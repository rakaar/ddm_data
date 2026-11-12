# %% Independent SciPy checks of the ASD0047 joint likelihood and saved posterior
import argparse
import json
import os
import pickle
from pathlib import Path
if os.environ.get('ASDFIT_CPUS'):
    os.sched_setaffinity(0, {int(x) for x in os.environ['ASDFIT_CPUS'].split(',')})
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_simpson, quad, simpson
from scipy.signal import fftconvolve
from scipy.special import ndtr
from scipy.stats import invgauss, norm
import jax
import jax.numpy as jnp
from numpyro import handlers
from numpyro.infer import init_to_value
from numpyro.infer.autoguide import AutoMultivariateNormal
from shank3_asd0047_joint_utils import (PARAMETER_COUNT, VARIANT, FIXED_PARAMETERS, BASE, OLD, ROOT, OUT, ABLS, ILDS, BLOCKS,
    PROACTIVE, REACTIVE, CONTAMINANTS, initialization, make_model, parameter_names,
    sample_matrix, contaminant, log_likelihood, sha256)
from human_plain_ddm_diagnostics_utils import scipy_ddm

# %% Editable tolerances; verification never changes a fit or its bound flags
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--round', type=int, default=0)
parser.add_argument('--seed', type=int, choices=[0, 1])
parser.add_argument('--pre-fit', action='store_true')
parser.add_argument('--report-checks', action='store_true')
args = parser.parse_args()
folder = OUT / f'round_{args.round:02d}'
priors = json.loads((folder / 'bounds.json').read_text())
grid_end = max(1.5, 1+priors['del_go'][-1])
prep = json.loads((OUT / 'preparation.json').read_text())
data = dict(np.load(OUT / 'observations.npz'))
jdata = {k: jnp.asarray(v) for k, v in data.items()}
MEAN_LL_TOLERANCE = 1e-4
assert prep['parent'] == 21786 and prep['valid'] == 18771
assert data['parent_count'] == len(data['rt'])+data['abort_counts'].sum()+len(data['late_onsets'])
assert len(parameter_names()) == PARAMETER_COUNT and len(BLOCKS) == (15 if FIXED_PARAMETERS else 16)
for name, h in prep['input_hashes'].items():
    assert sha256(OUT / name) == h

# %% Reconstruct the animal independently from the unchanged prepared source CSV
if args.pre_fit and args.round == 0:
    columns = ['animal', 'filename', 'session_type', 'training_level', 'success',
               'ABL', 'ILD', 'fix_time', 'intended_fix_time', 'timed_rt', 'response_poke']
    source = ROOT / 'asd_analysis/session_type_csvs/ASD_session_type_1.csv'
    raw = pd.read_csv(source, usecols=columns, float_precision='round_trip')
    raw['prepared_csv_line'] = np.arange(len(raw)) + 2
    animal = raw.loc[raw.animal.eq('ASD0047') & raw.filename.eq('SHANK3_cohort1.csv')].copy()
    animal['ABL_mapped'] = animal.ABL.replace({59: 60})
    animal = animal.loc[animal.ABL_mapped.isin(ABLS) & animal.ILD.isin(ILDS)]
    assert len(animal) == 21947 and animal.training_level.eq(16).all() and animal.session_type.eq(1).all()
    choices = animal.success.isin([1, -1]) & animal.timed_rt.between(0, 1)
    assert choices.sum() == 18771 and (animal.success.isin([1, -1]) & animal.timed_rt.gt(1)).sum() == 161
    saved = pd.read_csv(OUT / 'fit_trials.csv.gz', float_precision='round_trip')
    for key in ['prepared_csv_line', 'timed_rt', 'response_poke', 'fix_time', 'intended_fix_time', 'ILD']:
        np.testing.assert_array_equal(animal.loc[choices, key], saved[key])
    (OUT / 'raw_population_validation.json').write_text(json.dumps(dict(status='passed',
        animal='ASD0047', original_parent=21947, valid_choices=18771, long_valid_excluded=161,
        exact_trial_ids_and_recorded_values=True, source_sha256=sha256(source)), indent=2) + '\n')


def independent_grid(p):
    # Independent NPL algebra and K80 first passage, then finer endpoint-corrected FFT.
    abl, ild = np.repeat(ABLS, 10), np.tile(ILDS, 3)
    lam, ell, alpha = [float(p[k]) for k in ['rate_lambda', 'rate_norm_l', 'alpha']]
    gain = 10**(lam*(1-ell)*abl/20)
    right = gain*np.exp(lam*ild/17.37)/(np.exp(lam*ell*ild/17.37)+alpha*np.exp(-lam*ell*ild/17.37))
    left = gain*np.exp(-lam*ild/17.37)/(np.exp(-lam*ell*ild/17.37)+alpha*np.exp(lam*ell*ild/17.37))
    gamma = float(p['theta_E'])*(right-left)/(right+left)
    omega = (right+left)/(float(p['T_0'])*float(p['theta_E'])**2)
    mu, sd = np.repeat(p['delay_mu'], 10), np.repeat(p['delay_sigma'], 10)
    distance = 2*min(float(p['w']), 1-float(p['w']))
    mode = 2*distance**2/(omega*(3+np.sqrt(9+4*(gamma*distance)**2)))
    dt = min(.000125, float(np.min(sd/np.where(mu < 6*sd, 40., 8.))), float(mode.min()/20))
    times = np.linspace(0, grid_end, round(grid_end/dt)+1)
    step = times[1]-times[0]
    fs, cs = [], []
    for i in range(30):
        delay_pdf = norm.pdf((times-mu[i])/sd[i])/sd[i]/ndtr(mu[i]/sd[i])
        g0, gp0 = delay_pdf[0], mu[i]/sd[i]**2*delay_pdf[0]
        delay_pdf[0] *= .5
        decision = np.array([scipy_ddm(times, gamma[i], omega[i], float(p['w']), 0, side=s, K=80) for s in [-1, 1]])
        density = np.array([fftconvolve(f, delay_pdf)[:len(times)]*step for f in decision])
        density += step**2/12*(decision*gp0-np.gradient(decision, step, axis=-1, edge_order=2)*g0)
        density = np.maximum(density, 0)
        density[:, 0] = 0
        cdf = cumulative_simpson(density, x=times, axis=-1, initial=0)
        assert cdf.min() >= -1e-9 and np.diff(cdf, axis=-1).min() >= -1e-8
        assert cdf.sum(0).max() < 1+1e-5
        fs.append(density)
        cs.append(cdf)
    return times, np.asarray(fs), np.asarray(cs), dt


def independent_contaminant(t, p, cutoff):
    beta, d = float(p['beta_lapse']), float(FIXED_PARAMETERS.get('lapse_exp_weight', p.get('lapse_exp_weight', 1.)))
    t = np.asarray(t)
    u = np.clip(t, 0, cutoff)
    if beta == 0:
        epdf, ecdf, esf = np.full_like(u, 1/cutoff), u/cutoff, 1-u/cutoff
    else:
        z = -np.expm1(-beta*cutoff)
        epdf, ecdf = beta*np.exp(-beta*u)/z, -np.expm1(-beta*u)/z
        esf = (np.exp(-beta*u)-np.exp(-beta*cutoff))/z
    pdf = np.where((t >= 0) & (t <= cutoff), d*epdf+(1-d)/cutoff, 0)
    return pdf, d*ecdf+(1-d)*u/cutoff, d*esf+(1-d)*(1-u/cutoff)


def independent_likelihood(p, observed, grid):
    times, density, cdf, _ = grid
    v, a, delay, lapse, q = [float(p[k]) for k in ['V_A', 'theta_A', 't_nd', 'lapse_prob', 'lapse_prob_right']]
    def action(t):
        return invgauss.pdf(t-delay, mu=1/(v*a), scale=a*a), invgauss.sf(t-delay, mu=1/(v*a), scale=a*a)
    result = 0.
    for i in range(30):
        mask = observed['condition'] == i
        rt, onset = observed['rt'][mask], observed['onset'][mask]
        f = np.array([np.interp(rt, times, density[i, s]) for s in range(2)]).T
        c = np.array([np.interp(rt, times, cdf[i, s]) for s in range(2)]).T
        go = np.array([np.interp(rt+float(p['del_go']), times, cdf[i, s]) for s in range(2)]).T
        readout = .5*(1-go.sum(-1))[:, None]+go-c
        ap, ass = action(onset+rt)
        h, _, _ = independent_contaminant(onset+rt, p, float(observed['cutoff']))
        joint = (1-lapse)*(ap[:, None]*readout+ass[:, None]*f)+lapse*h[:, None]*np.array([1-q, q])
        values = joint[np.arange(len(rt)), (observed['choice'][mask] == 1).astype(int)]
        assert (values > 0).all()
        result += np.log(values).sum()
    abort_ap, _ = action(observed['abort_times'])
    abort_h, _, _ = independent_contaminant(observed['abort_times'], p, float(observed['cutoff']))
    result += np.dot(np.log((1-lapse)*abort_ap+lapse*abort_h), observed['abort_counts'])
    _, late_sf = action(observed['late_onsets'])
    _, _, hc = independent_contaminant(observed['late_onsets'], p, float(observed['cutoff']))
    onset_survival = (1-lapse)*late_sf+lapse*hc
    end_e = np.array([np.interp(1, times, cdf[i].sum(0)) for i in range(30)])
    def end_survival(onsets, conditions):
        _, ass = action(onsets+1)
        _, _, hsf = independent_contaminant(onsets+1, p, float(observed['cutoff']))
        return (1-lapse)*ass*(1-end_e[conditions])+lapse*hsf
    late_mass = onset_survival-end_survival(observed['late_onsets'], observed['late_condition'])
    assert (late_mass > 0).all()
    result += np.log(late_mass).sum()
    entry_sf = (1-lapse)*action(0.)[1]+lapse
    normalizers = entry_sf-end_survival(observed['parent_onsets'], observed['parent_condition'])
    assert ((normalizers > 0) & (normalizers <= 1)).all()
    result -= np.log(normalizers).sum()
    return float(result), float(entry_sf)



# %% Independent normalized contaminant checks, endpoint cases and parameter multiplicity
tests = []
for beta in [0., 1e-10, .5, 5., 50.]:
    for d in ([1.] if FIXED_PARAMETERS else [0., .4, 1.]):
        p = {**initialization(0), 'beta_lapse': beta, 'lapse_exp_weight': d}
        t = np.r_[-.1, np.linspace(0, 3.3, 331), 3.4]
        direct = [np.asarray(v) for v in contaminant(jnp.asarray(t), p, 3.3)]
        reference = independent_contaminant(t, p, 3.3)
        for actual, expected in zip(direct, reference):
            np.testing.assert_allclose(actual, expected, atol=2e-6 if beta == 1e-10 else 1e-11, rtol=1e-10)
        integral = quad(lambda u: independent_contaminant(u, p, 3.3)[0], 0, 3.3, epsabs=1e-11)[0]
        np.testing.assert_allclose(integral, 1., atol=1e-10)
        tests.append(dict(beta=beta, exponential_weight=d, integral=integral))
if args.pre_fit:
    probes = [initialization(0), initialization(1)]
    # Test negative proactive delay and near-zero Gaussian mu without changing priors.
    probes.append({**initialization(1), 't_nd': -.15, 'delay_mu': np.array([.002, .003, .004]), 'delay_sigma': np.array([.02, .025, .03])})
    records = []
    for index, params in enumerate(probes):
        grid = independent_grid(params)
        p = {k: jnp.asarray(v) for k, v in params.items()}
        value, gradient = jax.jit(jax.value_and_grad(lambda pp: log_likelihood(pp, jdata, grid_end)))(p)
        value = float(value)
        assert all(np.isfinite(np.asarray(v)).all() for v in gradient.values())
        independent, survival = independent_likelihood(params, data, grid)
        error = abs(value-independent)/prep['parent']
        assert error < MEAN_LL_TOLERANCE, (index, error)
        trace = handlers.trace(handlers.seed(make_model(priors, grid_end), 0)).get_trace(None)
        assert sum(np.size(v['value']) for v in trace.values() if v['type'] == 'sample') == PARAMETER_COUNT
        assert {k for k in trace if k.endswith('_trapezoid')} == {k+'_trapezoid' for k in BLOCKS}
        # Independently integrate both choice densities over the entire retained
        # interval, including pre-onset abort times and entry truncation.
        normalization = []
        times, density, cdf, _ = grid
        va, aa, delay = [float(params[k]) for k in ['V_A', 'theta_A', 't_nd']]
        lapse = float(params['lapse_prob'])
        for condition in [0, 14, 29]:
            for onset in [.2086220337, 1., 2.3]:
                absolute = np.linspace(0, onset+1, int((onset+1)/.000025)+1)
                rel = absolute-onset
                fe = np.interp(rel, times, density[condition].sum(0), left=0)
                ce = np.interp(rel, times, cdf[condition].sum(0), left=0)
                ap = invgauss.pdf(absolute-delay, mu=1/(va*aa), scale=aa*aa)
                ass = invgauss.sf(absolute-delay, mu=1/(va*aa), scale=aa*aa)
                h = independent_contaminant(absolute, params, 3.3)[0]
                raw = (1-lapse)*(ap*(1-ce)+ass*fe)+lapse*h
                s0 = (1-lapse)*invgauss.sf(-delay, mu=1/(va*aa), scale=aa*aa)+lapse
                send = (1-lapse)*ass[-1]*(1-ce[-1])+lapse*independent_contaminant(absolute[-1], params, 3.3)[2]
                area = simpson(raw, x=absolute)/(s0-send)
                np.testing.assert_allclose(area, 1, atol=2e-5)
                normalization.append(dict(condition=condition, onset=onset, area=float(area)))
        records.append(dict(probe=index, entry_conditioned=True, upper_truncated=True,
            jax_log_likelihood=value, independent_log_likelihood=independent,
            mean_absolute_error=error, independent_dt=grid[-1], entry_survival=survival,
            all_parameter_gradients_finite=True, retained_window_normalization=normalization))
        print(records[-1], flush=True)
    report = dict(status='passed', parameter_count=PARAMETER_COUNT, full_parent_count=21786,
        original_parent_count=21947, valid_choices=18771, pre_onset_aborts=3004,
        late_censors=11, long_RT_excluded=161, long_RT_censors=0,
        tested_entry_options=[True], tested_long_RT_policy='excluded_with_joint_retained_window_normalization',
        probes=records, contaminant_normalization_tests=tests, prior_multiplicity_verified=True,
        source_hashes={str(ROOT/'asd_analysis/shank3_asd0047_joint_utils.py'): sha256(ROOT/'asd_analysis/shank3_asd0047_joint_utils.py'),
            str(ROOT/'asd_analysis/shank3_individual_adaptive_bounds_utils.py'): sha256(ROOT/'asd_analysis/shank3_individual_adaptive_bounds_utils.py')})
    (folder / 'prefit_validation.json').write_text(json.dumps(report, indent=2)+'\n')
else:
    policy = json.loads((OUT / 'likelihood_policy.json').read_text())
    assert policy['status'] == 'confirmed' and policy['exclude_valid_RT_above_1'] and policy['joint_retained_window_normalization']
    records = []
    for seed in ([args.seed] if args.seed is not None else [0, 1]):
        run = folder / f'seed_{seed}'
        config = json.loads((run / 'configuration.json').read_text())
        summary = json.loads((run / 'run_summary.json').read_text())
        assert summary['status'] == 'converged' and summary['parameter_count'] == PARAMETER_COUNT
        samples = dict(np.load(run / 'posterior_samples.npz'))
        matrix = sample_matrix(samples)
        assert matrix.shape == (10000, PARAMETER_COUNT) and np.isfinite(matrix).all()
        means = {k: np.asarray(v.mean(0)) for k, v in samples.items()}
        saved = pd.read_csv(run / 'posterior_summary.csv', float_precision='round_trip')
        np.testing.assert_allclose(saved['mean'], matrix.mean(0), rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(saved.q025, np.quantile(matrix, .025, axis=0), rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(saved.q975, np.quantile(matrix, .975, axis=0), rtol=1e-12, atol=1e-14)
        for key, values in samples.items():
            assert np.min(values) >= priors[key][0] and np.max(values) <= priors[key][-1]
        best = pickle.loads((run / 'best_checkpoint.pkl').read_bytes())
        restored = pickle.loads((run / 'restored_guide_optimizer.pkl').read_bytes())
        for x, y in zip(jax.tree_util.tree_leaves(best['state']), jax.tree_util.tree_leaves(restored['state'])):
            np.testing.assert_array_equal(x, y)
        model = make_model(priors, grid_end)
        guide = AutoMultivariateNormal(model, init_loc_fn=init_to_value(values={k: np.asarray(v) for k, v in config['initial'].items()}), init_scale=config['guide_scale'])
        handlers.seed(guide, rng_seed=seed)(jdata)
        reproduced = guide.sample_posterior(jax.random.PRNGKey(seed+100000), best['params'], sample_shape=(10000,))
        posterior_reproduction_max_error = 0.
        posterior_reproduction_bitwise = True
        for k in samples:
            actual = np.asarray(reproduced[k])
            posterior_reproduction_max_error = max(posterior_reproduction_max_error,
                float(np.max(np.abs(samples[k]-actual))))
            posterior_reproduction_bitwise &= np.array_equal(samples[k], actual)
            # CPU/GPU constrained transforms differ at floating-point roundoff.
            # Keep the exact draws immutable and report agreement, not bitwise identity.
            np.testing.assert_allclose(samples[k], actual, rtol=1e-12, atol=1e-12)
        losses = np.load(run / 'losses.npy')
        convergence = pd.read_csv(run / 'convergence.csv', float_precision='round_trip')
        assert len(losses) == summary['final_checked_step'] == convergence.step.iloc[-1]
        best_loss, patience, best_step = np.inf, 0, 0
        for row in convergence.itertuples():
            mean = losses[row.step-config['window']:row.step].mean()
            np.testing.assert_allclose(mean, row.mean_negative_elbo, atol=1e-9)
            if mean < best_loss:
                best_loss, patience, best_step = mean, 0, row.step
            else:
                patience += 1
            assert patience == row.patience and best_step == row.best_step
        assert best_step == summary['restored_best_step'] and patience >= config['confirmation_windows']
        assert summary['final_checked_step'] >= config['min_steps']
        assert best['step'] == best_step and best['loss'] == best_loss
        for name, expected in config['source_hashes'].items():
            current = ROOT/name
            archived = OUT/'source_history'/expected/current.name
            assert sha256(current) == expected or (archived.exists() and sha256(archived) == expected), name
        grid = independent_grid(means)
        independent, entry_survival = independent_likelihood(means, data, grid)
        jparams = {k: jnp.asarray(v) for k, v in means.items()}
        value = float(jax.jit(lambda p: log_likelihood(p, jdata, grid_end))(jparams))
        error = abs(value-independent)/prep['parent']
        assert error < MEAN_LL_TOLERANCE
        np.testing.assert_allclose(value, summary['posterior_mean_joint_log_likelihood'], rtol=1e-12, atol=1e-6)
        records.append(dict(seed=seed, status='passed', mean_log_likelihood_error=error,
            independent_log_likelihood=independent, entry_survival_at_means=entry_survival,
            posterior_draws_reproduced=True, posterior_reproduction_bitwise=bool(posterior_reproduction_bitwise),
            posterior_reproduction_max_absolute_error=posterior_reproduction_max_error,
            posterior_reproduction_tolerance=dict(rtol=1e-12, atol=1e-12), checkpoint_restoration_verified=True,
            loss_confirmation_replayed=True, source_hashes_verified=True))
        print(records[-1], flush=True)
    (folder / ('independent_validation.json' if args.seed is None else f'independent_validation_seed{args.seed}.json')).write_text(
        json.dumps(dict(status='passed', runs=records, contaminant_tests=tests), indent=2)+'\n')

# %% Independent posterior-mean report predictions using the finer SciPy grid
if args.report_checks:
    assert not args.pre_fit and args.seed is not None
    report_folder = OUT / 'report'
    parent = pd.read_csv(OUT / 'parent_trials.csv.gz', float_precision='round_trip')
    valid = pd.read_csv(OUT / 'fit_trials.csv.gz', float_precision='round_trip')
    signed = pd.read_csv(report_folder / 'signed_model_densities.csv.gz', float_precision='round_trip')
    probabilities = pd.read_csv(report_folder / 'psychometric_probabilities.csv', float_precision='round_trip')
    normalizers = pd.read_csv(report_folder / 'onset_window_normalizers.csv.gz', float_precision='round_trip')
    v, a, nd, lapse, q = [float(means[k]) for k in ['V_A', 'theta_A', 't_nd', 'lapse_prob', 'lapse_prob_right']]
    s0 = (1-lapse)*invgauss.sf(-nd, mu=1/(v*a), scale=a*a)+lapse
    times, density, cdf, _ = grid
    from numpy.polynomial.legendre import leggauss
    cuts = np.linspace(0, 1, 21)
    nodes, weights = leggauss(32)
    qt = np.concatenate([.5*(hi-lo)*nodes+.5*(hi+lo) for lo, hi in zip(cuts[:-1], cuts[1:])])
    qw = np.concatenate([.5*(hi-lo)*weights for lo, hi in zip(cuts[:-1], cuts[1:])])
    checks = []
    for i in range(30):
        z = parent.loc[parent.condition.eq(i)]
        onset = z.onset_s.to_numpy()
        ce1 = np.interp(1., times, cdf[i].sum(0))
        send = (1-lapse)*invgauss.sf(onset+1-nd, mu=1/(v*a), scale=a*a)*(1-ce1)+lapse*independent_contaminant(onset+1, means, 3.3)[2]
        zn = s0-send
        saved = normalizers.loc[normalizers.condition.eq(i)]
        np.testing.assert_array_equal(saved.prepared_csv_line, z.prepared_csv_line)
        np.testing.assert_allclose(saved.retained_window_mass, zn, rtol=3e-5, atol=2e-6)
        block = signed.loc[signed.condition.eq(i)]
        # Cover both signs of RT, the entry edge, peak and tail for every condition.
        ix = np.unique(np.r_[np.arange(0, 2001, 40), np.argmax(block.density.to_numpy())])
        relative = block.rt_s.to_numpy()[ix]
        ep = np.interp(relative, times, density[i].sum(0), left=0)
        es = 1-np.interp(relative, times, cdf[i].sum(0), left=0)
        absolute = relative[:, None]+onset[None, :]
        ap = invgauss.pdf(absolute-nd, mu=1/(v*a), scale=a*a)
        ass = invgauss.sf(absolute-nd, mu=1/(v*a), scale=a*a)
        h = independent_contaminant(absolute, means, 3.3)[0]
        expected = (((1-lapse)*(ap*es[:, None]+ass*ep[:, None])+lapse*h)*(absolute >= 0)/zn).mean(1)
        np.testing.assert_allclose(block.density.to_numpy()[ix], expected, atol=3e-4, rtol=3e-4)
        z = valid.loc[valid.condition.eq(i)]
        onset = z.onset_s.to_numpy()
        sc = (1-lapse)*invgauss.sf(onset-nd, mu=1/(v*a), scale=a*a)+lapse*independent_contaminant(onset, means, 3.3)[2]
        send = (1-lapse)*invgauss.sf(onset+1-nd, mu=1/(v*a), scale=a*a)*(1-ce1)+lapse*independent_contaminant(onset+1, means, 3.3)[2]
        valid_mass = sc-send
        fe = np.array([np.interp(qt, times, density[i, side]) for side in range(2)]).T
        ce = np.array([np.interp(qt, times, cdf[i, side]) for side in range(2)]).T
        cg = np.array([np.interp(qt+float(means['del_go']), times, cdf[i, side]) for side in range(2)]).T
        readout = .5*(1-cg.sum(1))+cg[:, 1]-ce[:, 1]
        independent_right = 0.
        for first in range(0, len(qt), 64):
            sl = slice(first, first+64)
            absolute = qt[sl, None]+onset[None, :]
            ap = invgauss.pdf(absolute-nd, mu=1/(v*a), scale=a*a)
            ass = invgauss.sf(absolute-nd, mu=1/(v*a), scale=a*a)
            h = independent_contaminant(absolute, means, 3.3)[0]
            right = ((1-lapse)*(ap*readout[sl, None]+ass*fe[sl, 1, None])+lapse*q*h)/valid_mass
            independent_right += qw[sl]@right.mean(1)
        saved_probability = probabilities.loc[probabilities.condition.eq(i)].model_p_right.iloc[0]
        np.testing.assert_allclose(saved_probability, independent_right, atol=3e-5, rtol=3e-5)
        checks.append(dict(condition=i, independent_p_right=float(independent_right),
            plotted_p_right=saved_probability, absolute_error=float(abs(saved_probability-independent_right)),
            checked_rtd_times=len(ix), max_rtd_absolute_error=float(abs(block.density.to_numpy()[ix]-expected).max())))
    (report_folder / 'independent_model_prediction_checks.json').write_text(json.dumps(
        dict(status='passed', finer_scipy_grid_s=grid[-1], signed_conditions=30,
            exact_onsets=True, checks=checks), indent=2)+'\n')
    print('Independent report predictions passed for all30 signed conditions.', flush=True)
