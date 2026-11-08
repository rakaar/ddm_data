# %% Independent CSV reconstruction and SciPy probability checks
"""Run --pre-fit before SVI; run without arguments after figures are generated."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import invgauss
import jax
import jax.numpy as jnp
from proactive_lapse_no_led_utils import (PARAM_NAMES, PRIOR_BOUNDS, mixture_logpdf_logsf,
    proactive_logpdf_logsf, log_likelihood, trapezoid_logpdf)

REPO = Path(__file__).resolve().parents[1]
OUT = REPO/'asd_analysis/shank3_cohort1_proactive_lapse'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--pre-fit', action='store_true')
args = parser.parse_args()
prep = json.loads((OUT/'preparation.json').read_text())
GENOTYPES = ['wt', 'hom', 'het']
expected = {'wt': (5, 83921, 14205, 69716), 'hom': (6, 81721, 13580, 68141), 'het': (3, 59603, 10439, 49164)}
if args.pre_fit and not (OUT/'prior_artifact_hashes.json').exists():
    prior = {}
    for folder in ['shank3_cohort1_wt_rtds', 'shank3_cohort1_fixation_abort_durations']:
        for path in (OUT.parent/folder).rglob('*'):
            if path.is_file():
                prior[str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT/'prior_artifact_hashes.json').write_text(json.dumps(prior, indent=2)+'\n')


def scipy_values(t, p):
    v, a, delay, lapse, rate = p
    times = np.asarray(t)
    shifted = times-delay
    active = shifted > 0
    log_pdf_a = np.full_like(times, -np.inf, dtype=float)
    log_sf_a = np.zeros_like(times, dtype=float)
    log_pdf_a[active] = invgauss.logpdf(shifted[active], mu=1/(v*a), scale=a*a)
    log_sf_a[active] = invgauss.logsf(shifted[active], mu=1/(v*a), scale=a*a)
    with np.errstate(divide='ignore'):
        lp = np.logaddexp(np.log1p(-lapse)+log_pdf_a, np.log(lapse)+np.log(rate)-rate*times)
        ls = np.logaddexp(np.log1p(-lapse)+log_sf_a, np.log(lapse)-rate*times)
    return np.where(times >= 0, lp, -np.inf), np.where(times >= 0, ls, 0.)


# %% Parse source cells independently; do not use the fitting driver's selector
records, members, counts, classifications = [], Counter(), Counter(), Counter()
source = REPO/'asd_analysis/session_type_csvs/ASD_session_type_1.csv'
with source.open(newline='') as stream:
    for line, r in enumerate(csv.DictReader(stream), 2):
        assert float(r['session_type']) == 1
        if r['filename'] != 'SHANK3_cohort1.csv' or float(r['training_level']) != 16:
            continue
        abl, ild = float(r['ABL']), float(r['ILD'])
        if (60 if abl == 59 else abl) not in [20, 40, 60] or abs(ild) not in [1, 2, 4, 8, 16]:
            continue
        g, animal = r['genotype'], r['animal']
        t, c = float(r['fix_time'])/1000, float(r['intended_fix_time'])/1000
        assert np.isfinite(t) and np.isfinite(c) and t >= 0 and c > 0
        success = float(r['success'] or 'nan')
        if success in [-1, 1]:
            label = 'valid_choice_censored'
        else:
            assert r['abort_type'] == 'Fixation'
            label = 'pre_onset_fixation_abort' if t < c else 'late_fixation_abort_censored'
        event = label == 'pre_onset_fixation_abort'
        records.append((line, g, animal, t, c, event, label))
        counts[(g, 'total')] += 1
        counts[(g, 'event')] += int(event)
        members[(g, animal)] += 1
        classifications[label] += 1
independent = pd.DataFrame(records, columns=['prepared_csv_line', 'genotype', 'animal', 'fixation_s', 'censor_s', 'is_event', 'classification'])
saved = pd.read_csv(OUT/'trial_classifications.csv.gz', float_precision='round_trip')
for col in independent:
    np.testing.assert_array_equal(independent[col], saved[col])
for g in GENOTYPES:
    got = (sum(key[0] == g for key in members), counts[g, 'total'], counts[g, 'event'], counts[g, 'total']-counts[g, 'event'])
    assert got == expected[g]
    rows = independent.loc[independent.genotype.eq(g)]
    data = np.load(OUT/(g+'_likelihood_data.npz'))
    np.testing.assert_array_equal(data['all_censor_times'], rows.censor_s)
    np.testing.assert_array_equal(data['censor_times'], rows.loc[~rows.is_event, 'censor_s'])
    values, ns = np.unique(rows.loc[rows.is_event, 'fixation_s'], return_counts=True)
    np.testing.assert_array_equal(data['event_times'], values)
    np.testing.assert_array_equal(data['event_counts'], ns)
assert len(saved) == 225245 and saved.animal.nunique() == 14
assert classifications['late_fixation_abort_censored'] == 1381
assert ((saved.classification == 'valid_choice_censored') & (saved.fixation_s < saved.censor_s)).sum() == 26
assert (saved.is_event & saved.fixation_s.lt(.3)).any()
assert saved.timed_rt.gt(1).any() and saved.timed_rt.lt(-1).any()
membership = pd.read_csv(OUT/'animal_membership.csv')
for r in membership.itertuples():
    assert r.trials == members[r.genotype, r.animal]
assert set(membership.animal) == {'ASD0040','ASD0042','ASD0044','ASD0047','ASD0049','ASD0023',
    'ASD0024','ASD0043','ASD0045','ASD0046','ASD0048','ASD0038','ASD0039','ASD0041'}
original_types = pd.read_csv(REPO/'SHANK3_cohort1.csv', usecols=['session_type']).session_type.value_counts().to_dict()
assert set(original_types) == {1, 2, 23}
assert '1 - standard RT condition' in (REPO/'animal_structure_columns.txt').read_text()

# %% SciPy inverse-Gaussian reference: normalization, boundary cases, gradients
vectors = [np.array(x, dtype=float) for x in [[1.6,2.5,.09,.05,5], [3,1,.04,.15,2],
    [.1,8,0,.01,.001], [8,.1,.3,.3,20], [2,2,.1,0,2], [2,2,.1,1,2]]]
checks, ll_checks = [], []
for p in vectors:
    times = np.unique(np.r_[np.linspace(0, 3, 350), p[2], np.nextafter(p[2], 0), np.nextafter(p[2], np.inf)])
    actual = [np.asarray(x) for x in mixture_logpdf_logsf(times, jnp.array(p))]
    reference = scipy_values(times, p)
    for a, b in zip(actual, reference):
        np.testing.assert_allclose(a, b, rtol=2e-9, atol=2e-9)
    assert np.all(np.exp(actual[1]) <= 1+1e-14)
    assert np.all(np.diff(np.exp(actual[1])) <= 1e-13)
    # Integrate the shifted inverse-Gaussian on its own support; exponential is exact.
    mass_a = quad(lambda u: invgauss.pdf(u, mu=1/(p[0]*p[1]), scale=p[1]**2), 0, np.inf, epsabs=1e-9)[0]
    np.testing.assert_allclose((1-p[3])*mass_a+p[3], 1, atol=2e-8)
    checks.append({'params': p.tolist(), 'density_integral': (1-p[3])*mass_a+p[3]})
for name, bounds in PRIOR_BOUNDS.items():
    area = quad(lambda x: float(jnp.exp(trapezoid_logpdf(x, bounds))), bounds[0], bounds[-1], points=bounds[1:3])[0]
    np.testing.assert_allclose(area, 1, atol=1e-9)
for p in vectors[:2]:
    density = jax.jit(lambda t: jnp.exp(mixture_logpdf_logsf(t, jnp.array(p))[0]))
    integral = (quad(lambda t: float(density(t)), 0, p[2], epsabs=1e-10)[0]
                + quad(lambda t: float(density(t)), p[2], np.inf, epsabs=1e-9)[0])
    np.testing.assert_allclose(integral, 1, atol=1e-8)
    gradient_at_delay = jax.grad(lambda q: sum(mixture_logpdf_logsf(p[2], q)))(jnp.array(p))
    assert np.isfinite(gradient_at_delay).all()
for g in GENOTYPES:
    z = independent.loc[independent.genotype.eq(g)]
    stored = np.load(OUT/(g+'_likelihood_data.npz'))
    data = {k: jnp.array(stored[k]) for k in ['event_times', 'event_counts', 'censor_times']}
    for p in vectors[:2]:
        expected_ll = scipy_values(z.loc[z.is_event, 'fixation_s'], p)[0].sum()+scipy_values(z.loc[~z.is_event, 'censor_s'], p)[1].sum()
        actual_ll, gradient = jax.value_and_grad(log_likelihood)(jnp.array(p), data)
        np.testing.assert_allclose(actual_ll, expected_ll, rtol=1e-11, atol=1e-7)
        assert np.isfinite(gradient).all()
        # Central finite differences independently check the complete data gradient.
        fd = []
        for i in range(5):
            delta = np.zeros(5); delta[i] = 1e-6
            def ref(q):
                return scipy_values(z.loc[z.is_event, 'fixation_s'], q)[0].sum()+scipy_values(z.loc[~z.is_event, 'censor_s'], q)[1].sum()
            fd.append((ref(p+delta)-ref(p-delta))/(2e-6))
        np.testing.assert_allclose(gradient, fd, rtol=2e-5, atol=2e-4)
        ll_checks.append({'genotype': g, 'parameters': p.tolist(), 'jax_log_likelihood': float(actual_ll), 'scipy_log_likelihood': expected_ll})
report = {'status': 'passed', 'selected_trials': len(saved), 'animals': saved.animal.nunique(),
    'event_censor_counts': classifications, 'original_session_type_counts': original_types,
    'exact_trial_identities_and_clocks_verified': True, 'no_rt_window_verified': True,
    'density_integrals': checks, 'scipy_likelihood_checks': ll_checks, 'finite_gradients_and_finite_differences': True,
    'source_hashes': prep['source_hashes']}

# %% Post-fit checkpoint restoration, posterior draws and start selection
if not args.pre_fit:
    from numpyro.infer import SVI, Trace_ELBO, init_to_value
    from numpyro.infer.autoguide import AutoMultivariateNormal
    import numpyro
    from proactive_lapse_no_led_utils import model
    selected = json.loads((OUT/'selected_starts.json').read_text())
    run_checks = []
    for g in GENOTYPES:
        candidates = []
        stored = np.load(OUT/(g+'_likelihood_data.npz'))
        data = {k: jnp.array(stored[k]) for k in ['event_times', 'event_counts', 'censor_times']}
        z = independent.loc[independent.genotype.eq(g)]
        for seed in [0, 1]:
            folder = OUT/g/f'seed_{seed}'
            summary = json.loads((folder/'run_summary.json').read_text())
            config = json.loads((folder/'configuration.json').read_text())
            for name, digest in config['source_hashes'].items():
                assert hashlib.sha256((REPO/'asd_analysis'/name).read_bytes()).hexdigest() == digest
            windows = pd.read_csv(folder/'convergence_windows.csv')
            losses = pd.read_csv(folder/'losses.csv.gz').negative_elbo.to_numpy()
            assert len(losses) == summary['final_checked_step']
            np.testing.assert_allclose(losses.reshape(-1,1000).mean(1), windows.mean_negative_elbo, atol=1e-8)
            best_index = windows.mean_negative_elbo.idxmin()
            assert summary['restored_best_step'] == windows.loc[best_index, 'end_step']
            np.testing.assert_allclose(summary['restored_window_negative_elbo'], windows.loc[best_index, 'mean_negative_elbo'])
            if summary['status'] == 'converged':
                assert len(losses) >= 50000 and bool(windows.iloc[-1].can_stop)
                assert windows.iloc[-1].windows_since_best >= 12
            else:
                assert 'inconclusive' in summary['stop_reason']
            checkpoint = pickle.loads((folder/'checkpoint.pkl').read_bytes())
            restored = pickle.loads((folder/'restored_guide_and_optimizer.pkl').read_bytes())
            assert restored['step'] == checkpoint['best_step'] == summary['restored_best_step']
            for a, b in zip(jax.tree_util.tree_leaves(restored['state']), jax.tree_util.tree_leaves(checkpoint['best_state'])):
                np.testing.assert_array_equal(a, b)
            guide = AutoMultivariateNormal(model, init_loc_fn=init_to_value(values=dict(zip(PARAM_NAMES, config['init']))), init_scale=.1)
            svi = SVI(model, guide, numpyro.optim.ClippedAdam(.0002, clip_norm=1), Trace_ELBO())
            svi.init(jax.random.PRNGKey(seed), data)
            for a, b in zip(jax.tree_util.tree_leaves(svi.get_params(restored['state'])), jax.tree_util.tree_leaves(restored['params'])):
                np.testing.assert_array_equal(a, b)
            reproduced = guide.sample_posterior(jax.random.PRNGKey(seed+1000), restored['params'], sample_shape=(10000,))
            samples = np.load(folder/'posterior_samples.npz')
            estimates = pd.read_csv(folder/'posterior_summary.csv').set_index('parameter')
            for name in PARAM_NAMES:
                x = samples[name]
                assert x.shape == (10000,) and np.isfinite(x).all()
                assert (x > PRIOR_BOUNDS[name][0]).all() and (x < PRIOR_BOUNDS[name][-1]).all()
                np.testing.assert_array_equal(x, reproduced[name])
                np.testing.assert_allclose([x.mean(), *np.quantile(x, [.025,.975])], estimates.loc[name, ['mean','q025','q975']].astype(float))
            p = np.array([samples[name].mean() for name in PARAM_NAMES])
            ref_ll = scipy_values(z.loc[z.is_event, 'fixation_s'], p)[0].sum()+scipy_values(z.loc[~z.is_event, 'censor_s'], p)[1].sum()
            np.testing.assert_allclose(ref_ll, summary['posterior_mean_log_likelihood'], atol=1e-7)
            candidates.append(summary)
            run_checks.append(summary)
        best = min(candidates, key=lambda r:r['restored_window_negative_elbo'])
        tolerance = .001*max(1,abs(best['restored_window_negative_elbo']))
        near = [r for r in candidates if r['restored_window_negative_elbo'] <= best['restored_window_negative_elbo']+tolerance]
        winner = max(near, key=lambda r:(r['posterior_mean_log_likelihood'], -r['seed']))
        assert winner['seed'] == next(r['seed'] for r in selected if r['genotype']==g)
    report['run_checks'] = run_checks
    report['posterior_draws_reproduced'] = True
    report['checkpoint_restoration_and_start_selection_verified'] = True
    # Independent bin masses: average F(min(edge,c)) over every trial, using SciPy.
    hist = pd.read_csv(OUT/'diagnostic_histogram_bins.csv')
    predictions = pd.read_csv(OUT/'abort_probability_comparison.csv')
    for g in GENOTYPES:
        winner = next(r for r in selected if r['genotype']==g)
        estimates = pd.read_csv(OUT/g/f'seed_{winner["seed"]}/posterior_summary.csv').set_index('parameter')
        p = estimates.loc[PARAM_NAMES, 'mean'].to_numpy()
        z = independent.loc[independent.genotype.eq(g)]
        table = hist.loc[hist.genotype.eq(g)]
        previous = 0.
        for b in table.itertuples():
            cumulative = np.mean(-np.expm1(scipy_values(np.minimum(b.bin_right_s, z.censor_s), p)[1]))
            np.testing.assert_allclose(b.model_bin_mass, cumulative-previous, atol=2e-12)
            previous = cumulative
        edges = np.r_[table.bin_left_s.to_numpy(), table.bin_right_s.iloc[-1]]
        obs = np.histogram(z.loc[z.is_event, 'fixation_s'], edges)[0]
        np.testing.assert_array_equal(obs, table.event_count)
        np.testing.assert_allclose(table.empirical_density, obs/(len(z)*.02))
        np.testing.assert_allclose(np.sum(table.empirical_density*.02), z.is_event.mean())
        row = predictions.loc[predictions.genotype.eq(g)].iloc[0]
        np.testing.assert_allclose(row.predicted_abort_probability, previous, atol=2e-12)
        np.testing.assert_allclose(row.observed_abort_probability, z.is_event.mean())
    report['censoring_adjusted_model_histogram_masses_verified'] = True
    prior_hashes = json.loads((OUT/'prior_artifact_hashes.json').read_text())
    for name, digest in prior_hashes.items():
        assert hashlib.sha256((REPO/name).read_bytes()).hexdigest() == digest
    report['prior_shank3_figures_and_audits_unchanged'] = True

# %% Preservation is checked for the prepared and original inputs
for name, expected_sha in prep['source_hashes'].items():
    with (REPO/name).open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected_sha, name
report['input_hashes_unchanged'] = True
destination = OUT/('prefit_validation.json' if args.pre_fit else 'independent_validation.json')
destination.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({'status':'passed', 'report':str(destination), 'trials':len(saved), 'animals':saved.animal.nunique()}, indent=2))
