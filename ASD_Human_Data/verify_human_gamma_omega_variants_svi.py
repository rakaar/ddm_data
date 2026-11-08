# %%
"""Independent MAT, likelihood, posterior, convergence and figure audit for human variants."""
import os
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
from io import BytesIO
from functools import partial
import argparse
import json
import pickle

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.io.matlab._mio5 import MatFile5Reader
from scipy.integrate import quad
import jax
import jax.numpy as jnp
from numpyro import handlers
from numpyro.infer.util import log_density
from numpyro.infer.autoguide import AutoMultivariateNormal
from numpyro.infer.initialization import init_to_value
from PIL import Image
import fitz
from human_gamma_omega_variants_svi_utils import (
    ROOT, BASELINE_ROOT, OUTPUTS, MODES, RT_WINDOW, get_priors, parameter_names,
    parameter_blocks, sha256, load_data, prepare_group, sample_matrix, vector_params,
    numerical_check, gamma_omega_model, guide_support_model, trapezoidal_logpdf_jax,
    log_likelihood,
)
from human_plain_ddm_diagnostics_utils import scipy_ddm

# %% Required variant selection; preflight can run before posterior and plot exports.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--mode', required=True, choices=MODES)
parser.add_argument('--output-dir', type=Path)
parser.add_argument('--preflight-only', action='store_true')
args = parser.parse_args()
mode = args.mode
out = args.output_dir if args.output_dir is not None else OUTPUTS[mode]
priors = get_priors(mode)
names, blocks = parameter_names(mode), parameter_blocks(mode)
n_parameters = 61 if mode == 'rt_only' else 63
model = partial(gamma_omega_model, mode=mode)
support_model = partial(guide_support_model, mode=mode)
config = json.loads((out/'run_config.json').read_text())
assert config['free_parameters_per_group'] == n_parameters
assert config['mode'] == mode
assert config['parameter_names'] == list(names) and config['parameter_blocks'] == list(blocks)
assert config['priors'] == priors and config['rt_window_s'] == list(RT_WINDOW) == [0., 1.8]
assert config['guide'] == 'AutoMultivariateNormal' and config['elbo_particles'] == 1
assert config['learning_rate'] == .0002 and config['gradient_clip'] == 1. and config['guide_init_scale'] == .1
assert config['full_batch'] and config['dtype'] == 'float64' and config['K_max'] == 40
for path, digest in config['source_hashes'].items():
    assert sha256(ROOT/path) == digest, path
assert priors['t_E_aff'] == dict(hard=[0., .5], plausible=[.01, .2])
assert priors['gamma_positive'] == dict(hard=[-1., 20.], plausible=[0., 12.])
assert priors['gamma_negative'] == dict(hard=[-20., 1.], plausible=[-12., 0.])
assert priors['omega'] == dict(hard=[.01, 15.], plausible=[.1, 12.])
assert priors['w'] == dict(hard=[0., 1.], plausible=[.4, .6])
if mode == 'rt_choice_lapse':
    assert priors['lapse_prob'] == dict(hard=[.0001, .4], plausible=[.001, .1])
    assert priors['lapse_prob_right'] == dict(hard=[.001, .999], plausible=[.4, .6])
print('Independent verifier:', mode, out, flush=True)


def independent_prediction(params, time, condition=None, cumulative=False, side=None, prediction_mode=None):
    """NumPy/SciPy DDM, then a retained-window uniform lapse if requested."""
    ids = slice(None) if condition is None else condition
    g, o, tau = (np.asarray(params[n])[ids] for n in ['gamma', 'omega', 't_E_aff'])
    mass = scipy_ddm(RT_WINDOW[1], g, o, params['w'], tau, cumulative=True)
    result = scipy_ddm(time, g, o, params['w'], tau, cumulative=cumulative, side=side)/mass
    if (mode if prediction_mode is None else prediction_mode) == 'rt_choice_lapse':
        p, q = float(params['lapse_prob']), float(params['lapse_prob_right'])
        uniform = (np.clip(np.asarray(time)/RT_WINDOW[1], 0., 1.) if cumulative else
                   ((np.asarray(time) >= 0) & (np.asarray(time) <= RT_WINDOW[1]))/RT_WINDOW[1])
        choice_mass = 1. if side is None else np.where(np.asarray(side) == 1, q, 1.-q)
        result = (1.-p)*result + p*choice_mass*uniform
    return result


# %% Re-read the opaque source independently, retaining original one-based MAT rows.
full, selected, excluded, provenance = load_data(mode)
source = ROOT/'ASD_Human_Data'/provenance['source']
raw = loadmat(source)['__function_workspace__'].tobytes()
stream = BytesIO(b'MATLAB 5.0 MAT-file'.ljust(116, b' ')+b'\0'*8+raw[:4]+raw[8:])
reader = MatFile5Reader(stream)
reader.initialize_read(); reader.read_file_header(); header, _ = reader.read_var_header()
cells = list(reader.read_var_array(header)['MCOS'][0, 0][0]['arr'].flat)
source_names = [a.item() for a in cells[10].flat]
arrays = dict(zip(source_names, [np.asarray(a).ravel() for a in cells[5].flat]))
n_rows = int(cells[7].item())
chars = cells[4].ravel()[4+n_rows:].tobytes().decode('utf-16le')
condition = np.array([chars[i:i+2] for i in range(0, len(chars), 2)])
group_names = [a.item() for a in cells[2].flat]
original_groups = np.array([group_names[int(c)-1] for c in cells[3].ravel()])
ids = full.source_row.to_numpy()-1
assert len(full) == 11400 and len(selected) == 11344 and len(excluded) == 56
np.testing.assert_array_equal(ids, np.flatnonzero(condition == 'RT'))
np.testing.assert_array_equal(original_groups[ids], full.Group.to_numpy())
for name in full:
    if name in arrays and name not in ['Group', 'Condition']:
        np.testing.assert_allclose(full[name], arrays[name][ids], rtol=0, atol=1e-12, equal_nan=True)
assert selected.RT_s.between(*RT_WINDOW).all() and excluded.RT_s.gt(RT_WINDOW[1]).all()
assert int(selected.RT_s.lt(.2).sum()) == 4 and selected.source_row.is_unique
assert set(selected.source_row).isdisjoint(excluded.source_row)
assert set(selected.source_row) | set(excluded.source_row) == set(full.source_row)
np.testing.assert_array_equal(pd.read_csv(out/'selected_trials.csv').source_row, selected.source_row)
np.testing.assert_array_equal(pd.read_csv(out/'excluded_trials.csv').source_row, excluded.source_row)
assert selected.Valid.eq(1).all() and selected.Stim_duration_s.isna().all()
np.testing.assert_array_equal(selected.Choice.eq(1).astype(int), selected.ChoseRight)
np.testing.assert_allclose(selected.Delta_dB, selected.Stim_right_dB-selected.Stim_left_dB, rtol=0, atol=1e-12)
for filename in ['selected_trials.csv', 'excluded_trials.csv']:
    assert (BASELINE_ROOT/filename).read_bytes() == (out/filename).read_bytes(), filename

# %% Verify normalized trapezoids and the actual model's prior density correction.
prior_integrals = {}
for name, prior in priors.items():
    lo, hi = prior['hard']; pl, ph = prior['plausible']
    height = 1/((pl-lo)/2+(ph-pl)+(hi-ph)/2)
    area = quad(lambda x: np.interp(x, [lo, pl, ph, hi], [0., height, height, 0.]),
                lo, hi, points=[pl, ph], epsabs=1e-12)[0]
    np.testing.assert_allclose(area, 1., rtol=0, atol=1e-12)
    x = np.linspace(lo+1e-7, hi-1e-7, 1001)
    np.testing.assert_allclose(np.exp(np.asarray(trapezoidal_logpdf_jax(x, lo, pl, ph, hi))),
                               np.interp(x, [lo, pl, ph, hi], [0., height, height, 0.]), rtol=1e-12, atol=1e-12)
    prior_integrals[name] = area

if not args.preflight_only:
    hist = pd.read_csv(out/'rtd_histograms.csv', float_precision='round_trip')
    curves = pd.read_csv(out/'rtd_model_curves.csv', float_precision='round_trip')
    condition_curves = pd.read_csv(out/'condition_model_curves.csv', float_precision='round_trip')
    psych = pd.read_csv(out/'psychometric_comparison.csv', float_precision='round_trip')
    quality = pd.read_csv(out/'fit_quality.csv', float_precision='round_trip')
    all_summary = pd.read_csv(out/'parameter_summary.csv', float_precision='round_trip')
reports = {}

# %% Reconstruct group identities, support transforms, likelihood and posterior exports.
for group, n_trials, n_people in [('Control', 6574, 11), ('ASD', 4770, 8)]:
    folder = out/group.lower()
    sub, conditions, data, initial, initial_source = prepare_group(
        selected, group, mode, initial_root=Path(config['initial_root']))
    assert len(sub) == n_trials and sub.Participant.nunique() == n_people
    assert len(conditions) == 20 and conditions.condition_index.tolist() == list(range(20))
    expected_map = [(a, d) for a in [20., 40.] for d in [-2., -1.3, -.65, -.3, -.1, .1, .3, .65, 1.3, 2.]]
    np.testing.assert_allclose(conditions[['ASL_cd', 'signed_delta']], expected_map, rtol=0, atol=1e-12)
    observed = pd.read_csv(folder/'selected_trials.csv', float_precision='round_trip')
    np.testing.assert_array_equal(observed.source_row, sub.source_row)
    for col in ['RT_s', 'Choice', 'Participant', 'GroupID', 'condition_index', 'ASL_cd', 'signed_delta']:
        np.testing.assert_allclose(observed[col], sub[col], rtol=0, atol=1e-12)
    manifest = json.loads((folder/'data_manifest.json').read_text())
    assert manifest['initialization_sha256'] == sha256(initial_source)
    assert manifest['selected_trials_sha256'] == sha256(folder/'selected_trials.csv')
    pd.testing.assert_frame_equal(pd.read_csv(folder/'conditions.csv', float_precision='round_trip'),
                                  conditions, check_dtype=False, rtol=0, atol=1e-12)
    original_initial = json.loads((BASELINE_ROOT/group.lower()/'initialization.json').read_text())
    for block in ['gamma', 'omega', 't_E_aff', 'w']:
        np.testing.assert_array_equal(initial[block], original_initial[block])
    minimum = sub.groupby('condition_index').RT_s.min().reindex(range(20)).to_numpy()
    expected_caps = np.minimum(.5, minimum-1e-6) if mode == 'rt_only' else np.full(20, .5)
    np.testing.assert_allclose(conditions.minimum_RT_s, minimum, rtol=0, atol=1e-15)
    np.testing.assert_allclose(conditions.guide_delay_upper_s, expected_caps, rtol=0, atol=1e-15)
    for field in ['hard', 'plausible']:
        expected = np.array([priors['gamma_positive' if d > 0 else 'gamma_negative'][field]
                             for d in conditions.signed_delta])
        np.testing.assert_array_equal(np.column_stack(data['gamma_'+field]), expected)

    if args.preflight_only:
        params = {k: np.asarray(v) for k, v in initial.items()}
    else:
        posterior = dict(np.load(folder/'posterior_samples.npz'))
        matrix = sample_matrix(posterior, mode)
        assert matrix.shape == (10000, n_parameters) and np.isfinite(matrix).all()
        assert set(posterior) == set(blocks)
        guide_params = dict(np.load(folder/'guide_parameters.npz'))
        with (folder/'best_checkpoint.pkl').open('rb') as stream:
            best_checkpoint = pickle.load(stream)
        for key in guide_params:
            np.testing.assert_array_equal(guide_params[key], np.asarray(best_checkpoint['params'][key]))
        guide = AutoMultivariateNormal(support_model, init_loc_fn=init_to_value(values=initial), init_scale=.1)
        handlers.seed(guide, rng_seed=config['seed'])(data)
        assert guide.latent_dim == n_parameters
        regenerated = guide.sample_posterior(jax.random.PRNGKey(config['seed']+100000),
                                             guide_params, sample_shape=(10000,))
        for key in blocks:
            np.testing.assert_allclose(posterior[key], regenerated[key], rtol=0, atol=1e-13)
        saved = all_summary.loc[all_summary.Group.eq(group)]
        assert saved.parameter.tolist() == list(names)
        for col, values in [('mean', matrix.mean(0)), ('sd', matrix.std(0, ddof=1)),
                            ('q025', np.quantile(matrix, .025, axis=0)), ('q975', np.quantile(matrix, .975, axis=0))]:
            np.testing.assert_allclose(saved[col], values, rtol=1e-12, atol=1e-12)
        for filename, expected in [('posterior_covariance.csv', np.cov(matrix, rowvar=False)),
                                    ('posterior_correlation.csv', np.corrcoef(matrix, rowvar=False))]:
            saved_matrix = pd.read_csv(folder/filename, index_col=0)
            assert saved_matrix.index.tolist() == list(names) and saved_matrix.columns.tolist() == list(names)
            np.testing.assert_allclose(saved_matrix, expected, rtol=1e-11, atol=1e-12)
        assert np.all((matrix[:, 20:40] > .01) & (matrix[:, 20:40] < 15.))
        assert np.all((matrix[:, 40:60] > 0) & (matrix[:, 40:60] < expected_caps))
        assert np.all((matrix[:, 60] > 0) & (matrix[:, 60] < 1))
        assert np.all(matrix[:, :20] > np.asarray(data['gamma_hard'][0]))
        assert np.all(matrix[:, :20] < np.asarray(data['gamma_hard'][1]))
        if mode == 'rt_choice_lapse':
            assert np.all((matrix[:, 61] > .0001) & (matrix[:, 61] < .4))
            assert np.all((matrix[:, 62] > .001) & (matrix[:, 62] < .999))
        params = vector_params(matrix.mean(0), mode)
        for draw in [0, 4999, 9999]:
            probe = vector_params(matrix[draw], mode)
            np.testing.assert_allclose(float(log_likelihood(probe, data, mode, K=40)),
                                       float(log_likelihood(probe, data, mode, K=80)), rtol=1e-11, atol=1e-6)

    numerical = numerical_check(params, data, mode)
    actual = handlers.trace(handlers.seed(handlers.substitute(model, data=params), rng_seed=0)).get_trace(data)
    prototype = handlers.trace(handlers.seed(handlers.substitute(support_model, data=params), rng_seed=0)).get_trace(data)
    assert sum(np.size(s['value']) for s in actual.values()
               if s['type'] == 'sample' and not s.get('is_observed', False)) == n_parameters
    assert {n for n, s in actual.items() if s['type'] == 'sample' and not s.get('is_observed', False)} == set(blocks)
    actual_delay, guide_delay = actual['t_E_aff']['fn'], prototype['t_E_aff']['fn']
    while hasattr(actual_delay, 'base_dist'):
        actual_delay = actual_delay.base_dist
    while hasattr(guide_delay, 'base_dist'):
        guide_delay = guide_delay.base_dist
    np.testing.assert_allclose(actual_delay.high, .5, rtol=0, atol=0)
    np.testing.assert_allclose(guide_delay.high, expected_caps, rtol=0, atol=1e-15)
    prior_log = 0.
    for block in blocks:
        hard, flat = ((data['gamma_hard'], data['gamma_plausible']) if block == 'gamma' else
                      (priors[block]['hard'], priors[block]['plausible']))
        prior_log += float(np.asarray(trapezoidal_logpdf_jax(params[block], hard[0], flat[0], flat[1], hard[1])).sum())
    log_joint, _ = log_density(model, (data,), {}, params)
    target_ll = float(log_likelihood(params, data, mode))
    np.testing.assert_allclose(float(log_joint)-target_ll, prior_log, rtol=0, atol=1e-8)
    ix, rt, choice = (np.asarray(data[n]) for n in ['condition', 'rt', 'choice'])
    independent_rt_ll = float(np.log(independent_prediction(params, rt, ix)).sum())
    independent_joint_ll = float(np.log(independent_prediction(params, rt, ix, side=choice)).sum())
    np.testing.assert_allclose(target_ll, independent_rt_ll if mode == 'rt_only' else independent_joint_ll,
                               rtol=1e-10, atol=1e-6)
    np.testing.assert_allclose(float(log_likelihood(params, data, mode, marginal=True)),
                               independent_rt_ll, rtol=1e-10, atol=1e-6)

    # Flipping choices must leave the RT-only target exactly unchanged.
    flipped = dict(data, choice=-data['choice'])
    np.testing.assert_allclose(float(log_likelihood(params, flipped, mode, marginal=True)),
                               independent_rt_ll, rtol=1e-10, atol=1e-6)
    identity_checks = dict(choice_marginalization_invariant=True)
    if mode == 'rt_only':
        np.testing.assert_allclose(float(log_likelihood(params, flipped, mode)), target_ll, rtol=0, atol=1e-10)
    else:
        p0 = dict(params, lapse_prob=jnp.asarray(0.))
        p1 = dict(params, lapse_prob=jnp.asarray(1.))
        base = {k: params[k] for k in ['gamma', 'omega', 't_E_aff', 'w']}
        with np.errstate(divide='ignore'):
            ddm_joint = np.log(scipy_ddm(rt, np.asarray(params['gamma'])[ix], np.asarray(params['omega'])[ix],
                                       params['w'], np.asarray(params['t_E_aff'])[ix], side=choice))
        mass = scipy_ddm(1.8, params['gamma'], params['omega'], params['w'], params['t_E_aff'], cumulative=True)
        # Initial delays are causal, even when posterior mean delays exceed some RTs.
        p0_causal = dict(initial, lapse_prob=jnp.asarray(0.), lapse_prob_right=params['lapse_prob_right'])
        init_ddm = float(log_likelihood({k: initial[k] for k in base}, data, 'rt_only', marginal=True))
        np.testing.assert_allclose(float(log_likelihood(p0_causal, data, mode, marginal=True)), init_ddm, rtol=0, atol=1e-8)
        if np.isfinite(ddm_joint).all():
            np.testing.assert_allclose(float(log_likelihood(p0, data, mode)), float((ddm_joint-np.log(mass[ix])).sum()),
                                       rtol=1e-10, atol=1e-6)
        else:
            assert np.isneginf(float(log_likelihood(p0, data, mode)))
        q = float(params['lapse_prob_right'])
        uniform_joint_ll = float(np.log(np.where(choice == 1, q, 1.-q)/1.8).sum())
        np.testing.assert_allclose(float(log_likelihood(p1, data, mode)), uniform_joint_ll, rtol=0, atol=1e-8)
        np.testing.assert_allclose(float(log_likelihood(p1, data, mode, marginal=True)), -len(rt)*np.log(1.8), rtol=0, atol=1e-8)
        late = {k: jnp.asarray(v) for k, v in params.items()}
        late.update(t_E_aff=jnp.full(20, .49), lapse_prob=jnp.asarray(.07), lapse_prob_right=jnp.asarray(.6))
        late_ll = float(log_likelihood(late, data, mode))
        late_gradient = jax.grad(lambda p: log_likelihood(p, data, mode))(late)
        assert np.isfinite(late_ll) and all(np.isfinite(np.asarray(g)).all() for g in late_gradient.values())
        assert np.any(rt < .49)
        identity_checks.update(p_zero_equals_ddm=True, p_one_equals_uniform=True,
                               finite_gradient_when_delay_exceeds_RT=True)

    area_errors, cdf_errors = [], []
    for c in conditions.itertuples():
        i = c.condition_index
        observed_condition = sub.loc[sub.condition_index.eq(i)]
        assert observed_condition.ASL_cd.eq(c.ASL_cd).all()
        np.testing.assert_allclose(observed_condition.signed_delta, c.signed_delta, rtol=0, atol=1e-12)
        assert len(observed_condition) == c.n and int(observed_condition.Choice.eq(1).sum()) == c.right
        tau = float(params['t_E_aff'][i])
        area = quad(lambda t: float(independent_prediction(params, t, i)), 0., 1.8,
                    points=[tau], epsabs=1e-9, limit=150)[0]
        area_errors.append(abs(area-1.))
        right = float(independent_prediction(params, 1.8, i, cumulative=True, side=1))
        left = float(independent_prediction(params, 1.8, i, cumulative=True, side=-1))
        np.testing.assert_allclose(right+left, 1., rtol=0, atol=1e-8)
        if args.preflight_only:
            continue
        row = psych.loc[psych.Group.eq(group) & psych.condition_index.eq(i)].iloc[0]
        np.testing.assert_allclose([row.n, row.right, row.empirical_p_right],
                                   [c.n, c.right, c.right/c.n], rtol=0, atol=1e-12)
        np.testing.assert_allclose(row.model_p_right, right, rtol=1e-8, atol=1e-8)
        points = condition_curves.loc[condition_curves.Group.eq(group) & condition_curves.condition_index.eq(i)].iloc[::17]
        for col, side in [('density', None), ('right_density', 1), ('left_density', -1)]:
            np.testing.assert_allclose(points[col], independent_prediction(params, points.time_s.to_numpy(), i, side=side),
                                       rtol=1e-7, atol=1e-8)
        expected_cdf = independent_prediction(params, points.time_s.to_numpy(), i, cumulative=True)
        cdf_errors.append(float(np.max(np.abs(points.cdf-expected_cdf))))
        np.testing.assert_allclose(points.cdf, expected_cdf, rtol=1e-7, atol=1e-8)
        empirical_rt = np.sort(sub.loc[sub.condition_index.eq(i), 'RT_s'].to_numpy())
        f = independent_prediction(params, empirical_rt, i, cumulative=True)
        ecdf = np.arange(1, len(empirical_rt)+1)/len(empirical_rt)
        gap = max(np.abs(f-ecdf).max(), np.abs(f-ecdf+1/len(empirical_rt)).max())
        np.testing.assert_allclose(row.cdf_gap, gap, rtol=1e-7, atol=1e-8)
    assert max(area_errors) < 1e-7
    if args.preflight_only:
        reports[group] = dict(trials=n_trials, participants=n_people, parameters=n_parameters, numerical=numerical,
                              independent_marginal_rt_log_likelihood=independent_rt_ll,
                              independent_joint_log_likelihood=independent_joint_ll,
                              unit_conditional_mass_max_error=max(area_errors), identities=identity_checks)
        print(group, 'independent preflight passed', flush=True)
        continue
    assert max(cdf_errors) < 1e-7
    score = quality.loc[quality.Group.eq(group)].iloc[0]
    np.testing.assert_allclose(score.marginal_rt_log_likelihood, independent_rt_ll, rtol=1e-10, atol=1e-6)
    if mode == 'rt_only':
        assert pd.isna(score.joint_log_likelihood)
    else:
        np.testing.assert_allclose(score.joint_log_likelihood, independent_joint_ll, rtol=1e-10, atol=1e-6)
    errors = psych.loc[psych.Group.eq(group)].model_p_right-psych.loc[psych.Group.eq(group)].empirical_p_right
    for metric, expected in [('psychometric_mae', np.abs(errors).mean()),
                             ('psychometric_rmse', np.sqrt((errors**2).mean())),
                             ('mean_condition_cdf_gap', psych.loc[psych.Group.eq(group)].cdf_gap.mean()),
                             ('max_condition_cdf_gap', psych.loc[psych.Group.eq(group)].cdf_gap.max())]:
        np.testing.assert_allclose(score[metric], expected, rtol=1e-10, atol=1e-10)
    for level in [20., 40.]:
        for delta in [.1, .3, .65, 1.3, 2.]:
            obs = sub.loc[sub.ASL_cd.eq(level) & np.isclose(sub.signed_delta.abs(), delta)]
            bins = hist.loc[hist.Group.eq(group) & hist.ASL_cd.eq(level) & hist.abs_delta.eq(delta)]
            counts, edges = np.histogram(obs.RT_s, bins=np.linspace(0, 1.8, 91))
            np.testing.assert_array_equal(bins['count'], counts)
            np.testing.assert_allclose(bins.left_s, edges[:-1], rtol=0, atol=1e-12)
            np.testing.assert_allclose(bins.right_s, edges[1:], rtol=0, atol=1e-12)
            np.testing.assert_allclose(bins.density, counts/(len(obs)*.02), rtol=1e-12, atol=1e-12)
            mix = curves.loc[curves.Group.eq(group) & curves.ASL_cd.eq(level) & curves.abs_delta.eq(delta)]
            ns = obs.groupby('signed_delta').size()
            np.testing.assert_allclose([mix.negative_weight.iloc[0], mix.positive_weight.iloc[0]],
                                       ns.to_numpy()/len(obs), rtol=0, atol=1e-12)
            expected_density, expected_cdf = np.zeros(len(mix)), np.zeros(len(mix))
            for sign in [-1, 1]:
                single = condition_curves.loc[condition_curves.Group.eq(group) & condition_curves.ASL_cd.eq(level)
                                              & condition_curves.signed_delta.eq(sign*delta)]
                np.testing.assert_array_equal(single.time_s, mix.time_s)
                expected_density += single.density.to_numpy()*ns.loc[sign*delta]/len(obs)
                expected_cdf += single.cdf.to_numpy()*ns.loc[sign*delta]/len(obs)
            np.testing.assert_allclose(mix.density, expected_density, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(mix.cdf, expected_cdf, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(mix.cdf.iloc[-1], 1., rtol=0, atol=1e-12)

    # Reconstruct every reset, patience count, and the terminal three-checkpoint test.
    losses = np.load(folder/'losses.npy')
    trace = pd.read_csv(folder/'convergence.csv', float_precision='round_trip')
    report = json.loads((folder/'convergence_summary.json').read_text())
    window_means = losses.reshape(-1, 1000).mean(1)
    np.testing.assert_allclose(window_means, trace.mean_negative_elbo, rtol=1e-12, atol=1e-10)
    assert len(trace) == len(window_means)
    record, best_step, patience = np.inf, 0, 0
    for mean, row in zip(window_means, trace.itertuples()):
        improved = mean < record
        if improved:
            record, best_step, patience = mean, int(row.step), 0
        else:
            patience += 1
        assert bool(row.updated_best) == improved and int(row.best_step) == best_step and int(row.patience) == patience
    assert report['restored_best_step'] == best_step and report['completed_steps'] == len(losses)
    assert int(best_checkpoint['step']) == best_step
    assert report['confirmation_windows'] == patience == (len(losses)-best_step)//1000
    assert report['updates_after_restored_best'] == len(losses)-best_step
    history = trace[[n+'_mean' for n in names]].to_numpy()
    spreads = trace[[n+'_sd' for n in names]].to_numpy()
    change = np.ptp(history[-3:], axis=0)/np.min(spreads[-3:], axis=0)
    stable = bool(np.all(change < .1))
    assert stable == report['terminal_parameter_stable']
    bi = best_step//1000
    selected_stable = bool(np.all(np.ptp(history[max(0, bi-3):bi], axis=0)/
                                 np.min(spreads[max(0, bi-3):bi], axis=0) < .1)) if bi >= 3 else False
    assert selected_stable == report['restored_checkpoint_parameter_stable']
    np.testing.assert_allclose(change.max(), report['maximum_terminal_mean_change_sd'], rtol=1e-10, atol=1e-10)
    confirmed = patience >= config['confirmation_windows'] and stable
    assert report['flagged'] == (not confirmed)
    if report['stop_reason'] == 'confirmed_minimum_and_terminal_parameter_stability':
        assert confirmed and len(losses) >= config['min_steps']
    else:
        assert report['stop_reason'] == 'maximum_steps_without_confirmation' and len(losses) == config['max_steps']
    reports[group] = dict(trials=n_trials, participants=n_people, parameters=n_parameters, numerical=numerical,
                          independent_marginal_rt_log_likelihood=independent_rt_ll,
                          independent_joint_log_likelihood=independent_joint_ll,
                          unit_conditional_mass_max_error=max(area_errors), scipy_cdf_max_error=max(cdf_errors),
                          identities=identity_checks, convergence=report)
    print(group, 'posterior and calculation audit passed', flush=True)

# %% Optional three-model comparison, evaluated on identical raw trial identities.
comparison_figures, comparison_checks = [], {}
if not args.preflight_only and (out/'variant_comparison_manifest.json').exists():
    comparison = json.loads((out/'variant_comparison_manifest.json').read_text())
    comparison_figures = comparison['figures']
    metrics = pd.read_csv(out/'variant_comparison_metrics.csv', float_precision='round_trip')
    probabilities = pd.read_csv(out/'variant_comparison_psychometrics.csv', float_precision='round_trip')
    pooled = pd.read_csv(out/'variant_comparison_pooled_rtd_curves.csv', float_precision='round_trip')
    pooled_hist = pd.read_csv(out/'variant_comparison_pooled_rtd_histograms.csv', float_precision='round_trip')
    mapping = pd.read_csv(out/'variant_comparison_mixture_weights.csv', float_precision='round_trip')
    for model_mode, source_path in comparison['sources'].items():
        directory = ROOT/source_path
        for filename in ['selected_trials.csv', 'excluded_trials.csv']:
            assert (directory/filename).read_bytes() == (out/filename).read_bytes()
        for group in ['Control', 'ASD']:
            sub, conditions, data, _, _ = prepare_group(selected, group, mode, BASELINE_ROOT)
            posterior_file = directory/group.lower()/'posterior_samples.npz'
            assert sha256(posterior_file) == comparison['posterior_sha256'][f'{model_mode}/{group}']
            draw_mode = 'rt_only' if model_mode == 'rt_choice' else model_mode
            p = vector_params(sample_matrix(dict(np.load(posterior_file)), draw_mode).mean(0), draw_mode)
            ix, rt, choice = (np.asarray(data[n]) for n in ['condition', 'rt', 'choice'])
            rt_ll = float(np.log(independent_prediction(p, rt, ix, prediction_mode=model_mode)).sum())
            score = metrics.loc[metrics.Group.eq(group) & metrics['mode'].eq(model_mode)].iloc[0]
            np.testing.assert_allclose(score.marginal_rt_log_likelihood, rt_ll, rtol=1e-10, atol=1e-6)
            if model_mode == 'rt_only':
                assert pd.isna(score.joint_log_likelihood)
            else:
                joint_ll = float(np.log(independent_prediction(p, rt, ix, side=choice,
                                                               prediction_mode=model_mode)).sum())
                np.testing.assert_allclose(score.joint_log_likelihood, joint_ll, rtol=1e-10, atol=1e-6)
            for c in conditions.itertuples():
                row = probabilities.loc[probabilities.Group.eq(group) & probabilities['mode'].eq(model_mode)
                                        & probabilities.condition_index.eq(c.condition_index)].iloc[0]
                expected = independent_prediction(p, 1.8, c.condition_index, cumulative=True,
                                                  side=1, prediction_mode=model_mode)
                np.testing.assert_allclose(row.model_p_right, expected, rtol=1e-8, atol=1e-8)
                np.testing.assert_allclose(row.empirical_p_right, c.right/c.n, rtol=0, atol=1e-12)
            for level in [20., 40.]:
                chosen = conditions.loc[conditions.ASL_cd.eq(level)].sort_values('condition_index')
                ci = chosen.condition_index.to_numpy(dtype=int)
                weights = chosen.n.to_numpy()/chosen.n.sum()
                saved = mapping.loc[mapping.Group.eq(group) & mapping['mode'].eq(model_mode)
                                    & mapping.ASL_cd.eq(level)].sort_values('condition_index')
                np.testing.assert_array_equal(saved.condition_index, ci)
                np.testing.assert_array_equal(saved.n, chosen.n)
                np.testing.assert_allclose(saved.weight, weights, rtol=0, atol=1e-12)
                point = pooled.loc[pooled.Group.eq(group) & pooled['mode'].eq(model_mode)
                                   & pooled.ASL_cd.eq(level)].iloc[::17]
                time = point.time_s.to_numpy()[:, None]
                for column, cumulative in [('density', False), ('cdf', True)]:
                    expected = independent_prediction(p, time, ci, cumulative=cumulative,
                                                      prediction_mode=model_mode)@weights
                    np.testing.assert_allclose(point[column], expected, rtol=1e-7, atol=1e-8)
                if model_mode == 'rt_choice':
                    obs = sub.loc[sub.ASL_cd.eq(level), 'RT_s'].to_numpy()
                    h = pooled_hist.loc[pooled_hist.Group.eq(group) & pooled_hist.ASL_cd.eq(level)]
                    counts, _ = np.histogram(obs, bins=np.linspace(0, 1.8, 91))
                    np.testing.assert_array_equal(h['count'], counts)
                    np.testing.assert_allclose(h.density, counts/(len(obs)*.02), rtol=1e-12, atol=1e-12)
    comparison_checks = dict(common_scores_reconstructed=6, psychometric_predictions_reconstructed=120,
                             pooled_mixtures_reconstructed=12, pooled_histograms_reconstructed=4)

# %% Preserve prior analyses and unrelated work; check image/PDF geometry after plotting.
preserved = {}
for filename in ['preservation_baseline.json', 'unrelated_worktree_baseline.json']:
    baseline = json.loads((out/filename).read_text())
    for path, digest in baseline.items():
        assert sha256(ROOT/path) == digest, path
    preserved[filename] = len(baseline)
figure_checks = []
if not args.preflight_only:
    plot_manifest = json.loads((out/'plot_manifest.json').read_text())
    for name in plot_manifest['figures']+comparison_figures:
        path = out/name
        if path.suffix == '.png':
            with Image.open(path) as im:
                assert min(im.size) > 1000 and np.asarray(im.convert('RGB')).std() > 5
                figure_checks.append(dict(file=name, width=im.width, height=im.height))
        elif path.suffix == '.pdf':
            with fitz.open(path) as doc:
                for page in doc:
                    for block in page.get_text('blocks'):
                        if len(block) > 6 and block[6] == 0:
                            assert block[0] >= -.1 and block[1] >= -.1
                            assert block[2] <= page.rect.width+.1 and block[3] <= page.rect.height+.1, (name, block)
                figure_checks.append(dict(file=name, pages=len(doc), all_text_within_page=True))
result = dict(status='passed', mode=mode, preflight_only=args.preflight_only, source_rows=n_rows,
              original_RT_rows=11400, retained_rows=11344, excluded_rows=56, retained_below_200ms=4,
              priors_integrate_to_one=prior_integrals, groups=reports, preserved=preserved,
              comparison=comparison_checks, figures=figure_checks, verifier_sha256=sha256(Path(__file__)))
if not args.preflight_only:
    result.update(probabilities_reconstructed=40, histograms_reconstructed=20, signed_curves_reconstructed=40,
                  mixture_curves_reconstructed=20, posterior_parameters_reconstructed=2*n_parameters)
(out/('independent_preflight_validation.json' if args.preflight_only else 'independent_validation.json')).write_text(
    json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k not in ['groups', 'figures']}, indent=2))
