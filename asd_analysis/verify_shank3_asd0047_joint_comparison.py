# %% Independent checks of pooled-WT parameter comparisons, without plotting imports
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# %% Editable output root
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'asd_analysis/shank3_cohort1_ASD0047_joint_npl_alpha_gaussian_abl_contaminants'
OUT = BASE / 'joint_vs_two_stage_parameters'
chosen = json.loads((BASE / 'selected_fit.json').read_text())
MIXTURE = chosen['parameter_count'] == 20
manifest = json.loads((OUT / 'plot_manifest.json').read_text())
table = pd.read_csv(OUT / 'posterior_parameter_comparison.csv', float_precision='round_trip')
changes = pd.read_csv(OUT / 'posterior_mean_changes.csv', float_precision='round_trip')
assert len(table) == (40 if MIXTURE else 38) and len(changes) == (20 if MIXTURE else 19)
assert table.estimate_type.eq('variational_posterior').sum() == (38 if MIXTURE else 37)
assert table.loc[table.part.eq('reactive')].groupby('method').size().to_dict() == {'joint': 13, 'two_stage': 13}
assert table.loc[table.part.eq('proactive_lapse')].groupby('method').size().to_dict() == {'joint': 7 if MIXTURE else 6, 'two_stage': 7 if MIXTURE else 6}
draw_cache = {}
maximum_summary_error = 0.
for record in table.itertuples(index=False):
    config = json.loads((ROOT / record.source_run / 'configuration.json').read_text())
    if record.estimate_type == 'fixed_constant':
        assert record.parameter in ['lapse_prob_right', 'lapse_exp_weight'] and record.method == 'two_stage'
        assert record.mean == (.5 if record.parameter == 'lapse_prob_right' else 1.) and record.posterior_draws == 0
        assert np.isnan(record.q025) and np.isnan(record.q975)
        continue
    if record.source_run not in draw_cache:
        with np.load(ROOT / record.source_run / 'posterior_samples.npz') as samples:
            draw_cache[record.source_run] = {name: samples[name].copy() for name in samples.files}
    values = draw_cache[record.source_run][record.block]
    if values.ndim == 2:
        values = values[:, [20, 40, 60].index(int(record.ABL))]
    assert values.shape == (10000,) and np.isfinite(values).all()
    expected = np.array([np.average(values), *np.percentile(values, [2.5, 97.5])])
    observed = np.array([record.mean, record.q025, record.q975])
    maximum_summary_error = max(maximum_summary_error, float(np.max(np.abs(expected - observed))))
    np.testing.assert_allclose(expected, observed, rtol=1e-13, atol=1e-14)
    summary = pd.read_csv(ROOT / record.source_run / 'posterior_summary.csv', float_precision='round_trip').set_index('parameter')
    np.testing.assert_allclose(summary.loc[record.parameter, ['mean', 'q025', 'q975']].to_numpy(dtype=float), observed, rtol=1e-12, atol=1e-14)
    expected_scale = 1000. if record.block in ['T_0', 't_nd', 'del_go', 'delay_mu', 'delay_sigma'] else 1.
    assert record.display_scale == expected_scale
    np.testing.assert_allclose([record.display_mean, record.display_q025, record.display_q975], expected * expected_scale, rtol=1e-13, atol=1e-12)
    prior = config.get('priors', config.get('prior_bounds'))[record.block]
    np.testing.assert_array_equal(prior, [record.hard_lower, record.flat_lower, record.flat_upper, record.hard_upper])
for record in changes.itertuples(index=False):
    pair = table.loc[table.parameter.eq(record.parameter)].set_index('method')
    np.testing.assert_allclose(record.mean_change, pair.loc['joint', 'mean'] - pair.loc['two_stage', 'mean'], rtol=0, atol=1e-14)

# %% Exact trial identity and original frozen-value agreement
joint = ROOT / manifest['joint_run']
sequential = ROOT / 'asd_analysis/shank3_cohort1_wt_hom_individual_adaptive_bounds/wt/ASD0047'
joint_root = joint.parents[1]
new_valid = pd.read_csv(joint_root / 'fit_trials.csv.gz', float_precision='round_trip')
old_valid = pd.read_csv(sequential / 'fit_trials.csv.gz', float_precision='round_trip')
columns = ['animal', 'prepared_csv_line', 'condition', 'response_poke', 'timed_rt', 'onset_s', 'success', 'training_level', 'session_type', 'ABL_mapped', 'ILD']
left = new_valid[columns].sort_values('prepared_csv_line').reset_index(drop=True)
right = old_valid[columns].sort_values('prepared_csv_line').reset_index(drop=True)
# Exported choice/condition columns use int versus float storage; values must be exact.
pd.testing.assert_frame_equal(left, right, check_exact=True, check_dtype=False)
assert len(left) == 18771 and left.animal.nunique() == 1
assert left.response_poke.value_counts().to_dict() == {1.: 9592, -1.: 9179}
assert left.training_level.eq(16).all() and left.session_type.eq(1).all()
assert left.success.isin([1, -1]).all() and left.timed_rt.between(0, 1, inclusive='both').all()
assert set(left.ABL_mapped) == {20, 40, 60}
assert set(left.ILD) == {-16, -8, -4, -2, -1, 1, 2, 4, 8, 16}
assert set(left.animal) == {'ASD0047'}
new_parent = pd.read_csv(joint_root / 'parent_trials.csv.gz', float_precision='round_trip')
old_parent = pd.read_csv(sequential / 'parent_trials.csv.gz', float_precision='round_trip')
assert len(new_parent) == 21786 and len(old_parent) == 21947
assert new_parent.observation.value_counts().to_dict() == {'valid_RT_0_1': 18771, 'pre_onset_abort': 3004, 'late_fixation_censored': 11}
removed = old_parent.loc[~old_parent.prepared_csv_line.isin(new_parent.prepared_csv_line)]
assert len(removed) == 161 and removed.success.isin([1, -1]).all() and removed.timed_rt.gt(1).all()
assert int(old_parent.observed_abort.sum()) == 3004
fixed = json.loads((sequential / 'fixed_proactive.json').read_text())
original_abort = table.loc[table.method.eq('two_stage') & table.part.eq('proactive_lapse')].set_index('parameter')
for parameter, value in zip(fixed['parameter_order'], fixed['values']):
    np.testing.assert_allclose(original_abort.loc[parameter, 'mean'], value, rtol=0, atol=1e-12)

# %% Artist values, square geometry, full interval coverage, and input preservation
for artist in manifest['artists']:
    expected = table.loc[table.method.eq(artist['method']) & table.block.eq(artist['block'])].set_index('parameter').loc[artist['parameters']]
    assert artist['color'] == manifest['colors'][artist['method']]
    np.testing.assert_allclose(artist['mean'], expected.display_mean, rtol=0, atol=1e-12)
    if artist['fixed']:
        assert artist['q025'] is None and artist['q975'] is None
    else:
        np.testing.assert_allclose(artist['q025'], expected.display_q025, rtol=0, atol=1e-12)
        np.testing.assert_allclose(artist['q975'], expected.display_q975, rtol=0, atol=1e-12)
    if artist['block'].startswith('delay_'):
        np.testing.assert_array_equal(artist['x'], expected.ABL + manifest['abl_offsets_db'][artist['method']])
for geometry in manifest['geometry']:
    assert abs(geometry['width_inches'] - geometry['height_inches']) < 1e-12 and not geometry['grid']
    values = table.loc[table.block.eq(geometry['block'])]
    endpoints = values[['display_mean', 'display_q025', 'display_q975']].to_numpy().ravel()
    endpoints = endpoints[np.isfinite(endpoints)]
    assert min(endpoints) > geometry['ylim'][0] and max(endpoints) < geometry['ylim'][1]
assert len(manifest['geometry']) == (16 if MIXTURE else 15)
for name, expected in manifest['source_hashes'].items():
    with (ROOT / name).open('rb') as handle:
        assert hashlib.file_digest(handle, 'sha256').hexdigest() == expected, name
image_sizes = {}
for name, expected in manifest['figure_hashes'].items():
    assert hashlib.sha256((OUT / name).read_bytes()).hexdigest() == expected
    with Image.open(OUT / name) as img:
        image_sizes[name] = list(img.size)
        assert min(img.size) > 1500
report = dict(status='passed', posterior_marginals_verified=(38 if MIXTURE else 37), fixed_constants_verified=(2 if MIXTURE else 1),
    reactive_paired_parameters=13, proactive_lapse_comparisons=(7 if MIXTURE else 6), panels=(16 if MIXTURE else 15),
    posterior_draws_per_estimate=10000, maximum_summary_absolute_error=maximum_summary_error,
    identical_valid_trial_identities=18771, animals=1, right=9592, left=9179,
    two_stage_proactive_population=21947, joint_population=21786,
    long_valid_responses_removed_from_joint_only=161, images=image_sizes,
    source_hashes_unchanged=True, same_statistical_model_claimed=False)
(OUT / 'independent_validation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2), flush=True)
