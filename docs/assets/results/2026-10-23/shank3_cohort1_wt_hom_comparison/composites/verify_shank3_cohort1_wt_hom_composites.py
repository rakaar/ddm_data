# %% Independent checks of the actual plotted artists against saved fits and trials
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
from collections import Counter
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
COMPARISON = ROOT / 'asd_analysis/shank3_cohort1_wt_hom_comparison'
OUT = COMPARISON / 'composites'
ABLS = [20, 40, 60]
ILDS = [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]
ABS_ILDS = [1, 2, 4, 8, 16]
EXPECTED = {'wt': (68184, 5, 13750), 'hom': (67715, 6, 13274)}
ABL_COLORS = {20: '#1f77b4', 40: '#ff7f0e', 60: '#2ca02c'}
GROUP_COLORS = {'wt': '#1f77b4', 'hom': '#d62728'}
manifest = json.loads((OUT / 'plot_manifest.json').read_text())
original = json.loads((COMPARISON / 'plot_manifest.json').read_text())
with gzip.open(OUT / 'plotted_series.json.gz', 'rt') as stream:
    traces = json.load(stream)
parameters = pd.read_csv(COMPARISON / 'parameter_comparisons.csv', float_precision='round_trip')
points = pd.read_csv(COMPARISON / 'chronometric_points.csv', float_precision='round_trip')
signed = pd.read_csv(COMPARISON / 'signed_condition_means.csv', float_precision='round_trip')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def select_trace(role, group, abl=None, abs_ild=None, block=None):
    selected = [t for t in traces if t['role'] == role and t['group'] == group and t['ABL'] == abl
                and t.get('abs_ILD') == abs_ild and t.get('block') == block]
    assert len(selected) == 1, (role, group, abl, abs_ild, block)
    return selected[0]


# %% Posterior summaries, independent trial populations, and every RTD bin
parameter_count = 0
display_counts = {}
population = {}
model_grid_area_errors = []
for group, (n_valid, n_animals, n_aborts) in EXPECTED.items():
    folder = ROOT / f'asd_analysis/shank3_cohort1_{group}_gamma_omega'
    draws = dict(np.load(folder / 'condition_delay/posterior_samples.npz'))
    mapping = pd.read_csv(folder / 'conditions.csv').set_index('condition')
    for block, array in draws.items():
        vectors = array[:, None] if array.ndim == 1 else array
        assert vectors.shape == (10000, 30 if block in ['gamma', 'omega', 't_E_aff'] else 1)
        for i in range(vectors.shape[1]):
            name = block if array.ndim == 1 else f'{block}_{i:02d}'
            z = parameters.loc[parameters.group.eq(group) & parameters.parameter.eq(name)].iloc[0]
            mean = np.mean(vectors[:, i])
            lo, hi = np.percentile(vectors[:, i], [2.5, 97.5])
            scale = 1000 if block in ['t_E_aff', 'del_go'] else 1
            np.testing.assert_allclose([z.display_mean, z.display_lower95, z.display_upper95],
                                      np.array([mean, lo, hi]) * scale, atol=1e-12, rtol=1e-14)
            abl = int(mapping.loc[i, 'ABL']) if array.ndim > 1 else None
            trace = select_trace('posterior', group, abl, block=block)
            index = ILDS.index(int(mapping.loc[i, 'ILD'])) if array.ndim > 1 else 0
            actual = [trace[key][index] for key in ['y', 'lower', 'upper']]
            np.testing.assert_allclose(actual, np.array([mean, lo, hi]) * scale, atol=1e-12, rtol=1e-14)
            assert trace['color'] == GROUP_COLORS[group] and trace['linestyle'] == 'None'
            expected_x = index + (-.10 if group == 'wt' else .10) if array.ndim > 1 else ['wt', 'hom'].index(group)
            np.testing.assert_allclose(trace['x'][index], expected_x, atol=1e-15)
            parameter_count += 1

    parent = pd.read_csv(folder / 'parent_trials.csv.gz', float_precision='round_trip')
    selected = parent.success.isin([1, -1]) & np.isfinite(parent.timed_rt) & parent.timed_rt.between(0, 1)
    valid = parent.loc[selected].copy()
    valid['abs_ILD'] = valid.ILD.abs()
    assert len(valid) == n_valid and valid.animal.nunique() == n_animals
    assert valid.training_level.eq(16).all() and valid.session_type.eq(1).all()
    assert valid.genotype.eq(group).all() and valid.filename.eq('SHANK3_cohort1.csv').all()
    assert valid.ABL_mapped.isin(ABLS).all() and valid.ILD.isin(ILDS).all()
    np.testing.assert_array_equal(parent.fit_valid, selected)
    abort_rt = (parent.fix_time - parent.intended_fix_time) / 1000
    aborts = parent.abort_type.eq('Fixation') & np.isfinite(abort_rt) & abort_rt.ge(-1) & abort_rt.lt(0)
    assert int(aborts.sum()) == n_aborts
    np.testing.assert_array_equal(parent.display_abort, aborts)
    displayed = parent.loc[selected | aborts].copy()
    displayed['display_rt'] = np.where(selected.loc[displayed.index], displayed.timed_rt,
                                      displayed.fix_time / 1000 - displayed.intended_fix_time / 1000)
    displayed['abs_ILD'] = displayed.ILD.abs()
    display_counts[group] = len(displayed)
    population[group] = valid
    saved_bins = pd.read_csv(folder / 'rtd_data_bins.csv', float_precision='round_trip')
    saved_models = pd.read_csv(folder / 'condition_delay/pooled_rtd_model_1ms.csv.gz', float_precision='round_trip')
    model_bins = pd.read_csv(folder / 'condition_delay/pooled_rtd_model_bins.csv', float_precision='round_trip')
    predictions = pd.read_csv(folder / 'condition_delay/psychometric_predictions.csv', float_precision='round_trip')
    for abl in ABLS:
        for abs_ild in ABS_ILDS:
            actual = select_trace('rtd_data', group, abl, abs_ild)
            saved = saved_bins.loc[saved_bins.ABL.eq(abl) & saved_bins.abs_ILD.eq(abs_ild)].sort_values('bin')
            edges = np.asarray(actual['edges'])
            assert len(edges) == 101
            np.testing.assert_array_equal(edges[:-1], saved.left)
            np.testing.assert_allclose(np.diff(edges), .020, atol=5e-16, rtol=0)
            subset = displayed.loc[displayed.ABL_mapped.eq(abl) & displayed.abs_ILD.eq(abs_ild)]
            counts = np.histogram(subset.display_rt, bins=edges)[0]
            np.testing.assert_array_equal(counts, saved['count'])
            np.testing.assert_allclose(actual['y'], counts / (len(subset) * .020), atol=2e-13, rtol=1e-14)
            np.testing.assert_array_equal(actual['y'], saved.density)
            np.testing.assert_allclose(np.dot(actual['y'], np.diff(edges)), 1, atol=1e-13)
            assert actual['color'] == ABL_COLORS[abl] and actual['linewidth'] == 1.4
            model = select_trace('rtd_model', group, abl, abs_ild)
            saved = saved_models.loc[saved_models.ABL.eq(abl) & saved_models.abs_ILD.eq(abs_ild)].sort_values('rt_s')
            np.testing.assert_array_equal(model['x'], saved.rt_s)
            np.testing.assert_array_equal(model['y'], saved.density)
            np.testing.assert_allclose(np.diff(model['x']), .001, atol=5e-16, rtol=0)
            # Source curves use exact CDF window masses, not a second normalization
            # of the sampled 1 ms density. Retain that normalization unchanged.
            np.testing.assert_array_equal(saved.window_cdf.iloc[[0, -1]], [0, 1])
            masses = model_bins.loc[model_bins.ABL.eq(abl) & model_bins.abs_ILD.eq(abs_ild)]
            np.testing.assert_allclose(masses.window_mass.sum(), 1, atol=1e-12)
            grid_area = trapezoid(model['y'], model['x'])
            np.testing.assert_allclose(grid_area, 1, atol=3e-4, rtol=0)  # Existing source-verifier tolerance.
            model_grid_area_errors.append(abs(grid_area - 1))
            assert model['color'] == ABL_COLORS[abl] and model['linestyle'] == '--' and model['linewidth'] == .8
        data_trace = select_trace('psychometric_data', group, abl)
        model_trace = select_trace('psychometric_model', group, abl)
        saved = predictions.loc[predictions.ABL.eq(abl)].sort_values('ILD')
        np.testing.assert_array_equal(data_trace['x'], ILDS)
        np.testing.assert_array_equal(model_trace['x'], ILDS)
        np.testing.assert_array_equal(data_trace['y'], saved.data_p_right)
        np.testing.assert_array_equal(model_trace['y'], saved.model_p_right)
        assert data_trace['color'] == model_trace['color'] == ABL_COLORS[abl]
        assert model_trace['linewidth'] == 1 and model_trace['linestyle'] == '-'
        for i, ild in enumerate(ILDS):
            z = valid.loc[valid.ABL_mapped.eq(abl) & valid.ILD.eq(ild)]
            assert len(z) == saved.iloc[i].trials
            np.testing.assert_allclose(data_trace['y'][i], np.mean(z.response_poke.eq(1)), atol=1e-15)
    print(group.upper() + ': posterior intervals, all RTD bins, and psychometric points agree', flush=True)
assert parameter_count == 184 and display_counts == {'wt': 81934, 'hom': 80989}

# %% Independently reconstruct pooled means, equal-animal SEMs, and model weights
for row in points.itertuples():
    selected = population[row.group].loc[population[row.group].abs_ILD.eq(row.abs_ILD)]
    separate = row.view == 'ABL_separate'
    abl = int(row.ABL) if separate else None
    if separate:
        selected = selected.loc[selected.ABL_mapped.eq(abl)]
    animal_means = np.array([z.timed_rt.mean() for _, z in selected.groupby('animal')])
    n_animals = len(animal_means)
    sem = np.sqrt(np.sum((animal_means - np.mean(animal_means)) ** 2) / (n_animals - 1)) / np.sqrt(n_animals)
    mean = np.mean(selected.timed_rt)
    assert n_animals == row.animals == EXPECTED[row.group][1] and len(selected) == row.n
    np.testing.assert_allclose([row.data_mean_RT_s, row.animal_SEM_s], [mean, sem], atol=1e-14)
    z = signed.loc[signed.group.eq(row.group) & signed.abs_ILD.eq(row.abs_ILD)]
    if separate:
        z = z.loc[z.ABL.eq(abl)]
    assert z.n.sum() == len(selected)
    model_mean = np.average(z.model_mean_RT_s, weights=z.n)
    np.testing.assert_allclose(model_mean, row.model_mean_RT_s, atol=1e-14)
    actual_data = select_trace('chronometric_data', row.group, abl)
    actual_model = select_trace('chronometric_model', row.group, abl)
    i = ABS_ILDS.index(row.abs_ILD)
    np.testing.assert_allclose([actual_data[k][i] for k in ['y', 'lower', 'upper']],
                              [mean, mean - sem, mean + sem], atol=1e-14)
    np.testing.assert_array_equal(actual_model['y'][i], row.model_mean_RT_s)
    color = ABL_COLORS[abl] if separate else GROUP_COLORS[row.group]
    offset = {20: -.06, 40: 0, 60: .06}[abl] if separate else {'wt': -.04, 'hom': .04}[row.group]
    np.testing.assert_array_equal(actual_data['x'], np.arange(5) + offset)
    np.testing.assert_array_equal(actual_model['x'], actual_data['x'])
    assert actual_data['color'] == actual_model['color'] == color
    assert actual_data['linestyle'] == 'None' and actual_model['linewidth'] == 1
    assert 0 < mean - sem < mean + sem < .30 and 0 < model_mean < .30

# %% Geometry, full limits, output files, and preservation
counts = Counter(t['role'] for t in traces)
assert counts == {'posterior': 22, 'rtd_data': 30, 'rtd_model': 30, 'chronometric_data': 8,
                  'chronometric_model': 8, 'psychometric_data': 6, 'psychometric_model': 6}
assert sum(len(t['y']) for t in traces if t['role'] == 'posterior') == 184
assert len(figures := manifest['figures']) == 2
assert [len(f['panels']) for f in figures] == [11, 15]
for figure in figures:
    for panel in figure['panels']:
        pid = panel['panel']
        if figure is figures[0]:
            block = pid.rsplit('_', 1)[0] if pid not in ['w', 'del_go'] else pid
            assert panel['ylim'] == original['parameter_ylim'][block]
            for trace in [t for t in traces if t['figure'] == figure['stem'] and t['panel'] == pid]:
                assert min(trace['lower']) > panel['ylim'][0] and max(trace['upper']) < panel['ylim'][1]
        elif '_rtd_' in pid:
            assert panel['xlim'] == [-1, 1] and panel['ylim'] == manifest['rtd_ylim']
        else:
            np.testing.assert_allclose(panel['width_inches'], panel['height_inches'], atol=1e-12)
            if 'chronometric' in pid:
                assert panel['ylim'] == [0, .3] and panel['xticklabels'] == ['1', '2', '4', '8', '16']
            else:
                assert panel['xlim'] == [-18, 18] and panel['ylim'] == [0, 1]
for name, expected in manifest['artifacts'].items():
    assert digest(OUT / name) == expected
    if name.endswith('.pdf'):
        info = subprocess.check_output(['pdfinfo', str(OUT / name)], text=True)
        assert int(next(line.split(':')[1] for line in info.splitlines() if line.startswith('Pages:'))) == 1
    else:
        with Image.open(OUT / name) as im:
            assert im.width >= 2800 and im.height >= 2800
baseline = json.loads((OUT / 'preservation_baseline.json').read_text())
for name, expected in baseline.items():
    assert digest(ROOT / name) == expected, name
assert digest(ROOT / manifest['source']) == manifest['source_sha256']
assert digest(OUT / 'plotted_series.json.gz') == manifest['plotted_series_sha256']
assert manifest['refitting'] is False
report = dict(status='passed', parameter_means_and_intervals=184, parameter_panels=11, behavior_panels=15,
    plotted_artist_series=len(traces), plotted_role_counts=dict(counts),
    RTD_data_bins=3000, RTD_model_points=60030, RTD_unit_areas=60,
    max_1ms_model_trapezoid_area_error=max(model_grid_area_errors),
    model_normalization='Exact CDF/window masses retained; sampled-density integrals checked separately',
    chronometric_pairs_and_available_SEMs=40, psychometric_pairs=60,
    groups=manifest['groups'], displayed_RTD_observations=display_counts,
    posterior_summaries_reconstructed_from_saved_draws=True, RTD_counts_reconstructed_from_parent_trials=True,
    raw_trial_means_and_animal_SEMs_verified=True, model_values_and_weights_unchanged=True,
    compact_square_chronometric_and_psychometric_panels=True, single_page_PDFs=2,
    input_fits_prior_exports_and_published_assets_unchanged=True,
    preserved_files=len(baseline), figure_artifact_hashes=manifest['artifacts'],
    source_sha256=digest(Path(__file__)))
(OUT / 'independent_validation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2), flush=True)
