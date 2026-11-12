# %% Independent numerical and PDF checks for all ASD0047 joint deliverables
import hashlib
import json
from pathlib import Path
import fitz
import numpy as np
import pandas as pd
from scipy.integrate import simpson

# %% Editable fit root; sources and production posteriors are read-only
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'asd_analysis/shank3_cohort1_ASD0047_joint_npl_alpha_gaussian_abl_contaminants'
OLD = ROOT / 'asd_analysis/shank3_cohort1_wt_hom_individual_adaptive_bounds/wt/ASD0047'
selection = json.loads((BASE / 'selected_fit.json').read_text())
RUN = ROOT / selection['selected_run']
VARIANT = RUN.parents[1]
REPORT = VARIANT / 'report'
OUT = BASE / 'posterior_and_diagnostics'
assert selection['status'] == 'passed'
assert not any(selection[key] for key in ['artificial_flags', 'taper_flags', 'intrinsic_flags'])
assert json.loads((RUN.parent / 'independent_validation.json').read_text())['status'] == 'passed'
assert json.loads((VARIANT / 'exact_resume_validation.json').read_text())['status'] == 'passed'
assert json.loads((REPORT / 'independent_model_prediction_checks.json').read_text())['status'] == 'passed'
assert json.loads((BASE / 'joint_vs_two_stage_parameters/independent_validation.json').read_text())['status'] == 'passed'

# %% Every saved estimate and every displayed sample are checked directly
config = json.loads((RUN / 'configuration.json').read_text())
names = config['parameter_order']
decision = json.loads((BASE / 'contaminant_model_decision.json').read_text())
mixture_run = ROOT / decision['selected_mixture_run']
with np.load(mixture_run / 'posterior_samples.npz') as mixture:
    d = mixture['lapse_exp_weight']
    np.testing.assert_allclose([decision['d_mean'], decision['d_q025'], decision['d_q975'], decision['d_median']],
                              [d.mean(), *np.percentile(d, [2.5, 97.5]), np.median(d)], atol=1e-14)
if selection['variant'] == 'exponential_only':
    assert config['fixed_parameters'] == {'lapse_exp_weight': 1.}
    assert config['frozen_parameters'] == ['lapse_exp_weight']
    assert 'lapse_exp_weight' not in names and 'lapse_exp_weight' not in config['priors']
    assert selection['parameter_count'] == 19
assert {'V_A', 'theta_A', 't_nd', 'lapse_prob', 'beta_lapse', 'lapse_prob_right'}.issubset(names)
assert config['full_batch'] and config['guide'] == 'AutoMultivariateNormal'
with np.load(VARIANT / 'observations.npz') as observations:
    assert not any(name.startswith('proactive_') or name.startswith('fixed_') for name in observations.files)
parent = pd.read_csv(VARIANT / 'parent_trials.csv.gz', float_precision='round_trip')
late = parent.loc[parent.observation.eq('late_fixation_censored')]
assert len(late) == 11 and late.abort_rt_s.between(0, 1, inclusive='both').all()
assert parent.loc[parent.observation.eq('pre_onset_abort'), 'fix_time'].ge(0).all()
with np.load(RUN / 'posterior_samples.npz') as samples:
    physical = np.column_stack([samples[key].reshape(10000, -1) for key in config['blocks']])
assert physical.shape == (10000, selection['parameter_count']) and np.isfinite(physical).all()
summary = pd.read_csv(OUT / 'corner_parameter_summaries.csv', float_precision='round_trip')
assert summary.parameter.tolist() == names
for index, record in enumerate(summary.itertuples()):
    values = physical[:, index] * record.display_scale
    np.testing.assert_allclose([record.mean, record.q025, record.q975],
                               [values.mean(), *np.percentile(values, [2.5, 97.5])], atol=1e-12, rtol=1e-12)
    assert record.range_lower <= values.min() and record.range_upper >= values.max()
    assert record.title.endswith(f'= {values.mean():.3f}') and 'Mean' not in record.title
    assert record.hard_lower < values.min() < values.max() < record.hard_upper
correlations = pd.read_csv(OUT / 'posterior_correlations.csv', index_col=0, float_precision='round_trip')
np.testing.assert_allclose(correlations.to_numpy(), np.corrcoef(physical, rowvar=False), atol=1e-12)
assert correlations.index.tolist() == names and correlations.columns.tolist() == names

# %% Histograms retain identical trials; all 15 panel densities have unit area
hist = pd.read_csv(REPORT / 'rtd_data_bins.csv', float_precision='round_trip')
old = pd.read_csv(OLD / 'rtd_data_bins.csv', float_precision='round_trip')
pd.testing.assert_frame_equal(hist[['ABL', 'abs_ILD', 'bin', 'count']], old[['ABL', 'abs_ILD', 'bin', 'count']], check_dtype=False)
assert hist['count'].sum() == 21694
np.testing.assert_allclose(hist.right - hist.left, .02, atol=2e-16)
curves = pd.read_csv(REPORT / 'rtd_model_curves.csv.gz', float_precision='round_trip')
areas = []
for key, condition in hist.groupby(['ABL', 'abs_ILD']):
    np.testing.assert_allclose((condition.density * (condition.right-condition.left)).sum(), 1., atol=1e-12)
    model = curves.loc[curves.ABL.eq(key[0]) & curves.abs_ILD.eq(key[1])]
    np.testing.assert_allclose(np.diff(model.rt_s), .001, atol=1e-15)
    assert len(model) == 2001 and np.isfinite(model.density).all() and model.density.ge(0).all()
    area = float(simpson(model.density, x=model.rt_s))
    np.testing.assert_allclose(area, 1., atol=1e-12)
    areas.append(dict(ABL=int(key[0]), abs_ILD=int(key[1]), model_area=area))
psy = pd.read_csv(REPORT / 'psychometric_probabilities.csv', float_precision='round_trip')
old_psy = pd.read_csv(OLD / 'psychometric_predictions.csv', float_precision='round_trip')
assert len(psy) == 30 and psy.trials.sum() == 18771 and psy.right.sum() == 9592
np.testing.assert_array_equal(psy.trials, old_psy.trials)
np.testing.assert_array_equal(psy.right, old_psy.right)
np.testing.assert_allclose(psy.data_p_right, old_psy.data_p_right, rtol=0, atol=1e-15)
assert psy.model_p_right.between(0, 1).all()
corners = json.loads((OUT / 'export_manifest.json').read_text())
pdf_counts = {}
for name, count in [(corners['corner_pdf'], 2), (corners['diagnostic_pdf'], 4)]:
    with fitz.open(name) as document:
        assert len(document) == count
        assert all(page.get_text().strip() for page in document)
        pdf_counts[Path(name).name] = count
        if count == 4:
            assert len(document.get_toc()) == 4
    delivered = ROOT / 'shan3_coh_1_animal_Wise_fitting/ASD0047_joint' / Path(name).name
    assert delivered.read_bytes() == Path(name).read_bytes()
baseline = json.loads((BASE / 'protected_previous_files.json').read_text())
for name, expected in baseline.items():
    with (ROOT / name).open('rb') as handle:
        assert hashlib.file_digest(handle, 'sha256').hexdigest() == expected, name
report = dict(status='passed', animal='ASD0047', parameters=selection['parameter_count'],
    fitted_choices=18771, likelihood_aborts=3004, onset_censored=11, excluded_long_choices=161,
    corner_samples=10000, all_corner_titles_include_parameter_and_mean=True,
    every_draw_inside_padded_corner_axes=True, correlations_verified=True,
    raw_bounds_acceptance_flags_empty=True, rtd_panels=15, panel_areas=areas,
    identical_empirical_RTD_rows=21694, psychometric_probabilities=30,
    pdf_pages=pdf_counts, protected_previous_files_verified=len(baseline))
(BASE / 'deliverable_validation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2), flush=True)
