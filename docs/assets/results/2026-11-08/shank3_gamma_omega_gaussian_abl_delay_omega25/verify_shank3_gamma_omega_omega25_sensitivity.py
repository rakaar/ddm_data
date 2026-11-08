# %% Compare the 25-bound fits with the unchanged 15-bound and NPL+alpha fits
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'asd_analysis/shank3_cohort1_wt_hom_gamma_omega_gaussian_delay_by_abl'
OUT = ROOT/'asd_analysis/shank3_cohort1_wt_hom_gamma_omega_gaussian_delay_by_abl_omega25'
NPL = ROOT/'asd_analysis/shank3_cohort1_wt_hom_npl_alpha_gaussian_delay_by_abl'
ABLS = [20, 40, 60]
rows, metrics, boundary_rows = [], [], []

for group in ['wt', 'hom']:
    old, new = OLD/group, OUT/group
    a = json.loads((old/'seed_0/configuration.json').read_text())
    b = json.loads((new/'seed_0/configuration.json').read_text())
    assert b['priors']['omega'] == [.01, .1, 12., 25.]
    assert a['priors']['omega'] == [.01, .1, 12., 15.]
    for name in a['priors']:
        if name != 'omega':
            assert a['priors'][name] == b['priors'][name]
    for key in ['initial', 'seed', 'parameter_count', 'learning_rate', 'clip_norm', 'guide_scale',
                'window', 'min_steps', 'max_steps', 'confirmation_windows', 'parameter_stability_required',
                'delay_grid_s', 'delay_grid_end_s', 'adaptive_refinement_factors', 'quadrature_order',
                'draws', 'series_terms', 'jax', 'numpyro', 'float64', 'backend', 'device']:
        assert a[key] == b[key], key
    for name in ['fit_trials.csv.gz', 'parent_trials.csv.gz', 'likelihood_data.npz', 'conditions.csv',
                 'membership.csv', 'fixed_proactive.json', 'initialization.json', 'initialization_rng.json']:
        assert (old/name).read_bytes() == (new/name).read_bytes(), name
    conditions = pd.read_csv(new/'conditions.csv')
    estimates = [pd.read_csv(p/'parameter_estimates.csv', float_precision='round_trip')
                 for p in [OLD, OUT]]
    for block in ['gamma', 'omega', 'delay_mu', 'delay_sigma', 'w', 'del_go']:
        first, second = [z.loc[z.group.eq(group) & z.block.eq(block)].sort_values('index')
                         for z in estimates]
        assert first.parameter.tolist() == second.parameter.tolist()
        for i in range(len(first)):
            before, after = first.iloc[i], second.iloc[i]
            scale = 1000 if block in ['delay_mu', 'delay_sigma', 'del_go'] else 1
            rows.append(dict(group=group, block=block, parameter=after.parameter,
                ABL=after.ABL, ILD=after.ILD, old_mean=before['mean'], new_mean=after['mean'],
                mean_change=after['mean']-before['mean'], display_scale=scale,
                display_mean_change=(after['mean']-before['mean'])*scale,
                old_lower95=before.lower95, old_upper95=before.upper95,
                new_lower95=after.lower95, new_upper95=after.upper95))
    for ceiling, base in [(15, old), (25, new)]:
        with np.load(base/'seed_0/posterior_samples.npz') as saved:
            for ai, abl in enumerate(ABLS):
                x = saved['omega'][:, ai*10:(ai+1)*10]
                means = x.mean(0)
                upper95 = np.quantile(x, .975, axis=0)
                boundary_rows.append(dict(group=group, omega_hard_upper=ceiling, ABL=abl,
                    minimum_condition_mean=means.min(), maximum_condition_mean=means.max(),
                    means_above_14=int(np.sum(means > 14)), means_above_24=int(np.sum(means > 24)),
                    means_within_1_of_upper=int(np.sum(means > ceiling-1)),
                    maximum_upper95=upper95.max(),
                    equal_condition_draw_fraction_within_1_of_upper=float(np.mean(x > ceiling-1)),
                    equal_condition_draw_fraction_within_1pct_of_upper=float(np.mean(x > ceiling-.01*(ceiling-.01)))))
    for label, base in [('direct_omega15', old), ('direct_omega25', new), ('npl_alpha', NPL/group)]:
        run = json.loads((base/'seed_0/run_summary.json').read_text())
        assert run['status'] == 'converged'
        psych = pd.read_csv(base/'psychometric_predictions.csv', float_precision='round_trip')
        assert psych.trials.sum() == (68184 if group == 'wt' else 67715)
        metrics.append(dict(group=group, model=label, valid_trials=int(psych.trials.sum()),
            joint_log_likelihood=run['posterior_mean_joint_log_likelihood'],
            psychometric_rmse=np.sqrt(np.mean((psych.model_p_right-psych.data_p_right)**2)),
            mean_max_rt_cdf_gap=psych.rt_cdf_max_abs_gap_1ms.mean(),
            best_step=run['restored_best_step'], final_step=run['final_checked_step']))
    pd.testing.assert_frame_equal(pd.read_csv(old/'rtd_data_bins.csv'), pd.read_csv(new/'rtd_data_bins.csv'), check_exact=True)
    for name in ['ABL', 'ILD', 'trials', 'right', 'left', 'data_p_right']:
        np.testing.assert_array_equal(pd.read_csv(old/'psychometric_predictions.csv')[name],
                                      pd.read_csv(new/'psychometric_predictions.csv')[name])

baseline = json.loads((OUT/'preservation_baseline.json').read_text())['hashes']
for relative, digest in baseline.items():
    with (ROOT/relative).open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest, relative
pd.DataFrame(rows).to_csv(OUT/'omega_bound_parameter_sensitivity.csv', index=False, float_format='%.17g')
pd.DataFrame(boundary_rows).to_csv(OUT/'omega_bound_pressure_by_ABL.csv', index=False, float_format='%.17g')
pd.DataFrame(metrics).to_csv(OUT/'omega_bound_in_sample_comparison.csv', index=False, float_format='%.17g')
report = dict(status='passed', previous_files_preserved=len(baseline), parameter_comparisons=len(rows),
    only_changed_prior='Omega hard upper 15 to 25; flat interval remains 0.1 to 12',
    identical_initialization=True, identical_trials_and_fixed_values=True,
    identical_other_settings=True, empirical_arrays_unchanged=True,
    comparison='in-sample posterior-mean plug-in; one initialization per fit')
(OUT/'sensitivity_validation.json').write_text(json.dumps(report, indent=2)+'\n')
print(pd.DataFrame(metrics).to_string(index=False), flush=True)
print(json.dumps(report, indent=2), flush=True)
