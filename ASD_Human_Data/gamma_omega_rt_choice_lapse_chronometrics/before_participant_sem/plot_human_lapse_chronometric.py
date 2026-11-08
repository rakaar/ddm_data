# %%
"""Pooled human chronometrics from the saved joint RT/choice/lapse SVI fit."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/human_lapse_chronometric_mpl')
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.integrate import quad, trapezoid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from human_gamma_omega_variants_svi_utils import ROOT, sha256, load_data, sample_matrix
from human_plain_ddm_diagnostics_utils import scipy_ddm

# %% Editable analysis settings. Reuse the completed fit without optimization.
FIT_DIR = ROOT / 'ASD_Human_Data/gamma_omega_rt_choice_lapse_svi_0_1p8s_delay500ms'
OUTPUT_DIR = ROOT / 'ASD_Human_Data/gamma_omega_rt_choice_lapse_chronometrics'
GROUPS = ['Control', 'ASD']
ASLS = [20, 40]
ABS_DELTA = [.1, .3, .65, 1.3, 2.]
RT_UPPER = 1.8
ASL_COLORS = {20: '#1f77b4', 40: '#ff7f0e'}
GROUP_COLORS = {'Control': '#1976d2', 'ASD': '#d62728'}
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% Verify the saved selection and freeze the earlier fit artifacts.
config = json.loads((FIT_DIR / 'run_config.json').read_text())
assert config['mode'] == 'rt_choice_lapse' and config['free_parameters_per_group'] == 63
assert config['rt_window_s'] == [0., RT_UPPER]
for name, expected in config['source_hashes'].items():
    assert sha256(ROOT / name) == expected, name
prior_roots = [ROOT / 'ASD_Human_Data' / name for name in [
    'gamma_omega_rt_choice_svi_0_1p8s_delay500ms',
    'gamma_omega_rt_only_svi_0_1p8s_delay500ms',
    'gamma_omega_rt_choice_lapse_svi_0_1p8s_delay500ms',
]]
preserved = {str(p.relative_to(ROOT)): sha256(p) for folder in prior_roots
             for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
selected = pd.read_csv(FIT_DIR / 'selected_trials.csv', float_precision='round_trip')
_, reconstructed, _, _ = load_data('rt_choice_lapse')
np.testing.assert_array_equal(selected.source_row, reconstructed.source_row)
assert selected.source_row.is_unique and len(selected) == 11344
assert selected.Condition.eq('RT').all() and selected.Valid.eq(1).all()
assert selected.RT_s.between(0, RT_UPPER).all()
assert sorted(selected.ASL_cd.unique()) == ASLS
assert selected.Choice.isin([-1, 1]).all()
curves = pd.read_csv(FIT_DIR / 'condition_model_curves.csv', float_precision='round_trip')
signed_rows, point_rows, checks = [], [], []
sources = [Path(__file__), FIT_DIR / 'run_config.json', FIT_DIR / 'selected_trials.csv',
           FIT_DIR / 'condition_model_curves.csv']

# %% Condition means: integrate the saved analytic densities at posterior means.
# Independent SciPy quadrature checks the retained DDM mean and uniform lapse term.
for group in GROUPS:
    folder = FIT_DIR / group.lower()
    condition_table = pd.read_csv(folder / 'conditions.csv', float_precision='round_trip')
    summary = pd.read_csv(folder / 'posterior_summary.csv', float_precision='round_trip')
    samples = dict(np.load(folder / 'posterior_samples.npz'))
    matrix = sample_matrix(samples, 'rt_choice_lapse')
    np.testing.assert_allclose(matrix.mean(0), summary['mean'], rtol=1e-12, atol=1e-12)
    means = summary.set_index('parameter')['mean']
    p, w = float(means['lapse_prob']), float(means['w'])
    sub = selected.loc[selected.Group.eq(group)]
    assert (len(sub), sub.Participant.nunique()) == ((6574, 11) if group == 'Control' else (4770, 8))
    sources += [folder / name for name in ['conditions.csv', 'posterior_summary.csv', 'posterior_samples.npz']]
    for condition in condition_table.itertuples():
        i = condition.condition_index
        observed = sub.loc[sub.ASL_cd.eq(condition.ASL_cd) & np.isclose(sub.signed_delta, condition.signed_delta)]
        assert len(observed) == condition.n
        c = curves.loc[curves.Group.eq(group) & curves.condition_index.eq(i)].sort_values('time_s')
        t, density, cdf = c.time_s.to_numpy(), c.density.to_numpy(), c.cdf.to_numpy()
        assert t[0] == 0 and t[-1] == RT_UPPER and len(t) == 1801
        density_mean = float(trapezoid(t * density, t))
        cdf_mean = float(RT_UPPER - trapezoid(cdf, t))
        gamma, omega, delay = [float(means[f'{name}_{i:02d}']) for name in ['gamma', 'omega', 't_E_aff']]
        retained_mass = float(scipy_ddm(RT_UPPER, gamma, omega, w, delay, cumulative=True, K=100))
        switch = delay + .6 / omega
        splits = [switch] if delay < switch < RT_UPPER else []
        ddm_mass, _ = quad(lambda rt: float(scipy_ddm(rt, gamma, omega, w, delay, K=100)) / retained_mass,
                           delay, RT_UPPER, points=splits, epsabs=2e-10, epsrel=2e-10)
        ddm_mean, _ = quad(lambda rt: rt * float(scipy_ddm(rt, gamma, omega, w, delay, K=100)) / retained_mass,
                           delay, RT_UPPER, points=splits, epsabs=2e-10, epsrel=2e-10)
        independent_mean = (1-p) * ddm_mean + p * RT_UPPER / 2
        assert abs(ddm_mass - 1) < 1e-7
        assert abs(density_mean - cdf_mean) < 2e-6
        assert abs(density_mean - independent_mean) < 2e-6
        signed_rows.append(dict(Group=group, condition_index=i, ASL_cd=int(condition.ASL_cd),
            signed_delta=condition.signed_delta, n=condition.n, participants=observed.Participant.nunique(),
            data_mean_RT_s=float(observed.RT_s.mean()), model_mean_RT_s=density_mean,
            independent_model_mean_RT_s=independent_mean, ddm_mean_RT_s=ddm_mean,
            lapse_probability=p, lapse_mean_RT_s=RT_UPPER/2))
        checks.append(dict(Group=group, condition_index=i, ddm_mass_error=abs(ddm_mass - 1),
            mean_density_cdf_error_s=abs(density_mean-cdf_mean),
            mean_independent_quad_error_s=abs(density_mean-independent_mean)))
    group_means = pd.DataFrame(signed_rows).loc[lambda x: x.Group.eq(group)]
    for asl in ASLS:
        for delta in ABS_DELTA:
            observed = sub.loc[sub.ASL_cd.eq(asl) & np.isclose(sub.signed_delta.abs(), delta)]
            signed = group_means.loc[group_means.ASL_cd.eq(asl) &
                                     np.isclose(group_means.signed_delta.abs(), delta)].sort_values('signed_delta')
            assert len(signed) == 2 and int(signed.n.sum()) == len(observed)
            weights = signed.n.to_numpy() / len(observed)
            empirical = float(observed.RT_s.mean())
            np.testing.assert_allclose(empirical, signed.data_mean_RT_s.to_numpy() @ weights, atol=1e-14)
            point_rows.append(dict(Group=group, ASL_cd=asl, abs_delta=delta, n=len(observed),
                participants=observed.Participant.nunique(), data_mean_RT_s=empirical,
                model_mean_RT_s=float(signed.model_mean_RT_s.to_numpy() @ weights),
                independent_model_mean_RT_s=float(signed.independent_model_mean_RT_s.to_numpy() @ weights),
                negative_n=int(signed.n.iloc[0]), positive_n=int(signed.n.iloc[1]),
                negative_weight=weights[0], positive_weight=weights[1]))

signed_table, points, validation = pd.DataFrame(signed_rows), pd.DataFrame(point_rows), pd.DataFrame(checks)
assert len(signed_table) == 40 and len(points) == 20
signed_table.to_csv(OUTPUT_DIR / 'signed_condition_means.csv', index=False, float_format='%.17g')
points.to_csv(OUTPUT_DIR / 'chronometric_points.csv', index=False, float_format='%.17g')
validation.to_csv(OUTPUT_DIR / 'integration_checks.csv', index=False, float_format='%.17g')

# %% Two group figures; identical axes, colored ASLs, data dots and model lines.
all_means = points[['data_mean_RT_s', 'model_mean_RT_s']].to_numpy()
low, high = float(all_means.min()), float(all_means.max())
pad = max(.025, .15 * (high-low))
figures = []
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.facecolor': 'white'})
for group in GROUPS:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    fig.subplots_adjust(left=.13, right=.97, bottom=.21, top=.79)
    for asl in ASLS:
        p = points.loc[points.Group.eq(group) & points.ASL_cd.eq(asl)].sort_values('abs_delta')
        ax.plot(p.abs_delta, p.model_mean_RT_s, color=ASL_COLORS[asl], lw=2.2, zorder=2)
        ax.scatter(p.abs_delta, p.data_mean_RT_s, color=ASL_COLORS[asl],
                   marker='o' if asl == 20 else 's', s=64 if asl == 20 else 38,
                   edgecolor='white', linewidth=.6, zorder=3)
    ax.set(xlabel='Absolute ΔdB (visual ILD)', ylabel='Mean RT (s)',
           xlim=(.04, 2.06), ylim=(max(0, low-pad), high+pad),
           xticks=ABS_DELTA, xticklabels=['0.1', '0.3', '0.65', '1.3', '2'])
    ax.grid(alpha=.2)
    handles = [Line2D([], [], color=ASL_COLORS[asl], lw=2, label=f'ASL {asl} cd/m²') for asl in ASLS]
    handles += [Line2D([], [], color='.2', ls='', marker='o', label='Data mean'),
                Line2D([], [], color='.2', lw=2, label='Model mean')]
    ax.legend(handles=handles, ncol=2, fontsize=9, frameon=False, loc='upper right')
    n = int(points.loc[points.Group.eq(group), 'n'].sum())
    participants = selected.loc[selected.Group.eq(group), 'Participant'].nunique()
    fig.suptitle(f'{group} | RT + choice + lapse chronometric', y=.96, fontsize=15, color=GROUP_COLORS[group])
    fig.text(.5, .875, f'{participants} participants pooled | {n:,} RT trials | 0–1.8 s retained', ha='center', fontsize=10)
    fig.text(.5, .09, 'Both choices included; ±Δ pooled using actual retained trial counts.', ha='center', fontsize=9)
    fig.text(.5, .045, 'Model: arithmetic posterior means, including the Uniform(0, 1.8 s) lapse component.', ha='center', fontsize=9)
    stem = f'human_{group.lower()}_rt_choice_lapse_chronometric'
    for suffix in ['png', 'pdf']:
        name = f'{stem}.{suffix}'
        fig.savefig(OUTPUT_DIR / name, dpi=200)
        figures.append(name)
    plt.close(fig)

# %% Provenance and independent integration checks, without modifying prior fits.
for name, expected in preserved.items():
    assert sha256(ROOT / name) == expected, name
report = dict(status='passed', fit_root=str(FIT_DIR.relative_to(ROOT)), mode=config['mode'],
    selection='Condition == RT, Valid == 1, 0 <= RT_s <= 1.8; all choices; pooled raw trials',
    counts={g: int(selected.Group.eq(g).sum()) for g in GROUPS},
    asl_cd=ASLS, absolute_delta_dB=ABS_DELTA, plotted_points=len(points), signed_conditions=len(signed_table),
    data='Arithmetic mean of all retained RTs; no correctness filter or participant reweighting',
    model='Integral of RT times conditional either-choice density at arithmetic physical posterior means; signs mixed by retained counts',
    lapse='Retained-trial mixture, Uniform(0,1.8 s) in stored RT, no delay shift',
    line='Connect only five measured absolute-difference predictions; no new fit or stimulus interpolation',
    maximum_mean_quad_error_s=float(validation.mean_independent_quad_error_s.max()),
    maximum_mean_density_cdf_error_s=float(validation.mean_density_cdf_error_s.max()),
    maximum_conditional_mass_error=float(validation.ddm_mass_error.max()),
    preserved_prior_files=len(preserved), preserved_sha256=preserved,
    sources={str(p.relative_to(ROOT)): sha256(p) for p in sources},
    figures={name: sha256(OUTPUT_DIR / name) for name in figures})
(OUTPUT_DIR / 'validation.json').write_text(json.dumps(report, indent=2)+'\n')
print(points[['Group', 'ASL_cd', 'abs_delta', 'n', 'data_mean_RT_s', 'model_mean_RT_s']].to_string(index=False))
print('Validated', len(points), 'chronometric points; maximum independent mean error', report['maximum_mean_quad_error_s'], 's')
