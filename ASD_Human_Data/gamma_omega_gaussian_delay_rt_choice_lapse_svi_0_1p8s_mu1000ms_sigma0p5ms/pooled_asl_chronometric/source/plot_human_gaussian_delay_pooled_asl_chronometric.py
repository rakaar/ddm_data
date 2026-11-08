# %%
"""Add pooled Control/ASD chronometrics to the existing RTD/psychometric figure."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/human_gaussian_delay_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# %% Source and editable plotting settings.
ROOT = Path(__file__).resolve().parents[1]
FIT = ROOT / 'ASD_Human_Data/gamma_omega_gaussian_delay_rt_choice_lapse_svi_0_1p8s_mu1000ms_sigma0p5ms'
GROUPS = ['Control', 'ASD']
COLORS = {'Control': '#1976d2', 'ASD': '#d62728'}
DELTAS = [.1, .3, .65, 1.3, 2.]
STEM = 'human_gaussian_delay_pooled_asl_rtd_psychometric_chronometric'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--fit-dir', type=Path, default=FIT)
parser.add_argument('--output-dir', type=Path)
args = parser.parse_args()
fit = args.fit_dir.resolve()
pooled = fit / 'pooled_asl'
out = args.output_dir.resolve() if args.output_dir else fit / 'pooled_asl_chronometric'
assert out not in [fit, pooled]
out.mkdir(parents=True, exist_ok=True)
print(f'Source fit: {fit}\nOutputs: {out}', flush=True)


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


# %% Use the previously verified count-weighted predictions without a new fit.
manifest = json.loads((pooled / 'plot_manifest.json').read_text())
assert manifest['curve_parameters'] == 'arithmetic physical posterior means'
for path, digest in manifest['source_hashes'].items():
    assert sha256(path) == digest, path
for name, digest in manifest['files'].items():
    assert sha256(pooled / name) == digest, name
sources = [pooled / 'plot_manifest.json'] + [pooled / name for name in manifest['files']]
sources += [fit / 'condition_model_curves.csv', ROOT / 'ASD_Human_Data/simultaneous_rt_trials.csv']
source_hashes = {str(path): sha256(path) for path in sources}
tables = {name: pd.read_csv(pooled / name, float_precision='round_trip') for name in
          ['pooled_rtd_histograms.csv', 'pooled_model_curves.csv', 'pooled_psychometric.csv',
           'trial_membership.csv', 'mixture_weights.csv']}
hist = tables['pooled_rtd_histograms.csv']
curves = tables['pooled_model_curves.csv']
psych = tables['pooled_psychometric.csv']
trials = pd.read_csv(ROOT / 'ASD_Human_Data/simultaneous_rt_trials.csv', float_precision='round_trip')
trials = trials.loc[trials.Condition.eq('RT') & trials.Valid.eq(1) & trials.RT_s.between(0., 1.8)].copy()
trials['signed_delta'] = trials.Delta_dB.round(2)
trials['abs_delta'] = trials.signed_delta.abs()
membership = tables['trial_membership.csv']
pd.testing.assert_frame_equal(membership.sort_values('source_row').reset_index(drop=True),
    trials[membership.columns].sort_values('source_row').reset_index(drop=True))
assert len(trials) == 11344
assert trials.groupby('Group').size().to_dict() == {'Control': 6574, 'ASD': 4770}

# %% Mean RT from raw trials and from the normalized model, including lapses.
points = []
for group in GROUPS:
    for delta in DELTAS:
        obs = trials.loc[trials.Group.eq(group) & trials.abs_delta.eq(delta)]
        curve = curves.loc[curves.Group.eq(group) & curves.abs_delta.eq(delta)].sort_values('time_s')
        t = curve.time_s.to_numpy()
        mass = float(np.trapz(curve.density, t))
        mean = float(np.trapz(t * curve.density, t))
        survival_mean = float(np.trapz(1. - curve.cdf, t))
        assert abs(mass - 1.) < 1e-6
        assert abs(mean - survival_mean) < 1e-6
        points.append(dict(Group=group, abs_delta=delta, n=len(obs),
            participants=obs.Participant.nunique(), data_mean_RT_s=obs.RT_s.mean(),
            model_mean_RT_s=mean, model_mean_from_cdf_s=survival_mean, model_mass=mass))
points = pd.DataFrame(points)

# Check a second route: integrate each signed-ASL component, then weight its mean.
signed_curves = pd.read_csv(fit / 'condition_model_curves.csv', float_precision='round_trip')
weights = tables['mixture_weights.csv'].query("panel == 'rtd'")
component_checks = []
for c in weights.itertuples():
    component = signed_curves.loc[signed_curves.Group.eq(c.Group)
                                 & signed_curves.condition_index.eq(c.condition_index)].sort_values('time_s')
    mean = float(np.trapz(1. - component.cdf, component.time_s))
    component_checks.append(dict(Group=c.Group, abs_delta=c.pooled_delta,
        condition_index=c.condition_index, ASL_cd=c.ASL_cd, signed_delta=c.signed_delta,
        n=c.component_n, weight=c.weight, component_mean_RT_s=mean, weighted_mean_RT_s=c.weight * mean))
component_checks = pd.DataFrame(component_checks)
independent = component_checks.groupby(['Group', 'abs_delta']).weighted_mean_RT_s.sum()
reported = points.set_index(['Group', 'abs_delta']).model_mean_RT_s.sort_index()
np.testing.assert_allclose(reported, independent.sort_index(), atol=1e-6, rtol=0)
empirical = trials.groupby(['Group', 'abs_delta']).RT_s.mean()
np.testing.assert_allclose(points.set_index(['Group', 'abs_delta']).data_mean_RT_s.sort_index(), empirical, atol=1e-14)
points.to_csv(out / 'chronometric_points.csv', index=False, float_format='%.17g')
component_checks.to_csv(out / 'chronometric_component_checks.csv', index=False, float_format='%.17g')

# %% Five overlaid RTDs, psychometric, then chronometric; all means use 0–1.8 s.
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.facecolor': 'white'})
fig = plt.figure(figsize=(26, 5.5))
grid = fig.add_gridspec(1, 7, width_ratios=[1, 1, 1, 1, 1, 1.6, 1.4],
                       left=.035, right=.99, bottom=.25, top=.70, wspace=.38)
ymax = manifest['layout']['rtd_ymax']
for c, delta in enumerate(DELTAS):
    ax = fig.add_subplot(grid[0, c])
    counts = []
    for group in GROUPS:
        h = hist.loc[hist.Group.eq(group) & hist.abs_delta.eq(delta)]
        m = curves.loc[curves.Group.eq(group) & curves.abs_delta.eq(delta)]
        ax.stairs(h.density.to_numpy(), np.r_[h.left_s, h.right_s.iloc[-1]],
                  color=COLORS[group], alpha=.45, lw=1.4)
        ax.plot(m.time_s, m.density, color=COLORS[group], lw=2.2)
        counts.append(int(h.n.iloc[0]))
    ax.set(xlim=(0, 1.8), ylim=(0, ymax), xticks=[0, .6, 1.2, 1.8], xlabel='RT (s)',
           title=f'|ΔdB| = {delta:g}\nn: Control {counts[0]:,} · ASD {counts[1]:,}')
    ax.grid(alpha=.16)
    if c == 0:
        ax.set_ylabel('Density (s⁻¹)')
    else:
        ax.tick_params(labelleft=False)
ax = fig.add_subplot(grid[0, 5])
for group in GROUPS:
    p = psych.loc[psych.Group.eq(group)].sort_values('signed_delta')
    ax.plot(p.signed_delta, p.model_p_right, color=COLORS[group], lw=2.2, label=group)
    ax.scatter(p.signed_delta, p.empirical_p_right, color=COLORS[group], s=28,
               edgecolor='white', linewidth=.4, zorder=3)
ax.set(title='Psychometric\nBoth ASLs pooled', xlabel='Signed ΔdB (right − left)',
       ylabel='P(right)', xlim=(-2.15, 2.15), ylim=(-.015, 1.015), xticks=[-2, -1, 0, 1, 2])
ax.grid(alpha=.18)
ax.legend(loc='lower right', fontsize=9)

ax = fig.add_subplot(grid[0, 6])
for group in GROUPS:
    p = points.loc[points.Group.eq(group)].sort_values('abs_delta')
    ax.plot(p.abs_delta, p.model_mean_RT_s, color=COLORS[group], lw=2.2)
    ax.scatter(p.abs_delta, p.data_mean_RT_s, color=COLORS[group], s=28,
               edgecolor='white', linewidth=.4, zorder=3)
ax.set(title='Chronometric\nBoth ASLs pooled', xlabel='|ΔdB|', ylabel='Mean RT (s)',
       xticks=DELTAS, xticklabels=['0.1', '0.3', '0.65', '1.3', '2'], xlim=(0, 2.1))
values = points[['data_mean_RT_s', 'model_mean_RT_s']].to_numpy()
ax.set_ylim(np.floor((values.min() - .035) * 20) / 20, np.ceil((values.max() + .035) * 20) / 20)
ax.grid(alpha=.18)
fig.suptitle('Human simultaneous RT | both ASLs pooled | data and model', fontsize=19, y=.97)
fig.text(.5, .90, 'Control: 6,574 trials, 11 participants   |   ASD: 4,770 trials, 8 participants   |   ASL 20 + 40 cd/m²', ha='center', fontsize=11)
fig.legend(handles=[
    Line2D([], [], color='.45', lw=1.4, alpha=.6, drawstyle='steps-mid', label='Data: 20 ms RT bins'),
    Line2D([], [], color='.45', marker='o', linestyle='none', label='Data: choice proportions / mean RT'),
    Line2D([], [], color='.2', lw=2.2, label='Model at posterior means'),
    Line2D([], [], color=COLORS['Control'], lw=2.2, label='Control'),
    Line2D([], [], color=COLORS['ASD'], lw=2.2, label='ASD'),
], loc='upper center', bbox_to_anchor=(.5, .855), ncol=5, frameon=False)
fig.text(.5, .125, 'Condition = RT; Valid = 1; 0–1.8 s retained | Both choices; raw trials pooled across participants, sessions and ASLs', ha='center', fontsize=10)
fig.text(.5, .078, 'Gaussian delay + choice + uniform lapses | Existing ASL-specific predictions combined using retained-trial weights; no refit', ha='center', fontsize=10)
fig.text(.5, .031, 'RTDs and chronometrics pool ±Δ | Chronometric: raw-trial mean versus mean of the same normalized model RTD | Model points joined at measured differences', ha='center', fontsize=10)
for suffix in ['png', 'pdf']:
    fig.savefig(out / f'{STEM}.{suffix}', dpi=180)
plt.close(fig)

# %% Audit the new moment calculation and preservation of every reused artifact.
for path, digest in source_hashes.items():
    assert sha256(path) == digest, path
report = dict(passed=True, selected_trial_identities=len(trials), chronometric_points=len(points),
    x_axis='absolute Delta_dB; signs and ASLs pooled using retained-trial counts',
    data_definition='arithmetic raw-trial mean RT, both choices, no error bars',
    model_definition='integral of t*f(t) over 0–1.8 s, same lapse-inclusive normalized density as RTD',
    maximum_density_vs_cdf_mean_error_s=float(abs(points.model_mean_RT_s-points.model_mean_from_cdf_s).max()),
    maximum_component_mean_error_s=float(abs(reported-independent.sort_index()).max()),
    maximum_unit_mass_error=float(abs(points.model_mass-1).max()),
    original_artifacts_unchanged=True, source_sha256=sha256(__file__))
(out / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
manifest = dict(source=str(Path(__file__).resolve()), source_sha256=sha256(__file__),
    fit_dir=str(fit), pooled_source_dir=str(pooled), source_hashes=source_hashes,
    selection='Condition == RT, Valid == 1, 0 <= RT_s <= 1.8', refitted=False,
    curve_parameters='arithmetic physical posterior means',
    counts={'Control': 6574, 'ASD': 4770}, layout='five overlaid RTDs, psychometric, chronometric',
    files={p.name: sha256(p) for p in out.iterdir() if p.is_file() and p.suffix in ['.csv', '.png', '.pdf']})
(out / 'plot_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(points[['Group', 'abs_delta', 'n', 'data_mean_RT_s', 'model_mean_RT_s']].to_string(index=False))
print(f'Saved {out / (STEM + ".png")}')
