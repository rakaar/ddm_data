# %% ASD0047 only: joint estimation versus proactive-first estimation
"""Compare physical posterior draws without refitting or cross-animal pooling."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# %% Editable roots and appearance
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'asd_analysis/shank3_cohort1_ASD0047_joint_npl_alpha_gaussian_abl_contaminants'
chosen = json.loads((BASE / 'selected_fit.json').read_text())
JOINT = (ROOT / chosen['selected_run']).parents[1]
TWO_STAGE = ROOT / 'asd_analysis/shank3_cohort1_wt_hom_individual_adaptive_bounds/wt/ASD0047'
OUT = BASE / 'joint_vs_two_stage_parameters'
ABLS = [20, 40, 60]
GLOBALS = ['rate_lambda', 'T_0', 'theta_E', 'rate_norm_l', 'alpha', 'w', 'del_go']
PROACTIVE = ['V_A', 'theta_A', 't_nd', 'lapse_prob', 'beta_lapse', 'lapse_prob_right']
COLORS = {'two_stage': '#4D4D4D', 'joint': '#0072B2'}
LABELS = {'two_stage': 'Two-stage: proactive first, then freeze', 'joint': 'Joint fit: all estimated together'}
DPI, SIDE = 190, 2.65
ABL_OFFSETS = {'two_stage': -1., 'joint': 1.}  # Additive offsets in dB.
REACTIVE_STEM = 'SHANK3_cohort1_ASD0047_joint_vs_two_stage_reactive_parameters'
PROACTIVE_STEM = 'SHANK3_cohort1_ASD0047_joint_vs_two_stage_proactive_lapse_parameters'
OUT.mkdir(exist_ok=True)
MIXTURE = chosen['parameter_count'] == 20
if MIXTURE:
    PROACTIVE.append('lapse_exp_weight')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


# %% Resolve the exact selected fits and the original frozen-parameter source
selection = json.loads((JOINT / 'selected_fit.json').read_text())
assert selection['status'] == 'passed' and selection['parameter_count'] in [19, 20]
joint_run = ROOT / selection['selected_run']
two_stage_run = ROOT / json.loads((TWO_STAGE / 'reactive_selected_run.json').read_text())['selected_run']
fixed = json.loads((TWO_STAGE / 'fixed_proactive.json').read_text())
abort_run = (ROOT / fixed['source']).parent
joint_config = json.loads((joint_run / 'configuration.json').read_text())
reactive_config = json.loads((two_stage_run / 'configuration.json').read_text())
abort_config = json.loads((abort_run / 'configuration.json').read_text())
assert joint_config['fixed_parameters'] == ({} if MIXTURE else {'lapse_exp_weight': 1.})
sources = [JOINT / 'selected_fit.json', TWO_STAGE / 'fixed_proactive.json',
           JOINT / 'likelihood_policy.json', JOINT / 'preparation.json',
           TWO_STAGE / 'preparation.json', JOINT / 'fit_trials.csv.gz',
           TWO_STAGE / 'fit_trials.csv.gz', JOINT / 'parent_trials.csv.gz',
           TWO_STAGE / 'parent_trials.csv.gz', ROOT / 'asd_analysis/session_type_csvs/ASD_session_type_1.csv',
           Path(__file__)]
for folder in [joint_run, two_stage_run, abort_run]:
    sources.extend(folder / name for name in ['posterior_samples.npz', 'posterior_summary.csv', 'configuration.json'])
source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in sources}
with np.load(joint_run / 'posterior_samples.npz') as saved:
    joint_samples = {name: saved[name].copy() for name in saved.files}
with np.load(two_stage_run / 'posterior_samples.npz') as saved:
    reactive_samples = {name: saved[name].copy() for name in saved.files}
with np.load(abort_run / 'posterior_samples.npz') as saved:
    abort_samples = {name: saved[name].copy() for name in saved.files}
assert all(len(values) == 10000 for samples in [joint_samples, reactive_samples, abort_samples]
           for values in samples.values())
for name, value in zip(fixed['parameter_order'], fixed['values']):
    np.testing.assert_allclose(abort_samples[name].mean(), value, rtol=0, atol=1e-12)

# %% Reconstruct means and central 95% credible intervals from all saved draws
records = []
for method in ['two_stage', 'joint']:
    for part, blocks in [('reactive', GLOBALS + ['delay_mu', 'delay_sigma']), ('proactive_lapse', PROACTIVE)]:
        for block in blocks:
            is_fixed = method == 'two_stage' and block in ['lapse_prob_right', 'lapse_exp_weight']
            samples = joint_samples if method == 'joint' else (reactive_samples if part == 'reactive' else abort_samples)
            vectors = np.array([[.5 if block == 'lapse_prob_right' else 1.]]) if is_fixed else samples[block].reshape(10000, -1)
            config = joint_config if method == 'joint' else (reactive_config if part == 'reactive' else abort_config)
            prior = [np.nan] * 4 if is_fixed else config.get('priors', config.get('prior_bounds'))[block]
            for index in range(vectors.shape[1]):
                values = vectors[:, index]
                scale = 1000. if block in ['T_0', 'del_go', 't_nd', 'delay_mu', 'delay_sigma'] else 1.
                lower, upper = [np.nan, np.nan] if is_fixed else np.quantile(values, [.025, .975])
                records.append(dict(method=method, part=part, block=block,
                    parameter=f'{block}_ABL{ABLS[index]}' if block.startswith('delay_') else block,
                    ABL=ABLS[index] if block.startswith('delay_') else np.nan,
                    mean=values.mean(), q025=lower, q975=upper,
                    estimate_type='fixed_constant' if is_fixed else 'variational_posterior',
                    posterior_draws=0 if is_fixed else 10000,
                    source_run=str((joint_run if method == 'joint' else two_stage_run if part == 'reactive' else abort_run).relative_to(ROOT)),
                    display_scale=scale, display_mean=values.mean() * scale,
                    display_q025=lower * scale, display_q975=upper * scale,
                    hard_lower=prior[0], flat_lower=prior[1], flat_upper=prior[2], hard_upper=prior[3]))
estimates = pd.DataFrame(records)
assert len(estimates) == (40 if MIXTURE else 38) and estimates.estimate_type.eq('variational_posterior').sum() == (38 if MIXTURE else 37)
estimates.to_csv(OUT / 'posterior_parameter_comparison.csv', index=False, float_format='%.17g')
differences = []
for parameter, rows in estimates.groupby('parameter', sort=False):
    old, new = [rows.loc[rows.method.eq(method)].iloc[0] for method in ['two_stage', 'joint']]
    differences.append(dict(parameter=parameter, part=old.part, display_scale=old.display_scale,
        two_stage_mean=old['mean'], joint_mean=new['mean'], mean_change=new['mean'] - old['mean'],
        display_mean_change=new.display_mean - old.display_mean,
        identical_prior=all(old[key] == new[key] for key in ['hard_lower', 'flat_lower', 'flat_upper', 'hard_upper'])
                        if old.estimate_type != 'fixed_constant' else False))
pd.DataFrame(differences).to_csv(OUT / 'posterior_mean_changes.csv', index=False, float_format='%.17g')

# %% Fifteen equal square panels; no grids and no model refitting
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': False, 'savefig.facecolor': 'white'})
titles = {'rate_lambda': r'$\lambda$', 'T_0': r'$T_0$', 'theta_E': r'$\theta_E$',
          'rate_norm_l': r'$\ell$', 'alpha': r'$\alpha$', 'w': r'$w$', 'del_go': 'del_go',
          'delay_mu': r'Gaussian $\mu$ by ABL', 'delay_sigma': r'Gaussian $\sigma$ by ABL',
          'V_A': r'$V_A$', 'theta_A': r'$\theta_A$', 't_nd': r'$t_{\mathrm{nd}}$',
          'lapse_prob': r'$P_{\mathrm{lapse}}$', 'beta_lapse': r'$\beta_{\mathrm{lapse}}$',
          'lapse_prob_right': r'$P_{\mathrm{lapse,right}}$', 'lapse_exp_weight': r'$d_{\mathrm{exp}}$'}
geometry, artists = [], []
for part, blocks, stem, size, lower_y, title in [
    ('reactive', GLOBALS + ['delay_mu', 'delay_sigma'], REACTIVE_STEM, (12., 13.), 9.,
     'ASD0047 | SHANK3 cohort 1 WT | Reactive parameters'),
    ('proactive_lapse', PROACTIVE, PROACTIVE_STEM, ((12., 13.) if MIXTURE else (12., 9.3)), (9. if MIXTURE else 5.3),
     'ASD0047 | SHANK3 cohort 1 WT | Proactive + lapse parameters')]:
    fig = plt.figure(figsize=size)
    for panel, block in enumerate(blocks):
        row, column = divmod(panel, 3)
        ax = fig.add_axes([(1. + 3.8 * column) / size[0], (lower_y - 3.5 * row) / size[1],
                           SIDE / size[0], SIDE / size[1]])
        endpoints = []
        for position, method in enumerate(['two_stage', 'joint']):
            values = estimates.loc[estimates.method.eq(method) & estimates.block.eq(block)].sort_values('ABL')
            x = values.ABL.to_numpy() + ABL_OFFSETS[method] if block.startswith('delay_') else np.array([position])
            means = values.display_mean.to_numpy()
            lower, upper = values.display_q025.to_numpy(), values.display_q975.to_numpy()
            is_fixed = values.estimate_type.eq('fixed_constant').all()
            if is_fixed:
                ax.plot(x, means, marker='D', markersize=6, markerfacecolor='white',
                        markeredgecolor=COLORS[method], linestyle='none')
                ax.annotate(f'fixed {means[0]:g}', (x[0], means[0]), xytext=(0, 13), textcoords='offset points',
                            ha='center', fontsize=9, color=COLORS[method])
            else:
                bars = ax.errorbar(x, means, yerr=np.vstack([means - lower, upper - means]),
                    fmt='o' if method == 'two_stage' else 's', color=COLORS[method],
                    markersize=5.5, capsize=4, elinewidth=1.2)
                np.testing.assert_allclose(np.asarray(bars.lines[2][0].get_segments())[:, :, 1],
                                           np.column_stack([lower, upper]), atol=1e-12)
            endpoints.extend(means if is_fixed else np.r_[lower, upper])
            artists.append(dict(part=part, block=block, method=method, parameters=values.parameter.tolist(),
                x=x.tolist(), mean=means.tolist(), q025=None if is_fixed else lower.tolist(),
                q975=None if is_fixed else upper.tolist(), fixed=bool(is_fixed), color=COLORS[method]))
        padding = max(np.ptp(endpoints) * .16, .001)
        limits = [float(min(endpoints) - padding), float(max(endpoints) + padding)]
        unit = 'Time (ms)' if block in ['T_0', 'del_go', 't_nd', 'delay_mu', 'delay_sigma'] else (
               r'Rate (s$^{-1}$)' if block == 'beta_lapse' else 'Probability' if block.startswith('lapse_') else 'Value')
        ax.set(title=titles[block], ylabel=unit, ylim=limits)
        if block.startswith('delay_'):
            ax.set(xticks=ABLS, xticklabels=ABLS, xlim=(12, 68), xlabel='ABL (dB)')
        else:
            ax.set(xticks=[0, 1], xticklabels=['Two-stage', 'Joint'], xlim=(-.55, 1.55))
        physical_size = ax.get_position().size * np.array(size)
        np.testing.assert_allclose(physical_size, [SIDE, SIDE], rtol=0, atol=1e-12)
        geometry.append(dict(part=part, block=block, width_inches=float(physical_size[0]),
            height_inches=float(physical_size[1]), ylim=limits, grid=False,
            x_ticks=ax.get_xticks().tolist(), x_limits=list(ax.get_xlim())))
    fig.suptitle(title, fontsize=17, y=.982)
    fig.legend([Line2D([0], [0], color=COLORS[m], marker='o' if m == 'two_stage' else 's', ls='none')
                for m in ['two_stage', 'joint']], [LABELS[m] for m in ['two_stage', 'joint']],
               ncol=2, frameon=False, loc='upper center', bbox_to_anchor=(.5, .955))
    fig.text(.5, .908 if size[1] < 10 else .922,
             'Same animal: ASD0047 | Posterior means and central 95% credible intervals',
             ha='center', fontsize=10)
    footer = ('Same 18,771 valid choices; success ±1, measured RT [0,1] s. μ/σ are underlying Gaussian parameters.\n'
              'Two-stage reactive intervals condition on frozen proactive/lapse means; joint intervals estimate all together.\n'
              'Actual priors are audited; contaminant choice bias/support and likelihood conditioning differ.\n'
              'Delay-panel ABL positions use additive −1/+1 dB offsets for readability.') if part == 'reactive' else (
              'Two-stage bars reuse the original abort-fit posterior, before its means were frozen.\n'
              'Proactive delay is allowed below zero in both fits; reactive trial delays remain truncated below zero.\n'
              'Open diamonds denote fixed values without posterior intervals; all other bars are central 95% credible intervals.\n'
              'Actual priors are audited; contaminant support and conditioning differ. These are not animal SEMs.')
    fig.text(.5, .018, footer, ha='center', va='bottom', fontsize=9, linespacing=1.45)
    fig.savefig(OUT / f'{stem}.png', dpi=DPI)
    plt.close(fig)

# %% Comparison provenance and explicit non-equivalences
notes = [
    'Both fits use only WT animal ASD0047, with no cross-animal pooling or hierarchical model.',
    'The reactive data are the same 18,771 valid choices, with measured RT in inclusive [0,1] s.',
    'The two-stage proactive fit uses 21,947 event-or-onset-censored trials, including 3,004 abort events.',
    'The joint fit uses 21,786 retained observations: 18,771 valid choices, 3,004 aborts and 11 late fixation-labelled onset-censored rows; 161 long valid responses are excluded.',
    'The initial common priors match the final two-stage animal priors, including proactive delay [-3,0.3] s. Adaptive changes, if needed, remain explicit in the exported endpoints.',
    'All actual hard/flat endpoints are included in the parameter table; compare them before attributing changes only to estimation order.',
    'Two-stage exponential lapses are untruncated with fixed right-choice probability 0.5; joint contaminants use fixation-time [0,3.3] s with a fitted right-choice probability and the selected exponential/uniform weight.',
    'Two-stage reactive inference conditions on valid-window response [onset,onset+1]; joint inference conditions on survival to fixation and the retained [0,onset+1] window.',
    'The two-stage reactive intervals omit uncertainty in the frozen proactive/lapse means. Joint intervals are marginal variational intervals with these parameters estimated together.',
    'Differences cannot be attributed exclusively to fitting jointly versus fitting in two stages.',
    'Underlying Gaussian mu/sigma are plotted, not derived moments after zero truncation.',
]
manifest = dict(status='generated_pending_independent_verification', joint_run=str(joint_run.relative_to(ROOT)),
    two_stage_reactive_run=str(two_stage_run.relative_to(ROOT)), two_stage_abort_run=str(abort_run.relative_to(ROOT)),
    figures=[REACTIVE_STEM + '.png', PROACTIVE_STEM + '.png'], colors=COLORS,
    posterior_estimates=(38 if MIXTURE else 37), fixed_constants=(2 if MIXTURE else 1), total_displayed_estimates=len(estimates),
    reactive_panels=9, proactive_lapse_panels=len(PROACTIVE), draws_per_posterior=10000,
    abl_offsets_db=ABL_OFFSETS, geometry=geometry, artists=artists, comparison_notes=notes,
    source_hashes=source_hashes,
    figure_hashes={stem + '.png': sha256(OUT / f'{stem}.png') for stem in [REACTIVE_STEM, PROACTIVE_STEM]})
assert all(sha256(ROOT / name) == expected for name, expected in source_hashes.items())
(OUT / 'plot_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(f'Saved two PNGs and {len(estimates)} displayed estimates to {OUT}', flush=True)
