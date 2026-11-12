# %% ASD0047 joint posterior corners and diagnostic PDF assembly
import hashlib
import json
import shutil
from pathlib import Path
import corner
import fitz
import numpy as np
import pandas as pd
from scipy.stats import invgauss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# %% Editable output, units, appearance, and all-draw corner settings
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'asd_analysis/shank3_cohort1_ASD0047_joint_npl_alpha_gaussian_abl_contaminants'
selected = json.loads((BASE / 'selected_fit.json').read_text())
RUN = ROOT / selected['selected_run']
OUT = BASE / 'posterior_and_diagnostics'
REPORT = RUN.parents[1] / 'report'
DELIVERY = ROOT / 'shan3_coh_1_animal_Wise_fitting/ASD0047_joint'
BINS, SMOOTH, LEVELS = 45, 1., [.68, .95]
COLOR, DPI = '#4D4D4D', 150
OUT.mkdir(exist_ok=True)
DELIVERY.mkdir(parents=True, exist_ok=True)
config = json.loads((RUN / 'configuration.json').read_text())
prep = json.loads((RUN.parents[1] / 'preparation.json').read_text())
names = config['parameter_order']
start_rows = []
for seed in [0, 1]:
    start_run = RUN.parent / f'seed_{seed}'
    start_summary = json.loads((start_run / 'run_summary.json').read_text())
    start_parameters = pd.read_csv(start_run / 'posterior_summary.csv')
    for row in start_parameters.itertuples(index=False):
        start_rows.append(dict(seed=seed, parameter=row.parameter, mean=row.mean,
            q025=row.q025, q975=row.q975, selected=seed == selected['selected_seed'],
            best_window_negative_elbo=start_summary['best_window_negative_elbo'],
            posterior_mean_joint_log_likelihood=start_summary['posterior_mean_joint_log_likelihood']))
pd.DataFrame(start_rows).to_csv(OUT / 'all_start_parameter_estimates.csv', index=False, float_format='%.17g')
delay_by_start = [row for row in start_rows if row['parameter'] == 't_nd']
start_note = '; '.join(f'start {row["seed"]}: {row["mean"]*1000:.1f} [{row["q025"]*1000:.1f}, {row["q975"]*1000:.1f}] ms'
                       for row in delay_by_start)
with np.load(RUN / 'posterior_samples.npz') as samples:
    p = {key: samples[key].mean(0) for key in samples.files}
    latent_entry_mass = (1-samples['lapse_prob']) * invgauss.cdf(-samples['t_nd'],
        mu=1/(samples['V_A']*samples['theta_A']), scale=samples['theta_A']**2)
    negative_delay_probability = float(np.mean(samples['t_nd'] < 0))
    physical = np.column_stack([samples[key].reshape(10000, -1) for key in config['blocks']])
p.update(config['fixed_parameters'])
assert physical.shape == (10000, len(names))
pd.DataFrame(dict(draw=np.arange(10000), latent_probability_before_fixation=latent_entry_mass,
    log_survival_correction=-np.log1p(-latent_entry_mass))).to_csv(
    OUT / 'latent_entry_mass_posterior.csv.gz', index=False, float_format='%.17g')
(OUT / 'proactive_delay_and_entry_summary.json').write_text(json.dumps(dict(
    posterior_probability_t_nd_negative=negative_delay_probability,
    latent_entry_mass_draw_mean=float(latent_entry_mass.mean()),
    latent_entry_mass_q025=float(np.quantile(latent_entry_mass, .025)),
    latent_entry_mass_q975=float(np.quantile(latent_entry_mass, .975)),
    observed_model_conditioned_on_no_response_before_fixation=True,
    note='These are latent pre-entry probabilities; the fitted observation model conditions them away. Negative proactive t_nd is an effective timing shift, not a negative reactive sensory delay.'), indent=2) + '\n')
labels = {'V_A': r'$V_A$', 'theta_A': r'$\theta_A$', 't_nd': r'$t_{\rm nd}$ (ms)',
          'rate_lambda': r'$\lambda$', 'T_0': r'$T_0$ (ms)', 'theta_E': r'$\theta_E$',
          'rate_norm_l': r'$\ell$', 'alpha': r'$\alpha$', 'w': r'$w$', 'del_go': 'del_go (ms)',
          'lapse_prob': r'$P_{\rm lapse}$', 'lapse_prob_right': r'$P_{\rm lapse,right}$',
          'lapse_exp_weight': r'$d_{\rm exp}$', 'beta_lapse': r'$\beta_{\rm lapse}$ (s$^{-1}$)'}
for key, symbol in [('delay_mu', r'\mu'), ('delay_sigma', r'\sigma')]:
    for abl in [20, 40, 60]:
        labels[f'{key}_ABL{abl}'] = rf'${symbol}_{{{abl}}}$ (ms)'
scales = np.array([1000. if name in ['t_nd', 'T_0', 'del_go'] or name.startswith('delay_') else 1. for name in names])
display = physical * scales
means = display.mean(0)
intervals = np.percentile(display, [2.5, 97.5], axis=0)
records, ranges = [], []
for index, name in enumerate(names):
    key = name.split('_ABL')[0]
    lo, pl, ph, hi = config['priors'][key]
    values = display[:, index]
    pad = .06 * np.ptp(values)
    bounds = [max(lo * scales[index], values.min() - pad), min(hi * scales[index], values.max() + pad)]
    assert bounds[0] <= values.min() and bounds[1] >= values.max()
    ranges.append(bounds)
    records.append(dict(parameter=name, title=labels[name] + f' = {means[index]:.3f}',
        mean=means[index], q025=intervals[0, index], q975=intervals[1, index],
        display_scale=scales[index], range_lower=bounds[0], range_upper=bounds[1],
        hard_lower=lo * scales[index], flat_lower=pl * scales[index],
        flat_upper=ph * scales[index], hard_upper=hi * scales[index]))
pd.DataFrame(records).to_csv(OUT / 'corner_parameter_summaries.csv', index=False, float_format='%.17g')
pd.DataFrame(np.corrcoef(physical, rowvar=False), index=names, columns=names).to_csv(OUT / 'posterior_correlations.csv', float_format='%.17g')
plt.rcParams.update({'axes.grid': False, 'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
pages = []
pdf_path = OUT / 'SHANK3_cohort1_ASD0047_joint_posterior_corners.pdf'
with PdfPages(pdf_path) as pdf:
    for focus in ['all_parameters', 'proactive_and_contaminants']:
        indices = list(range(len(names))) if focus == 'all_parameters' else [i for i, name in enumerate(names)
            if name in ['V_A', 'theta_A', 't_nd', 'lapse_prob', 'lapse_prob_right', 'lapse_exp_weight', 'beta_lapse']]
        n = len(indices)
        size = (max(14, 1.8 * n), max(14, 1.8 * n))
        fig = plt.figure(figsize=size)
        corner.corner(display[:, indices], fig=fig, labels=[labels[names[i]] for i in indices],
            color=COLOR, range=[ranges[i] for i in indices], bins=BINS, smooth=SMOOTH,
            quantiles=[.025, .975], show_titles=False, levels=LEVELS, plot_datapoints=False,
            plot_density=True, plot_contours=True, fill_contours=False, max_n_ticks=3,
            hist_kwargs={'density': True, 'linewidth': 1.1}, contour_kwargs={'linewidths': .8},
            pcolor_kwargs={'rasterized': True}, label_kwargs={'fontsize': 9 if n > 10 else 11})
        fig.subplots_adjust(left=.07, right=.985, bottom=.08 if n > 10 else .14,
                            top=.92 if n > 10 else .86, hspace=.10, wspace=.10)
        axes = np.array(fig.axes).reshape(n, n)
        for j, index in enumerate(indices):
            axes[j, j].axvline(means[index], color=COLOR, lw=1.2)
            axes[j, j].set_title(records[index]['title'], fontsize=8 if n > 10 else 10, pad=5)
        for row in range(n):
            for column in range(row + 1):
                axes[row, column].tick_params(labelsize=7 if n > 10 else 9)
                axes[row, column].grid(False)
        fixed_note = 'd=1 is fixed and has no posterior axis.' if 'lapse_exp_weight' not in names else 'd is the exponential fraction within the contaminant branch.'
        fig.suptitle('ASD0047 | SHANK3 cohort 1 WT | Single-stage posterior\n' +
                     ('All fitted parameters' if focus == 'all_parameters' else 'Proactive + contaminant parameters'),
                     y=.975, fontsize=18)
        fig.text(.5, .945 if n > 10 else .915,
                 f'{len(names)} jointly fitted parameters; 10,000 full-rank variational draws; selected seed {config["seed"]}', ha='center', fontsize=11)
        fig.text(.5, .017, 'Diagonal titles: parameter name and posterior mean; solid mean, dashed central 95% limits.\n'
                 'Lower triangle: 68% and 95% joint contours. Every draw retained; padded axes do not depict full prior support.\n'
                 'Times in ms; only proactive t_nd can be negative. ' + fixed_note, ha='center', fontsize=10, linespacing=1.5)
        png = 'SHANK3_cohort1_ASD0047_joint_corner_' + focus + '.png'
        fig.savefig(OUT / png, dpi=120 if n > 10 else DPI)
        pdf.savefig(fig)
        pages.append(dict(page=len(pages) + 1, focus=focus, parameters=[names[i] for i in indices], png=png,
                          axes=len(fig.axes), diagonal_titles=[records[i]['title'] for i in indices]))
        plt.close(fig)

# %% Fixation-time abort data/model, convergence of both starts, and parameter table
out_variant = RUN.parents[1]
parent = pd.read_csv(out_variant / 'parent_trials.csv.gz', float_precision='round_trip')
aborts = parent.loc[parent.observation.eq('pre_onset_abort')]
edges = np.arange(0, 3.3001, .02)
hist, _ = np.histogram(aborts.fix_time / 1000, edges)
times = np.arange(0, 3.3001, .001)
onsets = parent.onset_s.to_numpy()
normalizers = pd.read_csv(REPORT / 'onset_window_normalizers.csv.gz', float_precision='round_trip').set_index('prepared_csv_line')
ordered = normalizers.loc[parent.prepared_csv_line, 'retained_window_mass'].to_numpy()
v, a, nd, lapse, beta, d = [float(p[key]) for key in ['V_A', 'theta_A', 't_nd', 'lapse_prob', 'beta_lapse', 'lapse_exp_weight']]
abort_density = ((1 - lapse) * invgauss.pdf(times - nd, mu=1/(v*a), scale=a*a) + lapse *
                 (d * beta * np.exp(-beta*times)/(-np.expm1(-beta*3.3)) + (1-d)/3.3))
# Each trial is observed as an abort only before its own planned stimulus onset.
exposure = np.array([np.mean((t < onsets)/ordered) for t in times])
model_abort = abort_density * exposure
clipped = np.minimum(edges[:, None], onsets[None, :])
contaminant_cdf = d * (-np.expm1(-beta * clipped)) / (-np.expm1(-beta * 3.3)) + (1-d) * clipped / 3.3
event_cdf = (1-lapse) * invgauss.cdf(clipped-nd, mu=1/(v*a), scale=a*a) + lapse * contaminant_cdf
entry_cdf = (1-lapse) * invgauss.cdf(-nd, mu=1/(v*a), scale=a*a)
censored_cdf = ((event_cdf-entry_cdf)/ordered[None, :]).mean(1)
expected_bins = np.diff(censored_cdf)
assert expected_bins.min() >= -1e-12
predicted_abort_fraction = float(expected_bins.sum())
pd.DataFrame(dict(left_s=edges[:-1], right_s=edges[1:], observed_abort_count=hist,
    observed_density=hist/len(parent)/.02, model_bin_probability=expected_bins,
    model_expected_count=expected_bins*len(parent))).to_csv(
    OUT / 'censor_adjusted_abort_bins.csv', index=False, float_format='%.17g')
(OUT / 'abort_probability_audit.json').write_text(json.dumps(dict(
    observed_abort_fraction=len(aborts)/len(parent), predicted_abort_fraction=predicted_abort_fraction,
    denominator_retained_trials=len(parent), all_pre_onset_aborts=len(aborts),
    exact_onset_censoring=True, entry_conditioned=True,
    model_bin_mass_calculation='Analytic proactive/contaminant CDF at min(bin edge, exact trial onset), minus entry CDF, divided by retained-window mass.'), indent=2) + '\n')
summary_fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes[0, 0].stairs(hist / len(parent) / .02, edges, color=COLOR, lw=1.4, label='Observed pre-onset aborts')
axes[0, 0].plot(times, model_abort, color='#0072B2', lw=1., label='Complete fitted proactive + contaminants')
axes[0, 0].set(xlabel='Time since fixation onset (s)', ylabel='Density / retained trials (s⁻¹)',
               title=f'Pre-onset aborts: n={len(aborts):,}; no early-abort cutoff\nObserved {len(aborts)/len(parent):.3%}; model {predicted_abort_fraction:.3%}')
axes[0, 0].legend(frameon=False, fontsize=9)
for seed in [0, 1]:
    convergence = pd.read_csv(RUN.parent / f'seed_{seed}/convergence.csv')
    summary = json.loads((RUN.parent / f'seed_{seed}/run_summary.json').read_text())
    axes[0, 1].plot(convergence.step, convergence.mean_negative_elbo, lw=1., label=f'Start {seed}')
    axes[0, 1].axvline(summary['restored_best_step'], color=['#4D4D4D', '#0072B2'][seed], ls='--', lw=.8)
axes[0, 1].set(xlabel='Updates', ylabel='Window mean negative ELBO', title='1,000-update windows; best checkpoints restored')
axes[0, 1].legend(frameon=False)
for ax in axes[1]:
    ax.axis('off')
summary_values = pd.DataFrame(records)
for column, subset in enumerate([summary_values.iloc[:10], summary_values.iloc[10:]]):
    cells = [[labels[row.parameter], f'{row.mean:.3f}', f'[{row.q025:.3f}, {row.q975:.3f}]'] for row in subset.itertuples()]
    table = axes[1, column].table(cellText=cells, colLabels=['Parameter', 'Posterior mean', '95% interval'],
                                 cellLoc='center', bbox=[0, .03, 1, .94])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('white')
        cell.set_facecolor('#F2F2F2' if row == 0 else 'white')
summary_fig.suptitle('ASD0047 | Joint-fit abort and convergence diagnostics', fontsize=17)
summary_fig.text(.5, .015, 'All times in parameter tables are ms. Abort histogram area is the observed abort fraction, not one.\n'
    'Parameters update together; no proactive/lapse estimates are frozen. Intervals are variational credible intervals.\n'
    'Proactive delay across starts: ' + start_note + '\n'
    'Competitive starts can reveal weak identification; the selected corner is not a claim of a unique solution.', ha='center', fontsize=9)
summary_fig.tight_layout(rect=[0, .105, 1, .96])
summary_fig.savefig(OUT / 'SHANK3_cohort1_ASD0047_joint_abort_and_convergence.png', dpi=DPI)
with PdfPages(OUT / 'diagnostic_summary_page.pdf') as pdf:
    pdf.savefig(summary_fig)
plt.close(summary_fig)
pd.DataFrame(dict(time_s=times, model_abort_density=model_abort, censor_exposure=exposure)).to_csv(
    OUT / 'censor_adjusted_abort_prediction.csv', index=False, float_format='%.17g')

# %% Four-page diagnostics report; the full-corner PDF remains separate and zoomable
prefix = json.loads((REPORT / 'report_manifest.json').read_text())
diagnostics = OUT / 'SHANK3_cohort1_ASD0047_joint_diagnostics.pdf'
with fitz.open(prefix['pdf']) as document:
    with fitz.open(OUT / 'diagnostic_summary_page.pdf') as addition:
        document.insert_pdf(addition)
    with fitz.open(pdf_path) as corners:
        document.insert_pdf(corners, from_page=1, to_page=1)
    document.set_toc([[1, 'ASD0047 joint RTDs', 1], [1, 'ASD0047 joint psychometric', 2],
                      [1, 'Abort distributions, convergence, and parameters', 3],
                      [1, 'Proactive and contaminant posterior corner', 4]])
    document.save(diagnostics)
for path in [pdf_path, diagnostics]:
    shutil.copy2(path, DELIVERY / path.name)
manifest = dict(status='generated_pending_verification', selected_run=str(RUN.relative_to(ROOT)),
    fitted_parameters=len(names), samples=10000, fixed_parameters=config['fixed_parameters'],
    corners=pages, corner_pdf=str(pdf_path), corner_pages=2, diagnostic_pdf=str(diagnostics), diagnostic_pages=4,
    display_ranges=ranges, all_draws_included=True, contour_masses=LEVELS,
    means_and_intervals_recomputed_from_draws=True, corner_diagonal_titles_include_names_and_means=True,
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT / 'export_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(str(diagnostics), flush=True)
