# %%
"""RT-only human TIED: 250-ms versus 200-ms delay caps on the same 0.25--1.5 s trials."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/human_tied_mpl')
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import jax
import jax.numpy as jnp
from human_reactive_tied_rt_only_utils import (
    ROOT, PARAM_NAMES, load_trials, group_data, bound_cdf, bound_density,
    retained_mass, log_likelihood, sha256,
)

# %% Read the fitted window; all comparisons use these same rows and denominator.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path, default=ROOT/'ASD_Human_Data/reactive_tied_rt_only_svi_0p25_1p5s_delay250ms')
args = parser.parse_args()
out = args.output_dir
config = json.loads((out/'run_config.json').read_text())
previous = Path(config['source_fit'])
lower, upper = config['retention_window_s']
frame, provenance = load_trials(lower, upper)
times = np.linspace(lower, upper, round((upper-lower)*1000)+1)
# Keep 20-ms bins; the final bin is 10 ms for the 0.25--1.5 s window.
edges = np.r_[np.arange(lower, upper-1e-9, .02), upper]
colors = ['#1f77b4', '#ff7f0e']
asls = [20, 40]
deltas = [.1, .3, .65, 1.3, 2.]
group_colors = {'Control': '#1f77b4', 'ASD': '#d62728'}
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
hist_rows, condition_rows, quality_rows, summary_rows, empirical_rows = [], [], [], [], []
mixture_rows, support_rows = [], []
payload = {}
statuses = {}

# %% Five physical posterior means; RT densities and CDFs sum both response bounds.
for group in ['Control', 'ASD']:
    folder = out/group.lower()
    sub, conditions, data, _ = group_data(frame, group)
    data.pop('choice')
    data.update(rt_lower=lower, rt_upper=upper)
    report = json.loads((folder/'convergence_summary.json').read_text())
    statuses[group] = f"{report['completed_steps']:,} steps; "+('25-window check passed' if not report['flagged'] else 'unconfirmed')
    payload[group] = {}
    for label, source in [('previous_rt_only_200ms', previous), ('rt_only', out)]:
        samples = np.load(source/group.lower()/'posterior_samples.npz')
        matrix = np.column_stack([samples[n] for n in PARAM_NAMES])
        means = matrix.mean(0)
        p = jnp.asarray(means)
        B = jnp.asarray(conditions.Mean_dB.to_numpy())[:, None]
        D = jnp.asarray(conditions.signed_delta.to_numpy())[:, None]
        norm = np.asarray(retained_mass(B, D, p, lower, upper))
        evaluate_pdf = jax.jit(lambda t: sum(bound_density(t, side, B, D, p) for side in [-1, 1]))
        evaluate_cdf = jax.jit(lambda t: sum(bound_cdf(t, side, B, D, p) for side in [-1, 1]))
        origin = np.asarray(evaluate_cdf(lower))
        density = np.asarray(evaluate_pdf(jnp.asarray(times)[None, :]))/norm
        cdf = (np.asarray(evaluate_cdf(jnp.asarray(times)[None, :]))-origin)/norm
        np.savez_compressed(folder/('mean_parameter_curves.npz' if label=='rt_only' else 'previous_rt_only_200ms_mean_parameter_curves.npz'),
            time_s=times, density=density, cdf=cdf, mean_parameters=means,
            condition_B=np.asarray(B).ravel(), condition_D=np.asarray(D).ravel(), retained_mass=norm.ravel())
        rows = []
        for row in conditions.itertuples():
            selected = sub.loc[sub.condition_index.eq(row.condition_index)]
            ordered = np.sort(selected.RT_s.to_numpy()); n = len(ordered)
            at_data = (sum(np.asarray(bound_cdf(jnp.asarray(ordered), side, row.Mean_dB,
                         row.signed_delta, p)) for side in [-1, 1])-origin[row.condition_index, 0])/norm[row.condition_index, 0]
            gap = max(np.max(np.arange(1, n+1)/n-at_data), np.max(at_data-np.arange(n)/n))
            rows.append(dict(Group=group, fit=label, ASL_cd=row.ASL_cd, Delta_dB=row.signed_delta, n=n,
                cdf_max_abs_error=float(gap), mean_rt_data_s=float(ordered.mean()),
                mean_rt_model_s=float(np.trapz(times*density[row.condition_index], times))))
        condition_rows.extend(rows)
        cells = pd.DataFrame(rows)
        quality_rows.append(dict(Group=group, fit=label, n_trials=len(sub), n_participants=sub.Participant.nunique(),
            rt_lower_s=lower, rt_upper_s=upper, mean_parameter_RT_log_likelihood=float(log_likelihood(p, data)),
            mean_condition_cdf_error=float(cells.cdf_max_abs_error.mean()), max_condition_cdf_error=float(cells.cdf_max_abs_error.max())))
        mixtures = {}
        for asl in asls:
            for delta in [None]+deltas:
                mask = conditions.ASL_cd.eq(asl)
                trial_mask = sub.ASL_cd.eq(asl)
                if delta is not None:
                    mask &= conditions.signed_delta.abs().eq(delta)
                    trial_mask &= sub.signed_delta.abs().eq(delta)
                selected = sub.loc[trial_mask]
                indices = conditions.loc[mask, 'condition_index'].to_numpy()
                weights = conditions.loc[mask, 'n'].to_numpy()/len(selected)
                pdf = np.average(density[indices], axis=0, weights=weights)
                distribution = np.average(cdf[indices], axis=0, weights=weights)
                counts = np.histogram(selected.RT_s, edges)[0]
                hist = counts/len(selected)/np.diff(edges)
                mixtures[(asl, delta)] = (hist, pdf, distribution)
                abs_label = 'all' if delta is None else delta
                for index, weight in zip(indices, weights):
                    mixture_rows.append(dict(Group=group, fit=label, ASL_cd=asl, abs_Delta_dB=abs_label,
                        condition_index=index, signed_Delta_dB=conditions.loc[index, 'signed_delta'], n=int(conditions.loc[index, 'n']), weight=weight))
                if label == 'rt_only':
                    hist_rows.extend(dict(Group=group, ASL_cd=asl, abs_Delta_dB=abs_label, bin_left_s=edges[i],
                        bin_right_s=edges[i+1], count=int(count), n=len(selected), density=hist[i]) for i, count in enumerate(counts))
                    if delta is None:
                        ordered = np.sort(selected.RT_s.to_numpy())
                        empirical_rows.extend(dict(Group=group, ASL_cd=asl, RT_s=t, empirical_cdf=(i+1)/len(ordered))
                                              for i, t in enumerate(ordered))
        payload[group][label] = mixtures
        if label == 'previous_rt_only_200ms':
            continue
        summary = pd.read_csv(folder/'posterior_summary.csv')
        summary.insert(0, 'Group', group)
        summary_rows.extend(summary.to_dict('records'))
        support_rows.append(dict(Group=group, delay_prior_lower_s=0., delay_prior_upper_s=config['delay_prior_upper_s'],
            delay_mean_s=means[4], delay_sd_s=matrix[:, 4].std(ddof=1),
            delay_q025_s=np.quantile(matrix[:, 4], .025), delay_q975_s=np.quantile(matrix[:, 4], .975),
            fraction_within_1ms_of_upper=float(np.mean(matrix[:, 4]>config['delay_prior_upper_s']-.001)),
            fraction_within_5ms_of_upper=float(np.mean(matrix[:, 4]>config['delay_prior_upper_s']-.005))))

        plotted = matrix.copy(); plotted[:, :3] = np.log10(plotted[:, :3]); plotted[:, 4] *= 1000
        labels = [r'log$_{10}$ λ', r'log$_{10}$ T₀ (s)', r'log$_{10}$ θE', 'w', 'tE aff (ms)']
        limits = np.quantile(plotted, [.001, .999], axis=0)
        limits[1,4] = config['delay_prior_upper_s']*1000+.2
        fig, axes = plt.subplots(5, 5, figsize=(13, 12))
        for i in range(5):
            for j in range(5):
                ax = axes[i, j]
                if i < j: ax.axis('off')
                elif i == j:
                    ax.hist(plotted[:, j], bins=50, color=group_colors[group], alpha=.8); ax.set_xlim(limits[:, j])
                    if j == 4: ax.axvline(config['delay_prior_upper_s']*1000, color='.3', ls=':', lw=1)
                else:
                    ax.hexbin(plotted[:, j], plotted[:, i], gridsize=35, mincnt=1, cmap='Blues' if group=='Control' else 'Reds')
                    ax.set_xlim(limits[:, j]); ax.set_ylim(limits[:, i])
                if i == 4: ax.set_xlabel(labels[j])
                elif i >= j: ax.tick_params(labelbottom=False)
                if j == 0 and i > 0: ax.set_ylabel(labels[i])
        fit_status = 'UNCONFIRMED' if report['flagged'] else 'checks passed'
        fig.suptitle(f'{group} | RT-only | 10,000 variational draws | {fit_status}\n'
                     'Pooled trials; uncertainty does not measure participant variability', fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, .95))
        for ext in ['png', 'pdf']: fig.savefig(folder/f'parameter_corner.{ext}', dpi=170)
        plt.close(fig)
        conv = pd.read_csv(folder/'convergence.csv')
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes.flat[0].plot(conv.step, conv.mean_negative_elbo, color=group_colors[group])
        axes.flat[0].set_ylabel('1,000-step mean negative ELBO')
        for ax, name in zip(list(axes.flat)[1:], PARAM_NAMES):
            ax.plot(conv.step, conv[name+'_mean'], color=group_colors[group])
            ax.fill_between(conv.step, conv[name+'_mean']-conv[name+'_sd'], conv[name+'_mean']+conv[name+'_sd'], alpha=.2, color=group_colors[group])
            ax.set_ylabel(name+' (mean ±1 SD)')
            if name == 't_E_aff': ax.axhline(config['delay_prior_upper_s'], color='.4', ls='--', lw=1)
        for ax in axes.flat:
            ax.axvline(report['restored_best_step'], color='k', ls=':')
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{value/1000:g}'))
            ax.set_xlabel('SVI steps (thousands)'); ax.grid(alpha=.15)
        fig.suptitle(f"{group} | RT-only | selected best {report['restored_best_step']:,}; final {report['completed_steps']:,}\n"
                     f"{report['confirmation_windows']} follow-up windows; local stability "+('passed' if not report['flagged'] else 'unconfirmed'))
        fig.tight_layout(rect=(0, 0, 1, .93))
        for ext in ['png', 'pdf']: fig.savefig(folder/f'convergence.{ext}', dpi=160)
        plt.close(fig)

# %% Saved calculations do not score or fit choices.
pd.DataFrame(condition_rows).to_csv(out/'condition_cdf_discrepancies.csv', index=False, float_format='%.17g')
pd.DataFrame(quality_rows).to_csv(out/'common_RT_fit_quality.csv', index=False, float_format='%.17g')
pd.DataFrame(hist_rows).to_csv(out/'rtd_histograms.csv', index=False, float_format='%.17g')
pd.DataFrame(mixture_rows).to_csv(out/'mixture_weights.csv', index=False, float_format='%.17g')
pd.DataFrame(empirical_rows).to_csv(out/'empirical_cdfs.csv', index=False, float_format='%.17g')
pd.DataFrame(summary_rows).to_csv(out/'parameter_summary.csv', index=False, float_format='%.17g')
pd.DataFrame(support_rows).to_csv(out/'delay_boundary_diagnostics.csv', index=False, float_format='%.17g')
legend = [Line2D([0], [0], color=c, label=f'ASL {asl} cd/m²') for c, asl in zip(colors, asls)]
legend += [Line2D([0], [0], color='.3', lw=2, label='RT-only model (250 ms cap)'),
           Line2D([0], [0], color='.3', lw=2, ls=':', label='Previous RT-only model (200 ms cap)'),
           Line2D([0], [0], color='.3', lw=1, alpha=.6, label='Data steps')]
footer = (f'Condition = RT | {lower:g} ≤ RT_s ≤ {upper:g} s | delay prior 0–{config["delay_prior_upper_s"]*1000:g} ms\n'
          'RT-only: sum left/right densities; response side ignored | all curves conditional on the fitted RT window\n'
          'Arithmetic physical posterior means | pooled raw trials; actual stimulus frequencies | variational approximation\n'
          f'Control: {statuses["Control"]} | ASD: {statuses["ASD"]}')
for limit, name in [(upper, 'full_0p25_1p5s'), (1., 'zoom_0p25_1s')]:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ri, group in enumerate(['Control', 'ASD']):
        selected = frame.loc[frame.Group.eq(group)]
        for color, asl in zip(colors, asls):
            hist, pdf, cdf = payload[group]['rt_only'][(asl, None)]
            _, old_pdf, old_cdf = payload[group]['previous_rt_only_200ms'][(asl, None)]
            axes[ri, 0].stairs(hist, edges, color=color, alpha=.55, lw=1)
            axes[ri, 0].plot(times, pdf, color=color, lw=2)
            axes[ri, 0].plot(times, old_pdf, color=color, lw=1.8, ls=':')
            ordered = np.sort(selected.loc[selected.ASL_cd.eq(asl), 'RT_s'].to_numpy())
            axes[ri, 1].step(np.r_[lower, ordered, upper], np.r_[0., np.arange(1, len(ordered)+1)/len(ordered), 1.],
                             where='post', color=color, alpha=.5, lw=1)
            axes[ri, 1].plot(times, cdf, color=color, lw=2)
            axes[ri, 1].plot(times, old_cdf, color=color, lw=1.8, ls=':')
        axes[ri, 0].set(title=f'{group} | {len(selected):,} trials | pooled RTD\n{statuses[group]}',
                         xlabel='Recorded RT (s)', ylabel='Density (s⁻¹)', xlim=(lower, limit), ylim=(0, None))
        axes[ri, 1].set(title=f'{group} | pooled RT CDF', xlabel='Recorded RT (s)', ylabel='Conditional CDF',
                         xlim=(lower, limit), ylim=(-.02, 1.02))
        for ax in axes[ri]: ax.grid(alpha=.15)
    ymax = max(ax.get_ylim()[1] for ax in axes[:, 0])
    for ax in axes[:, 0]: ax.set_ylim(0, ymax)
    fig.suptitle('Human reactive-only TIED | 250 ms delay bound | fit RT distributions; ignore choices', fontsize=17, y=.985)
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, .955), ncol=3, frameon=False)
    fig.text(.5, .018, footer, ha='center', va='bottom', fontsize=10)
    fig.tight_layout(rect=(0, .11, 1, .90), h_pad=3)
    for ext in ['png', 'pdf']: fig.savefig(out/f'human_RT_only_rtd_cdf_comparison_{name}.{ext}', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(21, 9), sharex=True, sharey=True)
    for ri, group in enumerate(['Control', 'ASD']):
        for ci, delta in enumerate(deltas):
            ax = axes[ri, ci]
            for color, asl in zip(colors, asls):
                hist, pdf, _ = payload[group]['rt_only'][(asl, delta)]
                ax.stairs(hist, edges, color=color, lw=1.2, alpha=.7, ls='--')
                ax.plot(times, pdf, color=color, lw=2)
            ax.set(title=f'{group} | |ΔdB| = {delta:g}', xlabel='Recorded RT (s)', xlim=(lower, limit))
            if ci == 0: ax.set_ylabel('Density (s⁻¹)')
            ax.grid(alpha=.15)
    ymax = max(float(np.max(a)) for group in payload for (asl, d), arrays in payload[group]['rt_only'].items()
               if d is not None for a in arrays[:2])
    axes[0, 0].set_ylim(0, 1.08*ymax)
    fig.suptitle('Human RT-only TIED | 250 ms delay bound | RTDs by absolute difference', fontsize=18, y=.985)
    handles = legend[:2]+[Line2D([0], [0], color='.3', lw=2, label='RT-only model (250 ms cap)'),
                         Line2D([0], [0], color='.3', ls='--', label='Data RTD')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .945), ncol=4, frameon=False)
    fig.text(.5, .02, footer, ha='center', va='bottom', fontsize=11)
    fig.tight_layout(rect=(0, .13, 1, .91), h_pad=3)
    for ext in ['png', 'pdf']: fig.savefig(out/f'human_RT_only_rtd_by_delta_{name}.{ext}', dpi=180)
    plt.close(fig)

(out/'plot_manifest.json').write_text(json.dumps(dict(source=provenance,
    source_sha256=sha256(__file__), normalization_window_s=[lower, upper], data_bins_s=.02, model_grid_s=.001,
    likelihood='RT only; no choice score', comparison='Previous 200-ms RT-only parameters evaluated with the same marginal RT likelihood and selected 0.25--1.5 s rows',
    curves='Arithmetic physical means of 10000 constrained variational draws',
    simultaneous_changes='Relative to the 200-ms RT-only fit: four additional Control RTs below 250 ms removed; delay prior raised to 250 ms',
    histogram_edges_s=edges.tolist(), histogram_bin_width_note='20 ms, with a final 10-ms bin from 1.49 to 1.5 s',
    figure_hashes={str(p.relative_to(out)):sha256(p) for p in out.rglob('*.png')}), indent=2))
print(pd.DataFrame(quality_rows).to_string(index=False))
