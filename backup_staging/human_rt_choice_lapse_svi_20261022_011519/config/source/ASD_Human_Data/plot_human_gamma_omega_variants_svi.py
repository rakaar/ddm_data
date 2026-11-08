# %%
"""Human RT-only and joint-with-lapse Gamma/Omega posterior-mean diagnostics."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/human_gamma_omega_variants_mpl')
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
import jax.numpy as jnp
from human_gamma_omega_variants_svi_utils import (
    ROOT, BASELINE_ROOT, OUTPUTS, MODES, get_priors, parameter_names, parameter_blocks,
    sha256, load_data, prepare_group, sample_matrix, vector_params, log_likelihood,
    prediction_components, ddm_log_cdf,
)

# %% Configuration: fit variants remain separate from the completed joint baseline.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--mode', choices=MODES, required=True)
parser.add_argument('--output-dir', type=Path)
parser.add_argument('--comparison-only', action='store_true',
                    help='Compare both completed variants with the existing joint fit; save under this mode/output')
parser.add_argument('--rt-only-dir', type=Path, default=OUTPUTS['rt_only'])
parser.add_argument('--lapse-dir', type=Path, default=OUTPUTS['rt_choice_lapse'])
parser.add_argument('--baseline-dir', type=Path, default=BASELINE_ROOT)
args = parser.parse_args()
mode = args.mode
out = args.output_dir or OUTPUTS[mode]
groups = ['Control', 'ASD']
levels = [20., 40.]
absolute = [.1, .3, .65, 1.3, 2.]
times = np.linspace(0, 1.8, 1801)
edges = np.linspace(0, 1.8, 91)
level_colors = {20.: '#1f77b4', 40.: '#ff7f0e'}
group_colors = {'Control': '#1976d2', 'ASD': '#d62728'}
labels = {'rt_only': 'RT-only', 'rt_choice_lapse': 'RT + choice + lapses', 'rt_choice': 'RT + choice baseline'}
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.facecolor': 'white'})


# %% Shared calculations: exact conditional predictions and trial-weighted mixtures.
def assess(params, data, sub, conditions, model_mode, label, group):
    p = {name: jnp.asarray(value) for name, value in params.items()}
    endpoint = prediction_components(p, np.array([1.8]), model_mode, K=40)
    ids = np.asarray(data['condition'], dtype=int)
    g, o, tau, w = (p[name] for name in ['gamma', 'omega', 't_E_aff', 'w'])
    log_mass = ddm_log_cdf(1.8, g, o, w, tau, 40)
    cdf = jnp.exp(ddm_log_cdf(data['rt'], g[ids], o[ids], w, tau[ids], 40) - log_mass[ids])
    if model_mode == 'rt_choice_lapse':
        cdf = (1 - p['lapse_prob']) * cdf + p['lapse_prob'] * data['rt'] / 1.8
    cdf = np.asarray(cdf)
    right = np.asarray(endpoint['p_right'])
    mass = np.asarray(endpoint['retained_mass'])
    rows = []
    for c in conditions.itertuples():
        i = c.condition_index
        fitted = cdf[ids == i][np.argsort(sub.loc[ids == i, 'RT_s'].to_numpy())]
        empirical = np.arange(1, c.n + 1) / c.n
        gap = float(max(np.abs(fitted - empirical).max(),
                        np.abs(fitted - (empirical - 1 / c.n)).max()))
        rows.append(dict(Group=group, model=label, mode=model_mode, condition_index=i,
                         ASL_cd=c.ASL_cd, signed_delta=c.signed_delta, n=c.n, right=c.right,
                         empirical_p_right=c.right / c.n, model_p_right=right[i],
                         retained_mass=mass[i], cdf_gap=gap))
    table = pd.DataFrame(rows)
    errors = table.model_p_right - table.empirical_p_right
    marginal_ll = float(log_likelihood(p, data, model_mode, K=40, marginal=True))
    objective_ll = float(log_likelihood(p, data, model_mode, K=40))
    score = dict(Group=group, model=label, mode=model_mode, n_trials=len(sub),
                 objective_log_likelihood=objective_ll, marginal_rt_log_likelihood=marginal_ll,
                 joint_log_likelihood=np.nan if model_mode == 'rt_only' else objective_ll,
                 psychometric_mae=float(np.abs(errors).mean()),
                 psychometric_rmse=float(np.sqrt((errors ** 2).mean())),
                 mean_condition_cdf_gap=float(table.cdf_gap.mean()),
                 max_condition_cdf_gap=float(table.cdf_gap.max()))
    return table, score


def pooled_tables(sub, conditions, density, cdf, group, label=None):
    """Signs are mixed only after each signed condition has been normalized."""
    histograms, mixtures = [], []
    for level in levels:
        for delta in absolute:
            choose = sub.ASL_cd.eq(level) & np.isclose(sub.signed_delta.abs(), delta)
            obs = sub.loc[choose, 'RT_s'].to_numpy()
            signed = conditions.loc[conditions.ASL_cd.eq(level) &
                                    np.isclose(conditions.signed_delta.abs(), delta)].sort_values('signed_delta')
            ids = signed.condition_index.to_numpy(dtype=int)
            weights = signed.n.to_numpy() / len(obs)
            assert len(ids) == 2 and np.isclose(weights.sum(), 1)
            counts, _ = np.histogram(obs, bins=edges)
            h = pd.DataFrame(dict(Group=group, ASL_cd=level, abs_delta=delta,
                                 left_s=edges[:-1], right_s=edges[1:], count=counts,
                                 n=len(obs), density=counts / (len(obs) * .02)))
            m = pd.DataFrame(dict(Group=group, ASL_cd=level, abs_delta=delta, time_s=times,
                                 density=density[:, ids] @ weights, cdf=cdf[:, ids] @ weights,
                                 negative_n=int(signed.n.iloc[0]), positive_n=int(signed.n.iloc[1]),
                                 negative_weight=weights[0], positive_weight=weights[1]))
            if label is not None:
                h['model'] = label
                m['model'] = label
            histograms.append(h)
            mixtures.append(m)
    return pd.concat(histograms, ignore_index=True), pd.concat(mixtures, ignore_index=True)


# %% Per-variant tables, group figures, parameter intervals and convergence diagnostics.
if not args.comparison_only:
    config = json.loads((out / 'run_config.json').read_text())
    assert config['mode'] == mode
    for path, digest in config['source_hashes'].items():
        assert sha256(ROOT / path) == digest, path
    _, selected, _, _ = load_data(mode)
    names = parameter_names(mode)
    blocks = parameter_blocks(mode)
    priors = get_priors(mode)
    n_params = len(names)
    summaries, quality, condition_quality = [], [], []
    histograms, mixtures, condition_curves, boundary, strongest = [], [], [], [], []
    loaded = {}
    for group in groups:
        folder = out / group.lower()
        sub, conditions, data, _, _ = prepare_group(selected, group, mode, Path(config['initial_root']))
        posterior = dict(np.load(folder / 'posterior_samples.npz'))
        matrix = sample_matrix(posterior, mode)
        assert matrix.shape == (10000, n_params)
        params = {name: jnp.asarray(value) for name, value in vector_params(matrix.mean(0), mode).items()}
        summary = pd.read_csv(folder / 'posterior_summary.csv', float_precision='round_trip')
        assert summary.parameter.tolist() == list(names)
        summary.insert(0, 'Group', group)
        summary['block'] = ['gamma'] * 20 + ['omega'] * 20 + ['t_E_aff'] * 20 + list(blocks[3:])
        summary['condition_index'] = list(range(20)) * 3 + [np.nan] * (n_params - 60)
        summary = summary.merge(conditions[['condition_index', 'ASL_cd', 'signed_delta']],
                                on='condition_index', how='left', validate='many_to_one')
        summaries.append(summary)
        fitted, score = assess(params, data, sub, conditions, mode, labels[mode], group)
        score['free_parameters'] = n_params
        quality.append(score)
        condition_quality.append(fitted)
        prediction = prediction_components(params, times, mode, K=40)
        right_density, left_density, density, cdf = [np.asarray(prediction[key]) for key in
                                                     ['right_density', 'left_density', 'density', 'cdf']]
        for c in conditions.itertuples():
            i = c.condition_index
            condition_curves.append(pd.DataFrame(dict(Group=group, condition_index=i,
                ASL_cd=c.ASL_cd, signed_delta=c.signed_delta, time_s=times,
                right_density=right_density[:, i], left_density=left_density[:, i],
                density=density[:, i], cdf=cdf[:, i])))
        h, m = pooled_tables(sub, conditions, density, cdf, group)
        histograms.append(h)
        mixtures.append(m)
        for i, name in enumerate(names):
            block = summary.block.iloc[i]
            prior_key = ('gamma_positive' if conditions.signed_delta.iloc[i] > 0 else 'gamma_negative') if i < 20 else block
            lo, hi = priors[prior_key]['hard']
            vals = matrix[:, i]
            effective = float(conditions.guide_delay_upper_s.iloc[i - 40]) if 40 <= i < 60 else hi
            boundary.append(dict(Group=group, parameter=name, hard_low=lo, hard_high=hi,
                fraction_within_1pct_lower_hard=float(np.mean(vals < lo + .01 * (hi - lo))),
                fraction_within_1pct_upper_hard=float(np.mean(vals > hi - .01 * (hi - lo))),
                effective_upper=effective,
                fraction_within_1pct_causal_upper=float(np.mean(vals > effective - .01 * effective)) if 40 <= i < 60 else np.nan))
        corr = np.corrcoef(matrix, rowvar=False)
        ia, ib = np.triu_indices(n_params, 1)
        for pos in np.argsort(-np.abs(corr[ia, ib]))[:20]:
            strongest.append(dict(Group=group, parameter_1=names[ia[pos]], parameter_2=names[ib[pos]],
                                  correlation=float(corr[ia[pos], ib[pos]])))
        loaded[group] = dict(summary=summary, conditions=conditions, params=params, posterior=posterior,
                            report=json.loads((folder / 'convergence_summary.json').read_text()),
                            sub=sub, fit=fitted, score=score, corr=corr)
    tables = {
        'parameter_summary.csv': pd.concat(summaries, ignore_index=True),
        'fit_quality.csv': pd.DataFrame(quality),
        'condition_fit_quality.csv': pd.concat(condition_quality, ignore_index=True),
        'psychometric_comparison.csv': pd.concat(condition_quality, ignore_index=True),
        'rtd_histograms.csv': pd.concat(histograms, ignore_index=True),
        'rtd_model_curves.csv': pd.concat(mixtures, ignore_index=True),
        'condition_model_curves.csv': pd.concat(condition_curves, ignore_index=True),
        'posterior_boundary_audit.csv': pd.DataFrame(boundary),
        'strongest_correlations.csv': pd.DataFrame(strongest),
    }
    for name, table in tables.items():
        table.to_csv(out / name, index=False, float_format='%.17g')
    hist, curves = tables['rtd_histograms.csv'], tables['rtd_model_curves.csv']
    ymax = np.ceil(max(hist.density.max(), curves.density.max()) * 1.08 * 2) / 2
    figure_paths = []
    for group in groups:
        info = loaded[group]
        fig = plt.figure(figsize=(21.8, 7.6))
        grid = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, 1, 1, 1.7], left=.055, right=.985,
                               bottom=.18, top=.79, hspace=.38, wspace=.33)
        for r, level in enumerate(levels):
            for c, delta in enumerate(absolute):
                ax = fig.add_subplot(grid[r, c])
                h = hist.loc[hist.Group.eq(group) & hist.ASL_cd.eq(level) & hist.abs_delta.eq(delta)]
                m = curves.loc[curves.Group.eq(group) & curves.ASL_cd.eq(level) & curves.abs_delta.eq(delta)]
                ax.stairs(h.density.to_numpy(), np.r_[h.left_s.to_numpy(), h.right_s.iloc[-1]],
                          color=level_colors[level], alpha=.58, lw=1.5)
                ax.plot(m.time_s, m.density, color=level_colors[level], lw=2.1)
                ax.set(xlim=(0, 1.8), ylim=(0, ymax), xticks=[0, .6, 1.2, 1.8],
                       title=f'|ΔdB| = {delta:g}  ·  n={int(h.n.iloc[0])}')
                ax.grid(alpha=.16)
                if c == 0:
                    ax.set_ylabel(f'ASL {level:g} cd/m²\nDensity (s⁻¹)')
                else:
                    ax.tick_params(labelleft=False)
                if r == 1:
                    ax.set_xlabel('RT (s)')
        ax = fig.add_subplot(grid[:, 5])
        for level in levels:
            p = info['fit'].loc[lambda x: x.ASL_cd.eq(level)].sort_values('signed_delta')
            ax.scatter(p.signed_delta, p.empirical_p_right, color=level_colors[level], s=25, zorder=3)
            ax.plot(p.signed_delta, p.model_p_right, color=level_colors[level], lw=2,
                    label=f'ASL {level:g} cd/m²')
        ax.set(title='Psychometric' + ('\nChoices not fitted' if mode == 'rt_only' else ''),
               xlabel='Signed ΔdB (right − left)', ylabel='P(right)', xlim=(-2.15, 2.15),
               ylim=(-.015, 1.015), xticks=[-2, -1, 0, 1, 2])
        ax.grid(alpha=.18)
        ax.legend(loc='lower right', fontsize=9)
        fig.suptitle(f'{group} | human simultaneous RT | {len(info["sub"]):,} trials from '
                     f'{info["sub"].Participant.nunique()} participants pooled', color=group_colors[group],
                     fontsize=18, fontweight='bold', y=.97)
        fig.text(.5, .915, f'Reactive-only, {labels[mode]} | 20 Gamma–Omega–delay triples + shared bias'
                 + (' + two shared lapse probabilities' if mode == 'rt_choice_lapse' else '')
                 + f' ({n_params} parameters) | delay prior 0–500 ms', ha='center', fontsize=10.5)
        fig.legend(handles=[Line2D([], [], color='.5', lw=1.5, alpha=.65, drawstyle='steps-mid',
                                   label='Data: 20 ms bins / choice dots'),
                            Line2D([], [], color='.2', lw=2.1,
                                   label='Analytic model at arithmetic posterior means')],
                   loc='upper center', bbox_to_anchor=(.5, .89), ncol=2, frameon=False)
        status = 'Convergence checks passed' if not info['report']['flagged'] else 'PROVISIONAL: convergence checks not satisfied'
        fig.text(.5, .103, 'Condition = RT; 0–1.8 s retained, both choices; raw trials pooled across participants and sessions',
                 ha='center', fontsize=10)
        fig.text(.5, .065, 'RTDs pool ±Δ using retained-trial frequencies; each signed prediction has unit mass on 0–1.8 s',
                 ha='center', fontsize=10)
        fig.text(.5, .027, f'{status} | model psychometric points joined only as guides; no sigmoid or interpolation',
                 ha='center', fontsize=10)
        stem = f'human_{group.lower()}_gamma_omega_rtd_psychometric'
        for suffix in ['png', 'pdf']:
            fig.savefig(out / f'{stem}.{suffix}', dpi=180)
            figure_paths.append(f'{stem}.{suffix}')
        plt.close(fig)

    # %% Parameter comparison with shared scalar panels in a separate right column.
    fig = plt.figure(figsize=(17.5, 13.2))
    grid = fig.add_gridspec(3, 3, width_ratios=[1, 1, .65], left=.065, right=.975,
                           bottom=.13, top=.88, wspace=.32, hspace=.32)
    ticks = sorted(selected.signed_delta.unique())
    for r, (block, label, scale) in enumerate([('gamma', 'Gamma (dimensionless)', 1.),
                                                ('omega', 'Omega (s⁻¹)', 1.),
                                                ('t_E_aff', 'Delay (ms)', 1000.)]):
        axes = []
        for c, level in enumerate(levels):
            ax = fig.add_subplot(grid[r, c], sharey=axes[0] if axes else None)
            axes.append(ax)
            for group, offset, marker in [('Control', -.018, 'o'), ('ASD', .018, '*')]:
                p = loaded[group]['summary'].loc[lambda x: x.block.eq(block) & x.ASL_cd.eq(level)].sort_values('signed_delta')
                y = p['mean'].to_numpy() * scale
                ax.errorbar(p.signed_delta + offset, y,
                            yerr=np.vstack([y - p.q025.to_numpy() * scale, p.q975.to_numpy() * scale - y]),
                            fmt=marker, color=group_colors[group], ms=4.5 if group == 'Control' else 7.,
                            capsize=2, lw=1.1, label=group)
            ax.set(xticks=ticks, xlim=(-2.14, 2.14), ylabel=label if c == 0 else '')
            ax.tick_params(axis='x', rotation=55, labelsize=8)
            ax.grid(alpha=.18)
            if block == 'gamma':
                ax.axhline(0, color='.4', lw=.8)
                ax.set_title(f'ASL {level:g} cd/m²', fontsize=14)
            if block == 'omega':
                ax.set_ylim(bottom=0)
            if block == 't_E_aff':
                ax.set_ylim(0, 510)
                ax.set_xlabel('Signed ΔdB (right − left)')
        # Setting a lower limit disables shared-axis autoscaling; include both ASLs' intervals explicitly.
        if block in ['gamma', 'omega']:
            intervals = pd.concat([loaded[group]['summary'].loc[lambda x: x.block.eq(block)]
                                   for group in groups], ignore_index=True)
            lower, upper = float(intervals.q025.min()) * scale, float(intervals.q975.max()) * scale
            margin = .08 * (upper - min(0, lower))
            axes[0].set_ylim(0 if block == 'omega' else lower - margin, upper + margin)
        if r == 0:
            axes[0].legend(frameon=False)
    scalars = [('w', 'Shared bias w', 1., 'Fractional starting position')]
    if mode == 'rt_choice_lapse':
        scalars += [('lapse_prob', 'Shared lapse probability p', .4, 'Lapse fraction among retained trials'),
                    ('lapse_prob_right', 'Shared lapse-right probability', 1., 'P(right | lapse)')]
    for r, (name, title, high, ylabel) in enumerate(scalars):
        ax = fig.add_subplot(grid[r, 2])
        for i, group in enumerate(groups):
            p = loaded[group]['summary'].loc[lambda x: x.parameter.eq(name)].iloc[0]
            ax.errorbar(i, p['mean'], yerr=[[p['mean'] - p.q025], [p.q975 - p['mean']]],
                        fmt='o' if group == 'Control' else '*', color=group_colors[group],
                        capsize=4, ms=6 if group == 'Control' else 9, lw=1.8)
            ax.text(i, p.q975 + .025 * high, f'{p["mean"]:.3f}\n[{p.q025:.3f}, {p.q975:.3f}]',
                    ha='center', va='bottom', fontsize=8, color=group_colors[group])
        if name in ['w', 'lapse_prob_right']:
            ax.axhline(.5, color='.5', ls='--', lw=1)
        ax.set(title=title, xticks=[0, 1], xticklabels=groups, xlim=(-.6, 1.6),
               ylim=(0, high * 1.12), ylabel=ylabel)
        ax.tick_params(axis='x', rotation=30)
        ax.grid(axis='y', alpha=.18)
    fig.suptitle(f'Human {labels[mode]} | Gamma–Omega posterior comparison', y=.97, fontsize=20)
    fig.text(.5, .925, f'Control blue; ASD red | means and equal-tailed 95% variational credible intervals | '
             f'{n_params} parameters per group', ha='center', fontsize=11)
    fig.text(.5, .064, 'Gamma, Omega and delay vary independently across all 20 signed stimuli; scalar parameters are shared within group',
             ha='center', fontsize=10)
    fig.text(.5, .036, 'Pooled-trial variational uncertainty does not represent participant-to-participant variability.',
             ha='center', fontsize=10)
    for suffix in ['png', 'pdf']:
        fig.savefig(out / f'human_gamma_omega_parameter_comparison.{suffix}', dpi=180)
        figure_paths.append(f'human_gamma_omega_parameter_comparison.{suffix}')
    plt.close(fig)

    # %% Each objective's ELBO is used only for its own convergence diagnostics.
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5), gridspec_kw={'width_ratios': [1.2, 1]})
    for r, group in enumerate(groups):
        trace = pd.read_csv(out / group.lower() / 'convergence.csv')
        report = loaded[group]['report']
        ax = axes[r, 0]
        ax.plot(trace.step / 1000, trace.mean_negative_elbo, color=group_colors[group])
        ax.axvline(report['restored_best_step'] / 1000, color='.2', ls='--',
                   label=f'Restored {report["restored_best_step"]:,}')
        ax.set(title=f'{group}: window-mean negative ELBO', xlabel='Updates (thousands)', ylabel='Negative ELBO')
        ax.legend(frameon=False)
        ax.grid(alpha=.2)
        ax.text(.98, .96, f'Follow-up: {report["confirmation_windows"]} windows\n'
                f'Terminal shift: {report["maximum_terminal_mean_change_sd"]:.3f} SD',
                transform=ax.transAxes, ha='right', va='top', fontsize=9)
        ax = axes[r, 1]
        im = ax.imshow(loaded[group]['corr'], vmin=-1, vmax=1, cmap='coolwarm', origin='lower')
        scalar_tick = (60 + n_params - 1) / 2
        ax.set(title=f'{group}: physical posterior correlations', xticks=[9.5, 29.5, 49.5, scalar_tick],
               xticklabels=['Gamma', 'Omega', 'Delay', 'w' if n_params == 61 else 'w, p, q'],
               yticks=[9.5, 29.5, 49.5, scalar_tick],
               yticklabels=['Gamma', 'Omega', 'Delay', 'w' if n_params == 61 else 'w, p, q'])
        for cut in [19.5, 39.5, 59.5]:
            ax.axhline(cut, color='white', lw=.6)
            ax.axvline(cut, color='white', lw=.6)
        fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    fig.suptitle(f'{labels[mode]} | SVI convergence and posterior dependence', fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, .96))
    for suffix in ['png', 'pdf']:
        fig.savefig(out / f'human_gamma_omega_convergence_correlations.{suffix}', dpi=180)
        figure_paths.append(f'human_gamma_omega_convergence_correlations.{suffix}')
    plt.close(fig)
    manifest = dict(mode=mode, figures=figure_paths, tables=list(tables), rt_bin_s=.02, curve_step_s=.001,
                    rt_window_s=[0, 1.8], curves='arithmetic posterior means of physical parameters',
                    empirical_pooling='raw trials; signed conditions pooled only for displayed RTDs',
                    interval='equal-tailed 95% variational credible',
                    psychometric_data='Observed P(right) dots without error bars',
                    psychometric_model='Conditional probabilities at measured stimuli joined only as visual guides',
                    choice_in_objective=mode != 'rt_only', source_sha256=sha256(Path(__file__)))
    (out / 'plot_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(pd.DataFrame(quality).to_string(index=False))
    print('Saved', len(figure_paths), 'figure files in', out)


# %% Common comparison: objective-independent RT scores on identical retained rows.
else:
    reference_paths = {'rt_choice': args.baseline_dir, 'rt_only': args.rt_only_dir,
                       'rt_choice_lapse': args.lapse_dir}
    styles = {'rt_choice': '--', 'rt_only': ':', 'rt_choice_lapse': '-'}
    for path in reference_paths.values():
        assert (path / 'selected_trials.csv').read_bytes() == (args.baseline_dir / 'selected_trials.csv').read_bytes()
        assert (path / 'excluded_trials.csv').read_bytes() == (args.baseline_dir / 'excluded_trials.csv').read_bytes()
    _, selected, _, _ = load_data(mode)
    scores, probabilities, pooled_curves, pooled_histograms, mapping = [], [], [], [], []
    for model_mode, path in reference_paths.items():
        for group in groups:
            sub, conditions, data, _, _ = prepare_group(selected, group, mode, BASELINE_ROOT)
            draw_mode = 'rt_only' if model_mode == 'rt_choice' else model_mode
            posterior_file = path / group.lower() / 'posterior_samples.npz'
            params = vector_params(sample_matrix(dict(np.load(posterior_file)), draw_mode).mean(0), draw_mode)
            fitted, score = assess(params, data, sub, conditions, model_mode, labels[model_mode], group)
            score['free_parameters'] = 63 if model_mode == 'rt_choice_lapse' else 61
            scores.append(score)
            probabilities.append(fitted)
            prediction = prediction_components(params, times, model_mode, K=40)
            density, cdf = np.asarray(prediction['density']), np.asarray(prediction['cdf'])
            for level in levels:
                c = conditions.loc[conditions.ASL_cd.eq(level)].sort_values('condition_index')
                ids = c.condition_index.to_numpy(dtype=int)
                weights = c.n.to_numpy() / c.n.sum()
                pooled_curves.append(pd.DataFrame(dict(Group=group, model=labels[model_mode], mode=model_mode,
                    ASL_cd=level, time_s=times, density=density[:, ids] @ weights,
                    cdf=cdf[:, ids] @ weights, n=int(c.n.sum()))))
                mapping.append(pd.DataFrame(dict(Group=group, model=labels[model_mode], mode=model_mode,
                    ASL_cd=level, condition_index=ids, signed_delta=c.signed_delta.to_numpy(),
                    n=c.n.to_numpy(), weight=weights)))
                if model_mode == 'rt_choice':
                    obs = sub.loc[sub.ASL_cd.eq(level), 'RT_s'].to_numpy()
                    counts, _ = np.histogram(obs, bins=edges)
                    pooled_histograms.append(pd.DataFrame(dict(Group=group, ASL_cd=level,
                        left_s=edges[:-1], right_s=edges[1:], count=counts,
                        n=len(obs), density=counts / (len(obs) * .02))))
    tables = {
        'variant_comparison_metrics.csv': pd.DataFrame(scores),
        'variant_comparison_psychometrics.csv': pd.concat(probabilities, ignore_index=True),
        'variant_comparison_condition_fit_quality.csv': pd.concat(probabilities, ignore_index=True),
        'variant_comparison_pooled_rtd_curves.csv': pd.concat(pooled_curves, ignore_index=True),
        'variant_comparison_pooled_rtd_histograms.csv': pd.concat(pooled_histograms, ignore_index=True),
        'variant_comparison_mixture_weights.csv': pd.concat(mapping, ignore_index=True),
    }
    for name, table in tables.items():
        table.to_csv(out / name, index=False, float_format='%.17g')
    curves = tables['variant_comparison_pooled_rtd_curves.csv']
    hist = tables['variant_comparison_pooled_rtd_histograms.csv']
    psych = tables['variant_comparison_psychometrics.csv']
    ymax = np.ceil(max(hist.density.max(), curves.density.max()) * 1.08 * 2) / 2
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.2), gridspec_kw={'width_ratios': [1, 1.45]})
    for r, group in enumerate(groups):
        ax = axes[r, 0]
        for level in levels:
            empirical = psych.loc[psych.Group.eq(group) & psych.ASL_cd.eq(level) & psych['mode'].eq('rt_choice')].sort_values('signed_delta')
            ax.scatter(empirical.signed_delta, empirical.empirical_p_right, color=level_colors[level], s=25, zorder=4)
            for model_mode in reference_paths:
                p = psych.loc[psych.Group.eq(group) & psych.ASL_cd.eq(level) & psych['mode'].eq(model_mode)].sort_values('signed_delta')
                ax.plot(p.signed_delta, p.model_p_right, color=level_colors[level], ls=styles[model_mode], lw=1.8)
        ax.set(title=f'{group}: psychometric', xlabel='Signed ΔdB (right − left)', ylabel='P(right)',
               xlim=(-2.15, 2.15), ylim=(-.015, 1.015))
        ax.grid(alpha=.18)
        ax = axes[r, 1]
        for level in levels:
            h = hist.loc[hist.Group.eq(group) & hist.ASL_cd.eq(level)]
            ax.stairs(h.density.to_numpy(), np.r_[h.left_s.to_numpy(), h.right_s.iloc[-1]],
                      color=level_colors[level], alpha=.35, lw=1.3)
            for model_mode in reference_paths:
                p = curves.loc[curves.Group.eq(group) & curves.ASL_cd.eq(level) & curves['mode'].eq(model_mode)]
                ax.plot(p.time_s, p.density, color=level_colors[level], ls=styles[model_mode], lw=1.8)
        ax.set(title=f'{group}: RTD pooled across signed differences', xlabel='Stored RT_s (s)',
               ylabel='Density (s⁻¹)', xlim=(0, 1.8), ylim=(0, ymax))
        ax.grid(alpha=.18)
    fig.suptitle('Human simultaneous RT | baseline, RT-only, and RT + choice + lapses', fontsize=17, y=.985)
    handles = [Line2D([], [], color=level_colors[level], lw=2, label=f'ASL {level:g} cd/m²') for level in levels]
    handles += [Line2D([], [], color='.25', lw=2, ls=styles[m], label=labels[m]) for m in reference_paths]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .945), ncol=5, frameon=False, fontsize=9)
    fig.text(.5, .052, 'Identical 0–1.8 s RT trials; raw-trial pooling and retained-frequency mixtures; data dots and 20 ms steps',
             ha='center', fontsize=10)
    fig.text(.5, .025, 'Curves use arithmetic posterior means. RT-only choices are diagnostic predictions, not fitted observations.',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(.02, .08, .99, .9))
    figure_paths = []
    for suffix in ['png', 'pdf']:
        name = f'human_gamma_omega_variant_comparison.{suffix}'
        fig.savefig(out / name, dpi=180)
        figure_paths.append(name)
    plt.close(fig)
    manifest = dict(figures=figure_paths, tables=list(tables), source_sha256=sha256(Path(__file__)),
                    sources={m: str(path.relative_to(ROOT)) for m, path in reference_paths.items()},
                    posterior_sha256={f'{m}/{g}': sha256(path / g.lower() / 'posterior_samples.npz')
                                      for m, path in reference_paths.items() for g in groups},
                    score='Common marginal RT log likelihood for all; joint likelihood only for joint models',
                    normalization='Each signed component conditioned on 0–1.8 s before retained-frequency mixing')
    (out / 'variant_comparison_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(pd.DataFrame(scores).to_string(index=False))
    print('Saved common comparison in', out)
