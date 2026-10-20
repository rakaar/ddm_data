# %%
"""Compare previous and refitted parameters on identical RT <= 1.5 s trials/window."""
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
import jax
import jax.numpy as jnp
from human_reactive_tied_window_utils import (
    ROOT, PARAM_NAMES, RT_UPPER, load_trials, group_data, bound_cdf,
    bound_density, retained_mass, log_likelihood, sha256,
)

# %% The baseline is the confirmed full-RT fit, reconditioned on the new window.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-dir', type=Path, default=ROOT/'ASD_Human_Data/reactive_tied_rt_svi_confirm25')
parser.add_argument('--output-dir', type=Path, default=ROOT/'ASD_Human_Data/reactive_tied_rt_svi_rt_le_1p5s')
args = parser.parse_args()
out = args.output_dir
frame, provenance = load_trials()
times = np.linspace(0, RT_UPPER, 1501)
edges = np.linspace(0, RT_UPPER, 76)
x = np.linspace(-2.2, 2.2, 441)
colors = ['#1f77b4', '#ff7f0e']
quality_rows, condition_rows, parameter_rows = [], [], []
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False})

for ri, group in enumerate(['Control', 'ASD']):
    sub, conditions, data, upper = group_data(frame, group)
    previous = np.load(args.source_dir/group.lower()/'posterior_samples.npz')
    refit = np.load(out/group.lower()/'posterior_samples.npz')
    oldmeans = np.array([previous[n].mean() for n in PARAM_NAMES])
    newmeans = np.array([refit[n].mean() for n in PARAM_NAMES])
    for i, n in enumerate(PARAM_NAMES):
        parameter_rows.append(dict(Group=group, parameter=n, previous_mean=oldmeans[i], refit_mean=newmeans[i],
                                   previous_sd=previous[n].std(ddof=1), refit_sd=refit[n].std(ddof=1)))
    for label, means, style in [('previous_reconditioned', oldmeans, ':'), ('refit', newmeans, '-')]:
        p = jnp.asarray(means)
        B = jnp.asarray(conditions.Mean_dB.to_numpy())[:, None]
        D = jnp.asarray(conditions.signed_delta.to_numpy())[:, None]
        norm = np.asarray(retained_mass(B, D, p))
        density = np.asarray(jax.jit(lambda t: sum(bound_density(t, c, B, D, p) for c in [-1, 1]))(
            jnp.asarray(times)[None, :]))/norm
        pc = np.asarray(bound_cdf(RT_UPPER, 1, B, D, p))/norm
        cellrows = []
        for row in conditions.itertuples():
            cell = sub.loc[sub.condition_index.eq(row.condition_index)]
            ordered = np.sort(cell.RT_s.to_numpy()); n = len(cell)
            cdf = sum(np.asarray(bound_cdf(jnp.asarray(ordered), c, row.Mean_dB, row.signed_delta, p))
                      for c in [-1, 1])/norm[row.condition_index, 0]
            ks = max(np.max(np.arange(1, n+1)/n-cdf), np.max(cdf-np.arange(n)/n))
            cellrows.append(dict(Group=group, fit=label, ASL_cd=row.ASL_cd, Delta_dB=row.signed_delta,
                                 n=n, n_right=int(cell.ChoseRight.sum()), p_right_data=cell.ChoseRight.mean(),
                                 p_right_model=float(pc[row.condition_index, 0]), cdf_max_abs_error=float(ks)))
        cells = pd.DataFrame(cellrows)
        condition_rows.extend(cellrows)
        quality_rows.append(dict(Group=group, fit=label, n_trials=len(sub),
            retention_upper_s=RT_UPPER, mean_parameter_joint_log_likelihood=float(log_likelihood(p, data)),
            psychometric_rmse=np.sqrt(np.mean((cells.p_right_data-cells.p_right_model)**2)),
            psychometric_max_abs_error=abs(cells.p_right_data-cells.p_right_model).max(),
            mean_condition_cdf_error=cells.cdf_max_abs_error.mean(), max_condition_cdf_error=cells.cdf_max_abs_error.max()))
        np.savez_compressed(out/group.lower()/f'comparison_{label}_curves.npz', density=density,
                            p_right=pc.ravel(), time_s=times, parameters=means,
                            condition_B=np.asarray(B).ravel(), condition_D=np.asarray(D).ravel())
        for color, asl in zip(colors, [20, 40]):
            keep = conditions.ASL_cd.eq(asl)
            selected = sub.loc[sub.ASL_cd.eq(asl)]
            level = conditions.loc[keep, 'Mean_dB'].mean()
            pright = np.asarray(bound_cdf(RT_UPPER, 1, level, jnp.asarray(x), p)
                                 /retained_mass(level, jnp.asarray(x), p))
            axes[ri, 0].plot(x, pright, color=color, ls=style, lw=2)
            weights = conditions.loc[keep, 'n'].to_numpy()/len(selected)
            axes[ri, 1].plot(times, np.average(density[keep], axis=0, weights=weights), color=color, ls=style, lw=2)
            if label == 'refit':
                empirical = cells.loc[cells.ASL_cd.eq(asl)]
                axes[ri, 0].scatter(empirical.Delta_dB, empirical.p_right_data, s=25, color=color)
                counts = np.histogram(selected.RT_s, edges)[0]
                axes[ri, 1].stairs(counts/len(selected)/np.diff(edges), edges, color=color, alpha=.5, lw=1)
    axes[ri, 0].set(title=f'{group} | {len(sub):,} retained trials', xlabel='Signed ΔdB (right − left)',
                    ylabel='P(right)', ylim=(-.02, 1.02))
    axes[ri, 1].set(title=f'{group} | pooled RTD', xlabel='Recorded RT (s)', ylabel='Density (s⁻¹)',
                    xlim=(0, RT_UPPER), ylim=(0, None))
    for ax in axes[ri]: ax.grid(alpha=.15)
ymax = max(ax.get_ylim()[1] for ax in axes[:, 1])
for ax in axes[:, 1]: ax.set_ylim(0, ymax)
legend = [Line2D([0], [0], color=c, label=f'ASL {asl} cd/m²') for c, asl in zip(colors, [20, 40])]
legend += [Line2D([0], [0], color='.3', lw=2, label='Refit: RT ≤ 1.5 s'),
           Line2D([0], [0], color='.3', ls=':', lw=2, label='Previous fit, reconditioned'),
           Line2D([0], [0], color='.5', alpha=.5, lw=1, label='Selected data')]
fig.suptitle('Human reactive-only TIED | testing the 1.5 s cutoff', fontsize=18, y=.985)
fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, .955), ncol=3, frameon=False)
fig.text(.5, .02, 'Both models evaluated on the same RT ≤ 1.5 s trials, conditional on 0–1.5 s\n'
         'Arithmetic physical posterior means | previous model was trained on all RTs (0–3 s)\n'
         'Five shared parameters/group; unchanged priors | posterior uncertainty is a variational approximation',
         ha='center', va='bottom', fontsize=10)
fig.tight_layout(rect=(0, .11, 1, .9), h_pad=3)
for ext in ['png', 'pdf']: fig.savefig(out/f'human_tied_rt_cutoff_comparison.{ext}', dpi=180)
plt.close(fig)
pd.DataFrame(quality_rows).to_csv(out/'common_window_fit_quality.csv', index=False, float_format='%.17g')
pd.DataFrame(condition_rows).to_csv(out/'common_window_conditions.csv', index=False, float_format='%.17g')
pd.DataFrame(parameter_rows).to_csv(out/'cutoff_parameter_changes.csv', index=False, float_format='%.17g')
(out/'comparison_manifest.json').write_text(json.dumps(dict(
    previous_fit=str(args.source_dir.resolve()), refit=str(out.resolve()), selection=provenance,
    comparison_window_s=[0., RT_UPPER], mixture_weights='Actual selected signed-condition counts',
    previous_predictions='Recomputed and reconditioned at 1.5 s; previous 0--3 s curve values are not reused',
    source_sha256=sha256(__file__)), indent=2))
print(pd.DataFrame(quality_rows).to_string(index=False))
