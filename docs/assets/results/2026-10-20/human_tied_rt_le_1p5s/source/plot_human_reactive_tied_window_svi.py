# %%
"""Posterior-mean diagnostics for RT <= 1.5 s; derived from the original human plotter."""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/human_tied_mpl")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import jax
import jax.numpy as jnp
from human_reactive_tied_window_utils import (
    ROOT, PARAM_NAMES, load_trials, group_data, bound_cdf, bound_density,
    retained_mass, log_likelihood, sha256,
)

# %% Inputs and common plotting coordinates; normalization never changes with zoom.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, default=ROOT/"ASD_Human_Data/reactive_tied_rt_svi_rt_le_1p5s")
args = parser.parse_args()
out = args.output_dir
frame, provenance = load_trials()
times = np.linspace(0, 1.5, 1501)
edges = np.linspace(0, 1.5, 76)
centers = (edges[1:]+edges[:-1])/2
asls = [20, 40]
abs_deltas = [.1, .3, .65, 1.3, 2.]
colors = ["#1f77b4", "#ff7f0e"]
group_colors = {"Control": "#1f77b4", "ASD": "#d62728"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.facecolor": "white"})
psych_rows, histogram_rows, quality_rows, cdf_rows, summary_rows = [], [], [], [], []
smooth_psych_rows = []
models = {}
statuses = {}

# %% Trial-level likelihood and condition curves, evaluated at arithmetic physical means.
for group in ["Control", "ASD"]:
    folder = out/group.lower()
    fit_report = json.loads((folder/"convergence_summary.json").read_text())
    if "minimum_followup_confirmed" in fit_report:
        statuses[group] = (f"{fit_report['completed_steps']:,} steps; "
                           + ("25-window check passed" if not fit_report["flagged"] else "unconfirmed"))
    else:
        statuses[group] = (f"{fit_report['completed_steps']:,} steps; mean stability "
                           + ("FAILED" if fit_report["flagged"] else "passed"))
    sub, conditions, data, delay_upper = group_data(frame, group)
    samples = np.load(folder/"posterior_samples.npz")
    matrix = np.column_stack([samples[n] for n in PARAM_NAMES])
    means = matrix.mean(axis=0)
    params = jnp.asarray(means)
    summary = pd.read_csv(folder/"posterior_summary.csv")
    summary.insert(0, "Group", group)
    summary_rows.extend(summary.to_dict("records"))
    B = jnp.asarray(conditions.Mean_dB.to_numpy())[:, None]
    D = jnp.asarray(conditions.signed_delta.to_numpy())[:, None]
    normalizer = np.asarray(retained_mass(B, D, params))
    evaluate_density = jax.jit(lambda t: sum(bound_density(t, c, B, D, params) for c in [-1, 1]))
    evaluate_cdf = jax.jit(lambda t: sum(bound_cdf(t, c, B, D, params) for c in [-1, 1]))
    density = np.asarray(evaluate_density(jnp.asarray(times)[None, :]))/normalizer
    cdf = np.asarray(evaluate_cdf(jnp.asarray(times)[None, :]))/normalizer
    right_mass = np.asarray(bound_cdf(1.5, 1, B, D, params))/normalizer
    smooth_psych = {}
    psych_x = np.linspace(-2.2, 2.2, 441)
    for asl in asls:
        level = float(conditions.loc[conditions.ASL_cd.eq(asl), "Mean_dB"].mean())
        probability = np.asarray(bound_cdf(1.5, 1, level, jnp.asarray(psych_x), params)
                                 /retained_mass(level, jnp.asarray(psych_x), params))
        smooth_psych[asl] = probability
        smooth_psych_rows.extend(dict(Group=group, ASL_cd=asl, Mean_dB=level, Delta_dB=x,
                                       p_right_model=y) for x,y in zip(psych_x, probability))
    np.savez_compressed(folder/"mean_parameter_curves.npz", time_s=times, density=density,
                        cdf=cdf, p_right=right_mass.ravel(), condition_B=np.asarray(B).ravel(),
                        condition_D=np.asarray(D).ravel(), mean_parameters=means)
    for row in conditions.itertuples():
        i = row.condition_index
        selected = sub.loc[sub.condition_index.eq(i)]
        empirical = selected.Choice.eq(1).mean()
        predicted = float(right_mass[i, 0])
        psych_rows.append(dict(Group=group, ASL_cd=row.ASL_cd, Delta_dB=row.signed_delta,
                               Mean_dB=row.Mean_dB, n=len(selected), n_right=int(selected.Choice.eq(1).sum()),
                               p_right_data=empirical, p_right_model=predicted,
                               binomial_sem=np.sqrt(empirical*(1-empirical)/len(selected)),
                               retained_mass=float(normalizer[i, 0])))
        ordered = np.sort(selected.RT_s.to_numpy())
        evaluated = sum(np.asarray(bound_cdf(jnp.asarray(ordered), c, row.Mean_dB,
                       row.signed_delta, params)) for c in [-1, 1])/normalizer[i, 0]
        n = len(ordered)
        ks = max(np.max(np.arange(1, n+1)/n-evaluated), np.max(evaluated-np.arange(n)/n))
        cdf_rows.append(dict(Group=group, ASL_cd=row.ASL_cd, Delta_dB=row.signed_delta, n=n,
                             cdf_max_abs_error=float(ks), mean_rt_data_s=float(ordered.mean()),
                             mean_rt_model_s=float(np.trapz(times*density[i], times))))
    group_psych = pd.DataFrame(psych_rows).query("Group == @group")
    group_cdf = pd.DataFrame(cdf_rows).query("Group == @group")
    quality_rows.append(dict(Group=group, n_trials=len(sub), n_participants=sub.Participant.nunique(),
                             mean_parameter_joint_log_likelihood=float(log_likelihood(params, data)),
                             psychometric_rmse=float(np.sqrt(np.mean((group_psych.p_right_model-group_psych.p_right_data)**2))),
                             psychometric_max_abs_error=float(np.max(abs(group_psych.p_right_model-group_psych.p_right_data))),
                             mean_condition_cdf_error=float(group_cdf.cdf_max_abs_error.mean()),
                             max_condition_cdf_error=float(group_cdf.cdf_max_abs_error.max())))
    mixtures = {}
    for asl in asls:
        for abs_delta in [None]+abs_deltas:
            keep = conditions.ASL_cd.eq(asl)
            trial_keep = sub.ASL_cd.eq(asl)
            if abs_delta is not None:
                keep &= conditions.signed_delta.abs().eq(abs_delta)
                trial_keep &= sub.signed_delta.abs().eq(abs_delta)
            indices = conditions.loc[keep, "condition_index"].to_numpy()
            weights = conditions.loc[keep, "n"].to_numpy()/conditions.loc[keep, "n"].sum()
            model = np.average(density[indices], axis=0, weights=weights)
            selected_rt = sub.loc[trial_keep, "RT_s"].to_numpy()
            counts = np.histogram(selected_rt, edges)[0]
            hist = counts/(len(selected_rt)*np.diff(edges))
            mixtures[(asl, abs_delta)] = (hist, model)
            for i, count in enumerate(counts):
                histogram_rows.append(dict(Group=group, ASL_cd=asl, abs_Delta_dB="all" if abs_delta is None else abs_delta,
                                           bin_left_s=edges[i], bin_right_s=edges[i+1], count=int(count),
                                           n=len(selected_rt), density=hist[i]))
    models[group] = (sub, conditions, right_mass.ravel(), mixtures, smooth_psych)

    # Parameter corner: log10 axes for the three positive scale parameters.
    plotted = matrix.copy()
    plotted[:, :3] = np.log10(plotted[:, :3])
    plotted[:, 4] *= 1000
    labels = [r"log$_{10}$ λ", r"log$_{10}$ T₀ (s)", r"log$_{10}$ θE", "w", "tE aff (ms)"]
    fig, axes = plt.subplots(5, 5, figsize=(13, 12))
    limits = np.quantile(plotted, [.001, .999], axis=0)
    for i in range(5):
        for j in range(5):
            ax = axes[i,j]
            if i < j:
                ax.axis("off")
            elif i == j:
                ax.hist(plotted[:,j], bins=50, color=group_colors[group], alpha=.8)
                ax.set_xlim(limits[:,j])
            else:
                ax.hexbin(plotted[:,j], plotted[:,i], gridsize=35, mincnt=1, cmap="Blues" if group=="Control" else "Reds")
                ax.set_xlim(limits[:,j]); ax.set_ylim(limits[:,i])
            if i == 4: ax.set_xlabel(labels[j])
            elif i >= j: ax.tick_params(labelbottom=False)
            if j == 0 and i > 0: ax.set_ylabel(labels[i])
    fig.suptitle(f"{group} | 10,000 draws from the variational approximation\n"
                 "Pooled trials; uncertainty does not measure participant variability", fontsize=15)
    fig.tight_layout(rect=(0,0,1,.95))
    for ext in ["png","pdf"]: fig.savefig(folder/f"parameter_corner.{ext}", dpi=170)
    plt.close(fig)

    convergence = pd.read_csv(folder/"convergence.csv")
    report = json.loads((folder/"convergence_summary.json").read_text())
    fig, axes = plt.subplots(2,3,figsize=(15,8))
    ax = axes.flat[0]
    ax.plot(convergence.step, convergence.mean_negative_elbo, color=group_colors[group])
    ax.set_ylabel("1,000-step mean negative ELBO")
    for ax, name in zip(list(axes.flat)[1:], PARAM_NAMES):
        ax.plot(convergence.step, convergence[name+"_mean"], color=group_colors[group])
        ax.fill_between(convergence.step, convergence[name+"_mean"]-convergence[name+"_sd"],
                        convergence[name+"_mean"]+convergence[name+"_sd"], alpha=.2,color=group_colors[group])
        ax.set_ylabel(name+" (mean ±1 SD)")
    for ax in axes.flat:
        ax.axvline(report["restored_best_step"], color="k",ls=":", label="Restored best")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value,position:f"{value/1000:g}"))
        ax.set_xlabel("SVI steps (thousands)")
        ax.grid(alpha=.15)
    fig.suptitle(f"{group} | selected best {report['restored_best_step']:,}; final checked {report['completed_steps']:,}\n"
                 f"{report['confirmation_windows']} follow-up windows; local mean stability "
                 + ("passed" if not report['flagged'] else "unconfirmed"))
    fig.tight_layout(rect=(0,0,1,.93))
    for ext in ["png","pdf"]: fig.savefig(folder/f"convergence.{ext}",dpi=160)
    plt.close(fig)

# %% Save all numerical points used in the comparisons.
psych = pd.DataFrame(psych_rows)
psych.to_csv(out/"psychometric_comparison.csv",index=False,float_format="%.17g")
pd.DataFrame(smooth_psych_rows).to_csv(out/"psychometric_model_curve.csv",index=False,float_format="%.17g")
pd.DataFrame(histogram_rows).to_csv(out/"rtd_histograms.csv",index=False,float_format="%.17g")
pd.DataFrame(quality_rows).to_csv(out/"fit_quality.csv",index=False,float_format="%.17g")
pd.DataFrame(cdf_rows).to_csv(out/"condition_cdf_discrepancies.csv",index=False,float_format="%.17g")
pd.DataFrame(summary_rows).to_csv(out/"parameter_summary.csv",index=False,float_format="%.17g")
footer = ("Condition = RT, RT ≤ 1.5 s | retained correct and incorrect choices | pooled raw trials and sessions\n"
          "Solid: model at arithmetic posterior means; steps/points: data | joint RT + choice fit, conditional on 0–1.5 s\n"
          "Five shared parameters per group | variational approximation; pooled uncertainty excludes participant variability")
legend = [Line2D([0],[0],color=c,label=f"ASL {asl} cd/m²") for c,asl in zip(colors,asls)]
legend += [Line2D([0],[0],color=".3",lw=2,label="Model"),Line2D([0],[0],color=".3",lw=1,ls="--",label="Data RTD")]

# %% Two comparison layouts, each with full and zoomed RTD views.
for limit, name in [(1.5, "full_0_1p5s"),(1.,"zoom_0_1s")]:
    fig, axes = plt.subplots(2,2,figsize=(15,10))
    for row,group in enumerate(["Control","ASD"]):
        sub, conditions, pright, mixtures, smooth_psych = models[group]
        for color,asl in zip(colors,asls):
            empirical = psych.loc[psych.Group.eq(group)&psych.ASL_cd.eq(asl)].sort_values("Delta_dB")
            axes[row,0].errorbar(empirical.Delta_dB,empirical.p_right_data,yerr=empirical.binomial_sem,
                                 fmt="o",ms=5,color=color,capsize=2,alpha=.85)
            axes[row,0].plot(psych_x,smooth_psych[asl],color=color,lw=2)
            hist, model = mixtures[(asl,None)]
            axes[row,1].stairs(hist,edges,color=color,lw=1.2,ls="--",alpha=.8)
            axes[row,1].plot(times,model,color=color,lw=2)
        axes[row,0].set(title=f"{group} | {sub.Participant.nunique()} participants | {len(sub):,} RT trials\n{statuses[group]}",
                        xlabel="Signed ΔdB (right − left)",ylabel="P(right)",ylim=(-.02,1.02))
        axes[row,0].axhline(.5,color=".7",ls=":",lw=.8)
        axes[row,1].set(title=f"{group} | RTD pooled across signed differences",
                        xlabel="Recorded RT (s)",ylabel="Density (s⁻¹)",xlim=(0,limit),ylim=(0,None))
        for ax in axes[row]: ax.grid(alpha=.15)
    ymax=max(ax.get_ylim()[1] for ax in axes[:,1])
    for ax in axes[:,1]: ax.set_ylim(0,ymax)
    fig.suptitle("Human RT ≤ 1.5 s | reactive-only vanilla/IPL TIED | one model per group",fontsize=17,y=.985)
    fig.legend(handles=legend,loc="upper center",bbox_to_anchor=(.5,.955),ncol=4,frameon=False)
    fig.text(.5,.022,footer,ha="center",va="bottom",fontsize=10)
    fig.tight_layout(rect=(0,.11,1,.915),h_pad=3.)
    for ext in ["png","pdf"]: fig.savefig(out/f"human_reactive_tied_psychometric_rtd_{name}.{ext}",dpi=180)
    plt.close(fig)

    fig,axes=plt.subplots(2,5,figsize=(21,9),sharex=True,sharey=True)
    for row,group in enumerate(["Control","ASD"]):
        for col,delta in enumerate(abs_deltas):
            ax=axes[row,col]
            for color,asl in zip(colors,asls):
                hist,model=models[group][3][(asl,delta)]
                ax.stairs(hist,edges,color=color,ls="--",lw=1.2,alpha=.8)
                ax.plot(times,model,color=color,lw=2)
            ax.set(title=f"{group} | |ΔdB| = {delta:g}\n{statuses[group]}",xlim=(0,limit),xlabel="Recorded RT (s)")
            if col==0: ax.set_ylabel("Density (s⁻¹)")
            ax.grid(alpha=.15)
    # Explicit common limit includes every data step and model peak. Setting a
    # lower limit early on shared axes can otherwise freeze the first panel's range.
    ymax=max(float(np.max(array)) for group in models for (asl,delta),(hist,model)
             in models[group][3].items() if delta is not None for array in [hist,model])
    axes[0,0].set_ylim(0,1.08*ymax)
    fig.suptitle("Human RT ≤ 1.5 s | reactive-only TIED RTDs | ±ΔdB pooled with observed trial weights",fontsize=18,y=.985)
    fig.legend(handles=legend,loc="upper center",bbox_to_anchor=(.5,.95),ncol=4,frameon=False)
    fig.text(.5,.02,footer,ha="center",va="bottom",fontsize=11)
    fig.tight_layout(rect=(0,.13,1,.91),h_pad=3.)
    for ext in ["png","pdf"]: fig.savefig(out/f"human_reactive_tied_rtd_by_delta_{name}.{ext}",dpi=180)
    plt.close(fig)

manifest = dict(source=provenance, data_bins_s=.02, model_grid_s=.001, normalization_window_s=[0,1.5],
                prediction_parameters="arithmetic means of 10000 physical constrained posterior draws",
                psychometric_error_bars="binomial SEM of pooled choices",
                mixture_weights="observed condition n / total n within plotted cell",
                figure_hashes={p.name:sha256(p) for p in out.glob("*.png")},
                source_hash=sha256(__file__))
(out/"plot_manifest.json").write_text(json.dumps(manifest,indent=2))
print(pd.DataFrame(quality_rows).to_string(index=False))
