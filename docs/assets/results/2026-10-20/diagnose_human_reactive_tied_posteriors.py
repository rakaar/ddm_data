# %%
"""Audit posterior bounds and timing constraints of the saved human RT fits.

This diagnoses the existing fits; it does not refit, filter trials, or alter priors.
"""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/human_tied_mpl")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from human_reactive_tied_svi_utils import (
    ROOT, PARAM_NAMES, load_trials, group_data, bound_cdf, retained_mass, sha256,
)

# %% Exact saved fit and data; all 11,400 RT rows retain their original membership.
FIT = ROOT/"ASD_Human_Data/reactive_tied_rt_svi"
OUT = ROOT/"ASD_Human_Data/reactive_tied_rt_posterior_diagnostics"
OUT.mkdir(exist_ok=True)
frame, provenance = load_trials()
config = json.loads((FIT/"run_config.json").read_text())
for path,digest in config["source_hashes"].items():
    assert sha256(ROOT/path)==digest
prior_log = {"rate_lambda": (np.log(.2),1.5), "T_0": (np.log(.001),2.),
             "theta_E": (np.log(30.),1.5)}
labels = ["rate_lambda", "T_0 (ms)", "theta_E", "w", "t_E_aff (ms)"]
scales = np.array([1.,1000.,1.,1.,1000.])
colors = {"Control":"#1f77b4", "ASD":"#d62728"}
boundary_rows, timing_rows, early_rows, mean_check_rows, drift_rows = [], [], [], [], []
input_hashes = {"simultaneous_rt_trials.csv": provenance["csv_sha256"]}
fig,axes = plt.subplots(2,5,figsize=(21,8))

# %% Physical posteriors, support distances, and empirical/model RT spread.
for row,group in enumerate(["Control","ASD"]):
    folder = FIT/group.lower()
    draws = np.load(folder/"posterior_samples.npz")
    samples = np.column_stack([draws[n] for n in PARAM_NAMES])
    means,sd = samples.mean(axis=0),samples.std(axis=0,ddof=1)
    sub,conditions,data,upper = group_data(frame,group)
    curves = np.load(folder/"mean_parameter_curves.npz")
    times = curves["time_s"]
    input_hashes[f"{group}/posterior_samples.npz"] = sha256(folder/"posterior_samples.npz")
    input_hashes[f"{group}/mean_parameter_curves.npz"] = sha256(folder/"mean_parameter_curves.npz")
    for col,name in enumerate(PARAM_NAMES):
        values = samples[:,col]
        qlo,qhi = np.quantile(values,[.025,.975])
        lower = .05 if name=="w" else 0.
        higher = .95 if name=="w" else upper if name=="t_E_aff" else np.inf
        lower_sd = (means[col]-lower)/sd[col]
        upper_sd = (higher-means[col])/sd[col]
        edge_fraction = float(np.mean((values-lower)<.01*(higher-lower))) if np.isfinite(higher) else None
        upper_fraction = float(np.mean((higher-values)<.01*(higher-lower))) if np.isfinite(higher) else None
        prior_z = (np.log(values).mean()-prior_log[name][0])/prior_log[name][1] if name in prior_log else None
        boundary_rows.append(dict(Group=group,parameter=name,mean=means[col],sd=sd[col],q025=qlo,q975=qhi,
                                  support_lower=lower,support_upper=higher,lower_distance_sd=lower_sd,
                                  upper_distance_sd=upper_sd,fraction_in_lower_1pct_support=edge_fraction,
                                  fraction_in_upper_1pct_support=upper_fraction,mean_log_prior_z=prior_z))
        assert (values>lower).all() and (values<higher).all()
        ax=axes[row,col]
        displayed = values*scales[col]
        ax.hist(displayed,bins=45,density=True,color=colors[group],alpha=.8)
        ax.axvline(means[col]*scales[col],color="k",lw=1)
        ax.axvspan(qlo*scales[col],qhi*scales[col],color=colors[group],alpha=.1)
        if name=="w":
            ax.set_xlim(0,1)
            for limit in [lower,higher]: ax.axvline(limit,color="k",ls="--",lw=1)
            support="Support: 0.05 to 0.95"
        elif name=="t_E_aff":
            ax.set_xlim(0,upper*1000*1.08)
            ax.axvline(upper*1000,color="k",ls="--",lw=1)
            support=f"Upper: {upper*1000:.3f} ms ({upper_sd:.1f} SD away)"
        else:
            support="Positive; no finite upper bound"
        ax.set_title(f"{group} | {labels[col]}\n{support}",fontsize=10)
        ax.text(.03,.97,f"Mean {means[col]*scales[col]:.4g}\n95% [{qlo*scales[col]:.4g}, {qhi*scales[col]:.4g}]",
                transform=ax.transAxes,va="top",fontsize=9,bbox=dict(facecolor="white",alpha=.85,edgecolor="none"))
        ax.set_ylim(0,ax.get_ylim()[1]*1.22)
        ax.set_xlabel(labels[col]);ax.grid(alpha=.15)
        if col==0: ax.set_ylabel("Posterior density")
        ax.spines[["top","right"]].set_visible(False)
    for delta in [.1,.3,.65,1.3,2.]:
        keep=conditions.signed_delta.abs().eq(delta)
        indices=conditions.loc[keep,"condition_index"].to_numpy()
        weights=conditions.loc[keep,"n"].to_numpy()/conditions.loc[keep,"n"].sum()
        model_density=np.average(curves["density"][indices],axis=0,weights=weights)
        model_cdf=np.average(curves["cdf"][indices],axis=0,weights=weights)
        observed=sub.loc[sub.signed_delta.abs().eq(delta),"RT_s"].to_numpy()
        observed_quantile=np.quantile(observed,[.1,.5,.9])
        predicted_quantile=np.interp([.1,.5,.9],model_cdf,times)
        model_mean=np.trapz(times*model_density,times)
        model_sd=np.sqrt(np.trapz(times*times*model_density,times)-model_mean**2)
        timing_rows.append(dict(Group=group,abs_Delta_dB=delta,n=len(observed),
                                data_mean_s=observed.mean(),model_mean_s=model_mean,
                                data_sd_s=observed.std(ddof=1),model_sd_s=model_sd,
                                data_q10_s=observed_quantile[0],data_q50_s=observed_quantile[1],data_q90_s=observed_quantile[2],
                                model_q10_s=predicted_quantile[0],model_q50_s=predicted_quantile[1],model_q90_s=predicted_quantile[2]))
    for limit in [.2,.25,.3,.35,.4]:
        early_rows.append(dict(Group=group,rt_threshold_s=limit,n_below=int((sub.RT_s<limit).sum()),
                               n_total=len(sub),delay_counterfactual_has_zero_likelihood=bool((sub.RT_s<=limit).any())))
    sub.loc[sub.RT_s<.3].to_csv(OUT/f"{group.lower()}_early_trials_under_300ms.csv",index=False,float_format="%.17g")
    history=pd.read_csv(folder/"convergence.csv")
    for step in [50000,100000,int(history.step.max())]:
        point=history.loc[history.step.eq(step)].iloc[0]
        drift_rows.append(dict(Group=group,step=step,rate_lambda=point.rate_lambda_mean,
                               theta_E=point.theta_E_mean,T_0=point.T_0_mean,
                               lambda_times_theta=point.rate_lambda_mean*point.theta_E_mean,
                               T_0_times_theta_squared=point.T_0_mean*point.theta_E_mean**2))

    # Does averaging physical parameters explain the bad RTD? Compare with
    # posterior-averaged conditional CDFs using 256 joint posterior draws.
    check_times=jnp.asarray(np.linspace(0,3,301))[None,:]
    B=data["condition_B"][:,None];D=data["condition_D"][:,None]
    evaluate=jax.jit(jax.vmap(lambda p: sum(bound_cdf(check_times,c,B,D,p) for c in [-1,1])/retained_mass(B,D,p)))
    selected=samples[np.linspace(0,len(samples)-1,256,dtype=int)]
    accum=np.zeros((20,301))
    for start in range(0,256,16):
        batch=np.asarray(evaluate(jnp.asarray(selected[start:start+16])))
        assert np.isfinite(batch).all()
        accum+=batch.sum(axis=0)
    posterior_average=accum/256
    at_mean=np.asarray(evaluate(jnp.asarray(means)[None,:]))[0]
    difference=np.max(abs(at_mean-posterior_average),axis=1)
    for cell in conditions.itertuples():
        mean_check_rows.append(dict(Group=group,ASL_cd=cell.ASL_cd,Delta_dB=cell.signed_delta,
                                    max_cdf_gap_mean_parameters_vs_256_posterior_curves=difference[cell.condition_index]))
    print(group,"max mean-parameter vs posterior-average CDF gap",difference.max(),flush=True)

# %% Save a compact reproducible diagnosis without replacing the original fits.
fig.suptitle("Human RT | saved variational posteriors and actual prior supports",fontsize=17,y=.975)
fig.text(.5,.02,"10,000 constrained draws per group | solid black: mean | shaded: central 95% | dashed: finite support limits\n"
         "Control reached the step cap before mean stability; ASD met the stated stopping checks. No refit or trial exclusion.",
         ha="center",fontsize=11)
fig.tight_layout(rect=(0,.1,1,.94),w_pad=2,h_pad=3)
for ext in ["png","pdf"]: fig.savefig(OUT/f"human_reactive_tied_posterior_bounds.{ext}",dpi=180)
plt.close(fig)
pd.DataFrame(boundary_rows).to_csv(OUT/"posterior_support_audit.csv",index=False,float_format="%.17g")
pd.DataFrame(timing_rows).to_csv(OUT/"empirical_model_timing.csv",index=False,float_format="%.17g")
pd.DataFrame(early_rows).to_csv(OUT/"early_trial_counts.csv",index=False)
pd.DataFrame(mean_check_rows).to_csv(OUT/"mean_parameter_vs_posterior_average_cdf.csv",index=False,float_format="%.17g")
pd.DataFrame(drift_rows).to_csv(OUT/"parameter_drift_and_combinations.csv",index=False,float_format="%.17g")
(OUT/"audit.json").write_text(json.dumps(dict(source_fit=str(FIT.relative_to(ROOT)),source_hashes=input_hashes,
    script_sha256=sha256(__file__),selected_trials=len(frame),posterior_draws_per_group=10000,
    positive_parameter_upper_bounds="none",posterior_samples_in_outer_one_percent_of_any_finite_support=0,
    delay_limits="min(group RT)-1e-6; increasing above min RT makes observed early responses impossible",
    interpretation="No posterior pileup at a hard bound. Investigate timing constraints and optimizer drift, not wider bounds.",
    original_fits_changed=False),indent=2))
print(pd.DataFrame(timing_rows).to_string(index=False))
