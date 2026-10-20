# %%
"""Compare original and strictly confirmed human TIED SVI fits on the same RT rows."""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/human_tied_mpl")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
from pathlib import Path
import argparse
import json
from functools import partial

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import jax
import jax.numpy as jnp
from jax import random
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoMultivariateNormal
from numpyro.infer.initialization import init_to_value
from numpyro.optim import ClippedAdam
from human_reactive_tied_svi_utils import (
    ROOT, PARAM_NAMES, INITIAL, load_trials, group_data, reactive_model, sha256,
)

# %% Inputs: all physical-mean predictions retain 0–3 s conditional normalization.
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--original-dir",type=Path,default=ROOT/"ASD_Human_Data/reactive_tied_rt_svi")
parser.add_argument("--continued-dir",type=Path,default=ROOT/"ASD_Human_Data/reactive_tied_rt_svi_confirm25")
args=parser.parse_args()
old,new=args.original_dir,args.continued_dir
out=new/"comparison";out.mkdir(exist_ok=True)
frame,provenance=load_trials()
colors=["#1f77b4","#ff7f0e"]
asls=[20,40]
quality_rows,parameter_rows,condition_rows,elbo_rows=[],[],[],[]
payload={}
plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})

# %% Compare the same counts, joint mean-parameter likelihoods, and guide ELBOs.
for group in ["Control","ASD"]:
    oq=pd.read_csv(old/"fit_quality.csv").query("Group == @group").iloc[0]
    nq=pd.read_csv(new/"fit_quality.csv").query("Group == @group").iloc[0]
    report=json.loads((new/group.lower()/"convergence_summary.json").read_text())
    for name in ["mean_parameter_joint_log_likelihood","psychometric_rmse","psychometric_max_abs_error",
                 "mean_condition_cdf_error","max_condition_cdf_error"]:
        quality_rows.append(dict(Group=group,metric=name,original=float(oq[name]),continued=float(nq[name]),
                                 continued_minus_original=float(nq[name]-oq[name])))
    osummary=pd.read_csv(old/group.lower()/"posterior_summary.csv").set_index("parameter")
    nsummary=pd.read_csv(new/group.lower()/"posterior_summary.csv").set_index("parameter")
    for name in PARAM_NAMES:
        parameter_rows.append(dict(Group=group,parameter=name,original_mean=osummary.loc[name,"mean"],
                                   continued_mean=nsummary.loc[name,"mean"],original_sd=osummary.loc[name,"sd"],
                                   continued_sd=nsummary.loc[name,"sd"],
                                   mean_change_in_original_sd=(nsummary.loc[name,"mean"]-osummary.loc[name,"mean"])/osummary.loc[name,"sd"]))
    oc=pd.read_csv(old/"condition_cdf_discrepancies.csv",float_precision="round_trip").query("Group == @group")
    nc=pd.read_csv(new/"condition_cdf_discrepancies.csv",float_precision="round_trip").query("Group == @group")
    merged=oc.merge(nc,on=["Group","ASL_cd","Delta_dB"],suffixes=("_original","_continued"),validate="one_to_one")
    merged["cdf_gap_change"]=merged.cdf_max_abs_error_continued-merged.cdf_max_abs_error_original
    condition_rows.extend(merged.to_dict("records"))
    sub,conditions,data,upper=group_data(frame,group)
    orig_curves=np.load(old/group.lower()/"mean_parameter_curves.npz")
    new_curves=np.load(new/group.lower()/"mean_parameter_curves.npz")
    np.testing.assert_array_equal(orig_curves["time_s"],new_curves["time_s"])
    payload[group]=(sub,conditions,orig_curves,new_curves,report)
    # Independent MC evaluation with common random draws checks whether the saved
    # guide improves expected negative ELBO, beyond stochastic window selection.
    model=partial(reactive_model,K_max=40)
    guide=AutoMultivariateNormal(model,init_loc_fn=init_to_value(values=INITIAL),init_scale=.1)
    svi=SVI(model,guide,ClippedAdam(.0002,clip_norm=1.),Trace_ELBO(num_particles=1))
    svi.init(random.PRNGKey(0),data,upper)
    score=jax.jit(jax.vmap(lambda key,p:Trace_ELBO(num_particles=1).loss(key,p,model,guide,data,upper),in_axes=(0,None)))
    keys=random.split(random.PRNGKey(314159),256)
    values={}
    for label,folder in [("original",old),("continued",new)]:
        saved=np.load(folder/group.lower()/"guide_parameters.npz")
        params={name:jnp.asarray(saved[name]) for name in saved.files}
        result=[]
        for start in range(0,256,8):
            result.extend(np.asarray(score(keys[start:start+8],params)).tolist())
        values[label]=np.asarray(result)
        assert np.isfinite(values[label]).all()
    for i in range(256):
        elbo_rows.append(dict(Group=group,draw=i,original_negative_elbo=values["original"][i],
                              continued_negative_elbo=values["continued"][i],
                              paired_difference=values["continued"][i]-values["original"][i]))
    print(group,"common-draw negative ELBO: original",values["original"].mean(),
          "continued",values["continued"].mean(),"paired change",(values["continued"]-values["original"]).mean(),flush=True)

pd.DataFrame(quality_rows).to_csv(out/"fit_quality_changes.csv",index=False,float_format="%.17g")
pd.DataFrame(parameter_rows).to_csv(out/"parameter_changes.csv",index=False,float_format="%.17g")
pd.DataFrame(condition_rows).to_csv(out/"condition_cdf_changes.csv",index=False,float_format="%.17g")
elbo=pd.DataFrame(elbo_rows)
elbo.to_csv(out/"common_draw_elbo.csv",index=False,float_format="%.17g")
elbo_summary=elbo.groupby("Group").agg(original_mean=("original_negative_elbo","mean"),
    continued_mean=("continued_negative_elbo","mean"),paired_mean=("paired_difference","mean"),
    paired_sd=("paired_difference","std"),n=("draw","size")).reset_index()
elbo_summary["paired_mc_sem"]=elbo_summary.paired_sd/np.sqrt(elbo_summary.n)
elbo_summary.to_csv(out/"common_draw_elbo_summary.csv",index=False,float_format="%.17g")

# %% Data and both model versions; same scales, ASL colors and normalization.
legend=[Line2D([0],[0],color=color,label=f"ASL {asl} cd/m²") for color,asl in zip(colors,asls)]
legend += [Line2D([0],[0],color=".2",lw=2,label="Continued mean parameters"),
           Line2D([0],[0],color=".2",lw=2,ls=":",label="Original mean parameters"),
           Line2D([0],[0],color=".5",lw=1,drawstyle="steps-mid",label="Data RTD")]
footer=("RT only | 6,600 Control / 4,800 ASD trials | same five shared parameters, priors, and joint RT/choice likelihood\n"
        "Arithmetic physical posterior means | RTDs conditional on 0–3 s; zoom retains normalization | blue/orange = ASL 20/40 cd/m²")
psych=pd.read_csv(new/"psychometric_comparison.csv")
smooth_old=pd.read_csv(old/"psychometric_model_curve.csv")
smooth_new=pd.read_csv(new/"psychometric_model_curve.csv")
edges=np.linspace(0,3,151)
for xmax,name in [(1.5,"zoom_0_1p5s"),(3.,"full_0_3s")]:
    fig,axes=plt.subplots(2,2,figsize=(15,10))
    for row,group in enumerate(["Control","ASD"]):
        sub,conditions,oc,nc,report=payload[group]
        for color,asl in zip(colors,asls):
            points=psych.loc[psych.Group.eq(group)&psych.ASL_cd.eq(asl)]
            axes[row,0].errorbar(points.Delta_dB,points.p_right_data,yerr=points.binomial_sem,fmt="o",color=color,ms=4,capsize=2)
            for curves,style in [(smooth_old,":"),(smooth_new,"-")]:
                q=curves.loc[curves.Group.eq(group)&curves.ASL_cd.eq(asl)]
                axes[row,0].plot(q.Delta_dB,q.p_right_model,color=color,ls=style,lw=2)
            selected=sub.loc[sub.ASL_cd.eq(asl)]
            hist=np.histogram(selected.RT_s,edges)[0]/len(selected)/np.diff(edges)
            axes[row,1].stairs(hist,edges,color=color,alpha=.4,lw=1)
            c=conditions.loc[conditions.ASL_cd.eq(asl)]
            idx=c.condition_index.to_numpy();weights=c.n.to_numpy()/c.n.sum()
            for curves,style in [(oc,":"),(nc,"-")]:
                axes[row,1].plot(curves["time_s"],np.average(curves["density"][idx],axis=0,weights=weights),color=color,ls=style,lw=2)
        status="25-window minimum confirmed" if not report["flagged"] else "convergence unconfirmed"
        axes[row,0].set(title=f"{group} | {status}",xlabel="Signed ΔdB (right − left)",ylabel="P(right)",ylim=(-.02,1.02))
        axes[row,1].set(title=f"{group} | original and continued RTD",xlabel="Recorded RT (s)",ylabel="Density (s⁻¹)",xlim=(0,xmax))
        for ax in axes[row]:ax.grid(alpha=.15)
    ymax=max(ax.get_ylim()[1] for ax in axes[:,1])
    for ax in axes[:,1]:ax.set_ylim(0,ymax)
    fig.suptitle("Human reactive-only TIED | does strict minimum confirmation improve fit quality?",fontsize=16,y=.98)
    fig.legend(handles=legend,loc="upper center",bbox_to_anchor=(.5,.95),ncol=3,frameon=False)
    fig.text(.5,.025,footer,ha="center",fontsize=10)
    fig.tight_layout(rect=(0,.1,1,.89),h_pad=3)
    for ext in ["png","pdf"]:fig.savefig(out/f"human_tied_original_vs_continued_{name}.{ext}",dpi=180)
    plt.close(fig)

# %% Show actual post-best waiting, with the original endpoint marked explicitly.
fig,axes=plt.subplots(2,3,figsize=(18,9))
for row,group in enumerate(["Control","ASD"]):
    hist=pd.read_csv(new/group.lower()/"convergence.csv")
    source=json.loads((new/group.lower()/"resume_manifest.json").read_text())
    report=payload[group][4]
    color="#1f77b4" if group=="Control" else "#d62728"
    for col in [0,1]:
        axes[row,col].plot(hist.step,hist.mean_negative_elbo,color=color,lw=1.4)
        axes[row,col].axvline(report["restored_best_step"],color="k",ls=":")
        axes[row,col].axvline(source["source_step"],color=".5",ls="--")
        axes[row,col].set_ylabel("1,000-step mean negative ELBO")
    tail=hist.loc[hist.step.ge(max(source["source_step"],report["completed_steps"]-100000))]
    axes[row,1].set_xlim(tail.step.min(),tail.step.max())
    ymin,ymax=tail.mean_negative_elbo.min(),tail.mean_negative_elbo.max()
    axes[row,1].set_ylim(ymin-.1*(ymax-ymin+.1),ymax+.1*(ymax-ymin+.1))
    axes[row,2].plot(hist.step,hist.patience,color=color)
    axes[row,2].axhline(25,color="k",ls=":")
    axes[row,2].set_ylabel("Windows since lower best loss")
    axes[row,2].axvline(source["source_step"],color=".5",ls="--")
    for col,title in enumerate(["Full loss history","Final 100,000 updates","Strict post-best counter"]):
        axes[row,col].set_title(f"{group} | {title}")
        axes[row,col].set_xlabel("Total SVI updates (thousands)")
        axes[row,col].xaxis.set_major_formatter(FuncFormatter(lambda value,pos:f"{value/1000:g}"))
        axes[row,col].grid(alpha=.15)
fig.suptitle("Human TIED continuation | every lower window mean resets confirmation patience",fontsize=17,y=.98)
fig.text(.5,.02,"Gray dashed: original endpoint | black dotted: selected best loss checkpoint / required 25 follow-up windows\n"
         "Checkpoint selection uses stochastic 1,000-step means; this establishes the specified follow-up, not a global optimum.",ha="center",fontsize=11)
fig.tight_layout(rect=(0,.1,1,.93),h_pad=3,w_pad=3)
for ext in ["png","pdf"]:fig.savefig(out/f"human_tied_continuation_confirmation.{ext}",dpi=170)
plt.close(fig)
(out/"manifest.json").write_text(json.dumps(dict(original_root=str(old),continued_root=str(new),
    source_csv_sha256=provenance["csv_sha256"],script_sha256=sha256(__file__),
    objective_evaluation="256 common-random-number draws per guide, no refitting",
    figure_sha256={p.name:sha256(p) for p in out.glob("*.png")}),indent=2))
print(pd.DataFrame(quality_rows).to_string(index=False))
