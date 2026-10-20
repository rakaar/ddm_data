# %%
"""Resume human TIED SVI; confirm a best loss window with strict post-best patience."""
import os
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/human_tied_mpl")
from pathlib import Path
import argparse
import json
import pickle
import shutil
import time
from functools import partial

import numpy as np
import pandas as pd
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

# %% Configuration: same data/model/optimizer, corrected stopping, no new start.
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source-dir",type=Path,default=ROOT/"ASD_Human_Data/reactive_tied_rt_svi")
parser.add_argument("--output-dir",type=Path,default=ROOT/"ASD_Human_Data/reactive_tied_rt_svi_confirm25")
parser.add_argument("--group",choices=["all","Control","ASD"],default="all")
parser.add_argument("--max-total-steps",type=int,default=1000000)
parser.add_argument("--confirmation-windows",type=int,default=25)
parser.add_argument("--resume-existing",action="store_true",help="Continue an interrupted run in the output directory.")
args=parser.parse_args()
assert args.source_dir.resolve()!=args.output_dir.resolve()
assert args.max_total_steps%1000==0 and args.confirmation_windows>0
source_config=json.loads((args.source_dir/"run_config.json").read_text())
for path,digest in source_config["source_hashes"].items(): assert sha256(ROOT/path)==digest
frame,provenance=load_trials()
args.output_dir.mkdir(parents=True,exist_ok=True)
config=dict(source_config)
config.update(max_steps=args.max_total_steps,patience=args.confirmation_windows,minimum_relative_improvement=0.,
              stopping_rule="strict post-best patience: any lower 1000-step mean resets counter; require terminal and restored mean stability",
              confirmation_windows=args.confirmation_windows,source_fit=str(args.source_dir),
              statistical_initializations_per_group=1,continuation_preserves_optimizer_and_rng=True)
config["source_hashes"]=dict(source_config["source_hashes"])
config["source_hashes"][str(Path(__file__).resolve().relative_to(ROOT))]=sha256(__file__)
(args.output_dir/"run_config.json").write_text(json.dumps(config,indent=2))
print("CPU:",jax.devices(),"source:",args.source_dir,"output:",args.output_dir,flush=True)

# %% Sequential continuation using the saved Adam moments, iteration and RNG state.
for group in (["Control","ASD"] if args.group=="all" else [args.group]):
    original=args.source_dir/group.lower()
    folder=args.output_dir/group.lower()
    folder.mkdir(exist_ok=True)
    if (folder/"posterior_samples.npz").exists():
        if args.resume_existing:
            print(group,"already completed; preserving its saved result",flush=True)
            continue
        raise FileExistsError(f"Completed continuation exists: {folder}")
    sub,conditions,data,upper=group_data(frame,group)
    continuing=args.resume_existing and (folder/"latest_checkpoint.pkl").exists()
    starting=folder if continuing else original
    with (starting/"latest_checkpoint.pkl").open("rb") as stream: current=pickle.load(stream)
    with (starting/"best_checkpoint.pkl").open("rb") as stream: best=pickle.load(stream)
    source_steps=int(current["step"])
    assert args.max_total_steps>source_steps
    model=partial(reactive_model,K_max=40)
    guide=AutoMultivariateNormal(model,init_loc_fn=init_to_value(values=INITIAL),init_scale=.1)
    svi=SVI(model,guide,ClippedAdam(.0002,clip_norm=1.),Trace_ELBO(num_particles=1))
    # Initialize transforms/program structure only; discard this fresh optimizer state.
    svi.init(random.PRNGKey(0),data,upper)
    state=jax.tree.map(jax.device_put,current["state"])
    best_state=jax.tree.map(jax.device_put,best["state"])
    best_params=svi.get_params(best_state)
    for name,value in svi.get_params(state).items():
        np.testing.assert_array_equal(np.asarray(value),np.asarray(current["params"][name]))
    assert int(state.optim_state[0])==source_steps
    history=pd.read_csv(starting/"convergence.csv")
    assert source_steps==int(history.step.iloc[-1])
    losses=np.load(starting/"losses.npy").tolist()
    assert len(losses)==source_steps
    best_loss=float(history.mean_negative_elbo.min())
    best_step=int(best["step"])
    assert best_step==int(history.loc[history.mean_negative_elbo.idxmin(),"step"])
    if not continuing:
        history["legacy_patience"]=history.patience
        lowest=np.inf;since_best=0;strict=[]
        for loss in history.mean_negative_elbo:
            if loss<lowest: lowest=loss;since_best=0
            else: since_best+=1
            strict.append(since_best)
        history["patience"]=strict
        for name in ["selected_trials.csv","conditions.csv","data_manifest.json"]:
            shutil.copy2(original/name,folder/name)
        shutil.copy2(original/"best_checkpoint.pkl",folder/"best_checkpoint.pkl")
        resume_manifest=dict(group=group,source_checkpoint=str(original/"latest_checkpoint.pkl"),
                             source_checkpoint_sha256=sha256(original/"latest_checkpoint.pkl"),
                             source_best_checkpoint_sha256=sha256(original/"best_checkpoint.pkl"),
                             source_convergence_sha256=sha256(original/"convergence.csv"),
                             source_loss_sha256=sha256(original/"losses.npy"),source_step=source_steps,
                             optimizer_iteration=source_steps,delay_upper_s=upper,
                             rng_state_preserved=True,guide_parameters_exactly_restored=True)
        (folder/"resume_manifest.json").write_text(json.dumps(resume_manifest,indent=2))
    rows=history.to_dict("records")
    mean_history=history[[n+"_mean" for n in PARAM_NAMES]].to_numpy().tolist()
    sd_history=history[[n+"_sd" for n in PARAM_NAMES]].to_numpy().tolist()
    patience=int(history.patience.iloc[-1])
    started=time.monotonic()
    prior_elapsed=float(history.elapsed_s.iloc[-1])
    diagnostic_draws=jax.jit(lambda p:guide.sample_posterior(random.PRNGKey(1729),p,sample_shape=(4096,)))
    print(f"\n{group}: resuming step {source_steps:,}; current best {best_step:,}; wait {args.confirmation_windows} windows",flush=True)
    stop_reason="maximum_steps_without_confirmation"
    for completed in range(source_steps+1000,args.max_total_steps+1,1000):
        result=svi.run(random.PRNGKey(0),1000,data,upper,progress_bar=False,init_state=state,stable_update=True)
        state=result.state
        chunk=np.asarray(result.losses)
        assert np.isfinite(chunk).all(),f"Nonfinite loss at {completed}"
        losses.extend(chunk.tolist())
        loss=float(chunk.mean())
        improved=loss<best_loss
        if improved:
            best_loss,best_step,best_state,best_params=loss,completed,state,result.params
            patience=0
            with (folder/"best_checkpoint.pkl").open("wb") as stream:
                pickle.dump(jax.device_get(dict(state=state,params=result.params,step=completed)),stream)
        else:
            patience+=1
        draws=diagnostic_draws(result.params)
        means=np.array([np.asarray(draws[n]).mean() for n in PARAM_NAMES])
        sds=np.array([np.asarray(draws[n]).std(ddof=1) for n in PARAM_NAMES])
        mean_history.append(means.tolist());sd_history.append(sds.tolist())
        changes=np.ptp(mean_history[-3:],axis=0)/np.min(sd_history[-3:],axis=0)
        stable=bool(np.all(changes<.1))
        row=dict(step=completed,mean_negative_elbo=loss,best_loss=best_loss,best_step=best_step,
                 patience=patience,updated_best=improved,parameter_stable=stable,
                 maximum_mean_change_sd=float(changes.max()),elapsed_s=prior_elapsed+time.monotonic()-started)
        for i,name in enumerate(PARAM_NAMES):
            row[name+"_mean"]=means[i];row[name+"_sd"]=sds[i];row[name+"_change_sd"]=changes[i]
        rows.append(row)
        pd.DataFrame(rows).to_csv(folder/"convergence.csv",index=False)
        np.save(folder/"losses.npy",np.asarray(losses))
        with (folder/"latest_checkpoint.pkl").open("wb") as stream:
            pickle.dump(jax.device_get(dict(state=state,params=result.params,step=completed)),stream)
        restored_stable=bool(rows[best_step//1000-1]["parameter_stable"])
        print(f"{group} {completed:7,d}: loss {loss:.4f}, best {best_step:,} ({best_loss:.4f}), "
              f"post-best {patience}/{args.confirmation_windows}, drift {changes.max():.3f} SD, "
              f"elapsed +{time.monotonic()-started:.1f}s",flush=True)
        if patience>=args.confirmation_windows and stable and restored_stable:
            stop_reason="confirmed_minimum_25_windows_and_parameter_stability"
            break
    samples=guide.sample_posterior(random.PRNGKey(100000),best_params,sample_shape=(10000,))
    matrix=np.column_stack([np.asarray(samples[n]) for n in PARAM_NAMES])
    assert np.isfinite(matrix).all()
    np.savez_compressed(folder/"posterior_samples.npz",**{n:np.asarray(samples[n]) for n in PARAM_NAMES})
    np.savez_compressed(folder/"guide_parameters.npz",**{n:np.asarray(value) for n,value in best_params.items()})
    summary=pd.DataFrame(dict(parameter=PARAM_NAMES,mean=matrix.mean(axis=0),sd=matrix.std(axis=0,ddof=1),
                              q025=np.quantile(matrix,.025,axis=0),median=np.median(matrix,axis=0),q975=np.quantile(matrix,.975,axis=0)))
    summary.to_csv(folder/"posterior_summary.csv",index=False,float_format="%.17g")
    pd.DataFrame(np.cov(matrix,rowvar=False),index=PARAM_NAMES,columns=PARAM_NAMES).to_csv(folder/"posterior_covariance.csv")
    pd.DataFrame(np.corrcoef(matrix,rowvar=False),index=PARAM_NAMES,columns=PARAM_NAMES).to_csv(folder/"posterior_correlation.csv")
    confirmed=bool(patience>=args.confirmation_windows)
    report=dict(stop_reason=stop_reason,completed_steps=completed,restored_best_step=best_step,
                best_window_mean_negative_elbo=best_loss,terminal_parameter_stable=stable,
                restored_checkpoint_parameter_stable=restored_stable,minimum_followup_confirmed=confirmed,
                confirmation_windows=patience,required_confirmation_windows=args.confirmation_windows,
                updates_after_restored_best=completed-best_step,
                flagged=not(confirmed and stable and restored_stable),wall_seconds=time.monotonic()-started,
                source_step=source_steps,new_updates=completed-source_steps,
                posterior_kind="full-rank Gaussian variational approximation in unconstrained coordinates",
                warning="Observed-window confirmation is not proof of a global optimum; pooled-trial uncertainty excludes participant variability")
    (folder/"convergence_summary.json").write_text(json.dumps(report,indent=2))
    print(group,"FINISHED",report,flush=True)
