# %%
"""Audit exact continuation, strict post-best patience, and same-data fit comparisons."""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from human_reactive_tied_svi_utils import ROOT, PARAM_NAMES, sha256

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--original-dir",type=Path,default=ROOT/"ASD_Human_Data/reactive_tied_rt_svi")
parser.add_argument("--continued-dir",type=Path,default=ROOT/"ASD_Human_Data/reactive_tied_rt_svi_confirm25")
args=parser.parse_args()
old,new=args.original_dir,args.continued_dir
config=json.loads((new/"run_config.json").read_text())
original_config=json.loads((old/"run_config.json").read_text())
for key in ["seed","check_every","parameter_stability_sd","learning_rate","gradient_clip","guide",
            "guide_init_scale","elbo_particles","posterior_draws","K_max","initial","parameter_names","priors","source"]:
    assert config[key]==original_config[key],key
assert config["minimum_relative_improvement"]==0.
assert config["confirmation_windows"]==25
reports={}
for group in ["Control","ASD"]:
    folder=new/group.lower();source=old/group.lower()
    resume=json.loads((folder/"resume_manifest.json").read_text())
    report=json.loads((folder/"convergence_summary.json").read_text())
    assert sha256(source/"latest_checkpoint.pkl")==resume["source_checkpoint_sha256"]
    assert sha256(source/"best_checkpoint.pkl")==resume["source_best_checkpoint_sha256"]
    for name in ["selected_trials.csv","conditions.csv","data_manifest.json"]:
        assert sha256(folder/name)==sha256(source/name),name
    before=pd.read_csv(source/"convergence.csv",float_precision="round_trip")
    after=pd.read_csv(folder/"convergence.csv",float_precision="round_trip")
    prefix=after.iloc[:len(before)]
    assert np.array_equal(before.step,prefix.step)
    for name in ["mean_negative_elbo"]+[n+suffix for n in PARAM_NAMES for suffix in ["_mean","_sd"]]:
        np.testing.assert_allclose(before[name],prefix[name],rtol=1e-12,atol=1e-14)
    previous_losses=np.load(source/"losses.npy")
    losses=np.load(folder/"losses.npy")
    np.testing.assert_array_equal(previous_losses,losses[:len(previous_losses)])
    assert np.isfinite(losses).all() and len(losses)==report["completed_steps"]
    np.testing.assert_allclose(after.mean_negative_elbo,losses.reshape(-1,1000).mean(axis=1),rtol=1e-12,atol=1e-10)
    # Independently reconstruct the corrected counter, including tiny improvements.
    best=np.inf;best_step=0;count=0;small_new_minima=0
    for row in after.itertuples():
        better=row.mean_negative_elbo<best
        relative_gain=(best-row.mean_negative_elbo)/max(1.,abs(best)) if np.isfinite(best) else np.inf
        if better:
            if 0<relative_gain<.001:small_new_minima+=1
            best=float(row.mean_negative_elbo);best_step=int(row.step);count=0
        else:count+=1
        assert int(row.patience)==count,(group,row.step,row.patience,count)
        assert int(row.best_step)==best_step
        assert bool(row.updated_best)==better
    assert best_step==report["restored_best_step"]
    assert count==report["confirmation_windows"]
    assert report["updates_after_restored_best"]==report["completed_steps"]-best_step==count*1000
    assert bool(count>=25)==report["minimum_followup_confirmed"]
    if not report["flagged"]:
        assert count>=25 and report["terminal_parameter_stable"] and report["restored_checkpoint_parameter_stable"]
    assert small_new_minima>0
    for checkpoint,flag in [(report["completed_steps"],report["terminal_parameter_stable"]),
                            (best_step,report["restored_checkpoint_parameter_stable"])]:
        points=after.loc[after.step.le(checkpoint)].tail(3)
        changes=np.ptp(points[[n+"_mean" for n in PARAM_NAMES]],axis=0)/points[[n+"_sd" for n in PARAM_NAMES]].min(axis=0).to_numpy()
        assert bool(np.all(changes<.1))==flag
    posterior=np.load(folder/"posterior_samples.npz")
    matrix=np.column_stack([posterior[n] for n in PARAM_NAMES])
    np.testing.assert_allclose(pd.read_csv(folder/"posterior_covariance.csv",index_col=0),
                               np.cov(matrix,rowvar=False),atol=1e-13,rtol=1e-10)
    reports[group]=dict(source_step=resume["source_step"],source_optimizer_iteration=resume["optimizer_iteration"],
        final_step=report["completed_steps"],best_step=best_step,post_best_windows=count,
        small_new_minima_that_correctly_reset_patience=small_new_minima,
        minimum_followup_confirmed=report["minimum_followup_confirmed"],
        original_loss_prefix_bit_identical=True,selected_trial_files_identical=True,
        posterior_covariance_checked=True)

# %% Independent arithmetic for all old-versus-new numerical comparisons.
comparison=new/"comparison"
table=pd.read_csv(comparison/"fit_quality_changes.csv")
for row in table.itertuples():
    original=pd.read_csv(old/"fit_quality.csv").set_index("Group").loc[row.Group,row.metric]
    continued=pd.read_csv(new/"fit_quality.csv").set_index("Group").loc[row.Group,row.metric]
    np.testing.assert_allclose([row.original,row.continued,row.continued_minus_original],
                               [original,continued,continued-original],rtol=1e-11,atol=1e-12)
elbo=pd.read_csv(comparison/"common_draw_elbo.csv")
np.testing.assert_allclose(elbo.paired_difference,elbo.continued_negative_elbo-elbo.original_negative_elbo,rtol=1e-10,atol=1e-9)
saved=pd.read_csv(comparison/"common_draw_elbo_summary.csv").set_index("Group")
for group,rows in elbo.groupby("Group"):
    assert len(rows)==256
    np.testing.assert_allclose(saved.loc[group,"paired_mean"],rows.paired_difference.mean(),rtol=1e-11,atol=1e-11)
    np.testing.assert_allclose(saved.loc[group,"paired_mc_sem"],rows.paired_difference.std(ddof=1)/16,rtol=1e-11,atol=1e-11)
baseline_path=new/"original_fit_hashes_before.json"
baseline=json.loads(baseline_path.read_text())
assert all(sha256(ROOT/name)==digest for name,digest in baseline.items())
audit=dict(status="passed",groups=reports,original_fit_artifacts_preserved=len(baseline),
           same_data_and_model=True,strict_patience_replayed=True,raw_updates_in_confirmation_window=1000,
           required_windows=25,comparison_values_verified=True,source_sha256=sha256(__file__))
(new/"continuation_validation.json").write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2))
