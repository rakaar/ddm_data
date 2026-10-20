# %%
"""Independent source/MAT, first-passage and strict-stopping audit for RT <= 1.5 s."""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/human_tied_mpl")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
from pathlib import Path
from io import BytesIO
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.io.matlab._mio5 import MatFile5Reader
from scipy.special import log_ndtr
from scipy.integrate import quad
from PIL import Image
import fitz
import jax
import jax.numpy as jnp
from numpyro import handlers
from human_reactive_tied_window_utils import (
    ROOT, PARAM_NAMES, INITIAL, group_data, log_likelihood, bound_cdf, bound_density,
    retained_mass, reactive_model, sha256,
)


# %% Independent implementations: image CDF uses log-normal tails; density switches
# to the eigenfunction series at long dimensionless times to avoid cancellation.
def scipy_first_passage(rt, choice, B, D, params, cumulative=False):
    rate,t0,theta,w,delay = params
    t = np.asarray(rt)-delay
    gamma = theta*np.tanh(rate*np.asarray(D)/17.37)
    omega = 10**(rate*np.asarray(B)/20)*np.cosh(rate*np.asarray(D)/17.37)/(t0*theta**2)
    v = np.where(np.asarray(choice)==1, -gamma, gamma)
    wb = np.where(np.asarray(choice)==1, 1-w, w)
    tt, v, wb, om = np.broadcast_arrays(omega*np.maximum(t,1e-14),v,wb,omega)
    result = np.zeros_like(tt)
    if cumulative:
        k = np.arange(81)
        r = np.where(k%2==0,2*k+2*wb[...,None],2*k+2*(1-wb[...,None]))
        root = np.sqrt(tt)[...,None]
        vf = v[...,None]
        pref = -2*vf*wb[...,None]
        terms = np.exp(pref-vf*r+log_ndtr((vf*tt[...,None]-r)/root))
        terms += np.exp(pref+vf*r+log_ndtr((-vf*tt[...,None]-r)/root))
        result = np.sum((-1.)**k*terms,axis=-1)
    else:
        small = tt < .6
        if small.any():
            ts,vs,ws = tt[small],v[small],wb[small]
            k = np.arange(-40,41)
            paths = ws[:,None]+2*k
            image_sum = np.sum(paths*np.exp(-2*paths**2/ts[:,None]),axis=1)
            result[small] = 2/np.sqrt(2*np.pi*ts**3)*np.exp(-2*vs*ws-vs**2*ts/2)*image_sum
        if (~small).any():
            tl,vl,wl = tt[~small],v[~small],wb[~small]
            n = np.arange(1,81)
            spectral_sum = np.sum(n*np.sin(n*np.pi*wl[:,None])*np.exp(-n*n*np.pi**2*tl[:,None]/8),axis=1)
            result[~small] = np.pi/4*np.exp(-2*vl*wl-vl**2*tl/2)*spectral_sum
        result *= om
    return np.where(t>0,result,0.)


# %% Configuration and re-read of original MATLAB MCOS rows without rewriting exports.
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir",type=Path,default=ROOT/"ASD_Human_Data/reactive_tied_rt_svi_rt_le_1p5s")
args=parser.parse_args()
out=args.output_dir
human=ROOT/"ASD_Human_Data"
extraction=json.loads((human/"simultaneous_rt_extraction.json").read_text())
csv=human/"simultaneous_rt_trials.csv"
source=human/extraction["source"]
assert sha256(csv)==extraction["output_sha256"] and sha256(source)==extraction["source_sha256"]
frame=pd.read_csv(csv)
raw=loadmat(source)["__function_workspace__"].tobytes()
stream=BytesIO(b"MATLAB 5.0 MAT-file".ljust(116,b" ")+b"\0"*8+raw[:4]+raw[8:])
reader=MatFile5Reader(stream);reader.initialize_read();reader.read_file_header()
header,_=reader.read_var_header()
cells=list(reader.read_var_array(header)["MCOS"][0,0][0]["arr"].flat)
names=[x.item() for x in cells[10].flat]
arrays=dict(zip(names,[np.asarray(x).ravel() for x in cells[5].flat]))
n_rows=int(cells[7].item())
encoded=cells[4].ravel()[4+n_rows:].tobytes().decode("utf-16le")
condition=np.array([encoded[i:i+2] for i in range(0,len(encoded),2)])
groups=[x.item() for x in cells[2].flat]
original_groups=np.array([groups[int(c)-1] for c in cells[3].ravel()])
assert np.array_equal(frame.source_row.to_numpy()-1,np.flatnonzero(condition=="RT"))
original_frame=frame.copy()
frame=frame.loc[frame.RT_s.le(1.5)].copy()
identities=frame.source_row.to_numpy()-1
assert np.array_equal(identities,np.flatnonzero((condition=="RT") & (arrays["RT_s"]<=1.5)))
assert np.array_equal(original_groups[identities],frame.Group.to_numpy())
for name in frame.columns:
    if name in arrays and name not in ["Group","Condition"]:
        np.testing.assert_allclose(frame[name],arrays[name][identities],rtol=0,atol=1e-12,equal_nan=True)
assert len(frame)==11235 and frame.source_row.is_unique
assert frame.Condition.eq("RT").all() and frame.Valid.eq(1).all()
assert np.array_equal(frame.Choice.eq(1).astype(int),frame.ChoseRight)
assert frame.Stim_duration_s.isna().all() and frame.RT_s.between(0,1.5,inclusive="both").all()
assert np.allclose(frame.Mean_dB,(frame.Stim_left_dB+frame.Stim_right_dB)/2,rtol=0,atol=1e-12)
selected_export=pd.read_csv(out/'selected_trials.csv')
excluded_export=pd.read_csv(out/'excluded_long_trials.csv')
assert np.array_equal(selected_export.source_row,frame.source_row)
assert np.array_equal(excluded_export.source_row,original_frame.loc[original_frame.RT_s.gt(1.5),'source_row'])
assert len(excluded_export)==165 and excluded_export.RT_s.gt(1.5).all()
assert set(selected_export.source_row).isdisjoint(excluded_export.source_row)
assert set(selected_export.source_row)|set(excluded_export.source_row)==set(original_frame.source_row)
assert frame.Participant.nunique()==19
frame["signed_delta"]=frame.Delta_dB.round(2)
psych=pd.read_csv(out/"psychometric_comparison.csv",float_precision="round_trip")
smooth_psych=pd.read_csv(out/"psychometric_model_curve.csv")
histograms=pd.read_csv(out/"rtd_histograms.csv")
quality=pd.read_csv(out/"fit_quality.csv")
cdf_table=pd.read_csv(out/"condition_cdf_discrepancies.csv",float_precision="round_trip")
config=json.loads((out/"run_config.json").read_text())
for file,digest in config["source_hashes"].items(): assert sha256(ROOT/file)==digest
assert config["parameters"]==PARAM_NAMES and len(PARAM_NAMES)==5
assert config["guide"]=="AutoMultivariateNormal" and config["seed"]==0
assert config["elbo_particles"]==1 and config["learning_rate"]==.0002
reports={}

# %% Independently reconstruct all empirical values and model weights.
for group,n_people,n_trials in [("Control",11,6531),("ASD",8,4704)]:
    folder=out/group.lower()
    sub,conditions,data,upper=group_data(frame,group)
    assert len(sub)==n_trials and sub.Participant.nunique()==n_people
    assert sub.groupby(["ASL_cd","signed_delta"]).size().sum()==n_trials
    exported=pd.read_csv(folder/"selected_trials.csv")
    data_manifest=json.loads((folder/"data_manifest.json").read_text())
    assert sha256(folder/"selected_trials.csv")==data_manifest["selected_source_rows_sha256"]
    np.testing.assert_allclose(data_manifest["delay_upper_s"],sub.RT_s.min()-1e-6,rtol=0,atol=1e-15)
    assert np.array_equal(exported.source_row,sub.source_row)
    assert np.array_equal(exported.Choice,sub.Choice)
    np.testing.assert_allclose(exported.RT_s,sub.RT_s,rtol=0,atol=1e-15)
    trace=handlers.trace(handlers.seed(reactive_model,0)).get_trace(data,upper)
    free=[k for k,v in trace.items() if v["type"]=="sample" and not v["is_observed"]]
    assert free==PARAM_NAMES
    posterior=np.load(folder/"posterior_samples.npz")
    assert set(posterior.files)==set(PARAM_NAMES)
    matrix=np.column_stack([posterior[n] for n in PARAM_NAMES])
    assert matrix.shape==(10000,5) and np.isfinite(matrix).all()
    assert (matrix[:,:3]>0).all() and ((matrix[:,3]>.05)&(matrix[:,3]<.95)).all()
    assert ((matrix[:,4]>0)&(matrix[:,4]<upper)).all()
    saved=pd.read_csv(folder/"posterior_summary.csv").set_index("parameter").loc[PARAM_NAMES]
    means=matrix.mean(axis=0)
    np.testing.assert_allclose(saved["mean"],means,rtol=1e-12)
    np.testing.assert_allclose(saved.sd,matrix.std(axis=0,ddof=1),rtol=1e-12)
    np.testing.assert_allclose(saved.q025,np.quantile(matrix,.025,axis=0),rtol=1e-12)
    np.testing.assert_allclose(saved.q975,np.quantile(matrix,.975,axis=0),rtol=1e-12)
    np.testing.assert_allclose(pd.read_csv(folder/"posterior_correlation.csv",index_col=0),np.corrcoef(matrix,rowvar=False),atol=1e-12)
    np.testing.assert_allclose(pd.read_csv(folder/'posterior_covariance.csv',index_col=0),np.cov(matrix,rowvar=False),atol=1e-10,rtol=1e-10)
    curves=np.load(folder/"mean_parameter_curves.npz")
    times=curves["time_s"]
    metrics={"K40_K80_max_density_abs":0.,"K40_K80_max_mass_abs":0.,
             "scipy_max_density_abs":0.,"scipy_max_cdf_abs":0.,"unit_mass_max_abs":0.,
             "cdf_derivative_max_abs":0.}
    # Initial, posterior mean/median and representative joint draws, not marginal combinations.
    points=[np.array(list(INITIAL.values())),means,np.median(matrix,axis=0)]
    points.extend(matrix[np.linspace(0,9999,12,dtype=int)])
    evaluate_gradient=jax.jit(jax.value_and_grad(log_likelihood))
    for point_index,p in enumerate(points):
        jp=jnp.asarray(p)
        ll,gradient=evaluate_gradient(jp,data)
        assert np.isfinite(ll) and np.isfinite(gradient).all()
        f40=np.asarray(bound_density(data["rt"],data["choice"],data["B"],data["D"],jp,40))
        f80=np.asarray(bound_density(data["rt"],data["choice"],data["B"],data["D"],jp,80))
        scipy_density=scipy_first_passage(np.asarray(data["rt"]),np.asarray(data["choice"]),
                         np.asarray(data["B"]),np.asarray(data["D"]),p)
        metrics["K40_K80_max_density_abs"]=max(metrics["K40_K80_max_density_abs"],float(np.max(abs(f40-f80))))
        metrics["scipy_max_density_abs"]=max(metrics["scipy_max_density_abs"],float(np.max(abs(f40-scipy_density))))
        assert (scipy_density>0).all()
        norm=np.asarray(retained_mass(data["condition_B"],data["condition_D"],jp,40))
        norm80=np.asarray(retained_mass(data["condition_B"],data["condition_D"],jp,80))
        metrics["K40_K80_max_mass_abs"]=max(metrics["K40_K80_max_mass_abs"],float(np.max(abs(norm-norm80))))
        assert (norm>0).all() and (norm<=1+1e-10).all()
        for row in conditions.itertuples():
            i,B,D=row.condition_index,row.Mean_dB,row.signed_delta
            check_times=np.linspace(p[4]+.001,1.5,121)
            for choice in [-1,1]:
                fc=np.asarray(bound_cdf(jnp.asarray(check_times),choice,B,D,jp))
                fi=scipy_first_passage(check_times,choice,B,D,p,cumulative=True)
                metrics["scipy_max_cdf_abs"]=max(metrics["scipy_max_cdf_abs"],float(np.max(abs(fc-fi))))
            # Quadrature and CDF derivative use initialization, mean and median.
            if point_index < 3:
                integral=sum(quad(lambda t:float(scipy_first_passage(t,c,B,D,p)),p[4],1.5,epsabs=1e-9)[0] for c in [-1,1])
                metrics["unit_mass_max_abs"]=max(metrics["unit_mass_max_abs"],abs(integral/norm[i]-1))
                tcheck=np.linspace(p[4]+.02,1.4,40)
                h=1e-5
                derivative=sum((scipy_first_passage(tcheck+h,c,B,D,p,True)-scipy_first_passage(tcheck-h,c,B,D,p,True))/(2*h) for c in [-1,1])
                dens=sum(scipy_first_passage(tcheck,c,B,D,p) for c in [-1,1])
                metrics["cdf_derivative_max_abs"]=max(metrics["cdf_derivative_max_abs"],float(np.max(abs(derivative-dens))))
    assert metrics["K40_K80_max_density_abs"]<1e-9
    assert metrics["K40_K80_max_mass_abs"]<1e-9
    assert metrics["scipy_max_density_abs"]<1e-8
    assert metrics["scipy_max_cdf_abs"]<1e-8
    assert metrics["unit_mass_max_abs"]<1e-7
    assert metrics["cdf_derivative_max_abs"]<1e-6
    unbiased=np.array(list(INITIAL.values()))
    np.testing.assert_allclose(bound_cdf(1.5,1,10.891984,.3,unbiased),
                               bound_cdf(1.5,-1,10.891984,-.3,unbiased),atol=1e-12)
    biased=unbiased.copy();biased[3]=.6
    assert float(bound_cdf(1.5,1,10.891984,0.,biased))>float(bound_cdf(1.5,-1,10.891984,0.,biased))
    independent_norm=sum(scipy_first_passage(1.5,c,conditions.Mean_dB.to_numpy(),
                               conditions.signed_delta.to_numpy(),means,True) for c in [-1,1])
    for asl,curve in smooth_psych.loc[smooth_psych.Group.eq(group)].groupby("ASL_cd"):
        norm=sum(scipy_first_passage(1.5,c,curve.Mean_dB.to_numpy(),curve.Delta_dB.to_numpy(),means,True) for c in [-1,1])
        predicted=scipy_first_passage(1.5,1,curve.Mean_dB.to_numpy(),curve.Delta_dB.to_numpy(),means,True)/norm
        np.testing.assert_allclose(curve.p_right_model,predicted,atol=1e-9,rtol=0)
    independent_ll=np.sum(np.log(scipy_first_passage(sub.RT_s,sub.Choice,sub.Mean_dB,sub.Delta_dB,means))
                           -np.log(independent_norm[sub.condition_index.to_numpy()]))
    q=quality.loc[quality.Group.eq(group)].iloc[0]
    np.testing.assert_allclose(q.mean_parameter_joint_log_likelihood,independent_ll,atol=1e-7,rtol=0)
    for row in conditions.itertuples():
        cell=sub.loc[sub.condition_index.eq(row.condition_index)]
        check=psych.loc[psych.Group.eq(group)&psych.ASL_cd.eq(row.ASL_cd)&psych.Delta_dB.eq(row.signed_delta)].iloc[0]
        assert check.n==len(cell) and check.n_right==cell.Choice.eq(1).sum()
        np.testing.assert_allclose(check.p_right_data,cell.Choice.eq(1).mean(),atol=1e-15,rtol=0)
        model_right=scipy_first_passage(1.5,1,row.Mean_dB,row.signed_delta,means,True)/independent_norm[row.condition_index]
        np.testing.assert_allclose(check.p_right_model,model_right,atol=1e-9,rtol=0)
        # Entire exported mean curve independently reconstructed on its 1 ms grid.
        independent_density=sum(scipy_first_passage(times,c,row.Mean_dB,row.signed_delta,means) for c in [-1,1])/independent_norm[row.condition_index]
        np.testing.assert_allclose(curves["density"][row.condition_index],independent_density,atol=1e-8,rtol=1e-6)
        ordered=np.sort(cell.RT_s.to_numpy());n=len(ordered)
        predicted=sum(scipy_first_passage(ordered,c,row.Mean_dB,row.signed_delta,means,True) for c in [-1,1])/independent_norm[row.condition_index]
        ks=max(np.max(np.arange(1,n+1)/n-predicted),np.max(predicted-np.arange(n)/n))
        saved_ks=cdf_table.loc[cdf_table.Group.eq(group)&cdf_table.ASL_cd.eq(row.ASL_cd)&cdf_table.Delta_dB.eq(row.signed_delta)].iloc[0]
        np.testing.assert_allclose(saved_ks.cdf_max_abs_error,ks,atol=1e-9,rtol=0)
    for (asl,absdelta),bins in histograms.loc[histograms.Group.eq(group)].groupby(["ASL_cd","abs_Delta_dB"]):
        subset=sub.loc[sub.ASL_cd.eq(asl)]
        if absdelta!="all": subset=subset.loc[subset.signed_delta.abs().eq(float(absdelta))]
        bins=bins.sort_values("bin_left_s")
        edges=np.linspace(0,1.5,76)
        counts=np.histogram(subset.RT_s,edges)[0]
        assert np.array_equal(counts,bins["count"]) and bins.n.eq(len(subset)).all()
        np.testing.assert_allclose(bins.density,counts/len(subset)/np.diff(edges),rtol=1e-12,atol=1e-12)
        # All signs mixed with observed frequencies, then both choices summed.
        weights=subset.groupby("condition_index").size()/len(subset)
        assert np.isclose(weights.sum(),1)
        mixture=np.sum([curves["density"][i]*weight for i,weight in weights.items()],axis=0)
        assert abs(np.trapz(mixture,times)-1)<1e-5
    report=json.loads((folder/"convergence_summary.json").read_text())
    conv=pd.read_csv(folder/"convergence.csv")
    loss=np.load(folder/"losses.npy")
    assert len(loss)==report["completed_steps"] and np.isfinite(loss).all()
    assert report["completed_steps"]>=config["min_steps"]
    assert report["restored_best_step"]==int(conv.loc[conv.mean_negative_elbo.idxmin(),"step"])
    assert np.isclose(report["best_window_mean_negative_elbo"],conv.mean_negative_elbo.min())
    lowest=np.inf; since_best=0; selected_step=0
    for row in conv.itertuples():
        raw_mean=loss[row.step-1000:row.step].mean()
        np.testing.assert_allclose(row.mean_negative_elbo,raw_mean,rtol=1e-12,atol=1e-9)
        lower=raw_mean<lowest
        if lower:
            lowest=raw_mean;since_best=0;selected_step=row.step
        else:
            since_best+=1
        assert row.updated_best==lower and row.patience==since_best
        assert row.best_step==selected_step
    assert report['confirmation_windows']==since_best
    assert report['updates_after_restored_best']==report['completed_steps']-selected_step
    assert report['flagged']==(not(report['minimum_followup_confirmed'] and report['terminal_parameter_stable'] and report['restored_checkpoint_parameter_stable']))
    if not report['flagged']:
        assert since_best>=25 and report['updates_after_restored_best']>=25000
    source_root=Path(config['source_fit'])
    initialization=json.loads((folder/'initialization.json').read_text())
    source_guide=source_root/group.lower()/'guide_parameters.npz'
    assert sha256(source_guide)==initialization['source_guide_sha256']
    assert initialization['optimizer_iteration']==0 and initialization['fresh_loss_history']
    initial_guide=np.load(folder/'initial_guide_parameters.npz')
    original_guide=np.load(source_guide)
    for name in initial_guide.files:
        np.testing.assert_array_equal(initial_guide[name],original_guide[name])
    for step,flag in [(report["completed_steps"],report["terminal_parameter_stable"]),
                      (report["restored_best_step"],report["restored_checkpoint_parameter_stable"])]:
        last=conv.loc[conv.step.le(step)].tail(3)
        means3=last[[n+"_mean" for n in PARAM_NAMES]].to_numpy()
        sds3=last[[n+"_sd" for n in PARAM_NAMES]].to_numpy()
        changes=np.ptp(means3,axis=0)/sds3.min(axis=0)
        assert bool((changes<.1).all())==flag
        np.testing.assert_allclose(changes.max(),last.iloc[-1].maximum_mean_change_sd,rtol=1e-9,atol=1e-10)
    if report["stop_reason"]=="patience12_and_parameter_stability":
        assert report["terminal_parameter_stable"] and int(conv.iloc[-1].patience)>=12
    reports[group]=dict(n_trials=n_trials,n_participants=n_people,source_rows_verified=n_trials,
                        source_row_id_bytes_sha256=hashlib.sha256(np.asarray(sub.source_row,dtype="<i8").tobytes()).hexdigest(),
                        five_free_parameters=free,delay_support_upper_s=upper,
                        parameter_points_checked=len(points),numerics=metrics,
                        independent_joint_log_likelihood=independent_ll,
                        convergence=report)
    print(group,reports[group],flush=True)

# %% Independently evaluate both parameter sets on the same selected population.
common=pd.read_csv(out/'common_window_fit_quality.csv')
common_cells=pd.read_csv(out/'common_window_conditions.csv',float_precision='round_trip')
for group in ['Control','ASD']:
    sub,conditions,_,_=group_data(frame,group)
    for label,root in [('previous_reconditioned',Path(config['source_fit'])),('refit',out)]:
        samples=np.load(root/group.lower()/'posterior_samples.npz')
        means=np.array([samples[n].mean() for n in PARAM_NAMES])
        norm=sum(scipy_first_passage(1.5,c,conditions.Mean_dB.to_numpy(),conditions.signed_delta.to_numpy(),means,True) for c in [-1,1])
        independent_ll=np.sum(np.log(scipy_first_passage(sub.RT_s,sub.Choice,sub.Mean_dB,sub.Delta_dB,means))-np.log(norm[sub.condition_index.to_numpy()]))
        q=common.loc[common.Group.eq(group)&common.fit.eq(label)].iloc[0]
        assert q.n_trials==len(sub) and q.retention_upper_s==1.5
        np.testing.assert_allclose(q.mean_parameter_joint_log_likelihood,independent_ll,atol=1e-7,rtol=0)
        cells=common_cells.loc[common_cells.Group.eq(group)&common_cells.fit.eq(label)]
        for row in conditions.itertuples():
            saved=cells.loc[cells.ASL_cd.eq(row.ASL_cd)&cells.Delta_dB.eq(row.signed_delta)].iloc[0]
            subset=sub.loc[sub.condition_index.eq(row.condition_index)]
            ordered=np.sort(subset.RT_s.to_numpy());n=len(ordered)
            assert saved.n==n and saved.n_right==subset.ChoseRight.sum()
            probability=scipy_first_passage(1.5,1,row.Mean_dB,row.signed_delta,means,True)/norm[row.condition_index]
            np.testing.assert_allclose(saved.p_right_model,probability,atol=1e-9,rtol=0)
            cdf=sum(scipy_first_passage(ordered,c,row.Mean_dB,row.signed_delta,means,True) for c in [-1,1])/norm[row.condition_index]
            gap=max(np.max(np.arange(1,n+1)/n-cdf),np.max(cdf-np.arange(n)/n))
            np.testing.assert_allclose(saved.cdf_max_abs_error,gap,atol=1e-9,rtol=0)
        np.testing.assert_allclose(q.psychometric_rmse,np.sqrt(np.mean((cells.p_right_data-cells.p_right_model)**2)),atol=1e-12)
        np.testing.assert_allclose(q.mean_condition_cdf_error,cells.cdf_max_abs_error.mean(),atol=1e-12)
        np.testing.assert_allclose(q.max_condition_cdf_error,cells.cdf_max_abs_error.max(),atol=1e-12)

# %% Figure files and prior-artifact preservation are checked after numerical audits.
figures={}
for path in sorted(out.rglob("*.png")):
    with Image.open(path) as im:
        im.verify()
    pdf=path.with_suffix(".pdf")
    with fitz.open(pdf) as doc:
        assert len(doc)==1
        page=doc[0]
        assert page.rect.width>0 and page.rect.height>0
        outside=[span["text"] for block in page.get_text("dict")["blocks"] if "lines" in block
                 for line in block["lines"] for span in line["spans"]
                 if span["bbox"][0]<-1 or span["bbox"][1]<-1 or span["bbox"][2]>page.rect.width+1 or span["bbox"][3]>page.rect.height+1]
        assert not outside,(pdf,outside)
    figures[str(path.relative_to(out))]=sha256(path)
preservation=out/"preserved_baseline_sha256.json"
if preservation.exists():
    baseline=json.loads(preservation.read_text())
    # README is intentionally extended with the new analysis; all earlier artifacts are preserved.
    unchanged=[name for name,digest in baseline.items() if name!="ASD_Human_Data/README.md"]
    assert all(sha256(ROOT/name)==baseline[name] for name in unchanged)
else:
    unchanged=[]
audit=dict(status="passed",original_mat_rows=n_rows,selected_source_rows=11235,
           groups=reports,probability_cells_verified=len(psych),histogram_bins_verified=len(histograms),
           smooth_psychometric_points_verified=len(smooth_psych),
           condition_cdf_comparisons_verified=len(cdf_table),figures=figures,
           prior_artifacts_hash_verified=len(unchanged),source_sha256=sha256(source),
           csv_sha256=sha256(csv),verifier_sha256=sha256(__file__),
           note="Numerical, row and PDF bounds checks; visual inspection recorded separately")
(out/"independent_validation.json").write_text(json.dumps(audit,indent=2))
print("PASSED:",len(psych),"probabilities,",len(histograms),"histogram bins;",len(figures),"PNG/PDF pairs")
