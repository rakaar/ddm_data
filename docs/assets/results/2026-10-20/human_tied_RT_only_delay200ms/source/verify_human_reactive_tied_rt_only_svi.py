# %%
"""Audit RT-only human fits, the 200-ms delay support and marginal RT predictions."""
import os
os.environ.setdefault('JAX_PLATFORM_NAME','cpu')
from pathlib import Path
from io import BytesIO
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.io.matlab._mio5 import MatFile5Reader
from scipy.integrate import quad
from scipy.special import expit
from PIL import Image
import fitz
import jax
import jax.numpy as jnp
from numpyro import handlers
from human_reactive_tied_rt_only_utils import (
    ROOT,PARAM_NAMES,INITIAL,group_data,log_likelihood,bound_cdf,bound_density,
    retained_mass,reactive_model,sha256,
)
from human_tied_independent_fpt import scipy_first_passage

# %% Configuration and re-read of original MATLAB MCOS rows without rewriting exports.
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir",type=Path,default=ROOT/"ASD_Human_Data/reactive_tied_rt_only_svi_0p2_1p5s_delay200ms")
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
frame=frame.loc[frame.RT_s.between(.2,1.5)].copy()
identities=frame.source_row.to_numpy()-1
assert np.array_equal(identities,np.flatnonzero((condition=="RT") & (arrays["RT_s"]>=.2) & (arrays["RT_s"]<=1.5)))
assert np.array_equal(original_groups[identities],frame.Group.to_numpy())
for name in frame.columns:
    if name in arrays and name not in ["Group","Condition"]:
        np.testing.assert_allclose(frame[name],arrays[name][identities],rtol=0,atol=1e-12,equal_nan=True)
assert len(frame)==11231 and frame.source_row.is_unique
assert frame.Condition.eq("RT").all() and frame.Valid.eq(1).all()
assert np.array_equal(frame.Choice.eq(1).astype(int),frame.ChoseRight)
assert frame.Stim_duration_s.isna().all() and frame.RT_s.between(.2,1.5,inclusive="both").all()
assert np.allclose(frame.Mean_dB,(frame.Stim_left_dB+frame.Stim_right_dB)/2,rtol=0,atol=1e-12)
selected_export=pd.read_csv(out/'selected_trials.csv')
excluded_export=pd.read_csv(out/'excluded_trials.csv')
assert np.array_equal(selected_export.source_row,frame.source_row)
assert np.array_equal(excluded_export.source_row,original_frame.loc[~original_frame.RT_s.between(.2,1.5),'source_row'])
assert len(excluded_export)==169
assert excluded_export.RT_s.lt(.2).sum()==4 and excluded_export.RT_s.gt(1.5).sum()==165
assert set(selected_export.source_row).isdisjoint(excluded_export.source_row)
assert set(selected_export.source_row)|set(excluded_export.source_row)==set(original_frame.source_row)
assert frame.Participant.nunique()==19
frame["signed_delta"]=frame.Delta_dB.round(2)

# %% Marginal RT likelihood, data-free response labels and common-window predictions.
config=json.loads((out/'run_config.json').read_text())
assert config['retention_window_s']==[.2,1.5] and config['delay_prior_upper_s']==.2
assert config['response_side_used_in_likelihood'] is False
assert config['parameters']==PARAM_NAMES and len(PARAM_NAMES)==5
assert config['guide']=='AutoMultivariateNormal' and config['seed']==0
assert config['elbo_particles']==1 and config['learning_rate']==.0002 and config['gradient_clip']==1.
for path,digest in config['source_hashes'].items(): assert sha256(ROOT/path)==digest
quality=pd.read_csv(out/'common_RT_fit_quality.csv')
cells=pd.read_csv(out/'condition_cdf_discrepancies.csv',float_precision='round_trip')
histograms=pd.read_csv(out/'rtd_histograms.csv')
weights=pd.read_csv(out/'mixture_weights.csv')
empirical=pd.read_csv(out/'empirical_cdfs.csv')
support=pd.read_csv(out/'delay_boundary_diagnostics.csv')
reports={}
for group,n_people,n_trials in [('Control',11,6528),('ASD',8,4703)]:
    folder=out/group.lower()
    sub,conditions,data,_=group_data(frame,group)
    data.pop('choice');data.update(rt_lower=.2,rt_upper=1.5)
    assert len(sub)==n_trials and sub.Participant.nunique()==n_people
    trace=handlers.trace(handlers.seed(reactive_model,0)).get_trace(data,.2)
    free=[n for n,v in trace.items() if v['type']=='sample' and not v['is_observed']]
    assert free==PARAM_NAMES
    assert float(trace['t_E_aff']['fn'].low)==0 and float(trace['t_E_aff']['fn'].high)==.2
    manifest=json.loads((folder/'data_manifest.json').read_text())
    assert manifest['delay_upper_s']==.2 and manifest['retention_window_s']==[.2,1.5]
    assert manifest['selected_source_rows_sha256']==sha256(folder/'selected_trials.csv')
    exported=pd.read_csv(folder/'selected_trials.csv')
    assert np.array_equal(sub.source_row,exported.source_row)
    np.testing.assert_allclose(sub.RT_s,exported.RT_s,atol=1e-15,rtol=0)
    assert 'choice' not in data
    samples=np.load(folder/'posterior_samples.npz')
    matrix=np.column_stack([samples[n] for n in PARAM_NAMES])
    assert matrix.shape==(10000,5) and np.isfinite(matrix).all()
    assert (matrix[:,:3]>0).all() and ((matrix[:,3]>.05)&(matrix[:,3]<.95)).all()
    assert ((matrix[:,4]>0)&(matrix[:,4]<.2)).all()
    summary=pd.read_csv(folder/'posterior_summary.csv').set_index('parameter').loc[PARAM_NAMES]
    means=matrix.mean(0)
    for name,expected in [('mean',means),('sd',matrix.std(0,ddof=1)),('q025',np.quantile(matrix,.025,axis=0)),('q975',np.quantile(matrix,.975,axis=0))]:
        np.testing.assert_allclose(summary[name],expected,rtol=1e-12,atol=1e-15)
    np.testing.assert_allclose(pd.read_csv(folder/'posterior_covariance.csv',index_col=0),np.cov(matrix,rowvar=False),rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(pd.read_csv(folder/'posterior_correlation.csv',index_col=0),np.corrcoef(matrix,rowvar=False),atol=1e-12)
    metrics=dict(K40_K80_max_density_abs=0.,K40_K80_max_mass_abs=0.,scipy_max_density_abs=0.,
                 scipy_max_cdf_abs=0.,unit_mass_max_abs=0.,cdf_derivative_max_abs=0.)
    points=[np.array(list(INITIAL.values())),means,np.median(matrix,axis=0)]
    points.extend(matrix[np.linspace(0,9999,12,dtype=int)])
    evaluate=jax.jit(jax.value_and_grad(log_likelihood))
    for point_index,p in enumerate(points):
        jp=jnp.asarray(p)
        ll,gradient=evaluate(jp,data)
        assert np.isfinite(ll) and np.isfinite(gradient).all()
        if point_index<2:
            a=dict(data,choice=jnp.ones(n_trials));b=dict(data,choice=-jnp.ones(n_trials))
            ll_a,grad_a=evaluate(jp,a);ll_b,grad_b=evaluate(jp,b)
            np.testing.assert_array_equal(ll,ll_a);np.testing.assert_array_equal(ll,ll_b)
            np.testing.assert_array_equal(gradient,grad_a);np.testing.assert_array_equal(gradient,grad_b)
        f40=sum(np.asarray(bound_density(data['rt'],side,data['B'],data['D'],jp,40)) for side in [-1,1])
        f80=sum(np.asarray(bound_density(data['rt'],side,data['B'],data['D'],jp,80)) for side in [-1,1])
        independent_pdf=sum(scipy_first_passage(np.asarray(data['rt']),side,np.asarray(data['B']),np.asarray(data['D']),p) for side in [-1,1])
        assert (independent_pdf>0).all()
        metrics['K40_K80_max_density_abs']=max(metrics['K40_K80_max_density_abs'],float(np.max(abs(f40-f80))))
        metrics['scipy_max_density_abs']=max(metrics['scipy_max_density_abs'],float(np.max(abs(f40-independent_pdf))))
        norm=np.asarray(retained_mass(data['condition_B'],data['condition_D'],jp,.2,1.5,40))
        norm80=np.asarray(retained_mass(data['condition_B'],data['condition_D'],jp,.2,1.5,80))
        assert (norm>0).all() and (norm<=1+1e-10).all()
        metrics['K40_K80_max_mass_abs']=max(metrics['K40_K80_max_mass_abs'],float(np.max(abs(norm-norm80))))
        for row in conditions.itertuples():
            B,D,i=row.Mean_dB,row.signed_delta,row.condition_index
            t=np.linspace(.2,1.5,101)
            for side in [-1,1]:
                fc=np.asarray(bound_cdf(jnp.asarray(t),side,B,D,jp))
                fi=scipy_first_passage(t,side,B,D,p,True)
                metrics['scipy_max_cdf_abs']=max(metrics['scipy_max_cdf_abs'],float(np.max(abs(fc-fi))))
            if point_index<3:
                integral=sum(quad(lambda t:float(scipy_first_passage(t,side,B,D,p)),.2,1.5,epsabs=1e-9)[0] for side in [-1,1])
                metrics['unit_mass_max_abs']=max(metrics['unit_mass_max_abs'],abs(integral/norm[i]-1))
                t=np.linspace(.22,1.4,40);h=1e-5
                derivative=sum((scipy_first_passage(t+h,c,B,D,p,True)-scipy_first_passage(t-h,c,B,D,p,True))/(2*h) for c in [-1,1])
                density=sum(scipy_first_passage(t,c,B,D,p) for c in [-1,1])
                metrics['cdf_derivative_max_abs']=max(metrics['cdf_derivative_max_abs'],float(np.max(abs(derivative-density))))
    for key,value in metrics.items():
        assert value<({'unit_mass_max_abs':1e-7,'cdf_derivative_max_abs':1e-6}.get(key,1e-8)),(key,value)

    # Both old and new parameters are scored with RT only on the same selected rows.
    for label,fit_root,curve_name in [('previous_joint',Path(config['source_fit']),'previous_joint_mean_parameter_curves.npz'),
                                      ('rt_only',out,'mean_parameter_curves.npz')]:
        posterior=np.load(fit_root/group.lower()/'posterior_samples.npz')
        p=np.array([posterior[n].mean() for n in PARAM_NAMES])
        curves=np.load(folder/curve_name);times=curves['time_s']
        norm=sum(scipy_first_passage(1.5,c,conditions.Mean_dB.to_numpy(),conditions.signed_delta.to_numpy(),p,True)
                 -scipy_first_passage(.2,c,conditions.Mean_dB.to_numpy(),conditions.signed_delta.to_numpy(),p,True) for c in [-1,1])
        origin=sum(scipy_first_passage(.2,c,conditions.Mean_dB.to_numpy(),conditions.signed_delta.to_numpy(),p,True) for c in [-1,1])
        density=sum(scipy_first_passage(sub.RT_s,c,sub.Mean_dB,sub.Delta_dB,p) for c in [-1,1])
        independent_ll=float(np.sum(np.log(density)-np.log(norm[sub.condition_index.to_numpy()])))
        saved=quality.loc[quality.Group.eq(group)&quality.fit.eq(label)].iloc[0]
        assert saved.n_trials==n_trials and saved.rt_lower_s==.2 and saved.rt_upper_s==1.5
        np.testing.assert_allclose(saved.mean_parameter_RT_log_likelihood,independent_ll,atol=1e-7,rtol=0)
        rows=cells.loc[cells.Group.eq(group)&cells.fit.eq(label)]
        for row in conditions.itertuples():
            selected=sub.loc[sub.condition_index.eq(row.condition_index)]
            ordered=np.sort(selected.RT_s.to_numpy());n=len(ordered);i=row.condition_index
            pdf=sum(scipy_first_passage(times,c,row.Mean_dB,row.signed_delta,p) for c in [-1,1])/norm[i]
            cdf=(sum(scipy_first_passage(times,c,row.Mean_dB,row.signed_delta,p,True) for c in [-1,1])-origin[i])/norm[i]
            np.testing.assert_allclose(curves['density'][i],pdf,atol=1e-8,rtol=1e-6)
            np.testing.assert_allclose(curves['cdf'][i],cdf,atol=1e-8,rtol=0)
            assert abs(cdf[0])<1e-10 and abs(cdf[-1]-1)<1e-10
            at_data=(sum(scipy_first_passage(ordered,c,row.Mean_dB,row.signed_delta,p,True) for c in [-1,1])-origin[i])/norm[i]
            gap=max(np.max(np.arange(1,n+1)/n-at_data),np.max(at_data-np.arange(n)/n))
            recorded=rows.loc[rows.ASL_cd.eq(row.ASL_cd)&rows.Delta_dB.eq(row.signed_delta)].iloc[0]
            assert recorded.n==n
            np.testing.assert_allclose(recorded.cdf_max_abs_error,gap,atol=1e-9,rtol=0)
        np.testing.assert_allclose(saved.mean_condition_cdf_error,rows.cdf_max_abs_error.mean(),atol=1e-12)
        np.testing.assert_allclose(saved.max_condition_cdf_error,rows.cdf_max_abs_error.max(),atol=1e-12)
        for (asl,d),w in weights.loc[weights.Group.eq(group)&weights.fit.eq(label)].groupby(['ASL_cd','abs_Delta_dB']):
            selected=sub.loc[sub.ASL_cd.eq(asl)]
            if d!='all': selected=selected.loc[selected.signed_delta.abs().eq(float(d))]
            expected=selected.groupby('condition_index').size()/len(selected)
            w=w.set_index('condition_index').sort_index()
            np.testing.assert_allclose(w.weight,expected,atol=1e-15,rtol=0)
            assert np.isclose(w.weight.sum(),1)
            mixture=sum(curves['density'][i]*weight for i,weight in expected.items())
            assert abs(np.trapz(mixture,times)-1)<1e-5
    for (asl,d),bins in histograms.loc[histograms.Group.eq(group)].groupby(['ASL_cd','abs_Delta_dB']):
        selected=sub.loc[sub.ASL_cd.eq(asl)]
        if d!='all': selected=selected.loc[selected.signed_delta.abs().eq(float(d))]
        bins=bins.sort_values('bin_left_s');edges=np.linspace(.2,1.5,66)
        counts=np.histogram(selected.RT_s,edges)[0]
        assert np.array_equal(bins['count'],counts) and bins.n.eq(len(selected)).all()
        np.testing.assert_allclose(bins.density,counts/len(selected)/np.diff(edges),rtol=1e-12,atol=1e-12)
    for asl,ecdf in empirical.loc[empirical.Group.eq(group)].groupby('ASL_cd'):
        ordered=np.sort(sub.loc[sub.ASL_cd.eq(asl),'RT_s'])
        np.testing.assert_allclose(ecdf.RT_s,ordered,atol=1e-15,rtol=0)
        np.testing.assert_allclose(ecdf.empirical_cdf,np.arange(1,len(ordered)+1)/len(ordered),atol=1e-15,rtol=0)
    diagnostic=support.loc[support.Group.eq(group)].iloc[0]
    assert diagnostic.delay_prior_upper_s==.2
    np.testing.assert_allclose(diagnostic.fraction_within_1ms_of_upper,np.mean(matrix[:,4]>.199),atol=1e-12)

    # Replay every strict patience reset and check the actual warm-start map.
    conv=pd.read_csv(folder/'convergence.csv');losses=np.load(folder/'losses.npy')
    report=json.loads((folder/'convergence_summary.json').read_text())
    assert len(losses)==report['completed_steps'] and len(losses)>=config['min_steps']
    best=np.inf;since_best=0;best_step=0
    for row in conv.itertuples():
        average=losses[row.step-1000:row.step].mean()
        np.testing.assert_allclose(row.mean_negative_elbo,average,rtol=1e-12,atol=1e-9)
        better=average<best
        if better: best=average;since_best=0;best_step=row.step
        else: since_best+=1
        assert row.updated_best==better and row.patience==since_best and row.best_step==best_step
    assert report['restored_best_step']==best_step and report['confirmation_windows']==since_best
    assert report['flagged']==(not(report['minimum_followup_confirmed'] and report['terminal_parameter_stable'] and report['restored_checkpoint_parameter_stable']))
    assert report['updates_after_restored_best']==report['completed_steps']-best_step
    for step,flag in [(best_step,report['restored_checkpoint_parameter_stable']),
                      (report['completed_steps'],report['terminal_parameter_stable'])]:
        last=conv.loc[conv.step.le(step)].tail(3)
        shift=np.ptp(last[[n+'_mean' for n in PARAM_NAMES]].to_numpy(),axis=0)/last[[n+'_sd' for n in PARAM_NAMES]].to_numpy().min(0)
        assert bool((shift<.1).all())==flag
    if not report['flagged']:
        assert since_best>=25 and report['updates_after_restored_best']>=25000
        assert report['terminal_parameter_stable'] and report['restored_checkpoint_parameter_stable']
    initialization=json.loads((folder/'initialization.json').read_text())
    original=Path(config['source_fit'])/group.lower()/'guide_parameters.npz'
    assert initialization['source_guide_sha256']==sha256(original)
    old=np.load(original);warm=np.load(folder/'initial_guide_parameters.npz')
    old_loc=old['auto_loc'];expected_loc=old_loc.copy();expected_scale=old['auto_scale_tril'].copy()
    center=initialization['source_delay_upper_s']*expit(old_loc[4])
    expected_loc[4]=np.log(center/(.2-center))
    factor=(1-center/initialization['source_delay_upper_s'])/(1-center/.2)
    expected_scale[4,:]*=factor
    np.testing.assert_allclose(warm['auto_loc'],expected_loc,atol=1e-14,rtol=1e-14)
    np.testing.assert_allclose(warm['auto_scale_tril'],expected_scale,atol=1e-14,rtol=1e-14)
    assert initialization['optimizer_iteration']==0 and initialization['fresh_loss_history']
    np.testing.assert_allclose(initialization['mapped_delay_center_s'],center,atol=1e-15,rtol=0)
    reports[group]=dict(n_trials=n_trials,n_participants=n_people,source_rows_verified=n_trials,
        five_free_parameters=free,response_labels_do_not_change_likelihood_or_gradients=True,
        delay_prior_upper_s=.2,parameter_points_checked=len(points),numerics=metrics,convergence=report)
    print(group,reports[group],flush=True)

# %% Verify rendered files, PDF text bounds and previous fit/asset hashes.
figures={}
for path in sorted(out.rglob('*.png')):
    with Image.open(path) as im: im.verify()
    with fitz.open(path.with_suffix('.pdf')) as document:
        assert len(document)==1
        page=document[0]
        outside=[span['text'] for block in page.get_text('dict')['blocks'] if 'lines' in block
                 for line in block['lines'] for span in line['spans']
                 if span['bbox'][0]<-1 or span['bbox'][1]<-1 or span['bbox'][2]>page.rect.width+1 or span['bbox'][3]>page.rect.height+1]
        assert not outside,(path,outside)
    figures[str(path.relative_to(out))]=sha256(path)
previous=json.loads((out/'preserved_baseline_sha256.json').read_text())
assert all(sha256(ROOT/path)==digest for path,digest in previous.items())
audit=dict(status='passed',original_mat_rows=n_rows,selected_source_rows=11231,excluded_source_rows=169,
    early_RT_excluded=4,late_RT_excluded=165,groups=reports,condition_CDF_checks=len(cells),
    histogram_bins_verified=len(histograms),empirical_CDF_points_verified=len(empirical),mixture_rows_verified=len(weights),
    figures=figures,prior_artifacts_hash_verified=len(previous),source_sha256=sha256(source),
    csv_sha256=sha256(csv),verifier_sha256=sha256(__file__),
    independent_fpt_sha256=sha256(ROOT/'ASD_Human_Data/human_tied_independent_fpt.py'))
(out/'independent_validation.json').write_text(json.dumps(audit,indent=2))
print('PASSED:',len(cells),'condition CDF comparisons;',len(histograms),'histogram bins;',len(figures),'PNG/PDF pairs')
