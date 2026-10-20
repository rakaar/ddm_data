# %%
"""Independent numerical, data, and artifact checks for human DDM debugging."""
import os
os.environ.setdefault('JAX_PLATFORM_NAME','cpu')
from io import BytesIO
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.io.matlab._mio5 import MatFile5Reader
from scipy.integrate import quad
from PIL import Image
import fitz
import jax
import jax.numpy as jnp
from human_reactive_tied_rt_only_utils import ROOT,load_trials,group_data,sha256
from human_plain_ddm_diagnostics_utils import ddm_log_density,ddm_log_cdf,scipy_ddm,tied_coordinates
sys.path.insert(0,str(ROOT/'fit_each_condn'))
from svi_gamma_omega_likelihood_utils import rho_E_gamma_omega_with_w_jax,CDF_E_gamma_omega_with_w_jax
from numpyro_vanilla_condition_delay_svi_utils import rho_E_vanilla_jax,CDF_E_vanilla_jax
import time_vary_norm_utils as paper

# %% Original MAT identities and fields, independently decoded.
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',type=Path,default=ROOT/'ASD_Human_Data/reactive_ddm_debug_0_1p8s_delay300ms')
args=parser.parse_args();out=args.output_dir
frame,provenance=load_trials(0.,1.8)
saved=pd.read_csv(out/'selected_trials.csv',float_precision='round_trip')
assert np.array_equal(frame.source_row,saved.source_row)
extraction=json.loads((ROOT/'ASD_Human_Data/simultaneous_rt_extraction.json').read_text())
source=ROOT/'ASD_Human_Data'/extraction['source']
raw=loadmat(source)['__function_workspace__'].tobytes()
stream=BytesIO(b'MATLAB 5.0 MAT-file'.ljust(116,b' ')+b'\0'*8+raw[:4]+raw[8:])
reader=MatFile5Reader(stream);reader.initialize_read();reader.read_file_header()
header,_=reader.read_var_header();cells=list(reader.read_var_array(header)['MCOS'][0,0][0]['arr'].flat)
names=[v.item() for v in cells[10].flat]
arrays=dict(zip(names,[np.asarray(v).ravel() for v in cells[5].flat]))
n_rows=int(cells[7].item());encoded=cells[4].ravel()[4+n_rows:].tobytes().decode('utf-16le')
condition=np.array([encoded[i:i+2] for i in range(0,len(encoded),2)])
groups=[v.item() for v in cells[2].flat];original_group=np.array([groups[int(c)-1] for c in cells[3].ravel()])
selected=np.flatnonzero((condition=='RT')&(arrays['RT_s']>=0)&(arrays['RT_s']<=1.8))
assert np.array_equal(selected,frame.source_row.to_numpy()-1)
assert np.array_equal(original_group[selected],frame.Group)
for name in frame.columns:
    if name in arrays and name not in ['Group','Condition']:
        np.testing.assert_allclose(frame[name],arrays[name][selected],rtol=0,atol=1e-12,equal_nan=True)
assert len(frame)==11344 and frame.source_row.is_unique and frame.RT_s.lt(.2).sum()==4
assert frame.groupby('Group').size().to_dict()=={'ASD':4770,'Control':6574}
assert frame.groupby('Group').Participant.nunique().to_dict()=={'ASD':8,'Control':11}
assert np.allclose(frame.Delta_dB,frame.Stim_right_dB-frame.Stim_left_dB,rtol=0,atol=1e-12)
assert np.allclose(frame.Mean_dB,(frame.Stim_right_dB+frame.Stim_left_dB)/2,rtol=0,atol=1e-12)
manifest=json.loads((out/'analysis_manifest.json').read_text())
for path,digest in manifest['source_hashes'].items():assert sha256(ROOT/path)==digest,path
for path,digest in json.loads((out/'baseline_sha256.json').read_text()).items():assert sha256(ROOT/path)==digest,path

# %% Per-condition fits and curves versus independent SciPy and old paper kernels.
parameters=pd.read_csv(out/'condition_parameters.csv',float_precision='round_trip')
quality=pd.read_csv(out/'fit_quality.csv',float_precision='round_trip')
condition_quality=pd.read_csv(out/'condition_fit_quality.csv',float_precision='round_trip')
histograms=pd.read_csv(out/'rtd_histograms.csv',float_precision='round_trip')
numerical={};low_rows=[];lambda_checks=[]
for group in ['Control','ASD']:
    folder=out/group.lower();sub,conditions,data,_=group_data(frame,group)
    saved_curves=np.load(folder/'comparison_curves.npz');times=saved_curves['time_s']
    assert len(times)==901 and np.isclose(times[-1],1.8)
    maxima=dict(scipy_pdf=0.,scipy_cdf=0.,K40_K80_pdf=0.,K40_K80_cdf=0.,
                legacy_gamma_omega_pdf=0.,legacy_gamma_omega_cdf=0.,conditional_mass=0.,
                finite_difference_gradient=0.,cdf_derivative=0.,likelihood_reconstruction=0.)
    models=parameters.loc[parameters.Group.eq(group),'model'].unique()
    for model in models:
        cells=parameters.loc[parameters.Group.eq(group)&parameters.model.eq(model)].sort_values(['ASL_cd','Delta_dB'])
        g=cells.gamma.to_numpy();o=cells.omega.to_numpy();w=float(cells.w.iloc[0]);delay=float(cells.delay_s.iloc[0])
        assert 0<=delay<sub.RT_s.min() and delay<=.3 and .05<=w<=.95
        assert np.isfinite(g).all() and np.isfinite(o).all() and (o>0).all()
        test_t=np.linspace(delay+.025,1.8,31)[None,:];G,O=g[:,None],o[:,None]
        pdf=np.exp(np.asarray(ddm_log_density(test_t,G,O,w,delay)))
        cdf=np.exp(np.asarray(ddm_log_cdf(test_t,G,O,w,delay)))
        norm=np.exp(np.asarray(ddm_log_cdf(1.8,g,o,w,delay)))
        independent_pdf=scipy_ddm(test_t,G,O,w,delay)
        independent_cdf=scipy_ddm(test_t,G,O,w,delay,cumulative=True)
        larger_pdf=np.exp(np.asarray(ddm_log_density(test_t,G,O,w,delay,K=80)))
        larger_cdf=np.exp(np.asarray(ddm_log_cdf(test_t,G,O,w,delay,K=80)))
        legacy_pdf=sum(np.asarray(rho_E_gamma_omega_with_w_jax(test_t-delay,G,O,b,w,40)) for b in [-1,1])
        legacy_cdf=sum(np.asarray(CDF_E_gamma_omega_with_w_jax(test_t-delay,G,O,b,w,40)) for b in [-1,1])
        for key,error in [('scipy_pdf',pdf-independent_pdf),('scipy_cdf',cdf-independent_cdf),
                          ('K40_K80_pdf',pdf-larger_pdf),('K40_K80_cdf',cdf-larger_cdf),
                          ('legacy_gamma_omega_pdf',pdf-legacy_pdf),('legacy_gamma_omega_cdf',cdf-legacy_cdf)]:
            maxima[key]=max(maxima[key],float(np.max(abs(error))))
        assert np.max(abs(pdf-independent_pdf))<1e-10 and np.max(abs(cdf-independent_cdf))<1e-10
        assert np.max(abs(pdf-larger_pdf))<1e-10 and np.max(abs(cdf-larger_cdf))<1e-10
        assert np.max(abs(pdf-legacy_pdf))<1e-9 and np.max(abs(cdf-legacy_cdf))<1e-9
        f=saved_curves[model+'_pdf'];F=saved_curves[model+'_cdf']
        np.testing.assert_allclose(F[:,0],0.,atol=1e-12);np.testing.assert_allclose(F[:,-1],1.,atol=1e-12)
        assert np.min(np.diff(F,axis=1))>=-1e-12 and np.isfinite(f).all() and (f>=0).all()
        for row in conditions.itertuples():
            i=row.condition_index;selected=sub.loc[sub.condition_index.eq(i)];rt=selected.RT_s.to_numpy()
            p=scipy_ddm(np.sort(rt),g[i],o[i],w,delay,cumulative=True)/norm[i]
            n=len(rt);gap=max(np.max(np.arange(1,n+1)/n-p),np.max(p-np.arange(n)/n))
            saved_row=condition_quality.loc[(condition_quality.Group==group)&(condition_quality.model==model)&
                (condition_quality.ASL_cd==row.ASL_cd)&(condition_quality.Delta_dB==row.signed_delta)].iloc[0]
            # Saved CDF metrics interpolate a 2-ms plotting grid.
            assert abs(gap-saved_row.cdf_max_error)<7e-5
            predicted=scipy_ddm(1.8,g[i],o[i],w,delay,cumulative=True,side=1)/norm[i]
            assert abs(predicted-cells.predicted_P_right.iloc[i])<1e-10
            assert abs(row.right/row.n-cells.empirical_P_right.iloc[i])<1e-14
        rt=data['rt'];ids=data['condition'];choice=data['choice']
        rtll=float(jnp.sum(ddm_log_density(rt,jnp.array(g)[ids],jnp.array(o)[ids],w,delay)-jnp.log(jnp.array(norm)[ids])))
        jointll=float(jnp.sum(ddm_log_density(rt,jnp.array(g)[ids],jnp.array(o)[ids],w,delay,side=choice)-jnp.log(jnp.array(norm)[ids])))
        row=quality.loc[quality.Group.eq(group)&quality.model.eq(model)].iloc[0]
        maxima['likelihood_reconstruction']=max(maxima['likelihood_reconstruction'],abs(rtll-row.RT_log_likelihood),abs(jointll-row.joint_log_likelihood))
        for i in [0,4,5,19]:
            mass=quad(lambda t:float(scipy_ddm(t,g[i],o[i],w,delay)),delay,1.8,epsabs=1e-11)[0]/norm[i]
            maxima['conditional_mass']=max(maxima['conditional_mass'],abs(mass-1))
        h=1e-5;at=np.linspace(delay+.08,1.65,19)[None,:]
        derivative=(scipy_ddm(at+h,G,O,w,delay,cumulative=True)-scipy_ddm(at-h,G,O,w,delay,cumulative=True))/(2*h)
        maxima['cdf_derivative']=max(maxima['cdf_derivative'],float(np.max(abs(derivative-scipy_ddm(at,G,O,w,delay)))))
        # A representative parameter-gradient check is separate from time-CDF differentiation.
        p0=np.array([g[4],o[4],w,delay]);at_rt=jnp.array([.4,.6,.9,1.3])
        fun=lambda p:jnp.sum(ddm_log_density(at_rt,p[0],p[1],p[2],p[3])-ddm_log_cdf(1.8,p[0],p[1],p[2],p[3]))
        analytic=np.asarray(jax.grad(fun)(jnp.array(p0)));finite=[]
        for i in range(4):
            step=1e-5 if i<3 else 1e-6
            a=p0.copy();b=p0.copy();a[i]+=step;b[i]-=step
            finite.append((float(fun(jnp.array(a)))-float(fun(jnp.array(b))))/(2*step))
        assert np.isfinite(analytic).all()
        error=float(np.max(abs(analytic-finite)/(1+abs(analytic))))
        maxima['finite_difference_gradient']=max(maxima['finite_difference_gradient'],error)
        assert error<2e-6
        if model in ['SVI_mean','TIED_RT','DDM_shared_RT','DDM_shared_joint','DDM_free_joint']:
            subset=cells.loc[cells.Delta_dB.abs().eq(.1)]
            p=np.where(subset.Delta_dB.gt(0),subset.predicted_P_right,1-subset.predicted_P_right)
            sel=sub.loc[sub.signed_delta.abs().eq(.1)]
            actual=np.mean(sel.Choice.to_numpy()==np.sign(sel.signed_delta.to_numpy()))
            low_rows.append(dict(Group=group,model=model,n_trials=len(sel),observed_accuracy=actual,
                                 predicted_accuracy=float(np.average(p,weights=conditions.loc[conditions.signed_delta.abs().eq(.1),'n']))))
    assert maxima['conditional_mass']<1e-8 and maxima['cdf_derivative']<1e-6 and maxima['likelihood_reconstruction']<1e-7

    fit=json.loads((folder/'fits.json').read_text());physical=np.array(fit['tied_physical'])
    rate,t0,theta,w,delay=physical;B=np.array(data['condition_B'])[:,None];D=np.array(data['condition_D'])[:,None]
    grid=np.linspace(.03,1.5,25)[None,:];Z=(w-.5)*2*theta
    old_pdf=sum(paper.rho_E_minus_small_t_NORM_rate_norm_time_varying_fn_vec(
        grid,b,B,D,rate,t0,theta,Z,1.,grid,0.,False,False,40) for b in [-1,1])
    old_cdf=sum(paper.CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn_vec(
        grid,b,B,D,rate,t0,theta,Z,grid,0.,False,False,40) for b in [-1,1])
    g,o,_,_=tied_coordinates(jnp.array(fit['tied']['x']),jnp.array(B),jnp.array(D),fit['reference_B'])
    f=np.exp(np.array(ddm_log_density(grid+delay,g,o,w,delay)))
    F=np.exp(np.array(ddm_log_cdf(grid+delay,g,o,w,delay)))
    maxima['paper_pdf']=float(np.max(abs(f-old_pdf)));maxima['paper_cdf']=float(np.max(abs(F-old_cdf)))
    assert maxima['paper_pdf']<1e-10 and maxima['paper_cdf']<1e-10
    if group=='Control':
        x=np.array(fit['tied']['x']);new=x.copy();new[0]=np.log(1e-6)
        B=data['condition_B'];D=data['condition_D'];ids=data['condition'];rt=data['rt']
        oldg,oldo,w,delay=tied_coordinates(jnp.array(x),B,D,fit['reference_B'])
        newg,newo,_,_=tied_coordinates(jnp.array(new),B,D,fit['reference_B'])
        ll=float(jnp.sum(ddm_log_density(rt,newg[ids],newo[ids],w,delay)-ddm_log_cdf(1.8,newg,newo,w,delay)[ids]))
        oldF=jnp.exp(ddm_log_cdf(jnp.array(times)[None,:],oldg[:,None],oldo[:,None],w,delay)-ddm_log_cdf(1.8,oldg,oldo,w,delay)[:,None])
        newF=jnp.exp(ddm_log_cdf(jnp.array(times)[None,:],newg[:,None],newo[:,None],w,delay)-ddm_log_cdf(1.8,newg,newo,w,delay)[:,None])
        lambda_checks.append(dict(Group=group,original_lambda=float(np.exp(x[0])),smaller_lambda=1e-6,
            log_likelihood_change=ll+fit['tied']['nll'],max_cdf_change=float(np.max(abs(np.array(newF-oldF))))))
    numerical[group]=maxima
    for asl in [20,40]:
        for delta in ['all',.1,.3,.65,1.3,2.]:
            selected=sub.loc[sub.ASL_cd.eq(asl)]
            if delta!='all':selected=selected.loc[selected.signed_delta.abs().eq(delta)]
            expected=np.histogram(selected.RT_s,np.linspace(0,1.8,91))[0]
            saved=histograms.loc[histograms.Group.eq(group)&histograms.ASL_cd.eq(asl)&histograms.abs_Delta_dB.astype(str).eq(str(delta))]
            np.testing.assert_array_equal(saved['count'],expected)
            np.testing.assert_allclose(saved.density,expected/len(selected)/.02,atol=1e-12)

# %% Readable, unclipped PNG/PDF pairs and source preservation.
figures={}
for path,digest in manifest['figures'].items():
    p=out/path;assert sha256(p)==digest
    with Image.open(p) as im:
        im.verify()
    with fitz.open(p.with_suffix('.pdf')) as pdf:
        assert len(pdf)==1
        page=pdf[0]
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines',[]):
                for span in line['spans']:
                    x0,y0,x1,y1=span['bbox']
                    assert x0>=-1 and y0>=-1 and x1<=page.rect.width+1 and y1<=page.rect.height+1,(path,span)
    figures[path]=digest
assert len(figures)==6
for path,digest in json.loads((out/'baseline_sha256.json').read_text()).items():assert sha256(ROOT/path)==digest,path
pd.DataFrame(low_rows).to_csv(out/'low_delta_choice_check.csv',index=False,float_format='%.17g')
report=dict(status='passed',source_mat_rows=n_rows,selected_trials=len(frame),sub200ms_retained=4,
            source_hashes=dict(mat=sha256(source),csv=sha256(ROOT/'ASD_Human_Data/simultaneous_rt_trials.csv')),
            groups=numerical,lambda_lower_bound_sensitivity=lambda_checks,figures=figures,
            comparison='Same trial identities, RT window, conditioning and mixture weights; MLE diagnostics have no priors',
            meaning='Improved free-DDM training fit is a representational diagnostic, not evidence of predictive model superiority or a likelihood bug',
            verifier_sha256=sha256(Path(__file__)))
(out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
