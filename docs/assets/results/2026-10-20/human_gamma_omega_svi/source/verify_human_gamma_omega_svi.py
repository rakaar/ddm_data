# %%
"""Independent MAT, NumPy/SciPy, posterior, normalization and figure audit."""
import os
os.environ.setdefault('JAX_PLATFORM_NAME','cpu')
from pathlib import Path
from io import BytesIO
import argparse
import json
import pickle

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.io.matlab._mio5 import MatFile5Reader
from scipy.integrate import quad
import jax
import jax.numpy as jnp
from numpyro import handlers
from numpyro.infer.util import log_density
from numpyro.infer.autoguide import AutoMultivariateNormal
from numpyro.infer.initialization import init_to_value
from PIL import Image
import fitz
from human_gamma_omega_svi_utils import (
    ROOT, OUTPUT, PRIORS, PARAMETER_NAMES, PARAMETER_BLOCKS, sha256, load_data, prepare_group,
    sample_matrix, vector_params, numerical_check, gamma_omega_model, guide_support_model,
    trapezoidal_logpdf_jax, log_likelihood, ddm_log_density, ddm_log_cdf,
)
from human_plain_ddm_diagnostics_utils import scipy_ddm

# %% Re-read the opaque MAT workspace, keeping original one-based row identities.
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',type=Path,default=OUTPUT)
args=parser.parse_args();out=args.output_dir
config=json.loads((out/'run_config.json').read_text())
assert config['free_parameters_per_group']==61 and config['parameter_names']==PARAMETER_NAMES
assert config['priors']==PRIORS and config['rt_window_s']==[0,1.8]
assert config['guide']=='AutoMultivariateNormal' and config['elbo_particles']==1
assert config['learning_rate']==.0002 and config['gradient_clip']==1. and config['guide_init_scale']==.1
for path,digest in config['source_hashes'].items(): assert sha256(ROOT/path)==digest
full,selected,excluded,provenance=load_data()
source=ROOT/'ASD_Human_Data'/provenance['source']
raw=loadmat(source)['__function_workspace__'].tobytes()
stream=BytesIO(b'MATLAB 5.0 MAT-file'.ljust(116,b' ')+b'\0'*8+raw[:4]+raw[8:])
reader=MatFile5Reader(stream);reader.initialize_read();reader.read_file_header();header,_=reader.read_var_header()
cells=list(reader.read_var_array(header)['MCOS'][0,0][0]['arr'].flat)
names=[a.item() for a in cells[10].flat]
arrays=dict(zip(names,[np.asarray(a).ravel() for a in cells[5].flat]))
n_rows=int(cells[7].item())
chars=cells[4].ravel()[4+n_rows:].tobytes().decode('utf-16le')
condition=np.array([chars[i:i+2] for i in range(0,len(chars),2)])
group_names=[a.item() for a in cells[2].flat]
original_groups=np.array([group_names[int(c)-1] for c in cells[3].ravel()])
ids=full.source_row.to_numpy()-1
assert np.array_equal(ids,np.flatnonzero(condition=='RT'))
assert np.array_equal(original_groups[ids],full.Group.to_numpy())
for name in full:
    if name in arrays and name not in ['Group','Condition']:
        np.testing.assert_allclose(full[name],arrays[name][ids],rtol=0,atol=1e-12,equal_nan=True)
assert int(selected.RT_s.lt(.2).sum())==4 and selected.source_row.is_unique
assert set(selected.source_row).isdisjoint(excluded.source_row)
assert set(selected.source_row)|set(excluded.source_row)==set(full.source_row)
assert np.array_equal(pd.read_csv(out/'selected_trials.csv').source_row,selected.source_row)
assert np.array_equal(pd.read_csv(out/'excluded_trials.csv').source_row,excluded.source_row)
assert selected.Valid.eq(1).all() and selected.Stim_duration_s.isna().all()
assert np.array_equal(selected.Choice.eq(1).astype(int),selected.ChoseRight)
np.testing.assert_allclose(selected.Delta_dB,selected.Stim_right_dB-selected.Stim_left_dB,rtol=0,atol=1e-12)

# %% Check the actual trapezoidal densities, not just the Uniform base supports.
prior_integrals={}
for name,prior in PRIORS.items():
    lo,hi=prior['hard'];pl,ph=prior['plausible']
    height=1/((pl-lo)/2+(ph-pl)+(hi-ph)/2)
    def density(x): return float(np.interp(x,[lo,pl,ph,hi],[0,height,height,0],left=0,right=0))
    area=quad(density,lo,hi,points=[pl,ph],epsabs=1e-12)[0]
    np.testing.assert_allclose(area,1,rtol=0,atol=1e-12)
    x=np.linspace(lo+1e-7,hi-1e-7,1001)
    np.testing.assert_allclose(np.exp(np.asarray(trapezoidal_logpdf_jax(x,lo,pl,ph,hi))),
                               np.interp(x,[lo,pl,ph,hi],[0,height,height,0]),rtol=1e-12,atol=1e-12)
    prior_integrals[name]=area
hist=pd.read_csv(out/'rtd_histograms.csv',float_precision='round_trip')
curves=pd.read_csv(out/'rtd_model_curves.csv',float_precision='round_trip')
condition_curves=pd.read_csv(out/'condition_model_curves.csv',float_precision='round_trip')
psych=pd.read_csv(out/'psychometric_comparison.csv',float_precision='round_trip')
quality=pd.read_csv(out/'fit_quality.csv',float_precision='round_trip')
all_summary=pd.read_csv(out/'parameter_summary.csv',float_precision='round_trip')
reports={}

# %% Reconstruct every posterior interval, bound probability, histogram and mixture.
for group,n_trials,n_people in [('Control',6574,11),('ASD',4770,8)]:
    folder=out/group.lower()
    sub,conditions,data,initial,initial_source=prepare_group(selected,group,Path(config['initial_root']))
    assert len(sub)==n_trials and sub.Participant.nunique()==n_people
    observed=pd.read_csv(folder/'selected_trials.csv',float_precision='round_trip')
    assert np.array_equal(observed.source_row,sub.source_row)
    for col in ['RT_s','Choice','Participant','GroupID','condition_index','ASL_cd','signed_delta']:
        np.testing.assert_allclose(observed[col],sub[col],rtol=0,atol=1e-12)
    manifest=json.loads((folder/'data_manifest.json').read_text())
    assert manifest['initialization_sha256']==sha256(initial_source)
    assert manifest['selected_trials_sha256']==sha256(folder/'selected_trials.csv')
    saved_conditions=pd.read_csv(folder/'conditions.csv',float_precision='round_trip')
    pd.testing.assert_frame_equal(saved_conditions,conditions,check_dtype=False,rtol=0,atol=1e-12)
    posterior=dict(np.load(folder/'posterior_samples.npz'));matrix=sample_matrix(posterior)
    assert matrix.shape==(10000,61) and np.isfinite(matrix).all()
    guide_params=dict(np.load(folder/'guide_parameters.npz'))
    with (folder/'best_checkpoint.pkl').open('rb') as stream:best_checkpoint=pickle.load(stream)
    for key in guide_params:
        np.testing.assert_array_equal(guide_params[key],np.asarray(best_checkpoint['params'][key]))
    guide=AutoMultivariateNormal(guide_support_model,init_loc_fn=init_to_value(values=initial),init_scale=.1)
    handlers.seed(guide,rng_seed=config['seed'])(data)
    regenerated=guide.sample_posterior(jax.random.PRNGKey(config['seed']+100000),guide_params,sample_shape=(10000,))
    for key in PARAMETER_BLOCKS:
        np.testing.assert_allclose(posterior[key],regenerated[key],rtol=0,atol=1e-13)
    saved=all_summary.loc[all_summary.Group.eq(group)]
    assert saved.parameter.tolist()==PARAMETER_NAMES
    for col,values in [('mean',matrix.mean(0)),('sd',matrix.std(0,ddof=1)),
                       ('q025',np.quantile(matrix,.025,axis=0)),('q975',np.quantile(matrix,.975,axis=0))]:
        np.testing.assert_allclose(saved[col],values,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(pd.read_csv(folder/'posterior_covariance.csv',index_col=0),np.cov(matrix,rowvar=False),rtol=1e-11,atol=1e-12)
    np.testing.assert_allclose(pd.read_csv(folder/'posterior_correlation.csv',index_col=0),np.corrcoef(matrix,rowvar=False),rtol=1e-11,atol=1e-12)
    assert np.all((matrix[:,20:40]>.01)&(matrix[:,20:40]<15))
    assert np.all((matrix[:,40:60]>0)&(matrix[:,40:60]<conditions.guide_delay_upper_s.to_numpy()))
    assert np.all((matrix[:,60]>0)&(matrix[:,60]<1))
    assert np.all(matrix[:,:20]>np.asarray(data['gamma_hard'][0])) and np.all(matrix[:,:20]<np.asarray(data['gamma_hard'][1]))
    params=vector_params(matrix.mean(0))
    numerical=numerical_check(params,data)
    # Model trace must keep delay prior high=.3, while only the guide prototype narrows it.
    actual=handlers.trace(handlers.seed(handlers.substitute(gamma_omega_model,data=params),rng_seed=0)).get_trace(data)
    guide_trace=handlers.trace(handlers.seed(handlers.substitute(guide_support_model,data=params),rng_seed=0)).get_trace(data)
    assert sum(np.size(site['value']) for site in actual.values() if site['type']=='sample' and not site.get('is_observed',False))==61
    actual_delay_distribution=actual['t_E_aff']['fn']
    while hasattr(actual_delay_distribution,'base_dist'):
        actual_delay_distribution=actual_delay_distribution.base_dist
    np.testing.assert_allclose(actual_delay_distribution.high,.3,rtol=0,atol=0)
    np.testing.assert_allclose(guide_trace['t_E_aff']['fn'].base_dist.high,conditions.guide_delay_upper_s,rtol=0,atol=1e-15)
    prior_log=0.
    for block in PARAMETER_BLOCKS:
        hard,flat=(data['gamma_hard'],data['gamma_plausible']) if block=='gamma' else (PRIORS[block]['hard'],PRIORS[block]['plausible'])
        prior_log+=float(np.asarray(trapezoidal_logpdf_jax(params[block],hard[0],flat[0],flat[1],hard[1])).sum())
    log_joint,_=log_density(gamma_omega_model,(data,),{},params)
    np.testing.assert_allclose(float(log_joint)-float(log_likelihood(params,data)),prior_log,rtol=0,atol=1e-8)
    mass=scipy_ddm(1.8,params['gamma'],params['omega'],params['w'],params['t_E_aff'],cumulative=True)
    side=data['choice'];ix=np.asarray(data['condition'])
    density=scipy_ddm(np.asarray(data['rt']),params['gamma'][ix],params['omega'][ix],params['w'],params['t_E_aff'][ix],side=np.asarray(side))
    independent_ll=float(np.sum(np.log(density)-np.log(mass[ix])))
    np.testing.assert_allclose(independent_ll,numerical['joint_log_likelihood_K40'],rtol=1e-10,atol=1e-6)
    np.testing.assert_allclose(independent_ll,quality.loc[quality.Group.eq(group),'joint_log_likelihood'].iloc[0],rtol=1e-10,atol=1e-6)
    area_errors=[];cdf_errors=[]
    for c in conditions.itertuples():
        i=c.condition_index;g=params['gamma'][i];o=params['omega'][i];tau=params['t_E_aff'][i];w=params['w']
        area=quad(lambda t:float(scipy_ddm(t,g,o,w,tau))/mass[i],tau,1.8,epsabs=1e-9,limit=150)[0]
        area_errors.append(abs(area-1))
        row=psych.loc[psych.Group.eq(group)&psych.condition_index.eq(i)].iloc[0]
        independent_right=float(scipy_ddm(1.8,g,o,w,tau,cumulative=True,side=1)/mass[i])
        np.testing.assert_allclose([row.n,row.right,row.empirical_p_right],[c.n,c.right,c.right/c.n],rtol=0,atol=1e-12)
        np.testing.assert_allclose(row.model_p_right,independent_right,rtol=1e-8,atol=1e-8)
        signed=condition_curves.loc[condition_curves.Group.eq(group)&condition_curves.condition_index.eq(i)]
        point=signed.iloc[::17]
        np.testing.assert_allclose(point.density,scipy_ddm(point.time_s.to_numpy(),g,o,w,tau)/mass[i],rtol=1e-7,atol=1e-8)
        independent_cdf=scipy_ddm(point.time_s.to_numpy(),g,o,w,tau,cumulative=True)/mass[i]
        cdf_errors.append(float(np.max(np.abs(point.cdf-independent_cdf))))
        np.testing.assert_allclose(point.cdf,independent_cdf,rtol=1e-7,atol=1e-8)
        rt=np.sort(sub.loc[sub.condition_index.eq(i),'RT_s'].to_numpy())
        f=scipy_ddm(rt,g,o,w,tau,cumulative=True)/mass[i]
        ecdf=np.arange(1,len(rt)+1)/len(rt)
        gap=max(np.abs(f-ecdf).max(),np.abs(f-ecdf+1/len(rt)).max())
        np.testing.assert_allclose(row.cdf_gap,gap,rtol=1e-7,atol=1e-8)
    assert max(area_errors)<1e-7 and max(cdf_errors)<1e-7
    for level in [20.,40.]:
        for delta in [.1,.3,.65,1.3,2.]:
            obs=sub.loc[sub.ASL_cd.eq(level)&np.isclose(sub.signed_delta.abs(),delta)]
            bins=hist.loc[hist.Group.eq(group)&hist.ASL_cd.eq(level)&hist.abs_delta.eq(delta)]
            counts,_=np.histogram(obs.RT_s,bins=np.linspace(0,1.8,91))
            assert np.array_equal(bins['count'],counts)
            np.testing.assert_allclose(bins.density,counts/(len(obs)*.02),rtol=1e-12,atol=1e-12)
            mix=curves.loc[curves.Group.eq(group)&curves.ASL_cd.eq(level)&curves.abs_delta.eq(delta)]
            ns=obs.groupby('signed_delta').size()
            np.testing.assert_allclose([mix.negative_weight.iloc[0],mix.positive_weight.iloc[0]],ns.to_numpy()/len(obs),rtol=0,atol=1e-12)
            expected=np.zeros(len(mix))
            for sign in [-1,1]:
                single=condition_curves.loc[condition_curves.Group.eq(group)&condition_curves.ASL_cd.eq(level)&condition_curves.signed_delta.eq(sign*delta)]
                expected+=single.density.to_numpy()*ns.loc[sign*delta]/len(obs)
            np.testing.assert_allclose(mix.density,expected,rtol=1e-12,atol=1e-12)
            assert abs(float(mix.cdf.iloc[-1])-1)<1e-12
    # Reconstruct the corrected patience rule from every recorded loss.
    losses=np.load(folder/'losses.npy');trace=pd.read_csv(folder/'convergence.csv',float_precision='round_trip')
    report=json.loads((folder/'convergence_summary.json').read_text())
    means=losses.reshape(-1,1000).mean(1)
    np.testing.assert_allclose(means,trace.mean_negative_elbo,rtol=1e-12,atol=1e-10)
    best_index=int(np.argmin(means));best_step=(best_index+1)*1000
    assert report['restored_best_step']==best_step and report['completed_steps']==len(losses)
    assert int(best_checkpoint['step'])==best_step
    assert report['confirmation_windows']==len(means)-best_index-1
    h=trace[[n+'_mean' for n in PARAMETER_NAMES]].to_numpy()[-3:]
    s=trace[[n+'_sd' for n in PARAMETER_NAMES]].to_numpy()[-3:]
    stable=bool(np.all(np.ptp(h,axis=0)/np.min(s,axis=0)<.1))
    assert stable==report['terminal_parameter_stable']
    reports[group]=dict(trials=n_trials,participants=n_people,parameters=61,numerical=numerical,
        independent_joint_log_likelihood=independent_ll,unit_conditional_mass_max_error=max(area_errors),
        scipy_cdf_max_error=max(cdf_errors),convergence=report)

# %% Preservation and exported figure geometry, alongside manual PNG inspection.
baseline=json.loads((out/'preservation_baseline.json').read_text())
for path,digest in baseline.items(): assert sha256(ROOT/path)==digest,path
unrelated=json.loads((out/'unrelated_worktree_baseline.json').read_text())
for path,digest in unrelated.items(): assert sha256(ROOT/path)==digest,path
plot_manifest=json.loads((out/'plot_manifest.json').read_text())
figure_checks=[]
for name in plot_manifest['figures']:
    path=out/name
    if path.suffix=='.png':
        with Image.open(path) as im:
            assert min(im.size)>1000
            assert np.asarray(im.convert('RGB')).std()>5
            figure_checks.append(dict(file=name,width=im.width,height=im.height))
    else:
        doc=fitz.open(path)
        for page in doc:
            for block in page.get_text('blocks'):
                if len(block)>6 and block[6]==0:
                    assert block[0]>=-.1 and block[1]>=-.1 and block[2]<=page.rect.width+.1 and block[3]<=page.rect.height+.1,(name,block)
        figure_checks.append(dict(file=name,pages=len(doc),all_text_within_page=True))
result=dict(status='passed',source_rows=n_rows,original_RT_rows=11400,retained_rows=11344,excluded_rows=56,
    retained_below_200ms=4,priors_integrate_to_one=prior_integrals,groups=reports,
    probabilities_reconstructed=40,histograms_reconstructed=20,signed_curves_reconstructed=40,
    mixture_curves_reconstructed=20,posterior_parameters_reconstructed=122,
    prior_artifacts_preserved=len(baseline),unrelated_modified_tracked_files_preserved=len(unrelated),
    figures=figure_checks,verifier_sha256=sha256(Path(__file__)))
(out/'independent_validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ['groups','figures']},indent=2))
