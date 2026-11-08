# %%
"""Independent source, quadrature, gradient, posterior and numerical checks."""
import os
os.environ.setdefault('JAX_PLATFORM_NAME','cpu')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
from io import BytesIO
import argparse,json,pickle
import numpy as np
import pandas as pd
from scipy.integrate import quad, trapezoid
from scipy.special import ndtr
from scipy.stats import truncnorm
from scipy.io import loadmat
from scipy.io.matlab._mio5 import MatFile5Reader
import jax
import jax.numpy as jnp
from numpyro import handlers
from human_shared_gaussian_delay_svi_utils import *
from human_gaussian_delay_svi_utils_wide import gaussian_delay_log_integral, gaussian_delay_trial_logpdf
from human_gamma_omega_variants_svi_utils import trapezoidal_logpdf_jax
from human_plain_ddm_diagnostics_utils import scipy_ddm

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--preflight-only',action='store_true')
parser.add_argument('--output-dir',type=Path,default=OUTPUT)
args=parser.parse_args();out=args.output_dir
config=json.loads((out/'run_config.json').read_text())
assert config['free_parameters_per_group']==16 and config['priors']==PRIORS
for path,digest in config['source_hashes'].items():assert sha256(ROOT/path)==digest,path


def reference(t,gamma,omega,w,mu,sd,side,cumulative=False):
    """Adaptive SciPy integration; independent NumPy DDM, no FFT/JAX quadrature."""
    if t<=0:return 0.
    a=2*(1-w if side==1 else w)
    low=min(a*a/(omega*2048),t*1e-8)
    norm=ndtr(mu/sd)
    def integrand(z):
        u=np.exp(z);d=t-u
        f=float(scipy_ddm(u,gamma,omega,w,0.,side=side,K=80))
        h=((ndtr((d-mu)/sd)-ndtr(-mu/sd))/norm if cumulative else
           np.exp(-.5*((d-mu)/sd)**2)/(sd*np.sqrt(2*np.pi)*norm))
        return f*max(0,h)*u
    mode=2*a*a/(omega*(3+np.sqrt(9+4*(gamma*a)**2)))
    points=[mode,.6/omega]+[t-mu-k*sd for k in [-10,-3,0,3,10]]
    points=sorted(set(np.log(x) for x in points if low<x<t))
    return quad(integrand,np.log(low),np.log(t),points=points,epsabs=1e-11,epsrel=2e-10,limit=200)[0]


# %% Independent MAT reconstruction of every retained/excluded source row.
full,selected,excluded,provenance=load_data()
assert json.loads(json.dumps(provenance))==config['data'], 'Input hashes or selection provenance changed since fitting'
source=ROOT/'ASD_Human_Data'/provenance['source']
raw=loadmat(source)['__function_workspace__'].tobytes()
stream=BytesIO(b'MATLAB 5.0 MAT-file'.ljust(116,b' ')+b'\0'*8+raw[:4]+raw[8:])
reader=MatFile5Reader(stream);reader.initialize_read();reader.read_file_header();header,_=reader.read_var_header()
cells=list(reader.read_var_array(header)['MCOS'][0,0][0]['arr'].flat)
names=[a.item() for a in cells[10].flat]
arrays=dict(zip(names,[np.asarray(a).ravel() for a in cells[5].flat]))
n=int(cells[7].item());chars=cells[4].ravel()[4+n:].tobytes().decode('utf-16le')
conditions=np.array([chars[i:i+2] for i in range(0,len(chars),2)])
group_names=[a.item() for a in cells[2].flat]
groups=np.array([group_names[int(c)-1] for c in cells[3].ravel()])
ids=full.source_row.to_numpy()-1
np.testing.assert_array_equal(ids,np.flatnonzero(conditions=='RT'))
np.testing.assert_array_equal(groups[ids],full.Group)
for name in full:
    if name in arrays and name not in ['Group','Condition']:
        np.testing.assert_allclose(full[name],arrays[name][ids],rtol=0,atol=1e-12,equal_nan=True)
assert len(selected)==11344 and len(excluded)==56 and selected.RT_s.lt(.2).sum()==4
assert selected.Valid.eq(1).all() and selected.RT_s.between(0,1.8).all()
for name,df in [('selected_trials.csv',selected),('excluded_trials.csv',excluded)]:
    np.testing.assert_array_equal(pd.read_csv(out/name).source_row,df.source_row)
    assert (out/name).read_bytes()==(BASELINE_ROOT/name).read_bytes()

# %% Check prior areas and all independent numerical edge cases.
for name,prior in PRIORS.items():
    lo,hi=prior['hard'];a,b=prior['plausible']
    height=1/((a-lo)/2+b-a+(hi-b)/2)
    integral=quad(lambda x:np.interp(x,[lo,a,b,hi],[0,height,height,0]),lo,hi,points=[a,b])[0]
    assert abs(integral-1)<1e-12
    x=np.linspace(lo+1e-7,hi-1e-7,101)
    np.testing.assert_allclose(np.exp(np.asarray(trapezoidal_logpdf_jax(x,lo,a,b,hi))),
        np.interp(x,[lo,a,b,hi],[0,height,height,0]),rtol=1e-12,atol=1e-12)

cases=np.array([[0,2.5,.5,.28,.12],[1.8,3.5,.38,.24,.18],[-3,1.4,.62,.4,.2],
                [20,15,.05,.1,.005],[0,15,.001,.1,.005],[0,15,.999,.1,.005],
                [0,.01,.5,.5,.005],[20,15,.5,0,.005],[-20,15,.95,.5,.3],
                [0,1,.5,0,.3],[1,2,.5,.5,.005],
                [1,2,.5,1.,.0005],[0,15,.001,.3,.0005],
                [2,2,.5,0.,.0005],[2,2,.5,.0005,.0005],[-20,15,.95,1.,.3]])
probe=np.array([.02,.08,.1,.12,.3,.8,1.8])
checks=[]
integral_jit=jax.jit(gaussian_delay_log_integral,static_argnames=('cumulative','K','order'))
for i,p in enumerate(cases):
    for side in [-1,1]:
        for cumulative in [False,True]:
            v=np.exp(np.asarray(integral_jit(probe,*p,side,cumulative,8,QUAD_ORDER)))
            higher=np.exp(np.asarray(integral_jit(probe,*p,side,cumulative,80,48)))
            expected=np.array([reference(t,*p,side,cumulative) for t in probe])
            error=float(np.max(np.abs(v-expected)))
            assert error < (1e-5 if cumulative else 1e-4),(i,side,cumulative,error)
            assert np.max(np.abs(v-higher)) < (1e-5 if cumulative else 1e-4)
            checks.append(dict(case=i,side=side,cumulative=cumulative,max_absolute_error=error))
    print('Independent quadrature case',i,'passed',flush=True)

# %% Verify loss-only stopping, including a new minimum resetting confirmation.
for completed,best,minimum,cap,patience,expected in [
    (25000,1000,50000,1000000,25,(False,False)),
    (50000,25000,50000,1000000,25,(True,True)),
    (50000,49000,50000,1000000,25,(False,False)),
    (74000,49000,50000,1000000,25,(True,True)),
    (1000000,999000,50000,1000000,25,(True,False))]:
    assert loss_stop_status(completed,best,minimum,cap,patience)==expected
sequence=[5.,4.,4.1,4.2,3.9,4.,4.1]
best=np.inf;best_step=0;observed=[]
for i,loss in enumerate(sequence,1):
    if loss<best: best,best_step=loss,i*1000
    observed.append((i*1000-best_step)//1000)
assert observed==[0,0,1,2,0,1,2] and best_step==5000

# %% Compare the ten-condition likelihood and chain gradients with the existing model.
import human_pooled_asl_shared_omega_svi_utils as reference34
group_checks={}
for group in ['Control','ASD']:
    sub,ct,data,initial,source=prepare_group(selected,group)
    old_sub,old_ct,old_data,_,_=reference34.prepare_group(selected,group)
    np.testing.assert_array_equal(sub.source_row,old_sub.source_row)
    np.testing.assert_array_equal(ct.signed_delta,SIGNED_DIFFERENCES)
    np.testing.assert_array_equal(ct.n,sub.groupby('signed_delta',sort=True).size())
    original=json.loads(source.read_text())
    for name in PARAMETER_BLOCKS:
        expected=np.asarray(original[name])
        if name in ['delay_mu','delay_sigma']: expected=np.mean(expected)
        np.testing.assert_array_equal(initial[name],expected)
    trace=handlers.trace(handlers.seed(handlers.substitute(gamma_omega_model,data=initial),0)).get_trace(data)
    assert sum(np.size(x['value']) for x in trace.values() if x['type']=='sample' and not x.get('is_observed',False))==16
    assert np.shape(trace['omega']['value'])==()
    for name in PARAMETER_BLOCKS:
        if name=='gamma':
            lo,hi=data['gamma_hard'];a,b=data['gamma_plausible']
        else:
            lo,hi=PRIORS[name]['hard'];a,b=PRIORS[name]['plausible']
        actual=trace[name]['fn'].log_prob(trace[name]['value']).sum()+trace[name+'_trapezoid_prior']['fn'].log_prob(trace[name+'_trapezoid_prior']['value']).sum()
        expected=trapezoidal_logpdf_jax(initial[name],lo,a,b,hi).sum()
        np.testing.assert_allclose(actual,expected,atol=1e-11)
    expanded={name:(jnp.broadcast_to(value,(10,)) if name in ['delay_mu','delay_sigma'] else jnp.asarray(value)) for name,value in initial.items()}
    score,gradient=jax.jit(jax.value_and_grad(log_likelihood))(initial,data)
    old_score,old_gradient=jax.jit(jax.value_and_grad(reference34.log_likelihood))(expanded,old_data)
    np.testing.assert_allclose(score,old_score,atol=1e-8,rtol=0)
    for name in PARAMETER_BLOCKS:
        expected=old_gradient[name]
        if name in ['delay_mu','delay_sigma']: expected=jnp.sum(expected)
        np.testing.assert_allclose(gradient[name],expected,atol=1e-8,rtol=1e-9)
    check=numerical_check(initial,data)
    group_checks[group]=dict(trials=len(sub),conditions=10,free_parameters=16,
        likelihood_broadcast_difference=float(abs(score-old_score)),priors_once=True,gradient_equivalence=True,initialization_verified=True,numerical=check)
    print(group,'initialization, 16 parameters, priors, likelihood/gradient equivalence passed',flush=True)
report=dict(status='passed',stage='preflight',parameters_per_group=16,source_rows=11344,
    loss_only_stopping_tests=True,groups=group_checks,independent_quadrature=checks,source_hashes=config['source_hashes'])
(out/'independent_preflight.json').write_text(json.dumps(report,indent=2)+'\n')
if args.preflight_only: raise SystemExit

# %% Reconstruct posterior draws and loss-only checkpoint restoration.
posterior_checks={}
for group in ['Control','ASD']:
    folder=out/group.lower();samples=dict(np.load(folder/'posterior_samples.npz'))
    matrix=sample_matrix(samples)
    assert matrix.shape==(10000,16) and np.isfinite(matrix).all()
    summary=pd.read_csv(folder/'posterior_summary.csv')
    for column,expected in [('mean',matrix.mean(0)),('sd',matrix.std(0,ddof=1)),
                            ('q025',np.quantile(matrix,.025,axis=0)),('q975',np.quantile(matrix,.975,axis=0))]:
        np.testing.assert_allclose(summary[column],expected,atol=1e-12)
    np.testing.assert_allclose(pd.read_csv(folder/'posterior_covariance.csv',index_col=0),np.cov(matrix,rowvar=False),atol=1e-12)
    np.testing.assert_allclose(pd.read_csv(folder/'posterior_correlation.csv',index_col=0),np.corrcoef(matrix,rowvar=False),atol=1e-12)
    sub,ct,data,initial,_=prepare_group(selected,group)
    trace=pd.read_csv(folder/'convergence.csv');losses=np.load(folder/'losses.npy')
    stop=json.loads((folder/'convergence_summary.json').read_text())
    np.testing.assert_array_equal(trace.step,np.arange(1000,len(losses)+1,1000))
    np.testing.assert_allclose(trace.mean_negative_elbo,losses.reshape(-1,1000).mean(1),atol=1e-10)
    best_index=int(np.argmin(trace.mean_negative_elbo));best_step=int(trace.step.iloc[best_index])
    assert stop['restored_best_step']==best_step and stop['completed_steps']==len(losses)
    assert stop['confirmation_windows']==len(trace)-1-best_index
    for i,row in trace.iterrows():
        best=int(np.argmin(trace.mean_negative_elbo.iloc[:i+1]))
        assert row.best_step==trace.step.iloc[best] and row.patience==i-best
        should_stop,confirmed=loss_stop_status(int(row.step),int(row.best_step),config['min_steps'],config['max_steps'],config['confirmation_windows'])
        if i<len(trace)-1: assert not should_stop,'Fit continued after its loss-only stop'
    assert should_stop and stop['flagged']==(not confirmed) and stop['parameter_stability_gate'] is False
    with (folder/'best_checkpoint.pkl').open('rb') as stream:best=pickle.load(stream)
    assert best['step']==best_step
    with (folder/'latest_checkpoint.pkl').open('rb') as stream:latest=pickle.load(stream)
    assert latest['step']==len(losses)==int(latest['state'].optim_state[0])
    saved_guide=dict(np.load(folder/'guide_parameters.npz'))
    for name,value in saved_guide.items(): np.testing.assert_array_equal(value,best['params'][name])
    from numpyro.infer.autoguide import AutoMultivariateNormal
    from numpyro.infer.initialization import init_to_value
    guide=AutoMultivariateNormal(guide_support_model,init_loc_fn=init_to_value(values=initial),init_scale=.1)
    handlers.seed(guide,rng_seed=config['seed'])(data)
    regenerated=guide.sample_posterior(jax.random.PRNGKey(config['seed']+100000),saved_guide,sample_shape=(10000,))
    np.testing.assert_allclose(sample_matrix(regenerated),matrix,atol=1e-11,rtol=1e-11)
    subset={k:(v[::20] if k in ['rt','choice','condition','B','D'] and np.shape(v)==(len(data['rt']),) else v)
            for k,v in data.items()}
    coarse=jax.jit(lambda p:log_likelihood(p,subset))
    refined=jax.jit(lambda p:log_likelihood(p,subset,dt=GRID_DT/2,order=48))
    draw_ids=np.random.default_rng(20261022).choice(10000,100,replace=False)
    discrepancies=[]
    for row in np.r_[[-1],draw_ids]:
        p={n:jnp.asarray(v) for n,v in vector_params(matrix.mean(0) if row==-1 else matrix[row]).items()}
        delta=abs(float(coarse(p)-refined(p)))/len(subset['rt'])
        assert delta<1e-4,(group,int(row),delta)
        t=jnp.array([.2,.4,.6,1.,1.8])[:,None,None]
        arguments=(t,p['gamma'][None,:,None],expand_shared(p)['omega'][None,:,None],p['w'],
                   expand_shared(p)['delay_mu'][None,:,None],expand_shared(p)['delay_sigma'][None,:,None],
                   jnp.array([-1,1])[None,None,:],True)
        cdf=np.exp(np.asarray(integral_jit(*arguments,8,24)))
        cdf_fine=np.exp(np.asarray(integral_jit(*arguments,80,48)))
        z=cdf[-1].sum(-1);z_fine=cdf_fine[-1].sum(-1)
        assert np.all((z>0)&(z<1+1e-6))
        conditional_error=float(np.max(abs(cdf.sum(-1)/z-cdf_fine.sum(-1)/z_fine)))
        choice_error=float(np.max(abs(cdf[-1,:,1]/z-cdf_fine[-1,:,1]/z_fine)))
        assert conditional_error<1e-5 and choice_error<1e-5
        discrepancies.append(dict(draw=int(row),mean_log_likelihood_difference=delta,
            maximum_conditional_cdf_error=conditional_error,maximum_choice_probability_error=choice_error))
    means=vector_params(matrix.mean(0))
    means_check=numerical_check(means,data)
    # Independent adaptive quadrature for every signed condition at posterior means.
    model=pd.read_csv(out/'condition_model_curves.csv')
    model=model.loc[model.Group.eq(group)]
    points=pd.read_csv(out/'psychometric_comparison.csv')
    points=points.loc[points.Group.eq(group)]
    numerical_rows=[]
    for c in ct.itertuples():
        i=c.condition_index
        pars=[float(means[n][i]) if n=='gamma' else float(means[n])
              for n in ['gamma','omega','w','delay_mu','delay_sigma']]
        end=np.array([reference(1.8,*pars,side,True) for side in [-1,1]])
        z=end.sum();p,q=means['lapse_prob'],means['lapse_prob_right']
        expected_choice=(1-p)*end[1]/z+p*q
        point=points.loc[points.condition_index.eq(i)].iloc[0]
        observed=sub.loc[sub.condition_index.eq(i)]
        assert point.n==len(observed) and point.right==observed.Choice.eq(1).sum()
        np.testing.assert_allclose(point.empirical_p_right,observed.Choice.eq(1).mean(),atol=1e-14)
        np.testing.assert_allclose(point.model_p_right,expected_choice,atol=1e-6)
        curve=model.loc[model.condition_index.eq(i)].sort_values('time_s')
        np.testing.assert_allclose(curve.density,curve.left_density+curve.right_density,atol=1e-12)
        mass=trapezoid(curve.density,curve.time_s)
        assert abs(mass-1)<3e-5,(group,i,mass)
        assert np.isfinite(curve[['density','cdf','left_density','right_density']]).all().all()
        assert curve[['density','left_density','right_density']].to_numpy().min()>=0
        assert abs(trapezoid(curve.right_density,curve.time_s)-point.model_p_right)<3e-5
        assert np.all(np.diff(curve.cdf)>-1e-8)
        for t in [.1,.2,.4,.6,1.,1.4,1.8]:
            independent_cdf=(1-p)*sum(reference(t,*pars,side,True) for side in [-1,1])/z+p*t/1.8
            independent_pdf=(1-p)*sum(reference(t,*pars,side) for side in [-1,1])/z+p/1.8
            cdf_error=abs(np.interp(t,curve.time_s,curve.cdf)-independent_cdf)
            pdf_error=abs(np.interp(t,curve.time_s,curve.density)-independent_pdf)
            assert cdf_error<1e-5 and pdf_error<1e-4,(group,i,t,cdf_error,pdf_error)
            numerical_rows.append(dict(condition_index=i,time_s=t,cdf_absolute_error=cdf_error,
                                       pdf_absolute_error=pdf_error,conditional_grid_mass=mass))
    pd.DataFrame(numerical_rows).to_csv(folder/'independent_curve_checks.csv',index=False)
    # scipy.stats supplies an independent implementation of truncated-normal moments.
    moments=pd.read_csv(out/'actual_truncated_delay_moments.csv')
    moments=moments.loc[moments.Group.eq(group)]
    mu,sd=samples['delay_mu'],samples['delay_sigma']
    actual_mu,actual_var=truncnorm.stats(-mu/sd,np.inf,loc=mu,scale=sd,moments='mv')
    for label,array in [('actual_delay_mean',actual_mu),('actual_delay_sd',np.sqrt(actual_var))]:
        saved=moments.loc[moments.quantity.eq(label)].iloc[0]
        np.testing.assert_allclose(saved.mean_s,array.mean(),atol=1e-12)
        np.testing.assert_allclose(saved.q025_s,np.quantile(array,.025),atol=1e-12)
        np.testing.assert_allclose(saved.q975_s,np.quantile(array,.975),atol=1e-12)
    posterior_checks[group]=dict(summary_reconstructed=True,checked_draws=draw_ids.tolist(),
        resolution_checks=discrepancies,posterior_mean=means_check,convergence_reconstructed=True,
        restored_guide_verified=True,independent_curves=numerical_rows,
        truncated_delay_moments_verified=True)
    pd.DataFrame(discrepancies).to_csv(folder/'posterior_numerical_checks.csv',index=False)
    print(group,'posterior checks passed',flush=True)

# %% Independently reconstruct the requested pooled panels and both bound tails.
hist=pd.read_csv(out/'pooled_rtd_histograms.csv',float_precision='round_trip')
curves=pd.read_csv(out/'pooled_model_curves.csv',float_precision='round_trip')
psych=pd.read_csv(out/'pooled_psychometric.csv',float_precision='round_trip')
chrono=pd.read_csv(out/'chronometric_points.csv',float_precision='round_trip')
weights=pd.read_csv(out/'mixture_weights.csv',float_precision='round_trip')
condition_curves=pd.read_csv(out/'condition_model_curves.csv',float_precision='round_trip')
condition_points=pd.read_csv(out/'condition_fit_quality.csv',float_precision='round_trip')
boundary=pd.read_csv(out/'posterior_boundary_audit.csv',float_precision='round_trip')
pool_checks=[]
for group in ['Control','ASD']:
    sub,ct,data,_,_=prepare_group(selected,group)
    for delta in [.1,.3,.65,1.3,2.]:
        obs=sub.loc[np.isclose(sub.signed_delta.abs(),delta)]
        h=hist.loc[hist.Group.eq(group)&np.isclose(hist.abs_delta,delta)]
        count=np.histogram(obs.RT_s,np.linspace(0,1.8,91))[0]
        np.testing.assert_array_equal(h['count'],count)
        np.testing.assert_allclose(h.density,count/(len(obs)*.02),atol=1e-12)
        w=weights.loc[weights.Group.eq(group)&weights.panel.eq('rtd')&np.isclose(weights.pooled_delta,delta)].sort_values('condition_index')
        expected=ct.loc[np.isclose(ct.signed_delta.abs(),delta)].sort_values('condition_index')
        assert len(w)==2 and abs(w.weight.sum()-1)<1e-14
        np.testing.assert_array_equal(w.component_n,expected.n)
        np.testing.assert_allclose(w.weight,expected.n/len(obs),atol=1e-14)
        mix=curves.loc[curves.Group.eq(group)&np.isclose(curves.abs_delta,delta)].sort_values('time_s')
        component_means=[]
        for column in ['density','cdf']:
            signed=np.column_stack([condition_curves.loc[condition_curves.Group.eq(group)&condition_curves.condition_index.eq(i)]
                                    .sort_values('time_s')[column] for i in w.condition_index])
            np.testing.assert_allclose(mix[column],signed@w.weight.to_numpy(),atol=1e-12)
            if column=='cdf':component_means=trapezoid(1-signed,mix.time_s,axis=0)
        mass=trapezoid(mix.density,mix.time_s)
        mean=trapezoid(mix.time_s*mix.density,mix.time_s)
        mean_cdf=trapezoid(1-mix.cdf,mix.time_s)
        assert abs(mass-1)<3e-5 and abs(mean-mean_cdf)<3e-5
        point=chrono.loc[chrono.Group.eq(group)&np.isclose(chrono.abs_delta,delta)].iloc[0]
        np.testing.assert_allclose(point.data_mean_RT_s,obs.RT_s.mean(),atol=1e-14)
        np.testing.assert_allclose(point.model_mean_RT_s,mean,atol=1e-14)
        np.testing.assert_allclose(point.model_mean_RT_s,np.dot(component_means,w.weight),atol=3e-5)
        pool_checks.append(dict(Group=group,abs_delta=delta,trials=len(obs),mass=float(mass),mean_density_cdf_gap_s=float(abs(mean-mean_cdf))))
    for delta in sorted(ct.signed_delta.unique()):
        obs=sub.loc[sub.signed_delta.eq(delta)]
        row=psych.loc[psych.Group.eq(group)&np.isclose(psych.signed_delta,delta)].iloc[0]
        cond=condition_points.loc[condition_points.Group.eq(group)&np.isclose(condition_points.signed_delta,delta)]
        w=weights.loc[weights.Group.eq(group)&weights.panel.eq('psychometric')&np.isclose(weights.pooled_delta,delta)].sort_values('condition_index')
        assert len(w)==1 and abs(w.weight.sum()-1)<1e-14
        assert row.n==len(obs) and row.right==obs.Choice.eq(1).sum()
        np.testing.assert_allclose(row.empirical_p_right,obs.Choice.eq(1).mean(),atol=1e-14)
        np.testing.assert_allclose(row.model_p_right,np.average(cond.model_p_right,weights=cond.n),atol=1e-14)
        np.testing.assert_allclose(w.weight,cond.sort_values('condition_index').n/len(obs),atol=1e-14)
    samples=dict(np.load(out/group.lower()/'posterior_samples.npz'))
    assert all(samples[n].shape==(10000,) for n in ['omega','delay_mu','delay_sigma','w','lapse_prob','lapse_prob_right'])
    matrix=sample_matrix(samples);b=boundary.loc[boundary.Group.eq(group)]
    np.testing.assert_array_equal(b.parameter,PARAMETER_NAMES)
    np.testing.assert_allclose(b.q005,np.quantile(matrix,.005,axis=0),atol=1e-12)
    np.testing.assert_allclose(b.q995,np.quantile(matrix,.995,axis=0),atol=1e-12)
    assert np.all(matrix>=b.hard_low.to_numpy()) and np.all(matrix<=b.hard_high.to_numpy())
    for qcolumn,dcolumn,lower in [('q005','lower_tail_distance_fraction',True),('q995','upper_tail_distance_fraction',False)]:
        distance=(b[qcolumn]-b.hard_low) if lower else (b.hard_high-b[qcolumn])
        np.testing.assert_allclose(b[dcolumn],distance/(b.hard_high-b.hard_low),atol=1e-12)
report['pooled_panel_checks']=pool_checks

# %% Reconstruct comparison metrics over the same ten pooled signed differences.
comparison=pd.read_csv(out/'baseline_vs_shared_gaussian_delay_fit_quality.csv')
old_curves=pd.read_csv(BASELINE_ROOT/'condition_model_curves.csv')
old_points=pd.read_csv(BASELINE_ROOT/'condition_fit_quality.csv')
old_psych=pd.read_csv(BASELINE_ROOT/'pooled_psychometric.csv')
old_chrono=pd.read_csv(BASELINE_ROOT/'chronometric_points.csv')
old_quality=pd.read_csv(BASELINE_ROOT/'fit_quality.csv')
comparison_checks=[]
for group in ['Control','ASD']:
    sub,ct,data,_,_=prepare_group(selected,group)
    for nparam,cc,cp,ps,ch in [(16,condition_curves,condition_points,psych,chrono),
                              (34,old_curves,old_points,old_psych,old_chrono)]:
        gaps=[]
        for delta in SIGNED_DIFFERENCES:
            obs=np.sort(sub.loc[sub.signed_delta.eq(delta),'RT_s'])
            cond=cp.loc[cp.Group.eq(group)&np.isclose(cp.signed_delta,delta)].sort_values('condition_index')
            arrays=[cc.loc[cc.Group.eq(group)&cc.condition_index.eq(i)].sort_values('time_s') for i in cond.condition_index]
            weights=cond.n.to_numpy()/len(obs)
            expected_cdf=np.column_stack([a.cdf for a in arrays])@weights
            model=np.interp(obs,arrays[0].time_s,expected_cdf)
            empirical=np.arange(1,len(obs)+1)/len(obs)
            gaps.append(max(abs(model-empirical).max(),abs(model-empirical+1/len(obs)).max()))
        psy=ps.loc[ps.Group.eq(group)];chro=ch.loc[ch.Group.eq(group)]
        err=psy.model_p_right-psy.empirical_p_right
        metrics=dict(psychometric_mae=float(abs(err).mean()),psychometric_rmse=float(np.sqrt(np.mean(err**2))),
            mean_condition_cdf_gap=float(np.mean(gaps)),max_condition_cdf_gap=float(max(gaps)),
            pooled_chronometric_rmse_s=float(np.sqrt(np.mean((chro.model_mean_RT_s-chro.data_mean_RT_s)**2))))
        row=comparison.loc[comparison.Group.eq(group)&comparison.free_parameters.eq(nparam)].iloc[0]
        assert row.n_trials==len(sub)
        for name,value in metrics.items():np.testing.assert_allclose(row[name],value,atol=1e-12)
        if nparam==16:
            samples=dict(np.load(out/group.lower()/'posterior_samples.npz'))
            means=vector_params(sample_matrix(samples).mean(0))
            joint=posterior_checks[group]['posterior_mean']['joint_log_likelihood']
            marginal=float(jax.jit(lambda p:log_likelihood(p,data,marginal=True))(means))
            np.testing.assert_allclose(row.joint_log_likelihood,joint,atol=1e-8)
            np.testing.assert_allclose(row.marginal_rt_log_likelihood,marginal,atol=1e-8)
        else:
            old_row=old_quality.loc[old_quality.Group.eq(group)].iloc[0]
            for name in ['joint_log_likelihood','marginal_rt_log_likelihood']:
                np.testing.assert_allclose(row[name],old_row[name],atol=1e-10)
        comparison_checks.append(dict(Group=group,free_parameters=nparam,verified=True,**metrics))
report['fit_quality_comparison_checks']=comparison_checks
report.update(stage='posterior',posterior=posterior_checks)
frozen=json.loads((out/'preservation_baseline.json').read_text())
changed=[name for name,digest in frozen.items() if not (ROOT/name).exists() or sha256(ROOT/name)!=digest]
assert not changed,changed
report['preservation']=dict(checked_files=len(frozen), unchanged_files=len(frozen)-len(changed), intended_changes=changed)
report['verifier_sha256']=sha256(Path(__file__))
(out/'independent_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print('PASS: posterior verification',flush=True)
