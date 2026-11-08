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
from human_shared_omega_gaussian_delay_svi_utils import *
from human_gaussian_delay_svi_utils_wide import gaussian_delay_log_integral, gaussian_delay_trial_logpdf
from human_gamma_omega_variants_svi_utils import trapezoidal_logpdf_jax
from human_plain_ddm_diagnostics_utils import scipy_ddm

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--preflight-only',action='store_true')
parser.add_argument('--output-dir',type=Path,default=OUTPUT)
args=parser.parse_args();out=args.output_dir
config=json.loads((out/'run_config.json').read_text())
assert config['free_parameters_per_group']==64 and config['priors']==PRIORS
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

group_checks={}
for group in ['Control','ASD']:
    sub,ct,data,initial,source=prepare_group(selected,group)
    assert len(sub)==(6574 if group=='Control' else 4770)
    assert sub.Participant.nunique()==(11 if group=='Control' else 8)
    np.testing.assert_allclose(ct[['ASL_cd','signed_delta']],[(a,d) for a in [20,40]
        for d in [-2,-1.3,-.65,-.3,-.1,.1,.3,.65,1.3,2]],atol=1e-12)
    trace=handlers.trace(handlers.seed(handlers.substitute(gamma_omega_model,data=initial),0)).get_trace(data)
    assert sum(np.size(x['value']) for x in trace.values() if x['type']=='sample' and not x.get('is_observed',False))==64
    assert np.shape(trace['omega']['value']) == ()
    prior = PRIORS['omega']; lo, hi = prior['hard']; a, b = prior['plausible']
    omega_prior = (trace['omega']['fn'].log_prob(trace['omega']['value'])
                   + trace['omega_trapezoid_prior']['fn'].log_prob(trace['omega_trapezoid_prior']['value']))
    np.testing.assert_allclose(omega_prior,
        trapezoidal_logpdf_jax(initial['omega'], lo, a, b, hi), atol=1e-12)
    expanded=expand_omega(initial)
    scalar_score,scalar_grad=jax.jit(jax.value_and_grad(log_likelihood))(initial,data)
    vector_score,vector_grad=jax.jit(jax.value_and_grad(wide.log_likelihood))(expanded,data)
    np.testing.assert_allclose(scalar_score,vector_score,atol=1e-9,rtol=0)
    np.testing.assert_allclose(scalar_grad['omega'],np.sum(vector_grad['omega']),atol=1e-8,rtol=1e-10)
    for name in PARAMETER_BLOCKS:
        if name!='omega':np.testing.assert_allclose(scalar_grad[name],vector_grad[name],atol=1e-8,rtol=1e-10)
    original=json.loads(source.read_text())
    for n in ['gamma','delay_mu','delay_sigma','w','lapse_prob','lapse_prob_right']:
        np.testing.assert_array_equal(initial[n],original[n])
    np.testing.assert_allclose(initial['omega'],np.mean(original['omega']),rtol=0,atol=0)
    np.testing.assert_array_equal(initial['delay_sigma'],np.full(20,.05))
    # A representative subset contains both choices from every signed condition.
    positions=np.r_[tuple(np.flatnonzero(np.asarray(data['condition'])==i)[::35] for i in range(20))]
    small={k:(v[positions] if k in ['rt','choice','condition','B','D'] and np.shape(v)==(len(sub),) else v)
           for k,v in data.items()}
    p={n:jnp.asarray(v) for n,v in initial.items()}
    grid=jnp.linspace(0,1.8,3601)
    joint,logmass=gaussian_delay_trial_logpdf(expand_omega(p),small,grid)
    no_lapse=float(log_likelihood(dict(p,lapse_prob=0.),small))
    pure_lapse=float(log_likelihood(dict(p,lapse_prob=1.),small))
    np.testing.assert_allclose(no_lapse,np.asarray(joint).sum(),atol=1e-8)
    expected=np.log(np.where(np.asarray(small['choice'])==1,.5,.5)/1.8).sum()
    np.testing.assert_allclose(pure_lapse,expected,atol=1e-8)
    expected_logs=[]
    for t,y,c in zip(np.asarray(small['rt']),np.asarray(small['choice']),np.asarray(small['condition'])):
        pars=[float(p[n][c]) if n not in ['w','omega'] else float(p[n]) for n in ['gamma','omega','w','delay_mu','delay_sigma']]
        f=reference(t,*pars,y)
        z=sum(reference(1.8,*pars,side,True) for side in [-1,1])
        expected_logs.append(np.log(.98*f/z+.02*.5/1.8))
    calculated=float(log_likelihood(p,small))
    ll_error=abs(calculated-sum(expected_logs))/len(positions)
    assert ll_error<1e-4
    # Gradients independently differenced for a bound-sensitive and a typical case.
    grad_checks=[]
    for pars in [cases[1],cases[4]]:
        x=jnp.asarray(pars)
        fun=lambda v:gaussian_delay_log_integral(.12,*v,1,False,8,QUAD_ORDER)
        g=np.asarray(jax.jit(jax.grad(fun))(x));fd=[]
        for j in range(5):
            h=min(1e-6,pars[j]*.001) if j in [1,2,4] else 1e-6
            up=pars.copy();down=pars.copy();up[j]+=h;down[j]-=h
            fd.append((np.log(reference(.12,*up,1))-np.log(reference(.12,*down,1)))/(2*h))
        np.testing.assert_allclose(g,fd,atol=1e-4,rtol=1e-3)
        grad_checks.append(dict(params=pars.tolist(),max_absolute_error=float(np.max(np.abs(g-fd)))))
    group_checks[group]=dict(source_rows=len(sub),parameters=64,subset_size=len(positions),
        independent_mean_log_likelihood_error=ll_error,zero_lapse_recovery=True,pure_lapse_recovery=True,
        gradients=grad_checks)
    print(group,'independent source, likelihood and gradients passed',flush=True)

report=dict(status='passed',stage='preflight',source_rows=11344,parameters_per_group=64,
            independent_quadrature=checks,groups=group_checks,source_hashes=config['source_hashes'])
(out/'independent_preflight.json').write_text(json.dumps(report,indent=2)+'\n')
if args.preflight_only:
    print('PASS: independent preflight',flush=True)
    raise SystemExit

# %% Posterior reconstruction and 100 deterministic posterior numerical checks.
posterior_checks={}
for group in ['Control','ASD']:
    folder=out/group.lower();samples=dict(np.load(folder/'posterior_samples.npz'))
    matrix=sample_matrix(samples)
    assert matrix.shape==(10000,64) and np.isfinite(matrix).all()
    summary=pd.read_csv(folder/'posterior_summary.csv')
    np.testing.assert_allclose(summary['mean'],matrix.mean(0),atol=1e-12)
    np.testing.assert_allclose(summary.sd,matrix.std(0,ddof=1),atol=1e-12)
    np.testing.assert_allclose(summary.q025,np.quantile(matrix,.025,axis=0),atol=1e-12)
    np.testing.assert_allclose(summary.q975,np.quantile(matrix,.975,axis=0),atol=1e-12)
    np.testing.assert_allclose(pd.read_csv(folder/'posterior_covariance.csv',index_col=0),np.cov(matrix,rowvar=False),atol=1e-12)
    np.testing.assert_allclose(pd.read_csv(folder/'posterior_correlation.csv',index_col=0),np.corrcoef(matrix,rowvar=False),atol=1e-12)
    sub,ct,data,_,_=prepare_group(selected,group)
    # Reconstruct the stopping decision and restoration from every saved loss window.
    trace=pd.read_csv(folder/'convergence.csv');losses=np.load(folder/'losses.npy')
    previous=folder/'before_compilation_cache'
    if previous.exists():
        old_losses=np.load(previous/'losses.npy')
        np.testing.assert_array_equal(losses[:len(old_losses)],old_losses)
        old_trace=pd.read_csv(previous/'convergence.csv')
        pd.testing.assert_frame_equal(trace.iloc[:len(old_trace)],old_trace)
        with (previous/'latest_checkpoint.pkl').open('rb') as stream:old_state=pickle.load(stream)
        assert old_state['step']==len(old_losses)==int(old_state['state'].optim_state[0])
        resume=json.loads((out/'compilation_cache_resume.json').read_text())
        assert resume['groups'][group.lower()]['resume_step']==len(old_losses)
    previous=folder/'before_gpu'
    if previous.exists():
        old_losses=np.load(previous/'losses.npy')
        np.testing.assert_array_equal(losses[:len(old_losses)],old_losses)
        old_trace=pd.read_csv(previous/'convergence.csv')
        pd.testing.assert_frame_equal(trace.iloc[:len(old_trace)],old_trace)
        with (previous/'latest_checkpoint.pkl').open('rb') as stream:old_state=pickle.load(stream)
        resume=json.loads((out/'gpu_resume.json').read_text())
        assert old_state['step']==len(old_losses)==int(old_state['state'].optim_state[0])==resume['groups'][group]['resume_step']
        assert sha256(previous/'latest_checkpoint.pkl')==resume['groups'][group]['latest_checkpoint_sha256']
        old_config=json.loads((out/'source_history/before_gpu_run_config.json').read_text())
        assert old_config['platform']=='cpu' and config['platform']=='gpu'
        assert {k:v for k,v in old_config.items() if k not in ('platform','source_hashes')}=={
            k:v for k,v in config.items() if k not in ('platform','source_hashes')}
        fitter='ASD_Human_Data/fit_human_shared_omega_gaussian_delay_svi.py'
        assert sha256(out/'source_history/fit_human_shared_omega_gaussian_delay_svi_cpu.py')==old_config['source_hashes'][fitter]
        assert {k:v for k,v in old_config['source_hashes'].items() if k!=fitter}=={
            k:v for k,v in config['source_hashes'].items() if k!=fitter}
        benchmark=json.loads((out/'cpu_gpu_benchmark.json').read_text())
        assert benchmark['passed'] and benchmark['cpu']['x64'] and benchmark['gpu']['x64']
        assert json.loads((folder/'execution.json').read_text())['backend']=='gpu'
    previous=folder/'before_compiled_loop'
    if previous.exists():
        old_losses=np.load(previous/'losses.npy')
        np.testing.assert_array_equal(losses[:len(old_losses)],old_losses)
        old_trace=pd.read_csv(previous/'convergence.csv')
        pd.testing.assert_frame_equal(trace.iloc[:len(old_trace)],old_trace)
        with (previous/'latest_checkpoint.pkl').open('rb') as stream:old_state=pickle.load(stream)
        resume=json.loads((out/'compiled_loop_resume.json').read_text())
        assert old_state['step']==len(old_losses)==int(old_state['state'].optim_state[0])==resume['groups'][group]['resume_step']
        assert sha256(previous/'latest_checkpoint.pkl')==resume['groups'][group]['latest_checkpoint_sha256']
        old_config=json.loads((out/'source_history/before_compiled_loop_run_config.json').read_text())
        assert {k:v for k,v in old_config.items() if k!='source_hashes'}=={k:v for k,v in config.items() if k!='source_hashes'}
        fitter='ASD_Human_Data/fit_human_shared_omega_gaussian_delay_svi.py'
        assert sha256(out/'source_history/fit_human_shared_omega_gaussian_delay_svi_before_compiled_loop.py')==old_config['source_hashes'][fitter]
        assert {k:v for k,v in old_config['source_hashes'].items() if k!=fitter}=={
            k:v for k,v in config['source_hashes'].items() if k!=fitter}
        assert json.loads((out/'gpu_compiled_loop_benchmark.json').read_text())[group]['passed']
    stop=json.loads((folder/'convergence_summary.json').read_text())
    if stop['stop_reason']=='user_requested_loss_plateau_restore_best':
        request=json.loads((out/'user_requested_finalize.json').read_text())
        assert request['group']==group and request['completed_steps']==len(losses)
        assert sha256(out/'run_config.json')==request['original_run_config_sha256']
        assert sha256(out/'user_requested_finalize.json')==stop['user_request_record_sha256']
        assert sha256(ROOT/stop['finalization_source'])==stop['finalization_source_sha256']
        for name,digest in request['checkpoint_hashes'].items():
            assert sha256(folder/'before_user_requested_finalize'/name)==digest,name
            assert sha256(folder/name)==digest,name
        assert request['restored_best_step']==stop['restored_best_step']
        assert not stop['terminal_parameter_stable'] and stop['flagged']
    np.testing.assert_array_equal(trace.step,np.arange(1000,len(losses)+1,1000))
    np.testing.assert_allclose(trace.mean_negative_elbo,losses.reshape(-1,1000).mean(1),atol=1e-10)
    best_index=int(np.argmin(trace.mean_negative_elbo));best_step=int(trace.step.iloc[best_index])
    assert stop['restored_best_step']==best_step
    assert stop['completed_steps']==len(losses)
    assert stop['confirmation_windows']==(len(trace)-1-best_index)
    for i,row in trace.iterrows():
        running_best=int(np.argmin(trace.mean_negative_elbo.iloc[:i+1]))
        assert row.best_step==trace.step.iloc[running_best] and row.patience==i-running_best
        if i>=2:
            means=trace.loc[i-2:i,[n+'_mean' for n in PARAMETER_NAMES]].to_numpy()
            sds=trace.loc[i-2:i,[n+'_sd' for n in PARAMETER_NAMES]].to_numpy()
            change=np.ptp(means,axis=0)/sds.min(0)
            np.testing.assert_allclose(row.maximum_mean_change_sd,change.max(),atol=1e-9)
            assert bool(row.parameter_stable)==bool(np.all(change<.1))
    confirmed=len(losses)>=config['min_steps'] and stop['confirmation_windows']>=25 and bool(trace.parameter_stable.iloc[-1])
    assert stop['flagged']==(not confirmed)
    with (folder/'best_checkpoint.pkl').open('rb') as f:best=pickle.load(f)
    assert best['step']==best_step
    saved_guide=dict(np.load(folder/'guide_parameters.npz'))
    for name,value in saved_guide.items():np.testing.assert_array_equal(value,best['params'][name])
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
        arguments=(t,p['gamma'][None,:,None],expand_omega(p)['omega'][None,:,None],p['w'],
                   p['delay_mu'][None,:,None],p['delay_sigma'][None,:,None],
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
        pars=[float(means[n][i]) if n not in ['w','omega'] else float(means[n])
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
    # Recreate raw 20 ms histograms and signed-condition mixture weights.
    hist=pd.read_csv(out/'rtd_histograms.csv');mixed=pd.read_csv(out/'rtd_model_curves.csv')
    for level in [20,40]:
        for delta in [.1,.3,.65,1.3,2.]:
            trials=sub.loc[sub.ASL_cd.eq(level)&np.isclose(sub.signed_delta.abs(),delta)]
            h=hist.loc[hist.Group.eq(group)&hist.ASL_cd.eq(level)&np.isclose(hist.abs_delta,delta,atol=1e-12,rtol=0)]
            count=np.histogram(trials.RT_s,np.linspace(0,1.8,91))[0]
            np.testing.assert_array_equal(h['count'],count)
            np.testing.assert_allclose(h.density,count/(len(trials)*.02),atol=1e-13)
            m=mixed.loc[mixed.Group.eq(group)&mixed.ASL_cd.eq(level)&np.isclose(mixed.abs_delta,delta,atol=1e-12,rtol=0)]
            pair=ct.loc[ct.ASL_cd.eq(level)&np.isclose(ct.signed_delta.abs(),delta)].sort_values('signed_delta')
            weights=pair.n.to_numpy()/len(trials)
            np.testing.assert_allclose(m.negative_weight,weights[0],atol=1e-14)
            np.testing.assert_allclose(m.positive_weight,weights[1],atol=1e-14)
            for column in ['density','cdf']:
                signed=np.column_stack([model.loc[model.condition_index.eq(i)].sort_values('time_s')[column]
                                        for i in pair.condition_index])
                np.testing.assert_allclose(m[column],signed@weights,atol=1e-12)
    # scipy.stats supplies an independent implementation of truncated-normal moments.
    moments=pd.read_csv(out/'actual_truncated_delay_moments.csv')
    moments=moments.loc[moments.Group.eq(group)]
    mu,sd=samples['delay_mu'],samples['delay_sigma']
    actual_mu,actual_var=truncnorm.stats(-mu/sd,np.inf,loc=mu,scale=sd,moments='mv')
    for label,array in [('actual_delay_mean',actual_mu),('actual_delay_sd',np.sqrt(actual_var))]:
        saved=moments.loc[moments.quantity.eq(label)].sort_values('condition_index')
        np.testing.assert_allclose(saved.mean_s,array.mean(0),atol=1e-12)
        np.testing.assert_allclose(saved.q025_s,np.quantile(array,.025,axis=0),atol=1e-12)
        np.testing.assert_allclose(saved.q975_s,np.quantile(array,.975,axis=0),atol=1e-12)
    posterior_checks[group]=dict(summary_reconstructed=True,checked_draws=draw_ids.tolist(),
        resolution_checks=discrepancies,posterior_mean=means_check,convergence_reconstructed=True,
        restored_guide_verified=True,independent_curves=numerical_rows,
        histograms_and_mixture_weights_verified=True,truncated_delay_moments_verified=True)
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
        assert len(w)==4 and abs(w.weight.sum()-1)<1e-14
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
        assert len(w)==2 and abs(w.weight.sum()-1)<1e-14
        assert row.n==len(obs) and row.right==obs.Choice.eq(1).sum()
        np.testing.assert_allclose(row.empirical_p_right,obs.Choice.eq(1).mean(),atol=1e-14)
        np.testing.assert_allclose(row.model_p_right,np.average(cond.model_p_right,weights=cond.n),atol=1e-14)
        np.testing.assert_allclose(w.weight,cond.sort_values('condition_index').n/len(obs),atol=1e-14)
    samples=dict(np.load(out/group.lower()/'posterior_samples.npz'))
    assert samples['omega'].shape==(10000,)
    matrix=sample_matrix(samples);b=boundary.loc[boundary.Group.eq(group)]
    np.testing.assert_array_equal(b.parameter,PARAMETER_NAMES)
    np.testing.assert_allclose(b.q005,np.quantile(matrix,.005,axis=0),atol=1e-12)
    np.testing.assert_allclose(b.q995,np.quantile(matrix,.995,axis=0),atol=1e-12)
    assert np.all(matrix>=b.hard_low.to_numpy()) and np.all(matrix<=b.hard_high.to_numpy())
    for qcolumn,dcolumn,lower in [('q005','lower_tail_distance_fraction',True),('q995','upper_tail_distance_fraction',False)]:
        distance=(b[qcolumn]-b.hard_low) if lower else (b.hard_high-b[qcolumn])
        np.testing.assert_allclose(b[dcolumn],distance/(b.hard_high-b.hard_low),atol=1e-12)
report['pooled_panel_checks']=pool_checks
report.update(stage='posterior',posterior=posterior_checks)
frozen=json.loads((out/'preservation_baseline.json').read_text())
changed=[name for name,digest in frozen.items() if not (ROOT/name).exists() or sha256(ROOT/name)!=digest]
assert not set(changed)-{'ASD_Human_Data/README.md'},changed
report['preservation']=dict(checked_files=len(frozen), unchanged_files=len(frozen)-len(changed), intended_changes=changed)
report['verifier_sha256']=sha256(Path(__file__))
(out/'independent_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print('PASS: posterior verification',flush=True)
