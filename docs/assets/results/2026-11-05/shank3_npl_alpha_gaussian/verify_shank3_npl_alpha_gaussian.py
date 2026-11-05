# %% Independent raw selection, SciPy convolution, priors and saved-fit checks
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import argparse
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import quad, cumulative_simpson, trapezoid
from scipy.signal import fftconvolve
from scipy.special import ndtr
from scipy.stats import norm, invgauss
import jax
import jax.numpy as jnp
from numpyro import handlers
from shank3_npl_alpha_gaussian_utils import *
from human_plain_ddm_diagnostics_utils import scipy_ddm

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--genotype',choices=['wt','hom','all'],default='all')
parser.add_argument('--pre-fit',action='store_true')
args=parser.parse_args()


def numpy_mapping(p):
    lam,ell,alpha=[float(p[n]) for n in ['rate_lambda','rate_norm_l','alpha']]
    abl,ild=np.asarray(ABL_ARRAY),np.asarray(ILD_ARRAY)
    gain=10**(lam*(1-ell)*abl/20)
    r=gain*np.exp(lam*ild/17.37)/(np.exp(lam*ell*ild/17.37)+alpha*np.exp(-lam*ell*ild/17.37))
    l=gain*np.exp(-lam*ild/17.37)/(np.exp(-lam*ell*ild/17.37)+alpha*np.exp(lam*ell*ild/17.37))
    return float(p['theta_E'])*(r-l)/(r+l),(r+l)/(float(p['T_0'])*float(p['theta_E'])**2)


def scipy_grids(p, dt=.000125):
    """Independent K=80 SciPy DDM + FFT + cumulative Simpson, fourfold finer."""
    times=np.linspace(0,1.2,round(1.2/dt)+1)
    gamma,omega=numpy_mapping(p)
    fs,cs=[],[]
    for i in range(30):
        h=norm.pdf((times-p['delay_mu'][i])/p['delay_sigma'][i])/p['delay_sigma'][i]/ndtr(p['delay_mu'][i]/p['delay_sigma'][i])
        h[0]*=.5
        f=np.array([np.maximum(fftconvolve(scipy_ddm(times,gamma[i],omega[i],float(p['w']),0,side=s),h)[:len(times)]*dt,0) for s in [-1,1]])
        f[:,0]=0.
        fs.append(f)
        cs.append(cumulative_simpson(f,x=times,axis=-1,initial=0))
    return times,np.array(fs),np.array(cs)


def scipy_log_likelihood(p,data,grid):
    t,f,c=grid
    result=0.
    for i in range(30):
        mask=np.asarray(data['condition'])==i
        rt=np.asarray(data['rt'])[mask]
        pdf=np.array([np.interp(rt,t,f[i,s]) for s in range(2)]).T
        cdf=np.array([np.interp(rt,t,c[i,s]) for s in range(2)]).T
        go=np.array([np.interp(rt+float(p['del_go']),t,c[i,s]) for s in range(2)]).T
        readout=.5*(1-go.sum(-1))[:,None]+go-cdf
        ordinary=np.asarray(data['proactive_pdf'])[mask,None]*readout+np.asarray(data['proactive_survival'])[mask,None]*pdf
        q=(1-float(data['lapse_prob']))*ordinary+.5*float(data['lapse_prob'])*np.asarray(data['lapse_pdf'])[mask,None]
        q=q[np.arange(len(rt)),(np.asarray(data['choice'])[mask]==1).astype(int)]
        end=sum(np.interp(1,t,c[i,s]) for s in range(2))
        z=(1-float(data['lapse_prob']))*(np.asarray(data['survival_onset'])[mask]-np.asarray(data['survival_end'])[mask]*(1-end)) \
            +float(data['lapse_prob'])*np.asarray(data['lapse_window_mass'])[mask]
        assert (q>0).all() and (z>0).all() and (z<=1).all()
        result+=np.sum(np.log(q)-np.log(z))
    return float(result)


raw=pd.read_csv(ROOT/'asd_analysis/session_type_csvs/ASD_session_type_1.csv',float_precision='round_trip',
    usecols=['filename','genotype','animal','session_type','training_level','ABL','ILD','success','response_poke','timed_rt','fix_time','intended_fix_time','abort_type'])
for group in (['wt','hom'] if args.genotype=='all' else [args.genotype]):
    folder=OUT/group
    prep=json.loads((folder/'preparation.json').read_text())
    for name,digest in prep['input_hashes'].items():
        assert sha256(folder/name)==digest
    for name,digest in prep['preservation_hashes'].items():
        assert sha256(ROOT/name)==digest,name
    valid=pd.read_csv(folder/'fit_trials.csv.gz',float_precision='round_trip')
    parent=pd.read_csv(folder/'parent_trials.csv.gz',float_precision='round_trip')
    fixed=np.array(json.loads((folder/'fixed_proactive.json').read_text())['values'])
    n=COUNTS[group]
    mask=raw.filename.eq('SHANK3_cohort1.csv') & raw.genotype.eq(group) & raw.session_type.eq(1) & raw.training_level.eq(16)
    mask &= raw.ABL.isin([20,40,59,60]) & raw.ILD.isin(np.asarray(ILD_ARRAY))
    eligible=raw.loc[mask]
    selected=eligible.loc[eligible.success.isin([-1,1]) & np.isfinite(eligible.timed_rt) & eligible.timed_rt.between(0,1)]
    np.testing.assert_array_equal(selected.index+2,valid.prepared_csv_line)
    np.testing.assert_array_equal(selected.timed_rt,valid.timed_rt)
    np.testing.assert_array_equal(selected.response_poke,valid.response_poke)
    np.testing.assert_array_equal(selected.intended_fix_time/1000,valid.onset_s)
    np.testing.assert_array_equal(eligible.index+2,parent.prepared_csv_line)
    assert len(selected)==n['valid'] and len(eligible)==n['parent'] and selected.animal.nunique()==n['animals']
    assert selected.response_poke.value_counts().to_dict()=={1.:n['right'],-1.:n['left']}
    aborts=eligible.abort_type.eq('Fixation') & (eligible.fix_time/1000-eligible.intended_fix_time/1000).between(-1,0,inclusive='left')
    np.testing.assert_array_equal(eligible.index[aborts]+2,parent.loc[parent.display_abort,'prepared_csv_line'])
    assert aborts.sum()==n['aborts']
    abort_fit=pd.read_csv(ROOT/f'asd_analysis/shank3_cohort1_proactive_lapse/{group}/seed_0/posterior_summary.csv',float_precision='round_trip').set_index('parameter')
    np.testing.assert_array_equal(fixed,abort_fit.loc[['V_A','theta_A','t_nd','lapse_prob','beta_lapse'],'mean'])
    initial,rng=initialization()
    saved=json.loads((folder/'initialization.json').read_text())
    for key,value in initial.items(): np.testing.assert_array_equal(value,saved[key])
    assert rng==json.loads((folder/'initialization_rng.json').read_text())
    for key,value in initial.items(): assert np.all((value>PRIORS[key][1])&(value<PRIORS[key][2]))
    trace=handlers.trace(handlers.seed(handlers.substitute(model,data=initial),rng_seed=0)).get_trace(None)
    sites={k:s for k,s in trace.items() if s['type']=='sample' and not s.get('is_observed',False)}
    assert sum(np.size(x['value']) for x in sites.values())==67
    for key,site in sites.items():
        lo,pl,ph,hi=PRIORS[key]
        value=np.asarray(site['value'])
        shape=np.minimum(np.minimum((value-lo)/(pl-lo),(hi-value)/(hi-ph)),1)
        expected=np.log(shape/((pl-lo+hi-ph)/2+ph-pl)).sum()
        actual=site['fn'].log_prob(site['value'])+trace[key+'_trapezoid']['fn'].log_prob(trace[key+'_trapezoid']['value'])
        np.testing.assert_allclose(actual,expected,atol=1e-12)
    assert loss_window_decision(9,10,50000,24)==(True,0,False)
    assert loss_window_decision(10,10,50000,24)==(False,25,True)
    assert loss_window_decision(11,10,49000,25)==(False,26,False)
    data={k:jnp.asarray(v) for k,v in np.load(folder/'likelihood_data.npz').items() if k!='onset'}
    if not args.pre_fit:
        initial={k:v.mean(0) for k,v in np.load(folder/'seed_0/posterior_samples.npz').items()}
    params={k:jnp.asarray(v) for k,v in initial.items()}
    gamma,omega=numpy_mapping(initial)
    actual=expanded(params)
    np.testing.assert_allclose(actual['gamma'],gamma,atol=1e-12)
    np.testing.assert_allclose(actual['omega'],omega,atol=1e-12)
    value,grad=jax.jit(jax.value_and_grad(log_likelihood))(params,data)
    assert np.isfinite(value) and all(np.isfinite(x).all() for x in grad.values())
    print(group,'full-data gradients finite; LL',float(value),flush=True)
    grid=scipy_grids(initial)
    independent=scipy_log_likelihood(initial,data,grid)
    per_trial=abs(float(value)-independent)/n['valid']
    assert per_trial<1e-4,(group,float(value),independent,per_trial)
    fine=float(jax.jit(lambda p:log_likelihood(p,data,dt=.00025,order=48))(params))
    assert abs(fine-float(value))/n['valid']<1e-4
    # Independent adaptive delay quadrature at all three ABLs and short/long RTs.
    direct=[]
    for i in [0,4,9,10,14,19,20,24,29]:
        mu,sd=float(initial['delay_mu'][i]),float(initial['delay_sigma'][i])
        g,o,w=gamma[i],omega[i],float(initial['w'])
        for t in [.035,.12,.35,1.]:
            for side in [-1,1]:
                scipy_value=quad(lambda d:float(scipy_ddm(t-d,g,o,w,0,side=side))*norm.pdf((d-mu)/sd)/sd/ndtr(mu/sd),
                    0,t,epsabs=1e-9,epsrel=1e-7,points=[x for x in [mu-3*sd,mu,mu+3*sd] if 0<x<t],limit=150)[0]
                kernel=float(jnp.exp(gaussian_delay_log_integral(t,g,o,w,mu,sd,side,False,40,48)))
                np.testing.assert_allclose(kernel,scipy_value,atol=1e-8,rtol=1e-5)
                direct.append(abs(kernel-scipy_value))
    # Gaussian support and the deterministic-delay limit are checked apart
    # from the fitted sigma bounds; this is a mathematical limit, not a prior.
    support_checks=[]
    for mu,sd in [(0.,.005),(.01,.15),(.5,.3)]:
        area=quad(lambda d:norm.pdf((d-mu)/sd)/sd/ndtr(mu/sd),0,np.inf,epsabs=1e-10)[0]
        np.testing.assert_allclose(area,1.,atol=1e-9)
        support_checks.append(dict(mu=mu,sigma=sd,area=area))
    for t in [.03,.12,.35,.8]:
        for side in [-1,1]:
            small=float(jnp.exp(gaussian_delay_log_integral(t,1.2,5.,.47,.08,1e-5,side,False,40,48)))
            point=float(scipy_ddm(t,1.2,5.,.47,.08,side=side))
            np.testing.assert_allclose(small,point,atol=1e-6,rtol=1e-5)
            small_cdf=float(jnp.exp(gaussian_delay_log_integral(t,1.2,5.,.47,.08,1e-5,side,True,40,48)))
            point_cdf=float(scipy_ddm(t,1.2,5.,.47,.08,side=side,cumulative=True))
            np.testing.assert_allclose(small_cdf,point_cdf,atol=1e-6,rtol=1e-5)
    normalization=[]
    ts,fs,cs=grid
    window=ts<=1.
    v,a,nd,lp,beta=fixed
    for i,c0 in [(0,.4),(14,.8),(29,1.2)]:
        t=ts[window]
        se=1-cs[i,:,window].sum(-1)
        ep=fs[i,:,window].sum(-1)
        # Advanced indexing returns (time,choice), so sum over the last axis.
        for probability in [0.,lp,1.]:
            ap=invgauss.pdf(c0+t-nd,mu=1/(v*a),scale=a*a)
            sa=invgauss.sf(c0+t-nd,mu=1/(v*a),scale=a*a)
            q=(1-probability)*(ap*se+sa*ep)+probability*beta*np.exp(-beta*(c0+t))
            sa0=invgauss.sf(c0-nd,mu=1/(v*a),scale=a*a)
            z=(1-probability)*(sa0-sa[-1]*se[-1])+probability*(np.exp(-beta*c0)-np.exp(-beta*(c0+1)))
            integral=float(trapezoid(q,t))
            np.testing.assert_allclose(integral,z,atol=2e-6,rtol=2e-6)
            normalization.append(dict(condition=i,onset=c0,lapse_prob=probability,mass=z,integrated_mass=integral))
    report=dict(status='passed',genotype=group,**n,parameter_count=67,conditions=30,
        joint_log_likelihood=float(value),independent_scipy_log_likelihood=independent,
        mean_log_likelihood_disagreement=per_trial,fine_log_likelihood=fine,
        fine_mean_log_likelihood_disagreement=abs(fine-float(value))/n['valid'],
        scipy_adaptive_quadrature_cases=len(direct),maximum_direct_density_error=max(direct),
        finite_gradients=True,prior_multiplicities=True,random_initialization_reproduced=True,
        gaussian_support_checks=support_checks,vanishing_sigma_limit='passed',normalization=normalization,
        stop_reset_tie_minimum_tests='passed',verifier_sha256=sha256(Path(__file__)))
    if not args.pre_fit:
        config=json.loads((folder/'seed_0/configuration.json').read_text())
        assert config['parameter_count']==67 and config['backend']=='gpu' and config['float64']
        for name,digest in config['source_hashes'].items():
            assert sha256(ROOT/name)==digest,name
        samples=dict(np.load(folder/'seed_0/posterior_samples.npz'))
        matrix=sample_matrix(samples)
        assert matrix.shape==(10000,67) and np.isfinite(matrix).all()
        summary=pd.read_csv(folder/'seed_0/posterior_summary.csv',float_precision='round_trip')
        np.testing.assert_array_equal(summary.parameter,parameter_names())
        np.testing.assert_allclose(summary['mean'],matrix.mean(0),atol=1e-12)
        np.testing.assert_allclose(summary.q025,np.quantile(matrix,.025,axis=0),atol=1e-12)
        np.testing.assert_allclose(summary.q975,np.quantile(matrix,.975,axis=0),atol=1e-12)
        run=json.loads((folder/'seed_0/run_summary.json').read_text())
        convergence=pd.read_csv(folder/'seed_0/convergence.csv',float_precision='round_trip')
        assert run['restored_best_step']==int(convergence.loc[convergence.mean_negative_elbo.idxmin(),'step'])
        assert run['confirmation_windows']>=25 or run['status']=='inconclusive'
        assert len(np.load(folder/'seed_0/losses.npy'))==run['final_checked_step']
        best=pickle.loads((folder/'seed_0/restored_guide_optimizer.pkl').read_bytes())
        assert best['step']==run['restored_best_step']
        from numpyro.infer.autoguide import AutoMultivariateNormal
        from numpyro.infer import init_to_value
        init,_=initialization()
        guide=AutoMultivariateNormal(model,init_loc_fn=init_to_value(values=init),init_scale=.1)
        handlers.seed(guide,rng_seed=0)(None)
        reproduced=guide.sample_posterior(jax.random.PRNGKey(100000),best['params'],sample_shape=(10000,))
        for name in samples:
            np.testing.assert_allclose(reproduced[name],samples[name],atol=1e-12,rtol=1e-12)
        report['posterior_summaries']=67
        report['checkpoint_restoration_verified']=True
        report['posterior_draws_reproduced']=True
    name='prefit_validation.json' if args.pre_fit else 'independent_validation.json'
    (folder/name).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
