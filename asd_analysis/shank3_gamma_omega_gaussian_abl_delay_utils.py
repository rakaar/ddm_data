# %% Direct condition-specific evidence and ABL-shared Gaussian sensory delays
"""68 parameters: 30 Gamma, 30 Omega, 3 mu, 3 sigma, shared w and del_go.

Reuse the accepted Gaussian convolution and rat race, without imposing the
NPL+alpha mapping. Delay marginalization precedes complete-mixture conditioning.
"""
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import shank3_npl_alpha_gaussian_utils as reference
from shank3_npl_alpha_gaussian_utils import (ROOT, COUNTS, GRID_DT, GRID_END,
    K_DEFAULT, QUAD_ORDER, gaussian_delay_log_integral, sha256, loss_window_decision)
from shank3_wt_gamma_omega_utils import ABLS, ILDS
from proactive_lapse_no_led_utils import trapezoid_logpdf

OUT = ROOT/'asd_analysis/shank3_cohort1_wt_hom_gamma_omega_gaussian_delay_by_abl'
REFERENCES = {g:ROOT/'asd_analysis/shank3_cohort1_wt_hom_npl_alpha_gaussian_delay_by_abl'/g for g in ['wt','hom']}
GLOBALS = ['w','del_go']
BLOCKS = ['gamma','omega','delay_mu','delay_sigma','w','del_go']
SIZES = dict(gamma=30,omega=30,delay_mu=3,delay_sigma=3,w=None,del_go=None)
ABL_ARRAY = jnp.repeat(jnp.asarray(ABLS,dtype=float),10)
ILD_ARRAY = jnp.tile(jnp.asarray(ILDS,dtype=float),3)
ABL_INDEX = jnp.repeat(jnp.arange(3),10)
PARAMETER_COUNT = 68
PRIORS = dict(gamma=np.where(np.asarray(ILD_ARRAY)[None,:]>0,
    np.array([-1.,0.,12.,20.])[:,None],np.array([-20.,-12.,0.,1.])[:,None]).tolist(),
    omega=[.01,.1,12.,15.],delay_mu=[0.,.01,.20,.50],
    delay_sigma=[.005,.01,.15,.30],w=[0.,.4,.6,1.],del_go=[0.,.02,.199,.20])


def initialization(seed=0):
    rng=np.random.default_rng(seed)
    before=rng.bit_generator.state
    params={}
    for name in BLOCKS:
        lo,pl,ph,hi=np.asarray(PRIORS[name])
        params[name]=rng.uniform(pl,ph,size=SIZES[name])
    return params,dict(generator='numpy.default_rng/PCG64',seed=seed,before=before,after=rng.bit_generator.state)


def parameter_names():
    return [f'{n}_{i:02d}' for n in ['gamma','omega'] for i in range(30)]+[
        f'{n}_ABL{abl}' for n in ['delay_mu','delay_sigma'] for abl in ABLS]+GLOBALS


def sample_matrix(samples):
    return np.column_stack([np.asarray(samples[n]).reshape(len(samples[n]),-1) for n in BLOCKS])


def expand_delay_parameters(params):
    mu,sd=jnp.asarray(params['delay_mu']),jnp.asarray(params['delay_sigma'])
    # Expanded arrays are also accepted for independent chain-gradient checks.
    assert mu.shape==sd.shape and mu.shape in [(3,),(30,)]
    return {**params,'delay_mu':mu[ABL_INDEX] if mu.size==3 else mu,
            'delay_sigma':sd[ABL_INDEX] if sd.size==3 else sd}


expanded = expand_delay_parameters

def evidence_grid(params,dt=GRID_DT,K=K_DEFAULT,order=QUAD_ORDER):
    time=jnp.linspace(0.,reference.GRID_END,round(reference.GRID_END/dt)+1)
    p=expanded(params)
    pdf,cdf=reference.gaussian_delay_grid(p,time,K)
    fast=reference.needs_time_rescaling(p['gamma'],p['omega'],p['w'],dt)
    def refine(_):
        fine_time=jnp.linspace(0.,reference.GRID_END,round(reference.GRID_END/dt)*5+1)
        pf,cf=reference.gaussian_delay_grid(p,fine_time,K)
        f=jnp.where(fast[:,None,None],pf[:,:,::5],pdf)
        c=jnp.where(fast[:,None,None],cf[:,:,::5],cdf)
        finer=fast & reference.needs_time_rescaling(p['gamma'],p['omega'],p['w'],dt/5)
        def at25(_):
            t25=jnp.linspace(0.,reference.GRID_END,round(reference.GRID_END/dt)*25+1)
            p25,c25=reference.gaussian_delay_grid(p,t25,K)
            return jnp.where(finer[:,None,None],p25[:,:,::25],f),jnp.where(finer[:,None,None],c25[:,:,::25],c)
        f,c=jax.lax.cond(jnp.any(finer),at25,lambda _:(f,c),None)
        unresolved=finer & reference.needs_time_rescaling(p['gamma'],p['omega'],p['w'],dt/25)
        def quadrature(_):
            def condition(args):
                g,o,mu,sd,ff,cc,need=args
                def integrate(_):
                    def at(side):
                        def integrals(t):
                            return jnp.exp(reference.gaussian_delay_log_integral(t,g,o,p['w'],mu,sd,side,False,K,order)), \
                                   jnp.exp(reference.gaussian_delay_log_integral(t,g,o,p['w'],mu,sd,side,True,K,order))
                        return jax.checkpoint(integrals)(time)
                    fm,cm=at(-1);fp,cp=at(1)
                    return jnp.stack([fm,fp]),jnp.stack([cm,cp])
                return jax.lax.cond(need,integrate,lambda _:(ff,cc),None)
            return jax.lax.map(condition,(p['gamma'],p['omega'],p['delay_mu'],p['delay_sigma'],f,c,unresolved))
        return jax.lax.cond(jnp.any(unresolved),quadrature,lambda _:(f,c),None)
    pdf,cdf=jax.lax.cond(jnp.any(fast),refine,lambda _:(pdf,cdf),None)
    return time,pdf,cdf


def log_likelihood(params,data,marginal=False,dt=reference.GRID_DT,K=reference.K_DEFAULT,order=reference.QUAD_ORDER):
    time,pdf,cdf=evidence_grid(params,dt,K,order)
    ids,rt=data['condition'],data['rt']
    ef=reference.interpolate(pdf,time,rt,ids)
    ec=reference.interpolate(cdf,time,rt,ids)
    eg=reference.interpolate(cdf,time,rt+params['del_go'],ids)
    readout=.5*(1-eg.sum(-1))[...,None]+eg-ec
    ordinary=data['proactive_pdf'][...,None]*readout+data['proactive_survival'][...,None]*ef
    joint=(1-data['lapse_prob'])*ordinary+.5*data['lapse_prob']*data['lapse_pdf'][...,None]
    observed=joint.sum(-1) if marginal else joint[jnp.arange(rt.size),(data['choice']==1).astype(int)]
    p=expanded(params)
    endpoint=jnp.exp(reference.gaussian_delay_log_integral(1.,p['gamma'][:,None],p['omega'][:,None],p['w'],
        p['delay_mu'][:,None],p['delay_sigma'][:,None],jnp.array([-1,1])[None,:],True,K,order)).sum(-1)
    mass=(1-data['lapse_prob'])*(data['survival_onset']-data['survival_end']*(1-endpoint[ids]))+data['lapse_prob']*data['lapse_window_mass']
    return jnp.sum(jnp.log(jnp.maximum(observed,1e-300))-jnp.log(jnp.maximum(mass,1e-300)))


# %% One trapezoidal prior for each independently fitted physical parameter

def model(data):
    params={}
    for name in BLOCKS:
        lo,pl,ph,hi=[jnp.asarray(x) for x in PRIORS[name]]
        prior=dist.Uniform(lo,hi)
        if SIZES[name] is not None:
            if name!='gamma': prior=prior.expand((SIZES[name],))
            prior=prior.to_event(1)
        x=numpyro.sample(name,prior)
        numpyro.factor(name+'_trapezoid',jnp.sum(trapezoid_logpdf(x,(lo,pl,ph,hi))+jnp.log(hi-lo)))
        params[name]=x
    if data is not None:
        numpyro.factor('valid_rt_choice',log_likelihood(params,data))
