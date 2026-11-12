# %% Pure likelihoods and model builders for adaptive-bound individual refits
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.stats import invgauss
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax.scipy.special import log_ndtr
import numpyro
import numpyro.distributions as dist
import shank3_npl_alpha_gaussian_utils as reference
from gaussian_delay_likelihood import ddm_grid_density
from shank3_npl_alpha_gaussian_abl_delay_utils import expand_delay_parameters, expanded, parameter_names
from proactive_lapse_no_led_utils import PARAM_NAMES, log_likelihood as abort_log_likelihood, trapezoid_logpdf
from shank3_wt_npl_alpha_utils import loss_window_decision

# %% Editable run settings; zero truncation remains exclusive to reactive delays
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'asd_analysis/shank3_cohort1_wt_hom_individual_adaptive_bounds'
ANIMALS = {'wt':['ASD0040','ASD0042','ASD0044','ASD0047','ASD0049'],
           'hom':['ASD0023','ASD0024','ASD0043','ASD0045','ASD0046','ASD0048']}
ABLS = [20,40,60]
ABORT_INITIALS = {0:[1.6,2.5,.09,.05,5.],1:[3.,1.,.04,.15,2.]}
BLOCKS = ['rate_lambda','T_0','theta_E','rate_norm_l','alpha','w','del_go','delay_mu','delay_sigma']
WINDOW, MIN_STEPS, CONFIRMATION, MAX_STEPS = 1000,50000,25,1000000
LEARNING_RATE, CLIP_NORM, INIT_SCALE, DRAWS = .0002,1.,.1,10000
GRID_DT, NARROW_DT, NARROW_SIGMA, K_DEFAULT, QUAD_ORDER = .0005,.0001,.0025,12,24


def sha256(path):
    with open(path,'rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def reactive_cache(valid,fixed):
    v,a,nd,p,beta=np.asarray(fixed)
    rt=valid.timed_rt.to_numpy();onset=valid.onset_s.to_numpy();t=onset+rt
    return dict(rt=rt,onset=onset,choice=valid.response_poke.to_numpy(dtype=int),
        condition=valid.condition.to_numpy(dtype=int),lapse_prob=np.asarray(p),
        proactive_pdf=invgauss.pdf(t-nd,mu=1/(v*a),scale=a*a),
        proactive_survival=invgauss.sf(t-nd,mu=1/(v*a),scale=a*a),
        lapse_pdf=beta*np.exp(-beta*t),lapse_survival=np.exp(-beta*t),
        survival_onset=invgauss.sf(onset-nd,mu=1/(v*a),scale=a*a),
        survival_end=invgauss.sf(onset+1-nd,mu=1/(v*a),scale=a*a),
        lapse_window_mass=np.exp(-beta*onset)*(-np.expm1(-beta)))


def initialization(seed=0):
    # Retain original physical seed-0 starts; the source file is saved per animal.
    from shank3_npl_alpha_gaussian_abl_delay_utils import initialization as original
    return original(seed)


def _need(gamma,omega,w,sigma,dt):
    # Both first-passage mass AND narrow Gaussian/truncation curvature must resolve.
    return reference.needs_time_rescaling(gamma,omega,w,dt) | (sigma < 20*dt)

def _gaussian_delay_grid(params,time,K):
    """Endpoint-corrected convolution; retain unconditional finite-window mass.

    A Gaussian truncated at zero has nonzero density/slope at zero. Ordinary
    trapezoidal convolution leaves O(dt**2) mass error, which would masquerade
    as unabsorbed evidence in late proactive readout. Euler--Maclaurin removes
    that term using f*g'(0)-f'*g(0); no finite-time curve is renormalized.
    """
    gamma,omega,w,mu,sigma=jnp.broadcast_arrays(*[
        jnp.atleast_1d(params[n]) for n in ['gamma','omega','w','delay_mu','delay_sigma']])
    dt=time[1]-time[0]
    f=ddm_grid_density(jnp.where(time>0,time,dt),gamma,omega,w,K).at[:,:,0].set(0.)
    g=jnp.exp(-.5*((time[None,:]-mu[:,None])/sigma[:,None])**2-jnp.log(sigma[:,None])
        -.5*jnp.log(2*jnp.pi)-log_ndtr(mu[:,None]/sigma[:,None]))
    g0=g[:,0];gprime0=mu/sigma**2*g0
    g=g.at[:,0].multiply(.5)
    n=time.size;fft_size=1<<(2*n-2).bit_length()
    raw=jnp.fft.irfft(jnp.fft.rfft(f,n=fft_size,axis=-1)
        *jnp.fft.rfft(g[:,None,:],n=fft_size,axis=-1),n=fft_size,axis=-1)[...,:n]*dt
    derivative=jnp.gradient(f,dt,axis=-1)
    raw+=dt**2/12*(f*gprime0[:,None,None]-derivative*g0[:,None,None])
    density=jnp.maximum(raw,0.).at[:,:,0].set(0.)
    if (n-1)%2==0:
        a,b,c=density[...,0:-2:2],density[...,1:-1:2],density[...,2::2]
        increments=jnp.stack([dt*(5*a+8*b-c)/12,dt*(-a+8*b+5*c)/12],axis=-1)
        increments=increments.reshape(density.shape[:-1]+(n-1,))
    else:
        increments=.5*dt*(density[...,:-1]+density[...,1:])
    cumulative=jnp.concatenate([jnp.zeros_like(density[...,:1]),jnp.cumsum(increments,axis=-1)],axis=-1)
    return density,jnp.maximum(cumulative,0.)


def _expanded_grid(params,dt=GRID_DT,K=K_DEFAULT,order=QUAD_ORDER,direct=False,grid_end=1.5):
    time=jnp.linspace(0.,grid_end,round(grid_end/dt)+1)
    p=reference.expanded(params)
    pdf,cdf=_gaussian_delay_grid(p,time,K)
    fast=_need(p['gamma'],p['omega'],p['w'],p['delay_sigma'],dt)
    if direct:
        fast=jnp.ones_like(fast,dtype=bool)
    def refine(_):
        # The usual5x refinement is batched for GPUthroughput; only exceptional
        #25xFFT or direct quadrature paths serialize conditions/workspaces.
        fine_time=jnp.linspace(0.,grid_end,round(grid_end/dt)*5+1)
        pf,cf=jax.checkpoint(lambda pp:_gaussian_delay_grid(pp,fine_time,K))(p)
        f=jnp.where(fast[:,None,None],pf[:,:,::5],pdf)
        c=jnp.where(fast[:,None,None],cf[:,:,::5],cdf)
        finer=jnp.ones_like(fast,dtype=bool) if direct else fast & _need(
            p['gamma'],p['omega'],p['w'],p['delay_sigma'],dt/5)
        def rare(_):
            def condition(args):
                g,o,mu,sd,ff,cc,need=args
                def integrate(_):
                    def at(side):
                        def integrals(t):
                            return jnp.exp(reference.gaussian_delay_log_integral(t,g,o,p['w'],mu,sd,side,False,K,order)), \
                                   jnp.exp(reference.gaussian_delay_log_integral(t,g,o,p['w'],mu,sd,side,True,K,order))
                        padded=jnp.pad(time,(0,(-time.size)%128)).reshape(-1,128)
                        values=jax.lax.map(jax.checkpoint(integrals),padded)
                        return tuple(x.reshape(-1)[:time.size] for x in values)
                    fm,cm=at(-1);fp,cp=at(1)
                    return jnp.stack([fm,fp]),jnp.stack([cm,cp])
                def fft_at25(_):
                    tt=jnp.linspace(0.,grid_end,round(grid_end/dt)*25+1)
                    local=dict(gamma=jnp.atleast_1d(g),omega=jnp.atleast_1d(o),w=p['w'],
                        delay_mu=jnp.atleast_1d(mu),delay_sigma=jnp.atleast_1d(sd))
                    a,b=_gaussian_delay_grid(local,tt,K)
                    return a[0,:,::25],b[0,:,::25]
                def refined(_):
                    return jax.lax.cond(_need(g,o,p['w'],sd,dt/25),integrate,fft_at25,None)
                return jax.lax.cond(need,integrate if direct else refined,lambda _:(ff,cc),None)
            arrays=(p['gamma'],p['omega'],p['delay_mu'],p['delay_sigma'],f,c,finer)
            return jax.lax.map(jax.checkpoint(condition),arrays)
        return jax.lax.cond(jnp.any(finer),rare,lambda _:(f,c),None)
    pdf,cdf=jax.lax.cond(jnp.any(fast),refine,lambda _:(pdf,cdf),None)
    return time,pdf,cdf


# %% Complete joint mixture; delay integration precedes each exact-onset Z

def evidence_grid(params,dt=GRID_DT,K=K_DEFAULT,order=QUAD_ORDER,direct=False,grid_end=1.5):
    return _expanded_grid(expand_delay_parameters(params),dt,K,order,direct,grid_end)


def _reactive_log_likelihood(params,data,marginal,grid_end,dt,K,order):
    time,pdf,cdf=evidence_grid(params,dt,K,order,grid_end=grid_end)
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


def log_likelihood(stage,params,data,grid_end=1.5,dt=GRID_DT,K=K_DEFAULT,order=QUAD_ORDER,marginal=False):
    if stage=='abort':
        return abort_log_likelihood(jnp.stack([params[name] for name in PARAM_NAMES]),data)
    # A narrow delay needs finer output interpolation as well as FFT quadrature.
    return jax.lax.cond(jnp.min(params['delay_sigma'])<NARROW_SIGMA,
        lambda _: _reactive_log_likelihood(params,data,marginal,grid_end,min(dt,NARROW_DT),K,order),
        lambda _: _reactive_log_likelihood(params,data,marginal,grid_end,dt,K,order),None)


def make_model(stage,priors,grid_end=1.5):
    names=PARAM_NAMES if stage=='abort' else BLOCKS
    def model(data):
        params={}
        for name in names:
            lo,pl,ph,hi=priors[name]
            assert lo<pl<=ph<hi,(name,priors[name])
            prior=dist.Uniform(lo,hi)
            if name in ['delay_mu','delay_sigma']:
                prior=prior.expand((3,)).to_event(1)
            value=numpyro.sample(name,prior)
            numpyro.factor(name+'_trapezoid',jnp.sum(trapezoid_logpdf(value,(lo,pl,ph,hi))+jnp.log(hi-lo)))
            params[name]=value
        if data is not None:
            numpyro.factor(stage+'_joint_likelihood',log_likelihood(stage,params,data,grid_end))
    return model
