# %% Shared settings and the Gaussian-delay extension of the verified rat race
"""NPL+alpha with 30 independent zero-truncated Gaussian delay distributions.

The human reference contributes convolution kernels, not its uniform-lapse
likelihood. All averaging over delays precedes the rat's complete-mixture
retained-window normalization. Gaussian mu/sigma are physical trial variability.
"""
import sys
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from shank3_wt_npl_alpha_utils import (ROOT, GLOBALS, PRIORS as POINT_PRIORS,
    ABL_ARRAY, ILD_ARRAY, gamma_omega_alpha_jax, loss_window_decision)
from shank3_wt_gamma_omega_utils import sha256, fixed_at
from proactive_lapse_no_led_utils import trapezoid_logpdf

sys.path.insert(0, str(ROOT/'ASD_Human_Data'))
from gaussian_delay_likelihood import (gaussian_delay_grid,
    gaussian_delay_log_integral, needs_time_rescaling)

OUT = ROOT/'asd_analysis/shank3_cohort1_wt_hom_npl_alpha_gaussian_delay'
REFERENCES = {g: ROOT/f'asd_analysis/shank3_cohort1_{g}_npl_alpha_condition_delay' for g in ['wt','hom']}
COUNTS = {'wt':dict(animals=5, valid=68184, right=37482, left=30702, parent=83921, aborts=13750, display=81934),
          'hom':dict(animals=6, valid=67715, right=35320, left=32395, parent=81721, aborts=13274, display=80989)}
BLOCKS = GLOBALS + ['delay_mu', 'delay_sigma']
PRIORS = {n: POINT_PRIORS[n] for n in GLOBALS}
PRIORS.update(delay_mu=[0., .01, .20, .50], delay_sigma=[.005, .01, .15, .30])
GRID_DT, GRID_END, K_DEFAULT, QUAD_ORDER = .0005, 1.2, 12, 24


def initialization(seed=0):
    rng = np.random.default_rng(seed)
    before = rng.bit_generator.state
    params = {n:rng.uniform(PRIORS[n][1], PRIORS[n][2], size=30 if n.startswith('delay_') else None)
              for n in BLOCKS}
    return params, dict(generator='numpy.default_rng/PCG64', seed=seed, before=before, after=rng.bit_generator.state)


def parameter_names():
    return GLOBALS + [f'{n}_{i:02d}' for n in ['delay_mu','delay_sigma'] for i in range(30)]


def sample_matrix(samples):
    return np.column_stack([np.asarray(samples[n]).reshape(len(samples[n]),-1) for n in BLOCKS])


def expanded(params):
    gamma, omega = gamma_omega_alpha_jax(ABL_ARRAY, ILD_ARRAY, params['rate_lambda'], params['T_0'],
        params['theta_E'], params['rate_norm_l'], params['alpha'])
    return dict(gamma=gamma, omega=omega, w=params['w'], del_go=params['del_go'],
                delay_mu=params['delay_mu'], delay_sigma=params['delay_sigma'])


def evidence_grid(params, dt=GRID_DT, K=K_DEFAULT, order=QUAD_ORDER, direct=False):
    """Unconditional joint evidence PDF/CDF, after marginalizing random delay.

    Resolve fast decisions with positive rescaled quadrature. Only one
    condition's quadrature is resident at a time. No finite-grid renormalizing.
    """
    time = jnp.linspace(0., GRID_END, round(GRID_END/dt)+1)
    p = expanded(params)
    pdf, cdf = gaussian_delay_grid(p, time, K)
    fast = needs_time_rescaling(p['gamma'], p['omega'], p['w'], dt)
    if direct:
        fast = jnp.ones(30, dtype=bool)
    def condition(args):
        g,o,mu,sd,f,c,need = args
        def integrate(_):
            def bound(side):
                # Rematerialization avoids storing all quadrature intermediates
                # for both sides and 30 conditions in the full-rank SVI gradient.
                def at(t):
                    return jnp.exp(gaussian_delay_log_integral(t,g,o,p['w'],mu,sd,side,False,K,order)), \
                           jnp.exp(gaussian_delay_log_integral(t,g,o,p['w'],mu,sd,side,True,K,order))
                a,b = jax.checkpoint(at)(time)
                return a,b
            fm,cm = bound(-1)
            fp,cp = bound(1)
            return jnp.stack([fm,fp]), jnp.stack([cm,cp])
        def refine(_):
            # Resolve the first-passage mass on a finer grid, then retain the
            # common output grid. Gaussian smoothing (sigma >=5 ms) makes the
            # 0.5-ms interpolation accurate even for sub-ms decision times.
            def fft_at(factor):
                fine_time=jnp.linspace(0.,GRID_END,round(GRID_END/dt)*factor+1)
                local=dict(gamma=jnp.atleast_1d(g),omega=jnp.atleast_1d(o),w=p['w'],
                           delay_mu=jnp.atleast_1d(mu),delay_sigma=jnp.atleast_1d(sd))
                a,b=gaussian_delay_grid(local,fine_time,K)
                return a[0,:,::factor],b[0,:,::factor]
            def finest(_):
                return jax.lax.cond(needs_time_rescaling(g,o,p['w'],dt/25),integrate,lambda _:fft_at(25),None)
            return jax.lax.cond(needs_time_rescaling(g,o,p['w'],dt/5),finest,lambda _:fft_at(5),None)
        return jax.lax.cond(need, integrate if direct else refine, lambda _:(f,c), None)
    pdf,cdf = jax.lax.map(condition,(p['gamma'],p['omega'],p['delay_mu'],p['delay_sigma'],pdf,cdf,fast))
    return time,pdf,cdf


def interpolate(curves, time, rt, ids):
    """Cubic Hermite interpolation on the common grid, zero before time zero."""
    position = jnp.clip(jnp.asarray(rt)/(time[1]-time[0]),0.,time.size-1.)
    i = jnp.minimum(jnp.floor(position).astype(int),time.size-2)
    z = position-i
    y0,y1 = curves[ids,:,i],curves[ids,:,i+1]
    ym,yp = curves[ids,:,jnp.maximum(i-1,0)],curves[ids,:,jnp.minimum(i+2,time.size-1)]
    m0=jnp.where((i>0)[...,None],(y1-ym)/2,y1-y0)
    m1=jnp.where((i<time.size-2)[...,None],(yp-y0)/2,y1-y0)
    z=z[...,None]
    value=(2*z**3-3*z**2+1)*y0+(z**3-2*z**2+z)*m0+(-2*z**3+3*z**2)*y1+(z**3-z**2)*m1
    return jnp.where((jnp.asarray(rt)>0)[...,None], jnp.maximum(value,0.), 0.)


def log_likelihood(params, data, marginal=False, dt=GRID_DT, K=K_DEFAULT, order=QUAD_ORDER):
    time,pdf,cdf = evidence_grid(params,dt,K,order)
    ids,rt = data['condition'],data['rt']
    ef=interpolate(pdf,time,rt,ids)
    ec=interpolate(cdf,time,rt,ids)
    eg=interpolate(cdf,time,rt+params['del_go'],ids)
    readout=.5*(1-eg.sum(-1))[...,None]+eg-ec
    ordinary=data['proactive_pdf'][...,None]*readout+data['proactive_survival'][...,None]*ef
    joint=(1-data['lapse_prob'])*ordinary+.5*data['lapse_prob']*data['lapse_pdf'][...,None]
    observed = joint.sum(-1) if marginal else joint[jnp.arange(rt.size),(data['choice']==1).astype(int)]
    # Direct endpoint quadrature keeps each exact-onset denominator independent
    # of FFT tail roundoff. This integrates delays BEFORE complete-mixture Z.
    p=expanded(params)
    endpoint=jnp.exp(gaussian_delay_log_integral(1.,p['gamma'][:,None],p['omega'][:,None],p['w'],
        p['delay_mu'][:,None],p['delay_sigma'][:,None],jnp.array([-1,1])[None,:],True,K,order)).sum(-1)
    mass=(1-data['lapse_prob'])*(data['survival_onset']-data['survival_end']*(1-endpoint[ids])) \
        +data['lapse_prob']*data['lapse_window_mass']
    return jnp.sum(jnp.log(jnp.maximum(observed,1e-300))-jnp.log(jnp.maximum(mass,1e-300)))


def model(data):
    params={}
    for n in BLOCKS:
        lo,pl,ph,hi=PRIORS[n]
        prior=dist.Uniform(lo,hi)
        if n.startswith('delay_'):
            prior=prior.expand((30,)).to_event(1)
        value=numpyro.sample(n,prior)
        numpyro.factor(n+'_trapezoid',jnp.sum(trapezoid_logpdf(value,(lo,pl,ph,hi))+jnp.log(hi-lo)))
        params[n]=value
    if data is not None:
        numpyro.factor('valid_rt_choice',log_likelihood(params,data))
