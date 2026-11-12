# %% Stable direct Gamma/Omega evidence process and fixed proactive/lapse race
"""Bounds -1/+1; start 2*w-1; evidence time omega*(RT-t_E_aff).

Short-time image and long-time spectral expansions follow the human DDM
implementation. The race/readout follows the existing animal valid likelihood.
All fixed proactive quantities are cached at exact trial times, never binned.
"""
from pathlib import Path
import sys
import hashlib
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax.scipy.special import log_ndtr, ndtr
import numpyro
import numpyro.distributions as dist
from proactive_lapse_no_led_utils import proactive_logpdf_logsf, trapezoid_logpdf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'asd_analysis/shank3_cohort1_wt_gamma_omega'
ABLS = [20, 40, 60]
ILDS = [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]
PARAM_BLOCKS = ['gamma', 'omega', 't_E_aff', 'w', 'del_go']
PRIORS = {'gamma_positive': [-1., 0., 12., 20.], 'gamma_negative': [-20., -12., 0., 1.],
          'omega': [.01, .1, 12., 15.], 't_E_aff': [0., .01, .2, .5],
          'w': [0., .4, .6, 1.], 'del_go': [0., .02, .199, .2]}
K_DEFAULT = 12  # Both expansions validated against K=40 and independent K=80.


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def evidence_bound(u, gamma, omega, w, side, K=K_DEFAULT):
    """Return bound PDF, bound CDF and remaining bound mass; u is evidence age.

Remaining bound masses add to survival. Keeping spectral tails directly avoids
subtracting a nearly-one CDF in the long-time regime. Masked branches have safe
arguments so gradients remain finite at the evidence onset.
"""
    u, gamma, omega, w, side = jnp.broadcast_arrays(u, gamma, omega, w, side)
    tt = jnp.where(u > 0, omega*u, .4)
    short = tt < .6
    ts, tl = jnp.where(short, tt, .4), jnp.where(short, 1., tt)
    v = jnp.where(side == 1, -gamma, gamma)
    wb = jnp.where(side == 1, 1-w, w)
    # Eventual mass at the specified bound, stable at zero and either drift sign.
    small_v = jnp.abs(v) < 1e-6
    av = jnp.where(small_v, 1., jnp.abs(v))
    denom = -jnp.expm1(-4*av)
    eventual_positive = jnp.exp(-4*av*wb)*(-jnp.expm1(-4*av*(1-wb)))/denom
    eventual_negative = -jnp.expm1(-4*av*(1-wb))/denom
    eventual = jnp.where(small_v, 1-wb-2*v*wb*(1-wb)+(4/3)*v*v*wb*(1-wb)*(2*wb-1),
                        jnp.where(v > 0, eventual_positive, eventual_negative))
    # Short-time PDF: image expansion with the full exponent combined.
    paths = wb[...,None]+2*jnp.arange(-K//2, K//2+1)
    short_pdf = 2*omega/jnp.sqrt(2*jnp.pi*ts**3)*jnp.sum(paths*jnp.exp(
        -2*paths**2/ts[...,None]-2*v[...,None]*wb[...,None]-v[...,None]**2*ts[...,None]/2), axis=-1)
    k = jnp.arange(K+1)
    r = jnp.where(k % 2 == 0, 2*k+2*wb[...,None], 2*k+2*(1-wb[...,None]))
    z1=(v[...,None]*ts[...,None]-r)/jnp.sqrt(ts)[...,None]
    z2=(-v[...,None]*ts[...,None]-r)/jnp.sqrt(ts)[...,None]
    exponent1=-2*v[...,None]*wb[...,None]-v[...,None]*r
    exponent2=-2*v[...,None]*wb[...,None]+v[...,None]*r
    if K <= 12:
        # Under the approved |gamma|<=20 bounds these exponents are <=560,
        # safely below float64 overflow. Underflowed tails are far below double
        # precision of the complete mixture. The K40/80 reference uses log CDFs.
        first=jnp.exp(exponent1)*ndtr(z1)
        second=jnp.exp(exponent2)*ndtr(z2)
    else:
        first=jnp.exp(exponent1+log_ndtr(z1))
        second=jnp.exp(exponent2+log_ndtr(z2))
    short_cdf = jnp.sum((-1.)**k*(first+second),axis=-1)
    # Long-time PDF and remaining mass from the same spectral series.
    n = jnp.arange(1,K+1)
    rate = v[...,None]**2/2+n**2*jnp.pi**2/8
    terms = n*jnp.sin(n*jnp.pi*wb[...,None])*jnp.exp(-2*v[...,None]*wb[...,None]-rate*tl[...,None])
    long_pdf = jnp.pi*omega/4*jnp.sum(terms,axis=-1)
    long_remaining = jnp.pi/4*jnp.sum(terms/rate,axis=-1)
    cdf = jnp.clip(jnp.where(short,short_cdf,eventual-long_remaining),0.,eventual)
    remaining = jnp.maximum(jnp.where(short,eventual-cdf,long_remaining),0.)
    pdf = jnp.maximum(jnp.where(short,short_pdf,long_pdf),0.)
    return jnp.where(u > 0,pdf,0.), jnp.where(u > 0,cdf,0.), jnp.where(u > 0,remaining,eventual)


def fixed_at(rt, onset, fixed):
    v,a,delay,p,beta = fixed
    t = jnp.asarray(onset)+jnp.asarray(rt)
    lp,ls = proactive_logpdf_logsf(t,v,a,delay)
    return {'proactive_pdf':jnp.exp(lp), 'proactive_survival':jnp.exp(ls),
            'lapse_pdf':jnp.where(t >= 0,beta*jnp.exp(-beta*jnp.maximum(t,0)),0.),
            'lapse_survival':jnp.exp(-beta*jnp.maximum(t,0))}


def choice_density(rt, side, gamma, omega, delay, w, del_go, cached, lapse_prob, K=K_DEFAULT):
    pe, fe, _ = evidence_bound(rt-delay,gamma,omega,w,side,K)
    _, fe_go, remaining_go = evidence_bound(rt-delay+del_go,gamma,omega,w,side,K)
    _, _, opposite_remaining_go = evidence_bound(rt-delay+del_go,gamma,omega,w,-side,K)
    readout = jnp.maximum(.5*(remaining_go+opposite_remaining_go)+fe_go-fe,0.)
    ordinary = cached['proactive_pdf']*readout+pe*cached['proactive_survival']
    return (1-lapse_prob)*ordinary+.5*lapse_prob*cached['lapse_pdf']


def total_cdf(rt, onset, gamma, omega, delay, w, fixed, K=K_DEFAULT):
    cached = fixed_at(rt,onset,fixed)
    _,_,su = evidence_bound(rt-delay,gamma,omega,w,1,K)
    _,_,sl = evidence_bound(rt-delay,gamma,omega,w,-1,K)
    return 1-((1-fixed[3])*cached['proactive_survival']*(su+sl)+fixed[3]*cached['lapse_survival'])


def delays(params, variant):
    return jnp.broadcast_to(params['t_E_aff'],(30,)) if variant == 'shared_delay' else params['t_E_aff']


def log_likelihood(params, data, variant, K=K_DEFAULT, marginal=False):
    ids = data['condition']
    delay = delays(params,variant)
    gamma,omega,w = params['gamma'],params['omega'],params['w']
    pdf = choice_density(data['rt'],data['choice'],gamma[ids],omega[ids],delay[ids],w,params['del_go'],
                         data, data['lapse_prob'],K)
    if marginal:
        pdf = pdf+choice_density(data['rt'],-data['choice'],gamma[ids],omega[ids],delay[ids],w,params['del_go'],
                                 data,data['lapse_prob'],K)
    _,_,su = evidence_bound(1-delay,gamma,omega,w,1,K)
    _,_,sl = evidence_bound(1-delay,gamma,omega,w,-1,K)
    # At RT=0 the evidence has not started. Normalize the COMPLETE mixture.
    mass = ((1-data['lapse_prob'])*(data['survival_onset']-data['survival_end']*(su+sl)[ids])
            +data['lapse_prob']*data['lapse_window_mass'])
    return jnp.sum(jnp.log(jnp.maximum(pdf,1e-300))-jnp.log(jnp.maximum(mass,1e-300)))


def model(data,variant):
    params = {}
    for name in PARAM_BLOCKS:
        if name == 'gamma':
            bounds = jnp.asarray([PRIORS['gamma_positive' if ild > 0 else 'gamma_negative'] for _ in ABLS for ild in ILDS]).T
        else:
            count = 30 if name == 'omega' or (name == 't_E_aff' and variant == 'condition_delay') else None
            bounds = jnp.asarray(PRIORS[name])
            if count:
                bounds = jnp.broadcast_to(bounds[:,None],(4,count))
        lo,pl,ph,hi = bounds
        prior = dist.Uniform(lo,hi)
        if jnp.ndim(lo):
            prior = prior.to_event(1)
        value = numpyro.sample(name,prior)
        numpyro.factor(name+'_trapezoid',jnp.sum(trapezoid_logpdf(value,(lo,pl,ph,hi))+jnp.log(hi-lo)))
        params[name] = value
    numpyro.factor('valid_rt_choice',log_likelihood(params,data,variant))


def sample_matrix(samples):
    return np.column_stack([np.asarray(samples[n]).reshape(len(samples[n]),-1) for n in PARAM_BLOCKS])


def parameter_names(variant):
    return ([f'gamma_{i:02d}' for i in range(30)]+[f'omega_{i:02d}' for i in range(30)]
        +(['t_E_aff'] if variant == 'shared_delay' else [f't_E_aff_{i:02d}' for i in range(30)])+['w','del_go'])
