# %% No-LED, untruncated proactive + exponential-lapse likelihood
"""Fixation-referenced density and right-censored survival; diffusion is one.

The exponential lapse starts at fixation onset. Only the proactive process is
shifted by t_nd. This module has no fitting or file-writing side effects.
"""
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax.scipy.special import log_ndtr
import numpyro
import numpyro.distributions as dist

PARAM_NAMES = ['V_A', 'theta_A', 't_nd', 'lapse_prob', 'beta_lapse']
# (hard low, plateau low, plateau high, hard high)
PRIOR_BOUNDS = {'V_A': (.1, .5, 3., 8.), 'theta_A': (.1, .5, 3., 8.),
                't_nd': (0., .02, .14, .30), 'lapse_prob': (0., .01, .30, 1.),
                'beta_lapse': (.001, .5, 5., 20.)}


def proactive_logpdf_logsf(t, drift, bound, delay):
    """Stable inverse-Gaussian log density and log survival, including t<=delay."""
    u = jnp.asarray(t, dtype=jnp.float64) - delay
    safe_u = jnp.where(u > 0, u, 1.)
    root = jnp.sqrt(safe_u)
    logpdf = (jnp.log(bound) - .5*jnp.log(2*jnp.pi) - 1.5*jnp.log(safe_u)
              - .5*(bound-drift*safe_u)**2/safe_u)
    # S = Phi((a-vu)/sqrt(u)) - exp(2va) Phi(-(a+vu)/sqrt(u)).
    # Log subtraction avoids cancellation from 1-CDF in the far right tail.
    first = log_ndtr((bound-drift*safe_u)/root)
    second = 2*drift*bound + log_ndtr(-(bound+drift*safe_u)/root)
    logsf = first + jnp.log(-jnp.expm1(jnp.minimum(second-first, -1e-15)))
    return jnp.where(u > 0, logpdf, -jnp.inf), jnp.where(u > 0, logsf, 0.)


def mixture_logpdf_logsf(t, params):
    v, a, delay, p, beta = params
    lp, ls = proactive_logpdf_logsf(t, v, a, delay)
    t = jnp.asarray(t, dtype=jnp.float64)
    # Safe interior expressions plus exact p=0/1 limits for independent checks.
    safe_p = jnp.clip(p, 1e-15, 1-1e-15)
    lapse_lp = jnp.log(beta) - beta*jnp.maximum(t, 0.)
    lapse_ls = -beta*jnp.maximum(t, 0.)
    mixture_lp = jnp.logaddexp(jnp.log1p(-safe_p)+lp, jnp.log(safe_p)+lapse_lp)
    mixture_ls = jnp.logaddexp(jnp.log1p(-safe_p)+ls, jnp.log(safe_p)+lapse_ls)
    mixture_lp = jnp.where(p <= 0, lp, jnp.where(p >= 1, lapse_lp, mixture_lp))
    mixture_ls = jnp.where(p <= 0, ls, jnp.where(p >= 1, lapse_ls, mixture_ls))
    return jnp.where(t >= 0, mixture_lp, -jnp.inf), jnp.where(t >= 0, mixture_ls, 0.)


def log_likelihood(params, data):
    # Exact repeated event times are counted, not rounded or histogrammed.
    event_lp, _ = mixture_logpdf_logsf(data['event_times'], params)
    _, censor_ls = mixture_logpdf_logsf(data['censor_times'], params)
    return jnp.dot(event_lp, data['event_counts']) + jnp.sum(censor_ls)


def trapezoid_logpdf(x, bounds):
    lo, pl, ph, hi = bounds
    height = 1/((pl-lo+hi-ph)/2 + ph-pl)
    shape = jnp.minimum(jnp.minimum((x-lo)/(pl-lo), (hi-x)/(hi-ph)), 1.)
    return jnp.where((x > lo) & (x < hi), jnp.log(jnp.maximum(shape, 1e-300)*height), -jnp.inf)


def model(data):
    values = []
    for name in PARAM_NAMES:
        lo, pl, ph, hi = PRIOR_BOUNDS[name]
        value = numpyro.sample(name, dist.Uniform(lo, hi))
        numpyro.factor(name+'_trapezoid', trapezoid_logpdf(value, PRIOR_BOUNDS[name])+jnp.log(hi-lo))
        values.append(value)
    numpyro.factor('abort_density_and_censored_survival', log_likelihood(jnp.stack(values), data))
