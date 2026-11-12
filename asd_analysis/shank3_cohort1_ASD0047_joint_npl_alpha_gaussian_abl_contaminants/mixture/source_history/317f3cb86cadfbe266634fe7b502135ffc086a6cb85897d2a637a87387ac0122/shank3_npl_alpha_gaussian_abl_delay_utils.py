# %% NPL+alpha with one zero-truncated Gaussian delay distribution per ABL
"""Thirteen free parameters; only the six delay parameters are tied across ILD.

The accepted 67-parameter kernels operate on expanded arrays of 30 conditions.
Sampling and trapezoidal priors operate on three ABL-specific means/SDs only.
"""
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import shank3_npl_alpha_gaussian_utils as reference
from shank3_npl_alpha_gaussian_utils import (ROOT, GLOBALS, BLOCKS, PRIORS, COUNTS,
    ABL_ARRAY, ILD_ARRAY, GRID_DT, GRID_END, K_DEFAULT, QUAD_ORDER,
    gaussian_delay_log_integral, sha256, loss_window_decision)
from proactive_lapse_no_led_utils import trapezoid_logpdf

OUT = ROOT/'asd_analysis/shank3_cohort1_wt_hom_npl_alpha_gaussian_delay_by_abl'
REFERENCES = {g:reference.OUT/g for g in ['wt','hom']}
ABLS = [20,40,60]
ABL_INDEX = jnp.repeat(jnp.arange(3),10)
PARAMETER_COUNT = 13


def initialization(seed=0):
    rng = np.random.default_rng(seed)
    before = rng.bit_generator.state
    p = {n:rng.uniform(PRIORS[n][1],PRIORS[n][2],size=3 if n.startswith('delay_') else None)
         for n in BLOCKS}
    return p,dict(generator='numpy.default_rng/PCG64',seed=seed,before=before,after=rng.bit_generator.state)


def parameter_names():
    return GLOBALS+[f'{n}_ABL{abl}' for n in ['delay_mu','delay_sigma'] for abl in ABLS]


def sample_matrix(samples):
    return np.column_stack([np.asarray(samples[n]).reshape(len(samples[n]),-1) for n in BLOCKS])


def expand_delay_parameters(params):
    return {**params,'delay_mu':jnp.asarray(params['delay_mu'])[ABL_INDEX],
            'delay_sigma':jnp.asarray(params['delay_sigma'])[ABL_INDEX]}


def expanded(params):
    return reference.expanded(expand_delay_parameters(params))


def evidence_grid(params,dt=GRID_DT,K=K_DEFAULT,order=QUAD_ORDER,direct=False):
    return reference.evidence_grid(expand_delay_parameters(params),dt,K,order,direct)


def log_likelihood(params,data,marginal=False,dt=GRID_DT,K=K_DEFAULT,order=QUAD_ORDER):
    return reference.log_likelihood(expand_delay_parameters(params),data,marginal,dt,K,order)


def model(data):
    params={}
    for n in BLOCKS:
        lo,pl,ph,hi=PRIORS[n]
        prior=dist.Uniform(lo,hi)
        if n.startswith('delay_'):
            prior=prior.expand((3,)).to_event(1)
        x=numpyro.sample(n,prior)
        numpyro.factor(n+'_trapezoid',jnp.sum(trapezoid_logpdf(x,(lo,pl,ph,hi))+jnp.log(hi-lo)))
        params[n]=x
    if data is not None:
        numpyro.factor('valid_rt_choice',log_likelihood(params,data))
