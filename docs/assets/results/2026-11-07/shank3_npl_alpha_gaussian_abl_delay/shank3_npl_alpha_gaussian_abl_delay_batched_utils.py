# %% Same 13-parameter model, with batched refinement for the WT optimizer
from shank3_npl_alpha_gaussian_abl_delay_utils import *
import shank3_gaussian_batched_refinement as batched


def log_likelihood(params,data,marginal=False,dt=GRID_DT,K=K_DEFAULT,order=QUAD_ORDER):
    return batched.log_likelihood(expand_delay_parameters(params),data,marginal,dt,K,order)


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
