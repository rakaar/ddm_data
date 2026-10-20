# %%
"""Choice-marginalized human TIED likelihood with an explicit retained RT window.

ASL and signed difference remain covariates. Response side is not an input to
the likelihood. Reuse the unchanged IPL/vanilla bound-specific primitives.
"""
import hashlib
import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from human_reactive_tied_svi_utils import (
    ROOT, PARAM_NAMES, INITIAL, sha256, group_data, bound_cdf, bound_density,
    load_trials as load_all_trials,
)

RT_LOWER, RT_UPPER = .2, 1.5


def load_trials(rt_lower=RT_LOWER, rt_upper=RT_UPPER):
    full, provenance = load_all_trials()
    selected = full.loc[full.RT_s.between(rt_lower, rt_upper)].copy()
    provenance = dict(provenance)
    provenance.update(
        selection=f'Condition == "RT" and {rt_lower} <= RT_s <= {rt_upper}; response side ignored in fitting',
        retention_window_s=[rt_lower, rt_upper], original_trials=len(full),
        selected_trials=len(selected), removed_trials=len(full)-len(selected),
        selected_source_rows_sha256=hashlib.sha256(
            np.asarray(selected.source_row, dtype='<i8').tobytes()).hexdigest(),
        excluded_source_rows_sha256=hashlib.sha256(
            np.asarray(full.loc[~full.RT_s.between(rt_lower, rt_upper), 'source_row'], dtype='<i8').tobytes()).hexdigest(),
    )
    return selected, provenance


def retained_mass(B, D, params, rt_lower, rt_upper, K_max=40):
    return sum(bound_cdf(rt_upper, side, B, D, params, K_max)
               - bound_cdf(rt_lower, side, B, D, params, K_max) for side in [-1, 1])


def log_likelihood(params, data, K_max=40):
    density = sum(bound_density(data['rt'], side, data['B'], data['D'], params, K_max)
                  for side in [-1, 1])
    mass = retained_mass(data['condition_B'], data['condition_D'], params,
                         data['rt_lower'], data['rt_upper'], K_max)
    return jnp.sum(jnp.log(density) - jnp.log(mass[data['condition']]))


def reactive_model(data, delay_upper, K_max=40):
    rate = numpyro.sample('rate_lambda', dist.LogNormal(np.log(.2), 1.5))
    t0 = numpyro.sample('T_0', dist.LogNormal(np.log(.001), 2.))
    theta = numpyro.sample('theta_E', dist.LogNormal(np.log(30.), 1.5))
    w = numpyro.sample('w', dist.Uniform(.05, .95))
    delay = numpyro.sample('t_E_aff', dist.Uniform(0., delay_upper))
    numpyro.factor('retained_rt', log_likelihood(
        jnp.stack([rate, t0, theta, w, delay]), data, K_max))
