# %%
"""Human reactive TIED adapter for the requested RT <= 1.5 s sensitivity fit.

Keep the original 0--3 s adapter and its published source hashes unchanged.
Both choices share the either-bound retention denominator in [0, 1.5] s.
"""
import hashlib
import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from human_reactive_tied_svi_utils import (
    ROOT, PARAM_NAMES, INITIAL, DELTAS, sha256, group_data, bound_cdf,
    bound_density, load_trials as load_all_trials,
)

RT_LOWER, RT_UPPER = 0., 1.5


def load_trials():
    full, provenance = load_all_trials()
    selected = full.loc[full.RT_s.le(RT_UPPER)].copy()
    provenance = dict(provenance)
    provenance.update(
        selection='Condition == "RT" and 0 < RT_s <= 1.5; all valid recorded choices',
        retention_window_s=[RT_LOWER, RT_UPPER], original_trials=len(full),
        selected_trials=len(selected), removed_trials=len(full)-len(selected),
        selected_source_rows_sha256=hashlib.sha256(
            np.asarray(selected.source_row, dtype='<i8').tobytes()).hexdigest(),
        excluded_source_rows_sha256=hashlib.sha256(
            np.asarray(full.loc[full.RT_s.gt(RT_UPPER), 'source_row'], dtype='<i8').tobytes()).hexdigest(),
    )
    return selected, provenance


def retained_mass(B, D, params, K_max=40):
    return sum(bound_cdf(RT_UPPER, c, B, D, params, K_max)
               - bound_cdf(RT_LOWER, c, B, D, params, K_max) for c in [-1, 1])


def log_likelihood(params, data, K_max=40):
    density = bound_density(data['rt'], data['choice'], data['B'], data['D'], params, K_max)
    mass = retained_mass(data['condition_B'], data['condition_D'], params, K_max)
    return jnp.sum(jnp.log(density) - jnp.log(mass[data['condition']]))


def reactive_model(data, delay_upper, K_max=40):
    rate = numpyro.sample('rate_lambda', dist.LogNormal(np.log(.2), 1.5))
    t0 = numpyro.sample('T_0', dist.LogNormal(np.log(.001), 2.))
    theta = numpyro.sample('theta_E', dist.LogNormal(np.log(30.), 1.5))
    w = numpyro.sample('w', dist.Uniform(.05, .95))
    delay = numpyro.sample('t_E_aff', dist.Uniform(0., delay_upper))
    numpyro.factor('joint_retained_rt_choice', log_likelihood(
        jnp.stack([rate, t0, theta, w, delay]), data, K_max))
