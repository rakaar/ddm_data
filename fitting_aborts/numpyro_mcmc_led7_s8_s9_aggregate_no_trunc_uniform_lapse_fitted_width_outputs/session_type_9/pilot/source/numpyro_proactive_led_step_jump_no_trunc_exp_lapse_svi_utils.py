# %%
"""JAX/NumPyro utilities for the FCT-matched no-truncation lapse model."""

# %%
from collections import OrderedDict

import numpy as np

import jax.numpy as jnp
import numpyro

import numpyro_proactive_led_step_jump_svi_utils as step_jump


# %%
# =============================================================================
# Historical seven-parameter metadata
# =============================================================================
PARAM_NAMES = [
    "V_A_base",
    "V_A_post_LED",
    "theta_A",
    "del_a_minus_del_LED",
    "del_m_plus_del_LED",
    "lapse_prob",
    "beta_lapse",
]

PARAM_BOUNDS = OrderedDict(
    [
        (name, dict(bounds))
        for name, bounds in step_jump.PARAM_BOUNDS.items()
    ]
    + [
        ("lapse_prob", {"hard": (0.0, 1.0), "plausible": (0.01, 0.3)}),
        ("beta_lapse", {"hard": (0.001, 20.0), "plausible": (0.5, 5.0)}),
    ]
)

# Historical FCT initialization after clipping V_A_post_LED to its plausible bound.
DEFAULT_INIT_VALUES = {
    "V_A_base": 1.6,
    "V_A_post_LED": 3.0,
    "theta_A": 2.5,
    "del_a_minus_del_LED": 0.04,
    "del_m_plus_del_LED": 0.05,
    "lapse_prob": 0.05,
    "beta_lapse": 5.0,
}

LIKELIHOOD_FLOOR = step_jump.LIKELIHOOD_FLOOR


# %%
# =============================================================================
# Exponential lapse process and stable probability mixtures
# =============================================================================
def exponential_lapse_logpdf_jax(t, beta_lapse):
    t = jnp.asarray(t, dtype=jnp.float64)
    beta_lapse = jnp.asarray(beta_lapse, dtype=jnp.float64)
    valid = (t >= 0.0) & (beta_lapse > 0.0)
    safe_t = jnp.where(valid, t, 0.0)
    safe_beta = jnp.where(beta_lapse > 0.0, beta_lapse, 1.0)
    value = jnp.log(safe_beta) - safe_beta * safe_t
    return jnp.where(valid, value, -jnp.inf)


def exponential_lapse_logsurvival_jax(t, beta_lapse):
    t = jnp.asarray(t, dtype=jnp.float64)
    beta_lapse = jnp.asarray(beta_lapse, dtype=jnp.float64)
    valid = (t >= 0.0) & (beta_lapse > 0.0)
    safe_t = jnp.where(valid, t, 0.0)
    safe_beta = jnp.where(beta_lapse > 0.0, beta_lapse, 1.0)
    value = -safe_beta * safe_t
    return jnp.where(valid, value, -jnp.inf)


def log_probability_mixture_jax(log_proactive, log_lapse, lapse_prob):
    """Return log((1-p) * proactive + p * lapse) without probability underflow."""
    lapse_prob = jnp.asarray(lapse_prob, dtype=jnp.float64)
    tiny = jnp.finfo(jnp.float64).tiny
    safe_prob = jnp.clip(lapse_prob, tiny, 1.0 - jnp.finfo(jnp.float64).eps)
    return jnp.logaddexp(
        jnp.log1p(-safe_prob) + log_proactive,
        jnp.log(safe_prob) + log_lapse,
    )


def _log_positive(value):
    value = jnp.asarray(value, dtype=jnp.float64)
    safe = jnp.where(
        jnp.isfinite(value) & (value > 0.0),
        value,
        LIKELIHOOD_FLOOR,
    )
    return jnp.log(jnp.maximum(safe, LIKELIHOOD_FLOOR))


# %%
# =============================================================================
# Unweighted, untruncated abort-density plus censored-survival likelihood
# =============================================================================
def proactive_led_step_jump_no_trunc_lapse_loglike_jax(params, data, n_quad=64):
    V_A_base = params["V_A_base"]
    V_A_post_LED = params["V_A_post_LED"]
    theta_A = params["theta_A"]
    del_a_minus_del_LED = params["del_a_minus_del_LED"]
    del_m_plus_del_LED = params["del_m_plus_del_LED"]
    lapse_prob = params["lapse_prob"]
    beta_lapse = params["beta_lapse"]

    on_abort_t = data["on_abort_t"]
    on_abort_t_led = data["on_abort_t_led"]
    on_censored_t_stim = data["on_censored_t_stim"]
    on_censored_t_led = data["on_censored_t_led"]
    off_abort_t = data["off_abort_t"]
    off_censored_t_stim = data["off_censored_t_stim"]

    on_abort_pdf = step_jump.led_on_pdf_jax(
        on_abort_t,
        on_abort_t_led,
        V_A_base,
        V_A_post_LED,
        theta_A,
        del_a_minus_del_LED,
        del_m_plus_del_LED,
    )
    on_abort_loglike = log_probability_mixture_jax(
        _log_positive(on_abort_pdf),
        exponential_lapse_logpdf_jax(on_abort_t, beta_lapse),
        lapse_prob,
    )

    on_censored_cdf = step_jump.led_on_cdf_jax(
        on_censored_t_stim,
        on_censored_t_led,
        V_A_base,
        V_A_post_LED,
        theta_A,
        del_a_minus_del_LED,
        del_m_plus_del_LED,
        n_quad=n_quad,
    )
    on_censored_loglike = log_probability_mixture_jax(
        _log_positive(1.0 - on_censored_cdf),
        exponential_lapse_logsurvival_jax(on_censored_t_stim, beta_lapse),
        lapse_prob,
    )

    off_abort_pdf = step_jump.led_off_pdf_jax(
        off_abort_t,
        V_A_base,
        theta_A,
        del_a_minus_del_LED,
        del_m_plus_del_LED,
    )
    off_abort_loglike = log_probability_mixture_jax(
        _log_positive(off_abort_pdf),
        exponential_lapse_logpdf_jax(off_abort_t, beta_lapse),
        lapse_prob,
    )

    off_censored_cdf = step_jump.led_off_cdf_jax(
        off_censored_t_stim,
        V_A_base,
        theta_A,
        del_a_minus_del_LED,
        del_m_plus_del_LED,
    )
    off_censored_loglike = log_probability_mixture_jax(
        _log_positive(1.0 - off_censored_cdf),
        exponential_lapse_logsurvival_jax(off_censored_t_stim, beta_lapse),
        lapse_prob,
    )

    return (
        jnp.sum(on_abort_loglike)
        + jnp.sum(on_censored_loglike)
        + jnp.sum(off_abort_loglike)
        + jnp.sum(off_censored_loglike)
    )


# %%
# =============================================================================
# Matching trapezoidal priors and full-rank guide
# =============================================================================
def proactive_led_step_jump_no_trunc_lapse_model(data, n_quad=64):
    params = {
        name: step_jump.sample_trapezoid(name, bounds)
        for name, bounds in PARAM_BOUNDS.items()
    }
    loglike = proactive_led_step_jump_no_trunc_lapse_loglike_jax(
        params,
        data,
        n_quad=n_quad,
    )
    numpyro.factor("proactive_led_no_trunc_lapse_loglike", loglike)


def make_fullrank_guide(model, init_values, init_scale=0.1):
    return step_jump.make_fullrank_guide(model, init_values, init_scale=init_scale)


def clip_init_to_hard_bounds(init_values):
    clipped = {}
    for name, bounds in PARAM_BOUNDS.items():
        low, high = bounds["hard"]
        eps = max(1e-9, 1e-6 * (high - low))
        clipped[name] = float(np.clip(init_values[name], low + eps, high - eps))
    return clipped


tree_all_finite = step_jump.tree_all_finite
led_on_pdf_jax = step_jump.led_on_pdf_jax
led_on_cdf_jax = step_jump.led_on_cdf_jax
led_off_pdf_jax = step_jump.led_off_pdf_jax
led_off_cdf_jax = step_jump.led_off_cdf_jax
