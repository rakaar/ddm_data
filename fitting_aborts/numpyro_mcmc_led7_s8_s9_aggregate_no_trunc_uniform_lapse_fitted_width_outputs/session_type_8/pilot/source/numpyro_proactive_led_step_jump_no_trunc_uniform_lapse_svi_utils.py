# %%
"""Proactive LED step-jump with fixation-time Uniform(0, 2.2 s) lapses."""

# %%
from collections import OrderedDict

import jax.numpy as jnp
import numpyro

import numpyro_proactive_led_step_jump_no_trunc_exp_lapse_svi_utils as exponential

UNIFORM_END_S = 2.2
PARAM_BOUNDS = OrderedDict(
    (name, dict(bounds))
    for name, bounds in exponential.PARAM_BOUNDS.items()
    if name != "beta_lapse"
)
PARAM_NAMES = list(PARAM_BOUNDS)
DEFAULT_INIT_VALUES = {
    name: value
    for name, value in exponential.DEFAULT_INIT_VALUES.items()
    if name != "beta_lapse"
}


# %%
# Both density and survival use fixation time, not time relative to LED onset.
def uniform_lapse_logpdf_jax(t, width=UNIFORM_END_S):
    t = jnp.asarray(t, dtype=jnp.float64)
    return jnp.where((t >= 0.0) & (t < width), -jnp.log(width), -jnp.inf)


def uniform_lapse_logsurvival_jax(t, width=UNIFORM_END_S):
    t = jnp.asarray(t, dtype=jnp.float64)
    inside = (t > 0.0) & (t < width)
    safe_fraction = jnp.where(inside, t / width, 0.0)
    return jnp.where(t >= width, -jnp.inf, jnp.log1p(-safe_fraction))


def proactive_led_step_jump_no_trunc_lapse_loglike_jax(params, data, n_quad=64):
    p = params["lapse_prob"]
    width = params.get("width_lapse", UNIFORM_END_S)
    common = (
        params["V_A_base"], params["V_A_post_LED"], params["theta_A"],
        params["del_a_minus_del_LED"], params["del_m_plus_del_LED"],
    )
    on_pdf = exponential.led_on_pdf_jax(data["on_abort_t"], data["on_abort_t_led"], *common)
    on_survival = 1.0 - exponential.led_on_cdf_jax(
        data["on_censored_t_stim"], data["on_censored_t_led"], *common, n_quad=n_quad
    )
    off_common = (common[0], common[2], common[3], common[4])
    off_pdf = exponential.led_off_pdf_jax(data["off_abort_t"], *off_common)
    off_survival = 1.0 - exponential.led_off_cdf_jax(data["off_censored_t_stim"], *off_common)
    terms = (
        (on_pdf, uniform_lapse_logpdf_jax(data["on_abort_t"], width)),
        (on_survival, uniform_lapse_logsurvival_jax(data["on_censored_t_stim"], width)),
        (off_pdf, uniform_lapse_logpdf_jax(data["off_abort_t"], width)),
        (off_survival, uniform_lapse_logsurvival_jax(data["off_censored_t_stim"], width)),
    )
    return sum(
        jnp.sum(exponential.log_probability_mixture_jax(exponential._log_positive(proactive), lapse, p))
        for proactive, lapse in terms
    )


def proactive_led_step_jump_no_trunc_lapse_model(data, n_quad=64):
    params = {
        name: exponential.step_jump.sample_trapezoid(name, bounds)
        for name, bounds in PARAM_BOUNDS.items()
    }
    numpyro.factor(
        "proactive_led_no_trunc_uniform_lapse_loglike",
        proactive_led_step_jump_no_trunc_lapse_loglike_jax(params, data, n_quad=n_quad),
    )


# Reuse the established full-rank guide and proactive first-passage functions.
make_fullrank_guide = exponential.make_fullrank_guide
tree_all_finite = exponential.tree_all_finite
led_on_pdf_jax = exponential.led_on_pdf_jax
led_on_cdf_jax = exponential.led_on_cdf_jax
led_off_pdf_jax = exponential.led_off_pdf_jax
led_off_cdf_jax = exponential.led_off_cdf_jax
