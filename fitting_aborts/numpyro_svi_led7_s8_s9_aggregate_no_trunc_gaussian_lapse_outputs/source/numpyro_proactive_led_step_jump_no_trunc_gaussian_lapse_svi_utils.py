# %%
"""Proactive step-jump plus a Gaussian lapse clock conditioned on fixation time >= 0."""

# %%
from collections import OrderedDict

import jax.numpy as jnp
from jax.scipy.special import log_ndtr
import numpyro

import numpyro_proactive_led_step_jump_no_trunc_exp_lapse_svi_utils as exponential

# Mu and sigma describe the underlying untruncated Gaussian, not its retained moments.
PARAM_BOUNDS = OrderedDict(
    (name, dict(bounds))
    for name, bounds in exponential.PARAM_BOUNDS.items()
    if name != "beta_lapse"
)
PARAM_BOUNDS.update([
    ("mu_lapse", {"hard": (-5.0, 5.0), "plausible": (-0.5, 0.5)}),
    ("sigma_lapse", {"hard": (0.001, 5.0), "plausible": (0.01, 0.8)}),
])
PARAM_NAMES = list(PARAM_BOUNDS)
DEFAULT_INIT_VALUES = {
    name: value for name, value in exponential.DEFAULT_INIT_VALUES.items()
    if name != "beta_lapse"
}
DEFAULT_INIT_VALUES.update(mu_lapse=0.05, sigma_lapse=0.2)


# %%
def gaussian_lapse_logpdf_jax(t, mu_lapse, sigma_lapse):
    t = jnp.asarray(t, dtype=jnp.float64)
    z = (jnp.maximum(t, 0.0) - mu_lapse) / sigma_lapse
    logpdf = -0.5 * z**2 - jnp.log(sigma_lapse) - 0.5 * jnp.log(2.0 * jnp.pi)
    logpdf -= log_ndtr(mu_lapse / sigma_lapse)
    return jnp.where(t >= 0.0, logpdf, -jnp.inf)


def gaussian_lapse_logsurvival_jax(t, mu_lapse, sigma_lapse):
    t = jnp.asarray(t, dtype=jnp.float64)
    value = log_ndtr((mu_lapse - jnp.maximum(t, 0.0)) / sigma_lapse)
    value -= log_ndtr(mu_lapse / sigma_lapse)
    return jnp.where(t <= 0.0, 0.0, value)


def proactive_led_step_jump_no_trunc_lapse_loglike_jax(params, data, n_quad=64):
    p, mu, sigma = params["lapse_prob"], params["mu_lapse"], params["sigma_lapse"]
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
        (on_pdf, gaussian_lapse_logpdf_jax(data["on_abort_t"], mu, sigma)),
        (on_survival, gaussian_lapse_logsurvival_jax(data["on_censored_t_stim"], mu, sigma)),
        (off_pdf, gaussian_lapse_logpdf_jax(data["off_abort_t"], mu, sigma)),
        (off_survival, gaussian_lapse_logsurvival_jax(data["off_censored_t_stim"], mu, sigma)),
    )
    return sum(jnp.sum(exponential.log_probability_mixture_jax(
        exponential._log_positive(proactive), lapse, p)) for proactive, lapse in terms)


def proactive_led_step_jump_no_trunc_lapse_model(data, n_quad=64):
    params = {name: exponential.step_jump.sample_trapezoid(name, bounds)
              for name, bounds in PARAM_BOUNDS.items()}
    numpyro.factor("proactive_led_no_trunc_gaussian_lapse_loglike",
                   proactive_led_step_jump_no_trunc_lapse_loglike_jax(params, data, n_quad=n_quad))


make_fullrank_guide = exponential.make_fullrank_guide
tree_all_finite = exponential.tree_all_finite
led_on_pdf_jax = exponential.led_on_pdf_jax
led_on_cdf_jax = exponential.led_on_cdf_jax
led_off_pdf_jax = exponential.led_off_pdf_jax
led_off_cdf_jax = exponential.led_off_cdf_jax
