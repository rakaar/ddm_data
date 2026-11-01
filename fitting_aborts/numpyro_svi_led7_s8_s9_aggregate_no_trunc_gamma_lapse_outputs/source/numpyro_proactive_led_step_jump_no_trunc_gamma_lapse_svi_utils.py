# %%
"""Proactive step-jump plus a Gamma lapse clock with location fixed at fixation start."""

# %%
from collections import OrderedDict

import jax.numpy as jnp
from jax.scipy.special import gammainc, gammaincc, gammaln
import numpyro

import numpyro_proactive_led_step_jump_no_trunc_exp_lapse_svi_utils as exponential

GAMMA_LOCATION_S = 0.0
# Only protects log(0) after double-precision survival underflow, far below the
# existing proactive likelihood floor. It is not a truncation of lapse times.
SURVIVAL_FLOOR = 1e-300
PARAM_BOUNDS = OrderedDict(
    (name, dict(bounds))
    for name, bounds in exponential.PARAM_BOUNDS.items()
    if name != "beta_lapse"
)
PARAM_BOUNDS.update([
    ("shape_lapse", {"hard": (0.05, 50.0), "plausible": (0.3, 10.0)}),
    ("scale_lapse", {"hard": (0.001, 5.0), "plausible": (0.01, 0.8)}),
])
PARAM_NAMES = list(PARAM_BOUNDS)
DEFAULT_INIT_VALUES = {
    name: value for name, value in exponential.DEFAULT_INIT_VALUES.items()
    if name != "beta_lapse"
}
# Shape one and scale 0.2 s exactly reproduce the established beta=5/s start.
DEFAULT_INIT_VALUES.update(shape_lapse=1.0, scale_lapse=0.2)


# %%
def gamma_lapse_logpdf_jax(t, shape_lapse, scale_lapse):
    t = jnp.asarray(t, dtype=jnp.float64)
    scaled = jnp.where(t > 0.0, t / scale_lapse, 1.0)
    value = ((shape_lapse - 1.0) * jnp.log(scaled) - scaled
             - gammaln(shape_lapse) - jnp.log(scale_lapse))
    at_zero = jnp.where(shape_lapse < 1.0, jnp.inf,
                       jnp.where(shape_lapse > 1.0, -jnp.inf, -jnp.log(scale_lapse)))
    return jnp.where(t < 0.0, -jnp.inf, jnp.where(t == 0.0, at_zero, value))


def gamma_lapse_logsurvival_jax(t, shape_lapse, scale_lapse):
    t = jnp.asarray(t, dtype=jnp.float64)
    # Avoid evaluating the derivative of Q(shape, x) at x=0 for shape < 1.
    scaled = jnp.where(t > 0.0, t / scale_lapse, 1.0)
    survival = gammaincc(shape_lapse, scaled)
    return jnp.where(t <= 0.0, 0.0, jnp.log(jnp.maximum(survival, SURVIVAL_FLOOR)))


def gamma_lapse_cdf_jax(t, shape_lapse, scale_lapse):
    t = jnp.asarray(t, dtype=jnp.float64)
    scaled = jnp.where(t > 0.0, t / scale_lapse, 1.0)
    return jnp.where(t <= 0.0, 0.0, gammainc(shape_lapse, scaled))


def proactive_led_step_jump_no_trunc_lapse_loglike_jax(params, data, n_quad=64):
    p, shape, scale = params["lapse_prob"], params["shape_lapse"], params["scale_lapse"]
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
        (on_pdf, gamma_lapse_logpdf_jax(data["on_abort_t"], shape, scale)),
        (on_survival, gamma_lapse_logsurvival_jax(data["on_censored_t_stim"], shape, scale)),
        (off_pdf, gamma_lapse_logpdf_jax(data["off_abort_t"], shape, scale)),
        (off_survival, gamma_lapse_logsurvival_jax(data["off_censored_t_stim"], shape, scale)),
    )
    return sum(jnp.sum(exponential.log_probability_mixture_jax(
        exponential._log_positive(proactive), lapse, p)) for proactive, lapse in terms)


def proactive_led_step_jump_no_trunc_lapse_model(data, n_quad=64):
    params = {name: exponential.step_jump.sample_trapezoid(name, bounds)
              for name, bounds in PARAM_BOUNDS.items()}
    numpyro.factor("proactive_led_no_trunc_gamma_lapse_loglike",
                   proactive_led_step_jump_no_trunc_lapse_loglike_jax(params, data, n_quad=n_quad))


make_fullrank_guide = exponential.make_fullrank_guide
tree_all_finite = exponential.tree_all_finite
led_on_pdf_jax = exponential.led_on_pdf_jax
led_on_cdf_jax = exponential.led_on_cdf_jax
led_off_pdf_jax = exponential.led_off_pdf_jax
led_off_cdf_jax = exponential.led_off_cdf_jax
