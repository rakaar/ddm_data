# %%
"""JAX/NumPyro utilities for the identifiable proactive LED step-jump model."""

# %%
from collections import OrderedDict
from functools import lru_cache

import numpy as np

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import erf, log_ndtr, ndtr
from numpyro.infer.autoguide import AutoMultivariateNormal
from numpyro.infer.initialization import init_to_value


# %%
# =============================================================================
# Parameter metadata copied from the historical VBMC fit
# =============================================================================
PARAM_NAMES = [
    "V_A_base",
    "V_A_post_LED",
    "theta_A",
    "del_a_minus_del_LED",
    "del_m_plus_del_LED",
]

PARAM_BOUNDS = OrderedDict(
    [
        ("V_A_base", {"hard": (0.1, 8.0), "plausible": (0.5, 3.0)}),
        ("V_A_post_LED", {"hard": (0.1, 8.0), "plausible": (0.5, 3.0)}),
        ("theta_A", {"hard": (0.1, 8.0), "plausible": (0.5, 3.0)}),
        ("del_a_minus_del_LED", {"hard": (-1.1, 1.1), "plausible": (0.01, 0.07)}),
        ("del_m_plus_del_LED", {"hard": (0.001, 0.2), "plausible": (0.01, 0.07)}),
    ]
)

# This is the deterministic seed-50 plausible point used by the old VBMC file.
DEFAULT_INIT_VALUES = {
    "V_A_base": 1.7365041138450537,
    "V_A_post_LED": 1.0702077611233405,
    "theta_A": 1.1386848093930284,
    "del_a_minus_del_LED": 0.03377979458336662,
    "del_m_plus_del_LED": 0.03263890586146744,
}

LIKELIHOOD_FLOOR = 1e-50


# %%
# =============================================================================
# Stable inverse-Gaussian first-passage functions
# =============================================================================
def inverse_gaussian_pdf_jax(t, drift, bound):
    """Single-bound Brownian first-passage density with diffusion coefficient 1."""
    t, drift, bound = jnp.broadcast_arrays(
        jnp.asarray(t, dtype=jnp.float64),
        jnp.asarray(drift, dtype=jnp.float64),
        jnp.asarray(bound, dtype=jnp.float64),
    )
    valid = (t > 0.0) & (bound > 0.0)
    safe_t = jnp.where(valid, t, 1.0)
    safe_bound = jnp.where(valid, bound, 1.0)
    log_pdf = (
        jnp.log(safe_bound)
        - 0.5 * jnp.log(2.0 * jnp.pi)
        - 1.5 * jnp.log(safe_t)
        - 0.5 * (safe_bound - drift * safe_t) ** 2 / safe_t
    )
    return jnp.where(valid, jnp.exp(log_pdf), 0.0)


def inverse_gaussian_cdf_jax(t, drift, bound):
    """Stable single-bound first-passage CDF."""
    t, drift, bound = jnp.broadcast_arrays(
        jnp.asarray(t, dtype=jnp.float64),
        jnp.asarray(drift, dtype=jnp.float64),
        jnp.asarray(bound, dtype=jnp.float64),
    )
    valid = (t > 0.0) & (bound > 0.0)
    safe_t = jnp.where(valid, t, 1.0)
    safe_bound = jnp.where(valid, bound, 1.0)
    sqrt_t = jnp.sqrt(safe_t)
    z_first = (drift * safe_t - safe_bound) / sqrt_t
    z_second = -(drift * safe_t + safe_bound) / sqrt_t
    second_term = jnp.exp(2.0 * drift * safe_bound + log_ndtr(z_second))
    cdf = ndtr(z_first) + second_term
    return jnp.where(valid, jnp.clip(cdf, 0.0, 1.0), 0.0)


# %%
# =============================================================================
# Closed-form density after the LED drift jump
# =============================================================================
def post_jump_pdf_closed_form_jax(V_A_base, V_A_post_LED, theta_A, elapsed_post, elapsed_pre):
    """JAX port of ``stupid_f_integral`` from the historical VBMC likelihood."""
    elapsed_post, elapsed_pre = jnp.broadcast_arrays(
        jnp.asarray(elapsed_post, dtype=jnp.float64),
        jnp.asarray(elapsed_pre, dtype=jnp.float64),
    )
    valid = (elapsed_post > 0.0) & (elapsed_pre > 0.0)
    t = jnp.where(valid, elapsed_post, 1.0)
    tp = jnp.where(valid, elapsed_pre, 1.0)
    v = jnp.asarray(V_A_base, dtype=jnp.float64)
    v_on = jnp.asarray(V_A_post_LED, dtype=jnp.float64)
    theta = jnp.asarray(theta_A, dtype=jnp.float64)

    a1 = 0.5 * (1.0 / t + 1.0 / tp)
    b1 = theta / t + (v - v_on)
    c1 = -0.5 * (v_on**2 * t - 2.0 * theta * v_on + theta**2 / t + v**2 * tp)

    a2 = a1
    b2 = theta * (1.0 / t + 2.0 / tp) + (v - v_on)
    c2 = -0.5 * (
        v_on**2 * t
        - 2.0 * theta * v_on
        + theta**2 / t
        + v**2 * tp
        + 4.0 * theta * v
        + 4.0 * theta**2 / tp
    ) + 2.0 * v * theta

    common = 1.0 / (4.0 * jnp.pi * a1 * jnp.sqrt(tp * t**3))
    T11 = b1**2 / (4.0 * a1)
    T12 = (2.0 * a1 * theta - b1) / (2.0 * jnp.sqrt(a1))
    T13 = theta * (b1 - theta * a1)

    T21 = b2**2 / (4.0 * a2)
    T22 = (2.0 * a2 * theta - b2) / (2.0 * jnp.sqrt(a2))
    T23 = theta * (b2 - theta * a2)

    I1 = common * (
        T12 * jnp.sqrt(jnp.pi) * jnp.exp(T11 + c1) * (erf(T12) + 1.0)
        + jnp.exp(T13 + c1)
    )
    I2 = common * (
        T22 * jnp.sqrt(jnp.pi) * jnp.exp(T21 + c2) * (erf(T22) + 1.0)
        + jnp.exp(T23 + c2)
    )
    density = I1 - I2
    return jnp.where(valid, jnp.maximum(density, 0.0), 0.0)


def led_on_pdf_jax(
    t,
    t_led,
    V_A_base,
    V_A_post_LED,
    theta_A,
    del_a_minus_del_LED,
    del_m_plus_del_LED,
):
    """Untruncated LED-ON first-passage PDF in observed fixation-time coordinates."""
    t, t_led = jnp.broadcast_arrays(
        jnp.asarray(t, dtype=jnp.float64),
        jnp.asarray(t_led, dtype=jnp.float64),
    )
    elapsed_pre = t_led - del_a_minus_del_LED
    elapsed_post = t - t_led - del_m_plus_del_LED
    elapsed_if_pre = t - (del_m_plus_del_LED + del_a_minus_del_LED)

    pre_pdf = inverse_gaussian_pdf_jax(elapsed_if_pre, V_A_base, theta_A)
    post_only_pdf = inverse_gaussian_pdf_jax(elapsed_post, V_A_post_LED, theta_A)
    post_jump_pdf = post_jump_pdf_closed_form_jax(
        V_A_base,
        V_A_post_LED,
        theta_A,
        elapsed_post,
        elapsed_pre,
    )
    return jnp.where(
        elapsed_pre <= 0.0,
        post_only_pdf,
        jnp.where(elapsed_post <= 0.0, pre_pdf, post_jump_pdf),
    )


def led_off_pdf_jax(t, V_A_base, theta_A, del_a_minus_del_LED, del_m_plus_del_LED):
    elapsed = jnp.asarray(t, dtype=jnp.float64) - (
        del_a_minus_del_LED + del_m_plus_del_LED
    )
    return inverse_gaussian_pdf_jax(elapsed, V_A_base, theta_A)


# %%
# =============================================================================
# Stable CDF after the LED drift jump
# =============================================================================
@lru_cache(maxsize=None)
def _legendre_nodes_weights(n_quad):
    if n_quad < 4:
        raise ValueError(f"n_quad must be at least 4, got {n_quad}.")
    nodes, weights = np.polynomial.legendre.leggauss(int(n_quad))
    return jnp.asarray(nodes, dtype=jnp.float64), jnp.asarray(weights, dtype=jnp.float64)


def _post_jump_cdf_mass_jax(
    elapsed_post,
    elapsed_pre,
    V_A_base,
    V_A_post_LED,
    theta_A,
    n_quad,
):
    """Probability of surviving to the jump and hitting during the remaining time."""
    elapsed_post, elapsed_pre = jnp.broadcast_arrays(
        jnp.asarray(elapsed_post, dtype=jnp.float64),
        jnp.asarray(elapsed_pre, dtype=jnp.float64),
    )
    original_shape = elapsed_post.shape
    U = elapsed_post.reshape(-1)
    tp = elapsed_pre.reshape(-1)
    valid = (U > 0.0) & (tp > 0.0)
    safe_U = jnp.where(valid, U, 1.0)
    safe_tp = jnp.where(valid, tp, 1.0)

    nodes, weights = _legendre_nodes_weights(int(n_quad))
    z_bound = (theta_A - V_A_base * safe_tp) / jnp.sqrt(safe_tp)
    z_high = jnp.minimum(z_bound, 10.0)
    z_low = jnp.minimum(-10.0, z_high - 10.0)
    half_width = 0.5 * (z_high - z_low)
    midpoint = 0.5 * (z_high + z_low)
    z = midpoint[:, None] + half_width[:, None] * nodes[None, :]
    quad_weights = half_width[:, None] * weights[None, :]

    x_at_jump = V_A_base * safe_tp[:, None] + jnp.sqrt(safe_tp)[:, None] * z
    remaining_bound = theta_A - x_at_jump
    reflection_exponent = 2.0 * theta_A * (x_at_jump - theta_A) / safe_tp[:, None]
    killed_density_z = (
        jnp.exp(-0.5 * z**2)
        / jnp.sqrt(2.0 * jnp.pi)
        * (-jnp.expm1(reflection_exponent))
    )
    post_hit_cdf = inverse_gaussian_cdf_jax(
        safe_U[:, None],
        V_A_post_LED,
        remaining_bound,
    )
    mass = jnp.sum(quad_weights * killed_density_z * post_hit_cdf, axis=1)
    return jnp.where(valid, jnp.maximum(mass, 0.0), 0.0).reshape(original_shape)


def led_on_cdf_jax(
    t,
    t_led,
    V_A_base,
    V_A_post_LED,
    theta_A,
    del_a_minus_del_LED,
    del_m_plus_del_LED,
    n_quad=64,
):
    """Stable untruncated LED-ON first-passage CDF."""
    t, t_led = jnp.broadcast_arrays(
        jnp.asarray(t, dtype=jnp.float64),
        jnp.asarray(t_led, dtype=jnp.float64),
    )
    elapsed_pre = t_led - del_a_minus_del_LED
    elapsed_post = t - t_led - del_m_plus_del_LED
    elapsed_if_pre = t - (del_m_plus_del_LED + del_a_minus_del_LED)

    pre_cdf = inverse_gaussian_cdf_jax(elapsed_if_pre, V_A_base, theta_A)
    post_only_cdf = inverse_gaussian_cdf_jax(elapsed_post, V_A_post_LED, theta_A)
    post_cdf = inverse_gaussian_cdf_jax(elapsed_pre, V_A_base, theta_A) + (
        _post_jump_cdf_mass_jax(
            elapsed_post,
            elapsed_pre,
            V_A_base,
            V_A_post_LED,
            theta_A,
            n_quad,
        )
    )
    cdf = jnp.where(
        elapsed_pre <= 0.0,
        post_only_cdf,
        jnp.where(elapsed_post <= 0.0, pre_cdf, post_cdf),
    )
    return jnp.clip(cdf, 0.0, 1.0)


def led_off_cdf_jax(t, V_A_base, theta_A, del_a_minus_del_LED, del_m_plus_del_LED):
    elapsed = jnp.asarray(t, dtype=jnp.float64) - (
        del_a_minus_del_LED + del_m_plus_del_LED
    )
    return inverse_gaussian_cdf_jax(elapsed, V_A_base, theta_A)


# %%
# =============================================================================
# Unweighted, left-truncated abort/censor likelihood
# =============================================================================
def _log_positive(value):
    value = jnp.asarray(value, dtype=jnp.float64)
    safe_value = jnp.where(jnp.isfinite(value) & (value > 0.0), value, LIKELIHOOD_FLOOR)
    return jnp.log(jnp.maximum(safe_value, LIKELIHOOD_FLOOR))


def proactive_led_step_jump_loglike_jax(params, data, n_quad=64):
    """Joint, unweighted LED-ON/OFF log likelihood."""
    V_A_base = params["V_A_base"]
    V_A_post_LED = params["V_A_post_LED"]
    theta_A = params["theta_A"]
    del_a_minus_del_LED = params["del_a_minus_del_LED"]
    del_m_plus_del_LED = params["del_m_plus_del_LED"]
    T_trunc = data["T_trunc"]

    on_abort_t = data["on_abort_t"]
    on_abort_t_led = data["on_abort_t_led"]
    on_censored_t_stim = data["on_censored_t_stim"]
    on_censored_t_led = data["on_censored_t_led"]
    off_abort_t = data["off_abort_t"]
    off_censored_t_stim = data["off_censored_t_stim"]

    on_abort_pdf = led_on_pdf_jax(
        on_abort_t,
        on_abort_t_led,
        V_A_base,
        V_A_post_LED,
        theta_A,
        del_a_minus_del_LED,
        del_m_plus_del_LED,
    )
    on_abort_cdf_trunc = led_on_cdf_jax(
        jnp.full_like(on_abort_t, T_trunc),
        on_abort_t_led,
        V_A_base,
        V_A_post_LED,
        theta_A,
        del_a_minus_del_LED,
        del_m_plus_del_LED,
        n_quad=n_quad,
    )
    on_abort_loglike = _log_positive(on_abort_pdf) - _log_positive(1.0 - on_abort_cdf_trunc)

    on_censored_cdf_stim = led_on_cdf_jax(
        on_censored_t_stim,
        on_censored_t_led,
        V_A_base,
        V_A_post_LED,
        theta_A,
        del_a_minus_del_LED,
        del_m_plus_del_LED,
        n_quad=n_quad,
    )
    on_censored_cdf_trunc = led_on_cdf_jax(
        jnp.full_like(on_censored_t_stim, T_trunc),
        on_censored_t_led,
        V_A_base,
        V_A_post_LED,
        theta_A,
        del_a_minus_del_LED,
        del_m_plus_del_LED,
        n_quad=n_quad,
    )
    on_censored_loglike = _log_positive(1.0 - on_censored_cdf_stim) - _log_positive(
        1.0 - on_censored_cdf_trunc
    )

    off_abort_pdf = led_off_pdf_jax(
        off_abort_t,
        V_A_base,
        theta_A,
        del_a_minus_del_LED,
        del_m_plus_del_LED,
    )
    off_cdf_trunc = led_off_cdf_jax(
        T_trunc,
        V_A_base,
        theta_A,
        del_a_minus_del_LED,
        del_m_plus_del_LED,
    )
    off_abort_loglike = _log_positive(off_abort_pdf) - _log_positive(1.0 - off_cdf_trunc)

    off_censored_cdf_stim = led_off_cdf_jax(
        off_censored_t_stim,
        V_A_base,
        theta_A,
        del_a_minus_del_LED,
        del_m_plus_del_LED,
    )
    off_censored_loglike = _log_positive(1.0 - off_censored_cdf_stim) - _log_positive(
        1.0 - off_cdf_trunc
    )

    return (
        jnp.sum(on_abort_loglike)
        + jnp.sum(on_censored_loglike)
        + jnp.sum(off_abort_loglike)
        + jnp.sum(off_censored_loglike)
    )


# %%
# =============================================================================
# Trapezoidal priors and NumPyro model
# =============================================================================
def trapezoidal_logpdf_jax(x, hard_low, plausible_low, plausible_high, hard_high):
    x = jnp.asarray(x, dtype=jnp.float64)
    area = ((plausible_low - hard_low) + (hard_high - plausible_high)) / 2.0 + (
        plausible_high - plausible_low
    )
    h_max = 1.0 / area
    rising = ((x - hard_low) / (plausible_low - hard_low)) * h_max
    flat = jnp.full_like(x, h_max)
    falling = ((hard_high - x) / (hard_high - plausible_high)) * h_max
    pdf = jnp.where(
        (hard_low <= x) & (x <= plausible_low),
        rising,
        jnp.where(
            (plausible_low < x) & (x < plausible_high),
            flat,
            jnp.where((plausible_high <= x) & (x <= hard_high), falling, 0.0),
        ),
    )
    return jnp.where(pdf > 0.0, jnp.log(pdf), -jnp.inf)


def sample_trapezoid(name, bounds):
    hard_low, hard_high = bounds["hard"]
    plausible_low, plausible_high = bounds["plausible"]
    value = numpyro.sample(name, dist.Uniform(hard_low, hard_high))
    target_logpdf = trapezoidal_logpdf_jax(
        value,
        hard_low,
        plausible_low,
        plausible_high,
        hard_high,
    )
    uniform_logpdf = -jnp.log(hard_high - hard_low)
    numpyro.factor(f"{name}_trapezoid_prior", target_logpdf - uniform_logpdf)
    return value


def proactive_led_step_jump_model(data, n_quad=64):
    params = {name: sample_trapezoid(name, bounds) for name, bounds in PARAM_BOUNDS.items()}
    loglike = proactive_led_step_jump_loglike_jax(params, data, n_quad=n_quad)
    numpyro.factor("proactive_led_loglike", loglike)


def make_fullrank_guide(model, init_values, init_scale=0.1):
    return AutoMultivariateNormal(
        model,
        init_loc_fn=init_to_value(values=init_values),
        init_scale=init_scale,
    )


def clip_init_to_hard_bounds(init_values):
    clipped = {}
    for name, bounds in PARAM_BOUNDS.items():
        low, high = bounds["hard"]
        eps = max(1e-9, 1e-6 * (high - low))
        clipped[name] = float(np.clip(init_values[name], low + eps, high - eps))
    return clipped


def tree_all_finite(tree):
    import jax

    return bool(
        all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree_util.tree_leaves(tree))
    )
