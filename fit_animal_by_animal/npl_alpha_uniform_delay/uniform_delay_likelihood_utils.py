# %%
"""Analytic uniform-delay helpers for the NPL+alpha PSIAM likelihood.

This is an experimental utility.  It leaves the production point-delay
likelihood in ``numpyro_npl_alpha_svi_utils.py`` unchanged.
"""

# %%
import jax.numpy as jnp
from jax.scipy.special import log_ndtr

import numpyro_npl_alpha_svi_utils as npl_utils


# %%
def cum_A_t_stable_jax(t, V_A, theta_A):
    """Inverse-Gaussian proactive CDF with a stable extreme-tail term."""
    t = jnp.asarray(t, dtype=jnp.float64)
    safe_t = jnp.maximum(t, 1e-12)
    sqrt_t = jnp.sqrt(safe_t)
    first_argument = V_A * (safe_t - theta_A / V_A) / sqrt_t
    second_argument = -V_A * (safe_t + theta_A / V_A) / sqrt_t
    first_term = jnp.exp(log_ndtr(first_argument))
    second_term = jnp.exp(
        2.0 * V_A * theta_A + log_ndtr(second_argument)
    )
    return jnp.where(t > 0.0, first_term + second_term, 0.0)


# %%
def proactive_pdf_cdf_stable_jax(
    t,
    V_A,
    theta_A,
    t_A_aff,
    truncation_time=None,
):
    """Proactive PDF/CDF, optionally conditioned on surviving a left cutoff."""
    t = jnp.asarray(t, dtype=jnp.float64)
    proactive_time = t - t_A_aff
    p_A = npl_utils.rho_A_t_jax(proactive_time, V_A, theta_A)
    c_A = cum_A_t_stable_jax(proactive_time, V_A, theta_A)
    if truncation_time is None:
        return p_A, c_A

    truncation_time = jnp.asarray(truncation_time, dtype=jnp.float64)
    c_A_at_truncation = cum_A_t_stable_jax(
        truncation_time - t_A_aff,
        V_A,
        theta_A,
    )
    truncation_survival = jnp.maximum(1.0 - c_A_at_truncation, 1e-12)
    after_truncation = t >= truncation_time
    p_A = jnp.where(after_truncation, p_A / truncation_survival, 0.0)
    c_A = jnp.where(
        after_truncation,
        (c_A - c_A_at_truncation) / truncation_survival,
        0.0,
    )
    return p_A, jnp.clip(c_A, 0.0, 1.0)


# %%
def lower_bound_hit_probability_jax(v, w, boundary_separation=2.0):
    """Eventual probability of hitting the transformed lower DDM bound."""
    v = jnp.asarray(v, dtype=jnp.float64)
    w = jnp.asarray(w, dtype=jnp.float64)
    q = -2.0 * v * boundary_separation
    near_zero = jnp.abs(q) < 1e-8
    safe_q = jnp.where(near_zero, 1.0, q)
    drifted_probability = (
        jnp.exp(safe_q * w)
        * jnp.expm1(safe_q * (1.0 - w))
        / jnp.expm1(safe_q)
    )
    return jnp.where(near_zero, 1.0 - w, drifted_probability)


def integrated_CDF_E_gamma_omega_with_w_jax(
    t,
    gamma,
    omega,
    bound,
    w,
    n_terms,
):
    """Finite spectral expression for H(t) = integral_0^t F_bound(u) du."""
    t_original = jnp.asarray(t, dtype=jnp.float64)
    gamma = jnp.asarray(gamma, dtype=jnp.float64)
    omega = jnp.asarray(omega, dtype=jnp.float64)
    bound = jnp.asarray(bound)
    w = jnp.asarray(w, dtype=jnp.float64)

    boundary_separation = 2.0
    v = jnp.where(bound == 1, -gamma, gamma)
    transformed_w = jnp.where(bound == 1, 1.0 - w, w)

    shape = jnp.broadcast_shapes(
        jnp.shape(t_original),
        jnp.shape(v),
        jnp.shape(omega),
        jnp.shape(transformed_w),
    )
    t_full = jnp.broadcast_to(t_original, shape)
    v_full = jnp.broadcast_to(v, shape)
    omega_full = jnp.broadcast_to(omega, shape)
    w_full = jnp.broadcast_to(transformed_w, shape)
    valid = t_full > 0.0
    tau = jnp.where(valid, omega_full * t_full, 0.0)

    k = jnp.arange(1, n_terms + 1, dtype=jnp.float64)
    k = k.reshape((1,) * len(shape) + (n_terms,))
    v_terms = v_full[..., None]
    w_terms = w_full[..., None]
    tau_terms = tau[..., None]

    eigenvalue = 0.5 * (
        v_terms**2 + (k * jnp.pi / boundary_separation) ** 2
    )
    coefficient = (
        jnp.pi
        / boundary_separation**2
        * jnp.exp(-v_terms * boundary_separation * w_terms)
        * k
        * jnp.sin(k * jnp.pi * w_terms)
    )
    integrated_transient = jnp.sum(
        coefficient * (-jnp.expm1(-eigenvalue * tau_terms)) / eigenvalue**2,
        axis=-1,
    )
    eventual_probability = lower_bound_hit_probability_jax(
        v_full,
        w_full,
        boundary_separation,
    )
    integrated_cdf = (
        eventual_probability * tau - integrated_transient
    ) / omega_full
    return jnp.where(valid, integrated_cdf, 0.0)


def integrated_CDF_E_alpha_jax(
    t,
    bound,
    ABL,
    ILD,
    rate_lambda,
    T_0,
    theta_E,
    Z_E,
    rate_norm_l,
    alpha,
    n_terms,
):
    """NPL+alpha wrapper around the generic Gamma/Omega antiderivative."""
    gamma, omega = npl_utils.gamma_omega_alpha_jax(
        ABL,
        ILD,
        rate_lambda,
        T_0,
        theta_E,
        rate_norm_l,
        alpha,
    )
    w = 0.5 + Z_E / (2.0 * theta_E)
    return integrated_CDF_E_gamma_omega_with_w_jax(
        t,
        gamma,
        omega,
        bound,
        w,
        n_terms,
    )


# %%
def uniform_delay_bound_cdf_alpha_jax(
    elapsed_without_delay,
    bound,
    delay_low,
    delay_high,
    ABL,
    ILD,
    rate_lambda,
    T_0,
    theta_E,
    Z_E,
    rate_norm_l,
    alpha,
    n_terms,
):
    """Bound CDF averaged over D ~ Uniform(delay_low, delay_high)."""
    delay_width = delay_high - delay_low
    h_low = integrated_CDF_E_alpha_jax(
        elapsed_without_delay - delay_low,
        bound,
        ABL,
        ILD,
        rate_lambda,
        T_0,
        theta_E,
        Z_E,
        rate_norm_l,
        alpha,
        n_terms,
    )
    h_high = integrated_CDF_E_alpha_jax(
        elapsed_without_delay - delay_high,
        bound,
        ABL,
        ILD,
        rate_lambda,
        T_0,
        theta_E,
        Z_E,
        rate_norm_l,
        alpha,
        n_terms,
    )
    return (h_low - h_high) / delay_width


def uniform_delay_bound_pdf_alpha_jax(
    elapsed_without_delay,
    bound,
    delay_low,
    delay_high,
    ABL,
    ILD,
    rate_lambda,
    T_0,
    theta_E,
    Z_E,
    rate_norm_l,
    alpha,
    K_max,
):
    """Bound PDF averaged over D, expressed as a CDF difference."""
    delay_width = delay_high - delay_low
    cdf_low = npl_utils.CDF_E_alpha_jax(
        elapsed_without_delay - delay_low,
        bound,
        ABL,
        ILD,
        rate_lambda,
        T_0,
        theta_E,
        Z_E,
        rate_norm_l,
        alpha,
        K_max,
    )
    cdf_high = npl_utils.CDF_E_alpha_jax(
        elapsed_without_delay - delay_high,
        bound,
        ABL,
        ILD,
        rate_lambda,
        T_0,
        theta_E,
        Z_E,
        rate_norm_l,
        alpha,
        K_max,
    )
    return (cdf_low - cdf_high) / delay_width


def up_or_down_alpha_uniform_delay_jax(
    t,
    bound,
    V_A,
    theta_A,
    t_A_aff,
    t_stim,
    ABL,
    ILD,
    rate_lambda,
    T_0,
    theta_E,
    Z_E,
    delay_low,
    delay_high,
    del_go,
    rate_norm_l,
    alpha,
    K_max,
    n_terms,
    proactive_truncation_time=None,
):
    """Joint RT/choice density after analytic uniform-delay marginalization."""
    t = jnp.asarray(t, dtype=jnp.float64)
    elapsed_without_delay = t - t_stim

    cdf_up_after_grace = uniform_delay_bound_cdf_alpha_jax(
        elapsed_without_delay + del_go,
        1,
        delay_low,
        delay_high,
        ABL,
        ILD,
        rate_lambda,
        T_0,
        theta_E,
        Z_E,
        rate_norm_l,
        alpha,
        n_terms,
    )
    cdf_down_after_grace = uniform_delay_bound_cdf_alpha_jax(
        elapsed_without_delay + del_go,
        -1,
        delay_low,
        delay_high,
        ABL,
        ILD,
        rate_lambda,
        T_0,
        theta_E,
        Z_E,
        rate_norm_l,
        alpha,
        n_terms,
    )
    random_readout_if_EA_survives = 0.5 * (
        1.0 - cdf_up_after_grace - cdf_down_after_grace
    )

    cdf_bound_after_grace = uniform_delay_bound_cdf_alpha_jax(
        elapsed_without_delay + del_go,
        bound,
        delay_low,
        delay_high,
        ABL,
        ILD,
        rate_lambda,
        T_0,
        theta_E,
        Z_E,
        rate_norm_l,
        alpha,
        n_terms,
    )
    cdf_bound_at_response = uniform_delay_bound_cdf_alpha_jax(
        elapsed_without_delay,
        bound,
        delay_low,
        delay_high,
        ABL,
        ILD,
        rate_lambda,
        T_0,
        theta_E,
        Z_E,
        rate_norm_l,
        alpha,
        n_terms,
    )
    evidence_hits_during_grace = (
        cdf_bound_after_grace - cdf_bound_at_response
    )
    evidence_pdf = uniform_delay_bound_pdf_alpha_jax(
        elapsed_without_delay,
        bound,
        delay_low,
        delay_high,
        ABL,
        ILD,
        rate_lambda,
        T_0,
        theta_E,
        Z_E,
        rate_norm_l,
        alpha,
        K_max,
    )

    p_A, c_A = proactive_pdf_cdf_stable_jax(
        t,
        V_A,
        theta_A,
        t_A_aff,
        proactive_truncation_time,
    )
    return p_A * (
        random_readout_if_EA_survives + evidence_hits_during_grace
    ) + evidence_pdf * (1.0 - c_A)


def cum_pro_and_reactive_alpha_uniform_delay_jax(
    t,
    V_A,
    theta_A,
    t_A_aff,
    t_stim,
    ABL,
    ILD,
    rate_lambda,
    T_0,
    theta_E,
    Z_E,
    delay_low,
    delay_high,
    rate_norm_l,
    alpha,
    n_terms,
    proactive_truncation_time=None,
):
    """Choice-collapsed race CDF, optionally after proactive left truncation."""
    t = jnp.asarray(t, dtype=jnp.float64)
    _, c_A = proactive_pdf_cdf_stable_jax(
        t,
        V_A,
        theta_A,
        t_A_aff,
        proactive_truncation_time,
    )
    elapsed_without_delay = t - t_stim
    c_E = uniform_delay_bound_cdf_alpha_jax(
        elapsed_without_delay,
        1,
        delay_low,
        delay_high,
        ABL,
        ILD,
        rate_lambda,
        T_0,
        theta_E,
        Z_E,
        rate_norm_l,
        alpha,
        n_terms,
    ) + uniform_delay_bound_cdf_alpha_jax(
        elapsed_without_delay,
        -1,
        delay_low,
        delay_high,
        ABL,
        ILD,
        rate_lambda,
        T_0,
        theta_E,
        Z_E,
        rate_norm_l,
        alpha,
        n_terms,
    )
    return c_A + c_E - c_A * c_E


def up_or_down_alpha_fixed_delay_stable_proactive_jax(
    t,
    bound,
    V_A,
    theta_A,
    t_A_aff,
    t_stim,
    ABL,
    ILD,
    rate_lambda,
    T_0,
    theta_E,
    Z_E,
    t_E_aff,
    del_go,
    rate_norm_l,
    alpha,
    K_max,
    proactive_truncation_time=None,
):
    """Point-delay reference likelihood using the stable proactive CDF."""
    t = jnp.asarray(t, dtype=jnp.float64)
    elapsed = t - t_stim - t_E_aff
    t1 = jnp.maximum(elapsed, 1e-6)
    t2 = jnp.maximum(elapsed + del_go, 1e-6)

    p_A, c_A = proactive_pdf_cdf_stable_jax(
        t,
        V_A,
        theta_A,
        t_A_aff,
        proactive_truncation_time,
    )
    cdf_up_after_grace = npl_utils.CDF_E_alpha_jax(
        elapsed + del_go,
        1,
        ABL,
        ILD,
        rate_lambda,
        T_0,
        theta_E,
        Z_E,
        rate_norm_l,
        alpha,
        K_max,
    )
    cdf_down_after_grace = npl_utils.CDF_E_alpha_jax(
        elapsed + del_go,
        -1,
        ABL,
        ILD,
        rate_lambda,
        T_0,
        theta_E,
        Z_E,
        rate_norm_l,
        alpha,
        K_max,
    )
    random_readout_if_EA_survives = 0.5 * (
        1.0 - cdf_up_after_grace - cdf_down_after_grace
    )
    evidence_hits_during_grace = npl_utils.CDF_E_alpha_jax(
        t2,
        bound,
        ABL,
        ILD,
        rate_lambda,
        T_0,
        theta_E,
        Z_E,
        rate_norm_l,
        alpha,
        K_max,
    ) - npl_utils.CDF_E_alpha_jax(
        t1,
        bound,
        ABL,
        ILD,
        rate_lambda,
        T_0,
        theta_E,
        Z_E,
        rate_norm_l,
        alpha,
        K_max,
    )
    evidence_pdf = npl_utils.rho_E_alpha_jax(
        elapsed,
        bound,
        ABL,
        ILD,
        rate_lambda,
        T_0,
        theta_E,
        Z_E,
        rate_norm_l,
        alpha,
        K_max,
    )
    return p_A * (
        random_readout_if_EA_survives + evidence_hits_during_grace
    ) + evidence_pdf * (1.0 - c_A)


def cum_pro_and_reactive_alpha_fixed_delay_stable_proactive_jax(
    t,
    V_A,
    theta_A,
    t_A_aff,
    t_stim,
    ABL,
    ILD,
    rate_lambda,
    T_0,
    theta_E,
    Z_E,
    t_E_aff,
    rate_norm_l,
    alpha,
    K_max,
    proactive_truncation_time=None,
):
    """Fixed-delay race CDF used by independent delay quadrature checks."""
    t = jnp.asarray(t, dtype=jnp.float64)
    _, c_A = proactive_pdf_cdf_stable_jax(
        t,
        V_A,
        theta_A,
        t_A_aff,
        proactive_truncation_time,
    )
    elapsed = t - t_stim - t_E_aff
    c_E = npl_utils.CDF_E_alpha_jax(
        elapsed,
        1,
        ABL,
        ILD,
        rate_lambda,
        T_0,
        theta_E,
        Z_E,
        rate_norm_l,
        alpha,
        K_max,
    ) + npl_utils.CDF_E_alpha_jax(
        elapsed,
        -1,
        ABL,
        ILD,
        rate_lambda,
        T_0,
        theta_E,
        Z_E,
        rate_norm_l,
        alpha,
        K_max,
    )
    return c_A + c_E - c_A * c_E
