# %%
"""NumPyro helpers for NPL+alpha RT+choice fits with uniform delays.

The completed point-delay fit is the reference likelihood.  This module changes
only the evidence afferent delay: every condition uses

    D ~ Uniform(t_E_aff_low, t_E_aff_high)

with fitted center and width.  The proactive race terms and the legacy retained
0--1 s window normalization match ``numpyro_npl_alpha_svi_utils.py``.
"""

# %%
from collections import OrderedDict

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

import numpyro_npl_alpha_svi_utils as base_utils
import uniform_delay_likelihood_utils as uniform_utils


# %%
# =============================================================================
# Parameter bounds
# =============================================================================
GLOBAL_PARAM_NAMES = list(base_utils.GLOBAL_PARAM_NAMES)
GLOBAL_BOUNDS = OrderedDict(base_utils.GLOBAL_BOUNDS)

DELAY_WIDTH_HARD = (1e-3, 100e-3)
DELAY_WIDTH_PLAUSIBLE = (5e-3, 50e-3)

# A half-minimum-width margin keeps both endpoints inside [0, 1].  The tiny
# numerical margin avoids a zero-width transform exactly at the hard boundary.
_CENTER_EDGE_MARGIN = DELAY_WIDTH_HARD[0] / 2.0 + 1e-9
DELAY_CENTER_HARD = (_CENTER_EDGE_MARGIN, 1.0 - _CENTER_EDGE_MARGIN)
DELAY_CENTER_PLAUSIBLE = (10e-3, 200e-3)


# %%
def trapezoidal_cdf_jax(x, hard_low, plausible_low, plausible_high, hard_high):
    """CDF matching ``base_utils.trapezoidal_logpdf_jax``."""
    x = jnp.asarray(x, dtype=jnp.float64)
    rising_width = plausible_low - hard_low
    plateau_width = plausible_high - plausible_low
    falling_width = hard_high - plausible_high
    area = 0.5 * rising_width + plateau_width + 0.5 * falling_width
    height = 1.0 / area

    rising_area = 0.5 * height * rising_width
    plateau_area = height * plateau_width

    rising_x = jnp.clip(x - hard_low, 0.0, rising_width)
    rising_cdf = height * rising_x**2 / (2.0 * rising_width)

    plateau_x = jnp.clip(x - plausible_low, 0.0, plateau_width)
    plateau_cdf = rising_area + height * plateau_x

    falling_x = jnp.clip(x - plausible_high, 0.0, falling_width)
    falling_cdf = (
        rising_area
        + plateau_area
        + height
        * (falling_x - falling_x**2 / (2.0 * falling_width))
    )

    result = jnp.where(
        x <= hard_low,
        0.0,
        jnp.where(
            x <= plausible_low,
            rising_cdf,
            jnp.where(
                x <= plausible_high,
                plateau_cdf,
                jnp.where(x < hard_high, falling_cdf, 1.0),
            ),
        ),
    )
    return jnp.clip(result, 0.0, 1.0)


def delay_width_cap_jax(center):
    """Largest allowed width given the center and physical endpoint bounds."""
    center = jnp.asarray(center, dtype=jnp.float64)
    return jnp.minimum(
        DELAY_WIDTH_HARD[1],
        jnp.minimum(2.0 * center, 2.0 * (1.0 - center)),
    )


def delay_width_from_unit_jax(center, width_unit):
    """Map a unit latent to the center-dependent physical width."""
    width_cap = delay_width_cap_jax(center)
    width_span = jnp.maximum(width_cap - DELAY_WIDTH_HARD[0], 1e-12)
    width = DELAY_WIDTH_HARD[0] + width_unit * width_span
    return width, width_cap, width_span


def delay_endpoints_jax(center, width):
    return center - 0.5 * width, center + 0.5 * width


def sample_condition_delay_distribution(n_conditions):
    """Sample centers and conditionally truncated trapezoidal widths."""
    center = base_utils.sample_trapezoid_vector(
        "t_E_aff_center",
        n_conditions,
        DELAY_CENTER_HARD,
        DELAY_CENTER_PLAUSIBLE,
    )
    width_unit = numpyro.sample(
        "t_E_aff_width_unit",
        dist.Uniform(0.0, 1.0).expand([n_conditions]).to_event(1),
    )
    width, width_cap, width_span = delay_width_from_unit_jax(center, width_unit)

    width_logpdf = base_utils.trapezoidal_logpdf_jax(
        width,
        DELAY_WIDTH_HARD[0],
        DELAY_WIDTH_PLAUSIBLE[0],
        DELAY_WIDTH_PLAUSIBLE[1],
        DELAY_WIDTH_HARD[1],
    )
    truncation_mass = trapezoidal_cdf_jax(
        width_cap,
        DELAY_WIDTH_HARD[0],
        DELAY_WIDTH_PLAUSIBLE[0],
        DELAY_WIDTH_PLAUSIBLE[1],
        DELAY_WIDTH_HARD[1],
    )
    # width = low + width_unit * span, so the target density in unit space
    # includes the transform Jacobian ``span``.
    width_unit_target = (
        width_logpdf
        - jnp.log(jnp.maximum(truncation_mass, 1e-300))
        + jnp.log(width_span)
    )
    numpyro.factor("t_E_aff_width_trapezoid_prior", jnp.sum(width_unit_target))

    delay_low, delay_high = delay_endpoints_jax(center, width)
    numpyro.deterministic("t_E_aff_width", width)
    numpyro.deterministic("t_E_aff_low", delay_low)
    numpyro.deterministic("t_E_aff_high", delay_high)
    return center, width_unit, width, delay_low, delay_high


# %%
# =============================================================================
# Uniform-delay RT+choice likelihood with the point-fit race convention
# =============================================================================
def up_or_down_alpha_uniform_delay_legacy_jax(
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
    integrated_cdf_terms,
):
    """Bound-specific race density after analytic delay marginalization."""
    t = jnp.asarray(t, dtype=jnp.float64)
    elapsed = t - t_stim

    cdf_up_after_grace = uniform_utils.uniform_delay_bound_cdf_alpha_jax(
        elapsed + del_go,
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
        integrated_cdf_terms,
    )
    cdf_down_after_grace = uniform_utils.uniform_delay_bound_cdf_alpha_jax(
        elapsed + del_go,
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
        integrated_cdf_terms,
    )
    random_readout_if_evidence_survives = 0.5 * (
        1.0 - cdf_up_after_grace - cdf_down_after_grace
    )

    cdf_bound_after_grace = uniform_utils.uniform_delay_bound_cdf_alpha_jax(
        elapsed + del_go,
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
        integrated_cdf_terms,
    )
    cdf_bound_at_response = uniform_utils.uniform_delay_bound_cdf_alpha_jax(
        elapsed,
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
        integrated_cdf_terms,
    )
    evidence_hits_during_grace = (
        cdf_bound_after_grace - cdf_bound_at_response
    )
    evidence_pdf = uniform_utils.uniform_delay_bound_pdf_alpha_jax(
        elapsed,
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

    p_A = base_utils.rho_A_t_jax(t - t_A_aff, V_A, theta_A)
    c_A = base_utils.cum_A_t_jax(t - t_A_aff, V_A, theta_A)
    return p_A * (
        random_readout_if_evidence_survives + evidence_hits_during_grace
    ) + evidence_pdf * (1.0 - c_A)


def cum_pro_and_reactive_alpha_uniform_delay_legacy_jax(
    t,
    c_A_trunc_time,
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
    integrated_cdf_terms,
):
    """Choice-collapsed race CDF matching the completed point-delay fit."""
    t = jnp.asarray(t, dtype=jnp.float64)
    c_A = base_utils.cum_A_t_jax(t - t_A_aff, V_A, theta_A)
    if c_A_trunc_time is not None:
        truncation_survival = 1.0 - base_utils.cum_A_t_jax(
            c_A_trunc_time - t_A_aff,
            V_A,
            theta_A,
        )
        c_A = jnp.where(
            t < c_A_trunc_time,
            0.0,
            c_A / jnp.maximum(truncation_survival, 1e-20),
        )

    elapsed = t - t_stim
    c_E = uniform_utils.uniform_delay_bound_cdf_alpha_jax(
        elapsed,
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
        integrated_cdf_terms,
    ) + uniform_utils.uniform_delay_bound_cdf_alpha_jax(
        elapsed,
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
        integrated_cdf_terms,
    )
    return c_A + c_E - c_A * c_E


def npl_alpha_uniform_delay_loglike(
    params,
    data,
    K_max=10,
    integrated_cdf_terms=200,
):
    condition_id = data["condition_id"]
    center = params["t_E_aff_center"][condition_id]
    width = params["t_E_aff_width"][condition_id]
    delay_low, delay_high = delay_endpoints_jax(center, width)
    Z_E = (params["w"] - 0.5) * 2.0 * params["theta_E"]

    pdf = up_or_down_alpha_uniform_delay_legacy_jax(
        data["total_fix"],
        data["choice"],
        data["V_A"],
        data["theta_A"],
        data["t_A_aff"],
        data["t_stim"],
        data["ABL"],
        data["ILD"],
        params["rate_lambda"],
        params["T_0"],
        params["theta_E"],
        Z_E,
        delay_low,
        delay_high,
        params["del_go"],
        params["rate_norm_l"],
        params["alpha"],
        K_max,
        integrated_cdf_terms,
    )

    def race_cdf(absolute_time):
        return cum_pro_and_reactive_alpha_uniform_delay_legacy_jax(
            absolute_time,
            data["T_trunc"],
            data["V_A"],
            data["theta_A"],
            data["t_A_aff"],
            data["t_stim"],
            data["ABL"],
            data["ILD"],
            params["rate_lambda"],
            params["T_0"],
            params["theta_E"],
            Z_E,
            delay_low,
            delay_high,
            params["rate_norm_l"],
            params["alpha"],
            integrated_cdf_terms,
        )

    retained_mass = race_cdf(data["t_stim"] + 1.0) - race_cdf(
        data["t_stim"]
    )
    normalized_pdf = jnp.maximum(pdf / (retained_mass + 1e-20), 1e-50)
    return jnp.sum(jnp.log(normalized_pdf))


# %%
def npl_alpha_uniform_delay_model(
    data,
    n_conditions,
    K_max=10,
    integrated_cdf_terms=200,
):
    params = {}
    for name, bounds in GLOBAL_BOUNDS.items():
        params[name] = base_utils.sample_trapezoid(
            name,
            bounds["hard"],
            bounds["plausible"],
        )

    center, width_unit, width, _, _ = sample_condition_delay_distribution(
        n_conditions
    )
    params["t_E_aff_center"] = center
    params["t_E_aff_width_unit"] = width_unit
    params["t_E_aff_width"] = width

    loglike = npl_alpha_uniform_delay_loglike(
        params,
        data,
        K_max=K_max,
        integrated_cdf_terms=integrated_cdf_terms,
    )
    numpyro.factor("ddm_loglike", loglike)


def parameter_count(n_conditions):
    return len(GLOBAL_PARAM_NAMES) + 2 * int(n_conditions)
