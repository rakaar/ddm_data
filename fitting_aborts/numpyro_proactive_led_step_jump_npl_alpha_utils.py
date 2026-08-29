# %%
"""JAX likelihood for LED step-jump proactive and NPL+alpha reactive races.

LED-OFF and LED-ON likelihoods are deliberately exposed separately.  They
share only the race algebra; the proactive PDF/CDF is selected explicitly by
the public entry point.
"""

# %%
from pathlib import Path
import sys

import jax.numpy as jnp


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
ANIMAL_FIT_DIR = REPO_DIR / "fit_animal_by_animal"
for import_dir in [SCRIPT_DIR, ANIMAL_FIT_DIR]:
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

import numpyro_npl_alpha_svi_utils as npl_alpha
import numpyro_proactive_led_step_jump_svi_utils as step_jump


# %%
LIKELIHOOD_FLOOR = 1e-50


def _z_e_from_params(params):
    return (params["w"] - 0.5) * 2.0 * params["theta_E"]


def _evidence_cdf(t_evidence, bound, params, ABL, ILD, K_max):
    return npl_alpha.CDF_E_alpha_jax(
        t_evidence,
        bound,
        ABL,
        ILD,
        params["rate_lambda"],
        params["T_0"],
        params["theta_E"],
        _z_e_from_params(params),
        params["rate_norm_l"],
        params["alpha"],
        K_max,
    )


def _evidence_pdf(t_evidence, bound, params, ABL, ILD, K_max):
    return npl_alpha.rho_E_alpha_jax(
        t_evidence,
        bound,
        ABL,
        ILD,
        params["rate_lambda"],
        params["T_0"],
        params["theta_E"],
        _z_e_from_params(params),
        params["rate_norm_l"],
        params["alpha"],
        K_max,
    )


def _ordinary_race_choice_pdf_jax(
    t,
    bound,
    proactive_pdf,
    proactive_cdf,
    params,
    t_stim,
    ABL,
    ILD,
    t_E_aff,
    K_max,
):
    """Choice-specific density for the ordinary proactive/reactive race."""
    t = jnp.asarray(t, dtype=jnp.float64)
    evidence_time = t - t_stim - t_E_aff
    evidence_time_go = evidence_time + params["del_go"]

    cdf_up_go = _evidence_cdf(
        evidence_time_go, 1, params, ABL, ILD, K_max
    )
    cdf_down_go = _evidence_cdf(
        evidence_time_go, -1, params, ABL, ILD, K_max
    )
    evidence_survival_after_go = jnp.clip(
        1.0 - cdf_up_go - cdf_down_go,
        0.0,
        1.0,
    )
    bound_mass_during_go = (
        _evidence_cdf(
            evidence_time_go,
            bound,
            params,
            ABL,
            ILD,
            K_max,
        )
        - _evidence_cdf(
            evidence_time,
            bound,
            params,
            ABL,
            ILD,
            K_max,
        )
    )
    evidence_pdf = _evidence_pdf(
        evidence_time,
        bound,
        params,
        ABL,
        ILD,
        K_max,
    )

    density = proactive_pdf * (
        0.5 * evidence_survival_after_go + bound_mass_during_go
    ) + evidence_pdf * (1.0 - proactive_cdf)
    return jnp.maximum(density, 0.0)


def _ordinary_race_total_cdf_jax(
    t,
    proactive_cdf,
    params,
    t_stim,
    ABL,
    ILD,
    t_E_aff,
    K_max,
):
    """CDF of the first ordinary proactive or reactive response."""
    evidence_time = jnp.asarray(t, dtype=jnp.float64) - t_stim - t_E_aff
    evidence_cdf = _evidence_cdf(
        evidence_time, 1, params, ABL, ILD, K_max
    ) + _evidence_cdf(
        evidence_time, -1, params, ABL, ILD, K_max
    )
    evidence_cdf = jnp.clip(evidence_cdf, 0.0, 1.0)
    race_cdf = proactive_cdf + evidence_cdf - proactive_cdf * evidence_cdf
    return jnp.clip(race_cdf, 0.0, 1.0)


def _exponential_lapse_pdf_jax(t, beta_lapse):
    t = jnp.asarray(t, dtype=jnp.float64)
    value = beta_lapse * jnp.exp(-beta_lapse * jnp.maximum(t, 0.0))
    return jnp.where((t >= 0.0) & (beta_lapse > 0.0), value, 0.0)


def _exponential_lapse_cdf_jax(t, beta_lapse):
    t = jnp.asarray(t, dtype=jnp.float64)
    value = -jnp.expm1(-beta_lapse * jnp.maximum(t, 0.0))
    return jnp.where((t >= 0.0) & (beta_lapse > 0.0), value, 0.0)


def _add_random_choice_lapse_to_pdf(ordinary_pdf, t, params, include_lapse):
    if not include_lapse:
        return ordinary_pdf
    lapse_prob = params["lapse_prob"]
    lapse_pdf = _exponential_lapse_pdf_jax(t, params["beta_lapse"])
    return (1.0 - lapse_prob) * ordinary_pdf + 0.5 * lapse_prob * lapse_pdf


def _add_random_choice_lapse_to_cdf(ordinary_cdf, t, params, include_lapse):
    if not include_lapse:
        return ordinary_cdf
    lapse_prob = params["lapse_prob"]
    lapse_cdf = _exponential_lapse_cdf_jax(t, params["beta_lapse"])
    return jnp.clip(
        (1.0 - lapse_prob) * ordinary_cdf + lapse_prob * lapse_cdf,
        0.0,
        1.0,
    )


# %%
# =============================================================================
# LED-OFF likelihood
# =============================================================================
def led_off_npl_alpha_choice_pdf_jax(
    t,
    bound,
    params,
    t_stim,
    ABL,
    ILD,
    t_E_aff,
    K_max=10,
    include_lapse=False,
):
    proactive_pdf = step_jump.led_off_pdf_jax(
        t,
        params["V_A_base"],
        params["theta_A"],
        params["del_a_minus_del_LED"],
        params["del_m_plus_del_LED"],
    )
    proactive_cdf = step_jump.led_off_cdf_jax(
        t,
        params["V_A_base"],
        params["theta_A"],
        params["del_a_minus_del_LED"],
        params["del_m_plus_del_LED"],
    )
    ordinary_pdf = _ordinary_race_choice_pdf_jax(
        t,
        bound,
        proactive_pdf,
        proactive_cdf,
        params,
        t_stim,
        ABL,
        ILD,
        t_E_aff,
        K_max,
    )
    return _add_random_choice_lapse_to_pdf(
        ordinary_pdf, t, params, include_lapse
    )


def led_off_npl_alpha_total_cdf_jax(
    t,
    params,
    t_stim,
    ABL,
    ILD,
    t_E_aff,
    K_max=10,
    include_lapse=False,
):
    proactive_cdf = step_jump.led_off_cdf_jax(
        t,
        params["V_A_base"],
        params["theta_A"],
        params["del_a_minus_del_LED"],
        params["del_m_plus_del_LED"],
    )
    ordinary_cdf = _ordinary_race_total_cdf_jax(
        t,
        proactive_cdf,
        params,
        t_stim,
        ABL,
        ILD,
        t_E_aff,
        K_max,
    )
    return _add_random_choice_lapse_to_cdf(
        ordinary_cdf, t, params, include_lapse
    )


def led_off_npl_alpha_valid_loglike_jax(
    params,
    data,
    K_max=10,
    include_lapse=False,
):
    pdf = led_off_npl_alpha_choice_pdf_jax(
        data["total_fix"],
        data["choice"],
        params,
        data["t_stim"],
        data["ABL"],
        data["ILD"],
        data["t_E_aff"],
        K_max=K_max,
        include_lapse=include_lapse,
    )
    upper = led_off_npl_alpha_total_cdf_jax(
        data["t_stim"] + data.get("rt_high", 1.0),
        params,
        data["t_stim"],
        data["ABL"],
        data["ILD"],
        data["t_E_aff"],
        K_max=K_max,
        include_lapse=include_lapse,
    )
    lower = led_off_npl_alpha_total_cdf_jax(
        data["t_stim"] + data.get("rt_low", 0.0),
        params,
        data["t_stim"],
        data["ABL"],
        data["ILD"],
        data["t_E_aff"],
        K_max=K_max,
        include_lapse=include_lapse,
    )
    normalizer = jnp.maximum(upper - lower, LIKELIHOOD_FLOOR)
    return jnp.sum(
        jnp.log(jnp.maximum(pdf, LIKELIHOOD_FLOOR))
        - jnp.log(normalizer)
    )


# %%
# =============================================================================
# LED-ON likelihood
# =============================================================================
def led_on_npl_alpha_choice_pdf_jax(
    t,
    bound,
    params,
    t_stim,
    t_LED,
    ABL,
    ILD,
    t_E_aff,
    K_max=10,
    n_quad=64,
    include_lapse=False,
):
    proactive_pdf = step_jump.led_on_pdf_jax(
        t,
        t_LED,
        params["V_A_base"],
        params["V_A_post_LED"],
        params["theta_A"],
        params["del_a_minus_del_LED"],
        params["del_m_plus_del_LED"],
    )
    proactive_cdf = step_jump.led_on_cdf_jax(
        t,
        t_LED,
        params["V_A_base"],
        params["V_A_post_LED"],
        params["theta_A"],
        params["del_a_minus_del_LED"],
        params["del_m_plus_del_LED"],
        n_quad=n_quad,
    )
    ordinary_pdf = _ordinary_race_choice_pdf_jax(
        t,
        bound,
        proactive_pdf,
        proactive_cdf,
        params,
        t_stim,
        ABL,
        ILD,
        t_E_aff,
        K_max,
    )
    return _add_random_choice_lapse_to_pdf(
        ordinary_pdf, t, params, include_lapse
    )


def led_on_npl_alpha_total_cdf_jax(
    t,
    params,
    t_stim,
    t_LED,
    ABL,
    ILD,
    t_E_aff,
    K_max=10,
    n_quad=64,
    include_lapse=False,
):
    proactive_cdf = step_jump.led_on_cdf_jax(
        t,
        t_LED,
        params["V_A_base"],
        params["V_A_post_LED"],
        params["theta_A"],
        params["del_a_minus_del_LED"],
        params["del_m_plus_del_LED"],
        n_quad=n_quad,
    )
    ordinary_cdf = _ordinary_race_total_cdf_jax(
        t,
        proactive_cdf,
        params,
        t_stim,
        ABL,
        ILD,
        t_E_aff,
        K_max,
    )
    return _add_random_choice_lapse_to_cdf(
        ordinary_cdf, t, params, include_lapse
    )


def led_on_npl_alpha_valid_loglike_jax(
    params,
    data,
    K_max=10,
    n_quad=64,
    include_lapse=False,
):
    pdf = led_on_npl_alpha_choice_pdf_jax(
        data["total_fix"],
        data["choice"],
        params,
        data["t_stim"],
        data["t_LED"],
        data["ABL"],
        data["ILD"],
        data["t_E_aff"],
        K_max=K_max,
        n_quad=n_quad,
        include_lapse=include_lapse,
    )
    upper = led_on_npl_alpha_total_cdf_jax(
        data["t_stim"] + data.get("rt_high", 1.0),
        params,
        data["t_stim"],
        data["t_LED"],
        data["ABL"],
        data["ILD"],
        data["t_E_aff"],
        K_max=K_max,
        n_quad=n_quad,
        include_lapse=include_lapse,
    )
    lower = led_on_npl_alpha_total_cdf_jax(
        data["t_stim"] + data.get("rt_low", 0.0),
        params,
        data["t_stim"],
        data["t_LED"],
        data["ABL"],
        data["ILD"],
        data["t_E_aff"],
        K_max=K_max,
        n_quad=n_quad,
        include_lapse=include_lapse,
    )
    normalizer = jnp.maximum(upper - lower, LIKELIHOOD_FLOOR)
    return jnp.sum(
        jnp.log(jnp.maximum(pdf, LIKELIHOOD_FLOOR))
        - jnp.log(normalizer)
    )
