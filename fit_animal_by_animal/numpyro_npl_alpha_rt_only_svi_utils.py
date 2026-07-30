# %%
"""
JAX/NumPyro helpers for choice-collapsed NPL+alpha condition-delay fits.

Both modes fit the same six global NPL+alpha parameters and one t_E_aff per
observed ABL/signed-ILD condition:

    reactive_only
    proactive_reactive

The proactive parameters are fixed data inputs in the second mode. `del_go`
is absent because it cancels exactly after summing the upper- and lower-choice
densities.
"""

# %%
from collections import OrderedDict

import jax.numpy as jnp
import numpy as np
import numpyro

import numpyro_npl_alpha_reactive_svi_utils as reactive_utils
import numpyro_npl_alpha_svi_utils as base_utils


# %%
# =============================================================================
# Parameter metadata and shared output helpers
# =============================================================================
PROCESS_MODES = ("reactive_only", "proactive_reactive")

GLOBAL_PARAM_NAMES = list(reactive_utils.GLOBAL_PARAM_NAMES)
GLOBAL_PARAM_LABELS = dict(reactive_utils.GLOBAL_PARAM_LABELS)
GLOBAL_BOUNDS = OrderedDict(reactive_utils.GLOBAL_BOUNDS)
DELAY_BOUNDS = dict(reactive_utils.DELAY_BOUNDS)

make_guide = reactive_utils.make_guide
tree_all_finite = reactive_utils.tree_all_finite
tree_to_numpy = reactive_utils.tree_to_numpy
finite_sample_report = reactive_utils.finite_sample_report
posterior_samples_to_frame = reactive_utils.posterior_samples_to_frame
clip_init_to_hard_bounds = reactive_utils.clip_init_to_hard_bounds


def normalize_process_mode(process_mode):
    mode = str(process_mode).strip().lower().replace("-", "_")
    if mode not in PROCESS_MODES:
        raise ValueError(
            f"process_mode must be one of {PROCESS_MODES}, got "
            f"{process_mode!r}."
        )
    return mode


# %%
# =============================================================================
# Choice-collapsed RT densities
# =============================================================================
def choice_collapsed_reactive_pdf(params, data, K_max=10):
    """Sum the reactive first-passage densities for both evidence bounds."""
    t_E_aff = params["t_E_aff"][data["condition_id"]]
    Z_E = (params["w"] - 0.5) * 2.0 * params["theta_E"]
    reactive_time = data["rt_wrt_stim"] - t_E_aff

    return (
        base_utils.rho_E_alpha_jax(
            reactive_time,
            1,
            data["ABL"],
            data["ILD"],
            params["rate_lambda"],
            params["T_0"],
            params["theta_E"],
            Z_E,
            params["rate_norm_l"],
            params["alpha"],
            K_max,
        )
        + base_utils.rho_E_alpha_jax(
            reactive_time,
            -1,
            data["ABL"],
            data["ILD"],
            params["rate_lambda"],
            params["T_0"],
            params["theta_E"],
            Z_E,
            params["rate_norm_l"],
            params["alpha"],
            K_max,
        )
    )


def choice_collapsed_proactive_reactive_pdf(
    params,
    data,
    K_max=10,
    del_go_s=0.0,
):
    """
    Sum the existing bound-specific proactive+reactive race densities.

    `del_go_s` is exposed only for validation. The returned sum is invariant
    to it up to floating-point roundoff.
    """
    t_E_aff = params["t_E_aff"][data["condition_id"]]
    Z_E = (params["w"] - 0.5) * 2.0 * params["theta_E"]

    common_args = (
        data["total_fix"],
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
        t_E_aff,
        jnp.asarray(del_go_s, dtype=jnp.float64),
        params["rate_norm_l"],
        params["alpha"],
        K_max,
    )
    return (
        base_utils.up_or_down_alpha_jax(
            common_args[0],
            1,
            *common_args[1:],
        )
        + base_utils.up_or_down_alpha_jax(
            common_args[0],
            -1,
            *common_args[1:],
        )
    )


def choice_collapsed_proactive_reactive_pdf_simplified(
    params,
    data,
    K_max=10,
):
    """Algebraically collapsed race density, with no `del_go` term."""
    t_E_aff = params["t_E_aff"][data["condition_id"]]
    Z_E = (params["w"] - 0.5) * 2.0 * params["theta_E"]
    reactive_time = data["rt_wrt_stim"] - t_E_aff

    p_A = base_utils.rho_A_t_jax(
        data["total_fix"] - data["t_A_aff"],
        data["V_A"],
        data["theta_A"],
    )
    c_A = base_utils.cum_A_t_jax(
        data["total_fix"] - data["t_A_aff"],
        data["V_A"],
        data["theta_A"],
    )
    p_E = (
        base_utils.rho_E_alpha_jax(
            reactive_time,
            1,
            data["ABL"],
            data["ILD"],
            params["rate_lambda"],
            params["T_0"],
            params["theta_E"],
            Z_E,
            params["rate_norm_l"],
            params["alpha"],
            K_max,
        )
        + base_utils.rho_E_alpha_jax(
            reactive_time,
            -1,
            data["ABL"],
            data["ILD"],
            params["rate_lambda"],
            params["T_0"],
            params["theta_E"],
            Z_E,
            params["rate_norm_l"],
            params["alpha"],
            K_max,
        )
    )
    c_E = (
        base_utils.CDF_E_alpha_jax(
            reactive_time,
            1,
            data["ABL"],
            data["ILD"],
            params["rate_lambda"],
            params["T_0"],
            params["theta_E"],
            Z_E,
            params["rate_norm_l"],
            params["alpha"],
            K_max,
        )
        + base_utils.CDF_E_alpha_jax(
            reactive_time,
            -1,
            data["ABL"],
            data["ILD"],
            params["rate_lambda"],
            params["T_0"],
            params["theta_E"],
            Z_E,
            params["rate_norm_l"],
            params["alpha"],
            K_max,
        )
    )
    return p_A * (1.0 - c_E) + p_E * (1.0 - c_A)


# %%
# =============================================================================
# Retained-window likelihood
# =============================================================================
def reactive_retained_window_mass(params, data, K_max=10):
    t_E_aff = params["t_E_aff"][data["condition_id"]]
    Z_E = (params["w"] - 0.5) * 2.0 * params["theta_E"]

    def both_bound_cdf(rt):
        reactive_time = rt - t_E_aff
        return (
            base_utils.CDF_E_alpha_jax(
                reactive_time,
                1,
                data["ABL"],
                data["ILD"],
                params["rate_lambda"],
                params["T_0"],
                params["theta_E"],
                Z_E,
                params["rate_norm_l"],
                params["alpha"],
                K_max,
            )
            + base_utils.CDF_E_alpha_jax(
                reactive_time,
                -1,
                data["ABL"],
                data["ILD"],
                params["rate_lambda"],
                params["T_0"],
                params["theta_E"],
                Z_E,
                params["rate_norm_l"],
                params["alpha"],
                K_max,
            )
        )

    return both_bound_cdf(data["rt_upper"]) - both_bound_cdf(
        data["rt_lower"]
    )


def proactive_reactive_retained_window_mass(params, data, K_max=10):
    t_E_aff = params["t_E_aff"][data["condition_id"]]
    Z_E = (params["w"] - 0.5) * 2.0 * params["theta_E"]

    def race_cdf(rt_wrt_stim):
        return base_utils.cum_pro_and_reactive_alpha_jax(
            data["t_stim"] + rt_wrt_stim,
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
            t_E_aff,
            params["rate_norm_l"],
            params["alpha"],
            K_max,
        )

    return race_cdf(data["rt_upper"]) - race_cdf(data["rt_lower"])


def npl_alpha_rt_only_condition_delay_loglike_terms(
    params,
    data,
    process_mode,
    K_max=10,
):
    mode = normalize_process_mode(process_mode)
    if mode == "reactive_only":
        pdf = choice_collapsed_reactive_pdf(
            params,
            data,
            K_max=K_max,
        )
        retained_window_mass = reactive_retained_window_mass(
            params,
            data,
            K_max=K_max,
        )
    else:
        pdf = choice_collapsed_proactive_reactive_pdf_simplified(
            params,
            data,
            K_max=K_max,
        )
        retained_window_mass = proactive_reactive_retained_window_mass(
            params,
            data,
            K_max=K_max,
        )

    retained_window_mass = jnp.maximum(retained_window_mass, 1e-20)
    normalized_pdf = jnp.maximum(pdf / retained_window_mass, 1e-50)
    return {
        "pdf": pdf,
        "retained_window_mass": retained_window_mass,
        "normalized_pdf": normalized_pdf,
        "loglike": jnp.log(normalized_pdf),
    }


def npl_alpha_rt_only_condition_delay_loglike(
    params,
    data,
    process_mode,
    K_max=10,
):
    terms = npl_alpha_rt_only_condition_delay_loglike_terms(
        params,
        data,
        process_mode,
        K_max=K_max,
    )
    return jnp.sum(terms["loglike"])


# %%
# =============================================================================
# Priors and NumPyro model
# =============================================================================
def npl_alpha_rt_only_condition_delay_model(
    data,
    n_conditions,
    process_mode,
    K_max=10,
):
    mode = normalize_process_mode(process_mode)
    params = {}
    for name, bounds in GLOBAL_BOUNDS.items():
        params[name] = base_utils.sample_trapezoid(
            name,
            bounds["hard"],
            bounds["plausible"],
        )

    params["t_E_aff"] = base_utils.sample_trapezoid_vector(
        "t_E_aff",
        n_conditions,
        DELAY_BOUNDS["hard"],
        DELAY_BOUNDS["plausible"],
    )

    loglike = npl_alpha_rt_only_condition_delay_loglike(
        params,
        data,
        mode,
        K_max=K_max,
    )
    numpyro.factor("ddm_loglike", loglike)


# %%
def parameter_count(n_conditions):
    return len(GLOBAL_PARAM_NAMES) + int(n_conditions)


def assert_no_del_go(params):
    if "del_go" in params:
        raise ValueError("RT-only parameter dictionaries must not contain del_go.")
    return True
