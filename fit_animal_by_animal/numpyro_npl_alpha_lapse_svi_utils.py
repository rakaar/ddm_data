# %%
"""
JAX/NumPyro helpers for exploratory NPL+alpha+lapse SVI fits.

This extends `numpyro_npl_alpha_svi_utils.py` with the same lapse mixture used
by the old norm+lapse VBMC scripts and by the vanilla/IPL+lapse SVI port.
"""

# %%
from collections import OrderedDict

import jax
import jax.numpy as jnp
import numpy as np
import numpyro

import numpyro_npl_alpha_svi_utils as npl_alpha_utils


# %%
# =============================================================================
# Parameter metadata
# =============================================================================
GLOBAL_PARAM_NAMES = [
    "rate_lambda",
    "T_0",
    "theta_E",
    "w",
    "del_go",
    "rate_norm_l",
    "alpha",
    "lapse_prob",
    "lapse_prob_right",
]

GLOBAL_PARAM_LABELS = {
    **npl_alpha_utils.GLOBAL_PARAM_LABELS,
    "lapse_prob": "lapse_prob",
    "lapse_prob_right": "lapse_prob_right",
}

GLOBAL_BOUNDS = OrderedDict(npl_alpha_utils.GLOBAL_BOUNDS)
GLOBAL_BOUNDS["lapse_prob"] = {"hard": (1e-4, 0.2), "plausible": (1e-3, 0.1)}
GLOBAL_BOUNDS["lapse_prob_right"] = {"hard": (0.001, 0.999), "plausible": (0.4, 0.6)}

DELAY_BOUNDS = dict(npl_alpha_utils.DELAY_BOUNDS)

DEFAULT_INIT_VALUES = {
    "rate_lambda": 1.8,
    "T_0": 150e-3,
    "theta_E": 5.0,
    "w": 0.51,
    "del_go": 0.13,
    "rate_norm_l": 0.9,
    "alpha": 1.0,
    "lapse_prob": 0.02,
    "lapse_prob_right": 0.5,
}


# %%
# =============================================================================
# Reused NPL+alpha likelihood pieces and guide helpers
# =============================================================================
gamma_omega_alpha_jax = npl_alpha_utils.gamma_omega_alpha_jax
CDF_E_alpha_jax = npl_alpha_utils.CDF_E_alpha_jax
rho_E_alpha_jax = npl_alpha_utils.rho_E_alpha_jax
up_or_down_alpha_jax = npl_alpha_utils.up_or_down_alpha_jax
cum_pro_and_reactive_alpha_jax = npl_alpha_utils.cum_pro_and_reactive_alpha_jax

trapezoidal_logpdf_jax = npl_alpha_utils.trapezoidal_logpdf_jax
sample_trapezoid = npl_alpha_utils.sample_trapezoid
sample_trapezoid_vector = npl_alpha_utils.sample_trapezoid_vector
make_guide = npl_alpha_utils.make_guide
tree_all_finite = npl_alpha_utils.tree_all_finite


# %%
# =============================================================================
# NPL+alpha+lapse proactive + evidence likelihood
# =============================================================================
def npl_alpha_lapse_condition_delay_loglike_terms(params, data, K_max=10):
    t_E_aff = params["t_E_aff"][data["condition_id"]]
    Z_E = (params["w"] - 0.5) * 2.0 * params["theta_E"]

    pdf = up_or_down_alpha_jax(
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
        t_E_aff,
        params["del_go"],
        params["rate_norm_l"],
        params["alpha"],
        K_max,
    )
    trunc_factor = (
        cum_pro_and_reactive_alpha_jax(
            data["t_stim"] + 1.0,
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
        - cum_pro_and_reactive_alpha_jax(
            data["t_stim"],
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
    )

    normalized_pdf = jnp.maximum(pdf / (trunc_factor + 1e-20), 1e-50)
    lapse_choice_pdf = jnp.where(
        data["choice"] == 1,
        params["lapse_prob_right"] / data["lapse_rt_window"],
        (1.0 - params["lapse_prob_right"]) / data["lapse_rt_window"],
    )
    mixture_pdf = (1.0 - params["lapse_prob"]) * normalized_pdf + params["lapse_prob"] * lapse_choice_pdf
    mixture_pdf = jnp.maximum(mixture_pdf, 1e-50)
    return pdf, trunc_factor, jnp.log(mixture_pdf)


def npl_alpha_lapse_condition_delay_loglike(params, data, K_max=10):
    _pdf, _trunc_factor, log_pdf = npl_alpha_lapse_condition_delay_loglike_terms(
        params,
        data,
        K_max=K_max,
    )
    return jnp.sum(log_pdf)


# %%
# =============================================================================
# Priors, model, and summary helpers
# =============================================================================
def npl_alpha_lapse_condition_delay_model(data, n_conditions, K_max=10):
    params = {}
    for name, bounds in GLOBAL_BOUNDS.items():
        params[name] = sample_trapezoid(name, bounds["hard"], bounds["plausible"])

    params["t_E_aff"] = sample_trapezoid_vector(
        "t_E_aff",
        n_conditions,
        DELAY_BOUNDS["hard"],
        DELAY_BOUNDS["plausible"],
    )

    loglike = npl_alpha_lapse_condition_delay_loglike(params, data, K_max=K_max)
    numpyro.factor("ddm_loglike", loglike)


def posterior_samples_to_frame(samples, condition_table):
    import pandas as pd

    rows = []
    for name in GLOBAL_PARAM_NAMES:
        values = np.asarray(samples[name], dtype=float)
        finite_values = values[np.isfinite(values)]
        rows.append(
            {
                "parameter": name,
                "mean": float(np.mean(finite_values)) if finite_values.size else np.nan,
                "sd": float(np.std(finite_values)) if finite_values.size else np.nan,
                "q025": float(np.quantile(finite_values, 0.025)) if finite_values.size else np.nan,
                "q500": float(np.quantile(finite_values, 0.5)) if finite_values.size else np.nan,
                "q975": float(np.quantile(finite_values, 0.975)) if finite_values.size else np.nan,
                "n_samples": int(values.size),
                "n_finite": int(finite_values.size),
            }
        )

    delay_values = np.asarray(samples["t_E_aff"], dtype=float)
    for idx, condition in condition_table.reset_index(drop=True).iterrows():
        values = delay_values[:, idx]
        finite_values = values[np.isfinite(values)]
        rows.append(
            {
                "parameter": f"t_E_aff_ABL{int(condition['ABL'])}_ILD{condition['ILD']:g}",
                "mean": float(np.mean(finite_values)) if finite_values.size else np.nan,
                "sd": float(np.std(finite_values)) if finite_values.size else np.nan,
                "q025": float(np.quantile(finite_values, 0.025)) if finite_values.size else np.nan,
                "q500": float(np.quantile(finite_values, 0.5)) if finite_values.size else np.nan,
                "q975": float(np.quantile(finite_values, 0.975)) if finite_values.size else np.nan,
                "n_samples": int(values.size),
                "n_finite": int(finite_values.size),
                "ABL": int(condition["ABL"]),
                "ILD": float(condition["ILD"]),
                "condition_id": int(idx),
            }
        )

    return pd.DataFrame(rows)


def finite_sample_report(samples):
    import pandas as pd

    rows = []
    all_finite = True
    for key, value in samples.items():
        arr = np.asarray(value)
        finite = np.isfinite(arr)
        all_finite = all_finite and bool(np.all(finite))
        rows.append(
            {
                "parameter": key,
                "shape": str(arr.shape),
                "n_total": int(arr.size),
                "n_finite": int(np.sum(finite)),
                "n_nan": int(np.sum(np.isnan(arr))),
                "n_inf": int(np.sum(np.isinf(arr))),
            }
        )
    return pd.DataFrame(rows), all_finite


def clip_init_to_hard_bounds(init_values, eps=1e-6):
    clipped = dict(init_values)
    for name, bounds in GLOBAL_BOUNDS.items():
        hard_low, hard_high = bounds["hard"]
        width = hard_high - hard_low
        clipped[name] = np.clip(float(clipped[name]), hard_low + eps * width, hard_high - eps * width)

    hard_low, hard_high = DELAY_BOUNDS["hard"]
    width = hard_high - hard_low
    clipped["t_E_aff"] = np.clip(
        np.asarray(clipped["t_E_aff"], dtype=float),
        hard_low + eps * width,
        hard_high - eps * width,
    )
    return clipped


def tree_to_numpy(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(x), tree)


# %%
