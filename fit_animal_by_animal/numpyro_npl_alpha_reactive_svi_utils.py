# %%
"""
Reactive-only JAX/NumPyro helpers for NPL+alpha condition-delay SVI fits.

The fit uses only valid trials retained in a stimulus-relative RT window. It
does not contain the proactive process or `del_go`. The likelihood for each
trial is the bound-specific reactive density divided by the probability that
the reactive process hits either bound inside the retained RT window.
"""

# %%
from collections import OrderedDict

import jax
import numpy as np
import jax.numpy as jnp
import numpyro

import numpyro_npl_alpha_svi_utils as base_utils


# %%
# =============================================================================
# Parameter metadata
# =============================================================================
GLOBAL_PARAM_NAMES = [
    "rate_lambda",
    "T_0",
    "theta_E",
    "w",
    "rate_norm_l",
    "alpha",
]

GLOBAL_PARAM_LABELS = {
    "rate_lambda": "lambda",
    "T_0": "T_0",
    "theta_E": "theta_E",
    "w": "w",
    "rate_norm_l": "rate_norm_l",
    "alpha": "alpha",
}

GLOBAL_BOUNDS = OrderedDict(
    (name, base_utils.GLOBAL_BOUNDS[name]) for name in GLOBAL_PARAM_NAMES
)
DELAY_BOUNDS = base_utils.DELAY_BOUNDS

CDF_E_alpha_jax = base_utils.CDF_E_alpha_jax
rho_E_alpha_jax = base_utils.rho_E_alpha_jax
make_guide = base_utils.make_guide
tree_all_finite = base_utils.tree_all_finite


# %%
# =============================================================================
# Retained-window reactive likelihood
# =============================================================================
def npl_alpha_reactive_condition_delay_loglike_terms(params, data, K_max=10):
    """
    Return per-trial pieces of the reactive RT+choice likelihood.

    `rt_wrt_stim` and `t_E_aff` are in seconds. The denominator collapses over
    both choices because trial inclusion depends on RT, not on the observed
    choice.
    """
    t_E_aff = params["t_E_aff"][data["condition_id"]]
    Z_E = (params["w"] - 0.5) * 2.0 * params["theta_E"]

    reactive_time = data["rt_wrt_stim"] - t_E_aff
    pdf = rho_E_alpha_jax(
        reactive_time,
        data["choice"],
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

    lower_reactive_time = data["rt_lower"] - t_E_aff
    upper_reactive_time = data["rt_upper"] - t_E_aff
    lower_mass = (
        CDF_E_alpha_jax(
            lower_reactive_time,
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
        + CDF_E_alpha_jax(
            lower_reactive_time,
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
    upper_mass = (
        CDF_E_alpha_jax(
            upper_reactive_time,
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
        + CDF_E_alpha_jax(
            upper_reactive_time,
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

    retained_window_mass = jnp.maximum(upper_mass - lower_mass, 1e-20)
    normalized_pdf = jnp.maximum(pdf / retained_window_mass, 1e-50)
    return {
        "pdf": pdf,
        "retained_window_mass": retained_window_mass,
        "normalized_pdf": normalized_pdf,
        "loglike": jnp.log(normalized_pdf),
    }


def npl_alpha_reactive_condition_delay_loglike(params, data, K_max=10):
    terms = npl_alpha_reactive_condition_delay_loglike_terms(
        params,
        data,
        K_max=K_max,
    )
    return jnp.sum(terms["loglike"])


# %%
# =============================================================================
# Priors and NumPyro model
# =============================================================================
def npl_alpha_reactive_condition_delay_model(data, n_conditions, K_max=10):
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

    loglike = npl_alpha_reactive_condition_delay_loglike(
        params,
        data,
        K_max=K_max,
    )
    numpyro.factor("ddm_loglike", loglike)


# %%
# =============================================================================
# Output helpers
# =============================================================================
def clip_init_to_hard_bounds(init_values, eps=1e-6):
    clipped = dict(init_values)
    for name, bounds in GLOBAL_BOUNDS.items():
        hard_low, hard_high = bounds["hard"]
        width = hard_high - hard_low
        clipped[name] = np.clip(
            float(clipped[name]),
            hard_low + eps * width,
            hard_high - eps * width,
        )

    hard_low, hard_high = DELAY_BOUNDS["hard"]
    width = hard_high - hard_low
    clipped["t_E_aff"] = np.clip(
        np.asarray(clipped["t_E_aff"], dtype=float),
        hard_low + eps * width,
        hard_high - eps * width,
    )
    return clipped


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
                "ABL": np.nan,
                "ILD": np.nan,
                "condition_id": np.nan,
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


def tree_to_numpy(tree):
    return jax.tree_util.tree_map(lambda value: np.asarray(value), tree)
