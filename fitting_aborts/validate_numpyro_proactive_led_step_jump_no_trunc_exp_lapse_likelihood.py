# %%
"""Validate the FCT-matched JAX no-truncation exponential-lapse likelihood."""

# %%
# =============================================================================
# Editable parameters
# =============================================================================
from pathlib import Path
import json
import os
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "proactive_led_step_jump_no_trunc_exp_lapse_svi_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

QUADRATURE_NODES = 64
REFERENCE_QUADRATURE_NODES = 256
PDF_ATOL = 1e-10
PDF_RTOL = 1e-6
CDF_ATOL = 1e-7
CDF_RTOL = 1e-5
MIXTURE_ATOL = 1e-7
LOGLIKE_MEAN_ABS_TOL = 1e-5

PARAMETER_SETS = [
    {
        "label": "no jump",
        "V_A_base": 2.0,
        "V_A_post_LED": 2.0,
        "theta_A": 2.0,
        "del_a_minus_del_LED": 0.04,
        "del_m_plus_del_LED": 0.04,
        "lapse_prob": 0.08,
        "beta_lapse": 4.0,
    },
    {
        "label": "upward jump",
        "V_A_base": 1.0,
        "V_A_post_LED": 3.5,
        "theta_A": 1.5,
        "del_a_minus_del_LED": 0.06,
        "del_m_plus_del_LED": 0.03,
        "lapse_prob": 0.18,
        "beta_lapse": 2.5,
    },
    {
        "label": "downward jump",
        "V_A_base": 3.5,
        "V_A_post_LED": 0.8,
        "theta_A": 2.5,
        "del_a_minus_del_LED": -0.02,
        "del_m_plus_del_LED": 0.08,
        "lapse_prob": 0.03,
        "beta_lapse": 7.0,
    },
]
T_LED_VALUES = [0.03, 0.28, 0.90]


# %%
# =============================================================================
# Imports
# =============================================================================
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import log_ndtr, ndtr

sys.path.insert(0, str(REPO_DIR / "fit_each_condn"))
from psiam_tied_dv_map_utils_with_PDFs import stupid_f_integral

sys.path.insert(0, str(SCRIPT_DIR))
import numpyro_proactive_led_step_jump_no_trunc_exp_lapse_svi_utils as svi_utils


# %%
# =============================================================================
# Independent NumPy/SciPy reference functions
# =============================================================================
def inverse_gaussian_pdf_numpy(t, drift, bound):
    t, drift, bound = np.broadcast_arrays(
        np.asarray(t, dtype=float),
        np.asarray(drift, dtype=float),
        np.asarray(bound, dtype=float),
    )
    valid = (t > 0.0) & (bound > 0.0)
    safe_t = np.where(valid, t, 1.0)
    safe_bound = np.where(valid, bound, 1.0)
    log_pdf = (
        np.log(safe_bound)
        - 0.5 * np.log(2.0 * np.pi)
        - 1.5 * np.log(safe_t)
        - 0.5 * (safe_bound - drift * safe_t) ** 2 / safe_t
    )
    return np.where(valid, np.exp(log_pdf), 0.0)


def inverse_gaussian_cdf_numpy(t, drift, bound):
    t, drift, bound = np.broadcast_arrays(
        np.asarray(t, dtype=float),
        np.asarray(drift, dtype=float),
        np.asarray(bound, dtype=float),
    )
    valid = (t > 0.0) & (bound > 0.0)
    safe_t = np.where(valid, t, 1.0)
    safe_bound = np.where(valid, bound, 1.0)
    sqrt_t = np.sqrt(safe_t)
    z_first = (drift * safe_t - safe_bound) / sqrt_t
    z_second = -(drift * safe_t + safe_bound) / sqrt_t
    cdf = ndtr(z_first) + np.exp(2.0 * drift * safe_bound + log_ndtr(z_second))
    return np.where(valid, np.clip(cdf, 0.0, 1.0), 0.0)


def led_on_pdf_numpy_scalar(t, t_led, params):
    elapsed_pre = t_led - params["del_a_minus_del_LED"]
    elapsed_post = t - t_led - params["del_m_plus_del_LED"]
    elapsed_if_pre = t - (
        params["del_m_plus_del_LED"] + params["del_a_minus_del_LED"]
    )
    if elapsed_pre <= 0.0:
        return float(
            inverse_gaussian_pdf_numpy(
                elapsed_post,
                params["V_A_post_LED"],
                params["theta_A"],
            )
        )
    if elapsed_post <= 0.0:
        return float(
            inverse_gaussian_pdf_numpy(
                elapsed_if_pre,
                params["V_A_base"],
                params["theta_A"],
            )
        )
    return float(
        max(
            stupid_f_integral(
                params["V_A_base"],
                params["V_A_post_LED"],
                params["theta_A"],
                elapsed_post,
                elapsed_pre,
            ),
            0.0,
        )
    )


def led_on_cdf_scipy_scalar(t, t_led, params):
    elapsed_pre = t_led - params["del_a_minus_del_LED"]
    elapsed_post = t - t_led - params["del_m_plus_del_LED"]
    elapsed_if_pre = t - (
        params["del_m_plus_del_LED"] + params["del_a_minus_del_LED"]
    )
    if elapsed_pre <= 0.0:
        return float(
            inverse_gaussian_cdf_numpy(
                elapsed_post,
                params["V_A_post_LED"],
                params["theta_A"],
            )
        )
    if elapsed_post <= 0.0:
        return float(
            inverse_gaussian_cdf_numpy(
                elapsed_if_pre,
                params["V_A_base"],
                params["theta_A"],
            )
        )

    z_bound = (
        params["theta_A"] - params["V_A_base"] * elapsed_pre
    ) / np.sqrt(elapsed_pre)

    def integrand(z):
        x_at_jump = (
            params["V_A_base"] * elapsed_pre + np.sqrt(elapsed_pre) * z
        )
        reflection_exponent = (
            2.0
            * params["theta_A"]
            * (x_at_jump - params["theta_A"])
            / elapsed_pre
        )
        killed_density = (
            np.exp(-0.5 * z**2)
            / np.sqrt(2.0 * np.pi)
            * (-np.expm1(reflection_exponent))
        )
        return killed_density * float(
            inverse_gaussian_cdf_numpy(
                elapsed_post,
                params["V_A_post_LED"],
                params["theta_A"] - x_at_jump,
            )
        )

    post_mass = quad(
        integrand,
        -np.inf,
        z_bound,
        epsabs=1e-11,
        epsrel=1e-10,
        limit=300,
    )[0]
    pre_mass = float(
        inverse_gaussian_cdf_numpy(
            elapsed_pre,
            params["V_A_base"],
            params["theta_A"],
        )
    )
    return float(np.clip(pre_mass + post_mass, 0.0, 1.0))


def led_on_cdf_numpy_vector(t, t_led, params, n_quad=256, chunk_size=1024):
    t, t_led = np.broadcast_arrays(
        np.asarray(t, dtype=float),
        np.asarray(t_led, dtype=float),
    )
    original_shape = t.shape
    t = t.reshape(-1)
    t_led = t_led.reshape(-1)
    result = np.empty_like(t)
    nodes, weights = np.polynomial.legendre.leggauss(int(n_quad))

    for start in range(0, len(t), chunk_size):
        stop = min(start + chunk_size, len(t))
        t_chunk = t[start:stop]
        led_chunk = t_led[start:stop]
        elapsed_pre = led_chunk - params["del_a_minus_del_LED"]
        elapsed_post = t_chunk - led_chunk - params["del_m_plus_del_LED"]
        elapsed_if_pre = t_chunk - (
            params["del_m_plus_del_LED"] + params["del_a_minus_del_LED"]
        )
        pre_cdf = inverse_gaussian_cdf_numpy(
            elapsed_if_pre,
            params["V_A_base"],
            params["theta_A"],
        )
        post_only_cdf = inverse_gaussian_cdf_numpy(
            elapsed_post,
            params["V_A_post_LED"],
            params["theta_A"],
        )

        valid_post = (elapsed_pre > 0.0) & (elapsed_post > 0.0)
        safe_pre = np.where(valid_post, elapsed_pre, 1.0)
        safe_post = np.where(valid_post, elapsed_post, 1.0)
        z_bound = (
            params["theta_A"] - params["V_A_base"] * safe_pre
        ) / np.sqrt(safe_pre)
        z_high = np.minimum(z_bound, 10.0)
        z_low = np.minimum(-10.0, z_high - 10.0)
        half_width = 0.5 * (z_high - z_low)
        midpoint = 0.5 * (z_high + z_low)
        z = midpoint[:, None] + half_width[:, None] * nodes[None, :]
        quad_weights = half_width[:, None] * weights[None, :]
        x_at_jump = (
            params["V_A_base"] * safe_pre[:, None]
            + np.sqrt(safe_pre)[:, None] * z
        )
        remaining_bound = params["theta_A"] - x_at_jump
        reflection_exponent = (
            2.0
            * params["theta_A"]
            * (x_at_jump - params["theta_A"])
            / safe_pre[:, None]
        )
        killed_density = (
            np.exp(-0.5 * z**2)
            / np.sqrt(2.0 * np.pi)
            * (-np.expm1(reflection_exponent))
        )
        post_hit_cdf = inverse_gaussian_cdf_numpy(
            safe_post[:, None],
            params["V_A_post_LED"],
            remaining_bound,
        )
        post_mass = np.sum(quad_weights * killed_density * post_hit_cdf, axis=1)
        post_cdf = inverse_gaussian_cdf_numpy(
            elapsed_pre,
            params["V_A_base"],
            params["theta_A"],
        ) + np.where(valid_post, np.maximum(post_mass, 0.0), 0.0)
        chunk_result = np.where(
            elapsed_pre <= 0.0,
            post_only_cdf,
            np.where(elapsed_post <= 0.0, pre_cdf, post_cdf),
        )
        result[start:stop] = np.clip(chunk_result, 0.0, 1.0)

    return result.reshape(original_shape)


def led_off_pdf_numpy(t, params):
    elapsed = np.asarray(t, dtype=float) - (
        params["del_a_minus_del_LED"] + params["del_m_plus_del_LED"]
    )
    return inverse_gaussian_pdf_numpy(
        elapsed,
        params["V_A_base"],
        params["theta_A"],
    )


def led_off_cdf_numpy(t, params):
    elapsed = np.asarray(t, dtype=float) - (
        params["del_a_minus_del_LED"] + params["del_m_plus_del_LED"]
    )
    return inverse_gaussian_cdf_numpy(
        elapsed,
        params["V_A_base"],
        params["theta_A"],
    )


def log_mix_numpy(log_proactive, log_lapse, lapse_prob):
    return np.logaddexp(
        np.log1p(-lapse_prob) + log_proactive,
        np.log(lapse_prob) + log_lapse,
    )


def log_positive_numpy(value):
    value = np.asarray(value, dtype=float)
    safe = np.where(
        np.isfinite(value) & (value > 0.0),
        value,
        svi_utils.LIKELIHOOD_FLOOR,
    )
    return np.log(np.maximum(safe, svi_utils.LIKELIHOOD_FLOOR))


# %%
# =============================================================================
# Function-level PDF, CDF, and lapse-mixture validation
# =============================================================================
rows = []
for params in PARAMETER_SETS:
    for t_led in T_LED_VALUES:
        change_time = t_led + params["del_m_plus_del_LED"]
        test_times = sorted(
            {
                0.001,
                0.01,
                0.10,
                max(0.001, change_time - 0.001),
                change_time + 0.001,
                change_time + 0.005,
                0.30,
                0.80,
                1.20,
                2.00,
            }
        )
        for t in test_times:
            pdf_reference = led_on_pdf_numpy_scalar(t, t_led, params)
            cdf_reference = led_on_cdf_scipy_scalar(t, t_led, params)
            pdf_jax = float(
                svi_utils.led_on_pdf_jax(
                    t,
                    t_led,
                    **{name: params[name] for name in svi_utils.PARAM_NAMES[:5]},
                )
            )
            cdf_jax = float(
                svi_utils.led_on_cdf_jax(
                    t,
                    t_led,
                    **{name: params[name] for name in svi_utils.PARAM_NAMES[:5]},
                    n_quad=QUADRATURE_NODES,
                )
            )
            lapse_pdf = params["beta_lapse"] * np.exp(-params["beta_lapse"] * t)
            lapse_survival = np.exp(-params["beta_lapse"] * t)
            abort_mix_reference = (
                (1.0 - params["lapse_prob"]) * pdf_reference
                + params["lapse_prob"] * lapse_pdf
            )
            censor_mix_reference = (
                (1.0 - params["lapse_prob"]) * (1.0 - cdf_reference)
                + params["lapse_prob"] * lapse_survival
            )
            abort_mix_jax = float(
                jnp.exp(
                    svi_utils.log_probability_mixture_jax(
                        jnp.log(jnp.maximum(pdf_jax, svi_utils.LIKELIHOOD_FLOOR)),
                        svi_utils.exponential_lapse_logpdf_jax(
                            t, params["beta_lapse"]
                        ),
                        params["lapse_prob"],
                    )
                )
            )
            censor_mix_jax = float(
                jnp.exp(
                    svi_utils.log_probability_mixture_jax(
                        jnp.log(
                            jnp.maximum(
                                1.0 - cdf_jax,
                                svi_utils.LIKELIHOOD_FLOOR,
                            )
                        ),
                        svi_utils.exponential_lapse_logsurvival_jax(
                            t, params["beta_lapse"]
                        ),
                        params["lapse_prob"],
                    )
                )
            )
            rows.append(
                {
                    "parameter_set": params["label"],
                    "t_led_s": t_led,
                    "t_s": t,
                    "pdf_reference": pdf_reference,
                    "pdf_jax": pdf_jax,
                    "pdf_abs_error": abs(pdf_jax - pdf_reference),
                    "cdf_reference": cdf_reference,
                    "cdf_jax": cdf_jax,
                    "cdf_abs_error": abs(cdf_jax - cdf_reference),
                    "abort_mix_reference": abort_mix_reference,
                    "abort_mix_jax": abort_mix_jax,
                    "abort_mix_abs_error": abs(abort_mix_jax - abort_mix_reference),
                    "censor_mix_reference": censor_mix_reference,
                    "censor_mix_jax": censor_mix_jax,
                    "censor_mix_abs_error": abs(
                        censor_mix_jax - censor_mix_reference
                    ),
                }
            )

comparison_df = pd.DataFrame(rows)
comparison_df["pdf_pass"] = comparison_df["pdf_abs_error"] <= (
    PDF_ATOL + PDF_RTOL * comparison_df["pdf_reference"].abs()
)
comparison_df["cdf_pass"] = comparison_df["cdf_abs_error"] <= (
    CDF_ATOL + CDF_RTOL * comparison_df["cdf_reference"].abs()
)
comparison_df["abort_mix_pass"] = (
    comparison_df["abort_mix_abs_error"] <= MIXTURE_ATOL
)
comparison_df["censor_mix_pass"] = (
    comparison_df["censor_mix_abs_error"] <= MIXTURE_ATOL
)
comparison_csv = OUTPUT_DIR / "no_trunc_exp_lapse_function_validation.csv"
comparison_df.to_csv(comparison_csv, index=False)


# %%
# =============================================================================
# Exact historical FCT rows for LED7/93, with no 300 ms removal
# =============================================================================
raw_df = pd.read_csv(REPO_DIR / "out_LED.csv")
source_df = raw_df[
    raw_df["repeat_trial"].isin([0, 2]) | raw_df["repeat_trial"].isna()
].copy()
source_df = source_df[
    (source_df["session_type"] == 7) & (source_df["training_level"] == 16)
].copy()
source_df = source_df.dropna(
    subset=["intended_fix", "LED_onset_time", "timed_fix"]
)
source_df = source_df[
    (source_df["abort_event"] == 3) | source_df["success"].isin([1, -1])
].copy()
animal_df = source_df[source_df["animal"].astype(int) == 93].copy()
on_df = animal_df[animal_df["LED_trial"] == 1].copy()
off_df = animal_df[
    (animal_df["LED_trial"] == 0) | animal_df["LED_trial"].isna()
].copy()

trial_df = pd.concat(
    [
        pd.DataFrame(
            {
                "RT": on_df["timed_fix"].to_numpy(dtype=float),
                "t_stim": on_df["intended_fix"].to_numpy(dtype=float),
                "t_LED": (
                    on_df["intended_fix"] - on_df["LED_onset_time"]
                ).to_numpy(dtype=float),
                "is_led": True,
            }
        ),
        pd.DataFrame(
            {
                "RT": off_df["timed_fix"].to_numpy(dtype=float),
                "t_stim": off_df["intended_fix"].to_numpy(dtype=float),
                "t_LED": np.zeros(len(off_df), dtype=float),
                "is_led": False,
            }
        ),
    ],
    ignore_index=True,
)
if len(trial_df) != 16486 or int(trial_df["is_led"].sum()) != 5595:
    raise RuntimeError(
        "LED7/93 FCT row counts changed: "
        f"total={len(trial_df)}, on={int(trial_df['is_led'].sum())}."
    )
early_abort_count = int(
    np.sum(
        (animal_df["abort_event"] == 3)
        & (animal_df["timed_fix"] < 0.3)
        & (
            (animal_df["LED_trial"] == 1)
            | (animal_df["LED_trial"] == 0)
            | animal_df["LED_trial"].isna()
        )
    )
)
if early_abort_count != 664:
    raise RuntimeError(f"Expected 664 early LED7/93 aborts, found {early_abort_count}.")

on_abort_df = trial_df[trial_df["is_led"] & (trial_df["RT"] < trial_df["t_stim"])]
on_censored_df = trial_df[
    trial_df["is_led"] & (trial_df["RT"] >= trial_df["t_stim"])
]
off_abort_df = trial_df[
    ~trial_df["is_led"] & (trial_df["RT"] < trial_df["t_stim"])
]
off_censored_df = trial_df[
    ~trial_df["is_led"] & (trial_df["RT"] >= trial_df["t_stim"])
]

jax_data = {
    "on_abort_t": jnp.asarray(on_abort_df["RT"].to_numpy(dtype=float)),
    "on_abort_t_led": jnp.asarray(on_abort_df["t_LED"].to_numpy(dtype=float)),
    "on_censored_t_stim": jnp.asarray(
        on_censored_df["t_stim"].to_numpy(dtype=float)
    ),
    "on_censored_t_led": jnp.asarray(
        on_censored_df["t_LED"].to_numpy(dtype=float)
    ),
    "off_abort_t": jnp.asarray(off_abort_df["RT"].to_numpy(dtype=float)),
    "off_censored_t_stim": jnp.asarray(
        off_censored_df["t_stim"].to_numpy(dtype=float)
    ),
}
if "T_trunc" in jax_data:
    raise RuntimeError("No-truncation likelihood data must not contain T_trunc.")


# %%
# =============================================================================
# Full likelihood comparison on all 16,486 LED7/93 rows in all three regimes
# =============================================================================
loglike_rows = []
for parameter_set in PARAMETER_SETS:
    likelihood_params = dict(parameter_set)
    parameter_label = likelihood_params.pop("label")

    on_abort_pdf_reference = np.asarray(
        [
            led_on_pdf_numpy_scalar(row.RT, row.t_LED, likelihood_params)
            for row in on_abort_df.itertuples(index=False)
        ],
        dtype=float,
    )
    on_censored_cdf_reference = led_on_cdf_numpy_vector(
        on_censored_df["t_stim"].to_numpy(dtype=float),
        on_censored_df["t_LED"].to_numpy(dtype=float),
        likelihood_params,
        n_quad=REFERENCE_QUADRATURE_NODES,
    )
    off_abort_pdf_reference = led_off_pdf_numpy(
        off_abort_df["RT"].to_numpy(dtype=float),
        likelihood_params,
    )
    off_censored_cdf_reference = led_off_cdf_numpy(
        off_censored_df["t_stim"].to_numpy(dtype=float),
        likelihood_params,
    )

    p = likelihood_params["lapse_prob"]
    beta = likelihood_params["beta_lapse"]
    reference_loglike = float(
        np.sum(
            log_mix_numpy(
                log_positive_numpy(on_abort_pdf_reference),
                np.log(beta) - beta * on_abort_df["RT"].to_numpy(dtype=float),
                p,
            )
        )
        + np.sum(
            log_mix_numpy(
                log_positive_numpy(1.0 - on_censored_cdf_reference),
                -beta * on_censored_df["t_stim"].to_numpy(dtype=float),
                p,
            )
        )
        + np.sum(
            log_mix_numpy(
                log_positive_numpy(off_abort_pdf_reference),
                np.log(beta) - beta * off_abort_df["RT"].to_numpy(dtype=float),
                p,
            )
        )
        + np.sum(
            log_mix_numpy(
                log_positive_numpy(1.0 - off_censored_cdf_reference),
                -beta * off_censored_df["t_stim"].to_numpy(dtype=float),
                p,
            )
        )
    )
    jax_params = {
        name: jnp.asarray(likelihood_params[name], dtype=jnp.float64)
        for name in svi_utils.PARAM_NAMES
    }
    jax_loglike = float(
        svi_utils.proactive_led_step_jump_no_trunc_lapse_loglike_jax(
            jax_params,
            jax_data,
            n_quad=QUADRATURE_NODES,
        )
    )
    absolute_difference = abs(jax_loglike - reference_loglike)
    mean_difference = absolute_difference / len(trial_df)
    gradient = jax.grad(
        lambda params: svi_utils.proactive_led_step_jump_no_trunc_lapse_loglike_jax(
            params,
            jax_data,
            n_quad=QUADRATURE_NODES,
        )
    )(jax_params)
    loglike_rows.append(
        {
            "parameter_set": parameter_label,
            "n_trials": len(trial_df),
            "reference_quadrature_nodes": REFERENCE_QUADRATURE_NODES,
            "jax_quadrature_nodes": QUADRATURE_NODES,
            "reference_loglike": reference_loglike,
            "jax_loglike": jax_loglike,
            "absolute_difference": absolute_difference,
            "mean_absolute_difference_per_trial": mean_difference,
            "gradient_finite": svi_utils.tree_all_finite(gradient),
        }
    )

loglike_df = pd.DataFrame(loglike_rows)
mean_loglike_difference = float(
    loglike_df["mean_absolute_difference_per_trial"].max()
)
gradient_finite = bool(loglike_df["gradient_finite"].all())
loglike_csv = OUTPUT_DIR / "no_trunc_exp_lapse_full_loglike_validation.csv"
loglike_df.to_csv(loglike_csv, index=False)


# %%
# =============================================================================
# Visual comparison
# =============================================================================
fig, axes = plt.subplots(3, 2, figsize=(10.0, 10.5), sharex=True)
plot_times = np.linspace(0.001, 1.5, 180)
for row_index, params in enumerate(PARAMETER_SETS):
    t_led = 0.28
    pdf_reference = np.asarray(
        [led_on_pdf_numpy_scalar(t, t_led, params) for t in plot_times]
    )
    cdf_reference = np.asarray(
        [led_on_cdf_scipy_scalar(t, t_led, params) for t in plot_times]
    )
    pdf_jax = np.asarray(
        svi_utils.led_on_pdf_jax(
            jnp.asarray(plot_times),
            jnp.full_like(jnp.asarray(plot_times), t_led),
            **{name: params[name] for name in svi_utils.PARAM_NAMES[:5]},
        )
    )
    cdf_jax = np.asarray(
        svi_utils.led_on_cdf_jax(
            jnp.asarray(plot_times),
            jnp.full_like(jnp.asarray(plot_times), t_led),
            **{name: params[name] for name in svi_utils.PARAM_NAMES[:5]},
            n_quad=QUADRATURE_NODES,
        )
    )
    axes[row_index, 0].plot(
        plot_times, pdf_reference, color="black", lw=1.2, label="NumPy reference"
    )
    axes[row_index, 0].plot(
        plot_times, pdf_jax, color="tab:blue", ls="--", lw=1.0, label="JAX"
    )
    axes[row_index, 1].plot(
        plot_times, cdf_reference, color="black", lw=1.2, label="SciPy reference"
    )
    axes[row_index, 1].plot(
        plot_times, cdf_jax, color="tab:blue", ls="--", lw=1.0, label="JAX"
    )
    axes[row_index, 0].set_ylabel(f"{params['label']}\nPDF")
    axes[row_index, 1].set_ylabel("CDF")
    for ax in axes[row_index]:
        ax.axvline(
            t_led + params["del_m_plus_del_LED"],
            color="0.6",
            ls=":",
            lw=0.8,
        )
        ax.spines[["top", "right"]].set_visible(False)

axes[0, 0].legend(frameon=False, fontsize=8)
axes[0, 1].legend(frameon=False, fontsize=8)
axes[-1, 0].set_xlabel("Observed fixation time (s)")
axes[-1, 1].set_xlabel("Observed fixation time (s)")
fig.tight_layout()
validation_png = OUTPUT_DIR / "no_trunc_exp_lapse_pdf_cdf_validation.png"
fig.savefig(validation_png, dpi=200, bbox_inches="tight")


# %%
# =============================================================================
# Save and report
# =============================================================================
function_pass = bool(
    comparison_df[
        ["pdf_pass", "cdf_pass", "abort_mix_pass", "censor_mix_pass"]
    ].all().all()
)
loglike_pass = bool(mean_loglike_difference <= LOGLIKE_MEAN_ABS_TOL)
status = "passed" if function_pass and loglike_pass and gradient_finite else "failed"
summary = {
    "status": status,
    "model": "proactive_led_step_jump_all_on_no_trunc_exp_lapse",
    "recommended_quadrature_nodes": QUADRATURE_NODES,
    "reference_quadrature_nodes": REFERENCE_QUADRATURE_NODES,
    "function_validation_pass": function_pass,
    "full_loglike_pass": loglike_pass,
    "gradient_finite": gradient_finite,
    "full_loglike_mean_abs_tolerance": LOGLIKE_MEAN_ABS_TOL,
    "full_loglike_mean_abs_difference_per_trial": mean_loglike_difference,
    "n_trials_led7_93": int(len(trial_df)),
    "n_led_on_trials_led7_93": int(trial_df["is_led"].sum()),
    "n_led_off_trials_led7_93": int((~trial_df["is_led"]).sum()),
    "n_early_aborts_below_300ms_led7_93": early_abort_count,
    "truncation": "none",
    "function_comparison_csv": str(comparison_csv),
    "full_loglike_csv": str(loglike_csv),
    "figure": str(validation_png),
}
summary_json = OUTPUT_DIR / "no_trunc_exp_lapse_validation_summary.json"
summary_json.write_text(json.dumps(summary, indent=2) + "\n")

print("\nNo-truncation exponential-lapse validation:")
print(f"  Function validation: {function_pass}")
print(f"  Full likelihood: {loglike_pass}")
print(f"  Finite gradients: {gradient_finite}")
print(f"  LED7/93 rows: {len(trial_df)} (early aborts <300 ms: {early_abort_count})")
print(f"  Mean full-loglike difference/trial: {mean_loglike_difference:.3g}")
print(f"  Summary: {summary_json}")
print(f"  Figure: {validation_png}")

if status != "passed":
    raise SystemExit("No-truncation exponential-lapse likelihood validation failed.")
