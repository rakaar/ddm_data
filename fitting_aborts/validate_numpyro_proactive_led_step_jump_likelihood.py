# %%
"""Validate the JAX proactive LED step-jump likelihood against NumPy references."""

# %%
# =============================================================================
# Editable parameters
# =============================================================================
from pathlib import Path
import json
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "proactive_led_step_jump_svi_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PDF_ATOL = 1e-10
PDF_RTOL = 1e-6
CDF_ATOL = 1e-7
CDF_RTOL = 1e-5
LOGLIKE_MEAN_ABS_TOL = 1e-5
QUADRATURE_CANDIDATES = [64, 96]
T_TRUNC = 0.3

PARAMETER_SETS = [
    {
        "label": "no jump",
        "V_A_base": 2.0,
        "V_A_post_LED": 2.0,
        "theta_A": 2.0,
        "del_a_minus_del_LED": 0.04,
        "del_m_plus_del_LED": 0.04,
    },
    {
        "label": "upward jump",
        "V_A_base": 1.0,
        "V_A_post_LED": 3.5,
        "theta_A": 1.5,
        "del_a_minus_del_LED": 0.06,
        "del_m_plus_del_LED": 0.03,
    },
    {
        "label": "downward jump",
        "V_A_base": 3.5,
        "V_A_post_LED": 0.8,
        "theta_A": 2.5,
        "del_a_minus_del_LED": -0.02,
        "del_m_plus_del_LED": 0.08,
    },
]
T_LED_VALUES = [0.03, 0.28, 0.90]


# %%
# =============================================================================
# Imports
# =============================================================================
import os

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
import numpyro_proactive_led_step_jump_svi_utils as svi_utils


# %%
# =============================================================================
# Stable NumPy/SciPy reference functions
# =============================================================================
def inverse_gaussian_pdf_numpy(t, drift, bound):
    if t <= 0.0 or bound <= 0.0:
        return 0.0
    return float(
        bound
        / np.sqrt(2.0 * np.pi * t**3)
        * np.exp(-0.5 * (bound - drift * t) ** 2 / t)
    )


def inverse_gaussian_cdf_numpy(t, drift, bound):
    if t <= 0.0 or bound <= 0.0:
        return 0.0
    sqrt_t = np.sqrt(t)
    z_first = (drift * t - bound) / sqrt_t
    z_second = -(drift * t + bound) / sqrt_t
    value = ndtr(z_first) + np.exp(2.0 * drift * bound + log_ndtr(z_second))
    return float(np.clip(value, 0.0, 1.0))


def led_on_pdf_numpy_reference(t, t_led, params):
    elapsed_pre = t_led - params["del_a_minus_del_LED"]
    elapsed_post = t - t_led - params["del_m_plus_del_LED"]
    elapsed_if_pre = t - (
        params["del_m_plus_del_LED"] + params["del_a_minus_del_LED"]
    )
    if elapsed_pre <= 0.0:
        return inverse_gaussian_pdf_numpy(
            elapsed_post,
            params["V_A_post_LED"],
            params["theta_A"],
        )
    if elapsed_post <= 0.0:
        return inverse_gaussian_pdf_numpy(
            elapsed_if_pre,
            params["V_A_base"],
            params["theta_A"],
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


def killed_density_in_z(z, elapsed_pre, V_A_base, theta_A):
    x_at_jump = V_A_base * elapsed_pre + np.sqrt(elapsed_pre) * z
    reflection_exponent = 2.0 * theta_A * (x_at_jump - theta_A) / elapsed_pre
    return (
        np.exp(-0.5 * z**2)
        / np.sqrt(2.0 * np.pi)
        * (-np.expm1(reflection_exponent))
    )


def led_on_cdf_numpy_reference(t, t_led, params):
    elapsed_pre = t_led - params["del_a_minus_del_LED"]
    elapsed_post = t - t_led - params["del_m_plus_del_LED"]
    elapsed_if_pre = t - (
        params["del_m_plus_del_LED"] + params["del_a_minus_del_LED"]
    )
    if elapsed_pre <= 0.0:
        return inverse_gaussian_cdf_numpy(
            elapsed_post,
            params["V_A_post_LED"],
            params["theta_A"],
        )
    if elapsed_post <= 0.0:
        return inverse_gaussian_cdf_numpy(
            elapsed_if_pre,
            params["V_A_base"],
            params["theta_A"],
        )

    z_bound = (
        params["theta_A"] - params["V_A_base"] * elapsed_pre
    ) / np.sqrt(elapsed_pre)

    def integrand(z):
        x_at_jump = params["V_A_base"] * elapsed_pre + np.sqrt(elapsed_pre) * z
        return killed_density_in_z(
            z,
            elapsed_pre,
            params["V_A_base"],
            params["theta_A"],
        ) * inverse_gaussian_cdf_numpy(
            elapsed_post,
            params["V_A_post_LED"],
            params["theta_A"] - x_at_jump,
        )

    post_mass = quad(
        integrand,
        -np.inf,
        z_bound,
        epsabs=1e-11,
        epsrel=1e-10,
        limit=300,
    )[0]
    cdf = inverse_gaussian_cdf_numpy(
        elapsed_pre,
        params["V_A_base"],
        params["theta_A"],
    ) + post_mass
    return float(np.clip(cdf, 0.0, 1.0))


def led_on_cdf_old_1ms_trapz(t, t_led, params):
    if t <= 0.0:
        return 0.0
    time_points = np.arange(0.0, t + 0.001, 0.001)
    density = np.asarray(
        [led_on_pdf_numpy_reference(time, t_led, params) for time in time_points],
        dtype=float,
    )
    return float(np.trapz(density, time_points))


def led_off_pdf_numpy(t, params):
    elapsed = t - params["del_a_minus_del_LED"] - params["del_m_plus_del_LED"]
    return inverse_gaussian_pdf_numpy(elapsed, params["V_A_base"], params["theta_A"])


def led_off_cdf_numpy(t, params):
    elapsed = t - params["del_a_minus_del_LED"] - params["del_m_plus_del_LED"]
    return inverse_gaussian_cdf_numpy(elapsed, params["V_A_base"], params["theta_A"])


# %%
# =============================================================================
# Three-regime PDF/CDF comparison
# =============================================================================
rows = []
for params in PARAMETER_SETS:
    for t_led in T_LED_VALUES:
        change_time = t_led + params["del_m_plus_del_LED"]
        test_times = sorted(
            {
                0.01,
                0.10,
                max(0.001, change_time - 0.01),
                max(0.001, change_time - 0.001),
                change_time + 0.001,
                change_time + 0.005,
                0.30,
                0.45,
                0.80,
                1.20,
                2.00,
            }
        )
        for t in test_times:
            pdf_reference = led_on_pdf_numpy_reference(t, t_led, params)
            cdf_reference = led_on_cdf_numpy_reference(t, t_led, params)
            pdf_jax = float(
                svi_utils.led_on_pdf_jax(
                    t,
                    t_led,
                    **{name: params[name] for name in svi_utils.PARAM_NAMES},
                )
            )
            row = {
                "parameter_set": params["label"],
                "t_led_s": t_led,
                "t_s": t,
                "pdf_reference": pdf_reference,
                "pdf_jax": pdf_jax,
                "pdf_abs_error": abs(pdf_jax - pdf_reference),
                "cdf_reference": cdf_reference,
                "cdf_old_1ms_trapz": led_on_cdf_old_1ms_trapz(t, t_led, params),
            }
            for n_quad in QUADRATURE_CANDIDATES:
                cdf_jax = float(
                    svi_utils.led_on_cdf_jax(
                        t,
                        t_led,
                        **{name: params[name] for name in svi_utils.PARAM_NAMES},
                        n_quad=n_quad,
                    )
                )
                row[f"cdf_jax_{n_quad}"] = cdf_jax
                row[f"cdf_abs_error_{n_quad}"] = abs(cdf_jax - cdf_reference)
            rows.append(row)

comparison_df = pd.DataFrame(rows)
comparison_df["pdf_pass"] = comparison_df["pdf_abs_error"] <= (
    PDF_ATOL + PDF_RTOL * comparison_df["pdf_reference"].abs()
)
for n_quad in QUADRATURE_CANDIDATES:
    comparison_df[f"cdf_pass_{n_quad}"] = comparison_df[f"cdf_abs_error_{n_quad}"] <= (
        CDF_ATOL + CDF_RTOL * comparison_df["cdf_reference"].abs()
    )

comparison_csv = OUTPUT_DIR / "proactive_led_step_jump_pdf_cdf_validation.csv"
comparison_df.to_csv(comparison_csv, index=False)


# %%
# =============================================================================
# Build exact LED7/93 bilateral-ON plus OFF fitting data
# =============================================================================
raw_df = pd.read_csv(REPO_DIR / "out_LED.csv")
fit_source_df = raw_df[
    raw_df["repeat_trial"].isin([0, 2]) | raw_df["repeat_trial"].isna()
].copy()
fit_source_df = fit_source_df[
    (fit_source_df["session_type"] == 7)
    & (fit_source_df["training_level"] == 16)
].copy()
fit_source_df = fit_source_df.dropna(
    subset=["intended_fix", "LED_onset_time", "timed_fix"]
)
fit_source_df = fit_source_df[
    (fit_source_df["abort_event"] == 3)
    | (fit_source_df["success"].isin([1, -1]))
].copy()
fit_source_df = fit_source_df[
    ~((fit_source_df["abort_event"] == 3) & (fit_source_df["timed_fix"] < T_TRUNC))
].copy()
animal_df = fit_source_df[fit_source_df["animal"].astype(int) == 93].copy()
on_df = animal_df[
    (animal_df["LED_trial"] == 1)
    & (animal_df["LED_powerR"] != 0)
    & (animal_df["LED_powerL"] != 0)
].copy()
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
trial_df = trial_df[
    ~((trial_df["RT"] < trial_df["t_stim"]) & (trial_df["RT"] <= T_TRUNC))
].copy()

on_abort_df = trial_df[trial_df["is_led"] & (trial_df["RT"] < trial_df["t_stim"])]
on_censored_df = trial_df[
    trial_df["is_led"]
    & (trial_df["RT"] >= trial_df["t_stim"])
    & (trial_df["t_stim"] > T_TRUNC)
]
off_abort_df = trial_df[~trial_df["is_led"] & (trial_df["RT"] < trial_df["t_stim"])]
off_censored_df = trial_df[
    ~trial_df["is_led"]
    & (trial_df["RT"] >= trial_df["t_stim"])
    & (trial_df["t_stim"] > T_TRUNC)
]

jax_data = {
    "on_abort_t": jnp.asarray(on_abort_df["RT"].to_numpy(dtype=float)),
    "on_abort_t_led": jnp.asarray(on_abort_df["t_LED"].to_numpy(dtype=float)),
    "on_censored_t_stim": jnp.asarray(on_censored_df["t_stim"].to_numpy(dtype=float)),
    "on_censored_t_led": jnp.asarray(on_censored_df["t_LED"].to_numpy(dtype=float)),
    "off_abort_t": jnp.asarray(off_abort_df["RT"].to_numpy(dtype=float)),
    "off_censored_t_stim": jnp.asarray(off_censored_df["t_stim"].to_numpy(dtype=float)),
    "T_trunc": jnp.asarray(T_TRUNC, dtype=jnp.float64),
}


def log_positive_numpy(value):
    if not np.isfinite(value) or value <= 0.0:
        value = svi_utils.LIKELIHOOD_FLOOR
    return float(np.log(max(value, svi_utils.LIKELIHOOD_FLOOR)))


def full_loglike_numpy_reference(params):
    total = 0.0
    for row in on_abort_df.itertuples(index=False):
        pdf = led_on_pdf_numpy_reference(row.RT, row.t_LED, params)
        cdf_trunc = led_on_cdf_numpy_reference(T_TRUNC, row.t_LED, params)
        total += log_positive_numpy(pdf) - log_positive_numpy(1.0 - cdf_trunc)
    for row in on_censored_df.itertuples(index=False):
        cdf_stim = led_on_cdf_numpy_reference(row.t_stim, row.t_LED, params)
        cdf_trunc = led_on_cdf_numpy_reference(T_TRUNC, row.t_LED, params)
        total += log_positive_numpy(1.0 - cdf_stim) - log_positive_numpy(1.0 - cdf_trunc)

    cdf_trunc_off = led_off_cdf_numpy(T_TRUNC, params)
    for row in off_abort_df.itertuples(index=False):
        total += log_positive_numpy(led_off_pdf_numpy(row.RT, params)) - log_positive_numpy(
            1.0 - cdf_trunc_off
        )
    for row in off_censored_df.itertuples(index=False):
        total += log_positive_numpy(1.0 - led_off_cdf_numpy(row.t_stim, params)) - log_positive_numpy(
            1.0 - cdf_trunc_off
        )
    return float(total)


# %%
# =============================================================================
# Full-likelihood comparison and quadrature selection
# =============================================================================
likelihood_params = {
    name: PARAMETER_SETS[0][name] for name in svi_utils.PARAM_NAMES
}
reference_loglike = full_loglike_numpy_reference(likelihood_params)
loglike_rows = []
for n_quad in QUADRATURE_CANDIDATES:
    jax_loglike = float(
        svi_utils.proactive_led_step_jump_loglike_jax(
            {name: jnp.asarray(value) for name, value in likelihood_params.items()},
            jax_data,
            n_quad=n_quad,
        )
    )
    absolute_difference = abs(jax_loglike - reference_loglike)
    loglike_rows.append(
        {
            "n_quad": n_quad,
            "reference_loglike": reference_loglike,
            "jax_loglike": jax_loglike,
            "absolute_difference": absolute_difference,
            "mean_absolute_difference_per_trial": absolute_difference / len(trial_df),
        }
    )
loglike_df = pd.DataFrame(loglike_rows)
loglike_csv = OUTPUT_DIR / "proactive_led_step_jump_full_loglike_validation.csv"
loglike_df.to_csv(loglike_csv, index=False)

pdf_pass = bool(comparison_df["pdf_pass"].all())
node_pass = {}
for n_quad in QUADRATURE_CANDIDATES:
    cdf_pass = bool(comparison_df[f"cdf_pass_{n_quad}"].all())
    mean_loglike_difference = float(
        loglike_df.loc[
            loglike_df["n_quad"] == n_quad,
            "mean_absolute_difference_per_trial",
        ].iloc[0]
    )
    node_pass[n_quad] = bool(
        pdf_pass and cdf_pass and mean_loglike_difference <= LOGLIKE_MEAN_ABS_TOL
    )

recommended_nodes = next(
    (n_quad for n_quad in QUADRATURE_CANDIDATES if node_pass[n_quad]),
    None,
)


# %%
# =============================================================================
# Visual comparison at the median LED timing
# =============================================================================
fig, axes = plt.subplots(3, 2, figsize=(10.0, 10.5), sharex=True)
plot_times = np.linspace(0.001, 1.5, 180)
for row_index, params in enumerate(PARAMETER_SETS):
    t_led = 0.28
    pdf_reference = np.asarray(
        [led_on_pdf_numpy_reference(t, t_led, params) for t in plot_times]
    )
    cdf_reference = np.asarray(
        [led_on_cdf_numpy_reference(t, t_led, params) for t in plot_times]
    )
    pdf_jax = np.asarray(
        svi_utils.led_on_pdf_jax(
            jnp.asarray(plot_times),
            jnp.full_like(jnp.asarray(plot_times), t_led),
            **{name: params[name] for name in svi_utils.PARAM_NAMES},
        )
    )
    axes[row_index, 0].plot(plot_times, pdf_reference, color="black", linewidth=1.2, label="NumPy reference")
    axes[row_index, 0].plot(plot_times, pdf_jax, color="tab:blue", linestyle="--", linewidth=1.0, label="JAX")

    axes[row_index, 1].plot(plot_times, cdf_reference, color="black", linewidth=1.2, label="NumPy reference")
    for n_quad, color in zip(QUADRATURE_CANDIDATES, ["tab:blue", "tab:orange"]):
        cdf_jax = np.asarray(
            svi_utils.led_on_cdf_jax(
                jnp.asarray(plot_times),
                jnp.full_like(jnp.asarray(plot_times), t_led),
                **{name: params[name] for name in svi_utils.PARAM_NAMES},
                n_quad=n_quad,
            )
        )
        axes[row_index, 1].plot(
            plot_times,
            cdf_jax,
            color=color,
            linestyle="--",
            linewidth=1.0,
            label=f"JAX {n_quad} nodes",
        )

    axes[row_index, 0].set_ylabel(f"{params['label']}\nPDF")
    axes[row_index, 1].set_ylabel("CDF")
    for ax in axes[row_index]:
        ax.axvline(t_led + params["del_m_plus_del_LED"], color="0.6", linestyle=":", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)

axes[0, 0].legend(frameon=False, fontsize=8)
axes[0, 1].legend(frameon=False, fontsize=8)
axes[-1, 0].set_xlabel("Observed fixation time (s)")
axes[-1, 1].set_xlabel("Observed fixation time (s)")
fig.tight_layout()
validation_png = OUTPUT_DIR / "proactive_led_step_jump_pdf_cdf_validation.png"
fig.savefig(validation_png, dpi=200, bbox_inches="tight")


# %%
# =============================================================================
# Save and report
# =============================================================================
validation_summary = {
    "status": "passed" if recommended_nodes is not None else "failed",
    "recommended_quadrature_nodes": recommended_nodes,
    "quadrature_candidates": QUADRATURE_CANDIDATES,
    "node_pass": {str(key): value for key, value in node_pass.items()},
    "pdf_pass": pdf_pass,
    "pdf_atol": PDF_ATOL,
    "pdf_rtol": PDF_RTOL,
    "cdf_atol": CDF_ATOL,
    "cdf_rtol": CDF_RTOL,
    "loglike_mean_abs_tolerance": LOGLIKE_MEAN_ABS_TOL,
    "n_trials_led7_93": int(len(trial_df)),
    "n_led_on_trials_led7_93": int(trial_df["is_led"].sum()),
    "n_led_off_trials_led7_93": int((~trial_df["is_led"]).sum()),
    "comparison_csv": str(comparison_csv),
    "full_loglike_csv": str(loglike_csv),
    "figure": str(validation_png),
    "reference_note": (
        "Pass/fail uses the stable intended step-jump CDF. The historical 1 ms "
        "trapz CDF is retained in the comparison CSV as an audit column only."
    ),
}
summary_json = OUTPUT_DIR / "proactive_led_step_jump_validation_summary.json"
summary_json.write_text(json.dumps(validation_summary, indent=2) + "\n")

print("\nFunction-level validation:")
print(f"  PDF pass: {pdf_pass}")
for n_quad in QUADRATURE_CANDIDATES:
    max_cdf_error = comparison_df[f"cdf_abs_error_{n_quad}"].max()
    mean_loglike_error = loglike_df.loc[
        loglike_df["n_quad"] == n_quad,
        "mean_absolute_difference_per_trial",
    ].iloc[0]
    print(
        f"  {n_quad:>2} nodes: pass={node_pass[n_quad]}, "
        f"max CDF abs error={max_cdf_error:.3g}, "
        f"full-loglike mean abs difference/trial={mean_loglike_error:.3g}"
    )
print(f"  Recommended nodes: {recommended_nodes}")
print(f"  Summary: {summary_json}")
print(f"  Figure: {validation_png}")

if recommended_nodes is None:
    raise SystemExit("JAX step-jump likelihood validation failed for both quadrature settings.")
