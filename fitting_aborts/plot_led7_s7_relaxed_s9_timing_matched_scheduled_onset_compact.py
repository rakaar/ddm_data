# %%
"""Compare all-s7 and relaxed-matched onset-aligned LED OFF/ON fits."""

# %%
# =============================================================================
# Editable parameters
# =============================================================================
from pathlib import Path
from time import perf_counter
import json
import os
import pickle
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
ORIGINAL_FIT_ROOT = (
    SCRIPT_DIR
    / "numpyro_svi_led7_s7_all_vs_s9_timing_matched_"
    "no_trunc_exp_lapse_outputs"
)
RELAXED_FIT_ROOT = (
    SCRIPT_DIR
    / "numpyro_svi_led7_s7_relaxed_s9_timing_matched_"
    "no_trunc_exp_lapse_outputs"
)
DIAGNOSTIC_PAYLOAD_PATH = (
    RELAXED_FIT_ROOT / "summary_figures" / "theory_data_diagnostic_payload.pkl"
)
OUTPUT_FIGURE_PATH = (
    SCRIPT_DIR
    / "led7_s7_all_vs_relaxed_s9_timing_matched_svi_"
    "abort_rate_scheduled_onset_1x2.png"
)
OUTPUT_AUDIT_PATH = (
    SCRIPT_DIR
    / "led7_s7_all_vs_relaxed_s9_timing_matched_svi_"
    "abort_rate_scheduled_onset_1x2_audit.csv"
)

EXPECTED_MATCHING_PROFILE = "relaxed_unequal"
DATASET_SPECS = (
    {
        "key": "s7_all",
        "title": "All session type 7 trials",
        "fit_dir": ORIGINAL_FIT_ROOT / "s7_all",
        "counts": {0: (14_215, 88_414), 1: (8_276, 42_485)},
    },
    {
        "key": "s7_matched_to_s9",
        "title": "Timing-matched subset",
        "fit_dir": RELAXED_FIT_ROOT / "s7_matched_to_s9",
        "counts": {0: (1_914, 13_238), 1: (1_416, 6_403)},
    },
)
LED_SPECS = (
    {"value": 0, "label": "LED OFF", "color": "tab:blue"},
    {"value": 1, "label": "LED ON", "color": "tab:red"},
)
DISPLAY_RANGE_S = (-1.0, 1.0)
DATA_BIN_S = 0.020
MODEL_DT_S = 0.001
MC_SAMPLES_PER_CURVE = 5_000
MC_RANDOM_SEED = 20_260_912
MODEL_TRIAL_CHUNK = 256
MASS_TOLERANCE = 2.0e-3
OUTPUT_DPI = 250
SHOW_PLOT = False


# %%
# =============================================================================
# Load and validate the completed fit and reusable theory/data payload
# =============================================================================
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

jax.config.update("jax_enable_x64", True)

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpyro_proactive_led_step_jump_no_trunc_exp_lapse_svi_utils as svi_utils

required_paths = [DIAGNOSTIC_PAYLOAD_PATH]
required_paths.extend(spec["fit_dir"] / "run_summary.json" for spec in DATASET_SPECS)
required_paths.extend(
    spec["fit_dir"] / "main_fullrank_posterior_summary.csv"
    for spec in DATASET_SPECS
)
required_paths.extend(
    spec["fit_dir"] / "source_row_manifest.csv" for spec in DATASET_SPECS
)
for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

for dataset_spec in DATASET_SPECS:
    run_summary = json.loads(
        (dataset_spec["fit_dir"] / "run_summary.json").read_text()
    )
    if run_summary.get("status") != "complete":
        raise RuntimeError(
            f"{dataset_spec['key']} status is {run_summary.get('status')!r}."
        )
    if run_summary.get("n_nonfinite_losses") != 0:
        raise RuntimeError(f"{dataset_spec['key']} reports non-finite losses.")
    if not run_summary.get("all_posterior_samples_finite", False):
        raise RuntimeError(
            f"{dataset_spec['key']} reports non-finite posterior samples."
        )
    config = run_summary.get("config", {})
    if config.get("truncation") != "none":
        raise RuntimeError(f"{dataset_spec['key']} is not the no-truncation fit.")
    if config.get("pooling") != "trial_weighted":
        raise RuntimeError(f"{dataset_spec['key']} is not trial pooled.")
    if (
        dataset_spec["key"] == "s7_matched_to_s9"
        and config.get("matching_profile") != EXPECTED_MATCHING_PROFILE
    ):
        raise RuntimeError("The subset fit is not the relaxed timing match.")

    parameter_summary = pd.read_csv(
        dataset_spec["fit_dir"] / "main_fullrank_posterior_summary.csv"
    )
    if parameter_summary["parameter"].tolist() != svi_utils.PARAM_NAMES:
        raise RuntimeError(
            f"{dataset_spec['key']} posterior parameter order changed."
        )
    dataset_spec["posterior_means"] = dict(
        zip(parameter_summary["parameter"], parameter_summary["mean"])
    )

    manifest = pd.read_csv(dataset_spec["fit_dir"] / "source_row_manifest.csv")
    required_manifest_columns = {
        "LED_trial",
        "intended_fix",
        "effective_scheduled_onset",
        "likelihood_role",
    }
    missing_columns = sorted(required_manifest_columns - set(manifest.columns))
    if missing_columns:
        raise RuntimeError(
            f"{dataset_spec['key']} manifest is missing {missing_columns}."
        )
    fit_frame = manifest.loc[
        manifest["likelihood_role"].isin(["abort", "censored"])
    ].copy()
    for led_value, (expected_abort, expected_total) in dataset_spec["counts"].items():
        led_frame = fit_frame.loc[fit_frame["LED_trial"].eq(led_value)]
        observed_abort = int(led_frame["likelihood_role"].eq("abort").sum())
        if len(led_frame) != expected_total or observed_abort != expected_abort:
            raise RuntimeError(
                f"{dataset_spec['key']} LED {led_value} manifest counts changed."
            )
    dataset_spec["fit_frame"] = fit_frame

with DIAGNOSTIC_PAYLOAD_PATH.open("rb") as handle:
    diagnostic_payload = pickle.load(handle)

if diagnostic_payload.get("diagnostic_profile") != EXPECTED_MATCHING_PROFILE:
    raise RuntimeError("Diagnostic payload is not from the relaxed matching profile.")
if not np.isclose(diagnostic_payload.get("data_bin_s", np.nan), DATA_BIN_S):
    raise RuntimeError("Empirical bin width changed.")
if not np.isclose(diagnostic_payload.get("model_dt_s", np.nan), MODEL_DT_S):
    raise RuntimeError("Theory resolution changed.")
if tuple(diagnostic_payload.get("onset_display_range_s", ())) != DISPLAY_RANGE_S:
    raise RuntimeError("Scheduled-onset display range changed.")

missing_datasets = [
    spec["key"]
    for spec in DATASET_SPECS
    if spec["key"] not in diagnostic_payload["diagnostics"]
]
if missing_datasets:
    raise RuntimeError(f"Diagnostic payload is missing {missing_datasets}.")


# %%
# =============================================================================
# Monte Carlo timing-average theory and comparison with the exact curves
# =============================================================================
def monte_carlo_scheduled_onset_rate(led_value, timing_frame, params, model_t):
    t_stim = timing_frame["intended_fix"].to_numpy(dtype=float)
    t_led = timing_frame["effective_scheduled_onset"].to_numpy(dtype=float)
    model_t_jax = jnp.asarray(model_t, dtype=jnp.float64)
    proactive_sum = np.zeros_like(model_t, dtype=float)
    lapse_sum = np.zeros_like(model_t, dtype=float)

    for start in range(0, len(timing_frame), MODEL_TRIAL_CHUNK):
        stop = min(start + MODEL_TRIAL_CHUNK, len(timing_frame))
        chunk_t_stim = jnp.asarray(t_stim[start:stop], dtype=jnp.float64)
        chunk_t_led = jnp.asarray(t_led[start:stop], dtype=jnp.float64)
        physical_t = model_t_jax[None, :] + chunk_t_led[:, None]
        risk_mask = (physical_t >= 0.0) & (
            physical_t < chunk_t_stim[:, None]
        )

        if led_value == 1:
            proactive_pdf = svi_utils.led_on_pdf_jax(
                physical_t,
                chunk_t_led[:, None],
                params["V_A_base"],
                params["V_A_post_LED"],
                params["theta_A"],
                params["del_a_minus_del_LED"],
                params["del_m_plus_del_LED"],
            )
        else:
            proactive_pdf = svi_utils.led_off_pdf_jax(
                physical_t,
                params["V_A_base"],
                params["theta_A"],
                params["del_a_minus_del_LED"],
                params["del_m_plus_del_LED"],
            )
        lapse_pdf = params["beta_lapse"] * jnp.exp(
            -params["beta_lapse"] * jnp.maximum(physical_t, 0.0)
        )
        proactive_sum += np.asarray(
            jax.device_get(
                jnp.sum(jnp.where(risk_mask, proactive_pdf, 0.0), axis=0)
            ),
            dtype=float,
        )
        lapse_sum += np.asarray(
            jax.device_get(
                jnp.sum(jnp.where(risk_mask, lapse_pdf, 0.0), axis=0)
            ),
            dtype=float,
        )

    proactive_rate = (
        (1.0 - params["lapse_prob"]) * proactive_sum / len(timing_frame)
    )
    lapse_rate = params["lapse_prob"] * lapse_sum / len(timing_frame)
    mixture_rate = proactive_rate + lapse_rate
    if not np.isfinite(mixture_rate).all() or (mixture_rate < -1e-12).any():
        raise RuntimeError("Monte Carlo theory curve is invalid.")
    return mixture_rate


audit_rows = []
plot_payloads = {}
sampling_rng = np.random.default_rng(MC_RANDOM_SEED)
theory_start = perf_counter()
for dataset_spec in DATASET_SPECS:
    dataset_key = dataset_spec["key"]
    plot_payloads[dataset_key] = {}
    for led_spec in LED_SPECS:
        led_value = led_spec["value"]
        payload = diagnostic_payload["diagnostics"][dataset_key][led_value][
            "scheduled_onset"
        ]
        empirical = payload["empirical"]
        model = payload["model"]
        analytic = payload["analytic_mass"]
        expected_abort, expected_total = dataset_spec["counts"][led_value]

        if empirical["n_abort"] != expected_abort:
            raise RuntimeError(f"{dataset_key} LED {led_value}: abort count changed.")
        if empirical["n_total"] != expected_total:
            raise RuntimeError(f"{dataset_key} LED {led_value}: denominator changed.")
        observed_mass = expected_abort / expected_total
        if not np.isclose(empirical["data_mass"], observed_mass, atol=1e-12):
            raise RuntimeError(f"{dataset_key} LED {led_value}: data area changed.")
        if not np.allclose(
            model["mixture_rate"],
            model["proactive_rate"] + model["lapse_rate"],
            atol=1e-12,
            rtol=1e-12,
        ):
            raise RuntimeError(
                f"{dataset_key} LED {led_value}: theory components do not sum."
            )
        if abs(model["mixture_mass"] - analytic["mixture"]) > MASS_TOLERANCE:
            raise RuntimeError(
                f"{dataset_key} LED {led_value}: numerical/analytic mass mismatch."
            )
        if not all(
            np.isfinite(np.asarray(values)).all()
            for values in (
                empirical["data_centers_s"],
                empirical["data_rate"],
                model["model_t_s"],
                model["mixture_rate"],
            )
        ):
            raise RuntimeError(
                f"{dataset_key} LED {led_value}: curve contains non-finite values."
            )

        source_frame = dataset_spec["fit_frame"].loc[
            dataset_spec["fit_frame"]["LED_trial"].eq(led_value)
        ].copy()
        sampled_positions = sampling_rng.integers(
            0,
            len(source_frame),
            size=MC_SAMPLES_PER_CURVE,
        )
        sampled_timing = source_frame.iloc[sampled_positions][
            ["intended_fix", "effective_scheduled_onset"]
        ].copy()
        curve_start = perf_counter()
        mc_theory = monte_carlo_scheduled_onset_rate(
            led_value,
            sampled_timing,
            dataset_spec["posterior_means"],
            np.asarray(model["model_t_s"], dtype=float),
        )
        curve_elapsed_seconds = perf_counter() - curve_start
        mc_mass = float(np.sum(mc_theory) * MODEL_DT_S)
        exact_theory = np.asarray(model["mixture_rate"], dtype=float)
        mc_minus_exact = mc_theory - exact_theory
        integrated_absolute_mc_error = float(
            np.sum(np.abs(mc_minus_exact)) * MODEL_DT_S
        )
        rmse_mc_error = float(np.sqrt(np.mean(mc_minus_exact**2)))

        plot_payloads[dataset_key][led_value] = {
            **payload,
            "mc_mixture_rate": mc_theory,
        }
        audit_rows.append(
            {
                "dataset": dataset_key,
                "LED_group": led_spec["label"],
                "n_abort": empirical["n_abort"],
                "n_total": empirical["n_total"],
                "observed_abort_fraction": empirical["data_mass"],
                "predicted_abort_fraction": analytic["mixture"],
                "available_schedule_rows": len(source_frame),
                "mc_timing_samples": MC_SAMPLES_PER_CURVE,
                "mc_sampling_with_replacement": True,
                "exact_theory_mass": model["mixture_mass"],
                "mc_theory_mass": mc_mass,
                "mc_minus_exact_mass": mc_mass - model["mixture_mass"],
                "integrated_absolute_mc_minus_exact": (
                    integrated_absolute_mc_error
                ),
                "rmse_mc_minus_exact_rate_per_s": rmse_mc_error,
                "max_abs_mc_minus_exact_rate_per_s": float(
                    np.max(np.abs(mc_minus_exact))
                ),
                "curve_elapsed_seconds_including_first_compile": (
                    curve_elapsed_seconds
                ),
            }
        )

theory_elapsed_seconds = perf_counter() - theory_start
audit = pd.DataFrame(audit_rows)
audit["total_four_curve_elapsed_seconds"] = theory_elapsed_seconds
audit.to_csv(OUTPUT_AUDIT_PATH, index=False)


# %%
# =============================================================================
# Compact 1 x 2 data-versus-theory figure
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), sharex=True, sharey=True)
for ax, dataset_spec in zip(axes, DATASET_SPECS):
    for led_spec in LED_SPECS:
        payload = plot_payloads[dataset_spec["key"]][led_spec["value"]]
        empirical = payload["empirical"]
        model = payload["model"]
        ax.step(
            empirical["data_centers_s"],
            empirical["data_rate"],
            where="mid",
            color=led_spec["color"],
            linewidth=1.0,
            alpha=0.48,
            label=f"{led_spec['label']} data",
        )
        ax.plot(
            model["model_t_s"],
            payload["mc_mixture_rate"],
            color=led_spec["color"],
            linewidth=2.2,
            label=f"{led_spec['label']} theory (5k MC)",
        )
    ax.axvline(0.0, color="0.55", linestyle=":", linewidth=0.9, zorder=0)
    ax.set_xlim(DISPLAY_RANGE_S)
    ax.set_title(dataset_spec["title"])
    ax.set_xlabel("Time from scheduled LED onset (s)")
    ax.grid(axis="y", color="0.88", linewidth=0.55)
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_ylabel(r"Abort rate (s$^{-1}$)")
fig.legend(
    handles=[
        Line2D([0], [0], color="tab:blue", linewidth=1.0, alpha=0.48, label="LED OFF data"),
        Line2D(
            [0],
            [0],
            color="tab:blue",
            linewidth=2.2,
            label="LED OFF theory (5k MC)",
        ),
        Line2D([0], [0], color="tab:red", linewidth=1.0, alpha=0.48, label="LED ON data"),
        Line2D(
            [0],
            [0],
            color="tab:red",
            linewidth=2.2,
            label="LED ON theory (5k MC)",
        ),
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=4,
    frameon=False,
)
fig.suptitle(
    "LED7 session type 7: abort rate aligned to scheduled LED onset",
    y=1.08,
)
fig.tight_layout()
fig.savefig(OUTPUT_FIGURE_PATH, dpi=OUTPUT_DPI, bbox_inches="tight")
if SHOW_PLOT:
    plt.show()
else:
    plt.close(fig)

print(f"Original fit root: {ORIGINAL_FIT_ROOT.resolve()}")
print(f"Relaxed fit root: {RELAXED_FIT_ROOT.resolve()}")
print(f"Monte Carlo timing samples per curve: {MC_SAMPLES_PER_CURVE:,}")
print(f"Monte Carlo seed: {MC_RANDOM_SEED}")
print(f"Four-curve theory time: {theory_elapsed_seconds:.3f} seconds")
print(audit.to_string(index=False))
print(f"\nFigure: {OUTPUT_FIGURE_PATH.resolve()}")
print(f"Audit: {OUTPUT_AUDIT_PATH.resolve()}")
