# %%
"""Plot pooled LED7 abort RTDs in fixation coordinates with lapse components."""

# %%
# =============================================================================
# Editable paths and plotting settings
# =============================================================================
from pathlib import Path
import json
import os
import pickle
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
DATA_CSV = REPO_DIR / "out_LED.csv"
FIT_DIR = (
    SCRIPT_DIR
    / "numpyro_svi_proactive_led_step_jump_all_on_no_trunc_exp_lapse_"
    "aggregate_patience12_min150k_restore_best_outputs"
    / "LED7_all_animals"
)
POSTERIOR_PATH = FIT_DIR / "main_fullrank_posterior_samples.npz"
RUN_SUMMARY_PATH = FIT_DIR / "run_summary.json"
OUTPUT_DIR = FIT_DIR / "summary_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIGURE_PATH = (
    OUTPUT_DIR
    / "led7_aggregate_no_trunc_exp_lapse_abort_rtd_wrt_fixation_components.png"
)
PAYLOAD_PATH = (
    OUTPUT_DIR
    / "led7_aggregate_no_trunc_exp_lapse_abort_rtd_wrt_fixation_components.pkl"
)
METRICS_PATH = (
    OUTPUT_DIR
    / "led7_aggregate_no_trunc_exp_lapse_abort_rtd_wrt_fixation_components.csv"
)

ANIMALS = (92, 93, 98, 99, 100, 103)
EXPECTED_TOTAL = 94433
EXPECTED_COUNTS = {
    "off": {"total": 63523, "abort": 10051},
    "on": {"total": 30910, "abort": 6336},
}

DATA_BIN_S = 0.020
MODEL_DT_S = 0.001
TIME_RANGE_S = (0.0, 2.2)
MODEL_TRIAL_CHUNK = 512
QUADRATURE_NODES = 64
CDF_FRACTION_TOL = 2.0e-3

DATA_COLOR = "black"
MIXTURE_COLOR = "#6A3D9A"
PROACTIVE_COLOR = "#1F77B4"
LAPSE_COLOR = "#E66101"


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

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpyro_proactive_led_step_jump_no_trunc_exp_lapse_svi_utils as svi_utils


# %%
# =============================================================================
# Load and validate the completed aggregate posterior
# =============================================================================
for path in [DATA_CSV, POSTERIOR_PATH, RUN_SUMMARY_PATH]:
    if not path.exists():
        raise FileNotFoundError(path)

with RUN_SUMMARY_PATH.open() as handle:
    run_summary = json.load(handle)

if run_summary.get("status") != "complete":
    raise RuntimeError(f"Aggregate fit is not complete: {run_summary.get('status')}")
config = run_summary.get("config", {})
if config.get("truncation") != "none":
    raise RuntimeError("Expected the no-truncation aggregate fit.")
if config.get("pooling") != "trial_weighted":
    raise RuntimeError("Expected trial-weighted aggregate fitting.")
if config.get("led_on_scope") != "all LED_trial == 1 rows":
    raise RuntimeError("Expected all LED-ON trials, including unilateral trials.")

with np.load(POSTERIOR_PATH) as posterior:
    missing = sorted(set(svi_utils.PARAM_NAMES) - set(posterior.files))
    if missing:
        raise RuntimeError(f"Aggregate posterior is missing {missing}.")
    posterior_samples = {
        name: np.asarray(posterior[name], dtype=float)
        for name in svi_utils.PARAM_NAMES
    }

if not all(np.all(np.isfinite(values)) for values in posterior_samples.values()):
    raise RuntimeError("Aggregate posterior contains non-finite samples.")
posterior_means = {
    name: float(np.mean(values)) for name, values in posterior_samples.items()
}


# %%
# =============================================================================
# Reconstruct the exact no-truncation, all-ON plus OFF fitting population
# =============================================================================
raw_df = pd.read_csv(DATA_CSV)
source_df = raw_df[
    raw_df["repeat_trial"].isin([0, 2]) | raw_df["repeat_trial"].isna()
].copy()
source_df = source_df[
    (source_df["session_type"] == 7)
    & (source_df["training_level"] == 16)
    & source_df["animal"].astype(int).isin(ANIMALS)
].copy()
source_df = source_df.dropna(
    subset=["intended_fix", "LED_onset_time", "timed_fix"]
)
source_df = source_df[
    (source_df["abort_event"] == 3) | source_df["success"].isin([1, -1])
].copy()
source_df["t_LED"] = source_df["intended_fix"] - source_df["LED_onset_time"]

category_frames = {
    "off": source_df[
        (source_df["LED_trial"] == 0) | source_df["LED_trial"].isna()
    ].copy(),
    "on": source_df[source_df["LED_trial"] == 1].copy(),
}

if len(source_df) != EXPECTED_TOTAL:
    raise RuntimeError(f"Expected {EXPECTED_TOTAL} rows, found {len(source_df)}.")
if int(run_summary["trial_counts"]["total"]) != len(source_df):
    raise RuntimeError("Fit summary and reconstructed fitting rows disagree.")

for category, frame in category_frames.items():
    frame["is_abort"] = frame["timed_fix"] < frame["intended_fix"]
    n_total = len(frame)
    n_abort = int(frame["is_abort"].sum())
    expected = EXPECTED_COUNTS[category]
    if n_total != expected["total"] or n_abort != expected["abort"]:
        raise RuntimeError(
            f"{category}: expected {expected}, found "
            f"{{'total': {n_total}, 'abort': {n_abort}}}."
        )


# %%
# =============================================================================
# Exact timing-averaged component densities and CDF masses
# =============================================================================
model_t = np.arange(
    TIME_RANGE_S[0] + 0.5 * MODEL_DT_S,
    TIME_RANGE_S[1],
    MODEL_DT_S,
)
model_t_jax = jnp.asarray(model_t, dtype=jnp.float64)


def component_densities(category, frame):
    """Return censoring-conditioned mixture and weighted component densities."""
    t_stim = frame["intended_fix"].to_numpy(dtype=float)
    t_led = frame["t_LED"].to_numpy(dtype=float)
    n_trials = len(frame)

    proactive_sum = np.zeros_like(model_t)
    lapse_sum = np.zeros_like(model_t)
    proactive_cdf_sum = 0.0
    lapse_cdf_sum = 0.0

    for start in range(0, n_trials, MODEL_TRIAL_CHUNK):
        stop = min(start + MODEL_TRIAL_CHUNK, n_trials)
        chunk_t_stim = jnp.asarray(t_stim[start:stop], dtype=jnp.float64)
        chunk_t_led = jnp.asarray(t_led[start:stop], dtype=jnp.float64)
        t = model_t_jax[None, :]
        risk_mask = t < chunk_t_stim[:, None]

        if category == "on":
            proactive_pdf = svi_utils.led_on_pdf_jax(
                t,
                chunk_t_led[:, None],
                posterior_means["V_A_base"],
                posterior_means["V_A_post_LED"],
                posterior_means["theta_A"],
                posterior_means["del_a_minus_del_LED"],
                posterior_means["del_m_plus_del_LED"],
            )
            proactive_cdf = svi_utils.led_on_cdf_jax(
                chunk_t_stim,
                chunk_t_led,
                posterior_means["V_A_base"],
                posterior_means["V_A_post_LED"],
                posterior_means["theta_A"],
                posterior_means["del_a_minus_del_LED"],
                posterior_means["del_m_plus_del_LED"],
                n_quad=QUADRATURE_NODES,
            )
        else:
            proactive_pdf = svi_utils.led_off_pdf_jax(
                t,
                posterior_means["V_A_base"],
                posterior_means["theta_A"],
                posterior_means["del_a_minus_del_LED"],
                posterior_means["del_m_plus_del_LED"],
            )
            proactive_cdf = svi_utils.led_off_cdf_jax(
                chunk_t_stim,
                posterior_means["V_A_base"],
                posterior_means["theta_A"],
                posterior_means["del_a_minus_del_LED"],
                posterior_means["del_m_plus_del_LED"],
            )

        lapse_pdf = posterior_means["beta_lapse"] * jnp.exp(
            -posterior_means["beta_lapse"] * t
        )
        proactive_sum += np.asarray(
            jax.device_get(jnp.sum(jnp.where(risk_mask, proactive_pdf, 0.0), axis=0))
        )
        lapse_sum += np.asarray(
            jax.device_get(jnp.sum(jnp.where(risk_mask, lapse_pdf, 0.0), axis=0))
        )
        proactive_cdf_sum += float(
            np.sum(np.asarray(jax.device_get(proactive_cdf), dtype=float))
        )
        lapse_cdf_sum += float(
            np.sum(1.0 - np.exp(-posterior_means["beta_lapse"] * t_stim[start:stop]))
        )

    proactive_raw = proactive_sum / n_trials
    lapse_raw = lapse_sum / n_trials
    lapse_prob = posterior_means["lapse_prob"]
    proactive_weighted = (1.0 - lapse_prob) * proactive_raw
    lapse_weighted = lapse_prob * lapse_raw
    mixture_raw = proactive_weighted + lapse_weighted

    numerical_total_mass = float(np.sum(mixture_raw) * MODEL_DT_S)
    if not np.isfinite(numerical_total_mass) or numerical_total_mass <= 0.0:
        raise RuntimeError(f"{category}: invalid numerical model mass.")

    proactive_density = proactive_weighted / numerical_total_mass
    lapse_density = lapse_weighted / numerical_total_mass
    mixture_density = proactive_density + lapse_density

    proactive_area = float(np.sum(proactive_density) * MODEL_DT_S)
    lapse_area = float(np.sum(lapse_density) * MODEL_DT_S)
    mixture_area = float(np.sum(mixture_density) * MODEL_DT_S)

    analytic_proactive_mass = (1.0 - lapse_prob) * proactive_cdf_sum / n_trials
    analytic_lapse_mass = lapse_prob * lapse_cdf_sum / n_trials
    analytic_total_mass = analytic_proactive_mass + analytic_lapse_mass
    analytic_lapse_fraction = analytic_lapse_mass / analytic_total_mass

    arrays = [proactive_density, lapse_density, mixture_density]
    if not all(np.all(np.isfinite(values)) for values in arrays):
        raise RuntimeError(f"{category}: model density contains non-finite values.")
    if any(np.any(values < -1e-12) for values in arrays):
        raise RuntimeError(f"{category}: model density contains negative values.")
    if not np.allclose(
        mixture_density,
        proactive_density + lapse_density,
        atol=1e-12,
        rtol=1e-12,
    ):
        raise RuntimeError(f"{category}: components do not sum to the mixture.")
    if not np.isclose(mixture_area, 1.0, atol=1e-12, rtol=1e-12):
        raise RuntimeError(f"{category}: normalized mixture area is {mixture_area}.")
    if not np.isclose(proactive_area + lapse_area, 1.0, atol=1e-12, rtol=1e-12):
        raise RuntimeError(f"{category}: component areas do not sum to one.")
    if not np.isclose(
        lapse_area,
        analytic_lapse_fraction,
        atol=CDF_FRACTION_TOL,
        rtol=0.0,
    ):
        raise RuntimeError(
            f"{category}: numerical lapse area {lapse_area:.6f} differs from "
            f"CDF fraction {analytic_lapse_fraction:.6f}."
        )

    return {
        "model_t_s": model_t.copy(),
        "mixture_density": mixture_density,
        "proactive_density": proactive_density,
        "lapse_density": lapse_density,
        "numerical_mixture_area": mixture_area,
        "numerical_proactive_area": proactive_area,
        "numerical_lapse_area": lapse_area,
        "analytic_raw_proactive_abort_mass": analytic_proactive_mass,
        "analytic_raw_lapse_abort_mass": analytic_lapse_mass,
        "analytic_model_abort_probability": analytic_total_mass,
        "analytic_conditional_lapse_fraction": analytic_lapse_fraction,
        "analytic_conditional_proactive_fraction": 1.0 - analytic_lapse_fraction,
        "numerical_raw_model_abort_mass": numerical_total_mass,
        "t_stim_s": t_stim,
        "t_LED_s": t_led,
    }


# %%
# =============================================================================
# Empirical RTDs and category payloads
# =============================================================================
data_edges = np.arange(
    TIME_RANGE_S[0],
    TIME_RANGE_S[1] + 0.5 * DATA_BIN_S,
    DATA_BIN_S,
)
data_centers = 0.5 * (data_edges[:-1] + data_edges[1:])
category_payloads = {}
metric_rows = []

for category, frame in category_frames.items():
    abort_times = frame.loc[frame["is_abort"], "timed_fix"].to_numpy(dtype=float)
    if np.any(abort_times < TIME_RANGE_S[0]) or np.any(abort_times > TIME_RANGE_S[1]):
        raise RuntimeError(f"{category}: plotting range omits empirical aborts.")
    counts, _ = np.histogram(abort_times, bins=data_edges)
    data_density = counts / (len(abort_times) * DATA_BIN_S)
    data_area = float(np.sum(data_density) * DATA_BIN_S)
    if not np.isclose(data_area, 1.0, atol=1e-12, rtol=1e-12):
        raise RuntimeError(f"{category}: empirical area is {data_area}.")

    model_payload = component_densities(category, frame)
    category_payloads[category] = {
        "n_total": int(len(frame)),
        "n_abort": int(len(abort_times)),
        "observed_abort_fraction": float(len(abort_times) / len(frame)),
        "abort_times_s": abort_times,
        "data_edges_s": data_edges.copy(),
        "data_centers_s": data_centers.copy(),
        "data_density": data_density,
        "data_area": data_area,
        **model_payload,
    }
    metric_rows.append(
        {
            "category": category,
            "n_total": len(frame),
            "n_abort": len(abort_times),
            "observed_abort_fraction": len(abort_times) / len(frame),
            "fitted_lapse_prob": posterior_means["lapse_prob"],
            "fitted_beta_lapse_per_s": posterior_means["beta_lapse"],
            "data_area": data_area,
            "model_mixture_area": model_payload["numerical_mixture_area"],
            "model_proactive_area": model_payload["numerical_proactive_area"],
            "model_lapse_area": model_payload["numerical_lapse_area"],
            "cdf_proactive_fraction": model_payload[
                "analytic_conditional_proactive_fraction"
            ],
            "cdf_lapse_fraction": model_payload[
                "analytic_conditional_lapse_fraction"
            ],
            "model_abort_probability": model_payload[
                "analytic_model_abort_probability"
            ],
            **posterior_means,
        }
    )


# %%
# =============================================================================
# 1 x 2 fixation-time RTD decomposition
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.7), sharex=True, sharey=True)
panel_settings = {
    "off": "LED OFF",
    "on": "LED ON (all ON trials)",
}

for ax, category in zip(axes, ["off", "on"]):
    payload = category_payloads[category]
    ax.step(
        payload["data_centers_s"],
        payload["data_density"],
        where="mid",
        color=DATA_COLOR,
        alpha=0.52,
        linewidth=1.05,
        label="Data abort RTD (area=1)",
    )
    ax.plot(
        payload["model_t_s"],
        payload["mixture_density"],
        color=MIXTURE_COLOR,
        linewidth=2.1,
        label="Model mixture (area=1)",
    )
    ax.plot(
        payload["model_t_s"],
        payload["proactive_density"],
        color=PROACTIVE_COLOR,
        linestyle="--",
        linewidth=1.7,
        label=(
            "Proactive contribution "
            f"(area={payload['numerical_proactive_area']:.3f})"
        ),
    )
    ax.plot(
        payload["model_t_s"],
        payload["lapse_density"],
        color=LAPSE_COLOR,
        linestyle="-.",
        linewidth=1.7,
        label=(
            "Lapse contribution "
            f"(area={payload['numerical_lapse_area']:.3f})"
        ),
    )
    ax.set_title(panel_settings[category])
    ax.set_xlim(TIME_RANGE_S)
    ax.set_xlabel("Time from fixation onset (s)")
    ax.grid(axis="y", alpha=0.14, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.98,
        0.97,
        f"aborts: {payload['n_abort']:,}/{payload['n_total']:,}\n"
        f"fitted lapse p: {100 * posterior_means['lapse_prob']:.2f}%\n"
        "lapse share among model aborts: "
        f"{100 * payload['analytic_conditional_lapse_fraction']:.2f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.2,
    )
    ax.legend(frameon=False, fontsize=8.0, loc="upper right", bbox_to_anchor=(1, 0.77))

axes[0].set_ylabel("Density conditioned on abort (s$^{-1}$)")
fig.suptitle(
    "LED7 aggregate proactive-step-jump + exponential-lapse abort RTDs",
    y=1.01,
)
fig.tight_layout()
fig.savefig(FIGURE_PATH, dpi=220, bbox_inches="tight")


# %%
# =============================================================================
# Reusable payload and numerical audit table
# =============================================================================
payload = {
    "schema_version": 1,
    "data_source": str(DATA_CSV.resolve()),
    "fit_root": str(FIT_DIR.resolve()),
    "animals": list(ANIMALS),
    "pooling": "trial weighted",
    "led_on_scope": "all LED_trial == 1 rows",
    "truncation": "none",
    "posterior_point_estimate": "posterior means",
    "posterior_means": posterior_means,
    "data_bin_s": DATA_BIN_S,
    "model_dt_s": MODEL_DT_S,
    "time_range_s": TIME_RANGE_S,
    "quadrature_nodes": QUADRATURE_NODES,
    "normalization": (
        "weighted proactive and lapse densities share the same conditional-abort "
        "normalization; component areas sum to one"
    ),
    "categories": category_payloads,
}
with PAYLOAD_PATH.open("wb") as handle:
    pickle.dump(payload, handle)

metrics = pd.DataFrame(metric_rows)
metrics.to_csv(METRICS_PATH, index=False)

print(f"Fit: {FIT_DIR.resolve()}")
print(f"Figure: {FIGURE_PATH.resolve()}")
print(f"Payload: {PAYLOAD_PATH.resolve()}")
print(f"Metrics: {METRICS_PATH.resolve()}")
print(metrics.to_string(index=False))
