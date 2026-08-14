# %%
"""Plot pooled LED7 RT-relative-to-LED data against one aggregate SVI fit."""

# %%
# =============================================================================
# Parameters
# =============================================================================
from pathlib import Path
import json
import os
import pickle
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
DATA_CSV = REPO_DIR / "out_LED.csv"
ANIMALS = (92, 93, 98, 99, 100, 103)

FIT_ROOT = (
    SCRIPT_DIR
    / "numpyro_svi_proactive_led_step_jump_bilateral_unweighted_"
    "aggregate_patience12_min50k_restore_best_outputs"
)
FIT_DIR = FIT_ROOT / "LED7_all_animals"
RUN_SUMMARY_PATH = FIT_DIR / "run_summary.json"
POSTERIOR_PATH = FIT_DIR / "main_fullrank_posterior_samples.npz"

T_TRUNC_S = 0.3
QUADRATURE_NODES = 64
THEORY_DT_S = 0.005
THEORY_X_S = np.arange(-0.3, 0.4 + 0.5 * THEORY_DT_S, THEORY_DT_S)
N_MC_TIMING_PAIRS = 5000
MC_SEED = 20260813
MODEL_TRIAL_CHUNK_SIZE = 512

HIST_RANGE_S = (-3.0, 3.0)
REGULAR_BIN_S = 0.010
ZOOM_BIN_S = 0.005
REGULAR_XLIM_MS = (-300.0, 400.0)
ZOOM_XLIM_MS = (-200.0, 200.0)

SUMMARY_DIR = FIT_DIR / "summary_figures"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
FIG_PATH = SUMMARY_DIR / "led7_aggregate_proactive_step_jump_rtwrtled_1x2.png"
PAYLOAD_PATH = SUMMARY_DIR / "led7_aggregate_proactive_step_jump_rtwrtled_1x2.pkl"

LED_ON_COLOR = "#D62728"
LED_OFF_COLOR = "#1F77B4"
DATA_ALPHA = 0.45


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
from matplotlib.lines import Line2D
import pandas as pd

sys.path.insert(0, str(SCRIPT_DIR))
import numpyro_proactive_led_step_jump_svi_utils as svi_utils


# %%
# =============================================================================
# Density helpers
# =============================================================================
def histogram_rate(values, n_total, bin_width_s):
    edges = np.arange(
        HIST_RANGE_S[0],
        HIST_RANGE_S[1] + 0.5 * bin_width_s,
        bin_width_s,
    )
    counts, _ = np.histogram(np.asarray(values, dtype=float), bins=edges)
    density = counts.astype(float) / (float(n_total) * bin_width_s)
    centers = 0.5 * (edges[:-1] + edges[1:])
    expected_area = len(values) / float(n_total)
    observed_area = float(np.sum(density * np.diff(edges)))
    if not np.isclose(observed_area, expected_area, atol=1e-12, rtol=1e-12):
        raise RuntimeError(
            f"Histogram area={observed_area}, expected abort fraction={expected_area}."
        )
    return centers, density, edges


def conditioned_theory_density(t_led, t_stim, params, led_on):
    t_led = np.asarray(t_led, dtype=float)
    t_stim = np.asarray(t_stim, dtype=float)
    if len(t_led) != N_MC_TIMING_PAIRS or len(t_stim) != N_MC_TIMING_PAIRS:
        raise RuntimeError(
            f"Expected {N_MC_TIMING_PAIRS} paired timing draws, "
            f"found {len(t_led)} and {len(t_stim)}."
        )

    x = jnp.asarray(THEORY_X_S, dtype=jnp.float64)
    density_sum = np.zeros(len(THEORY_X_S), dtype=float)

    if led_on:
        cdf_trunc = svi_utils.led_on_cdf_jax(
            jnp.full(len(t_led), T_TRUNC_S, dtype=jnp.float64),
            jnp.asarray(t_led, dtype=jnp.float64),
            params["V_A_base"],
            params["V_A_post_LED"],
            params["theta_A"],
            params["del_a_minus_del_LED"],
            params["del_m_plus_del_LED"],
            n_quad=QUADRATURE_NODES,
        )
        trunc_denominators = np.asarray(
            jax.device_get(1.0 - cdf_trunc),
            dtype=float,
        )
    else:
        cdf_trunc = svi_utils.led_off_cdf_jax(
            T_TRUNC_S,
            params["V_A_base"],
            params["theta_A"],
            params["del_a_minus_del_LED"],
            params["del_m_plus_del_LED"],
        )
        trunc_denominators = np.full(
            len(t_led),
            float(np.asarray(jax.device_get(1.0 - cdf_trunc))),
        )

    if (
        not np.isfinite(trunc_denominators).all()
        or np.any(trunc_denominators <= 0.0)
    ):
        raise RuntimeError("Invalid 300 ms survival denominator in aggregate theory.")

    for start in range(0, len(t_led), MODEL_TRIAL_CHUNK_SIZE):
        stop = min(start + MODEL_TRIAL_CHUNK_SIZE, len(t_led))
        chunk_t_led = jnp.asarray(t_led[start:stop], dtype=jnp.float64)
        chunk_t_stim = jnp.asarray(t_stim[start:stop], dtype=jnp.float64)
        t_fix = chunk_t_led[:, None] + x[None, :]
        risk_mask = (
            (chunk_t_stim[:, None] > T_TRUNC_S)
            & (t_fix > T_TRUNC_S)
            & (t_fix < chunk_t_stim[:, None])
        )

        if led_on:
            raw_pdf = svi_utils.led_on_pdf_jax(
                t_fix,
                chunk_t_led[:, None],
                params["V_A_base"],
                params["V_A_post_LED"],
                params["theta_A"],
                params["del_a_minus_del_LED"],
                params["del_m_plus_del_LED"],
            )
        else:
            raw_pdf = svi_utils.led_off_pdf_jax(
                t_fix,
                params["V_A_base"],
                params["theta_A"],
                params["del_a_minus_del_LED"],
                params["del_m_plus_del_LED"],
            )

        conditioned = jnp.where(
            risk_mask,
            raw_pdf
            / jnp.asarray(
                trunc_denominators[start:stop, None],
                dtype=jnp.float64,
            ),
            0.0,
        )
        density_sum += np.asarray(
            jax.device_get(jnp.sum(conditioned, axis=0)),
            dtype=float,
        )

    density = density_sum / float(len(t_led))
    if not np.isfinite(density).all() or np.any(density < -1e-12):
        raise RuntimeError("Aggregate theory contains non-finite or negative values.")
    return np.maximum(density, 0.0), trunc_denominators


# %%
# =============================================================================
# Load and validate the aggregate posterior
# =============================================================================
if not RUN_SUMMARY_PATH.exists() or not POSTERIOR_PATH.exists():
    raise FileNotFoundError(
        f"Aggregate fit is incomplete or missing: {FIT_DIR}"
    )

with RUN_SUMMARY_PATH.open("r", encoding="utf-8") as handle:
    run_summary = json.load(handle)
if run_summary.get("status") != "complete":
    raise RuntimeError("Aggregate fit run_summary status is not complete.")
if tuple(run_summary["animals"]) != ANIMALS:
    raise RuntimeError("Aggregate fit contains the wrong animal set.")
if not np.isclose(float(run_summary["config"]["T_trunc"]), T_TRUNC_S):
    raise RuntimeError("Aggregate fit does not use the expected 300 ms truncation.")
if run_summary["trial_counts"]["total"] != 71537:
    raise RuntimeError("Aggregate fit does not contain the expected 71,537 rows.")
if int(run_summary.get("n_nonfinite_losses", -1)) != 0:
    raise RuntimeError("Aggregate fit has non-finite losses.")
if not bool(run_summary.get("all_posterior_samples_finite", False)):
    raise RuntimeError("Aggregate fit has non-finite posterior samples.")

with np.load(POSTERIOR_PATH) as posterior:
    missing = sorted(set(svi_utils.PARAM_NAMES) - set(posterior.files))
    if missing:
        raise RuntimeError(f"Aggregate posterior is missing parameters: {missing}")
    posterior_means = {}
    for name in svi_utils.PARAM_NAMES:
        values = np.asarray(posterior[name], dtype=float)
        if not np.isfinite(values).all():
            raise RuntimeError(f"Aggregate posterior {name} is non-finite.")
        posterior_means[name] = float(np.mean(values))

print("Aggregate posterior means:")
for name, value in posterior_means.items():
    print(f"  {name:<24} {value:.8g}")


# %%
# =============================================================================
# Rebuild the exact pooled dataset with scheduled LED times for plotting
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
source_df = source_df[
    ~((source_df["abort_event"] == 3) & (source_df["timed_fix"] < T_TRUNC_S))
].copy()

on_df = source_df[
    (source_df["LED_trial"] == 1)
    & (source_df["LED_powerR"] != 0)
    & (source_df["LED_powerL"] != 0)
].copy()
off_df = source_df[
    (source_df["LED_trial"] == 0) | source_df["LED_trial"].isna()
].copy()

condition_frames = {}
for condition, condition_df in (("on", on_df), ("off", off_df)):
    trial_df = pd.DataFrame(
        {
            "RT": condition_df["timed_fix"].to_numpy(dtype=float),
            "t_stim": condition_df["intended_fix"].to_numpy(dtype=float),
            "t_LED": (
                condition_df["intended_fix"] - condition_df["LED_onset_time"]
            ).to_numpy(dtype=float),
            "animal": condition_df["animal"].to_numpy(dtype=int),
        }
    )
    trial_df = trial_df[
        ~(
            (trial_df["RT"] < trial_df["t_stim"])
            & (trial_df["RT"] <= T_TRUNC_S)
        )
    ].copy()
    condition_frames[condition] = trial_df

if len(condition_frames["on"]) != 9914 or len(condition_frames["off"]) != 61623:
    raise RuntimeError("Diagnostic data do not match the aggregate fit trial counts.")

timing_pool = pd.concat(
    [
        condition_frames["on"][["t_LED", "t_stim", "animal"]],
        condition_frames["off"][["t_LED", "t_stim", "animal"]],
    ],
    ignore_index=True,
)
rng = np.random.default_rng(MC_SEED)
sampled_indices = rng.integers(0, len(timing_pool), size=N_MC_TIMING_PAIRS)
sampled_timing = timing_pool.iloc[sampled_indices].copy()
if not np.array_equal(
    sampled_timing["t_LED"].to_numpy(),
    timing_pool.iloc[sampled_indices]["t_LED"].to_numpy(),
) or not np.array_equal(
    sampled_timing["t_stim"].to_numpy(),
    timing_pool.iloc[sampled_indices]["t_stim"].to_numpy(),
):
    raise RuntimeError("Paired timing-row sampling was not preserved.")


# %%
# =============================================================================
# Data histograms and aggregate theory
# =============================================================================
conditions = {}
for condition, led_on in (("on", True), ("off", False)):
    trial_df = condition_frames[condition]
    abort_df = trial_df[
        (trial_df["RT"] < trial_df["t_stim"])
        & (trial_df["RT"] > T_TRUNC_S)
    ]
    rt_wrt_led = (abort_df["RT"] - abort_df["t_LED"]).to_numpy(dtype=float)
    n_total = len(trial_df)

    regular_x, regular_density, regular_edges = histogram_rate(
        rt_wrt_led,
        n_total,
        REGULAR_BIN_S,
    )
    zoom_x, zoom_density, zoom_edges = histogram_rate(
        rt_wrt_led,
        n_total,
        ZOOM_BIN_S,
    )
    theory_density, trunc_denominators = conditioned_theory_density(
        sampled_timing["t_LED"].to_numpy(dtype=float),
        sampled_timing["t_stim"].to_numpy(dtype=float),
        posterior_means,
        led_on=led_on,
    )

    conditions[condition] = {
        "n_total": int(n_total),
        "n_abort_post_trunc": int(len(abort_df)),
        "abort_fraction": float(len(abort_df) / n_total),
        "rt_wrt_led_s": rt_wrt_led,
        "regular_x_s": regular_x,
        "regular_edges_s": regular_edges,
        "regular_data_density": regular_density,
        "zoom_x_s": zoom_x,
        "zoom_edges_s": zoom_edges,
        "zoom_data_density": zoom_density,
        "theory_x_s": THEORY_X_S.copy(),
        "theory_density": theory_density,
        "trunc_survival_denominators": trunc_denominators,
    }
    print(
        f"LED {condition.upper()}: {len(abort_df)}/{n_total} retained aborts; "
        f"fraction={len(abort_df) / n_total:.5f}"
    )


# %%
# =============================================================================
# Plot regular and zoomed aggregate views
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), sharey=True)
switch_ms = 1000.0 * posterior_means["del_m_plus_del_LED"]

for ax, view, xlim, title in (
    (axes[0], "regular", REGULAR_XLIM_MS, "Regular view (10 ms data bins)"),
    (axes[1], "zoom", ZOOM_XLIM_MS, "Zoomed view (5 ms data bins)"),
):
    for condition, color in (("on", LED_ON_COLOR), ("off", LED_OFF_COLOR)):
        payload = conditions[condition]
        ax.step(
            1000.0 * payload[f"{view}_x_s"],
            payload[f"{view}_data_density"],
            where="mid",
            color=color,
            alpha=DATA_ALPHA,
            lw=1.0,
            zorder=2,
        )
        ax.plot(
            1000.0 * payload["theory_x_s"],
            payload["theory_density"],
            color=color,
            lw=2.0,
            zorder=3,
        )

    ax.axvline(0.0, color="0.25", ls="--", lw=0.9, alpha=0.75, zorder=1)
    ax.axvline(
        switch_ms,
        color=LED_ON_COLOR,
        ls=":",
        lw=1.1,
        alpha=0.75,
        zorder=1,
    )
    ax.set_xlim(xlim)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("RT wrt LED (ms)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.12, lw=0.5)

axes[0].axvline(
    1000.0 * T_TRUNC_S,
    color="0.4",
    ls="-.",
    lw=0.9,
    alpha=0.55,
    zorder=1,
)
axes[0].set_xticks([-300, 0, 300, 400])
axes[1].set_xticks([-200, 0, 200])
axes[0].set_ylabel("Abort rate (Hz)")

row_max = 0.0
for view, xlim in (("regular", REGULAR_XLIM_MS), ("zoom", ZOOM_XLIM_MS)):
    for condition in ("on", "off"):
        payload = conditions[condition]
        data_x_ms = 1000.0 * payload[f"{view}_x_s"]
        theory_x_ms = 1000.0 * payload["theory_x_s"]
        data_mask = (data_x_ms >= xlim[0]) & (data_x_ms <= xlim[1])
        theory_mask = (theory_x_ms >= xlim[0]) & (theory_x_ms <= xlim[1])
        row_max = max(
            row_max,
            float(np.max(payload[f"{view}_data_density"][data_mask])),
            float(np.max(payload["theory_density"][theory_mask])),
        )
axes[0].set_ylim(0.0, 1.08 * row_max)

legend_handles = [
    Line2D([0], [0], color=LED_ON_COLOR, lw=1.0, alpha=DATA_ALPHA, label="Data LED ON"),
    Line2D([0], [0], color=LED_ON_COLOR, lw=2.0, label="Model LED ON"),
    Line2D([0], [0], color=LED_OFF_COLOR, lw=1.0, alpha=DATA_ALPHA, label="Data LED OFF"),
    Line2D([0], [0], color=LED_OFF_COLOR, lw=2.0, label="Model LED OFF"),
    Line2D([0], [0], color="0.25", lw=0.9, ls="--", label="LED onset"),
    Line2D(
        [0],
        [0],
        color=LED_ON_COLOR,
        lw=1.1,
        ls=":",
        label=rf"Fitted switch ({switch_ms:.0f} ms)",
    ),
    Line2D(
        [0],
        [0],
        color="0.4",
        lw=0.9,
        ls="-.",
        label="300 ms truncation boundary",
    ),
]
fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.955),
    ncol=4,
    frameon=False,
    fontsize=8.5,
)
fig.suptitle(
    "LED7 pooled trials: one aggregate proactive step-jump SVI fit",
    fontsize=13,
    y=1.02,
)
fig.subplots_adjust(left=0.075, right=0.99, bottom=0.13, top=0.79, wspace=0.16)
fig.savefig(FIG_PATH, dpi=250, bbox_inches="tight")


# %%
# =============================================================================
# Save reusable plot payload
# =============================================================================
result_payload = {
    "config": {
        "animals": ANIMALS,
        "pooling": "trial_weighted",
        "fit_dir": str(FIT_DIR.resolve()),
        "data_csv": str(DATA_CSV.resolve()),
        "T_trunc_s": T_TRUNC_S,
        "quadrature_nodes": QUADRATURE_NODES,
        "theory_dt_s": THEORY_DT_S,
        "n_mc_timing_pairs": N_MC_TIMING_PAIRS,
        "mc_seed": MC_SEED,
        "regular_bin_s": REGULAR_BIN_S,
        "zoom_bin_s": ZOOM_BIN_S,
        "density_scaling": "area equals retained post-truncation abort fraction",
        "timing_sampling": "paired t_LED/t_stim rows sampled with replacement",
        "shared_parameter_vector": True,
    },
    "posterior_means": posterior_means,
    "run_summary": run_summary,
    "sampled_timing": {
        "animal": sampled_timing["animal"].to_numpy(dtype=int),
        "t_LED_s": sampled_timing["t_LED"].to_numpy(dtype=float),
        "t_stim_s": sampled_timing["t_stim"].to_numpy(dtype=float),
        "source_row_index": sampled_indices,
    },
    "conditions": conditions,
    "figure_path": str(FIG_PATH.resolve()),
}
with PAYLOAD_PATH.open("wb") as handle:
    pickle.dump(result_payload, handle)

print(f"Saved aggregate figure: {FIG_PATH}")
print(f"Saved aggregate payload: {PAYLOAD_PATH}")

# %%
