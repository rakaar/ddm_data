# %%
"""Compare the historical FCT LED-ON theory with the aggregate SVI curve."""

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

FIT_DIR = (
    SCRIPT_DIR
    / "numpyro_svi_proactive_led_step_jump_bilateral_unweighted_"
    "aggregate_patience12_min50k_restore_best_outputs"
    / "LED7_all_animals"
)
POSTERIOR_PATH = FIT_DIR / "main_fullrank_posterior_samples.npz"
CURRENT_PAYLOAD_PATH = (
    FIT_DIR
    / "summary_figures"
    / "led7_aggregate_proactive_step_jump_rtwrtled_1x2.pkl"
)
FCT_PAYLOAD_PATH = REPO_DIR / "fct_march_26" / "rt_wrt_led_theory_data_all_animals.pkl"

T_TRUNC_S = 0.3
N_TIMING_DRAWS = 5000
TIMING_SEED = 20260813
THEORY_DT_S = 0.005
THEORY_X_S = np.arange(-0.3, 0.4 + 0.5 * THEORY_DT_S, THEORY_DT_S)
CHUNK_SIZE = 512
N_QUAD = 64

OUTPUT_DIR = FIT_DIR / "summary_figures"
FIG_PATH = OUTPUT_DIR / "fct_vs_aggregate_svi_led_on_theory_comparison.png"
METRICS_PATH = OUTPUT_DIR / "fct_vs_aggregate_svi_led_on_theory_comparison.json"


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
import pandas as pd

sys.path.insert(0, str(REPO_DIR / "fit_each_condn"))
from psiam_tied_dv_map_utils_with_PDFs import d_A_RT, stupid_f_integral

sys.path.insert(0, str(SCRIPT_DIR))
import numpyro_proactive_led_step_jump_svi_utils as svi_utils


# %%
# =============================================================================
# Historical FCT scalar PDF, copied from schematics_for_fct.py
# =============================================================================
def fct_led_on_pdf(t, t_led, params):
    elapsed_pre = t_led - params["del_a_minus_del_LED"]
    elapsed_post = t - t_led - params["del_m_plus_del_LED"]
    elapsed_if_pre = t - (
        params["del_m_plus_del_LED"] + params["del_a_minus_del_LED"]
    )

    if elapsed_pre > 0.0 and elapsed_post <= 0.0:
        return d_A_RT(
            params["V_A_base"] * params["theta_A"],
            elapsed_if_pre / params["theta_A"] ** 2,
        ) / params["theta_A"] ** 2
    if elapsed_pre <= 0.0:
        return d_A_RT(
            params["V_A_post_LED"] * params["theta_A"],
            elapsed_post / params["theta_A"] ** 2,
        ) / params["theta_A"] ** 2
    return stupid_f_integral(
        params["V_A_base"],
        params["V_A_post_LED"],
        params["theta_A"],
        elapsed_post,
        elapsed_pre,
    )


def fct_raw_curve(t_led, t_stim, params):
    density = np.zeros(len(THEORY_X_S), dtype=float)
    for one_t_led, one_t_stim in zip(t_led, t_stim):
        t_fix = THEORY_X_S + one_t_led
        mask = (t_fix > 0.0) & (t_fix < one_t_stim)
        density[mask] += np.asarray(
            [fct_led_on_pdf(t, one_t_led, params) for t in t_fix[mask]],
            dtype=float,
        )
    return density / len(t_led)


def jax_curve(t_led, t_stim, params, conditioned, normalize_truncation=True):
    density_sum = np.zeros(len(THEORY_X_S), dtype=float)
    if conditioned and normalize_truncation:
        cdf_at_trunc = svi_utils.led_on_cdf_jax(
            jnp.full(len(t_led), T_TRUNC_S, dtype=jnp.float64),
            jnp.asarray(t_led, dtype=jnp.float64),
            params["V_A_base"],
            params["V_A_post_LED"],
            params["theta_A"],
            params["del_a_minus_del_LED"],
            params["del_m_plus_del_LED"],
            n_quad=N_QUAD,
        )
        denominators = np.asarray(jax.device_get(1.0 - cdf_at_trunc), dtype=float)
    else:
        denominators = np.ones(len(t_led), dtype=float)

    for start in range(0, len(t_led), CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, len(t_led))
        chunk_t_led = jnp.asarray(t_led[start:stop], dtype=jnp.float64)
        chunk_t_stim = jnp.asarray(t_stim[start:stop], dtype=jnp.float64)
        t_fix = chunk_t_led[:, None] + jnp.asarray(THEORY_X_S)[None, :]
        if conditioned:
            mask = (
                (chunk_t_stim[:, None] > T_TRUNC_S)
                & (t_fix > T_TRUNC_S)
                & (t_fix < chunk_t_stim[:, None])
            )
        else:
            mask = (t_fix > 0.0) & (t_fix < chunk_t_stim[:, None])

        raw_pdf = svi_utils.led_on_pdf_jax(
            t_fix,
            chunk_t_led[:, None],
            params["V_A_base"],
            params["V_A_post_LED"],
            params["theta_A"],
            params["del_a_minus_del_LED"],
            params["del_m_plus_del_LED"],
        )
        values = jnp.where(
            mask,
            raw_pdf / jnp.asarray(denominators[start:stop, None]),
            0.0,
        )
        density_sum += np.asarray(jax.device_get(jnp.sum(values, axis=0)), dtype=float)
    return density_sum / len(t_led)


# %%
# =============================================================================
# Load current posterior and both saved curves
# =============================================================================
for path in (POSTERIOR_PATH, CURRENT_PAYLOAD_PATH, FCT_PAYLOAD_PATH):
    if not path.exists():
        raise FileNotFoundError(path)

with np.load(POSTERIOR_PATH) as posterior:
    params = {
        name: float(np.mean(np.asarray(posterior[name], dtype=float)))
        for name in svi_utils.PARAM_NAMES
    }

with CURRENT_PAYLOAD_PATH.open("rb") as handle:
    current_payload = pickle.load(handle)
with FCT_PAYLOAD_PATH.open("rb") as handle:
    fct_payload = pickle.load(handle)

current_x_s = np.asarray(
    current_payload["conditions"]["on"]["theory_x_s"], dtype=float
)
current_curve = np.asarray(
    current_payload["conditions"]["on"]["theory_density"], dtype=float
)
if not np.array_equal(current_x_s, THEORY_X_S):
    raise RuntimeError("Current aggregate payload uses an unexpected theory grid.")

fct_x_s = np.asarray(fct_payload["theory_x_ms"], dtype=float) / 1000.0
fct_curve = np.asarray(fct_payload["rtd_theory_on_wrt_led"], dtype=float)


# %%
# =============================================================================
# Rebuild the exact current timing pool and paired Monte Carlo draw
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
    ~(
        (source_df["abort_event"] == 3)
        & (source_df["timed_fix"] < T_TRUNC_S)
    )
].copy()

on_df = source_df[
    (source_df["LED_trial"] == 1)
    & (source_df["LED_powerR"] != 0)
    & (source_df["LED_powerL"] != 0)
].copy()
off_df = source_df[
    (source_df["LED_trial"] == 0) | source_df["LED_trial"].isna()
].copy()
timing_pool = pd.concat(
    [
        on_df[["intended_fix", "LED_onset_time"]],
        off_df[["intended_fix", "LED_onset_time"]],
    ],
    ignore_index=True,
)
timing_pool["t_LED"] = (
    timing_pool["intended_fix"] - timing_pool["LED_onset_time"]
)

rng = np.random.default_rng(TIMING_SEED)
indices = rng.integers(0, len(timing_pool), size=N_TIMING_DRAWS)
t_led = timing_pool.iloc[indices]["t_LED"].to_numpy(dtype=float)
t_stim = timing_pool.iloc[indices]["intended_fix"].to_numpy(dtype=float)
all_t_led = timing_pool["t_LED"].to_numpy(dtype=float)
all_t_stim = timing_pool["intended_fix"].to_numpy(dtype=float)


# %%
# =============================================================================
# Controlled evaluator comparison
# =============================================================================
fct_raw = fct_raw_curve(t_led, t_stim, params)
jax_raw = jax_curve(t_led, t_stim, params, conditioned=False)
jax_conditioned = jax_curve(t_led, t_stim, params, conditioned=True)
jax_conditioned_all_pairs = jax_curve(
    all_t_led,
    all_t_stim,
    params,
    conditioned=True,
)
jax_support_mask_only_all_pairs = jax_curve(
    all_t_led,
    all_t_stim,
    params,
    conditioned=True,
    normalize_truncation=False,
)
positive_t_led_mask = ~np.isclose(all_t_led, 0.0, atol=1e-12, rtol=0.0)
jump_rows_mask = (~positive_t_led_mask) & (all_t_stim > T_TRUNC_S)
jax_conditioned_positive_t_led = jax_curve(
    all_t_led[positive_t_led_mask],
    all_t_stim[positive_t_led_mask],
    params,
    conditioned=True,
)
jax_conditioned_zero_t_led = jax_curve(
    all_t_led[~positive_t_led_mask],
    all_t_stim[~positive_t_led_mask],
    params,
    conditioned=True,
)

raw_abs_difference = np.abs(fct_raw - jax_raw)
conditioned_abs_difference = np.abs(jax_conditioned - current_curve)
metrics = {
    "fct_source": str(REPO_DIR / "schematics_for_fct.py"),
    "fct_payload": str(FCT_PAYLOAD_PATH),
    "current_payload": str(CURRENT_PAYLOAD_PATH),
    "n_timing_draws": N_TIMING_DRAWS,
    "timing_seed": TIMING_SEED,
    "current_posterior_means": params,
    "controlled_fct_vs_jax_raw_max_abs": float(np.max(raw_abs_difference)),
    "controlled_fct_vs_jax_raw_rmse": float(
        np.sqrt(np.mean((fct_raw - jax_raw) ** 2))
    ),
    "recomputed_conditioned_vs_saved_max_abs": float(
        np.max(conditioned_abs_difference)
    ),
    "recomputed_conditioned_vs_saved_rmse": float(
        np.sqrt(np.mean((jax_conditioned - current_curve) ** 2))
    ),
    "all_timing_pairs": int(len(all_t_led)),
    "zero_t_led_timing_pairs": int(np.sum(~positive_t_led_mask)),
    "zero_t_led_timing_pair_fraction": float(np.mean(~positive_t_led_mask)),
    "zero_t_led_with_t_stim_above_truncation": int(np.sum(jump_rows_mask)),
    "jump_row_fraction_of_all_timing_pairs": float(np.mean(jump_rows_mask)),
    "sampled_vs_all_pairs_conditioned_max_abs": float(
        np.max(np.abs(jax_conditioned - jax_conditioned_all_pairs))
    ),
}

for x_ms in (285, 290, 295, 300, 305, 310, 315):
    index = int(np.argmin(np.abs(THEORY_X_S - x_ms / 1000.0)))
    metrics[f"tail_{x_ms}ms"] = {
        "fct_evaluator_raw": float(fct_raw[index]),
        "conditioned_300ms": float(jax_conditioned[index]),
        "conditioned_all_pairs": float(jax_conditioned_all_pairs[index]),
        "conditioned_positive_t_led_only": float(
            jax_conditioned_positive_t_led[index]
        ),
        "support_mask_only_all_pairs": float(
            jax_support_mask_only_all_pairs[index]
        ),
        "conditioned_zero_t_led_only": float(
            jax_conditioned_zero_t_led[index]
        ),
    }

with METRICS_PATH.open("w", encoding="utf-8") as handle:
    json.dump(metrics, handle, indent=2)


# %%
# =============================================================================
# Plot
# =============================================================================
fig, axes = plt.subplots(1, 4, figsize=(20.0, 4.4))

axes[0].plot(1000.0 * fct_x_s, fct_curve, color="#D62728", lw=2.2, label="FCT")
axes[0].plot(
    1000.0 * current_x_s,
    current_curve,
    color="#7B3294",
    lw=1.8,
    label="Current aggregate SVI",
)
axes[0].set_title("Saved red theory curves")
axes[0].set_ylabel("Abort rate (Hz)")
axes[0].legend(frameon=False)

axes[1].plot(
    1000.0 * THEORY_X_S,
    fct_raw,
    color="#D62728",
    lw=3.0,
    alpha=0.45,
    label="FCT Python PDF",
)
axes[1].plot(
    1000.0 * THEORY_X_S,
    jax_raw,
    color="#1F77B4",
    lw=1.0,
    label="JAX PDF",
)
axes[1].set_title("Same params, pairs, mask")
axes[1].legend(frameon=False)

axes[2].plot(
    1000.0 * THEORY_X_S,
    jax_raw,
    color="#1F77B4",
    lw=1.4,
    label="No truncation",
)
axes[2].plot(
    1000.0 * THEORY_X_S,
    jax_support_mask_only_all_pairs,
    color="#E66101",
    lw=2.0,
    label="300 ms support mask only",
)
axes[2].plot(
    1000.0 * THEORY_X_S,
    jax_conditioned_all_pairs,
    color="#D62728",
    lw=1.3,
    ls="--",
    label="Mask + survival normalization",
)
axes[2].set_title("Which truncation operation adds kink?")
axes[2].legend(frameon=False)

zero_weight = float(np.mean(~positive_t_led_mask))
positive_weight = 1.0 - zero_weight
axes[3].plot(
    1000.0 * THEORY_X_S,
    jax_conditioned_all_pairs,
    color="#D62728",
    lw=2.0,
    label="All timing rows",
)
axes[3].plot(
    1000.0 * THEORY_X_S,
    positive_weight * jax_conditioned_positive_t_led,
    color="#008837",
    lw=1.4,
    label=r"Contribution from $t_{LED}>0$ rows",
)
axes[3].plot(
    1000.0 * THEORY_X_S,
    zero_weight * jax_conditioned_zero_t_led,
    color="#5E3C99",
    lw=1.6,
    label=r"Contribution from $t_{LED}=0$ rows",
)
axes[3].set_title("Conditioned curve by timing group")
axes[3].legend(frameon=False)

for ax in axes:
    ax.axvline(0.0, color="0.35", ls="--", lw=0.8)
    ax.set_xlim(-300.0, 400.0)
    ax.set_xlabel("RT wrt LED (ms)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.12, lw=0.5)

fig.suptitle("Historical FCT versus current aggregate LED-ON theory", fontsize=14)
fig.tight_layout()
fig.savefig(FIG_PATH, dpi=220, bbox_inches="tight")

print(f"Saved: {FIG_PATH}")
print(f"Saved: {METRICS_PATH}")
print(
    "Controlled FCT-Python vs JAX raw max abs difference: "
    f"{metrics['controlled_fct_vs_jax_raw_max_abs']:.3e}"
)
print(
    "Recomputed conditioned vs saved current max abs difference: "
    f"{metrics['recomputed_conditioned_vs_saved_max_abs']:.3e}"
)
