# %%
"""Plot LED7 data and proactive step-jump SVI theory relative to LED onset."""

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

STANDARD_OUTPUT_ROOT = (
    SCRIPT_DIR
    / "numpyro_svi_proactive_led_step_jump_bilateral_unweighted_"
    "patience12_min50k_restore_best_outputs"
)
EXTENDED_OUTPUT_ROOT = (
    SCRIPT_DIR
    / "numpyro_svi_proactive_led_step_jump_bilateral_unweighted_"
    "min150k_restore_best_audit_outputs"
)
EXTENDED_ANIMALS = (92, 100, 103)
ANIMALS = (92, 93, 98, 99, 100, 103)

T_TRUNC_S = 0.3
QUADRATURE_NODES = 64
THEORY_DT_S = 0.005
THEORY_XLIM_S = (-0.3, 0.4)
THEORY_X_S = np.arange(
    THEORY_XLIM_S[0],
    THEORY_XLIM_S[1] + 0.5 * THEORY_DT_S,
    THEORY_DT_S,
)

HIST_RANGE_S = (-3.0, 3.0)
N_MC_TIMING_PAIRS = 5000
MC_SEED = 20260813
INDIVIDUAL_REGULAR_BIN_S = 0.020
INDIVIDUAL_ZOOM_BIN_S = 0.010
POOLED_REGULAR_BIN_S = 0.010
POOLED_ZOOM_BIN_S = 0.005
REGULAR_XLIM_MS = (-300.0, 400.0)
ZOOM_XLIM_MS = (-200.0, 200.0)
MODEL_TRIAL_CHUNK_SIZE = 512

SUMMARY_DIR = STANDARD_OUTPUT_ROOT / "summary_figures"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
FIG_PATH = SUMMARY_DIR / "led7_proactive_led_step_jump_rtwrtled_data_model_2x7.png"
PAYLOAD_PATH = SUMMARY_DIR / "led7_proactive_led_step_jump_rtwrtled_data_model_2x7.pkl"

LED_ON_COLOR = "#D62728"
LED_OFF_COLOR = "#1F77B4"
DATA_ALPHA = 0.42


# %%
# =============================================================================
# Imports configured for deterministic, noninteractive evaluation
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
# Reusable density helpers
# =============================================================================
def histogram_rate(values, n_total, bin_width_s):
    """Histogram density whose area is the retained abort fraction."""
    edges = np.arange(
        HIST_RANGE_S[0],
        HIST_RANGE_S[1] + 0.5 * bin_width_s,
        bin_width_s,
    )
    counts, _ = np.histogram(values, bins=edges)
    density = counts.astype(float) / (float(n_total) * bin_width_s)
    centers = 0.5 * (edges[:-1] + edges[1:])
    expected_area = len(values) / float(n_total)
    observed_area = float(np.sum(density * np.diff(edges)))
    if not np.isclose(observed_area, expected_area, atol=1e-12, rtol=1e-12):
        raise RuntimeError(
            "Histogram range omitted retained aborts: "
            f"area={observed_area}, expected={expected_area}."
        )
    return centers, density, edges


def conditioned_theory_density(t_led, t_stim, params, led_on):
    """Average trial-aligned density conditioned on proactive survival to 300 ms."""
    t_led = np.asarray(t_led, dtype=float)
    t_stim = np.asarray(t_stim, dtype=float)
    if len(t_led) == 0 or len(t_led) != len(t_stim):
        raise RuntimeError("Theory requires a nonempty, paired t_LED/t_stim trial set.")

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
        trunc_denominators = np.asarray(jax.device_get(1.0 - cdf_trunc), dtype=float)
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
        raise RuntimeError("Invalid 300 ms survival denominator in model density.")

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
        raise RuntimeError("Model density contains non-finite or negative values.")
    return np.maximum(density, 0.0)


# %%
# =============================================================================
# Load the exact fitting rows once
# =============================================================================
if not DATA_CSV.exists():
    raise FileNotFoundError(DATA_CSV)

raw_df = pd.read_csv(DATA_CSV)
source_df = raw_df[
    raw_df["repeat_trial"].isin([0, 2]) | raw_df["repeat_trial"].isna()
].copy()
source_df = source_df[
    (source_df["session_type"] == 7)
    & (source_df["training_level"] == 16)
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


# %%
# =============================================================================
# Build each animal's data and posterior-mean theory
# =============================================================================
per_animal = {}
timing_pools = {}

for animal in ANIMALS:
    fit_root = (
        EXTENDED_OUTPUT_ROOT if animal in EXTENDED_ANIMALS else STANDARD_OUTPUT_ROOT
    )
    fit_dir = fit_root / f"LED7_{animal}"
    run_summary_path = fit_dir / "run_summary.json"
    posterior_path = fit_dir / "main_fullrank_posterior_samples.npz"
    if not run_summary_path.exists() or not posterior_path.exists():
        raise FileNotFoundError(f"Missing selected fit artifacts for LED7/{animal}: {fit_dir}")

    with run_summary_path.open("r", encoding="utf-8") as handle:
        run_summary = json.load(handle)
    if run_summary.get("status") != "complete":
        raise RuntimeError(f"LED7/{animal} fit is not complete.")
    if not bool(run_summary.get("all_posterior_samples_finite", False)):
        raise RuntimeError(f"LED7/{animal} has non-finite posterior samples.")
    if int(run_summary.get("n_nonfinite_losses", -1)) != 0:
        raise RuntimeError(f"LED7/{animal} has non-finite losses.")
    if not np.isclose(float(run_summary["config"]["T_trunc"]), T_TRUNC_S):
        raise RuntimeError(f"LED7/{animal} was not fit with T_trunc={T_TRUNC_S}.")

    with np.load(posterior_path) as posterior:
        missing = sorted(set(svi_utils.PARAM_NAMES) - set(posterior.files))
        if missing:
            raise RuntimeError(f"LED7/{animal} posterior is missing {missing}.")
        params = {
            name: float(np.mean(np.asarray(posterior[name], dtype=float)))
            for name in svi_utils.PARAM_NAMES
        }
        for name in svi_utils.PARAM_NAMES:
            if not np.isfinite(np.asarray(posterior[name], dtype=float)).all():
                raise RuntimeError(f"LED7/{animal} posterior {name} is non-finite.")

    animal_df = source_df[source_df["animal"].astype(int) == animal].copy()
    on_df = animal_df[
        (animal_df["LED_trial"] == 1)
        & (animal_df["LED_powerR"] != 0)
        & (animal_df["LED_powerL"] != 0)
    ].copy()
    off_df = animal_df[
        (animal_df["LED_trial"] == 0) | animal_df["LED_trial"].isna()
    ].copy()

    condition_frames = {}
    for condition, condition_df in [("on", on_df), ("off", off_df)]:
        trial_df = pd.DataFrame(
            {
                "RT": condition_df["timed_fix"].to_numpy(dtype=float),
                "t_stim": condition_df["intended_fix"].to_numpy(dtype=float),
                "t_LED": (
                    condition_df["intended_fix"] - condition_df["LED_onset_time"]
                ).to_numpy(dtype=float),
            }
        )
        trial_df = trial_df[
            ~(
                (trial_df["RT"] < trial_df["t_stim"])
                & (trial_df["RT"] <= T_TRUNC_S)
            )
        ].copy()
        if trial_df.empty:
            raise RuntimeError(f"No retained LED {condition.upper()} rows for LED7/{animal}.")
        condition_frames[condition] = trial_df

    animal_payload = {
        "animal": animal,
        "fit_root": str(fit_root.resolve()),
        "fit_dir": str(fit_dir.resolve()),
        "completed_steps": int(run_summary["completed_steps"]),
        "restored_best_step": int(run_summary["restored_best_step"]),
        "posterior_means": params,
        "conditions": {},
    }

    for condition in ("on", "off"):
        trial_df = condition_frames[condition]
        abort_df = trial_df[
            (trial_df["RT"] < trial_df["t_stim"])
            & (trial_df["RT"] > T_TRUNC_S)
        ]
        rt_wrt_led = (abort_df["RT"] - abort_df["t_LED"]).to_numpy(dtype=float)
        n_total = len(trial_df)
        abort_fraction = len(abort_df) / float(n_total)

        regular_x, regular_density, regular_edges = histogram_rate(
            rt_wrt_led,
            n_total,
            INDIVIDUAL_REGULAR_BIN_S,
        )
        zoom_x, zoom_density, zoom_edges = histogram_rate(
            rt_wrt_led,
            n_total,
            INDIVIDUAL_ZOOM_BIN_S,
        )
        animal_payload["conditions"][condition] = {
            "n_total": int(n_total),
            "n_abort_post_trunc": int(len(abort_df)),
            "n_censored_contributing": int(
                np.sum(
                    (trial_df["RT"] >= trial_df["t_stim"])
                    & (trial_df["t_stim"] > T_TRUNC_S)
                )
            ),
            "n_censored_t_stim_le_trunc": int(
                np.sum(
                    (trial_df["RT"] >= trial_df["t_stim"])
                    & (trial_df["t_stim"] <= T_TRUNC_S)
                )
            ),
            "abort_fraction": float(abort_fraction),
            "rt_wrt_led_s": rt_wrt_led,
            "regular_x_s": regular_x,
            "regular_edges_s": regular_edges,
            "regular_data_density": regular_density,
            "zoom_x_s": zoom_x,
            "zoom_edges_s": zoom_edges,
            "zoom_data_density": zoom_density,
            "theory_x_s": THEORY_X_S.copy(),
        }

    timing_pool = pd.concat(
        [
            condition_frames["on"][["t_LED", "t_stim"]],
            condition_frames["off"][["t_LED", "t_stim"]],
        ],
        ignore_index=True,
    )
    timing_pools[animal] = timing_pool
    rng = np.random.default_rng(MC_SEED + animal)
    sampled_indices = rng.integers(0, len(timing_pool), size=N_MC_TIMING_PAIRS)
    sampled_timing = timing_pool.iloc[sampled_indices]
    for condition, led_on in (("on", True), ("off", False)):
        animal_payload["conditions"][condition]["theory_density"] = (
            conditioned_theory_density(
                sampled_timing["t_LED"].to_numpy(dtype=float),
                sampled_timing["t_stim"].to_numpy(dtype=float),
                params,
                led_on=led_on,
            )
        )
    animal_payload["theory_timing_pool_n"] = int(len(timing_pool))
    animal_payload["theory_mc_paired_draws"] = N_MC_TIMING_PAIRS
    animal_payload["theory_sampled_t_LED_s"] = sampled_timing["t_LED"].to_numpy(
        dtype=float
    )
    animal_payload["theory_sampled_t_stim_s"] = sampled_timing["t_stim"].to_numpy(
        dtype=float
    )

    per_animal[animal] = animal_payload
    print(
        f"LED7/{animal}: "
        f"ON {animal_payload['conditions']['on']['n_abort_post_trunc']}/"
        f"{animal_payload['conditions']['on']['n_total']}, "
        f"OFF {animal_payload['conditions']['off']['n_abort_post_trunc']}/"
        f"{animal_payload['conditions']['off']['n_total']}; "
        f"best={animal_payload['restored_best_step']}"
    )


# %%
# =============================================================================
# Pooled-all-animals data and trial-weighted model column
# =============================================================================
pooled_timing_pool = pd.concat(
    [
        timing_pools[animal].assign(animal=animal)
        for animal in ANIMALS
    ],
    ignore_index=True,
)
pooled_rng = np.random.default_rng(MC_SEED)
pooled_sample_indices = pooled_rng.integers(
    0,
    len(pooled_timing_pool),
    size=N_MC_TIMING_PAIRS,
)
pooled_sampled_timing = pooled_timing_pool.iloc[pooled_sample_indices]

pooled_payload = {
    "label": "All animals pooled",
    "posterior_mean_del_m_plus_del_LED": float(
        np.average(
            [
                per_animal[animal]["posterior_means"]["del_m_plus_del_LED"]
                for animal in ANIMALS
            ],
            weights=[len(timing_pools[animal]) for animal in ANIMALS],
        )
    ),
    "theory_timing_pool_n": int(len(pooled_timing_pool)),
    "theory_mc_paired_draws": N_MC_TIMING_PAIRS,
    "theory_sampled_animal": pooled_sampled_timing["animal"].to_numpy(dtype=int),
    "theory_sampled_t_LED_s": pooled_sampled_timing["t_LED"].to_numpy(dtype=float),
    "theory_sampled_t_stim_s": pooled_sampled_timing["t_stim"].to_numpy(dtype=float),
    "conditions": {},
}

for condition, led_on in (("on", True), ("off", False)):
    reference = per_animal[ANIMALS[0]]["conditions"][condition]
    pooled_rt_wrt_led = np.concatenate(
        [
            per_animal[animal]["conditions"][condition]["rt_wrt_led_s"]
            for animal in ANIMALS
        ]
    )
    pooled_n_total = int(
        np.sum(
            [
                per_animal[animal]["conditions"][condition]["n_total"]
                for animal in ANIMALS
            ]
        )
    )
    regular_x, regular_density, regular_edges = histogram_rate(
        pooled_rt_wrt_led,
        pooled_n_total,
        POOLED_REGULAR_BIN_S,
    )
    zoom_x, zoom_density, zoom_edges = histogram_rate(
        pooled_rt_wrt_led,
        pooled_n_total,
        POOLED_ZOOM_BIN_S,
    )

    pooled_theory = np.zeros_like(THEORY_X_S, dtype=float)
    n_grouped_draws = 0
    for animal, sampled_group in pooled_sampled_timing.groupby("animal", sort=True):
        animal = int(animal)
        group_n = len(sampled_group)
        n_grouped_draws += group_n
        pooled_theory += (group_n / float(N_MC_TIMING_PAIRS)) * conditioned_theory_density(
            sampled_group["t_LED"].to_numpy(dtype=float),
            sampled_group["t_stim"].to_numpy(dtype=float),
            per_animal[animal]["posterior_means"],
            led_on=led_on,
        )
    if n_grouped_draws != N_MC_TIMING_PAIRS:
        raise RuntimeError("Pooled Monte Carlo timing groups lost paired draws.")

    pooled_payload["conditions"][condition] = {
        "n_total": pooled_n_total,
        "n_abort_post_trunc": int(len(pooled_rt_wrt_led)),
        "abort_fraction": float(len(pooled_rt_wrt_led) / pooled_n_total),
        "rt_wrt_led_s": pooled_rt_wrt_led,
        "regular_x_s": regular_x,
        "regular_edges_s": regular_edges,
        "regular_data_density": regular_density,
        "zoom_x_s": zoom_x,
        "zoom_edges_s": zoom_edges,
        "zoom_data_density": zoom_density,
        "theory_x_s": reference["theory_x_s"].copy(),
        "theory_density": pooled_theory,
    }


# %%
# =============================================================================
# Plot the regular and zoomed FCT-style rows
# =============================================================================
columns = [per_animal[animal] for animal in ANIMALS] + [pooled_payload]
column_titles = [f"LED7/{animal}" for animal in ANIMALS] + ["All animals\npooled"]

fig, axes = plt.subplots(2, 7, figsize=(24.0, 7.0), sharey="row")

for column_index, (payload, title) in enumerate(zip(columns, column_titles)):
    delay_ms = 1000.0 * (
        payload["posterior_means"]["del_m_plus_del_LED"]
        if "posterior_means" in payload
        else payload["posterior_mean_del_m_plus_del_LED"]
    )

    for row_index, view in enumerate(("regular", "zoom")):
        ax = axes[row_index, column_index]
        for condition, color in (("on", LED_ON_COLOR), ("off", LED_OFF_COLOR)):
            condition_payload = payload["conditions"][condition]
            ax.step(
                1000.0 * condition_payload[f"{view}_x_s"],
                condition_payload[f"{view}_data_density"],
                where="mid",
                color=color,
                alpha=DATA_ALPHA,
                lw=1.0,
                zorder=2,
            )
            ax.plot(
                1000.0 * condition_payload["theory_x_s"],
                condition_payload["theory_density"],
                color=color,
                lw=1.8,
                zorder=3,
            )

        ax.axvline(0.0, color="0.25", ls="--", lw=0.9, alpha=0.7, zorder=1)
        ax.axvline(delay_ms, color="0.45", ls=":", lw=0.9, alpha=0.65, zorder=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.12, lw=0.5)
        ax.tick_params(axis="both", labelsize=8)
        if row_index == 0:
            ax.set_xlim(REGULAR_XLIM_MS)
            ax.set_xticks([-300, 0, 400])
            ax.set_title(title, fontsize=10)
        else:
            ax.set_xlim(ZOOM_XLIM_MS)
            ax.set_xticks([-200, 0, 200])
            ax.set_xlabel("RT wrt LED (ms)", fontsize=9)

axes[0, 0].set_ylabel("Abort rate (Hz)", fontsize=10)
axes[1, 0].set_ylabel("Abort rate (Hz)", fontsize=10)

for row_index, xlim in enumerate((REGULAR_XLIM_MS, ZOOM_XLIM_MS)):
    row_max = 0.0
    for column_index, payload in enumerate(columns):
        view = "regular" if row_index == 0 else "zoom"
        for condition in ("on", "off"):
            condition_payload = payload["conditions"][condition]
            data_x_ms = 1000.0 * condition_payload[f"{view}_x_s"]
            theory_x_ms = 1000.0 * condition_payload["theory_x_s"]
            data_mask = (data_x_ms >= xlim[0]) & (data_x_ms <= xlim[1])
            theory_mask = (theory_x_ms >= xlim[0]) & (theory_x_ms <= xlim[1])
            row_max = max(
                row_max,
                float(np.max(condition_payload[f"{view}_data_density"][data_mask])),
                float(np.max(condition_payload["theory_density"][theory_mask])),
            )
    axes[row_index, 0].set_ylim(0.0, 1.08 * row_max)

legend_handles = [
    Line2D([0], [0], color=LED_ON_COLOR, lw=1.0, alpha=DATA_ALPHA, label="Data LED ON"),
    Line2D([0], [0], color=LED_ON_COLOR, lw=1.8, label="Model LED ON"),
    Line2D([0], [0], color=LED_OFF_COLOR, lw=1.0, alpha=DATA_ALPHA, label="Data LED OFF"),
    Line2D([0], [0], color=LED_OFF_COLOR, lw=1.8, label="Model LED OFF"),
    Line2D([0], [0], color="0.25", lw=0.9, ls="--", label="LED onset"),
    Line2D([0], [0], color="0.45", lw=0.9, ls=":", label=r"$\delta_m + \delta_{LED}$"),
]
fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.955),
    ncol=6,
    frameon=False,
    fontsize=9,
)
fig.suptitle(
    "LED7 proactive LED step-jump SVI: RT relative to LED onset",
    fontsize=13,
    y=0.995,
)
fig.subplots_adjust(
    left=0.055,
    right=0.995,
    bottom=0.10,
    top=0.86,
    wspace=0.20,
    hspace=0.32,
)
fig.savefig(FIG_PATH, dpi=250, bbox_inches="tight")


# %%
# =============================================================================
# Save a reusable payload with full provenance
# =============================================================================
result_payload = {
    "config": {
        "data_csv": str(DATA_CSV.resolve()),
        "animals": ANIMALS,
        "extended_animals": EXTENDED_ANIMALS,
        "standard_output_root": str(STANDARD_OUTPUT_ROOT.resolve()),
        "extended_output_root": str(EXTENDED_OUTPUT_ROOT.resolve()),
        "T_trunc_s": T_TRUNC_S,
        "quadrature_nodes": QUADRATURE_NODES,
        "theory_dt_s": THEORY_DT_S,
        "n_mc_timing_pairs": N_MC_TIMING_PAIRS,
        "mc_seed": MC_SEED,
        "individual_regular_bin_s": INDIVIDUAL_REGULAR_BIN_S,
        "individual_zoom_bin_s": INDIVIDUAL_ZOOM_BIN_S,
        "pooled_regular_bin_s": POOLED_REGULAR_BIN_S,
        "pooled_zoom_bin_s": POOLED_ZOOM_BIN_S,
        "regular_xlim_ms": REGULAR_XLIM_MS,
        "zoom_xlim_ms": ZOOM_XLIM_MS,
        "seventh_column": "pooled retained rows across all six animals",
        "pooled_model": (
            "5000 paired timing rows sampled from the pooled trial table; "
            "each row is evaluated with its source animal posterior mean"
        ),
        "timing_sampling": (
            "paired t_LED/t_stim rows sampled with replacement from the combined "
            "LED-ON and LED-OFF retained trial table"
        ),
        "density_scaling": "area equals retained post-truncation abort fraction",
        "led_off_alignment": "scheduled t_LED = intended_fix - LED_onset_time",
    },
    "per_animal": per_animal,
    "pooled_all_animals": pooled_payload,
    "figure_path": str(FIG_PATH.resolve()),
}
with PAYLOAD_PATH.open("wb") as handle:
    pickle.dump(result_payload, handle)

print(f"Saved figure: {FIG_PATH}")
print(f"Saved payload: {PAYLOAD_PATH}")

# %%
