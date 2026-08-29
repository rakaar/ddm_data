# %%
"""Diagnostics for pooled LED7 valid OFF and bilateral-ON NPL+alpha fits."""

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
ANIMAL_FIT_DIR = REPO_DIR / "fit_animal_by_animal"
ANIMALS = (92, 93, 98, 99, 100, 103)

FIT_ROOT = Path(
    os.environ.get(
        "LED7_VALID_NPL_AGG_DIAGNOSTIC_FIT_ROOT",
        str(
            SCRIPT_DIR
            / "numpyro_svi_proactive_led_step_jump_npl_alpha_valid_led7_"
            "aggregate_patience12_min50k_restore_best_outputs"
            / "LED7_all_animals"
        ),
    )
).expanduser()
PROACTIVE_ROOT = (
    SCRIPT_DIR
    / "numpyro_svi_proactive_led_step_jump_all_on_no_trunc_exp_lapse_"
    "aggregate_patience12_min150k_restore_best_outputs"
    / "LED7_all_animals"
)
PROACTIVE_NPZ = PROACTIVE_ROOT / "main_fullrank_posterior_samples.npz"
DATA_CSV = REPO_DIR / "out_LED.csv"
OUTPUT_DIR = Path(
    os.environ.get(
        "LED7_VALID_NPL_AGG_DIAGNOSTIC_OUTPUT_DIR",
        str(FIT_ROOT / "diagnostics"),
    )
).expanduser()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

K_MAX = 10
QUADRATURE_NODES = 64
MODEL_DT_S = 0.001
MODEL_TIME_CHUNK = 100
MODEL_TRIAL_CHUNK = 256
DATA_BIN_S = 0.020
XLIM = (-1.0, 1.0)

EXPECTED_VALID_COUNTS = {"off": 52799, "on_bilateral": 8138}
EXPECTED_FULL_COUNTS = {"off": 63744, "on_bilateral": 10313}
CATEGORY_SETTINGS = {
    "off": {"label": "LED OFF", "model_color": "tab:blue"},
    "on_bilateral": {
        "label": "bilateral LED ON",
        "model_color": "tab:red",
    },
}


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
import numpy as np
import pandas as pd

for import_dir in [SCRIPT_DIR, ANIMAL_FIT_DIR]:
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

import numpyro_npl_alpha_svi_utils as npl_alpha
import numpyro_proactive_led_step_jump_npl_alpha_utils as combined_likelihood


# %%
# =============================================================================
# Load selection records, posterior samples, and convergence traces
# =============================================================================
if not DATA_CSV.exists():
    raise FileNotFoundError(DATA_CSV)
if not PROACTIVE_NPZ.exists():
    raise FileNotFoundError(PROACTIVE_NPZ)

with np.load(PROACTIVE_NPZ) as saved:
    fixed_proactive_params = {
        name: float(np.mean(np.asarray(saved[name], dtype=float)))
        for name in [
            "V_A_base",
            "V_A_post_LED",
            "theta_A",
            "del_a_minus_del_LED",
            "del_m_plus_del_LED",
            "lapse_prob",
            "beta_lapse",
        ]
    }

category_payloads = {}
for category in CATEGORY_SETTINGS:
    selection_path = FIT_ROOT / category / "selection_summary.json"
    if not selection_path.exists():
        raise FileNotFoundError(selection_path)
    with selection_path.open() as handle:
        selection = json.load(handle)
    if selection.get("status") != "complete":
        raise RuntimeError(f"{category} selection is not complete: {selection}")

    branch_payloads = {}
    for init_mode in ["warm", "displaced"]:
        branch_dir = FIT_ROOT / category / init_mode
        summary_path = branch_dir / "run_summary.json"
        convergence_path = branch_dir / "main_fullrank_convergence_checks.csv"
        for path in [summary_path, convergence_path]:
            if not path.exists():
                raise FileNotFoundError(path)
        with summary_path.open() as handle:
            summary = json.load(handle)
        if summary.get("status") != "complete":
            raise RuntimeError(f"{category}/{init_mode} is not complete.")
        branch_payloads[init_mode] = {
            "fit_dir": branch_dir,
            "summary": summary,
            "convergence": pd.read_csv(convergence_path),
        }

    accepted_mode = selection["accepted_initialization_mode"]
    accepted_dir = FIT_ROOT / category / accepted_mode
    sample_path = accepted_dir / "main_fullrank_posterior_samples.npz"
    condition_path = accepted_dir / "condition_table.csv"
    for path in [sample_path, condition_path]:
        if not path.exists():
            raise FileNotFoundError(path)
    with np.load(sample_path) as saved:
        posterior = {name: np.asarray(saved[name]) for name in saved.files}
    if not all(np.all(np.isfinite(values)) for values in posterior.values()):
        raise RuntimeError(f"{category} accepted posterior contains non-finite samples.")
    condition_table = pd.read_csv(condition_path).sort_values("condition_id")
    if len(condition_table) != 30:
        raise RuntimeError(f"{category}: expected 30 accepted conditions.")

    category_payloads[category] = {
        "selection": selection,
        "branches": branch_payloads,
        "accepted_mode": accepted_mode,
        "accepted_fit_dir": accepted_dir,
        "posterior": posterior,
        "posterior_means": {
            name: (
                float(np.mean(values))
                if values.ndim == 1
                else np.mean(values, axis=0)
            )
            for name, values in posterior.items()
        },
        "condition_table": condition_table,
    }

print(f"Fit root: {FIT_ROOT.resolve()}")
for category, payload in category_payloads.items():
    print(
        f"{category}: accepted {payload['accepted_mode']} at "
        f"{payload['accepted_fit_dir'].resolve()}"
    )


# %%
# =============================================================================
# 1 x 2 convergence figure with both initialization branches
# =============================================================================
convergence_png = OUTPUT_DIR / "led7_super_animal_valid_npl_alpha_convergence.png"
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), sharey=False)
branch_style = {
    "warm": {"color": "tab:blue", "linestyle": "-", "linewidth": 1.0},
    "displaced": {
        "color": "tab:orange",
        "linestyle": "--",
        "linewidth": 2.2,
    },
}

for ax, category in zip(axes, CATEGORY_SETTINGS):
    payload = category_payloads[category]
    for init_mode in ["warm", "displaced"]:
        branch = payload["branches"][init_mode]
        convergence = branch["convergence"]
        summary = branch["summary"]
        style = branch_style[init_mode]
        alpha = 1.0 if payload["accepted_mode"] == init_mode else 0.55
        ax.plot(
            convergence["end_step"],
            convergence["mean_loss"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            alpha=alpha,
            label=(
                f"{init_mode} (accepted)"
                if payload["accepted_mode"] == init_mode
                else init_mode
            ),
        )
        ax.axvline(
            summary["restored_best_step"],
            color=style["color"],
            linewidth=1.0,
            alpha=alpha,
        )
        ax.axvline(
            summary["completed_steps"],
            color=style["color"],
            linestyle=":",
            linewidth=1.0,
            alpha=alpha,
        )
    ax.set_title(
        f"{CATEGORY_SETTINGS[category]['label']}\n"
        f"accepted: {payload['accepted_mode']}"
    )
    ax.set_xlabel("SVI step")
    ax.set_ylabel("negative ELBO")
    ax.spines[["top", "right"]].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    handles.extend(
        [
            Line2D([0], [0], color="0.25", linewidth=1.0),
            Line2D([0], [0], color="0.25", linewidth=1.0, linestyle=":"),
        ]
    )
    labels.extend(["restored best", "final checked"])
    ax.legend(handles, labels, frameon=False, fontsize=7.5, loc="best")

fig.suptitle("LED7 super-animal valid RT+choice NPL+alpha fits", y=1.02)
fig.tight_layout()
fig.savefig(convergence_png, dpi=220, bbox_inches="tight")


# %%
# =============================================================================
# Rebuild pooled abort-event 3 plus successful-trial RTD datasets
# =============================================================================
raw_df = pd.read_csv(DATA_CSV)
full_df = raw_df[
    raw_df["repeat_trial"].isin([0, 2]) | raw_df["repeat_trial"].isna()
].copy()
full_df = full_df[
    (full_df["session_type"] == 7)
    & (full_df["training_level"] == 16)
    & full_df["animal"].astype(int).isin(ANIMALS)
].copy()
full_df = full_df.dropna(
    subset=["timed_fix", "intended_fix", "ABL", "ILD", "LED_onset_time"]
)
full_df = full_df[
    (full_df["abort_event"].isin([3, 4]) | full_df["success"].isin([1, -1]))
    & full_df["ABL"].isin([20, 40, 60])
].copy()
full_df["RTwrtStim"] = full_df["timed_fix"] - full_df["intended_fix"]
full_df["t_LED"] = full_df["intended_fix"] - full_df["LED_onset_time"]
full_df["abs_ILD"] = full_df["ILD"].abs().astype(float)

full_category_frames = {
    "off": full_df[
        (full_df["LED_trial"] == 0) | full_df["LED_trial"].isna()
    ].copy(),
    "on_bilateral": full_df[
        (full_df["LED_trial"] == 1)
        & (full_df["LED_powerL"] > 0)
        & (full_df["LED_powerR"] > 0)
        & (full_df["abort_event"] != 4)
    ].copy(),
}
for category, frame in full_category_frames.items():
    if len(frame) != EXPECTED_FULL_COUNTS[category]:
        raise RuntimeError(
            f"{category}: expected {EXPECTED_FULL_COUNTS[category]} RTD rows, "
            f"found {len(frame)}."
        )
    if frame.groupby(["ABL", "abs_ILD"], observed=True).ngroups != 15:
        raise RuntimeError(f"{category}: expected 15 RTD cells.")


# %%
# =============================================================================
# Exact-timing, fixed-shape model evaluators
# =============================================================================
def make_model_chunk_evaluator(category, model_params):
    params = {
        name: jnp.asarray(value, dtype=jnp.float64)
        for name, value in {**fixed_proactive_params, **model_params}.items()
        if name != "t_E_aff"
    }

    if category == "off":

        @jax.jit
        def evaluate(rt, t_stim, ABL, ILD, t_E_aff, row_mask):
            t = rt[:, None] + t_stim[None, :]
            common = {
                "params": params,
                "t_stim": t_stim[None, :],
                "ABL": ABL[None, :],
                "ILD": ILD[None, :],
                "t_E_aff": t_E_aff[None, :],
                "K_max": K_MAX,
                "include_lapse": True,
            }
            density = combined_likelihood.led_off_npl_alpha_choice_pdf_jax(
                t, 1, **common
            ) + combined_likelihood.led_off_npl_alpha_choice_pdf_jax(
                t, -1, **common
            )
            return jnp.sum(density * row_mask[None, :], axis=1)

    else:

        @jax.jit
        def evaluate(rt, t_stim, t_LED, ABL, ILD, t_E_aff, row_mask):
            t = rt[:, None] + t_stim[None, :]
            common = {
                "params": params,
                "t_stim": t_stim[None, :],
                "t_LED": t_LED[None, :],
                "ABL": ABL[None, :],
                "ILD": ILD[None, :],
                "t_E_aff": t_E_aff[None, :],
                "K_max": K_MAX,
                "n_quad": QUADRATURE_NODES,
                "include_lapse": True,
            }
            density = combined_likelihood.led_on_npl_alpha_choice_pdf_jax(
                t, 1, **common
            ) + combined_likelihood.led_on_npl_alpha_choice_pdf_jax(
                t, -1, **common
            )
            return jnp.sum(density * row_mask[None, :], axis=1)

    return evaluate


def pad(values, size):
    values = np.asarray(values, dtype=float)
    padded = np.zeros(size, dtype=float)
    padded[: len(values)] = values
    return padded


# %%
# =============================================================================
# Separate 3 x 5 pooled RTD figures
# =============================================================================
model_rt = np.arange(XLIM[0], XLIM[1] + 0.5 * MODEL_DT_S, MODEL_DT_S)
data_edges = np.arange(XLIM[0], XLIM[1] + 0.5 * DATA_BIN_S, DATA_BIN_S)
data_centers = 0.5 * (data_edges[:-1] + data_edges[1:])

rtd_payload = {
    "schema_version": 1,
    "animals": list(ANIMALS),
    "pooling": "trial_weighted",
    "data_source": str(DATA_CSV.resolve()),
    "fit_root": str(FIT_ROOT.resolve()),
    "proactive_source": str(PROACTIVE_ROOT.resolve()),
    "model_dt_s": MODEL_DT_S,
    "data_bin_s": DATA_BIN_S,
    "display_xlim_s": XLIM,
    "data_response_conventions": {
        "off": "abort_event in {3, 4} plus successful trials",
        "on_bilateral": "abort_event == 3 plus successful trials; event 4 excluded",
    },
    "categories": {},
}
metric_rows = []
rtd_pngs = {}

for category, settings in CATEGORY_SETTINGS.items():
    category_df = full_category_frames[category]
    payload = category_payloads[category]
    condition_table = payload["condition_table"]
    delay_means = np.asarray(payload["posterior_means"]["t_E_aff"], dtype=float)
    delay_lookup = {
        (float(row.ABL), float(row.ILD)): delay_means[int(row.condition_id)]
        for row in condition_table.itertuples()
    }
    model_params = {
        name: payload["posterior_means"][name]
        for name in npl_alpha.GLOBAL_PARAM_NAMES
    }
    evaluate_chunk = make_model_chunk_evaluator(category, model_params)

    cell_payload = {}
    max_density = 0.0
    for ABL in [20.0, 40.0, 60.0]:
        for abs_ILD in [1.0, 2.0, 4.0, 8.0, 16.0]:
            cell_df = category_df[
                (category_df["ABL"].astype(float) == ABL)
                & (category_df["abs_ILD"] == abs_ILD)
            ].copy()
            if cell_df.empty:
                raise RuntimeError(
                    f"{category}: empty ABL={ABL:g}, |ILD|={abs_ILD:g} cell."
                )
            n_trials = len(cell_df)
            hist_counts, _ = np.histogram(cell_df["RTwrtStim"], bins=data_edges)
            data_density = hist_counts / (n_trials * DATA_BIN_S)
            data_mass = float(np.sum(data_density) * DATA_BIN_S)
            expected_data_mass = float(
                np.mean(
                    (cell_df["RTwrtStim"] >= data_edges[0])
                    & (cell_df["RTwrtStim"] <= data_edges[-1])
                )
            )
            if not np.isclose(data_mass, expected_data_mass, atol=1e-12, rtol=0.0):
                raise RuntimeError(f"{category}: empirical histogram mass mismatch.")

            t_stim_values = cell_df["intended_fix"].to_numpy(dtype=float)
            t_led_values = cell_df["t_LED"].to_numpy(dtype=float)
            ABL_values = cell_df["ABL"].to_numpy(dtype=float)
            ILD_values = cell_df["ILD"].to_numpy(dtype=float)
            t_E_values = np.array(
                [
                    delay_lookup[(float(abl), float(ild))]
                    for abl, ild in zip(ABL_values, ILD_values)
                ]
            )
            model_sum = np.zeros_like(model_rt)

            for trial_start in range(0, n_trials, MODEL_TRIAL_CHUNK):
                trial_stop = min(trial_start + MODEL_TRIAL_CHUNK, n_trials)
                n_chunk_trials = trial_stop - trial_start
                row_mask = np.zeros(MODEL_TRIAL_CHUNK, dtype=float)
                row_mask[:n_chunk_trials] = 1.0
                trial_slice = slice(trial_start, trial_stop)
                padded_t_stim = pad(t_stim_values[trial_slice], MODEL_TRIAL_CHUNK)
                padded_t_led = pad(t_led_values[trial_slice], MODEL_TRIAL_CHUNK)
                padded_ABL = pad(ABL_values[trial_slice], MODEL_TRIAL_CHUNK)
                padded_ILD = pad(ILD_values[trial_slice], MODEL_TRIAL_CHUNK)
                padded_t_E = pad(t_E_values[trial_slice], MODEL_TRIAL_CHUNK)

                for time_start in range(0, len(model_rt), MODEL_TIME_CHUNK):
                    time_stop = min(time_start + MODEL_TIME_CHUNK, len(model_rt))
                    n_chunk_times = time_stop - time_start
                    padded_rt = np.full(MODEL_TIME_CHUNK, model_rt[time_stop - 1])
                    padded_rt[:n_chunk_times] = model_rt[time_start:time_stop]
                    if category == "off":
                        chunk_density = evaluate_chunk(
                            jnp.asarray(padded_rt),
                            jnp.asarray(padded_t_stim),
                            jnp.asarray(padded_ABL),
                            jnp.asarray(padded_ILD),
                            jnp.asarray(padded_t_E),
                            jnp.asarray(row_mask),
                        )
                    else:
                        chunk_density = evaluate_chunk(
                            jnp.asarray(padded_rt),
                            jnp.asarray(padded_t_stim),
                            jnp.asarray(padded_t_led),
                            jnp.asarray(padded_ABL),
                            jnp.asarray(padded_ILD),
                            jnp.asarray(padded_t_E),
                            jnp.asarray(row_mask),
                        )
                    model_sum[time_start:time_stop] += np.asarray(
                        jax.device_get(chunk_density[:n_chunk_times]), dtype=float
                    )

            model_density = model_sum / n_trials
            if not np.all(np.isfinite(model_density)) or np.any(model_density < -1e-12):
                raise RuntimeError(f"{category}: invalid model density.")
            model_density = np.maximum(model_density, 0.0)
            model_mass = float(np.trapz(model_density, model_rt))
            max_density = max(
                max_density,
                float(np.max(data_density)),
                float(np.max(model_density)),
            )
            key = f"ABL{int(ABL)}_absILD{int(abs_ILD)}"
            cell_payload[key] = {
                "ABL": ABL,
                "abs_ILD": abs_ILD,
                "n_trials": n_trials,
                "data_centers_s": data_centers,
                "data_density": data_density,
                "data_displayed_mass": data_mass,
                "model_rt_s": model_rt,
                "model_density": model_density,
                "model_displayed_mass": model_mass,
            }
            metric_rows.append(
                {
                    "category": category,
                    "accepted_initialization": payload["accepted_mode"],
                    "ABL": ABL,
                    "abs_ILD": abs_ILD,
                    "n_eligible_trials": n_trials,
                    "data_displayed_mass": data_mass,
                    "model_displayed_mass": model_mass,
                }
            )

    rtd_payload["categories"][category] = {
        "accepted_initialization": payload["accepted_mode"],
        "accepted_fit_root": str(payload["accepted_fit_dir"].resolve()),
        "cells": cell_payload,
    }
    fig, axes = plt.subplots(3, 5, figsize=(14.5, 8.0), sharex=True, sharey=True)
    for row_index, ABL in enumerate([20.0, 40.0, 60.0]):
        for col_index, abs_ILD in enumerate([1.0, 2.0, 4.0, 8.0, 16.0]):
            ax = axes[row_index, col_index]
            cell = cell_payload[f"ABL{int(ABL)}_absILD{int(abs_ILD)}"]
            ax.step(
                cell["data_centers_s"],
                cell["data_density"],
                where="mid",
                color="black",
                alpha=0.58,
                linewidth=0.9,
                label="pooled data" if row_index == 0 and col_index == 0 else None,
            )
            ax.plot(
                cell["model_rt_s"],
                cell["model_density"],
                color=settings["model_color"],
                linewidth=1.35,
                label="model" if row_index == 0 and col_index == 0 else None,
            )
            if row_index == 0:
                ax.set_title(rf"$|ILD|={abs_ILD:g}$ dB")
            if col_index == 0:
                ax.set_ylabel(f"ABL {ABL:g} dB\ndensity")
            if row_index == 2:
                ax.set_xlabel("RT wrt stimulus (s)")
            ax.text(
                0.98,
                0.94,
                f"n={cell['n_trials']}\n"
                f"mass {cell['data_displayed_mass']:.3f}/"
                f"{cell['model_displayed_mass']:.3f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=6.8,
            )
            ax.set_xlim(XLIM)
            ax.set_ylim(0.0, 1.06 * max_density)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    response_label = (
        "abort events 3/4 + successful RTDs"
        if category == "off"
        else "abort-event 3 + successful RTDs"
    )
    fig.suptitle(
        f"LED7 super animal {settings['label']}: {response_label}",
        y=1.01,
    )
    fig.tight_layout()
    rtd_png = OUTPUT_DIR / f"led7_super_animal_{category}_valid_npl_alpha_rtds.png"
    fig.savefig(rtd_png, dpi=220, bbox_inches="tight")
    rtd_pngs[category] = rtd_png


# %%
# =============================================================================
# Save reusable RTD payload and metrics
# =============================================================================
rtd_payload_path = OUTPUT_DIR / "led7_super_animal_valid_npl_alpha_rtd_payload.pkl"
rtd_metrics_path = OUTPUT_DIR / "led7_super_animal_valid_npl_alpha_rtd_metrics.csv"
with rtd_payload_path.open("wb") as handle:
    pickle.dump(rtd_payload, handle)
pd.DataFrame(metric_rows).to_csv(rtd_metrics_path, index=False)

print(f"Convergence figure: {convergence_png.resolve()}")
for category, path in rtd_pngs.items():
    print(f"{category} RTD figure: {path.resolve()}")
print(f"RTD payload: {rtd_payload_path.resolve()}")
print(f"RTD metrics: {rtd_metrics_path.resolve()}")
