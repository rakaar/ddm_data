# %%
"""Diagnostics for the LED7/98 OFF and bilateral-ON valid-trial pilot fits."""

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
ANIMAL = 98

PILOT_ROOT = Path(
    os.environ.get(
        "LED7_VALID_NPL_DIAGNOSTIC_PILOT_ROOT",
        str(
            SCRIPT_DIR
            / "numpyro_svi_proactive_led_step_jump_npl_alpha_valid_"
            "led7_98_pilot_outputs"
        ),
    )
).expanduser()
CATEGORY_FIT_DIRS = {
    "off": Path(
        os.environ.get(
            "LED7_VALID_NPL_DIAGNOSTIC_OFF_FIT_DIR",
            str(PILOT_ROOT / "off"),
        )
    ).expanduser(),
    "on_bilateral": Path(
        os.environ.get(
            "LED7_VALID_NPL_DIAGNOSTIC_ON_FIT_DIR",
            str(PILOT_ROOT / "on_bilateral"),
        )
    ).expanduser(),
}
REFERENCE_ROOT = (
    ANIMAL_FIT_DIR
    / "numpyro_svi_npl_alpha_condition_delay_patience12_restore_best_outputs"
    / f"LED7_{ANIMAL}"
)
PROACTIVE_ROOT = (
    SCRIPT_DIR
    / "numpyro_svi_proactive_led_step_jump_all_on_no_trunc_exp_lapse_"
    "patience12_min50k_restore_best_outputs"
    / f"LED7_{ANIMAL}"
)
DATA_CSV = REPO_DIR / "out_LED.csv"

OUTPUT_DIR = Path(
    os.environ.get(
        "LED7_VALID_NPL_DIAGNOSTIC_OUTPUT_DIR",
        str(PILOT_ROOT / "diagnostics"),
    )
).expanduser()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

K_MAX = 10
QUADRATURE_NODES = 64
MODEL_DT_S = 0.001
MODEL_TIME_CHUNK = 100
MODEL_TRIAL_CHUNK = 256
OFF_LOSS_XMAX_STEPS = 2500
OFF_LOSS_SMOOTHING_STEPS = 100

CATEGORY_SETTINGS = {
    "off": {
        "xlim": (-1.0, 1.0),
        "data_bin_s": 0.020,
        "model_color": "tab:blue",
        "label": "LED OFF",
    },
    "on_bilateral": {
        "xlim": (-1.0, 1.0),
        "data_bin_s": 0.050,
        "model_color": "tab:red",
        "label": "bilateral LED ON",
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
import numpy as np
import pandas as pd

for import_dir in [SCRIPT_DIR, ANIMAL_FIT_DIR]:
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

import numpyro_npl_alpha_svi_utils as npl_alpha
import numpyro_proactive_led_step_jump_npl_alpha_utils as combined_likelihood


# %%
# =============================================================================
# Load and validate the three posterior sources
# =============================================================================
category_payloads = {}
for category in CATEGORY_SETTINGS:
    fit_dir = CATEGORY_FIT_DIRS[category]
    summary_path = fit_dir / "run_summary.json"
    sample_path = fit_dir / "main_fullrank_posterior_samples.npz"
    condition_path = fit_dir / "condition_table.csv"
    convergence_path = fit_dir / "main_fullrank_convergence_checks.csv"
    loss_path = fit_dir / "main_fullrank_loss.csv"
    for path in [summary_path, sample_path, condition_path, convergence_path, loss_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    with summary_path.open() as handle:
        run_summary = json.load(handle)
    if run_summary.get("status") != "complete":
        raise RuntimeError(
            f"{category} fit is not patience-converged: "
            f"status={run_summary.get('status')!r}."
        )
    with np.load(sample_path) as saved:
        posterior = {name: np.asarray(saved[name]) for name in saved.files}
    if not all(np.all(np.isfinite(values)) for values in posterior.values()):
        raise RuntimeError(f"{category} posterior contains non-finite samples.")

    condition_table = pd.read_csv(condition_path).sort_values("condition_id")
    if len(condition_table) != 30:
        raise RuntimeError(f"{category}: expected 30 condition rows.")
    category_payloads[category] = {
        "fit_dir": fit_dir,
        "run_summary": run_summary,
        "posterior": posterior,
        "posterior_means": {
            name: float(np.mean(values)) if values.ndim == 1 else np.mean(values, axis=0)
            for name, values in posterior.items()
        },
        "condition_table": condition_table,
        "convergence": pd.read_csv(convergence_path),
        "loss": pd.read_csv(loss_path),
    }

reference_sample_path = REFERENCE_ROOT / "main_fullrank_posterior_samples.npz"
reference_condition_path = REFERENCE_ROOT / "condition_table.csv"
proactive_sample_path = PROACTIVE_ROOT / "main_fullrank_posterior_samples.npz"
for path in [reference_sample_path, reference_condition_path, proactive_sample_path, DATA_CSV]:
    if not path.exists():
        raise FileNotFoundError(path)

with np.load(reference_sample_path) as saved:
    reference_posterior = {name: np.asarray(saved[name]) for name in saved.files}
reference_conditions = pd.read_csv(reference_condition_path).sort_values("condition_id")
with np.load(proactive_sample_path) as saved:
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


# %%
# =============================================================================
# 1 x 2 convergence figure
# =============================================================================
convergence_png = OUTPUT_DIR / "led7_98_valid_npl_alpha_pilot_convergence.png"
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), sharey=False)
for ax, category in zip(axes, CATEGORY_SETTINGS):
    payload = category_payloads[category]
    convergence = payload["convergence"]
    summary = payload["run_summary"]
    best_step = int(summary["restored_best_step"])
    checked_step = int(summary["completed_steps"])
    if category == "off":
        loss = payload["loss"]
        smoothed_loss = loss["negative_elbo"].rolling(
            OFF_LOSS_SMOOTHING_STEPS,
            min_periods=20,
        ).mean()
        visible = (loss["step"] <= OFF_LOSS_XMAX_STEPS) & smoothed_loss.notna()
        ax.plot(
            loss.loc[visible, "step"],
            smoothed_loss.loc[visible],
            color=CATEGORY_SETTINGS[category]["model_color"],
            linewidth=1.0,
            label=f"{OFF_LOSS_SMOOTHING_STEPS}-step mean",
        )
        ax.set_xlim(0, OFF_LOSS_XMAX_STEPS)
    else:
        ax.plot(
            convergence["end_step"],
            convergence["mean_loss"],
            color=CATEGORY_SETTINGS[category]["model_color"],
            linewidth=1.0,
            label="1k-window mean",
        )
    ax.axvline(best_step, color="tab:green", linewidth=1.2, label="restored best")
    if category != "off":
        ax.axvline(
            checked_step,
            color="tab:red",
            linestyle="--",
            linewidth=1.2,
            label="final checked",
        )
    ax.set_title(
        f"{CATEGORY_SETTINGS[category]['label']}\n"
        f"best {best_step / 1000:.0f}k, checked {checked_step / 1000:.0f}k"
    )
    ax.set_xlabel("SVI step")
    ax.set_ylabel("negative ELBO")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="best")
fig.suptitle("LED7/98 fixed-proactive NPL+alpha valid-trial fits", y=1.02)
fig.tight_layout()
fig.savefig(convergence_png, dpi=220, bbox_inches="tight")
plt.close(fig)


# %%
# =============================================================================
# 2 x 4 global and delay comparison
# =============================================================================
parameter_png = OUTPUT_DIR / "led7_98_valid_npl_alpha_pilot_parameter_comparison.png"
global_names = npl_alpha.GLOBAL_PARAM_NAMES
global_labels = {
    "rate_lambda": r"$\lambda$",
    "T_0": r"$T_0$ (ms)",
    "theta_E": r"$\theta_E$",
    "w": r"$w$",
    "del_go": r"$t_{go}$ (ms)",
    "rate_norm_l": r"$\ell$",
    "alpha": r"$\alpha$",
}
source_specs = [
    (
        "Previous OFF\n300 ms proactive truncation, no lapse",
        reference_posterior,
        "0.35",
        "o",
        True,
    ),
    (
        "New OFF\nno proactive truncation + lapse",
        category_payloads["off"]["posterior"],
        "tab:blue",
        "o",
        False,
    ),
    (
        "Bilateral ON\nno proactive truncation + lapse",
        category_payloads["on_bilateral"]["posterior"],
        "tab:red",
        "x",
        False,
    ),
]

fig, axes = plt.subplots(2, 4, figsize=(13.0, 6.8))
for ax, name in zip(axes.flat[:7], global_names):
    for x_index, (source_label, samples, color, marker, open_marker) in enumerate(
        source_specs
    ):
        values = np.asarray(samples[name], dtype=float).reshape(-1)
        if name in {"T_0", "del_go"}:
            values = 1000.0 * values
        mean = float(np.mean(values))
        low, high = np.quantile(values, [0.025, 0.975])
        ax.errorbar(
            x_index,
            mean,
            yerr=[[mean - low], [high - mean]],
            fmt=marker,
            color=color,
            markerfacecolor="white" if open_marker else color,
            markersize=5.5,
            capsize=2.5,
            linewidth=1.0,
            label=source_label,
        )
    ax.set_title(global_labels[name])
    ax.set_xticks(range(3), ["Previous\nOFF", "New\nOFF", "Bilateral\nON"])
    ax.spines[["top", "right"]].set_visible(False)

delay_ax = axes.flat[7]
reference_delay_means = np.mean(reference_posterior["t_E_aff"], axis=0)
reference_delay_lookup = {
    (float(row.ABL), float(row.ILD)): reference_delay_means[int(row.condition_id)]
    for row in reference_conditions.itertuples()
}
all_delay_values = [1000.0 * reference_delay_means]
for category, color, marker, label in [
    ("off", "tab:blue", "o", "New OFF"),
    (
        "on_bilateral",
        "tab:red",
        "x",
        "Bilateral ON",
    ),
]:
    condition_table = category_payloads[category]["condition_table"]
    new_means = np.asarray(category_payloads[category]["posterior_means"]["t_E_aff"])
    old_aligned = np.array(
        [
            reference_delay_lookup[(float(row.ABL), float(row.ILD))]
            for row in condition_table.itertuples()
        ]
    )
    delay_ax.scatter(
        1000.0 * old_aligned,
        1000.0 * new_means,
        color=color,
        marker=marker,
        s=24,
        alpha=0.8,
        label=label,
    )
    all_delay_values.extend([1000.0 * old_aligned, 1000.0 * new_means])
delay_limits = [
    min(float(np.min(values)) for values in all_delay_values),
    max(float(np.max(values)) for values in all_delay_values),
]
padding = max(2.0, 0.05 * (delay_limits[1] - delay_limits[0]))
delay_limits = [delay_limits[0] - padding, delay_limits[1] + padding]
delay_ax.plot(delay_limits, delay_limits, color="0.5", linestyle="--", linewidth=0.9)
delay_ax.set_xlim(delay_limits)
delay_ax.set_ylim(delay_limits)
delay_ax.set_aspect("equal", adjustable="box")
delay_ax.set_title(r"Condition $t_{E,aff}$ means")
delay_ax.set_xlabel("Previous OFF delay (ms)")
delay_ax.set_ylabel("New-fit delay (ms)")
delay_ax.spines[["top", "right"]].set_visible(False)
delay_ax.legend(frameon=False, fontsize=7, loc="lower right")

handles, labels = axes.flat[0].get_legend_handles_labels()
fig.suptitle(
    "LED7/98 reactive posterior comparison across proactive conventions",
    y=0.995,
)
fig.legend(
    handles,
    labels,
    frameon=False,
    ncol=3,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.955),
    fontsize=8.0,
)
fig.tight_layout(rect=[0, 0, 1, 0.84])
fig.savefig(parameter_png, dpi=220, bbox_inches="tight")
plt.close(fig)


# %%
# =============================================================================
# Rebuild full abort + successful-trial datasets for unconditional RTDs
# =============================================================================
raw_df = pd.read_csv(DATA_CSV)
full_df = raw_df[
    raw_df["repeat_trial"].isin([0, 2]) | raw_df["repeat_trial"].isna()
].copy()
full_df = full_df[
    (full_df["session_type"] == 7)
    & (full_df["training_level"] == 16)
    & (full_df["animal"].astype(int) == ANIMAL)
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


# %%
# =============================================================================
# Fixed-shape JAX evaluators for exact-timing model averages
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
# Separate 3 x 5 OFF and bilateral-ON RTD figures
# =============================================================================
rtd_payload = {
    "schema_version": 1,
    "animal": ANIMAL,
    "data_source": str(DATA_CSV.resolve()),
    "pilot_root": str(PILOT_ROOT.resolve()),
    "category_fit_dirs": {
        category: str(path.resolve()) for category, path in CATEGORY_FIT_DIRS.items()
    },
    "proactive_source": str(PROACTIVE_ROOT.resolve()),
    "model_dt_s": MODEL_DT_S,
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

    x_low, x_high = settings["xlim"]
    model_rt = np.arange(x_low, x_high + 0.5 * MODEL_DT_S, MODEL_DT_S)
    data_edges = np.arange(
        x_low,
        x_high + 0.5 * settings["data_bin_s"],
        settings["data_bin_s"],
    )
    data_centers = 0.5 * (data_edges[:-1] + data_edges[1:])

    cell_payload = {}
    max_density = 0.0
    for ABL in [20.0, 40.0, 60.0]:
        for abs_ILD in [1.0, 2.0, 4.0, 8.0, 16.0]:
            cell_df = category_df[
                (category_df["ABL"].astype(float) == ABL)
                & (category_df["abs_ILD"] == abs_ILD)
            ].copy()
            if cell_df.empty:
                raise RuntimeError(f"{category}: empty ABL={ABL:g}, |ILD|={abs_ILD:g} cell.")
            n_trials = len(cell_df)
            hist_counts, _ = np.histogram(cell_df["RTwrtStim"], bins=data_edges)
            data_density = hist_counts / (n_trials * settings["data_bin_s"])
            data_mass = float(np.sum(data_density) * settings["data_bin_s"])
            expected_data_mass = float(
                np.mean(
                    (cell_df["RTwrtStim"] >= x_low)
                    & (cell_df["RTwrtStim"] < data_edges[-1])
                )
            )
            if not np.isclose(data_mass, expected_data_mass, atol=1e-12, rtol=0.0):
                raise RuntimeError(f"{category}: histogram mass assertion failed.")

            t_stim_values = cell_df["intended_fix"].to_numpy(dtype=float)
            t_led_values = cell_df["t_LED"].to_numpy(dtype=float)
            ABL_values = cell_df["ABL"].to_numpy(dtype=float)
            ILD_values = cell_df["ILD"].to_numpy(dtype=float)
            t_E_values = np.array(
                [delay_lookup[(float(abl), float(ild))] for abl, ild in zip(ABL_values, ILD_values)]
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
                raise RuntimeError(f"{category}: invalid model density in a cell.")
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
                "data_bin_s": settings["data_bin_s"],
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
                    "ABL": ABL,
                    "abs_ILD": abs_ILD,
                    "n_eligible_trials": n_trials,
                    "data_bin_ms": 1000.0 * settings["data_bin_s"],
                    "data_displayed_mass": data_mass,
                    "model_displayed_mass": model_mass,
                }
            )

    rtd_payload["categories"][category] = {
        "settings": settings,
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
                label="data" if row_index == 0 and col_index == 0 else None,
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
                f"mass {cell['data_displayed_mass']:.3f}/{cell['model_displayed_mass']:.3f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=6.8,
            )
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlim(settings["xlim"])
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    for ax in axes.flat:
        ax.set_ylim(0.0, 1.06 * max_density)
    response_label = (
        "abort events 3/4 + successful-trial RTDs"
        if category == "off"
        else "abort-event 3 + successful-trial RTDs"
    )
    fig.suptitle(f"LED7/{ANIMAL} {settings['label']}: {response_label}", y=1.01)
    fig.tight_layout()
    rtd_png = OUTPUT_DIR / f"led7_98_{category}_valid_npl_alpha_choice_collapsed_rtds.png"
    fig.savefig(rtd_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    rtd_pngs[category] = rtd_png


# %%
# =============================================================================
# Save reusable RTD payload and metrics
# =============================================================================
rtd_payload_path = OUTPUT_DIR / "led7_98_valid_npl_alpha_pilot_rtd_payload.pkl"
rtd_metrics_path = OUTPUT_DIR / "led7_98_valid_npl_alpha_pilot_rtd_metrics.csv"
with rtd_payload_path.open("wb") as handle:
    pickle.dump(rtd_payload, handle)
pd.DataFrame(metric_rows).to_csv(rtd_metrics_path, index=False)

print(f"Convergence figure: {convergence_png}")
print(f"Parameter figure: {parameter_png}")
for category, path in rtd_pngs.items():
    print(f"{category} RTD figure: {path}")
print(f"RTD payload: {rtd_payload_path}")
print(f"RTD metrics: {rtd_metrics_path}")
