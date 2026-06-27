# %%
"""
Diagnose SVI loss-minimum selection rules on the six no-early-stop audit runs.

The goal is to choose an online rule that lets the big Gamma/Omega/delay SVI
run past the premature stable-window stop, but still returns parameters from
the useful loss minimum rather than from the later rising part of the curve.
"""

# %%
# =============================================================================
# Parameters
# =============================================================================
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

ORIGINAL_OUTPUT_ROOT = SCRIPT_DIR / "svi_big_gamma_omega_delay_all_animals_outputs"
AUDIT_OUTPUT_ROOT = (
    SCRIPT_DIR
    / "svi_big_gamma_omega_delay_convergence_audit_outputs"
    / "no_early_stop_50k"
)
OUTPUT_DIR = AUDIT_OUTPUT_ROOT / "loss_minima_method_diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SELECTED_PAIRS = [
    ("SD", 49),
    ("LED8", 105),
    ("SD", 52),
    ("LED6", 81),
    ("LED34", 63),
    ("LED7", 103),
]

ROLLING_WINDOW = 5
ROLLING_MIN_PERIODS = 3
PATIENCE_MIN_STEPS = 5000
PATIENCE_WINDOWS = 8
PATIENCE_MIN_DELTA = 0.0
REBOUND_MIN_STEPS = 5000
REBOUND_PER_TRIAL = 0.0002
REBOUND_ABS_FLOOR = 1.0
REBOUND_PATIENCE_WINDOWS = 8
DERIVATIVE_CROSS_MIN_STEPS = 5000
DERIVATIVE_YLIM = (-2.5, 1.0)

SUMMARY_CSV = OUTPUT_DIR / "big_gamma_omega_delay_loss_minima_method_summary.csv"
DERIVATIVE_SUMMARY_CSV = OUTPUT_DIR / "big_gamma_omega_delay_loss_minima_derivative_summary.csv"
FIG_PATH = OUTPUT_DIR / "big_gamma_omega_delay_loss_minima_methods.png"
DERIVATIVE_FIG_PATH = OUTPUT_DIR / "big_gamma_omega_delay_loss_derivative_at_selected_minima.png"

METHOD_COLORS = {
    "original_early_stop": "tab:red",
    "raw_window_min": "tab:blue",
    "rolling5_median_min": "tab:purple",
    "patience8_restore_best": "tab:orange",
    "derivative_zero_cross_bracket": "tab:brown",
    "rebound_confirmed_min": "tab:green",
    "rebound_confirmed_stop": "0.35",
}


# %%
# =============================================================================
# Helpers
# =============================================================================
def animal_prefix(batch: str, animal: int) -> str:
    return f"{batch}_{animal}_big_gamma_omega_delay"


def animal_paths(root: Path, batch: str, animal: int):
    output_dir = root / f"{batch}_{animal}"
    prefix = animal_prefix(batch, animal)
    return {
        "convergence": output_dir / f"{prefix}_convergence_checks.csv",
        "condition_summary": output_dir / f"{prefix}_condition_summary.csv",
    }


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def select_rebound_confirmed_min(convergence_df: pd.DataFrame, n_trials: int):
    best_idx = None
    best_mean = np.inf
    rebound_count = 0
    stop_idx = None
    rebound_threshold = max(REBOUND_ABS_FLOOR, REBOUND_PER_TRIAL * n_trials)

    for idx, row in convergence_df.iterrows():
        mean_loss = float(row["mean_loss"])
        end_step = int(row["end_step"])
        if not np.isfinite(mean_loss):
            continue

        if mean_loss < best_mean:
            best_idx = idx
            best_mean = mean_loss
            rebound_count = 0
            continue

        if end_step >= REBOUND_MIN_STEPS and mean_loss > best_mean + rebound_threshold:
            rebound_count += 1
        else:
            rebound_count = 0

        if rebound_count >= REBOUND_PATIENCE_WINDOWS:
            stop_idx = idx
            break

    if best_idx is None:
        raise RuntimeError("Could not select a rebound-confirmed minimum.")
    if stop_idx is None:
        stop_idx = len(convergence_df) - 1

    return {
        "selected_idx": int(best_idx),
        "selected_step": int(convergence_df.loc[best_idx, "end_step"]),
        "selected_loss": float(convergence_df.loc[best_idx, "mean_loss"]),
        "stop_idx": int(stop_idx),
        "stop_step": int(convergence_df.loc[stop_idx, "end_step"]),
        "rebound_threshold": float(rebound_threshold),
    }


def select_patience_restore_best(convergence_df: pd.DataFrame):
    best_idx = None
    best_mean = np.inf
    no_improve_count = 0
    stop_idx = None

    for idx, row in convergence_df.iterrows():
        mean_loss = float(row["mean_loss"])
        end_step = int(row["end_step"])
        if not np.isfinite(mean_loss):
            continue

        if mean_loss < best_mean - PATIENCE_MIN_DELTA:
            best_idx = idx
            best_mean = mean_loss
            no_improve_count = 0
        elif end_step >= PATIENCE_MIN_STEPS:
            no_improve_count += 1

        if end_step >= PATIENCE_MIN_STEPS and no_improve_count >= PATIENCE_WINDOWS:
            stop_idx = idx
            break

    if best_idx is None:
        raise RuntimeError("Could not select a patience restore-best minimum.")
    if stop_idx is None:
        stop_idx = len(convergence_df) - 1

    return {
        "selected_idx": int(best_idx),
        "selected_step": int(convergence_df.loc[best_idx, "end_step"]),
        "selected_loss": float(convergence_df.loc[best_idx, "mean_loss"]),
        "stop_idx": int(stop_idx),
        "stop_step": int(convergence_df.loc[stop_idx, "end_step"]),
    }


def centered_derivative_per_1k(steps, values):
    steps = np.asarray(steps, dtype=float)
    values = np.asarray(values, dtype=float)
    derivative = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(steps) & np.isfinite(values)
    if finite.sum() >= 2:
        derivative[finite] = np.gradient(values[finite], steps[finite]) * 1000.0
    return derivative


def select_derivative_zero_cross_bracket(convergence_df: pd.DataFrame, rolling_derivative: np.ndarray):
    cross_idx = None
    for idx in range(1, len(convergence_df)):
        end_step = int(convergence_df.loc[idx, "end_step"])
        prev_deriv = float(rolling_derivative[idx - 1])
        this_deriv = float(rolling_derivative[idx])
        if end_step < DERIVATIVE_CROSS_MIN_STEPS:
            continue
        if np.isfinite(prev_deriv) and np.isfinite(this_deriv) and prev_deriv <= 0 <= this_deriv:
            cross_idx = idx
            break

    if cross_idx is None:
        cross_idx = int(convergence_df["mean_loss"].idxmin())

    candidate_indices = sorted(set([max(0, cross_idx - 1), cross_idx]))
    selected_idx = min(candidate_indices, key=lambda idx: float(convergence_df.loc[idx, "mean_loss"]))

    return {
        "selected_idx": int(selected_idx),
        "selected_step": int(convergence_df.loc[selected_idx, "end_step"]),
        "selected_loss": float(convergence_df.loc[selected_idx, "mean_loss"]),
        "stop_idx": int(cross_idx),
        "stop_step": int(convergence_df.loc[cross_idx, "end_step"]),
    }


# %%
# =============================================================================
# Method comparison
# =============================================================================
summary_rows = []
derivative_rows = []
plot_payloads = []

for batch, animal in SELECTED_PAIRS:
    label = f"{batch}/{animal}"
    audit_paths = animal_paths(AUDIT_OUTPUT_ROOT, batch, animal)
    original_paths = animal_paths(ORIGINAL_OUTPUT_ROOT, batch, animal)

    audit_conv = load_csv(audit_paths["convergence"]).copy()
    original_conv = load_csv(original_paths["convergence"]).copy()
    condition_summary = load_csv(audit_paths["condition_summary"])

    audit_conv = audit_conv.sort_values("end_step").reset_index(drop=True)
    original_conv = original_conv.sort_values("end_step").reset_index(drop=True)
    n_trials = int(condition_summary["n_trials"].sum())

    original_stop_step = int(original_conv["end_step"].max())
    original_stop_loss = float(original_conv["mean_loss"].iloc[-1])

    raw_idx = int(audit_conv["mean_loss"].idxmin())
    rolling_loss = audit_conv["mean_loss"].rolling(
        ROLLING_WINDOW,
        min_periods=ROLLING_MIN_PERIODS,
    ).median()
    raw_derivative = centered_derivative_per_1k(audit_conv["end_step"], audit_conv["mean_loss"])
    rolling_derivative = centered_derivative_per_1k(audit_conv["end_step"], rolling_loss)
    rolling_idx = int(rolling_loss.idxmin())
    patience_selection = select_patience_restore_best(audit_conv)
    derivative_selection = select_derivative_zero_cross_bracket(audit_conv, rolling_derivative)
    rebound_selection = select_rebound_confirmed_min(audit_conv, n_trials)

    method_results = {
        "raw_window_min": {
            "selected_idx": raw_idx,
            "selected_step": int(audit_conv.loc[raw_idx, "end_step"]),
            "selected_loss": float(audit_conv.loc[raw_idx, "mean_loss"]),
            "score": float(audit_conv.loc[raw_idx, "mean_loss"]),
            "stop_step": np.nan,
            "rebound_threshold": np.nan,
        },
        "rolling5_median_min": {
            "selected_idx": rolling_idx,
            "selected_step": int(audit_conv.loc[rolling_idx, "end_step"]),
            "selected_loss": float(audit_conv.loc[rolling_idx, "mean_loss"]),
            "score": float(rolling_loss.iloc[rolling_idx]),
            "stop_step": np.nan,
            "rebound_threshold": np.nan,
        },
        "patience8_restore_best": {
            "selected_idx": patience_selection["selected_idx"],
            "selected_step": patience_selection["selected_step"],
            "selected_loss": patience_selection["selected_loss"],
            "score": patience_selection["selected_loss"],
            "stop_step": patience_selection["stop_step"],
            "rebound_threshold": np.nan,
        },
        "derivative_zero_cross_bracket": {
            "selected_idx": derivative_selection["selected_idx"],
            "selected_step": derivative_selection["selected_step"],
            "selected_loss": derivative_selection["selected_loss"],
            "score": derivative_selection["selected_loss"],
            "stop_step": derivative_selection["stop_step"],
            "rebound_threshold": np.nan,
        },
        "rebound_confirmed_min": {
            "selected_idx": rebound_selection["selected_idx"],
            "selected_step": rebound_selection["selected_step"],
            "selected_loss": rebound_selection["selected_loss"],
            "score": rebound_selection["selected_loss"],
            "stop_step": rebound_selection["stop_step"],
            "rebound_threshold": rebound_selection["rebound_threshold"],
        },
    }

    raw_best_loss = method_results["raw_window_min"]["selected_loss"]
    for method, result in method_results.items():
        selected_loss = float(result["selected_loss"])
        selected_idx = int(result["selected_idx"])
        prev_idx = max(0, selected_idx - 1)
        next_idx = min(len(audit_conv) - 1, selected_idx + 1)
        summary_rows.append(
            {
                "batch_name": batch,
                "animal": animal,
                "animal_label": label,
                "n_trials": n_trials,
                "method": method,
                "selected_step": int(result["selected_step"]),
                "selected_loss": selected_loss,
                "method_score": float(result["score"]),
                "original_stop_step": original_stop_step,
                "original_stop_loss": original_stop_loss,
                "stop_step": result["stop_step"],
                "raw_best_step": int(method_results["raw_window_min"]["selected_step"]),
                "raw_best_loss": raw_best_loss,
                "delta_from_raw_best": selected_loss - raw_best_loss,
                "delta_from_raw_best_per_trial": (selected_loss - raw_best_loss) / n_trials,
                "improvement_over_original_stop": original_stop_loss - selected_loss,
                "improvement_over_original_stop_per_trial": (original_stop_loss - selected_loss) / n_trials,
                "rebound_threshold": result["rebound_threshold"],
            }
        )
        derivative_rows.append(
            {
                "batch_name": batch,
                "animal": animal,
                "animal_label": label,
                "n_trials": n_trials,
                "method": method,
                "selected_step": int(result["selected_step"]),
                "raw_loss_derivative_at_selected_per_1k": float(raw_derivative[selected_idx]),
                "raw_loss_derivative_prev_per_1k": float(raw_derivative[prev_idx]),
                "raw_loss_derivative_next_per_1k": float(raw_derivative[next_idx]),
                "rolling_loss_derivative_at_selected_per_1k": float(rolling_derivative[selected_idx]),
                "rolling_loss_derivative_prev_per_1k": float(rolling_derivative[prev_idx]),
                "rolling_loss_derivative_next_per_1k": float(rolling_derivative[next_idx]),
                "raw_loss_prev": float(audit_conv.loc[prev_idx, "mean_loss"]),
                "raw_loss_selected": selected_loss,
                "raw_loss_next": float(audit_conv.loc[next_idx, "mean_loss"]),
            }
        )

    plot_payloads.append(
        {
            "label": label,
            "n_trials": n_trials,
            "convergence": audit_conv,
            "rolling_loss": rolling_loss,
            "raw_derivative": raw_derivative,
            "rolling_derivative": rolling_derivative,
            "original_stop_step": original_stop_step,
            "original_stop_loss": original_stop_loss,
            "method_results": method_results,
            "patience_stop_step": patience_selection["stop_step"],
            "derivative_cross_step": derivative_selection["stop_step"],
            "rebound_stop_step": rebound_selection["stop_step"],
        }
    )

summary_df = pd.DataFrame(summary_rows)
derivative_df = pd.DataFrame(derivative_rows)
summary_df.to_csv(SUMMARY_CSV, index=False)
derivative_df.to_csv(DERIVATIVE_SUMMARY_CSV, index=False)

print("Saved summary:")
print(f"  {SUMMARY_CSV}")
print(f"  {DERIVATIVE_SUMMARY_CSV}")
print("\nMethod selections:")
print(
    summary_df[
        [
            "animal_label",
            "method",
            "selected_step",
            "delta_from_raw_best_per_trial",
            "improvement_over_original_stop_per_trial",
            "stop_step",
        ]
    ].to_string(index=False)
)

print("\nDerivative at selected online minima:")
print(
    derivative_df[
        derivative_df["method"].isin(
            [
                "patience8_restore_best",
                "derivative_zero_cross_bracket",
                "rebound_confirmed_min",
            ]
        )
    ][
        [
            "animal_label",
            "method",
            "selected_step",
            "raw_loss_derivative_prev_per_1k",
            "raw_loss_derivative_at_selected_per_1k",
            "raw_loss_derivative_next_per_1k",
            "rolling_loss_derivative_at_selected_per_1k",
        ]
    ].to_string(index=False)
)


# %%
# =============================================================================
# Plot minima methods
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 8.2), sharex=False, sharey=False)
axes_flat = axes.ravel()

for ax, payload in zip(axes_flat, plot_payloads):
    conv = payload["convergence"]
    method_results = payload["method_results"]

    ax.plot(
        conv["end_step"],
        conv["mean_loss"],
        color="0.15",
        linewidth=1.0,
        label="1k-window mean",
    )
    ax.plot(
        conv["end_step"],
        payload["rolling_loss"],
        color=METHOD_COLORS["rolling5_median_min"],
        linewidth=1.1,
        alpha=0.55,
        label="rolling-5 median",
    )

    ax.axvline(
        payload["original_stop_step"],
        color=METHOD_COLORS["original_early_stop"],
        linestyle="--",
        linewidth=1.4,
        alpha=0.8,
    )
    for method, result in method_results.items():
        if method == "derivative_zero_cross_bracket":
            linestyle = "-."
        elif method == "patience8_restore_best":
            linestyle = "--"
        else:
            linestyle = "-"
        ax.axvline(
            result["selected_step"],
            color=METHOD_COLORS[method],
            linestyle=linestyle,
            linewidth=1.4 if method != "rebound_confirmed_min" else 2.0,
            alpha=0.85,
        )

    rebound_stop_step = payload["rebound_stop_step"]
    patience_stop_step = payload["patience_stop_step"]
    derivative_cross_step = payload["derivative_cross_step"]
    ax.axvline(
        rebound_stop_step,
        color=METHOD_COLORS["rebound_confirmed_stop"],
        linestyle=":",
        linewidth=1.4,
        alpha=0.75,
    )
    ax.axvline(
        patience_stop_step,
        color=METHOD_COLORS["patience8_restore_best"],
        linestyle=":",
        linewidth=1.2,
        alpha=0.45,
    )
    ax.axvline(
        derivative_cross_step,
        color=METHOD_COLORS["derivative_zero_cross_bracket"],
        linestyle=":",
        linewidth=1.2,
        alpha=0.45,
    )

    raw_step = method_results["raw_window_min"]["selected_step"]
    rebound_step = method_results["rebound_confirmed_min"]["selected_step"]
    rolling_step = method_results["rolling5_median_min"]["selected_step"]
    patience_step = method_results["patience8_restore_best"]["selected_step"]
    derivative_step = method_results["derivative_zero_cross_bracket"]["selected_step"]
    text = (
        f"old stop={payload['original_stop_step'] / 1000:.0f}k\n"
        f"raw={raw_step / 1000:.0f}k, roll={rolling_step / 1000:.0f}k\n"
        f"patience={patience_step / 1000:.0f}k, deriv={derivative_step / 1000:.0f}k\n"
        f"green pick={rebound_step / 1000:.0f}k, stop={rebound_stop_step / 1000:.0f}k"
    )
    ax.text(
        0.98,
        0.96,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "0.85", "alpha": 0.9},
    )

    ax.set_title(payload["label"])
    ax.set_xlabel("SVI step")
    ax.set_ylabel("1k-window mean negative ELBO")
    ax.grid(True, alpha=0.25)

legend_handles = [
    Line2D([0], [0], color="0.15", lw=1.0, label="1k-window mean"),
    Line2D([0], [0], color=METHOD_COLORS["original_early_stop"], lw=1.4, linestyle="--", label="old early stop"),
    Line2D([0], [0], color=METHOD_COLORS["raw_window_min"], lw=1.5, label="raw window min"),
    Line2D([0], [0], color=METHOD_COLORS["rolling5_median_min"], lw=1.5, label="rolling-5 median min"),
    Line2D([0], [0], color=METHOD_COLORS["patience8_restore_best"], lw=1.4, linestyle="--", label="patience restore-best"),
    Line2D([0], [0], color=METHOD_COLORS["derivative_zero_cross_bracket"], lw=1.4, linestyle="-.", label="derivative bracket"),
    Line2D([0], [0], color=METHOD_COLORS["rebound_confirmed_min"], lw=2.0, label="rebound-confirmed pick"),
    Line2D([0], [0], color=METHOD_COLORS["rebound_confirmed_stop"], lw=1.4, linestyle=":", label="rebound stop"),
]
fig.legend(
    handles=legend_handles,
    frameon=False,
    fontsize=9,
    loc="upper right",
    bbox_to_anchor=(0.995, 1.01),
    ncol=4,
)
fig.suptitle("Loss-minimum selection methods on 50k no-early-stop audit curves", y=1.03)
fig.tight_layout()
fig.savefig(FIG_PATH, dpi=200, bbox_inches="tight")
print(f"Saved figure: {FIG_PATH}")


# %%
# =============================================================================
# Plot derivative at selected minima
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 8.2), sharex=False, sharey=False)
axes_flat = axes.ravel()

for ax, payload in zip(axes_flat, plot_payloads):
    conv = payload["convergence"]
    method_results = payload["method_results"]
    rebound_pick_step = method_results["rebound_confirmed_min"]["selected_step"]
    patience_pick_step = method_results["patience8_restore_best"]["selected_step"]
    derivative_pick_step = method_results["derivative_zero_cross_bracket"]["selected_step"]
    rebound_stop_step = payload["rebound_stop_step"]

    ax.plot(
        conv["end_step"],
        payload["raw_derivative"],
        color="0.45",
        linewidth=1.0,
        alpha=0.7,
        label="raw derivative",
    )
    ax.plot(
        conv["end_step"],
        payload["rolling_derivative"],
        color="tab:blue",
        linewidth=1.4,
        label="rolling-5 derivative",
    )
    ax.axhline(0.0, color="0.15", linestyle="-", linewidth=1.0, alpha=0.75)
    ax.axvline(
        payload["original_stop_step"],
        color=METHOD_COLORS["original_early_stop"],
        linestyle="--",
        linewidth=1.4,
        alpha=0.8,
    )
    ax.axvline(
        patience_pick_step,
        color=METHOD_COLORS["patience8_restore_best"],
        linestyle="--",
        linewidth=1.4,
        alpha=0.85,
    )
    ax.axvline(
        derivative_pick_step,
        color=METHOD_COLORS["derivative_zero_cross_bracket"],
        linestyle="-.",
        linewidth=1.4,
        alpha=0.85,
    )
    ax.axvline(
        rebound_pick_step,
        color=METHOD_COLORS["rebound_confirmed_min"],
        linestyle="-",
        linewidth=2.0,
        alpha=0.85,
    )
    ax.axvline(
        rebound_stop_step,
        color=METHOD_COLORS["rebound_confirmed_stop"],
        linestyle=":",
        linewidth=1.4,
        alpha=0.75,
    )

    rebound_derivative_row = derivative_df[
        (derivative_df["animal_label"] == payload["label"])
        & (derivative_df["method"] == "rebound_confirmed_min")
    ].iloc[0]
    derivative_method_row = derivative_df[
        (derivative_df["animal_label"] == payload["label"])
        & (derivative_df["method"] == "derivative_zero_cross_bracket")
    ].iloc[0]
    text = (
        f"green={rebound_pick_step / 1000:.0f}k, patience={patience_pick_step / 1000:.0f}k\n"
        f"deriv={derivative_pick_step / 1000:.0f}k, stop={rebound_stop_step / 1000:.0f}k\n"
        f"green dL={rebound_derivative_row['raw_loss_derivative_at_selected_per_1k']:.3g}/1k, "
        f"deriv dL={derivative_method_row['raw_loss_derivative_at_selected_per_1k']:.3g}/1k"
    )
    ax.text(
        0.98,
        0.96,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "0.85", "alpha": 0.9},
    )

    ax.set_title(payload["label"])
    ax.set_xlabel("SVI step")
    ax.set_ylabel("d(negative ELBO)/dstep x 1000")
    ax.set_ylim(DERIVATIVE_YLIM)
    ax.grid(True, alpha=0.25)

derivative_legend_handles = [
    Line2D([0], [0], color="0.45", lw=1.0, label="raw centered derivative"),
    Line2D([0], [0], color="tab:blue", lw=1.4, label="rolling-5 centered derivative"),
    Line2D([0], [0], color="0.15", lw=1.0, label="zero derivative"),
    Line2D([0], [0], color=METHOD_COLORS["original_early_stop"], lw=1.4, linestyle="--", label="old early stop"),
    Line2D([0], [0], color=METHOD_COLORS["patience8_restore_best"], lw=1.4, linestyle="--", label="patience restore-best"),
    Line2D([0], [0], color=METHOD_COLORS["derivative_zero_cross_bracket"], lw=1.4, linestyle="-.", label="derivative bracket"),
    Line2D([0], [0], color=METHOD_COLORS["rebound_confirmed_min"], lw=2.0, label="rebound-confirmed pick"),
    Line2D([0], [0], color=METHOD_COLORS["rebound_confirmed_stop"], lw=1.4, linestyle=":", label="rebound stop"),
]
fig.legend(
    handles=derivative_legend_handles,
    frameon=False,
    fontsize=9,
    loc="upper right",
    bbox_to_anchor=(0.995, 1.01),
    ncol=4,
)
fig.suptitle("Derivative check for rebound-confirmed selected minima", y=1.03)
fig.tight_layout()
fig.savefig(DERIVATIVE_FIG_PATH, dpi=200, bbox_inches="tight")
print(f"Saved derivative figure: {DERIVATIVE_FIG_PATH}")

plt.show()

# %%
