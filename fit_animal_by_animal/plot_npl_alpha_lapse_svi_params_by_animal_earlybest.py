# %%
"""
Plot all-animal NPL+alpha+lapse SVI parameters with early-best checkpoint marks.

This diagnostic checks whether animals whose patience12 restored-best checkpoint
was at 1k/2k steps have outlier posterior parameters.
"""

# %%
# =============================================================================
# Parameters
# =============================================================================
from pathlib import Path
import os

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = Path(
    os.environ.get(
        "NPL_ALPHA_LAPSE_SVI_OUTPUT_ROOT",
        str(
            SCRIPT_DIR
            / "numpyro_svi_npl_alpha_lapse_condition_delay_patience12_min50k_restore_best_outputs"
        ),
    )
).expanduser()
LOG_DIR = OUTPUT_ROOT / "_batch_logs"
LEDGER_CSV = LOG_DIR / "batch_run_status.csv"
SUMMARY_DIR = OUTPUT_ROOT / "summary_figures"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

FIG_PATH = SUMMARY_DIR / "npl_alpha_lapse_condition_delay_patience12_min50k_params_by_animal_earlybest.png"
PARAM_CSV = SUMMARY_DIR / "npl_alpha_lapse_condition_delay_patience12_min50k_params_by_animal_earlybest.csv"
CONVERGENCE_CSV = SUMMARY_DIR / "npl_alpha_lapse_condition_delay_patience12_min50k_earlybest_summary.csv"

EXPECTED_N_ANIMALS = 30
EARLY_BEST_THRESHOLD = 2000

PARAM_SPECS = [
    ("rate_lambda", "rate_lambda", 1.0),
    ("T_0", "T_0 (ms)", 1000.0),
    ("theta_E", "theta_E", 1.0),
    ("w", "w", 1.0),
    ("del_go", "del_go (ms)", 1000.0),
    ("rate_norm_l", "rate_norm_l", 1.0),
    ("alpha", "alpha", 1.0),
    ("lapse_prob", "lapse rate (%)", 100.0),
    ("lapse_prob_right", "lapse_prob_right", 1.0),
]

EARLY_COLOR = "#D55E00"
OTHER_COLOR = "#0072B2"
ERROR_ALPHA = 0.55


# %%
# =============================================================================
# Imports
# =============================================================================
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# %%
# =============================================================================
# Load parameter summaries and convergence metadata
# =============================================================================
if not LEDGER_CSV.exists():
    raise FileNotFoundError(LEDGER_CSV)

ledger_df = pd.read_csv(LEDGER_CSV).sort_values("run_index").reset_index(drop=True)
if len(ledger_df) != EXPECTED_N_ANIMALS:
    raise RuntimeError(f"Expected {EXPECTED_N_ANIMALS} animals in {LEDGER_CSV}, found {len(ledger_df)}.")

accepted_statuses = {"completed", "skipped_existing"}
bad_status = ledger_df[~ledger_df["status"].isin(accepted_statuses)]
if not bad_status.empty:
    raise RuntimeError("Some animals are not completed or skipped-existing:\n" + bad_status.to_string(index=False))

param_rows = []
convergence_rows = []

for _, ledger_row in ledger_df.iterrows():
    batch_name = str(ledger_row["batch_name"])
    animal = int(ledger_row["animal"])
    animal_label = f"{batch_name}/{animal}"
    animal_dir = OUTPUT_ROOT / f"{batch_name}_{animal}"
    summary_csv = animal_dir / "main_fullrank_posterior_summary.csv"
    convergence_csv = animal_dir / "main_fullrank_convergence_checks.csv"

    if not summary_csv.exists():
        raise FileNotFoundError(summary_csv)
    if not convergence_csv.exists():
        raise FileNotFoundError(convergence_csv)

    summary_df = pd.read_csv(summary_csv)
    conv_df = pd.read_csv(convergence_csv).sort_values("end_step").reset_index(drop=True)
    final_conv = conv_df.iloc[-1]

    best_end_step = int(final_conv["best_end_step_so_far"])
    checked_end_step = int(final_conv["end_step"])
    best_mean_loss = float(final_conv["best_mean_loss_so_far"])
    final_checked_mean_loss = float(final_conv["mean_loss"])
    no_improve_windows = int(final_conv["no_improve_window_count"])
    early_best = best_end_step <= EARLY_BEST_THRESHOLD

    convergence_rows.append(
        {
            "run_index": int(ledger_row["run_index"]),
            "batch_name": batch_name,
            "animal": animal,
            "animal_label": animal_label,
            "best_end_step": best_end_step,
            "checked_end_step": checked_end_step,
            "best_mean_loss": best_mean_loss,
            "final_checked_mean_loss": final_checked_mean_loss,
            "loss_rebound_from_best": final_checked_mean_loss - best_mean_loss,
            "no_improve_windows": no_improve_windows,
            "n_windows": int(len(conv_df)),
            "early_best_le_2k": bool(early_best),
        }
    )

    for param_name, display_name, scale in PARAM_SPECS:
        rows = summary_df[summary_df["parameter"].astype(str) == param_name]
        if len(rows) != 1:
            raise RuntimeError(f"{summary_csv} has {len(rows)} rows for {param_name}")
        row = rows.iloc[0]
        values = {
            "mean": float(row["mean"]),
            "q025": float(row["q025"]),
            "q500": float(row["q500"]),
            "q975": float(row["q975"]),
        }
        if not all(np.isfinite(list(values.values()))):
            raise RuntimeError(f"Non-finite posterior summary for {animal_label}, {param_name}: {values}")

        param_rows.append(
            {
                "run_index": int(ledger_row["run_index"]),
                "batch_name": batch_name,
                "animal": animal,
                "animal_label": animal_label,
                "parameter": param_name,
                "display_parameter": display_name,
                "mean": values["mean"],
                "q025": values["q025"],
                "q500": values["q500"],
                "q975": values["q975"],
                "display_mean": values["mean"] * scale,
                "display_q025": values["q025"] * scale,
                "display_q500": values["q500"] * scale,
                "display_q975": values["q975"] * scale,
                "scale": scale,
                "best_end_step": best_end_step,
                "checked_end_step": checked_end_step,
                "early_best_le_2k": bool(early_best),
            }
        )

param_df = pd.DataFrame(param_rows)
convergence_df = pd.DataFrame(convergence_rows)

expected_param_rows = EXPECTED_N_ANIMALS * len(PARAM_SPECS)
if len(param_df) != expected_param_rows:
    raise RuntimeError(f"Expected {expected_param_rows} parameter rows, found {len(param_df)}.")

best_counts = convergence_df["best_end_step"].value_counts().sort_index()
n_1k = int(best_counts.get(1000, 0))
n_2k = int(best_counts.get(2000, 0))
n_early = int(convergence_df["early_best_le_2k"].sum())
if n_1k != 6 or n_2k != 3:
    raise RuntimeError(f"Expected 6 animals at 1k and 3 at 2k; found {n_1k} and {n_2k}.")

param_df.to_csv(PARAM_CSV, index=False)
convergence_df.to_csv(CONVERGENCE_CSV, index=False)

print(f"Loaded {len(ledger_df)} animals from {OUTPUT_ROOT}")
print("Restored-best checkpoint counts:")
print(best_counts.to_string())
print(f"Early-best <= {EARLY_BEST_THRESHOLD} steps: {n_early}")
print(f"Saved parameter CSV: {PARAM_CSV}")
print(f"Saved convergence CSV: {CONVERGENCE_CSV}")


# %%
# =============================================================================
# Plot 3 x 3 parameter grid
# =============================================================================
animal_labels = ledger_df.apply(lambda row: f"{row['batch_name']}/{int(row['animal'])}", axis=1).to_list()
x = np.arange(len(animal_labels), dtype=float)
early_by_label = convergence_df.set_index("animal_label")["early_best_le_2k"].to_dict()
point_colors = [EARLY_COLOR if early_by_label[label] else OTHER_COLOR for label in animal_labels]

fig, axes = plt.subplots(3, 3, figsize=(22, 13.5), sharex=True)
axes = axes.ravel()

for ax, (param_name, display_name, _scale) in zip(axes, PARAM_SPECS):
    sub = (
        param_df[param_df["parameter"] == param_name]
        .set_index("animal_label")
        .loc[animal_labels]
        .reset_index()
    )
    y = sub["display_mean"].to_numpy(dtype=float)
    low = sub["display_q025"].to_numpy(dtype=float)
    high = sub["display_q975"].to_numpy(dtype=float)
    yerr = np.vstack([y - low, high - y])

    for idx, label in enumerate(animal_labels):
        color = point_colors[idx]
        marker = "o" if early_by_label[label] else "."
        marker_size = 5.5 if early_by_label[label] else 5.0
        ax.errorbar(
            x[idx],
            y[idx],
            yerr=yerr[:, idx : idx + 1],
            fmt=marker,
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=2.2,
            markersize=marker_size,
            alpha=0.95 if early_by_label[label] else ERROR_ALPHA,
            linestyle="none",
        )

    ax.set_title(display_name)
    ax.set_ylabel(display_name)
    ax.grid(axis="y", alpha=0.22, linewidth=0.6)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.set_xticks(x)
    ax.set_xticklabels(animal_labels, rotation=55, ha="right", fontsize=7)

legend_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        color=EARLY_COLOR,
        linestyle="none",
        markersize=7,
        label=f"restored-best <= {EARLY_BEST_THRESHOLD // 1000}k ({n_early} animals)",
    ),
    Line2D(
        [0],
        [0],
        marker=".",
        color=OTHER_COLOR,
        linestyle="none",
        markersize=8,
        alpha=ERROR_ALPHA,
        label=f"restored-best > {EARLY_BEST_THRESHOLD // 1000}k",
    ),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=2, frameon=False, fontsize=11)
fig.suptitle(
    "Patience12 min-50k NPL+alpha+lapse SVI parameters by animal",
    fontsize=15,
    y=0.995,
)
fig.tight_layout(rect=[0, 0.055, 1, 0.965])
fig.savefig(FIG_PATH, dpi=200, bbox_inches="tight")

print(f"Saved figure: {FIG_PATH}")

# %%
