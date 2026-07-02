# %%
"""
Plot all-animal loss curves for the patience12 restore-best vanilla/IPL SVI run.

Each subplot shows the 1k-window mean negative ELBO used by the stopping rule.
The green vertical line is the restored-best checkpoint used for posterior
sampling, and the red dashed line is the final step checked before stopping.
"""

# %%
# =============================================================================
# Parameters
# =============================================================================
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = Path(
    os.environ.get(
        "NUMPYRO_VANILLA_SVI_PATIENCE12_OUTPUT_ROOT",
        str(SCRIPT_DIR / "numpyro_svi_vanilla_condition_delay_patience12_restore_best_outputs"),
    )
).expanduser()
LOG_DIR = OUTPUT_ROOT / "_batch_logs"
LEDGER_CSV = LOG_DIR / "batch_run_status.csv"
SUMMARY_DIR = OUTPUT_ROOT / "summary_figures"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

FIG_PATH = SUMMARY_DIR / "vanilla_condition_delay_patience12_loss_grid.png"
SUMMARY_CSV = SUMMARY_DIR / "vanilla_condition_delay_patience12_loss_grid_summary.csv"

N_ROWS = 5
N_COLS = 6
LOSS_COLOR = "0.15"
BEST_COLOR = "tab:green"
STOP_COLOR = "tab:red"


# %%
# =============================================================================
# Load convergence summaries
# =============================================================================
if not LEDGER_CSV.exists():
    raise FileNotFoundError(LEDGER_CSV)

ledger_df = pd.read_csv(LEDGER_CSV)
if len(ledger_df) != N_ROWS * N_COLS:
    raise RuntimeError(f"Expected {N_ROWS * N_COLS} animals in {LEDGER_CSV}, found {len(ledger_df)}.")

accepted_statuses = {"completed", "skipped_existing"}
bad_status = ledger_df[~ledger_df["status"].isin(accepted_statuses)]
if not bad_status.empty:
    raise RuntimeError("Some animals are not completed or skipped-existing:\n" + bad_status.to_string(index=False))

plot_payload = []
summary_rows = []

for _, ledger_row in ledger_df.sort_values("run_index").iterrows():
    batch = str(ledger_row["batch_name"])
    animal = int(ledger_row["animal"])
    label = f"{batch}/{animal}"
    output_dir = OUTPUT_ROOT / f"{batch}_{animal}"
    convergence_csv = output_dir / "main_fullrank_convergence_checks.csv"

    if not convergence_csv.exists():
        raise FileNotFoundError(convergence_csv)

    conv_df = pd.read_csv(convergence_csv).sort_values("end_step").reset_index(drop=True)
    final_row = conv_df.iloc[-1]
    checked_step = int(final_row["end_step"])
    best_step = int(final_row["best_end_step_so_far"])
    best_loss = float(final_row["best_mean_loss_so_far"])
    final_loss = float(final_row["mean_loss"])
    no_improve_windows = int(final_row["no_improve_window_count"])

    if int(final_row.get("n_nonfinite", 0)) > 0:
        stop_reason = "nonfinite_loss"
    elif bool(final_row.get("no_improve_stop_candidate", False)):
        stop_reason = "patience_restore_best_12_windows"
    elif bool(final_row.get("early_stop_candidate", False)):
        stop_reason = "stable_12_windows"
    else:
        stop_reason = "max_steps_or_manual_stop"

    plot_payload.append(
        {
            "label": label,
            "steps": conv_df["end_step"].to_numpy(dtype=float),
            "mean_loss": conv_df["mean_loss"].to_numpy(dtype=float),
            "best_step": best_step,
            "checked_step": checked_step,
            "best_loss": best_loss,
            "final_loss": final_loss,
            "no_improve_windows": no_improve_windows,
            "stop_reason": stop_reason,
        }
    )
    summary_rows.append(
        {
            "run_index": int(ledger_row["run_index"]),
            "batch_name": batch,
            "animal": animal,
            "stop_reason": stop_reason,
            "best_end_step": best_step,
            "checked_end_step": checked_step,
            "best_mean_loss": best_loss,
            "final_checked_mean_loss": final_loss,
            "loss_rebound_from_best": final_loss - best_loss,
            "no_improve_windows": no_improve_windows,
            "n_windows": len(conv_df),
        }
    )

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_CSV, index=False)

print("Loaded completed patience12 vanilla/IPL condition-delay SVI runs:")
print(summary_df[["batch_name", "animal", "best_end_step", "checked_end_step", "stop_reason"]].to_string(index=False))
print("\nStep summary:")
print(
    summary_df[["best_end_step", "checked_end_step", "loss_rebound_from_best", "n_windows"]]
    .describe()
    .to_string()
)
print(f"\nSaved summary CSV: {SUMMARY_CSV}")


# %%
# =============================================================================
# Plot 5 x 6 loss grid
# =============================================================================
fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(18, 12.5), sharex=False, sharey=False)
axes = axes.ravel()

for ax, payload in zip(axes, plot_payload):
    ax.plot(payload["steps"], payload["mean_loss"], color=LOSS_COLOR, lw=1.1)
    ax.axvline(payload["best_step"], color=BEST_COLOR, lw=1.4)
    ax.axvline(payload["checked_step"], color=STOP_COLOR, lw=1.4, ls="--")
    ax.scatter([payload["best_step"]], [payload["best_loss"]], s=18, color=BEST_COLOR, zorder=3)
    title_lines = [
        payload["label"],
        f"min {payload['best_step'] / 1000:.0f}k, checked {payload['checked_step'] / 1000:.0f}k",
    ]
    if payload["best_step"] == payload["checked_step"]:
        title_lines.append(f"no >0.1%: {payload['no_improve_windows']}w")
    ax.set_title("\n".join(title_lines), fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(alpha=0.18, lw=0.5)
    ax.set_xlabel("SVI step", fontsize=7)
    ax.set_ylabel("mean -ELBO", fontsize=7)

for ax in axes[len(plot_payload) :]:
    ax.axis("off")

legend_handles = [
    Line2D([0], [0], color=LOSS_COLOR, lw=1.4, label="1k-window mean -ELBO"),
    Line2D([0], [0], color=BEST_COLOR, lw=1.6, label="restored-best checkpoint"),
    Line2D([0], [0], color=STOP_COLOR, lw=1.6, ls="--", label="final checked step"),
]
fig.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.005),
    ncol=3,
    frameon=False,
    fontsize=10,
)
fig.suptitle(
    "Patience12 restore-best vanilla/IPL condition-delay SVI loss curves",
    y=0.995,
    fontsize=14,
)
fig.tight_layout(rect=[0, 0.04, 1, 0.965])
fig.savefig(FIG_PATH, dpi=200, bbox_inches="tight")

print(f"Saved figure: {FIG_PATH}")

# %%
