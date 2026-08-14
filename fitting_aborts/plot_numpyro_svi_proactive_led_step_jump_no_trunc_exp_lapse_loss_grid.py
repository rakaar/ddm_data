# %%
"""Plot convergence traces for the six LED7 no-truncation lapse SVI fits."""

# %%
from pathlib import Path
import json
import os

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
FIT_ROOT = (
    SCRIPT_DIR
    / "numpyro_svi_proactive_led_step_jump_all_on_no_trunc_exp_lapse_"
    "patience12_min50k_restore_best_outputs"
)
ANIMALS = (92, 93, 98, 99, 100, 103)
SUMMARY_DIR = FIT_ROOT / "summary_figures"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
FIG_PATH = SUMMARY_DIR / "led7_no_trunc_exp_lapse_svi_loss_grid.png"
SUMMARY_CSV = SUMMARY_DIR / "led7_no_trunc_exp_lapse_svi_convergence_summary.csv"


# %%
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# %%
# =============================================================================
# Load and validate all completed summaries
# =============================================================================
records = []
for animal in ANIMALS:
    fit_dir = FIT_ROOT / f"LED7_{animal}"
    summary_path = fit_dir / "run_summary.json"
    convergence_path = fit_dir / "main_fullrank_convergence_checks.csv"
    if not summary_path.exists() or not convergence_path.exists():
        raise FileNotFoundError(f"Missing fit artifacts for LED7/{animal}: {fit_dir}")
    summary = json.loads(summary_path.read_text())
    if summary.get("status") not in {"complete", "max_steps_restore_best"}:
        raise RuntimeError(f"LED7/{animal} has status {summary.get('status')}.")
    if summary["config"].get("truncation") != "none":
        raise RuntimeError(f"LED7/{animal} is not a no-truncation fit.")
    convergence = pd.read_csv(convergence_path)
    if convergence.empty or not np.isfinite(convergence["mean_loss"]).all():
        raise RuntimeError(f"LED7/{animal} has invalid convergence windows.")
    records.append(
        {
            "animal": animal,
            "summary": summary,
            "convergence": convergence,
            "fit_dir": fit_dir,
        }
    )


# %%
# =============================================================================
# Plot
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.4), constrained_layout=True)
summary_rows = []

for ax, record in zip(axes.flat, records):
    animal = record["animal"]
    summary = record["summary"]
    convergence = record["convergence"]
    best_step = int(summary["restored_best_step"])
    checked_step = int(summary["completed_steps"])

    ax.plot(
        convergence["end_step"],
        convergence["mean_loss"],
        color="tab:blue",
        lw=1.0,
    )
    ax.axvline(best_step, color="tab:green", lw=1.2)
    ax.axvline(checked_step, color="tab:red", ls="--", lw=1.2)
    ax.set_title(
        f"LED7/{animal} | best {best_step // 1000}k, checked {checked_step // 1000}k",
        fontsize=9,
    )
    ax.set_xlabel("SVI step")
    ax.set_ylabel("negative ELBO")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(3, 3))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.15, lw=0.5)

    summary_rows.append(
        {
            "animal": animal,
            "status": summary["status"],
            "stop_reason": summary["stop_reason"],
            "restored_best_step": best_step,
            "completed_steps": checked_step,
            "best_mean_negative_elbo": float(
                convergence.iloc[-1]["best_mean_loss_so_far"]
            ),
            "final_no_improve_windows": int(
                convergence.iloc[-1]["no_improve_window_count"]
            ),
            "fit_elapsed_minutes": float(summary["fit_elapsed_minutes"]),
            "early_aborts_below_300ms_retained": int(
                summary["trial_counts"]["early_aborts_below_300ms_retained"]
            ),
        }
    )

fig.legend(
    handles=[
        Line2D([0], [0], color="tab:blue", lw=1.0, label="1k-window mean"),
        Line2D([0], [0], color="tab:green", lw=1.2, label="restored best"),
        Line2D(
            [0],
            [0],
            color="tab:red",
            ls="--",
            lw=1.2,
            label="final checked",
        ),
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, 1.035),
    ncol=3,
    frameon=False,
    fontsize=9,
)
fig.suptitle(
    "LED7 all-ON no-truncation exponential-lapse proactive SVI",
    fontsize=12,
    y=1.075,
)
fig.savefig(FIG_PATH, dpi=220, bbox_inches="tight")
pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False)

print(f"Saved figure: {FIG_PATH}")
print(f"Saved summary: {SUMMARY_CSV}")
