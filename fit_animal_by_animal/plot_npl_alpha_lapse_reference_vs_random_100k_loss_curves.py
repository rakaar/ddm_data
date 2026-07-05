# %%
"""
Compare early-best reference-init and random-init 100k NPL+alpha+lapse SVI ELBO curves.

Rows are the four suspicious animals. Columns compare the low-learning-rate
reference-initialized rerun, whose restored-best checkpoint was at 1k/2k, against
the random-plausible low-learning-rate rerun that was forced to 100k steps.
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
REFERENCE_LOW_LR_ROOT = SCRIPT_DIR / "numpyro_svi_npl_alpha_lapse_condition_delay_low_lr_patience12_min12k_restore_best_reruns"
RANDOM_100K_ROOT = SCRIPT_DIR / "numpyro_svi_npl_alpha_lapse_condition_delay_random_plausible_low_lr_100k_earlybest_reruns"

ANIMALS = [
    ("LED7", 98),
    ("LED7", 100),
    ("LED34_even", 52),
    ("LED34_even", 60),
]

SUMMARY_DIR = RANDOM_100K_ROOT / "summary_figures"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

FIG_PATH = SUMMARY_DIR / "npl_alpha_lapse_reference_vs_random_100k_elbo_curves.png"
SUMMARY_CSV = SUMMARY_DIR / "npl_alpha_lapse_reference_vs_random_100k_elbo_summary.csv"

LOSS_COLOR = "0.12"
BEST_COLOR = "tab:green"
CHECKED_COLOR = "tab:red"


# %%
# =============================================================================
# Load convergence windows
# =============================================================================
payloads = []
for batch_name, animal in ANIMALS:
    animal_dir = f"{batch_name}_{animal}"
    for source, root in [
        ("reference init", REFERENCE_LOW_LR_ROOT),
        ("random init 100k", RANDOM_100K_ROOT),
    ]:
        convergence_csv = root / animal_dir / "main_fullrank_convergence_checks.csv"
        if not convergence_csv.exists():
            raise FileNotFoundError(convergence_csv)

        conv_df = pd.read_csv(convergence_csv).sort_values("end_step").reset_index(drop=True)
        final_row = conv_df.iloc[-1]
        payloads.append(
            {
                "batch_name": batch_name,
                "animal": animal,
                "label": f"{batch_name}/{animal}",
                "source": source,
                "steps": conv_df["end_step"].to_numpy(dtype=float),
                "mean_loss": conv_df["mean_loss"].to_numpy(dtype=float),
                "best_step": int(final_row["best_end_step_so_far"]),
                "checked_step": int(final_row["end_step"]),
                "best_mean_loss": float(final_row["best_mean_loss_so_far"]),
                "final_mean_loss": float(final_row["mean_loss"]),
                "no_improve_windows": int(final_row["no_improve_window_count"]),
                "n_windows": len(conv_df),
            }
        )

summary_rows = []
for payload in payloads:
    summary_rows.append(
        {
            "batch_name": payload["batch_name"],
            "animal": payload["animal"],
            "source": payload["source"],
            "best_step": payload["best_step"],
            "checked_step": payload["checked_step"],
            "best_mean_negative_elbo": payload["best_mean_loss"],
            "final_mean_negative_elbo": payload["final_mean_loss"],
            "final_minus_best": payload["final_mean_loss"] - payload["best_mean_loss"],
            "no_improve_windows": payload["no_improve_windows"],
            "n_windows": payload["n_windows"],
        }
    )

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_CSV, index=False)
print(summary_df.to_string(index=False))
print(f"Saved summary CSV: {SUMMARY_CSV}")


# %%
# =============================================================================
# Plot 4 x 2 comparison
# =============================================================================
fig, axes = plt.subplots(4, 2, figsize=(11.5, 12.0), sharex=False, sharey=False)

payload_by_key = {(p["batch_name"], p["animal"], p["source"]): p for p in payloads}

for row_idx, (batch_name, animal) in enumerate(ANIMALS):
    for col_idx, source in enumerate(["reference init", "random init 100k"]):
        ax = axes[row_idx, col_idx]
        payload = payload_by_key[(batch_name, animal, source)]

        ax.plot(payload["steps"], payload["mean_loss"], color=LOSS_COLOR, lw=1.1)
        ax.axvline(payload["best_step"], color=BEST_COLOR, lw=1.4)
        ax.axvline(payload["checked_step"], color=CHECKED_COLOR, lw=1.3, ls="--")
        ax.scatter([payload["best_step"]], [payload["best_mean_loss"]], s=24, color=BEST_COLOR, zorder=3)

        title = (
            f"{payload['label']} ELBO - {source}\n"
            f"best {payload['best_step'] / 1000:.0f}k, checked {payload['checked_step'] / 1000:.0f}k, "
            f"chosen -ELBO={payload['best_mean_loss']:.2f}"
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("SVI step")
        ax.set_ylabel("1k-window mean -ELBO")
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.grid(alpha=0.18, lw=0.5)

        if source == "random init 100k":
            ax.set_xlim(0, 100000)

handles = [
    Line2D([0], [0], color=LOSS_COLOR, lw=1.4, label="1k-window mean -ELBO"),
    Line2D([0], [0], color=BEST_COLOR, lw=1.6, label="restored-best checkpoint"),
    Line2D([0], [0], color=CHECKED_COLOR, lw=1.6, ls="--", label="final checked step"),
]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10)
fig.suptitle("ELBO comparison: early-best reference init vs random-init 100k NPL+alpha+lapse SVI", y=0.995, fontsize=14)
fig.tight_layout(rect=[0, 0.04, 1, 0.97])
fig.savefig(FIG_PATH, dpi=200, bbox_inches="tight")
print(f"Saved figure: {FIG_PATH}")

# %%
