# %%
"""
Plot the six completed LED7 reactive-only NPL+alpha SVI loss traces.

The blue curve is the 1k-window mean used by the patience12 restore-best rule.
Green marks the checkpoint restored for posterior sampling and red marks the
final checked step.
"""

# %%
# =============================================================================
# Parameters
# =============================================================================
from pathlib import Path
import json
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
        "NPL_REACTIVE_LED7_OUTPUT_ROOT",
        str(
            SCRIPT_DIR
            / "numpyro_svi_npl_alpha_reactive_only_rt_ge_100ms_condition_delay_"
            "patience12_min50k_restore_best_outputs"
        ),
    )
).expanduser()
LEDGER_CSV = OUTPUT_ROOT / "_batch_logs" / "batch_run_status.csv"
SUMMARY_DIR = OUTPUT_ROOT / "summary_figures"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

FIG_PATH = (
    SUMMARY_DIR
    / "led7_npl_alpha_reactive_ge100ms_patience12_loss_grid.png"
)
SUMMARY_CSV = (
    SUMMARY_DIR
    / "led7_npl_alpha_reactive_ge100ms_patience12_loss_grid_summary.csv"
)

EXPECTED_ANIMALS = (92, 93, 98, 99, 100, 103)
WINDOW_COLOR = "#0072B2"
BEST_COLOR = "#009E73"
STOP_COLOR = "#D55E00"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Helvetica",
            "Nimbus Sans",
            "Helvetica Neue",
            "Arial",
            "Liberation Sans",
            "sans-serif",
        ],
    }
)


# %%
# =============================================================================
# Load and validate completed fits
# =============================================================================
if not LEDGER_CSV.exists():
    raise FileNotFoundError(LEDGER_CSV)

ledger_df = pd.read_csv(LEDGER_CSV).sort_values("run_index").reset_index(drop=True)
animals = tuple(ledger_df["animal"].astype(int))
if animals != EXPECTED_ANIMALS:
    raise RuntimeError(
        f"Expected LED7 animals {EXPECTED_ANIMALS}, found {animals}."
    )
if not ledger_df["status"].eq("completed").all():
    raise RuntimeError(
        "Not all reactive-only fits are complete:\n"
        + ledger_df[["animal", "status"]].to_string(index=False)
    )

plot_payload = []
summary_rows = []
for ledger_row in ledger_df.itertuples(index=False):
    animal = int(ledger_row.animal)
    fit_dir = OUTPUT_ROOT / f"LED7_{animal}"
    loss_csv = fit_dir / "main_fullrank_loss.csv"
    convergence_csv = fit_dir / "main_fullrank_convergence_checks.csv"
    metadata_json = fit_dir / "main_fullrank_run_metadata.json"
    finite_csv = fit_dir / "main_fullrank_posterior_finite_report.csv"
    for required_path in [
        loss_csv,
        convergence_csv,
        metadata_json,
        finite_csv,
    ]:
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    loss_df = pd.read_csv(loss_csv).sort_values("step").reset_index(drop=True)
    convergence_df = (
        pd.read_csv(convergence_csv)
        .sort_values("end_step")
        .reset_index(drop=True)
    )
    finite_df = pd.read_csv(finite_csv)
    with metadata_json.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    raw_losses = loss_df["loss"].to_numpy(dtype=float)
    window_losses = convergence_df["mean_loss"].to_numpy(dtype=float)
    if not np.isfinite(raw_losses).all() or not np.isfinite(window_losses).all():
        raise RuntimeError(f"Non-finite loss values for LED7/{animal}.")
    if not (
        finite_df["n_total"].to_numpy(dtype=int)
        == finite_df["n_finite"].to_numpy(dtype=int)
    ).all():
        raise RuntimeError(f"Non-finite posterior samples for LED7/{animal}.")

    final_row = convergence_df.iloc[-1]
    best_step = int(final_row["best_end_step_so_far"])
    checked_step = int(final_row["end_step"])
    best_loss = float(final_row["best_mean_loss_so_far"])
    final_loss = float(final_row["mean_loss"])
    if best_step != int(metadata["best_step"]):
        raise RuntimeError(f"Best-step metadata mismatch for LED7/{animal}.")
    if checked_step != int(metadata["final_checked_step"]):
        raise RuntimeError(f"Checked-step metadata mismatch for LED7/{animal}.")

    y_low = min(np.min(window_losses), best_loss)
    y_high = max(np.max(window_losses), best_loss)
    y_pad = max(1.0, 0.06 * (y_high - y_low))

    plot_payload.append(
        {
            "animal": animal,
            "window_steps": convergence_df["end_step"].to_numpy(dtype=float),
            "window_losses": window_losses,
            "best_step": best_step,
            "checked_step": checked_step,
            "best_loss": best_loss,
            "final_loss": final_loss,
            "y_limits": (y_low - y_pad, y_high + y_pad),
        }
    )
    summary_rows.append(
        {
            "batch_name": "LED7",
            "animal": animal,
            "n_retained_trials": int(metadata["n_retained_trials"]),
            "best_end_step": best_step,
            "checked_end_step": checked_step,
            "best_mean_negative_elbo": best_loss,
            "final_checked_mean_negative_elbo": final_loss,
            "final_minus_best": final_loss - best_loss,
            "n_windows": len(convergence_df),
            "all_losses_finite": True,
            "all_posterior_samples_finite": True,
        }
    )

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_CSV, index=False)

print("Reactive-only LED7 SVI convergence summary:")
print(
    summary_df[
        [
            "animal",
            "best_end_step",
            "checked_end_step",
            "best_mean_negative_elbo",
            "final_checked_mean_negative_elbo",
            "final_minus_best",
        ]
    ].to_string(index=False)
)
print(f"Saved summary: {SUMMARY_CSV}")


# %%
# =============================================================================
# Plot 2 x 3 loss grid
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.6))
axes = axes.ravel()

for ax, payload in zip(axes, plot_payload):
    ax.plot(
        payload["window_steps"],
        payload["window_losses"],
        color=WINDOW_COLOR,
        lw=1.25,
        marker="o",
        markersize=2.2,
        zorder=3,
    )
    ax.axvline(
        payload["best_step"],
        color=BEST_COLOR,
        lw=1.4,
        zorder=2,
    )
    ax.axvline(
        payload["checked_step"],
        color=STOP_COLOR,
        lw=1.4,
        ls="--",
        zorder=2,
    )
    ax.scatter(
        [payload["best_step"]],
        [payload["best_loss"]],
        color=BEST_COLOR,
        s=24,
        zorder=4,
    )
    ax.set_xlim(0, payload["checked_step"])
    ax.set_ylim(*payload["y_limits"])
    ax.set_title(
        f"LED7/{payload['animal']}  "
        f"best={payload['best_step'] / 1000:g}k, "
        f"checked={payload['checked_step'] / 1000:g}k",
        fontsize=10,
    )
    ax.set_xlabel("SVI step")
    ax.set_ylabel("negative ELBO")
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(alpha=0.18, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

legend_handles = [
    Line2D(
        [0],
        [0],
        color=WINDOW_COLOR,
        lw=1.4,
        marker="o",
        markersize=3,
        label="1k-window mean",
    ),
    Line2D(
        [0],
        [0],
        color=BEST_COLOR,
        lw=1.5,
        label="restored-best checkpoint",
    ),
    Line2D(
        [0],
        [0],
        color=STOP_COLOR,
        lw=1.5,
        ls="--",
        label="final checked step",
    ),
]
fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.955),
    ncol=3,
    frameon=False,
    fontsize=9,
)
fig.suptitle(
    "LED7 reactive-only NPL+alpha SVI loss traces "
    "(valid RT 100-1000 ms)",
    y=0.995,
    fontsize=13,
)
fig.subplots_adjust(
    left=0.075,
    right=0.985,
    bottom=0.075,
    top=0.88,
    wspace=0.25,
    hspace=0.34,
)
fig.savefig(FIG_PATH, dpi=250, bbox_inches="tight")
print(f"Saved figure: {FIG_PATH}")

# %%
