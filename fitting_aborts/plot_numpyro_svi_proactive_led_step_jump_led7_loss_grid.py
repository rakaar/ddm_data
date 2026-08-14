# %%
"""Plot the six completed LED7 proactive LED step-jump SVI loss traces."""

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
STANDARD_OUTPUT_ROOT = Path(
    os.environ.get(
        "PROACTIVE_LED_SVI_OUTPUT_ROOT",
        str(
            SCRIPT_DIR
            / "numpyro_svi_proactive_led_step_jump_bilateral_unweighted_"
            "patience12_min50k_restore_best_outputs"
        ),
    )
).expanduser()
EXTENDED_OUTPUT_ROOT = Path(
    os.environ.get(
        "PROACTIVE_LED_SVI_EXTENDED_OUTPUT_ROOT",
        str(
            SCRIPT_DIR
            / "numpyro_svi_proactive_led_step_jump_bilateral_unweighted_"
            "min150k_restore_best_audit_outputs"
        ),
    )
).expanduser()
EXTENDED_ANIMALS = (92, 100, 103)
SUMMARY_DIR = STANDARD_OUTPUT_ROOT / "summary_figures"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

FIG_PATH = SUMMARY_DIR / "led7_proactive_led_step_jump_svi_loss_grid.png"
SUMMARY_CSV = SUMMARY_DIR / "led7_proactive_led_step_jump_svi_loss_grid_summary.csv"
ANIMALS = (92, 93, 98, 99, 100, 103)

LOSS_COLOR = "#0072B2"
BEST_COLOR = "#009E73"
CHECKED_COLOR = "#D55E00"

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
plot_payload = []
summary_rows = []

for animal in ANIMALS:
    fit_root = (
        EXTENDED_OUTPUT_ROOT if animal in EXTENDED_ANIMALS else STANDARD_OUTPUT_ROOT
    )
    animal_dir = fit_root / f"LED7_{animal}"
    run_summary_path = animal_dir / "run_summary.json"
    convergence_path = animal_dir / "main_fullrank_convergence_checks.csv"
    posterior_summary_path = animal_dir / "main_fullrank_posterior_summary.csv"
    for required_path in [run_summary_path, convergence_path, posterior_summary_path]:
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    with run_summary_path.open("r", encoding="utf-8") as handle:
        run_summary = json.load(handle)
    convergence_df = pd.read_csv(convergence_path).sort_values("end_step")
    posterior_summary_df = pd.read_csv(posterior_summary_path)

    if run_summary.get("status") != "complete":
        raise RuntimeError(f"LED7/{animal} is not complete: {run_summary.get('status')}")
    if int(run_summary.get("n_nonfinite_losses", -1)) != 0:
        raise RuntimeError(f"LED7/{animal} has non-finite losses.")
    if not bool(run_summary.get("all_posterior_samples_finite", False)):
        raise RuntimeError(f"LED7/{animal} has non-finite posterior samples.")
    if posterior_summary_df["n_nonfinite"].astype(int).ne(0).any():
        raise RuntimeError(f"LED7/{animal} posterior summary contains non-finite draws.")

    window_steps = convergence_df["end_step"].to_numpy(dtype=int)
    window_losses = convergence_df["mean_loss"].to_numpy(dtype=float)
    if not np.isfinite(window_losses).all():
        raise RuntimeError(f"LED7/{animal} convergence curve contains non-finite values.")

    restored_best_step = int(run_summary["restored_best_step"])
    final_checked_step = int(run_summary["completed_steps"])
    best_row = convergence_df.loc[
        convergence_df["end_step"].astype(int).eq(restored_best_step)
    ]
    if len(best_row) != 1 or int(window_steps[-1]) != final_checked_step:
        raise RuntimeError(f"LED7/{animal} checkpoint metadata does not match its loss table.")

    best_loss = float(best_row.iloc[0]["mean_loss"])
    final_loss = float(window_losses[-1])
    no_improve_windows = int(convergence_df.iloc[-1]["no_improve_window_count"])
    y_low = float(np.min(window_losses))
    y_high = float(np.max(window_losses))
    y_pad = max(1.0, 0.06 * (y_high - y_low))

    plot_payload.append(
        {
            "animal": animal,
            "window_steps": window_steps,
            "window_losses": window_losses,
            "restored_best_step": restored_best_step,
            "final_checked_step": final_checked_step,
            "best_loss": best_loss,
            "y_limits": (y_low - y_pad, y_high + y_pad),
        }
    )
    summary_rows.append(
        {
            "batch_name": "LED7",
            "animal": animal,
            "fit_root": str(fit_root),
            "restored_best_step": restored_best_step,
            "final_checked_step": final_checked_step,
            "best_mean_negative_elbo": best_loss,
            "final_checked_mean_negative_elbo": final_loss,
            "final_minus_best": final_loss - best_loss,
            "final_no_improve_window_count": no_improve_windows,
            "fit_elapsed_minutes": float(run_summary["fit_elapsed_minutes"]),
            "all_losses_finite": True,
            "all_posterior_samples_finite": True,
        }
    )

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_CSV, index=False)
print("LED7 proactive LED step-jump SVI convergence summary:")
print(summary_df.to_string(index=False))
print(f"Saved summary: {SUMMARY_CSV}")


# %%
# =============================================================================
# Plot 2 x 3 loss grid
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.6))

for ax, payload in zip(axes.ravel(), plot_payload):
    ax.plot(
        payload["window_steps"],
        payload["window_losses"],
        color=LOSS_COLOR,
        lw=1.25,
        marker="o",
        markersize=2.2,
        zorder=3,
    )
    ax.axvline(
        payload["restored_best_step"],
        color=BEST_COLOR,
        lw=1.4,
        zorder=2,
    )
    ax.axvline(
        payload["final_checked_step"],
        color=CHECKED_COLOR,
        lw=1.4,
        ls="--",
        zorder=2,
    )
    ax.scatter(
        [payload["restored_best_step"]],
        [payload["best_loss"]],
        color=BEST_COLOR,
        s=24,
        zorder=4,
    )
    ax.set_xlim(0, payload["final_checked_step"] * 1.025)
    ax.set_ylim(*payload["y_limits"])
    ax.set_title(
        f"LED7/{payload['animal']}  "
        f"best={payload['restored_best_step'] / 1000:g}k, "
        f"checked={payload['final_checked_step'] / 1000:g}k",
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
        color=LOSS_COLOR,
        lw=1.4,
        marker="o",
        markersize=3,
        label="1k-window mean",
    ),
    Line2D([0], [0], color=BEST_COLOR, lw=1.5, label="restored-best checkpoint"),
    Line2D(
        [0],
        [0],
        color=CHECKED_COLOR,
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
    "LED7 proactive LED step-jump SVI loss traces\n"
    "150k audits for animals 92, 100, 103",
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
