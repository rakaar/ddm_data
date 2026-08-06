# %%
"""
Plot focused loss grids for the two completed LED7 RT-only NPL+alpha fits.

Each panel stops at the restored-best checkpoint plus 5,000 SVI steps, capped
at the final checked step when the run did not continue that far. This keeps
late loss rebound from compressing the initial descent.
"""

# %%
# =============================================================================
# Editable parameters
# =============================================================================
from pathlib import Path
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

SCRIPT_DIR = Path(__file__).resolve().parent
EXPECTED_ANIMALS = (92, 93, 98, 99, 100, 103)
DISPLAY_STEPS_AFTER_BEST = 5_000
PLOT_DPI = 250

FIT_CONFIGS = (
    {
        "mode": "proactive_reactive",
        "label": "proactive + reactive",
        "root": SCRIPT_DIR
        / (
            "numpyro_svi_npl_alpha_rt_only_proactive_reactive_0_to_1s_"
            "condition_delay_patience12_min50k_restore_best_outputs"
        ),
        "figure_name": (
            "led7_npl_alpha_rt_only_proactive_reactive_"
            "loss_grid_best_plus_5k.png"
        ),
    },
    {
        "mode": "reactive_only",
        "label": "reactive only",
        "root": SCRIPT_DIR
        / (
            "numpyro_svi_npl_alpha_rt_only_reactive_only_0_to_1s_"
            "condition_delay_patience12_min50k_restore_best_outputs"
        ),
        "figure_name": (
            "led7_npl_alpha_rt_only_reactive_only_"
            "loss_grid_best_plus_5k.png"
        ),
    },
)


# %%
# =============================================================================
# Imports and plotting defaults
# =============================================================================
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

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
        "axes.linewidth": 0.8,
    }
)


# %%
# =============================================================================
# Load, validate, and plot each RT-only process model
# =============================================================================
all_summary_rows = []

for fit_config in FIT_CONFIGS:
    output_root = fit_config["root"]
    ledger_csv = output_root / "_batch_logs" / "batch_run_status.csv"
    summary_dir = output_root / "summary_figures"
    summary_dir.mkdir(parents=True, exist_ok=True)
    figure_path = summary_dir / fit_config["figure_name"]

    if not ledger_csv.exists():
        raise FileNotFoundError(ledger_csv)

    ledger_df = (
        pd.read_csv(ledger_csv)
        .sort_values("run_index")
        .reset_index(drop=True)
    )
    animals = tuple(ledger_df["animal"].astype(int))
    if animals != EXPECTED_ANIMALS:
        raise RuntimeError(
            f"Expected LED7 animals {EXPECTED_ANIMALS}, found {animals} "
            f"for {fit_config['mode']}."
        )
    if not ledger_df["status"].eq("completed").all():
        raise RuntimeError(
            f"Incomplete {fit_config['mode']} fits:\n"
            + ledger_df[["animal", "status"]].to_string(index=False)
        )

    plot_payload = []
    for ledger_row in ledger_df.itertuples(index=False):
        animal = int(ledger_row.animal)
        fit_dir = output_root / f"LED7_{animal}"
        loss_csv = fit_dir / "main_fullrank_loss.csv"
        convergence_csv = fit_dir / "main_fullrank_convergence_checks.csv"
        metadata_json = fit_dir / "main_fullrank_run_metadata.json"
        finite_csv = fit_dir / "main_fullrank_posterior_finite_report.csv"

        for required_path in (
            loss_csv,
            convergence_csv,
            metadata_json,
            finite_csv,
        ):
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
        window_steps = convergence_df["end_step"].to_numpy(dtype=int)
        window_losses = convergence_df["mean_loss"].to_numpy(dtype=float)
        if not np.isfinite(raw_losses).all() or not np.isfinite(window_losses).all():
            raise RuntimeError(
                f"Non-finite loss values for {fit_config['mode']} LED7/{animal}."
            )
        if not (
            finite_df["n_total"].to_numpy(dtype=int)
            == finite_df["n_finite"].to_numpy(dtype=int)
        ).all():
            raise RuntimeError(
                f"Non-finite posterior samples for {fit_config['mode']} "
                f"LED7/{animal}."
            )

        final_row = convergence_df.iloc[-1]
        best_step = int(metadata["best_step"])
        checked_step = int(metadata["final_checked_step"])
        if best_step != int(final_row["best_end_step_so_far"]):
            raise RuntimeError(
                f"Best-step mismatch for {fit_config['mode']} LED7/{animal}."
            )
        if checked_step != int(final_row["end_step"]):
            raise RuntimeError(
                f"Checked-step mismatch for {fit_config['mode']} LED7/{animal}."
            )

        display_end_step = min(
            checked_step,
            best_step + DISPLAY_STEPS_AFTER_BEST,
        )
        display_mask = window_steps <= display_end_step
        display_steps = window_steps[display_mask]
        display_losses = window_losses[display_mask]
        if len(display_steps) == 0 or best_step not in display_steps:
            raise RuntimeError(
                f"Focused display does not contain the best checkpoint for "
                f"{fit_config['mode']} LED7/{animal}."
            )

        best_loss = float(
            convergence_df.loc[
                convergence_df["end_step"].eq(best_step), "mean_loss"
            ].iloc[0]
        )
        y_low = float(np.min(display_losses))
        y_high = float(np.max(display_losses))
        y_pad = max(1.0, 0.07 * (y_high - y_low))

        plot_payload.append(
            {
                "animal": animal,
                "steps": display_steps,
                "losses": display_losses,
                "best_step": best_step,
                "best_loss": best_loss,
                "checked_step": checked_step,
                "display_end_step": display_end_step,
                "y_limits": (y_low - y_pad, y_high + y_pad),
            }
        )
        all_summary_rows.append(
            {
                "process_mode": fit_config["mode"],
                "animal": animal,
                "best_end_step": best_step,
                "final_checked_step": checked_step,
                "display_end_step": display_end_step,
                "displayed_steps_after_best": display_end_step - best_step,
                "best_window_mean_negative_elbo": best_loss,
            }
        )

    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.6))
    axes = axes.ravel()

    for ax, payload in zip(axes, plot_payload):
        ax.plot(
            payload["steps"],
            payload["losses"],
            color=LOSS_COLOR,
            lw=1.25,
            marker="o",
            markersize=2.3,
            zorder=3,
        )
        ax.axvline(
            payload["best_step"],
            color=BEST_COLOR,
            lw=1.5,
            zorder=2,
        )
        ax.scatter(
            [payload["best_step"]],
            [payload["best_loss"]],
            color=BEST_COLOR,
            s=25,
            zorder=4,
        )
        if payload["checked_step"] <= payload["display_end_step"]:
            ax.axvline(
                payload["checked_step"],
                color=CHECKED_COLOR,
                lw=1.3,
                ls="--",
                zorder=2,
            )

        ax.set_xlim(0, payload["display_end_step"])
        ax.set_ylim(*payload["y_limits"])
        ax.set_title(
            f"LED7/{payload['animal']}  "
            f"best={payload['best_step'] / 1000:g}k, "
            f"shown={payload['display_end_step'] / 1000:g}k, "
            f"checked={payload['checked_step'] / 1000:g}k",
            fontsize=9.5,
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
            color=CHECKED_COLOR,
            lw=1.3,
            ls="--",
            label="final checked step (when visible)",
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
        f"LED7 RT-only NPL+alpha SVI: {fit_config['label']} "
        f"(restored best + {DISPLAY_STEPS_AFTER_BEST // 1000}k view)",
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
    fig.savefig(figure_path, dpi=PLOT_DPI, bbox_inches="tight")

    print(f"\n{fit_config['label']} convergence view:")
    print(
        pd.DataFrame(
            [
                row
                for row in all_summary_rows
                if row["process_mode"] == fit_config["mode"]
            ]
        )[
            [
                "animal",
                "best_end_step",
                "display_end_step",
                "final_checked_step",
                "displayed_steps_after_best",
            ]
        ].to_string(index=False)
    )
    print(f"Saved figure: {figure_path}")


# %%
# =============================================================================
# Save one compact audit table beside the script
# =============================================================================
SUMMARY_CSV = SCRIPT_DIR / "npl_alpha_rt_only_led7_loss_grid_best_plus_5k_summary.csv"
pd.DataFrame(all_summary_rows).to_csv(SUMMARY_CSV, index=False)
print(f"\nSaved combined summary: {SUMMARY_CSV}")

# %%
