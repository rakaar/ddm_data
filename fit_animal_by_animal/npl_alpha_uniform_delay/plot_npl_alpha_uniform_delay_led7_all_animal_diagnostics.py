# %%
"""Across-animal diagnostics for the six LED7 uniform-delay SVI fits."""

# %%
# =============================================================================
# Editable parameters
# =============================================================================
from pathlib import Path
import json
import os

SCRIPT_DIR = Path(__file__).resolve().parent
FIT_ROOT = Path(
    os.environ.get(
        "NUMPYRO_SVI_OUTPUT_ROOT",
        str(
            SCRIPT_DIR
            / (
                "numpyro_svi_npl_alpha_uniform_delay_rt_choice_"
                "patience12_min50k_restore_best_outputs"
            )
        ),
    )
).expanduser()
ANIMALS = (92, 93, 98, 99, 100, 103)
ABLS = (20, 40, 60)
SIGNED_ILDS = (-16.0, -8.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0, 16.0)
ABL_COLORS = {
    20: "#0072B2",
    40: "#E69F00",
    60: "#009E73",
}

SUMMARY_DIR = FIT_ROOT / "summary_figures"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

LOSS_FIG = SUMMARY_DIR / "led7_npl_alpha_uniform_delay_loss_curves_1x6.png"
LOSS_SUMMARY_CSV = (
    SUMMARY_DIR / "led7_npl_alpha_uniform_delay_loss_curve_summary.csv"
)
ANIMAL_SUPPORT_FIG = (
    SUMMARY_DIR
    / "led7_npl_alpha_uniform_delay_center_support_by_animal_1x6.png"
)
AVERAGE_SUPPORT_FIG = (
    SUMMARY_DIR
    / "led7_npl_alpha_uniform_delay_center_support_across_animals.png"
)
AVERAGE_SUPPORT_CSV = (
    SUMMARY_DIR
    / "led7_npl_alpha_uniform_delay_center_support_across_animals.csv"
)

WINDOW_COLOR = "#0072B2"
BEST_COLOR = "#009E73"
STOP_COLOR = "#D55E00"
PLOT_DPI = 300


# %%
# =============================================================================
# Imports and plotting style
# =============================================================================
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

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
# Load and validate all six completed fits
# =============================================================================
fit_payloads = []
loss_summary_rows = []
condition_frames = []

for animal in ANIMALS:
    fit_dir = FIT_ROOT / f"LED7_{animal}"
    loss_csv = fit_dir / "main_fullrank_loss.csv"
    convergence_csv = fit_dir / "main_fullrank_convergence_checks.csv"
    metadata_json = fit_dir / "main_fullrank_run_metadata.json"
    posterior_npz = fit_dir / "main_fullrank_posterior_samples.npz"
    condition_csv = fit_dir / "condition_delay_summary.csv"

    for required_path in (
        loss_csv,
        convergence_csv,
        metadata_json,
        posterior_npz,
        condition_csv,
    ):
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    loss_df = pd.read_csv(loss_csv).sort_values("step").reset_index(drop=True)
    convergence_df = (
        pd.read_csv(convergence_csv)
        .sort_values("end_step")
        .reset_index(drop=True)
    )
    condition_df = (
        pd.read_csv(condition_csv)
        .sort_values(["ABL", "ILD"])
        .reset_index(drop=True)
    )
    with metadata_json.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    if len(condition_df) != 30:
        raise RuntimeError(
            f"Expected 30 conditions for LED7/{animal}, found {len(condition_df)}."
        )
    observed_grid = set(
        zip(condition_df["ABL"].astype(int), condition_df["ILD"].astype(float))
    )
    expected_grid = {(abl, ild) for abl in ABLS for ild in SIGNED_ILDS}
    if observed_grid != expected_grid:
        raise RuntimeError(f"Unexpected ABL/ILD grid for LED7/{animal}.")

    raw_losses = loss_df["negative_elbo"].to_numpy(dtype=float)
    window_losses = convergence_df["mean_loss"].to_numpy(dtype=float)
    if not np.isfinite(raw_losses).all() or not np.isfinite(window_losses).all():
        raise RuntimeError(f"Non-finite loss values for LED7/{animal}.")
    if convergence_df["n_nonfinite"].astype(int).sum() != 0:
        raise RuntimeError(f"A convergence window is non-finite for LED7/{animal}.")

    with np.load(posterior_npz) as posterior:
        posterior_is_finite = all(
            np.isfinite(np.asarray(posterior[key])).all() for key in posterior.files
        )
    if not posterior_is_finite:
        raise RuntimeError(f"Non-finite posterior samples for LED7/{animal}.")

    stopping = metadata["stopping"]
    best_step = int(stopping["best_step"])
    checked_step = int(stopping["final_checked_step"])
    best_loss = float(stopping["best_window_mean_loss"])
    first_loss = float(window_losses[0])
    final_loss = float(window_losses[-1])
    if best_step != int(convergence_df.iloc[-1]["best_end_step_so_far"]):
        raise RuntimeError(f"Best-step mismatch for LED7/{animal}.")
    if checked_step != int(convergence_df.iloc[-1]["end_step"]):
        raise RuntimeError(f"Final-step mismatch for LED7/{animal}.")
    if not np.isclose(
        best_loss,
        float(convergence_df.iloc[-1]["best_mean_loss_so_far"]),
        rtol=0,
        atol=1e-8,
    ):
        raise RuntimeError(f"Best-loss mismatch for LED7/{animal}.")

    initial_to_best_drop = first_loss - best_loss
    final_rebound = final_loss - best_loss
    if initial_to_best_drop <= 0:
        raise RuntimeError(
            f"LED7/{animal} never improved beyond its first loss window."
        )

    first_window_count = min(10, len(window_losses))
    last_window_count = min(10, len(window_losses))
    first_slope = float(
        np.polyfit(
            convergence_df["end_step"].to_numpy(dtype=float)[:first_window_count],
            window_losses[:first_window_count],
            deg=1,
        )[0]
        * 1000.0
    )
    last_slope = float(
        np.polyfit(
            convergence_df["end_step"].to_numpy(dtype=float)[-last_window_count:],
            window_losses[-last_window_count:],
            deg=1,
        )[0]
        * 1000.0
    )

    fit_payloads.append(
        {
            "animal": animal,
            "steps": convergence_df["end_step"].to_numpy(dtype=float),
            "window_losses": window_losses,
            "best_step": best_step,
            "checked_step": checked_step,
            "best_loss": best_loss,
            "condition_df": condition_df,
        }
    )
    loss_summary_rows.append(
        {
            "batch_name": "LED7",
            "animal": animal,
            "n_windows": len(convergence_df),
            "best_step": best_step,
            "final_checked_step": checked_step,
            "first_window_mean_negative_elbo": first_loss,
            "best_window_mean_negative_elbo": best_loss,
            "final_window_mean_negative_elbo": final_loss,
            "initial_to_best_loss_drop": initial_to_best_drop,
            "final_minus_best_loss": final_rebound,
            "final_rebound_fraction_of_initial_drop": (
                final_rebound / initial_to_best_drop
            ),
            "first_10_window_slope_per_1k": first_slope,
            "last_10_window_slope_per_1k": last_slope,
            "n_nonfinite_losses": 0,
            "posterior_samples_all_finite": True,
            "stop_reason": stopping["stop_reason"],
        }
    )

    condition_copy = condition_df.copy()
    condition_copy["animal"] = animal
    condition_frames.append(condition_copy)

loss_summary_df = pd.DataFrame(loss_summary_rows)
loss_summary_df.to_csv(LOSS_SUMMARY_CSV, index=False)
all_conditions_df = pd.concat(condition_frames, ignore_index=True)

print("LED7 uniform-delay convergence summary:")
print(
    loss_summary_df[
        [
            "animal",
            "best_step",
            "final_checked_step",
            "initial_to_best_loss_drop",
            "final_minus_best_loss",
            "final_rebound_fraction_of_initial_drop",
            "first_10_window_slope_per_1k",
            "last_10_window_slope_per_1k",
        ]
    ].to_string(index=False)
)
print(f"Saved: {LOSS_SUMMARY_CSV}")


# %%
# =============================================================================
# One-row convergence diagnostic
# =============================================================================
loss_figure, loss_axes = plt.subplots(
    1,
    6,
    figsize=(22.5, 4.3),
    sharex=True,
)

for ax, payload in zip(loss_axes, fit_payloads):
    losses = payload["window_losses"]
    y_low = min(float(np.min(losses)), payload["best_loss"])
    y_high = max(float(np.max(losses)), payload["best_loss"])
    y_padding = max(1.0, 0.07 * (y_high - y_low))

    ax.plot(
        payload["steps"],
        losses,
        color=WINDOW_COLOR,
        linewidth=1.2,
        marker="o",
        markersize=2.2,
        zorder=3,
    )
    ax.axvline(
        payload["best_step"],
        color=BEST_COLOR,
        linewidth=1.4,
        zorder=2,
    )
    ax.axvline(
        payload["checked_step"],
        color=STOP_COLOR,
        linewidth=1.4,
        linestyle="--",
        zorder=2,
    )
    ax.scatter(
        [payload["best_step"]],
        [payload["best_loss"]],
        color=BEST_COLOR,
        s=22,
        zorder=4,
    )
    ax.set_xlim(0, payload["checked_step"] + 1000)
    ax.set_ylim(y_low - y_padding, y_high + y_padding)
    ax.set_xticks([0, 25000, 50000])
    ax.set_xticklabels(["0", "25k", "50k"])
    ax.set_title(
        f"LED7/{payload['animal']}\n"
        f"best {payload['best_step'] // 1000}k, checked "
        f"{payload['checked_step'] // 1000}k",
        fontsize=9,
    )
    ax.set_xlabel("SVI step")
    ax.grid(alpha=0.18, linewidth=0.5)
    ax.tick_params(axis="both", labelsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)

loss_axes[0].set_ylabel("negative ELBO")
loss_legend = [
    Line2D(
        [0],
        [0],
        color=WINDOW_COLOR,
        linewidth=1.3,
        marker="o",
        markersize=3,
        label="1k-window mean",
    ),
    Line2D(
        [0],
        [0],
        color=BEST_COLOR,
        linewidth=1.5,
        label="restored-best checkpoint",
    ),
    Line2D(
        [0],
        [0],
        color=STOP_COLOR,
        linewidth=1.5,
        linestyle="--",
        label="final checked step",
    ),
]
loss_figure.legend(
    handles=loss_legend,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.97),
    ncol=3,
    frameon=False,
    fontsize=9,
)
loss_figure.suptitle(
    "LED7 NPL+alpha condition-wise Uniform-delay SVI convergence",
    y=1.035,
    fontsize=13,
)
loss_figure.subplots_adjust(
    left=0.055,
    right=0.995,
    bottom=0.16,
    top=0.78,
    wspace=0.34,
)
loss_figure.savefig(LOSS_FIG, dpi=PLOT_DPI, bbox_inches="tight")
print(f"Saved: {LOSS_FIG}")


# %%
# =============================================================================
# One-row animal-wise posterior-mean centers and Uniform support
# =============================================================================
support_low = float(all_conditions_df["low_mean_ms"].min())
support_high = float(all_conditions_df["high_mean_ms"].max())
support_padding = max(3.0, 0.05 * (support_high - support_low))

animal_figure, animal_axes = plt.subplots(
    1,
    6,
    figsize=(22.5, 4.7),
    sharex=True,
    sharey=True,
)

for ax, payload in zip(animal_axes, fit_payloads):
    condition_df = payload["condition_df"]
    for abl in ABLS:
        abl_df = (
            condition_df.loc[condition_df["ABL"].astype(int).eq(abl)]
            .set_index("ILD")
            .loc[list(SIGNED_ILDS)]
            .reset_index()
        )
        centers = abl_df["center_mean_ms"].to_numpy(dtype=float)
        lows = abl_df["low_mean_ms"].to_numpy(dtype=float)
        highs = abl_df["high_mean_ms"].to_numpy(dtype=float)
        support = np.vstack([centers - lows, highs - centers])
        ax.errorbar(
            SIGNED_ILDS,
            centers,
            yerr=support,
            fmt="o-",
            color=ABL_COLORS[abl],
            ecolor=ABL_COLORS[abl],
            linewidth=1.1,
            elinewidth=0.9,
            capsize=2.2,
            markersize=3.6,
            label=f"ABL {abl} dB",
        )

    ax.axvline(0.0, color="0.84", linewidth=0.8, zorder=0)
    ax.set_ylim(
        max(0.0, support_low - support_padding),
        support_high + support_padding,
    )
    ax.set_xticks(SIGNED_ILDS)
    ax.set_xticklabels(
        [f"{ild:g}" for ild in SIGNED_ILDS],
        rotation=45,
        ha="right",
    )
    ax.set_title(f"LED7/{payload['animal']}", fontsize=10)
    ax.set_xlabel("ILD (dB)")
    ax.grid(axis="y", alpha=0.18, linewidth=0.5)
    ax.tick_params(axis="both", labelsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)

animal_axes[0].set_ylabel(r"$t_{E,\mathrm{aff}}$ (ms)")
animal_legend = [
    Line2D(
        [0],
        [0],
        color=ABL_COLORS[abl],
        marker="o",
        linewidth=1.2,
        markersize=4,
        label=f"ABL {abl} dB",
    )
    for abl in ABLS
]
animal_figure.legend(
    handles=animal_legend,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.96),
    ncol=3,
    frameon=False,
    fontsize=9,
)
animal_figure.suptitle(
    "LED7 fitted Uniform evidence-delay centers and support by animal",
    y=1.025,
    fontsize=13,
)
animal_figure.subplots_adjust(
    left=0.055,
    right=0.995,
    bottom=0.23,
    top=0.79,
    wspace=0.16,
)
animal_figure.savefig(ANIMAL_SUPPORT_FIG, dpi=PLOT_DPI, bbox_inches="tight")
print(f"Saved: {ANIMAL_SUPPORT_FIG}")


# %%
# =============================================================================
# Across-animal center, mean Uniform support, and center SEM
# =============================================================================
average_rows = []
for abl in ABLS:
    for ild in SIGNED_ILDS:
        rows = all_conditions_df.loc[
            all_conditions_df["ABL"].astype(int).eq(abl)
            & all_conditions_df["ILD"].astype(float).eq(ild)
        ]
        if len(rows) != len(ANIMALS):
            raise RuntimeError(
                f"Expected {len(ANIMALS)} animals for ABL={abl}, ILD={ild:g}; "
                f"found {len(rows)}."
            )
        centers = rows["center_mean_ms"].to_numpy(dtype=float)
        average_rows.append(
            {
                "ABL": abl,
                "ILD": ild,
                "n_animals": len(rows),
                "mean_center_ms": float(np.mean(centers)),
                "sem_center_ms": float(
                    np.std(centers, ddof=1) / np.sqrt(len(centers))
                ),
                "mean_low_support_ms": float(np.mean(rows["low_mean_ms"])),
                "mean_high_support_ms": float(np.mean(rows["high_mean_ms"])),
                "mean_width_ms": float(np.mean(rows["width_mean_ms"])),
            }
        )

average_df = pd.DataFrame(average_rows)
average_df.to_csv(AVERAGE_SUPPORT_CSV, index=False)

average_figure, average_ax = plt.subplots(
    figsize=(9.2, 5.4),
    constrained_layout=True,
)
for abl in ABLS:
    abl_df = average_df.loc[average_df["ABL"].eq(abl)].set_index("ILD")
    abl_df = abl_df.loc[list(SIGNED_ILDS)].reset_index()
    centers = abl_df["mean_center_ms"].to_numpy(dtype=float)
    support_low_values = abl_df["mean_low_support_ms"].to_numpy(dtype=float)
    support_high_values = abl_df["mean_high_support_ms"].to_numpy(dtype=float)
    support = np.vstack(
        [centers - support_low_values, support_high_values - centers]
    )

    # The broad translucent bars are average fitted Uniform support, not error.
    average_ax.errorbar(
        SIGNED_ILDS,
        centers,
        yerr=support,
        fmt="none",
        ecolor=ABL_COLORS[abl],
        elinewidth=5.0,
        capsize=4.5,
        capthick=1.5,
        alpha=0.25,
        zorder=1,
    )
    # These narrow capped bars quantify between-animal uncertainty in the centers.
    average_ax.errorbar(
        SIGNED_ILDS,
        centers,
        yerr=abl_df["sem_center_ms"].to_numpy(dtype=float),
        fmt="o-",
        color=ABL_COLORS[abl],
        ecolor=ABL_COLORS[abl],
        linewidth=1.3,
        elinewidth=1.0,
        capsize=2.5,
        markersize=4.8,
        label=f"ABL {abl} dB",
        zorder=3,
    )

average_ax.axvline(0.0, color="0.84", linewidth=0.9, zorder=0)
average_ax.set_xticks(SIGNED_ILDS)
average_ax.set_xticklabels(
    [f"{ild:g}" for ild in SIGNED_ILDS],
    rotation=45,
    ha="right",
)
average_ax.set_xlabel("ILD (dB)")
average_ax.set_ylabel(r"$t_{E,\mathrm{aff}}$ (ms)")
average_ax.set_title(
    "LED7 NPL+alpha Uniform delays averaged equally across six animals"
)
average_ax.grid(axis="y", alpha=0.2, linewidth=0.6)
average_ax.spines[["top", "right"]].set_visible(False)
average_ax.legend(frameon=False, loc="upper right")
average_ax.text(
    0.02,
    0.03,
    "thick translucent bar: mean fitted Uniform support\n"
    "thin capped bar: SEM of animal posterior-mean centers",
    transform=average_ax.transAxes,
    fontsize=8.5,
    va="bottom",
    ha="left",
)
average_figure.savefig(AVERAGE_SUPPORT_FIG, dpi=PLOT_DPI, bbox_inches="tight")

print(f"Saved: {AVERAGE_SUPPORT_CSV}")
print(f"Saved: {AVERAGE_SUPPORT_FIG}")
