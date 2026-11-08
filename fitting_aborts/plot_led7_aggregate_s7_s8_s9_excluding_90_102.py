# %%
"""Trial-pooled LED7 timing and abort comparison, excluding animals 90 and 102."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# %%
############ Parameters (edit here) ############
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RAW_DATA_DIR = REPO_ROOT / "raw_data"
SESSION_TYPES = (7, 8, 9)
ANIMALS = (92, 93, 98, 99, 100, 103)
LED_TRIALS = (0, 1)
ABORT_EVENT = 3
BIN_WIDTH_S = 0.020
TIMING_BINS = np.arange(0.0, 2.2 + BIN_WIDTH_S / 2, BIN_WIDTH_S)
ABORT_BINS = np.arange(-2.0, 2.0 + BIN_WIDTH_S / 2, BIN_WIDTH_S)
DISPLAY_XLIMS = ((-0.4, 0.5), (-0.1, 0.3))
FIGURE_PATH = SCRIPT_DIR / "led7_aggregate_s7_s8_s9_excluding_90_102_3x4.png"
AUDIT_PATH = SCRIPT_DIR / "led7_aggregate_s7_s8_s9_excluding_90_102_audit.csv"
LED_COLORS = {0: "tab:blue", 1: "tab:red"}


# %%
############ Load and validate standardized CSVs ############
datasets = {}
for session_type in SESSION_TYPES:
    path = RAW_DATA_DIR / f"LED7_s{session_type}.csv"
    frame = pd.read_csv(path, float_precision="round_trip")
    required = {
        "animal", "session_type", "training_level", "repeat_trial", "LED_trial",
        "abort_event", "timed_fix", "intended_fix", "LED_onset_time",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"{path} lacks required columns: {missing}")
    if not frame["session_type"].eq(session_type).all():
        raise RuntimeError(f"Unexpected session type in {path}.")
    if not frame["training_level"].eq(16).all():
        raise RuntimeError(f"Unexpected training level in {path}.")
    if not (frame["repeat_trial"].isin((0, 2)) | frame["repeat_trial"].isna()).all():
        raise RuntimeError(f"Unexpected repeat_trial value in {path}.")
    if not frame["LED_trial"].isin(LED_TRIALS).all():
        raise RuntimeError(f"Unexpected LED_trial value in {path}.")
    if set(frame["animal"].astype(int).unique()) != set((*ANIMALS, 90, 102)):
        raise RuntimeError(f"Unexpected source animal set in {path}.")
    frame = frame.loc[frame["animal"].isin(ANIMALS)].copy()
    if set(frame["animal"].astype(int).unique()) != set(ANIMALS):
        raise RuntimeError(f"Requested cohort is incomplete in {path}.")
    aborts = frame.loc[frame["abort_event"].eq(ABORT_EVENT)].copy()
    finite = aborts.loc[np.isfinite(aborts["timed_fix"])]
    if not (finite["timed_fix"] < finite["intended_fix"]).all():
        raise RuntimeError(f"Finite event-3 abort occurs after intended_fix in {path}.")
    datasets[session_type] = {"all": frame, "abort": aborts}
    print(
        f"s{session_type}: {len(frame):,} eligible trials, {len(aborts):,} "
        f"event-3 aborts ({len(aborts) - len(finite):,} missing timed_fix)"
    )


# %%
############ Pool trials, calculate histograms, and draw 3 x 4 figure ############
fig, axes = plt.subplots(3, 4, figsize=(19, 11), sharex="col")
timing_ymax = np.zeros(2)
rate_ymax = 0.0
audit_rows = []

for row, session_type in enumerate(SESSION_TYPES):
    all_trials = datasets[session_type]["all"]
    abort_trials = datasets[session_type]["abort"]

    for col, field in enumerate(("intended_fix", "LED_onset_time")):
        all_values = all_trials[field].to_numpy(dtype=float)
        abort_values = abort_trials[field].to_numpy(dtype=float)
        all_counts, _ = np.histogram(all_values, bins=TIMING_BINS)
        abort_counts, _ = np.histogram(abort_values, bins=TIMING_BINS)
        if all_counts.sum() != len(all_values) or abort_counts.sum() != len(abort_values):
            raise RuntimeError(f"Timing histogram omitted values in s{session_type}.")
        all_density = all_counts / (len(all_values) * BIN_WIDTH_S)
        abort_density = abort_counts / (len(abort_values) * BIN_WIDTH_S)
        if not np.isclose(np.sum(all_density * BIN_WIDTH_S), 1.0):
            raise RuntimeError("All-trial timing density does not integrate to one.")
        if not np.isclose(np.sum(abort_density * BIN_WIDTH_S), 1.0):
            raise RuntimeError("Abort timing density does not integrate to one.")
        ax = axes[row, col]
        ax.stairs(all_density, TIMING_BINS, color="black", linewidth=1.35)
        ax.stairs(abort_density, TIMING_BINS, color="tab:red", linewidth=1.35)
        ax.text(
            0.97, 0.95,
            f"all n = {len(all_values):,}\nabort n = {len(abort_values):,}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
        )
        timing_ymax[col] = max(timing_ymax[col], all_density.max(), abort_density.max())

    for led in LED_TRIALS:
        group_trials = all_trials.loc[all_trials["LED_trial"].eq(led)]
        group_aborts = abort_trials.loc[abort_trials["LED_trial"].eq(led)]
        finite = group_aborts.loc[np.isfinite(group_aborts["timed_fix"])]
        aligned = (finite["timed_fix"] - finite["LED_onset_time"]).to_numpy(dtype=float)
        counts, _ = np.histogram(aligned, bins=ABORT_BINS)
        if counts.sum() != len(aligned):
            raise RuntimeError(f"Abort histogram omitted values in s{session_type}, LED {led}.")
        heights = counts / (len(group_trials) * BIN_WIDTH_S)
        area = float(np.sum(heights * BIN_WIDTH_S))
        expected_area = len(finite) / len(group_trials)
        if not np.isclose(area, expected_area, rtol=0, atol=1e-14):
            raise RuntimeError("Abort histogram area does not equal finite abort fraction.")
        for col in (2, 3):
            axes[row, col].stairs(heights, ABORT_BINS, color=LED_COLORS[led], linewidth=1.4)
        rate_ymax = max(rate_ymax, float(heights.max()))
        audit_rows.append({
            "animal_cohort": ",".join(map(str, ANIMALS)),
            "session_type": session_type,
            "LED_trial": led,
            "all_trials": len(group_trials),
            "coded_event3_aborts": len(group_aborts),
            "finite_event3_aborts": len(finite),
            "missing_timed_fix": len(group_aborts) - len(finite),
            "full_histogram_area": area,
            "finite_abort_fraction": expected_area,
        })

    for col in (2, 3):
        axes[row, col].axvline(0, color="0.35", linestyle=":", linewidth=1.0)
    off, on = audit_rows[-2:]
    for col in (2, 3):
        axes[row, col].text(
            0.03, 0.95,
            f"OFF area = {off['full_histogram_area']:.3f}\n"
            f"ON area = {on['full_histogram_area']:.3f}",
            transform=axes[row, col].transAxes, ha="left", va="top", fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88},
        )
    axes[row, 0].set_ylabel(f"Session type {session_type}\nDensity (s$^{{-1}}$)")
    axes[row, 2].set_ylabel("Abort rate (s$^{-1}$)")

titles = (
    "intended_fix",
    "LED_onset_time",
    "Abort rate from LED onset\n−0.4 to +0.5 s",
    "Abort rate from LED onset\n−0.1 to +0.3 s",
)
for col, title in enumerate(titles):
    axes[0, col].set_title(title, fontsize=12)
    for row in range(3):
        ax = axes[row, col]
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.16, linewidth=0.6)
        ax.tick_params(labelsize=8)
        if col < 2:
            ax.set_xlim(TIMING_BINS[0], TIMING_BINS[-1])
            ax.set_ylim(0, 1.12 * timing_ymax[col])
        else:
            ax.set_xlim(DISPLAY_XLIMS[col - 2])
            ax.set_ylim(0, 1.15 * rate_ymax)
    axes[-1, col].set_xlabel(
        "Time from fixation onset (s)" if col < 2 else "Abort time from LED onset (s)"
    )

fig.legend(
    handles=[
        Line2D([0], [0], color="black", lw=1.5, label="Timing: all eligible trials"),
        Line2D([0], [0], color="tab:red", lw=1.5, label="Timing: event-3 aborts"),
        Line2D([0], [0], color="tab:blue", lw=1.5, label="Abort rate: LED OFF"),
        Line2D([0], [0], color="tab:red", lw=1.5, label="Abort rate: LED ON"),
    ],
    loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=4, frameon=False, fontsize=9.5,
)
fig.suptitle(
    "LED7 trial-pooled aggregate (animals 92, 93, 98, 99, 100, 103): "
    "session types 7, 8, and 9",
    fontsize=14,
)
fig.text(
    0.5, 0.012,
    "Training level 16; repeat_trial 0/2/NaN; LED_trial 0/1. Timing curves each "
    "have area 1. Abort-rate areas use the full −2 to +2 s histogram and equal "
    "finite event-3 aborts / all trials in that LED group. Missing-timing aborts "
    "remain in denominators. LED OFF onset is scheduled/counterfactual.",
    ha="center", fontsize=9,
)
fig.tight_layout(rect=(0.025, 0.05, 1, 0.90), w_pad=1.15, h_pad=1.2)
fig.savefig(FIGURE_PATH, dpi=220, bbox_inches="tight")
plt.close(fig)

audit = pd.DataFrame(audit_rows)
if len(audit) != len(SESSION_TYPES) * len(LED_TRIALS):
    raise RuntimeError("Unexpected number of pooled LED-group audit rows.")
audit.to_csv(AUDIT_PATH, index=False)
print(f"Saved figure: {FIGURE_PATH}")
print(f"Saved audit: {AUDIT_PATH}")
print(audit.to_string(index=False))
