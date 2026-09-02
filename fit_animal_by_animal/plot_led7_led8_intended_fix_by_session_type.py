# %%
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


# %%
############ Parameters (edit here) ############
TRAINING_LEVEL = 16
BIN_START_S = 0.0
BIN_STOP_S = 2.5
BIN_SIZE_S = 0.020

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "raw_data"

DATA_PATHS = {
    "LED7": DATA_DIR / "out_LED.csv",
    "LED8": DATA_DIR / "outLED8.csv",
}

PANELS = [
    ("LED7", 1, "tab:blue"),
    ("LED7", 7, "tab:blue"),
    ("LED7", 8, "tab:blue"),
    ("LED8", 1, "tab:orange"),
]

OUTPUT_PATH = SCRIPT_DIR / "led7_led8_training16_intended_fix_by_session_type_1x4.png"


# %%
############ Load data, filtering training level first ############
required_columns = ["training_level", "session_type", "intended_fix"]
training_level_data = {}

for dataset_name, csv_path in DATA_PATHS.items():
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find input CSV: {csv_path}")

    raw_df = pd.read_csv(csv_path, usecols=required_columns)
    training_level_data[dataset_name] = raw_df.loc[
        raw_df["training_level"].eq(TRAINING_LEVEL)
    ].copy()


# %%
############ Plot session-type distributions ############
bin_edges = np.arange(BIN_START_S, BIN_STOP_S + BIN_SIZE_S / 2, BIN_SIZE_S)
if not np.isclose(bin_edges[-1], BIN_STOP_S):
    raise RuntimeError("Histogram bins do not end at the requested upper limit.")

fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharex=True, sharey=True)
audit_rows = []

for ax, (dataset_name, session_type, color) in zip(axes, PANELS):
    training_df = training_level_data[dataset_name]
    session_df = training_df.loc[training_df["session_type"].eq(session_type)].copy()
    intended_fix = pd.to_numeric(session_df["intended_fix"], errors="coerce")
    intended_fix = intended_fix.loc[np.isfinite(intended_fix)].to_numpy(dtype=float)

    if intended_fix.size == 0:
        raise RuntimeError(
            f"No finite intended_fix values for {dataset_name}, "
            f"training_level={TRAINING_LEVEL}, session_type={session_type}."
        )

    in_display_range = (intended_fix >= BIN_START_S) & (intended_fix <= BIN_STOP_S)
    histogram_weights = np.full(intended_fix.size, 1.0 / intended_fix.size)

    ax.hist(
        intended_fix,
        bins=bin_edges,
        weights=histogram_weights,
        color=color,
        alpha=0.78,
        edgecolor=color,
        linewidth=0.35,
    )
    ax.set_title(f"{dataset_name}: session type {session_type}\nn = {intended_fix.size:,}")
    ax.set_xlim(BIN_START_S, BIN_STOP_S)
    ax.set_xticks(np.arange(BIN_START_S, BIN_STOP_S + 0.001, 0.5))
    ax.set_xlabel("intended_fix (s)")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    audit_rows.append(
        {
            "dataset": dataset_name,
            "training_level": TRAINING_LEVEL,
            "session_type": session_type,
            "n_finite": int(intended_fix.size),
            "n_in_0_to_2p5_s": int(in_display_range.sum()),
            "fraction_in_0_to_2p5_s": float(in_display_range.mean()),
            "intended_fix_min_s": float(intended_fix.min()),
            "intended_fix_max_s": float(intended_fix.max()),
        }
    )

axes[0].set_ylabel("Fraction of trials per 20 ms bin")
axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))

fig.suptitle(
    "intended_fix distributions after filtering training_level = 16",
    fontsize=14,
)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUTPUT_PATH, dpi=250, bbox_inches="tight")
plt.close(fig)


# %%
############ Report saved output and plotted rows ############
audit_df = pd.DataFrame(audit_rows)
print(audit_df.to_string(index=False))
print(f"\nSaved: {OUTPUT_PATH.resolve()}")
