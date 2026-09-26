# %%
"""Check whether LED7 s7 fixation aborts retain the assigned fixation schedule."""

from pathlib import Path
import importlib.util
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")


# %%
############ Parameters (edit here) ############
SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_PATH = SCRIPT_DIR.parent / "raw_data" / "outMatrix_LED7_latest.mat"
TABLE_NAME = "totalout_stGtACRII"
TRAINING_LEVEL = 16
SESSION_TYPE = 7
ABORT_EVENT = 3
COMPLETED_SUCCESS_CODES = (-1, 1)
BIN_WIDTH_S = 0.020
FIGURE_PATH = SCRIPT_DIR / "led7_s7_abort_vs_outcome_union_intended_fix.png"
AUDIT_PATH = SCRIPT_DIR / "led7_s7_abort_vs_outcome_union_intended_fix_bins.csv"

# MATIO_SITE_PACKAGES can point to an isolated mat-io installation. The project
# environment is deliberately left unchanged.
MATIO_SITE_PACKAGES = os.environ.get("MATIO_SITE_PACKAGES")
if importlib.util.find_spec("matio") is None:
    if MATIO_SITE_PACKAGES is None:
        raise ModuleNotFoundError(
            "MATLAB table reader not found; set MATIO_SITE_PACKAGES to an "
            "isolated site-packages directory containing mat-io."
        )
    sys.path.insert(0, str(Path(MATIO_SITE_PACKAGES).expanduser()))

from matio import load_from_mat
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

BINS_S = np.arange(0.0, 2.2 + BIN_WIDTH_S / 2, BIN_WIDTH_S)


# %%
############ Read MAT table and define the two requested populations ############
if not SOURCE_PATH.exists():
    raise FileNotFoundError(SOURCE_PATH)
source = load_from_mat(SOURCE_PATH, variable_names=[TABLE_NAME])[TABLE_NAME]
if not isinstance(source, pd.DataFrame):
    raise TypeError(f"{TABLE_NAME} did not decode as a table.")
required = {"training_level", "session_type", "abort_event", "success", "intended_fix", "repeat_trial", "LED_trial"}
if missing := required.difference(source.columns):
    raise RuntimeError(f"Missing source columns: {sorted(missing)}")

# Deliberately no repeat_trial or LED_trial filter: this is the user's exact
# training-level-16, session-type-7 comparison directly from the MAT table.
level_16 = source.loc[source["training_level"].eq(TRAINING_LEVEL)]
s7 = level_16.loc[level_16["session_type"].eq(SESSION_TYPE)]
is_abort = s7["abort_event"].eq(ABORT_EVENT)
is_completed = s7["success"].isin(COMPLETED_SUCCESS_CODES)
if (is_abort & is_completed).any():
    raise RuntimeError("Event-3 aborts and completed outcomes overlap.")
is_union = is_abort | is_completed

abort_values = s7.loc[is_abort, "intended_fix"].to_numpy(dtype=float)
union_values = s7.loc[is_union, "intended_fix"].to_numpy(dtype=float)
if not np.isfinite(abort_values).all() or not np.isfinite(union_values).all():
    raise RuntimeError("A requested intended_fix population has non-finite times.")
if not len(abort_values) or not len(union_values):
    raise RuntimeError("A requested outcome population is empty.")
if (abort_values < BINS_S[0]).any() or (abort_values > BINS_S[-1]).any():
    raise RuntimeError("Event-3 intended_fix values fall outside plotting bins.")
if (union_values < BINS_S[0]).any() or (union_values > BINS_S[-1]).any():
    raise RuntimeError("Outcome-union intended_fix values fall outside plotting bins.")

abort_counts, _ = np.histogram(abort_values, bins=BINS_S)
union_counts, _ = np.histogram(union_values, bins=BINS_S)
if abort_counts.sum() != len(abort_values) or union_counts.sum() != len(union_values):
    raise RuntimeError("Histogram omitted intended_fix observations.")
abort_density = abort_counts / (len(abort_values) * BIN_WIDTH_S)
union_density = union_counts / (len(union_values) * BIN_WIDTH_S)
if not np.isclose(np.sum(abort_density * np.diff(BINS_S)), 1.0):
    raise RuntimeError("Abort density does not integrate to one.")
if not np.isclose(np.sum(union_density * np.diff(BINS_S)), 1.0):
    raise RuntimeError("Outcome-union density does not integrate to one.")

abort_fraction = np.divide(
    abort_counts,
    union_counts,
    out=np.full(len(abort_counts), np.nan),
    where=union_counts > 0,
)
if ((abort_fraction < 0) | (abort_fraction > 1)).any():
    raise RuntimeError("Invalid conditional abort fractions.")
ks_distance = float(ks_2samp(abort_values, union_values).statistic)

print(f"Source: {SOURCE_PATH}::{TABLE_NAME}")
print(f"training_level == {TRAINING_LEVEL}: {len(level_16):,} rows")
print(f"session_type == {SESSION_TYPE}: {len(s7):,} rows")
print(f"abort_event == {ABORT_EVENT}: {len(abort_values):,} rows")
print(f"success in {COMPLETED_SUCCESS_CODES}: {int(is_completed.sum()):,} rows")
print(f"union: {len(union_values):,} rows; other outcomes: {len(s7)-len(union_values):,}")
print(
    f"Median intended_fix: abort {np.median(abort_values):.6f} s; "
    f"union {np.median(union_values):.6f} s; KS {ks_distance:.6f}"
)

# The earlier animal-wise figure used the regular repeat/LED filters and all
# eligible outcomes. Print that counterpart to isolate the role of filtering.
regular = s7.loc[
    (s7["repeat_trial"].isin((0, 2)) | s7["repeat_trial"].isna())
    & (s7["LED_trial"].isin((0, 1)) | s7["LED_trial"].isna())
]
regular_abort = regular.loc[regular["abort_event"].eq(ABORT_EVENT), "intended_fix"]
regular_union = regular.loc[
    regular["abort_event"].eq(ABORT_EVENT)
    | regular["success"].isin(COMPLETED_SUCCESS_CODES),
    "intended_fix",
]
print(
    f"With previous repeat/LED filters: {len(regular):,} all, "
    f"{len(regular_abort):,} abort, {len(regular_union):,} union; "
    f"abort-versus-union KS "
    f"{ks_2samp(regular_abort, regular_union).statistic:.6f}"
)

audit = pd.DataFrame(
    {
        "intended_fix_bin_left_s": BINS_S[:-1],
        "intended_fix_bin_right_s": BINS_S[1:],
        "abort_count": abort_counts,
        "abort_or_completed_count": union_counts,
        "abort_density_per_s": abort_density,
        "abort_or_completed_density_per_s": union_density,
        "abort_fraction_in_union": abort_fraction,
    }
)
audit.to_csv(AUDIT_PATH, index=False)


# %%
############ Plot the distributions and their selection effect ############
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
axes[0].stairs(
    union_density,
    BINS_S,
    color="black",
    linewidth=1.6,
    label=f"Abort 3 or success ±1  (n = {len(union_values):,})",
)
axes[0].stairs(
    abort_density,
    BINS_S,
    color="tab:red",
    linewidth=1.6,
    label=f"Abort 3  (n = {len(abort_values):,})",
)
axes[0].set(xlabel="intended_fix (s)", ylabel="Density (s$^{-1}$)")
axes[0].legend(frameon=False, fontsize=9)
axes[0].text(
    0.97,
    0.95,
    f"KS = {ks_distance:.3f}\nmedian: {np.median(abort_values):.3f} vs "
    f"{np.median(union_values):.3f} s",
    ha="right",
    va="top",
    transform=axes[0].transAxes,
    fontsize=9,
)

axes[1].stairs(abort_fraction, BINS_S, color="tab:red", linewidth=1.6)
axes[1].axhline(
    len(abort_values) / len(union_values),
    color="0.55",
    linestyle="--",
    linewidth=1.0,
)
axes[1].set(
    xlabel="intended_fix (s)",
    ylabel="Fraction abort 3 among abort 3 or success ±1",
    ylim=(0, 1),
)
axes[1].text(
    0.97,
    0.95,
    f"Overall = {len(abort_values)/len(union_values):.3f}",
    ha="right",
    va="top",
    transform=axes[1].transAxes,
    fontsize=9,
)

for ax in axes:
    ax.set_xlim(BINS_S[0], BINS_S[-1])
    ax.grid(alpha=0.15)
fig.suptitle("LED7 · session type 7 · training level 16")
fig.tight_layout()
fig.savefig(FIGURE_PATH, dpi=250, bbox_inches="tight")
plt.close(fig)
print(f"Figure: {FIGURE_PATH}")
print(f"Bin audit: {AUDIT_PATH}")
