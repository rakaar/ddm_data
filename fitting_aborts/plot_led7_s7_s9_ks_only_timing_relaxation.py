# %%
"""Show LED7 s7 timing matches as row counts increase, audited by KS only."""

# %%
# Editable parameters
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_MAT_PATH = SCRIPT_DIR.parent / "raw_data" / "outMatrix_LED7_latest.mat"
TABLE_NAME = "totalout_stGtACRII"
OUTPUT_FIGURE_PATH = SCRIPT_DIR / "led7_s7_s9_ks_only_timing_relaxation_2x5.png"
OUTPUT_AUDIT_PATH = SCRIPT_DIR / "led7_s7_s9_ks_only_timing_relaxation_audit.csv"

TRAINING_LEVEL = 16
REPEAT_TRIAL_VALUES = {0, 2}
LED_TRIAL_VALUES = (0, 1)
EXPECTED_S7_COUNTS = {0: 92_057, 1: 46_383}
EXPECTED_S9_ROWS = 100_788
MATCH_RANDOM_SEED = 20_260_909

# The first pair reproduces the previously tested ~0.03-KS selection. The
# others ask the same matcher to retain more rows; KS is measured afterward.
STAGES = (
    ("Current pair", {0: 14_140, 1: 7_355}),
    ("Larger 1", {0: 15_000, 1: 8_000}),
    ("Larger 2", {0: 17_000, 1: 9_000}),
    ("Larger 3", {0: 20_000, 1: 10_000}),
    ("All s7", EXPECTED_S7_COUNTS),
)

MATCHING_COLUMNS = ["intended_fix", "effective_scheduled_onset"]
BIN_WIDTH_S = 0.020
MATCH_INTENDED_BINS = np.arange(200, 2200 + 20, 20) / 1000
MATCH_ONSET_BINS = np.arange(100, 1100 + 20, 20) / 1000
DISPLAY_BINS = np.arange(0, 2200 + 20, 20) / 1000
OUTPUT_DPI = 250
SHOW_PLOT = False


# %%
# Imports and reusable timing calculations
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from scipy.stats import ks_2samp

from led7_scheduled_timing_matching_utils import (
    largest_remainder_counts,
    select_rows_matching_two_margins,
)

try:
    from matio import load_from_mat
except ImportError as exc:
    raise ImportError(
        "Run with: env PYTHONPATH=/tmp/codex_mat_io .venv/bin/python "
        "fitting_aborts/plot_led7_s7_s9_ks_only_timing_relaxation.py"
    ) from exc


def density_histogram(values, bins):
    values = np.asarray(values, dtype=float)
    counts, _ = np.histogram(values, bins=bins)
    if int(counts.sum()) != len(values):
        raise RuntimeError("The displayed timing range omitted source rows.")
    density = counts / (len(values) * np.diff(bins))
    area = float(np.sum(density * np.diff(bins)))
    if not np.isclose(area, 1.0, atol=1e-12):
        raise RuntimeError("Timing histogram does not have unit area.")
    return density, area


# %%
# Filter before matching and define each session type's effective onset
mat_contents = load_from_mat(INPUT_MAT_PATH, variable_names=[TABLE_NAME])
raw = mat_contents[TABLE_NAME]
if not isinstance(raw, pd.DataFrame):
    raise TypeError(f"Expected a DataFrame, found {type(raw).__name__}.")
required_columns = {
    "training_level", "session_type", "repeat_trial", "LED_trial",
    "intended_fix", "LED_onset_time",
}
missing_columns = sorted(required_columns - set(raw.columns))
if missing_columns:
    raise RuntimeError(f"Source table is missing {missing_columns}.")

filtered = raw.loc[raw["training_level"].eq(TRAINING_LEVEL)].copy()
filtered = filtered.loc[filtered["session_type"].isin((7, 9))].copy()
filtered = filtered.loc[
    filtered["repeat_trial"].isin(REPEAT_TRIAL_VALUES)
    | filtered["repeat_trial"].isna()
].copy()
filtered = filtered.loc[filtered["LED_trial"].isin(LED_TRIAL_VALUES)].copy()

s7 = filtered.loc[filtered["session_type"].eq(7)].copy()
s9 = filtered.loc[filtered["session_type"].eq(9)].copy()
s7["effective_scheduled_onset"] = s7["intended_fix"] - s7["LED_onset_time"]
s9["effective_scheduled_onset"] = s9["LED_onset_time"]

if len(s9) != EXPECTED_S9_ROWS or not s9.index.is_unique:
    raise RuntimeError("The pooled s9 timing reference changed.")
if not s7.index.is_unique:
    raise RuntimeError("s7 source-row indices are not unique.")
for led in LED_TRIAL_VALUES:
    if int(s7["LED_trial"].eq(led).sum()) != EXPECTED_S7_COUNTS[led]:
        raise RuntimeError(f"s7 LED {led} count changed.")
for label, frame in (("s7", s7), ("s9", s9)):
    if not np.isfinite(frame[MATCHING_COLUMNS].to_numpy(dtype=float)).all():
        raise RuntimeError(f"{label} has non-finite timing.")
    if (frame["effective_scheduled_onset"] < 0).any():
        raise RuntimeError(f"{label} has negative scheduled onset.")
    if (frame["effective_scheduled_onset"] > frame["intended_fix"] + 1e-12).any():
        raise RuntimeError(f"{label} has onset after intended_fix.")

reference_intended_counts, _ = np.histogram(s9["intended_fix"], MATCH_INTENDED_BINS)
reference_onset_counts, _ = np.histogram(
    s9["effective_scheduled_onset"], MATCH_ONSET_BINS
)
if int(reference_intended_counts.sum()) != len(s9):
    raise RuntimeError("s9 intended_fix exceeds matching support.")
if int(reference_onset_counts.sum()) != len(s9):
    raise RuntimeError("s9 onset exceeds matching support.")


# %%
# Select intact trial-wise timing pairs using only the two timing columns
stage_groups = {}
for stage_label, sizes in STAGES:
    groups = {}
    matching_rng = np.random.default_rng(MATCH_RANDOM_SEED)
    for led in LED_TRIAL_VALUES:
        candidate = s7.loc[s7["LED_trial"].eq(led)].copy()
        selected_n = sizes[led]
        if stage_label == "All s7":
            group = candidate
        else:
            target_intended = largest_remainder_counts(
                reference_intended_counts, selected_n
            )
            target_onset = largest_remainder_counts(
                reference_onset_counts, selected_n
            )
            selected_indices, _ = select_rows_matching_two_margins(
                timing_frame=candidate[MATCHING_COLUMNS].copy(),
                target_intended_counts=target_intended,
                target_onset_counts=target_onset,
                rng=matching_rng,
                matching_columns=MATCHING_COLUMNS,
                intended_bins=MATCH_INTENDED_BINS,
                onset_bins=MATCH_ONSET_BINS,
                sample_size=selected_n,
                bin_width_s=BIN_WIDTH_S,
            )
            group = candidate.loc[selected_indices].copy()
        if len(group) != selected_n or not group.index.is_unique:
            raise RuntimeError(f"{stage_label} LED {led} contains repeated or missing rows.")
        groups[led] = group
    if not groups[0].index.intersection(groups[1].index).empty:
        raise RuntimeError(f"{stage_label} OFF and ON selections overlap.")
    stage_groups[stage_label] = groups


# %%
# Measure achieved unbinned KS and prepare unit-area histograms
variables = (
    ("intended_fix", "intended_fix (s)"),
    ("effective_scheduled_onset", "Scheduled LED onset from fixation (s)"),
)
reference_densities = {
    column: density_histogram(s9[column], DISPLAY_BINS)[0]
    for column, _ in variables
}
plot_payload = {}
audit_rows = []
for stage_label, sizes in STAGES:
    for led in LED_TRIAL_VALUES:
        group = stage_groups[stage_label][led]
        for column, _ in variables:
            density, area = density_histogram(group[column], DISPLAY_BINS)
            ks = float(ks_2samp(group[column], s9[column], method="asymp").statistic)
            if not np.isfinite(ks) or not (0 <= ks <= 1):
                raise RuntimeError("Invalid KS result.")
            plot_payload[(stage_label, led, column)] = {"density": density, "ks": ks}
            audit_rows.append(
                {
                    "stage": stage_label,
                    "variable": column,
                    "LED_trial": led,
                    "selected_n": len(group),
                    "reference_n": len(s9),
                    "KS_distance": ks,
                    "histogram_area": area,
                }
            )

audit = pd.DataFrame(audit_rows)
if len(audit) != 2 * len(STAGES) * len(LED_TRIAL_VALUES):
    raise RuntimeError("Timing audit is incomplete.")
audit.to_csv(OUTPUT_AUDIT_PATH, index=False)


# %%
# Compare progressively relaxed selections against the same pooled s9 reference
colors = {0: "tab:blue", 1: "tab:red"}
bin_centers = 0.5 * (DISPLAY_BINS[:-1] + DISPLAY_BINS[1:])
fig, axes = plt.subplots(2, len(STAGES), figsize=(22.0, 8.8), sharex=True, sharey="row")
for column_index, (stage_label, sizes) in enumerate(STAGES):
    for row_index, (column, xlabel) in enumerate(variables):
        ax = axes[row_index, column_index]
        ax.step(
            bin_centers, reference_densities[column], where="mid",
            color="black", linestyle="--", linewidth=1.7, zorder=3,
        )
        for led in LED_TRIAL_VALUES:
            payload = plot_payload[(stage_label, led, column)]
            ax.step(
                bin_centers, payload["density"], where="mid",
                color=colors[led], linewidth=1.3, zorder=2,
            )
            ax.text(
                0.97, 0.95 - 0.09 * led,
                f"{'OFF' if led == 0 else 'ON'} KS = {payload['ks']:.3f}",
                color=colors[led], transform=ax.transAxes,
                ha="right", va="top", fontsize=9,
            )
        if row_index == 0:
            ax.set_title(f"{stage_label}\nOFF {sizes[0]:,} · ON {sizes[1]:,}", fontsize=12)
        ax.set_xlabel(xlabel)
        if column_index == 0:
            ax.set_ylabel("Density (s$^{-1}$)")
        ax.set_xlim(0, 2.2)
        ax.grid(axis="y", color="0.88", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)

legend_handles = [
    Line2D([0], [0], color="black", linestyle="--", linewidth=1.7, label="Pooled s9"),
    Line2D([0], [0], color=colors[0], linewidth=1.3, label="s7 LED OFF"),
    Line2D([0], [0], color=colors[1], linewidth=1.3, label="s7 LED ON"),
]
fig.suptitle("LED7 s7 timing-match relaxation versus pooled s9 (achieved KS)", y=0.995)
fig.legend(
    handles=legend_handles, loc="upper center",
    bbox_to_anchor=(0.5, 0.96), ncol=3, frameon=False,
)
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.savefig(OUTPUT_FIGURE_PATH, dpi=OUTPUT_DPI, bbox_inches="tight")
if SHOW_PLOT:
    plt.show()
else:
    plt.close(fig)

print(audit.to_string(index=False))
print(f"\nFigure: {OUTPUT_FIGURE_PATH.resolve()}")
print(f"Audit: {OUTPUT_AUDIT_PATH.resolve()}")
