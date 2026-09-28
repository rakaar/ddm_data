# %%
"""Compare LED7 timing schedules and onset-aligned abort rates by animal."""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


# %%
############ Parameters (edit here) ############
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RAW_DATA_DIR = REPO_ROOT / "raw_data"

SESSION_TYPES = (7, 8, 9)
ANIMALS = (90, 92, 93, 98, 99, 100, 102, 103)
EXPERIMENTAL_ANIMALS = (92, 93, 98, 99, 100, 103)
COHORTS = (*ANIMALS, "aggregate", "aggregate_no_controls")
TRAINING_LEVEL = 16
ALLOWED_REPEAT_TRIALS = (0, 2)
ALLOWED_LED_TRIALS = (0, 1)
ABORT_EVENT = 3
EXPECTED_COLUMNS = 52
EXPECTED_ALL_ROWS = {7: 138_440, 8: 148_428, 9: 100_788}
EXPECTED_ABORT_ROWS = {7: 22_494, 8: 22_926, 9: 18_860}
EXPECTED_MISSING_TIMED_FIX = {7: 3, 8: 2, 9: 6}
EXPECTED_ALL_LED_COUNTS = {
    7: {0: 92_057, 1: 46_383},
    8: {0: 92_784, 1: 55_644},
    9: {0: 61_408, 1: 39_380},
}
EXPECTED_ABORT_LED_COUNTS = {
    7: {0: 14_218, 1: 8_276},
    8: {0: 12_570, 1: 10_356},
    9: {0: 10_108, 1: 8_752},
}
EXPECTED_NO_CONTROL_ALL_ROWS = {7: 100_251, 8: 113_941, 9: 74_745}
EXPECTED_NO_CONTROL_ABORT_ROWS = {7: 16_389, 8: 17_725, 9: 14_001}
EXPECTED_NO_CONTROL_MISSING_TIMED_FIX = {7: 2, 8: 2, 9: 5}
EXPECTED_NO_CONTROL_LED_COUNTS = {
    7: {0: 66_226, 1: 34_025},
    8: {0: 70_892, 1: 43_049},
    9: {0: 45_124, 1: 29_621},
}

BIN_WIDTH_S = 0.020
TIMING_BINS = np.arange(0.0, 2.2 + BIN_WIDTH_S / 2, BIN_WIDTH_S)
ABORT_BINS = np.arange(-2.0, 2.0 + BIN_WIDTH_S / 2, BIN_WIDTH_S)
REGULAR_XLIM_S = (-0.4, 0.5)
ZOOM_XLIM_S = (-0.1, 0.3)

ALL_COLOR = "black"
TIMING_ABORT_COLOR = "tab:red"
LED_COLORS = {0: "tab:blue", 1: "tab:red"}
FIGURE_DPI = 220
AUDIT_CSV = SCRIPT_DIR / "led7_s7_s8_s9_animal_timing_abort_rates_audit.csv"


# %%
############ Reused histogram calculations ############
def unit_area_histogram(values, bins):
    counts, _ = np.histogram(values, bins=bins)
    if int(counts.sum()) != len(values):
        raise RuntimeError("Timing histogram omitted observations.")
    heights = counts / (len(values) * BIN_WIDTH_S)
    if not np.isclose(np.sum(heights * np.diff(bins)), 1.0, atol=1e-12):
        raise RuntimeError("Timing histogram area is not one.")
    return heights


def abort_rate_histogram(values, n_total_trials, bins):
    counts, _ = np.histogram(values, bins=bins)
    if int(counts.sum()) != len(values):
        raise RuntimeError("Abort histogram omitted finite aborts.")
    heights = counts / (n_total_trials * BIN_WIDTH_S)
    area = float(np.sum(heights * np.diff(bins)))
    expected_area = len(values) / n_total_trials
    if not np.isclose(area, expected_area, rtol=0, atol=1e-14):
        raise RuntimeError("Abort histogram area differs from finite abort fraction.")
    return heights, area


# %%
############ Read and validate only standardized CSVs ############
datasets = {}
required_columns = [
    "animal", "session_type", "training_level", "repeat_trial", "LED_trial",
    "abort_event", "timed_fix", "intended_fix", "LED_onset_time",
]

for session_type in SESSION_TYPES:
    all_path = RAW_DATA_DIR / f"LED7_s{session_type}.csv"
    abort_path = RAW_DATA_DIR / f"LED7_s{session_type}_aborts.csv"
    for path in (all_path, abort_path):
        if not path.exists():
            raise FileNotFoundError(f"Required standardized CSV not found: {path}")

    all_df = pd.read_csv(all_path, float_precision="round_trip")
    abort_df = pd.read_csv(abort_path, float_precision="round_trip")
    for label, frame in (("all", all_df), ("abort", abort_df)):
        if len(frame.columns) != EXPECTED_COLUMNS:
            raise RuntimeError(f"s{session_type} {label}: expected 52 columns.")
        missing_columns = [column for column in required_columns if column not in frame]
        if missing_columns:
            raise RuntimeError(
                f"s{session_type} {label}: missing columns {missing_columns}."
            )
        if frame.columns.tolist().count("LED_onset_time") != 1:
            raise RuntimeError(f"s{session_type} {label}: duplicate onset column.")
        if not frame["training_level"].eq(TRAINING_LEVEL).all():
            raise RuntimeError(f"s{session_type} {label}: unexpected training level.")
        if not frame["session_type"].eq(session_type).all():
            raise RuntimeError(f"s{session_type} {label}: unexpected session type.")
        if not (
            frame["repeat_trial"].isin(ALLOWED_REPEAT_TRIALS)
            | frame["repeat_trial"].isna()
        ).all():
            raise RuntimeError(f"s{session_type} {label}: unexpected repeat trial.")
        if not (
            frame["LED_trial"].isin(ALLOWED_LED_TRIALS)
            | frame["LED_trial"].isna()
        ).all():
            raise RuntimeError(f"s{session_type} {label}: unexpected LED trial.")
        if frame["animal"].isna().any():
            raise RuntimeError(f"s{session_type} {label}: missing animal.")
        if not np.isfinite(frame[["intended_fix", "LED_onset_time"]]).all(axis=None):
            raise RuntimeError(f"s{session_type} {label}: non-finite timing.")
        if (
            (frame["LED_onset_time"] < -1e-12)
            | (frame["LED_onset_time"] > frame["intended_fix"] + 1e-12)
        ).any():
            raise RuntimeError(f"s{session_type} {label}: onset outside fixation.")
        if frame.duplicated().any():
            raise RuntimeError(f"s{session_type} {label}: duplicate full rows.")

    if all_df.columns.tolist() != abort_df.columns.tolist():
        raise RuntimeError(f"s{session_type}: all and abort CSV schemas differ.")
    if len(all_df) != EXPECTED_ALL_ROWS[session_type]:
        raise RuntimeError(f"s{session_type}: unexpected all-trial row count.")
    if len(abort_df) != EXPECTED_ABORT_ROWS[session_type]:
        raise RuntimeError(f"s{session_type}: unexpected abort row count.")
    if not abort_df["abort_event"].eq(ABORT_EVENT).all():
        raise RuntimeError(f"s{session_type}: abort CSV contains other outcomes.")
    if all_df["LED_trial"].isna().any() or abort_df["LED_trial"].isna().any():
        raise RuntimeError(f"s{session_type}: unexpected missing LED trial.")

    all_led_counts = all_df["LED_trial"].value_counts().sort_index().to_dict()
    abort_led_counts = abort_df["LED_trial"].value_counts().sort_index().to_dict()
    if all_led_counts != EXPECTED_ALL_LED_COUNTS[session_type]:
        raise RuntimeError(f"s{session_type}: unexpected all-trial LED counts.")
    if abort_led_counts != EXPECTED_ABORT_LED_COUNTS[session_type]:
        raise RuntimeError(f"s{session_type}: unexpected abort LED counts.")
    if int(abort_df["timed_fix"].isna().sum()) != EXPECTED_MISSING_TIMED_FIX[session_type]:
        raise RuntimeError(f"s{session_type}: unexpected missing abort timing count.")

    expected_abort_df = all_df.loc[all_df["abort_event"].eq(ABORT_EVENT)]
    np.testing.assert_allclose(
        expected_abort_df.to_numpy(dtype=float),
        abort_df.to_numpy(dtype=float),
        rtol=0,
        atol=0,
        equal_nan=True,
        err_msg=f"s{session_type}: abort CSV does not match all-trial CSV rows",
    )
    animals_all = tuple(sorted(all_df["animal"].astype(int).unique()))
    animals_abort = tuple(sorted(abort_df["animal"].astype(int).unique()))
    if animals_all != ANIMALS or animals_abort != ANIMALS:
        raise RuntimeError(f"s{session_type}: unexpected animal cohort.")

    finite_abort_df = abort_df.loc[np.isfinite(abort_df["timed_fix"])]
    if not (finite_abort_df["timed_fix"] < finite_abort_df["intended_fix"]).all():
        raise RuntimeError(f"s{session_type}: finite event-3 abort after stimulus.")
    datasets[session_type] = {"all": all_df, "abort": abort_df}
    print(
        f"s{session_type}: {len(all_df):,} all trials, {len(abort_df):,} "
        f"event-3 aborts, {len(abort_df) - len(finite_abort_df)} missing timed_fix"
    )


# %%
############ Plot each animal and both trial-pooled aggregates ############
audit_rows = []
for cohort in COHORTS:
    is_aggregate = cohort in ("aggregate", "aggregate_no_controls")
    fig, axes = plt.subplots(3, 4, figsize=(19, 11), sharex="col")
    column_y_max = np.zeros(3, dtype=float)

    for row_index, session_type in enumerate(SESSION_TYPES):
        all_animal = datasets[session_type]["all"]
        abort_animal = datasets[session_type]["abort"]
        if cohort == "aggregate_no_controls":
            all_animal = all_animal.loc[all_animal["animal"].isin(EXPERIMENTAL_ANIMALS)]
            abort_animal = abort_animal.loc[abort_animal["animal"].isin(EXPERIMENTAL_ANIMALS)]
        elif not is_aggregate:
            all_animal = all_animal.loc[all_animal["animal"].eq(cohort)]
            abort_animal = abort_animal.loc[abort_animal["animal"].eq(cohort)]
        if all_animal.empty or abort_animal.empty:
            raise RuntimeError(f"Cohort {cohort} s{session_type}: empty dataset.")

        timing_ks = {}
        for col_index, field in enumerate(("intended_fix", "LED_onset_time")):
            all_values = all_animal[field].to_numpy(dtype=float)
            abort_values = abort_animal[field].to_numpy(dtype=float)
            all_heights = unit_area_histogram(all_values, TIMING_BINS)
            abort_heights = unit_area_histogram(abort_values, TIMING_BINS)
            timing_ks[field] = float(ks_2samp(all_values, abort_values).statistic)
            ax = axes[row_index, col_index]
            ax.stairs(all_heights, TIMING_BINS, color=ALL_COLOR, linewidth=1.35)
            ax.stairs(
                abort_heights,
                TIMING_BINS,
                color=TIMING_ABORT_COLOR,
                linewidth=1.35,
            )
            ax.text(
                0.97,
                0.94,
                f"all n = {len(all_values):,}\n"
                f"abort n = {len(abort_values):,}\n"
                f"KS = {timing_ks[field]:.3f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
            )
            column_y_max[col_index] = max(
                column_y_max[col_index],
                float(all_heights.max()),
                float(abort_heights.max()),
            )

        for led_trial in ALLOWED_LED_TRIALS:
            all_condition = all_animal.loc[all_animal["LED_trial"].eq(led_trial)]
            abort_condition = abort_animal.loc[abort_animal["LED_trial"].eq(led_trial)]
            if all_condition.empty:
                raise RuntimeError(
                    f"Cohort {cohort} s{session_type} LED {led_trial}: no trials."
                )
            finite_abort = abort_condition.loc[
                np.isfinite(abort_condition["timed_fix"])
            ]
            aligned_times = (
                finite_abort["timed_fix"] - finite_abort["LED_onset_time"]
            ).to_numpy(dtype=float)
            if not (
                (aligned_times >= ABORT_BINS[0])
                & (aligned_times <= ABORT_BINS[-1])
            ).all():
                raise RuntimeError(
                    f"Cohort {cohort} s{session_type} LED {led_trial}: "
                    "abort outside full histogram bins."
                )
            heights, area = abort_rate_histogram(
                aligned_times, len(all_condition), ABORT_BINS
            )
            for col_index in (2, 3):
                axes[row_index, col_index].stairs(
                    heights,
                    ABORT_BINS,
                    color=LED_COLORS[led_trial],
                    linewidth=1.4,
                )
            column_y_max[2] = max(column_y_max[2], float(heights.max()))
            audit_rows.append(
                {
                    "animal": cohort,
                    "session_type": session_type,
                    "LED_trial": led_trial,
                    "all_condition_trials": len(all_condition),
                    "coded_event3_aborts": len(abort_condition),
                    "finite_plotted_aborts": len(finite_abort),
                    "missing_timed_fix": len(abort_condition) - len(finite_abort),
                    "full_histogram_area": area,
                    "finite_abort_fraction": len(finite_abort) / len(all_condition),
                    "intended_fix_all_vs_abort_KS": timing_ks["intended_fix"],
                    "LED_onset_time_all_vs_abort_KS": timing_ks["LED_onset_time"],
                    "regular_xlim_left_s": REGULAR_XLIM_S[0],
                    "regular_xlim_right_s": REGULAR_XLIM_S[1],
                    "zoom_xlim_left_s": ZOOM_XLIM_S[0],
                    "zoom_xlim_right_s": ZOOM_XLIM_S[1],
                }
            )

        for col_index in (2, 3):
            axes[row_index, col_index].axvline(
                0, color="0.3", linestyle=":", linewidth=1.0
            )
            off_row, on_row = audit_rows[-2:]
            axes[row_index, col_index].text(
                0.03,
                0.94,
                f"OFF area = {off_row['full_histogram_area']:.3f}\n"
                f"ON area = {on_row['full_histogram_area']:.3f}",
                transform=axes[row_index, col_index].transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9},
            )

        axes[row_index, 0].set_ylabel(
            f"Session type {session_type}\nDensity (s$^{{-1}}$)", fontsize=10
        )
        axes[row_index, 2].set_ylabel("Abort rate (s$^{-1}$)", fontsize=10)

    column_titles = (
        "intended_fix",
        "LED_onset_time",
        "Abort rate from LED onset\n−0.4 to +0.5 s",
        "Abort rate from LED onset\n−0.1 to +0.3 s",
    )
    for col_index, title in enumerate(column_titles):
        axes[0, col_index].set_title(title, fontsize=12)
        for row_index in range(len(SESSION_TYPES)):
            ax = axes[row_index, col_index]
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.16, linewidth=0.6)
            ax.tick_params(labelsize=8)
            if col_index < 2:
                ax.set_xlim(TIMING_BINS[0], TIMING_BINS[-1])
                ax.set_ylim(0, 1.12 * column_y_max[col_index])
            else:
                ax.set_xlim(REGULAR_XLIM_S if col_index == 2 else ZOOM_XLIM_S)
                ax.set_ylim(0, 1.15 * column_y_max[2])

    for col_index in range(4):
        axes[-1, col_index].set_xlabel(
            "Time from fixation onset (s)"
            if col_index < 2
            else "Abort time from LED onset (s)",
            fontsize=10,
        )

    fig.legend(
        handles=[
            Line2D([0], [0], color=ALL_COLOR, lw=1.5, label="Timing: all trials"),
            Line2D(
                [0], [0], color=TIMING_ABORT_COLOR, lw=1.5,
                label="Timing: fixation aborts",
            ),
            Line2D(
                [0], [0], color=LED_COLORS[0], lw=1.5,
                label="Abort rate: LED OFF",
            ),
            Line2D(
                [0], [0], color=LED_COLORS[1], lw=1.5,
                label="Abort rate: LED ON",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.962),
        ncol=4,
        frameon=False,
        fontsize=9.5,
    )
    title = (
        "LED7 trial-pooled aggregate (no controls): session types 7, 8, and 9"
        if cohort == "aggregate_no_controls"
        else "LED7 trial-pooled aggregate: session types 7, 8, and 9"
        if cohort == "aggregate"
        else f"LED7 animal {cohort}: session types 7, 8, and 9"
    )
    fig.suptitle(title, fontsize=15)
    fig.text(
        0.5,
        0.012,
        "Timing curves each have area 1. Abort-rate areas use the full −2 to +2 s "
        "histogram and equal finite aborts / all LED-group trials. "
        "For LED OFF, onset is scheduled/counterfactual.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.025, 0.045, 1, 0.91), w_pad=1.15, h_pad=1.2)
    output_name = (
        "led7_aggregate_no_controls_s7_s8_s9_timing_abort_rate_3x4.png"
        if cohort == "aggregate_no_controls"
        else "led7_aggregate_s7_s8_s9_timing_abort_rate_3x4.png"
        if cohort == "aggregate"
        else f"led7_animal_{cohort}_s7_s8_s9_timing_abort_rate_3x4.png"
    )
    output_path = SCRIPT_DIR / output_name
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {cohort}: {output_path}")


# %%
############ Export and print numerical audit ############
audit_df = pd.DataFrame(audit_rows)
if len(audit_df) != len(COHORTS) * len(SESSION_TYPES) * len(ALLOWED_LED_TRIALS):
    raise RuntimeError("Audit table has an unexpected number of rows.")
if audit_df.duplicated(["animal", "session_type", "LED_trial"]).any():
    raise RuntimeError("Audit table has duplicate animal/session/LED groups.")
for session_type in SESSION_TYPES:
    session_audit = audit_df.loc[
        audit_df["session_type"].eq(session_type)
        & audit_df["animal"].isin(ANIMALS)
    ]
    if int(session_audit["all_condition_trials"].sum()) != EXPECTED_ALL_ROWS[session_type]:
        raise RuntimeError(f"s{session_type}: per-animal all-trial counts do not sum.")
    if int(session_audit["coded_event3_aborts"].sum()) != EXPECTED_ABORT_ROWS[session_type]:
        raise RuntimeError(f"s{session_type}: per-animal abort counts do not sum.")
    if int(session_audit["missing_timed_fix"].sum()) != EXPECTED_MISSING_TIMED_FIX[session_type]:
        raise RuntimeError(f"s{session_type}: missing-timing counts do not sum.")
    for led_trial in ALLOWED_LED_TRIALS:
        animal_groups = session_audit.loc[session_audit["LED_trial"].eq(led_trial)]
        aggregate_group = audit_df.loc[
            audit_df["animal"].eq("aggregate")
            & audit_df["session_type"].eq(session_type)
            & audit_df["LED_trial"].eq(led_trial)
        ].iloc[0]
        for count_column in (
            "all_condition_trials",
            "coded_event3_aborts",
            "finite_plotted_aborts",
            "missing_timed_fix",
        ):
            if int(animal_groups[count_column].sum()) != int(aggregate_group[count_column]):
                raise RuntimeError(
                    f"s{session_type} LED {led_trial}: pooled {count_column} "
                    "does not equal the animal sum."
                )

    no_control = audit_df.loc[
        audit_df["session_type"].eq(session_type)
        & audit_df["animal"].eq("aggregate_no_controls")
    ]
    if int(no_control["all_condition_trials"].sum()) != EXPECTED_NO_CONTROL_ALL_ROWS[session_type]:
        raise RuntimeError(f"s{session_type}: no-control trial count changed.")
    if int(no_control["coded_event3_aborts"].sum()) != EXPECTED_NO_CONTROL_ABORT_ROWS[session_type]:
        raise RuntimeError(f"s{session_type}: no-control abort count changed.")
    if int(no_control["missing_timed_fix"].sum()) != EXPECTED_NO_CONTROL_MISSING_TIMED_FIX[session_type]:
        raise RuntimeError(f"s{session_type}: no-control missing-timing count changed.")
    for led_trial in ALLOWED_LED_TRIALS:
        group = no_control.loc[no_control["LED_trial"].eq(led_trial)].iloc[0]
        if int(group["all_condition_trials"]) != EXPECTED_NO_CONTROL_LED_COUNTS[session_type][led_trial]:
            raise RuntimeError(f"s{session_type} LED {led_trial}: no-control count changed.")
        animal_groups = session_audit.loc[
            session_audit["animal"].isin(EXPERIMENTAL_ANIMALS)
            & session_audit["LED_trial"].eq(led_trial)
        ]
        for count_column in (
            "all_condition_trials", "coded_event3_aborts",
            "finite_plotted_aborts", "missing_timed_fix",
        ):
            if int(animal_groups[count_column].sum()) != int(group[count_column]):
                raise RuntimeError(
                    f"s{session_type} LED {led_trial}: no-control {count_column} "
                    "does not equal the six-animal sum."
                )

audit_df.to_csv(AUDIT_CSV, index=False)
print("\nPer-animal and trial-pooled session/LED audit:")
print(audit_df.to_string(index=False))
print(f"Saved audit: {AUDIT_CSV}")
