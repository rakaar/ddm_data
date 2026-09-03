# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# %%
############ Parameters (edit here) ############
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

INPUT_CSV_PATH = REPO_ROOT / "raw_data" / "LED7_s9.csv"
OUTPUT_FIGURE_PATH = SCRIPT_DIR / "led7_s9_fixation_abort_two_alignments_2x9.png"

TRAINING_LEVEL = 16
SESSION_TYPE = 9
ALLOWED_REPEAT_TRIALS = {0, 2}
EXPECTED_ANIMALS = [90, 92, 93, 98, 99, 100, 102, 103]

EXPECTED_ROWS = 100_788
EXPECTED_LED_TRIAL_COUNTS = {0: 61_408, 1: 39_380}
EXPECTED_ABORT_EVENT_3_COUNTS = {0: 10_108, 1: 8_752}
EXPECTED_FINITE_ABORT_COUNTS = {0: 10_107, 1: 8_747}

BIN_WIDTH_S = 0.020
FULL_MIN_S = -1.1
FULL_MAX_S = 1.1
ZOOM_MIN_S = -1.0
ZOOM_MAX_S = 1.0
FULL_BINS = np.arange(
    FULL_MIN_S,
    FULL_MAX_S + BIN_WIDTH_S / 2,
    BIN_WIDTH_S,
)

LED_COLORS = {0: "tab:blue", 1: "tab:red"}
LED_LABELS = {0: "LED OFF", 1: "LED ON"}

OUTPUT_DPI = 250
SHOW_PLOT = False


# %%
############ Load and validate the session-9 export ############
if not INPUT_CSV_PATH.exists():
    raise FileNotFoundError(f"Could not find input CSV: {INPUT_CSV_PATH}")

df = pd.read_csv(INPUT_CSV_PATH)

required_columns = [
    "animal",
    "training_level",
    "session_type",
    "repeat_trial",
    "LED_trial",
    "abort_event",
    "timed_fix",
    "intended_fix",
    "LED_onset_time",
]
missing_columns = [column for column in required_columns if column not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns in {INPUT_CSV_PATH}: {missing_columns}")

if len(df) != EXPECTED_ROWS:
    raise RuntimeError(f"Expected {EXPECTED_ROWS:,} rows, found {len(df):,}.")
if not df["training_level"].eq(TRAINING_LEVEL).all():
    raise RuntimeError(f"Input contains rows outside training_level == {TRAINING_LEVEL}.")
if not df["session_type"].eq(SESSION_TYPE).all():
    raise RuntimeError(f"Input contains rows outside session_type == {SESSION_TYPE}.")

repeat_values = set(df["repeat_trial"].dropna().astype(int).unique())
if not repeat_values.issubset(ALLOWED_REPEAT_TRIALS):
    raise RuntimeError(f"Unexpected repeat_trial values: {sorted(repeat_values)}")
if df["LED_trial"].isna().any():
    raise RuntimeError("LED_trial contains missing values; expected only 0 and 1.")
if df["animal"].isna().any():
    raise RuntimeError("animal contains missing values.")

df["animal"] = df["animal"].astype(int)
df["LED_trial"] = df["LED_trial"].astype(int)

animals = sorted(df["animal"].unique().tolist())
if animals != EXPECTED_ANIMALS:
    raise RuntimeError(f"Expected animals {EXPECTED_ANIMALS}, found {animals}.")

led_values = set(df["LED_trial"].unique())
if led_values != {0, 1}:
    raise RuntimeError(f"Expected LED_trial values {{0, 1}}, found {sorted(led_values)}.")
led_trial_counts = df["LED_trial"].value_counts().sort_index().to_dict()
if led_trial_counts != EXPECTED_LED_TRIAL_COUNTS:
    raise RuntimeError(
        f"Expected LED trial counts {EXPECTED_LED_TRIAL_COUNTS}, found {led_trial_counts}."
    )


# %%
############ Confirm the fixation-abort subset ############
event_3_df = df.loc[df["abort_event"].eq(3)].copy()
finite_abort_df = event_3_df.loc[
    np.isfinite(event_3_df[["timed_fix", "intended_fix", "LED_onset_time"]]).all(
        axis=1
    )
].copy()

event_3_counts = (
    event_3_df["LED_trial"].value_counts().sort_index().astype(int).to_dict()
)
finite_abort_counts = (
    finite_abort_df["LED_trial"].value_counts().sort_index().astype(int).to_dict()
)
if event_3_counts != EXPECTED_ABORT_EVENT_3_COUNTS:
    raise RuntimeError(
        f"Expected event-3 counts {EXPECTED_ABORT_EVENT_3_COUNTS}, found {event_3_counts}."
    )
if finite_abort_counts != EXPECTED_FINITE_ABORT_COUNTS:
    raise RuntimeError(
        f"Expected finite abort counts {EXPECTED_FINITE_ABORT_COUNTS}, "
        f"found {finite_abort_counts}."
    )

finite_abort_df["RTwrtStim"] = (
    finite_abort_df["timed_fix"] - finite_abort_df["intended_fix"]
)
if not (finite_abort_df["RTwrtStim"] < 0).all():
    raise RuntimeError("Some finite event-3 aborts do not have RTwrtStim < 0.")

missing_abort_timing_counts = (
    event_3_df.loc[~np.isfinite(event_3_df["timed_fix"]), "LED_trial"]
    .value_counts()
    .reindex([0, 1], fill_value=0)
    .astype(int)
    .to_dict()
)
if missing_abort_timing_counts != {0: 1, 1: 5}:
    raise RuntimeError(
        "Expected one LED-OFF and five LED-ON event-3 aborts with missing timed_fix; "
        f"found {missing_abort_timing_counts}."
    )


# %%
############ Build both aligned histograms and audit table ############
def abort_rate_histogram(values, n_total_trials):
    counts, _ = np.histogram(values, bins=FULL_BINS)
    heights = counts.astype(float) / (n_total_trials * BIN_WIDTH_S)
    return counts, heights


group_definitions = [
    (f"Animal {animal}", df.loc[df["animal"].eq(animal)].copy())
    for animal in animals
]
group_definitions.append(("Aggregate", df.copy()))

panel_results = {}
audit_rows = []
global_y_max = 0.0

for group_label, group_df in group_definitions:
    for led_trial in [0, 1]:
        condition_df = group_df.loc[group_df["LED_trial"].eq(led_trial)].copy()
        coded_abort_df = condition_df.loc[condition_df["abort_event"].eq(3)].copy()
        plotted_abort_df = coded_abort_df.loc[
            np.isfinite(
                coded_abort_df[["timed_fix", "intended_fix", "LED_onset_time"]]
            ).all(axis=1)
        ].copy()

        n_total_trials = len(condition_df)
        n_coded_aborts = len(coded_abort_df)
        n_plotted_aborts = len(plotted_abort_df)

        aligned_to_led_onset = (
            plotted_abort_df["timed_fix"] - plotted_abort_df["LED_onset_time"]
        ).to_numpy(dtype=float)
        aligned_to_intended_minus_onset = (
            plotted_abort_df["timed_fix"]
            - (
                plotted_abort_df["intended_fix"]
                - plotted_abort_df["LED_onset_time"]
            )
        ).to_numpy(dtype=float)

        for alignment_label, values in [
            ("LED onset", aligned_to_led_onset),
            ("intended_fix - LED_onset_time", aligned_to_intended_minus_onset),
        ]:
            if not ((values >= FULL_BINS[0]) & (values <= FULL_BINS[-1])).all():
                raise RuntimeError(
                    f"{group_label} {LED_LABELS[led_trial]} has {alignment_label} "
                    f"values outside [{FULL_BINS[0]}, {FULL_BINS[-1]}] s."
                )

        led_onset_counts, led_onset_heights = abort_rate_histogram(
            aligned_to_led_onset,
            n_total_trials,
        )
        intended_minus_onset_counts, intended_minus_onset_heights = (
            abort_rate_histogram(
                aligned_to_intended_minus_onset,
                n_total_trials,
            )
        )

        expected_full_area = n_plotted_aborts / n_total_trials
        led_onset_full_area = float(led_onset_heights.sum() * BIN_WIDTH_S)
        intended_minus_onset_full_area = float(
            intended_minus_onset_heights.sum() * BIN_WIDTH_S
        )
        if not np.isclose(
            led_onset_full_area, expected_full_area, rtol=0, atol=1e-14
        ):
            raise RuntimeError(
                f"LED-onset histogram area mismatch for {group_label} "
                f"{LED_LABELS[led_trial]}."
            )
        if not np.isclose(
            intended_minus_onset_full_area,
            expected_full_area,
            rtol=0,
            atol=1e-14,
        ):
            raise RuntimeError(
                f"intended-minus-onset histogram area mismatch for {group_label} "
                f"{LED_LABELS[led_trial]}."
            )

        led_onset_in_zoom = (
            (aligned_to_led_onset >= ZOOM_MIN_S)
            & (aligned_to_led_onset <= ZOOM_MAX_S)
        )
        intended_minus_onset_in_zoom = (
            (aligned_to_intended_minus_onset >= ZOOM_MIN_S)
            & (aligned_to_intended_minus_onset <= ZOOM_MAX_S)
        )

        panel_results[(group_label, led_trial)] = {
            "led_onset_counts": led_onset_counts,
            "led_onset_heights": led_onset_heights,
            "intended_minus_onset_counts": intended_minus_onset_counts,
            "intended_minus_onset_heights": intended_minus_onset_heights,
            "n_total_trials": n_total_trials,
            "n_coded_aborts": n_coded_aborts,
            "n_plotted_aborts": n_plotted_aborts,
            "full_area": expected_full_area,
        }

        global_y_max = max(
            global_y_max,
            float(led_onset_heights.max(initial=0.0)),
            float(intended_minus_onset_heights.max(initial=0.0)),
        )

        audit_rows.append(
            {
                "group": group_label,
                "condition": LED_LABELS[led_trial],
                "n_total_trials": n_total_trials,
                "n_abort_event_3": n_coded_aborts,
                "n_plotted_aborts": n_plotted_aborts,
                "n_missing_timed_fix": n_coded_aborts - n_plotted_aborts,
                "full_abort_fraction": expected_full_area,
                "led_onset_full_hist_area": led_onset_full_area,
                "led_onset_zoom_fraction": float(
                    led_onset_in_zoom.sum() / n_total_trials
                ),
                "intended_minus_onset_full_hist_area": (
                    intended_minus_onset_full_area
                ),
                "intended_minus_onset_zoom_fraction": float(
                    intended_minus_onset_in_zoom.sum() / n_total_trials
                ),
                "led_onset_aligned_min_s": float(aligned_to_led_onset.min()),
                "led_onset_aligned_max_s": float(aligned_to_led_onset.max()),
                "intended_minus_onset_aligned_min_s": float(
                    aligned_to_intended_minus_onset.min()
                ),
                "intended_minus_onset_aligned_max_s": float(
                    aligned_to_intended_minus_onset.max()
                ),
            }
        )

audit_df = pd.DataFrame(audit_rows)


# %%
############ Verify the pooled aggregate against animal sums ############
for led_trial in [0, 1]:
    condition_label = LED_LABELS[led_trial]
    animal_audit = audit_df.loc[
        audit_df["group"].ne("Aggregate")
        & audit_df["condition"].eq(condition_label)
    ]
    aggregate_audit = audit_df.loc[
        audit_df["group"].eq("Aggregate")
        & audit_df["condition"].eq(condition_label)
    ].iloc[0]

    for column in [
        "n_total_trials",
        "n_abort_event_3",
        "n_plotted_aborts",
        "n_missing_timed_fix",
    ]:
        if int(animal_audit[column].sum()) != int(aggregate_audit[column]):
            raise RuntimeError(f"Aggregate {column} mismatch for {condition_label}.")

    for count_key in ["led_onset_counts", "intended_minus_onset_counts"]:
        animal_count_sum = sum(
            panel_results[(f"Animal {animal}", led_trial)][count_key]
            for animal in animals
        )
        np.testing.assert_array_equal(
            animal_count_sum,
            panel_results[("Aggregate", led_trial)][count_key],
        )


# %%
############ Plot the 2 x (N animals + aggregate) grid ############
n_columns = len(group_definitions)
fig, axes = plt.subplots(
    2,
    n_columns,
    figsize=(3.15 * n_columns, 6.8),
    sharex=True,
    sharey=True,
)

y_max = 1.16 * global_y_max if global_y_max > 0 else 1.0

for column_index, (group_label, _) in enumerate(group_definitions):
    for led_trial in [0, 1]:
        result = panel_results[(group_label, led_trial)]
        axes[0, column_index].stairs(
            result["led_onset_heights"],
            FULL_BINS,
            color=LED_COLORS[led_trial],
            linewidth=1.5,
            label=LED_LABELS[led_trial],
        )
        axes[1, column_index].stairs(
            result["intended_minus_onset_heights"],
            FULL_BINS,
            color=LED_COLORS[led_trial],
            linewidth=1.5,
            label=LED_LABELS[led_trial],
        )

    for row_index in [0, 1]:
        axes[row_index, column_index].axvline(
            0.0,
            color="black",
            linestyle="--",
            linewidth=1.0,
            alpha=0.6,
        )
        axes[row_index, column_index].set_xlim(ZOOM_MIN_S, ZOOM_MAX_S)
        axes[row_index, column_index].set_ylim(0.0, y_max)
        axes[row_index, column_index].grid(axis="y", alpha=0.18, linewidth=0.6)
        axes[row_index, column_index].tick_params(labelsize=8)

        off_result = panel_results[(group_label, 0)]
        on_result = panel_results[(group_label, 1)]
        axes[row_index, column_index].text(
            0.98,
            0.96,
            f"OFF area={off_result['full_area']:.3f}\n"
            f"ON area={on_result['full_area']:.3f}",
            transform=axes[row_index, column_index].transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
        )

    axes[0, column_index].set_title(group_label, fontsize=11)
    axes[0, column_index].set_xlabel(
        "Abort time relative to LED onset (s)",
        fontsize=8,
    )
    axes[1, column_index].set_xlabel(
        "Abort time relative to intended_fix - LED_onset_time (s)",
        fontsize=8,
    )

axes[0, 0].set_ylabel("LED OFF / ON\nAbort probability density (s$^{-1}$)")
axes[1, 0].set_ylabel("LED OFF / ON\nAbort probability density (s$^{-1}$)")

axes[0, -1].legend(loc="upper left", fontsize=8, frameon=False)
axes[1, -1].legend(loc="upper left", fontsize=8, frameon=False)

fig.suptitle(
    "LED7 session 9 fixation aborts: two timing alignments (training level 16)\n"
    "20 ms bins; full histogram area = finite plotted aborts / all trials "
    "in LED condition",
    fontsize=14,
    y=0.995,
)
fig.tight_layout(rect=(0.01, 0.01, 1.0, 0.94))
fig.savefig(OUTPUT_FIGURE_PATH, dpi=OUTPUT_DPI, bbox_inches="tight")

if SHOW_PLOT:
    plt.show()
else:
    plt.close(fig)


# %%
############ Print the numerical audit ############
print(f"Loaded: {INPUT_CSV_PATH}")
print(f"Rows: {len(df):,}")
print(f"Animals: {animals}")
print(f"Finite event-3 aborts: {len(finite_abort_df):,}")
print(f"Missing event-3 timed_fix counts: {missing_abort_timing_counts}")
print(
    "Top coordinate: timed_fix - LED_onset_time.\n"
    "Bottom coordinate: timed_fix - (intended_fix - LED_onset_time)."
)
print(
    f"Both rows show xlim [{ZOOM_MIN_S:.1f}, {ZOOM_MAX_S:.1f}] s from "
    f"full [{FULL_MIN_S:.1f}, {FULL_MAX_S:.1f}] s histograms."
)
print("\nHistogram audit")
print(audit_df.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
print(f"\nSaved figure: {OUTPUT_FIGURE_PATH}")
