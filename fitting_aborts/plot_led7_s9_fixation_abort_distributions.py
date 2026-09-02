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
OUTPUT_FIGURE_PATH = SCRIPT_DIR / "led7_s9_fixation_abort_distributions_4x9.png"

TRAINING_LEVEL = 16
SESSION_TYPE = 9
ALLOWED_REPEAT_TRIALS = {0, 2}
EXPECTED_ANIMALS = [90, 92, 93, 98, 99, 100, 102, 103]

EXPECTED_ROWS = 100_788
EXPECTED_LED_TRIAL_COUNTS = {0: 61_408, 1: 39_380}
EXPECTED_ABORT_EVENT_3_COUNTS = {0: 10_108, 1: 8_752}
EXPECTED_FINITE_ABORT_COUNTS = {0: 10_107, 1: 8_747}

BIN_WIDTH_S = 0.020
FIXATION_TIME_MIN_S = 0.0
FIXATION_TIME_MAX_S = 2.0
LED_ALIGNED_MIN_S = -1.1
LED_ALIGNED_MAX_S = 1.1
LED_ALIGNED_ZOOM_MIN_S = -0.1
LED_ALIGNED_ZOOM_MAX_S = 0.5

FIXATION_TIME_BINS = np.arange(
    FIXATION_TIME_MIN_S,
    FIXATION_TIME_MAX_S + BIN_WIDTH_S / 2,
    BIN_WIDTH_S,
)
LED_ALIGNED_BINS = np.arange(
    LED_ALIGNED_MIN_S,
    LED_ALIGNED_MAX_S + BIN_WIDTH_S / 2,
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
led_values = set(df["LED_trial"].astype(int).unique())
if led_values != {0, 1}:
    raise RuntimeError(f"Expected LED_trial values {{0, 1}}, found {sorted(led_values)}.")

if df["animal"].isna().any():
    raise RuntimeError("animal contains missing values.")
df["animal"] = df["animal"].astype(int)
df["LED_trial"] = df["LED_trial"].astype(int)

animals = sorted(df["animal"].unique().tolist())
if animals != EXPECTED_ANIMALS:
    raise RuntimeError(f"Expected animals {EXPECTED_ANIMALS}, found {animals}.")

led_trial_counts = df["LED_trial"].value_counts().sort_index().to_dict()
if led_trial_counts != EXPECTED_LED_TRIAL_COUNTS:
    raise RuntimeError(
        f"Expected LED trial counts {EXPECTED_LED_TRIAL_COUNTS}, found {led_trial_counts}."
    )


# %%
############ Confirm the fixation-abort definition ############
df["RTwrtStim"] = df["timed_fix"] - df["intended_fix"]

event_3_mask = df["abort_event"].eq(3)
finite_abort_timing_mask = event_3_mask & np.isfinite(
    df[["timed_fix", "intended_fix", "LED_onset_time"]]
).all(axis=1)

event_3_df = df.loc[event_3_mask].copy()
finite_abort_df = df.loc[finite_abort_timing_mask].copy()

if len(event_3_df) != sum(EXPECTED_ABORT_EVENT_3_COUNTS.values()):
    raise RuntimeError(
        f"Expected {sum(EXPECTED_ABORT_EVENT_3_COUNTS.values()):,} event-3 aborts, "
        f"found {len(event_3_df):,}."
    )
if len(finite_abort_df) != sum(EXPECTED_FINITE_ABORT_COUNTS.values()):
    raise RuntimeError(
        f"Expected {sum(EXPECTED_FINITE_ABORT_COUNTS.values()):,} finite event-3 aborts, "
        f"found {len(finite_abort_df):,}."
    )

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

if not (finite_abort_df["RTwrtStim"] < 0).all():
    raise RuntimeError("Some finite event-3 aborts do not have RTwrtStim < 0.")
if not (finite_abort_df["timed_fix"] < finite_abort_df["intended_fix"]).all():
    raise RuntimeError("Some finite event-3 aborts do not end before intended fixation.")

finite_negative_rt_mask = np.isfinite(df["RTwrtStim"]) & df["RTwrtStim"].lt(0)
if not df.loc[finite_negative_rt_mask, "abort_event"].eq(3).all():
    raise RuntimeError("Some finite negative-RT trials are not coded abort_event == 3.")

if not np.isfinite(df[["intended_fix", "LED_onset_time"]]).all(axis=None):
    raise RuntimeError("intended_fix or LED_onset_time contains a non-finite value.")
if not (df["LED_onset_time"] > 0).all():
    raise RuntimeError("LED_onset_time contains a non-positive value.")
if not (df["LED_onset_time"] < df["intended_fix"]).all():
    raise RuntimeError("Some raw LED_onset_time values do not precede intended_fix.")

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
############ Build rate-scaled histograms and audit table ############
def abort_rate_histogram(values, n_total_trials, bins):
    counts, _ = np.histogram(values, bins=bins)
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
        n_missing_abort_timing = n_coded_aborts - n_plotted_aborts

        fixation_times = plotted_abort_df["timed_fix"].to_numpy(dtype=float)

        # LED_onset_time in this MAT-derived session-9 export is already in the
        # fixation-time coordinate. Do not apply the legacy out_LED.csv transform.
        abort_times_wrt_led = (
            plotted_abort_df["timed_fix"] - plotted_abort_df["LED_onset_time"]
        ).to_numpy(dtype=float)

        fixation_in_range = (
            (fixation_times >= FIXATION_TIME_BINS[0])
            & (fixation_times <= FIXATION_TIME_BINS[-1])
        )
        led_aligned_in_range = (
            (abort_times_wrt_led >= LED_ALIGNED_BINS[0])
            & (abort_times_wrt_led <= LED_ALIGNED_BINS[-1])
        )
        if not fixation_in_range.all():
            raise RuntimeError(
                f"{group_label} {LED_LABELS[led_trial]} has fixation aborts outside "
                f"[{FIXATION_TIME_BINS[0]}, {FIXATION_TIME_BINS[-1]}] s."
            )
        if not led_aligned_in_range.all():
            raise RuntimeError(
                f"{group_label} {LED_LABELS[led_trial]} has LED-aligned aborts outside "
                f"[{LED_ALIGNED_BINS[0]}, {LED_ALIGNED_BINS[-1]}] s."
            )

        fixation_counts, fixation_heights = abort_rate_histogram(
            fixation_times,
            n_total_trials,
            FIXATION_TIME_BINS,
        )
        led_aligned_counts, led_aligned_heights = abort_rate_histogram(
            abort_times_wrt_led,
            n_total_trials,
            LED_ALIGNED_BINS,
        )

        expected_area = n_plotted_aborts / n_total_trials
        fixation_area = float(fixation_heights.sum() * BIN_WIDTH_S)
        led_aligned_area = float(led_aligned_heights.sum() * BIN_WIDTH_S)
        if not np.isclose(fixation_area, expected_area, rtol=0, atol=1e-14):
            raise RuntimeError(
                f"Fixation histogram area mismatch for {group_label} "
                f"{LED_LABELS[led_trial]}."
            )
        if not np.isclose(led_aligned_area, expected_area, rtol=0, atol=1e-14):
            raise RuntimeError(
                f"LED-aligned histogram area mismatch for {group_label} "
                f"{LED_LABELS[led_trial]}."
            )

        panel_results[(group_label, led_trial)] = {
            "fixation_counts": fixation_counts,
            "fixation_heights": fixation_heights,
            "led_aligned_counts": led_aligned_counts,
            "led_aligned_heights": led_aligned_heights,
            "n_total_trials": n_total_trials,
            "n_coded_aborts": n_coded_aborts,
            "n_plotted_aborts": n_plotted_aborts,
            "n_missing_abort_timing": n_missing_abort_timing,
            "plotted_abort_fraction": expected_area,
        }

        global_y_max = max(
            global_y_max,
            float(fixation_heights.max(initial=0.0)),
            float(led_aligned_heights.max(initial=0.0)),
        )

        audit_rows.append(
            {
                "group": group_label,
                "condition": LED_LABELS[led_trial],
                "n_total_trials": n_total_trials,
                "n_abort_event_3": n_coded_aborts,
                "n_plotted_aborts": n_plotted_aborts,
                "n_missing_abort_timing": n_missing_abort_timing,
                "coded_abort_fraction": n_coded_aborts / n_total_trials,
                "plotted_abort_fraction": expected_area,
                "fixation_histogram_area": fixation_area,
                "led_aligned_histogram_area": led_aligned_area,
                "fixation_time_min_s": float(fixation_times.min()),
                "fixation_time_max_s": float(fixation_times.max()),
                "led_aligned_min_s": float(abort_times_wrt_led.min()),
                "led_aligned_max_s": float(abort_times_wrt_led.max()),
            }
        )

audit_df = pd.DataFrame(audit_rows)


# %%
############ Verify pooled aggregate counts and histogram sums ############
count_columns = [
    "n_total_trials",
    "n_abort_event_3",
    "n_plotted_aborts",
    "n_missing_abort_timing",
]

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

    for column in count_columns:
        animal_sum = int(animal_audit[column].sum())
        aggregate_value = int(aggregate_audit[column])
        if animal_sum != aggregate_value:
            raise RuntimeError(
                f"Aggregate {column} mismatch for {condition_label}: "
                f"animal sum={animal_sum}, aggregate={aggregate_value}."
            )

    animal_fixation_counts = sum(
        panel_results[(f"Animal {animal}", led_trial)]["fixation_counts"]
        for animal in animals
    )
    animal_led_aligned_counts = sum(
        panel_results[(f"Animal {animal}", led_trial)]["led_aligned_counts"]
        for animal in animals
    )
    np.testing.assert_array_equal(
        animal_fixation_counts,
        panel_results[("Aggregate", led_trial)]["fixation_counts"],
    )
    np.testing.assert_array_equal(
        animal_led_aligned_counts,
        panel_results[("Aggregate", led_trial)]["led_aligned_counts"],
    )


# %%
############ Plot the 4 x (N animals + aggregate) grid ############
n_columns = len(group_definitions)
fig, axes = plt.subplots(
    4,
    n_columns,
    figsize=(3.15 * n_columns, 12.4),
    sharex="row",
    sharey=True,
)

y_max = 1.16 * global_y_max if global_y_max > 0 else 1.0

for column_index, (group_label, _) in enumerate(group_definitions):
    off_result = panel_results[(group_label, 0)]
    on_result = panel_results[(group_label, 1)]

    axes[0, column_index].stairs(
        off_result["fixation_heights"],
        FIXATION_TIME_BINS,
        color=LED_COLORS[0],
        linewidth=1.5,
        fill=False,
    )
    axes[1, column_index].stairs(
        on_result["fixation_heights"],
        FIXATION_TIME_BINS,
        color=LED_COLORS[1],
        linewidth=1.5,
        fill=False,
    )
    for row_index in [2, 3]:
        axes[row_index, column_index].stairs(
            off_result["led_aligned_heights"],
            LED_ALIGNED_BINS,
            color=LED_COLORS[0],
            linewidth=1.5,
            label=LED_LABELS[0],
        )
        axes[row_index, column_index].stairs(
            on_result["led_aligned_heights"],
            LED_ALIGNED_BINS,
            color=LED_COLORS[1],
            linewidth=1.5,
            label=LED_LABELS[1],
        )
        axes[row_index, column_index].axvline(
            0.0,
            color="black",
            linestyle="--",
            linewidth=1.0,
            alpha=0.6,
        )

    axes[0, column_index].set_title(group_label, fontsize=11)

    off_annotation = (
        f"a={off_result['n_plotted_aborts']:,}/{off_result['n_total_trials']:,}\n"
        f"area={off_result['plotted_abort_fraction']:.3f}"
    )
    on_annotation = (
        f"a={on_result['n_plotted_aborts']:,}/{on_result['n_total_trials']:,}\n"
        f"area={on_result['plotted_abort_fraction']:.3f}"
    )
    axes[0, column_index].text(
        0.98,
        0.96,
        off_annotation,
        transform=axes[0, column_index].transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
    )
    axes[1, column_index].text(
        0.98,
        0.96,
        on_annotation,
        transform=axes[1, column_index].transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
    )
    for row_index in [2, 3]:
        axes[row_index, column_index].text(
            0.98,
            0.96,
            f"OFF area={off_result['plotted_abort_fraction']:.3f}\n"
            f"ON area={on_result['plotted_abort_fraction']:.3f}",
            transform=axes[row_index, column_index].transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
        )

    axes[0, column_index].set_xlim(FIXATION_TIME_MIN_S, FIXATION_TIME_MAX_S)
    axes[1, column_index].set_xlim(FIXATION_TIME_MIN_S, FIXATION_TIME_MAX_S)
    axes[2, column_index].set_xlim(LED_ALIGNED_MIN_S, LED_ALIGNED_MAX_S)
    axes[3, column_index].set_xlim(
        LED_ALIGNED_ZOOM_MIN_S,
        LED_ALIGNED_ZOOM_MAX_S,
    )

    for row_index in range(4):
        axes[row_index, column_index].set_ylim(0.0, y_max)
        axes[row_index, column_index].grid(axis="y", alpha=0.18, linewidth=0.6)
        axes[row_index, column_index].tick_params(labelsize=8)

    axes[0, column_index].set_xlabel("Fixation time from onset (s)", fontsize=8)
    axes[1, column_index].set_xlabel("Fixation time from onset (s)", fontsize=8)
    axes[2, column_index].set_xlabel("Abort time relative to LED onset (s)", fontsize=8)
    axes[3, column_index].set_xlabel("Abort time relative to LED onset (s)", fontsize=8)

axes[0, 0].set_ylabel("LED OFF\nAbort probability density (s$^{-1}$)")
axes[1, 0].set_ylabel("LED ON\nAbort probability density (s$^{-1}$)")
axes[2, 0].set_ylabel("LED OFF / ON\nAbort probability density (s$^{-1}$)")
axes[3, 0].set_ylabel("LED OFF / ON\nAbort probability density (s$^{-1}$)")

axes[2, -1].legend(loc="upper left", fontsize=8, frameon=False)
axes[3, -1].legend(loc="upper left", fontsize=8, frameon=False)

fig.suptitle(
    "LED7 session 9 fixation-abort distributions (training level 16)\n"
    "20 ms bins; histogram area = finite plotted aborts / all trials in LED condition",
    fontsize=14,
    y=0.995,
)
fig.tight_layout(rect=(0.01, 0.01, 1.0, 0.965))
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
print(f"Event-3 aborts: {len(event_3_df):,}")
print(f"Finite event-3 aborts: {len(finite_abort_df):,}")
print(
    "Confirmed: every finite event-3 abort has RTwrtStim < 0 and "
    "timed_fix < intended_fix."
)
print("Confirmed: every finite negative-RT trial has abort_event == 3.")
print("Confirmed: every raw LED_onset_time is positive and precedes intended_fix.")
print(f"Missing event-3 timed_fix counts: {missing_abort_timing_counts}")
print("\nHistogram audit")
print(audit_df.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
print(f"\nSaved figure: {OUTPUT_FIGURE_PATH}")
