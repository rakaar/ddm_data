# %%
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

try:
    from matio import load_from_mat
except ImportError as exc:
    raise ImportError(
        "This script needs the isolated mat-io decoder. Run it as:\n"
        "env PYTHONPATH=/tmp/codex_mat_io .venv/bin/python "
        "fitting_aborts/plot_led7_s7_s8_all_vs_nonzero_derived_onset_4x5.py"
    ) from exc


# %%
############ Parameters (edit here) ############
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

INPUT_MAT_PATH = REPO_ROOT / "raw_data" / "outMatrix_LED7_latest.mat"
TABLE_NAME = "totalout_stGtACRII"
OUTPUT_FIGURE_PATH = (
    SCRIPT_DIR / "led7_s7_s8_all_vs_nonzero_derived_onset_4x5.png"
)

TRAINING_LEVEL = 16
ALLOWED_REPEAT_TRIALS = {0, 2}
LED_COLORS = {0: "tab:blue", 1: "tab:red"}
LED_LABELS = {0: "LED OFF", 1: "LED ON"}

BIN_WIDTH_S = 0.020
INTENDED_FIX_BINS = np.arange(0, 2200 + 20, 20, dtype=float) / 1000
LED_ONSET_BINS = np.arange(0, 1200 + 20, 20, dtype=float) / 1000
DERIVED_ONSET_BINS = np.arange(0, 2200 + 20, 20, dtype=float) / 1000
ALIGNED_FULL_BINS = np.arange(-2000, 2000 + 20, 20, dtype=float) / 1000
ALIGNED_DISPLAY_LIMITS_S = (-1.0, 1.0)

OUTPUT_DPI = 250
SHOW_PLOT = False

SESSION_EXPECTATIONS = {
    7: {
        "animals": [90, 92, 93, 98, 99, 100, 102, 103],
        "condition_trials": {0: 92_057, 1: 46_383},
        "zero_derived_onset_on_trials": 3_406,
        "coded_aborts": {0: 14_218, 1: 8_276},
        "finite_aborts": {0: 14_215, 1: 8_276},
        "nonzero_on_coded_aborts": 8_102,
        "nonzero_on_finite_aborts": 8_102,
    },
    8: {
        "animals": [90, 92, 93, 98, 99, 100, 102, 103],
        "condition_trials": {0: 92_784, 1: 55_644},
        "zero_derived_onset_on_trials": 4_073,
        "coded_aborts": {0: 12_570, 1: 10_356},
        "finite_aborts": {0: 12_570, 1: 10_354},
        "nonzero_on_coded_aborts": 9_918,
        "nonzero_on_finite_aborts": 9_916,
    },
}

ROW_SPECS = [
    {
        "key": "s7_all",
        "session_type": 7,
        "exclude_zero_onset_from_on": False,
        "row_label": "LED7 session type 7\nall eligible trials",
    },
    {
        "key": "s8_all",
        "session_type": 8,
        "exclude_zero_onset_from_on": False,
        "row_label": "LED7 session type 8\nall eligible trials",
    },
    {
        "key": "s7_nonzero",
        "session_type": 7,
        "exclude_zero_onset_from_on": True,
        "row_label": "LED7 session type 7\nzero-onset ON removed",
    },
    {
        "key": "s8_nonzero",
        "session_type": 8,
        "exclude_zero_onset_from_on": True,
        "row_label": "LED7 session type 8\nzero-onset ON removed",
    },
]


# %%
############ Histogram helper ############
def scaled_histogram(values, bin_edges, normalization_count):
    counts, _ = np.histogram(values, bins=bin_edges)
    if int(counts.sum()) != len(values):
        raise RuntimeError(
            f"Histogram range contains {counts.sum():,} of {len(values):,} values."
        )

    heights = counts.astype(float) / (normalization_count * BIN_WIDTH_S)
    area = float(np.sum(heights * np.diff(bin_edges)))
    return counts, heights, area


# %%
############ Load and filter LED7 session types 7 and 8 ############
if not INPUT_MAT_PATH.exists():
    raise FileNotFoundError(f"Could not find input MAT file: {INPUT_MAT_PATH}")

mat_contents = load_from_mat(INPUT_MAT_PATH, variable_names=[TABLE_NAME])
if TABLE_NAME not in mat_contents:
    raise KeyError(f"{TABLE_NAME!r} is absent from {INPUT_MAT_PATH}.")

raw_df = mat_contents[TABLE_NAME]
if not isinstance(raw_df, pd.DataFrame):
    raise TypeError(
        f"Expected {TABLE_NAME!r} to decode as a DataFrame; "
        f"found {type(raw_df).__name__}."
    )

required_columns = [
    "animal",
    "session_type",
    "training_level",
    "repeat_trial",
    "LED_trial",
    "abort_event",
    "timed_fix",
    "intended_fix",
    "LED_onset_time",
]
missing_columns = [
    column for column in required_columns if column not in raw_df.columns
]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

training_df = raw_df.loc[
    raw_df["training_level"].eq(TRAINING_LEVEL)
].copy()

session_frames = {}
for session_type, expected in SESSION_EXPECTATIONS.items():
    session_df = training_df.loc[
        training_df["session_type"].eq(session_type)
    ].copy()
    session_df = session_df.loc[
        session_df["repeat_trial"].isin(ALLOWED_REPEAT_TRIALS)
        | session_df["repeat_trial"].isna()
    ].copy()

    if session_df.empty:
        raise RuntimeError(f"No rows remain for session type {session_type}.")
    if not session_df["training_level"].eq(TRAINING_LEVEL).all():
        raise RuntimeError(f"Session type {session_type} has another training level.")
    if not session_df["session_type"].eq(session_type).all():
        raise RuntimeError(f"Session type {session_type} filtering failed.")

    repeat_values = set(session_df["repeat_trial"].dropna().astype(int).unique())
    if not repeat_values.issubset(ALLOWED_REPEAT_TRIALS):
        raise RuntimeError(
            f"Session type {session_type} has repeat values {repeat_values}."
        )

    animals = sorted(session_df["animal"].dropna().astype(int).unique().tolist())
    if animals != expected["animals"]:
        raise RuntimeError(
            f"Session type {session_type} expected animals {expected['animals']}, "
            f"found {animals}."
        )

    condition_counts = {
        led_trial: int(session_df["LED_trial"].eq(led_trial).sum())
        for led_trial in [0, 1]
    }
    if condition_counts != expected["condition_trials"]:
        raise RuntimeError(
            f"Session type {session_type} expected condition counts "
            f"{expected['condition_trials']}, found {condition_counts}."
        )

    if session_df["LED_trial"].isna().any():
        raise RuntimeError(f"Session type {session_type} has missing LED_trial.")

    finite_rt_mask = np.isfinite(
        session_df[["timed_fix", "intended_fix"]]
    ).all(axis=1)
    negative_rt = (
        session_df.loc[finite_rt_mask, "timed_fix"]
        < session_df.loc[finite_rt_mask, "intended_fix"]
    )
    event_3 = session_df.loc[finite_rt_mask, "abort_event"].eq(3)
    if not np.array_equal(negative_rt.to_numpy(), event_3.to_numpy()):
        raise RuntimeError(
            f"Session type {session_type} has finite RT/event-3 mismatches."
        )

    led_on_df = session_df.loc[session_df["LED_trial"].eq(1)]
    derived_onset = led_on_df["intended_fix"] - led_on_df["LED_onset_time"]
    if not np.isfinite(derived_onset).all() or (derived_onset < 0).any():
        raise RuntimeError(
            f"Session type {session_type} has invalid derived LED-onset timing."
        )
    exact_zero_mask = derived_onset.eq(0)
    numerical_zero_mask = derived_onset.abs().le(1e-12)
    if not exact_zero_mask.equals(numerical_zero_mask):
        raise RuntimeError(
            f"Session type {session_type} has near-zero values distinct from zero."
        )
    if int(exact_zero_mask.sum()) != expected["zero_derived_onset_on_trials"]:
        raise RuntimeError(
            f"Session type {session_type} expected "
            f"{expected['zero_derived_onset_on_trials']:,} zero-onset ON trials, "
            f"found {exact_zero_mask.sum():,}."
        )

    session_frames[session_type] = session_df


# %%
############ Build all-onset and nonzero-onset row payloads ############
timing_definitions = [
    ("intended_fix", INTENDED_FIX_BINS),
    ("LED_onset_time", LED_ONSET_BINS),
    ("derived_LED_onset", DERIVED_ONSET_BINS),
]

row_payloads = []
timing_audit_rows = []
abort_audit_rows = []

for row_spec in ROW_SPECS:
    session_type = row_spec["session_type"]
    expected = SESSION_EXPECTATIONS[session_type]
    session_df = session_frames[session_type]

    off_df = session_df.loc[session_df["LED_trial"].eq(0)].copy()
    all_on_df = session_df.loc[session_df["LED_trial"].eq(1)].copy()
    all_on_derived_onset = (
        all_on_df["intended_fix"] - all_on_df["LED_onset_time"]
    )
    zero_onset_mask = all_on_derived_onset.eq(0)

    if row_spec["exclude_zero_onset_from_on"]:
        on_df = all_on_df.loc[~zero_onset_mask].copy()
        n_removed_on_trials = int(zero_onset_mask.sum())
    else:
        on_df = all_on_df.copy()
        n_removed_on_trials = 0

    expected_on_trials = expected["condition_trials"][1] - n_removed_on_trials
    if len(on_df) != expected_on_trials:
        raise RuntimeError(
            f"{row_spec['key']} expected {expected_on_trials:,} ON trials, "
            f"found {len(on_df):,}."
        )

    intended_fix = on_df["intended_fix"].to_numpy(dtype=float)
    led_onset_time = on_df["LED_onset_time"].to_numpy(dtype=float)
    derived_led_onset = intended_fix - led_onset_time
    if not np.isfinite(
        np.column_stack([intended_fix, led_onset_time, derived_led_onset])
    ).all():
        raise RuntimeError(f"{row_spec['key']} has non-finite timing values.")
    if (derived_led_onset < 0).any():
        raise RuntimeError(f"{row_spec['key']} has negative derived LED onsets.")
    if (
        row_spec["exclude_zero_onset_from_on"]
        and np.equal(derived_led_onset, 0).any()
    ):
        raise RuntimeError(f"{row_spec['key']} retained a zero-onset ON trial.")

    timing_values = {
        "intended_fix": intended_fix,
        "LED_onset_time": led_onset_time,
        "derived_LED_onset": derived_led_onset,
    }
    timing_payload = {}
    for timing_name, bin_edges in timing_definitions:
        values = timing_values[timing_name]
        counts, heights, area = scaled_histogram(
            values,
            bin_edges,
            normalization_count=len(values),
        )
        if not np.isclose(area, 1.0, atol=1e-14, rtol=0):
            raise RuntimeError(
                f"{row_spec['key']} {timing_name} area is {area}, expected 1."
            )
        timing_payload[timing_name] = {
            "counts": counts,
            "heights": heights,
            "bin_edges": bin_edges,
            "area": area,
        }
        timing_audit_rows.append(
            {
                "row": row_spec["key"],
                "distribution": timing_name,
                "n": len(values),
                "removed_zero_onset_ON_trials": n_removed_on_trials,
                "min_s": float(values.min()),
                "median_s": float(np.median(values)),
                "max_s": float(values.max()),
                "area": area,
            }
        )

    condition_frames = {0: off_df, 1: on_df}
    abort_payload = {}
    for led_trial, condition_df in condition_frames.items():
        coded_abort_df = condition_df.loc[
            condition_df["abort_event"].eq(3)
        ].copy()
        finite_abort_mask = np.isfinite(
            coded_abort_df[["timed_fix", "intended_fix", "LED_onset_time"]]
        ).all(axis=1)
        plotted_abort_df = coded_abort_df.loc[finite_abort_mask].copy()

        if led_trial == 0 or not row_spec["exclude_zero_onset_from_on"]:
            expected_coded = expected["coded_aborts"][led_trial]
            expected_finite = expected["finite_aborts"][led_trial]
        else:
            expected_coded = expected["nonzero_on_coded_aborts"]
            expected_finite = expected["nonzero_on_finite_aborts"]

        if len(coded_abort_df) != expected_coded:
            raise RuntimeError(
                f"{row_spec['key']} {LED_LABELS[led_trial]} expected "
                f"{expected_coded:,} coded aborts, found {len(coded_abort_df):,}."
            )
        if len(plotted_abort_df) != expected_finite:
            raise RuntimeError(
                f"{row_spec['key']} {LED_LABELS[led_trial]} expected "
                f"{expected_finite:,} finite aborts, "
                f"found {len(plotted_abort_df):,}."
            )
        if not (
            plotted_abort_df["timed_fix"] < plotted_abort_df["intended_fix"]
        ).all():
            raise RuntimeError(
                f"{row_spec['key']} {LED_LABELS[led_trial]} has a finite "
                "event-3 abort with timed_fix >= intended_fix."
            )

        aligned_to_raw_led_onset = (
            plotted_abort_df["timed_fix"]
            - plotted_abort_df["LED_onset_time"]
        ).to_numpy(dtype=float)
        aligned_to_derived_led_onset = (
            plotted_abort_df["timed_fix"]
            - (
                plotted_abort_df["intended_fix"]
                - plotted_abort_df["LED_onset_time"]
            )
        ).to_numpy(dtype=float)

        raw_counts, raw_heights, raw_area = scaled_histogram(
            aligned_to_raw_led_onset,
            ALIGNED_FULL_BINS,
            normalization_count=len(condition_df),
        )
        derived_counts, derived_heights, derived_area = scaled_histogram(
            aligned_to_derived_led_onset,
            ALIGNED_FULL_BINS,
            normalization_count=len(condition_df),
        )
        expected_area = len(plotted_abort_df) / len(condition_df)
        if not np.isclose(raw_area, expected_area, atol=1e-14, rtol=0):
            raise RuntimeError(f"{row_spec['key']} raw-onset area mismatch.")
        if not np.isclose(derived_area, expected_area, atol=1e-14, rtol=0):
            raise RuntimeError(f"{row_spec['key']} derived-onset area mismatch.")

        raw_outside = int(
            (
                (aligned_to_raw_led_onset < ALIGNED_DISPLAY_LIMITS_S[0])
                | (aligned_to_raw_led_onset > ALIGNED_DISPLAY_LIMITS_S[1])
            ).sum()
        )
        derived_outside = int(
            (
                (aligned_to_derived_led_onset < ALIGNED_DISPLAY_LIMITS_S[0])
                | (aligned_to_derived_led_onset > ALIGNED_DISPLAY_LIMITS_S[1])
            ).sum()
        )

        abort_payload[led_trial] = {
            "n_total_trials": len(condition_df),
            "n_coded_aborts": len(coded_abort_df),
            "n_finite_aborts": len(plotted_abort_df),
            "abort_fraction": expected_area,
            "raw_counts": raw_counts,
            "raw_heights": raw_heights,
            "derived_counts": derived_counts,
            "derived_heights": derived_heights,
        }
        abort_audit_rows.append(
            {
                "row": row_spec["key"],
                "condition": LED_LABELS[led_trial],
                "all_condition_trials": len(condition_df),
                "coded_event_3_aborts": len(coded_abort_df),
                "finite_plotted_aborts": len(plotted_abort_df),
                "missing_abort_timing": len(coded_abort_df)
                - len(plotted_abort_df),
                "full_abort_fraction": expected_area,
                "outside_raw_onset_display": raw_outside,
                "outside_derived_onset_display": derived_outside,
            }
        )

    row_payloads.append(
        {
            "spec": row_spec,
            "n_removed_on_trials": n_removed_on_trials,
            "timing": timing_payload,
            "aborts": abort_payload,
        }
    )


# %%
############ Confirm that the OFF baseline is unchanged ############
for all_row_index, nonzero_row_index in [(0, 2), (1, 3)]:
    all_off = row_payloads[all_row_index]["aborts"][0]
    nonzero_off = row_payloads[nonzero_row_index]["aborts"][0]
    for key in ["raw_counts", "raw_heights", "derived_counts", "derived_heights"]:
        if not np.array_equal(all_off[key], nonzero_off[key]):
            raise RuntimeError(
                f"OFF baseline changed between {ROW_SPECS[all_row_index]['key']} "
                f"and {ROW_SPECS[nonzero_row_index]['key']}."
            )
    if all_off["n_total_trials"] != nonzero_off["n_total_trials"]:
        raise RuntimeError("OFF denominator changed in the excluded-onset rows.")


# %%
############ Plot the 4 x 5 matched comparison ############
column_titles = [
    "intended_fix",
    "LED_onset_time",
    "intended_fix - LED_onset_time",
    "Abort time relative to\nLED_onset_time",
    "Abort time relative to\nintended_fix - LED_onset_time",
]
timing_keys = ["intended_fix", "LED_onset_time", "derived_LED_onset"]

fig, axes = plt.subplots(
    len(row_payloads),
    len(column_titles),
    figsize=(22, 13),
    sharex="col",
)

for column_index, title in enumerate(column_titles):
    axes[0, column_index].set_title(title, fontsize=12)

for row_index, payload in enumerate(row_payloads):
    for column_index, timing_key in enumerate(timing_keys):
        result = payload["timing"][timing_key]
        axes[row_index, column_index].stairs(
            result["heights"],
            result["bin_edges"],
            color=LED_COLORS[1],
            linewidth=1.5,
        )

    for led_trial in [0, 1]:
        abort_result = payload["aborts"][led_trial]
        axes[row_index, 3].stairs(
            abort_result["raw_heights"],
            ALIGNED_FULL_BINS,
            color=LED_COLORS[led_trial],
            linewidth=1.4,
        )
        axes[row_index, 4].stairs(
            abort_result["derived_heights"],
            ALIGNED_FULL_BINS,
            color=LED_COLORS[led_trial],
            linewidth=1.4,
        )

    n_on = int(payload["timing"]["intended_fix"]["counts"].sum())
    n_text = f"n = {n_on:,}"
    if payload["n_removed_on_trials"]:
        n_text += f"\nremoved = {payload['n_removed_on_trials']:,}"
    axes[row_index, 0].text(
        0.97,
        0.94,
        n_text,
        transform=axes[row_index, 0].transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )

    off_area = payload["aborts"][0]["abort_fraction"]
    on_area = payload["aborts"][1]["abort_fraction"]
    area_text = f"OFF area = {off_area:.3f}\nON area = {on_area:.3f}"
    for column_index in [3, 4]:
        axes[row_index, column_index].text(
            0.97,
            0.94,
            area_text,
            transform=axes[row_index, column_index].transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
        )
        axes[row_index, column_index].axvline(
            0,
            color="black",
            linestyle="--",
            linewidth=0.9,
            alpha=0.65,
        )
        axes[row_index, column_index].set_xlim(ALIGNED_DISPLAY_LIMITS_S)

    axes[row_index, 0].set_ylabel(
        f"{payload['spec']['row_label']}\nDensity (s$^{{-1}}$)",
        fontsize=9.5,
    )

for column_index, timing_key in enumerate(timing_keys):
    column_y_max = max(
        float(payload["timing"][timing_key]["heights"].max())
        for payload in row_payloads
    )
    for row_index in range(len(row_payloads)):
        axes[row_index, column_index].set_ylim(0, 1.10 * column_y_max)

aligned_y_max = max(
    float(payload["aborts"][led_trial][height_key].max())
    for payload in row_payloads
    for led_trial in [0, 1]
    for height_key in ["raw_heights", "derived_heights"]
)
for row_index in range(len(row_payloads)):
    for column_index in [3, 4]:
        axes[row_index, column_index].set_ylim(0, 1.15 * aligned_y_max)

for ax in axes.flat:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.16, linewidth=0.6)
    ax.tick_params(axis="both", labelsize=8)

for column_index in range(3):
    axes[-1, column_index].set_xlabel("Time (s)", fontsize=10)
for column_index in [3, 4]:
    axes[-1, column_index].set_xlabel("Aligned abort time (s)", fontsize=10)

fig.legend(
    handles=[
        Line2D([0], [0], color=LED_COLORS[0], lw=1.5, label=LED_LABELS[0]),
        Line2D([0], [0], color=LED_COLORS[1], lw=1.5, label=LED_LABELS[1]),
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, 0.965),
    ncol=2,
    frameon=False,
    fontsize=10,
)
fig.suptitle(
    "LED7 session types 7 and 8: all trials versus zero-onset ON removed\n"
    "training level 16; repeat_trial = 0, 2, or NaN; 20 ms bins",
    fontsize=15,
    y=0.995,
)
fig.text(
    0.5,
    0.012,
    "Bottom rows remove intended_fix - LED_onset_time = 0 only from LED ON; "
    "LED OFF is unchanged. Aligned panels display -1 to +1 s; full-area "
    "normalization uses -2 to +2 s.",
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=(0.015, 0.04, 1.0, 0.935), w_pad=1.4, h_pad=1.3)
fig.savefig(OUTPUT_FIGURE_PATH, dpi=OUTPUT_DPI, bbox_inches="tight")

if SHOW_PLOT:
    plt.show()
else:
    plt.close(fig)


# %%
############ Print the numerical audit ############
timing_audit_df = pd.DataFrame(timing_audit_rows)
abort_audit_df = pd.DataFrame(abort_audit_rows)

print(f"Source: {INPUT_MAT_PATH} :: {TABLE_NAME}")
print(
    "Base filters in order: training_level == 16; session_type 7 or 8; "
    "repeat_trial in {0, 2, NaN}."
)
print(
    "Top rows retain all LED-ON timing values. Bottom rows remove exact "
    "intended_fix - LED_onset_time == 0 rows from LED ON only; LED OFF is "
    "unchanged. Abort panels use abort_event == 3."
)
print("\nTiming-distribution audit")
print(
    timing_audit_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)
print("\nAbort-distribution audit")
print(
    abort_audit_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)
print(f"\nSaved figure: {OUTPUT_FIGURE_PATH.resolve()}")
