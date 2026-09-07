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
        "fitting_aborts/plot_led7_s7_s8_all_vs_s9_onset_floor_4x5.py"
    ) from exc


# %%
############ Parameters (edit here) ############
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

INPUT_MAT_PATH = REPO_ROOT / "raw_data" / "outMatrix_LED7_latest.mat"
TABLE_NAME = "totalout_stGtACRII"
OUTPUT_FIGURE_PATH = (
    SCRIPT_DIR / "led7_s7_s8_all_vs_s9_onset_floor_4x5.png"
)

TRAINING_LEVEL = 16
ALLOWED_REPEAT_TRIALS = {0, 2}
REFERENCE_SESSION_TYPE = 9
COMPARISON_SESSION_TYPES = [7, 8]
EXPECTED_REFERENCE_ONSET_FLOOR_S = 0.100

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

EXPECTED = {
    7: {
        "off_trials": 92_057,
        "all_on_trials": 46_383,
        "early_off_trials": 18_457,
        "early_on_trials": 9_256,
        "kept_off_trials": 73_600,
        "kept_on_trials": 37_127,
        "off_coded_aborts": 14_218,
        "off_finite_aborts": 14_215,
        "all_on_coded_aborts": 8_276,
        "all_on_finite_aborts": 8_276,
        "kept_off_coded_aborts": 13_632,
        "kept_off_finite_aborts": 13_629,
        "kept_on_coded_aborts": 7_803,
        "kept_on_finite_aborts": 7_803,
    },
    8: {
        "off_trials": 92_784,
        "all_on_trials": 55_644,
        "early_off_trials": 18_801,
        "early_on_trials": 11_151,
        "kept_off_trials": 73_983,
        "kept_on_trials": 44_493,
        "off_coded_aborts": 12_570,
        "off_finite_aborts": 12_570,
        "all_on_coded_aborts": 10_356,
        "all_on_finite_aborts": 10_354,
        "kept_off_coded_aborts": 11_971,
        "kept_off_finite_aborts": 11_971,
        "kept_on_coded_aborts": 9_344,
        "kept_on_finite_aborts": 9_342,
    },
}

ROW_SPECS = [
    {
        "key": "s7_all",
        "session_type": 7,
        "apply_onset_floor": False,
        "row_label": "LED7 session type 7\nall eligible trials",
    },
    {
        "key": "s8_all",
        "session_type": 8,
        "apply_onset_floor": False,
        "row_label": "LED7 session type 8\nall eligible trials",
    },
    {
        "key": "s7_floor",
        "session_type": 7,
        "apply_onset_floor": True,
        "row_label": "LED7 session type 7\nOFF/ON onset >= 100 ms",
    },
    {
        "key": "s8_floor",
        "session_type": 8,
        "apply_onset_floor": True,
        "row_label": "LED7 session type 8\nOFF/ON onset >= 100 ms",
    },
]


# %%
############ Histogram helper ############
def scaled_histogram(values, bin_edges, normalization_count):
    counts, _ = np.histogram(values, bins=bin_edges)
    if int(counts.sum()) != len(values):
        raise RuntimeError(
            f"Histogram contains {counts.sum():,} of {len(values):,} values."
        )
    heights = counts.astype(float) / (normalization_count * BIN_WIDTH_S)
    area = float(np.sum(heights * np.diff(bin_edges)))
    return counts, heights, area


# %%
############ Load the LED7 MATLAB table ############
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
repeat_df = training_df.loc[
    training_df["repeat_trial"].isin(ALLOWED_REPEAT_TRIALS)
    | training_df["repeat_trial"].isna()
].copy()


# %%
############ Derive the onset floor from session type 9 ############
reference_on_df = repeat_df.loc[
    repeat_df["session_type"].eq(REFERENCE_SESSION_TYPE)
    & repeat_df["LED_trial"].eq(1)
].copy()
if len(reference_on_df) != 39_380:
    raise RuntimeError(
        f"Expected 39,380 session-type-9 LED-ON rows, found {len(reference_on_df):,}."
    )
if not np.isfinite(reference_on_df["LED_onset_time"]).all():
    raise RuntimeError("Session-type-9 LED_onset_time contains non-finite values.")

reference_onsets = reference_on_df["LED_onset_time"].to_numpy(dtype=float)
reference_min_onset_s = float(reference_onsets.min())
reference_onset_floor_s = (
    np.floor(reference_min_onset_s / BIN_WIDTH_S) * BIN_WIDTH_S
)
if not np.isclose(
    reference_onset_floor_s,
    EXPECTED_REFERENCE_ONSET_FLOOR_S,
    atol=1e-12,
    rtol=0,
):
    raise RuntimeError(
        "Expected the session-type-9 onset floor to be 100 ms; found "
        f"{1000 * reference_onset_floor_s:.6f} ms."
    )
if (reference_onsets < reference_onset_floor_s).any():
    raise RuntimeError("Session type 9 has LED onsets below its inferred floor.")

reference_counts, _ = np.histogram(reference_onsets, bins=LED_ONSET_BINS)
first_reference_bin = int(np.flatnonzero(reference_counts)[0])
if not np.isclose(
    LED_ONSET_BINS[first_reference_bin],
    reference_onset_floor_s,
    atol=1e-12,
    rtol=0,
):
    raise RuntimeError("The first occupied session-type-9 onset bin is unexpected.")


# %%
############ Prepare session types 7 and 8 ############
session_frames = {}
expected_animals = [90, 92, 93, 98, 99, 100, 102, 103]

for session_type in COMPARISON_SESSION_TYPES:
    session_df = repeat_df.loc[
        repeat_df["session_type"].eq(session_type)
    ].copy()
    if session_df.empty:
        raise RuntimeError(f"No rows remain for session type {session_type}.")

    animals = sorted(session_df["animal"].dropna().astype(int).unique().tolist())
    if animals != expected_animals:
        raise RuntimeError(
            f"Session type {session_type} expected animals {expected_animals}, "
            f"found {animals}."
        )
    if session_df["LED_trial"].isna().any():
        raise RuntimeError(f"Session type {session_type} has missing LED_trial.")

    observed_condition_counts = {
        0: int(session_df["LED_trial"].eq(0).sum()),
        1: int(session_df["LED_trial"].eq(1).sum()),
    }
    expected_condition_counts = {
        0: EXPECTED[session_type]["off_trials"],
        1: EXPECTED[session_type]["all_on_trials"],
    }
    if observed_condition_counts != expected_condition_counts:
        raise RuntimeError(
            f"Session type {session_type} expected {expected_condition_counts}, "
            f"found {observed_condition_counts}."
        )

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

    for led_trial, condition_name in [(0, "OFF"), (1, "ON")]:
        condition_df = session_df.loc[session_df["LED_trial"].eq(led_trial)]
        derived_onsets = (
            condition_df["intended_fix"] - condition_df["LED_onset_time"]
        )
        if not np.isfinite(derived_onsets).all() or (derived_onsets < 0).any():
            raise RuntimeError(
                f"Session type {session_type} has invalid derived "
                f"{condition_name} timing."
            )
        n_early_trials = int(
            (derived_onsets < reference_onset_floor_s).sum()
        )
        expected_early_trials = EXPECTED[session_type][
            f"early_{condition_name.lower()}_trials"
        ]
        if n_early_trials != expected_early_trials:
            raise RuntimeError(
                f"Session type {session_type} expected "
                f"{expected_early_trials:,} {condition_name} rows below 100 ms, "
                f"found {n_early_trials:,}."
            )

    session_frames[session_type] = session_df


# %%
############ Build all-onset and >=100-ms-onset row payloads ############
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
    expected = EXPECTED[session_type]
    session_df = session_frames[session_type]

    all_off_df = session_df.loc[session_df["LED_trial"].eq(0)].copy()
    all_on_df = session_df.loc[session_df["LED_trial"].eq(1)].copy()
    all_off_derived_onset = (
        all_off_df["intended_fix"] - all_off_df["LED_onset_time"]
    )
    all_on_derived_onset = (
        all_on_df["intended_fix"] - all_on_df["LED_onset_time"]
    )
    early_off_mask = all_off_derived_onset < reference_onset_floor_s
    early_on_mask = all_on_derived_onset < reference_onset_floor_s

    if row_spec["apply_onset_floor"]:
        off_df = all_off_df.loc[~early_off_mask].copy()
        on_df = all_on_df.loc[~early_on_mask].copy()
        n_removed_off_trials = int(early_off_mask.sum())
        n_removed_on_trials = int(early_on_mask.sum())
        expected_off_trials = expected["kept_off_trials"]
        expected_on_trials = expected["kept_on_trials"]
    else:
        off_df = all_off_df.copy()
        on_df = all_on_df.copy()
        n_removed_off_trials = 0
        n_removed_on_trials = 0
        expected_off_trials = expected["off_trials"]
        expected_on_trials = expected["all_on_trials"]

    if len(off_df) != expected_off_trials:
        raise RuntimeError(
            f"{row_spec['key']} expected {expected_off_trials:,} OFF trials, "
            f"found {len(off_df):,}."
        )
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
    if (
        row_spec["apply_onset_floor"]
        and (derived_led_onset < reference_onset_floor_s).any()
    ):
        raise RuntimeError(f"{row_spec['key']} retained an ON onset below 100 ms.")
    if row_spec["apply_onset_floor"]:
        off_derived_onset = (
            off_df["intended_fix"] - off_df["LED_onset_time"]
        )
        if (off_derived_onset < reference_onset_floor_s).any():
            raise RuntimeError(
                f"{row_spec['key']} retained an OFF onset below 100 ms."
            )

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
        }
        timing_audit_rows.append(
            {
                "row": row_spec["key"],
                "distribution": timing_name,
                "n": len(values),
                "removed_ON_trials": n_removed_on_trials,
                "min_s": float(values.min()),
                "median_s": float(np.median(values)),
                "max_s": float(values.max()),
                "area": area,
            }
        )

    abort_payload = {}
    for led_trial, condition_df in {0: off_df, 1: on_df}.items():
        coded_abort_df = condition_df.loc[
            condition_df["abort_event"].eq(3)
        ].copy()
        finite_abort_mask = np.isfinite(
            coded_abort_df[["timed_fix", "intended_fix", "LED_onset_time"]]
        ).all(axis=1)
        plotted_abort_df = coded_abort_df.loc[finite_abort_mask].copy()

        if led_trial == 0:
            if row_spec["apply_onset_floor"]:
                expected_coded = expected["kept_off_coded_aborts"]
                expected_finite = expected["kept_off_finite_aborts"]
            else:
                expected_coded = expected["off_coded_aborts"]
                expected_finite = expected["off_finite_aborts"]
        elif row_spec["apply_onset_floor"]:
            expected_coded = expected["kept_on_coded_aborts"]
            expected_finite = expected["kept_on_finite_aborts"]
        else:
            expected_coded = expected["all_on_coded_aborts"]
            expected_finite = expected["all_on_finite_aborts"]

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
                f"{row_spec['key']} contains a finite non-abort event-3 row."
            )

        aligned_to_raw_onset = (
            plotted_abort_df["timed_fix"]
            - plotted_abort_df["LED_onset_time"]
        ).to_numpy(dtype=float)
        aligned_to_derived_onset = (
            plotted_abort_df["timed_fix"]
            - (
                plotted_abort_df["intended_fix"]
                - plotted_abort_df["LED_onset_time"]
            )
        ).to_numpy(dtype=float)

        raw_counts, raw_heights, raw_area = scaled_histogram(
            aligned_to_raw_onset,
            ALIGNED_FULL_BINS,
            normalization_count=len(condition_df),
        )
        derived_counts, derived_heights, derived_area = scaled_histogram(
            aligned_to_derived_onset,
            ALIGNED_FULL_BINS,
            normalization_count=len(condition_df),
        )
        expected_area = len(plotted_abort_df) / len(condition_df)
        if not np.isclose(raw_area, expected_area, atol=1e-14, rtol=0):
            raise RuntimeError(f"{row_spec['key']} raw-onset area mismatch.")
        if not np.isclose(derived_area, expected_area, atol=1e-14, rtol=0):
            raise RuntimeError(f"{row_spec['key']} derived-onset area mismatch.")

        pre_onset_bin_mask = ALIGNED_FULL_BINS[1:] <= 0
        derived_pre_onset_fraction = float(
            derived_counts[pre_onset_bin_mask].sum() / len(condition_df)
        )

        raw_outside = int(
            (
                (aligned_to_raw_onset < ALIGNED_DISPLAY_LIMITS_S[0])
                | (aligned_to_raw_onset > ALIGNED_DISPLAY_LIMITS_S[1])
            ).sum()
        )
        derived_outside = int(
            (
                (aligned_to_derived_onset < ALIGNED_DISPLAY_LIMITS_S[0])
                | (aligned_to_derived_onset > ALIGNED_DISPLAY_LIMITS_S[1])
            ).sum()
        )

        abort_payload[led_trial] = {
            "n_total_trials": len(condition_df),
            "abort_fraction": expected_area,
            "derived_pre_onset_fraction": derived_pre_onset_fraction,
            "raw_counts": raw_counts,
            "raw_heights": raw_heights,
            "derived_counts": derived_counts,
            "derived_heights": derived_heights,
        }
        abort_audit_rows.append(
            {
                "row": row_spec["key"],
                "condition": LED_LABELS[led_trial],
                "removed_condition_trials": (
                    n_removed_off_trials if led_trial == 0 else n_removed_on_trials
                ),
                "all_condition_trials": len(condition_df),
                "coded_event_3_aborts": len(coded_abort_df),
                "finite_plotted_aborts": len(plotted_abort_df),
                "missing_abort_timing": len(coded_abort_df)
                - len(plotted_abort_df),
                "full_abort_fraction": expected_area,
                "derived_pre_onset_fraction": derived_pre_onset_fraction,
                "outside_raw_onset_display": raw_outside,
                "outside_derived_onset_display": derived_outside,
            }
        )

    row_payloads.append(
        {
            "spec": row_spec,
            "n_removed_off_trials": n_removed_off_trials,
            "n_removed_on_trials": n_removed_on_trials,
            "timing": timing_payload,
            "aborts": abort_payload,
        }
    )


# %%
############ Confirm pre-onset agreement after symmetric selection ############
pre_onset_audit_rows = []
for floor_row_index in [2, 3]:
    payload = row_payloads[floor_row_index]
    off_fraction = payload["aborts"][0]["derived_pre_onset_fraction"]
    on_fraction = payload["aborts"][1]["derived_pre_onset_fraction"]
    absolute_difference = abs(on_fraction - off_fraction)
    if absolute_difference > 0.005:
        raise RuntimeError(
            f"{payload['spec']['key']} pre-onset OFF/ON fractions differ by "
            f"{absolute_difference:.6f}."
        )
    pre_onset_audit_rows.append(
        {
            "row": payload["spec"]["key"],
            "OFF_pre_onset_fraction": off_fraction,
            "ON_pre_onset_fraction": on_fraction,
            "ON_minus_OFF": on_fraction - off_fraction,
        }
    )


# %%
############ Plot the 4 x 5 comparison ############
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
        n_text += (
            f"\nremoved OFF/ON = "
            f"{payload['n_removed_off_trials']:,}/"
            f"{payload['n_removed_on_trials']:,}"
        )
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
    "LED7 session types 7 and 8: symmetric session-9 onset floor\n"
    "training level 16; repeat_trial = 0, 2, or NaN; 20 ms bins",
    fontsize=15,
    y=0.995,
)
fig.text(
    0.5,
    0.012,
    "Session type 9 begins at 100 ms. Bottom rows remove derived LED onset "
    "< 100 ms from both OFF and ON, then renormalize within each retained "
    "condition. Aligned panels display -1 to +1 s; full-area normalization "
    "uses -2 to +2 s.",
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
    "Reference: session type 9, training level 16, repeat_trial in "
    "{0, 2, NaN}, LED_trial == 1."
)
print(
    f"Reference minimum raw LED onset: {1000 * reference_min_onset_s:.6f} ms; "
    f"first occupied 20 ms bin: {1000 * reference_onset_floor_s:.0f}-"
    f"{1000 * (reference_onset_floor_s + BIN_WIDTH_S):.0f} ms."
)
print(
    "Bottom rows remove derived onset < 100 ms from both OFF and ON, then "
    "renormalize each retained condition. Abort panels use abort_event == 3."
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
print("\nBottom-row pre-onset agreement")
print(
    pd.DataFrame(pre_onset_audit_rows).to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)
print(f"\nSaved figure: {OUTPUT_FIGURE_PATH.resolve()}")
