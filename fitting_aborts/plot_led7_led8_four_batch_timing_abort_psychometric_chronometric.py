# %%
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator, ScalarFormatter
import numpy as np
import pandas as pd

try:
    from matio import load_from_mat
except ImportError as exc:
    raise ImportError(
        "This script needs the isolated mat-io decoder. Run it as:\n"
        "env PYTHONPATH=/tmp/codex_mat_io .venv/bin/python "
        "fitting_aborts/"
        "plot_led7_led8_four_batch_timing_abort_psychometric_chronometric.py"
    ) from exc


# %%
############ Parameters (edit here) ############
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

LED7_MAT_PATH = REPO_ROOT / "raw_data" / "outMatrix_LED7_latest.mat"
LED8_MAT_PATH = REPO_ROOT / "raw_data" / "outMatrix_LED8.mat"
OUTPUT_FIGURE_PATH = (
    SCRIPT_DIR
    / "led7_led8_four_batch_timing_abort_psychometric_chronometric_4x8.png"
)

TRAINING_LEVEL = 16
ALLOWED_REPEAT_TRIALS = {0, 2}
LED_COLORS = {0: "tab:blue", 1: "tab:red"}
LED_LABELS = {0: "LED OFF", 1: "LED ON"}

BIN_WIDTH_S = 0.020
INTENDED_FIX_BINS = np.arange(0, 2200 + 20, 20, dtype=float) / 1000
LED_ONSET_BINS = np.arange(0, 1200 + 20, 20, dtype=float) / 1000
INTENDED_MINUS_ONSET_BINS = np.arange(0, 2200 + 20, 20, dtype=float) / 1000
ALIGNED_FULL_BINS = np.arange(-2000, 2000 + 20, 20, dtype=float) / 1000
ALIGNED_DISPLAY_LIMITS_S = (-1.0, 1.0)

RT_WINDOW_S = (0.0, 1.0)
SIGNED_ILDS = [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]
ABS_ILDS = [1, 2, 4, 8, 16]
EXPECTED_ABLS = {20, 40, 60}
ACQUISITION_TICK_S = 1.0 / 1024.0
CHRONOMETRIC_Y_LIMITS_S = (0.10, 0.35)

OUTPUT_DPI = 250
SHOW_PLOT = False

BATCH_SPECS = [
    {
        "key": "LED7_s7",
        "row_label": "LED7\nsession type 7",
        "mat_path": LED7_MAT_PATH,
        "table_name": "totalout_stGtACRII",
        "session_type": 7,
        "expected_animals": [90, 92, 93, 98, 99, 100, 102, 103],
        "expected_sessions": 199,
        "expected_condition_trials": {0: 92_057, 1: 46_383},
        "expected_coded_aborts": {0: 14_218, 1: 8_276},
        "expected_finite_aborts": {0: 14_215, 1: 8_276},
        "expected_completed_choices": {0: 73_248, 1: 33_834},
        "expected_correct_choices": {0: 61_089, 1: 26_417},
    },
    {
        "key": "LED7_s8",
        "row_label": "LED7\nsession type 8",
        "mat_path": LED7_MAT_PATH,
        "table_name": "totalout_stGtACRII",
        "session_type": 8,
        "expected_animals": [90, 92, 93, 98, 99, 100, 102, 103],
        "expected_sessions": 215,
        "expected_condition_trials": {0: 92_784, 1: 55_644},
        "expected_coded_aborts": {0: 12_570, 1: 10_356},
        "expected_finite_aborts": {0: 12_570, 1: 10_354},
        "expected_completed_choices": {0: 75_431, 1: 38_807},
        "expected_correct_choices": {0: 63_574, 1: 29_762},
    },
    {
        "key": "LED7_s9",
        "row_label": "LED7\nsession type 9",
        "mat_path": LED7_MAT_PATH,
        "table_name": "totalout_stGtACRII",
        "session_type": 9,
        "expected_animals": [90, 92, 93, 98, 99, 100, 102, 103],
        "expected_sessions": 191,
        "expected_condition_trials": {0: 61_408, 1: 39_380},
        "expected_coded_aborts": {0: 10_108, 1: 8_752},
        "expected_finite_aborts": {0: 10_107, 1: 8_747},
        "expected_completed_choices": {0: 46_722, 1: 25_265},
        "expected_correct_choices": {0: 37_522, 1: 19_234},
    },
    {
        "key": "LED8_s8",
        "row_label": "LED8\nsession type 8",
        "mat_path": LED8_MAT_PATH,
        "table_name": "totalout",
        "session_type": 8,
        "expected_animals": [105, 107, 108, 109, 112],
        "expected_sessions": 120,
        "expected_condition_trials": {0: 36_372, 1: 23_836},
        "expected_coded_aborts": {0: 6_763, 1: 4_057},
        "expected_finite_aborts": {0: 6_761, 1: 4_055},
        "expected_completed_choices": {0: 27_679, 1: 15_336},
        "expected_correct_choices": {0: 22_530, 1: 12_342},
    },
]


# %%
############ Reused histogram calculation ############
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
############ Load each MAT table once ############
required_columns = [
    "animal",
    "session",
    "training_level",
    "session_type",
    "repeat_trial",
    "LED_trial",
    "abort_event",
    "timed_fix",
    "intended_fix",
    "LED_onset_time",
    "timed_RT",
    "ABL",
    "ILD",
    "success",
    "response_poke",
]

loaded_tables = {}
for spec in BATCH_SPECS:
    mat_path = spec["mat_path"]
    table_name = spec["table_name"]
    source_key = (mat_path, table_name)

    if not mat_path.exists():
        raise FileNotFoundError(f"Could not find input MAT file: {mat_path}")

    if source_key not in loaded_tables:
        mat_contents = load_from_mat(mat_path, variable_names=[table_name])
        if table_name not in mat_contents:
            raise KeyError(f"{table_name!r} is absent from {mat_path}.")

        table_df = mat_contents[table_name]
        if not isinstance(table_df, pd.DataFrame):
            raise TypeError(
                f"Expected {table_name!r} in {mat_path} to decode as a DataFrame; "
                f"found {type(table_df).__name__}."
            )

        missing_columns = [
            column for column in required_columns if column not in table_df.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Missing columns in {mat_path}::{table_name}: {missing_columns}"
            )
        loaded_tables[source_key] = table_df


# %%
############ Apply filters and build all eight distributions ############
batch_payloads = []
timing_audit_rows = []
abort_audit_rows = []
valid_count_audit_rows = []
rt_consistency_audit_rows = []
psychometric_audit_rows = []
chronometric_audit_rows = []

timing_definitions = [
    ("intended_fix", INTENDED_FIX_BINS),
    ("LED_onset_time", LED_ONSET_BINS),
    ("intended_fix_minus_LED_onset_time", INTENDED_MINUS_ONSET_BINS),
]

for spec in BATCH_SPECS:
    source_df = loaded_tables[(spec["mat_path"], spec["table_name"])]

    # Keep the requested order explicit: training level, session type, repeat trial.
    training_df = source_df.loc[
        source_df["training_level"].eq(TRAINING_LEVEL)
    ].copy()
    session_df = training_df.loc[
        training_df["session_type"].eq(spec["session_type"])
    ].copy()
    filtered_df = session_df.loc[
        session_df["repeat_trial"].isin(ALLOWED_REPEAT_TRIALS)
        | session_df["repeat_trial"].isna()
    ].copy()

    if filtered_df.empty:
        raise RuntimeError(f"No rows remain for {spec['key']} after filtering.")
    if not filtered_df["training_level"].eq(TRAINING_LEVEL).all():
        raise RuntimeError(f"{spec['key']} contains another training level.")
    if not filtered_df["session_type"].eq(spec["session_type"]).all():
        raise RuntimeError(f"{spec['key']} contains another session type.")

    repeat_values = set(filtered_df["repeat_trial"].dropna().astype(int).unique())
    if not repeat_values.issubset(ALLOWED_REPEAT_TRIALS):
        raise RuntimeError(
            f"{spec['key']} has unexpected repeat_trial values: {repeat_values}"
        )

    led_values = set(filtered_df["LED_trial"].dropna().astype(int).unique())
    if led_values != {0, 1} or filtered_df["LED_trial"].isna().any():
        raise RuntimeError(
            f"{spec['key']} expected LED_trial values {{0, 1}}, found "
            f"{sorted(led_values)} with "
            f"{filtered_df['LED_trial'].isna().sum():,} missing rows."
        )

    animals = sorted(filtered_df["animal"].dropna().astype(int).unique().tolist())
    if animals != spec["expected_animals"]:
        raise RuntimeError(
            f"{spec['key']} expected animals {spec['expected_animals']}, "
            f"found {animals}."
        )

    n_sessions = filtered_df[["animal", "session"]].drop_duplicates().shape[0]
    if n_sessions != spec["expected_sessions"]:
        raise RuntimeError(
            f"{spec['key']} expected {spec['expected_sessions']} animal-session "
            f"pairs, found {n_sessions}."
        )

    condition_counts = {
        led_trial: int(filtered_df["LED_trial"].eq(led_trial).sum())
        for led_trial in [0, 1]
    }
    if condition_counts != spec["expected_condition_trials"]:
        raise RuntimeError(
            f"{spec['key']} expected condition counts "
            f"{spec['expected_condition_trials']}, found {condition_counts}."
        )

    finite_rt_mask = np.isfinite(
        filtered_df[["timed_fix", "intended_fix"]]
    ).all(axis=1)
    negative_rt_mask = (
        filtered_df.loc[finite_rt_mask, "timed_fix"]
        < filtered_df.loc[finite_rt_mask, "intended_fix"]
    )
    coded_abort_mask = filtered_df.loc[finite_rt_mask, "abort_event"].eq(3)
    if not np.array_equal(
        negative_rt_mask.to_numpy(), coded_abort_mask.to_numpy()
    ):
        n_mismatch = int((negative_rt_mask != coded_abort_mask).sum())
        raise RuntimeError(
            f"{spec['key']} has {n_mismatch} finite rows where "
            "timed_fix < intended_fix and abort_event == 3 disagree."
        )

    # Columns 1-3 remain the original all-LED-ON timing distributions.
    led_on_df = filtered_df.loc[filtered_df["LED_trial"].eq(1)].copy()
    if len(led_on_df) != spec["expected_condition_trials"][1]:
        raise RuntimeError(f"Unexpected LED-ON timing count for {spec['key']}.")
    if not np.isfinite(
        led_on_df[["intended_fix", "LED_onset_time"]]
    ).all(axis=None):
        raise RuntimeError(
            f"{spec['key']} has non-finite intended-fix or LED-onset timing."
        )

    intended_fix = led_on_df["intended_fix"].to_numpy(dtype=float)
    led_onset_time = led_on_df["LED_onset_time"].to_numpy(dtype=float)
    intended_minus_onset = intended_fix - led_onset_time
    if (intended_minus_onset < 0).any():
        raise RuntimeError(
            f"{spec['key']} contains LED_onset_time later than intended_fix."
        )

    timing_values = {
        "intended_fix": intended_fix,
        "LED_onset_time": led_onset_time,
        "intended_fix_minus_LED_onset_time": intended_minus_onset,
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
                f"{spec['key']} {timing_name} area is {area}, expected 1."
            )
        timing_payload[timing_name] = {
            "counts": counts,
            "heights": heights,
            "bin_edges": bin_edges,
        }
        timing_audit_rows.append(
            {
                "batch": spec["key"],
                "distribution": timing_name,
                "n": len(values),
                "min_s": float(values.min()),
                "median_s": float(np.median(values)),
                "max_s": float(values.max()),
                "histogram_area": area,
            }
        )

    # Columns 4-5 remain the original event-3 fixation-abort comparisons.
    abort_payload = {}
    for led_trial in [0, 1]:
        condition_df = filtered_df.loc[
            filtered_df["LED_trial"].eq(led_trial)
        ].copy()
        coded_abort_df = condition_df.loc[
            condition_df["abort_event"].eq(3)
        ].copy()
        finite_abort_mask = np.isfinite(
            coded_abort_df[["timed_fix", "intended_fix", "LED_onset_time"]]
        ).all(axis=1)
        plotted_abort_df = coded_abort_df.loc[finite_abort_mask].copy()

        n_total_trials = len(condition_df)
        n_coded_aborts = len(coded_abort_df)
        n_plotted_aborts = len(plotted_abort_df)
        if n_coded_aborts != spec["expected_coded_aborts"][led_trial]:
            raise RuntimeError(
                f"{spec['key']} {LED_LABELS[led_trial]} expected "
                f"{spec['expected_coded_aborts'][led_trial]:,} coded aborts, "
                f"found {n_coded_aborts:,}."
            )
        if n_plotted_aborts != spec["expected_finite_aborts"][led_trial]:
            raise RuntimeError(
                f"{spec['key']} {LED_LABELS[led_trial]} expected "
                f"{spec['expected_finite_aborts'][led_trial]:,} finite aborts, "
                f"found {n_plotted_aborts:,}."
            )
        if not (
            plotted_abort_df["timed_fix"] < plotted_abort_df["intended_fix"]
        ).all():
            raise RuntimeError(
                f"{spec['key']} {LED_LABELS[led_trial]} has a finite event-3 "
                "abort with timed_fix >= intended_fix."
            )

        aligned_to_led_onset = (
            plotted_abort_df["timed_fix"]
            - plotted_abort_df["LED_onset_time"]
        ).to_numpy(dtype=float)
        aligned_to_intended_minus_onset = (
            plotted_abort_df["timed_fix"]
            - (
                plotted_abort_df["intended_fix"]
                - plotted_abort_df["LED_onset_time"]
            )
        ).to_numpy(dtype=float)

        led_onset_counts, led_onset_heights, led_onset_area = scaled_histogram(
            aligned_to_led_onset,
            ALIGNED_FULL_BINS,
            normalization_count=n_total_trials,
        )
        gap_counts, gap_heights, gap_area = scaled_histogram(
            aligned_to_intended_minus_onset,
            ALIGNED_FULL_BINS,
            normalization_count=n_total_trials,
        )

        expected_abort_area = n_plotted_aborts / n_total_trials
        for alignment_name, area in [
            ("LED onset", led_onset_area),
            ("intended_fix - LED_onset_time", gap_area),
        ]:
            if not np.isclose(area, expected_abort_area, atol=1e-14, rtol=0):
                raise RuntimeError(
                    f"{spec['key']} {LED_LABELS[led_trial]} {alignment_name} "
                    f"area is {area}, expected {expected_abort_area}."
                )

        outside_led_onset_display = int(
            (
                (aligned_to_led_onset < ALIGNED_DISPLAY_LIMITS_S[0])
                | (aligned_to_led_onset > ALIGNED_DISPLAY_LIMITS_S[1])
            ).sum()
        )
        outside_gap_display = int(
            (
                (aligned_to_intended_minus_onset < ALIGNED_DISPLAY_LIMITS_S[0])
                | (aligned_to_intended_minus_onset > ALIGNED_DISPLAY_LIMITS_S[1])
            ).sum()
        )

        abort_payload[led_trial] = {
            "abort_fraction": expected_abort_area,
            "led_onset_heights": led_onset_heights,
            "gap_heights": gap_heights,
        }
        abort_audit_rows.append(
            {
                "batch": spec["key"],
                "condition": LED_LABELS[led_trial],
                "all_filtered_trials": n_total_trials,
                "coded_event_3_aborts": n_coded_aborts,
                "finite_plotted_aborts": n_plotted_aborts,
                "missing_abort_timing": n_coded_aborts - n_plotted_aborts,
                "full_abort_fraction": expected_abort_area,
                "outside_led_onset_display": outside_led_onset_display,
                "outside_gap_display": outside_gap_display,
            }
        )

    # Columns 6-8 use the requested derived RT window and choice filters.
    filtered_df["RTwrtStim"] = (
        filtered_df["timed_fix"] - filtered_df["intended_fix"]
    )
    finite_rt_comparison = np.isfinite(
        filtered_df[["RTwrtStim", "timed_RT"]]
    ).all(axis=1)
    if not finite_rt_comparison.any():
        raise RuntimeError(f"{spec['key']} has no finite timed_RT comparison rows.")
    rt_delta = (
        filtered_df.loc[finite_rt_comparison, "RTwrtStim"]
        - filtered_df.loc[finite_rt_comparison, "timed_RT"]
    ).abs()
    maximum_rt_delta = float(rt_delta.max())
    if maximum_rt_delta > ACQUISITION_TICK_S + 1e-9:
        raise RuntimeError(
            f"{spec['key']} derived RT and timed_RT differ by "
            f"{maximum_rt_delta:.9f} s, more than one acquisition tick."
        )
    rt_consistency_audit_rows.append(
        {
            "batch": spec["key"],
            "finite_comparison_rows": int(finite_rt_comparison.sum()),
            "maximum_abs_delta_s": maximum_rt_delta,
            "median_abs_delta_s": float(rt_delta.median()),
        }
    )

    valid_rt_df = filtered_df.loc[
        np.isfinite(filtered_df["RTwrtStim"])
        & filtered_df["RTwrtStim"].between(
            RT_WINDOW_S[0], RT_WINDOW_S[1], inclusive="both"
        )
    ].copy()
    completed_choice_df = valid_rt_df.loc[
        valid_rt_df["success"].isin([-1, 1])
    ].copy()
    if not completed_choice_df["response_poke"].isin([2, 3]).all():
        raise RuntimeError(
            f"{spec['key']} has completed choices without response_poke 2 or 3."
        )
    correct_choice_df = completed_choice_df.loc[
        completed_choice_df["success"].eq(1)
    ].copy()

    completed_counts = {
        led_trial: int(completed_choice_df["LED_trial"].eq(led_trial).sum())
        for led_trial in [0, 1]
    }
    correct_counts = {
        led_trial: int(correct_choice_df["LED_trial"].eq(led_trial).sum())
        for led_trial in [0, 1]
    }
    if completed_counts != spec["expected_completed_choices"]:
        raise RuntimeError(
            f"{spec['key']} expected completed-choice counts "
            f"{spec['expected_completed_choices']}, found {completed_counts}."
        )
    if correct_counts != spec["expected_correct_choices"]:
        raise RuntimeError(
            f"{spec['key']} expected correct-choice counts "
            f"{spec['expected_correct_choices']}, found {correct_counts}."
        )

    completed_choice_df["right_choice"] = (
        completed_choice_df["response_poke"].eq(3)
    )
    completed_choice_df["abs_ILD"] = completed_choice_df["ILD"].abs()
    correct_choice_df["abs_ILD"] = correct_choice_df["ILD"].abs()

    observed_abls = set(completed_choice_df["ABL"].dropna().astype(int).unique())
    if observed_abls != EXPECTED_ABLS:
        raise RuntimeError(
            f"{spec['key']} expected ABLs {EXPECTED_ABLS}, found {observed_abls}."
        )

    valid_count_audit_rows.append(
        {
            "batch": spec["key"],
            "RT_window_rows": len(valid_rt_df),
            "excluded_nonchoice_rows": len(valid_rt_df) - len(completed_choice_df),
            "completed_OFF": completed_counts[0],
            "completed_ON": completed_counts[1],
            "correct_OFF": correct_counts[0],
            "correct_ON": correct_counts[1],
        }
    )

    psychometric_payload = {}
    correct_chronometric_payload = {}
    all_choice_chronometric_payload = {}

    for led_trial in [0, 1]:
        completed_condition_df = completed_choice_df.loc[
            completed_choice_df["LED_trial"].eq(led_trial)
        ].copy()
        correct_condition_df = correct_choice_df.loc[
            correct_choice_df["LED_trial"].eq(led_trial)
        ].copy()

        observed_signed_ilds = sorted(
            completed_condition_df["ILD"].dropna().astype(int).unique().tolist()
        )
        if observed_signed_ilds != SIGNED_ILDS:
            raise RuntimeError(
                f"{spec['key']} {LED_LABELS[led_trial]} expected signed ILDs "
                f"{SIGNED_ILDS}, found {observed_signed_ilds}."
            )
        observed_abs_ilds = sorted(
            completed_condition_df["abs_ILD"]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        if observed_abs_ilds != ABS_ILDS:
            raise RuntimeError(
                f"{spec['key']} {LED_LABELS[led_trial]} expected absolute ILDs "
                f"{ABS_ILDS}, found {observed_abs_ilds}."
            )

        psycho_summary = (
            completed_condition_df.groupby("ILD", sort=True)["right_choice"]
            .agg(["size", "sum", "mean"])
            .reindex(SIGNED_ILDS)
        )
        if psycho_summary.isna().any(axis=None):
            raise RuntimeError(
                f"{spec['key']} {LED_LABELS[led_trial]} has an empty "
                "psychometric cell."
            )
        if int(psycho_summary["size"].sum()) != len(completed_condition_df):
            raise RuntimeError(f"{spec['key']} psychometric count mismatch.")
        if not psycho_summary["mean"].between(0.0, 1.0).all():
            raise RuntimeError(f"{spec['key']} psychometric probability is invalid.")

        psychometric_payload[led_trial] = {
            "ILD": np.asarray(SIGNED_ILDS, dtype=float),
            "probability_right": psycho_summary["mean"].to_numpy(dtype=float),
        }
        for ild, row in psycho_summary.iterrows():
            psychometric_audit_rows.append(
                {
                    "batch": spec["key"],
                    "condition": LED_LABELS[led_trial],
                    "ILD": int(ild),
                    "n": int(row["size"]),
                    "right_choices": int(row["sum"]),
                    "probability_right": float(row["mean"]),
                }
            )

        for chrono_name, chrono_df, chrono_payload in [
            (
                "correct choices",
                correct_condition_df,
                correct_chronometric_payload,
            ),
            (
                "all valid trials",
                completed_condition_df,
                all_choice_chronometric_payload,
            ),
        ]:
            chrono_summary = (
                chrono_df.groupby("abs_ILD", sort=True)["RTwrtStim"]
                .agg(["size", "mean"])
                .reindex(ABS_ILDS)
            )
            if chrono_summary.isna().any(axis=None):
                raise RuntimeError(
                    f"{spec['key']} {LED_LABELS[led_trial]} {chrono_name} "
                    "has an empty or non-finite chronometric cell."
                )
            if int(chrono_summary["size"].sum()) != len(chrono_df):
                raise RuntimeError(f"{spec['key']} chronometric count mismatch.")
            if not chrono_summary["mean"].between(
                RT_WINDOW_S[0], RT_WINDOW_S[1]
            ).all():
                raise RuntimeError(f"{spec['key']} chronometric mean is invalid.")
            chrono_payload[led_trial] = {
                "abs_ILD": np.asarray(ABS_ILDS, dtype=float),
                "mean_RT": chrono_summary["mean"].to_numpy(dtype=float),
            }
            for abs_ild, row in chrono_summary.iterrows():
                chronometric_audit_rows.append(
                    {
                        "batch": spec["key"],
                        "population": chrono_name,
                        "condition": LED_LABELS[led_trial],
                        "abs_ILD": int(abs_ild),
                        "n": int(row["size"]),
                        "mean_RT_s": float(row["mean"]),
                    }
                )

    batch_payloads.append(
        {
            "spec": spec,
            "animals": animals,
            "n_sessions": n_sessions,
            "timing": timing_payload,
            "aborts": abort_payload,
            "psychometric": psychometric_payload,
            "correct_chronometric": correct_chronometric_payload,
            "all_choice_chronometric": all_choice_chronometric_payload,
        }
    )


# %%
############ Plot the pooled 4 x 8 comparison ############
column_titles = [
    "intended_fix",
    "LED_onset_time",
    "intended_fix - LED_onset_time",
    "Abort time relative to\nLED_onset_time",
    "Abort time relative to\nintended_fix - LED_onset_time",
    "Psychometric",
    "Chronometric\ncorrect choices",
    "Chronometric\nall valid trials",
]
timing_keys = [
    "intended_fix",
    "LED_onset_time",
    "intended_fix_minus_LED_onset_time",
]

fig, axes = plt.subplots(
    len(batch_payloads),
    len(column_titles),
    figsize=(34, 13),
    sharex="col",
)

for column_index, title in enumerate(column_titles):
    axes[0, column_index].set_title(title, fontsize=12)

for row_index, payload in enumerate(batch_payloads):
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
            abort_result["led_onset_heights"],
            ALIGNED_FULL_BINS,
            color=LED_COLORS[led_trial],
            linewidth=1.4,
        )
        axes[row_index, 4].stairs(
            abort_result["gap_heights"],
            ALIGNED_FULL_BINS,
            color=LED_COLORS[led_trial],
            linewidth=1.4,
        )

        psycho_result = payload["psychometric"][led_trial]
        axes[row_index, 5].plot(
            psycho_result["ILD"],
            psycho_result["probability_right"],
            color=LED_COLORS[led_trial],
            marker="o",
            markersize=3.4,
            linewidth=1.4,
        )

        for column_index, chrono_key in [
            (6, "correct_chronometric"),
            (7, "all_choice_chronometric"),
        ]:
            chrono_result = payload[chrono_key][led_trial]
            axes[row_index, column_index].plot(
                chrono_result["abs_ILD"],
                chrono_result["mean_RT"],
                color=LED_COLORS[led_trial],
                marker="o",
                markersize=3.4,
                linewidth=1.4,
            )

    axes[row_index, 0].text(
        0.97,
        0.94,
        f"n = {payload['timing']['intended_fix']['counts'].sum():,}",
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
        fontsize=10,
    )
    axes[row_index, 5].set_ylabel("P(right)", fontsize=9)
    for column_index in [6, 7]:
        axes[row_index, column_index].set_ylabel("Mean RT (s)", fontsize=9)

for column_index in range(3):
    column_y_max = max(
        float(payload["timing"][timing_keys[column_index]]["heights"].max())
        for payload in batch_payloads
    )
    for row_index in range(len(batch_payloads)):
        axes[row_index, column_index].set_ylim(0, 1.10 * column_y_max)

aligned_y_max = max(
    float(payload["aborts"][led_trial][height_key].max())
    for payload in batch_payloads
    for led_trial in [0, 1]
    for height_key in ["led_onset_heights", "gap_heights"]
)
for row_index in range(len(batch_payloads)):
    for column_index in [3, 4]:
        axes[row_index, column_index].set_ylim(0, 1.15 * aligned_y_max)

    axes[row_index, 5].set_xlim(-17, 17)
    axes[row_index, 5].set_ylim(0, 1)
    axes[row_index, 5].set_xticks([-16, -8, 0, 8, 16])
    axes[row_index, 5].set_yticks([0, 0.5, 1])

    for column_index in [6, 7]:
        axes[row_index, column_index].set_xscale("log", base=2)
        axes[row_index, column_index].set_xlim(0.8, 20)
        axes[row_index, column_index].set_ylim(CHRONOMETRIC_Y_LIMITS_S)
        axes[row_index, column_index].set_xticks(ABS_ILDS)
        axes[row_index, column_index].xaxis.set_major_formatter(ScalarFormatter())
        axes[row_index, column_index].xaxis.set_minor_locator(NullLocator())

for ax in axes.flat:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.16, linewidth=0.6)
    ax.tick_params(axis="both", labelsize=8)

for column_index in range(3):
    axes[-1, column_index].set_xlabel("Time (s)", fontsize=10)
for column_index in [3, 4]:
    axes[-1, column_index].set_xlabel("Aligned abort time (s)", fontsize=10)
axes[-1, 5].set_xlabel("ILD (dB)", fontsize=10)
for column_index in [6, 7]:
    axes[-1, column_index].set_xlabel("|ILD| (dB)", fontsize=10)

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
    "LED timing, fixation-abort, psychometric, and chronometric comparisons\n"
    "training level 16; repeat_trial = 0, 2, or NaN",
    fontsize=15,
    y=0.995,
)
fig.text(
    0.5,
    0.012,
    "Columns 1-3 use all LED-ON trials. Columns 4-5 use event-3 aborts and "
    "condition-trial denominators. Columns 6-8 use derived RT in [0, 1] s; "
    "chronometric curves show pooled-trial means without error bars.",
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=(0.012, 0.04, 1.0, 0.93), w_pad=1.2, h_pad=1.3)
fig.savefig(OUTPUT_FIGURE_PATH, dpi=OUTPUT_DPI, bbox_inches="tight")

if SHOW_PLOT:
    plt.show()
else:
    plt.close(fig)


# %%
############ Print the complete numerical audit ############
timing_audit_df = pd.DataFrame(timing_audit_rows)
abort_audit_df = pd.DataFrame(abort_audit_rows)
valid_count_audit_df = pd.DataFrame(valid_count_audit_rows)
rt_consistency_audit_df = pd.DataFrame(rt_consistency_audit_rows)
psychometric_audit_df = pd.DataFrame(psychometric_audit_rows)
chronometric_audit_df = pd.DataFrame(chronometric_audit_rows)

print("Sources:")
print(f"  {LED7_MAT_PATH} :: totalout_stGtACRII")
print(f"  {LED8_MAT_PATH} :: totalout")
print(
    "Base filters: training_level=16; row-specific session_type; "
    "repeat_trial in {0, 2, NaN}."
)
print(
    "Valid-panel filters: derived RTwrtStim=timed_fix-intended_fix in "
    "[0, 1] s; psychometric/all-choice chronometric success in {-1, 1}; "
    "correct chronometric success=1."
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
print("\nDerived RT versus timed_RT audit")
print(
    rt_consistency_audit_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.9f}",
    )
)
print("\nValid-choice count audit")
print(valid_count_audit_df.to_string(index=False))
print("\nPsychometric audit")
print(
    psychometric_audit_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)
print("\nChronometric audit")
print(
    chronometric_audit_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)
print(f"\nSaved figure: {OUTPUT_FIGURE_PATH.resolve()}")
