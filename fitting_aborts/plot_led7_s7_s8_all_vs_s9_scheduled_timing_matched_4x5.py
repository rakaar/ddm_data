# %%
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from led7_scheduled_timing_matching_utils import (
    largest_remainder_counts,
    select_rows_matching_two_margins,
)

try:
    from matio import load_from_mat
except ImportError as exc:
    raise ImportError(
        "This script needs the isolated mat-io decoder. Run it as:\n"
        "env PYTHONPATH=/tmp/codex_mat_io .venv/bin/python "
        "fitting_aborts/"
        "plot_led7_s7_s8_all_vs_s9_scheduled_timing_matched_4x5.py"
    ) from exc


# %%
############ Parameters (edit here) ############
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

INPUT_MAT_PATH = REPO_ROOT / "raw_data" / "outMatrix_LED7_latest.mat"
TABLE_NAME = "totalout_stGtACRII"
OUTPUT_FIGURE_PATH = (
    SCRIPT_DIR
    / "led7_s7_s8_all_vs_s9_scheduled_timing_matched_4x5.png"
)
OUTPUT_COMPACT_FIGURE_PATH = (
    SCRIPT_DIR
    / "led7_s7_s8_all_vs_s9_scheduled_timing_matched_4x3.png"
)
OUTPUT_ABORT_FIGURE_PATH = (
    SCRIPT_DIR
    / "led7_s7_s8_all_vs_s9_scheduled_timing_matched_abort_rate_2x2.png"
)

TRAINING_LEVEL = 16
ALLOWED_REPEAT_TRIALS = {0, 2}
COMPARISON_SESSION_TYPES = [7, 8]
REFERENCE_SESSION_TYPE = 9
LED_TRIAL_VALUES = [0, 1]
LED_COLORS = {0: "tab:blue", 1: "tab:red"}
LED_LABELS = {0: "LED OFF", 1: "LED ON"}

MATCH_SAMPLE_SIZE = 5_500
RANDOM_SEED = 20_260_909
BIN_WIDTH_S = 0.020
MAX_KS_DISTANCE = 0.015
MAX_WASSERSTEIN_MS = 10.0

# Matching uses these two marginal schedules only. No stimulus, choice,
# success, response, reaction-time, or abort field enters row selection.
MATCHING_COLUMNS = ["intended_fix", "effective_scheduled_onset"]
MATCH_INTENDED_BINS = np.arange(200, 2200 + 20, 20, dtype=float) / 1000
MATCH_ONSET_BINS = np.arange(100, 1100 + 20, 20, dtype=float) / 1000

PLOT_INTENDED_BINS = np.arange(0, 2200 + 20, 20, dtype=float) / 1000
PLOT_ONSET_BINS = np.arange(0, 2200 + 20, 20, dtype=float) / 1000
PLOT_GAP_BINS = np.arange(0, 2200 + 20, 20, dtype=float) / 1000
ABORT_FROM_FIXATION_BINS = (
    np.arange(0, 2200 + 20, 20, dtype=float) / 1000
)
ABORT_FROM_ONSET_FULL_BINS = (
    np.arange(-2000, 2000 + 20, 20, dtype=float) / 1000
)
ABORT_FROM_ONSET_DISPLAY_LIMITS_S = (-1.0, 1.0)

OUTPUT_DPI = 250
SHOW_PLOT = False

EXPECTED_ANIMALS = [90, 92, 93, 98, 99, 100, 102, 103]
EXPECTED = {
    7: {
        "animal_sessions": 199,
        "trials": {0: 92_057, 1: 46_383},
    },
    8: {
        "animal_sessions": 215,
        "trials": {0: 92_784, 1: 55_644},
    },
    9: {
        "animal_sessions": 191,
        "trials": {0: 61_408, 1: 39_380},
    },
}

ROW_SPECS = [
    {
        "key": "s7_all",
        "session_type": 7,
        "matched": False,
        "label": "LED7 session type 7\nall eligible trials",
    },
    {
        "key": "s8_all",
        "session_type": 8,
        "matched": False,
        "label": "LED7 session type 8\nall eligible trials",
    },
    {
        "key": "s7_matched",
        "session_type": 7,
        "matched": True,
        "label": "LED7 session type 7\nmatched to session type 9",
    },
    {
        "key": "s8_matched",
        "session_type": 8,
        "matched": True,
        "label": "LED7 session type 8\nmatched to session type 9",
    },
]


# %%
############ Reused histogram and matching helpers ############
def density_histogram(values, bin_edges):
    counts, _ = np.histogram(values, bins=bin_edges)
    if int(counts.sum()) != len(values):
        raise RuntimeError(
            f"Histogram contains {counts.sum():,} of {len(values):,} values."
        )
    heights = counts.astype(float) / (len(values) * BIN_WIDTH_S)
    area = float(np.sum(heights * np.diff(bin_edges)))
    return counts, heights, area


def scaled_abort_histogram(values, bin_edges, denominator):
    counts, _ = np.histogram(values, bins=bin_edges)
    if int(counts.sum()) != len(values):
        raise RuntimeError(
            f"Abort histogram contains {counts.sum():,} of "
            f"{len(values):,} finite aborts."
        )
    heights = counts.astype(float) / (denominator * BIN_WIDTH_S)
    area = float(np.sum(heights * np.diff(bin_edges)))
    return counts, heights, area


# %%
############ Load and filter LED7 session types 7, 8, and 9 ############
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
    "session",
    "training_level",
    "session_type",
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

# Keep the requested filter order explicit.
training_df = raw_df.loc[
    raw_df["training_level"].eq(TRAINING_LEVEL)
].copy()
session_df = training_df.loc[
    training_df["session_type"].isin(
        COMPARISON_SESSION_TYPES + [REFERENCE_SESSION_TYPE]
    )
].copy()
repeat_df = session_df.loc[
    session_df["repeat_trial"].isin(ALLOWED_REPEAT_TRIALS)
    | session_df["repeat_trial"].isna()
].copy()
filtered_df = repeat_df.loc[
    repeat_df["LED_trial"].isin(LED_TRIAL_VALUES)
].copy()

if not filtered_df["training_level"].eq(TRAINING_LEVEL).all():
    raise RuntimeError("Filtered table contains another training level.")
if not filtered_df["session_type"].isin(EXPECTED).all():
    raise RuntimeError("Filtered table contains another session type.")
observed_repeat_values = set(
    filtered_df["repeat_trial"].dropna().astype(int).unique()
)
if not observed_repeat_values.issubset(ALLOWED_REPEAT_TRIALS):
    raise RuntimeError(
        f"Unexpected repeat_trial values: {observed_repeat_values}"
    )
if set(filtered_df["LED_trial"].astype(int).unique()) != {0, 1}:
    raise RuntimeError("Expected both LED_trial values 0 and 1.")

session_frames = {}
source_audit_rows = []
for session_type in COMPARISON_SESSION_TYPES + [REFERENCE_SESSION_TYPE]:
    current = filtered_df.loc[
        filtered_df["session_type"].eq(session_type)
    ].copy()
    animals = sorted(current["animal"].dropna().astype(int).unique().tolist())
    if animals != EXPECTED_ANIMALS:
        raise RuntimeError(
            f"Session type {session_type} expected animals {EXPECTED_ANIMALS}, "
            f"found {animals}."
        )
    animal_sessions = current[["animal", "session"]].drop_duplicates().shape[0]
    if animal_sessions != EXPECTED[session_type]["animal_sessions"]:
        raise RuntimeError(
            f"Session type {session_type} expected "
            f"{EXPECTED[session_type]['animal_sessions']} animal-session pairs, "
            f"found {animal_sessions}."
        )
    trial_counts = {
        led_trial: int(current["LED_trial"].eq(led_trial).sum())
        for led_trial in LED_TRIAL_VALUES
    }
    if trial_counts != EXPECTED[session_type]["trials"]:
        raise RuntimeError(
            f"Session type {session_type} expected LED-group counts "
            f"{EXPECTED[session_type]['trials']}, found {trial_counts}."
        )

    current["effective_scheduled_onset"] = (
        current["LED_onset_time"]
        if session_type == REFERENCE_SESSION_TYPE
        else current["intended_fix"] - current["LED_onset_time"]
    )
    current["scheduled_onset_to_stimulus_gap"] = (
        current["intended_fix"] - current["effective_scheduled_onset"]
    )
    if not np.isfinite(
        current[
            [
                "intended_fix",
                "effective_scheduled_onset",
                "scheduled_onset_to_stimulus_gap",
            ]
        ]
    ).all(axis=None):
        raise RuntimeError(f"Session type {session_type} has non-finite timing.")
    if (current["effective_scheduled_onset"] < 0).any():
        raise RuntimeError(f"Session type {session_type} has a negative onset.")
    if (current["scheduled_onset_to_stimulus_gap"] < -1e-12).any():
        raise RuntimeError(
            f"Session type {session_type} has onset after intended fixation."
        )

    finite_response_time = np.isfinite(
        current[["timed_fix", "intended_fix"]]
    ).all(axis=1)
    negative_rt = (
        current.loc[finite_response_time, "timed_fix"]
        < current.loc[finite_response_time, "intended_fix"]
    )
    event_3 = current.loc[finite_response_time, "abort_event"].eq(3)
    if not np.array_equal(negative_rt.to_numpy(), event_3.to_numpy()):
        raise RuntimeError(
            f"Session type {session_type} has finite RT/event-3 mismatches."
        )

    for led_trial in LED_TRIAL_VALUES:
        source_audit_rows.append(
            {
                "session_type": session_type,
                "LED_group": LED_LABELS[led_trial],
                "trials": trial_counts[led_trial],
                "animal_sessions": animal_sessions,
            }
        )
    session_frames[session_type] = current


# %%
############ Build the pooled session-type-9 timing target ############
reference_df = session_frames[REFERENCE_SESSION_TYPE].copy()
if len(reference_df) != 100_788:
    raise RuntimeError(
        f"Expected 100,788 pooled session-type-9 rows, found "
        f"{len(reference_df):,}."
    )

reference_intended_counts, _ = np.histogram(
    reference_df["intended_fix"], bins=MATCH_INTENDED_BINS
)
reference_onset_counts, _ = np.histogram(
    reference_df["effective_scheduled_onset"], bins=MATCH_ONSET_BINS
)
if int(reference_intended_counts.sum()) != len(reference_df):
    raise RuntimeError("Reference intended-fix values exceed matching support.")
if int(reference_onset_counts.sum()) != len(reference_df):
    raise RuntimeError("Reference onset values exceed matching support.")

target_intended_counts = largest_remainder_counts(
    reference_intended_counts, MATCH_SAMPLE_SIZE
)
target_onset_counts = largest_remainder_counts(
    reference_onset_counts, MATCH_SAMPLE_SIZE
)

reference_joint_counts, _, _ = np.histogram2d(
    reference_df["intended_fix"],
    reference_df["effective_scheduled_onset"],
    bins=[MATCH_INTENDED_BINS, MATCH_ONSET_BINS],
)
reference_joint_probability = reference_joint_counts / len(reference_df)
reference_correlation = float(
    reference_df[
        ["intended_fix", "effective_scheduled_onset"]
    ].corr().iloc[0, 1]
)


# %%
############ Match each session-type and LED assignment independently ############
matched_frames = {}
matching_audit_rows = []
matching_rng = np.random.default_rng(RANDOM_SEED)

for session_type in COMPARISON_SESSION_TYPES:
    for led_trial in LED_TRIAL_VALUES:
        candidate_df = session_frames[session_type].loc[
            session_frames[session_type]["LED_trial"].eq(led_trial)
        ].copy()
        timing_only = candidate_df[MATCHING_COLUMNS].copy()
        selected_indices, flow_audit = select_rows_matching_two_margins(
            timing_frame=timing_only,
            target_intended_counts=target_intended_counts,
            target_onset_counts=target_onset_counts,
            rng=matching_rng,
            matching_columns=MATCHING_COLUMNS,
            intended_bins=MATCH_INTENDED_BINS,
            onset_bins=MATCH_ONSET_BINS,
            sample_size=MATCH_SAMPLE_SIZE,
            bin_width_s=BIN_WIDTH_S,
        )
        matched_df = candidate_df.loc[selected_indices].copy()

        if len(matched_df) != MATCH_SAMPLE_SIZE:
            raise RuntimeError("Matched frame has the wrong row count.")
        if not matched_df.index.is_unique:
            raise RuntimeError("Matched frame repeats a source row.")
        if not matched_df["LED_trial"].eq(led_trial).all():
            raise RuntimeError("Matched frame crossed LED assignments.")
        if not matched_df.index.isin(candidate_df.index).all():
            raise RuntimeError("Matched frame contains a non-candidate row.")

        intended_ks = float(
            ks_2samp(
                matched_df["intended_fix"],
                reference_df["intended_fix"],
                method="asymp",
            ).statistic
        )
        intended_wasserstein_ms = float(
            1000
            * wasserstein_distance(
                matched_df["intended_fix"],
                reference_df["intended_fix"],
            )
        )
        onset_ks = float(
            ks_2samp(
                matched_df["effective_scheduled_onset"],
                reference_df["effective_scheduled_onset"],
                method="asymp",
            ).statistic
        )
        onset_wasserstein_ms = float(
            1000
            * wasserstein_distance(
                matched_df["effective_scheduled_onset"],
                reference_df["effective_scheduled_onset"],
            )
        )
        if intended_ks > MAX_KS_DISTANCE:
            raise RuntimeError(
                f"Session {session_type} {LED_LABELS[led_trial]} intended-fix "
                f"KS distance {intended_ks:.6f} exceeds {MAX_KS_DISTANCE}."
            )
        if intended_wasserstein_ms > MAX_WASSERSTEIN_MS:
            raise RuntimeError(
                f"Session {session_type} {LED_LABELS[led_trial]} intended-fix "
                f"Wasserstein distance {intended_wasserstein_ms:.6f} ms "
                f"exceeds {MAX_WASSERSTEIN_MS} ms."
            )
        if onset_ks > MAX_KS_DISTANCE:
            raise RuntimeError(
                f"Session {session_type} {LED_LABELS[led_trial]} onset KS "
                f"distance {onset_ks:.6f} exceeds {MAX_KS_DISTANCE}."
            )
        if onset_wasserstein_ms > MAX_WASSERSTEIN_MS:
            raise RuntimeError(
                f"Session {session_type} {LED_LABELS[led_trial]} onset "
                f"Wasserstein distance {onset_wasserstein_ms:.6f} ms "
                f"exceeds {MAX_WASSERSTEIN_MS} ms."
            )

        selected_joint_counts, _, _ = np.histogram2d(
            matched_df["intended_fix"],
            matched_df["effective_scheduled_onset"],
            bins=[MATCH_INTENDED_BINS, MATCH_ONSET_BINS],
        )
        selected_joint_probability = selected_joint_counts / MATCH_SAMPLE_SIZE
        joint_total_variation = float(
            0.5
            * np.abs(
                selected_joint_probability - reference_joint_probability
            ).sum()
        )
        selected_correlation = float(
            matched_df[
                ["intended_fix", "effective_scheduled_onset"]
            ].corr().iloc[0, 1]
        )

        matching_audit_rows.append(
            {
                "session_type": session_type,
                "LED_group": LED_LABELS[led_trial],
                **flow_audit,
                "selected_rows": len(matched_df),
                "intended_KS": intended_ks,
                "intended_W_ms": intended_wasserstein_ms,
                "onset_KS": onset_ks,
                "onset_W_ms": onset_wasserstein_ms,
                "joint_20ms_TV": joint_total_variation,
                "reference_r": reference_correlation,
                "selected_r": selected_correlation,
                "delta_r": selected_correlation - reference_correlation,
            }
        )
        matched_frames[(session_type, led_trial)] = matched_df


# %%
############ Build before/after timing and abort payloads ############
timing_definitions = [
    ("intended_fix", PLOT_INTENDED_BINS),
    ("effective_scheduled_onset", PLOT_ONSET_BINS),
    ("scheduled_onset_to_stimulus_gap", PLOT_GAP_BINS),
]

reference_timing_payload = {}
for timing_name, bin_edges in timing_definitions:
    values = reference_df[timing_name].to_numpy(dtype=float)
    counts, heights, area = density_histogram(values, bin_edges)
    if not np.isclose(area, 1.0, atol=1e-14, rtol=0):
        raise RuntimeError(f"Reference {timing_name} area is not one.")
    reference_timing_payload[timing_name] = {
        "counts": counts,
        "heights": heights,
        "bin_edges": bin_edges,
    }

row_payloads = []
timing_audit_rows = []
abort_audit_rows = []

for row_spec in ROW_SPECS:
    session_type = row_spec["session_type"]
    group_frames = {}
    for led_trial in LED_TRIAL_VALUES:
        if row_spec["matched"]:
            group_df = matched_frames[(session_type, led_trial)].copy()
        else:
            group_df = session_frames[session_type].loc[
                session_frames[session_type]["LED_trial"].eq(led_trial)
            ].copy()
        group_frames[led_trial] = group_df

    timing_payload = {}
    abort_payload = {}
    for led_trial, group_df in group_frames.items():
        timing_payload[led_trial] = {}
        for timing_name, bin_edges in timing_definitions:
            values = group_df[timing_name].to_numpy(dtype=float)
            counts, heights, area = density_histogram(values, bin_edges)
            if not np.isclose(area, 1.0, atol=1e-14, rtol=0):
                raise RuntimeError(
                    f"{row_spec['key']} {LED_LABELS[led_trial]} "
                    f"{timing_name} area is not one."
                )
            timing_payload[led_trial][timing_name] = {
                "counts": counts,
                "heights": heights,
                "bin_edges": bin_edges,
            }
            timing_audit_rows.append(
                {
                    "row": row_spec["key"],
                    "LED_group": LED_LABELS[led_trial],
                    "distribution": timing_name,
                    "n": len(values),
                    "min_s": float(values.min()),
                    "median_s": float(np.median(values)),
                    "max_s": float(values.max()),
                    "area": area,
                }
            )

        coded_abort_df = group_df.loc[group_df["abort_event"].eq(3)].copy()
        finite_abort_df = coded_abort_df.loc[
            np.isfinite(coded_abort_df["timed_fix"])
        ].copy()
        if not (
            finite_abort_df["timed_fix"] < finite_abort_df["intended_fix"]
        ).all():
            raise RuntimeError(
                f"{row_spec['key']} {LED_LABELS[led_trial]} has a finite "
                "event-3 row at or after intended fixation."
            )

        abort_from_fixation = finite_abort_df["timed_fix"].to_numpy(dtype=float)
        abort_from_onset = (
            finite_abort_df["timed_fix"]
            - finite_abort_df["effective_scheduled_onset"]
        ).to_numpy(dtype=float)
        fixation_counts, fixation_heights, fixation_area = (
            scaled_abort_histogram(
                abort_from_fixation,
                ABORT_FROM_FIXATION_BINS,
                denominator=len(group_df),
            )
        )
        onset_counts, onset_heights, onset_area = scaled_abort_histogram(
            abort_from_onset,
            ABORT_FROM_ONSET_FULL_BINS,
            denominator=len(group_df),
        )
        expected_area = len(finite_abort_df) / len(group_df)
        if not np.isclose(fixation_area, expected_area, atol=1e-14, rtol=0):
            raise RuntimeError(f"{row_spec['key']} fixation-area mismatch.")
        if not np.isclose(onset_area, expected_area, atol=1e-14, rtol=0):
            raise RuntimeError(f"{row_spec['key']} onset-area mismatch.")

        pre_onset_fraction = float(
            np.count_nonzero(abort_from_onset < 0) / len(group_df)
        )
        outside_display = int(
            np.count_nonzero(
                (abort_from_onset < ABORT_FROM_ONSET_DISPLAY_LIMITS_S[0])
                | (abort_from_onset > ABORT_FROM_ONSET_DISPLAY_LIMITS_S[1])
            )
        )
        abort_payload[led_trial] = {
            "denominator": len(group_df),
            "coded_aborts": len(coded_abort_df),
            "finite_aborts": len(finite_abort_df),
            "abort_fraction": expected_area,
            "fixation_counts": fixation_counts,
            "fixation_heights": fixation_heights,
            "onset_counts": onset_counts,
            "onset_heights": onset_heights,
        }
        abort_audit_rows.append(
            {
                "row": row_spec["key"],
                "LED_group": LED_LABELS[led_trial],
                "denominator_trials": len(group_df),
                "coded_event_3_aborts": len(coded_abort_df),
                "finite_plotted_aborts": len(finite_abort_df),
                "missing_abort_timing": len(coded_abort_df)
                - len(finite_abort_df),
                "abort_fraction": expected_area,
                "pre_scheduled_onset_fraction": pre_onset_fraction,
                "outside_onset_display": outside_display,
                "fixation_hist_area": fixation_area,
                "onset_hist_area": onset_area,
            }
        )

    row_payloads.append(
        {
            "spec": row_spec,
            "timing": timing_payload,
            "aborts": abort_payload,
        }
    )


# %%
############ Plot the 4 x 5 before/after comparison ############
column_titles = [
    "Intended fixation",
    "Scheduled onset\nfrom fixation",
    "Scheduled onset-to-stimulus\ngap",
    "Fixation-abort time\nfrom fixation",
    "Fixation-abort time\nfrom scheduled onset",
]
timing_keys = [
    "intended_fix",
    "effective_scheduled_onset",
    "scheduled_onset_to_stimulus_gap",
]

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
        for led_trial in LED_TRIAL_VALUES:
            result = payload["timing"][led_trial][timing_key]
            axes[row_index, column_index].stairs(
                result["heights"],
                result["bin_edges"],
                color=LED_COLORS[led_trial],
                linewidth=1.35,
            )
        reference_result = reference_timing_payload[timing_key]
        axes[row_index, column_index].stairs(
            reference_result["heights"],
            reference_result["bin_edges"],
            color="black",
            linestyle="--",
            linewidth=1.15,
            alpha=0.9,
        )

    for led_trial in LED_TRIAL_VALUES:
        abort_result = payload["aborts"][led_trial]
        axes[row_index, 3].stairs(
            abort_result["fixation_heights"],
            ABORT_FROM_FIXATION_BINS,
            color=LED_COLORS[led_trial],
            linewidth=1.35,
        )
        axes[row_index, 4].stairs(
            abort_result["onset_heights"],
            ABORT_FROM_ONSET_FULL_BINS,
            color=LED_COLORS[led_trial],
            linewidth=1.35,
        )

    off_n = payload["aborts"][0]["denominator"]
    on_n = payload["aborts"][1]["denominator"]
    axes[row_index, 0].text(
        0.97,
        0.94,
        f"OFF n = {off_n:,}\nON n = {on_n:,}",
        transform=axes[row_index, 0].transAxes,
        ha="right",
        va="top",
        fontsize=7.8,
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
    axes[row_index, 4].axvline(
        0,
        color="black",
        linestyle=":",
        linewidth=0.9,
        alpha=0.75,
    )

    axes[row_index, 0].set_ylabel(
        f"{payload['spec']['label']}\nDensity (s$^{{-1}}$)",
        fontsize=9.5,
    )

for column_index, timing_key in enumerate(timing_keys):
    column_y_max = max(
        [
            float(
                payload["timing"][led_trial][timing_key]["heights"].max()
            )
            for payload in row_payloads
            for led_trial in LED_TRIAL_VALUES
        ]
        + [float(reference_timing_payload[timing_key]["heights"].max())]
    )
    for row_index in range(len(row_payloads)):
        axes[row_index, column_index].set_ylim(0, 1.12 * column_y_max)

fixation_abort_y_max = max(
    float(payload["aborts"][led_trial]["fixation_heights"].max())
    for payload in row_payloads
    for led_trial in LED_TRIAL_VALUES
)
onset_abort_y_max = max(
    float(payload["aborts"][led_trial]["onset_heights"].max())
    for payload in row_payloads
    for led_trial in LED_TRIAL_VALUES
)
for row_index in range(len(row_payloads)):
    axes[row_index, 3].set_ylim(0, 1.15 * fixation_abort_y_max)
    axes[row_index, 4].set_ylim(0, 1.15 * onset_abort_y_max)

for row_index in range(len(row_payloads)):
    axes[row_index, 0].set_xlim(PLOT_INTENDED_BINS[[0, -1]])
    axes[row_index, 1].set_xlim(PLOT_ONSET_BINS[[0, -1]])
    axes[row_index, 2].set_xlim(PLOT_GAP_BINS[[0, -1]])
    axes[row_index, 3].set_xlim(ABORT_FROM_FIXATION_BINS[[0, -1]])
    axes[row_index, 4].set_xlim(ABORT_FROM_ONSET_DISPLAY_LIMITS_S)

for ax in axes.flat:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.16, linewidth=0.6)
    ax.tick_params(axis="both", labelsize=8)

for column_index in range(3):
    axes[-1, column_index].set_xlabel("Time (s)", fontsize=10)
axes[-1, 3].set_xlabel("Abort time from fixation (s)", fontsize=10)
axes[-1, 4].set_xlabel("Abort time from scheduled onset (s)", fontsize=10)

fig.legend(
    handles=[
        Line2D([0], [0], color=LED_COLORS[0], lw=1.5, label=LED_LABELS[0]),
        Line2D([0], [0], color=LED_COLORS[1], lw=1.5, label=LED_LABELS[1]),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            lw=1.3,
            label="Session type 9 timing reference",
        ),
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, 0.965),
    ncol=3,
    frameon=False,
    fontsize=10,
)
fig.suptitle(
    "LED7 fixation aborts before and after matching scheduled timing\n"
    "session types 7 and 8 matched to pooled session type 9",
    fontsize=15,
    y=0.995,
)
fig.text(
    0.5,
    0.010,
    "training_level = 16; repeat_trial = 0, 2, or NaN; 20 ms bins. "
    "Matching uses intended fixation and scheduled onset only, before "
    "selecting abort_event = 3. For OFF, onset is the scheduled/pseudo-onset; "
    "actual illumination occurs only for ON. Aligned panels display -1 to "
    "+1 s; full-area normalization uses -2 to +2 s.",
    ha="center",
    fontsize=8.5,
)
fig.tight_layout(rect=(0.015, 0.045, 1.0, 0.935), w_pad=1.35, h_pad=1.25)
fig.savefig(OUTPUT_FIGURE_PATH, dpi=OUTPUT_DPI, bbox_inches="tight")

if SHOW_PLOT:
    plt.show()
else:
    plt.close(fig)


# %%
############ Plot the compact 4 x 3 presentation figure ############
compact_column_titles = [
    "intended_fix",
    "intended_fix - LED_onset_time",
    "abort rate aligned to LED onset\nfrom t = 0",
]
compact_timing_keys = ["intended_fix", "effective_scheduled_onset"]

compact_fig, compact_axes = plt.subplots(
    len(row_payloads),
    len(compact_column_titles),
    figsize=(14.5, 13),
    sharex="col",
)

for column_index, title in enumerate(compact_column_titles):
    compact_axes[0, column_index].set_title(title, fontsize=12)

for row_index, payload in enumerate(row_payloads):
    for column_index, timing_key in enumerate(compact_timing_keys):
        for led_trial in LED_TRIAL_VALUES:
            result = payload["timing"][led_trial][timing_key]
            compact_axes[row_index, column_index].stairs(
                result["heights"],
                result["bin_edges"],
                color=LED_COLORS[led_trial],
                linewidth=1.4,
            )
        reference_result = reference_timing_payload[timing_key]
        compact_axes[row_index, column_index].stairs(
            reference_result["heights"],
            reference_result["bin_edges"],
            color="black",
            linestyle="--",
            linewidth=1.2,
            alpha=0.9,
        )

    for led_trial in LED_TRIAL_VALUES:
        abort_result = payload["aborts"][led_trial]
        compact_axes[row_index, 2].stairs(
            abort_result["onset_heights"],
            ABORT_FROM_ONSET_FULL_BINS,
            color=LED_COLORS[led_trial],
            linewidth=1.4,
        )

    off_n = payload["aborts"][0]["denominator"]
    on_n = payload["aborts"][1]["denominator"]
    compact_axes[row_index, 0].text(
        0.97,
        0.94,
        f"OFF n = {off_n:,}\nON n = {on_n:,}",
        transform=compact_axes[row_index, 0].transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )

    off_area = payload["aborts"][0]["abort_fraction"]
    on_area = payload["aborts"][1]["abort_fraction"]
    compact_axes[row_index, 2].text(
        0.97,
        0.94,
        f"OFF area = {off_area:.3f}\nON area = {on_area:.3f}",
        transform=compact_axes[row_index, 2].transAxes,
        ha="right",
        va="top",
        fontsize=7.8,
    )
    compact_axes[row_index, 2].axvline(
        0,
        color="black",
        linestyle=":",
        linewidth=0.9,
        alpha=0.75,
    )
    compact_axes[row_index, 0].set_ylabel(
        f"{payload['spec']['label']}\nDensity (s$^{{-1}}$)",
        fontsize=9.5,
    )

for column_index, timing_key in enumerate(compact_timing_keys):
    column_y_max = max(
        [
            float(
                payload["timing"][led_trial][timing_key]["heights"].max()
            )
            for payload in row_payloads
            for led_trial in LED_TRIAL_VALUES
        ]
        + [float(reference_timing_payload[timing_key]["heights"].max())]
    )
    for row_index in range(len(row_payloads)):
        compact_axes[row_index, column_index].set_ylim(
            0, 1.12 * column_y_max
        )
        compact_axes[row_index, column_index].set_xlim(
            PLOT_INTENDED_BINS[[0, -1]]
            if column_index == 0
            else PLOT_ONSET_BINS[[0, -1]]
        )

for row_index in range(len(row_payloads)):
    compact_axes[row_index, 2].set_ylim(0, 1.15 * onset_abort_y_max)
    compact_axes[row_index, 2].set_xlim(
        ABORT_FROM_ONSET_DISPLAY_LIMITS_S
    )

for ax in compact_axes.flat:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.16, linewidth=0.6)
    ax.tick_params(axis="both", labelsize=8)

compact_axes[-1, 0].set_xlabel("Time (s)", fontsize=10)
compact_axes[-1, 1].set_xlabel("Time (s)", fontsize=10)
compact_axes[-1, 2].set_xlabel(
    "Abort time from LED onset (s)", fontsize=10
)

compact_fig.legend(
    handles=[
        Line2D([0], [0], color=LED_COLORS[0], lw=1.5, label=LED_LABELS[0]),
        Line2D([0], [0], color=LED_COLORS[1], lw=1.5, label=LED_LABELS[1]),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            lw=1.3,
            label="Session type 9 timing reference",
        ),
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, 0.965),
    ncol=3,
    frameon=False,
    fontsize=10,
)
compact_fig.suptitle(
    "LED7 fixation aborts before and after matching scheduled timing",
    fontsize=15,
    y=0.995,
)
compact_fig.text(
    0.5,
    0.010,
    "For session types 7/8, LED onset is intended_fix - LED_onset_time; "
    "the dashed session-type-9 reference uses raw LED_onset_time. Matching "
    "is performed before selecting abort_event = 3. OFF alignment uses the "
    "scheduled/pseudo-onset.",
    ha="center",
    fontsize=8.5,
)
compact_fig.tight_layout(
    rect=(0.02, 0.045, 1.0, 0.935),
    w_pad=1.35,
    h_pad=1.25,
)
compact_fig.savefig(
    OUTPUT_COMPACT_FIGURE_PATH,
    dpi=OUTPUT_DPI,
    bbox_inches="tight",
)

if SHOW_PLOT:
    plt.show()
else:
    plt.close(compact_fig)


# %%
############ Plot only the onset-aligned abort rates as a 2 x 2 ############
abort_grid_payloads = [
    [row_payloads[0], row_payloads[1]],
    [row_payloads[2], row_payloads[3]],
]
abort_grid_fig, abort_grid_axes = plt.subplots(
    2,
    2,
    figsize=(11.5, 8.5),
    sharex=True,
    sharey="row",
)

for column_index, session_type in enumerate(COMPARISON_SESSION_TYPES):
    abort_grid_axes[0, column_index].set_title(
        f"Session type {session_type}", fontsize=12
    )

for row_index in range(2):
    for column_index in range(2):
        payload = abort_grid_payloads[row_index][column_index]
        ax = abort_grid_axes[row_index, column_index]
        for led_trial in LED_TRIAL_VALUES:
            abort_result = payload["aborts"][led_trial]
            ax.stairs(
                abort_result["onset_heights"],
                ABORT_FROM_ONSET_FULL_BINS,
                color=LED_COLORS[led_trial],
                linewidth=1.6,
            )

        off_area = payload["aborts"][0]["abort_fraction"]
        on_area = payload["aborts"][1]["abort_fraction"]
        ax.text(
            0.97,
            0.94,
            f"OFF area = {off_area:.3f}\nON area = {on_area:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
        )
        ax.axvline(
            0,
            color="black",
            linestyle=":",
            linewidth=1.0,
            alpha=0.75,
        )
        ax.set_xlim(ABORT_FROM_ONSET_DISPLAY_LIMITS_S)
        ax.set_ylim(
            0,
            0.45 if row_index == 0 else 1.15 * onset_abort_y_max,
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.16, linewidth=0.6)
        ax.tick_params(axis="both", labelsize=9)

abort_grid_axes[0, 0].set_ylabel(
    "All eligible trials\nAbort rate (s$^{-1}$)", fontsize=10
)
abort_grid_axes[1, 0].set_ylabel(
    "Matched to session type 9\nAbort rate (s$^{-1}$)", fontsize=10
)
for column_index in range(2):
    abort_grid_axes[1, column_index].set_xlabel(
        "Abort time from LED onset (s)", fontsize=10
    )

abort_grid_fig.legend(
    handles=[
        Line2D([0], [0], color=LED_COLORS[0], lw=1.7, label=LED_LABELS[0]),
        Line2D([0], [0], color=LED_COLORS[1], lw=1.7, label=LED_LABELS[1]),
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, 0.947),
    ncol=2,
    frameon=False,
    fontsize=10,
)
abort_grid_fig.suptitle(
    "LED7 fixation-abort rate aligned to LED onset from t = 0",
    fontsize=15,
    y=0.995,
)
abort_grid_fig.text(
    0.5,
    0.015,
    "Top: all eligible trials. Bottom: 5,500 trials per LED group after "
    "matching intended_fix and LED-onset marginals to pooled session type 9. "
    "For OFF trials, onset is scheduled/pseudo-onset.",
    ha="center",
    fontsize=8.5,
)
abort_grid_fig.tight_layout(rect=(0.04, 0.065, 1.0, 0.91), w_pad=1.5, h_pad=1.6)
abort_grid_fig.savefig(
    OUTPUT_ABORT_FIGURE_PATH,
    dpi=OUTPUT_DPI,
    bbox_inches="tight",
)

if SHOW_PLOT:
    plt.show()
else:
    plt.close(abort_grid_fig)


# %%
############ Print the complete numerical audit ############
source_audit_df = pd.DataFrame(source_audit_rows)
matching_audit_df = pd.DataFrame(matching_audit_rows)
timing_audit_df = pd.DataFrame(timing_audit_rows)
abort_audit_df = pd.DataFrame(abort_audit_rows)

print(f"Source: {INPUT_MAT_PATH} :: {TABLE_NAME}")
print(
    "Filters: training_level == 16; session_type in {7, 8, 9}; "
    "repeat_trial in {0, 2, NaN}; LED_trial in {0, 1}."
)
print(
    "No sensory condition exists for these fixation aborts. Matching fields: "
    f"{MATCHING_COLUMNS}. ABL, ILD, success, response_poke, timed_fix, and "
    "abort_event do not enter selection."
)
print(
    "Effective scheduled onset: intended_fix - LED_onset_time for session "
    "types 7/8; raw LED_onset_time for session type 9."
)
print(
    f"Reference: pooled session-type-9 OFF + ON, n = {len(reference_df):,}. "
    f"Matched sample: {MATCH_SAMPLE_SIZE:,} rows per session-type/LED group."
)
print(f"Fixed base random seed: {RANDOM_SEED}.")

print("\nSource audit")
print(source_audit_df.to_string(index=False))

print("\nMarginal matching and joint-distribution audit")
print(
    matching_audit_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)
print(
    "\nBalance requirements: intended/onset KS <= "
    f"{MAX_KS_DISTANCE:.3f}; Wasserstein <= {MAX_WASSERSTEIN_MS:.1f} ms. "
    "joint_20ms_TV and correlation are reported diagnostics, not matching "
    "constraints."
)

print("\nTiming-distribution audit")
print(
    timing_audit_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)

print("\nFixation-abort audit")
print(
    abort_audit_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)
print(f"\nSaved figure: {OUTPUT_FIGURE_PATH.resolve()}")
print(f"Saved compact figure: {OUTPUT_COMPACT_FIGURE_PATH.resolve()}")
print(f"Saved abort-rate figure: {OUTPUT_ABORT_FIGURE_PATH.resolve()}")
