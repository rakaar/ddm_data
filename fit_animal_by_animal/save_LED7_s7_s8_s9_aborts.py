# %%
"""Export LED7 fixation aborts and their all-trial timing denominators."""

from pathlib import Path
import importlib.util
import os
import sys


# %%
############ Parameters (edit here) ############
MAT_TABLE_NAME = "totalout_stGtACRII"
TRAINING_LEVEL = 16
SESSION_TYPES = (7, 8, 9)
ALLOWED_REPEAT_TRIALS = (0, 2)
ALLOWED_LED_TRIALS = (0, 1)
ABORT_EVENT = 3

EXPECTED_SOURCE_COLUMNS = 52
EXPECTED_ANIMALS = (90, 92, 93, 98, 99, 100, 102, 103)
EXPECTED_ALL_ROWS = {7: 138_440, 8: 148_428}
EXPECTED_ABORT_ROWS = {7: 22_494, 8: 22_926, 9: 18_860}
EXPECTED_MISSING_TIMED_FIX = {7: 3, 8: 2, 9: 6}

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
INPUT_PATH = REPO_ROOT / "raw_data" / "outMatrix_LED7_latest.mat"
OUTPUT_PATHS = {
    session_type: REPO_ROOT / "raw_data" / f"LED7_s{session_type}_aborts.csv"
    for session_type in SESSION_TYPES
}
ALL_TRIAL_OUTPUT_PATHS = {
    session_type: REPO_ROOT / "raw_data" / f"LED7_s{session_type}.csv"
    for session_type in (7, 8)
}

# Supply an isolated mat-io installation with MATIO_SITE_PACKAGES if needed.
MATIO_SITE_PACKAGES = os.environ.get("MATIO_SITE_PACKAGES")
if importlib.util.find_spec("matio") is None:
    if MATIO_SITE_PACKAGES is None:
        raise ModuleNotFoundError(
            "MATLAB MCOS table reader not found. Set MATIO_SITE_PACKAGES to an "
            "isolated site-packages directory containing mat-io>=1.0.0."
        )
    matio_site_packages_path = Path(MATIO_SITE_PACKAGES).expanduser()
    if not matio_site_packages_path.exists():
        raise FileNotFoundError(
            f"MATIO_SITE_PACKAGES does not exist: {matio_site_packages_path}"
        )
    sys.path.insert(0, str(matio_site_packages_path))

from matio import load_from_mat
import numpy as np
import pandas as pd


# %%
############ Load the LED7 source table ############
if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Could not find input MAT file: {INPUT_PATH}")

source_df = load_from_mat(
    INPUT_PATH, variable_names=[MAT_TABLE_NAME]
)[MAT_TABLE_NAME]
if not isinstance(source_df, pd.DataFrame):
    raise TypeError(f"Expected {MAT_TABLE_NAME} to decode as a DataFrame.")
if len(source_df.columns) != EXPECTED_SOURCE_COLUMNS:
    raise RuntimeError(
        f"Expected {EXPECTED_SOURCE_COLUMNS} source columns, "
        f"found {len(source_df.columns)}."
    )
if source_df.columns.tolist().count("LED_onset_time") != 1:
    raise RuntimeError("Expected exactly one source LED_onset_time column.")

required_columns = [
    "animal", "session", "session_type", "training_level", "repeat_trial",
    "LED_trial", "abort_event", "timed_fix", "intended_fix", "LED_onset_time",
]
missing_columns = [column for column in required_columns if column not in source_df]
if missing_columns:
    raise RuntimeError(f"Source table is missing columns: {missing_columns}")


# %%
############ Apply the requested filters in order ############
training_df = source_df.loc[
    source_df["training_level"].eq(TRAINING_LEVEL)
]
session_df = training_df.loc[
    training_df["session_type"].isin(SESSION_TYPES)
]
repeat_df = session_df.loc[
    session_df["repeat_trial"].isin(ALLOWED_REPEAT_TRIALS)
    | session_df["repeat_trial"].isna()
]
led_df = repeat_df.loc[
    repeat_df["LED_trial"].isin(ALLOWED_LED_TRIALS)
    | repeat_df["LED_trial"].isna()
]
abort_df = led_df.loc[
    led_df["abort_event"].eq(ABORT_EVENT)
]

print(f"Source rows: {len(source_df):,}")
print(f"After training_level == {TRAINING_LEVEL}: {len(training_df):,}")
print(f"After session_type in {SESSION_TYPES}: {len(session_df):,}")
print(f"After repeat_trial in {ALLOWED_REPEAT_TRIALS} or NaN: {len(repeat_df):,}")
print(f"After LED_trial in {ALLOWED_LED_TRIALS} or NaN: {len(led_df):,}")
print(f"After abort_event == {ABORT_EVENT}: {len(abort_df):,}")


# %%
############ Export all eligible s7/s8 trials for timing and rate denominators ############
for session_type, output_path in ALL_TRIAL_OUTPUT_PATHS.items():
    source_subset = led_df.loc[led_df["session_type"].eq(session_type)].copy()
    output_df = source_subset.copy()
    output_df["LED_onset_time"] = (
        source_subset["intended_fix"] - source_subset["LED_onset_time"]
    )

    if len(output_df) != EXPECTED_ALL_ROWS[session_type]:
        raise RuntimeError(
            f"Session type {session_type}: expected "
            f"{EXPECTED_ALL_ROWS[session_type]:,} eligible trials, "
            f"found {len(output_df):,}."
        )
    if output_df.columns.tolist() != source_df.columns.tolist():
        raise RuntimeError(f"Session type {session_type}: all-trial schema changed.")
    if not output_df.drop(columns="LED_onset_time").equals(
        source_subset.drop(columns="LED_onset_time")
    ):
        raise RuntimeError(f"Session type {session_type}: other columns changed.")
    animals = tuple(sorted(output_df["animal"].dropna().astype(int).unique()))
    if animals != EXPECTED_ANIMALS:
        raise RuntimeError(f"Session type {session_type}: unexpected animals {animals}.")
    if not output_df["training_level"].eq(TRAINING_LEVEL).all():
        raise RuntimeError(f"Session type {session_type}: unexpected training level.")
    if not output_df["session_type"].eq(session_type).all():
        raise RuntimeError(f"Session type {session_type}: unexpected session type.")
    if not (
        output_df["repeat_trial"].isin(ALLOWED_REPEAT_TRIALS)
        | output_df["repeat_trial"].isna()
    ).all():
        raise RuntimeError(f"Session type {session_type}: unexpected repeat trial.")
    if not (
        output_df["LED_trial"].isin(ALLOWED_LED_TRIALS)
        | output_df["LED_trial"].isna()
    ).all():
        raise RuntimeError(f"Session type {session_type}: unexpected LED trial.")
    if output_df["repeat_trial"].isna().any() or output_df["LED_trial"].isna().any():
        raise RuntimeError(f"Session type {session_type}: unexpected NaN category.")
    if not np.isfinite(output_df[["intended_fix", "LED_onset_time"]]).all(axis=None):
        raise RuntimeError(f"Session type {session_type}: non-finite timing.")
    if (
        (output_df["LED_onset_time"] < -1e-12)
        | (output_df["LED_onset_time"] > output_df["intended_fix"] + 1e-12)
    ).any():
        raise RuntimeError(f"Session type {session_type}: onset outside fixation.")
    if output_df.duplicated().any():
        raise RuntimeError(f"Session type {session_type}: duplicate full rows.")

    if not output_path.exists():
        output_df.to_csv(output_path, index=False)
    saved_df = pd.read_csv(output_path, float_precision="round_trip")
    if saved_df.shape != output_df.shape:
        raise RuntimeError(f"Session type {session_type}: saved all-trial shape changed.")
    if saved_df.columns.tolist() != source_df.columns.tolist():
        raise RuntimeError(f"Session type {session_type}: saved all-trial schema changed.")
    np.testing.assert_allclose(
        saved_df.to_numpy(dtype=float),
        output_df.to_numpy(dtype=float),
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    print(
        f"Session type {session_type}: {len(output_df):,} all-trial rows; "
        f"saved/read back {output_path}"
    )


# %%
############ Export and read back each session type ############
for session_type in SESSION_TYPES:
    source_subset = abort_df.loc[
        abort_df["session_type"].eq(session_type)
    ].copy()
    output_df = source_subset.copy()

    # For LED OFF, this is a scheduled/counterfactual onset, not illumination.
    # Session type 9 already stores fixation-referenced onset in this column.
    expected_onset = (
        source_subset["LED_onset_time"]
        if session_type == 9
        else source_subset["intended_fix"] - source_subset["LED_onset_time"]
    )
    output_df["LED_onset_time"] = expected_onset

    if len(output_df) != EXPECTED_ABORT_ROWS[session_type]:
        raise RuntimeError(
            f"Session type {session_type}: expected "
            f"{EXPECTED_ABORT_ROWS[session_type]:,} aborts, found {len(output_df):,}."
        )
    if output_df.columns.tolist() != source_df.columns.tolist():
        raise RuntimeError(f"Session type {session_type}: source schema changed.")
    if output_df.columns.tolist().count("LED_onset_time") != 1:
        raise RuntimeError(f"Session type {session_type}: duplicate onset column.")
    if not output_df.drop(columns="LED_onset_time").equals(
        source_subset.drop(columns="LED_onset_time")
    ):
        raise RuntimeError(f"Session type {session_type}: other columns changed.")

    animals = tuple(sorted(output_df["animal"].dropna().astype(int).unique()))
    if animals != EXPECTED_ANIMALS:
        raise RuntimeError(f"Session type {session_type}: unexpected animals {animals}.")
    if not output_df["training_level"].eq(TRAINING_LEVEL).all():
        raise RuntimeError(f"Session type {session_type}: unexpected training level.")
    if not output_df["session_type"].eq(session_type).all():
        raise RuntimeError(f"Session type {session_type}: unexpected session type.")
    if not (
        output_df["repeat_trial"].isin(ALLOWED_REPEAT_TRIALS)
        | output_df["repeat_trial"].isna()
    ).all():
        raise RuntimeError(f"Session type {session_type}: unexpected repeat trial.")
    if not (
        output_df["LED_trial"].isin(ALLOWED_LED_TRIALS)
        | output_df["LED_trial"].isna()
    ).all():
        raise RuntimeError(f"Session type {session_type}: unexpected LED trial.")
    if not output_df["abort_event"].eq(ABORT_EVENT).all():
        raise RuntimeError(f"Session type {session_type}: non-fixation abort.")
    missing_timed_fix = int(output_df["timed_fix"].isna().sum())
    if missing_timed_fix != EXPECTED_MISSING_TIMED_FIX[session_type]:
        raise RuntimeError(
            f"Session type {session_type}: expected "
            f"{EXPECTED_MISSING_TIMED_FIX[session_type]} missing timed_fix, "
            f"found {missing_timed_fix}."
        )
    if output_df["repeat_trial"].isna().any() or output_df["LED_trial"].isna().any():
        raise RuntimeError(f"Session type {session_type}: unexpected NaN category.")
    if not np.isfinite(output_df["LED_onset_time"]).all():
        raise RuntimeError(f"Session type {session_type}: non-finite onset.")
    if (
        (output_df["LED_onset_time"] < -1e-12)
        | (output_df["LED_onset_time"] > output_df["intended_fix"] + 1e-12)
    ).any():
        raise RuntimeError(f"Session type {session_type}: onset outside fixation.")
    if output_df.duplicated().any():
        raise RuntimeError(f"Session type {session_type}: duplicate full rows.")

    output_path = OUTPUT_PATHS[session_type]
    if not output_path.exists():
        output_df.to_csv(output_path, index=False)
    saved_df = pd.read_csv(output_path, float_precision="round_trip")
    if saved_df.shape != output_df.shape:
        raise RuntimeError(f"Session type {session_type}: saved CSV shape changed.")
    if saved_df.columns.tolist() != source_df.columns.tolist():
        raise RuntimeError(f"Session type {session_type}: saved CSV schema changed.")
    np.testing.assert_allclose(
        saved_df.to_numpy(dtype=float),
        output_df.to_numpy(dtype=float),
        rtol=0,
        atol=0,
        equal_nan=True,
    )

    print(
        f"Session type {session_type}: {len(output_df):,} aborts, "
        f"{missing_timed_fix} missing timed_fix; "
        f"LED_trial counts {output_df['LED_trial'].value_counts(dropna=False).to_dict()}; "
        f"saved/read back {output_path}"
    )
