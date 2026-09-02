# %%
from pathlib import Path
import importlib.util
import os


# %%
############ Parameters (edit here) ############
MAT_TABLE_NAME = "totalout_stGtACRII"
TRAINING_LEVEL = 16
SESSION_TYPE = 9
ALLOWED_REPEAT_TRIALS = [0, 2]
ALLOWED_LED_TRIALS = [0, 1]

EXPECTED_ROWS = 100_788
EXPECTED_COLUMNS = 52
EXPECTED_ANIMALS = [90, 92, 93, 98, 99, 100, 102, 103]
EXPECTED_SESSIONS = 56
EXPECTED_REPEAT_COUNTS = {0.0: 97_440, 2.0: 3_348}
EXPECTED_LED_COUNTS = {0.0: 61_408, 1.0: 39_380}

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
INPUT_PATH = REPO_ROOT / "raw_data" / "outMatrix_LED7_latest.mat"
OUTPUT_PATH = REPO_ROOT / "raw_data" / "LED7_s9.csv"

# mat-io is kept isolated because version 1.0.0 requires a newer NumPy/SciPy
# stack than this repository. Override this path with MATIO_SITE_PACKAGES when
# using another isolated installation.
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
    import sys

    sys.path.insert(0, str(matio_site_packages_path))

from matio import load_from_mat
import numpy as np
import pandas as pd


# %%
############ Load the source MATLAB table ############
if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Could not find input MAT file: {INPUT_PATH}")

source_df = load_from_mat(
    INPUT_PATH,
    variable_names=[MAT_TABLE_NAME],
)[MAT_TABLE_NAME]

if len(source_df.columns) != EXPECTED_COLUMNS:
    raise RuntimeError(
        f"Expected {EXPECTED_COLUMNS} source columns, found {len(source_df.columns)}."
    )


# %%
############ Apply the requested filters in order ############
training_df = source_df.loc[
    source_df["training_level"].eq(TRAINING_LEVEL)
].copy()

session_df = training_df.loc[
    training_df["session_type"].eq(SESSION_TYPE)
].copy()

repeat_mask = (
    session_df["repeat_trial"].isin(ALLOWED_REPEAT_TRIALS)
    | session_df["repeat_trial"].isna()
)
repeat_df = session_df.loc[repeat_mask].copy()

led_mask = (
    repeat_df["LED_trial"].isin(ALLOWED_LED_TRIALS)
    | repeat_df["LED_trial"].isna()
)
output_df = repeat_df.loc[led_mask].copy()


# %%
############ Validate and save all original columns ############
animals = sorted(output_df["animal"].dropna().astype(int).unique().tolist())
repeat_counts = output_df["repeat_trial"].value_counts(dropna=False).sort_index().to_dict()
led_counts = output_df["LED_trial"].value_counts(dropna=False).sort_index().to_dict()

if len(output_df) != EXPECTED_ROWS:
    raise RuntimeError(f"Expected {EXPECTED_ROWS:,} rows, found {len(output_df):,}.")
if output_df.columns.tolist() != source_df.columns.tolist():
    raise RuntimeError("Output columns or their order differ from the source table.")
if animals != EXPECTED_ANIMALS:
    raise RuntimeError(f"Unexpected animals: {animals}")
if output_df["session"].nunique() != EXPECTED_SESSIONS:
    raise RuntimeError(
        f"Expected {EXPECTED_SESSIONS} sessions, found {output_df['session'].nunique()}."
    )
if repeat_counts != EXPECTED_REPEAT_COUNTS:
    raise RuntimeError(f"Unexpected repeat_trial counts: {repeat_counts}")
if led_counts != EXPECTED_LED_COUNTS:
    raise RuntimeError(f"Unexpected LED_trial counts: {led_counts}")
if output_df.duplicated().any():
    raise RuntimeError("Filtered output contains duplicated full rows.")

output_df.to_csv(OUTPUT_PATH, index=False)


# %%
############ Read-back verification ############
saved_df = pd.read_csv(OUTPUT_PATH, float_precision="round_trip")
if saved_df.columns.tolist() != output_df.columns.tolist():
    raise RuntimeError("Saved CSV columns or their order differ from the filtered table.")
if saved_df.shape != output_df.shape:
    raise RuntimeError(
        f"Saved CSV shape {saved_df.shape} differs from expected {output_df.shape}."
    )

np.testing.assert_allclose(
    saved_df.to_numpy(dtype=float),
    output_df.to_numpy(dtype=float),
    rtol=0,
    atol=0,
    equal_nan=True,
)

print(f"Source rows: {len(source_df):,}")
print(f"After training_level == {TRAINING_LEVEL}: {len(training_df):,}")
print(f"After session_type == {SESSION_TYPE}: {len(session_df):,}")
print(f"After repeat_trial in {ALLOWED_REPEAT_TRIALS} or NaN: {len(repeat_df):,}")
print(f"After LED_trial in {ALLOWED_LED_TRIALS} or NaN: {len(output_df):,}")
print(f"Columns preserved: {len(output_df.columns)}")
print(f"Animals: {animals}")
print(f"Sessions: {output_df['session'].nunique()}")
print(f"repeat_trial counts: {repeat_counts}")
print(f"LED_trial counts: {led_counts}")
print("Full-row duplicates: 0")
print(f"Saved and read back exactly: {OUTPUT_PATH}")
