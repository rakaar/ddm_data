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
OUTPUT_FIGURE_PATH = (
    SCRIPT_DIR / "led7_s9_led_on_abort_intended_minus_onset_distribution.png"
)

TRAINING_LEVEL = 16
SESSION_TYPE = 9
LED_TRIAL = 1
ABORT_EVENT = 3
ALLOWED_REPEAT_TRIALS = {0, 2}
EXPECTED_ANIMALS = [90, 92, 93, 98, 99, 100, 102, 103]
EXPECTED_ROWS = 100_788
EXPECTED_LED_ON_ABORTS = 8_752

BIN_WIDTH_S = 0.020
HISTOGRAM_MIN_S = 0.1
HISTOGRAM_MAX_S = 1.1
HISTOGRAM_BINS = np.arange(
    HISTOGRAM_MIN_S,
    HISTOGRAM_MAX_S + BIN_WIDTH_S / 2,
    BIN_WIDTH_S,
)

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
    "LED_onset_time",
    "intended_fix",
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
if df["animal"].isna().any():
    raise RuntimeError("animal contains missing values.")

df["animal"] = df["animal"].astype(int)
animals = sorted(df["animal"].unique().tolist())
if animals != EXPECTED_ANIMALS:
    raise RuntimeError(f"Expected animals {EXPECTED_ANIMALS}, found {animals}.")


# %%
############ Pool LED ON fixation aborts and calculate the difference ############
led_on_aborts = df.loc[
    df["LED_trial"].eq(LED_TRIAL) & df["abort_event"].eq(ABORT_EVENT)
].copy()

if len(led_on_aborts) != EXPECTED_LED_ON_ABORTS:
    raise RuntimeError(
        f"Expected {EXPECTED_LED_ON_ABORTS:,} LED ON event-3 aborts, "
        f"found {len(led_on_aborts):,}."
    )
if not np.isfinite(
    led_on_aborts[["LED_onset_time", "intended_fix"]]
).all(axis=None):
    raise RuntimeError("LED ON aborts contain non-finite onset or intended-fix values.")

intended_minus_onset = (
    led_on_aborts["intended_fix"] - led_on_aborts["LED_onset_time"]
).to_numpy(dtype=float)

if not (intended_minus_onset > 0).all():
    raise RuntimeError("Some LED ON aborts have LED_onset_time >= intended_fix.")
if not (
    (intended_minus_onset >= HISTOGRAM_BINS[0])
    & (intended_minus_onset <= HISTOGRAM_BINS[-1])
).all():
    raise RuntimeError(
        "Some intended_fix - LED_onset_time values fall outside the histogram bins."
    )

counts, _ = np.histogram(intended_minus_onset, bins=HISTOGRAM_BINS)
if int(counts.sum()) != EXPECTED_LED_ON_ABORTS:
    raise RuntimeError(
        f"Histogram contains {counts.sum():,} of {EXPECTED_LED_ON_ABORTS:,} values."
    )

animal_counts = (
    led_on_aborts["animal"]
    .value_counts()
    .reindex(EXPECTED_ANIMALS)
    .astype(int)
    .rename_axis("animal")
    .rename("n_aborts")
)
if int(animal_counts.sum()) != EXPECTED_LED_ON_ABORTS:
    raise RuntimeError("Per-animal counts do not sum to the pooled abort count.")


# %%
############ Plot the pooled raw-count distribution ############
fig, ax = plt.subplots(figsize=(8.2, 5.2))

ax.stairs(
    counts,
    HISTOGRAM_BINS,
    color="tab:red",
    linewidth=1.8,
)
ax.text(
    0.97,
    0.94,
    f"n = {len(intended_minus_onset):,}",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=10,
)
ax.set_xlim(HISTOGRAM_MIN_S, HISTOGRAM_MAX_S)
ax.set_xlabel("intended_fix - LED_onset_time (s)")
ax.set_ylabel("Count")
ax.set_title(
    "All animals: intended_fix - LED_onset_time for aborts LED ON "
    "(LED7 sessiontype 9)"
)
ax.grid(axis="y", alpha=0.18, linewidth=0.6)
fig.tight_layout()
fig.savefig(OUTPUT_FIGURE_PATH, dpi=OUTPUT_DPI, bbox_inches="tight")

if SHOW_PLOT:
    plt.show()
else:
    plt.close(fig)


# %%
############ Print the numerical audit ############
print(f"Loaded: {INPUT_CSV_PATH}")
print(
    f"Subset: training_level={TRAINING_LEVEL}, session_type={SESSION_TYPE}, "
    f"LED_trial={LED_TRIAL}, abort_event={ABORT_EVENT}"
)
print(f"Animals pooled: {EXPECTED_ANIMALS}")
print(f"LED ON fixation aborts plotted: {len(intended_minus_onset):,}")
print(f"Bin width: {1_000 * BIN_WIDTH_S:.0f} ms")
print(
    "intended_fix - LED_onset_time: "
    f"min={intended_minus_onset.min():.6f} s, "
    f"median={np.median(intended_minus_onset):.6f} s, "
    f"mean={intended_minus_onset.mean():.6f} s, "
    f"max={intended_minus_onset.max():.6f} s"
)
print("\nPer-animal contribution")
print(animal_counts.to_string())
print(f"\nSaved figure: {OUTPUT_FIGURE_PATH}")
