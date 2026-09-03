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
    SCRIPT_DIR / "led7_s9_led_on_abort_onset_vs_intended_fix_by_animal.png"
)

TRAINING_LEVEL = 16
SESSION_TYPE = 9
LED_TRIAL = 1
ABORT_EVENT = 3
ALLOWED_REPEAT_TRIALS = {0, 2}
EXPECTED_ANIMALS = [90, 92, 93, 98, 99, 100, 102, 103]
EXPECTED_ROWS = 100_788
EXPECTED_LED_ON_ABORTS = 8_752

AXIS_MIN_S = 0.0
AXIS_MAX_S = 2.1
POINT_COLOR = "tab:red"
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
############ Select LED ON fixation aborts and audit the ordering ############
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

led_on_aborts["onset_before_intended_s"] = (
    led_on_aborts["intended_fix"] - led_on_aborts["LED_onset_time"]
)
led_on_aborts["ordering_violation"] = led_on_aborts["LED_onset_time"].ge(
    led_on_aborts["intended_fix"]
)

if led_on_aborts["ordering_violation"].any():
    n_violations = int(led_on_aborts["ordering_violation"].sum())
    raise RuntimeError(
        f"Found {n_violations:,} LED ON aborts with LED_onset_time >= intended_fix."
    )

if not (
    led_on_aborts[["LED_onset_time", "intended_fix"]] <= AXIS_MAX_S
).all(axis=None):
    raise RuntimeError(f"A plotted value exceeds AXIS_MAX_S = {AXIS_MAX_S}.")
if not (
    led_on_aborts[["LED_onset_time", "intended_fix"]] >= AXIS_MIN_S
).all(axis=None):
    raise RuntimeError(f"A plotted value is below AXIS_MIN_S = {AXIS_MIN_S}.")

audit_df = (
    led_on_aborts.groupby("animal", sort=True)
    .agg(
        n_aborts=("animal", "size"),
        onset_min_s=("LED_onset_time", "min"),
        onset_max_s=("LED_onset_time", "max"),
        intended_min_s=("intended_fix", "min"),
        intended_max_s=("intended_fix", "max"),
        minimum_gap_s=("onset_before_intended_s", "min"),
        median_gap_s=("onset_before_intended_s", "median"),
        maximum_gap_s=("onset_before_intended_s", "max"),
        n_violations=("ordering_violation", "sum"),
    )
    .reset_index()
)
audit_df["animal"] = audit_df["animal"].astype(int)
audit_df["n_aborts"] = audit_df["n_aborts"].astype(int)
audit_df["n_violations"] = audit_df["n_violations"].astype(int)

if audit_df["animal"].tolist() != EXPECTED_ANIMALS:
    raise RuntimeError("Not every expected animal is represented in the abort subset.")
if int(audit_df["n_aborts"].sum()) != EXPECTED_LED_ON_ABORTS:
    raise RuntimeError("Per-animal abort counts do not sum to the pooled count.")


# %%
############ Plot one panel per animal ############
fig, axes = plt.subplots(
    2,
    4,
    figsize=(14.5, 7.8),
    sharex=True,
    sharey=True,
)

identity_x = np.array([AXIS_MIN_S, AXIS_MAX_S])

for ax, animal in zip(axes.flat, EXPECTED_ANIMALS):
    animal_df = led_on_aborts.loc[led_on_aborts["animal"].eq(animal)]
    animal_audit = audit_df.loc[audit_df["animal"].eq(animal)].iloc[0]

    ax.fill_between(
        identity_x,
        identity_x,
        AXIS_MAX_S,
        color="black",
        alpha=0.035,
        linewidth=0,
    )
    ax.plot(
        identity_x,
        identity_x,
        color="black",
        linestyle="--",
        linewidth=1.1,
    )
    ax.scatter(
        animal_df["intended_fix"],
        animal_df["LED_onset_time"],
        s=8,
        alpha=0.38,
        color=POINT_COLOR,
        edgecolors="none",
        rasterized=True,
    )

    ax.set_title(f"Animal {animal}", fontsize=11)
    ax.text(
        0.04,
        0.96,
        f"n = {int(animal_audit['n_aborts']):,}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
    )
    ax.set_xlim(AXIS_MIN_S, AXIS_MAX_S)
    ax.set_ylim(AXIS_MIN_S, AXIS_MAX_S)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.tick_params(labelsize=8.5)

fig.supxlabel("intended_fix (s)", fontsize=11)
fig.supylabel("LED_onset_time (s)", fontsize=11)
fig.suptitle(
    "Animal wise LED_onset_time vs intended_fix for aborts LED ON "
    "(LED7 sessiontype 9)",
    fontsize=14,
    y=0.985,
)
fig.tight_layout(rect=(0.025, 0.025, 1.0, 0.945), h_pad=2.0, w_pad=1.0)
fig.savefig(OUTPUT_FIGURE_PATH, dpi=OUTPUT_DPI, bbox_inches="tight")

if SHOW_PLOT:
    plt.show()
else:
    plt.close(fig)


# %%
############ Print the numerical audit ############
pooled_minimum_gap_s = float(led_on_aborts["onset_before_intended_s"].min())
pooled_median_gap_s = float(led_on_aborts["onset_before_intended_s"].median())
pooled_maximum_gap_s = float(led_on_aborts["onset_before_intended_s"].max())

print(f"Loaded: {INPUT_CSV_PATH}")
print(
    f"Subset: training_level={TRAINING_LEVEL}, session_type={SESSION_TYPE}, "
    f"LED_trial={LED_TRIAL}, abort_event={ABORT_EVENT}"
)
print(f"LED ON fixation aborts plotted: {len(led_on_aborts):,}")
print("Confirmed: all onset/intended-fix pairs are finite.")
print("Confirmed: LED_onset_time < intended_fix for every plotted abort.")
print(
    "Pooled intended_fix - LED_onset_time gap: "
    f"min={1_000 * pooled_minimum_gap_s:.3f} ms, "
    f"median={1_000 * pooled_median_gap_s:.3f} ms, "
    f"max={1_000 * pooled_maximum_gap_s:.3f} ms"
)
print("\nPer-animal audit")
print(audit_df.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
print(f"\nSaved figure: {OUTPUT_FIGURE_PATH}")
