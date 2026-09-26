# %%
"""Diagnose the variable LED_duration and onset-coordinate change in LED7 s9."""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# %%
############ Parameters (edit here) ############
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV = SCRIPT_DIR.parent / "raw_data" / "LED7_s9.csv"
OUTPUT_PNG = SCRIPT_DIR / "led7_s9_led_duration_timing_relations_2x2.png"
AUDIT_CSV = SCRIPT_DIR / "led7_s9_led_duration_session_formula_audit.csv"
TRAINING_LEVEL = 16
SESSION_TYPE = 9
ALLOWED_REPEAT_TRIALS = (0, 2)
ALLOWED_LED_TRIALS = (0, 1)
EXPECTED_ROWS = 100_788
EXPECTED_ANIMALS = (90, 92, 93, 98, 99, 100, 102, 103)
BIN_WIDTH_S = 0.020
DURATION_BINS = np.arange(1.1, 2.1 + BIN_WIDTH_S / 2, BIN_WIDTH_S)
FORMULA_TOLERANCE_S = 1e-9
MEASUREMENT_TOLERANCE_S = 0.001
SCATTER_PER_MODE = 3_000
RANDOM_SEED = 20260925


# %%
############ Read standardized CSV and audit the exact relationships ############
df = pd.read_csv(INPUT_CSV, float_precision="round_trip")
required = {
    "animal", "session", "training_level", "session_type", "repeat_trial",
    "LED_trial", "abort_event", "intended_fix", "LED_onset_time",
    "LED_duration", "timed_LED", "timed_fix",
}
if missing := required.difference(df.columns):
    raise RuntimeError(f"Missing columns: {sorted(missing)}")
if len(df.columns) != 52 or len(df) != EXPECTED_ROWS:
    raise RuntimeError("Unexpected session-type-9 CSV schema or row count.")
if not df["training_level"].eq(TRAINING_LEVEL).all():
    raise RuntimeError("Unexpected training level in CSV.")
if not df["session_type"].eq(SESSION_TYPE).all():
    raise RuntimeError("Unexpected session type in CSV.")
if not (
    df["repeat_trial"].isin(ALLOWED_REPEAT_TRIALS) | df["repeat_trial"].isna()
).all():
    raise RuntimeError("Unexpected repeat_trial in CSV.")
if not (
    df["LED_trial"].isin(ALLOWED_LED_TRIALS) | df["LED_trial"].isna()
).all():
    raise RuntimeError("Unexpected LED_trial in CSV.")
if tuple(sorted(df["animal"].astype(int).unique())) != EXPECTED_ANIMALS:
    raise RuntimeError("Unexpected animal cohort.")
if not np.isfinite(df[["intended_fix", "LED_onset_time", "LED_duration"]]).all(axis=None):
    raise RuntimeError("A required duration or schedule field is non-finite.")

raw_onset = df["LED_onset_time"].to_numpy(dtype=float)
intended_fix = df["intended_fix"].to_numpy(dtype=float)
duration = df["LED_duration"].to_numpy(dtype=float)
stimulus_gap_from_raw = intended_fix - raw_onset
error_raw = duration - (1.0 + raw_onset)
error_gap = duration - (1.0 + stimulus_gap_from_raw)
raw_formula = np.abs(error_raw) <= FORMULA_TOLERANCE_S
gap_formula = np.abs(error_gap) <= FORMULA_TOLERANCE_S
if not (raw_formula ^ gap_formula).all():
    raise RuntimeError("Every row should satisfy exactly one duration formula.")

df["duration_formula"] = np.where(
    raw_formula,
    "1 + raw LED_onset_time",
    "1 + intended_fix - raw LED_onset_time",
)
formula_colors = {
    "1 + raw LED_onset_time": "tab:blue",
    "1 + intended_fix - raw LED_onset_time": "tab:orange",
}

session_modes = df.groupby(["animal", "session"])["duration_formula"].nunique()
if not session_modes.eq(1).all():
    raise RuntimeError("A session mixes LED_duration formulas.")
session_audit = (
    df.groupby(["animal", "session", "duration_formula"], as_index=False)
    .agg(
        all_trials=("LED_duration", "size"),
        event3_aborts=("abort_event", lambda x: int(x.eq(3).sum())),
    )
    .sort_values(["animal", "session"])
)
session_audit["session_index_within_type9"] = (
    session_audit.groupby("animal").cumcount() + 1
)
for animal, animal_sessions in session_audit.groupby("animal"):
    is_early_formula = animal_sessions["duration_formula"].eq(
        "1 + raw LED_onset_time"
    ).to_numpy()
    if not np.array_equal(
        is_early_formula,
        np.arange(len(is_early_formula)) < is_early_formula.sum(),
    ):
        raise RuntimeError(f"Animal {animal}: formula does not switch once.")
session_audit.to_csv(AUDIT_CSV, index=False)

timed_led = df["timed_LED"].to_numpy(dtype=float)
measured_led = np.isfinite(timed_led)
measurement_error = np.abs(duration[measured_led] - timed_led[measured_led])
if not (measurement_error <= MEASUREMENT_TOLERANCE_S).all():
    raise RuntimeError("Measured timed_LED differs from LED_duration by over 1 ms.")
is_abort = df["abort_event"].eq(3).to_numpy()
timed_fix = df["timed_fix"].to_numpy(dtype=float)
finite_abort_fix = is_abort & np.isfinite(timed_fix)
abort_fix_error = np.abs(duration[finite_abort_fix] - timed_fix[finite_abort_fix])

counts, _ = np.histogram(duration, bins=DURATION_BINS)
if int(counts.sum()) != len(df):
    raise RuntimeError("Duration histogram omitted rows.")
print(f"Source: {INPUT_CSV} (original MAT columns; no s9 duration transform)")
print(f"All filtered s9 rows: {len(df):,}; duration range: {duration.min():.6f}–{duration.max():.6f} s")
print(f"Median duration: {np.median(duration):.6f} s")
print(f"Duration = 1 + raw onset: {int(raw_formula.sum()):,} rows")
print(f"Duration = 1 + intended_fix − raw onset: {int(gap_formula.sum()):,} rows")
print(
    "Sessions by formula: "
    f"{int(raw_formula.sum()):,} rows in "
    f"{int((session_audit['duration_formula'] == '1 + raw LED_onset_time').sum())} "
    "early sessions; "
    f"{int(gap_formula.sum()):,} rows in "
    f"{int((session_audit['duration_formula'] == '1 + intended_fix - raw LED_onset_time').sum())} "
    "later sessions"
)
print(
    f"timed_LED present in {int(measured_led.sum()):,} rows; "
    f"max |LED_duration − timed_LED| = {measurement_error.max():.6f} s"
)
print(
    f"Event-3 aborts with finite timed_fix: {int(finite_abort_fix.sum()):,}; "
    f"within 1 ms of LED_duration: "
    f"{int((abort_fix_error <= MEASUREMENT_TOLERANCE_S).sum())}"
)
print(f"Session audit: {AUDIT_CSV}")


# %%
############ Plot duration distribution, exact formulas, and session switch ############
fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))

ax = axes[0, 0]
all_density = counts / (len(duration) * BIN_WIDTH_S)
if not np.isclose(np.sum(all_density * np.diff(DURATION_BINS)), 1.0):
    raise RuntimeError("Full duration density does not integrate to one.")
ax.stairs(all_density, DURATION_BINS, color="black", linewidth=2.0, label="All s9")
for formula, color in formula_colors.items():
    values = duration[df["duration_formula"].eq(formula)]
    formula_counts, _ = np.histogram(values, bins=DURATION_BINS)
    if int(formula_counts.sum()) != len(values):
        raise RuntimeError(f"Duration histogram omitted {formula} rows.")
    density = formula_counts / (len(values) * BIN_WIDTH_S)
    if not np.isclose(np.sum(density * np.diff(DURATION_BINS)), 1.0):
        raise RuntimeError(f"Duration density is not unit-area: {formula}.")
    label = "Early sessions" if formula.startswith("1 + raw") else "Later sessions"
    ax.stairs(density, DURATION_BINS, color=color, linewidth=1.3, label=label)
ax.set(
    title="LED_duration distribution",
    xlabel="LED_duration (s)",
    ylabel="Density (s$^{-1}$)",
    xlim=(1.1, 2.1),
)
ax.legend(frameon=False, fontsize=9)

rng = np.random.default_rng(RANDOM_SEED)
for ax, x_values, x_label in (
    (axes[0, 1], raw_onset, "Raw LED_onset_time (s)"),
    (axes[1, 0], stimulus_gap_from_raw, "intended_fix − raw LED_onset_time (s)"),
):
    for formula, color in formula_colors.items():
        row_indices = np.flatnonzero(df["duration_formula"].eq(formula).to_numpy())
        sample = rng.choice(
            row_indices, size=min(SCATTER_PER_MODE, len(row_indices)), replace=False
        )
        label = "Early sessions" if formula.startswith("1 + raw") else "Later sessions"
        ax.scatter(
            x_values[sample], duration[sample] - 1.0,
            color=color, s=5, alpha=0.25, linewidths=0, label=label,
        )
    ax.plot([0, 1.1], [0, 1.1], color="black", linestyle="--", linewidth=1)
    ax.set(
        xlabel=x_label,
        ylabel="LED_duration − 1 (s)",
        xlim=(0, 1.1),
        ylim=(0, 1.1),
    )
axes[0, 1].set_title("Matches raw onset in early s9 sessions")
axes[1, 0].set_title("Matches stimulus gap in later s9 sessions")

ax = axes[1, 1]
for formula, color in formula_colors.items():
    group = session_audit.loc[session_audit["duration_formula"].eq(formula)]
    ax.scatter(
        group["session_index_within_type9"], group["animal"],
        color=color, marker="s", s=55, label=(
            "Early: 1 + raw onset" if formula.startswith("1 + raw")
            else "Later: 1 + intended − raw onset"
        ),
    )
ax.set(
    title="Formula switches once within every animal",
    xlabel="Session index within session type 9",
    ylabel="Animal",
    yticks=EXPECTED_ANIMALS,
    xlim=(0, session_audit["session_index_within_type9"].max() + 1),
)

for ax in axes.flat:
    ax.grid(alpha=0.15)
fig.suptitle("LED7 session type 9 · training level 16 · LED_duration audit", fontsize=15)
fig.text(
    0.5, 0.015,
    "Both formulas are exact in their respective sessions. The arithmetic changes "
    "within s9; whether onset coordinates or duration programming changed needs protocol code.",
    ha="center", fontsize=9,
)
fig.tight_layout(rect=(0, 0.04, 1, 0.96))
fig.savefig(OUTPUT_PNG, dpi=230, bbox_inches="tight")
plt.close(fig)
print(f"Figure: {OUTPUT_PNG}")
