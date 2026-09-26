# %%
"""Plot LED7 fixation-abort fractions by actual session number and animal."""

from pathlib import Path
import importlib.util
import os
import sys


# %%
############ Parameters (edit here) ############
SCRIPT_DIR = Path(__file__).resolve().parent
MAT_PATH = SCRIPT_DIR.parent / "raw_data" / "outMatrix_LED7_latest.mat"
MAT_TABLE_NAME = "totalout_stGtACRII"
FIGURE_PATH = SCRIPT_DIR / "led7_s7_s8_s9_abort_fraction_by_session_8x3.png"
AUDIT_PATH = SCRIPT_DIR / "led7_s7_s8_s9_abort_fraction_by_session_audit.csv"
ZERO_AUDIT_PATH = SCRIPT_DIR / "led7_s7_s8_s9_zero_abort_sessions_audit.csv"

TRAINING_LEVEL = 16
SESSION_TYPES = (7, 8, 9)
ANIMALS = (90, 92, 93, 98, 99, 100, 102, 103)
ABORT_EVENT = 3
LOW_TRIAL_THRESHOLD = 100
Y_LIM = (0.0, 0.4)
FIGURE_DPI = 220

EXPECTED_COLUMNS = 52
EXPECTED_ROWS = 391_183
EXPECTED_SESSION_GROUPS = 609
EXPECTED_ROWS_BY_TYPE = {7: 138_806, 8: 151_083, 9: 101_294}
EXPECTED_GROUPS_BY_TYPE = {7: 199, 8: 219, 9: 191}

# The isolated MAT-table reader may be supplied outside the project venv.
MATIO_SITE_PACKAGES = os.environ.get("MATIO_SITE_PACKAGES")
if importlib.util.find_spec("matio") is None:
    if MATIO_SITE_PACKAGES is None:
        raise ModuleNotFoundError(
            "MATLAB table reader not found. Set MATIO_SITE_PACKAGES to an "
            "isolated site-packages directory containing mat-io."
        )
    matio_path = Path(MATIO_SITE_PACKAGES).expanduser()
    if not matio_path.exists():
        raise FileNotFoundError(f"MATIO_SITE_PACKAGES does not exist: {matio_path}")
    sys.path.insert(0, str(matio_path))

from matio import load_from_mat
import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator


# %%
############ Load MAT table and apply only the requested training filter ############
if not MAT_PATH.exists():
    raise FileNotFoundError(f"Missing source MAT file: {MAT_PATH}")
source = load_from_mat(MAT_PATH, variable_names=[MAT_TABLE_NAME])[MAT_TABLE_NAME]
if not isinstance(source, pd.DataFrame):
    raise TypeError(f"Expected {MAT_TABLE_NAME} to decode as a DataFrame.")
if len(source.columns) != EXPECTED_COLUMNS:
    raise RuntimeError(f"Expected {EXPECTED_COLUMNS} MAT columns, got {len(source.columns)}.")
required = {"animal", "session", "trial", "session_type", "training_level", "abort_event"}
if missing := required.difference(source.columns):
    raise RuntimeError(f"Missing MAT columns: {sorted(missing)}")

level_16 = source.loc[source["training_level"].eq(TRAINING_LEVEL)]
data = level_16.loc[level_16["session_type"].isin(SESSION_TYPES)]
if len(data) != EXPECTED_ROWS:
    raise RuntimeError(f"Expected {EXPECTED_ROWS:,} filtered rows, got {len(data):,}.")
if data[["animal", "session_type", "session", "trial"]].isna().any(axis=None):
    raise RuntimeError("Missing animal, session type, session, or trial ID.")
if data.duplicated(["animal", "session", "trial"]).any():
    raise RuntimeError("Duplicate (animal, session, trial) source keys.")
if tuple(sorted(data["animal"].astype(int).unique())) != ANIMALS:
    raise RuntimeError("Unexpected animal cohort.")
if not np.equal(data["session"], np.floor(data["session"])).all():
    raise RuntimeError("Non-integer session IDs cannot be plotted on the requested axis.")

# No repeat-trial, LED-trial, success, or timing filter is applied. A coded
# event-3 row contributes to the numerator even when timed_fix is missing.
print(f"Source: {MAT_PATH}::{MAT_TABLE_NAME}")
print(f"training_level == {TRAINING_LEVEL}: {len(level_16):,} rows")
print(f"session types {SESSION_TYPES}: {len(data):,} rows")
print(f"Event-{ABORT_EVENT} rows: {int(data['abort_event'].eq(ABORT_EVENT).sum()):,}")


# %%
############ Count trials and coded fixation aborts per actual session ############
audit = (
    data.groupby(["animal", "session_type", "session"], sort=True)
    .agg(
        all_trials=("trial", "size"),
        event3_aborts=("abort_event", lambda values: int(values.eq(ABORT_EVENT).sum())),
    )
    .reset_index()
)
for column in ("animal", "session_type", "session"):
    audit[column] = audit[column].astype(int)
audit["abort_fraction"] = audit["event3_aborts"] / audit["all_trials"]
audit["below_trial_threshold"] = audit["all_trials"].lt(LOW_TRIAL_THRESHOLD)

if len(audit) != EXPECTED_SESSION_GROUPS:
    raise RuntimeError(
        f"Expected {EXPECTED_SESSION_GROUPS} animal/session groups, got {len(audit)}."
    )
if audit.duplicated(["animal", "session_type", "session"]).any():
    raise RuntimeError("Duplicate animal/session-type/session audit groups.")
if int(audit["all_trials"].sum()) != len(data):
    raise RuntimeError("Session denominators do not sum to filtered MAT rows.")
if int(audit["event3_aborts"].sum()) != int(data["abort_event"].eq(ABORT_EVENT).sum()):
    raise RuntimeError("Session abort numerators do not sum to coded MAT aborts.")
if not audit["abort_fraction"].between(0, 1).all():
    raise RuntimeError("An abort fraction lies outside [0, 1].")
if audit["abort_fraction"].max() >= Y_LIM[1]:
    raise RuntimeError("Configured common y-limit would clip an abort fraction.")
for session_type in SESSION_TYPES:
    source_rows = int(data["session_type"].eq(session_type).sum())
    subset = audit.loc[audit["session_type"].eq(session_type)]
    if source_rows != EXPECTED_ROWS_BY_TYPE[session_type]:
        raise RuntimeError(f"Unexpected source row count for session type {session_type}.")
    if len(subset) != EXPECTED_GROUPS_BY_TYPE[session_type]:
        raise RuntimeError(f"Unexpected session-group count for type {session_type}.")
    if int(subset["all_trials"].sum()) != source_rows:
        raise RuntimeError(f"Grouped denominators differ for type {session_type}.")

audit.to_csv(AUDIT_PATH, index=False)
print("\nPer-type totals (all repeat and LED categories retained):")
print(
    audit.groupby("session_type")
    .agg(
        sessions=("session", "size"),
        trials=("all_trials", "sum"),
        event3_aborts=("event3_aborts", "sum"),
        low_trial_sessions=("below_trial_threshold", "sum"),
    )
    .to_string()
)
print(f"Audit CSV: {AUDIT_PATH}")


# %%
############ Inspect sessions with no coded fixation aborts ############
zero_keys = audit.loc[
    audit["event3_aborts"].eq(0), ["animal", "session_type", "session"]
]
zero_rows = data.merge(zero_keys, on=["animal", "session_type", "session"])
zero_audit = (
    zero_rows.groupby(["animal", "session_type", "session"], sort=True)
    .agg(
        all_trials=("trial", "size"),
        event2_trials=("abort_event", lambda x: int(x.eq(2).sum())),
        no_center_poke_time=("time_to_CNP", lambda x: int(x.isna().sum())),
        no_fixation_time=("timed_fix", lambda x: int(x.isna().sum())),
        no_LED_time=("timed_LED", lambda x: int(x.isna().sum())),
        led_on_trials=("LED_trial", lambda x: int(x.eq(1).sum())),
        blocks=("block", "nunique"),
        max_wait_seconds=("max_wait", "median"),
        median_trial_duration_seconds=("trial_duration", "median"),
        trials_about_31_seconds=(
            "trial_duration", lambda x: int(x.between(30.99, 31.01).sum())
        ),
    )
    .reset_index()
)
for column in ("animal", "session_type", "session"):
    zero_audit[column] = zero_audit[column].astype(int)
if not zero_audit["all_trials"].eq(zero_audit["event2_trials"]).all():
    raise RuntimeError("A zero-fixation-abort session has outcomes besides event 2.")
zero_audit.to_csv(ZERO_AUDIT_PATH, index=False)
print("\nZero-fixation-abort session audit:")
print(zero_audit.to_string(index=False))
print(f"Zero-session audit CSV: {ZERO_AUDIT_PATH}")


# %%
############ Plot eight animals by three session types ############
fig, axes = plt.subplots(len(ANIMALS), len(SESSION_TYPES), figsize=(16.5, 23))
line_color = "#245a80"
missing_session_groups = []

for row, animal in enumerate(ANIMALS):
    for column, session_type in enumerate(SESSION_TYPES):
        ax = axes[row, column]
        panel = audit.loc[
            audit["animal"].eq(animal) & audit["session_type"].eq(session_type)
        ].sort_values("session")
        if panel.empty:
            raise RuntimeError(f"No training-level-16 sessions for animal {animal} s{session_type}.")

        session_ids = panel["session"].to_numpy(dtype=int)
        fractions = panel["abort_fraction"].to_numpy(dtype=float)
        sparse = panel["below_trial_threshold"].to_numpy(dtype=bool)
        trial_counts = panel["all_trials"]
        median_trials = float(trial_counts.median())
        median_label = (
            f"{median_trials:,.0f}" if median_trials.is_integer()
            else f"{median_trials:,.1f}"
        )
        full_ids = np.arange(session_ids[0], session_ids[-1] + 1)
        full_fractions = np.full(len(full_ids), np.nan)
        full_fractions[session_ids - session_ids[0]] = fractions
        missing_ids = np.setdiff1d(full_ids, session_ids)
        if len(missing_ids):
            missing_session_groups.append((animal, session_type, missing_ids.tolist()))
        if int(np.isfinite(full_fractions).sum()) != len(panel):
            raise RuntimeError(f"Plot lookup lost sessions for animal {animal} s{session_type}.")

        # NaNs at missing session IDs break the line; no gaps are interpolated.
        ax.plot(full_ids, full_fractions, color=line_color, linewidth=1.25)
        ax.scatter(
            session_ids[~sparse], fractions[~sparse],
            color=line_color, s=19, zorder=3,
        )
        ax.scatter(
            session_ids[sparse], fractions[sparse],
            facecolors="white", edgecolors=line_color, linewidths=1.2,
            s=34, zorder=4, clip_on=False,
        )
        ax.set_xlim(session_ids[0] - 0.7, session_ids[-1] + 0.7)
        ax.set_ylim(*Y_LIM)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.grid(axis="y", alpha=0.16, linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8.5)
        trial_count_title = (
            f"Trials/session: {trial_counts.min():,}–{trial_counts.max():,}; "
            f"median {median_label}"
        )
        if row == 0:
            trial_count_title = f"Session type {session_type}\n{trial_count_title}"
        ax.set_title(trial_count_title, fontsize=9.5, pad=6)
        if column == 0:
            ax.set_ylabel(f"Animal {animal}\nAbort fraction", fontsize=10)
        if row == len(ANIMALS) - 1:
            ax.set_xlabel("Session number", fontsize=10)

fig.suptitle("LED7 fixation aborts by session · training level 16", fontsize=15)
fig.text(
    0.5, 0.012,
    "Fraction = event-3 aborts / all trials in that session. "
    "LED OFF and ON and all repeat-trial values included. "
    f"Open circles: fewer than {LOW_TRIAL_THRESHOLD} trials; "
    "missing session IDs break the line.",
    ha="center", fontsize=9,
)
fig.tight_layout(rect=(0.015, 0.03, 1, 0.977), w_pad=1.2, h_pad=1.35)
fig.savefig(FIGURE_PATH, dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)
print(f"Missing-ID panel groups: {missing_session_groups}")
print(f"Figure: {FIGURE_PATH}")
