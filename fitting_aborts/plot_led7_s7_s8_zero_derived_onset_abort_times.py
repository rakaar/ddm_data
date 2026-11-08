# %%
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

try:
    from matio import load_from_mat
except ImportError as exc:
    raise ImportError(
        "This script needs the isolated mat-io decoder. Run it as:\n"
        "env PYTHONPATH=/tmp/codex_mat_io .venv/bin/python "
        "fitting_aborts/plot_led7_s7_s8_zero_derived_onset_abort_times.py"
    ) from exc


# %%
############ Parameters (edit here) ############
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

INPUT_MAT_PATH = REPO_ROOT / "raw_data" / "outMatrix_LED7_latest.mat"
TABLE_NAME = "totalout_stGtACRII"
OUTPUT_FIGURE_PATH = (
    SCRIPT_DIR / "led7_s7_s8_zero_derived_onset_abort_times_1x2.png"
)

TRAINING_LEVEL = 16
SESSION_TYPES = [7, 8]
ALLOWED_REPEAT_TRIALS = {0, 2}
LED_TRIAL = 1
FIXATION_ABORT_EVENT = 3

BIN_WIDTH_S = 0.020
BIN_EDGES_S = np.arange(-100, 700 + 20, 20, dtype=float) / 1000
DISPLAY_LIMITS_S = (-0.1, 0.7)

EXPECTED_COUNTS = {
    7: {
        "led_on_trials": 46_383,
        "zero_onset_trials": 3_406,
        "coded_aborts": 174,
        "finite_aborts": 174,
    },
    8: {
        "led_on_trials": 55_644,
        "zero_onset_trials": 4_073,
        "coded_aborts": 438,
        "finite_aborts": 438,
    },
}

COLOR = "tab:red"
OUTPUT_DPI = 250
SHOW_PLOT = False


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


# %%
############ Select zero-derived-onset LED-ON fixation aborts ############
training_df = raw_df.loc[
    raw_df["training_level"].eq(TRAINING_LEVEL)
].copy()

panel_payloads = []
audit_rows = []

for session_type in SESSION_TYPES:
    session_df = training_df.loc[
        training_df["session_type"].eq(session_type)
    ].copy()
    repeat_df = session_df.loc[
        session_df["repeat_trial"].isin(ALLOWED_REPEAT_TRIALS)
        | session_df["repeat_trial"].isna()
    ].copy()
    led_on_df = repeat_df.loc[repeat_df["LED_trial"].eq(LED_TRIAL)].copy()

    derived_led_onset = (
        led_on_df["intended_fix"] - led_on_df["LED_onset_time"]
    )
    if not np.isfinite(derived_led_onset).all():
        raise RuntimeError(
            f"Session type {session_type} has non-finite derived LED onsets."
        )
    if (derived_led_onset < 0).any():
        raise RuntimeError(
            f"Session type {session_type} contains a negative derived LED onset."
        )

    exact_zero_mask = derived_led_onset.eq(0)
    numerical_zero_mask = derived_led_onset.abs().le(1e-12)
    if not exact_zero_mask.equals(numerical_zero_mask):
        raise RuntimeError(
            f"Session type {session_type} has near-zero derived onsets that are "
            "not represented by exact zero."
        )

    zero_onset_df = led_on_df.loc[exact_zero_mask].copy()
    coded_abort_df = zero_onset_df.loc[
        zero_onset_df["abort_event"].eq(FIXATION_ABORT_EVENT)
    ].copy()
    finite_abort_mask = np.isfinite(
        coded_abort_df[["timed_fix", "intended_fix", "LED_onset_time"]]
    ).all(axis=1)
    plotted_abort_df = coded_abort_df.loc[finite_abort_mask].copy()

    observed_counts = {
        "led_on_trials": len(led_on_df),
        "zero_onset_trials": len(zero_onset_df),
        "coded_aborts": len(coded_abort_df),
        "finite_aborts": len(plotted_abort_df),
    }
    if observed_counts != EXPECTED_COUNTS[session_type]:
        raise RuntimeError(
            f"Session type {session_type} expected counts "
            f"{EXPECTED_COUNTS[session_type]}, found {observed_counts}."
        )

    plotted_derived_led_onset = (
        plotted_abort_df["intended_fix"]
        - plotted_abort_df["LED_onset_time"]
    ).to_numpy(dtype=float)
    if not np.equal(plotted_derived_led_onset, 0).all():
        raise RuntimeError(
            f"Session type {session_type} plotted rows are not all at zero onset."
        )

    aligned_abort_time = (
        plotted_abort_df["timed_fix"].to_numpy(dtype=float)
        - plotted_derived_led_onset
    )
    if not (
        plotted_abort_df["timed_fix"] < plotted_abort_df["intended_fix"]
    ).all():
        raise RuntimeError(
            f"Session type {session_type} has an event-3 row with "
            "timed_fix >= intended_fix."
        )

    counts, _ = np.histogram(aligned_abort_time, bins=BIN_EDGES_S)
    if int(counts.sum()) != len(aligned_abort_time):
        raise RuntimeError(
            f"Session type {session_type} histogram contains {counts.sum():,} "
            f"of {len(aligned_abort_time):,} finite aborts."
        )

    quantile_probabilities = [0, 0.05, 0.25, 0.50, 0.75, 0.95, 1]
    quantiles = np.quantile(aligned_abort_time, quantile_probabilities)
    panel_payloads.append(
        {
            "session_type": session_type,
            "counts": counts,
            "aligned_abort_time": aligned_abort_time,
            "n_zero_onset_trials": len(zero_onset_df),
            "n_aborts": len(plotted_abort_df),
        }
    )
    audit_rows.append(
        {
            "session_type": session_type,
            "LED_ON_trials": len(led_on_df),
            "zero_derived_onset_trials": len(zero_onset_df),
            "zero_onset_trial_fraction": len(zero_onset_df) / len(led_on_df),
            "coded_event_3_aborts": len(coded_abort_df),
            "finite_plotted_aborts": len(plotted_abort_df),
            "abort_fraction_among_zero_onset_trials": (
                len(plotted_abort_df) / len(zero_onset_df)
            ),
            "min_s": quantiles[0],
            "p05_s": quantiles[1],
            "p25_s": quantiles[2],
            "median_s": quantiles[3],
            "p75_s": quantiles[4],
            "p95_s": quantiles[5],
            "max_s": quantiles[6],
            "fraction_within_100_ms": float(
                np.mean(aligned_abort_time <= 0.100)
            ),
        }
    )


# %%
############ Plot session types 7 and 8 separately ############
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharex=True)

for ax, payload in zip(axes, panel_payloads):
    ax.stairs(
        payload["counts"],
        BIN_EDGES_S,
        color=COLOR,
        linewidth=1.7,
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlim(DISPLAY_LIMITS_S)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(f"Session type {payload['session_type']}", fontsize=12)
    ax.set_xlabel(
        "timed_fix - (intended_fix - LED_onset_time) (s)",
        fontsize=10,
    )
    ax.text(
        0.97,
        0.94,
        f"n = {payload['n_aborts']:,}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.16, linewidth=0.6)

axes[0].set_ylabel("Count per 20 ms bin", fontsize=10)
fig.suptitle(
    "LED7 LED-ON fixation aborts with intended_fix - LED_onset_time = 0\n"
    "training level 16; repeat_trial = 0, 2, or NaN",
    fontsize=14,
    y=0.995,
)
fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=2.2)
fig.savefig(OUTPUT_FIGURE_PATH, dpi=OUTPUT_DPI, bbox_inches="tight")

if SHOW_PLOT:
    plt.show()
else:
    plt.close(fig)


# %%
############ Print the numerical audit ############
audit_df = pd.DataFrame(audit_rows)

print(f"Source: {INPUT_MAT_PATH} :: {TABLE_NAME}")
print(
    "Filters in order: training_level == 16; session_type in {7, 8} "
    "analyzed separately; repeat_trial in {0, 2, NaN}; LED_trial == 1; "
    "intended_fix - LED_onset_time == 0; abort_event == 3; finite timing."
)
print(
    "Plotted coordinate: timed_fix - (intended_fix - LED_onset_time). "
    "Because the selected derived onset is exactly zero, this equals timed_fix."
)
print("\nAudit")
print(
    audit_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)
print(f"\nSaved figure: {OUTPUT_FIGURE_PATH.resolve()}")
