# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

try:
    from matio import load_from_mat
except ImportError as exc:
    raise ImportError(
        "This script needs the isolated mat-io decoder. Run it as:\n"
        "env PYTHONPATH=/tmp/codex_mat_io .venv/bin/python "
        "fitting_aborts/plot_led7_s7_filtered_vs_s9_intended_fix_theory.py"
    ) from exc


# %%
############ Parameters (edit here) ############
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

INPUT_MAT_PATH = REPO_ROOT / "raw_data" / "outMatrix_LED7_latest.mat"
TABLE_NAME = "totalout_stGtACRII"
OUTPUT_FIGURE_PATH = (
    SCRIPT_DIR / "led7_s7_filtered_vs_s9_intended_fix_theory.png"
)

TRAINING_LEVEL = 16
SESSION_TYPE_7 = 7
SESSION_TYPE_9 = 9
ALLOWED_REPEAT_TRIALS = {0, 2}
LED_TRIAL = 1
DERIVED_ONSET_FLOOR_S = 0.100

BASE_PER_STAGE_S = 0.100
EXPONENTIAL_MEAN_S = 0.200
EXPONENTIAL_UPPER_S = 1.000

HISTOGRAM_START_S = 0.0
HISTOGRAM_STOP_S = 2.5
HISTOGRAM_BIN_S = 0.020
THEORY_STEP_S = 0.001

SESSION_9_COLOR = "#4477AA"
SESSION_7_COLOR = "#CC6677"
THEORY_COLOR = "black"
OUTPUT_DPI = 300
SHOW_PLOT = False

EXPECTED_COUNTS = {
    "session_9_on": 39_380,
    "session_7_on_before_cutoff": 46_383,
    "session_7_on_removed": 9_256,
    "session_7_on_retained": 37_127,
}


# %%
############ Exact theory: (0.1 + X1) + (0.1 + X2) ############
rate = 1.0 / EXPONENTIAL_MEAN_S
component_normalizer = 1.0 - np.exp(-rate * EXPONENTIAL_UPPER_S)
total_base_s = 2.0 * BASE_PER_STAGE_S


def intended_fix_cdf(values):
    """CDF of 2*base + X1 + X2 for iid truncated exponentials Xi."""
    summed_exponential_time = np.asarray(values, dtype=float) - total_base_s
    cdf = np.zeros_like(summed_exponential_time)

    first_half = (
        (summed_exponential_time > 0)
        & (summed_exponential_time <= EXPONENTIAL_UPPER_S)
    )
    s = summed_exponential_time[first_half]
    cdf[first_half] = (
        1.0 - np.exp(-rate * s) * (1.0 + rate * s)
    ) / component_normalizer**2

    second_half = (
        (summed_exponential_time > EXPONENTIAL_UPPER_S)
        & (summed_exponential_time < 2.0 * EXPONENTIAL_UPPER_S)
    )
    remaining = (
        2.0 * EXPONENTIAL_UPPER_S
        - summed_exponential_time[second_half]
    )
    upper_tail = (
        np.exp(-2.0 * rate * EXPONENTIAL_UPPER_S)
        * (np.exp(rate * remaining) * (rate * remaining - 1.0) + 1.0)
        / component_normalizer**2
    )
    cdf[second_half] = 1.0 - upper_tail

    cdf[
        summed_exponential_time >= 2.0 * EXPONENTIAL_UPPER_S
    ] = 1.0
    return cdf


def intended_fix_pdf(values):
    """PDF of 2*base + X1 + X2 for iid truncated exponentials Xi."""
    summed_exponential_time = np.asarray(values, dtype=float) - total_base_s
    convolution_width = np.zeros_like(summed_exponential_time)

    first_half = (
        (summed_exponential_time >= 0)
        & (summed_exponential_time <= EXPONENTIAL_UPPER_S)
    )
    convolution_width[first_half] = summed_exponential_time[first_half]

    second_half = (
        (summed_exponential_time > EXPONENTIAL_UPPER_S)
        & (summed_exponential_time <= 2.0 * EXPONENTIAL_UPPER_S)
    )
    convolution_width[second_half] = (
        2.0 * EXPONENTIAL_UPPER_S
        - summed_exponential_time[second_half]
    )

    return (
        rate**2
        * np.exp(-rate * summed_exponential_time)
        * convolution_width
        / component_normalizer**2
    )


def empirical_cdf_distance(values):
    ordered = np.sort(np.asarray(values, dtype=float))
    n_values = ordered.size
    theory_cdf = intended_fix_cdf(ordered)
    ranks = np.arange(1, n_values + 1)
    return float(
        max(
            np.max(ranks / n_values - theory_cdf),
            np.max(theory_cdf - (ranks - 1) / n_values),
        )
    )


# %%
############ Load and filter the authoritative LED7 MATLAB table ############
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
    "session_type",
    "training_level",
    "repeat_trial",
    "LED_trial",
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

session_9_on_df = repeat_df.loc[
    repeat_df["session_type"].eq(SESSION_TYPE_9)
    & repeat_df["LED_trial"].eq(LED_TRIAL)
].copy()
session_7_on_df = repeat_df.loc[
    repeat_df["session_type"].eq(SESSION_TYPE_7)
    & repeat_df["LED_trial"].eq(LED_TRIAL)
].copy()

session_7_derived_onset = (
    session_7_on_df["intended_fix"] - session_7_on_df["LED_onset_time"]
)
if not np.isfinite(session_7_derived_onset).all():
    raise RuntimeError("Session type 7 contains non-finite derived LED onsets.")

session_7_removed_mask = session_7_derived_onset < DERIVED_ONSET_FLOOR_S
session_7_retained_df = session_7_on_df.loc[
    ~session_7_removed_mask
].copy()

observed_counts = {
    "session_9_on": len(session_9_on_df),
    "session_7_on_before_cutoff": len(session_7_on_df),
    "session_7_on_removed": int(session_7_removed_mask.sum()),
    "session_7_on_retained": len(session_7_retained_df),
}
if observed_counts != EXPECTED_COUNTS:
    raise RuntimeError(
        f"Expected counts {EXPECTED_COUNTS}, found {observed_counts}."
    )

session_9_values = session_9_on_df["intended_fix"].to_numpy(dtype=float)
session_7_values = session_7_retained_df["intended_fix"].to_numpy(
    dtype=float
)
if not np.isfinite(session_9_values).all():
    raise RuntimeError("Session type 9 intended_fix contains non-finite values.")
if not np.isfinite(session_7_values).all():
    raise RuntimeError("Retained session type 7 intended_fix is non-finite.")
if (
    session_7_retained_df["intended_fix"]
    - session_7_retained_df["LED_onset_time"]
).lt(DERIVED_ONSET_FLOOR_S).any():
    raise RuntimeError("A session type 7 onset below 100 ms was retained.")


# %%
############ Unit-area histograms and numerical comparison ############
histogram_edges = np.arange(
    HISTOGRAM_START_S,
    HISTOGRAM_STOP_S + HISTOGRAM_BIN_S / 2,
    HISTOGRAM_BIN_S,
)
theory_times = np.arange(
    HISTOGRAM_START_S,
    HISTOGRAM_STOP_S + THEORY_STEP_S / 2,
    THEORY_STEP_S,
)
theory_density = intended_fix_pdf(theory_times)

session_9_counts, _ = np.histogram(session_9_values, bins=histogram_edges)
session_7_counts, _ = np.histogram(session_7_values, bins=histogram_edges)
if int(session_9_counts.sum()) != len(session_9_values):
    raise RuntimeError("The histogram omits session type 9 observations.")
if int(session_7_counts.sum()) != len(session_7_values):
    raise RuntimeError("The histogram omits retained session type 7 observations.")

session_9_density = session_9_counts / (
    len(session_9_values) * HISTOGRAM_BIN_S
)
session_7_density = session_7_counts / (
    len(session_7_values) * HISTOGRAM_BIN_S
)
session_9_area = float(
    np.sum(session_9_density * np.diff(histogram_edges))
)
session_7_area = float(
    np.sum(session_7_density * np.diff(histogram_edges))
)
if not np.isclose(session_9_area, 1.0, atol=1e-14, rtol=0):
    raise RuntimeError(f"Session type 9 histogram area is {session_9_area}.")
if not np.isclose(session_7_area, 1.0, atol=1e-14, rtol=0):
    raise RuntimeError(f"Session type 7 histogram area is {session_7_area}.")

session_9_theory_distance = empirical_cdf_distance(session_9_values)
session_7_theory_distance = empirical_cdf_distance(session_7_values)
two_sample_ks = ks_2samp(session_7_values, session_9_values, method="asymp")
wasserstein_ms = 1000.0 * wasserstein_distance(
    session_7_values,
    session_9_values,
)


# %%
############ Plot the two data distributions and fixed theory ############
fig, ax = plt.subplots(figsize=(10.2, 5.7))

ax.stairs(
    session_9_density,
    histogram_edges,
    color=SESSION_9_COLOR,
    linewidth=1.8,
    label=f"Session type 9 (n = {len(session_9_values):,})",
)
ax.stairs(
    session_7_density,
    histogram_edges,
    color=SESSION_7_COLOR,
    linewidth=1.8,
    label=(
        "Session type 7, derived onset >= 100 ms "
        f"(n = {len(session_7_values):,})"
    ),
)
ax.plot(
    theory_times,
    theory_density,
    color=THEORY_COLOR,
    linewidth=2.2,
    label=(
        r"Theory: $(0.1+X_1)+(0.1+X_2)$, "
        r"$X_i\sim\mathrm{Exp}(\mathrm{mean}=0.2\,\mathrm{s})$"
    ),
)

ax.set_title("LED7 intended_fix: session type 9 and filtered session type 7")
ax.set_xlabel("intended_fix (s)")
ax.set_ylabel(r"Density ($\mathrm{s}^{-1}$)")
ax.set_xlim(HISTOGRAM_START_S, HISTOGRAM_STOP_S)
ax.set_xticks(np.arange(0, HISTOGRAM_STOP_S + 0.001, 0.5))
ax.grid(axis="y", alpha=0.22)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False)

fig.tight_layout()
fig.savefig(OUTPUT_FIGURE_PATH, dpi=OUTPUT_DPI, bbox_inches="tight")
if SHOW_PLOT:
    plt.show()
else:
    plt.close(fig)


# %%
############ Audit output ############
print("Filters: training_level=16; repeat_trial in {0, 2, NaN}; LED_trial=1")
print(
    "Session type 7 cutoff: intended_fix - LED_onset_time >= "
    f"{DERIVED_ONSET_FLOOR_S:.3f} s"
)
print(
    f"Session type 7: before={len(session_7_on_df):,}, "
    f"removed={int(session_7_removed_mask.sum()):,}, "
    f"retained={len(session_7_values):,}"
)
print(
    f"Session type 9: n={len(session_9_values):,}, "
    f"mean={session_9_values.mean():.9f} s, "
    f"theory CDF D={session_9_theory_distance:.9f}"
)
print(
    f"Filtered session type 7: n={len(session_7_values):,}, "
    f"mean={session_7_values.mean():.9f} s, "
    f"theory CDF D={session_7_theory_distance:.9f}"
)
print(
    f"Session 7 versus session 9: KS D={two_sample_ks.statistic:.9f}, "
    f"p={two_sample_ks.pvalue:.9g}, Wasserstein={wasserstein_ms:.6f} ms"
)
print(f"Theory resolution: {THEORY_STEP_S:.3f} s")
print(f"Saved: {OUTPUT_FIGURE_PATH}")
