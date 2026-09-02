# %%
from pathlib import Path


# %%
############ Parameters (fixed theoretical model; not fitted) ############
TRAINING_LEVEL = 16
SESSION_TYPE = 9
BASE_PER_STAGE_S = 0.1
EXPONENTIAL_MEAN_S = 0.2
EXPONENTIAL_UPPER_S = 1.0

HISTOGRAM_START_S = 0.0
HISTOGRAM_STOP_S = 2.5
HISTOGRAM_BIN_S = 0.020
THEORY_STEP_S = 0.001

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_PATH = REPO_ROOT / "raw_data" / "LED7_s9.csv"
OUTPUT_PATH = (
    SCRIPT_DIR
    / "led7_stgtacrii_session9_training16_intended_fix_data_vs_theory.png"
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# %%
############ Exact theory for the sum of two truncated exponentials ############
rate = 1.0 / EXPONENTIAL_MEAN_S
normalizer = 1.0 - np.exp(-rate * EXPONENTIAL_UPPER_S)
total_base = 2.0 * BASE_PER_STAGE_S


def intended_fix_cdf(values):
    """CDF of 2*base + X1 + X2, with iid truncated exponential Xi."""
    summed_exponential_time = np.asarray(values, dtype=float) - total_base
    cdf = np.zeros_like(summed_exponential_time)

    first_half = (
        (summed_exponential_time > 0)
        & (summed_exponential_time <= EXPONENTIAL_UPPER_S)
    )
    s = summed_exponential_time[first_half]
    cdf[first_half] = (
        1.0 - np.exp(-rate * s) * (1.0 + rate * s)
    ) / normalizer**2

    second_half = (
        (summed_exponential_time > EXPONENTIAL_UPPER_S)
        & (summed_exponential_time < 2.0 * EXPONENTIAL_UPPER_S)
    )
    remaining = 2.0 * EXPONENTIAL_UPPER_S - summed_exponential_time[second_half]
    upper_tail = (
        np.exp(-2.0 * rate * EXPONENTIAL_UPPER_S)
        * (np.exp(rate * remaining) * (rate * remaining - 1.0) + 1.0)
        / normalizer**2
    )
    cdf[second_half] = 1.0 - upper_tail

    cdf[summed_exponential_time >= 2.0 * EXPONENTIAL_UPPER_S] = 1.0
    return cdf


def intended_fix_pdf(values):
    """PDF of 2*base + X1 + X2, with iid truncated exponential Xi."""
    summed_exponential_time = np.asarray(values, dtype=float) - total_base
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
        2.0 * EXPONENTIAL_UPPER_S - summed_exponential_time[second_half]
    )

    return (
        rate**2
        * np.exp(-rate * summed_exponential_time)
        * convolution_width
        / normalizer**2
    )


def empirical_cdf_distance(values):
    """Maximum absolute empirical-versus-theoretical CDF difference."""
    ordered = np.sort(values)
    n = ordered.size
    theory = intended_fix_cdf(ordered)
    ranks = np.arange(1, n + 1)
    return max(
        np.max(ranks / n - theory),
        np.max(theory - (ranks - 1) / n),
    )


# %%
############ Load the filtered session-9 CSV ############
table = pd.read_csv(DATA_PATH)

training_level_table = table.loc[
    table["training_level"].eq(TRAINING_LEVEL)
].copy()
session_table = training_level_table.loc[
    training_level_table["session_type"].eq(SESSION_TYPE)
].copy()

if len(session_table) != len(table):
    raise RuntimeError(
        "LED7_s9.csv contains rows outside training_level=16 and session_type=9."
    )
if not (
    session_table["repeat_trial"].isin([0, 2])
    | session_table["repeat_trial"].isna()
).all():
    raise RuntimeError("LED7_s9.csv contains an unexpected repeat_trial value.")
if not (
    session_table["LED_trial"].isin([0, 1])
    | session_table["LED_trial"].isna()
).all():
    raise RuntimeError("LED7_s9.csv contains an unexpected LED_trial value.")


# %%
############ Raw-count histograms with a 1 ms theoretical curve ############
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

truncated_exponential_mean = (
    1.0 / rate
    - EXPONENTIAL_UPPER_S / np.expm1(rate * EXPONENTIAL_UPPER_S)
)
theory_intended_fix_mean = total_base + 2.0 * truncated_exponential_mean

groups = [
    (0, "LED OFF", "#4477AA"),
    (1, "LED ON", "#EE7733"),
]

fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.7), sharex=True)
audit_rows = []

for ax, (led_trial, label, color) in zip(axes, groups):
    intended_fix = session_table.loc[
        session_table["LED_trial"].eq(led_trial), "intended_fix"
    ].to_numpy(dtype=float)
    intended_fix = intended_fix[np.isfinite(intended_fix)]
    cdf_distance = empirical_cdf_distance(intended_fix)

    ax.hist(
        intended_fix,
        bins=histogram_edges,
        color=color,
        alpha=0.68,
        linewidth=0,
    )
    ax.plot(
        theory_times,
        intended_fix.size * HISTOGRAM_BIN_S * theory_density,
        color="black",
        linewidth=2.0,
        zorder=3,
    )

    ax.set_title(label)
    ax.set_xlim(HISTOGRAM_START_S, HISTOGRAM_STOP_S)
    ax.set_xticks(np.arange(HISTOGRAM_START_S, HISTOGRAM_STOP_S + 0.001, 0.5))
    ax.set_xlabel("intended_fix (s)")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    audit_rows.append(
        (
            label,
            intended_fix.size,
            intended_fix.mean(),
            intended_fix.min(),
            intended_fix.max(),
            cdf_distance,
        )
    )

fig.suptitle(
    "LED7 totalout_stGtACRII: intended_fix data versus fixed theory\n"
    "training level 16, session type 9;  "
    "$T=(0.1+X_1)+(0.1+X_2)$,  "
    "$X_i\\sim\\mathrm{Exp}(\\mathrm{mean}=0.2\\,s)$",
    fontsize=12.5,
)
fig.tight_layout(rect=(0, 0, 1, 0.88))
fig.savefig(OUTPUT_PATH, dpi=250, bbox_inches="tight")
plt.close(fig)


# %%
############ Audit output ############
print(
    f"Theory: rate={rate:.6f} s^-1, per-stage base={BASE_PER_STAGE_S:.3f} s, "
    f"component support=[0, {EXPONENTIAL_UPPER_S:.1f}] s"
)
print(f"Histogram bin: {HISTOGRAM_BIN_S:.3f} s")
print(f"Theory resolution: {THEORY_STEP_S:.3f} s")
print(f"Theory intended_fix mean: {theory_intended_fix_mean:.9f} s")
for label, n, mean, minimum, maximum, distance in audit_rows:
    print(
        f"{label}: n={n:,}, mean={mean:.9f}, min={minimum:.9f}, "
        f"max={maximum:.9f}, CDF D={distance:.9f}"
    )
print(f"Saved: {OUTPUT_PATH}")
