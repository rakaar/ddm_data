# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# %%
############ Parameters (edit here) ############
TRAINING_LEVEL = 16
SESSION_TYPE = 7
LED_TRIAL = 1
ALLOWED_REPEAT_TRIALS = [0, 2]
ANIMALS = [92, 93, 98, 99, 100, 103]

BIN_START_S = 0.0
BIN_STOP_S = 1.2
BIN_SIZE_S = 0.020
GAP_BIN_START_S = -0.2
GAP_BIN_STOP_S = 2.2

EXPECTED_ROWS = 30_910
EXPECTED_ABORTS = 6_336
EXPECTED_CENSORED = 24_574

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
INPUT_PATH = REPO_ROOT / "raw_data" / "out_LED.csv"
OUTPUT_PATH = SCRIPT_DIR / "led7_s7_svi_led_on_onset_distribution.png"
GAP_OUTPUT_PATH = (
    SCRIPT_DIR
    / "led7_s7_svi_led_on_intended_fix_minus_onset_distribution.png"
)


# %%
############ Reproduce the current no-truncation SVI fit rows ############
if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Could not find input CSV: {INPUT_PATH}")

raw_df = pd.read_csv(INPUT_PATH)

repeat_df = raw_df.loc[
    raw_df["repeat_trial"].isin(ALLOWED_REPEAT_TRIALS)
    | raw_df["repeat_trial"].isna()
].copy()

session_df = repeat_df.loc[
    repeat_df["session_type"].eq(SESSION_TYPE)
    & repeat_df["training_level"].eq(TRAINING_LEVEL)
    & repeat_df["animal"].astype(int).isin(ANIMALS)
].copy()

finite_df = session_df.dropna(
    subset=["intended_fix", "LED_onset_time", "timed_fix"]
).copy()

fit_df = finite_df.loc[
    finite_df["abort_event"].eq(3)
    | finite_df["success"].isin([1, -1])
].copy()

led_on_df = fit_df.loc[fit_df["LED_trial"].eq(LED_TRIAL)].copy()

abort_mask = led_on_df["abort_event"].eq(3)
censored_mask = led_on_df["success"].isin([1, -1])

if len(led_on_df) != EXPECTED_ROWS:
    raise RuntimeError(
        f"Expected {EXPECTED_ROWS:,} fitted LED-ON rows, found {len(led_on_df):,}."
    )
if int(abort_mask.sum()) != EXPECTED_ABORTS:
    raise RuntimeError(
        f"Expected {EXPECTED_ABORTS:,} LED-ON aborts, found {abort_mask.sum():,}."
    )
if int(censored_mask.sum()) != EXPECTED_CENSORED:
    raise RuntimeError(
        "Expected "
        f"{EXPECTED_CENSORED:,} LED-ON censored trials, "
        f"found {censored_mask.sum():,}."
    )
if (abort_mask & censored_mask).any():
    raise RuntimeError("Abort and censored trial classifications overlap.")


# %%
############ Plot the LED onset-time distribution ############
led_onset_time = led_on_df["LED_onset_time"].to_numpy(dtype=float)
bin_edges = np.arange(
    BIN_START_S,
    BIN_STOP_S + BIN_SIZE_S / 2,
    BIN_SIZE_S,
)

if led_onset_time.min() < BIN_START_S or led_onset_time.max() > BIN_STOP_S:
    raise RuntimeError("Configured bins do not contain every LED onset time.")

fig, ax = plt.subplots(figsize=(7.2, 4.5))
ax.hist(
    led_onset_time,
    bins=bin_edges,
    histtype="step",
    color="tab:red",
    linewidth=1.8,
)
ax.set_xlim(BIN_START_S, BIN_STOP_S)
ax.set_xlabel("LED_onset_time (s)")
ax.set_ylabel("Count per 20 ms bin")
ax.set_title(
    "LED7 session type 7, training level 16, LED ON\n"
    f"n = {len(led_on_df):,}"
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(OUTPUT_PATH, dpi=250, bbox_inches="tight")
plt.close(fig)


# %%
############ Plot intended fixation minus LED onset ############
intended_minus_onset = (
    led_on_df["intended_fix"] - led_on_df["LED_onset_time"]
).to_numpy(dtype=float)

bin_size_ms = int(round(BIN_SIZE_S * 1000))
gap_bin_edges = np.arange(
    int(round(GAP_BIN_START_S * 1000)),
    int(round(GAP_BIN_STOP_S * 1000)) + bin_size_ms,
    bin_size_ms,
    dtype=float,
) / 1000

if (
    intended_minus_onset.min() < GAP_BIN_START_S
    or intended_minus_onset.max() > GAP_BIN_STOP_S
):
    raise RuntimeError(
        "Configured gap bins do not contain every intended-fix minus onset value."
    )

fig, ax = plt.subplots(figsize=(7.2, 4.5))
ax.hist(
    intended_minus_onset,
    bins=gap_bin_edges,
    histtype="step",
    color="tab:red",
    linewidth=1.8,
)
ax.axvline(0, color="0.35", linestyle="--", linewidth=1)
ax.set_xlim(GAP_BIN_START_S, GAP_BIN_STOP_S)
ax.set_xlabel("intended_fix - LED_onset_time (s)")
ax.set_ylabel("Count per 20 ms bin")
ax.set_title(
    "LED7 session type 7, training level 16, LED ON\n"
    f"n = {len(led_on_df):,}"
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(GAP_OUTPUT_PATH, dpi=250, bbox_inches="tight")
plt.close(fig)


# %%
############ Numerical audit ############
quantiles = np.quantile(
    led_onset_time,
    [0.0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0],
)
gap_quantiles = np.quantile(
    intended_minus_onset,
    [0.0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0],
)

print(f"Loaded: {INPUT_PATH}")
print(
    "Filters: repeat_trial in {0, 2} or NaN; session_type=7; "
    "training_level=16; animals 92, 93, 98, 99, 100, 103; "
    "finite intended_fix/LED_onset_time/timed_fix; "
    "abort_event=3 or success in {-1, 1}; LED_trial=1"
)
print(f"Fitted LED-ON rows: {len(led_on_df):,}")
print(f"Abort rows: {int(abort_mask.sum()):,}")
print(f"Censored successful rows: {int(censored_mask.sum()):,}")
print(f"Animals: {sorted(led_on_df['animal'].astype(int).unique().tolist())}")
print(f"Bin width: {BIN_SIZE_S * 1000:.0f} ms")
print(f"LED_onset_time mean: {led_onset_time.mean():.6f} s")
print(f"LED_onset_time SD: {led_onset_time.std(ddof=1):.6f} s")
print(
    "LED_onset_time min/p01/p05/p25/median/p75/p95/p99/max (s): "
    + np.array2string(quantiles, precision=6, separator=", ")
)
print(
    "intended_fix - LED_onset_time: "
    f"negative={int((intended_minus_onset < 0).sum()):,}, "
    f"zero={int((intended_minus_onset == 0).sum()):,}, "
    f"positive={int((intended_minus_onset > 0).sum()):,}"
)
print(f"Gap mean: {intended_minus_onset.mean():.6f} s")
print(f"Gap SD: {intended_minus_onset.std(ddof=1):.6f} s")
print(
    "Gap min/p01/p05/p25/median/p75/p95/p99/max (s): "
    + np.array2string(gap_quantiles, precision=6, separator=", ")
)
print("Saved figures:")
print(OUTPUT_PATH.resolve())
print(GAP_OUTPUT_PATH.resolve())
