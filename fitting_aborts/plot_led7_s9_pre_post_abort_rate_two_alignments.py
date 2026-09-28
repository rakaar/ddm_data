# %%
"""Compare session-9 abort alignments before and after 2 December 2024."""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# %%
############ Editable parameters ############
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OUR_CSV = REPO_ROOT / "raw_data" / "LED7_s9.csv"
JUAN_CSV = REPO_ROOT / "juan_LED7_csvs_sess_type" / "outLED7_sess_type_9.csv"
OUTPUT_PNG = SCRIPT_DIR / "led7_s9_pre_post_abort_rate_with_juan_3x2.png"
AUDIT_CSV = SCRIPT_DIR / "led7_s9_pre_post_abort_rate_with_juan_3x2_audit.csv"
CUTOFF = pd.Timestamp("2024-12-02")
ANIMALS = (92, 93, 98, 99, 100, 103)
EXPECTED_PERIOD_ROWS = (16_511, 58_234)
BIN_WIDTH_S = 0.020
FULL_BINS = np.round(np.arange(-1.2, 1.2 + BIN_WIDTH_S / 2, BIN_WIDTH_S), 10)
DISPLAY_RANGE_S = (-0.5, 0.6)
COLORS = {0: "tab:blue", 1: "tab:red"}


# %%
############ Join Juan's dates to our unmodified raw timing column ############
key = ["animal", "session", "trial"]
our = pd.read_csv(
    OUR_CSV,
    usecols=key + ["training_level", "session_type", "repeat_trial", "LED_trial",
                   "abort_event", "timed_fix", "intended_fix", "LED_onset_time", "LED_duration"],
    float_precision="round_trip",
)
our = our.loc[our["animal"].isin(ANIMALS)].copy()
juan = pd.read_csv(
    JUAN_CSV,
    usecols=key + ["date_time_start", "date_time_end", "LED_onset_time", "LED_onset"],
    float_precision="round_trip",
)
assert not our.duplicated(key).any() and not juan.duplicated(key).any()
assert len(our) == len(juan) == sum(EXPECTED_PERIOD_ROWS)
assert our["training_level"].eq(16).all() and our["session_type"].eq(9).all()
assert (our["repeat_trial"].isin([0, 2]) | our["repeat_trial"].isna()).all()
assert our["LED_trial"].isin([0, 1]).all()

df = our.merge(juan, on=key, how="outer", validate="one_to_one", indicator=True,
               suffixes=("_raw", "_juan"))
assert df["_merge"].eq("both").all()
start = pd.to_datetime(df["date_time_start"], format="%d-%b-%Y %H:%M:%S.%f", errors="coerce")
end = pd.to_datetime(df["date_time_end"], format="%d-%b-%Y %H:%M:%S.%f", errors="coerce")
assert start.isna().sum() == 6 and end.loc[start.isna()].notna().all()
date = start.fillna(end)
assert date.notna().all() and (end.loc[start.isna()] >= CUTOFF).all()
df["period"] = np.where(date < CUTOFF, "Before 2 Dec 2024", "From 2 Dec 2024")
assert tuple(df["period"].value_counts().reindex(["Before 2 Dec 2024", "From 2 Dec 2024"])) == EXPECTED_PERIOD_ROWS

raw = df["LED_onset_time_raw"]
derived = df["intended_fix"] - raw
before = df["period"].eq("Before 2 Dec 2024")
after = ~before
assert np.allclose(df.loc[before, "LED_onset_time_juan"], raw.loc[before], rtol=0, atol=1e-9)
assert np.allclose(df.loc[before, "LED_onset"], derived.loc[before], rtol=0, atol=1e-9)
assert np.allclose(df.loc[before, "LED_duration"], 1 + raw.loc[before], rtol=0, atol=1e-9)
assert np.allclose(df.loc[after, "LED_duration"], 1 + derived.loc[after], rtol=0, atol=1e-9)
known_after = after & start.notna()
assert np.allclose(df.loc[known_after, "LED_onset"], raw.loc[known_after], rtol=0, atol=1e-9)


# %%
############ Same trials and denominator for both candidates and Juan's column ############
periods = ("Before 2 Dec 2024", "From 2 Dec 2024")
df["derived_onset"] = derived
panels = (
    (0, 0, periods[0], "Raw LED_onset_time", "LED_onset_time_raw"),
    (0, 1, periods[0], "intended_fix − raw LED_onset_time", "derived_onset"),
    (1, 0, periods[1], "Raw LED_onset_time", "LED_onset_time_raw"),
    (1, 1, periods[1], "intended_fix − raw LED_onset_time", "derived_onset"),
    (2, 0, periods[0], "Juan LED_onset: before 2 Dec", "LED_onset"),
    (2, 1, periods[1], "Juan LED_onset: from 2 Dec", "LED_onset"),
)
fig, axes = plt.subplots(3, 2, figsize=(12.6, 12.2), sharex=True, sharey=True)
audit_rows = []
rates = {}
for row, col, period, title, onset_col in panels:
    pool = df.loc[df["period"].eq(period)]
    ax = axes[row, col]
    for led in (0, 1):
        group = pool.loc[pool["LED_trial"].eq(led)]
        coded = group.loc[group["abort_event"].eq(3)]
        finite = coded.loc[np.isfinite(coded["timed_fix"])]
        assert finite["timed_fix"].lt(finite["intended_fix"]).all()
        times = (finite["timed_fix"] - finite[onset_col]).to_numpy(float)
        counts, _ = np.histogram(times, bins=FULL_BINS)
        assert counts.sum() == len(finite)
        heights = counts / (len(group) * BIN_WIDTH_S)
        area = float(np.sum(heights * np.diff(FULL_BINS)))
        pre_area = float(np.sum(heights[FULL_BINS[1:] <= 0] * BIN_WIDTH_S))
        assert np.isclose(area, len(finite) / len(group))
        assert np.isclose(pre_area, (times < 0).sum() / len(group))
        rates[period, onset_col, led] = heights
        ax.stairs(heights, FULL_BINS, color=COLORS[led], linewidth=1.45,
                  label=f"LED {'OFF' if led == 0 else 'ON'}")
        audit_rows.append({
            "period": period,
            "alignment": title,
            "LED_trial": led,
            "all_trials": len(group),
            "coded_event3_aborts": len(coded),
            "finite_plotted_aborts": len(finite),
            "missing_abort_timing": len(coded) - len(finite),
            "aborts_before_candidate_onset": int((times < 0).sum()),
            "pre_onset_area": pre_area,
            "full_histogram_area": area,
        })
    ax.axvline(0, color="0.35", linestyle="--", linewidth=1)
    ax.set_xlim(*DISPLAY_RANGE_S)
    ax.grid(axis="y", alpha=0.2)
    if row in (0, 2):
        ax.set_title(title)
    if col == 0:
        ax.set_ylabel(f"{period if row < 2 else 'Juan LED_onset'}\nAbort rate (s⁻¹)")
    if row == 2:
        ax.set_xlabel("Abort time relative to candidate onset (s)")

for led in (0, 1):
    assert np.array_equal(rates[periods[0], "LED_onset", led],
                          rates[periods[0], "derived_onset", led])
    assert np.array_equal(rates[periods[1], "LED_onset", led],
                          rates[periods[1], "LED_onset_time_raw", led])
axes[0, 1].legend(frameon=False, loc="upper right")
fig.suptitle("LED7 session type 9: candidate alignments and Juan's LED_onset", fontsize=14)
fig.tight_layout()
fig.savefig(OUTPUT_PNG, dpi=250, bbox_inches="tight")
plt.close(fig)

audit = pd.DataFrame(audit_rows)
audit.to_csv(AUDIT_CSV, index=False)
print(f"Before: {int(before.sum()):,}; from cutoff: {int(after.sum()):,} (six dates use trial_end)")
print(audit.to_string(index=False))
print(f"Figure: {OUTPUT_PNG}")
print(f"Audit: {AUDIT_CSV}")
