# %%
"""Pooled LED7 session-9 fixation-abort rates aligned to Juan's LED_onset."""

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
INPUT_CSV = SCRIPT_DIR.parent / "juan_LED7_csvs_sess_type" / "outLED7_sess_type_9.csv"
OUTPUT_PNG = SCRIPT_DIR / "juan_led7_s9_abort_rate_aligned_to_led_onset.png"
AUDIT_CSV = SCRIPT_DIR / "juan_led7_s9_abort_rate_aligned_to_led_onset_audit.csv"
ANIMALS = (92, 93, 98, 99, 100, 103)
EXPECTED_ROWS = 74_745
EXPECTED_LED_COUNTS = {0: 45_124, 1: 29_621}
EXPECTED_CODED_ABORTS = 14_001
BIN_WIDTH_S = 0.020
FULL_BINS = np.round(np.arange(-2.0, 2.0 + BIN_WIDTH_S / 2, BIN_WIDTH_S), 10)
DISPLAY_RANGE_S = (-0.5, 0.6)
COLORS = {0: "tab:blue", 1: "tab:red"}


# %%
############ Use Juan's CSV and his LED_onset value directly ############
df = pd.read_csv(INPUT_CSV, float_precision="round_trip")
required = {"animal", "session", "trial", "training_level", "session_type",
            "repeat_trial", "LED_trial", "abort_event", "TotalFixTime",
            "intended_fix", "LED_onset"}
assert required.issubset(df.columns)
assert len(df) == EXPECTED_ROWS and tuple(sorted(df["animal"].unique())) == ANIMALS
assert not df.duplicated(["animal", "session", "trial"]).any()
assert df["training_level"].eq(16).all() and df["session_type"].eq(9).all()
assert (df["repeat_trial"].isin([0, 2]) | df["repeat_trial"].isna()).all()
assert df["LED_trial"].isin([0, 1]).all()
assert int(df["abort_event"].eq(3).sum()) == EXPECTED_CODED_ABORTS
assert np.isfinite(df[["LED_onset", "intended_fix"]]).all().all()
assert df["LED_onset"].ge(0).all()
assert df["LED_onset"].le(df["intended_fix"]).all()


# %%
############ Area-scaled 20 ms OFF/ON step histograms ############
fig, ax = plt.subplots(figsize=(9.5, 5.2))
audit_rows = []
for led in (0, 1):
    trials = df.loc[df["LED_trial"].eq(led)]
    coded = trials.loc[trials["abort_event"].eq(3)]
    finite = coded.loc[np.isfinite(coded["TotalFixTime"])]
    assert len(trials) == EXPECTED_LED_COUNTS[led]
    assert finite["TotalFixTime"].lt(finite["intended_fix"]).all()
    relative_time = (finite["TotalFixTime"] - finite["LED_onset"]).to_numpy(float)
    counts, _ = np.histogram(relative_time, bins=FULL_BINS)
    assert counts.sum() == len(finite)
    rate = counts / (len(trials) * BIN_WIDTH_S)
    area = float(np.sum(rate * np.diff(FULL_BINS)))
    before_area = float(np.sum(rate[FULL_BINS[1:] <= 0] * BIN_WIDTH_S))
    assert np.isclose(area, len(finite) / len(trials))
    assert np.isclose(before_area, (relative_time < 0).sum() / len(trials))
    label = f"LED {'OFF' if led == 0 else 'ON'}"
    ax.stairs(rate, FULL_BINS, color=COLORS[led], linewidth=1.6, label=label)
    audit_rows.append({
        "LED_trial": led,
        "all_trials": len(trials),
        "coded_event3_aborts": len(coded),
        "finite_plotted_aborts": len(finite),
        "missing_abort_timing": len(coded) - len(finite),
        "aborts_before_juan_onset": int((relative_time < 0).sum()),
        "pre_onset_area": before_area,
        "full_histogram_area": area,
    })

ax.axvline(0, color="0.35", linestyle="--", linewidth=1)
ax.set_xlim(*DISPLAY_RANGE_S)
ax.set_xlabel("Abort time − Juan's LED_onset (s)")
ax.set_ylabel("Abort rate (s⁻¹)")
ax.set_title("LED7 session type 9: Juan's LED_onset alignment")
ax.grid(axis="y", alpha=0.2)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUTPUT_PNG, dpi=250, bbox_inches="tight")
plt.close(fig)

audit = pd.DataFrame(audit_rows)
assert audit["all_trials"].sum() == EXPECTED_ROWS
assert audit["coded_event3_aborts"].sum() == EXPECTED_CODED_ABORTS
audit.to_csv(AUDIT_CSV, index=False)
print(audit.to_string(index=False))
print(f"Figure: {OUTPUT_PNG}")
print(f"Audit: {AUDIT_CSV}")
