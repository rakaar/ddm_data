# %%
"""Compare two raw session-9 onset coordinates before 2 December 2024."""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance


# %%
############ Editable parameters ############
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
JUAN_CSV = REPO_ROOT / "juan_LED7_csvs_sess_type" / "outLED7_sess_type_9.csv"
OUR_CSV = REPO_ROOT / "raw_data" / "LED7_s9.csv"
OUTPUT_PNG = SCRIPT_DIR / "led7_s9_pre_dec_raw_vs_derived_onset_1x2.png"
ABORT_OUTPUT_PNG = SCRIPT_DIR / "led7_s9_pre_dec_abort_rate_two_alignments_1x2.png"
ABORT_AUDIT_CSV = SCRIPT_DIR / "led7_s9_pre_dec_abort_rate_two_alignments_audit.csv"
DATE_CUTOFF = pd.Timestamp("2024-12-02")
ANIMALS = (92, 93, 98, 99, 100, 103)
EXPECTED_ROWS = 16_511
BIN_WIDTH_S = 0.020
BINS = np.arange(0, 1.2 + BIN_WIDTH_S / 2, BIN_WIDTH_S)


# %%
############ Select pre-cutoff trials and check the two data sources ############
juan = pd.read_csv(JUAN_CSV, float_precision="round_trip")
ours = pd.read_csv(OUR_CSV, float_precision="round_trip")
key = ["animal", "session", "trial"]
assert not juan.duplicated(key).any() and not ours.duplicated(key).any()
assert juan["training_level"].eq(16).all() and juan["session_type"].eq(9).all()
assert (juan["repeat_trial"].isin([0, 2]) | juan["repeat_trial"].isna()).all()
assert juan["LED_trial"].isin([0, 1]).all()

juan["date_time_start"] = pd.to_datetime(
    juan["date_time_start"], format="%d-%b-%Y %H:%M:%S.%f", errors="coerce"
)
pre = juan.loc[juan["date_time_start"] < DATE_CUTOFF].copy()
assert len(pre) == EXPECTED_ROWS
assert tuple(sorted(pre["animal"].unique())) == ANIMALS
assert pre["date_time_start"].notna().all()

source = pre[key + ["intended_fix", "LED_onset_time", "LED_duration", "LED_onset"]].merge(
    ours[key + ["intended_fix", "LED_onset_time", "LED_duration"]],
    on=key, how="left", validate="one_to_one", indicator=True,
    suffixes=("_juan", "_ours"),
)
assert source["_merge"].eq("both").all()
for col in ("intended_fix", "LED_onset_time", "LED_duration"):
    assert np.allclose(source[f"{col}_juan"], source[f"{col}_ours"], rtol=0, atol=1e-9)

raw = source["LED_onset_time_ours"].to_numpy(float)
derived = (source["intended_fix_ours"] - source["LED_onset_time_ours"]).to_numpy(float)
assert np.allclose(source["LED_onset"], derived, rtol=0, atol=1e-9)
assert np.allclose(source["LED_duration_ours"], 1 + raw, rtol=0, atol=1e-9)
assert np.isfinite(raw).all() and np.isfinite(derived).all()
assert ((raw >= BINS[0]) & (raw <= BINS[-1])).all()
assert ((derived >= BINS[0]) & (derived <= BINS[-1])).all()


# %%
############ Side-by-side unit-area histograms ############
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, sharey=True)
for ax, values, title, color in zip(
    axes,
    (raw, derived),
    ("Raw LED_onset_time", "intended_fix − raw LED_onset_time"),
    ("tab:blue", "tab:orange"),
):
    counts, _ = np.histogram(values, bins=BINS)
    assert counts.sum() == len(values)
    density = counts / (len(values) * BIN_WIDTH_S)
    assert np.isclose(np.sum(density * np.diff(BINS)), 1)
    ax.stairs(density, BINS, color=color, linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.grid(axis="y", alpha=0.2)
    ax.text(0.97, 0.95, f"n = {len(values):,}\nmedian = {np.median(values):.3f} s",
            transform=ax.transAxes, ha="right", va="top")
axes[0].set_ylabel("Probability density (s⁻¹)")
axes[0].set_xlim(BINS[0], BINS[-1])
fig.suptitle("LED7 session type 9, before 2 Dec 2024", fontsize=14)
fig.tight_layout()
fig.savefig(OUTPUT_PNG, dpi=250, bbox_inches="tight")
plt.close(fig)

ks = ks_2samp(raw, derived).statistic
w_distance_ms = 1000 * wasserstein_distance(raw, derived)
print(f"Pre-cutoff rows: {len(pre):,}; animals: {ANIMALS}")
print(f"Raw median: {np.median(raw):.6f} s; derived median: {np.median(derived):.6f} s")
print(f"Marginal KS: {ks:.6f}; Wasserstein: {w_distance_ms:.3f} ms")
print(f"Figure: {OUTPUT_PNG}")


# %%
############ Compare OFF/ON fixation-abort rates under each alignment ############
abort_bins = np.round(np.arange(-1.2, 1.2 + BIN_WIDTH_S / 2, BIN_WIDTH_S), 10)
alignments = (
    ("Raw LED_onset_time", lambda frame: frame["LED_onset_time"]),
    ("intended_fix − raw LED_onset_time",
     lambda frame: frame["intended_fix"] - frame["LED_onset_time"]),
)
colors = {0: "tab:blue", 1: "tab:red"}
labels = {0: "LED OFF", 1: "LED ON"}
fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharex=True, sharey=True)
audit_rows = []
for ax, (title, onset_function) in zip(axes, alignments):
    for led in (0, 1):
        group = pre.loc[pre["LED_trial"].eq(led)]
        coded = group.loc[group["abort_event"].eq(3)]
        finite = coded.loc[np.isfinite(coded["TotalFixTime"])]
        assert finite["TotalFixTime"].lt(finite["intended_fix"]).all()
        times = (finite["TotalFixTime"] - onset_function(finite)).to_numpy(float)
        counts, _ = np.histogram(times, bins=abort_bins)
        assert counts.sum() == len(finite)
        heights = counts / (len(group) * BIN_WIDTH_S)
        full_area = float(np.sum(heights * np.diff(abort_bins)))
        before_area = float(np.sum(heights[abort_bins[1:] <= 0] * BIN_WIDTH_S))
        assert np.isclose(full_area, len(finite) / len(group))
        assert np.isclose(before_area, (times < 0).sum() / len(group))
        ax.stairs(heights, abort_bins, color=colors[led], linewidth=1.6, label=labels[led])
        audit_rows.append({
            "alignment": title,
            "LED_trial": led,
            "all_trials": len(group),
            "coded_event3_aborts": len(coded),
            "finite_plotted_aborts": len(finite),
            "missing_abort_timing": len(coded) - len(finite),
            "aborts_before_candidate_onset": int((times < 0).sum()),
            "pre_onset_area": before_area,
            "full_histogram_area": full_area,
        })
    ax.axvline(0, color="0.35", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Abort time relative to candidate onset (s)")
    ax.grid(axis="y", alpha=0.2)
axes[0].set_ylabel("Abort rate (s⁻¹)")
axes[0].set_xlim(-0.5, 0.6)
axes[1].legend(frameon=False, loc="upper right")
fig.suptitle("LED7 session type 9: fixation aborts before 2 Dec 2024", fontsize=14)
fig.tight_layout()
fig.savefig(ABORT_OUTPUT_PNG, dpi=250, bbox_inches="tight")
plt.close(fig)

audit = pd.DataFrame(audit_rows)
audit.to_csv(ABORT_AUDIT_CSV, index=False)
print(audit.to_string(index=False))
print(f"Abort-rate figure: {ABORT_OUTPUT_PNG}")
