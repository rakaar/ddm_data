# %%
"""Pooled LED7 fixation-abort rates by shared OFF/ON scheduled-onset terciles."""

from pathlib import Path
import os


# %%
############ Editable parameters ############
SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = SCRIPT_DIR.parent / "raw_data"
OUTPUT_STEM = "led7_s7_s8_s9_abort_rates_by_onset_tercile_3x3"
FIGURE_PATH = SCRIPT_DIR / f"{OUTPUT_STEM}.png"
AUDIT_PATH = SCRIPT_DIR / f"{OUTPUT_STEM}_audit.csv"
HISTOGRAM_PATH = SCRIPT_DIR / f"{OUTPUT_STEM}_histograms.csv"
SESSION_TYPES = (7, 8, 9)
ANIMALS = (90, 92, 93, 98, 99, 100, 102, 103)
EXPECTED_ROWS = {7: 138_440, 8: 148_428, 9: 100_788}
EXPECTED_LED_COUNTS = {7: (92_057, 46_383), 8: (92_784, 55_644), 9: (61_408, 39_380)}
EXPECTED_ABORTS = {7: 22_494, 8: 22_926, 9: 18_860}
EXPECTED_MISSING_ABORT_TIMES = {7: 3, 8: 2, 9: 6}
EXPECTED_TERCILE_COUNTS = {
    7: (46_147, 46_146, 46_147),
    8: (49_478, 49_474, 49_476),
    9: (33_596, 33_596, 33_596),
}
TERCILE_NAMES = ("Early onset", "Middle onset", "Late onset")
BIN_WIDTH_S = 0.020
FULL_RANGE_S = (-2.0, 2.0)
DISPLAY_RANGE_S = (-1.0, 1.0)
LED_COLORS = {0: "tab:blue", 1: "tab:red"}
LED_LABELS = {0: "OFF", 1: "ON"}
FIGURE_DPI = 250

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

bins = np.linspace(*FULL_RANGE_S, round((FULL_RANGE_S[1] - FULL_RANGE_S[0]) / BIN_WIDTH_S) + 1)
centers = (bins[:-1] + bins[1:]) / 2


# %%
############ Read standardized CSVs and select terciles before aborts ############
datasets = {}
tercile_edges = {}
for session_type in SESSION_TYPES:
    source = RAW_DATA_DIR / f"LED7_s{session_type}.csv"
    frame = pd.read_csv(source, float_precision="round_trip")
    assert len(frame.columns) == 52 and len(frame) == EXPECTED_ROWS[session_type]
    assert frame["training_level"].eq(16).all()
    assert frame["session_type"].eq(session_type).all()
    assert (frame["repeat_trial"].isin([0, 2]) | frame["repeat_trial"].isna()).all()
    # These three exports have no missing LED flags; OFF and ON cover every row.
    assert frame["LED_trial"].isin([0, 1]).all()
    assert tuple(frame["LED_trial"].value_counts().sort_index()) == EXPECTED_LED_COUNTS[session_type]
    assert tuple(sorted(frame["animal"].unique())) == ANIMALS
    assert not frame.duplicated(["animal", "session", "trial"]).any()
    assert np.isfinite(frame[["LED_onset_time", "intended_fix"]]).all().all()
    assert frame["LED_onset_time"].ge(0).all()
    assert frame["LED_onset_time"].le(frame["intended_fix"]).all()

    # LED_onset_time is already fixation referenced in all three CSVs.
    # Use only scheduled onset to define shared bins, pooling OFF and ON.
    # qcut keeps equal onset values together: s8 has a three-row boundary tie.
    labels, edges = pd.qcut(frame["LED_onset_time"], q=3, labels=False, retbins=True)
    assert not labels.isna().any() and np.all(np.diff(edges) > 0)
    assert tuple(labels.value_counts().sort_index()) == EXPECTED_TERCILE_COUNTS[session_type]
    frame["onset_tercile"] = labels.astype(int)
    abort = frame["abort_event"].eq(3)
    finite = np.isfinite(frame["timed_fix"])
    assert int(abort.sum()) == EXPECTED_ABORTS[session_type]
    assert int((abort & ~finite).sum()) == EXPECTED_MISSING_ABORT_TIMES[session_type]
    assert frame.loc[abort & finite, "timed_fix"].lt(frame.loc[abort & finite, "intended_fix"]).all()
    datasets[session_type] = frame
    tercile_edges[session_type] = edges
    print(f"s{session_type}: {len(frame):,} trials; onset edges (s): {edges}", flush=True)


# %%
############ Count and normalize within each session type, tercile, and LED group ############
audit_rows = []
histogram_frames = []
rates = {}
for session_type, frame in datasets.items():
    edges = tercile_edges[session_type]
    for tercile in range(3):
        pool = frame.loc[frame["onset_tercile"].eq(tercile)]
        for led in (0, 1):
            group = pool.loc[pool["LED_trial"].eq(led)]
            aborts = group.loc[group["abort_event"].eq(3)]
            finite_aborts = aborts.loc[np.isfinite(aborts["timed_fix"])]
            relative_times = (finite_aborts["timed_fix"] - finite_aborts["LED_onset_time"]).to_numpy()
            assert len(group) > 0
            counts, _ = np.histogram(relative_times, bins=bins)
            assert counts.sum() == len(relative_times), "Full bins omitted finite aborts."
            height = counts / (len(group) * BIN_WIDTH_S)
            area = float(np.sum(height * np.diff(bins)))
            assert np.isclose(area, len(finite_aborts) / len(group), atol=1e-14, rtol=0)
            rates[session_type, tercile, led] = height
            audit_rows.append({
                "session_type": session_type,
                "tercile": tercile + 1,
                "tercile_name": TERCILE_NAMES[tercile],
                "onset_lower_s": edges[tercile],
                "onset_upper_s": edges[tercile + 1],
                "lower_boundary_inclusive": tercile == 0,
                "upper_boundary_inclusive": True,
                "pooled_tercile_trials": len(pool),
                "LED_trial": led,
                "all_trials": len(group),
                "coded_event3_aborts": len(aborts),
                "finite_plotted_aborts": len(finite_aborts),
                "missing_abort_timing": len(aborts) - len(finite_aborts),
                "coded_abort_fraction": len(aborts) / len(group),
                "finite_abort_fraction": len(finite_aborts) / len(group),
                "full_histogram_area": area,
                "aborts_outside_display": int(((relative_times < DISPLAY_RANGE_S[0]) | (relative_times > DISPLAY_RANGE_S[1])).sum()),
                "aborts_before_scheduled_onset": int((relative_times < 0).sum()),
                "median_intended_fix_s": group["intended_fix"].median(),
                "median_LED_onset_s": group["LED_onset_time"].median(),
            })
            histogram_frames.append(pd.DataFrame({
                "session_type": session_type,
                "tercile": tercile + 1,
                "LED_trial": led,
                "bin_left_s": bins[:-1],
                "bin_right_s": bins[1:],
                "bin_center_s": centers,
                "abort_count": counts,
                "all_trials": len(group),
                "abort_rate_per_s": height,
            }))

audit = pd.DataFrame(audit_rows)
for session_type in SESSION_TYPES:
    rows = audit.loc[audit["session_type"].eq(session_type)]
    assert rows["all_trials"].sum() == EXPECTED_ROWS[session_type]
    assert rows["coded_event3_aborts"].sum() == EXPECTED_ABORTS[session_type]
    assert rows["missing_abort_timing"].sum() == EXPECTED_MISSING_ABORT_TIMES[session_type]
    assert tuple(rows.groupby("LED_trial")["all_trials"].sum()) == EXPECTED_LED_COUNTS[session_type]
audit.to_csv(AUDIT_PATH, index=False)
pd.concat(histogram_frames, ignore_index=True).to_csv(HISTOGRAM_PATH, index=False)


# %%
############ Three session rows by three shared-onset terciles ############
fig, axes = plt.subplots(3, 3, figsize=(15.5, 11.5), sharex=True, sharey=True)
visible_bins = (centers >= DISPLAY_RANGE_S[0]) & (centers <= DISPLAY_RANGE_S[1])
y_max = max(float(y[visible_bins].max()) for y in rates.values())
for row, session_type in enumerate(SESSION_TYPES):
    edges = tercile_edges[session_type]
    for tercile, ax in enumerate(axes[row]):
        panel_audit = audit.loc[audit["session_type"].eq(session_type) & audit["tercile"].eq(tercile + 1)]
        for led in (0, 1):
            ax.stairs(rates[session_type, tercile, led], bins, color=LED_COLORS[led], linewidth=1.35)
            info = panel_audit.loc[panel_audit["LED_trial"].eq(led)].iloc[0]
            ax.text(
                0.035, 0.95 - led * 0.08,
                f"{LED_LABELS[led]}: {int(info.finite_plotted_aborts):,}/{int(info.all_trials):,} = {info.full_histogram_area:.1%}",
                transform=ax.transAxes, va="top", color=LED_COLORS[led], fontsize=9,
            )
        n_pool = int(panel_audit["all_trials"].sum())
        ax.set_title(
            f"{TERCILE_NAMES[tercile]}\n"
            f"{edges[tercile] * 1000:.1f}–{edges[tercile + 1] * 1000:.1f} ms · n = {n_pool:,}",
            fontsize=11,
        )
        ax.axvline(0, color="0.45", linestyle="--", linewidth=0.9, zorder=0)
        ax.set_xlim(DISPLAY_RANGE_S)
        ax.set_ylim(0, y_max * 1.15)
        ax.grid(axis="y", color="0.9", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        if tercile == 0:
            ax.set_ylabel(f"Session type {session_type}\nAbort rate (s$^{{-1}}$)")
        if row == 2:
            ax.set_xlabel("Time from scheduled LED onset (s)")

fig.suptitle("LED7 aggregate: fixation aborts by scheduled LED-onset tercile", fontsize=16, y=0.99)
fig.legend(handles=[
    Line2D([0], [0], color=LED_COLORS[led], linewidth=1.5, label=f"LED {LED_LABELS[led]}")
    for led in (0, 1)
], loc="upper center", bbox_to_anchor=(0.5, 0.958), ncol=2, frameon=False)
fig.text(
    0.5, 0.016,
    "Shared OFF/ON cutoffs use all eligible trials; equal onset values stay together. 20 ms bins.\n"
    "Annotations: finite event-3 aborts / all group trials = full histogram area. OFF onset is scheduled/counterfactual.",
    ha="center", fontsize=9,
)
fig.tight_layout(rect=(0, 0.055, 1, 0.92), h_pad=1.5, w_pad=1.0)
fig.savefig(FIGURE_PATH, dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)
print(audit.to_string(index=False))
print(f"Figure: {FIGURE_PATH}")
print(f"Audit: {AUDIT_PATH}")
print(f"Histogram values: {HISTOGRAM_PATH}")
