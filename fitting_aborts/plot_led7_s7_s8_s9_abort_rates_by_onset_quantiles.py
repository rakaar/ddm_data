# %%
"""Pooled LED7 onset-quantile abort rates, excluding control animals 90/102."""

from pathlib import Path
import os


# %%
############ Editable parameters ############
SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = SCRIPT_DIR.parent / "raw_data"
OUTPUT_PREFIX = "led7_s7_s8_s9_no_controls_abort_rates_by_onset_quantiles"
AUDIT_PATH = SCRIPT_DIR / f"{OUTPUT_PREFIX}_audit.csv"
HISTOGRAM_PATH = SCRIPT_DIR / f"{OUTPUT_PREFIX}_histograms.csv"
SESSION_TYPES = (7, 8, 9)
N_GROUPS = (2, 3, 4, 6)
CONTROL_ANIMALS = (90, 102)
ANIMALS = (92, 93, 98, 99, 100, 103)
EXPECTED_SOURCE_ROWS = {7: 138_440, 8: 148_428, 9: 100_788}
EXPECTED_ROWS = {7: 100_251, 8: 113_941, 9: 74_745}
EXPECTED_LED_COUNTS = {7: (66_226, 34_025), 8: (70_892, 43_049), 9: (45_124, 29_621)}
EXPECTED_ABORTS = {7: 16_389, 8: 17_725, 9: 14_001}
EXPECTED_MISSING_ABORT_TIMES = {7: 2, 8: 2, 9: 5}
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
############ Read standardized onset values and exclude controls before grouping ############
datasets = {}
for session_type in SESSION_TYPES:
    source = RAW_DATA_DIR / f"LED7_s{session_type}.csv"
    frame = pd.read_csv(source, float_precision="round_trip")
    assert len(frame.columns) == 52 and len(frame) == EXPECTED_SOURCE_ROWS[session_type]
    assert frame["training_level"].eq(16).all()
    assert frame["session_type"].eq(session_type).all()
    assert (frame["repeat_trial"].isin([0, 2]) | frame["repeat_trial"].isna()).all()
    assert frame["LED_trial"].isin([0, 1]).all()  # No missing LED flags in these exports.

    frame = frame.loc[~frame["animal"].isin(CONTROL_ANIMALS)].copy()
    assert len(frame) == EXPECTED_ROWS[session_type]
    assert tuple(sorted(frame["animal"].unique())) == ANIMALS
    assert not frame["animal"].isin(CONTROL_ANIMALS).any()
    assert tuple(frame["LED_trial"].value_counts().sort_index()) == EXPECTED_LED_COUNTS[session_type]
    assert not frame.duplicated(["animal", "session", "trial"]).any()
    assert np.isfinite(frame[["LED_onset_time", "intended_fix"]]).all().all()
    assert frame["LED_onset_time"].ge(0).all()
    assert frame["LED_onset_time"].le(frame["intended_fix"]).all()
    abort = frame["abort_event"].eq(3)
    finite = np.isfinite(frame["timed_fix"])
    assert int(abort.sum()) == EXPECTED_ABORTS[session_type]
    assert int((abort & ~finite).sum()) == EXPECTED_MISSING_ABORT_TIMES[session_type]
    assert frame.loc[abort & finite, "timed_fix"].lt(frame.loc[abort & finite, "intended_fix"]).all()

    # These CSVs already use LED onset relative to fixation t=0:
    # s7/s8 were exported as intended_fix - raw MAT LED_onset_time;
    # s9 retains raw MAT LED_onset_time. Do not transform them again.
    datasets[session_type] = frame
    print(f"s{session_type}: {len(frame):,} retained trials; animals {ANIMALS}", flush=True)


# %%
############ Quantiles within each session type, pooling OFF and ON ############
rates = {}
quantile_edges = {}
audit_rows = []
histogram_frames = []
for n_groups in N_GROUPS:
    for session_type, frame in datasets.items():
        # All retained trials determine the boundaries. Aborts are selected later.
        # Identical onset values stay together, including ties at quantile edges.
        labels, edges = pd.qcut(frame["LED_onset_time"], q=n_groups, labels=False, retbins=True)
        assert not labels.isna().any() and np.all(np.diff(edges) > 0)
        sizes = labels.value_counts().sort_index()
        assert len(sizes) == n_groups and int(sizes.sum()) == len(frame)
        assert sizes.max() - sizes.min() <= 4, "Unexpected quantile imbalance; inspect ties."
        quantile_edges[n_groups, session_type] = edges
        print(f"{n_groups} groups, s{session_type}: sizes {sizes.tolist()}, edges (ms) {np.round(edges * 1000, 3)}")

        for group_index in range(n_groups):
            pool = frame.loc[labels.eq(group_index)]
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
                rates[n_groups, session_type, group_index, led] = height
                audit_rows.append({
                    "n_groups": n_groups,
                    "session_type": session_type,
                    "onset_group": group_index + 1,
                    "onset_lower_s": edges[group_index],
                    "onset_upper_s": edges[group_index + 1],
                    "lower_boundary_inclusive": group_index == 0,
                    "upper_boundary_inclusive": True,
                    "pooled_group_trials": len(pool),
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
                    "included_animals": ",".join(map(str, ANIMALS)),
                    "excluded_controls": ",".join(map(str, CONTROL_ANIMALS)),
                })
                histogram_frames.append(pd.DataFrame({
                    "n_groups": n_groups,
                    "session_type": session_type,
                    "onset_group": group_index + 1,
                    "LED_trial": led,
                    "bin_left_s": bins[:-1],
                    "bin_right_s": bins[1:],
                    "bin_center_s": centers,
                    "abort_count": counts,
                    "all_trials": len(group),
                    "abort_rate_per_s": height,
                }))

audit = pd.DataFrame(audit_rows)
for n_groups in N_GROUPS:
    for session_type in SESSION_TYPES:
        rows = audit.loc[audit["n_groups"].eq(n_groups) & audit["session_type"].eq(session_type)]
        assert rows["all_trials"].sum() == EXPECTED_ROWS[session_type]
        assert rows["coded_event3_aborts"].sum() == EXPECTED_ABORTS[session_type]
        assert rows["missing_abort_timing"].sum() == EXPECTED_MISSING_ABORT_TIMES[session_type]
        assert tuple(rows.groupby("LED_trial")["all_trials"].sum()) == EXPECTED_LED_COUNTS[session_type]
audit.to_csv(AUDIT_PATH, index=False)
pd.concat(histogram_frames, ignore_index=True).to_csv(HISTOGRAM_PATH, index=False)


# %%
############ Four figures with identical time and rate axes ############
visible_bins = (centers >= DISPLAY_RANGE_S[0]) & (centers <= DISPLAY_RANGE_S[1])
y_max = max(float(height[visible_bins].max()) for height in rates.values())
for n_groups in N_GROUPS:
    fig, axes = plt.subplots(3, n_groups, figsize=(5.0 * n_groups + 0.5, 11.5), sharex=True, sharey=True)
    for row, session_type in enumerate(SESSION_TYPES):
        edges = quantile_edges[n_groups, session_type]
        for group_index, ax in enumerate(axes[row]):
            panel = audit.loc[
                audit["n_groups"].eq(n_groups) & audit["session_type"].eq(session_type)
                & audit["onset_group"].eq(group_index + 1)
            ]
            for led in (0, 1):
                ax.stairs(rates[n_groups, session_type, group_index, led], bins, color=LED_COLORS[led], linewidth=1.35)
                info = panel.loc[panel["LED_trial"].eq(led)].iloc[0]
                ax.text(
                    0.035, 0.95 - led * 0.08,
                    f"{LED_LABELS[led]}: {int(info.finite_plotted_aborts):,}/{int(info.all_trials):,} = {info.full_histogram_area:.1%}",
                    transform=ax.transAxes, va="top", color=LED_COLORS[led], fontsize=9,
                )
            if n_groups == 2:
                name = ("Early onset", "Late onset")[group_index]
            elif n_groups == 3:
                name = ("Early onset", "Middle onset", "Late onset")[group_index]
            else:
                name = f"Onset group {group_index + 1}/{n_groups}"
            ax.set_title(
                f"{name}\n{edges[group_index] * 1000:.1f}–{edges[group_index + 1] * 1000:.1f} ms"
                f" · n = {int(panel['all_trials'].sum()):,}", fontsize=11,
            )
            ax.axvline(0, color="0.45", linestyle="--", linewidth=0.9, zorder=0)
            ax.set_xlim(DISPLAY_RANGE_S)
            ax.set_ylim(0, y_max * 1.15)
            ax.grid(axis="y", color="0.9", linewidth=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            if group_index == 0:
                ax.set_ylabel(f"Session type {session_type}\nAbort rate (s$^{{-1}}$)")
            if row == 2:
                ax.set_xlabel("Time from scheduled LED onset (s)")

    fig.suptitle(f"LED7 aggregate: {n_groups} LED-onset groups · controls 90 and 102 excluded", fontsize=16, y=0.99)
    fig.legend(handles=[
        Line2D([0], [0], color=LED_COLORS[led], linewidth=1.5, label=f"LED {LED_LABELS[led]}")
        for led in (0, 1)
    ], loc="upper center", bbox_to_anchor=(0.5, 0.958), ncol=2, frameon=False)
    fig.text(
        0.5, 0.016,
        "Onset groups use fixation-referenced LED_onset_time, separately within each session type, with shared OFF/ON cutoffs.\n"
        "Annotations: finite event-3 aborts / all group trials = full histogram area. 20 ms bins; OFF onset is scheduled/counterfactual.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.92), h_pad=1.5, w_pad=1.0)
    figure_path = SCRIPT_DIR / f"{OUTPUT_PREFIX}_3x{n_groups}.png"
    fig.savefig(figure_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure: {figure_path}", flush=True)

print(audit[["n_groups", "session_type", "onset_group", "LED_trial", "all_trials", "finite_plotted_aborts", "missing_abort_timing", "full_histogram_area"]].to_string(index=False))
print(f"Audit: {AUDIT_PATH}")
print(f"Histogram values: {HISTOGRAM_PATH}")
