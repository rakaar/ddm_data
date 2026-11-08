# %%
"""Current LED-ON abort timing by exposure during the previous LED-ON abort.

Save both equal-animal mean +/- SEM and trial-pooled views. In either view,
each group uses current LED-ON event-3-or-valid trials as its denominator.
The standardized CSV's LED_onset_time is already fixation referenced.
"""

from pathlib import Path
import os


# %%
############ Editable parameters ############
SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
OUTPUT_STEM = SCRIPT_DIR / "led7_s7_s8_previous_led_on_abort_exposure"
ANIMALS = (92, 93, 98, 99, 100, 103)
SESSION_TYPES = (7, 8)
BIN_WIDTH_S = 0.020
FULL_TIME_RANGE_S = (-2.0, 2.0)
DISPLAY_TIME_RANGE_S = (-0.4, 0.5)
FIGURE_DPI = 240
GROUPS = (
    ("previous_abort_before_LED", "Previous abort before LED onset", "black"),
    ("previous_abort_after_LED", "Previous abort after LED onset", "tab:red"),
)
EXPECTED_COUNTS = {
    7: {"previous_abort_before_LED": (722, 3105),
        "previous_abort_after_LED": (639, 2908)},
    8: {"previous_abort_before_LED": (800, 3372),
        "previous_abort_after_LED": (1136, 4709)},
}
EXPECTED_MISSING_CURRENT_TIMING = {
    7: {"previous_abort_before_LED": 0, "previous_abort_after_LED": 0},
    8: {"previous_abort_before_LED": 1, "previous_abort_after_LED": 1},
}

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# %%
############ Select genuinely adjacent pairs and classify previous exposure ############
bins = np.linspace(*FULL_TIME_RANGE_S,
                   int(round((FULL_TIME_RANGE_S[1] - FULL_TIME_RANGE_S[0]) / BIN_WIDTH_S)) + 1)
np.testing.assert_allclose(np.diff(bins), BIN_WIDTH_S, rtol=0, atol=1e-14)
curves = np.empty((2, 2, len(ANIMALS), len(bins) - 1))
bin_counts = np.zeros((2, 2, len(ANIMALS), len(bins) - 1), dtype=int)
denominators = np.zeros((2, 2, len(ANIMALS)), dtype=int)
audit_records = []

for session_index, session_type in enumerate(SESSION_TYPES):
    source_path = REPO / f"raw_data/LED7_s{session_type}.csv"
    source = pd.read_csv(source_path, float_precision="round_trip")
    assert len(source.columns) == 52
    assert source.training_level.eq(16).all()
    assert source.session_type.eq(session_type).all()
    assert (source.repeat_trial.isin([0, 2]) | source.repeat_trial.isna()).all()
    assert source.LED_trial.isin([0, 1]).all()

    trials = source.loc[source.animal.isin(ANIMALS)].copy()
    trials.sort_values(["animal", "session", "trial"], kind="stable", inplace=True)
    assert not trials.duplicated(["animal", "session", "trial"]).any()
    previous = trials.groupby(["animal", "session"], sort=False)[
        ["trial", "LED_trial", "abort_event", "timed_fix", "LED_onset_time"]
    ].shift(1)
    adjacent = previous.trial.eq(trials.trial - 1)
    previous_on_abort = adjacent & previous.LED_trial.eq(1) & previous.abort_event.eq(3)
    current_on_eligible = (trials.LED_trial.eq(1)
                           & (trials.abort_event.eq(3) | trials.success.isin([-1, 1])))
    pair_eligible = previous_on_abort & current_on_eligible
    previous_relative_time = previous.timed_fix - previous.LED_onset_time
    assert int((pair_eligible & previous_relative_time.isna()).sum()) == 0
    assert int((pair_eligible & previous_relative_time.eq(0)).sum()) == 0
    group_masks = {
        GROUPS[0][0]: pair_eligible & previous_relative_time.lt(0),
        GROUPS[1][0]: pair_eligible & previous_relative_time.gt(0),
    }
    assert int(group_masks[GROUPS[0][0]].sum() + group_masks[GROUPS[1][0]].sum()) == int(pair_eligible.sum())
    assert not (group_masks[GROUPS[0][0]] & group_masks[GROUPS[1][0]]).any()

    for group_index, (group_name, _, _) in enumerate(GROUPS):
        grouped = trials.loc[group_masks[group_name]]
        selected = grouped.loc[grouped.abort_event.eq(3)]
        expected_abort, expected_denominator = EXPECTED_COUNTS[session_type][group_name]
        assert (len(selected), len(grouped)) == (expected_abort, expected_denominator)
        missing = int(selected.timed_fix.isna().sum())
        assert missing == EXPECTED_MISSING_CURRENT_TIMING[session_type][group_name]
        assert not (grouped.abort_event.eq(3) & grouped.success.isin([-1, 1])).any()

        for animal_index, animal in enumerate(ANIMALS):
            animal_group = grouped.loc[grouped.animal.eq(animal)]
            animal_aborts = animal_group.loc[animal_group.abort_event.eq(3)]
            relative_time = (animal_aborts.timed_fix - animal_aborts.LED_onset_time).dropna()
            assert len(animal_group) > 0
            assert np.isfinite(relative_time).all()
            counts = np.histogram(relative_time, bins=bins)[0]
            assert int(counts.sum()) == len(relative_time)
            rate = counts / (len(animal_group) * BIN_WIDTH_S)
            bin_counts[session_index, group_index, animal_index] = counts
            denominators[session_index, group_index, animal_index] = len(animal_group)
            curves[session_index, group_index, animal_index] = rate
            area = float(rate @ np.diff(bins))
            expected_area = len(relative_time) / len(animal_group)
            assert np.isclose(area, expected_area, rtol=0, atol=1e-12)
            audit_records.append({
                "source_csv": str(source_path.relative_to(REPO)),
                "session_type": session_type,
                "animal": animal,
                "previous_exposure_group": group_name,
                "eligible_current_LED_ON_trials": len(animal_group),
                "current_event3_aborts": len(animal_aborts),
                "current_valid_trials": int(animal_group.success.isin([-1, 1]).sum()),
                "missing_current_timed_fix": int(animal_aborts.timed_fix.isna().sum()),
                "plotted_current_aborts": len(relative_time),
                "full_histogram_area": area,
                "expected_area": expected_area,
            })

audit = pd.DataFrame(audit_records)
assert len(audit) == len(SESSION_TYPES) * len(GROUPS) * len(ANIMALS)
for session_type in SESSION_TYPES:
    for group_name, _, _ in GROUPS:
        rows = audit.loc[audit.session_type.eq(session_type)
                         & audit.previous_exposure_group.eq(group_name)]
        assert len(rows) == 6
        assert (int(rows.current_event3_aborts.sum()),
                int(rows.eligible_current_LED_ON_trials.sum())) == EXPECTED_COUNTS[session_type][group_name]
        assert int(rows.missing_current_timed_fix.sum()) == EXPECTED_MISSING_CURRENT_TIMING[session_type][group_name]
        assert (rows.current_event3_aborts + rows.current_valid_trials
                == rows.eligible_current_LED_ON_trials).all()


# %%
############ Equal-animal means and the two-panel figure ############
means = curves.mean(axis=2)
sems = curves.std(axis=2, ddof=1) / np.sqrt(len(ANIMALS))
assert np.isfinite(means).all() and np.isfinite(sems).all()
for session_index, session_type in enumerate(SESSION_TYPES):
    for group_index, (group_name, _, _) in enumerate(GROUPS):
        rows = audit.loc[audit.session_type.eq(session_type)
                         & audit.previous_exposure_group.eq(group_name)]
        mean_area = float(means[session_index, group_index] @ np.diff(bins))
        assert np.isclose(mean_area, rows.full_histogram_area.mean(), rtol=0, atol=1e-12)
        assert np.isclose(mean_area, rows.expected_area.mean(), rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            sems[session_index, group_index],
            np.sqrt(((curves[session_index, group_index]
                      - means[session_index, group_index]) ** 2).sum(axis=0)
                    / (len(ANIMALS) - 1) / len(ANIMALS)),
            rtol=0, atol=1e-14,
        )

fig, axes = plt.subplots(1, 2, figsize=(13, 5.1), sharex=True, sharey=True)
visible = (bins[:-1] < DISPLAY_TIME_RANGE_S[1]) & (bins[1:] > DISPLAY_TIME_RANGE_S[0])
ymax = 1.17 * float((means + sems)[:, :, visible].max())
for session_index, session_type in enumerate(SESSION_TYPES):
    ax = axes[session_index]
    for group_index, (group_name, label, color) in enumerate(GROUPS):
        mean = means[session_index, group_index]
        sem = sems[session_index, group_index]
        ax.fill_between(bins, np.r_[np.maximum(mean - sem, 0), max(mean[-1] - sem[-1], 0)],
                        np.r_[mean + sem, mean[-1] + sem[-1]], step="post",
                        color=color, alpha=0.15, linewidth=0)
        ax.stairs(mean, bins, color=color, linewidth=1.8, label=label)
        rows = audit.loc[audit.session_type.eq(session_type)
                         & audit.previous_exposure_group.eq(group_name)]
        ax.text(0.98, 0.97 - 0.09 * group_index,
                f"{'Black' if group_index == 0 else 'Red'}: "
                f"n={int(rows.plotted_current_aborts.sum()):,}, "
                f"N={int(rows.eligible_current_LED_ON_trials.sum()):,}, "
                f"area={rows.full_histogram_area.mean():.3f}",
                ha="right", va="top", transform=ax.transAxes, color=color, fontsize=9)
    ax.axvline(0, linestyle=":", color="0.4", linewidth=1)
    ax.set(xlim=DISPLAY_TIME_RANGE_S, ylim=(0, ymax),
           title=f"Session type {session_type}",
           xlabel="Current abort time from current LED onset (s)")
    ax.grid(axis="y", linewidth=0.5, color="0.9")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Conditional abort rate (s⁻¹)")
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2,
           bbox_to_anchor=(0.5, 0.985), frameon=False)
fig.suptitle("LED7 current LED-ON aborts by previous LED-ON abort exposure", y=1.07,
             fontsize=15)
fig.text(0.5, -0.015,
         "Equal-animal mean ± SEM across animals 92, 93, 98, 99, 100, 103. "
         "N: current LED-ON event-3 or valid trials immediately after a previous LED-ON event-3 abort\n"
         "in that exposure group. The full area is the mean animal-level abort fraction, not pooled n/N. "
         "One s8 abort per color lacks timed_fix and remains in N.\n"
         "No matching or subsampling. Previous exposure is defined using observed abort time; "
         "this comparison is descriptive, not causal.",
         ha="center", fontsize=9, linespacing=1.35)
fig.tight_layout(rect=(0, 0.06, 1, 0.94), w_pad=1.8)
figure_path = Path(f"{OUTPUT_STEM}_1x2.png")
fig.savefig(figure_path, dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)

audit_path = Path(f"{OUTPUT_STEM}_count_area_audit.csv")
audit.to_csv(audit_path, index=False)


# %%
############ Trial-pooled version: sum aborts and eligible trials first ############
pooled_counts = bin_counts.sum(axis=2)
pooled_denominators = denominators.sum(axis=2)
pooled_rates = pooled_counts / (pooled_denominators[:, :, None] * BIN_WIDTH_S)
pooled_records = []
for session_index, session_type in enumerate(SESSION_TYPES):
    for group_index, (group_name, _, _) in enumerate(GROUPS):
        rows = audit.loc[audit.session_type.eq(session_type)
                         & audit.previous_exposure_group.eq(group_name)]
        count = int(pooled_counts[session_index, group_index].sum())
        denominator = int(pooled_denominators[session_index, group_index])
        assert count == int(rows.plotted_current_aborts.sum())
        assert denominator == int(rows.eligible_current_LED_ON_trials.sum())
        rate = pooled_rates[session_index, group_index]
        np.testing.assert_allclose(
            rate,
            (curves[session_index, group_index] * denominators[session_index, group_index, :, None]
             ).sum(axis=0) / denominator,
            rtol=0, atol=1e-14,
        )
        area = float(rate @ np.diff(bins))
        assert np.isclose(area, count / denominator, rtol=0, atol=1e-12)
        pooled_records.append({
            "session_type": session_type,
            "previous_exposure_group": group_name,
            "animals": ",".join(map(str, ANIMALS)),
            "eligible_current_LED_ON_trials": denominator,
            "current_event3_aborts": int(rows.current_event3_aborts.sum()),
            "current_valid_trials": int(rows.current_valid_trials.sum()),
            "missing_current_timed_fix": int(rows.missing_current_timed_fix.sum()),
            "plotted_current_aborts": count,
            "full_histogram_area": area,
            "expected_area": count / denominator,
        })
pooled_audit = pd.DataFrame(pooled_records)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.1), sharex=True, sharey=True)
pooled_ymax = 1.17 * float(pooled_rates[:, :, visible].max())
for session_index, session_type in enumerate(SESSION_TYPES):
    ax = axes[session_index]
    for group_index, (group_name, label, color) in enumerate(GROUPS):
        rate = pooled_rates[session_index, group_index]
        ax.stairs(rate, bins, color=color, linewidth=1.8, label=label)
        row = pooled_audit.loc[pooled_audit.session_type.eq(session_type)
                               & pooled_audit.previous_exposure_group.eq(group_name)].iloc[0]
        ax.text(0.98, 0.97 - 0.09 * group_index,
                f"{'Black' if group_index == 0 else 'Red'}: "
                f"n={int(row.plotted_current_aborts):,}, "
                f"N={int(row.eligible_current_LED_ON_trials):,}, "
                f"area={row.full_histogram_area:.3f}",
                ha="right", va="top", transform=ax.transAxes, color=color, fontsize=9)
    ax.axvline(0, linestyle=":", color="0.4", linewidth=1)
    ax.set(xlim=DISPLAY_TIME_RANGE_S, ylim=(0, pooled_ymax),
           title=f"Session type {session_type}",
           xlabel="Current abort time from current LED onset (s)")
    ax.grid(axis="y", linewidth=0.5, color="0.9")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Conditional abort rate (s⁻¹)")
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2,
           bbox_to_anchor=(0.5, 0.985), frameon=False)
fig.suptitle("LED7 current LED-ON aborts by previous LED-ON abort exposure: trial pooled",
             y=1.07, fontsize=15)
fig.text(0.5, -0.015,
         "Trial-pooled animals 92, 93, 98, 99, 100, 103 (controls 90 and 102 excluded). "
         "N: all current LED-ON event-3 or valid trials immediately after a previous LED-ON event-3 abort\n"
         "in that exposure group. Each full-histogram area is plotted aborts / N. "
         "One s8 abort per color lacks timed_fix and remains in N.\n"
         "No matching, subsampling, or error band. Previous exposure uses observed abort time; "
         "this comparison is descriptive, not causal.",
         ha="center", fontsize=9, linespacing=1.35)
fig.tight_layout(rect=(0, 0.06, 1, 0.94), w_pad=1.8)
pooled_figure_path = Path(f"{OUTPUT_STEM}_pooled_1x2.png")
fig.savefig(pooled_figure_path, dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)

pooled_audit_path = Path(f"{OUTPUT_STEM}_pooled_count_area_audit.csv")
pooled_audit.to_csv(pooled_audit_path, index=False)
print(audit.groupby(["session_type", "previous_exposure_group"], sort=False)[
    ["current_event3_aborts", "eligible_current_LED_ON_trials",
     "missing_current_timed_fix", "plotted_current_aborts"]
].sum().to_string())
print(f"Saved {figure_path}, {pooled_figure_path}, and their count/area audits")
