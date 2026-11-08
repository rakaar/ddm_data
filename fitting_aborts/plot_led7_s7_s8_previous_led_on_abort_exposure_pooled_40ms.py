# %%
"""Trial-pooled current LED-ON abort rates by previous LED-ON abort exposure.

This is the separate 40 ms-bin version of the existing 20 ms pooled figure.
The standardized CSV's LED_onset_time is already fixation referenced.
"""

from pathlib import Path
import os


# %%
############ Editable parameters ############
SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
OUTPUT_STEM = SCRIPT_DIR / "led7_s7_s8_previous_led_on_abort_exposure_pooled_40ms"
ANIMALS = (92, 93, 98, 99, 100, 103)
SESSION_TYPES = (7, 8)
BIN_WIDTH_S = 0.040
FULL_TIME_RANGE_S = (-2.0, 2.0)
DISPLAY_TIME_RANGE_S = (-0.4, 0.5)
FIGURE_DPI = 240
GROUPS = (
    ("previous_abort_before_LED", "Previous abort before LED onset", "black"),
    ("previous_abort_after_LED", "Previous abort after LED onset", "tab:red"),
)
EXPECTED = {
    7: {"previous_abort_before_LED": (722, 3105, 0),
        "previous_abort_after_LED": (639, 2908, 0)},
    8: {"previous_abort_before_LED": (800, 3372, 1),
        "previous_abort_after_LED": (1136, 4709, 1)},
}  # coded current aborts, eligible current trials, missing current timed_fix

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# %%
############ Select adjacent pairs and calculate pooled 40 ms rates ############
bins = np.linspace(*FULL_TIME_RANGE_S,
                   int(round((FULL_TIME_RANGE_S[1] - FULL_TIME_RANGE_S[0]) / BIN_WIDTH_S)) + 1)
np.testing.assert_allclose(np.diff(bins), BIN_WIDTH_S, rtol=0, atol=1e-14)
rates = np.empty((2, 2, len(bins) - 1))
records = []

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
    eligible = (previous.trial.eq(trials.trial - 1)
                & previous.LED_trial.eq(1)
                & previous.abort_event.eq(3)
                & trials.LED_trial.eq(1)
                & (trials.abort_event.eq(3) | trials.success.isin([-1, 1])))
    previous_relative_time = previous.timed_fix - previous.LED_onset_time
    assert int((eligible & previous_relative_time.isna()).sum()) == 0
    assert int((eligible & previous_relative_time.eq(0)).sum()) == 0
    group_masks = (
        eligible & previous_relative_time.lt(0),
        eligible & previous_relative_time.gt(0),
    )
    assert int(group_masks[0].sum() + group_masks[1].sum()) == int(eligible.sum())
    assert not (group_masks[0] & group_masks[1]).any()

    for group_index, (group_name, _, _) in enumerate(GROUPS):
        group = trials.loc[group_masks[group_index]]
        aborts = group.loc[group.abort_event.eq(3)]
        missing = int(aborts.timed_fix.isna().sum())
        assert (len(aborts), len(group), missing) == EXPECTED[session_type][group_name]
        assert not (group.abort_event.eq(3) & group.success.isin([-1, 1])).any()
        relative_time = (aborts.timed_fix - aborts.LED_onset_time).dropna()
        assert np.isfinite(relative_time).all()
        counts = np.histogram(relative_time, bins=bins)[0]
        assert int(counts.sum()) == len(relative_time)
        rate = counts / (len(group) * BIN_WIDTH_S)
        rates[session_index, group_index] = rate
        area = float(rate @ np.diff(bins))
        assert np.isclose(area, len(relative_time) / len(group), rtol=0, atol=1e-12)
        records.append({
            "source_csv": str(source_path.relative_to(REPO)),
            "session_type": session_type,
            "previous_exposure_group": group_name,
            "animals": ",".join(map(str, ANIMALS)),
            "bin_width_s": BIN_WIDTH_S,
            "eligible_current_LED_ON_trials": len(group),
            "current_event3_aborts": len(aborts),
            "current_valid_trials": int(group.success.isin([-1, 1]).sum()),
            "missing_current_timed_fix": missing,
            "plotted_current_aborts": len(relative_time),
            "full_histogram_area": area,
        })

audit = pd.DataFrame(records)
assert len(audit) == len(SESSION_TYPES) * len(GROUPS)
assert (audit.current_event3_aborts + audit.current_valid_trials
        == audit.eligible_current_LED_ON_trials).all()
assert np.isfinite(rates).all() and (rates >= 0).all()


# %%
############ Two-panel figure ############
fig, axes = plt.subplots(1, 2, figsize=(13, 5.1), sharex=True, sharey=True)
visible = (bins[:-1] < DISPLAY_TIME_RANGE_S[1]) & (bins[1:] > DISPLAY_TIME_RANGE_S[0])
ymax = 1.32 * float(rates[:, :, visible].max())
for session_index, session_type in enumerate(SESSION_TYPES):
    ax = axes[session_index]
    for group_index, (group_name, label, color) in enumerate(GROUPS):
        ax.stairs(rates[session_index, group_index], bins, color=color,
                  linewidth=1.8, label=label)
        record = audit.loc[audit.session_type.eq(session_type)
                           & audit.previous_exposure_group.eq(group_name)].iloc[0]
        ax.text(0.98, 0.97 - 0.09 * group_index,
                f"{'Black' if group_index == 0 else 'Red'}: "
                f"n={int(record.plotted_current_aborts):,}, "
                f"N={int(record.eligible_current_LED_ON_trials):,}, "
                f"area={record.full_histogram_area:.3f}",
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
fig.suptitle("LED7 current LED-ON aborts by previous LED-ON abort exposure: 40 ms bins",
             y=1.07, fontsize=15)
fig.text(0.5, -0.015,
         "Trial-pooled animals 92, 93, 98, 99, 100, 103; controls 90 and 102 excluded. "
         "Each 40 ms bin is divided by all eligible current LED-ON event-3-or-valid trials\n"
         "following a previous LED-ON event-3 abort in that exposure group and by 0.040 s. "
         "One s8 abort per color lacks timed_fix and remains in N.\n"
         "Same trials as the 20 ms version; no matching or subsampling. "
         "Previous exposure uses observed abort time, so this is descriptive, not causal.",
         ha="center", fontsize=9, linespacing=1.35)
fig.tight_layout(rect=(0, 0.06, 1, 0.94), w_pad=1.8)
figure_path = Path(f"{OUTPUT_STEM}_1x2.png")
fig.savefig(figure_path, dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)

audit_path = Path(f"{OUTPUT_STEM}_count_area_audit.csv")
audit.to_csv(audit_path, index=False)
print(audit[["session_type", "previous_exposure_group", "plotted_current_aborts",
             "eligible_current_LED_ON_trials", "missing_current_timed_fix",
             "full_histogram_area"]].to_string(index=False))
print(f"Saved {figure_path} and {audit_path}")
