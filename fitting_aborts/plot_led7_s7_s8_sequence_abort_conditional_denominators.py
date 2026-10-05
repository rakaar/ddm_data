# %%
"""Four-row LED7 sequence-abort figure with conditional rate denominators.

This is a focused view of the audited six-row conditional analysis. A is an
event-3 abort and V is a valid trial (success +/-1). For a current trial,
the rate denominators are AA+AV after an abort and VAV+VVV between valid
neighbors. The LED flag belongs to the current trial, not its neighbors.
"""

from pathlib import Path
import os


# %%
############ Editable parameters ############
SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
SOURCE_STEM = SCRIPT_DIR / "led7_s7_s8_sequence_abort_valid_neighbor_conditioning"
TOP_MANIFEST_STEM = SCRIPT_DIR / "led7_s7_s8_sequence_abort_pre_onset_matched_summary"
OUTPUT_STEM = SCRIPT_DIR / "led7_s7_s8_sequence_abort_conditional_denominators"
ANIMALS = (92, 93, 98, 99, 100, 103)
CASES = ("s7_after_abort", "s8_after_abort", "s7_isolated", "s8_isolated")
METRICS = ("intended_fix", "LED_onset_time", "abort_from_LED_onset", "matched_abort_from_LED_onset")
RATE_VIEW_S = (-0.4, 0.5)
MAX_PRE_BIN_RATE_DIFFERENCE = 0.005  # s^-1, within each animal.
MAX_PRE_FRACTION_DIFFERENCE = 0.001
FIGURE_DPI = 240
COLORS = {0: "tab:blue", 1: "tab:red"}
EXPECTED = {
    "s7_after_abort": (3097, 15618, 1674, 14195),
    "s8_after_abort": (3426, 16852, 2287, 15713),
    "s7_isolated": (9863, 62238, 8612, 60987),
    "s8_isolated": (10511, 71855, 9107, 70451),
}  # Original selected/N, then pre-onset-matched selected/N.

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# %%
############ Read the already validated conditional curves and audits ############
with np.load(f"{SOURCE_STEM}_curves.npz") as payload:
    curves = {name: payload[name].copy() for name in payload.files}
source_cases = curves["cases"].tolist()
assert source_cases == ["s7_first", "s8_first", *CASES]
assert curves["rows"].tolist() == [*[str(animal) for animal in ANIMALS], "equal_animal_mean"]
timing_bins, abort_bins = curves["timing_bins"], curves["abort_bins"]
np.testing.assert_allclose(np.diff(timing_bins), 0.040, rtol=0, atol=1e-14)
np.testing.assert_allclose(np.diff(abort_bins), 0.020, rtol=0, atol=1e-14)
np.testing.assert_allclose([timing_bins[0], timing_bins[-1]], [0, 2.2], rtol=0, atol=1e-14)
np.testing.assert_allclose([abort_bins[0], abort_bins[-1]], [-2, 2], rtol=0, atol=1e-14)
source_audit = pd.read_csv(f"{SOURCE_STEM}_count_area_audit.csv", dtype={"row": str})
audit = source_audit.loc[source_audit.case.isin(CASES)].copy()
assert len(audit) == 4 * 7 * 2 * 4


# %%
############ Independently check trial-ID selections and rate normalization ############
keys = ["animal", "session", "trial"]
for session_type in (7, 8):
    source = pd.read_csv(REPO / f"raw_data/LED7_s{session_type}.csv", float_precision="round_trip")
    assert source.shape == ({7: 138440, 8: 148428}[session_type], 52)
    assert source.training_level.eq(16).all() and source.session_type.eq(session_type).all()
    assert (source.repeat_trial.isin([0, 2]) | source.repeat_trial.isna()).all()
    assert source.LED_trial.isin([0, 1]).all()
    source["source_csv_row_index"] = np.arange(len(source))
    data = source.loc[source.animal.isin(ANIMALS)].sort_values(keys, kind="stable").copy()
    assert not data.duplicated(keys).any()
    data = data.set_index("source_csv_row_index", drop=False)
    neighbors = data.groupby(["animal", "session"], sort=False)
    previous = neighbors[["trial", "abort_event", "success"]].shift(1)
    following = neighbors[["trial", "success"]].shift(-1)
    abort = data.abort_event.eq(3)
    valid = data.success.isin([-1, 1])
    assert not (abort & valid).any()
    after_abort = previous.trial.eq(data.trial - 1) & previous.abort_event.eq(3)
    between_valid = (previous.trial.eq(data.trial - 1) & following.trial.eq(data.trial + 1)
                     & previous.success.isin([-1, 1]) & following.success.isin([-1, 1]))

    for selection, neighbor_condition in (("after_abort", after_abort), ("isolated", between_valid)):
        case = f"s{session_type}_{selection}"
        frame = data.loc[(abort | valid) & neighbor_condition].copy()
        frame["selected_abort"] = abort.loc[frame.index]
        if selection == "after_abort":
            manifest_path = Path(f"{TOP_MANIFEST_STEM}_{case}_selection_manifest.csv")
            saved = pd.read_csv(manifest_path, float_precision="round_trip")
            retained_field = "retained"
        else:
            manifest_path = Path(f"{SOURCE_STEM}_s{session_type}_manifest.csv")
            saved = pd.read_csv(manifest_path, float_precision="round_trip")
            saved = saved.loc[saved.new_eligible]
            retained_field = "new_matched_retained"
        assert saved.source_csv_row_index.is_unique
        saved = saved.set_index("source_csv_row_index").sort_index()
        frame = frame.sort_index()
        np.testing.assert_array_equal(frame.index.to_numpy(), saved.index.to_numpy())
        for field in (*keys, "LED_trial"):
            np.testing.assert_array_equal(frame[field].to_numpy(), saved[field].to_numpy())
        for field in ("intended_fix", "LED_onset_time", "timed_fix"):
            np.testing.assert_allclose(frame[field], saved[field], rtol=0, atol=0, equal_nan=True)
        np.testing.assert_array_equal(frame.selected_abort, saved.selected_abort)
        frame["retained"] = saved[retained_field].to_numpy(dtype=bool)
        frame["abort_from_LED_onset"] = frame.timed_fix - frame.LED_onset_time
        removed = frame.loc[~frame.retained]
        assert removed.selected_abort.all() and removed.abort_from_LED_onset.notna().all()
        assert removed.abort_from_LED_onset.lt(0).all()
        assert frame.loc[~frame.selected_abort, "retained"].all()
        counts = (int(frame.selected_abort.sum()), len(frame),
                  int((frame.selected_abort & frame.retained).sum()), int(frame.retained.sum()))
        assert counts == EXPECTED[case], (case, counts)

        case_index = source_cases.index(case)
        for animal_index, animal in enumerate(ANIMALS):
            animal_frame = frame.loc[frame.animal.eq(animal)]
            for led in (0, 1):
                original = animal_frame.loc[animal_frame.LED_trial.eq(led)]
                for metric in METRICS:
                    matched = metric == "matched_abort_from_LED_onset"
                    current = original.loc[original.retained] if matched else original
                    selected = current.loc[current.selected_abort]
                    value_column = metric if metric in METRICS[:2] else "abort_from_LED_onset"
                    values = selected[value_column].to_numpy()
                    values = values[np.isfinite(values)]
                    bins = timing_bins if metric in METRICS[:2] else abort_bins
                    hist = np.histogram(values, bins=bins)[0]
                    assert hist.sum() == len(values)
                    normalizer = len(values) if metric in METRICS[:2] else len(current)
                    bin_width = 0.040 if metric in METRICS[:2] else 0.020
                    calculated = hist / (normalizer * bin_width)
                    np.testing.assert_allclose(calculated, curves[metric][case_index, animal_index, led],
                                               rtol=0, atol=1e-14)
                    record = audit.loc[audit.case.eq(case) & audit.row.eq(str(animal))
                                       & audit.LED_trial.eq(led) & audit.metric.eq(metric)]
                    assert len(record) == 1
                    record = record.iloc[0]
                    assert (int(record.selected_aborts), int(record.eligible_denominator)) == (len(selected), len(current))
                    assert int(record.plotted_observations) == len(values)
                    expected_area = len(values) / normalizer
                    assert np.isclose(record.full_histogram_area, expected_area, rtol=0, atol=1e-12)
                    assert np.isclose(calculated @ np.diff(bins), expected_area, rtol=0, atol=1e-12)

        for metric in METRICS:
            bins = timing_bins if metric in METRICS[:2] else abort_bins
            for led in (0, 1):
                six = curves[metric][case_index, :6, led]
                mean = six.mean(axis=0)
                sem = six.std(axis=0, ddof=1) / np.sqrt(6)
                np.testing.assert_allclose(mean, curves[metric][case_index, 6, led], rtol=0, atol=1e-14)
                np.testing.assert_allclose(sem, curves[f"{metric}_sem"][case_index, 6, led], rtol=0, atol=1e-14)
                mean_record = audit.loc[audit.case.eq(case) & audit.row.eq("equal_animal_mean")
                                        & audit.LED_trial.eq(led) & audit.metric.eq(metric)]
                assert len(mean_record) == 1
                assert np.isclose(mean @ np.diff(bins), mean_record.iloc[0].full_histogram_area,
                                  rtol=0, atol=1e-12)

        for animal_index in range(6):
            off = curves[METRICS[3]][case_index, animal_index, 0, :100]
            on = curves[METRICS[3]][case_index, animal_index, 1, :100]
            assert np.max(np.abs(off - on)) <= MAX_PRE_BIN_RATE_DIFFERENCE + 1e-12
            assert abs(np.sum(off - on) * 0.020) <= MAX_PRE_FRACTION_DIFFERENCE + 1e-12


# %%
############ Four-row figure: reuse the exact checked curves, not a new match ############
fig, axes = plt.subplots(4, 4, figsize=(20, 14), sharex="col")
titles = ("intended_fix", "LED_onset_time", "Abort rate aligned to LED onset",
          "Pre-onset matched abort rate")
visible = (abort_bins[:-1] < RATE_VIEW_S[1]) & (abort_bins[1:] > RATE_VIEW_S[0])
timing_limits = [1.22 * max(float((curves[metric][source_cases.index(case), 6, led]
                                   + curves[f"{metric}_sem"][source_cases.index(case), 6, led]).max())
                            for case in CASES for led in (0, 1)) for metric in METRICS[:2]]
rate_limit = 1.25 * max(float((curves[metric][source_cases.index(case), 6, led]
                                + curves[f"{metric}_sem"][source_cases.index(case), 6, led])[visible].max())
                        for case in CASES for metric in METRICS[2:] for led in (0, 1))

for row, case in enumerate(CASES):
    session_type = int(case[1])
    is_isolated = case.endswith("isolated")
    row_name = "Isolated V→A→V" if is_isolated else "Abort after abort"
    formula = "VAV / (VAV + VVV)" if is_isolated else "AA / (AA + AV)"
    case_index = source_cases.index(case)
    for col, metric in enumerate(METRICS):
        ax = axes[row, col]
        is_rate = col >= 2
        bins = abort_bins if is_rate else timing_bins
        for led in (0, 1):
            mean = curves[metric][case_index, 6, led]
            sem = curves[f"{metric}_sem"][case_index, 6, led]
            ax.fill_between(bins, np.r_[np.maximum(mean - sem, 0), max(mean[-1] - sem[-1], 0)],
                            np.r_[mean + sem, mean[-1] + sem[-1]], step="post",
                            color=COLORS[led], alpha=0.16, linewidth=0)
            ax.stairs(mean, bins, color=COLORS[led], linewidth=1.5)
            record = audit.loc[audit.case.eq(case) & audit.row.eq("equal_animal_mean")
                               & audit.LED_trial.eq(led) & audit.metric.eq(metric)].iloc[0]
            label = "OFF" if led == 0 else "ON"
            note = (f"{label} n={int(record.plotted_observations):,}, N={int(record.eligible_denominator):,}; "
                    f"mean area={record.full_histogram_area:.3f}" if is_rate
                    else f"{label} n={int(record.selected_aborts):,}")
            ax.text(0.98, 0.97 - 0.085 * led, note, transform=ax.transAxes,
                    ha="right", va="top", color=COLORS[led], fontsize=8)
        ax.set_xlim(*(RATE_VIEW_S if is_rate else (0, 2.2)))
        ax.set_ylim(0, rate_limit if is_rate else timing_limits[col])
        if is_rate:
            ax.axvline(0, color="0.45", linestyle=":", linewidth=1)
        ax.grid(axis="y", color="0.91", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)
        if row == 0:
            ax.set_title(titles[col], fontsize=12, pad=12)
        if col == 0:
            ax.set_ylabel(f"Session type {session_type}\n{row_name}\n{formula}\nDensity (s⁻¹)", fontsize=10)
        elif col == 2:
            ax.set_ylabel("Conditional abort rate (s⁻¹)", fontsize=10)
        if row == 3:
            ax.set_xlabel("Time from scheduled LED onset (s)" if is_rate else "Time from fixation onset (s)",
                          fontsize=10)
        if col == 3:
            matched_audit = audit.loc[audit.case.eq(case) & audit.row.eq("equal_animal_mean")
                                      & audit.LED_trial.eq(0) & audit.metric.eq(metric)].iloc[0]
            ax.text(0.02, 0.77,
                    f"Pre: max |Δrate|={matched_audit.max_pre_bin_rate_difference_s_inv:.4f} s⁻¹\n"
                    f"|Δfraction|={100 * matched_audit.pre_onset_fraction_difference:.3f} pp",
                    transform=ax.transAxes, va="top", fontsize=8)

fig.suptitle("LED7 sequence-abort timing: conditional denominators", fontsize=18, y=0.995)
handles = [Line2D([0], [0], color=COLORS[led], lw=2,
                  label=f"Current trial LED {'OFF' if led == 0 else 'ON'}") for led in (0, 1)]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.966),
           ncol=2, frameon=False, fontsize=12)
fig.text(0.5, 0.022,
         "Equal-animal mean ± SEM: animals 92, 93, 98, 99, 100, 103. A = event-3 abort; V = valid (success ±1). "
         "LED condition is that of the current trial.\n"
         "Top: AA/(AA+AV); bottom: VAV/(VAV+VVV). Actual adjacent trial IDs required within animal/session. "
         "Timing: 40 ms unit-area bins; rates: 20 ms bins.\n"
         "Displayed n and N are summed counts; full rate area is the mean of six animal fractions, not pooled n/N. "
         "OFF onset is scheduled/counterfactual.\n"
         "Only selected pre-onset aborts were removed for column 4; outcome-based matching and future-outcome "
         "conditioning are descriptive, not causal evidence.",
         ha="center", fontsize=9, linespacing=1.45)
fig.tight_layout(rect=(0.005, 0.105, 0.995, 0.945), h_pad=1.6, w_pad=1.5)
figure_path = Path(f"{OUTPUT_STEM}_4x4.png")
fig.savefig(figure_path, dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)

audit.to_csv(f"{OUTPUT_STEM}_count_area_audit.csv", index=False)
print(f"Verified conditional identities, areas, equal-animal SEMs and matched tolerances. Saved {figure_path}")
