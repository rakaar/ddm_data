# %%
"""Compare assigned LED flags with light exposure at a sequential abort.

The top rows reproduce the existing AA/(AA+AV) conditional analysis. In the
bottom rows, an assigned-ON event-3 abort before its scheduled LED onset moves
to the effective-OFF group in both the numerator and denominator. This is an
outcome-defined descriptive grouping, not a causal comparison.
"""

from pathlib import Path
import os


# %%
############ Editable parameters ############
SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
SOURCE_STEM = SCRIPT_DIR / "led7_s7_s8_sequence_abort_valid_neighbor_conditioning"
TOP_MANIFEST_STEM = SCRIPT_DIR / "led7_s7_s8_sequence_abort_pre_onset_matched_summary"
OUTPUT_STEM = SCRIPT_DIR / "led7_s7_s8_sequence_abort_effective_exposure"
ANIMALS = (92, 93, 98, 99, 100, 103)
SESSION_TYPES = (7, 8)
CASES = ("s7_assigned", "s8_assigned", "s7_effective", "s8_effective")
METRICS = ("intended_fix", "LED_onset_time", "abort_from_LED_onset",
           "matched_abort_from_LED_onset")
RATE_VIEW_S = (-0.4, 0.5)
FIGURE_DPI = 240
COLORS = {0: "tab:blue", 1: "tab:red"}
EXPECTED_ASSIGNED = {
    7: {0: (1528, 8630), 1: (1569, 6988)},
    8: {0: (963, 6181), 1: (2463, 10671)},
}
EXPECTED_EFFECTIVE = {
    7: {0: (2398, 9500), 1: (699, 6118)},
    8: {0: (2097, 7315), 1: (1329, 9537)},
}
EXPECTED_MOVED = {7: 870, 8: 1134}
EXPECTED_PREVIOUS_MOVED = {7: 3105, 8: 3372}

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# %%
############ Load the completed original curves; the top rows are exact copies ############
with np.load(f"{SOURCE_STEM}_curves.npz") as payload:
    source_curves = {name: payload[name].copy() for name in payload.files}
source_cases = source_curves["cases"].tolist()
assert source_cases == ["s7_first", "s8_first", "s7_after_abort", "s8_after_abort",
                        "s7_isolated", "s8_isolated"]
rows = source_curves["rows"].tolist()
assert rows == [*[str(animal) for animal in ANIMALS], "equal_animal_mean"]
timing_bins = source_curves["timing_bins"]
abort_bins = source_curves["abort_bins"]
np.testing.assert_allclose(np.diff(timing_bins), 0.040, rtol=0, atol=1e-14)
np.testing.assert_allclose(np.diff(abort_bins), 0.020, rtol=0, atol=1e-14)
np.testing.assert_allclose([timing_bins[0], timing_bins[-1]], [0, 2.2], rtol=0, atol=1e-14)
np.testing.assert_allclose([abort_bins[0], abort_bins[-1]], [-2, 2], rtol=0, atol=1e-14)

curves = {"cases": np.asarray(CASES), "rows": np.asarray(rows),
          "timing_bins": timing_bins.copy(), "abort_bins": abort_bins.copy()}
for metric in METRICS:
    n_bins = len(timing_bins) - 1 if metric in METRICS[:2] else len(abort_bins) - 1
    curves[metric] = np.full((4, 7, 2, n_bins), np.nan)
    curves[f"{metric}_sem"] = np.full((4, 7, 2, n_bins), np.nan)
    for new_row, source_case in enumerate(("s7_after_abort", "s8_after_abort")):
        old_row = source_cases.index(source_case)
        curves[metric][new_row] = source_curves[metric][old_row]
        curves[f"{metric}_sem"][new_row] = source_curves[f"{metric}_sem"][old_row]
        np.testing.assert_array_equal(curves[metric][new_row], source_curves[metric][old_row])

source_audit = pd.read_csv(f"{SOURCE_STEM}_count_area_audit.csv", dtype={"row": str})
audit_records = []
for new_row, source_case in enumerate(("s7_after_abort", "s8_after_abort")):
    for _, record in source_audit.loc[source_audit.case.eq(source_case)].iterrows():
        audit_records.append({
            "case": CASES[new_row], "session_type": int(record.session_type),
            "grouping": "assigned_LED_trial", "row": record.row,
            "LED_group": int(record.LED_trial), "metric": record.metric,
            "selected_aborts": int(record.selected_aborts),
            "plotted_observations": int(record.plotted_observations),
            "eligible_denominator": int(record.eligible_denominator),
            "full_histogram_area": float(record.full_histogram_area),
            "moved_pre_onset_aborts": 0,
            "unknown_exposure_aborts": (int(record.missing_aligned_time)
                                        if int(record.LED_trial) == 1 else 0),
            "denominator_definition": "adjacent preceding A; current A or V; current assigned LED flag",
        })


# %%
############ Classify intact adjacent trial pairs and recompute the effective rows ############
manifest_parts = []
keys = ["animal", "session", "trial"]
for session_type in SESSION_TYPES:
    source_path = REPO / f"raw_data/LED7_s{session_type}.csv"
    source = pd.read_csv(source_path, float_precision="round_trip")
    assert source.shape == ({7: 138440, 8: 148428}[session_type], 52)
    assert source.training_level.eq(16).all() and source.session_type.eq(session_type).all()
    assert (source.repeat_trial.isin([0, 2]) | source.repeat_trial.isna()).all()
    assert source.LED_trial.isin([0, 1]).all()
    source["source_csv_row_index"] = np.arange(len(source))
    data = source.loc[source.animal.isin(ANIMALS)].sort_values(keys, kind="stable").copy()
    assert not data.duplicated(keys).any()
    data = data.set_index("source_csv_row_index", drop=False)
    is_abort = data.abort_event.eq(3)
    is_valid = data.success.isin([-1, 1])
    assert not (is_abort & is_valid).any()
    data["selected_abort"] = is_abort
    data["moved_pre_onset_abort"] = (
        data.LED_trial.eq(1) & is_abort & data.timed_fix.notna()
        & data.timed_fix.lt(data.LED_onset_time)
    )
    data["exposure_unknown"] = data.LED_trial.eq(1) & is_abort & data.timed_fix.isna()
    data["effective_LED_trial"] = data.LED_trial.mask(data.moved_pre_onset_abort, 0).astype(int)
    data["abort_from_LED_onset"] = data.timed_fix - data.LED_onset_time

    previous = data.groupby(["animal", "session"], sort=False)[[
        "trial", "abort_event", "LED_trial", "effective_LED_trial",
        "moved_pre_onset_abort", "exposure_unknown", "source_csv_row_index",
    ]].shift(1)
    has_previous_abort = previous.trial.eq(data.trial - 1) & previous.abort_event.eq(3)
    pair = data.loc[(is_abort | is_valid) & has_previous_abort].copy()
    pair["previous_trial"] = previous.loc[pair.index, "trial"].to_numpy()
    pair["previous_source_csv_row_index"] = previous.loc[pair.index, "source_csv_row_index"].to_numpy()
    pair["previous_assigned_LED_trial"] = previous.loc[pair.index, "LED_trial"].to_numpy()
    pair["previous_effective_LED_trial"] = previous.loc[pair.index, "effective_LED_trial"].to_numpy()
    pair["previous_moved_pre_onset_abort"] = previous.loc[pair.index, "moved_pre_onset_abort"].to_numpy(bool)
    pair["previous_exposure_unknown"] = previous.loc[pair.index, "exposure_unknown"].to_numpy(bool)
    assert pair.previous_trial.eq(pair.trial - 1).all()
    assert int(pair.previous_moved_pre_onset_abort.sum()) == EXPECTED_PREVIOUS_MOVED[session_type]
    assert not (pair.LED_trial.eq(1) & ~pair.selected_abort
                & pair.timed_fix.lt(pair.LED_onset_time)).any()
    assert int(pair.moved_pre_onset_abort.sum()) == EXPECTED_MOVED[session_type]
    assert int(pair.exposure_unknown.sum()) == ({7: 0, 8: 2}[session_type])
    assert pair.effective_LED_trial.isin([0, 1]).all()
    assert pair.loc[pair.moved_pre_onset_abort, "effective_LED_trial"].eq(0).all()
    assert pair.loc[~pair.moved_pre_onset_abort, "effective_LED_trial"].eq(
        pair.loc[~pair.moved_pre_onset_abort, "LED_trial"]
    ).all()

    # The original top-row source manifest fixes pair identity and timing values.
    original_manifest = pd.read_csv(
        f"{TOP_MANIFEST_STEM}_s{session_type}_after_abort_selection_manifest.csv",
        float_precision="round_trip",
    ).set_index("source_csv_row_index").sort_index()
    pair = pair.sort_index()
    np.testing.assert_array_equal(pair.index.to_numpy(), original_manifest.index.to_numpy())
    for field in (*keys, "LED_trial", "abort_event", "success", "intended_fix",
                  "LED_onset_time", "timed_fix"):
        np.testing.assert_allclose(pair[field], original_manifest[field], rtol=0, atol=0,
                                   equal_nan=True)
    np.testing.assert_array_equal(pair.selected_abort, original_manifest.selected_abort)
    np.testing.assert_array_equal(pair.previous_trial, original_manifest.previous_trial)
    assert len(pair) == sum(n for _, n in EXPECTED_ASSIGNED[session_type].values())
    assert int(pair.selected_abort.sum()) == sum(a for a, _ in EXPECTED_ASSIGNED[session_type].values())

    for led in (0, 1):
        assigned = pair.loc[pair.LED_trial.eq(led)]
        effective = pair.loc[pair.effective_LED_trial.eq(led)]
        assert (int(assigned.selected_abort.sum()), len(assigned)) == EXPECTED_ASSIGNED[session_type][led]
        assert (int(effective.selected_abort.sum()), len(effective)) == EXPECTED_EFFECTIVE[session_type][led]

    # Recalculate original columns 1-3 and compare them with the stored top rows.
    top_row = SESSION_TYPES.index(session_type)
    bottom_row = top_row + 2
    for animal_index, animal in enumerate(ANIMALS):
        animal_pair = pair.loc[pair.animal.eq(animal)]
        for led in (0, 1):
            for row_index, led_column in ((top_row, "LED_trial"),
                                          (bottom_row, "effective_LED_trial")):
                eligible = animal_pair.loc[animal_pair[led_column].eq(led)]
                selected = eligible.loc[eligible.selected_abort]
                assert len(eligible) > 0 and len(selected) > 0
                assert np.isfinite(selected[["intended_fix", "LED_onset_time"]]).all().all()
                for metric in METRICS[:3]:
                    bins = timing_bins if metric in METRICS[:2] else abort_bins
                    values = selected[metric].to_numpy()
                    values = values[np.isfinite(values)]
                    counts = np.histogram(values, bins=bins)[0]
                    assert int(counts.sum()) == len(values)
                    normalizer = len(values) if metric in METRICS[:2] else len(eligible)
                    bin_width = 0.040 if metric in METRICS[:2] else 0.020
                    calculated = counts / (normalizer * bin_width)
                    area = float(calculated @ np.diff(bins))
                    assert np.isclose(area, len(values) / normalizer, rtol=0, atol=1e-12)
                    if row_index == top_row:
                        np.testing.assert_allclose(calculated, curves[metric][row_index, animal_index, led],
                                                   rtol=0, atol=1e-14)
                    else:
                        curves[metric][row_index, animal_index, led] = calculated
                        audit_records.append({
                            "case": CASES[row_index], "session_type": session_type,
                            "grouping": "effective_at_abort", "row": str(animal),
                            "LED_group": led, "metric": metric,
                            "selected_aborts": len(selected),
                            "plotted_observations": len(values),
                            "eligible_denominator": len(eligible),
                            "full_histogram_area": area,
                            "moved_pre_onset_aborts": int(eligible.moved_pre_onset_abort.sum()),
                            "unknown_exposure_aborts": int(eligible.exposure_unknown.sum()),
                            "denominator_definition": "adjacent preceding A; current A or V; current effective LED flag",
                        })

    manifest_parts.append(pair[[
        "source_csv_row_index", *keys, "session_type", "training_level", "repeat_trial",
        "LED_trial", "effective_LED_trial", "abort_event", "success", "selected_abort",
        "intended_fix", "LED_onset_time", "timed_fix", "abort_from_LED_onset",
        "moved_pre_onset_abort", "exposure_unknown", "previous_trial",
        "previous_source_csv_row_index", "previous_assigned_LED_trial",
        "previous_effective_LED_trial", "previous_moved_pre_onset_abort",
        "previous_exposure_unknown",
    ]].copy())

    for metric in METRICS[:3]:
        bins = timing_bins if metric in METRICS[:2] else abort_bins
        for led in (0, 1):
            six = curves[metric][bottom_row, :6, led]
            curves[metric][bottom_row, 6, led] = six.mean(axis=0)
            curves[f"{metric}_sem"][bottom_row, 6, led] = six.std(axis=0, ddof=1) / np.sqrt(6)
            np.testing.assert_allclose(curves[metric][bottom_row, 6, led], six.sum(axis=0) / 6,
                                       rtol=0, atol=1e-14)
            np.testing.assert_allclose(curves[f"{metric}_sem"][bottom_row, 6, led],
                                       np.sqrt(((six - six.mean(axis=0)) ** 2).sum(axis=0) / 5 / 6),
                                       rtol=0, atol=1e-14)
            animal_records = [record for record in audit_records if record["case"] == CASES[bottom_row]
                              and record["metric"] == metric and record["LED_group"] == led]
            assert len(animal_records) == 6
            mean_area = float(curves[metric][bottom_row, 6, led] @ np.diff(bins))
            assert np.isclose(mean_area,
                              np.mean([record["full_histogram_area"] for record in animal_records]),
                              rtol=0, atol=1e-12)
            audit_records.append({
                "case": CASES[bottom_row], "session_type": session_type,
                "grouping": "effective_at_abort", "row": "equal_animal_mean",
                "LED_group": led, "metric": metric,
                "selected_aborts": sum(record["selected_aborts"] for record in animal_records),
                "plotted_observations": sum(record["plotted_observations"] for record in animal_records),
                "eligible_denominator": sum(record["eligible_denominator"] for record in animal_records),
                "full_histogram_area": mean_area,
                "moved_pre_onset_aborts": sum(record["moved_pre_onset_aborts"] for record in animal_records),
                "unknown_exposure_aborts": sum(record["unknown_exposure_aborts"] for record in animal_records),
                "denominator_definition": "mean of six animal-normalized AA/(AA+AV) curves",
            })

manifest = pd.concat(manifest_parts, ignore_index=True)
assert not manifest.duplicated(["session_type", *keys]).any()
assert len(manifest) == 15618 + 16852
assert int(manifest.moved_pre_onset_abort.sum()) == 870 + 1134
assert int(manifest.exposure_unknown.sum()) == 2
audit = pd.DataFrame(audit_records)
assert len(audit) == 2 * 7 * 2 * 4 + 2 * 7 * 2 * 3


# %%
############ Display assigned versus effective status ############
fig, axes = plt.subplots(4, 4, figsize=(20, 14), sharex="col")
titles = ("intended_fix", "LED_onset_time", "Abort rate aligned to LED onset",
          "Pre-onset matched abort rate")
visible = (abort_bins[:-1] < RATE_VIEW_S[1]) & (abort_bins[1:] > RATE_VIEW_S[0])
timing_limits = [1.22 * max(float((curves[metric][row, 6, led]
                                   + curves[f"{metric}_sem"][row, 6, led]).max())
                            for row in range(4) for led in (0, 1)) for metric in METRICS[:2]]
rate_limit = 1.25 * max(
    float((curves[metric][row, 6, led] + curves[f"{metric}_sem"][row, 6, led])[visible].max())
    for row in range(4) for metric in METRICS[2:] for led in (0, 1)
    if not (row >= 2 and metric == METRICS[3])
)

for row, case in enumerate(CASES):
    session_type = 7 if row % 2 == 0 else 8
    effective = row >= 2
    for col, metric in enumerate(METRICS):
        ax = axes[row, col]
        is_rate = col >= 2
        bins = abort_bins if is_rate else timing_bins
        ax.set_xlim(*(RATE_VIEW_S if is_rate else (0, 2.2)))
        ax.set_ylim(0, rate_limit if is_rate else timing_limits[col])
        if is_rate and not (effective and col == 3):
            ax.axvline(0, color="0.45", linestyle=":", linewidth=1)
        ax.grid(axis="y", color="0.91", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)
        if row == 0:
            ax.set_title(titles[col], fontsize=12, pad=12)
        if col == 0:
            label = "Effective at abort" if effective else "Assigned LED"
            ax.set_ylabel(f"Session type {session_type}\n{label}\nAA / (AA + AV)\nDensity (s⁻¹)",
                          fontsize=10)
        elif col == 2:
            ax.set_ylabel("Conditional abort rate (s⁻¹)", fontsize=10)
        if row == 3:
            ax.set_xlabel("Time from scheduled LED onset (s)" if is_rate
                          else "Time from fixation onset (s)", fontsize=10)

        if effective and col == 3:
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14)
            continue

        for led in (0, 1):
            mean = curves[metric][row, 6, led]
            sem = curves[f"{metric}_sem"][row, 6, led]
            ax.fill_between(bins, np.r_[np.maximum(mean - sem, 0), max(mean[-1] - sem[-1], 0)],
                            np.r_[mean + sem, mean[-1] + sem[-1]], step="post",
                            color=COLORS[led], alpha=0.16, linewidth=0)
            ax.stairs(mean, bins, color=COLORS[led], linewidth=1.5)
            record = audit.loc[audit.case.eq(case) & audit.row.eq("equal_animal_mean")
                               & audit.LED_group.eq(led) & audit.metric.eq(metric)].iloc[0]
            label = "OFF" if led == 0 else "ON"
            note = (f"{label} n={int(record.plotted_observations):,}, N={int(record.eligible_denominator):,}; "
                    f"mean area={record.full_histogram_area:.3f}" if is_rate
                    else f"{label} n={int(record.selected_aborts):,}")
            ax.text(0.98, 0.97 - 0.085 * led, note, transform=ax.transAxes,
                    ha="right", va="top", color=COLORS[led], fontsize=8)
        if not effective and col == 3:
            original = source_audit.loc[source_audit.case.eq(f"s{session_type}_after_abort")
                                        & source_audit.row.eq("equal_animal_mean")
                                        & source_audit.LED_trial.eq(0)
                                        & source_audit.metric.eq(metric)].iloc[0]
            ax.text(0.02, 0.77,
                    f"Pre: max |Δrate|={original.max_pre_bin_rate_difference_s_inv:.4f} s⁻¹\n"
                    f"|Δfraction|={100 * original.pre_onset_fraction_difference:.3f} pp",
                    transform=ax.transAxes, va="top", fontsize=8)

fig.suptitle("LED7 sequential aborts: assigned versus effective LED status", fontsize=18, y=0.995)
handles = [Line2D([0], [0], color=COLORS[led], lw=2,
                  label=f"Current trial {'OFF' if led == 0 else 'ON'} group") for led in (0, 1)]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.966),
           ncol=2, frameon=False, fontsize=12)
fig.text(0.5, 0.022,
         "Animals 92, 93, 98, 99, 100, 103; equal-animal mean ± SEM. A = event-3 abort; V = valid "
         "(success ±1). Adjacent trial IDs required within animal/session.\n"
         "Top: assigned LED flag (exact original curves); bottom: assigned-ON aborts before scheduled onset "
         "move to effective OFF. Every rate denominator is AA+AV in the displayed current-trial group.\n"
         "Timing: 40 ms unit-area bins; rate: 20 ms bins. Displayed n/N are summed counts; full rate area "
         "is the mean of animal fractions, not pooled n/N. OFF onset is scheduled/counterfactual.\n"
         "Two s7 OFF and two s8 ON aborts lack timed_fix: all stay in denominators, not rate histograms; "
         "s8 ON exposure is unknown. This regrouping and pre-onset matching are descriptive, not causal evidence.",
         ha="center", fontsize=9, linespacing=1.45)
fig.tight_layout(rect=(0.005, 0.105, 0.995, 0.945), h_pad=1.6, w_pad=1.5)
figure_path = Path(f"{OUTPUT_STEM}_4x4.png")
fig.savefig(figure_path, dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)

manifest_path = Path(f"{OUTPUT_STEM}_pair_classification_manifest.csv")
audit_path = Path(f"{OUTPUT_STEM}_count_area_audit.csv")
curves_path = Path(f"{OUTPUT_STEM}_curves.npz")
manifest.to_csv(manifest_path, index=False)
audit.to_csv(audit_path, index=False)
np.savez_compressed(curves_path, **curves)
print(f"Saved {figure_path}")
print(f"Saved {manifest_path}, {audit_path}, and {curves_path}")
for session_type in SESSION_TYPES:
    moved = int(manifest.loc[manifest.session_type.eq(session_type), "moved_pre_onset_abort"].sum())
    print(f"s{session_type}: moved {moved:,} assigned-ON pre-onset aborts to effective OFF")
