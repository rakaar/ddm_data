# %%
"""LED7 pair-conditioned abort timing, without subsampling or onset matching.

First and second mean the members of every adjacent AA pair, not positions in
a complete run. With A = event-3 abort and V = success +/-1, the denominators
are AA+VA (first member), AA+AV (second member), and VAV+VVV (isolated).
Each trial is grouped by its own LED flag. CSV LED_onset_time is already
fixation-referenced, including the scheduled/counterfactual onset for OFF.
"""

from pathlib import Path
import hashlib
import json
import os


# %%
############ Editable parameters ############
SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
OUTPUT_STEM = SCRIPT_DIR / "led7_s7_s8_pair_conditioned_sequence_abort_timing"
REFERENCE_STEM = SCRIPT_DIR / "led7_s7_s8_sequence_abort_valid_neighbor_conditioning"
ANIMALS = (92, 93, 98, 99, 100, 103)
SESSION_TYPES = (7, 8)
POSITIONS = ("first_member", "second_member", "isolated")
CASES = tuple((st, position) for position in POSITIONS for st in SESSION_TYPES)
TIMING_BIN_WIDTH_S = 0.040
ABORT_BIN_WIDTH_S = 0.020
TIMING_RANGE_S = (0.0, 2.2)
ABORT_RANGE_S = (-2.0, 2.0)
ABORT_DISPLAY_S = (-0.4, 0.5)
FIGURE_DPI = 240
COLORS = {0: "tab:blue", 1: "tab:red"}
METRICS = ("intended_fix", "LED_onset_time", "abort_from_LED_onset")
COLUMN_TITLES = ("intended_fix", "LED_onset_time", "Abort rate aligned to LED onset")
DENOMINATOR_FORMULAS = {"first_member": "AA + VA", "second_member": "AA + AV", "isolated": "VAV + VVV"}
ROW_LABELS = {"first_member": "First member of AA", "second_member": "Second member of AA",
              "isolated": "Valid → abort → valid"}
# Each LED condition: selected aborts, finite aligned aborts, denominator trials.
EXPECTED_COUNTS = {
    (7, "first_member"): {0: (1736, 1736, 10485), 1: (1361, 1361, 5240)},
    (8, "first_member"): {0: (1490, 1490, 10443), 1: (1936, 1936, 6487)},
    (7, "second_member"): {0: (1528, 1526, 8630), 1: (1569, 1569, 6988)},
    (8, "second_member"): {0: (963, 963, 6181), 1: (2463, 2461, 10671)},
    (7, "isolated"): {0: (6498, 6498, 43706), 1: (3365, 3365, 18532)},
    (8, "isolated"): {0: (6406, 6406, 50811), 1: (4105, 4105, 21044)},
}

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

keys = ["animal", "session", "trial"]
timing_bins = np.linspace(*TIMING_RANGE_S, round(np.diff(TIMING_RANGE_S)[0] / TIMING_BIN_WIDTH_S) + 1)
abort_bins = np.linspace(*ABORT_RANGE_S, round(np.diff(ABORT_RANGE_S)[0] / ABORT_BIN_WIDTH_S) + 1)
bins_by_metric = dict(zip(METRICS, (timing_bins, timing_bins, abort_bins)))
width_by_metric = dict(zip(METRICS, (TIMING_BIN_WIDTH_S, TIMING_BIN_WIDTH_S, ABORT_BIN_WIDTH_S)))
for metric, bins in bins_by_metric.items():
    np.testing.assert_allclose(np.diff(bins), width_by_metric[metric], rtol=0, atol=1e-14)

reference_selections = {
    (st, position): SCRIPT_DIR / f"led7_s{st}_{'abort_after_abort' if position == 'second_member' else 'isolated_abort'}_timing_selected_trials.csv"
    for st in SESSION_TYPES for position in POSITIONS[1:]
}
protected_paths = [REPO / f"raw_data/LED7_s{st}.csv" for st in SESSION_TYPES]
protected_paths += list(reference_selections.values())
protected_paths += list(SCRIPT_DIR.glob("led7_s7_s8_sequence_abort_valid_neighbor_conditioning*"))
protected_paths += list(SCRIPT_DIR.glob("led7_s7_s8_sequence_abort_pre_onset_matched_summary*"))
protected_paths += [SCRIPT_DIR / "plot_led7_s7_s8_sequence_abort_valid_neighbor_conditioning.py",
                    SCRIPT_DIR / "plot_led7_s7_s8_sequence_abort_pre_onset_matched_summary.py"]
input_hashes = {}
for path in protected_paths:
    with path.open("rb") as handle:
        input_hashes[str(path.relative_to(REPO))] = hashlib.file_digest(handle, "sha256").hexdigest()
with np.load(f"{REFERENCE_STEM}_curves.npz") as payload:
    reference_arrays = {key: payload[key].copy() for key in payload.files}
reference_panels = pd.read_csv(f"{REFERENCE_STEM}_count_area_audit.csv", float_precision="round_trip")


# %%
############ Shared classification for synthetic tests and both real CSV streams ############
def classify_trial_neighbors(source):
    """Preserve trial IDs; classify before splitting outcomes or LED conditions."""
    data = source.sort_values(keys, kind="stable").copy()
    assert not data.duplicated(keys).any() and data.source_csv_row_index.is_unique
    grouped = data.groupby(["animal", "session"], sort=False)
    neighbor_fields = ["trial", "abort_event", "success", "LED_trial", "source_csv_row_index"]
    previous, following = grouped[neighbor_fields].shift(1), grouped[neighbor_fields].shift(-1)
    for prefix, neighbor in (("previous", previous), ("next", following)):
        for field in neighbor_fields:
            data[f"{prefix}_{field}"] = neighbor[field]
    is_abort, is_valid = data.abort_event.eq(3), data.success.isin([-1, 1])
    assert not (is_abort & is_valid).any()
    data["event_or_valid"] = is_abort | is_valid
    data["previous_adjacent"] = previous.trial.eq(data.trial - 1)
    data["next_adjacent"] = following.trial.eq(data.trial + 1)
    conditions = {
        "first_member": data.next_adjacent & following.abort_event.eq(3),
        "second_member": data.previous_adjacent & previous.abort_event.eq(3),
        "isolated": data.previous_adjacent & data.next_adjacent
        & previous.success.isin([-1, 1]) & following.success.isin([-1, 1]),
    }
    for position, condition in conditions.items():
        data[f"{position}_eligible"] = data.event_or_valid & condition
        data[f"{position}_selected"] = is_abort & condition
    data["abort_from_LED_onset"] = data.timed_fix - data.LED_onset_time
    return data


# %%
############ Small examples: overlapping pairs, LED switches, gaps and boundaries ############
examples = {
    1: [(1, "V", 0), (2, "A", 0), (3, "A", 1), (4, "A", 1), (5, "V", 0), (6, "V", 1), (7, "V", 0)],
    2: [(10, "V", 1), (11, "A", 0), (12, "V", 0), (13, "V", 1), (14, "V", 1)],
    3: [(1, "V", 0), (3, "A", 0), (4, "A", 1), (6, "V", 0)],
    4: [(1, "A", 0)],
    5: [(2, "A", 1)],
    6: [(1, "O", 0), (2, "A", 1), (3, "V", 0), (4, "A", 0), (5, "O", 1)],
}
records = []
for session, example in examples.items():
    for trial, outcome, led in example:
        records.append({"animal": 1, "session": session, "trial": trial, "LED_trial": led,
                        "abort_event": 3 if outcome == "A" else (2 if outcome == "O" else np.nan),
                        "success": 1 if outcome == "V" else np.nan, "timed_fix": 0.3, "LED_onset_time": 0.2})
# Same session number but another animal must not complete a pair.
records.append({**records[0], "animal": 2, "session": 4, "trial": 2, "abort_event": 3, "success": np.nan})
test_data = pd.DataFrame(records)
test_data["source_csv_row_index"] = np.arange(len(test_data))
test_data.loc[test_data.session.eq(1) & test_data.trial.eq(4), "timed_fix"] = np.nan
test_data = classify_trial_neighbors(test_data)
expected_selected = {
    "first_member": {(1, 1, 2), (1, 1, 3), (1, 3, 3)},
    "second_member": {(1, 1, 3), (1, 1, 4), (1, 3, 4)},
    "isolated": {(1, 2, 11)},
}
expected_eligible = {
    "first_member": {(1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 2, 10), (1, 3, 3), (1, 6, 3)},
    "second_member": {(1, 1, 3), (1, 1, 4), (1, 1, 5), (1, 2, 12), (1, 3, 4), (1, 6, 3)},
    "isolated": {(1, 1, 6), (1, 2, 11), (1, 2, 13)},
}
for position in POSITIONS:
    assert set(map(tuple, test_data.loc[test_data[f"{position}_selected"], keys].to_numpy())) == expected_selected[position]
    assert set(map(tuple, test_data.loc[test_data[f"{position}_eligible"], keys].to_numpy())) == expected_eligible[position]
assert test_data.loc[test_data.second_member_selected, "timed_fix"].isna().sum() == 1
switched = test_data.loc[test_data.first_member_selected & test_data.LED_trial.ne(test_data.next_LED_trial)]
assert len(switched) == 2


# %%
############ Read standardized CSVs; independently verify neighbors and AA linkage ############
streams, pair_manifests = {}, []
for st in SESSION_TYPES:
    source_path = REPO / f"raw_data/LED7_s{st}.csv"
    source = pd.read_csv(source_path, float_precision="round_trip")
    assert source.shape == ({7: 138440, 8: 148428}[st], 52)
    assert source.training_level.eq(16).all() and source.session_type.eq(st).all()
    assert (source.repeat_trial.isin([0, 2]) | source.repeat_trial.isna()).all()
    assert source.LED_trial.isin([0, 1]).all()
    source["source_csv_row_index"] = np.arange(len(source))
    data = source.loc[source.animal.isin(ANIMALS)].copy()
    assert len(data) == {7: 100251, 8: 113941}[st]
    assert tuple(sorted(data.animal.unique())) == ANIMALS
    for field in [*keys, "LED_trial"]:
        assert data[field].notna().all() and data[field].eq(np.floor(data[field])).all()
        data[field] = data[field].astype(int)
    data = classify_trial_neighbors(data)
    # A separate dictionary lookup cannot mistake adjacent rows for adjacent IDs.
    abort_keys = set(map(tuple, data.loc[data.abort_event.eq(3), keys].to_numpy()))
    valid_keys = set(map(tuple, data.loc[data.success.isin([-1, 1]), keys].to_numpy()))
    expected_conditions = {
        "first_member": np.array([(a, s, t + 1) in abort_keys for a, s, t in data[keys].to_numpy()]),
        "second_member": np.array([(a, s, t - 1) in abort_keys for a, s, t in data[keys].to_numpy()]),
        "isolated": np.array([(a, s, t - 1) in valid_keys and (a, s, t + 1) in valid_keys
                              for a, s, t in data[keys].to_numpy()]),
    }
    for position, condition in expected_conditions.items():
        np.testing.assert_array_equal(data[f"{position}_eligible"], condition & data.event_or_valid.to_numpy())
        np.testing.assert_array_equal(data[f"{position}_selected"], condition & data.abort_event.eq(3).to_numpy())
    first = data.loc[data.first_member_selected]
    second = data.loc[data.second_member_selected]
    np.testing.assert_array_equal(first.next_source_csv_row_index.astype(int).to_numpy(), second.source_csv_row_index.to_numpy())
    np.testing.assert_array_equal(second.previous_source_csv_row_index.astype(int).to_numpy(), first.source_csv_row_index.to_numpy())
    assert len(first) == len(second) == {7: 3097, 8: 3426}[st]
    pair_manifests.append(pd.DataFrame({
        "session_type": st, "animal": first.animal.to_numpy(), "session": first.session.to_numpy(),
        "first_trial": first.trial.to_numpy(), "second_trial": second.trial.to_numpy(),
        "first_source_csv_row_index": first.source_csv_row_index.to_numpy(),
        "second_source_csv_row_index": second.source_csv_row_index.to_numpy(),
        "first_LED_trial": first.LED_trial.to_numpy(), "second_LED_trial": second.LED_trial.to_numpy(),
    }))
    for position in POSITIONS[1:]:
        saved = pd.read_csv(reference_selections[(st, position)], float_precision="round_trip").set_index("source_csv_row_index").sort_index()
        selected = data.loc[data[f"{position}_selected"]].set_index("source_csv_row_index").sort_index()
        fields = [*keys, "LED_trial", "abort_event", "success", "timed_fix", "intended_fix", "LED_onset_time"]
        pd.testing.assert_frame_equal(selected[fields], saved[fields], check_dtype=False)
    streams[st] = data


# %%
############ Per-animal histograms with their own conditional denominators ############
row_names = [*[str(animal) for animal in ANIMALS], "equal_animal_mean"]
curves = {metric: np.zeros((6, 7, 2, len(bins_by_metric[metric]) - 1)) for metric in METRICS}
sems = {metric: np.zeros_like(curves[metric]) for metric in METRICS}
panel_records, curve_parts, selected_manifests, eligible_manifests = [], [], [], {}
manifest_fields = ["source_csv_row_index", *keys, "session_type", "training_level", "repeat_trial", "LED_trial",
                   "abort_event", "success", "timed_fix", "intended_fix", "LED_onset_time", "abort_from_LED_onset",
                   "previous_trial", "next_trial", "previous_abort_event", "next_abort_event",
                   "previous_success", "next_success", "previous_LED_trial", "next_LED_trial",
                   "previous_source_csv_row_index", "next_source_csv_row_index", "previous_adjacent", "next_adjacent"]
for ci, (st, position) in enumerate(CASES):
    case_id = f"s{st}_{position}"
    data = streams[st]
    eligible = data.loc[data[f"{position}_eligible"], manifest_fields].copy()
    eligible.insert(0, "source_csv", f"raw_data/LED7_s{st}.csv")
    eligible.insert(0, "case", case_id)
    eligible["selected_abort"] = data.loc[eligible.index, f"{position}_selected"]
    eligible["denominator_formula"] = DENOMINATOR_FORMULAS[position]
    eligible_manifests[case_id] = eligible
    selected_manifests.append(eligible.loc[eligible.selected_abort])
    selected = eligible.loc[eligible.selected_abort]
    assert np.isfinite(selected[["intended_fix", "LED_onset_time"]]).all().all()
    finite = np.isfinite(selected.abort_from_LED_onset)
    assert selected.loc[finite, "timed_fix"].lt(selected.loc[finite, "intended_fix"]).all()
    for led in (0, 1):
        group = eligible.loc[eligible.LED_trial.eq(led)]
        chosen = group.loc[group.selected_abort]
        actual = len(chosen), int(np.isfinite(chosen.abort_from_LED_onset).sum()), len(group)
        assert actual == EXPECTED_COUNTS[(st, position)][led], (case_id, led, actual)
    missing = selected.loc[~finite]
    expected_missing = ({7: {(92, 0): 1, (100, 0): 1}, 8: {(93, 1): 2}}[st]
                        if position == "second_member" else {})
    assert missing.groupby(["animal", "LED_trial"]).size().to_dict() == expected_missing

    for ai, animal in enumerate(ANIMALS):
        for led in (0, 1):
            group = eligible.loc[eligible.animal.eq(animal) & eligible.LED_trial.eq(led)]
            chosen = group.loc[group.selected_abort]
            assert len(group) > 0 and len(chosen) > 0
            for metric in METRICS:
                values = chosen[metric].to_numpy()
                values = values[np.isfinite(values)]
                bins, width = bins_by_metric[metric], width_by_metric[metric]
                counts = np.histogram(values, bins=bins)[0]
                assert counts.sum() == len(values) and len(values) > 0
                normalizer = len(group) if metric == METRICS[2] else len(values)
                heights = counts / (normalizer * width)
                area = float(heights @ np.diff(bins))
                expected_area = len(values) / normalizer
                assert np.isclose(area, expected_area, rtol=0, atol=1e-12)
                curves[metric][ci, ai, led] = heights
                panel_records.append({
                    "case": case_id, "session_type": st, "position": position, "row": str(animal),
                    "LED_trial": led, "metric": metric, "n_animals": 1,
                    "selected_aborts": len(chosen), "finite_aligned_aborts": int(np.isfinite(chosen.abort_from_LED_onset).sum()),
                    "missing_aligned_time": int((~np.isfinite(chosen.abort_from_LED_onset)).sum()),
                    "plotted_observations": len(values), "eligible_denominator": len(group),
                    "histogram_normalizer": normalizer, "full_histogram_area": area, "expected_area": expected_area,
                    "selected_abort_fraction": len(chosen) / len(group), "bin_width_s": width,
                    "denominator_formula": DENOMINATOR_FORMULAS[position],
                })
                curve_parts.append(pd.DataFrame({
                    "case": case_id, "row": str(animal), "LED_trial": led, "metric": metric,
                    "bin_left_s": bins[:-1], "bin_right_s": bins[1:], "height": heights,
                    "sem": np.nan, "bin_count": counts,
                }))
    animal_audits = pd.DataFrame(panel_records)
    animal_audits = animal_audits.loc[animal_audits.case.eq(case_id)]
    for led in (0, 1):
        for metric in METRICS:
            matrix = curves[metric][ci, :6, led]
            mean = matrix.mean(axis=0)
            sem = matrix.std(axis=0, ddof=1) / np.sqrt(6)
            np.testing.assert_allclose(sem, np.sqrt(((matrix - mean) ** 2).sum(axis=0) / 30), atol=1e-14)
            curves[metric][ci, 6, led], sems[metric][ci, 6, led] = mean, sem
            group_audit = animal_audits.loc[animal_audits.LED_trial.eq(led) & animal_audits.metric.eq(metric)]
            assert len(group_audit) == 6
            bins = bins_by_metric[metric]
            area = float(mean @ np.diff(bins))
            assert np.isclose(area, group_audit.expected_area.mean(), rtol=0, atol=1e-12)
            record = {"case": case_id, "session_type": st, "position": position, "row": "equal_animal_mean",
                      "LED_trial": led, "metric": metric, "n_animals": 6, "histogram_normalizer": np.nan,
                      "full_histogram_area": area, "expected_area": group_audit.expected_area.mean(),
                      "selected_abort_fraction": group_audit.selected_abort_fraction.mean(),
                      "bin_width_s": width_by_metric[metric], "denominator_formula": DENOMINATOR_FORMULAS[position]}
            for field in ("selected_aborts", "finite_aligned_aborts", "missing_aligned_time", "plotted_observations", "eligible_denominator"):
                record[field] = int(group_audit[field].sum())
            panel_records.append(record)
            curve_parts.append(pd.DataFrame({"case": case_id, "row": "equal_animal_mean", "LED_trial": led,
                                             "metric": metric, "bin_left_s": bins[:-1], "bin_right_s": bins[1:],
                                             "height": mean, "sem": sem, "bin_count": np.nan}))
    if position != "first_member":
        # Reproduce all existing unmatched animal curves, means and SEMs exactly.
        reference_case = f"s{st}_{'after_abort' if position == 'second_member' else 'isolated'}"
        reference_index = reference_arrays["cases"].tolist().index(reference_case)
        for metric in METRICS:
            np.testing.assert_array_equal(curves[metric][ci], reference_arrays[metric][reference_index])
            np.testing.assert_array_equal(sems[metric][ci], reference_arrays[f"{metric}_sem"][reference_index])
        expected_denominators = reference_panels.loc[reference_panels.case.eq(reference_case)
                                                     & reference_panels.metric.eq(METRICS[2])]
        for ai, animal in enumerate(ANIMALS):
            for led in (0, 1):
                reference_n = expected_denominators.loc[expected_denominators.row.eq(str(animal))
                                                       & expected_denominators.LED_trial.eq(led), "eligible_denominator"].item()
                assert len(eligible.loc[eligible.animal.eq(animal) & eligible.LED_trial.eq(led)]) == reference_n
    print(f"{case_id}: selected {len(selected):,}; denominator {len(eligible):,}; missing aligned times {len(missing)}", flush=True)


# %%
############ Save new manifests, counts, curves and averaging metadata ############
panels = pd.DataFrame(panel_records)
curve_audit = pd.concat(curve_parts, ignore_index=True)
assert len(panels) == 6 * 7 * 2 * 3
assert not panels.duplicated(["case", "row", "LED_trial", "metric"]).any()
assert not curve_audit.duplicated(["case", "row", "LED_trial", "metric", "bin_left_s"]).any()
for metric in METRICS:
    assert np.isfinite(curves[metric]).all() and (curves[metric] >= 0).all()
    assert np.isfinite(sems[metric]).all() and (sems[metric] >= 0).all()
for case_id, manifest in eligible_manifests.items():
    assert not manifest.duplicated(keys).any() and manifest.source_csv_row_index.is_unique
    manifest.to_csv(f"{OUTPUT_STEM}_{case_id}_eligible_trials.csv", index=False)
pd.concat(selected_manifests, ignore_index=True).to_csv(f"{OUTPUT_STEM}_selected_trials.csv", index=False)
pair_manifest = pd.concat(pair_manifests, ignore_index=True)
assert pair_manifest.second_trial.sub(pair_manifest.first_trial).eq(1).all()
pair_manifest.to_csv(f"{OUTPUT_STEM}_pair_manifest.csv", index=False)
panels.to_csv(f"{OUTPUT_STEM}_count_area_audit.csv", index=False)
mean_panels = panels.loc[panels.row.eq("equal_animal_mean")].copy()
mean_panels.to_csv(f"{OUTPUT_STEM}_mean_area_audit.csv", index=False)
curve_audit.to_csv(f"{OUTPUT_STEM}_curve_audit.csv", index=False)
np.savez_compressed(
    f"{OUTPUT_STEM}_curves.npz", cases=np.array([f"s{st}_{position}" for st, position in CASES]),
    rows=np.array(row_names), led_conditions=np.array([0, 1]), timing_bins=timing_bins, abort_bins=abort_bins,
    **curves, **{f"{metric}_sem": sems[metric] for metric in METRICS},
)


# %%
############ Six equal-animal rows, three columns, no matching ############
fig, axes = plt.subplots(6, 3, figsize=(16, 19), sharex="col", sharey="col")
upper_limits = []
for metric in METRICS:
    bins = bins_by_metric[metric]
    visible = ((bins[:-1] < ABORT_DISPLAY_S[1]) & (bins[1:] > ABORT_DISPLAY_S[0])
               if metric == METRICS[2] else np.ones(len(bins) - 1, dtype=bool))
    upper_limits.append(1.30 * float((curves[metric][:, 6] + sems[metric][:, 6])[..., visible].max()))
for ri, (st, position) in enumerate(CASES):
    case_id = f"s{st}_{position}"
    for col, metric in enumerate(METRICS):
        ax, bins = axes[ri, col], bins_by_metric[metric]
        for led in (0, 1):
            mean, sem = curves[metric][ri, 6, led], sems[metric][ri, 6, led]
            ax.fill_between(bins, np.r_[mean - sem, (mean - sem)[-1]], np.r_[mean + sem, (mean + sem)[-1]],
                            step="post", color=COLORS[led], alpha=0.16, linewidth=0)
            ax.stairs(mean, bins, color=COLORS[led], linewidth=1.55)
            audit = mean_panels.loc[mean_panels.case.eq(case_id) & mean_panels.LED_trial.eq(led)
                                   & mean_panels.metric.eq(metric)].iloc[0]
            label = "OFF" if led == 0 else "ON"
            note = (f"{label} n={int(audit.plotted_observations):,}" if col < 2 else
                    f"{label} n={int(audit.finite_aligned_aborts):,}, N={int(audit.eligible_denominator):,}; area={audit.full_histogram_area:.3f}")
            ax.text(0.98, 0.97 - 0.08 * led, note, transform=ax.transAxes, ha="right", va="top",
                    fontsize=8.5, color=COLORS[led])
        ax.set_xlim(*(ABORT_DISPLAY_S if col == 2 else TIMING_RANGE_S))
        ax.set_ylim(0, upper_limits[col])
        if col == 2:
            ax.axvline(0, color="0.45", linestyle=":", linewidth=1)
            ax.set_ylabel("Conditional abort rate (s⁻¹)", fontsize=10)
        elif col == 0:
            ax.set_ylabel(f"Session type {st}\n{ROW_LABELS[position]}\nDensity (s⁻¹)", fontsize=11)
        if ri == 0:
            ax.set_title(COLUMN_TITLES[col], fontsize=13, pad=12)
        if ri == 5:
            ax.set_xlabel("Time from scheduled LED onset (s)" if col == 2 else "Time from fixation onset (s)", fontsize=11)
        ax.grid(axis="y", color="0.91", linewidth=0.5)
        ax.tick_params(labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
for col in range(3):
    assert len({tuple(axes[ri, col].get_ylim()) for ri in range(6)}) == 1
fig.suptitle("LED7 pair-conditioned abort timing: equal-animal mean ± SEM", fontsize=18, y=0.994)
fig.legend(handles=[Line2D([0], [0], color=COLORS[led], lw=2, label=f"Plotted trial LED {'OFF' if led == 0 else 'ON'}")
                    for led in (0, 1)], loc="upper center", bbox_to_anchor=(0.5, 0.973), ncol=2, frameon=False, fontsize=12)
fig.text(0.5, 0.019,
         "Animals 92, 93, 98, 99, 100, 103. A = event-3 abort; V = success ±1. Denominators: first member AA+VA; second member AA+AV; isolated VAV+VVV.\n"
         "Actual adjacent trial IDs within each animal/session; overlapping AA pairs included. Group by the plotted trial's LED flag, without requiring matching neighboring flags.\n"
         "Timing: 40 ms unit-area histograms. Abort rates: 20 ms bins; full areas over −2 to +2 s. n and N are summed counts; area is the mean of six animal-level n/N fractions.\n"
         "Two s7 second-member OFF and two s8 second-member ON aborts lack timing: retained in classification/denominators, omitted only from aligned histograms.\n"
         "No subsampling or matching. OFF onset is scheduled/counterfactual. Conditioning on future outcomes is descriptive, not causal evidence.",
         ha="center", fontsize=8.5, linespacing=1.45)
fig.tight_layout(rect=(0.008, 0.105, 0.997, 0.947), h_pad=1.8, w_pad=1.7)
figure_path = Path(f"{OUTPUT_STEM}_6x3.png")
fig.savefig(figure_path, dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)

for path in protected_paths:
    with path.open("rb") as handle:
        assert hashlib.file_digest(handle, "sha256").hexdigest() == input_hashes[str(path.relative_to(REPO))]
summary = {
    "source_script": str(Path(__file__).relative_to(REPO)), "figure": str(figure_path.relative_to(REPO)),
    "input_sha256": input_hashes, "old_artifacts_preserved": True,
    "animals": ANIMALS, "case_order": [f"s{st}_{position}" for st, position in CASES],
    "denominator_formulas": DENOMINATOR_FORMULAS,
    "LED_grouping": "Plotted trial's own LED flag; no requirement on neighbor LED flags.",
    "adjacency": "Actual trial +/-1 in same animal/session, established before outcome or LED splitting.",
    "pair_membership": "All overlapping AA pairs; first/second are pair members, not run positions.",
    "onset": "Standardized CSV LED_onset_time, already fixation-referenced. OFF is scheduled/counterfactual.",
    "averaging": "Equal mean of six animal-normalized histograms, sample SD/sqrt(6) SEM; no pooling.",
    "missing_timing": "Classified and included in denominators; omitted only from aligned histograms.",
    "subsampling_or_matching": False,
    "histogram_bin_widths_s": {"timing": TIMING_BIN_WIDTH_S, "abort_rate": ABORT_BIN_WIDTH_S},
    "figure_y_upper_limits": dict(zip(METRICS, upper_limits)),
    "validation": ["Synthetic overlap/LED-switch/gap/boundary/other-outcome/missing-timing checks",
                   "Independent trial-key neighbor lookup", "Exact first/second pair linkage",
                   "Exact prior second-member and isolated trial identities and unmatched curves/SEMs",
                   "Expected counts, finite-time ranges, full histogram areas and independent SEM formula",
                   "Preserved all recorded input and old-artifact hashes"],
    "caveat": "Future-outcome conditioning is descriptive, not causal evidence.",
}
Path(f"{OUTPUT_STEM}_run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(mean_panels.loc[mean_panels.metric.eq(METRICS[2]),
                      ["case", "LED_trial", "selected_aborts", "finite_aligned_aborts", "missing_aligned_time",
                       "eligible_denominator", "full_histogram_area"]].to_string(index=False))
print(f"Saved {figure_path}")
print("Passed: sequence tests, source identities, expected counts, areas, equal-animal means/SEMs and old-file hashes.")
