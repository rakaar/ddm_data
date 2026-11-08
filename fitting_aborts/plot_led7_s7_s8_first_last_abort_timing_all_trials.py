# %%
"""First/last aborts in complete runs and isolated aborts, per eligible trial.

Use the standardized CSV onset directly. All rate denominators contain every
event-3 OR success +/-1 trial in the animal/session-type/current-LED group.
Run classification uses the intact CSV stream and actual adjacent trial IDs.
"""

from pathlib import Path
import hashlib
import json
import os


# %%
############ Editable parameters ############
SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
OUTPUT_STEM = SCRIPT_DIR / "led7_s7_s8_first_last_abort_timing_all_trials"
ORIGINAL_MANIFEST = SCRIPT_DIR / "led7_s7_s8_abort_rates_by_sequence_position_sequence_manifest.csv"
REFERENCE_STEM = SCRIPT_DIR / "led7_s7_s8_sequence_abort_pre_onset_matched_summary_with_first"
ANIMALS = (92, 93, 98, 99, 100, 103)
SESSION_TYPES = (7, 8)
POSITIONS = ("first", "last", "isolated")
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
ROW_LABELS = {"first": "First abort in a run", "last": "Last abort in a run",
              "isolated": "Valid → abort → valid"}
DENOMINATOR_DESCRIPTION = "all abort_event == 3 OR success in {-1, +1} trials"
EXPECTED_DENOMINATORS = {7: {0: 63525, 1: 30910}, 8: {0: 68584, 1: 38486}}
EXPECTED_SELECTED = {
    (7, "first"): {0: 1425, 1: 956}, (8, "first"): {0: 1309, 1: 1284},
    (7, "last"): {0: 1218, 1: 1163}, (8, "last"): {0: 786, 1: 1807},
    (7, "isolated"): {0: 6498, 1: 3365}, (8, "isolated"): {0: 6406, 1: 4105},
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

protected_paths = [REPO / f"raw_data/LED7_s{st}.csv" for st in SESSION_TYPES]
protected_paths += [ORIGINAL_MANIFEST, Path(f"{REFERENCE_STEM}_curves.npz"),
                    SCRIPT_DIR / "plot_led7_s7_s8_abort_rates_by_sequence_position.py",
                    SCRIPT_DIR / "plot_led7_s7_s8_pair_conditioned_sequence_abort_timing.py"]
protected_paths += list(SCRIPT_DIR.glob("led7_s7_s8_pair_conditioned_sequence_abort_timing*"))
protected_paths += list(SCRIPT_DIR.glob("led7_s*_abort_rates_by_sequence_position_8x3.png"))
protected_paths += list((REPO / "docs/assets/results/2026-10-04").glob("led7_s7_s8_pair_conditioned*"))
input_hashes = {}
for path in protected_paths:
    with path.open("rb") as handle:
        input_hashes[str(path.relative_to(REPO))] = hashlib.file_digest(handle, "sha256").hexdigest()
old_manifest = pd.read_csv(ORIGINAL_MANIFEST, float_precision="round_trip")
with np.load(f"{REFERENCE_STEM}_curves.npz") as payload:
    reference_arrays = {name: payload[name].copy() for name in payload.files}


# %%
############ Reconstruct complete runs before applying the denominator filter ############
streams, selections, run_parts, denominator_parts = {}, {}, [], []
for st in SESSION_TYPES:
    source = pd.read_csv(REPO / f"raw_data/LED7_s{st}.csv", float_precision="round_trip")
    assert source.shape == ({7: 138440, 8: 148428}[st], 52)
    assert source.training_level.eq(16).all() and source.session_type.eq(st).all()
    assert (source.repeat_trial.isin([0, 2]) | source.repeat_trial.isna()).all()
    assert source.LED_trial.isin([0, 1]).all() and not source.duplicated(keys).any()
    source["source_csv_row_index"] = np.arange(len(source))
    source["source_csv"] = f"raw_data/LED7_s{st}.csv"
    data = source.loc[source.animal.isin(ANIMALS)].sort_values(keys, kind="stable").copy()
    assert len(data) == {7: 100251, 8: 113941}[st]
    assert tuple(sorted(data.animal.unique())) == ANIMALS
    for field in [*keys, "LED_trial"]:
        assert data[field].notna().all() and data[field].eq(np.floor(data[field])).all()
        data[field] = data[field].astype(int)
    grouped = data.groupby(["animal", "session"], sort=False)
    neighbor_fields = ["trial", "abort_event", "success", "LED_trial", "source_csv_row_index"]
    previous, following = grouped[neighbor_fields].shift(1), grouped[neighbor_fields].shift(-1)
    for name, neighbor, direction in (("previous", previous, -1), ("next", following, 1)):
        for field in neighbor_fields:
            data[f"{name}_{field}"] = neighbor[field]
        data[f"{name}_adjacent"] = neighbor.trial.eq(data.trial + direction)
    is_abort, is_valid = data.abort_event.eq(3), data.success.isin([-1, 1])
    assert not (is_abort & is_valid).any()
    data["eligible_denominator_trial"] = is_abort | is_valid
    data["abort_from_LED_onset"] = data.timed_fix - data.LED_onset_time

    # An abort begins a new run whenever the exact previous ID is not an abort.
    run_numbers = (is_abort & ~(data.previous_adjacent & previous.abort_event.eq(3))).cumsum()
    aborts = data.loc[is_abort].copy()
    aborts["run_number"] = run_numbers.loc[aborts.index]
    runs = aborts.groupby("run_number", sort=False)
    aborts["run_position"] = runs.cumcount() + 1
    aborts["observed_run_length"] = runs.trial.transform("size")
    aborts["run_start_trial"] = runs.trial.transform("min")
    aborts["run_end_trial"] = runs.trial.transform("max")
    aborts["run_id"] = ("s" + str(st) + "_a" + aborts.animal.astype(str)
                        + "_session" + aborts.session.astype(str) + "_trial" + aborts.run_start_trial.astype(str))
    first = aborts.loc[aborts.run_position.eq(1)].set_index("run_number")
    last = aborts.loc[aborts.run_position.eq(aborts.observed_run_length)].set_index("run_number")
    run_audit = first[["source_csv", "run_id", "session_type", "animal", "session", "observed_run_length",
                       "run_start_trial", "run_end_trial", "previous_trial", "previous_abort_event", "previous_success"]].copy()
    for field in ("next_trial", "next_abort_event", "next_success"):
        run_audit[field] = last[field]
    run_audit["complete"] = (first.previous_adjacent & last.next_adjacent
                             & first.previous_abort_event.ne(3) & last.next_abort_event.ne(3))
    run_audit["valid_neighbors"] = first.previous_success.isin([-1, 1]) & last.next_success.isin([-1, 1])
    run_audit["first_source_csv_row_index"] = first.source_csv_row_index
    run_audit["last_source_csv_row_index"] = last.source_csv_row_index
    run_audit["first_LED_trial"] = first.LED_trial
    run_audit["last_LED_trial"] = last.LED_trial
    run_audit["complete_multi"] = run_audit.complete & run_audit.observed_run_length.ge(2)
    run_audit["isolated"] = run_audit.complete & run_audit.observed_run_length.eq(1) & run_audit.valid_neighbors
    assert run_audit.complete_multi.sum() == {7: 2381, 8: 2593}[st]
    assert (~run_audit.complete).sum() == 84
    assert (aborts.run_end_trial - aborts.run_start_trial + 1).eq(aborts.observed_run_length).all()
    assert aborts.groupby("run_number").trial.diff().dropna().eq(1).all()
    aborts["complete_run"] = aborts.run_number.map(run_audit.complete)
    for field in ("previous_trial", "next_trial", "previous_abort_event", "next_abort_event"):
        aborts[f"run_boundary_{field}"] = aborts.run_number.map(run_audit[field])
    multi = aborts.run_number.map(run_audit.complete_multi)
    aborts["selection"] = np.select(
        [multi & aborts.run_position.eq(1), multi & aborts.run_position.eq(aborts.observed_run_length),
         aborts.run_number.map(run_audit.isolated)], ["first", "last", "isolated"], default="unselected",
    )

    # Verify endpoints with an independent exact-key lookup on the intact stream.
    keyed = data.set_index(keys)
    complete = run_audit.loc[run_audit.complete]
    before_keys = pd.MultiIndex.from_arrays([complete.animal, complete.session, complete.run_start_trial - 1], names=keys)
    after_keys = pd.MultiIndex.from_arrays([complete.animal, complete.session, complete.run_end_trial + 1], names=keys)
    before, after = keyed.reindex(before_keys), keyed.reindex(after_keys)
    assert before.source_csv_row_index.notna().all() and after.source_csv_row_index.notna().all()
    assert before.abort_event.ne(3).all() and after.abort_event.ne(3).all()
    for position in POSITIONS:
        selected = aborts.loc[aborts.selection.eq(position)].copy()
        saved = old_manifest.loc[old_manifest.session_type.eq(st) & old_manifest.selection.eq(position)]
        selected_by_id = selected.set_index("source_csv_row_index").sort_index()
        saved_by_id = saved.set_index("source_csv_row_index").sort_index()
        fields = [*keys, "LED_trial", "abort_event", "success", "timed_fix", "intended_fix", "LED_onset_time",
                  "run_id", "run_position", "observed_run_length", "run_start_trial", "run_end_trial", "complete_run"]
        pd.testing.assert_frame_equal(selected_by_id[fields], saved_by_id[fields], check_dtype=False)
        assert selected.run_id.is_unique
        assert np.isfinite(selected[["intended_fix", "LED_onset_time", "timed_fix"]]).all().all()
        assert selected.timed_fix.lt(selected.intended_fix).all()
        for led in (0, 1):
            assert selected.LED_trial.eq(led).sum() == EXPECTED_SELECTED[(st, position)][led]
        if position == "isolated":
            assert selected.previous_success.isin([-1, 1]).all() and selected.next_success.isin([-1, 1]).all()
        selections[(st, position)] = selected
        data[f"selected_{position}"] = data.source_csv_row_index.isin(selected.source_csv_row_index)
    assert set(selections[(st, "first")].run_id) == set(selections[(st, "last")].run_id)

    eligible = data.loc[data.eligible_denominator_trial].copy()
    assert eligible.groupby("LED_trial").size().to_dict() == EXPECTED_DENOMINATORS[st]
    assert eligible.loc[eligible.abort_event.eq(3), "timed_fix"].isna().sum() == 2
    streams[st] = eligible
    denominator_parts.append(eligible[["source_csv", "source_csv_row_index", "session_type", *keys,
                                      "training_level", "repeat_trial", "LED_trial", "abort_event", "success",
                                      "timed_fix", "intended_fix", "LED_onset_time", "eligible_denominator_trial",
                                      "selected_first", "selected_last", "selected_isolated"]])
    run_parts.append(run_audit.reset_index(drop=True))


# %%
############ Same denominator for first, last and isolated; then equal-animal means ############
row_names = [*[str(animal) for animal in ANIMALS], "equal_animal_mean"]
curves = {metric: np.zeros((len(CASES), 7, 2, len(bins_by_metric[metric]) - 1)) for metric in METRICS}
sems = {metric: np.zeros_like(curves[metric]) for metric in METRICS}
panel_records, curve_parts, selected_parts = [], [], []
for ci, (st, position) in enumerate(CASES):
    case_id = f"s{st}_{position}"
    eligible, selected = streams[st], selections[(st, position)]
    selected = selected.drop(columns="run_number").copy()
    selected.insert(0, "case", case_id)
    selected_parts.append(selected)
    for ai, animal in enumerate(ANIMALS):
        for led in (0, 1):
            denominator = eligible.loc[eligible.animal.eq(animal) & eligible.LED_trial.eq(led)]
            chosen = selected.loc[selected.animal.eq(animal) & selected.LED_trial.eq(led)]
            assert len(chosen) > 0 and len(denominator) > 0
            for metric in METRICS:
                bins, width = bins_by_metric[metric], width_by_metric[metric]
                values = chosen[metric].to_numpy()
                values = values[np.isfinite(values)]
                counts = np.histogram(values, bins=bins)[0]
                assert len(values) > 0 and counts.sum() == len(values)
                normalizer = len(denominator) if metric == METRICS[2] else len(values)
                heights = counts / (normalizer * width)
                area = float(heights @ np.diff(bins))
                expected_area = len(values) / normalizer
                assert np.isclose(area, expected_area, rtol=0, atol=1e-12)
                curves[metric][ci, ai, led] = heights
                panel_records.append({
                    "case": case_id, "session_type": st, "position": position, "row": str(animal),
                    "LED_trial": led, "metric": metric, "n_animals": 1, "selected_aborts": len(chosen),
                    "finite_aligned_aborts": int(np.isfinite(chosen.abort_from_LED_onset).sum()),
                    "missing_selected_timing": int((~np.isfinite(chosen.abort_from_LED_onset)).sum()),
                    "plotted_observations": len(values), "eligible_denominator": len(denominator),
                    "denominator_missing_timed_fix": int(denominator.timed_fix.isna().sum()),
                    "histogram_normalizer": normalizer, "full_histogram_area": area,
                    "expected_area": expected_area, "bin_width_s": width,
                    "denominator_definition": DENOMINATOR_DESCRIPTION,
                })
                curve_parts.append(pd.DataFrame({"case": case_id, "row": str(animal), "LED_trial": led,
                                                 "metric": metric, "bin_left_s": bins[:-1], "bin_right_s": bins[1:],
                                                 "height": heights, "sem": np.nan, "bin_count": counts}))
    animal_audit = pd.DataFrame(panel_records)
    animal_audit = animal_audit.loc[animal_audit.case.eq(case_id)]
    for led in (0, 1):
        for metric in METRICS:
            bins = bins_by_metric[metric]
            matrix = curves[metric][ci, :6, led]
            mean, sem = matrix.mean(axis=0), matrix.std(axis=0, ddof=1) / np.sqrt(6)
            np.testing.assert_allclose(sem, np.sqrt(((matrix - mean) ** 2).sum(axis=0) / 30), atol=1e-14)
            curves[metric][ci, 6, led], sems[metric][ci, 6, led] = mean, sem
            group_audit = animal_audit.loc[animal_audit.LED_trial.eq(led) & animal_audit.metric.eq(metric)]
            assert len(group_audit) == 6
            area = float(mean @ np.diff(bins))
            assert np.isclose(area, group_audit.expected_area.mean(), rtol=0, atol=1e-12)
            record = {"case": case_id, "session_type": st, "position": position, "row": "equal_animal_mean",
                      "LED_trial": led, "metric": metric, "n_animals": 6, "histogram_normalizer": np.nan,
                      "full_histogram_area": area, "expected_area": group_audit.expected_area.mean(),
                      "bin_width_s": width_by_metric[metric], "denominator_definition": DENOMINATOR_DESCRIPTION}
            for field in ("selected_aborts", "finite_aligned_aborts", "missing_selected_timing", "plotted_observations",
                          "eligible_denominator", "denominator_missing_timed_fix"):
                record[field] = int(group_audit[field].sum())
            panel_records.append(record)
            curve_parts.append(pd.DataFrame({"case": case_id, "row": "equal_animal_mean", "LED_trial": led,
                                             "metric": metric, "bin_left_s": bins[:-1], "bin_right_s": bins[1:],
                                             "height": mean, "sem": sem, "bin_count": np.nan}))
    # Earlier summary already used these exact all-eligible denominators for first/isolated.
    if position in ("first", "isolated"):
        reference_index = reference_arrays["cases"].tolist().index(case_id)
        for metric in METRICS:
            np.testing.assert_array_equal(curves[metric][ci], reference_arrays[metric][reference_index])
            np.testing.assert_array_equal(sems[metric][ci], reference_arrays[f"{metric}_sem"][reference_index])
    print(f"{case_id}: {len(selected):,} selected; {len(eligible):,} denominator trials; selected timing finite.", flush=True)


# %%
############ Save new source identities, run audit and exact histogram arrays ############
panels = pd.DataFrame(panel_records)
curve_audit = pd.concat(curve_parts, ignore_index=True)
denominators = pd.concat(denominator_parts, ignore_index=True)
selected_manifest = pd.concat(selected_parts, ignore_index=True)
run_audit = pd.concat(run_parts, ignore_index=True)
assert not denominators.duplicated(["session_type", *keys]).any()
assert not selected_manifest.duplicated(["case", *keys]).any()
assert len(panels) == 6 * 7 * 2 * 3
assert panels.groupby(["session_type", "row", "LED_trial"]).eligible_denominator.nunique().eq(1).all()
assert not curve_audit.duplicated(["case", "row", "LED_trial", "metric", "bin_left_s"]).any()
for metric in METRICS:
    assert np.isfinite(curves[metric]).all() and (curves[metric] >= 0).all()
    assert np.isfinite(sems[metric]).all() and (sems[metric] >= 0).all()
selected_manifest.to_csv(f"{OUTPUT_STEM}_selected_trials.csv", index=False)
denominators.to_csv(f"{OUTPUT_STEM}_denominator_trials.csv", index=False)
run_audit.to_csv(f"{OUTPUT_STEM}_run_audit.csv", index=False)
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
############ Six rows: first, last and isolated, each s7 then s8 ############
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
            ax.set_ylabel("Abort rate per eligible trial (s⁻¹)", fontsize=10)
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
fig.suptitle("LED7 first, last and isolated aborts: equal-animal mean ± SEM", fontsize=18, y=0.994)
fig.legend(handles=[Line2D([0], [0], color=COLORS[led], lw=2, label=f"Plotted trial LED {'OFF' if led == 0 else 'ON'}")
                    for led in (0, 1)], loc="upper center", bbox_to_anchor=(0.5, 0.973), ncol=2, frameon=False, fontsize=12)
fig.text(0.5, 0.021,
         "Animals 92, 93, 98, 99, 100, 103. Every rate denominator contains all event-3 OR success ±1 trials in that animal/session-type/LED group.\n"
         "First and last: one endpoint each from complete runs of ≥2 consecutive aborts, with observed non-event-3 trials on both sides. Isolated: valid→abort→valid.\n"
         "Timing distributions: 40 ms bins, unit area. Abort rates: 20 ms bins, full areas over −2 to +2 s. n and N are summed counts; area averages six animal-level n/N fractions.\n"
         "The full rate area measures sequence-category frequency among all eligible trials. Missing timing stays in denominators; all selected first/last/isolated aborts here have finite timing.\n"
         "Actual trial-ID adjacency within animal/session. LED flag belongs to the plotted trial. OFF onset is scheduled/counterfactual. No subsampling or matching.",
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
    "input_sha256": input_hashes, "old_artifacts_preserved": True, "animals": ANIMALS,
    "case_order": [f"s{st}_{position}" for st, position in CASES],
    "denominator": DENOMINATOR_DESCRIPTION,
    "sequence_definition": "Maximal event-3 runs, adjacent trial IDs in same animal/session, observed non-event-3 boundaries at both ends.",
    "selection": "First and last of complete runs of length >=2; isolated event-3 trials have adjacent success +/-1 neighbors.",
    "LED_grouping": "Plotted trial's own LED flag; neighboring flags unrestricted.",
    "onset": "Standardized CSV LED_onset_time, already fixation-referenced; OFF is scheduled/counterfactual.",
    "averaging": "Equal mean of six animal-normalized curves, sample SD/sqrt(6) SEM.",
    "histogram_bin_widths_s": {"timing": TIMING_BIN_WIDTH_S, "abort_rate": ABORT_BIN_WIDTH_S},
    "figure_y_upper_limits": dict(zip(METRICS, upper_limits)),
    "missing_timing": "Retained in denominators and classification. All selected first/last/isolated aborts in this source have finite timing.",
    "subsampling_or_matching": False,
    "interpretation": "Full rate area is category frequency among all eligible trials, averaged equally across animals.",
    "validation": ["Exact original selected source identities", "One first/last per complete run",
                   "Actual-ID adjacency and non-event-3 run boundaries", "Expected selected and denominator counts",
                   "Identical denominators across first/last/isolated", "Full histogram integrals and means/SEMs",
                   "Exact prior first/isolated all-eligible curves and SEMs", "Old artifact hashes unchanged"],
}
Path(f"{OUTPUT_STEM}_run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(mean_panels.loc[mean_panels.metric.eq(METRICS[2]),
                      ["case", "LED_trial", "selected_aborts", "eligible_denominator", "full_histogram_area"]].to_string(index=False))
print(f"Saved {figure_path}")
print("Passed: original selections, run boundaries, shared denominators, histogram areas, means/SEMs, and old-file hashes.")
