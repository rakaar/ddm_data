# %%
"""Original complete-run/isolated abort timing, with an LED-to-stimulus gap panel.

CSV LED_onset_time is already fixation referenced. No matching, subsampling,
new fitting, or changes to the original event-3-or-valid denominators.
"""

from pathlib import Path
import hashlib
import json
import os


# %%
############ Editable parameters ############
SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
ORIGINAL_STEM = SCRIPT_DIR / "led7_s7_s8_first_last_abort_timing_all_trials"
OUTPUT_STEM = SCRIPT_DIR / "led7_s7_s8_first_last_abort_timing_stimulus_gap"
ANIMALS = (92, 93, 98, 99, 100, 103)
CASES = tuple((st, position) for position in ("first", "last", "isolated") for st in (7, 8))
TIMING_BIN_WIDTH_S = 0.040
ABORT_BIN_WIDTH_S = 0.020
TIMING_RANGE_S = (0.0, 2.2)
ABORT_RANGE_S = (-2.0, 2.0)
ABORT_DISPLAY_S = (-0.4, 0.5)
FIGURE_DPI = 240
COLORS = {0: "tab:blue", 1: "tab:red"}
METRICS = ("intended_fix", "LED_onset_time", "stimulus_gap", "abort_from_LED_onset")
COLUMN_TITLES = ("intended_fix", "LED_onset_time", "intended_fix − LED_onset_time",
                 "Abort rate aligned to LED onset")
ROW_LABELS = {"first": "First abort in a run", "last": "Last abort in a run",
              "isolated": "Valid → abort → valid"}
EXPECTED_DENOMINATORS = {7: (63525, 30910), 8: (68584, 38486)}
EXPECTED_SELECTED = {(7, "first"): (1425, 956), (8, "first"): (1309, 1284),
                     (7, "last"): (1218, 1163), (8, "last"): (786, 1807),
                     (7, "isolated"): (6498, 3365), (8, "isolated"): (6406, 4105)}

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

case_names = [f"s{st}_{position}" for st, position in CASES]
row_names = [*[str(animal) for animal in ANIMALS], "equal_animal_mean"]
keys = ["animal", "session", "trial"]
timing_bins = np.linspace(*TIMING_RANGE_S, round(2.2 / TIMING_BIN_WIDTH_S) + 1)
abort_bins = np.linspace(*ABORT_RANGE_S, round(4 / ABORT_BIN_WIDTH_S) + 1)
bins_by_metric = dict(zip(METRICS, (timing_bins, timing_bins, timing_bins, abort_bins)))
width_by_metric = dict(zip(METRICS, (TIMING_BIN_WIDTH_S,) * 3 + (ABORT_BIN_WIDTH_S,)))
original_summary = json.loads(Path(f"{ORIGINAL_STEM}_run_summary.json").read_text())
protected_paths = list(SCRIPT_DIR.glob(f"{ORIGINAL_STEM.name}*"))
protected_paths += list(SCRIPT_DIR.glob("led7_s7_s8_first_last_abort_timing_pre_onset_matched*"))
protected_paths += [SCRIPT_DIR / "plot_led7_s7_s8_first_last_abort_timing_all_trials.py",
                    SCRIPT_DIR / "plot_led7_s7_s8_first_last_abort_timing_pre_onset_matched.py"]
protected_paths += [REPO / f"raw_data/LED7_s{st}.csv" for st in (7, 8)]
protected_paths += list((REPO / "docs/assets/results/2026-10-04").glob("led7_s7_s8_first_last_abort_timing_all_trials*.png"))
protected_paths += list((REPO / "docs/assets/results/2026-10-04").glob("led7_s7_s8_first_last_abort_timing_pre_onset_matched*.png"))
input_hashes = {}
for path in protected_paths:
    with path.open("rb") as handle:
        input_hashes[str(path.relative_to(REPO))] = hashlib.file_digest(handle, "sha256").hexdigest()
for st in (7, 8):
    assert input_hashes[f"raw_data/LED7_s{st}.csv"] == original_summary["input_sha256"][f"raw_data/LED7_s{st}.csv"]
with np.load(f"{ORIGINAL_STEM}_curves.npz") as payload:
    original_arrays = {key: payload[key].copy() for key in payload.files}
np.testing.assert_array_equal(original_arrays["cases"], case_names)
np.testing.assert_array_equal(original_arrays["rows"], row_names)
np.testing.assert_array_equal(original_arrays["timing_bins"], timing_bins)
np.testing.assert_array_equal(original_arrays["abort_bins"], abort_bins)
original_selected = pd.read_csv(f"{ORIGINAL_STEM}_selected_trials.csv", float_precision="round_trip")
original_denominators = pd.read_csv(f"{ORIGINAL_STEM}_denominator_trials.csv", float_precision="round_trip")
original_panels = pd.read_csv(f"{ORIGINAL_STEM}_count_area_audit.csv", float_precision="round_trip")


# %%
############ Verify original selections and denominators on intact CSV trial streams ############
eligible_by_session, selected_by_case = {}, {}
for st in (7, 8):
    raw = pd.read_csv(REPO / f"raw_data/LED7_s{st}.csv", float_precision="round_trip")
    raw_columns = list(raw.columns)
    assert raw.shape == ({7: 138440, 8: 148428}[st], 52)
    assert raw.training_level.eq(16).all() and raw.session_type.eq(st).all()
    assert (raw.repeat_trial.isin([0, 2]) | raw.repeat_trial.isna()).all()
    assert raw.LED_trial.isin([0, 1]).all() and not raw.duplicated(keys).any()
    raw["source_csv_row_index"] = np.arange(len(raw))
    data = raw.loc[raw.animal.isin(ANIMALS)].sort_values(keys, kind="stable").copy()
    assert tuple(sorted(data.animal.unique())) == ANIMALS
    ids_by_selection = {position: set() for position in ("first", "last", "isolated")}
    # Scan actual trial IDs, not row numbers, before excluding other outcomes.
    for _, session in data.groupby(["animal", "session"], sort=False):
        trial, ids = session.trial.to_numpy(), session.source_csv_row_index.to_numpy()
        abort, valid = session.abort_event.eq(3).to_numpy(), session.success.isin([-1, 1]).to_numpy()
        i = 0
        while i < len(session):
            if not abort[i]:
                i += 1
                continue
            end = i
            while end + 1 < len(session) and abort[end + 1] and trial[end + 1] == trial[end] + 1:
                end += 1
            complete = (i > 0 and end + 1 < len(session) and trial[i - 1] == trial[i] - 1
                        and trial[end + 1] == trial[end] + 1 and not abort[i - 1] and not abort[end + 1])
            if complete and end > i:
                ids_by_selection["first"].add(ids[i])
                ids_by_selection["last"].add(ids[end])
            if complete and end == i and valid[i - 1] and valid[end + 1]:
                ids_by_selection["isolated"].add(ids[i])
            i = end + 1
    eligible = data.loc[data.abort_event.eq(3) | data.success.isin([-1, 1])].set_index("source_csv_row_index", drop=False)
    saved_denominator = original_denominators.loc[original_denominators.session_type.eq(st)].set_index("source_csv_row_index")
    common_columns = [key for key in raw_columns if key in saved_denominator]
    pd.testing.assert_frame_equal(eligible.sort_index()[common_columns], saved_denominator.sort_index()[common_columns], check_dtype=False)
    assert tuple(eligible.groupby("LED_trial").size()) == EXPECTED_DENOMINATORS[st]
    assert eligible.loc[eligible.abort_event.eq(3), "timed_fix"].isna().sum() == 2
    eligible_by_session[st] = eligible
    for position, selected_ids in ids_by_selection.items():
        saved = original_selected.loc[original_selected.case.eq(f"s{st}_{position}")].copy()
        assert saved.source_csv_row_index.is_unique and set(saved.source_csv_row_index) == selected_ids
        pd.testing.assert_frame_equal(eligible.loc[sorted(selected_ids), raw_columns],
                                      saved.set_index("source_csv_row_index").sort_index()[raw_columns], check_dtype=False)
        assert tuple(saved.groupby("LED_trial").size()) == EXPECTED_SELECTED[(st, position)]
        assert np.isfinite(saved[["intended_fix", "LED_onset_time", "timed_fix"]]).all().all()
        assert saved.LED_onset_time.ge(0).all() and saved.timed_fix.lt(saved.intended_fix).all()
        saved["stimulus_gap"] = saved.intended_fix - saved.LED_onset_time
        np.testing.assert_array_equal(saved.abort_from_LED_onset, saved.timed_fix - saved.LED_onset_time)
        assert saved.stimulus_gap.gt(0).all() and saved.stimulus_gap.between(*TIMING_RANGE_S).all()
        selected_by_case[(st, position)] = saved
    print(f"s{st}: source identities, complete-run endpoints, isolated trials and all-eligible denominators verified.", flush=True)


# %%
############ Animal histograms, descriptive comparisons, then equal-animal mean/SEM ############
curves = {metric: np.zeros((6, 7, 2, len(bins_by_metric[metric]) - 1)) for metric in METRICS}
sems = {metric: np.zeros_like(curves[metric]) for metric in METRICS}
panel_records, comparison_records, curve_parts = [], [], []
for ci, (st, position) in enumerate(CASES):
    case_id = case_names[ci]
    eligible, selected = eligible_by_session[st], selected_by_case[(st, position)]
    for ai, animal in enumerate(ANIMALS):
        for led in (0, 1):
            denominator = eligible.loc[eligible.animal.eq(animal) & eligible.LED_trial.eq(led)]
            chosen = selected.loc[selected.animal.eq(animal) & selected.LED_trial.eq(led)]
            assert len(chosen) > 0 and len(denominator) > 0
            intended, gap = chosen.intended_fix.to_numpy(), chosen.stimulus_gap.to_numpy()
            comparison_records.append({
                "case": case_id, "animal": animal, "LED_trial": led, "n": len(chosen),
                "intended_fix_mean_s": intended.mean(), "intended_fix_median_s": np.median(intended),
                "stimulus_gap_mean_s": gap.mean(), "stimulus_gap_median_s": np.median(gap),
                "mean_difference_s": intended.mean() - gap.mean(),
                "unbinned_KS_distance": ks_2samp(intended, gap).statistic,
                "zero_onset_trials": int(chosen.LED_onset_time.eq(0).sum()),
                "stimulus_gap_min_s": gap.min(), "stimulus_gap_max_s": gap.max(),
            })
            assert np.isclose(intended.mean() - gap.mean(), chosen.LED_onset_time.mean(), rtol=0, atol=1e-12)
            for metric in METRICS:
                bins, width = bins_by_metric[metric], width_by_metric[metric]
                values = chosen[metric].to_numpy()
                values = values[np.isfinite(values)]
                counts = np.histogram(values, bins=bins)[0]
                assert counts.sum() == len(values)
                normalizer = len(denominator) if metric == METRICS[3] else len(values)
                heights = counts / (normalizer * width)
                area = float(heights @ np.diff(bins))
                assert np.isclose(area, len(values) / normalizer, rtol=0, atol=1e-12)
                curves[metric][ci, ai, led] = heights
                panel_records.append({"case": case_id, "row": str(animal), "LED_trial": led, "metric": metric,
                                      "selected_aborts": len(chosen), "plotted_observations": len(values),
                                      "eligible_denominator": len(denominator), "histogram_normalizer": normalizer,
                                      "full_histogram_area": area, "expected_area": len(values) / normalizer,
                                      "bin_width_s": width,
                                      "denominator_missing_timed_fix": int(denominator.timed_fix.isna().sum())})
                curve_parts.append(pd.DataFrame({"case": case_id, "row": str(animal), "LED_trial": led, "metric": metric,
                                                 "bin_left_s": bins[:-1], "bin_right_s": bins[1:],
                                                 "height": heights, "sem": np.nan, "bin_count": counts}))
    animal_panels = pd.DataFrame(panel_records)
    animal_panels = animal_panels.loc[animal_panels.case.eq(case_id)]
    for metric in METRICS:
        bins = bins_by_metric[metric]
        for led in (0, 1):
            matrix = curves[metric][ci, :6, led]
            mean, sem = matrix.mean(axis=0), matrix.std(axis=0, ddof=1) / np.sqrt(6)
            np.testing.assert_allclose(sem, np.sqrt(((matrix - mean) ** 2).sum(axis=0) / 30), atol=1e-14)
            curves[metric][ci, 6, led], sems[metric][ci, 6, led] = mean, sem
            group = animal_panels.loc[animal_panels.metric.eq(metric) & animal_panels.LED_trial.eq(led)]
            assert len(group) == 6
            area = float(mean @ np.diff(bins))
            assert np.isclose(area, group.expected_area.mean(), rtol=0, atol=1e-12)
            record = {"case": case_id, "row": "equal_animal_mean", "LED_trial": led, "metric": metric,
                      "histogram_normalizer": np.nan, "full_histogram_area": area,
                      "expected_area": group.expected_area.mean(), "bin_width_s": width_by_metric[metric]}
            for field in ("selected_aborts", "plotted_observations", "eligible_denominator", "denominator_missing_timed_fix"):
                record[field] = int(group[field].sum())
            panel_records.append(record)
            curve_parts.append(pd.DataFrame({"case": case_id, "row": "equal_animal_mean", "LED_trial": led, "metric": metric,
                                             "bin_left_s": bins[:-1], "bin_right_s": bins[1:],
                                             "height": mean, "sem": sem, "bin_count": np.nan}))
        if metric != "stimulus_gap":
            np.testing.assert_array_equal(curves[metric][ci], original_arrays[metric][ci])
            np.testing.assert_array_equal(sems[metric][ci], original_arrays[f"{metric}_sem"][ci])


# %%
############ Save exact curves, source timing pairs and count/comparison audits ############
panels, comparisons = pd.DataFrame(panel_records), pd.DataFrame(comparison_records)
curve_audit = pd.concat(curve_parts, ignore_index=True)
mean_panels = panels.loc[panels.row.eq("equal_animal_mean")].copy()
assert len(panels) == 6 * 7 * 2 * 4 and len(comparisons) == 6 * 6 * 2
assert not curve_audit.duplicated(["case", "row", "LED_trial", "metric", "bin_left_s"]).any()
assert np.isfinite(curve_audit.height).all() and curve_audit.height.ge(0).all()
check_keys = ["case", "row", "LED_trial", "metric"]
check_fields = ["selected_aborts", "plotted_observations", "eligible_denominator", "full_histogram_area"]
pd.testing.assert_frame_equal(
    panels.loc[panels.metric.ne("stimulus_gap")].set_index(check_keys).sort_index()[check_fields],
    original_panels.set_index(check_keys).sort_index()[check_fields], check_exact=True,
)
selected_values = pd.concat([selected_by_case[case] for case in CASES], ignore_index=True)
selected_values[["case", "source_csv", "source_csv_row_index", *keys, "LED_trial", "intended_fix", "LED_onset_time",
                 "stimulus_gap", "timed_fix", "abort_from_LED_onset", "run_id", "run_position",
                 "observed_run_length", "complete_run"]].to_csv(f"{OUTPUT_STEM}_selected_timing_values.csv", index=False)
panels.to_csv(f"{OUTPUT_STEM}_count_area_audit.csv", index=False)
mean_panels.to_csv(f"{OUTPUT_STEM}_mean_area_audit.csv", index=False)
curve_audit.to_csv(f"{OUTPUT_STEM}_curve_audit.csv", index=False)
comparisons.to_csv(f"{OUTPUT_STEM}_distribution_comparison.csv", index=False)
np.savez_compressed(f"{OUTPUT_STEM}_curves.npz", cases=np.array(case_names), rows=np.array(row_names),
                    led_conditions=np.array([0, 1]), timing_bins=timing_bins, abort_bins=abort_bins,
                    **curves, **{f"{metric}_sem": sems[metric] for metric in METRICS})
# These means have equal animal weights, consistent with the plotted densities.
# The KS ranges are animal-level descriptive distances, not KS of a pooled sample.
comparison_summary = comparisons.groupby(["case", "LED_trial"], sort=False).agg(
    n=("n", "sum"), n_animals=("animal", "nunique"),
    equal_animal_intended_mean_s=("intended_fix_mean_s", "mean"),
    equal_animal_gap_mean_s=("stimulus_gap_mean_s", "mean"),
    animal_KS_min=("unbinned_KS_distance", "min"), animal_KS_max=("unbinned_KS_distance", "max"),
    zero_onset_trials=("zero_onset_trials", "sum"),
).reset_index()
assert comparison_summary.n_animals.eq(6).all()
comparison_summary.to_csv(f"{OUTPUT_STEM}_comparison_summary.csv", index=False)


# %%
############ Plot the new gap panel between onset and the unchanged original rate ############
upper_limits = {metric: original_summary["figure_y_upper_limits"][metric] for metric in METRICS if metric != "stimulus_gap"}
upper_limits["stimulus_gap"] = 1.30 * float((curves["stimulus_gap"][:, 6] + sems["stimulus_gap"][:, 6]).max())
fig, axes = plt.subplots(6, 4, figsize=(21, 19), sharex="col", sharey="col")
for ci, (st, position) in enumerate(CASES):
    case_id = case_names[ci]
    for col, metric in enumerate(METRICS):
        ax, bins = axes[ci, col], bins_by_metric[metric]
        for led in (0, 1):
            mean, sem = curves[metric][ci, 6, led], sems[metric][ci, 6, led]
            ax.fill_between(bins, np.r_[mean - sem, (mean - sem)[-1]], np.r_[mean + sem, (mean + sem)[-1]],
                            step="post", color=COLORS[led], alpha=0.16, linewidth=0)
            ax.stairs(mean, bins, color=COLORS[led], linewidth=1.55)
            audit = mean_panels.loc[mean_panels.case.eq(case_id) & mean_panels.LED_trial.eq(led)
                                   & mean_panels.metric.eq(metric)].iloc[0]
            label = "OFF" if led == 0 else "ON"
            note = (f"{label} n={int(audit.plotted_observations):,}" if col < 3 else
                    f"{label} n={int(audit.plotted_observations):,}, N={int(audit.eligible_denominator):,}; area={audit.full_histogram_area:.3f}")
            ax.text(0.98, 0.97 - 0.08 * led, note, transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=COLORS[led])
        ax.set_xlim(*(ABORT_DISPLAY_S if col == 3 else TIMING_RANGE_S))
        ax.set_ylim(0, upper_limits[metric])
        if col == 0:
            ax.set_ylabel(f"Session type {st}\n{ROW_LABELS[position]}\nDensity (s⁻¹)", fontsize=11)
        elif col == 2:
            ax.set_ylabel("Density (s⁻¹)", fontsize=10)
        elif col == 3:
            ax.axvline(0, color="0.45", linestyle=":", linewidth=1)
            ax.set_ylabel("Abort rate per eligible trial (s⁻¹)", fontsize=10)
        if ci == 0:
            ax.set_title(COLUMN_TITLES[col], fontsize=13, pad=12)
        if ci == 5:
            ax.set_xlabel("Time from fixation onset (s)" if col < 2 else
                          "Scheduled LED-to-stimulus interval (s)" if col == 2 else
                          "Time from scheduled LED onset (s)", fontsize=11)
        ax.grid(axis="y", color="0.91", linewidth=0.5)
        ax.tick_params(labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
for col in range(4):
    assert len({tuple(axes[ci, col].get_ylim()) for ci in range(6)}) == 1
fig.suptitle("LED7 first, last and isolated aborts: timing-gap diagnostic", fontsize=19, y=0.994)
fig.legend(handles=[Line2D([0], [0], color=COLORS[led], lw=2, label=f"Plotted trial LED {'OFF' if led == 0 else 'ON'}")
                    for led in (0, 1)], loc="upper center", bbox_to_anchor=(0.5, 0.973), ncol=2, frameon=False, fontsize=12)
fig.text(0.5, 0.025,
         "Equal-animal mean ± SEM across animals 92, 93, 98, 99, 100, 103. First/last: one endpoint each from complete runs of ≥2 aborts; isolated: valid→abort→valid.\n"
         "The first three columns are selected-abort unit-area densities with 40 ms bins. Column 3 is the scheduled gap from LED onset to stimulus onset, not an alternative onset definition.\n"
         "Column 4: original LED-aligned rates, 20 ms bins; denominator is all event-3 OR success ±1 trials per animal/session-type/LED group. Full areas cover −2 to +2 s.\n"
         "n and N are summed counts; rate areas average six animal fractions. Missing timing stays in denominators; all selected aborts here have finite timing. OFF onset is scheduled/counterfactual.\n"
         "No subsampling, matching or new fitting. The original intended-fixation, onset and abort-rate curves and their y-axis scales are unchanged.",
         ha="center", fontsize=9, linespacing=1.45)
fig.tight_layout(rect=(0.005, 0.108, 0.998, 0.947), h_pad=1.8, w_pad=1.7)
figure_path = Path(f"{OUTPUT_STEM}_6x4.png")
fig.savefig(figure_path, dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)
for path in protected_paths:
    with path.open("rb") as handle:
        assert hashlib.file_digest(handle, "sha256").hexdigest() == input_hashes[str(path.relative_to(REPO))]
summary = {
    "source_script": str(Path(__file__).relative_to(REPO)), "figure": str(figure_path.relative_to(REPO)),
    "input_sha256": input_hashes, "old_artifacts_preserved": True, "case_order": case_names,
    "animals": ANIMALS, "subsampling_or_matching": False, "new_fit": False,
    "gap_definition": "intended_fix - standardized fixation-referenced LED_onset_time; scheduled LED-to-stimulus interval.",
    "denominator": "All abort_event == 3 OR success in {-1,+1} trials, unchanged for all selections.",
    "selection": "Original complete-run first/last endpoints and isolated VAV trials; actual trial-ID adjacency within animal/session.",
    "averaging": "Equal mean of six animal-normalized curves, sample SD/sqrt(6) SEM.",
    "original_curves": "Exact original animal curves, means, SEMs, counts, areas and y-axis scales reproduced.",
    "bin_width_s": {"timing": TIMING_BIN_WIDTH_S, "abort_rate": ABORT_BIN_WIDTH_S},
    "figure_y_upper_limits": upper_limits, "comparison_summary": comparison_summary.to_dict(orient="records"),
    "KS_interpretation": "Descriptive unbinned distance for paired within-animal/LED quantities; no independent-samples significance claim.",
    "exact_matching_constraint": "Because intended_fix - gap equals nonnegative LED onset, exact equality of empirical distributions on intact retained rows requires zero onset throughout.",
    "validation": ["Independent actual-ID sequence scan", "Unchanged selected and denominator source identities",
                   "Finite in-range gap values", "Unit-area densities and original abort fractions",
                   "Unchanged original curves and SEMs", "Equal-animal means and sample-SD/sqrt(6) SEMs",
                   "Input and historical-artifact hashes unchanged"],
}
Path(f"{OUTPUT_STEM}_run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(comparison_summary.to_string(index=False))
print("Passed: original identities/curves/scales, gap histogram areas, rate fractions, equal-animal means/SEMs and old-file hashes.")
print(f"Saved {figure_path}")
