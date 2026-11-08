# %%
"""Two intact-row selection experiments: intended fixation versus stimulus gap.

The KS bound applies within each animal and LED condition, not between OFF/ON.
Sequence classifications are frozen on the original trial streams. No fits,
abort-time matching, sequence reconstruction or changes to historical outputs.
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
GAP_STEM = SCRIPT_DIR / "led7_s7_s8_first_last_abort_timing_stimulus_gap"
OUTPUT_STEM = SCRIPT_DIR / "led7_s7_s8_first_last_abort_intended_gap_matching"
ANIMALS = (92, 93, 98, 99, 100, 103)
CASES = tuple((st, position) for position in ("first", "last", "isolated") for st in (7, 8))
SEED = 20261004
KS_INVERSE = 5                       # Exact bound 1/5 = 0.20.
SPARSE_SELECTION_THRESHOLD = 10      # Flag only; never exclude or constrain outcomes.
TIMING_BIN_WIDTH_S = 0.040
ABORT_BIN_WIDTH_S = 0.020
TIMING_RANGE_S = (0.0, 2.2)
ABORT_RANGE_S = (-2.0, 2.0)
ABORT_DISPLAY_S = (-0.4, 0.5)
FIGURE_DPI = 240
COLORS = {0: "tab:blue", 1: "tab:red"}
STATES = ("original", "selected_matched", "eligible_matched")
METRICS = ("intended_fix", "LED_onset_time", "stimulus_gap", "abort_from_LED_onset")
COLUMN_TITLES = ("intended_fix", "LED_onset_time", "intended_fix − LED_onset_time",
                 "Abort rate aligned to LED onset")
ROW_LABELS = {"first": "First abort", "last": "Last abort", "isolated": "Valid → abort → valid"}
EXPECTED_DENOMINATORS = {7: (63525, 30910), 8: (68584, 38486)}
EXPECTED_SELECTED = {(7, "first"): (1425, 956), (8, "first"): (1309, 1284),
                     (7, "last"): (1218, 1163), (8, "last"): (786, 1807),
                     (7, "isolated"): (6498, 3365), (8, "isolated"): (6406, 4105)}
EXPECTED_SELECTED_MATCHED = {(7, "first"): (85, 76), (8, "first"): (92, 239),
                             (7, "last"): (120, 145), (8, "last"): (62, 389),
                             (7, "isolated"): (453, 325), (8, "isolated"): (578, 853)}
EXPECTED_ELIGIBLE_MATCHED = {7: (17779, 8733), 8: (19403, 10694)}

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from led7_intended_gap_matching_utils import match_intended_gap, paired_ks

case_names = [f"s{st}_{position}" for st, position in CASES]
row_names = [*[str(animal) for animal in ANIMALS], "equal_animal_mean"]
keys = ["animal", "session", "trial"]
timing_bins = np.linspace(*TIMING_RANGE_S, round(2.2 / TIMING_BIN_WIDTH_S) + 1)
abort_bins = np.linspace(*ABORT_RANGE_S, round(4 / ABORT_BIN_WIDTH_S) + 1)
bins_by_metric = dict(zip(METRICS, (timing_bins, timing_bins, timing_bins, abort_bins)))
width_by_metric = dict(zip(METRICS, (TIMING_BIN_WIDTH_S,) * 3 + (ABORT_BIN_WIDTH_S,)))


# %%
############ Verify source hashes, saved identities and actual-ID sequence definitions ############
protected_paths = []
for stem in (ORIGINAL_STEM, GAP_STEM, SCRIPT_DIR / "led7_s7_s8_first_last_abort_timing_pre_onset_matched"):
    protected_paths.extend(SCRIPT_DIR.glob(f"{stem.name}*"))
protected_paths.extend(REPO / f"raw_data/LED7_s{st}.csv" for st in (7, 8))
protected_paths.extend((REPO / "docs/assets/results/2026-10-04").glob("*.png"))
# Exclude this script's published figures on a subsequent run.
protected_paths = [p for p in protected_paths if not p.name.startswith(OUTPUT_STEM.name)]
input_hashes = {}
for path in protected_paths:
    with path.open("rb") as handle:
        input_hashes[str(path.relative_to(REPO))] = hashlib.file_digest(handle, "sha256").hexdigest()
source_summary = json.loads(Path(f"{ORIGINAL_STEM}_run_summary.json").read_text())
for st in (7, 8):
    key = f"raw_data/LED7_s{st}.csv"
    assert input_hashes[key] == source_summary["input_sha256"][key]
original_denominators = pd.read_csv(f"{ORIGINAL_STEM}_denominator_trials.csv", float_precision="round_trip")
original_selected = pd.read_csv(f"{GAP_STEM}_selected_timing_values.csv", float_precision="round_trip")
original_panels = pd.read_csv(f"{GAP_STEM}_count_area_audit.csv", float_precision="round_trip")
with np.load(f"{GAP_STEM}_curves.npz") as payload:
    original_arrays = {key: payload[key].copy() for key in payload.files}
np.testing.assert_array_equal(original_arrays["cases"], case_names)
np.testing.assert_array_equal(original_arrays["rows"], row_names)
np.testing.assert_array_equal(original_arrays["timing_bins"], timing_bins)
np.testing.assert_array_equal(original_arrays["abort_bins"], abort_bins)
validation_path = Path(f"{OUTPUT_STEM}_optimizer_validation.csv")
optimizer_validation = pd.read_csv(validation_path)
assert optimizer_validation.passed.all() and len(optimizer_validation) == 106

eligible_by_session = {}
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
    complete_runs = 0
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
                complete_runs += 1
            if complete and end == i and valid[i - 1] and valid[end + 1]:
                ids_by_selection["isolated"].add(ids[i])
            i = end + 1
    assert complete_runs == {7: 2381, 8: 2593}[st]
    eligible = data.loc[data.abort_event.eq(3) | data.success.isin([-1, 1])].set_index("source_csv_row_index", drop=False)
    saved = original_denominators.loc[original_denominators.session_type.eq(st)].set_index("source_csv_row_index")
    common = [column for column in raw_columns if column in saved]
    pd.testing.assert_frame_equal(eligible.sort_index()[common], saved.sort_index()[common], check_dtype=False)
    assert tuple(eligible.groupby("LED_trial").size()) == EXPECTED_DENOMINATORS[st]
    for position, selected_ids in ids_by_selection.items():
        original = original_selected.loc[original_selected.case.eq(f"s{st}_{position}")].set_index("source_csv_row_index")
        assert original.index.is_unique and set(original.index) == selected_ids
        columns = [*keys, "LED_trial", "intended_fix", "LED_onset_time", "timed_fix"]
        pd.testing.assert_frame_equal(eligible.loc[sorted(selected_ids), columns], original.sort_index()[columns], check_dtype=False)
        eligible[f"selected_{position}"] = eligible.index.isin(selected_ids)
        np.testing.assert_array_equal(eligible.sort_index()[f"selected_{position}"], saved.sort_index()[f"selected_{position}"])
        assert tuple(original.groupby("LED_trial").size()) == EXPECTED_SELECTED[(st, position)]
    # Keep only analysis/provenance fields in new manifests; raw CSVs retain all 52 columns.
    columns = ["source_csv_row_index", "session_type", *keys, "training_level", "repeat_trial", "LED_trial",
               "abort_event", "success", "intended_fix", "LED_onset_time", "timed_fix",
               "selected_first", "selected_last", "selected_isolated"]
    eligible = eligible[columns].copy()
    eligible["source_csv"] = f"raw_data/LED7_s{st}.csv"
    eligible["stimulus_gap"] = eligible.intended_fix - eligible.LED_onset_time
    eligible["abort_from_LED_onset"] = eligible.timed_fix - eligible.LED_onset_time
    assert np.isfinite(eligible[["intended_fix", "LED_onset_time", "stimulus_gap"]]).all().all()
    assert eligible.LED_onset_time.ge(0).all() and eligible.stimulus_gap.ge(0).all()
    finite_aborts = eligible.loc[eligible.abort_event.eq(3) & eligible.timed_fix.notna()]
    assert finite_aborts.timed_fix.lt(finite_aborts.intended_fix).all()
    assert eligible.loc[eligible.abort_event.eq(3), "timed_fix"].isna().sum() == 2
    eligible_by_session[st] = eligible
    print(f"s{st}: original identities, complete runs, isolated trials and denominators verified.", flush=True)


# %%
############ Maximum-retention matching: scheduled timing only, intact rows ############
optimizer_records, selected_keep, population_keep = [], {}, {}
rng = np.random.default_rng(SEED)
selected_groups = []
for st, position in CASES:
    group = eligible_by_session[st].loc[eligible_by_session[st][f"selected_{position}"]].copy()
    group["case"] = f"s{st}_{position}"
    selected_groups.append(group.reset_index(drop=True))
selected_groups = pd.concat(selected_groups, ignore_index=True)
for (case, animal, led), group in selected_groups.groupby(["case", "animal", "LED_trial"], sort=True):
    ordered = group.sort_values("source_csv_row_index").iloc[rng.permutation(len(group))]
    keep, audit = match_intended_gap(ordered.intended_fix.to_numpy(), ordered.stimulus_gap.to_numpy(), KS_INVERSE)
    chosen = ordered.loc[keep]
    assert len(chosen) > 0 and chosen.source_csv_row_index.is_unique
    ks_method = "exact" if len(chosen) == 1 else "asymp"  # Only the distance is used, never its p-value.
    assert np.isclose(audit["ks_after"], ks_2samp(chosen.intended_fix, chosen.stimulus_gap, method=ks_method).statistic, atol=1e-14)
    selected_keep[(case, int(animal), int(led))] = set(chosen.source_csv_row_index)
    optimizer_records.append({"analysis": "selected_matched", "case": case, "session_type": int(group.session_type.iloc[0]),
                              "animal": int(animal), "LED_trial": int(led), **audit})

# Reset tie ordering independently for the all-eligible analysis.
rng = np.random.default_rng(SEED)
eligible_groups = pd.concat([eligible_by_session[st].reset_index(drop=True) for st in (7, 8)], ignore_index=True)
for (st, animal, led), group in eligible_groups.groupby(["session_type", "animal", "LED_trial"], sort=True):
    ordered = group.sort_values("source_csv_row_index").iloc[rng.permutation(len(group))]
    keep, audit = match_intended_gap(ordered.intended_fix.to_numpy(), ordered.stimulus_gap.to_numpy(), KS_INVERSE)
    chosen = ordered.loc[keep]
    assert len(chosen) > 0 and chosen.source_csv_row_index.is_unique
    assert np.isclose(audit["ks_after"], ks_2samp(chosen.intended_fix, chosen.stimulus_gap, method="asymp").statistic, atol=1e-14)
    population_keep[(int(st), int(animal), int(led))] = set(chosen.source_csv_row_index)
    optimizer_records.append({"analysis": "eligible_matched", "case": "all_categories", "session_type": int(st),
                              "animal": int(animal), "LED_trial": int(led), **audit})
optimizer_audit = pd.DataFrame(optimizer_records)
assert len(optimizer_audit) == 96 and optimizer_audit.maximum_retention.all()
assert optimizer_audit.capacity_boundary_verified.all() and optimizer_audit.ks_after.le(1 / KS_INVERSE + 1e-12).all()
optimizer_audit.to_csv(f"{OUTPUT_STEM}_optimization_audit.csv", index=False)


# %%
############ Freeze classifications; save kept/removed manifests and recompute denominators ############
retained_by_case = {}
retention_records = []
for st in (7, 8):
    eligible = eligible_by_session[st]
    ids = set().union(*(population_keep[(st, animal, led)] for animal in ANIMALS for led in (0, 1)))
    population = eligible.copy()
    population["retained"] = population.index.isin(ids)
    assert population.index.is_unique and len(ids) == population.retained.sum()
    population.to_csv(f"{OUTPUT_STEM}_eligible_s{st}_kept_removed_rows.csv", index=False)
    population = population.loc[population.retained].drop(columns="retained")
    assert tuple(population.groupby("LED_trial").size()) == EXPECTED_ELIGIBLE_MATCHED[st]
    for position in ("first", "last", "isolated"):
        case, flag = f"s{st}_{position}", f"selected_{position}"
        chosen_ids = set().union(*(selected_keep[(case, animal, led)] for animal in ANIMALS for led in (0, 1)))
        selected_manifest = eligible.copy()
        selected_manifest["case"] = case
        selected_manifest["selected_in_case"] = selected_manifest[flag]
        selected_manifest["retained"] = ~selected_manifest[flag] | selected_manifest.index.isin(chosen_ids)
        removed = selected_manifest.loc[~selected_manifest.retained]
        assert removed[flag].all() and removed.abort_event.eq(3).all()
        assert selected_manifest.loc[~selected_manifest[flag], "retained"].all()
        selected_manifest.to_csv(f"{OUTPUT_STEM}_{case}_selected_kept_removed_rows.csv", index=False)
        matched = eligible.loc[selected_manifest.retained]
        assert len(matched) == len(eligible) - len(removed)
        assert tuple(matched.loc[matched[flag]].groupby("LED_trial").size()) == EXPECTED_SELECTED_MATCHED[(st, position)]
        retained_by_case[("original", case)] = eligible
        retained_by_case[("selected_matched", case)] = matched
        retained_by_case[("eligible_matched", case)] = population
        for state in STATES:
            retained = retained_by_case[(state, case)]
            pd.testing.assert_frame_equal(retained, eligible.loc[retained.index], check_exact=True)
            for led in (0, 1):
                denominator = retained.loc[retained.LED_trial.eq(led)]
                chosen = denominator.loc[denominator[flag]]
                per_animal = chosen.groupby("animal").size().reindex(ANIMALS, fill_value=0)
                assert per_animal.gt(0).all() and denominator.index.is_unique
                retention_records.append({"state": state, "case": case, "LED_trial": led,
                                          "original_selected": EXPECTED_SELECTED[(st, position)][led],
                                          "retained_selected": len(chosen), "original_denominator": EXPECTED_DENOMINATORS[st][led],
                                          "retained_denominator": len(denominator), "n_animals": len(per_animal),
                                          "min_animal_selected": int(per_animal.min()),
                                          "sparse_animal_groups": int(per_animal.lt(SPARSE_SELECTION_THRESHOLD).sum())})
    # The very same eligible subset is used in all three category rows.
    for position in ("last", "isolated"):
        pd.testing.assert_frame_equal(retained_by_case[("eligible_matched", f"s{st}_first")],
                                      retained_by_case[("eligible_matched", f"s{st}_{position}")], check_exact=True)
retention_audit = pd.DataFrame(retention_records)
assert retention_audit.loc[retention_audit.state.eq("selected_matched"), "retained_selected"].sum() == 3417
assert sum(len(retained_by_case[("eligible_matched", f"s{st}_first")]) for st in (7, 8)) == 56609
retention_audit.to_csv(f"{OUTPUT_STEM}_retention_summary.csv", index=False)
print("Maximum-retention subsets and frozen source-row manifests saved.", flush=True)


# %%
############ Animal curves and population-versus-selected matching audits ############
curves = {metric: np.zeros((3, 6, 7, 2, len(bins_by_metric[metric]) - 1)) for metric in METRICS}
sems = {metric: np.zeros_like(curves[metric]) for metric in METRICS}
panel_records, comparison_records, curve_parts = [], [], []
for si, state in enumerate(STATES):
    for ci, (st, position) in enumerate(CASES):
        case, flag = case_names[ci], f"selected_{position}"
        retained = retained_by_case[(state, case)]
        original = eligible_by_session[st]
        animal_records = []
        for ai, animal in enumerate(ANIMALS):
            for led in (0, 1):
                denominator = retained.loc[retained.animal.eq(animal) & retained.LED_trial.eq(led)]
                chosen = denominator.loc[denominator[flag]]
                original_group = original.loc[original.animal.eq(animal) & original.LED_trial.eq(led)]
                original_chosen = original_group.loc[original_group[flag]]
                assert len(chosen) > 0 and len(denominator) > 0
                population_ks = paired_ks(denominator.intended_fix, denominator.stimulus_gap)
                selected_ks = paired_ks(chosen.intended_fix, chosen.stimulus_gap)
                ks_method = "exact" if len(chosen) == 1 else "asymp"
                np.testing.assert_allclose(selected_ks, ks_2samp(chosen.intended_fix, chosen.stimulus_gap, method=ks_method).statistic, atol=1e-14)
                if state == "selected_matched":
                    assert selected_ks <= 1 / KS_INVERSE + 1e-12
                    assert len(original_group) - len(denominator) == len(original_chosen) - len(chosen)
                if state == "eligible_matched":
                    assert population_ks <= 1 / KS_INVERSE + 1e-12
                comparison_records.append({
                    "state": state, "case": case, "animal": animal, "LED_trial": led,
                    "population_original_n": len(original_group), "population_retained_n": len(denominator),
                    "selected_original_n": len(original_chosen), "selected_retained_n": len(chosen),
                    "population_original_KS": paired_ks(original_group.intended_fix, original_group.stimulus_gap),
                    "population_retained_KS": population_ks,
                    "selected_original_KS": paired_ks(original_chosen.intended_fix, original_chosen.stimulus_gap),
                    "selected_retained_KS": selected_ks,
                    "KS_target_population": "none" if state == "original" else "selected_aborts" if state == "selected_matched" else "all_eligible",
                    "selected_sparse": len(chosen) < SPARSE_SELECTION_THRESHOLD,
                    "selected_zero_onset_n": int(chosen.LED_onset_time.eq(0).sum()),
                    "selected_mean_intended_s": chosen.intended_fix.mean(),
                    "selected_mean_gap_s": chosen.stimulus_gap.mean(),
                    "selected_mean_onset_s": chosen.LED_onset_time.mean(),
                })
                for metric in METRICS:
                    bins, width = bins_by_metric[metric], width_by_metric[metric]
                    values = chosen[metric].to_numpy()
                    values = values[np.isfinite(values)]
                    counts = np.histogram(values, bins=bins)[0]
                    assert counts.sum() == len(values) and len(values) > 0
                    normalizer = len(denominator) if metric == METRICS[3] else len(values)
                    heights = counts / (normalizer * width)
                    area = float(heights @ np.diff(bins))
                    assert np.isclose(area, len(values) / normalizer, rtol=0, atol=1e-12)
                    curves[metric][si, ci, ai, led] = heights
                    record = {"state": state, "case": case, "row": str(animal), "LED_trial": led, "metric": metric,
                              "selected_aborts": len(chosen), "plotted_observations": len(values),
                              "eligible_denominator": len(denominator), "histogram_normalizer": normalizer,
                              "full_histogram_area": area, "expected_area": len(values) / normalizer,
                              "bin_width_s": width, "denominator_missing_timed_fix": int(denominator.timed_fix.isna().sum())}
                    panel_records.append(record)
                    animal_records.append(record)
                    curve_parts.append(pd.DataFrame({"state": state, "case": case, "row": str(animal), "LED_trial": led,
                                                     "metric": metric, "bin_left_s": bins[:-1], "bin_right_s": bins[1:],
                                                     "height": heights, "sem": np.nan, "bin_count": counts}))
        animal_panels = pd.DataFrame(animal_records)
        for metric in METRICS:
            bins = bins_by_metric[metric]
            for led in (0, 1):
                matrix = curves[metric][si, ci, :6, led]
                mean, sem = matrix.mean(axis=0), matrix.std(axis=0, ddof=1) / np.sqrt(6)
                # Independent arithmetic check; all six animals contribute, including sparse groups.
                np.testing.assert_allclose(mean, sum(matrix[i] for i in range(6)) / 6, atol=1e-14)
                np.testing.assert_allclose(sem, np.sqrt(sum((matrix[i] - mean) ** 2 for i in range(6)) / 30), atol=1e-14)
                curves[metric][si, ci, 6, led], sems[metric][si, ci, 6, led] = mean, sem
                group = animal_panels.loc[animal_panels.metric.eq(metric) & animal_panels.LED_trial.eq(led)]
                assert len(group) == 6
                area = float(mean @ np.diff(bins))
                assert np.isclose(area, group.expected_area.mean(), rtol=0, atol=1e-12)
                record = {"state": state, "case": case, "row": "equal_animal_mean", "LED_trial": led, "metric": metric,
                          "histogram_normalizer": np.nan, "full_histogram_area": area,
                          "expected_area": group.expected_area.mean(), "bin_width_s": width_by_metric[metric]}
                for field in ("selected_aborts", "plotted_observations", "eligible_denominator", "denominator_missing_timed_fix"):
                    record[field] = int(group[field].sum())
                panel_records.append(record)
                curve_parts.append(pd.DataFrame({"state": state, "case": case, "row": "equal_animal_mean", "LED_trial": led,
                                                 "metric": metric, "bin_left_s": bins[:-1], "bin_right_s": bins[1:],
                                                 "height": mean, "sem": sem, "bin_count": np.nan}))
        if state == "original":
            for metric in METRICS:
                np.testing.assert_array_equal(curves[metric][si, ci], original_arrays[metric][ci])
                np.testing.assert_array_equal(sems[metric][si, ci], original_arrays[f"{metric}_sem"][ci])


# %%
############ Save numerical outputs and check the original rows exactly ############
panels = pd.DataFrame(panel_records)
comparisons = pd.DataFrame(comparison_records)
curve_audit = pd.concat(curve_parts, ignore_index=True)
mean_panels = panels.loc[panels.row.eq("equal_animal_mean")].copy()
assert len(panels) == 3 * 6 * 7 * 2 * 4 and len(comparisons) == 3 * 6 * 6 * 2
assert not curve_audit.duplicated(["state", "case", "row", "LED_trial", "metric", "bin_left_s"]).any()
assert np.isfinite(curve_audit.height).all() and curve_audit.height.ge(0).all()
check_keys = ["case", "row", "LED_trial", "metric"]
check_fields = ["selected_aborts", "plotted_observations", "eligible_denominator", "full_histogram_area"]
pd.testing.assert_frame_equal(panels.loc[panels.state.eq("original")].set_index(check_keys).sort_index()[check_fields],
                              original_panels.set_index(check_keys).sort_index()[check_fields], check_exact=True)
panels.to_csv(f"{OUTPUT_STEM}_count_area_audit.csv", index=False)
mean_panels.to_csv(f"{OUTPUT_STEM}_mean_area_audit.csv", index=False)
comparisons.to_csv(f"{OUTPUT_STEM}_population_vs_selected_KS.csv", index=False)
curve_audit.to_csv(f"{OUTPUT_STEM}_curve_audit.csv", index=False)
np.savez_compressed(f"{OUTPUT_STEM}_curves.npz", states=np.array(STATES), cases=np.array(case_names), rows=np.array(row_names),
                    led_conditions=np.array([0, 1]), timing_bins=timing_bins, abort_bins=abort_bins,
                    **curves, **{f"{metric}_sem": sems[metric] for metric in METRICS})
comparison_summary = comparisons.groupby(["state", "case", "LED_trial"], sort=False).agg(
    n_animals=("animal", "nunique"), selected_n=("selected_retained_n", "sum"),
    selected_n_min=("selected_retained_n", "min"), sparse_animal_groups=("selected_sparse", "sum"),
    population_n=("population_retained_n", "sum"),
    selected_KS_min=("selected_retained_KS", "min"), selected_KS_max=("selected_retained_KS", "max"),
    population_KS_min=("population_retained_KS", "min"), population_KS_max=("population_retained_KS", "max"),
    selected_equal_animal_mean_onset_s=("selected_mean_onset_s", "mean"),
).reset_index()
assert comparison_summary.n_animals.eq(6).all()
comparison_summary.to_csv(f"{OUTPUT_STEM}_KS_summary.csv", index=False)
print(retention_audit.loc[retention_audit.state.ne("original")].to_string(index=False), flush=True)


# %%
############ Two figures with common column scales; no post-hoc smoothing ############
upper_limits = {metric: 1.34 * float((curves[metric][:, :, 6] + sems[metric][:, :, 6]).max()) for metric in METRICS}
layouts = {
    "selected_aborts_12x4": [(si, ci) for ci in range(6) for si in (0, 1)],
    "all_eligible_6x4": [(2, ci) for ci in range(6)],
}
figure_paths = []
for figure_name, layout in layouts.items():
    nrows = len(layout)
    fig, axes = plt.subplots(nrows, 4, figsize=(21, 3.0 * nrows + 2.1), sharex="col", sharey="col")
    for ri, (si, ci) in enumerate(layout):
        state, case = STATES[si], case_names[ci]
        st, position = CASES[ci]
        stats = comparison_summary.loc[comparison_summary.state.eq(state) & comparison_summary.case.eq(case)].set_index("LED_trial")
        row_label = "Original" if state == "original" else "Selected-abort matched" if state == "selected_matched" else "All-eligible matched"
        for col, metric in enumerate(METRICS):
            ax, bins = axes[ri, col], bins_by_metric[metric]
            if state == "selected_matched":
                ax.set_facecolor("#f8faf7")
            for led in (0, 1):
                mean, sem = curves[metric][si, ci, 6, led], sems[metric][si, ci, 6, led]
                ax.fill_between(bins, np.r_[mean - sem, (mean - sem)[-1]], np.r_[mean + sem, (mean + sem)[-1]],
                                step="post", color=COLORS[led], alpha=0.16, linewidth=0)
                ax.stairs(mean, bins, color=COLORS[led], linewidth=1.5)
                audit = mean_panels.loc[mean_panels.state.eq(state) & mean_panels.case.eq(case)
                                       & mean_panels.LED_trial.eq(led) & mean_panels.metric.eq(metric)].iloc[0]
                label = "OFF" if led == 0 else "ON"
                original_n = EXPECTED_SELECTED[(st, position)][led]
                note = (f"{label} n={int(audit.plotted_observations):,}" if state == "original" else
                        f"{label} n={int(audit.plotted_observations):,}/{original_n:,}")
                if col == 3:
                    note += f", N={int(audit.eligible_denominator):,}; area={audit.full_histogram_area:.3f}"
                elif col == 2:
                    note = f"{label} selected KS={stats.loc[led, 'selected_KS_max']:.3f} max"
                    if state == "eligible_matched":
                        note += f"; eligible={stats.loc[led, 'population_KS_max']:.3f}"
                ax.text(0.98, 0.975 - 0.075 * led, note, transform=ax.transAxes, ha="right", va="top", fontsize=8.2, color=COLORS[led])
            if col == 3 and state != "original":
                minima = "/".join(str(int(stats.loc[led, "selected_n_min"])) for led in (0, 1))
                sparse_count = int(stats.sparse_animal_groups.sum())
                note = f"Min animal n (OFF/ON): {minima}"
                if sparse_count:
                    note += f"; {sparse_count}/12 groups n<{SPARSE_SELECTION_THRESHOLD}"
                ax.text(0.98, 0.805, note, transform=ax.transAxes, ha="right", va="top", fontsize=8,
                        color="#9c4a00" if sparse_count else "0.35")
            ax.set_xlim(*(ABORT_DISPLAY_S if col == 3 else TIMING_RANGE_S))
            ax.set_ylim(0, upper_limits[metric])
            if col == 0:
                ax.set_ylabel(f"s{st} · {ROW_LABELS[position]}\n{row_label}\nDensity (s⁻¹)", fontsize=10)
            elif col == 2:
                ax.set_ylabel("Density (s⁻¹)", fontsize=10)
            elif col == 3:
                ax.axvline(0, color="0.45", linestyle=":", linewidth=1)
                ax.set_ylabel("Abort rate per eligible trial (s⁻¹)", fontsize=10)
            if ri == 0:
                ax.set_title(COLUMN_TITLES[col], fontsize=13, pad=12)
            if ri == nrows - 1:
                ax.set_xlabel("Time from fixation onset (s)" if col < 2 else
                              "Scheduled LED-to-stimulus interval (s)" if col == 2 else
                              "Time from scheduled LED onset (s)", fontsize=10)
            ax.grid(axis="y", color="0.91", linewidth=0.5)
            ax.tick_params(labelsize=9)
            ax.spines[["top", "right"]].set_visible(False)
    for col in range(4):
        assert all(axes[ri, col].get_ylim() == (0, upper_limits[METRICS[col]]) for ri in range(nrows))
    title = ("LED7 intended-fixation versus stimulus-gap matching: selected aborts" if nrows == 12 else
             "LED7 intended-fixation versus stimulus-gap matching: all eligible trials")
    fig.suptitle(title, fontsize=19, y=0.994)
    fig.legend(handles=[Line2D([0], [0], color=COLORS[led], lw=2, label=f"Plotted trial LED {'OFF' if led == 0 else 'ON'}")
                        for led in (0, 1)], loc="upper center", bbox_to_anchor=(0.5, 0.98 if nrows == 12 else 0.97),
               ncol=2, frameon=False, fontsize=12)
    method_note = ("Each original row is followed by its matched row. Match selected aborts only; other eligible trials remain. N = original N minus rejected selected aborts."
                   if nrows == 12 else
                   "Match all eligible trials once per animal/session-type/LED; reuse that retained subset for first, last and isolated categories. Timing panels show selected aborts only.")
    constraint_note = ("KS ≤ 0.20 applies to selected-abort intended_fix versus stimulus gap within each animal/LED group. Sparse groups are retained, not excluded."
                       if nrows == 12 else
                       "KS ≤ 0.20 applies to the eligible population, NOT its abort subgroups. Selected-abort KS and eligible-population KS are reported separately (max across six animals).")
    footer = (
        "Equal-animal mean ± SEM: animals 92, 93, 98, 99, 100, 103. First/last: complete runs ≥2; isolated: valid→abort→valid. Classifications frozen before selection.\n"
        + method_note + "\n" + constraint_note + "\n"
        "Timing densities: 40 ms bins, unit area per animal. Rates: 20 ms bins; N counts retained event-3 OR success ±1 trials. Full areas cover −2 to +2 s.\n"
        "n/N labels are summed counts; the plotted area is the mean of six animal fractions, not pooled n/N. OFF onset is scheduled/counterfactual.\n"
        "Maximum intact-row retention using scheduled timing only; no timed_fix in matching. Does not match OFF against ON or enforce pre-onset overlap. Descriptive, not causal."
    )
    fig.text(0.5, 0.018, footer, ha="center", fontsize=8.6, linespacing=1.5)
    fig.tight_layout(rect=(0.005, 0.062 if nrows == 12 else 0.114, 0.998, 0.964 if nrows == 12 else 0.945),
                     h_pad=1.6, w_pad=1.6)
    figure_path = Path(f"{OUTPUT_STEM}_{figure_name}.png")
    fig.savefig(figure_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    figure_paths.append(figure_path)
    print(f"Saved {figure_path}", flush=True)


# %%
############ Verify preservation and save a reproducible run receipt ############
for path in protected_paths:
    with path.open("rb") as handle:
        assert hashlib.file_digest(handle, "sha256").hexdigest() == input_hashes[str(path.relative_to(REPO))]
summary = {
    "source_script": str(Path(__file__).relative_to(REPO)),
    "figures": [str(path.relative_to(REPO)) for path in figure_paths],
    "input_sha256": input_hashes, "old_artifacts_preserved": True, "case_order": case_names,
    "animals": ANIMALS, "seed_per_analysis": SEED, "KS_limit": 1 / KS_INVERSE,
    "algorithm": "Exact interval-capacity sweep; binary-search largest feasible k with retained_count >= 5k; keep zero-length intervals.",
    "optimization_inputs": ["intended_fix", "intended_fix - LED_onset_time"],
    "tie_order": "Reset NumPy default_rng seed per analysis. Sorted group keys; shuffle source-row-sorted group once, then stable timing order.",
    "selection": "Original complete-run first/last endpoints and isolated VAV, frozen before matching; actual-ID adjacency within animal/session.",
    "selected_mode_denominator": "Original event-3-or-valid denominator minus rejected selected aborts only.",
    "eligible_mode_denominator": "All retained event-3-or-valid rows; identical retained subset across the three categories.",
    "averaging": "Equal mean of all six animal-normalized curves; sample SD/sqrt(6) SEM. No pooling or sparse-group exclusion.",
    "original_curves": "All original per-animal histograms, means, SEMs, counts and areas reproduce exactly.",
    "figure_y_upper_limits": upper_limits,
    "histogram_bin_width_s": {"timing": TIMING_BIN_WIDTH_S, "abort_rate": ABORT_BIN_WIDTH_S},
    "sparse_threshold_flag_only": SPARSE_SELECTION_THRESHOLD,
    "optimizer_validation": {"exhaustive_examples": 33, "certified_MILP_comparisons": 73,
                             "audit": str(validation_path.relative_to(REPO))},
    "retention_summary": retention_audit.to_dict(orient="records"),
    "KS_summary": comparison_summary.to_dict(orient="records"),
    "interpretation": "Approximate within-LED timing matching, not OFF/ON rate matching. Descriptive selection; no causal conclusion.",
    "new_fit": False,
}
Path(f"{OUTPUT_STEM}_run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("Passed: exact original curves, source identities, retention totals, capacity boundaries, KS, histogram areas, means/SEMs and preserved historical files.")
