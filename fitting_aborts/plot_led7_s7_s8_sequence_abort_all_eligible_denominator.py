# %%
"""Four-row LED7 sequence-abort figure with one all-eligible rate denominator.

After-abort selections remain pairwise: A1→A2→A3 contributes A2 and A3.
Classify on the original trial stream, then remove only selected pre-onset
aborts to match OFF/ON rates. The isolated rows reuse their saved row identities.
"""

from pathlib import Path
import hashlib
import json
import os
import time


# %%
############ Editable parameters ############
SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
REFERENCE_STEM = SCRIPT_DIR / "led7_s7_s8_sequence_abort_pre_onset_matched_summary"
OUTPUT_STEM = SCRIPT_DIR / "led7_s7_s8_sequence_abort_all_eligible_denominator"
ANIMALS = (92, 93, 98, 99, 100, 103)
CASES = ((7, "after_abort"), (8, "after_abort"), (7, "isolated"), (8, "isolated"))
SEED = 20261001
MAX_BIN_RATE_DIFFERENCE = 0.005
MAX_PRE_ONSET_FRACTION_DIFFERENCE = 0.001
SOLVER_TIME_LIMIT_S = 60
TIMING_BIN_WIDTH_S = 0.040
ABORT_BIN_WIDTH_S = 0.020
ABORT_DISPLAY_S = (-0.4, 0.5)
FIGURE_DPI = 240
COLORS = {0: "tab:blue", 1: "tab:red"}
METRICS = ("intended_fix", "LED_onset_time", "abort_from_LED_onset", "matched_abort_from_LED_onset")
ROW_LABELS = {"after_abort": "Abort → abort", "isolated": "Valid → abort → valid"}
EXPECTED_ORIGINAL_SELECTED = {(7, "after_abort"): (1528, 1569), (8, "after_abort"): (963, 2463),
                              (7, "isolated"): (6498, 3365), (8, "isolated"): (6406, 4105)}
EXPECTED_ALL_ELIGIBLE = {7: (63525, 30910), 8: (68584, 38486)}
EXPECTED_TOP_REMOVED = {7: 634, 8: 828}

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, eye, hstack, vstack

case_names = [f"s{st}_{selection}" for st, selection in CASES]
rows = [*[str(animal) for animal in ANIMALS], "equal_animal_mean"]
timing_bins = np.linspace(0, 2.2, round(2.2 / TIMING_BIN_WIDTH_S) + 1)
abort_bins = np.linspace(-2, 2, round(4 / ABORT_BIN_WIDTH_S) + 1)
pre_bins = np.linspace(-2, 0, round(2 / ABORT_BIN_WIDTH_S) + 1)
n_pre_bins = len(pre_bins) - 1
np.testing.assert_array_equal(abort_bins[:n_pre_bins + 1], pre_bins)
bins_by_metric = dict(zip(METRICS, (timing_bins, timing_bins, abort_bins, abort_bins)))
width_by_metric = dict(zip(METRICS, (TIMING_BIN_WIDTH_S,) * 2 + (ABORT_BIN_WIDTH_S,) * 2))


# %%
############ Protected inputs, original selections and trial-ID adjacency ############
protected_paths = [REPO / f"raw_data/LED7_s{st}.csv" for st in (7, 8)]
protected_paths += [Path(f"{REFERENCE_STEM}_curves.npz"), Path(f"{REFERENCE_STEM}_count_area_audit.csv"),
                    Path(f"{REFERENCE_STEM}_4x4.png")]
protected_paths += [Path(f"{REFERENCE_STEM}_{case}_selection_manifest.csv") for case in case_names]
protected_paths += [REPO / "docs/assets/results/2026-10-01/led7_s7_s8_sequence_abort_pre_onset_matched_summary_4x4.png"]
input_hashes = {}
for path in protected_paths:
    assert path.is_file(), path
    with path.open("rb") as handle:
        input_hashes[str(path.relative_to(REPO))] = hashlib.file_digest(handle, "sha256").hexdigest()
with np.load(f"{REFERENCE_STEM}_curves.npz") as payload:
    reference_arrays = {key: payload[key].copy() for key in payload.files}
np.testing.assert_array_equal(reference_arrays["cases"], case_names)
np.testing.assert_array_equal(reference_arrays["rows"], rows)
np.testing.assert_array_equal(reference_arrays["timing_bins"], timing_bins)
np.testing.assert_array_equal(reference_arrays["abort_bins"], abort_bins)
reference_panels = pd.read_csv(f"{REFERENCE_STEM}_count_area_audit.csv", float_precision="round_trip")

eligible_by_session, source_by_session, saved_manifests = {}, {}, {}
keys = ["animal", "session", "trial"]
for st in (7, 8):
    source = pd.read_csv(REPO / f"raw_data/LED7_s{st}.csv", float_precision="round_trip")
    assert source.shape == ({7: 138440, 8: 148428}[st], 52)
    assert source.training_level.eq(16).all() and source.session_type.eq(st).all()
    assert (source.repeat_trial.isin([0, 2]) | source.repeat_trial.isna()).all()
    assert source.LED_trial.isin([0, 1]).all() and not source.duplicated(keys).any()
    source["source_csv_row_index"] = np.arange(len(source))
    stream = source.loc[source.animal.isin(ANIMALS)].sort_values(keys, kind="stable").copy()
    assert tuple(sorted(stream.animal.unique())) == ANIMALS
    assert len(stream) == {7: 100251, 8: 113941}[st]
    prior = stream.groupby(["animal", "session"], sort=False)[["trial", "abort_event", "success"]].shift(1)
    following = stream.groupby(["animal", "session"], sort=False)[["trial", "success"]].shift(-1)
    is_abort, is_valid = stream.abort_event.eq(3), stream.success.isin([-1, 1])
    assert not (is_abort & is_valid).any()
    stream["previous_is_abort"] = prior.trial.eq(stream.trial - 1) & prior.abort_event.eq(3)
    stream["isolated_abort"] = (is_abort & prior.trial.eq(stream.trial - 1)
                                & following.trial.eq(stream.trial + 1)
                                & prior.success.isin([-1, 1]) & following.success.isin([-1, 1]))
    stream["abort_from_LED_onset"] = stream.timed_fix - stream.LED_onset_time
    stream = stream.set_index("source_csv_row_index", drop=False)
    eligible = stream.loc[is_abort | is_valid].copy()
    assert tuple(eligible.groupby("LED_trial").size()) == EXPECTED_ALL_ELIGIBLE[st]
    source_by_session[st], eligible_by_session[st] = stream, eligible

    for selection in ("after_abort", "isolated"):
        case = f"s{st}_{selection}"
        saved = pd.read_csv(f"{REFERENCE_STEM}_{case}_selection_manifest.csv", float_precision="round_trip")
        assert saved.source_csv_row_index.is_unique and not saved.duplicated(keys).any()
        saved = saved.set_index("source_csv_row_index", drop=False)
        should_select = (stream.abort_event.eq(3) & stream.previous_is_abort if selection == "after_abort"
                         else stream.isolated_abort)
        selected_ids = set(stream.index[should_select])
        assert set(saved.index[saved.selected_abort]) == selected_ids
        assert tuple(saved.loc[saved.selected_abort].groupby("LED_trial").size()) == EXPECTED_ORIGINAL_SELECTED[(st, selection)]
        if selection == "isolated":
            assert set(saved.index) == set(eligible.index)
        else:
            assert set(saved.index) == set(stream.index[stream.previous_is_abort & (is_abort | is_valid)])
        compare = [*keys, "LED_trial", "abort_event", "success", "intended_fix", "LED_onset_time", "timed_fix"]
        pd.testing.assert_frame_equal(stream.loc[saved.index, compare].sort_index(),
                                      saved[compare].sort_index(), check_dtype=False, check_exact=True)
        assert saved.LED_onset_time.ge(0).all()
        assert np.isfinite(saved.loc[saved.selected_abort, ["intended_fix", "LED_onset_time"]]).all().all()
        assert saved.loc[saved.selected_abort & saved.timed_fix.notna(), "timed_fix"].lt(
            saved.loc[saved.selected_abort & saved.timed_fix.notna(), "intended_fix"]).all()
        saved_manifests[(st, selection)] = saved
    print(f"s{st}: exact saved abort identities, actual adjacent trial IDs and all-eligible denominators verified.", flush=True)


# %%
############ Certified integer quotas with the corrected, all-eligible denominators ############
def optimize_pre_onset_counts(available, denominators):
    """Maximize retained pre-onset abort counts under the established linear rule."""
    mandatory = denominators - available.sum(axis=1)
    assert (mandatory > 0).all()
    ratio = mandatory[0] / mandatory[1]
    u_max = float((available / mandatory[:, None]).max())
    epsilon = ABORT_BIN_WIDTH_S * MAX_BIN_RATE_DIFFERENCE - u_max * MAX_PRE_ONSET_FRACTION_DIFFERENCE
    assert epsilon > 0
    matrix = vstack([
        hstack([eye(n_pre_bins), -ratio * eye(n_pre_bins)]),
        csr_matrix(np.r_[np.ones(n_pre_bins), -ratio * np.ones(n_pre_bins)][None, :]),
    ], format="csr")
    tolerance = np.r_[np.full(n_pre_bins, mandatory[0] * epsilon),
                      mandatory[0] * MAX_PRE_ONSET_FRACTION_DIFFERENCE]
    started = time.perf_counter()
    solution = milp(
        -np.ones(2 * n_pre_bins), integrality=np.ones(2 * n_pre_bins),
        bounds=Bounds(np.zeros(2 * n_pre_bins), available.ravel()),
        constraints=LinearConstraint(matrix, -tolerance, tolerance),
        options={"mip_rel_gap": 0.0, "time_limit": SOLVER_TIME_LIMIT_S},
    )
    assert solution.success and solution.status == 0, solution.message
    assert solution.mip_gap == 0 and np.isfinite(solution.x).all()
    quotas = np.rint(solution.x).astype(int).reshape(2, n_pre_bins)
    np.testing.assert_allclose(solution.x, quotas.ravel(), rtol=0, atol=1e-7)
    assert ((quotas >= 0) & (quotas <= available)).all()
    assert np.all(np.abs(matrix @ quotas.ravel()) <= tolerance + 1e-7)
    assert np.isclose(solution.fun, -quotas.sum(), rtol=0, atol=1e-7)
    assert np.isclose(solution.mip_dual_bound, solution.fun, rtol=0, atol=1e-7)
    retained_n = mandatory + quotas.sum(axis=1)
    rates = quotas / (retained_n[:, None] * ABORT_BIN_WIDTH_S)
    max_difference = float(np.abs(rates[0] - rates[1]).max())
    fractions = quotas.sum(axis=1) / retained_n
    fraction_difference = float(abs(fractions[0] - fractions[1]))
    assert max_difference <= MAX_BIN_RATE_DIFFERENCE + 1e-12
    assert fraction_difference <= MAX_PRE_ONSET_FRACTION_DIFFERENCE + 1e-12
    audit = {
        "solver": "scipy.optimize.milp / HiGHS", "certified_optimal": True,
        "status": solution.status, "mip_gap": solution.mip_gap, "objective": solution.fun,
        "dual_bound": solution.mip_dual_bound, "elapsed_s": time.perf_counter() - started,
        "original_OFF": int(denominators[0]), "original_ON": int(denominators[1]),
        "mandatory_OFF": int(mandatory[0]), "mandatory_ON": int(mandatory[1]),
        "retained_OFF": int(retained_n[0]), "retained_ON": int(retained_n[1]),
        "removed": int((available - quotas).sum()), "r": ratio, "u_max": u_max, "epsilon": epsilon,
        "max_pre_bin_rate_difference_s_inv": max_difference,
        "pre_onset_fraction_difference": fraction_difference,
    }
    return quotas, audit


# %%
############ Freeze categories, rematch the top rows and reuse bottom-row identities ############
case_frames = {}
optimizer_records, quota_records, retention_records = [], [], []
for st, selection in CASES:
    case = f"s{st}_{selection}"
    saved = saved_manifests[(st, selection)]
    frame = eligible_by_session[st].copy()
    selected_ids = set(saved.index[saved.selected_abort])
    frame["selected_abort"] = frame.index.isin(selected_ids)
    frame["abort_from_LED_onset"] = frame.timed_fix - frame.LED_onset_time
    frame["pre_onset_candidate"] = (frame.selected_abort & frame.abort_from_LED_onset.notna()
                                    & frame.abort_from_LED_onset.lt(0))
    assert frame.loc[frame.pre_onset_candidate, "abort_from_LED_onset"].ge(-2).all()
    frame["pre_onset_bin"] = -1
    frame.loc[frame.pre_onset_candidate, "pre_onset_bin"] = (
        np.searchsorted(pre_bins, frame.loc[frame.pre_onset_candidate, "abort_from_LED_onset"], side="right") - 1
    )
    assert frame.loc[frame.pre_onset_candidate, "pre_onset_bin"].between(0, n_pre_bins - 1).all()
    original_selected = frame.loc[frame.selected_abort]
    assert tuple(original_selected.groupby("LED_trial").size()) == EXPECTED_ORIGINAL_SELECTED[(st, selection)]
    assert set(original_selected.index) == selected_ids
    if selection == "isolated":
        pd.testing.assert_frame_equal(frame[keys].sort_index(), saved[keys].sort_index(), check_dtype=False)
        frame["retained"] = saved.retained.reindex(frame.index)
        assert frame.retained.notna().all()
        for column in ("selected_abort", "pre_onset_candidate", "pre_onset_bin"):
            np.testing.assert_array_equal(frame[column].sort_index(), saved[column].sort_index())
    else:
        frame["retained"] = True
        rng = np.random.default_rng(SEED)  # Independent deterministic draw for s7 and s8.
        for animal in ANIMALS:
            group = frame.loc[frame.animal.eq(animal)]
            denominators = group.groupby("LED_trial").size().reindex([0, 1]).to_numpy(dtype=int)
            available = np.stack([
                np.histogram(group.loc[group.LED_trial.eq(led) & group.pre_onset_candidate,
                                       "abort_from_LED_onset"], bins=pre_bins)[0]
                for led in (0, 1)
            ])
            quotas, diagnostic = optimize_pre_onset_counts(available, denominators)
            for led in (0, 1):
                for bin_index in range(n_pre_bins):
                    pool = group.index[group.LED_trial.eq(led) & group.pre_onset_candidate
                                       & group.pre_onset_bin.eq(bin_index)].to_numpy()
                    pool.sort()
                    assert len(pool) == available[led, bin_index]
                    chosen = rng.choice(pool, size=quotas[led, bin_index], replace=False)
                    frame.loc[pool, "retained"] = False
                    frame.loc[chosen, "retained"] = True
                    assert len(chosen) == quotas[led, bin_index] and len(np.unique(chosen)) == len(chosen)
                    quota_records.append({"case": case, "session_type": st, "animal": animal,
                                          "LED_trial": led, "bin_index": bin_index,
                                          "bin_left_s": pre_bins[bin_index], "bin_right_s": pre_bins[bin_index + 1],
                                          "available": len(pool), "retained": len(chosen),
                                          "removed": len(pool) - len(chosen)})
            optimizer_records.append({"case": case, "session_type": st, "animal": animal, **diagnostic})

    kept, removed = frame.loc[frame.retained], frame.loc[~frame.retained]
    assert frame.index.is_unique and kept.index.is_unique
    assert removed.selected_abort.all() and removed.pre_onset_candidate.all()
    assert removed.abort_event.eq(3).all() and not removed.success.isin([-1, 1]).any()
    assert frame.loc[~frame.pre_onset_candidate, "retained"].all()
    assert len(kept) + len(removed) == len(frame)
    for animal in ANIMALS:
        for led in (0, 1):
            original = frame.loc[frame.animal.eq(animal) & frame.LED_trial.eq(led)]
            retained = original.loc[original.retained]
            assert len(retained) > 0
            assert original.success.isin([-1, 1]).sum() == retained.success.isin([-1, 1]).sum()
            assert original.loc[original.selected_abort & original.abort_from_LED_onset.ge(0)].shape[0] == (
                retained.selected_abort & retained.abort_from_LED_onset.ge(0)).sum()
            assert original.timed_fix.isna().sum() == retained.timed_fix.isna().sum()
    if selection == "after_abort":
        assert len(removed) == EXPECTED_TOP_REMOVED[st]
        assert tuple(frame.groupby("LED_trial").size()) == EXPECTED_ALL_ELIGIBLE[st]
        assert frame.loc[frame.selected_abort, "timed_fix"].isna().sum() == 2
    else:
        np.testing.assert_array_equal(frame.retained.sort_index(), saved.retained.sort_index())
    retained_selected = kept.loc[kept.selected_abort].groupby("LED_trial").size().reindex([0, 1]).to_numpy(int)
    retained_denominators = kept.groupby("LED_trial").size().reindex([0, 1]).to_numpy(int)
    retention_records.append({"case": case, "session_type": st, "selection": selection,
                              "original_selected_OFF": EXPECTED_ORIGINAL_SELECTED[(st, selection)][0],
                              "original_selected_ON": EXPECTED_ORIGINAL_SELECTED[(st, selection)][1],
                              "retained_selected_OFF": retained_selected[0], "retained_selected_ON": retained_selected[1],
                              "original_denominator_OFF": EXPECTED_ALL_ELIGIBLE[st][0],
                              "original_denominator_ON": EXPECTED_ALL_ELIGIBLE[st][1],
                              "retained_denominator_OFF": retained_denominators[0],
                              "retained_denominator_ON": retained_denominators[1],
                              "removed_pre_onset_selected": len(removed),
                              "missing_selected_timed_fix": int(frame.loc[frame.selected_abort, "timed_fix"].isna().sum()),
                              "saved_bottom_selection_reused": selection == "isolated"})
    case_frames[(st, selection)] = frame
    print(f"{case}: selected {sum(EXPECTED_ORIGINAL_SELECTED[(st, selection)]):,}, "
          f"retained {retained_selected.sum():,}; eligible {len(frame):,}, "
          f"retained {len(kept):,}; removed {len(removed):,}.", flush=True)

optimizer_audit = pd.DataFrame(optimizer_records)
quota_audit = pd.DataFrame(quota_records)
retention_audit = pd.DataFrame(retention_records)
assert len(optimizer_audit) == 12 and optimizer_audit.certified_optimal.all()
assert optimizer_audit.mip_gap.eq(0).all()
assert len(quota_audit) == 2 * len(ANIMALS) * 2 * n_pre_bins


# %%
############ Normalize each animal first, then compute equal-animal mean and SEM ############
curves = {metric: np.zeros((4, 7, 2, len(bins_by_metric[metric]) - 1)) for metric in METRICS}
sems = {metric: np.zeros_like(curves[metric]) for metric in METRICS}
panel_records, curve_parts = [], []
for ci, (st, selection) in enumerate(CASES):
    case = case_names[ci]
    frame = case_frames[(st, selection)]
    for ai, animal in enumerate(ANIMALS):
        for led in (0, 1):
            original = frame.loc[frame.animal.eq(animal) & frame.LED_trial.eq(led)]
            for mi, metric in enumerate(METRICS):
                current = original.loc[original.retained] if mi == 3 else original
                chosen = current.loc[current.selected_abort]
                values = chosen["abort_from_LED_onset" if mi == 3 else metric].dropna().to_numpy()
                bins, width = bins_by_metric[metric], width_by_metric[metric]
                counts = np.histogram(values, bins=bins)[0]
                assert counts.sum() == len(values) and len(values) > 0
                normalizer = len(values) if mi < 2 else len(current)
                heights = counts / (normalizer * width)
                area = float(heights @ np.diff(bins))
                assert np.isclose(area, len(values) / normalizer, rtol=0, atol=1e-12)
                curves[metric][ci, ai, led] = heights
                if selection == "isolated" or mi < 2:
                    np.testing.assert_array_equal(heights, reference_arrays[metric][ci, ai, led])
                panel_records.append({"case": case, "row": str(animal), "LED_trial": led, "metric": metric,
                                      "selected_aborts": len(chosen), "plotted_observations": len(values),
                                      "eligible_denominator": len(current), "histogram_normalizer": normalizer,
                                      "full_histogram_area": area, "expected_area": len(values) / normalizer,
                                      "bin_width_s": width,
                                      "missing_selected_timed_fix": int(chosen.timed_fix.isna().sum())})
                curve_parts.append(pd.DataFrame({"case": case, "row": str(animal), "LED_trial": led,
                                                 "metric": metric, "bin_left_s": bins[:-1], "bin_right_s": bins[1:],
                                                 "height": heights, "sem": np.nan, "bin_count": counts}))
    for metric in METRICS:
        bins = bins_by_metric[metric]
        for led in (0, 1):
            matrix = curves[metric][ci, :6, led]
            mean, sem = matrix.mean(axis=0), matrix.std(axis=0, ddof=1) / np.sqrt(6)
            np.testing.assert_allclose(mean, sum(matrix[i] for i in range(6)) / 6, atol=1e-14)
            np.testing.assert_allclose(sem, np.sqrt(sum((matrix[i] - mean) ** 2 for i in range(6)) / 30), atol=1e-14)
            curves[metric][ci, 6, led], sems[metric][ci, 6, led] = mean, sem
            group = pd.DataFrame(panel_records)
            group = group.loc[group.case.eq(case) & group.row.ne("equal_animal_mean")
                              & group.LED_trial.eq(led) & group.metric.eq(metric)]
            assert len(group) == 6
            area = float(mean @ np.diff(bins))
            assert np.isclose(area, group.expected_area.mean(), rtol=0, atol=1e-12)
            record = {"case": case, "row": "equal_animal_mean", "LED_trial": led, "metric": metric,
                      "histogram_normalizer": np.nan, "full_histogram_area": area,
                      "expected_area": group.expected_area.mean(), "bin_width_s": width_by_metric[metric]}
            for field in ("selected_aborts", "plotted_observations", "eligible_denominator", "missing_selected_timed_fix"):
                record[field] = int(group[field].sum())
            panel_records.append(record)
            curve_parts.append(pd.DataFrame({"case": case, "row": "equal_animal_mean", "LED_trial": led,
                                             "metric": metric, "bin_left_s": bins[:-1], "bin_right_s": bins[1:],
                                             "height": mean, "sem": sem, "bin_count": np.nan}))
        if selection == "isolated" or metric in METRICS[:2]:
            np.testing.assert_array_equal(curves[metric][ci], reference_arrays[metric][ci])
            np.testing.assert_array_equal(sems[metric][ci], reference_arrays[f"{metric}_sem"][ci])

panels = pd.DataFrame(panel_records)
curve_audit = pd.concat(curve_parts, ignore_index=True)
assert len(panels) == len(CASES) * 7 * 2 * len(METRICS)
assert curve_audit.height.ge(0).all() and np.isfinite(curve_audit.height).all()
assert not curve_audit.duplicated(["case", "row", "LED_trial", "metric", "bin_left_s"]).any()
for ci, (st, selection) in enumerate(CASES):
    for row in rows:
        for metric in METRICS[2:]:
            ri = rows.index(row)
            difference = curves[metric][ci, ri, 0, :n_pre_bins] - curves[metric][ci, ri, 1, :n_pre_bins]
            max_rate = float(np.abs(difference).max())
            fraction = float(abs(difference.sum() * ABORT_BIN_WIDTH_S))
            mask = panels.case.eq(f"s{st}_{selection}") & panels.row.eq(row) & panels.metric.eq(metric)
            panels.loc[mask, "max_pre_bin_rate_difference_s_inv"] = max_rate
            panels.loc[mask, "pre_onset_fraction_difference"] = fraction
            if metric == METRICS[3]:
                assert max_rate <= MAX_BIN_RATE_DIFFERENCE + 1e-12
                assert fraction <= MAX_PRE_ONSET_FRACTION_DIFFERENCE + 1e-12
print("All bottom-row curves and top-row timing curves reproduced exactly; corrected top-row rates validated.", flush=True)


# %%
############ Save source identities, numerical curves, quotas and matching receipts ############
manifest_columns = ["source_csv_row_index", *keys, "session_type", "training_level", "repeat_trial",
                    "LED_trial", "abort_event", "success", "intended_fix", "LED_onset_time", "timed_fix",
                    "abort_from_LED_onset", "previous_is_abort", "isolated_abort", "selected_abort",
                    "pre_onset_candidate", "pre_onset_bin", "retained"]
for st, selection in CASES:
    case = f"s{st}_{selection}"
    manifest = case_frames[(st, selection)][manifest_columns].copy()
    manifest.insert(0, "source_csv", f"raw_data/LED7_s{st}.csv")
    manifest.insert(0, "case", case)
    manifest.to_csv(f"{OUTPUT_STEM}_{case}_selection_manifest.csv", index=False)
optimizer_audit.to_csv(f"{OUTPUT_STEM}_optimizer_diagnostics.csv", index=False)
quota_audit.to_csv(f"{OUTPUT_STEM}_bin_quotas.csv", index=False)
retention_audit.to_csv(f"{OUTPUT_STEM}_retention_audit.csv", index=False)
panels.to_csv(f"{OUTPUT_STEM}_count_area_audit.csv", index=False)
panels.loc[panels.row.eq("equal_animal_mean")].to_csv(f"{OUTPUT_STEM}_mean_area_audit.csv", index=False)
curve_audit.to_csv(f"{OUTPUT_STEM}_curve_audit.csv", index=False)
np.savez_compressed(f"{OUTPUT_STEM}_curves.npz", cases=np.array(case_names), rows=np.array(rows),
                    led_conditions=np.array([0, 1]), timing_bins=timing_bins, abort_bins=abort_bins,
                    **curves, **{f"{metric}_sem": sems[metric] for metric in METRICS})


# %%
############ 4 × 4 figure: all rate rows are frequencies among eligible trials ############
mean_panels = panels.loc[panels.row.eq("equal_animal_mean")].copy()
fig, axes = plt.subplots(4, 4, figsize=(20, 13.2), sharex="col", sharey=False)
rate_peak = max(float((curves[metric][:, 6] + sems[metric][:, 6]).max()) for metric in METRICS[2:])
rate_upper = float(np.ceil(rate_peak * 1.25 / 0.05) * 0.05)
timing_upper = {metric: float(np.ceil((curves[metric][:, 6] + sems[metric][:, 6]).max() * 1.18 / 0.25) * 0.25)
                for metric in METRICS[:2]}
for ci, (st, selection) in enumerate(CASES):
    case = case_names[ci]
    for mi, metric in enumerate(METRICS):
        ax, bins = axes[ci, mi], bins_by_metric[metric]
        for led in (0, 1):
            mean, sem = curves[metric][ci, 6, led], sems[metric][ci, 6, led]
            ax.fill_between(bins, np.r_[mean - sem, (mean - sem)[-1]], np.r_[mean + sem, (mean + sem)[-1]],
                            step="post", color=COLORS[led], alpha=0.16, linewidth=0)
            ax.stairs(mean, bins, color=COLORS[led], linewidth=1.6)
            audit = mean_panels.loc[mean_panels.case.eq(case) & mean_panels.LED_trial.eq(led)
                                    & mean_panels.metric.eq(metric)].iloc[0]
            label = "OFF" if led == 0 else "ON"
            note = (f"{label} n={int(audit.selected_aborts):,}" if mi < 2 else
                    f"{label} n={int(audit.plotted_observations):,}, N={int(audit.eligible_denominator):,}; "
                    f"area={audit.full_histogram_area:.3f}")
            ax.text(0.98, 0.97 - 0.085 * led, note, transform=ax.transAxes,
                    ha="right", va="top", color=COLORS[led], fontsize=8.4)
        if mi < 2:
            ax.set_xlim(0, 2.2)
            ax.set_ylim(0, timing_upper[metric])
        else:
            ax.axvline(0, color="0.45", linestyle=":", linewidth=1)
            ax.set_xlim(*ABORT_DISPLAY_S)
            ax.set_ylim(0, rate_upper)
            if mi == 3:
                audit = mean_panels.loc[mean_panels.case.eq(case) & mean_panels.metric.eq(metric)].iloc[0]
                kept = retention_audit.loc[retention_audit.case.eq(case)].iloc[0]
                note = (f"Selected retained: {int(kept.retained_selected_OFF + kept.retained_selected_ON):,}/"
                        f"{int(kept.original_selected_OFF + kept.original_selected_ON):,}\n"
                        f"Pre: max |Δrate| = {audit.max_pre_bin_rate_difference_s_inv:.4f} s⁻¹\n"
                        f"|Δfraction| = {100 * audit.pre_onset_fraction_difference:.3f} pp")
                ax.text(0.02, 0.75, note, transform=ax.transAxes, ha="left", va="top", fontsize=8)
        if mi == 0:
            ax.set_ylabel(f"Session type {st}\n{ROW_LABELS[selection]}\nDensity (s⁻¹)", fontsize=11)
        elif mi == 2:
            ax.set_ylabel("Abort rate per eligible trial (s⁻¹)", fontsize=10)
        if ci == 0:
            ax.set_title(("intended_fix", "LED_onset_time", "Abort rate aligned to LED onset",
                          "Pre-onset matched abort rate")[mi], fontsize=12, pad=12)
        if ci == len(CASES) - 1:
            ax.set_xlabel("Time from fixation onset (s)" if mi < 2 else
                          "Time from scheduled LED onset (s)", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.grid(axis="y", color="0.91", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
assert all(axes[ci, mi].get_ylim() == (0, rate_upper) for ci in range(4) for mi in (2, 3))
fig.suptitle("LED7 sequence-abort timing: equal-animal mean ± SEM", fontsize=19, y=0.995)
fig.legend(handles=[Line2D([0], [0], color=COLORS[led], lw=2,
                           label=f"Current trial LED {'OFF' if led == 0 else 'ON'}") for led in (0, 1)],
           loc="upper center", bbox_to_anchor=(0.5, 0.972), ncol=2, frameon=False, fontsize=12)
fig.text(0.5, 0.020,
         "Animals 92, 93, 98, 99, 100, 103. Timing: 40 ms unit-area densities; abort rates: 20 ms bins, full areas over −2 to +2 s.\n"
         "Each rate denominator is ALL event-3 or valid trials for that animal/session type/current LED condition, regardless of adjacent outcomes.\n"
         "Abort→abort selects every abort following an abort (including later aborts in longer runs); isolated selects valid→abort→valid.\n"
         "Curves average six animal-normalized rates; n/N labels are summed counts, while areas are mean animal fractions.\n"
         "Only pre-onset selected aborts are removed for column 4; all other eligible rows remain. OFF onset is scheduled/counterfactual. "
         "Outcome-based pre-onset matching is descriptive, not causal evidence.",
         ha="center", fontsize=8.9, linespacing=1.4)
fig.tight_layout(rect=(0.005, 0.145, 0.995, 0.945), h_pad=1.5, w_pad=1.6)
figure_path = Path(f"{OUTPUT_STEM}_4x4.png")
fig.savefig(figure_path, dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)


# %%
############ Final source and historical-figure preservation receipt ############
for path in protected_paths:
    with path.open("rb") as handle:
        assert hashlib.file_digest(handle, "sha256").hexdigest() == input_hashes[str(path.relative_to(REPO))]
summary = {
    "source_script": str(Path(__file__).relative_to(REPO)),
    "figure": str(figure_path.relative_to(REPO)),
    "source_sha256": input_hashes,
    "previous_figure_preserved": True,
    "animals": ANIMALS, "case_order": case_names,
    "classification": "Actual adjacent trial IDs within animal/session; original labels frozen before removal.",
    "denominator": "All abort_event == 3 OR success in {-1,+1} trials for animal/session type/current LED condition.",
    "alignment": "timed_fix - standardized fixation-referenced LED_onset_time; OFF onset scheduled/counterfactual.",
    "top_matching": "Fresh certified per-animal OFF/ON integer quotas under corrected all-eligible denominators.",
    "bottom_matching": "Exact saved isolated-abort retained source-row identities and all numerical curves reused.",
    "seed": SEED, "bin_rate_tolerance_s_inv": MAX_BIN_RATE_DIFFERENCE,
    "pre_onset_fraction_tolerance": MAX_PRE_ONSET_FRACTION_DIFFERENCE,
    "all_top_solvers_certified_optimal": bool(optimizer_audit.certified_optimal.all()),
    "retention": retention_audit.to_dict(orient="records"),
    "figure_y_upper_limits": {**timing_upper, "abort_rates": rate_upper},
    "averaging": "Equal-animal mean of six animal-normalized curves, sample SD/sqrt(6) SEM.",
    "interpretation": "Outcome-based matching is descriptive and does not establish a causal LED effect.",
    "numpy_version": np.__version__, "scipy_version": scipy.__version__,
}
Path(f"{OUTPUT_STEM}_run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(retention_audit.to_string(index=False))
print("Passed: all-eligible denominators, certified top matching, saved bottom curves, missing timing, full areas and equal-animal SEMs.")
print(f"Saved {figure_path}")
