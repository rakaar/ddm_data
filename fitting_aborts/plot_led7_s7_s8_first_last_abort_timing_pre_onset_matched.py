# %%
"""Extend the first/last/isolated 6x3 with outcome-based pre-onset matching.

The original three columns are unchanged. Only the fourth column removes rows,
independently by animal and selection, with classifications frozen beforehand.
Overlap is enforced using observed abort times: descriptive, not causal evidence.
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
ORIGINAL_STEM = SCRIPT_DIR / "led7_s7_s8_first_last_abort_timing_all_trials"
PREVIOUS_MATCH_STEM = SCRIPT_DIR / "led7_s7_s8_sequence_abort_pre_onset_matched_summary_with_first"
OUTPUT_STEM = SCRIPT_DIR / "led7_s7_s8_first_last_abort_timing_pre_onset_matched"
ANIMALS = (92, 93, 98, 99, 100, 103)
CASES = tuple((st, position) for position in ("first", "last", "isolated") for st in (7, 8))
SEED = 20261001  # Reset per case; new last-abort selections use this same rule.
MAX_BIN_RATE_DIFFERENCE = 0.005  # s^-1 in every 20 ms pre-onset bin, per animal.
MAX_PRE_ONSET_FRACTION_DIFFERENCE = 0.001
REUSE_PREVIOUS_FIRST_ISOLATED = True  # Preserve their exact saved row identities.
SOLVER_TIME_LIMIT_S = 60
TIMING_BIN_WIDTH_S = 0.040
ABORT_BIN_WIDTH_S = 0.020
ABORT_DISPLAY_S = (-0.4, 0.5)
FIGURE_DPI = 240
COLORS = {0: "tab:blue", 1: "tab:red"}
METRICS = ("intended_fix", "LED_onset_time", "abort_from_LED_onset", "matched_abort_from_LED_onset")
COLUMN_TITLES = ("intended_fix", "LED_onset_time", "Abort rate aligned to LED onset",
                 "Pre-onset matched abort rate")
ROW_LABELS = {"first": "First abort in a run", "last": "Last abort in a run",
              "isolated": "Valid → abort → valid"}
EXPECTED_SELECTED = {(7, "first"): (1425, 956), (8, "first"): (1309, 1284),
                     (7, "last"): (1218, 1163), (8, "last"): (786, 1807),
                     (7, "isolated"): (6498, 3365), (8, "isolated"): (6406, 4105)}
EXPECTED_DENOMINATORS = {7: (63525, 30910), 8: (68584, 38486)}

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

keys = ["animal", "session", "trial"]
timing_bins = np.linspace(0, 2.2, round(2.2 / TIMING_BIN_WIDTH_S) + 1)
abort_bins = np.linspace(-2, 2, round(4 / ABORT_BIN_WIDTH_S) + 1)
pre_bins = np.linspace(-2, 0, round(2 / ABORT_BIN_WIDTH_S) + 1)
n_pre_bins = len(pre_bins) - 1
np.testing.assert_array_equal(abort_bins[:n_pre_bins + 1], pre_bins)
bins_by_metric = dict(zip(METRICS, (timing_bins, timing_bins, abort_bins, abort_bins)))
width_by_metric = dict(zip(METRICS, (TIMING_BIN_WIDTH_S, TIMING_BIN_WIDTH_S,
                                   ABORT_BIN_WIDTH_S, ABORT_BIN_WIDTH_S)))
case_names = [f"s{st}_{position}" for st, position in CASES]
row_names = [*[str(animal) for animal in ANIMALS], "equal_animal_mean"]
protected = list(SCRIPT_DIR.glob(f"{ORIGINAL_STEM.name}*"))
protected += [SCRIPT_DIR / "plot_led7_s7_s8_first_last_abort_timing_all_trials.py"]
protected += [REPO / f"raw_data/LED7_s{st}.csv" for st in (7, 8)]
protected += list((REPO / "docs/assets/results/2026-10-04").glob("*.png"))
protected += [Path(f"{PREVIOUS_MATCH_STEM}_curves.npz")]
protected += [Path(f"{PREVIOUS_MATCH_STEM}_s{st}_{position}_selection_manifest.csv")
              for st, position in CASES if position != "last"]
input_hashes = {}
for path in protected:
    with path.open("rb") as handle:
        input_hashes[str(path.relative_to(REPO))] = hashlib.file_digest(handle, "sha256").hexdigest()


# %%
############ Same conservative MILP rule used by the earlier matching plots ############
def optimize_pre_onset_counts(available, denominators, saved_quotas=None):
    """Maximize retention under the linear rule and certify the integer optimum."""
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
    if saved_quotas is not None:
        quotas = saved_quotas  # Preserve old identities if more than one optimum exists.
    assert ((quotas >= 0) & (quotas <= available)).all()
    assert np.all(np.abs(matrix @ quotas.ravel()) <= tolerance + 1e-7)
    assert np.isclose(solution.fun, -quotas.sum(), rtol=0, atol=1e-7)
    assert np.isclose(solution.mip_dual_bound, solution.fun, rtol=0, atol=1e-7)
    new_n = mandatory + quotas.sum(axis=1)
    rates = quotas / (new_n[:, None] * ABORT_BIN_WIDTH_S)
    max_difference = float(np.abs(rates[0] - rates[1]).max())
    fractions = quotas.sum(axis=1) / new_n
    fraction_difference = float(abs(fractions[0] - fractions[1]))
    assert max_difference <= MAX_BIN_RATE_DIFFERENCE + 1e-12
    assert fraction_difference <= MAX_PRE_ONSET_FRACTION_DIFFERENCE + 1e-12
    return quotas, {
        "solver": "scipy.optimize.milp / HiGHS", "certified_optimal": True,
        "status": solution.status, "message": solution.message, "mip_gap": solution.mip_gap,
        "objective": solution.fun, "dual_bound": solution.mip_dual_bound,
        "mip_node_count": solution.mip_node_count, "elapsed_s": time.perf_counter() - started,
        "original_OFF": denominators[0], "original_ON": denominators[1],
        "mandatory_OFF": mandatory[0], "mandatory_ON": mandatory[1],
        "retained_OFF": new_n[0], "retained_ON": new_n[1], "removed": int((available - quotas).sum()),
        "r": ratio, "u_max": u_max, "epsilon": epsilon,
        "max_pre_bin_rate_difference_s_inv": max_difference,
        "pre_onset_fraction_difference": fraction_difference,
    }


# %%
############ Verify the saved source identities against intact CSV trial streams ############
selected_manifest = pd.read_csv(f"{ORIGINAL_STEM}_selected_trials.csv", float_precision="round_trip")
denominator_manifest = pd.read_csv(f"{ORIGINAL_STEM}_denominator_trials.csv", float_precision="round_trip")
original_panels = pd.read_csv(f"{ORIGINAL_STEM}_count_area_audit.csv", float_precision="round_trip")
with np.load(f"{ORIGINAL_STEM}_curves.npz") as payload:
    original_arrays = {name: payload[name].copy() for name in payload.files}
with np.load(f"{PREVIOUS_MATCH_STEM}_curves.npz") as payload:
    previous_arrays = {name: payload[name].copy() for name in payload.files}
np.testing.assert_array_equal(original_arrays["cases"], case_names)
np.testing.assert_array_equal(original_arrays["rows"], row_names)
np.testing.assert_array_equal(original_arrays["timing_bins"], timing_bins)
np.testing.assert_array_equal(original_arrays["abort_bins"], abort_bins)
streams = {}
for st in (7, 8):
    raw = pd.read_csv(REPO / f"raw_data/LED7_s{st}.csv", float_precision="round_trip")
    assert raw.shape == ({7: 138440, 8: 148428}[st], 52)
    raw_columns = list(raw.columns)
    assert raw.training_level.eq(16).all() and raw.session_type.eq(st).all()
    assert (raw.repeat_trial.isin([0, 2]) | raw.repeat_trial.isna()).all()
    assert raw.LED_trial.isin([0, 1]).all() and not raw.duplicated(keys).any()
    raw["source_csv_row_index"] = np.arange(len(raw))
    data = raw.loc[raw.animal.isin(ANIMALS)].sort_values(keys, kind="stable").copy()
    assert tuple(sorted(data.animal.unique())) == ANIMALS
    expected = {position: set() for position in ("first", "last", "isolated")}
    # Direct sequential scan independently reproduces the whole-run endpoints.
    for _, session in data.groupby(["animal", "session"], sort=False):
        trial = session.trial.to_numpy()
        is_abort = session.abort_event.eq(3).to_numpy()
        is_valid = session.success.isin([-1, 1]).to_numpy()
        ids = session.source_csv_row_index.to_numpy()
        i = 0
        while i < len(session):
            if not is_abort[i]:
                i += 1
                continue
            end = i
            while end + 1 < len(session) and is_abort[end + 1] and trial[end + 1] == trial[end] + 1:
                end += 1
            complete = (i > 0 and end + 1 < len(session) and trial[i - 1] == trial[i] - 1
                        and trial[end + 1] == trial[end] + 1 and not is_abort[i - 1] and not is_abort[end + 1])
            if complete and end > i:
                expected["first"].add(ids[i])
                expected["last"].add(ids[end])
            if complete and end == i and is_valid[i - 1] and is_valid[end + 1]:
                expected["isolated"].add(ids[i])
            i = end + 1
    eligible = data.loc[data.abort_event.eq(3) | data.success.isin([-1, 1])].copy()
    saved = denominator_manifest.loc[denominator_manifest.session_type.eq(st)].set_index("source_csv_row_index").sort_index()
    actual = eligible.set_index("source_csv_row_index").sort_index()
    compare = [column for column in raw_columns if column in saved]
    pd.testing.assert_frame_equal(actual[compare], saved[compare], check_dtype=False)
    assert tuple(eligible.groupby("LED_trial").size()) == EXPECTED_DENOMINATORS[st]
    assert eligible.loc[eligible.abort_event.eq(3), "timed_fix"].isna().sum() == 2
    for position, expected_ids in expected.items():
        selected = selected_manifest.loc[selected_manifest.case.eq(f"s{st}_{position}")]
        assert selected.source_csv_row_index.is_unique and set(selected.source_csv_row_index) == expected_ids
        assert tuple(selected.groupby("LED_trial").size()) == EXPECTED_SELECTED[(st, position)]
        pd.testing.assert_frame_equal(actual.loc[sorted(expected_ids), raw_columns],
                                      selected.set_index("source_csv_row_index").sort_index()[raw_columns], check_dtype=False)
        assert np.isfinite(selected[["intended_fix", "LED_onset_time", "timed_fix"]]).all().all()
        assert selected.timed_fix.lt(selected.intended_fix).all()
        eligible[f"selected_{position}"] = eligible.source_csv_row_index.isin(expected_ids)
    eligible["abort_from_LED_onset"] = eligible.timed_fix - eligible.LED_onset_time
    streams[st] = eligible.set_index("source_csv_row_index", drop=False)


# %%
############ Match independently per case/animal; never reclassify after removal ############
case_frames, optimizer_records, quota_records, retention_records = {}, [], [], []
for st, position in CASES:
    case_id = f"s{st}_{position}"
    frame = streams[st].copy()
    frame["selected_abort"] = frame[f"selected_{position}"]
    frame["pre_onset_candidate"] = (frame.selected_abort & np.isfinite(frame.abort_from_LED_onset)
                                     & frame.abort_from_LED_onset.lt(0))
    pre = frame.pre_onset_candidate
    assert frame.loc[pre, "abort_from_LED_onset"].ge(-2).all()
    frame["pre_onset_bin"] = -1
    frame.loc[pre, "pre_onset_bin"] = np.searchsorted(pre_bins, frame.loc[pre, "abort_from_LED_onset"], side="right") - 1
    assert frame.loc[pre, "pre_onset_bin"].between(0, n_pre_bins - 1).all()
    frame["retained"] = True
    reuse = REUSE_PREVIOUS_FIRST_ISOLATED and position in ("first", "isolated")
    if reuse:
        assert (MAX_BIN_RATE_DIFFERENCE, MAX_PRE_ONSET_FRACTION_DIFFERENCE, ABORT_BIN_WIDTH_S) == (0.005, 0.001, 0.020)
        saved = pd.read_csv(f"{PREVIOUS_MATCH_STEM}_{case_id}_selection_manifest.csv", float_precision="round_trip")
        saved = saved.set_index("source_csv_row_index").sort_index()
        fields = [*keys, "LED_trial", "abort_event", "success", "intended_fix", "LED_onset_time", "timed_fix",
                  "selected_abort", "pre_onset_candidate", "pre_onset_bin"]
        pd.testing.assert_frame_equal(frame.sort_index()[fields], saved[fields], check_dtype=False)
        frame["retained"] = saved.retained.reindex(frame.index)
        assert frame.retained.notna().all()
    rng = np.random.default_rng(SEED)
    for animal in ANIMALS:
        group = frame.loc[frame.animal.eq(animal)]
        denominators = group.groupby("LED_trial").size().reindex([0, 1]).to_numpy(int)
        available = np.stack([np.histogram(group.loc[group.LED_trial.eq(led) & group.pre_onset_candidate,
                                                    "abort_from_LED_onset"], bins=pre_bins)[0] for led in (0, 1)])
        saved_quotas = None
        if reuse:
            saved_quotas = np.stack([np.histogram(group.loc[group.LED_trial.eq(led) & group.pre_onset_candidate
                                                           & group.retained, "abort_from_LED_onset"], bins=pre_bins)[0]
                                    for led in (0, 1)])
        quotas, diagnostic = optimize_pre_onset_counts(available, denominators, saved_quotas)
        for led in (0, 1):
            for bi in range(n_pre_bins):
                pool = np.sort(group.index[group.LED_trial.eq(led) & group.pre_onset_candidate & group.pre_onset_bin.eq(bi)].to_numpy())
                assert len(pool) == available[led, bi]
                if reuse:
                    chosen = pool[frame.loc[pool, "retained"].to_numpy()]
                else:
                    chosen = rng.choice(pool, size=quotas[led, bi], replace=False)
                    frame.loc[pool, "retained"] = False
                    frame.loc[chosen, "retained"] = True
                assert len(chosen) == quotas[led, bi] and len(np.unique(chosen)) == len(chosen)
                quota_records.append({"case": case_id, "animal": animal, "LED_trial": led, "bin_index": bi,
                                      "bin_left_s": pre_bins[bi], "bin_right_s": pre_bins[bi + 1],
                                      "available": len(pool), "retained": len(chosen), "removed": len(pool) - len(chosen)})
        optimizer_records.append({"case": case_id, "animal": animal, "reused_saved_identities": reuse, **diagnostic})
    kept, removed = frame.loc[frame.retained], frame.loc[~frame.retained]
    assert frame.loc[~frame.pre_onset_candidate, "retained"].all()
    assert removed.selected_abort.all() and removed.abort_event.eq(3).all() and removed.abort_from_LED_onset.lt(0).all()
    assert kept.index.is_unique and not kept.duplicated(keys).any()
    assert len(kept) + len(removed) == len(frame)
    for animal in ANIMALS:
        for led in (0, 1):
            original = frame.loc[frame.animal.eq(animal) & frame.LED_trial.eq(led)]
            retained = original.loc[original.retained]
            for mask in (original.success.isin([-1, 1]), original.abort_from_LED_onset.ge(0),
                         ~np.isfinite(original.abort_from_LED_onset), ~original.selected_abort):
                assert original.loc[mask, "retained"].all()
            assert len(original) - len(retained) == original.selected_abort.sum() - retained.selected_abort.sum()
    if reuse:
        np.testing.assert_array_equal(frame.sort_index().retained, saved.retained)
    retention_records.append({
        "case": case_id, "original_selected": int(frame.selected_abort.sum()),
        "retained_selected": int(kept.selected_abort.sum()), "original_denominator": len(frame),
        "retained_denominator": len(kept), "removed": len(removed),
        "retained_selected_OFF": int(kept.loc[kept.LED_trial.eq(0), "selected_abort"].sum()),
        "retained_selected_ON": int(kept.loc[kept.LED_trial.eq(1), "selected_abort"].sum()),
        "valid_trials_preserved": int(kept.success.isin([-1, 1]).sum()),
        "post_onset_selected_preserved": int((kept.selected_abort & kept.abort_from_LED_onset.ge(0)).sum()),
        "missing_timed_fix_preserved": int(kept.timed_fix.isna().sum()), "reused_saved_identities": reuse,
    })
    # Compact manifest: all eligible rows, not only the selected events.
    columns = ["source_csv_row_index", *keys, "session_type", "training_level", "repeat_trial", "LED_trial",
               "abort_event", "success", "timed_fix", "intended_fix", "LED_onset_time", "abort_from_LED_onset",
               "selected_first", "selected_last", "selected_isolated", "selected_abort",
               "pre_onset_candidate", "pre_onset_bin", "retained"]
    manifest = frame[columns].copy()
    manifest.insert(0, "source_csv", f"raw_data/LED7_s{st}.csv")
    manifest.insert(0, "case", case_id)
    manifest.to_csv(f"{OUTPUT_STEM}_{case_id}_selection_manifest.csv", index=False)
    case_frames[(st, position)] = frame
    print(f"{case_id}: retained {kept.selected_abort.sum():,}/{frame.selected_abort.sum():,} selected aborts; "
          f"{len(kept):,}/{len(frame):,} eligible trials. Certified optimum under the matching rule.", flush=True)


# %%
############ Reproduce original columns, then compute retained-denominator rates ############
curves = {metric: np.zeros((6, 7, 2, len(bins_by_metric[metric]) - 1)) for metric in METRICS}
sems = {metric: np.zeros_like(curves[metric]) for metric in METRICS}
panel_records, curve_parts = [], []
for ci, case in enumerate(CASES):
    case_id, frame = case_names[ci], case_frames[case]
    for ai, animal in enumerate(ANIMALS):
        for led in (0, 1):
            group = frame.loc[frame.animal.eq(animal) & frame.LED_trial.eq(led)]
            for metric in METRICS:
                matched = metric == METRICS[3]
                current = group.loc[group.retained] if matched else group
                selected = current.loc[current.selected_abort]
                values = selected["abort_from_LED_onset" if matched else metric].to_numpy()
                values = values[np.isfinite(values)]
                bins, width = bins_by_metric[metric], width_by_metric[metric]
                counts = np.histogram(values, bins=bins)[0]
                assert counts.sum() == len(values) and len(current) > 0
                normalizer = len(current) if metric in METRICS[2:] else len(values)
                heights = counts / (normalizer * width)
                area = float(heights @ np.diff(bins))
                assert np.isclose(area, len(values) / normalizer, atol=1e-12, rtol=0)
                curves[metric][ci, ai, led] = heights
                panel_records.append({"case": case_id, "row": str(animal), "LED_trial": led, "metric": metric,
                                      "selected_aborts": len(selected), "plotted_observations": len(values),
                                      "original_denominator": len(group), "eligible_denominator": len(current),
                                      "missing_selected_timing": int((~np.isfinite(selected.abort_from_LED_onset)).sum()),
                                      "histogram_normalizer": normalizer, "full_histogram_area": area,
                                      "expected_area": len(values) / normalizer, "bin_width_s": width})
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
            animal_audit = animal_panels.loc[animal_panels.metric.eq(metric) & animal_panels.LED_trial.eq(led)]
            assert len(animal_audit) == 6
            area = float(mean @ np.diff(bins))
            assert np.isclose(area, animal_audit.expected_area.mean(), rtol=0, atol=1e-12)
            record = {"case": case_id, "row": "equal_animal_mean", "LED_trial": led, "metric": metric,
                      "histogram_normalizer": np.nan, "full_histogram_area": area,
                      "expected_area": animal_audit.expected_area.mean(), "bin_width_s": width_by_metric[metric]}
            for field in ("selected_aborts", "plotted_observations", "original_denominator", "eligible_denominator", "missing_selected_timing"):
                record[field] = int(animal_audit[field].sum())
            panel_records.append(record)
            curve_parts.append(pd.DataFrame({"case": case_id, "row": "equal_animal_mean", "LED_trial": led, "metric": metric,
                                             "bin_left_s": bins[:-1], "bin_right_s": bins[1:], "height": mean, "sem": sem, "bin_count": np.nan}))
        if metric in METRICS[:3]:
            np.testing.assert_array_equal(curves[metric][ci], original_arrays[metric][ci])
            np.testing.assert_array_equal(sems[metric][ci], original_arrays[f"{metric}_sem"][ci])
        elif REUSE_PREVIOUS_FIRST_ISOLATED and case[1] != "last":
            old_ci = previous_arrays["cases"].tolist().index(case_id)
            np.testing.assert_array_equal(curves[metric][ci], previous_arrays[metric][old_ci])
            np.testing.assert_array_equal(sems[metric][ci], previous_arrays[f"{metric}_sem"][old_ci])

panels, curve_audit = pd.DataFrame(panel_records), pd.concat(curve_parts, ignore_index=True)
for ci, case_id in enumerate(case_names):
    for ai, row in enumerate(row_names):
        for metric in METRICS[2:]:
            off, on = curves[metric][ci, ai, :, :n_pre_bins]
            delta_rate = float(np.abs(off - on).max())
            delta_fraction = float(abs((off - on).sum() * ABORT_BIN_WIDTH_S))
            mask = panels.case.eq(case_id) & panels.row.eq(row) & panels.metric.eq(metric)
            panels.loc[mask, "max_pre_bin_rate_difference_s_inv"] = delta_rate
            panels.loc[mask, "pre_onset_fraction_difference"] = delta_fraction
            if metric == METRICS[3]:
                assert delta_rate <= MAX_BIN_RATE_DIFFERENCE + 1e-12
                assert delta_fraction <= MAX_PRE_ONSET_FRACTION_DIFFERENCE + 1e-12
assert len(panels) == 6 * 7 * 2 * 4
assert not curve_audit.duplicated(["case", "row", "LED_trial", "metric", "bin_left_s"]).any()
assert np.isfinite(curve_audit.height).all() and curve_audit.height.ge(0).all()
old_check = original_panels.set_index(["case", "row", "LED_trial", "metric"]).sort_index()
new_check = panels.loc[panels.metric.isin(METRICS[:3])].set_index(["case", "row", "LED_trial", "metric"]).sort_index()
pd.testing.assert_frame_equal(old_check[["selected_aborts", "plotted_observations", "eligible_denominator", "full_histogram_area"]],
                              new_check[["selected_aborts", "plotted_observations", "eligible_denominator", "full_histogram_area"]], check_exact=True)
optimizer, quotas, retention = (pd.DataFrame(records) for records in (optimizer_records, quota_records, retention_records))
assert len(optimizer) == 36 and optimizer.certified_optimal.all()
optimizer.to_csv(f"{OUTPUT_STEM}_optimizer_diagnostics.csv", index=False)
quotas.to_csv(f"{OUTPUT_STEM}_bin_quotas.csv", index=False)
retention.to_csv(f"{OUTPUT_STEM}_retention_audit.csv", index=False)
panels.to_csv(f"{OUTPUT_STEM}_count_area_audit.csv", index=False)
mean_panels = panels.loc[panels.row.eq("equal_animal_mean")].copy()
mean_panels.to_csv(f"{OUTPUT_STEM}_mean_area_audit.csv", index=False)
curve_audit.to_csv(f"{OUTPUT_STEM}_curve_audit.csv", index=False)
np.savez_compressed(f"{OUTPUT_STEM}_curves.npz", cases=np.array(case_names), rows=np.array(row_names),
                    led_conditions=np.array([0, 1]), timing_bins=timing_bins, abort_bins=abort_bins,
                    **curves, **{f"{metric}_sem": sems[metric] for metric in METRICS})


# %%
############ Original three columns plus the matched rate; identical rate axes ############
fig, axes = plt.subplots(6, 4, figsize=(21, 19), sharex="col", sharey="col")
visible = (abort_bins[:-1] < ABORT_DISPLAY_S[1]) & (abort_bins[1:] > ABORT_DISPLAY_S[0])
old_summary = json.loads(Path(f"{ORIGINAL_STEM}_run_summary.json").read_text())
upper_limits = [old_summary["figure_y_upper_limits"][metric] for metric in METRICS[:3]]
matched_max = float((curves[METRICS[3]][:, 6] + sems[METRICS[3]][:, 6])[..., visible].max())
rate_upper = max(upper_limits[2], 1.30 * matched_max)
upper_limits[2:] = [rate_upper, rate_upper]
for ci, (st, position) in enumerate(CASES):
    case_id = case_names[ci]
    kept = retention.loc[retention.case.eq(case_id)].iloc[0]
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
            note = (f"{label} n={int(audit.plotted_observations):,}" if col < 2 else
                    f"{label} n={int(audit.plotted_observations):,}, N={int(audit.eligible_denominator):,}; area={audit.full_histogram_area:.3f}")
            ax.text(0.98, 0.97 - 0.08 * led, note, transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=COLORS[led])
        if col >= 2:
            ax.axvline(0, color="0.45", linestyle=":", linewidth=1)
        if col == 3:
            ax.text(0.02, 0.76,
                    f"Retained {int(kept.retained_selected):,}/{int(kept.original_selected):,} selected aborts\n"
                    f"Pre: max |Δrate|={audit.max_pre_bin_rate_difference_s_inv:.4f} s⁻¹\n"
                    f"|Δfraction|={100 * audit.pre_onset_fraction_difference:.3f} pp",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8)
        ax.set_xlim(*(ABORT_DISPLAY_S if col >= 2 else (0, 2.2)))
        ax.set_ylim(0, upper_limits[col])
        if col == 0:
            ax.set_ylabel(f"Session type {st}\n{ROW_LABELS[position]}\nDensity (s⁻¹)", fontsize=11)
        elif col == 2:
            ax.set_ylabel("Abort rate per eligible trial (s⁻¹)", fontsize=10)
        if ci == 0:
            ax.set_title(COLUMN_TITLES[col], fontsize=13, pad=12)
        if ci == 5:
            ax.set_xlabel("Time from scheduled LED onset (s)" if col >= 2 else "Time from fixation onset (s)", fontsize=11)
        ax.grid(axis="y", color="0.91", linewidth=0.5)
        ax.tick_params(labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("LED7 first, last and isolated aborts: equal-animal mean ± SEM", fontsize=19, y=0.994)
fig.legend(handles=[Line2D([0], [0], color=COLORS[led], lw=2, label=f"Plotted trial LED {'OFF' if led == 0 else 'ON'}")
                    for led in (0, 1)], loc="upper center", bbox_to_anchor=(0.5, 0.973), ncol=2, frameon=False, fontsize=12)
fig.text(0.5, 0.025,
         "Animals 92, 93, 98, 99, 100, 103. First/last: one endpoint each per complete run of ≥2 adjacent aborts; isolated: valid→abort→valid.\n"
         "Columns 1–3 unchanged. Rate denominator: all event-3 OR success ±1 trials in each animal/session-type/LED group; column 4 recomputes it after removals.\n"
         "Only pre-onset selected aborts removed; valid, post-onset, zero-time and missing-time rows retained. Original sequence labels frozen; matching is separate for each animal and row.\n"
         "Timing: 40 ms unit-area bins. Rates: 20 ms bins; full areas over −2 to +2 s. n/N are summed counts; areas average six animal fractions. OFF onset is scheduled/counterfactual.\n"
         "Pre-onset overlap is enforced using observed outcomes: descriptive, not causal evidence. No post-onset counts are removed, but their rates change with the new denominators.",
         ha="center", fontsize=9, linespacing=1.45)
fig.tight_layout(rect=(0.005, 0.108, 0.998, 0.947), h_pad=1.8, w_pad=1.7)
figure_path = Path(f"{OUTPUT_STEM}_6x4.png")
fig.savefig(figure_path, dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)
for path in protected:
    with path.open("rb") as handle:
        assert hashlib.file_digest(handle, "sha256").hexdigest() == input_hashes[str(path.relative_to(REPO))]
summary = {
    "source_script": str(Path(__file__).relative_to(REPO)), "figure": str(figure_path.relative_to(REPO)),
    "input_sha256": input_hashes, "old_artifacts_preserved": True, "case_order": case_names,
    "animals": ANIMALS, "seed_per_new_case": SEED, "numpy_version": np.__version__, "scipy_version": scipy.__version__,
    "max_bin_rate_difference_s_inv": MAX_BIN_RATE_DIFFERENCE,
    "max_pre_onset_fraction_difference": MAX_PRE_ONSET_FRACTION_DIFFERENCE,
    "all_36_optimizations_certified": bool(optimizer.certified_optimal.all()),
    "optimality_scope": "Maximum retention under the specified conservative linear rule, independently per animal/case.",
    "retention": retention.to_dict(orient="records"),
    "previous_first_isolated_identities_reused": REUSE_PREVIOUS_FIRST_ISOLATED,
    "original_three_columns": "Exact original animal curves, equal-animal means, SEMs, counts and areas reproduced.",
    "denominator": "All event-3 OR success +/-1 trials; matched column excludes only removed pre-onset selected aborts.",
    "classification": "Complete-run first/last and isolated VAV labels frozen before matching; actual trial-ID adjacency.",
    "onset": "CSV LED_onset_time directly, fixation-referenced; OFF is scheduled/counterfactual.",
    "averaging": "Equal mean of six animal-normalized curves, sample SD/sqrt(6) SEM.",
    "caveat": "Outcome-based descriptive matching; enforced overlap is not causal evidence. Post-onset counts unchanged, rates renormalized.",
}
Path(f"{OUTPUT_STEM}_run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(retention.to_string(index=False))
print("All checks passed: original identities/curves, protected rows, maximal integer retention, matching tolerances, areas, means/SEMs and old-file hashes.")
print(f"Saved {figure_path}")
