# %%
"""Equal-animal sequence-abort timing, before/after descriptive onset matching.

Rows are s7/s8 first aborts in complete runs, then abort-after-abort and isolated
valid-abort-valid. Set INCLUDE_FIRST_ROWS=False to reproduce the earlier 4x4.
Classifications and conditional denominators are established on the original
trial stream and frozen before removing any rows. Matching uses observed abort
outcomes, so enforced overlap is not evidence of a causal LED effect.
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
BASE_OUTPUT_STEM = SCRIPT_DIR / "led7_s7_s8_sequence_abort_pre_onset_matched_summary"
INCLUDE_FIRST_ROWS = True
OUTPUT_STEM = Path(f"{BASE_OUTPUT_STEM}_with_first") if INCLUDE_FIRST_ROWS else BASE_OUTPUT_STEM
ANIMALS = (92, 93, 98, 99, 100, 103)
ORIGINAL_CASES = ((7, "after_abort"), (8, "after_abort"), (7, "isolated"), (8, "isolated"))
CASES = ((7, "first"), (8, "first"), *ORIGINAL_CASES) if INCLUDE_FIRST_ROWS else ORIGINAL_CASES
SEQUENCE_MANIFEST_PATH = SCRIPT_DIR / "led7_s7_s8_abort_rates_by_sequence_position_sequence_manifest.csv"
SEED = 20261001  # Reset independently for each new selection.
MAX_BIN_RATE_DIFFERENCE = 0.005  # s^-1, each animal, every pre-onset bin.
MAX_PRE_ONSET_FRACTION_DIFFERENCE = 0.001
TIMING_BIN_WIDTH_S = 0.040
ABORT_BIN_WIDTH_S = 0.020
ABORT_DISPLAY_S = (-0.4, 0.5)
FIRST_ABORT_RATE_YLIM = (0.0, 0.2)
SOLVER_TIME_LIMIT_S = 60
FIGURE_DPI = 240
COLORS = {0: "tab:blue", 1: "tab:red"}
METRICS = ("intended_fix", "LED_onset_time", "abort_from_LED_onset", "matched_abort_from_LED_onset")
COLUMN_TITLES = ("intended_fix", "LED_onset_time", "Abort rate aligned to LED onset",
                 "Pre-onset matched abort rate")
EXISTING_MATCH_STEM = SCRIPT_DIR / "led7_s8_isolated_abort_pre_onset_matched_timing"
# Each condition: original selected, original finite aligned, original denominator.
EXPECTED_ORIGINAL = {
    (7, "first"): {0: (1425, 1425, 63525), 1: (956, 956, 30910)},
    (8, "first"): {0: (1309, 1309, 68584), 1: (1284, 1284, 38486)},
    (7, "after_abort"): {0: (1528, 1526, 8630), 1: (1569, 1569, 6988)},
    (8, "after_abort"): {0: (963, 963, 6181), 1: (2463, 2461, 10671)},
    (7, "isolated"): {0: (6498, 6498, 63525), 1: (3365, 3365, 30910)},
    (8, "isolated"): {0: (6406, 6406, 68584), 1: (4105, 4105, 38486)},
}
# Matched selected OFF/ON, then combined eligible denominator (includes missing times).
EXPECTED_MATCHED = {(7, "first"): (1158, 743, 93955), (8, "first"): (1078, 1120, 106675),
                    (7, "after_abort"): (771, 903, 14195), (8, "after_abort"): (552, 1735, 15713),
                    (7, "isolated"): (5372, 3144, 93088), (8, "isolated"): (4846, 3959, 105364)}

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

timing_bins = np.linspace(0, 2.2, round(2.2 / TIMING_BIN_WIDTH_S) + 1)
abort_bins = np.linspace(-2, 2, round(4 / ABORT_BIN_WIDTH_S) + 1)
pre_bins = np.linspace(-2, 0, round(2 / ABORT_BIN_WIDTH_S) + 1)
n_pre_bins = len(pre_bins) - 1
np.testing.assert_array_equal(abort_bins[:n_pre_bins + 1], pre_bins)
bins_by_metric = dict(zip(METRICS, (timing_bins, timing_bins, abort_bins, abort_bins)))
width_by_metric = dict(zip(METRICS, (TIMING_BIN_WIDTH_S, TIMING_BIN_WIDTH_S,
                                   ABORT_BIN_WIDTH_S, ABORT_BIN_WIDTH_S)))
original_stems = {
    case: SCRIPT_DIR / f"led7_s{case[0]}_{'abort_after_abort' if case[1] == 'after_abort' else 'isolated_abort'}_timing"
    for case in ORIGINAL_CASES
}
protected_paths = [REPO / f"raw_data/LED7_s{st}.csv" for st in (7, 8)]
for stem in original_stems.values():
    protected_paths.extend(Path(f"{stem}{suffix}") for suffix in
                           ("_selected_trials.csv", "_count_area_audit.csv", "_curve_audit.csv", "_7x3.png"))
protected_paths.extend(Path(f"{EXISTING_MATCH_STEM}{suffix}") for suffix in
                       ("_eligible_trial_selection_manifest.csv", "_bin_quotas.csv", "_optimizer_diagnostics.csv",
                        "_curve_audit.csv", "_count_area_audit.csv", "_7x4.png", "_run_summary.json"))
if INCLUDE_FIRST_ROWS:
    protected_paths.extend([SEQUENCE_MANIFEST_PATH, Path(f"{BASE_OUTPUT_STEM}_curve_audit.csv"),
                            Path(f"{BASE_OUTPUT_STEM}_4x4.png")])
    protected_paths.extend(Path(f"{BASE_OUTPUT_STEM}_s{st}_{selection}_selection_manifest.csv")
                           for st, selection in ORIGINAL_CASES)
input_hashes = {}
for path in protected_paths:
    with path.open("rb") as handle:
        input_hashes[str(path.relative_to(REPO))] = hashlib.file_digest(handle, "sha256").hexdigest()


# %%
############ Reusable integer-quota step: original conservative linear rule ############
def optimize_pre_onset_counts(available, denominators, saved_quotas=None):
    """Certify maximal retention; optionally certify an existing quota solution."""
    mandatory = denominators - available.sum(axis=1)
    ratio = mandatory[0] / mandatory[1]
    u_max = float((available / mandatory[:, None]).max())
    epsilon = ABORT_BIN_WIDTH_S * MAX_BIN_RATE_DIFFERENCE - u_max * MAX_PRE_ONSET_FRACTION_DIFFERENCE
    assert (mandatory > 0).all() and epsilon > 0
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
        quotas = saved_quotas  # Keep old row identities, even if another optimum exists.
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
    audit = {
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
    return quotas, audit


# %%
############ Establish both sequence definitions on the intact CSV trial streams ############
keys = ["animal", "session", "trial"]
streams, raw_columns_by_session = {}, {}
for session_type in (7, 8):
    source = pd.read_csv(REPO / f"raw_data/LED7_s{session_type}.csv", float_precision="round_trip")
    assert source.shape == ({7: 138440, 8: 148428}[session_type], 52)
    raw_columns_by_session[session_type] = list(source.columns)
    assert source.training_level.eq(16).all() and source.session_type.eq(session_type).all()
    assert (source.repeat_trial.isin([0, 2]) | source.repeat_trial.isna()).all()
    assert source.LED_trial.isin([0, 1]).all() and not source.duplicated(keys).any()
    source["source_csv_row_index"] = np.arange(len(source))
    data = source.loc[source.animal.isin(ANIMALS)].sort_values(keys, kind="stable").copy()
    assert len(data) == {7: 100251, 8: 113941}[session_type]
    for column in [*keys, "LED_trial"]:
        assert data[column].notna().all() and data[column].eq(np.floor(data[column])).all()
        data[column] = data[column].astype(int)
    data = data.set_index("source_csv_row_index", drop=False)
    grouped = data.groupby(["animal", "session"], sort=False)
    neighbor_columns = ["trial", "success", "abort_event", "LED_trial", "source_csv_row_index"]
    previous, following = grouped[neighbor_columns].shift(1), grouped[neighbor_columns].shift(-1)
    is_abort, is_valid = data.abort_event.eq(3), data.success.isin([-1, 1])
    assert not (is_abort & is_valid).any()
    data["event_or_valid"] = is_abort | is_valid
    data["previous_is_abort"] = previous.trial.eq(data.trial - 1) & previous.abort_event.eq(3)
    data["isolated_abort"] = (is_abort & previous.trial.eq(data.trial - 1)
                              & following.trial.eq(data.trial + 1)
                              & previous.success.isin([-1, 1]) & following.success.isin([-1, 1]))
    for prefix, neighbors in (("previous", previous), ("next", following)):
        for column in neighbor_columns:
            data[f"{prefix}_{column}"] = neighbors[column]
    # Independent trial-key lookup confirms that filtered-out IDs cannot be bridged.
    lookup = data.set_index(keys)
    prev_keys = pd.MultiIndex.from_arrays([data.animal, data.session, data.trial - 1], names=keys)
    next_keys = pd.MultiIndex.from_arrays([data.animal, data.session, data.trial + 1], names=keys)
    np.testing.assert_array_equal(data.previous_is_abort, lookup.reindex(prev_keys).abort_event.eq(3))
    np.testing.assert_array_equal(data.isolated_abort, is_abort.to_numpy()
                                  & lookup.reindex(prev_keys).success.isin([-1, 1]).to_numpy()
                                  & lookup.reindex(next_keys).success.isin([-1, 1]).to_numpy())
    data["abort_from_LED_onset"] = data.timed_fix - data.LED_onset_time
    if INCLUDE_FIRST_ROWS:
        # Maximal event-3 runs are detected before LED splitting. A complete run
        # needs observed adjacent non-event-3 trials on both sides.
        run_number = (is_abort & ~data.previous_is_abort).cumsum()
        aborts = data.loc[is_abort].copy()
        aborts["run_number"] = run_number.loc[aborts.index]
        runs = aborts.groupby("run_number", sort=False)
        aborts["run_position"] = runs.cumcount() + 1
        aborts["observed_run_length"] = runs.trial.transform("size")
        aborts["run_start_trial"] = runs.trial.transform("min")
        aborts["run_end_trial"] = runs.trial.transform("max")
        starts = aborts.loc[aborts.run_position.eq(1)].set_index("run_number")
        ends = aborts.loc[aborts.run_position.eq(aborts.observed_run_length)].set_index("run_number")
        complete = (starts.previous_trial.eq(starts.trial - 1) & starts.previous_abort_event.ne(3)
                    & ends.next_trial.eq(ends.trial + 1) & ends.next_abort_event.ne(3))
        aborts["complete_run"] = aborts.run_number.map(complete)
        aborts["run_id"] = ("s" + str(session_type) + "_a" + aborts.animal.astype(str)
                            + "_session" + aborts.session.astype(str)
                            + "_trial" + aborts.run_start_trial.astype(str))
        aborts["run_boundary_previous_trial"] = aborts.run_number.map(starts.previous_trial)
        aborts["run_boundary_next_trial"] = aborts.run_number.map(ends.next_trial)
        aborts["first_abort"] = aborts.run_position.eq(1) & aborts.observed_run_length.ge(2) & aborts.complete_run
        assert (aborts.run_end_trial - aborts.run_start_trial + 1).eq(aborts.observed_run_length).all()
        assert aborts.groupby("run_number").trial.diff().dropna().eq(1).all()
        data["first_abort"] = False
        data.loc[aborts.index, "first_abort"] = aborts.first_abort
        for column in ("run_position", "observed_run_length", "run_start_trial", "run_end_trial", "run_id",
                       "complete_run", "run_boundary_previous_trial", "run_boundary_next_trial"):
            data[column] = aborts[column].reindex(data.index)
        first = aborts.loc[aborts.first_abort]
        assert first.run_id.is_unique and len(first) == {7: 2381, 8: 2593}[session_type]
        assert first.previous_trial.eq(first.trial - 1).all() and first.previous_abort_event.ne(3).all()
        assert first.next_trial.eq(first.trial + 1).all() and first.next_abort_event.eq(3).all()
    streams[session_type] = data

saved_s8 = pd.read_csv(f"{EXISTING_MATCH_STEM}_eligible_trial_selection_manifest.csv", float_precision="round_trip")
assert saved_s8.source_csv_row_index.is_unique
saved_s8 = saved_s8.set_index("source_csv_row_index", drop=False)
saved_s8_curves = pd.read_csv(f"{EXISTING_MATCH_STEM}_curve_audit.csv", float_precision="round_trip")
if INCLUDE_FIRST_ROWS:
    saved_sequences = pd.read_csv(SEQUENCE_MANIFEST_PATH, float_precision="round_trip")
    saved_summary_curves = pd.read_csv(f"{BASE_OUTPUT_STEM}_curve_audit.csv", float_precision="round_trip")
case_frames, original_curves_by_case = {}, {}
optimizer_records, quota_records, retention_records = [], [], []


# %%
############ Verify saved selections; match each animal with frozen classifications ############
for case in CASES:
    session_type, selection = case
    case_id = f"s{session_type}_{selection}"
    data = streams[session_type]
    is_after = selection == "after_abort"
    denominator_mask = data.event_or_valid & data.previous_is_abort if is_after else data.event_or_valid
    if selection == "first":
        selected_mask = data.first_abort
    else:
        selected_mask = data.previous_is_abort & data.abort_event.eq(3) if is_after else data.isolated_abort
    eligible = data.loc[denominator_mask].copy()
    eligible["selected_abort"] = selected_mask.loc[eligible.index]
    if selection == "first":
        old_selected = saved_sequences.loc[saved_sequences.session_type.eq(session_type)
                                           & saved_sequences.selection.eq("first")].copy()
    else:
        old_selected = pd.read_csv(f"{original_stems[case]}_selected_trials.csv", float_precision="round_trip")
    assert not old_selected.duplicated(keys).any() and old_selected.source_csv_row_index.is_unique
    old_selected = old_selected.set_index("source_csv_row_index").sort_index()
    selected = eligible.loc[eligible.selected_abort].sort_index()
    raw_columns = raw_columns_by_session[session_type]
    compare_columns = [column for column in raw_columns if column in old_selected]
    pd.testing.assert_frame_equal(selected[compare_columns], old_selected[compare_columns], check_dtype=False)
    for column in ("previous_trial", "next_trial", "previous_success", "next_success",
                   "previous_source_csv_row_index", "next_source_csv_row_index"):
        if column in old_selected:
            np.testing.assert_array_equal(selected[column], old_selected[column])
    if selection == "first":
        for column in ("run_id", "run_position", "observed_run_length", "run_start_trial", "run_end_trial", "complete_run"):
            np.testing.assert_array_equal(selected[column], old_selected[column])
    assert np.isfinite(selected[["intended_fix", "LED_onset_time"]]).all().all()
    finite = np.isfinite(selected.abort_from_LED_onset)
    assert selected.loc[finite, "timed_fix"].lt(selected.loc[finite, "intended_fix"]).all()
    for led in (0, 1):
        condition = eligible.loc[eligible.LED_trial.eq(led)]
        n_selected = int(condition.selected_abort.sum())
        n_finite = int((condition.selected_abort & np.isfinite(condition.abort_from_LED_onset)).sum())
        assert (n_selected, n_finite, len(condition)) == EXPECTED_ORIGINAL[case][led]
    eligible["pre_onset_candidate"] = (eligible.selected_abort & np.isfinite(eligible.abort_from_LED_onset)
                                       & eligible.abort_from_LED_onset.lt(0))
    pre = eligible.pre_onset_candidate
    assert eligible.loc[pre, "abort_from_LED_onset"].ge(-2).all()
    eligible["pre_onset_bin"] = -1
    eligible.loc[pre, "pre_onset_bin"] = np.searchsorted(pre_bins, eligible.loc[pre, "abort_from_LED_onset"], side="right") - 1
    assert eligible.loc[pre, "pre_onset_bin"].between(0, n_pre_bins - 1).all()
    eligible["retained"] = True
    reuse = case == (8, "isolated") or (INCLUDE_FIRST_ROWS and case in ORIGINAL_CASES)
    if reuse:
        # Preserve every saved retained/removed identity, rather than resampling it.
        saved_selection = saved_s8 if case == (8, "isolated") else pd.read_csv(
            f"{BASE_OUTPUT_STEM}_{case_id}_selection_manifest.csv", float_precision="round_trip"
        ).set_index("source_csv_row_index", drop=False)
        pd.testing.assert_frame_equal(eligible[keys].sort_index(), saved_selection[keys].sort_index(), check_dtype=False)
        for column in ("LED_trial", "abort_event", "success", "intended_fix", "LED_onset_time",
                       "timed_fix", "pre_onset_candidate", "pre_onset_bin", "isolated_abort"):
            np.testing.assert_array_equal(eligible.sort_index()[column], saved_selection.sort_index()[column])
        eligible["retained"] = saved_selection.retained.reindex(eligible.index)
        assert eligible.retained.notna().all()
    rng = np.random.default_rng(SEED)
    for animal in ANIMALS:
        group = eligible.loc[eligible.animal.eq(animal)]
        denominators = group.groupby("LED_trial").size().reindex([0, 1]).to_numpy(dtype=int)
        available = np.stack([np.histogram(group.loc[group.LED_trial.eq(led) & group.pre_onset_candidate,
                                                    "abort_from_LED_onset"], bins=pre_bins)[0] for led in (0, 1)])
        saved_quotas = None
        if reuse:
            saved_quotas = np.stack([np.histogram(group.loc[group.LED_trial.eq(led) & group.pre_onset_candidate
                                                           & group.retained, "abort_from_LED_onset"], bins=pre_bins)[0]
                                    for led in (0, 1)])
        quotas, diagnostic = optimize_pre_onset_counts(available, denominators, saved_quotas)
        for led in (0, 1):
            for bin_index in range(n_pre_bins):
                pool = group.index[group.LED_trial.eq(led) & group.pre_onset_candidate
                                   & group.pre_onset_bin.eq(bin_index)].to_numpy()
                pool.sort()
                assert len(pool) == available[led, bin_index]
                if reuse:
                    chosen = pool[eligible.loc[pool, "retained"].to_numpy()]
                else:
                    chosen = rng.choice(pool, size=quotas[led, bin_index], replace=False)
                    eligible.loc[pool, "retained"] = False
                    eligible.loc[chosen, "retained"] = True
                assert len(chosen) == quotas[led, bin_index] and len(np.unique(chosen)) == len(chosen)
                quota_records.append({"case": case_id, "session_type": session_type, "selection": selection,
                                      "animal": animal, "LED_trial": led, "bin_index": bin_index,
                                      "bin_left_s": pre_bins[bin_index], "bin_right_s": pre_bins[bin_index + 1],
                                      "available": len(pool), "retained": len(chosen), "removed": len(pool) - len(chosen)})
        optimizer_records.append({"case": case_id, "session_type": session_type, "selection": selection,
                                  "animal": animal, "reused_saved_identities": reuse, **diagnostic})
    kept, removed = eligible.loc[eligible.retained], eligible.loc[~eligible.retained]
    assert eligible.loc[~eligible.pre_onset_candidate, "retained"].all()
    assert removed.pre_onset_candidate.all() and removed.selected_abort.all() and removed.abort_event.eq(3).all()
    assert not removed.success.isin([-1, 1]).any()
    assert kept.index.is_unique and not kept.duplicated(keys).any()
    assert len(kept) + len(removed) == len(eligible)
    for animal in ANIMALS:
        for led in (0, 1):
            original = eligible.loc[eligible.animal.eq(animal) & eligible.LED_trial.eq(led)]
            retained = original.loc[original.retained]
            assert original.success.isin([-1, 1]).sum() == retained.success.isin([-1, 1]).sum()
            assert original.abort_from_LED_onset.ge(0).sum() == retained.abort_from_LED_onset.ge(0).sum()
            assert original.timed_fix.isna().sum() == retained.timed_fix.isna().sum()
    retained_off_on = kept.groupby("LED_trial").selected_abort.sum().reindex([0, 1]).to_numpy(int)
    if (MAX_BIN_RATE_DIFFERENCE, MAX_PRE_ONSET_FRACTION_DIFFERENCE, ABORT_BIN_WIDTH_S) == (0.005, 0.001, 0.020):
        assert (*retained_off_on, len(kept)) == EXPECTED_MATCHED[case]
    if reuse:
        np.testing.assert_array_equal(eligible.sort_index().retained, saved_selection.sort_index().retained)
    retention_records.append({
        "case": case_id, "session_type": session_type, "selection": selection,
        "original_selected": int(eligible.selected_abort.sum()), "retained_selected": int(kept.selected_abort.sum()),
        "retained_selected_OFF": retained_off_on[0], "retained_selected_ON": retained_off_on[1],
        "original_denominator": len(eligible), "retained_denominator": len(kept), "removed": len(removed),
        "missing_selected_time": int((eligible.selected_abort & ~np.isfinite(eligible.abort_from_LED_onset)).sum()),
        "valid_trials_preserved": int(kept.success.isin([-1, 1]).sum()),
        "post_onset_selected_preserved": int((kept.selected_abort & kept.abort_from_LED_onset.ge(0)).sum()),
        "reused_saved_identities": reuse,
    })
    case_frames[case] = eligible
    original_curves_by_case[case] = (pd.read_csv(f"{original_stems[case]}_curve_audit.csv", float_precision="round_trip")
                                   if case in ORIGINAL_CASES else None)
    print(f"{case_id}: certified optimum; retained {kept.selected_abort.sum():,}/{eligible.selected_abort.sum():,} "
          f"selected aborts; {len(kept):,}/{len(eligible):,} eligible trials. "
          f"{'Reused saved rows.' if reuse else 'Seed reset to 20261001.'}", flush=True)


# %%
############ Animal histograms first; equal-animal means and SEMs second ############
curves, sems = {}, {}
panel_records, curve_parts = [], []
for case in CASES:
    session_type, selection = case
    case_id = f"s{session_type}_{selection}"
    eligible, old_curves = case_frames[case], original_curves_by_case[case]
    old_panels = (pd.read_csv(f"{original_stems[case]}_count_area_audit.csv", float_precision="round_trip")
                  if case in ORIGINAL_CASES else None)
    for animal in ANIMALS:
        for led in (0, 1):
            group = eligible.loc[eligible.animal.eq(animal) & eligible.LED_trial.eq(led)]
            for metric in METRICS:
                matched = metric == METRICS[3]
                current = group.loc[group.retained] if matched else group
                selected = current.loc[current.selected_abort]
                value_column = "abort_from_LED_onset" if matched else metric
                values = selected[value_column].to_numpy()
                values = values[np.isfinite(values)]
                bins, width = bins_by_metric[metric], width_by_metric[metric]
                counts = np.histogram(values, bins=bins)[0]
                assert len(values) > 0 and counts.sum() == len(values)
                normalizer = len(current) if metric in METRICS[2:] else len(values)
                heights = counts / (normalizer * width)
                area = float(heights @ np.diff(bins))
                expected_area = len(values) / normalizer
                assert np.isclose(area, expected_area, rtol=0, atol=1e-12)
                if not matched and case in ORIGINAL_CASES:
                    reference = old_curves.loc[old_curves.row.eq(str(animal)) & old_curves.LED_trial.eq(led)
                                               & old_curves.metric.eq(metric)]
                    np.testing.assert_array_equal(heights, reference.height)
                    np.testing.assert_array_equal(counts, reference.bin_count)
                    reference_panel = old_panels.loc[old_panels.row.eq(str(animal)) & old_panels.LED_trial.eq(led)
                                                     & old_panels.metric.eq(metric)].iloc[0]
                    assert len(current) == reference_panel.eligible_denominator
                    assert len(selected) == reference_panel.selected_aborts
                elif case == (8, "isolated"):
                    reference = saved_s8_curves.loc[saved_s8_curves.row.eq(str(animal)) & saved_s8_curves.LED_trial.eq(led)
                                                    & saved_s8_curves.metric.eq(metric)]
                    np.testing.assert_array_equal(heights, reference.height)
                    np.testing.assert_array_equal(counts, reference.bin_count)
                curves[(case, str(animal), led, metric)] = heights
                sems[(case, str(animal), led, metric)] = np.zeros_like(heights)
                panel_records.append({
                    "case": case_id, "session_type": session_type, "selection": selection,
                    "row": str(animal), "LED_trial": led, "metric": metric, "n_animals": 1,
                    "original_selected_aborts": int(group.selected_abort.sum()), "selected_aborts": len(selected),
                    "finite_aligned_aborts": int(np.isfinite(selected.abort_from_LED_onset).sum()),
                    "missing_aligned_time": int((~np.isfinite(selected.abort_from_LED_onset)).sum()),
                    "plotted_observations": len(values), "original_denominator": len(group),
                    "eligible_denominator": len(current), "removed": len(group) - len(current),
                    "histogram_normalizer": normalizer, "full_histogram_area": area,
                    "expected_area": expected_area, "bin_width_s": width,
                    "denominator_definition": "event-3 or valid current trials after an adjacent event-3 abort"
                    if selection == "after_abort" else "all event-3 or valid trials",
                })
                curve_parts.append(pd.DataFrame({
                    "case": case_id, "row": str(animal), "LED_trial": led, "metric": metric,
                    "bin_left_s": bins[:-1], "bin_right_s": bins[1:],
                    "height": heights, "sem": np.nan, "bin_count": counts,
                }))
    # Neither summed counts nor summed denominators define the plotted average.
    animal_panels = pd.DataFrame(panel_records)
    animal_panels = animal_panels.loc[animal_panels.case.eq(case_id) & ~animal_panels.row.eq("equal_animal_mean")]
    for led in (0, 1):
        for metric in METRICS:
            bins = bins_by_metric[metric]
            matrix = np.stack([curves[(case, str(animal), led, metric)] for animal in ANIMALS])
            mean = matrix.mean(axis=0)
            sem = matrix.std(axis=0, ddof=1) / np.sqrt(len(ANIMALS))
            np.testing.assert_allclose(sem, np.sqrt(((matrix - mean) ** 2).sum(axis=0) / 30), atol=1e-14)
            curves[(case, "equal_animal_mean", led, metric)] = mean
            sems[(case, "equal_animal_mean", led, metric)] = sem
            if case in ORIGINAL_CASES and (metric != METRICS[3] or case == (8, "isolated")):
                reference_curves = saved_s8_curves if metric == METRICS[3] else old_curves
                reference = reference_curves.loc[reference_curves.row.eq("equal_animal_mean")
                                                & reference_curves.LED_trial.eq(led) & reference_curves.metric.eq(metric)]
                np.testing.assert_array_equal(mean, reference.height)
                np.testing.assert_array_equal(sem, reference["sem"])
            audits = animal_panels.loc[animal_panels.LED_trial.eq(led) & animal_panels.metric.eq(metric)]
            area = float(mean @ np.diff(bins))
            assert np.isclose(area, audits.expected_area.mean(), rtol=0, atol=1e-12)
            record = {"case": case_id, "session_type": session_type, "selection": selection,
                      "row": "equal_animal_mean", "LED_trial": led, "metric": metric, "n_animals": 6,
                      "histogram_normalizer": np.nan, "full_histogram_area": area,
                      "expected_area": audits.expected_area.mean(), "bin_width_s": width_by_metric[metric],
                      "denominator_definition": audits.iloc[0].denominator_definition}
            for field in ("original_selected_aborts", "selected_aborts", "finite_aligned_aborts", "missing_aligned_time",
                          "plotted_observations", "original_denominator", "eligible_denominator", "removed"):
                record[field] = int(audits[field].sum())
            panel_records.append(record)
            curve_parts.append(pd.DataFrame({
                "case": case_id, "row": "equal_animal_mean", "LED_trial": led, "metric": metric,
                "bin_left_s": bins[:-1], "bin_right_s": bins[1:],
                "height": mean, "sem": sem, "bin_count": np.nan,
            }))
panels, curve_audit = pd.DataFrame(panel_records), pd.concat(curve_parts, ignore_index=True)
for case in CASES:
    case_id = f"s{case[0]}_{case[1]}"
    for row in [*[str(animal) for animal in ANIMALS], "equal_animal_mean"]:
        for metric in METRICS[2:]:
            off, on = (curves[(case, row, led, metric)][:n_pre_bins] for led in (0, 1))
            max_difference = float(np.abs(off - on).max())
            fraction_difference = float(abs((off - on).sum() * ABORT_BIN_WIDTH_S))
            mask = panels.case.eq(case_id) & panels.row.eq(row) & panels.metric.eq(metric)
            panels.loc[mask, "max_pre_bin_rate_difference_s_inv"] = max_difference
            panels.loc[mask, "pre_onset_fraction_difference"] = fraction_difference
            if metric == METRICS[3]:
                assert max_difference <= MAX_BIN_RATE_DIFFERENCE + 1e-12
                assert fraction_difference <= MAX_PRE_ONSET_FRACTION_DIFFERENCE + 1e-12
assert len(panels) == len(CASES) * 7 * 2 * 4
assert not curve_audit.duplicated(["case", "row", "LED_trial", "metric", "bin_left_s"]).any()
assert curve_audit.height.ge(0).all() and np.isfinite(curve_audit.height).all()
if INCLUDE_FIRST_ROWS:
    # Every bin, including matched curves and SEMs, must reproduce the old four rows.
    comparison_keys = ["case", "row", "LED_trial", "metric", "bin_left_s"]
    old_cases = [f"s{st}_{selection}" for st, selection in ORIGINAL_CASES]
    current = curve_audit.loc[curve_audit.case.isin(old_cases)].set_index(comparison_keys).sort_index()
    reference = saved_summary_curves.set_index(comparison_keys).sort_index()
    pd.testing.assert_frame_equal(current, reference, check_exact=True)


# %%
############ Save source-row selections, exact curves, and optimization audits ############
manifest_columns = ["source_csv_row_index", *keys, "session_type", "training_level", "repeat_trial", "LED_trial",
                    "abort_event", "success", "intended_fix", "LED_onset_time", "timed_fix", "abort_from_LED_onset",
                    "selected_abort", "previous_is_abort", "isolated_abort", "previous_trial", "next_trial",
                    "previous_success", "next_success", "previous_abort_event", "next_abort_event",
                    "previous_LED_trial", "next_LED_trial", "previous_source_csv_row_index", "next_source_csv_row_index",
                    "pre_onset_candidate", "pre_onset_bin", "retained"]
if INCLUDE_FIRST_ROWS:
    manifest_columns.extend(["first_abort", "run_id", "run_position", "observed_run_length", "run_start_trial",
                             "run_end_trial", "complete_run", "run_boundary_previous_trial", "run_boundary_next_trial"])
for case in CASES:
    case_id = f"s{case[0]}_{case[1]}"
    manifest = case_frames[case][manifest_columns].copy()
    manifest.insert(0, "source_csv", f"raw_data/LED7_s{case[0]}.csv")
    manifest.insert(0, "case", case_id)
    manifest.to_csv(f"{OUTPUT_STEM}_{case_id}_selection_manifest.csv", index=False)
optimizer, quota_audit, retention = (pd.DataFrame(records) for records in (optimizer_records, quota_records, retention_records))
optimizer.to_csv(f"{OUTPUT_STEM}_optimizer_diagnostics.csv", index=False)
quota_audit.to_csv(f"{OUTPUT_STEM}_bin_quotas.csv", index=False)
retention.to_csv(f"{OUTPUT_STEM}_retention_audit.csv", index=False)
panels.to_csv(f"{OUTPUT_STEM}_count_area_audit.csv", index=False)
curve_audit.to_csv(f"{OUTPUT_STEM}_curve_audit.csv", index=False)
panels.loc[panels.row.eq("equal_animal_mean")].to_csv(f"{OUTPUT_STEM}_mean_area_audit.csv", index=False)
array_rows = [*[str(animal) for animal in ANIMALS], "equal_animal_mean"]
np.savez_compressed(
    f"{OUTPUT_STEM}_curves.npz", cases=np.array([f"s{st}_{selection}" for st, selection in CASES]),
    rows=np.array(array_rows), led_conditions=np.array([0, 1]), timing_bins=timing_bins, abort_bins=abort_bins,
    **{metric: np.stack([[[curves[(case, row, led, metric)] for led in (0, 1)] for row in array_rows] for case in CASES])
       for metric in METRICS},
    **{f"{metric}_sem": np.stack([[[sems[(case, row, led, metric)] for led in (0, 1)] for row in array_rows] for case in CASES])
       for metric in METRICS},
)
summary = {
    "source_hashes_sha256": input_hashes, "seed_per_new_case": SEED,
    "numpy_version": np.__version__, "scipy_version": scipy.__version__, "animals": ANIMALS,
    "case_order": [f"s{st}_{selection}" for st, selection in CASES], "retention": retention.to_dict(orient="records"),
    "max_bin_rate_difference_s_inv": MAX_BIN_RATE_DIFFERENCE,
    "max_pre_onset_fraction_difference": MAX_PRE_ONSET_FRACTION_DIFFERENCE,
    "all_solvers_certified_optimal": bool(optimizer.certified_optimal.all()),
    "optimality_scope": "Maximum retention under the specified conservative linear rule, independently per animal/case.",
    "s8_isolated_selection": "Exact saved retained/removed source-row identities reused; not resampled.",
    "existing_four_rows": "Original and matched animal curves, equal-animal means and SEMs reproduced exactly.",
    "first_abort_definition": "First trial in a complete run of at least two event-3 aborts; adjacent non-event-3 boundaries required.",
    "first_abort_denominator": "All event-3-or-valid trials for the current animal, session type, and LED condition.",
    "classification": "Frozen on original trial streams, using actual adjacent trial IDs.",
    "averaging": "Equal mean of six animal-normalized curves; across-animal sample SD / sqrt(6) SEM.",
    "missing_time": "Kept in denominators and timing panels, omitted from aligned time histograms.",
    "caveat": "Observed-outcome matching enforces pre-onset overlap; it is descriptive, not causal evidence.",
}
Path(f"{OUTPUT_STEM}_run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


# %%
############ Equal-animal summary: optional first-in-run rows at the top ############
fig, axes = plt.subplots(len(CASES), 4, figsize=(20, 3 * len(CASES) + 1), sharex="col", sharey=False)
for row_index, case in enumerate(CASES):
    session_type, selection = case
    case_id = f"s{session_type}_{selection}"
    selected_label = {"first": "First abort in a run", "after_abort": "Abort → abort",
                      "isolated": "Valid → abort → valid"}[selection]
    kept_counts = retention.loc[retention.case.eq(case_id)].iloc[0]
    for column, metric in enumerate(METRICS):
        ax, bins = axes[row_index, column], bins_by_metric[metric]
        for led in (0, 1):
            mean = curves[(case, "equal_animal_mean", led, metric)]
            sem = sems[(case, "equal_animal_mean", led, metric)]
            ax.fill_between(bins, np.r_[mean - sem, (mean - sem)[-1]], np.r_[mean + sem, (mean + sem)[-1]],
                            step="post", color=COLORS[led], alpha=0.16, linewidth=0)
            ax.stairs(mean, bins, color=COLORS[led], linewidth=1.6)
            audit = panels.loc[panels.case.eq(case_id) & panels.row.eq("equal_animal_mean")
                               & panels.LED_trial.eq(led) & panels.metric.eq(metric)].iloc[0]
            label = "OFF" if led == 0 else "ON"
            if column < 2:
                note = f"{label} n = {int(audit.selected_aborts):,}"
            else:
                note = f"{label} n={int(audit.finite_aligned_aborts):,}, N={int(audit.eligible_denominator):,}; area={audit.full_histogram_area:.3f}"
            ax.text(0.98, 0.97 - 0.09 * led, note, transform=ax.transAxes, ha="right", va="top",
                    color=COLORS[led], fontsize=8.5)
        if column >= 2:
            ax.axvline(0, color="0.45", linestyle=":", linewidth=1)
            ax.set_xlim(*ABORT_DISPLAY_S)
            if column == 3:
                note = (f"Selected retained: {int(kept_counts.retained_selected):,}/{int(kept_counts.original_selected):,}\n"
                        f"Pre: max |Δrate| = {audit.max_pre_bin_rate_difference_s_inv:.4f} s⁻¹\n"
                        f"|Δfraction| = {100 * audit.pre_onset_fraction_difference:.3f} pp")
                ax.text(0.02, 0.76, note, transform=ax.transAxes, ha="left", va="top", fontsize=8)
        else:
            ax.set_xlim(0, 2.2)
        if column == 0:
            ax.set_ylabel(f"Session type {session_type}\n{selected_label}\nDensity (s⁻¹)", fontsize=11)
        elif column == 2:
            ax.set_ylabel({"first": "First-abort rate (s⁻¹)", "after_abort": "Conditional abort rate (s⁻¹)",
                           "isolated": "Isolated-abort rate (s⁻¹)"}[selection], fontsize=10)
        if row_index == 0:
            ax.set_title(COLUMN_TITLES[column], fontsize=12, pad=12)
        if row_index == len(CASES) - 1:
            ax.set_xlabel("Time from scheduled LED onset (s)" if column >= 2 else "Time from fixation onset (s)", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.grid(axis="y", color="0.91", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
upper_limits = []
for column, metric in enumerate(METRICS):
    bins = bins_by_metric[metric]
    visible = ((bins[:-1] < ABORT_DISPLAY_S[1]) & (bins[1:] > ABORT_DISPLAY_S[0])) if column >= 2 else np.ones(len(bins) - 1, dtype=bool)
    upper = max(float((curves[(case, "equal_animal_mean", led, metric)]
                       + sems[(case, "equal_animal_mean", led, metric)])[visible].max())
                for case in CASES for led in (0, 1))
    upper_limits.append(upper * 1.35)
upper_limits[2] = upper_limits[3] = max(upper_limits[2:])
for column, upper in enumerate(upper_limits):
    for row_index, case in enumerate(CASES):
        limits = FIRST_ABORT_RATE_YLIM if case[1] == "first" and column >= 2 else (0, upper)
        axes[row_index, column].set_ylim(*limits)
fig.suptitle("LED7 sequence-abort timing: equal-animal mean ± SEM", fontsize=19, y=0.995)
fig.legend(handles=[Line2D([0], [0], color=COLORS[led], lw=2, label=f"Current trial LED {'OFF' if led == 0 else 'ON'}")
                    for led in (0, 1)], loc="upper center", bbox_to_anchor=(0.5, 0.974), ncol=2, frameon=False, fontsize=12)
fig.text(0.5, 0.018,
         "Animals 92, 93, 98, 99, 100, 103. Timing: 40 ms unit-area histograms; abort rates: 20 ms bins, full areas over −2 to +2 s.\n"
         "First-in-run: one first abort per complete run of ≥2. First/isolated denominators: all event-3 or valid trials; abort→abort: eligible trials following an abort. "
         "Each curve averages six animal-normalized curves, not pooled trials.\n"
         "n/N annotations are summed plotted-abort/eligible counts; areas are means of animal fractions. Retained selected counts include two missing-time aborts in each abort→abort row.\n"
         "Only pre-onset selected aborts removed; other eligible rows retained. OFF onset is scheduled/counterfactual. "
         "Pre-onset overlap is enforced using outcomes: descriptive, not causal evidence.",
         ha="center", fontsize=9, linespacing=1.45)
fig.tight_layout(rect=(0.005, 0.105, 0.995, 0.945), h_pad=1.5, w_pad=1.6)
figure_path = Path(f"{OUTPUT_STEM}_{len(CASES)}x4.png")
fig.savefig(figure_path, dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)

for path in protected_paths:
    with path.open("rb") as handle:
        assert hashlib.file_digest(handle, "sha256").hexdigest() == input_hashes[str(path.relative_to(REPO))]
print(retention.to_string(index=False))
print(f"All {len(CASES) * len(ANIMALS)} optimizations certified; existing curve/SEM and membership checks passed.")
print("Trial adjacency, protected rows, both matching tolerances, histogram areas, and equal-animal SEM checks passed.")
print(f"Saved {figure_path}")
