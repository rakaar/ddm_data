# %%
"""First and second members of adjacent AA pairs with conditional denominators.

A is an event-3 abort and V is a valid trial (success +/-1). The new first
member rows use AA/(AA+VA); second-member rows use AA/(AA+AV); isolated rows
use VAV/(VAV+VVV). Pair labels stay fixed during the onset-matching selection.
"""

from pathlib import Path
import os


# %%
############ Editable parameters ############
SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
PAIR_STEM = SCRIPT_DIR / "led7_s7_s8_pair_conditioned_sequence_abort_timing"
SOURCE_STEM = SCRIPT_DIR / "led7_s7_s8_sequence_abort_valid_neighbor_conditioning"
FOUR_STEM = SCRIPT_DIR / "led7_s7_s8_sequence_abort_conditional_denominators"
OUTPUT_STEM = SCRIPT_DIR / "led7_s7_s8_sequence_abort_conditional_denominators_with_first"
ANIMALS = (92, 93, 98, 99, 100, 103)
SESSION_TYPES = (7, 8)
CASES = ("s7_first_member", "s8_first_member", "s7_after_abort", "s8_after_abort",
         "s7_isolated", "s8_isolated")
METRICS = ("intended_fix", "LED_onset_time", "abort_from_LED_onset",
           "matched_abort_from_LED_onset")
SEED = 20261001  # Reset for each session type.
MAX_BIN_RATE_DIFFERENCE = 0.005  # s^-1 within each 20 ms pre-onset bin.
MAX_PRE_FRACTION_DIFFERENCE = 0.001
SOLVER_TIME_LIMIT_S = 60
RATE_DISPLAY_S = (-0.4, 0.5)
FIGURE_DPI = 240
COLORS = {0: "tab:blue", 1: "tab:red"}
EXPECTED_ORIGINAL = {
    7: {0: (1736, 10485), 1: (1361, 5240)},
    8: {0: (1490, 10443), 1: (1936, 6487)},
}
EXPECTED_MATCHED = {
    7: {0: (1157, 9906), 1: (911, 4790)},
    8: {0: (946, 9899), 1: (1396, 5947)},
}

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, eye, hstack, vstack


# %%
############ Load exact saved curves for all original panels ############
with np.load(f"{PAIR_STEM}_curves.npz") as payload:
    pair_curves = {key: payload[key].copy() for key in payload.files}
with np.load(f"{SOURCE_STEM}_curves.npz") as payload:
    source_curves = {key: payload[key].copy() for key in payload.files}
pair_cases = pair_curves["cases"].tolist()
source_cases = source_curves["cases"].tolist()
assert pair_cases == ["s7_first_member", "s8_first_member", "s7_second_member",
                      "s8_second_member", "s7_isolated", "s8_isolated"]
assert source_cases == ["s7_first", "s8_first", "s7_after_abort", "s8_after_abort",
                        "s7_isolated", "s8_isolated"]
rows = [*[str(animal) for animal in ANIMALS], "equal_animal_mean"]
assert pair_curves["rows"].tolist() == source_curves["rows"].tolist() == rows
timing_bins, abort_bins = pair_curves["timing_bins"], pair_curves["abort_bins"]
np.testing.assert_array_equal(timing_bins, source_curves["timing_bins"])
np.testing.assert_array_equal(abort_bins, source_curves["abort_bins"])
np.testing.assert_allclose(np.diff(timing_bins), 0.040, rtol=0, atol=1e-14)
np.testing.assert_allclose(np.diff(abort_bins), 0.020, rtol=0, atol=1e-14)
pre_bins = abort_bins[:101]
np.testing.assert_allclose([pre_bins[0], pre_bins[-1]], [-2, 0], rtol=0, atol=1e-14)

curves = {"cases": np.asarray(CASES), "rows": np.asarray(rows),
          "timing_bins": timing_bins.copy(), "abort_bins": abort_bins.copy()}
for metric in METRICS:
    n_bins = len(timing_bins) - 1 if metric in METRICS[:2] else len(abort_bins) - 1
    curves[metric] = np.full((6, 7, 2, n_bins), np.nan)
    curves[f"{metric}_sem"] = np.full((6, 7, 2, n_bins), np.nan)
    if metric in METRICS[:3]:
        for new_row, session_type in enumerate(SESSION_TYPES):
            old_row = pair_cases.index(f"s{session_type}_first_member")
            curves[metric][new_row] = pair_curves[metric][old_row]
            curves[f"{metric}_sem"][new_row] = pair_curves[f"{metric}_sem"][old_row]
    for new_row, case in enumerate(CASES[2:], start=2):
        old_row = source_cases.index(case)
        curves[metric][new_row] = source_curves[metric][old_row]
        curves[f"{metric}_sem"][new_row] = source_curves[f"{metric}_sem"][old_row]
        np.testing.assert_array_equal(curves[metric][new_row], source_curves[metric][old_row])

old_audit = pd.read_csv(f"{FOUR_STEM}_count_area_audit.csv", dtype={"row": str})
pair_audit = pd.read_csv(f"{PAIR_STEM}_count_area_audit.csv", dtype={"row": str})
assert set(old_audit.case) == set(CASES[2:])
first_audit = pair_audit.loc[pair_audit.case.isin(CASES[:2])
                             & pair_audit.metric.isin(METRICS[:3])].copy()
first_audit["selection"] = "first_member"
first_audit["original_selected_aborts"] = first_audit.selected_aborts
first_audit["original_denominator"] = first_audit.eligible_denominator
first_audit["removed"] = 0
first_audit["denominator_definition"] = "current A or V immediately before adjacent A (AA+VA)"
first_audit["max_pre_bin_rate_difference_s_inv"] = np.nan
first_audit["pre_onset_fraction_difference"] = np.nan
assert len(first_audit) == 2 * 7 * 2 * 3


# %%
############ Reusable integer-quota optimization for the new rows ############
def optimize_pre_onset_counts(available, denominators):
    """Maximize retained aborts under the earlier conservative linear rule."""
    mandatory = denominators - available.sum(axis=1)
    assert (mandatory > 0).all()
    ratio = mandatory[0] / mandatory[1]
    u_max = float((available / mandatory[:, None]).max())
    epsilon = 0.020 * MAX_BIN_RATE_DIFFERENCE - u_max * MAX_PRE_FRACTION_DIFFERENCE
    assert epsilon > 0
    n_bins = available.shape[1]
    matrix = vstack([
        hstack([eye(n_bins), -ratio * eye(n_bins)]),
        csr_matrix(np.r_[np.ones(n_bins), -ratio * np.ones(n_bins)][None, :]),
    ], format="csr")
    tolerance = np.r_[np.full(n_bins, mandatory[0] * epsilon),
                      mandatory[0] * MAX_PRE_FRACTION_DIFFERENCE]
    result = milp(
        -np.ones(2 * n_bins), integrality=np.ones(2 * n_bins),
        bounds=Bounds(np.zeros(2 * n_bins), available.ravel()),
        constraints=LinearConstraint(matrix, -tolerance, tolerance),
        options={"mip_rel_gap": 0.0, "time_limit": SOLVER_TIME_LIMIT_S},
    )
    assert result.success and result.status == 0 and result.mip_gap == 0, result.message
    quotas = np.rint(result.x).astype(int).reshape(2, n_bins)
    np.testing.assert_allclose(result.x, quotas.ravel(), rtol=0, atol=1e-7)
    np.testing.assert_allclose([result.fun, result.mip_dual_bound],
                               [-quotas.sum(), -quotas.sum()], rtol=0, atol=1e-7)
    assert ((quotas >= 0) & (quotas <= available)).all()
    retained_n = mandatory + quotas.sum(axis=1)
    rates = quotas / (retained_n[:, None] * 0.020)
    max_rate_difference = float(abs(rates[0] - rates[1]).max())
    fraction_difference = float(abs(quotas[0].sum() / retained_n[0]
                                    - quotas[1].sum() / retained_n[1]))
    assert max_rate_difference <= MAX_BIN_RATE_DIFFERENCE + 1e-12
    assert fraction_difference <= MAX_PRE_FRACTION_DIFFERENCE + 1e-12
    return quotas, {
        "certified_optimal": True, "solver_status": result.status, "mip_gap": result.mip_gap,
        "objective": result.fun, "dual_bound": result.mip_dual_bound,
        "original_OFF": int(denominators[0]), "original_ON": int(denominators[1]),
        "retained_OFF": int(retained_n[0]), "retained_ON": int(retained_n[1]),
        "available_pre_OFF": int(available[0].sum()), "available_pre_ON": int(available[1].sum()),
        "retained_pre_OFF": int(quotas[0].sum()), "retained_pre_ON": int(quotas[1].sum()),
        "max_pre_bin_rate_difference_s_inv": max_rate_difference,
        "pre_onset_fraction_difference": fraction_difference,
    }


# %%
############ Verify pair IDs, then match the first-member rows ############
manifests, matching_records, quota_records, matched_audits = [], [], [], []
keys = ["animal", "session", "trial"]
for session_type in SESSION_TYPES:
    source = pd.read_csv(REPO / f"raw_data/LED7_s{session_type}.csv", float_precision="round_trip")
    assert source.shape == ({7: 138440, 8: 148428}[session_type], 52)
    assert source.training_level.eq(16).all() and source.session_type.eq(session_type).all()
    assert (source.repeat_trial.isin([0, 2]) | source.repeat_trial.isna()).all()
    assert source.LED_trial.isin([0, 1]).all()
    source["source_csv_row_index"] = np.arange(len(source))
    data = source.loc[source.animal.isin(ANIMALS)].sort_values(keys, kind="stable").copy()
    assert not data.duplicated(keys).any()
    data = data.set_index("source_csv_row_index", drop=False)
    following = data.groupby(["animal", "session"], sort=False)[
        ["trial", "abort_event", "source_csv_row_index"]
    ].shift(-1)
    abort = data.abort_event.eq(3)
    valid = data.success.isin([-1, 1])
    assert not (abort & valid).any()
    first_condition = following.trial.eq(data.trial + 1) & following.abort_event.eq(3)
    eligible = data.loc[(abort | valid) & first_condition].copy().sort_index()
    eligible["selected_abort"] = eligible.abort_event.eq(3)
    eligible["abort_from_LED_onset"] = eligible.timed_fix - eligible.LED_onset_time
    eligible["next_trial"] = following.loc[eligible.index, "trial"].to_numpy()
    eligible["next_source_csv_row_index"] = following.loc[
        eligible.index, "source_csv_row_index"
    ].to_numpy()
    assert eligible.next_trial.eq(eligible.trial + 1).all()
    assert np.isfinite(eligible.loc[eligible.selected_abort,
                                   ["intended_fix", "LED_onset_time", "timed_fix"]]).all().all()
    assert eligible.loc[eligible.selected_abort, "timed_fix"].lt(
        eligible.loc[eligible.selected_abort, "intended_fix"]
    ).all()

    saved = pd.read_csv(
        f"{PAIR_STEM}_s{session_type}_first_member_eligible_trials.csv",
        float_precision="round_trip",
    ).set_index("source_csv_row_index").sort_index()
    np.testing.assert_array_equal(eligible.index.to_numpy(), saved.index.to_numpy())
    for field in (*keys, "LED_trial", "abort_event", "success", "intended_fix",
                  "LED_onset_time", "timed_fix", "next_trial"):
        np.testing.assert_allclose(eligible[field], saved[field], rtol=0, atol=0,
                                   equal_nan=True)
    np.testing.assert_array_equal(eligible.selected_abort, saved.selected_abort)

    case = f"s{session_type}_first_member"
    case_index = CASES.index(case)
    for led in (0, 1):
        group = eligible.loc[eligible.LED_trial.eq(led)]
        assert (int(group.selected_abort.sum()), len(group)) == EXPECTED_ORIGINAL[session_type][led]
        for animal_index, animal in enumerate(ANIMALS):
            animal_group = group.loc[group.animal.eq(animal)]
            selected = animal_group.loc[animal_group.selected_abort]
            for metric in METRICS[:3]:
                bins = timing_bins if metric in METRICS[:2] else abort_bins
                values = selected[metric].to_numpy()
                counts = np.histogram(values, bins=bins)[0]
                assert int(counts.sum()) == len(values)
                width = 0.040 if metric in METRICS[:2] else 0.020
                normalizer = len(values) if metric in METRICS[:2] else len(animal_group)
                calculated = counts / (normalizer * width)
                np.testing.assert_allclose(calculated, curves[metric][case_index, animal_index, led],
                                           rtol=0, atol=1e-14)

    eligible["pre_onset_candidate"] = (eligible.selected_abort
                                       & eligible.abort_from_LED_onset.lt(0))
    assert eligible.loc[eligible.pre_onset_candidate, "abort_from_LED_onset"].ge(-2).all()
    eligible["pre_onset_bin"] = -1
    pre = eligible.pre_onset_candidate
    eligible.loc[pre, "pre_onset_bin"] = np.searchsorted(
        pre_bins, eligible.loc[pre, "abort_from_LED_onset"], side="right"
    ) - 1
    assert eligible.loc[pre, "pre_onset_bin"].between(0, 99).all()
    eligible["retained"] = True
    rng = np.random.default_rng(SEED)
    for animal_index, animal in enumerate(ANIMALS):
        animal_trials = eligible.loc[eligible.animal.eq(animal)]
        denominators = np.array([int(animal_trials.LED_trial.eq(led).sum()) for led in (0, 1)])
        available = np.stack([
            np.histogram(animal_trials.loc[animal_trials.LED_trial.eq(led)
                                                 & animal_trials.pre_onset_candidate,
                                           "abort_from_LED_onset"], pre_bins)[0]
            for led in (0, 1)
        ])
        quotas, matching = optimize_pre_onset_counts(available, denominators)
        matching_records.append({"case": case, "session_type": session_type,
                                 "animal": animal, **matching})
        for led in (0, 1):
            for bin_index in range(100):
                pool = animal_trials.index[animal_trials.LED_trial.eq(led)
                                           & animal_trials.pre_onset_candidate
                                           & animal_trials.pre_onset_bin.eq(bin_index)].to_numpy()
                pool.sort()
                assert len(pool) == available[led, bin_index]
                chosen = rng.choice(pool, size=quotas[led, bin_index], replace=False)
                removed = np.setdiff1d(pool, chosen, assume_unique=True)
                eligible.loc[removed, "retained"] = False
                quota_records.append({
                    "case": case, "animal": animal, "LED_trial": led, "bin_index": bin_index,
                    "bin_left_s": pre_bins[bin_index], "bin_right_s": pre_bins[bin_index + 1],
                    "available": len(pool), "retained": int(quotas[led, bin_index]),
                    "removed": len(removed),
                })
        retained_animal = eligible.loc[eligible.animal.eq(animal) & eligible.retained]
        assert len(retained_animal) == matching["retained_OFF"] + matching["retained_ON"]
        assert int((retained_animal.selected_abort
                    & retained_animal.abort_from_LED_onset.ge(0)).sum()) == int(
            (animal_trials.selected_abort & animal_trials.abort_from_LED_onset.ge(0)).sum()
        )
        for led in (0, 1):
            original = animal_trials.loc[animal_trials.LED_trial.eq(led)]
            current = retained_animal.loc[retained_animal.LED_trial.eq(led)]
            selected = current.loc[current.selected_abort]
            values = selected.abort_from_LED_onset.to_numpy()
            assert np.isfinite(values).all()
            counts = np.histogram(values, bins=abort_bins)[0]
            assert int(counts.sum()) == len(values)
            heights = counts / (len(current) * 0.020)
            curves[METRICS[3]][case_index, animal_index, led] = heights
            curves[f"{METRICS[3]}_sem"][case_index, animal_index, led] = 0
            area = float(heights @ np.diff(abort_bins))
            assert np.isclose(area, len(values) / len(current), rtol=0, atol=1e-12)
            matched_audits.append({
                "case": case, "session_type": session_type, "selection": "first_member",
                "row": str(animal), "LED_trial": led, "metric": METRICS[3], "n_animals": 1,
                "original_selected_aborts": int(original.selected_abort.sum()),
                "selected_aborts": len(selected), "finite_aligned_aborts": len(values),
                "missing_aligned_time": 0, "plotted_observations": len(values),
                "original_denominator": len(original), "eligible_denominator": len(current),
                "removed": len(original) - len(current), "histogram_normalizer": len(current),
                "full_histogram_area": area, "expected_area": len(values) / len(current),
                "bin_width_s": 0.020,
                "denominator_definition": "current A or V immediately before adjacent A (AA+VA)",
                "max_pre_bin_rate_difference_s_inv": matching["max_pre_bin_rate_difference_s_inv"],
                "pre_onset_fraction_difference": matching["pre_onset_fraction_difference"],
            })

    removed = eligible.loc[~eligible.retained]
    assert removed.selected_abort.all() and removed.pre_onset_candidate.all()
    assert eligible.loc[~eligible.selected_abort, "retained"].all()
    assert eligible.loc[eligible.selected_abort & eligible.abort_from_LED_onset.ge(0),
                        "retained"].all()
    for led in (0, 1):
        current = eligible.loc[eligible.LED_trial.eq(led) & eligible.retained]
        assert (int(current.selected_abort.sum()), len(current)) == EXPECTED_MATCHED[session_type][led]
    manifest = eligible[["source_csv_row_index", *keys, "session_type", "training_level",
                         "repeat_trial", "LED_trial", "abort_event", "success", "selected_abort",
                         "intended_fix", "LED_onset_time", "timed_fix", "abort_from_LED_onset",
                         "next_trial", "next_source_csv_row_index", "pre_onset_candidate",
                         "pre_onset_bin", "retained"]].copy()
    manifest.insert(0, "source_csv", f"raw_data/LED7_s{session_type}.csv")
    manifests.append(manifest)

    for led in (0, 1):
        six = curves[METRICS[3]][case_index, :6, led]
        mean = six.mean(axis=0)
        sem = six.std(axis=0, ddof=1) / np.sqrt(6)
        curves[METRICS[3]][case_index, 6, led] = mean
        curves[f"{METRICS[3]}_sem"][case_index, 6, led] = sem
        np.testing.assert_allclose(mean, six.sum(axis=0) / 6, rtol=0, atol=1e-14)
        np.testing.assert_allclose(sem,
                                   np.sqrt(((six - mean) ** 2).sum(axis=0) / 5 / 6),
                                   rtol=0, atol=1e-14)
        group_records = [record for record in matched_audits
                         if record["case"] == case and record["LED_trial"] == led]
        assert len(group_records) == 6
        area = float(mean @ np.diff(abort_bins))
        assert np.isclose(area, np.mean([record["full_histogram_area"] for record in group_records]),
                          rtol=0, atol=1e-12)
        matched_audits.append({
            "case": case, "session_type": session_type, "selection": "first_member",
            "row": "equal_animal_mean", "LED_trial": led, "metric": METRICS[3], "n_animals": 6,
            "original_selected_aborts": sum(record["original_selected_aborts"] for record in group_records),
            "selected_aborts": sum(record["selected_aborts"] for record in group_records),
            "finite_aligned_aborts": sum(record["finite_aligned_aborts"] for record in group_records),
            "missing_aligned_time": 0,
            "plotted_observations": sum(record["plotted_observations"] for record in group_records),
            "original_denominator": sum(record["original_denominator"] for record in group_records),
            "eligible_denominator": sum(record["eligible_denominator"] for record in group_records),
            "removed": sum(record["removed"] for record in group_records),
            "histogram_normalizer": np.nan, "full_histogram_area": area,
            "expected_area": np.mean([record["expected_area"] for record in group_records]),
            "bin_width_s": 0.020,
            "denominator_definition": "mean of six animal-normalized AA/(AA+VA) curves",
            "max_pre_bin_rate_difference_s_inv": np.nan,
            "pre_onset_fraction_difference": np.nan,
        })
    off, on = curves[METRICS[3]][case_index, 6, :, :100]
    for record in matched_audits:
        if record["case"] == case and record["row"] == "equal_animal_mean":
            record["max_pre_bin_rate_difference_s_inv"] = float(abs(off - on).max())
            record["pre_onset_fraction_difference"] = float(abs((off - on).sum() * 0.020))

manifest = pd.concat(manifests, ignore_index=True)
assert len(manifest) == sum(sum(n for _, n in EXPECTED_ORIGINAL[st].values()) for st in SESSION_TYPES)
assert not manifest.duplicated(["session_type", *keys]).any()
matching_audit = pd.DataFrame(matching_records)
assert len(matching_audit) == 12 and matching_audit.certified_optimal.all()
assert matching_audit.max_pre_bin_rate_difference_s_inv.le(MAX_BIN_RATE_DIFFERENCE + 1e-12).all()
assert matching_audit.pre_onset_fraction_difference.le(MAX_PRE_FRACTION_DIFFERENCE + 1e-12).all()
audit = pd.concat([old_audit, first_audit[old_audit.columns],
                   pd.DataFrame(matched_audits)[old_audit.columns]], ignore_index=True)
assert len(audit) == 6 * 7 * 2 * 4
assert not audit.duplicated(["case", "row", "LED_trial", "metric"]).any()

for case_index, case in enumerate(CASES):
    for metric in METRICS:
        bins = timing_bins if metric in METRICS[:2] else abort_bins
        for led in (0, 1):
            six = curves[metric][case_index, :6, led]
            mean = six.mean(axis=0)
            sem = six.std(axis=0, ddof=1) / np.sqrt(6)
            np.testing.assert_allclose(curves[metric][case_index, 6, led], mean, rtol=0, atol=1e-14)
            np.testing.assert_allclose(curves[f"{metric}_sem"][case_index, 6, led], sem,
                                       rtol=0, atol=1e-14)
            record = audit.loc[audit.case.eq(case) & audit.row.eq("equal_animal_mean")
                               & audit.LED_trial.eq(led) & audit.metric.eq(metric)]
            assert len(record) == 1
            assert np.isclose(mean @ np.diff(bins), record.iloc[0].full_histogram_area,
                              rtol=0, atol=1e-12)
            if metric in METRICS[:2]:
                assert np.isclose(mean @ np.diff(bins), 1, rtol=0, atol=1e-12)
            else:
                animal_records = audit.loc[audit.case.eq(case) & audit.row.isin(rows[:6])
                                           & audit.LED_trial.eq(led) & audit.metric.eq(metric)]
                assert np.isclose(record.iloc[0].full_histogram_area,
                                  animal_records.full_histogram_area.mean(), rtol=0, atol=1e-12)
assert np.isfinite(np.concatenate([curves[metric].ravel() for metric in METRICS])).all()


# %%
############ Figure: first, second, and isolated rows ############
fig, axes = plt.subplots(6, 4, figsize=(20, 20), sharex="col")
titles = ("intended_fix", "LED_onset_time", "Abort rate aligned to LED onset",
          "Pre-onset matched abort rate")
visible = (abort_bins[:-1] < RATE_DISPLAY_S[1]) & (abort_bins[1:] > RATE_DISPLAY_S[0])
timing_limits = [1.22 * max(float((curves[metric][case_index, 6, led]
                                   + curves[f"{metric}_sem"][case_index, 6, led]).max())
                            for case_index in range(6) for led in (0, 1))
                 for metric in METRICS[:2]]
rate_limit = 1.23 * max(float((curves[metric][case_index, 6, led]
                                + curves[f"{metric}_sem"][case_index, 6, led])[visible].max())
                        for case_index in range(6) for metric in METRICS[2:] for led in (0, 1))

for case_index, case in enumerate(CASES):
    session_type = int(case[1])
    if case.endswith("first_member"):
        row_name, formula = "First member of A→A", "AA / (AA + VA)"
    elif case.endswith("after_abort"):
        row_name, formula = "Second member of A→A", "AA / (AA + AV)"
    else:
        row_name, formula = "Isolated V→A→V", "VAV / (VAV + VVV)"
    for col, metric in enumerate(METRICS):
        ax = axes[case_index, col]
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
            note = (f"{label} n={int(record.plotted_observations):,}, "
                    f"N={int(record.eligible_denominator):,}; area={record.full_histogram_area:.3f}"
                    if is_rate else f"{label} n={int(record.selected_aborts):,}")
            ax.text(0.98, 0.97 - 0.085 * led, note, transform=ax.transAxes,
                    ha="right", va="top", color=COLORS[led], fontsize=8)
        ax.set_xlim(*(RATE_DISPLAY_S if is_rate else (0, 2.2)))
        ax.set_ylim(0, rate_limit if is_rate else timing_limits[col])
        if is_rate:
            ax.axvline(0, color="0.45", linestyle=":", linewidth=1)
        ax.grid(axis="y", color="0.91", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)
        if case_index == 0:
            ax.set_title(titles[col], fontsize=12, pad=12)
        if col == 0:
            ax.set_ylabel(f"Session type {session_type}\n{row_name}\n{formula}\nDensity (s⁻¹)",
                          fontsize=10)
        elif col == 2:
            ax.set_ylabel("Conditional abort rate (s⁻¹)", fontsize=10)
        if case_index == 5:
            ax.set_xlabel("Time from scheduled LED onset (s)" if is_rate
                          else "Time from fixation onset (s)", fontsize=10)
        if col == 3:
            matched_record = audit.loc[audit.case.eq(case) & audit.row.eq("equal_animal_mean")
                                       & audit.LED_trial.eq(0) & audit.metric.eq(metric)].iloc[0]
            ax.text(0.02, 0.77,
                    f"Pre: max |Δrate|={matched_record.max_pre_bin_rate_difference_s_inv:.4f} s⁻¹\n"
                    f"|Δfraction|={100 * matched_record.pre_onset_fraction_difference:.3f} pp",
                    transform=ax.transAxes, va="top", fontsize=8)

fig.suptitle("LED7 sequence-abort timing: conditional denominators", fontsize=18, y=0.995)
handles = [Line2D([0], [0], color=COLORS[led], lw=2,
                  label=f"Plotted trial LED {'OFF' if led == 0 else 'ON'}") for led in (0, 1)]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.971),
           ncol=2, frameon=False, fontsize=12)
fig.text(0.5, 0.019,
         "Equal-animal mean ± SEM: animals 92, 93, 98, 99, 100, 103. A = event-3 abort; "
         "V = valid (success ±1). LED flag belongs to the plotted trial.\n"
         "First pair member: AA/(AA+VA); second: AA/(AA+AV); isolated: VAV/(VAV+VVV). "
         "A→A→A contributes two overlapping pairs. Adjacent trial IDs within animal/session are required.\n"
         "Timing: 40 ms unit-area bins; rates: 20 ms bins. Displayed n and N are summed counts; "
         "full rate area is the mean of six animal fractions, not pooled n/N. OFF onset is scheduled/counterfactual.\n"
         "Column 4 removes only selected pre-onset aborts using observed abort times; "
         "outcome-based matching and future-outcome conditioning are descriptive, not causal evidence.",
         ha="center", fontsize=9, linespacing=1.45)
fig.tight_layout(rect=(0.005, 0.095, 0.995, 0.95), h_pad=1.55, w_pad=1.5)
figure_path = Path(f"{OUTPUT_STEM}_6x4.png")
fig.savefig(figure_path, dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)


# %%
############ Save selection and numerical audits ############
manifest.to_csv(f"{OUTPUT_STEM}_first_member_retained_manifest.csv", index=False)
matching_audit.to_csv(f"{OUTPUT_STEM}_matching_audit.csv", index=False)
pd.DataFrame(quota_records).to_csv(f"{OUTPUT_STEM}_matching_bin_quotas.csv", index=False)
audit.to_csv(f"{OUTPUT_STEM}_count_area_audit.csv", index=False)
np.savez_compressed(f"{OUTPUT_STEM}_curves.npz", **curves)
print(audit.loc[audit.row.eq("equal_animal_mean") & audit.metric.eq(METRICS[3]),
                ["case", "LED_trial", "selected_aborts", "eligible_denominator",
                 "full_histogram_area"]].to_string(index=False))
print(f"Certified {len(matching_audit)} first-member optimizations; saved {figure_path}")
