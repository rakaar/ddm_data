# %%
"""Descriptive pre-onset OFF/ON matching of s8 isolated fixation aborts.

Classification is frozen before subsampling. Only pre-onset isolated aborts may
be removed; every other event-3-or-valid trial stays. Matching uses observed
abort outcomes and is not evidence of a causal LED effect. The integer optimum
is maximal under the specified conservative linear rule, not under every
possible formulation of the nonlinear rate constraints.
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
SOURCE_PATH = REPO / "raw_data/LED7_s8.csv"
ORIGINAL_STEM = SCRIPT_DIR / "led7_s8_isolated_abort_timing"
OUTPUT_STEM = SCRIPT_DIR / "led7_s8_isolated_abort_pre_onset_matched_timing"
ANIMALS = (92, 93, 98, 99, 100, 103)
SEED = 20261001
MAX_BIN_RATE_DIFFERENCE = 0.005  # s^-1
MAX_PRE_ONSET_FRACTION_DIFFERENCE = 0.001
TIMING_BIN_WIDTH_S = 0.040
ABORT_BIN_WIDTH_S = 0.020
PRE_ONSET_RANGE_S = (-2.0, 0.0)
ABORT_DISPLAY_S = (-0.4, 0.5)
SOLVER_TIME_LIMIT_S = 60
FIGURE_DPI = 220
COLORS = {0: "tab:blue", 1: "tab:red"}
METRICS = ("intended_fix", "LED_onset_time", "abort_from_LED_onset",
           "matched_abort_from_LED_onset")
TITLES = ("intended_fix", "LED_onset_time", "Original abort rate aligned to LED onset",
          "Pre-onset matched abort rate")

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
pre_bins = np.linspace(*PRE_ONSET_RANGE_S, round(2 / ABORT_BIN_WIDTH_S) + 1)
n_bins = len(pre_bins) - 1
np.testing.assert_array_equal(abort_bins[:n_bins + 1], pre_bins)
bins_by_metric = dict(zip(METRICS, (timing_bins, timing_bins, abort_bins, abort_bins)))
width_by_metric = dict(zip(METRICS, (TIMING_BIN_WIDTH_S, TIMING_BIN_WIDTH_S,
                                   ABORT_BIN_WIDTH_S, ABORT_BIN_WIDTH_S)))
protected_paths = [SOURCE_PATH, Path(f"{ORIGINAL_STEM}_selected_trials.csv"),
                   Path(f"{ORIGINAL_STEM}_count_area_audit.csv"),
                   Path(f"{ORIGINAL_STEM}_curve_audit.csv"),
                   Path(f"{ORIGINAL_STEM}_7x3.png"),
                   REPO / "docs/assets/results/2026-09-30/led7_s8_isolated_abort_timing_7x3.png"]
input_hashes = {}
for path in protected_paths:
    with path.open("rb") as handle:
        input_hashes[str(path.relative_to(REPO))] = hashlib.file_digest(handle, "sha256").hexdigest()


# %%
############ Reproduce original identities using actual neighboring trial IDs ############
source = pd.read_csv(SOURCE_PATH, float_precision="round_trip")
assert source.shape == (148_428, 52)
raw_columns = list(source.columns)
assert source["training_level"].eq(16).all() and source["session_type"].eq(8).all()
assert (source["repeat_trial"].isin([0, 2]) | source["repeat_trial"].isna()).all()
assert source["LED_trial"].isin([0, 1]).all()
keys = ["animal", "session", "trial"]
assert not source.duplicated(keys).any()
source["source_csv_row_index"] = np.arange(len(source))
data = source.loc[source["animal"].isin(ANIMALS)].sort_values(keys, kind="stable").copy()
assert len(data) == 113_941
for column in [*keys, "LED_trial"]:
    assert data[column].notna().all() and data[column].eq(np.floor(data[column])).all()
    data[column] = data[column].astype(int)
data = data.set_index("source_csv_row_index", drop=False)
grouped = data.groupby(["animal", "session"], sort=False)
neighbor_columns = ["trial", "success", "abort_event", "LED_trial", "source_csv_row_index"]
previous = grouped[neighbor_columns].shift(1)
following = grouped[neighbor_columns].shift(-1)
is_abort = data["abort_event"].eq(3)
is_valid = data["success"].isin([-1, 1])
assert not (is_abort & is_valid).any()
data["eligible_denominator_trial"] = is_abort | is_valid
data["isolated_abort"] = (is_abort & previous["trial"].eq(data["trial"] - 1)
                          & following["trial"].eq(data["trial"] + 1)
                          & previous["success"].isin([-1, 1])
                          & following["success"].isin([-1, 1]))
for prefix, neighbors in (("previous", previous), ("next", following)):
    for column in neighbor_columns:
        data[f"{prefix}_{column}"] = neighbors[column]
# Independent exact-key lookup: gaps and sessions cannot be bridged by shifts.
lookup = data.set_index(keys)
prev_keys = pd.MultiIndex.from_arrays([data.animal, data.session, data.trial - 1], names=keys)
next_keys = pd.MultiIndex.from_arrays([data.animal, data.session, data.trial + 1], names=keys)
np.testing.assert_array_equal(
    data["isolated_abort"], is_abort.to_numpy()
    & lookup.reindex(prev_keys)["success"].isin([-1, 1]).to_numpy()
    & lookup.reindex(next_keys)["success"].isin([-1, 1]).to_numpy(),
)
original_selection = pd.read_csv(f"{ORIGINAL_STEM}_selected_trials.csv", float_precision="round_trip")
assert len(original_selection) == 10_511 and not original_selection.duplicated(keys).any()
assert not original_selection["source_csv_row_index"].duplicated().any()
selected = data.loc[data["isolated_abort"]].sort_index()
saved_selected = original_selection.set_index("source_csv_row_index").sort_index()
pd.testing.assert_frame_equal(selected[raw_columns], saved_selected[raw_columns], check_dtype=False)
for column in ["previous_trial", "next_trial", "previous_success", "next_success",
               "previous_source_csv_row_index", "next_source_csv_row_index"]:
    np.testing.assert_array_equal(selected[column], saved_selected[column])
data["abort_from_LED_onset"] = data["timed_fix"] - data["LED_onset_time"]
assert np.isfinite(data.loc[data["isolated_abort"], ["intended_fix", "LED_onset_time",
                                                   "abort_from_LED_onset"]]).all().all()
assert selected["timed_fix"].lt(selected["intended_fix"]).all()
eligible = data.loc[data["eligible_denominator_trial"]].copy()
assert len(eligible) == 107_070
assert eligible.groupby("LED_trial").size().to_dict() == {0: 68_584, 1: 38_486}
assert eligible.groupby("LED_trial")["isolated_abort"].sum().to_dict() == {0: 6_406, 1: 4_105}
eligible["pre_onset_candidate"] = eligible["isolated_abort"] & eligible["abort_from_LED_onset"].lt(0)
candidate = eligible["pre_onset_candidate"]
assert eligible.loc[candidate, "abort_from_LED_onset"].ge(-2).all()
eligible["pre_onset_bin"] = -1
eligible.loc[candidate, "pre_onset_bin"] = (
    np.searchsorted(pre_bins, eligible.loc[candidate, "abort_from_LED_onset"], side="right") - 1
)
assert eligible.loc[candidate, "pre_onset_bin"].between(0, n_bins - 1).all()
eligible["retained"] = True


# %%
############ Certified integer quotas, then intact-row sampling without replacement ############
rng = np.random.default_rng(SEED)
optimizer_records, quota_records = [], []
for animal in ANIMALS:
    group = eligible.loc[eligible.animal.eq(animal)]
    denominators = group.groupby("LED_trial").size().reindex([0, 1]).to_numpy(dtype=int)
    available = np.stack([
        np.histogram(group.loc[group.LED_trial.eq(led) & group.pre_onset_candidate,
                               "abort_from_LED_onset"], bins=pre_bins)[0]
        for led in (0, 1)
    ])
    mandatory = denominators - available.sum(axis=1)
    ratio = mandatory[0] / mandatory[1]
    u_max = float((available / mandatory[:, None]).max())
    epsilon = ABORT_BIN_WIDTH_S * MAX_BIN_RATE_DIFFERENCE - u_max * MAX_PRE_ONSET_FRACTION_DIFFERENCE
    assert (mandatory > 0).all() and epsilon > 0, "Requested conservative rule is infeasible as specified."
    constraint_matrix = vstack([
        hstack([eye(n_bins), -ratio * eye(n_bins)]),
        csr_matrix(np.r_[np.ones(n_bins), -ratio * np.ones(n_bins)][None, :]),
    ], format="csr")
    tolerance = np.r_[np.full(n_bins, mandatory[0] * epsilon),
                      mandatory[0] * MAX_PRE_ONSET_FRACTION_DIFFERENCE]
    started = time.perf_counter()
    solution = milp(
        c=-np.ones(2 * n_bins), integrality=np.ones(2 * n_bins),
        bounds=Bounds(np.zeros(2 * n_bins), available.ravel()),
        constraints=LinearConstraint(constraint_matrix, -tolerance, tolerance),
        options={"time_limit": SOLVER_TIME_LIMIT_S, "mip_rel_gap": 0.0},
    )
    elapsed = time.perf_counter() - started
    assert solution.success and solution.status == 0, (animal, solution.message)
    assert solution.mip_gap == 0 and np.isfinite(solution.x).all()
    quotas = np.rint(solution.x).astype(int).reshape(2, n_bins)
    np.testing.assert_allclose(solution.x, quotas.ravel(), rtol=0, atol=1e-7)
    assert ((quotas >= 0) & (quotas <= available)).all()
    assert np.all(np.abs(constraint_matrix @ quotas.ravel()) <= tolerance + 1e-7)
    assert np.isclose(solution.fun, -quotas.sum(), rtol=0, atol=1e-7)
    assert np.isclose(solution.mip_dual_bound, solution.fun, rtol=0, atol=1e-7)

    for led in (0, 1):
        for bin_index in range(n_bins):
            pool = group.index[group.LED_trial.eq(led) & group.pre_onset_candidate
                               & group.pre_onset_bin.eq(bin_index)].to_numpy()
            pool.sort()
            assert len(pool) == available[led, bin_index]
            chosen = rng.choice(pool, size=quotas[led, bin_index], replace=False)
            assert len(np.unique(chosen)) == len(chosen)
            eligible.loc[pool, "retained"] = False
            eligible.loc[chosen, "retained"] = True
            quota_records.append({
                "animal": animal, "LED_trial": led, "bin_index": bin_index,
                "bin_left_s": pre_bins[bin_index], "bin_right_s": pre_bins[bin_index + 1],
                "available": len(pool), "retained": len(chosen), "removed": len(pool) - len(chosen),
            })
    new_denominators = mandatory + quotas.sum(axis=1)
    matched_rates = quotas / (new_denominators[:, None] * ABORT_BIN_WIDTH_S)
    max_rate_difference = float(np.abs(matched_rates[0] - matched_rates[1]).max())
    pre_fractions = quotas.sum(axis=1) / new_denominators
    pre_fraction_difference = float(abs(pre_fractions[0] - pre_fractions[1]))
    assert max_rate_difference <= MAX_BIN_RATE_DIFFERENCE + 1e-12
    assert pre_fraction_difference <= MAX_PRE_ONSET_FRACTION_DIFFERENCE + 1e-12
    optimizer_records.append({
        "animal": animal, "solver": "scipy.optimize.milp / HiGHS", "status": int(solution.status),
        "message": solution.message, "certified_optimal": True, "mip_gap": solution.mip_gap,
        "objective": solution.fun, "dual_bound": solution.mip_dual_bound,
        "mip_node_count": solution.mip_node_count, "elapsed_s": elapsed,
        "original_OFF": denominators[0], "original_ON": denominators[1],
        "mandatory_OFF": mandatory[0], "mandatory_ON": mandatory[1],
        "retained_OFF": new_denominators[0], "retained_ON": new_denominators[1],
        "removed": int((available - quotas).sum()), "r": ratio, "u_max": u_max, "epsilon": epsilon,
        "max_bin_rate_difference_s_inv": max_rate_difference,
        "pre_onset_fraction_difference": pre_fraction_difference,
    })
    print(f"Animal {animal}: optimal, retained {new_denominators.sum():,}/{denominators.sum():,}; "
          f"max bin difference {max_rate_difference:.6f} s^-1; fraction difference {pre_fraction_difference:.6f}", flush=True)
optimizer = pd.DataFrame(optimizer_records)
quota_audit = pd.DataFrame(quota_records)


# %%
############ Check membership and invariants before computing the new rates ############
kept = eligible.loc[eligible.retained]
removed = eligible.loc[~eligible.retained]
assert removed["pre_onset_candidate"].all() and removed["abort_event"].eq(3).all()
assert not removed["success"].isin([-1, 1]).any()
assert kept.index.is_unique and not kept.duplicated(keys).any()
assert len(kept) + len(removed) == len(eligible)
assert eligible.loc[~eligible.pre_onset_candidate, "retained"].all()
for led in (0, 1):
    original_group = eligible.loc[eligible.LED_trial.eq(led)]
    kept_group = kept.loc[kept.LED_trial.eq(led)]
    for mask_column in ("success", "abort_from_LED_onset"):
        protected = (original_group[mask_column].isin([-1, 1]) if mask_column == "success"
                     else original_group[mask_column].ge(0))
        assert original_group.index[protected].isin(kept_group.index).all()
    assert (original_group.success.isin([-1, 1]).sum() == kept_group.success.isin([-1, 1]).sum())
    assert (original_group.abort_from_LED_onset.ge(0).sum() == kept_group.abort_from_LED_onset.ge(0).sum())
if (MAX_BIN_RATE_DIFFERENCE, MAX_PRE_ONSET_FRACTION_DIFFERENCE, ABORT_BIN_WIDTH_S) == (0.005, 0.001, 0.020):
    assert len(removed) == 1_706 and len(kept) == 105_364
    assert kept.groupby("LED_trial")["isolated_abort"].sum().to_dict() == {0: 4_846, 1: 3_959}
    assert kept.groupby("LED_trial").size().to_dict() == {0: 67_024, 1: 38_340}


# %%
############ Reproduce the original columns; add matched rates and equal-animal SEM ############
original_curves = pd.read_csv(f"{ORIGINAL_STEM}_curve_audit.csv", float_precision="round_trip")
original_panels = pd.read_csv(f"{ORIGINAL_STEM}_count_area_audit.csv", float_precision="round_trip")
curves, errors = {}, {}
panel_records, curve_parts = [], []
for animal in ANIMALS:
    for led in (0, 1):
        group = eligible.loc[eligible.animal.eq(animal) & eligible.LED_trial.eq(led)]
        for metric in METRICS:
            matched = metric == METRICS[3]
            current = group.loc[group.retained] if matched else group
            aborts = current.loc[current.isolated_abort]
            value_column = "abort_from_LED_onset" if matched else metric
            values = aborts[value_column].to_numpy()
            values = values[np.isfinite(values)]
            bins, width = bins_by_metric[metric], width_by_metric[metric]
            normalizer = len(current) if metric in METRICS[2:] else len(values)
            counts = np.histogram(values, bins=bins)[0]
            assert counts.sum() == len(values)
            heights = counts / (normalizer * width)
            area = float(heights @ np.diff(bins))
            expected_area = len(values) / normalizer
            assert np.isclose(area, expected_area, rtol=0, atol=1e-12)
            if not matched:
                old = original_curves.loc[original_curves.row.eq(str(animal))
                                          & original_curves.LED_trial.eq(led) & original_curves.metric.eq(metric)]
                np.testing.assert_array_equal(heights, old.height)
                np.testing.assert_array_equal(counts, old.bin_count)
                old_panel = original_panels.loc[original_panels.row.eq(str(animal))
                                               & original_panels.LED_trial.eq(led)
                                               & original_panels.metric.eq(metric)].iloc[0]
                assert len(current) == old_panel.eligible_denominator
                assert len(aborts) == old_panel.selected_aborts
            else:
                expected_quotas = quota_audit.loc[quota_audit.animal.eq(animal) & quota_audit.LED_trial.eq(led)]
                np.testing.assert_array_equal(counts[:n_bins], expected_quotas.retained)
            curves[(str(animal), led, metric)] = heights
            errors[(str(animal), led, metric)] = np.zeros_like(heights)
            panel_records.append({
                "row": str(animal), "LED_trial": led, "metric": metric, "n_animals": 1,
                "original_isolated_aborts": int(group.isolated_abort.sum()),
                "selected_aborts": len(aborts), "finite_plotted": len(values),
                "original_eligible_trials": len(group), "eligible_denominator": len(current),
                "removed": len(group) - len(current), "histogram_normalizer": normalizer,
                "full_histogram_area": area, "expected_area": expected_area,
                "pre_onset_fraction": float(counts[:n_bins].sum() / normalizer) if metric in METRICS[2:] else np.nan,
                "bin_width_s": width,
            })
            curve_parts.append(pd.DataFrame({
                "row": str(animal), "LED_trial": led, "metric": metric,
                "bin_left_s": bins[:-1], "bin_right_s": bins[1:],
                "height": heights, "sem": np.nan, "bin_count": counts,
            }))
animal_panels = pd.DataFrame(panel_records)
for led in (0, 1):
    for metric in METRICS:
        bins = bins_by_metric[metric]
        matrix = np.stack([curves[(str(animal), led, metric)] for animal in ANIMALS])
        mean = matrix.mean(axis=0)
        sem = matrix.std(axis=0, ddof=1) / np.sqrt(len(ANIMALS))
        np.testing.assert_allclose(sem, np.sqrt(((matrix - mean) ** 2).sum(axis=0) / 30), atol=1e-14)
        curves[("equal_animal_mean", led, metric)] = mean
        errors[("equal_animal_mean", led, metric)] = sem
        if metric != METRICS[3]:
            old = original_curves.loc[original_curves.row.eq("equal_animal_mean")
                                     & original_curves.LED_trial.eq(led) & original_curves.metric.eq(metric)]
            np.testing.assert_array_equal(mean, old.height)
            np.testing.assert_array_equal(sem, old["sem"])
        animal_audits = animal_panels.loc[animal_panels.LED_trial.eq(led) & animal_panels.metric.eq(metric)]
        mean_area = float(mean @ np.diff(bins))
        assert np.isclose(mean_area, animal_audits.expected_area.mean(), rtol=0, atol=1e-12)
        record = {"row": "equal_animal_mean", "LED_trial": led, "metric": metric, "n_animals": 6,
                  "full_histogram_area": mean_area, "expected_area": animal_audits.expected_area.mean(),
                  "pre_onset_fraction": animal_audits.pre_onset_fraction.mean(),
                  "histogram_normalizer": np.nan, "bin_width_s": width_by_metric[metric]}
        for column in ("original_isolated_aborts", "selected_aborts", "finite_plotted",
                       "original_eligible_trials", "eligible_denominator", "removed"):
            record[column] = int(animal_audits[column].sum())
        panel_records.append(record)
        curve_parts.append(pd.DataFrame({
            "row": "equal_animal_mean", "LED_trial": led, "metric": metric,
            "bin_left_s": bins[:-1], "bin_right_s": bins[1:],
            "height": mean, "sem": sem, "bin_count": np.nan,
        }))
panels = pd.DataFrame(panel_records)
curve_audit = pd.concat(curve_parts, ignore_index=True)
row_labels = [*[str(animal) for animal in ANIMALS], "equal_animal_mean"]
for row in row_labels:
    for metric in METRICS[2:]:
        off, on = (curves[(row, led, metric)][:n_bins] for led in (0, 1))
        max_difference = float(np.abs(off - on).max())
        mass_difference = float(abs((off - on).sum() * ABORT_BIN_WIDTH_S))
        mask = panels.row.eq(row) & panels.metric.eq(metric)
        panels.loc[mask, "max_pre_bin_rate_difference_s_inv"] = max_difference
        panels.loc[mask, "pre_onset_fraction_difference"] = mass_difference
        if metric == METRICS[3]:
            assert max_difference <= MAX_BIN_RATE_DIFFERENCE + 1e-12
            assert mass_difference <= MAX_PRE_ONSET_FRACTION_DIFFERENCE + 1e-12
assert len(panels) == 7 * 2 * 4
assert not curve_audit.duplicated(["row", "LED_trial", "metric", "bin_left_s"]).any()
assert curve_audit.height.ge(0).all() and np.isfinite(curve_audit.height).all()


# %%
############ Save auditable row identities, quotas, solver certificates, and curves ############
manifest_columns = ["source_csv_row_index", *keys, "session_type", "training_level", "repeat_trial",
                    "LED_trial", "abort_event", "success", "intended_fix", "LED_onset_time", "timed_fix",
                    "abort_from_LED_onset", "eligible_denominator_trial", "isolated_abort",
                    "previous_trial", "next_trial", "previous_success", "next_success",
                    "previous_source_csv_row_index", "next_source_csv_row_index",
                    "pre_onset_candidate", "pre_onset_bin", "retained"]
selection_manifest = eligible[manifest_columns].copy()
selection_manifest.insert(0, "source_csv", str(SOURCE_PATH.relative_to(REPO)))
selection_manifest.to_csv(f"{OUTPUT_STEM}_eligible_trial_selection_manifest.csv", index=False)
quota_audit.to_csv(f"{OUTPUT_STEM}_bin_quotas.csv", index=False)
optimizer.to_csv(f"{OUTPUT_STEM}_optimizer_diagnostics.csv", index=False)
panels.to_csv(f"{OUTPUT_STEM}_count_area_audit.csv", index=False)
curve_audit.to_csv(f"{OUTPUT_STEM}_curve_audit.csv", index=False)
# Array axes: animal/mean row, LED condition, histogram bin. CSV stores the same arrays.
np.savez_compressed(
    f"{OUTPUT_STEM}_curves.npz", rows=np.array(row_labels), led_conditions=np.array([0, 1]),
    timing_bins=timing_bins, abort_bins=abort_bins,
    **{metric: np.stack([[curves[(row, led, metric)] for led in (0, 1)] for row in row_labels]) for metric in METRICS},
    **{f"{metric}_sem": np.stack([[errors[(row, led, metric)] for led in (0, 1)] for row in row_labels]) for metric in METRICS},
)
summary = {
    "source_hashes_sha256": input_hashes, "seed": SEED, "numpy_version": np.__version__,
    "scipy_version": scipy.__version__, "max_bin_rate_difference_s_inv": MAX_BIN_RATE_DIFFERENCE,
    "max_pre_onset_fraction_difference": MAX_PRE_ONSET_FRACTION_DIFFERENCE,
    "original_eligible_trials": len(eligible), "retained_eligible_trials": len(kept),
    "removed_pre_onset_isolated_aborts": len(removed),
    "original_isolated_aborts": int(eligible.isolated_abort.sum()),
    "retained_isolated_aborts": int(kept.isolated_abort.sum()),
    "valid_trials_unchanged": int(kept.success.isin([-1, 1]).sum()),
    "post_onset_isolated_aborts_unchanged": int((kept.isolated_abort & kept.abort_from_LED_onset.ge(0)).sum()),
    "all_solvers_certified_optimal": bool(optimizer.certified_optimal.all()),
    "optimality_scope": "Maximum retention under the specified conservative linear matching rule.",
    "classification": "Frozen original valid-abort-valid labels based on actual adjacent trial IDs.",
    "normalization": "Selected finite isolated aborts / retained event-3-or-valid trials / bin width.",
    "caveat": "Overlap is enforced using observed abort times; descriptive, not causal evidence.",
}
Path(f"{OUTPUT_STEM}_run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


# %%
############ Original three columns plus matched rates, without smoothing ############
fig, axes = plt.subplots(7, 4, figsize=(20, 20), sharex="col", sharey="col")
for row_index, row in enumerate(row_labels):
    for column, metric in enumerate(METRICS):
        ax, bins = axes[row_index, column], bins_by_metric[metric]
        for led in (0, 1):
            heights, sem = curves[(row, led, metric)], errors[(row, led, metric)]
            if row == "equal_animal_mean":
                ax.fill_between(bins, np.r_[heights - sem, (heights - sem)[-1]],
                                np.r_[heights + sem, (heights + sem)[-1]], step="post",
                                color=COLORS[led], alpha=0.16, linewidth=0)
            ax.stairs(heights, bins, color=COLORS[led], linewidth=1.25)
            audit = panels.loc[panels.row.eq(row) & panels.LED_trial.eq(led) & panels.metric.eq(metric)].iloc[0]
            label = "OFF" if led == 0 else "ON"
            if column < 2:
                note = f"{label} {'total n' if row_index == 6 else 'n'} = {int(audit.selected_aborts):,}"
            elif row_index == 6:
                note = f"{label} mean area = {audit.full_histogram_area:.3f}"
            else:
                note = f"{label} {int(audit.selected_aborts):,}/{int(audit.eligible_denominator):,}; area = {audit.full_histogram_area:.3f}"
            ax.text(0.98, 0.97 - 0.085 * led, note, transform=ax.transAxes,
                    ha="right", va="top", color=COLORS[led], fontsize=8.4)
        if column >= 2:
            ax.axvline(0, color="0.45", linestyle=":", linewidth=1)
            ax.set_xlim(*ABORT_DISPLAY_S)
            if column == 3:
                note = (f"Pre: max Δrate = {audit.max_pre_bin_rate_difference_s_inv:.4f} s⁻¹\n"
                        f"|Δfraction| = {100 * audit.pre_onset_fraction_difference:.3f} pp")
                ax.text(0.02, 0.79, note, transform=ax.transAxes, ha="left", va="top", fontsize=7.8)
        else:
            ax.set_xlim(0, 2.2)
        if column == 0:
            label = "Equal-animal mean ± SEM" if row_index == 6 else f"Animal {row}"
            ax.set_ylabel(f"{label}\nDensity (s⁻¹)", fontsize=10)
        elif column == 2:
            ax.set_ylabel("Isolated-abort rate (s⁻¹)", fontsize=10)
        if row_index == 0:
            ax.set_title(TITLES[column], fontsize=12, pad=10)
        if row_index == 6:
            ax.set_xlabel("Time from scheduled LED onset (s)" if column >= 2 else "Time from fixation onset (s)", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.grid(axis="y", color="0.91", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)

# Match the original first-three-column scales; allow extra headroom only if
# the newly normalized fourth-column curves require it. Rate columns share y.
upper_limits = []
for column, metric in enumerate(METRICS):
    bins = bins_by_metric[metric]
    visible = ((bins[:-1] < ABORT_DISPLAY_S[1]) & (bins[1:] > ABORT_DISPLAY_S[0])) if column >= 2 else np.ones(len(bins) - 1, dtype=bool)
    peak = max(float((curves[(row, led, metric)] + errors[(row, led, metric)])[visible].max())
               for row in row_labels for led in (0, 1))
    upper_limits.append(peak * 1.28)
upper_limits[2] = upper_limits[3] = max(upper_limits[2:])
for column, upper in enumerate(upper_limits):
    axes[0, column].set_ylim(0, upper)
fig.suptitle("LED7 session type 8: isolated valid–abort–valid trials", fontsize=18, y=0.995)
fig.legend(handles=[Line2D([0], [0], color=COLORS[led], lw=1.7, label=f"Current trial LED {'OFF' if led == 0 else 'ON'}")
                    for led in (0, 1)], loc="upper center", bbox_to_anchor=(0.5, 0.981), ncol=2, frameon=False, fontsize=12)
fig.text(0.5, 0.013,
         f"Timing columns unchanged (40 ms, unit area); abort rates use 20 ms bins. Areas refer to full −2 to +2 s histograms.\n"
         f"Matched: {int(kept.isolated_abort.sum()):,}/{int(eligible.isolated_abort.sum()):,} isolated aborts; {len(kept):,}/{len(eligible):,} eligible trials retained. "
         "Rate denominators include all retained event-3 or success ±1 trials.\n"
         "Only pre-onset isolated aborts removed; all valid and post-onset rows retained. OFF zero is scheduled/counterfactual onset.\n"
         "Overlap is enforced using observed abort times: descriptive matching, not evidence of a causal LED effect.",
         ha="center", fontsize=9, linespacing=1.4)
fig.tight_layout(rect=(0.012, 0.067, 0.995, 0.96), h_pad=1.4, w_pad=1.3)
figure_path = Path(f"{OUTPUT_STEM}_7x4.png")
fig.savefig(figure_path, dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)

for path in protected_paths:
    with path.open("rb") as handle:
        assert hashlib.file_digest(handle, "sha256").hexdigest() == input_hashes[str(path.relative_to(REPO))]
print(optimizer[["animal", "retained_OFF", "retained_ON", "removed", "mip_gap",
                 "max_bin_rate_difference_s_inv", "pre_onset_fraction_difference"]].to_string(index=False))
print(f"Retained {kept.isolated_abort.sum():,}/{eligible.isolated_abort.sum():,} isolated aborts and "
      f"{len(kept):,}/{len(eligible):,} eligible trials; removed {len(removed):,} pre-onset isolated aborts.")
print("All original identities/curves, solver optimality, protected rows, tolerances, areas, and mean/SEM checks passed.")
print(f"Saved {figure_path}")
