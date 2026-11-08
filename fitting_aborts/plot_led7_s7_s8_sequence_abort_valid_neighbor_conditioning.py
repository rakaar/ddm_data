# %%
"""Compare isolated-abort incidence with abort probability given valid neighbors.

The numerator remains valid -> event-3 abort -> valid. The new denominator is
valid -> (event-3 abort OR valid) -> valid, with actual adjacent trial IDs.
The other four rows of the existing six-row figure are unchanged. Old scripts,
figures, manifests, and data are read-only inputs. No result-book publication.
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
OLD_STEM = SCRIPT_DIR / "led7_s7_s8_sequence_abort_pre_onset_matched_summary_with_first"
OUTPUT_STEM = SCRIPT_DIR / "led7_s7_s8_sequence_abort_valid_neighbor_conditioning"
ANIMALS = (92, 93, 98, 99, 100, 103)
SESSION_TYPES = (7, 8)
SEED = 20261001  # Reset for each session type, as in the earlier matcher.
ABORT_BIN_WIDTH_S = 0.020
TIMING_BIN_WIDTH_S = 0.040
MAX_BIN_RATE_DIFFERENCE = 0.005  # s^-1, independently for each animal.
MAX_PRE_ONSET_FRACTION_DIFFERENCE = 0.001
SOLVER_TIME_LIMIT_S = 60
ABORT_DISPLAY_S = (-0.4, 0.5)
FIGURE_DPI = 240
COLORS = {0: "tab:blue", 1: "tab:red"}
METRICS = ("intended_fix", "LED_onset_time", "abort_from_LED_onset", "matched_abort_from_LED_onset")
EXPECTED = {
    7: {0: (6498, 63525, 43706), 1: (3365, 30910, 18532)},
    8: {0: (6406, 68584, 50811), 1: (4105, 38486, 21044)},
}  # Selected aborts, old denominator, new denominator, before matching.
NEW_DENOMINATOR = "event-3 or valid current trials with immediately adjacent valid trials on both sides"

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

assert OUTPUT_STEM != OLD_STEM
protected_paths = [REPO / f"raw_data/LED7_s{st}.csv" for st in SESSION_TYPES]
protected_paths += list(SCRIPT_DIR.glob("led7_s7_s8_sequence_abort_pre_onset_matched_summary*"))
protected_paths += [SCRIPT_DIR / "plot_led7_s7_s8_sequence_abort_pre_onset_matched_summary.py"]
protected_paths += list((REPO / "docs/assets/results/2026-10-01").glob(
    "led7_s7_s8_sequence_abort_pre_onset_matched_summary*"))
input_hashes = {}
for path in protected_paths:
    with path.open("rb") as handle:
        input_hashes[str(path.relative_to(REPO))] = hashlib.file_digest(handle, "sha256").hexdigest()

with np.load(f"{OLD_STEM}_curves.npz") as payload:
    old_arrays = {key: payload[key].copy() for key in payload.files}
arrays = {key: value.copy() for key, value in old_arrays.items()}
cases, rows = arrays["cases"].tolist(), arrays["rows"].tolist()
assert cases == ["s7_first", "s8_first", "s7_after_abort", "s8_after_abort", "s7_isolated", "s8_isolated"]
assert rows == [*[str(animal) for animal in ANIMALS], "equal_animal_mean"]
old_panels = pd.read_csv(f"{OLD_STEM}_count_area_audit.csv", float_precision="round_trip")
panels = old_panels.copy()
timing_bins, abort_bins = arrays["timing_bins"], arrays["abort_bins"]
pre_bins = np.linspace(-2, 0, 101)
n_pre_bins = len(pre_bins) - 1
np.testing.assert_array_equal(abort_bins[:n_pre_bins + 1], pre_bins)
np.testing.assert_allclose(np.diff(abort_bins), ABORT_BIN_WIDTH_S, rtol=0, atol=1e-14)
np.testing.assert_allclose(np.diff(timing_bins), TIMING_BIN_WIDTH_S, rtol=0, atol=1e-14)


# %%
############ Same conservative MILP rule, now with the conditional denominator ############
def optimize_pre_onset_counts(available, denominators):
    """Maximize retained rows under the previous script's linear matching rule."""
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
    start = time.perf_counter()
    result = milp(
        -np.ones(2 * n_pre_bins), integrality=np.ones(2 * n_pre_bins),
        bounds=Bounds(np.zeros(2 * n_pre_bins), available.ravel()),
        constraints=LinearConstraint(matrix, -tolerance, tolerance),
        options={"mip_rel_gap": 0.0, "time_limit": SOLVER_TIME_LIMIT_S},
    )
    assert result.success and result.status == 0, result.message
    assert result.mip_gap == 0 and np.isfinite(result.x).all()
    quotas = np.rint(result.x).astype(int).reshape(2, n_pre_bins)
    np.testing.assert_allclose(result.x, quotas.ravel(), rtol=0, atol=1e-7)
    assert ((quotas >= 0) & (quotas <= available)).all()
    assert np.all(np.abs(matrix @ quotas.ravel()) <= tolerance + 1e-7)
    np.testing.assert_allclose([result.fun, result.mip_dual_bound], -quotas.sum(), rtol=0, atol=1e-7)
    new_n = mandatory + quotas.sum(axis=1)
    rates = quotas / (new_n[:, None] * ABORT_BIN_WIDTH_S)
    max_difference = float(abs(rates[0] - rates[1]).max())
    fraction_difference = float(abs(quotas[0].sum() / new_n[0] - quotas[1].sum() / new_n[1]))
    assert max_difference <= MAX_BIN_RATE_DIFFERENCE + 1e-12
    assert fraction_difference <= MAX_PRE_ONSET_FRACTION_DIFFERENCE + 1e-12
    return quotas, {
        "solver": "scipy.optimize.milp / HiGHS", "certified_optimal": True,
        "status": result.status, "mip_gap": result.mip_gap, "objective": result.fun,
        "dual_bound": result.mip_dual_bound, "elapsed_s": time.perf_counter() - start,
        "r": ratio, "u_max": u_max, "epsilon": epsilon,
        "original_OFF": denominators[0], "original_ON": denominators[1],
        "retained_OFF": new_n[0], "retained_ON": new_n[1],
        "removed": int((available - quotas).sum()),
        "max_pre_bin_rate_difference_s_inv": max_difference,
        "pre_onset_fraction_difference": fraction_difference,
    }


# %%
############ Classify neighbors on the intact CSV stream, then freeze membership ############
keys = ["animal", "session", "trial"]
manifests, optimizer_records, quota_records, retention_records = {}, [], [], []
for st in SESSION_TYPES:
    data = pd.read_csv(REPO / f"raw_data/LED7_s{st}.csv", float_precision="round_trip")
    assert data.shape == ({7: 138440, 8: 148428}[st], 52)
    assert data.training_level.eq(16).all() and data.session_type.eq(st).all()
    assert (data.repeat_trial.isin([0, 2]) | data.repeat_trial.isna()).all()
    assert data.LED_trial.isin([0, 1]).all() and not data.duplicated(keys).any()
    data["source_csv_row_index"] = np.arange(len(data))
    data = data.loc[data.animal.isin(ANIMALS)].sort_values(keys, kind="stable").copy()
    data = data.set_index("source_csv_row_index", drop=False)
    grouped = data.groupby(["animal", "session"], sort=False)
    fields = ["trial", "success", "abort_event", "source_csv_row_index"]
    previous, following = grouped[fields].shift(1), grouped[fields].shift(-1)
    for name, neighbors in (("previous", previous), ("next", following)):
        for field in fields:
            data[f"{name}_{field}"] = neighbors[field]
    is_abort, is_valid = data.abort_event.eq(3), data.success.isin([-1, 1])
    assert not (is_abort & is_valid).any()
    data["old_eligible"] = is_abort | is_valid
    data["both_neighbors_valid"] = (
        previous.trial.eq(data.trial - 1) & following.trial.eq(data.trial + 1)
        & previous.success.isin([-1, 1]) & following.success.isin([-1, 1])
    )
    data["new_eligible"] = data.old_eligible & data.both_neighbors_valid
    data["selected_abort"] = is_abort & data.both_neighbors_valid
    data["valid_valid_valid"] = is_valid & data.both_neighbors_valid
    np.testing.assert_array_equal(data.new_eligible, data.selected_abort | data.valid_valid_valid)
    # Independent exact-key lookup verifies trial-ID adjacency, gaps and boundaries.
    lookup = data.set_index(keys)
    prev_keys = pd.MultiIndex.from_arrays([data.animal, data.session, data.trial - 1], names=keys)
    next_keys = pd.MultiIndex.from_arrays([data.animal, data.session, data.trial + 1], names=keys)
    np.testing.assert_array_equal(
        data.both_neighbors_valid,
        lookup.reindex(prev_keys).success.isin([-1, 1]).to_numpy()
        & lookup.reindex(next_keys).success.isin([-1, 1]).to_numpy(),
    )
    data["abort_from_LED_onset"] = data.timed_fix - data.LED_onset_time
    saved = pd.read_csv(f"{OLD_STEM}_s{st}_isolated_selection_manifest.csv", float_precision="round_trip")
    saved = saved.set_index("source_csv_row_index", drop=False)
    old_eligible = data.loc[data.old_eligible].sort_index()
    saved = saved.sort_index()
    compare_fields = [*keys, "LED_trial", "abort_event", "success", "intended_fix", "LED_onset_time", "timed_fix"]
    pd.testing.assert_frame_equal(old_eligible[compare_fields], saved[compare_fields], check_dtype=False)
    np.testing.assert_array_equal(old_eligible.selected_abort, saved.selected_abort)
    # Keep the old matching flags as an audit only; the new matcher starts afresh.
    data["old_matched_retained"] = data.index.to_series().map(saved.retained)
    for led in (0, 1):
        group = data.loc[data.LED_trial.eq(led)]
        assert (group.selected_abort.sum(), group.old_eligible.sum(), group.new_eligible.sum()) == EXPECTED[st][led]

    eligible = data.loc[data.new_eligible].copy()
    selected = eligible.loc[eligible.selected_abort]
    assert np.isfinite(selected[["intended_fix", "LED_onset_time", "timed_fix"]]).all().all()
    assert selected.timed_fix.lt(selected.intended_fix).all()
    eligible["pre_onset_candidate"] = eligible.selected_abort & eligible.abort_from_LED_onset.lt(0)
    eligible["pre_onset_bin"] = -1
    pre = eligible.pre_onset_candidate
    assert eligible.loc[pre, "abort_from_LED_onset"].ge(-2).all()
    eligible.loc[pre, "pre_onset_bin"] = np.searchsorted(
        pre_bins, eligible.loc[pre, "abort_from_LED_onset"], side="right"
    ) - 1
    assert eligible.loc[pre, "pre_onset_bin"].between(0, n_pre_bins - 1).all()
    eligible["retained"] = True
    rng = np.random.default_rng(SEED)
    for animal in ANIMALS:
        group = eligible.loc[eligible.animal.eq(animal)]
        denominators = group.groupby("LED_trial").size().reindex([0, 1]).to_numpy(int)
        available = np.stack([
            np.histogram(group.loc[group.LED_trial.eq(led) & group.pre_onset_candidate, "abort_from_LED_onset"], pre_bins)[0]
            for led in (0, 1)
        ])
        quotas, audit = optimize_pre_onset_counts(available, denominators)
        optimizer_records.append({"session_type": st, "animal": animal, **audit})
        for led in (0, 1):
            for b in range(n_pre_bins):
                pool = group.index[group.LED_trial.eq(led) & group.pre_onset_candidate & group.pre_onset_bin.eq(b)].to_numpy()
                pool.sort()
                chosen = rng.choice(pool, size=quotas[led, b], replace=False)
                assert len(np.unique(chosen)) == len(chosen)
                eligible.loc[pool, "retained"] = False
                eligible.loc[chosen, "retained"] = True
                quota_records.append({"session_type": st, "animal": animal, "LED_trial": led,
                                      "bin_left_s": pre_bins[b], "bin_right_s": pre_bins[b + 1],
                                      "available": len(pool), "retained": len(chosen)})
        print(f"s{st}, animal {animal}: optimum certified, {audit['removed']} pre-onset rows removed", flush=True)
    removed, kept = eligible.loc[~eligible.retained], eligible.loc[eligible.retained]
    assert eligible.loc[~eligible.pre_onset_candidate, "retained"].all()
    assert removed.pre_onset_candidate.all() and removed.selected_abort.all()
    assert eligible.valid_valid_valid.sum() == kept.valid_valid_valid.sum()
    assert (eligible.selected_abort & eligible.abort_from_LED_onset.ge(0)).sum() == (
        kept.selected_abort & kept.abort_from_LED_onset.ge(0)).sum()
    assert kept.index.is_unique and not kept.duplicated(keys).any()
    # Explicitly distinguish exclusion by the new denominator from matching removal.
    data["new_matched_retained"] = False
    data.loc[eligible.index, "new_matched_retained"] = eligible.retained
    data["removed_by_new_matching"] = data.new_eligible & ~data.new_matched_retained
    data["excluded_by_new_condition"] = data.old_eligible & ~data.new_eligible
    manifests[st] = data.loc[data.old_eligible].copy()
    retention_records.append({"session_type": st, "original_selected": int(eligible.selected_abort.sum()),
                              "retained_selected": int(kept.selected_abort.sum()), "removed": len(removed),
                              "old_denominator": int(data.old_eligible.sum()),
                              "new_denominator": len(eligible), "new_matched_denominator": len(kept)})

    # Each animal's unchanged numerator is now divided by its conditional N.
    case, ci = f"s{st}_isolated", cases.index(f"s{st}_isolated")
    for ai, animal in enumerate(ANIMALS):
        for led in (0, 1):
            group = eligible.loc[eligible.animal.eq(animal) & eligible.LED_trial.eq(led)]
            original_n = int(saved.loc[saved.animal.eq(animal) & saved.LED_trial.eq(led)].shape[0])
            for mi, metric in enumerate(METRICS):
                matched = mi == 3
                current = group.loc[group.retained] if matched else group
                selection = current.loc[current.selected_abort]
                bins = timing_bins if mi < 2 else abort_bins
                width = TIMING_BIN_WIDTH_S if mi < 2 else ABORT_BIN_WIDTH_S
                values = selection[metric if mi < 2 else "abort_from_LED_onset"].to_numpy()
                values = values[np.isfinite(values)]
                counts = np.histogram(values, bins=bins)[0]
                assert counts.sum() == len(values)
                normalizer = len(values) if mi < 2 else len(current)
                heights = counts / (normalizer * width)
                assert np.isclose(heights @ np.diff(bins), len(values) / normalizer, atol=1e-12, rtol=0)
                if mi < 2:
                    np.testing.assert_array_equal(heights, old_arrays[metric][ci, ai, led])
                elif mi == 2:
                    np.testing.assert_array_equal(counts / (original_n * width), old_arrays[metric][ci, ai, led])
                    np.testing.assert_allclose(heights, old_arrays[metric][ci, ai, led] * original_n / len(group), atol=1e-14)
                arrays[metric][ci, ai, led] = heights
                mask = panels.case.eq(case) & panels.row.eq(str(animal)) & panels.LED_trial.eq(led) & panels.metric.eq(metric)
                assert mask.sum() == 1
                record = {
                    "original_selected_aborts": int(group.selected_abort.sum()), "selected_aborts": len(selection),
                    "finite_aligned_aborts": int(np.isfinite(selection.abort_from_LED_onset).sum()),
                    "missing_aligned_time": int((~np.isfinite(selection.abort_from_LED_onset)).sum()),
                    "plotted_observations": len(values), "original_denominator": len(group),
                    "eligible_denominator": len(current), "removed": len(group) - len(current),
                    "histogram_normalizer": normalizer, "full_histogram_area": float(heights @ np.diff(bins)),
                    "expected_area": len(values) / normalizer, "denominator_definition": NEW_DENOMINATOR,
                }
                for field, value in record.items():
                    panels.loc[mask, field] = value
    for led in (0, 1):
        for mi, metric in enumerate(METRICS):
            animal_curves = arrays[metric][ci, :6, led]
            mean = animal_curves.mean(axis=0)
            sem = animal_curves.std(axis=0, ddof=1) / np.sqrt(6)
            np.testing.assert_allclose(sem, np.sqrt(((animal_curves - mean) ** 2).sum(axis=0) / 30), atol=1e-14)
            arrays[metric][ci, 6, led], arrays[f"{metric}_sem"][ci, 6, led] = mean, sem
            animal_mask = panels.case.eq(case) & ~panels.row.eq("equal_animal_mean") & panels.LED_trial.eq(led) & panels.metric.eq(metric)
            mean_mask = panels.case.eq(case) & panels.row.eq("equal_animal_mean") & panels.LED_trial.eq(led) & panels.metric.eq(metric)
            animal_audit = panels.loc[animal_mask]
            for field in ("original_selected_aborts", "selected_aborts", "finite_aligned_aborts", "missing_aligned_time",
                          "plotted_observations", "original_denominator", "eligible_denominator", "removed"):
                panels.loc[mean_mask, field] = animal_audit[field].sum()
            area = float(mean @ np.diff(timing_bins if mi < 2 else abort_bins))
            assert np.isclose(area, animal_audit.expected_area.mean(), atol=1e-12, rtol=0)
            panels.loc[mean_mask, "full_histogram_area"] = area
            panels.loc[mean_mask, "expected_area"] = animal_audit.expected_area.mean()
            panels.loc[mean_mask, "denominator_definition"] = NEW_DENOMINATOR
    for ri, row in enumerate(rows):
        for metric in METRICS[2:]:
            off, on = arrays[metric][ci, ri, :, :n_pre_bins]
            max_difference = float(abs(off - on).max())
            fraction_difference = float(abs((off - on).sum() * ABORT_BIN_WIDTH_S))
            mask = panels.case.eq(case) & panels.row.eq(row) & panels.metric.eq(metric)
            panels.loc[mask, "max_pre_bin_rate_difference_s_inv"] = max_difference
            panels.loc[mask, "pre_onset_fraction_difference"] = fraction_difference
            if metric == METRICS[3]:
                assert max_difference <= MAX_BIN_RATE_DIFFERENCE + 1e-12
                assert fraction_difference <= MAX_PRE_ONSET_FRACTION_DIFFERENCE + 1e-12


# %%
############ Save new artifacts only, with an explicit old/new normalization audit ############
for metric in METRICS:
    for field in (metric, f"{metric}_sem"):
        np.testing.assert_array_equal(arrays[field][:4], old_arrays[field][:4])
        if metric in METRICS[:2]:
            np.testing.assert_array_equal(arrays[field], old_arrays[field])
        assert np.isfinite(arrays[field]).all() and (arrays[field] >= 0).all()
pd.testing.assert_frame_equal(panels.loc[~panels.selection.eq("isolated")], old_panels.loc[~old_panels.selection.eq("isolated")])
comparison = panels.loc[panels.selection.eq("isolated") & panels.metric.isin(METRICS[2:])].merge(
    old_panels.loc[old_panels.selection.eq("isolated") & old_panels.metric.isin(METRICS[2:])],
    on=["case", "row", "LED_trial", "metric"], suffixes=("_new", "_old"), validate="one_to_one",
)
comparison = comparison[["case", "row", "LED_trial", "metric",
                         "finite_aligned_aborts_old", "eligible_denominator_old", "full_histogram_area_old",
                         "finite_aligned_aborts_new", "eligible_denominator_new", "full_histogram_area_new",
                         "max_pre_bin_rate_difference_s_inv_old", "max_pre_bin_rate_difference_s_inv_new"]]
comparison["area_ratio_new_to_old"] = comparison.full_histogram_area_new / comparison.full_histogram_area_old
comparison.to_csv(f"{OUTPUT_STEM}_old_vs_new_audit.csv", index=False)
panels.to_csv(f"{OUTPUT_STEM}_count_area_audit.csv", index=False)
pd.DataFrame(optimizer_records).to_csv(f"{OUTPUT_STEM}_optimizer_diagnostics.csv", index=False)
pd.DataFrame(quota_records).to_csv(f"{OUTPUT_STEM}_bin_quotas.csv", index=False)
pd.DataFrame(retention_records).to_csv(f"{OUTPUT_STEM}_retention_audit.csv", index=False)
manifest_fields = ["source_csv_row_index", *keys, "LED_trial", "abort_event", "success", "intended_fix",
                   "LED_onset_time", "timed_fix", "abort_from_LED_onset", "previous_trial", "next_trial",
                   "previous_success", "next_success", "previous_source_csv_row_index", "next_source_csv_row_index",
                   "selected_abort", "valid_valid_valid", "both_neighbors_valid", "old_eligible", "new_eligible",
                   "old_matched_retained", "new_matched_retained", "removed_by_new_matching", "excluded_by_new_condition"]
for st, data in manifests.items():
    manifest = data[manifest_fields].copy()
    manifest.insert(0, "source_csv", f"raw_data/LED7_s{st}.csv")
    manifest.to_csv(f"{OUTPUT_STEM}_s{st}_manifest.csv", index=False)
np.savez_compressed(f"{OUTPUT_STEM}_curves.npz", **arrays)


# %%
############ Shared drawing block for the updated figure and the direct comparison ############
def draw_panel(ax, payload, audit, case, metric):
    """Draw the equal-animal histogram, SEM and summed-count annotations."""
    ci = cases.index(case)
    is_rate = metric in METRICS[2:]
    bins = abort_bins if is_rate else timing_bins
    for led in (0, 1):
        mean, sem = payload[metric][ci, 6, led], payload[f"{metric}_sem"][ci, 6, led]
        ax.fill_between(bins, np.r_[mean - sem, (mean - sem)[-1]], np.r_[mean + sem, (mean + sem)[-1]],
                        step="post", color=COLORS[led], alpha=0.16, linewidth=0)
        ax.stairs(mean, bins, color=COLORS[led], linewidth=1.5)
        record = audit.loc[audit.case.eq(case) & audit.row.eq("equal_animal_mean")
                           & audit.LED_trial.eq(led) & audit.metric.eq(metric)].iloc[0]
        label = "OFF" if led == 0 else "ON"
        note = (f"{label} n={int(record.finite_aligned_aborts):,}, N={int(record.eligible_denominator):,}; area={record.full_histogram_area:.3f}"
                if is_rate else f"{label} n={int(record.selected_aborts):,}")
        ax.text(0.98, 0.97 - 0.075 * led, note, transform=ax.transAxes, ha="right", va="top",
                color=COLORS[led], fontsize=8)
    ax.set_xlim(*(ABORT_DISPLAY_S if is_rate else (0, 2.2)))
    if is_rate:
        ax.axvline(0, color="0.45", linestyle=":", linewidth=1)
    ax.grid(axis="y", color="0.91", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)


# %%
############ Full replacement-style figure, but saved under a new filename ############
fig, axes = plt.subplots(6, 4, figsize=(20, 19), sharex="col")
titles = ("intended_fix", "LED_onset_time", "Abort rate aligned to LED onset", "Pre-onset matched abort rate")
visible = (abort_bins[:-1] < ABORT_DISPLAY_S[1]) & (abort_bins[1:] > ABORT_DISPLAY_S[0])
rate_upper = 1.35 * max(float((arrays[m][ci, 6, led] + arrays[f"{m}_sem"][ci, 6, led])[visible].max())
                        for m in METRICS[2:] for ci in range(2, 6) for led in (0, 1))
timing_uppers = [1.35 * max(float((arrays[m][ci, 6, led] + arrays[f"{m}_sem"][ci, 6, led]).max())
                           for ci in range(6) for led in (0, 1)) for m in METRICS[:2]]
for ri, case in enumerate(cases):
    st, selection = int(case[1]), case[3:]
    row_label = {"first": "First abort in a run", "after_abort": "Abort → abort",
                 "isolated": "Valid → abort → valid\nGiven valid neighbors"}[selection]
    for col, metric in enumerate(METRICS):
        ax = axes[ri, col]
        draw_panel(ax, arrays, panels, case, metric)
        ax.set_ylim(0, timing_uppers[col] if col < 2 else (0.2 if selection == "first" else rate_upper))
        if col == 0:
            ax.set_ylabel(f"Session type {st}\n{row_label}\nDensity (s⁻¹)", fontsize=10)
        elif col == 2:
            ax.set_ylabel("First-abort rate (s⁻¹)" if selection == "first" else "Conditional abort rate (s⁻¹)", fontsize=10)
        if ri == 0:
            ax.set_title(titles[col], fontsize=12, pad=12)
        if ri == 5:
            ax.set_xlabel("Time from scheduled LED onset (s)" if col >= 2 else "Time from fixation onset (s)", fontsize=10)
        if col == 3:
            record = panels.loc[panels.case.eq(case) & panels.row.eq("equal_animal_mean")
                                & panels.metric.eq(metric) & panels.LED_trial.eq(0)].iloc[0]
            ax.text(0.02, 0.76, f"Pre: max |Δrate|={record.max_pre_bin_rate_difference_s_inv:.4f} s⁻¹\n"
                    f"|Δfraction|={100 * record.pre_onset_fraction_difference:.3f} pp",
                    transform=ax.transAxes, va="top", fontsize=8)
fig.suptitle("LED7 sequence-abort timing: isolated aborts conditioned on valid neighbors", fontsize=18, y=0.995)
legend_handles = [Line2D([0], [0], color=COLORS[led], lw=2, label=f"Current trial LED {'OFF' if led == 0 else 'ON'}") for led in (0, 1)]
fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.974), ncol=2, frameon=False, fontsize=12)
fig.text(0.5, 0.02,
         "Equal-animal mean ± SEM: animals 92, 93, 98, 99, 100, 103. First four rows unchanged; only isolated-abort denominators/matching changed.\n"
         "A = event-3 abort, V = success ±1. Isolated denominator: V→A→V + V→V→V. Abort→abort denominator: A→A + A→V. First-in-run denominator: all A or V.\n"
         "Timing histograms: 40 ms, unit area. Abort bins: 20 ms; full areas over −2 to +2 s. n and N are sums; areas average six animal-level n/N fractions.\n"
         "Isolated matching rerun using the conditional denominators; classifications frozen before removal. Valid and post-onset trials retained.\n"
         "OFF onset is scheduled/counterfactual. Conditioning includes the next outcome; pre-onset matching uses outcomes. Both are descriptive, not causal evidence.",
         ha="center", fontsize=9, linespacing=1.45)
fig.tight_layout(rect=(0.005, 0.10, 0.995, 0.945), h_pad=1.5, w_pad=1.6)
full_figure_path = Path(f"{OUTPUT_STEM}_6x4.png")
fig.savefig(full_figure_path, dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)


# %%
############ Focused old/new comparison: identical numerators before matching ############
fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=True, sharey=True)
comparisons = ((old_arrays, old_panels, METRICS[2]), (arrays, panels, METRICS[2]),
               (old_arrays, old_panels, METRICS[3]), (arrays, panels, METRICS[3]))
titles = ("Old: all A or V trials", "New: given valid neighbors",
          "Old: pre-onset matched", "New: conditional + re-matched")
upper = 1.35 * max(float((payload[m][ci, 6, led] + payload[f"{m}_sem"][ci, 6, led])[visible].max())
                  for payload, _, m in comparisons for ci in (4, 5) for led in (0, 1))
for ri, st in enumerate(SESSION_TYPES):
    for col, (payload, audit, metric) in enumerate(comparisons):
        ax = axes[ri, col]
        draw_panel(ax, payload, audit, f"s{st}_isolated", metric)
        ax.set_ylim(0, upper)
        if ri == 0:
            ax.set_title(titles[col], fontsize=12, pad=12)
        if col == 0:
            ax.set_ylabel(f"Session type {st}\nAbort rate (s⁻¹)", fontsize=12)
        if ri == 1:
            ax.set_xlabel("Time from scheduled LED onset (s)", fontsize=10)
fig.suptitle("Valid → abort → valid: old versus valid-neighbor-conditioned normalization", fontsize=18, y=0.99)
fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=2, frameon=False, fontsize=12)
fig.text(0.5, 0.024,
         "Six animals, equal-animal mean ± SEM. A = event-3 abort; V = success ±1. Current LED OFF/ON is unchanged.\n"
         "Columns 1–2: same isolated aborts, old N = all A or V, new N = V→A→V + V→V→V. Columns 3–4: matching optimized separately for each denominator.\n"
         "20 ms bins; n/N annotations are summed counts, but full areas are means of animal fractions over −2 to +2 s (not just the displayed window).\n"
         "Conditioning on the next outcome and enforced pre-onset overlap are descriptive, not causal evidence. Original files are preserved.",
         ha="center", fontsize=9, linespacing=1.5)
fig.tight_layout(rect=(0.005, 0.15, 0.995, 0.89), h_pad=1.8, w_pad=1.6)
comparison_figure_path = Path(f"{OUTPUT_STEM}_old_vs_new_2x4.png")
fig.savefig(comparison_figure_path, dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)

for path in protected_paths:
    with path.open("rb") as handle:
        assert hashlib.file_digest(handle, "sha256").hexdigest() == input_hashes[str(path.relative_to(REPO))]
summary = {
    "source_script": str(Path(__file__).relative_to(REPO)), "input_sha256": input_hashes,
    "source_figures_preserved": True, "seed_per_session_type": SEED,
    "numpy_version": np.__version__, "scipy_version": scipy.__version__, "animals": ANIMALS,
    "new_isolated_denominator": NEW_DENOMINATOR,
    "unchanged": "Isolated numerator identities, both timing panels, and all curves/SEMs in first-in-run and abort-after-abort rows.",
    "averaging": "Mean of six individually normalized animal curves; sample SD/sqrt(6) SEM.",
    "adjacency": "Actual trial -1 and +1 in the same animal/session; classification frozen before matching.",
    "matching": "Fresh per-animal MILP with new denominator; maximum retention only under the existing conservative linear rule.",
    "all_solvers_certified_optimal": all(record["certified_optimal"] for record in optimizer_records),
    "max_bin_rate_difference_s_inv": MAX_BIN_RATE_DIFFERENCE,
    "max_pre_onset_fraction_difference": MAX_PRE_ONSET_FRACTION_DIFFERENCE,
    "retention": retention_records,
    "caveat": "Conditional on both neighboring outcomes. Matching uses observed abort times; descriptive, not causal evidence.",
    "figures": [str(path.relative_to(REPO)) for path in (full_figure_path, comparison_figure_path)],
}
Path(f"{OUTPUT_STEM}_run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(pd.DataFrame(retention_records).to_string(index=False))
print(comparison.loc[comparison.row.eq("equal_animal_mean")].to_string(index=False))
print(f"Saved {full_figure_path}\nSaved {comparison_figure_path}")
print("Passed: exact neighbors, unchanged numerators/timing panels/other rows, certified matching, areas, SEMs, old-file hashes.")
