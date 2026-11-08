# %%
"""Split LED7 abort-after-abort curves by the preceding abort's LED flag."""

from pathlib import Path
import os
import time

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
############ Parameters (edit here) ############
SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
OUTPUT_PREFIX = SCRIPT_DIR / "led7_s7_s8_previous_abort_led"
TOP_SOURCES = {
    "all_eligible": SCRIPT_DIR / "led7_s7_s8_sequence_abort_all_eligible_denominator",
    "conditional": SCRIPT_DIR / "led7_s7_s8_sequence_abort_valid_neighbor_conditioning",
}
ANIMALS = (92, 93, 98, 99, 100, 103)
SESSION_TYPES = (7, 8)
CASE_DEFS = (
    (7, 0, "s7_previous_off"),
    (8, 0, "s8_previous_off"),
    (7, 1, "s7_previous_on"),
    (8, 1, "s8_previous_on"),
)
METRICS = (
    "intended_fix", "LED_onset_time", "abort_from_LED_onset",
    "matched_abort_from_LED_onset",
)
EXPECTED_PAIR_COUNTS = {
    7: {(0, 0): (1528, 8630), (0, 1): (208, 975), (1, 1): (1361, 6013)},
    8: {(0, 0): (963, 6181), (0, 1): (527, 2590), (1, 1): (1936, 8081)},
}
EXPECTED_ALL_ELIGIBLE = {7: (63525, 30910), 8: (68584, 38486)}
EXPECTED_MISSING_SELECTED_TIME = {(7, 0, 0): 2, (8, 1, 1): 2}
SEED = 20261001
MAX_BIN_RATE_DIFFERENCE = 0.005  # s^-1, separately for each animal.
MAX_PRE_FRACTION_DIFFERENCE = 0.001
SOLVER_TIME_LIMIT_S = 60
TIMING_WIDTH_S = 0.040
ABORT_WIDTH_S = 0.020
RATE_VIEW_S = (-0.4, 0.5)
FIGURE_DPI = 240
COLORS = {0: "tab:blue", 1: "tab:red"}

timing_bins = np.linspace(0, 2.2, 56)
abort_bins = np.linspace(-2, 2, 201)
pre_bins = np.linspace(-2, 0, 101)
n_pre_bins = len(pre_bins) - 1
np.testing.assert_allclose(np.diff(timing_bins), TIMING_WIDTH_S, rtol=0, atol=1e-14)
np.testing.assert_allclose(np.diff(abort_bins), ABORT_WIDTH_S, rtol=0, atol=1e-14)
np.testing.assert_array_equal(abort_bins[: n_pre_bins + 1], pre_bins)
case_names = ("s7_after_abort", "s8_after_abort", *(case for _, _, case in CASE_DEFS))
row_names = (*map(str, ANIMALS), "equal_animal_mean")


# %%
############ Classify actual adjacent trials in the intact CSV stream ############
source_by_session = {}
for st in SESSION_TYPES:
    source = pd.read_csv(REPO / f"raw_data/LED7_s{st}.csv", float_precision="round_trip")
    assert source.shape == ({7: 138440, 8: 148428}[st], 52)
    assert source.training_level.eq(16).all() and source.session_type.eq(st).all()
    assert (source.repeat_trial.isin((0, 2)) | source.repeat_trial.isna()).all()
    assert source.LED_trial.isin((0, 1)).all()
    source["source_csv_row_index"] = np.arange(len(source))
    stream = source.loc[source.animal.isin(ANIMALS)].sort_values(
        ["animal", "session", "trial"], kind="stable"
    ).copy()
    assert set(stream.animal.astype(int).unique()) == set(ANIMALS)
    assert not stream.duplicated(["animal", "session", "trial"]).any()
    previous = stream.groupby(["animal", "session"], sort=False)[
        ["trial", "abort_event", "LED_trial"]
    ].shift(1)
    stream["previous_is_abort"] = (
        previous.trial.eq(stream.trial - 1) & previous.abort_event.eq(3)
    )
    stream["previous_LED_trial"] = previous.LED_trial
    stream["is_abort"] = stream.abort_event.eq(3)
    stream["is_valid"] = stream.success.isin((-1, 1))
    assert not (stream.is_abort & stream.is_valid).any()
    stream["abort_from_LED_onset"] = stream.timed_fix - stream.LED_onset_time
    stream = stream.set_index("source_csv_row_index", drop=False)
    eligible = stream.loc[stream.is_abort | stream.is_valid].copy()
    assert tuple(eligible.groupby("LED_trial").size()) == EXPECTED_ALL_ELIGIBLE[st]

    pair = eligible.loc[eligible.previous_is_abort]
    assert not ((pair.previous_LED_trial.eq(1)) & (pair.LED_trial.eq(0))).any()
    for (previous_led, current_led), (expected_abort, expected_n) in EXPECTED_PAIR_COUNTS[st].items():
        group = pair.loc[
            pair.previous_LED_trial.eq(previous_led) & pair.LED_trial.eq(current_led)
        ]
        assert (int(group.is_abort.sum()), len(group)) == (expected_abort, expected_n)
    assert pair.is_abort.sum() == {7: 3097, 8: 3426}[st]
    assert pair.loc[pair.LED_trial.eq(0), "is_abort"].sum() == {
        7: 1528, 8: 963
    }[st]
    assert pair.loc[pair.LED_trial.eq(1), "is_abort"].sum() == {
        7: 1569, 8: 2463
    }[st]
    source_by_session[st] = eligible
    print(f"s{st}: classified {len(pair):,} eligible trials after an abort", flush=True)


# %%
############ Established certified integer matching rule ############
def optimize_pre_onset_counts(available, denominators):
    """Maximize retained pre-onset aborts under the previous linear rule."""
    mandatory = denominators - available.sum(axis=1)
    assert (mandatory > 0).all()
    ratio = mandatory[0] / mandatory[1]
    u_max = float((available / mandatory[:, None]).max())
    epsilon = ABORT_WIDTH_S * MAX_BIN_RATE_DIFFERENCE - u_max * MAX_PRE_FRACTION_DIFFERENCE
    assert epsilon > 0
    matrix = vstack(
        [
            hstack([eye(n_pre_bins), -ratio * eye(n_pre_bins)]),
            csr_matrix(
                np.r_[np.ones(n_pre_bins), -ratio * np.ones(n_pre_bins)][None, :]
            ),
        ],
        format="csr",
    )
    tolerance = np.r_[
        np.full(n_pre_bins, mandatory[0] * epsilon),
        mandatory[0] * MAX_PRE_FRACTION_DIFFERENCE,
    ]
    started = time.perf_counter()
    solution = milp(
        -np.ones(2 * n_pre_bins),
        integrality=np.ones(2 * n_pre_bins),
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
    rates = quotas / (retained_n[:, None] * ABORT_WIDTH_S)
    max_difference = float(np.abs(rates[0] - rates[1]).max())
    fraction_difference = float(abs(quotas[0].sum() / retained_n[0]
                                    - quotas[1].sum() / retained_n[1]))
    assert max_difference <= MAX_BIN_RATE_DIFFERENCE + 1e-12
    assert fraction_difference <= MAX_PRE_FRACTION_DIFFERENCE + 1e-12
    return quotas, {
        "solver": "scipy.optimize.milp / HiGHS",
        "certified_optimal": True,
        "mip_gap": solution.mip_gap,
        "objective": solution.fun,
        "dual_bound": solution.mip_dual_bound,
        "elapsed_s": time.perf_counter() - started,
        "original_OFF": int(denominators[0]),
        "original_ON": int(denominators[1]),
        "retained_OFF": int(retained_n[0]),
        "retained_ON": int(retained_n[1]),
        "removed": int((available - quotas).sum()),
        "max_pre_bin_rate_difference_s_inv": max_difference,
        "pre_onset_fraction_difference": fraction_difference,
    }


# %%
############ Reuse existing top rows and build four preceding-LED rows ############
results = {}
for mode, source_stem in TOP_SOURCES.items():
    with np.load(f"{source_stem}_curves.npz") as payload:
        reference = {key: payload[key].copy() for key in payload.files}
    source_cases = reference["cases"].tolist()
    top_indices = [source_cases.index("s7_after_abort"), source_cases.index("s8_after_abort")]
    assert reference["rows"].tolist() == list(row_names)
    np.testing.assert_array_equal(reference["timing_bins"], timing_bins)
    np.testing.assert_array_equal(reference["abort_bins"], abort_bins)
    original_audit = pd.read_csv(
        f"{source_stem}_count_area_audit.csv", dtype={"row": str}
    )
    top_audit = original_audit.loc[
        original_audit.case.isin(case_names[:2])
    ].copy()
    assert len(top_audit) == 2 * 7 * 2 * 4

    curves = {
        metric: np.full(
            (6, 7, 2, len(timing_bins if metric in METRICS[:2] else abort_bins) - 1),
            np.nan,
        )
        for metric in METRICS
    }
    sems = {metric: np.full_like(curves[metric], np.nan) for metric in METRICS}
    for metric in METRICS:
        curves[metric][:2] = reference[metric][top_indices]
        sems[metric][:2] = reference[f"{metric}_sem"][top_indices]
        np.testing.assert_array_equal(curves[metric][:2], reference[metric][top_indices])
        np.testing.assert_array_equal(sems[metric][:2], reference[f"{metric}_sem"][top_indices])

    records = []
    for _, r in top_audit.iterrows():
        records.append({
            "mode": mode, "case": r.case, "row": str(r.row),
            "previous_LED_trial": np.nan, "LED_trial": int(r.LED_trial),
            "metric": r.metric, "defined": True,
            "selected_aborts": int(r.selected_aborts),
            "plotted_observations": int(r.plotted_observations),
            "eligible_denominator": int(r.eligible_denominator),
            "full_histogram_area": float(r.full_histogram_area),
            "missing_selected_timed_fix": int(
                r.get("missing_selected_timed_fix", r.get("missing_aligned_time"))
            ),
        })

    frames = {}
    optimizer_records, quota_records, manifest_parts = [], [], []
    for case_index, (st, previous_led, case) in enumerate(CASE_DEFS, start=2):
        eligible = source_by_session[st]
        if mode == "conditional":
            frame = eligible.loc[
                eligible.previous_is_abort
                & eligible.previous_LED_trial.eq(previous_led)
            ].copy()
        else:
            frame = eligible.copy()
        frame["selected_abort"] = (
            frame.is_abort & frame.previous_is_abort
            & frame.previous_LED_trial.eq(previous_led)
        )
        frame["pre_onset_candidate"] = (
            frame.selected_abort & frame.abort_from_LED_onset.notna()
            & frame.abort_from_LED_onset.lt(0)
        )
        assert frame.loc[frame.pre_onset_candidate, "abort_from_LED_onset"].ge(-2).all()
        frame["pre_onset_bin"] = -1
        frame.loc[frame.pre_onset_candidate, "pre_onset_bin"] = (
            np.searchsorted(
                pre_bins,
                frame.loc[frame.pre_onset_candidate, "abort_from_LED_onset"],
                side="right",
            ) - 1
        )
        assert frame.loc[frame.pre_onset_candidate, "pre_onset_bin"].between(
            0, n_pre_bins - 1
        ).all()
        frame["matched_retained"] = True

        available_leds = (0, 1) if previous_led == 0 else (1,)
        for led in available_leds:
            selected = frame.loc[frame.LED_trial.eq(led) & frame.selected_abort]
            expected_abort, expected_conditional_n = EXPECTED_PAIR_COUNTS[st][
                (previous_led, led)
            ]
            assert len(selected) == expected_abort
            if mode == "conditional":
                assert len(frame.loc[frame.LED_trial.eq(led)]) == expected_conditional_n
            else:
                assert len(frame.loc[frame.LED_trial.eq(led)]) == (
                    EXPECTED_ALL_ELIGIBLE[st][led]
                )
            finite = selected.loc[selected.timed_fix.notna()]
            assert finite.timed_fix.lt(finite.intended_fix).all()
            missing = int(selected.timed_fix.isna().sum())
            assert missing == EXPECTED_MISSING_SELECTED_TIME.get(
                (st, previous_led, led), 0
            )
        if previous_led == 1:
            assert not frame.loc[
                frame.selected_abort & frame.LED_trial.eq(0)
            ].shape[0]

        if previous_led == 0:
            rng = np.random.default_rng(SEED)  # Reset for each mode and session type.
            for animal in ANIMALS:
                group = frame.loc[frame.animal.eq(animal)]
                denominators = group.groupby("LED_trial").size().reindex(
                    (0, 1)
                ).to_numpy(int)
                available = np.stack([
                    np.histogram(
                        group.loc[
                            group.LED_trial.eq(led) & group.pre_onset_candidate,
                            "abort_from_LED_onset",
                        ],
                        bins=pre_bins,
                    )[0]
                    for led in (0, 1)
                ])
                quotas, diagnostic = optimize_pre_onset_counts(
                    available, denominators
                )
                for led in (0, 1):
                    for b in range(n_pre_bins):
                        pool = group.index[
                            group.LED_trial.eq(led)
                            & group.pre_onset_candidate
                            & group.pre_onset_bin.eq(b)
                        ].to_numpy()
                        pool.sort()
                        assert len(pool) == available[led, b]
                        chosen = rng.choice(
                            pool, size=quotas[led, b], replace=False
                        )
                        frame.loc[pool, "matched_retained"] = False
                        frame.loc[chosen, "matched_retained"] = True
                        assert len(chosen) == quotas[led, b]
                        assert len(np.unique(chosen)) == len(chosen)
                        quota_records.append({
                            "mode": mode, "case": case, "animal": animal,
                            "LED_trial": led, "bin_index": b,
                            "bin_left_s": pre_bins[b],
                            "bin_right_s": pre_bins[b + 1],
                            "available": len(pool), "retained": len(chosen),
                            "removed": len(pool) - len(chosen),
                        })
                optimizer_records.append({
                    "mode": mode, "case": case, "session_type": st,
                    "animal": animal, **diagnostic,
                })
            removed = frame.loc[~frame.matched_retained]
            assert removed.selected_abort.all()
            assert removed.pre_onset_candidate.all()
            assert frame.loc[~frame.pre_onset_candidate, "matched_retained"].all()
            for animal in ANIMALS:
                group = frame.loc[frame.animal.eq(animal)]
                kept = group.loc[group.matched_retained]
                assert int(group.is_valid.sum()) == int(kept.is_valid.sum())
                assert (
                    group.loc[
                        group.selected_abort
                        & group.abort_from_LED_onset.ge(0)
                    ].shape[0]
                    == kept.loc[
                        kept.selected_abort & kept.abort_from_LED_onset.ge(0)
                    ].shape[0]
                )
        else:
            assert frame.matched_retained.all()

        manifest = frame.loc[frame.selected_abort, [
            "source_csv_row_index", "animal", "session", "trial", "session_type",
            "LED_trial", "previous_LED_trial", "abort_event", "success",
            "intended_fix", "LED_onset_time", "timed_fix",
            "abort_from_LED_onset", "pre_onset_candidate", "pre_onset_bin",
            "matched_retained",
        ]].copy()
        manifest.insert(0, "source_csv", f"raw_data/LED7_s{st}.csv")
        manifest.insert(0, "case", case)
        manifest.insert(0, "mode", mode)
        manifest["matching_applicable"] = previous_led == 0
        manifest_parts.append(manifest)
        frames[case] = frame

        for animal_index, animal in enumerate(ANIMALS):
            animal_frame = frame.loc[frame.animal.eq(animal)]
            for led in available_leds:
                original = animal_frame.loc[animal_frame.LED_trial.eq(led)]
                assert len(original) > 0
                for metric in METRICS:
                    if metric == METRICS[3] and previous_led == 1:
                        continue
                    current = (
                        original.loc[original.matched_retained]
                        if metric == METRICS[3] else original
                    )
                    selected = current.loc[current.selected_abort]
                    column = (
                        "abort_from_LED_onset"
                        if metric in METRICS[2:] else metric
                    )
                    values = selected[column].to_numpy(float)
                    values = values[np.isfinite(values)]
                    bins = timing_bins if metric in METRICS[:2] else abort_bins
                    width = TIMING_WIDTH_S if metric in METRICS[:2] else ABORT_WIDTH_S
                    counts = np.histogram(values, bins=bins)[0]
                    assert counts.sum() == len(values)
                    normalizer = len(values) if metric in METRICS[:2] else len(current)
                    assert normalizer > 0
                    heights = counts / (normalizer * width)
                    area = float(heights @ np.diff(bins))
                    assert np.isclose(
                        area, len(values) / normalizer, rtol=0, atol=1e-12
                    )
                    curves[metric][case_index, animal_index, led] = heights
                    records.append({
                        "mode": mode, "case": case, "row": str(animal),
                        "previous_LED_trial": previous_led,
                        "LED_trial": led, "metric": metric, "defined": True,
                        "selected_aborts": len(selected),
                        "plotted_observations": len(values),
                        "eligible_denominator": len(current),
                        "full_histogram_area": area,
                        "missing_selected_timed_fix": int(
                            selected.timed_fix.isna().sum()
                        ),
                    })

        for metric in METRICS:
            if metric == METRICS[3] and previous_led == 1:
                continue
            bins = timing_bins if metric in METRICS[:2] else abort_bins
            for led in available_leds:
                six = curves[metric][case_index, :6, led]
                assert np.isfinite(six).all() and (six >= 0).all()
                mean = six.mean(axis=0)
                sem = six.std(axis=0, ddof=1) / np.sqrt(6)
                curves[metric][case_index, 6, led] = mean
                sems[metric][case_index, 6, led] = sem
                selected_rows = [
                    r for r in records
                    if r["case"] == case and r["row"] != "equal_animal_mean"
                    and r["LED_trial"] == led and r["metric"] == metric
                ]
                assert len(selected_rows) == 6
                area = float(mean @ np.diff(bins))
                assert np.isclose(
                    area,
                    np.mean([r["full_histogram_area"] for r in selected_rows]),
                    rtol=0, atol=1e-12,
                )
                records.append({
                    "mode": mode, "case": case, "row": "equal_animal_mean",
                    "previous_LED_trial": previous_led,
                    "LED_trial": led, "metric": metric, "defined": True,
                    "selected_aborts": sum(
                        r["selected_aborts"] for r in selected_rows
                    ),
                    "plotted_observations": sum(
                        r["plotted_observations"] for r in selected_rows
                    ),
                    "eligible_denominator": sum(
                        r["eligible_denominator"] for r in selected_rows
                    ),
                    "full_histogram_area": area,
                    "missing_selected_timed_fix": sum(
                        r["missing_selected_timed_fix"] for r in selected_rows
                    ),
                })
        print(
            f"{mode} {case}: selected {int(frame.selected_abort.sum()):,}, "
            f"matched selected {int((frame.selected_abort & frame.matched_retained).sum()):,}"
            if previous_led == 0
            else f"{mode} {case}: selected {int(frame.selected_abort.sum()):,}; "
                 "no current LED-OFF group",
            flush=True,
        )

    panels = pd.DataFrame(records)
    assert not panels.duplicated(
        ["case", "row", "LED_trial", "metric"]
    ).any()
    assert len(optimizer_records) == 12
    assert all(r["certified_optimal"] and r["mip_gap"] == 0 for r in optimizer_records)
    assert len(quota_records) == 2 * len(ANIMALS) * 2 * n_pre_bins

    # The new previous-LED groups partition the unchanged top-row original rate.
    for st in SESSION_TYPES:
        top_index = case_names.index(f"s{st}_after_abort")
        off_index = case_names.index(f"s{st}_previous_off")
        on_index = case_names.index(f"s{st}_previous_on")
        for animal_index, animal in enumerate(ANIMALS):
            np.testing.assert_allclose(
                curves[METRICS[2]][top_index, animal_index, 0],
                curves[METRICS[2]][off_index, animal_index, 0],
                rtol=0, atol=1e-14,
            )
            off_on = curves[METRICS[2]][off_index, animal_index, 1]
            on_on = curves[METRICS[2]][on_index, animal_index, 1]
            if mode == "all_eligible":
                reconstructed = off_on + on_on
            else:
                off_n = len(frames[f"s{st}_previous_off"].loc[
                    frames[f"s{st}_previous_off"].animal.eq(animal)
                    & frames[f"s{st}_previous_off"].LED_trial.eq(1)
                ])
                on_n = len(frames[f"s{st}_previous_on"].loc[
                    frames[f"s{st}_previous_on"].animal.eq(animal)
                    & frames[f"s{st}_previous_on"].LED_trial.eq(1)
                ])
                reconstructed = (off_n * off_on + on_n * on_on) / (off_n + on_n)
            np.testing.assert_allclose(
                curves[METRICS[2]][top_index, animal_index, 1],
                reconstructed, rtol=0, atol=1e-14,
            )
        for animal_index in range(6):
            off = curves[METRICS[3]][off_index, animal_index, 0, :n_pre_bins]
            on = curves[METRICS[3]][off_index, animal_index, 1, :n_pre_bins]
            assert np.max(abs(off - on)) <= MAX_BIN_RATE_DIFFERENCE + 1e-12
            assert abs(np.sum(off - on) * ABORT_WIDTH_S) <= MAX_PRE_FRACTION_DIFFERENCE + 1e-12

    # The old s7/s8 top curves and their uncertainty bands are copied exactly.
    for metric in METRICS:
        np.testing.assert_array_equal(curves[metric][:2], reference[metric][top_indices])
        np.testing.assert_array_equal(
            sems[metric][:2], reference[f"{metric}_sem"][top_indices]
        )
    results[mode] = {
        "curves": curves, "sems": sems, "panels": panels,
        "optimizer": pd.DataFrame(optimizer_records),
        "quotas": pd.DataFrame(quota_records),
        "manifest": pd.concat(manifest_parts, ignore_index=True),
    }


# %%
############ Plot the two six-row comparisons and save numerical receipts ############
for mode, result in results.items():
    curves, sems, panels = result["curves"], result["sems"], result["panels"]
    stem = Path(f"{OUTPUT_PREFIX}_{mode}")
    visible = (abort_bins[:-1] < RATE_VIEW_S[1]) & (abort_bins[1:] > RATE_VIEW_S[0])
    timing_limits = []
    for metric in METRICS[:2]:
        values = [
            float(np.nanmax(curves[metric][ci, 6, led]
                            + sems[metric][ci, 6, led]))
            for ci in range(6) for led in (0, 1)
            if np.isfinite(curves[metric][ci, 6, led]).any()
        ]
        timing_limits.append(1.2 * max(values))
    rate_limit = 1.2 * max(
        float(np.nanmax(
            (curves[metric][ci, 6, led] + sems[metric][ci, 6, led])[visible]
        ))
        for ci in range(6) for led in (0, 1) for metric in METRICS[2:]
        if np.isfinite(curves[metric][ci, 6, led]).any()
    )

    fig, axes = plt.subplots(6, 4, figsize=(20, 19.5), sharex="col")
    titles = (
        "intended_fix", "LED_onset_time", "Abort rate aligned to LED onset",
        "Pre-onset matched abort rate",
    )
    for ci, case in enumerate(case_names):
        st = int(case[1])
        if ci < 2:
            row_label = f"Session type {st}\nAbort after abort"
        else:
            previous_led = CASE_DEFS[ci - 2][1]
            row_label = (
                f"Session type {st}\nPrevious abort LED "
                f"{'OFF' if previous_led == 0 else 'ON'}"
            )
        for mi, metric in enumerate(METRICS):
            ax = axes[ci, mi]
            is_rate = mi >= 2
            bins = abort_bins if is_rate else timing_bins
            if ci >= 4 and mi == 3:
                ax.set_facecolor("0.97")
                ax.text(
                    0.5, 0.5,
                    "Not applicable\nNo current LED-OFF trials\nafter an LED-ON abort",
                    ha="center", va="center", transform=ax.transAxes, fontsize=11,
                )
            else:
                present_leds = (0, 1) if ci < 4 else (1,)
                for led in present_leds:
                    mean = curves[metric][ci, 6, led]
                    sem = sems[metric][ci, 6, led]
                    assert np.isfinite(mean).all() and np.isfinite(sem).all()
                    lower = np.maximum(mean - sem, 0)
                    upper = mean + sem
                    ax.fill_between(
                        bins, np.r_[lower, lower[-1]], np.r_[upper, upper[-1]],
                        step="post", color=COLORS[led], alpha=0.16, linewidth=0,
                    )
                    ax.stairs(mean, bins, color=COLORS[led], linewidth=1.5)
                    record = panels.loc[
                        panels.case.eq(case)
                        & panels.row.eq("equal_animal_mean")
                        & panels.LED_trial.eq(led)
                        & panels.metric.eq(metric)
                    ]
                    assert len(record) == 1
                    record = record.iloc[0]
                    label = "OFF" if led == 0 else "ON"
                    note = (
                        f"{label} n={int(record.plotted_observations):,}, "
                        f"N={int(record.eligible_denominator):,}; "
                        f"mean area={record.full_histogram_area:.3f}"
                        if is_rate else
                        f"{label} n={int(record.selected_aborts):,}"
                    )
                    ax.text(
                        0.98, 0.96 - 0.085 * led, note, transform=ax.transAxes,
                        ha="right", va="top", color=COLORS[led], fontsize=8,
                    )
            ax.set_xlim(*(RATE_VIEW_S if is_rate else (0, 2.2)))
            ax.set_ylim(0, rate_limit if is_rate else timing_limits[mi])
            if is_rate:
                ax.axvline(0, color="0.45", linestyle=":", linewidth=1)
            ax.grid(axis="y", color="0.91", linewidth=0.5)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=9)
            if ci == 0:
                ax.set_title(titles[mi], fontsize=12, pad=12)
            if mi == 0:
                ax.set_ylabel(f"{row_label}\nDensity (s⁻¹)", fontsize=10)
            elif mi == 2:
                ax.set_ylabel(
                    "Conditional abort rate (s⁻¹)"
                    if mode == "conditional" else
                    "Abort rate per eligible trial (s⁻¹)",
                    fontsize=10,
                )
            if ci == 5:
                ax.set_xlabel(
                    "Time from scheduled LED onset (s)"
                    if is_rate else "Time from fixation onset (s)", fontsize=10,
                )
            if mi == 3 and ci in (2, 3):
                matching = result["optimizer"].loc[
                    result["optimizer"].case.eq(case)
                ]
                assert len(matching) == 6
                ax.text(
                    0.02, 0.76,
                    f"Pre: max |Δrate|={matching.max_pre_bin_rate_difference_s_inv.max():.4f} s⁻¹\n"
                    f"|Δfraction|={100 * matching.pre_onset_fraction_difference.max():.3f} pp",
                    transform=ax.transAxes, va="top", fontsize=8,
                )

    fig.suptitle(
        "LED7 abort-after-abort timing by preceding LED: "
        + ("all eligible denominator" if mode == "all_eligible"
           else "conditional denominator"),
        fontsize=17, y=0.995,
    )
    handles = [
        Line2D(
            [0], [0], color=COLORS[led], lw=2,
            label=f"Current trial LED {'OFF' if led == 0 else 'ON'}",
        )
        for led in (0, 1)
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.967),
        ncol=2, frameon=False, fontsize=11,
    )
    denominator_note = (
        "All-eligible N: all event-3 or valid current trials in that LED group."
        if mode == "all_eligible" else
        "Conditional N: current event-3 or valid trials after an abort with the specified previous LED flag."
    )
    fig.text(
        0.5, 0.013,
        "Six equal-animal mean ± SEM curves: 92, 93, 98, 99, 100, 103. "
        "Adjacent trial IDs must differ by one in the same session. "
        "A = event-3 abort; V = success ±1.\n"
        + denominator_note
        + " Timing: 40 ms unit-area bins; rate: 20 ms bins. "
        "Full rate area is mean animal fraction, not pooled n/N.\n"
        "Only selected pre-onset aborts are removed for matched preceding-OFF rows. "
        "LED OFF onset is scheduled/counterfactual. Outcome-based matching is descriptive.",
        ha="center", fontsize=8.5, linespacing=1.35,
    )
    fig.tight_layout(rect=(0.005, 0.08, 0.995, 0.947), h_pad=1.55, w_pad=1.5)
    figure_path = Path(f"{stem}_6x4.png")
    fig.savefig(figure_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)

    panels.to_csv(f"{stem}_count_area_audit.csv", index=False)
    result["optimizer"].to_csv(f"{stem}_optimizer_audit.csv", index=False)
    result["quotas"].to_csv(f"{stem}_bin_quotas.csv", index=False)
    result["manifest"].to_csv(f"{stem}_selection_manifest.csv", index=False)
    np.savez_compressed(
        f"{stem}_curves.npz",
        cases=np.array(case_names), rows=np.array(row_names),
        timing_bins=timing_bins, abort_bins=abort_bins,
        **curves,
        **{f"{metric}_sem": sems[metric] for metric in METRICS},
    )
    print(f"Saved {figure_path}", flush=True)

print("Validated new pair counts, top-row equality, mass, means, and certified matching.")
