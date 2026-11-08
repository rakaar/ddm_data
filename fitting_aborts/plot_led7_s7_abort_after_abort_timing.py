# %%
"""Timing of aborts after aborts, or isolated valid→abort→valid trials.

Trial IDs establish adjacency within animal/session. The conditional denominator
contains current event-3 or success ±1 trials immediately following an event-3
abort. For isolated aborts, the denominator is all event-3 or success ±1 trials.
LED groups refer to the current trial, irrespective of neighboring LED flags.
"""

from pathlib import Path
import os


# %%
############ Editable parameters ############
SCRIPT_DIR = Path(__file__).resolve().parent
SESSION_TYPE = int(os.environ.get("LED7_SESSION_TYPE", "7"))  # 7 or 8
assert SESSION_TYPE in (7, 8)
SOURCE_PATH = SCRIPT_DIR.parent / "raw_data" / f"LED7_s{SESSION_TYPE}.csv"
ANIMALS = (92, 93, 98, 99, 100, 103)
SELECTION = os.environ.get("LED7_ABORT_SELECTION", "after_abort")  # after_abort or isolated
TIMING_BIN_WIDTH_S = 0.040
ABORT_BIN_WIDTH_S = 0.020
TIMING_RANGE_S = (0.0, 2.2)
ABORT_RANGE_S = (-2.0, 2.0)
ABORT_DISPLAY_S = (-0.4, 0.5)
FIGURE_DPI = 220
LED_COLORS = {0: "tab:blue", 1: "tab:red"}
METRICS = ("intended_fix", "LED_onset_time", "abort_from_LED_onset")
COLUMN_TITLES = ("intended_fix", "LED_onset_time", "Abort rate aligned to LED onset")
assert SELECTION in ("after_abort", "isolated")
OUTPUT_STEM = f"led7_s{SESSION_TYPE}_abort_after_abort_timing" if SELECTION == "after_abort" else f"led7_s{SESSION_TYPE}_isolated_abort_timing"
FIGURE_PATH = SCRIPT_DIR / f"{OUTPUT_STEM}_7x3.png"
# (coded selected aborts, finite aligned aborts, eligible current trials)
EXPECTED_COUNTS = {
    (7, "after_abort"): {0: (1_528, 1_526, 8_630), 1: (1_569, 1_569, 6_988)},
    (7, "isolated"): {0: (6_498, 6_498, 63_525), 1: (3_365, 3_365, 30_910)},
    (8, "after_abort"): {0: (963, 963, 6_181), 1: (2_463, 2_461, 10_671)},
    (8, "isolated"): {0: (6_406, 6_406, 68_584), 1: (4_105, 4_105, 38_486)},
}[SESSION_TYPE, SELECTION]
SELECTION_TITLE = "aborts following aborts" if SELECTION == "after_abort" else "isolated valid–abort–valid trials"
DENOMINATOR_DESCRIPTION = ("all current event-3 or success ±1 trials immediately following an event-3 abort"
                           if SELECTION == "after_abort" else "all event-3 or success ±1 trials in the current LED condition")

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

timing_bins = np.linspace(*TIMING_RANGE_S, round((TIMING_RANGE_S[1] - TIMING_RANGE_S[0]) / TIMING_BIN_WIDTH_S) + 1)
abort_bins = np.linspace(*ABORT_RANGE_S, round((ABORT_RANGE_S[1] - ABORT_RANGE_S[0]) / ABORT_BIN_WIDTH_S) + 1)
bins_by_metric = {"intended_fix": timing_bins, "LED_onset_time": timing_bins,
                  "abort_from_LED_onset": abort_bins}
width_by_metric = {"intended_fix": TIMING_BIN_WIDTH_S, "LED_onset_time": TIMING_BIN_WIDTH_S,
                   "abort_from_LED_onset": ABORT_BIN_WIDTH_S}
for metric, bins in bins_by_metric.items():
    np.testing.assert_allclose(np.diff(bins), width_by_metric[metric], rtol=0, atol=1e-14)


# %%
############ Select by actual previous trial ID, retaining the source stream ############
source = pd.read_csv(SOURCE_PATH, float_precision="round_trip")
assert source.shape == ({7: 138_440, 8: 148_428}[SESSION_TYPE], 52)
assert source["training_level"].eq(16).all() and source["session_type"].eq(SESSION_TYPE).all()
assert (source["repeat_trial"].isin([0, 2]) | source["repeat_trial"].isna()).all()
assert source["LED_trial"].isin([0, 1]).all()
keys = ["animal", "session", "trial"]
assert not source.duplicated(keys).any()
source["source_csv_row_index"] = np.arange(len(source))
data = source.loc[source["animal"].isin(ANIMALS)].sort_values(keys, kind="stable").reset_index(drop=True)
assert len(data) == {7: 100_251, 8: 113_941}[SESSION_TYPE] and tuple(sorted(data["animal"].unique())) == ANIMALS
for column in [*keys, "LED_trial"]:
    assert data[column].notna().all() and data[column].eq(np.floor(data[column])).all()
    data[column] = data[column].astype(int)

grouped = data.groupby(["animal", "session"], sort=False)
neighbor_columns = ["trial", "abort_event", "success", "LED_trial", "source_csv_row_index"]
previous = grouped[neighbor_columns].shift(1)
following = grouped[neighbor_columns].shift(-1)
adjacent = previous["trial"].eq(data["trial"] - 1)
after_abort = adjacent & previous["abort_event"].eq(3)
is_abort = data["abort_event"].eq(3)
is_valid = data["success"].isin([-1, 1])
assert not (is_abort & is_valid).any()
isolated = (is_abort & adjacent & following["trial"].eq(data["trial"] + 1)
            & previous["success"].isin([-1, 1]) & following["success"].isin([-1, 1]))
data["eligible_denominator_trial"] = after_abort & (is_abort | is_valid) if SELECTION == "after_abort" else is_abort | is_valid
data["selected_abort"] = after_abort & is_abort if SELECTION == "after_abort" else isolated
assert (is_abort | is_valid).sum() == {7: 94_435, 8: 107_070}[SESSION_TYPE]

# Verify shifted neighbors against a separate lookup by exact trial key.
previous_keys = pd.MultiIndex.from_arrays(
    [data["animal"], data["session"], data["trial"] - 1], names=keys,
)
keyed_previous = data.set_index(keys).reindex(previous_keys)
np.testing.assert_array_equal(after_abort, keyed_previous["abort_event"].eq(3))
following_keys = pd.MultiIndex.from_arrays(
    [data["animal"], data["session"], data["trial"] + 1], names=keys,
)
keyed_following = data.set_index(keys).reindex(following_keys)
np.testing.assert_array_equal(isolated, is_abort.to_numpy()
                              & keyed_previous["success"].isin([-1, 1]).to_numpy()
                              & keyed_following["success"].isin([-1, 1]).to_numpy())
for column in previous:
    data[f"previous_{column}"] = previous[column]
    data[f"next_{column}"] = following[column]

# Record current selected aborts and both observed neighbors. After-abort mode
# includes the second and third abort in a three-abort sequence; isolated mode
# requires success ±1 immediately before and after the current abort.
manifest = data.loc[data["selected_abort"]].copy()
manifest.insert(0, "source_csv", str(SOURCE_PATH.relative_to(SCRIPT_DIR.parent)))
assert not manifest.duplicated(keys).any() and len(manifest) == sum(v[0] for v in EXPECTED_COUNTS.values())
assert (manifest["trial"] - manifest["previous_trial"]).eq(1).all()
if SELECTION == "after_abort":
    assert manifest["previous_abort_event"].eq(3).all()
else:
    assert (manifest["next_trial"] - manifest["trial"]).eq(1).all()
    assert manifest["previous_success"].isin([-1, 1]).all() and manifest["next_success"].isin([-1, 1]).all()
assert manifest["eligible_denominator_trial"].all()
for column in ["previous_trial", "previous_LED_trial", "previous_source_csv_row_index"]:
    manifest[column] = manifest[column].astype(int)
assert np.isfinite(manifest[["intended_fix", "LED_onset_time"]]).all().all()
manifest["abort_from_LED_onset"] = manifest["timed_fix"] - manifest["LED_onset_time"]
manifest["finite_aligned_time"] = np.isfinite(manifest["abort_from_LED_onset"])
finite = manifest.loc[manifest["finite_aligned_time"]]
assert finite["timed_fix"].lt(finite["intended_fix"]).all()
missing = manifest.loc[~manifest["finite_aligned_time"]]
expected_missing = ({7: {(92, 0): 1, (100, 0): 1}, 8: {(93, 1): 2}}[SESSION_TYPE]
                    if SELECTION == "after_abort" else {})
assert missing.groupby(["animal", "LED_trial"]).size().to_dict() == expected_missing

count_table = data.groupby(["animal", "LED_trial"])[["eligible_denominator_trial", "selected_abort"]].sum()
assert count_table["eligible_denominator_trial"].gt(0).all() and count_table["selected_abort"].gt(0).all()
for led, expected in EXPECTED_COUNTS.items():
    selected = manifest.loc[manifest["LED_trial"].eq(led)]
    denominator = int(data.loc[data["LED_trial"].eq(led), "eligible_denominator_trial"].sum())
    assert (len(selected), int(selected["finite_aligned_time"].sum()), denominator) == expected


# %%
############ Animal histograms and full-support area/count audits ############
curves = {}
panel_records, curve_parts = [], []
for animal in ANIMALS:
    for led in (0, 1):
        selected = manifest.loc[manifest["animal"].eq(animal) & manifest["LED_trial"].eq(led)]
        denominator = int(count_table.loc[(animal, led), "eligible_denominator_trial"])
        n_finite = int(selected["finite_aligned_time"].sum())
        for metric in METRICS:
            bins = bins_by_metric[metric]
            values = selected[metric].to_numpy()
            values = values[np.isfinite(values)]
            normalizer = denominator if metric == "abort_from_LED_onset" else len(values)
            counts, _ = np.histogram(values, bins=bins)
            assert len(values) > 0 and counts.sum() == len(values), (animal, led, metric)
            heights = counts / (normalizer * width_by_metric[metric])
            area = float(np.sum(heights * np.diff(bins)))
            expected_area = n_finite / denominator if metric == "abort_from_LED_onset" else 1.0
            assert np.isclose(area, expected_area, rtol=0, atol=1e-12)
            display = ABORT_DISPLAY_S if metric == "abort_from_LED_onset" else TIMING_RANGE_S
            outside = int(((values < display[0]) | (values > display[1])).sum())
            curves[(animal, led, metric)] = heights
            panel_records.append({
                "row": str(animal), "LED_trial": led, "metric": metric, "n_animals": 1,
                "selected_aborts": len(selected), "finite_aligned_aborts": n_finite,
                "missing_aligned_time": len(selected) - n_finite,
                "eligible_denominator": denominator, "denominator_definition": DENOMINATOR_DESCRIPTION,
                "selection": SELECTION, "plotted_observations": len(values),
                "outside_display": outside, "histogram_normalizer": normalizer,
                "full_histogram_area": area, "expected_area": expected_area,
                "coded_selected_abort_fraction": len(selected) / denominator,
                "bin_width_s": width_by_metric[metric], "display_left_s": display[0], "display_right_s": display[1],
            })
            curve_parts.append(pd.DataFrame({
                "row": str(animal), "LED_trial": led, "metric": metric, "n_animals": 1,
                "bin_left_s": bins[:-1], "bin_right_s": bins[1:],
                "height": heights, "sem": np.nan, "bin_count": counts,
            }))
panels = pd.DataFrame(panel_records)


# %%
############ Equal-animal mean and sample-SD/sqrt(6) bands in every column ############
mean_curves, sem_curves = {}, {}
for led in (0, 1):
    for metric in METRICS:
        bins = bins_by_metric[metric]
        matrix = np.stack([curves[(animal, led, metric)] for animal in ANIMALS])
        mean = matrix.mean(axis=0)
        sem = matrix.std(axis=0, ddof=1) / np.sqrt(len(ANIMALS))
        np.testing.assert_allclose(sem, np.sqrt(((matrix - mean) ** 2).sum(axis=0) / (6 * 5)), atol=1e-14)
        mean_curves[(led, metric)], sem_curves[(led, metric)] = mean, sem
        area = float(np.sum(mean * np.diff(bins)))
        animals_panel = panels.loc[panels["LED_trial"].eq(led) & panels["metric"].eq(metric)]
        expected_area = float(animals_panel["expected_area"].mean())
        assert np.isclose(area, expected_area, rtol=0, atol=1e-12)
        record = {"row": "equal_animal_mean", "LED_trial": led, "metric": metric, "n_animals": 6,
                  "full_histogram_area": area, "expected_area": expected_area,
                  "coded_selected_abort_fraction": animals_panel["coded_selected_abort_fraction"].mean(),
                  "selection": SELECTION, "denominator_definition": DENOMINATOR_DESCRIPTION,
                  "histogram_normalizer": np.nan, "bin_width_s": width_by_metric[metric],
                  "display_left_s": animals_panel.iloc[0]["display_left_s"],
                  "display_right_s": animals_panel.iloc[0]["display_right_s"]}
        # Count sums document contributing observations, not a pooled normalizer.
        for column in ["selected_aborts", "finite_aligned_aborts", "missing_aligned_time",
                       "eligible_denominator", "plotted_observations", "outside_display"]:
            record[column] = int(animals_panel[column].sum())
        panel_records.append(record)
        curve_parts.append(pd.DataFrame({
            "row": "equal_animal_mean", "LED_trial": led, "metric": metric, "n_animals": 6,
            "bin_left_s": bins[:-1], "bin_right_s": bins[1:],
            "height": mean, "sem": sem, "bin_count": np.nan,
        }))
panels = pd.DataFrame(panel_records)
curve_audit = pd.concat(curve_parts, ignore_index=True)
assert len(panels) == 7 * 2 * 3
assert not curve_audit.duplicated(["row", "LED_trial", "metric", "bin_left_s"]).any()
assert curve_audit["height"].ge(0).all() and np.isfinite(curve_audit["height"]).all()
assert np.isfinite(curve_audit.loc[curve_audit["row"].eq("equal_animal_mean"), "sem"]).all()

manifest.to_csv(SCRIPT_DIR / f"{OUTPUT_STEM}_selected_trials.csv", index=False)
panels.to_csv(SCRIPT_DIR / f"{OUTPUT_STEM}_count_area_audit.csv", index=False)
curve_audit.to_csv(SCRIPT_DIR / f"{OUTPUT_STEM}_curve_audit.csv", index=False)


# %%
############ Seven rows, including equal-animal mean ± SEM ############
fig, axes = plt.subplots(7, 3, figsize=(15, 20), sharex="col", sharey="col")
for row_index, row in enumerate([*[str(a) for a in ANIMALS], "equal_animal_mean"]):
    for column, metric in enumerate(METRICS):
        ax = axes[row_index, column]
        bins = bins_by_metric[metric]
        for led in (0, 1):
            if row == "equal_animal_mean":
                values, sem = mean_curves[(led, metric)], sem_curves[(led, metric)]
                ax.fill_between(bins, np.r_[values - sem, (values - sem)[-1]],
                                np.r_[values + sem, (values + sem)[-1]], step="post",
                                color=LED_COLORS[led], alpha=0.16, linewidth=0)
            else:
                values = curves[(int(row), led, metric)]
            ax.stairs(values, bins, color=LED_COLORS[led], linewidth=1.25)
            audit = panels.loc[panels["row"].eq(row) & panels["LED_trial"].eq(led) & panels["metric"].eq(metric)].iloc[0]
            label = "OFF" if led == 0 else "ON"
            if metric == "abort_from_LED_onset":
                if row == "equal_animal_mean":
                    note = f"{label} mean area = {audit.full_histogram_area:.3f}"
                else:
                    note = (f"{label} {int(audit.finite_aligned_aborts)}/{int(audit.eligible_denominator)}"
                            f" = {audit.full_histogram_area:.3f}")
                    if audit.missing_aligned_time:
                        note += f" (+{int(audit.missing_aligned_time)} missing time)"
            else:
                prefix = "total n" if row == "equal_animal_mean" else "n"
                note = f"{label} {prefix} = {int(audit.selected_aborts):,}"
            ax.text(0.98, 0.96 - 0.085 * led, note, ha="right", va="top",
                    transform=ax.transAxes, color=LED_COLORS[led], fontsize=8.5)
        if metric == "abort_from_LED_onset":
            ax.axvline(0, color="0.45", linestyle=":", linewidth=1)
            ax.set_xlim(*ABORT_DISPLAY_S)
        else:
            ax.set_xlim(*TIMING_RANGE_S)
        if column == 0:
            row_label = "Equal-animal mean ± SEM" if row == "equal_animal_mean" else f"Animal {row}"
            ax.set_ylabel(f"{row_label}\nDensity (s⁻¹)", fontsize=9)
        elif column == 2:
            ax.set_ylabel("Conditional abort rate (s⁻¹)" if SELECTION == "after_abort" else "Isolated-abort rate (s⁻¹)", fontsize=9)
        if row_index == 0:
            ax.set_title(COLUMN_TITLES[column], fontsize=12, pad=10)
        if row_index == 6:
            ax.set_xlabel("Time from scheduled LED onset (s)" if column == 2 else "Time from fixation onset (s)", fontsize=10)
        ax.tick_params(labelsize=8.5)
        ax.grid(axis="y", color="0.91", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)

# Each column shares its scale across animals and the mean; reserve annotation space.
for column, metric in enumerate(METRICS):
    bins = bins_by_metric[metric]
    visible = ((bins[:-1] < ABORT_DISPLAY_S[1]) & (bins[1:] > ABORT_DISPLAY_S[0])) if column == 2 else np.ones(len(bins) - 1, dtype=bool)
    upper = max(float(curves[(animal, led, metric)][visible].max()) for animal in ANIMALS for led in (0, 1))
    upper = max(upper, *(float((mean_curves[(led, metric)] + sem_curves[(led, metric)])[visible].max()) for led in (0, 1)))
    axes[0, column].set_ylim(0, upper * 1.28)

fig.suptitle(f"LED7 session type {SESSION_TYPE}: {SELECTION_TITLE}", fontsize=17, y=0.995)
fig.legend(handles=[Line2D([0], [0], color=LED_COLORS[led], lw=1.7, label=f"Current trial LED {'OFF' if led == 0 else 'ON'}")
                    for led in (0, 1)], loc="upper center", bbox_to_anchor=(0.5, 0.981), ncol=2, frameon=False, fontsize=11)
fig.text(0.5, 0.011,
         f"Timing panels: {TIMING_BIN_WIDTH_S * 1000:g} ms bins, selected current aborts normalized separately to unit area. Abort panel: {ABORT_BIN_WIDTH_S * 1000:g} ms bins.\n"
         f"Abort rate: selected finite aborts / {DENOMINATOR_DESCRIPTION}.\n"
         "Area labels use full −2 to +2 s bins. OFF zero is scheduled/counterfactual onset. Last row: equal-animal mean ± SEM.",
         ha="center", fontsize=9, linespacing=1.4)
fig.tight_layout(rect=(0.015, 0.052, 0.99, 0.96), h_pad=1.4, w_pad=1.4)
fig.savefig(FIGURE_PATH, dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)

print(panels.loc[panels["metric"].eq("abort_from_LED_onset"),
                 ["row", "LED_trial", "selected_aborts", "finite_aligned_aborts", "eligible_denominator",
                  "missing_aligned_time", "outside_display", "full_histogram_area"]].to_string(index=False))
print(f"\nSaved {FIGURE_PATH}")
print(f"Saved selected-trial manifest, count/area audit, and animal/mean/SEM curve audit under {OUTPUT_STEM}_*.csv")
print("All source, adjacency, count, timing-range, histogram-area, and equal-animal SEM checks passed.")
