# %%
"""LED7 abort timing at the first/last position of a run or between valid trials.

Consecutiveness uses the values in `trial`, never CSV row numbers. Run detection
precedes LED splitting and keeps event-3 trials even when their timing is missing.
"""

from pathlib import Path
import os


# %%
############ Editable parameters ############
SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = SCRIPT_DIR.parent / "raw_data"
SESSION_TYPES = (7, 8)
ANIMALS = (92, 93, 98, 99, 100, 103)
BIN_WIDTH_S = 0.020
FULL_RANGE_S = (-2.0, 2.0)
DISPLAY_RANGE_S = (-0.4, 0.5)
FIGURE_DPI = 220
LED_COLORS = {0: "tab:blue", 1: "tab:red"}
POSITIONS = ("first", "last", "isolated")
COLUMN_TITLES = ("First abort in a run", "Last abort in a run", "Isolated: valid–abort–valid")
ROW_LABELS = (*[str(animal) for animal in ANIMALS], "equal_animal_mean", "pooled")
OUTPUT_PREFIX = "led7_s7_s8_abort_rates_by_sequence_position"

EXPECTED_SOURCE_ROWS = {7: 138_440, 8: 148_428}
EXPECTED_COHORT_ROWS = {7: 100_251, 8: 113_941}
EXPECTED_LED_COUNTS = {7: {0: 66_226, 1: 34_025}, 8: {0: 70_892, 1: 43_049}}
EXPECTED_ABORTS = {7: 16_389, 8: 17_725}
EXPECTED_MULTI_RUNS = {7: 2_381, 8: 2_593}
EXPECTED_ISOLATED = {7: 9_863, 8: 10_511}
EXPECTED_BOUNDARY_RUNS = {7: 84, 8: 84}
EXPECTED_BOUNDARY_ABORTS = {7: 113, 8: 112}
EXPECTED_OTHER_SINGLETONS = {7: 964, 8: 1_111}
EXPECTED_SELECTED_LED_COUNTS = {
    7: {"first": {0: 1_425, 1: 956}, "last": {0: 1_218, 1: 1_163},
        "isolated": {0: 6_498, 1: 3_365}},
    8: {"first": {0: 1_309, 1: 1_284}, "last": {0: 786, 1: 1_807},
        "isolated": {0: 6_406, 1: 4_105}},
}

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

bins = np.linspace(*FULL_RANGE_S, round(np.diff(FULL_RANGE_S)[0] / BIN_WIDTH_S) + 1)
widths = np.diff(bins)
centers = (bins[:-1] + bins[1:]) / 2
visible_bins = (bins[:-1] < DISPLAY_RANGE_S[1]) & (bins[1:] > DISPLAY_RANGE_S[0])
np.testing.assert_allclose(widths, BIN_WIDTH_S, rtol=0, atol=1e-14)


# %%
############ Reusable classification, also exercised on synthetic sequences ############
def classify_abort_runs(data):
    """Return all event-3 rows and one audit row per observed abort run."""
    ordered = data.sort_values(["session_type", "animal", "session", "trial"], kind="stable").copy()
    keys = ["session_type", "animal", "session"]
    grouped = ordered.groupby(keys, sort=False)
    previous = grouped[["trial", "abort_event", "success"]].shift(1)
    following = grouped[["trial", "abort_event", "success"]].shift(-1)
    has_previous = previous["trial"].eq(ordered["trial"] - 1)
    has_next = following["trial"].eq(ordered["trial"] + 1)
    is_abort = ordered["abort_event"].eq(3)

    # Shifts find candidate neighbors, but only actual `trial` IDs establish
    # adjacency. LED_trial is deliberately absent from grouping and adjacency.
    begins_run = is_abort & ~(has_previous & previous["abort_event"].eq(3))
    ordered["run_number"] = begins_run.cumsum()
    for side, neighbor, adjacent in (
        ("previous", previous, has_previous), ("next", following, has_next)
    ):
        ordered[f"{side}_trial"] = neighbor["trial"]
        ordered[f"{side}_abort_event"] = neighbor["abort_event"]
        ordered[f"{side}_success"] = neighbor["success"]
        ordered[f"{side}_adjacent"] = adjacent
        ordered[f"{side}_boundary"] = np.select(
            [neighbor["trial"].isna(), ~adjacent],
            ["session_edge", "trial_id_gap"], default="observed",
        )

    manifest = ordered.loc[is_abort].copy()
    abort_groups = manifest.groupby("run_number", sort=False)
    manifest["run_position"] = abort_groups.cumcount() + 1
    manifest["observed_run_length"] = abort_groups["trial"].transform("size")
    manifest["run_start_trial"] = abort_groups["trial"].transform("min")
    manifest["run_end_trial"] = abort_groups["trial"].transform("max")
    manifest["run_id"] = (
        "s" + manifest["session_type"].astype(int).astype(str)
        + "_a" + manifest["animal"].astype(int).astype(str)
        + "_session" + manifest["session"].astype(int).astype(str)
        + "_trial" + manifest["run_start_trial"].astype(int).astype(str)
    )

    first = manifest.loc[manifest["run_position"].eq(1)].set_index("run_id")
    last = manifest.loc[manifest["run_position"].eq(manifest["observed_run_length"])].set_index("run_id")
    runs = first[
        ["source_csv", "session_type", "animal", "session", "observed_run_length",
         "run_start_trial", "run_end_trial", "previous_trial", "previous_abort_event",
         "previous_success", "previous_boundary"]
    ].copy()
    for column in ("next_trial", "next_abort_event", "next_success", "next_boundary"):
        runs[column] = last[column]
    runs["first_LED_trial"] = first["LED_trial"]
    runs["last_LED_trial"] = last["LED_trial"]
    runs["first_source_csv_row_index"] = first["source_csv_row_index"]
    runs["last_source_csv_row_index"] = last["source_csv_row_index"]
    runs["complete"] = first["previous_adjacent"] & last["next_adjacent"]
    runs["valid_neighbors"] = first["previous_success"].isin([-1, 1]) & last["next_success"].isin([-1, 1])
    runs["run_category"] = np.select(
        [~runs["complete"], runs["observed_run_length"].ge(2), runs["valid_neighbors"]],
        ["boundary_or_gap", "complete_multi", "isolated_valid_abort_valid"],
        default="singleton_other",
    )
    manifest["complete_run"] = manifest["run_id"].map(runs["complete"])
    manifest["run_category"] = manifest["run_id"].map(runs["run_category"])
    multi = manifest["run_category"].eq("complete_multi")
    manifest["selection"] = np.select(
        [multi & manifest["run_position"].eq(1),
         multi & manifest["run_position"].eq(manifest["observed_run_length"]),
         multi, manifest["run_category"].eq("isolated_valid_abort_valid"),
         manifest["run_category"].eq("boundary_or_gap")],
        ["first", "last", "interior", "isolated", "boundary_or_gap"],
        default="singleton_other",
    )
    manifest["finite_timed_fix"] = np.isfinite(manifest["timed_fix"])
    manifest["abort_from_LED_onset"] = manifest["timed_fix"] - manifest["LED_onset_time"]
    runs["missing_timed_fix"] = manifest.groupby("run_id")["finite_timed_fix"].agg(lambda v: int((~v).sum()))

    assert not runs.index.duplicated().any()
    assert len(manifest) == int(is_abort.sum())
    assert (runs["run_end_trial"] - runs["run_start_trial"] + 1).eq(runs["observed_run_length"]).all()
    internal_trial_steps = manifest.groupby("run_id")["trial"].diff().dropna()
    assert internal_trial_steps.eq(1).all()
    complete = runs.loc[runs["complete"]]
    assert (complete["run_start_trial"] - complete["previous_trial"]).eq(1).all()
    assert (complete["next_trial"] - complete["run_end_trial"]).eq(1).all()
    assert complete[["previous_abort_event", "next_abort_event"]].ne(3).all().all()
    first_ids = set(manifest.loc[manifest["selection"].eq("first"), "run_id"])
    last_ids = set(manifest.loc[manifest["selection"].eq("last"), "run_id"])
    assert first_ids == last_ids == set(runs.index[runs["run_category"].eq("complete_multi")])
    isolated = runs.loc[runs["run_category"].eq("isolated_valid_abort_valid")]
    assert isolated[["previous_success", "next_success"]].isin([-1, 1]).all().all()
    return manifest.drop(columns="run_number"), runs.reset_index()


# %%
############ Small classification checks: shuffled rows, gaps, LED switches and edges ############
cases = [
    ("mixed_LED", [100, 101, 102, 103], "V A A V", [0, 0, 1, 1], ["first", "last"]),
    ("long_run", list(range(200, 208)), "V A A A A A A V", [0] * 8,
     ["first", "interior", "interior", "interior", "interior", "last"]),
    ("trial_gap", [300, 301, 303, 304], "V A A V", [0] * 4,
     ["boundary_or_gap", "boundary_or_gap"]),
    ("left_edge", [400, 401, 402], "A A V", [0] * 3, ["boundary_or_gap"] * 2),
    ("right_edge", [500, 501, 502], "V A A", [0] * 3, ["boundary_or_gap"] * 2),
    ("isolated", [600, 601, 602], "E A V", [1] * 3, ["isolated"]),
    ("other_singleton", [700, 701, 702], "O A V", [0] * 3, ["singleton_other"]),
    ("other_bounded_run", [800, 801, 802, 803], "O A A O", [1] * 4, ["first", "last"]),
    ("session_end", [900, 901], "V A", [0] * 2, ["boundary_or_gap"]),
    ("next_session_start", [902, 903], "A V", [0] * 2, ["boundary_or_gap"]),
]
synthetic_rows = []
for session, (case, trials, outcomes, leds, expected) in enumerate(cases, start=1):
    for trial, outcome, led in zip(trials, outcomes.split(), leds):
        abort_event, success = {"A": (3, 0), "V": (-2, 1), "E": (-1, -1), "O": (2, 0)}[outcome]
        synthetic_rows.append({
            "source_csv": "synthetic", "source_csv_row_index": len(synthetic_rows),
            "session_type": 7, "animal": 92, "session": session, "trial": trial,
            "LED_trial": led, "abort_event": abort_event, "success": success,
            "timed_fix": np.nan if trial == 102 else 0.3,
            "intended_fix": 0.5, "LED_onset_time": 0.2,
        })
synthetic = pd.DataFrame(synthetic_rows).sample(frac=1, random_state=123)
synthetic.index = np.arange(len(synthetic)) * 17 + 42
test_manifest, test_runs = classify_abort_runs(synthetic)
for session, (case, trials, outcomes, leds, expected) in enumerate(cases, start=1):
    actual = test_manifest.loc[test_manifest["session"].eq(session)]
    assert actual["selection"].tolist() == expected, case
mixed = test_manifest.loc[test_manifest["session"].eq(1)]
assert mixed["LED_trial"].tolist() == [0, 1] and mixed["run_id"].nunique() == 1
assert mixed["finite_timed_fix"].tolist() == [True, False]
assert test_runs.loc[test_runs["session"].eq(3), "observed_run_length"].tolist() == [1, 1]
assert test_runs.loc[test_runs["session"].eq(2), "observed_run_length"].item() == 6
print("Synthetic sequence checks passed; adjacency is based on trial IDs.", flush=True)


# %%
############ Read the all-trial denominators and classify every event-3 abort ############
data_by_type = {}
manifest_parts = []
run_parts = []
source_columns = [
    "source_csv", "source_csv_row_index", "session_type", "animal", "session", "trial",
    "training_level", "repeat_trial", "LED_trial", "abort_event", "success",
    "timed_fix", "intended_fix", "LED_onset_time",
]
for session_type in SESSION_TYPES:
    source_path = RAW_DATA_DIR / f"LED7_s{session_type}.csv"
    source = pd.read_csv(source_path, float_precision="round_trip")
    assert source.shape == (EXPECTED_SOURCE_ROWS[session_type], 52)
    assert source["training_level"].eq(16).all()
    assert source["session_type"].eq(session_type).all()
    assert (source["repeat_trial"].isin([0, 2]) | source["repeat_trial"].isna()).all()
    assert (source["LED_trial"].isin([0, 1]) | source["LED_trial"].isna()).all()
    assert source["LED_trial"].notna().all(), "Missing LED flags need a separate unlabeled audit."
    assert not source.duplicated(["animal", "session", "trial"]).any()
    source["source_csv"] = str(source_path.relative_to(SCRIPT_DIR.parent))
    # This index is provenance only: it never determines consecutiveness.
    source["source_csv_row_index"] = np.arange(len(source))
    data = source.loc[source["animal"].isin(ANIMALS), source_columns].copy()
    assert len(data) == EXPECTED_COHORT_ROWS[session_type]
    assert data["LED_trial"].value_counts().to_dict() == EXPECTED_LED_COUNTS[session_type]
    assert np.isfinite(data[["animal", "session", "trial", "LED_onset_time", "intended_fix"]]).all().all()
    for column in ("animal", "session_type", "session", "trial", "LED_trial"):
        assert data[column].eq(np.floor(data[column])).all()
        data[column] = data[column].astype(int)
    assert data["LED_onset_time"].between(0, data["intended_fix"]).all()
    event3 = data["abort_event"].eq(3)
    finite = np.isfinite(data["timed_fix"])
    assert int(event3.sum()) == EXPECTED_ABORTS[session_type]
    assert int((event3 & ~finite).sum()) == 2
    assert data.loc[event3 & finite, "timed_fix"].lt(data.loc[event3 & finite, "intended_fix"]).all()

    manifest, runs = classify_abort_runs(data)
    counts = manifest["selection"].value_counts()
    assert counts["first"] == counts["last"] == EXPECTED_MULTI_RUNS[session_type]
    assert counts["isolated"] == EXPECTED_ISOLATED[session_type]
    assert counts["boundary_or_gap"] == EXPECTED_BOUNDARY_ABORTS[session_type]
    assert counts["singleton_other"] == EXPECTED_OTHER_SINGLETONS[session_type]
    assert int(runs["run_category"].eq("boundary_or_gap").sum()) == EXPECTED_BOUNDARY_RUNS[session_type]
    assert int(runs["observed_run_length"].sum()) == len(manifest)
    for position in POSITIONS:
        selection = manifest.loc[manifest["selection"].eq(position)]
        assert selection["LED_trial"].value_counts().to_dict() == EXPECTED_SELECTED_LED_COUNTS[session_type][position]
    print(f"\ns{session_type}: {len(data):,} eligible trials; {len(manifest):,} event-3 aborts")
    print(runs.groupby("run_category").agg(runs=("run_id", "size"), aborts=("observed_run_length", "sum")).to_string())
    print("Abort positions:", counts.to_dict(), flush=True)
    data_by_type[session_type] = data
    manifest_parts.append(manifest)
    run_parts.append(runs)

manifest = pd.concat(manifest_parts, ignore_index=True)
run_audit = pd.concat(run_parts, ignore_index=True)
assert not manifest.duplicated(["session_type", "animal", "session", "trial"]).any()
assert not run_audit["run_id"].duplicated().any()


# %%
############ Individual curves, then equal-animal mean/SEM and pooled counts ############
rates = {}
sems = {}
bin_counts = {}
panel_records = []
curve_records = []
for session_type in SESSION_TYPES:
    data = data_by_type[session_type]
    aborts = manifest.loc[manifest["session_type"].eq(session_type)]
    for row_label in ROW_LABELS:
        individual = row_label not in ("equal_animal_mean", "pooled")
        eligible = data.loc[data["animal"].eq(int(row_label))] if individual else data
        row_aborts = aborts.loc[aborts["animal"].eq(int(row_label))] if individual else aborts
        for position in POSITIONS:
            for led in (0, 1):
                key = (session_type, row_label, position, led)
                denominator = int(eligible["LED_trial"].eq(led).sum())
                assert denominator > 0
                selected = row_aborts.loc[row_aborts["selection"].eq(position) & row_aborts["LED_trial"].eq(led)]
                plotted = selected.loc[selected["finite_timed_fix"]]
                times = plotted["abort_from_LED_onset"].to_numpy()
                counts, _ = np.histogram(times, bins=bins)
                assert int(counts.sum()) == len(plotted), "Full bins must contain every selected finite abort."
                bin_counts[key] = counts

                if row_label == "equal_animal_mean":
                    animal_rates = np.stack([rates[session_type, str(a), position, led] for a in ANIMALS])
                    heights = animal_rates.mean(axis=0)
                    sem = animal_rates.std(axis=0, ddof=1) / np.sqrt(len(ANIMALS))
                    fractions = []
                    for animal in ANIMALS:
                        animal_denominator = int((data["animal"].eq(animal) & data["LED_trial"].eq(led)).sum())
                        fractions.append(int(plotted["animal"].eq(animal).sum()) / animal_denominator)
                    reference_area = float(np.mean(fractions))
                    manual_sem = np.sqrt(np.sum((animal_rates - heights) ** 2, axis=0) / (len(ANIMALS) * (len(ANIMALS) - 1)))
                    np.testing.assert_allclose(sem, manual_sem, rtol=1e-12, atol=1e-14)
                    normalization = "equal mean of six animal rates"
                else:
                    heights = counts / (denominator * BIN_WIDTH_S)
                    sem = np.full(len(counts), np.nan)
                    reference_area = len(plotted) / denominator
                    normalization = "selected finite aborts / all LED-group trials"
                area = float(np.sum(heights * widths))
                assert np.isclose(area, reference_area, rtol=0, atol=1e-14)
                assert np.isfinite(heights).all() and (heights >= 0).all()
                rates[key], sems[key] = heights, sem

                if not individual:
                    animal_counts = np.stack([bin_counts[session_type, str(a), position, led] for a in ANIMALS])
                    np.testing.assert_array_equal(counts, animal_counts.sum(axis=0))
                    if row_label == "pooled":
                        weighted = sum(
                            rates[session_type, str(a), position, led]
                            * int((data["animal"].eq(a) & data["LED_trial"].eq(led)).sum())
                            for a in ANIMALS
                        ) / denominator
                        np.testing.assert_allclose(heights, weighted, rtol=1e-12, atol=1e-14)

                record = {
                    "session_type": session_type, "row": row_label, "position": position,
                    "LED_trial": led, "n_animals": 1 if individual else len(ANIMALS),
                    "all_condition_trials": denominator,
                    "selected_coded_aborts": len(selected), "finite_plotted_aborts": len(plotted),
                    "missing_timed_fix": len(selected) - len(plotted),
                    "full_histogram_area": area, "reference_fraction": reference_area,
                    "normalization": normalization, "bin_width_s": BIN_WIDTH_S,
                    "display_min_s": DISPLAY_RANGE_S[0], "display_max_s": DISPLAY_RANGE_S[1],
                    "aborts_outside_display": int(((times < DISPLAY_RANGE_S[0]) | (times > DISPLAY_RANGE_S[1])).sum()),
                }
                for length in (2, 3, 4, 5):
                    record[f"n_runs_length_{length}"] = int(selected["observed_run_length"].eq(length).sum())
                record["n_runs_length_ge6"] = int(selected["observed_run_length"].ge(6).sum())
                panel_records.append(record)
                curve_records.append(pd.DataFrame({
                    "session_type": session_type, "row": row_label, "position": position,
                    "LED_trial": led, "bin_left_s": bins[:-1], "bin_right_s": bins[1:],
                    "selected_bin_count": counts, "rate_per_s": heights, "across_animal_sem": sem,
                    "normalization": normalization,
                }))

panel_audit = pd.DataFrame(panel_records)
curve_audit = pd.concat(curve_records, ignore_index=True)
panel_lookup = panel_audit.set_index(["session_type", "row", "position", "LED_trial"])
assert len(panel_audit) == len(SESSION_TYPES) * len(ROW_LABELS) * len(POSITIONS) * 2
for session_type in SESSION_TYPES:
    for position in POSITIONS:
        rows = panel_audit.loc[panel_audit["session_type"].eq(session_type) & panel_audit["position"].eq(position)]
        animals_only = rows.loc[rows["row"].isin([str(a) for a in ANIMALS])]
        assert int(animals_only["all_condition_trials"].sum()) == EXPECTED_COHORT_ROWS[session_type]
        for led in (0, 1):
            same_led = animals_only.loc[animals_only["LED_trial"].eq(led)]
            pooled = panel_lookup.loc[(session_type, "pooled", position, led)]
            for field in ("all_condition_trials", "selected_coded_aborts", "finite_plotted_aborts", "missing_timed_fix"):
                assert same_led[field].sum() == pooled[field]


# %%
############ Save source-trial provenance, exact run lengths, and all plotted values ############
outputs = {
    "sequence_manifest": manifest,
    "run_audit": run_audit,
    "panel_audit": panel_audit,
    "curve_audit": curve_audit,
}
for suffix, frame in outputs.items():
    output_path = SCRIPT_DIR / f"{OUTPUT_PREFIX}_{suffix}.csv"
    frame.to_csv(output_path, index=False)
    print(f"Saved {output_path} ({len(frame):,} rows)", flush=True)
print("\nPooled condition audit:")
print(panel_audit.loc[panel_audit["row"].eq("pooled"), [
    "session_type", "position", "LED_trial", "selected_coded_aborts", "finite_plotted_aborts",
    "all_condition_trials", "full_histogram_area", "aborts_outside_display",
]].to_string(index=False))


# %%
############ Plot with the same row-wise scales in both session-type figures ############
row_ymax = {}
for row_label in ROW_LABELS:
    maximum = 0.0
    for session_type in SESSION_TYPES:
        for position in POSITIONS:
            for led in (0, 1):
                key = (session_type, row_label, position, led)
                upper = rates[key] + np.nan_to_num(sems[key], nan=0)
                maximum = max(maximum, float(upper[visible_bins].max()))
    # Keep the top third clear for selection and sequence-length annotations.
    row_ymax[row_label] = max(maximum * 1.55, 0.01)

for session_type in SESSION_TYPES:
    fig, axes = plt.subplots(len(ROW_LABELS), len(POSITIONS), figsize=(18.8, 25.5), sharex=True)
    for row_index, row_label in enumerate(ROW_LABELS):
        for col_index, position in enumerate(POSITIONS):
            ax = axes[row_index, col_index]
            for led in (0, 1):
                key = (session_type, row_label, position, led)
                heights, sem = rates[key], sems[key]
                color = LED_COLORS[led]
                if row_label == "equal_animal_mean":
                    lower, upper = np.maximum(0, heights - sem), heights + sem
                    ax.fill_between(
                        bins, np.r_[lower, lower[-1]], np.r_[upper, upper[-1]],
                        step="post", color=color, alpha=0.17, linewidth=0,
                    )
                ax.stairs(heights, bins, color=color, linewidth=1.35, baseline=None)
                panel = panel_lookup.loc[key]
                condition = "OFF" if led == 0 else "ON"
                count_text = (
                    f"{condition}: n={int(panel['finite_plotted_aborts']):,}; "
                    f"N={int(panel['all_condition_trials']):,}"
                )
                if panel["missing_timed_fix"]:
                    count_text += f"; missing time={int(panel['missing_timed_fix'])}"
                if position != "isolated":
                    length_counts = [int(panel[f"n_runs_length_{length}"]) for length in (2, 3, 4, 5)]
                    length_counts.append(int(panel["n_runs_length_ge6"]))
                    count_text += "\nL=2/3/4/5/6+: " + "/".join(f"{n:,}" for n in length_counts)
                ax.text(
                    0.025, 0.98 - led * 0.155, count_text, transform=ax.transAxes,
                    va="top", ha="left", color=color, fontsize=8.4, linespacing=1.25,
                )
            ax.axvline(0, color="0.4", linestyle="--", linewidth=0.9, zorder=0)
            ax.set_xlim(*DISPLAY_RANGE_S)
            ax.set_ylim(0, row_ymax[row_label])
            ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4])
            ax.tick_params(labelsize=9, labelbottom=True)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.grid(axis="y", color="0.90", linewidth=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            if row_index == 0:
                ax.set_title(COLUMN_TITLES[col_index], fontsize=14, pad=12)
            if col_index == 0:
                label = {
                    "equal_animal_mean": "Equal-animal mean ± SEM",
                    "pooled": "Pooled trials",
                }.get(row_label, f"Animal {row_label}")
                ax.set_ylabel(label + "\nAbort rate (s⁻¹)", fontsize=10.5)
            if row_index == len(ROW_LABELS) - 1:
                ax.set_xlabel("Time from scheduled LED onset (s)", fontsize=11)

    fig.suptitle(f"LED7 session type {session_type}: abort timing by sequence position", fontsize=19, y=0.990)
    fig.legend(
        handles=[Line2D([0], [0], color=LED_COLORS[led], lw=1.6, label="LED OFF" if led == 0 else "LED ON") for led in (0, 1)],
        loc="upper center", bbox_to_anchor=(0.5, 0.978), ncol=2, frameon=False, fontsize=12,
    )
    fig.text(
        0.5, 0.018,
        "20 ms bins; n = plotted aborts, N = all eligible LED-group trials; L = number of consecutive event-3 trials in a run.\n"
        "Trial IDs must differ by 1 within the same animal/session; runs touching gaps or session edges are excluded.\n"
        "Equal-animal row averages six rates ± SEM; its n/N annotations are totals across animals. Pooled row divides summed counts by summed N.\n"
        "LED OFF uses scheduled/counterfactual onset. Full histogram areas use −2…2 s; the displayed range is cropped.",
        ha="center", va="bottom", fontsize=9.1, linespacing=1.45,
    )
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.070, top=0.943, hspace=0.32, wspace=0.22)
    figure_path = SCRIPT_DIR / f"led7_s{session_type}_abort_rates_by_sequence_position_8x3.png"
    fig.savefig(figure_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure: {figure_path}", flush=True)

print("All sequence, denominator, histogram-area, pooling, and SEM checks passed.", flush=True)
