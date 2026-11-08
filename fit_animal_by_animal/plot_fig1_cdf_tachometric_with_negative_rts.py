# %% Imports
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# %% Parameters (edit here)
DESIRED_BATCHES = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]
ABLS = [20, 40, 60]
ABS_ILDS = [1, 2, 4, 8, 16]
ABS_ILD_COLORS = {1: "tab:blue", 2: "tab:orange", 4: "tab:green", 8: "tab:red", 16: "tab:purple"}
RT_MIN_S = -1.0
RT_MAX_S = 1.0
BIN_WIDTH_S = 0.020
RANDOM_SEED = 42
DISPLAY_LIMITS_MS = [(-1000, 600), (-100, 600)]
PNG_DPI = 300
SHOW_PLOT = False

SCRIPT_PATH = Path(__file__).resolve()
REPO_DIR = SCRIPT_PATH.parent.parent
CSV_DIR = REPO_DIR / "raw_data" / "batch_csvs"
COHORT_MANIFEST = REPO_DIR / "all_30_fit_consolidation_manifest.md"
ORIGINAL_DIR = SCRIPT_PATH.parent / "fig1_tachometric_by_abl_abs_ild"
ORIGINAL_SUMMARY = ORIGINAL_DIR / "run_summary.json"
OUTPUT_DIR = SCRIPT_PATH.parent / "fig1_tachometric_with_negative_rts"


# %% Read the same batch CSVs and verify the 30-animal cohort and source hashes
original_summary = json.loads(ORIGINAL_SUMMARY.read_text())
expected_hashes = {record["batch"]: record["sha256"] for record in original_summary["sources"]}
expected_animals = {
    (batch, int(animal))
    for batch, animal in re.findall(
        r"^- `([^/`]+)/([0-9]+)`: source=", COHORT_MANIFEST.read_text(), flags=re.MULTILINE
    )
}
if len(expected_animals) != 30:
    raise ValueError("The paper manifest must contain exactly 30 batch/animal pairs.")

frames = []
source_records = []
for batch in DESIRED_BATCHES:
    csv_path = CSV_DIR / f"batch_{batch}_valid_and_aborts.csv"
    with csv_path.open("rb") as source_file:
        source_hash = hashlib.file_digest(source_file, "sha256").hexdigest()
    if source_hash != expected_hashes[batch]:
        raise ValueError(f"{csv_path} changed since the original positive-RT analysis.")
    frame = pd.read_csv(
        csv_path,
        usecols=["batch_name", "animal", "success", "abort_event", "RTwrtStim", "ABL", "ILD"],
    )
    if set(frame["batch_name"].unique()) != {batch}:
        raise ValueError(f"Unexpected batch labels in {csv_path}.")
    frame["source_csv"] = str(csv_path)
    frame["source_row_index"] = np.arange(len(frame))  # Zero-based data row; excludes CSV header.
    frames.append(frame)
    source_records.append({"batch": batch, "path": str(csv_path), "sha256": source_hash, "n_rows": len(frame)})

all_data = pd.concat(frames, ignore_index=True)
observed_animals = {
    (batch, int(animal))
    for batch, animal in all_data[["batch_name", "animal"]].drop_duplicates().itertuples(index=False, name=None)
}
if observed_animals != expected_animals:
    raise ValueError(
        f"Cohort mismatch: missing={sorted(expected_animals - observed_animals)}, "
        f"extra={sorted(observed_animals - expected_animals)}."
    )
animal_pairs = sorted(expected_animals)


# %% Select valid choices plus fixation aborts, then randomize abort choices once
selected = (
    all_data["RTwrtStim"].between(RT_MIN_S, RT_MAX_S, inclusive="both")
    & all_data["ABL"].isin(ABLS)
    & all_data["ILD"].abs().isin(ABS_ILDS)
    & (all_data["success"].isin([1, -1]) | all_data["abort_event"].eq(3))
)
plot_trials = all_data.loc[selected].copy()
plot_trials["abs_ILD"] = plot_trials["ILD"].abs().astype(int)
plot_trials["is_valid_choice"] = plot_trials["success"].isin([1, -1])
plot_trials["choice_is_simulated"] = ~plot_trials["is_valid_choice"]
plot_trials["correct"] = plot_trials["success"].eq(1).astype(int)
abort_mask = plot_trials["choice_is_simulated"]
n_aborts = int(abort_mask.sum())

# Draw in fixed batch order and original CSV row order, before grouping or plotting.
rng = np.random.default_rng(RANDOM_SEED)
abort_choices = rng.choice([-1, 1], size=n_aborts, p=[0.5, 0.5])
plot_trials.loc[abort_mask, "correct"] = (
    plot_trials.loc[abort_mask, "ILD"].to_numpy() * abort_choices > 0
).astype(int)
plot_trials["correct_valid"] = (plot_trials["is_valid_choice"] & plot_trials["correct"].eq(1)).astype(int)
plot_trials["correct_simulated"] = (abort_mask & plot_trials["correct"].eq(1)).astype(int)

abort_audit = plot_trials.loc[
    abort_mask,
    ["source_csv", "source_row_index", "batch_name", "animal", "success", "abort_event", "RTwrtStim", "ABL", "ILD", "choice_is_simulated"],
].copy()
abort_audit["csv_line_number"] = abort_audit["source_row_index"] + 2
abort_audit["random_draw_index"] = np.arange(n_aborts)
abort_audit["simulated_choice"] = abort_choices  # -1 = left, +1 = right.
abort_audit["simulated_correct"] = plot_trials.loc[abort_mask, "correct"].to_numpy()

# Mirror the original positive edges, preserving every positive-time assignment exactly.
positive_edges_s = np.asarray(original_summary["binning"]["edges_s"])
bin_edges_s = np.concatenate((-positive_edges_s[:0:-1], positive_edges_s))
if not (
    np.isclose(bin_edges_s[0], RT_MIN_S)
    and np.isclose(bin_edges_s[-1], RT_MAX_S)
    and np.allclose(np.diff(bin_edges_s), BIN_WIDTH_S)
):
    raise ValueError("Requested RT window/bin width disagrees with the mirrored original bin edges.")
n_bins = len(bin_edges_s) - 1
bin_centers_s = (bin_edges_s[:-1] + bin_edges_s[1:]) / 2
plot_trials["bin_index"] = np.minimum(
    np.searchsorted(bin_edges_s, plot_trials["RTwrtStim"].to_numpy(), side="right") - 1,
    n_bins - 1,
)
print(
    f"Cohort: {len(animal_pairs)} animals. Selected {len(plot_trials):,} trials: "
    f"{int(plot_trials['is_valid_choice'].sum()):,} valid choices and {n_aborts:,} simulated-choice aborts."
)


# %% Animal-bin accuracy, followed by equal-animal mean and sample SEM
animal_keys = ["batch_name", "animal", "ABL", "abs_ILD"]
animal_bins = plot_trials.groupby(animal_keys + ["bin_index"]).agg(
    n_trials=("correct", "size"),
    n_correct=("correct", "sum"),
    n_valid_choices=("is_valid_choice", "sum"),
    n_simulated_aborts=("choice_is_simulated", "sum"),
    n_correct_valid=("correct_valid", "sum"),
    n_correct_simulated=("correct_simulated", "sum"),
)
complete_index = pd.MultiIndex.from_tuples(
    [
        (batch, animal, abl, ild, index)
        for batch, animal in animal_pairs
        for abl in ABLS
        for ild in ABS_ILDS
        for index in range(n_bins)
    ],
    names=animal_keys + ["bin_index"],
)
animal_bins = animal_bins.reindex(complete_index, fill_value=0).reset_index()
animal_bins["accuracy"] = animal_bins["n_correct"] / animal_bins["n_trials"].replace(0, np.nan)
group_bins = animal_bins.groupby(["ABL", "abs_ILD", "bin_index"], as_index=False).agg(
    mean_accuracy=("accuracy", "mean"),
    std_accuracy=("accuracy", "std"),
    n_animals=("accuracy", "count"),
    n_trials=("n_trials", "sum"),
    n_correct=("n_correct", "sum"),
    n_valid_choices=("n_valid_choices", "sum"),
    n_simulated_aborts=("n_simulated_aborts", "sum"),
    n_correct_valid=("n_correct_valid", "sum"),
    n_correct_simulated=("n_correct_simulated", "sum"),
)
group_bins["sem_accuracy"] = group_bins["std_accuracy"] / np.sqrt(group_bins["n_animals"])
for table in [animal_bins, group_bins]:
    indices = table["bin_index"].to_numpy()
    table["bin_start_s"] = bin_edges_s[indices]
    table["bin_end_s"] = bin_edges_s[indices + 1]
    table["bin_center_s"] = bin_centers_s[indices]


# %% CDF: normalize each animal over the full -1 to 1 s window
animal_cdf = animal_bins[animal_keys + ["bin_index", "bin_end_s", "n_trials"]].copy()
animal_groups = animal_cdf.groupby(animal_keys, sort=False)
animal_cdf["n_condition_trials"] = animal_groups["n_trials"].transform("sum")
animal_cdf["n_cumulative_trials"] = animal_groups["n_trials"].cumsum()
animal_cdf["cumulative_fraction"] = (
    animal_cdf["n_cumulative_trials"] / animal_cdf["n_condition_trials"].replace(0, np.nan)
)
animal_cdf = animal_cdf.rename(columns={"bin_end_s": "time_s"})
start_points = animal_cdf.groupby(animal_keys, as_index=False).first()
start_points["bin_index"] = -1
start_points["time_s"] = RT_MIN_S
start_points["n_trials"] = 0
start_points["n_cumulative_trials"] = 0
start_points["cumulative_fraction"] = np.where(start_points["n_condition_trials"].gt(0), 0.0, np.nan)
animal_cdf = pd.concat([start_points, animal_cdf], ignore_index=True)
animal_cdf = animal_cdf.sort_values(animal_keys + ["bin_index"]).reset_index(drop=True)
group_cdf = animal_cdf.groupby(["ABL", "abs_ILD", "bin_index", "time_s"], as_index=False).agg(
    mean_cumulative_fraction=("cumulative_fraction", "mean"),
    std_cumulative_fraction=("cumulative_fraction", "std"),
    n_animals=("cumulative_fraction", "count"),
    n_condition_trials=("n_condition_trials", "sum"),
    n_cumulative_trials=("n_cumulative_trials", "sum"),
)
group_cdf["sem_cumulative_fraction"] = group_cdf["std_cumulative_fraction"] / np.sqrt(group_cdf["n_animals"])
assert int(animal_bins["n_trials"].sum()) == len(plot_trials)
assert int(animal_bins["n_simulated_aborts"].sum()) == n_aborts
assert group_bins.loc[group_bins["n_animals"].lt(2), "sem_accuracy"].isna().all()
assert np.allclose(group_cdf.loc[group_cdf["time_s"].eq(RT_MAX_S), "mean_cumulative_fraction"], 1)


# %% Save the numerical results and the exact simulated abort choices
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
table_paths = {
    "abort_random_choices": OUTPUT_DIR / "abort_random_choices.csv",
    "animal_bin_summary": OUTPUT_DIR / "animal_bin_summary.csv",
    "group_bin_summary": OUTPUT_DIR / "group_bin_summary.csv",
    "animal_cumulative_rtd": OUTPUT_DIR / "animal_cumulative_rtd.csv",
    "group_cumulative_rtd": OUTPUT_DIR / "group_cumulative_rtd.csv",
}
for name, table in [
    ("abort_random_choices", abort_audit), ("animal_bin_summary", animal_bins),
    ("group_bin_summary", group_bins), ("animal_cumulative_rtd", animal_cdf),
    ("group_cumulative_rtd", group_cdf),
]:
    table.to_csv(table_paths[name], index=False)


# %% Two views of identical full-window curves: ABL rows, CDF left, accuracy right
plt.rcParams.update({"font.family": "DejaVu Sans"})
figures = {}
figure_records = []
for xmin_ms, xmax_ms in DISPLAY_LIMITS_MS:
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), sharex=True, sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.195, top=0.865, hspace=0.26, wspace=0.22)
    for row, abl in enumerate(ABLS):
        cdf_ax, accuracy_ax = axes[row]
        for ild in ABS_ILDS:
            cdf = group_cdf.loc[group_cdf["ABL"].eq(abl) & group_cdf["abs_ILD"].eq(ild)].sort_values("bin_index")
            accuracy = group_bins.loc[group_bins["ABL"].eq(abl) & group_bins["abs_ILD"].eq(ild)].sort_values("bin_index")
            for ax, x_ms, mean, sem, marker in [
                (
                    cdf_ax, cdf["time_s"].to_numpy() * 1000,
                    cdf["mean_cumulative_fraction"].to_numpy(), cdf["sem_cumulative_fraction"].to_numpy(), None,
                ),
                (
                    accuracy_ax, accuracy["bin_center_s"].to_numpy() * 1000,
                    accuracy["mean_accuracy"].to_numpy(), accuracy["sem_accuracy"].to_numpy(), "o",
                ),
            ]:
                ax.fill_between(x_ms, mean - sem, mean + sem, color=ABS_ILD_COLORS[ild], alpha=0.16, linewidth=0)
                ax.plot(x_ms, mean, color=ABS_ILD_COLORS[ild], linewidth=1.7, marker=marker, markersize=2, label=f"{ild} dB")
        accuracy_ax.axhline(0.5, color="0.6", linestyle="--", linewidth=0.8, zorder=0)
        for ax in [cdf_ax, accuracy_ax]:
            ax.axvline(0, color="0.45", linestyle=":", linewidth=1, zorder=0)
            ax.set_xlim(xmin_ms, xmax_ms)
            ax.set_xticks(np.arange(xmin_ms, xmax_ms + 1, 200 if xmin_ms == -1000 else 100))
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 0.5, 1])
            ax.tick_params(labelsize=11, labelleft=True)
            ax.spines[["top", "right"]].set_visible(False)
        cdf_ax.set_title(f"ABL = {abl} dB: cumulative RTD", fontsize=13, pad=10)
        accuracy_ax.set_title(f"ABL = {abl} dB: accuracy", fontsize=13, pad=10)
        cdf_ax.set_ylabel("Cumulative fraction of trials", fontsize=12)
        accuracy_ax.set_ylabel("Accuracy", fontsize=13)

    fig.suptitle("Cumulative reaction times and accuracy at each ABL", fontsize=18, y=0.985)
    fig.text(
        0.5, 0.925,
        f"Valid trials + fixation aborts  |  {BIN_WIDTH_S * 1000:g} ms bins  |  Equal-animal mean ± SEM\n"
        f"Abort choices simulated: P(left) = P(right) = 0.5; seed {RANDOM_SEED}",
        ha="center", fontsize=11, linespacing=1.6,
    )
    fig.supxlabel("Reaction time from stimulus onset (ms)", fontsize=14, y=0.13)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title="|ILD| (dB)", loc="lower center", bbox_to_anchor=(0.5, 0.06), ncol=5, frameon=False, fontsize=11, title_fontsize=11)
    fig.text(
        0.5, 0.005,
        "CDF denominator: each animal's valid + abort trials over −1 to 1 s; accuracy denominator: trials in each time bin.\n"
        "|ILD| = 1–8 dB: up to 30 animals; 16 dB: up to 24. Negative-time accuracy uses simulated choices, not observed responses.",
        ha="center", fontsize=9.5, linespacing=1.5,
    )
    output_stem = f"fig1_cdf_tachometric_negative_rts_xlim_minus{abs(xmin_ms)}_to_{xmax_ms}ms"
    png_path = OUTPUT_DIR / f"{output_stem}.png"
    fig.savefig(png_path, dpi=PNG_DPI, bbox_inches="tight", facecolor="white")
    figures[output_stem] = fig
    figure_records.append({"png": str(png_path), "layout_rows_columns": [3, 2], "display_xlim_ms": [xmin_ms, xmax_ms]})
    print(f"Saved {png_path}")


# %% Provenance, definitions and counts
summary = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "script_path": str(SCRIPT_PATH),
    "script_sha256": hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest(),
    "numpy_version": np.__version__,
    "sources": source_records,
    "cohort_manifest": {"path": str(COHORT_MANIFEST), "sha256": hashlib.sha256(COHORT_MANIFEST.read_bytes()).hexdigest()},
    "original_summary": {"path": str(ORIGINAL_SUMMARY), "sha256": hashlib.sha256(ORIGINAL_SUMMARY.read_bytes()).hexdigest()},
    "cohort": [{"batch_name": batch, "animal": animal} for batch, animal in animal_pairs],
    "filters": {
        "rt_field": "RTwrtStim", "rt_reference": "stimulus onset; existing CSV values",
        "rt_window_s_inclusive": [RT_MIN_S, RT_MAX_S],
        "outcomes": "success in [-1, 1] OR abort_event == 3; valid outcomes take precedence",
        "ABL": ABLS, "abs_ILD": ABS_ILDS,
        "batch_preprocessing": "Preserved from the six original Figure 1 batch CSVs.",
    },
    "counts": {
        "source_rows": len(all_data), "selected_trials": len(plot_trials),
        "valid_choice_trials": int(plot_trials["is_valid_choice"].sum()), "simulated_abort_trials": n_aborts,
        "negative_rt_trials": int(plot_trials["RTwrtStim"].lt(0).sum()),
        "observed_correct": int(plot_trials["correct_valid"].sum()),
        "simulated_correct": int(plot_trials["correct_simulated"].sum()),
    },
    "simulation": {
        "seed": RANDOM_SEED, "bit_generator": type(rng.bit_generator).__name__,
        "call": "np.random.default_rng(seed).choice([-1, 1], size=n_aborts, p=[0.5, 0.5])",
        "choice_codes": {"-1": "left", "1": "right"},
        "draw_order": "DESIRED_BATCHES order, then original zero-based CSV data-row order, after trial selection",
        "correctness": "ILD * simulated_choice > 0 for aborts; success == 1 for valid choices",
        "source_accuracy_column": "Not read or used.",
        "reuse": "One draw per abort, reused in all bins and both PNG views.",
    },
    "binning": {"n_bins": n_bins, "width_s": BIN_WIDTH_S, "edges_s": bin_edges_s.tolist(), "boundary_rule": "[left, right), except the final bin includes 1 s", "positive_edges": "Copied exactly from the original 0–1 s analysis."},
    "aggregation": {
        "animal_id": ["batch_name", "animal"], "ILD_signs": "Pooled within each animal and condition.",
        "accuracy": "(observed correct valid choices + simulated correct abort choices) / all selected trials in each animal/condition/bin",
        "cdf": "Cumulative trial count / all selected trials in each animal/condition over the full -1 to 1 s window",
        "cdf_time_points": "Start of retained window then bin right edges; internal upper edges exclude the next bin, final edge includes 1 s.",
        "weighting": "Normalize within animal, then equal-animal means; SEM = sample SD (ddof=1) / sqrt(n contributing animals).",
        "missing_conditions": "Remain missing; six SD animals lack |ILD| = 16 dB.",
        "empty_accuracy_bins": "Missing mean and SEM; leave a gap.", "singleton_accuracy_bins": "Mean shown; SEM missing.",
        "empty_cdf_bins": "Cumulative fraction stays constant and all condition-available animals still contribute.",
        "display_limits": "Only crop axes; never recalculate or renormalize curves for a view.",
    },
    "outputs": {"tables": {name: str(path) for name, path in table_paths.items()}, "figures": figure_records},
}
(OUTPUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

if SHOW_PLOT:
    plt.show()
else:
    for fig in figures.values():
        plt.close(fig)
