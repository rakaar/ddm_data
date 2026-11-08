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
ABL_COLORS = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}
RT_MIN_S = 0.0
RT_MAX_S = 1.0
BIN_WIDTH_S = 0.020
PNG_DPI = 300
SHOW_PLOT = False

SCRIPT_PATH = Path(__file__).resolve()
REPO_DIR = SCRIPT_PATH.parent.parent
CSV_DIR = REPO_DIR / "raw_data" / "batch_csvs"
COHORT_MANIFEST = REPO_DIR / "all_30_fit_consolidation_manifest.md"
OUTPUT_DIR = SCRIPT_PATH.parent / "fig1_tachometric_by_abl_abs_ild"
OUTPUT_STEM = "fig1_tachometric_by_abl_abs_ild"


# %% Load the existing Figure 1 batch CSVs and verify the paper cohort
manifest_text = COHORT_MANIFEST.read_text()
expected_animals = {
    (batch, int(animal))
    for batch, animal in re.findall(
        r"^- `([^/`]+)/([0-9]+)`: source=", manifest_text, flags=re.MULTILINE
    )
}
if len(expected_animals) != 30:
    raise ValueError(f"Expected 30 animals in {COHORT_MANIFEST}; found {len(expected_animals)}.")

data_frames = []
source_records = []
for batch in DESIRED_BATCHES:
    csv_path = CSV_DIR / f"batch_{batch}_valid_and_aborts.csv"
    batch_data = pd.read_csv(
        csv_path, usecols=["batch_name", "animal", "success", "RTwrtStim", "ABL", "ILD"]
    )
    if set(batch_data["batch_name"].unique()) != {batch}:
        raise ValueError(f"Unexpected batch_name values in {csv_path}.")
    data_frames.append(batch_data)
    with csv_path.open("rb") as source_file:
        source_sha256 = hashlib.file_digest(source_file, "sha256").hexdigest()
    source_records.append(
        {
            "batch": batch,
            "path": str(csv_path),
            "sha256": source_sha256,
            "n_rows": len(batch_data),
            "n_valid_trials": int(batch_data["success"].isin([1, -1]).sum()),
        }
    )

all_data = pd.concat(data_frames, ignore_index=True)
observed_animals = {
    (batch, int(animal))
    for batch, animal in all_data[["batch_name", "animal"]].drop_duplicates().itertuples(index=False, name=None)
}
if observed_animals != expected_animals:
    raise ValueError(
        f"Paper cohort mismatch: missing={sorted(expected_animals - observed_animals)}, "
        f"extra={sorted(observed_animals - expected_animals)}."
    )
animal_pairs = sorted(observed_animals)
print(f"Paper cohort verified: {len(animal_pairs)} animals; zero missing or extra.")


# %% Select valid trials using the existing stimulus-relative reaction times
valid_trials = all_data.loc[all_data["success"].isin([1, -1])].copy()
rt_trials = valid_trials.loc[
    valid_trials["RTwrtStim"].between(RT_MIN_S, RT_MAX_S, inclusive="both")
].copy()
plot_trials = rt_trials.loc[
    rt_trials["ABL"].isin(ABLS) & rt_trials["ILD"].abs().isin(ABS_ILDS)
].copy()
plot_trials["abs_ILD"] = plot_trials["ILD"].abs().astype(int)
plot_trials["correct"] = plot_trials["success"].eq(1).astype(int)

selected_animals = set(
    plot_trials[["batch_name", "animal"]].itertuples(index=False, name=None)
)
if selected_animals != expected_animals:
    raise ValueError("At least one paper animal has no trials after the plotting filters.")

n_bins = int(round((RT_MAX_S - RT_MIN_S) / BIN_WIDTH_S))
if not np.isclose(n_bins * BIN_WIDTH_S, RT_MAX_S - RT_MIN_S):
    raise ValueError("The RT window must contain an integer number of bins.")
bin_edges_s = np.linspace(RT_MIN_S, RT_MAX_S, n_bins + 1)
bin_centers_s = (bin_edges_s[:-1] + bin_edges_s[1:]) / 2

# Left-inclusive bins; the final bin also contains RT_MAX_S.
plot_trials["bin_index"] = np.minimum(
    np.searchsorted(bin_edges_s, plot_trials["RTwrtStim"].to_numpy(), side="right") - 1,
    n_bins - 1,
)
print(
    f"Trials: {len(all_data):,} source rows -> {len(valid_trials):,} valid outcomes "
    f"-> {len(rt_trials):,} in the RT window -> {len(plot_trials):,} plotted."
)


# %% Accuracy within each animal, then equal-animal means and sample SEMs
animal_keys = ["batch_name", "animal", "ABL", "abs_ILD", "bin_index"]
animal_bins = plot_trials.groupby(animal_keys)["correct"].agg(
    n_trials="size", n_correct="sum"
)

# Explicit zero-count rows retain missing animals/conditions/bins in the audit.
complete_index = pd.MultiIndex.from_tuples(
    [
        (batch, animal, abl, abs_ild, bin_index)
        for batch, animal in animal_pairs
        for abl in ABLS
        for abs_ild in ABS_ILDS
        for bin_index in range(n_bins)
    ],
    names=animal_keys,
)
animal_bins = animal_bins.reindex(complete_index, fill_value=0).reset_index()
animal_bins["accuracy"] = animal_bins["n_correct"] / animal_bins["n_trials"].replace(0, np.nan)

group_bins = animal_bins.groupby(["ABL", "abs_ILD", "bin_index"], as_index=False).agg(
    mean_accuracy=("accuracy", "mean"),
    std_accuracy=("accuracy", "std"),  # pandas uses the sample SD, ddof=1.
    n_animals=("accuracy", "count"),
    n_trials=("n_trials", "sum"),
    n_correct=("n_correct", "sum"),
)
group_bins["sem_accuracy"] = group_bins["std_accuracy"] / np.sqrt(group_bins["n_animals"])

for table in [animal_bins, group_bins]:
    indices = table["bin_index"].to_numpy()
    table["bin_start_s"] = bin_edges_s[indices]
    table["bin_end_s"] = bin_edges_s[indices + 1]
    table["bin_center_s"] = bin_centers_s[indices]

condition_records = []
for abl in ABLS:
    for abs_ild in ABS_ILDS:
        condition_trials = plot_trials.loc[
            plot_trials["ABL"].eq(abl) & plot_trials["abs_ILD"].eq(abs_ild)
        ]
        contributors = set(
            condition_trials[["batch_name", "animal"]].itertuples(index=False, name=None)
        )
        condition_bins = group_bins.loc[
            group_bins["ABL"].eq(abl) & group_bins["abs_ILD"].eq(abs_ild)
        ]
        condition_records.append(
            {
                "ABL": abl,
                "abs_ILD": abs_ild,
                "n_animals_with_trials": len(contributors),
                "animals_without_trials": [
                    {"batch_name": batch, "animal": animal}
                    for batch, animal in sorted(expected_animals - contributors)
                ],
                "n_trials": len(condition_trials),
                "min_animals_per_bin": int(condition_bins["n_animals"].min()),
                "max_animals_per_bin": int(condition_bins["n_animals"].max()),
                "n_empty_bins": int(condition_bins["n_animals"].eq(0).sum()),
                "n_singleton_bins": int(condition_bins["n_animals"].eq(1).sum()),
            }
        )

assert int(animal_bins["n_trials"].sum()) == len(plot_trials)
assert int(group_bins["n_correct"].sum()) == int(plot_trials["correct"].sum())
assert group_bins.loc[group_bins["n_animals"].lt(2), "sem_accuracy"].isna().all()


# %% Save the numerical results and provenance
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
animal_csv = OUTPUT_DIR / "animal_bin_accuracy.csv"
group_csv = OUTPUT_DIR / "group_bin_summary.csv"
summary_path = OUTPUT_DIR / "run_summary.json"
png_path = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
pdf_path = OUTPUT_DIR / f"{OUTPUT_STEM}.pdf"
animal_bins.to_csv(animal_csv, index=False)
group_bins.to_csv(group_csv, index=False)

summary = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "script_path": str(SCRIPT_PATH),
    "script_sha256": hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest(),
    "cohort_manifest_path": str(COHORT_MANIFEST),
    "cohort_manifest_sha256": hashlib.sha256(COHORT_MANIFEST.read_bytes()).hexdigest(),
    "sources": source_records,
    "cohort": [{"batch_name": batch, "animal": animal} for batch, animal in animal_pairs],
    "cohort_comparison": {"n_expected": 30, "n_observed": len(animal_pairs), "missing": [], "extra": []},
    "filters": {
        "success": [1, -1],
        "rt_field": "RTwrtStim",
        "rt_reference": "stimulus onset; use the existing CSV values",
        "rt_window_s_inclusive": [RT_MIN_S, RT_MAX_S],
        "ABL": ABLS,
        "abs_ILD": ABS_ILDS,
        "batch_preprocessing": "Preserved from the existing six Figure 1 batch CSVs.",
    },
    "counts": {
        "source_rows": len(all_data),
        "valid_outcomes": len(valid_trials),
        "valid_in_rt_window": len(rt_trials),
        "plotted_trials": len(plot_trials),
        "plotted_correct": int(plot_trials["correct"].sum()),
        "excluded_by_rt_window": len(valid_trials) - len(rt_trials),
        "excluded_by_ABL_or_abs_ILD": len(rt_trials) - len(plot_trials),
    },
    "binning": {
        "width_s": BIN_WIDTH_S,
        "n_bins": n_bins,
        "edges_s": bin_edges_s.tolist(),
        "boundary_rule": "[left, right), except the final bin includes its right edge",
        "plotted_x": "bin centers in milliseconds",
    },
    "aggregation": {
        "accuracy": "n(success == 1) / n(success in [1, -1]) within animal, ABL, |ILD|, RT bin",
        "ILD_signs": "Pool positive and negative ILD trials within each animal before calculating accuracy.",
        "animal_weighting": "Equal weight for animals with at least one trial in the bin.",
        "sem": "Sample SD (ddof=1) of animal accuracies / sqrt(n contributing animals)",
        "empty_bin": "Mean and SEM missing; leave a gap in the curve.",
        "singleton_bin": "Show the animal accuracy; SEM missing and no shading.",
        "minimum_trials_per_animal_bin": 1,
        "smoothing": "None",
    },
    "conditions": condition_records,
    "outputs": {
        "animal_bin_accuracy": str(animal_csv),
        "group_bin_summary": str(group_csv),
        "summary": str(summary_path),
        "png": str(png_path),
        "pdf": str(pdf_path),
    },
}


# %% Plot the 3 x 5 grid
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
fig, axes = plt.subplots(
    len(ABLS), len(ABS_ILDS), figsize=(17, 10), sharex=True, sharey=True, squeeze=False
)
fig.subplots_adjust(left=0.07, right=0.985, bottom=0.17, top=0.835, hspace=0.22, wspace=0.14)

for row, abl in enumerate(ABLS):
    for col, abs_ild in enumerate(ABS_ILDS):
        ax = axes[row, col]
        condition = group_bins.loc[
            group_bins["ABL"].eq(abl) & group_bins["abs_ILD"].eq(abs_ild)
        ].sort_values("bin_index")
        x_ms = condition["bin_center_s"].to_numpy() * 1000
        mean = condition["mean_accuracy"].to_numpy()
        sem = condition["sem_accuracy"].to_numpy()
        ax.fill_between(x_ms, mean - sem, mean + sem, color=ABL_COLORS[abl], alpha=0.22, linewidth=0)
        ax.plot(x_ms, mean, color=ABL_COLORS[abl], linewidth=1.7, marker="o", markersize=2.6)
        ax.axhline(0.5, color="0.6", linestyle="--", linewidth=0.8, zorder=0)
        ax.set_xlim(RT_MIN_S * 1000, RT_MAX_S * 1000)
        ax.set_ylim(0, 1)
        ax.set_xticks(np.linspace(RT_MIN_S * 1000, RT_MAX_S * 1000, 5))
        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(labelsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        if row == 0:
            n_condition_animals = next(
                record["n_animals_with_trials"]
                for record in condition_records
                if record["ABL"] == abl and record["abs_ILD"] == abs_ild
            )
            ax.set_title(f"|ILD| = {abs_ild} dB\nUp to {n_condition_animals} animals", fontsize=13, pad=12)
        if col == 0:
            ax.set_ylabel(f"ABL = {abl} dB\nAccuracy", fontsize=13)

fig.suptitle("Tachometric curves: Figure 1 cohort (30 animals)", fontsize=19, y=0.975)
fig.text(
    0.5, 0.932,
    f"Equal-animal mean ± SEM  |  {BIN_WIDTH_S * 1000:g} ms bins  |  Positive and negative ILDs pooled within animal",
    ha="center", fontsize=12,
)
fig.supxlabel("Reaction time from stimulus onset (ms)", fontsize=14, y=0.105)
fig.text(
    0.5, 0.035,
    "Animals with trials in each bin contribute equally; contributor counts vary with reaction time.\n"
    "At |ILD| = 16 dB, the six SD animals have no trials (maximum n = 24). Empty bins are gaps; singleton bins have no SEM.",
    ha="center", fontsize=10, linespacing=1.6,
)
fig.savefig(png_path, dpi=PNG_DPI, bbox_inches="tight", facecolor="white")
fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
print(f"Saved figure and numerical audits to {OUTPUT_DIR}")

if SHOW_PLOT:
    plt.show()
else:
    plt.close(fig)
