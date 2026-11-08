# %% Imports
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# %% Parameters (edit here)
ABLS = [20, 40, 60]
ABS_ILDS = [1, 2, 4, 8, 16]
ABL_COLORS = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}
ABS_ILD_COLORS = {1: "tab:blue", 2: "tab:orange", 4: "tab:green", 8: "tab:red", 16: "tab:purple"}
ABL_PANEL_XMAX_MS = 600  # Display limit for the 3 x 2 figure; retain the full 0–1 s dataset.
PNG_DPI = 300
SHOW_PLOT = False

SCRIPT_PATH = Path(__file__).resolve()
OUTPUT_DIR = SCRIPT_PATH.parent / "fig1_tachometric_by_abl_abs_ild"
ANIMAL_CSV = OUTPUT_DIR / "animal_bin_accuracy.csv"
GROUP_CSV = OUTPUT_DIR / "group_bin_summary.csv"
SOURCE_SUMMARY = OUTPUT_DIR / "run_summary.json"


# %% Read the verified Figure 1 counts and the existing accuracy curves
animal_bins = pd.read_csv(ANIMAL_CSV)
accuracy_bins = pd.read_csv(GROUP_CSV)
source_summary = json.loads(SOURCE_SUMMARY.read_text())
rt_min_s, rt_max_s = source_summary["filters"]["rt_window_s_inclusive"]
bin_width_ms = source_summary["binning"]["width_s"] * 1000
animal_keys = ["batch_name", "animal", "ABL", "abs_ILD"]
animal_bins = animal_bins.sort_values(animal_keys + ["bin_index"]).reset_index(drop=True)

expected_conditions = {(abl, ild) for abl in ABLS for ild in ABS_ILDS}
observed_conditions = set(accuracy_bins[["ABL", "abs_ILD"]].itertuples(index=False, name=None))
if observed_conditions != expected_conditions:
    raise ValueError("The saved curves do not match the requested three ABLs and five |ILD|s.")
if int(animal_bins["n_trials"].sum()) != source_summary["counts"]["plotted_trials"]:
    raise ValueError("Saved animal-bin counts disagree with the original trial-count audit.")


# %% Cumulative RTD within each animal and condition, then mean and sample SEM
# The denominator includes every retained RT bin, not only the bin at time t.
animal_cdf = animal_bins[animal_keys + ["bin_index", "bin_end_s", "n_trials"]].copy()
animal_groups = animal_cdf.groupby(animal_keys, sort=False)
animal_cdf["n_condition_trials"] = animal_groups["n_trials"].transform("sum")
animal_cdf["n_cumulative_trials"] = animal_groups["n_trials"].cumsum()
animal_cdf["cumulative_fraction"] = (
    animal_cdf["n_cumulative_trials"] / animal_cdf["n_condition_trials"].replace(0, np.nan)
)
animal_cdf = animal_cdf.rename(columns={"bin_end_s": "time_s"})

# Add the start of the binned CDF: zero before accumulating the first bin.
start_points = animal_cdf.groupby(animal_keys, as_index=False).first()
start_points["bin_index"] = -1
start_points["time_s"] = rt_min_s
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
group_cdf["sem_cumulative_fraction"] = (
    group_cdf["std_cumulative_fraction"] / np.sqrt(group_cdf["n_animals"])
)
assert np.allclose(group_cdf.loc[group_cdf["time_s"].eq(rt_max_s), "mean_cumulative_fraction"], 1)
assert group_cdf.groupby(["ABL", "abs_ILD"])["n_animals"].nunique().eq(1).all()

animal_cdf_path = OUTPUT_DIR / "animal_cumulative_rtd.csv"
group_cdf_path = OUTPUT_DIR / "group_cumulative_rtd.csv"
animal_cdf.to_csv(animal_cdf_path, index=False)
group_cdf.to_csv(group_cdf_path, index=False)


# %% Pair cumulative RTD and accuracy in both overlay arrangements
plt.rcParams.update({"font.family": "DejaVu Sans"})
field_labels = {"ABL": "ABL", "abs_ILD": "|ILD|"}
layout_specs = [
    {
        "panel_field": "abs_ILD", "panel_values": ABS_ILDS,
        "curve_field": "ABL", "curve_values": ABLS, "colors": ABL_COLORS,
        "title": "Cumulative reaction times and accuracy at each |ILD|",
        "output_stem": "fig1_cdf_tachometric_abl_overlay_by_abs_ild",
    },
    {
        "panel_field": "ABL", "panel_values": ABLS,
        "curve_field": "abs_ILD", "curve_values": ABS_ILDS, "colors": ABS_ILD_COLORS,
        "title": "Cumulative reaction times and accuracy at each ABL",
        "output_stem": "fig1_cdf_tachometric_abs_ild_overlay_by_abl",
    },
]
figures = {}
output_records = []
for spec in layout_specs:
    wide_layout = spec["panel_field"] == "abs_ILD"
    display_max_ms = rt_max_s * 1000 if wide_layout else ABL_PANEL_XMAX_MS
    n_rows, n_cols = (2, len(ABS_ILDS)) if wide_layout else (len(ABLS), 2)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(19, 8) if wide_layout else (13, 12),
        sharex=True, sharey=True, squeeze=False,
    )
    fig.subplots_adjust(
        left=0.06 if wide_layout else 0.075, right=0.99,
        bottom=0.23 if wide_layout else 0.195, top=0.85 if wide_layout else 0.90,
        hspace=0.25, wspace=0.14 if wide_layout else 0.22,
    )

    for panel_index, panel_value in enumerate(spec["panel_values"]):
        cdf_ax, accuracy_ax = (
            (axes[0, panel_index], axes[1, panel_index]) if wide_layout
            else (axes[panel_index, 0], axes[panel_index, 1])
        )
        for curve_value in spec["curve_values"]:
            cdf = group_cdf.loc[
                group_cdf[spec["panel_field"]].eq(panel_value)
                & group_cdf[spec["curve_field"]].eq(curve_value)
            ].sort_values("bin_index")
            accuracy = accuracy_bins.loc[
                accuracy_bins[spec["panel_field"]].eq(panel_value)
                & accuracy_bins[spec["curve_field"]].eq(curve_value)
            ].sort_values("bin_index")
            color = spec["colors"][curve_value]
            for ax, x_ms, mean, sem, marker in [
                (
                    cdf_ax, cdf["time_s"].to_numpy() * 1000,
                    cdf["mean_cumulative_fraction"].to_numpy(),
                    cdf["sem_cumulative_fraction"].to_numpy(), None,
                ),
                (
                    accuracy_ax, accuracy["bin_center_s"].to_numpy() * 1000,
                    accuracy["mean_accuracy"].to_numpy(),
                    accuracy["sem_accuracy"].to_numpy(), "o",
                ),
            ]:
                ax.fill_between(x_ms, mean - sem, mean + sem, color=color, alpha=0.16, linewidth=0)
                ax.plot(
                    x_ms, mean, color=color, linewidth=1.7,
                    marker=marker, markersize=2, label=f"{curve_value} dB",
                )

        accuracy_ax.axhline(0.5, color="0.6", linestyle="--", linewidth=0.8, zorder=0)
        for ax in [cdf_ax, accuracy_ax]:
            ax.set_xlim(rt_min_s * 1000, display_max_ms)
            ax.set_ylim(0, 1)
            ax.set_xticks(np.linspace(rt_min_s * 1000, display_max_ms, 5 if wide_layout else 7))
            ax.set_yticks([0, 0.5, 1])
            ax.tick_params(labelsize=11)
            ax.spines[["top", "right"]].set_visible(False)
        panel_title = f"{field_labels[spec['panel_field']]} = {panel_value} dB"
        if wide_layout:
            cdf_ax.set_title(panel_title, fontsize=14, pad=12)
            if panel_index == 0:
                cdf_ax.set_ylabel("Cumulative fraction of trials", fontsize=12)
                accuracy_ax.set_ylabel("Accuracy", fontsize=13)
        else:
            cdf_ax.set_title(f"{panel_title}: cumulative RTD", fontsize=13, pad=10)
            accuracy_ax.set_title(f"{panel_title}: accuracy", fontsize=13, pad=10)
            cdf_ax.set_ylabel("Cumulative fraction of trials", fontsize=12)
            accuracy_ax.set_ylabel("Accuracy", fontsize=13)
            accuracy_ax.tick_params(labelleft=True)

    fig.suptitle(spec["title"], fontsize=19 if wide_layout else 18, y=0.98)
    fig.text(
        0.5, 0.935 if wide_layout else 0.948,
        f"Same Figure 1 trials and conditions  |  {bin_width_ms:g} ms bins  |  Equal-animal mean ± SEM",
        ha="center", fontsize=12,
    )
    fig.supxlabel("Reaction time from stimulus onset (ms)", fontsize=14, y=0.15 if wide_layout else 0.13)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, title=f"{field_labels[spec['curve_field']]} (dB)",
        loc="lower center", bbox_to_anchor=(0.5, 0.06),
        ncol=len(spec["curve_values"]), frameon=False, fontsize=11, title_fontsize=11,
    )
    fig.text(
        0.5, 0.005,
        "CDF denominator: each animal's valid 0–1 s trials in that condition; accuracy denominator: trials in each time bin.\n"
        "|ILD| = 1–8 dB: 30 animals; 16 dB: 24. CDF averages all available animals; accuracy contributors vary by time bin.",
        ha="center", fontsize=9.5, linespacing=1.5,
    )
    png_path = OUTPUT_DIR / f"{spec['output_stem']}.png"
    fig.savefig(png_path, dpi=PNG_DPI, bbox_inches="tight", facecolor="white")
    figures[spec["output_stem"]] = fig
    output_records.append({
        "png": str(png_path), "layout_rows_columns": [n_rows, n_cols],
        "display_xlim_ms": [rt_min_s * 1000, display_max_ms],
    })
    print(f"Saved {png_path}")


# %% Save the CDF definition and source provenance
summary = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "script_path": str(SCRIPT_PATH),
    "script_sha256": hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest(),
    "inputs": [
        {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in [ANIMAL_CSV, GROUP_CSV, SOURCE_SUMMARY]
    ],
    "source_trial_count": source_summary["counts"]["plotted_trials"],
    "filters": source_summary["filters"],
    "cohort": source_summary["cohort"],
    "cdf_definition": {
        "resolution_ms": bin_width_ms,
        "numerator": "Cumulative n_trials through the displayed bin for each animal, ABL and |ILD|.",
        "denominator": "All retained valid 0–1 s trials for that animal, ABL and |ILD|.",
        "time_points": "Start of the retained window, then bin right edges; internal upper edges exclude the next bin, final edge includes 1 s.",
        "weighting": "Normalize within each animal, then equal-animal mean and sample SEM (ddof=1).",
        "missing_conditions": "Missing throughout, not assigned zero CDF; six SD animals lack |ILD| = 16 dB.",
        "empty_time_bins": "Cumulative fraction stays at its previous value; animals still contribute.",
        "conditioning": "CDF is conditional on the retained valid RT window, with endpoint 1 at 1 s.",
        "condition_pooling": "None; preserve all 15 ABL x |ILD| conditions.",
    },
    "accuracy": "All original bin centers, mean accuracies, SEMs and gaps copied without modification.",
    "outputs": {
        "animal_cumulative_rtd": str(animal_cdf_path),
        "group_cumulative_rtd": str(group_cdf_path),
        "figures": output_records,
    },
}
(OUTPUT_DIR / "cumulative_rtd_run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

if SHOW_PLOT:
    plt.show()
else:
    for fig in figures.values():
        plt.close(fig)
