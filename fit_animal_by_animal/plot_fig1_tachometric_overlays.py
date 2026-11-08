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
PNG_DPI = 300
SHOW_PLOT = False

SCRIPT_PATH = Path(__file__).resolve()
OUTPUT_DIR = SCRIPT_PATH.parent / "fig1_tachometric_by_abl_abs_ild"
GROUP_CSV = OUTPUT_DIR / "group_bin_summary.csv"
SOURCE_SUMMARY = OUTPUT_DIR / "run_summary.json"


# %% Read the original curves; this script only changes their arrangement
group_bins = pd.read_csv(GROUP_CSV)
source_summary = json.loads(SOURCE_SUMMARY.read_text())
rt_min_s, rt_max_s = source_summary["filters"]["rt_window_s_inclusive"]
bin_width_ms = source_summary["binning"]["width_s"] * 1000
field_labels = {"ABL": "ABL", "abs_ILD": "|ILD|"}

expected_conditions = {(abl, ild) for abl in ABLS for ild in ABS_ILDS}
observed_conditions = set(group_bins[["ABL", "abs_ILD"]].itertuples(index=False, name=None))
if observed_conditions != expected_conditions:
    raise ValueError("The saved curves do not match the requested three ABLs and five |ILD|s.")
if group_bins.duplicated(["ABL", "abs_ILD", "bin_index"]).any():
    raise ValueError("Duplicate condition/bin rows in the saved tachometric curves.")

overlay_specs = [
    {
        "panel_field": "abs_ILD",
        "panel_values": ABS_ILDS,
        "curve_field": "ABL",
        "curve_values": ABLS,
        "colors": ABL_COLORS,
        "title": "Tachometric curves: three ABLs overlaid at each |ILD|",
        "output_stem": "fig1_tachometric_abl_overlay_by_abs_ild",
    },
    {
        "panel_field": "ABL",
        "panel_values": ABLS,
        "curve_field": "abs_ILD",
        "curve_values": ABS_ILDS,
        "colors": ABS_ILD_COLORS,
        "title": "Tachometric curves: five |ILD|s overlaid at each ABL",
        "output_stem": "fig1_tachometric_abs_ild_overlay_by_abl",
    },
]


# %% Draw both layouts with the original bin centers, means, and SEMs
plt.rcParams.update({"font.family": "DejaVu Sans"})
figures = {}
output_records = []
for spec in overlay_specs:
    wide_layout = spec["panel_field"] == "abs_ILD"
    n_rows, n_cols = (1, len(spec["panel_values"])) if wide_layout else (len(spec["panel_values"]), 1)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(19, 5.2) if wide_layout else (8.5, 12),
        sharex=True, sharey=True, squeeze=False,
    )
    if wide_layout:
        fig.subplots_adjust(left=0.055, right=0.99, bottom=0.31, top=0.77, wspace=0.14)
    else:
        fig.subplots_adjust(left=0.12, right=0.97, bottom=0.20, top=0.895, hspace=0.24)

    for panel_value, ax in zip(spec["panel_values"], axes.flat):
        for curve_value in spec["curve_values"]:
            curve = group_bins.loc[
                group_bins[spec["panel_field"]].eq(panel_value)
                & group_bins[spec["curve_field"]].eq(curve_value)
            ].sort_values("bin_index")
            x_ms = curve["bin_center_s"].to_numpy() * 1000
            mean = curve["mean_accuracy"].to_numpy()
            sem = curve["sem_accuracy"].to_numpy()
            color = spec["colors"][curve_value]
            ax.fill_between(x_ms, mean - sem, mean + sem, color=color, alpha=0.16, linewidth=0)
            ax.plot(
                x_ms, mean, color=color, linewidth=1.6,
                marker="o", markersize=2.3, label=f"{curve_value} dB",
            )
        ax.axhline(0.5, color="0.6", linestyle="--", linewidth=0.8, zorder=0)
        ax.set_xlim(rt_min_s * 1000, rt_max_s * 1000)
        ax.set_ylim(0, 1)
        ax.set_xticks(np.linspace(rt_min_s * 1000, rt_max_s * 1000, 5))
        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(labelsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"{field_labels[spec['panel_field']]} = {panel_value} dB", fontsize=14, pad=10)
        if not wide_layout or ax is axes[0, 0]:
            ax.set_ylabel("Accuracy", fontsize=13)

    fig.suptitle(spec["title"], fontsize=18 if wide_layout else 16, y=0.98)
    fig.text(
        0.5, 0.91 if wide_layout else 0.945,
        f"Original equal-animal mean ± SEM  |  {bin_width_ms:g} ms bins  |  No aggregation across conditions",
        ha="center", fontsize=12 if wide_layout else 10.5,
    )
    fig.supxlabel("Reaction time from stimulus onset (ms)", fontsize=14, y=0.18 if wide_layout else 0.13)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, title=f"{field_labels[spec['curve_field']]} (dB)",
        loc="lower center", bbox_to_anchor=(0.5, 0.055 if wide_layout else 0.06),
        ncol=len(spec["curve_values"]), frameon=False, fontsize=11, title_fontsize=11,
    )
    footnote = (
        "Up to 30 animals for |ILD| = 1–8 dB; up to 24 for 16 dB (six SD animals lack this condition).\n"
        "Counts vary by time bin; empty bins are gaps and singleton bins have no SEM."
    )
    fig.text(0.5, 0.005, footnote, ha="center", fontsize=9.5, linespacing=1.5)

    png_path = OUTPUT_DIR / f"{spec['output_stem']}.png"
    fig.savefig(png_path, dpi=PNG_DPI, bbox_inches="tight", facecolor="white")
    figures[spec["output_stem"]] = fig
    output_records.append(
        {
            "png": str(png_path),
            "panel_field": spec["panel_field"],
            "panel_values": spec["panel_values"],
            "colored_curve_field": spec["curve_field"],
            "colored_curve_values": spec["curve_values"],
            "layout_rows_columns": [n_rows, n_cols],
        }
    )
    print(f"Saved {png_path}")


# %% Record the exact saved curves used to make these layout-only figures
summary = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "script_path": str(SCRIPT_PATH),
    "script_sha256": hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest(),
    "source_group_summary_path": str(GROUP_CSV),
    "source_group_summary_sha256": hashlib.sha256(GROUP_CSV.read_bytes()).hexdigest(),
    "source_run_summary_path": str(SOURCE_SUMMARY),
    "source_run_summary_sha256": hashlib.sha256(SOURCE_SUMMARY.read_bytes()).hexdigest(),
    "operation": "Overlay existing condition curves; no trial pooling or averaging across conditions.",
    "values": "Copy bin centers, mean_accuracy, and sem_accuracy directly from group_bin_summary.csv.",
    "source_condition_bin_rows": len(group_bins),
    "source_trial_count": source_summary["counts"]["plotted_trials"],
    "cohort": source_summary["cohort"],
    "outputs": output_records,
}
(OUTPUT_DIR / "overlay_run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

if SHOW_PLOT:
    plt.show()
else:
    for fig in figures.values():
        plt.close(fig)
