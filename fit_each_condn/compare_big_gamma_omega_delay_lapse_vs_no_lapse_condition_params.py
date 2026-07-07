# %%
"""
Compare completed condition-wise Gamma/Omega/delay SVI fits with and without lapses.

The comparison uses each animal's posterior mean for each fitted condition.
Error bars are SEM across animals. For the middle Gamma panel, each animal is
first averaged across available ABLs at each ILD, then SEM is computed across
animals so ABLs are not treated as independent observations.
"""

# %%
# =============================================================================
# Parameters
# =============================================================================
from pathlib import Path
import os
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
NO_LAPSE_ROOT = SCRIPT_DIR / "svi_big_gamma_omega_delay_patience12_restore_best_all_animals_outputs"
LAPSE_ROOT = SCRIPT_DIR / "svi_big_gamma_omega_delay_lapse_patience12_restore_best_all_animals_outputs"

SUMMARY_DIR = LAPSE_ROOT / "summary_figures"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

FIG_PATH = SUMMARY_DIR / "big_gamma_omega_delay_lapse_vs_no_lapse_gamma_omega_1x3.png"
ANIMAL_VALUES_CSV = SUMMARY_DIR / "big_gamma_omega_delay_lapse_vs_no_lapse_condition_param_animal_values.csv"
PLOT_SUMMARY_CSV = SUMMARY_DIR / "big_gamma_omega_delay_lapse_vs_no_lapse_gamma_omega_1x3_summary.csv"

EXPECTED_N_ANIMALS = 30
EXPECTED_N_CONDITION_ROWS = 864
ABLS = [20, 40, 60]
ILDS = np.array([-16, -8, -4, -2, -1, 1, 2, 4, 8, 16], dtype=int)
COLORS = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}
MODEL_SPECS = {
    "No lapse": {
        "root": NO_LAPSE_ROOT,
        "file_suffix": "_big_gamma_omega_delay_condition_summary.csv",
        "prefix_suffix": "_big_gamma_omega_delay",
        "marker": ".",
        "linestyle": "-",
        "alpha": 0.85,
    },
    "Lapse": {
        "root": LAPSE_ROOT,
        "file_suffix": "_big_gamma_omega_delay_lapse_condition_summary.csv",
        "prefix_suffix": "_big_gamma_omega_delay_lapse",
        "marker": "x",
        "linestyle": "--",
        "alpha": 0.9,
    },
}


# %%
# =============================================================================
# Load and validate condition summaries
# =============================================================================
all_model_dfs = []
for model_name, spec in MODEL_SPECS.items():
    root = spec["root"]
    summary_paths = sorted(root.glob(f"*/*{spec['file_suffix']}"))
    if len(summary_paths) != EXPECTED_N_ANIMALS:
        raise RuntimeError(f"{model_name}: expected {EXPECTED_N_ANIMALS} condition summaries, found {len(summary_paths)}")

    animal_dfs = []
    for summary_path in summary_paths:
        match = re.match(r"^(?P<batch>.+)_(?P<animal>\d+)$", summary_path.parent.name)
        if match is None:
            raise RuntimeError(f"{model_name}: could not parse animal folder name {summary_path.parent}")

        df = pd.read_csv(summary_path)
        required_cols = ["batch_name", "animal", "ABL", "ILD", "gamma_mean", "omega_mean"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise KeyError(f"{summary_path} missing columns: {missing_cols}")

        df = df[required_cols].copy()
        df["model"] = model_name
        df["animal"] = df["animal"].astype(int)
        df["ABL"] = df["ABL"].astype(int)
        df["ILD"] = df["ILD"].astype(int)
        animal_dfs.append(df)

    model_df = pd.concat(animal_dfs, ignore_index=True)
    model_df = model_df.sort_values(["batch_name", "animal", "ABL", "ILD"]).reset_index(drop=True)

    n_animals = model_df[["batch_name", "animal"]].drop_duplicates().shape[0]
    if n_animals != EXPECTED_N_ANIMALS:
        raise RuntimeError(f"{model_name}: expected {EXPECTED_N_ANIMALS} animals, found {n_animals}")
    if len(model_df) != EXPECTED_N_CONDITION_ROWS:
        raise RuntimeError(f"{model_name}: expected {EXPECTED_N_CONDITION_ROWS} condition rows, found {len(model_df)}")
    duplicate_count = int(model_df.duplicated(["batch_name", "animal", "ABL", "ILD"]).sum())
    if duplicate_count:
        raise RuntimeError(f"{model_name}: found {duplicate_count} duplicate animal/condition rows")
    if not np.all(np.isfinite(model_df[["gamma_mean", "omega_mean"]].to_numpy(dtype=float))):
        raise RuntimeError(f"{model_name}: gamma/omega means contain NaN/Inf values")

    counts = model_df.groupby(["ABL", "ILD"])[["batch_name", "animal"]].apply(
        lambda values: values.drop_duplicates().shape[0]
    )
    for (abl, ild), n_animals_for_condition in counts.items():
        expected_n = 24 if abs(int(ild)) == 16 else 30
        if int(n_animals_for_condition) != expected_n:
            raise RuntimeError(
                f"{model_name}: ABL={abl}, ILD={ild} has n={n_animals_for_condition}, expected {expected_n}"
            )

    all_model_dfs.append(model_df)

condition_df = pd.concat(all_model_dfs, ignore_index=True)
no_lapse_animals = set(
    map(tuple, condition_df[condition_df["model"] == "No lapse"][["batch_name", "animal"]].drop_duplicates().to_numpy())
)
lapse_animals = set(
    map(tuple, condition_df[condition_df["model"] == "Lapse"][["batch_name", "animal"]].drop_duplicates().to_numpy())
)
if no_lapse_animals != lapse_animals:
    raise RuntimeError("No-lapse and lapse model roots do not contain the same animal set.")

condition_df.to_csv(ANIMAL_VALUES_CSV, index=False)
print(f"Loaded {len(no_lapse_animals)} animals per model")
print(f"Loaded {len(condition_df)} model/condition rows")
print(f"Saved animal-wise values: {ANIMAL_VALUES_CSV}")


# %%
# =============================================================================
# Summarize plotted values
# =============================================================================
summary_rows = []

for (model_name, abl, ild), group in condition_df.groupby(["model", "ABL", "ILD"], sort=True):
    n = group[["batch_name", "animal"]].drop_duplicates().shape[0]
    for param_name in ["gamma", "omega"]:
        values = group[f"{param_name}_mean"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "panel": f"{param_name}_by_abl",
                "model": model_name,
                "ABL": int(abl),
                "ILD": int(ild),
                "n_animals": int(n),
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)) if n > 1 else np.nan,
                "sem": float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
            }
        )

gamma_by_animal_ablavg = (
    condition_df.groupby(["model", "batch_name", "animal", "ILD"], as_index=False)["gamma_mean"].mean()
)
for (model_name, ild), group in gamma_by_animal_ablavg.groupby(["model", "ILD"], sort=True):
    n = group[["batch_name", "animal"]].drop_duplicates().shape[0]
    values = group["gamma_mean"].to_numpy(dtype=float)
    summary_rows.append(
        {
            "panel": "gamma_abl_average",
            "model": model_name,
            "ABL": np.nan,
            "ILD": int(ild),
            "n_animals": int(n),
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)) if n > 1 else np.nan,
            "sem": float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
        }
    )

plot_summary_df = pd.DataFrame(summary_rows).sort_values(["panel", "model", "ABL", "ILD"]).reset_index(drop=True)
plot_summary_df.to_csv(PLOT_SUMMARY_CSV, index=False)

for row in plot_summary_df.itertuples(index=False):
    expected_n = 24 if abs(int(row.ILD)) == 16 else 30
    if int(row.n_animals) != expected_n:
        raise RuntimeError(f"{row.panel}, {row.model}, ILD={row.ILD} has n={row.n_animals}, expected {expected_n}")

print(f"Saved plot summary: {PLOT_SUMMARY_CSV}")


# %%
# =============================================================================
# Plot 1 x 3 comparison
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9), sharex=True)
x_offset = {"No lapse": -0.08, "Lapse": 0.08}

for model_name, spec in MODEL_SPECS.items():
    model_summary = plot_summary_df[plot_summary_df["model"] == model_name]
    for abl in ABLS:
        for ax, panel, ylabel in [
            (axes[0], "gamma_by_abl", "Gamma"),
            (axes[2], "omega_by_abl", "Omega"),
        ]:
            abl_df = model_summary[(model_summary["panel"] == panel) & (model_summary["ABL"] == abl)].sort_values("ILD")
            if panel == "gamma_by_abl":
                plot_marker = "." if model_name == "No lapse" else "x"
                plot_linestyle = "none"
                plot_alpha = 0.95 if model_name == "No lapse" else 0.55
                plot_markersize = 5.0 if model_name == "No lapse" else 7.5
                plot_linewidth = 0.0
                plot_elinewidth = 0.9
                plot_capsize = 1.6
            else:
                plot_marker = spec["marker"]
                plot_linestyle = spec["linestyle"]
                plot_alpha = spec["alpha"]
                plot_markersize = 5.5 if model_name == "Lapse" else 7.0
                plot_linewidth = 1.1
                plot_elinewidth = 1.0
                plot_capsize = 2.0
            ax.errorbar(
                abl_df["ILD"].to_numpy(dtype=float) + x_offset[model_name],
                abl_df["mean"],
                yerr=abl_df["sem"],
                fmt=plot_marker,
                linestyle=plot_linestyle,
                color=COLORS[abl],
                ecolor=COLORS[abl],
                alpha=plot_alpha,
                capsize=plot_capsize,
                markersize=plot_markersize,
                linewidth=plot_linewidth,
                elinewidth=plot_elinewidth,
                markeredgewidth=1.15 if model_name == "Lapse" else 0.8,
                label=f"ABL {abl} {model_name}",
            )
            ax.set_ylabel(ylabel)
            ax.set_xlabel("ILD")
            ax.grid(True, alpha=0.25)

axes[0].axhline(0.0, color="0.75", linewidth=0.8, zorder=0)

for model_name, color, marker, offset in [
    ("No lapse", "black", ".", -0.08),
    ("Lapse", "tab:red", "x", 0.08),
]:
    panel_df = plot_summary_df[
        (plot_summary_df["panel"] == "gamma_abl_average") & (plot_summary_df["model"] == model_name)
    ].sort_values("ILD")
    axes[1].errorbar(
        panel_df["ILD"].to_numpy(dtype=float) + offset,
        panel_df["mean"],
        yerr=panel_df["sem"],
        fmt=marker,
        linestyle="none",
        color=color,
        ecolor=color,
        capsize=2.0,
        markersize=7.0 if model_name == "No lapse" else 5.5,
        label=model_name,
    )

axes[1].axhline(0.0, color="0.75", linewidth=0.8, zorder=0)
axes[1].set_ylabel("Gamma averaged across ABLs")
axes[1].set_xlabel("ILD")
axes[1].grid(True, alpha=0.25)

for ax in axes:
    ax.set_xticks(ILDS)
    ax.set_xticklabels([f"{ild:+d}" for ild in ILDS], rotation=45, ha="right", fontsize=8)

axes[0].set_title("Gamma by ABL")
axes[1].set_title("Gamma averaged across ABLs")
axes[2].set_title("Omega by ABL")

abl_handles = [Line2D([0], [0], color=COLORS[abl], marker="o", lw=1.2, label=f"ABL {abl}") for abl in ABLS]
model_handles = [
    Line2D([0], [0], color="0.25", marker=".", linestyle="-", lw=1.2, markersize=7, label="No lapse"),
    Line2D([0], [0], color="0.25", marker="x", linestyle="--", lw=1.2, markersize=5.5, label="Lapse"),
]
fig.legend(
    handles=abl_handles + model_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.015),
    ncol=5,
    frameon=False,
)
fig.suptitle("Condition-wise Gamma/Omega SVI fits with and without lapses", y=1.02)
fig.tight_layout(rect=[0, 0.06, 1, 0.98])
fig.savefig(FIG_PATH, dpi=200, bbox_inches="tight")

print(f"Saved figure: {FIG_PATH}")

# %%
