# %%
"""
Overlay old-rule and patience-12 restore-best all-animal condition parameters.

Both curves are across-animal mean +/- SEM of posterior means. The old-rule
curve is dashed with x markers so it can be compared against the new
patience-12 restore-best fit in the same axes.
"""

# %%
# =============================================================================
# Parameters
# =============================================================================
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
OLD_OUTPUT_ROOT = SCRIPT_DIR / "svi_big_gamma_omega_delay_all_animals_outputs"
NEW_OUTPUT_ROOT = SCRIPT_DIR / "svi_big_gamma_omega_delay_patience12_restore_best_all_animals_outputs"

OLD_SUMMARY_CSV = (
    OLD_OUTPUT_ROOT
    / "group_summary"
    / "big_gamma_omega_delay_all_animals_condition_param_summary.csv"
)
NEW_SUMMARY_CSV = (
    NEW_OUTPUT_ROOT
    / "group_summary"
    / "big_gamma_omega_delay_patience12_all_animals_condition_param_summary.csv"
)

GROUP_OUTPUT_DIR = NEW_OUTPUT_ROOT / "group_summary"
GROUP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_PATH = GROUP_OUTPUT_DIR / "big_gamma_omega_delay_patience12_vs_old_condition_params.png"
DELTA_CSV = GROUP_OUTPUT_DIR / "big_gamma_omega_delay_patience12_vs_old_condition_param_deltas.csv"
DELTA_FIG_PATH = GROUP_OUTPUT_DIR / "big_gamma_omega_delay_patience12_vs_old_condition_param_deltas.png"
OLD_ANIMAL_VALUES_CSV = (
    OLD_OUTPUT_ROOT
    / "group_summary"
    / "big_gamma_omega_delay_all_animals_condition_param_animal_values.csv"
)
NEW_ANIMAL_VALUES_CSV = (
    NEW_OUTPUT_ROOT
    / "group_summary"
    / "big_gamma_omega_delay_patience12_all_animals_condition_param_animal_values.csv"
)

ABLS = [20, 40, 60]
ILDS = np.array([-16, -8, -4, -2, -1, 1, 2, 4, 8, 16], dtype=int)
COLORS = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}

PLOT_SPECS = [
    ("gamma", "Gamma"),
    ("omega", "Omega"),
    ("t_E_aff_ms", "t_E_aff (ms)"),
]


# %%
# =============================================================================
# Load summaries and validate matching conditions
# =============================================================================
if not OLD_SUMMARY_CSV.exists():
    raise FileNotFoundError(OLD_SUMMARY_CSV)
if not NEW_SUMMARY_CSV.exists():
    raise FileNotFoundError(NEW_SUMMARY_CSV)
if not OLD_ANIMAL_VALUES_CSV.exists():
    raise FileNotFoundError(OLD_ANIMAL_VALUES_CSV)
if not NEW_ANIMAL_VALUES_CSV.exists():
    raise FileNotFoundError(NEW_ANIMAL_VALUES_CSV)

old_df = pd.read_csv(OLD_SUMMARY_CSV).copy()
new_df = pd.read_csv(NEW_SUMMARY_CSV).copy()
old_animal_df = pd.read_csv(OLD_ANIMAL_VALUES_CSV).copy()
new_animal_df = pd.read_csv(NEW_ANIMAL_VALUES_CSV).copy()

for name, df in [("old", old_df), ("patience12", new_df)]:
    df["ABL"] = df["ABL"].astype(int)
    df["ILD"] = df["ILD"].astype(int)
    if len(df) != len(ABLS) * len(ILDS):
        raise RuntimeError(f"{name} summary has {len(df)} rows, expected {len(ABLS) * len(ILDS)}.")
    duplicate_count = int(df.duplicated(["ABL", "ILD"]).sum())
    if duplicate_count:
        raise RuntimeError(f"{name} summary has {duplicate_count} duplicate ABL/ILD rows.")
    for row in df.itertuples(index=False):
        expected_n = 24 if abs(int(row.ILD)) == 16 else 30
        if int(row.n_animals) != expected_n:
            raise RuntimeError(f"{name} ABL={row.ABL}, ILD={row.ILD} has n={row.n_animals}, expected {expected_n}.")

old_keys = old_df[["ABL", "ILD"]].sort_values(["ABL", "ILD"]).reset_index(drop=True)
new_keys = new_df[["ABL", "ILD"]].sort_values(["ABL", "ILD"]).reset_index(drop=True)
if not old_keys.equals(new_keys):
    raise RuntimeError("Old and patience-12 summaries do not have matching ABL/ILD grids.")

merged_df = new_df.merge(
    old_df,
    on=["ABL", "ILD"],
    suffixes=("_patience12", "_old"),
    validate="one_to_one",
)

delta_rows = []
for row in merged_df.itertuples(index=False):
    delta_row = {
        "ABL": int(row.ABL),
        "ILD": int(row.ILD),
        "n_animals_patience12": int(row.n_animals_patience12),
        "n_animals_old": int(row.n_animals_old),
    }
    for short_name, _ in PLOT_SPECS:
        new_mean = float(getattr(row, f"{short_name}_mean_patience12"))
        old_mean = float(getattr(row, f"{short_name}_mean_old"))
        delta_row[f"{short_name}_delta_patience12_minus_old"] = new_mean - old_mean
    delta_rows.append(delta_row)

delta_df = pd.DataFrame(delta_rows).sort_values(["ABL", "ILD"]).reset_index(drop=True)
delta_df.to_csv(DELTA_CSV, index=False)

animal_delta_df = new_animal_df.merge(
    old_animal_df,
    on=["batch_name", "animal", "ABL", "ILD"],
    suffixes=("_patience12", "_old"),
    validate="one_to_one",
)
animal_delta_rows = []
for row in animal_delta_df.itertuples(index=False):
    delta_row = {
        "batch_name": row.batch_name,
        "animal": int(row.animal),
        "ABL": int(row.ABL),
        "ILD": int(row.ILD),
    }
    for short_name, _ in PLOT_SPECS:
        delta_row[f"{short_name}_delta_patience12_minus_old"] = float(
            getattr(row, f"{short_name}_mean_patience12")
            - getattr(row, f"{short_name}_mean_old")
        )
    animal_delta_rows.append(delta_row)

animal_delta_df = pd.DataFrame(animal_delta_rows)
delta_summary_rows = []
for (abl, ild), group in animal_delta_df.groupby(["ABL", "ILD"], sort=True):
    summary_row = {
        "ABL": int(abl),
        "ILD": int(ild),
        "n_animals": int(group[["batch_name", "animal"]].drop_duplicates().shape[0]),
    }
    for short_name, _ in PLOT_SPECS:
        values = group[f"{short_name}_delta_patience12_minus_old"].to_numpy(dtype=float)
        summary_row[f"{short_name}_delta_mean"] = float(np.mean(values))
        summary_row[f"{short_name}_delta_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
        summary_row[f"{short_name}_delta_sem"] = (
            float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan
        )
    delta_summary_rows.append(summary_row)

delta_summary_df = pd.DataFrame(delta_summary_rows).sort_values(["ABL", "ILD"]).reset_index(drop=True)

print("Loaded matching old-rule and patience-12 condition summaries.")
print("Maximum absolute patience12-old mean differences:")
for short_name, label in PLOT_SPECS:
    max_abs_delta = float(np.max(np.abs(delta_df[f"{short_name}_delta_patience12_minus_old"])))
    print(f"  {label}: {max_abs_delta:.6g}")
print(f"Saved delta CSV: {DELTA_CSV}")


# %%
# =============================================================================
# Plot overlay
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharex=True)

for ax, (short_name, ylabel) in zip(axes, PLOT_SPECS):
    for abl in ABLS:
        new_abl_df = new_df[new_df["ABL"] == abl].sort_values("ILD")
        old_abl_df = old_df[old_df["ABL"] == abl].sort_values("ILD")

        ax.errorbar(
            new_abl_df["ILD"],
            new_abl_df[f"{short_name}_mean"],
            yerr=new_abl_df[f"{short_name}_sem"],
            fmt="o-",
            color=COLORS[abl],
            ecolor=COLORS[abl],
            capsize=2.0,
            markersize=4,
            linewidth=1.2,
            label=f"ABL {abl}",
        )
        ax.errorbar(
            old_abl_df["ILD"],
            old_abl_df[f"{short_name}_mean"],
            yerr=old_abl_df[f"{short_name}_sem"],
            fmt="x--",
            color=COLORS[abl],
            ecolor=COLORS[abl],
            capsize=2.0,
            markersize=5,
            linewidth=3.0,
            alpha=0.4,
        )

    ax.set_ylabel(ylabel)
    ax.set_xlabel("ILD")
    ax.grid(True, alpha=0.25)
    if short_name == "gamma":
        ax.axhline(0.0, color="0.75", linewidth=0.8, zorder=0)

for ax in axes:
    ax.set_xticks(ILDS)
    ax.set_xticklabels([f"{ild:+d}" for ild in ILDS], rotation=45, ha="right", fontsize=8)

abl_handles = [
    Line2D([0], [0], color=COLORS[abl], marker="o", lw=1.5, label=f"ABL {abl}")
    for abl in ABLS
]
style_handles = [
    Line2D([0], [0], color="0.2", marker="o", lw=1.2, label="patience-12 restore-best"),
    Line2D([0], [0], color="0.2", marker="x", lw=3.0, ls="--", alpha=0.4, label="old stopping rule"),
]
axes[0].legend(handles=abl_handles, frameon=False, loc="best")
axes[1].legend(handles=style_handles, frameon=False, loc="best")

fig.suptitle("Old stopping rule vs patience-12 restore-best condition parameters", y=1.02)
fig.tight_layout()
fig.savefig(FIG_PATH, dpi=200, bbox_inches="tight")
print(f"Saved figure: {FIG_PATH}")

plt.show()


# %%
# =============================================================================
# Plot direct deltas
# =============================================================================
delta_fig, delta_axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharex=True)

for ax, (short_name, ylabel) in zip(delta_axes, PLOT_SPECS):
    for abl in ABLS:
        abl_df = delta_summary_df[delta_summary_df["ABL"] == abl].sort_values("ILD")
        ax.errorbar(
            abl_df["ILD"],
            abl_df[f"{short_name}_delta_mean"],
            yerr=abl_df[f"{short_name}_delta_sem"],
            fmt="o-",
            color=COLORS[abl],
            ecolor=COLORS[abl],
            capsize=2.0,
            markersize=4,
            linewidth=1.2,
            label=f"ABL {abl}",
        )
    ax.axhline(0.0, color="0.2", linewidth=0.8, zorder=0)
    ax.set_ylabel(f"Delta {ylabel}\n(patience-12 - old)")
    ax.set_xlabel("ILD")
    ax.grid(True, alpha=0.25)

for ax in delta_axes:
    ax.set_xticks(ILDS)
    ax.set_xticklabels([f"{ild:+d}" for ild in ILDS], rotation=45, ha="right", fontsize=8)

delta_axes[0].legend(frameon=False, loc="best")
delta_fig.suptitle("Direct parameter shifts from old stopping rule to patience-12 restore-best", y=1.02)
delta_fig.tight_layout()
delta_fig.savefig(DELTA_FIG_PATH, dpi=200, bbox_inches="tight")
print(f"Saved delta figure: {DELTA_FIG_PATH}")

plt.show()

# %%
