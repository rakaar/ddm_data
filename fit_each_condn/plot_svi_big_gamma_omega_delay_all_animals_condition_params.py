# %%
"""
Plot all-animal means for the big Gamma/Omega/delay SVI condition parameters.

Each animal contributes one posterior mean per fitted condition. Error bars are
SEM across animals.
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
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / "svi_big_gamma_omega_delay_all_animals_outputs"
GROUP_OUTPUT_DIR = OUTPUT_ROOT / "group_summary"
GROUP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_PATH = GROUP_OUTPUT_DIR / "big_gamma_omega_delay_all_animals_condition_params.png"
ANIMAL_VALUES_CSV = GROUP_OUTPUT_DIR / "big_gamma_omega_delay_all_animals_condition_param_animal_values.csv"
SUMMARY_CSV = GROUP_OUTPUT_DIR / "big_gamma_omega_delay_all_animals_condition_param_summary.csv"

EXPECTED_N_ANIMALS = 30
EXPECTED_N_CONDITION_ROWS = 864
ABLS = [20, 40, 60]
ILDS = np.array([-16, -8, -4, -2, -1, 1, 2, 4, 8, 16], dtype=int)
COLORS = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}

PLOT_SPECS = [
    ("gamma", "Gamma", "gamma_mean", 1.0),
    ("omega", "Omega", "omega_mean", 1.0),
    ("t_E_aff_ms", "t_E_aff (ms)", "t_E_aff_ms_mean", 1.0),
]


# %%
# =============================================================================
# Load all condition summaries
# =============================================================================
summary_paths = sorted(OUTPUT_ROOT.glob("*/*_big_gamma_omega_delay_condition_summary.csv"))
if len(summary_paths) != EXPECTED_N_ANIMALS:
    raise RuntimeError(f"Expected {EXPECTED_N_ANIMALS} condition-summary CSVs, found {len(summary_paths)}")

animal_dfs = []
for summary_path in summary_paths:
    match = re.match(r"^(?P<batch>.+)_(?P<animal>\d+)$", summary_path.parent.name)
    if match is None:
        raise RuntimeError(f"Could not parse animal folder name: {summary_path.parent}")

    df = pd.read_csv(summary_path)
    required_cols = [
        "batch_name",
        "animal",
        "ABL",
        "ILD",
        "gamma_mean",
        "omega_mean",
        "t_E_aff_ms_mean",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"{summary_path} missing columns: {missing_cols}")

    df["animal"] = df["animal"].astype(int)
    df["ABL"] = df["ABL"].astype(int)
    df["ILD"] = df["ILD"].astype(int)
    animal_dfs.append(df[required_cols].copy())

condition_df = pd.concat(animal_dfs, ignore_index=True)
condition_df = condition_df.sort_values(["batch_name", "animal", "ABL", "ILD"]).reset_index(drop=True)

if len(condition_df) != EXPECTED_N_CONDITION_ROWS:
    raise RuntimeError(f"Expected {EXPECTED_N_CONDITION_ROWS} condition rows, found {len(condition_df)}")
if condition_df[["batch_name", "animal"]].drop_duplicates().shape[0] != EXPECTED_N_ANIMALS:
    raise RuntimeError("Did not load exactly 30 unique animals.")
duplicate_count = int(condition_df.duplicated(["batch_name", "animal", "ABL", "ILD"]).sum())
if duplicate_count:
    raise RuntimeError(f"Found {duplicate_count} duplicate animal/condition rows.")

value_cols = [spec[2] for spec in PLOT_SPECS]
if not np.all(np.isfinite(condition_df[value_cols].to_numpy(dtype=float))):
    raise RuntimeError("Condition parameter means contain NaN/Inf values.")

condition_df.to_csv(ANIMAL_VALUES_CSV, index=False)
print(f"Loaded {condition_df[['batch_name', 'animal']].drop_duplicates().shape[0]} animals")
print(f"Loaded {len(condition_df)} condition rows")
print(f"Saved animal-wise values: {ANIMAL_VALUES_CSV}")


# %%
# =============================================================================
# Average per condition across animals
# =============================================================================
summary_rows = []
for (abl, ild), group in condition_df.groupby(["ABL", "ILD"], sort=True):
    row = {"ABL": int(abl), "ILD": int(ild)}
    n_animals = group[["batch_name", "animal"]].drop_duplicates().shape[0]
    row["n_animals"] = int(n_animals)

    for short_name, _, value_col, scale in PLOT_SPECS:
        values = group[value_col].to_numpy(dtype=float) * scale
        finite = np.isfinite(values)
        n = int(np.sum(finite))
        row[f"{short_name}_mean"] = float(np.nanmean(values)) if n else np.nan
        row[f"{short_name}_sd"] = float(np.nanstd(values, ddof=1)) if n > 1 else np.nan
        row[f"{short_name}_sem"] = float(np.nanstd(values, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        row[f"{short_name}_n"] = n
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows).sort_values(["ABL", "ILD"]).reset_index(drop=True)
summary_df.to_csv(SUMMARY_CSV, index=False)
print(f"Saved condition summary: {SUMMARY_CSV}")

for row in summary_df.itertuples(index=False):
    expected_n = 24 if abs(int(row.ILD)) == 16 else 30
    if int(row.n_animals) != expected_n:
        raise RuntimeError(f"ABL={row.ABL}, ILD={row.ILD} has n={row.n_animals}, expected {expected_n}")

print("Animal counts per ABL/ILD validated: |ILD|=16 has n=24, all other ILDs have n=30.")


# %%
# =============================================================================
# Plot all-animal means
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)

for ax, (short_name, ylabel, _, _) in zip(axes, PLOT_SPECS):
    for abl in ABLS:
        abl_df = summary_df[summary_df["ABL"] == abl].sort_values("ILD")
        ax.errorbar(
            abl_df["ILD"],
            abl_df[f"{short_name}_mean"],
            yerr=abl_df[f"{short_name}_sem"],
            fmt="o-",
            color=COLORS[abl],
            ecolor=COLORS[abl],
            capsize=2.0,
            markersize=4,
            linewidth=1.2,
            label=f"ABL {abl}",
        )
    ax.set_ylabel(ylabel)
    ax.set_xlabel("ILD")
    ax.grid(True, alpha=0.25)
    if short_name == "gamma":
        ax.axhline(0.0, color="0.75", linewidth=0.8, zorder=0)

for ax in axes:
    ax.set_xticks(ILDS)
    ax.set_xticklabels([f"{ild:+d}" for ild in ILDS], rotation=45, ha="right", fontsize=8)

axes[0].legend(frameon=False, loc="best")
fig.suptitle("Big Gamma/Omega/delay SVI condition parameters averaged across animals", y=1.02)
fig.tight_layout()
fig.savefig(FIG_PATH, dpi=200, bbox_inches="tight")
print(f"Saved figure: {FIG_PATH}")

plt.show()

# %%
