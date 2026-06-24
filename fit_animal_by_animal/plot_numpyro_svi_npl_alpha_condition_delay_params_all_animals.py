# %%
"""
All-animal posterior parameter plots for the NumPyro SVI
NPL+alpha condition-delay fits.

This script makes:
- a 3 x 3 plot of the 7 shared non-delay parameters by animal, with
  posterior means and 95% posterior intervals;
- a separate across-animal t_E_aff-vs-ILD plot by ABL.
"""

# %%
# =============================================================================
# Editable parameters
# =============================================================================
from pathlib import Path
import os

SCRIPT_DIR = Path(__file__).resolve().parent

SVI_LABEL = os.environ.get("NUMPYRO_SVI_PARAM_LABEL", "main_fullrank")
SVI_OUTPUT_ROOT = SCRIPT_DIR / "numpyro_svi_npl_alpha_condition_delay_single_animal_outputs"
OUTPUT_DIR = SCRIPT_DIR / "numpyro_svi_npl_alpha_condition_delay_all_animals_diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_N_ANIMALS = int(os.environ.get("NUMPYRO_SVI_EXPECTED_N_ANIMALS", "30"))
DESIRED_BATCHES = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]
ABLS = [20.0, 40.0, 60.0]
ILD_GRID = [-16.0, -8.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0, 16.0]

PARAMS_BY_ANIMAL_PNG = OUTPUT_DIR / f"{SVI_LABEL}_all_animals_global_params_by_animal.png"
DELAY_BY_ILD_PNG = OUTPUT_DIR / f"{SVI_LABEL}_all_animals_t_E_aff_vs_ild_by_abl.png"

ABL_COLORS = {
    20.0: "#1f77b4",
    40.0: "#ff7f0e",
    60.0: "#2ca02c",
}


# %%
# =============================================================================
# Imports and small helpers
# =============================================================================
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PARAM_SPECS = [
    ("rate_lambda", "lambda", 1.0),
    ("T_0", "T_0 (ms)", 1000.0),
    ("theta_E", "theta_E", 1.0),
    ("w", "w", 1.0),
    ("del_go", "del_go (ms)", 1000.0),
    ("rate_norm_l", "rate_norm_L", 1.0),
    ("alpha", "alpha", 1.0),
]


def sort_key_for_animal_dir(path):
    batch_name, animal_text = path.name.rsplit("_", 1)
    batch_idx = DESIRED_BATCHES.index(batch_name) if batch_name in DESIRED_BATCHES else len(DESIRED_BATCHES)
    return batch_idx, int(animal_text)


def sem_from_values(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return np.nan
    return np.std(values, ddof=1) / np.sqrt(values.size)


# %%
# =============================================================================
# Load posterior samples and condition tables
# =============================================================================
animal_dirs = sorted(
    [p.parent for p in SVI_OUTPUT_ROOT.glob(f"*/{SVI_LABEL}_posterior_samples.npz")],
    key=sort_key_for_animal_dir,
)

if len(animal_dirs) != EXPECTED_N_ANIMALS:
    raise RuntimeError(f"Expected {EXPECTED_N_ANIMALS} animals, found {len(animal_dirs)}")

print(f"Loaded {len(animal_dirs)} animals from {SVI_OUTPUT_ROOT}")

animal_labels = []
param_rows = []
delay_rows = []

for animal_dir in animal_dirs:
    batch_name, animal_text = animal_dir.name.rsplit("_", 1)
    animal_label = f"{batch_name}/{animal_text}"
    animal_labels.append(animal_label)

    samples_path = animal_dir / f"{SVI_LABEL}_posterior_samples.npz"
    condition_table_path = animal_dir / "condition_table.csv"
    samples = np.load(samples_path)
    condition_table = pd.read_csv(condition_table_path)

    for param_name, display_name, scale in PARAM_SPECS:
        vals = np.asarray(samples[param_name], dtype=float) * scale
        param_rows.append(
            {
                "animal_label": animal_label,
                "batch_name": batch_name,
                "animal": int(animal_text),
                "parameter": param_name,
                "display_name": display_name,
                "mean": np.mean(vals),
                "q025": np.quantile(vals, 0.025),
                "q975": np.quantile(vals, 0.975),
            }
        )

    t_e_aff_samples = np.asarray(samples["t_E_aff"], dtype=float)
    for _, row in condition_table.iterrows():
        condition_id = int(row["condition_id"])
        vals_ms = t_e_aff_samples[:, condition_id] * 1000.0
        delay_rows.append(
            {
                "animal_label": animal_label,
                "batch_name": batch_name,
                "animal": int(animal_text),
                "ABL": float(row["ABL"]),
                "ILD": float(row["ILD"]),
                "t_E_aff_mean_ms": np.mean(vals_ms),
            }
        )

param_df = pd.DataFrame(param_rows)
delay_df = pd.DataFrame(delay_rows)

print("Condition counts per animal:")
print(delay_df.groupby("animal_label").size().value_counts().sort_index())
print(f"Global parameter rows: {len(param_df)}")
print(f"Delay rows: {len(delay_df)}")


# %%
# =============================================================================
# Shared non-delay parameter plot by animal
# =============================================================================
fig, axes = plt.subplots(3, 3, figsize=(20, 11), constrained_layout=True)
axes_flat = axes.ravel()
x = np.arange(len(animal_labels))

for ax, (param_name, display_name, _scale) in zip(axes_flat, PARAM_SPECS):
    sub = (
        param_df[param_df["parameter"] == param_name]
        .set_index("animal_label")
        .loc[animal_labels]
        .reset_index()
    )
    means = sub["mean"].to_numpy(dtype=float)
    lower = means - sub["q025"].to_numpy(dtype=float)
    upper = sub["q975"].to_numpy(dtype=float) - means

    ax.errorbar(
        x,
        means,
        yerr=np.vstack([lower, upper]),
        fmt="o",
        markersize=4,
        color="#0072B2",
        ecolor="#0072B2",
        elinewidth=1,
        capsize=2,
        linestyle="none",
    )
    ax.set_title(display_name)
    ax.set_ylabel(display_name)
    ax.set_xticks(x)
    ax.set_xticklabels(animal_labels, rotation=45, ha="right", fontsize=7)
    ax.grid(axis="y", alpha=0.25)

for ax in axes_flat[len(PARAM_SPECS) :]:
    ax.axis("off")

fig.suptitle("NumPyro SVI shared parameters by animal", fontsize=16)
fig.savefig(PARAMS_BY_ANIMAL_PNG, dpi=200, bbox_inches="tight")
print(f"Saved {PARAMS_BY_ANIMAL_PNG}")


# %%
# =============================================================================
# Across-animal t_E_aff vs ILD by ABL
# =============================================================================
delay_summary_rows = []

for abl in ABLS:
    for ild in ILD_GRID:
        vals = delay_df.loc[
            (delay_df["ABL"] == abl) & (delay_df["ILD"] == ild),
            "t_E_aff_mean_ms",
        ].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        delay_summary_rows.append(
            {
                "ABL": abl,
                "ILD": ild,
                "n_animals": int(vals.size),
                "mean_ms": np.mean(vals) if vals.size else np.nan,
                "sem_ms": sem_from_values(vals),
            }
        )

delay_summary_df = pd.DataFrame(delay_summary_rows)
print("Delay summary n_animals by ABL/ILD:")
print(delay_summary_df.pivot(index="ILD", columns="ABL", values="n_animals"))

fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
for abl in ABLS:
    sub = delay_summary_df[delay_summary_df["ABL"] == abl].sort_values("ILD")
    ax.errorbar(
        sub["ILD"],
        sub["mean_ms"],
        yerr=sub["sem_ms"],
        fmt="o-",
        color=ABL_COLORS[abl],
        ecolor=ABL_COLORS[abl],
        elinewidth=1.25,
        capsize=3,
        markersize=5,
        label=f"ABL {int(abl)}",
    )

ax.axvline(0, color="0.8", linewidth=1)
ax.set_xticks(ILD_GRID)
ax.set_xticklabels([str(int(v)) for v in ILD_GRID], rotation=45, ha="right")
ax.set_xlabel("ILD")
ax.set_ylabel("t_E_aff (ms)")
ax.set_title("NumPyro SVI condition delays averaged across animals")
ax.grid(axis="y", alpha=0.25)
ax.legend(frameon=False)
fig.savefig(DELAY_BY_ILD_PNG, dpi=200, bbox_inches="tight")
print(f"Saved {DELAY_BY_ILD_PNG}")


# %%
# =============================================================================
# Keep data frames available in interactive sessions
# =============================================================================
print("\nOutput files:")
print(PARAMS_BY_ANIMAL_PNG)
print(DELAY_BY_ILD_PNG)
