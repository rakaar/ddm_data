# %%
"""
3 x 3 animal-wise parameter comparison between:
- NPL+alpha with condition t_E_aff fixed from condition-by-condition fits
- NPL+alpha with ABL-specific ILD2 delay

Only the seven shared non-delay parameters are plotted. Delay parameters are
excluded.
"""

# %%
# =============================================================================
# Editable parameters
# =============================================================================
from pathlib import Path
import os
import pickle
import re

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

FIXED_DELAY_DIR = SCRIPT_DIR / "NPL_alpha_condition_t_E_aff_fixed_delay_fit_results_all_30"
ABL_SPECIFIC_ILD2_DIR = REPO_DIR / "all_30_NPL_alpha_ABL_specific_ILD2_delay_fit_results"

OUTPUT_DIR = SCRIPT_DIR / "fixed_condition_t_E_aff_vs_abl_specific_ild2_param_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_N_ANIMALS = 30
DESIRED_BATCHES = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]

FIG_PNG = OUTPUT_DIR / "fixed_condition_t_E_aff_vs_abl_specific_ild2_params_3x3.png"
VALUES_CSV = OUTPUT_DIR / "fixed_condition_t_E_aff_vs_abl_specific_ild2_params_3x3_values.csv"

MODEL_CONFIGS = [
    {
        "model": "fixed_condition_t_E_aff",
        "label": "condition t_E_aff fixed",
        "color": "#0072B2",
        "x_jitter": -0.12,
        "result_dir": FIXED_DELAY_DIR,
        "glob": "results_*_NORM_ALPHA_CONDITION_T_E_AFF_FIXED_DELAY_FROM_ABORTS.pkl",
        "regex": (
            r"^results_(?P<batch>.+)_animal_(?P<animal>\d+)_"
            r"NORM_ALPHA_CONDITION_T_E_AFF_FIXED_DELAY_FROM_ABORTS\.pkl$"
        ),
        "result_key": "vbmc_norm_alpha_condition_t_E_aff_fixed_delay_tied_results",
    },
    {
        "model": "abl_specific_ild2_delay",
        "label": "NPL+alpha+ABL-specific ILD2",
        "color": "#D62728",
        "x_jitter": 0.12,
        "result_dir": ABL_SPECIFIC_ILD2_DIR,
        "glob": "results_*_NORM_ALPHA_ABL_SPECIFIC_ILD2_DELAY_FROM_ABORTS.pkl",
        "regex": (
            r"^results_(?P<batch>.+)_animal_(?P<animal>\d+)_"
            r"NORM_ALPHA_ABL_SPECIFIC_ILD2_DELAY_FROM_ABORTS\.pkl$"
        ),
        "result_key": "vbmc_norm_alpha_abl_specific_ild2_delay_tied_results",
    },
]

PARAM_SPECS = [
    {"param": "lambda", "sample_key": "rate_lambda_samples", "scale": 1.0, "ylabel": "lambda"},
    {"param": "T_0_ms", "sample_key": "T_0_samples", "scale": 1e3, "ylabel": "T_0 (ms)"},
    {"param": "theta_E", "sample_key": "theta_E_samples", "scale": 1.0, "ylabel": "theta_E"},
    {"param": "w", "sample_key": "w_samples", "scale": 1.0, "ylabel": "w"},
    {"param": "del_go_ms", "sample_key": "del_go_samples", "scale": 1e3, "ylabel": "del_go (ms)"},
    {"param": "rate_norm_L", "sample_key": "rate_norm_l_samples", "scale": 1.0, "ylabel": "rate_norm_L"},
    {"param": "alpha", "sample_key": "alpha_samples", "scale": 1.0, "ylabel": "alpha"},
]


# %%
# =============================================================================
# Imports and helpers
# =============================================================================
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


def animal_sort_key(batch_name, animal):
    batch_idx = DESIRED_BATCHES.index(batch_name) if batch_name in DESIRED_BATCHES else len(DESIRED_BATCHES)
    return batch_idx, int(animal)


# %%
# =============================================================================
# Load posterior samples
# =============================================================================
rows = []

for model_config in MODEL_CONFIGS:
    result_paths = sorted(model_config["result_dir"].glob(model_config["glob"]))
    if len(result_paths) != EXPECTED_N_ANIMALS:
        raise RuntimeError(
            f"Expected {EXPECTED_N_ANIMALS} result pkls for {model_config['label']}, "
            f"found {len(result_paths)} in {model_config['result_dir']}"
        )

    for result_path in result_paths:
        match = re.match(model_config["regex"], result_path.name)
        if match is None:
            raise RuntimeError(f"Could not parse result filename: {result_path.name}")

        batch_name = match.group("batch")
        animal = int(match.group("animal"))

        with result_path.open("rb") as f:
            saved = pickle.load(f)

        result_key = model_config["result_key"]
        if result_key not in saved:
            raise KeyError(f"{result_path} is missing `{result_key}`")

        result = saved[result_key]
        message = str(result.get("message", ""))
        if "stable" not in message.lower():
            raise RuntimeError(f"{model_config['label']} fit is not stable for {batch_name}/{animal}: {message}")

        for param_spec in PARAM_SPECS:
            sample_key = param_spec["sample_key"]
            if sample_key not in result:
                raise KeyError(f"{result_path} is missing `{sample_key}`")

            values = np.asarray(result[sample_key], dtype=float) * param_spec["scale"]
            q025, q975 = np.percentile(values, [2.5, 97.5])
            rows.append(
                {
                    "batch_name": batch_name,
                    "animal": animal,
                    "animal_label": f"{batch_name}/{animal}",
                    "model": model_config["model"],
                    "model_label": model_config["label"],
                    "parameter": param_spec["param"],
                    "ylabel": param_spec["ylabel"],
                    "mean": float(np.mean(values)),
                    "q025": float(q025),
                    "q975": float(q975),
                    "posterior_sd": float(np.std(values, ddof=1)),
                    "n_samples": int(values.size),
                    "source_pkl": str(result_path),
                    "message": message,
                }
            )

param_df = pd.DataFrame(rows)
param_df["batch_order"] = param_df["batch_name"].map(
    {batch: idx for idx, batch in enumerate(DESIRED_BATCHES)}
).fillna(len(DESIRED_BATCHES))
param_df = param_df.sort_values(["batch_order", "animal", "parameter", "model"]).reset_index(drop=True)
param_df.drop(columns=["batch_order"]).to_csv(VALUES_CSV, index=False)

animal_order_df = (
    param_df[["batch_name", "animal", "animal_label"]]
    .drop_duplicates()
    .sort_values(["batch_name", "animal"], key=lambda col: col)
)
animal_order_df["sort_key"] = animal_order_df.apply(
    lambda row: animal_sort_key(row["batch_name"], row["animal"]), axis=1
)
animal_order_df = animal_order_df.sort_values("sort_key").drop(columns=["sort_key"]).reset_index(drop=True)
animal_labels = animal_order_df["animal_label"].tolist()

if len(animal_labels) != EXPECTED_N_ANIMALS:
    raise RuntimeError(f"Expected {EXPECTED_N_ANIMALS} unique animals, found {len(animal_labels)}")

counts = param_df.groupby(["model_label", "parameter"]).size().unstack("parameter")
print("Parameter counts by model:")
print(counts.to_string())
print(f"Saved parameter table: {VALUES_CSV}")


# %%
# =============================================================================
# Plot 3 x 3 animal-wise parameter comparison
# =============================================================================
fig, axes = plt.subplots(3, 3, figsize=(21, 12), constrained_layout=True)
axes_flat = axes.ravel()
x_base = np.arange(len(animal_labels))
animal_to_x = {label: idx for idx, label in enumerate(animal_labels)}

for ax, param_spec in zip(axes_flat, PARAM_SPECS):
    param = param_spec["param"]
    subset = param_df[param_df["parameter"] == param].copy()

    for model_config in MODEL_CONFIGS:
        model_subset = (
            subset[subset["model"] == model_config["model"]]
            .set_index("animal_label")
            .loc[animal_labels]
            .reset_index()
        )
        x = np.asarray([animal_to_x[label] for label in model_subset["animal_label"]], dtype=float)
        x = x + model_config["x_jitter"]
        y = model_subset["mean"].to_numpy(dtype=float)
        yerr = np.vstack(
            [
                y - model_subset["q025"].to_numpy(dtype=float),
                model_subset["q975"].to_numpy(dtype=float) - y,
            ]
        )

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o",
            markersize=4.4,
            markerfacecolor=model_config["color"],
            markeredgecolor="white",
            markeredgewidth=0.5,
            color=model_config["color"],
            ecolor=model_config["color"],
            elinewidth=0.9,
            capsize=2,
            capthick=0.9,
            alpha=0.92,
            linestyle="none",
        )

    ax.set_title(param_spec["ylabel"])
    ax.set_ylabel(param_spec["ylabel"])
    ax.set_xticks(x_base)
    ax.set_xticklabels(animal_labels, rotation=45, ha="right", fontsize=7)
    ax.grid(axis="y", alpha=0.25)

for ax in axes_flat[len(PARAM_SPECS) :]:
    ax.axis("off")

legend_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markerfacecolor=model_config["color"],
        markeredgecolor="white",
        markersize=7,
        label=model_config["label"],
    )
    for model_config in MODEL_CONFIGS
]
axes_flat[-1].legend(handles=legend_handles, loc="center", frameon=False, fontsize=12)

fig.suptitle("NPL+alpha shared parameter comparison by animal", fontsize=16)
fig.savefig(FIG_PNG, dpi=250, bbox_inches="tight")
print(f"Saved figure: {FIG_PNG}")


# %%
print("\nOutput files:")
print(FIG_PNG)
print(VALUES_CSV)
