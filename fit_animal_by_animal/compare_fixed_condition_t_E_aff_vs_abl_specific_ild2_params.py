# %%
"""
Compare NPL+alpha animal-wise parameters between fixed-condition-t_E_aff and
ABL-specific ILD2-delay fits.

For each animal, this script extracts posterior means for:
lambda, T_0, theta_E, rate_norm_L, and alpha.
"""

# %%
# =============================================================================
# Imports
# =============================================================================
import os
import pickle
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


# %%
# =============================================================================
# Parameters
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

FIXED_DELAY_DIR = SCRIPT_DIR / "NPL_alpha_condition_t_E_aff_fixed_delay_fit_results_all_30"
ABL_SPECIFIC_ILD2_DIR = REPO_DIR / "all_30_NPL_alpha_ABL_specific_ILD2_delay_fit_results"

OUTPUT_DIR = SCRIPT_DIR / "fixed_condition_t_E_aff_vs_abl_specific_ild2_param_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PARAM_VALUES_CSV = OUTPUT_DIR / "fixed_condition_t_E_aff_vs_abl_specific_ild2_param_values.csv"
PARAM_DIFF_CSV = OUTPUT_DIR / "fixed_condition_t_E_aff_minus_abl_specific_ild2_param_differences.csv"
SUMMARY_CSV = OUTPUT_DIR / "fixed_condition_t_E_aff_vs_abl_specific_ild2_param_summary.csv"
FIG_PNG = OUTPUT_DIR / "fixed_condition_t_E_aff_vs_abl_specific_ild2_params.png"
FIG_PDF = OUTPUT_DIR / "fixed_condition_t_E_aff_vs_abl_specific_ild2_params.pdf"

EXPECTED_N_ANIMALS = 30

MODEL_CONFIGS = [
    {
        "model": "fixed_condition_t_E_aff",
        "label": "fixed condition t_E_aff",
        "color": "tab:blue",
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
        "label": "ABL-specific ILD2 delay",
        "color": "tab:orange",
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
    {"param": "rate_norm_L", "sample_key": "rate_norm_l_samples", "scale": 1.0, "ylabel": "rate_norm_L"},
    {"param": "alpha", "sample_key": "alpha_samples", "scale": 1.0, "ylabel": "alpha"},
]

X_JITTER = {
    "fixed_condition_t_E_aff": -0.12,
    "abl_specific_ild2_delay": 0.12,
}


# %%
# =============================================================================
# Load posterior means
# =============================================================================
rows = []

for model_config in MODEL_CONFIGS:
    result_paths = sorted(model_config["result_dir"].glob(model_config["glob"]))
    if len(result_paths) != EXPECTED_N_ANIMALS:
        raise RuntimeError(
            f"Expected {EXPECTED_N_ANIMALS} result pkls for {model_config['label']}, "
            f"found {len(result_paths)}"
        )

    for result_path in result_paths:
        match = re.match(model_config["regex"], result_path.name)
        if match is None:
            raise RuntimeError(f"Could not parse result filename: {result_path.name}")

        batch_name = match.group("batch")
        animal = int(match.group("animal"))

        with result_path.open("rb") as f:
            saved = pickle.load(f)
        if model_config["result_key"] not in saved:
            raise KeyError(f"{result_path} is missing `{model_config['result_key']}`")

        result = saved[model_config["result_key"]]
        message = str(result.get("message", ""))
        if "stable" not in message.lower():
            raise RuntimeError(f"{model_config['label']} fit is not stable for {batch_name}/{animal}: {message}")

        for param_spec in PARAM_SPECS:
            sample_key = param_spec["sample_key"]
            if sample_key not in result:
                raise KeyError(f"{result_path} is missing `{sample_key}`")
            values = np.asarray(result[sample_key], dtype=float) * param_spec["scale"]
            ci95_lower, ci95_upper = np.percentile(values, [2.5, 97.5])
            rows.append(
                {
                    "batch_name": batch_name,
                    "animal": animal,
                    "animal_label": f"{batch_name}/{animal}",
                    "model": model_config["model"],
                    "model_label": model_config["label"],
                    "parameter": param_spec["param"],
                    "value": float(np.mean(values)),
                    "posterior_sd": float(np.std(values)),
                    "ci95_lower": float(ci95_lower),
                    "ci95_upper": float(ci95_upper),
                    "n_samples": int(values.size),
                    "source_pkl": str(result_path),
                    "message": message,
                }
            )

param_df = pd.DataFrame(rows).sort_values(["batch_name", "animal", "parameter", "model"]).reset_index(drop=True)
param_df.to_csv(PARAM_VALUES_CSV, index=False)
print(f"Saved parameter values: {PARAM_VALUES_CSV}")


# %%
# =============================================================================
# Pair fixed-delay and ABL-specific ILD2 parameter values
# =============================================================================
animal_order_df = (
    param_df[["batch_name", "animal", "animal_label"]]
    .drop_duplicates()
    .sort_values(["batch_name", "animal"])
    .reset_index(drop=True)
)
if len(animal_order_df) != EXPECTED_N_ANIMALS:
    raise RuntimeError(f"Expected {EXPECTED_N_ANIMALS} unique animals, found {len(animal_order_df)}")

counts = param_df.groupby(["model", "parameter"]).size().unstack("parameter")
print("Parameter counts by model:")
print(counts.to_string())

wide_df = param_df.pivot_table(
    index=["batch_name", "animal", "animal_label", "parameter"],
    columns="model",
    values="value",
    aggfunc="first",
).reset_index()
wide_df["fixed_minus_abl_specific_ild2"] = (
    wide_df["fixed_condition_t_E_aff"] - wide_df["abl_specific_ild2_delay"]
)
wide_df["fixed_minus_abl_specific_ild2_pct"] = (
    100.0 * wide_df["fixed_minus_abl_specific_ild2"] / wide_df["abl_specific_ild2_delay"]
)
wide_df.to_csv(PARAM_DIFF_CSV, index=False)
print(f"Saved parameter differences: {PARAM_DIFF_CSV}")

summary_rows = []
for param_spec in PARAM_SPECS:
    param = param_spec["param"]
    subset = wide_df[wide_df["parameter"] == param].copy()
    fixed = subset["fixed_condition_t_E_aff"].to_numpy(dtype=float)
    abl_ild2 = subset["abl_specific_ild2_delay"].to_numpy(dtype=float)
    diff = fixed - abl_ild2
    finite = np.isfinite(fixed) & np.isfinite(abl_ild2)
    summary_rows.append(
        {
            "parameter": param,
            "n_animals": int(np.sum(finite)),
            "fixed_mean": float(np.nanmean(fixed)),
            "abl_specific_ild2_mean": float(np.nanmean(abl_ild2)),
            "mean_fixed_minus_abl_specific_ild2": float(np.nanmean(diff)),
            "rmse_fixed_vs_abl_specific_ild2": float(np.sqrt(np.nanmean(diff**2))),
            "pearson_r": float(np.corrcoef(fixed[finite], abl_ild2[finite])[0, 1]) if np.sum(finite) >= 2 else np.nan,
        }
    )
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_CSV, index=False)
print(f"Saved parameter summary: {SUMMARY_CSV}")
print(summary_df.to_string(index=False))


# %%
# =============================================================================
# Plot 2 x 3 parameter scatter by animal
# =============================================================================
animal_labels = animal_order_df["animal_label"].tolist()
animal_to_x = {label: idx for idx, label in enumerate(animal_labels)}

fig, axes = plt.subplots(2, 3, figsize=(17.0, 8.0), sharex=True)
axes_flat = axes.ravel()

for ax, param_spec in zip(axes_flat, PARAM_SPECS):
    param = param_spec["param"]
    subset = param_df[param_df["parameter"] == param].copy()

    for model_config in MODEL_CONFIGS:
        model_subset = subset[subset["model"] == model_config["model"]].copy()
        x = np.asarray([animal_to_x[label] for label in model_subset["animal_label"]], dtype=float)
        x = x + X_JITTER[model_config["model"]]
        y = model_subset["value"].to_numpy(dtype=float)
        yerr = np.vstack(
            [
                y - model_subset["ci95_lower"].to_numpy(dtype=float),
                model_subset["ci95_upper"].to_numpy(dtype=float) - y,
            ]
        )
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o",
            markersize=4.8,
            markerfacecolor=model_config["color"],
            markeredgecolor="white",
            markeredgewidth=0.5,
            color=model_config["color"],
            ecolor=model_config["color"],
            elinewidth=0.85,
            capsize=1.8,
            capthick=0.85,
            alpha=0.92,
            label=model_config["label"],
        )

    param_summary = summary_df[summary_df["parameter"] == param].iloc[0]
    ax.text(
        0.02,
        0.96,
        (
            f"r={param_summary['pearson_r']:.2f}\n"
            f"RMSE={param_summary['rmse_fixed_vs_abl_specific_ild2']:.3g}"
        ),
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.85},
    )
    ax.set_title(param_spec["ylabel"])
    ax.set_ylabel(param_spec["ylabel"])
    ax.grid(True, axis="y", alpha=0.25)
    ax.grid(True, axis="x", alpha=0.12)

axes_flat[-1].axis("off")

for ax in axes[-1, :]:
    ax.set_xticks(np.arange(len(animal_labels)))
    ax.set_xticklabels(animal_labels, rotation=90, fontsize=7)
    ax.set_xlabel("animal")

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
axes_flat[-1].legend(handles=legend_handles, loc="center", frameon=False, fontsize=11)

fig.suptitle("Animal-wise NPL+alpha parameter comparison", y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIG_PNG, dpi=300, bbox_inches="tight")
fig.savefig(FIG_PDF, bbox_inches="tight")
plt.close(fig)
print(f"Saved figure: {FIG_PNG}")
print(f"Saved figure: {FIG_PDF}")

# %%
