# %%
"""
Plot lapse-specific parameters from NPL+alpha+lapse SVI with IPL+lapse overlay.

The first panel shows lapse_prob as a percentage. The second panel shows
lapse_prob_right directly. NPL+alpha+lapse is plotted as circles; IPL+lapse is
overlaid with x markers for comparison.
"""

# %%
# =============================================================================
# Editable parameters
# =============================================================================
from pathlib import Path
import os

SCRIPT_DIR = Path(__file__).resolve().parent
NPL_LAPSE_OUTPUT_ROOT = (
    SCRIPT_DIR / "numpyro_svi_npl_alpha_lapse_condition_delay_patience12_min50k_restore_best_outputs"
)
IPL_LAPSE_OUTPUT_ROOT = (
    SCRIPT_DIR / "numpyro_svi_vanilla_lapse_condition_delay_patience12_min50k_restore_best_outputs"
)
OUTPUT_DIR = NPL_LAPSE_OUTPUT_ROOT / "summary_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_N_ANIMALS = 30
DESIRED_BATCHES = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]

FIG_PATH = OUTPUT_DIR / "npl_alpha_lapse_svi_lapse_params_by_animal_with_ipl_overlay.png"
VALUES_CSV = OUTPUT_DIR / "npl_alpha_lapse_svi_lapse_params_by_animal_with_ipl_overlay.csv"

MODEL_NPL_LAPSE = "NPL+alpha+lapse SVI"
MODEL_IPL_LAPSE = "IPL+lapse SVI"
MODEL_ORDER = [MODEL_NPL_LAPSE, MODEL_IPL_LAPSE]
MODEL_ROOTS = {
    MODEL_NPL_LAPSE: NPL_LAPSE_OUTPUT_ROOT,
    MODEL_IPL_LAPSE: IPL_LAPSE_OUTPUT_ROOT,
}
MODEL_COLORS = {
    MODEL_NPL_LAPSE: "#0072B2",
    MODEL_IPL_LAPSE: "#D55E00",
}
MODEL_MARKERS = {
    MODEL_NPL_LAPSE: "o",
    MODEL_IPL_LAPSE: "x",
}
MODEL_OFFSETS = {
    MODEL_NPL_LAPSE: -0.11,
    MODEL_IPL_LAPSE: 0.11,
}

PARAMS = [
    ("lapse_prob", "Lapse rate (%)", 100.0),
    ("lapse_prob_right", "lapse_prob_right", 1.0),
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
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def parse_animal_dir_name(path):
    batch_name, animal_text = path.name.rsplit("_", 1)
    return batch_name, int(animal_text)


def animal_sort_key(item):
    batch_name, animal = item
    batch_idx = DESIRED_BATCHES.index(batch_name) if batch_name in DESIRED_BATCHES else len(DESIRED_BATCHES)
    return batch_idx, int(animal)


def load_lapse_params(output_root, model_label):
    summary_paths = sorted(output_root.glob("*/main_fullrank_posterior_summary.csv"))
    if len(summary_paths) != EXPECTED_N_ANIMALS:
        raise RuntimeError(
            f"Expected {EXPECTED_N_ANIMALS} posterior summaries for {model_label}, "
            f"found {len(summary_paths)} in {output_root}"
        )

    rows = []
    for summary_path in summary_paths:
        batch_name, animal = parse_animal_dir_name(summary_path.parent)
        summary_df = pd.read_csv(summary_path)

        for param_name, _ylabel, scale in PARAMS:
            param_rows = summary_df[summary_df["parameter"] == param_name]
            if len(param_rows) != 1:
                raise RuntimeError(f"{summary_path} has {len(param_rows)} rows for {param_name}")
            row = param_rows.iloc[0]
            rows.append(
                {
                    "model": model_label,
                    "batch_name": batch_name,
                    "animal": animal,
                    "animal_label": f"{batch_name}/{animal}",
                    "parameter": param_name,
                    "mean": float(row["mean"]) * scale,
                    "q025": float(row["q025"]) * scale,
                    "q975": float(row["q975"]) * scale,
                    "unit": "%" if param_name == "lapse_prob" else "probability",
                }
            )
    return rows


# %%
# =============================================================================
# Load posterior summaries
# =============================================================================
rows = []
for model_label in MODEL_ORDER:
    rows.extend(load_lapse_params(MODEL_ROOTS[model_label], model_label))

params_df = pd.DataFrame(rows)
expected_rows = EXPECTED_N_ANIMALS * len(PARAMS) * len(MODEL_ORDER)
if len(params_df) != expected_rows:
    raise RuntimeError(f"Expected {expected_rows} parameter rows, found {len(params_df)}")
if params_df.duplicated(["model", "batch_name", "animal", "parameter"]).any():
    raise RuntimeError("Duplicate model/animal/parameter rows found")
if not np.all(np.isfinite(params_df[["mean", "q025", "q975"]].to_numpy(dtype=float))):
    raise RuntimeError("Parameter values contain NaN/Inf")

animal_sets = {}
for model in MODEL_ORDER:
    animal_sets[model] = set(
        params_df.loc[params_df["model"] == model, ["batch_name", "animal"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
reference_animals = animal_sets[MODEL_NPL_LAPSE]
for model, model_animals in animal_sets.items():
    if model_animals != reference_animals:
        missing_from_model = sorted(reference_animals - model_animals, key=animal_sort_key)
        extra_in_model = sorted(model_animals - reference_animals, key=animal_sort_key)
        raise RuntimeError(
            f"Animal set mismatch for {model}. Missing: {missing_from_model}; extra: {extra_in_model}"
        )

animal_keys = sorted(reference_animals, key=animal_sort_key)
animal_labels = [f"{batch}/{animal}" for batch, animal in animal_keys]
if len(animal_keys) != EXPECTED_N_ANIMALS:
    raise RuntimeError(f"Expected {EXPECTED_N_ANIMALS} matched animals, found {len(animal_keys)}")

params_df["model_order"] = params_df["model"].map({model: idx for idx, model in enumerate(MODEL_ORDER)})
params_df = params_df.sort_values(["model_order", "batch_name", "animal", "parameter"]).reset_index(drop=True)
params_df.drop(columns=["model_order"]).to_csv(VALUES_CSV, index=False)

print(f"Loaded lapse parameters for {len(animal_keys)} animals and {len(MODEL_ORDER)} models")
print(params_df.groupby(["model", "parameter"])["mean"].describe().to_string())
print(f"Saved values: {VALUES_CSV}")


# %%
# =============================================================================
# Plot 1 x 2 lapse-parameter panels
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(17, 5.8), constrained_layout=True)
x = np.arange(len(animal_labels))

for ax, (param_name, ylabel, _scale) in zip(axes, PARAMS):
    for model in MODEL_ORDER:
        sub = (
            params_df[(params_df["model"] == model) & (params_df["parameter"] == param_name)]
            .set_index(["batch_name", "animal"])
            .loc[animal_keys]
            .reset_index()
        )
        y = sub["mean"].to_numpy(dtype=float)
        yerr = np.vstack(
            [
                y - sub["q025"].to_numpy(dtype=float),
                sub["q975"].to_numpy(dtype=float) - y,
            ]
        )
        ax.errorbar(
            x + MODEL_OFFSETS[model],
            y,
            yerr=yerr,
            fmt=MODEL_MARKERS[model],
            color=MODEL_COLORS[model],
            ecolor=MODEL_COLORS[model],
            elinewidth=1.0,
            capsize=2.5,
            markersize=5.8,
            markeredgewidth=1.6,
            alpha=0.9,
            label=model,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(animal_labels, rotation=55, ha="right", fontsize=8)
    ax.set_xlabel("Animal")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.grid(axis="y", alpha=0.25)
    if param_name == "lapse_prob_right":
        ax.set_ylim(0.0, 1.0)

legend_handles = [
    Line2D(
        [0],
        [0],
        color=MODEL_COLORS[model],
        marker=MODEL_MARKERS[model],
        linestyle="none",
        label=model,
        markersize=7,
        markeredgewidth=1.6,
    )
    for model in MODEL_ORDER
]
axes[0].legend(handles=legend_handles, frameon=False, loc="best", fontsize=9)
fig.suptitle("Patience-12 min-50k NPL+alpha+lapse and IPL+lapse SVI lapse parameters", fontsize=14)
fig.savefig(FIG_PATH, dpi=200, bbox_inches="tight")

print(f"Saved figure: {FIG_PATH}")

plt.show()


# %%
