# %%
"""
Compare fixed condition-fit w with NPL+alpha+ILD2 animal-wise w.

The 4-param condition fits use w fixed to each animal's mean_w from the
5-param condition-fit summary. The NPL+alpha+ILD2 model estimates one animal-wise
w, so this comparison is animal-wise rather than ABL/ILD-wise.
"""
import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from gamma_omega_alpha_utils import load_batch_animal_pairs, print_batch_animal_table


# %% Parameters
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DESIRED_BATCHES = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]
BATCH_DIR = os.path.join(REPO_DIR, "fit_animal_by_animal", "batch_csvs")
BATCH_ORDER = {batch: idx for idx, batch in enumerate(DESIRED_BATCHES)}

COND_FIT_SOURCE = "gamma_omega_t_E_aff_del_go_fix_w_mean"
W_SUMMARY_CSV_PATH = os.path.join(SCRIPT_DIR, "five_param_w_mean_median_by_animal.csv")

MODEL_RESULTS_DIR = os.path.join(REPO_DIR, "NPL_alpha_ILD2_fit_results", "result_pkls")
MODEL_RESULT_KEY = "vbmc_norm_alpha_ild2_delay_tied_results"
MODEL_LABEL = "NPL + alpha + ILD2"
MODEL_RESULT_PATTERN = re.compile(
    r"^results_(?P<batch>.+)_animal_(?P<animal>\d+)_NORM_ALPHA_ILD2_DELAY_FROM_ABORTS\.pkl$"
)

OUTPUT_DIR = os.path.join(REPO_DIR, "NPL_alpha_ILD2_fit_results", "param_comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FIG_PATH = os.path.join(OUTPUT_DIR, "fixed_w_vs_npl_alpha_ild2_w.png")
CSV_PATH = os.path.join(OUTPUT_DIR, "fixed_w_vs_npl_alpha_ild2_w.csv")


# %% Helpers
def parse_model_result_filename(fname):
    match = MODEL_RESULT_PATTERN.match(fname)
    if match is None:
        return None
    return match.group("batch"), int(match.group("animal"))


def load_model_result_paths():
    result_paths = {}
    for fname in os.listdir(MODEL_RESULTS_DIR):
        parsed = parse_model_result_filename(fname)
        if parsed is None:
            continue
        result_paths[parsed] = os.path.join(MODEL_RESULTS_DIR, fname)
    return result_paths


def load_model_w_mean(pkl_path):
    with open(pkl_path, "rb") as f:
        saved_data = pickle.load(f)

    if MODEL_RESULT_KEY not in saved_data:
        raise KeyError(f"{pkl_path} is missing `{MODEL_RESULT_KEY}`")

    model_results = saved_data[MODEL_RESULT_KEY]
    if "w_samples" not in model_results:
        raise KeyError(f"{pkl_path} is missing `w_samples`")

    return float(np.mean(np.asarray(model_results["w_samples"], dtype=float)))


def finite_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 2:
        return np.nan
    if np.nanstd(x[valid]) == 0 or np.nanstd(y[valid]) == 0:
        return np.nan
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


# %%
# Load matched animals with fixed condition-fit w and model w.
print(f"Using condition-fit source: {COND_FIT_SOURCE}")
print(f"Condition fixed-w summary: {W_SUMMARY_CSV_PATH}")
print(f"Model result directory: {MODEL_RESULTS_DIR}")

all_batch_animal_pairs = load_batch_animal_pairs(BATCH_DIR, DESIRED_BATCHES)
model_result_paths = load_model_result_paths()

w_summary_df = pd.read_csv(W_SUMMARY_CSV_PATH)
w_summary_df["animal"] = w_summary_df["animal"].astype(str)
w_by_batch_animal = {
    (row["batch_name"], row["animal"]): {
        "condition_fixed_mean_w": float(row["mean_w"]),
        "condition_fixed_median_w": float(row["median_w"]),
        "n_conditions_with_w": int(row["n_conditions_with_w"]),
        "n_conditions_expected": int(row["n_conditions_expected"]),
    }
    for _, row in w_summary_df.iterrows()
}

matched_pairs = [
    (batch_name, int(animal_id))
    for batch_name, animal_id in all_batch_animal_pairs
    if (batch_name, int(animal_id)) in model_result_paths
    and (batch_name, str(int(animal_id))) in w_by_batch_animal
]
matched_pairs = sorted(
    matched_pairs,
    key=lambda pair: (BATCH_ORDER.get(pair[0], len(BATCH_ORDER)), pair[1]),
)

print_batch_animal_table(matched_pairs)
print(f"Matched animals with both fixed condition w and {MODEL_LABEL} w: {len(matched_pairs)}")
if len(matched_pairs) == 0:
    raise RuntimeError("No animals have both fixed condition w and model result pickles.")

rows = []
for batch_name, animal_id in matched_pairs:
    fixed_w_info = w_by_batch_animal[(batch_name, str(animal_id))]
    model_w = load_model_w_mean(model_result_paths[(batch_name, animal_id)])
    condition_w = fixed_w_info["condition_fixed_mean_w"]

    rows.append(
        {
            "batch_name": batch_name,
            "animal": animal_id,
            "condition_fixed_mean_w": condition_w,
            "condition_fixed_median_w": fixed_w_info["condition_fixed_median_w"],
            "n_conditions_with_w": fixed_w_info["n_conditions_with_w"],
            "n_conditions_expected": fixed_w_info["n_conditions_expected"],
            "npl_alpha_ild2_w": model_w,
            "delta_model_minus_condition_w": model_w - condition_w,
        }
    )

comparison_df = pd.DataFrame(rows)
comparison_df = comparison_df.sort_values(
    ["condition_fixed_mean_w", "batch_name", "animal"],
    ignore_index=True,
)
comparison_df.to_csv(CSV_PATH, index=False)
print(f"Saved comparison CSV: {CSV_PATH}")


# %%
# Summary statistics.
condition_w_values = comparison_df["condition_fixed_mean_w"].to_numpy(dtype=float)
model_w_values = comparison_df["npl_alpha_ild2_w"].to_numpy(dtype=float)
delta_values = comparison_df["delta_model_minus_condition_w"].to_numpy(dtype=float)

n_animals = len(comparison_df)
condition_mean = float(np.nanmean(condition_w_values))
model_mean = float(np.nanmean(model_w_values))
delta_mean = float(np.nanmean(delta_values))
rmse = float(np.sqrt(np.nanmean(delta_values**2)))
corr = finite_corr(condition_w_values, model_w_values)

print("w comparison summary:")
print(f"  n animals: {n_animals}")
print(f"  condition fixed mean_w: {condition_mean:.6g}")
print(f"  {MODEL_LABEL} w: {model_mean:.6g}")
print(f"  model - condition mean delta: {delta_mean:.6g}")
print(f"  RMSE: {rmse:.6g}")
print(f"  Pearson r: {corr:.6g}")


# %%
# Plot animal-wise paired values and model-vs-condition scatter.
batches = list(dict.fromkeys(comparison_df["batch_name"].tolist()))
cmap = plt.get_cmap("tab10")
batch_colors = {batch: cmap(idx % 10) for idx, batch in enumerate(batches)}
point_colors = [batch_colors[batch] for batch in comparison_df["batch_name"]]
x_positions = np.arange(n_animals)

all_w_values = np.concatenate([condition_w_values, model_w_values])
pad = max(0.02, 0.08 * (np.nanmax(all_w_values) - np.nanmin(all_w_values)))
w_min = float(np.nanmin(all_w_values) - pad)
w_max = float(np.nanmax(all_w_values) + pad)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))

for idx, row in comparison_df.iterrows():
    color = batch_colors[row["batch_name"]]
    ax[0].plot(
        [idx, idx],
        [row["condition_fixed_mean_w"], row["npl_alpha_ild2_w"]],
        color=color,
        alpha=0.25,
        linewidth=1,
        zorder=1,
    )

ax[0].scatter(
    x_positions,
    condition_w_values,
    marker="o",
    s=45,
    facecolors="white",
    edgecolors=point_colors,
    linewidths=1.5,
    label="condition fixed mean_w",
    zorder=3,
)
ax[0].scatter(
    x_positions,
    model_w_values,
    marker="s",
    s=38,
    c=point_colors,
    label=f"{MODEL_LABEL} w",
    zorder=4,
)
ax[0].axhline(0.5, color="0.55", linestyle="--", linewidth=1, alpha=0.7)
ax[0].set_xlabel("animal, sorted by condition fixed mean_w")
ax[0].set_ylabel("w")
ax[0].set_ylim(w_min, w_max)
ax[0].set_title("Animal-wise paired w")
ax[0].grid(True, alpha=0.25)

ax[1].scatter(
    condition_w_values,
    model_w_values,
    c=point_colors,
    s=55,
    edgecolors="black",
    linewidths=0.5,
)
ax[1].plot([w_min, w_max], [w_min, w_max], color="0.35", linestyle="--", linewidth=1.5)
ax[1].axvline(0.5, color="0.75", linestyle=":", linewidth=1)
ax[1].axhline(0.5, color="0.75", linestyle=":", linewidth=1)
ax[1].set_xlim(w_min, w_max)
ax[1].set_ylim(w_min, w_max)
ax[1].set_xlabel("condition fixed mean_w")
ax[1].set_ylabel(f"{MODEL_LABEL} w")
ax[1].set_title("Model w vs fixed condition-fit w")
ax[1].grid(True, alpha=0.25)

batch_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor=color,
        markeredgecolor="black",
        markersize=6,
        label=batch,
    )
    for batch, color in batch_colors.items()
]
source_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="black",
        markerfacecolor="white",
        linestyle="none",
        markersize=6,
        label="condition fixed mean_w",
    ),
    Line2D(
        [0],
        [0],
        marker="s",
        color="black",
        markerfacecolor="black",
        linestyle="none",
        markersize=6,
        label=f"{MODEL_LABEL} w",
    ),
]
ax[0].legend(handles=source_handles, fontsize=8, loc="best")
ax[1].legend(handles=batch_handles, fontsize=8, loc="best", title="Batch", title_fontsize=8)

fig.suptitle(
    "Fixed condition-fit w vs NPL+alpha+ILD2 w\n"
    f"n={n_animals}; condition mean={condition_mean:.3f}; model mean={model_mean:.3f}; "
    f"mean delta={delta_mean:+.3f}; RMSE={rmse:.3f}; r={corr:.3f}"
)
fig.tight_layout(rect=[0, 0, 1, 0.88])
fig.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
print(f"Saved figure: {FIG_PATH}")

plt.show()

# %%
