# %%
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np


# %%
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

ILD2_DIAGNOSTIC_DIR = os.path.join(REPO_DIR, "NPL_alpha_ILD2_fit_results", "figure_4_diagnostics")

EMPIRICAL_PSY_PKL = os.path.join(ILD2_DIAGNOSTIC_DIR, "empirical_psychometric_data_for_npl_alpha_ild2.pkl")
NPL_THEORETICAL_PSY_PKL = os.path.join(SCRIPT_DIR, "theoretical_psychometric_data_norm.pkl")
ILD2_THEORETICAL_PSY_PKL = os.path.join(ILD2_DIAGNOSTIC_DIR, "theoretical_psychometric_data_npl_alpha_ild2.pkl")

OUTPUT_PNG = os.path.join(ILD2_DIAGNOSTIC_DIR, "npl_vs_ild2_accuracy_percent_change.png")
OUTPUT_PDF = os.path.join(ILD2_DIAGNOSTIC_DIR, "npl_vs_ild2_accuracy_percent_change.pdf")

ABL_ARR = [20, 40, 60]


# %%
def get_right_prob_at_ild(condition_data, ILD):
    ild_values = np.asarray(condition_data["ild_values"], dtype=float)
    right_choice_probs = np.asarray(condition_data["right_choice_probs"], dtype=float)
    matches = np.where(np.isclose(ild_values, ILD))[0]
    if len(matches) == 0:
        return np.nan
    return float(right_choice_probs[matches[0]])


def signed_accuracy_from_right_prob(ILD, right_prob):
    if not np.isfinite(right_prob):
        return np.nan
    if ILD > 0:
        return right_prob
    if ILD < 0:
        return 1 - right_prob
    return np.nan


def mean_model_accuracy_on_empirical_conditions(model_psy_data, animal_key, empirical_psy_data):
    accuracies = []
    empirical_grids = {}
    for ABL in ABL_ARR:
        empirical_condition = empirical_psy_data.get(animal_key, {}).get(ABL, {}).get("empirical", None)
        model_condition = model_psy_data.get(animal_key, {}).get(ABL, {}).get("theoretical", None)
        if empirical_condition is None or model_condition is None:
            continue

        empirical_ilds = np.asarray(empirical_condition["ild_values"], dtype=float)
        empirical_grids[ABL] = tuple(float(ILD) for ILD in empirical_ilds)
        for ILD in empirical_ilds:
            right_prob = get_right_prob_at_ild(model_condition, ILD)
            accuracies.append(signed_accuracy_from_right_prob(ILD, right_prob))

    accuracies = np.asarray(accuracies, dtype=float)
    if accuracies.size == 0 or np.all(~np.isfinite(accuracies)):
        return np.nan, empirical_grids
    return float(np.nanmean(accuracies)), empirical_grids


# %%
with open(EMPIRICAL_PSY_PKL, "rb") as handle:
    empirical_psy_data = pickle.load(handle)
with open(NPL_THEORETICAL_PSY_PKL, "rb") as handle:
    npl_theoretical_psy_data = pickle.load(handle)
with open(ILD2_THEORETICAL_PSY_PKL, "rb") as handle:
    ild2_theoretical_psy_data = pickle.load(handle)

animal_keys = sorted(
    set(empirical_psy_data) & set(npl_theoretical_psy_data) & set(ild2_theoretical_psy_data),
    key=lambda item: (item[0], item[1]),
)

npl_accuracy = []
ild2_accuracy = []
empirical_grids_by_animal = {}
for animal_key in animal_keys:
    npl_acc, empirical_grid = mean_model_accuracy_on_empirical_conditions(
        npl_theoretical_psy_data,
        animal_key,
        empirical_psy_data,
    )
    ild2_acc, _ = mean_model_accuracy_on_empirical_conditions(
        ild2_theoretical_psy_data,
        animal_key,
        empirical_psy_data,
    )
    npl_accuracy.append(npl_acc)
    ild2_accuracy.append(ild2_acc)
    empirical_grids_by_animal[animal_key] = empirical_grid

npl_accuracy = np.asarray(npl_accuracy, dtype=float)
ild2_accuracy = np.asarray(ild2_accuracy, dtype=float)
accuracy_diff = ild2_accuracy - npl_accuracy
percent_change = 100 * accuracy_diff / npl_accuracy
animal_labels = np.asarray([f"{batch}-{animal_id}" for batch, animal_id in animal_keys])

print(f"Using {len(animal_keys)} animals")
print("SD empirical ILDs used for model accuracy:")
for animal_key in animal_keys:
    if animal_key[0] != "SD":
        continue
    grid_strings = []
    for ABL in ABL_ARR:
        ilds = empirical_grids_by_animal[animal_key].get(ABL, ())
        grid_strings.append(f"ABL {ABL}: {list(ilds)}")
    print(f"  {animal_key[0]}-{animal_key[1]}: " + "; ".join(grid_strings))

negative_indices = np.where(np.isfinite(percent_change) & (percent_change < 0))[0]
negative_indices = negative_indices[np.argsort(percent_change[negative_indices])]

print("\nAnimals where NPL + alpha + ILD2 accuracy is lower than NPL:")
for idx in negative_indices:
    print(
        f"  {animal_labels[idx]}: "
        f"NPL={npl_accuracy[idx]:.4f}, "
        f"ILD2={ild2_accuracy[idx]:.4f}, "
        f"diff={accuracy_diff[idx]:+.4f}, "
        f"percent_change={percent_change[idx]:+.2f}%"
    )


# %%
sort_idx = np.argsort(percent_change)
sorted_labels = animal_labels[sort_idx]
sorted_percent_change = percent_change[sort_idx]
sorted_npl_accuracy = npl_accuracy[sort_idx]
sorted_ild2_accuracy = ild2_accuracy[sort_idx]

fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))

ax = axes[0]
below_npl = ild2_accuracy < npl_accuracy
point_colors = np.where(below_npl, "tab:red", "tab:green")
ax.scatter(
    npl_accuracy,
    ild2_accuracy,
    s=72,
    c=point_colors,
    alpha=0.75,
    edgecolors="none",
)
lim_left = 0.76
lim_right = 0.88
ax.plot([lim_left, lim_right], [lim_left, lim_right], color="grey", linestyle="--", linewidth=1.5, alpha=0.6)
ax.set_xlim(lim_left, lim_right)
ax.set_ylim(lim_left, lim_right)
ax.set_xticks([0.76, 0.82, 0.88])
ax.set_yticks([0.76, 0.82, 0.88])
ax.set_xlabel("NPL model accuracy", fontsize=10)
ax.set_ylabel("NPL + alpha + ILD2 model accuracy", fontsize=10)
ax.set_title("Model Accuracy on Empirical Stimulus Grids", fontsize=11)
ax.tick_params(axis="both", labelsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_box_aspect(1)

ax = axes[1]
x = np.arange(len(sorted_labels))
bar_colors = np.where(sorted_percent_change < 0, "tab:red", "tab:green")
ax.scatter(
    x,
    sorted_percent_change,
    s=56,
    c=bar_colors,
    alpha=0.85,
    edgecolors="none",
)
for local_idx, sorted_idx in enumerate(sort_idx):
    ax.plot(
        [local_idx, local_idx],
        [0, sorted_percent_change[local_idx]],
        color=bar_colors[local_idx],
        alpha=0.32,
        linewidth=1.4,
    )
ax.axhline(0, color="grey", linestyle="--", linewidth=1.2, alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(sorted_labels, rotation=90, fontsize=7)
ax.set_ylabel("Percent change vs NPL (%)", fontsize=10)
ax.set_title("100 x (ILD2 accuracy - NPL accuracy) / NPL accuracy", fontsize=11)
ax.tick_params(axis="y", labelsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.suptitle(
    "NPL vs NPL + alpha + ILD2 Accuracy\n"
    "Accuracy is evaluated only at each animal's empirical ABL/ILD conditions; SD uses |ILD| = 1, 2, 4, 8",
    fontsize=12,
    y=0.98,
)
fig.subplots_adjust(left=0.07, right=0.99, bottom=0.27, top=0.80, wspace=0.30)
fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUTPUT_PDF, dpi=300, bbox_inches="tight")

print(f"\nSaved {OUTPUT_PNG}")
print(f"Saved {OUTPUT_PDF}")

# %%
