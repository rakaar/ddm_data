# %%
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


# %%
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

ILD2_DIAGNOSTIC_DIR = os.path.join(REPO_DIR, "NPL_alpha_ILD2_fit_results", "figure_4_diagnostics")

EMPIRICAL_PSY_PKL = os.path.join(ILD2_DIAGNOSTIC_DIR, "empirical_psychometric_data_for_npl_alpha_ild2.pkl")
IPL_THEORETICAL_PSY_PKL = os.path.join(SCRIPT_DIR, "theoretical_psychometric_data_vanilla.pkl")
NPL_THEORETICAL_PSY_PKL = os.path.join(SCRIPT_DIR, "theoretical_psychometric_data_norm.pkl")
ILD2_THEORETICAL_PSY_PKL = os.path.join(ILD2_DIAGNOSTIC_DIR, "theoretical_psychometric_data_npl_alpha_ild2.pkl")

OUTPUT_PNG = os.path.join(ILD2_DIAGNOSTIC_DIR, "permutation_test_model_vs_data_accuracy.png")
OUTPUT_PDF = os.path.join(ILD2_DIAGNOSTIC_DIR, "permutation_test_model_vs_data_accuracy.pdf")

ABL_ARR = [20, 40, 60]
N_SHUFFLES = 100_000
RNG_SEED = 0
N_BINS = 60

MODEL_CONFIGS = {
    "IPL / vanilla TIED": {
        "color": "tab:green",
        "pkl": IPL_THEORETICAL_PSY_PKL,
    },
    "NPL": {
        "color": "tab:blue",
        "pkl": NPL_THEORETICAL_PSY_PKL,
    },
    "NPL + alpha + ILD2": {
        "color": "tab:red",
        "pkl": ILD2_THEORETICAL_PSY_PKL,
    },
}


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


def mean_accuracy_on_empirical_ilds(psy_data, animal_key, data_key, empirical_psy_data):
    accuracies = []
    for ABL in ABL_ARR:
        empirical_condition = empirical_psy_data.get(animal_key, {}).get(ABL, {}).get("empirical", None)
        model_condition = psy_data.get(animal_key, {}).get(ABL, {}).get(data_key, None)
        if empirical_condition is None or model_condition is None:
            continue

        empirical_ild_values = np.asarray(empirical_condition["ild_values"], dtype=float)
        for ILD in empirical_ild_values:
            right_prob = get_right_prob_at_ild(model_condition, ILD)
            accuracies.append(signed_accuracy_from_right_prob(ILD, right_prob))

    accuracies = np.asarray(accuracies, dtype=float)
    if accuracies.size == 0 or np.all(~np.isfinite(accuracies)):
        return np.nan
    return float(np.nanmean(accuracies))


def paired_label_swap_test(model_accuracy, data_accuracy, rng):
    model_accuracy = np.asarray(model_accuracy, dtype=float)
    data_accuracy = np.asarray(data_accuracy, dtype=float)
    finite = np.isfinite(model_accuracy) & np.isfinite(data_accuracy)
    diff_values = model_accuracy[finite] - data_accuracy[finite]

    true_stat = float(np.mean(diff_values))
    signs = rng.choice([-1.0, 1.0], size=(N_SHUFFLES, len(diff_values)))
    null_stats = np.mean(signs * diff_values, axis=1)

    p_two_sided = (np.sum(np.abs(null_stats) >= abs(true_stat)) + 1) / (N_SHUFFLES + 1)
    p_model_gt_data = (np.sum(null_stats >= true_stat) + 1) / (N_SHUFFLES + 1)
    p_model_lt_data = (np.sum(null_stats <= true_stat) + 1) / (N_SHUFFLES + 1)

    ttest_result = ttest_rel(model_accuracy[finite], data_accuracy[finite], nan_policy="omit")
    try:
        wilcoxon_result = wilcoxon(model_accuracy[finite], data_accuracy[finite], alternative="two-sided")
        wilcoxon_p = float(wilcoxon_result.pvalue)
    except ValueError:
        wilcoxon_p = np.nan

    return {
        "n_animals": int(len(diff_values)),
        "data_mean": float(np.mean(data_accuracy[finite])),
        "model_mean": float(np.mean(model_accuracy[finite])),
        "diff_values": diff_values,
        "true_stat": true_stat,
        "null_stats": null_stats,
        "p_two_sided": float(p_two_sided),
        "p_model_gt_data": float(p_model_gt_data),
        "p_model_lt_data": float(p_model_lt_data),
        "paired_t_stat": float(ttest_result.statistic),
        "paired_t_p": float(ttest_result.pvalue),
        "wilcoxon_p": wilcoxon_p,
    }


# %%
with open(EMPIRICAL_PSY_PKL, "rb") as handle:
    empirical_psy_data = pickle.load(handle)

model_psy_data_by_label = {}
common_animal_sets = [set(empirical_psy_data)]
for model_label, config in MODEL_CONFIGS.items():
    with open(config["pkl"], "rb") as handle:
        model_psy_data = pickle.load(handle)
    model_psy_data_by_label[model_label] = model_psy_data
    common_animal_sets.append(set(model_psy_data))

animal_keys = sorted(set.intersection(*common_animal_sets), key=lambda item: (item[0], item[1]))
sd_animals = [animal_key for animal_key in animal_keys if animal_key[0] == "SD"]
print(f"Using {len(animal_keys)} animals")
print(f"SD animals included: {sd_animals}")
for animal_key in sd_animals:
    sd_ilds = sorted(
        set(
            np.concatenate([
                np.asarray(empirical_psy_data[animal_key][ABL]["empirical"]["ild_values"], dtype=float)
                for ABL in ABL_ARR
            ])
        )
    )
    print(f"  {animal_key[0]}-{animal_key[1]} empirical ILDs used for model accuracy: {sd_ilds}")


# %%
data_accuracy = np.asarray([
    mean_accuracy_on_empirical_ilds(empirical_psy_data, animal_key, "empirical", empirical_psy_data)
    for animal_key in animal_keys
], dtype=float)

rng = np.random.default_rng(RNG_SEED)
plot_results = {}
summary_rows = []

for model_label, model_psy_data in model_psy_data_by_label.items():
    model_accuracy = np.asarray([
        mean_accuracy_on_empirical_ilds(model_psy_data, animal_key, "theoretical", empirical_psy_data)
        for animal_key in animal_keys
    ], dtype=float)
    result = paired_label_swap_test(model_accuracy, data_accuracy, rng)
    result["model_accuracy"] = model_accuracy
    result["data_accuracy"] = data_accuracy
    plot_results[model_label] = result

    summary_rows.append({
        "model": model_label,
        "n_animals": result["n_animals"],
        "data_mean_accuracy": result["data_mean"],
        "model_mean_accuracy": result["model_mean"],
        "mean_model_minus_data": result["true_stat"],
        "p_two_sided": result["p_two_sided"],
        "p_model_gt_data": result["p_model_gt_data"],
        "p_model_lt_data": result["p_model_lt_data"],
        "paired_t_stat": result["paired_t_stat"],
        "paired_t_p": result["paired_t_p"],
        "wilcoxon_p": result["wilcoxon_p"],
    })

summary_df = pd.DataFrame(summary_rows)
print("\nLabel-swap accuracy test summary: statistic = mean(model accuracy - data accuracy)")
print(summary_df.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


# %%
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True)

for ax, (model_label, result) in zip(axes, plot_results.items()):
    color = MODEL_CONFIGS[model_label]["color"]
    null_stats = result["null_stats"]
    true_stat = result["true_stat"]

    ax.hist(null_stats, bins=N_BINS, color="0.7", edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.85)
    ax.axvline(true_stat, color=color, linestyle="-", linewidth=2.8)
    ax.set_title(
        f"{model_label}\n"
        f"mean(model-data)={true_stat:+.4f}, n={result['n_animals']}\n"
        f"p2={result['p_two_sided']:.4g}, "
        f"p(model>data)={result['p_model_gt_data']:.4g}, "
        f"p(model<data)={result['p_model_lt_data']:.4g}\n"
        f"paired t p={result['paired_t_p']:.4g}, "
        f"Wilcoxon p={result['wilcoxon_p']:.4g}",
        fontsize=9.5,
    )
    ax.set_xlabel("Shuffled mean(model accuracy - data accuracy)", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Shuffle count", fontsize=10)
fig.suptitle(
    "Paired Label-Swap Test: Model Accuracy vs Data Accuracy\n"
    "Model accuracy evaluated only on empirical ABL/ILD conditions",
    fontsize=13,
    y=1.05,
)
fig.subplots_adjust(left=0.07, right=0.99, bottom=0.17, top=0.74, wspace=0.26)
fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUTPUT_PDF, dpi=300, bbox_inches="tight")

print(f"\nSaved {OUTPUT_PNG}")
print(f"Saved {OUTPUT_PDF}")

# %%
