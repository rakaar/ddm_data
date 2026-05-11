# %%
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figure_template as ft


# %%
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

ABL_ARR = [20, 40, 60]
N_SHUFFLES = 100_000
RNG_SEED = 0
N_BINS = 60

ILD2_DIAGNOSTIC_DIR = os.path.join(REPO_DIR, "NPL_alpha_ILD2_fit_results", "figure_4_diagnostics")
NPL_PSY_PKL = os.path.join(SCRIPT_DIR, "theoretical_psychometric_data_norm.pkl")
ILD2_PSY_PKL = os.path.join(ILD2_DIAGNOSTIC_DIR, "theoretical_psychometric_data_npl_alpha_ild2.pkl")

OUTPUT_PNG = os.path.join(ILD2_DIAGNOSTIC_DIR, "permutation_test_ild2_minus_npl_slopes.png")


# %%
def load_psychometric_fit_data():
    with open(NPL_PSY_PKL, "rb") as handle:
        npl_psychometric_data = pickle.load(handle)
    with open(ILD2_PSY_PKL, "rb") as handle:
        ild2_psychometric_data = pickle.load(handle)
    return npl_psychometric_data, ild2_psychometric_data


def extract_slopes(data_dict):
    slopes = {}
    for batch_animal, abl_dict in data_dict.items():
        slopes[batch_animal] = {}
        for ABL in ABL_ARR:
            fit = abl_dict.get(ABL, {}).get("fit")
            if fit is not None and "params" in fit:
                slopes[batch_animal][ABL] = float(fit["params"][3])
            else:
                slopes[batch_animal][ABL] = np.nan
    return slopes


def paired_slopes_for_abl(slopes_npl, slopes_ild2, ABL):
    common_pairs = sorted(set(slopes_npl) & set(slopes_ild2), key=lambda item: (item[0], item[1]))
    matched_pairs, npl_values, ild2_values = [], [], []

    for pair in common_pairs:
        npl_slope = slopes_npl[pair].get(ABL, np.nan)
        ild2_slope = slopes_ild2[pair].get(ABL, np.nan)
        if np.isfinite(npl_slope) and np.isfinite(ild2_slope):
            matched_pairs.append(pair)
            npl_values.append(npl_slope)
            ild2_values.append(ild2_slope)

    return matched_pairs, np.asarray(npl_values, dtype=float), np.asarray(ild2_values, dtype=float)


def permutation_test_paired_mean(diff_values, rng):
    true_stat = float(np.nanmean(diff_values))
    signs = rng.choice([-1.0, 1.0], size=(N_SHUFFLES, len(diff_values)))
    null_stats = np.mean(signs * diff_values, axis=1)
    p_value = (np.sum(np.abs(null_stats) >= abs(true_stat)) + 1) / (N_SHUFFLES + 1)
    return true_stat, null_stats, float(p_value)


# %%
npl_psychometric_data, ild2_psychometric_data = load_psychometric_fit_data()
slopes_npl = extract_slopes(npl_psychometric_data)
slopes_ild2 = extract_slopes(ild2_psychometric_data)

rng = np.random.default_rng(RNG_SEED)
summary_rows = []
plot_results = {}

for ABL in ABL_ARR:
    matched_pairs, npl_values, ild2_values = paired_slopes_for_abl(slopes_npl, slopes_ild2, ABL)
    diff_values = ild2_values - npl_values
    true_stat, null_stats, p_value = permutation_test_paired_mean(diff_values, rng)

    plot_results[ABL] = {
        "matched_pairs": matched_pairs,
        "npl_values": npl_values,
        "ild2_values": ild2_values,
        "diff_values": diff_values,
        "true_stat": true_stat,
        "null_stats": null_stats,
        "p_value": p_value,
    }
    summary_rows.append({
        "ABL": ABL,
        "n_animals": len(matched_pairs),
        "mean_npl_slope": float(np.nanmean(npl_values)),
        "mean_ild2_slope": float(np.nanmean(ild2_values)),
        "true_diff_ild2_minus_npl": true_stat,
        "p_value": p_value,
    })


# %%
summary_df = pd.DataFrame(summary_rows)
print("\nPermutation test summary: statistic = mean(ILD2 slope - NPL slope)")
print(summary_df.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


# %%
fig, axes = plt.subplots(1, len(ABL_ARR), figsize=(15, 4.8), sharey=True)

for ax, ABL in zip(axes, ABL_ARR):
    result = plot_results[ABL]
    null_stats = result["null_stats"]
    true_stat = result["true_stat"]
    p_value = result["p_value"]
    n_animals = len(result["matched_pairs"])

    ax.hist(null_stats, bins=N_BINS, color="0.65", edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.axvline(true_stat, color="tab:red", linestyle="-", linewidth=2.5)
    ax.set_title(
        f"ABL={ABL}\nmean diff={true_stat:.4f}, p={p_value:.4g}, n={n_animals}",
        fontsize=ft.STYLE.LEGEND_FONTSIZE,
    )
    ax.set_xlabel("Mean slope diff (ILD2 - NPL)", fontsize=ft.STYLE.LEGEND_FONTSIZE)
    ax.tick_params(axis="both", labelsize=ft.STYLE.LEGEND_FONTSIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Shuffle count", fontsize=ft.STYLE.LEGEND_FONTSIZE)
fig.suptitle(
    "permutation test for psychometric slopes",
    fontsize=ft.STYLE.LABEL_FONTSIZE,
    y=1.03,
)

plt.tight_layout()
fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
print(f"\nSaved {OUTPUT_PNG}")

# %%
