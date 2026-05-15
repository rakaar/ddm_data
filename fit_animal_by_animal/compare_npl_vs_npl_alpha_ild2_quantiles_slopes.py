# %%
import os
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import figure_template as ft


# %%
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

ILD2_DIAGNOSTIC_DIR = os.path.join(REPO_DIR, "NPL_alpha_ILD2_fit_results", "figure_4_diagnostics")

NPL_QUANT_PKL = os.path.join(SCRIPT_DIR, "norm_quant_fig2_data.pkl")
NPL_SLOPES_PKL = os.path.join(SCRIPT_DIR, "norm_slopes_fig2_data.pkl")
VANILLA_SLOPES_PKL = os.path.join(SCRIPT_DIR, "vanilla_slopes_fig2_data.pkl")
ILD2_QUANT_PKL = os.path.join(ILD2_DIAGNOSTIC_DIR, "npl_alpha_ild2_quant_fig4_data.pkl")
ILD2_SLOPES_PKL = os.path.join(ILD2_DIAGNOSTIC_DIR, "npl_alpha_ild2_slopes_fig4_data.pkl")
VANILLA_THEORETICAL_PSY_PKL = os.path.join(SCRIPT_DIR, "theoretical_psychometric_data_vanilla.pkl")
NPL_THEORETICAL_PSY_PKL = os.path.join(SCRIPT_DIR, "theoretical_psychometric_data_norm.pkl")
ILD2_THEORETICAL_PSY_PKL = os.path.join(ILD2_DIAGNOSTIC_DIR, "theoretical_psychometric_data_npl_alpha_ild2.pkl")
EMPIRICAL_PSY_PKL = os.path.join(ILD2_DIAGNOSTIC_DIR, "empirical_psychometric_data_for_npl_alpha_ild2.pkl")

OUTPUT_PNG = os.path.join(ILD2_DIAGNOSTIC_DIR, "compare_npl_vs_npl_alpha_ild2_quantiles_slopes.png")
OUTPUT_PDF = os.path.join(ILD2_DIAGNOSTIC_DIR, "compare_npl_vs_npl_alpha_ild2_quantiles_slopes.pdf")
SLOPE_ACCURACY_OUTPUT_PNG = os.path.join(ILD2_DIAGNOSTIC_DIR, "compare_npl_vs_npl_alpha_ild2_slope_vs_accuracy.png")
SLOPE_ACCURACY_OUTPUT_PDF = os.path.join(ILD2_DIAGNOSTIC_DIR, "compare_npl_vs_npl_alpha_ild2_slope_vs_accuracy.pdf")
SIX_ANIMAL_PSY_OUTPUT_PNG = os.path.join(ILD2_DIAGNOSTIC_DIR, "six_animals_above_below_accuracy_psychometric.png")
SIX_ANIMAL_PSY_OUTPUT_PDF = os.path.join(ILD2_DIAGNOSTIC_DIR, "six_animals_above_below_accuracy_psychometric.pdf")
SIX_ANIMAL_PSY_ABL_AVG_OUTPUT_PNG = os.path.join(
    ILD2_DIAGNOSTIC_DIR,
    "six_animals_above_below_accuracy_psychometric_abl_average.png",
)
SIX_ANIMAL_PSY_ABL_AVG_OUTPUT_PDF = os.path.join(
    ILD2_DIAGNOSTIC_DIR,
    "six_animals_above_below_accuracy_psychometric_abl_average.pdf",
)
SIX_ANIMAL_PCORRECT_ABL_AVG_OUTPUT_PNG = os.path.join(
    ILD2_DIAGNOSTIC_DIR,
    "six_animals_above_below_accuracy_pcorrect_abl_average.png",
)
SIX_ANIMAL_PCORRECT_ABL_AVG_OUTPUT_PDF = os.path.join(
    ILD2_DIAGNOSTIC_DIR,
    "six_animals_above_below_accuracy_pcorrect_abl_average.pdf",
)
ILD2_LABEL_FONTSIZE = 6
HIGHLIGHT_ILD2_ANIMAL = ("LED7", 93)
ABL_ARR = [20, 40, 60]
ILD_ARR = [-16.0, -8.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0, 16.0]
ABL_COLORS = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}
MAIN_LABEL_FONTSIZE = 10
MAIN_TITLE_FONTSIZE = 12
MAIN_TICK_FONTSIZE = 9
MAIN_SUPTITLE_FONTSIZE = 13
MAIN_ANNOTATION_FONTSIZE = 8
N_ABOVE_DIAGONAL_ACCURACY_ANIMALS_TO_LABEL = 7


# %%
def _create_innermost_dict():
    return {"empirical": [], "theoretical": []}


def _create_inner_defaultdict():
    return defaultdict(_create_innermost_dict)


def load_data():
    with open(NPL_QUANT_PKL, "rb") as handle:
        npl_quant_data = pickle.load(handle)
    with open(NPL_SLOPES_PKL, "rb") as handle:
        npl_slopes_data = pickle.load(handle)
    with open(VANILLA_SLOPES_PKL, "rb") as handle:
        vanilla_slopes_data = pickle.load(handle)
    with open(ILD2_QUANT_PKL, "rb") as handle:
        ild2_quant_data = pickle.load(handle)
    with open(ILD2_SLOPES_PKL, "rb") as handle:
        ild2_slopes_data = pickle.load(handle)
    with open(VANILLA_THEORETICAL_PSY_PKL, "rb") as handle:
        vanilla_theoretical_psy_data = pickle.load(handle)
    with open(NPL_THEORETICAL_PSY_PKL, "rb") as handle:
        npl_theoretical_psy_data = pickle.load(handle)
    with open(ILD2_THEORETICAL_PSY_PKL, "rb") as handle:
        ild2_theoretical_psy_data = pickle.load(handle)
    with open(EMPIRICAL_PSY_PKL, "rb") as handle:
        empirical_psy_data = pickle.load(handle)

    return (
        npl_quant_data,
        npl_slopes_data,
        vanilla_slopes_data,
        ild2_quant_data,
        ild2_slopes_data,
        vanilla_theoretical_psy_data,
        npl_theoretical_psy_data,
        ild2_theoretical_psy_data,
        empirical_psy_data,
    )


# %%
def aggregate_empirical_quantiles(data, q_idx):
    plot_data = data["plot_data"]
    abs_ild_sorted = data["abs_ild_sorted"]
    ABL_arr = data["ABL_arr"]

    emp_means, emp_sems = [], []
    for abs_ild in abs_ild_sorted:
        all_abl_emp_quantiles = []
        for ABL in ABL_arr:
            emp_values = np.asarray(plot_data[ABL][abs_ild]["empirical"], dtype=float)
            if emp_values.size > 0:
                all_abl_emp_quantiles.extend(emp_values[:, q_idx])

        all_abl_emp_quantiles = np.asarray(all_abl_emp_quantiles, dtype=float)
        emp_means.append(np.nanmean(all_abl_emp_quantiles))
        emp_sems.append(sem(all_abl_emp_quantiles, nan_policy="omit"))

    return np.asarray(abs_ild_sorted, dtype=float), np.asarray(emp_means), np.asarray(emp_sems)


def aggregate_model_quantiles(data, q_idx):
    plot_data = data["plot_data"]
    continuous_plot_data = data.get("continuous_plot_data", None)
    continuous_abs_ild = data.get("continuous_abs_ild", None)
    abs_ild_sorted = data["abs_ild_sorted"]
    ABL_arr = data["ABL_arr"]

    theo_abs_ild_plot, theo_means, theo_sems = [], [], []

    if continuous_plot_data is not None and continuous_abs_ild is not None:
        for abs_ild in continuous_abs_ild:
            all_abl_theo_quantiles = []
            for ABL in ABL_arr:
                if len(continuous_plot_data[ABL][abs_ild]["theoretical"]) > 0:
                    theo_values = np.asarray(continuous_plot_data[ABL][abs_ild]["theoretical"], dtype=float)
                    all_abl_theo_quantiles.extend(theo_values[:, q_idx])

            if len(all_abl_theo_quantiles) > 0:
                all_abl_theo_quantiles = np.asarray(all_abl_theo_quantiles, dtype=float)
                theo_abs_ild_plot.append(abs_ild)
                theo_means.append(np.nanmean(all_abl_theo_quantiles))
                theo_sems.append(sem(all_abl_theo_quantiles, nan_policy="omit"))
    else:
        for abs_ild in abs_ild_sorted:
            all_abl_theo_quantiles = []
            for ABL in ABL_arr:
                theo_values = np.asarray(plot_data[ABL][abs_ild]["theoretical"], dtype=float)
                if theo_values.size > 0:
                    all_abl_theo_quantiles.extend(theo_values[:, q_idx])

            all_abl_theo_quantiles = np.asarray(all_abl_theo_quantiles, dtype=float)
            theo_abs_ild_plot.append(abs_ild)
            theo_means.append(np.nanmean(all_abl_theo_quantiles))
            theo_sems.append(sem(all_abl_theo_quantiles, nan_policy="omit"))

    return np.asarray(theo_abs_ild_plot, dtype=float), np.asarray(theo_means), np.asarray(theo_sems)


def aggregate_model_quantiles_at_observed_abs_ild(data, q_idx):
    plot_data = data["plot_data"]
    abs_ild_sorted = data["abs_ild_sorted"]
    continuous_plot_data = data.get("continuous_plot_data", None)
    continuous_abs_ild = data.get("continuous_abs_ild", None)
    ABL_arr = data["ABL_arr"]

    theo_means = []
    for abs_ild in abs_ild_sorted:
        all_abl_theo_quantiles = []
        if continuous_plot_data is not None and continuous_abs_ild is not None:
            nearest_idx = int(np.nanargmin(np.abs(np.asarray(continuous_abs_ild, dtype=float) - float(abs_ild))))
            continuous_abs_ild_key = continuous_abs_ild[nearest_idx]
            for ABL in ABL_arr:
                theo_values = np.asarray(
                    continuous_plot_data[ABL][continuous_abs_ild_key]["theoretical"],
                    dtype=float,
                )
                if theo_values.size > 0:
                    all_abl_theo_quantiles.extend(theo_values[:, q_idx])
        else:
            for ABL in ABL_arr:
                theo_values = np.asarray(plot_data[ABL][abs_ild]["theoretical"], dtype=float)
                if theo_values.size > 0:
                    all_abl_theo_quantiles.extend(theo_values[:, q_idx])

        all_abl_theo_quantiles = np.asarray(all_abl_theo_quantiles, dtype=float)
        theo_means.append(np.nanmean(all_abl_theo_quantiles))

    return np.asarray(theo_means, dtype=float)


def print_quantile_fit_metrics(npl_quant_data, ild2_quant_data):
    quantiles_to_plot = npl_quant_data["QUANTILES_TO_PLOT"]

    empirical_values, npl_values, ild2_values = [], [], []
    for q_idx, _q in enumerate(quantiles_to_plot):
        _emp_x, emp_mean, _emp_sem = aggregate_empirical_quantiles(npl_quant_data, q_idx)
        empirical_values.extend(emp_mean)
        npl_values.extend(aggregate_model_quantiles_at_observed_abs_ild(npl_quant_data, q_idx))
        ild2_values.extend(aggregate_model_quantiles_at_observed_abs_ild(ild2_quant_data, q_idx))

    empirical_values = np.asarray(empirical_values, dtype=float)
    npl_values = np.asarray(npl_values, dtype=float)
    ild2_values = np.asarray(ild2_values, dtype=float)

    finite_npl = np.isfinite(empirical_values) & np.isfinite(npl_values)
    finite_ild2 = np.isfinite(empirical_values) & np.isfinite(ild2_values)

    if np.sum(finite_npl) >= 2:
        print(
            "Quantile fit metrics for NPL: "
            f"R²={r2_score(empirical_values[finite_npl], npl_values[finite_npl]):.3f}, "
            f"MSE={mean_squared_error(empirical_values[finite_npl], npl_values[finite_npl]):.6f}"
        )
    if np.sum(finite_ild2) >= 2:
        print(
            "Quantile fit metrics for NPL + alpha + ILD2 delay: "
            f"R²={r2_score(empirical_values[finite_ild2], ild2_values[finite_ild2]):.3f}, "
            f"MSE={mean_squared_error(empirical_values[finite_ild2], ild2_values[finite_ild2]):.6f}"
        )


# %%
def signed_accuracy_from_right_prob(ild, right_prob):
    if not np.isfinite(right_prob):
        return np.nan
    if ild > 0:
        return right_prob
    if ild < 0:
        return 1 - right_prob
    return np.nan


def get_right_prob_at_ild(condition_data, ILD):
    ild_values = np.asarray(condition_data["ild_values"], dtype=float)
    right_choice_probs = np.asarray(condition_data["right_choice_probs"], dtype=float)
    matches = np.where(np.isclose(ild_values, ILD))[0]
    if len(matches) == 0:
        return np.nan
    return right_choice_probs[matches[0]]


def mean_accuracy_for_animal_on_empirical_conditions(psy_data, animal_key, data_key, empirical_psy_data):
    accuracies = []
    empirical_animal_data = empirical_psy_data.get(animal_key, {})
    model_animal_data = psy_data.get(animal_key, {})
    for ABL in ABL_ARR:
        empirical_condition_data = empirical_animal_data.get(ABL, {}).get("empirical", None)
        model_condition_data = model_animal_data.get(ABL, {}).get(data_key, None)
        if empirical_condition_data is None or model_condition_data is None:
            continue

        empirical_ild_values = np.asarray(empirical_condition_data["ild_values"], dtype=float)
        for ILD in empirical_ild_values:
            right_prob = get_right_prob_at_ild(model_condition_data, ILD)
            accuracies.append(signed_accuracy_from_right_prob(ILD, right_prob))

    accuracies = np.asarray(accuracies, dtype=float)
    if accuracies.size == 0 or np.all(~np.isfinite(accuracies)):
        return np.nan
    return float(np.nanmean(accuracies))


def get_accuracy_comparison_data(
    vanilla_theoretical_psy_data,
    npl_theoretical_psy_data,
    ild2_theoretical_psy_data,
    empirical_psy_data,
    ild2_slopes_data,
):
    animal_keys = ild2_slopes_data.get("common_pairs_sorted", sorted(set(empirical_psy_data)))

    data_accuracy = []
    vanilla_accuracy = []
    npl_accuracy = []
    ild2_accuracy = []
    for animal_key in animal_keys:
        data_accuracy.append(
            mean_accuracy_for_animal_on_empirical_conditions(
                empirical_psy_data,
                animal_key,
                "empirical",
                empirical_psy_data,
            )
        )
        vanilla_accuracy.append(
            mean_accuracy_for_animal_on_empirical_conditions(
                vanilla_theoretical_psy_data,
                animal_key,
                "theoretical",
                empirical_psy_data,
            )
        )
        npl_accuracy.append(
            mean_accuracy_for_animal_on_empirical_conditions(
                npl_theoretical_psy_data,
                animal_key,
                "theoretical",
                empirical_psy_data,
            )
        )
        ild2_accuracy.append(
            mean_accuracy_for_animal_on_empirical_conditions(
                ild2_theoretical_psy_data,
                animal_key,
                "theoretical",
                empirical_psy_data,
            )
        )

    return (
        animal_keys,
        np.asarray(data_accuracy, dtype=float),
        np.asarray(vanilla_accuracy, dtype=float),
        np.asarray(npl_accuracy, dtype=float),
        np.asarray(ild2_accuracy, dtype=float),
    )


def get_six_animal_psychometric_keys(animal_keys, data_accuracy, ild2_accuracy):
    ild2_accuracy_difference = ild2_accuracy - data_accuracy
    finite_diff = np.isfinite(ild2_accuracy_difference)
    above_indices = [
        idx
        for idx, animal_key in enumerate(animal_keys)
        if finite_diff[idx] and ild2_accuracy_difference[idx] > 0
    ]
    top_indices = sorted(above_indices, key=lambda idx: ild2_accuracy_difference[idx], reverse=True)[:3]

    below_indices = [
        idx
        for idx in range(len(animal_keys))
        if finite_diff[idx] and ild2_accuracy_difference[idx] < 0
    ]
    bottom_indices = sorted(below_indices, key=lambda idx: ild2_accuracy_difference[idx])[:3]

    print("Six-animal psychometric diagnostic selection:")
    print("Top animals above diagonal:")
    for idx in top_indices:
        batch, animal_id = animal_keys[idx]
        print(f"  {batch}-{animal_id}: ILD2-data={ild2_accuracy_difference[idx]:+.4f}")
    print("Lowest animals below diagonal:")
    for idx in bottom_indices:
        batch, animal_id = animal_keys[idx]
        print(f"  {batch}-{animal_id}: ILD2-data={ild2_accuracy_difference[idx]:+.4f}")

    return top_indices, bottom_indices, ild2_accuracy_difference


def print_accuracy_fit_metrics(data_accuracy, vanilla_accuracy, npl_accuracy, ild2_accuracy):
    finite_vanilla = np.isfinite(data_accuracy) & np.isfinite(vanilla_accuracy)
    finite_npl = np.isfinite(data_accuracy) & np.isfinite(npl_accuracy)
    finite_ild2 = np.isfinite(data_accuracy) & np.isfinite(ild2_accuracy)

    if np.sum(finite_vanilla) >= 2:
        print(
            "Accuracy fit metrics for vanilla TIED: "
            f"R²={r2_score(data_accuracy[finite_vanilla], vanilla_accuracy[finite_vanilla]):.3f}, "
            f"MSE={mean_squared_error(data_accuracy[finite_vanilla], vanilla_accuracy[finite_vanilla]):.6f}"
        )
    if np.sum(finite_npl) >= 2:
        print(
            "Accuracy fit metrics for NPL: "
            f"R²={r2_score(data_accuracy[finite_npl], npl_accuracy[finite_npl]):.3f}, "
            f"MSE={mean_squared_error(data_accuracy[finite_npl], npl_accuracy[finite_npl]):.6f}"
        )
    if np.sum(finite_ild2) >= 2:
        print(
            "Accuracy fit metrics for NPL + alpha + ILD2 delay: "
            f"R²={r2_score(data_accuracy[finite_ild2], ild2_accuracy[finite_ild2]):.3f}, "
            f"MSE={mean_squared_error(data_accuracy[finite_ild2], ild2_accuracy[finite_ild2]):.6f}"
        )


# %%
def plot_quantile_comparison(ax, npl_quant_data, ild2_quant_data):
    quantiles_to_plot = npl_quant_data["QUANTILES_TO_PLOT"]
    abs_ild_sorted = npl_quant_data["abs_ild_sorted"]

    if list(quantiles_to_plot) != list(ild2_quant_data["QUANTILES_TO_PLOT"]):
        print("Warning: NPL and ILD2 quantile levels differ; plotting NPL quantile levels.")

    for q_idx, q in enumerate(quantiles_to_plot):
        emp_x, emp_mean, emp_sem = aggregate_empirical_quantiles(npl_quant_data, q_idx)
        ax.errorbar(
            emp_x,
            emp_mean,
            yerr=emp_sem,
            fmt="o",
            color="black",
            markersize=7,
            capsize=0,
            zorder=5,
        )

        npl_x, npl_mean, npl_sem = aggregate_model_quantiles(npl_quant_data, q_idx)
        ax.plot(
            npl_x,
            npl_mean,
            "-",
            color="tab:blue",
            linewidth=1.8,
            alpha=0.95,
        )
        ax.fill_between(
            npl_x,
            npl_mean - npl_sem,
            npl_mean + npl_sem,
            color="tab:blue",
            alpha=0.08,
            linewidth=0,
        )

        ild2_x, ild2_mean, ild2_sem = aggregate_model_quantiles(ild2_quant_data, q_idx)
        ax.plot(
            ild2_x,
            ild2_mean,
            "--",
            color="tab:red",
            linewidth=1.8,
            alpha=0.95,
        )
        ax.fill_between(
            ild2_x,
            ild2_mean - ild2_sem,
            ild2_mean + ild2_sem,
            color="tab:red",
            alpha=0.08,
            linewidth=0,
        )

    ax.set_xlabel("|ILD| (dB)", fontsize=MAIN_LABEL_FONTSIZE)
    ax.set_ylabel("RT Quantile (s)", fontsize=MAIN_LABEL_FONTSIZE)
    ax.set_xscale("log", base=2)
    ax.set_xticks(abs_ild_sorted)
    ax.set_yticks([0.1, 0.2, 0.3, 0.4])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.tick_params(axis="both", which="major", labelsize=MAIN_TICK_FONTSIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_box_aspect(1)


def plot_slope_comparison(ax, npl_slopes_data, ild2_slopes_data):
    npl_data_means = np.asarray(npl_slopes_data["data_means"], dtype=float)
    npl_model_means = np.asarray(npl_slopes_data["norm_means"], dtype=float)

    ild2_data_means = np.asarray(ild2_slopes_data["data_means"], dtype=float)
    ild2_model_means = np.asarray(
        ild2_slopes_data.get("npl_alpha_ild2_means", ild2_slopes_data["norm_means"]),
        dtype=float,
    )
    ild2_labels = [
        f"{batch}-{animal_id}"
        for batch, animal_id in ild2_slopes_data.get("common_pairs_sorted", [])
    ]

    ax.scatter(
        npl_data_means,
        npl_model_means,
        marker="o",
        s=70,
        color="tab:blue",
        alpha=0.5,
        edgecolors="none",
    )
    ax.scatter(
        ild2_data_means,
        ild2_model_means,
        marker="o",
        s=70,
        color="tab:red",
        alpha=0.5,
        edgecolors="none",
    )

    for idx, (x_val, y_val, label) in enumerate(zip(ild2_data_means, ild2_model_means, ild2_labels)):
        if not (np.isfinite(x_val) and np.isfinite(y_val)):
            continue
        if label != f"{HIGHLIGHT_ILD2_ANIMAL[0]}-{HIGHLIGHT_ILD2_ANIMAL[1]}":
            continue
        ax.scatter(
            [x_val],
            [y_val],
            marker="o",
            s=165,
            facecolors="none",
            edgecolors="black",
            linewidths=1.6,
            zorder=7,
        )
        ax.annotate(
            label,
            xy=(x_val, y_val),
            xytext=(24, -22),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.0},
            fontsize=MAIN_ANNOTATION_FONTSIZE,
            color="black",
            zorder=8,
        )

    ax.plot([0.1, 0.9], [0.1, 0.9], color="grey", alpha=0.5, linestyle="--", linewidth=2, zorder=0)
    ax.set_xlabel("Data", fontsize=MAIN_LABEL_FONTSIZE)
    ax.set_ylabel("Model", fontsize=MAIN_LABEL_FONTSIZE)
    ax.set_xticks([0.1, 0.5, 0.9])
    ax.set_yticks([0.1, 0.5, 0.9])
    ax.set_xlim(0.1, 0.9)
    ax.set_ylim(0.1, 0.9)
    ax.tick_params(axis="both", labelsize=MAIN_TICK_FONTSIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_box_aspect(1)

    finite_npl = np.isfinite(npl_data_means) & np.isfinite(npl_model_means)
    finite_ild2 = np.isfinite(ild2_data_means) & np.isfinite(ild2_model_means)
    if np.sum(finite_npl) >= 2:
        print(f"R² for NPL slopes: {r2_score(npl_data_means[finite_npl], npl_model_means[finite_npl]):.2f}")
    if np.sum(finite_ild2) >= 2:
        print(
            "R² for NPL + alpha + ILD2 delay slopes: "
            f"{r2_score(ild2_data_means[finite_ild2], ild2_model_means[finite_ild2]):.2f}"
        )


def plot_vanilla_slope_comparison(ax, vanilla_slopes_data):
    data_means = np.asarray(vanilla_slopes_data["data_means"], dtype=float)
    vanilla_means = np.asarray(vanilla_slopes_data["vanilla_means"], dtype=float)

    ax.scatter(
        data_means,
        vanilla_means,
        marker="o",
        s=70,
        color="tab:green",
        alpha=0.5,
        edgecolors="none",
    )

    ax.plot([0.1, 0.9], [0.1, 0.9], color="grey", alpha=0.5, linestyle="--", linewidth=2, zorder=0)
    ax.set_xlabel("Data", fontsize=MAIN_LABEL_FONTSIZE)
    ax.set_ylabel("Vanilla TIED", fontsize=MAIN_LABEL_FONTSIZE)
    ax.set_xticks([0.1, 0.5, 0.9])
    ax.set_yticks([0.1, 0.5, 0.9])
    ax.set_xlim(0.1, 0.9)
    ax.set_ylim(0.1, 0.9)
    ax.tick_params(axis="both", labelsize=MAIN_TICK_FONTSIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_box_aspect(1)

    finite = np.isfinite(data_means) & np.isfinite(vanilla_means)
    if np.sum(finite) >= 2:
        print(f"R² for vanilla TIED slopes: {r2_score(data_means[finite], vanilla_means[finite]):.2f}")

# %%
# acc vs acc 
def plot_accuracy_comparison(ax, animal_keys, data_accuracy, npl_accuracy, ild2_accuracy):
    ax.scatter(
        data_accuracy,
        npl_accuracy,
        marker="o",
        s=70,
        color="tab:blue",
        alpha=0.5,
        edgecolors="none",
    )
    ax.scatter(
        data_accuracy,
        ild2_accuracy,
        marker="o",
        s=70,
        color="tab:red",
        alpha=0.5,
        edgecolors="none",
    )

    accuracy_residuals = np.maximum(npl_accuracy - data_accuracy, ild2_accuracy - data_accuracy)
    finite_residuals = np.isfinite(data_accuracy) & np.isfinite(npl_accuracy) & np.isfinite(ild2_accuracy)
    above_diagonal_order = np.argsort(accuracy_residuals)[::-1]
    highlighted_accuracy_indices = [
        idx
        for idx in above_diagonal_order
        if finite_residuals[idx] and accuracy_residuals[idx] > 0
    ][:N_ABOVE_DIAGONAL_ACCURACY_ANIMALS_TO_LABEL]

    print("Animals most above the accuracy diagonal:")
    for rank, idx in enumerate(highlighted_accuracy_indices, start=1):
        batch, animal_id = animal_keys[idx]
        print(
            f"{rank}. {batch}-{animal_id}: "
            f"data={data_accuracy[idx]:.4f}, "
            f"NPL={npl_accuracy[idx]:.4f} ({npl_accuracy[idx] - data_accuracy[idx]:+.4f}), "
            f"ILD2={ild2_accuracy[idx]:.4f} ({ild2_accuracy[idx] - data_accuracy[idx]:+.4f})"
        )

    rank_label_offsets = [
        (-12, -2),
        (14, 0),
        (-4, 16),
        (-20, 14),
        (-12, 14),
        (12, 12),
        (14, -12),
    ]
    for label_rank, idx in enumerate(highlighted_accuracy_indices):
        label_y = max(npl_accuracy[idx], ild2_accuracy[idx])
        ax.scatter(
            [data_accuracy[idx]],
            [npl_accuracy[idx]],
            marker="o",
            s=140,
            facecolors="none",
            edgecolors="black",
            linewidths=1.2,
            zorder=7,
        )
        ax.scatter(
            [data_accuracy[idx]],
            [ild2_accuracy[idx]],
            marker="o",
            s=140,
            facecolors="none",
            edgecolors="black",
            linewidths=1.2,
            zorder=7,
        )
        ax.annotate(
            str(label_rank + 1),
            xy=(data_accuracy[idx], label_y),
            xytext=rank_label_offsets[label_rank % len(rank_label_offsets)],
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "color": "black", "lw": 0.8},
            fontsize=MAIN_ANNOTATION_FONTSIZE,
            fontweight="bold",
            color="black",
            zorder=8,
        )

    lim_left = 0.7
    lim_right = 0.9
    lim_middle = (lim_left + lim_right) / 2
    ax.plot([lim_left, lim_right], [lim_left, lim_right], color="grey", alpha=0.5, linestyle="--", linewidth=2, zorder=0)
    ax.set_xlabel("Data Accuracy", fontsize=MAIN_LABEL_FONTSIZE)
    ax.set_ylabel("Model Accuracy", fontsize=MAIN_LABEL_FONTSIZE)
    ax.set_xticks([lim_left, lim_middle, lim_right])
    ax.set_yticks([lim_left, lim_middle, lim_right])
    ax.set_xlim(lim_left, lim_right)
    ax.set_ylim(lim_left, lim_right)
    ax.tick_params(axis="both", labelsize=MAIN_TICK_FONTSIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_box_aspect(1)


def plot_vanilla_accuracy_comparison(ax, data_accuracy, vanilla_accuracy):
    ax.scatter(
        data_accuracy,
        vanilla_accuracy,
        marker="o",
        s=70,
        color="tab:green",
        alpha=0.5,
        edgecolors="none",
    )

    lim_left = 0.7
    lim_right = 0.9
    lim_middle = (lim_left + lim_right) / 2
    ax.plot([lim_left, lim_right], [lim_left, lim_right], color="grey", alpha=0.5, linestyle="--", linewidth=2, zorder=0)
    ax.set_xlabel("Data Accuracy", fontsize=MAIN_LABEL_FONTSIZE)
    ax.set_ylabel("Vanilla TIED Accuracy", fontsize=MAIN_LABEL_FONTSIZE)
    ax.set_xticks([lim_left, lim_middle, lim_right])
    ax.set_yticks([lim_left, lim_middle, lim_right])
    ax.set_xlim(lim_left, lim_right)
    ax.set_ylim(lim_left, lim_right)
    ax.tick_params(axis="both", labelsize=MAIN_TICK_FONTSIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_box_aspect(1)


def plot_slope_vs_accuracy_panel(ax, slopes, accuracies, color, title, axis_limits, axis_ticks):
    local_label_fontsize = 12
    local_title_fontsize = 13
    local_tick_fontsize = 11

    ax.scatter(
        slopes,
        accuracies,
        marker="o",
        s=70,
        color=color,
        alpha=0.65,
        edgecolors="none",
    )
    ax.set_xlabel("Psychometric Slope", fontsize=local_label_fontsize)
    ax.set_ylabel("Accuracy", fontsize=local_label_fontsize)
    ax.set_title(title, fontsize=local_title_fontsize)
    ax.set_xlim(axis_limits)
    ax.set_ylim(axis_limits)
    ax.set_xticks(axis_ticks)
    ax.set_yticks(axis_ticks)
    ax.tick_params(axis="both", labelsize=local_tick_fontsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_box_aspect(1)


def plot_slope_vs_accuracy_figure(npl_slopes_data, ild2_slopes_data, data_accuracy, npl_accuracy, ild2_accuracy, axis_limits):
    data_slopes = np.asarray(ild2_slopes_data["data_means"], dtype=float)
    npl_slopes = np.asarray(npl_slopes_data["norm_means"], dtype=float)
    ild2_slopes = np.asarray(
        ild2_slopes_data.get("npl_alpha_ild2_means", ild2_slopes_data["norm_means"]),
        dtype=float,
    )

    npl_data_slopes = np.asarray(npl_slopes_data["data_means"], dtype=float)
    if not np.allclose(npl_data_slopes, data_slopes, equal_nan=True):
        print("Warning: NPL and ILD2 slope data order differs; slope-vs-accuracy plot assumes matching order.")

    axis_ticks = [axis_limits[0], (axis_limits[0] + axis_limits[1]) / 2, axis_limits[1]]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plot_slope_vs_accuracy_panel(axes[0], data_slopes, data_accuracy, "black", "Data", axis_limits, axis_ticks)
    plot_slope_vs_accuracy_panel(axes[1], npl_slopes, npl_accuracy, "tab:blue", "NPL", axis_limits, axis_ticks)
    plot_slope_vs_accuracy_panel(
        axes[2],
        ild2_slopes,
        ild2_accuracy,
        "tab:red",
        "NPL + alpha\n+ ILD2 delay",
        axis_limits,
        axis_ticks,
    )
    fig.suptitle(
        "Slope vs Accuracy Across Animals",
        fontsize=14,
        y=0.98,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.78, wspace=0.35)
    fig.savefig(SLOPE_ACCURACY_OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(SLOPE_ACCURACY_OUTPUT_PDF, dpi=300, bbox_inches="tight")
    print(f"Saved {SLOPE_ACCURACY_OUTPUT_PNG}")
    print(f"Saved {SLOPE_ACCURACY_OUTPUT_PDF}")


def plot_one_animal_psychometric_panel(
    ax,
    animal_key,
    accuracy_difference,
    empirical_psy_data,
    ild2_theoretical_psy_data,
):
    batch, animal_id = animal_key
    for ABL in ABL_ARR:
        empirical_condition = empirical_psy_data[animal_key][ABL]["empirical"]
        ild2_condition = ild2_theoretical_psy_data[animal_key][ABL]["theoretical"]

        emp_ild = np.asarray(empirical_condition["ild_values"], dtype=float)
        emp_prob = np.asarray(empirical_condition["right_choice_probs"], dtype=float)
        ild2_prob = np.asarray([get_right_prob_at_ild(ild2_condition, ILD) for ILD in emp_ild], dtype=float)

        emp_order = np.argsort(emp_ild)

        ax.scatter(
            emp_ild[emp_order],
            emp_prob[emp_order],
            marker="o",
            s=30,
            color=ABL_COLORS[ABL],
            edgecolors="none",
            alpha=0.75,
            zorder=5,
        )
        ax.plot(
            emp_ild[emp_order],
            ild2_prob[emp_order],
            color=ABL_COLORS[ABL],
            linestyle="-",
            linewidth=2.0,
            alpha=0.95,
        )

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.35, zorder=0)
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.35, zorder=0)
    ax.set_title(
        f"{batch}-{animal_id}\nILD2-data={accuracy_difference:+.4f}",
        fontsize=11,
    )
    ax.set_xlim(-17, 17)
    ax.set_ylim(0, 1)
    ax.set_xticks([-16, -8, 0, 8, 16])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(axis="both", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_box_aspect(1)


def plot_six_animal_psychometric_diagnostic(
    animal_keys,
    data_accuracy,
    ild2_accuracy,
    empirical_psy_data,
    ild2_theoretical_psy_data,
):
    top_indices, bottom_indices, ild2_accuracy_difference = get_six_animal_psychometric_keys(
        animal_keys,
        data_accuracy,
        ild2_accuracy,
    )
    selected_indices = top_indices + bottom_indices

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7))
    for ax, animal_idx in zip(axes.ravel(), selected_indices):
        plot_one_animal_psychometric_panel(
            ax,
            animal_keys[animal_idx],
            ild2_accuracy_difference[animal_idx],
            empirical_psy_data,
            ild2_theoretical_psy_data,
        )

    for ax in axes[1]:
        ax.set_xlabel("ILD (dB)", fontsize=11)
    for ax in axes[:, 0]:
        ax.set_ylabel("P(right)", fontsize=11)

    fig.suptitle(
        "Psychometric Diagnostics: Dots are Data, Lines are NPL + alpha + ILD2\n"
        "ABL 20/40/60 shown as blue/orange/green",
        fontsize=12,
        y=0.98,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.85, wspace=0.28, hspace=0.42)
    fig.savefig(SIX_ANIMAL_PSY_OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(SIX_ANIMAL_PSY_OUTPUT_PDF, dpi=300, bbox_inches="tight")
    print(f"Saved {SIX_ANIMAL_PSY_OUTPUT_PNG}")
    print(f"Saved {SIX_ANIMAL_PSY_OUTPUT_PDF}")


def empirical_ild_values_for_animal(empirical_psy_data, animal_key):
    ild_values = []
    for ABL in ABL_ARR:
        empirical_condition = empirical_psy_data.get(animal_key, {}).get(ABL, {}).get("empirical", None)
        if empirical_condition is None:
            continue
        ild_values.extend(np.asarray(empirical_condition["ild_values"], dtype=float))
    return np.asarray(sorted(set(ild_values)), dtype=float)


def average_psychometric_across_abl(psy_data, animal_key, data_key, empirical_psy_data):
    ild_values = empirical_ild_values_for_animal(empirical_psy_data, animal_key)
    probs_by_abl = []
    for ABL in ABL_ARR:
        empirical_condition = empirical_psy_data.get(animal_key, {}).get(ABL, {}).get("empirical", None)
        if empirical_condition is None:
            continue

        empirical_condition_ild_values = np.asarray(empirical_condition["ild_values"], dtype=float)
        condition_data = psy_data[animal_key][ABL][data_key]

        condition_probs = []
        for ILD in ild_values:
            empirical_matches = np.where(np.isclose(empirical_condition_ild_values, ILD))[0]
            if len(empirical_matches) == 0:
                condition_probs.append(np.nan)
            else:
                condition_probs.append(get_right_prob_at_ild(condition_data, ILD))
        probs_by_abl.append(condition_probs)

    probs_by_abl = np.asarray(probs_by_abl, dtype=float)
    mean_probs = np.nanmean(probs_by_abl, axis=0)
    sem_probs = sem(probs_by_abl, axis=0, nan_policy="omit")
    return ild_values, mean_probs, sem_probs


def plot_one_animal_abl_average_psychometric_panel(
    ax,
    animal_key,
    accuracy_difference,
    empirical_psy_data,
    ild2_theoretical_psy_data,
):
    batch, animal_id = animal_key
    emp_ild, emp_mean, emp_sem = average_psychometric_across_abl(
        empirical_psy_data,
        animal_key,
        "empirical",
        empirical_psy_data,
    )
    ild2_ild, ild2_mean, ild2_sem = average_psychometric_across_abl(
        ild2_theoretical_psy_data,
        animal_key,
        "theoretical",
        empirical_psy_data,
    )

    ax.errorbar(
        emp_ild,
        emp_mean,
        yerr=emp_sem,
        fmt="o",
        color="black",
        markersize=5,
        elinewidth=1.1,
        capsize=0,
        alpha=0.9,
        zorder=5,
    )
    ax.errorbar(
        ild2_ild,
        ild2_mean,
        yerr=ild2_sem,
        fmt="-o",
        color="tab:red",
        markersize=3,
        linewidth=2.0,
        elinewidth=1.0,
        capsize=0,
        alpha=0.9,
        zorder=4,
    )

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.35, zorder=0)
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.35, zorder=0)
    ax.set_title(
        f"{batch}-{animal_id}\nILD2-data={accuracy_difference:+.4f}",
        fontsize=11,
    )
    ax.set_xlim(-17, 17)
    ax.set_ylim(0, 1)
    ax.set_xticks([-16, -8, 0, 8, 16])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(axis="both", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_box_aspect(1)


def plot_six_animal_abl_average_psychometric_diagnostic(
    animal_keys,
    data_accuracy,
    ild2_accuracy,
    empirical_psy_data,
    ild2_theoretical_psy_data,
):
    top_indices, bottom_indices, ild2_accuracy_difference = get_six_animal_psychometric_keys(
        animal_keys,
        data_accuracy,
        ild2_accuracy,
    )
    selected_indices = top_indices + bottom_indices

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7))
    for ax, animal_idx in zip(axes.ravel(), selected_indices):
        plot_one_animal_abl_average_psychometric_panel(
            ax,
            animal_keys[animal_idx],
            ild2_accuracy_difference[animal_idx],
            empirical_psy_data,
            ild2_theoretical_psy_data,
        )

    for ax in axes[1]:
        ax.set_xlabel("ILD (dB)", fontsize=11)
    for ax in axes[:, 0]:
        ax.set_ylabel("P(right)", fontsize=11)

    fig.suptitle(
        "Psychometric Diagnostics Averaged Across ABL\n"
        "Black dots: data mean +/- SEM; red line: NPL + alpha + ILD2 mean +/- SEM",
        fontsize=12,
        y=0.98,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.85, wspace=0.28, hspace=0.42)
    fig.savefig(SIX_ANIMAL_PSY_ABL_AVG_OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(SIX_ANIMAL_PSY_ABL_AVG_OUTPUT_PDF, dpi=300, bbox_inches="tight")
    print(f"Saved {SIX_ANIMAL_PSY_ABL_AVG_OUTPUT_PNG}")
    print(f"Saved {SIX_ANIMAL_PSY_ABL_AVG_OUTPUT_PDF}")


def average_pcorrect_across_abl(psy_data, animal_key, data_key, empirical_psy_data):
    ild_values = empirical_ild_values_for_animal(empirical_psy_data, animal_key)
    pcorrect_by_abl = []
    for ABL in ABL_ARR:
        empirical_condition = empirical_psy_data.get(animal_key, {}).get(ABL, {}).get("empirical", None)
        if empirical_condition is None:
            continue

        empirical_condition_ild_values = np.asarray(empirical_condition["ild_values"], dtype=float)
        condition_data = psy_data[animal_key][ABL][data_key]

        condition_pcorrect = []
        for ILD in ild_values:
            empirical_matches = np.where(np.isclose(empirical_condition_ild_values, ILD))[0]
            if len(empirical_matches) == 0:
                condition_pcorrect.append(np.nan)
            else:
                condition_pcorrect.append(signed_accuracy_from_right_prob(ILD, get_right_prob_at_ild(condition_data, ILD)))
        pcorrect_by_abl.append(condition_pcorrect)

    pcorrect_by_abl = np.asarray(pcorrect_by_abl, dtype=float)
    mean_pcorrect = np.nanmean(pcorrect_by_abl, axis=0)
    sem_pcorrect = sem(pcorrect_by_abl, axis=0, nan_policy="omit")
    return ild_values, mean_pcorrect, sem_pcorrect


def plot_one_animal_abl_average_pcorrect_panel(
    ax,
    animal_key,
    accuracy_difference,
    empirical_psy_data,
    ild2_theoretical_psy_data,
):
    batch, animal_id = animal_key
    emp_ild, emp_mean, emp_sem = average_pcorrect_across_abl(
        empirical_psy_data,
        animal_key,
        "empirical",
        empirical_psy_data,
    )
    ild2_ild, ild2_mean, ild2_sem = average_pcorrect_across_abl(
        ild2_theoretical_psy_data,
        animal_key,
        "theoretical",
        empirical_psy_data,
    )

    ax.errorbar(
        emp_ild,
        emp_mean,
        yerr=emp_sem,
        fmt="o",
        color="black",
        markersize=5,
        elinewidth=1.1,
        capsize=0,
        alpha=0.9,
        zorder=5,
    )
    ax.errorbar(
        ild2_ild,
        ild2_mean,
        yerr=ild2_sem,
        fmt="-o",
        color="tab:red",
        markersize=3,
        linewidth=2.0,
        elinewidth=1.0,
        capsize=0,
        alpha=0.9,
        zorder=4,
    )

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.35, zorder=0)
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.35, zorder=0)
    ax.set_title(
        f"{batch}-{animal_id}\nILD2-data={accuracy_difference:+.4f}",
        fontsize=11,
    )
    ax.set_xlim(-17, 17)
    ax.set_ylim(0.45, 1.02)
    ax.set_xticks([-16, -8, 0, 8, 16])
    ax.set_yticks([0.5, 0.75, 1])
    ax.tick_params(axis="both", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_box_aspect(1)


def plot_six_animal_abl_average_pcorrect_diagnostic(
    animal_keys,
    data_accuracy,
    ild2_accuracy,
    empirical_psy_data,
    ild2_theoretical_psy_data,
):
    top_indices, bottom_indices, ild2_accuracy_difference = get_six_animal_psychometric_keys(
        animal_keys,
        data_accuracy,
        ild2_accuracy,
    )
    selected_indices = top_indices + bottom_indices

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7))
    for ax, animal_idx in zip(axes.ravel(), selected_indices):
        plot_one_animal_abl_average_pcorrect_panel(
            ax,
            animal_keys[animal_idx],
            ild2_accuracy_difference[animal_idx],
            empirical_psy_data,
            ild2_theoretical_psy_data,
        )

    for ax in axes[1]:
        ax.set_xlabel("ILD (dB)", fontsize=11)
    for ax in axes[:, 0]:
        ax.set_ylabel("P(correct)", fontsize=11)

    fig.suptitle(
        "Signed Accuracy Averaged Across ABL\n"
        "Black dots: data mean +/- SEM; red line: NPL + alpha + ILD2 mean +/- SEM",
        fontsize=12,
        y=0.98,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.85, wspace=0.28, hspace=0.42)
    fig.savefig(SIX_ANIMAL_PCORRECT_ABL_AVG_OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(SIX_ANIMAL_PCORRECT_ABL_AVG_OUTPUT_PDF, dpi=300, bbox_inches="tight")
    print(f"Saved {SIX_ANIMAL_PCORRECT_ABL_AVG_OUTPUT_PNG}")
    print(f"Saved {SIX_ANIMAL_PCORRECT_ABL_AVG_OUTPUT_PDF}")


# %%
npl_quant_data, npl_slopes_data, vanilla_slopes_data, ild2_quant_data, ild2_slopes_data, vanilla_theoretical_psy_data, npl_theoretical_psy_data, ild2_theoretical_psy_data, empirical_psy_data = load_data()
animal_keys, data_accuracy, vanilla_accuracy, npl_accuracy, ild2_accuracy = get_accuracy_comparison_data(
    vanilla_theoretical_psy_data,
    npl_theoretical_psy_data,
    ild2_theoretical_psy_data,
    empirical_psy_data,
    ild2_slopes_data,
)

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
plot_quantile_comparison(axes[0], npl_quant_data, ild2_quant_data)
plot_slope_comparison(axes[1], npl_slopes_data, ild2_slopes_data)
plot_accuracy_comparison(axes[2], animal_keys, data_accuracy, npl_accuracy, ild2_accuracy)
plot_vanilla_slope_comparison(axes[3], vanilla_slopes_data)
plot_vanilla_accuracy_comparison(axes[4], data_accuracy, vanilla_accuracy)
print_quantile_fit_metrics(npl_quant_data, ild2_quant_data)
print_accuracy_fit_metrics(data_accuracy, vanilla_accuracy, npl_accuracy, ild2_accuracy)

axes[0].set_title("RT Quantiles", fontsize=MAIN_TITLE_FONTSIZE)
axes[1].set_title("Psychometric Slope", fontsize=MAIN_TITLE_FONTSIZE)
axes[2].set_title("Accuracy", fontsize=MAIN_TITLE_FONTSIZE)
axes[3].set_title("Vanilla TIED Slope", fontsize=MAIN_TITLE_FONTSIZE)
axes[4].set_title("Vanilla TIED Accuracy", fontsize=MAIN_TITLE_FONTSIZE)
fig.suptitle(
    "Black: data    Blue: NPL    Red: NPL + alpha + ILD2 delay    Green: vanilla TIED",
    fontsize=MAIN_SUPTITLE_FONTSIZE,
    y=0.96,
)

fig.subplots_adjust(left=0.055, right=0.99, bottom=0.2, top=0.84, wspace=0.55)
fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUTPUT_PDF, dpi=300, bbox_inches="tight")

print(f"Saved {OUTPUT_PNG}")
print(f"Saved {OUTPUT_PDF}")
# %%
axis_limits = [0.2, 1]
plot_slope_vs_accuracy_figure(npl_slopes_data, ild2_slopes_data, data_accuracy, npl_accuracy, ild2_accuracy, axis_limits)

# %%
plot_six_animal_psychometric_diagnostic(
    animal_keys,
    data_accuracy,
    ild2_accuracy,
    empirical_psy_data,
    ild2_theoretical_psy_data,
)

# %%
plot_six_animal_abl_average_psychometric_diagnostic(
    animal_keys,
    data_accuracy,
    ild2_accuracy,
    empirical_psy_data,
    ild2_theoretical_psy_data,
)

# %%
plot_six_animal_abl_average_pcorrect_diagnostic(
    animal_keys,
    data_accuracy,
    ild2_accuracy,
    empirical_psy_data,
    ild2_theoretical_psy_data,
)

# %%
