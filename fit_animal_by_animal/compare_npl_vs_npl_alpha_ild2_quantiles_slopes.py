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
ILD2_QUANT_PKL = os.path.join(ILD2_DIAGNOSTIC_DIR, "npl_alpha_ild2_quant_fig4_data.pkl")
ILD2_SLOPES_PKL = os.path.join(ILD2_DIAGNOSTIC_DIR, "npl_alpha_ild2_slopes_fig4_data.pkl")

OUTPUT_PNG = os.path.join(ILD2_DIAGNOSTIC_DIR, "compare_npl_vs_npl_alpha_ild2_quantiles_slopes.png")
OUTPUT_PDF = os.path.join(ILD2_DIAGNOSTIC_DIR, "compare_npl_vs_npl_alpha_ild2_quantiles_slopes.pdf")
ILD2_LABEL_FONTSIZE = 6
HIGHLIGHT_ILD2_ANIMAL = ("LED7", 93)


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
    with open(ILD2_QUANT_PKL, "rb") as handle:
        ild2_quant_data = pickle.load(handle)
    with open(ILD2_SLOPES_PKL, "rb") as handle:
        ild2_slopes_data = pickle.load(handle)

    return npl_quant_data, npl_slopes_data, ild2_quant_data, ild2_slopes_data


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

    ax.set_xlabel("|ILD| (dB)", fontsize=ft.STYLE.LABEL_FONTSIZE)
    ax.set_ylabel("RT Quantile (s)", fontsize=ft.STYLE.LABEL_FONTSIZE)
    ax.set_xscale("log", base=2)
    ax.set_xticks(abs_ild_sorted)
    ax.set_yticks([0.1, 0.2, 0.3, 0.4])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.tick_params(axis="both", which="major", labelsize=ft.STYLE.TICK_FONTSIZE)
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
            fontsize=ft.STYLE.LEGEND_FONTSIZE,
            color="black",
            zorder=8,
        )

    ax.plot([0.1, 0.9], [0.1, 0.9], color="grey", alpha=0.5, linestyle="--", linewidth=2, zorder=0)
    ax.set_xlabel("Data", fontsize=ft.STYLE.LABEL_FONTSIZE)
    ax.set_ylabel("Model", fontsize=ft.STYLE.LABEL_FONTSIZE)
    ax.set_xticks([0.1, 0.5, 0.9])
    ax.set_yticks([0.1, 0.5, 0.9])
    ax.set_xlim(0.1, 0.9)
    ax.set_ylim(0.1, 0.9)
    ax.tick_params(axis="both", labelsize=ft.STYLE.TICK_FONTSIZE)
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


# %%
npl_quant_data, npl_slopes_data, ild2_quant_data, ild2_slopes_data = load_data()

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
plot_quantile_comparison(axes[0], npl_quant_data, ild2_quant_data)
plot_slope_comparison(axes[1], npl_slopes_data, ild2_slopes_data)
print_quantile_fit_metrics(npl_quant_data, ild2_quant_data)

axes[0].set_title("RT Quantiles", fontsize=ft.STYLE.LABEL_FONTSIZE)
axes[1].set_title("Psychometric Slope", fontsize=ft.STYLE.LABEL_FONTSIZE)
fig.suptitle(
    "Black: data    Blue: NPL    Red dashed: NPL + alpha + ILD2 delay",
    fontsize=ft.STYLE.LEGEND_FONTSIZE,
    y=1.02,
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUTPUT_PDF, dpi=300, bbox_inches="tight")

print(f"Saved {OUTPUT_PNG}")
print(f"Saved {OUTPUT_PDF}")

# %%
