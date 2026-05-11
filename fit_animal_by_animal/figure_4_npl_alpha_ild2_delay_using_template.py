# %%
import os
import pickle
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import sem
from sklearn.metrics import r2_score

import figure_template as ft


# %%
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

DATA_DIR = os.path.join(REPO_DIR, "NPL_alpha_ILD2_fit_results", "figure_4_diagnostics")
PSY_DATA_PKL = os.path.join(DATA_DIR, "npl_alpha_ild2_psy_fig4_data.pkl")
QUANT_DATA_PKL = os.path.join(DATA_DIR, "npl_alpha_ild2_quant_fig4_data.pkl")
SLOPES_DATA_PKL = os.path.join(DATA_DIR, "npl_alpha_ild2_slopes_fig4_data.pkl")

OUTPUT_PNG = os.path.join(DATA_DIR, "figure_4_npl_alpha_ild2_delay_psy_quant_slopes.png")
OUTPUT_PDF = os.path.join(DATA_DIR, "figure_4_npl_alpha_ild2_delay_psy_quant_slopes.pdf")


# %%
def _create_innermost_dict():
    return {"empirical": [], "theoretical": []}


def _create_inner_defaultdict():
    return defaultdict(_create_innermost_dict)


def load_data():
    with open(PSY_DATA_PKL, "rb") as handle:
        psy_data = pickle.load(handle)
    with open(QUANT_DATA_PKL, "rb") as handle:
        quant_data = pickle.load(handle)
    with open(SLOPES_DATA_PKL, "rb") as handle:
        slopes_data = pickle.load(handle)
    return psy_data, quant_data, slopes_data


# %%
def plot_psychometric(ax, data):
    empirical_agg = data["empirical_agg"]
    theory_agg = data["theory_agg"]
    ILD_arr = data["ILD_arr"]
    colors = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}

    for ABL in [20, 40, 60]:
        emp = empirical_agg[ABL]
        theo = theory_agg[ABL]
        emp_mean = np.nanmean(emp, axis=0)
        theo_mean = np.nanmean(theo, axis=0)
        ilds = np.array(ILD_arr)

        n_emp = np.sum(~np.isnan(emp), axis=0)
        emp_sem = np.nanstd(emp, axis=0) / np.sqrt(np.maximum(n_emp - 1, 1))
        ax.errorbar(
            ilds,
            emp_mean,
            yerr=emp_sem,
            fmt="o",
            color=colors[ABL],
            capsize=0,
            label=f"Data ABL={ABL}",
            markersize=8,
        )

        valid_idx = np.isfinite(theo_mean)
        if np.sum(valid_idx) >= 4:
            try:
                def sigmoid(x, upper, lower, x0, k):
                    return lower + (upper - lower) / (1 + np.exp(-k * (x - x0)))

                popt, _ = curve_fit(
                    sigmoid,
                    ilds[valid_idx],
                    theo_mean[valid_idx],
                    p0=[1.0, 0.0, 0.0, 1.0],
                    bounds=([0, 0, -np.inf, 0], [1, 1, np.inf, np.inf]),
                )
                ilds_smooth = np.linspace(min(ilds), max(ilds), 200)
                ax.plot(
                    ilds_smooth,
                    sigmoid(ilds_smooth, *popt),
                    linestyle="-",
                    color=colors[ABL],
                    label=f"Model ABL={ABL}",
                )
            except Exception as exc:
                print(f"Could not fit logistic for ABL={ABL}: {exc}")

    ax.set_xlabel("ILD (dB)", fontsize=ft.STYLE.LABEL_FONTSIZE)
    ax.set_ylabel("P(choice = right)", fontsize=ft.STYLE.LABEL_FONTSIZE)
    ax.set_xticks([-15, -5, 5, 15])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(axis="both", labelsize=ft.STYLE.TICK_FONTSIZE)
    ax.axvline(0, alpha=0.5, color="grey", linestyle="--")
    ax.axhline(0.5, alpha=0.5, color="grey", linestyle="--")
    ax.set_ylim(-0.05, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_box_aspect(1)


def plot_quantiles(ax, data):
    plot_data = data["plot_data"]
    continuous_plot_data = data.get("continuous_plot_data", None)
    continuous_abs_ild = data.get("continuous_abs_ild", None)
    QUANTILES_TO_PLOT = data["QUANTILES_TO_PLOT"]
    abs_ild_sorted = data["abs_ild_sorted"]
    ABL_arr = data["ABL_arr"]

    for q_idx, q in enumerate(QUANTILES_TO_PLOT):
        emp_means, emp_sems = [], []
        theo_means, theo_sems, theo_abs_ild_plot = [], [], []

        for abs_ild in abs_ild_sorted:
            all_abl_emp_quantiles = np.concatenate([
                np.array(plot_data[ABL][abs_ild]["empirical"])[:, q_idx] for ABL in ABL_arr
            ])
            emp_means.append(np.nanmean(all_abl_emp_quantiles))
            emp_sems.append(sem(all_abl_emp_quantiles, nan_policy="omit"))

        if continuous_plot_data is not None and continuous_abs_ild is not None:
            for abs_ild in continuous_abs_ild:
                all_abl_theo_q = []
                for ABL in ABL_arr:
                    if len(continuous_plot_data[ABL][abs_ild]["theoretical"]) > 0:
                        all_abl_theo_q.extend(np.array(continuous_plot_data[ABL][abs_ild]["theoretical"])[:, q_idx])
                if len(all_abl_theo_q) > 0:
                    theo_abs_ild_plot.append(abs_ild)
                    theo_means.append(np.nanmean(all_abl_theo_q))
                    theo_sems.append(sem(all_abl_theo_q, nan_policy="omit"))

        ax.errorbar(
            abs_ild_sorted,
            emp_means,
            yerr=emp_sems,
            fmt="o",
            color="black",
            markersize=8,
            capsize=0,
            label=f"Data q={q:.2f}" if q_idx == 0 else "_nolegend_",
        )
        if len(theo_abs_ild_plot) > 0:
            ax.plot(
                theo_abs_ild_plot,
                theo_means,
                "-",
                color="tab:red",
                linewidth=1.5,
                label=f"Model q={q:.2f}" if q_idx == 0 else "_nolegend_",
            )
            ax.fill_between(
                theo_abs_ild_plot,
                np.array(theo_means) - np.array(theo_sems),
                np.array(theo_means) + np.array(theo_sems),
                color="tab:red",
                alpha=0.2,
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


def plot_slopes(ax, data):
    data_means = data["data_means"]
    ild2_means = data.get("npl_alpha_ild2_means", data["norm_means"])

    ax.scatter(data_means, ild2_means, marker="o", s=64, facecolors="w", edgecolors="k", linewidths=1.5)
    ax.set_xlabel("Data", fontsize=ft.STYLE.LABEL_FONTSIZE)
    ax.set_ylabel("Model", fontsize=ft.STYLE.LABEL_FONTSIZE)
    ax.set_xticks([0.1, 0.5, 0.9])
    ax.set_yticks([0.1, 0.5, 0.9])
    ax.set_xlim(0.1, 0.9)
    ax.set_ylim(0.1, 0.9)
    ax.tick_params(axis="both", labelsize=ft.STYLE.TICK_FONTSIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.plot([0.1, 0.9], [0.1, 0.9], color="grey", alpha=0.5, linestyle="--", linewidth=2, zorder=0)
    ax.set_box_aspect(1)

    finite = np.isfinite(data_means) & np.isfinite(ild2_means)
    if np.sum(finite) >= 2:
        r2 = r2_score(data_means[finite], ild2_means[finite])
        print(f"R² for NPL + alpha + ILD2 delay slopes: {r2:.2f}")


# %%
psy_data, quant_data, slopes_data = load_data()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
plot_psychometric(axes[0], psy_data)
plot_quantiles(axes[1], quant_data)
plot_slopes(axes[2], slopes_data)

for ax, label in zip(axes, ["Psychometric", "RT Quantiles", "Psychometric Slope"]):
    ax.set_title(label, fontsize=ft.STYLE.LABEL_FONTSIZE)

plt.tight_layout()
fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUTPUT_PDF, dpi=300, bbox_inches="tight")

print(f"Saved {OUTPUT_PNG}")
print(f"Saved {OUTPUT_PDF}")

# %%
