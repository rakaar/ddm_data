# %%
import os
import pickle

os.makedirs("/tmp/matplotlib", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem

import figure_template as ft

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]


# %%
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

DATA_DIR = os.path.join(REPO_DIR, "NPL_alpha_ILD2_fit_results", "figure_4_diagnostics")
OUTPUT_DIR = os.path.join(REPO_DIR, "NPL_alpha_ILD2_fit_results", "figure_4_diagnostics_part2")

MODEL_PLOTS = [
    {
        "label": "NPL + alpha + ILD2 delay",
        "quant_data_pkl": os.path.join(DATA_DIR, "npl_alpha_ild2_quant_fig4_data.pkl"),
        "output_stem": "npl_alpha_ild2_delay_quantiles_per_abl",
    },
    {
        "label": "NPL + alpha + ILD2 delay (q10)",
        "quant_data_pkl": os.path.join(DATA_DIR, "npl_alpha_ild2_quant_fig4_data_q10.pkl"),
        "output_stem": "npl_alpha_ild2_delay_quantiles_q10_abl20_60",
        "abl_arr": [20, 60],
    },
    {
        "label": "NPL",
        "quant_data_pkl": os.path.join(SCRIPT_DIR, "norm_quant_fig2_data.pkl"),
        "output_stem": "npl_quantiles_per_abl",
    },
]


# %%
def load_quantile_data(quant_data_pkl):
    with open(quant_data_pkl, "rb") as handle:
        return pickle.load(handle)


def plot_quantiles_for_abl(ax, data, ABL):
    plot_data = data["plot_data"]
    continuous_plot_data = data["continuous_plot_data"]
    continuous_abs_ild = data["continuous_abs_ild"]
    quantiles_to_plot = data["QUANTILES_TO_PLOT"]
    abs_ild_sorted = data["abs_ild_sorted"]

    for q_idx, q in enumerate(quantiles_to_plot):
        emp_means, emp_sems = [], []
        theo_means, theo_sems, theo_abs_ild_plot = [], [], []

        for abs_ild in abs_ild_sorted:
            empirical_values = np.asarray(plot_data[ABL][abs_ild]["empirical"], dtype=float)
            empirical_quantiles = empirical_values[:, q_idx]
            emp_means.append(np.nanmean(empirical_quantiles))
            emp_sems.append(sem(empirical_quantiles, nan_policy="omit"))

        for abs_ild in continuous_abs_ild:
            theoretical_values = np.asarray(continuous_plot_data[ABL][abs_ild]["theoretical"], dtype=float)
            if len(theoretical_values) == 0:
                continue
            theoretical_quantiles = theoretical_values[:, q_idx]
            theo_abs_ild_plot.append(abs_ild)
            theo_means.append(np.nanmean(theoretical_quantiles))
            theo_sems.append(sem(theoretical_quantiles, nan_policy="omit"))

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

    ax.set_title(f"ABL = {ABL}", fontsize=ft.STYLE.LEGEND_FONTSIZE)
    ax.set_xlabel("|ILD| (dB)", fontsize=ft.STYLE.LABEL_FONTSIZE)
    ax.set_xscale("log", base=2)
    ax.set_xticks(abs_ild_sorted)
    ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.tick_params(axis="both", which="major", labelsize=ft.STYLE.TICK_FONTSIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_box_aspect(1)


def print_sanity_counts(label, data):
    print(f"\n{label}: per-ABL quantile sample counts")
    for ABL in data["ABL_arr"]:
        empirical_counts = [
            len(data["plot_data"][ABL][abs_ild]["empirical"])
            for abs_ild in data["abs_ild_sorted"]
        ]
        theory_counts = [
            len(data["continuous_plot_data"][ABL][abs_ild]["theoretical"])
            for abs_ild in [1.0, 2.0, 4.0, 8.0, 16.0]
        ]
        print(
            f"  ABL={ABL}: empirical entries by |ILD|={empirical_counts}; "
            f"model signed entries at |ILD| 1/2/4/8/16={theory_counts}"
        )


def save_per_abl_quantile_figure(model_plot):
    quantile_data = load_quantile_data(model_plot["quant_data_pkl"])
    print_sanity_counts(model_plot["label"], quantile_data)
    abl_arr = model_plot.get("abl_arr", quantile_data["ABL_arr"])

    fig, axes = plt.subplots(1, len(abl_arr), figsize=(5 * len(abl_arr), 5), sharex=True, sharey=True)
    if len(abl_arr) == 1:
        axes = [axes]
    for ax, ABL in zip(axes, abl_arr):
        plot_quantiles_for_abl(ax, quantile_data, ABL)

    axes[0].set_ylabel("RT Quantile (s)", fontsize=ft.STYLE.LABEL_FONTSIZE)
    for ax in axes[1:]:
        ax.set_ylabel("")

    for ax in axes:
        ax.set_ylim(0.06, 0.7)

    output_png = os.path.join(OUTPUT_DIR, f"{model_plot['output_stem']}.png")
    output_pdf = os.path.join(OUTPUT_DIR, f"{model_plot['output_stem']}.pdf")
    fig.suptitle(model_plot["label"], fontsize=ft.STYLE.LEGEND_FONTSIZE)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.8, wspace=0.24)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {output_png}")
    print(f"Saved {output_pdf}")


# %%
os.makedirs(OUTPUT_DIR, exist_ok=True)
for model_plot in MODEL_PLOTS:
    save_per_abl_quantile_figure(model_plot)

# %%
