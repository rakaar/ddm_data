# %%
"""
Assemble a paper Fig. 4-style panel for NPL+alpha Gamma+Omega MSE params.

Inputs are produced by build_mse_gamma_omega_fig4_v2_data.py in this folder.
"""

# %%
# =============================================================================
# Editable parameters
# =============================================================================
from pathlib import Path
import os
import pickle
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


SCRIPT_DIR = Path(__file__).resolve().parent
ANIMAL_DIR = SCRIPT_DIR.parent
if str(ANIMAL_DIR) not in sys.path:
    sys.path.insert(0, str(ANIMAL_DIR))

import figure_template as ft

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Nimbus Sans", "Helvetica Neue", "Arial", "Liberation Sans", "sans-serif"],
    "pdf.use14corefonts": True,
    "ps.useafm": True,
})


PSY_DATA_PKL = SCRIPT_DIR / "mse_gamma_omega_npl_alpha_psy_fig4_v2_data.pkl"
QUANT_DATA_PKL = SCRIPT_DIR / "mse_gamma_omega_npl_alpha_quant_fig4_v2_data.pkl"
SLOPES_DATA_PKL = SCRIPT_DIR / "mse_gamma_omega_npl_alpha_slopes_fig4_v2_data.pkl"
GAMMA_OMEGA_DATA_PKL = SCRIPT_DIR / "mse_gamma_omega_npl_alpha_gamma_fig4_v2_data.pkl"

OUTPUT_PNG = SCRIPT_DIR / "figure4_v2_mse_gamma_omega_npl_alpha.png"
OUTPUT_PDF = SCRIPT_DIR / "figure4_v2_mse_gamma_omega_npl_alpha.pdf"

ABLS = [20, 40, 60]
ABL_COLORS = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}
PLOT_LABEL_FONTSIZE = ft.STYLE.LABEL_FONTSIZE
TICK_FONTSIZE = ft.STYLE.TICK_FONTSIZE
DATA_MARKER_SIZE = 8.0
THEORY_LINEWIDTH = 1.5
REFERENCE_LINEWIDTH = 1.5
AXIS_LINEWIDTH = plt.rcParams["axes.linewidth"]
SHADE_ALPHA = 0.2


# %%
# =============================================================================
# Helpers
# =============================================================================
def load_pickle(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def sigmoid(x, upper, lower, x0, k):
    return lower + (upper - lower) / (1 + np.exp(-k * (x - x0)))


def fit_psychometric_sigmoid(ild_values, right_choice_probs):
    ild_values = np.asarray(ild_values, dtype=float)
    right_choice_probs = np.asarray(right_choice_probs, dtype=float)
    valid_idx = np.isfinite(ild_values) & np.isfinite(right_choice_probs)
    if np.sum(valid_idx) < 4:
        return None
    try:
        popt, _ = curve_fit(
            sigmoid,
            ild_values[valid_idx],
            right_choice_probs[valid_idx],
            p0=[1.0, 0.0, 0.0, 1.0],
            bounds=([0, 0, -np.inf, 0], [1, 1, np.inf, np.inf]),
            maxfev=20000,
        )
        return popt
    except Exception:
        return None


def mean_sem(values, axis=0):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    n = np.sum(finite, axis=axis)
    mean = np.nanmean(values, axis=axis)
    sd = np.nanstd(values, axis=axis, ddof=1)
    curr_sem = sd / np.sqrt(np.maximum(n, 1))
    curr_sem = np.where(n > 1, curr_sem, np.nan)
    return mean, curr_sem, n


def flat_mean_sem(values):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    n = int(np.sum(finite))
    if n == 0:
        return np.nan, np.nan, 0
    mean = float(np.nanmean(values))
    curr_sem = float(np.nanstd(values, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return mean, curr_sem, n


def pearson_r(data_values, model_values):
    data_values = np.asarray(data_values, dtype=float)
    model_values = np.asarray(model_values, dtype=float)
    valid = np.isfinite(data_values) & np.isfinite(model_values)
    if np.sum(valid) < 2:
        return np.nan
    if np.nanstd(data_values[valid]) == 0 or np.nanstd(model_values[valid]) == 0:
        return np.nan
    return float(np.corrcoef(data_values[valid], model_values[valid])[0, 1])


# %%
# =============================================================================
# Plotting
# =============================================================================
def plot_psychometric(ax, data):
    empirical_agg = data["empirical_agg"]
    theory_agg = data["theory_agg"]
    ild_arr = np.asarray(data["ILD_arr"], dtype=float)

    for abl in ABLS:
        color = ABL_COLORS[abl]
        emp = np.asarray(empirical_agg[abl], dtype=float)
        theo = np.asarray(theory_agg[abl], dtype=float)
        emp_mean, emp_sem, _n_emp = mean_sem(emp, axis=0)
        theo_mean = np.nanmean(theo, axis=0)

        ax.errorbar(
            ild_arr,
            emp_mean,
            yerr=emp_sem,
            fmt="o",
            color=color,
            markersize=DATA_MARKER_SIZE,
            capsize=0,
            linestyle="none",
        )

        popt = fit_psychometric_sigmoid(ild_arr, theo_mean)
        if popt is not None:
            valid_ilds = ild_arr[np.isfinite(theo_mean)]
            ild_smooth = np.linspace(np.nanmin(valid_ilds), np.nanmax(valid_ilds), 200)
            ax.plot(ild_smooth, sigmoid(ild_smooth, *popt), color=color, linewidth=THEORY_LINEWIDTH)

    ax.set_xlabel("ILD (dB)", fontsize=PLOT_LABEL_FONTSIZE)
    ax.set_ylabel("P(choice = right)", fontsize=PLOT_LABEL_FONTSIZE)
    ax.set_xlim(-17, 17)
    ax.set_xticks([-15, -5, 5, 15])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.axvline(0, alpha=0.5, color="grey", linestyle="--", linewidth=REFERENCE_LINEWIDTH)
    ax.axhline(0.5, alpha=0.5, color="grey", linestyle="--", linewidth=REFERENCE_LINEWIDTH)
    ax.set_ylim(-0.05, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(AXIS_LINEWIDTH)
    ax.spines["left"].set_linewidth(AXIS_LINEWIDTH)


def plot_quantiles(ax, data):
    plot_data = data["plot_data"]
    theory_source = data["continuous_plot_data"]
    theory_x = np.asarray(data["continuous_abs_ild"], dtype=float)
    quantiles = [float(q) for q in data["QUANTILES_TO_PLOT"]]
    abs_ild_sorted = [float(x) for x in data["abs_ild_sorted"]]
    abl_arr = [int(x) for x in data["ABL_arr"]]

    for q_idx, _q in enumerate(quantiles):
        emp_means = []
        emp_sems = []
        for abs_ild in abs_ild_sorted:
            values = []
            for abl in abl_arr:
                entries = plot_data[abl][float(abs_ild)]["empirical"]
                if len(entries) > 0:
                    values.extend(np.asarray(entries, dtype=float)[:, q_idx])
            mean, curr_sem, _n = flat_mean_sem(values)
            emp_means.append(mean)
            emp_sems.append(curr_sem)

        ax.errorbar(
            abs_ild_sorted,
            emp_means,
            yerr=emp_sems,
            fmt="o",
            color="black",
            markersize=DATA_MARKER_SIZE,
            capsize=0,
            alpha=0.85,
        )

        theo_means = []
        theo_sems = []
        valid_x = []
        for abs_ild in theory_x:
            values = []
            for abl in abl_arr:
                entries = theory_source[abl][float(abs_ild)]["theoretical"]
                if len(entries) > 0:
                    values.extend(np.asarray(entries, dtype=float)[:, q_idx])
            mean, curr_sem, n = flat_mean_sem(values)
            if n > 0:
                valid_x.append(float(abs_ild))
                theo_means.append(mean)
                theo_sems.append(curr_sem)

        if valid_x:
            valid_x = np.asarray(valid_x, dtype=float)
            theo_means = np.asarray(theo_means, dtype=float)
            theo_sems = np.asarray(theo_sems, dtype=float)
            ax.plot(valid_x, theo_means, color="tab:red", linewidth=THEORY_LINEWIDTH)
            ax.fill_between(
                valid_x,
                theo_means - theo_sems,
                theo_means + theo_sems,
                color="tab:red",
                alpha=SHADE_ALPHA,
                linewidth=0,
            )

    ax.set_xlabel("|ILD| (dB)", fontsize=PLOT_LABEL_FONTSIZE)
    ax.set_ylabel("RT Quantile (s)", fontsize=PLOT_LABEL_FONTSIZE)
    ax.set_xscale("log", base=2)
    ax.set_xticks(abs_ild_sorted)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_yticks([0.1, 0.2, 0.3, 0.4])
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(AXIS_LINEWIDTH)
    ax.spines["left"].set_linewidth(AXIS_LINEWIDTH)


def plot_slopes(ax, data):
    data_means = np.asarray(data["data_means"], dtype=float)
    model_means = np.asarray(data["model_means"], dtype=float)

    ax.scatter(
        data_means,
        model_means,
        marker="o",
        s=64,
        facecolors="white",
        edgecolors="black",
        linewidths=1.5,
    )
    ax.plot([0.1, 0.9], [0.1, 0.9], color="grey", alpha=0.5, linestyle="--", linewidth=2, zorder=0)
    ax.set_xlabel("Data", fontsize=PLOT_LABEL_FONTSIZE)
    ax.set_ylabel("Model", fontsize=PLOT_LABEL_FONTSIZE)
    ax.set_xticks([0.1, 0.5, 0.9])
    ax.set_yticks([0.1, 0.5, 0.9])
    ax.set_xlim(0.1, 0.9)
    ax.set_ylim(0.1, 0.9)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(AXIS_LINEWIDTH)
    ax.spines["left"].set_linewidth(AXIS_LINEWIDTH)

    r = pearson_r(data_means, model_means)
    if np.isfinite(r):
        print(f"Psychometric slope Pearson r = {r:.3f}")


def plot_gamma_omega(ax, data, param):
    condition_df = pd.DataFrame(data["condition_rows"])
    model_df = pd.DataFrame(data["model_summary_rows"])
    if param not in {"gamma", "omega"}:
        raise ValueError("param must be 'gamma' or 'omega'")

    condition_col = f"condition_{param}"
    model_mean_col = f"model_{param}_mean"
    model_sem_col = f"model_{param}_sem"
    ylabel = r"Discriminability $\Gamma$" if param == "gamma" else r"Omega $\omega$"

    for abl in ABLS:
        color = ABL_COLORS[abl]
        cond_subset = condition_df[condition_df["ABL"] == abl].copy()
        cond_summary = (
            cond_subset.groupby("ILD", sort=True)[condition_col]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        cond_summary["sem"] = cond_summary["std"] / np.sqrt(cond_summary["count"])
        ax.errorbar(
            cond_summary["ILD"],
            cond_summary["mean"],
            yerr=cond_summary["sem"],
            fmt="o",
            ms=DATA_MARKER_SIZE,
            mfc="white",
            mec=color,
            ecolor=color,
            color=color,
            capsize=0,
            linestyle="none",
        )

        model_subset = model_df[model_df["ABL"] == abl].sort_values("ILD")
        x = model_subset["ILD"].to_numpy(dtype=float)
        y = model_subset[model_mean_col].to_numpy(dtype=float)
        curr_sem = model_subset[model_sem_col].to_numpy(dtype=float)
        ax.plot(x, y, color=color, linestyle="-", linewidth=THEORY_LINEWIDTH)
        ax.fill_between(x, y - curr_sem, y + curr_sem, color=color, alpha=SHADE_ALPHA, linewidth=0)

    ax.set_xlabel("ILD", fontsize=PLOT_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=PLOT_LABEL_FONTSIZE)
    ax.set_xlim(-17, 17)
    ax.set_xticks([-15, -5, 5, 15])
    if param == "gamma":
        ax.set_yticks([-2, 0, 2])
        ax.set_ylim(-3, 3)
    else:
        ax.set_yticks([3, 5, 7])
        ax.set_ylim(bottom=2)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(AXIS_LINEWIDTH)
    ax.spines["left"].set_linewidth(AXIS_LINEWIDTH)


# %%
# =============================================================================
# Load data and assemble figure
# =============================================================================
psy_data = load_pickle(PSY_DATA_PKL)
quant_data = load_pickle(QUANT_DATA_PKL)
slopes_data = load_pickle(SLOPES_DATA_PKL)
gamma_omega_data = load_pickle(GAMMA_OMEGA_DATA_PKL)

print(f"Psychometric data: {PSY_DATA_PKL}")
print(f"Quantile data: {QUANT_DATA_PKL}")
print(f"Slope data: {SLOPES_DATA_PKL}")
print(f"Gamma/Omega data: {GAMMA_OMEGA_DATA_PKL}")
print(f"Quantiles used: {quant_data['QUANTILES_TO_PLOT']}")
print(f"Gamma/Omega model source: {gamma_omega_data['model_source']}")

builder = ft.FigureBuilder(
    sup_title="",
    n_rows=2,
    n_cols=3,
    figsize=(18, 10),
    hspace=0.3,
    wspace=0.3,
)

ax_psych = builder.fig.add_subplot(builder.gs[0, 0])
plot_psychometric(ax_psych, psy_data)

ax_quant = builder.fig.add_subplot(builder.gs[0, 1])
plot_quantiles(ax_quant, quant_data)

ax_gamma = builder.fig.add_subplot(builder.gs[1, 0])
plot_gamma_omega(ax_gamma, gamma_omega_data, "gamma")

ax_omega = builder.fig.add_subplot(builder.gs[1, 1])
plot_gamma_omega(ax_omega, gamma_omega_data, "omega")

ax_slopes = builder.fig.add_subplot(builder.gs[1, 2])
plot_slopes(ax_slopes, slopes_data)

fig = builder.finish()
fig.tight_layout()
fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUTPUT_PDF, dpi=300, bbox_inches="tight")

print(f"Saved figure: {OUTPUT_PNG}")
print(f"Saved figure: {OUTPUT_PDF}")
