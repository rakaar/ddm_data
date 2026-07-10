# %%
"""
Plot the direct NPL+alpha SVI supplementary Figure 4 diagnostic.

The five paper panels match Figure 4 v2. The upper-triangular parameter block
shows animal-wise posterior means, 95% covariance ellipses, and marginal 95%
credible intervals from the patience-12 variational posterior samples.
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
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
from matplotlib.ticker import FormatStrFormatter
from scipy.optimize import curve_fit


SCRIPT_DIR = Path(__file__).resolve().parent
ANIMAL_DIR = SCRIPT_DIR.parent
if str(ANIMAL_DIR) not in sys.path:
    sys.path.insert(0, str(ANIMAL_DIR))

import figure_template as ft


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Helvetica",
            "Nimbus Sans",
            "Helvetica Neue",
            "Arial",
            "Liberation Sans",
            "sans-serif",
        ],
        "pdf.use14corefonts": True,
        "ps.useafm": True,
    }
)

BUNDLE_PKL = SCRIPT_DIR / "npl_svi_patience12_fig4_supplementary_bundle.pkl"
OUTPUT_PNG = SCRIPT_DIR / "figure4_supplementary_npl_svi_patience12.png"
OUTPUT_PDF = SCRIPT_DIR / "figure4_supplementary_npl_svi_patience12.pdf"

ABLS = [20, 40, 60]
ABL_COLORS = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}
PLOT_LABEL_FONTSIZE = ft.STYLE.LABEL_FONTSIZE
TICK_FONTSIZE = ft.STYLE.TICK_FONTSIZE
CORNER_LABEL_FONTSIZE = 22
CORNER_TICK_FONTSIZE = 15
DATA_MARKER_SIZE = 8.0
THEORY_LINEWIDTH = 1.5
REFERENCE_LINEWIDTH = 1.5
AXIS_LINEWIDTH = plt.rcParams["axes.linewidth"]
SHADE_ALPHA = 0.2

PARAMS = ["rate_lambda", "T_0", "theta_E", "rate_norm_l", "alpha"]
PARAM_LABELS = {
    "rate_lambda": r"$\lambda^\prime$",
    "T_0": r"$T_0$",
    "theta_E": r"$\theta_E$",
    "rate_norm_l": r"$\ell$",
    "alpha": r"$\alpha$",
}
PARAM_TICKS = {
    "rate_lambda": [2.0, 4.0],
    "T_0": [0.1, 0.2],
    "theta_E": [2.0, 3.0],
    "rate_norm_l": [0.8, 0.9, 1.0],
    "alpha": [0.5, 1.0],
}
CORNER_COLOR = "#2b6cb0"
ELLIPSE_QUANTILE = 0.95


# %%
# =============================================================================
# Shared plotting helpers
# =============================================================================
def sigmoid(x, upper, lower, x0, k):
    return lower + (upper - lower) / (1 + np.exp(-k * (x - x0)))


def fit_psychometric_sigmoid(ild_values, right_choice_probs):
    ild_values = np.asarray(ild_values, dtype=float)
    right_choice_probs = np.asarray(right_choice_probs, dtype=float)
    valid = np.isfinite(ild_values) & np.isfinite(right_choice_probs)
    if np.sum(valid) < 4:
        return None
    try:
        parameters, _covariance = curve_fit(
            sigmoid,
            ild_values[valid],
            right_choice_probs[valid],
            p0=[1.0, 0.0, 0.0, 1.0],
            bounds=([0, 0, -np.inf, 0], [1, 1, np.inf, np.inf]),
            maxfev=20_000,
        )
        return parameters
    except Exception:
        return None


def mean_sem(values, axis=0):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    n = np.sum(finite, axis=axis)
    mean = np.nanmean(values, axis=axis)
    sd = np.nanstd(values, axis=axis, ddof=1)
    curr_sem = sd / np.sqrt(np.maximum(n, 1))
    return mean, np.where(n > 1, curr_sem, np.nan), n


def flat_mean_sem(values):
    values = np.asarray(values, dtype=float)
    n = int(np.sum(np.isfinite(values)))
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
    if np.std(data_values[valid]) == 0 or np.std(model_values[valid]) == 0:
        return np.nan
    return float(np.corrcoef(data_values[valid], model_values[valid])[0, 1])


def ticks_for_param(param, limits):
    ticks = [
        tick
        for tick in PARAM_TICKS.get(param, [])
        if limits[0] <= tick <= limits[1]
    ]
    if len(ticks) >= 2:
        return ticks
    span = limits[1] - limits[0]
    return [limits[0] + 0.15 * span, limits[1] - 0.15 * span]


# %%
# =============================================================================
# Figure 4 paper panels
# =============================================================================
def plot_psychometric(ax, data):
    ild_arr = np.asarray(data["ILD_arr"], dtype=float)
    for abl in ABLS:
        color = ABL_COLORS[abl]
        empirical = np.asarray(data["empirical_agg"][abl], dtype=float)
        theoretical = np.asarray(data["theory_agg"][abl], dtype=float)
        empirical_mean, empirical_sem, _n = mean_sem(empirical, axis=0)
        theoretical_mean = np.nanmean(theoretical, axis=0)

        ax.errorbar(
            ild_arr,
            empirical_mean,
            yerr=empirical_sem,
            fmt="o",
            color=color,
            markersize=DATA_MARKER_SIZE,
            capsize=0,
            linestyle="none",
        )
        fit_parameters = fit_psychometric_sigmoid(ild_arr, theoretical_mean)
        if fit_parameters is not None:
            valid_ilds = ild_arr[np.isfinite(theoretical_mean)]
            smooth_ilds = np.linspace(np.min(valid_ilds), np.max(valid_ilds), 200)
            ax.plot(
                smooth_ilds,
                sigmoid(smooth_ilds, *fit_parameters),
                color=color,
                linewidth=THEORY_LINEWIDTH,
            )

    ax.set_xlabel("ILD (dB)", fontsize=PLOT_LABEL_FONTSIZE)
    ax.set_ylabel("P(choice = right)", fontsize=PLOT_LABEL_FONTSIZE)
    ax.set_xlim(-17, 17)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([-15, -5, 5, 15])
    ax.set_yticks([0, 0.5, 1])
    ax.axvline(0, alpha=0.5, color="grey", linestyle="--", linewidth=REFERENCE_LINEWIDTH)
    ax.axhline(0.5, alpha=0.5, color="grey", linestyle="--", linewidth=REFERENCE_LINEWIDTH)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(AXIS_LINEWIDTH)
    ax.spines["left"].set_linewidth(AXIS_LINEWIDTH)
    ax.set_box_aspect(1)


def plot_quantiles(ax, data):
    plot_data = data["plot_data"]
    theory_source = data["continuous_plot_data"]
    theory_x = np.asarray(data["continuous_abs_ild"], dtype=float)
    quantiles = [float(value) for value in data["QUANTILES_TO_PLOT"]]
    abs_ilds = [float(value) for value in data["abs_ild_sorted"]]
    abl_arr = [int(value) for value in data["ABL_arr"]]

    for quantile_index, _quantile in enumerate(quantiles):
        empirical_means = []
        empirical_sems = []
        for abs_ild in abs_ilds:
            values = []
            for abl in abl_arr:
                entries = plot_data[abl][abs_ild]["empirical"]
                if len(entries) > 0:
                    values.extend(np.asarray(entries, dtype=float)[:, quantile_index])
            mean, curr_sem, _n = flat_mean_sem(values)
            empirical_means.append(mean)
            empirical_sems.append(curr_sem)
        ax.errorbar(
            abs_ilds,
            empirical_means,
            yerr=empirical_sems,
            fmt="o",
            color="black",
            markersize=DATA_MARKER_SIZE,
            capsize=0,
            alpha=0.85,
        )

        valid_x = []
        theory_means = []
        theory_sems = []
        for abs_ild in theory_x:
            values = []
            for abl in abl_arr:
                entries = theory_source[abl][float(abs_ild)]["theoretical"]
                if len(entries) > 0:
                    values.extend(np.asarray(entries, dtype=float)[:, quantile_index])
            mean, curr_sem, n = flat_mean_sem(values)
            if n > 0:
                valid_x.append(float(abs_ild))
                theory_means.append(mean)
                theory_sems.append(curr_sem)
        if valid_x:
            valid_x = np.asarray(valid_x, dtype=float)
            theory_means = np.asarray(theory_means, dtype=float)
            theory_sems = np.asarray(theory_sems, dtype=float)
            ax.plot(valid_x, theory_means, color="tab:red", linewidth=THEORY_LINEWIDTH)
            ax.fill_between(
                valid_x,
                theory_means - theory_sems,
                theory_means + theory_sems,
                color="tab:red",
                alpha=SHADE_ALPHA,
                linewidth=0,
            )

    ax.set_xlabel("|ILD| (dB)", fontsize=PLOT_LABEL_FONTSIZE)
    ax.set_ylabel("RT Quantile (s)", fontsize=PLOT_LABEL_FONTSIZE)
    ax.set_xscale("log", base=2)
    ax.set_xticks(abs_ilds)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_yticks([0.1, 0.2, 0.3, 0.4])
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(AXIS_LINEWIDTH)
    ax.spines["left"].set_linewidth(AXIS_LINEWIDTH)
    ax.set_box_aspect(1)


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
    ax.plot(
        [0.1, 0.9],
        [0.1, 0.9],
        color="grey",
        alpha=0.5,
        linestyle="--",
        linewidth=2,
        zorder=0,
    )
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
    ax.set_box_aspect(1)

    correlation = pearson_r(data_means, model_means)
    if np.isfinite(correlation):
        print(f"Psychometric slope Pearson r = {correlation:.3f}")


def plot_gamma_omega(ax, data, parameter):
    condition_df = pd.DataFrame(data["condition_rows"])
    model_df = pd.DataFrame(data["model_summary_rows"])
    condition_column = f"condition_{parameter}"
    model_mean_column = f"model_{parameter}_mean"
    model_sem_column = f"model_{parameter}_sem"

    for abl in ABLS:
        color = ABL_COLORS[abl]
        condition_subset = condition_df[condition_df["ABL"] == abl]
        condition_summary = (
            condition_subset.groupby("ILD", sort=True)[condition_column]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        condition_summary["sem"] = (
            condition_summary["std"] / np.sqrt(condition_summary["count"])
        )
        ax.errorbar(
            condition_summary["ILD"],
            condition_summary["mean"],
            yerr=condition_summary["sem"],
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
        x_values = model_subset["ILD"].to_numpy(dtype=float)
        means = model_subset[model_mean_column].to_numpy(dtype=float)
        sems = model_subset[model_sem_column].to_numpy(dtype=float)
        ax.plot(x_values, means, color=color, linewidth=THEORY_LINEWIDTH)
        ax.fill_between(
            x_values,
            means - sems,
            means + sems,
            color=color,
            alpha=SHADE_ALPHA,
            linewidth=0,
        )

    ax.set_xlabel("ILD", fontsize=PLOT_LABEL_FONTSIZE)
    if parameter == "gamma":
        ax.set_ylabel(r"Discriminability $\Gamma$", fontsize=PLOT_LABEL_FONTSIZE)
        ax.set_yticks([-2, 0, 2])
        ax.set_ylim(-3, 3)
    else:
        ax.set_ylabel(r"Omega $\omega$", fontsize=PLOT_LABEL_FONTSIZE)
        ax.set_yticks([3, 5, 7])
        ax.set_ylim(bottom=2)
    ax.set_xlim(-17, 17)
    ax.set_xticks([-15, -5, 5, 15])
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(AXIS_LINEWIDTH)
    ax.spines["left"].set_linewidth(AXIS_LINEWIDTH)
    ax.set_box_aspect(1)


# %%
# =============================================================================
# Upper-triangular posterior corner
# =============================================================================
def plot_posterior_upper_corner(axes, posterior_rows):
    param_index = {param: index for index, param in enumerate(PARAMS)}
    means = {
        param: np.array([row["mean"][param] for row in posterior_rows], dtype=float)
        for param in PARAMS
    }
    q025 = {
        param: np.array([row["q025"][param] for row in posterior_rows], dtype=float)
        for param in PARAMS
    }
    q975 = {
        param: np.array([row["q975"][param] for row in posterior_rows], dtype=float)
        for param in PARAMS
    }
    covariances = np.asarray(
        [row["covariance"] for row in posterior_rows], dtype=float
    )
    chi_square_scale = -2.0 * np.log(1.0 - ELLIPSE_QUANTILE)

    limits = {}
    tick_map = {}
    for param in PARAMS:
        index = param_index[param]
        ellipse_radius = np.sqrt(
            chi_square_scale * np.maximum(covariances[:, index, index], 0.0)
        )
        low = float(
            np.min(np.concatenate([q025[param], means[param] - ellipse_radius]))
        )
        high = float(
            np.max(np.concatenate([q975[param], means[param] + ellipse_radius]))
        )
        padding = 0.08 * (high - low) if high > low else 0.1 * abs(high)
        if padding == 0:
            padding = 0.5
        limits[param] = (low - padding, high + padding)
        tick_map[param] = ticks_for_param(param, limits[param])

    ranked_order = {param: np.argsort(means[param])[::-1] for param in PARAMS}
    n_params = len(PARAMS)
    for row_index, y_param in enumerate(PARAMS):
        for column_index, x_param in enumerate(PARAMS):
            ax = axes[row_index, column_index]
            if row_index > column_index:
                ax.axis("off")
                continue

            if row_index == column_index:
                order = ranked_order[x_param]
                y_values = np.arange(len(order), dtype=float)
                x_values = means[x_param][order]
                lower_error = x_values - q025[x_param][order]
                upper_error = q975[x_param][order] - x_values
                ax.axvline(
                    np.median(means[x_param]),
                    color="0.65",
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.45,
                    zorder=1,
                )
                ax.errorbar(
                    x_values,
                    y_values,
                    xerr=np.vstack([lower_error, upper_error]),
                    fmt="o",
                    color=CORNER_COLOR,
                    ecolor=CORNER_COLOR,
                    elinewidth=1.0,
                    capsize=0,
                    markersize=3.8,
                    markeredgecolor="black",
                    markeredgewidth=0.4,
                    alpha=0.9,
                    zorder=4,
                )
                ax.set_xlim(limits[x_param])
                ax.set_ylim(-0.5, len(order) - 0.5)
                if row_index == 0:
                    ax.set_yticks([0, len(order) - 1])
                    ax.set_yticklabels(["30", "1"])
                    ax.set_ylabel(
                        "Rat ID", fontsize=CORNER_LABEL_FONTSIZE, labelpad=19
                    )
                else:
                    ax.set_yticks([])
            else:
                x_index = param_index[x_param]
                y_index = param_index[y_param]
                for animal_index, covariance in enumerate(covariances):
                    pair_covariance = covariance[
                        np.ix_([x_index, y_index], [x_index, y_index])
                    ]
                    eigenvalues, eigenvectors = np.linalg.eigh(pair_covariance)
                    eigen_order = np.argsort(eigenvalues)[::-1]
                    eigenvalues = np.maximum(eigenvalues[eigen_order], 0.0)
                    eigenvectors = eigenvectors[:, eigen_order]
                    width = 2.0 * np.sqrt(chi_square_scale * eigenvalues[0])
                    height = 2.0 * np.sqrt(chi_square_scale * eigenvalues[1])
                    angle = np.degrees(
                        np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
                    )
                    if width > 0 and height > 0:
                        ax.add_patch(
                            Ellipse(
                                (means[x_param][animal_index], means[y_param][animal_index]),
                                width=width,
                                height=height,
                                angle=angle,
                                facecolor="none",
                                edgecolor=CORNER_COLOR,
                                linewidth=0.8,
                                alpha=0.75,
                                zorder=3,
                            )
                        )
                ax.scatter(
                    means[x_param],
                    means[y_param],
                    s=24,
                    c=CORNER_COLOR,
                    edgecolor="black",
                    linewidths=0.4,
                    alpha=0.9,
                    zorder=4,
                )
                ax.set_xlim(limits[x_param])
                ax.set_ylim(limits[y_param])

            ax.set_xticks(tick_map[x_param])
            if x_param in {"rate_lambda", "alpha"}:
                ax.xaxis.set_major_formatter(FormatStrFormatter("%.2g"))
            else:
                ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            if row_index != column_index:
                ax.set_yticks(tick_map[y_param])
                if y_param in {"rate_lambda", "alpha"}:
                    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2g"))
                else:
                    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

            if row_index == 0:
                ax.set_xlabel(
                    PARAM_LABELS[x_param], fontsize=CORNER_LABEL_FONTSIZE, labelpad=8
                )
                ax.xaxis.set_label_position("top")
                ax.xaxis.tick_top()
                ax.tick_params(axis="x", labeltop=True, labelbottom=False)
            else:
                ax.set_xticklabels([])

            if column_index == n_params - 1 and row_index < column_index:
                ax.set_ylabel(
                    PARAM_LABELS[y_param], fontsize=CORNER_LABEL_FONTSIZE, labelpad=8
                )
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.tick_params(axis="y", labelright=True, labelleft=False)
            elif not (row_index == column_index and row_index == 0):
                ax.set_yticklabels([])

            ax.tick_params(
                axis="both", which="major", labelsize=CORNER_TICK_FONTSIZE, pad=1
            )
            for spine in ["left", "bottom", "right", "top"]:
                ax.spines[spine].set_linewidth(0.9)
                ax.spines[spine].set_visible(True)
            ax.grid(False)
            ax.set_box_aspect(1)


# %%
# =============================================================================
# Load data and assemble the matched Figure 4 v2 layout
# =============================================================================
with BUNDLE_PKL.open("rb") as handle:
    bundle = pickle.load(handle)

if bundle["params"] != PARAMS:
    raise RuntimeError(f"Unexpected posterior parameter order: {bundle['params']}")
posterior_rows = bundle["posterior_rows"]
if len(posterior_rows) != 30:
    raise RuntimeError(f"Expected 30 posterior summaries, found {len(posterior_rows)}")

print(f"Supplementary bundle: {BUNDLE_PKL}")
print(f"Direct NPL SVI root: {bundle['npl_svi_root']}")
print(f"Big Gamma/Omega SVI root: {bundle['big_svi_root']}")
print(f"Posterior animals: {len(posterior_rows)}")
print(f"Corner parameters: {PARAMS}")

fig = plt.figure(figsize=(22, 10))
left_grid = GridSpec(
    2,
    3,
    figure=fig,
    left=0.05,
    right=0.546,
    top=0.95,
    bottom=0.08,
    hspace=0.15,
    wspace=0.9,
)

psychometric_ax = fig.add_subplot(left_grid[0, 0])
plot_psychometric(psychometric_ax, bundle["psy_data"])

quantile_ax = fig.add_subplot(left_grid[0, 1])
plot_quantiles(quantile_ax, bundle["quant_data"])

gamma_ax = fig.add_subplot(left_grid[1, 0])
plot_gamma_omega(gamma_ax, bundle["gamma_omega_data"], "gamma")

omega_ax = fig.add_subplot(left_grid[1, 1])
plot_gamma_omega(omega_ax, bundle["gamma_omega_data"], "omega")

slope_ax = fig.add_subplot(left_grid[1, 2])
plot_slopes(slope_ax, bundle["slopes_data"])

fig.canvas.draw()
left_block_top = max(
    psychometric_ax.get_position().y1,
    quantile_ax.get_position().y1,
)
left_block_bottom = min(
    gamma_ax.get_position().y0,
    omega_ax.get_position().y0,
    slope_ax.get_position().y0,
)
corner_left = 0.44
corner_top = left_block_top - 0.045
corner_bottom = left_block_bottom
corner_width = (
    (corner_top - corner_bottom)
    * fig.get_size_inches()[1]
    / fig.get_size_inches()[0]
)
corner_right = corner_left + corner_width

corner_grid = GridSpec(
    len(PARAMS),
    len(PARAMS),
    figure=fig,
    left=corner_left,
    right=corner_right,
    top=corner_top,
    bottom=corner_bottom,
    hspace=0.08,
    wspace=0.08,
)
corner_axes = np.empty((len(PARAMS), len(PARAMS)), dtype=object)
for row_index in range(len(PARAMS)):
    for column_index in range(len(PARAMS)):
        corner_axes[row_index, column_index] = fig.add_subplot(
            corner_grid[row_index, column_index]
        )
plot_posterior_upper_corner(corner_axes, posterior_rows)

fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUTPUT_PDF, dpi=300, bbox_inches="tight")
print(f"Saved figure: {OUTPUT_PNG}")
print(f"Saved figure: {OUTPUT_PDF}")
