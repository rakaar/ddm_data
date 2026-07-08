# %%
"""
Plot the SVI-based lapses supplementary figure v2.

Run `build_svi_lapses_supp_v2_data.py` first. This script intentionally keeps
the visual style close to the old `lapses_figure_using_template.py` 2 x 4
figure while reading the compact SVI data bundle from this folder.
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

SCRIPT_DIR = Path(__file__).resolve().parent
ANIMAL_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
PLOT_DATA_PKL = OUTPUT_DIR / "svi_lapses_supp_v2_plot_data.pkl"
FIG_PNG = OUTPUT_DIR / "svi_lapses_supp_v2_2x4.png"
FIG_PDF = OUTPUT_DIR / "svi_lapses_supp_v2_2x4.pdf"

sys.path.insert(0, str(ANIMAL_DIR))
import figure_template as ft

PARAM_ORDER = ["rate_norm_l", "rate_lambda", "theta_E", "T_0"]
PARAM_LABELS = {
    "rate_norm_l": r"$\ell$",
    "rate_lambda": r"$\lambda'$",
    "theta_E": r"$\theta_E$",
    "T_0": r"$T_0$ (s)",
}
PARAM_TICKS = {
    "rate_lambda": [1.5, 2.5],
}


# %%
# =============================================================================
# Plot helpers
# =============================================================================
def set_paper_axes(ax):
    ax.tick_params(axis="both", labelsize=ft.STYLE.TICK_FONTSIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_box_aspect(1)


def finite_ylim_with_zero(values, pad_fraction=0.08):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -1.0, 1.0
    ymin = min(float(np.min(values)), 0.0)
    ymax = max(float(np.max(values)), 0.0)
    if ymin == ymax:
        pad = max(abs(ymin) * 0.1, 1.0)
    else:
        pad = pad_fraction * (ymax - ymin)
    return ymin - pad, ymax + pad


def plot_lapse_distribution(ax, lapse_df, median_lapse_rate_pct):
    lapse_rates_sorted = np.sort(lapse_df["avg_lapse_rate_pct"].to_numpy(dtype=float))
    animal_indices = np.arange(1, len(lapse_rates_sorted) + 1)

    ax.scatter(animal_indices, lapse_rates_sorted, color="black", s=50, alpha=0.7)
    ax.axhline(
        median_lapse_rate_pct,
        color="gray",
        linestyle="--",
        linewidth=2,
        label=f"Median={median_lapse_rate_pct:.2f}%",
    )

    ax.set_xlabel("Animal", fontsize=ft.STYLE.TICK_FONTSIZE)
    ax.set_ylabel("Lapse Rate (%)", fontsize=ft.STYLE.TICK_FONTSIZE)
    ax.legend(fontsize=ft.STYLE.LEGEND_FONTSIZE, frameon=False, loc="best")
    ax.set_xticks([])
    ax.set_ylim(0, 25)
    ax.set_yticks([0, 25])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _p: f"{x:.10g}"))
    set_paper_axes(ax)


def plot_ll_diff(ax, ll_diff_df, y_key, ylabel):
    x = ll_diff_df["avg_lapse_rate_pct"].to_numpy(dtype=float)
    y = ll_diff_df[y_key].to_numpy(dtype=float)
    colors = ["green" if value > 0 else "red" for value in y]
    ax.scatter(x, y, c=colors, alpha=0.7, s=45, edgecolors="black", linewidth=0.5)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.axvline(x=float(np.median(x)), color="black", linestyle=":", linewidth=1)

    ax.set_xlabel("Lapse Rate (%)", fontsize=ft.STYLE.TICK_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=ft.STYLE.TICK_FONTSIZE)
    ax.set_xticks([0, 6])
    ax.set_ylim(*finite_ylim_with_zero(y))
    set_paper_axes(ax)


def plot_gamma_big_svi(ax, gamma_summary_df):
    style = {
        "Lapse": {"color": "red"},
        "No lapse": {"color": "black"},
    }
    for model_name, s in style.items():
        sub = gamma_summary_df[gamma_summary_df["model"] == model_name].sort_values("ILD")
        ax.errorbar(
            sub["ILD"].to_numpy(dtype=float),
            sub["mean"].to_numpy(dtype=float),
            yerr=sub["sem"].to_numpy(dtype=float),
            fmt="o",
            color=s["color"],
            ecolor=s["color"],
            capsize=0,
            markersize=8,
        )

    ax.set_xlabel("ILD", fontsize=ft.STYLE.LABEL_FONTSIZE)
    ax.set_ylabel(r"$\Gamma$", fontsize=ft.STYLE.LABEL_FONTSIZE)
    ax.set_xticks([-15, -5, 5, 15])
    ax.set_yticks([-2, 0, 2])
    set_paper_axes(ax)


def plot_param_comparison(ax, param_df, param_name, median_lapse_x):
    style = {
        "NPL": {"color": "black", "marker": "o", "offset": -0.10},
        "NPL_L": {"color": "red", "marker": "s", "offset": 0.10},
    }
    for model_name, s in style.items():
        sub = param_df[(param_df["parameter"] == param_name) & (param_df["model"] == model_name)].sort_values("x_pos")
        means = sub["mean"].to_numpy(dtype=float)
        q025 = sub["q025"].to_numpy(dtype=float)
        q975 = sub["q975"].to_numpy(dtype=float)
        x_pos = sub["x_pos"].to_numpy(dtype=float) + s["offset"]
        ax.errorbar(
            x_pos,
            means,
            yerr=[means - q025, q975 - means],
            fmt=s["marker"],
            color=s["color"],
            alpha=0.7,
            capsize=0,
            markersize=6,
            linewidth=1.5,
            label=model_name,
        )

    ax.axvline(median_lapse_x, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Rat", fontsize=ft.STYLE.LABEL_FONTSIZE)
    ax.set_ylabel(PARAM_LABELS[param_name], fontsize=ft.STYLE.LABEL_FONTSIZE)
    ax.set_xticks([])
    if param_name in PARAM_TICKS:
        ax.set_yticks(PARAM_TICKS[param_name])
    set_paper_axes(ax)


# %%
# =============================================================================
# Load data
# =============================================================================
if not PLOT_DATA_PKL.exists():
    raise FileNotFoundError(
        f"{PLOT_DATA_PKL} does not exist. Run build_svi_lapses_supp_v2_data.py first."
    )

with PLOT_DATA_PKL.open("rb") as handle:
    plot_data = pickle.load(handle)

lapse_df = plot_data["lapse_df"]
ll_diff_df = plot_data["ll_diff_df"]
param_df = plot_data["param_df"]
gamma_summary_df = plot_data["gamma_summary_df"]
median_lapse_rate_pct = float(plot_data["median_lapse_rate_pct"])
median_lapse_x = float(plot_data["median_lapse_x"])

print(f"Loaded plot data: {PLOT_DATA_PKL}")
print(f"Median average lapse rate: {median_lapse_rate_pct:.3f}%")


# %%
# =============================================================================
# Assemble 2 x 4 figure
# =============================================================================
builder = ft.FigureBuilder(
    sup_title="",
    n_rows=2,
    n_cols=4,
    figsize=(18, 9),
    hspace=0.4,
    wspace=0.8,
)

ax1 = builder.fig.add_subplot(builder.gs[0, 0])
plot_lapse_distribution(ax1, lapse_df, median_lapse_rate_pct)

ax2 = builder.fig.add_subplot(builder.gs[0, 1])
plot_ll_diff(ax2, ll_diff_df, "npl_minus_ipl_lapse", r"$\Delta$LL (NPL $-$ IPL$_L$)")
ax2.set_yticks([-300, 0, 300])

ax3 = builder.fig.add_subplot(builder.gs[0, 2])
plot_ll_diff(ax3, ll_diff_df, "npl_lapse_minus_ipl_lapse", r"$\Delta$LL (NPL$_L$ $-$ IPL$_L$)")
ax3.set_yticks([0, 300])

ax4 = builder.fig.add_subplot(builder.gs[0, 3])
plot_gamma_big_svi(ax4, gamma_summary_df)

bottom_axes = []
for col_idx, param_name in enumerate(PARAM_ORDER):
    ax = builder.fig.add_subplot(builder.gs[1, col_idx])
    plot_param_comparison(ax, param_df, param_name, median_lapse_x)
    bottom_axes.append(ax)

ft.shift_axes([ax4, bottom_axes[-1]], dx=-0.03)

fig = builder.finish()
fig.savefig(FIG_PNG, dpi=300, bbox_inches="tight")
fig.savefig(FIG_PDF, format="pdf", bbox_inches="tight")

print("\nFigure saved to:")
print(f"  {FIG_PNG}")
print(f"  {FIG_PDF}")
print("Done.")

# %%
