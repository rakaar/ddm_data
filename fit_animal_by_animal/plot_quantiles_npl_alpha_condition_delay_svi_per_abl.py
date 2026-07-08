# %%
"""
Per-ABL RT quantile diagnostic for direct NPL+alpha condition-delay SVI.

This is the NPL+alpha SVI version of the older per-ABL quantile plot. It uses
the already-computed quantile payload from the three-source patience12
comparison and extracts only the direct 37-param NPL+alpha SVI method.
"""

# %%
# =============================================================================
# Parameters
# =============================================================================
from pathlib import Path
import os
import pickle

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem

import figure_template as ft


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

NPL_SVI_ROOT = SCRIPT_DIR / "numpyro_svi_npl_alpha_condition_delay_patience12_restore_best_outputs"
PAYLOAD_PKL = Path(
    os.environ.get(
        "NPL_ALPHA_SVI_QUANTILE_PAYLOAD",
        str(NPL_SVI_ROOT / "three_npl_param_source_comparison" / "three_npl_param_sources_patience12_3x5.pkl"),
    )
).expanduser()
OUTPUT_DIR = NPL_SVI_ROOT / "three_npl_param_source_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PNG = OUTPUT_DIR / "npl_alpha_condition_delay_svi_quantiles_per_abl.png"

METHOD_KEY = "direct_37_param_svi"
QUANTILES_TO_SHOW = [0.1, 0.3, 0.5, 0.7, 0.9]
EXPECTED_ABLS = [20, 40, 60]
EXPECTED_ABS_ILDS = [1.0, 2.0, 4.0, 8.0, 16.0]

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]


# %%
# =============================================================================
# Load and validate payload
# =============================================================================
if not PAYLOAD_PKL.exists():
    raise FileNotFoundError(PAYLOAD_PKL)

with PAYLOAD_PKL.open("rb") as handle:
    payload = pickle.load(handle)

if Path(payload["npl_svi_root"]).resolve() != NPL_SVI_ROOT.resolve():
    raise RuntimeError(
        "Payload NPL SVI root does not match the expected patience12 condition-delay root:\n"
        f"  payload: {payload['npl_svi_root']}\n"
        f"  expected: {NPL_SVI_ROOT}"
    )

quantile_by_method = payload.get("quantile_by_method", {})
if METHOD_KEY not in quantile_by_method:
    raise KeyError(f"Missing {METHOD_KEY!r} in {PAYLOAD_PKL}")

quantile_data = quantile_by_method[METHOD_KEY]
all_quantiles = np.asarray(quantile_data["QUANTILES_TO_PLOT"], dtype=float)
abl_arr = [int(abl) for abl in quantile_data["ABL_arr"]]
abs_ild_sorted = [float(abs_ild) for abs_ild in quantile_data["abs_ild_sorted"]]
continuous_abs_ild = [float(abs_ild) for abs_ild in quantile_data["continuous_abs_ild"]]

if abl_arr != EXPECTED_ABLS:
    raise RuntimeError(f"Expected ABLs {EXPECTED_ABLS}, found {abl_arr}")
if abs_ild_sorted != EXPECTED_ABS_ILDS:
    raise RuntimeError(f"Expected |ILD| values {EXPECTED_ABS_ILDS}, found {abs_ild_sorted}")

quantile_indices = []
for quantile in QUANTILES_TO_SHOW:
    matches = np.where(np.isclose(all_quantiles, quantile))[0]
    if len(matches) != 1:
        raise RuntimeError(f"Could not find quantile {quantile} in {all_quantiles.tolist()}")
    quantile_indices.append(int(matches[0]))

print(f"Payload: {PAYLOAD_PKL}")
print(f"NPL+alpha SVI root: {NPL_SVI_ROOT}")
print(f"Method: {METHOD_KEY}")
print(f"Quantiles shown: {QUANTILES_TO_SHOW}")


# %%
# =============================================================================
# Helpers
# =============================================================================
def nanmean_sem(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan, np.nan, 0
    if finite.size == 1:
        return float(finite[0]), np.nan, 1
    return float(np.mean(finite)), float(sem(finite, nan_policy="omit")), int(finite.size)


def print_sanity_counts():
    print("\nDirect NPL+alpha SVI quantile sample counts")
    for abl in abl_arr:
        empirical_counts = [
            len(quantile_data["plot_data"][abl][abs_ild]["empirical"])
            for abs_ild in abs_ild_sorted
        ]
        standard_model_counts = [
            len(quantile_data["continuous_plot_data"][abl][abs_ild]["theoretical"])
            for abs_ild in abs_ild_sorted
        ]
        sd_flat_model_counts = [
            len(quantile_data["continuous_plot_data_sd_flat"][abl][abs_ild]["theoretical"])
            for abs_ild in abs_ild_sorted
        ]
        print(
            f"  ABL={abl}: data={empirical_counts}; "
            f"model={standard_model_counts}; model_sd_flat={sd_flat_model_counts}"
        )


def plot_quantiles_for_abl(ax, abl):
    plot_data = quantile_data["plot_data"]
    theory_source = quantile_data["continuous_plot_data_sd_flat"]

    for plot_idx, (q_idx, q) in enumerate(zip(quantile_indices, QUANTILES_TO_SHOW)):
        emp_means, emp_sems = [], []
        for abs_ild in abs_ild_sorted:
            entries = plot_data[abl][abs_ild]["empirical"]
            if len(entries) > 0:
                values = np.asarray(entries, dtype=float)[:, q_idx]
            else:
                values = []
            mean, curr_sem, _n = nanmean_sem(values)
            emp_means.append(mean)
            emp_sems.append(curr_sem)

        theo_x, theo_means, theo_sems = [], [], []
        for abs_ild in continuous_abs_ild:
            entries = theory_source[abl][float(abs_ild)]["theoretical"]
            if len(entries) == 0:
                continue
            values = np.asarray(entries, dtype=float)[:, q_idx]
            mean, curr_sem, n = nanmean_sem(values)
            if n > 0:
                theo_x.append(float(abs_ild))
                theo_means.append(mean)
                theo_sems.append(curr_sem)

        ax.errorbar(
            abs_ild_sorted,
            emp_means,
            yerr=emp_sems,
            fmt="o",
            color="black",
            markersize=6,
            capsize=0,
            alpha=0.9,
            label=f"Data q={q:.1f}" if plot_idx == 0 else "_nolegend_",
        )

        ax.plot(
            theo_x,
            theo_means,
            "-",
            color="tab:red",
            linewidth=1.4,
            label=f"Model q={q:.1f}" if plot_idx == 0 else "_nolegend_",
        )
        ax.fill_between(
            theo_x,
            np.asarray(theo_means) - np.asarray(theo_sems),
            np.asarray(theo_means) + np.asarray(theo_sems),
            color="tab:red",
            alpha=0.16,
            linewidth=0,
        )

    ax.set_title(f"ABL = {abl}", fontsize=ft.STYLE.LEGEND_FONTSIZE)
    ax.set_xlabel("|ILD| (dB)", fontsize=ft.STYLE.LABEL_FONTSIZE)
    ax.set_xscale("log", base=2)
    ax.set_xticks(abs_ild_sorted)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.tick_params(axis="both", which="major", labelsize=ft.STYLE.TICK_FONTSIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.16)
    ax.set_box_aspect(1)


# %%
# =============================================================================
# Plot and save
# =============================================================================
print_sanity_counts()

fig, axes = plt.subplots(1, len(abl_arr), figsize=(15, 5), sharex=True, sharey=True)
for ax, abl in zip(axes, abl_arr):
    plot_quantiles_for_abl(ax, abl)

axes[0].set_ylabel("RT quantile (s)", fontsize=ft.STYLE.LABEL_FONTSIZE)
for ax in axes:
    ax.set_ylim(0.06, 0.7)

fig.suptitle(
    "Direct patience12 NPL+alpha condition-delay SVI: RT quantiles",
    fontsize=ft.STYLE.LEGEND_FONTSIZE,
)
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.8, wspace=0.24)
fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")

print(f"\nSaved figure: {OUTPUT_PNG}")

# %%
