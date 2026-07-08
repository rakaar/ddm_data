# %%
"""
Five-source per-ABL RT quantile diagnostic.

This renders only the RT-quantile part of the current patience12 diagnostics:
the three NPL+alpha parameter sources, the direct IPL/vanilla condition-delay
SVI fit, and the IPL Gamma/Omega MSE row. It uses already-computed payloads
and does not recompute fits or RTDs.
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
NPL_COMPARE_DIR = NPL_SVI_ROOT / "three_npl_param_source_comparison"
IPL_SVI_ROOT = SCRIPT_DIR / "numpyro_svi_vanilla_condition_delay_patience12_min50k_restore_best_outputs"
IPL_DIAGNOSTIC_DIR = IPL_SVI_ROOT / "fig2_like_diagnostics"

FOUR_SOURCE_PAYLOAD_PKL = Path(
    os.environ.get(
        "FIVE_SOURCE_QUANTILE_FOUR_PAYLOAD",
        str(NPL_COMPARE_DIR / "three_npl_param_sources_plus_ipl_patience12_4x5.pkl"),
    )
).expanduser()
DIRECT_IPL_PAYLOAD_PKL = Path(
    os.environ.get(
        "FIVE_SOURCE_QUANTILE_IPL_PAYLOAD",
        str(IPL_DIAGNOSTIC_DIR / "ipl_svi_50k_fig2_like_diagnostics.pkl"),
    )
).expanduser()
OUTPUT_PNG = Path(
    os.environ.get(
        "FIVE_SOURCE_QUANTILE_OUTPUT",
        str(NPL_COMPARE_DIR / "five_param_sources_patience12_quantiles_per_abl_5x3.png"),
    )
).expanduser()

METHODS = [
    {
        "key": "mse_gamma_omega",
        "payload_name": "four_source",
        "label": "NPL+alpha Gamma+Omega MSE",
        "short_label": "NPL+alpha\nG+O MSE",
    },
    {
        "key": "mse_omega_only",
        "payload_name": "four_source",
        "label": "NPL+alpha Omega-only MSE",
        "short_label": "NPL+alpha\nOmega MSE",
    },
    {
        "key": "direct_37_param_svi",
        "payload_name": "four_source",
        "label": "Direct 37-param NPL+alpha SVI",
        "short_label": "NPL+alpha\nSVI",
    },
    {
        "key": "ipl_svi_50k",
        "payload_name": "direct_ipl",
        "label": "Direct IPL SVI 50k",
        "short_label": "IPL SVI",
    },
    {
        "key": "ipl_gamma_omega",
        "payload_name": "four_source",
        "label": "IPL Gamma+Omega MSE",
        "short_label": "IPL G+O MSE",
    },
]
QUANTILES_TO_SHOW = [0.1, 0.3, 0.5, 0.7, 0.9]
EXPECTED_ABLS = [20, 40, 60]
EXPECTED_ABS_ILDS = [1.0, 2.0, 4.0, 8.0, 16.0]
EXPECTED_CONTINUOUS_ABS_ILD_COUNT = 151

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]


# %%
# =============================================================================
# Load and validate payloads
# =============================================================================
if not FOUR_SOURCE_PAYLOAD_PKL.exists():
    raise FileNotFoundError(FOUR_SOURCE_PAYLOAD_PKL)
if not DIRECT_IPL_PAYLOAD_PKL.exists():
    raise FileNotFoundError(DIRECT_IPL_PAYLOAD_PKL)

with FOUR_SOURCE_PAYLOAD_PKL.open("rb") as handle:
    four_source_payload = pickle.load(handle)
with DIRECT_IPL_PAYLOAD_PKL.open("rb") as handle:
    direct_ipl_payload = pickle.load(handle)

payloads = {
    "four_source": four_source_payload,
    "direct_ipl": direct_ipl_payload,
}


def normalized_animal_keys(raw_keys):
    return [(str(batch), int(animal)) for batch, animal in raw_keys]


canonical_animal_keys = normalized_animal_keys(four_source_payload["animal_keys"])
direct_ipl_animal_keys = normalized_animal_keys(direct_ipl_payload["animal_keys"])
if canonical_animal_keys != direct_ipl_animal_keys:
    raise RuntimeError("Direct IPL SVI animal ordering does not match the four-source payload.")

validated_quantile_data = {}
quantile_indices_by_method = {}
for method in METHODS:
    method_key = method["key"]
    source_payload = payloads[method["payload_name"]]
    quantile_by_method = source_payload.get("quantile_by_method", {})
    if method_key not in quantile_by_method:
        raise KeyError(f"Missing {method_key!r} in {method['payload_name']} payload.")

    quantile_data = quantile_by_method[method_key]
    all_quantiles = np.asarray(quantile_data["QUANTILES_TO_PLOT"], dtype=float)
    abl_arr = sorted(int(abl) for abl in quantile_data["plot_data"].keys())
    abs_ild_sorted = [float(abs_ild) for abs_ild in quantile_data["plot_data"][abl_arr[0]].keys()]
    continuous_abs_ild = [float(abs_ild) for abs_ild in quantile_data["continuous_abs_ild"]]
    method_animal_keys = normalized_animal_keys(quantile_data["animal_keys"])

    if method_animal_keys != canonical_animal_keys:
        raise RuntimeError(f"{method_key}: animal ordering does not match the canonical payload.")
    if abl_arr != EXPECTED_ABLS:
        raise RuntimeError(f"{method_key}: expected ABLs {EXPECTED_ABLS}, found {abl_arr}")
    if abs_ild_sorted != EXPECTED_ABS_ILDS:
        raise RuntimeError(
            f"{method_key}: expected |ILD| values {EXPECTED_ABS_ILDS}, found {abs_ild_sorted}"
        )
    if len(continuous_abs_ild) != EXPECTED_CONTINUOUS_ABS_ILD_COUNT:
        raise RuntimeError(
            f"{method_key}: expected {EXPECTED_CONTINUOUS_ABS_ILD_COUNT} continuous |ILD| values, "
            f"found {len(continuous_abs_ild)}"
        )

    quantile_indices = []
    for quantile in QUANTILES_TO_SHOW:
        matches = np.where(np.isclose(all_quantiles, quantile))[0]
        if len(matches) != 1:
            raise RuntimeError(
                f"{method_key}: could not find quantile {quantile} in {all_quantiles.tolist()}"
            )
        quantile_indices.append(int(matches[0]))

    validated_quantile_data[method_key] = {
        "payload": quantile_data,
        "abl_arr": abl_arr,
        "abs_ild_sorted": abs_ild_sorted,
        "continuous_abs_ild": continuous_abs_ild,
    }
    quantile_indices_by_method[method_key] = quantile_indices

print(f"Four-source payload: {FOUR_SOURCE_PAYLOAD_PKL}")
print(f"Direct IPL SVI payload: {DIRECT_IPL_PAYLOAD_PKL}")
print(f"Output: {OUTPUT_PNG}")
print(f"Methods: {[method['key'] for method in METHODS]}")
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
    print("\nFive-source quantile sample counts")
    for method in METHODS:
        method_key = method["key"]
        data = validated_quantile_data[method_key]
        quantile_data = data["payload"]
        print(f"\n{method_key}: {method['label']}")
        for abl in data["abl_arr"]:
            empirical_counts = [
                len(quantile_data["plot_data"][abl][abs_ild]["empirical"])
                for abs_ild in data["abs_ild_sorted"]
            ]
            standard_model_counts = [
                len(quantile_data["continuous_plot_data"][abl][abs_ild]["theoretical"])
                for abs_ild in data["abs_ild_sorted"]
            ]
            sd_flat_model_counts = [
                len(quantile_data["continuous_plot_data_sd_flat"][abl][abs_ild]["theoretical"])
                for abs_ild in data["abs_ild_sorted"]
            ]
            print(
                f"  ABL={abl}: data={empirical_counts}; "
                f"model={standard_model_counts}; model_sd_flat={sd_flat_model_counts}"
            )


def plot_quantiles_for_method_abl(ax, method_key, abl):
    data = validated_quantile_data[method_key]
    quantile_data = data["payload"]
    plot_data = quantile_data["plot_data"]
    theory_source = quantile_data["continuous_plot_data_sd_flat"]

    for plot_idx, (q_idx, q) in enumerate(zip(quantile_indices_by_method[method_key], QUANTILES_TO_SHOW)):
        emp_means, emp_sems = [], []
        for abs_ild in data["abs_ild_sorted"]:
            entries = plot_data[abl][abs_ild]["empirical"]
            if len(entries) > 0:
                values = np.asarray(entries, dtype=float)[:, q_idx]
            else:
                values = []
            mean, curr_sem, _n = nanmean_sem(values)
            emp_means.append(mean)
            emp_sems.append(curr_sem)

        theo_x, theo_means, theo_sems = [], [], []
        for abs_ild in data["continuous_abs_ild"]:
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
            data["abs_ild_sorted"],
            emp_means,
            yerr=emp_sems,
            fmt="o",
            color="black",
            markersize=4.0,
            capsize=0,
            alpha=0.9,
            label=f"Data q={q:.1f}" if plot_idx == 0 else "_nolegend_",
        )

        ax.plot(
            theo_x,
            theo_means,
            "-",
            color="tab:red",
            linewidth=1.1,
            label=f"Model q={q:.1f}" if plot_idx == 0 else "_nolegend_",
        )
        ax.fill_between(
            theo_x,
            np.asarray(theo_means) - np.asarray(theo_sems),
            np.asarray(theo_means) + np.asarray(theo_sems),
            color="tab:red",
            alpha=0.12,
            linewidth=0,
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(data["abs_ild_sorted"])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    ax.tick_params(axis="both", which="major", labelsize=ft.STYLE.TICK_FONTSIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.16)


# %%
# =============================================================================
# Plot and save
# =============================================================================
print_sanity_counts()

OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(
    len(METHODS),
    len(EXPECTED_ABLS),
    figsize=(15.5, 18.5),
    sharex=True,
    sharey=True,
)

for row_idx, method in enumerate(METHODS):
    method_key = method["key"]
    for col_idx, abl in enumerate(EXPECTED_ABLS):
        ax = axes[row_idx, col_idx]
        plot_quantiles_for_method_abl(ax, method_key, abl)

        if row_idx == 0:
            ax.set_title(f"ABL = {abl}", fontsize=ft.STYLE.LEGEND_FONTSIZE)
        if row_idx == len(METHODS) - 1:
            ax.set_xlabel("|ILD| (dB)", fontsize=ft.STYLE.LABEL_FONTSIZE)

    axes[row_idx, 0].text(
        -0.32,
        0.5,
        method["short_label"],
        transform=axes[row_idx, 0].transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=15,
        fontweight="bold",
    )

for ax in axes.ravel():
    ax.set_ylim(0.06, 0.7)

fig.suptitle(
    "Five patience12 parameter sources: per-ABL RT quantiles",
    fontsize=ft.STYLE.LEGEND_FONTSIZE,
)
fig.text(0.025, 0.5, "RT quantile (s)", rotation=90, va="center", ha="center", fontsize=22)
fig.subplots_adjust(left=0.13, right=0.98, bottom=0.06, top=0.94, wspace=0.12, hspace=0.22)
fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")

print(f"\nSaved figure: {OUTPUT_PNG}")

# %%
