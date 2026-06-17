# %%
"""
Compare animal-wise VBMC delay posteriors with unconstrained MSE delay coefficients.

Each subplot overlays:
- one stepped posterior histogram per animal from the NPL + alpha +
  ABL-specific ILD2 delay fit;
- one dashed vertical line per animal at that animal's unconstrained MSE delay
  coefficient fit from condition-by-condition t_E_aff values.

The delay function is:

    delay_ms = bias_ms + abs_ild_coeff_ms_per_unit * |ILD| + ild2_coeff_ms_per_unit2 * ILD^2
"""

# %%
# =============================================================================
# Imports
# =============================================================================
import os
import pickle
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.optimize import least_squares


# %%
# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

UPSTREAM_DIR = REPO_DIR / "all_30_NPL_alpha_ABL_specific_ILD2_delay_fit_results"
COND_CACHE_CSV = (
    REPO_DIR
    / "fit_each_condn"
    / "abl_specific_ild2_delay_agreement"
    / "condition_t_E_aff_extraction_cache.csv"
)

OUTPUT_DIR = SCRIPT_DIR / "abl_specific_ild2_delay_posterior_bounds"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_PNG = OUTPUT_DIR / "abl_specific_ild2_delay_posteriors_with_mse_coefficients.png"

MODEL_KEY = "vbmc_norm_alpha_abl_specific_ild2_delay_tied_results"
EXPECTED_N_ANIMALS = 30
EXPECTED_N_COND_FITS = 864
ABLS = [20, 40, 60]

DELAY_PARAMS = [
    {
        "sample_key": "bias_ms_by_abl_samples",
        "mse_col": "bias_ms",
        "bounds_key": "bias_ms",
        "label": "bias (ms)",
        "short_label": "bias",
        "bounds": (10.0, 200.0),
        "plausible": (50.0, 150.0),
        "bin_width": 2.5,
    },
    {
        "sample_key": "abs_ild_delay_coeff_ms_per_unit_by_abl_samples",
        "mse_col": "abs_ild_coeff_ms_per_unit",
        "bounds_key": "abs_ild_delay_coeff_ms_per_unit",
        "label": "|ILD| coeff (ms/unit)",
        "short_label": "|ILD| coeff",
        "bounds": (-2.0, 0.5),
        "plausible": (-1.8, 0.3),
        "bin_width": 0.025,
    },
    {
        "sample_key": "ild2_delay_coeff_ms_per_unit2_by_abl_samples",
        "mse_col": "ild2_coeff_ms_per_unit2",
        "bounds_key": "ild2_delay_coeff_ms_per_unit2",
        "label": "ILD^2 coeff (ms/unit^2)",
        "short_label": "ILD^2 coeff",
        "bounds": (-1.0, 0.1),
        "plausible": (-0.1, 0.05),
        "bin_width": 0.01,
    },
]

BATCH_COLORS = {
    "LED34": "tab:blue",
    "LED34_even": "tab:cyan",
    "LED6": "tab:orange",
    "LED7": "tab:green",
    "LED8": "tab:red",
    "SD": "tab:purple",
}

PLOT_SAMPLE_STRIDE = int(os.environ.get("PLOT_SAMPLE_STRIDE", "1"))


# %%
# =============================================================================
# Helpers
# =============================================================================
def parse_upstream_result_name(path):
    match = re.match(
        r"^results_(?P<batch>.+)_animal_(?P<animal>\d+)_NORM_ALPHA_ABL_SPECIFIC_ILD2_DELAY_FROM_ABORTS\.pkl$",
        path.name,
    )
    if match is None:
        return None
    return match.group("batch"), int(match.group("animal"))


def delay_curve_ms(params, ild_values):
    bias_ms, abs_ild_coeff, ild2_coeff = np.asarray(params, dtype=float)
    ild_values = np.asarray(ild_values, dtype=float)
    return bias_ms + abs_ild_coeff * np.abs(ild_values) + ild2_coeff * (ild_values ** 2)


def apply_saved_fit_config_bounds(fit_config_bounds):
    if not fit_config_bounds:
        return
    for param in DELAY_PARAMS:
        saved_bounds = fit_config_bounds.get(param["bounds_key"])
        if not isinstance(saved_bounds, dict):
            continue
        if "hard" in saved_bounds:
            param["bounds"] = tuple(float(x) for x in saved_bounds["hard"])
        if "plausible" in saved_bounds:
            param["plausible"] = tuple(float(x) for x in saved_bounds["plausible"])


def load_vbmc_delay_posteriors():
    result_paths = sorted(UPSTREAM_DIR.glob("results_*_NORM_ALPHA_ABL_SPECIFIC_ILD2_DELAY_FROM_ABORTS.pkl"))
    if len(result_paths) != EXPECTED_N_ANIMALS:
        raise RuntimeError(f"Expected {EXPECTED_N_ANIMALS} upstream result pkls, found {len(result_paths)}")

    animal_results = []
    npl_coeff_rows = []
    fit_config_bounds = None

    for result_path in result_paths:
        parsed = parse_upstream_result_name(result_path)
        if parsed is None:
            raise RuntimeError(f"Could not parse upstream result name: {result_path.name}")
        batch_name, animal_id = parsed

        with result_path.open("rb") as f:
            saved = pickle.load(f)
        if MODEL_KEY not in saved:
            raise KeyError(f"{result_path} is missing `{MODEL_KEY}`")

        result = saved[MODEL_KEY]
        message = str(result.get("message", ""))
        if "stable" not in message.lower():
            raise RuntimeError(f"Upstream fit is not stable for {batch_name}/{animal_id}: {message}")

        abl_levels = np.asarray(result["delay_abl_levels"], dtype=float)
        if len(abl_levels) != len(ABLS) or not np.allclose(abl_levels, ABLS):
            raise RuntimeError(f"{batch_name}/{animal_id} has delay_abl_levels={abl_levels}, expected {ABLS}")

        fit_config = saved.get("fit_config", {})
        if fit_config_bounds is None and isinstance(fit_config, dict):
            fit_config_bounds = fit_config.get("delay_coefficient_bounds")

        animal_samples = {
            param["sample_key"]: np.asarray(result[param["sample_key"]], dtype=float)[::PLOT_SAMPLE_STRIDE]
            for param in DELAY_PARAMS
        }
        animal_results.append(
            {
                "batch_name": batch_name,
                "animal": animal_id,
                "samples": animal_samples,
                "source_pkl": str(result_path),
            }
        )

        bias_mean = np.mean(np.asarray(result["bias_ms_by_abl_samples"], dtype=float), axis=0)
        abs_mean = np.mean(np.asarray(result["abs_ild_delay_coeff_ms_per_unit_by_abl_samples"], dtype=float), axis=0)
        ild2_mean = np.mean(np.asarray(result["ild2_delay_coeff_ms_per_unit2_by_abl_samples"], dtype=float), axis=0)
        for abl_idx, abl in enumerate(ABLS):
            npl_coeff_rows.append(
                {
                    "batch_name": batch_name,
                    "animal": animal_id,
                    "ABL": abl,
                    "bias_ms": float(bias_mean[abl_idx]),
                    "abs_ild_coeff_ms_per_unit": float(abs_mean[abl_idx]),
                    "ild2_coeff_ms_per_unit2": float(ild2_mean[abl_idx]),
                }
            )

    return animal_results, pd.DataFrame(npl_coeff_rows), fit_config_bounds


def fit_unconstrained_mse_coefficients(cond_df, npl_coeff_df):
    coeff_lookup = {
        (row.batch_name, int(row.animal), int(row.ABL)): np.array(
            [row.bias_ms, row.abs_ild_coeff_ms_per_unit, row.ild2_coeff_ms_per_unit2],
            dtype=float,
        )
        for row in npl_coeff_df.itertuples(index=False)
    }

    fit_rows = []
    for (batch_name, animal_id, abl), group in cond_df.groupby(["batch_name", "animal", "ABL"], sort=True):
        x0 = coeff_lookup[(batch_name, int(animal_id), int(abl))]
        ild_values = group["ILD"].to_numpy(dtype=float)
        target_ms = group["t_E_aff_ms"].to_numpy(dtype=float)
        finite = np.isfinite(ild_values) & np.isfinite(target_ms)
        ild_values = ild_values[finite]
        target_ms = target_ms[finite]
        if len(target_ms) < 3:
            raise RuntimeError(f"Too few points for MSE delay fit: {batch_name}/{animal_id}, ABL={abl}")

        def residuals(params):
            return delay_curve_ms(params, ild_values) - target_ms

        fit_result = least_squares(residuals, x0)
        params = fit_result.x
        fitted_values = delay_curve_ms(params, ild_values)
        fit_rows.append(
            {
                "batch_name": batch_name,
                "animal": int(animal_id),
                "ABL": int(abl),
                "bias_ms": float(params[0]),
                "abs_ild_coeff_ms_per_unit": float(params[1]),
                "ild2_coeff_ms_per_unit2": float(params[2]),
                "rmse_ms": float(np.sqrt(np.mean((fitted_values - target_ms) ** 2))),
                "success": bool(fit_result.success),
                "message": fit_result.message,
            }
        )

    return pd.DataFrame(fit_rows)


def make_bins(x_min, x_max, bin_width):
    bins = np.arange(x_min, x_max + 0.5 * bin_width, bin_width)
    if len(bins) < 2:
        bins = np.array([x_min, x_max], dtype=float)
    if bins[-1] < x_max:
        bins = np.append(bins, x_max)
    else:
        bins[-1] = x_max
    return bins


# %%
# =============================================================================
# Main analysis
# =============================================================================
print(f"Upstream folder: {UPSTREAM_DIR}")
print(f"Condition cache: {COND_CACHE_CSV}")
print(f"Output figure: {FIG_PNG}")

if not COND_CACHE_CSV.exists():
    raise FileNotFoundError(f"Missing condition t_E_aff cache: {COND_CACHE_CSV}")

animal_results, npl_coeff_df, fit_config_bounds = load_vbmc_delay_posteriors()
if fit_config_bounds is not None:
    print(f"Delay coefficient bounds from saved fit_config: {fit_config_bounds}")
    apply_saved_fit_config_bounds(fit_config_bounds)
print(f"Loaded stable upstream fits for {len(animal_results)} animals")

cond_df = pd.read_csv(COND_CACHE_CSV)
if len(cond_df) != EXPECTED_N_COND_FITS:
    raise RuntimeError(f"Expected {EXPECTED_N_COND_FITS} condition rows, found {len(cond_df)}")
print(f"Loaded condition rows: {len(cond_df)}")

mse_fit_df = fit_unconstrained_mse_coefficients(cond_df, npl_coeff_df)
if len(mse_fit_df) != EXPECTED_N_ANIMALS * len(ABLS):
    raise RuntimeError(f"Expected {EXPECTED_N_ANIMALS * len(ABLS)} MSE coefficient rows, found {len(mse_fit_df)}")
print(f"MSE delay fits completed: {len(mse_fit_df)} animal-ABL fits")
print(f"MSE fit success count: {int(mse_fit_df['success'].sum())}/{len(mse_fit_df)}")

print("\nMSE coefficients outside hard VBMC bounds:")
for abl in ABLS:
    abl_mse = mse_fit_df[mse_fit_df["ABL"] == abl]
    for param in DELAY_PARAMS:
        lower, upper = param["bounds"]
        values = abl_mse[param["mse_col"]].to_numpy(dtype=float)
        n_low = int(np.sum(values < lower))
        n_high = int(np.sum(values > upper))
        print(
            f"  ABL={abl}, {param['short_label']}: "
            f"{n_low} below / {n_high} above; range [{np.min(values):.4g}, {np.max(values):.4g}]"
        )

fig, axes = plt.subplots(len(ABLS), len(DELAY_PARAMS), figsize=(15, 9), sharey=False)

for row_idx, abl in enumerate(ABLS):
    for col_idx, param in enumerate(DELAY_PARAMS):
        ax = axes[row_idx, col_idx]
        hard_lower, hard_upper = param["bounds"]
        plausible_lower, plausible_upper = param["plausible"]
        mse_subset = mse_fit_df[mse_fit_df["ABL"] == abl].copy()
        mse_values = mse_subset[param["mse_col"]].to_numpy(dtype=float)

        x_min = min(hard_lower, float(np.nanmin(mse_values)))
        x_max = max(hard_upper, float(np.nanmax(mse_values)))
        pad = 0.04 * (x_max - x_min)
        x_min -= pad
        x_max += pad
        bins = make_bins(x_min, x_max, param["bin_width"])

        for animal_result in animal_results:
            batch_name = animal_result["batch_name"]
            animal_id = animal_result["animal"]
            color = BATCH_COLORS.get(batch_name, "0.35")
            posterior_samples = animal_result["samples"][param["sample_key"]][:, row_idx]

            ax.hist(
                posterior_samples,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=0.75,
                alpha=0.42,
                color=color,
            )

            mse_row = mse_subset[
                (mse_subset["batch_name"] == batch_name)
                & (mse_subset["animal"] == animal_id)
            ]
            if len(mse_row) != 1:
                raise RuntimeError(f"Could not find one MSE row for {batch_name}/{animal_id}, ABL={abl}")
            ax.axvline(
                float(mse_row.iloc[0][param["mse_col"]]),
                color=color,
                linestyle="--",
                linewidth=0.7,
                alpha=0.48,
            )

        ax.axvspan(hard_lower, hard_upper, color="0.85", alpha=0.12, linewidth=0)
        ax.axvline(hard_lower, color="black", linestyle="-", linewidth=1.1)
        ax.axvline(hard_upper, color="black", linestyle="-", linewidth=1.1)
        ax.axvline(plausible_lower, color="0.35", linestyle=":", linewidth=1.0)
        ax.axvline(plausible_upper, color="0.35", linestyle=":", linewidth=1.0)
        ax.set_xlim(x_min, x_max)
        ax.grid(True, alpha=0.18)
        ax.tick_params(labelsize=8)

        n_low = int(np.sum(mse_values < hard_lower))
        n_high = int(np.sum(mse_values > hard_upper))
        ax.text(
            0.03,
            0.92,
            f"MSE outside: {n_low} low / {n_high} high",
            transform=ax.transAxes,
            fontsize=7.5,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
        )

        if row_idx == 0:
            ax.set_title(param["label"], fontsize=10)
        if col_idx == 0:
            ax.set_ylabel(f"ABL={abl}\ndensity", fontsize=10)
        if row_idx == len(ABLS) - 1:
            ax.set_xlabel(param["label"], fontsize=9)

batch_handles = [
    Line2D([0], [0], color=color, linewidth=1.8, label=batch_name)
    for batch_name, color in BATCH_COLORS.items()
]
method_handles = [
    Line2D([0], [0], color="black", linewidth=1.4, label="hard VBMC bound"),
    Line2D([0], [0], color="0.35", linestyle=":", linewidth=1.4, label="plausible bound"),
    Line2D([0], [0], color="black", linestyle="-", linewidth=1.4, alpha=0.35, label="VBMC posterior hist"),
    Line2D([0], [0], color="black", linestyle="--", linewidth=1.4, alpha=0.65, label="animal MSE coefficient"),
]
fig.legend(
    handles=batch_handles + method_handles,
    loc="upper center",
    ncol=5,
    frameon=False,
    bbox_to_anchor=(0.5, 0.94),
    fontsize=8.8,
)
fig.suptitle(
    "Animal-wise VBMC delay posteriors with unconstrained MSE coefficients\n"
    "Stepped histograms are VBMC posterior samples; dashed vertical lines are the same animal's MSE coefficient.",
    fontsize=12,
    y=0.99,
)
fig.tight_layout(rect=[0, 0, 1, 0.85])
fig.savefig(FIG_PNG, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved figure: {FIG_PNG}")

# %%
