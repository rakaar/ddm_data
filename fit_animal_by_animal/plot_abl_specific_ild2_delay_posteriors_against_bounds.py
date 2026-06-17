# %%
"""
Overlay animal-wise posteriors for ABL-specific ILD2 delay coefficients.

Each subplot shows one histogram per animal with histtype="step". The x-axis is
the hard VBMC bound used for that coefficient, so this is a quick visual check
for whether the upstream NPL + alpha + ABL-specific ILD2 delay posteriors are
piling up near the fitted bounds.
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
from matplotlib.lines import Line2D


# %%
# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

UPSTREAM_DIR = REPO_DIR / "all_30_NPL_alpha_ABL_specific_ILD2_delay_fit_results"
OUTPUT_DIR = SCRIPT_DIR / "abl_specific_ild2_delay_posterior_bounds"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_PNG = OUTPUT_DIR / "abl_specific_ild2_delay_posteriors_against_vbmc_bounds.png"

MODEL_KEY = "vbmc_norm_alpha_abl_specific_ild2_delay_tied_results"
EXPECTED_N_ANIMALS = 30
ABLS = [20, 40, 60]

DELAY_PARAMS = [
    {
        "key": "bias_ms_by_abl_samples",
        "bounds_key": "bias_ms",
        "label": "bias (ms)",
        "short_label": "bias",
        "bounds": (10.0, 200.0),
        "plausible": (50.0, 150.0),
        "bin_width": 2.5,
    },
    {
        "key": "abs_ild_delay_coeff_ms_per_unit_by_abl_samples",
        "bounds_key": "abs_ild_delay_coeff_ms_per_unit",
        "label": "|ILD| coeff (ms/unit)",
        "short_label": "|ILD| coeff",
        "bounds": (-2.0, 0.5),
        "plausible": (-1.8, 0.3),
        "bin_width": 0.025,
    },
    {
        "key": "ild2_delay_coeff_ms_per_unit2_by_abl_samples",
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
EDGE_FRACTION_OF_RANGE = 0.01


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


def load_delay_posterior_samples():
    result_paths = sorted(UPSTREAM_DIR.glob("results_*_NORM_ALPHA_ABL_SPECIFIC_ILD2_DELAY_FROM_ABORTS.pkl"))
    if len(result_paths) != EXPECTED_N_ANIMALS:
        raise RuntimeError(f"Expected {EXPECTED_N_ANIMALS} upstream result pkls, found {len(result_paths)}")

    animal_results = []
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

        animal_results.append(
            {
                "batch_name": batch_name,
                "animal": animal_id,
                "samples": {
                    param["key"]: np.asarray(result[param["key"]], dtype=float)[::PLOT_SAMPLE_STRIDE]
                    for param in DELAY_PARAMS
                },
                "source_pkl": str(result_path),
            }
        )

    return animal_results, fit_config_bounds


def edge_mass_percent(samples, bounds):
    lower, upper = bounds
    edge_width = EDGE_FRACTION_OF_RANGE * (upper - lower)
    lower_mass = 100.0 * np.mean(samples <= lower + edge_width)
    upper_mass = 100.0 * np.mean(samples >= upper - edge_width)
    return lower_mass, upper_mass


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


# %%
# =============================================================================
# Plot
# =============================================================================
print(f"Upstream folder: {UPSTREAM_DIR}")
print(f"Output figure: {FIG_PNG}")

animal_results, fit_config_bounds = load_delay_posterior_samples()
print(f"Loaded stable upstream fits for {len(animal_results)} animals")
if fit_config_bounds is not None:
    print(f"Delay coefficient bounds from saved fit_config: {fit_config_bounds}")
    apply_saved_fit_config_bounds(fit_config_bounds)

fig, axes = plt.subplots(len(ABLS), len(DELAY_PARAMS), figsize=(14, 9), sharey=False)

print("\nMaximum per-animal edge mass within 1% of each hard bound:")
for row_idx, abl in enumerate(ABLS):
    for col_idx, param in enumerate(DELAY_PARAMS):
        ax = axes[row_idx, col_idx]
        lower, upper = param["bounds"]
        plausible_lower, plausible_upper = param["plausible"]
        bin_width = param["bin_width"]
        bins = np.arange(lower, upper + 0.5 * bin_width, bin_width)
        if bins[-1] < upper:
            bins = np.append(bins, upper)
        else:
            bins[-1] = upper

        max_lower_mass = 0.0
        max_upper_mass = 0.0
        max_lower_animal = None
        max_upper_animal = None

        for animal_result in animal_results:
            batch_name = animal_result["batch_name"]
            animal_id = animal_result["animal"]
            color = BATCH_COLORS.get(batch_name, "0.35")
            samples = animal_result["samples"][param["key"]][:, row_idx]

            lower_mass, upper_mass = edge_mass_percent(samples, param["bounds"])
            if lower_mass > max_lower_mass:
                max_lower_mass = lower_mass
                max_lower_animal = f"{batch_name}/{animal_id}"
            if upper_mass > max_upper_mass:
                max_upper_mass = upper_mass
                max_upper_animal = f"{batch_name}/{animal_id}"

            ax.hist(
                samples,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=0.7,
                alpha=0.42,
                color=color,
            )

        ax.axvline(lower, color="black", linestyle="-", linewidth=1.0)
        ax.axvline(upper, color="black", linestyle="-", linewidth=1.0)
        ax.axvline(plausible_lower, color="0.35", linestyle=":", linewidth=1.0)
        ax.axvline(plausible_upper, color="0.35", linestyle=":", linewidth=1.0)
        ax.set_xlim(lower, upper)
        ax.grid(True, alpha=0.18)
        ax.tick_params(labelsize=8)

        if row_idx == 0:
            ax.set_title(param["label"], fontsize=10)
        if col_idx == 0:
            ax.set_ylabel(f"ABL={abl}\ndensity", fontsize=10)
        if row_idx == len(ABLS) - 1:
            ax.set_xlabel(param["label"], fontsize=9)

        ax.text(
            0.03,
            0.92,
            f"max edge: L {max_lower_mass:.2f}% / U {max_upper_mass:.2f}%",
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )

        print(
            f"  ABL={abl}, {param['short_label']}: "
            f"lower max {max_lower_mass:.3f}% ({max_lower_animal}), "
            f"upper max {max_upper_mass:.3f}% ({max_upper_animal})"
        )

batch_handles = [
    Line2D([0], [0], color=color, linewidth=1.8, label=batch_name)
    for batch_name, color in BATCH_COLORS.items()
]
bound_handles = [
    Line2D([0], [0], color="black", linewidth=1.2, label="hard VBMC bound"),
    Line2D([0], [0], color="0.35", linestyle=":", linewidth=1.2, label="plausible bound"),
]
fig.legend(
    handles=batch_handles + bound_handles,
    loc="upper center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, 0.935),
    fontsize=9,
)
fig.suptitle(
    "Animal-wise NPL + alpha + ABL-specific ILD2 delay posteriors against VBMC bounds\n"
    "Each stepped histogram is one animal; x-axes span the hard VBMC bounds.",
    fontsize=12,
    y=0.99,
)
fig.tight_layout(rect=[0, 0, 1, 0.86])
fig.savefig(FIG_PNG, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved figure: {FIG_PNG}")

# %%
