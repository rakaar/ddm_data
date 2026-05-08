# %%
"""
Compare condition-by-condition Gamma/Omega fits with NPL+alpha model curves.

This script uses the gamma/omega-only condition fits as the empirical target,
then overlays Gamma/Omega implied by animal-wise NPL+alpha fits and two MSE
alpha-model fits.
"""
import os
import pickle

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from gamma_omega_alpha_utils import (
    build_cond_fit_arrays,
    gamma_omega_alpha_model,
    load_batch_animal_pairs,
    print_batch_animal_table,
)


# %%
# Parameters

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DESIRED_BATCHES = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]
BATCH_DIR = os.path.join(REPO_DIR, "fit_animal_by_animal", "batch_csvs")

COND_FIT_SOURCE = "gamma_omega_only"
COND_FIT_CONFIG = {
    "pkl_dir": os.path.join(SCRIPT_DIR, "each_animal_cond_fit_gama_omega_pkl_files"),
    "filename_suffix": "_FIX_t_E_w_del_go_same_as_parametric",
    "expected_n_params": 2,
    "label": "Gamma/Omega only",
}

ALPHA_FIT_DIR = os.path.join(REPO_DIR, "fit_animal_by_animal", "NPL_alpha_animal_fits")
ALPHA_RESULT_KEY = "vbmc_norm_alpha_tied_results"

ABLS = [20, 40, 60]
ILDS = np.sort([-16, -8, -4, -2, -1, 1, 2, 4, 8, 16])
SMOOTH_ILDS = np.round(np.arange(-16, 16 + 0.1, 0.1), 10)

P_0 = 20e-6
N_POSTERIOR_SAMPLES = int(1e5)

PARAM_NAMES = ["rate_lambda", "ell", "alpha", "theta", "T_0"]
P0_FIT = np.array([2.0, 0.9, 0.5, 2.4, 100e-3])
LOWER_BOUNDS = np.array([0.1, 0.0, 0.0, 0.5, 1e-3])
UPPER_BOUNDS = np.array([6.0, 2.0, 5.0, 15.0, 2.0])

COLORS = ["tab:blue", "tab:orange", "tab:green"]

NPL_ALPHA_FIG_PATH = os.path.join(SCRIPT_DIR, "cond_gamma_omega_vs_npl_alpha_model.png")
MSE_COMPARE_FIG_PATH = os.path.join(SCRIPT_DIR, "cond_gamma_omega_mse_fit_comparison.png")
ALPHA_COMPARE_FIG_PATH = os.path.join(SCRIPT_DIR, "animalwise_alpha_npl_vs_mse_fit.png")
RATE_LAMBDA_COMPARE_FIG_PATH = os.path.join(SCRIPT_DIR, "animalwise_rate_lambda_npl_vs_mse_fit.png")


# %%
# Small local helpers

def mean_sem_n(arr, axis=0):
    arr = np.asarray(arr, dtype=float)
    n = np.sum(np.isfinite(arr), axis=axis)
    mean = np.nanmean(arr, axis=axis)
    sem = np.nanstd(arr, axis=axis) / np.sqrt(n)
    sem = np.where(n > 0, sem, np.nan)
    return mean, sem, n


def alpha_result_path(batch_name, animal_id):
    return os.path.join(
        ALPHA_FIT_DIR,
        f"results_{batch_name}_animal_{animal_id}_NORM_ALPHA_FROM_ABORTS.pkl",
    )


def load_alpha_params(batch_name, animal_id):
    pkl_path = alpha_result_path(batch_name, animal_id)
    with open(pkl_path, "rb") as f:
        saved_data = pickle.load(f)

    if ALPHA_RESULT_KEY not in saved_data:
        raise KeyError(f"{pkl_path} is missing `{ALPHA_RESULT_KEY}`")

    alpha_results = saved_data[ALPHA_RESULT_KEY]
    required_keys = [
        "rate_lambda_samples",
        "rate_norm_l_samples",
        "alpha_samples",
        "theta_E_samples",
        "T_0_samples",
    ]
    missing_keys = [key for key in required_keys if key not in alpha_results]
    if missing_keys:
        raise KeyError(f"{pkl_path} is missing keys: {missing_keys}")

    return {
        "rate_lambda": float(np.mean(alpha_results["rate_lambda_samples"])),
        "ell": float(np.mean(alpha_results["rate_norm_l_samples"])),
        "alpha": float(np.mean(alpha_results["alpha_samples"])),
        "theta": float(np.mean(alpha_results["theta_E_samples"])),
        "T_0": float(np.mean(alpha_results["T_0_samples"])),
    }


def model_curves_for_params(params, ild_grid):
    gamma_by_abl = {}
    omega_by_abl = {}
    for ABL in ABLS:
        gamma_by_abl[str(ABL)], omega_by_abl[str(ABL)] = gamma_omega_alpha_model(
            ABL,
            ild_grid,
            params["rate_lambda"],
            params["ell"],
            params["alpha"],
            params["theta"],
            params["T_0"],
            P_0,
        )
    return gamma_by_abl, omega_by_abl


def finite_fit_vectors(gamma_by_abl, omega_by_abl, animal_idx=None):
    fit_abls = []
    fit_ilds = []
    fit_gamma = []
    fit_omega = []

    for ABL in ABLS:
        if animal_idx is None:
            gamma_values = gamma_by_abl[str(ABL)]
            omega_values = omega_by_abl[str(ABL)]
        else:
            gamma_values = gamma_by_abl[str(ABL)][animal_idx, :]
            omega_values = omega_by_abl[str(ABL)][animal_idx, :]

        valid = np.isfinite(gamma_values) & np.isfinite(omega_values)
        fit_abls.extend(np.full(np.sum(valid), ABL))
        fit_ilds.extend(ILDS[valid])
        fit_gamma.extend(gamma_values[valid])
        fit_omega.extend(omega_values[valid])

    return (
        np.asarray(fit_abls, dtype=float),
        np.asarray(fit_ilds, dtype=float),
        np.asarray(fit_gamma, dtype=float),
        np.asarray(fit_omega, dtype=float),
    )


def fit_gamma_omega_alpha(fit_abls, fit_ilds, target_gamma, target_omega):
    gamma_scale = np.nanstd(target_gamma)
    omega_scale = np.nanstd(target_omega)
    if gamma_scale == 0 or not np.isfinite(gamma_scale):
        gamma_scale = 1.0
    if omega_scale == 0 or not np.isfinite(omega_scale):
        omega_scale = 1.0

    def residuals(params):
        rate_lambda, ell, alpha, theta, T_0 = params
        pred_gamma, pred_omega = gamma_omega_alpha_model(
            fit_abls,
            fit_ilds,
            rate_lambda,
            ell,
            alpha,
            theta,
            T_0,
            P_0,
        )
        gamma_residuals = (pred_gamma - target_gamma) / gamma_scale
        omega_residuals = (pred_omega - target_omega) / omega_scale
        return np.concatenate([gamma_residuals, omega_residuals])

    return least_squares(
        residuals,
        P0_FIT,
        bounds=(LOWER_BOUNDS, UPPER_BOUNDS),
    )


def params_from_fit_result(fit_result):
    return {name: float(value) for name, value in zip(PARAM_NAMES, fit_result.x)}


def print_params(label, params, cost=None):
    print(label)
    for name in PARAM_NAMES:
        value = params[name]
        if name == "T_0":
            print(f"  {name}: {value * 1e3:.6g} ms")
        else:
            print(f"  {name}: {value:.6g}")
    if cost is not None:
        print(f"  cost: {cost:.6g}")


def plot_data_points(ax_gamma, ax_omega, mean_gamma_by_abl, sem_gamma_by_abl,
                     mean_omega_by_abl, sem_omega_by_abl):
    for abl_idx, ABL in enumerate(ABLS):
        color = COLORS[abl_idx]
        ax_gamma.errorbar(
            ILDS,
            mean_gamma_by_abl[str(ABL)],
            yerr=sem_gamma_by_abl[str(ABL)],
            fmt="o",
            linestyle="none",
            capsize=3,
            color=color,
        )
        ax_omega.errorbar(
            ILDS,
            mean_omega_by_abl[str(ABL)],
            yerr=sem_omega_by_abl[str(ABL)],
            fmt="o",
            linestyle="none",
            capsize=3,
            color=color,
        )


def plot_curve_with_sem(ax, x, mean_by_abl, sem_by_abl, linestyle,
                        show_sem=True):
    for abl_idx, ABL in enumerate(ABLS):
        color = COLORS[abl_idx]
        mean_values = mean_by_abl[str(ABL)]
        sem_values = sem_by_abl[str(ABL)]
        ax.plot(
            x,
            mean_values,
            color=color,
            linestyle=linestyle,
            linewidth=2,
        )
        if show_sem:
            ax.fill_between(
                x,
                mean_values - sem_values,
                mean_values + sem_values,
                color=color,
                alpha=0.08,
                linewidth=0,
            )


def format_gamma_omega_axes(ax_gamma, ax_omega, title_gamma, title_omega):
    ax_gamma.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.4)
    ax_gamma.set_title(title_gamma)
    ax_omega.set_title(title_omega)
    for curr_ax in (ax_gamma, ax_omega):
        curr_ax.set_xlabel("ILD")
        curr_ax.grid(True, alpha=0.25)
    ax_gamma.set_ylabel("Gamma")
    ax_omega.set_ylabel("Omega")


def plot_animal_param_comparison(param_name, ylabel, fig_path):
    labels = [f"{batch}-{animal}" for batch, animal in alpha_matched_pairs]
    x = np.arange(len(labels))

    npl_values = np.array(
        [alpha_params_by_animal[(batch, animal)][param_name] for batch, animal in alpha_matched_pairs],
        dtype=float,
    )
    mse_by_pair = {
        (row["batch_name"], row["animal"]): row
        for row in successful_animal_fits
    }
    mse_values = np.array(
        [
            mse_by_pair.get((batch, animal), {}).get(param_name, np.nan)
            for batch, animal in alpha_matched_pairs
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(1, 1, figsize=(max(12, 0.35 * len(labels)), 5))
    ax.scatter(x - 0.13, npl_values, color="tab:red", s=28, label="NPL+alpha")
    ax.scatter(x + 0.13, mse_values, color="tab:blue", s=28, label="MSE per-animal fit")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right")
    ax.set_xlabel("Animal")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Animal-wise {ylabel}: NPL+alpha vs MSE per-animal fit")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    # plt.close(fig)
    print(f"Saved figure: {fig_path}")


# %%
# Load shared animal list and condition-fit Gamma/Omega.

print(f"Using condition-fit source: {COND_FIT_SOURCE} ({COND_FIT_CONFIG['label']})")
print(f"Pickle directory: {COND_FIT_CONFIG['pkl_dir']}")
print(f"Filename suffix: {COND_FIT_CONFIG['filename_suffix']}")

all_batch_animal_pairs = load_batch_animal_pairs(BATCH_DIR, DESIRED_BATCHES)
alpha_matched_pairs = [
    (batch_name, int(animal_id))
    for batch_name, animal_id in all_batch_animal_pairs
    if os.path.exists(alpha_result_path(batch_name, int(animal_id)))
]

print_batch_animal_table(alpha_matched_pairs)
print(f"Matched animals with NPL+alpha fits: {len(alpha_matched_pairs)}")
if len(alpha_matched_pairs) == 0:
    raise RuntimeError("No batch-animal pairs have matching NPL+alpha fit pickles.")

gamma_cond_by_cond_fit_all_animals, omega_cond_by_cond_fit_all_animals, missing_files = build_cond_fit_arrays(
    alpha_matched_pairs,
    ABLS,
    ILDS,
    COND_FIT_CONFIG["pkl_dir"],
    n_samples=N_POSTERIOR_SAMPLES,
    filename_suffix=COND_FIT_CONFIG["filename_suffix"],
    expected_n_params=COND_FIT_CONFIG["expected_n_params"],
)

print(f"Missing condition-fit pickle files: {len(missing_files)}")


# %%
# Average condition-fit Gamma/Omega across animals.

mean_gamma_by_abl = {}
sem_gamma_by_abl = {}
n_gamma_by_abl = {}
mean_omega_by_abl = {}
sem_omega_by_abl = {}
n_omega_by_abl = {}

for ABL in ABLS:
    mean_gamma_by_abl[str(ABL)], sem_gamma_by_abl[str(ABL)], n_gamma_by_abl[str(ABL)] = mean_sem_n(
        gamma_cond_by_cond_fit_all_animals[str(ABL)], axis=0
    )
    mean_omega_by_abl[str(ABL)], sem_omega_by_abl[str(ABL)], n_omega_by_abl[str(ABL)] = mean_sem_n(
        omega_cond_by_cond_fit_all_animals[str(ABL)], axis=0
    )

fit_abls, fit_ilds, fit_mean_gamma, fit_mean_omega = finite_fit_vectors(
    mean_gamma_by_abl,
    mean_omega_by_abl,
)

print(f"Finite averaged Gamma/Omega points for MSE fit: {len(fit_mean_gamma)}")
if len(fit_mean_gamma) == 0:
    raise RuntimeError("No finite mean Gamma/Omega values found.")


# %%
# Compute NPL+alpha Gamma/Omega curves for each animal and average them.

n_animals = len(alpha_matched_pairs)
alpha_gamma_curves = {str(ABL): np.full((n_animals, len(SMOOTH_ILDS)), np.nan) for ABL in ABLS}
alpha_omega_curves = {str(ABL): np.full((n_animals, len(SMOOTH_ILDS)), np.nan) for ABL in ABLS}
alpha_params_by_animal = {}

for animal_idx, (batch_name, animal_id) in enumerate(alpha_matched_pairs):
    alpha_params = load_alpha_params(batch_name, animal_id)
    alpha_params_by_animal[(batch_name, animal_id)] = alpha_params
    gamma_by_abl, omega_by_abl = model_curves_for_params(alpha_params, SMOOTH_ILDS)
    for ABL in ABLS:
        alpha_gamma_curves[str(ABL)][animal_idx, :] = gamma_by_abl[str(ABL)]
        alpha_omega_curves[str(ABL)][animal_idx, :] = omega_by_abl[str(ABL)]

alpha_mean_gamma_by_abl = {}
alpha_sem_gamma_by_abl = {}
alpha_mean_omega_by_abl = {}
alpha_sem_omega_by_abl = {}
for ABL in ABLS:
    alpha_mean_gamma_by_abl[str(ABL)], alpha_sem_gamma_by_abl[str(ABL)], _ = mean_sem_n(
        alpha_gamma_curves[str(ABL)], axis=0
    )
    alpha_mean_omega_by_abl[str(ABL)], alpha_sem_omega_by_abl[str(ABL)], _ = mean_sem_n(
        alpha_omega_curves[str(ABL)], axis=0
    )

alpha_param_summary = {
    name: np.array([params[name] for params in alpha_params_by_animal.values()], dtype=float)
    for name in PARAM_NAMES
}
print("NPL+alpha parameter means across animals:")
for name in PARAM_NAMES:
    values = alpha_param_summary[name]
    if name == "T_0":
        print(f"  {name}: {np.nanmean(values) * 1e3:.6g} ms +/- {np.nanstd(values) / np.sqrt(len(values)) * 1e3:.6g} SEM")
    else:
        print(f"  {name}: {np.nanmean(values):.6g} +/- {np.nanstd(values) / np.sqrt(len(values)):.6g} SEM")


# %%
# Fit the alpha Gamma/Omega model to animal-averaged condition-fit values.

avg_fit_result = fit_gamma_omega_alpha(fit_abls, fit_ilds, fit_mean_gamma, fit_mean_omega)
avg_fit_params = params_from_fit_result(avg_fit_result)
print_params("Best MSE fit to animal-averaged condition Gamma/Omega:", avg_fit_params, avg_fit_result.cost)
print(f"  success: {avg_fit_result.success} ({avg_fit_result.message})")

avg_fit_gamma_by_abl, avg_fit_omega_by_abl = model_curves_for_params(avg_fit_params, SMOOTH_ILDS)


# %%
# Fit each animal separately, then average the fitted curves across animals.

animal_fit_gamma_curves = {str(ABL): np.full((n_animals, len(SMOOTH_ILDS)), np.nan) for ABL in ABLS}
animal_fit_omega_curves = {str(ABL): np.full((n_animals, len(SMOOTH_ILDS)), np.nan) for ABL in ABLS}
animal_fit_rows = []

for animal_idx, (batch_name, animal_id) in enumerate(alpha_matched_pairs):
    animal_abls, animal_ilds, animal_gamma, animal_omega = finite_fit_vectors(
        gamma_cond_by_cond_fit_all_animals,
        omega_cond_by_cond_fit_all_animals,
        animal_idx=animal_idx,
    )
    if len(animal_gamma) < len(PARAM_NAMES):
        animal_fit_rows.append(
            {
                "batch_name": batch_name,
                "animal": animal_id,
                "success": False,
                "n_points": len(animal_gamma),
                "message": "too few finite condition points",
            }
        )
        continue

    try:
        fit_result = fit_gamma_omega_alpha(animal_abls, animal_ilds, animal_gamma, animal_omega)
        fit_params = params_from_fit_result(fit_result)
        gamma_by_abl, omega_by_abl = model_curves_for_params(fit_params, SMOOTH_ILDS)
        for ABL in ABLS:
            animal_fit_gamma_curves[str(ABL)][animal_idx, :] = gamma_by_abl[str(ABL)]
            animal_fit_omega_curves[str(ABL)][animal_idx, :] = omega_by_abl[str(ABL)]
        animal_fit_rows.append(
            {
                "batch_name": batch_name,
                "animal": animal_id,
                "success": bool(fit_result.success),
                "n_points": len(animal_gamma),
                "message": fit_result.message,
                "cost": float(fit_result.cost),
                **fit_params,
            }
        )
    except Exception as exc:
        animal_fit_rows.append(
            {
                "batch_name": batch_name,
                "animal": animal_id,
                "success": False,
                "n_points": len(animal_gamma),
                "message": str(exc),
            }
        )

successful_animal_fits = [row for row in animal_fit_rows if row["success"]]
print(f"Successful per-animal MSE fits: {len(successful_animal_fits)} / {len(animal_fit_rows)}")
if len(successful_animal_fits) == 0:
    raise RuntimeError("No per-animal MSE fits succeeded.")

animal_fit_mean_gamma_by_abl = {}
animal_fit_sem_gamma_by_abl = {}
animal_fit_mean_omega_by_abl = {}
animal_fit_sem_omega_by_abl = {}
for ABL in ABLS:
    animal_fit_mean_gamma_by_abl[str(ABL)], animal_fit_sem_gamma_by_abl[str(ABL)], _ = mean_sem_n(
        animal_fit_gamma_curves[str(ABL)], axis=0
    )
    animal_fit_mean_omega_by_abl[str(ABL)], animal_fit_sem_omega_by_abl[str(ABL)], _ = mean_sem_n(
        animal_fit_omega_curves[str(ABL)], axis=0
    )

print("Per-animal MSE fit parameter means across successful fits:")
for name in PARAM_NAMES:
    values = np.array([row[name] for row in successful_animal_fits], dtype=float)
    if name == "T_0":
        print(f"  {name}: {np.nanmean(values) * 1e3:.6g} ms +/- {np.nanstd(values) / np.sqrt(len(values)) * 1e3:.6g} SEM")
    else:
        print(f"  {name}: {np.nanmean(values):.6g} +/- {np.nanstd(values) / np.sqrt(len(values)):.6g} SEM")


# %%
# Figure 1: condition data and NPL+alpha model curves.

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
plot_data_points(
    ax[0],
    ax[1],
    mean_gamma_by_abl,
    sem_gamma_by_abl,
    mean_omega_by_abl,
    sem_omega_by_abl,
)
plot_curve_with_sem(
    ax[0],
    SMOOTH_ILDS,
    alpha_mean_gamma_by_abl,
    alpha_sem_gamma_by_abl,
    "-",
)
plot_curve_with_sem(
    ax[1],
    SMOOTH_ILDS,
    alpha_mean_omega_by_abl,
    alpha_sem_omega_by_abl,
    "-",
)
format_gamma_omega_axes(
    ax[0],
    ax[1],
    "Gamma: condition fits vs NPL+alpha",
    "Omega: condition fits vs NPL+alpha",
)
fig.suptitle(
    f"{COND_FIT_CONFIG['label']}; averaged across {n_animals} NPL+alpha animals\n"
    "Source: dots = condition data, solid lines = NPL+alpha model"
)
fig.tight_layout(rect=[0, 0, 1, 0.88])
fig.savefig(NPL_ALPHA_FIG_PATH, dpi=300, bbox_inches="tight")
print(f"Saved figure: {NPL_ALPHA_FIG_PATH}")


# %%
# Figure 2: condition data and two MSE-fit summaries.

zero_sem_avg_fit_gamma = {str(ABL): np.zeros_like(avg_fit_gamma_by_abl[str(ABL)]) for ABL in ABLS}
zero_sem_avg_fit_omega = {str(ABL): np.zeros_like(avg_fit_omega_by_abl[str(ABL)]) for ABL in ABLS}

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
plot_data_points(
    ax[0],
    ax[1],
    mean_gamma_by_abl,
    sem_gamma_by_abl,
    mean_omega_by_abl,
    sem_omega_by_abl,
)
plot_curve_with_sem(
    ax[0],
    SMOOTH_ILDS,
    avg_fit_gamma_by_abl,
    zero_sem_avg_fit_gamma,
    "-",
    show_sem=False,
)
plot_curve_with_sem(
    ax[1],
    SMOOTH_ILDS,
    avg_fit_omega_by_abl,
    zero_sem_avg_fit_omega,
    "-",
    show_sem=False,
)
plot_curve_with_sem(
    ax[0],
    SMOOTH_ILDS,
    animal_fit_mean_gamma_by_abl,
    animal_fit_sem_gamma_by_abl,
    "--",
)
plot_curve_with_sem(
    ax[1],
    SMOOTH_ILDS,
    animal_fit_mean_omega_by_abl,
    animal_fit_sem_omega_by_abl,
    "--",
)
format_gamma_omega_axes(
    ax[0],
    ax[1],
    "Gamma: MSE fit comparison",
    "Omega: MSE fit comparison",
)
fig.suptitle(
    f"{COND_FIT_CONFIG['label']}; mean-fit vs per-animal-fit average\n"
    "Source: dots = condition data, solid lines = MSE fit to mean, dashed lines = per-animal MSE average"
)
fig.tight_layout(rect=[0, 0, 1, 0.88])
fig.savefig(MSE_COMPARE_FIG_PATH, dpi=300, bbox_inches="tight")
print(f"Saved figure: {MSE_COMPARE_FIG_PATH}")


# %%
# Figure 3/4: compare per-animal parameter values from NPL+alpha and MSE fits.

plot_animal_param_comparison("alpha", "alpha", ALPHA_COMPARE_FIG_PATH)
plot_animal_param_comparison("rate_lambda", "rate_lambda", RATE_LAMBDA_COMPARE_FIG_PATH)

# %%
