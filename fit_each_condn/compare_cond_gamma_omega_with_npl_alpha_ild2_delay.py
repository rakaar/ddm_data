# %%
"""
Compare condition-by-condition Gamma/Omega fits with NPL+alpha+ILD2-delay fits.

The ILD2-delay model changes the fitted animal-wise parameters, but Gamma/Omega
are still implied by the NPL+alpha firing-rate expression. This script overlays
those implied curves on the condition-by-condition Gamma/Omega posterior means.
"""
import os
import pickle

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
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

COND_FIT_SOURCE = "gamma_omega_t_E_aff_del_go_fix_w_mean"
COND_FIT_SOURCES = {
    "gamma_omega_only": {
        "pkl_dir": os.path.join(SCRIPT_DIR, "each_animal_cond_fit_gama_omega_pkl_files"),
        "filename_suffix": "_FIX_t_E_w_del_go_same_as_parametric",
        "expected_n_params": 2,
        "label": "Gamma/Omega only",
        "output_tag": "gamma_omega_only",
    },
    "gamma_omega_t_E_aff_w_del_go": {
        "pkl_dir": os.path.join(SCRIPT_DIR, "each_animal_cond_fit_5_params_pkl_files"),
        "filename_suffix": "_5_params",
        "expected_n_params": 5,
        "label": "Gamma/Omega + t_E_aff/w/del_go",
        "output_tag": "gamma_omega_5_params",
    },
    "gamma_omega_t_E_aff_fix_w_del_go": {
        "pkl_dir": SCRIPT_DIR,
        "filename_suffix": "_FIX_w_pt_5_del_go",
        "expected_n_params": 3,
        "label": "Gamma/Omega + t_E_aff; fixed w/del_go",
        "output_tag": "gamma_omega_t_E_aff_fix_w_del_go",
    },
    "gamma_omega_t_E_aff_del_go_fix_w_mean": {
        "pkl_dir": os.path.join(SCRIPT_DIR, "each_animal_cond_fit_4_params_fix_w_mean_pkl_files"),
        "filename_suffix": "_FIX_w_mean_4_params",
        "expected_n_params": 4,
        "label": "Gamma/Omega + t_E_aff/del_go; fixed animal mean w",
        "output_tag": "gamma_omega_t_E_aff_del_go_fix_w_mean",
    },
}
COND_FIT_CONFIG = COND_FIT_SOURCES[COND_FIT_SOURCE]
COND_FIT_PKL_DIR = COND_FIT_CONFIG["pkl_dir"]
COND_FIT_FILENAME_SUFFIX = COND_FIT_CONFIG["filename_suffix"]
COND_FIT_EXPECTED_N_PARAMS = COND_FIT_CONFIG["expected_n_params"]
COND_FIT_LABEL = COND_FIT_CONFIG["label"]
OUTPUT_TAG = COND_FIT_CONFIG["output_tag"]

MODEL_RESULTS_DIR = os.path.join(REPO_DIR, "NPL_alpha_ILD2_fit_results", "result_pkls")
MODEL_RESULT_KEY = "vbmc_norm_alpha_ild2_delay_tied_results"
MODEL_LABEL = "NPL + alpha + ILD2 delay"

OUTPUT_DIR = os.path.join(REPO_DIR, "NPL_alpha_ILD2_fit_results", "gamma_omega_comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ABLS = [20, 40, 60]
ILDS = np.sort([-16, -8, -4, -2, -1, 1, 2, 4, 8, 16])
SMOOTH_ILDS = np.round(np.arange(-16, 16 + 0.1, 0.1), 10)

P_0 = 20e-6
N_POSTERIOR_SAMPLES = int(1e5)

PARAM_SAMPLES_TO_MEAN = {
    "rate_lambda": "rate_lambda_samples",
    "ell": "rate_norm_l_samples",
    "alpha": "alpha_samples",
    "theta": "theta_E_samples",
    "T_0": "T_0_samples",
}
PARAM_NAMES = ["rate_lambda", "ell", "alpha", "theta", "T_0"]
P0_FIT = np.array([2.0, 0.9, 0.5, 2.4, 100e-3])
LOWER_BOUNDS = np.array([0.1, 0.0, 0.0, 0.5, 1e-3])
UPPER_BOUNDS = np.array([6.0, 2.0, 5.0, 15.0, 2.0])

COLORS = ["tab:blue", "tab:orange", "tab:green"]

FIG_PATH = os.path.join(OUTPUT_DIR, f"cond_gamma_omega_vs_npl_alpha_ild2_delay_model_{OUTPUT_TAG}.png")
CSV_PATH = os.path.join(OUTPUT_DIR, f"cond_gamma_omega_vs_npl_alpha_ild2_delay_metrics_{OUTPUT_TAG}.csv")
PARAM_CSV_PATH = os.path.join(OUTPUT_DIR, f"npl_alpha_ild2_delay_gamma_omega_params_by_animal_{OUTPUT_TAG}.csv")
MSE_PARAM_CSV_PATH = os.path.join(OUTPUT_DIR, f"per_animal_mse_gamma_omega_alpha_params_{OUTPUT_TAG}.csv")


# %%
# Helpers

def mean_sem_n(arr, axis=0):
    arr = np.asarray(arr, dtype=float)
    n = np.sum(np.isfinite(arr), axis=axis)
    mean = np.nanmean(arr, axis=axis)
    sem = np.nanstd(arr, axis=axis) / np.sqrt(n)
    sem = np.where(n > 0, sem, np.nan)
    return mean, sem, n


def model_result_path(batch_name, animal_id):
    return os.path.join(
        MODEL_RESULTS_DIR,
        f"results_{batch_name}_animal_{animal_id}_NORM_ALPHA_ILD2_DELAY_FROM_ABORTS.pkl",
    )


def load_model_params(batch_name, animal_id):
    pkl_path = model_result_path(batch_name, animal_id)
    with open(pkl_path, "rb") as f:
        saved_data = pickle.load(f)

    if MODEL_RESULT_KEY not in saved_data:
        raise KeyError(f"{pkl_path} is missing `{MODEL_RESULT_KEY}`")

    model_results = saved_data[MODEL_RESULT_KEY]
    missing_keys = [
        sample_key
        for sample_key in PARAM_SAMPLES_TO_MEAN.values()
        if sample_key not in model_results
    ]
    if missing_keys:
        raise KeyError(f"{pkl_path} is missing keys: {missing_keys}")

    return {
        param_name: float(np.mean(np.asarray(model_results[sample_key], dtype=float)))
        for param_name, sample_key in PARAM_SAMPLES_TO_MEAN.items()
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


def finite_fit_vectors(gamma_by_abl, omega_by_abl, animal_idx):
    fit_abls = []
    fit_ilds = []
    fit_gamma = []
    fit_omega = []

    for ABL in ABLS:
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


def plot_curve_with_sem(ax, x, mean_by_abl, sem_by_abl, linestyle="-", alpha=0.10):
    for abl_idx, ABL in enumerate(ABLS):
        color = COLORS[abl_idx]
        mean_values = mean_by_abl[str(ABL)]
        sem_values = sem_by_abl[str(ABL)]
        ax.plot(x, mean_values, color=color, linestyle=linestyle, linewidth=2)
        ax.fill_between(
            x,
            mean_values - sem_values,
            mean_values + sem_values,
            color=color,
            alpha=alpha,
            linewidth=0,
        )


# %%
# Load shared animal list and condition-by-condition Gamma/Omega.

print(f"Using condition-fit source: {COND_FIT_SOURCE} ({COND_FIT_LABEL})")
print(f"Pickle directory: {COND_FIT_PKL_DIR}")
print(f"Filename suffix: {COND_FIT_FILENAME_SUFFIX}")
print(f"Expected sampled params per pickle: {COND_FIT_EXPECTED_N_PARAMS}")
print(f"Model result directory: {MODEL_RESULTS_DIR}")

all_batch_animal_pairs = load_batch_animal_pairs(BATCH_DIR, DESIRED_BATCHES)
matched_pairs = [
    (batch_name, int(animal_id))
    for batch_name, animal_id in all_batch_animal_pairs
    if os.path.exists(model_result_path(batch_name, int(animal_id)))
]

print_batch_animal_table(matched_pairs)
print(f"Matched animals with {MODEL_LABEL} fits: {len(matched_pairs)}")
if len(matched_pairs) == 0:
    raise RuntimeError(f"No batch-animal pairs have matching result pickles in {MODEL_RESULTS_DIR}")

gamma_cond_by_abl, omega_cond_by_abl, missing_files = build_cond_fit_arrays(
    matched_pairs,
    ABLS,
    ILDS,
    COND_FIT_PKL_DIR,
    n_samples=N_POSTERIOR_SAMPLES,
    filename_suffix=COND_FIT_FILENAME_SUFFIX,
    expected_n_params=COND_FIT_EXPECTED_N_PARAMS,
)

print(f"Missing condition-fit pickle files: {len(missing_files)}")


# %%
# Average condition-fit Gamma/Omega across matched animals.

mean_gamma_by_abl = {}
sem_gamma_by_abl = {}
n_gamma_by_abl = {}
mean_omega_by_abl = {}
sem_omega_by_abl = {}
n_omega_by_abl = {}

for ABL in ABLS:
    mean_gamma_by_abl[str(ABL)], sem_gamma_by_abl[str(ABL)], n_gamma_by_abl[str(ABL)] = mean_sem_n(
        gamma_cond_by_abl[str(ABL)], axis=0
    )
    mean_omega_by_abl[str(ABL)], sem_omega_by_abl[str(ABL)], n_omega_by_abl[str(ABL)] = mean_sem_n(
        omega_cond_by_abl[str(ABL)], axis=0
    )


# %%
# Compute animal-wise model-implied Gamma/Omega curves and average them.

n_animals = len(matched_pairs)
model_gamma_curves = {str(ABL): np.full((n_animals, len(SMOOTH_ILDS)), np.nan) for ABL in ABLS}
model_omega_curves = {str(ABL): np.full((n_animals, len(SMOOTH_ILDS)), np.nan) for ABL in ABLS}
model_gamma_at_conditions = {str(ABL): np.full((n_animals, len(ILDS)), np.nan) for ABL in ABLS}
model_omega_at_conditions = {str(ABL): np.full((n_animals, len(ILDS)), np.nan) for ABL in ABLS}
param_rows = []

for animal_idx, (batch_name, animal_id) in enumerate(matched_pairs):
    model_params = load_model_params(batch_name, animal_id)
    param_rows.append({"batch_name": batch_name, "animal": animal_id, **model_params})

    smooth_gamma_by_abl, smooth_omega_by_abl = model_curves_for_params(model_params, SMOOTH_ILDS)
    cond_gamma_by_abl, cond_omega_by_abl = model_curves_for_params(model_params, ILDS)

    for ABL in ABLS:
        model_gamma_curves[str(ABL)][animal_idx, :] = smooth_gamma_by_abl[str(ABL)]
        model_omega_curves[str(ABL)][animal_idx, :] = smooth_omega_by_abl[str(ABL)]
        model_gamma_at_conditions[str(ABL)][animal_idx, :] = cond_gamma_by_abl[str(ABL)]
        model_omega_at_conditions[str(ABL)][animal_idx, :] = cond_omega_by_abl[str(ABL)]

param_df = pd.DataFrame(param_rows)
param_df.to_csv(PARAM_CSV_PATH, index=False)
print(f"Saved parameter summary: {PARAM_CSV_PATH}")

model_mean_gamma_by_abl = {}
model_sem_gamma_by_abl = {}
model_mean_omega_by_abl = {}
model_sem_omega_by_abl = {}
model_cond_mean_gamma_by_abl = {}
model_cond_mean_omega_by_abl = {}

for ABL in ABLS:
    model_mean_gamma_by_abl[str(ABL)], model_sem_gamma_by_abl[str(ABL)], _ = mean_sem_n(
        model_gamma_curves[str(ABL)], axis=0
    )
    model_mean_omega_by_abl[str(ABL)], model_sem_omega_by_abl[str(ABL)], _ = mean_sem_n(
        model_omega_curves[str(ABL)], axis=0
    )
    model_cond_mean_gamma_by_abl[str(ABL)], _, _ = mean_sem_n(
        model_gamma_at_conditions[str(ABL)], axis=0
    )
    model_cond_mean_omega_by_abl[str(ABL)], _, _ = mean_sem_n(
        model_omega_at_conditions[str(ABL)], axis=0
    )

print(f"{MODEL_LABEL} parameter means across animals:")
for param_name in ["rate_lambda", "ell", "alpha", "theta", "T_0"]:
    values = param_df[param_name].to_numpy(dtype=float)
    if param_name == "T_0":
        print(
            f"  {param_name}: {np.nanmean(values) * 1e3:.6g} ms "
            f"+/- {np.nanstd(values) / np.sqrt(len(values)) * 1e3:.6g} SEM"
        )
    else:
        print(
            f"  {param_name}: {np.nanmean(values):.6g} "
            f"+/- {np.nanstd(values) / np.sqrt(len(values)):.6g} SEM"
        )


# %%
# Fit the same Gamma/Omega alpha model to each animal's condition-fit Gamma/Omega,
# then average those fitted curves across animals.

mse_gamma_curves = {str(ABL): np.full((n_animals, len(SMOOTH_ILDS)), np.nan) for ABL in ABLS}
mse_omega_curves = {str(ABL): np.full((n_animals, len(SMOOTH_ILDS)), np.nan) for ABL in ABLS}
mse_fit_rows = []

for animal_idx, (batch_name, animal_id) in enumerate(matched_pairs):
    animal_abls, animal_ilds, animal_gamma, animal_omega = finite_fit_vectors(
        gamma_cond_by_abl,
        omega_cond_by_abl,
        animal_idx=animal_idx,
    )
    if len(animal_gamma) < len(PARAM_NAMES):
        mse_fit_rows.append(
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
            mse_gamma_curves[str(ABL)][animal_idx, :] = gamma_by_abl[str(ABL)]
            mse_omega_curves[str(ABL)][animal_idx, :] = omega_by_abl[str(ABL)]
        mse_fit_rows.append(
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
        mse_fit_rows.append(
            {
                "batch_name": batch_name,
                "animal": animal_id,
                "success": False,
                "n_points": len(animal_gamma),
                "message": str(exc),
            }
        )

mse_fit_df = pd.DataFrame(mse_fit_rows)
mse_fit_df.to_csv(MSE_PARAM_CSV_PATH, index=False)
print(f"Saved per-animal MSE fit summary: {MSE_PARAM_CSV_PATH}")

successful_mse_fits = [row for row in mse_fit_rows if row["success"]]
print(f"Successful per-animal MSE fits: {len(successful_mse_fits)} / {len(mse_fit_rows)}")
if len(successful_mse_fits) == 0:
    raise RuntimeError("No per-animal MSE fits succeeded.")

mse_mean_gamma_by_abl = {}
mse_sem_gamma_by_abl = {}
mse_mean_omega_by_abl = {}
mse_sem_omega_by_abl = {}

for ABL in ABLS:
    mse_mean_gamma_by_abl[str(ABL)], mse_sem_gamma_by_abl[str(ABL)], _ = mean_sem_n(
        mse_gamma_curves[str(ABL)], axis=0
    )
    mse_mean_omega_by_abl[str(ABL)], mse_sem_omega_by_abl[str(ABL)], _ = mean_sem_n(
        mse_omega_curves[str(ABL)], axis=0
    )

print("Per-animal MSE fit parameter means across successful fits:")
for param_name in PARAM_NAMES:
    values = np.asarray([row[param_name] for row in successful_mse_fits], dtype=float)
    if param_name == "T_0":
        print(
            f"  {param_name}: {np.nanmean(values) * 1e3:.6g} ms "
            f"+/- {np.nanstd(values) / np.sqrt(len(values)) * 1e3:.6g} SEM"
        )
    else:
        print(
            f"  {param_name}: {np.nanmean(values):.6g} "
            f"+/- {np.nanstd(values) / np.sqrt(len(values)):.6g} SEM"
        )


# %%
# Save per-condition mean comparison metrics.

metric_rows = []
for ABL in ABLS:
    for ild_idx, ILD in enumerate(ILDS):
        cond_gamma = mean_gamma_by_abl[str(ABL)][ild_idx]
        cond_omega = mean_omega_by_abl[str(ABL)][ild_idx]
        model_gamma = model_cond_mean_gamma_by_abl[str(ABL)][ild_idx]
        model_omega = model_cond_mean_omega_by_abl[str(ABL)][ild_idx]
        metric_rows.append(
            {
                "ABL": ABL,
                "ILD": ILD,
                "condition_gamma_mean": cond_gamma,
                "model_gamma_mean": model_gamma,
                "gamma_diff_model_minus_condition": model_gamma - cond_gamma,
                "n_condition_gamma": n_gamma_by_abl[str(ABL)][ild_idx],
                "condition_omega_mean": cond_omega,
                "model_omega_mean": model_omega,
                "omega_diff_model_minus_condition": model_omega - cond_omega,
                "n_condition_omega": n_omega_by_abl[str(ABL)][ild_idx],
            }
        )

metrics_df = pd.DataFrame(metric_rows)
metrics_df.to_csv(CSV_PATH, index=False)

gamma_rmse = np.sqrt(np.nanmean(metrics_df["gamma_diff_model_minus_condition"] ** 2))
omega_rmse = np.sqrt(np.nanmean(metrics_df["omega_diff_model_minus_condition"] ** 2))
print(f"Saved comparison metrics: {CSV_PATH}")
print(f"Mean-condition RMSE: gamma={gamma_rmse:.6g}, omega={omega_rmse:.6g}")


# %%
# Plot condition-fit points and model-implied curves.

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

for abl_idx, ABL in enumerate(ABLS):
    color = COLORS[abl_idx]
    ax[0].errorbar(
        ILDS,
        mean_gamma_by_abl[str(ABL)],
        yerr=sem_gamma_by_abl[str(ABL)],
        fmt="o",
        linestyle="none",
        capsize=3,
        color=color,
        label=f"ABL={ABL}",
    )
    ax[1].errorbar(
        ILDS,
        mean_omega_by_abl[str(ABL)],
        yerr=sem_omega_by_abl[str(ABL)],
        fmt="o",
        linestyle="none",
        capsize=3,
        color=color,
        label=f"ABL={ABL}",
    )

plot_curve_with_sem(ax[0], SMOOTH_ILDS, model_mean_gamma_by_abl, model_sem_gamma_by_abl)
plot_curve_with_sem(ax[1], SMOOTH_ILDS, model_mean_omega_by_abl, model_sem_omega_by_abl)
plot_curve_with_sem(ax[0], SMOOTH_ILDS, mse_mean_gamma_by_abl, mse_sem_gamma_by_abl, linestyle="--", alpha=0.07)
plot_curve_with_sem(ax[1], SMOOTH_ILDS, mse_mean_omega_by_abl, mse_sem_omega_by_abl, linestyle="--", alpha=0.07)

ax[0].axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.4)
ax[0].set_title("Gamma")
ax[1].set_title("Omega")

for curr_ax in ax:
    curr_ax.set_xlabel("ILD")
    curr_ax.grid(True, alpha=0.25)

ax[0].set_ylabel("Gamma")
ax[1].set_ylabel("Omega")

abl_handles = [
    Line2D([0], [0], marker="o", color=color, linestyle="none", label=f"ABL={ABL}")
    for ABL, color in zip(ABLS, COLORS)
]
source_handles = [
    Line2D([0], [0], marker="o", color="black", linestyle="none", label="condition fit mean +/- SEM"),
    Line2D([0], [0], color="black", linestyle="-", linewidth=2, label=f"{MODEL_LABEL} mean +/- SEM"),
    Line2D([0], [0], color="black", linestyle="--", linewidth=2, label="per-animal MSE fit mean +/- SEM"),
]
ax[0].legend(handles=abl_handles + source_handles, fontsize=8)

fig.suptitle(
    f"{COND_FIT_LABEL}; averaged across {n_animals} matched animals\n"
    f"solid={MODEL_LABEL}; dashed=per-animal MSE fit; "
    f"RMSE at condition grid: gamma={gamma_rmse:.3g}, omega={omega_rmse:.3g}"
)
fig.tight_layout(rect=[0, 0, 1, 0.88])
fig.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
print(f"Saved figure: {FIG_PATH}")


# %%
