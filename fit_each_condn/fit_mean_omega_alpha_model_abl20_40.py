# %%
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from gamma_omega_alpha_utils import (
    build_cond_fit_arrays,
    gamma_omega_alpha_model,
    load_batch_animal_pairs,
    mean_and_sem_by_abl,
    print_batch_animal_table,
)


# %% Parameters
SCRIPT_DIR = os.path.dirname(__file__)
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

FIT_ABLS = [20, 40]
PLOT_ABLS = [20, 40, 60]
ILDS = np.sort([-16, -8, -4, -2, -1, 1, 2, 4, 8, 16])

P_0 = 20e-6
N_POSTERIOR_SAMPLES = int(1e5)

PARAM_NAMES = ["rate_lambda", "ell", "alpha", "theta", "T_0"]
P0_FIT = np.array([2.0, 0.9, 0.5, 2.4, 100e-3])
LOWER_BOUNDS = np.array([0.1, 0.0, 0.0, 0.5, 1e-3])
UPPER_BOUNDS = np.array([6.0, 2.0, 5.0, 15.0, 2.0])

COLORS = ["tab:blue", "tab:orange", "tab:green"]
FIG_PATH = os.path.join(SCRIPT_DIR, f"mean_gamma_omega_alpha_joint_fit_ABL20_40_{OUTPUT_TAG}.png")
FIT_RESULT_PATH = os.path.join(SCRIPT_DIR, f"mean_gamma_omega_alpha_joint_fit_ABL20_40_{OUTPUT_TAG}.npz")


# %% Load condition-fit Gamma/Omega for ABL 20, 40, and 60
print(f"Using condition-fit source: {COND_FIT_SOURCE} ({COND_FIT_LABEL})")
print(f"Pickle directory: {COND_FIT_PKL_DIR}")
print(f"Filename suffix: {COND_FIT_FILENAME_SUFFIX}")
print(f"Expected sampled params per pickle: {COND_FIT_EXPECTED_N_PARAMS}")
print(f"Fitting only ABLs: {FIT_ABLS}")
print(f"Plotting ABLs: {PLOT_ABLS}")

batch_animal_pairs = load_batch_animal_pairs(BATCH_DIR, DESIRED_BATCHES)
print_batch_animal_table(batch_animal_pairs)

gamma_cond_by_cond_fit_all_animals, omega_cond_by_cond_fit_all_animals, missing_files = build_cond_fit_arrays(
    batch_animal_pairs,
    PLOT_ABLS,
    ILDS,
    COND_FIT_PKL_DIR,
    n_samples=N_POSTERIOR_SAMPLES,
    filename_suffix=COND_FIT_FILENAME_SUFFIX,
    expected_n_params=COND_FIT_EXPECTED_N_PARAMS,
)

if len(missing_files) > 0:
    print(f"Missing condition-fit pickle files: {len(missing_files)}")


# %% Average across animals
mean_gamma_by_abl, sem_gamma_by_abl, n_gamma_by_abl = mean_and_sem_by_abl(gamma_cond_by_cond_fit_all_animals, PLOT_ABLS)
mean_omega_by_abl, sem_omega_by_abl, n_omega_by_abl = mean_and_sem_by_abl(omega_cond_by_cond_fit_all_animals, PLOT_ABLS)

fit_abls = []
fit_ilds = []
fit_mean_gamma = []
fit_mean_omega = []
for ABL in FIT_ABLS:
    valid = np.isfinite(mean_gamma_by_abl[str(ABL)]) & np.isfinite(mean_omega_by_abl[str(ABL)])
    fit_abls.extend(np.full(np.sum(valid), ABL))
    fit_ilds.extend(ILDS[valid])
    fit_mean_gamma.extend(mean_gamma_by_abl[str(ABL)][valid])
    fit_mean_omega.extend(mean_omega_by_abl[str(ABL)][valid])

fit_abls = np.asarray(fit_abls, dtype=float)
fit_ilds = np.asarray(fit_ilds, dtype=float)
fit_mean_gamma = np.asarray(fit_mean_gamma, dtype=float)
fit_mean_omega = np.asarray(fit_mean_omega, dtype=float)

gamma_scale = np.nanstd(fit_mean_gamma)
omega_scale = np.nanstd(fit_mean_omega)
if gamma_scale == 0 or not np.isfinite(gamma_scale):
    gamma_scale = 1.0
if omega_scale == 0 or not np.isfinite(omega_scale):
    omega_scale = 1.0

print(f"Fitting Gamma and Omega on {len(fit_mean_omega)} averaged ABL/ILD points.")
if len(fit_mean_omega) == 0:
    raise RuntimeError("No finite mean Gamma/Omega values found to fit.")
print(f"Residual scales: gamma={gamma_scale:.6g}, omega={omega_scale:.6g}")


# %% MSE-style fit to animal-averaged Gamma and Omega
def gamma_omega_residuals(params):
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
    gamma_residuals = (pred_gamma - fit_mean_gamma) / gamma_scale
    omega_residuals = (pred_omega - fit_mean_omega) / omega_scale
    return np.concatenate([gamma_residuals, omega_residuals])


fit_result = least_squares(
    gamma_omega_residuals,
    P0_FIT,
    bounds=(LOWER_BOUNDS, UPPER_BOUNDS),
)

best_fit_params = {name: float(value) for name, value in zip(PARAM_NAMES, fit_result.x)}
pred_gamma_fit, pred_omega_fit = gamma_omega_alpha_model(
    fit_abls,
    fit_ilds,
    best_fit_params["rate_lambda"],
    best_fit_params["ell"],
    best_fit_params["alpha"],
    best_fit_params["theta"],
    best_fit_params["T_0"],
    P_0,
)
gamma_mse = float(np.mean((pred_gamma_fit - fit_mean_gamma) ** 2))
omega_mse = float(np.mean((pred_omega_fit - fit_mean_omega) ** 2))
scaled_joint_mse = float(np.mean(gamma_omega_residuals(fit_result.x) ** 2))

print("Best-fit joint Gamma/Omega alpha model params for ABL 20 and 40:")
for name in PARAM_NAMES:
    print(f"  {name}: {best_fit_params[name]:.6g}")
print(f"  gamma_mse: {gamma_mse:.6g}")
print(f"  omega_mse: {omega_mse:.6g}")
print(f"  scaled_joint_mse: {scaled_joint_mse:.6g}")
print(f"  cost: {fit_result.cost:.6g}")
print(f"  success: {fit_result.success} ({fit_result.message})")

fitted_gamma_by_abl = {}
fitted_omega_by_abl = {}
for ABL in PLOT_ABLS:
    fitted_gamma_by_abl[str(ABL)], fitted_omega_by_abl[str(ABL)] = gamma_omega_alpha_model(
        ABL,
        ILDS,
        best_fit_params["rate_lambda"],
        best_fit_params["ell"],
        best_fit_params["alpha"],
        best_fit_params["theta"],
        best_fit_params["T_0"],
        P_0,
    )

np.savez(
    FIT_RESULT_PATH,
    param_names=np.asarray(PARAM_NAMES),
    best_fit_params=fit_result.x,
    gamma_mse=gamma_mse,
    omega_mse=omega_mse,
    scaled_joint_mse=scaled_joint_mse,
    fit_abls=fit_abls,
    fit_ilds=fit_ilds,
    fit_target_abls=np.asarray(FIT_ABLS),
    plot_abls=np.asarray(PLOT_ABLS),
    fit_mean_gamma=fit_mean_gamma,
    fit_mean_omega=fit_mean_omega,
    pred_gamma_fit=pred_gamma_fit,
    pred_omega_fit=pred_omega_fit,
)
print(f"Saved fit result data: {FIT_RESULT_PATH}")


# %% Plot mean Gamma/Omega and fitted curves
fig, ax = plt.subplots(1, 2, figsize=(11, 5))

for abl_idx, ABL in enumerate(PLOT_ABLS):
    color = COLORS[abl_idx]
    is_fit_abl = ABL in FIT_ABLS
    fit_label_suffix = "fit" if is_fit_abl else "extrap"
    fit_linestyle = "-" if is_fit_abl else "--"
    mean_alpha = 1.0 if is_fit_abl else 0.7
    ax[0].errorbar(
        ILDS,
        mean_gamma_by_abl[str(ABL)],
        yerr=sem_gamma_by_abl[str(ABL)],
        fmt="o",
        linestyle="none",
        capsize=3,
        color=color,
        alpha=mean_alpha,
        label=f"ABL={ABL} mean",
    )
    ax[0].plot(
        ILDS,
        fitted_gamma_by_abl[str(ABL)],
        linestyle=fit_linestyle,
        color=color,
        label=f"ABL={ABL} {fit_label_suffix}",
    )
    ax[1].errorbar(
        ILDS,
        mean_omega_by_abl[str(ABL)],
        yerr=sem_omega_by_abl[str(ABL)],
        fmt="o",
        linestyle="none",
        capsize=3,
        color=color,
        alpha=mean_alpha,
        label=f"ABL={ABL} mean",
    )
    ax[1].plot(
        ILDS,
        fitted_omega_by_abl[str(ABL)],
        linestyle=fit_linestyle,
        color=color,
        label=f"ABL={ABL} {fit_label_suffix}",
    )

ax[0].axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.4)
ax[0].set_title("Mean Gamma with alpha-model fit")
ax[1].set_title("Mean Omega with alpha-model fit")

for curr_ax in ax:
    curr_ax.set_xlabel("ILD")
    curr_ax.grid(True, alpha=0.25)

ax[0].set_ylabel("Gamma")
ax[1].set_ylabel("Omega")
ax[0].legend(fontsize=8)
ax[1].legend(fontsize=8)

fig.suptitle(
    f"Fit to ABL 20/40 only, ABL 60 extrapolated; {COND_FIT_LABEL}; "
    f"lambda'={best_fit_params['rate_lambda']:.3g}, ell={best_fit_params['ell']:.3g}, "
    f"alpha={best_fit_params['alpha']:.3g}, theta={best_fit_params['theta']:.3g}, "
    f"T_0={best_fit_params['T_0']*1e3:.3g} ms"
)
fig.tight_layout()
fig.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
print(f"Saved figure: {FIG_PATH}")

plt.show()
