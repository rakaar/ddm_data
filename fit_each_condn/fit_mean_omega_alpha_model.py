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
COND_FIT_PKL_DIR = os.path.join(SCRIPT_DIR, "each_animal_cond_fit_gama_omega_pkl_files")

ABLS = [20, 40, 60]
# ILDS = np.sort([-16, -8, -4, -2, -1, 1, 2, 4, 8, 16])
# ILDS = np.sort([ -8, -4, -2, -1, 1, 2, 4, 8])
ILDS = np.sort([ -4, -2, -1, 1, 2, 4])

P_0 = 20e-6
N_POSTERIOR_SAMPLES = int(1e5)

PARAM_NAMES = ["rate_lambda", "ell", "alpha", "theta", "T_0"]
P0_FIT = np.array([2.0, 0.9, 0.5, 2.4, 100e-3])
LOWER_BOUNDS = np.array([0.1, 0.0, 0.0, 0.5, 1e-3])
UPPER_BOUNDS = np.array([6.0, 2.0, 5.0, 15.0, 2.0])

COLORS = ["tab:blue", "tab:orange", "tab:green"]
FIG_PATH = os.path.join(SCRIPT_DIR, "mean_gamma_omega_alpha_joint_fit.png")
ELL_SWEEP_FIG_PATH = os.path.join(SCRIPT_DIR, "abl60_omega_ell_sweep.png")


# %% Load condition-fit Gamma/Omega for each animal
batch_animal_pairs = load_batch_animal_pairs(BATCH_DIR, DESIRED_BATCHES)
print_batch_animal_table(batch_animal_pairs)

gamma_cond_by_cond_fit_all_animals, omega_cond_by_cond_fit_all_animals, missing_files = build_cond_fit_arrays(
    batch_animal_pairs,
    ABLS,
    ILDS,
    COND_FIT_PKL_DIR,
    n_samples=N_POSTERIOR_SAMPLES,
)

if len(missing_files) > 0:
    print(f"Missing condition-fit pickle files: {len(missing_files)}")


# %% Average across animals
mean_gamma_by_abl, sem_gamma_by_abl, n_gamma_by_abl = mean_and_sem_by_abl(gamma_cond_by_cond_fit_all_animals, ABLS)
mean_omega_by_abl, sem_omega_by_abl, n_omega_by_abl = mean_and_sem_by_abl(omega_cond_by_cond_fit_all_animals, ABLS)

fit_abls = []
fit_ilds = []
fit_mean_gamma = []
fit_mean_omega = []
for ABL in ABLS:
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


# %% Fit new expression to animal-averaged Gamma and Omega
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
print("Best-fit joint Gamma/Omega alpha model params:")
for name in PARAM_NAMES:
    print(f"  {name}: {best_fit_params[name]:.6g}")
print(f"  cost: {fit_result.cost:.6g}")
print(f"  success: {fit_result.success} ({fit_result.message})")

fitted_gamma_by_abl = {}
fitted_omega_by_abl = {}
for ABL in ABLS:
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


# %% Plot mean Gamma/Omega and fitted curves
fig, ax = plt.subplots(1, 2, figsize=(11, 5))

for abl_idx, ABL in enumerate(ABLS):
    color = COLORS[abl_idx]
    ax[0].plot(
        ILDS,
        mean_gamma_by_abl[str(ABL)],
        marker="o",
        linestyle="none",
        color=color,
        label=f"ABL={ABL} mean",
    )
    ax[0].plot(
        ILDS,
        fitted_gamma_by_abl[str(ABL)],
        linestyle="-",
        color=color,
        label=f"ABL={ABL} fit",
    )
    ax[1].plot(
        ILDS,
        mean_omega_by_abl[str(ABL)],
        marker="o",
        linestyle="none",
        color=color,
        label=f"ABL={ABL} mean",
    )
    ax[1].plot(
        ILDS,
        fitted_omega_by_abl[str(ABL)],
        linestyle="-",
        color=color,
        label=f"ABL={ABL} fit",
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
# ax[1].legend(fontsize=8)

fig.suptitle(
    f"lambda'={best_fit_params['rate_lambda']:.3g}, ell={best_fit_params['ell']:.3g}, "
    f"alpha={best_fit_params['alpha']:.3g}, theta={best_fit_params['theta']:.3g}, "
    f"T_0={best_fit_params['T_0']*1e3:.3g} ms"
)
fig.tight_layout()
fig.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
print(f"Saved figure: {FIG_PATH}")


plt.show()

# %%
# %% Plot ABL 60 Omega while sweeping ell
ABL_TO_SWEEP = 60
ELL_SWEEP = [0.9, 0.92, 0.99]

ild_smooth = np.linspace(np.min(ILDS), np.max(ILDS), 200)

fig, ax = plt.subplots(1, 1, figsize=(6, 5))
ax.plot(
    ILDS,
    mean_omega_by_abl[str(ABL_TO_SWEEP)],
    marker="o",
    linestyle="none",
    color="black",
    label=f"ABL={ABL_TO_SWEEP} mean",
)

for ell_value in ELL_SWEEP:
    _, omega_ell = gamma_omega_alpha_model(
        ABL_TO_SWEEP,
        ild_smooth,
        best_fit_params["rate_lambda"],
        ell_value,
        best_fit_params["alpha"],
        best_fit_params["theta"],
        best_fit_params["T_0"],
        P_0,
    )
    ax.plot(ild_smooth, omega_ell, label=f"ell={ell_value}")

ax.set_title(f"ABL {ABL_TO_SWEEP} Omega for different ell values")
ax.set_xlabel("ILD")
ax.set_ylabel("Omega")
ax.grid(True, alpha=0.25)
ax.legend()

fig.suptitle(
    f"Fixed params: lambda'={best_fit_params['rate_lambda']:.3g}, "
    f"alpha={best_fit_params['alpha']:.3g}, theta={best_fit_params['theta']:.3g}, "
    f"T_0={best_fit_params['T_0']*1e3:.3g} ms"
)
fig.tight_layout()
fig.savefig(ELL_SWEEP_FIG_PATH, dpi=300, bbox_inches="tight")
print(f"Saved figure: {ELL_SWEEP_FIG_PATH}")

plt.show()

# %%
