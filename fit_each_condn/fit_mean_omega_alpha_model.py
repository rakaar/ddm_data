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
ILDS = np.sort([-16, -8, -4, -2, -1, 1, 2, 4, 8, 16])
# ILDS = np.sort([ -8, -4, -2, -1, 1, 2, 4, 8])
# ILDS = np.sort([ -4, -2, -1, 1, 2, 4])

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

# %% Compare analytical Gamma/Omega formulas with firing-rate implementation
COMPARE_RATE_LAMBDA = 2.75641
COMPARE_ELL = 0.943883
COMPARE_ALPHA = 0.498281
COMPARE_THETA = 2.55864
COMPARE_T_0 = 0.0986694

ANALYTICAL_COMPARE_FIG_PATH = os.path.join(SCRIPT_DIR, "analytical_vs_firing_rate_gamma_omega.png")

chi = 40 / np.log(10)
# ild_smooth = np.linspace(np.min(ILDS), np.max(ILDS), 100)
ild_smooth = np.arange(-40, 40, 0.2)
fig, ax = plt.subplots(1, 2, figsize=(11, 5))

print("Analytical vs firing-rate Gamma/Omega max absolute differences:")
for abl_idx, ABL in enumerate(ABLS):
    color = COLORS[abl_idx]

    firing_gamma_smooth, firing_omega_smooth = gamma_omega_alpha_model(
        ABL,
        ild_smooth,
        COMPARE_RATE_LAMBDA,
        COMPARE_ELL,
        COMPARE_ALPHA,
        COMPARE_THETA,
        COMPARE_T_0,
        P_0,
    )

    LD = ild_smooth
    lambda_ld_over_chi = COMPARE_RATE_LAMBDA * LD / chi
    lambda_ell_ld_over_chi = COMPARE_RATE_LAMBDA * COMPARE_ELL * LD / chi
    lambda_ell_plus_one_ld_over_chi = COMPARE_RATE_LAMBDA * (COMPARE_ELL + 1) * LD / chi
    alpha_minus_one_half = (COMPARE_ALPHA - 1) / 2

    analytical_gamma_smooth = COMPARE_THETA * (
        np.sinh(lambda_ld_over_chi)
        + alpha_minus_one_half
        * (np.sinh(lambda_ell_plus_one_ld_over_chi) / np.cosh(lambda_ell_ld_over_chi))
    ) / (
        np.cosh(lambda_ld_over_chi)
        + alpha_minus_one_half
        * (np.cosh(lambda_ell_plus_one_ld_over_chi) / np.cosh(lambda_ell_ld_over_chi))
    )

    f_omega_smooth = (
        np.cosh(lambda_ld_over_chi) * np.cosh(lambda_ell_ld_over_chi)
        + alpha_minus_one_half * np.cosh(lambda_ell_plus_one_ld_over_chi)
    ) / (
        COMPARE_ALPHA * (np.cosh(lambda_ell_ld_over_chi) ** 2)
        + (alpha_minus_one_half**2)
    )
    analytical_omega_smooth = (
        (1 / (COMPARE_T_0 * COMPARE_THETA**2))
        * (10 ** (COMPARE_RATE_LAMBDA * (1 - COMPARE_ELL) * ABL / 20))
        * f_omega_smooth
    )

    firing_gamma_ild, firing_omega_ild = gamma_omega_alpha_model(
        ABL,
        ILDS,
        COMPARE_RATE_LAMBDA,
        COMPARE_ELL,
        COMPARE_ALPHA,
        COMPARE_THETA,
        COMPARE_T_0,
        P_0,
    )

    LD = ILDS
    lambda_ld_over_chi = COMPARE_RATE_LAMBDA * LD / chi
    lambda_ell_ld_over_chi = COMPARE_RATE_LAMBDA * COMPARE_ELL * LD / chi
    lambda_ell_plus_one_ld_over_chi = COMPARE_RATE_LAMBDA * (COMPARE_ELL + 1) * LD / chi

    analytical_gamma_ild = COMPARE_THETA * (
        np.sinh(lambda_ld_over_chi)
        + alpha_minus_one_half
        * (np.sinh(lambda_ell_plus_one_ld_over_chi) / np.cosh(lambda_ell_ld_over_chi))
    ) / (
        np.cosh(lambda_ld_over_chi)
        + alpha_minus_one_half
        * (np.cosh(lambda_ell_plus_one_ld_over_chi) / np.cosh(lambda_ell_ld_over_chi))
    )

    f_omega_ild = (
        np.cosh(lambda_ld_over_chi) * np.cosh(lambda_ell_ld_over_chi)
        + alpha_minus_one_half * np.cosh(lambda_ell_plus_one_ld_over_chi)
    ) / (
        COMPARE_ALPHA * (np.cosh(lambda_ell_ld_over_chi) ** 2)
        + (alpha_minus_one_half**2)
    )
    analytical_omega_ild = (
        (1 / (COMPARE_T_0 * COMPARE_THETA**2))
        * (10 ** (COMPARE_RATE_LAMBDA * (1 - COMPARE_ELL) * ABL / 20))
        * f_omega_ild
    )

    max_gamma_diff = np.max(np.abs(firing_gamma_ild - analytical_gamma_ild))
    max_omega_diff = np.max(np.abs(firing_omega_ild - analytical_omega_ild))
    print(f"  ABL={ABL}: gamma={max_gamma_diff:.12g}, omega={max_omega_diff:.12g}")

    ax[0].scatter(
        ild_smooth,
        firing_gamma_smooth,
        linestyle="--",
        color='red',
        alpha = 0.2
    )
    ax[0].plot(
        ild_smooth,
        analytical_gamma_smooth,
        color='black',
        lw=2
    )
    ax[1].scatter(
        ild_smooth,
        firing_omega_smooth,
        linestyle="--",
        color='red',
        alpha = 0.2

    )
    ax[1].plot(
        ild_smooth,
        analytical_omega_smooth,
        color='black',
        lw=2
    )

ax[0].axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.4)
ax[0].set_title("Gamma: firing rates vs analytical")
ax[1].set_title("Omega: firing rates vs analytical")

for curr_ax in ax:
    curr_ax.set_xlabel("ILD")
    curr_ax.grid(True, alpha=0.25)

ax[0].set_ylabel("Gamma")
ax[1].set_ylabel("Omega")
ax[0].legend(fontsize=8)
ax[1].legend(fontsize=8)

fig.suptitle(
    f"lambda'={COMPARE_RATE_LAMBDA:.6g}, ell={COMPARE_ELL:.6g}, "
    f"alpha={COMPARE_ALPHA:.6g}, theta={COMPARE_THETA:.6g}, "
    f"T_0={COMPARE_T_0*1e3:.6g} ms"
)
fig.tight_layout()
fig.savefig(ANALYTICAL_COMPARE_FIG_PATH, dpi=300, bbox_inches="tight")
print(f"Saved figure: {ANALYTICAL_COMPARE_FIG_PATH}")

plt.show()

# %%

# %% Sweep alpha at ILD=+16 to inspect omega numerator/denominator saturation
ALPHA_SWEEP_ABL = 40
ALPHA_SWEEP_ILD = 16
ALPHA_SWEEP_VALUES = np.arange(0.1, 3.0 + 0.1, 0.1)
ALPHA_SWEEP_FIG_PATH = os.path.join(SCRIPT_DIR, "omega_alpha_sweep_ild_16.png")

LD = ALPHA_SWEEP_ILD
lambda_ld_over_chi = COMPARE_RATE_LAMBDA * LD / chi
lambda_ell_ld_over_chi = COMPARE_RATE_LAMBDA * COMPARE_ELL * LD / chi
lambda_ell_plus_one_ld_over_chi = COMPARE_RATE_LAMBDA * (COMPARE_ELL + 1) * LD / chi
alpha_minus_one_half = (ALPHA_SWEEP_VALUES - 1) / 2

omega_numerator = (
    np.cosh(lambda_ld_over_chi) * np.cosh(lambda_ell_ld_over_chi)
    + alpha_minus_one_half * np.cosh(lambda_ell_plus_one_ld_over_chi)
)
omega_denominator = (
    ALPHA_SWEEP_VALUES * (np.cosh(lambda_ell_ld_over_chi) ** 2)
    + (alpha_minus_one_half**2)
)
omega_ratio = omega_numerator / omega_denominator

fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))

ax[0].plot(ALPHA_SWEEP_VALUES, omega_numerator, marker="o", color="tab:blue")
ax[0].set_title("Omega numerator")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("Numerator")

ax[1].plot(ALPHA_SWEEP_VALUES, omega_denominator, marker="o", color="tab:orange")
ax[1].set_title("Omega denominator")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("Denominator")

omega_scale_for_abl = (
    (1 / (COMPARE_T_0 * COMPARE_THETA**2))
    * (10 ** (COMPARE_RATE_LAMBDA * (1 - COMPARE_ELL) * ALPHA_SWEEP_ABL / 20))
)
ax[2].plot(
    ALPHA_SWEEP_VALUES,
    omega_scale_for_abl * omega_ratio,
    marker="o",
    color="tab:green",
)

ax[2].set_title(f"Omega at ABL={ALPHA_SWEEP_ABL}, ILD={ALPHA_SWEEP_ILD}")
ax[2].set_xlabel("alpha")
ax[2].set_ylabel("Omega")

for curr_ax in ax:
    curr_ax.grid(True, alpha=0.25)

fig.suptitle(
    f"Fixed: ABL={ALPHA_SWEEP_ABL}, ILD={ALPHA_SWEEP_ILD}, lambda'={COMPARE_RATE_LAMBDA:.6g}, "
    f"ell={COMPARE_ELL:.6g}, theta={COMPARE_THETA:.6g}, "
    f"T_0={COMPARE_T_0*1e3:.6g} ms"
)
fig.tight_layout()
fig.savefig(ALPHA_SWEEP_FIG_PATH, dpi=300, bbox_inches="tight")
print(f"Saved figure: {ALPHA_SWEEP_FIG_PATH}")

print(f"Alpha sweep at ABL={ALPHA_SWEEP_ABL}, ILD={ALPHA_SWEEP_ILD}:")
for alpha_value, numerator_value, denominator_value, ratio_value in zip(
    ALPHA_SWEEP_VALUES,
    omega_numerator,
    omega_denominator,
    omega_ratio,
):
    omega_value = omega_scale_for_abl * ratio_value
    print(
        f"  alpha={alpha_value:.1f}: numerator={numerator_value:.6g}, "
        f"denominator={denominator_value:.6g}, numerator/denominator={ratio_value:.6g}, "
        f"omega={omega_value:.6g}"
    )

plt.show()

# %%
