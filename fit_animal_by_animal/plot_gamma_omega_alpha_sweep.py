# %%
import matplotlib.pyplot as plt
import numpy as np


# %% Parameters
ABLS = [20, 40, 60]
ILDS = np.sort([-16, -8, -4, -2, -1, 1, 2, 4, 8, 16])
ALPHAS = [0, 0.25, 0.5, 0.75, 1]

lambda_prime = 2
ell = 0.9
T_0 = 100e-3
theta = 2.4
P_0 = 20e-6

COLORS = ["tab:blue", "tab:orange", "tab:green"]
FIGSIZE = (18, 4)


# %% Functions
def levels_from_abl_ild(abl, ild):
    R_dB = abl + ild / 2
    L_dB = abl - ild / 2
    return R_dB, L_dB


def pressure_from_db(db, P_0):
    return P_0 * (10 ** (db / 20))


def firing_rates(P_R, P_L, alpha, lambda_prime, ell, P_0):
    P_R_scaled = P_R / P_0
    P_L_scaled = P_L / P_0

    r_R = (P_R_scaled ** lambda_prime) / (
        (P_R_scaled ** (lambda_prime * ell)) + alpha * (P_L_scaled ** (lambda_prime * ell))
    )
    r_L = (P_L_scaled ** lambda_prime) / (
        (P_L_scaled ** (lambda_prime * ell)) + alpha * (P_R_scaled ** (lambda_prime * ell))
    )
    return r_R, r_L


def gamma_from_rates(r_R, r_L, theta):
    return theta * (r_R - r_L) / (r_R + r_L)


def omega_from_rates(r_R, r_L, T_0, theta):
    return (r_R + r_L) / (T_0 * theta**2)


# %% Compute Gamma and Omega for each alpha, ABL, ILD
gamma_by_alpha_abl = {}
omega_by_alpha_abl = {}

for alpha in ALPHAS:
    gamma_by_alpha_abl[alpha] = {}
    omega_by_alpha_abl[alpha] = {}

    for abl in ABLS:
        R_dB, L_dB = levels_from_abl_ild(abl, ILDS)

        if not np.allclose(R_dB - L_dB, ILDS):
            raise ValueError(f"Sound-level conversion failed for ABL={abl}: R_dB - L_dB != ILD")
        if not np.allclose((R_dB + L_dB) / 2, abl):
            raise ValueError(f"Sound-level conversion failed for ABL={abl}: average != ABL")

        P_R = pressure_from_db(R_dB, P_0)
        P_L = pressure_from_db(L_dB, P_0)
        r_R, r_L = firing_rates(P_R, P_L, alpha, lambda_prime, ell, P_0)

        gamma_by_alpha_abl[alpha][abl] = gamma_from_rates(r_R, r_L, theta)
        omega_by_alpha_abl[alpha][abl] = omega_from_rates(r_R, r_L, T_0, theta)


# %% Plot Gamma alpha sweep
fig, axes = plt.subplots(1, len(ALPHAS), figsize=FIGSIZE, sharex=True, sharey=True)

for alpha_idx, alpha in enumerate(ALPHAS):
    ax = axes[alpha_idx]
    for abl_idx, abl in enumerate(ABLS):
        ax.plot(
            ILDS,
            gamma_by_alpha_abl[alpha][abl],
            marker="o",
            color=COLORS[abl_idx],
            label=f"ABL={abl}",
        )

    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.4)
    ax.set_title(f"alpha={alpha}")
    ax.set_xlabel("ILD")
    ax.grid(True, alpha=0.25)

axes[0].set_ylabel("Gamma")
axes[0].legend()
fig.suptitle("Gamma vs ILD for alpha sweep")
fig.tight_layout()


# %% Plot Omega alpha sweep
fig, axes = plt.subplots(1, len(ALPHAS), figsize=FIGSIZE, sharex=True, sharey=True)

for alpha_idx, alpha in enumerate(ALPHAS):
    ax = axes[alpha_idx]
    for abl_idx, abl in enumerate(ABLS):
        ax.plot(
            ILDS,
            omega_by_alpha_abl[alpha][abl],
            marker="o",
            color=COLORS[abl_idx],
            label=f"ABL={abl}",
        )

    ax.set_title(f"alpha={alpha}")
    ax.set_xlabel("ILD")
    ax.grid(True, alpha=0.25)

axes[0].set_ylabel("Omega")
axes[0].legend()
fig.suptitle("Omega vs ILD for alpha sweep")
fig.tight_layout()

plt.show()

# %%
