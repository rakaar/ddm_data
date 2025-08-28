# %%
#  INCOMPLETE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

def gama(lam, theta ,ild):
    return theta * np.tanh(lam * ild / 17.37)

# Publication grade styling
TITLE_FONTSIZE = 24
LABEL_FONTSIZE = 25
TICK_FONTSIZE = 24
LEGEND_FONTSIZE = 16
SUPTITLE_FONTSIZE = 24
font_family = "Helvetica"


mpl.rcParams.update({
    "savefig.pad_inches": 0.6,
    "font.family": "sans-serif",
    "font.sans-serif": [
        font_family,
        "Helvetica Neue",
        "TeX Gyre Heros",
        "Arial",
        "sans-serif",
    ],
    "axes.labelpad": 12,
})

# Gamma = theta0*tanh(lambda0*ILD/Chi);
# omega = 2*10.^(lambda0*ABL/20)/theta0^2/T0;

# RT_tau = tanh(Gamma)./Gamma;
# RT_tau(1) = 1;
# RT = RT_tau./omega + delta_E;

def mean_rt_fn(lam, theta, T0, del_E, ild, abl, l = 0):
    gamma = theta * np.tanh(lam * ild / 17.37)
    omega = (2/(T0 * (theta ** 2))) * (10 ** (lam * abl / 20))

    tau = np.tanh(gamma) / gamma

    rt = (tau / omega) + del_E
    return rt

def mean_rt_fn_2(lam, theta, T0, del_E, ild, abl, l):
    gamma = theta * np.tanh(lam * ild / 17.37)
    omega = (2/(T0 * (theta ** 2))) * (10 ** (lam * (1-l) * abl / 20)) * (np.cosh(lam * ild / 17.37) / np.cosh(lam * ild * l / 17.37))

    tau = np.tanh(gamma) / gamma

    rt = (tau / omega) + del_E
    return rt

ABLs = [20, 40, 60]
ilds_range = np.arange(0, 16, 0.1)

# params 1
lam = 0.2
theta = 30
T0 = 0.5 * 1e-3

# params 2
lam_2 = 2.7
theta_2 = 2
T0_2 = 100 * 1e-3

# 1 x 2 - 2 models
# each plot - mean RT vs ILD for 3 ABLs
# figure 1/2
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

colors = ['tab:blue', 'tab:orange', 'tab:green']

# Left subplot - Model 1 (mean_rt_fn)
for i, abl in enumerate(ABLs):
    mean_rt_1 = mean_rt_fn(lam, theta, T0, 0, ilds_range, abl)
    ax[0].plot(ilds_range, mean_rt_1, color=colors[i], label=f'ABL {abl}')
ax[0].set_xlabel('ILD')
ax[0].set_ylabel('Mean RT')
ax[0].set_title('Model 1')
ax[0].legend()

# Right subplot - Model 2 (mean_rt_fn_2)
for i, abl in enumerate(ABLs):
    mean_rt_2 = mean_rt_fn_2(lam_2, theta_2, T0_2, 0, ilds_range, abl, 0.5)
    ax[1].plot(ilds_range, mean_rt_2, color=colors[i], label=f'ABL {abl}')
ax[1].set_xlabel('ILD')
ax[1].set_ylabel('Mean RT')
ax[1].set_title('Model 2')
ax[1].legend()

plt.tight_layout()
plt.show()
