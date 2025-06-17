# %%
import pickle

# %% 
# Load parametric fit params
pkl_file_name = 'vbmc_mutiple_gama_omega_at_once_but_parametric.pkl'
with open(pkl_file_name, 'rb') as f:
    parametric_fit_params = pickle.load(f)

parametric_fit_params = parametric_fit_params.vp
vp_samples = parametric_fit_params.sample(int(1e5))[0]

# Extract new parameter samples according to the new order
g_tanh_scale_20_samples = vp_samples[:, 0]
g_ild_scale_20_samples = vp_samples[:, 1]
g_ild_offset_20_samples = vp_samples[:, 2]
o_ratio_scale_20_samples = vp_samples[:, 3]
o_ild_scale_20_samples = vp_samples[:, 4]
o_ild_offset_20_samples = vp_samples[:, 5]
norm_factor_20_samples = vp_samples[:, 6]

g_tanh_scale_40_samples = vp_samples[:, 7]
g_ild_scale_40_samples = vp_samples[:, 8]
g_ild_offset_40_samples = vp_samples[:, 9]
o_ratio_scale_40_samples = vp_samples[:, 10]
o_ild_scale_40_samples = vp_samples[:, 11]
o_ild_offset_40_samples = vp_samples[:, 12]
norm_factor_40_samples = vp_samples[:, 13]

g_tanh_scale_60_samples = vp_samples[:, 14]
g_ild_scale_60_samples = vp_samples[:, 15]
g_ild_offset_60_samples = vp_samples[:, 16]
o_ratio_scale_60_samples = vp_samples[:, 17]
o_ild_scale_60_samples = vp_samples[:, 18]
o_ild_offset_60_samples = vp_samples[:, 19]
norm_factor_60_samples = vp_samples[:, 20]


# Means (if needed elsewhere)
g_tanh_scale_20 = g_tanh_scale_20_samples.mean()
g_ild_scale_20 = g_ild_scale_20_samples.mean()
g_ild_offset_20 = g_ild_offset_20_samples.mean()
o_ratio_scale_20 = o_ratio_scale_20_samples.mean()
o_ild_scale_20 = o_ild_scale_20_samples.mean()
o_ild_offset_20 = o_ild_offset_20_samples.mean()
norm_factor_20 = norm_factor_20_samples.mean()

g_tanh_scale_40 = g_tanh_scale_40_samples.mean()
g_ild_scale_40 = g_ild_scale_40_samples.mean()
g_ild_offset_40 = g_ild_offset_40_samples.mean()
o_ratio_scale_40 = o_ratio_scale_40_samples.mean()
o_ild_scale_40 = o_ild_scale_40_samples.mean()
o_ild_offset_40 = o_ild_offset_40_samples.mean()
norm_factor_40 = norm_factor_40_samples.mean()

g_tanh_scale_60 = g_tanh_scale_60_samples.mean()
g_ild_scale_60 = g_ild_scale_60_samples.mean()
g_ild_offset_60 = g_ild_offset_60_samples.mean()
o_ratio_scale_60 = o_ratio_scale_60_samples.mean()
o_ild_scale_60 = o_ild_scale_60_samples.mean()
o_ild_offset_60 = o_ild_offset_60_samples.mean()
norm_factor_60 = norm_factor_60_samples.mean()

    

def get_gamma_parametric(ABL, ILD):
    if ABL == 20:
        return g_tanh_scale_20 * np.tanh(g_ild_scale_20 * (ILD - g_ild_offset_20))
    elif ABL == 40:
        return g_tanh_scale_40 * np.tanh(g_ild_scale_40 * (ILD - g_ild_offset_40))
    elif ABL == 60:
        return g_tanh_scale_60 * np.tanh(g_ild_scale_60 * (ILD - g_ild_offset_60))
    else:
        return None

def get_omega_parametric(ABL, ILD):
    if ABL == 20:
        return o_ratio_scale_20 * np.cosh(o_ild_scale_20 * (ILD - o_ild_offset_20)) / np.cosh(o_ild_scale_20 * norm_factor_20 * (ILD - o_ild_offset_20))
    elif ABL == 40:
        return o_ratio_scale_40 * np.cosh(o_ild_scale_40 * (ILD - o_ild_offset_40)) / np.cosh(o_ild_scale_40 * norm_factor_40 * (ILD - o_ild_offset_40))
    elif ABL == 60:
        return o_ratio_scale_60 * np.cosh(o_ild_scale_60 * (ILD - o_ild_offset_60)) / np.cosh(o_ild_scale_60 * norm_factor_60 * (ILD - o_ild_offset_60))
    else:
        return None
# %%
# Cond by cond fit
import pickle

# Load cond-by-cond fit gamma parameters
with open('gamma_parametric_params_LED7_103_on_cond_by_cond_fit.pkl', 'rb') as f:
    gamma_params_dict = pickle.load(f)

# Load cond-by-cond fit omega parameters
with open('omega_parametric_params_LED7_103_on_cond_by_cond_fit.pkl', 'rb') as f:
    omega_params_dict = pickle.load(f)

def get_gamma_cond_fit(ABL, ILD):
    if ABL == 20:
        return gamma_params_dict[f'gamma_tanh_scale_{ABL}'] * np.tanh(gamma_params_dict[f'gamma_ILD_scale_{ABL}'] * (ILD - gamma_params_dict[f'gamma_ILD_offset_{ABL}']))
    elif ABL == 40:
        return gamma_params_dict[f'gamma_tanh_scale_{ABL}'] * np.tanh(gamma_params_dict[f'gamma_ILD_scale_{ABL}'] * (ILD - gamma_params_dict[f'gamma_ILD_offset_{ABL}']))
    elif ABL == 60:
        return gamma_params_dict[f'gamma_tanh_scale_{ABL}'] * np.tanh(gamma_params_dict[f'gamma_ILD_scale_{ABL}'] * (ILD - gamma_params_dict[f'gamma_ILD_offset_{ABL}']))
    else:
        return None

def get_omega_cond_fit(ABL, ILD):
    if ABL == 20:
        return omega_params_dict[f'omega_ratio_scale_{ABL}'] * np.cosh(omega_params_dict[f'omega_ild_scale_{ABL}'] * (ILD - omega_params_dict[f'omega_ild_offset_{ABL}'])) / np.cosh(omega_params_dict[f'omega_ild_scale_{ABL}']*omega_params_dict[f'omega_norm_factor_{ABL}'] * (ILD - omega_params_dict[f'omega_ild_offset_{ABL}']))
    elif ABL == 40:
        return omega_params_dict[f'omega_ratio_scale_{ABL}'] * np.cosh(omega_params_dict[f'omega_ild_scale_{ABL}'] * (ILD - omega_params_dict[f'omega_ild_offset_{ABL}'])) / np.cosh(omega_params_dict[f'omega_ild_scale_{ABL}']*omega_params_dict[f'omega_norm_factor_{ABL}'] * (ILD - omega_params_dict[f'omega_ild_offset_{ABL}']))
    elif ABL == 60:
        return omega_params_dict[f'omega_ratio_scale_{ABL}'] * np.cosh(omega_params_dict[f'omega_ild_scale_{ABL}'] * (ILD - omega_params_dict[f'omega_ild_offset_{ABL}'])) / np.cosh(omega_params_dict[f'omega_ild_scale_{ABL}']*omega_params_dict[f'omega_norm_factor_{ABL}'] * (ILD - omega_params_dict[f'omega_ild_offset_{ABL}']))
    else:
        return None

# %%
ABLs = [20, 40, 60]
# Continuous ILD range for parametric plots
ILDs_cont = np.linspace(-16, 16, 200)
ABL_color_map = {20: 'tab:blue', 40: 'tab:orange', 60: 'tab:green'}

import numpy as np
import matplotlib.pyplot as plt

# 1. Gamma vs ILD for each ABL (continuous)
plt.figure(figsize=(8, 6))
for abl in ABLs:
    gamma_vals = [get_gamma_parametric(abl, ild) for ild in ILDs_cont]
    gamma_cond_vals = [get_gamma_cond_fit(abl, ild) for ild in ILDs_cont]
    plt.plot(ILDs_cont, gamma_vals, label=f'ABL={abl}', color=ABL_color_map[abl])
    plt.plot(ILDs_cont, gamma_cond_vals, label=f'ABL={abl} cond fit', ls='--', color=ABL_color_map[abl])
plt.xlabel('ILD (dB)')
plt.ylabel('gamma')
plt.title('Gamma vs ILD for Each ABL (continuous)')
# plt.legend()
plt.tight_layout()
plt.show()

# 2. Omega vs ILD for each ABL (continuous)
plt.figure(figsize=(8, 6))
for abl in ABLs:
    omega_vals = [get_omega_parametric(abl, ild) for ild in ILDs_cont]
    omega_cond_vals = [get_omega_cond_fit(abl, ild) for ild in ILDs_cont]
    plt.plot(ILDs_cont, omega_vals, label=f'ABL={abl}', color=ABL_color_map[abl])
    plt.plot(ILDs_cont, omega_cond_vals, label=f'ABL={abl} cond fit', ls='--', color=ABL_color_map[abl])
plt.xlabel('ILD (dB)')
plt.ylabel('omega')
plt.title('Omega vs ILD for Each ABL (continuous)')
# plt.legend()
plt.tight_layout()
plt.show()

