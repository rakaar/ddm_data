# %%
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

def get_param_means_by_ABL_ILD(batch_name, animal_id, ABLs_to_fit, ILDs_to_fit):
    """
    Returns a dictionary with keys (ABL, ILD) and values as dicts of mean parameter values.
    Only includes (ABL, ILD) combinations for which the corresponding pickle file exists.
    param_names: list of parameter names in the order of columns in vp_samples, default uses [gamma, omega, t_E_aff, w, del_go]
    """
    import os
    import pickle
    import numpy as np
    param_names = ['gamma', 'omega']
    param_dict = {}
    for ABL in ABLs_to_fit:
        for ILD in ILDs_to_fit:
            pkl_file = f'vbmc_cond_by_cond_{batch_name}_{animal_id}_{ABL}_ILD_{ILD}_FIX_t_E_w_del_go_same_as_parametric.pkl'
            if not os.path.exists(pkl_file):
                continue
            with open(pkl_file, 'rb') as f:
                vp = pickle.load(f)
            vp = vp.vp
            vp_samples = vp.sample(int(1e5))[0]
            means = {name: float(np.mean(vp_samples[:, i])) for i, name in enumerate(param_names)}
            param_dict[(ABL, ILD)] = means
    return param_dict

# === USER: Set these values as needed ===
batch_name = 'LED7' 
animal_id = '103'    
ABLs_to_fit = [20, 40, 60]  
ILDs_to_fit = [1, 2, 4, 8, 16, -1, -2, -4, -8, -16]  

# =======================================

# Get mean parameters
## Cond by Cond fit
d = get_param_means_by_ABL_ILD(batch_name, animal_id, ABLs_to_fit, ILDs_to_fit)

# Prepare data for scatter plot
ABLs_flat, ILDs_flat, omega_flat, gamma_flat = [], [], [], []
for (ABL, ILD), vals in d.items():
    ABLs_flat.append(ABL)
    ILDs_flat.append(ILD)
    omega_flat.append(vals['omega'])
    gamma_flat.append(vals['gamma'])
ABLs_flat = np.array(ABLs_flat)
ILDs_flat = np.array(ILDs_flat)
omega_flat = np.array(omega_flat)
gamma_flat = np.array(gamma_flat)

# %%
# Parametric fit
saved_vbmc_file = f'vbmc_mutiple_gama_omega_at_once_but_parametric_batch_{batch_name}_animal_{animal_id}_BETTER_BOUNDS_V2.pkl'
with open(saved_vbmc_file, 'rb') as f:
    vp = pickle.load(f)
vp = vp.vp
vp_samples = vp.sample(int(1e5))[0]

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

# Mean
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

def get_omega(ABL, ILD):
    if ABL == 20:
        return o_ratio_scale_20 * np.cosh(o_ild_scale_20 * (ILD - o_ild_offset_20)) / np.cosh(o_ild_scale_20 * norm_factor_20 * (ILD - o_ild_offset_20))
    elif ABL == 40:
        return o_ratio_scale_40 * np.cosh(o_ild_scale_40 * (ILD - o_ild_offset_40)) / np.cosh(o_ild_scale_40 * norm_factor_40 * (ILD - o_ild_offset_40))
    elif ABL == 60:
        return o_ratio_scale_60 * np.cosh(o_ild_scale_60 * (ILD - o_ild_offset_60)) / np.cosh(o_ild_scale_60 * norm_factor_60 * (ILD - o_ild_offset_60))
    else:
        return None

ILDs_cont = np.linspace(-16, 16, 100)
omega_cont = np.array([get_omega(ABL, ILD) for ABL in ABLs_to_fit for ILD in ILDs_cont])

# Scatter plot: omega vs ILD for each ABL
plt.figure(figsize=(6, 4))
colors = ['tab:blue', 'tab:orange', 'tab:green']
for i, ABL in enumerate(ABLs_to_fit):
    idx = np.where(ABLs_flat == ABL)[0]
    ILDs = ILDs_flat[idx]
    omega_obs = omega_flat[idx]
    plt.scatter(ILDs, omega_obs, color=colors[i%3], label=f'ABL={ABL} data')
    plt.plot(ILDs_cont, omega_cont[i*100:(i+1)*100], color=colors[i%3], label=f'ABL={ABL} parametric')
plt.xlabel('ILD (dB)')
plt.ylabel('omega')
plt.title(f'omega vs ILD, Animal: {animal_id}, Batch: {batch_name}')
plt.savefig(f'omega_vs_ILD_{animal_id}_{batch_name}_parametric_and_cond_by_cond.png')
plt.tight_layout()
plt.show()

# --- GAMMA PLOTS ---
def get_gamma(ABL, ILD):
    if ABL == 20:
        return g_tanh_scale_20 * np.tanh(g_ild_scale_20 * (ILD - g_ild_offset_20))
    elif ABL == 40:
        return g_tanh_scale_40 * np.tanh(g_ild_scale_40 * (ILD - g_ild_offset_40))
    elif ABL == 60:
        return g_tanh_scale_60 * np.tanh(g_ild_scale_60 * (ILD - g_ild_offset_60))
    else:
        return None

gamma_cont = np.array([get_gamma(ABL, ILD) for ABL in ABLs_to_fit for ILD in ILDs_cont])

# Scatter plot: gamma vs ILD for each ABL
plt.figure(figsize=(6, 4))
for i, ABL in enumerate(ABLs_to_fit):
    idx = np.where(ABLs_flat == ABL)[0]
    ILDs = ILDs_flat[idx]
    gamma_obs = gamma_flat[idx]
    plt.scatter(ILDs, gamma_obs, color=colors[i%3], label=f'ABL={ABL} data')
    plt.plot(ILDs_cont, gamma_cont[i*100:(i+1)*100], color=colors[i%3], label=f'ABL={ABL} parametric')
plt.xlabel('ILD (dB)')
plt.ylabel('gamma')
plt.title(f'gamma vs ILD, Animal: {animal_id}, Batch: {batch_name}')
plt.savefig(f'gamma_vs_ILD_{animal_id}_{batch_name}_parametric_and_cond_by_cond.png')
plt.tight_layout()
plt.show()

# %%
