# %%
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# --- Settings ---
# Directory containing parametric fit pickle files
import pandas as pd
import random

batch_name = 'LED7'
og_df = pd.read_csv('../out_LED.csv')
all_animals = og_df['animal'].unique()
ABLs = [20, 40, 60]
ILDs = np.linspace(-16, 16, 200)

# Find animals for which the pkl exists
param_dir = os.path.join(os.path.dirname(__file__), '../fit_each_condn')
existing_files = set(os.listdir(param_dir))
animal_ids_with_pkl = []
for animal_id in all_animals:
    animal_id_str = str(animal_id)
    pkl_name = f'vbmc_mutiple_gama_omega_at_once_but_parametric_batch_{batch_name}_animal_{animal_id_str}.pkl'
    if pkl_name in existing_files:
        animal_ids_with_pkl.append(animal_id_str)

# For plotting colors
dark_color = 'k'

# Store gamma/omega curves for averaging
all_gammas = {abl: [] for abl in ABLs}
all_omegas = {abl: [] for abl in ABLs}

for animal_id in animal_ids_with_pkl:
    print(f'##### Starting animal {animal_id} #####')
    pkl_name = f'vbmc_mutiple_gama_omega_at_once_but_parametric_batch_{batch_name}_animal_{animal_id}.pkl'
    with open(os.path.join(param_dir, pkl_name), 'rb') as f:
        vp = pickle.load(f)
    vp = vp.vp
    vp_samples = vp.sample(int(1e5))[0]
    # Means for each parameter
    g_tanh_scale_20 = vp_samples[:, 0].mean()
    g_ild_scale_20 = vp_samples[:, 1].mean()
    g_ild_offset_20 = vp_samples[:, 2].mean()
    o_ratio_scale_20 = vp_samples[:, 3].mean()
    o_ild_scale_20 = vp_samples[:, 4].mean()
    o_ild_offset_20 = vp_samples[:, 5].mean()
    norm_factor_20 = vp_samples[:, 6].mean()

    g_tanh_scale_40 = vp_samples[:, 7].mean()
    g_ild_scale_40 = vp_samples[:, 8].mean()
    g_ild_offset_40 = vp_samples[:, 9].mean()
    o_ratio_scale_40 = vp_samples[:, 10].mean()
    o_ild_scale_40 = vp_samples[:, 11].mean()
    o_ild_offset_40 = vp_samples[:, 12].mean()
    norm_factor_40 = vp_samples[:, 13].mean()

    g_tanh_scale_60 = vp_samples[:, 14].mean()
    g_ild_scale_60 = vp_samples[:, 15].mean()
    g_ild_offset_60 = vp_samples[:, 16].mean()
    o_ratio_scale_60 = vp_samples[:, 17].mean()
    o_ild_scale_60 = vp_samples[:, 18].mean()
    o_ild_offset_60 = vp_samples[:, 19].mean()
    norm_factor_60 = vp_samples[:, 20].mean()

    gammas = {}
    omegas = {}
    for ABL in ABLs:
        if ABL == 20:
            gammas[ABL] = g_tanh_scale_20 * np.tanh(g_ild_scale_20 * (ILDs - g_ild_offset_20))
            omegas[ABL] = o_ratio_scale_20 * np.cosh(o_ild_scale_20 * (ILDs - o_ild_offset_20)) / np.cosh(o_ild_scale_20 * norm_factor_20 * (ILDs - o_ild_offset_20))
        elif ABL == 40:
            gammas[ABL] = g_tanh_scale_40 * np.tanh(g_ild_scale_40 * (ILDs - g_ild_offset_40))
            omegas[ABL] = o_ratio_scale_40 * np.cosh(o_ild_scale_40 * (ILDs - o_ild_offset_40)) / np.cosh(o_ild_scale_40 * norm_factor_40 * (ILDs - o_ild_offset_40))
        elif ABL == 60:
            gammas[ABL] = g_tanh_scale_60 * np.tanh(g_ild_scale_60 * (ILDs - g_ild_offset_60))
            omegas[ABL] = o_ratio_scale_60 * np.cosh(o_ild_scale_60 * (ILDs - o_ild_offset_60)) / np.cosh(o_ild_scale_60 * norm_factor_60 * (ILDs - o_ild_offset_60))
        all_gammas[ABL].append(gammas[ABL])
        all_omegas[ABL].append(omegas[ABL])

# %%
    # Plot all ABLs in one figure for gamma and one for omega
abl_colors = {20: 'tab:blue', 40: 'tab:orange', 60: 'tab:green'}

plt.figure('gamma_all_ABLs', figsize=(6,4))
for idx in range(len(animal_ids_with_pkl)):
    for ABL in ABLs:
        plt.plot(ILDs, all_gammas[ABL][idx], color=abl_colors[ABL], alpha=0.4, lw=1)
# Plot mean
for ABL in ABLs:
    gamma_mean = np.mean(np.stack(all_gammas[ABL]), axis=0)
    plt.plot(ILDs, gamma_mean, color=abl_colors[ABL], lw=3, label=f'ABL={ABL} mean')
plt.xlabel('ILD (dB)')
plt.ylabel('gamma')
plt.title('gamma vs ILD (all ABLs)')
plt.savefig(f'gamma_parametric_fit_{batch_name}.png')
plt.legend()
plt.tight_layout()

plt.figure('omega_all_ABLs', figsize=(6,4))
for idx in range(len(animal_ids_with_pkl)):
    for ABL in ABLs:
        plt.plot(ILDs, all_omegas[ABL][idx], color=abl_colors[ABL], alpha=0.4, lw=1)
# Plot mean
for ABL in ABLs:
    omega_mean = np.mean(np.stack(all_omegas[ABL]), axis=0)
    plt.plot(ILDs, omega_mean, color=abl_colors[ABL], lw=2, label=f'ABL={ABL} mean')
plt.xlabel('ILD (dB)')
plt.ylabel('omega')
plt.title('omega vs ILD (all ABLs)')
plt.savefig(f'omega_parametric_fit_{batch_name}.png')
plt.legend()
plt.tight_layout()

plt.show()

# %%
# --- Parameter vs Animal Plot ---
import math

# Define parameter names and bounds (order matches VBMC vector)

new_g_tanh_scale_bounds = [0.01, 6]
new_g_tanh_scale_plausible_bounds = [0.5, 4]

new_g_ild_scale_bounds = [0.001, 0.7]
new_g_ild_scale_plausible_bounds = [0.1, 0.5]

new_o_ratio_scale_bounds = [0.1, 7]
new_o_ratio_scale_plausible_bounds = [0.5, 6]

new_o_ild_offset_bounds = [-3,3]
new_o_ild_offset_plausible_bounds = [-1,1]

new_norm_factor_bounds = [0.3,1.7]
new_norm_factor_plausible_bounds = [0.9, 1.1]

new_t_E_aff_bounds = [0.01, 0.12]
new_t_E_aff_plausible_bounds = [0.06, 0.1]



param_info = [
    ('g_tanh_scale_20', new_g_tanh_scale_bounds, new_g_tanh_scale_plausible_bounds),
    ('g_ild_scale_20', new_g_ild_scale_bounds, new_g_ild_scale_plausible_bounds),
    ('g_ild_offset_20', [-5, 5], [-3, 3]),
    ('o_ratio_scale_20', new_o_ratio_scale_bounds, new_o_ratio_scale_plausible_bounds),
    ('o_ild_scale_20', [0.01, 0.6], [0.05, 0.5]),
    ('o_ild_offset_20', new_o_ild_offset_bounds, new_o_ild_offset_plausible_bounds),
    ('norm_factor_20', new_norm_factor_bounds, new_norm_factor_plausible_bounds),
    ('g_tanh_scale_40', new_g_tanh_scale_bounds, new_g_tanh_scale_plausible_bounds),
    ('g_ild_scale_40', new_g_ild_scale_bounds, new_g_ild_scale_plausible_bounds),
    ('g_ild_offset_40', [-5, 5], [-3, 3]),
    ('o_ratio_scale_40', new_o_ratio_scale_bounds, new_o_ratio_scale_plausible_bounds),
    ('o_ild_scale_40', [0.01, 0.6], [0.05, 0.5]),
    ('o_ild_offset_40', new_o_ild_offset_bounds, new_o_ild_offset_plausible_bounds),
    ('norm_factor_40', new_norm_factor_bounds, new_norm_factor_plausible_bounds),
    ('g_tanh_scale_60', new_g_tanh_scale_bounds, new_g_tanh_scale_plausible_bounds),
    ('g_ild_scale_60', new_g_ild_scale_bounds, new_g_ild_scale_plausible_bounds),
    ('g_ild_offset_60', [-5, 5], [-3, 3]),
    ('o_ratio_scale_60', new_o_ratio_scale_bounds, new_o_ratio_scale_plausible_bounds),
    ('o_ild_scale_60', [0.01, 0.6], [0.05, 0.5]),
    ('o_ild_offset_60', new_o_ild_offset_bounds, new_o_ild_offset_plausible_bounds),
    ('norm_factor_60', new_norm_factor_bounds, new_norm_factor_plausible_bounds),
    ('w_20', [0.2, 0.8], [0.3, 0.7]),
    ('w_40', [0.2, 0.8], [0.3, 0.7]),
    ('w_60', [0.2, 0.8], [0.3, 0.7]),
    ('t_E_aff', new_t_E_aff_bounds, new_t_E_aff_plausible_bounds),
    ('del_go', [0.001, 0.2], [0.11, 0.15]),
]

# Only use the first 22 parameters (as in VBMC vector)
param_names = [x[0] for x in param_info]

# Collect parameter means for each animal
param_means_by_animal = {name: [] for name in param_names}
animal_id_strs = []

for animal_id in animal_ids_with_pkl:
    pkl_name = f'vbmc_mutiple_gama_omega_at_once_but_parametric_batch_{batch_name}_animal_{animal_id}.pkl'
    with open(os.path.join(param_dir, pkl_name), 'rb') as f:
        vp = pickle.load(f)
    vp = vp.vp
    vp_samples = vp.sample(int(1e5))[0]
    for i, pname in enumerate(param_names):
        param_means_by_animal[pname].append(vp_samples[:, i].mean())
    animal_id_strs.append(str(animal_id))

n_params = len(param_names)
n_animals = len(animal_id_strs)
ncols = 7
nrows = math.ceil(n_params / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*3), squeeze=False)
for i, (pname, bounds, plausible) in enumerate(param_info):
    row, col = divmod(i, ncols)
    ax = axes[row][col]
    y = param_means_by_animal[pname]
    ax.scatter(animal_id_strs, y)
    ax.set_title(pname)
    ax.set_xticks(range(len(animal_id_strs)))
    ax.set_xticklabels(animal_id_strs, rotation=90, fontsize=8)
    # Set y-limits and ticks
    yticks = [bounds[0], plausible[0], plausible[1], bounds[1]]
    ax.set_ylim(bounds[0], bounds[1])
    ax.set_yticks(yticks)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    # Highlight plausible bounds
    ax.axhline(plausible[0], color='orange', ls='--', lw=1)
    ax.axhline(plausible[1], color='orange', ls='--', lw=1)
    ax.axhline(bounds[0], color='red', ls='-', lw=1)
    ax.axhline(bounds[1], color='red', ls='-', lw=1)

# Hide unused subplots
for i in range(n_params, nrows*ncols):
    row, col = divmod(i, ncols)
    axes[row][col].axis('off')

fig.tight_layout()
fig.subplots_adjust(bottom=0.18)
fig.savefig(f'param_vs_animal_{batch_name}_parametric_fit.png')
plt.show()