# %%
# Psychometric fitting for all animals
import time  # ensure this is only imported once at the top

import numpy as np
from scipy.integrate import cumulative_trapezoid as cumtrapz
# Set non-interactive backend before importing plt to avoid tkinter errors
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from pyvbmc import VBMC
import corner
from time_vary_norm_utils import (
    up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn, 
    up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn_vec,
    cum_pro_and_reactive_time_vary_fn, 
    rho_A_t_fn, 
    cum_A_t_fn
)
import sys
import multiprocessing
from scipy.integrate import trapezoid
import random
import glob
import os
import time

from scipy.interpolate import interp1d
import pickle
# %%
DESIRED_BATCHES = ['LED7']

# Base directory paths
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_dir = os.path.join(base_dir, 'batch_csvs')
results_dir = base_dir  # Directory containing the pickle files

def find_batch_animal_pairs():
    pairs = []
    pattern = os.path.join(results_dir, 'results_*_animal_*.pkl')
    pickle_files = glob.glob(pattern)
    for pickle_file in pickle_files:
        filename = os.path.basename(pickle_file)
        parts = filename.split('_')
        if len(parts) >= 4:
            batch_index = parts.index('animal') - 1 if 'animal' in parts else 1
            animal_index = parts.index('animal') + 1 if 'animal' in parts else 2
            batch_name = parts[batch_index]
            animal_id = parts[animal_index].split('.')[0]
            if batch_name in DESIRED_BATCHES:
                pairs.append((batch_name, animal_id))
        else:
            print(f"Warning: Invalid filename format: {filename}")
    return pairs

batch_animal_pairs = find_batch_animal_pairs()
# with open('high_slope_animals.pkl', 'rb') as f:
#     batch_animal_pairs = pickle.load(f)


print(f"Found {len(batch_animal_pairs)} batch-animal pairs: {batch_animal_pairs}")


psycho_fits_repo_path = '/home/rlab/raghavendra/ddm_data/fit_valid_trials/psycho_fits/'

def get_psycho_params(batch_name, animal_id):
    filename = os.path.join(psycho_fits_repo_path, f'psycho_fit_{batch_name}_{animal_id}.pkl')
    with open(filename, 'rb') as f:
        vp = pickle.load(f)
    vp = vp.vp
    samples = vp.sample(int(1e6))[0]
    tied_params = {
        'rate_lambda' : samples[:,0].mean(),
        'T_0' : samples[:,1].mean(),
        'theta_E' : samples[:,2].mean(),
        'w' : samples[:,3].mean(),
        't_E_aff' : samples[:,4].mean(),
        'del_go' : samples[:,5].mean()
    }
    return tied_params

# %%
# --- Collect parameter values for all animals into a dict of arrays ---
param_dict = {}
all_params = []
for batch_name, animal_id in batch_animal_pairs:
    try:
        params = get_psycho_params(batch_name, animal_id)
        all_params.append(params)
    except Exception as e:
        print(f"Failed to get params for {batch_name}, {animal_id}: {e}")
        continue

# Collect all parameter names present
if all_params:
    param_names = all_params[0].keys()
    for pname in param_names:
        param_dict[pname] = [p[pname] for p in all_params]
    print("Parameter dictionary (param_dict) created:")
    for k, v in param_dict.items():
        print(f"{k}: {v}")
else:
    print("No parameter data collected.")

# %%
MODEL_TYPE = 'norm'
def get_params_from_animal_pkl_file(batch_name, animal_id):
    pkl_file = f'results_{batch_name}_animal_{animal_id}.pkl'
    with open(pkl_file, 'rb') as f:
        fit_results_data = pickle.load(f)
    vbmc_aborts_param_keys_map = {
        'V_A_samples': 'V_A',
        'theta_A_samples': 'theta_A',
        't_A_aff_samp': 't_A_aff'
    }
    vbmc_vanilla_tied_param_keys_map = {
        'rate_lambda_samples': 'rate_lambda',
        'T_0_samples': 'T_0',
        'theta_E_samples': 'theta_E',
        'w_samples': 'w',
        't_E_aff_samples': 't_E_aff',
        'del_go_samples': 'del_go'
    }
    vbmc_norm_tied_param_keys_map = {
        **vbmc_vanilla_tied_param_keys_map,
        'rate_norm_l_samples': 'rate_norm_l'
    }
    abort_keyname = "vbmc_aborts_results"
    if MODEL_TYPE == 'vanilla':
        tied_keyname = "vbmc_vanilla_tied_results"
        tied_param_keys_map = vbmc_vanilla_tied_param_keys_map
        is_norm = False
    elif MODEL_TYPE == 'norm':
        tied_keyname = "vbmc_norm_tied_results"
        tied_param_keys_map = vbmc_norm_tied_param_keys_map
        is_norm = True
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")
    abort_params = {}
    tied_params = {}
    rate_norm_l = 0
    if abort_keyname in fit_results_data:
        abort_samples = fit_results_data[abort_keyname]
        for param_samples_name, param_label in vbmc_aborts_param_keys_map.items():
            abort_params[param_label] = np.mean(abort_samples[param_samples_name])
    if tied_keyname in fit_results_data:
        tied_samples = fit_results_data[tied_keyname]
        for param_samples_name, param_label in tied_param_keys_map.items():
            tied_params[param_label] = np.mean(tied_samples[param_samples_name])
        if is_norm:
            rate_norm_l = tied_params.get('rate_norm_l', np.nan)
        else:
            rate_norm_l = 0
    else:
        print(f"Warning: {tied_keyname} not found in pickle for {batch_name}, {animal_id}")
    return abort_params, tied_params, rate_norm_l, is_norm

# --- Collect norm model parameter values for all animals into a dict of arrays ---
norm_param_dict = {}
norm_all_params = []
for batch_name, animal_id in batch_animal_pairs:
    try:
        _, tied_params, _, _ = get_params_from_animal_pkl_file(batch_name, animal_id)
        norm_all_params.append(tied_params)
    except Exception as e:
        print(f"Failed to get norm model params for {batch_name}, {animal_id}: {e}")
        continue

# Collect all parameter names present
if norm_all_params:
    norm_param_names = norm_all_params[0].keys()
    for pname in norm_param_names:
        norm_param_dict[pname] = [p[pname] for p in norm_all_params]
    print("Norm model parameter dictionary (norm_param_dict) created:")
    for k, v in norm_param_dict.items():
        print(f"{k}: {v}")
else:
    print("No norm model parameter data collected.")

# %%
import matplotlib.pyplot as plt

# Find common parameter names (those present in both dicts)
common_params = list(set(param_dict.keys()) & set(norm_param_dict.keys()))
# Optionally exclude norm-only parameter if present
common_params = [p for p in common_params if not p.startswith('rate_norm_l')]

num_params = len(common_params)
num_animals = len(batch_animal_pairs)
animal_labels = [f"{b},{a}" for b, a in batch_animal_pairs]

fig, axes = plt.subplots(1, num_params, figsize=(4*num_params, 5))  # <-- No sharey

if num_params == 1:
    axes = [axes]

for idx, pname in enumerate(common_params):
    psycho_vals = param_dict[pname]
    norm_vals = norm_param_dict[pname]
    ax = axes[idx]
    x = range(num_animals)
    if pname == 'T_0':
        psycho_vals = [v * 1000 for v in psycho_vals]
        norm_vals = [v * 1000 for v in norm_vals]
        ax.set_title('T_0 (ms)')
    else:
        ax.set_title(pname)
    ax.scatter(x, psycho_vals, color='blue', label='psycho', s=60)
    ax.scatter(x, norm_vals, color='red', label='norm', marker='x', s=60)
    ax.set_xticks(x)
    ax.set_xticklabels(animal_labels, rotation=45, ha='right')
    if idx == 0:
        ax.set_ylabel('Parameter value')
    ax.legend()

# %%
import matplotlib.pyplot as plt
import numpy as np

# --- Compute gamma curves for each animal and model ---
ILD_arr = np.arange(-16, 16, 0.05)

def gamma_from_params(tied_params, ild_theory):
    rate_lambda = tied_params['rate_lambda']
    theta_E = tied_params['theta_E']
    return theta_E * np.tanh(rate_lambda * ild_theory / 17.37)

# Gather gamma curves for all animals for both models
psycho_gammas = []
norm_gammas = []
for i in range(len(batch_animal_pairs)):
    psycho_params = {k: param_dict[k][i] for k in param_dict}
    norm_params = {k: norm_param_dict[k][i] for k in norm_param_dict}
    psycho_gammas.append(gamma_from_params(psycho_params, ILD_arr))
    norm_gammas.append(gamma_from_params(norm_params, ILD_arr))

psycho_gammas = np.vstack(psycho_gammas)
norm_gammas = np.vstack(norm_gammas)

# --- 1 x 2 subplot: all animals, psycho vs norm ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for i in range(len(batch_animal_pairs)):
    axes[0].plot(ILD_arr, psycho_gammas[i], alpha=0.7, label=f'Animal {i+1}')
    axes[1].plot(ILD_arr, norm_gammas[i], alpha=0.7, label=f'Animal {i+1}')
axes[0].set_title('Psycho model')
axes[1].set_title('Norm model')
for ax in axes:
    ax.set_xlabel('ILD')
    ax.set_ylabel('gamma')
axes[0].legend(fontsize='small', loc='best')
plt.tight_layout()
plt.show()

# --- Single plot: average gamma curve for both models ---
plt.figure(figsize=(6, 5))
plt.plot(ILD_arr, psycho_gammas.mean(axis=0), color='blue', label='Psycho (avg)')
plt.plot(ILD_arr, norm_gammas.mean(axis=0), color='red', label='Norm (avg)')
plt.xlabel('ILD')
plt.ylabel('gamma')
plt.title('Average gamma vs ILD (all animals)')
plt.legend()
plt.tight_layout()
plt.show()

# %%
# --- Extract vanilla model params and plot average gamma curve alongside psycho and norm ---

# Save current MODEL_TYPE and set to 'vanilla'
MODEL_TYPE_ORIG = MODEL_TYPE
MODEL_TYPE = 'vanilla'

# Collect vanilla model parameter dict
vanilla_param_dict = {}
vanilla_all_params = []
for batch_name, animal_id in batch_animal_pairs:
    try:
        _, tied_params, _, _ = get_params_from_animal_pkl_file(batch_name, animal_id)
        vanilla_all_params.append(tied_params)
    except Exception as e:
        print(f"Failed to get vanilla model params for {batch_name}, {animal_id}: {e}")
        continue

if vanilla_all_params:
    vanilla_param_names = vanilla_all_params[0].keys()
    for pname in vanilla_param_names:
        vanilla_param_dict[pname] = [p[pname] for p in vanilla_all_params]
else:
    print("No vanilla model parameter data collected.")

# Compute gamma curves for vanilla
vanilla_gammas = []
for i in range(len(batch_animal_pairs)):
    vanilla_params = {k: vanilla_param_dict[k][i] for k in vanilla_param_dict}
    vanilla_gammas.append(gamma_from_params(vanilla_params, ILD_arr))
vanilla_gammas = np.vstack(vanilla_gammas)


# --- Single plot: average gamma curve for all three models ---
plt.figure(figsize=(7, 5))
plt.plot(ILD_arr, psycho_gammas.mean(axis=0), color='blue', label='Psycho (avg)')
plt.plot(ILD_arr, norm_gammas.mean(axis=0), color='red', label='Norm (avg)')
plt.plot(ILD_arr, vanilla_gammas.mean(axis=0), color='green', label='Vanilla (avg)')
plt.xlabel('ILD')
plt.ylabel('gamma')
plt.title('Average gamma vs ILD (all animals)')
plt.legend(frameon=False)
plt.xticks([-15, -5, 0, 5, 15])
plt.yticks([-4, 0, 4])
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(1, num_params, figsize=(4*num_params, 5))  # <-- No sharey

if num_params == 1:
    axes = [axes]

for idx, pname in enumerate(common_params):
    psycho_vals = param_dict[pname]
    norm_vals = norm_param_dict[pname]
    vanilla_vals = vanilla_param_dict[pname] if pname in vanilla_param_dict else [np.nan]*num_animals  # fallback if missing
    ax = axes[idx]
    x = range(num_animals)
    if pname == 'T_0':
        psycho_vals = [v * 1000 for v in psycho_vals]
        norm_vals = [v * 1000 for v in norm_vals]
        vanilla_vals = [v * 1000 for v in vanilla_vals]
        ax.set_title('T_0 (ms)')
    else:
        ax.set_title(pname)
    ax.scatter(x, psycho_vals, color='blue', label='psycho', s=60)
    # ax.scatter(x, norm_vals, color='red', label='norm', marker='x', s=60)
    ax.scatter(x, vanilla_vals, color='green', label='vanilla', marker='^', s=60)
    ax.set_xticks(x)
    ax.set_xticklabels(animal_labels, rotation=45, ha='right')
    if idx == 0:
        ax.set_ylabel('Parameter value')
    ax.legend()

plt.tight_layout()