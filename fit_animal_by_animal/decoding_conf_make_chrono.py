"""
Unified RTD analysis script for vanilla and normalized TIED models.

Set MODEL_TYPE = 'vanilla' or 'norm' at the top to switch between models.
- 'vanilla': uses vbmc_vanilla_tied_results from pickle, is_norm=False, rate_norm_l=0
- 'norm':    uses vbmc_norm_tied_results from pickle, is_norm=True, rate_norm_l from pickle

All downstream logic is automatically adjusted based on this flag.
"""
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from joblib import Parallel, delayed
from tqdm import tqdm
from time_vary_and_norm_simulators import psiam_tied_data_gen_wrapper_rate_norm_fn
import pickle
import warnings
from types import SimpleNamespace
from animal_wise_plotting_utils import calculate_theoretical_curves
from time_vary_norm_utils import (
    up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn, 
    up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn_vec,
    cum_pro_and_reactive_time_vary_fn, 
    rho_A_t_fn, 
    cum_A_t_fn
)
from collections import defaultdict
import random
from scipy.stats import gaussian_kde

def get_simulation_RTD_KDE(
    abort_params, tied_params, rate_norm_l, Z_E, ABL, ILD, t_stim_samples, N_sim, N_print, dt, n_jobs=30
):
    """
    Run the simulation for given parameters and return KDE arrays for RTD.
    Returns: x_vals, kde_vals
    """
    sim_results = Parallel(n_jobs=n_jobs)(
        delayed(psiam_tied_data_gen_wrapper_rate_norm_fn)(
            abort_params['V_A'], abort_params['theta_A'], ABL, ILD, tied_params['rate_lambda'], tied_params['T_0'],
            tied_params['theta_E'], Z_E, abort_params['t_A_aff'], tied_params['t_E_aff'], tied_params['del_go'],
            t_stim_samples[iter_num], rate_norm_l, iter_num, N_print, dt
        ) for iter_num in range(N_sim)
    )
    sim_results_df = pd.DataFrame(sim_results)
    sim_results_df_valid = sim_results_df[sim_results_df['rt'] - sim_results_df['t_stim'] > -0.1]
    sim_results_df_valid_lt_1 = sim_results_df_valid[sim_results_df_valid['rt'] - sim_results_df_valid['t_stim'] <= 1]
    sim_rt = sim_results_df_valid_lt_1['rt'] - sim_results_df_valid_lt_1['t_stim']
    kde = gaussian_kde(sim_rt)
    x_vals = np.arange(-0.12, 1, 0.01)
    kde_vals = kde(x_vals)
    return x_vals, kde_vals

# %%
# Define desired batches
DESIRED_BATCHES = ['Comparable', 'SD', 'LED2', 'LED1', 'LED34', 'LED6']
# DESIRED_BATCHES = ['LED7']

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
# %%
ABL_arr = [20, 40, 60]
abs_ILD_arr = [1,2,4,8,16]

# %%
# remove SD 49 due to issue in sensory delay
def get_mean_RT_by_ABL_absILD(batch_name, animal_id):
    """
    For a given batch and animal, compute mean RT for each (ABL, absILD) pair.
    Returns a dictionary with keys (ABL, absILD), values mean RT.
    Applies filter: 0 <= RT <= 1.
    """
    file_name = f'batch_csvs/batch_{batch_name}_valid_and_aborts.csv'
    df = pd.read_csv(file_name)
    df = df[df['animal'] == int(animal_id)]
    if 'absILD' not in df.columns:
        df['absILD'] = df['ILD'].abs()
    df = df[(df['RTwrtStim'] >= 0) & (df['RTwrtStim'] <= 1)]
    grouped = df.groupby(['ABL', 'absILD'])['RTwrtStim'].mean().reset_index()
    mean_rt_dict = {(row['ABL'], row['absILD']): row['RTwrtStim'] for _, row in grouped.iterrows()}
    return mean_rt_dict

# Compute and store mean RT for each batch, animal, (ABL, absILD)
mean_rt_by_batch_animal = {}
for batch, animal in batch_animal_pairs:
    mean_rt_by_batch_animal[(batch, animal)] = get_mean_RT_by_ABL_absILD(batch, animal)

print(f"Computed mean RT for {len(mean_rt_by_batch_animal)} batch-animal pairs.")
# Print a sample entry
if mean_rt_by_batch_animal:
    sample_key = next(iter(mean_rt_by_batch_animal))
    print(f"Sample for {sample_key}: {mean_rt_by_batch_animal[sample_key]}")

# %%
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from decoding_conf_make_chrono import mean_rt_by_batch_animal

# --- Chronometric plot: mean ± SEM across all animals for each (ABL, abs_ILD) ---
ABL_arr = [20, 40, 60]
abs_ILD_arr = [1, 2, 4, 8, 16]

# Prepare data structure: for each ABL and abs_ILD, collect all mean RTs across animals
all_rt = {(ABL, abs_ILD): [] for ABL in ABL_arr for abs_ILD in abs_ILD_arr}

for (batch, animal), rt_dict in mean_rt_by_batch_animal.items():
    for ABL in ABL_arr:
        for abs_ILD in abs_ILD_arr:
            key = (ABL, abs_ILD)
            if key in rt_dict:
                all_rt[key].append(rt_dict[key])

# Compute mean and SEM for each (ABL, abs_ILD)
mean_mat = np.full((len(ABL_arr), len(abs_ILD_arr)), np.nan)
sem_mat = np.full((len(ABL_arr), len(abs_ILD_arr)), np.nan)
for i, ABL in enumerate(ABL_arr):
    for j, abs_ILD in enumerate(abs_ILD_arr):
        vals = np.array(all_rt[(ABL, abs_ILD)])
        if len(vals) > 0:
            mean_mat[i, j] = np.nanmean(vals)
            sem_mat[i, j] = np.nanstd(vals, ddof=1) / np.sqrt(len(vals))

# %%
# Plotting
fig, ax = plt.subplots(figsize=(4, 3))
colors = ['C0', 'C1', 'C2']
for i, (ABL, color) in enumerate(zip(ABL_arr, colors)):
    ax.errorbar(abs_ILD_arr, mean_mat[i], yerr=sem_mat[i], marker='o', label=f'ABL {ABL} group mean', color=color, capsize=0)
# Increase label and tick font sizes
label_fontsize = 14
tick_fontsize = 12
ax.set_xlabel('|ILD|', fontsize=label_fontsize)
ax.set_ylabel('Mean RT', fontsize=label_fontsize)
# ax.set_title('Chronometric curve (mean ± SEM across animals)', fontsize=label_fontsize)
ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# Set custom y-ticks and limits
ax.set_yticks([0.15, 0.25, 0.35])
ax.set_ylim(0.1, 0.4)
# Set custom x-ticks and limits
ax.set_xticks([1, 2, 4, 8, 16])
ax.set_xlim(0.7, 16.3)
# Remove legend for publication style
# ax.legend().set_visible(False)  # Not needed, just don't call legend
plt.tight_layout()
plt.savefig('N_26_chronometric_curve.pdf')
plt.show()
