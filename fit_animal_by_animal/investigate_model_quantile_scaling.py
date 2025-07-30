# %%
#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
This script calculates theoretical RT quantiles for different models ('vanilla' or 'norm').

Set MODEL_TYPE at the top to switch between models.
- 'vanilla': uses vbmc_vanilla_tied_results from pickle, is_norm=False, rate_norm_l=0
- 'norm':    uses vbmc_norm_tied_results from pickle, is_norm=True, rate_norm_l from pickle

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from joblib import Parallel, delayed
from tqdm import tqdm
import pickle
import warnings
from types import SimpleNamespace
from animal_wise_plotting_utils import calculate_theoretical_curves
from time_vary_norm_utils import (
    up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn, 
    cum_pro_and_reactive_time_vary_fn, 
    rho_A_t_fn, 
    cum_A_t_fn
)
from collections import defaultdict


# In[2]:


# --- Model Type Declaration ---
MODEL_TYPE = 'norm' # or 'norm'
print(f"Processing MODEL_TYPE: {MODEL_TYPE}")


# In[3]:


# --- Get Batch-Animal Pairs ---
DESIRED_BATCHES = ['SD', 'LED34', 'LED6', 'LED8', 'LED7', 'LED34_even']
batch_dir = os.path.join(os.path.dirname(__file__), 'batch_csvs')
batch_files = [f'batch_{batch_name}_valid_and_aborts.csv' for batch_name in DESIRED_BATCHES]

# Check for file existence before attempting to read
existing_files = [os.path.join(batch_dir, fname) for fname in batch_files if os.path.exists(os.path.join(batch_dir, fname))]
if not existing_files:
    raise FileNotFoundError(f"No batch CSVs found in {batch_dir} for desired batches.")

merged_data = pd.concat([
    pd.read_csv(f) for f in existing_files
], ignore_index=True)

merged_valid = merged_data[merged_data['success'].isin([1, -1])].copy()
batch_animal_pairs = sorted(list(map(tuple, merged_valid[['batch_name', 'animal']].drop_duplicates().values)))

print(f"Found {len(batch_animal_pairs)} batch-animal pairs to process.")


# In[4]:


# --- Parameter and RTD Calculation Functions ---

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

def get_P_A_C_A(batch, animal_id, abort_params):
    N_theory = int(1e3)
    file_name = f'batch_csvs/batch_{batch}_valid_and_aborts.csv'
    df = pd.read_csv(file_name)
    df_animal = df[df['animal'] == animal_id]
    t_pts = np.arange(-2, 2, 0.001)
    P_A_mean, C_A_mean, t_stim_samples = calculate_theoretical_curves(
        df_animal, N_theory, t_pts, abort_params['t_A_aff'], abort_params['V_A'], abort_params['theta_A'], rho_A_t_fn
    )
    return P_A_mean, C_A_mean, t_stim_samples

def get_theoretical_RTD_from_params(P_A_mean, C_A_mean, t_stim_samples, abort_params, tied_params, rate_norm_l, is_norm, ABL, ILD, batch_name):
    phi_params_obj = np.nan
    K_max = 10
    if batch_name == 'LED34_even':
        T_trunc = 0.15
    else:
        T_trunc = 0.3
    t_pts = np.arange(-2, 2, 0.001)
    trunc_fac_samples = np.zeros((len(t_stim_samples)))
    Z_E = (tied_params['w'] - 0.5) * 2 * tied_params['theta_E']
    for idx, t_stim in enumerate(t_stim_samples):
        trunc_fac_samples[idx] = cum_pro_and_reactive_time_vary_fn(
            t_stim + 1, T_trunc,
            abort_params['V_A'], abort_params['theta_A'], abort_params['t_A_aff'],
            t_stim, ABL, ILD, tied_params['rate_lambda'], tied_params['T_0'], tied_params['theta_E'], Z_E, tied_params['t_E_aff'],
            phi_params_obj, rate_norm_l, 
            is_norm, False, K_max) \
            - cum_pro_and_reactive_time_vary_fn(
            t_stim, T_trunc,
            abort_params['V_A'], abort_params['theta_A'], abort_params['t_A_aff'],
            t_stim, ABL, ILD, tied_params['rate_lambda'], tied_params['T_0'], tied_params['theta_E'], Z_E, tied_params['t_E_aff'],
            phi_params_obj, rate_norm_l, 
            is_norm, False, K_max) + 1e-10
    trunc_factor = np.mean(trunc_fac_samples)
    up_mean = np.array([up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn(
        t, 1,
        P_A_mean[i], C_A_mean[i],
        ABL, ILD, tied_params['rate_lambda'], tied_params['T_0'], tied_params['theta_E'], Z_E, tied_params['t_E_aff'], tied_params['del_go'],
        phi_params_obj, rate_norm_l, 
        is_norm, False, K_max) for i, t in enumerate(t_pts)])
    down_mean = np.array([up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn(
        t, -1,
        P_A_mean[i], C_A_mean[i],
        ABL, ILD, tied_params['rate_lambda'], tied_params['T_0'], tied_params['theta_E'], Z_E, tied_params['t_E_aff'], tied_params['del_go'],
        phi_params_obj, rate_norm_l, 
        is_norm, False, K_max) for i, t in enumerate(t_pts)])
    mask_0_1 = (t_pts >= 0) & (t_pts <= 1)
    t_pts_0_1 = t_pts[mask_0_1]
    up_mean_0_1 = up_mean[mask_0_1]
    down_mean_0_1 = down_mean[mask_0_1]
    up_theory_mean_norm = up_mean_0_1 / trunc_factor
    down_theory_mean_norm = down_mean_0_1 / trunc_factor
    up_plus_down_mean = up_theory_mean_norm + down_theory_mean_norm
    return t_pts_0_1, up_plus_down_mean

def find_quantile_from_cdf(q, cdf, x_axis):
    """Inverts a CDF to find the value corresponding to a given quantile."""
    idx = np.searchsorted(cdf, q, side='left')
    if idx == 0:
        return x_axis[0]
    if idx == len(cdf):
        return x_axis[-1]
    x1, x2 = x_axis[idx - 1], x_axis[idx]
    y1, y2 = cdf[idx - 1], cdf[idx]
    if y2 == y1:
        return x1
    return x1 + (x2 - x1) * (q - y1) / (y2 - y1)


# In[5]:


# --- Main Processing Function ---

ABL_arr = [20, 40, 60]
ILD_arr = [-16., -8., -4., -2., -1., 1., 2., 4., 8., 16.]
# QUANTILES_TO_PLOT = [0.1, 0.3, 0.5, 0.7, 0.9]
QUANTILES_TO_PLOT = np.arange(0.1, 0.9, 0.05)


def get_animal_raw_RTs(batch_name, animal_id, ABL, ILD):
    """Fetches raw RTwrtStim from the dataframe for a given animal and stimulus condition."""
    file_name = f'batch_csvs/batch_{batch_name}_valid_and_aborts.csv'
    df = pd.read_csv(file_name)
    df_stim = df[(df['animal'] == animal_id) & (df['ABL'] == ABL) & (df['ILD'] == ILD) & (df['success'].isin([1, -1]))]
    # RTwrtStim is the column for reaction time
    df_stim = df_stim[(df_stim['RTwrtStim'] >= 0) & (df_stim['RTwrtStim'] <= 1)]
    return df_stim['RTwrtStim'].values

def process_animal_for_quantiles(batch_animal_pair):
    """Processes a single animal to get empirical and theoretical quantiles for all stimuli."""
    batch_name, animal_id = batch_animal_pair
    print(f"Processing {batch_name} / {animal_id}...")
    animal_quantile_data = {}
    try:
        abort_params, tied_params, rate_norm_l, is_norm = get_params_from_animal_pkl_file(batch_name, int(animal_id))
        p_a, c_a, ts_samp = get_P_A_C_A(batch_name, int(animal_id), abort_params)

        for abl in ABL_arr:
            for ild in ILD_arr:
                stim_key = (abl, ild)
                try:
                    t_pts, rtd = get_theoretical_RTD_from_params(
                        p_a, c_a, ts_samp, abort_params, tied_params, rate_norm_l, is_norm, abl, ild, batch_name
                    )
                    if np.all(np.isnan(rtd)) or len(t_pts) < 2:
                        raise ValueError("Theoretical RTD is all NaN or too short")
                    
                    cdf = np.cumsum(rtd) * (t_pts[1] - t_pts[0])
                    if cdf[-1] > 1e-6:
                        cdf /= cdf[-1] # Normalize
                    else:
                        raise ValueError("Theoretical CDF sum is close to zero")
                    
                    theo_quantiles = [find_quantile_from_cdf(q, cdf, t_pts) for q in QUANTILES_TO_PLOT]
                except Exception as e:
                    # print(f"  Warn: Theoretical quantiles failed for {stim_key}: {e}")
                    theo_quantiles = [np.nan] * len(QUANTILES_TO_PLOT)
                
                # --- Empirical Quantiles ---
                raw_rts = get_animal_raw_RTs(batch_name, int(animal_id), abl, ild)
                if len(raw_rts) > 5: # Need a few trials to be meaningful
                    emp_quantiles = np.quantile(raw_rts, QUANTILES_TO_PLOT)
                else:
                    emp_quantiles = [np.nan] * len(QUANTILES_TO_PLOT)

                animal_quantile_data[stim_key] = {
                    'empirical': emp_quantiles,
                    'theoretical': theo_quantiles
                }

    except Exception as e:
        print(f"ERROR processing animal {batch_name}/{animal_id}: {e}")
    
    return animal_quantile_data


# In[6]:


# --- Main Execution ---
n_jobs = max(1, os.cpu_count() - 1)
print(f"Running parallel processing with {n_jobs} jobs...")

all_animal_results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(process_animal_for_quantiles)(pair) for pair in batch_animal_pairs
)

print("\nCalculation finished.")
print(f"Processed {len(all_animal_results)} animals.")

# You can now inspect the `all_animal_results` variable
# For example, print the results for the first animal
if all_animal_results:
    print("\nExample result for the first animal:")
    first_result = all_animal_results[0]
    if first_result:
        for stim, data in list(first_result.items())[:5]: # Print first 5 stimuli
            print(f"  Stim {stim}: Quantiles={data['theoretical']}")


# %%
# for each abs ILD, plot quantiles of ABL 60 vs ABL 20, ABL 40 vs ABL 60

# %% 
# --- Data Restructuring and Analysis ---

from collections import defaultdict
import scipy.stats

# Restructure data: {abs_ild: {abl: {animal_idx: [quantiles]}}}
restructured_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

abs_ild_values = sorted(list(set(np.abs(ILD_arr))))
num_animals = len(all_animal_results)

for animal_idx, animal_data in enumerate(all_animal_results):
    if not animal_data: # Skip empty results
        continue
    for (abl, ild), quantiles_data in animal_data.items():
        abs_ild = abs(ild)
        # Average quantiles for +ILD and -ILD for each animal before storing
        restructured_data[abs_ild][abl][animal_idx].append(quantiles_data)

# Average the lists of quantiles if both +ILD and -ILD were present
for abs_ild in restructured_data:
    for abl in restructured_data[abs_ild]:
        for animal_idx in restructured_data[abs_ild][abl]:
            # It's a list of dicts. We need to average 'theoretical' and 'empirical' across the list.
            theo_quantiles_list = [d['theoretical'] for d in restructured_data[abs_ild][abl][animal_idx] if 'theoretical' in d]
            emp_quantiles_list = [d['empirical'] for d in restructured_data[abs_ild][abl][animal_idx] if 'empirical' in d]

            mean_theo_quantiles = np.nanmean(np.array(theo_quantiles_list), axis=0)
            mean_emp_quantiles = np.nanmean(np.array(emp_quantiles_list), axis=0)
            
            restructured_data[abs_ild][abl][animal_idx] = {
                'theoretical': mean_theo_quantiles,
                'empirical': mean_emp_quantiles
            }

# --- Plotting ---
# %%
fig, axes = plt.subplots(2, len(abs_ild_values), figsize=(25, 10), sharex=True, sharey=True)
fig.suptitle(f'Quantile-Quantile Plots (Model: {MODEL_TYPE})', fontsize=16)

for i, abs_ild in enumerate(abs_ild_values):
    ax_top = axes[0, i]
    ax_bottom = axes[1, i]

    ax_top.set_title(f'|ILD| = {abs_ild}')
    ax_top.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax_bottom.plot([0, 1], [0, 1], 'k--', alpha=0.5)

    # --- Common Data for x-axis (ABL 60) ---
    q_60_data_theo = np.array([restructured_data[abs_ild][60][j]['theoretical'] for j in range(num_animals) if 60 in restructured_data[abs_ild] and j in restructured_data[abs_ild][60] and 'theoretical' in restructured_data[abs_ild][60][j]])
    q_60_data_emp = np.array([restructured_data[abs_ild][60][j]['empirical'] for j in range(num_animals) if 60 in restructured_data[abs_ild] and j in restructured_data[abs_ild][60] and 'empirical' in restructured_data[abs_ild][60][j]])

    if q_60_data_theo.size == 0 and q_60_data_emp.size == 0:
        print(f"Skipping |ILD|={abs_ild} due to missing ABL 60 data.")
        ax_top.text(0.5, 0.5, 'No Data', ha='center', va='center')
        ax_bottom.text(0.5, 0.5, 'No Data', ha='center', va='center')
        continue

    mean_q_60_theo = np.nanmean(q_60_data_theo, axis=0)
    sem_q_60_theo = scipy.stats.sem(q_60_data_theo, axis=0, nan_policy='omit')
    mean_q_60_emp = np.nanmean(q_60_data_emp, axis=0)
    sem_q_60_emp = scipy.stats.sem(q_60_data_emp, axis=0, nan_policy='omit')

    # --- Top Row: ABL 20 vs 60 ---
    q_20_data_theo = np.array([restructured_data[abs_ild][20][j]['theoretical'] for j in range(num_animals) if 20 in restructured_data[abs_ild] and j in restructured_data[abs_ild][20] and 'theoretical' in restructured_data[abs_ild][20][j]])
    q_20_data_emp = np.array([restructured_data[abs_ild][20][j]['empirical'] for j in range(num_animals) if 20 in restructured_data[abs_ild] and j in restructured_data[abs_ild][20] and 'empirical' in restructured_data[abs_ild][20][j]])

    if q_20_data_theo.size > 0:
        mean_q_20_theo = np.nanmean(q_20_data_theo, axis=0)
        sem_q_20_theo = scipy.stats.sem(q_20_data_theo, axis=0, nan_policy='omit')
        ax_top.plot(mean_q_60_theo, mean_q_20_theo, 'o', label='Theo', alpha=0.8, color='red')

    if q_20_data_emp.size > 0:
        mean_q_20_emp = np.nanmean(q_20_data_emp, axis=0)
        sem_q_20_emp = scipy.stats.sem(q_20_data_emp, axis=0, nan_policy='omit')
        ax_top.plot(mean_q_60_emp, mean_q_20_emp, 'o', label='Data', alpha=0.8, color='blue', markerfacecolor='none')

    # --- Bottom Row: ABL 40 vs 60 ---
    q_40_data_theo = np.array([restructured_data[abs_ild][40][j]['theoretical'] for j in range(num_animals) if 40 in restructured_data[abs_ild] and j in restructured_data[abs_ild][40] and 'theoretical' in restructured_data[abs_ild][40][j]])
    q_40_data_emp = np.array([restructured_data[abs_ild][40][j]['empirical'] for j in range(num_animals) if 40 in restructured_data[abs_ild] and j in restructured_data[abs_ild][40] and 'empirical' in restructured_data[abs_ild][40][j]])

    if q_40_data_theo.size > 0:
        mean_q_40_theo = np.nanmean(q_40_data_theo, axis=0)
        sem_q_40_theo = scipy.stats.sem(q_40_data_theo, axis=0, nan_policy='omit')
        ax_bottom.plot(mean_q_60_theo, mean_q_40_theo, 'o', label='Theo', alpha=0.8, color='red')

    if q_40_data_emp.size > 0:
        mean_q_40_emp = np.nanmean(q_40_data_emp, axis=0)
        sem_q_40_emp = scipy.stats.sem(q_40_data_emp, axis=0, nan_policy='omit')
        ax_bottom.plot(mean_q_60_emp, mean_q_40_emp, 'o', label='Data', alpha=0.8, color='blue', markerfacecolor='none')

    # --- Labels and Legends ---
    ax_bottom.set_xlabel('RT for ABL 60')
    # ax_top.grid(True, linestyle='--', alpha=0.6)
    # ax_bottom.grid(True, linestyle='--', alpha=0.6)
    # ax_top.legend()
    # ax_bottom.legend()
    
    # Set x-axis limits
    ax_top.set_xlim(0, 0.35)
    ax_bottom.set_xlim(0, 0.35)
    ax_top.set_ylim(0, 0.6)
    ax_bottom.set_ylim(0, 0.6)
    if i == 0:
        ax_top.set_ylabel('RT for ABL 20')
        ax_bottom.set_ylabel('RT for ABL 40')
        

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()