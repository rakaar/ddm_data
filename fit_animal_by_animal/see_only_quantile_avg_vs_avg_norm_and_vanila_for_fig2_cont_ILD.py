# %%
"""
Unified RTD analysis script for vanilla and normalized TIED models.

Set MODEL_TYPE = 'vanilla' or 'norm' at the top to switch between models.
- 'vanilla': uses vbmc_vanilla_tied_results from pickle, is_norm=False, rate_norm_l=0
- 'norm':    uses vbmc_norm_tied_results from pickle, is_norm=True, rate_norm_l from pickle

All downstream logic is automatically adjusted based on this flag.
"""
# %%
MODEL_TYPE = 'norm'
print(f"Processing MODEL_TYPE: {MODEL_TYPE}")


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
    cum_pro_and_reactive_time_vary_fn, 
    rho_A_t_fn, 
    cum_A_t_fn
)
from collections import defaultdict
import random
from scipy.stats import gaussian_kde


# %%
# Define desired batches
DESIRED_BATCHES = ['SD', 'LED34', 'LED6', 'LED8', 'LED7', 'LED34_even']
batch_dir = os.path.join(os.path.dirname(__file__), 'batch_csvs')
batch_files = [f'batch_{batch_name}_valid_and_aborts.csv' for batch_name in DESIRED_BATCHES]

merged_data = pd.concat([
    pd.read_csv(os.path.join(batch_dir, fname)) for fname in batch_files if os.path.exists(os.path.join(batch_dir, fname))
], ignore_index=True)

merged_valid = merged_data[merged_data['success'].isin([1, -1])].copy()

# --- Print animal table ---
batch_animal_pairs = sorted(list(map(tuple, merged_valid[['batch_name', 'animal']].drop_duplicates().values)))

print(f"Found {len(batch_animal_pairs)} batch-animal pairs from {len(set(p[0] for p in batch_animal_pairs))} batches:")

if batch_animal_pairs:
    batch_to_animals = defaultdict(list)
    for batch, animal in batch_animal_pairs:
        # Ensure animal is a string and we don't add duplicates
        animal_str = str(animal)
        if animal_str not in batch_to_animals[batch]:
            batch_to_animals[batch].append(animal_str)

    # Determine column widths for formatting
    max_batch_len = max(len(b) for b in batch_to_animals.keys()) if batch_to_animals else 0
    animal_strings = {b: ', '.join(sorted(a)) for b, a in batch_to_animals.items()}
    max_animals_len = max(len(s) for s in animal_strings.values()) if animal_strings else 0

    # Header
    print(f"{'Batch':<{max_batch_len}}  {'Animals'}")
    print(f"{'=' * max_batch_len}  {'=' * max_animals_len}")

    # Rows
    for batch in sorted(animal_strings.keys()):
        animals_str = animal_strings[batch]
        print(f"{batch:<{max_batch_len}}  {animals_str}")

# remove SD 49 due to issue in sensory delay
# batch_animal_pairs = [(batch, animal) for batch, animal in batch_animal_pairs if not (batch == 'SD' and animal == '49')]
# print(f"Removed SD 49 due to issue in sensory delay. Found {len(batch_animal_pairs)} batch-animal pairs: {batch_animal_pairs}")

def get_animal_RTD_data(batch_name, animal_id, ABL, ILD, bins):
    file_name = f'batch_csvs/batch_{batch_name}_valid_and_aborts.csv'
    df = pd.read_csv(file_name)
    df = df[(df['animal'] == animal_id) & (df['ABL'] == ABL) & (df['ILD'] == ILD) & (df['success'].isin([1, -1]))]

    bin_centers = (bins[:-1] + bins[1:]) / 2
    if df.empty:
        print(f"No data found for batch {batch_name}, animal {animal_id}, ABL {ABL}, ILD {ILD}. Returning NaNs.")
        rtd_hist = np.full_like(bin_centers, np.nan)
        return bin_centers, rtd_hist
    df = df[(df['RTwrtStim'] >= 0) & (df['RTwrtStim'] <= 1)]
    if len(df) == 0:
        print(f"No trials with RTwrtStim <= 1 for batch {batch_name}, animal {animal_id}, ABL {ABL}, ILD {ILD}. Returning NaNs.")
        rtd_hist = np.full_like(bin_centers, np.nan)
        return bin_centers, rtd_hist
    rtd_hist, _ = np.histogram(df['RTwrtStim'], bins=bins, density=True)
    return bin_centers, rtd_hist

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

from scipy.stats import sem
import pickle

# %% 
# Main analysis loop

ABL_arr = [20, 40, 60]
ILD_arr = [-16., -8., -4., -2., -1., 1., 2., 4., 8., 16.]
QUANTILES_TO_PLOT = [0.1, 0.3, 0.5, 0.7, 0.9]

def get_animal_raw_RTs(batch_name, animal_id, ABL, ILD):
    """Fetches raw RTwrtStim for a given animal and stimulus condition."""
    file_name = f'batch_csvs/batch_{batch_name}_valid_and_aborts.csv'
    df = pd.read_csv(file_name)
    df_stim = df[(df['animal'] == animal_id) & (df['ABL'] == ABL) & (df['ILD'] == ILD) & (df['success'].isin([1, -1]))]
    df_stim = df_stim[(df_stim['RTwrtStim'] >= 0) & (df_stim['RTwrtStim'] <= 1)]
    return df_stim['RTwrtStim'].values

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

def process_animal_for_quantiles(batch_animal_pair):
    """Processes a single animal to get empirical and theoretical quantiles.
    
    Empirical quantiles are calculated for discrete ILD values present in the data.
    Theoretical quantiles are calculated for continuous ILD values from -16 to 16 in steps of 0.1.
    """
    batch_name, animal_id = batch_animal_pair
    print(f"Processing {batch_name} / {animal_id}...")
    animal_quantile_data = {}
    try:
        abort_params, tied_params, rate_norm_l, is_norm = get_params_from_animal_pkl_file(batch_name, int(animal_id))
        p_a, c_a, ts_samp = get_P_A_C_A(batch_name, int(animal_id), abort_params)

        # --- Empirical Quantiles (discrete ILD values from data) ---
        for abl in ABL_arr:
            for ild in ILD_arr:
                stim_key = (abl, ild)
                
                raw_rts = get_animal_raw_RTs(batch_name, int(animal_id), abl, ild)
                if len(raw_rts) > 5: # Need a few trials to be meaningful
                    emp_quantiles = np.quantile(raw_rts, QUANTILES_TO_PLOT)
                else:
                    emp_quantiles = [np.nan] * len(QUANTILES_TO_PLOT)
                
                animal_quantile_data[stim_key] = {
                    'empirical': emp_quantiles,
                }
        
        # --- Theoretical Quantiles (continuous ILD values) ---
        # Create continuous ILD array from -16 to 16 in steps of 0.1
        continuous_ild_values = np.arange(-16.0, 16.1, 0.1)
        
        for abl in ABL_arr:
            for ild in continuous_ild_values:
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
                    
                    # Store continuous theoretical quantiles with a special key
                    continuous_key = (abl, ild, 'continuous')
                    animal_quantile_data[continuous_key] = {
                        'theoretical': theo_quantiles
                    }
                    
                    # Also update discrete ILD entries if they match (within tolerance)
                    for discrete_ild in ILD_arr:
                        if abs(ild - discrete_ild) < 0.05:  # Match within 0.05 tolerance
                            discrete_key = (abl, discrete_ild)
                            if discrete_key in animal_quantile_data:
                                animal_quantile_data[discrete_key]['theoretical'] = theo_quantiles
                            break
                    
                except Exception as e:
                    # Silently skip failed theoretical quantiles for continuous values
                    pass
                    
    except Exception as e:
        print(f"ERROR processing animal {batch_name}/{animal_id}: {e}")
    
    return animal_quantile_data

# %% 
# LONG TIME TAKING --- Main Execution ---

n_jobs = max(1, os.cpu_count() - 1)
print(f"Running parallel processing with {n_jobs} jobs...")

all_animal_results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(process_animal_for_quantiles)(pair) for pair in batch_animal_pairs
)

# %% 
# === Debugging: inspect stored theoretical stim keys ===
# Rounding off messes things up, so have to round off
# Example of issue:
# First 10 negative ILDs (raw):  [-16.0, -15.9, -15.8, -15.700000000000001, -15.600000000000001, -15.500000000000002, -15.400000000000002, -15.300000000000002, -15.200000000000003, -15.100000000000003]
# First 10 negative ILDs (rounded): [-16.0, -15.9, -15.8, -15.7, -15.6, -15.5, -15.4, -15.3, -15.2, -15.1]
# Last 10 positive ILDs  (raw):  [15.099999999999888, 15.19999999999989, 15.29999999999989, 15.399999999999888, 15.499999999999886, 15.599999999999888, 15.69999999999989, 15.799999999999887, 15.899999999999885, 15.999999999999886]
# Last 10 positive ILDs  (rounded): [15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 15.7, 15.8, 15.9, 16.0]



# Inspect keys for one ABL: show positive and negative ILD entries separately
abl_to_inspect = ABL_arr[0] if len(ABL_arr) else None
if abl_to_inspect is not None:
    # take first non-empty animal
    for animal_data in all_animal_results:
        if not animal_data:
            continue
        ild_vals = sorted([k[1] for k in animal_data.keys()
                           if isinstance(k, tuple) and len(k)==3
                           and k[2]=='continuous' and k[0]==abl_to_inspect])
        neg_vals = [v for v in ild_vals if v < 0][:10]
        pos_vals = [v for v in ild_vals if v > 0][-10:]
        neg_vals_round = [round(v,1) for v in neg_vals]
        pos_vals_round = [round(v,1) for v in pos_vals]
        print(f"\nDetailed key sample for ABL {abl_to_inspect} (animal):")
        print("First 10 negative ILDs (raw): ", neg_vals)
        print("First 10 negative ILDs (rounded):", neg_vals_round)
        print("Last 10 positive ILDs  (raw): ", pos_vals)
        print("Last 10 positive ILDs  (rounded):", pos_vals_round)
        break

# %%
# data aggregate for plot
print("Aggregating data for plotting...")
abs_ild_sorted = sorted(list(set(abs(ild) for ild in ILD_arr)))

# Create continuous abs_ild array for theoretical curves (absolute values only for plotting)
continuous_abs_ild = np.round(np.arange(1.0, 16.1, 0.1), 1)

def _create_innermost_dict():
    return {'empirical': [], 'theoretical': []}

def _create_inner_defaultdict():
    return defaultdict(_create_innermost_dict)

# Discrete data for empirical quantiles
plot_data = defaultdict(_create_inner_defaultdict)

# Continuous data for theoretical quantiles
continuous_plot_data = defaultdict(_create_inner_defaultdict)

for animal_data in all_animal_results:
    if not animal_data: continue
    
    # Process discrete empirical data
    for abl in ABL_arr:
        for abs_ild in abs_ild_sorted:
            # Combine data from ILD and -ILD
            emp_quantiles_combined = []
            for ild_sign in [abs_ild, -abs_ild]:
                stim_key = (abl, ild_sign)
                if stim_key in animal_data:
                    emp_quantiles_combined.append(animal_data[stim_key]['empirical'])
            # Average empirical quantiles across ±ILD for this animal
            if emp_quantiles_combined:
                plot_data[abl][abs_ild]['empirical'].append(np.nanmean(emp_quantiles_combined, axis=0))
    
    # Process continuous theoretical data (iterate over keys to avoid float mismatch)
    for abl in ABL_arr:
        # select this animal's continuous keys for current ABL
        abl_cont_keys = [k for k in animal_data.keys() if isinstance(k, tuple) and len(k)==3 and k[2]=='continuous' and k[0]==abl]
        for key in abl_cont_keys:
            ild_val = key[1]
            rounded_abs = np.round(abs(ild_val), 1)  # plotting key
            continuous_plot_data[abl][rounded_abs]['theoretical'].append(animal_data[key]['theoretical'])


# %% --- Plotting: ABLs separate ---
print("Generating quantile plot (separate ABLs)...")
fig, axes = plt.subplots(1, len(ABL_arr), figsize=(12, 4), sharey=True)
if len(ABL_arr) == 1: axes = [axes] # Ensure axes is always iterable

for i, abl in enumerate(ABL_arr):
    ax = axes[i]
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for q_idx, q in enumerate(QUANTILES_TO_PLOT):
        # Empirical data (discrete points)
        emp_means, emp_sems = [], []
        for abs_ild in abs_ild_sorted:
            emp_quantiles_across_animals = np.array(plot_data[abl][abs_ild]['empirical'])[:, q_idx]
            emp_means.append(np.nanmean(emp_quantiles_across_animals))
            emp_sems.append(sem(emp_quantiles_across_animals, nan_policy='omit'))
        
        # Theoretical data (continuous curve)
        theo_means, theo_sems = [], []
        continuous_abs_ild_valid = []
        for abs_ild in continuous_abs_ild:
            if len(continuous_plot_data[abl][abs_ild]['theoretical']) > 0:
                theo_quantiles_across_animals = np.array(continuous_plot_data[abl][abs_ild]['theoretical'])[:, q_idx]
                theo_means.append(np.nanmean(theo_quantiles_across_animals))
                theo_sems.append(sem(theo_quantiles_across_animals, nan_policy='omit'))
                continuous_abs_ild_valid.append(abs_ild)

        # Plot empirical with error bars (discrete points)
        ax.errorbar(abs_ild_sorted, emp_means, yerr=emp_sems, fmt='o-', color='b', markersize=4, capsize=3, label='Data' if q_idx == 0 else "")
        
        # Plot theoretical as smooth curve (continuous)
        if len(continuous_abs_ild_valid) > 0:
            ax.plot(continuous_abs_ild_valid, theo_means, '-', color='r', linewidth=1.5, label='Theory' if q_idx == 0 else "")
            # SEM shading around theoretical curve
            ax.fill_between(continuous_abs_ild_valid,
                             np.array(theo_means) - np.array(theo_sems),
                             np.array(theo_means) + np.array(theo_sems),
                             color='r', alpha=0.2, linewidth=0)

    ax.set_title(f'ABL = {abl}')
    ax.set_xlabel('|ILD| (dB)')
    ax.set_xscale('log', base=2)
    ax.set_xticks(abs_ild_sorted)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

axes[0].set_ylabel('RT Quantile (s)')
# fig.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95))
plt.tight_layout(rect=[0, 0, 0.95, 1])
plt.savefig(f'quantile_plot_separate_abls_{MODEL_TYPE}.png', dpi=300)
plt.show()

# %% --- Plotting: Aggregating across ABLs (Concatenation) ---
# Fig 2

# We will populate quantile_summary after computing aggregated means/SEMs below.
quantile_summary = []

LABEL_FONTSIZE: int = 25
TICK_FONTSIZE: int = 24

fig, ax = plt.subplots(1, 1, figsize=(6, 5))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for q_idx, q in enumerate(QUANTILES_TO_PLOT):
    # Empirical data (discrete points)
    emp_means, emp_sems = [], []
    for abs_ild in abs_ild_sorted:
        # Aggregate across all ABLs for this ILD and quantile
        all_abl_emp_quantiles = np.concatenate([
            np.array(plot_data[abl][abs_ild]['empirical'])[:, q_idx] for abl in ABL_arr
        ])
        emp_means.append(np.nanmean(all_abl_emp_quantiles))
        emp_sems.append(sem(all_abl_emp_quantiles, nan_policy='omit'))
    
    # Theoretical data (continuous curve)
    theo_means, theo_sems = [], []
    continuous_abs_ild_valid = []
    for abs_ild in continuous_abs_ild:
        # Aggregate across all ABLs for this continuous ILD and quantile
        all_abl_theo_quantiles = []
        for abl in ABL_arr:
            if len(continuous_plot_data[abl][abs_ild]['theoretical']) > 0:
                all_abl_theo_quantiles.extend(np.array(continuous_plot_data[abl][abs_ild]['theoretical'])[:, q_idx])
        
        if len(all_abl_theo_quantiles) > 0:
            theo_means.append(np.nanmean(all_abl_theo_quantiles))
            theo_sems.append(sem(all_abl_theo_quantiles, nan_policy='omit'))
            continuous_abs_ild_valid.append(abs_ild)

    # Save aggregated statistics for this quantile
    quantile_summary.append({
        'q': q,
        'emp_abs_ild': abs_ild_sorted,
        'emp_means': emp_means,
        'emp_sems': emp_sems,
        'theo_abs_ild': continuous_abs_ild_valid,
        'theo_means': theo_means,
        'theo_sems': theo_sems
    })

    # Plot empirical with error bars (discrete points)
    ax.errorbar(abs_ild_sorted, emp_means, yerr=emp_sems, fmt='o-', color='black', markersize=4, capsize=0)
    
    # Plot theoretical as smooth curve (continuous)
    if len(continuous_abs_ild_valid) > 0:
        ax.plot(continuous_abs_ild_valid, theo_means, '-', color='tab:red', linewidth=1.5)
        # SEM shading around theoretical curve
        ax.fill_between(continuous_abs_ild_valid,
                         np.array(theo_means) - np.array(theo_sems),
                         np.array(theo_means) + np.array(theo_sems),
                         color='tab:red', alpha=0.2, linewidth=0)

ax.set_xlabel('|ILD| (dB)', fontsize=LABEL_FONTSIZE)
ax.set_ylabel('RT Quantile (s)', fontsize=LABEL_FONTSIZE)
ax.set_xscale('log', base=2)
ax.set_xticks(abs_ild_sorted)
ax.set_yticks([0.1, 0.2, 0.3, 0.4])
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)

plt.tight_layout()
plt.savefig(f'quantile_plot_with_errorbars_{MODEL_TYPE}.png', dpi=300)
plt.show()

# ---------------- Save all necessary data for Fig 2 ---------------- #
quantile_plot_data = {
    'plot_data': plot_data,
    'continuous_plot_data': continuous_plot_data,
    'quantile_summary': quantile_summary,
    'QUANTILES_TO_PLOT': QUANTILES_TO_PLOT,
    'abs_ild_sorted': abs_ild_sorted,
    'continuous_abs_ild': continuous_abs_ild,
    'ABL_arr': ABL_arr,
    'MODEL_TYPE': MODEL_TYPE
}
with open(f'{MODEL_TYPE}_quant_fig2_data.pkl', 'wb') as f:
    pickle.dump(quantile_plot_data, f)


# %% --- Plotting: Averaging the ABL means ---

# To average the means from each ABL, we also need to propagate the standard errors.
# The standard error of an average of N means (m_i) with individual standard errors (s_i) is:
# SEM_avg = (1/N) * sqrt(s_1^2 + s_2^2 + ... + s_N^2)

# fig, ax = plt.subplots(1, 1, figsize=(6, 5))
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# for q_idx, q in enumerate(QUANTILES_TO_PLOT):
#     avg_emp_means, prop_emp_sems = [], []
#     avg_theo_means, prop_theo_sems = [], []

#     for abs_ild in abs_ild_sorted:
#         # For each ILD, get the mean and sem for each ABL
#         abl_emp_means, abl_emp_sems = [], []
#         abl_theo_means, abl_theo_sems = [], []

#         for abl in ABL_arr:
#             emp_quantiles = np.array(plot_data[abl][abs_ild]['empirical'])[:, q_idx]
#             theo_quantiles = np.array(plot_data[abl][abs_ild]['theoretical'])[:, q_idx]

#             abl_emp_means.append(np.nanmean(emp_quantiles))
#             abl_emp_sems.append(sem(emp_quantiles, nan_policy='omit'))
            
#             abl_theo_means.append(np.nanmean(theo_quantiles))
#             abl_theo_sems.append(sem(theo_quantiles, nan_policy='omit'))

#         # Average the means and propagate the SEMs
#         avg_emp_means.append(np.nanmean(abl_emp_means))
#         prop_emp_sems.append(np.sqrt(np.nansum(np.square(abl_emp_sems))) / len(ABL_arr))

#         avg_theo_means.append(np.nanmean(abl_theo_means))
#         prop_theo_sems.append(np.sqrt(np.nansum(np.square(abl_theo_sems))) / len(ABL_arr))

#     # Plot empirical with error bars
#     ax.errorbar(abs_ild_sorted, avg_emp_means, yerr=prop_emp_sems, fmt='o-', color='b', markersize=4, capsize=3, label='Data' if q_idx == 0 else "")
#     # Plot theoretical with error bars
#     ax.errorbar(abs_ild_sorted, avg_theo_means, yerr=prop_theo_sems, fmt='^-', color='r', markersize=4, capsize=3, label='Theory' if q_idx == 0 else "")

# ax.set_title('RT Quantiles (Averaged ABL Means)')
# ax.set_xlabel('|ILD| (dB)')
# ax.set_ylabel('RT Quantile (s)')
# ax.set_xscale('log', base=2)
# ax.set_xticks(abs_ild_sorted)
# ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

# fig.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95))
# plt.tight_layout(rect=[0, 0, 0.9, 1])
# plt.savefig(f'quantile_plot_avg_of_means_{MODEL_TYPE}.png', dpi=300)
# plt.show()

# %%
