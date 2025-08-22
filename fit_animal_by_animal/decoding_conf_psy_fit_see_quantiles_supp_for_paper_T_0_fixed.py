# %%
"""
Unified QUANTILE analysis script for vanilla and normalized TIED models.

Set MODEL_TYPE = 'vanilla' or 'norm' at the top to switch between models.
- 'vanilla': uses vbmc_vanilla_tied_results from pickle, is_norm=False, rate_norm_l=0
- 'norm':    uses vbmc_norm_tied_results from pickle, is_norm=True, rate_norm_l from pickle

PARAM_SOURCE controls where tied parameters come from:
- 'results': read tied param means/samples from results_{batch}_{animal}.pkl
- 'psycho' : read 3-parameter VBMC samples (T_0 fixed externally) from psycho_fits_T_0_fixed_from_vanilla
In both cases, abort params and fixed T_0, t_E_aff, del_go are read from VANILLA results PKL only.

This script computes empirical RT quantiles directly from raw RTs, and theoretical
quantiles by building the CDF from the theoretical RTD and inverting it.
It produces:
- 1x3 ABL panels: 5 quantiles vs |ILD| with mean±SEM across animals (Data vs Theory)
- Fig-2 style single panel: aggregated across ABLs (Data vs Theory)
"""
# %%
MODEL_TYPE = 'vanilla'
# Parameter source: 'results' or 'psycho'
PARAM_SOURCE = 'psycho'
print(f"Processing MODEL_TYPE: {MODEL_TYPE}, PARAM_SOURCE: {PARAM_SOURCE}")
# NOTE: V2 is being used

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from joblib import Parallel, delayed
import pickle
from animal_wise_plotting_utils import calculate_theoretical_curves
from time_vary_norm_utils import (
    up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn, 
    up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn_vec,
    cum_pro_and_reactive_time_vary_fn, 
    rho_A_t_fn, 
    cum_A_t_fn
)
from collections import defaultdict
from scipy.stats import sem

# def get_simulation_RTD_KDE(
#     abort_params, tied_params, rate_norm_l, Z_E, ABL, ILD, t_stim_samples, N_sim, N_print, dt, n_jobs=30
# ):
#     """
#     Run the simulation for given parameters and return KDE arrays for RTD.
#     Returns: x_vals, kde_vals
#     """
#     sim_results = Parallel(n_jobs=n_jobs)(
#         delayed(psiam_tied_data_gen_wrapper_rate_norm_fn)(
#             abort_params['V_A'], abort_params['theta_A'], ABL, ILD, tied_params['rate_lambda'], tied_params['T_0'],
#             tied_params['theta_E'], Z_E, abort_params['t_A_aff'], tied_params['t_E_aff'], tied_params['del_go'],
#             t_stim_samples[iter_num], rate_norm_l, iter_num, N_print, dt
#         ) for iter_num in range(N_sim)
#     )
#     sim_results_df = pd.DataFrame(sim_results)
#     sim_results_df_valid = sim_results_df[sim_results_df['rt'] - sim_results_df['t_stim'] > -0.1]
#     sim_results_df_valid_lt_1 = sim_results_df_valid[sim_results_df_valid['rt'] - sim_results_df_valid['t_stim'] <= 1]
#     sim_rt = sim_results_df_valid_lt_1['rt'] - sim_results_df_valid_lt_1['t_stim']
#     kde = gaussian_kde(sim_rt)

def load_and_merge_batches(batches):
    dfs = []
    for b in batches:
        fn = f'batch_csvs/batch_{b}_valid_and_aborts.csv'
        if not os.path.exists(fn):
            print(f"Warning: missing {fn}; skipping")
            continue
        df = pd.read_csv(fn)
        df['batch_name'] = b
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError("No batch CSVs found. Check DESIRED_BATCHES and paths.")
    return pd.concat(dfs, ignore_index=True)

# Choose batches to include
DESIRED_BATCHES = ['SD', 'LED34', 'LED6', 'LED8', 'LED7', 'LED34_even']
merged_data = load_and_merge_batches(DESIRED_BATCHES)
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

# %%
# (RTD histogram helper removed; not used in quantile workflow)

psycho_fits_repo_path = '/home/rlab/raghavendra/ddm_data/fit_valid_trials/psycho_fits_T_0_fixed_from_vanilla/'

def get_psycho_params(batch_name, animal_id):
    filename = os.path.join(psycho_fits_repo_path, f'psycho_fit_3-params-T_0_fixed_from_vanilla_{batch_name}_{animal_id}.pkl')
    with open(filename, 'rb') as f:
        vbmc_obj = pickle.load(f)
    vp = vbmc_obj.vp
    samples = vp.sample(int(1e6))[0]
    print(f"Loaded 3-param VBMC samples for {batch_name}, {animal_id}: shape {samples.shape}")
    tied_params = {
        'rate_lambda' : samples[:,0].mean(),
        'theta_E' : samples[:,1].mean(),
        'w' : samples[:,2].mean(),
    }
    return tied_params


def get_psycho_vp_samples(batch_name, animal_id, n_samples):
    """Sample n_samples parameter draws from the psycho-fit 3-param VBMC VP (T_0 fixed externally).
    Returns a dict with arrays for keys: rate_lambda, theta_E, w
    """
    filename = os.path.join(psycho_fits_repo_path, f'psycho_fit_3-params-T_0_fixed_from_vanilla_{batch_name}_{animal_id}.pkl')
    with open(filename, 'rb') as f:
        vbmc_obj = pickle.load(f)
    vp = vbmc_obj.vp
    samples = vp.sample(n_samples)[0]
    return {
        'rate_lambda': samples[:, 0],
        'theta_E': samples[:, 1],
        'w': samples[:, 2],
    }


def get_fixed_params_from_vanilla(fit_results_data, batch_name, animal_id):
    """Fetch fixed T_0, t_E_aff and del_go from vanilla tied results only."""
    vanilla_keyname = "vbmc_vanilla_tied_results"
    if vanilla_keyname not in fit_results_data:
        raise KeyError(f"Missing required key in results_{batch_name}_animal_{animal_id}.pkl: {vanilla_keyname}")
    vanilla_tied = fit_results_data[vanilla_keyname]
    fixed_T_0 = np.mean(vanilla_tied['T_0_samples'])
    fixed_t_E_aff = np.mean(vanilla_tied['t_E_aff_samples'])
    fixed_del_go = np.mean(vanilla_tied['del_go_samples'])
    print(f"Using fixed (vanilla) T_0={fixed_T_0:.6f}, t_E_aff={fixed_t_E_aff:.6f}, del_go={fixed_del_go:.6f} for batch {batch_name}, animal {animal_id}")
    return fixed_T_0, fixed_t_E_aff, fixed_del_go


def get_params_from_animal_pkl_file(batch_name, animal_id):
    pkl_file = f'results_{batch_name}_animal_{animal_id}.pkl'
    with open(pkl_file, 'rb') as f:
        fit_results_data = pickle.load(f)
    vbmc_aborts_param_keys_map = {
        'V_A_samples': 'V_A',
        'theta_A_samples': 'theta_A',
        't_A_aff_samp': 't_A_aff'
    }
    abort_keyname = "vbmc_aborts_results"
    abort_params = {}
    
    if abort_keyname in fit_results_data:
        abort_samples = fit_results_data[abort_keyname]
        for param_samples_name, param_label in vbmc_aborts_param_keys_map.items():
            abort_params[param_label] = np.mean(abort_samples[param_samples_name])
    
    # Load fixed T_0, t_E_aff and del_go from vanilla tied results
    fixed_T_0, fixed_t_E_aff, fixed_del_go = get_fixed_params_from_vanilla(fit_results_data, batch_name, animal_id)

    # Choose source of tied parameters based on MODEL_TYPE
    if MODEL_TYPE == 'vanilla':
        source_key = 'vbmc_vanilla_tied_results'
        is_norm = False
        rate_norm_l = 0.0
    elif MODEL_TYPE == 'norm':
        source_key = 'vbmc_norm_tied_results'
        is_norm = True
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

    tied_params = {}
    if PARAM_SOURCE == 'results':
        if source_key not in fit_results_data:
            raise KeyError(f"Missing {source_key} in results_{batch_name}_animal_{animal_id}.pkl")
        model_samples = fit_results_data[source_key]
        tied_map = {
            'rate_lambda_samples': 'rate_lambda',
            'theta_E_samples': 'theta_E',
            'w_samples': 'w',
        }
        for s_key, label in tied_map.items():
            if s_key not in model_samples:
                raise KeyError(f"Key '{s_key}' not found in {source_key} for batch {batch_name}, animal {animal_id}")
            tied_params[label] = float(np.mean(np.asarray(model_samples[s_key])))
        if MODEL_TYPE == 'norm':
            if 'rate_norm_l_samples' not in model_samples:
                raise KeyError(f"Key 'rate_norm_l_samples' not found in {source_key} for batch {batch_name}, animal {animal_id}")
            rate_norm_l = float(np.mean(np.asarray(model_samples['rate_norm_l_samples'])))
            print(f"Using rate_norm_l={rate_norm_l:.6f} for batch {batch_name}, animal {animal_id}")
    elif PARAM_SOURCE == 'psycho':
        # Use psycho-fit 3-param (T_0 fixed externally) VBMC means for tied params
        psycho_means = get_psycho_params(batch_name, animal_id)
        tied_params.update(psycho_means)
        # For normalized model, we still need rate_norm_l from results PKL
        if MODEL_TYPE == 'norm':
            if source_key not in fit_results_data:
                raise KeyError(f"Missing {source_key} (for rate_norm_l) in results_{batch_name}_animal_{animal_id}.pkl")
            model_samples = fit_results_data[source_key]
            if 'rate_norm_l_samples' not in model_samples:
                raise KeyError(f"Key 'rate_norm_l_samples' not found in {source_key} for batch {batch_name}, animal {animal_id}")
            rate_norm_l = float(np.mean(np.asarray(model_samples['rate_norm_l_samples'])))
            print(f"Using rate_norm_l (from results) = {rate_norm_l:.6f} for batch {batch_name}, animal {animal_id}")
    else:
        raise ValueError(f"Unknown PARAM_SOURCE: {PARAM_SOURCE}")

    # Attach fixed T_0, t_E_aff and del_go
    tied_params['T_0'] = fixed_T_0
    tied_params['t_E_aff'] = fixed_t_E_aff
    tied_params['del_go'] = fixed_del_go

    return abort_params, tied_params, rate_norm_l, is_norm


def get_P_A_C_A(batch, animal_id, abort_params):
    N_theory = int(1e3)
    file_name = f'batch_csvs/batch_{batch}_valid_and_aborts.csv'
    df = pd.read_csv(file_name)
    df_animal = df[df['animal'] == int(animal_id)]
    t_pts = np.arange(-2, 2, 0.001)
    P_A_mean, C_A_mean, t_stim_samples = calculate_theoretical_curves(
        df_animal, N_theory, t_pts, abort_params['t_A_aff'], abort_params['V_A'], abort_params['theta_A'], rho_A_t_fn
    )
    return P_A_mean, C_A_mean, t_stim_samples

def get_theoretical_RTD_from_params(batch_name, P_A_mean, C_A_mean, t_stim_samples, abort_params, tied_params, rate_norm_l, is_norm, ABL, ILD):
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
    # Use vectorized version for better performance
    up_mean = up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn_vec(
        t_pts, 1,
        P_A_mean, C_A_mean,
        ABL, ILD, tied_params['rate_lambda'], tied_params['T_0'], tied_params['theta_E'], Z_E, tied_params['t_E_aff'], tied_params['del_go'],
        np.nan, np.nan, np.nan, np.nan, np.nan,  # int_phi_t_E_g, phi_t_e, int_phi_t_e, int_phi_t2, int_phi_t1
        rate_norm_l, is_norm, False, K_max
    )
    down_mean = up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn_vec(
        t_pts, -1,
        P_A_mean, C_A_mean,
        ABL, ILD, tied_params['rate_lambda'], tied_params['T_0'], tied_params['theta_E'], Z_E, tied_params['t_E_aff'], tied_params['del_go'],
        np.nan, np.nan, np.nan, np.nan, np.nan,
        rate_norm_l, is_norm, False, K_max
    )
    mask_0_1 = (t_pts >= 0) & (t_pts <= 1)
    t_pts_0_1 = t_pts[mask_0_1]
    up_mean_0_1 = up_mean[mask_0_1]
    down_mean_0_1 = down_mean[mask_0_1]
    up_theory_mean_norm = up_mean_0_1 / trunc_factor
    down_theory_mean_norm = down_mean_0_1 / trunc_factor
    up_plus_down_mean = up_theory_mean_norm + down_theory_mean_norm
    return t_pts_0_1, up_plus_down_mean

# %%
# Main analysis loop (Quantiles only)
ABL_arr = [20, 40, 60]
ILD_arr = [-16., -8., -4., -2., -1., 1., 2., 4., 8., 16.]
QUANTILES_TO_PLOT = [0.1, 0.3, 0.5, 0.7, 0.9]

def get_animal_raw_RTs(batch_name, animal_id, ABL, ILD):
    """Fetch raw RTwrtStim for a given animal and stimulus condition (valid trials only)."""
    file_name = f'batch_csvs/batch_{batch_name}_valid_and_aborts.csv'
    df = pd.read_csv(file_name)
    df_stim = df[(df['animal'] == int(animal_id)) & (df['ABL'] == ABL) & (df['ILD'] == ILD) & (df['success'].isin([1, -1]))]
    df_stim = df_stim[(df_stim['RTwrtStim'] >= 0) & (df_stim['RTwrtStim'] <= 1)]
    return df_stim['RTwrtStim'].values

def find_quantile_from_cdf(q, cdf, x_axis):
    """Invert a CDF to find the value corresponding to a given quantile."""
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
    """Compute empirical and theoretical quantiles for one animal across all stimuli."""
    batch_name, animal_id = batch_animal_pair
    print(f"Processing {batch_name} / {animal_id} for quantiles...")
    animal_quantile_data = {}
    try:
        abort_params, tied_params, rate_norm_l, is_norm = get_params_from_animal_pkl_file(batch_name, int(animal_id))
        P_A_mean, C_A_mean, t_stim_samples = get_P_A_C_A(batch_name, int(animal_id), abort_params)
        for abl in ABL_arr:
            for ild in ILD_arr:
                stim_key = (abl, ild)
                # Empirical quantiles
                rts = get_animal_raw_RTs(batch_name, int(animal_id), abl, ild)
                if len(rts) > 5:
                    emp_q = np.quantile(rts, QUANTILES_TO_PLOT)
                else:
                    emp_q = [np.nan] * len(QUANTILES_TO_PLOT)
                # Theoretical quantiles via CDF inversion
                try:
                    t_pts, rtd = get_theoretical_RTD_from_params(
                        batch_name, P_A_mean, C_A_mean, t_stim_samples,
                        abort_params, tied_params, rate_norm_l, is_norm,
                        abl, ild
                    )
                    if np.all(np.isnan(rtd)) or len(t_pts) < 2:
                        raise ValueError("Theoretical RTD invalid")
                    cdf = np.cumsum(rtd) * (t_pts[1] - t_pts[0])
                    if cdf[-1] > 1e-6:
                        cdf = cdf / cdf[-1]
                    else:
                        raise ValueError("CDF sum too small")
                    theo_q = [find_quantile_from_cdf(q, cdf, t_pts) for q in QUANTILES_TO_PLOT]
                except Exception as e:
                    # print(f"  Warn: theory quantiles failed for {stim_key}: {e}")
                    theo_q = [np.nan] * len(QUANTILES_TO_PLOT)
                animal_quantile_data[stim_key] = {
                    'empirical': emp_q,
                    'theoretical': theo_q,
                }
    except Exception as e:
        print(f"ERROR processing animal {batch_name}/{animal_id}: {e}")
    return animal_quantile_data

# Parallel over animals
n_jobs = max(1, os.cpu_count() - 1)
print(f"Running parallel processing (quantiles) with {n_jobs} jobs...")
all_animal_results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(process_animal_for_quantiles)(pair) for pair in batch_animal_pairs
)

# Aggregate across ±ILD (within-animal), then across animals
print("Aggregating data for plotting (quantiles)...")
abs_ild_sorted = sorted(list(set(abs(ild) for ild in ILD_arr)))
def _create_innermost_dict():
    return {'empirical': [], 'theoretical': []}
def _create_inner_defaultdict():
    return defaultdict(_create_innermost_dict)
plot_data = defaultdict(_create_inner_defaultdict)

for animal_data in all_animal_results:
    if not animal_data:
        continue
    for abl in ABL_arr:
        for abs_ild in abs_ild_sorted:
            emp_combined = []
            theo_combined = []
            for ild_sign in [abs_ild, -abs_ild]:
                stim_key = (abl, ild_sign)
                if stim_key in animal_data:
                    emp_combined.append(animal_data[stim_key]['empirical'])
                    theo_combined.append(animal_data[stim_key]['theoretical'])
            if emp_combined:
                plot_data[abl][abs_ild]['empirical'].append(np.nanmean(emp_combined, axis=0))
            if theo_combined:
                plot_data[abl][abs_ild]['theoretical'].append(np.nanmean(theo_combined, axis=0))

# Plot 1: 1x3 panels (per ABL), five quantiles vs |ILD|
print("Generating quantile plot (separate ABLs)...")
fig, axes = plt.subplots(1, len(ABL_arr), figsize=(12, 4), sharey=True)
if len(ABL_arr) == 1:
    axes = [axes]
for i, abl in enumerate(ABL_arr):
    ax = axes[i]
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for q_idx, q in enumerate(QUANTILES_TO_PLOT):
        emp_means, emp_sems = [], []
        theo_means, theo_sems = [], []
        for abs_ild in abs_ild_sorted:
            emp_qs = np.array(plot_data[abl][abs_ild]['empirical'])[:, q_idx] if plot_data[abl][abs_ild]['empirical'] else np.array([np.nan])
            theo_qs = np.array(plot_data[abl][abs_ild]['theoretical'])[:, q_idx] if plot_data[abl][abs_ild]['theoretical'] else np.array([np.nan])
            emp_means.append(np.nanmean(emp_qs))
            emp_sems.append(sem(emp_qs, nan_policy='omit'))
            theo_means.append(np.nanmean(theo_qs))
            theo_sems.append(sem(theo_qs, nan_policy='omit'))
        ax.errorbar(abs_ild_sorted, emp_means, yerr=emp_sems, fmt='o-', color='b', markersize=4, capsize=3, label='Data' if q_idx == 0 else "")
        ax.errorbar(abs_ild_sorted, theo_means, yerr=theo_sems, fmt='^-', color='r', markersize=4, capsize=3, label='Theory' if q_idx == 0 else "")
    ax.set_title(f'ABL = {abl}')
    ax.set_xlabel('|ILD| (dB)')
    ax.set_xscale('log', base=2)
    ax.set_xticks(abs_ild_sorted)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
axes[0].set_ylabel('RT Quantile (s)')
plt.tight_layout(rect=[0, 0, 0.95, 1])
plt.savefig(f'quantile_plot_separate_abls_psycho_fit_T_0_also_fixed_{MODEL_TYPE}_{PARAM_SOURCE}.png', dpi=300)
plt.show()

# Plot 2: Fig-2 style aggregate across ABLs
LABEL_FONTSIZE: int = 25
TICK_FONTSIZE: int = 24
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for q_idx, q in enumerate(QUANTILES_TO_PLOT):
    emp_means, emp_sems = [], []
    theo_means, theo_sems = [], []
    for abs_ild in abs_ild_sorted:
        all_abl_emp = []
        all_abl_theo = []
        for abl in ABL_arr:
            if plot_data[abl][abs_ild]['empirical']:
                all_abl_emp.append(np.array(plot_data[abl][abs_ild]['empirical'])[:, q_idx])
            if plot_data[abl][abs_ild]['theoretical']:
                all_abl_theo.append(np.array(plot_data[abl][abs_ild]['theoretical'])[:, q_idx])
        all_abl_emp = np.concatenate(all_abl_emp) if len(all_abl_emp) > 0 else np.array([np.nan])
        all_abl_theo = np.concatenate(all_abl_theo) if len(all_abl_theo) > 0 else np.array([np.nan])
        emp_means.append(np.nanmean(all_abl_emp))
        emp_sems.append(sem(all_abl_emp, nan_policy='omit'))
        theo_means.append(np.nanmean(all_abl_theo))
        theo_sems.append(sem(all_abl_theo, nan_policy='omit'))
    ax.errorbar(abs_ild_sorted, emp_means, yerr=emp_sems, fmt='o-', color='black', markersize=4, capsize=0)
    ax.errorbar(abs_ild_sorted, theo_means, yerr=theo_sems, fmt='^-', color='tab:red', markersize=4, capsize=0)
ax.set_xlabel('|ILD| (dB)', fontsize=LABEL_FONTSIZE)
ax.set_ylabel('RT Quantile (s)', fontsize=LABEL_FONTSIZE)
ax.set_xscale('log', base=2)
ax.set_xticks(abs_ild_sorted)
ax.set_yticks([0.1, 0.2, 0.3, 0.4])
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
plt.tight_layout()
plt.savefig(f'quantile_plot_with_errorbars_psycho_fit_T_0_also_fixed_{MODEL_TYPE}_{PARAM_SOURCE}.png', dpi=300)
plt.show()