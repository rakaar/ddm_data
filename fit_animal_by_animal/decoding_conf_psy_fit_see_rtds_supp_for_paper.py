# %%
"""
Unified RTD analysis script for vanilla and normalized TIED models.

Set MODEL_TYPE = 'vanilla' or 'norm' at the top to switch between models.
- 'vanilla': uses vbmc_vanilla_tied_results from pickle, is_norm=False, rate_norm_l=0
- 'norm':    uses vbmc_norm_tied_results from pickle, is_norm=True, rate_norm_l from pickle

PARAM_SOURCE controls where tied parameters come from:
- 'results': read tied param means/samples from results_{batch}_{animal}.pkl
- 'psycho' : read 4-parameter VBMC samples from psycho_fits_4-params-del_E_go_fixed_as_avg
In both cases, abort params and fixed t_E_aff/del_go are read from results PKL.

All downstream logic is automatically adjusted based on these flags.
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
#     x_vals = np.arange(-0.12, 1, 0.01)
#     kde_vals = kde(x_vals)
#     return x_vals, kde_vals

# %%
# Define desired batches
# DESIRED_BATCHES = ['Comparable', 'SD', 'LED2', 'LED1', 'LED34', 'LED6']
DESIRED_BATCHES = ['SD', 'LED34', 'LED6', 'LED8', 'LED7', 'LED34_even']



batch_dir = '/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/batch_csvs'
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

# %%

def get_animal_RTD_data(batch_name, animal_id, ABL, ILD, bins):
    file_name = f'batch_csvs/batch_{batch_name}_valid_and_aborts.csv'
    df = pd.read_csv(file_name)
    df = df[(df['animal'] == animal_id) & (df['ABL'] == ABL) & (df['ILD'] == ILD) & (df['success'].isin([1, -1]))]
    # df = df[(df['animal'] == animal_id) & (df['ABL'] == ABL) & (df['ILD'] == ILD) \
    #     & ((df['RTwrtStim'] <= 1) & (df['RTwrtStim'] >= -0.1))]

    bin_centers = (bins[:-1] + bins[1:]) / 2
    if df.empty:
        print(f"No data found for batch {batch_name}, animal {animal_id}, ABL {ABL}, ILD {ILD}. Returning NaNs.")
        rtd_hist = np.full_like(bin_centers, np.nan)
        return bin_centers, rtd_hist
    df = df[df['RTwrtStim'] <= 1]
    if len(df) == 0:
        print(f"No trials with RTwrtStim <= 1 for batch {batch_name}, animal {animal_id}, ABL {ABL}, ILD {ILD}. Returning NaNs.")
        rtd_hist = np.full_like(bin_centers, np.nan)
        return bin_centers, rtd_hist
    rtd_hist, _ = np.histogram(df['RTwrtStim'], bins=bins, density=True)
    return bin_centers, rtd_hist

psycho_fits_repo_path = '/home/rlab/raghavendra/ddm_data/fit_valid_trials/psycho_fits_4-params-del_E_go_fixed_as_avg/'

def get_psycho_params(batch_name, animal_id):
    filename = os.path.join(psycho_fits_repo_path, f'psycho_fit_4-params-del_E_go_fixed_as_avg_{batch_name}_{animal_id}.pkl')
    with open(filename, 'rb') as f:
        vbmc_obj = pickle.load(f)
    vp = vbmc_obj.vp
    samples = vp.sample(int(1e6))[0]
    print(f"Loaded 4-param VBMC samples for {batch_name}, {animal_id}: shape {samples.shape}")
    tied_params = {
        'rate_lambda' : samples[:,0].mean(),
        'T_0' : samples[:,1].mean(),
        'theta_E' : samples[:,2].mean(),
        'w' : samples[:,3].mean(),
    }
    return tied_params


def get_psycho_vp_samples(batch_name, animal_id, n_samples):
    """Sample n_samples parameter draws from the psycho-fit 4-param VBMC VP.
    Returns a dict with arrays for keys: rate_lambda, T_0, theta_E, w
    """
    filename = os.path.join(psycho_fits_repo_path, f'psycho_fit_4-params-del_E_go_fixed_as_avg_{batch_name}_{animal_id}.pkl')
    with open(filename, 'rb') as f:
        vbmc_obj = pickle.load(f)
    vp = vbmc_obj.vp
    samples = vp.sample(n_samples)[0]
    return {
        'rate_lambda': samples[:, 0],
        'T_0': samples[:, 1],
        'theta_E': samples[:, 2],
        'w': samples[:, 3],
    }


def get_fixed_t_E_aff_and_del_go(fit_results_data, batch_name, animal_id):
    """Compute fixed t_E_aff and del_go by averaging vanilla and norm tied fits."""
    vanilla_keyname = "vbmc_vanilla_tied_results"
    norm_keyname = "vbmc_norm_tied_results"
    if vanilla_keyname not in fit_results_data or norm_keyname not in fit_results_data:
        missing = []
        if vanilla_keyname not in fit_results_data:
            missing.append(vanilla_keyname)
        if norm_keyname not in fit_results_data:
            missing.append(norm_keyname)
        raise KeyError(f"Missing required keys in results_{batch_name}_animal_{animal_id}.pkl: {', '.join(missing)}")
    vanilla_tied = fit_results_data[vanilla_keyname]
    norm_tied = fit_results_data[norm_keyname]
    fixed_t_E_aff = 0.5 * (np.mean(vanilla_tied['t_E_aff_samples']) + np.mean(norm_tied['t_E_aff_samples']))
    fixed_del_go = 0.5 * (np.mean(vanilla_tied['del_go_samples']) + np.mean(norm_tied['del_go_samples']))
    print(f"Using fixed t_E_aff={fixed_t_E_aff:.6f}, del_go={fixed_del_go:.6f} for batch {batch_name}, animal {animal_id}")
    return fixed_t_E_aff, fixed_del_go


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
    
    # Load t_E_aff and del_go (averaged across vanilla and norm tied fits)
    fixed_t_E_aff, fixed_del_go = get_fixed_t_E_aff_and_del_go(fit_results_data, batch_name, animal_id)

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
            'T_0_samples': 'T_0',
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
        # Use psycho-fit 4-param VBMC means for tied params
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

    # Attach fixed t_E_aff and del_go
    tied_params['t_E_aff'] = fixed_t_E_aff
    tied_params['del_go'] = fixed_del_go

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
# Main analysis loop
ABL_arr = [20, 40, 60]
ILD_arr = [-16., -8., -4., -2., -1., 1., 2., 4., 8., 16.]
rt_bins = np.arange(0, 1.02, 0.02)  # 0 to 1 second in 0.02s bins

def process_batch_animal(batch_animal_pair):
    batch_name, animal_id = batch_animal_pair
    print(f"Processing batch {batch_name}, animal {animal_id}")
    animal_rtd_data = {}
    try:
        abort_params, tied_params, rate_norm_l, is_norm = get_params_from_animal_pkl_file(batch_name, int(animal_id))
        p_a, c_a, ts_samp = get_P_A_C_A(batch_name, int(animal_id), abort_params)
        for abl in ABL_arr:
            print(f"Animal = {batch_name},{animal_id}, Processing ABL {abl}")
            for ild in ILD_arr:
                stim_key = (abl, ild)
                try:
                    bin_centers, rtd_hist = get_animal_RTD_data(batch_name, int(animal_id), abl, ild, rt_bins)
                    try:
                        t_pts_0_1, up_plus_down = get_theoretical_RTD_from_params(
                            batch_name, p_a, c_a, ts_samp, abort_params, tied_params, rate_norm_l, is_norm, abl, ild
                        )
                    except Exception as e:
                        print(f"  Error calculating theoretical RTD for ABL={abl}, ILD={ild}: {str(e)}")
                        t_pts_0_1 = np.linspace(0, 1, 100)
                        up_plus_down = np.full_like(t_pts_0_1, np.nan)
                    # --- Simulation KDE ---
                    # Prepare t_stim_samples for simulation: sample from this animal's intended_fix
                    try:
                        file_name = f"batch_csvs/batch_{batch_name}_valid_and_aborts.csv"
                        df = pd.read_csv(file_name)
                        df_animal = df[df['animal'] == int(animal_id)]
                        N_sim = int(1e5)  # Use a reasonable number for speed; adjust as needed
                        N_print = max(1, N_sim // 5)
                        dt = 1e-3
                        t_stim_samples = df_animal['intended_fix'].sample(N_sim, replace=True).values
                        # NOTE: TEMPORILY, NO SIMULATION. TAKES A LOT OF TIME
                        # Z_E = (tied_params['w'] - 0.5) * 2 * tied_params['theta_E']
                        # x_vals, kde_vals = get_simulation_RTD_KDE(
                        #     abort_params, tied_params, rate_norm_l, Z_E, abl, ild, t_stim_samples, N_sim, N_print, dt
                        # )
                        # sim_dict = {'x_vals': x_vals, 'kde_vals': kde_vals}
                        sim_dict = {'x_vals': np.array([]), 'kde_vals': np.array([])}
                    except Exception as e:
                        print(f"  Error running simulation for ABL={abl}, ILD={ild}: {str(e)}")
                        sim_dict = {'x_vals': np.array([]), 'kde_vals': np.array([])}

                    animal_rtd_data[stim_key] = {
                        'empirical': {
                            'bin_centers': bin_centers,
                            'rtd_hist': rtd_hist
                        },
                        'theoretical': {
                            't_pts': t_pts_0_1,
                            'rtd': up_plus_down
                        },
                        'simulation': sim_dict
                    }
                    print(f"  Processed stimulus ABL={abl}, ILD={ild}")
                except Exception as e:
                    print(f"  Error processing stimulus ABL={abl}, ILD={ild}: {str(e)}")
                    continue
    except Exception as e:
        print(f"Error processing batch {batch_name}, animal {animal_id}: {str(e)}")
    return batch_animal_pair, animal_rtd_data

n_jobs = max(1, os.cpu_count() - 1)
print(f"Running with {n_jobs} parallel jobs")

results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(process_batch_animal)(batch_animal_pair) for batch_animal_pair in batch_animal_pairs
)

rtd_data = {}
for batch_animal_pair, animal_rtd_data in results:
    if animal_rtd_data:
        rtd_data[batch_animal_pair] = animal_rtd_data
print(f"Completed processing {len(rtd_data)} batch-animal pairs")

# %%
# PLOT data average and theory average for each stimulus
fig, axes = plt.subplots(3, 10, figsize=(20, 8), sharex=True, sharey=True)
for ax_row in axes:
    for ax in ax_row:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
for i, abl in enumerate(ABL_arr):
    for j, ild in enumerate(ILD_arr):
        stim_key = (abl, ild)
        empirical_rtds = []
        theoretical_rtds = []
        bin_centers = None
        t_pts = None
        sim_kdes = []
        sim_x_vals = None
        for batch_animal_pair, animal_data in rtd_data.items():
            if stim_key in animal_data:
                emp_data = animal_data[stim_key]['empirical']
                if not np.all(np.isnan(emp_data['rtd_hist'])):
                    empirical_rtds.append(emp_data['rtd_hist'])
                    bin_centers = emp_data['bin_centers']
                theo_data = animal_data[stim_key]['theoretical']
                if not np.all(np.isnan(theo_data['rtd'])):
                    theoretical_rtds.append(theo_data['rtd'])
                    t_pts = theo_data['t_pts']
                sim_data = animal_data[stim_key].get('simulation', None)
                if sim_data is not None and sim_data['x_vals'].size > 0:
                    sim_kdes.append(sim_data['kde_vals'])
                    sim_x_vals = sim_data['x_vals']  # All should be the same
        ax = axes[i, j]
        if empirical_rtds and bin_centers is not None:
            avg_empirical_rtd = np.nanmean(empirical_rtds, axis=0)
            ax.plot(bin_centers, avg_empirical_rtd, 'b-', linewidth=1.5, label='Data')
        if theoretical_rtds and t_pts is not None:
            avg_theoretical_rtd = np.nanmean(theoretical_rtds, axis=0)
            ax.plot(t_pts, avg_theoretical_rtd, 'r-', linewidth=1.5, label='Theory')
        if sim_kdes and sim_x_vals is not None:
            avg_sim_kde = np.nanmean(sim_kdes, axis=0)
            ax.plot(sim_x_vals, avg_sim_kde, color='green', alpha=0.3, lw=3, label='Simulation')
        ax = axes[i, j]
        if empirical_rtds and bin_centers is not None:
            avg_empirical_rtd = np.nanmean(empirical_rtds, axis=0)
            ax.plot(bin_centers, avg_empirical_rtd, 'b-', linewidth=1.5, label='Data')
        if theoretical_rtds and t_pts is not None:
            avg_theoretical_rtd = np.nanmean(theoretical_rtds, axis=0)
            ax.plot(t_pts, avg_theoretical_rtd, 'r-', linewidth=1.5, label='Theory')
        ax.set_title(f'ABL={abl}, ILD={ild}', fontsize=10)
        if i == 2:
            ax.set_xlabel('RT (s)')
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['0', '1'], fontsize=12)
        ax.set_xlim(0, 1)
        if j == 0:
            ax.set_ylabel('Density')
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=2)
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.savefig(f'rtd_average_by_stimulus_4-params-del_E_go_fixed_as_avg_{MODEL_TYPE}.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
abs_ILD_arr = [abs(ild) for ild in ILD_arr]
abs_ILD_arr = sorted(list(set(abs_ILD_arr)))
max_xlim_RT = 1.0
fig, axes = plt.subplots(len(ABL_arr), len(abs_ILD_arr), figsize=(10,6), sharex=True, sharey=True)
for ax_row in axes:
    for ax in ax_row:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
for i, abl in enumerate(ABL_arr):
    for j, abs_ild in enumerate(abs_ILD_arr):
        # For each (ABL, abs(ILD)), average over (ABL, ILD) and (ABL, -ILD)
        empirical_rtds = []
        theoretical_rtds = []
        bin_centers = None
        t_pts = None
        sim_kdes = []
        sim_x_vals = None
        for ild in [abs_ild, -abs_ild]:
            stim_key = (abl, ild)
            for batch_animal_pair, animal_data in rtd_data.items():
                if stim_key in animal_data:
                    emp_data = animal_data[stim_key]['empirical']
                    if not np.all(np.isnan(emp_data['rtd_hist'])):
                        empirical_rtds.append(emp_data['rtd_hist'])
                        bin_centers = emp_data['bin_centers']
                    theo_data = animal_data[stim_key]['theoretical']
                    if not np.all(np.isnan(theo_data['rtd'])):
                        theoretical_rtds.append(theo_data['rtd'])
                        t_pts = theo_data['t_pts']
                    sim_data = animal_data[stim_key].get('simulation', None)
                    if sim_data is not None and sim_data['x_vals'].size > 0:
                        sim_kdes.append(sim_data['kde_vals'])
                        sim_x_vals = sim_data['x_vals']  # All should be the same
        ax = axes[i, j]
        if empirical_rtds and bin_centers is not None:
            avg_empirical_rtd = np.nanmean(empirical_rtds, axis=0)
            ax.plot(bin_centers, avg_empirical_rtd, 'b-', linewidth=1.5, label='Data')
        if theoretical_rtds and t_pts is not None:
            avg_theoretical_rtd = np.nanmean(theoretical_rtds, axis=0)
            ax.plot(t_pts, avg_theoretical_rtd, 'r-', linewidth=1.5, label='Theory')
        if sim_kdes and sim_x_vals is not None:
            avg_sim_kde = np.nanmean(sim_kdes, axis=0)
            ax.plot(sim_x_vals, avg_sim_kde, color='green', alpha=0.4, lw=3, label='Simulation')
        ax = axes[i, j]
        if empirical_rtds and bin_centers is not None:
            avg_empirical_rtd = np.nanmean(empirical_rtds, axis=0)
            ax.plot(bin_centers, avg_empirical_rtd, 'b-', linewidth=1.5, label='Data')
        if theoretical_rtds and t_pts is not None:
            avg_theoretical_rtd = np.nanmean(theoretical_rtds, axis=0)
            ax.plot(t_pts, avg_theoretical_rtd, 'r-', linewidth=1, label='Theory')
        if i == len(ABL_arr) - 1:
            ax.set_xlabel('RT (s)', fontsize=12)
            ax.set_xticks([0, max_xlim_RT])
            ax.set_xticklabels(['0', max_xlim_RT], fontsize=12)
        ax.set_xlim(0, max_xlim_RT)
        ax.set_yticks([0, 12])
        ax.set_ylim(0, 14)
        ax.axvline(x=0, color='k', linestyle='--', linewidth=1)
        if j == 0:
            ax.set_ylabel(f'ABL={abl}', fontsize=12, rotation=0, ha='right', va='center')
        if i == 0:
            ax.set_title(f'|ILD|={abs_ild}', fontsize=12)
for ax_row in axes:
    for ax in ax_row:
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.savefig(f'rtd_average_by_abs_ILD_FOLDED_4-params-del_E_go_fixed_as_avg_{MODEL_TYPE}.png', dpi=300, bbox_inches='tight')
plt.show()



# %%
# === RTD plot: 5 param samples per panel (NO EMPIRICAL) ===
print('\nPlotting 5 individual sampled-theory RTDs and their mean per (ABL, |ILD|) panel (first animal per panel)...')

abs_ILD_arr = [abs(ild) for ild in ILD_arr]
abs_ILD_arr = sorted(list(set(abs_ILD_arr)))
max_xlim_RT = 1.0
fig, axes = plt.subplots(len(ABL_arr), len(abs_ILD_arr), figsize=(10,6), sharex=True, sharey=True)
for ax_row in axes:
    for ax in ax_row:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

for i, abl in enumerate(ABL_arr):
    for j, abs_ild in enumerate(abs_ILD_arr):
        found = False
        for ild in [abs_ild, -abs_ild]:
            stim_key = (abl, ild)
            for batch_animal_pair, animal_data in rtd_data.items():
                if stim_key in animal_data and not found:
                    batch_name, animal_id = batch_animal_pair
                    # Load parameter samples directly from results_{batch}_animal_{id}.pkl
                    results_pkl = f'results_{batch_name}_animal_{animal_id}.pkl'
                    with open(results_pkl, 'rb') as f:
                        fit_results_data = pickle.load(f)
                    # Abort params (means)
                    abort_keyname = "vbmc_aborts_results"
                    abort_params = {}
                    if abort_keyname in fit_results_data:
                        abort_samples = fit_results_data[abort_keyname]
                        abort_params['V_A'] = np.mean(np.asarray(abort_samples['V_A_samples']))
                        abort_params['theta_A'] = np.mean(np.asarray(abort_samples['theta_A_samples']))
                        abort_params['t_A_aff'] = np.mean(np.asarray(abort_samples['t_A_aff_samp']))
                    else:
                        continue
                    # Choose tied source and pull sample arrays according to PARAM_SOURCE
                    if PARAM_SOURCE == 'results':
                        source_key = 'vbmc_vanilla_tied_results' if MODEL_TYPE == 'vanilla' else 'vbmc_norm_tied_results'
                        if source_key not in fit_results_data:
                            print(f"Missing {source_key} in {results_pkl}")
                            continue
                        model_samples = fit_results_data[source_key]
                        try:
                            rate_lambda_s = np.asarray(model_samples['rate_lambda_samples'])
                            T0_s = np.asarray(model_samples['T_0_samples'])
                            theta_E_s = np.asarray(model_samples['theta_E_samples'])
                            w_s = np.asarray(model_samples['w_samples'])
                            if MODEL_TYPE == 'norm':
                                rate_norm_l_s = np.asarray(model_samples['rate_norm_l_samples'])
                            else:
                                rate_norm_l_s = np.zeros_like(w_s)
                        except KeyError as e:
                            print(f"Missing key in {source_key} for {batch_name},{animal_id}: {e}")
                            continue
                        n_pool = min(len(rate_lambda_s), len(T0_s), len(theta_E_s), len(w_s), len(rate_norm_l_s))
                        if n_pool < 5:
                            print(f"Not enough samples in {source_key} for {batch_name},{animal_id} (n={n_pool})")
                            continue
                        idxs = np.random.choice(n_pool, size=5, replace=False)
                    elif PARAM_SOURCE == 'psycho':
                        try:
                            draws = get_psycho_vp_samples(batch_name, int(animal_id), 5)
                        except Exception as e:
                            print(f"Error sampling psycho VP for {batch_name},{animal_id}: {e}")
                            continue
                        rate_lambda_s = np.asarray(draws['rate_lambda'])
                        T0_s = np.asarray(draws['T_0'])
                        theta_E_s = np.asarray(draws['theta_E'])
                        w_s = np.asarray(draws['w'])
                        if MODEL_TYPE == 'norm':
                            # Pull 5 rate_norm_l draws from results PKL
                            if 'vbmc_norm_tied_results' in fit_results_data and 'rate_norm_l_samples' in fit_results_data['vbmc_norm_tied_results']:
                                rn_pool = np.asarray(fit_results_data['vbmc_norm_tied_results']['rate_norm_l_samples'])
                                if rn_pool.size >= 5:
                                    rn_idxs = np.random.choice(rn_pool.size, size=5, replace=False)
                                    rate_norm_l_s = rn_pool[rn_idxs]
                                else:
                                    rate_norm_l_s = np.full(5, float(np.mean(rn_pool)) if rn_pool.size > 0 else 0.0)
                            else:
                                print(f"Missing rate_norm_l_samples for norm model in {results_pkl}; using zeros")
                                rate_norm_l_s = np.zeros(5, dtype=float)
                        else:
                            rate_norm_l_s = np.zeros(5, dtype=float)
                        idxs = range(5)
                    else:
                        print(f"Unknown PARAM_SOURCE: {PARAM_SOURCE}")
                        continue
                    # Get fixed t_E_aff and del_go for this animal
                    try:
                        fixed_t_E_aff, fixed_del_go = get_fixed_t_E_aff_and_del_go(fit_results_data, batch_name, animal_id)
                    except Exception as e:
                        print(f"Error computing fixed t_E_aff/del_go for {batch_name}, {animal_id}: {e}")
                        continue
                    # Get theoretical RTDs from the previous plots
                    theoretical_rtds = []
                    t_pts = None
                    
                    # Get the theoretical RTD from the previous plots (same as in the earlier code)
                    for ild_check in [abs_ild, -abs_ild]:
                        stim_key_check = (abl, ild_check)
                        for batch_animal_pair_check, animal_data_check in rtd_data.items():
                            if stim_key_check in animal_data_check:
                                theo_data = animal_data_check[stim_key_check]['theoretical']
                                if not np.all(np.isnan(theo_data['rtd'])):
                                    theoretical_rtds.append(theo_data['rtd'])
                                    t_pts = theo_data['t_pts']
                    
                    # Also plot individual samples for comparison
                    P_A_mean, C_A_mean, t_stim_samples = get_P_A_C_A(batch_name, int(animal_id), abort_params)
                    rtds = []
                    for k, idx in enumerate(idxs):
                        param_set = {
                            'rate_lambda': float(rate_lambda_s[idx]),
                            'T_0': float(T0_s[idx]),
                            'theta_E': float(theta_E_s[idx]),
                            'w': float(w_s[idx]),
                            't_E_aff': float(fixed_t_E_aff),
                            'del_go': float(fixed_del_go),
                        }
                        rn_l = float(rate_norm_l_s[idx]) if MODEL_TYPE == 'norm' else 0.0
                        is_norm_flag = (MODEL_TYPE == 'norm')
                        try:
                            t_pts_0_1, rtd = get_theoretical_RTD_from_params(
                                batch_name, P_A_mean, C_A_mean, t_stim_samples, abort_params, param_set, rn_l, is_norm_flag, abl, ild
                            )
                            if t_pts is None:
                                t_pts = t_pts_0_1
                            rtds.append(rtd)
                        except Exception as e:
                            print(f"Error computing RTD for {batch_name},{animal_id}, sample {idx}: {e}")
                            continue
                    
                    ax = axes[i, j]
                    colors = ['r', 'g', 'b', 'm', 'orange']
                    for k, rtd in enumerate(rtds):
                        ax.plot(t_pts, rtd, color=colors[k % len(colors)], lw=1, label=f'Sample {k+1}' if i==0 and j==0 else None)
                    
                    # Plot the average theoretical RTD from previous plots instead of mean of samples
                    if theoretical_rtds and t_pts is not None:
                        avg_theoretical_rtd = np.nanmean(theoretical_rtds, axis=0)
                        ax.plot(t_pts, avg_theoretical_rtd, color='k', lw=1.5, label='Theory Average' if i==0 and j==0 else None)
                    if i == len(ABL_arr) - 1:
                        ax.set_xlabel('RT (s)', fontsize=12)
                        ax.set_xticks([0, max_xlim_RT])
                        ax.set_xticklabels(['0', max_xlim_RT], fontsize=12)
                    ax.set_xlim(0, max_xlim_RT)
                    ax.set_yticks([0, 15])
                    ax.set_ylim(0, 30)
                    ax.axvline(x=0, color='k', linestyle='--', linewidth=1)
                    if j == 0:
                        ax.set_ylabel(f'ABL={abl}', fontsize=12, rotation=0, ha='right', va='center')
                    if i == 0:
                        ax.set_title(f'|ILD|={abs_ild}', fontsize=12)
                    found = True
                    break
            if found:
                break
for ax_row in axes:
    for ax in ax_row:
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.savefig(f'5_rtd_individual_and_mean_by_abs_ILD_FOLDED_4-params-del_E_go_fixed_as_avg_{MODEL_TYPE}.png', dpi=300, bbox_inches='tight')
plt.show()