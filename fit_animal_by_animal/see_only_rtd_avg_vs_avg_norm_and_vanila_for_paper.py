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
DESIRED_BATCHES = ['SD', 'LED2', 'LED1', 'LED34', 'LED6', 'LED8', 'LED7']

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
                # Exclude animals 40, 41, 43 from LED2 batch
                if not (batch_name == 'LED2' and animal_id in ['40', '41', '43']):
                    pairs.append((batch_name, animal_id))
        else:
            print(f"Warning: Invalid filename format: {filename}")
    return pairs

batch_animal_pairs = find_batch_animal_pairs()
# with open('high_slope_animals.pkl', 'rb') as f:
#     batch_animal_pairs = pickle.load(f)

print(f"Found {len(batch_animal_pairs)} batch-animal pairs: {batch_animal_pairs}")

# remove SD 49 due to issue in sensory delay
# batch_animal_pairs = [(batch, animal) for batch, animal in batch_animal_pairs if not (batch == 'SD' and animal == '49')]
# print(f"Removed SD 49 due to issue in sensory delay. Found {len(batch_animal_pairs)} batch-animal pairs: {batch_animal_pairs}")

def get_animal_RTD_data(batch_name, animal_id, ABL, ILD, bins):
    file_name = f'batch_csvs/batch_{batch_name}_valid_and_aborts.csv'
    df = pd.read_csv(file_name)
    # df = df[(df['animal'] == animal_id) & (df['ABL'] == ABL) & (df['ILD'] == ILD) & (df['success'].isin([1, -1]))]
    df = df[(df['animal'] == animal_id) & (df['ABL'] == ABL) & (df['ILD'] == ILD) \
        & ((df['RTwrtStim'] <= 1) & (df['RTwrtStim'] >= -0.1))]

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

def get_theoretical_RTD_from_params(P_A_mean, C_A_mean, t_stim_samples, abort_params, tied_params, rate_norm_l, is_norm, ABL, ILD):
    phi_params_obj = np.nan
    K_max = 10
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
            t_stim - 0.1, T_trunc,
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
    mask_0_1 = (t_pts >= -0.1) & (t_pts <= 1)
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
rt_bins = np.arange(-0.12, 1.02, 0.02)  # 0 to 1 second in 0.02s bins

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
                            p_a, c_a, ts_samp, abort_params, tied_params, rate_norm_l, is_norm, abl, ild
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
        if j == 0:
            ax.set_ylabel('Density')
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=2)
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.savefig(f'rtd_average_by_stimulus_{MODEL_TYPE}.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
abs_ILD_arr = [abs(ild) for ild in ILD_arr]
abs_ILD_arr = sorted(list(set(abs_ILD_arr)))
max_xlim_RT = 0.6
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
            ax.plot(t_pts, avg_theoretical_rtd, 'r-', linewidth=1.5, label='Theory')
        if i == len(ABL_arr) - 1:
            ax.set_xlabel('RT (s)', fontsize=12)
            ax.set_xticks([-0.1, max_xlim_RT])
            ax.set_xticklabels(['-0.1', max_xlim_RT], fontsize=12)
        ax.set_xlim(-0.1, max_xlim_RT)
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
plt.savefig(f'rtd_average_by_abs_ILD_FOLDED_{MODEL_TYPE}.png', dpi=300, bbox_inches='tight')
plt.show()


# %%
# issue in sensory delay  in one 8 animals  in rate norm model
# vanilla_sensory_delays = np.zeros((len(batch_animal_pairs),))
# norm_sensory_delays = np.zeros((len(batch_animal_pairs),))

# MODEL_TYPE = 'vanilla'
# for idx, (batch, animal) in enumerate(batch_animal_pairs):
#     abort_params, tied_params, rate_norm_l, is_norm = get_params_from_animal_pkl_file(batch, int(animal))
#     vanilla_sensory_delays[idx] = tied_params['t_E_aff']
#     print(f'idx {idx}, batch {batch}, animal {animal}')

# MODEL_TYPE = 'norm'
# for idx, (batch, animal) in enumerate(batch_animal_pairs):
#     abort_params, tied_params, rate_norm_l, is_norm = get_params_from_animal_pkl_file(batch, int(animal))
#     norm_sensory_delays[idx] = tied_params['t_E_aff']

# # plot both the sensory delays
# plt.figure(figsize=(6, 4))
# plt.plot(vanilla_sensory_delays * 1000, 'ro', label='Vanilla')
# plt.plot(norm_sensory_delays * 1000, 'bo', label='Norm')
# # plot vertical line with mean of vanilla sensory delay and norm sensory delay
# plt.axhline(y=np.mean(vanilla_sensory_delays * 1000), color='k', linestyle='--', linewidth=1)
# plt.axhline(y=np.mean(norm_sensory_delays * 1000), color='k', linestyle='--', linewidth=1)
# plt.ylabel('Sensory Delay (ms)')
# plt.xlabel('Animal')
# plt.legend()
# plt.show()

# print batch 
# %%
