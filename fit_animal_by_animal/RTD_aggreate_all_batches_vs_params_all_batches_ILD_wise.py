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
    cum_pro_and_reactive_time_vary_fn, 
    rho_A_t_fn, 
    cum_A_t_fn
)
from collections import defaultdict
import random

# %%
bins = np.arange(0,1,0.02)
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_dir = os.path.join(base_dir, 'batch_csvs')

# Batch names to load, as specified by the user
DESIRED_BATCHES = ['Comparable', 'SD', 'LED2', 'LED1', 'LED34']

all_data_list = []

for batch_name in DESIRED_BATCHES:
    file_name = f'batch_{batch_name}_valid_and_aborts.csv'
    file_path = os.path.join(csv_dir, file_name)
    df = pd.read_csv(file_path)
    df['batch_name'] = batch_name
    all_data_list.append(df)
    
all_data = pd.concat(all_data_list, ignore_index=True)
valid_trials = all_data[all_data['success'].isin([1, -1])].copy()
filtered_trials = valid_trials[valid_trials['RTwrtStim'] <= 1]

# %% Calculate and store RTDs per animal for specified stimuli
print("\nCalculating RTDs per animal for specified stimuli...")

# Define target stimuli based on user request
ABL_TARGETS = [20, 40, 60] 
ILD_ACTUAL_TARGETS = [-16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16]

if 'abs_ILD' not in filtered_trials.columns:
    filtered_trials.loc[:, 'abs_ILD'] = filtered_trials['ILD'].abs().astype(float)

rtd_calc_data = filtered_trials[
    filtered_trials['ABL'].isin(ABL_TARGETS) &
    filtered_trials['ILD'].isin(ILD_ACTUAL_TARGETS)
].copy() # Use .copy() if further modifications or to be safe

animal_rtds_by_stimulus = {}
rtd_bin_edges = bins

if rtd_calc_data.empty:
    print(f"No data found in filtered_trials for ABLs {ABL_TARGETS} and ILDs {ILD_ACTUAL_TARGETS}. No RTDs calculated.")
else:
    grouped_for_rtd = rtd_calc_data.groupby(['animal', 'ABL', 'ILD'])

    for name, group_df in grouped_for_rtd:
        animal_id, abl_val, ild_val = name
        rts_for_group = group_df['RTwrtStim'].dropna()

        if not rts_for_group.empty:
            hist_values, _ = np.histogram(rts_for_group, bins=rtd_bin_edges, density=True)
            
            stim_key = (abl_val, ild_val)

            if stim_key not in animal_rtds_by_stimulus:
                animal_rtds_by_stimulus[stim_key] = []
            
            animal_rtds_by_stimulus[stim_key].append({
                'animal': str(animal_id),
                'rtd_hist': hist_values,
                'num_trials': len(rts_for_group)
            })
    
# %%
# PLOT for each ILD animal averages


ABL_colors = {20: '#219ebc', 40: '#fb8500', 60: 'green'}

# Calculate bin centers for plotting histograms
# rtd_bin_edges is available from the RTD calculation section (it's the global 'bins')
bin_centers = (rtd_bin_edges[:-1] + rtd_bin_edges[1:]) / 2

num_ilds = len(ILD_ACTUAL_TARGETS)
num_abls = len(ABL_TARGETS) # Should be 3 for [20, 40, 60]

fig, axes = plt.subplots(num_abls, num_ilds, 
                         figsize=(4 * num_ilds, 3.5 * num_abls), 
                         sharex=True, sharey=True)
    
# If only one ABL or one ILD, axes might not be a 2D array. Handle this.
if num_abls == 1 and num_ilds == 1:
    axes = np.array([[axes]])
elif num_abls == 1:
    axes = np.array([axes])
elif num_ilds == 1:
    axes = np.array([[ax] for ax in axes]) # Make it a 2D array with 1 column

for row_idx, abl_val in enumerate(ABL_TARGETS):
    for col_idx, current_ild in enumerate(ILD_ACTUAL_TARGETS):
        ax = axes[row_idx, col_idx]
        stim_key = (float(abl_val), float(current_ild))
        
        plot_successful = False
        if stim_key in animal_rtds_by_stimulus and animal_rtds_by_stimulus[stim_key]:
            rtd_hist_arrays = [item['rtd_hist'] for item in animal_rtds_by_stimulus[stim_key]]
            
            if rtd_hist_arrays:
                mean_rtd_hist = np.mean(np.array(rtd_hist_arrays), axis=0)
                ax.plot(bin_centers, mean_rtd_hist, drawstyle='steps-mid', 
                        color=ABL_colors.get(abl_val, 'gray'), 
                        # No label per line needed now, ABL is by row
                        linewidth=2)
                plot_successful = True
        
        # Titles for columns (top row only)
        if row_idx == 0:
            ax.set_title(f'ILD = {current_ild}')
        
        # X-axis labels (bottom row only)
        if row_idx == num_abls - 1:
            ax.set_xlabel('RT (s)')
        
        # Y-axis labels (first column only, indicating ABL)
        if col_idx == 0:
            ax.set_ylabel(f'ABL {int(abl_val)}\nDensity')
        
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylim(0, 12) # User had set this, keeping it.
        ax.set_xlim(0,1)   # User had set this, keeping it.
        
        # If no data was plotted, indicate it to avoid blank plots misleadingly
        if not plot_successful:
            ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10, color='red')

fig.suptitle('Average Reaction Time Distributions (RTDs) by Stimulus Condition', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect for suptitle and x-labels
plt.show()


# %%
# Load model parameters for animals/batches that contributed to RTD averages
if 'rtd_calc_data' in locals() and not rtd_calc_data.empty and \
   all(col in rtd_calc_data.columns for col in ['animal', 'batch_name']):

    RESULTS_DIR_PARAMS = '/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/' # Explicit path

    unique_animal_batches = rtd_calc_data[['animal', 'batch_name']].drop_duplicates().to_records(index=False)
    
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

    animal_batch_params_means = {}

    for animal_id, batch_name in unique_animal_batches:
        animal_id_str = str(animal_id) # Ensure animal_id is string for filename formatting
        batch_name_str = str(batch_name)
        
        pickle_fname = f'results_{batch_name_str}_animal_{animal_id_str}.pkl'
        pickle_path = os.path.join(RESULTS_DIR_PARAMS, pickle_fname)

        current_animal_params = {
            'vbmc_aborts_means': {label: np.nan for label in vbmc_aborts_param_keys_map.values()},
            'vbmc_vanilla_tied_means': {label: np.nan for label in vbmc_vanilla_tied_param_keys_map.values()}
        }

        if not os.path.exists(pickle_path):
            print(f"Warning: Pickle file not found for animal {animal_id_str}, batch {batch_name_str} at {pickle_path}")
            animal_batch_params_means[(animal_id_str, batch_name_str)] = current_animal_params
            continue

        try:
            with open(pickle_path, 'rb') as f:
                fit_results_data = pickle.load(f)
            
            # Process VBMC Aborts Results
            if 'vbmc_aborts_results' in fit_results_data:
                model_data = fit_results_data['vbmc_aborts_results']
                for p_key_sample, p_label in vbmc_aborts_param_keys_map.items():
                    if p_key_sample in model_data and model_data[p_key_sample] is not None:
                        samples = np.asarray(model_data[p_key_sample])
                        if samples.size > 0:
                            current_animal_params['vbmc_aborts_means'][p_label] = np.mean(samples)
                        else:
                            print(f"Warning: Empty samples for {p_key_sample} in {pickle_fname} for vbmc_aborts_results")
                    else:
                        print(f"Warning: Key '{p_key_sample}' not found or is None in {pickle_fname} for vbmc_aborts_results")
            else:
                print(f"Warning: 'vbmc_aborts_results' not found in {pickle_fname}")

            # Process VBMC Vanilla TIED Results
            if 'vbmc_vanilla_tied_results' in fit_results_data:
                model_data = fit_results_data['vbmc_vanilla_tied_results']
                for p_key_sample, p_label in vbmc_vanilla_tied_param_keys_map.items():
                    if p_key_sample in model_data and model_data[p_key_sample] is not None:
                        samples = np.asarray(model_data[p_key_sample])
                        if samples.size > 0:
                            current_animal_params['vbmc_vanilla_tied_means'][p_label] = np.mean(samples)
                        else:
                            print(f"Warning: Empty samples for {p_key_sample} in {pickle_fname} for vbmc_vanilla_tied_results")
                    else:
                        print(f"Warning: Key '{p_key_sample}' not found or is None in {pickle_fname} for vbmc_vanilla_tied_results")
            else:
                print(f"Warning: 'vbmc_vanilla_tied_results' not found in {pickle_fname}")
            
            animal_batch_params_means[(animal_id_str, batch_name_str)] = current_animal_params

        except Exception as e:
            print(f"Error loading or processing pickle file {pickle_path}: {e}")
            animal_batch_params_means[(animal_id_str, batch_name_str)] = current_animal_params # Store with NaNs

    print("\n--- Finished loading model parameters ---")
    
else:
    print("Skipping model parameter loading: 'rtd_calc_data' is not available, empty, or missing 'animal'/'batch_name' columns.")
    animal_batch_params_means = {} # Ensure it's defined for the next step
    all_animal_batch_theoretical_rtds = {} # Ensure it's defined

# --- END OF PARAMETER LOADING AND PREPARATION ---

# %%
from time_vary_norm_utils import rho_A_t_fn
from animal_wise_plotting_utils import calculate_theoretical_curves

V_A = 1.4425311495431423
theta_A = 1.610192771910792
t_A_aff = 0.03724630477308381
rate_lambda = 0.11469178637304395
T_0 = 0.00034996254264112425
theta_E = 34.69426235080911
w = 0.5522561464121016
t_E_aff = 0.0632140115545335
del_go = 0.1440168192595432
Z_E = 3.6259769061274927 # Calculated from w and theta_E

N_theory = int(1e3)
t_pts = np.arange(-1, 2, 0.001)
P_A_mean, C_A_mean, t_stim_samples = calculate_theoretical_curves(
    rtd_calc_data, N_theory, t_pts, t_A_aff, V_A, theta_A, rho_A_t_fn
)
T_trunc = 0.3 
phi_params_obj = np.nan
rate_norm_l = 0
is_norm = False
is_time_vary = False
K_max = 10
bw = 0.02
bins = np.arange(0, 1, bw)
bin_centers = bins[:-1] + (0.5 * bw)


for i_idx, ILD in enumerate([1]):
    for a_idx, ABL in enumerate([40]):
        trunc_fac_samples = np.zeros((len(t_stim_samples)))
        for idx, t_stim in enumerate(t_stim_samples):
            trunc_fac_samples[idx] = cum_pro_and_reactive_time_vary_fn(
                            t_stim + 1, T_trunc,
                            V_A, theta_A, t_A_aff,
                            t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_E_aff,
                            phi_params_obj, rate_norm_l, 
                            is_norm, is_time_vary, K_max) \
                            - \
                            cum_pro_and_reactive_time_vary_fn(
                            t_stim, T_trunc,
                            V_A, theta_A, t_A_aff,
                            t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_E_aff,
                            phi_params_obj, rate_norm_l, 
                            is_norm, is_time_vary, K_max) + 1e-10
        trunc_factor = np.mean(trunc_fac_samples)
        
        up_mean = np.array([up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn(
                    t, 1,
                    P_A_mean[i], C_A_mean[i],
                    ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go,
                    phi_params_obj, rate_norm_l, 
                    is_norm, is_time_vary, K_max) for i, t in enumerate(t_pts)])
        down_mean = np.array([up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn(
                t, -1,
                P_A_mean[i], C_A_mean[i],
                ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go,
                phi_params_obj, rate_norm_l, 
                is_norm, is_time_vary, K_max) for i, t in enumerate(t_pts)])
        
        # Filter to relevant time window
        mask_0_1 = (t_pts >= 0) & (t_pts <= 1)
        t_pts_0_1 = t_pts[mask_0_1]
        up_mean_0_1 = up_mean[mask_0_1]
        down_mean_0_1 = down_mean[mask_0_1]
        
        # Normalize theory curves
        up_theory_mean_norm = up_mean_0_1 / trunc_factor
        down_theory_mean_norm = down_mean_0_1 / trunc_factor

        up_plus_down_mean = up_theory_mean_norm + down_theory_mean_norm
                

plt.plot(t_pts_0_1, up_plus_down_mean)
plt.show()


# %%
N_sim = int(1e5)
ILD = 1
ABL = 40
N_print = N_sim // 5
dt = 1e-3
rate_norm_l = 0
is_norm = False
is_time_vary = False
phi_params_obj = np.nan
t_stim_samples = rtd_calc_data['intended_fix'].sample(N_sim, replace=True).values
sim_results = Parallel(n_jobs=30)(
    delayed(psiam_tied_data_gen_wrapper_rate_norm_fn)(
        V_A, theta_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_A_aff, t_E_aff, del_go, 
        t_stim_samples[iter_num], rate_norm_l, iter_num, N_print, dt
    ) for iter_num in tqdm(range(N_sim))
)

# %%
sim_results_df = pd.DataFrame(sim_results)
sim_results_df_valid = sim_results_df[sim_results_df['rt'] > sim_results_df['t_stim']]
sim_results_df_valid_lt_1 = sim_results_df_valid[sim_results_df_valid['rt'] - sim_results_df_valid['t_stim'] <= 1]
sim_rt = sim_results_df_valid_lt_1['rt'] - sim_results_df_valid_lt_1['t_stim']

# %%
plt.hist(sim_rt, bins=np.arange(0, 1, 0.01), density=True, histtype='step', label='Simulated')
plt.plot(t_pts_0_1, up_plus_down_mean, label='Theoretical')
plt.legend()
plt.show()
