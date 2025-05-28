# %%
from scipy.integrate import trapezoid
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
# Define desired batches
DESIRED_BATCHES = ['Comparable', 'SD', 'LED2', 'LED1', 'LED34']
# DESIRED_BATCHES = ['Comparable', 'SD', 'LED1', 'LED34']
# DESIRED_BATCHES = ['LED7']

# Base directory paths
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_dir = os.path.join(base_dir, 'batch_csvs')
results_dir = base_dir  # Directory containing the pickle files

# Dynamically discover available pickle files for the desired batches
def find_batch_animal_pairs():
    pairs = []
    pattern = os.path.join(results_dir, 'results_*_animal_*.pkl')
    pickle_files = glob.glob(pattern)
    
    for pickle_file in pickle_files:
        # Extract batch name and animal ID from filename
        filename = os.path.basename(pickle_file)
        # Format: results_{batch}_animal_{animal}.pkl
        parts = filename.split('_')
        
        if len(parts) >= 4:
            batch_index = parts.index('animal') - 1 if 'animal' in parts else 1
            animal_index = parts.index('animal') + 1 if 'animal' in parts else 2
            
            batch_name = parts[batch_index]
            animal_id = parts[animal_index].split('.')[0]  # Remove .pkl extension
            
            if batch_name in DESIRED_BATCHES:
                pairs.append((batch_name, animal_id))
        else:
            print(f"Warning: Invalid filename format: {filename}")
    
    return pairs

# Get batch-animal pairs from available pickle files
batch_animal_pairs = find_batch_animal_pairs()
print(f"Found {len(batch_animal_pairs)} batch-animal pairs: {batch_animal_pairs}")

# %%
# function that takes batch, animal, stim and gives RTD with RTs < 1
def get_animal_RTD_data(batch_name, animal_id, ABL, ILD, bins):
    file_name = f'batch_csvs/batch_{batch_name}_valid_and_aborts.csv'
    df = pd.read_csv(file_name)
    
    # get valid trials of that animal for stim
    df = df[(df['animal'] == animal_id) & (df['ABL'] == ABL) & (df['ILD'] == ILD) &\
                 (df['success'].isin([1, -1]))]

    # Calculate bin centers regardless of whether data exists
    bin_centers = (bins[:-1] + bins[1:]) / 2

    if df.empty:
        print(f"No data found for batch {batch_name}, animal {animal_id}, ABL {ABL}, ILD {ILD}. Returning NaNs.")
        # Return NaN array of same length as bin_centers
        rtd_hist = np.full_like(bin_centers, np.nan)
        return bin_centers, rtd_hist

    # < 1s
    df = df[df['RTwrtStim'] <= 1]
    
    if len(df) == 0:
        print(f"No trials with RTwrtStim <= 1 for batch {batch_name}, animal {animal_id}, ABL {ABL}, ILD {ILD}. Returning NaNs.")
        rtd_hist = np.full_like(bin_centers, np.nan)
        return bin_centers, rtd_hist

    # calculate RTD
    rtd_hist, _ = np.histogram(df['RTwrtStim'], bins=bins, density=True)
    
    return bin_centers, rtd_hist
    
# %%
# function that gets theoretical RTD

def get_params_from_animal_pkl_file(batch_name, animal_id):
    # read the pkl file
    pkl_file = f'results_{batch_name}_animal_{animal_id}.pkl'
    with open(pkl_file, 'rb') as f:
        fit_results_data = pickle.load(f)
    
    vbmc_aborts_param_keys_map = {
        'V_A_samples': 'V_A',
        'theta_A_samples': 'theta_A',
        't_A_aff_samp': 't_A_aff'
    }
    

    vbmc_norm_tied_param_keys_map = {
        'rate_lambda_samples': 'rate_lambda',
        'T_0_samples': 'T_0',
        'theta_E_samples': 'theta_E',
        'w_samples': 'w',
        't_E_aff_samples': 't_E_aff',
        'del_go_samples': 'del_go',
        'rate_norm_l_samples': 'rate_norm_l'
    }
    
    abort_keyname = "vbmc_aborts_results"
    norm_tied_keyname = "vbmc_norm_tied_results"

    abort_params = {}
    norm_tied_params = {}

    if abort_keyname in fit_results_data:
        abort_samples = fit_results_data[abort_keyname]
        for param_samples_name, param_label in vbmc_aborts_param_keys_map.items():
            abort_params[param_label] = np.mean(abort_samples[param_samples_name])
    
    if norm_tied_keyname in fit_results_data:
        norm_tied_samples = fit_results_data[norm_tied_keyname]
        for param_samples_name, param_label in vbmc_norm_tied_param_keys_map.items():
            norm_tied_params[param_label] = np.mean(norm_tied_samples[param_samples_name])
    
    return abort_params, norm_tied_params

    
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


def get_theoretical_RTD_from_params(P_A_mean, C_A_mean, t_stim_samples, abort_params, tied_params, ABL, ILD):
    phi_params_obj = np.nan
    rate_norm_l = tied_params.get('rate_norm_l', np.nan)
    is_norm = True
    is_time_vary = False
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
                                is_norm, is_time_vary, K_max) \
                                - \
                                cum_pro_and_reactive_time_vary_fn(
                                t_stim, T_trunc,
                                abort_params['V_A'], abort_params['theta_A'], abort_params['t_A_aff'],
                                t_stim, ABL, ILD, tied_params['rate_lambda'], tied_params['T_0'], tied_params['theta_E'], Z_E, tied_params['t_E_aff'],
                                phi_params_obj, rate_norm_l, 
                                is_norm, is_time_vary, K_max) + 1e-10
    trunc_factor = np.mean(trunc_fac_samples)
    
    up_mean = np.array([up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn(
                t, 1,
                P_A_mean[i], C_A_mean[i],
                ABL, ILD, tied_params['rate_lambda'], tied_params['T_0'], tied_params['theta_E'], Z_E, tied_params['t_E_aff'], tied_params['del_go'],
                phi_params_obj, rate_norm_l, 
                is_norm, is_time_vary, K_max) for i, t in enumerate(t_pts)])
    down_mean = np.array([up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn(
            t, -1,
            P_A_mean[i], C_A_mean[i],
            ABL, ILD, tied_params['rate_lambda'], tied_params['T_0'], tied_params['theta_E'], Z_E, tied_params['t_E_aff'], tied_params['del_go'],
            phi_params_obj, rate_norm_l, 
            is_norm, is_time_vary, K_max) for i, t in enumerate(t_pts)])
            
    
   
    mask_0_1 = (t_pts >= 0) & (t_pts <= 1)
    t_pts_0_1 = t_pts[mask_0_1]
    up_mean_0_1 = up_mean[mask_0_1]
    down_mean_0_1 = down_mean[mask_0_1]
    
    # Normalize theory curves
    up_theory_mean_norm = up_mean_0_1 / trunc_factor
    down_theory_mean_norm = down_mean_0_1 / trunc_factor

    up_plus_down_mean = up_theory_mean_norm + down_theory_mean_norm
    return t_pts_0_1, up_plus_down_mean
    
    
# %%
### TEST THAT THEORY and SIM agree ###
# abort_params, tied_params = get_params_from_animal_pkl_file('SD', 50)
# p,c,ts_samp = get_P_A_C_A('SD', 50, abort_params)
# ABL = 20
# ILD = 2
# t_pts_0_1,ud = get_theoretical_RTD_from_params(p, c, ts_samp, abort_params, tied_params, ABL, ILD)



# N_sim = int(1e4)

# N_print = N_sim // 5
# dt = 1e-4

# is_norm = True
# rate_norm_l = tied_params.get('rate_norm_l', np.nan)

# is_time_vary = False
# phi_params_obj = np.nan

# batch = 'SD'
# animal_id = 50
# file_name = f'batch_csvs/batch_{batch}_valid_and_aborts.csv'
# df = pd.read_csv(file_name)
# df_animal = df[df['animal'] == animal_id]

# t_stim_samples = df_animal['intended_fix'].sample(N_sim, replace=True).values
# Z_E = (tied_params['w'] - 0.5) * 2 * tied_params['theta_E']
# sim_results = Parallel(n_jobs=30)(
#     delayed(psiam_tied_data_gen_wrapper_rate_norm_fn)(
#         abort_params['V_A'], abort_params['theta_A'], ABL, ILD, tied_params['rate_lambda'], tied_params['T_0'], \
#             tied_params['theta_E'], Z_E, abort_params['t_A_aff'], tied_params['t_E_aff'], tied_params['del_go'], 
#         t_stim_samples[iter_num], rate_norm_l, iter_num, N_print, dt
#     ) for iter_num in tqdm(range(N_sim))
# )

# sim_results_df = pd.DataFrame(sim_results)
# sim_results_df_valid = sim_results_df[sim_results_df['rt'] > sim_results_df['t_stim']]
# sim_results_df_valid_lt_1 = sim_results_df_valid[sim_results_df_valid['rt'] - sim_results_df_valid['t_stim'] <= 1]
# sim_rt = sim_results_df_valid_lt_1['rt'] - sim_results_df_valid_lt_1['t_stim']


# plt.hist(sim_rt, bins=np.arange(0, 1, 0.01), density=True, histtype='step', label='Simulated')
# plt.plot(t_pts_0_1, ud, label='Theoretical')

# plt.legend()
# plt.show()


# %%
# SERIAL
# ABL_arr = [20, 40, 60]
# ILD_arr = [-16., -8., -4., -2., -1., 1., 2., 4., 8., 16.]

# # ABL_arr = [20]
# # ILD_arr = [4]

# # Create bins for RTD histograms
# rt_bins = np.arange(0, 1.02, 0.02)  # 0 to 1 second in 0.01s bins

# # Dictionary to store RTD data for each batch-animal-stimulus combination
# rtd_data = {}

# # Iterate through each batch-animal pair
# for batch_name, animal_id in tqdm(batch_animal_pairs):
# # for batch_name, animal_id in [batch_animal_pairs[0]]:

#     print(f"Processing batch {batch_name}, animal {animal_id}")
    
#     # Initialize nested dictionary for this batch-animal pair
#     if (batch_name, animal_id) not in rtd_data:
#         rtd_data[(batch_name, animal_id)] = {}
    
#     # Try to get parameters for this animal
#     try:
#         abort_params, vanilla_tied_params = get_params_from_animal_pkl_file(batch_name, int(animal_id))
#         p_a, c_a, ts_samp = get_P_A_C_A(batch_name, int(animal_id), abort_params)
        
#         # Iterate through each ABL-ILD combination (stimulus)
#         for abl in ABL_arr:
#             print(f"Animal = {batch_name},{animal_id}, Processing ABL {abl}")
#             for ild in ILD_arr:
#                 stim_key = (abl, ild)
                
#                 try:
#                     # Get empirical RTD data
#                     bin_centers, rtd_hist = get_animal_RTD_data(batch_name, int(animal_id), abl, ild, rt_bins)
                    
#                     # Get theoretical RTD data
#                     try:
#                         t_pts_0_1, up_plus_down = get_theoretical_RTD_from_params(
#                             p_a, c_a, ts_samp, abort_params, vanilla_tied_params, abl, ild
#                         )
#                     except Exception as e:
#                         print(f"  Error calculating theoretical RTD for ABL={abl}, ILD={ild}: {str(e)}")
#                         # Create NaN arrays of appropriate size for theoretical data
#                         t_pts_0_1 = np.linspace(0, 1, 100)  # Typical range for t_pts_0_1
#                         up_plus_down = np.full_like(t_pts_0_1, np.nan)
                    
#                     # Store both empirical and theoretical data
#                     rtd_data[(batch_name, animal_id)][stim_key] = {
#                         'empirical': {
#                             'bin_centers': bin_centers,
#                             'rtd_hist': rtd_hist
#                         },
#                         'theoretical': {
#                             't_pts': t_pts_0_1,
#                             'rtd': up_plus_down
#                         }
#                     }
                    
#                     print(f"  Processed stimulus ABL={abl}, ILD={ild}")
                    
#                 except Exception as e:
#                     print(f"  Error processing stimulus ABL={abl}, ILD={ild}: {str(e)}")
#                     continue
                
#     except Exception as e:
#         print(f"Error processing batch {batch_name}, animal {animal_id}: {str(e)}")
#         continue



# print(f"Completed processing {len(rtd_data)} batch-animal pairs")

# %% 
# PARALLEL
ABL_arr = [20, 40, 60]
ILD_arr = [-16., -8., -4., -2., -1., 1., 2., 4., 8., 16.]

# Create bins for RTD histograms
rt_bins = np.arange(0, 1.02, 0.02)  # 0 to 1 second in 0.02s bins

# Track animals and ABLs included in analyses
included_animals_rtd = []  # All animals are now included for RTD
included_animals_psychometric = []  # Animals with at least one valid ABL
included_abls_psychometric = {}  # Dictionary to track which ABLs were included for each animal

# Function to process a single batch-animal pair
def process_batch_animal(batch_animal_pair):
    batch_name, animal_id = batch_animal_pair
    print(f"Processing batch {batch_name}, animal {animal_id}")
    
    # Dictionary to store RTD data for this batch-animal pair
    animal_rtd_data = {}
    
    try:
        # Try to get parameters for this animal
        abort_params, tied_params = get_params_from_animal_pkl_file(batch_name, int(animal_id))
        p_a, c_a, ts_samp = get_P_A_C_A(batch_name, int(animal_id), abort_params)
        
        # Iterate through each ABL-ILD combination (stimulus)
        for abl in ABL_arr:
            print(f"Animal = {batch_name},{animal_id}, Processing ABL {abl}")
            for ild in ILD_arr:
                stim_key = (abl, ild)
                
                try:
                    # Get empirical RTD data
                    bin_centers, rtd_hist = get_animal_RTD_data(batch_name, int(animal_id), abl, ild, rt_bins)
                    
                    # Get theoretical RTD data
                    try:
                        t_pts_0_1, up_plus_down = get_theoretical_RTD_from_params(
                            p_a, c_a, ts_samp, abort_params, tied_params, abl, ild
                        )
                    except Exception as e:
                        print(f"  Error calculating theoretical RTD for ABL={abl}, ILD={ild}: {str(e)}")
                        # Create NaN arrays of appropriate size for theoretical data
                        t_pts_0_1 = np.linspace(0, 1, 100)  # Typical range for t_pts_0_1
                        up_plus_down = np.full_like(t_pts_0_1, np.nan)
                    
                    # Store both empirical and theoretical data
                    animal_rtd_data[stim_key] = {
                        'empirical': {
                            'bin_centers': bin_centers,
                            'rtd_hist': rtd_hist
                        },
                        'theoretical': {
                            't_pts': t_pts_0_1,
                            'rtd': up_plus_down
                        }
                    }
                    
                    print(f"  Processed stimulus ABL={abl}, ILD={ild}")
                    
                except Exception as e:
                    print(f"  Error processing stimulus ABL={abl}, ILD={ild}: {str(e)}")
                    # Just continue to the next stimulus, no longer skipping the animal
                    continue
        
        # Always include this animal in the RTD analysis, regardless of missing data
        included_animals_rtd.append(batch_animal_pair)
        return batch_animal_pair, animal_rtd_data
                
    except Exception as e:
        print(f"Error processing batch {batch_name}, animal {animal_id}: {str(e)}")
        return batch_animal_pair, {}

# Determine number of CPU cores to use (leave 1 core free for system processes)
n_jobs = max(1, os.cpu_count() - 1)
print(f"Running with {n_jobs} parallel jobs")

# Process all batch-animal pairs in parallel
results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(process_batch_animal)(batch_animal_pair) for batch_animal_pair in batch_animal_pairs
)

# Combine results from all parallel processes
rtd_data = {}
for batch_animal_pair, animal_rtd_data in results:
    if animal_rtd_data:  # Only add if data was successfully processed
        rtd_data[batch_animal_pair] = animal_rtd_data


print(f"Completed processing {len(rtd_data)} batch-animal pairs")

# %%
# PLOT data average and theory average for each stimulus




fig, axes = plt.subplots(3, 10, figsize=(20, 8), sharex=True, sharey=True)

# Remove top and right margins from all subplots
for ax_row in axes:
    for ax in ax_row:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

# Calculate average RTD data and theoretical curves for each ABL-ILD combination
for i, abl in enumerate(ABL_arr):
    for j, ild in enumerate(ILD_arr):
        stim_key = (abl, ild)
        
        # Lists to collect data for averaging
        empirical_rtds = []
        theoretical_rtds = []
        bin_centers = None
        t_pts = None
        
        # Collect data from all animals for this stimulus
        for batch_animal_pair, animal_data in rtd_data.items():
            if stim_key in animal_data:
                # Get empirical data
                emp_data = animal_data[stim_key]['empirical']
                if not np.all(np.isnan(emp_data['rtd_hist'])):
                    empirical_rtds.append(emp_data['rtd_hist'])
                    bin_centers = emp_data['bin_centers']
                
                # Get theoretical data
                theo_data = animal_data[stim_key]['theoretical']
                if not np.all(np.isnan(theo_data['rtd'])):
                    theoretical_rtds.append(theo_data['rtd'])
                    t_pts = theo_data['t_pts']
        
        # Plot average data if available
        ax = axes[i, j]
        
        if empirical_rtds and bin_centers is not None:
            # Calculate average empirical RTD
            avg_empirical_rtd = np.nanmean(empirical_rtds, axis=0)
            ax.plot(bin_centers, avg_empirical_rtd, 'b-', linewidth=1.5, label='Data')
        
        if theoretical_rtds and t_pts is not None:
            # Calculate average theoretical RTD
            avg_theoretical_rtd = np.nanmean(theoretical_rtds, axis=0)
            ax.plot(t_pts, avg_theoretical_rtd, 'r-', linewidth=1.5, label='Theory')
        
        # Set title with ABL and ILD values
        ax.set_title(f'ABL={abl}, ILD={ild}', fontsize=10)
        
        # Only add x-label for bottom row
        if i == 2:
            ax.set_xlabel('RT (s)')
        
        # Only add y-label for leftmost column
        if j == 0:
            ax.set_ylabel('Density')

# Add a single legend for the entire figure
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=2)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)  # Make room for the legend

# Save the figure
plt.savefig('rtd_average_by_stimulus.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
###### Psychometric ####################

# Function to get empirical psychometric data for an animal in a batch for each ABL
def get_animal_psychometric_data(batch_name, animal_id, ABL):
    file_name = f'batch_csvs/batch_{batch_name}_valid_and_aborts.csv'
    df = pd.read_csv(file_name)
    
    # Get valid trials of that animal for the specified ABL
    df = df[(df['animal'] == animal_id) & (df['ABL'] == ABL) & 
            (df['success'].isin([1, -1]))]
    
    
    if df.empty:
        print(f"No data found for batch {batch_name}, animal {animal_id}, ABL {ABL}. Returning NaNs.")
        # Return empty results
        return None
    
    df = df[df['RTwrtStim'] <= 1]
    
    # Get unique ILD values
    ild_values = sorted(df['ILD'].unique())
    right_choice_probs = []
    
    # Calculate probability of right choice for each ILD
    for ild in ild_values:
        ild_trials = df[df['ILD'] == ild]
        if len(ild_trials) > 0:
            # Calculate proportion of right choices (choice == 1)
            right_prob = np.mean(ild_trials['choice'] == 1)
            right_choice_probs.append(right_prob)
        else:
            right_choice_probs.append(np.nan)
    
    return {
        'ild_values': np.array(ild_values),
        'right_choice_probs': np.array(right_choice_probs)
    }

# Function to fit a 4-parameter sigmoid to psychometric data
def fit_psychometric_sigmoid(ild_values, right_choice_probs):
    from scipy.optimize import curve_fit
    
    # Define 4-parameter sigmoid function
    def sigmoid(x, base, amplitude, inflection, slope):
        # Calculate sigmoid values and clip to [0,1] range
        values = base + amplitude / (1 + np.exp(-slope * (x - inflection)))
        return np.clip(values, 0, 1)
    
    # Initial parameter guess
    p0 = [0.0, 1.0, 0.0, 1.0]  # [base, amplitude, inflection, slope]
    
    # Filter out NaN values before fitting
    valid_idx = ~np.isnan(right_choice_probs)
    if np.sum(valid_idx) < 4:  # Need at least 4 points for 4 parameters
        return None
    
    x = ild_values[valid_idx]
    y = right_choice_probs[valid_idx]
    
    try:
        # Fit sigmoid function to data
        popt, _ = curve_fit(sigmoid, x, y, p0=p0)
        
        # Return fitted parameters and function with explicit clipping
        return {
            'params': popt,
            'sigmoid_fn': lambda x: np.clip(sigmoid(x, *popt), 0, 1)  # Ensure output is in [0,1] range
        }
    except Exception as e:
        print(f"Error fitting sigmoid: {str(e)}")
        return None

# Function to process psychometric data for a batch-animal pair
def process_batch_animal_psychometric(batch_animal_pair):
    batch_name, animal_id = batch_animal_pair
    print(f"Processing psychometric data for batch {batch_name}, animal {animal_id}")
    
    # Dictionary to store psychometric data for this batch-animal pair
    animal_psychometric_data = {}
    
    # Track which ABLs have complete data for this animal
    valid_abls = []
    
    try:
        # Get the animal's data from CSV
        file_name = f'batch_csvs/batch_{batch_name}_valid_and_aborts.csv'
        df = pd.read_csv(file_name)
        df_animal = df[(df['animal'] == animal_id) & (df['success'].isin([1, -1]))]
        df_animal = df_animal[df_animal['RTwrtStim'] <= 1]  # Only trials with RT <= 1s
        
        # Process each ABL
        for abl in [20, 40, 60]:  # The 3 ABLs we're interested in
            # Print ILD data types for debugging (only once)
            if abl == 20 and batch_animal_pair == batch_animal_pairs[0]:
                print(f"ILD data type in DataFrame: {df_animal['ILD'].dtype}")
                print(f"ILD data type in ILD_arr: {type(ILD_arr[0])}")
                print(f"Sample values from DataFrame: {df_animal['ILD'].head().tolist()}")
                print(f"Sample values from ILD_arr: {ILD_arr[:3]}")
            
            # Check if all ILDs exist for this ABL
            missing_ilds = []
            for ild in ILD_arr:
                # Use approximate float comparison with small tolerance to handle floating point precision issues
                subset = df_animal[(df_animal['ABL'] == abl) & (df_animal['ILD'].apply(lambda x: abs(float(x) - float(ild)) < 1e-5))]
                if len(subset) == 0:
                    missing_ilds.append(ild)
            
            if missing_ilds:
                suffix = '...' if len(missing_ilds) > 3 else ''
                # print(f"  ABL={abl} is missing data for {len(missing_ilds)} ILDs: {missing_ilds[:3]}{suffix}")
                # print(f"  Skipping ABL={abl} for animal {batch_name},{animal_id}")
                continue  # Skip this ABL but continue with other ABLs
            
            print('Accepted animal = {batch_name},{animal_id} for ABL={abl}')
            try:
                # Get empirical psychometric data
                psychometric_data = get_animal_psychometric_data(batch_name, int(animal_id), abl)
                
                if psychometric_data is not None:
                    # Fit sigmoid to the data
                    fit_result = fit_psychometric_sigmoid(
                        psychometric_data['ild_values'], 
                        psychometric_data['right_choice_probs']
                    )
                    
                    # Store the data and fit
                    animal_psychometric_data[abl] = {
                        'empirical': psychometric_data,
                        'fit': fit_result
                    }
                    
                    valid_abls.append(abl)  # Record that this ABL has valid data
                    # print(f"  Processed ABL={abl}")
                else:
                    print(f"  No data for ABL={abl}. Skipping this ABL.")
                    
            except Exception as e:
                print(f"  Error processing ABL={abl}: {str(e)}")
                # Skip only this ABL, not the entire animal
                continue
        
        # Include the animal if at least one ABL was processed successfully
        if valid_abls:
            included_animals_psychometric.append(batch_animal_pair)
            included_abls_psychometric[batch_animal_pair] = valid_abls
            print(f"Animal {batch_name},{animal_id} included with ABLs: {valid_abls}")
            return batch_animal_pair, animal_psychometric_data
        else:
            print(f"Animal {batch_name},{animal_id} has no valid ABLs, skipping entirely.")
            return batch_animal_pair, {}
        
    except Exception as e:
        print(f"Error processing psychometric data for {batch_name}, animal {animal_id}: {str(e)}")
        return batch_animal_pair, {}

# %%
# Process psychometric data for all batch-animal pairs in parallel
def run_psychometric_processing():
    # Determine number of CPU cores to use (leave 1 core free for system processes)
    n_jobs = max(1, os.cpu_count() - 1)
    print(f"Running psychometric processing with {n_jobs} parallel jobs")
    
    # Process all batch-animal pairs in parallel
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_batch_animal_psychometric)(batch_animal_pair) for batch_animal_pair in batch_animal_pairs
    )
    
    # Combine results from all parallel processes
    psychometric_data = {}
    for batch_animal_pair, animal_psychometric_data in results:
        if animal_psychometric_data:  # Only add if data was successfully processed
            psychometric_data[batch_animal_pair] = animal_psychometric_data
    
    print(f"Completed psychometric processing for {len(psychometric_data)} batch-animal pairs")
    return psychometric_data

# %%
# Plot average psychometric data and fits
def plot_average_psychometric_data(psychometric_data):
    # Create figure with 3 subplots (one for each ABL)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    # Define ILD range for smooth curves
    ild_smooth = np.linspace(-16, 16, 100)
    
    # Define ABL values and colors
    abls = [20, 40, 60]
    abl_colors = ['b', 'g', 'r']
    
    # Process each ABL
    for i, abl in enumerate(abls):
        ax = axes[i]
        
        # Lists to collect individual sigmoid fits
        individual_fits = []
        
        # Collect fitted curves from all animals for this ABL
        for batch_animal_pair, animal_data in psychometric_data.items():
            if abl in animal_data and animal_data[abl]['fit'] is not None:
                # Get fitted curve
                fit = animal_data[abl]['fit']
                fit_values = [fit['sigmoid_fn'](x) for x in ild_smooth]
                individual_fits.append(fit_values)
                
                # Plot individual animal sigmoid fit with low alpha
                ax.plot(ild_smooth, fit_values, color=abl_colors[i], alpha=0.4, linewidth=1)
        
        # Plot average fit if available
        if individual_fits:
            avg_fit = np.nanmean(individual_fits, axis=0)
            ax.plot(ild_smooth, avg_fit, color=abl_colors[i], linewidth=3, label=f'ABL={abl}')
        
        # Set title and labels
        ax.set_title(f'ABL = {abl} dB')
        ax.set_xlabel('ILD (dB)')
        if i == 0:
            ax.set_ylabel('P(right choice)')
        
        # Set axis limits
        ax.set_xlim(-16, 16)
        ax.set_ylim(0, 1)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add legend
        ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig('average_psychometric_by_abl.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

# %%
# Data psychometric - sigmoid averages
# Run the psychometric data processing and plotting
# Process the psychometric data
psychometric_data = run_psychometric_processing()

# Plot the average psychometric data
fig = plot_average_psychometric_data(psychometric_data)

print("Saved psychometric data to psychometric_data.pkl")


# %%
# Theoretical psychometric - sigmoid averages

# Function to calculate theoretical psychometric data for a given animal and ABL
def get_theoretical_psychometric_data(batch_name, animal_id, ABL):
    # Get animal parameters
    try:
        abort_params, norm_tied_params = get_params_from_animal_pkl_file(batch_name, int(animal_id))
        p_a, c_a, ts_samp = get_P_A_C_A(batch_name, int(animal_id), abort_params)
        
        # Define ILD range
        ild_values = np.array([-16., -8., -4., -2., -1., 1., 2., 4., 8., 16.])
        right_choice_probs = []
        
        # Calculate theoretical right choice probability for each ILD
        for ild in ild_values:
            try:
                # Get theoretical RTD for up (rightward) and down (leftward) choices
                t_pts_0_1, up_mean, down_mean = get_theoretical_RTD_up_down(
                    p_a, c_a, ts_samp, abort_params, norm_tied_params, ABL, ild
                )
                
                # Calculate probability of rightward choice using the correct formula
                # Integrate both curves and take the ratio of the up (right) area to total area
                up_area = trapezoid(up_mean, t_pts_0_1)
                down_area = trapezoid(down_mean, t_pts_0_1)
                right_prob = up_area / (up_area + down_area)  # Correct formula
                
                right_choice_probs.append(right_prob)
            except Exception as e:
                print(f"  Error calculating theoretical psychometric for ABL={ABL}, ILD={ild}: {str(e)}")
                right_choice_probs.append(np.nan)
        
        return {
            'ild_values': ild_values,
            'right_choice_probs': np.array(right_choice_probs)
        }
    except Exception as e:
        print(f"Error getting parameters for batch {batch_name}, animal {animal_id}: {str(e)}")
        return None

# Function to get both up (rightward) and down (leftward) theoretical RTD
def get_theoretical_RTD_up_down(P_A_mean, C_A_mean, t_stim_samples, abort_params, tied_params, ABL, ILD):
    phi_params_obj = np.nan
    rate_norm_l = tied_params.get('rate_norm_l', np.nan)
    is_norm = True
    is_time_vary = False
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
                                is_norm, is_time_vary, K_max) \
                                - \
                                cum_pro_and_reactive_time_vary_fn(
                                t_stim, T_trunc,
                                abort_params['V_A'], abort_params['theta_A'], abort_params['t_A_aff'],
                                t_stim, ABL, ILD, tied_params['rate_lambda'], tied_params['T_0'], tied_params['theta_E'], Z_E, tied_params['t_E_aff'],
                                phi_params_obj, rate_norm_l, 
                                is_norm, is_time_vary, K_max) + 1e-10
    trunc_factor = np.mean(trunc_fac_samples)
    
    # Calculate the up (rightward) mean
    up_mean = np.array([up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn(
                t, 1,  # 1 = up/right
                P_A_mean[i], C_A_mean[i],
                ABL, ILD, tied_params['rate_lambda'], tied_params['T_0'], tied_params['theta_E'], Z_E, tied_params['t_E_aff'], tied_params['del_go'],
                phi_params_obj, rate_norm_l, 
                is_norm, is_time_vary, K_max) for i, t in enumerate(t_pts)])
    
    # Calculate the down (leftward) mean
    down_mean = np.array([up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn(
                t, -1,  # -1 = down/left
                P_A_mean[i], C_A_mean[i],
                ABL, ILD, tied_params['rate_lambda'], tied_params['T_0'], tied_params['theta_E'], Z_E, tied_params['t_E_aff'], tied_params['del_go'],
                phi_params_obj, rate_norm_l, 
                is_norm, is_time_vary, K_max) for i, t in enumerate(t_pts)])
   
    # Get time points in the 0-1 range
    mask_0_1 = (t_pts >= 0) & (t_pts <= 1)
    t_pts_0_1 = t_pts[mask_0_1]
    up_mean_0_1 = up_mean[mask_0_1]
    down_mean_0_1 = down_mean[mask_0_1]
    
    # Normalize theory curves
    up_theory_mean_norm = up_mean_0_1 / trunc_factor
    down_theory_mean_norm = down_mean_0_1 / trunc_factor

    return t_pts_0_1, up_theory_mean_norm, down_theory_mean_norm

# Function to process theoretical psychometric data for a batch-animal pair
def process_batch_animal_theoretical_psychometric(batch_animal_pair):
    batch_name, animal_id = batch_animal_pair
    print(f"Processing theoretical psychometric data for batch {batch_name}, animal {animal_id}")
    
    # Dictionary to store theoretical psychometric data for this batch-animal pair
    animal_theoretical_psychometric_data = {}
    
    # Process each ABL
    for abl in [20, 40, 60]:  # The 3 ABLs we're interested in
        try:
            # Get theoretical psychometric data
            psychometric_data = get_theoretical_psychometric_data(batch_name, int(animal_id), abl)
            
            if psychometric_data is not None:
                # Fit sigmoid to the data
                fit_result = fit_psychometric_sigmoid(
                    psychometric_data['ild_values'], 
                    psychometric_data['right_choice_probs']
                )
                
                # Store the data and fit
                animal_theoretical_psychometric_data[abl] = {
                    'theoretical': psychometric_data,
                    'fit': fit_result
                }
                
                print(f"  Processed theoretical ABL={abl}")
            else:
                # Store NaN data if ABL is absent
                animal_theoretical_psychometric_data[abl] = {
                    'theoretical': None,
                    'fit': None
                }
                print(f"  No theoretical data for ABL={abl}")
                
        except Exception as e:
            print(f"  Error processing theoretical ABL={abl}: {str(e)}")
            # Store NaN data for this ABL
            animal_theoretical_psychometric_data[abl] = {
                'theoretical': None,
                'fit': None
            }
    
    return batch_animal_pair, animal_theoretical_psychometric_data

# Process theoretical psychometric data for all batch-animal pairs in parallel
def run_theoretical_psychometric_processing():
    # Determine number of CPU cores to use (leave 1 core free for system processes)
    n_jobs = max(1, os.cpu_count() - 1)
    print(f"Running theoretical psychometric processing with {n_jobs} parallel jobs")
    
    # Process all batch-animal pairs in parallel
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_batch_animal_theoretical_psychometric)(batch_animal_pair) for batch_animal_pair in batch_animal_pairs
    )
    
    # Combine results from all parallel processes
    theoretical_psychometric_data = {}
    for batch_animal_pair, animal_theoretical_psychometric_data in results:
        if animal_theoretical_psychometric_data:  # Only add if data was successfully processed
            theoretical_psychometric_data[batch_animal_pair] = animal_theoretical_psychometric_data
    
    print(f"Completed theoretical psychometric processing for {len(theoretical_psychometric_data)} batch-animal pairs")
    return theoretical_psychometric_data

# Plot theoretical psychometric data and fits
def plot_theoretical_psychometric_data(theoretical_psychometric_data):
    # Create figure with 3 subplots (one for each ABL)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    # Define ILD range for smooth curves
    ild_smooth = np.linspace(-16, 16, 100)
    
    # Define ABL values and colors
    abls = [20, 40, 60]
    abl_colors = ['b', 'g', 'r']
    
    # Process each ABL
    for i, abl in enumerate(abls):
        ax = axes[i]
        
        # Lists to collect individual sigmoid fits
        individual_fits = []
        
        # Collect fitted curves from all animals for this ABL
        for batch_animal_pair, animal_data in theoretical_psychometric_data.items():
            if abl in animal_data and animal_data[abl]['fit'] is not None:
                # Get fitted curve
                fit = animal_data[abl]['fit']
                fit_values = [fit['sigmoid_fn'](x) for x in ild_smooth]
                individual_fits.append(fit_values)
                
                # Plot individual animal sigmoid fit with low alpha
                ax.plot(ild_smooth, fit_values, color=abl_colors[i], alpha=0.4, linewidth=1)
        
        # Plot average fit if available
        if individual_fits:
            avg_fit = np.nanmean(individual_fits, axis=0)
            ax.plot(ild_smooth, avg_fit, color=abl_colors[i], linewidth=3, label=f'ABL={abl}')
        
        # Set title and labels
        ax.set_title(f'Theoretical ABL = {abl} dB')
        ax.set_xlabel('ILD (dB)')
        if i == 0:
            ax.set_ylabel('P(right choice)')
        
        # Set axis limits
        ax.set_xlim(-16, 16)
        ax.set_ylim(0, 1)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add legend
        ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig('theoretical_psychometric_by_abl.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

# Run the theoretical psychometric data processing and plotting
theoretical_psychometric_data = run_theoretical_psychometric_processing()

# Plot the theoretical average psychometric data
theory_fig = plot_theoretical_psychometric_data(theoretical_psychometric_data)
# %%
# Compare empirical data vs theoretical psychometrics

# Function to plot combined empirical and theoretical psychometric curves
def plot_combined_psychometric_data(psychometric_data, theoretical_psychometric_data):
    # Create figure with 3 subplots (one for each ABL)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    # Define ILD range for smooth curves
    ild_smooth = np.linspace(-16, 16, 100)
    
    # Define ABL values
    abls = [20, 40, 60]
    
    # Process each ABL
    for i, abl in enumerate(abls):
        ax = axes[i]
        
        # --- EMPIRICAL DATA ---
        # Lists to collect individual sigmoid fits for empirical data
        empirical_individual_fits = []
        
        # Collect fitted curves from all animals for this ABL
        for batch_animal_pair, animal_data in psychometric_data.items():
            if abl in animal_data and animal_data[abl]['fit'] is not None:
                # Get fitted curve
                fit = animal_data[abl]['fit']
                fit_values = np.clip([fit['sigmoid_fn'](x) for x in ild_smooth], 0, 1)
                empirical_individual_fits.append(fit_values)
                
                # Plot individual animal sigmoid fit with low alpha in blue
                ax.plot(ild_smooth, fit_values, color='blue', alpha=0.2, linewidth=1)
        
        # Plot average empirical fit if available
        if empirical_individual_fits:
            avg_empirical_fit = np.nanmean(empirical_individual_fits, axis=0)
            ax.plot(ild_smooth, avg_empirical_fit, color='blue', linewidth=3, label='Data Average')
        
        # --- THEORETICAL DATA ---
        # Lists to collect individual sigmoid fits for theoretical data
        theoretical_individual_fits = []
        
        # Collect theoretical fitted curves from all animals for this ABL
        for batch_animal_pair, animal_data in theoretical_psychometric_data.items():
            if abl in animal_data and animal_data[abl]['fit'] is not None:
                # Get fitted curve
                fit = animal_data[abl]['fit']
                fit_values = np.clip([fit['sigmoid_fn'](x) for x in ild_smooth], 0, 1)
                theoretical_individual_fits.append(fit_values)
        
        # Plot average theoretical fit if available
        if theoretical_individual_fits:
            avg_theoretical_fit = np.nanmean(theoretical_individual_fits, axis=0)
            ax.plot(ild_smooth, avg_theoretical_fit, color='red', linewidth=3, label='Theory Average')
        
        # Set title and labels
        ax.set_title(f'ABL = {abl} dB')
        ax.set_xlabel('ILD (dB)')
        if i == 0:
            ax.set_ylabel('P(right choice)')
        
        # Set axis limits
        ax.set_xlim(-16, 16)
        ax.set_ylim(0, 1)
        
        # Set custom ticks
        ax.set_xticks([-15, -5, 5, 15])
        ax.set_yticks([0, 0.5, 1])
        
        # Add reference lines
        ax.axhline(y=0.5, color='grey', alpha=0.5, linestyle='-')  # Horizontal line at 0.5
        ax.axvline(x=0, color='grey', alpha=0.5, linestyle='-')    # Vertical line at 0
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add legend
        ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig('combined_psychometric_by_abl.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

# Run the combined plotting
combined_fig = plot_combined_psychometric_data(psychometric_data, theoretical_psychometric_data)

# %%
# Single plot with all data and theory averages together
def plot_all_averages_together(psychometric_data, theoretical_psychometric_data):
    # Create a single figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define ILD range for smooth curves
    ild_smooth = np.linspace(-16, 16, 100)
    
    # Define ABL values and colors
    abls = [20, 40, 60]
    abl_colors = ['b', 'g', 'r']
    
    # Process each ABL
    for i, abl in enumerate(abls):
        color = abl_colors[i]
        
        # --- EMPIRICAL DATA AVERAGE ---
        empirical_individual_fits = []
        
        # Collect empirical fitted curves for this ABL
        for batch_animal_pair, animal_data in psychometric_data.items():
            if abl in animal_data and animal_data[abl]['fit'] is not None:
                fit = animal_data[abl]['fit']
                fit_values = [fit['sigmoid_fn'](x) for x in ild_smooth]
                empirical_individual_fits.append(fit_values)
        
        # Plot average empirical fit if available (dotted line)
        if empirical_individual_fits:
            avg_empirical_fit = np.nanmean(empirical_individual_fits, axis=0)
            ax.plot(ild_smooth, avg_empirical_fit, color=color, linestyle='--', linewidth=2, 
                   label=f'Data ABL={abl}')
        
        # --- THEORETICAL DATA AVERAGE ---
        theoretical_individual_fits = []
        
        # Collect theoretical fitted curves for this ABL
        for batch_animal_pair, animal_data in theoretical_psychometric_data.items():
            if abl in animal_data and animal_data[abl]['fit'] is not None:
                fit = animal_data[abl]['fit']
                fit_values = [fit['sigmoid_fn'](x) for x in ild_smooth]
                theoretical_individual_fits.append(fit_values)
        
        # Plot average theoretical fit if available (solid line)
        if theoretical_individual_fits:
            avg_theoretical_fit = np.nanmean(theoretical_individual_fits, axis=0)
            ax.plot(ild_smooth, avg_theoretical_fit, color=color, linestyle='-', linewidth=2, 
                   label=f'Theory ABL={abl}')
    
    # Set title and labels
    ax.set_title('Average Psychometric Curves: Data vs Theory', fontsize=14)
    ax.set_xlabel('ILD (dB)', fontsize=12)
    ax.set_ylabel('P(right choice)', fontsize=12)
    
    # Set axis limits and ticks
    ax.set_xlim(-16, 16)
    ax.set_ylim(0, 1)
    ax.set_xticks([-15, -5, 5, 15])
    ax.set_yticks([0, 0.5, 1])
    
    # Add reference lines
    ax.axhline(y=0.5, color='grey', alpha=0.5, linestyle='-')  # Horizontal line at 0.5
    ax.axvline(x=0, color='grey', alpha=0.5, linestyle='-')    # Vertical line at 0
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add legend with smaller font size
    ax.legend(fontsize=10, loc='best')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig('all_averages_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

# Run the all-averages plot
all_averages_fig = plot_all_averages_together(psychometric_data, theoretical_psychometric_data)
# %%
# Get all animal keys and determine grid size
animal_keys = list(theoretical_psychometric_data.keys())
total_animals = len(animal_keys)
print(f'Total animals: {total_animals}')

# Calculate number of rows needed (5 animals per row)
row_count = (total_animals + 4) // 5  # Ceiling division to get number of rows

# Create a figure with subplots arranged in a grid, 5 per row
fig, axes = plt.subplots(row_count, 5, figsize=(20, 4*row_count))

# Make axes 2D if there's only one row
if row_count == 1:
    axes = axes.reshape(1, -1)

# Define colors for different ABLs
abl_colors = {20: 'blue', 40: 'green', 60: 'red'}

# Create plots for each animal
for i, key in enumerate(animal_keys):
    # Calculate row and column position
    row = i // 5
    col = i % 5
    
    # Get the appropriate axis
    ax = axes[row, col]
    
    psycho_animal = theoretical_psychometric_data[key]
    
    # Set title for each subplot
    ax.set_title(f'Animal {key}')
    
    # Plot each ABL
    for abl in [20, 40, 60]:
        if abl in psycho_animal:
            psycho_abl = psycho_animal[abl]
            ild_values = psycho_abl['theoretical']['ild_values']
            right_choice_probs = psycho_abl['theoretical']['right_choice_probs']
            ax.plot(ild_values, right_choice_probs, color=abl_colors[abl], label=f'ABL {abl}')
    
    # Add reference lines
    ax.axhline(y=0.5, color='grey', alpha=0.5, linestyle='--')  # Horizontal line at 0.5
    ax.axvline(x=0, color='grey', alpha=0.5, linestyle='--')    # Vertical line at 0
    
    # Set axis labels and limits
    ax.set_xlabel('ILD (dB)')
    if col == 0:  # Only add y-label for leftmost plots
        ax.set_ylabel('P(right choice)')
    ax.set_xlim(-16, 16)
    ax.set_ylim(0, 1)
    
    # Add legend only for the first subplot
    if i == 0:
        ax.legend()

# Hide empty subplots
for i in range(total_animals, row_count * 5):
    row = i // 5
    col = i % 5
    axes[row, col].axis('off')

# Adjust layout
plt.tight_layout()
plt.savefig('theoretical_psychometric_by_animal_grid.png', dpi=300, bbox_inches='tight')
plt.show()
# %%
# check if theory and sim agree
### TEST THAT THEORY and SIM agree ###


# Set up the simulation to compare psychometric curves between simulation and theory
batch = 'SD'
animal_id = 50

# Get model parameters
abort_params, tied_params = get_params_from_animal_pkl_file(batch, animal_id)

# Simulation settings
N_sim = int(1e6)
N_print = N_sim // 5
dt = 1e-3

# Model settings
is_norm = True
rate_norm_l = tied_params.get('rate_norm_l', np.nan)
is_time_vary = False
phi_params_obj = np.nan

# Load animal data for getting intended_fix sample distribution
file_name = f'batch_csvs/batch_{batch}_valid_and_aborts.csv'
df = pd.read_csv(file_name)
df_animal = df[df['animal'] == animal_id]

# Define ILD and ABL values to test
ild_values = np.array([-16., -8., -4., -2., -1., 1., 2., 4., 8., 16.])
abl_values = np.array([20, 40, 60])

# Create figure for psychometric curves
fig, axes = plt.subplots(1, len(abl_values), figsize=(15, 5), sharey=True)
fig.suptitle(f'Psychometric Curves: Simulated vs Theoretical for Animal {animal_id}, Batch {batch}')

# Sample intended_fix times from animal data
t_stim_samples = df_animal['intended_fix'].sample(N_sim, replace=True).values
Z_E = (tied_params['w'] - 0.5) * 2 * tied_params['theta_E']

# Generate random ABL and ILD samples from the set of values
ABL_samples = np.random.choice(abl_values, N_sim)
ILD_samples = np.random.choice(ild_values, N_sim)

# Run a single large simulation with all samples
print("Running simulation for all conditions...")
sim_results = Parallel(n_jobs=30)(
    delayed(psiam_tied_data_gen_wrapper_rate_norm_fn)(
        abort_params['V_A'], abort_params['theta_A'], ABL_samples[i], ILD_samples[i],
        tied_params['rate_lambda'], tied_params['T_0'], tied_params['theta_E'], Z_E,
        abort_params['t_A_aff'], tied_params['t_E_aff'], tied_params['del_go'],
        t_stim_samples[i], rate_norm_l, i, N_print, dt
    ) for i in tqdm(range(N_sim))
)

# Convert to DataFrame and filter for valid trials
sim_results_df = pd.DataFrame(sim_results)
sim_results_df_valid = sim_results_df[
    (sim_results_df['rt'] > sim_results_df['t_stim']) & 
    (sim_results_df['rt'] - sim_results_df['t_stim'] <= 1)
]

# Process each ABL level
for abl_idx, abl in enumerate(abl_values):
    print(f"Processing ABL = {abl}")
    
    # Get theoretical psychometric data for this ABL
    theo_psycho = get_theoretical_psychometric_data(batch, animal_id, abl)
    
    # Calculate simulated psychometric curve for this ABL
    sim_right_probs = []
    
    # For each ILD value, calculate the probability of rightward choice
    for ild in ild_values:
        # Filter trials with current ABL and ILD
        abl_ild_trials = sim_results_df_valid[
            (sim_results_df_valid['ABL'] == abl) & 
            (sim_results_df_valid['ILD'] == ild)
        ]
        
        # Calculate probability of right choice (choice == 1)
        if len(abl_ild_trials) > 0:
            right_prob = (abl_ild_trials['choice'] == 1).mean()
            print(f"  ILD={ild}, trials={len(abl_ild_trials)}, P(right)={right_prob:.3f}")
        else:
            right_prob = np.nan
            print(f"  ILD={ild}, no valid trials found")
            
        sim_right_probs.append(right_prob)
    
    # Plot simulation results
    axes[abl_idx].plot(ild_values, sim_right_probs, 'bo-', label='Simulation')
    
    # Plot theoretical results if available
    if theo_psycho is not None:
        axes[abl_idx].plot(theo_psycho['ild_values'], theo_psycho['right_choice_probs'], 
                         'r*-', label='Theory')
    
    axes[abl_idx].set_title(f'ABL = {abl} dB')
    axes[abl_idx].set_xlabel('ILD (dB)')
    axes[abl_idx].set_ylim([0, 1])
    axes[abl_idx].grid(True, alpha=0.3)
    
    if abl_idx == 0:
        axes[abl_idx].set_ylabel('P(Right Choice)')
    
    axes[abl_idx].legend()

plt.tight_layout()
plt.savefig(f'psychometric_sim_vs_theory_animal_{animal_id}.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Create a new figure for psychometric data with sigmoid fits using actual experimental data
# Load experimental data from CSV
file_name = f'batch_csvs/batch_{batch}_valid_and_aborts.csv'
df = pd.read_csv(file_name)
df_animal = df[df['animal'] == animal_id]

# Only use valid trials (not aborts) where the animal responded after sound onset
df_valid = df_animal[(df_animal['success'] == 1) | (df_animal['success'] == -1)]

# Create figure
plt.figure(figsize=(12, 4))
n_abls = len(abl_values)
axes = plt.subplots(1, n_abls, figsize=(4*n_abls, 4))[1]

# Set x-axis range for fine sigmoid interpolation
x_interp = np.linspace(-16, 16, 100)

for abl_idx, abl in enumerate(abl_values):
    print(f"Processing psychometric fit for ABL = {abl} using real data")
    
    # Get actual data points for this ABL
    abl_data = df_valid[df_valid['ABL'] == abl]
    
    # Calculate probability of right choice for each ILD
    ild_data = []
    right_probs = []
    trial_counts = []
    
    for ild in sorted(abl_data['ILD'].unique()):
        ild_trials = abl_data[abl_data['ILD'] == ild]
        if len(ild_trials) > 0:
            right_prob = (ild_trials['choice'] == 1).mean()
            ild_data.append(ild)
            right_probs.append(right_prob)
            trial_counts.append(len(ild_trials))
            print(f"  ILD={ild}, trials={len(ild_trials)}, P(right)={right_prob:.3f}")
    
    # Convert to numpy arrays
    ild_data = np.array(ild_data)
    right_probs = np.array(right_probs)
    trial_counts = np.array(trial_counts)
    
    # Plot actual data points with fixed marker size
    if len(ild_data) > 0:
        axes[abl_idx].plot(ild_data, right_probs, 'bo', markersize=8, label='Data')
    
    # Fit sigmoid and plot it
    fit_result = fit_psychometric_sigmoid(ild_data, right_probs)
    
    if fit_result is not None:
        # Generate smooth curve from fitted sigmoid
        y_interp = fit_result['sigmoid_fn'](x_interp)
        axes[abl_idx].plot(x_interp, y_interp, 'r-', label='Sigmoid fit')
        
        # Extract parameters for display
        base, amplitude, inflection, slope = fit_result['params']
        print(f"  Sigmoid fit: base={base:.2f}, amplitude={amplitude:.2f}, inflection={inflection:.2f}, slope={slope:.2f}")
    
    # Set plot properties
    axes[abl_idx].set_title(f'ABL = {abl} dB')
    axes[abl_idx].set_xlabel('ILD (dB)')
    axes[abl_idx].set_xlim([-16, 16])
    axes[abl_idx].set_ylim([0, 1])
    axes[abl_idx].grid(True, alpha=0.3)
    axes[abl_idx].set_yticks([0, 0.5, 1.05])
    axes[abl_idx].axhline(y=1, color='grey', alpha=0.5, linestyle='--')
    
    if abl_idx == 0:
        axes[abl_idx].set_ylabel('P(Right Choice)')
    
    axes[abl_idx].legend()

plt.tight_layout()
plt.savefig(f'psychometric_real_data_fits_animal_{animal_id}.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# ONLY DISCRETE POINTS, NO SMOOTHING
# Get all animal keys and determine grid size
animal_keys = list(theoretical_psychometric_data.keys())
total_animals = len(animal_keys)
print(f'Total animals: {total_animals}')

# Calculate number of rows needed (5 animals per row)
row_count = (total_animals + 4) // 5  # Ceiling division to get number of rows

# Create a figure with subplots arranged in a grid, 5 per row
fig, axes = plt.subplots(row_count, 5, figsize=(20, 4*row_count))

# Make axes 2D if there's only one row
if row_count == 1:
    axes = axes.reshape(1, -1)

# Define colors for different ABLs
abl_colors = {20: 'blue', 40: 'green', 60: 'red'}

# Create plots for each animal
for i, key in enumerate(animal_keys):
    # Calculate row and column position
    row = i // 5
    col = i % 5
    
    # Get the appropriate axis
    ax = axes[row, col]
    
    psycho_animal = theoretical_psychometric_data[key]
    
    # Set title for each subplot
    ax.set_title(f'Animal {key}')
    
    # Plot each ABL
    for abl in [20, 40, 60]:
        if abl in psycho_animal:
            psycho_abl = psycho_animal[abl]
            ild_values = psycho_abl['theoretical']['ild_values']
            right_choice_probs = psycho_abl['theoretical']['right_choice_probs']
            ax.scatter(ild_values, right_choice_probs, color=abl_colors[abl], label=f'ABL {abl}')
            # ax.scatter(animal_psychometric_data[abl]['empirical']['ild_values'], animal_psychometric_data[abl]['empirical']['right_choice_probs'], color=abl_colors[abl], marker='x')
    
    # Add reference lines
    ax.axhline(y=0.5, color='grey', alpha=0.5, linestyle='--')  # Horizontal line at 0.5
    ax.axvline(x=0, color='grey', alpha=0.5, linestyle='--')    # Vertical line at 0
    
    # Set axis labels and limits
    ax.set_xlabel('ILD (dB)')
    if col == 0:  # Only add y-label for leftmost plots
        ax.set_ylabel('P(right choice)')
    ax.set_xlim(-17, 17)
    ax.set_ylim(0, 1)
    
    # Add legend only for the first subplot
    if i == 0:
        ax.legend()

# Hide empty subplots
for i in range(total_animals, row_count * 5):
    row = i // 5
    col = i % 5
    axes[row, col].axis('off')

# Adjust layout
plt.tight_layout()
plt.show()

# %%
# %%
psychometric_data = run_psychometric_processing()
print(psychometric_data)
