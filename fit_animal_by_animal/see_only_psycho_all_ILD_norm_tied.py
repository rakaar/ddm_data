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

# %%
# Define desired batches
DESIRED_BATCHES = ['Comparable', 'SD', 'LED2', 'LED1', 'LED34', 'LED6']
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
   

# %% 
# PARALLEL
ABL_arr = [20, 40, 60]
ILD_arr = [-16., -8., -4., -2., -1., 1., 2., 4., 8., 16.]


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



# Function to process psychometric data for a batch-l pair
def process_batch_animal_psychometric(batch_animal_pair):
    batch_name, animal_id = batch_animal_pair
    animal_id = int(animal_id)
    
    # Dictionary to store psychometric data for this batch-animal pair
    animal_psychometric_data = {}
    
    # Track which ABLs have complete data for this animal
    valid_abls = []
    included_animals_psychometric = []
    
    try:
        # Get the animal's data from CSV
        file_name = f'batch_csvs/batch_{batch_name}_valid_and_aborts.csv'
        df = pd.read_csv(file_name)
        df_animal = df[(df['animal'] == animal_id) & (df['success'].isin([1, -1]))]
        df_animal = df_animal[df_animal['RTwrtStim'] <= 1]  # Only trials with RT <= 1s
        # Process each ABL in ABL_arr, include any available data for each ILD in ILD_arr
        for abl in ABL_arr:
            print(f'Processing animal = {batch_name},{animal_id} for ABL={abl}')
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
                        'fit': None
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
            print(f"Animal {batch_name},{animal_id} included with ABLs: {valid_abls}")
            return batch_animal_pair, animal_psychometric_data
        else:
            # Even if no ABLs had full ILD coverage, still return whatever was found (even if empty)
            print(f"Animal {batch_name},{animal_id} has no valid ABLs, but returning empty psychometric data.")
            return batch_animal_pair, animal_psychometric_data
        
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

# %%
# exp data
psychometric_data = run_psychometric_processing()
# %%
print(f'len of theory psycho data = {len(theoretical_psychometric_data)}')
print(f'len of empirical psycho data = {len(psychometric_data)}')


# %%
# ONLY DISCRETE POINTS, NO SMOOTHING
# Get all animal keys and determine grid size
animal_keys = list(psychometric_data.keys())
total_animals = len(animal_keys)
# print(f'Total animals: {len(animal_keys)}')

# Calculate number of rows needed (5 animals per row)
row_count = (total_animals + 4) // 5  # Ceiling division to get number of rows
print(f'Number of animals: {total_animals}')
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
    empirical_animal = psychometric_data[key]
    # Set title for each subplot
    ax.set_title(f'Animal {key}')
    
    # Plot each ABL
    for abl in [20, 40, 60]:
        if abl in psycho_animal:
            psycho_abl = psycho_animal[abl]
            ild_values = psycho_abl['theoretical']['ild_values']
            right_choice_probs = psycho_abl['theoretical']['right_choice_probs']
            ax.scatter(ild_values, right_choice_probs, color=abl_colors[abl], label=f'ABL {abl}')
            if abl in empirical_animal:
                ax.scatter(empirical_animal[abl]['empirical']['ild_values'], empirical_animal[abl]['empirical']['right_choice_probs'], color=abl_colors[abl], marker='x')    
    # Add reference lines
    ax.axhline(y=0.5, color='grey', alpha=0.5, linestyle='--')  # Horizontal line at 0.5
    ax.axvline(x=0, color='grey', alpha=0.5, linestyle='--')    # Vertical line at 0
    
    # Set axis labels and limits
    ax.set_xlabel('ILD (dB)')
    if col == 0:  # Only add y-label for leftmost plots
        ax.set_ylabel('P(right choice)')
    ax.set_xlim(-17, 17)
    ax.set_ylim(0, 1.02)
    
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
# === Start of new code for average psychometric plots ===
# Ensure numpy and pyplot are available (likely already imported as np and plt)
theory_agg = {}
empirical_agg = {}
for abl in [20, 40, 60]:
    theory_agg[abl] = np.full((len(animal_keys), len(ILD_arr)), np.nan)
    empirical_agg[abl] = np.full((len(animal_keys), len(ILD_arr)), np.nan)

for idx, key in enumerate(animal_keys):
    animal_data = psychometric_data[key]
    theory_data = theoretical_psychometric_data[key]
    
    for abl_key in [20, 40, 60]:
        if abl_key in theory_data:
            theory_abl_psycho = theory_data[abl_key]['theoretical']['right_choice_probs']
            theory_agg[abl_key][idx] = theory_abl_psycho
        if abl_key in animal_data:
            empirical = animal_data[abl_key]['empirical']
            # Restrict to ILD_arr only, in order
            ild_values = empirical['ild_values']
            right_choice_probs = empirical['right_choice_probs']
            # Build array for ILD_arr
            selected_probs = np.full(len(ILD_arr), np.nan)
            for i, ild in enumerate(ILD_arr):
                matches = np.where(ild_values == ild)[0]
                if len(matches) > 0:
                    selected_probs[i] = right_choice_probs[matches[0]]
            try:
                empirical_agg[abl_key][idx] = selected_probs
            except:
                print(key, abl_key)
                print('empirical ild_values:', ild_values)
                print('empirical right_choice_probs:', right_choice_probs)
                print('selected_probs:', selected_probs)
        
# %%


# Plot average psychometric curves for each ABL
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for i, abl in enumerate([20, 40, 60]):
    emp = empirical_agg[abl]  # shape: (n_animals, n_ilds)
    theo = theory_agg[abl]
    emp_mean = np.nanmean(emp, axis=0)
    emp_std = np.nanstd(emp, axis=0)
    theo_mean = np.nanmean(theo, axis=0)
    theo_std = np.nanstd(theo, axis=0)

    ax = axes[i]
    ilds = ILD_arr
    # Empirical: blue dots with error bars (std), no caps
    ax.errorbar(ilds, emp_mean, yerr=emp_std, fmt='o', color='blue', label='data', capsize=0)
    ax.plot(ilds, emp_mean, color='blue', linestyle='-', linewidth=2, alpha=0.7)  # Blue line joining dots
    # Theoretical: red dots with error bars (std), no caps
    ax.errorbar(ilds, theo_mean, yerr=theo_std, fmt='o', color='red', label='theory', capsize=0)
    ax.plot(ilds, theo_mean, color='red', linestyle='-', linewidth=2, alpha=0.7)  # Red line joining dots
    ax.set_title(f'ABL = {abl}')
    ax.set_xlabel('ILD (dB)')
    if i == 0:
        ax.set_ylabel('P(choice = right)')
    ax.set_xticks(ilds)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
plt.tight_layout()
plt.show()


# %%
# Plot all three ABLs in a single figure: data (dotted), theory (solid), each ABL a color
import matplotlib.pyplot as plt
import numpy as np

colors = {20: 'tab:blue', 40: 'tab:orange', 60: 'tab:green'}
plt.figure(figsize=(8, 6))
for abl in [20, 40, 60]:
    emp = empirical_agg[abl]
    theo = theory_agg[abl]
    emp_mean = np.nanmean(emp, axis=0)
    theo_mean = np.nanmean(theo, axis=0)
    ilds = ILD_arr
    # Empirical: dotted line
    plt.plot(ilds, emp_mean, linestyle=':', marker='o', color=colors[abl], label=f'Data ABL={abl}')
    # Theory: solid line
    plt.plot(ilds, theo_mean, linestyle='-', marker=None, color=colors[abl], label=f'Theory ABL={abl}')
plt.xlabel('ILD (dB)')
plt.ylabel('P(choice = right)')
plt.title('Average Psychometric Curves (All ABLs)')
plt.xticks([-15,-5,5,15])
plt.yticks([0, 0.5, 1])
plt.axvline(0, alpha=0.5, color='grey')
plt.axhline(0.5, alpha=0.5, color='grey')

plt.ylim(-0.05, 1.05)
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Plot: data for all ABLs in one plot, theory for all ABLs in another plot
import matplotlib.pyplot as plt
import numpy as np

colors = {20: 'tab:blue', 40: 'tab:orange', 60: 'tab:green'}
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Data plot (left)
for abl in [20, 40, 60]:
    emp = empirical_agg[abl]
    emp_mean = np.nanmean(emp, axis=0)
    ilds = ILD_arr
    axes[0].plot(ilds, emp_mean, linestyle=':', marker='o', color=colors[abl], label=f'Data ABL={abl}')
axes[0].set_xlabel('ILD (dB)')
axes[0].set_ylabel('P(choice = right)')
axes[0].set_title('Empirical Data (All ABLs)')
axes[0].set_xticks([-15, -5, 5, 15])
axes[0].set_yticks([0, 0.5, 1])
axes[0].axvline(0, alpha=0.5, color='grey')
axes[0].axhline(0.5, alpha=0.5, color='grey')
axes[0].set_ylim(-0.05, 1.05)
axes[0].legend()

# Theory plot (right)
for abl in [20, 40, 60]:
    theo = theory_agg[abl]
    theo_mean = np.nanmean(theo, axis=0)
    ilds = ILD_arr
    axes[1].plot(ilds, theo_mean, linestyle='-', marker='o', color=colors[abl], label=f'Theory ABL={abl}')
axes[1].set_xlabel('ILD (dB)')
axes[1].set_title('Theoretical (All ABLs)')
axes[1].set_xticks([-15, -5, 5, 15])
axes[1].set_yticks([0, 0.5, 1])
axes[1].axvline(0, alpha=0.5, color='grey')
axes[1].axhline(0.5, alpha=0.5, color='grey')
axes[1].set_ylim(-0.05, 1.05)
axes[1].legend()

plt.tight_layout()
plt.show()
# %%
# Plot all three ABLs in a single figure: data (dotted), theory (solid), each ABL a color

import matplotlib.pyplot as plt
import numpy as np

colors = {20: 'tab:blue', 40: 'tab:orange', 60: 'tab:green'}
plt.figure(figsize=(4, 3))  # Smaller figure for publication
for abl in [20, 40, 60]:
    emp = empirical_agg[abl]
    theo = theory_agg[abl]
    emp_mean = np.nanmean(emp, axis=0)
    theo_mean = np.nanmean(theo, axis=0)
    ilds = np.array(ILD_arr)
    theo_mean = np.array(theo_mean)
    # Empirical: dotted line
    n_emp = np.sum(~np.isnan(emp), axis=0)
    print(f'emp n valid: {n_emp}')
    emp_sem = np.nanstd(emp, axis=0) / np.sqrt(np.maximum(n_emp - 1, 1))
    print(f'denominator: {np.sqrt(np.maximum(n_emp - 1, 1))}')
    plt.errorbar(ilds, emp_mean, yerr=emp_sem, fmt='o', color=colors[abl], label=f'Data ABL={abl}', capsize=0, markersize=4)
    # Logistic fit to theory: solid line
    valid_idx = ~np.isnan(theo_mean)
    if np.sum(valid_idx) >= 4:
        try:
            from scipy.optimize import curve_fit
            def logistic(x, base, amplitude, inflection, slope):
                values = base + amplitude / (1 + np.exp(-slope * (x - inflection)))
                return np.clip(values, 0, 1)
            p0 = [0.0, 1.0, 0.0, 1.0]
            popt, _ = curve_fit(logistic, ilds[valid_idx], theo_mean[valid_idx], p0=p0)
            ilds_smooth = np.linspace(min(ilds), max(ilds), 200)
            fit_curve = logistic(ilds_smooth, *popt)
            plt.plot(ilds_smooth, fit_curve, linestyle='-', color=colors[abl], label=f'Logistic fit (Theory) ABL={abl}')
        except Exception as e:
            print(f"Could not fit logistic for ABL={abl}: {e}")
    else:
        print(f"Not enough valid theory points for ABL={abl} to fit.")
plt.xlabel('ILD (dB)', fontsize=16)
plt.ylabel('P(choice = right)', fontsize=16)
# plt.title('Average Psychometric Curves (All ABLs)')
plt.xticks([-15,-5,5,15], fontsize=14)
plt.yticks([0, 0.5, 1], fontsize=14)
plt.axvline(0, alpha=0.5, color='grey', linestyle='--')
plt.axhline(0.5, alpha=0.5, color='grey', linestyle='--')

plt.ylim(-0.05, 1.05)
# Remove top and right spines for publication style
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()


# %%
# --- Logistic fit slopes for theory and data, print and save ---
import pickle
from scipy.optimize import curve_fit

logistic_fit_results = {'theory': {}, 'data': {}}

for abl in [20, 40, 60]:
    emp = empirical_agg[abl]
    theo = theory_agg[abl]
    emp_mean = np.nanmean(emp, axis=0)
    theo_mean = np.nanmean(theo, axis=0)
    ilds = np.array(ILD_arr)
    # Theory fit
    valid_theo = ~np.isnan(theo_mean)
    if np.sum(valid_theo) >= 4:
        try:
            def logistic(x, base, amplitude, inflection, slope):
                values = base + amplitude / (1 + np.exp(-slope * (x - inflection)))
                return np.clip(values, 0, 1)
            p0 = [0.0, 1.0, 0.0, 1.0]
            popt, _ = curve_fit(logistic, ilds[valid_theo], theo_mean[valid_theo], p0=p0)
            slope = popt[3]
            logistic_fit_results['theory'][abl] = {'params': popt, 'slope': slope}
            print(f"[THEORY] ABL={abl} logistic slope: {slope:.4f}")
        except Exception as e:
            logistic_fit_results['theory'][abl] = {'params': None, 'slope': None, 'error': str(e)}
            print(f"[THEORY] Could not fit logistic for ABL={abl}: {e}")
    else:
        logistic_fit_results['theory'][abl] = {'params': None, 'slope': None, 'error': 'Insufficient valid points'}
        print(f"[THEORY] Not enough valid theory points for ABL={abl} to fit.")
    # Empirical fit
    valid_emp = ~np.isnan(emp_mean)
    if np.sum(valid_emp) >= 4:
        try:
            p0 = [0.0, 1.0, 0.0, 1.0]
            popt, _ = curve_fit(logistic, ilds[valid_emp], emp_mean[valid_emp], p0=p0)
            slope = popt[3]
            logistic_fit_results['data'][abl] = {'params': popt, 'slope': slope}
            print(f"[DATA]   ABL={abl} logistic slope: {slope:.4f}")
        except Exception as e:
            logistic_fit_results['data'][abl] = {'params': None, 'slope': None, 'error': str(e)}
            print(f"[DATA]   Could not fit logistic for ABL={abl}: {e}")
    else:
        logistic_fit_results['data'][abl] = {'params': None, 'slope': None, 'error': 'Insufficient valid points'}
        print(f"[DATA]   Not enough valid data points for ABL={abl} to fit.")

# Save slopes and fit params to pickle
with open('psychometric_logistic_slopes_NORM_TIED.pkl', 'wb') as f:
    pickle.dump(logistic_fit_results, f)
print('Logistic fit slopes and parameters saved to psychometric_logistic_slopes.pkl')

#%%
logistic_fit_results