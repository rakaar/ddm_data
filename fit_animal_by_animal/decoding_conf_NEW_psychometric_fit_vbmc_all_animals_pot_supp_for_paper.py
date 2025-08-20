# %%
# Psychometric fitting for all animals
import time  # ensure this is only imported once at the top

import numpy as np
from scipy.integrate import cumulative_trapezoid as cumtrapz
# Set non-interactive backend before importing plt to avoid tkinter errors
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend which doesn't require a GUI
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
from collections import defaultdict


# %%
# Define desired batches
MODEL_TYPE = 'vanilla'
DESIRED_BATCHES = ['SD', 'LED34', 'LED6', 'LED8', 'LED7', 'LED34_even']


# Base directory paths
base_dir = '/home/rlab/raghavendra/ddm_data/fit_animal_by_animal'
csv_dir = os.path.join(base_dir, 'batch_csvs')
results_dir = base_dir  # Directory containing the pickle files

# Output directory for psychometric fits
output_dir = '/home/rlab/raghavendra/ddm_data/fit_valid_trials/psycho_fits_4-params-del_E_go_fixed_as_avg'
os.makedirs(output_dir, exist_ok=True)

# Dynamically discover available pickle files for the desired batches
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
def rho_A_t_VEC_fn(t_pts, V_A, theta_A):
    """
    Proactive PDF, takes input as vector, delays should be already subtracted before calling func
    """
    t_pts = np.asarray(t_pts)
    rho = np.zeros_like(t_pts, dtype=float)

    # Apply the formula only where t > 0
    valid_idx = t_pts > 0
    t_valid = t_pts[valid_idx]
    
    rho[valid_idx] = (theta_A / np.sqrt(2 * np.pi * t_valid**3)) * \
        np.exp(-0.5 * (V_A**2) * ((t_valid - theta_A / V_A)**2) / t_valid)
    
    return rho


# Function to perform psychometric fitting for a single animal
def fit_psychometric_for_animal(batch_name, animal_id_str):
    start_time = time.time()
    print(f"\n\n{'='*80}\nProcessing batch {batch_name}, animal {animal_id_str}\n{'='*80}")
    
    try:
        animal_id = int(animal_id_str)
        
        # Get data
        file_name = os.path.join(csv_dir, f'batch_{batch_name}_valid_and_aborts.csv')
        df = pd.read_csv(file_name)
        
        # Get valid trials of that animal
        df = df[(df['animal'] == animal_id) & (df['success'].isin([1, -1]))]
        df = df[df['RTwrtStim'] <= 1]
        
        # If no data for this animal, skip
        if df.empty:
            print(f"No valid data found for batch {batch_name}, animal {animal_id}. Skipping.")
            return None
        
        ABL_arr = df['ABL'].unique()
        ILD_arr = df['ILD'].unique()
        print(f'ABL_arr: {ABL_arr}')
        print(f'ILD_arr: {ILD_arr}')
        
        # Read proactive params from pkl
        pkl_file = os.path.join(results_dir, f'results_{batch_name}_animal_{animal_id}.pkl')
        with open(pkl_file, 'rb') as f:
            fit_results_data = pickle.load(f)
        
        vbmc_aborts_param_keys_map = {
            'V_A_samples': 'V_A',
            'theta_A_samples': 'theta_A',
            't_A_aff_samp': 't_A_aff'
        }
        
        abort_keyname = "vbmc_aborts_results"
        if abort_keyname not in fit_results_data:
            print(f"No abort parameters found for batch {batch_name}, animal {animal_id}. Skipping.")
            return None
            
        abort_samples = fit_results_data[abort_keyname]
        abort_params = {}
        for param_samples_name, param_label in vbmc_aborts_param_keys_map.items():
            abort_params[param_label] = np.mean(abort_samples[param_samples_name])
        
        V_A = abort_params['V_A']
        theta_A = abort_params['theta_A']
        t_A_aff = abort_params['t_A_aff']
        
        # Get fixed t_E_aff and del_go by averaging vanilla and norm tied fits
        vanilla_keyname = "vbmc_vanilla_tied_results"
        norm_keyname = "vbmc_norm_tied_results"
        if vanilla_keyname not in fit_results_data or norm_keyname not in fit_results_data:
            missing = []
            if vanilla_keyname not in fit_results_data:
                missing.append(vanilla_keyname)
            if norm_keyname not in fit_results_data:
                missing.append(norm_keyname)
            raise KeyError(f"Missing required keys in {pkl_file}: {', '.join(missing)}")
        vanilla_tied = fit_results_data[vanilla_keyname]
        norm_tied = fit_results_data[norm_keyname]
        fixed_t_E_aff = 0.5 * (np.mean(vanilla_tied['t_E_aff_samples']) + np.mean(norm_tied['t_E_aff_samples']))
        fixed_del_go = 0.5 * (np.mean(vanilla_tied['del_go_samples']) + np.mean(norm_tied['del_go_samples']))
        print(f"Using fixed t_E_aff={fixed_t_E_aff:.6f}, del_go={fixed_del_go:.6f} for batch {batch_name}, animal {animal_id}")
        
        # Setup for integration
        t_stim_arr = np.arange(0.2, 2.2, 0.1)
        t_pts_pa = np.arange(-1,2,0.001)
        if batch_name == 'LED34_even':
            t_trunc = 0.15
        else:
            t_trunc = 0.3
        PA_vs_t_stim_dict = {}
        CA_vs_t_stim_dict = {}
        print(f'starting integral for multiple t_stim')
        for t_stim_idx, t_stim in enumerate(t_stim_arr):
            # calc RTD
            P_A = np.zeros_like(t_pts_pa)
            mask = (t_pts_pa + t_stim) > t_trunc
            if np.any(mask):
                P_A[mask] = rho_A_t_VEC_fn((t_pts_pa + t_stim - t_A_aff)[mask], V_A, theta_A)

            truncated_PA_area = trapezoid(P_A, t_pts_pa)
            norm_PA = P_A / truncated_PA_area
            C_A = cumtrapz(norm_PA, t_pts_pa, initial = 0)
            PA_vs_t_stim_dict[t_stim] = P_A
            CA_vs_t_stim_dict[t_stim] = C_A

        def compute_integral(ABL, ILD, t_stim_arr, rate_lambda, T_0, theta_E, w):
            rate_norm_l = 0
            is_norm = False
            is_time_vary = False
            integrals = np.zeros_like(t_stim_arr)
            phi_params_obj = np.nan
            K_max = 10
            t_pts = np.arange(-1,2,0.001)
            # t_trunc = 0.3
            for t_stim_idx, t_stim in enumerate(t_stim_arr):

                
                P_A = PA_vs_t_stim_dict[t_stim]
                C_A = CA_vs_t_stim_dict[t_stim]
                
                Z_E = (w - 0.5) * 2 * theta_E
                # up_and_down_start_time = time.time()
                up_mean = up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn_vec(
                    t_pts, 1, P_A, C_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, fixed_t_E_aff, fixed_del_go,
                    np.nan, np.nan, np.nan, np.nan, np.nan,  # int_phi_t_E_g, phi_t_e, int_phi_t_e, int_phi_t2, int_phi_t1
                    rate_norm_l, is_norm, is_time_vary, K_max
                )
                down_mean = up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn_vec(
                    t_pts, -1, P_A, C_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, fixed_t_E_aff, fixed_del_go,
                    np.nan, np.nan, np.nan, np.nan, np.nan,
                    rate_norm_l, is_norm, is_time_vary, K_max
                )

                # up_and_down_end_time = time.time()
                # print(f'Time taken for up and down: {up_and_down_end_time - up_and_down_start_time:.3f} s')                
                mask_0_1 = (t_pts >= 0) & (t_pts <= 1)
                t_pts_0_1 = t_pts[mask_0_1]
                up_mean_0_1 = up_mean[mask_0_1]
                down_mean_0_1 = down_mean[mask_0_1]
                up_area = trapezoid(up_mean_0_1, t_pts_0_1)
                down_area = trapezoid(down_mean_0_1, t_pts_0_1)
                right_prob = up_area / (up_area + down_area)  # Correct formula

                integrals[t_stim_idx] = right_prob
            
            return ((ABL, ILD), integrals)
        
        # Compute log-likelihood for each trial
        def compute_loglike(row, integral_vs_t_stim):
            intended_fix = row['intended_fix']
            ILD = row['ILD']
            ABL = row['ABL']
            choice = 2*row['response_poke'] - 5
        
            t_stim = intended_fix

            p_up_vs_t_stim = integral_vs_t_stim[(ABL, ILD)]
            cubic_interp_func = interp1d(
                t_stim_arr, 
                p_up_vs_t_stim, 
                kind='cubic', 
                fill_value="extrapolate",  # Allows extrapolation outside the data range
                assume_sorted=True         # Assumes input data is sorted
            )
            p_up = cubic_interp_func(t_stim)
        
        
            if choice == 1:
                likelihood = p_up
            elif choice == -1:
                likelihood = 1 - p_up
            
            if likelihood <= 0:
                likelihood = 1e-50
        
            return np.log(likelihood)    
        
        # Define log-likelihood function for VBMC
        def psiam_tied_loglike_fn(params):
            rate_lambda, T_0, theta_E, w = params
            
            integral_vs_t_stim = {}
            tasks = [(ABL, ILD) for ABL in ABL_arr for ILD in ILD_arr]
            results = Parallel(n_jobs=30)(
                delayed(compute_integral)(
                    ABL, ILD, t_stim_arr, rate_lambda, T_0, theta_E, w
                ) for (ABL, ILD) in tasks
            )
            integral_vs_t_stim = {}
            for (ABL, ILD), integrals in results:
                integral_vs_t_stim[(ABL, ILD)] = integrals
        
            all_loglike = Parallel(n_jobs=30)(delayed(compute_loglike)(row, integral_vs_t_stim)\
                                               for _, row in df.iterrows())
        
            loglike = np.sum(all_loglike)
            return loglike
        
        # Define parameter bounds
        rate_lambda_bounds = [0.01, 0.2]
        theta_E_bounds = [10, 80]
        T_0_bounds = [0.01*(1e-3), 2*(1e-3)]
        w_bounds = [0.2, 0.8]

        rate_lambda_plausible_bounds = [0.05, 0.15]
        T_0_plausible_bounds = [0.1*(1e-3), 0.9*(1e-3)]
        theta_E_plausible_bounds = [30, 60]
        w_plausible_bounds = [0.4, 0.6]
        
        # Define prior function
        def trapezoidal_logpdf(x, a, b, c, d):
            if x < a or x > d:
                return -np.inf  # Logarithm of zero
            area = ((b - a) + (d - c)) / 2 + (c - b)
            h_max = 1.0 / area  # Height of the trapezoid to normalize the area to 1
            
            if a <= x <= b:
                pdf_value = ((x - a) / (b - a)) * h_max
            elif b < x < c:
                pdf_value = h_max
            elif c <= x <= d:
                pdf_value = ((d - x) / (d - c)) * h_max
            else:
                pdf_value = 0.0  # This case is redundant due to the initial check
        
            if pdf_value <= 0.0:
                return -np.inf
            else:
                return np.log(pdf_value)
        
        def psiam_tied_prior_fn(params):
            rate_lambda, T_0, theta_E, w = params
        
            rate_lambda_logpdf = trapezoidal_logpdf(rate_lambda, rate_lambda_bounds[0], rate_lambda_plausible_bounds[0], rate_lambda_plausible_bounds[1], rate_lambda_bounds[1])
            theta_E_logpdf = trapezoidal_logpdf(theta_E, theta_E_bounds[0], theta_E_plausible_bounds[0], theta_E_plausible_bounds[1], theta_E_bounds[1])
            T_0_logpdf = trapezoidal_logpdf(T_0, T_0_bounds[0], T_0_plausible_bounds[0], T_0_plausible_bounds[1], T_0_bounds[1])
            w_logpdf = trapezoidal_logpdf(w, w_bounds[0], w_plausible_bounds[0], w_plausible_bounds[1], w_bounds[1])
        
            return rate_lambda_logpdf + T_0_logpdf + theta_E_logpdf + w_logpdf
        
        # Define joint function (prior + likelihood)
        def psiam_tied_joint_fn(params):
            priors = psiam_tied_prior_fn(params) 
            loglike = psiam_tied_loglike_fn(params)
        
            joint = priors + loglike
            return joint
        
        # Run VBMC
        lb = np.array([rate_lambda_bounds[0], T_0_bounds[0], theta_E_bounds[0], w_bounds[0]])
        ub = np.array([rate_lambda_bounds[1], T_0_bounds[1], theta_E_bounds[1], w_bounds[1]])
        
        plb = np.array([rate_lambda_plausible_bounds[0], T_0_plausible_bounds[0], theta_E_plausible_bounds[0], w_plausible_bounds[0]])
        pub = np.array([rate_lambda_plausible_bounds[1], T_0_plausible_bounds[1], theta_E_plausible_bounds[1], w_plausible_bounds[1]])
        
        np.random.seed(49)
        rate_lambda_0 = np.random.uniform(rate_lambda_plausible_bounds[0], rate_lambda_plausible_bounds[1])
        T_0_0 = np.random.uniform(T_0_plausible_bounds[0], T_0_plausible_bounds[1])
        theta_E_0 = np.random.uniform(theta_E_plausible_bounds[0], theta_E_plausible_bounds[1])
        w_0 = np.random.uniform(w_plausible_bounds[0], w_plausible_bounds[1])
        
        x_0 = np.array([rate_lambda_0, T_0_0, theta_E_0, w_0])
        
        print(f"Starting VBMC optimization for batch {batch_name}, animal {animal_id}...")
        vbmc = VBMC(psiam_tied_joint_fn, x_0, lb, ub, plb, pub, options={'display': 'on'})
        vp, results = vbmc.optimize()
        
        # Generate and save corner plot
        vp_samples = vp.sample(int(1e5))[0]
        vp_samples[:,1] = vp_samples[:,1] * 1e3  # Scale T_0 for better visualization
        
        param_labels = ['lambda', 'T0', 'theta_E', 'w']
        percentiles = np.percentile(vp_samples, [1, 99], axis=0)
        _ranges = [(percentiles[0, i], percentiles[1, i]) for i in np.arange(vp_samples.shape[1])]
        
        # Create corner plot
        fig = plt.figure(figsize=(10, 10))
        corner.corner(
            vp_samples,
            labels=param_labels,
            show_titles=True,
            quantiles=[0.025, 0.5, 0.975],
            range=_ranges,
            fig=fig
        )
        plt.suptitle(f"Batch: {batch_name}, Animal: {animal_id}", fontsize=16)
        
        # Save corner plot
        corner_plot_file = os.path.join(output_dir, f"psycho_corner_4-params-del_E_go_fixed_as_avg_{batch_name}_{animal_id}.png")
        plt.savefig(corner_plot_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Convert T_0 back to original scale for saving
        vp_samples[:,1] = vp_samples[:,1] / 1e3
        
        # Save VBMC results
        vbmc_file = os.path.join(output_dir, f"psycho_fit_4-params-del_E_go_fixed_as_avg_{batch_name}_{animal_id}.pkl")
        vbmc.save(vbmc_file, overwrite=True)
        
        end_time = time.time()
        print(f"Completed processing batch {batch_name}, animal {animal_id} in {end_time - start_time:.2f} seconds")
        return (batch_name, animal_id, True)  # Success
        
    except Exception as e:
        end_time = time.time()
        print(f"Error processing batch {batch_name}, animal {animal_id}: {str(e)}")
        print(f"Processing time: {end_time - start_time:.2f} seconds")
        return (batch_name, animal_id, False)  # Failed


# %%
# Main execution
# Get all batch-animal pairs
# batch_animal_pairs is already computed above
print(f"Found {len(batch_animal_pairs)} batch-animal pairs: {batch_animal_pairs}")

# Process each batch-animal pair sequentially
results = []
for batch_name, animal_id_str in batch_animal_pairs:
    result = fit_psychometric_for_animal(batch_name, animal_id_str)
    results.append(result)

# Summarize results
success_count = sum(1 for r in results if r is not None and r[2])
fail_count = sum(1 for r in results if r is not None and not r[2])
skip_count = sum(1 for r in results if r is None)

print(f"\n\n{'='*80}\nProcessing Summary\n{'='*80}")
print(f"Total animals: {len(batch_animal_pairs)}")
print(f"Successfully processed: {success_count}")
print(f"Failed to process: {fail_count}")
print(f"Skipped: {skip_count}")
print(f"{'='*80}")

# %%
