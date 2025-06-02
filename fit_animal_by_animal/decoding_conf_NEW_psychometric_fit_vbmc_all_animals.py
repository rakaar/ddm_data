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


# %%
# Define desired batches
DESIRED_BATCHES = ['SD', 'LED2', 'LED1']
MODEL_TYPE = 'vanilla'
# DESIRED_BATCHES = ['Comparable']

# Base directory paths
base_dir = '/home/rlab/raghavendra/ddm_data/fit_animal_by_animal'
csv_dir = os.path.join(base_dir, 'batch_csvs')
results_dir = base_dir  # Directory containing the pickle files

# Output directory for psychometric fits
output_dir = '/home/rlab/raghavendra/ddm_data/fit_valid_trials/psycho_fits'
os.makedirs(output_dir, exist_ok=True)

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
        
        # Setup for integration
        t_stim_arr = np.arange(0.2, 2.2, 0.1)
        t_pts_pa = np.arange(-1,2,0.001)
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

        def compute_integral(ABL, ILD, t_stim_arr, rate_lambda, T_0, theta_E, w, t_E_aff, del_go):
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
                    t_pts, 1, P_A, C_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go,
                    np.nan, np.nan, np.nan, np.nan, np.nan,  # int_phi_t_E_g, phi_t_e, int_phi_t_e, int_phi_t2, int_phi_t1
                    rate_norm_l, is_norm, is_time_vary, K_max
                )
                down_mean = up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn_vec(
                    t_pts, -1, P_A, C_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go,
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
            choice = row['response_poke']
        
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
        
        
            if choice == 3:
                likelihood = p_up
            elif choice == 2:
                likelihood = 1 - p_up
            
            if likelihood <= 0:
                likelihood = 1e-50
        
            return np.log(likelihood)    
        
        # Define log-likelihood function for VBMC
        def psiam_tied_loglike_fn(params):
            rate_lambda, T_0, theta_E, w, t_E_aff, del_go = params
            
            integral_vs_t_stim = {}
            tasks = [(ABL, ILD) for ABL in ABL_arr for ILD in ILD_arr]
            results = Parallel(n_jobs=30)(
                delayed(compute_integral)(
                    ABL, ILD, t_stim_arr, rate_lambda, T_0, theta_E, w, t_E_aff, del_go
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
        t_E_aff_bounds = [0, 0.2]
        del_go_bounds = [0, 0.2]

        rate_lambda_plausible_bounds = [0.05, 0.15]
        T_0_plausible_bounds = [0.1*(1e-3), 0.9*(1e-3)]
        theta_E_plausible_bounds = [30, 60]
        w_plausible_bounds = [0.4, 0.6]
        t_E_aff_plausible_bounds = [0.05, 0.12]
        del_go_plausible_bounds = [0.1, 0.15]
        
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
            rate_lambda, T_0, theta_E, w, t_E_aff, del_go = params
        
            rate_lambda_logpdf = trapezoidal_logpdf(rate_lambda, rate_lambda_bounds[0], rate_lambda_plausible_bounds[0], rate_lambda_plausible_bounds[1], rate_lambda_bounds[1])
            theta_E_logpdf = trapezoidal_logpdf(theta_E, theta_E_bounds[0], theta_E_plausible_bounds[0], theta_E_plausible_bounds[1], theta_E_bounds[1])
            T_0_logpdf = trapezoidal_logpdf(T_0, T_0_bounds[0], T_0_plausible_bounds[0], T_0_plausible_bounds[1], T_0_bounds[1])
            w_logpdf = trapezoidal_logpdf(w, w_bounds[0], w_plausible_bounds[0], w_plausible_bounds[1], w_bounds[1])
            t_E_aff_logpdf = trapezoidal_logpdf(t_E_aff, t_E_aff_bounds[0], t_E_aff_plausible_bounds[0], t_E_aff_plausible_bounds[1], t_E_aff_bounds[1])
            del_go_logpdf = trapezoidal_logpdf(del_go, del_go_bounds[0], del_go_plausible_bounds[0], del_go_plausible_bounds[1], del_go_bounds[1])
        
            return rate_lambda_logpdf + T_0_logpdf + theta_E_logpdf + w_logpdf + t_E_aff_logpdf + del_go_logpdf
        
        # Define joint function (prior + likelihood)
        def psiam_tied_joint_fn(params):
            priors = psiam_tied_prior_fn(params) 
            loglike = psiam_tied_loglike_fn(params)
        
            joint = priors + loglike
            return joint
        
        # Run VBMC
        lb = np.array([rate_lambda_bounds[0], T_0_bounds[0], theta_E_bounds[0], w_bounds[0], t_E_aff_bounds[0], del_go_bounds[0]])
        ub = np.array([rate_lambda_bounds[1], T_0_bounds[1], theta_E_bounds[1], w_bounds[1], t_E_aff_bounds[1], del_go_bounds[1]])
        
        plb = np.array([rate_lambda_plausible_bounds[0], T_0_plausible_bounds[0], theta_E_plausible_bounds[0], w_plausible_bounds[0], t_E_aff_plausible_bounds[0], del_go_plausible_bounds[0]])
        pub = np.array([rate_lambda_plausible_bounds[1], T_0_plausible_bounds[1], theta_E_plausible_bounds[1], w_plausible_bounds[1], t_E_aff_plausible_bounds[1], del_go_plausible_bounds[1]])
        
        np.random.seed(49)
        rate_lambda_0 = np.random.uniform(rate_lambda_plausible_bounds[0], rate_lambda_plausible_bounds[1])
        T_0_0 = np.random.uniform(T_0_plausible_bounds[0], T_0_plausible_bounds[1])
        theta_E_0 = np.random.uniform(theta_E_plausible_bounds[0], theta_E_plausible_bounds[1])
        w_0 = np.random.uniform(w_plausible_bounds[0], w_plausible_bounds[1])
        t_E_aff_0 = np.random.uniform(t_E_aff_plausible_bounds[0], t_E_aff_plausible_bounds[1])
        del_go_0 = np.random.uniform(del_go_plausible_bounds[0], del_go_plausible_bounds[1])
        
        x_0 = np.array([rate_lambda_0, T_0_0, theta_E_0, w_0, t_E_aff_0, del_go_0])
        
        print(f"Starting VBMC optimization for batch {batch_name}, animal {animal_id}...")
        vbmc = VBMC(psiam_tied_joint_fn, x_0, lb, ub, plb, pub, options={'display': 'on'})
        vp, results = vbmc.optimize()
        
        # Generate and save corner plot
        vp_samples = vp.sample(int(1e5))[0]
        vp_samples[:,1] = vp_samples[:,1] * 1e3  # Scale T_0 for better visualization
        
        param_labels = ['lambda', 'T0', 'theta_E', 'w', 't_E_aff', 'del_go']
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
        corner_plot_file = os.path.join(output_dir, f"psycho_corner_{batch_name}_{animal_id}.png")
        plt.savefig(corner_plot_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Convert T_0 back to original scale for saving
        vp_samples[:,1] = vp_samples[:,1] / 1e3
        
        # Save VBMC results
        vbmc_file = os.path.join(output_dir, f"psycho_fit_{batch_name}_{animal_id}.pkl")
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
batch_animal_pairs = find_batch_animal_pairs()
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
