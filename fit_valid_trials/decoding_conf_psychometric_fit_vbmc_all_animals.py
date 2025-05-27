# %%
# Psychometric fitting for all animals
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from pyvbmc import VBMC
import corner
from psiam_tied_dv_map_utils import rho_A_t_fn, up_RTs_fit_fn, down_RTs_fit_fn, up_RTs_fit_single_t_fn, psiam_tied_data_gen_wrapper, psiam_tied_data_gen_wrapper_V2
import sys
import multiprocessing
from psiam_tied_no_dv_map_utils import cum_A_t_fn, all_RTs_fit_OPTIM_fn
from psiam_tied_no_dv_map_utils import rho_A_t_fn, cum_A_t_fn, rho_E_minus_small_t_NORM_fn
from psiam_tied_dv_map_utils import cum_E_t_fn
from tqdm import tqdm
from scipy.integrate import trapezoid
import random
import glob
import os
import time

from psiam_tied_no_dv_map_utils import CDF_E_minus_small_t_NORM_fn_vectorized, rho_A_t_fn_vectorized, P_small_t_btn_x1_x2_vectorized
from scipy.interpolate import interp1d
import pickle


# %%
# Define desired batches
DESIRED_BATCHES = ['Comparable', 'SD', 'LED2', 'LED1', 'LED34']

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
        t_stim_arr = np.arange(0.2, 1.3, 0.1)
        t_pts_for_integ = np.arange(0, 1, 0.001)
        tasks = [(ABL, ILD) for ABL in ABL_arr for ILD in ILD_arr]
        
        # Define P_up_optim_fn for integration
        def P_up_optim_fn(t, V_A, theta_A, x1, x2, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, bound, K_max, t_A_aff, t_stim):
            return rho_A_t_fn_vectorized(t - t_A_aff, V_A, theta_A) * (
                CDF_E_minus_small_t_NORM_fn_vectorized(t - t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, bound, K_max) + \
                P_small_t_btn_x1_x2_vectorized(x1, x2, t - t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, K_max)
            )
        
        # Compute integral for each ABL-ILD combination
        def compute_integral(ABL, ILD, t_stim_arr, t_pts_for_integ, P_up_optim_fn, V_A, theta_A, x1, x2,
                            rate_lambda, T_0, theta_E, Z_E, bound, K_max, t_A_aff):
            integrals = np.zeros_like(t_stim_arr)
            for t_stim_idx, t_stim in enumerate(t_stim_arr):
                unknown_integ_arr = P_up_optim_fn(t_pts_for_integ, V_A, theta_A, x1, x2,
                                            ABL, ILD, rate_lambda, T_0, theta_E, Z_E,
                                            bound, K_max, t_A_aff, t_stim)
                integrals[t_stim_idx] = trapezoid(unknown_integ_arr, t_pts_for_integ)
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
            rate_lambda, T_0, theta_E, w = params
            Z_E = (w - 0.5) * 2 * theta_E
            
            bound = 1
            K_max = 10
            x1 = 1; x2 = 2
            integral_vs_t_stim = {}
        
            results = Parallel(n_jobs=30)(
                delayed(compute_integral)(
                    ABL, ILD, t_stim_arr, t_pts_for_integ, P_up_optim_fn, V_A, theta_A, x1, x2,
                    rate_lambda, T_0, theta_E, Z_E, bound, K_max, t_A_aff
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
        theta_E_bounds = [30, 60]
        T_0_bounds = [0.1*(1e-3), 1*(1e-3)]
        w_bounds = [0.2, 0.8]
        
        rate_lambda_plausible_bounds = [0.05, 0.09]
        T_0_plausible_bounds = [0.15*(1e-3), 0.5*(1e-3)]
        theta_E_plausible_bounds = [40, 55]
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
            w_logpdf = trapezoidal_logpdf(w, w_bounds[0], 0.5, 0.5, w_bounds[1])
        
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
if __name__ == "__main__":
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
