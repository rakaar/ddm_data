# %%
#!/usr/bin/env python3
"""
Compare log-likelihood values from vanilla+lapse and norm+lapse model fits - V2.
This version manually calculates log-likelihoods from data and parameters
instead of extracting pre-computed values.
"""
import pickle
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from joblib import Parallel, delayed

# Add paths to import utility functions
sys.path.append('../lapses')
from lapses_utils import simulate_psiam_tied_rate_norm
from vbmc_animal_wise_fit_utils import trapezoidal_logpdf
from time_vary_norm_utils import up_or_down_RTs_fit_fn, cum_pro_and_reactive_time_vary_fn

# %%
# Constants
phi_params_obj = np.nan
is_time_vary = False
K_max = 10
DO_RIGHT_TRUNCATE = True

# %%
# Helper functions for manual log-likelihood calculation

def load_animal_data(batch, animal, is_stim_filtered=False):
    """
    Load data for a specific batch and animal.
    Returns df_valid_animal (filtered valid trials).
    """
    csv_filename = f'batch_csvs/batch_{batch}_valid_and_aborts.csv'
    exp_df = pd.read_csv(csv_filename)
    
    df_valid_and_aborts = exp_df[
        (exp_df['success'].isin([1,-1])) |
        (exp_df['abort_event'] == 3)
    ].copy()
    
    df_all_trials_animal = df_valid_and_aborts[df_valid_and_aborts['animal'] == animal]
    df_valid_animal = df_all_trials_animal[df_all_trials_animal['success'].isin([1,-1])]
    
    # Stimulus filtering (ABL and ILD)
    if is_stim_filtered:
        allowed_abls = [20, 40, 60]
        allowed_ilds = [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]
        df_valid_animal = df_valid_animal[
            (df_valid_animal['ABL'].isin(allowed_abls)) & 
            (df_valid_animal['ILD'].isin(allowed_ilds))
        ]
    
    # Right truncation
    if DO_RIGHT_TRUNCATE:
        df_valid_animal = df_valid_animal[df_valid_animal['RTwrtStim'] < 1]
        max_rt = 1
    else:
        max_rt = df_valid_animal['RTwrtStim'].max()
    
    return df_valid_animal, max_rt


def load_abort_params(batch, animal, results_dir):
    """Load abort parameters from results pickle file."""
    pkl_file = os.path.join(results_dir, f'results_{batch}_animal_{animal}.pkl')
    
    if not os.path.exists(pkl_file):
        return None
    
    with open(pkl_file, 'rb') as f:
        fit_results_data = pickle.load(f)
    
    vbmc_aborts_param_keys_map = {
        'V_A_samples': 'V_A',
        'theta_A_samples': 'theta_A',
        't_A_aff_samp': 't_A_aff'
    }
    abort_keyname = "vbmc_aborts_results"
    
    if abort_keyname not in fit_results_data:
        return None
    
    abort_samples = fit_results_data[abort_keyname]
    abort_params = {}
    for param_samples_name, param_label in vbmc_aborts_param_keys_map.items():
        abort_params[param_label] = np.mean(abort_samples[param_samples_name])
    
    return abort_params


def get_T_trunc(batch):
    """Get truncation time based on batch."""
    if batch == 'LED34_even':
        return 0.15
    else:
        return 0.3


def compute_loglike_vanilla(row, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go, 
                            lapse_prob, lapse_prob_right, max_rt, abort_params, T_trunc):
    """
    Compute log-likelihood for a single trial - vanilla + lapse model.
    """
    rt = row['TotalFixTime']
    t_stim = row['intended_fix']
    ILD = row['ILD']
    ABL = row['ABL']
    choice = row['choice']
    lapse_rt_window = max_rt
    
    V_A = abort_params['V_A']
    theta_A = abort_params['theta_A']
    t_A_aff = abort_params['t_A_aff']
    
    rate_norm_l = 0  # Vanilla model doesn't have norm
    is_norm = False
    
    pdf = up_or_down_RTs_fit_fn(
            rt, choice,
            V_A, theta_A, t_A_aff,
            t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go,
            phi_params_obj, rate_norm_l, 
            is_norm, is_time_vary, K_max)
    
    if DO_RIGHT_TRUNCATE:
        trunc_factor_p_joint = cum_pro_and_reactive_time_vary_fn(
                            t_stim + 1, T_trunc,
                            V_A, theta_A, t_A_aff,
                            t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_E_aff,
                            phi_params_obj, rate_norm_l, 
                            is_norm, is_time_vary, K_max)  - \
                            cum_pro_and_reactive_time_vary_fn(
                            t_stim, T_trunc,
                            V_A, theta_A, t_A_aff,
                            t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_E_aff,
                            phi_params_obj, rate_norm_l, 
                            is_norm, is_time_vary, K_max)
    else:
        trunc_factor_p_joint = 1 - cum_pro_and_reactive_time_vary_fn(
                            t_stim, T_trunc,
                            V_A, theta_A, t_A_aff,
                            t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_E_aff,
                            phi_params_obj, rate_norm_l, 
                            is_norm, is_time_vary, K_max)
    
    pdf /= (trunc_factor_p_joint + 1e-20)
    
    # Lapse probability depends on choice direction
    if choice == 1:
        lapse_choice_pdf = lapse_prob_right * (1/lapse_rt_window)
    else:  # choice == -1
        lapse_choice_pdf = (1 - lapse_prob_right) * (1/lapse_rt_window)
    
    included_lapse_pdf = (1 - lapse_prob) * pdf + lapse_prob * lapse_choice_pdf
    included_lapse_pdf = max(included_lapse_pdf, 1e-50)
    
    return np.log(included_lapse_pdf)


def compute_loglike_norm(row, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go, rate_norm_l,
                         lapse_prob, lapse_prob_right, max_rt, abort_params, T_trunc):
    """
    Compute log-likelihood for a single trial - norm + lapse model.
    """
    rt = row['TotalFixTime']
    t_stim = row['intended_fix']
    ILD = row['ILD']
    ABL = row['ABL']
    choice = row['choice']
    lapse_rt_window = max_rt
    
    V_A = abort_params['V_A']
    theta_A = abort_params['theta_A']
    t_A_aff = abort_params['t_A_aff']
    
    is_norm = True
    
    pdf = up_or_down_RTs_fit_fn(
            rt, choice,
            V_A, theta_A, t_A_aff,
            t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go,
            phi_params_obj, rate_norm_l, 
            is_norm, is_time_vary, K_max)
    
    if DO_RIGHT_TRUNCATE:
        trunc_factor_p_joint = cum_pro_and_reactive_time_vary_fn(
                            t_stim + 1, T_trunc,
                            V_A, theta_A, t_A_aff,
                            t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_E_aff,
                            phi_params_obj, rate_norm_l, 
                            is_norm, is_time_vary, K_max)  - \
                            cum_pro_and_reactive_time_vary_fn(
                            t_stim, T_trunc,
                            V_A, theta_A, t_A_aff,
                            t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_E_aff,
                            phi_params_obj, rate_norm_l, 
                            is_norm, is_time_vary, K_max)
    else:
        trunc_factor_p_joint = 1 - cum_pro_and_reactive_time_vary_fn(
                            t_stim, T_trunc,
                            V_A, theta_A, t_A_aff,
                            t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_E_aff,
                            phi_params_obj, rate_norm_l, 
                            is_norm, is_time_vary, K_max)
    
    pdf /= (trunc_factor_p_joint + 1e-20)
    
    # Lapse probability depends on choice direction
    if choice == 1:
        lapse_choice_pdf = lapse_prob_right * (1/lapse_rt_window)
    else:  # choice == -1
        lapse_choice_pdf = (1 - lapse_prob_right) * (1/lapse_rt_window)
    
    included_lapse_pdf = (1 - lapse_prob) * pdf + lapse_prob * lapse_choice_pdf
    included_lapse_pdf = max(included_lapse_pdf, 1e-50)
    
    return np.log(included_lapse_pdf)

# %%

def calculate_loglike_from_vbmc(pkl_path, batch, animal, is_vanilla=True, results_dir='.'):
    """
    Calculate log-likelihood manually from VBMC pickle file.
    
    Args:
        pkl_path: Path to VBMC lapse model pickle
        batch: Batch name
        animal: Animal ID
        is_vanilla: True for vanilla model, False for norm model
        results_dir: Directory containing results files
    
    Returns:
        dict with keys: loglike, elbo_sd, stable, n_iterations
    """
    try:
        # Load VBMC object
        with open(pkl_path, 'rb') as f:
            vbmc = pickle.load(f)
        
        # Extract stability info from iteration_history
        result = {}
        
        if hasattr(vbmc, 'iteration_history'):
            iter_hist = vbmc.iteration_history
            
            if 'elbo_sd' in iter_hist:
                result['elbo_sd'] = float(iter_hist['elbo_sd'][-1])
            else:
                result['elbo_sd'] = None
            
            if 'stable' in iter_hist:
                result['stable'] = bool(iter_hist['stable'][-1])
            else:
                result['stable'] = None
            
            if 'iter' in iter_hist:
                result['n_iterations'] = int(iter_hist['iter'][-1])
            else:
                result['n_iterations'] = len(iter_hist)
            
            # Extract VP object for sampling parameters
            if 'vp' in iter_hist:
                vp_arr = iter_hist['vp']
                last_vp = vp_arr[-1]
            else:
                result['loglike'] = None
                return result
        else:
            result['loglike'] = None
            result['elbo_sd'] = None
            result['stable'] = None
            result['n_iterations'] = None
            return result
        
        # Sample from VP and compute mean parameters
        # sample() returns (samples, log_weights), we need just the samples
        vp_samples, _ = last_vp.sample(int(1e6))
        
        if is_vanilla:
            # Vanilla: rate_lambda, T_0, theta_E, w, t_E_aff, del_go, lapse_prob, lapse_prob_right
            rate_lambda = np.mean(vp_samples[:, 0])
            T_0 = np.mean(vp_samples[:, 1])
            theta_E = np.mean(vp_samples[:, 2])
            w = np.mean(vp_samples[:, 3])
            t_E_aff = np.mean(vp_samples[:, 4])
            del_go = np.mean(vp_samples[:, 5])
            lapse_prob = np.mean(vp_samples[:, 6])
            lapse_prob_right = np.mean(vp_samples[:, 7])
            rate_norm_l = None
        else:
            # Norm: rate_lambda, T_0, theta_E, w, t_E_aff, del_go, rate_norm_l, lapse_prob, lapse_prob_right
            rate_lambda = np.mean(vp_samples[:, 0])
            T_0 = np.mean(vp_samples[:, 1])
            theta_E = np.mean(vp_samples[:, 2])
            w = np.mean(vp_samples[:, 3])
            t_E_aff = np.mean(vp_samples[:, 4])
            del_go = np.mean(vp_samples[:, 5])
            rate_norm_l = np.mean(vp_samples[:, 6])
            lapse_prob = np.mean(vp_samples[:, 7])
            lapse_prob_right = np.mean(vp_samples[:, 8])
        
        Z_E = (w - 0.5) * 2 * theta_E
        
        # Load animal data
        df_valid_animal, max_rt = load_animal_data(batch, animal, is_stim_filtered=False)
        
        # Store number of trials
        n_trials = len(df_valid_animal)
        
        # Load abort parameters
        abort_params = load_abort_params(batch, animal, results_dir)
        if abort_params is None:
            result['loglike'] = None
            result['n_trials'] = None
            return result
        
        # Get T_trunc
        T_trunc = get_T_trunc(batch)
        
        # Calculate log-likelihood for all trials
        if is_vanilla:
            all_loglike = Parallel(n_jobs=-5)(
                delayed(compute_loglike_vanilla)(
                    row, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go,
                    lapse_prob, lapse_prob_right, max_rt, abort_params, T_trunc
                ) for _, row in df_valid_animal.iterrows()
            )
        else:
            all_loglike = Parallel(n_jobs=-5)(
                delayed(compute_loglike_norm)(
                    row, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go, rate_norm_l,
                    lapse_prob, lapse_prob_right, max_rt, abort_params, T_trunc
                ) for _, row in df_valid_animal.iterrows()
            )
        
        result['loglike'] = float(np.sum(all_loglike))
        result['n_trials'] = int(n_trials)
        result['lapse_prob'] = float(lapse_prob)
        result['lapse_prob_right'] = float(lapse_prob_right)
        return result
    
    except Exception as e:
        print(f"Error calculating loglike from {pkl_path}: {e}")
        import traceback
        traceback.print_exc()
        return {'loglike': None, 'n_trials': None, 'elbo_sd': None, 'stable': None, 'n_iterations': None, 'error': str(e)}


def parse_filename_vanilla_lapse(filename):
    """
    Parse vanilla+lapse pickle filename to extract batch and animal.
    Expected format: vbmc_vanilla_tied_results_batch_{batch}_animal_{animal}_lapses_truncate_1s.pkl
    """
    # Remove .pkl extension
    name = filename.replace('.pkl', '')
    
    # Split by underscores
    parts = name.split('_')
    
    # Find batch name (after 'batch_')
    batch_idx = parts.index('batch') + 1
    batch_parts = []
    animal_idx = None
    
    for i in range(batch_idx, len(parts)):
        if parts[i] == 'animal':
            animal_idx = i + 1
            break
        batch_parts.append(parts[i])
    
    batch = '_'.join(batch_parts)
    
    # Get animal ID
    animal = int(parts[animal_idx])
    
    return batch, animal


def parse_filename_norm_lapse(filename):
    """
    Parse norm+lapse pickle filename to extract batch and animal.
    Expected format: vbmc_norm_tied_results_batch_{batch}_animal_{animal}_lapses_truncate_1s_norm.pkl
    """
    # Remove .pkl extension
    name = filename.replace('.pkl', '')
    
    # Split by underscores
    parts = name.split('_')
    
    # Find batch name (after 'batch_')
    batch_idx = parts.index('batch') + 1
    batch_parts = []
    animal_idx = None
    
    for i in range(batch_idx, len(parts)):
        if parts[i] == 'animal':
            animal_idx = i + 1
            break
        batch_parts.append(parts[i])
    
    batch = '_'.join(batch_parts)
    
    # Get animal ID
    animal = int(parts[animal_idx])
    
    return batch, animal


def get_original_loglikes(batch, animal_id, results_dir):
    """
    Load original vanilla and norm log-likelihood values from results pickle.
    
    Returns:
        dict with keys: og_vanilla_loglike, og_norm_loglike
    """
    result = {'og_vanilla_loglike': None, 'og_norm_loglike': None}
    
    # Special handling for LED34 batch
    if batch == 'LED34':
        # For vanilla: read from led34_filter_files/vanila/results_LED34_animal_{id}_VANILLA_ABL_ILD_filtered.pkl
        vanilla_pkl_fname = f'results_LED34_animal_{animal_id}_VANILLA_ABL_ILD_filtered.pkl'
        vanilla_pkl_path = os.path.join(results_dir, 'led34_filter_files', 'vanila', vanilla_pkl_fname)
        
        if os.path.exists(vanilla_pkl_path):
            try:
                with open(vanilla_pkl_path, 'rb') as f:
                    vanilla_results = pickle.load(f)
                # Extract from nested vbmc_vanilla_tied_results dict
                if 'vbmc_vanilla_tied_results' in vanilla_results:
                    result['og_vanilla_loglike'] = vanilla_results['vbmc_vanilla_tied_results'].get('loglike', None)
            except Exception as e:
                print(f"Warning: Could not load vanilla log-likelihood from {vanilla_pkl_path}: {e}")
        
        # For norm: read from led34_filter_files/norm/results_LED34_animal_{id}_NORM_filtered.pkl
        norm_pkl_fname = f'results_LED34_animal_{animal_id}_NORM_filtered.pkl'
        norm_pkl_path = os.path.join(results_dir, 'led34_filter_files', 'norm', norm_pkl_fname)
        
        if os.path.exists(norm_pkl_path):
            try:
                with open(norm_pkl_path, 'rb') as f:
                    norm_results = pickle.load(f)
                # Extract from nested vbmc_norm_tied_results dict
                if 'vbmc_norm_tied_results' in norm_results:
                    result['og_norm_loglike'] = norm_results['vbmc_norm_tied_results'].get('loglike', None)
            except Exception as e:
                print(f"Warning: Could not load norm log-likelihood from {norm_pkl_path}: {e}")
        
        return result
    
    # Standard handling for other batches
    pkl_fname = f'results_{batch}_animal_{animal_id}.pkl'
    pkl_path = os.path.join(results_dir, pkl_fname)
    
    if not os.path.exists(pkl_path):
        return result
    
    try:
        with open(pkl_path, 'rb') as f:
            results = pickle.load(f)
        
        # Extract vanilla log-likelihood
        if 'vbmc_vanilla_tied_results' in results:
            result['og_vanilla_loglike'] = results['vbmc_vanilla_tied_results'].get('loglike', None)
        
        # Extract norm log-likelihood
        if 'vbmc_norm_tied_results' in results:
            result['og_norm_loglike'] = results['vbmc_norm_tied_results'].get('loglike', None)
        
        return result
    except Exception as e:
        print(f"Warning: Could not load original log-likelihoods from {pkl_path}: {e}")
        return result


def format_bool(val):
    """Format boolean for display"""
    if val is None:
        return "N/A"
    return "True" if val else "False"


def format_float(val):
    """Format float for display"""
    if val is None:
        return "N/A"
    return f"{val:.2f}"

# %%
# Configuration
base_dir = '/home/rlab/raghavendra/ddm_data/fit_animal_by_animal'
vanilla_lapse_dir = os.path.join(base_dir, 'oct_9_10_vanila_lapse_model_fit_files')
norm_lapse_dir = os.path.join(base_dir, 'oct_9_10_norm_lapse_model_fit_files')
results_dir = base_dir  # Results files are in the main directory
    
# Find all pickle files in both directories
vanilla_lapse_files = glob.glob(os.path.join(vanilla_lapse_dir, '*.pkl'))
norm_lapse_files = glob.glob(os.path.join(norm_lapse_dir, '*.pkl'))

print(f"Found {len(vanilla_lapse_files)} vanilla+lapse pickle files")
print(f"Found {len(norm_lapse_files)} norm+lapse pickle files")

# Extract batch, animal pairs from vanilla+lapse files and calculate log-likelihoods
vanilla_lapse_data = {}
for pkl_path in vanilla_lapse_files:
    filename = os.path.basename(pkl_path)
    try:
        batch, animal = parse_filename_vanilla_lapse(filename)
        print(f"Calculating vanilla+lapse loglike for {batch} animal {animal}...")
        conv_info = calculate_loglike_from_vbmc(pkl_path, batch, animal, is_vanilla=True, results_dir=results_dir)
        vanilla_lapse_data[(batch, animal)] = conv_info
    except Exception as e:
        print(f"Error processing {filename}: {e}")

# Extract batch, animal pairs from norm+lapse files and calculate log-likelihoods
norm_lapse_data = {}
for pkl_path in norm_lapse_files:
    filename = os.path.basename(pkl_path)
    try:
        batch, animal = parse_filename_norm_lapse(filename)
        print(f"Calculating norm+lapse loglike for {batch} animal {animal}...")
        conv_info = calculate_loglike_from_vbmc(pkl_path, batch, animal, is_vanilla=False, results_dir=results_dir)
        norm_lapse_data[(batch, animal)] = conv_info
    except Exception as e:
        print(f"Error processing {filename}: {e}")

# Find common (batch, animal) pairs
vanilla_keys = set(vanilla_lapse_data.keys())
norm_keys = set(norm_lapse_data.keys())
common_keys = vanilla_keys & norm_keys

print(f"\nFound {len(common_keys)} common (batch, animal) pairs")

# %%
# Build results table
rows = []
for batch, animal in sorted(common_keys):
    vanilla_info = vanilla_lapse_data[(batch, animal)]
    norm_info = norm_lapse_data[(batch, animal)]
    og_loglikes = get_original_loglikes(batch, animal, results_dir)
    
    row = {
        'batch': batch,
        'animal': animal,
        'vanilla_lapse_stable': vanilla_info.get('stable'),
        'norm_lapse_stable': norm_info.get('stable'),
        'vanilla_lapse_loglike': vanilla_info.get('loglike'),
        'norm_lapse_loglike': norm_info.get('loglike'),
        'og_vanilla_loglike': og_loglikes['og_vanilla_loglike'],
        'og_norm_loglike': og_loglikes['og_norm_loglike'],
        'vanilla_lapse_prob': vanilla_info.get('lapse_prob'),
        'vanilla_lapse_prob_right': vanilla_info.get('lapse_prob_right'),
        'norm_lapse_prob': norm_info.get('lapse_prob'),
        'norm_lapse_prob_right': norm_info.get('lapse_prob_right'),
        'n_trials': vanilla_info.get('n_trials'),  # Same for both vanilla and norm
    }
    rows.append(row)

# %%
# Print table header
print("\n" + "="*160)
print("Log-Likelihood Comparison Table")
print("="*160)

# Column headers
header = f"{'Batch':<15} {'Animal':<8} {'V+L Stable':<12} {'N+L Stable':<12} {'V+L LogLike':<14} {'N+L LogLike':<14} {'OG V LogLike':<14} {'OG N LogLike':<14}"
print(header)
print("-" * 160)

# Print rows
for row in rows:
    line = f"{row['batch']:<15} {row['animal']:<8} "
    line += f"{format_bool(row['vanilla_lapse_stable']):<12} "
    line += f"{format_bool(row['norm_lapse_stable']):<12} "
    line += f"{format_float(row['vanilla_lapse_loglike']):<14} "
    line += f"{format_float(row['norm_lapse_loglike']):<14} "
    line += f"{format_float(row['og_vanilla_loglike']):<14} "
    line += f"{format_float(row['og_norm_loglike']):<14}"
    print(line)

# %%
# Summary statistics
print("\n" + "="*160)
print("Summary Statistics")
print("="*160)

print(f"\nTotal animals analyzed: {len(rows)}")

# Count stable
v_stable = sum(1 for row in rows if row['vanilla_lapse_stable'])
n_stable = sum(1 for row in rows if row['norm_lapse_stable'])
print(f"\nVanilla+Lapse stable: {v_stable}/{len(rows)}")
print(f"Norm+Lapse stable: {n_stable}/{len(rows)}")

# Compute log-likelihood differences
v_diffs = []
n_diffs = []
for row in rows:
    if row['vanilla_lapse_loglike'] is not None and row['og_vanilla_loglike'] is not None:
        v_diffs.append(row['vanilla_lapse_loglike'] - row['og_vanilla_loglike'])
    if row['norm_lapse_loglike'] is not None and row['og_norm_loglike'] is not None:
        n_diffs.append(row['norm_lapse_loglike'] - row['og_norm_loglike'])

if v_diffs:
    print(f"\nVanilla+Lapse log-likelihood improvement over original:")
    print(f"  Mean: {sum(v_diffs)/len(v_diffs):.2f}")
    print(f"  Median: {sorted(v_diffs)[len(v_diffs)//2]:.2f}")
    print(f"  Min: {min(v_diffs):.2f}")
    print(f"  Max: {max(v_diffs):.2f}")

if n_diffs:
    print(f"\nNorm+Lapse log-likelihood improvement over original:")
    print(f"  Mean: {sum(n_diffs)/len(n_diffs):.2f}")
    print(f"  Median: {sorted(n_diffs)[len(n_diffs)//2]:.2f}")
    print(f"  Min: {min(n_diffs):.2f}")
    print(f"  Max: {max(n_diffs):.2f}")

# %%
# Save to CSV
output_csv = os.path.join(base_dir, 'vanilla_norm_lapse_loglike_comparison_v2_div_by_N.csv')
with open(output_csv, 'w') as f:
    # Write header
    f.write("batch,animal,vanilla_lapse_stable,norm_lapse_stable,vanilla_lapse_loglike,norm_lapse_loglike,og_vanilla_loglike,og_norm_loglike,n_trials\n")
    # Write rows
    for row in rows:
        f.write(f"{row['batch']},{row['animal']},")
        f.write(f"{row['vanilla_lapse_stable']},{row['norm_lapse_stable']},")
        f.write(f"{row['vanilla_lapse_loglike']},{row['norm_lapse_loglike']},")
        f.write(f"{row['og_vanilla_loglike']},{row['og_norm_loglike']},")
        f.write(f"{row['n_trials']}\n")

print(f"\nResults saved to: {output_csv}")

# %%
# Log-Likelihood Comparison Bar Plots

# Prepare data for plotting
animal_labels = [f"{row['batch']}_{row['animal']}" for row in rows]
x_pos = np.arange(len(rows))

# Compute all four comparisons
comparison_1 = []  # Vanilla+Lapse - Vanilla
comparison_2 = []  # Vanilla+Lapse - Norm
comparison_3 = []  # Norm+Lapse - Norm
comparison_4 = []  # Vanilla+Lapse - Norm+Lapse

for row in rows:
    n_trials = row['n_trials']
    
    # Comparison 1: (Vanilla+Lapse log-likelihood - Vanilla log-likelihood) / n_trials
    if row['vanilla_lapse_loglike'] is not None and row['og_vanilla_loglike'] is not None and n_trials is not None:
        comparison_1.append((row['vanilla_lapse_loglike'] - row['og_vanilla_loglike']) / n_trials)
    else:
        comparison_1.append(0)
    
    # Comparison 2: (Vanilla+Lapse log-likelihood - Norm log-likelihood) / n_trials
    if row['vanilla_lapse_loglike'] is not None and row['og_norm_loglike'] is not None and n_trials is not None:
        comparison_2.append((row['vanilla_lapse_loglike'] - row['og_norm_loglike']) / n_trials)
    else:
        comparison_2.append(0)
    
    # Comparison 3: (Norm+Lapse log-likelihood - Norm log-likelihood) / n_trials
    if row['norm_lapse_loglike'] is not None and row['og_norm_loglike'] is not None and n_trials is not None:
        comparison_3.append((row['norm_lapse_loglike'] - row['og_norm_loglike']) / n_trials)
    else:
        comparison_3.append(0)
    
    # Comparison 4: (Vanilla+Lapse log-likelihood - Norm+Lapse log-likelihood) / n_trials
    if row['vanilla_lapse_loglike'] is not None and row['norm_lapse_loglike'] is not None and n_trials is not None:
        comparison_4.append((row['vanilla_lapse_loglike'] - row['norm_lapse_loglike']) / n_trials)
    else:
        comparison_4.append(0)
# %%
# Create figure with 4 subplots
fig, axes = plt.subplots(4, 1, figsize=(14, 16))

# Plot 1: Vanilla+Lapse - Vanilla
ax1 = axes[0]
colors_1 = ['green' if val > 0 else 'red' for val in comparison_1]
ax1.bar(x_pos, comparison_1, color=colors_1, alpha=0.7)
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax1.set_ylabel('Log-Likelihood Difference Per Trial', fontsize=12, fontweight='bold')
ax1.set_title('(Vanilla+Lapse - Original Vanilla) Log-Likelihood Per Trial', fontsize=14, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(animal_labels, rotation=45, ha='right')
ax1.grid(axis='y', alpha=0.3)
ax1.set_xlabel('Batch_Animal', fontsize=11)
# ax1.set_ylim(-100, 100)

# Plot 2: Vanilla+Lapse - Norm
ax2 = axes[1]
colors_2 = ['green' if val > 0 else 'red' for val in comparison_2]
ax2.bar(x_pos, comparison_2, color=colors_2, alpha=0.7)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.set_ylabel('Log-Likelihood Difference Per Trial', fontsize=12, fontweight='bold')
ax2.set_title('(Vanilla+Lapse - Original Norm) Log-Likelihood Per Trial', fontsize=14, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(animal_labels, rotation=45, ha='right')
ax2.grid(axis='y', alpha=0.3)
ax2.set_xlabel('Batch_Animal', fontsize=11)
# ax2.set_ylim(-100, 100)

# Plot 3: Norm+Lapse - Norm
ax3 = axes[2]
colors_3 = ['green' if val > 0 else 'red' for val in comparison_3]
ax3.bar(x_pos, comparison_3, color=colors_3, alpha=0.7)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax3.set_ylabel('Log-Likelihood Difference Per Trial', fontsize=12, fontweight='bold')
ax3.set_title('(Norm+Lapse - Original Norm) Log-Likelihood Per Trial', fontsize=14, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(animal_labels, rotation=45, ha='right')
ax3.grid(axis='y', alpha=0.3)
ax3.set_xlabel('Batch_Animal', fontsize=11)
# ax3.set_ylim(-100, 100)

# Plot 4: Vanilla+Lapse - Norm+Lapse
ax4 = axes[3]
colors_4 = ['green' if val > 0 else 'red' for val in comparison_4]
ax4.bar(x_pos, comparison_4, color=colors_4, alpha=0.7)
ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax4.set_ylabel('Log-Likelihood Difference Per Trial', fontsize=12, fontweight='bold')
ax4.set_title('(Vanilla+Lapse - Norm+Lapse) Log-Likelihood Per Trial', fontsize=14, fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(animal_labels, rotation=45, ha='right')
ax4.grid(axis='y', alpha=0.3)
ax4.set_xlabel('Batch_Animal', fontsize=11)
# ax4.set_ylim(-100, 100)

plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'loglike_comparisons_bar_plots_v2_div_by_N.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"\nBar plots saved to: {os.path.join(base_dir, 'loglike_comparisons_bar_plots_v2_div_by_N.png')}")

# %%
# Find batches with mixed positive/negative Vanilla+Lapse - Norm log-likelihood differences
# (excluding LED34 batch)

print("\n" + "="*160)
print("Batches with Mixed Signs in Vanilla+Lapse - Norm Log-Likelihood Difference (excluding LED34)")
print("="*160)

# Group by batch
batch_differences = {}
for i, row in enumerate(rows):
    batch = row['batch']
    animal = row['animal']
    diff = comparison_2[i]  # Vanilla+Lapse - Norm
    
    if batch not in batch_differences:
        batch_differences[batch] = []
    
    batch_differences[batch].append({
        'animal': animal,
        'difference': diff
    })

# Find batches with mixed signs (excluding LED34)
mixed_batches = []
for batch, data in sorted(batch_differences.items()):
    # Skip LED34
    if batch == 'LED34':
        continue
    
    # Check if there are both positive and negative differences
    differences = [d['difference'] for d in data]
    has_positive = any(d > 0 for d in differences)
    has_negative = any(d < 0 for d in differences)
    
    if has_positive and has_negative:
        mixed_batches.append(batch)
        print(f"\n{batch}:")
        for d in data:
            sign = "+" if d['difference'] > 0 else "-"
            print(f"  Animal {d['animal']}: {sign}{abs(d['difference']):.2f}")

print(f"\n\nSummary: Found {len(mixed_batches)} batches with mixed signs (excluding LED34):")
print(f"  {', '.join(mixed_batches)}")

# %%
# Find batches with uniform signs (all positive or all negative) in Vanilla+Lapse - Norm log-likelihood
# (excluding LED34 batch)

print("\n" + "="*160)
print("Batches with Uniform Signs in Vanilla+Lapse - Norm Log-Likelihood Difference (excluding LED34)")
print("="*160)

all_positive_batches = []
all_negative_batches = []

for batch, data in sorted(batch_differences.items()):
    # Skip LED34
    if batch == 'LED34':
        continue
    
    # Check if all differences have the same sign
    differences = [d['difference'] for d in data]
    has_positive = any(d > 0 for d in differences)
    has_negative = any(d < 0 for d in differences)
    
    # All positive
    if has_positive and not has_negative:
        all_positive_batches.append(batch)
        print(f"\n{batch} (ALL POSITIVE):")
        for d in data:
            print(f"  Animal {d['animal']}: +{d['difference']:.2f}")
    
    # All negative
    elif has_negative and not has_positive:
        all_negative_batches.append(batch)
        print(f"\n{batch} (ALL NEGATIVE):")
        for d in data:
            print(f"  Animal {d['animal']}: {d['difference']:.2f}")

print(f"\n\nSummary:")
print(f"  All positive ({len(all_positive_batches)} batches): {', '.join(all_positive_batches) if all_positive_batches else 'None'}")
print(f"  All negative ({len(all_negative_batches)} batches): {', '.join(all_negative_batches) if all_negative_batches else 'None'}")
# %%
# Scatter plot: Lapse Probability vs Log-Likelihood Difference
# X-axis: vanilla lapse_prob mean
# Y-axis: Vanilla+Lapse LogLike - Original Norm LogLike

print("\n" + "="*160)
print("Creating scatter plot: Lapse Probability vs Log-Likelihood Improvement")
print("="*160)

# Prepare data for scatter plot
lapse_probs = []
loglike_diffs = []  # vanilla+lapse - og_norm
batch_labels = []

for row in rows:
    vanilla_lapse_ll = row.get('vanilla_lapse_loglike')
    og_norm_ll = row.get('og_norm_loglike')
    lapse_prob = row.get('vanilla_lapse_prob')
    n_trials = row.get('n_trials')
    
    # Only include rows where we have all necessary data
    if (vanilla_lapse_ll is not None and 
        og_norm_ll is not None and 
        lapse_prob is not None and
        n_trials is not None):
        
        loglike_diff = (vanilla_lapse_ll - og_norm_ll) / n_trials
        lapse_probs.append(lapse_prob)
        loglike_diffs.append(loglike_diff)
        batch_labels.append(f"{row['batch']}_{row['animal']}")

# Create scatter plot
fig, ax = plt.subplots(figsize=(10, 7))

# Color points by whether they improve (green) or worsen (red) the log-likelihood
colors = ['green' if diff > 0 else 'red' for diff in loglike_diffs]

ax.scatter(np.array(lapse_probs)*100, loglike_diffs, c=colors, alpha=0.6, s=100, edgecolors='black', linewidth=0.5)

# Add horizontal line at y=0
ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, label='No difference')

# Labels and title
ax.set_xlabel('Lapse Probability % (from Vanilla+Lapse Model)', fontsize=14, fontweight='bold')
ax.set_ylabel('Log-Likelihood Difference Per Trial\n(Vanilla+Lapse - Original Norm)', fontsize=14, fontweight='bold')
ax.set_title('Lapse Probability vs Log-Likelihood Per Trial Improvement Over Norm Model', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)
# ax.legend()
ax.axvline(x=1, alpha=0.4)

# Add text annotation with correlation
if len(lapse_probs) > 0:
    correlation = np.corrcoef(lapse_probs, loglike_diffs)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
scatter_plot_path = os.path.join(base_dir, 'lapse_prob_vs_loglike_improvement_v2_div_by_N.png')
plt.savefig(scatter_plot_path, dpi=150, bbox_inches='tight')
plt.show()

print(f"\nScatter plot saved to: {scatter_plot_path}")
print(f"Number of animals plotted: {len(lapse_probs)}")
if len(lapse_probs) > 0:
    print(f"Lapse probability range: [{min(lapse_probs):.4f}, {max(lapse_probs):.4f}]")
    print(f"Log-likelihood difference range: [{min(loglike_diffs):.2f}, {max(loglike_diffs):.2f}]")
    print(f"Correlation coefficient: {correlation:.3f}")

# %%
# Scatter plot: Lapse Probability Consistency between Vanilla+Lapse and Norm+Lapse
# X-axis: lapse_prob from vanilla+lapse
# Y-axis: lapse_prob from norm+lapse

print("\n" + "="*160)
print("Creating scatter plot: Lapse Probability Consistency (Vanilla+Lapse vs Norm+Lapse)")
print("="*160)

# Prepare data
vanilla_lapse_probs = []
norm_lapse_probs = []
consistency_labels = []

for row in rows:
    vanilla_lp = row.get('vanilla_lapse_prob')
    norm_lp = row.get('norm_lapse_prob')
    
    # Only include rows where we have both lapse probabilities
    if vanilla_lp is not None and norm_lp is not None:
        vanilla_lapse_probs.append(vanilla_lp)
        norm_lapse_probs.append(norm_lp)
        consistency_labels.append(f"{row['batch']}_{row['animal']}")

# Create scatter plot
fig, ax = plt.subplots(figsize=(10, 10))

# Scatter plot
ax.scatter(vanilla_lapse_probs, norm_lapse_probs, alpha=0.6, s=100, 
           c='blue', edgecolors='black', linewidth=0.5, label='Animals')

# Add diagonal line y=x (perfect consistency)
if len(vanilla_lapse_probs) > 0:
    min_val = min(min(vanilla_lapse_probs), min(norm_lapse_probs))
    max_val = max(max(vanilla_lapse_probs), max(norm_lapse_probs))
    ax.plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=2, label='Perfect consistency (y=x)')

# Labels and title
ax.set_xlabel('Lapse Probability (Vanilla+Lapse Model)', fontsize=14, fontweight='bold')
ax.set_ylabel('Lapse Probability (Norm+Lapse Model)', fontsize=14, fontweight='bold')
ax.set_title('Lapse Probability Consistency Between Models', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)
# ax.legend(fontsize=12)
ax.set_aspect('equal', adjustable='box')

# Add correlation and statistics
if len(vanilla_lapse_probs) > 0:
    correlation = np.corrcoef(vanilla_lapse_probs, norm_lapse_probs)[0, 1]
    
    # Calculate mean absolute difference
    differences = np.array(vanilla_lapse_probs) - np.array(norm_lapse_probs)
    mean_abs_diff = np.mean(np.abs(differences))
    
    stats_text = f'Correlation: {correlation:.3f}\nMean |Δ|: {mean_abs_diff:.4f}'
    ax.text(0.05, 0.95, stats_text, 
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
consistency_plot_path = os.path.join(base_dir, 'lapse_prob_consistency_vanilla_vs_norm_v2.png')
plt.savefig(consistency_plot_path, dpi=150, bbox_inches='tight')
plt.show()

print(f"\nConsistency scatter plot saved to: {consistency_plot_path}")
print(f"Number of animals plotted: {len(vanilla_lapse_probs)}")
if len(vanilla_lapse_probs) > 0:
    print(f"Vanilla lapse prob range: [{min(vanilla_lapse_probs):.4f}, {max(vanilla_lapse_probs):.4f}]")
    print(f"Norm lapse prob range: [{min(norm_lapse_probs):.4f}, {max(norm_lapse_probs):.4f}]")
    print(f"Correlation coefficient: {correlation:.3f}")
    print(f"Mean absolute difference: {mean_abs_diff:.4f}")
    
    # Count how many are close to diagonal
    threshold = 0.05  # 5% difference threshold
    close_to_diagonal = np.sum(np.abs(differences) < threshold)
    print(f"Animals within {threshold:.2f} of diagonal: {close_to_diagonal}/{len(vanilla_lapse_probs)} ({100*close_to_diagonal/len(vanilla_lapse_probs):.1f}%)")

# %%
# Scatter plot: Lapse Probability Right Consistency between Vanilla+Lapse and Norm+Lapse
# X-axis: lapse_prob_right from vanilla+lapse
# Y-axis: lapse_prob_right from norm+lapse

print("\n" + "="*160)
print("Creating scatter plot: Lapse Probability Right Consistency (Vanilla+Lapse vs Norm+Lapse)")
print("="*160)

# Prepare data
vanilla_lapse_probs_right = []
norm_lapse_probs_right = []
consistency_right_labels = []

for row in rows:
    vanilla_lpr = row.get('vanilla_lapse_prob_right')
    norm_lpr = row.get('norm_lapse_prob_right')
    
    # Only include rows where we have both lapse_prob_right values
    if vanilla_lpr is not None and norm_lpr is not None:
        vanilla_lapse_probs_right.append(vanilla_lpr)
        norm_lapse_probs_right.append(norm_lpr)
        consistency_right_labels.append(f"{row['batch']}_{row['animal']}")

# Create scatter plot
fig, ax = plt.subplots(figsize=(10, 10))

# Scatter plot
ax.scatter(vanilla_lapse_probs_right, norm_lapse_probs_right, alpha=0.6, s=100, 
           c='purple', edgecolors='black', linewidth=0.5, label='Animals')

# Add diagonal line y=x (perfect consistency)
if len(vanilla_lapse_probs_right) > 0:
    min_val = min(min(vanilla_lapse_probs_right), min(norm_lapse_probs_right))
    max_val = max(max(vanilla_lapse_probs_right), max(norm_lapse_probs_right))
    ax.plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=2, label='Perfect consistency (y=x)')
    
    # Add horizontal line at y=0.5 (no directional bias)
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label='No right bias (0.5)')
    ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)

# Labels and title
ax.set_xlabel('Lapse Prob Right (Vanilla+Lapse Model)', fontsize=14, fontweight='bold')
ax.set_ylabel('Lapse Prob Right (Norm+Lapse Model)', fontsize=14, fontweight='bold')
ax.set_title('Lapse Probability Right Consistency Between Models', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)
# ax.legend(fontsize=12)
ax.set_aspect('equal', adjustable='box')

# Add correlation and statistics
if len(vanilla_lapse_probs_right) > 0:
    correlation = np.corrcoef(vanilla_lapse_probs_right, norm_lapse_probs_right)[0, 1]
    
    # Calculate mean absolute difference
    differences = np.array(vanilla_lapse_probs_right) - np.array(norm_lapse_probs_right)
    mean_abs_diff = np.mean(np.abs(differences))
    
    # Calculate mean values
    mean_vanilla = np.mean(vanilla_lapse_probs_right)
    mean_norm = np.mean(norm_lapse_probs_right)
    
    # stats_text = f'Correlation: {correlation:.3f}\nMean |Δ|: {mean_abs_diff:.4f}\nMean V: {mean_vanilla:.3f}\nMean N: {mean_norm:.3f}'
    stats_text = f'Correlation: {correlation:.3f}\nMean |Δ|: {mean_abs_diff:.4f}'

    ax.text(0.05, 0.95, stats_text, 
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
consistency_right_plot_path = os.path.join(base_dir, 'lapse_prob_right_consistency_vanilla_vs_norm_v2.png')
plt.savefig(consistency_right_plot_path, dpi=150, bbox_inches='tight')
plt.show()

print(f"\nLapse Prob Right consistency scatter plot saved to: {consistency_right_plot_path}")
print(f"Number of animals plotted: {len(vanilla_lapse_probs_right)}")
if len(vanilla_lapse_probs_right) > 0:
    print(f"Vanilla lapse_prob_right range: [{min(vanilla_lapse_probs_right):.4f}, {max(vanilla_lapse_probs_right):.4f}]")
    print(f"Norm lapse_prob_right range: [{min(norm_lapse_probs_right):.4f}, {max(norm_lapse_probs_right):.4f}]")
    print(f"Mean vanilla lapse_prob_right: {mean_vanilla:.4f}")
    print(f"Mean norm lapse_prob_right: {mean_norm:.4f}")
    print(f"Correlation coefficient: {correlation:.3f}")
    print(f"Mean absolute difference: {mean_abs_diff:.4f}")
    
    # Count how many are close to diagonal
    threshold = 0.05  # 5% difference threshold
    close_to_diagonal = np.sum(np.abs(differences) < threshold)
    print(f"Animals within {threshold:.2f} of diagonal: {close_to_diagonal}/{len(vanilla_lapse_probs_right)} ({100*close_to_diagonal/len(vanilla_lapse_probs_right):.1f}%)")
    
    # Count animals with right bias (> 0.5) in both models
    vanilla_right_bias = np.sum(np.array(vanilla_lapse_probs_right) > 0.5)
    norm_right_bias = np.sum(np.array(norm_lapse_probs_right) > 0.5)
    print(f"Animals with right bias (>0.5) - Vanilla: {vanilla_right_bias}/{len(vanilla_lapse_probs_right)}, Norm: {norm_right_bias}/{len(norm_lapse_probs_right)}")

# %%
