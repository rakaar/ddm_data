# %%
"""
Fit condition-by-condition with 5 parameters (gamma, omega, t_E_aff, w, del_go)
for ALL animals across all batches.

This is the 5-param version where t_E_aff, w, del_go are FIT per condition (not fixed).
Unlike the 2-param version (_FIX_t_E_w_del_go_same_as_parametric), this fits all 5 params.

Saves pkl files to: each_animal_cond_fit_5_params_pkl_files/
"""
import os
import numpy as np
from joblib import Parallel, delayed
import pandas as pd
from pyvbmc import VBMC
import pickle
from led_off_gamma_omega_pdf_utils import cum_pro_and_reactive_trunc_fn, up_or_down_RTs_fit_OPTIM_V_A_change_gamma_omega_with_w_fn
from collections import defaultdict

# %%
# =============================================================================
# Configuration
# =============================================================================
DESIRED_BATCHES = ['SD', 'LED34', 'LED6', 'LED8', 'LED7', 'LED34_even']
batch_dir = '/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/batch_csvs'

# Output folder for 5-param fits
OUTPUT_FOLDER = '/home/rlab/raghavendra/ddm_data/fit_each_condn/each_animal_cond_fit_5_params_pkl_files'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

all_ABLs_cond = [20, 40, 60]
all_ILDs_cond = [1, -1, 2, -2, 4, -4, 8, -8, 16, -16]
K_max = 10
N_JOBS = 30  # Parallel jobs for likelihood computation

# %%
# =============================================================================
# Get batch-animal pairs
# =============================================================================
batch_files = [f'batch_{batch_name}_valid_and_aborts.csv' for batch_name in DESIRED_BATCHES]

merged_data = pd.concat([
    pd.read_csv(os.path.join(batch_dir, fname)) for fname in batch_files if os.path.exists(os.path.join(batch_dir, fname))
], ignore_index=True)

merged_valid = merged_data[merged_data['success'].isin([1, -1])].copy()
batch_animal_pairs = sorted(list(map(tuple, merged_valid[['batch_name', 'animal']].drop_duplicates().values)))

print(f"Found {len(batch_animal_pairs)} batch-animal pairs from {len(set(p[0] for p in batch_animal_pairs))} batches:")

if batch_animal_pairs:
    batch_to_animals = defaultdict(list)
    for batch, animal in batch_animal_pairs:
        animal_str = str(animal)
        if animal_str not in batch_to_animals[batch]:
            batch_to_animals[batch].append(animal_str)
    
    max_batch_len = max(len(b) for b in batch_to_animals.keys()) if batch_to_animals else 0
    animal_strings = {b: ', '.join(sorted(a)) for b, a in batch_to_animals.items()}
    
    print(f"{'Batch':<{max_batch_len}}  {'Animals'}")
    print(f"{'=' * max_batch_len}  {'=' * 50}")
    for batch in sorted(animal_strings.keys()):
        print(f"{batch:<{max_batch_len}}  {animal_strings[batch]}")

# %%
# =============================================================================
# Helper functions
# =============================================================================

def get_abort_params_from_animal_pkl_file(batch_name, animal_id):
    """Load abort parameters (V_A, theta_A, t_A_aff) from animal pkl file."""
    pkl_file = f'/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/results_{batch_name}_animal_{animal_id}.pkl'
    if not os.path.exists(pkl_file):
        return None
    with open(pkl_file, 'rb') as f:
        fit_results_data = pickle.load(f)
    
    abort_keyname = "vbmc_aborts_results"
    if abort_keyname not in fit_results_data:
        return None
    
    abort_samples = fit_results_data[abort_keyname]
    return {
        'V_A': np.mean(abort_samples['V_A_samples']),
        'theta_A': np.mean(abort_samples['theta_A_samples']),
        't_A_aff': np.mean(abort_samples['t_A_aff_samp'])
    }

def trapezoidal_logpdf(x, a, b, c, d):
    """Trapezoidal prior log-pdf."""
    if x < a or x > d:
        return -np.inf
    area = ((b - a) + (d - c)) / 2 + (c - b)
    h_max = 1.0 / area
    
    if a <= x <= b:
        pdf_value = ((x - a) / (b - a)) * h_max
    elif b < x < c:
        pdf_value = h_max
    elif c <= x <= d:
        pdf_value = ((d - x) / (d - c)) * h_max
    else:
        pdf_value = 0.0
    
    if pdf_value <= 0.0:
        return -np.inf
    else:
        return np.log(pdf_value)

# %%
# =============================================================================
# Parameter bounds (5 params: gamma, omega, t_E_aff, w, del_go)
# =============================================================================
omega_bounds = [0.1, 15]
omega_plausible_bounds = [2, 12]

t_E_aff_bounds = [0, 1]
t_E_aff_plausible_bounds = [0.01, 0.2]

w_bounds = [0.1, 0.9]
w_plausible_bounds = [0.3, 0.7]

del_go_bounds = [0.001, 0.2]
del_go_plausible_bounds = [0.11, 0.15]

# %%
# =============================================================================
# Main fitting loop
# =============================================================================

for batch_name, animal_id in batch_animal_pairs:
    print('\n' + '='*60)
    print(f'Batch: {batch_name}, Animal: {animal_id}')
    print('='*60)
    
    # Get abort params
    abort_params = get_abort_params_from_animal_pkl_file(batch_name, animal_id)
    if abort_params is None:
        print(f"  WARNING: No abort params found for {batch_name}/{animal_id}. Skipping.")
        continue
    
    V_A = abort_params['V_A']
    theta_A = abort_params['theta_A']
    t_A_aff = abort_params['t_A_aff']
    
    # Load data
    file_name = os.path.join(batch_dir, f'batch_{batch_name}_valid_and_aborts.csv')
    df = pd.read_csv(file_name)
    df_animal = df[df['animal'] == int(animal_id)]
    df_animal_success = df_animal[df_animal['success'].isin([1, -1])]
    df_animal_success_rt_filter = df_animal_success[
        (df_animal_success['RTwrtStim'] <= 1) & (df_animal_success['RTwrtStim'] > 0)
    ]
    
    for cond_ABL in all_ABLs_cond:
        for cond_ILD in all_ILDs_cond:
            # Check if already exists
            pkl_file = os.path.join(OUTPUT_FOLDER, 
                f'vbmc_cond_by_cond_{batch_name}_{animal_id}_{cond_ABL}_ILD_{cond_ILD}_5_params.pkl')
            
            if os.path.exists(pkl_file):
                print(f'  [{cond_ABL}, {cond_ILD:+3d}] Already exists, skipping')
                continue
            
            # Filter data for this condition
            df_cond = df_animal_success_rt_filter[
                (df_animal_success_rt_filter['ABL'] == cond_ABL) & 
                (df_animal_success_rt_filter['ILD'] == cond_ILD)
            ]
            
            if len(df_cond) < 10:
                print(f'  [{cond_ABL}, {cond_ILD:+3d}] Only {len(df_cond)} trials, skipping')
                continue
            
            print(f'  [{cond_ABL}, {cond_ILD:+3d}] Fitting with {len(df_cond)} trials...')
            
            # Define likelihood function for this condition
            def compute_loglike_trial(row, gamma, omega, t_E_aff, w, del_go):
                c_A_trunc_time = 0.3
                if 'timed_fix' in row:
                    rt = row['timed_fix']
                else:
                    rt = row['TotalFixTime']
                t_stim = row['intended_fix']
                response_poke = row['response_poke']
                
                trunc_factor_p_joint = cum_pro_and_reactive_trunc_fn(
                    t_stim + 1, c_A_trunc_time,
                    V_A, theta_A, t_A_aff,
                    t_stim, t_E_aff, gamma, omega, w, K_max) - \
                    cum_pro_and_reactive_trunc_fn(
                    t_stim, c_A_trunc_time,
                    V_A, theta_A, t_A_aff,
                    t_stim, t_E_aff, gamma, omega, w, K_max)
                
                choice = 2*response_poke - 5
                P_joint_rt_choice = up_or_down_RTs_fit_OPTIM_V_A_change_gamma_omega_with_w_fn(
                    rt, V_A, theta_A, gamma, omega, t_stim, t_A_aff, t_E_aff, del_go, choice, w, K_max)
                
                P_joint_rt_choice_trunc = max(P_joint_rt_choice / (trunc_factor_p_joint + 1e-10), 1e-10)
                return np.log(P_joint_rt_choice_trunc)
            
            # Set gamma bounds based on ILD sign
            if cond_ILD > 0:
                gamma_bounds = [-1, 5]
                gamma_plausible_bounds = [0, 3]
            else:
                gamma_bounds = [-5, 1]
                gamma_plausible_bounds = [-3, 0]
            
            def vbmc_prior_fn(params):
                gamma, omega, t_E_aff, w, del_go = params
                gamma_logpdf = trapezoidal_logpdf(gamma, gamma_bounds[0], gamma_plausible_bounds[0], 
                                                   gamma_plausible_bounds[1], gamma_bounds[1])
                omega_logpdf = trapezoidal_logpdf(omega, omega_bounds[0], omega_plausible_bounds[0], 
                                                   omega_plausible_bounds[1], omega_bounds[1])
                t_E_aff_logpdf = trapezoidal_logpdf(t_E_aff, t_E_aff_bounds[0], t_E_aff_plausible_bounds[0], 
                                                    t_E_aff_plausible_bounds[1], t_E_aff_bounds[1])
                w_logpdf = trapezoidal_logpdf(w, w_bounds[0], w_plausible_bounds[0], 
                                               w_plausible_bounds[1], w_bounds[1])
                del_go_logpdf = trapezoidal_logpdf(del_go, del_go_bounds[0], del_go_plausible_bounds[0], 
                                                    del_go_plausible_bounds[1], del_go_bounds[1])
                return gamma_logpdf + omega_logpdf + t_E_aff_logpdf + w_logpdf + del_go_logpdf
            
            def vbmc_loglike_fn(params):
                gamma, omega, t_E_aff, w, del_go = params
                all_loglike = Parallel(n_jobs=N_JOBS)(
                    delayed(compute_loglike_trial)(row, gamma, omega, t_E_aff, w, del_go) 
                    for _, row in df_cond.iterrows()
                )
                return np.sum(all_loglike)
            
            def vbmc_joint_fn(params):
                return vbmc_prior_fn(params) + vbmc_loglike_fn(params)
            
            # Bounds arrays
            lb = np.array([gamma_bounds[0], omega_bounds[0], t_E_aff_bounds[0], w_bounds[0], del_go_bounds[0]])
            ub = np.array([gamma_bounds[1], omega_bounds[1], t_E_aff_bounds[1], w_bounds[1], del_go_bounds[1]])
            plb = np.array([gamma_plausible_bounds[0], omega_plausible_bounds[0], t_E_aff_plausible_bounds[0], 
                           w_plausible_bounds[0], del_go_plausible_bounds[0]])
            pub = np.array([gamma_plausible_bounds[1], omega_plausible_bounds[1], t_E_aff_plausible_bounds[1], 
                           w_plausible_bounds[1], del_go_plausible_bounds[1]])
            
            # Initialize
            np.random.seed(42)
            x_0 = np.array([
                np.random.uniform(gamma_plausible_bounds[0], gamma_plausible_bounds[1]),
                np.random.uniform(omega_plausible_bounds[0], omega_plausible_bounds[1]),
                np.random.uniform(t_E_aff_plausible_bounds[0], t_E_aff_plausible_bounds[1]),
                np.random.uniform(w_plausible_bounds[0], w_plausible_bounds[1]),
                np.random.uniform(del_go_plausible_bounds[0], del_go_plausible_bounds[1])
            ])
            
            # Run VBMC
            try:
                vbmc = VBMC(vbmc_joint_fn, x_0, lb, ub, plb, pub, options={'display': 'iter'})
                vp, results = vbmc.optimize()
                
                # Save
                vbmc.save(pkl_file, overwrite=True)
                print(f'    -> Saved: {os.path.basename(pkl_file)}')
            except Exception as e:
                print(f'    -> ERROR: {e}')

print('\n' + '='*60)
print('DONE: All animals processed!')
print('='*60)
# %%
