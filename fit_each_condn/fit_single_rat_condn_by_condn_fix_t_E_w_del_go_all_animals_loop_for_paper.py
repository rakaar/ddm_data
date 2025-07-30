# %%
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import pandas as pd
import random
from scipy.integrate import trapezoid as trapz
from pyvbmc import VBMC
import corner
from scipy.integrate import cumulative_trapezoid as cumtrapz
import pickle
from led_off_gamma_omega_pdf_utils import cum_pro_and_reactive_trunc_fn, up_or_down_RTs_fit_OPTIM_V_A_change_gamma_omega_with_w_fn
from led_off_gamma_omega_pdf_utils import cum_pro_and_reactive, up_or_down_RTs_fit_OPTIM_V_A_change_gamma_omega_fn,\
         rho_A_t_VEC_fn, up_or_down_RTs_fit_OPTIM_V_A_change_gamma_omega_P_A_C_A_wrt_stim_fn
from led_off_gamma_omega_pdf_utils import up_or_down_RTs_fit_OPTIM_V_A_change_gamma_omega_with_w_PA_CA_fn
from collections import defaultdict

# %%
# collect animals
# --- Get Batch-Animal Pairs ---
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
def get_params_from_animal_pkl_file(batch_name, animal_id):
    pkl_file = f'/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/results_{batch_name}_animal_{animal_id}.pkl'
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

# %%
# VBMC funcs
def compute_loglike_trial(row, gamma, omega):
        # data
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
        
        P_joint_rt_choice = up_or_down_RTs_fit_OPTIM_V_A_change_gamma_omega_with_w_fn(rt, V_A, theta_A, gamma, omega, t_stim, t_A_aff, t_E_aff, del_go, choice, w, K_max)
        

        
        P_joint_rt_choice_trunc = max(P_joint_rt_choice / (trunc_factor_p_joint + 1e-10), 1e-10)
        
        wt_log_like = np.log(P_joint_rt_choice_trunc)

        return wt_log_like




omega_bounds = [0.1, 15]
omega_plausible_bounds = [2, 12]

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
    

def vbmc_prior_fn(params):
    gamma, omega = params
    gamma_logpdf = trapezoidal_logpdf(gamma, gamma_bounds[0], gamma_plausible_bounds[0], gamma_plausible_bounds[1], gamma_bounds[1])
    omega_logpdf = trapezoidal_logpdf(omega, omega_bounds[0], omega_plausible_bounds[0], omega_plausible_bounds[1], omega_bounds[1])

    return gamma_logpdf + omega_logpdf

# %%
# all_ABLs_cond = [20, 40, 60]
all_ABLs_cond = [40]
all_ILDs_cond = [1, -1, 2, -2, 4, -4, 8, -8, 16, -16]
vbmc_fit_saving_path = '/home/rlab/raghavendra/ddm_data/fit_each_condn/each_animal_cond_fit_gama_omega_pkl_files'
K_max = 10

for batch_name, animal_id in batch_animal_pairs:

    print('##########################################')
    print(f'Batch: {batch_name}, Animal: {animal_id}')
    print('##########################################')

    MODEL_TYPE = 'vanilla'
    abort_params, vanilla_tied_params, rate_norm_l, is_norm = get_params_from_animal_pkl_file(batch_name, animal_id)
    MODEL_TYPE = 'norm'
    abort_params, norm_tied_params, rate_norm_l, is_norm = get_params_from_animal_pkl_file(batch_name, animal_id)
    
    # take w, t_E_aff, del_go avg from both vanilla and norm tied params
    w = (vanilla_tied_params['w'] + norm_tied_params['w']) / 2
    t_E_aff = (vanilla_tied_params['t_E_aff'] + norm_tied_params['t_E_aff']) / 2
    del_go = (vanilla_tied_params['del_go'] + norm_tied_params['del_go']) / 2
    
    print(f"Batch: {batch_name}, Animal: {animal_id}")
    print(f"w: {w}")
    print(f"t_E_aff: {t_E_aff}")
    print(f"del_go: {del_go}")
    print("\n")

    # abort params
    V_A = abort_params['V_A']
    theta_A = abort_params['theta_A']
    t_A_aff = abort_params['t_A_aff']

    # get the database from batch_csvs
    file_name = f'../fit_animal_by_animal/batch_csvs/batch_{batch_name}_valid_and_aborts.csv'
    df = pd.read_csv(file_name)
    df_animal = df[df['animal'] == int(animal_id)]
    df_animal_success = df_animal[df_animal['success'].isin([1, -1])]
    df_animal_success_rt_filter = df_animal_success[(df_animal_success['RTwrtStim'] <= 1) & (df_animal_success['RTwrtStim'] > 0)]

    for cond_ABL in all_ABLs_cond:
        for cond_ILD in all_ILDs_cond:
            pkl_file = os.path.join(vbmc_fit_saving_path, f'vbmc_cond_by_cond_{batch_name}_{animal_id}_{cond_ABL}_ILD_{cond_ILD}_FIX_t_E_w_del_go_same_as_parametric.pkl')
            if os.path.exists(pkl_file):
                print(f'{pkl_file} already exists, skipping')
                continue
            else:
                print(f'{pkl_file} NOT EXISTS, fitting')

            print('********************************')
            print(f'ABL: {cond_ABL}, ILD: {cond_ILD}')
            print('********************************')
            
            conditions = {'ABL': [cond_ABL], 'ILD': [cond_ILD]}
            df_animal_cond_filter = df_animal_success_rt_filter[
                (df_animal_success_rt_filter['ABL'].isin(conditions['ABL'])) & 
                (df_animal_success_rt_filter['ILD'].isin(conditions['ILD']))
            ]
            print(f'len(df_animal_cond_filter): {len(df_animal_cond_filter)}')

            if len(df_animal_cond_filter) == 0:
                continue
            def vbmc_loglike_fn(params):
                gamma, omega = params
                all_loglike = Parallel(n_jobs=30)(delayed(compute_loglike_trial)(row, gamma, omega) \
                                                for _, row in df_animal_cond_filter.iterrows())
                return np.sum(all_loglike)

            def vbmc_joint_fn(params):
                priors = vbmc_prior_fn(params)
                loglike = vbmc_loglike_fn(params)

                return priors + loglike

            
            print(f'++++++++++ ABL = {cond_ABL}, ILD = {cond_ILD} +++++++++++++++++++++')
            if cond_ILD > 0:
                gamma_bounds = [-1, 5]
                gamma_plausible_bounds = [0, 3]
            elif cond_ILD < 0:
                gamma_bounds = [-5, 1]
                gamma_plausible_bounds = [-3, 0]


            lb = np.array([gamma_bounds[0], omega_bounds[0]])
            ub = np.array([gamma_bounds[1], omega_bounds[1]])

            plb = np.array([gamma_plausible_bounds[0], omega_plausible_bounds[0]])
            pub = np.array([gamma_plausible_bounds[1], omega_plausible_bounds[1]])

            # Initialize with random values within plausible bounds
            np.random.seed(42)
            gamma_0 = np.random.uniform(gamma_plausible_bounds[0], gamma_plausible_bounds[1])
            omega_0 = np.random.uniform(omega_plausible_bounds[0], omega_plausible_bounds[1])

            x_0 = np.array([gamma_0, omega_0])

            # Run VBMC
            vbmc = VBMC(vbmc_joint_fn, x_0, lb, ub, plb, pub, options={'display': 'on'})
            vp, results = vbmc.optimize()

            # save vbmc 
            vbmc.save(os.path.join(vbmc_fit_saving_path, f'vbmc_cond_by_cond_{batch_name}_{animal_id}_{cond_ABL}_ILD_{cond_ILD}_FIX_t_E_w_del_go_same_as_parametric.pkl'), overwrite=True)

# %%
