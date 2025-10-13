# %%
"""
Animal-wise Normalized TIED Model Fitting Script
Loads abort parameters from pickle and fits normalized model.
Includes stimulus filtering for specific ABLs and ILDs.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed
import pickle
from pyvbmc import VBMC
import os
from time_vary_norm_utils import up_or_down_RTs_fit_fn, cum_pro_and_reactive_time_vary_fn
from vbmc_animal_wise_fit_utils import trapezoidal_logpdf
from animal_wise_config import T_trunc
from matplotlib.backends.backend_pdf import PdfPages

############3 Params #############
batch_name = 'LED34'
animal_to_fit = [63]
K_max = 10

# Create output directory
output_dir = '/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/led34_filter_files/norm'
os.makedirs(output_dir, exist_ok=True)


######### Normalized TIED ###############
def compute_loglike_norm_fn(row, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go, rate_norm_l):
    rt = row['TotalFixTime']
    t_stim = row['intended_fix']
    ILD = row['ILD']
    ABL = row['ABL']
    choice = row['choice']

    pdf = up_or_down_RTs_fit_fn(
        rt, choice, V_A, theta_A, t_A_aff,
        t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go,
        phi_params_obj, rate_norm_l, is_norm, is_time_vary, K_max
    )

    trunc_factor_p_joint = cum_pro_and_reactive_time_vary_fn(
        t_stim + 1, T_trunc, V_A, theta_A, t_A_aff,
        t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_E_aff,
        phi_params_obj, rate_norm_l, is_norm, is_time_vary, K_max
    ) - cum_pro_and_reactive_time_vary_fn(
        t_stim, T_trunc, V_A, theta_A, t_A_aff,
        t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_E_aff,
        phi_params_obj, rate_norm_l, is_norm, is_time_vary, K_max
    )

    pdf /= (trunc_factor_p_joint + 1e-20)
    pdf = max(pdf, 1e-50)
    if np.isnan(pdf):
        raise ValueError(f'nan pdf rt = {rt}, t_stim = {t_stim}')
    return np.log(pdf)

def vbmc_norm_tied_loglike_fn(params):
    rate_lambda, T_0, theta_E, w, t_E_aff, del_go, rate_norm_l = params
    Z_E = (w - 0.5) * 2 * theta_E
    all_loglike = Parallel(n_jobs=30)(
        delayed(compute_loglike_norm_fn)(row, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go, rate_norm_l)
        for _, row in df_valid_animal_less_than_1.iterrows()
    )
    return np.sum(all_loglike)

def vbmc_prior_norm_tied_fn(params):
    rate_lambda, T_0, theta_E, w, t_E_aff, del_go, rate_norm_l = params
    
    logpdfs = [
        trapezoidal_logpdf(rate_lambda, norm_tied_rate_lambda_bounds[0], norm_tied_rate_lambda_plausible_bounds[0], 
                          norm_tied_rate_lambda_plausible_bounds[1], norm_tied_rate_lambda_bounds[1]),
        trapezoidal_logpdf(T_0, norm_tied_T_0_bounds[0], norm_tied_T_0_plausible_bounds[0], 
                          norm_tied_T_0_plausible_bounds[1], norm_tied_T_0_bounds[1]),
        trapezoidal_logpdf(theta_E, norm_tied_theta_E_bounds[0], norm_tied_theta_E_plausible_bounds[0], 
                          norm_tied_theta_E_plausible_bounds[1], norm_tied_theta_E_bounds[1]),
        trapezoidal_logpdf(w, norm_tied_w_bounds[0], norm_tied_w_plausible_bounds[0], 
                          norm_tied_w_plausible_bounds[1], norm_tied_w_bounds[1]),
        trapezoidal_logpdf(t_E_aff, norm_tied_t_E_aff_bounds[0], norm_tied_t_E_aff_plausible_bounds[0], 
                          norm_tied_t_E_aff_plausible_bounds[1], norm_tied_t_E_aff_bounds[1]),
        trapezoidal_logpdf(del_go, norm_tied_del_go_bounds[0], norm_tied_del_go_plausible_bounds[0], 
                          norm_tied_del_go_plausible_bounds[1], norm_tied_del_go_bounds[1]),
        trapezoidal_logpdf(rate_norm_l, norm_tied_rate_norm_bounds[0], norm_tied_rate_norm_plausible_bounds[0], 
                          norm_tied_rate_norm_plausible_bounds[1], norm_tied_rate_norm_bounds[1])
    ]
    return sum(logpdfs)

def vbmc_norm_tied_joint_fn(params):
    return vbmc_prior_norm_tied_fn(params) + vbmc_norm_tied_loglike_fn(params)

# Normalized bounds
norm_tied_rate_lambda_bounds = [0.5, 5]
norm_tied_T_0_bounds = [50e-3, 800e-3]
norm_tied_theta_E_bounds = [1, 15]
norm_tied_w_bounds = [0.3, 0.7]
norm_tied_t_E_aff_bounds = [0.01, 0.2]
norm_tied_del_go_bounds = [0, 0.2]
norm_tied_rate_norm_bounds = [0, 2]

norm_tied_rate_lambda_plausible_bounds = [1, 3]
norm_tied_T_0_plausible_bounds = [90e-3, 400e-3]
norm_tied_theta_E_plausible_bounds = [1.5, 10]
norm_tied_w_plausible_bounds = [0.4, 0.6]
norm_tied_t_E_aff_plausible_bounds = [0.03, 0.09]
norm_tied_del_go_plausible_bounds = [0.05, 0.15]
norm_tied_rate_norm_plausible_bounds = [0.8, 0.99]

norm_tied_lb = np.array([norm_tied_rate_lambda_bounds[0], norm_tied_T_0_bounds[0], norm_tied_theta_E_bounds[0], 
                         norm_tied_w_bounds[0], norm_tied_t_E_aff_bounds[0], norm_tied_del_go_bounds[0], 
                         norm_tied_rate_norm_bounds[0]])
norm_tied_ub = np.array([norm_tied_rate_lambda_bounds[1], norm_tied_T_0_bounds[1], norm_tied_theta_E_bounds[1], 
                         norm_tied_w_bounds[1], norm_tied_t_E_aff_bounds[1], norm_tied_del_go_bounds[1], 
                         norm_tied_rate_norm_bounds[1]])
norm_tied_plb = np.array([norm_tied_rate_lambda_plausible_bounds[0], norm_tied_T_0_plausible_bounds[0], 
                          norm_tied_theta_E_plausible_bounds[0], norm_tied_w_plausible_bounds[0], 
                          norm_tied_t_E_aff_plausible_bounds[0], norm_tied_del_go_plausible_bounds[0], 
                          norm_tied_rate_norm_plausible_bounds[0]])
norm_tied_pub = np.array([norm_tied_rate_lambda_plausible_bounds[1], norm_tied_T_0_plausible_bounds[1], 
                          norm_tied_theta_E_plausible_bounds[1], norm_tied_w_plausible_bounds[1], 
                          norm_tied_t_E_aff_plausible_bounds[1], norm_tied_del_go_plausible_bounds[1], 
                          norm_tied_rate_norm_plausible_bounds[1]])

# %%
### Read csv from batch_csvs/ folder ###
csv_filename = f'batch_csvs/batch_{batch_name}_valid_and_aborts.csv'
exp_df = pd.read_csv(csv_filename)

# Data is already processed and batch-filtered in the CSV
### DF - valid and aborts ###
df_valid_and_aborts = exp_df[
    (exp_df['success'].isin([1,-1])) |
    (exp_df['abort_event'] == 3)
].copy()

df_aborts = df_valid_and_aborts[df_valid_and_aborts['abort_event'] == 3]
animal_ids = df_valid_and_aborts['animal'].unique() 

print('####################################')
print(f'Aborts Truncation Time: {T_trunc}')
print('####################################')

# %%
# Main fitting loop
for animal in animal_to_fit:
    if animal not in animal_ids:
        print(f"Animal {animal} not found in batch {batch_name}. Available animals: {animal_ids}")
        continue
    
    df_all_trials_animal = df_valid_and_aborts[df_valid_and_aborts['animal'] == animal]
    df_aborts_animal = df_aborts[df_aborts['animal'] == animal]

    print(f'Batch: {batch_name}, sample animal: {animal}')
    pdf_filename = os.path.join(output_dir, f'results_{batch_name}_animal_{animal}_NORM_ABL_ILD_filtered.pdf')
    pdf = PdfPages(pdf_filename)

    # Page 1: Title
    fig_text = plt.figure(figsize=(8.5, 11))
    fig_text.clf()
    fig_text.text(0.1, 0.9, f"Normalized Model Analysis Report", fontsize=20, weight='bold')
    fig_text.text(0.1, 0.8, f"Batch Name: {batch_name}", fontsize=14)
    fig_text.text(0.1, 0.75, f"Animal ID: {animal}", fontsize=14)
    fig_text.text(0.1, 0.68, f"Data Filtering:", fontsize=14, weight='bold')
    fig_text.text(0.1, 0.63, f"  - ABLs: 20, 40, 60", fontsize=12)
    fig_text.text(0.1, 0.59, f"  - ILDs: ±1, ±2, ±4, ±8, ±16", fontsize=12)
    fig_text.text(0.1, 0.55, f"  - Abort params loaded from pickle", fontsize=12)
    fig_text.gca().axis("off")
    pdf.savefig(fig_text, bbox_inches='tight')
    plt.close(fig_text)

    ####################################################
    ########### Load Abort Parameters ##################
    ####################################################
    print("\n### Loading Abort Parameters from Pickle ###")
    
    pkl_file = f'results_{batch_name}_animal_{animal}.pkl'
    try:
        with open(pkl_file, 'rb') as f:
            fit_results_data = pickle.load(f)
        
        vbmc_aborts_param_keys_map = {
            'V_A_samples': 'V_A',
            'theta_A_samples': 'theta_A',
            't_A_aff_samp': 't_A_aff'
        }
        abort_keyname = "vbmc_aborts_results"
        
        abort_samples = fit_results_data[abort_keyname]
        abort_params = {}
        for param_samples_name, param_label in vbmc_aborts_param_keys_map.items():
            abort_params[param_label] = np.mean(abort_samples[param_samples_name])
        
        V_A = abort_params['V_A']
        theta_A = abort_params['theta_A']
        t_A_aff = abort_params['t_A_aff']
        
        print(f"Loaded abort parameters:")
        print(f"  V_A = {V_A:.4f}")
        print(f"  theta_A = {theta_A:.4f}")
        print(f"  t_A_aff = {t_A_aff:.4f}")
        
        # Store the abort samples for later saving
        vbmc_aborts_results = abort_samples
        
        # Add abort params page to PDF
        fig_text = plt.figure(figsize=(8.5, 11))
        fig_text.clf()
        fig_text.text(0.1, 0.9, f"Abort Parameters (Loaded from {pkl_file})", fontsize=16, weight='bold')
        fig_text.text(0.1, 0.8, f"V_A: {V_A:.4f}", fontsize=12)
        fig_text.text(0.1, 0.75, f"theta_A: {theta_A:.4f}", fontsize=12)
        fig_text.text(0.1, 0.7, f"t_A_aff: {t_A_aff:.4f}", fontsize=12)
        fig_text.gca().axis("off")
        pdf.savefig(fig_text, bbox_inches='tight')
        plt.close(fig_text)
        
    except FileNotFoundError:
        print(f"ERROR: Pickle file {pkl_file} not found. Please run abort fitting first.")
        pdf.close()
        continue
    except KeyError as e:
        print(f"ERROR: Missing key {e} in pickle file {pkl_file}")
        pdf.close()
        continue

    ####################################################
    ########### Filter Data for Specific ABLs/ILDs #####
    ####################################################
    print("\n### Filtering data for specific ABLs and ILDs ###")
    allowed_abls = [20, 40, 60]
    allowed_ilds = [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]
    
    df_all_trials_animal = df_all_trials_animal[
        (df_all_trials_animal['ABL'].isin(allowed_abls)) &
        (df_all_trials_animal['ILD'].isin(allowed_ilds))
    ]
    
    print(f"After filtering: {len(df_all_trials_animal)} trials")
    
    # Extract valid trials
    df_valid_animal = df_all_trials_animal[df_all_trials_animal['success'].isin([1,-1])]
    df_valid_animal_less_than_1 = df_valid_animal[df_valid_animal['RTwrtStim'] < 1]
    
    print(f"Valid trials with RT < 1s: {len(df_valid_animal_less_than_1)}")

    ####################################################
    ########### Normalized Model Fitting ###############
    ####################################################
    print("\n### Fitting Normalized Model ###")
    
    is_norm = True
    is_time_vary = False
    phi_params_obj = np.nan

    rate_lambda_0 = 2.3
    T_0_0 = 100 * 1e-3
    theta_E_0 = 3
    w_0 = 0.51
    t_E_aff_0 = 0.071
    del_go_0 = 0.13  # Changed from 0.19 to fit within plausible bounds [0.05, 0.15]
    rate_norm_l_0 = 0.95

    x_0 = np.array([
        rate_lambda_0,
        T_0_0,
        theta_E_0,
        w_0,
        t_E_aff_0,
        del_go_0,
        rate_norm_l_0
    ])

    # Run VBMC
    vbmc = VBMC(vbmc_norm_tied_joint_fn, x_0, norm_tied_lb, norm_tied_ub, norm_tied_plb, norm_tied_pub, 
                options={'display': 'on', 'max_fun_evals': 200 * (2 + 7)})
    vp, results = vbmc.optimize()
    vbmc.save(os.path.join(output_dir, f'vbmc_PKL_file_norm_tied_results_batch_{batch_name}_animal_{animal}_FILTERED.pkl'))

    # Sample from posterior
    vp_samples = vp.sample(int(1e5))[0]

    # Extract posterior means
    rate_lambda = vp_samples[:, 0].mean()
    T_0 = vp_samples[:, 1].mean()
    theta_E = vp_samples[:, 2].mean()
    w = vp_samples[:, 3].mean()
    Z_E = (w - 0.5) * 2 * theta_E
    t_E_aff = vp_samples[:, 4].mean()
    del_go = vp_samples[:, 5].mean()
    rate_norm_l = vp_samples[:, 6].mean()

    print("\nPosterior Means:")
    print(f"rate_lambda  = {rate_lambda:.5f}")
    print(f"T_0 (ms)     = {1e3*T_0:.5f}")
    print(f"theta_E      = {theta_E:.5f}")
    print(f"Z_E          = {Z_E:.5f}")
    print(f"t_E_aff (ms) = {1e3*t_E_aff:.5f}")
    print(f"del_go       = {del_go:.5f}")
    print(f"rate_norm_l  = {rate_norm_l:.5f}")

    norm_tied_loglike = vbmc_norm_tied_loglike_fn([rate_lambda, T_0, theta_E, w, t_E_aff, del_go, rate_norm_l])

    vbmc_norm_tied_results = {
        'rate_lambda_samples': vp_samples[:, 0],
        'T_0_samples': vp_samples[:, 1],
        'theta_E_samples': vp_samples[:, 2],
        'w_samples': vp_samples[:, 3],
        't_E_aff_samples': vp_samples[:, 4],
        'del_go_samples': vp_samples[:, 5],
        'rate_norm_l_samples': vp_samples[:, 6],
        'message': results['message'],
        'elbo': results['elbo'],
        'elbo_sd': results['elbo_sd'],
        'loglike': norm_tied_loglike
    }

    # Close PDF (only title and abort params pages)
    pdf.close()
    print(f"PDF saved: {pdf_filename}")

    ####################################################
    ########### Save Results to Pickle #################
    ####################################################
    pkl_output = os.path.join(output_dir, f'results_{batch_name}_animal_{animal}_NORM_filtered.pkl')
    
    # Combine abort results (from loaded file) with new norm results
    save_dict = {
        'vbmc_aborts_results': vbmc_aborts_results,
        'vbmc_norm_tied_results': vbmc_norm_tied_results
    }
    
    with open(pkl_output, 'wb') as f:
        pickle.dump(save_dict, f)
    
    print(f"Results saved to: {pkl_output}")
    print(f"Completed fitting for animal {animal}\n")
    print("=" * 60)

print("\n### All animals completed ###")
