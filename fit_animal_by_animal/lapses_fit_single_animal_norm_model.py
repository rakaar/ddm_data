# VBMC fit on exp data single animal - lapses + norm model
# Now fits rate_norm_l and lapse_prob_right as additional parameters
# %%
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
from pyvbmc import VBMC
from matplotlib.backends.backend_pdf import PdfPages
import pickle
import sys
import argparse
import os
sys.path.append('../lapses')
from lapses_utils import simulate_psiam_tied_rate_norm
from vbmc_animal_wise_fit_utils import trapezoidal_logpdf
from time_vary_norm_utils import up_or_down_RTs_fit_fn, cum_pro_and_reactive_time_vary_fn
# %%
# CLI args
parser = argparse.ArgumentParser(description='Fit norm+lapse model for a single animal')
parser.add_argument('--batch', required=True, help='Batch name, e.g., LED8')
parser.add_argument('--animal', required=True, type=int, help='Animal ID (int)')
parser.add_argument('--init-type', required=True, choices=['vanilla', 'norm'], help='Initialization type: vanilla or norm')
parser.add_argument('--output-dir', default='oct_6_7_large_bounds_diff_init_lapse_fit', help='Directory to save results')
args = parser.parse_args()

batch_name = args.batch
animal_ids = [args.animal]
output_dir = args.output_dir
init_type = args.init_type


os.makedirs(output_dir, exist_ok=True)

if batch_name == 'LED34_even':
    T_trunc = 0.15
else:
    T_trunc = 0.3

phi_params_obj = np.nan
is_norm = True
is_time_vary = False
K_max = 10

DO_RIGHT_TRUNCATE = True
if DO_RIGHT_TRUNCATE:
    print(f'Right truncation at 1s')
else:
    print(f'No right truncation')

# load animal data directly from preprocessed batch CSV
csv_filename = f'batch_csvs/batch_{batch_name}_valid_and_aborts.csv'
exp_df = pd.read_csv(csv_filename)

# Data is already processed and batch-filtered in the CSV
### DF - valid and aborts ###
df_valid_and_aborts = exp_df[
    (exp_df['success'].isin([1,-1])) |
    (exp_df['abort_event'] == 3)
].copy()

df_aborts = df_valid_and_aborts[df_valid_and_aborts['abort_event'] == 3]

# animal_ids = df_valid_and_aborts['animal'].unique()
# animal = animal_ids[-1]
# for animal_idx in [-1]:

print('####################################')
print(f'Aborts Truncation Time: {T_trunc}')
print('####################################')
# %%

# load proactive params

# %%
# VBMC helper funcs
def compute_loglike_norm(row, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go, rate_norm_l, lapse_prob, lapse_prob_right):
    
    rt = row['TotalFixTime']
    t_stim = row['intended_fix']
    
    
    ILD = row['ILD']
    ABL = row['ABL']
    choice = row['choice']
    lapse_rt_window = max_rt

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
    if np.isnan(included_lapse_pdf):
        print(f'row["abort_event"] = {row["abort_event"]}')
        print(f'row["RTwrtStim"] = {row["RTwrtStim"]}')
        raise ValueError(f'nan pdf rt = {rt}, t_stim = {t_stim}')

    
    return np.log(included_lapse_pdf)


def vbmc_norm_tied_loglike_fn(params):
    rate_lambda, T_0, theta_E, w, t_E_aff, del_go, rate_norm_l, lapse_prob, lapse_prob_right = params
    Z_E = (w - 0.5) * 2 * theta_E
    all_loglike = Parallel(n_jobs=30)(delayed(compute_loglike_norm)(row, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go, rate_norm_l, lapse_prob, lapse_prob_right)\
                                       for _, row in df_valid_animal.iterrows() )
    return np.sum(all_loglike)


def vbmc_norm_tied_prior_fn(params):
    rate_lambda, T_0, theta_E, w, t_E_aff, del_go, rate_norm_l, lapse_prob, lapse_prob_right = params

    rate_lambda_logpdf = trapezoidal_logpdf(
        rate_lambda,
        norm_rate_lambda_bounds[0],
        norm_rate_lambda_plausible_bounds[0],
        norm_rate_lambda_plausible_bounds[1],
        norm_rate_lambda_bounds[1]
    )
    
    T_0_logpdf = trapezoidal_logpdf(
        T_0,
        norm_T_0_bounds[0],
        norm_T_0_plausible_bounds[0],
        norm_T_0_plausible_bounds[1],
        norm_T_0_bounds[1]
    )
    
    theta_E_logpdf = trapezoidal_logpdf(
        theta_E,
        norm_theta_E_bounds[0],
        norm_theta_E_plausible_bounds[0],
        norm_theta_E_plausible_bounds[1],
        norm_theta_E_bounds[1]
    )
    
    w_logpdf = trapezoidal_logpdf(
        w,
        norm_w_bounds[0],
        norm_w_plausible_bounds[0],
        norm_w_plausible_bounds[1],
        norm_w_bounds[1]
    )
    
    t_E_aff_logpdf = trapezoidal_logpdf(
        t_E_aff,
        norm_t_E_aff_bounds[0],
        norm_t_E_aff_plausible_bounds[0],
        norm_t_E_aff_plausible_bounds[1],
        norm_t_E_aff_bounds[1]
    )
    
    del_go_logpdf = trapezoidal_logpdf(
        del_go,
        norm_del_go_bounds[0],
        norm_del_go_plausible_bounds[0],
        norm_del_go_plausible_bounds[1],
        norm_del_go_bounds[1]
    )
    
    rate_norm_l_logpdf = trapezoidal_logpdf(
        rate_norm_l,
        norm_rate_norm_bounds[0],
        norm_rate_norm_plausible_bounds[0],
        norm_rate_norm_plausible_bounds[1],
        norm_rate_norm_bounds[1]
    )
    
    lapse_prob_logpdf = trapezoidal_logpdf(
        lapse_prob,
        norm_lapse_prob_bounds[0],
        norm_lapse_prob_plausible_bounds[0],
        norm_lapse_prob_plausible_bounds[1],
        norm_lapse_prob_bounds[1]
    )
    
    lapse_prob_right_logpdf = trapezoidal_logpdf(
        lapse_prob_right,
        norm_lapse_prob_right_bounds[0],
        norm_lapse_prob_right_plausible_bounds[0],
        norm_lapse_prob_right_plausible_bounds[1],
        norm_lapse_prob_right_bounds[1]
    )
    
    return (
        rate_lambda_logpdf +
        T_0_logpdf +
        theta_E_logpdf +
        w_logpdf +
        t_E_aff_logpdf +
        del_go_logpdf +
        rate_norm_l_logpdf +
        lapse_prob_logpdf +
        lapse_prob_right_logpdf
    )

def vbmc_norm_tied_joint_fn(params):
    priors = vbmc_norm_tied_prior_fn(params)
    loglike = vbmc_norm_tied_loglike_fn(params)

    return priors + loglike


# Bounds for normalized model (from animal_wise_fit_3_models_script_refactor.py)
# large bounds to accomodate vanilla params too
# Updated bounds to accommodate both vanilla and norm model parameters
norm_rate_lambda_bounds = [0.01, 5]  # covers vanilla [0.01, 1] and norm [0.01, 5]
norm_T_0_bounds = [0.1e-3, 800e-3]  # covers vanilla [0.1e-3, 2.2e-3] and norm [50e-3, 800e-3]
norm_theta_E_bounds = [1, 65]  # covers vanilla [5, 65] and norm [1, 15]
norm_w_bounds = [0.3, 0.7]  # same for both
norm_t_E_aff_bounds = [0.01, 0.2]  # same for both
norm_del_go_bounds = [0, 0.2]  # same for both
norm_rate_norm_bounds = [0, 2]  # norm-specific parameter
norm_lapse_prob_bounds = [1e-4, 0.2]  # same for both
norm_lapse_prob_right_bounds = [0.001, 0.999]  # same for both

norm_rate_lambda_plausible_bounds = [0.1, 3]  # covers vanilla [0.1, 0.3] and norm [1, 3]
norm_T_0_plausible_bounds = [0.5e-3, 400e-3]  # covers vanilla [0.5e-3, 1.5e-3] and norm [90e-3, 400e-3]
norm_theta_E_plausible_bounds = [1.5, 55]  # covers vanilla [15, 55] and norm [1.5, 10]
norm_w_plausible_bounds = [0.4, 0.6]  # same for both
norm_t_E_aff_plausible_bounds = [0.03, 0.09]  # same for both
norm_del_go_plausible_bounds = [0.05, 0.15]  # same for both
norm_rate_norm_plausible_bounds = [0.8, 0.99]  # norm-specific parameter
norm_lapse_prob_plausible_bounds = [1e-3, 0.1]  # same for both
norm_lapse_prob_right_plausible_bounds = [0.4, 0.6]  # same for both


norm_tied_lb = np.array([
    norm_rate_lambda_bounds[0],
    norm_T_0_bounds[0],
    norm_theta_E_bounds[0],
    norm_w_bounds[0],
    norm_t_E_aff_bounds[0],
    norm_del_go_bounds[0],
    norm_rate_norm_bounds[0],
    norm_lapse_prob_bounds[0],
    norm_lapse_prob_right_bounds[0]
])

norm_tied_ub = np.array([
    norm_rate_lambda_bounds[1],
    norm_T_0_bounds[1],
    norm_theta_E_bounds[1],
    norm_w_bounds[1],
    norm_t_E_aff_bounds[1],
    norm_del_go_bounds[1],
    norm_rate_norm_bounds[1],
    norm_lapse_prob_bounds[1],
    norm_lapse_prob_right_bounds[1]
])

norm_plb = np.array([
    norm_rate_lambda_plausible_bounds[0],
    norm_T_0_plausible_bounds[0],
    norm_theta_E_plausible_bounds[0],
    norm_w_plausible_bounds[0],
    norm_t_E_aff_plausible_bounds[0],
    norm_del_go_plausible_bounds[0],
    norm_rate_norm_plausible_bounds[0],
    norm_lapse_prob_plausible_bounds[0],
    norm_lapse_prob_right_plausible_bounds[0]
])

norm_pub = np.array([
    norm_rate_lambda_plausible_bounds[1],
    norm_T_0_plausible_bounds[1],
    norm_theta_E_plausible_bounds[1],
    norm_w_plausible_bounds[1],
    norm_t_E_aff_plausible_bounds[1],
    norm_del_go_plausible_bounds[1],
    norm_rate_norm_plausible_bounds[1],
    norm_lapse_prob_plausible_bounds[1],
    norm_lapse_prob_right_plausible_bounds[1]
])
# %%
print(f'len of animal_ids = {len(animal_ids)}')

# %%
for animal_idx in [0]:
    animal = animal_ids[animal_idx]

    df_all_trials_animal = df_valid_and_aborts[df_valid_and_aborts['animal'] == animal]
    df_aborts_animal = df_aborts[df_aborts['animal'] == animal]

    df_valid_animal = df_all_trials_animal[df_all_trials_animal['success'].isin([1,-1])]
    # no right Truncation
    if DO_RIGHT_TRUNCATE:
        df_valid_animal = df_valid_animal[df_valid_animal['RTwrtStim'] < 1]
        max_rt = 1
    else:
        max_rt = df_valid_animal['RTwrtStim'].max()

    # df_valid_animal_less_than_1 = df_valid_animal[df_valid_animal['RTwrtStim'] < 1]
    # max value of RTwrtStim


    print(f'Batch: {batch_name},sample animal: {animal}')
    pdf_filename = os.path.join(output_dir, f'results_{batch_name}_animal_{animal}_lapse_fit_{init_type}.pdf')
    pdf = PdfPages(pdf_filename)
    fig_text = plt.figure(figsize=(8.5, 11)) # Standard page size looks better
    fig_text.clf() # Clear the figure
    fig_text.text(0.1, 0.9, f"Analysis Report", fontsize=20, weight='bold')
    fig_text.text(0.1, 0.8, f"Batch Name: {batch_name}", fontsize=14)
    fig_text.text(0.1, 0.75, f"Animal ID: {animal}", fontsize=14)
    fig_text.gca().axis("off")
    pdf.savefig(fig_text, bbox_inches='tight')
    plt.close(fig_text)

    ABL_arr = df_all_trials_animal['ABL'].unique()
    ILD_arr = df_all_trials_animal['ILD'].unique()


    # sort ILD arr in ascending order
    ILD_arr = np.sort(ILD_arr)
    ABL_arr = np.sort(ABL_arr)

    pkl_file = f'results_{batch_name}_animal_{animal}.pkl'
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

    # Initialize based on init_type
    if init_type == 'vanilla':
        # Initialize from vanilla lapse model fit results (values in ms converted to seconds)
        rate_lambda_0 = 0.173793
        T_0_0 = 1.958508e-3  
        theta_E_0 = 14.194689
        w_0 = 0.502231
        t_E_aff_0 = 87.604997e-3
        del_go_0 = 169.585535e-3
        rate_norm_l_0 = 0.00001
        lapse_prob_0 = 0.02
        lapse_prob_right_0 = 0.5
    elif init_type == 'norm':
        # Initialize from norm model typical values (values in ms converted to seconds)
        rate_lambda_0 = 1.8
        T_0_0 = 150e-3
        theta_E_0 = 5
        w_0 = 0.51
        t_E_aff_0 = 0.071
        del_go_0 = 0.13
        rate_norm_l_0 = 0.9
        lapse_prob_0 = 0.02
        lapse_prob_right_0 = 0.5

    x_0 = np.array([
        rate_lambda_0,
        T_0_0,
        theta_E_0,
        w_0,
        t_E_aff_0,
        del_go_0,
        rate_norm_l_0,
        lapse_prob_0,
        lapse_prob_right_0,
    ])
    
    vbmc = VBMC(vbmc_norm_tied_joint_fn, x_0, norm_tied_lb, norm_tied_ub, norm_plb, norm_pub, options={'display': 'on', 'max_fun_evals': 200 * (2 + 8)})
    vp, results = vbmc.optimize()

    if DO_RIGHT_TRUNCATE:
        vbmc_pkl_path = os.path.join(output_dir, f'vbmc_norm_tied_results_batch_{batch_name}_animal_{animal}_lapses_truncate_1s_{init_type}.pkl')
    else:
        vbmc_pkl_path = os.path.join(output_dir, f'vbmc_norm_tied_results_batch_{batch_name}_animal_{animal}_lapses_{init_type}.pkl')
    vbmc.save(vbmc_pkl_path, overwrite=True)

# %%

vp_samples = vp.sample(int(1e6))[0]
# %%
#    rate_lambda, T_0, theta_E, w, t_E_aff, del_go, rate_norm_l, lapse_prob, lapse_prob_right = params
rate_lambda_samples = vp_samples[:, 0]
T_0_samples = vp_samples[:, 1]
theta_E_samples = vp_samples[:, 2]
w_samples = vp_samples[:, 3]
t_E_aff_samples = vp_samples[:, 4]
del_go_samples = vp_samples[:, 5]
rate_norm_l_samples = vp_samples[:, 6]
lapse_prob_samples = vp_samples[:, 7]
lapse_prob_right_samples = vp_samples[:, 8]



# print mean of each sample
print("rate_lambda_samples mean: ", np.mean(rate_lambda_samples))
print("T_0_samples mean (ms): ", 1000* np.mean(T_0_samples))
print("theta_E_samples mean: ", np.mean(theta_E_samples))
print("w_samples mean: ", np.mean(w_samples))
# Z_E = (w - 0.5) * 2 * theta_E
print(f'Z_E = {(np.mean(w_samples) - 0.5) * 2 * np.mean(theta_E_samples)}')
print("t_E_aff_samples mean (ms): ", 1000* np.mean(t_E_aff_samples))
print("del_go_samples mean (ms): ", 1000* np.mean(del_go_samples))
print("rate_norm_l_samples mean: ", np.mean(rate_norm_l_samples))
print("lapse_prob_samples mean: ", np.mean(lapse_prob_samples))
print("lapse_prob_right_samples mean: ", np.mean(lapse_prob_right_samples))


# %%
# read mean from pkl file and compare those with current fit
# Load norm TIED params from pkl file
pkl_file = f'results_{batch_name}_animal_{animal_ids[0]}.pkl'
with open(pkl_file, 'rb') as f:
    fit_results_data = pickle.load(f)

# Extract norm tied samples
norm_tied_samples = fit_results_data['vbmc_norm_tied_results']
norm_rate_lambda = np.mean(norm_tied_samples['rate_lambda_samples'])
norm_T_0 = np.mean(norm_tied_samples['T_0_samples'])
norm_theta_E = np.mean(norm_tied_samples['theta_E_samples'])
norm_w = np.mean(norm_tied_samples['w_samples'])
norm_t_E_aff = np.mean(norm_tied_samples['t_E_aff_samples'])
norm_del_go = np.mean(norm_tied_samples['del_go_samples'])
norm_rate_norm_l = np.mean(norm_tied_samples['rate_norm_l_samples'])

# Compute means from lapse model
lapse_rate_lambda = np.mean(rate_lambda_samples)
lapse_T_0 = np.mean(T_0_samples)
lapse_theta_E = np.mean(theta_E_samples)
lapse_w = np.mean(w_samples)
lapse_t_E_aff = np.mean(t_E_aff_samples)
lapse_del_go = np.mean(del_go_samples)
lapse_rate_norm_l = np.mean(rate_norm_l_samples)
lapse_prob_mean = np.mean(lapse_prob_samples)
lapse_prob_right_mean = np.mean(lapse_prob_right_samples)

# Compute absolute differences
diff_rate_lambda = abs(lapse_rate_lambda - norm_rate_lambda)
diff_T_0_ms = abs(lapse_T_0 - norm_T_0) * 1000  # Convert to ms
diff_theta_E = abs(lapse_theta_E - norm_theta_E)
diff_w = abs(lapse_w - norm_w)
diff_t_E_aff_ms = abs(lapse_t_E_aff - norm_t_E_aff) * 1000  # Convert to ms
diff_del_go_ms = abs(lapse_del_go - norm_del_go) * 1000  # Convert to ms
diff_rate_norm_l = abs(lapse_rate_norm_l - norm_rate_norm_l)

# Compute percentage changes: 100 * (Lapse - Norm) / Norm
pct_rate_lambda = 100 * (lapse_rate_lambda - norm_rate_lambda) / norm_rate_lambda
pct_T_0 = 100 * (lapse_T_0 - norm_T_0) / norm_T_0
pct_theta_E = 100 * (lapse_theta_E - norm_theta_E) / norm_theta_E
pct_w = 100 * (lapse_w - norm_w) / norm_w
pct_t_E_aff = 100 * (lapse_t_E_aff - norm_t_E_aff) / norm_t_E_aff
pct_del_go = 100 * (lapse_del_go - norm_del_go) / norm_del_go
pct_rate_norm_l = 100 * (lapse_rate_norm_l - norm_rate_norm_l) / norm_rate_norm_l

# Print comparison
print("\n" + "="*95)
print(f"PARAMETER COMPARISON: Norm vs Norm+Lapse Model (Batch {batch_name}, Animal {animal_ids[0]})")
print("="*95)
print(f"{'Parameter':<20} {'Norm':<15} {'Norm+Lapse':<15} {'|Diff|':<15} {'% Change':<15}")
print("-"*95)
print(f"{'rate_lambda':<20} {norm_rate_lambda:<15.6f} {lapse_rate_lambda:<15.6f} {diff_rate_lambda:<15.6f} {pct_rate_lambda:<+15.2f}")
print(f"{'T_0 (ms)':<20} {norm_T_0*1000:<15.6f} {lapse_T_0*1000:<15.6f} {diff_T_0_ms:<15.6f} {pct_T_0:<+15.2f}")
print(f"{'theta_E':<20} {norm_theta_E:<15.6f} {lapse_theta_E:<15.6f} {diff_theta_E:<15.6f} {pct_theta_E:<+15.2f}")
print(f"{'w':<20} {norm_w:<15.6f} {lapse_w:<15.6f} {diff_w:<15.6f} {pct_w:<+15.2f}")
print(f"{'t_E_aff (ms)':<20} {norm_t_E_aff*1000:<15.6f} {lapse_t_E_aff*1000:<15.6f} {diff_t_E_aff_ms:<15.6f} {pct_t_E_aff:<+15.2f}")
print(f"{'del_go (ms)':<20} {norm_del_go*1000:<15.6f} {lapse_del_go*1000:<15.6f} {diff_del_go_ms:<15.6f} {pct_del_go:<+15.2f}")
print(f"{'rate_norm_l':<20} {norm_rate_norm_l:<15.6f} {lapse_rate_norm_l:<15.6f} {diff_rate_norm_l:<15.6f} {pct_rate_norm_l:<+15.2f}")
print(f"{'lapse_prob':<20} {'N/A':<15} {lapse_prob_mean:<15.6f} {'N/A':<15} {'N/A':<15}")
print(f"{'lapse_prob_right':<20} {'N/A':<15} {lapse_prob_right_mean:<15.6f} {'N/A':<15} {'N/A':<15}")
print("="*95)

# Save the above comparison to a text file
comparison_lines = [
    "",
    "="*95,
    f"PARAMETER COMPARISON: Norm vs Norm+Lapse Model (Batch {batch_name}, Animal {animal_ids[0]})",
    "="*95,
    f"{'Parameter':<20} {'Norm':<15} {'Norm+Lapse':<15} {'|Diff|':<15} {'% Change':<15}",
    "-"*95,
    f"{'rate_lambda':<20} {norm_rate_lambda:<15.6f} {lapse_rate_lambda:<15.6f} {diff_rate_lambda:<15.6f} {pct_rate_lambda:<+15.2f}",
    f"{'T_0 (ms)':<20} {norm_T_0*1000:<15.6f} {lapse_T_0*1000:<15.6f} {diff_T_0_ms:<15.6f} {pct_T_0:<+15.2f}",
    f"{'theta_E':<20} {norm_theta_E:<15.6f} {lapse_theta_E:<15.6f} {diff_theta_E:<15.6f} {pct_theta_E:<+15.2f}",
    f"{'w':<20} {norm_w:<15.6f} {lapse_w:<15.6f} {diff_w:<15.6f} {pct_w:<+15.2f}",
    f"{'t_E_aff (ms)':<20} {norm_t_E_aff*1000:<15.6f} {lapse_t_E_aff*1000:<15.6f} {diff_t_E_aff_ms:<15.6f} {pct_t_E_aff:<+15.2f}",
    f"{'del_go (ms)':<20} {norm_del_go*1000:<15.6f} {lapse_del_go*1000:<15.6f} {diff_del_go_ms:<15.6f} {pct_del_go:<+15.2f}",
    f"{'rate_norm_l':<20} {norm_rate_norm_l:<15.6f} {lapse_rate_norm_l:<15.6f} {diff_rate_norm_l:<15.6f} {pct_rate_norm_l:<+15.2f}",
    f"{'lapse_prob':<20} {'N/A':<15} {lapse_prob_mean:<15.6f} {'N/A':<15} {'N/A':<15}",
    f"{'lapse_prob_right':<20} {'N/A':<15} {lapse_prob_right_mean:<15.6f} {'N/A':<15} {'N/A':<15}",
    "="*95,
    ""
]
comparison_text = "\n".join(comparison_lines)
comparison_txt_path = os.path.join(output_dir, f'param_comparison_batch_{batch_name}_animal_{animal_ids[0]}_{init_type}.txt')
with open(comparison_txt_path, 'w') as f:
    f.write(comparison_text)
print(f"Saved parameter comparison to {comparison_txt_path}")

# %%
# Plot distributions of samples for each param from both fits
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
fig.suptitle(f'Parameter Distributions: Norm (blue) vs Norm+Lapse (red) - Batch {batch_name}, Animal {animal_ids[0]}', 
             fontsize=14, fontweight='bold')

params_info = [
    ('rate_lambda', norm_tied_samples['rate_lambda_samples'], rate_lambda_samples, 1, ''),
    ('T_0', norm_tied_samples['T_0_samples'], T_0_samples, 1000, '(ms)'),
    ('theta_E', norm_tied_samples['theta_E_samples'], theta_E_samples, 1, ''),
    ('w', norm_tied_samples['w_samples'], w_samples, 1, ''),
    ('t_E_aff', norm_tied_samples['t_E_aff_samples'], t_E_aff_samples, 1000, '(ms)'),
    ('del_go', norm_tied_samples['del_go_samples'], del_go_samples, 1000, '(ms)'),
    ('rate_norm_l', norm_tied_samples['rate_norm_l_samples'], rate_norm_l_samples, 1, '')
]

for idx, (param_name, norm_samples, lapse_samples, scale, unit) in enumerate(params_info):
    ax = axes[idx // 3, idx % 3]
    
    # Scale samples if needed (for time parameters)
    norm_scaled = norm_samples * scale
    lapse_scaled = lapse_samples * scale
    
    # Plot histograms with step style
    ax.hist(norm_scaled, bins=50, density=True, histtype='step', 
            color='blue', linewidth=2, label='Norm')
    ax.hist(lapse_scaled, bins=50, density=True, histtype='step', 
            color='red', linewidth=2, label='Norm+Lapse')
    
    # Add vertical lines for means
    norm_mean = np.mean(norm_scaled)
    lapse_mean = np.mean(lapse_scaled)
    ax.axvline(norm_mean, color='blue', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.axvline(lapse_mean, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    
    ax.set_xlabel(f'{param_name} {unit}', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'{param_name}', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

# Hide unused subplots
for idx in range(len(params_info), 9):
    axes[idx // 3, idx % 3].axis('off')

plt.tight_layout()
param_dist_png = os.path.join(output_dir, f'param_distributions_batch_{batch_name}_animal_{animal_ids[0]}_{init_type}.png')
fig.savefig(param_dist_png, dpi=300, bbox_inches='tight')
print(f"Saved {param_dist_png}")
plt.show()

# %%
# Plot lapse probability distributions separately
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Lapse probability
bins_lapse = np.arange(norm_lapse_prob_bounds[0], norm_lapse_prob_bounds[1], 1e-4)
mean_lapse_prob = np.mean(lapse_prob_samples)
ax1.hist(lapse_prob_samples, bins=bins_lapse, density=True, histtype='step', color='red', linewidth=2)
ax1.set_xlabel('Lapse probability', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.set_title(f'Lapse Probability Distribution', fontsize=13)
ax1.set_xlim(0, 1e-1)
ax1.axvline(x=mean_lapse_prob, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_lapse_prob:.4f}')
ax1.legend(fontsize=10)
ax1.grid(axis='y', alpha=0.3)

# Lapse probability right
bins_right = np.arange(norm_lapse_prob_right_bounds[0], norm_lapse_prob_right_bounds[1], 1e-3)
mean_lapse_prob_right = np.mean(lapse_prob_right_samples)
ax2.hist(lapse_prob_right_samples, bins=bins_right, density=True, histtype='step', color='orange', linewidth=2)
ax2.set_xlabel('Lapse P(right)', fontsize=12)
ax2.set_ylabel('Density', fontsize=12)
ax2.set_title(f'Lapse Right Bias Distribution', fontsize=13)
ax2.axvline(x=mean_lapse_prob_right, color='orange', linestyle='--', linewidth=2, label=f'Mean: {mean_lapse_prob_right:.4f}')
ax2.axvline(x=0.5, color='gray', linestyle=':', linewidth=1, label='Symmetric (0.5)')
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)

fig.suptitle(f'Lapse Parameters - Batch {batch_name}, Animal {animal_ids[0]}', fontsize=14, fontweight='bold')
plt.tight_layout()
lapse_params_png = os.path.join(output_dir, f'lapse_params_batch_{batch_name}_animal_{animal_ids[0]}_{init_type}.png')
fig.savefig(lapse_params_png, dpi=300, bbox_inches='tight')
print(f"Saved {lapse_params_png}")
plt.show()

# %%
# Simulate data with both norm and norm+lapse model params to compare RTDs

# Simulation parameters
N_sim = int(1e6)
dt = 1e-4
N_print = int(N_sim / 5)
T_lapse_max = max_rt  # Use actual max RT from data

print(f"\n{'='*60}")
print(f"SIMULATING RTDs: Norm vs Norm+Lapse Model")
print(f"Batch: {batch_name}, Animal: {animal_ids[0]}")
print(f"N_sim: {N_sim}, dt: {dt}")
print(f"{'='*60}\n")

# Sample t_stim, ABL, ILD from the animal's valid trials
t_stim_samples = df_valid_animal['intended_fix'].sample(N_sim, replace=True).values
ABL_samples = df_valid_animal['ABL'].sample(N_sim, replace=True).values
ILD_samples = df_valid_animal['ILD'].sample(N_sim, replace=True).values

# Load abort params (needed for simulation)
pkl_file = f'results_{batch_name}_animal_{animal_ids[0]}.pkl'
with open(pkl_file, 'rb') as f:
    fit_results_data = pickle.load(f)
abort_samples = fit_results_data['vbmc_aborts_results']
V_A = np.mean(abort_samples['V_A_samples'])
theta_A = np.mean(abort_samples['theta_A_samples'])
t_A_aff = np.mean(abort_samples['t_A_aff_samp'])

def simulate_single_trial_norm(i):
    choice, rt, is_act = simulate_psiam_tied_rate_norm(
        V_A, theta_A, ABL_samples[i], ILD_samples[i],
        norm_rate_lambda, norm_T_0, norm_theta_E, norm_Z_E,
        t_stim_samples[i], t_A_aff, norm_t_E_aff, norm_del_go,
        norm_rate_norm_l, dt, lapse_prob=0.0, T_lapse_max=T_lapse_max
    )
    return {
        'choice': choice,
        'rt': rt,
        'is_act': is_act,
        'ABL': ABL_samples[i],
        'ILD': ILD_samples[i],
        't_stim': t_stim_samples[i]
    }

def simulate_single_trial_lapse(i):
    choice, rt, is_act = simulate_psiam_tied_rate_norm(
        V_A, theta_A, ABL_samples[i], ILD_samples[i],
        lapse_rate_lambda, lapse_T_0, lapse_theta_E, lapse_Z_E,
        t_stim_samples[i], t_A_aff, lapse_t_E_aff, lapse_del_go,
        lapse_rate_norm_l, dt, lapse_prob=lapse_prob_mean, T_lapse_max=T_lapse_max,
        lapse_prob_right=lapse_prob_right_mean
    )
    return {
        'choice': choice,
        'rt': rt,
        'is_act': is_act,
        'ABL': ABL_samples[i],
        'ILD': ILD_samples[i],
        't_stim': t_stim_samples[i]
    }

print("Simulating with NORM model (lapse_prob=0)...")
# Norm model simulation (lapse_prob = 0)
norm_Z_E = (norm_w - 0.5) * 2 * norm_theta_E
norm_sim_results = Parallel(n_jobs=-2, verbose=5)(
    delayed(simulate_single_trial_norm)(i) for i in tqdm(range(N_sim))
)

print("\nSimulating with NORM+LAPSE model (lapse_prob={:.4f}, lapse_prob_right={:.4f})...".format(lapse_prob_mean, lapse_prob_right_mean))
# Lapse model simulation
lapse_Z_E = (lapse_w - 0.5) * 2 * lapse_theta_E
lapse_sim_results = Parallel(n_jobs=-2, verbose=5)(
    delayed(simulate_single_trial_lapse)(i) for i in tqdm(range(N_sim))
)

# Convert to DataFrames
norm_sim_df = pd.DataFrame(norm_sim_results)
lapse_sim_df = pd.DataFrame(lapse_sim_results)

# Compute RT relative to stimulus onset
norm_sim_df['rt_minus_t_stim'] = norm_sim_df['rt'] - norm_sim_df['t_stim']
lapse_sim_df['rt_minus_t_stim'] = lapse_sim_df['rt'] - lapse_sim_df['t_stim']

print(f"\nNorm simulation: {len(norm_sim_df)} trials")
print(f"Norm+Lapse simulation: {len(lapse_sim_df)} trials")
print(f"{'='*60}\n")

# %%
# Filter trials where rt_minus_t_stim > 0
norm_sim_df_filtered = norm_sim_df[norm_sim_df['rt_minus_t_stim'] > 0].copy()
lapse_sim_df_filtered = lapse_sim_df[lapse_sim_df['rt_minus_t_stim'] > 0].copy()

# Apply right truncation to simulated data if flag is set
if DO_RIGHT_TRUNCATE:
    norm_sim_df_filtered = norm_sim_df_filtered[norm_sim_df_filtered['rt_minus_t_stim'] < 1].copy()
    lapse_sim_df_filtered = lapse_sim_df_filtered[lapse_sim_df_filtered['rt_minus_t_stim'] < 1].copy()

# Create abs_ILD column
norm_sim_df_filtered['abs_ILD'] = np.abs(norm_sim_df_filtered['ILD'])
lapse_sim_df_filtered['abs_ILD'] = np.abs(lapse_sim_df_filtered['ILD'])

# Prepare empirical data
df_valid_animal_filtered = df_valid_animal[df_valid_animal['RTwrtStim'] > 0].copy()
if DO_RIGHT_TRUNCATE:
    df_valid_animal_filtered = df_valid_animal_filtered[df_valid_animal_filtered['RTwrtStim'] < 1].copy()
df_valid_animal_filtered['abs_ILD'] = np.abs(df_valid_animal_filtered['ILD'])

print(f"Filtered norm trials (rt > 0): {len(norm_sim_df_filtered)}")
print(f"Filtered norm+lapse trials (rt > 0): {len(lapse_sim_df_filtered)}")
print(f"Filtered empirical trials (rt > 0): {len(df_valid_animal_filtered)}")

# Get unique ABLs and abs_ILDs
ABL_vals = sorted(norm_sim_df_filtered['ABL'].unique())
abs_ILD_vals = sorted(norm_sim_df_filtered['abs_ILD'].unique())

print(f"ABL values: {ABL_vals}")
print(f"Absolute ILD values: {abs_ILD_vals}")

# Plot RT distributions: 3 rows (ABL) x 5 columns (abs_ILD)
fig, axes = plt.subplots(3, 5, figsize=(20, 12))
fig.suptitle(f'RT Distributions: Norm (blue) vs Norm+Lapse (red) - Batch {batch_name}, Animal {animal_ids[0]}', 
             fontsize=16, fontweight='bold')

bins = np.arange(0, 1.3, 0.01)

for row_idx, abl in enumerate(ABL_vals[:3]):  # Limit to 3 ABLs
    for col_idx, abs_ild in enumerate(abs_ILD_vals[:5]):  # Limit to 5 abs_ILDs
        ax = axes[row_idx, col_idx]
        
        # Filter data for this ABL and abs_ILD
        norm_data = norm_sim_df_filtered[
            (norm_sim_df_filtered['ABL'] == abl) & 
            (norm_sim_df_filtered['abs_ILD'] == abs_ild)
        ]['rt_minus_t_stim']
        
        lapse_data = lapse_sim_df_filtered[
            (lapse_sim_df_filtered['ABL'] == abl) & 
            (lapse_sim_df_filtered['abs_ILD'] == abs_ild)
        ]['rt_minus_t_stim']
        
        empirical_data = df_valid_animal_filtered[
            (df_valid_animal_filtered['ABL'] == abl) & 
            (df_valid_animal_filtered['abs_ILD'] == abs_ild)
        ]['RTwrtStim']
        
        # Plot histograms
        if len(norm_data) > 0:
            ax.hist(norm_data, bins=bins, density=True, histtype='step', 
                   color='blue', linewidth=2, label='Norm')
        
        if len(lapse_data) > 0:
            ax.hist(lapse_data, bins=bins, density=True, histtype='step', 
                   color='red', linewidth=2, label='Norm+Lapse')
        
        if len(empirical_data) > 0:
            ax.hist(empirical_data, bins=bins, density=True, histtype='step', 
                   color='green', linewidth=2, label='Data')
        
        # Set labels and title
        if row_idx == 0:
            ax.set_title(f'|ILD|={abs_ild}', fontsize=11)
        if col_idx == 0:
            ax.set_ylabel(f'ABL={abl}\nDensity', fontsize=10)
        if row_idx == 2:
            ax.set_xlabel('RT (s)', fontsize=10)
        
        # Add legend only to top-left subplot
        if row_idx == 0 and col_idx == 0:
            ax.legend(fontsize=9)
        
        # Add grid and format
        ax.set_xlim(0, 0.7)
        
        # Add trial counts as text
        n_norm = len(norm_data)
        n_lapse = len(lapse_data)
        n_empirical = len(empirical_data)
        
plt.tight_layout()
rt_dists_png = os.path.join(output_dir, f'rtds_norm_vs_lapse_vs_data_batch_{batch_name}_animal_{animal_ids[0]}_{init_type}.png')
fig.savefig(rt_dists_png, dpi=300, bbox_inches='tight')
print(f"Saved {rt_dists_png}")
plt.show()

# %%
# Compute log-odds: log(P(choice=1) / P(choice=-1)) vs ILD for each ABL
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f'Log-Odds: Norm (blue) vs Norm+Lapse (red) vs Data (green) - Batch {batch_name}, Animal {animal_ids[0]}', 
             fontsize=14, fontweight='bold')

# Get unique ILD values (not absolute)
ILD_vals = sorted(norm_sim_df_filtered['ILD'].unique())

for idx, abl in enumerate(ABL_vals[:3]):  # 3 ABLs
    ax = axes[idx]
    
    norm_log_odds = []
    lapse_log_odds = []
    empirical_log_odds = []
    
    for ild in ILD_vals:
        # Norm model
        norm_subset = norm_sim_df_filtered[
            (norm_sim_df_filtered['ABL'] == abl) & 
            (norm_sim_df_filtered['ILD'] == ild)
        ]
        if len(norm_subset) > 0:
            p_right = np.mean(norm_subset['choice'] == 1)
            p_left = np.mean(norm_subset['choice'] == -1)
            if p_left > 0 and p_right > 0:
                log_odds_norm = np.log(p_right / p_left)
            else:
                log_odds_norm = np.nan
        else:
            log_odds_norm = np.nan
        norm_log_odds.append(log_odds_norm)
        
        # Lapse model
        lapse_subset = lapse_sim_df_filtered[
            (lapse_sim_df_filtered['ABL'] == abl) & 
            (lapse_sim_df_filtered['ILD'] == ild)
        ]
        if len(lapse_subset) > 0:
            p_right = np.mean(lapse_subset['choice'] == 1)
            p_left = np.mean(lapse_subset['choice'] == -1)
            if p_left > 0 and p_right > 0:
                log_odds_lapse = np.log(p_right / p_left)
            else:
                log_odds_lapse = np.nan
        else:
            log_odds_lapse = np.nan
        lapse_log_odds.append(log_odds_lapse)
        
        # Empirical data
        empirical_subset = df_valid_animal_filtered[
            (df_valid_animal_filtered['ABL'] == abl) & 
            (df_valid_animal_filtered['ILD'] == ild)
        ]
        if len(empirical_subset) > 0:
            p_right = np.mean(empirical_subset['choice'] == 1)
            p_left = np.mean(empirical_subset['choice'] == -1)
            if p_left > 0 and p_right > 0:
                log_odds_empirical = np.log(p_right / p_left)
            else:
                log_odds_empirical = np.nan
        else:
            log_odds_empirical = np.nan
        empirical_log_odds.append(log_odds_empirical)
    
    # Plot
    ax.plot(ILD_vals, norm_log_odds, 'o', color='blue', markersize=8, 
            label='Norm', markerfacecolor='blue', markeredgecolor='blue')
    ax.plot(ILD_vals, lapse_log_odds, 'x', color='red', markersize=10, 
            markeredgewidth=2, label='Norm+Lapse')
    ax.plot(ILD_vals, empirical_log_odds, 's', color='green', markersize=8, 
            label='Data', markerfacecolor='green', markeredgecolor='green')
    
    # Formatting
    ax.set_title(f'ABL = {abl} dB', fontsize=12)
    ax.set_xlabel('ILD (dB)', fontsize=11)
    if idx == 0:
        ax.set_ylabel('log(P(right) / P(left))', fontsize=11)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Set consistent y-limits across subplots
    ax.set_ylim(-5, 5)

plt.tight_layout()
log_odds_png = os.path.join(output_dir, f'log_odds_norm_vs_lapse_vs_data_batch_{batch_name}_animal_{animal_ids[0]}_{init_type}.png')
fig.savefig(log_odds_png, dpi=300, bbox_inches='tight')
print(f"Saved {log_odds_png}")
plt.show()

# %%
# Psychometric curves: P(choice=1) vs ILD for each ABL
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f'Psychometric Curves: Norm (blue) vs Norm+Lapse (red) vs Data (green) - Batch {batch_name}, Animal {animal_ids[0]}', 
             fontsize=14, fontweight='bold')

for idx, abl in enumerate(ABL_vals[:3]):  # 3 ABLs
    ax = axes[idx]
    
    norm_p_right = []
    lapse_p_right = []
    empirical_p_right = []
    
    for ild in ILD_vals:
        # Norm model
        norm_subset = norm_sim_df_filtered[
            (norm_sim_df_filtered['ABL'] == abl) & 
            (norm_sim_df_filtered['ILD'] == ild)
        ]
        if len(norm_subset) > 0:
            p_right_norm = np.mean(norm_subset['choice'] == 1)
        else:
            p_right_norm = np.nan
        norm_p_right.append(p_right_norm)
        
        # Lapse model
        lapse_subset = lapse_sim_df_filtered[
            (lapse_sim_df_filtered['ABL'] == abl) & 
            (lapse_sim_df_filtered['ILD'] == ild)
        ]
        if len(lapse_subset) > 0:
            p_right_lapse = np.mean(lapse_subset['choice'] == 1)
        else:
            p_right_lapse = np.nan
        lapse_p_right.append(p_right_lapse)
        
        # Empirical data
        empirical_subset = df_valid_animal_filtered[
            (df_valid_animal_filtered['ABL'] == abl) & 
            (df_valid_animal_filtered['ILD'] == ild)
        ]
        if len(empirical_subset) > 0:
            p_right_empirical = np.mean(empirical_subset['choice'] == 1)
        else:
            p_right_empirical = np.nan
        empirical_p_right.append(p_right_empirical)
    
    # Plot
    ax.plot(ILD_vals, norm_p_right, 'o', color='blue', markersize=8, 
            label='Norm', markerfacecolor='blue', markeredgecolor='blue')
    ax.plot(ILD_vals, lapse_p_right, 'x', color='red', markersize=10, 
            markeredgewidth=2, label='Norm+Lapse')
    ax.plot(ILD_vals, empirical_p_right, 's', color='green', markersize=8, 
            label='Data', markerfacecolor='green', markeredgecolor='green', alpha=0.3)
    
    # Formatting
    ax.set_title(f'ABL = {abl} dB', fontsize=12)
    ax.set_xlabel('ILD (dB)', fontsize=11)
    if idx == 0:
        ax.set_ylabel('P(choice = right)', fontsize=11)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Set consistent y-limits
    ax.set_ylim(0, 1)

plt.tight_layout()
psycho_png = os.path.join(output_dir, f'psychometric_norm_vs_lapse_vs_data_batch_{batch_name}_animal_{animal_ids[0]}_{init_type}.png')
fig.savefig(psycho_png, dpi=300, bbox_inches='tight')
print(f"Saved {psycho_png}")
plt.show()

# %%
# Close PDF file handle if created
try:
    pdf.close()
    print(f"Saved PDF report to {pdf_filename}")
except Exception:
    pass

# %%
# initilziation values
print(f'rate_lamda_0 = {rate_lambda_0}')
print(f'T_0_0 = {T_0_0* 1e3} ms')
print(f'theta_E_0 = {theta_E_0}')
print(f'w_0 = {w_0}')
print(f't_E_aff_0 = {t_E_aff_0 * 1e3} ms')
print(f'del_go_0 = {del_go_0 * 1e3} ms')
print(f'rate_norm_l_0 = {rate_norm_l_0}')
print(f'lapse_prob_0 = {lapse_prob_0}')
print(f'lapse_prob_right_0 = {lapse_prob_right_0}')
