# VBMC fit on exp data single animal - lapses + vanila model
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
parser = argparse.ArgumentParser(description='Fit vanilla+lapse model for a single animal')
parser.add_argument('--batch', required=True, help='Batch name, e.g., LED8')
parser.add_argument('--animal', required=True, type=int, help='Animal ID (int)')
parser.add_argument('--output-dir', default='oct_9_10_vanila_lapse_model_fit_files', help='Directory to save results')
args = parser.parse_args()

batch_name = args.batch
animal_ids = [args.animal]
output_dir = args.output_dir
# batch_name = 'LED8'
# animal_ids = [109]
# output_dir = 'oct_9_10_vanila_lapse_model_fit_files'

os.makedirs(output_dir, exist_ok=True)

phi_params_obj = np.nan
rate_norm_l = np.nan
is_norm = False
is_time_vary = False
K_max = 10

if batch_name == 'LED34_even':
    T_trunc = 0.15
else:
    T_trunc = 0.3

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
def compute_loglike_vanilla(row, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go, lapse_prob, lapse_prob_right):
    
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
                            is_norm, is_time_vary, K_max) - \
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


def vbmc_vanilla_tied_loglike_fn(params):
    rate_lambda, T_0, theta_E, w, t_E_aff, del_go, lapse_prob, lapse_prob_right = params
    Z_E = (w - 0.5) * 2 * theta_E
    all_loglike = Parallel(n_jobs=30)(delayed(compute_loglike_vanilla)(row, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go, lapse_prob, lapse_prob_right)\
                                       for _, row in df_valid_animal.iterrows() )
    return np.sum(all_loglike)


def vbmc_vanilla_tied_prior_fn(params):
    rate_lambda, T_0, theta_E, w, t_E_aff, del_go, lapse_prob, lapse_prob_right = params

    rate_lambda_logpdf = trapezoidal_logpdf(
        rate_lambda,
        vanilla_rate_lambda_bounds[0],
        vanilla_rate_lambda_plausible_bounds[0],
        vanilla_rate_lambda_plausible_bounds[1],
        vanilla_rate_lambda_bounds[1]
    )
    
    T_0_logpdf = trapezoidal_logpdf(
        T_0,
        vanilla_T_0_bounds[0],
        vanilla_T_0_plausible_bounds[0],
        vanilla_T_0_plausible_bounds[1],
        vanilla_T_0_bounds[1]
    )
    
    theta_E_logpdf = trapezoidal_logpdf(
        theta_E,
        vanilla_theta_E_bounds[0],
        vanilla_theta_E_plausible_bounds[0],
        vanilla_theta_E_plausible_bounds[1],
        vanilla_theta_E_bounds[1]
    )
    
    w_logpdf = trapezoidal_logpdf(
        w,
        vanilla_w_bounds[0],
        vanilla_w_plausible_bounds[0],
        vanilla_w_plausible_bounds[1],
        vanilla_w_bounds[1]
    )
    
    t_E_aff_logpdf = trapezoidal_logpdf(
        t_E_aff,
        vanilla_t_E_aff_bounds[0],
        vanilla_t_E_aff_plausible_bounds[0],
        vanilla_t_E_aff_plausible_bounds[1],
        vanilla_t_E_aff_bounds[1]
    )
    
    del_go_logpdf = trapezoidal_logpdf(
        del_go,
        vanilla_del_go_bounds[0],
        vanilla_del_go_plausible_bounds[0],
        vanilla_del_go_plausible_bounds[1],
        vanilla_del_go_bounds[1]
    )
    
    lapse_prob_logpdf = trapezoidal_logpdf(
        lapse_prob,
        vanilla_lapse_prob_bounds[0],
        vanilla_lapse_prob_plausible_bounds[0],
        vanilla_lapse_prob_plausible_bounds[1],
        vanilla_lapse_prob_bounds[1]
    )
    
    lapse_prob_right_logpdf = trapezoidal_logpdf(
        lapse_prob_right,
        vanilla_lapse_prob_right_bounds[0],
        vanilla_lapse_prob_right_plausible_bounds[0],
        vanilla_lapse_prob_right_plausible_bounds[1],
        vanilla_lapse_prob_right_bounds[1]
    )
    
    return (
        rate_lambda_logpdf +
        T_0_logpdf +
        theta_E_logpdf +
        w_logpdf +
        t_E_aff_logpdf +
        del_go_logpdf +
        lapse_prob_logpdf +
        lapse_prob_right_logpdf
    )

def vbmc_vanilla_tied_joint_fn(params):
    priors = vbmc_vanilla_tied_prior_fn(params)
    loglike = vbmc_vanilla_tied_loglike_fn(params)

    return priors + loglike


vanilla_rate_lambda_bounds = [0.01, 1]
vanilla_T_0_bounds = [0.1e-3, 2.2e-3]
vanilla_theta_E_bounds = [5, 65]
vanilla_w_bounds = [0.3, 0.7]
vanilla_t_E_aff_bounds = [0.01, 0.2]
vanilla_del_go_bounds = [0, 0.2]
vanilla_lapse_prob_bounds = [1e-4, 0.2]
vanilla_lapse_prob_right_bounds = [0.001, 0.999]

vanilla_rate_lambda_plausible_bounds = [0.1, 0.3]
vanilla_T_0_plausible_bounds = [0.5e-3, 1.5e-3]
vanilla_theta_E_plausible_bounds = [15, 55]
vanilla_w_plausible_bounds = [0.4, 0.6]
vanilla_t_E_aff_plausible_bounds = [0.03, 0.09]
vanilla_del_go_plausible_bounds = [0.05, 0.15]
vanilla_lapse_prob_plausible_bounds = [1e-3, 0.1]
vanilla_lapse_prob_right_plausible_bounds = [0.4, 0.6]


vanilla_tied_lb = np.array([
    vanilla_rate_lambda_bounds[0],
    vanilla_T_0_bounds[0],
    vanilla_theta_E_bounds[0],
    vanilla_w_bounds[0],
    vanilla_t_E_aff_bounds[0],
    vanilla_del_go_bounds[0],
    vanilla_lapse_prob_bounds[0],
    vanilla_lapse_prob_right_bounds[0]
])

vanilla_tied_ub = np.array([
    vanilla_rate_lambda_bounds[1],
    vanilla_T_0_bounds[1],
    vanilla_theta_E_bounds[1],
    vanilla_w_bounds[1],
    vanilla_t_E_aff_bounds[1],
    vanilla_del_go_bounds[1],
    vanilla_lapse_prob_bounds[1],
    vanilla_lapse_prob_right_bounds[1]
])

vanilla_plb = np.array([
    vanilla_rate_lambda_plausible_bounds[0],
    vanilla_T_0_plausible_bounds[0],
    vanilla_theta_E_plausible_bounds[0],
    vanilla_w_plausible_bounds[0],
    vanilla_t_E_aff_plausible_bounds[0],
    vanilla_del_go_plausible_bounds[0],
    vanilla_lapse_prob_plausible_bounds[0],
    vanilla_lapse_prob_right_plausible_bounds[0]
])

vanilla_pub = np.array([
    vanilla_rate_lambda_plausible_bounds[1],
    vanilla_T_0_plausible_bounds[1],
    vanilla_theta_E_plausible_bounds[1],
    vanilla_w_plausible_bounds[1],
    vanilla_t_E_aff_plausible_bounds[1],
    vanilla_del_go_plausible_bounds[1],
    vanilla_lapse_prob_plausible_bounds[1],
    vanilla_lapse_prob_right_plausible_bounds[1]
])
# %%
print(f'len of animal_ids = {len(animal_ids)}')

# %%
for animal_idx in [0]:
    animal = animal_ids[animal_idx]

    df_all_trials_animal = df_valid_and_aborts[df_valid_and_aborts['animal'] == animal]
    df_aborts_animal = df_aborts[df_aborts['animal'] == animal]

    df_valid_animal = df_all_trials_animal[df_all_trials_animal['success'].isin([1,-1])]
    # Right Truncation
    if DO_RIGHT_TRUNCATE:
        df_valid_animal = df_valid_animal[df_valid_animal['RTwrtStim'] < 1]
        max_rt = 1
    else:
        max_rt = df_valid_animal['RTwrtStim'].max()

    # df_valid_animal_less_than_1 = df_valid_animal[df_valid_animal['RTwrtStim'] < 1]
    # max value of RTwrtStim


    print(f'Batch: {batch_name},sample animal: {animal}')
    pdf_filename = os.path.join(output_dir, f'results_{batch_name}_animal_{animal}_lapse_vanilla_model.pdf')
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

    rate_lambda_0 = 0.17
    T_0_0 = 1.4 * 1e-3
    theta_E_0 = 20
    w_0 = 0.51
    t_E_aff_0 = 0.071
    del_go_0 = 0.13
    lapse_prob_0 = 0.02
    lapse_prob_right_0 = 0.5  # Start with symmetric assumption
    x_0 = np.array([
        rate_lambda_0,
        T_0_0,
        theta_E_0,
        w_0,
        t_E_aff_0,
        del_go_0,
        lapse_prob_0,
        lapse_prob_right_0,
    ])
    
    vbmc = VBMC(vbmc_vanilla_tied_joint_fn, x_0, vanilla_tied_lb, vanilla_tied_ub, vanilla_plb, vanilla_pub, options={'display': 'on', 'max_fun_evals': 50 * (2 + 7)})
    vp, results = vbmc.optimize()

    if DO_RIGHT_TRUNCATE:
        vbmc_pkl_path = os.path.join(output_dir, f'vbmc_vanilla_tied_results_batch_{batch_name}_animal_{animal}_lapses_truncate_1s.pkl')
    else:
        vbmc_pkl_path = os.path.join(output_dir, f'vbmc_vanilla_tied_results_batch_{batch_name}_animal_{animal}_lapses.pkl')
    vbmc.save(vbmc_pkl_path, overwrite=True)

# %%

vp_samples = vp.sample(int(1e6))[0]
# %%
#    rate_lambda, T_0, theta_E, w, t_E_aff, del_go, lapse_prob, lapse_prob_right = params
rate_lambda_samples = vp_samples[:, 0]
T_0_samples = vp_samples[:, 1]
theta_E_samples = vp_samples[:, 2]
w_samples = vp_samples[:, 3]
t_E_aff_samples = vp_samples[:, 4]
del_go_samples = vp_samples[:, 5]
lapse_prob_samples = vp_samples[:, 6]
lapse_prob_right_samples = vp_samples[:, 7]



# print mean of each sample
print("rate_lambda_samples mean: ", np.mean(rate_lambda_samples))
print("T_0_samples mean (ms): ", 1000* np.mean(T_0_samples))
print("theta_E_samples mean: ", np.mean(theta_E_samples))
print("w_samples mean: ", np.mean(w_samples))
# Z_E = (w - 0.5) * 2 * theta_E
print(f'Z_E = {(np.mean(w_samples) - 0.5) * 2 * np.mean(theta_E_samples)}')
print("t_E_aff_samples mean (ms): ", 1000* np.mean(t_E_aff_samples))
print("del_go_samples mean (ms): ", 1000* np.mean(del_go_samples))
print("lapse_prob_samples mean: ", np.mean(lapse_prob_samples))
print("lapse_prob_right_samples mean: ", np.mean(lapse_prob_right_samples))


# %%
# read mean from pkl file and compare those with current fit
# Load vanilla TIED params from pkl file
pkl_file = f'results_{batch_name}_animal_{animal_ids[0]}.pkl'
with open(pkl_file, 'rb') as f:
    fit_results_data = pickle.load(f)

# Extract vanilla tied samples
vanilla_tied_samples = fit_results_data['vbmc_vanilla_tied_results']
vanilla_rate_lambda = np.mean(vanilla_tied_samples['rate_lambda_samples'])
vanilla_T_0 = np.mean(vanilla_tied_samples['T_0_samples'])
vanilla_theta_E = np.mean(vanilla_tied_samples['theta_E_samples'])
vanilla_w = np.mean(vanilla_tied_samples['w_samples'])
vanilla_t_E_aff = np.mean(vanilla_tied_samples['t_E_aff_samples'])
vanilla_del_go = np.mean(vanilla_tied_samples['del_go_samples'])

# Compute means from lapse model
lapse_rate_lambda = np.mean(rate_lambda_samples)
lapse_T_0 = np.mean(T_0_samples)
lapse_theta_E = np.mean(theta_E_samples)
lapse_w = np.mean(w_samples)
lapse_t_E_aff = np.mean(t_E_aff_samples)
lapse_del_go = np.mean(del_go_samples)
lapse_prob_mean = np.mean(lapse_prob_samples)
lapse_prob_right_mean = np.mean(lapse_prob_right_samples)

# Compute absolute differences
diff_rate_lambda = abs(lapse_rate_lambda - vanilla_rate_lambda)
diff_T_0_ms = abs(lapse_T_0 - vanilla_T_0) * 1000  # Convert to ms
diff_theta_E = abs(lapse_theta_E - vanilla_theta_E)
diff_w = abs(lapse_w - vanilla_w)
diff_t_E_aff_ms = abs(lapse_t_E_aff - vanilla_t_E_aff) * 1000  # Convert to ms
diff_del_go_ms = abs(lapse_del_go - vanilla_del_go) * 1000  # Convert to ms

# Compute percentage changes: 100 * (Lapse - Vanilla) / Vanilla
pct_rate_lambda = 100 * (lapse_rate_lambda - vanilla_rate_lambda) / vanilla_rate_lambda
pct_T_0 = 100 * (lapse_T_0 - vanilla_T_0) / vanilla_T_0
pct_theta_E = 100 * (lapse_theta_E - vanilla_theta_E) / vanilla_theta_E
pct_w = 100 * (lapse_w - vanilla_w) / vanilla_w
pct_t_E_aff = 100 * (lapse_t_E_aff - vanilla_t_E_aff) / vanilla_t_E_aff
pct_del_go = 100 * (lapse_del_go - vanilla_del_go) / vanilla_del_go

# Print comparison
print("\n" + "="*95)
print(f"PARAMETER COMPARISON: Vanilla vs Lapse Model (Batch {batch_name}, Animal {animal_ids[0]})")
print("="*95)
print(f"{'Parameter':<20} {'Vanilla':<15} {'Lapse':<15} {'|Diff|':<15} {'% Change':<15}")
print("-"*95)
print(f"{'rate_lambda':<20} {vanilla_rate_lambda:<15.6f} {lapse_rate_lambda:<15.6f} {diff_rate_lambda:<15.6f} {pct_rate_lambda:<+15.2f}")
print(f"{'T_0 (ms)':<20} {vanilla_T_0*1000:<15.6f} {lapse_T_0*1000:<15.6f} {diff_T_0_ms:<15.6f} {pct_T_0:<+15.2f}")
print(f"{'theta_E':<20} {vanilla_theta_E:<15.6f} {lapse_theta_E:<15.6f} {diff_theta_E:<15.6f} {pct_theta_E:<+15.2f}")
print(f"{'w':<20} {vanilla_w:<15.6f} {lapse_w:<15.6f} {diff_w:<15.6f} {pct_w:<+15.2f}")
print(f"{'t_E_aff (ms)':<20} {vanilla_t_E_aff*1000:<15.6f} {lapse_t_E_aff*1000:<15.6f} {diff_t_E_aff_ms:<15.6f} {pct_t_E_aff:<+15.2f}")
print(f"{'del_go (ms)':<20} {vanilla_del_go*1000:<15.6f} {lapse_del_go*1000:<15.6f} {diff_del_go_ms:<15.6f} {pct_del_go:<+15.2f}")
print(f"{'lapse_prob':<20} {'N/A':<15} {lapse_prob_mean:<15.6f} {'N/A':<15} {'N/A':<15}")
print(f"{'lapse_prob_right':<20} {'N/A':<15} {lapse_prob_right_mean:<15.6f} {'N/A':<15} {'N/A':<15}")
print("="*95)

# Save the above comparison to a text file
comparison_lines = [
    "",
    "="*95,
    f"PARAMETER COMPARISON: Vanilla vs Lapse Model (Batch {batch_name}, Animal {animal_ids[0]})",
    "="*95,
    f"{'Parameter':<20} {'Vanilla':<15} {'Lapse':<15} {'|Diff|':<15} {'% Change':<15}",
    "-"*95,
    f"{'rate_lambda':<20} {vanilla_rate_lambda:<15.6f} {lapse_rate_lambda:<15.6f} {diff_rate_lambda:<15.6f} {pct_rate_lambda:<+15.2f}",
    f"{'T_0 (ms)':<20} {vanilla_T_0*1000:<15.6f} {lapse_T_0*1000:<15.6f} {diff_T_0_ms:<15.6f} {pct_T_0:<+15.2f}",
    f"{'theta_E':<20} {vanilla_theta_E:<15.6f} {lapse_theta_E:<15.6f} {diff_theta_E:<15.6f} {pct_theta_E:<+15.2f}",
    f"{'w':<20} {vanilla_w:<15.6f} {lapse_w:<15.6f} {diff_w:<15.6f} {pct_w:<+15.2f}",
    f"{'t_E_aff (ms)':<20} {vanilla_t_E_aff*1000:<15.6f} {lapse_t_E_aff*1000:<15.6f} {diff_t_E_aff_ms:<15.6f} {pct_t_E_aff:<+15.2f}",
    f"{'del_go (ms)':<20} {vanilla_del_go*1000:<15.6f} {lapse_del_go*1000:<15.6f} {diff_del_go_ms:<15.6f} {pct_del_go:<+15.2f}",
    f"{'lapse_prob':<20} {'N/A':<15} {lapse_prob_mean:<15.6f} {'N/A':<15} {'N/A':<15}",
    f"{'lapse_prob_right':<20} {'N/A':<15} {lapse_prob_right_mean:<15.6f} {'N/A':<15} {'N/A':<15}",
    "="*95,
    ""
]
comparison_text = "\n".join(comparison_lines)
comparison_txt_path = os.path.join(output_dir, f'param_comparison_batch_{batch_name}_animal_{animal_ids[0]}_vanilla_lapse.txt')
with open(comparison_txt_path, 'w') as f:
    f.write(comparison_text)
print(f"Saved parameter comparison to {comparison_txt_path}")

# %%
# Plot distributions of samples for each param from both fits
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle(f'Parameter Distributions: Vanilla (blue) vs Lapse (red) - Batch {batch_name}, Animal {animal_ids[0]}', 
             fontsize=14, fontweight='bold')

params_info = [
    ('rate_lambda', vanilla_tied_samples['rate_lambda_samples'], rate_lambda_samples, 1, ''),
    ('T_0', vanilla_tied_samples['T_0_samples'], T_0_samples, 1000, '(ms)'),
    ('theta_E', vanilla_tied_samples['theta_E_samples'], theta_E_samples, 1, ''),
    ('w', vanilla_tied_samples['w_samples'], w_samples, 1, ''),
    ('t_E_aff', vanilla_tied_samples['t_E_aff_samples'], t_E_aff_samples, 1000, '(ms)'),
    ('del_go', vanilla_tied_samples['del_go_samples'], del_go_samples, 1000, '(ms)')
]

for idx, (param_name, vanilla_samples, lapse_samples, scale, unit) in enumerate(params_info):
    ax = axes[idx // 3, idx % 3]
    
    # Scale samples if needed (for time parameters)
    vanilla_scaled = vanilla_samples * scale
    lapse_scaled = lapse_samples * scale
    
    # Plot histograms with step style
    ax.hist(vanilla_scaled, bins=50, density=True, histtype='step', 
            color='blue', linewidth=2, label='Vanilla')
    ax.hist(lapse_scaled, bins=50, density=True, histtype='step', 
            color='red', linewidth=2, label='Lapse')
    
    # Add vertical lines for means
    vanilla_mean = np.mean(vanilla_scaled)
    lapse_mean = np.mean(lapse_scaled)
    ax.axvline(vanilla_mean, color='blue', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.axvline(lapse_mean, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    
    ax.set_xlabel(f'{param_name} {unit}', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'{param_name}', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
param_dist_png = os.path.join(output_dir, f'param_distributions_batch_{batch_name}_animal_{animal_ids[0]}_vanilla_lapse.png')
fig.savefig(param_dist_png, dpi=300, bbox_inches='tight')
print(f"Saved {param_dist_png}")
plt.show()

# %%
# Plot lapse probability distribution separately
plt.figure(figsize=(6, 4))
bins = np.arange(vanilla_lapse_prob_bounds[0], vanilla_lapse_prob_bounds[1], 1e-4)
# bins = np.arange(vanilla_lapse_prob_bounds[0], 0.1, 1e-4)

mean_lapse_prob = np.mean(lapse_prob_samples)
plt.hist(lapse_prob_samples, bins=bins, density=True, histtype='step', color='red', linewidth=2)

plt.xlabel('Lapse probability', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title(f'Lapse Probability Distribution - Batch {batch_name}, Animal {animal_ids[0]}', fontsize=13)
plt.xlim(0, 1e-1)
plt.axvline(x=mean_lapse_prob, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_lapse_prob:.4f}')
plt.legend(fontsize=10)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
lapse_prob_png = os.path.join(output_dir, f'lapse_prob_distribution_batch_{batch_name}_animal_{animal_ids[0]}_vanilla_lapse.png')
plt.savefig(lapse_prob_png, dpi=300, bbox_inches='tight')
print(f"Saved {lapse_prob_png}")
plt.show()

# %%
# Simulate data with both vanilla and lapse model params to compare RTDs

# Simulation parameters
N_sim = int(1e6)
dt = 1e-4
N_print = int(N_sim / 5)
T_lapse_max = max_rt  # Use actual max RT from data

print(f"\n{'='*60}")
print(f"SIMULATING RTDs: Vanilla vs Lapse Model")
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

def simulate_single_trial_vanilla(i):
    choice, rt, is_act = simulate_psiam_tied_rate_norm(
        V_A, theta_A, ABL_samples[i], ILD_samples[i],
        vanilla_rate_lambda, vanilla_T_0, vanilla_theta_E, vanilla_Z_E,
        t_stim_samples[i], t_A_aff, vanilla_t_E_aff, vanilla_del_go,
        0.0, dt, lapse_prob=0.0, T_lapse_max=T_lapse_max
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
        0.0, dt, lapse_prob=lapse_prob_mean, T_lapse_max=T_lapse_max,
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

print("Simulating with VANILLA model (lapse_prob=0)...")
# Vanilla model simulation (lapse_prob = 0)
vanilla_Z_E = (vanilla_w - 0.5) * 2 * vanilla_theta_E
vanilla_sim_results = Parallel(n_jobs=-2, verbose=5)(
    delayed(simulate_single_trial_vanilla)(i) for i in tqdm(range(N_sim))
)

print("\nSimulating with LAPSE model (lapse_prob={:.4f}, lapse_prob_right={:.4f})...".format(lapse_prob_mean, lapse_prob_right_mean))
# Lapse model simulation
lapse_Z_E = (lapse_w - 0.5) * 2 * lapse_theta_E
lapse_sim_results = Parallel(n_jobs=-2, verbose=5)(
    delayed(simulate_single_trial_lapse)(i) for i in tqdm(range(N_sim))
)

# Convert to DataFrames
vanilla_sim_df = pd.DataFrame(vanilla_sim_results)
lapse_sim_df = pd.DataFrame(lapse_sim_results)

# Compute RT relative to stimulus onset
vanilla_sim_df['rt_minus_t_stim'] = vanilla_sim_df['rt'] - vanilla_sim_df['t_stim']
lapse_sim_df['rt_minus_t_stim'] = lapse_sim_df['rt'] - lapse_sim_df['t_stim']

print(f"\nVanilla simulation: {len(vanilla_sim_df)} trials")
print(f"Lapse simulation: {len(lapse_sim_df)} trials")
print(f"{'='*60}\n")

# %%
# Filter trials where rt_minus_t_stim > 0
vanilla_sim_df_filtered = vanilla_sim_df[vanilla_sim_df['rt_minus_t_stim'] > 0].copy()
lapse_sim_df_filtered = lapse_sim_df[lapse_sim_df['rt_minus_t_stim'] > 0].copy()

# Apply right truncation to simulated data if flag is set
if DO_RIGHT_TRUNCATE:
    vanilla_sim_df_filtered = vanilla_sim_df_filtered[vanilla_sim_df_filtered['rt_minus_t_stim'] < 1].copy()
    lapse_sim_df_filtered = lapse_sim_df_filtered[lapse_sim_df_filtered['rt_minus_t_stim'] < 1].copy()

# Create abs_ILD column
vanilla_sim_df_filtered['abs_ILD'] = np.abs(vanilla_sim_df_filtered['ILD'])
lapse_sim_df_filtered['abs_ILD'] = np.abs(lapse_sim_df_filtered['ILD'])

# Prepare empirical data
df_valid_animal_filtered = df_valid_animal[df_valid_animal['RTwrtStim'] > 0].copy()
if DO_RIGHT_TRUNCATE:
    df_valid_animal_filtered = df_valid_animal_filtered[df_valid_animal_filtered['RTwrtStim'] < 1].copy()
df_valid_animal_filtered['abs_ILD'] = np.abs(df_valid_animal_filtered['ILD'])

print(f"Filtered vanilla trials (rt > 0): {len(vanilla_sim_df_filtered)}")
print(f"Filtered lapse trials (rt > 0): {len(lapse_sim_df_filtered)}")
print(f"Filtered empirical trials (rt > 0): {len(df_valid_animal_filtered)}")

# Get unique ABLs and abs_ILDs
ABL_vals = sorted(vanilla_sim_df_filtered['ABL'].unique())
abs_ILD_vals = sorted(vanilla_sim_df_filtered['abs_ILD'].unique())

print(f"ABL values: {ABL_vals}")
print(f"Absolute ILD values: {abs_ILD_vals}")

# Plot RT distributions: 3 rows (ABL) x 5 columns (abs_ILD)
fig, axes = plt.subplots(3, 5, figsize=(20, 12))
fig.suptitle(f'RT Distributions: Vanilla (blue) vs Lapse (red) - Batch {batch_name}, Animal {animal_ids[0]}', 
             fontsize=16, fontweight='bold')

bins = np.arange(0, 1.3, 0.01)

for row_idx, abl in enumerate(ABL_vals[:3]):  # Limit to 3 ABLs
    for col_idx, abs_ild in enumerate(abs_ILD_vals[:5]):  # Limit to 5 abs_ILDs
        ax = axes[row_idx, col_idx]
        
        # Filter data for this ABL and abs_ILD
        vanilla_data = vanilla_sim_df_filtered[
            (vanilla_sim_df_filtered['ABL'] == abl) & 
            (vanilla_sim_df_filtered['abs_ILD'] == abs_ild)
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
        if len(vanilla_data) > 0:
            ax.hist(vanilla_data, bins=bins, density=True, histtype='step', 
                   color='blue', linewidth=2, label='Vanilla')
        
        if len(lapse_data) > 0:
            ax.hist(lapse_data, bins=bins, density=True, histtype='step', 
                   color='red', linewidth=2, label='Lapse')
        
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
        n_vanilla = len(vanilla_data)
        n_lapse = len(lapse_data)
        n_empirical = len(empirical_data)
        
plt.tight_layout()
rt_dists_png = os.path.join(output_dir, f'rtds_vanilla_vs_lapse_vs_data_batch_{batch_name}_animal_{animal_ids[0]}.png')
fig.savefig(rt_dists_png, dpi=300, bbox_inches='tight')
print(f"Saved {rt_dists_png}")
plt.show()

# %%
# Compute log-odds: log(P(choice=1) / P(choice=-1)) vs ILD for each ABL
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f'Log-Odds: Vanilla (blue) vs Lapse (red) vs Data (green) - Batch {batch_name}, Animal {animal_ids[0]}', 
             fontsize=14, fontweight='bold')

# Get unique ILD values (not absolute)
ILD_vals = sorted(vanilla_sim_df_filtered['ILD'].unique())

for idx, abl in enumerate(ABL_vals[:3]):  # 3 ABLs
    ax = axes[idx]
    
    vanilla_log_odds = []
    lapse_log_odds = []
    empirical_log_odds = []
    
    for ild in ILD_vals:
        # Vanilla model
        vanilla_subset = vanilla_sim_df_filtered[
            (vanilla_sim_df_filtered['ABL'] == abl) & 
            (vanilla_sim_df_filtered['ILD'] == ild)
        ]
        if len(vanilla_subset) > 0:
            p_right = np.mean(vanilla_subset['choice'] == 1)
            p_left = np.mean(vanilla_subset['choice'] == -1)
            if p_left > 0 and p_right > 0:
                log_odds_vanilla = np.log(p_right / p_left)
            else:
                log_odds_vanilla = np.nan
        else:
            log_odds_vanilla = np.nan
        vanilla_log_odds.append(log_odds_vanilla)
        
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
    ax.plot(ILD_vals, vanilla_log_odds, 'o', color='blue', markersize=8, 
            label='Vanilla', markerfacecolor='blue', markeredgecolor='blue')
    ax.plot(ILD_vals, lapse_log_odds, 'x', color='red', markersize=10, 
            markeredgewidth=2, label='Lapse')
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
log_odds_png = os.path.join(output_dir, f'log_odds_vanilla_vs_lapse_vs_data_batch_{batch_name}_animal_{animal_ids[0]}.png')
fig.savefig(log_odds_png, dpi=300, bbox_inches='tight')
print(f"Saved {log_odds_png}")
plt.show()

# %%
# Psychometric curves: P(choice=1) vs ILD for each ABL
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f'Psychometric Curves: Vanilla (blue) vs Lapse (red) vs Data (green) - Batch {batch_name}, Animal {animal_ids[0]}', 
             fontsize=14, fontweight='bold')

for idx, abl in enumerate(ABL_vals[:3]):  # 3 ABLs
    ax = axes[idx]
    
    vanilla_p_right = []
    lapse_p_right = []
    empirical_p_right = []
    
    for ild in ILD_vals:
        # Vanilla model
        vanilla_subset = vanilla_sim_df_filtered[
            (vanilla_sim_df_filtered['ABL'] == abl) & 
            (vanilla_sim_df_filtered['ILD'] == ild)
        ]
        if len(vanilla_subset) > 0:
            p_right_vanilla = np.mean(vanilla_subset['choice'] == 1)
        else:
            p_right_vanilla = np.nan
        vanilla_p_right.append(p_right_vanilla)
        
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
    ax.plot(ILD_vals, vanilla_p_right, 'o', color='blue', markersize=8, 
            label='Vanilla', markerfacecolor='blue', markeredgecolor='blue')
    ax.plot(ILD_vals, lapse_p_right, 'x', color='red', markersize=10, 
            markeredgewidth=2, label='Lapse')
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
psycho_png = os.path.join(output_dir, f'psychometric_vanilla_vs_lapse_vs_data_batch_{batch_name}_animal_{animal_ids[0]}.png')
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