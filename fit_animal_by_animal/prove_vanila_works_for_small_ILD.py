# %%
import matplotlib
matplotlib.use('Agg')     # or 'PDF', 'SVG', etc., any non-Tk backend

import pandas as pd
import matplotlib.pyplot as plt

  
import numpy as np
import os
from joblib import Parallel, delayed
from tqdm import tqdm
from animal_wise_plotting_utils import prepare_simulation_data, calculate_theoretical_curves, plot_rt_distributions, plot_tachometric_curves, plot_grand_summary
import pickle
import warnings
from types import SimpleNamespace
from collections import defaultdict
import random
from time_vary_and_norm_simulators import psiam_tied_data_gen_wrapper_rate_norm_fn
from animal_wise_plotting_utils import save_posterior_summary_page

from pyvbmc import VBMC
import corner
from matplotlib.backends.backend_pdf import PdfPages
from vbmc_animal_wise_fit_utils import trapezoidal_logpdf
from time_vary_norm_utils import (
    up_or_down_RTs_fit_fn, cum_pro_and_reactive_time_vary_fn,
    rho_A_t_VEC_fn, up_or_down_RTs_fit_wrt_stim_fn, rho_A_t_fn, cum_A_t_fn)
from time_vary_norm_utils import up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn
from animal_wise_config import T_trunc

# %%
N_theory = int(1e3)
N_sim = int(1e6)
dt  = 1e-3
N_print = int(N_sim / 5)
K_max = 10

# %%
# Vanilla VBMC stuff
######### Vanilla TIED ###############
def compute_loglike_vanilla(row, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go):
    
    rt = row['TotalFixTime']
    t_stim = row['intended_fix']
    
    
    ILD = row['ILD']
    ABL = row['ABL']
    choice = row['choice']

    pdf = up_or_down_RTs_fit_fn(
            rt, choice,
            V_A, theta_A, t_A_aff,
            t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go,
            phi_params_obj, rate_norm_l, 
            is_norm, is_time_vary, K_max)

    trunc_factor_p_joint = cum_pro_and_reactive_time_vary_fn(
                            t_stim + 1, T_trunc,
                            V_A, theta_A, t_A_aff,
                            t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_E_aff,
                            phi_params_obj, rate_norm_l, 
                            is_norm, is_time_vary, K_max) \
                            - \
                            cum_pro_and_reactive_time_vary_fn(
                            t_stim, T_trunc,
                            V_A, theta_A, t_A_aff,
                            t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_E_aff,
                            phi_params_obj, rate_norm_l, 
                            is_norm, is_time_vary, K_max)
                           

    pdf /= (trunc_factor_p_joint + 1e-20)
    pdf = max(pdf, 1e-50)
    if np.isnan(pdf):
        print(f'row["abort_event"] = {row["abort_event"]}')
        print(f'row["RTwrtStim"] = {row["RTwrtStim"]}')
        raise ValueError(f'nan pdf rt = {rt}, t_stim = {t_stim}')
    return np.log(pdf)
    
    

## loglike
def vbmc_vanilla_tied_loglike_fn(params):
    rate_lambda, T_0, theta_E, w, t_E_aff, del_go = params
    Z_E = (w - 0.5) * 2 * theta_E
    all_loglike = Parallel(n_jobs=30)(delayed(compute_loglike_vanilla)(row, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go)\
                                       for _, row in df_animal_success_lt_1_ILD_124.iterrows() )
    return np.sum(all_loglike)


def vbmc_vanilla_tied_prior_fn(params):
    rate_lambda, T_0, theta_E, w, t_E_aff, del_go = params

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
    
    return (
        rate_lambda_logpdf +
        T_0_logpdf +
        theta_E_logpdf +
        w_logpdf +
        t_E_aff_logpdf +
        del_go_logpdf
    )


## joint
def vbmc_vanilla_tied_joint_fn(params):
    priors = vbmc_vanilla_tied_prior_fn(params)
    loglike = vbmc_vanilla_tied_loglike_fn(params)

    return priors + loglike

## bounds
vanilla_rate_lambda_bounds = [0.01, 1]
vanilla_T_0_bounds = [0.1e-3, 2.2e-3]
vanilla_theta_E_bounds = [5, 65]
vanilla_w_bounds = [0.3, 0.7]
vanilla_t_E_aff_bounds = [0.01, 0.2]
vanilla_del_go_bounds = [0, 0.2]


vanilla_rate_lambda_plausible_bounds = [0.1, 0.3]
vanilla_T_0_plausible_bounds = [0.5e-3, 1.5e-3]
vanilla_theta_E_plausible_bounds = [15, 55]
vanilla_w_plausible_bounds = [0.4, 0.6]
vanilla_t_E_aff_plausible_bounds = [0.03, 0.09]
vanilla_del_go_plausible_bounds = [0.05, 0.15]

## bounds array
vanilla_tied_lb = np.array([
    vanilla_rate_lambda_bounds[0],
    vanilla_T_0_bounds[0],
    vanilla_theta_E_bounds[0],
    vanilla_w_bounds[0],
    vanilla_t_E_aff_bounds[0],
    vanilla_del_go_bounds[0]
])

vanilla_tied_ub = np.array([
    vanilla_rate_lambda_bounds[1],
    vanilla_T_0_bounds[1],
    vanilla_theta_E_bounds[1],
    vanilla_w_bounds[1],
    vanilla_t_E_aff_bounds[1],
    vanilla_del_go_bounds[1]
])

vanilla_plb = np.array([
    vanilla_rate_lambda_plausible_bounds[0],
    vanilla_T_0_plausible_bounds[0],
    vanilla_theta_E_plausible_bounds[0],
    vanilla_w_plausible_bounds[0],
    vanilla_t_E_aff_plausible_bounds[0],
    vanilla_del_go_plausible_bounds[0]
])

vanilla_pub = np.array([
    vanilla_rate_lambda_plausible_bounds[1],
    vanilla_T_0_plausible_bounds[1],
    vanilla_theta_E_plausible_bounds[1],
    vanilla_w_plausible_bounds[1],
    vanilla_t_E_aff_plausible_bounds[1],
    vanilla_del_go_plausible_bounds[1]
])

# %%
# Read 30 animals
DESIRED_BATCHES = ['SD', 'LED34', 'LED6', 'LED8', 'LED7', 'LED34_even']
batch_dir = os.path.join(os.path.dirname(__file__), 'batch_csvs')
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
    pkl_file = f'results_{batch_name}_animal_{animal_id}.pkl'
    with open(pkl_file, 'rb') as f:
        fit_results_data = pickle.load(f)
    vbmc_aborts_param_keys_map = {
        'V_A_samples': 'V_A',
        'theta_A_samples': 'theta_A',
        't_A_aff_samp': 't_A_aff'
    }
    abort_keyname = "vbmc_aborts_results"
    abort_params = {}
    if abort_keyname in fit_results_data:
        abort_samples = fit_results_data[abort_keyname]
        for param_samples_name, param_label in vbmc_aborts_param_keys_map.items():
            abort_params[param_label] = np.mean(abort_samples[param_samples_name])
    return abort_params

# %%
for batch, animal in batch_animal_pairs:
# for batch, animal in [batch_animal_pairs[0]]:

    # get csv
    file_name = f'batch_csvs/batch_{batch}_valid_and_aborts.csv'
    df = pd.read_csv(file_name)
    df_animal = df[df['animal'] == animal]
    # success trials 
    df_animal_success = df_animal[df_animal['success'].isin([1, -1])]
    # RTs <= 1
    df_animal_success_lt_1 = df_animal_success[df_animal_success['RTwrtStim'] <= 1]
    # ILDs: 1,2,4
    df_animal_success_lt_1_ILD_124 = df_animal_success_lt_1[df_animal_success_lt_1['ILD'].isin([1, 2, 4, -1, -2, -4])]

    # ABL and ILDs
    ABL_arr = np.sort(df_animal_success_lt_1_ILD_124['ABL'].unique())
    ILD_arr = np.sort(df_animal_success_lt_1_ILD_124['ILD'].unique())
    print('===================================================================')
    print(f'Batch: {batch}, Animal: {animal}, ABL: {ABL_arr}, ILD: {ILD_arr}')
    print('===================================================================')
    
    # get abort params
    abort_params = get_params_from_animal_pkl_file(batch, animal)
    V_A = abort_params['V_A']
    theta_A = abort_params['theta_A']
    t_A_aff = abort_params['t_A_aff']

    # Vanilla TIED fitting
    is_norm = False
    is_time_vary = False
    phi_params_obj = np.nan
    rate_norm_l = np.nan

    rate_lambda_0 = 0.17
    T_0_0 = 1.4 * 1e-3
    theta_E_0 = 20
    w_0 = 0.51
    t_E_aff_0 = 0.071
    del_go_0 = 0.13

    x_0 = np.array([
        rate_lambda_0,
        T_0_0,
        theta_E_0,
        w_0,
        t_E_aff_0,
        del_go_0
    ])

    # Run VBMC
    vbmc = VBMC(vbmc_vanilla_tied_joint_fn, x_0, vanilla_tied_lb, vanilla_tied_ub, vanilla_plb, vanilla_pub, options={'display': 'on', 'max_fun_evals': 200 * (2 + 6)})
    vp, results = vbmc.optimize()

    #### Diagnostics ####
    pdf_filename = f'small_ILDs_only_results_{batch}_animal_{animal}.pdf'
    pdf = PdfPages(pdf_filename)

    vp_samples = vp.sample(int(1e5))[0]
    vp_samples[:, 1] *= 1e3
    param_labels = [
        r'$\lambda$',       # 3
        r'$T_0$ (ms)',      # 4
        r'$\theta_E$',      # 5
        r'$w$',             # 6
        r'$t_E^{aff}$',     # 7
        r'$\Delta_{go}$'    # 8
    ]
    percentiles = np.percentile(vp_samples, [1, 99], axis=0)
    _ranges = [(percentiles[0, i], percentiles[1, i]) for i in range(vp_samples.shape[1])]
    vanilla_tied_corner_fig = corner.corner(
        vp_samples,
        labels=param_labels,
        show_titles=True,
        quantiles=[0.025, 0.5, 0.975],
        range=_ranges,
        title_fmt=".3f"
    )
    vanilla_tied_corner_fig.suptitle(f'Vanilla Tied Posterior (Animal: {animal})', y=1.02) # Add a title to the corner plot figure
    vp_samples[:, 1] /= 1e3
 
    rate_lambda = vp_samples[:, 0].mean()
    T_0 = vp_samples[:, 1].mean()
    theta_E = vp_samples[:, 2].mean()
    w = vp_samples[:, 3].mean()
    Z_E = (w - 0.5) * 2 * theta_E
    t_E_aff = vp_samples[:, 4].mean()
    del_go = vp_samples[:, 5].mean()

    # Print them out
    print("Posterior Means:")
    print(f"rate_lambda  = {rate_lambda:.5f}")
    print(f"T_0 (ms)      = {1e3*T_0:.5f}")
    print(f"theta_E       = {theta_E:.5f}")
    print(f"Z_E           = {Z_E:.5f}")
    print(f"t_E_aff       = {1e3*t_E_aff:.5f} ms")
    print(f"del_go   = {del_go:.5f}")

    vanilla_tied_loglike = vbmc_vanilla_tied_loglike_fn([rate_lambda, T_0, theta_E, w, t_E_aff, del_go])
    # --- Page: Vanilla Tied Model Posterior Means ---
    save_posterior_summary_page(
        pdf_pages=pdf,
        title=f'Vanilla Tied Model - Posterior Means ({animal})',
        posterior_means=pd.Series({
            'rate_lambda': rate_lambda,
            'T_0': 1e3*T_0,
            'theta_E': theta_E,
            'w': w,
            'Z_E': Z_E,
            't_E_aff': 1e3*t_E_aff,
            'del_go': del_go
        }),
        param_labels={
            'rate_lambda': r'$\lambda$',
            'T_0': r'$T_0$ (ms)',
            'theta_E': r'$\theta_E$',
            'w': r'$w$',
            'Z_E': r'$Z_E$',
            't_E_aff': r'$t_E^{aff}$',
            'del_go': r'$\Delta_{go}$'
        },
        vbmc_results={'message': results['message'], 'elbo': results['elbo'], 'elbo_sd': results['elbo_sd'], 'loglike': vanilla_tied_loglike, 'convergence_status': results.get('convergence_status'), 'r_index': results.get('r_index'), 'success_flag': results.get('success_flag')},
        extra_text=f"T_trunc = {T_trunc:.3f}"
    )

    # Create the corner plot

    pdf.savefig(vanilla_tied_corner_fig, bbox_inches='tight')
    plt.close(vanilla_tied_corner_fig) # Close the figure
    # Convert T_0 back to original units if needed

        # --- Page 4: Vanilla Tied Results ---

    vbmc_vanilla_tied_results = {
        'rate_lambda_samples': vp_samples[:, 0],
        'T_0_samples': vp_samples[:, 1],
        'theta_E_samples': vp_samples[:, 2],
        'w_samples': vp_samples[:, 3],
        't_E_aff_samples': vp_samples[:, 4],
        'del_go_samples': vp_samples[:, 5],
        'message': results['message'],
        'elbo': results['elbo'],
        'elbo_sd': results['elbo_sd'],
        'loglike': vanilla_tied_loglike
    }

    ######################################################
    ########### VANILLA TIED diagnostics #####################
    ########################################################


    rate_norm_l = 0
    is_norm = False
    is_time_vary = False
    phi_params_obj = np.nan

    ### Stimulus samples for all 3 TIED models
    t_stim_samples = df_animal_success_lt_1_ILD_124['intended_fix'].sample(N_sim, replace=True).values
    ABL_samples = df_animal_success_lt_1_ILD_124['ABL'].sample(N_sim, replace=True).values
    ILD_samples = df_animal_success_lt_1_ILD_124['ILD'].sample(N_sim, replace=True).values
    
    # P_A and C_A vs t wrt stim for all 3 TIED models
    t_pts = np.arange(-1, 2, 0.001)
    P_A_mean, C_A_mean, t_stim_samples_for_diag = calculate_theoretical_curves(
        df_animal_success_lt_1_ILD_124, N_theory, t_pts, t_A_aff, V_A, theta_A, rho_A_t_fn
    )

    # Run simulations in parallel
    sim_results = Parallel(n_jobs=30)(
        delayed(psiam_tied_data_gen_wrapper_rate_norm_fn)(
            V_A, theta_A, ABL_samples[iter_num], ILD_samples[iter_num], rate_lambda, T_0, theta_E, Z_E, t_A_aff, t_E_aff, del_go, 
            t_stim_samples[iter_num], rate_norm_l, iter_num, N_print, dt
        ) for iter_num in tqdm(range(N_sim))
    )

    # Convert results to DataFrame and prepare simulation data
    sim_results_df = pd.DataFrame(sim_results)
    sim_df_1, data_df_1 = prepare_simulation_data(sim_results_df, df_animal_success_lt_1_ILD_124)

    

    # Plot RT distributions and get theoretical results for later use
    theory_results_up_and_down, theory_time_axis, bins, bin_centers = plot_rt_distributions(
        sim_df_1, data_df_1, ILD_arr, ABL_arr, t_pts, P_A_mean, C_A_mean, 
        t_stim_samples_for_diag, V_A, theta_A, t_A_aff, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go,
        phi_params_obj, rate_norm_l, is_norm, is_time_vary, K_max, T_trunc,
        cum_pro_and_reactive_time_vary_fn, up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn,
        animal, pdf, model_name="Vanilla Tied"
    )

    plot_tachometric_curves(
        sim_df_1, data_df_1, ILD_arr, ABL_arr, theory_results_up_and_down,
        theory_time_axis, bins, animal, pdf, model_name="Vanilla Tied"
    )

    plot_grand_summary(
        sim_df_1, data_df_1, ILD_arr, ABL_arr, bins, bin_centers,
        animal, pdf, model_name="Vanilla Tied"
    )

    save_dict = {
        'vbmc_vanilla_tied_results': vbmc_vanilla_tied_results,
    }
    pkl_filename = f'small_ILDs_only_results_{batch}_animal_{animal}.pkl'

    with open(pkl_filename, 'wb') as f:
        pickle.dump(save_dict, f)

    print(f"Saved results to {pkl_filename}")
    pdf.close()
    print(f"Saved PDF report to {pdf_filename}")

