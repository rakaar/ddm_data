# %%
"""
Animal-wise Vanilla TIED Model Fitting Script
Extracts vanilla model fitting from the 3-models refactor script.
Includes abort model fitting (needed for vanilla model dependencies).
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from animal_wise_plotting_utils import (
    render_df_to_pdf, create_abort_table, create_tied_table,
    save_posterior_summary_page, save_corner_plot, plot_abort_diagnostic,
    prepare_simulation_data, calculate_theoretical_curves, 
    plot_rt_distributions, plot_tachometric_curves, plot_grand_summary
)
from joblib import Parallel, delayed
from tqdm import tqdm
import pickle
from types import SimpleNamespace
from pyvbmc import VBMC
import random
from time_vary_norm_utils import (
    up_or_down_RTs_fit_fn, cum_pro_and_reactive_time_vary_fn,
    rho_A_t_VEC_fn, rho_A_t_fn, cum_A_t_fn, 
    up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn
)
import corner
from time_vary_and_norm_simulators import psiam_tied_data_gen_wrapper_rate_norm_fn
from vbmc_animal_wise_fit_utils import trapezoidal_logpdf
from animal_wise_config import T_trunc
from matplotlib.backends.backend_pdf import PdfPages

############3 Params #############
batch_name = 'LED34'
animal_to_fit = [63]
K_max = 10
N_theory = int(1e3)
N_sim = int(1e6)
dt  = 1e-3
N_print = int(N_sim / 5)



######### Vanilla TIED ###############
def compute_loglike_vanilla(row, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go):
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

def vbmc_vanilla_tied_loglike_fn(params):
    rate_lambda, T_0, theta_E, w, t_E_aff, del_go = params
    Z_E = (w - 0.5) * 2 * theta_E
    all_loglike = Parallel(n_jobs=30)(
        delayed(compute_loglike_vanilla)(row, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go)
        for _, row in df_valid_animal_less_than_1.iterrows()
    )
    return np.sum(all_loglike)

def vbmc_vanilla_tied_prior_fn(params):
    rate_lambda, T_0, theta_E, w, t_E_aff, del_go = params
    
    logpdfs = [
        trapezoidal_logpdf(rate_lambda, vanilla_rate_lambda_bounds[0], vanilla_rate_lambda_plausible_bounds[0], 
                          vanilla_rate_lambda_plausible_bounds[1], vanilla_rate_lambda_bounds[1]),
        trapezoidal_logpdf(T_0, vanilla_T_0_bounds[0], vanilla_T_0_plausible_bounds[0], 
                          vanilla_T_0_plausible_bounds[1], vanilla_T_0_bounds[1]),
        trapezoidal_logpdf(theta_E, vanilla_theta_E_bounds[0], vanilla_theta_E_plausible_bounds[0], 
                          vanilla_theta_E_plausible_bounds[1], vanilla_theta_E_bounds[1]),
        trapezoidal_logpdf(w, vanilla_w_bounds[0], vanilla_w_plausible_bounds[0], 
                          vanilla_w_plausible_bounds[1], vanilla_w_bounds[1]),
        trapezoidal_logpdf(t_E_aff, vanilla_t_E_aff_bounds[0], vanilla_t_E_aff_plausible_bounds[0], 
                          vanilla_t_E_aff_plausible_bounds[1], vanilla_t_E_aff_bounds[1]),
        trapezoidal_logpdf(del_go, vanilla_del_go_bounds[0], vanilla_del_go_plausible_bounds[0], 
                          vanilla_del_go_plausible_bounds[1], vanilla_del_go_bounds[1])
    ]
    return sum(logpdfs)

def vbmc_vanilla_tied_joint_fn(params):
    return vbmc_vanilla_tied_prior_fn(params) + vbmc_vanilla_tied_loglike_fn(params)

# Vanilla bounds
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

vanilla_tied_lb = np.array([vanilla_rate_lambda_bounds[0], vanilla_T_0_bounds[0], vanilla_theta_E_bounds[0], 
                            vanilla_w_bounds[0], vanilla_t_E_aff_bounds[0], vanilla_del_go_bounds[0]])
vanilla_tied_ub = np.array([vanilla_rate_lambda_bounds[1], vanilla_T_0_bounds[1], vanilla_theta_E_bounds[1], 
                            vanilla_w_bounds[1], vanilla_t_E_aff_bounds[1], vanilla_del_go_bounds[1]])
vanilla_plb = np.array([vanilla_rate_lambda_plausible_bounds[0], vanilla_T_0_plausible_bounds[0], vanilla_theta_E_plausible_bounds[0], 
                        vanilla_w_plausible_bounds[0], vanilla_t_E_aff_plausible_bounds[0], vanilla_del_go_plausible_bounds[0]])
vanilla_pub = np.array([vanilla_rate_lambda_plausible_bounds[1], vanilla_T_0_plausible_bounds[1], vanilla_theta_E_plausible_bounds[1], 
                        vanilla_w_plausible_bounds[1], vanilla_t_E_aff_plausible_bounds[1], vanilla_del_go_plausible_bounds[1]])

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
    pdf_filename = f'results_{batch_name}_animal_{animal}_VANILLA_ABL_ILD_filtered.pdf'
    pdf = PdfPages(pdf_filename)

    # Page 1: Title
    fig_text = plt.figure(figsize=(8.5, 11))
    fig_text.clf()
    fig_text.text(0.1, 0.9, f"Vanilla Model Analysis Report", fontsize=20, weight='bold')
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
        
        print(f'Loaded abort params - V_A: {V_A:.4f}, theta_A: {theta_A:.4f}, t_A_aff: {t_A_aff:.4f}')
        
        # Save abort summary to PDF
        fig_text = plt.figure(figsize=(8.5, 11))
        fig_text.clf()
        fig_text.text(0.1, 0.9, f"Abort Parameters (Loaded from {pkl_file})", fontsize=16, weight='bold')
        fig_text.text(0.1, 0.8, f"V_A: {V_A:.4f}", fontsize=12)
        fig_text.text(0.1, 0.75, f"theta_A: {theta_A:.4f}", fontsize=12)
        fig_text.text(0.1, 0.7, f"t_A_aff: {t_A_aff:.4f}", fontsize=12)
        fig_text.gca().axis("off")
        pdf.savefig(fig_text, bbox_inches='tight')
        plt.close(fig_text)
        
        vbmc_aborts_results = abort_samples
        
    except FileNotFoundError:
        print(f"ERROR: Could not find {pkl_file}. Please run abort fitting first.")
        pdf.close()
        continue

    ######################################################
    ############### VANILLA TIED MODEL ###################
    ######################################################
    print("\n### Fitting Vanilla TIED Model ###")
    
    is_norm, is_time_vary = False, False
    phi_params_obj, rate_norm_l = np.nan, np.nan

    # Filter data for specific ABLs and ILDs
    print("\n### Filtering data for specific ABLs and ILDs ###")
    allowed_abls = [20, 40, 60]
    allowed_ilds = [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]
    
    df_all_trials_animal_filtered = df_all_trials_animal[
        (df_all_trials_animal['ABL'].isin(allowed_abls)) &
        (df_all_trials_animal['ILD'].isin(allowed_ilds))
    ].copy()
    
    print(f"Original trials: {len(df_all_trials_animal)}")
    print(f"Filtered trials: {len(df_all_trials_animal_filtered)}")
    print(f"ABLs in filtered data: {sorted(df_all_trials_animal_filtered['ABL'].unique())}")
    print(f"ILDs in filtered data: {sorted(df_all_trials_animal_filtered['ILD'].unique())}")
    
    # Update arrays for plotting
    ABL_arr = np.sort(df_all_trials_animal_filtered['ABL'].unique())
    ILD_arr = np.sort(df_all_trials_animal_filtered['ILD'].unique())
    
    df_valid_animal = df_all_trials_animal_filtered[df_all_trials_animal_filtered['success'].isin([1,-1])]
    df_valid_animal_less_than_1 = df_valid_animal[df_valid_animal['RTwrtStim'] < 1]

    rate_lambda_0, T_0_0, theta_E_0, w_0, t_E_aff_0, del_go_0 = 0.17, 1.4e-3, 20, 0.51, 0.071, 0.13
    x_0 = np.array([rate_lambda_0, T_0_0, theta_E_0, w_0, t_E_aff_0, del_go_0])

    vbmc = VBMC(vbmc_vanilla_tied_joint_fn, x_0, vanilla_tied_lb, vanilla_tied_ub, vanilla_plb, vanilla_pub, 
                options={'display': 'on', 'max_fun_evals': 200 * (2 + 6)})
    vp, results = vbmc.optimize()
    vbmc.save(f'vbmc_PKL_file_vanilla_tied_results_batch_{batch_name}_animal_{animal}_FILTERED.pkl')

    vp_samples = vp.sample(int(1e5))[0]
    vp_samples[:, 1] *= 1e3  # Convert T_0 to ms for display

    param_labels = [r'$\lambda$', r'$T_0$ (ms)', r'$\theta_E$', r'$w$', r'$t_E^{aff}$', r'$\Delta_{go}$']
    percentiles = np.percentile(vp_samples, [1, 99], axis=0)
    _ranges = [(percentiles[0, i], percentiles[1, i]) for i in range(vp_samples.shape[1])]

    vanilla_tied_corner_fig = corner.corner(vp_samples, labels=param_labels, show_titles=True,
                                            quantiles=[0.025, 0.5, 0.975], range=_ranges, title_fmt=".3f")
    vanilla_tied_corner_fig.suptitle(f'Vanilla Tied Posterior (Animal: {animal})', y=1.02)
    
    vp_samples[:, 1] /= 1e3  # Convert back to seconds

    rate_lambda, T_0, theta_E, w, t_E_aff, del_go = vp_samples.mean(axis=0)
    Z_E = (w - 0.5) * 2 * theta_E

    print(f"rate_lambda={rate_lambda:.5f}, T_0={1e3*T_0:.5f}ms, theta_E={theta_E:.5f}, "
          f"w={w:.5f}, t_E_aff={1e3*t_E_aff:.5f}ms, del_go={del_go:.5f}")

    vanilla_tied_loglike = vbmc_vanilla_tied_loglike_fn([rate_lambda, T_0, theta_E, w, t_E_aff, del_go])

    save_posterior_summary_page(
        pdf, f'Vanilla Tied Model - Posterior Means ({animal})',
        pd.Series({'rate_lambda': rate_lambda, 'T_0': 1e3*T_0, 'theta_E': theta_E, 
                  'w': w, 'Z_E': Z_E, 't_E_aff': 1e3*t_E_aff, 'del_go': del_go}),
        {'rate_lambda': r'$\lambda$', 'T_0': r'$T_0$ (ms)', 'theta_E': r'$\theta_E$', 
         'w': r'$w$', 'Z_E': r'$Z_E$', 't_E_aff': r'$t_E^{aff}$', 'del_go': r'$\Delta_{go}$'},
        {'message': results['message'], 'elbo': results['elbo'], 'elbo_sd': results['elbo_sd'], 
         'loglike': vanilla_tied_loglike},
        f"T_trunc = {T_trunc:.3f}"
    )

    pdf.savefig(vanilla_tied_corner_fig, bbox_inches='tight')
    plt.close(vanilla_tied_corner_fig)

    vbmc_vanilla_tied_results = {
        'rate_lambda_samples': vp_samples[:, 0], 'T_0_samples': vp_samples[:, 1],
        'theta_E_samples': vp_samples[:, 2], 'w_samples': vp_samples[:, 3],
        't_E_aff_samples': vp_samples[:, 4], 'del_go_samples': vp_samples[:, 5],
        'message': results['message'], 'elbo': results['elbo'], 
        'elbo_sd': results['elbo_sd'], 'loglike': vanilla_tied_loglike
    }

    ######################################################
    ########### VANILLA TIED diagnostics #################
    ########################################################
    print("\n### Running Vanilla TIED Diagnostics ###")
    
    rate_norm_l = 0

    t_stim_samples = df_valid_animal_less_than_1['intended_fix'].sample(N_sim, replace=True).values
    ABL_samples = df_valid_animal_less_than_1['ABL'].sample(N_sim, replace=True).values
    ILD_samples = df_valid_animal_less_than_1['ILD'].sample(N_sim, replace=True).values
    
    t_pts = np.arange(-1, 2, 0.001)
    P_A_mean, C_A_mean, t_stim_samples_for_diag = calculate_theoretical_curves(
        df_valid_and_aborts, N_theory, t_pts, t_A_aff, V_A, theta_A, rho_A_t_fn
    )

    sim_results = Parallel(n_jobs=30)(
        delayed(psiam_tied_data_gen_wrapper_rate_norm_fn)(
            V_A, theta_A, ABL_samples[i], ILD_samples[i], rate_lambda, T_0, theta_E, Z_E, 
            t_A_aff, t_E_aff, del_go, t_stim_samples[i], rate_norm_l, i, N_print, dt
        ) for i in tqdm(range(N_sim))
    )

    sim_results_df = pd.DataFrame(sim_results)
    sim_df_1, data_df_1 = prepare_simulation_data(sim_results_df, df_valid_animal_less_than_1)

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

    ############### END Of vanilla tied model #####################

    # Create result tables
    print("\nGenerating model comparison tables...")
    abort_df = create_abort_table(vbmc_aborts_results)
    if abort_df is not None:
        render_df_to_pdf(abort_df, f"Abort Model Results - Animal {animal}", pdf)

    save_dict = {
        'vbmc_aborts_results': vbmc_aborts_results,
        'vbmc_vanilla_tied_results': vbmc_vanilla_tied_results
    }
    
    # Save results
    pkl_filename = f'results_{batch_name}_animal_{animal}_VANILLA_ABL_ILD_filtered.pkl'
    with open(pkl_filename, 'wb') as f:
        pickle.dump(save_dict, f)
    print(f"Saved results to {pkl_filename}")

    pdf.close()
    print(f"Saved PDF report to {pdf_filename}")
