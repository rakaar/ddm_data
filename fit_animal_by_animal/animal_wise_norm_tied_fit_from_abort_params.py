# %%
"""
Animal-wise normalized TIED fit using previously fitted proactive parameters.

This is the normalized TIED portion of animal_wise_fit_3_models_script_refactor.py
ported into a standalone script. It loads V_A, theta_A, and t_A_aff from an
existing animal-wise result pickle and skips abort re-fitting.
"""
import os
import pickle
import random

import corner
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.backends.backend_pdf import PdfPages
from pyvbmc import VBMC
from tqdm.notebook import tqdm

from animal_wise_config import T_trunc
from animal_wise_plotting_utils import (
    calculate_theoretical_curves,
    create_abort_table,
    create_tied_table,
    plot_grand_summary,
    plot_rt_distributions,
    plot_tachometric_curves,
    prepare_simulation_data,
    render_df_to_pdf,
    save_posterior_summary_page,
)
from time_vary_and_norm_simulators import psiam_tied_data_gen_wrapper_rate_norm_fn
from time_vary_norm_utils import (
    cum_pro_and_reactive_time_vary_fn,
    rho_A_t_fn,
    up_or_down_RTs_fit_fn,
    up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn,
)
from vbmc_animal_wise_fit_utils import trapezoidal_logpdf


# %%
############3 Params #############
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

batch_name = 'SD'
animal_to_fit = [49]
K_max = 10

N_theory = int(1e3)
N_sim = int(1e6)
dt = 1e-3
N_print = int(N_sim / 5)


# %%
###########  Normalized TIED ##############
def compute_loglike_norm_fn(row, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go, rate_norm_l):
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
        is_norm, is_time_vary, K_max
    )

    trunc_factor_p_joint = cum_pro_and_reactive_time_vary_fn(
        t_stim + 1, T_trunc,
        V_A, theta_A, t_A_aff,
        t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_E_aff,
        phi_params_obj, rate_norm_l,
        is_norm, is_time_vary, K_max
    ) - cum_pro_and_reactive_time_vary_fn(
        t_stim, T_trunc,
        V_A, theta_A, t_A_aff,
        t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_E_aff,
        phi_params_obj, rate_norm_l,
        is_norm, is_time_vary, K_max
    )

    pdf /= (trunc_factor_p_joint + 1e-20)
    pdf = max(pdf, 1e-50)
    if np.isnan(pdf):
        print(f'row["abort_event"] = {row["abort_event"]}')
        print(f'row["RTwrtStim"] = {row["RTwrtStim"]}')
        raise ValueError(f'nan pdf rt = {rt}, t_stim = {t_stim}')
    return np.log(pdf)


def vbmc_norm_tied_loglike_fn(params):
    rate_lambda, T_0, theta_E, w, t_E_aff, del_go, rate_norm_l = params
    Z_E = (w - 0.5) * 2 * theta_E
    all_loglike = Parallel(n_jobs=30)(
        delayed(compute_loglike_norm_fn)(
            row, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go, rate_norm_l
        )
        for _, row in df_valid_animal_less_than_1.iterrows()
    )
    return np.sum(all_loglike)


def vbmc_prior_norm_tied_fn(params):
    rate_lambda, T_0, theta_E, w, t_E_aff, del_go, rate_norm_l = params

    rate_lambda_logpdf = trapezoidal_logpdf(
        rate_lambda,
        norm_tied_rate_lambda_bounds[0],
        norm_tied_rate_lambda_plausible_bounds[0],
        norm_tied_rate_lambda_plausible_bounds[1],
        norm_tied_rate_lambda_bounds[1]
    )

    T_0_logpdf = trapezoidal_logpdf(
        T_0,
        norm_tied_T_0_bounds[0],
        norm_tied_T_0_plausible_bounds[0],
        norm_tied_T_0_plausible_bounds[1],
        norm_tied_T_0_bounds[1]
    )

    theta_E_logpdf = trapezoidal_logpdf(
        theta_E,
        norm_tied_theta_E_bounds[0],
        norm_tied_theta_E_plausible_bounds[0],
        norm_tied_theta_E_plausible_bounds[1],
        norm_tied_theta_E_bounds[1]
    )

    w_logpdf = trapezoidal_logpdf(
        w,
        norm_tied_w_bounds[0],
        norm_tied_w_plausible_bounds[0],
        norm_tied_w_plausible_bounds[1],
        norm_tied_w_bounds[1]
    )

    t_E_aff_logpdf = trapezoidal_logpdf(
        t_E_aff,
        norm_tied_t_E_aff_bounds[0],
        norm_tied_t_E_aff_plausible_bounds[0],
        norm_tied_t_E_aff_plausible_bounds[1],
        norm_tied_t_E_aff_bounds[1]
    )

    del_go_logpdf = trapezoidal_logpdf(
        del_go,
        norm_tied_del_go_bounds[0],
        norm_tied_del_go_plausible_bounds[0],
        norm_tied_del_go_plausible_bounds[1],
        norm_tied_del_go_bounds[1]
    )

    rate_norm_l_logpdf = trapezoidal_logpdf(
        rate_norm_l,
        norm_tied_rate_norm_bounds[0],
        norm_tied_rate_norm_plausible_bounds[0],
        norm_tied_rate_norm_plausible_bounds[1],
        norm_tied_rate_norm_bounds[1]
    )

    return (
        rate_lambda_logpdf +
        T_0_logpdf +
        theta_E_logpdf +
        w_logpdf +
        t_E_aff_logpdf +
        del_go_logpdf +
        rate_norm_l_logpdf
    )


def vbmc_norm_tied_joint_fn(params):
    priors = vbmc_prior_norm_tied_fn(params)
    loglike = vbmc_norm_tied_loglike_fn(params)

    return priors + loglike


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

norm_tied_lb = np.array([
    norm_tied_rate_lambda_bounds[0],
    norm_tied_T_0_bounds[0],
    norm_tied_theta_E_bounds[0],
    norm_tied_w_bounds[0],
    norm_tied_t_E_aff_bounds[0],
    norm_tied_del_go_bounds[0],
    norm_tied_rate_norm_bounds[0]
])

norm_tied_ub = np.array([
    norm_tied_rate_lambda_bounds[1],
    norm_tied_T_0_bounds[1],
    norm_tied_theta_E_bounds[1],
    norm_tied_w_bounds[1],
    norm_tied_t_E_aff_bounds[1],
    norm_tied_del_go_bounds[1],
    norm_tied_rate_norm_bounds[1]
])

norm_tied_plb = np.array([
    norm_tied_rate_lambda_plausible_bounds[0],
    norm_tied_T_0_plausible_bounds[0],
    norm_tied_theta_E_plausible_bounds[0],
    norm_tied_w_plausible_bounds[0],
    norm_tied_t_E_aff_plausible_bounds[0],
    norm_tied_del_go_plausible_bounds[0],
    norm_tied_rate_norm_plausible_bounds[0]
])

norm_tied_pub = np.array([
    norm_tied_rate_lambda_plausible_bounds[1],
    norm_tied_T_0_plausible_bounds[1],
    norm_tied_theta_E_plausible_bounds[1],
    norm_tied_w_plausible_bounds[1],
    norm_tied_t_E_aff_plausible_bounds[1],
    norm_tied_del_go_plausible_bounds[1],
    norm_tied_rate_norm_plausible_bounds[1]
])


# %%
### Read csv and get batch data###
exp_df = pd.read_csv('../outExp.csv')

if 'timed_fix' in exp_df.columns:
    exp_df.loc[:, 'RTwrtStim'] = exp_df['timed_fix'] - exp_df['intended_fix']
    exp_df = exp_df.rename(columns={'timed_fix': 'TotalFixTime'})

exp_df = exp_df[~((exp_df['RTwrtStim'].isna()) & (exp_df['abort_event'] == 3))].copy()

# In some cases, response_poke is nan, but success and ILD are present.
mask_nan = exp_df['response_poke'].isna()
mask_success_1 = (exp_df['success'] == 1)
mask_success_neg1 = (exp_df['success'] == -1)
mask_ild_pos = (exp_df['ILD'] > 0)
mask_ild_neg = (exp_df['ILD'] < 0)

exp_df.loc[mask_nan & mask_success_1 & mask_ild_pos, 'response_poke'] = 3
exp_df.loc[mask_nan & mask_success_1 & mask_ild_neg, 'response_poke'] = 2

exp_df.loc[mask_nan & mask_success_neg1 & mask_ild_pos, 'response_poke'] = 2
exp_df.loc[mask_nan & mask_success_neg1 & mask_ild_neg, 'response_poke'] = 3

exp_df_batch = exp_df[
    (exp_df['batch_name'] == batch_name) &
    (exp_df['LED_trial'].isin([np.nan, 0])) &
    (exp_df['animal'].isin(animal_to_fit)) &
    (exp_df['session_type'].isin([1, 7]))
].copy()

exp_df_batch['choice'] = exp_df_batch['response_poke'].apply(
    lambda x: 1 if x == 3 else (-1 if x == 2 else random.choice([1, -1]))
)
exp_df_batch['accuracy'] = (exp_df_batch['ILD'] * exp_df_batch['choice']).apply(lambda x: 1 if x > 0 else 0)


# %%
### DF - valid and aborts ###
df_valid_and_aborts = exp_df_batch[
    (exp_df_batch['success'].isin([1, -1])) |
    (exp_df_batch['abort_event'] == 3)
].copy()

df_aborts = df_valid_and_aborts[df_valid_and_aborts['abort_event'] == 3]

animal_ids = df_valid_and_aborts['animal'].unique()

print('####################################')
print(f'Aborts Truncation Time: {T_trunc}')
print('####################################')


# %%
for animal in animal_to_fit:
    if animal not in animal_ids:
        print(f"Animal {animal} not found in filtered {batch_name} data. Available animals: {animal_ids}")
        continue

    df_all_trials_animal = df_valid_and_aborts[df_valid_and_aborts['animal'] == animal]
    df_aborts_animal = df_aborts[df_aborts['animal'] == animal]

    print(f'Batch: {batch_name}, sample animal: {animal}')
    pdf_filename = f'results_{batch_name}_animal_{animal}_NORM_FROM_ABORTS.pdf'
    pdf = PdfPages(pdf_filename)

    fig_text = plt.figure(figsize=(8.5, 11))
    fig_text.clf()
    fig_text.text(0.1, 0.9, "Normalized TIED From Loaded Abort Parameters", fontsize=20, weight='bold')
    fig_text.text(0.1, 0.8, f"Batch Name: {batch_name}", fontsize=14)
    fig_text.text(0.1, 0.75, f"Animal ID: {animal}", fontsize=14)
    fig_text.text(0.1, 0.68, "Abort params loaded from existing result pickle", fontsize=12)
    fig_text.text(0.1, 0.63, f"T_trunc = {T_trunc:.3f}", fontsize=12)
    fig_text.gca().axis("off")
    pdf.savefig(fig_text, bbox_inches='tight')
    plt.close(fig_text)

    # %%
    ####################################################
    ########### Load Abort Parameters ##################
    ####################################################
    print("\n### Loading Abort Parameters from Pickle ###")

    pkl_file = f'results_{batch_name}_animal_{animal}.pkl'
    try:
        with open(pkl_file, 'rb') as f:
            fit_results_data = pickle.load(f)

        abort_keyname = "vbmc_aborts_results"
        abort_samples = fit_results_data[abort_keyname]

        V_A = np.mean(abort_samples['V_A_samples'])
        theta_A = np.mean(abort_samples['theta_A_samples'])
        t_A_aff = np.mean(abort_samples['t_A_aff_samp'])
        vbmc_aborts_results = abort_samples

        print("Loaded abort parameters:")
        print(f"  V_A = {V_A:.4f}")
        print(f"  theta_A = {theta_A:.4f}")
        print(f"  t_A_aff = {t_A_aff:.4f}")

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

    # %%
    ########################################################
    ########## Normalized model ############################
    ########################################################
    is_norm = True
    is_time_vary = False
    phi_params_obj = np.nan

    df_valid_animal = df_all_trials_animal[df_all_trials_animal['success'].isin([1, -1])]
    df_valid_animal_less_than_1 = df_valid_animal[df_valid_animal['RTwrtStim'] < 1]

    ABL_arr = np.sort(df_all_trials_animal['ABL'].unique())
    ILD_arr = np.sort(df_all_trials_animal['ILD'].unique())

    rate_lambda_0 = 2.3
    T_0_0 = 100 * 1e-3
    theta_E_0 = 3
    w_0 = 0.51
    t_E_aff_0 = 0.071
    del_go_0 = 0.19
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

    vbmc = VBMC(
        vbmc_norm_tied_joint_fn,
        x_0,
        norm_tied_lb,
        norm_tied_ub,
        norm_tied_plb,
        norm_tied_pub,
        options={'display': 'on', 'max_fun_evals': 200 * (2 + 7)}
    )
    vp, results = vbmc.optimize()
    vbmc.save(f'vbmc_PKL_file_norm_tied_results_batch_{batch_name}_animal_{animal}_FROM_ABORTS.pkl')

    vp_samples = vp.sample(int(1e5))[0]
    vp_samples[:, 1] *= 1e3
    param_labels = [
        r'$\lambda$',
        r'$T_0$ (ms)',
        r'$\theta_E$',
        r'$w$',
        r'$t_E^{aff}$',
        r'$\Delta_{go}$',
        r'rate_norm'
    ]
    percentiles = np.percentile(vp_samples, [1, 99], axis=0)
    _ranges = [(percentiles[0, i], percentiles[1, i]) for i in range(vp_samples.shape[1])]

    norm_tied_corner_fig = corner.corner(
        vp_samples,
        labels=param_labels,
        show_titles=True,
        quantiles=[0.025, 0.5, 0.975],
        range=_ranges,
        title_fmt=".3f"
    )
    norm_tied_corner_fig.suptitle(f'Normalized Tied Posterior (Animal: {animal})', y=1.02)
    vp_samples[:, 1] /= 1e3

    rate_lambda = vp_samples[:, 0].mean()
    T_0 = vp_samples[:, 1].mean()
    theta_E = vp_samples[:, 2].mean()
    w = vp_samples[:, 3].mean()
    Z_E = (w - 0.5) * 2 * theta_E
    t_E_aff = vp_samples[:, 4].mean()
    del_go = vp_samples[:, 5].mean()
    rate_norm_l = vp_samples[:, 6].mean()

    print("Posterior Means:")
    print(f"rate_lambda  = {rate_lambda:.5f}")
    print(f"T_0 (ms)      = {1e3*T_0:.5f}")
    print(f"theta_E       = {theta_E:.5f}")
    print(f"Z_E           = {Z_E:.5f}")
    print(f"t_E_aff       = {1e3*t_E_aff:.5f} ms")
    print(f"del_go        = {del_go:.5f}")
    print(f"rate_norm_l   = {rate_norm_l:.5f}")

    norm_tied_loglike = vbmc_norm_tied_loglike_fn([
        rate_lambda, T_0, theta_E, w, t_E_aff, del_go, rate_norm_l
    ])
    save_posterior_summary_page(
        pdf_pages=pdf,
        title=f'Normalized Tied Model - Posterior Means ({animal})',
        posterior_means=pd.Series({
            'rate_lambda': rate_lambda,
            'T_0': 1e3*T_0,
            'theta_E': theta_E,
            'w': w,
            'Z_E': Z_E,
            't_E_aff': 1e3*t_E_aff,
            'del_go': del_go,
            'rate_norm_l': rate_norm_l
        }),
        param_labels={
            'rate_lambda': r'$\lambda$',
            'T_0': r'$T_0$ (ms)',
            'theta_E': r'$\theta_E$',
            'w': r'$w$',
            'Z_E': r'$Z_E$',
            't_E_aff': r'$t_E^{aff}$',
            'del_go': r'$\Delta_{go}$',
            'rate_norm_l': r'rate_norm'
        },
        vbmc_results={
            'message': results['message'],
            'elbo': results['elbo'],
            'elbo_sd': results['elbo_sd'],
            'loglike': norm_tied_loglike,
            'convergence_status': results.get('convergence_status'),
            'r_index': results.get('r_index'),
            'success_flag': results.get('success_flag')
        },
        extra_text=f"T_trunc = {T_trunc:.3f}"
    )

    pdf.savefig(norm_tied_corner_fig, bbox_inches='tight')
    plt.close(norm_tied_corner_fig)

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

    # %%
    #######################################
    ##### norm tied diagnostics ###########
    #######################################
    print(f'Rate norm is {rate_norm_l}')

    t_stim_samples = df_valid_animal_less_than_1['intended_fix'].sample(N_sim, replace=True).values
    ABL_samples = df_valid_animal_less_than_1['ABL'].sample(N_sim, replace=True).values
    ILD_samples = df_valid_animal_less_than_1['ILD'].sample(N_sim, replace=True).values

    t_pts = np.arange(-1, 2, 0.001)
    P_A_mean, C_A_mean, t_stim_samples_for_diag = calculate_theoretical_curves(
        df_valid_and_aborts, N_theory, t_pts, t_A_aff, V_A, theta_A, rho_A_t_fn
    )

    sim_results = Parallel(n_jobs=30)(
        delayed(psiam_tied_data_gen_wrapper_rate_norm_fn)(
            V_A, theta_A, ABL_samples[iter_num], ILD_samples[iter_num],
            rate_lambda, T_0, theta_E, Z_E, t_A_aff, t_E_aff, del_go,
            t_stim_samples[iter_num], rate_norm_l, iter_num, N_print, dt
        )
        for iter_num in tqdm(range(N_sim))
    )
    sim_results_df = pd.DataFrame(sim_results)
    sim_results_df_valid = sim_results_df[
        (sim_results_df['rt'] > sim_results_df['t_stim']) &
        (sim_results_df['rt'] - sim_results_df['t_stim'] < 1)
    ].copy()

    sim_df_1, data_df_1 = prepare_simulation_data(sim_results_df_valid, df_valid_animal_less_than_1)

    theory_results_up_and_down, theory_time_axis, bins, bin_centers = plot_rt_distributions(
        sim_df_1, data_df_1, ILD_arr, ABL_arr, t_pts, P_A_mean, C_A_mean,
        t_stim_samples_for_diag, V_A, theta_A, t_A_aff,
        rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go,
        phi_params_obj, rate_norm_l, True, False, K_max, T_trunc,
        cum_pro_and_reactive_time_vary_fn, up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn,
        animal, pdf, model_name="Normalized Tied"
    )

    plot_tachometric_curves(
        sim_df_1, data_df_1, ILD_arr, ABL_arr, theory_results_up_and_down,
        theory_time_axis, bins, animal, pdf, model_name="Normalized Tied"
    )

    plot_grand_summary(
        sim_df_1, data_df_1, ILD_arr, ABL_arr, bins, bin_centers,
        animal, pdf, model_name="Normalized Tied"
    )

    # %%
    print("\nGenerating model comparison tables...")
    save_dict = {
        'vbmc_aborts_results': vbmc_aborts_results,
        'vbmc_norm_tied_results': vbmc_norm_tied_results
    }

    abort_df = create_abort_table(save_dict['vbmc_aborts_results'])
    if abort_df is not None:
        render_df_to_pdf(abort_df, f"Abort Model Results - Animal {animal}", pdf)
        print(f"Added abort model results table to PDF for animal {animal}")

    tied_df = create_tied_table(save_dict)
    if tied_df is not None:
        render_df_to_pdf(tied_df, f"Tied Models Comparison - Animal {animal}", pdf)
        print(f"Added normalized TIED results table to PDF for animal {animal}")

    pkl_filename = f'results_{batch_name}_animal_{animal}_NORM_FROM_ABORTS.pkl'
    with open(pkl_filename, 'wb') as f:
        pickle.dump(save_dict, f)

    print(f"Saved results to {pkl_filename}")

    pdf.close()
    print(f"Saved PDF report to {pdf_filename}")
