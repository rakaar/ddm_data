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
from vbmc_animal_wise_fit_utils import trapezoidal_logpdf
from time_vary_norm_utils import up_or_down_RTs_fit_fn, cum_pro_and_reactive_time_vary_fn
# %%
T_trunc = 0.3
batch_name = 'SD'
phi_params_obj = np.nan
rate_norm_l = np.nan
is_norm = False
is_time_vary = False
K_max = 10

# load animal data
exp_df = pd.read_csv('../outExp.csv')
# out_LED.csv
# exp_df = pd.read_csv('../out_LED.csv')
if 'timed_fix' in exp_df.columns:
    exp_df.loc[:, 'RTwrtStim'] = exp_df['timed_fix'] - exp_df['intended_fix']
    exp_df = exp_df.rename(columns={'timed_fix': 'TotalFixTime'})
# 
# remove rows where abort happened, and RT is nan
exp_df = exp_df[~((exp_df['RTwrtStim'].isna()) & (exp_df['abort_event'] == 3))].copy()

# in some cases, response_poke is nan, but succcess and ILD are present
# reconstruct response_poke from success and ILD 
# if success and ILD are present, a response should also be present
mask_nan = exp_df['response_poke'].isna()
mask_success_1 = (exp_df['success'] == 1)
mask_success_neg1 = (exp_df['success'] == -1)
mask_ild_pos = (exp_df['ILD'] > 0)
mask_ild_neg = (exp_df['ILD'] < 0)

# For success == 1
exp_df.loc[mask_nan & mask_success_1 & mask_ild_pos, 'response_poke'] = 3
exp_df.loc[mask_nan & mask_success_1 & mask_ild_neg, 'response_poke'] = 2

# For success == -1
exp_df.loc[mask_nan & mask_success_neg1 & mask_ild_pos, 'response_poke'] = 2
exp_df.loc[mask_nan & mask_success_neg1 & mask_ild_neg, 'response_poke'] = 3

exp_df_batch = exp_df[
    (exp_df['batch_name'] == batch_name) &
    (exp_df['LED_trial'].isin([np.nan, 0])) &
    (exp_df['animal'].isin([49])) &
    (exp_df['session_type'].isin([1,7]))  
].copy()

# aborts don't have choice, so assign random 
exp_df_batch['choice'] = exp_df_batch['response_poke'].apply(lambda x: 1 if x == 3 else (-1 if x == 2 else random.choice([1, -1])))
# 1 or 0 if the choice was correct or not
exp_df_batch['accuracy'] = (exp_df_batch['ILD'] * exp_df_batch['choice']).apply(lambda x: 1 if x > 0 else 0)


### DF - valid and aborts ###
df_valid_and_aborts = exp_df_batch[
    (exp_df_batch['success'].isin([1,-1])) |
    (exp_df_batch['abort_event'] == 3)
].copy()

df_aborts = df_valid_and_aborts[df_valid_and_aborts['abort_event'] == 3]

### Animal selection ###
animal_ids = df_valid_and_aborts['animal'].unique()
# animal = animal_ids[-1]
# for animal_idx in [-1]:

print('####################################')
print(f'Aborts Truncation Time: {T_trunc}')
print('####################################')


# load proactive params

# %%
# VBMC helper funcs
def compute_loglike_vanilla(row, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go, lapse_prob, lapse_rt_window):
    
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

    included_lapse_pdf = (1 - lapse_prob) * pdf  + lapse_prob * (0.5 * (1/lapse_rt_window))
    included_lapse_pdf = max(included_lapse_pdf, 1e-50)
    if np.isnan(included_lapse_pdf):
        print(f'row["abort_event"] = {row["abort_event"]}')
        print(f'row["RTwrtStim"] = {row["RTwrtStim"]}')
        raise ValueError(f'nan pdf rt = {rt}, t_stim = {t_stim}')

    
    return np.log(included_lapse_pdf)


def vbmc_vanilla_tied_loglike_fn(params):
    rate_lambda, T_0, theta_E, w, t_E_aff, del_go, lapse_prob, lapse_rt_window = params
    Z_E = (w - 0.5) * 2 * theta_E
    all_loglike = Parallel(n_jobs=30)(delayed(compute_loglike_vanilla)(row, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go, lapse_prob, lapse_rt_window)\
                                       for _, row in df_valid_animal_less_than_1.iterrows() )
    return np.sum(all_loglike)


def vbmc_vanilla_tied_prior_fn(params):
    rate_lambda, T_0, theta_E, w, t_E_aff, del_go, lapse_prob, lapse_rt_window = params

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
    
    return (
        rate_lambda_logpdf +
        T_0_logpdf +
        theta_E_logpdf +
        w_logpdf +
        t_E_aff_logpdf +
        del_go_logpdf +
        lapse_prob_logpdf
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
vanilla_lapse_prob_bounds = [0, 0.3]
vanilla_lapse_rt_window_bounds = [0.0001, 1]

vanilla_rate_lambda_plausible_bounds = [0.1, 0.3]
vanilla_T_0_plausible_bounds = [0.5e-3, 1.5e-3]
vanilla_theta_E_plausible_bounds = [15, 55]
vanilla_w_plausible_bounds = [0.4, 0.6]
vanilla_t_E_aff_plausible_bounds = [0.03, 0.09]
vanilla_del_go_plausible_bounds = [0.05, 0.15]
vanilla_lapse_prob_plausible_bounds = [0.01, 0.1]
vanilla_lapse_rt_window_plausible_bounds = [0.005, 0.9]


vanilla_tied_lb = np.array([
    vanilla_rate_lambda_bounds[0],
    vanilla_T_0_bounds[0],
    vanilla_theta_E_bounds[0],
    vanilla_w_bounds[0],
    vanilla_t_E_aff_bounds[0],
    vanilla_del_go_bounds[0],
    vanilla_lapse_prob_bounds[0],
    vanilla_lapse_rt_window_bounds[0]
])

vanilla_tied_ub = np.array([
    vanilla_rate_lambda_bounds[1],
    vanilla_T_0_bounds[1],
    vanilla_theta_E_bounds[1],
    vanilla_w_bounds[1],
    vanilla_t_E_aff_bounds[1],
    vanilla_del_go_bounds[1],
    vanilla_lapse_prob_bounds[1],
    vanilla_lapse_rt_window_bounds[1]
])

vanilla_plb = np.array([
    vanilla_rate_lambda_plausible_bounds[0],
    vanilla_T_0_plausible_bounds[0],
    vanilla_theta_E_plausible_bounds[0],
    vanilla_w_plausible_bounds[0],
    vanilla_t_E_aff_plausible_bounds[0],
    vanilla_del_go_plausible_bounds[0],
    vanilla_lapse_prob_plausible_bounds[0],
    vanilla_lapse_rt_window_plausible_bounds[0]
])

vanilla_pub = np.array([
    vanilla_rate_lambda_plausible_bounds[1],
    vanilla_T_0_plausible_bounds[1],
    vanilla_theta_E_plausible_bounds[1],
    vanilla_w_plausible_bounds[1],
    vanilla_t_E_aff_plausible_bounds[1],
    vanilla_del_go_plausible_bounds[1],
    vanilla_lapse_prob_plausible_bounds[1],
    vanilla_lapse_rt_window_plausible_bounds[1]
])

# %%
for animal_idx in [0]:
    animal = animal_ids[animal_idx]

    df_all_trials_animal = df_valid_and_aborts[df_valid_and_aborts['animal'] == animal]
    df_aborts_animal = df_aborts[df_aborts['animal'] == animal]

    df_valid_animal = df_all_trials_animal[df_all_trials_animal['success'].isin([1,-1])]
    df_valid_animal_less_than_1 = df_valid_animal[df_valid_animal['RTwrtStim'] < 1]


    print(f'Batch: {batch_name},sample animal: {animal}')
    pdf_filename = f'results_{batch_name}_animal_{animal}_lapse_fit.pdf'
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
    lapse_rt_window_0 = 0.9

    x_0 = np.array([
        rate_lambda_0,
        T_0_0,
        theta_E_0,
        w_0,
        t_E_aff_0,
        del_go_0,
        lapse_prob_0,
        lapse_rt_window_0
    ])
    
    vbmc = VBMC(vbmc_vanilla_tied_joint_fn, x_0, vanilla_tied_lb, vanilla_tied_ub, vanilla_plb, vanilla_pub, options={'display': 'on', 'max_fun_evals': 200 * (2 + 6)})
    vp, results = vbmc.optimize()

    vbmc.save(f'vbmc_vanilla_tied_results_batch_{batch_name}_animal_{animal}_lapses.pkl', overwrite=True)

# %%

