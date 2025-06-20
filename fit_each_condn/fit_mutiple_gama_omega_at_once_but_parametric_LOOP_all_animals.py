# %%
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



# %%
# repeat_trial, T16, S7
batch_name = 'LED7'
og_df = pd.read_csv('../out_LED.csv')
animal_ids = og_df['animal'].unique()
# remove animal 92
# for animal_id in animal_ids:
for animal_id in [103]:
    print(f'##### Starting animal {animal_id} #####')
    og_df = pd.read_csv('../out_LED.csv')
    df = og_df[ og_df['repeat_trial'].isin([0,2]) | og_df['repeat_trial'].isna() ]
    session_type = 7    
    df = df[ df['session_type'].isin([session_type]) ]
    training_level = 16
    df = df[ df['training_level'].isin([training_level]) ]


    # t_stim, t_LED, ABL, ILD
    t_stim_and_led_tuple = [(row['intended_fix'], row['intended_fix'] - row['LED_onset_time']) for _, row in df.iterrows()]
    ABL_arr = df['ABL'].unique(); ABL_arr.sort()
    ILD_arr = df['ILD'].unique(); ILD_arr.sort()


    # 1 is right , -1 is left
    df['choice'] = df['response_poke'].apply(lambda x: 1 if x == 3 else (-1 if x == 2 else random.choice([1, -1])))
    # 1 or 0 if the choice was correct or not
    df['correct'] = (df['ILD'] * df['choice']).apply(lambda x: 1 if x > 0 else 0)

    # %%
    # find the animal with largest number of trials and take that
    df = df[df['animal'] == int(animal_id)]

    # %% [markdown]
    # # data for vbmc

    # %%
    # LED OFF
    df_led_off = df[df['LED_trial'] == 0]
    print(f'len of LED off: {len(df_led_off)}')

    # valid trials
    df_led_off_valid_trials = df_led_off[df_led_off['success'].isin([1,-1])]
    print(f'len of led off valid trials = {len(df_led_off_valid_trials)}')

    # remove trials with RT > 1s
    df_led_off_valid_trials = df_led_off_valid_trials[df_led_off_valid_trials['timed_fix'] - df_led_off_valid_trials['intended_fix'] < 1]
    print(f'len of valid trials < 1s : {len(df_led_off_valid_trials)}')

    # %%
    # Filter the ABLs and ILDs 
    ABLs_to_fit = [20, 40, 60]
    ILDs_to_fit = [1,-1, 2, -2, 4, -4, 8, -8, 16, -16]

    df_led_off_valid_trials_cond_filtered = df_led_off_valid_trials[
        (df_led_off_valid_trials['ABL'].isin(ABLs_to_fit)) & 
        (df_led_off_valid_trials['ILD'].isin(ILDs_to_fit))
    ]
    print(f'len of filtered trials: {len(df_led_off_valid_trials_cond_filtered)}')



    # %%
    print(f'len of conditioned trials = {len(df_led_off_valid_trials_cond_filtered)}')
    ABLs_cond = df_led_off_valid_trials_cond_filtered['ABL'].unique()
    ILDs_cond = df_led_off_valid_trials_cond_filtered['ILD'].unique()
    print(ABLs_cond)
    print(ILDs_cond)


    # %%
    # Proactive params

    pkl_file = f'/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/results_{batch_name}_animal_{animal_id}.pkl'
    with open(pkl_file, 'rb') as f:
        fit_results_data = pickle.load(f)

    vbmc_aborts_param_keys_map = {
        'V_A_samples': 'V_A',
        'theta_A_samples': 'theta_A',
        't_A_aff_samp': 't_A_aff'
    }

    abort_keyname = "vbmc_aborts_results"
    if abort_keyname not in fit_results_data:
        raise Exception(f"No abort parameters found for batch {batch_name}, animal {animal_id}. Skipping.")
        
    abort_samples = fit_results_data[abort_keyname]
    abort_params = {}
    for param_samples_name, param_label in vbmc_aborts_param_keys_map.items():
        abort_params[param_label] = np.mean(abort_samples[param_samples_name])

    V_A = abort_params['V_A']
    theta_A = abort_params['theta_A']
    t_A_aff = abort_params['t_A_aff']


    # other params
    K_max = 10

    # %%
    # VBMC loglike
    def compute_loglike_trial(row, g_tanh_scale_20, g_ild_scale_20, g_ild_offset_20, o_ratio_scale_20, o_ild_scale_20, o_ild_offset_20, norm_factor_20,
        g_tanh_scale_40, g_ild_scale_40, g_ild_offset_40, o_ratio_scale_40, o_ild_scale_40, o_ild_offset_40, norm_factor_40,
        g_tanh_scale_60, g_ild_scale_60, g_ild_offset_60, o_ratio_scale_60, o_ild_scale_60, o_ild_offset_60, norm_factor_60,
        w_20, w_40, w_60, t_E_aff, del_go):
        # data
        c_A_trunc_time = 0.3
        rt = row['timed_fix']
        t_stim = row['intended_fix']
        response_poke = row['response_poke']
        
        ABL = row['ABL']
        ILD = row['ILD']

        if ABL == 20:
            gamma = g_tanh_scale_20 * np.tanh(g_ild_scale_20 * (ILD - g_ild_offset_20))
            omega = o_ratio_scale_20 * np.cosh(o_ild_scale_20 * (ILD - o_ild_offset_20)) / np.cosh(o_ild_scale_20 * norm_factor_20 * (ILD - o_ild_offset_20))
            w = w_20
        elif ABL == 40:
            gamma = g_tanh_scale_40 * np.tanh(g_ild_scale_40 * (ILD - g_ild_offset_40))
            omega = o_ratio_scale_40 * np.cosh(o_ild_scale_40 * (ILD - o_ild_offset_40)) / np.cosh(o_ild_scale_40 * norm_factor_40 * (ILD - o_ild_offset_40))
            w = w_40
        elif ABL == 60:
            gamma = g_tanh_scale_60 * np.tanh(g_ild_scale_60 * (ILD - g_ild_offset_60))
            omega = o_ratio_scale_60 * np.cosh(o_ild_scale_60 * (ILD - o_ild_offset_60)) / np.cosh(o_ild_scale_60 * norm_factor_60 * (ILD - o_ild_offset_60))
            w = w_60
        else:
            gamma = None
            omega = None
        
        if gamma is None or omega is None:
            raise ValueError(f"gamma or omega is None for ABL {ABL}, ILD {ILD}")
        
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


    def vbmc_loglike_fn(params):
        # gamma, omega, t_E_aff, w, del_go = params
        # gamma for each ABL = g_tanh_scale * tanh( g_ild_scale * (ILD - g_ild_offset) )
        # omega for each ABL = o_ratio_scale * cosh(o_ild_scale * (ILD - o_ild_offset)) / cosh(o_ild_scale * norm_factor * (ILD - o_ild_offset)) 
        (
            g_tanh_scale_20, g_ild_scale_20, g_ild_offset_20,
            o_ratio_scale_20, o_ild_scale_20, o_ild_offset_20, norm_factor_20,

            g_tanh_scale_40, g_ild_scale_40, g_ild_offset_40,
            o_ratio_scale_40, o_ild_scale_40, o_ild_offset_40, norm_factor_40,

            g_tanh_scale_60, g_ild_scale_60, g_ild_offset_60,
            o_ratio_scale_60, o_ild_scale_60, o_ild_offset_60, norm_factor_60,

            w_20, w_40, w_60, t_E_aff, del_go
        ) = params

        all_loglike = Parallel(n_jobs=30)(
            delayed(compute_loglike_trial)(
                row,
                g_tanh_scale_20, g_ild_scale_20, g_ild_offset_20,
                o_ratio_scale_20, o_ild_scale_20, o_ild_offset_20, norm_factor_20,
                g_tanh_scale_40, g_ild_scale_40, g_ild_offset_40,
                o_ratio_scale_40, o_ild_scale_40, o_ild_offset_40, norm_factor_40,
                g_tanh_scale_60, g_ild_scale_60, g_ild_offset_60,
                o_ratio_scale_60, o_ild_scale_60, o_ild_offset_60, norm_factor_60,
                w_20, w_40, w_60, t_E_aff, del_go
            )
            for _, row in df_led_off_valid_trials_cond_filtered.iterrows()
        )
        return np.sum(all_loglike)

    # %% [markdown]
    # # bounds

    # %%
    g_tanh_scale_bounds = [0.01, 6]
    g_tanh_scale_plausible_bounds = [0.5, 4]

    g_ild_scale_bounds = [0.001, 0.7]
    g_ild_scale_plausible_bounds = [0.1, 0.5]

    g_ild_offset_bounds = [-5, 5]
    g_ild_offset_plausible_bounds = [-3, 3]

    o_ratio_scale_bounds = [0.1, 7]
    o_ratio_scale_plausible_bounds = [0.5, 6]

    o_ild_scale_bounds = [0.01, 0.6]
    o_ild_scale_plausible_bounds = [0.05, 0.5]

    o_ild_offset_bounds = [-6, 6]
    o_ild_offset_plausible_bounds = [-1, 1]

    norm_factor_bounds = [0.3, 1.6]
    norm_factor_plausible_bounds = [0.75, 1]

    w_bounds = [0.2, 0.8]
    w_plausible_bounds = [0.3, 0.7]

    t_E_aff_bounds = [0.01, 0.12]
    t_E_aff_plausible_bounds = [0.06, 0.1]

    del_go_bounds = [0.001, 0.2]
    del_go_plausible_bounds = [0.11, 0.15]

    g_tanh_scale_20_bounds = g_tanh_scale_40_bounds = g_tanh_scale_60_bounds = g_tanh_scale_bounds
    g_tanh_scale_20_plausible_bounds = g_tanh_scale_40_plausible_bounds = g_tanh_scale_60_plausible_bounds = g_tanh_scale_plausible_bounds

    g_ild_scale_20_bounds = g_ild_scale_40_bounds = g_ild_scale_60_bounds = g_ild_scale_bounds
    g_ild_scale_20_plausible_bounds = g_ild_scale_40_plausible_bounds = g_ild_scale_60_plausible_bounds = g_ild_scale_plausible_bounds

    g_ild_offset_20_bounds = g_ild_offset_40_bounds = g_ild_offset_60_bounds = g_ild_offset_bounds
    g_ild_offset_20_plausible_bounds = g_ild_offset_40_plausible_bounds = g_ild_offset_60_plausible_bounds = g_ild_offset_plausible_bounds

    o_ratio_scale_20_bounds = o_ratio_scale_40_bounds = o_ratio_scale_60_bounds = o_ratio_scale_bounds
    o_ratio_scale_20_plausible_bounds = o_ratio_scale_40_plausible_bounds = o_ratio_scale_60_plausible_bounds = o_ratio_scale_plausible_bounds

    o_ild_scale_20_bounds = o_ild_scale_40_bounds = o_ild_scale_60_bounds = o_ild_scale_bounds
    o_ild_scale_20_plausible_bounds = o_ild_scale_40_plausible_bounds = o_ild_scale_60_plausible_bounds = o_ild_scale_plausible_bounds

    o_ild_offset_20_bounds = o_ild_offset_40_bounds = o_ild_offset_60_bounds = o_ild_offset_bounds
    o_ild_offset_20_plausible_bounds = o_ild_offset_40_plausible_bounds = o_ild_offset_60_plausible_bounds = o_ild_offset_plausible_bounds

    norm_factor_20_bounds = norm_factor_40_bounds = norm_factor_60_bounds = norm_factor_bounds
    norm_factor_20_plausible_bounds = norm_factor_40_plausible_bounds = norm_factor_60_plausible_bounds = norm_factor_plausible_bounds

    w_20_bounds = w_40_bounds = w_60_bounds = w_bounds
    w_20_plausible_bounds = w_40_plausible_bounds = w_60_plausible_bounds = w_plausible_bounds

    # %% [markdown]
    # # prior

    # %%
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
        (
            g_tanh_scale_20, g_ild_scale_20, g_ild_offset_20, o_ratio_scale_20, o_ild_scale_20, o_ild_offset_20, norm_factor_20,
            g_tanh_scale_40, g_ild_scale_40, g_ild_offset_40, o_ratio_scale_40, o_ild_scale_40, o_ild_offset_40, norm_factor_40,
            g_tanh_scale_60, g_ild_scale_60, g_ild_offset_60, o_ratio_scale_60, o_ild_scale_60, o_ild_offset_60, norm_factor_60,
            w_20, w_40, w_60, t_E_aff, del_go
        ) = params

        logpdf = 0

        # 20
        logpdf += trapezoidal_logpdf(g_tanh_scale_20, g_tanh_scale_20_bounds[0], g_tanh_scale_20_plausible_bounds[0], g_tanh_scale_20_plausible_bounds[1], g_tanh_scale_20_bounds[1])
        logpdf += trapezoidal_logpdf(g_ild_scale_20, g_ild_scale_20_bounds[0], g_ild_scale_20_plausible_bounds[0], g_ild_scale_20_plausible_bounds[1], g_ild_scale_20_bounds[1])
        logpdf += trapezoidal_logpdf(g_ild_offset_20, g_ild_offset_20_bounds[0], g_ild_offset_20_plausible_bounds[0], g_ild_offset_20_plausible_bounds[1], g_ild_offset_20_bounds[1])
        logpdf += trapezoidal_logpdf(o_ratio_scale_20, o_ratio_scale_20_bounds[0], o_ratio_scale_20_plausible_bounds[0], o_ratio_scale_20_plausible_bounds[1], o_ratio_scale_20_bounds[1])
        logpdf += trapezoidal_logpdf(o_ild_scale_20, o_ild_scale_20_bounds[0], o_ild_scale_20_plausible_bounds[0], o_ild_scale_20_plausible_bounds[1], o_ild_scale_20_bounds[1])
        logpdf += trapezoidal_logpdf(o_ild_offset_20, o_ild_offset_20_bounds[0], o_ild_offset_20_plausible_bounds[0], o_ild_offset_20_plausible_bounds[1], o_ild_offset_20_bounds[1])
        logpdf += trapezoidal_logpdf(norm_factor_20, norm_factor_20_bounds[0], norm_factor_20_plausible_bounds[0], norm_factor_20_plausible_bounds[1], norm_factor_20_bounds[1])

        # 40
        logpdf += trapezoidal_logpdf(g_tanh_scale_40, g_tanh_scale_40_bounds[0], g_tanh_scale_40_plausible_bounds[0], g_tanh_scale_40_plausible_bounds[1], g_tanh_scale_40_bounds[1])
        logpdf += trapezoidal_logpdf(g_ild_scale_40, g_ild_scale_40_bounds[0], g_ild_scale_40_plausible_bounds[0], g_ild_scale_40_plausible_bounds[1], g_ild_scale_40_bounds[1])
        logpdf += trapezoidal_logpdf(g_ild_offset_40, g_ild_offset_40_bounds[0], g_ild_offset_40_plausible_bounds[0], g_ild_offset_40_plausible_bounds[1], g_ild_offset_40_bounds[1])
        logpdf += trapezoidal_logpdf(o_ratio_scale_40, o_ratio_scale_40_bounds[0], o_ratio_scale_40_plausible_bounds[0], o_ratio_scale_40_plausible_bounds[1], o_ratio_scale_40_bounds[1])
        logpdf += trapezoidal_logpdf(o_ild_scale_40, o_ild_scale_40_bounds[0], o_ild_scale_40_plausible_bounds[0], o_ild_scale_40_plausible_bounds[1], o_ild_scale_40_bounds[1])
        logpdf += trapezoidal_logpdf(o_ild_offset_40, o_ild_offset_40_bounds[0], o_ild_offset_40_plausible_bounds[0], o_ild_offset_40_plausible_bounds[1], o_ild_offset_40_bounds[1])
        logpdf += trapezoidal_logpdf(norm_factor_40, norm_factor_40_bounds[0], norm_factor_40_plausible_bounds[0], norm_factor_40_plausible_bounds[1], norm_factor_40_bounds[1])

        # 60
        logpdf += trapezoidal_logpdf(g_tanh_scale_60, g_tanh_scale_60_bounds[0], g_tanh_scale_60_plausible_bounds[0], g_tanh_scale_60_plausible_bounds[1], g_tanh_scale_60_bounds[1])
        logpdf += trapezoidal_logpdf(g_ild_scale_60, g_ild_scale_60_bounds[0], g_ild_scale_60_plausible_bounds[0], g_ild_scale_60_plausible_bounds[1], g_ild_scale_60_bounds[1])
        logpdf += trapezoidal_logpdf(g_ild_offset_60, g_ild_offset_60_bounds[0], g_ild_offset_60_plausible_bounds[0], g_ild_offset_60_plausible_bounds[1], g_ild_offset_60_bounds[1])
        logpdf += trapezoidal_logpdf(o_ratio_scale_60, o_ratio_scale_60_bounds[0], o_ratio_scale_60_plausible_bounds[0], o_ratio_scale_60_plausible_bounds[1], o_ratio_scale_60_bounds[1])
        logpdf += trapezoidal_logpdf(o_ild_scale_60, o_ild_scale_60_bounds[0], o_ild_scale_60_plausible_bounds[0], o_ild_scale_60_plausible_bounds[1], o_ild_scale_60_bounds[1])
        logpdf += trapezoidal_logpdf(o_ild_offset_60, o_ild_offset_60_bounds[0], o_ild_offset_60_plausible_bounds[0], o_ild_offset_60_plausible_bounds[1], o_ild_offset_60_bounds[1])
        logpdf += trapezoidal_logpdf(norm_factor_60, norm_factor_60_bounds[0], norm_factor_60_plausible_bounds[0], norm_factor_60_plausible_bounds[1], norm_factor_60_bounds[1])

        # Shared params
        logpdf += trapezoidal_logpdf(w_20, w_20_bounds[0], w_20_plausible_bounds[0], w_20_plausible_bounds[1], w_20_bounds[1])
        logpdf += trapezoidal_logpdf(w_40, w_40_bounds[0], w_40_plausible_bounds[0], w_40_plausible_bounds[1], w_40_bounds[1])
        logpdf += trapezoidal_logpdf(w_60, w_60_bounds[0], w_60_plausible_bounds[0], w_60_plausible_bounds[1], w_60_bounds[1])
        logpdf += trapezoidal_logpdf(t_E_aff, t_E_aff_bounds[0], t_E_aff_plausible_bounds[0], t_E_aff_plausible_bounds[1], t_E_aff_bounds[1])
        logpdf += trapezoidal_logpdf(del_go, del_go_bounds[0], del_go_plausible_bounds[0], del_go_plausible_bounds[1], del_go_bounds[1])

        return logpdf
    # %% [markdown]
    # # prior + loglike

    # %%
    def vbmc_joint_fn(params):
        priors = vbmc_prior_fn(params)
        loglike = vbmc_loglike_fn(params)

        return priors + loglike

    # %% [markdown]
    # # vbmc

    # %%
    lb = np.array([
        g_tanh_scale_20_bounds[0], g_ild_scale_20_bounds[0], g_ild_offset_20_bounds[0], o_ratio_scale_20_bounds[0], o_ild_scale_20_bounds[0], o_ild_offset_20_bounds[0], norm_factor_20_bounds[0],
        g_tanh_scale_40_bounds[0], g_ild_scale_40_bounds[0], g_ild_offset_40_bounds[0], o_ratio_scale_40_bounds[0], o_ild_scale_40_bounds[0], o_ild_offset_40_bounds[0], norm_factor_40_bounds[0],
        g_tanh_scale_60_bounds[0], g_ild_scale_60_bounds[0], g_ild_offset_60_bounds[0], o_ratio_scale_60_bounds[0], o_ild_scale_60_bounds[0], o_ild_offset_60_bounds[0], norm_factor_60_bounds[0],
        w_20_bounds[0], w_40_bounds[0], w_60_bounds[0], t_E_aff_bounds[0], del_go_bounds[0]
    ])
    ub = np.array([
        g_tanh_scale_20_bounds[1], g_ild_scale_20_bounds[1], g_ild_offset_20_bounds[1], o_ratio_scale_20_bounds[1], o_ild_scale_20_bounds[1], o_ild_offset_20_bounds[1], norm_factor_20_bounds[1],
        g_tanh_scale_40_bounds[1], g_ild_scale_40_bounds[1], g_ild_offset_40_bounds[1], o_ratio_scale_40_bounds[1], o_ild_scale_40_bounds[1], o_ild_offset_40_bounds[1], norm_factor_40_bounds[1],
        g_tanh_scale_60_bounds[1], g_ild_scale_60_bounds[1], g_ild_offset_60_bounds[1], o_ratio_scale_60_bounds[1], o_ild_scale_60_bounds[1], o_ild_offset_60_bounds[1], norm_factor_60_bounds[1],
        w_20_bounds[1], w_40_bounds[1], w_60_bounds[1], t_E_aff_bounds[1], del_go_bounds[1]
    ])
    plb = np.array([
        g_tanh_scale_20_plausible_bounds[0], g_ild_scale_20_plausible_bounds[0], g_ild_offset_20_plausible_bounds[0], o_ratio_scale_20_plausible_bounds[0], o_ild_scale_20_plausible_bounds[0], o_ild_offset_20_plausible_bounds[0], norm_factor_20_plausible_bounds[0],
        g_tanh_scale_40_plausible_bounds[0], g_ild_scale_40_plausible_bounds[0], g_ild_offset_40_plausible_bounds[0], o_ratio_scale_40_plausible_bounds[0], o_ild_scale_40_plausible_bounds[0], o_ild_offset_40_plausible_bounds[0], norm_factor_40_plausible_bounds[0],
        g_tanh_scale_60_plausible_bounds[0], g_ild_scale_60_plausible_bounds[0], g_ild_offset_60_plausible_bounds[0], o_ratio_scale_60_plausible_bounds[0], o_ild_scale_60_plausible_bounds[0], o_ild_offset_60_plausible_bounds[0], norm_factor_60_plausible_bounds[0],
        w_20_plausible_bounds[0], w_40_plausible_bounds[0], w_60_plausible_bounds[0], t_E_aff_plausible_bounds[0], del_go_plausible_bounds[0]
    ])
    pub = np.array([
        g_tanh_scale_20_plausible_bounds[1], g_ild_scale_20_plausible_bounds[1], g_ild_offset_20_plausible_bounds[1], o_ratio_scale_20_plausible_bounds[1], o_ild_scale_20_plausible_bounds[1], o_ild_offset_20_plausible_bounds[1], norm_factor_20_plausible_bounds[1],
        g_tanh_scale_40_plausible_bounds[1], g_ild_scale_40_plausible_bounds[1], g_ild_offset_40_plausible_bounds[1], o_ratio_scale_40_plausible_bounds[1], o_ild_scale_40_plausible_bounds[1], o_ild_offset_40_plausible_bounds[1], norm_factor_40_plausible_bounds[1],
        g_tanh_scale_60_plausible_bounds[1], g_ild_scale_60_plausible_bounds[1], g_ild_offset_60_plausible_bounds[1], o_ratio_scale_60_plausible_bounds[1], o_ild_scale_60_plausible_bounds[1], o_ild_offset_60_plausible_bounds[1], norm_factor_60_plausible_bounds[1],
        w_20_plausible_bounds[1], w_40_plausible_bounds[1], w_60_plausible_bounds[1], t_E_aff_plausible_bounds[1], del_go_plausible_bounds[1]
    ])

    # Initialize with random values within plausible bounds
    np.random.seed(42)
    g_tanh_scale_20_0 = np.random.uniform(g_tanh_scale_20_plausible_bounds[0], g_tanh_scale_20_plausible_bounds[1])
    g_ild_scale_20_0 = np.random.uniform(g_ild_scale_20_plausible_bounds[0], g_ild_scale_20_plausible_bounds[1])
    g_ild_offset_20_0 = np.random.uniform(g_ild_offset_20_plausible_bounds[0], g_ild_offset_20_plausible_bounds[1])
    o_ratio_scale_20_0 = np.random.uniform(o_ratio_scale_20_plausible_bounds[0], o_ratio_scale_20_plausible_bounds[1])
    o_ild_scale_20_0 = np.random.uniform(o_ild_scale_20_plausible_bounds[0], o_ild_scale_20_plausible_bounds[1])
    o_ild_offset_20_0 = np.random.uniform(o_ild_offset_20_plausible_bounds[0], o_ild_offset_20_plausible_bounds[1])
    norm_factor_20_0 = np.random.uniform(norm_factor_20_plausible_bounds[0], norm_factor_20_plausible_bounds[1])

    g_tanh_scale_40_0 = np.random.uniform(g_tanh_scale_40_plausible_bounds[0], g_tanh_scale_40_plausible_bounds[1])
    g_ild_scale_40_0 = np.random.uniform(g_ild_scale_40_plausible_bounds[0], g_ild_scale_40_plausible_bounds[1])
    g_ild_offset_40_0 = np.random.uniform(g_ild_offset_40_plausible_bounds[0], g_ild_offset_40_plausible_bounds[1])
    o_ratio_scale_40_0 = np.random.uniform(o_ratio_scale_40_plausible_bounds[0], o_ratio_scale_40_plausible_bounds[1])
    o_ild_scale_40_0 = np.random.uniform(o_ild_scale_40_plausible_bounds[0], o_ild_scale_40_plausible_bounds[1])
    o_ild_offset_40_0 = np.random.uniform(o_ild_offset_40_plausible_bounds[0], o_ild_offset_40_plausible_bounds[1])
    norm_factor_40_0 = np.random.uniform(norm_factor_40_plausible_bounds[0], norm_factor_40_plausible_bounds[1])

    g_tanh_scale_60_0 = np.random.uniform(g_tanh_scale_60_plausible_bounds[0], g_tanh_scale_60_plausible_bounds[1])
    g_ild_scale_60_0 = np.random.uniform(g_ild_scale_60_plausible_bounds[0], g_ild_scale_60_plausible_bounds[1])
    g_ild_offset_60_0 = np.random.uniform(g_ild_offset_60_plausible_bounds[0], g_ild_offset_60_plausible_bounds[1])
    o_ratio_scale_60_0 = np.random.uniform(o_ratio_scale_60_plausible_bounds[0], o_ratio_scale_60_plausible_bounds[1])
    o_ild_scale_60_0 = np.random.uniform(o_ild_scale_60_plausible_bounds[0], o_ild_scale_60_plausible_bounds[1])
    o_ild_offset_60_0 = np.random.uniform(o_ild_offset_60_plausible_bounds[0], o_ild_offset_60_plausible_bounds[1])
    norm_factor_60_0 = np.random.uniform(norm_factor_60_plausible_bounds[0], norm_factor_60_plausible_bounds[1])

    w_20_0 = np.random.uniform(w_20_plausible_bounds[0], w_20_plausible_bounds[1])
    w_40_0 = np.random.uniform(w_40_plausible_bounds[0], w_40_plausible_bounds[1])
    w_60_0 = np.random.uniform(w_60_plausible_bounds[0], w_60_plausible_bounds[1])
    t_E_aff_0 = np.random.uniform(t_E_aff_plausible_bounds[0], t_E_aff_plausible_bounds[1])
    del_go_0 = np.random.uniform(del_go_plausible_bounds[0], del_go_plausible_bounds[1])

    x_0 = np.array([
        g_tanh_scale_20_0, g_ild_scale_20_0, g_ild_offset_20_0, o_ratio_scale_20_0, o_ild_scale_20_0, o_ild_offset_20_0, norm_factor_20_0,
        g_tanh_scale_40_0, g_ild_scale_40_0, g_ild_offset_40_0, o_ratio_scale_40_0, o_ild_scale_40_0, o_ild_offset_40_0, norm_factor_40_0,
        g_tanh_scale_60_0, g_ild_scale_60_0, g_ild_offset_60_0, o_ratio_scale_60_0, o_ild_scale_60_0, o_ild_offset_60_0, norm_factor_60_0,
        w_20_0, w_40_0, w_60_0, t_E_aff_0, del_go_0
    ])
    # Run VBMC
    vbmc = VBMC(vbmc_joint_fn, x_0, lb, ub, plb, pub, options={'display': 'on'})
    vp, results = vbmc.optimize()

    # %%
    vbmc.save(f'vbmc_mutiple_gama_omega_at_once_but_parametric_batch_{batch_name}_animal_{animal_id}_BETTER_BOUNDS_V2.pkl', overwrite=True)
