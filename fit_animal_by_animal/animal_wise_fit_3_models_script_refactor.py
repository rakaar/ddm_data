# %%
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend which doesn't require tkinter
import matplotlib.pyplot as plt
import pandas as pd
# Import model table creation functions from utils
from animal_wise_plotting_utils import render_df_to_pdf, create_abort_table, create_tied_table
from joblib import Parallel, delayed
from tqdm.notebook import tqdm
from scipy.integrate import quad
import pandas as pd
import pickle
from types import SimpleNamespace
from pyvbmc import VBMC
import random
import os
from time_vary_norm_utils import (
    up_or_down_RTs_fit_fn, cum_pro_and_reactive_time_vary_fn,
    rho_A_t_VEC_fn, up_or_down_RTs_fit_wrt_stim_fn, rho_A_t_fn, cum_A_t_fn)
import corner
from scipy.integrate import trapezoid as trapz
from time_vary_and_norm_simulators import (
    psiam_tied_data_gen_wrapper_rate_norm_fn, 
    psiam_tied_data_gen_wrapper_rate_norm_time_vary_fn
)
from scipy.integrate import cumulative_trapezoid as cumtrapz
from time_vary_norm_utils import up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn

from vbmc_animal_wise_fit_utils import trapezoidal_logpdf
from animal_wise_config import T_trunc
from animal_wise_plotting_utils import save_posterior_summary_page, save_corner_plot, plot_abort_diagnostic

from matplotlib.backends.backend_pdf import PdfPages
from animal_wise_plotting_utils import prepare_simulation_data, calculate_theoretical_curves, plot_rt_distributions, plot_tachometric_curves, plot_grand_summary


############3 Params #############
batch_name = 'Comparable'
K_max = 10

N_theory = int(1e3)
N_sim = int(1e6)
dt  = 1e-3
N_print = int(N_sim / 5)


####### Aborts ##############
def compute_loglike_aborts(row, V_A, theta_A, t_A_aff, pdf_trunc_factor):
    t_stim = row['intended_fix']
    rt = row['TotalFixTime']

    if rt < T_trunc:
        likelihood = 0
    else:
        if rt < t_stim:
            likelihood =  rho_A_t_fn(rt - t_A_aff, V_A, theta_A) / pdf_trunc_factor
        elif rt > t_stim:
            if t_stim <= T_trunc:
                likelihood = 1
            else:
                likelihood = ( 1 - cum_A_t_fn(t_stim - t_A_aff, V_A, theta_A) ) / pdf_trunc_factor

    if likelihood <= 0:
        likelihood = 1e-50

    if np.isnan(likelihood):
        print(f'likelihood is nan for rt={rt}, t_stim={t_stim}, t_A_aff={t_A_aff}, pdf_trunc_factor={pdf_trunc_factor}')
        print(f'rho_A_t_fn(rt - t_A_aff, V_A, theta_A) = {rho_A_t_fn(rt - t_A_aff, V_A, theta_A)}')
        print(f'pdf_trunc_factor = {pdf_trunc_factor}')
        print(f'cum_A_t_fn(t_stim - t_A_aff, V_A, theta_A) = {cum_A_t_fn(t_stim - t_A_aff, V_A, theta_A)}')
        raise ValueError('likelihood is nan')

    
    return np.log(likelihood)    


## loglike
def vbmc_aborts_loglike_fn(params):
    V_A, theta_A, t_A_aff = params

    pdf_trunc_factor = 1 - cum_A_t_fn(T_trunc - t_A_aff, V_A, theta_A)
    all_loglike = Parallel(n_jobs=-1)(delayed(compute_loglike_aborts)(row, V_A, theta_A, t_A_aff, pdf_trunc_factor) \
                                       for _, row in df_all_trials_animal.iterrows()  )
                                   


    loglike = np.sum(all_loglike)
    return loglike

## prior
def vbmc_prior_abort_fn(params):
    V_A, theta_A,t_A_aff = params

    V_A_logpdf = trapezoidal_logpdf(V_A, V_A_bounds[0], V_A_plausible_bounds[0], V_A_plausible_bounds[1], V_A_bounds[1])
    theta_A_logpdf = trapezoidal_logpdf(theta_A, theta_A_bounds[0], theta_A_plausible_bounds[0], theta_A_plausible_bounds[1], theta_A_bounds[1])
    t_A_aff_logpdf = trapezoidal_logpdf(t_A_aff, t_A_aff_bounds[0], t_A_aff_plausible_bounds[0], t_A_aff_plausible_bounds[1], t_A_aff_bounds[1])
    return V_A_logpdf + theta_A_logpdf + t_A_aff_logpdf

## joint
def vbmc_joint_aborts_fn(params):
    return vbmc_prior_abort_fn(params) + vbmc_aborts_loglike_fn(params)

######## BOUNDS ########
V_A_bounds = [0.1, 10]
theta_A_bounds = [0.1, 10]
t_A_aff_bounds = [-5, 0.1]

V_A_plausible_bounds = [0.5, 3]
theta_A_plausible_bounds = [0.5, 3]
t_A_aff_plausible_bounds = [-2, 0.06]

## bounds array
aborts_lb = [V_A_bounds[0], theta_A_bounds[0], t_A_aff_bounds[0]]
aborts_ub = [V_A_bounds[1], theta_A_bounds[1], t_A_aff_bounds[1]]

aborts_plb = [V_A_plausible_bounds[0], theta_A_plausible_bounds[0], t_A_aff_plausible_bounds[0]]
aborts_pub = [V_A_plausible_bounds[1], theta_A_plausible_bounds[1], t_A_aff_plausible_bounds[1]]

############ End of aborts #######################

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
                                       for _, row in df_valid_animal_less_than_1.iterrows() )
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

###### End of vanilla tied ###########

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
    

def vbmc_norm_tied_loglike_fn(params):
    rate_lambda, T_0, theta_E, w, t_E_aff, del_go, rate_norm_l = params
    Z_E = (w - 0.5) * 2 * theta_E
    all_loglike = Parallel(n_jobs=30)(delayed(compute_loglike_norm_fn)(row, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go, rate_norm_l)\
                                       for _, row in df_valid_animal_less_than_1.iterrows() )
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


# Add bounds for all parameters (order: V_A, theta_A, t_A_aff, rate_lambda, T_0, theta_E, w, t_E_aff, del_go)
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


######### End of Normalized TIED #############

############# Time vary norm TIED ##############
def compute_loglike_time_vary_norm_fn(row, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go, rate_norm_l, bump_height, bump_width, dip_height, dip_width):
    
    rt = row['TotalFixTime']
    t_stim = row['intended_fix']
    
    
    ILD = row['ILD']
    ABL = row['ABL']
    choice = row['choice']

    phi_params = {
        'h1': bump_width,
        'a1': bump_height,
        'h2': dip_width,
        'a2': dip_height,
        'b1': bump_offset
    }

    phi_params_obj = SimpleNamespace(**phi_params)

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
    
    


def vbmc_time_vary_norm_loglike_fn(params):
    rate_lambda, T_0, theta_E, w, t_E_aff, del_go, rate_norm_l, bump_height, bump_width, dip_height, dip_width = params
    Z_E = (w - 0.5) * 2 * theta_E
    all_loglike = Parallel(n_jobs=30)(delayed(compute_loglike_time_vary_norm_fn)(row, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go, rate_norm_l, bump_height, bump_width, dip_height, dip_width)\
                                       for _, row in df_valid_animal_less_than_1.iterrows() )
    return np.sum(all_loglike)

def vbmc_time_vary_norm_prior_fn(params):
    rate_lambda, T_0, theta_E, w, t_E_aff, del_go, rate_norm_l, bump_height, bump_width, dip_height, dip_width = params

    rate_lambda_logpdf = trapezoidal_logpdf(
        rate_lambda,
        time_vary_norm_tied_rate_lambda_bounds[0],
        time_vary_norm_tied_rate_lambda_plausible_bounds[0],
        time_vary_norm_tied_rate_lambda_plausible_bounds[1],
        time_vary_norm_tied_rate_lambda_bounds[1]
    )
    
    T_0_logpdf = trapezoidal_logpdf(
        T_0,
        time_vary_norm_tied_T_0_bounds[0],
        time_vary_norm_tied_T_0_plausible_bounds[0],
        time_vary_norm_tied_T_0_plausible_bounds[1],
        time_vary_norm_tied_T_0_bounds[1]
    )
    
    theta_E_logpdf = trapezoidal_logpdf(
        theta_E,
        time_vary_norm_tied_theta_E_bounds[0],
        time_vary_norm_tied_theta_E_plausible_bounds[0],
        time_vary_norm_tied_theta_E_plausible_bounds[1],
        time_vary_norm_tied_theta_E_bounds[1]
    )
    
    w_logpdf = trapezoidal_logpdf(
        w,
        time_vary_norm_tied_w_bounds[0],
        time_vary_norm_tied_w_plausible_bounds[0],
        time_vary_norm_tied_w_plausible_bounds[1],
        time_vary_norm_tied_w_bounds[1]
    )
    
    t_E_aff_logpdf = trapezoidal_logpdf(
        t_E_aff,
        time_vary_norm_tied_t_E_aff_bounds[0],
        time_vary_norm_tied_t_E_aff_plausible_bounds[0],
        time_vary_norm_tied_t_E_aff_plausible_bounds[1],
        time_vary_norm_tied_t_E_aff_bounds[1]
    )
    
    del_go_logpdf = trapezoidal_logpdf(
        del_go,
        time_vary_norm_tied_del_go_bounds[0],
        time_vary_norm_tied_del_go_plausible_bounds[0],
        time_vary_norm_tied_del_go_plausible_bounds[1],
        time_vary_norm_tied_del_go_bounds[1]
    )
    rate_norm_l_logpdf = trapezoidal_logpdf(
        rate_norm_l,
        time_vary_norm_tied_rate_norm_l_bounds[0],
        time_vary_norm_tied_rate_norm_l_plausible_bounds[0],
        time_vary_norm_tied_rate_norm_l_plausible_bounds[1],
        time_vary_norm_tied_rate_norm_l_bounds[1]
    )
    
    bump_height_logpdf = trapezoidal_logpdf(
        bump_height,
        bump_height_bounds[0],
        bump_height_plausible_bounds[0],
        bump_height_plausible_bounds[1],
        bump_height_bounds[1]
    )

    bump_width_logpdf = trapezoidal_logpdf(
        bump_width,
        bump_width_bounds[0],
        bump_width_plausible_bounds[0],
        bump_width_plausible_bounds[1],
        bump_width_bounds[1]
    )

    dip_height_logpdf = trapezoidal_logpdf(
        dip_height,
        dip_height_bounds[0],
        dip_height_plausible_bounds[0],
        dip_height_plausible_bounds[1],
        dip_height_bounds[1]
    )

    dip_width_logpdf = trapezoidal_logpdf(
        dip_width,
        dip_width_bounds[0],
        dip_width_plausible_bounds[0],
        dip_width_plausible_bounds[1],
        dip_width_bounds[1]
    )

    return (
        rate_lambda_logpdf +
        T_0_logpdf +
        theta_E_logpdf +
        w_logpdf +
        t_E_aff_logpdf +
        del_go_logpdf + 
        rate_norm_l_logpdf + 
        bump_height_logpdf +
        bump_width_logpdf +
        dip_height_logpdf +
        dip_width_logpdf
    )

def vbmc_time_vary_norm_joint_fn(params):
    priors = vbmc_time_vary_norm_prior_fn(params)
    loglike = vbmc_time_vary_norm_loglike_fn(params)

    return priors + loglike

time_vary_norm_tied_rate_lambda_bounds = [0.5, 5]
time_vary_norm_tied_T_0_bounds = [50e-3, 800e-3]
time_vary_norm_tied_theta_E_bounds = [1, 15]
time_vary_norm_tied_w_bounds = [0.3, 0.7]
time_vary_norm_tied_t_E_aff_bounds = [0.01, 0.2]
time_vary_norm_tied_del_go_bounds = [0, 0.2]
time_vary_norm_tied_rate_norm_l_bounds = [0, 2]
bump_height_bounds = [0.02, 1]
bump_width_bounds = [0.1, 1]
dip_width_bounds = [0.01, 1]
dip_height_bounds = [0.001, 1]

time_vary_norm_tied_rate_lambda_plausible_bounds = [1, 3]
time_vary_norm_tied_T_0_plausible_bounds = [90e-3, 400e-3]
time_vary_norm_tied_theta_E_plausible_bounds = [1.5, 10]
time_vary_norm_tied_w_plausible_bounds = [0.4, 0.6]
time_vary_norm_tied_t_E_aff_plausible_bounds = [0.03, 0.09]
time_vary_norm_tied_del_go_plausible_bounds = [0.05, 0.15]
time_vary_norm_tied_rate_norm_l_plausible_bounds = [0.8, 0.99]
bump_height_plausible_bounds = [0.1, 0.5]
bump_width_plausible_bounds = [0.2, 0.3]
dip_width_plausible_bounds = [0.025, 0.05]
dip_height_plausible_bounds = [0.2, 0.5]

# Add bounds for all parameters (order: V_A, theta_A, t_A_aff, rate_lambda, T_0, theta_E, w, t_E_aff, del_go)
time_vary_norm_tied_lb = np.array([
    time_vary_norm_tied_rate_lambda_bounds[0],
    time_vary_norm_tied_T_0_bounds[0],
    time_vary_norm_tied_theta_E_bounds[0],
    time_vary_norm_tied_w_bounds[0],
    time_vary_norm_tied_t_E_aff_bounds[0],
    time_vary_norm_tied_del_go_bounds[0],
    time_vary_norm_tied_rate_norm_l_bounds[0],
    bump_height_bounds[0],
    bump_width_bounds[0],
    dip_height_bounds[0],
    dip_width_bounds[0]
])

time_vary_norm_tied_ub = np.array([
    time_vary_norm_tied_rate_lambda_bounds[1],
    time_vary_norm_tied_T_0_bounds[1],
    time_vary_norm_tied_theta_E_bounds[1],
    time_vary_norm_tied_w_bounds[1],
    time_vary_norm_tied_t_E_aff_bounds[1],
    time_vary_norm_tied_del_go_bounds[1],
    time_vary_norm_tied_rate_norm_l_bounds[1],
    bump_height_bounds[1],
    bump_width_bounds[1],
    dip_height_bounds[1],
    dip_width_bounds[1]
])

time_vary_norm_tied_plb = np.array([
    time_vary_norm_tied_rate_lambda_plausible_bounds[0],
    time_vary_norm_tied_T_0_plausible_bounds[0],
    time_vary_norm_tied_theta_E_plausible_bounds[0],
    time_vary_norm_tied_w_plausible_bounds[0],
    time_vary_norm_tied_t_E_aff_plausible_bounds[0],
    time_vary_norm_tied_del_go_plausible_bounds[0],
    time_vary_norm_tied_rate_norm_l_plausible_bounds[0],
    bump_height_plausible_bounds[0],
    bump_width_plausible_bounds[0],
    dip_height_plausible_bounds[0],
    dip_width_plausible_bounds[0]
])

time_vary_norm_tied_pub = np.array([
    time_vary_norm_tied_rate_lambda_plausible_bounds[1],
    time_vary_norm_tied_T_0_plausible_bounds[1],
    time_vary_norm_tied_theta_E_plausible_bounds[1],
    time_vary_norm_tied_w_plausible_bounds[1],
    time_vary_norm_tied_t_E_aff_plausible_bounds[1],
    time_vary_norm_tied_del_go_plausible_bounds[1],
    time_vary_norm_tied_rate_norm_l_plausible_bounds[1],
    bump_height_plausible_bounds[1],
    bump_width_plausible_bounds[1],
    dip_height_plausible_bounds[1],
    dip_width_plausible_bounds[1]
])


###############End of time vary norm TIED #############

# %%
### Read csv and get batch data###
exp_df = pd.read_csv('../outExp.csv')

exp_df = exp_df[~((exp_df['RTwrtStim'].isna()) & (exp_df['abort_event'] == 3))].copy()

exp_df_batch = exp_df[
    (exp_df['batch_name'] == batch_name) &
    (exp_df['LED_trial'].isin([np.nan, 0]))
].copy()

exp_df_batch['choice'] = exp_df_batch['response_poke'].apply(lambda x: 1 if x == 3 else (-1 if x == 2 else random.choice([1, -1])))
# 1 or 0 if the choice was correct or not
exp_df_batch['accuracy'] = (exp_df_batch['ILD'] * exp_df_batch['choice']).apply(lambda x: 1 if x > 0 else 0)

# %%
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
for animal_idx in range(len(animal_ids)):
    animal = animal_ids[animal_idx]

    df_all_trials_animal = df_valid_and_aborts[df_valid_and_aborts['animal'] == animal]
    df_aborts_animal = df_aborts[df_aborts['animal'] == animal]

    ######### NOTE: Sake of testing, remove half of trials ############
    # df_all_trials_animal = df_all_trials_animal.sample(frac=0.01)
    # df_aborts_animal = df_aborts_animal.sample(frac=0.01)
    ############ END NOTE ############################################

    print(f'Batch: {batch_name},sample animal: {animal}')
    pdf_filename = f'results_{batch_name}_animal_{animal}.pdf'
    pdf = PdfPages(pdf_filename)

    # --- Page 1: Batch Name and Animal ID ---
    fig_text = plt.figure(figsize=(8.5, 11)) # Standard page size looks better
    fig_text.clf() # Clear the figure
    fig_text.text(0.1, 0.9, f"Analysis Report", fontsize=20, weight='bold')
    fig_text.text(0.1, 0.8, f"Batch Name: {batch_name}", fontsize=14)
    fig_text.text(0.1, 0.75, f"Animal ID: {animal}", fontsize=14)
    fig_text.gca().axis("off")
    pdf.savefig(fig_text, bbox_inches='tight')
    plt.close(fig_text)


    # find ABL and ILD
    ABL_arr = df_valid_and_aborts['ABL'].unique()
    ILD_arr = df_valid_and_aborts['ILD'].unique()


    # sort ILD arr in ascending order
    ILD_arr = np.sort(ILD_arr)
    ABL_arr = np.sort(ABL_arr)

    ####################################################
    ########### Abort Model ##############################
    ####################################################

    V_A_0 = 3.3
    theta_A_0 = 3.8
    t_A_aff_0 = -0.27

    x_0 = np.array([V_A_0, theta_A_0, t_A_aff_0])
    vbmc = VBMC(vbmc_joint_aborts_fn, x_0, aborts_lb, aborts_ub, aborts_plb, aborts_pub, options={'display': 'on', 'max_fun_evals': 200 * (2 + 3)})    
    # vbmc = VBMC(vbmc_joint_aborts_fn, x_0, aborts_lb, aborts_ub, aborts_plb, aborts_pub, options={'display': 'on'})

    vp, results = vbmc.optimize()

    # %%
    vp_samples = vp.sample(int(1e6))[0]
    V_A_samp = vp_samples[:,0]
    theta_A_samp = vp_samples[:,1]
    t_A_aff_samp = vp_samples[:,2]
    # %%
    V_A = vp_samples[:,0].mean()
    theta_A = vp_samples[:,1].mean()
    t_A_aff = vp_samples[:,2].mean()


    print(f'V_A: {V_A}')
    print(f'theta_A: {theta_A}')
    print(f't_A_aff: {t_A_aff}')

    combined_samples = np.transpose(np.vstack((V_A_samp, theta_A_samp, t_A_aff_samp)))
    param_labels = ['V_A', 'theta_A', 't_A_aff']
    # Calculate log likelihood at the mean parameter values
    aborts_loglike = vbmc_aborts_loglike_fn([V_A, theta_A, t_A_aff])
    # --- Page: Abort Model Posterior Means ---
    save_posterior_summary_page(
        pdf_pages=pdf,
        title=f'Abort Model - Posterior Means ({animal})',
        posterior_means=pd.Series({'V_A': V_A, 'theta_A': theta_A, 't_A_aff': t_A_aff}),
        param_labels={'V_A': 'V_A', 'theta_A': 'theta_A', 't_A_aff': 't_A_aff'},
        vbmc_results={'message': results['message'], 'elbo': results['elbo'], 'elbo_sd': results['elbo_sd'], 'loglike': aborts_loglike},
        extra_text=f"T_trunc = {T_trunc:.3f}"
    )

    # --- Page: Abort Model Corner Plot ---
    save_corner_plot(
        pdf_pages=pdf,
        samples=combined_samples,
        param_labels=param_labels,
        title=f'Abort Model - Corner Plot ({animal})',
        truths=[V_A, theta_A, t_A_aff] # Show posterior means as truths
    )

    # --- Page: Abort Model Diagnostic Plot ---
    plot_abort_diagnostic(
        pdf_pages=pdf,
        df_aborts_animal=df_aborts_animal,
        df_valid_and_aborts=df_valid_and_aborts, # Use combined df for sampling intended fix times
        N_theory=N_theory,
        V_A=V_A,
        theta_A=theta_A,
        t_A_aff=t_A_aff,
        T_trunc=T_trunc,
        rho_A_t_VEC_fn=rho_A_t_VEC_fn,
        cum_A_t_fn=cum_A_t_fn,
        title=f'Abort Model RTD Diagnostic ({animal})'
    )

    vbmc_aborts_results = {
        'V_A_samples': V_A_samp,
        'theta_A_samples': theta_A_samp,
        't_A_aff_samp': t_A_aff_samp,
        'message': results['message'],
        'elbo': results['elbo'],
        'elbo_sd': results['elbo_sd'],
        'loglike': aborts_loglike
    }

    ######################################################
    ############### VANILLA TIED MODEL #####################
    ########################################################

    # %%
    is_norm = False
    is_time_vary = False
    phi_params_obj = np.nan
    rate_norm_l = np.nan

    # df_all_trials_animal
    df_valid_animal = df_all_trials_animal[df_all_trials_animal['success'].isin([1,-1])]
    df_valid_animal_less_than_1 = df_valid_animal[df_valid_animal['RTwrtStim'] < 1]


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
    vbmc = VBMC(vbmc_vanilla_tied_joint_fn, x_0, vanilla_tied_lb, vanilla_tied_ub, vanilla_plb, vanilla_pub, options={'display': 'on'})
    vp, results = vbmc.optimize()

    # %%
    # Sample from the VBMC posterior (returns tuple: samples, log weights)
    vp_samples = vp.sample(int(1e5))[0]

    # Convert T_0 to ms (T_0 is at index 4)
    vp_samples[:, 1] *= 1e3

    # Parameter labels (order matters!)
    param_labels = [
        r'$\lambda$',       # 3
        r'$T_0$ (ms)',      # 4
        r'$\theta_E$',      # 5
        r'$w$',             # 6
        r'$t_E^{aff}$',     # 7
        r'$\Delta_{go}$'    # 8
    ]

    # Compute 1st and 99th percentiles for each param to restrict range
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

    # %%
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
        vbmc_results={'message': results['message'], 'elbo': results['elbo'], 'elbo_sd': results['elbo_sd'], 'loglike': vanilla_tied_loglike},
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
    t_stim_samples = df_all_trials_animal['intended_fix'].sample(N_sim, replace=True).values
    ABL_samples = df_all_trials_animal['ABL'].sample(N_sim, replace=True).values
    ILD_samples = df_all_trials_animal['ILD'].sample(N_sim, replace=True).values
    
    # P_A and C_A vs t wrt stim for all 3 TIED models
    t_pts = np.arange(-1, 2, 0.001)
    P_A_mean, C_A_mean, t_stim_samples_for_diag = calculate_theoretical_curves(
        df_valid_and_aborts, N_theory, t_pts, t_A_aff, V_A, theta_A, rho_A_t_fn
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
    sim_df_1, data_df_1 = prepare_simulation_data(sim_results_df, df_valid_animal_less_than_1)

    

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


    ############### END Of vanilla tied model #####################

    #########################################################
    ########## Normalized model ###############################
    ########################################################

    is_norm =  True
    is_time_vary = False
    phi_params_obj = np.nan

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

    vbmc = VBMC(vbmc_norm_tied_joint_fn, x_0, norm_tied_lb, norm_tied_ub, norm_tied_plb, norm_tied_pub, options={'display': 'on'})
    vp, results = vbmc.optimize()

    vp_samples = vp.sample(int(1e5))[0]
    vp_samples[:, 1] *= 1e3
    param_labels = [
        r'$\lambda$',       # 3
        r'$T_0$ (ms)',      # 4
        r'$\theta_E$',      # 5
        r'$w$',             # 6
        r'$t_E^{aff}$',     # 7
        r'$\Delta_{go}$',  # 8,
        r'rate_norm'        # 9
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
    norm_tied_corner_fig.suptitle(f'Normalized Tied Posterior (Animal: {animal})', y=1.02) # Add a title to the corner plot figure
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

    norm_tied_loglike = vbmc_norm_tied_loglike_fn([rate_lambda, T_0, theta_E, w, t_E_aff, del_go, rate_norm_l])
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
        vbmc_results={'message': results['message'], 'elbo': results['elbo'], 'elbo_sd': results['elbo_sd'], 'loglike': norm_tied_loglike},
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

    #######################################
    ##### norm tied diagnostics ###########
    ######################################
    print(f'Rate norm is {rate_norm_l}')
    
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
    
    sim_results = Parallel(n_jobs=30)(
        delayed(psiam_tied_data_gen_wrapper_rate_norm_fn)(
            V_A, theta_A, ABL_samples[iter_num], ILD_samples[iter_num], rate_lambda, T_0, theta_E, Z_E, t_A_aff, t_E_aff, del_go, 
            t_stim_samples[iter_num], rate_norm_l, iter_num, N_print, dt
        ) for iter_num in tqdm(range(N_sim))
    )
    sim_results_df = pd.DataFrame(sim_results)
    sim_results_df_valid = sim_results_df[
        (sim_results_df['rt'] > sim_results_df['t_stim']) &
        (sim_results_df['rt'] - sim_results_df['t_stim'] < 1)
    ].copy()
    
    sim_df_1, data_df_1 = prepare_simulation_data(sim_results_df_valid, df_valid_animal_less_than_1)
    
    theory_results_up_and_down, theory_time_axis, bins, bin_centers = plot_rt_distributions(
        sim_df_1, data_df_1, ILD_arr, ABL_arr, t_pts, P_A_mean, C_A_mean, 
        t_stim_samples_for_diag, V_A, theta_A, t_A_aff, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go,
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
    ############### END Of normalized tied model #####################

    #################################################
    ####### Time-Varying Normalized TIED Model ########
    #################################################

    
    is_norm = True
    is_time_vary = True
    bump_offset = 0

    rate_lambda_0 = 2.3
    T_0_0 = 100 * 1e-3
    theta_E_0 = 3
    w_0 = 0.51
    t_E_aff_0 = 0.071
    del_go_0 = 0.13
    rate_norm_l_0 = 0.95
    bump_height_0 = 0.35846
    bump_width_0 = 0.28043
    dip_height_0 = 0.29911
    dip_width_0 = 0.01818

    x_0 = np.array([
        rate_lambda_0,
        T_0_0,
        theta_E_0,
        w_0,
        t_E_aff_0,
        del_go_0,
        rate_norm_l_0,
        bump_height_0,
        bump_width_0,
        dip_height_0,
        dip_width_0
    ])

    vbmc = VBMC(vbmc_time_vary_norm_joint_fn, x_0, time_vary_norm_tied_lb, time_vary_norm_tied_ub, time_vary_norm_tied_plb, time_vary_norm_tied_pub, options={'display': 'on'})
    vp, results = vbmc.optimize()

    #### time vary norm tied model ####
    vp_samples = vp.sample(int(1e5))[0]
    vp_samples[:, 1] *= 1e3  # Convert T_0 to ms for display

    param_labels = [
        r'$\lambda$',       # 0
        r'$T_0$ (ms)',      # 1
        r'$\theta_E$',      # 2
        r'$w$',             # 3
        r'$t_E^{aff}$',     # 4
        r'$\Delta_{go}$',   # 5
        r'rate_norm',       # 6
        r'bump_height',     # 7
        r'bump_width',      # 8
        r'dip_height',      # 9
        r'dip_width'        # 10
    ]

    percentiles = np.percentile(vp_samples, [1, 99], axis=0)
    _ranges = [(percentiles[0, i], percentiles[1, i]) for i in range(vp_samples.shape[1])]
    time_vary_norm_tied_corner_fig = corner.corner(
        vp_samples,
        labels=param_labels,
        show_titles=True,
        quantiles=[0.025, 0.5, 0.975],
        range=_ranges,
        title_fmt=".3f"
    )
    time_vary_norm_tied_corner_fig.suptitle(f'Time-Varying Normalized Tied Posterior (Animal: {animal})', y=1.02)
    vp_samples[:, 1] /= 1e3

    # Calculate posterior means
    rate_lambda = vp_samples[:, 0].mean()
    T_0 = vp_samples[:, 1].mean()
    theta_E = vp_samples[:, 2].mean()
    w = vp_samples[:, 3].mean()
    Z_E = (w - 0.5) * 2 * theta_E
    t_E_aff = vp_samples[:, 4].mean()
    del_go = vp_samples[:, 5].mean()
    rate_norm_l = vp_samples[:, 6].mean()
    bump_height = vp_samples[:, 7].mean()
    bump_width = vp_samples[:, 8].mean()
    dip_height = vp_samples[:, 9].mean()
    dip_width = vp_samples[:, 10].mean()

    # Print them out
    print("Posterior Means:")
    print(f"rate_lambda  = {rate_lambda:.5f}")
    print(f"T_0 (ms)      = {1e3*T_0:.5f}")
    print(f"theta_E       = {theta_E:.5f}")
    print(f"Z_E           = {Z_E:.5f}")
    print(f"t_E_aff       = {1e3*t_E_aff:.5f} ms")
    print(f"del_go        = {del_go:.5f}")
    print(f"rate_norm_l   = {rate_norm_l:.5f}")
    print(f"bump_height   = {bump_height:.5f}")
    print(f"bump_width    = {bump_width:.5f}")
    print(f"dip_height    = {dip_height:.5f}")
    print(f"dip_width     = {dip_width:.5f}")

    phi_params = {
        'h1': bump_width,
        'a1': bump_height,
        'h2': dip_width,
        'a2': dip_height,
        'b1': bump_offset
    }
    phi_params_obj = SimpleNamespace(**phi_params)

    time_vary_norm_loglike = vbmc_time_vary_norm_loglike_fn([rate_lambda, T_0, theta_E, w, t_E_aff, del_go, rate_norm_l, bump_height, bump_width, dip_height, dip_width])
    save_posterior_summary_page(
        pdf_pages=pdf,
        title=f'Time-Varying Normalized Tied Model - Posterior Means ({animal})',
        posterior_means=pd.Series({
            'rate_lambda': rate_lambda,
            'T_0': 1e3*T_0,
            'theta_E': theta_E,
            'w': w,
            'Z_E': Z_E,
            't_E_aff': 1e3*t_E_aff,
            'del_go': del_go,
            'rate_norm_l': rate_norm_l,
            'bump_height': bump_height,
            'bump_width': bump_width,
            'dip_height': dip_height,
            'dip_width': dip_width
        }),
        param_labels={
            'rate_lambda': r'$\lambda$',
            'T_0': r'$T_0$ (ms)',
            'theta_E': r'$\theta_E$',
            'w': r'$w$',
            'Z_E': r'$Z_E$',
            't_E_aff': r'$t_E^{aff}$',
            'del_go': r'$\Delta_{go}$',
            'rate_norm_l': r'rate_norm',
            'bump_height': r'bump_height',
            'bump_width': r'bump_width',
            'dip_height': r'dip_height',
            'dip_width': r'dip_width'
        },
        vbmc_results={'message': results['message'], 'elbo': results['elbo'], 'elbo_sd': results['elbo_sd'], 'loglike': time_vary_norm_loglike},
        extra_text=f"T_trunc = {T_trunc:.3f}"
    )
    pdf.savefig(time_vary_norm_tied_corner_fig, bbox_inches='tight')

    vbmc_time_vary_norm_tied_results = {
        'rate_lambda_samples': vp_samples[:, 0],
        'T_0_samples': vp_samples[:, 1],
        'theta_E_samples': vp_samples[:, 2],
        'w_samples': vp_samples[:, 3],
        't_E_aff_samples': vp_samples[:, 4],
        'del_go_samples': vp_samples[:, 5],
        'rate_norm_l_samples': vp_samples[:, 6],
        'bump_height_samples': vp_samples[:, 7],
        'bump_width_samples': vp_samples[:, 8],
        'dip_height_samples': vp_samples[:, 9],
        'dip_width_samples': vp_samples[:, 10],
        'message': results['message'],
        'elbo': results['elbo'],
        'elbo_sd': results['elbo_sd'],
        'loglike': time_vary_norm_loglike
    }
    
    ################################################
    ######## Time-Varying Normalized TIED Model Diagnostics #######
    ################################################
    print(f'Rate norm is {rate_norm_l}, bump height is {bump_height}, bump width is {bump_width}, dip height is {dip_height}, dip width is {dip_width}')
    
    sim_results = Parallel(n_jobs=30)(
        delayed(psiam_tied_data_gen_wrapper_rate_norm_time_vary_fn)(
            V_A, theta_A, ABL_samples[iter_num], ILD_samples[iter_num], rate_lambda, T_0, theta_E, Z_E, t_A_aff, t_E_aff, del_go, 
            t_stim_samples[iter_num], rate_norm_l, iter_num, N_print, phi_params_obj, dt
        ) for iter_num in tqdm(range(N_sim))
    )

    sim_results_df = pd.DataFrame(sim_results)
    sim_results_df_valid = sim_results_df[
        (sim_results_df['rt'] > sim_results_df['t_stim']) &
        (sim_results_df['rt'] - sim_results_df['t_stim'] < 1)
    ].copy()
    
    sim_df_1, data_df_1 = prepare_simulation_data(sim_results_df_valid, df_valid_animal_less_than_1)
    
    theory_results_up_and_down, theory_time_axis, bins, bin_centers = plot_rt_distributions(
        sim_df_1, data_df_1, ILD_arr, ABL_arr, t_pts, P_A_mean, C_A_mean, 
        t_stim_samples_for_diag, V_A, theta_A, t_A_aff, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go,
        phi_params_obj, rate_norm_l, True, True, K_max, T_trunc,
        cum_pro_and_reactive_time_vary_fn, up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn,
        animal, pdf, model_name="Time-Varying Normalized Tied"
    )
    
    plot_tachometric_curves(
        sim_df_1, data_df_1, ILD_arr, ABL_arr, theory_results_up_and_down,
        theory_time_axis, bins, animal, pdf, model_name="Time-Varying Normalized Tied"
    )
    
    plot_grand_summary(
        sim_df_1, data_df_1, ILD_arr, ABL_arr, bins, bin_centers,
        animal, pdf, model_name="Time-Varying Normalized Tied"
    )

    #### end of time vary norm tied ##############

    # Create model tables and add them to the PDF
    # These functions are now imported from animal_wise_plotting_utils
    abort_df = None
    tied_df = None
    
    save_dict = {
        'vbmc_aborts_results': vbmc_aborts_results,
        'vbmc_vanilla_tied_results': vbmc_vanilla_tied_results,
        'vbmc_norm_tied_results': vbmc_norm_tied_results,
        'vbmc_time_vary_norm_tied_results': vbmc_time_vary_norm_tied_results
    }
    
    print("\nGenerating model comparison tables...")
    if 'vbmc_aborts_results' in save_dict:
        abort_df = create_abort_table(save_dict['vbmc_aborts_results'])
        if abort_df is not None:
            render_df_to_pdf(abort_df, f"Abort Model Results - Animal {animal}", pdf)
            print(f"Added abort model results table to PDF for animal {animal}")
    
    tied_df = create_tied_table(save_dict)
    if tied_df is not None:
        render_df_to_pdf(tied_df, f"Tied Models Comparison - Animal {animal}", pdf)
        print(f"Added tied models comparison table to PDF for animal {animal}")
    
    ### pkl file - vbmc samples ###
    pkl_filename = f'results_{batch_name}_animal_{animal}.pkl'

    with open(pkl_filename, 'wb') as f:
        pickle.dump(save_dict, f)

    print(f"Saved results to {pkl_filename}")

    ### PDF save  ###
    pdf.close()
    print(f"Saved PDF report to {pdf_filename}")
