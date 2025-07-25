# %%
# %%
import numpy as np
import matplotlib
# matplotlib.use('Agg')  # Use the Agg backend which doesn't require tkinter
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

# %%
############3 Params #############
batch_name = 'LED34_even'
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

    if pdf_trunc_factor == 0:
        likelihood = 1e-50
    
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

V_A_plausible_bounds = [0.5, 4]
theta_A_plausible_bounds = [0.5, 4]
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
# outExp.csv
exp_df = pd.read_csv('../outUni.csv')
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
    (exp_df['batch_name'] == 'LED34') &
    (exp_df['LED_trial'].isin([np.nan, 0])) &
    (exp_df['animal'].isin([48,52,56,60])) &
    (exp_df['session_type'].isin([1,2]))  
].copy()

# aborts don't have choice, so assign random 
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




# %%
# for animal_idx in range(len(animal_ids)):
for animal_idx in [1]:
    animal = animal_ids[animal_idx]

    df_all_trials_animal = df_valid_and_aborts[df_valid_and_aborts['animal'] == animal]
    df_aborts_animal = df_aborts[df_aborts['animal'] == animal]
    ######### NOTE: Sake of testing, remove half of trials ############
    # df_all_trials_animal = df_all_trials_animal.sample(frac=0.01)
    # df_aborts_animal = df_aborts_animal.sample(frac=0.01)
    ############ END NOTE ############################################

    print(f'Batch: {batch_name},sample animal: {animal}')
    # pdf_filename = f'results_{batch_name}_animal_{animal}.pdf'
    # pdf = PdfPages(pdf_filename)

    # --- Page 1: Batch Name and Animal ID ---
    fig_text = plt.figure(figsize=(8.5, 11)) # Standard page size looks better
    fig_text.clf() # Clear the figure
    fig_text.text(0.1, 0.9, f"Analysis Report", fontsize=20, weight='bold')
    fig_text.text(0.1, 0.8, f"Batch Name: {batch_name}", fontsize=14)
    fig_text.text(0.1, 0.75, f"Animal ID: {animal}", fontsize=14)
    fig_text.gca().axis("off")
    # pdf.savefig(fig_text, bbox_inches='tight')
    plt.close(fig_text)


    # find ABL and ILD
    ABL_arr = df_all_trials_animal['ABL'].unique()
    ILD_arr = df_all_trials_animal['ILD'].unique()


    # sort ILD arr in ascending order
    ILD_arr = np.sort(ILD_arr)
    ABL_arr = np.sort(ABL_arr)

    ####################################################
    ########### Abort Model ##############################
    ####################################################

    # V_A_0 = 3.3
    # theta_A_0 = 3.8
    # t_A_aff_0 = -0.27

    # x_0 = np.array([V_A_0, theta_A_0, t_A_aff_0])
    # vbmc = VBMC(vbmc_joint_aborts_fn, x_0, aborts_lb, aborts_ub, aborts_plb, aborts_pub, options={'display': 'on', 'max_fun_evals': 200 * (2 + 3)})    
    # # vbmc = VBMC(vbmc_joint_aborts_fn, x_0, aborts_lb, aborts_ub, aborts_plb, aborts_pub, options={'display': 'on'})

    # vp, results = vbmc.optimize()

    # # %%
    # vp_samples = vp.sample(int(1e6))[0]
    # V_A_samp = vp_samples[:,0]
    # theta_A_samp = vp_samples[:,1]
    # t_A_aff_samp = vp_samples[:,2]
    # # %%
    # V_A = vp_samples[:,0].mean()
    # theta_A = vp_samples[:,1].mean()
    # t_A_aff = vp_samples[:,2].mean()


    # print(f'theta_A: {theta_A}')
    # print(f't_A_aff: {t_A_aff}')

    # combined_samples = np.transpose(np.vstack((V_A_samp, theta_A_samp, t_A_aff_samp)))
    # param_labels = ['V_A', 'theta_A', 't_A_aff']
    # # Calculate log likelihood at the mean parameter values
    # aborts_loglike = vbmc_aborts_loglike_fn([V_A, theta_A, t_A_aff])
    # t_pts = np.arange(0, 2, 0.001)
    # pdf_samples = np.zeros((N_theory, len(t_pts)))

    # t_stim_samples_df = df_valid_and_aborts.sample(n=N_theory, replace=True).copy()
    # t_stim_samples = t_stim_samples_df['intended_fix'].values

    # for i, t_stim in enumerate(t_stim_samples):
    #     t_stim_idx = np.searchsorted(t_pts, t_stim)
    #     proactive_trunc_idx = np.searchsorted(t_pts, T_trunc)
    #     pdf_samples[i, :proactive_trunc_idx] = 0
    #     pdf_samples[i, t_stim_idx:] = 0
    #     t_btn = t_pts[proactive_trunc_idx:t_stim_idx-1]
    #     pdf_samples[i, proactive_trunc_idx:t_stim_idx-1] = rho_A_t_VEC_fn(t_btn - t_A_aff, V_A, theta_A) / (1 - cum_A_t_fn(T_trunc - t_A_aff, V_A, theta_A))
    # avg_pdf = np.mean(pdf_samples, axis=0)


    # Plotting
    fig_aborts_diag = plt.figure(figsize=(10, 5))
    bins = np.arange(0, 0.5, 0.01)

    # Get empirical abort RTs and filter by truncation time
    animal_abort_RT = df_aborts_animal['TotalFixTime'].dropna().values
    # Plot histogram of animal abort RTs
    fig_hist, ax = plt.subplots(figsize=(10, 5))
    ax.hist(animal_abort_RT, bins=np.arange(0, 0.5, 0.01), alpha=0.5, label='animal abort RTs')
    ax.axvline(0.5)
    ax.set_xlabel('RT (s)')
    ax.set_ylabel('Count')
    # ax.legend()
    ax.set_title(f'Animal {animal} Abort RTs')
    ax.axvline(0.17, color='r')
    fig_hist.savefig(f'test_34_animal_{animal}_aborts_hist.png', bbox_inches='tight')
    # plt.close(fig_hist)

    # animal_abort_RT_trunc = animal_abort_RT[animal_abort_RT > T_trunc]
    print(animal_abort_RT.min(), animal_abort_RT.max())
    # save hist
    # Plot empirical histogram (scaled by fraction of aborts after truncation)
    if len(animal_abort_RT_trunc) > 0:
        # Compute N_valid_and_trunc_aborts as in the reference code
        # Need df_all_trials_animal and df_before_trunc_animal
        if 'animal' in df_valid_and_aborts.columns:
            animal_id = df_aborts_animal['animal'].iloc[0] if len(df_aborts_animal) > 0 else None
            df_all_trials_animal = df_valid_and_aborts[df_valid_and_aborts['animal'] == animal_id]
        else:
            df_all_trials_animal = df_valid_and_aborts
        df_before_trunc_animal = df_aborts_animal[df_aborts_animal['TotalFixTime'] < T_trunc]
        N_valid_and_trunc_aborts = len(df_all_trials_animal) - len(df_before_trunc_animal)
        frac_aborts = len(animal_abort_RT_trunc) / N_valid_and_trunc_aborts if N_valid_and_trunc_aborts > 0 else 0
        aborts_hist, _ = np.histogram(animal_abort_RT_trunc, bins=bins, density=True)
        # Scale the histogram by frac_aborts
        plt.plot(bins[:-1], aborts_hist * frac_aborts, label='Data (Aborts > T_trunc)')
    else:
        # Add a note if no data to plot
        plt.text(0.5, 0.5, 'No empirical abort data > T_trunc', 
                 horizontalalignment='center', verticalalignment='center', 
                 transform=plt.gca().transAxes)

    # # Plot theoretical PDF
    # plt.plot(t_pts, avg_pdf, 'r-', lw=2, label='Theory (Abort Model)')

    # plt.title('aborts ')
    # plt.xlabel('Reaction Time (s)')
    # plt.ylabel('Probability Density')
    # plt.legend()
    # plt.xlim([0, np.max(bins)]) # Limit x-axis to the bin range
    # # Add a reasonable upper limit to y-axis if needed, e.g., based on max density
    # if len(animal_abort_RT_trunc) > 0:
    #     max_density = np.max(aborts_hist * frac_aborts) if len(aborts_hist) > 0 else 1
    #     plt.ylim([0, max(np.max(avg_pdf), max_density) * 1.1])
    # elif np.any(avg_pdf > 0):
    #      plt.ylim([0, np.max(avg_pdf) * 1.1])
    # else:
    #     plt.ylim([0, 1]) # Default ylim if no data and no theory\
    # plt.savefig(f'test_34_aborts_{animal}.png')
    break

# %%
t_pts = np.arange(0, 1, 0.01)


V_A, theta_A, t_A_aff = [1, 2, -0.01]

density = np.zeros_like(t_pts)
for i, rt in enumerate(t_pts):
    t_stim = np.random.choice(df_aborts_animal['intended_fix'].values)
    pdf_trunc_factor = 1 - cum_A_t_fn(T_trunc - t_A_aff, V_A, theta_A)

    if rt < T_trunc:
        density[i] = 0
    else:
        if rt < t_stim:
            density[i] =  rho_A_t_fn(rt - t_A_aff, V_A, theta_A) / pdf_trunc_factor
        elif rt > t_stim:
            if t_stim <= T_trunc:
                density[i] = 1
            else:
                density[i] = ( 1 - cum_A_t_fn(t_stim - t_A_aff, V_A, theta_A) ) / pdf_trunc_factor


plt.plot(t_pts, density)
plt.axvline(0.15, color='r')
plt.hist(animal_abort_RT, bins=np.arange(0, 0.5, 0.01), alpha=0.5, label='animal abort RTs', density=True);
plt.savefig(f'test_34_aborts_{animal}.png')
# %%
for animal_idx in [1]:
    animal = animal_ids[animal_idx]

    df_all_trials_animal = df_valid_and_aborts[df_valid_and_aborts['animal'] == animal]
    df_aborts_animal = df_aborts[df_aborts['animal'] == animal]

    df_success_trials = df_all_trials_animal[df_all_trials_animal['success'].isin([1, -1])]
    df_success_trials_1 = df_success_trials[df_success_trials['TotalFixTime'] - df_success_trials['intended_fix'] < 1]


    ######### NOTE: Sake of testing, remove half of trials ############
    # df_all_trials_animal = df_all_trials_animal.sample(frac=0.01)
    # df_aborts_animal = df_aborts_animal.sample(frac=0.01)
    ############ END NOTE ############################################

    print(f'Batch: {batch_name},sample animal: {animal}')
    # remove aborts RT < 0.15
    df_aborts_animal = df_aborts_animal[df_aborts_animal['TotalFixTime'] > 0.15]
    animal_abort_RT = df_aborts_animal['TotalFixTime'].dropna().values
    print(f'len animal abort RTs = {len(animal_abort_RT)}')

    ABL_unique = df_all_trials_animal['ABL'].unique()
    ILD_unique = df_all_trials_animal['ILD'].unique()
    print(f'ABL_unique = {ABL_unique}')
    print(f'ILD_unique = {ILD_unique}')

    # for ABL in np.sort(ABL_unique):
    for ABL in [40]:

        for ILD in np.sort(ILD_unique):
            df_ABL_ILD_animal = df_success_trials_1[(df_success_trials_1['ABL'] == ABL) & (df_success_trials_1['ILD'] == ILD)]
            # trunc factor area 
            print(f'ABL = {ABL}, ILD = {ILD}')
            num_total = len(df_ABL_ILD_animal) + len(df_aborts_animal[df_aborts_animal['TotalFixTime'] > 0.15])
            num_ABL_ILD = len(df_ABL_ILD_animal)
            trunc_factor = num_ABL_ILD / num_total
            print(f'trunc factor = {trunc_factor}')

            # # area up and area down
            # # response_poke == 3
            df_ABL_ILD_up = df_ABL_ILD_animal[df_ABL_ILD_animal['response_poke'] == 3]
            df_ABL_ILD_down = df_ABL_ILD_animal[df_ABL_ILD_animal['response_poke'] == 2]
            n_up = len(df_ABL_ILD_up)
            n_down = len(df_ABL_ILD_down)
            area_up = n_up / num_total
            area_down = n_down / num_total
            print(f'area up = {area_up}, area down = {area_down}')
            print(f'ratio of up + down / trunc = {(area_up + area_down)/trunc_factor}')
            # df_ABL_ILD_animal_up = df_ABL_ILD_animal[df_ABL_ILD_animal['response_poke'] == 3]
            # df_ABL_ILD_animal_down = df_ABL_ILD_animal[df_ABL_ILD_animal['response_poke'] == 2]
            # n_up = len(df_ABL_ILD_animal_up)
            # n_down = len(df_ABL_ILD_animal_down)
            # area_up = n_up / num_total
            # area_down = n_down / num_total
            # print(f'area up = {area_up}, area down = {area_down}')

            # print(f'ratio of up + down / trunc')
            # print((area_up + area_down) / trunc_factor)

            
        
    