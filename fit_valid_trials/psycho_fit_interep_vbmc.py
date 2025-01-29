# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from pyvbmc import VBMC
import corner
from psiam_tied_dv_map_utils import rho_A_t_fn, up_RTs_fit_fn, down_RTs_fit_fn, up_RTs_fit_single_t_fn, psiam_tied_data_gen_wrapper, psiam_tied_data_gen_wrapper_V2
import sys
import multiprocessing
from psiam_tied_no_dv_map_utils import cum_A_t_fn, all_RTs_fit_OPTIM_fn
from psiam_tied_no_dv_map_utils import rho_A_t_fn, cum_A_t_fn, rho_E_minus_small_t_NORM_fn
from psiam_tied_dv_map_utils import cum_E_t_fn
from tqdm import tqdm
from scipy.integrate import trapezoid
import random

from psiam_tied_no_dv_map_utils import CDF_E_minus_small_t_NORM_fn_vectorized, rho_A_t_fn_vectorized,P_small_t_btn_x1_x2_vectorized
import time
from scipy.interpolate import interp1d


# %%
# read out_LED.csv as dataframe
og_df = pd.read_csv('../out_LED.csv')

# %%
# chose non repeat trials - 0 or 2 or missing
df = og_df[ og_df['repeat_trial'].isin([0,2]) | og_df['repeat_trial'].isna() ]

# only session type 7
session_type = 7    
df = df[ df['session_type'].isin([session_type]) ]

# training level 16
training_level = 16
df = df[ df['training_level'].isin([training_level]) ]

df = df[  df['LED_trial'] == 0 ]

# 1 is right , -1 is left
df['choice'] = df['response_poke'].apply(lambda x: 1 if x == 3 else (-1 if x == 2 else random.choice([1, -1])))

# 1 or 0 if the choice was correct or not
df['correct'] = (df['ILD'] * df['choice']).apply(lambda x: 1 if x > 0 else 0)

# %%
# find ABL and ILD
ABL_arr = df['ABL'].unique()
ILD_arr = df['ILD'].unique()


# sort ILD arr in ascending order
ILD_arr = np.sort(ILD_arr)
ABL_arr = np.sort(ABL_arr)

print('ABL:', ABL_arr)
print('ILD:', ILD_arr)

# %%
# LED off rows
df_1 = df[ df['LED_trial'] == 0 ]
print(f'len of LED off rows: {len(df_1)}')
# remove trunc aborts
df_1 = df_1 [ ~( (df_1['abort_event']==3) & (df_1['timed_fix'] < 0.3) ) ]
print(f'len of LED off rows after removing trunc aborts: {len(df_1)}')

df_1 = df_1[df_1['timed_fix'] > df_1['intended_fix']]
print(f'len of rows after reoving all aborts = {len(df_1)}')

# %%
import pickle
with open('../fitting_aborts/post_led_censor_test_vbmc.pkl', 'rb') as f:
    vp = pickle.load(f)

# %%
vp_sample = vp.sample(int(1e6))[0]

# %%
V_A = np.mean(vp_sample[:,0])
theta_A = np.mean(vp_sample[:,1])
t_motor = 0.04
t_A_aff = np.mean(vp_sample[:,2]) - t_motor

# %%
t_A_aff

# %% [markdown]
# # VBMC

# %% [markdown]
# ## loglike fn

# %%
t_stim_arr = np.arange(0.2, 2.3, 0.1)
t_pts_for_integ = np.arange(0, 10, 0.001)
tasks = [(ABL, ILD) for ABL in ABL_arr for ILD in ILD_arr]

# %%
def P_up_optim_fn(t, V_A, theta_A, x1, x2, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, bound, K_max, t_A_aff, t_motor, t_stim):
    return rho_A_t_fn_vectorized(t - t_A_aff - t_motor, V_A, theta_A) * ( CDF_E_minus_small_t_NORM_fn_vectorized(t - t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, bound, K_max) + \
                                                 P_small_t_btn_x1_x2_vectorized(x1, x2, t - t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, K_max) )

# %%
def compute_integral(ABL, ILD, t_stim_arr, t_pts_for_integ, P_up_optim_fn, V_A, theta_A, x1, x2,
                    rate_lambda, T_0, theta_E, Z_E, bound, K_max, t_A_aff, t_motor):
    integrals = np.zeros_like(t_stim_arr)
    for t_stim_idx, t_stim in enumerate(t_stim_arr):
        unknown_integ_arr = P_up_optim_fn(t_pts_for_integ, V_A, theta_A, x1, x2,
                                    ABL, ILD, rate_lambda, T_0, theta_E, Z_E,
                                    bound, K_max, t_A_aff, t_motor, t_stim)
        integrals[t_stim_idx] = trapezoid(unknown_integ_arr, t_pts_for_integ)
    return ((ABL, ILD), integrals)

# %%
def compute_loglike(row, integral_vs_t_stim):
    intended_fix = row['intended_fix']
    ILD = row['ILD']
    ABL = row['ABL']
    choice = row['response_poke']

    t_stim = intended_fix

    p_up_vs_t_stim = integral_vs_t_stim[(ABL, ILD)]
    cubic_interp_func = interp1d(
        t_stim_arr, 
        p_up_vs_t_stim, 
        kind='cubic', 
        fill_value="extrapolate",  # Allows extrapolation outside the data range
        assume_sorted=True         # Assumes input data is sorted
    )
    p_up = cubic_interp_func(t_stim)


    if choice == 3:
        likelihood = p_up
    elif choice == 2:
        likelihood = 1 - p_up
    
    if likelihood <= 0:
        likelihood = 1e-50

    return np.log(likelihood)    


def psiam_tied_loglike_fn(params):
    rate_lambda, T_0, theta_E, Z_E = params
    
    bound = 1
    K_max = 10
    x1 = 1; x2 = 2
    integral_vs_t_stim = {}
    # start_time = time.time()

    results = Parallel(n_jobs=30)(
        delayed(compute_integral)(
            ABL, ILD, t_stim_arr, t_pts_for_integ, P_up_optim_fn, V_A, theta_A, x1, x2,
            rate_lambda, T_0, theta_E, Z_E, bound, K_max, t_A_aff, t_motor
        ) for (ABL, ILD) in tasks
    )
    integral_vs_t_stim = {}
    for (ABL, ILD), integrals in results:
        integral_vs_t_stim[(ABL, ILD)] = integrals



    # end_time = time.time()
    # print(f"Time taken: {end_time - start_time} seconds")


    all_loglike = Parallel(n_jobs=30)(delayed(compute_loglike)(row, integral_vs_t_stim)\
                                       for _, row in df_1.iterrows() if (row['timed_fix'] > row['intended_fix']) \
                                        and (row['response_poke'] in [2,3]))

    loglike = np.sum(all_loglike)
    return loglike

# %% [markdown]
# ## Bounds

# %%
rate_lambda_bounds = [0.01, 0.2]
theta_E_bounds = [30, 60]
T_0_bounds = [0.1*(1e-3), 1*(1e-3)]


# t_E_aff_bounds = [0.001, 0.1]
Z_E_bounds = [-10, 10]
# L_bounds = [0.1, 1.99]

# ---
rate_lambda_plausible_bounds =  [0.05, 0.09]
T_0_plausible_bounds = [0.15*(1e-3), 0.5*(1e-3)]
theta_E_plausible_bounds = [40, 55]

# t_E_aff_plausible_bounds = [0.01, 0.05]
Z_E_plausible_bounds = [-5, 5]
# L_plausible_bounds = [0.5, 1.5]

# %% [markdown]
# ## prior

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
    

def psiam_tied_prior_fn(params):
    rate_lambda, T_0, theta_E, Z_E = params

    rate_lambda_logpdf = trapezoidal_logpdf(rate_lambda, rate_lambda_bounds[0], rate_lambda_plausible_bounds[0], rate_lambda_plausible_bounds[1], rate_lambda_bounds[1])
    theta_E_logpdf = trapezoidal_logpdf(theta_E, theta_E_bounds[0], theta_E_plausible_bounds[0], theta_E_plausible_bounds[1], theta_E_bounds[1])
    T_0_logpdf = trapezoidal_logpdf(T_0, T_0_bounds[0], T_0_plausible_bounds[0], T_0_plausible_bounds[1], T_0_bounds[1])
    
    Z_E_logpdf = trapezoidal_logpdf(Z_E, Z_E_bounds[0], Z_E_plausible_bounds[0], Z_E_plausible_bounds[1], Z_E_bounds[1])
    # L_logpdf = trapezoidal_logpdf(L, L_bounds[0], L_plausible_bounds[0], L_plausible_bounds[1], L_bounds[1])

    return rate_lambda_logpdf + T_0_logpdf + theta_E_logpdf + Z_E_logpdf


# %% [markdown]
# ## prior + loglike

# %%
def psiam_tied_joint_fn(params):
    priors = psiam_tied_prior_fn(params) 
    loglike = psiam_tied_loglike_fn(params)

    joint = priors + loglike
    return joint

# %% [markdown]
# ## run vbmc

# %%
lb = np.array([ rate_lambda_bounds[0], T_0_bounds[0], theta_E_bounds[0], \
                Z_E_bounds[0]])
ub = np.array([ rate_lambda_bounds[1], T_0_bounds[1], theta_E_bounds[1], \
                 Z_E_bounds[1]])

plb = np.array([ rate_lambda_plausible_bounds[0], T_0_plausible_bounds[0], theta_E_plausible_bounds[0], \
                 Z_E_plausible_bounds[0]])

pub = np.array([rate_lambda_plausible_bounds[1], T_0_plausible_bounds[1], theta_E_plausible_bounds[1], \
                 Z_E_plausible_bounds[1]])


np.random.seed(43)
rate_lambda_0 = np.random.uniform(rate_lambda_plausible_bounds[0], rate_lambda_plausible_bounds[1])
T_0_0 = np.random.uniform(T_0_plausible_bounds[0], T_0_plausible_bounds[1])
theta_E_0 = np.random.uniform(theta_E_plausible_bounds[0], theta_E_plausible_bounds[1])

Z_E_0 = np.random.uniform(Z_E_plausible_bounds[0], Z_E_plausible_bounds[1])
# L_0 = np.random.uniform(L_plausible_bounds[0], L_plausible_bounds[1])

x_0 = np.array([rate_lambda_0, T_0_0, theta_E_0, Z_E_0])

vbmc = VBMC(psiam_tied_joint_fn, x_0, lb, ub, plb, pub, options={'display': 'on'})
vp, results = vbmc.optimize()

# %%
vbmc.save('INTP_PSYCHO_choice_all_trials.pkl', overwrite=True)

