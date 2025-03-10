# %%
import numpy as np
import matplotlib.pyplot as plt

from joblib import Parallel, delayed, parallel_backend
from psiam_tied_dv_map_utils_for_noise import psiam_tied_data_gen_wrapper_noise_change_no_L_T0_change
from psiam_tied_dv_map_utils_with_PDFs import all_RTs_fit_OPTIM_V_A_change_added_noise_fn, up_RTs_fit_OPTIM_V_A_change_added_noise_fn, down_RTs_fit_OPTIM_V_A_change_added_noise_fn, PA_with_LEDON_2
from psiam_tied_dv_map_utils_with_PDFs import up_RTs_fit_OPTIM_V_A_change_added_noise_M3_delGO_fn, down_RTs_fit_OPTIM_V_A_change_added_noise_M3_delGO_fn
import pandas as pd
import random
from tqdm.notebook import tqdm
from joblib import Parallel, delayed
from scipy.integrate import trapezoid as trapz
from pyvbmc import VBMC
import corner
from diagnostics_class import Diagnostics

# %% [markdown]
# # data

# %%
# repeat_trial, T16, S7
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


#### LED OFF #####
df_led_off = df[df['LED_trial'] == 0]


### LED ON ###
df_led_on = df[df['LED_trial'] == 1]
df_led_on = df_led_on[df_led_on['LED_powerL'] == df_led_on['LED_powerR']]  # Bilateral, Left and right same power = 100

# %%
df_led_on_valid_trials = df_led_on[df_led_on['success'].isin([1,-1])]
print(f'len of led on valid trials = {len(df_led_on_valid_trials)}')

# %% [markdown]
# # loglike fn

# %%
# proactive
V_A = 1.6
theta_A = 2.53
V_A_post_LED = V_A + 1.8

# delays
t_A_aff = -0.187
t_E_aff = 0.075
Z_E = 0
del_go = 0.12


K_max = 10

# %%
def compute_loglike_trial(row, rate_lambda, T_0, theta_E, noise):
    # data
    rt = row['timed_fix']
    t_stim = row['intended_fix']
    t_LED = row['intended_fix'] - row['LED_onset_time']

    
    ABL = row['ABL']
    ILD = row['ILD']

    response_poke = row['response_poke']
    
    t_pts = np.arange(0, t_stim, 0.001)
    P_A_LED_change = np.array([PA_with_LEDON_2(i, V_A, V_A_post_LED, theta_A, 0, t_LED, t_A_aff) for i in t_pts])
    CDF_PA_till_stim = trapz(P_A_LED_change, t_pts)
    trunc_factor = 1 - CDF_PA_till_stim

    if response_poke == 3:
        # up
        likelihood = up_RTs_fit_OPTIM_V_A_change_added_noise_M3_delGO_fn(rt, t_LED, V_A, V_A_post_LED, theta_A, \
                                                         ABL, ILD, rate_lambda, T_0, noise,\
                                                              theta_E, Z_E, t_stim, t_A_aff, t_E_aff, del_go, K_max)
    elif response_poke == 2:
        # down
        likelihood = down_RTs_fit_OPTIM_V_A_change_added_noise_M3_delGO_fn(rt, t_LED, V_A, V_A_post_LED, theta_A, \
                                                         ABL, ILD, rate_lambda, T_0, noise,\
                                                              theta_E, Z_E, t_stim, t_A_aff, t_E_aff, del_go, K_max)

    
    likelihood /= (trunc_factor + 1e-10)
    if likelihood <= 0:
        likelihood = 1e-50


    return np.log(likelihood)


def vbmc_loglike_fn(params):
    rate_lambda, T_0, theta_E, noise = params

    all_loglike = Parallel(n_jobs=30)(delayed(compute_loglike_trial)(row, rate_lambda, T_0, theta_E, noise) \
                                     for _, row in df_led_on_valid_trials.iterrows())
    
    return np.sum(all_loglike)

# %% [markdown]
# # bounds

# %%
rate_lambda_bounds = [0.01, 0.2]
rate_lambda_plausible_bounds =  [0.05, 0.15]

theta_E_bounds = [1, 90]
theta_E_plausible_bounds = [40, 50]

T_0_bounds = [0.1 * (1e-3), 5 * (1e-3)]
T_0_plausible_bounds = [0.5 * (1e-3), 2.5 * (1e-3)]

noise_bounds = [0, 100]
noise_plausible_bounds = [20, 80]

# %% [markdown]
# # priors

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
    rate_lambda, T_0, theta_E, noise = params

    rate_lambda_logpdf = trapezoidal_logpdf(rate_lambda, rate_lambda_bounds[0], rate_lambda_plausible_bounds[0], rate_lambda_plausible_bounds[1], rate_lambda_bounds[1])
    T_0_logpdf = trapezoidal_logpdf(T_0, T_0_bounds[0], T_0_plausible_bounds[0], T_0_plausible_bounds[1], T_0_bounds[1])
    theta_E_logpdf = trapezoidal_logpdf(theta_E, theta_E_bounds[0], theta_E_plausible_bounds[0], theta_E_plausible_bounds[1], theta_E_bounds[1])
    noise_logpdf = trapezoidal_logpdf(noise, noise_bounds[0], noise_plausible_bounds[0], noise_plausible_bounds[1], noise_bounds[1])

    return rate_lambda_logpdf + T_0_logpdf + theta_E_logpdf + noise_logpdf

# %% [markdown]
# # prior + loglike

# %%
def vbmc_joint_fn(params):
    priors = vbmc_prior_fn(params)
    loglike = vbmc_loglike_fn(params)

    return priors + loglike

# %% [markdown]
# # run vbmc

# %%
# lb = np.array([T_0_bounds[0], noise_bounds[0]])
# ub = np.array([T_0_bounds[1], noise_bounds[1]])

# plb = np.array([T_0_plausible_bounds[0], noise_plausible_bounds[0]])
# pub = np.array([T_0_plausible_bounds[1], noise_plausible_bounds[1]])


lb = np.array([rate_lambda_bounds[0], T_0_bounds[0], theta_E_bounds[0], noise_bounds[0]])
ub = np.array([rate_lambda_bounds[1], T_0_bounds[1], theta_E_bounds[1], noise_bounds[1]])

plb = np.array([rate_lambda_plausible_bounds[0], T_0_plausible_bounds[0], theta_E_plausible_bounds[0], noise_plausible_bounds[0]])
pub = np.array([rate_lambda_plausible_bounds[1], T_0_plausible_bounds[1], theta_E_plausible_bounds[1], noise_plausible_bounds[1]])
               
np.random.seed(42)
# T_0_0 = np.random.uniform(T_0_plausible_bounds[0], T_0_plausible_bounds[1])
# noise_0 = np.random.uniform(noise_plausible_bounds[0], noise_plausible_bounds[1])
rate_lambda_0 = np.random.uniform(rate_lambda_plausible_bounds[0], rate_lambda_plausible_bounds[1])
T_0_0 = np.random.uniform(T_0_plausible_bounds[0], T_0_plausible_bounds[1])
theta_E_0 = np.random.uniform(theta_E_plausible_bounds[0], theta_E_plausible_bounds[1])
noise_0 = np.random.uniform(noise_plausible_bounds[0], noise_plausible_bounds[1])


# x_0 = np.array([T_0_0, noise_0])
x_0 = np.array([rate_lambda_0, T_0_0, theta_E_0, noise_0])


vbmc = VBMC(vbmc_joint_fn, x_0, lb, ub, plb, pub, options={'display': 'on'})
vp, results = vbmc.optimize()

# %%


# %%
# vbmc.save('added_noise_vbmc.pkl', overwrite=True)

# %%
# import pickle 
# with open('added_noise_vbmc.pkl', 'rb') as f:
#     vp = pickle.load(f)


# vp = vp.vp
