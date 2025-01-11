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
import multiprocessing
from psiam_tied_no_dv_map_utils import cum_A_t_fn
from psiam_tied_no_dv_map_utils import rho_A_t_fn, cum_A_t_fn, rho_E_minus_small_t_NORM_fn
from psiam_tied_dv_map_utils import cum_E_t_fn

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
df_1 = df_1[ df_1['timed_fix'] > df_1['intended_fix'] ]

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
# t_A_aff = 0.05 # NOTE: TEMP, to test if negative afferent delay is causing VBMC to not converge

# %%
t_A_aff

# %% [markdown]
# # VBMC

# %% [markdown]
# ## loglike fn

# %%
T_trunc_right_wrt_stim = 0.21
T_trunc_left_aborts = 0.3

# %%
def cum_psiam_tied_fn(t, V_A, theta_A, t_A_aff, t_motor, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_stim, t_E_aff, K_max):
    c_A  = cum_A_t_fn(t - t_A_aff - t_motor, V_A, theta_A)
    c_E = cum_E_t_fn(t - t_E_aff - t_motor - t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, K_max)
    return c_A + c_E - (c_A * c_E)

# %%


def compute_loglike(row, rate_lambda, T_0, theta_E, t_E_aff, Z_E, L):
    timed_fix = row['timed_fix']
    intended_fix = row['intended_fix']
    
    ILD = row['ILD']
    ABL = row['ABL']
    # choice = row['response_poke']

    rt = timed_fix
    t_stim = intended_fix
    
    K_max = 10
    
    trunc_time = max(t_stim, T_trunc_left_aborts)
    trunc_factor = 1 - cum_A_t_fn(trunc_time - t_A_aff - t_motor, V_A, theta_A) + 1e-10

    if rt - t_stim > T_trunc_right_wrt_stim:
        cum_inf = 1
        cum_trunc = cum_psiam_tied_fn(T_trunc_right_wrt_stim + t_stim, V_A, theta_A, t_A_aff, t_motor, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_stim, t_E_aff, K_max)
        likelihood = cum_inf - cum_trunc
    else:
        # fit shape only
        p_a = rho_A_t_fn(rt - t_A_aff - t_motor, V_A, theta_A)
        c_e = cum_E_t_fn(rt - t_E_aff - t_motor - t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, K_max)
        p_e = rho_E_minus_small_t_NORM_fn(rt - t_E_aff - t_motor - t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, 1, K_max) \
                + rho_E_minus_small_t_NORM_fn(rt - t_E_aff - t_motor - t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, -1, K_max) 
        c_a = cum_A_t_fn(rt - t_A_aff - t_motor, V_A, theta_A)
        likelihood = p_a * (1 - c_e) + p_e * (1 - c_a)
        

    likelihood /= trunc_factor

    if likelihood <= 0:
        likelihood = 1e-50

    return np.log(likelihood)    


def psiam_tied_loglike_fn(params):
    rate_lambda, T_0, theta_E, t_E_aff, Z_E, L = params


    all_loglike = Parallel(n_jobs=30)(delayed(compute_loglike)(row, rate_lambda, T_0, theta_E, t_E_aff, Z_E, L)\
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


t_E_aff_bounds = [0.001, 0.1]
Z_E_bounds = [-10, 10]
L_bounds = [0.1, 1.99]

# ---
rate_lambda_plausible_bounds =  [0.05, 0.09]
T_0_plausible_bounds = [0.15*(1e-3), 0.5*(1e-3)]
theta_E_plausible_bounds = [40, 55]

t_E_aff_plausible_bounds = [0.01, 0.05]
Z_E_plausible_bounds = [-5, 5]
L_plausible_bounds = [0.5, 1.5]

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
    rate_lambda, T_0, theta_E, t_E_aff, Z_E, L = params

    rate_lambda_logpdf = trapezoidal_logpdf(rate_lambda, rate_lambda_bounds[0], rate_lambda_plausible_bounds[0], rate_lambda_plausible_bounds[1], rate_lambda_bounds[1])
    theta_E_logpdf = trapezoidal_logpdf(theta_E, theta_E_bounds[0], theta_E_plausible_bounds[0], theta_E_plausible_bounds[1], theta_E_bounds[1])
    T_0_logpdf = trapezoidal_logpdf(T_0, T_0_bounds[0], T_0_plausible_bounds[0], T_0_plausible_bounds[1], T_0_bounds[1])
    
    t_E_aff_logpdf = trapezoidal_logpdf(t_E_aff, t_E_aff_bounds[0], t_E_aff_plausible_bounds[0], t_E_aff_plausible_bounds[1], t_E_aff_bounds[1])
    Z_E_logpdf = trapezoidal_logpdf(Z_E, Z_E_bounds[0], Z_E_plausible_bounds[0], Z_E_plausible_bounds[1], Z_E_bounds[1])
    L_logpdf = trapezoidal_logpdf(L, L_bounds[0], L_plausible_bounds[0], L_plausible_bounds[1], L_bounds[1])

    return rate_lambda_logpdf + T_0_logpdf + theta_E_logpdf + t_E_aff_logpdf + Z_E_logpdf + L_logpdf


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
               t_E_aff_bounds[0], Z_E_bounds[0], L_bounds[0]])
ub = np.array([ rate_lambda_bounds[1], T_0_bounds[1], theta_E_bounds[1], \
                t_E_aff_bounds[1], Z_E_bounds[1], L_bounds[1]])

plb = np.array([ rate_lambda_plausible_bounds[0], T_0_plausible_bounds[0], theta_E_plausible_bounds[0], \
                t_E_aff_plausible_bounds[0], Z_E_plausible_bounds[0], L_plausible_bounds[0]])

pub = np.array([rate_lambda_plausible_bounds[1], T_0_plausible_bounds[1], theta_E_plausible_bounds[1], \
                t_E_aff_plausible_bounds[1], Z_E_plausible_bounds[1], L_plausible_bounds[1]])


np.random.seed(42)
rate_lambda_0 = np.random.uniform(rate_lambda_plausible_bounds[0], rate_lambda_plausible_bounds[1])
T_0_0 = np.random.uniform(T_0_plausible_bounds[0], T_0_plausible_bounds[1])
theta_E_0 = np.random.uniform(theta_E_plausible_bounds[0], theta_E_plausible_bounds[1])

t_E_aff_0 = np.random.uniform(t_E_aff_plausible_bounds[0], t_E_aff_plausible_bounds[1])
Z_E_0 = np.random.uniform(Z_E_plausible_bounds[0], Z_E_plausible_bounds[1])
L_0 = np.random.uniform(L_plausible_bounds[0], L_plausible_bounds[1])

x_0 = np.array([rate_lambda_0, T_0_0, theta_E_0, t_E_aff_0, Z_E_0, L_0])

vbmc = VBMC(psiam_tied_joint_fn, x_0, lb, ub, plb, pub, options={'display': 'on'})
vp, results = vbmc.optimize()

# %%
vbmc.save('shape_only.pkl', overwrite=True)