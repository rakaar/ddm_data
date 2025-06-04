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
####### TEMP ###############
###### NOTE ############
# Fitting super rat ########
#######################
animal_with_largest_trials = df['animal'].value_counts().idxmax()
print(f'animal with largest number of trials: {animal_with_largest_trials}')
df = df[df['animal'] == animal_with_largest_trials]

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
ABLs_to_fit = [20, 60]
ILDs_to_fit = [1, 2, 4, 8, 16, -1, -2, -4, -8, -16]

df_led_off_valid_trials_cond_filtered = df_led_off_valid_trials[
    (df_led_off_valid_trials['ABL'].isin(ABLs_to_fit)) & 
    (df_led_off_valid_trials['ILD'].isin(ILDs_to_fit))
]
print(f'len of filtered trials: {len(df_led_off_valid_trials_cond_filtered)}')



# %%
print(f'lenm of conditioned trials = {len(df_led_off_valid_trials_cond_filtered)}')
ABLs_cond = df_led_off_valid_trials_cond_filtered['ABL'].unique()
ILDs_cond = df_led_off_valid_trials_cond_filtered['ILD'].unique()
print(ABLs_cond)
print(ILDs_cond)


# %%
# Proactive params
batch_name = 'LED7'
animal_id = '92'

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
### TEMP: NOTE for super rate #########
# V_A = 1.6
# theta_A = 2.5
# t_A_aff = -0.22

# other params
K_max = 10

# %%
# VBMC loglike
def compute_loglike_trial(row,
    gamma_ABL_20_ILD_1, gamma_ABL_20_ILD_2, gamma_ABL_20_ILD_4, gamma_ABL_20_ILD_8, gamma_ABL_20_ILD_16,
    gamma_ABL_60_ILD_1, gamma_ABL_60_ILD_2, gamma_ABL_60_ILD_4, gamma_ABL_60_ILD_8, gamma_ABL_60_ILD_16,
    omega_ABL_20_ILD_1, omega_ABL_20_ILD_2, omega_ABL_20_ILD_4, omega_ABL_20_ILD_8, omega_ABL_20_ILD_16,
    omega_ABL_60_ILD_1, omega_ABL_60_ILD_2, omega_ABL_60_ILD_4, omega_ABL_60_ILD_8, omega_ABL_60_ILD_16,
    t_E_aff, w, del_go):
    # data
    c_A_trunc_time = 0.3
    rt = row['timed_fix']
    t_stim = row['intended_fix']
    response_poke = row['response_poke']
    
    ABL = row['ABL']
    ILD = row['ILD']

    gamma = None
    omega = None
    abs_ILD = abs(ILD)
    if ABL == 20:
        if abs_ILD == 1:
            gamma = gamma_ABL_20_ILD_1
            omega = omega_ABL_20_ILD_1
        elif abs_ILD == 2:
            gamma = gamma_ABL_20_ILD_2
            omega = omega_ABL_20_ILD_2
        elif abs_ILD == 4:
            gamma = gamma_ABL_20_ILD_4
            omega = omega_ABL_20_ILD_4
        elif abs_ILD == 8:
            gamma = gamma_ABL_20_ILD_8
            omega = omega_ABL_20_ILD_8
        elif abs_ILD == 16:
            gamma = gamma_ABL_20_ILD_16
            omega = omega_ABL_20_ILD_16
    elif ABL == 60:
        if abs_ILD == 1:
            gamma = gamma_ABL_60_ILD_1
            omega = omega_ABL_60_ILD_1
        elif abs_ILD == 2:
            gamma = gamma_ABL_60_ILD_2
            omega = omega_ABL_60_ILD_2
        elif abs_ILD == 4:
            gamma = gamma_ABL_60_ILD_4
            omega = omega_ABL_60_ILD_4
        elif abs_ILD == 8:
            gamma = gamma_ABL_60_ILD_8
            omega = omega_ABL_60_ILD_8
        elif abs_ILD == 16:
            gamma = gamma_ABL_60_ILD_16
            omega = omega_ABL_60_ILD_16

    if gamma is not None:
        gamma *= np.sign(ILD)

    if gamma is None or omega is None:
        print(f"gamma or omega is None for ABL {ABL}, ILD {ILD}")
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
    # gamma_ABL_x_ILD_y: x is [20,60]. y is [1,2,4,8,16]
    # omega_ABL_x_ILD_y: x is [20,60]. y is [1,2,4,8,16]
    (
        gamma_ABL_20_ILD_1, gamma_ABL_20_ILD_2, gamma_ABL_20_ILD_4, gamma_ABL_20_ILD_8, gamma_ABL_20_ILD_16,
        gamma_ABL_60_ILD_1, gamma_ABL_60_ILD_2, gamma_ABL_60_ILD_4, gamma_ABL_60_ILD_8, gamma_ABL_60_ILD_16,
        omega_ABL_20_ILD_1, omega_ABL_20_ILD_2, omega_ABL_20_ILD_4, omega_ABL_20_ILD_8, omega_ABL_20_ILD_16,
        omega_ABL_60_ILD_1, omega_ABL_60_ILD_2, omega_ABL_60_ILD_4, omega_ABL_60_ILD_8, omega_ABL_60_ILD_16,
        t_E_aff, w, del_go
    ) = params

    all_loglike = Parallel(n_jobs=30)(delayed(compute_loglike_trial)(
        row,
        gamma_ABL_20_ILD_1, gamma_ABL_20_ILD_2, gamma_ABL_20_ILD_4, gamma_ABL_20_ILD_8, gamma_ABL_20_ILD_16,
        gamma_ABL_60_ILD_1, gamma_ABL_60_ILD_2, gamma_ABL_60_ILD_4, gamma_ABL_60_ILD_8, gamma_ABL_60_ILD_16,
        omega_ABL_20_ILD_1, omega_ABL_20_ILD_2, omega_ABL_20_ILD_4, omega_ABL_20_ILD_8, omega_ABL_20_ILD_16,
        omega_ABL_60_ILD_1, omega_ABL_60_ILD_2, omega_ABL_60_ILD_4, omega_ABL_60_ILD_8, omega_ABL_60_ILD_16,
        t_E_aff, w, del_go
    ) for _, row in df_led_off_valid_trials_cond_filtered.iterrows())

    return np.sum(all_loglike)

# %% [markdown]
# # bounds

# %%
# gamma_bounds = [0.02, 2]
# gamma_plausible_bounds = [0.09, 0.9]
# gamma bounds

# For explicitness, also assign to local variables (for later use)
gamma_ABL_20_ILD_1_bounds = [0.001, 5]
gamma_ABL_20_ILD_2_bounds = [0.001, 5]
gamma_ABL_20_ILD_4_bounds = [0.001, 5]
gamma_ABL_20_ILD_8_bounds = [0.001, 5]
gamma_ABL_20_ILD_16_bounds = [0.001, 5]
gamma_ABL_60_ILD_1_bounds = [0.001, 5]
gamma_ABL_60_ILD_2_bounds = [0.001, 5]
gamma_ABL_60_ILD_4_bounds = [0.001, 5]
gamma_ABL_60_ILD_8_bounds = [0.001, 5]
gamma_ABL_60_ILD_16_bounds = [0.001, 5]

gamma_ABL_20_ILD_1_plausible_bounds = [0.01, 3]
gamma_ABL_20_ILD_2_plausible_bounds = [0.01, 3]
gamma_ABL_20_ILD_4_plausible_bounds = [0.01, 3]
gamma_ABL_20_ILD_8_plausible_bounds = [0.01, 3]
gamma_ABL_20_ILD_16_plausible_bounds = [0.01, 3]
gamma_ABL_60_ILD_1_plausible_bounds = [0.01, 3]
gamma_ABL_60_ILD_2_plausible_bounds = [0.01, 3]
gamma_ABL_60_ILD_4_plausible_bounds = [0.01, 3]
gamma_ABL_60_ILD_8_plausible_bounds = [0.01, 3]
gamma_ABL_60_ILD_16_plausible_bounds = [0.01, 3]

omega_ABL_20_ILD_1_bounds = [0.05, 50]
omega_ABL_20_ILD_2_bounds = [0.05, 50]
omega_ABL_20_ILD_4_bounds = [0.05, 50]
omega_ABL_20_ILD_8_bounds = [0.05, 50]
omega_ABL_20_ILD_16_bounds = [0.05, 50]
omega_ABL_60_ILD_1_bounds = [0.05, 50]
omega_ABL_60_ILD_2_bounds = [0.05, 50]
omega_ABL_60_ILD_4_bounds = [0.05, 50]
omega_ABL_60_ILD_8_bounds = [0.05, 50]
omega_ABL_60_ILD_16_bounds = [0.05, 50]

omega_ABL_20_ILD_1_plausible_bounds = [0.5, 10]
omega_ABL_20_ILD_2_plausible_bounds = [0.5, 10]
omega_ABL_20_ILD_4_plausible_bounds = [0.5, 10]
omega_ABL_20_ILD_8_plausible_bounds = [0.5, 10]
omega_ABL_20_ILD_16_plausible_bounds = [0.5, 10]
omega_ABL_60_ILD_1_plausible_bounds = [0.5, 10]
omega_ABL_60_ILD_2_plausible_bounds = [0.5, 10]
omega_ABL_60_ILD_4_plausible_bounds = [0.5, 10]
omega_ABL_60_ILD_8_plausible_bounds = [0.5, 10]
omega_ABL_60_ILD_16_plausible_bounds = [0.5, 10]


t_E_aff_bounds = [0, 1]
t_E_aff_plausible_bounds = [0.01, 0.2]

w_bounds = [0.2, 0.8]
w_plausible_bounds = [0.3, 0.7]

del_go_bounds = [0.001, 0.2]
del_go_plausible_bounds = [0.11, 0.15]

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
        gamma_ABL_20_ILD_1, gamma_ABL_20_ILD_2, gamma_ABL_20_ILD_4, gamma_ABL_20_ILD_8, gamma_ABL_20_ILD_16,
        gamma_ABL_60_ILD_1, gamma_ABL_60_ILD_2, gamma_ABL_60_ILD_4, gamma_ABL_60_ILD_8, gamma_ABL_60_ILD_16,
        omega_ABL_20_ILD_1, omega_ABL_20_ILD_2, omega_ABL_20_ILD_4, omega_ABL_20_ILD_8, omega_ABL_20_ILD_16,
        omega_ABL_60_ILD_1, omega_ABL_60_ILD_2, omega_ABL_60_ILD_4, omega_ABL_60_ILD_8, omega_ABL_60_ILD_16,
        t_E_aff, w, del_go
    ) = params

    # Compute logpdf for all gamma and omega parameters
    gamma_logpdfs = [
        trapezoidal_logpdf(gamma_ABL_20_ILD_1, gamma_ABL_20_ILD_1_bounds[0], gamma_ABL_20_ILD_1_plausible_bounds[0], gamma_ABL_20_ILD_1_plausible_bounds[1], gamma_ABL_20_ILD_1_bounds[1]),
        trapezoidal_logpdf(gamma_ABL_20_ILD_2, gamma_ABL_20_ILD_2_bounds[0], gamma_ABL_20_ILD_2_plausible_bounds[0], gamma_ABL_20_ILD_2_plausible_bounds[1], gamma_ABL_20_ILD_2_bounds[1]),
        trapezoidal_logpdf(gamma_ABL_20_ILD_4, gamma_ABL_20_ILD_4_bounds[0], gamma_ABL_20_ILD_4_plausible_bounds[0], gamma_ABL_20_ILD_4_plausible_bounds[1], gamma_ABL_20_ILD_4_bounds[1]),
        trapezoidal_logpdf(gamma_ABL_20_ILD_8, gamma_ABL_20_ILD_8_bounds[0], gamma_ABL_20_ILD_8_plausible_bounds[0], gamma_ABL_20_ILD_8_plausible_bounds[1], gamma_ABL_20_ILD_8_bounds[1]),
        trapezoidal_logpdf(gamma_ABL_20_ILD_16, gamma_ABL_20_ILD_16_bounds[0], gamma_ABL_20_ILD_16_plausible_bounds[0], gamma_ABL_20_ILD_16_plausible_bounds[1], gamma_ABL_20_ILD_16_bounds[1]),
        trapezoidal_logpdf(gamma_ABL_60_ILD_1, gamma_ABL_60_ILD_1_bounds[0], gamma_ABL_60_ILD_1_plausible_bounds[0], gamma_ABL_60_ILD_1_plausible_bounds[1], gamma_ABL_60_ILD_1_bounds[1]),
        trapezoidal_logpdf(gamma_ABL_60_ILD_2, gamma_ABL_60_ILD_2_bounds[0], gamma_ABL_60_ILD_2_plausible_bounds[0], gamma_ABL_60_ILD_2_plausible_bounds[1], gamma_ABL_60_ILD_2_bounds[1]),
        trapezoidal_logpdf(gamma_ABL_60_ILD_4, gamma_ABL_60_ILD_4_bounds[0], gamma_ABL_60_ILD_4_plausible_bounds[0], gamma_ABL_60_ILD_4_plausible_bounds[1], gamma_ABL_60_ILD_4_bounds[1]),
        trapezoidal_logpdf(gamma_ABL_60_ILD_8, gamma_ABL_60_ILD_8_bounds[0], gamma_ABL_60_ILD_8_plausible_bounds[0], gamma_ABL_60_ILD_8_plausible_bounds[1], gamma_ABL_60_ILD_8_bounds[1]),
        trapezoidal_logpdf(gamma_ABL_60_ILD_16, gamma_ABL_60_ILD_16_bounds[0], gamma_ABL_60_ILD_16_plausible_bounds[0], gamma_ABL_60_ILD_16_plausible_bounds[1], gamma_ABL_60_ILD_16_bounds[1]),
    ]
    omega_logpdfs = [
        trapezoidal_logpdf(omega_ABL_20_ILD_1, omega_ABL_20_ILD_1_bounds[0], omega_ABL_20_ILD_1_plausible_bounds[0], omega_ABL_20_ILD_1_plausible_bounds[1], omega_ABL_20_ILD_1_bounds[1]),
        trapezoidal_logpdf(omega_ABL_20_ILD_2, omega_ABL_20_ILD_2_bounds[0], omega_ABL_20_ILD_2_plausible_bounds[0], omega_ABL_20_ILD_2_plausible_bounds[1], omega_ABL_20_ILD_2_bounds[1]),
        trapezoidal_logpdf(omega_ABL_20_ILD_4, omega_ABL_20_ILD_4_bounds[0], omega_ABL_20_ILD_4_plausible_bounds[0], omega_ABL_20_ILD_4_plausible_bounds[1], omega_ABL_20_ILD_4_bounds[1]),
        trapezoidal_logpdf(omega_ABL_20_ILD_8, omega_ABL_20_ILD_8_bounds[0], omega_ABL_20_ILD_8_plausible_bounds[0], omega_ABL_20_ILD_8_plausible_bounds[1], omega_ABL_20_ILD_8_bounds[1]),
        trapezoidal_logpdf(omega_ABL_20_ILD_16, omega_ABL_20_ILD_16_bounds[0], omega_ABL_20_ILD_16_plausible_bounds[0], omega_ABL_20_ILD_16_plausible_bounds[1], omega_ABL_20_ILD_16_bounds[1]),
        trapezoidal_logpdf(omega_ABL_60_ILD_1, omega_ABL_60_ILD_1_bounds[0], omega_ABL_60_ILD_1_plausible_bounds[0], omega_ABL_60_ILD_1_plausible_bounds[1], omega_ABL_60_ILD_1_bounds[1]),
        trapezoidal_logpdf(omega_ABL_60_ILD_2, omega_ABL_60_ILD_2_bounds[0], omega_ABL_60_ILD_2_plausible_bounds[0], omega_ABL_60_ILD_2_plausible_bounds[1], omega_ABL_60_ILD_2_bounds[1]),
        trapezoidal_logpdf(omega_ABL_60_ILD_4, omega_ABL_60_ILD_4_bounds[0], omega_ABL_60_ILD_4_plausible_bounds[0], omega_ABL_60_ILD_4_plausible_bounds[1], omega_ABL_60_ILD_4_bounds[1]),
        trapezoidal_logpdf(omega_ABL_60_ILD_8, omega_ABL_60_ILD_8_bounds[0], omega_ABL_60_ILD_8_plausible_bounds[0], omega_ABL_60_ILD_8_plausible_bounds[1], omega_ABL_60_ILD_8_bounds[1]),
        trapezoidal_logpdf(omega_ABL_60_ILD_16, omega_ABL_60_ILD_16_bounds[0], omega_ABL_60_ILD_16_plausible_bounds[0], omega_ABL_60_ILD_16_plausible_bounds[1], omega_ABL_60_ILD_16_bounds[1]),
    ]
    t_E_aff_logpdf = trapezoidal_logpdf(t_E_aff, t_E_aff_bounds[0], t_E_aff_plausible_bounds[0], t_E_aff_plausible_bounds[1], t_E_aff_bounds[1])
    w_logpdf = trapezoidal_logpdf(w, w_bounds[0], w_plausible_bounds[0], w_plausible_bounds[1], w_bounds[1])
    del_go_logpdf = trapezoidal_logpdf(del_go, del_go_bounds[0], del_go_plausible_bounds[0], del_go_plausible_bounds[1], del_go_bounds[1])
    return sum(gamma_logpdfs) + sum(omega_logpdfs) + t_E_aff_logpdf + w_logpdf + del_go_logpdf

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
# lb = np.array([gamma_bounds[0], omega_bounds[0], t_E_aff_bounds[0]])
# ub = np.array([gamma_bounds[1], omega_bounds[1], t_E_aff_bounds[1]])

# plb = np.array([gamma_plausible_bounds[0], omega_plausible_bounds[0], t_E_aff_plausible_bounds[0]])
# pub = np.array([gamma_plausible_bounds[1], omega_plausible_bounds[1], t_E_aff_plausible_bounds[1]])
lb = np.array([
    gamma_ABL_20_ILD_1_bounds[0], gamma_ABL_20_ILD_2_bounds[0], gamma_ABL_20_ILD_4_bounds[0], gamma_ABL_20_ILD_8_bounds[0], gamma_ABL_20_ILD_16_bounds[0],
    gamma_ABL_60_ILD_1_bounds[0], gamma_ABL_60_ILD_2_bounds[0], gamma_ABL_60_ILD_4_bounds[0], gamma_ABL_60_ILD_8_bounds[0], gamma_ABL_60_ILD_16_bounds[0],
    omega_ABL_20_ILD_1_bounds[0], omega_ABL_20_ILD_2_bounds[0], omega_ABL_20_ILD_4_bounds[0], omega_ABL_20_ILD_8_bounds[0], omega_ABL_20_ILD_16_bounds[0],
    omega_ABL_60_ILD_1_bounds[0], omega_ABL_60_ILD_2_bounds[0], omega_ABL_60_ILD_4_bounds[0], omega_ABL_60_ILD_8_bounds[0], omega_ABL_60_ILD_16_bounds[0],
    t_E_aff_bounds[0], w_bounds[0], del_go_bounds[0]
])

ub = np.array([
    gamma_ABL_20_ILD_1_bounds[1], gamma_ABL_20_ILD_2_bounds[1], gamma_ABL_20_ILD_4_bounds[1], gamma_ABL_20_ILD_8_bounds[1], gamma_ABL_20_ILD_16_bounds[1],
    gamma_ABL_60_ILD_1_bounds[1], gamma_ABL_60_ILD_2_bounds[1], gamma_ABL_60_ILD_4_bounds[1], gamma_ABL_60_ILD_8_bounds[1], gamma_ABL_60_ILD_16_bounds[1],
    omega_ABL_20_ILD_1_bounds[1], omega_ABL_20_ILD_2_bounds[1], omega_ABL_20_ILD_4_bounds[1], omega_ABL_20_ILD_8_bounds[1], omega_ABL_20_ILD_16_bounds[1],
    omega_ABL_60_ILD_1_bounds[1], omega_ABL_60_ILD_2_bounds[1], omega_ABL_60_ILD_4_bounds[1], omega_ABL_60_ILD_8_bounds[1], omega_ABL_60_ILD_16_bounds[1],
    t_E_aff_bounds[1], w_bounds[1], del_go_bounds[1]
])

plb = np.array([
    gamma_ABL_20_ILD_1_plausible_bounds[0], gamma_ABL_20_ILD_2_plausible_bounds[0], gamma_ABL_20_ILD_4_plausible_bounds[0], gamma_ABL_20_ILD_8_plausible_bounds[0], gamma_ABL_20_ILD_16_plausible_bounds[0],
    gamma_ABL_60_ILD_1_plausible_bounds[0], gamma_ABL_60_ILD_2_plausible_bounds[0], gamma_ABL_60_ILD_4_plausible_bounds[0], gamma_ABL_60_ILD_8_plausible_bounds[0], gamma_ABL_60_ILD_16_plausible_bounds[0],
    omega_ABL_20_ILD_1_plausible_bounds[0], omega_ABL_20_ILD_2_plausible_bounds[0], omega_ABL_20_ILD_4_plausible_bounds[0], omega_ABL_20_ILD_8_plausible_bounds[0], omega_ABL_20_ILD_16_plausible_bounds[0],
    omega_ABL_60_ILD_1_plausible_bounds[0], omega_ABL_60_ILD_2_plausible_bounds[0], omega_ABL_60_ILD_4_plausible_bounds[0], omega_ABL_60_ILD_8_plausible_bounds[0], omega_ABL_60_ILD_16_plausible_bounds[0],
    t_E_aff_plausible_bounds[0], w_plausible_bounds[0], del_go_plausible_bounds[0]
])

pub = np.array([
    gamma_ABL_20_ILD_1_plausible_bounds[1], gamma_ABL_20_ILD_2_plausible_bounds[1], gamma_ABL_20_ILD_4_plausible_bounds[1], gamma_ABL_20_ILD_8_plausible_bounds[1], gamma_ABL_20_ILD_16_plausible_bounds[1],
    gamma_ABL_60_ILD_1_plausible_bounds[1], gamma_ABL_60_ILD_2_plausible_bounds[1], gamma_ABL_60_ILD_4_plausible_bounds[1], gamma_ABL_60_ILD_8_plausible_bounds[1], gamma_ABL_60_ILD_16_plausible_bounds[1],
    omega_ABL_20_ILD_1_plausible_bounds[1], omega_ABL_20_ILD_2_plausible_bounds[1], omega_ABL_20_ILD_4_plausible_bounds[1], omega_ABL_20_ILD_8_plausible_bounds[1], omega_ABL_20_ILD_16_plausible_bounds[1],
    omega_ABL_60_ILD_1_plausible_bounds[1], omega_ABL_60_ILD_2_plausible_bounds[1], omega_ABL_60_ILD_4_plausible_bounds[1], omega_ABL_60_ILD_8_plausible_bounds[1], omega_ABL_60_ILD_16_plausible_bounds[1],
    t_E_aff_plausible_bounds[1], w_plausible_bounds[1], del_go_plausible_bounds[1]
])

# Initialize with random values within plausible bounds
np.random.seed(42)
gamma_ABL_20_ILD_1_0 = np.random.uniform(gamma_ABL_20_ILD_1_plausible_bounds[0], gamma_ABL_20_ILD_1_plausible_bounds[1])
gamma_ABL_20_ILD_2_0 = np.random.uniform(gamma_ABL_20_ILD_2_plausible_bounds[0], gamma_ABL_20_ILD_2_plausible_bounds[1])
gamma_ABL_20_ILD_4_0 = np.random.uniform(gamma_ABL_20_ILD_4_plausible_bounds[0], gamma_ABL_20_ILD_4_plausible_bounds[1])
gamma_ABL_20_ILD_8_0 = np.random.uniform(gamma_ABL_20_ILD_8_plausible_bounds[0], gamma_ABL_20_ILD_8_plausible_bounds[1])
gamma_ABL_20_ILD_16_0 = np.random.uniform(gamma_ABL_20_ILD_16_plausible_bounds[0], gamma_ABL_20_ILD_16_plausible_bounds[1])
gamma_ABL_60_ILD_1_0 = np.random.uniform(gamma_ABL_60_ILD_1_plausible_bounds[0], gamma_ABL_60_ILD_1_plausible_bounds[1])
gamma_ABL_60_ILD_2_0 = np.random.uniform(gamma_ABL_60_ILD_2_plausible_bounds[0], gamma_ABL_60_ILD_2_plausible_bounds[1])
gamma_ABL_60_ILD_4_0 = np.random.uniform(gamma_ABL_60_ILD_4_plausible_bounds[0], gamma_ABL_60_ILD_4_plausible_bounds[1])
gamma_ABL_60_ILD_8_0 = np.random.uniform(gamma_ABL_60_ILD_8_plausible_bounds[0], gamma_ABL_60_ILD_8_plausible_bounds[1])
gamma_ABL_60_ILD_16_0 = np.random.uniform(gamma_ABL_60_ILD_16_plausible_bounds[0], gamma_ABL_60_ILD_16_plausible_bounds[1])
omega_ABL_20_ILD_1_0 = np.random.uniform(omega_ABL_20_ILD_1_plausible_bounds[0], omega_ABL_20_ILD_1_plausible_bounds[1])
omega_ABL_20_ILD_2_0 = np.random.uniform(omega_ABL_20_ILD_2_plausible_bounds[0], omega_ABL_20_ILD_2_plausible_bounds[1])
omega_ABL_20_ILD_4_0 = np.random.uniform(omega_ABL_20_ILD_4_plausible_bounds[0], omega_ABL_20_ILD_4_plausible_bounds[1])
omega_ABL_20_ILD_8_0 = np.random.uniform(omega_ABL_20_ILD_8_plausible_bounds[0], omega_ABL_20_ILD_8_plausible_bounds[1])
omega_ABL_20_ILD_16_0 = np.random.uniform(omega_ABL_20_ILD_16_plausible_bounds[0], omega_ABL_20_ILD_16_plausible_bounds[1])
omega_ABL_60_ILD_1_0 = np.random.uniform(omega_ABL_60_ILD_1_plausible_bounds[0], omega_ABL_60_ILD_1_plausible_bounds[1])
omega_ABL_60_ILD_2_0 = np.random.uniform(omega_ABL_60_ILD_2_plausible_bounds[0], omega_ABL_60_ILD_2_plausible_bounds[1])
omega_ABL_60_ILD_4_0 = np.random.uniform(omega_ABL_60_ILD_4_plausible_bounds[0], omega_ABL_60_ILD_4_plausible_bounds[1])
omega_ABL_60_ILD_8_0 = np.random.uniform(omega_ABL_60_ILD_8_plausible_bounds[0], omega_ABL_60_ILD_8_plausible_bounds[1])
omega_ABL_60_ILD_16_0 = np.random.uniform(omega_ABL_60_ILD_16_plausible_bounds[0], omega_ABL_60_ILD_16_plausible_bounds[1])
t_E_aff_0 = np.random.uniform(t_E_aff_plausible_bounds[0], t_E_aff_plausible_bounds[1])
w_0 = np.random.uniform(w_plausible_bounds[0], w_plausible_bounds[1])
del_go_0 = np.random.uniform(del_go_plausible_bounds[0], del_go_plausible_bounds[1])

x_0 = np.array([
    gamma_ABL_20_ILD_1_0, gamma_ABL_20_ILD_2_0, gamma_ABL_20_ILD_4_0, gamma_ABL_20_ILD_8_0, gamma_ABL_20_ILD_16_0,
    gamma_ABL_60_ILD_1_0, gamma_ABL_60_ILD_2_0, gamma_ABL_60_ILD_4_0, gamma_ABL_60_ILD_8_0, gamma_ABL_60_ILD_16_0,
    omega_ABL_20_ILD_1_0, omega_ABL_20_ILD_2_0, omega_ABL_20_ILD_4_0, omega_ABL_20_ILD_8_0, omega_ABL_20_ILD_16_0,
    omega_ABL_60_ILD_1_0, omega_ABL_60_ILD_2_0, omega_ABL_60_ILD_4_0, omega_ABL_60_ILD_8_0, omega_ABL_60_ILD_16_0,
    t_E_aff_0, w_0, del_go_0
])

# Run VBMC
vbmc = VBMC(vbmc_joint_fn, x_0, lb, ub, plb, pub, options={'display': 'on'})
vp, results = vbmc.optimize()

# %%
vbmc.save(f'vbmc_mutiple_gama_omega_at_once_ILDs_1_2_4_8_16.pkl', overwrite=True)

# %%
# Load pickle
with open('vbmc_mutiple_gama_omega_at_once_ILDs_1_2_4_8_16.pkl', 'rb') as f:
    vp = pickle.load(f)

vp = vp.vp


# %%
vp_samples = vp.sample(int(1e5))[0]

# gamma_samples = vp_samples[:, 0]
# omega_samples = vp_samples[:, 1]
# t_E_aff_samples = vp_samples[:, 2]
gamma_ABL_20_ILD_1_samples = vp_samples[:, 0]
gamma_ABL_20_ILD_2_samples = vp_samples[:, 1]
gamma_ABL_20_ILD_4_samples = vp_samples[:, 2]
gamma_ABL_20_ILD_8_samples = vp_samples[:, 3]
gamma_ABL_20_ILD_16_samples = vp_samples[:, 4]
gamma_ABL_60_ILD_1_samples = vp_samples[:, 5]
gamma_ABL_60_ILD_2_samples = vp_samples[:, 6]
gamma_ABL_60_ILD_4_samples = vp_samples[:, 7]
gamma_ABL_60_ILD_8_samples = vp_samples[:, 8]
gamma_ABL_60_ILD_16_samples = vp_samples[:, 9]
omega_ABL_20_ILD_1_samples = vp_samples[:, 10]
omega_ABL_20_ILD_2_samples = vp_samples[:, 11]
omega_ABL_20_ILD_4_samples = vp_samples[:, 12]
omega_ABL_20_ILD_8_samples = vp_samples[:, 13]
omega_ABL_20_ILD_16_samples = vp_samples[:, 14]
omega_ABL_60_ILD_1_samples = vp_samples[:, 15]
omega_ABL_60_ILD_2_samples = vp_samples[:, 16]
omega_ABL_60_ILD_4_samples = vp_samples[:, 17]
omega_ABL_60_ILD_8_samples = vp_samples[:, 18]
omega_ABL_60_ILD_16_samples = vp_samples[:, 19]
t_E_aff_samples = vp_samples[:, 20]
w_samples = vp_samples[:, 21]
del_go_samples = vp_samples[:, 22]

# %%
gamma_ABL_20_ILD_1 = gamma_ABL_20_ILD_1_samples.mean()
gamma_ABL_20_ILD_2 = gamma_ABL_20_ILD_2_samples.mean()
gamma_ABL_20_ILD_4 = gamma_ABL_20_ILD_4_samples.mean()
gamma_ABL_20_ILD_8 = gamma_ABL_20_ILD_8_samples.mean()
gamma_ABL_20_ILD_16 = gamma_ABL_20_ILD_16_samples.mean()
gamma_ABL_60_ILD_1 = gamma_ABL_60_ILD_1_samples.mean()
gamma_ABL_60_ILD_2 = gamma_ABL_60_ILD_2_samples.mean()
gamma_ABL_60_ILD_4 = gamma_ABL_60_ILD_4_samples.mean()
gamma_ABL_60_ILD_8 = gamma_ABL_60_ILD_8_samples.mean()
gamma_ABL_60_ILD_16 = gamma_ABL_60_ILD_16_samples.mean()
omega_ABL_20_ILD_1 = omega_ABL_20_ILD_1_samples.mean()
omega_ABL_20_ILD_2 = omega_ABL_20_ILD_2_samples.mean()
omega_ABL_20_ILD_4 = omega_ABL_20_ILD_4_samples.mean()
omega_ABL_20_ILD_8 = omega_ABL_20_ILD_8_samples.mean()
omega_ABL_20_ILD_16 = omega_ABL_20_ILD_16_samples.mean()
omega_ABL_60_ILD_1 = omega_ABL_60_ILD_1_samples.mean()
omega_ABL_60_ILD_2 = omega_ABL_60_ILD_2_samples.mean()
omega_ABL_60_ILD_4 = omega_ABL_60_ILD_4_samples.mean()
omega_ABL_60_ILD_8 = omega_ABL_60_ILD_8_samples.mean()
omega_ABL_60_ILD_16 = omega_ABL_60_ILD_16_samples.mean()
t_E_aff = t_E_aff_samples.mean()
w = w_samples.mean()
del_go = del_go_samples.mean()

# %% [markdown]
# # corner

# %%
# plot the corner plot
corner_samples = np.vstack([
    gamma_ABL_20_ILD_1_samples, gamma_ABL_20_ILD_2_samples, gamma_ABL_20_ILD_4_samples, gamma_ABL_20_ILD_8_samples, gamma_ABL_20_ILD_16_samples,
    gamma_ABL_60_ILD_1_samples, gamma_ABL_60_ILD_2_samples, gamma_ABL_60_ILD_4_samples, gamma_ABL_60_ILD_8_samples, gamma_ABL_60_ILD_16_samples,
    omega_ABL_20_ILD_1_samples, omega_ABL_20_ILD_2_samples, omega_ABL_20_ILD_4_samples, omega_ABL_20_ILD_8_samples, omega_ABL_20_ILD_16_samples,
    omega_ABL_60_ILD_1_samples, omega_ABL_60_ILD_2_samples, omega_ABL_60_ILD_4_samples, omega_ABL_60_ILD_8_samples, omega_ABL_60_ILD_16_samples,
    t_E_aff_samples, w_samples, del_go_samples
]).T
percentiles = np.percentile(corner_samples, [0, 100], axis=0)
_ranges = [(percentiles[0, i], percentiles[1, i]) for i in np.arange(corner_samples.shape[1])]
param_labels = [
    'gamma_ABL_20_ILD_1', 'gamma_ABL_20_ILD_2', 'gamma_ABL_20_ILD_4', 'gamma_ABL_20_ILD_8', 'gamma_ABL_20_ILD_16',
    'gamma_ABL_60_ILD_1', 'gamma_ABL_60_ILD_2', 'gamma_ABL_60_ILD_4', 'gamma_ABL_60_ILD_8', 'gamma_ABL_60_ILD_16',
    'omega_ABL_20_ILD_1', 'omega_ABL_20_ILD_2', 'omega_ABL_20_ILD_4', 'omega_ABL_20_ILD_8', 'omega_ABL_20_ILD_16',
    'omega_ABL_60_ILD_1', 'omega_ABL_60_ILD_2', 'omega_ABL_60_ILD_4', 'omega_ABL_60_ILD_8', 'omega_ABL_60_ILD_16',
    't_E_aff', 'w', 'del_go']

corner.corner(
    corner_samples,
    labels=param_labels,
    show_titles=True,
    quantiles=[0.025, 0.5, 0.975],
    range=_ranges,
    title_fmt=".4f"
);



# %%
# Diagnostics - RTD choice
N_theory = int(1e3)
t_pts = np.arange(-1, 2, 0.001)
t_stim_samples = df_led_off_valid_trials_cond_filtered['intended_fix'].sample(N_theory, replace=True).values
P_A_samples = np.zeros((N_theory, len(t_pts)))
t_trunc = 0.3 # wrt fix
for idx, t_stim in enumerate(t_stim_samples):
    # t is wrt t_stim, t + t_stim is wrt fix
    # Vectorized version using rho_A_t_VEC_fn
    t_shifted = t_pts + t_stim
    mask = t_shifted > t_trunc
    vals = np.zeros_like(t_pts)
    if np.any(mask):
        vals[mask] = rho_A_t_VEC_fn(t_shifted[mask] - t_A_aff, V_A, theta_A)
    P_A_samples[idx, :] = vals


from scipy.integrate import trapezoid
P_A_mean = np.mean(P_A_samples, axis=0)
area = trapezoid(P_A_mean, t_pts)

if area != 0:
    P_A_mean = P_A_mean / area
C_A_mean = cumtrapz(P_A_mean, t_pts, initial=0)


# %%
# --- Compute and store theory/data for all ABL/ILD combinations ---
theory_curves = {}  # (ABL, ILD): dict with up_mean, down_mean, up_plus_down, t_pts_0_1, up_plus_down_mean
rt_data = {}        # (ABL, ILD): data_a_i_rt

for ABL in ABLs_to_fit:
    for ILD in ILDs_to_fit:
        gamma = None
        omega = None
        if ABL == 20:
            if abs(ILD) == 1:
                gamma = gamma_ABL_20_ILD_1
                omega = omega_ABL_20_ILD_1
            elif abs(ILD) == 2:
                gamma = gamma_ABL_20_ILD_2
                omega = omega_ABL_20_ILD_2
            elif abs(ILD) == 4:
                gamma = gamma_ABL_20_ILD_4
                omega = omega_ABL_20_ILD_4
            elif abs(ILD) == 8:
                gamma = gamma_ABL_20_ILD_8
                omega = omega_ABL_20_ILD_8
            elif abs(ILD) == 16:
                gamma = gamma_ABL_20_ILD_16
                omega = omega_ABL_20_ILD_16
        elif ABL == 60:
            if abs(ILD) == 1:
                gamma = gamma_ABL_60_ILD_1
                omega = omega_ABL_60_ILD_1
            elif abs(ILD) == 2:
                gamma = gamma_ABL_60_ILD_2
                omega = omega_ABL_60_ILD_2
            elif abs(ILD) == 4:
                gamma = gamma_ABL_60_ILD_4
                omega = omega_ABL_60_ILD_4
            elif abs(ILD) == 8:
                gamma = gamma_ABL_60_ILD_8
                omega = omega_ABL_60_ILD_8
            elif abs(ILD) == 16:
                gamma = gamma_ABL_60_ILD_16
                omega = omega_ABL_60_ILD_16
        if gamma is not None:
            gamma *= np.sign(ILD)
        if gamma is None or omega is None:
            print(f"Skipping ABL={ABL}, ILD={ILD} (no gamma/omega)")
            continue
        bound = 1
        up_mean = np.array([
            up_or_down_RTs_fit_OPTIM_V_A_change_gamma_omega_with_w_PA_CA_fn(
                t, P_A_mean[idx], C_A_mean[idx],
                gamma, omega, 0, t_E_aff, del_go, bound, w, K_max
            ) for idx, t in enumerate(t_pts)
        ])
        down_mean = np.array([
            up_or_down_RTs_fit_OPTIM_V_A_change_gamma_omega_with_w_PA_CA_fn(
                t, P_A_mean[idx], C_A_mean[idx],
                gamma, omega, 0, t_E_aff, del_go, -bound, w, K_max
            ) for idx, t in enumerate(t_pts)
        ])
        mask_0_1 = (t_pts >= 0) & (t_pts <= 1)
        t_pts_0_1 = t_pts[mask_0_1]
        up_plus_down = up_mean + down_mean
        # area up and down
        print(f'ABL={ABL}, ILD={ILD}')
        print(f'area up {trapezoid(up_mean, t_pts) :.2f}')
        print(f'area down {trapezoid(down_mean, t_pts) :.2f}')
        up_plus_down_masked = up_plus_down[mask_0_1]
        area_masked = trapezoid(up_plus_down_masked, t_pts_0_1)

        if area_masked != 0:
            up_plus_down_mean = up_plus_down_masked / area_masked
        else:
            up_plus_down_mean = up_plus_down_masked
        theory_curves[(ABL, ILD)] = {
            'up_mean': up_mean,
            'down_mean': down_mean,
            't_pts': t_pts,

            'up_mean_mask': up_mean[mask_0_1],
            'down_mean_mask': down_mean[mask_0_1],
            't_pts_0_1': t_pts_0_1,
            'up_plus_down_mean': up_plus_down_mean
        }
        # Data
        data_a_i = df_led_off_valid_trials_cond_filtered[
            (df_led_off_valid_trials_cond_filtered['ABL'] == ABL) &
            (df_led_off_valid_trials_cond_filtered['ILD'] == ILD)
        ]
        data_a_i_rt = data_a_i['timed_fix'] - data_a_i['intended_fix']
        rt_data[(ABL, ILD)] = data_a_i_rt

# %%
# --- Plot from stored arrays ---
n_ABLs = len(ABLs_to_fit)
n_ILDs = len(ILDs_to_fit)
fig, axes = plt.subplots(n_ABLs, n_ILDs, figsize=(4*n_ILDs, 3*n_ABLs), sharex=True, sharey=True)
for i_ABL, ABL in enumerate(ABLs_to_fit):
    for i_ILD, ILD in enumerate(ILDs_to_fit):
        ax = axes[i_ABL, i_ILD] if n_ABLs > 1 and n_ILDs > 1 else (
            axes[i_ILD] if n_ABLs == 1 else axes[i_ABL]
        )
        if (ABL, ILD) not in theory_curves:
            ax.set_visible(False)
            continue
        tc = theory_curves[(ABL, ILD)]
        ax.plot(tc['t_pts_0_1'], tc['up_plus_down_mean'], label="theory")
        ax.hist(rt_data[(ABL, ILD)], bins=np.arange(0,1,0.02), density=True, histtype='step', label="data")
        ax.set_title(f"ABL={ABL}, ILD={ILD}")
        if i_ABL == n_ABLs-1:
            ax.set_xlabel("RT (s)")
        if i_ILD == 0:
            ax.set_ylabel("Density")
        ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %%
# psychometrics:
# Plot psychometric function: prob(choice==1) vs ILD for each ABL in data
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
for ABL in ABLs_to_fit:
    # --- Data psychometric ---
    df_abl = df_led_off_valid_trials_cond_filtered[df_led_off_valid_trials_cond_filtered['ABL'] == ABL]
    prob_right_data = df_abl.groupby('ILD')['choice'].apply(lambda x: (x == 1).mean())
    ILDs_data = prob_right_data.index.values
    plt.scatter(ILDs_data, prob_right_data.values, label=f'Data ABL={ABL}', alpha=0.7, marker='o')
    # --- Theory psychometric ---
    ILDs_theory = []
    prob_right_theory = []
    for ILD in np.sort(ILDs_to_fit):
        key = (ABL, ILD)
        if key not in theory_curves:
            continue
        tc = theory_curves[key]
        area_up = trapezoid(tc['up_mean_mask'], tc['t_pts_0_1'])
        area_down = trapezoid(tc['down_mean_mask'], tc['t_pts_0_1'])
        p_right = area_up / (area_up + area_down) if (area_up + area_down) > 0 else np.nan
        print(f'ILD = {ILD}, p_right = {p_right}')
        ILDs_theory.append(ILD)
        prob_right_theory.append(p_right)
    # Sort for line plot
    ILDs_theory = np.array(ILDs_theory)
    prob_right_theory = np.array(prob_right_theory)
    idx_sort = np.argsort(ILDs_theory)
    plt.plot(ILDs_theory[idx_sort], prob_right_theory[idx_sort], label=f'Theory ABL={ABL}', marker='x')
plt.axhline(0.5, color='gray', ls='--', lw=1)
plt.xlabel('ILD (dB)')
plt.ylabel('P(choice=right)')
plt.title('Psychometric curve: Data vs Theory')
plt.legend(title='Curve')
plt.tight_layout()
plt.show()


# %%
import matplotlib.pyplot as plt

def get_gamma(ABL, ILD):
    if ABL == 20:
        if abs(ILD) == 1:
            return gamma_ABL_20_ILD_1
        elif abs(ILD) == 2:
            return gamma_ABL_20_ILD_2
        elif abs(ILD) == 4:
            return gamma_ABL_20_ILD_4
        elif abs(ILD) == 8:
            return gamma_ABL_20_ILD_8
        elif abs(ILD) == 16:
            return gamma_ABL_20_ILD_16
    elif ABL == 60:
        if abs(ILD) == 1:
            return gamma_ABL_60_ILD_1
        elif abs(ILD) == 2:
            return gamma_ABL_60_ILD_2
        elif abs(ILD) == 4:
            return gamma_ABL_60_ILD_4
        elif abs(ILD) == 8:
            return gamma_ABL_60_ILD_8
        elif abs(ILD) == 16:
            return gamma_ABL_60_ILD_16
    return None

def get_omega(ABL, ILD):
    if ABL == 20:
        if abs(ILD) == 1:
            return omega_ABL_20_ILD_1
        elif abs(ILD) == 2:
            return omega_ABL_20_ILD_2
        elif abs(ILD) == 4:
            return omega_ABL_20_ILD_4
        elif abs(ILD) == 8:
            return omega_ABL_20_ILD_8
        elif abs(ILD) == 16:
            return omega_ABL_20_ILD_16
    elif ABL == 60:
        if abs(ILD) == 1:
            return omega_ABL_60_ILD_1
        elif abs(ILD) == 2:
            return omega_ABL_60_ILD_2
        elif abs(ILD) == 4:
            return omega_ABL_60_ILD_4
        elif abs(ILD) == 8:
            return omega_ABL_60_ILD_8
        elif abs(ILD) == 16:
            return omega_ABL_60_ILD_16
    return None

plt.figure(figsize=(4, 3))
all_ILDs = sorted(set(ILD for ILD in ILDs_to_fit if ILD > 0))
for ABL in ABLs_to_fit:
    gammas = []
    ILDs_plot = []
    for ILD in ILDs_to_fit:
        if ILD > 0:
            gamma = get_gamma(ABL, ILD)
            if gamma is not None:
                gammas.append(gamma)
                ILDs_plot.append(ILD)
    plt.plot(ILDs_plot, gammas, marker='o', label=f'ABL={ABL}')
plt.xlabel('ILD (dB)')
plt.ylabel('gamma')
plt.title('gamma vs ILD for each ABL')
plt.xticks(all_ILDs)
plt.tight_layout()
plt.show()

plt.figure(figsize=(4, 3))
all_ILDs = sorted(set(ILD for ILD in ILDs_to_fit if ILD > 0))
for ABL in ABLs_to_fit:
    omegas = []
    ILDs_plot = []
    for ILD in ILDs_to_fit:
        if ILD > 0:
            omega = get_omega(ABL, ILD)
            if omega is not None:
                omegas.append(omega)
                ILDs_plot.append(ILD)
    plt.plot(ILDs_plot, omegas, marker='o', label=f'ABL={ABL}')
plt.xlabel('ILD (dB)')
plt.ylabel('omega')
plt.title('omega vs ILD for each ABL')
plt.xticks(all_ILDs)
plt.tight_layout()
plt.show()


# %%