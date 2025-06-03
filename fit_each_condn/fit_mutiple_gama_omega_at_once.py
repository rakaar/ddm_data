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

# %%
from led_off_gamma_omega_pdf_utils import cum_pro_and_reactive, up_or_down_RTs_fit_OPTIM_V_A_change_gamma_omega_fn,\
         rho_A_t_VEC_fn, up_or_down_RTs_fit_OPTIM_V_A_change_gamma_omega_P_A_C_A_wrt_stim_fn

# %% [markdown]
# # get og data

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
ILDs_to_fit = [1,4,16, -1,-4, -16]

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

# other params
K_max = 10

# %%
# VBMC loglike
def compute_loglike_trial(row, gamma_ABL_20_ILD_1, gamma_ABL_20_ILD_4, gamma_ABL_20_ILD_16, \
                                        gamma_ABL_60_ILD_1, gamma_ABL_60_ILD_4, gamma_ABL_60_ILD_16, \
                                        omega_ABL_20_ILD_1, omega_ABL_20_ILD_4, omega_ABL_20_ILD_16, \
                                        omega_ABL_60_ILD_1, omega_ABL_60_ILD_4, omega_ABL_60_ILD_16, \
                                        t_E_aff, w, del_go):
    # data
    c_A_trunc_time = 0.3
    rt = row['timed_fix']
    t_stim = row['intended_fix']
    response_poke = row['response_poke']
    
    ABL = row['ABL']
    ILD = row['ILD']

    gamma = (
    gamma_ABL_20_ILD_1 if ABL == 20 and abs(ILD) == 1 else
    gamma_ABL_20_ILD_4 if ABL == 20 and abs(ILD) == 4 else
    gamma_ABL_20_ILD_16 if ABL == 20 and abs(ILD) == 16 else
    gamma_ABL_60_ILD_1 if ABL == 60 and abs(ILD) == 1 else
    gamma_ABL_60_ILD_4 if ABL == 60 and abs(ILD) == 4 else
    gamma_ABL_60_ILD_16 if ABL == 60 and abs(ILD) == 16 else
    None
)

    omega = (
    omega_ABL_20_ILD_1 if ABL == 20 and abs(ILD) == 1 else
    omega_ABL_20_ILD_4 if ABL == 20 and abs(ILD) == 4 else
    omega_ABL_20_ILD_16 if ABL == 20 and abs(ILD) == 16 else
    omega_ABL_60_ILD_1 if ABL == 60 and abs(ILD) == 1 else
    omega_ABL_60_ILD_4 if ABL == 60 and abs(ILD) == 4 else
    omega_ABL_60_ILD_16 if ABL == 60 and abs(ILD) == 16 else
    None
)

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
    # gamma_ABL_x_ILD_y: x is [20,60]. y is [1,4,16]
    # omega_ABL_x_ILD_y: x is [20,60]. y is [1,4,16]
    gamma_ABL_20_ILD_1, gamma_ABL_20_ILD_4, gamma_ABL_20_ILD_16, \
    gamma_ABL_60_ILD_1, gamma_ABL_60_ILD_4, gamma_ABL_60_ILD_16, \
    omega_ABL_20_ILD_1, omega_ABL_20_ILD_4, omega_ABL_20_ILD_16, \
    omega_ABL_60_ILD_1, omega_ABL_60_ILD_4, omega_ABL_60_ILD_16, \
    t_E_aff, w, del_go = params

    all_loglike = Parallel(n_jobs=30)(delayed(compute_loglike_trial)(row, gamma_ABL_20_ILD_1, gamma_ABL_20_ILD_4, gamma_ABL_20_ILD_16, \
                                        gamma_ABL_60_ILD_1, gamma_ABL_60_ILD_4, gamma_ABL_60_ILD_16, \
                                        omega_ABL_20_ILD_1, omega_ABL_20_ILD_4, omega_ABL_20_ILD_16, \
                                        omega_ABL_60_ILD_1, omega_ABL_60_ILD_4, omega_ABL_60_ILD_16, \
                                        t_E_aff, w, del_go) \
                                     for _, row in df_led_off_valid_trials_cond_filtered.iterrows())
    
    return np.sum(all_loglike)

# %% [markdown]
# # bounds

# %%
# gamma_bounds = [0.02, 2]
# gamma_plausible_bounds = [0.09, 0.9]
gamma_ABL_20_ILD_1_bounds = gamma_ABL_20_ILD_4_bounds = gamma_ABL_20_ILD_16_bounds = gamma_ABL_60_ILD_1_bounds = gamma_ABL_60_ILD_4_bounds = gamma_ABL_60_ILD_16_bounds = [0.02, 2]
gamma_ABL_20_ILD_1_plausible_bounds = gamma_ABL_20_ILD_4_plausible_bounds = gamma_ABL_20_ILD_16_plausible_bounds = gamma_ABL_60_ILD_1_plausible_bounds = gamma_ABL_60_ILD_4_plausible_bounds = gamma_ABL_60_ILD_16_plausible_bounds = [0.09, 0.9]

omega_ABL_20_ILD_1_bounds = omega_ABL_20_ILD_4_bounds = omega_ABL_20_ILD_16_bounds = omega_ABL_60_ILD_1_bounds = omega_ABL_60_ILD_4_bounds = omega_ABL_60_ILD_16_bounds = [0.05, 50]
omega_ABL_20_ILD_1_plausible_bounds = omega_ABL_20_ILD_4_plausible_bounds = omega_ABL_20_ILD_16_plausible_bounds = omega_ABL_60_ILD_1_plausible_bounds = omega_ABL_60_ILD_4_plausible_bounds = omega_ABL_60_ILD_16_plausible_bounds = [0.5, 10]

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
    gamma_ABL_20_ILD_1, gamma_ABL_20_ILD_4, gamma_ABL_20_ILD_16, \
    gamma_ABL_60_ILD_1, gamma_ABL_60_ILD_4, gamma_ABL_60_ILD_16, \
    omega_ABL_20_ILD_1, omega_ABL_20_ILD_4, omega_ABL_20_ILD_16, \
    omega_ABL_60_ILD_1, omega_ABL_60_ILD_4, omega_ABL_60_ILD_16, \
    t_E_aff, w, del_go = params

    # gamma_logpdf = trapezoidal_logpdf(gamma, gamma_bounds[0], gamma_plausible_bounds[0], gamma_plausible_bounds[1], gamma_bounds[1])
    # omega_logpdf = trapezoidal_logpdf(omega, omega_bounds[0], omega_plausible_bounds[0], omega_plausible_bounds[1], omega_bounds[1])
    # t_E_aff_logpdf = trapezoidal_logpdf(t_E_aff, t_E_aff_bounds[0], t_E_aff_plausible_bounds[0], t_E_aff_plausible_bounds[1], t_E_aff_bounds[1])
    # return gamma_logpdf + omega_logpdf + t_E_aff_logpdf

    gamma_ABL_20_ILD_1_logpdf = trapezoidal_logpdf(gamma_ABL_20_ILD_1, gamma_ABL_20_ILD_1_bounds[0], gamma_ABL_20_ILD_1_plausible_bounds[0], gamma_ABL_20_ILD_1_plausible_bounds[1], gamma_ABL_20_ILD_1_bounds[1])
    gamma_ABL_20_ILD_4_logpdf = trapezoidal_logpdf(gamma_ABL_20_ILD_4, gamma_ABL_20_ILD_4_bounds[0], gamma_ABL_20_ILD_4_plausible_bounds[0], gamma_ABL_20_ILD_4_plausible_bounds[1], gamma_ABL_20_ILD_4_bounds[1])
    gamma_ABL_20_ILD_16_logpdf = trapezoidal_logpdf(gamma_ABL_20_ILD_16, gamma_ABL_20_ILD_16_bounds[0], gamma_ABL_20_ILD_16_plausible_bounds[0], gamma_ABL_20_ILD_16_plausible_bounds[1], gamma_ABL_20_ILD_16_bounds[1])
    gamma_ABL_60_ILD_1_logpdf = trapezoidal_logpdf(gamma_ABL_60_ILD_1, gamma_ABL_60_ILD_1_bounds[0], gamma_ABL_60_ILD_1_plausible_bounds[0], gamma_ABL_60_ILD_1_plausible_bounds[1], gamma_ABL_60_ILD_1_bounds[1])
    gamma_ABL_60_ILD_4_logpdf = trapezoidal_logpdf(gamma_ABL_60_ILD_4, gamma_ABL_60_ILD_4_bounds[0], gamma_ABL_60_ILD_4_plausible_bounds[0], gamma_ABL_60_ILD_4_plausible_bounds[1], gamma_ABL_60_ILD_4_bounds[1])
    gamma_ABL_60_ILD_16_logpdf = trapezoidal_logpdf(gamma_ABL_60_ILD_16, gamma_ABL_60_ILD_16_bounds[0], gamma_ABL_60_ILD_16_plausible_bounds[0], gamma_ABL_60_ILD_16_plausible_bounds[1], gamma_ABL_60_ILD_16_bounds[1])
    omega_ABL_20_ILD_1_logpdf = trapezoidal_logpdf(omega_ABL_20_ILD_1, omega_ABL_20_ILD_1_bounds[0], omega_ABL_20_ILD_1_plausible_bounds[0], omega_ABL_20_ILD_1_plausible_bounds[1], omega_ABL_20_ILD_1_bounds[1])
    omega_ABL_20_ILD_4_logpdf = trapezoidal_logpdf(omega_ABL_20_ILD_4, omega_ABL_20_ILD_4_bounds[0], omega_ABL_20_ILD_4_plausible_bounds[0], omega_ABL_20_ILD_4_plausible_bounds[1], omega_ABL_20_ILD_4_bounds[1])
    omega_ABL_20_ILD_16_logpdf = trapezoidal_logpdf(omega_ABL_20_ILD_16, omega_ABL_20_ILD_16_bounds[0], omega_ABL_20_ILD_16_plausible_bounds[0], omega_ABL_20_ILD_16_plausible_bounds[1], omega_ABL_20_ILD_16_bounds[1])
    omega_ABL_60_ILD_1_logpdf = trapezoidal_logpdf(omega_ABL_60_ILD_1, omega_ABL_60_ILD_1_bounds[0], omega_ABL_60_ILD_1_plausible_bounds[0], omega_ABL_60_ILD_1_plausible_bounds[1], omega_ABL_60_ILD_1_bounds[1])
    omega_ABL_60_ILD_4_logpdf = trapezoidal_logpdf(omega_ABL_60_ILD_4, omega_ABL_60_ILD_4_bounds[0], omega_ABL_60_ILD_4_plausible_bounds[0], omega_ABL_60_ILD_4_plausible_bounds[1], omega_ABL_60_ILD_4_bounds[1])
    omega_ABL_60_ILD_16_logpdf = trapezoidal_logpdf(omega_ABL_60_ILD_16, omega_ABL_60_ILD_16_bounds[0], omega_ABL_60_ILD_16_plausible_bounds[0], omega_ABL_60_ILD_16_plausible_bounds[1], omega_ABL_60_ILD_16_bounds[1])
    t_E_aff_logpdf = trapezoidal_logpdf(t_E_aff, t_E_aff_bounds[0], t_E_aff_plausible_bounds[0], t_E_aff_plausible_bounds[1], t_E_aff_bounds[1])
    w_logpdf = trapezoidal_logpdf(w, w_bounds[0], w_plausible_bounds[0], w_plausible_bounds[1], w_bounds[1])
    del_go_logpdf = trapezoidal_logpdf(del_go, del_go_bounds[0], del_go_plausible_bounds[0], del_go_plausible_bounds[1], del_go_bounds[1])
    return gamma_ABL_20_ILD_1_logpdf + gamma_ABL_20_ILD_4_logpdf + gamma_ABL_20_ILD_16_logpdf + \
           gamma_ABL_60_ILD_1_logpdf + gamma_ABL_60_ILD_4_logpdf + gamma_ABL_60_ILD_16_logpdf + \
           omega_ABL_20_ILD_1_logpdf + omega_ABL_20_ILD_4_logpdf + omega_ABL_20_ILD_16_logpdf + \
           omega_ABL_60_ILD_1_logpdf + omega_ABL_60_ILD_4_logpdf + omega_ABL_60_ILD_16_logpdf + \
           t_E_aff_logpdf + w_logpdf + del_go_logpdf

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
lb = np.array([gamma_ABL_20_ILD_1_bounds[0], gamma_ABL_20_ILD_4_bounds[0], gamma_ABL_20_ILD_16_bounds[0], \
                    gamma_ABL_60_ILD_1_bounds[0], gamma_ABL_60_ILD_4_bounds[0], gamma_ABL_60_ILD_16_bounds[0], \
                    omega_ABL_20_ILD_1_bounds[0], omega_ABL_20_ILD_4_bounds[0], omega_ABL_20_ILD_16_bounds[0], \
                    omega_ABL_60_ILD_1_bounds[0], omega_ABL_60_ILD_4_bounds[0], omega_ABL_60_ILD_16_bounds[0], \
                    t_E_aff_bounds[0], w_bounds[0], del_go_bounds[0]])

ub = np.array([gamma_ABL_20_ILD_1_bounds[1], gamma_ABL_20_ILD_4_bounds[1], gamma_ABL_20_ILD_16_bounds[1], \
                    gamma_ABL_60_ILD_1_bounds[1], gamma_ABL_60_ILD_4_bounds[1], gamma_ABL_60_ILD_16_bounds[1], \
                    omega_ABL_20_ILD_1_bounds[1], omega_ABL_20_ILD_4_bounds[1], omega_ABL_20_ILD_16_bounds[1], \
                    omega_ABL_60_ILD_1_bounds[1], omega_ABL_60_ILD_4_bounds[1], omega_ABL_60_ILD_16_bounds[1], \
                    t_E_aff_bounds[1], w_bounds[1], del_go_bounds[1]])

plb = np.array([gamma_ABL_20_ILD_1_plausible_bounds[0], gamma_ABL_20_ILD_4_plausible_bounds[0], gamma_ABL_20_ILD_16_plausible_bounds[0], \
                    gamma_ABL_60_ILD_1_plausible_bounds[0], gamma_ABL_60_ILD_4_plausible_bounds[0], gamma_ABL_60_ILD_16_plausible_bounds[0], \
                    omega_ABL_20_ILD_1_plausible_bounds[0], omega_ABL_20_ILD_4_plausible_bounds[0], omega_ABL_20_ILD_16_plausible_bounds[0], \
                    omega_ABL_60_ILD_1_plausible_bounds[0], omega_ABL_60_ILD_4_plausible_bounds[0], omega_ABL_60_ILD_16_plausible_bounds[0], \
                    t_E_aff_plausible_bounds[0], w_plausible_bounds[0], del_go_plausible_bounds[0]])
pub = np.array([gamma_ABL_20_ILD_1_plausible_bounds[1], gamma_ABL_20_ILD_4_plausible_bounds[1], gamma_ABL_20_ILD_16_plausible_bounds[1], \
                    gamma_ABL_60_ILD_1_plausible_bounds[1], gamma_ABL_60_ILD_4_plausible_bounds[1], gamma_ABL_60_ILD_16_plausible_bounds[1], \
                    omega_ABL_20_ILD_1_plausible_bounds[1], omega_ABL_20_ILD_4_plausible_bounds[1], omega_ABL_20_ILD_16_plausible_bounds[1], \
                    omega_ABL_60_ILD_1_plausible_bounds[1], omega_ABL_60_ILD_4_plausible_bounds[1], omega_ABL_60_ILD_16_plausible_bounds[1], \
                    t_E_aff_plausible_bounds[1], w_plausible_bounds[1], del_go_plausible_bounds[1]])

# Initialize with random values within plausible bounds
np.random.seed(42)
# gamma_0 = np.random.uniform(gamma_plausible_bounds[0], gamma_plausible_bounds[1])
# omega_0 = np.random.uniform(omega_plausible_bounds[0], omega_plausible_bounds[1])
# t_E_aff_0 = np.random.uniform(t_E_aff_plausible_bounds[0], t_E_aff_plausible_bounds[1])
gamma_ABL_20_ILD_1_0 = np.random.uniform(gamma_ABL_20_ILD_1_plausible_bounds[0], gamma_ABL_20_ILD_1_plausible_bounds[1])
gamma_ABL_20_ILD_4_0 = np.random.uniform(gamma_ABL_20_ILD_4_plausible_bounds[0], gamma_ABL_20_ILD_4_plausible_bounds[1])
gamma_ABL_20_ILD_16_0 = np.random.uniform(gamma_ABL_20_ILD_16_plausible_bounds[0], gamma_ABL_20_ILD_16_plausible_bounds[1])
gamma_ABL_60_ILD_1_0 = np.random.uniform(gamma_ABL_60_ILD_1_plausible_bounds[0], gamma_ABL_60_ILD_1_plausible_bounds[1])
gamma_ABL_60_ILD_4_0 = np.random.uniform(gamma_ABL_60_ILD_4_plausible_bounds[0], gamma_ABL_60_ILD_4_plausible_bounds[1])
gamma_ABL_60_ILD_16_0 = np.random.uniform(gamma_ABL_60_ILD_16_plausible_bounds[0], gamma_ABL_60_ILD_16_plausible_bounds[1])
omega_ABL_20_ILD_1_0 = np.random.uniform(omega_ABL_20_ILD_1_plausible_bounds[0], omega_ABL_20_ILD_1_plausible_bounds[1])
omega_ABL_20_ILD_4_0 = np.random.uniform(omega_ABL_20_ILD_4_plausible_bounds[0], omega_ABL_20_ILD_4_plausible_bounds[1])
omega_ABL_20_ILD_16_0 = np.random.uniform(omega_ABL_20_ILD_16_plausible_bounds[0], omega_ABL_20_ILD_16_plausible_bounds[1])
omega_ABL_60_ILD_1_0 = np.random.uniform(omega_ABL_60_ILD_1_plausible_bounds[0], omega_ABL_60_ILD_1_plausible_bounds[1])
omega_ABL_60_ILD_4_0 = np.random.uniform(omega_ABL_60_ILD_4_plausible_bounds[0], omega_ABL_60_ILD_4_plausible_bounds[1])
omega_ABL_60_ILD_16_0 = np.random.uniform(omega_ABL_60_ILD_16_plausible_bounds[0], omega_ABL_60_ILD_16_plausible_bounds[1])
t_E_aff_0 = np.random.uniform(t_E_aff_plausible_bounds[0], t_E_aff_plausible_bounds[1])
w_0 = np.random.uniform(w_plausible_bounds[0], w_plausible_bounds[1])
del_go_0 = np.random.uniform(del_go_plausible_bounds[0], del_go_plausible_bounds[1])

# x_0 = np.array([gamma_0, omega_0, t_E_aff_0])
x_0 = np.array([gamma_ABL_20_ILD_1_0, gamma_ABL_20_ILD_4_0, gamma_ABL_20_ILD_16_0, gamma_ABL_60_ILD_1_0, gamma_ABL_60_ILD_4_0, gamma_ABL_60_ILD_16_0, omega_ABL_20_ILD_1_0, omega_ABL_20_ILD_4_0, omega_ABL_20_ILD_16_0, omega_ABL_60_ILD_1_0, omega_ABL_60_ILD_4_0, omega_ABL_60_ILD_16_0, t_E_aff_0, w_0, del_go_0])

# Run VBMC
vbmc = VBMC(vbmc_joint_fn, x_0, lb, ub, plb, pub, options={'display': 'on'})
vp, results = vbmc.optimize()

# %%
vbmc.save(f'vbmc_mutiple_gama_omega_at_once.pkl', overwrite=True)

# %%
# vbmc.save(f'vbmc_single_condn_ABL_{conditions['ABL'][0]}_ILD_{conditions['ILD'][0]}.pkl')

# %%
# with open(f"vbmc_single_condn_ABL_{conditions['ABL'][0]}_ILD_{conditions['ILD'][0]}.pkl", 'rb') as f:
#     vp = pickle.load(f)

# vp = vp.vp

# %% [markdown]
# # vbmc sample

# %%
vp_samples = vp.sample(int(1e5))[0]

gamma_samples = vp_samples[:, 0]
omega_samples = vp_samples[:, 1]
t_E_aff_samples = vp_samples[:, 2]

# %%
gamma = gamma_samples.mean()
omega = omega_samples.mean()
t_E_aff = t_E_aff_samples.mean()

# %% [markdown]
# # corner

# %%
# plot the corner plot
corner_samples = np.vstack([gamma_samples, omega_samples, t_E_aff_samples]).T
percentiles = np.percentile(corner_samples, [0, 100], axis=0)
_ranges = [(percentiles[0, i], percentiles[1, i]) for i in np.arange(corner_samples.shape[1])]
param_labels = [ 'gamma', 'omega', 't_E_aff']

corner.corner(
    corner_samples,
    labels=param_labels,
    show_titles=True,
    quantiles=[0.025, 0.5, 0.975],
    range=_ranges,
    title_fmt=".4f"
);


# %% [markdown]
# # Diagnostics

# %% [markdown]
# ## exp data df prepare

# %%
# DATA
df_led_off = df[df['LED_trial'] == 0]

# < 1s RTs
df_led_off = df_led_off[df_led_off['timed_fix'] - df_led_off['intended_fix'] < 1]
# remove truncated aborts
data_df_led_off_with_aborts = df_led_off[ ~( (df_led_off['abort_event'] == 3) & (df_led_off['timed_fix'] < 0.3) ) ]
# renaming
data_df_led_off_with_aborts = data_df_led_off_with_aborts.rename(
    columns={'timed_fix': 'rt', 'intended_fix': 't_stim'}
)

### ABORTS + VALID TRIALS + ABL, ILD CONDITION
data_df_led_off_with_aborts_cond_filtered = data_df_led_off_with_aborts[
    (data_df_led_off_with_aborts['ABL'].isin(conditions['ABL'])) & 
    (data_df_led_off_with_aborts['ILD'].isin(conditions['ILD']))
]

data_df_led_off_valid = data_df_led_off_with_aborts[ data_df_led_off_with_aborts['success'].isin([1,-1]) ]

# VALID TRIALS CONDITION
df_led_off_valid_trials_cond_filtered = data_df_led_off_valid[
    (data_df_led_off_valid['ABL'].isin(conditions['ABL'])) & 
    (data_df_led_off_valid['ILD'].isin(conditions['ILD']))
]

df_led_off_valid_trials_cond_filtered['ABL'].unique(), df_led_off_valid_trials_cond_filtered['ILD'].unique()

# %% [markdown]
# ## up and down RT

# %%
N_theory = int(1e3)
random_indices = np.random.randint(0, len(t_stim_and_led_tuple), N_theory)
t_pts = np.arange(0, 1, 0.001)

P_A_samples = np.zeros((N_theory, len(t_pts)))
for idx in range(N_theory):
    t_stim, t_LED = t_stim_and_led_tuple[random_indices[idx]]
    pdf = rho_A_t_VEC_fn(t_pts + t_stim - t_A_aff, V_A, theta_A)
    P_A_samples[idx, :] = pdf

# %%
P_A_samples_mean = np.mean(P_A_samples, axis=0)
C_A_mean = cumtrapz(P_A_samples_mean, t_pts, initial=0)

# %%
up_wrt_stim = np.zeros_like(t_pts)
down_wrt_stim = np.zeros_like(t_pts)
for idx, t in enumerate(t_pts):
    P_A = P_A_samples_mean[idx]
    C_A = C_A_mean[idx]
    up_wrt_stim[idx] =  up_or_down_RTs_fit_OPTIM_V_A_change_gamma_omega_P_A_C_A_wrt_stim_fn(t, P_A, C_A, gamma, omega, t_E_aff, del_go, 1, K_max)
    down_wrt_stim[idx] = up_or_down_RTs_fit_OPTIM_V_A_change_gamma_omega_P_A_C_A_wrt_stim_fn(t, P_A, C_A, gamma, omega, t_E_aff, del_go, -1, K_max)


# %%
bins = np.arange(-1,1,0.02)
bin_centers = (bins[:-1] + bins[1:]) / 2



## data
data_up = df_led_off_valid_trials_cond_filtered[df_led_off_valid_trials_cond_filtered['choice'] == 1]
data_down = df_led_off_valid_trials_cond_filtered[df_led_off_valid_trials_cond_filtered['choice'] == -1]

data_up_rt = data_up['rt'] - data_up['t_stim']
data_up_rt_hist, _ = np.histogram(data_up_rt, bins=bins, density=True)

data_down_rt = data_down['rt'] - data_down['t_stim']
data_down_rt_hist, _ = np.histogram(data_down_rt, bins=bins, density=True)

frac_up_data = len(data_up) / len(df_led_off_valid_trials_cond_filtered)
frac_down_data = len(data_down) / len(df_led_off_valid_trials_cond_filtered)

# %%
# RTDs - up and down
plt.plot(t_pts, up_wrt_stim, ls='--', color='r')
plt.plot(t_pts, -down_wrt_stim, ls='--', color='r')

plt.plot(bin_centers, data_up_rt_hist*frac_up_data, color='b')
plt.plot(bin_centers, -data_down_rt_hist*frac_down_data, color='b')

theory_area_up = trapz(up_wrt_stim, t_pts)
theory_area_down = trapz(down_wrt_stim, t_pts)

print(f'areas theory up = {theory_area_up :.3f}, down = {theory_area_down :.3f}')
print(f'frac up data = {frac_up_data :.3f}, down data = {frac_down_data :.3f}')
plt.xlim(0,1)
plt.title(f'Areas: +T:{theory_area_up:.3f},+E:{frac_up_data:.3f},-T:{theory_area_down: .3f},-E:{frac_down_data :.3f}')

plt.xlabel('rt wrt stim')
plt.ylabel('density')

# %% [markdown]
# ## accuracy

# %%
## 2. Accuracy
xlabels = ['data', 'vbmc']
if conditions['ILD'][0] > 0:
    accuracy_data_and_theory = [frac_up_data, theory_area_up]
else:
    accuracy_data_and_theory = [frac_down_data, theory_area_down]

plt.bar(xlabels, accuracy_data_and_theory)
plt.ylabel('accuracy')
plt.title('between stim start and stim + 1s')
plt.ylim(0,1)
plt.yticks(np.arange(0, 1.1, 0.1));

# %%


# %% [markdown]
# # tachometric

# %%
tacho = np.zeros_like(t_pts)
for idx, t in enumerate(t_pts):
    P_A = P_A_samples_mean[idx]
    C_A = C_A_mean[idx]
    P_up = up_or_down_RTs_fit_OPTIM_V_A_change_gamma_omega_P_A_C_A_wrt_stim_fn(t, P_A, C_A, gamma, omega, t_E_aff, del_go, 1, K_max)
    P_down = up_or_down_RTs_fit_OPTIM_V_A_change_gamma_omega_P_A_C_A_wrt_stim_fn(t, P_A, C_A, gamma, omega, t_E_aff, del_go, -1, K_max)

    if conditions['ILD'][0] > 0:
        P_rt_c = P_up
    else:
        P_rt_c = P_down
        
    P_rt = P_up + P_down

    tacho[idx] = P_rt_c / (P_rt + 1e-10)

# %%
df_led_off_valid_trials_cond_filtered_copy = df_led_off_valid_trials_cond_filtered.copy()
df_led_off_valid_trials_cond_filtered_copy.loc[:, 'RT_bin'] = pd.cut(df_led_off_valid_trials_cond_filtered_copy['rt'] - df_led_off_valid_trials_cond_filtered_copy['t_stim'],\
                                                              bins = bins, include_lowest=True)
grouped_by_rt_bin = df_led_off_valid_trials_cond_filtered_copy.groupby('RT_bin', observed=False)['correct'].agg(['mean', 'count'])
grouped_by_rt_bin['bin_mid'] = grouped_by_rt_bin.index.map(lambda x: x.mid)

# %%
##  3. Tacho
plt.plot(t_pts, tacho)
plt.plot(grouped_by_rt_bin['bin_mid'], grouped_by_rt_bin['mean'], label='data')

plt.ylim(0.5,1)
plt.xlabel('rt - t_stim')
plt.ylabel('accuracy')
plt.title('btn stim start and stim + 1s')

# %% [markdown]
# # combined diagnostics

# %% [markdown]
# # des

# %%
import io
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import corner

# --------------------------
# Font size parameters
# Adjust these as needed
FONT_SIZE_LABEL = 22   # for x/y labels
FONT_SIZE_TICKS = 22   # for x/y tick labels
FONT_SIZE_TITLE = 22   # for subplot titles
FONT_SIZE_INSET_TITLE = 22  # for inset bar plot title

# --------------------------
# Create a figure and a GridSpec with 3 columns,
# giving the third column more space (e.g., 1.4 times the width).
fig = plt.figure(figsize=(30, 8))
gs = gridspec.GridSpec(nrows=1, ncols=3, width_ratios=[1, 1, 1.4])

ax_rtd = fig.add_subplot(gs[0, 0])
ax_tacho = fig.add_subplot(gs[0, 1])
ax_corner = fig.add_subplot(gs[0, 2])

# ------------------------------------------------------------------------------
# 1) RTD Plot in ax_rtd
ax_rtd.plot(t_pts, up_wrt_stim, ls='--', color='r')
ax_rtd.plot(t_pts, -down_wrt_stim, ls='--', color='r')
ax_rtd.plot(bin_centers, data_up_rt_hist * frac_up_data, color='b')
ax_rtd.plot(bin_centers, -data_down_rt_hist * frac_down_data, color='b')

ax_rtd.set_xlim(0, 1)

# Set labels and title with custom font sizes
ax_rtd.set_xlabel('rt wrt stim', fontsize=FONT_SIZE_LABEL)
ax_rtd.set_ylabel('density', fontsize=FONT_SIZE_LABEL)
ax_rtd.set_title(
    f"Areas: +T:{theory_area_up:.3f},+E:{frac_up_data:.3f},"
    f"-T:{theory_area_down:.3f},-E:{frac_down_data:.3f}",
    fontsize=FONT_SIZE_TITLE
)

# Increase tick label size
ax_rtd.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICKS)

# Inset bar chart
inset_ax = ax_rtd.inset_axes([0.65, 0.55, 0.3, 0.4])
bar_positions = [0, 1]
inset_ax.bar(bar_positions, accuracy_data_and_theory, color=['C0', 'C1'])
inset_ax.set_xticks(bar_positions)
inset_ax.set_xticklabels(['data', 'vbmc'], fontsize=FONT_SIZE_TICKS)
inset_ax.set_ylim(0, 1)
inset_ax.set_ylabel('accuracy', fontsize=FONT_SIZE_LABEL)
inset_ax.set_title(
    f"AL={conditions['ABL'][0]},ID={conditions['ILD'][0]}",
    fontsize=FONT_SIZE_INSET_TITLE
)
inset_ax.tick_params(axis='y', which='major', labelsize=FONT_SIZE_TICKS)

# ------------------------------------------------------------------------------
# 2) Tacho Plot in ax_tacho
ax_tacho.plot(t_pts, tacho, label='tacho')
ax_tacho.plot(grouped_by_rt_bin['bin_mid'], grouped_by_rt_bin['mean'], label='data')
ax_tacho.set_ylim(0.5, 1)

# Set labels and title with custom font sizes
ax_tacho.set_xlabel('rt - t_stim', fontsize=FONT_SIZE_LABEL)
ax_tacho.set_ylabel('accuracy', fontsize=FONT_SIZE_LABEL)
ax_tacho.set_title('btn stim start and stim + 1s', fontsize=FONT_SIZE_TITLE)

# Increase tick label size
ax_tacho.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICKS)
ax_tacho.legend(fontsize=FONT_SIZE_TICKS)


## 3. Corner
fig_corner = corner.corner(
    corner_samples,
    labels=['Γ', 'ω', 'tE'],
    show_titles=True,
    quantiles=[0.025, 0.5, 0.975],
    range=_ranges,
    title_fmt=".4f",
    title_kwargs={"fontsize": 18}
)

# Adjust subplots to reduce margins around the corner plot.
# Tweak left, right, bottom, top until the left whitespace is minimized.
fig_corner.subplots_adjust(left=-1, right=0.95, bottom=0.15, top=0.9)

# Save with minimal padding
buf = io.BytesIO()
fig_corner.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
buf.seek(0)

# Read into an image array and close the figure
corner_img = plt.imread(buf)
plt.close(fig_corner)

# Now display in your third subplot
ax_corner.imshow(corner_img)
ax_corner.axis('off')
ax_corner.set_title('Corner Plot', fontsize=20, pad=0)

# plt.savefig("my_figure.png", dpi=300, bbox_inches="tight")


# %%


# %%



