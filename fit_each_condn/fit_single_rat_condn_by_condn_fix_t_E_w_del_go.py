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

# animal filter 
batch_name = 'LED7'
animal_id = 103
print(f'animal filter: {animal_id}')
df = df[df['animal'] == int(animal_id)]

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
## VBMC params
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

K_max = 10
# Average t_E_aff: 0.08411045805617333
# Average w: 0.4594308751578441
# Average del_go: 0.12118261009394682

t_E_aff = 0.078
w_20 = 0.435
w_40 = 0.458
w_60 = 0.443
del_go = 0.144

# %%
def compute_loglike_trial(row, gamma, omega):
        # data
        c_A_trunc_time = 0.3
        rt = row['timed_fix']
        t_stim = row['intended_fix']
        response_poke = row['response_poke']
        
        if row['ABL'] == 20:
            w = w_20
        elif row['ABL'] == 40:
            w = w_40
        elif row['ABL'] == 60:
            w = w_60
        
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
    gamma, omega = params
    all_loglike = Parallel(n_jobs=30)(delayed(compute_loglike_trial)(
        row,
        gamma, omega
    ) for _, row in df_led_off_valid_trials_cond_filtered.iterrows())

    return np.sum(all_loglike)



# %%
# gamma_bounds = [-1, 5]
# gamma_plausible_bounds = [0, 3]

omega_bounds = [0.1, 15]
omega_plausible_bounds = [2, 12]


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
    gamma, omega = params
    gamma_logpdf = trapezoidal_logpdf(gamma, gamma_bounds[0], gamma_plausible_bounds[0], gamma_plausible_bounds[1], gamma_bounds[1])
    omega_logpdf = trapezoidal_logpdf(omega, omega_bounds[0], omega_plausible_bounds[0], omega_plausible_bounds[1], omega_bounds[1])

    return gamma_logpdf + omega_logpdf

# %%
all_ABLs_cond = [20, 40, 60]
all_ILDs_cond = [1, -1, 2, -2, 4, -4, 8, -8, 16, -16]
for cond_ABL in all_ABLs_cond:
    for cond_ILD in all_ILDs_cond:
        conditions = {'ABL': [cond_ABL], 'ILD': [cond_ILD]}

        # Applying the filter
        df_led_off_valid_trials_cond_filtered = df_led_off_valid_trials[
            (df_led_off_valid_trials['ABL'].isin(conditions['ABL'])) & 
            (df_led_off_valid_trials['ILD'].isin(conditions['ILD']))
        ]


        def vbmc_loglike_fn(params):
            gamma, omega = params

            all_loglike = Parallel(n_jobs=30)(delayed(compute_loglike_trial)(row, gamma, omega) \
                                            for _, row in df_led_off_valid_trials_cond_filtered.iterrows())
            
            return np.sum(all_loglike)

        def vbmc_joint_fn(params):
            priors = vbmc_prior_fn(params)
            loglike = vbmc_loglike_fn(params)

            return priors + loglike

        
        print(f'++++++++++ ABL = {cond_ABL}, ILD = {cond_ILD} +++++++++++++++++++++')
        if cond_ILD > 0:
            gamma_bounds = [-1, 5]
            gamma_plausible_bounds = [0, 3]
        elif cond_ILD < 0:
            gamma_bounds = [-5, 1]
            gamma_plausible_bounds = [-3, 0]


        lb = np.array([gamma_bounds[0], omega_bounds[0]])
        ub = np.array([gamma_bounds[1], omega_bounds[1]])

        plb = np.array([gamma_plausible_bounds[0], omega_plausible_bounds[0]])
        pub = np.array([gamma_plausible_bounds[1], omega_plausible_bounds[1]])

        # Initialize with random values within plausible bounds
        np.random.seed(42)
        gamma_0 = np.random.uniform(gamma_plausible_bounds[0], gamma_plausible_bounds[1])
        omega_0 = np.random.uniform(omega_plausible_bounds[0], omega_plausible_bounds[1])

        x_0 = np.array([gamma_0, omega_0])

        # Run VBMC
        vbmc = VBMC(vbmc_joint_fn, x_0, lb, ub, plb, pub, options={'display': 'on'})
        vp, results = vbmc.optimize()

        # save vbmc 
        vbmc.save(f'vbmc_cond_by_cond_{batch_name}_{animal_id}_{cond_ABL}_ILD_{cond_ILD}_FIX_t_E_w_del_go_same_as_parametric.pkl', overwrite=True)

        