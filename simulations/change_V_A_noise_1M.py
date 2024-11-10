import numpy as np
import matplotlib.pyplot as plt

from joblib import Parallel, delayed, parallel_backend

import pandas as pd
from increase_V_A_and_add_noise_utils import psiam_tied_data_gen_wrapper_noise_and_V_A_change

V_A = 1.1
theta_A = 1.8

rate_lambda = 0.14
T_0 = 0.66 * (1e-3)
theta_E = 40.5

t_A_aff = 0.03
t_E_aff = 0.05
t_motor = 0.03

Z_E = 0
L = 0.5

N_sim = int(1e6)
dt = 1e-4


N_print = int(1e5)
N_params = 11

new_noise_step_size = 4
new_V_A = 1.4

og_df = pd.read_csv('../out_LED.csv')
df = og_df[ og_df['repeat_trial'].isin([0,2]) | og_df['repeat_trial'].isna() ]
session_type = 7    
df = df[ df['session_type'].isin([session_type]) ]
training_level = 16
df = df[ df['training_level'].isin([training_level]) ]

t_stim_and_led_tuple = [(row['intended_fix'], row['LED_onset_time']) for _, row in df.iterrows()]

ABL_arr = df['ABL'].unique()
ABL_arr.sort()

ILD_arr = df['ILD'].unique()
ILD_arr.sort()

# percentage of LED on trials
frac_of_led = df['LED_trial'].values.sum() / df['LED_trial'].values.size
print(f'frac_of_led: {frac_of_led}')

noise_change_results_dict = {}
for i in range(N_params):
    new_noise = 0 + i * new_noise_step_size
    new_noise = new_noise**2

    with parallel_backend('threading', n_jobs=64):
        sim_results = Parallel()(
            delayed(psiam_tied_data_gen_wrapper_noise_and_V_A_change)(V_A, theta_A, ABL_arr, ILD_arr, rate_lambda, T_0, theta_E, Z_E,\
                                                               t_A_aff, t_E_aff, t_motor, L,t_stim_and_led_tuple, new_noise, new_V_A, iter_num, \
                                                                N_print, dt)
            for iter_num in range(N_sim)
        )

    print(f'V_A: {V_A}, new noise: {new_noise}')

    noise_change_results_dict[i] = {'results': sim_results, 'new_noise':  new_noise, 'new_V_A': new_V_A }

import pickle
with open(f'v_A_and_noise_change_1M.pkl', 'wb') as f:
    pickle.dump(noise_change_results_dict, f)