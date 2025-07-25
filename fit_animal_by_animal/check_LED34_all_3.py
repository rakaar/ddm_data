# %%
import pandas as pd
import os
import numpy as np
# %%
# Data
df_batch = pd.read_csv('batch_csvs/batch_LED34_even_valid_and_aborts.csv')
animal = 52
df_animal = df_batch[df_batch['animal'] == animal]

df_animal_aborts = df_animal[df_animal['abort_event'] == 3]
df_animal_success = df_animal[df_animal['success'].isin([1,-1])]

df_aborts_filter = df_animal_aborts[df_animal_aborts['TotalFixTime'] > 0.17]
df_success_filter = df_animal_success[df_animal_success['TotalFixTime'] \
                - df_animal_success['intended_fix'] <= 1]


ABL_arr = [20, 40, 60]
ILD_arr = [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]

print(f"{'ABL':<5} | {'ILD':<5} | {'Area Up':<10} | {'Area Down':<10} | {'Trunc Factor':<15}")
print("-" * 60)

for ABL in ABL_arr:
    for ILD in ILD_arr:
        df_success_filter_ABL_ILD = df_success_filter[(df_success_filter['ABL'] == ABL)\
                         & (df_success_filter['ILD'] == ILD)]
        
        up_choices = df_success_filter_ABL_ILD[df_success_filter_ABL_ILD['choice'] == 1]
        down_choices = df_success_filter_ABL_ILD[df_success_filter_ABL_ILD['choice'] == -1]
        
        n_up = len(up_choices)
        n_down = len(down_choices)
        n_aborts = len(df_aborts_filter)

        denominator = n_up + n_down + n_aborts
        if denominator > 0:
            area_up = n_up / denominator
            area_down = n_down / denominator
            trunc_factor = (n_up + n_down) / denominator
        else:
            area_up = 0
            area_down = 0
            trunc_factor = 0

        print(f'{ABL:<5} | {ILD:<5} | {area_up:<10.3f} | {area_down:<10.3f} | {trunc_factor:<15.3f}')
        
# %%
# theory
# what truncaction factor do u get when you use such sim params

from time_vary_norm_utils import cum_A_t_fn

V_A = 0.6284559742615939
theta_A = 3.1637365154388015
t_A_aff = -2.60894206221376


t_stim_samples = df_animal['intended_fix'].sample(int(1e5), replace=True)
trunc_fac = np.zeros_like(t_stim_samples)


for i, t_stim in enumerate(t_stim_samples):
    area_till_stim = cum_A_t_fn(t_stim - t_A_aff, V_A, theta_A)
    left_trunc = 1 - cum_A_t_fn(0.15 - t_A_aff, V_A, theta_A)
    trunc_fac[i] =  1 - (area_till_stim / left_trunc)


print(trunc_fac.mean())


# %%
t_stim_samples = df_animal['intended_fix'].sample(int(1e5), replace=True)

t_pts = np.arange(-2, 2, 0.001)
trunc_fac_samples = np.zeros((len(t_stim_samples)))

V_A = 0.6284559742615939
theta_A = 3.1637365154388015
t_A_aff = -2.60894206221376

rate_lambda  = 0.13652
T_0       = 1.94948 * 1e-3
theta_E       = 17.27168
Z_E           = -1.28608
t_E_aff       = 67.47528 *1e-3
del_go   = 0.16574
is_norm = False
rate_norm_l = 0
phi_params_obj = np.nan
N_theory = int(1e3)
t_pts = np.arange(-2, 2, 0.001)
K_max = 10

ABL = 20
ILD = 8
T_trunc = 0.17
MODEL_TYPE = 'vanilla'

abort_params, tied_params, rate_norm_l, is_norm = get_params_from_animal_pkl_file('LED34_even', 52)

P_A_mean, C_A_mean, t_stim_samples = calculate_theoretical_curves(
    df_animal, N_theory, t_pts, abort_params['t_A_aff'], abort_params['V_A'], abort_params['theta_A'], rho_A_t_fn
    )

for idx, t_stim in enumerate(t_stim_samples):
    trunc_fac_samples[idx] = cum_pro_and_reactive_time_vary_fn(
        t_stim + 1, T_trunc,
        abort_params['V_A'], abort_params['theta_A'], abort_params['t_A_aff'],
        t_stim, ABL, ILD, tied_params['rate_lambda'], tied_params['T_0'], tied_params['theta_E'], Z_E, t_E_aff,
        phi_params_obj, rate_norm_l, 
        is_norm, False, K_max) \
        - cum_pro_and_reactive_time_vary_fn(
        t_stim, T_trunc,
        abort_params['V_A'], abort_params['theta_A'], abort_params['t_A_aff'],
        t_stim, ABL, ILD, tied_params['rate_lambda'], tied_params['T_0'], tied_params['theta_E'], Z_E, t_E_aff,
        phi_params_obj, rate_norm_l, 
        is_norm, False, K_max) + 1e-10
trunc_factor = np.mean(trunc_fac_samples)

up_mean = np.array([up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn(
    t, 1,
    P_A_mean[i], C_A_mean[i],
    ABL, ILD, tied_params['rate_lambda'], tied_params['T_0'], tied_params['theta_E'], (tied_params['w'] - 0.5)  * 2 * tied_params['theta_E'], tied_params['t_E_aff'], tied_params['del_go'],
    phi_params_obj, rate_norm_l, 
    is_norm, False, K_max) for i, t in enumerate(t_pts)])
down_mean = np.array([up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn(
    t, -1,
    P_A_mean[i], C_A_mean[i],
    ABL, ILD, tied_params['rate_lambda'], tied_params['T_0'], tied_params['theta_E'], (tied_params['w'] - 0.5)  * 2 * tied_params['theta_E'], tied_params['t_E_aff'], tied_params['del_go'],
    phi_params_obj, rate_norm_l, 
    is_norm, False, K_max) for i, t in enumerate(t_pts)])
mask_0_1 = (t_pts >= 0) & (t_pts <= 1)
t_pts_0_1 = t_pts[mask_0_1]
up_mean_0_1 = up_mean[mask_0_1]
down_mean_0_1 = down_mean[mask_0_1]
# %%
plt.plot(t_pts, up_mean, color='g')
plt.plot(t_pts, -down_mean, color='r')
plt.plot(t_pts, up_mean + down_mean, color='k')
print(f'up area = {np.trapz(up_mean, t_pts)}')
print(f'down area = {np.trapz(down_mean, t_pts)}')
print(np.trapz(up_mean, t_pts) + np.trapz(down_mean, t_pts))
plt.show()

# %%
# area under curve
up_area = trapz(up_mean_0_1, t_pts_0_1)
down_area = trapz(down_mean_0_1, t_pts_0_1)
print(f"Up area: {up_area}")
print(f"Down area: {down_area}")
print(f"Trunc factor: {trunc_factor}")
# %%
# plt.hist(df_success_filter['TotalFixTime'] - df_success_filter['intended_fix'], density=True);
print(len(df_success_filter)/ (len(df_success_filter) + len(df_aborts_filter)))
# %%
print(len(df_aborts_filter))
print(len(df_success_filter))
257/(13376 + 257)

# %%
from joblib import Parallel, delayed
n_jobs = 29
N_sim = int(1e6)
N_print = max(1, N_sim // 5)
dt = 1e-3
t_stim_samples = df_success_filter['intended_fix'].sample(N_sim, replace=True).values
ABL_samples = df_success_filter['ABL'].sample(N_sim, replace=True).values
ILD_samples = df_success_filter['ILD'].sample(N_sim, replace=True).values

sim_results = Parallel(n_jobs=n_jobs)(
    delayed(psiam_tied_data_gen_wrapper_rate_norm_fn)(
        abort_params['V_A'], abort_params['theta_A'], ABL_samples[iter_num], ILD_samples[iter_num], tied_params['rate_lambda'], tied_params['T_0'],
        tied_params['theta_E'], Z_E, abort_params['t_A_aff'], tied_params['t_E_aff'], tied_params['del_go'],
        t_stim_samples[iter_num], rate_norm_l, iter_num, N_print, dt
    ) for iter_num in range(N_sim)
)
# %%
sim_results_df = pd.DataFrame(sim_results)
sim_results_df_valid = sim_results_df[sim_results_df['rt'] - sim_results_df['t_stim'] > 0]
sim_results_df_valid_lt_1 = sim_results_df_valid[sim_results_df_valid['rt'] - sim_results_df_valid['t_stim'] <= 1]
sim_results_aborts = sim_results_df[sim_results_df['rt'] - sim_results_df['t_stim'] <= 0]
sim_results_abort_filter = sim_results_aborts[sim_results_aborts['rt'] > 0.17]

# 

# %%
print(len(sim_results_abort_filter) / (len(sim_results_df_valid_lt_1) + len(sim_results_abort_filter)))

print("\n--- Simulation Results Analysis ---")
print(f"{'ABL':<5} | {'ILD':<5} | {'Area Up':<10} | {'Area Down':<10} | {'Trunc Factor':<15}")
print("-" * 60)

for ABL in ABL_arr:
    for ILD in ILD_arr:
        sim_valid_filter_ABL_ILD = sim_results_df_valid_lt_1[(sim_results_df_valid_lt_1['ABL'] == ABL) & (sim_results_df_valid_lt_1['ILD'] == ILD)]
        # sim_abort_filter_ABL_ILD = sim_results_abort_filter[(sim_results_abort_filter['ABL'] == ABL) & (sim_results_abort_filter['ILD'] == ILD)]

        up_choices = sim_valid_filter_ABL_ILD[sim_valid_filter_ABL_ILD['choice'] == 1]
        down_choices = sim_valid_filter_ABL_ILD[sim_valid_filter_ABL_ILD['choice'] == -1]

        n_up = len(up_choices)
        n_down = len(down_choices)
        n_aborts = len(sim_results_abort_filter)

        denominator = n_up + n_down + n_aborts
        if denominator > 0:
            area_up = n_up / denominator
            area_down = n_down / denominator
            trunc_factor = (n_up + n_down) / denominator
        else:
            area_up = 0
            area_down = 0
            trunc_factor = 0

        print(f'{ABL:<5} | {ILD:<5} | {area_up:<10.3f} | {area_down:<10.3f} | {trunc_factor:<15.3f}')

#%% Theoretical Truncation Factor Analysis
print("\n--- Theoretical Truncation Factor Analysis ---")
print(f"{'ABL':<5} | {'ILD':<5} | {'Theoretical Trunc Factor':<25}")
print("-" * 45)

trunc_fac_samples = np.zeros(int(1e5))
t_stim_samples = df_animal['intended_fix'].sample(int(1e5), replace=True)
print(T_trunc)
for ABL in ABL_arr:
    for ILD in ILD_arr:
        for idx, t_stim in enumerate(t_stim_samples):
            trunc_fac_samples[idx] = cum_pro_and_reactive_time_vary_fn(
                t_stim + 1, T_trunc,
                abort_params['V_A'], abort_params['theta_A'], abort_params['t_A_aff'],
                t_stim, ABL, ILD, tied_params['rate_lambda'], tied_params['T_0'], tied_params['theta_E'], Z_E, t_E_aff,
                phi_params_obj, rate_norm_l, 
                is_norm, False, K_max) \
                - cum_pro_and_reactive_time_vary_fn(
                t_stim, T_trunc,
                abort_params['V_A'], abort_params['theta_A'], abort_params['t_A_aff'],
                t_stim, ABL, ILD, tied_params['rate_lambda'], tied_params['T_0'], tied_params['theta_E'], Z_E, t_E_aff,
                phi_params_obj, rate_norm_l, 
                is_norm, False, K_max) + 1e-10
        trunc_factor = np.mean(trunc_fac_samples)
        print(f'{ABL:<5} | {ILD:<5} | {trunc_factor:<25.3f}')
# %%
t_stim_samples = df_animal['intended_fix'].sample(int(1e5), replace=True)
for idx, t_stim in enumerate(t_stim_samples):
    trunc_fac_samples[idx] = cum_pro_and_reactive_time_vary_fn(
        t_stim + 1, T_trunc,
        abort_params['V_A'], abort_params['theta_A'], abort_params['t_A_aff'],
        t_stim, ABL, ILD, tied_params['rate_lambda'], tied_params['T_0'], tied_params['theta_E'], Z_E, t_E_aff,
        phi_params_obj, rate_norm_l, 
        is_norm, False, K_max) \
        - cum_pro_and_reactive_time_vary_fn(
        t_stim, T_trunc,
        abort_params['V_A'], abort_params['theta_A'], abort_params['t_A_aff'],
        t_stim, ABL, ILD, tied_params['rate_lambda'], tied_params['T_0'], tied_params['theta_E'], Z_E, t_E_aff,
        phi_params_obj, rate_norm_l, 
        is_norm, False, K_max) + 1e-10
trunc_factor = np.mean(trunc_fac_samples)
# %%
print(trunc_factor)
T_trunc 

# %%
plt.hist(t_stim_samples)

# %%
t_pts = np.arange(-3, 5, 0.001)
print(abort_params['t_A_aff'], abort_params['V_A'], abort_params['theta_A'])
rho_den = [rho_A_t_fn(t- abort_params['t_A_aff'], abort_params['V_A'], abort_params['theta_A']) for t in t_pts]
plt.axvline(0)
plt.axvline(0.25)

plt.plot(t_pts, rho_den)