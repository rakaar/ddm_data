# %%
from scipy.integrate import trapezoid
from scipy.integrate import cumulative_trapezoid as cumtrapz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from joblib import Parallel, delayed
from tqdm import tqdm
from time_vary_and_norm_simulators import psiam_tied_data_gen_wrapper_rate_norm_fn
import pickle
import warnings
from types import SimpleNamespace
from animal_wise_plotting_utils import calculate_theoretical_curves
from time_vary_norm_utils import (
    up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn, 
    cum_pro_and_reactive_time_vary_fn, 
    rho_A_t_fn, 
    cum_A_t_fn
)
from collections import defaultdict
import random

# %%
def get_params_from_animal_pkl_file(batch_name, animal_id):
    # read the pkl file
    pkl_file = f'results_{batch_name}_animal_{animal_id}.pkl'
    with open(pkl_file, 'rb') as f:
        fit_results_data = pickle.load(f)
    
    vbmc_aborts_param_keys_map = {
        'V_A_samples': 'V_A',
        'theta_A_samples': 'theta_A',
        't_A_aff_samp': 't_A_aff'
    }
    vbmc_vanilla_tied_param_keys_map = {
        'rate_lambda_samples': 'rate_lambda',
        'T_0_samples': 'T_0',
        'theta_E_samples': 'theta_E',
        'w_samples': 'w',
        't_E_aff_samples': 't_E_aff',
        'del_go_samples': 'del_go'
    }
    
    abort_keyname = "vbmc_aborts_results"
    vanilla_tied_keyname = "vbmc_vanilla_tied_results"

    abort_params = {}
    tied_params = {}

    if abort_keyname in fit_results_data:
        abort_samples = fit_results_data[abort_keyname]
        for param_samples_name, param_label in vbmc_aborts_param_keys_map.items():
            abort_params[param_label] = np.mean(abort_samples[param_samples_name])
    
    if vanilla_tied_keyname in fit_results_data:
        vanilla_tied_samples = fit_results_data[vanilla_tied_keyname]
        for param_samples_name, param_label in vbmc_vanilla_tied_param_keys_map.items():
            tied_params[param_label] = np.mean(vanilla_tied_samples[param_samples_name])
    
    return abort_params, tied_params    


# %%

batch = 'SD'
animal_id = 50
file_name = f'batch_csvs/batch_{batch}_valid_and_aborts.csv'
df = pd.read_csv(file_name)
df_animal = df[df['animal'] == animal_id]

N_sim = int(1e6)
ILD = 1
ABL = 20
t_stim_samples = df_animal['intended_fix'].sample(N_sim, replace=True).values


abort_params, tied_params = get_params_from_animal_pkl_file('SD', 50)
Z_E = (tied_params['w'] - 0.5) * 2 * tied_params['theta_E']

rate_norm_l = 0
N_print = N_sim // 5
dt = 1e-3

# Run a single large simulation with all samples
print("Running simulation for all conditions...")
sim_results = Parallel(n_jobs=30)(
    delayed(psiam_tied_data_gen_wrapper_rate_norm_fn)(
        abort_params['V_A'], abort_params['theta_A'], ABL, ILD,
        tied_params['rate_lambda'], tied_params['T_0'], tied_params['theta_E'], Z_E,
        abort_params['t_A_aff'], tied_params['t_E_aff'], tied_params['del_go'],
        t_stim_samples[i], rate_norm_l, i, N_print, dt
    ) for i in tqdm(range(N_sim))
)
# %%

sim_results_df = pd.DataFrame(sim_results)
rt_wrt_stim  = sim_results_df['rt'] - sim_results_df['t_stim']
plt.hist(rt_wrt_stim, bins=np.arange(-2,2,0.02))
plt.show()

# %%
rt_wrt_fix  = sim_results_df['rt']
plt.hist(rt_wrt_fix, bins=np.arange(-2,2,0.02))
plt.show()

# %%
# remove (rt < t_stim AND rt < 0.3)
sim_results_df_trunc_aborts = sim_results_df[~((sim_results_df['rt'] < sim_results_df['t_stim']) & (sim_results_df['rt'] < 0.3))].reset_index(drop=True)
print(f'len sim results df trunc aborts: {len(sim_results_df_trunc_aborts)}')
# %%
# rows where rt - t_stim is between 0 and 1
valid_btn_1 = sim_results_df_trunc_aborts[(sim_results_df_trunc_aborts['rt'] - sim_results_df_trunc_aborts['t_stim'] >= 0) & (sim_results_df_trunc_aborts['rt'] - sim_results_df_trunc_aborts['t_stim'] <= 1)]
print(f'len valid btn 1: {len(valid_btn_1)}')

# %%
valid_btn_1_rtd = valid_btn_1['rt'] - valid_btn_1['t_stim']
plt.hist(valid_btn_1_rtd, bins=np.arange(0,1,0.02), density=True)
plt.show()

# %%
aborts_df = sim_results_df[sim_results_df['rt'] < sim_results_df['t_stim']]
aborts_to_be_removed = aborts_df[aborts_df['rt'] < 0.3]

# PA area = 
print(f'area of PA remaining = {(len(aborts_df) - len(aborts_to_be_removed)) / len(aborts_df)}')
# %%
def calculate_theoretical_curves(df_valid_and_aborts, N_theory, t_pts, t_A_aff, V_A, theta_A, rho_A_t_fn):
    """
    Calculate theoretical P_A_mean and C_A_mean curves.
    
    Args:
        df_valid_and_aborts: Dataframe with valid trials and aborts
        N_theory: Number of samples for theoretical calculation
        t_pts: Time points for evaluation
        t_A_aff: Afferent time
        V_A: V_A parameter
        theta_A: theta_A parameter
        rho_A_t_fn: Function to compute rho_A_t
        
    Returns:
        P_A_mean: Mean probability
        C_A_mean: Cumulative mean probability
        t_stim_samples: Samples used for calculation
    """
    t_stim_samples = df_valid_and_aborts['intended_fix'].sample(N_theory, replace=True).values
    
    P_A_samples = np.zeros((N_theory, len(t_pts)))
    t_trunc = 0.3
    for idx, t_stim in enumerate(t_stim_samples):
        # Apply user-defined logic for each t
        def _pa_val(t):
            if (t + t_stim) <= t_trunc:
                return 0
            else:
                return rho_A_t_fn(t + t_stim - t_A_aff, V_A, theta_A)
        P_A_samples[idx, :] = [_pa_val(t) for t in t_pts]

    
    from scipy.integrate import trapezoid
    P_A_mean = np.mean(P_A_samples, axis=0)
    area = trapezoid(P_A_mean, t_pts)
    # mask_neg2_0 = (t_pts >= -2) & (t_pts <= 0)
    # area_neg2_0 = trapezoid(P_A_mean[mask_neg2_0], t_pts[mask_neg2_0])
    # print(f'area of PA from -2 to 0 = {area_neg2_0}')

    if area != 0:
        P_A_mean = P_A_mean / area
    C_A_mean = cumtrapz(P_A_mean, t_pts, initial=0)
    
    return P_A_mean, C_A_mean, t_stim_samples

def get_P_A_C_A(batch, animal_id, abort_params):
    N_theory = int(1e4)
    file_name = f'batch_csvs/batch_{batch}_valid_and_aborts.csv'
    df = pd.read_csv(file_name)
    df_animal = df[df['animal'] == animal_id]
    t_pts = np.arange(-2, 2, 0.001)
    P_A_mean, C_A_mean, t_stim_samples = calculate_theoretical_curves(
        df_animal, N_theory, t_pts, abort_params['t_A_aff'], abort_params['V_A'], abort_params['theta_A'], rho_A_t_fn
        )
    
    return P_A_mean, C_A_mean, t_stim_samples


def get_theoretical_RTD_from_params(P_A_mean, C_A_mean, t_stim_samples, abort_params, vanilla_tied_params, ABL, ILD):
    phi_params_obj = np.nan
    rate_norm_l = 0
    is_norm = False
    is_time_vary = False
    K_max = 10
    T_trunc = 0.3
    t_pts = np.arange(-2, 2, 0.001)
    trunc_fac_samples = np.zeros((len(t_stim_samples)))

    Z_E = (vanilla_tied_params['w'] - 0.5) * 2 * vanilla_tied_params['theta_E']
    
    up_mean = np.array([up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn(
                t, 1,
                P_A_mean[i], C_A_mean[i],
                ABL, ILD, vanilla_tied_params['rate_lambda'], vanilla_tied_params['T_0'], vanilla_tied_params['theta_E'], Z_E, vanilla_tied_params['t_E_aff'], vanilla_tied_params['del_go'],
                phi_params_obj, rate_norm_l, 
                is_norm, is_time_vary, K_max) for i, t in enumerate(t_pts)])
    down_mean = np.array([up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn(
            t, -1,
            P_A_mean[i], C_A_mean[i],
            ABL, ILD, vanilla_tied_params['rate_lambda'], vanilla_tied_params['T_0'], vanilla_tied_params['theta_E'], Z_E, vanilla_tied_params['t_E_aff'], vanilla_tied_params['del_go'],
            phi_params_obj, rate_norm_l, 
            is_norm, is_time_vary, K_max) for i, t in enumerate(t_pts)])
            
    
   
    # --- Old approach: mask first, then normalize ---
    # mask_0_1 = (t_pts >= 0) & (t_pts <= 1)
    # t_pts_0_1 = t_pts[mask_0_1]
    # up_mean_0_1 = up_mean[mask_0_1]
    # down_mean_0_1 = down_mean[mask_0_1]
    # 
    # # Normalize theory curves
    # up_theory_mean_norm = up_mean_0_1 
    # down_theory_mean_norm = down_mean_0_1 
    # up_plus_down_mean = up_theory_mean_norm + down_theory_mean_norm
    # area = trapezoid(up_plus_down_mean, t_pts_0_1)
    # if area != 0:
    #     up_plus_down_mean = up_plus_down_mean / area
    # print(f'area: {area}')
    # return t_pts_0_1, up_plus_down_mean

    # --- New approach: normalize first, then mask ---
    up_plus_down = up_mean + down_mean
    # area_full = trapezoid(up_plus_down, t_pts)
    # if area_full != 0:
    #     up_plus_down_norm = up_plus_down / area_full
    # else:
    #     up_plus_down_norm = up_plus_down
    # mask_0_1 = (t_pts >= 0) & (t_pts <= 1)
    # t_pts_0_1 = t_pts[mask_0_1]
    # up_plus_down_mean = up_plus_down_norm[mask_0_1]
    # print(f'area (full): {area_full}')
    # return t_pts_0_1, up_plus_down_mean

    # --- Corrected: mask first, then normalize ---
    mask_0_1 = (t_pts >= 0) & (t_pts <= 1)
    t_pts_0_1 = t_pts[mask_0_1]
    up_plus_down_masked = up_plus_down[mask_0_1]
    area_masked = trapezoid(up_plus_down_masked, t_pts_0_1)
    if area_masked != 0:
        up_plus_down_mean = up_plus_down_masked / area_masked
    else:
        up_plus_down_mean = up_plus_down_masked
    print(f'area (masked): {area_masked}')
    return t_pts_0_1, up_plus_down_mean

p_a, c_a, ts_samp = get_P_A_C_A(batch, int(animal_id), abort_params)
t_pts_0_1, up_plus_down = get_theoretical_RTD_from_params(
                            p_a, c_a, ts_samp, abort_params, tied_params, ABL, ILD
                        )

# %%
plt.plot(t_pts_0_1, up_plus_down)
plt.hist(valid_btn_1_rtd, bins=np.arange(0,1,0.02), density=True, histtype='step', color='r')

plt.show()
# area t_pts_0_1, up_plus_down
area = trapezoid(up_plus_down, t_pts_0_1)
print(f'area: {area}')
# %%
trunc_factor_sim = len(valid_btn_1) / len(sim_results_df_trunc_aborts)
print(trunc_factor_sim)

# %%
# theory
T_trunc = 0.3
N_theory = int(1e5)
phi_params_obj = np.nan
is_norm = False
is_time_vary = False
K_max = 10
t_stim_samples = np.random.choice(sim_results_df_trunc_aborts['t_stim'], size=N_theory)
trunc_fac_samples = np.zeros((N_theory))
for idx, t_stim in enumerate(t_stim_samples):
        trunc_fac_samples[idx] = cum_pro_and_reactive_time_vary_fn(
                        t_stim + 1, T_trunc,
                        abort_params['V_A'], abort_params['theta_A'], abort_params['t_A_aff'],
                        t_stim, ABL, ILD, tied_params['rate_lambda'], tied_params['T_0'], tied_params['theta_E'], Z_E, tied_params['t_E_aff'],
                        phi_params_obj, rate_norm_l, 
                        is_norm, is_time_vary, K_max) \
                        - \
                        cum_pro_and_reactive_time_vary_fn(
                        t_stim, T_trunc,
                        abort_params['V_A'], abort_params['theta_A'], abort_params['t_A_aff'],
                        t_stim, ABL, ILD, tied_params['rate_lambda'], tied_params['T_0'], tied_params['theta_E'], Z_E, tied_params['t_E_aff'],
                        phi_params_obj, rate_norm_l, 
                        is_norm, is_time_vary, K_max) + 1e-10
trunc_factor_theory = np.mean(trunc_fac_samples)
print(trunc_factor_theory)

# %%
# print theory and sim trunc factor
print(f"Sim trunc factor: {trunc_factor_sim}")
print(f"Theory trunc factor: {trunc_factor_theory}")
# diff btn them
print(f"Diff: {trunc_factor_sim - trunc_factor_theory}")
# %%
