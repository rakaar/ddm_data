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

batch = 'LED7'
animal_id = 93
file_name = f'batch_csvs/batch_{batch}_valid_and_aborts.csv'
df = pd.read_csv(file_name)
df_animal = df[df['animal'] == animal_id]

N_sim = int(5e5)
ABL_arr = [20, 40, 60]
ILD_arr = np.sort([1,-1,2,-2,4,-4,8,-8])
ABL_samples = np.random.choice(ABL_arr, N_sim)
ILD_samples = np.random.choice(ILD_arr, N_sim)
t_stim_samples = df_animal['intended_fix'].sample(N_sim, replace=True).values


abort_params, tied_params = get_params_from_animal_pkl_file(batch, animal_id)
Z_E = (tied_params['w'] - 0.5) * 2 * tied_params['theta_E']

rate_norm_l = 0
N_print = N_sim // 5
dt = 1e-3

# Run a single large simulation with all samples
print("Running simulation for all conditions...")
sim_results = Parallel(n_jobs=30)(
    delayed(psiam_tied_data_gen_wrapper_rate_norm_fn)(
        abort_params['V_A'], abort_params['theta_A'], ABL_samples[i], ILD_samples[i],
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

# %%
# theory
IS_NORM_TIED = False

from time_vary_norm_utils import up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn_vec, rho_A_t_VEC_fn
from scipy.integrate import cumtrapz, trapezoid

def compute_PA_CA_curves(V_A, theta_A, t_A_aff, t_pts, t_stim):
    P_A = np.zeros_like(t_pts)
    mask = (t_pts + t_stim) > 0.3
    if np.any(mask):
        P_A[mask] = rho_A_t_VEC_fn((t_pts + t_stim - t_A_aff)[mask], V_A, theta_A)
    truncated_PA_area = trapezoid(P_A, t_pts)
    norm_PA = P_A / truncated_PA_area if truncated_PA_area > 0 else P_A
    C_A = cumtrapz(norm_PA, t_pts, initial=0)
    return P_A, C_A

abort_params, tied_params = get_params_from_animal_pkl_file(batch, int(animal_id))
V_A = abort_params['V_A']
theta_A = abort_params['theta_A']
t_A_aff = abort_params['t_A_aff']
rate_lambda = tied_params['rate_lambda']
T_0 = tied_params['T_0']
theta_E = tied_params['theta_E']
w = tied_params['w']
t_E_aff = tied_params['t_E_aff']
del_go = tied_params['del_go']
Z_E = (w - 0.5) * 2 * theta_E
ild_values = np.array([-16., -8., -4., -2., -1., 1., 2., 4., 8., 16.])
t_pts = np.arange(-1, 2, 0.001)
def get_P_A_C_A(batch, animal_id, abort_params):
    N_theory = int(1e3)
    file_name = f'batch_csvs/batch_{batch}_valid_and_aborts.csv'
    df = pd.read_csv(file_name)
    df_animal = df[df['animal'] == animal_id]
    P_A_mean, C_A_mean, t_stim_samples = calculate_theoretical_curves(
        df_animal, N_theory, t_pts, abort_params['t_A_aff'], abort_params['V_A'], abort_params['theta_A'], rho_A_t_fn
    )
    return P_A_mean, C_A_mean, t_stim_samples
P_A, C_A, _ = get_P_A_C_A(batch, int(animal_id), abort_params)

for ABL in sorted(valid_btn_1['ABL'].unique()):
    right_choice_probs = []
    for ILD in ild_values:

        up_mean = up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn_vec(
            t_pts, 1, P_A, C_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go,
            np.nan, np.nan, np.nan, np.nan, np.nan, 0, False, False, 10)
        down_mean = up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn_vec(
            t_pts, -1, P_A, C_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go,
            np.nan, np.nan, np.nan, np.nan, np.nan, 0, False, False, 10)
        mask_0_1 = (t_pts >= 0) & (t_pts <= 1)
        t_pts_0_1 = t_pts[mask_0_1]
        up_mean_0_1 = up_mean[mask_0_1]
        down_mean_0_1 = down_mean[mask_0_1]
        up_area = trapezoid(up_mean_0_1, t_pts_0_1)
        down_area = trapezoid(down_mean_0_1, t_pts_0_1)
        right_prob = up_area / (up_area + down_area)
        right_choice_probs.append(right_prob)
    theory_psycho[ABL] = np.array(right_choice_probs)

# %%
# --- Plot: Theory curves for different sensory delays (t_E_aff) for each ABL ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
sensory_delays = [t_E_aff, 0.01, 0.03, 0.05, 0.5]  # fitted, 10 ms, 30 ms, 50 ms, 90 ms
labels = [f'fit: {t_E_aff*1000:.0f} ms', '10 ms', '30 ms', '50 ms', '500 ms']
colors_theory = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
P_A, C_A, _ = get_P_A_C_A(batch, int(animal_id), abort_params)

for i, abl in enumerate(sorted(valid_btn_1['ABL'].unique())):
    ax = axes[i]
    # Plot simulation data
    df_abl = valid_btn_1[valid_btn_1['ABL'] == abl]
    grouped = df_abl.groupby('ILD')
    ilds = np.array(sorted(df_abl['ILD'].unique()))
    p_right = np.array([(grp['choice'] == 1).mean() for _, grp in grouped])
    ax.scatter(ilds, p_right, color='black', label='Sim', s=60)
    # Plot theory curves for each sensory delay
    for j, t_E_aff_val in enumerate(sensory_delays):
        right_choice_probs = []
        for ILD in ild_values:
            up_mean = up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn_vec(
                t_pts, 1, P_A, C_A, abl, ILD, rate_lambda, T_0, theta_E, Z_E, t_E_aff_val, del_go,
                np.nan, np.nan, np.nan, np.nan, np.nan, 0, False, False, 10)
            down_mean = up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn_vec(
                t_pts, -1, P_A, C_A, abl, ILD, rate_lambda, T_0, theta_E, Z_E, t_E_aff_val, del_go,
                np.nan, np.nan, np.nan, np.nan, np.nan, 0, False, False, 10)
            mask_0_1 = (t_pts >= 0) & (t_pts <= 1)
            t_pts_0_1 = t_pts[mask_0_1]
            up_mean_0_1 = up_mean[mask_0_1]
            down_mean_0_1 = down_mean[mask_0_1]
            up_area = trapezoid(up_mean_0_1, t_pts_0_1)
            down_area = trapezoid(down_mean_0_1, t_pts_0_1)
            right_prob = up_area / (up_area + down_area)
            right_choice_probs.append(right_prob)
        ax.plot(ild_values, right_choice_probs, color=colors_theory[j], label=labels[j], lw=0.5)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(f'ABL={abl} dB')
    ax.set_xlabel('ILD (dB)')
    if i == 0:
        ax.set_ylabel('P(Right)')
    ax.legend()
plt.tight_layout()
plt.show()

# %%
# data
# Plot psychometric curve for valid_btn_1 for each ABL (scatter only, no sigmoid)
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
colors = {20: 'tab:blue', 40: 'tab:orange', 60: 'tab:green'}
for i, abl in enumerate(sorted(valid_btn_1['ABL'].unique())):
    ax = axes[i]
    df_abl = valid_btn_1[valid_btn_1['ABL'] == abl]
    grouped = df_abl.groupby('ILD')
    ilds = np.array(sorted(df_abl['ILD'].unique()))
    p_right = np.array([(grp['choice'] == 1).mean() for _, grp in grouped])
    ax.scatter(ilds, p_right, color=colors.get(abl, None), label='Sim', s=60)
    ax.plot(ild_values, theory_psycho[abl], color=colors.get(abl, None), label='Theory')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('ILD (dB)')
    ax.set_title(f'ABL {abl}')
    ax.set_ylim(0, 1)
    if i == 0:
        ax.set_ylabel('P(choice = 1)')
    ax.legend()
fig.suptitle('Psychometric Function by ABL (valid_btn_1)', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
DEL_GO_vals = [del_go,0.01, 0.04, 0.08, 0.13, 0.16, 0.2]
labels = [f'fit: {del_go*1000:.0f} s', '0.01 s', '0.04 s', '0.08 s', '0.13 s', '0.16 s', '0.2 s']
colors_theory = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
P_A, C_A, _ = get_P_A_C_A(batch, int(animal_id), abort_params)

for i, abl in enumerate(sorted(valid_btn_1['ABL'].unique())):
    ax = axes[i]
    # Plot simulation data
    df_abl = valid_btn_1[valid_btn_1['ABL'] == abl]
    grouped = df_abl.groupby('ILD')
    ilds = np.array(sorted(df_abl['ILD'].unique()))
    p_right = np.array([(grp['choice'] == 1).mean() for _, grp in grouped])
    ax.scatter(ilds, p_right, color='black', label='Sim', s=60)
    # Plot theory curves for each sensory delay
    for j, del_go_val in enumerate(DEL_GO_vals):
        right_choice_probs = []
        for ILD in ild_values:
            up_mean = up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn_vec(
                t_pts, 1, P_A, C_A, abl, ILD, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go_val,
                np.nan, np.nan, np.nan, np.nan, np.nan, 0, False, False, 10)
            down_mean = up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn_vec(
                t_pts, -1, P_A, C_A, abl, ILD, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go_val,
                np.nan, np.nan, np.nan, np.nan, np.nan, 0, False, False, 10)
            mask_0_1 = (t_pts >= 0) & (t_pts <= 1)
            t_pts_0_1 = t_pts[mask_0_1]
            up_mean_0_1 = up_mean[mask_0_1]
            down_mean_0_1 = down_mean[mask_0_1]
            up_area = trapezoid(up_mean_0_1, t_pts_0_1)
            down_area = trapezoid(down_mean_0_1, t_pts_0_1)
            right_prob = up_area / (up_area + down_area)
            right_choice_probs.append(right_prob)
        ax.plot(ild_values, right_choice_probs, lw=0.5)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(f'ABL={abl} dB')
    ax.set_xlabel('ILD (dB)')
    if i == 0:
        ax.set_ylabel('P(Right)')
    ax.legend()
plt.tight_layout()
plt.show()

# %%
# read from files
import pickle
psycho_fits_repo_path = '/home/rlab/raghavendra/ddm_data/fit_valid_trials/psycho_fits/'

def get_params_from_animal_pkl_file(batch_name, animal_id):
    filename = os.path.join(psycho_fits_repo_path, f'psycho_fit_{batch_name}_{animal_id}.pkl')
    with open(filename, 'rb') as f:
        vp = pickle.load(f)
    vp = vp.vp
    samples = vp.sample(int(1e6))[0]
    tied_params = {
        'rate_lambda' : samples[:,0].mean(),
        'T_0' : samples[:,1].mean(),
        'theta_E' : samples[:,2].mean(),
        'w' : samples[:,3].mean(),
        't_E_aff' : samples[:,4].mean(),
        'del_go' : samples[:,5].mean()
    }
    return tied_params

params = get_params_from_animal_pkl_file('LED7', 92)
print(params)