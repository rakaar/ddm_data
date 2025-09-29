# Testing likelihood with simulated data - lapse model
# Modified to read parameters from pickle file instead of hardcoding
# %%
import pickle
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from joblib import Parallel, delayed

# Set model type (False for vanilla TIED, True for norm TIED)
IS_NORM_TIED = False

# Function to read animal parameters from pickle file
def get_params_from_animal_pkl_file(batch_name, animal_id):
    pkl_file = f'../fit_animal_by_animal/results_{batch_name}_animal_{animal_id}.pkl'
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
    vbmc_norm_tied_param_keys_map = {
        'rate_lambda_samples': 'rate_lambda',
        'T_0_samples': 'T_0',
        'theta_E_samples': 'theta_E',
        'w_samples': 'w',
        't_E_aff_samples': 't_E_aff',
        'del_go_samples': 'del_go',
        'rate_norm_l_samples': 'rate_norm_l'
    }
    abort_keyname = "vbmc_aborts_results"
    vanilla_tied_keyname = "vbmc_vanilla_tied_results"
    norm_tied_keyname = "vbmc_norm_tied_results"
    abort_params = {}
    vanilla_tied_params = {}
    norm_tied_params = {}
    if abort_keyname in fit_results_data:
        abort_samples = fit_results_data[abort_keyname]
        for param_samples_name, param_label in vbmc_aborts_param_keys_map.items():
            abort_params[param_label] = np.mean(abort_samples[param_samples_name])
    if vanilla_tied_keyname in fit_results_data:
        vanilla_tied_samples = fit_results_data[vanilla_tied_keyname]
        for param_samples_name, param_label in vbmc_vanilla_tied_param_keys_map.items():
            vanilla_tied_params[param_label] = np.mean(vanilla_tied_samples[param_samples_name])
    if norm_tied_keyname in fit_results_data:
        norm_tied_samples = fit_results_data[norm_tied_keyname]
        for param_samples_name, param_label in vbmc_norm_tied_param_keys_map.items():
            norm_tied_params[param_label] = np.mean(norm_tied_samples[param_samples_name])
    if IS_NORM_TIED:
        return abort_params, norm_tied_params
    else:
        return abort_params, vanilla_tied_params
# %%
# read random animal params
animal_id = 112
batch_name = 'LED8'

# Read parameters for the animal
abort_params, tied_params = get_params_from_animal_pkl_file(batch_name, animal_id)

# Extract individual parameters for easier access
V_A = abort_params.get('V_A', np.nan)
theta_A = abort_params.get('theta_A', np.nan)
t_A_aff = abort_params.get('t_A_aff', np.nan)

rate_lambda = tied_params.get('rate_lambda', np.nan)
T_0 = tied_params.get('T_0', np.nan)
theta_E = tied_params.get('theta_E', np.nan)
w = tied_params.get('w', np.nan)
t_E_aff = tied_params.get('t_E_aff', np.nan)
del_go = tied_params.get('del_go', np.nan)

# For norm model, also get rate_norm_l
if IS_NORM_TIED:
    rate_norm_l = tied_params.get('rate_norm_l', np.nan)
else:
    rate_norm_l = 0

print(f"Loaded parameters for {batch_name} animal {animal_id}:")
print(f"Abort params: V_A={V_A}, theta_A={theta_A}, t_A_aff={t_A_aff}")
print(f"Tied params: rate_lambda={rate_lambda}, T_0={T_0}, theta_E={theta_E}, w={w}")
print(f"t_E_aff={t_E_aff}, del_go={del_go}, rate_norm_l={rate_norm_l}")

# Fixed test parameters
t_stim_fixed = 0.25  # Fixed t_stim
ABL_fixed = 20       # Fixed ABL
ILD_fixed = 4       # Fixed ILD
lapse_prob = 0.3    # 10% lapse rate
T_lapse_max = 0.7   # Max lapse RT
N_sim = int(50e3)        # Number of simulations
dt = 1e-4       # Time step

# Calculate Z_E from w and theta_E
Z_E = (w - 0.5) * 2 * theta_E

print("=== Simple Lapse Test ===")
print(f"Fixed parameters: t_stim={t_stim_fixed}, ABL={ABL_fixed}, ILD={ILD_fixed}")
print(f"Lapse prob: {lapse_prob}, T_lapse_max: {T_lapse_max}")
print(f"Running {N_sim} simulations...")

def simulate_psiam_tied_rate_norm(V_A, theta_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_stim,
                                  t_A_aff, t_E_aff, del_go, rate_norm_l, dt, lapse_prob=0.0, T_lapse_max=1.0):

    # Lapse mechanism: with probability lapse_prob, generate random choice and RT
    if np.random.rand() < lapse_prob:
        choice = 1 if np.random.rand() >= 0.5 else -1
        rt = t_stim + np.random.uniform(0, T_lapse_max)  # Uniform distribution between 0 and T_lapse_max
        is_act = 1  # Mark as lapse
        return choice, rt, is_act

    # Normal simulation process (with probability 1 - lapse_prob)
    AI = 0; DV = Z_E; t = t_A_aff; dB = dt**0.5

    chi = 17.37; q_e = 1
    theta = theta_E * q_e
    lambda_ABL_term = (10 ** (rate_lambda * (1 - rate_norm_l) * ABL / 20))
    lambda_ILD_arg = rate_lambda * ILD / chi
    lambda_ILD_L_arg = rate_lambda * rate_norm_l * ILD / chi
    mu = (1/T_0) * lambda_ABL_term * (np.sinh(lambda_ILD_arg) / np.cosh(lambda_ILD_L_arg))
    sigma = np.sqrt( (1/T_0) * lambda_ABL_term * ( np.cosh(lambda_ILD_arg) / np.cosh(lambda_ILD_L_arg) ) )

    is_act = 0
    while True:
        AI += V_A*dt + np.random.normal(0, dB)

        if t > t_stim + t_E_aff:
            DV += mu*dt + sigma*np.random.normal(0, dB)

        t += dt

        if DV >= theta:
            choice = +1; RT = t
            break
        elif DV <= -theta:
            choice = -1; RT = t
            break

        if AI >= theta_A:
            both_AI_hit_and_EA_hit = 0
            is_act = 1
            AI_hit_time = t
            while t <= (AI_hit_time + del_go):
                if t > t_stim + t_E_aff:
                    DV += mu*dt + sigma*np.random.normal(0, dB)
                    if DV >= theta:
                        DV = theta
                        both_AI_hit_and_EA_hit = 1
                        break
                    elif DV <= -theta:
                        DV = -theta
                        both_AI_hit_and_EA_hit = -1
                        break
                t += dt
            break

    if is_act == 1:
        RT = AI_hit_time
        if both_AI_hit_and_EA_hit != 0:
            choice = both_AI_hit_and_EA_hit
        else:
            randomly_choose_up = np.random.rand() >= 0.5
            if randomly_choose_up:
                choice = 1
            else:
                choice = -1

    return choice, RT, is_act

def simulate_single_trial(iter_num):
    """Wrapper function for parallel simulation of a single trial"""
    choice, rt, is_act = simulate_psiam_tied_rate_norm(
        V_A, theta_A, ABL_fixed, ILD_fixed, rate_lambda, T_0, theta_E, Z_E, t_stim_fixed,
        t_A_aff, t_E_aff, del_go, rate_norm_l, dt, lapse_prob, T_lapse_max
    )
    return {'choice': choice, 'rt': rt, 't_stim': t_stim_fixed, 'is_act': is_act}

# Run simulations with fixed parameters in parallel
print(f"Running {N_sim} simulations in parallel with dt={dt}, lapse_prob={lapse_prob}...")
sim_results = Parallel(n_jobs=30)(
    delayed(simulate_single_trial)(iter_num) for iter_num in tqdm(range(N_sim))
)

# Convert to DataFrame
sim_df = pd.DataFrame(sim_results)

# Calculate rt - t_stim
sim_df['rt_minus_t_stim'] = sim_df['rt'] - sim_df['t_stim']

# Filter out invalid trials: remove rows where rt < t_stim AND rt < 0.3
sim_df = sim_df[~((sim_df['rt'] < sim_df['t_stim']) & (sim_df['rt'] < 0.3))]

# %%
# Filter valid trials (rt - t_stim between 0 and 1)
valid_trials = sim_df[(sim_df['rt_minus_t_stim'] >= 0) & (sim_df['rt_minus_t_stim'] <= 1)]
valid_trials_ch_1 = valid_trials[valid_trials['choice'] == 1]
valid_trials_ch_2 = valid_trials[valid_trials['choice'] == -1]

# %%
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'fit_animal_by_animal'))
from time_vary_norm_utils import up_or_down_RTs_fit_fn, cum_pro_and_reactive_time_vary_fn

def likeli(rt, choice):
    t_stim = t_stim_fixed
    phi_params_obj = np.nan
    rate_norm_l = np.nan
    is_norm = False
    is_time_vary = False
    K_max = 10
    T_trunc = 0.3
    lapse_rt_window = T_lapse_max
    pdf = up_or_down_RTs_fit_fn(
            rt, choice,
            V_A, theta_A, t_A_aff,
            t_stim, ABL_fixed, ILD_fixed, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go,
            phi_params_obj, rate_norm_l,
            is_norm, is_time_vary, K_max)

    trunc_factor_p_joint = cum_pro_and_reactive_time_vary_fn(
                            t_stim + 1, T_trunc,
                            V_A, theta_A, t_A_aff,
                            t_stim, ABL_fixed, ILD_fixed, rate_lambda, T_0, theta_E, Z_E, t_E_aff,
                            phi_params_obj, rate_norm_l,
                            is_norm, is_time_vary, K_max) \
                            - \
                            cum_pro_and_reactive_time_vary_fn(
                            t_stim, T_trunc,
                            V_A, theta_A, t_A_aff,
                            t_stim, ABL_fixed, ILD_fixed, rate_lambda, T_0, theta_E, Z_E, t_E_aff,
                            phi_params_obj, rate_norm_l,
                            is_norm, is_time_vary, K_max)

    pdf /= (trunc_factor_p_joint + 1e-20)

    # Lapse mechanism: uniform distribution between 0 and T_lapse_max
    # Since T_lapse_max <= 1.0 and data is filtered to rt-t_stim <= 1,
    # all lapse RTs are included without truncation
    in_lapse_window = (rt >= t_stim) and (rt < t_stim + lapse_rt_window)
    lapse_pdf = (0.5 / lapse_rt_window) if in_lapse_window else 0.0


    included_lapse_pdf = (1 - lapse_prob) * pdf + lapse_prob * lapse_pdf
    included_lapse_pdf = max(included_lapse_pdf, 1e-50)
    if np.isnan(included_lapse_pdf):
        print(f'nan pdf rt = {rt}, t_stim = {t_stim}')
        raise ValueError(f'nan pdf rt = {rt}, t_stim = {t_stim}')

    return included_lapse_pdf
# %%

t_pts = np.arange(0,1,0.001)

pdf_ch_1 = np.array([likeli(t+t_stim_fixed, 1) for t in t_pts])
pdf_ch_2 = np.array([likeli(t+t_stim_fixed, -1) for t in t_pts])


bins = np.arange(0,1,0.005)
sim_up_hist, _ = np.histogram(valid_trials_ch_1['rt_minus_t_stim'], bins=bins, density=True)
sim_down_hist, _ = np.histogram(valid_trials_ch_2['rt_minus_t_stim'], bins=bins, density=True)

frac_up = len(valid_trials_ch_1) / len(valid_trials)
frac_down = len(valid_trials_ch_2) / len(valid_trials)

sim_up_hist *= frac_up
sim_down_hist *= frac_down
bin_centers = (bins[:-1] + bins[1:]) / 2
plt.plot(bin_centers, sim_up_hist)
plt.plot(t_pts, pdf_ch_1)

plt.plot(bin_centers, -sim_down_hist)
plt.plot(t_pts, -pdf_ch_2)
plt.savefig('lapses_sim_and_theory.png', dpi=300, bbox_inches='tight')

# %%
from scipy.integrate import trapezoid
a1 = trapezoid(pdf_ch_1, t_pts)
a2 = trapezoid(pdf_ch_2, t_pts)


print(f'theory up: {a1 :.3f}, sim up = {frac_up :.3f}')
print(f'theory down: {a2 :.3f}, sim down = {frac_down :.3f}')
print(f'theory up + down: {a1 + a2 :.3f}, sim up + down = {frac_up + frac_down :.3f}')
# %%
