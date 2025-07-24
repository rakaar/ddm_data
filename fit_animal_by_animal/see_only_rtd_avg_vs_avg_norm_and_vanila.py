"""
Unified RTD analysis script for vanilla and normalized TIED models.

Set MODEL_TYPE = 'vanilla' or 'norm' at the top to switch between models.
- 'vanilla': uses vbmc_vanilla_tied_results from pickle, is_norm=False, rate_norm_l=0
- 'norm':    uses vbmc_norm_tied_results from pickle, is_norm=True, rate_norm_l from pickle

All downstream logic is automatically adjusted based on this flag.
"""
# %%
MODEL_TYPE = 'norm'
print(f"Processing MODEL_TYPE: {MODEL_TYPE}")


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
from scipy.stats import gaussian_kde

def get_simulation_RTD_KDE(
    abort_params, tied_params, rate_norm_l, Z_E, ABL, ILD, t_stim_samples, N_sim, N_print, dt, n_jobs=30
):
    """
    Run the simulation for given parameters and return KDE arrays for RTD.
    Returns: x_vals, kde_vals
    """
    sim_results = Parallel(n_jobs=n_jobs)(
        delayed(psiam_tied_data_gen_wrapper_rate_norm_fn)(
            abort_params['V_A'], abort_params['theta_A'], ABL, ILD, tied_params['rate_lambda'], tied_params['T_0'],
            tied_params['theta_E'], Z_E, abort_params['t_A_aff'], tied_params['t_E_aff'], tied_params['del_go'],
            t_stim_samples[iter_num], rate_norm_l, iter_num, N_print, dt
        ) for iter_num in range(N_sim)
    )
    sim_results_df = pd.DataFrame(sim_results)
    sim_results_df_valid = sim_results_df[sim_results_df['rt'] - sim_results_df['t_stim'] > -0.1]
    sim_results_df_valid_lt_1 = sim_results_df_valid[sim_results_df_valid['rt'] - sim_results_df_valid['t_stim'] <= 1]
    sim_rt = sim_results_df_valid_lt_1['rt'] - sim_results_df_valid_lt_1['t_stim']
    kde = gaussian_kde(sim_rt)
    x_vals = np.arange(-0.12, 1, 0.01)
    kde_vals = kde(x_vals)
    return x_vals, kde_vals

# %%
# Define desired batches
DESIRED_BATCHES = ['SD', 'LED34', 'LED6', 'LED8', 'LED7', 'LED34_even']
batch_dir = os.path.join(os.path.dirname(__file__), 'batch_csvs')
batch_files = [f'batch_{batch_name}_valid_and_aborts.csv' for batch_name in DESIRED_BATCHES]

merged_data = pd.concat([
    pd.read_csv(os.path.join(batch_dir, fname)) for fname in batch_files if os.path.exists(os.path.join(batch_dir, fname))
], ignore_index=True)

merged_valid = merged_data[merged_data['success'].isin([1, -1])].copy()

# --- Print animal table ---
batch_animal_pairs = sorted(list(map(tuple, merged_valid[['batch_name', 'animal']].drop_duplicates().values)))

print(f"Found {len(batch_animal_pairs)} batch-animal pairs from {len(set(p[0] for p in batch_animal_pairs))} batches:")

if batch_animal_pairs:
    batch_to_animals = defaultdict(list)
    for batch, animal in batch_animal_pairs:
        # Ensure animal is a string and we don't add duplicates
        animal_str = str(animal)
        if animal_str not in batch_to_animals[batch]:
            batch_to_animals[batch].append(animal_str)

    # Determine column widths for formatting
    max_batch_len = max(len(b) for b in batch_to_animals.keys()) if batch_to_animals else 0
    animal_strings = {b: ', '.join(sorted(a)) for b, a in batch_to_animals.items()}
    max_animals_len = max(len(s) for s in animal_strings.values()) if animal_strings else 0

    # Header
    print(f"{'Batch':<{max_batch_len}}  {'Animals'}")
    print(f"{'=' * max_batch_len}  {'=' * max_animals_len}")

    # Rows
    for batch in sorted(animal_strings.keys()):
        animals_str = animal_strings[batch]
        print(f"{batch:<{max_batch_len}}  {animals_str}")

# %%

def get_animal_RTD_data(batch_name, animal_id, ABL, ILD, bins):
    file_name = f'batch_csvs/batch_{batch_name}_valid_and_aborts.csv'
    df = pd.read_csv(file_name)
    df = df[(df['animal'] == animal_id) & (df['ABL'] == ABL) & (df['ILD'] == ILD) & (df['success'].isin([1, -1]))]
    # df = df[(df['animal'] == animal_id) & (df['ABL'] == ABL) & (df['ILD'] == ILD) \
    #     & ((df['RTwrtStim'] <= 1) & (df['RTwrtStim'] >= -0.1))]

    bin_centers = (bins[:-1] + bins[1:]) / 2
    if df.empty:
        print(f"No data found for batch {batch_name}, animal {animal_id}, ABL {ABL}, ILD {ILD}. Returning NaNs.")
        rtd_hist = np.full_like(bin_centers, np.nan)
        return bin_centers, rtd_hist
    df = df[df['RTwrtStim'] <= 1]
    if len(df) == 0:
        print(f"No trials with RTwrtStim <= 1 for batch {batch_name}, animal {animal_id}, ABL {ABL}, ILD {ILD}. Returning NaNs.")
        rtd_hist = np.full_like(bin_centers, np.nan)
        return bin_centers, rtd_hist
    rtd_hist, _ = np.histogram(df['RTwrtStim'], bins=bins, density=True)
    return bin_centers, rtd_hist

def get_params_from_animal_pkl_file(batch_name, animal_id):
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
    vbmc_norm_tied_param_keys_map = {
        **vbmc_vanilla_tied_param_keys_map,
        'rate_norm_l_samples': 'rate_norm_l'
    }
    abort_keyname = "vbmc_aborts_results"
    if MODEL_TYPE == 'vanilla':
        tied_keyname = "vbmc_vanilla_tied_results"
        tied_param_keys_map = vbmc_vanilla_tied_param_keys_map
        is_norm = False
    elif MODEL_TYPE == 'norm':
        tied_keyname = "vbmc_norm_tied_results"
        tied_param_keys_map = vbmc_norm_tied_param_keys_map
        is_norm = True
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")
    abort_params = {}
    tied_params = {}
    rate_norm_l = 0
    if abort_keyname in fit_results_data:
        abort_samples = fit_results_data[abort_keyname]
        for param_samples_name, param_label in vbmc_aborts_param_keys_map.items():
            abort_params[param_label] = np.mean(abort_samples[param_samples_name])
    if tied_keyname in fit_results_data:
        tied_samples = fit_results_data[tied_keyname]
        for param_samples_name, param_label in tied_param_keys_map.items():
            tied_params[param_label] = np.mean(tied_samples[param_samples_name])
        if is_norm:
            rate_norm_l = tied_params.get('rate_norm_l', np.nan)
        else:
            rate_norm_l = 0
    else:
        print(f"Warning: {tied_keyname} not found in pickle for {batch_name}, {animal_id}")
    return abort_params, tied_params, rate_norm_l, is_norm

def get_P_A_C_A(batch, animal_id, abort_params):
    N_theory = int(1e3)
    file_name = f'batch_csvs/batch_{batch}_valid_and_aborts.csv'
    df = pd.read_csv(file_name)
    df_animal = df[df['animal'] == animal_id]
    t_pts = np.arange(-2, 2, 0.001)
    P_A_mean, C_A_mean, t_stim_samples = calculate_theoretical_curves(
        df_animal, N_theory, t_pts, abort_params['t_A_aff'], abort_params['V_A'], abort_params['theta_A'], rho_A_t_fn
    )
    return P_A_mean, C_A_mean, t_stim_samples

def get_theoretical_RTD_from_params(P_A_mean, C_A_mean, t_stim_samples, abort_params, tied_params, rate_norm_l, is_norm, ABL, ILD, batch_name, animal_id):
    phi_params_obj = np.nan
    K_max = 10
    if batch_name == 'LED34_even':
        T_trunc = 0.15
    else:
        T_trunc = 0.3
    t_pts = np.arange(-2, 2, 0.001)
    trunc_fac_samples = np.zeros((len(t_stim_samples)))
    Z_E = (tied_params['w'] - 0.5) * 2 * tied_params['theta_E']
    for idx, t_stim in enumerate(t_stim_samples):
        trunc_fac_samples[idx] = cum_pro_and_reactive_time_vary_fn(
            t_stim + 1, T_trunc,
            abort_params['V_A'], abort_params['theta_A'], abort_params['t_A_aff'],
            t_stim, ABL, ILD, tied_params['rate_lambda'], tied_params['T_0'], tied_params['theta_E'], Z_E, tied_params['t_E_aff'],
            phi_params_obj, rate_norm_l, 
            is_norm, False, K_max) \
            - cum_pro_and_reactive_time_vary_fn(
            t_stim, T_trunc,
            abort_params['V_A'], abort_params['theta_A'], abort_params['t_A_aff'],
            t_stim, ABL, ILD, tied_params['rate_lambda'], tied_params['T_0'], tied_params['theta_E'], Z_E, tied_params['t_E_aff'],
            phi_params_obj, rate_norm_l, 
            is_norm, False, K_max) + 1e-10
    trunc_factor = np.mean(trunc_fac_samples)
    # if batch_name == 'LED34_even':
    #     print(f"Trunc factor for {batch_name}, {animal_id} = {trunc_factor}")
    up_mean = np.array([up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn(
        t, 1,
        P_A_mean[i], C_A_mean[i],
        ABL, ILD, tied_params['rate_lambda'], tied_params['T_0'], tied_params['theta_E'], Z_E, tied_params['t_E_aff'], tied_params['del_go'],
        phi_params_obj, rate_norm_l, 
        is_norm, False, K_max) for i, t in enumerate(t_pts)])
    down_mean = np.array([up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn(
        t, -1,
        P_A_mean[i], C_A_mean[i],
        ABL, ILD, tied_params['rate_lambda'], tied_params['T_0'], tied_params['theta_E'], Z_E, tied_params['t_E_aff'], tied_params['del_go'],
        phi_params_obj, rate_norm_l, 
        is_norm, False, K_max) for i, t in enumerate(t_pts)])
    mask_0_1 = (t_pts >= 0) & (t_pts <= 1)
    t_pts_0_1 = t_pts[mask_0_1]
    up_mean_0_1 = up_mean[mask_0_1]
    down_mean_0_1 = down_mean[mask_0_1]
    if batch_name == 'LED34_even' and animal_id == 52:
        print('=================')
        print('ABL = ', ABL)
        print('ILD = ', ILD)
        print(f'batch_name = {batch_name}, animal_id = {animal_id}')
        print(f'area up = {np.trapz(up_mean_0_1, t_pts_0_1)}')
        print(f'area down = {np.trapz(down_mean_0_1, t_pts_0_1)}')
        print(f'trunc factor = {trunc_factor}')
        print('=================')
    up_theory_mean_norm = up_mean_0_1 / trunc_factor
    down_theory_mean_norm = down_mean_0_1 / trunc_factor
    up_plus_down_mean = up_theory_mean_norm + down_theory_mean_norm

    return t_pts_0_1, up_plus_down_mean

# %%
# Main analysis loop
ABL_arr = [20, 40, 60]
ILD_arr = [-16., -8., -4., -2., -1., 1., 2., 4., 8., 16.]
rt_bins = np.arange(0, 1.02, 0.02)  # 0 to 1 second in 0.02s bins

def process_batch_animal(batch_animal_pair):
    batch_name, animal_id = batch_animal_pair
    # print(f"Processing batch {batch_name}, animal {animal_id}")
    animal_rtd_data = {}
    try:
        abort_params, tied_params, rate_norm_l, is_norm = get_params_from_animal_pkl_file(batch_name, int(animal_id))
        p_a, c_a, ts_samp = get_P_A_C_A(batch_name, int(animal_id), abort_params)
        for abl in ABL_arr:
            # print(f"Animal = {batch_name},{animal_id}, Processing ABL {abl}")
            for ild in ILD_arr:
                stim_key = (abl, ild)
                try:
                    bin_centers, rtd_hist = get_animal_RTD_data(batch_name, int(animal_id), abl, ild, rt_bins)
                    try:
                        t_pts_0_1, up_plus_down = get_theoretical_RTD_from_params(
                            p_a, c_a, ts_samp, abort_params, tied_params, rate_norm_l, is_norm, abl, ild, batch_name, animal_id
                        )
                    except Exception as e:
                        print(f"  Error calculating theoretical RTD for ABL={abl}, ILD={ild}: {str(e)}")
                        t_pts_0_1 = np.linspace(0, 1, 100)
                        up_plus_down = np.full_like(t_pts_0_1, np.nan)
                    # --- Simulation KDE ---
                    # Prepare t_stim_samples for simulation: sample from this animal's intended_fix
                    try:
                        file_name = f"batch_csvs/batch_{batch_name}_valid_and_aborts.csv"
                        df = pd.read_csv(file_name)
                        df_animal = df[df['animal'] == int(animal_id)]
                        N_sim = int(1e5)  # Use a reasonable number for speed; adjust as needed
                        N_print = max(1, N_sim // 5)
                        dt = 1e-3
                        t_stim_samples = df_animal['intended_fix'].sample(N_sim, replace=True).values
                        # NOTE: TEMPORILY, NO SIMULATION. TAKES A LOT OF TIME
                        # Z_E = (tied_params['w'] - 0.5) * 2 * tied_params['theta_E']
                        # x_vals, kde_vals = get_simulation_RTD_KDE(
                        #     abort_params, tied_params, rate_norm_l, Z_E, abl, ild, t_stim_samples, N_sim, N_print, dt
                        # )
                        # sim_dict = {'x_vals': x_vals, 'kde_vals': kde_vals}
                        sim_dict = {'x_vals': np.array([]), 'kde_vals': np.array([])}
                    except Exception as e:
                        print(f"  Error running simulation for ABL={abl}, ILD={ild}: {str(e)}")
                        sim_dict = {'x_vals': np.array([]), 'kde_vals': np.array([])}

                    animal_rtd_data[stim_key] = {
                        'empirical': {
                            'bin_centers': bin_centers,
                            'rtd_hist': rtd_hist
                        },
                        'theoretical': {
                            't_pts': t_pts_0_1,
                            'rtd': up_plus_down
                        },
                        'simulation': sim_dict
                    }
                    # print(f"  Processed stimulus ABL={abl}, ILD={ild}")
                except Exception as e:
                    print(f"  Error processing stimulus ABL={abl}, ILD={ild}: {str(e)}")
                    continue
    except Exception as e:
        print(f"Error processing batch {batch_name}, animal {animal_id}: {str(e)}")
    return batch_animal_pair, animal_rtd_data

n_jobs = max(1, os.cpu_count() - 1)
print(f"Running with {n_jobs} parallel jobs")

results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(process_batch_animal)(batch_animal_pair) for batch_animal_pair in batch_animal_pairs
)

rtd_data = {}
for batch_animal_pair, animal_rtd_data in results:
    if animal_rtd_data:
        rtd_data[batch_animal_pair] = animal_rtd_data
print(f"Completed processing {len(rtd_data)} batch-animal pairs")

# %%
# PLOT data average and theory average for each stimulus
fig, axes = plt.subplots(3, 10, figsize=(20, 8), sharex=True, sharey=True)
for ax_row in axes:
    for ax in ax_row:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
for i, abl in enumerate(ABL_arr):
    for j, ild in enumerate(ILD_arr):
        stim_key = (abl, ild)
        empirical_rtds = []
        theoretical_rtds = []
        theoretical_rtd_areas = []
        bin_centers = None
        t_pts = None
        sim_kdes = []
        sim_x_vals = None
        for batch_animal_pair, animal_data in rtd_data.items():
            if stim_key in animal_data:
                emp_data = animal_data[stim_key]['empirical']
                if not np.all(np.isnan(emp_data['rtd_hist'])):
                    empirical_rtds.append(emp_data['rtd_hist'])
                    bin_centers = emp_data['bin_centers']
                theo_data = animal_data[stim_key]['theoretical']
                if not np.all(np.isnan(theo_data['rtd'])):
                    theoretical_rtds.append(theo_data['rtd'])
                    t_pts = theo_data['t_pts']
                    area = np.trapz(theo_data['rtd'], t_pts)
                    if area > 1.1:
                        print('++++++++++++++++++++++++++=')
                        print(f'ABL={abl}, ILD={ild}: Theoretical Area = {area:.4f}, animal {batch_animal_pair}')
                    theoretical_rtd_areas.append(area)
                sim_data = animal_data[stim_key].get('simulation', None)
                if sim_data is not None and sim_data['x_vals'].size > 0:
                    sim_kdes.append(sim_data['kde_vals'])
                    sim_x_vals = sim_data['x_vals']  # All should be the same
        # ax = axes[i, j]
        # if empirical_rtds and bin_centers is not None:
        #     avg_empirical_rtd = np.nanmean(empirical_rtds, axis=0)
        #     ax.plot(bin_centers, avg_empirical_rtd, 'b-', linewidth=1.5, label='Data')
        # if theoretical_rtds and t_pts is not None:
        #     avg_theoretical_rtd = np.nanmean(theoretical_rtds, axis=0)
        #     ax.plot(t_pts, avg_theoretical_rtd, 'r-', linewidth=1.5, label='Theory')
        # if sim_kdes and sim_x_vals is not None:
        #     avg_sim_kde = np.nanmean(sim_kdes, axis=0)
        #     ax.plot(sim_x_vals, avg_sim_kde, color='green', alpha=0.3, lw=3, label='Simulation')
        ax = axes[i, j]
        if empirical_rtds and bin_centers is not None:
            avg_empirical_rtd = np.nanmean(empirical_rtds, axis=0)
            ax.plot(bin_centers, avg_empirical_rtd, 'b-', linewidth=1.5, label='Data')
            # print area 
            bin_width = np.diff(bin_centers)[0]
            empirical_area = np.sum(avg_empirical_rtd * bin_width)
            # print(f"ABL={abl}, ILD={ild}: Empirical Area = {empirical_area:.4f}")
        if theoretical_rtds and t_pts is not None:
            avg_theoretical_rtd = np.nanmean(theoretical_rtds, axis=0)
            ax.plot(t_pts, avg_theoretical_rtd, 'r-', linewidth=1.5, label='Theory')
            # print area 
            t_width = np.diff(t_pts)[0]
            theoretical_area = np.sum(avg_theoretical_rtd * t_width)
            print(f"ABL={abl}, ILD={ild}: Theoretical Area = {theoretical_area:.4f}")
        ax.set_title(f'ABL={abl}, ILD={ild}', fontsize=10)
        if i == 2:
            ax.set_xlabel('RT (s)')
        if j == 0:
            ax.set_ylabel('Density')
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=2)
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.savefig(f'rtd_average_by_stimulus_{MODEL_TYPE}.png', dpi=300, bbox_inches='tight')
plt.show()
# %%
plt.hist(theoretical_rtd_areas);
# %%
abs_ILD_arr = [abs(ild) for ild in ILD_arr]
abs_ILD_arr = sorted(list(set(abs_ILD_arr)))
max_xlim_RT = 1
fig, axes = plt.subplots(len(ABL_arr), len(abs_ILD_arr), figsize=(10,6), sharex=True, sharey=True)
for ax_row in axes:
    for ax in ax_row:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
for i, abl in enumerate(ABL_arr):
    for j, abs_ild in enumerate(abs_ILD_arr):
        # For each (ABL, abs(ILD)), average over (ABL, ILD) and (ABL, -ILD)
        empirical_rtds = []
        theoretical_rtds = []
        bin_centers = None
        t_pts = None
        sim_kdes = []
        sim_x_vals = None
        for ild in [abs_ild, -abs_ild]:
            stim_key = (abl, ild)
            for batch_animal_pair, animal_data in rtd_data.items():
                if stim_key in animal_data:
                    emp_data = animal_data[stim_key]['empirical']
                    if not np.all(np.isnan(emp_data['rtd_hist'])):
                        empirical_rtds.append(emp_data['rtd_hist'])
                        bin_centers = emp_data['bin_centers']
                    theo_data = animal_data[stim_key]['theoretical']
                    if not np.all(np.isnan(theo_data['rtd'])):
                        theoretical_rtds.append(theo_data['rtd'])
                        t_pts = theo_data['t_pts']
                    sim_data = animal_data[stim_key].get('simulation', None)
                    if sim_data is not None and sim_data['x_vals'].size > 0:
                        sim_kdes.append(sim_data['kde_vals'])
                        sim_x_vals = sim_data['x_vals']  # All should be the same
        ax = axes[i, j]
        if empirical_rtds and bin_centers is not None:
            avg_empirical_rtd = np.nanmean(empirical_rtds, axis=0)
            ax.plot(bin_centers, avg_empirical_rtd, 'b-', linewidth=0.5, label='Data')
            bin_width = np.diff(bin_centers)[0]
            empirical_area = np.sum(avg_empirical_rtd * bin_width)
            print(f"ABL={abl}, |ILD|={abs_ild}: Empirical Area = {empirical_area:.4f}")
        if theoretical_rtds and t_pts is not None:
            avg_theoretical_rtd = np.nanmean(theoretical_rtds, axis=0)
            ax.plot(t_pts, avg_theoretical_rtd, 'r-', linewidth=0.5, label='Theory')
            t_width = np.diff(t_pts)[0]
            theoretical_area = np.sum(avg_theoretical_rtd * t_width)
            print(f"ABL={abl}, |ILD|={abs_ild}: Theoretical Area = {theoretical_area:.4f}")
        if i == len(ABL_arr) - 1:
            ax.set_xlabel('RT (s)', fontsize=12)
            ax.set_xticks([0, max_xlim_RT])
            ax.set_xticklabels(['0', max_xlim_RT], fontsize=12)
        ax.set_xlim(0, max_xlim_RT)
        ax.set_yticks([0, 12])
        ax.set_ylim(0, 14)
        ax.axvline(x=0, color='k', linestyle='--', linewidth=1)
        if j == 0:
            ax.set_ylabel(f'ABL={abl}', fontsize=12, rotation=0, ha='right', va='center')
        if i == 0:
            ax.set_title(f'|ILD|={abs_ild}', fontsize=12)
for ax_row in axes:
    for ax in ax_row:
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.savefig(f'rtd_average_by_abs_ILD_FOLDED_{MODEL_TYPE}.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
def get_quantiles_from_rtd(rtd, time_points, quantiles):
    """Calculate quantiles from a given RTD."""
    # Normalize RTD to get a PDF, then compute CDF
    pdf = rtd / np.sum(rtd)
    cdf = np.cumsum(pdf)
    # Use interpolation to find the time points corresponding to the desired quantiles
    # We use np.interp to find the x-values (time_points) for given y-values (quantiles) of the CDF.
    # We need to provide interp with (x_new, x_old, y_old)
    return np.interp(quantiles, cdf, time_points)


# %% Quantile Plot from Averaged RTDs
print("\nGenerating quantile plot from averaged RTDs...")

QUANTILES_TO_PLOT = [0.1, 0.3, 0.5, 0.7, 0.9]

# 1. Calculate quantiles from the averaged RTDs
quantile_results = {abl: {abs_ild: {'empirical': [], 'theoretical': []} for abs_ild in abs_ILD_arr} for abl in ABL_arr}

for i, abl in enumerate(ABL_arr):
    for j, abs_ild in enumerate(abs_ILD_arr):
        # This logic is similar to the RTD plot section to get averaged RTDs
        empirical_rtds = []
        theoretical_rtds = []
        bin_centers = None
        t_pts = None
        
        for ild in [abs_ild, -abs_ild]:
            stim_key = (abl, ild)
            for batch_animal_pair, animal_data in rtd_data.items():
                if stim_key in animal_data:
                    emp_data = animal_data[stim_key]['empirical']
                    if not np.all(np.isnan(emp_data['rtd_hist'])):
                        empirical_rtds.append(emp_data['rtd_hist'])
                        bin_centers = emp_data['bin_centers']
                    
                    theo_data = animal_data[stim_key]['theoretical']
                    if not np.all(np.isnan(theo_data['rtd'])):
                        theoretical_rtds.append(theo_data['rtd'])
                        t_pts = theo_data['t_pts']

        if empirical_rtds and bin_centers is not None:
            avg_empirical_rtd = np.nanmean(empirical_rtds, axis=0)
            emp_quantiles = get_quantiles_from_rtd(avg_empirical_rtd, bin_centers, QUANTILES_TO_PLOT)
            quantile_results[abl][abs_ild]['empirical'] = emp_quantiles

        if theoretical_rtds and t_pts is not None:
            avg_theoretical_rtd = np.nanmean(theoretical_rtds, axis=0)
            theo_quantiles = get_quantiles_from_rtd(avg_theoretical_rtd, t_pts, QUANTILES_TO_PLOT)
            quantile_results[abl][abs_ild]['theoretical'] = theo_quantiles

# 2. Plot the quantiles
fig, axes = plt.subplots(1, len(ABL_arr), figsize=(12, 4), sharey=True)
if len(ABL_arr) == 1: axes = [axes]  # Ensure axes is always iterable

abs_ild_sorted = sorted(abs_ILD_arr)

for i, abl in enumerate(ABL_arr):
    ax = axes[i]
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for q_idx, q in enumerate(QUANTILES_TO_PLOT):
        emp_quantiles_for_abl = [quantile_results[abl][abs_ild]['empirical'][q_idx] for abs_ild in abs_ild_sorted]
        theo_quantiles_for_abl = [quantile_results[abl][abs_ild]['theoretical'][q_idx] for abs_ild in abs_ild_sorted]
        
        valid_abs_ild_emp = [abs_ild for abs_ild in abs_ild_sorted]
        valid_abs_ild_theo = [abs_ild for abs_ild in abs_ild_sorted]

        # Plot empirical data
        if emp_quantiles_for_abl:
            ax.plot(valid_abs_ild_emp, emp_quantiles_for_abl, 'o-', color='b', markersize=4, label='Data' if q_idx == 0 else "")
        
        # Plot theoretical data
        if theo_quantiles_for_abl:
            ax.plot(valid_abs_ild_theo, theo_quantiles_for_abl, '^-', color='r', markersize=4, label='Theory' if q_idx == 0 else "")

    ax.set_title(f'ABL = {abl}')
    ax.set_xlabel('|ILD| (dB)')
    ax.set_xscale('log', base=2)
    ax.set_xticks(abs_ild_sorted)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

axes[0].set_ylabel('RT Quantile (s)')
if len(ABL_arr) > 1:
    fig.legend(loc='upper right', bbox_to_anchor=(0.98, 0.95))

plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.savefig(f'quantile_plot_from_avg_rtd_{MODEL_TYPE}.png', dpi=300)
plt.show()

# %% RTD Plots with Quantiles Overlaid
print("\nGenerating RTD plots with quantiles overlaid...")

fig, axes = plt.subplots(len(ABL_arr), len(abs_ILD_arr), figsize=(20,6), sharex=True, sharey=True)
for ax_row in axes:
    for ax in ax_row:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

for i, abl in enumerate(ABL_arr):
    for j, abs_ild in enumerate(abs_ILD_arr):
        ax = axes[i, j]
        
        # This logic is similar to the original RTD plot section to get averaged RTDs
        empirical_rtds = []
        theoretical_rtds = []
        bin_centers = None
        t_pts = None
        
        for ild in [abs_ild, -abs_ild]:
            stim_key = (abl, ild)
            for batch_animal_pair, animal_data in rtd_data.items():
                if stim_key in animal_data:
                    emp_data = animal_data[stim_key]['empirical']
                    if not np.all(np.isnan(emp_data['rtd_hist'])):
                        empirical_rtds.append(emp_data['rtd_hist'])
                        bin_centers = emp_data['bin_centers']
                    
                    theo_data = animal_data[stim_key]['theoretical']
                    if not np.all(np.isnan(theo_data['rtd'])):
                        theoretical_rtds.append(theo_data['rtd'])
                        t_pts = theo_data['t_pts']

        # Plot RTDs
        if empirical_rtds is not None and bin_centers is not None:
            avg_empirical_rtd = np.nanmean(empirical_rtds, axis=0)
            ax.plot(bin_centers, avg_empirical_rtd, 'b-', linewidth=1.5, label='Data')
        
        if theoretical_rtds is not None and t_pts is not None:
            avg_theoretical_rtd = np.nanmean(theoretical_rtds, axis=0)
            ax.plot(t_pts, avg_theoretical_rtd, 'r-', linewidth=1.5, label='Theory')

        # Overlay Quantiles
        emp_quantiles = quantile_results[abl][abs_ild]['empirical']
        if emp_quantiles is not None:
            for q_val in emp_quantiles:
                ax.axvline(x=q_val, color='b', linestyle='--', linewidth=1)

        theo_quantiles = quantile_results[abl][abs_ild]['theoretical']
        if theo_quantiles is not None:
            for q_val in theo_quantiles:
                ax.axvline(x=q_val, color='r', linestyle=':', linewidth=1)

        # Formatting
        if i == len(ABL_arr) - 1:
            ax.set_xlabel('RT (s)', fontsize=12)
            ax.set_xticks(np.arange(0, max_xlim_RT+0.1, 0.1))
            ax.set_xticklabels([f'{x:.1f}' for x in np.arange(0, max_xlim_RT+0.1, 0.1)], fontsize=12)
        ax.set_xlim(0, max_xlim_RT)
        ax.set_yticks([0, 12])
        ax.set_ylim(0, 14)
        ax.axvline(x=0, color='k', linestyle='--', linewidth=1)
        if j == 0:
            ax.set_ylabel(f'ABL={abl}', fontsize=12, rotation=0, ha='right', va='center')
        if i == 0:
            ax.set_title(f'|ILD|={abs_ild}', fontsize=12)

for ax_row in axes:
    for ax in ax_row:
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.savefig(f'rtd_average_with_quantiles_{MODEL_TYPE}.png', dpi=300, bbox_inches='tight')
plt.show()

# %% CDF of RTDs
print("\nGenerating CDF plots...")

fig, axes = plt.subplots(len(ABL_arr), len(abs_ILD_arr), figsize=(20, 6), sharex=True, sharey=True)
for ax_row in axes:
    for ax in ax_row:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

for i, abl in enumerate(ABL_arr):
    for j, abs_ild in enumerate(abs_ILD_arr):
        ax = axes[i, j]
        
        # This logic is similar to the original RTD plot section to get averaged RTDs
        empirical_rtds = []
        theoretical_rtds = []
        bin_centers = None
        t_pts = None
        
        for ild in [abs_ild, -abs_ild]:
            stim_key = (abl, ild)
            for batch_animal_pair, animal_data in rtd_data.items():
                if stim_key in animal_data:
                    emp_data = animal_data[stim_key]['empirical']
                    if not np.all(np.isnan(emp_data['rtd_hist'])):
                        empirical_rtds.append(emp_data['rtd_hist'])
                        bin_centers = emp_data['bin_centers']
                    
                    theo_data = animal_data[stim_key]['theoretical']
                    if not np.all(np.isnan(theo_data['rtd'])):
                        theoretical_rtds.append(theo_data['rtd'])
                        t_pts = theo_data['t_pts']

        # Calculate and Plot CDFs
        if empirical_rtds and bin_centers is not None:
            avg_empirical_rtd = np.nanmean(empirical_rtds, axis=0)
            bin_width = np.diff(bin_centers)[0]
            empirical_cdf = np.cumsum(avg_empirical_rtd * bin_width)
            ax.plot(bin_centers, empirical_cdf, 'b-', linewidth=1.5, label='Data CDF')
        
        if theoretical_rtds and t_pts is not None:
            avg_theoretical_rtd = np.nanmean(theoretical_rtds, axis=0)
            t_width = np.diff(t_pts)[0]
            theoretical_cdf = np.cumsum(avg_theoretical_rtd * t_width)
            ax.plot(t_pts, theoretical_cdf, 'r-', linewidth=1.5, label='Theory CDF')

        # Add horizontal line for 0.9 quantile
        ax.axhline(y=0.9, color='k', linestyle='--', linewidth=1, label='0.9 Quantile')

        # Formatting
        if i == len(ABL_arr) - 1:
            ax.set_xlabel('RT (s)', fontsize=12)
            ax.set_xticks(np.arange(0, max_xlim_RT+0.1, 0.1))
            ax.set_xticklabels([f'{x:.1f}' for x in np.arange(0, max_xlim_RT+0.1, 0.1)], fontsize=12)
        ax.set_xlim(0, max_xlim_RT)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.5, 1.0])
        ax.axvline(x=0, color='k', linestyle=':', linewidth=1)
        if j == 0:
            ax.set_ylabel(f'ABL={abl}', fontsize=12, rotation=0, ha='right', va='center')
        if i == 0:
            ax.set_title(f'|ILD|={abs_ild}', fontsize=12)

# Common legend for the figure
lines, labels = fig.axes[-1].get_legend_handles_labels()
# fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3, fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to make space for the legend
plt.subplots_adjust(bottom=0.15)
plt.savefig(f'cdf_average_{MODEL_TYPE}.png', dpi=300, bbox_inches='tight')
plt.show()


# %%