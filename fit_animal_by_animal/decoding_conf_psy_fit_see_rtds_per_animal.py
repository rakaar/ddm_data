# %%
"""
Per-animal RTD plots (no averaging across animals).

- Uses the same machinery as decoding_conf_psy_fit_see_rtds_supp_for_paper.py
- Supports MODEL_TYPE in {'vanilla','norm'} and PARAM_SOURCE in {'results','psycho'}
- Plots one 3x10 grid (ABL x ILD) per animal with Data (empirical) and Theory curves
- Choose how many animals to process via MAX_ANIMALS

Abort parameters and fixed timing (t_E_aff, del_go) are always read from results_{batch}_{animal}.pkl.
Tied parameters come from either results PKL (means) or psycho-fit VBMC (means), depending on PARAM_SOURCE.
"""
# %%
MODEL_TYPE = 'vanilla'  # 'vanilla' or 'norm'
PARAM_SOURCE = 'psycho'  # 'results' or 'psycho'
MAX_ANIMALS = 8  # set to None for all
# Configuration knobs
K_MAX = 10          # series truncation for theory computation
T_TRUNC = 0.3       # truncation window for normalization factor
N_VP_MEAN_SAMPLES = 50000  # psycho-fit VP draws to estimate means (avoid 1e6)
print(f"Per-animal RTD | MODEL_TYPE={MODEL_TYPE}, PARAM_SOURCE={PARAM_SOURCE}, MAX_ANIMALS={MAX_ANIMALS}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from collections import defaultdict
from time_vary_and_norm_simulators import psiam_tied_data_gen_wrapper_rate_norm_fn  # imported for parity with main script
from animal_wise_plotting_utils import calculate_theoretical_curves
from time_vary_norm_utils import (
    up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn, 
    up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn_vec,
    cum_pro_and_reactive_time_vary_fn, 
    rho_A_t_fn, 
    cum_A_t_fn
)

# --- Configuration ---
DESIRED_BATCHES = ['SD', 'LED34', 'LED6', 'LED8', 'LED7', 'LED34_even']
psycho_fits_repo_path = '/home/rlab/raghavendra/ddm_data/fit_valid_trials/psycho_fits_4-params-del_E_go_fixed_as_avg/'
ABL_arr = [20, 40, 60]
ILD_arr = [-16., -8., -4., -2., -1., 1., 2., 4., 8., 16.]
rt_bins = np.arange(0, 1.02, 0.02)

# --- Data helpers ---

def get_animal_RTD_data(batch_name, animal_id, ABL, ILD, bins):
    file_name = f'batch_csvs/batch_{batch_name}_valid_and_aborts.csv'
    df = pd.read_csv(file_name)
    df = df[(df['animal'] == animal_id) & (df['ABL'] == ABL) & (df['ILD'] == ILD) & (df['success'].isin([1, -1]))]
    bin_centers = (bins[:-1] + bins[1:]) / 2
    if df.empty:
        return bin_centers, np.full_like(bin_centers, np.nan)
    df = df[df['RTwrtStim'] <= 1]
    if len(df) == 0:
        return bin_centers, np.full_like(bin_centers, np.nan)
    rtd_hist, _ = np.histogram(df['RTwrtStim'], bins=bins, density=True)
    return bin_centers, rtd_hist

# --- Psycho VBMC helpers ---

def get_psycho_params(batch_name, animal_id):
    filename = os.path.join(psycho_fits_repo_path, f'psycho_fit_4-params-del_E_go_fixed_as_avg_{batch_name}_{animal_id}.pkl')
    with open(filename, 'rb') as f:
        vbmc_obj = pickle.load(f)
    vp = vbmc_obj.vp
    samples = vp.sample(int(N_VP_MEAN_SAMPLES))[0]
    tied_params = {
        'rate_lambda' : samples[:,0].mean(),
        'T_0' : samples[:,1].mean(),
        'theta_E' : samples[:,2].mean(),
        'w' : samples[:,3].mean(),
    }
    return tied_params


def get_psycho_vp_samples(batch_name, animal_id, n_samples):
    filename = os.path.join(psycho_fits_repo_path, f'psycho_fit_4-params-del_E_go_fixed_as_avg_{batch_name}_{animal_id}.pkl')
    with open(filename, 'rb') as f:
        vbmc_obj = pickle.load(f)
    vp = vbmc_obj.vp
    samples = vp.sample(n_samples)[0]
    return {
        'rate_lambda': samples[:, 0],
        'T_0': samples[:, 1],
        'theta_E': samples[:, 2],
        'w': samples[:, 3],
    }

# --- Parameter extraction ---

def get_fixed_t_E_aff_and_del_go(fit_results_data, batch_name, animal_id):
    """Compute fixed t_E_aff and del_go by averaging vanilla and norm tied fits.
    Requires both vbmc_vanilla_tied_results and vbmc_norm_tied_results.
    """
    vanilla_keyname = "vbmc_vanilla_tied_results"
    norm_keyname = "vbmc_norm_tied_results"
    if vanilla_keyname not in fit_results_data or norm_keyname not in fit_results_data:
        missing = []
        if vanilla_keyname not in fit_results_data:
            missing.append(vanilla_keyname)
        if norm_keyname not in fit_results_data:
            missing.append(norm_keyname)
        raise KeyError(f"Missing required keys in results_{batch_name}_animal_{animal_id}.pkl: {', '.join(missing)}")
    vanilla_tied = fit_results_data[vanilla_keyname]
    norm_tied = fit_results_data[norm_keyname]
    fixed_t_E_aff = 0.5 * (np.mean(vanilla_tied['t_E_aff_samples']) + np.mean(norm_tied['t_E_aff_samples']))
    fixed_del_go = 0.5 * (np.mean(vanilla_tied['del_go_samples']) + np.mean(norm_tied['del_go_samples']))
    return fixed_t_E_aff, fixed_del_go


def get_params_from_animal_pkl_file(batch_name, animal_id):
    pkl_file = f'results_{batch_name}_animal_{animal_id}.pkl'
    with open(pkl_file, 'rb') as f:
        fit_results_data = pickle.load(f)

    # Abort params (means)
    vbmc_aborts_param_keys_map = {
        'V_A_samples': 'V_A',
        'theta_A_samples': 'theta_A',
        't_A_aff_samp': 't_A_aff'
    }
    abort_keyname = "vbmc_aborts_results"
    abort_params = {}
    if abort_keyname in fit_results_data:
        abort_samples = fit_results_data[abort_keyname]
        for param_samples_name, param_label in vbmc_aborts_param_keys_map.items():
            abort_params[param_label] = float(np.mean(np.asarray(abort_samples[param_samples_name])))
    else:
        raise KeyError(f"Missing {abort_keyname} in {pkl_file}")

    # Fixed t_E_aff and del_go
    fixed_t_E_aff, fixed_del_go = get_fixed_t_E_aff_and_del_go(fit_results_data, batch_name, animal_id)

    # MODEL_TYPE settings
    if MODEL_TYPE == 'vanilla':
        source_key = 'vbmc_vanilla_tied_results'
        is_norm = False
        rate_norm_l = 0.0
    elif MODEL_TYPE == 'norm':
        source_key = 'vbmc_norm_tied_results'
        is_norm = True
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

    # Tied params depending on PARAM_SOURCE
    tied_params = {}
    if PARAM_SOURCE == 'results':
        if source_key not in fit_results_data:
            raise KeyError(f"Missing {source_key} in results_{batch_name}_animal_{animal_id}.pkl")
        model_samples = fit_results_data[source_key]
        tied_map = {
            'rate_lambda_samples': 'rate_lambda',
            'T_0_samples': 'T_0',
            'theta_E_samples': 'theta_E',
            'w_samples': 'w',
        }
        for s_key, label in tied_map.items():
            if s_key not in model_samples:
                raise KeyError(f"Key '{s_key}' not found in {source_key} for batch {batch_name}, animal {animal_id}")
            tied_params[label] = float(np.mean(np.asarray(model_samples[s_key])))
        if MODEL_TYPE == 'norm':
            if 'rate_norm_l_samples' not in model_samples:
                raise KeyError(f"Key 'rate_norm_l_samples' not found in {source_key} for batch {batch_name}, animal {animal_id}")
            rate_norm_l = float(np.mean(np.asarray(model_samples['rate_norm_l_samples'])))
    elif PARAM_SOURCE == 'psycho':
        tied_params.update(get_psycho_params(batch_name, animal_id))
        if MODEL_TYPE == 'norm':
            if source_key not in fit_results_data or 'rate_norm_l_samples' not in fit_results_data[source_key]:
                raise KeyError(f"Missing rate_norm_l_samples in {source_key} for batch {batch_name}, animal {animal_id}")
            model_samples = fit_results_data[source_key]
            rate_norm_l = float(np.mean(np.asarray(model_samples['rate_norm_l_samples'])))
    else:
        raise ValueError(f"Unknown PARAM_SOURCE: {PARAM_SOURCE}")

    tied_params['t_E_aff'] = fixed_t_E_aff
    tied_params['del_go'] = fixed_del_go

    return abort_params, tied_params, rate_norm_l, is_norm

# --- Theory helpers ---

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


def get_theoretical_RTD_from_params(P_A_mean, C_A_mean, t_stim_samples, abort_params, tied_params, rate_norm_l, is_norm, ABL, ILD, K_max=K_MAX):
    phi_params_obj = np.nan
    T_trunc = T_TRUNC
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
    up_mean = up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn_vec(
        t_pts, 1,
        P_A_mean, C_A_mean,
        ABL, ILD, tied_params['rate_lambda'], tied_params['T_0'], tied_params['theta_E'], Z_E, tied_params['t_E_aff'], tied_params['del_go'],
        np.nan, np.nan, np.nan, np.nan, np.nan,
        rate_norm_l, is_norm, False, K_max
    )
    down_mean = up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn_vec(
        t_pts, -1,
        P_A_mean, C_A_mean,
        ABL, ILD, tied_params['rate_lambda'], tied_params['T_0'], tied_params['theta_E'], Z_E, tied_params['t_E_aff'], tied_params['del_go'],
        np.nan, np.nan, np.nan, np.nan, np.nan,
        rate_norm_l, is_norm, False, K_max
    )
    mask_0_1 = (t_pts >= 0) & (t_pts <= 1)
    t_pts_0_1 = t_pts[mask_0_1]
    up_theory_mean_norm = up_mean[mask_0_1] / trunc_factor
    down_theory_mean_norm = down_mean[mask_0_1] / trunc_factor
    up_plus_down_mean = up_theory_mean_norm + down_theory_mean_norm
    return t_pts_0_1, up_plus_down_mean

# --- Build batch-animal list ---

batch_dir = '/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/batch_csvs'
batch_files = [f'batch_{batch_name}_valid_and_aborts.csv' for batch_name in DESIRED_BATCHES]
merged_data = pd.concat([
    pd.read_csv(os.path.join(batch_dir, fname)) for fname in batch_files if os.path.exists(os.path.join(batch_dir, fname))
], ignore_index=True)
merged_valid = merged_data[merged_data['success'].isin([1, -1])].copy()
batch_animal_pairs = sorted(list(map(tuple, merged_valid[['batch_name', 'animal']].drop_duplicates().values)))
print(f"Found {len(batch_animal_pairs)} batch-animal pairs.")

# Optionally cap number of animals
pairs_to_process = batch_animal_pairs[:MAX_ANIMALS] if (MAX_ANIMALS is not None) else batch_animal_pairs
print(f"Processing {len(pairs_to_process)} animals...")

# --- Per-animal plotting ---

for batch_name, animal_id in pairs_to_process:
    print(f"\n=== Animal: batch={batch_name}, animal={animal_id} ===")
    try:
        abort_params, tied_params, rate_norm_l, is_norm = get_params_from_animal_pkl_file(batch_name, animal_id)
        P_A_mean, C_A_mean, t_stim_samples = get_P_A_C_A(batch_name, animal_id, abort_params)
    except Exception as e:
        print(f"  Skipping (param extraction failed): {e}")
        continue

    fig, axes = plt.subplots(3, 10, figsize=(20, 8), sharex=True, sharey=True)
    for ax_row in axes:
        for ax in ax_row:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    for i, abl in enumerate(ABL_arr):
        for j, ild in enumerate(ILD_arr):
            ax = axes[i, j]
            try:
                bin_centers, rtd_hist = get_animal_RTD_data(batch_name, animal_id, abl, ild, rt_bins)
                try:
                    t_pts_0_1, up_plus_down = get_theoretical_RTD_from_params(
                        P_A_mean, C_A_mean, t_stim_samples, abort_params, tied_params, rate_norm_l, is_norm, abl, ild
                    )
                except Exception as e:
                    print(f"    Theory failed for ABL={abl}, ILD={ild}: {e}")
                    t_pts_0_1 = np.linspace(0, 1, 100)
                    up_plus_down = np.full_like(t_pts_0_1, np.nan)

                if not np.all(np.isnan(rtd_hist)):
                    ax.plot(bin_centers, rtd_hist, 'b-', linewidth=1.5, label='Data')
                if not np.all(np.isnan(up_plus_down)):
                    ax.plot(t_pts_0_1, up_plus_down, 'r-', linewidth=1.5, label='Theory')
                ax.set_title(f'ABL={abl}, ILD={ild}', fontsize=10)
                if i == 2:
                    ax.set_xlabel('RT (s)')
                    ax.set_xticks([0, 1])
                    ax.set_xticklabels(['0', '1'], fontsize=12)
                ax.set_xlim(0, 1)
                if j == 0:
                    ax.set_ylabel('Density')
            except Exception as e:
                print(f"    Error at ABL={abl}, ILD={ild}: {e}")
                continue

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=2)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    out_png = f'rtd_by_stimulus_PER_ANIMAL_{MODEL_TYPE}_{PARAM_SOURCE}_{batch_name}_{animal_id}.png'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  Saved {out_png}")
