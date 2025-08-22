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
MAX_ANIMALS = None # set to None for all
# Configuration knobs
K_MAX = 10          # series truncation for theory computation
T_TRUNC = 0.3       # truncation window for normalization factor
N_VP_MEAN_SAMPLES = 50000  # psycho-fit VP draws to estimate means (avoid 1e6)
print(f"Per-animal RTD | MODEL_TYPE={MODEL_TYPE}, PARAM_SOURCE={PARAM_SOURCE}, MAX_ANIMALS={MAX_ANIMALS}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import pickle
from collections import defaultdict
import concurrent.futures as cf
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
psycho_fits_repo_path = '/home/rlab/raghavendra/ddm_data/fit_valid_trials/psycho_fits_T_0_fixed_from_vanilla/'
ABL_arr = [20, 40, 60]
ILD_arr = [-16., -8., -4., -2., -1., 1., 2., 4., 8., 16.]
rt_bins = np.arange(0, 1.02, 0.02)

# Parallel settings
PARALLEL = True
N_WORKERS = max(1, (os.cpu_count() or 4) - 1)

# Common axes for consistent averaging
EMP_BIN_CENTERS = (rt_bins[:-1] + rt_bins[1:]) / 2
THEO_T_AXIS = np.arange(0, 1.0001, 0.001)  # 0..1 inclusive, matches internal theory grid (1001 pts)

# Aggregated plotting toggles
PLOT_OVERLAY_ALL_STIM = False  # overlay of all stimuli in one axes (off by default)
PLOT_AVG_GRID = True           # 3x10 grid style averaged across animals (on by default)

# Output controls
SAVE_MULTIPAGE_PDF = True
OUTPUT_PDF = f"rtd_by_stimulus_ALL_ANIMALS_T_0_also_fixed_{MODEL_TYPE}_{PARAM_SOURCE}.pdf"
SHOW_FIGS = False  # when saving to PDF, keep False to avoid interactive windows
fig_avg = None  # will hold the averaged 3x10 figure
per_animal_figs = []  # collect per-animal figures for multipage PDF

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

# Optimized empirical histogram computation (avoid repeated CSV reads per panel)
def get_empirical_hist_map(df_animal, ABL_list, ILD_list, bins):
    bin_centers = (bins[:-1] + bins[1:]) / 2
    out = {}
    if df_animal is None or df_animal.empty:
        for abl in ABL_list:
            for ild in ILD_list:
                out[(abl, ild)] = (bin_centers, np.full_like(bin_centers, np.nan))
        return out
    dfv = df_animal[df_animal['success'].isin([1, -1])]
    dfv = dfv[dfv['RTwrtStim'] <= 1]
    for abl in ABL_list:
        dfa = dfv[dfv['ABL'] == abl]
        for ild in ILD_list:
            dfi = dfa[dfa['ILD'] == ild]
            if dfi.empty:
                out[(abl, ild)] = (bin_centers, np.full_like(bin_centers, np.nan))
            else:
                rtd_hist, _ = np.histogram(dfi['RTwrtStim'], bins=bins, density=True)
                out[(abl, ild)] = (bin_centers, rtd_hist)
    return out

# --- Psycho VBMC helpers ---

def get_psycho_params(batch_name, animal_id):
    filename = os.path.join(psycho_fits_repo_path, f'psycho_fit_3-params-T_0_fixed_from_vanilla_{batch_name}_{animal_id}.pkl')
    with open(filename, 'rb') as f:
        vbmc_obj = pickle.load(f)
    vp = vbmc_obj.vp
    samples = vp.sample(int(N_VP_MEAN_SAMPLES))[0]
    tied_params = {
        'rate_lambda' : samples[:,0].mean(),
        'theta_E' : samples[:,1].mean(),
        'w' : samples[:,2].mean(),
    }
    return tied_params


def get_psycho_vp_samples(batch_name, animal_id, n_samples):
    filename = os.path.join(psycho_fits_repo_path, f'psycho_fit_3-params-T_0_fixed_from_vanilla_{batch_name}_{animal_id}.pkl')
    with open(filename, 'rb') as f:
        vbmc_obj = pickle.load(f)
    vp = vbmc_obj.vp
    samples = vp.sample(n_samples)[0]
    return {
        'rate_lambda': samples[:, 0],
        'theta_E': samples[:, 1],
        'w': samples[:, 2],
    }

# --- Parameter extraction ---

def get_fixed_t_E_aff_and_del_go(fit_results_data, batch_name, animal_id):
    """Read fixed T_0, t_E_aff, and del_go from vanilla tied results only."""
    vanilla_keyname = "vbmc_vanilla_tied_results"
    if vanilla_keyname not in fit_results_data:
        raise KeyError(f"Missing required key '{vanilla_keyname}' in results_{batch_name}_animal_{animal_id}.pkl")
    vanilla_tied = fit_results_data[vanilla_keyname]
    fixed_T_0 = float(np.mean(np.asarray(vanilla_tied['T_0_samples'])))
    fixed_t_E_aff = float(np.mean(np.asarray(vanilla_tied['t_E_aff_samples'])))
    fixed_del_go = float(np.mean(np.asarray(vanilla_tied['del_go_samples'])))
    return fixed_T_0, fixed_t_E_aff, fixed_del_go


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

    # Fixed T_0, t_E_aff, and del_go from vanilla tied results
    fixed_T_0, fixed_t_E_aff, fixed_del_go = get_fixed_t_E_aff_and_del_go(fit_results_data, batch_name, animal_id)

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

    tied_params['T_0'] = fixed_T_0
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

# Worker for parallel theory computation (one panel)
def _theory_worker(P_A_mean, C_A_mean, t_stim_samples, abort_params, tied_params, rate_norm_l, is_norm, abl, ild, K_max):
    try:
        t_pts_0_1, up_plus_down = get_theoretical_RTD_from_params(
            P_A_mean, C_A_mean, t_stim_samples, abort_params, tied_params, rate_norm_l, is_norm, abl, ild, K_max
        )
    except Exception as e:
        t_pts_0_1 = THEO_T_AXIS
        up_plus_down = np.full_like(t_pts_0_1, np.nan)
    return abl, ild, t_pts_0_1, up_plus_down

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

# Aggregators for animal-averaged plot
agg_emp = defaultdict(list)   # key: (ABL, ILD) -> list of empirical hist arrays (len EMP_BIN_CENTERS)
agg_theo = defaultdict(list)  # key: (ABL, ILD) -> list of theory arrays (len THEO_T_AXIS)

for batch_name, animal_id in pairs_to_process:
    print(f"\n=== Animal: batch={batch_name}, animal={animal_id} ===")
    try:
        abort_params, tied_params, rate_norm_l, is_norm = get_params_from_animal_pkl_file(batch_name, animal_id)
        P_A_mean, C_A_mean, t_stim_samples = get_P_A_C_A(batch_name, animal_id, abort_params)
    except Exception as e:
        print(f"  Skipping (param extraction failed): {e}")
        continue

    # Preload data once per animal for faster empirical histograms
    try:
        df_file = f'batch_csvs/batch_{batch_name}_valid_and_aborts.csv'
        df_all = pd.read_csv(df_file)
        df_animal = df_all[df_all['animal'] == animal_id]
    except Exception as e:
        print(f"  Failed reading CSV for empirical histograms: {e}")
        df_animal = pd.DataFrame()

    emp_map = get_empirical_hist_map(df_animal, ABL_arr, ILD_arr, rt_bins)

    # Compute theory in parallel across the 3x10 grid
    theory_map = {}
    cond_list = [(abl, ild) for abl in ABL_arr for ild in ILD_arr]
    if PARALLEL and len(cond_list) > 1:
        with cf.ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            futs = {
                ex.submit(
                    _theory_worker,
                    P_A_mean, C_A_mean, t_stim_samples,
                    abort_params, tied_params, rate_norm_l, is_norm,
                    abl, ild, K_MAX
                ): (abl, ild) for (abl, ild) in cond_list
            }
            for fut in cf.as_completed(futs):
                abl, ild = futs[fut]
                try:
                    abl_out, ild_out, t_pts_0_1, up_plus_down = fut.result()
                    theory_map[(abl_out, ild_out)] = (t_pts_0_1, up_plus_down)
                except Exception as e:
                    print(f"    Theory failed for ABL={abl}, ILD={ild}: {e}")
                    t_pts_0_1 = np.linspace(0, 1, 100)
                    theory_map[(abl, ild)] = (t_pts_0_1, np.full_like(t_pts_0_1, np.nan))
    else:
        # Sequential fallback
        for (abl, ild) in cond_list:
            try:
                t_pts_0_1, up_plus_down = get_theoretical_RTD_from_params(
                    P_A_mean, C_A_mean, t_stim_samples, abort_params, tied_params, rate_norm_l, is_norm, abl, ild
                )
            except Exception as e:
                print(f"    Theory failed for ABL={abl}, ILD={ild}: {e}")
                t_pts_0_1 = THEO_T_AXIS
                up_plus_down = np.full_like(t_pts_0_1, np.nan)
            theory_map[(abl, ild)] = (t_pts_0_1, up_plus_down)

    # Plot on the main thread
    fig, axes = plt.subplots(3, 10, figsize=(20, 8), sharex=True, sharey=True)
    for ax_row in axes:
        for ax in ax_row:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    for i, abl in enumerate(ABL_arr):
        for j, ild in enumerate(ILD_arr):
            ax = axes[i, j]
            try:
                bin_centers, rtd_hist = emp_map.get((abl, ild), (None, None))
                t_pts_0_1, up_plus_down = theory_map.get((abl, ild), (None, None))

                if bin_centers is not None and rtd_hist is not None and not np.all(np.isnan(rtd_hist)):
                    ax.plot(bin_centers, rtd_hist, 'b-', linewidth=1.5, label='Data')
                if t_pts_0_1 is not None and up_plus_down is not None and not np.all(np.isnan(up_plus_down)):
                    ax.plot(t_pts_0_1, up_plus_down, 'r-', linewidth=1.5, label='Theory')
                ax.set_title(f'ABL={abl}, ILD={ild}', fontsize=10)
                if i == 2:
                    ax.set_xlabel('RT (s)')
                    ax.set_xticks([0, 0.6])
                    ax.set_xticklabels(['0', '0.6'], fontsize=12)
                ax.set_xlim(0, 0.6)
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
    # out_png = f'rtd_by_stimulus_PER_ANIMAL_{MODEL_TYPE}_{PARAM_SOURCE}_{batch_name}_{animal_id}.png'
    # plt.savefig(out_png, dpi=300, bbox_inches='tight')
    if SAVE_MULTIPAGE_PDF:
        try:
            per_animal_figs
        except NameError:
            per_animal_figs = []
        per_animal_figs.append(fig)
    if SHOW_FIGS:
        plt.show()

    # Collect per-stimulus arrays for animal-averaged plot
    for (abl, ild) in cond_list:
        # Empirical
        bc, rtd_hist = emp_map.get((abl, ild), (None, None))
        if rtd_hist is None:
            emp_arr = np.full_like(EMP_BIN_CENTERS, np.nan)
        else:
            # ensure consistent length
            if bc is None or len(bc) != len(EMP_BIN_CENTERS) or not np.allclose(bc, EMP_BIN_CENTERS):
                # interpolate onto EMP_BIN_CENTERS if needed
                if bc is not None and len(bc) > 1 and np.any(np.isfinite(rtd_hist)):
                    valid = np.isfinite(rtd_hist)
                    if np.sum(valid) >= 2:
                        emp_arr = np.interp(EMP_BIN_CENTERS, bc[valid], rtd_hist[valid])
                    else:
                        emp_arr = np.full_like(EMP_BIN_CENTERS, np.nan)
                else:
                    emp_arr = np.full_like(EMP_BIN_CENTERS, np.nan)
            else:
                emp_arr = rtd_hist
        agg_emp[(abl, ild)].append(emp_arr)

        # Theory
        t_pts, theo = theory_map.get((abl, ild), (None, None))
        if theo is None:
            theo_arr = np.full_like(THEO_T_AXIS, np.nan)
        else:
            if t_pts is None or len(t_pts) != len(THEO_T_AXIS) or not np.allclose(t_pts, THEO_T_AXIS):
                # interpolate onto THEO_T_AXIS if possible
                if t_pts is not None and len(t_pts) > 1 and np.any(np.isfinite(theo)):
                    valid = np.isfinite(theo)
                    if np.sum(valid) >= 2:
                        theo_arr = np.interp(THEO_T_AXIS, t_pts[valid], theo[valid])
                    else:
                        theo_arr = np.full_like(THEO_T_AXIS, np.nan)
                else:
                    theo_arr = np.full_like(THEO_T_AXIS, np.nan)
            else:
                theo_arr = theo
        agg_theo[(abl, ild)].append(theo_arr)

# --- Aggregated across animals plots ---

# Optional overlay of all stimuli in one axes
if PLOT_OVERLAY_ALL_STIM:
    fig, ax = plt.subplots(figsize=(10, 5))
    added_data_lbl = False
    added_theory_lbl = False
    for abl in ABL_arr:
        for ild in ILD_arr:
            key = (abl, ild)
            emp_list = agg_emp.get(key, [])
            theo_list = agg_theo.get(key, [])
            if len(emp_list) == 0 and len(theo_list) == 0:
                continue
            emp_avg = np.nanmean(np.vstack(emp_list), axis=0) if len(emp_list) > 0 else np.full_like(EMP_BIN_CENTERS, np.nan)
            theo_avg = np.nanmean(np.vstack(theo_list), axis=0) if len(theo_list) > 0 else np.full_like(THEO_T_AXIS, np.nan)

            if not np.all(np.isnan(emp_avg)):
                ax.plot(EMP_BIN_CENTERS, emp_avg, '-', linewidth=1.5, alpha=0.9, label=('Data' if not added_data_lbl else None))
                added_data_lbl = True
            if not np.all(np.isnan(theo_avg)):
                ax.plot(THEO_T_AXIS, theo_avg, '--', linewidth=1.5, alpha=0.9, label=('Theory' if not added_theory_lbl else None))
                added_theory_lbl = True

    ax.set_title('Animal-averaged RTD across stimuli (Data solid, Theory dashed)')
    ax.set_xlabel('RT (s)')
    ax.set_ylabel('Density')
    ax.set_xlim(0, 0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    if SHOW_FIGS:
        plt.show()

# 3x10 grid averaged across animals (matching per-animal style)
if PLOT_AVG_GRID:
    fig_avg, axes_avg = plt.subplots(3, 10, figsize=(20, 8), sharex=True, sharey=True)
    for ax_row in axes_avg:
        for axx in ax_row:
            axx.spines['top'].set_visible(False)
            axx.spines['right'].set_visible(False)

    for i, abl in enumerate(ABL_arr):
        for j, ild in enumerate(ILD_arr):
            axx = axes_avg[i, j]
            key = (abl, ild)
            emp_list = agg_emp.get(key, [])
            theo_list = agg_theo.get(key, [])
            emp_avg = np.nanmean(np.vstack(emp_list), axis=0) if len(emp_list) > 0 else np.full_like(EMP_BIN_CENTERS, np.nan)
            theo_avg = np.nanmean(np.vstack(theo_list), axis=0) if len(theo_list) > 0 else np.full_like(THEO_T_AXIS, np.nan)

            if not np.all(np.isnan(emp_avg)):
                axx.plot(EMP_BIN_CENTERS, emp_avg, 'b-', linewidth=1.5, label='Data')
            if not np.all(np.isnan(theo_avg)):
                axx.plot(THEO_T_AXIS, theo_avg, 'r-', linewidth=1.5, label='Theory')
            axx.set_title(f'ABL={abl}, ILD={ild}', fontsize=10)
            if i == 2:
                axx.set_xlabel('RT (s)')
                axx.set_xticks([0, 0.6])
                axx.set_xticklabels(['0', '0.6'], fontsize=12)
            axx.set_xlim(0, 0.6)
            if j == 0:
                axx.set_ylabel('Density')

    handles_avg, labels_avg = axes_avg[0, 0].get_legend_handles_labels()
    if handles_avg:
        fig_avg.legend(handles_avg, labels_avg, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=2)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    # do not show unless requested; we'll save to PDF later
    if SHOW_FIGS:
        plt.show()

# --- Save multi-page PDF (first: averaged grid, then: each animal) ---
if SAVE_MULTIPAGE_PDF:
    try:
        with PdfPages(OUTPUT_PDF) as pdf:
            if fig_avg is not None:
                pdf.savefig(fig_avg, bbox_inches='tight')
            for fig in per_animal_figs:
                pdf.savefig(fig, bbox_inches='tight')
            info = pdf.infodict()
            info['Title'] = f'RTD by Stimulus — Averaged + Per Animal ({MODEL_TYPE}, {PARAM_SOURCE})'
            info['Creator'] = 'decoding_conf_psy_fit_see_rtds_per_animal.py'
        print(f"Saved multi-page PDF: {OUTPUT_PDF} with {1 if fig_avg is not None else 0} + {len(per_animal_figs)} pages")
    finally:
        if not SHOW_FIGS:
            if fig_avg is not None:
                plt.close(fig_avg)
            for fig in per_animal_figs:
                plt.close(fig)
