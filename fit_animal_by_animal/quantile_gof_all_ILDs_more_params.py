# %%
"""
Compute quantile goodness-of-fit (R², RMSE, SSE) for cond-fit with MORE PARAMS
(gamma, omega, t_E_aff, w, del_go all fitted per condition) across all ILD values.

This differs from quantile_gof_all_ILDs.py in that:
- The cond-fit pkl files contain 5 params (gamma, omega, t_E_aff, w, del_go)
- NOT the 2-param version where t_E_aff, w, del_go are fixed from parametric fit

Outputs for each ILD:
- PNG: quantiles_gof_ILD_{ild}_more_params.png
- PKL: quantiles_gof_ILD_{ild}_more_params.pkl (contains theory, empirical data, and R² metrics)
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Add fit_each_condn to path (needed for unpickling vbmc objects)
sys.path.insert(0, '/home/rlab/raghavendra/ddm_data/fit_each_condn')

from led_off_gamma_omega_pdf_utils import cum_A_t_fn, CDF_E_minus_small_t_NORM_omega_gamma_with_w_fn

# %%
# =============================================================================
# Configuration
# =============================================================================
ILD_VALUES = [16, 1, 2, 4, 8]
ABL_arr = [20, 40, 60]
DESIRED_BATCHES = ['LED34', 'LED6', 'LED8', 'LED7', 'LED34_even']  # Exclude SD (no ILD=16)
K_max = 10
N_theory = int(1e3)
N_WORKERS = min(25, multiprocessing.cpu_count())  # Parallel workers

batch_dir = '/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/batch_csvs'
abl_colors = {20: 'tab:blue', 40: 'tab:orange', 60: 'tab:green'}

# Folder where 5-param cond-fit pkl files are stored
COND_FIT_MORE_PARAMS_FOLDER = '/home/rlab/raghavendra/ddm_data/fit_each_condn/each_animal_cond_fit_5_params_pkl_files'

# %%
# =============================================================================
# Load empirical quantile data
# =============================================================================
with open('fig1_quantiles_plot_data.pkl', 'rb') as f:
    quantile_data = pickle.load(f)

abs_ILD_arr = quantile_data['abs_ILD_arr']  # [1, 2, 4, 8, 16]
plotting_quantiles = quantile_data['plotting_quantiles']
mean_unscaled = quantile_data['mean_unscaled']
sem_unscaled = quantile_data['sem_unscaled']

# %%
# =============================================================================
# Get batch-animal pairs
# =============================================================================
batch_files = [f'batch_{batch_name}_valid_and_aborts.csv' for batch_name in DESIRED_BATCHES]
merged_data = pd.concat([
    pd.read_csv(os.path.join(batch_dir, fname)) 
    for fname in batch_files if os.path.exists(os.path.join(batch_dir, fname))
], ignore_index=True)
merged_valid = merged_data[merged_data['success'].isin([1, -1])].copy()
batch_animal_pairs = sorted(list(map(tuple, merged_valid[['batch_name', 'animal']].drop_duplicates().values)))
print(f"Found {len(batch_animal_pairs)} batch-animal pairs")

# %%
# =============================================================================
# Helper functions
# =============================================================================

def get_cond_fit_more_params(batch_name, animal_id, ABL, ILD):
    """
    Load gamma, omega, t_E_aff, w, del_go from 5-param condition-by-condition fit pkl file.
    File pattern: vbmc_cond_by_cond_{batch}_{animal}_{ABL}_ILD_{ILD}_5_params.pkl
    """
    pkl_file = os.path.join(COND_FIT_MORE_PARAMS_FOLDER, 
                            f'vbmc_cond_by_cond_{batch_name}_{animal_id}_{ABL}_ILD_{ILD}_5_params.pkl')
    if not os.path.exists(pkl_file):
        return None
    with open(pkl_file, 'rb') as f:
        vp = pickle.load(f)
    vp = vp.vp
    vp_samples = vp.sample(int(1e4))[0]
    # 5 params: gamma, omega, t_E_aff, w, del_go
    gamma = float(np.mean(vp_samples[:, 0]))
    omega = float(np.mean(vp_samples[:, 1]))
    t_E_aff = float(np.mean(vp_samples[:, 2]))
    w = float(np.mean(vp_samples[:, 3]))
    del_go = float(np.mean(vp_samples[:, 4]))
    return {'gamma': gamma, 'omega': omega, 't_E_aff': t_E_aff, 'w': w, 'del_go': del_go}

def get_abort_params(batch_name, animal_id):
    """Load abort params (V_A, theta_A, t_A_aff) from animal pkl file."""
    pkl_file = f'/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/results_{batch_name}_animal_{animal_id}.pkl'
    if not os.path.exists(pkl_file):
        return None
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    if 'vbmc_aborts_results' not in data:
        return None
    
    abort_samples = data['vbmc_aborts_results']
    return {
        'V_A': np.mean(abort_samples['V_A_samples']),
        'theta_A': np.mean(abort_samples['theta_A_samples']),
        't_A_aff': np.mean(abort_samples['t_A_aff_samp'])
    }

def invert_cdf(cdf_vals, t_vals, quantile_levels):
    """Invert CDF to get RT values at given quantile levels."""
    quantile_rts = np.zeros(len(quantile_levels))
    for i, q in enumerate(quantile_levels):
        idx = np.searchsorted(cdf_vals, q)
        if idx == 0:
            quantile_rts[i] = t_vals[0]
        elif idx >= len(t_vals):
            quantile_rts[i] = t_vals[-1]
        else:
            t0, t1 = t_vals[idx-1], t_vals[idx]
            c0, c1 = cdf_vals[idx-1], cdf_vals[idx]
            if c1 != c0:
                quantile_rts[i] = t0 + (q - c0) * (t1 - t0) / (c1 - c0)
            else:
                quantile_rts[i] = t0
    return quantile_rts

def compute_gof_metrics(theory_quantiles, theory_q_levels, emp_quantiles, emp_q_levels, emp_sem):
    """Compute goodness of fit metrics between theoretical and empirical quantiles."""
    theory_interp = np.interp(emp_q_levels, theory_q_levels, theory_quantiles)
    residuals = theory_interp - emp_quantiles
    sse = np.sum(residuals**2)
    weights = 1.0 / (emp_sem**2 + 1e-10)
    weighted_sse = np.sum(weights * residuals**2)
    rmse = np.sqrt(np.mean(residuals**2))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((emp_quantiles - np.mean(emp_quantiles))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    return {'SSE': sse, 'weighted_SSE': weighted_sse, 'RMSE': rmse, 'R2': r_squared}

def process_single_animal(args):
    """Worker function to process a single animal for a given ILD. Returns CDFs per ABL."""
    batch_name, animal_id, ILD_target, t_pts_wrt_stim = args
    animal_id_str = str(animal_id)
    
    # Get abort params
    abort_params = get_abort_params(batch_name, animal_id_str)
    if abort_params is None:
        return None
    
    # Check if this animal has data for this ILD
    csv_file = os.path.join(batch_dir, f'batch_{batch_name}_valid_and_aborts.csv')
    df = pd.read_csv(csv_file)
    df_animal = df[df['animal'] == int(animal_id)]
    if ILD_target not in df_animal['ILD'].abs().unique():
        return None
    
    # Sample t_stim
    t_stim_samples = df_animal['intended_fix'].sample(N_theory, replace=True).values
    
    V_A = abort_params['V_A']
    theta_A = abort_params['theta_A']
    t_A_aff = abort_params['t_A_aff']
    
    result = {'cond_more': {}}
    
    for ABL in ABL_arr:
        # --- COND FIT MORE PARAMS MODEL (5 params per condition) ---
        cond_params_pos = get_cond_fit_more_params(batch_name, animal_id_str, ABL, ILD_target)
        cond_params_neg = get_cond_fit_more_params(batch_name, animal_id_str, ABL, -ILD_target)
        
        if cond_params_pos is not None and cond_params_neg is not None:
            gamma_pos, omega_pos = cond_params_pos['gamma'], cond_params_pos['omega']
            gamma_neg, omega_neg = cond_params_neg['gamma'], cond_params_neg['omega']
            w_pos, t_E_aff_pos = cond_params_pos['w'], cond_params_pos['t_E_aff']
            w_neg, t_E_aff_neg = cond_params_neg['w'], cond_params_neg['t_E_aff']
            
            cdf_samples_pos = np.zeros((N_theory, len(t_pts_wrt_stim)))
            cdf_samples_neg = np.zeros((N_theory, len(t_pts_wrt_stim)))
            
            for idx, t_stim in enumerate(t_stim_samples):
                t_pts_wrt_fix = t_pts_wrt_stim + t_stim
                for t_idx, t_fix in enumerate(t_pts_wrt_fix):
                    c_A = cum_A_t_fn(t_fix - t_A_aff, V_A, theta_A)
                    
                    # Positive ILD
                    t_evidence_pos = t_pts_wrt_stim[t_idx] - t_E_aff_pos
                    c_E_up_pos = CDF_E_minus_small_t_NORM_omega_gamma_with_w_fn(t_evidence_pos, gamma_pos, omega_pos, 1, w_pos, K_max)
                    c_E_down_pos = CDF_E_minus_small_t_NORM_omega_gamma_with_w_fn(t_evidence_pos, gamma_pos, omega_pos, -1, w_pos, K_max)
                    cdf_samples_pos[idx, t_idx] = c_A + (c_E_up_pos + c_E_down_pos) - c_A * (c_E_up_pos + c_E_down_pos)
                    
                    # Negative ILD
                    t_evidence_neg = t_pts_wrt_stim[t_idx] - t_E_aff_neg
                    c_E_up_neg = CDF_E_minus_small_t_NORM_omega_gamma_with_w_fn(t_evidence_neg, gamma_neg, omega_neg, 1, w_neg, K_max)
                    c_E_down_neg = CDF_E_minus_small_t_NORM_omega_gamma_with_w_fn(t_evidence_neg, gamma_neg, omega_neg, -1, w_neg, K_max)
                    cdf_samples_neg[idx, t_idx] = c_A + (c_E_up_neg + c_E_down_neg) - c_A * (c_E_up_neg + c_E_down_neg)
            
            # Normalize each CDF separately, then average
            cdf_mean_pos = np.mean(cdf_samples_pos, axis=0)
            cdf_mean_neg = np.mean(cdf_samples_neg, axis=0)
            mask_0_1 = (t_pts_wrt_stim >= 0) & (t_pts_wrt_stim <= 1)
            cdf_0_1_pos = cdf_mean_pos[mask_0_1]
            cdf_0_1_neg = cdf_mean_neg[mask_0_1]
            
            if cdf_0_1_pos[-1] > cdf_0_1_pos[0]:
                cdf_0_1_pos = (cdf_0_1_pos - cdf_0_1_pos[0]) / (cdf_0_1_pos[-1] - cdf_0_1_pos[0])
            if cdf_0_1_neg[-1] > cdf_0_1_neg[0]:
                cdf_0_1_neg = (cdf_0_1_neg - cdf_0_1_neg[0]) / (cdf_0_1_neg[-1] - cdf_0_1_neg[0])
            
            result['cond_more'][ABL] = 0.5 * (cdf_0_1_pos + cdf_0_1_neg)
    
    return {'batch': batch_name, 'animal': animal_id, 'cdfs': result}

# %%
# =============================================================================
# Main loop: Process each ILD
# =============================================================================

for ILD_target in ILD_VALUES:
    print(f"\n{'='*60}")
    print(f"Processing |ILD| = {ILD_target} (MORE PARAMS fit, using {N_WORKERS} workers)")
    print(f"{'='*60}")
    
    ild_idx = abs_ILD_arr.index(ILD_target)
    
    # Time points
    t_pts_wrt_stim = np.arange(-1, 1, 0.01)
    quantile_levels = np.linspace(0.09, 0.91, 50)
    
    # Storage for CDFs
    cdf_by_abl_cond_more = {20: [], 40: [], 60: []}
    
    # -------------------------------------------------------------------
    # Process animals in parallel
    # -------------------------------------------------------------------
    args_list = [(batch_name, animal_id, ILD_target, t_pts_wrt_stim) 
                 for batch_name, animal_id in batch_animal_pairs]
    
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(process_single_animal, args): args for args in args_list}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            if result is not None:
                has_data = any(ABL in result['cdfs']['cond_more'] for ABL in ABL_arr)
                if has_data:
                    print(f"  [{completed}/{len(args_list)}] ILD={ILD_target} | {result['batch']}/{result['animal']} done")
                    for ABL in ABL_arr:
                        if ABL in result['cdfs']['cond_more']:
                            cdf_by_abl_cond_more[ABL].append(result['cdfs']['cond_more'][ABL])
    
    n_animals_processed = sum(len(cdf_by_abl_cond_more[ABL]) for ABL in ABL_arr) // len(ABL_arr)
    print(f"Processed {n_animals_processed} animals with valid 5-param cond-fit data")
    
    if n_animals_processed == 0:
        print(f"WARNING: No 5-param cond-fit data found for |ILD|={ILD_target}. Skipping.")
        # Save empty pkl
        output_data = {
            'ILD_target': ILD_target,
            'ABL_arr': ABL_arr,
            'quantile_levels': quantile_levels,
            'theory': {'cond_more': {'mean_quantiles': {}, 'sem_quantiles': {}}},
            'empirical': {
                'plotting_quantiles': plotting_quantiles,
                'mean_unscaled': {ABL: mean_unscaled[ABL][:, ild_idx] for ABL in ABL_arr},
                'sem_unscaled': {ABL: sem_unscaled[ABL][:, ild_idx] for ABL in ABL_arr}
            },
            'gof': {'cond_more': {}},
            'R2_per_ABL': {'cond_more': {ABL: np.nan for ABL in ABL_arr}},
            'mean_R2': {'cond_more': np.nan}
        }
        pkl_file = f'quantiles_gof_ILD_{ILD_target}_more_params.pkl'
        with open(pkl_file, 'wb') as f:
            pickle.dump(output_data, f)
        print(f"Saved (empty): {pkl_file}")
        continue
    
    # -------------------------------------------------------------------
    # Compute mean quantiles across animals
    # -------------------------------------------------------------------
    t_cdf = t_pts_wrt_stim[(t_pts_wrt_stim >= 0) & (t_pts_wrt_stim <= 1)]
    
    mean_quantiles_cond_more = {}
    sem_quantiles_cond_more = {}
    
    for ABL in ABL_arr:
        if len(cdf_by_abl_cond_more[ABL]) > 0:
            quantiles_list = [invert_cdf(cdf, t_cdf, quantile_levels) for cdf in cdf_by_abl_cond_more[ABL]]
            q_arr = np.array(quantiles_list)
            mean_quantiles_cond_more[ABL] = np.nanmean(q_arr, axis=0)
            sem_quantiles_cond_more[ABL] = np.nanstd(q_arr, axis=0) / np.sqrt(np.sum(~np.isnan(q_arr), axis=0))
    
    # -------------------------------------------------------------------
    # Compute R² metrics
    # -------------------------------------------------------------------
    gof_results = {'cond_more': {}}
    
    for ABL in ABL_arr:
        emp_q = mean_unscaled[ABL][:, ild_idx]
        emp_sem = sem_unscaled[ABL][:, ild_idx]
        
        if ABL in mean_quantiles_cond_more:
            gof_results['cond_more'][ABL] = compute_gof_metrics(
                mean_quantiles_cond_more[ABL], quantile_levels, emp_q, plotting_quantiles, emp_sem)
    
    # Print R² table
    print(f"\nR² for |ILD| = {ILD_target} (MORE PARAMS):")
    print(f"{'ABL':<8} {'Cond-More R²':<12}")
    print("-"*20)
    for ABL in ABL_arr:
        r2 = gof_results['cond_more'].get(ABL, {}).get('R2', np.nan)
        print(f"{ABL:<8} {r2:<12.4f}")
    
    # Aggregate
    if gof_results['cond_more']:
        mean_r2_cond_more = np.mean([g['R2'] for g in gof_results['cond_more'].values()])
    else:
        mean_r2_cond_more = np.nan
    print(f"{'Mean':<8} {mean_r2_cond_more:<12.4f}")
    
    # -------------------------------------------------------------------
    # Save PKL
    # -------------------------------------------------------------------
    output_data = {
        'ILD_target': ILD_target,
        'ABL_arr': ABL_arr,
        'quantile_levels': quantile_levels,
        'theory': {
            'cond_more': {'mean_quantiles': mean_quantiles_cond_more, 'sem_quantiles': sem_quantiles_cond_more}
        },
        'empirical': {
            'plotting_quantiles': plotting_quantiles,
            'mean_unscaled': {ABL: mean_unscaled[ABL][:, ild_idx] for ABL in ABL_arr},
            'sem_unscaled': {ABL: sem_unscaled[ABL][:, ild_idx] for ABL in ABL_arr}
        },
        'gof': gof_results,
        'R2_per_ABL': {
            'cond_more': {ABL: gof_results['cond_more'].get(ABL, {}).get('R2', np.nan) for ABL in ABL_arr}
        },
        'mean_R2': {'cond_more': mean_r2_cond_more}
    }
    
    pkl_file = f'quantiles_gof_ILD_{ILD_target}_more_params.pkl'
    with open(pkl_file, 'wb') as f:
        pickle.dump(output_data, f)
    print(f"Saved: {pkl_file}")
    
    # -------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    
    for i, ABL in enumerate(ABL_arr):
        color = abl_colors[ABL]
        
        # Cond fit more params (dashed)
        if ABL in mean_quantiles_cond_more:
            ax.plot(quantile_levels, mean_quantiles_cond_more[ABL], color=color, 
                    linewidth=2, linestyle='--', label='Cond-More' if i == 0 else None)
        
        # Empirical data
        emp_q = mean_unscaled[ABL][:, ild_idx]
        emp_sem = sem_unscaled[ABL][:, ild_idx]
        ax.errorbar(plotting_quantiles, emp_q, yerr=emp_sem, marker='o', linestyle='none',
                    color=color, markersize=8, capsize=3, label='Data' if i == 0 else None)
    
    ax.set_xlabel('Quantile', fontsize=14)
    ax.set_ylabel('RT (s)', fontsize=14)
    ax.set_title(f'|ILD| = {ILD_target} dB (5-param fit)  |  R²={mean_r2_cond_more:.3f}', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0.05, 0.95)
    ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
    ax.set_xticklabels(['10', '30', '50', '70', '90'])
    
    plt.tight_layout()
    png_file = f'quantiles_gof_ILD_{ILD_target}_more_params.png'
    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {png_file}")

print("\n" + "="*60)
print("All ILDs processed (MORE PARAMS)!")
print("="*60)
# %%
