# %%
"""
Compute quantile goodness-of-fit (R², RMSE, SSE) for both cond-fit and vanilla models
across all ILD values (1, 2, 4, 8, 16).

Outputs for each ILD:
- PNG: quantiles_gof_ILD_{ild}.png
- PKL: quantiles_gof_ILD_{ild}.pkl (contains theory, empirical data, and R² metrics)
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
from time_vary_norm_utils import cum_pro_and_reactive_time_vary_fn

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

def get_cond_fit_params(batch_name, animal_id, ABL, ILD):
    """Load gamma, omega from condition-by-condition fit pkl file."""
    pkl_folder = '/home/rlab/raghavendra/ddm_data/fit_each_condn/each_animal_cond_fit_gama_omega_pkl_files'
    pkl_file = os.path.join(pkl_folder, f'vbmc_cond_by_cond_{batch_name}_{animal_id}_{ABL}_ILD_{ILD}_FIX_t_E_w_del_go_same_as_parametric.pkl')
    if not os.path.exists(pkl_file):
        return None
    with open(pkl_file, 'rb') as f:
        vp = pickle.load(f)
    vp = vp.vp
    vp_samples = vp.sample(int(1e4))[0]
    gamma = float(np.mean(vp_samples[:, 0]))
    omega = float(np.mean(vp_samples[:, 1]))
    return {'gamma': gamma, 'omega': omega}

def get_animal_params(batch_name, animal_id):
    """Load w, t_E_aff, del_go (avg of vanilla and norm) and abort params."""
    pkl_file = f'/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/results_{batch_name}_animal_{animal_id}.pkl'
    if not os.path.exists(pkl_file):
        return None
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    # Abort params
    abort_params = {}
    if 'vbmc_aborts_results' in data:
        abort_samples = data['vbmc_aborts_results']
        abort_params['V_A'] = np.mean(abort_samples['V_A_samples'])
        abort_params['theta_A'] = np.mean(abort_samples['theta_A_samples'])
        abort_params['t_A_aff'] = np.mean(abort_samples['t_A_aff_samp'])
    
    # Vanilla tied params (for vanilla model)
    vanilla_tied_params = {}
    if 'vbmc_vanilla_tied_results' in data:
        v = data['vbmc_vanilla_tied_results']
        vanilla_tied_params = {
            'rate_lambda': np.mean(v['rate_lambda_samples']),
            'T_0': np.mean(v['T_0_samples']),
            'theta_E': np.mean(v['theta_E_samples']),
            'w': np.mean(v['w_samples']),
            't_E_aff': np.mean(v['t_E_aff_samples']),
            'del_go': np.mean(v['del_go_samples'])
        }
    
    # Get w, t_E_aff, del_go from both vanilla and norm, then average (for cond fit)
    vanilla_params = {}
    norm_params = {}
    if 'vbmc_vanilla_tied_results' in data:
        v = data['vbmc_vanilla_tied_results']
        vanilla_params = {
            'w': np.mean(v['w_samples']),
            't_E_aff': np.mean(v['t_E_aff_samples']),
            'del_go': np.mean(v['del_go_samples'])
        }
    if 'vbmc_norm_tied_results' in data:
        n = data['vbmc_norm_tied_results']
        norm_params = {
            'w': np.mean(n['w_samples']),
            't_E_aff': np.mean(n['t_E_aff_samples']),
            'del_go': np.mean(n['del_go_samples'])
        }
    
    if vanilla_params and norm_params:
        avg_params = {
            'w': (vanilla_params['w'] + norm_params['w']) / 2,
            't_E_aff': (vanilla_params['t_E_aff'] + norm_params['t_E_aff']) / 2,
            'del_go': (vanilla_params['del_go'] + norm_params['del_go']) / 2
        }
    elif vanilla_params:
        avg_params = vanilla_params
    elif norm_params:
        avg_params = norm_params
    else:
        return None
    
    return {
        'abort_params': abort_params, 
        'avg_params': avg_params,
        'vanilla_tied_params': vanilla_tied_params
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
    
    # Get animal params
    animal_data = get_animal_params(batch_name, animal_id_str)
    if animal_data is None:
        return None
    
    # Check if this animal has data for this ILD
    csv_file = os.path.join(batch_dir, f'batch_{batch_name}_valid_and_aborts.csv')
    df = pd.read_csv(csv_file)
    df_animal = df[df['animal'] == int(animal_id)]
    if ILD_target not in df_animal['ILD'].abs().unique():
        return None
    
    # Sample t_stim
    t_stim_samples = df_animal['intended_fix'].sample(N_theory, replace=True).values
    
    # Get abort params
    V_A = animal_data['abort_params']['V_A']
    theta_A = animal_data['abort_params']['theta_A']
    t_A_aff = animal_data['abort_params']['t_A_aff']
    
    result = {'cond': {}, 'vanilla': {}}
    
    for ABL in ABL_arr:
        # --- COND FIT MODEL ---
        cond_params_pos = get_cond_fit_params(batch_name, animal_id_str, ABL, ILD_target)
        cond_params_neg = get_cond_fit_params(batch_name, animal_id_str, ABL, -ILD_target)
        
        if cond_params_pos is not None and cond_params_neg is not None:
            gamma_pos, omega_pos = cond_params_pos['gamma'], cond_params_pos['omega']
            gamma_neg, omega_neg = cond_params_neg['gamma'], cond_params_neg['omega']
            w_cond = animal_data['avg_params']['w']
            t_E_aff_cond = animal_data['avg_params']['t_E_aff']
            
            cdf_samples_pos = np.zeros((N_theory, len(t_pts_wrt_stim)))
            cdf_samples_neg = np.zeros((N_theory, len(t_pts_wrt_stim)))
            
            for idx, t_stim in enumerate(t_stim_samples):
                t_pts_wrt_fix = t_pts_wrt_stim + t_stim
                for t_idx, t_fix in enumerate(t_pts_wrt_fix):
                    c_A = cum_A_t_fn(t_fix - t_A_aff, V_A, theta_A)
                    t_evidence = t_pts_wrt_stim[t_idx] - t_E_aff_cond
                    
                    c_E_up_pos = CDF_E_minus_small_t_NORM_omega_gamma_with_w_fn(t_evidence, gamma_pos, omega_pos, 1, w_cond, K_max)
                    c_E_down_pos = CDF_E_minus_small_t_NORM_omega_gamma_with_w_fn(t_evidence, gamma_pos, omega_pos, -1, w_cond, K_max)
                    cdf_samples_pos[idx, t_idx] = c_A + (c_E_up_pos + c_E_down_pos) - c_A * (c_E_up_pos + c_E_down_pos)
                    
                    c_E_up_neg = CDF_E_minus_small_t_NORM_omega_gamma_with_w_fn(t_evidence, gamma_neg, omega_neg, 1, w_cond, K_max)
                    c_E_down_neg = CDF_E_minus_small_t_NORM_omega_gamma_with_w_fn(t_evidence, gamma_neg, omega_neg, -1, w_cond, K_max)
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
            
            result['cond'][ABL] = 0.5 * (cdf_0_1_pos + cdf_0_1_neg)
        
        # --- VANILLA MODEL ---
        if animal_data['vanilla_tied_params']:
            vp = animal_data['vanilla_tied_params']
            rate_lambda = vp['rate_lambda']
            T_0 = vp['T_0']
            theta_E = vp['theta_E']
            w_van = vp['w']
            t_E_aff_van = vp['t_E_aff']
            Z_E = (w_van - 0.5) * 2 * theta_E
            
            cdf_samples_pos = np.zeros((N_theory, len(t_pts_wrt_stim)))
            cdf_samples_neg = np.zeros((N_theory, len(t_pts_wrt_stim)))
            
            for idx, t_stim in enumerate(t_stim_samples):
                t_pts_wrt_fix = t_pts_wrt_stim + t_stim
                for t_idx, t_fix in enumerate(t_pts_wrt_fix):
                    cdf_samples_pos[idx, t_idx] = cum_pro_and_reactive_time_vary_fn(
                        t=t_fix, c_A_trunc_time=None,
                        V_A=V_A, theta_A=theta_A, t_A_aff=t_A_aff,
                        t_stim=t_stim, ABL=ABL, ILD=ILD_target,
                        rate_lambda=rate_lambda, T0=T_0, theta_E=theta_E, Z_E=Z_E, t_E_aff=t_E_aff_van,
                        phi_params=None, rate_norm_l=0,
                        is_norm=False, is_time_vary=False, K_max=K_max
                    )
                    cdf_samples_neg[idx, t_idx] = cum_pro_and_reactive_time_vary_fn(
                        t=t_fix, c_A_trunc_time=None,
                        V_A=V_A, theta_A=theta_A, t_A_aff=t_A_aff,
                        t_stim=t_stim, ABL=ABL, ILD=-ILD_target,
                        rate_lambda=rate_lambda, T0=T_0, theta_E=theta_E, Z_E=Z_E, t_E_aff=t_E_aff_van,
                        phi_params=None, rate_norm_l=0,
                        is_norm=False, is_time_vary=False, K_max=K_max
                    )
            
            cdf_mean_pos = np.mean(cdf_samples_pos, axis=0)
            cdf_mean_neg = np.mean(cdf_samples_neg, axis=0)
            mask_0_1 = (t_pts_wrt_stim >= 0) & (t_pts_wrt_stim <= 1)
            cdf_0_1_pos = cdf_mean_pos[mask_0_1]
            cdf_0_1_neg = cdf_mean_neg[mask_0_1]
            
            if cdf_0_1_pos[-1] > cdf_0_1_pos[0]:
                cdf_0_1_pos = (cdf_0_1_pos - cdf_0_1_pos[0]) / (cdf_0_1_pos[-1] - cdf_0_1_pos[0])
            if cdf_0_1_neg[-1] > cdf_0_1_neg[0]:
                cdf_0_1_neg = (cdf_0_1_neg - cdf_0_1_neg[0]) / (cdf_0_1_neg[-1] - cdf_0_1_neg[0])
            
            result['vanilla'][ABL] = 0.5 * (cdf_0_1_pos + cdf_0_1_neg)
    
    return {'batch': batch_name, 'animal': animal_id, 'cdfs': result}

# %%
# =============================================================================
# Main loop: Process each ILD
# =============================================================================

for ILD_target in ILD_VALUES:
    print(f"\n{'='*60}")
    print(f"Processing |ILD| = {ILD_target} (using {N_WORKERS} workers)")
    print(f"{'='*60}")
    
    ild_idx = abs_ILD_arr.index(ILD_target)
    
    # Time points
    t_pts_wrt_stim = np.arange(-1, 1, 0.01)
    quantile_levels = np.linspace(0.09, 0.91, 50)
    
    # Storage for CDFs
    cdf_by_abl_cond = {20: [], 40: [], 60: []}
    cdf_by_abl_vanilla = {20: [], 40: [], 60: []}
    
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
                print(f"  [{completed}/{len(args_list)}] ILD={ILD_target} | {result['batch']}/{result['animal']} done")
                for ABL in ABL_arr:
                    if ABL in result['cdfs']['cond']:
                        cdf_by_abl_cond[ABL].append(result['cdfs']['cond'][ABL])
                    if ABL in result['cdfs']['vanilla']:
                        cdf_by_abl_vanilla[ABL].append(result['cdfs']['vanilla'][ABL])
    
    n_animals_processed = len(cdf_by_abl_cond[20])  # Use any ABL as proxy
    print(f"Processed {n_animals_processed} animals with valid data")
    
    # -------------------------------------------------------------------
    # Compute mean quantiles across animals
    # -------------------------------------------------------------------
    t_cdf = t_pts_wrt_stim[(t_pts_wrt_stim >= 0) & (t_pts_wrt_stim <= 1)]
    
    mean_quantiles_cond = {}
    sem_quantiles_cond = {}
    mean_quantiles_vanilla = {}
    sem_quantiles_vanilla = {}
    
    for ABL in ABL_arr:
        # Cond fit
        if len(cdf_by_abl_cond[ABL]) > 0:
            quantiles_list = [invert_cdf(cdf, t_cdf, quantile_levels) for cdf in cdf_by_abl_cond[ABL]]
            q_arr = np.array(quantiles_list)
            mean_quantiles_cond[ABL] = np.nanmean(q_arr, axis=0)
            sem_quantiles_cond[ABL] = np.nanstd(q_arr, axis=0) / np.sqrt(np.sum(~np.isnan(q_arr), axis=0))
        
        # Vanilla
        if len(cdf_by_abl_vanilla[ABL]) > 0:
            quantiles_list = [invert_cdf(cdf, t_cdf, quantile_levels) for cdf in cdf_by_abl_vanilla[ABL]]
            q_arr = np.array(quantiles_list)
            mean_quantiles_vanilla[ABL] = np.nanmean(q_arr, axis=0)
            sem_quantiles_vanilla[ABL] = np.nanstd(q_arr, axis=0) / np.sqrt(np.sum(~np.isnan(q_arr), axis=0))
    
    # -------------------------------------------------------------------
    # Compute R² metrics
    # -------------------------------------------------------------------
    gof_results = {'cond': {}, 'vanilla': {}}
    
    for ABL in ABL_arr:
        emp_q = mean_unscaled[ABL][:, ild_idx]
        emp_sem = sem_unscaled[ABL][:, ild_idx]
        
        if ABL in mean_quantiles_cond:
            gof_results['cond'][ABL] = compute_gof_metrics(
                mean_quantiles_cond[ABL], quantile_levels, emp_q, plotting_quantiles, emp_sem)
        
        if ABL in mean_quantiles_vanilla:
            gof_results['vanilla'][ABL] = compute_gof_metrics(
                mean_quantiles_vanilla[ABL], quantile_levels, emp_q, plotting_quantiles, emp_sem)
    
    # Print R² table
    print(f"\nR² for |ILD| = {ILD_target}:")
    print(f"{'ABL':<8} {'Cond R²':<12} {'Vanilla R²':<12}")
    print("-"*32)
    for ABL in ABL_arr:
        cond_r2 = gof_results['cond'].get(ABL, {}).get('R2', np.nan)
        van_r2 = gof_results['vanilla'].get(ABL, {}).get('R2', np.nan)
        print(f"{ABL:<8} {cond_r2:<12.4f} {van_r2:<12.4f}")
    
    # Aggregate
    if gof_results['cond']:
        mean_r2_cond = np.mean([g['R2'] for g in gof_results['cond'].values()])
    else:
        mean_r2_cond = np.nan
    if gof_results['vanilla']:
        mean_r2_vanilla = np.mean([g['R2'] for g in gof_results['vanilla'].values()])
    else:
        mean_r2_vanilla = np.nan
    print(f"{'Mean':<8} {mean_r2_cond:<12.4f} {mean_r2_vanilla:<12.4f}")
    
    # -------------------------------------------------------------------
    # Save PKL
    # -------------------------------------------------------------------
    output_data = {
        'ILD_target': ILD_target,
        'ABL_arr': ABL_arr,
        'quantile_levels': quantile_levels,
        'theory': {
            'cond': {'mean_quantiles': mean_quantiles_cond, 'sem_quantiles': sem_quantiles_cond},
            'vanilla': {'mean_quantiles': mean_quantiles_vanilla, 'sem_quantiles': sem_quantiles_vanilla}
        },
        'empirical': {
            'plotting_quantiles': plotting_quantiles,
            'mean_unscaled': {ABL: mean_unscaled[ABL][:, ild_idx] for ABL in ABL_arr},
            'sem_unscaled': {ABL: sem_unscaled[ABL][:, ild_idx] for ABL in ABL_arr}
        },
        'gof': gof_results,
        'R2_per_ABL': {
            'cond': {ABL: gof_results['cond'].get(ABL, {}).get('R2', np.nan) for ABL in ABL_arr},
            'vanilla': {ABL: gof_results['vanilla'].get(ABL, {}).get('R2', np.nan) for ABL in ABL_arr}
        },
        'mean_R2': {'cond': mean_r2_cond, 'vanilla': mean_r2_vanilla}
    }
    
    pkl_file = f'quantiles_gof_ILD_{ILD_target}.pkl'
    with open(pkl_file, 'wb') as f:
        pickle.dump(output_data, f)
    print(f"Saved: {pkl_file}")
    
    # -------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    
    for i, ABL in enumerate(ABL_arr):
        color = abl_colors[ABL]
        
        # Cond fit (solid)
        if ABL in mean_quantiles_cond:
            ax.plot(quantile_levels, mean_quantiles_cond[ABL], color=color, 
                    linewidth=2, linestyle='-', label='Cond fit' if i == 0 else None)
        
        # Vanilla (dotted)
        if ABL in mean_quantiles_vanilla:
            ax.plot(quantile_levels, mean_quantiles_vanilla[ABL], color=color, 
                    linewidth=2, linestyle=':', label='Vanilla' if i == 0 else None)
        
        # Empirical data
        emp_q = mean_unscaled[ABL][:, ild_idx]
        emp_sem = sem_unscaled[ABL][:, ild_idx]
        ax.errorbar(plotting_quantiles, emp_q, yerr=emp_sem, marker='o', linestyle='none',
                    color=color, markersize=8, capsize=3, label='Data' if i == 0 else None)
    
    ax.set_xlabel('Quantile', fontsize=14)
    ax.set_ylabel('RT (s)', fontsize=14)
    ax.set_title(f'|ILD| = {ILD_target} dB  |  R²: Cond={mean_r2_cond:.3f}, Vanilla={mean_r2_vanilla:.3f}', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0.05, 0.95)
    ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
    ax.set_xticklabels(['10', '30', '50', '70', '90'])
    
    plt.tight_layout()
    png_file = f'quantiles_gof_ILD_{ILD_target}.png'
    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {png_file}")

print("\n" + "="*60)
print("All ILDs processed!")
print("="*60)
# %%
