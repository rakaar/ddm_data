# %%
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the quantile data
with open('fig1_quantiles_plot_data.pkl', 'rb') as f:
    quantile_data = pickle.load(f)

# Extract data
ABL_arr = quantile_data['ABL_arr']  # [20, 40, 60]
abs_ILD_arr = quantile_data['abs_ILD_arr']  # [1, 2, 4, 8, 16]
plotting_quantiles = quantile_data['plotting_quantiles']  # e.g., [0.1, 0.3, 0.5, 0.7, 0.9]
mean_unscaled = quantile_data['mean_unscaled']  # dict: ABL -> (n_quantiles, n_ilds)
sem_unscaled = quantile_data['sem_unscaled']

# Get index for ILD = 16
ild_idx = abs_ILD_arr.index(16)

# Colors for each ABL
abl_colors = {20: 'tab:blue', 40: 'tab:orange', 60: 'tab:green'}

# Create figure
fig, ax = plt.subplots(figsize=(6, 5))

for abl in ABL_arr:
    # Extract quantile values for ILD=16
    q_vals = mean_unscaled[abl][:, ild_idx]  # shape: (n_quantiles,)
    q_sem = sem_unscaled[abl][:, ild_idx]
    
    ax.errorbar(
        plotting_quantiles,
        q_vals,
        yerr=q_sem,
        marker='o',
        linestyle='none',
        color=abl_colors[abl],
        label=f'ABL {abl} dB',
        capsize=2
    )

ax.set_xlabel('Quantile', fontsize=14)
ax.set_ylabel('RT (s)', fontsize=14)
ax.set_title('RT Quantiles at |ILD| = 16 dB (Avg across animals)', fontsize=14)
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()
plt.savefig('cond_fit_goodness_ild16_v3.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# =============================================================================
# STEP 1: Load batch-animal pairs and get vanilla_TIED_params
#         for ILD=16 at ABL=20,40,60
# =============================================================================
import os
import sys
from collections import defaultdict

# Add fit_each_condn to path (needed for unpickling vbmc objects)
sys.path.insert(0, '/home/rlab/raghavendra/ddm_data/fit_each_condn')

# --- Get Batch-Animal Pairs ---
DESIRED_BATCHES = ['SD', 'LED34', 'LED6', 'LED8', 'LED7', 'LED34_even']
batch_dir = '/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/batch_csvs'
batch_files = [f'batch_{batch_name}_valid_and_aborts.csv' for batch_name in DESIRED_BATCHES]

import pandas as pd
merged_data = pd.concat([
    pd.read_csv(os.path.join(batch_dir, fname)) for fname in batch_files if os.path.exists(os.path.join(batch_dir, fname))
], ignore_index=True)

merged_valid = merged_data[merged_data['success'].isin([1, -1])].copy()

# --- Get unique batch-animal pairs ---
batch_animal_pairs = sorted(list(map(tuple, merged_valid[['batch_name', 'animal']].drop_duplicates().values)))
print(f"Found {len(batch_animal_pairs)} batch-animal pairs")

# --- Function to load vanilla_tied params and abort params from animal pkl file ---
def get_animal_params(batch_name, animal_id):
    """Load vanilla_tied params (rate_lambda, T_0, theta_E, w, t_E_aff, del_go) and abort params."""
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
    
    # Get all params from vanilla_tied only
    vanilla_params = {}
    if 'vbmc_vanilla_tied_results' in data:
        v = data['vbmc_vanilla_tied_results']
        vanilla_params = {
            'rate_lambda': np.mean(v['rate_lambda_samples']),
            'T_0': np.mean(v['T_0_samples']),
            'theta_E': np.mean(v['theta_E_samples']),
            'w': np.mean(v['w_samples']),
            't_E_aff': np.mean(v['t_E_aff_samples']),
            'del_go': np.mean(v['del_go_samples'])
        }
    else:
        return None
    
    return {'abort_params': abort_params, 'vanilla_params': vanilla_params}

# --- Load params for ILD=16, ABL=20,40,60 for all animals ---
ABL_arr = [20, 40, 60]
ILD_target = 16  # We only care about ILD=16

# Store: {(batch, animal): {'vanilla_params': {...}, 'abort_params': {...}}}
animal_params_all = {}

for batch_name, animal_id in batch_animal_pairs:
    animal_id_str = str(animal_id)
    
    # Check if animal has ILD=16 data in their batch CSV
    csv_file = os.path.join(batch_dir, f'batch_{batch_name}_valid_and_aborts.csv')
    df_check = pd.read_csv(csv_file)
    df_animal_check = df_check[(df_check['animal'] == int(animal_id)) & (df_check['success'].isin([1, -1]))]
    if ILD_target not in df_animal_check['ILD'].abs().unique():
        print(f"Skipping {batch_name}/{animal_id_str}: no ILD={ILD_target} data")
        continue
    
    # Get vanilla_tied params and abort params
    animal_data = get_animal_params(batch_name, animal_id_str)
    if animal_data is None:
        print(f"Skipping {batch_name}/{animal_id_str}: no animal pkl or missing vanilla_tied")
        continue
    
    vanilla_params = animal_data['vanilla_params']
    abort_params = animal_data['abort_params']
    
    # Compute Z_E from w: w = 0.5 + (Z_E / (2 * theta_E)) => Z_E = (w - 0.5) * 2 * theta_E
    Z_E = (vanilla_params['w'] - 0.5) * 2 * vanilla_params['theta_E']
    vanilla_params['Z_E'] = Z_E
    
    animal_params_all[(batch_name, animal_id)] = {
        'vanilla_params': vanilla_params,
        'abort_params': abort_params
    }

# Print summary
print(f"\nLoaded params for {len(animal_params_all)} animals")

# Show a sample
sample_key = list(animal_params_all.keys())[0]
sample_v = animal_params_all[sample_key]['vanilla_params']
print(f"\nSample params for {sample_key}:")
print(f"  rate_lambda={sample_v['rate_lambda']:.3f}, T_0={sample_v['T_0']:.6f}, theta_E={sample_v['theta_E']:.3f}")
print(f"  w={sample_v['w']:.3f}, Z_E={sample_v['Z_E']:.3f}, t_E_aff={sample_v['t_E_aff']:.3f}")

# %%
# =============================================================================
# STEP 2 (V3): Compute theoretical CDF using cum_pro_and_reactive_time_vary_fn
#              which takes rate_lambda, T_0, theta_E directly (no gamma/omega)
#              Time is wrt fixation: t_wrt_fix = t_wrt_stim + t_stim
# =============================================================================
from time_vary_norm_utils import cum_pro_and_reactive_time_vary_fn

K_max = 10
N_theory = int(1e3)

# Time points wrt stimulus onset
t_pts_wrt_stim = np.arange(-1, 1, 0.01)

# Store CDFs: {ABL: list of CDF arrays (one per animal)}
cdf_by_abl = {20: [], 40: [], 60: []}

n_animals = len(animal_params_all)
for animal_num, ((batch_name, animal_id), params_dict) in enumerate(animal_params_all.items(), 1):
    animal_id_str = str(animal_id)
    
    print(f"  [{animal_num}/{n_animals}] Processing {batch_name}/{animal_id_str}...", end=' ', flush=True)
    
    # Load animal's data for t_stim sampling
    csv_file = os.path.join(batch_dir, f'batch_{batch_name}_valid_and_aborts.csv')
    df = pd.read_csv(csv_file)
    df_animal = df[df['animal'] == int(animal_id)]
    
    # Sample t_stim from this animal's intended_fix distribution
    t_stim_samples = df_animal['intended_fix'].sample(N_theory, replace=True).values
    
    # Extract params
    vanilla_params = params_dict['vanilla_params']
    abort_params = params_dict['abort_params']
    
    V_A = abort_params['V_A']
    theta_A = abort_params['theta_A']
    t_A_aff = abort_params['t_A_aff']
    
    rate_lambda = vanilla_params['rate_lambda']
    T_0 = vanilla_params['T_0']
    theta_E = vanilla_params['theta_E']
    Z_E = vanilla_params['Z_E']
    t_E_aff = vanilla_params['t_E_aff']
    
    # Compute CDF for each ABL
    for ABL in ABL_arr:
        # Compute CDF samples for each t_stim, averaging over ILD=+16 and ILD=-16
        cdf_samples_pos = np.zeros((N_theory, len(t_pts_wrt_stim)))
        cdf_samples_neg = np.zeros((N_theory, len(t_pts_wrt_stim)))
        
        for idx, t_stim in enumerate(t_stim_samples):
            # Convert to time wrt fixation
            t_pts_wrt_fix = t_pts_wrt_stim + t_stim
            
            # Compute CDF at each time point for both ILD signs
            for t_idx, t_fix in enumerate(t_pts_wrt_fix):
                # ILD = +16
                cdf_samples_pos[idx, t_idx] = cum_pro_and_reactive_time_vary_fn(
                    t=t_fix,
                    c_A_trunc_time=None,
                    V_A=V_A, theta_A=theta_A, t_A_aff=t_A_aff,
                    t_stim=t_stim, ABL=ABL, ILD=ILD_target,
                    rate_lambda=rate_lambda, T0=T_0, theta_E=theta_E, Z_E=Z_E, t_E_aff=t_E_aff,
                    phi_params=None, rate_norm_l=0,
                    is_norm=False, is_time_vary=False, K_max=K_max
                )
                # ILD = -16
                cdf_samples_neg[idx, t_idx] = cum_pro_and_reactive_time_vary_fn(
                    t=t_fix,
                    c_A_trunc_time=None,
                    V_A=V_A, theta_A=theta_A, t_A_aff=t_A_aff,
                    t_stim=t_stim, ABL=ABL, ILD=-ILD_target,
                    rate_lambda=rate_lambda, T0=T_0, theta_E=theta_E, Z_E=Z_E, t_E_aff=t_E_aff,
                    phi_params=None, rate_norm_l=0,
                    is_norm=False, is_time_vary=False, K_max=K_max
                )
        
        # Average CDF across t_stim samples for each ILD sign
        cdf_mean_pos = np.mean(cdf_samples_pos, axis=0)
        cdf_mean_neg = np.mean(cdf_samples_neg, axis=0)
        
        # Extract CDF for t_wrt_stim in [0, 1]
        mask_0_1 = (t_pts_wrt_stim >= 0) & (t_pts_wrt_stim <= 1)
        cdf_0_1_pos = cdf_mean_pos[mask_0_1]
        cdf_0_1_neg = cdf_mean_neg[mask_0_1]
        
        # Normalize each CDF separately before averaging
        # This removes the abort probability that occurred before stimulus
        cdf_at_0_pos = cdf_0_1_pos[0]
        cdf_at_end_pos = cdf_0_1_pos[-1]
        if cdf_at_end_pos > cdf_at_0_pos:
            cdf_0_1_pos = (cdf_0_1_pos - cdf_at_0_pos) / (cdf_at_end_pos - cdf_at_0_pos)
        
        cdf_at_0_neg = cdf_0_1_neg[0]
        cdf_at_end_neg = cdf_0_1_neg[-1]
        if cdf_at_end_neg > cdf_at_0_neg:
            cdf_0_1_neg = (cdf_0_1_neg - cdf_at_0_neg) / (cdf_at_end_neg - cdf_at_0_neg)
        
        # Average the normalized CDFs (matching how data pools +16 and -16 trials)
        cdf_0_1 = 0.5 * (cdf_0_1_pos + cdf_0_1_neg)
        
        cdf_by_abl[ABL].append(cdf_0_1)
    
    print(f"done (3 ABLs)")

# Time points for CDF (wrt stim, in [0, 1])
t_cdf = t_pts_wrt_stim[(t_pts_wrt_stim >= 0) & (t_pts_wrt_stim <= 1)]

# Convert to arrays and compute mean/SEM
mean_cdf = {}
sem_cdf = {}
for ABL in ABL_arr:
    if len(cdf_by_abl[ABL]) > 0:
        cdf_arr = np.array(cdf_by_abl[ABL])
        mean_cdf[ABL] = np.nanmean(cdf_arr, axis=0)
        n_valid = np.sum(~np.isnan(cdf_arr), axis=0)
        sem_cdf[ABL] = np.nanstd(cdf_arr, axis=0) / np.sqrt(n_valid)
        print(f"ABL={ABL}: {len(cdf_by_abl[ABL])} animals")

# --- Plot average CDF with SEM shaded ---
abl_colors = {20: 'tab:blue', 40: 'tab:orange', 60: 'tab:green'}

fig, ax = plt.subplots(figsize=(6, 5))
for ABL in ABL_arr:
    if ABL in mean_cdf:
        ax.plot(t_cdf, mean_cdf[ABL], color=abl_colors[ABL], label=f'ABL {ABL} dB')
        ax.fill_between(t_cdf, mean_cdf[ABL] - sem_cdf[ABL], mean_cdf[ABL] + sem_cdf[ABL],
                        color=abl_colors[ABL], alpha=0.3)

ax.set_xlabel('RT (s)', fontsize=14)
ax.set_ylabel('CDF', fontsize=14)
ax.set_title('Theoretical CDF at |ILD| = 16 dB (V3 - using vanilla_tied params)', fontsize=14)
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig('cond_fit_theoretical_cdf_ild16_v3.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nCDF computation complete. Saved plot to cond_fit_theoretical_cdf_ild16_v3.png")

# %%
# =============================================================================
# STEP 3-5: Invert CDFs to get quantiles (10-90%, 50 pts), average across animals,
#           and overlay on empirical data quantiles
# =============================================================================

# Define quantile levels (10% to 90%, 50 points)
quantile_levels = np.linspace(0.09, 0.91, 50)

# Function to invert CDF: given CDF values and time points, find time for each quantile
def invert_cdf(cdf_vals, t_vals, quantile_levels):
    """Invert CDF to get RT values at given quantile levels."""
    quantile_rts = np.zeros(len(quantile_levels))
    for i, q in enumerate(quantile_levels):
        # Find first index where CDF >= q
        idx = np.searchsorted(cdf_vals, q)
        if idx == 0:
            quantile_rts[i] = t_vals[0]
        elif idx >= len(t_vals):
            quantile_rts[i] = t_vals[-1]
        else:
            # Linear interpolation
            t0, t1 = t_vals[idx-1], t_vals[idx]
            c0, c1 = cdf_vals[idx-1], cdf_vals[idx]
            if c1 != c0:
                quantile_rts[i] = t0 + (q - c0) * (t1 - t0) / (c1 - c0)
            else:
                quantile_rts[i] = t0
    return quantile_rts

# Invert each animal's CDF to get quantiles
quantiles_by_abl = {20: [], 40: [], 60: []}

for ABL in ABL_arr:
    for cdf in cdf_by_abl[ABL]:
        q_rts = invert_cdf(cdf, t_cdf, quantile_levels)
        quantiles_by_abl[ABL].append(q_rts)

# Compute mean and SEM of quantiles across animals
mean_quantiles = {}
sem_quantiles = {}
for ABL in ABL_arr:
    if len(quantiles_by_abl[ABL]) > 0:
        q_arr = np.array(quantiles_by_abl[ABL])
        mean_quantiles[ABL] = np.nanmean(q_arr, axis=0)
        n_valid = np.sum(~np.isnan(q_arr), axis=0)
        sem_quantiles[ABL] = np.nanstd(q_arr, axis=0) / np.sqrt(n_valid)
        print(f"ABL={ABL}: {len(quantiles_by_abl[ABL])} animals for quantiles")

# --- Plot: Theoretical quantile curves with empirical data overlaid ---
abl_colors = {20: 'tab:blue', 40: 'tab:orange', 60: 'tab:green'}

# Save data for plotting in a separate file
quantiles_data = {
    'theory': {
        'quantile_levels': quantile_levels,
        'mean_quantiles': mean_quantiles,
        'sem_quantiles': sem_quantiles,
    },
    'empirical': {
        'plotting_quantiles': plotting_quantiles,
        'mean_unscaled': mean_unscaled,
        'sem_unscaled': sem_unscaled,
        'ild_idx': ild_idx,
    },
    'ABL_arr': ABL_arr,
    'abl_colors': abl_colors,
    'ILD_target': ILD_target,
}
with open('ILD_16_vanila_model_quantiles.pkl', 'wb') as f:
    pickle.dump(quantiles_data, f)
print("Saved quantiles data to ILD_16_vanila_model_quantiles.pkl")

fig, ax = plt.subplots(figsize=(7, 6))

# Plot theoretical quantile curves (continuous, with SEM shaded)
for ABL in ABL_arr:
    if ABL in mean_quantiles:
        ax.plot(quantile_levels, mean_quantiles[ABL], color=abl_colors[ABL], 
                linewidth=2, label=f'Theory ABL {ABL} dB')
        ax.fill_between(quantile_levels, 
                        mean_quantiles[ABL] - sem_quantiles[ABL], 
                        mean_quantiles[ABL] + sem_quantiles[ABL],
                        color=abl_colors[ABL], alpha=0.3)

# Overlay empirical data quantiles (from the pickle file)
for abl in ABL_arr:
    q_vals = mean_unscaled[abl][:, ild_idx]  # ILD=16
    q_sem = sem_unscaled[abl][:, ild_idx]
    ax.errorbar(
        plotting_quantiles,
        q_vals,
        yerr=q_sem,
        marker='o',
        linestyle='none',
        color=abl_colors[abl],
        markersize=8,
        capsize=3,
        label=f'Data ABL {abl} dB'
    )

ax.set_xlabel('Quantile', fontsize=14)
ax.set_ylabel('RT (s)', fontsize=14)
# ax.set_title('Theoretical vs Empirical RT Quantiles at |ILD| = 16 dB (V3)', fontsize=14)
# ax.legend(loc='upper left', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_xlim(0.05, 0.95)
ax.set_ylim(0, 0.35)
ax.set_yticks([0, 0.35])
ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
ax.set_xticklabels(['10', '30', '50', '70', '90'])

plt.tight_layout()
plt.savefig('cond_fit_quantiles_theory_vs_data_ild16_v3.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nQuantile comparison saved to cond_fit_quantiles_theory_vs_data_ild16_v3.png")
# %%
