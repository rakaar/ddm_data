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
plt.savefig('cond_fit_goodness_ild16_v2.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# =============================================================================
# STEP 1: Load batch-animal pairs and get gamma, omega from condition-fit pkl files
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

# --- Function to load gamma, omega from condition-fit pkl files ---
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

# --- Function to load w, t_E_aff, del_go and abort params from animal pkl file ---
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
    
    # Get w, t_E_aff, del_go from both vanilla and norm, then average
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
    
    # Average
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
    
    return {'abort_params': abort_params, 'avg_params': avg_params}

# --- Load params for ILD=16, ABL=20,40,60 for all animals ---
ABL_arr = [20, 40, 60]
ILD_target = 16  # We only care about ILD=16

# Store: {(batch, animal): {ABL: {'gamma', 'omega', 'w', 't_E_aff', 'del_go', 'V_A', 'theta_A', 't_A_aff'}}}
animal_params_ild16 = {}

for batch_name, animal_id in batch_animal_pairs:
    animal_id_str = str(animal_id)
    
    # Get w, t_E_aff, del_go, abort params
    animal_data = get_animal_params(batch_name, animal_id_str)
    if animal_data is None:
        print(f"Skipping {batch_name}/{animal_id_str}: no animal pkl")
        continue
    
    animal_params_ild16[(batch_name, animal_id)] = {}
    
    for ABL in ABL_arr:
        cond_params = get_cond_fit_params(batch_name, animal_id_str, ABL, ILD_target)
        if cond_params is None:
            print(f"  Missing cond-fit for {batch_name}/{animal_id_str} ABL={ABL} ILD={ILD_target}")
            continue
        
        animal_params_ild16[(batch_name, animal_id)][ABL] = {
            'gamma': cond_params['gamma'],
            'omega': cond_params['omega'],
            **animal_data['avg_params'],
            **animal_data['abort_params']
        }

# Print summary
n_complete = sum(1 for v in animal_params_ild16.values() if len(v) == 3)
print(f"\nLoaded params for {len(animal_params_ild16)} animals, {n_complete} have all 3 ABLs")

# Show a sample
sample_key = list(animal_params_ild16.keys())[0]
print(f"\nSample params for {sample_key}:")
for abl, params in animal_params_ild16[sample_key].items():
    print(f"  ABL={abl}: gamma={params['gamma']:.3f}, omega={params['omega']:.3f}, w={params['w']:.3f}")

# %%
# =============================================================================
# STEP 2 (V2): Compute theoretical CDF using simpler formula:
#              CDF = c_A + c_E - c_A * c_E
#              where c_E uses CDF_E_minus_small_t_NORM_omega_gamma_with_w_fn
#              Time is wrt fixation: t_wrt_fix = t_wrt_stim + t_stim
# =============================================================================
from led_off_gamma_omega_pdf_utils import cum_A_t_fn, CDF_E_minus_small_t_NORM_omega_gamma_with_w_fn

K_max = 10
N_theory = int(1e3)

# Time points wrt stimulus onset
t_pts_wrt_stim = np.arange(-1, 1, 0.01)

# Store CDFs: {ABL: list of CDF arrays (one per animal)}
cdf_by_abl = {20: [], 40: [], 60: []}

n_animals = len(animal_params_ild16)
for animal_num, ((batch_name, animal_id), abl_dict) in enumerate(animal_params_ild16.items(), 1):
    animal_id_str = str(animal_id)
    
    # Skip if no ABLs available for this animal
    if len(abl_dict) == 0:
        print(f"  [{animal_num}/{n_animals}] Skipping {batch_name}/{animal_id_str}: no ABLs for ILD=16")
        continue
    
    print(f"  [{animal_num}/{n_animals}] Processing {batch_name}/{animal_id_str}...", end=' ', flush=True)
    
    # Load animal's data for t_stim sampling
    csv_file = os.path.join(batch_dir, f'batch_{batch_name}_valid_and_aborts.csv')
    df = pd.read_csv(csv_file)
    df_animal = df[df['animal'] == int(animal_id)]
    
    # Sample t_stim from this animal's intended_fix distribution
    t_stim_samples = df_animal['intended_fix'].sample(N_theory, replace=True).values
    
    # Get abort params (same for all ABLs) - use first available ABL
    sample_params = list(abl_dict.values())[0]
    
    V_A = sample_params['V_A']
    theta_A = sample_params['theta_A']
    t_A_aff = sample_params['t_A_aff']
    
    # Compute CDF for each ABL
    for ABL in ABL_arr:
        if ABL not in abl_dict:
            continue
        
        params = abl_dict[ABL]
        gamma = params['gamma']
        omega = params['omega']
        w = params['w']
        t_E_aff = params['t_E_aff']
        
        # Compute CDF samples for each t_stim
        cdf_samples = np.zeros((N_theory, len(t_pts_wrt_stim)))
        
        for idx, t_stim in enumerate(t_stim_samples):
            # Convert to time wrt fixation
            t_pts_wrt_fix = t_pts_wrt_stim + t_stim
            
            # Compute CDF at each time point
            for t_idx, t_fix in enumerate(t_pts_wrt_fix):
                # Abort CDF
                c_A = cum_A_t_fn(t_fix - t_A_aff, V_A, theta_A)
                
                # Evidence CDF (both bounds, with w)
                t_evidence = t_pts_wrt_stim[t_idx] - t_E_aff  # time for evidence accumulator
                c_E_up = CDF_E_minus_small_t_NORM_omega_gamma_with_w_fn(t_evidence, gamma, omega, 1, w, K_max)
                c_E_down = CDF_E_minus_small_t_NORM_omega_gamma_with_w_fn(t_evidence, gamma, omega, -1, w, K_max)
                c_E = c_E_up + c_E_down
                
                # Combined CDF
                cdf_samples[idx, t_idx] = c_A + c_E - c_A * c_E
        
        # Average CDF across t_stim samples
        cdf_mean = np.mean(cdf_samples, axis=0)
        
        # Extract CDF for t_wrt_stim in [0, 1]
        mask_0_1 = (t_pts_wrt_stim >= 0) & (t_pts_wrt_stim <= 1)
        cdf_0_1 = cdf_mean[mask_0_1]
        
        # Normalize: subtract CDF(0) and rescale so CDF goes from 0 to 1
        # This removes the abort probability that occurred before stimulus
        cdf_at_0 = cdf_0_1[0]
        cdf_at_end = cdf_0_1[-1]
        if cdf_at_end > cdf_at_0:
            cdf_0_1 = (cdf_0_1 - cdf_at_0) / (cdf_at_end - cdf_at_0)
        
        cdf_by_abl[ABL].append(cdf_0_1)
    
    print(f"done ({len(abl_dict)} ABLs)")

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
ax.set_title('Theoretical CDF at |ILD| = 16 dB (V2 - simpler formula)', fontsize=14)
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig('cond_fit_theoretical_cdf_ild16_v2.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nCDF computation complete. Saved plot to cond_fit_theoretical_cdf_ild16_v2.png")

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
# ax.set_title('Theoretical vs Empirical RT Quantiles at |ILD| = 16 dB (V2)', fontsize=14)
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
plt.savefig('cond_fit_quantiles_theory_vs_data_ild16_v2.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nQuantile comparison saved to cond_fit_quantiles_theory_vs_data_ild16_v2.png")
# %%
