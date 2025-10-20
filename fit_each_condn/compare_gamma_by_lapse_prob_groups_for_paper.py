# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from collections import defaultdict

# %%
# --- Get Batch-Animal Pairs ---
DESIRED_BATCHES = ['SD', 'LED34', 'LED6', 'LED8', 'LED7', 'LED34_even']
batch_dir = '/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/batch_csvs'
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
# --- Load Lapse Parameters ---
lapse_pkl_path = '/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/lapse_parameters_all_animals.pkl'
with open(lapse_pkl_path, 'rb') as f:
    lapse_params = pickle.load(f)

print(f"\nLoaded lapse parameters for {len(lapse_params)} animals")

# %%
# --- Calculate Vanilla Lapse Probability for Each Animal ---
animal_avg_lapse = {}
for batch, animal in batch_animal_pairs:
    key = (batch, int(animal))
    if key in lapse_params:
        data = lapse_params[key]
        vanilla_lp = data['vanilla_lapse']['lapse_prob']
        
        if vanilla_lp is not None:
            animal_avg_lapse[(batch, str(animal))] = vanilla_lp
        else:
            print(f"Warning: Missing vanilla lapse data for {batch}_{animal}")
    else:
        print(f"Warning: No lapse parameters found for {batch}_{animal}")

print(f"\nCalculated vanilla lapse probability for {len(animal_avg_lapse)} animals")

# %%
# --- Separate Animals into Two Groups ---
LAPSE_THRESHOLD = 0.015  # 1.5%

low_lapse_animals = []
high_lapse_animals = []

for (batch, animal), avg_lapse in animal_avg_lapse.items():
    if avg_lapse < LAPSE_THRESHOLD:
        low_lapse_animals.append((batch, animal))
    else:
        high_lapse_animals.append((batch, animal))

print(f"\n--- Grouping by Lapse Probability (threshold = {LAPSE_THRESHOLD*100:.1f}%) ---")
print(f"Low lapse animals (< {LAPSE_THRESHOLD*100:.1f}%): {len(low_lapse_animals)}")
print(f"High lapse animals (>= {LAPSE_THRESHOLD*100:.1f}%): {len(high_lapse_animals)}")

# Print animals in each group
print("\nLow lapse animals:")
for batch, animal in sorted(low_lapse_animals):
    print(f"  {batch}_{animal}: {animal_avg_lapse[(batch, animal)]*100:.2f}%")

print("\nHigh lapse animals:")
for batch, animal in sorted(high_lapse_animals):
    print(f"  {batch}_{animal}: {animal_avg_lapse[(batch, animal)]*100:.2f}%")

# %%
# --- Function to Get Gamma/Omega from Condition-by-Condition Fit ---
def get_param_means_by_ABL_ILD(batch_name, animal_id, ABLs_to_fit, ILDs_to_fit, param_names=None):
    """
    Returns a dictionary with keys (ABL, ILD) and values as dicts of mean parameter values.
    Only includes (ABL, ILD) combinations for which the corresponding pickle file exists.
    param_names: list of parameter names in the order of columns in vp_samples, default uses [gamma, omega]
    """
    if param_names is None:
        param_names = ['gamma', 'omega']
    
    param_dict = {}
    for ABL in ABLs_to_fit:
        for ILD in ILDs_to_fit:
            pkl_folder = '/home/rlab/raghavendra/ddm_data/fit_each_condn/each_animal_cond_fit_gama_omega_pkl_files'
            pkl_file = os.path.join(pkl_folder, f'vbmc_cond_by_cond_{batch_name}_{animal_id}_{ABL}_ILD_{ILD}_FIX_t_E_w_del_go_same_as_parametric.pkl')
            if not os.path.exists(pkl_file):
                continue
            with open(pkl_file, 'rb') as f:
                vp = pickle.load(f)
            vp = vp.vp
            vp_samples = vp.sample(int(1e5))[0]
            means = {name: float(np.mean(vp_samples[:, i])) for i, name in enumerate(param_names)}
            param_dict[(ABL, ILD)] = means
    return param_dict

# %%
# --- Extract Gamma for Both Groups ---
all_ABL = [20, 40, 60]
all_ILD_sorted = np.sort([1, -1, 2, -2, 4, -4, 8, -8, 16, -16])

# Initialize storage for low lapse group
gamma_low_lapse = {
    '20': np.full((len(low_lapse_animals), len(all_ILD_sorted)), np.nan), 
    '40': np.full((len(low_lapse_animals), len(all_ILD_sorted)), np.nan), 
    '60': np.full((len(low_lapse_animals), len(all_ILD_sorted)), np.nan)
}

# Initialize storage for high lapse group
gamma_high_lapse = {
    '20': np.full((len(high_lapse_animals), len(all_ILD_sorted)), np.nan), 
    '40': np.full((len(high_lapse_animals), len(all_ILD_sorted)), np.nan), 
    '60': np.full((len(high_lapse_animals), len(all_ILD_sorted)), np.nan)
}

# Fill in gamma for low lapse animals
print("\n--- Processing Low Lapse Animals ---")
for animal_idx, (batch_name, animal_id) in enumerate(low_lapse_animals):
    print(f'Processing {batch_name}_{animal_id} (lapse: {animal_avg_lapse[(batch_name, animal_id)]*100:.2f}%)')
    param_dict = get_param_means_by_ABL_ILD(batch_name, animal_id, all_ABL, all_ILD_sorted)
    for ABL in all_ABL:
        for ild_idx, ILD in enumerate(all_ILD_sorted):
            if (ABL, ILD) in param_dict:
                gamma_low_lapse[str(ABL)][animal_idx, ild_idx] = param_dict[(ABL, ILD)]['gamma']

# Fill in gamma for high lapse animals
print("\n--- Processing High Lapse Animals ---")
for animal_idx, (batch_name, animal_id) in enumerate(high_lapse_animals):
    print(f'Processing {batch_name}_{animal_id} (lapse: {animal_avg_lapse[(batch_name, animal_id)]*100:.2f}%)')
    param_dict = get_param_means_by_ABL_ILD(batch_name, animal_id, all_ABL, all_ILD_sorted)
    for ABL in all_ABL:
        for ild_idx, ILD in enumerate(all_ILD_sorted):
            if (ABL, ILD) in param_dict:
                gamma_high_lapse[str(ABL)][animal_idx, ild_idx] = param_dict[(ABL, ILD)]['gamma']

# %%
# --- Plot Gamma Comparison Between Low and High Lapse Groups ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for abl_idx, ABL in enumerate(all_ABL):
    ax_gamma = axes[abl_idx]
    
    # --- Gamma Plot ---
    # Low lapse group
    mean_gamma_low = np.nanmean(gamma_low_lapse[str(ABL)], axis=0)
    sem_gamma_low = np.nanstd(gamma_low_lapse[str(ABL)], axis=0) / np.sqrt(np.sum(~np.isnan(gamma_low_lapse[str(ABL)]), axis=0))
    
    # High lapse group
    mean_gamma_high = np.nanmean(gamma_high_lapse[str(ABL)], axis=0)
    sem_gamma_high = np.nanstd(gamma_high_lapse[str(ABL)], axis=0) / np.sqrt(np.sum(~np.isnan(gamma_high_lapse[str(ABL)]), axis=0))
    
    ax_gamma.errorbar(all_ILD_sorted, mean_gamma_low, yerr=sem_gamma_low, fmt='o-', 
                     color='blue', label=f'Low lapse (n={len(low_lapse_animals)})', capsize=0, alpha=0.7)
    ax_gamma.errorbar(all_ILD_sorted, mean_gamma_high, yerr=sem_gamma_high, fmt='s-', 
                     color='red', label=f'High lapse (n={len(high_lapse_animals)})', capsize=0, alpha=0.7)
    
    ax_gamma.set_title(f'Gamma at ABL={ABL}', fontsize=12, fontweight='bold')
    ax_gamma.set_xlabel('ILD', fontsize=11)
    ax_gamma.set_ylabel('Gamma', fontsize=11)
    ax_gamma.legend(fontsize=9)
    # ax_gamma.grid(True, alpha=0.3)

plt.suptitle(f'Gamma Comparison: Low vs High Vanilla Lapse Probability (threshold = {LAPSE_THRESHOLD*100:.1f}%)', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

# Save figure
output_dir = '/home/rlab/raghavendra/ddm_data/fit_each_condn'
output_file = os.path.join(output_dir, 'gamma_omega_by_lapse_prob_groups.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {output_file}")

plt.show()

# %%
# --- Print Summary Statistics ---
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

for ABL in all_ABL:
    print(f"\n--- ABL = {ABL} ---")
    
    # Gamma statistics
    gamma_low_mean = np.nanmean(gamma_low_lapse[str(ABL)])
    gamma_high_mean = np.nanmean(gamma_high_lapse[str(ABL)])
    print(f"Gamma (Low lapse):  mean = {gamma_low_mean:.4f}")
    print(f"Gamma (High lapse): mean = {gamma_high_mean:.4f}")
    print(f"Gamma difference: {gamma_high_mean - gamma_low_mean:.4f}")

print("\n" + "="*60)

# %%
# --- Plot Gamma Separated by Log-Likelihood Comparison ---
print("\n" + "="*60)
print("GROUPING BY LOG-LIKELIHOOD COMPARISON")
print("="*60)

# Load lapse parameters with loglike_per_trial
lapse_params_loglike_pkl = '/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/lapse_parameters_all_animals.pkl'
with open(lapse_params_loglike_pkl, 'rb') as f:
    lapse_params_loglike = pickle.load(f)

print(f"\nLoaded log-likelihood data for {len(lapse_params_loglike)} animals")

# Separate animals based on vanilla_lapse vs norm loglike comparison
vanilla_better_animals = []  # vanilla_lapse > norm
norm_better_animals = []      # vanilla_lapse < norm

for batch, animal in batch_animal_pairs:
    key = (batch, int(animal))
    if key in lapse_params_loglike:
        data = lapse_params_loglike[key]
        vanilla_lapse_ll = data['vanilla_lapse'].get('loglike_per_trial')
        norm_ll = data['norm'].get('loglike_per_trial')
        
        if vanilla_lapse_ll is not None and norm_ll is not None:
            if vanilla_lapse_ll > norm_ll:
                vanilla_better_animals.append((batch, str(animal)))
            else:
                norm_better_animals.append((batch, str(animal)))
        else:
            print(f"Warning: Missing loglike data for {batch}_{animal}")
    else:
        print(f"Warning: No loglike data found for {batch}_{animal}")

print(f"\nAnimals where Vanilla+Lapse > Norm: {len(vanilla_better_animals)}")
print(f"Animals where Vanilla+Lapse < Norm: {len(norm_better_animals)}")

# Print animals in each group with their loglike differences
print("\nVanilla+Lapse > Norm loglike:")
for batch, animal in sorted(vanilla_better_animals):
    key = (batch, int(animal))
    data = lapse_params_loglike[key]
    diff = data['vanilla_lapse']['loglike_per_trial'] - data['norm']['loglike_per_trial']
    print(f"  {batch}_{animal}: Δloglike/trial = +{diff:.6f}")

print("\nVanilla+Lapse < Norm loglike:")
for batch, animal in sorted(norm_better_animals):
    key = (batch, int(animal))
    data = lapse_params_loglike[key]
    diff = data['vanilla_lapse']['loglike_per_trial'] - data['norm']['loglike_per_trial']
    print(f"  {batch}_{animal}: Δloglike/trial = {diff:.6f}")

# %%
# --- Extract Gamma for Both LogLike Groups ---
# Initialize storage for vanilla_better group
gamma_vanilla_better = {
    '20': np.full((len(vanilla_better_animals), len(all_ILD_sorted)), np.nan), 
    '40': np.full((len(vanilla_better_animals), len(all_ILD_sorted)), np.nan), 
    '60': np.full((len(vanilla_better_animals), len(all_ILD_sorted)), np.nan)
}

# Initialize storage for norm_better group
gamma_norm_better = {
    '20': np.full((len(norm_better_animals), len(all_ILD_sorted)), np.nan), 
    '40': np.full((len(norm_better_animals), len(all_ILD_sorted)), np.nan), 
    '60': np.full((len(norm_better_animals), len(all_ILD_sorted)), np.nan)
}

# Fill in gamma for vanilla_better animals
print("\n--- Processing Vanilla+Lapse > Norm Animals ---")
for animal_idx, (batch_name, animal_id) in enumerate(vanilla_better_animals):
    key = (batch_name, int(animal_id))
    data = lapse_params_loglike[key]
    diff = data['vanilla_lapse']['loglike_per_trial'] - data['norm']['loglike_per_trial']
    print(f'Processing {batch_name}_{animal_id} (Δloglike/trial: +{diff:.6f})')
    param_dict = get_param_means_by_ABL_ILD(batch_name, animal_id, all_ABL, all_ILD_sorted)
    for ABL in all_ABL:
        for ild_idx, ILD in enumerate(all_ILD_sorted):
            if (ABL, ILD) in param_dict:
                gamma_vanilla_better[str(ABL)][animal_idx, ild_idx] = param_dict[(ABL, ILD)]['gamma']

# Fill in gamma for norm_better animals
print("\n--- Processing Vanilla+Lapse < Norm Animals ---")
for animal_idx, (batch_name, animal_id) in enumerate(norm_better_animals):
    key = (batch_name, int(animal_id))
    data = lapse_params_loglike[key]
    diff = data['vanilla_lapse']['loglike_per_trial'] - data['norm']['loglike_per_trial']
    print(f'Processing {batch_name}_{animal_id} (Δloglike/trial: {diff:.6f})')
    param_dict = get_param_means_by_ABL_ILD(batch_name, animal_id, all_ABL, all_ILD_sorted)
    for ABL in all_ABL:
        for ild_idx, ILD in enumerate(all_ILD_sorted):
            if (ABL, ILD) in param_dict:
                gamma_norm_better[str(ABL)][animal_idx, ild_idx] = param_dict[(ABL, ILD)]['gamma']

# %%
# --- Plot Gamma Comparison Between LogLike Groups ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for abl_idx, ABL in enumerate(all_ABL):
    ax_gamma = axes[abl_idx]
    
    # --- Gamma Plot ---
    # Vanilla+Lapse better group (RED since vanilla+lapse wins)
    mean_gamma_vanilla_better = np.nanmean(gamma_vanilla_better[str(ABL)], axis=0)
    sem_gamma_vanilla_better = np.nanstd(gamma_vanilla_better[str(ABL)], axis=0) / np.sqrt(np.sum(~np.isnan(gamma_vanilla_better[str(ABL)]), axis=0))
    
    # Norm better group (BLUE since norm wins, vanilla+lapse loses)
    mean_gamma_norm_better = np.nanmean(gamma_norm_better[str(ABL)], axis=0)
    sem_gamma_norm_better = np.nanstd(gamma_norm_better[str(ABL)], axis=0) / np.sqrt(np.sum(~np.isnan(gamma_norm_better[str(ABL)]), axis=0))
    
    ax_gamma.errorbar(all_ILD_sorted, mean_gamma_vanilla_better, yerr=sem_gamma_vanilla_better, fmt='o-', 
                     color='red', label=f'V+L > Norm (n={len(vanilla_better_animals)})', capsize=0, alpha=0.7)
    ax_gamma.errorbar(all_ILD_sorted, mean_gamma_norm_better, yerr=sem_gamma_norm_better, fmt='s-', 
                     color='blue', label=f'V+L < Norm (n={len(norm_better_animals)})', capsize=0, alpha=0.7)
    
    ax_gamma.set_title(f'Gamma at ABL={ABL}', fontsize=12, fontweight='bold')
    ax_gamma.set_xlabel('ILD', fontsize=11)
    ax_gamma.set_ylabel('Gamma', fontsize=11)
    ax_gamma.legend(fontsize=9)

plt.suptitle(f'Gamma Comparison: Grouped by Vanilla+Lapse vs Norm LogLike Per Trial', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

# Save figure
output_file_loglike = os.path.join(output_dir, 'gamma_by_loglike_comparison_groups.png')
plt.savefig(output_file_loglike, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {output_file_loglike}")

plt.show()

# %%
# --- Print Summary Statistics for LogLike Groups ---
print("\n" + "="*60)
print("SUMMARY STATISTICS - LOGLIKE GROUPS")
print("="*60)

for ABL in all_ABL:
    print(f"\n--- ABL = {ABL} ---")
    
    # Gamma statistics
    gamma_vanilla_better_mean = np.nanmean(gamma_vanilla_better[str(ABL)])
    gamma_norm_better_mean = np.nanmean(gamma_norm_better[str(ABL)])
    print(f"Gamma (V+L > Norm):  mean = {gamma_vanilla_better_mean:.4f}")
    print(f"Gamma (V+L < Norm):  mean = {gamma_norm_better_mean:.4f}")
    print(f"Gamma difference: {gamma_vanilla_better_mean - gamma_norm_better_mean:.4f}")

print("\n" + "="*60)

# %%
