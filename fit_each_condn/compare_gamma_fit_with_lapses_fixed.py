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
def get_gamma_means_by_ABL_ILD_LAPSES(batch_name, animal_id, ABLs_to_fit, ILDs_to_fit):
    """
    Returns a dictionary with keys (ABL, ILD) and values as mean gamma values.
    Only includes (ABL, ILD) combinations for which the corresponding pickle file exists.
    Reads from LAPSES fit pickle files.
    """
    gamma_dict = {}
    for ABL in ABLs_to_fit:
        for ILD in ILDs_to_fit:
            pkl_folder = '/home/rlab/raghavendra/ddm_data/fit_each_condn/each_animal_cond_fit_gama_omega_pkl_files_LAPSES'
            pkl_file = os.path.join(pkl_folder, f'vbmc_cond_by_cond_{batch_name}_{animal_id}_{ABL}_ILD_{ILD}_FIX_t_E_w_del_go_same_as_parametric_LAPSES.pkl')
            if not os.path.exists(pkl_file):
                print(f'{pkl_file} does not exist')
                continue
            with open(pkl_file, 'rb') as f:
                vp = pickle.load(f)
            vp = vp.vp
            vp_samples = vp.sample(int(1e5))[0]
            # First column is gamma, second is omega
            gamma_mean = float(np.mean(vp_samples[:, 0]))
            gamma_dict[(ABL, ILD)] = gamma_mean
    return gamma_dict


# %%
# Collect gamma values from LAPSES fits for all animals
all_ABL = [20, 40, 60]
all_ILD_sorted = np.sort([1, -1, 2, -2, 4, -4, 8, -8, 16, -16])

gamma_lapse_fit_all_animals = {
    '20': np.full((len(batch_animal_pairs), len(all_ILD_sorted)), np.nan), 
    '40': np.full((len(batch_animal_pairs), len(all_ILD_sorted)), np.nan), 
    '60': np.full((len(batch_animal_pairs), len(all_ILD_sorted)), np.nan)
}
    
for animal_idx, (batch_name, animal_id) in enumerate(batch_animal_pairs):
    print('==========================================')
    print(f'Batch: {batch_name}, Animal: {animal_id}')
    print('==========================================')

    gamma_dict = get_gamma_means_by_ABL_ILD_LAPSES(batch_name, animal_id, all_ABL, all_ILD_sorted)
    
    for ABL in all_ABL:
        for ild_idx, ILD in enumerate(all_ILD_sorted):
            if (ABL, ILD) in gamma_dict:
                gamma_lapse_fit_all_animals[str(ABL)][animal_idx, ild_idx] = gamma_dict[(ABL, ILD)]

# %%
# Plot average gamma across all animals from LAPSES fits
fig, ax = plt.subplots(1, 1, figsize=(7, 5))

for ABL in all_ABL:
    # Calculate mean and standard error of mean
    mean_gamma = np.nanmean(gamma_lapse_fit_all_animals[str(ABL)], axis=0)
    sem_gamma = np.nanstd(gamma_lapse_fit_all_animals[str(ABL)], axis=0) / np.sqrt(np.sum(~np.isnan(gamma_lapse_fit_all_animals[str(ABL)]), axis=0))
    
    # Create scatter plots with error bars
    ax.errorbar(all_ILD_sorted, mean_gamma, yerr=sem_gamma, fmt='o', 
                color=f'tab:{["blue", "orange", "green"][ABL//20-1]}', 
                label=f'ABL={ABL}', capsize=0)

ax.set_title('Gamma from LAPSES Fits (Average across animals)', fontsize=14)
ax.set_xlabel('ILD', fontsize=12)
ax.set_ylabel('Gamma', fontsize=12)
ax.legend()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig('gamma_lapse_fit_average.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Save data to pickle for later use
gamma_lapse_plot_data = {
    'all_ABL': all_ABL,
    'gamma_lapse_fit_all_animals': gamma_lapse_fit_all_animals,
    'all_ILD_sorted': all_ILD_sorted,
    'batch_animal_pairs': batch_animal_pairs
}
with open('gamma_lapse_fit_data.pkl', 'wb') as f:
    pickle.dump(gamma_lapse_plot_data, f)

print('Saved gamma lapse fit data to gamma_lapse_fit_data.pkl')

# %%
# Load lapse parameters to split animals by lapse rate
lapse_params_pkl_path = '/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/lapse_parameters_all_animals.pkl'
with open(lapse_params_pkl_path, 'rb') as f:
    lapse_params_all = pickle.load(f)

# For each animal, compute average lapse_prob from vanilla_lapse and norm_lapse
animal_lapse_rates = {}
for animal_idx, (batch_name, animal_id) in enumerate(batch_animal_pairs):
    key = (batch_name, int(animal_id))
    if key in lapse_params_all:
        data = lapse_params_all[key]
        vanilla_lapse_prob = data['vanilla_lapse']['lapse_prob']
        norm_lapse_prob = data['norm_lapse']['lapse_prob']
        
        if vanilla_lapse_prob is not None and norm_lapse_prob is not None:
            avg_lapse_prob = (vanilla_lapse_prob + norm_lapse_prob) / 2
            animal_lapse_rates[animal_idx] = avg_lapse_prob
        else:
            print(f"Warning: None lapse_prob for {batch_name}_{animal_id}")
    else:
        print(f"Warning: {batch_name}_{animal_id} not found in lapse_params_all")

# Split animals by lapse rate threshold
lapse_threshold = 0.015  # 1.5%
low_lapse_animals = [idx for idx, lapse in animal_lapse_rates.items() if lapse < lapse_threshold]
high_lapse_animals = [idx for idx, lapse in animal_lapse_rates.items() if lapse >= lapse_threshold]

print(f"\nAnimals with lapse rate < {lapse_threshold*100:.1f}%: {len(low_lapse_animals)}")
print(f"Animals with lapse rate >= {lapse_threshold*100:.1f}%: {len(high_lapse_animals)}")

# %%
# Plot 1x3: Gamma split by lapse rate for each ABL
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, ABL in enumerate(all_ABL):
    ax = axes[i]
    
    # Low lapse rate animals (< 1.5%)
    if len(low_lapse_animals) > 0:
        gamma_low_lapse = gamma_lapse_fit_all_animals[str(ABL)][low_lapse_animals, :]
        mean_gamma_low = np.nanmean(gamma_low_lapse, axis=0)
        sem_gamma_low = np.nanstd(gamma_low_lapse, axis=0) / np.sqrt(np.sum(~np.isnan(gamma_low_lapse), axis=0))
        
        ax.errorbar(all_ILD_sorted, mean_gamma_low, yerr=sem_gamma_low, fmt='o-', 
                    color='blue', label=f'Lapse < {lapse_threshold*100:.1f}% (n={len(low_lapse_animals)})', 
                    capsize=0, alpha=0.7)
    
    # High lapse rate animals (>= 1.5%)
    if len(high_lapse_animals) > 0:
        gamma_high_lapse = gamma_lapse_fit_all_animals[str(ABL)][high_lapse_animals, :]
        mean_gamma_high = np.nanmean(gamma_high_lapse, axis=0)
        sem_gamma_high = np.nanstd(gamma_high_lapse, axis=0) / np.sqrt(np.sum(~np.isnan(gamma_high_lapse), axis=0))
        
        ax.errorbar(all_ILD_sorted, mean_gamma_high, yerr=sem_gamma_high, fmt='o-', 
                    color='red', label=f'Lapse >= {lapse_threshold*100:.1f}% (n={len(high_lapse_animals)})', 
                    capsize=0, alpha=0.7)
    
    ax.set_title(f'ABL = {ABL}', fontsize=14)
    ax.set_xlabel('ILD', fontsize=12)
    ax.set_ylabel('Gamma', fontsize=12)
    ax.legend(fontsize=10)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(True, alpha=0.3)

plt.suptitle('Gamma by Lapse Rate (LAPSES Fits)', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('gamma_by_lapse_rate.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Split animals by which model has better log-likelihood: vanilla_lapse vs norm
animal_model_preference = {}
for animal_idx, (batch_name, animal_id) in enumerate(batch_animal_pairs):
    key = (batch_name, int(animal_id))
    if key in lapse_params_all:
        data = lapse_params_all[key]
        vanilla_lapse_ll = data['vanilla_lapse']['loglike_per_trial']
        norm_ll = data['norm']['loglike_per_trial']
        
        if vanilla_lapse_ll is not None and norm_ll is not None:
            # Store which model is better: 'norm' if norm > vanilla_lapse, else 'vanilla_lapse'
            if norm_ll > vanilla_lapse_ll:
                animal_model_preference[animal_idx] = 'norm_better'
            else:
                animal_model_preference[animal_idx] = 'vanilla_lapse_better'
        else:
            print(f"Warning: None loglike for {batch_name}_{animal_id}")
    else:
        print(f"Warning: {batch_name}_{animal_id} not found in lapse_params_all")

# Split animals by model preference
norm_better_animals = [idx for idx, pref in animal_model_preference.items() if pref == 'norm_better']
vanilla_lapse_better_animals = [idx for idx, pref in animal_model_preference.items() if pref == 'vanilla_lapse_better']

print(f"\nAnimals where Norm loglike > Vanilla+Lapse loglike: {len(norm_better_animals)}")
print(f"Animals where Vanilla+Lapse loglike > Norm loglike: {len(vanilla_lapse_better_animals)}")

# %%
# Plot 1x3: Gamma split by model preference for each ABL
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, ABL in enumerate(all_ABL):
    ax = axes[i]
    
    # Norm better (blue) - vanilla_lapse loglike < norm loglike
    if len(norm_better_animals) > 0:
        gamma_norm_better = gamma_lapse_fit_all_animals[str(ABL)][norm_better_animals, :]
        mean_gamma_norm = np.nanmean(gamma_norm_better, axis=0)
        sem_gamma_norm = np.nanstd(gamma_norm_better, axis=0) / np.sqrt(np.sum(~np.isnan(gamma_norm_better), axis=0))
        
        ax.errorbar(all_ILD_sorted, mean_gamma_norm, yerr=sem_gamma_norm, fmt='o-', 
                    color='blue', label=f'Norm better (n={len(norm_better_animals)})', 
                    capsize=0, alpha=0.7)
    
    # Vanilla+Lapse better (red) - vanilla_lapse loglike > norm loglike
    if len(vanilla_lapse_better_animals) > 0:
        gamma_vanilla_better = gamma_lapse_fit_all_animals[str(ABL)][vanilla_lapse_better_animals, :]
        mean_gamma_vanilla = np.nanmean(gamma_vanilla_better, axis=0)
        sem_gamma_vanilla = np.nanstd(gamma_vanilla_better, axis=0) / np.sqrt(np.sum(~np.isnan(gamma_vanilla_better), axis=0))
        
        ax.errorbar(all_ILD_sorted, mean_gamma_vanilla, yerr=sem_gamma_vanilla, fmt='o-', 
                    color='red', label=f'Vanilla+Lapse better (n={len(vanilla_lapse_better_animals)})', 
                    capsize=0, alpha=0.7)
    
    ax.set_title(f'ABL = {ABL}', fontsize=14)
    ax.set_xlabel('ILD', fontsize=12)
    ax.set_ylabel('Gamma', fontsize=12)
    ax.legend(fontsize=10)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(True, alpha=0.3)

plt.suptitle('Gamma by Model Preference: Vanilla+Lapse vs Norm (LAPSES Fits)', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('gamma_by_model_preference.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Compare gamma: without lapses (blue) vs with lapses (red)

def get_gamma_means_by_ABL_ILD_NO_LAPSES(batch_name, animal_id, ABLs_to_fit, ILDs_to_fit):
    """
    Returns a dictionary with keys (ABL, ILD) and values as mean gamma values.
    Only includes (ABL, ILD) combinations for which the corresponding pickle file exists.
    Reads from NON-LAPSES fit pickle files.
    """
    gamma_dict = {}
    for ABL in ABLs_to_fit:
        for ILD in ILDs_to_fit:
            pkl_folder = '/home/rlab/raghavendra/ddm_data/fit_each_condn/each_animal_cond_fit_gama_omega_pkl_files'
            pkl_file = os.path.join(pkl_folder, f'vbmc_cond_by_cond_{batch_name}_{animal_id}_{ABL}_ILD_{ILD}_FIX_t_E_w_del_go_same_as_parametric.pkl')
            if not os.path.exists(pkl_file):
                print(f'{pkl_file} does not exist')
                continue
            with open(pkl_file, 'rb') as f:
                vp = pickle.load(f)
            vp = vp.vp
            vp_samples = vp.sample(int(1e5))[0]
            # First column is gamma, second is omega
            gamma_mean = float(np.mean(vp_samples[:, 0]))
            gamma_dict[(ABL, ILD)] = gamma_mean
    return gamma_dict

# Collect gamma values from NON-LAPSES fits for all animals
gamma_no_lapse_fit_all_animals = {
    '20': np.full((len(batch_animal_pairs), len(all_ILD_sorted)), np.nan), 
    '40': np.full((len(batch_animal_pairs), len(all_ILD_sorted)), np.nan), 
    '60': np.full((len(batch_animal_pairs), len(all_ILD_sorted)), np.nan)
}
    
for animal_idx, (batch_name, animal_id) in enumerate(batch_animal_pairs):
    print('==========================================')
    print(f'NO LAPSE: Batch: {batch_name}, Animal: {animal_id}')
    print('==========================================')

    gamma_dict = get_gamma_means_by_ABL_ILD_NO_LAPSES(batch_name, animal_id, all_ABL, all_ILD_sorted)
    
    for ABL in all_ABL:
        for ild_idx, ILD in enumerate(all_ILD_sorted):
            if (ABL, ILD) in gamma_dict:
                gamma_no_lapse_fit_all_animals[str(ABL)][animal_idx, ild_idx] = gamma_dict[(ABL, ILD)]

# %%
# Plot 1x3: Gamma comparison - without lapses (blue) vs with lapses (red)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, ABL in enumerate(all_ABL):
    ax = axes[i]
    
    # Without lapses (blue)
    mean_gamma_no_lapse = np.nanmean(gamma_no_lapse_fit_all_animals[str(ABL)], axis=0)
    sem_gamma_no_lapse = np.nanstd(gamma_no_lapse_fit_all_animals[str(ABL)], axis=0) / np.sqrt(np.sum(~np.isnan(gamma_no_lapse_fit_all_animals[str(ABL)]), axis=0))
    
    ax.errorbar(all_ILD_sorted, mean_gamma_no_lapse, yerr=sem_gamma_no_lapse, fmt='o-', 
                color='blue', label='Without lapses', capsize=0, alpha=0.7, linewidth=2, markersize=6)
    
    # With lapses (red)
    mean_gamma_lapse = np.nanmean(gamma_lapse_fit_all_animals[str(ABL)], axis=0)
    sem_gamma_lapse = np.nanstd(gamma_lapse_fit_all_animals[str(ABL)], axis=0) / np.sqrt(np.sum(~np.isnan(gamma_lapse_fit_all_animals[str(ABL)]), axis=0))
    
    ax.errorbar(all_ILD_sorted, mean_gamma_lapse, yerr=sem_gamma_lapse, fmt='s-', 
                color='red', label='With lapses', capsize=0, alpha=0.7, linewidth=2, markersize=6)
    
    ax.set_title(f'ABL = {ABL}', fontsize=14)
    ax.set_xlabel('ILD', fontsize=12)
    ax.set_ylabel('Gamma', fontsize=12)
    ax.legend(fontsize=10)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.grid(True, alpha=0.3)

plt.suptitle('Gamma: Without Lapses vs With Lapses', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('gamma_comparison_no_lapses_vs_lapses.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Plot 1x3: Gamma comparison for LOW lapse rate animals (< 1.5%)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, ABL in enumerate(all_ABL):
    ax = axes[i]
    
    # Without lapses (blue) - only low lapse animals
    if len(low_lapse_animals) > 0:
        gamma_no_lapse_low = gamma_no_lapse_fit_all_animals[str(ABL)][low_lapse_animals, :]
        mean_gamma_no_lapse = np.nanmean(gamma_no_lapse_low, axis=0)
        sem_gamma_no_lapse = np.nanstd(gamma_no_lapse_low, axis=0) / np.sqrt(np.sum(~np.isnan(gamma_no_lapse_low), axis=0))
        
        ax.errorbar(all_ILD_sorted, mean_gamma_no_lapse, yerr=sem_gamma_no_lapse, fmt='o-', 
                    color='blue', label='Without lapses', capsize=0, alpha=0.7, linewidth=2, markersize=6)
    
    # With lapses (red) - only low lapse animals
    if len(low_lapse_animals) > 0:
        gamma_lapse_low = gamma_lapse_fit_all_animals[str(ABL)][low_lapse_animals, :]
        mean_gamma_lapse = np.nanmean(gamma_lapse_low, axis=0)
        sem_gamma_lapse = np.nanstd(gamma_lapse_low, axis=0) / np.sqrt(np.sum(~np.isnan(gamma_lapse_low), axis=0))
        
        ax.errorbar(all_ILD_sorted, mean_gamma_lapse, yerr=sem_gamma_lapse, fmt='s-', 
                    color='red', label='With lapses', capsize=0, alpha=0.7, linewidth=2, markersize=6)
    
    ax.set_title(f'ABL = {ABL}', fontsize=14)
    ax.set_xlabel('ILD', fontsize=12)
    ax.set_ylabel('Gamma', fontsize=12)
    ax.legend(fontsize=10)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

plt.suptitle(f'Gamma: Without vs With Lapses (Low lapse rate < 1.5%, n={len(low_lapse_animals)})', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('gamma_comparison_low_lapse_rate.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Plot 1x3: Gamma comparison for HIGH lapse rate animals (>= 1.5%)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, ABL in enumerate(all_ABL):
    ax = axes[i]
    
    # Without lapses (blue) - only high lapse animals
    if len(high_lapse_animals) > 0:
        gamma_no_lapse_high = gamma_no_lapse_fit_all_animals[str(ABL)][high_lapse_animals, :]
        mean_gamma_no_lapse = np.nanmean(gamma_no_lapse_high, axis=0)
        sem_gamma_no_lapse = np.nanstd(gamma_no_lapse_high, axis=0) / np.sqrt(np.sum(~np.isnan(gamma_no_lapse_high), axis=0))
        
        ax.errorbar(all_ILD_sorted, mean_gamma_no_lapse, yerr=sem_gamma_no_lapse, fmt='o-', 
                    color='blue', label='Without lapses', capsize=0, alpha=0.7, linewidth=2, markersize=6)
    
    # With lapses (red) - only high lapse animals
    if len(high_lapse_animals) > 0:
        gamma_lapse_high = gamma_lapse_fit_all_animals[str(ABL)][high_lapse_animals, :]
        mean_gamma_lapse = np.nanmean(gamma_lapse_high, axis=0)
        sem_gamma_lapse = np.nanstd(gamma_lapse_high, axis=0) / np.sqrt(np.sum(~np.isnan(gamma_lapse_high), axis=0))
        
        ax.errorbar(all_ILD_sorted, mean_gamma_lapse, yerr=sem_gamma_lapse, fmt='s-', 
                    color='red', label='With lapses', capsize=0, alpha=0.7, linewidth=2, markersize=6)
    
    ax.set_title(f'ABL = {ABL}', fontsize=14)
    ax.set_xlabel('ILD', fontsize=12)
    ax.set_ylabel('Gamma', fontsize=12)
    ax.legend(fontsize=10)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

plt.suptitle(f'Gamma: Without vs With Lapses (High lapse rate >= 1.5%, n={len(high_lapse_animals)})', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('gamma_comparison_high_lapse_rate.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Scatter plot: Lapse rate vs Gamma for ILD=16 and ILD=-16
# One subplot per ABL, all animals plotted individually

# Find indices for ILD=16 and ILD=-16
ild_16_idx = np.where(all_ILD_sorted == 16)[0][0]
ild_minus_16_idx = np.where(all_ILD_sorted == -16)[0][0]

# Prepare data for scatter plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, ABL in enumerate(all_ABL):
    ax = axes[i]
    
    # Collect data for all animals
    lapse_rates = []
    gamma_ild_16 = []
    gamma_ild_minus_16 = []
    
    for animal_idx, (batch_name, animal_id) in enumerate(batch_animal_pairs):
        # Get lapse rate for this animal
        if animal_idx in animal_lapse_rates:
            lapse_rate = animal_lapse_rates[animal_idx]
            
            # Get gamma at ILD=16 and ILD=-16 for this ABL
            g_16 = gamma_lapse_fit_all_animals[str(ABL)][animal_idx, ild_16_idx]
            g_minus_16 = gamma_lapse_fit_all_animals[str(ABL)][animal_idx, ild_minus_16_idx]
            
            # Only plot if both gamma values exist
            if not np.isnan(g_16) and not np.isnan(g_minus_16):
                lapse_rates.append(lapse_rate)
                gamma_ild_16.append(g_16)
                gamma_ild_minus_16.append(g_minus_16)
    
    # Scatter plot
    ax.scatter(np.array(lapse_rates)*100, gamma_ild_16, color='blue', alpha=0.6, s=50, label='ILD = 16')
    ax.scatter(np.array(lapse_rates)*100, np.abs(gamma_ild_minus_16), color='red', alpha=0.6, s=50, label='ILD = -16')
    
    # Add vertical line at lapse rate threshold
    ax.axvline(x=lapse_threshold*100, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    ax.set_title(f'ABL = {ABL}', fontsize=14)
    ax.set_xlabel('Lapse Rate % (vanilla_lapse)', fontsize=12)
    ax.set_ylabel('Gamma', fontsize=12)
    ax.legend(fontsize=10)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.grid(True, alpha=0.3)

plt.suptitle('Gamma vs Lapse Rate: Individual Animals at ILD=±16', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('gamma_vs_lapse_rate_scatter.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Scatter plot: Lapse rate vs Gamma for ILD=16 and ILD=-16 (POOLED across all ABLs)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# Collect data for all animals across all ABLs
lapse_rates_pooled = []
gamma_ild_16_pooled = []
gamma_ild_minus_16_pooled = []

for ABL in all_ABL:
    for animal_idx, (batch_name, animal_id) in enumerate(batch_animal_pairs):
        # Get lapse rate for this animal
        if animal_idx in animal_lapse_rates:
            lapse_rate = animal_lapse_rates[animal_idx]
            
            # Get gamma at ILD=16 and ILD=-16 for this ABL
            g_16 = gamma_lapse_fit_all_animals[str(ABL)][animal_idx, ild_16_idx]
            g_minus_16 = gamma_lapse_fit_all_animals[str(ABL)][animal_idx, ild_minus_16_idx]
            
            # Only plot if both gamma values exist
            if not np.isnan(g_16) and not np.isnan(g_minus_16):
                lapse_rates_pooled.append(lapse_rate)
                gamma_ild_16_pooled.append(g_16)
                gamma_ild_minus_16_pooled.append(g_minus_16)

# Scatter plot
ax.scatter(np.array(lapse_rates_pooled)*100, gamma_ild_16_pooled, color='blue', alpha=0.6, s=50, label='ILD = 16')
ax.scatter(np.array(lapse_rates_pooled)*100, np.abs(gamma_ild_minus_16_pooled), color='red', alpha=0.6, s=50, label='|ILD = -16|')

# Add vertical line at lapse rate threshold
ax.axvline(x=lapse_threshold*100, color='gray', linestyle='--', alpha=0.5, linewidth=1)

ax.set_xlabel('Lapse Rate % (vanilla_lapse)', fontsize=14)
ax.set_ylabel('Gamma', fontsize=14)
ax.legend(fontsize=12)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_title('Gamma vs Lapse Rate: Pooled Across All ABLs', fontsize=16)

plt.tight_layout()
plt.savefig('gamma_vs_lapse_rate_scatter_pooled.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Scatter plot: Lapse rate vs Gamma averaged across all ABLs for each animal

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# Collect data: average gamma across ABLs for each animal
lapse_rates_avg = []
gamma_ild_16_avg = []
gamma_ild_minus_16_avg = []

for animal_idx, (batch_name, animal_id) in enumerate(batch_animal_pairs):
    # Get lapse rate for this animal
    if animal_idx in animal_lapse_rates:
        lapse_rate = animal_lapse_rates[animal_idx]
        
        # Collect gamma values across all ABLs for this animal
        g_16_values = []
        g_minus_16_values = []
        
        for ABL in all_ABL:
            g_16 = gamma_lapse_fit_all_animals[str(ABL)][animal_idx, ild_16_idx]
            g_minus_16 = gamma_lapse_fit_all_animals[str(ABL)][animal_idx, ild_minus_16_idx]
            
            if not np.isnan(g_16):
                g_16_values.append(g_16)
            if not np.isnan(g_minus_16):
                g_minus_16_values.append(g_minus_16)
        
        # Only plot if we have gamma values for at least one ABL
        if len(g_16_values) > 0 and len(g_minus_16_values) > 0:
            lapse_rates_avg.append(lapse_rate)
            gamma_ild_16_avg.append(np.mean(g_16_values))
            gamma_ild_minus_16_avg.append(np.mean(g_minus_16_values))

# Scatter plot
ax.scatter(np.array(lapse_rates_avg)*100, gamma_ild_16_avg, color='blue', alpha=0.6, s=50, label='ILD = 16')
ax.scatter(np.array(lapse_rates_avg)*100, np.abs(gamma_ild_minus_16_avg), color='red', alpha=0.6, s=50, label='|ILD = -16|')

# Add vertical line at lapse rate threshold
ax.axvline(x=lapse_threshold*100, color='gray', linestyle='--', alpha=0.5, linewidth=1)

ax.set_xlabel('Lapse Rate % (vanilla_lapse)', fontsize=14)
ax.set_ylabel('Gamma (averaged across ABLs)', fontsize=14)
ax.legend(fontsize=12)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_title('Gamma (ABL-averaged) vs Lapse Rate', fontsize=16)

plt.tight_layout()
plt.savefig('gamma_vs_lapse_rate_scatter_abl_averaged.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Scatter plot: Lapse rate vs Gamma averaged across ABLs AND across ILD=±16

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# Collect data: average gamma across ABLs and across ILD=±16 for each animal
lapse_rates_final = []
gamma_final_avg = []

for animal_idx, (batch_name, animal_id) in enumerate(batch_animal_pairs):
    # Get lapse rate for this animal
    if animal_idx in animal_lapse_rates:
        lapse_rate = animal_lapse_rates[animal_idx]
        
        # Collect gamma values across all ABLs for this animal
        g_16_values = []
        g_minus_16_values = []
        
        for ABL in all_ABL:
            g_16 = gamma_lapse_fit_all_animals[str(ABL)][animal_idx, ild_16_idx]
            g_minus_16 = gamma_lapse_fit_all_animals[str(ABL)][animal_idx, ild_minus_16_idx]
            
            if not np.isnan(g_16):
                g_16_values.append(g_16)
            if not np.isnan(g_minus_16):
                g_minus_16_values.append(np.abs(g_minus_16))
        
        # Only plot if we have gamma values for at least one ABL
        if len(g_16_values) > 0 and len(g_minus_16_values) > 0:
            # Average across ABLs for each ILD
            avg_g_16 = np.mean(g_16_values)
            avg_g_minus_16 = np.mean(g_minus_16_values)
            
            # Average across ILD=±16
            final_gamma = (avg_g_16 + avg_g_minus_16) / 2
            
            lapse_rates_final.append(lapse_rate)
            gamma_final_avg.append(final_gamma)

# Scatter plot
lapse_rates_percent = np.array(lapse_rates_final) * 100
ax.scatter(lapse_rates_percent, gamma_final_avg, color='black', alpha=0.6, s=50, label='Data')

# Fit linear regression line
slope, intercept = np.polyfit(lapse_rates_percent, gamma_final_avg, 1)
x_line = np.linspace(lapse_rates_percent.min(), lapse_rates_percent.max(), 100)
y_line = slope * x_line + intercept

# Compute correlation coefficient
correlation = np.corrcoef(lapse_rates_percent, gamma_final_avg)[0, 1]

# Plot regression line
ax.plot(x_line, y_line, 'r--', linewidth=2, label=f'Fit: r = {correlation:.3f}')

# Add vertical line at lapse rate threshold
ax.axvline(x=lapse_threshold*100, color='gray', linestyle='--', alpha=0.5, linewidth=1)

ax.set_xlabel('Lapse Rate % (vanilla_lapse)', fontsize=14)
ax.set_ylabel('Gamma (averaged across ABLs & ILD=±16)', fontsize=14)
ax.legend(fontsize=12)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_title('Gamma (ABL & ILD averaged) vs Lapse Rate', fontsize=16)

plt.tight_layout()
plt.savefig('gamma_vs_lapse_rate_scatter_fully_averaged.png', dpi=300, bbox_inches='tight')
plt.show()

# %%