# %%
# Compare vanilla vs vanilla+lapse models for LED34 animals

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# %%
# Helper function to extract ELBO from VBMC object
def extract_elbo(vbmc_obj):
    """Extract ELBO from VBMC object's iteration_history"""
    if hasattr(vbmc_obj, 'iteration_history'):
        iter_hist = vbmc_obj.iteration_history
        if 'elbo' in iter_hist:
            elbo_arr = iter_hist['elbo']
            return float(elbo_arr[-1])
    return np.nan

# %%
# Configuration
batch_name = 'LED34'
animals = [45, 57, 59, 61]

vanilla_dir = 'led34_filter_files/vanila'
lapse_dir = 'led34_filter_files/vanila_lapse'

# %%
# Load all results
results = {}

for animal in animals:
    # Load vanilla results
    vanilla_file = os.path.join(vanilla_dir, f'vbmc_PKL_file_vanilla_tied_results_batch_{batch_name}_animal_{animal}_FILTERED.pkl')
    with open(vanilla_file, 'rb') as f:
        vanilla_vbmc = pickle.load(f)
    
    # Load vanilla+lapse results
    lapse_file = os.path.join(lapse_dir, f'vbmc_vanilla_tied_results_batch_{batch_name}_animal_{animal}_lapses_truncate_1s_stim_filtered.pkl')
    with open(lapse_file, 'rb') as f:
        lapse_vbmc = pickle.load(f)
    
    results[animal] = {
        'vanilla': vanilla_vbmc,
        'lapse': lapse_vbmc
    }

print("Loaded results for all animals")

# %%
# Extract ELBOs and compute differences
elbo_data = {}

for animal in animals:
    # Extract ELBOs from both VBMC objects
    vanilla_elbo = extract_elbo(results[animal]['vanilla'])
    lapse_elbo = extract_elbo(results[animal]['lapse'])
    
    elbo_diff = lapse_elbo - vanilla_elbo
    
    elbo_data[animal] = {
        'vanilla': vanilla_elbo,
        'lapse': lapse_elbo,
        'diff': elbo_diff
    }

# %%
# Plot ELBO differences as bar plot
fig, ax = plt.subplots(figsize=(10, 6))

x_labels = [f'{batch_name}-{animal}' for animal in animals]
x_pos = np.arange(len(animals))
diffs = [elbo_data[animal]['diff'] for animal in animals]

bars = ax.bar(x_pos, diffs, color='steelblue', alpha=0.7, edgecolor='black')

ax.axhline(0, color='black', linestyle='--', linewidth=1)
ax.set_xlabel('Batch-Animal', fontsize=12, fontweight='bold')
ax.set_ylabel('ELBO Difference\n(Lapse - Vanilla)', fontsize=12, fontweight='bold')
ax.set_title('ELBO Improvement with Lapse Model', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, diff) in enumerate(zip(bars, diffs)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{diff:.1f}',
            ha='center', va='bottom' if height > 0 else 'top',
            fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

# %%
# Print parameter comparison tables for each animal
def extract_vanilla_params(vanilla_vbmc):
    """Extract parameters from vanilla VBMC object"""
    # Sample from variational posterior
    vp = vanilla_vbmc.vp
    vp_samples = vp.sample(int(1e6))[0]
    
    # Parameter order: rate_lambda, T_0, theta_E, w, t_E_aff, del_go
    params = {
        'rate_lambda': np.mean(vp_samples[:, 0]),
        'T_0_ms': np.mean(vp_samples[:, 1]) * 1e3,
        'theta_E': np.mean(vp_samples[:, 2]),
        'w': np.mean(vp_samples[:, 3]),
        't_E_aff_ms': np.mean(vp_samples[:, 4]) * 1e3,
        'del_go_ms': np.mean(vp_samples[:, 5]) * 1e3,
    }
    
    return params


def extract_lapse_params(lapse_vbmc):
    """Extract parameters from lapse VBMC object"""
    # Sample from variational posterior
    vp = lapse_vbmc.vp
    vp_samples = vp.sample(int(1e6))[0]
    
    # Parameter order: rate_lambda, T_0, theta_E, w, t_E_aff, del_go, lapse_prob, lapse_prob_right
    params = {
        'rate_lambda': np.mean(vp_samples[:, 0]),
        'T_0_ms': np.mean(vp_samples[:, 1]) * 1e3,
        'theta_E': np.mean(vp_samples[:, 2]),
        'w': np.mean(vp_samples[:, 3]),
        't_E_aff_ms': np.mean(vp_samples[:, 4]) * 1e3,
        'del_go_ms': np.mean(vp_samples[:, 5]) * 1e3,
        'lapse_prob': np.mean(vp_samples[:, 6]),
        'lapse_prob_right': np.mean(vp_samples[:, 7]),
    }
    
    return params

# %%
# Print tables for each animal
for animal in animals:
    print(f"\n{'='*80}")
    print(f"Animal: {batch_name}-{animal}")
    print(f"{'='*80}")
    
    # Extract parameters
    vanilla_params = extract_vanilla_params(results[animal]['vanilla'])
    lapse_params = extract_lapse_params(results[animal]['lapse'])
    
    # Print ELBOs
    print(f"\nELBO:")
    print(f"  Vanilla:       {elbo_data[animal]['vanilla']:>10.3f}")
    print(f"  Vanilla+Lapse: {elbo_data[animal]['lapse']:>10.3f}")
    print(f"  Difference:    {elbo_data[animal]['diff']:>10.3f}")
    
    # Print parameter comparison table
    print(f"\nParameters:")
    print(f"{'Parameter':<20} {'Vanilla':>12} {'Vanilla+Lapse':>15}")
    print(f"{'-'*50}")
    
    param_names = ['rate_lambda', 'T_0_ms', 'theta_E', 'w', 't_E_aff_ms', 'del_go_ms']
    
    for param in param_names:
        vanilla_val = vanilla_params[param]
        lapse_val = lapse_params[param]
        print(f"{param:<20} {vanilla_val:>12.3f} {lapse_val:>15.3f}")
    
    # Print lapse-specific parameters
    print(f"\nLapse-specific parameters:")
    print(f"{'lapse_prob':<20} {'N/A':>12} {lapse_params['lapse_prob']:>15.3f}")
    print(f"{'lapse_prob_right':<20} {'N/A':>12} {lapse_params['lapse_prob_right']:>15.3f}")

print(f"\n{'='*80}\n")

# %%
# ==================================================================================
# Compare Vanilla Filtered vs Vanilla Unfiltered (NO lapse models)
# ==================================================================================

# %%
# Load vanilla unfiltered results
vanilla_unfiltered_results = {}

for animal in animals:
    unfiltered_file = os.path.join(vanilla_dir, f'results_{batch_name}_animal_{animal}.pkl')
    with open(unfiltered_file, 'rb') as f:
        unfiltered_data = pickle.load(f)
    
    # Extract vbmc_vanilla_tied_results (this is a dict, not a VBMC object)
    vanilla_unfiltered_results[animal] = unfiltered_data['vbmc_vanilla_tied_results']

print("Loaded vanilla unfiltered results for all animals")

# %%
# Extract ELBOs for filtered and unfiltered
filter_comparison_data = {}

for animal in animals:
    # Filtered ELBO (already loaded as VBMC object)
    filtered_elbo = extract_elbo(results[animal]['vanilla'])
    
    # Unfiltered ELBO (from dict)
    unfiltered_elbo = vanilla_unfiltered_results[animal].get('elbo', np.nan)
    
    elbo_diff = filtered_elbo - unfiltered_elbo
    
    filter_comparison_data[animal] = {
        'filtered': filtered_elbo,
        'unfiltered': unfiltered_elbo,
        'diff': elbo_diff
    }

# %%
# Plot ELBO differences: Filtered - Unfiltered
fig, ax = plt.subplots(figsize=(10, 6))

x_labels = [f'{batch_name}-{animal}' for animal in animals]
x_pos = np.arange(len(animals))
diffs = [filter_comparison_data[animal]['diff'] for animal in animals]

bars = ax.bar(x_pos, diffs, color='coral', alpha=0.7, edgecolor='black')

ax.axhline(0, color='black', linestyle='--', linewidth=1)
ax.set_xlabel('Batch-Animal', fontsize=12, fontweight='bold')
ax.set_ylabel('ELBO Difference\n(Filtered - Unfiltered)', fontsize=12, fontweight='bold')
ax.set_title('ELBO: Vanilla Filtered vs Vanilla Unfiltered', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, diff) in enumerate(zip(bars, diffs)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{diff:.1f}',
            ha='center', va='bottom' if height > 0 else 'top',
            fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

# %%
# Extract parameters for unfiltered vanilla
def extract_vanilla_unfiltered_params(vanilla_dict):
    """Extract parameters from vanilla unfiltered results dict"""
    # This is a dict with sample arrays, not a VBMC object
    params = {}
    param_keys = {
        'rate_lambda': ('rate_lambda_samples', 1.0),
        'T_0_ms': ('T_0_samples', 1e3),
        'theta_E': ('theta_E_samples', 1.0),
        'w': ('w_samples', 1.0),
        't_E_aff_ms': ('t_E_aff_samples', 1e3),
        'del_go_ms': ('del_go_samples', 1e3),
    }
    
    for param_name, (sample_key, scale) in param_keys.items():
        if sample_key in vanilla_dict:
            samples = vanilla_dict[sample_key]
            params[param_name] = np.mean(samples) * scale
        else:
            params[param_name] = np.nan
    
    return params

# %%
# Print parameter comparison tables for filtered vs unfiltered
for animal in animals:
    print(f"\n{'='*80}")
    print(f"Animal: {batch_name}-{animal}")
    print(f"Vanilla Filtered vs Vanilla Unfiltered")
    print(f"{'='*80}")
    
    # Extract parameters
    filtered_params = extract_vanilla_params(results[animal]['vanilla'])
    unfiltered_params = extract_vanilla_unfiltered_params(vanilla_unfiltered_results[animal])
    
    # Print ELBOs
    # print(f"\nELBO:")
    # print(f"  Filtered:   {filter_comparison_data[animal]['filtered']:>10.3f}")
    # print(f"  Unfiltered: {filter_comparison_data[animal]['unfiltered']:>10.3f}")
    # print(f"  Difference: {filter_comparison_data[animal]['diff']:>10.3f}")
    

    # Print parameter comparison table
    print(f"\nParameters:")
    print(f"{'Parameter':<20} {'Filtered':>12} {'Unfiltered':>15}")
    print(f"{'-'*50}")
    
    param_names = ['rate_lambda', 'T_0_ms', 'theta_E', 'w', 't_E_aff_ms', 'del_go_ms']
    
    for param in param_names:
        filtered_val = filtered_params[param]
        unfiltered_val = unfiltered_params[param]
        print(f"{param:<20} {filtered_val:>12.3f} {unfiltered_val:>15.3f}")

print(f"\n{'='*80}\n")

# %%
# ==================================================================================
# Compare Vanilla+Lapse Filtered vs Vanilla+Lapse Unfiltered
# ==================================================================================

# %%
# Load vanilla+lapse unfiltered results
lapse_unfiltered_dir = '/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/oct_9_10_vanila_lapse_model_fit_files'
lapse_unfiltered_results = {}

for animal in animals:
    unfiltered_file = os.path.join(lapse_unfiltered_dir, f'vbmc_vanilla_tied_results_batch_{batch_name}_animal_{animal}_lapses_truncate_1s.pkl')
    with open(unfiltered_file, 'rb') as f:
        lapse_unfiltered_vbmc = pickle.load(f)
    
    lapse_unfiltered_results[animal] = lapse_unfiltered_vbmc

print("Loaded vanilla+lapse unfiltered results for all animals")

# %%
# Print parameter comparison tables for lapse filtered vs unfiltered
for animal in animals:
    print(f"\n{'='*80}")
    print(f"Animal: {batch_name}-{animal}")
    print(f"Vanilla+Lapse Filtered vs Vanilla+Lapse Unfiltered")
    print(f"{'='*80}")
    
    # Extract parameters
    lapse_filtered_params = extract_lapse_params(results[animal]['lapse'])
    lapse_unfiltered_params = extract_lapse_params(lapse_unfiltered_results[animal])
    
    # Print parameter comparison table
    print(f"\nParameters:")
    print(f"{'Parameter':<20} {'Filtered':>12} {'Unfiltered':>15}")
    print(f"{'-'*50}")
    
    param_names = ['rate_lambda', 'T_0_ms', 'theta_E', 'w', 't_E_aff_ms', 'del_go_ms', 'lapse_prob', 'lapse_prob_right']
    
    for param in param_names:
        filtered_val = lapse_filtered_params[param]
        unfiltered_val = lapse_unfiltered_params[param]
        print(f"{param:<20} {filtered_val:>12.3f} {unfiltered_val:>15.3f}")

print(f"\n{'='*80}\n")
# %%
