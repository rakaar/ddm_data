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