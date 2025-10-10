# %%
#!/usr/bin/env python3
"""
Compare ELBO values from vanilla+lapse and norm+lapse model fits.
Extracts ELBO and stability information from VBMC pickle files in both folders,
and compares with original vanilla and norm model ELBOs.
"""
import pickle
import os
import glob
import matplotlib.pyplot as plt
import numpy as np


def extract_convergence_info(pkl_path):
    """
    Extract convergence information from a VBMC pickle file.
    
    Returns:
        dict with keys: elbo, elbo_sd, stable, n_iterations
    """
    try:
        with open(pkl_path, 'rb') as f:
            vbmc = pickle.load(f)
        
        # Extract from iteration_history
        if hasattr(vbmc, 'iteration_history'):
            iter_hist = vbmc.iteration_history
            
            result = {}
            
            if 'elbo' in iter_hist:
                elbo_arr = iter_hist['elbo']
                result['elbo'] = float(elbo_arr[-1])
            else:
                result['elbo'] = None
            
            if 'elbo_sd' in iter_hist:
                elbo_sd_arr = iter_hist['elbo_sd']
                result['elbo_sd'] = float(elbo_sd_arr[-1])
            else:
                result['elbo_sd'] = None
            
            if 'stable' in iter_hist:
                stable_arr = iter_hist['stable']
                result['stable'] = bool(stable_arr[-1])
            else:
                result['stable'] = None
            
            if 'iter' in iter_hist:
                iter_arr = iter_hist['iter']
                result['n_iterations'] = int(iter_arr[-1])
            else:
                result['n_iterations'] = len(iter_hist)
            
            return result
        else:
            return {'elbo': None, 'elbo_sd': None, 'stable': None, 'n_iterations': None}
    
    except Exception as e:
        print(f"Error reading {pkl_path}: {e}")
        return {'elbo': None, 'elbo_sd': None, 'stable': None, 'n_iterations': None, 'error': str(e)}


def parse_filename_vanilla_lapse(filename):
    """
    Parse vanilla+lapse pickle filename to extract batch and animal.
    Expected format: vbmc_vanilla_tied_results_batch_{batch}_animal_{animal}_lapses_truncate_1s.pkl
    """
    # Remove .pkl extension
    name = filename.replace('.pkl', '')
    
    # Split by underscores
    parts = name.split('_')
    
    # Find batch name (after 'batch_')
    batch_idx = parts.index('batch') + 1
    batch_parts = []
    animal_idx = None
    
    for i in range(batch_idx, len(parts)):
        if parts[i] == 'animal':
            animal_idx = i + 1
            break
        batch_parts.append(parts[i])
    
    batch = '_'.join(batch_parts)
    
    # Get animal ID
    animal = int(parts[animal_idx])
    
    return batch, animal


def parse_filename_norm_lapse(filename):
    """
    Parse norm+lapse pickle filename to extract batch and animal.
    Expected format: vbmc_norm_tied_results_batch_{batch}_animal_{animal}_lapses_truncate_1s_norm.pkl
    """
    # Remove .pkl extension
    name = filename.replace('.pkl', '')
    
    # Split by underscores
    parts = name.split('_')
    
    # Find batch name (after 'batch_')
    batch_idx = parts.index('batch') + 1
    batch_parts = []
    animal_idx = None
    
    for i in range(batch_idx, len(parts)):
        if parts[i] == 'animal':
            animal_idx = i + 1
            break
        batch_parts.append(parts[i])
    
    batch = '_'.join(batch_parts)
    
    # Get animal ID
    animal = int(parts[animal_idx])
    
    return batch, animal


def get_original_elbos(batch, animal_id, results_dir):
    """
    Load original vanilla and norm ELBO values from results pickle.
    
    Returns:
        dict with keys: og_vanilla_elbo, og_norm_elbo
    """
    pkl_fname = f'results_{batch}_animal_{animal_id}.pkl'
    pkl_path = os.path.join(results_dir, pkl_fname)
    
    result = {'og_vanilla_elbo': None, 'og_norm_elbo': None}
    
    if not os.path.exists(pkl_path):
        return result
    
    try:
        with open(pkl_path, 'rb') as f:
            results = pickle.load(f)
        
        # Extract vanilla ELBO
        if 'vbmc_vanilla_tied_results' in results:
            result['og_vanilla_elbo'] = results['vbmc_vanilla_tied_results'].get('elbo', None)
        
        # Extract norm ELBO
        if 'vbmc_norm_tied_results' in results:
            result['og_norm_elbo'] = results['vbmc_norm_tied_results'].get('elbo', None)
        
        return result
    except Exception as e:
        print(f"Warning: Could not load original ELBOs from {pkl_path}: {e}")
        return result


def format_bool(val):
    """Format boolean for display"""
    if val is None:
        return "N/A"
    return "True" if val else "False"


def format_float(val):
    """Format float for display"""
    if val is None:
        return "N/A"
    return f"{val:.2f}"

# %%
# Configuration
base_dir = '/home/rlab/raghavendra/ddm_data/fit_animal_by_animal'
vanilla_lapse_dir = os.path.join(base_dir, 'oct_9_10_vanila_lapse_model_fit_files')
norm_lapse_dir = os.path.join(base_dir, 'oct_9_10_norm_lapse_model_fit_files')
results_dir = base_dir  # Results files are in the main directory
    
# Find all pickle files in both directories
vanilla_lapse_files = glob.glob(os.path.join(vanilla_lapse_dir, '*.pkl'))
norm_lapse_files = glob.glob(os.path.join(norm_lapse_dir, '*.pkl'))

print(f"Found {len(vanilla_lapse_files)} vanilla+lapse pickle files")
print(f"Found {len(norm_lapse_files)} norm+lapse pickle files")

# Extract batch, animal pairs from vanilla+lapse files
vanilla_lapse_data = {}
for pkl_path in vanilla_lapse_files:
    filename = os.path.basename(pkl_path)
    try:
        batch, animal = parse_filename_vanilla_lapse(filename)
        conv_info = extract_convergence_info(pkl_path)
        vanilla_lapse_data[(batch, animal)] = conv_info
    except Exception as e:
        print(f"Error parsing {filename}: {e}")

# Extract batch, animal pairs from norm+lapse files
norm_lapse_data = {}
for pkl_path in norm_lapse_files:
    filename = os.path.basename(pkl_path)
    try:
        batch, animal = parse_filename_norm_lapse(filename)
        conv_info = extract_convergence_info(pkl_path)
        norm_lapse_data[(batch, animal)] = conv_info
    except Exception as e:
        print(f"Error parsing {filename}: {e}")

# Find common (batch, animal) pairs
vanilla_keys = set(vanilla_lapse_data.keys())
norm_keys = set(norm_lapse_data.keys())
common_keys = vanilla_keys & norm_keys

print(f"\nFound {len(common_keys)} common (batch, animal) pairs")

# %%
# Build results table
rows = []
for batch, animal in sorted(common_keys):
    vanilla_info = vanilla_lapse_data[(batch, animal)]
    norm_info = norm_lapse_data[(batch, animal)]
    og_elbos = get_original_elbos(batch, animal, results_dir)
    
    row = {
        'batch': batch,
        'animal': animal,
        'vanilla_lapse_stable': vanilla_info.get('stable'),
        'norm_lapse_stable': norm_info.get('stable'),
        'vanilla_lapse_elbo': vanilla_info.get('elbo'),
        'norm_lapse_elbo': norm_info.get('elbo'),
        'og_vanilla_elbo': og_elbos['og_vanilla_elbo'],
        'og_norm_elbo': og_elbos['og_norm_elbo'],
    }
    rows.append(row)

# %%
# Print table header
print("\n" + "="*150)
print("ELBO Comparison Table")
print("="*150)

# Column headers
header = f"{'Batch':<15} {'Animal':<8} {'V+L Stable':<12} {'N+L Stable':<12} {'V+L ELBO':<12} {'N+L ELBO':<12} {'OG V ELBO':<12} {'OG N ELBO':<12}"
print(header)
print("-" * 150)

# Print rows
for row in rows:
    line = f"{row['batch']:<15} {row['animal']:<8} "
    line += f"{format_bool(row['vanilla_lapse_stable']):<12} "
    line += f"{format_bool(row['norm_lapse_stable']):<12} "
    line += f"{format_float(row['vanilla_lapse_elbo']):<12} "
    line += f"{format_float(row['norm_lapse_elbo']):<12} "
    line += f"{format_float(row['og_vanilla_elbo']):<12} "
    line += f"{format_float(row['og_norm_elbo']):<12}"
    print(line)

# %%
# Summary statistics
print("\n" + "="*150)
print("Summary Statistics")
print("="*150)

print(f"\nTotal animals analyzed: {len(rows)}")

# Count stable
v_stable = sum(1 for row in rows if row['vanilla_lapse_stable'])
n_stable = sum(1 for row in rows if row['norm_lapse_stable'])
print(f"\nVanilla+Lapse stable: {v_stable}/{len(rows)}")
print(f"Norm+Lapse stable: {n_stable}/{len(rows)}")

# Compute ELBO differences
v_diffs = []
n_diffs = []
for row in rows:
    if row['vanilla_lapse_elbo'] is not None and row['og_vanilla_elbo'] is not None:
        v_diffs.append(row['vanilla_lapse_elbo'] - row['og_vanilla_elbo'])
    if row['norm_lapse_elbo'] is not None and row['og_norm_elbo'] is not None:
        n_diffs.append(row['norm_lapse_elbo'] - row['og_norm_elbo'])

if v_diffs:
    print(f"\nVanilla+Lapse ELBO improvement over original:")
    print(f"  Mean: {sum(v_diffs)/len(v_diffs):.2f}")
    print(f"  Median: {sorted(v_diffs)[len(v_diffs)//2]:.2f}")
    print(f"  Min: {min(v_diffs):.2f}")
    print(f"  Max: {max(v_diffs):.2f}")

if n_diffs:
    print(f"\nNorm+Lapse ELBO improvement over original:")
    print(f"  Mean: {sum(n_diffs)/len(n_diffs):.2f}")
    print(f"  Median: {sorted(n_diffs)[len(n_diffs)//2]:.2f}")
    print(f"  Min: {min(n_diffs):.2f}")
    print(f"  Max: {max(n_diffs):.2f}")

# %%
# Save to CSV
output_csv = os.path.join(base_dir, 'vanilla_norm_lapse_elbo_comparison.csv')
with open(output_csv, 'w') as f:
    # Write header
    f.write("batch,animal,vanilla_lapse_stable,norm_lapse_stable,vanilla_lapse_elbo,norm_lapse_elbo,og_vanilla_elbo,og_norm_elbo\n")
    # Write rows
    for row in rows:
        f.write(f"{row['batch']},{row['animal']},")
        f.write(f"{row['vanilla_lapse_stable']},{row['norm_lapse_stable']},")
        f.write(f"{row['vanilla_lapse_elbo']},{row['norm_lapse_elbo']},")
        f.write(f"{row['og_vanilla_elbo']},{row['og_norm_elbo']}\n")

print(f"\nResults saved to: {output_csv}")

# %%
# ELBO Comparison Bar Plots

# Prepare data for plotting
animal_labels = [f"{row['batch']}_{row['animal']}" for row in rows]
x_pos = np.arange(len(rows))

# Compute all three comparisons
comparison_1 = []  # Vanilla+Lapse - Vanilla
comparison_2 = []  # Vanilla+Lapse - Norm
comparison_3 = []  # Norm+Lapse - Norm

for row in rows:
    # Comparison 1: Vanilla+Lapse ELBO - Vanilla ELBO
    if row['vanilla_lapse_elbo'] is not None and row['og_vanilla_elbo'] is not None:
        comparison_1.append(row['vanilla_lapse_elbo'] - row['og_vanilla_elbo'])
    else:
        comparison_1.append(0)
    
    # Comparison 2: Vanilla+Lapse ELBO - Norm ELBO
    if row['vanilla_lapse_elbo'] is not None and row['og_norm_elbo'] is not None:
        comparison_2.append(row['vanilla_lapse_elbo'] - row['og_norm_elbo'])
    else:
        comparison_2.append(0)
    
    # Comparison 3: Norm+Lapse ELBO - Norm ELBO
    if row['norm_lapse_elbo'] is not None and row['og_norm_elbo'] is not None:
        comparison_3.append(row['norm_lapse_elbo'] - row['og_norm_elbo'])
    else:
        comparison_3.append(0)
# %%
# Create figure with 3 subplots
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Plot 1: Vanilla+Lapse - Vanilla
ax1 = axes[0]
colors_1 = ['green' if val > 0 else 'red' for val in comparison_1]
ax1.bar(x_pos, comparison_1, color=colors_1, alpha=0.7)
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax1.set_ylabel('ELBO Difference', fontsize=12, fontweight='bold')
ax1.set_title('Vanilla+Lapse ELBO - Original Vanilla ELBO', fontsize=14, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(animal_labels, rotation=45, ha='right')
ax1.grid(axis='y', alpha=0.3)
ax1.set_xlabel('Batch_Animal', fontsize=11)
ax1.set_ylim(-100, 100)

# Plot 2: Vanilla+Lapse - Norm
ax2 = axes[1]
colors_2 = ['green' if val > 0 else 'red' for val in comparison_2]
ax2.bar(x_pos, comparison_2, color=colors_2, alpha=0.7)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.set_ylabel('ELBO Difference', fontsize=12, fontweight='bold')
ax2.set_title('Vanilla+Lapse ELBO - Original Norm ELBO', fontsize=14, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(animal_labels, rotation=45, ha='right')
ax2.grid(axis='y', alpha=0.3)
ax2.set_xlabel('Batch_Animal', fontsize=11)
ax2.set_ylim(-100, 100)

# Plot 3: Norm+Lapse - Norm
ax3 = axes[2]
colors_3 = ['green' if val > 0 else 'red' for val in comparison_3]
ax3.bar(x_pos, comparison_3, color=colors_3, alpha=0.7)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax3.set_ylabel('ELBO Difference', fontsize=12, fontweight='bold')
ax3.set_title('Norm+Lapse ELBO - Original Norm ELBO', fontsize=14, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(animal_labels, rotation=45, ha='right')
ax3.grid(axis='y', alpha=0.3)
ax3.set_xlabel('Batch_Animal', fontsize=11)
ax3.set_ylim(-100, 100)

plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'elbo_comparisons_bar_plots.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"\nBar plots saved to: {os.path.join(base_dir, 'elbo_comparisons_bar_plots.png')}") 