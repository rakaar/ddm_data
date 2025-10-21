#!/usr/bin/env python3
"""
Extract and plot rate_norm_l vs lapse_prob for norm+lapse model fits across all animals.
"""
import pickle
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import sys

# %%
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


def extract_params_from_norm_lapse_pkl(pkl_path, n_samples):
    """
    Extract rate_norm_l and lapse_prob samples from norm+lapse VBMC pickle file.
    
    Args:
        pkl_path: Path to pickle file
        n_samples: Number of samples to draw from the VP (default 100)
    
    Returns:
        dict with 'rate_norm_l_samples' and 'lapse_prob_samples' arrays (or None if extraction fails)
    """
    try:
        # Load VBMC object (not a dictionary!)
        with open(pkl_path, 'rb') as f:
            vbmc = pickle.load(f)
        
        # Extract VP object from iteration_history
        if not hasattr(vbmc, 'iteration_history'):
            print(f"No iteration_history in VBMC object from {pkl_path}")
            return None
        
        iter_hist = vbmc.iteration_history
        
        if 'vp' not in iter_hist:
            print(f"No 'vp' in iteration_history from {pkl_path}")
            return None
        
        # Get the last VP (variational posterior)
        vp_arr = iter_hist['vp']
        last_vp = vp_arr[-1]
        
        # Sample from VP
        # sample() returns (samples, log_weights), we need just the samples
        vp_samples, _ = last_vp.sample(n_samples)
        
        # Norm model parameter order: 
        # rate_lambda, T_0, theta_E, w, t_E_aff, del_go, rate_norm_l, lapse_prob, lapse_prob_right
        rate_norm_l_samples = vp_samples[:, 6]
        lapse_prob_samples = vp_samples[:, 7]
        
        return {
            'rate_norm_l_samples': rate_norm_l_samples,
            'lapse_prob_samples': lapse_prob_samples
        }
    
    except Exception as e:
        print(f"Error extracting parameters from {pkl_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


# %%
# Configuration
base_dir = '/home/rlab/raghavendra/ddm_data/fit_animal_by_animal'
norm_lapse_dir = os.path.join(base_dir, 'oct_9_10_norm_lapse_model_fit_files')

# Find all pickle files in norm+lapse directory
norm_lapse_files = glob.glob(os.path.join(norm_lapse_dir, '*.pkl'))

print(f"Found {len(norm_lapse_files)} norm+lapse pickle files")

# Extract rate_norm_l and lapse_prob samples for all animals
animal_data = []
n_samples_per_animal = 1000

for pkl_path in norm_lapse_files:
    filename = os.path.basename(pkl_path)
    try:
        batch, animal = parse_filename_norm_lapse(filename)
        print(f"Processing {batch} animal {animal}...")
        params = extract_params_from_norm_lapse_pkl(pkl_path, n_samples_per_animal)
        
        if params is not None:
            animal_data.append({
                'batch': batch,
                'animal': animal,
                'rate_norm_l_samples': params['rate_norm_l_samples'],
                'lapse_prob_samples': params['lapse_prob_samples']
            })
    except Exception as e:
        print(f"Error processing {filename}: {e}")

print(f"\nSuccessfully processed {len(animal_data)} animals")
print(f"Total number of points to plot: {len(animal_data) * n_samples_per_animal}")

# %%
# Create scatter plot with samples from each animal's posterior
if len(animal_data) > 0:
    # Flatten all samples across all animals
    all_rate_norm_l = []
    all_lapse_prob = []
    
    for d in animal_data:
        all_rate_norm_l.extend(d['rate_norm_l_samples'])
        all_lapse_prob.extend(d['lapse_prob_samples'])
    
    all_rate_norm_l = np.array(all_rate_norm_l)
    all_lapse_prob = np.array(all_lapse_prob)
    
    # Compute mean per animal for statistics
    rate_norm_l_means = [np.mean(d['rate_norm_l_samples']) for d in animal_data]
    lapse_prob_means = [np.mean(d['lapse_prob_samples']) for d in animal_data]
    
    # Compute correlation using all samples
    correlation = np.corrcoef(all_rate_norm_l, all_lapse_prob)[0, 1]
    
    # Fit a linear regression line using all samples
    z = np.polyfit(all_rate_norm_l, all_lapse_prob, 1)
    p = np.poly1d(z)
    x_line = np.linspace(np.min(all_rate_norm_l), np.max(all_rate_norm_l), 100)
    y_line = p(x_line)
    
    plt.figure(figsize=(8, 6))
    # Plot all samples with high transparency
    plt.scatter(all_rate_norm_l, all_lapse_prob, alpha=0.15, s=20, 
                c='steelblue', edgecolors='none')
    # Plot linear fit
    plt.plot(x_line, y_line, 'r--', linewidth=2, alpha=0.8, 
             label=f'Linear fit (r = {correlation:.3f})')
    
    plt.xlabel('rate_norm_l', fontsize=14)
    plt.ylabel('lapse_prob', fontsize=14)
    plt.title('ell vs Lapse rate', fontsize=15, fontweight='bold')
    # plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='best')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(base_dir, 'rate_norm_l_vs_lapse_prob_scatter.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    
    plt.show()
    
    # Print summary statistics (based on all samples)
    print("\n" + "="*60)
    print("SUMMARY STATISTICS (all posterior samples)")
    print("="*60)
    print(f"rate_norm_l:  mean={np.mean(all_rate_norm_l):.4f}, "
          f"std={np.std(all_rate_norm_l):.4f}, "
          f"min={np.min(all_rate_norm_l):.4f}, "
          f"max={np.max(all_rate_norm_l):.4f}")
    print(f"lapse_prob:   mean={np.mean(all_lapse_prob):.4f}, "
          f"std={np.std(all_lapse_prob):.4f}, "
          f"min={np.min(all_lapse_prob):.4f}, "
          f"max={np.max(all_lapse_prob):.4f}")
    print(f"\nPearson correlation: {correlation:.4f}")
    
    # Print per-animal mean statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS (per-animal means)")
    print("="*60)
    print(f"rate_norm_l:  mean={np.mean(rate_norm_l_means):.4f}, "
          f"std={np.std(rate_norm_l_means):.4f}, "
          f"min={np.min(rate_norm_l_means):.4f}, "
          f"max={np.max(rate_norm_l_means):.4f}")
    print(f"lapse_prob:   mean={np.mean(lapse_prob_means):.4f}, "
          f"std={np.std(lapse_prob_means):.4f}, "
          f"min={np.min(lapse_prob_means):.4f}, "
          f"max={np.max(lapse_prob_means):.4f}")
    correlation_means = np.corrcoef(rate_norm_l_means, lapse_prob_means)[0, 1]
    print(f"\nPearson correlation (means): {correlation_means:.4f}")
    print("="*60)
else:
    print("No data to plot!")
# %%
