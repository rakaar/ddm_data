# %%
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add fit_each_condn to path for loading pickle files
sys.path.insert(0, '/home/rlab/raghavendra/ddm_data/fit_each_condn')
import led_off_gamma_omega_pdf_utils  # Required for unpickling VBMC files

# %%
# 1. lapses distribution, median vertical line
def plot_lapse_distribution():
    """
    Plot distribution of lapse probabilities across all animals (average of Vanilla+Lapse and Norm+Lapse models).
    Shows histogram with median vertical line.
    """
    # Load lapse parameters
    base_dir = '/home/rlab/raghavendra/ddm_data/fit_animal_by_animal'
    pkl_path = os.path.join(base_dir, 'lapse_parameters_all_animals.pkl')
    
    with open(pkl_path, 'rb') as f:
        lapse_params = pickle.load(f)
    
    # Extract average lapse_prob (vanilla_lapse + norm_lapse) for all animals
    lapse_probs = []
    for (batch, animal), data in lapse_params.items():
        vanilla_lp = data['vanilla_lapse']['lapse_prob']
        norm_lp = data['norm_lapse']['lapse_prob']
        
        # Only include if both models have valid lapse_prob
        if vanilla_lp is not None and norm_lp is not None:
            # Average the values from both models
            avg_lapse_prob = (vanilla_lp + norm_lp) / 2
            lapse_probs.append(avg_lapse_prob)
    
    lapse_probs = np.array(lapse_probs)
    
    # Convert to percentage (lapse rate)
    lapse_rates = lapse_probs * 100
    
    # Compute median
    median_lapse_rate = np.median(lapse_rates)
    
    # Create histogram
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot histogram
    lapse_rate_bins = np.arange(0, 5.5, 0.1)
    counts, bins, patches = ax.hist(lapse_rates, bins=lapse_rate_bins, color='grey', 
                                     alpha=0.7, edgecolor='black', linewidth=1.2)
    
    # Add median vertical line
    ax.axvline(median_lapse_rate, color='r', linestyle='--', linewidth=2, 
               label=f'Median = {median_lapse_rate:.2f}%')
    
    # Labels (publication-grade: no title, no grid)
    ax.set_xlabel('Lapse Rate (%)', fontsize=20)
    ax.set_ylabel('Count', fontsize=20)
    ax.legend(fontsize=12)
    
    # Set specific ticks
    # X-axis: 0 to 5 in steps of 1
    ax.set_xticks(np.arange(0, 6, 1))
    ax.tick_params(axis='x', labelsize=18)
    
    # Y-axis: only 0 and max count
    max_count = int(np.max(counts))
    ax.set_yticks([0, max_count])
    ax.tick_params(axis='y', labelsize=18)
    
    # Remove top and right spines for publication-grade appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Print statistics
    print(f"Total animals: {len(lapse_rates)}")
    print(f"Mean lapse rate: {np.mean(lapse_rates):.2f}%")
    print(f"Median lapse rate: {median_lapse_rate:.2f}%")
    print(f"Std lapse rate: {np.std(lapse_rates):.2f}%")
    print(f"Min lapse rate: {np.min(lapse_rates):.2f}%")
    print(f"Max lapse rate: {np.max(lapse_rates):.2f}%")
    
    # Save figure
    output_path = os.path.join(base_dir, 'lapse_distribution.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    
    # Save plot data to pickle
    plot_data = {
        'lapse_rates': lapse_rates,
        'median_lapse_rate': median_lapse_rate,
        'bins': bins,
        'counts': counts,
        'lapse_rate_bins': lapse_rate_bins,
        'statistics': {
            'mean': np.mean(lapse_rates),
            'median': median_lapse_rate,
            'std': np.std(lapse_rates),
            'min': np.min(lapse_rates),
            'max': np.max(lapse_rates),
            'n_animals': len(lapse_rates)
        }
    }
    
    pkl_output_path = os.path.join(base_dir, 'supp_lapses_distr_plot.pkl')
    with open(pkl_output_path, 'wb') as f:
        pickle.dump(plot_data, f)
    print(f"Plot data saved to: {pkl_output_path}")
    
    plt.show()
    
    return lapse_rates, median_lapse_rate

# Run the plot
lapse_rates, median_lapse_rate = plot_lapse_distribution()


# %%
# 2. gamma less than median lapse rate, > median lapse rate
# Step 1: Load data
def load_gamma_by_median_lapse_data():
    """
    Load and process gamma data separated by median lapse rate (average of Vanilla+Lapse and Norm+Lapse models).
    Returns all data needed for plotting.
    """
    import pandas as pd
    from collections import defaultdict
    
    base_dir = '/home/rlab/raghavendra/ddm_data/fit_animal_by_animal'
    
    # Load lapse parameters
    pkl_path = os.path.join(base_dir, 'lapse_parameters_all_animals.pkl')
    with open(pkl_path, 'rb') as f:
        lapse_params = pickle.load(f)
    
    # Get batch-animal pairs
    DESIRED_BATCHES = ['SD', 'LED34', 'LED6', 'LED8', 'LED7', 'LED34_even']
    batch_dir = os.path.join(base_dir, 'batch_csvs')
    batch_files = [f'batch_{batch_name}_valid_and_aborts.csv' for batch_name in DESIRED_BATCHES]
    
    merged_data = pd.concat([
        pd.read_csv(os.path.join(batch_dir, fname)) for fname in batch_files 
        if os.path.exists(os.path.join(batch_dir, fname))
    ], ignore_index=True)
    
    merged_valid = merged_data[merged_data['success'].isin([1, -1])].copy()
    batch_animal_pairs = sorted(list(map(tuple, merged_valid[['batch_name', 'animal']].drop_duplicates().values)))
    
    print(f"\nFound {len(batch_animal_pairs)} batch-animal pairs")
    
    # Calculate average lapse probability (vanilla_lapse + norm_lapse) for each animal
    animal_avg_lapse = {}
    for batch, animal in batch_animal_pairs:
        key = (batch, int(animal))
        if key in lapse_params:
            data = lapse_params[key]
            vanilla_lp = data['vanilla_lapse']['lapse_prob']
            norm_lp = data['norm_lapse']['lapse_prob']
            
            # Only include if both models have valid lapse_prob
            if vanilla_lp is not None and norm_lp is not None:
                # Average the values from both models
                avg_lapse_prob = (vanilla_lp + norm_lp) / 2
                animal_avg_lapse[(batch, str(animal))] = avg_lapse_prob
    
    print(f"Calculated average lapse probability (vanilla + norm) for {len(animal_avg_lapse)} animals")
    
    # Compute median lapse rate (in decimal form for threshold comparison)
    lapse_probs_list = list(animal_avg_lapse.values())
    median_lapse_prob = np.median(lapse_probs_list)
    median_lapse_rate_pct = median_lapse_prob * 100
    
    print(f"\nMedian lapse rate: {median_lapse_rate_pct:.2f}%")
    
    # Separate animals into two groups by median
    low_lapse_animals = []  # Below median
    high_lapse_animals = []  # At or above median
    
    for (batch, animal), avg_lapse in animal_avg_lapse.items():
        if avg_lapse < median_lapse_prob:
            low_lapse_animals.append((batch, animal))
        else:
            high_lapse_animals.append((batch, animal))
    
    print(f"\n--- Grouping by Median Lapse Rate (threshold = {median_lapse_rate_pct:.2f}%) ---")
    print(f"Below median (< {median_lapse_rate_pct:.2f}%): {len(low_lapse_animals)} animals")
    print(f"At or above median (>= {median_lapse_rate_pct:.2f}%): {len(high_lapse_animals)} animals")
    
    # Print animals in each group
    print("\nBelow median animals:")
    for batch, animal in sorted(low_lapse_animals):
        print(f"  {batch}_{animal}: {animal_avg_lapse[(batch, animal)]*100:.2f}%")
    
    print("\nAt or above median animals:")
    for batch, animal in sorted(high_lapse_animals):
        print(f"  {batch}_{animal}: {animal_avg_lapse[(batch, animal)]*100:.2f}%")
    
    # Function to get gamma from condition-by-condition fit
    def get_param_means_by_ABL_ILD(batch_name, animal_id, ABLs_to_fit, ILDs_to_fit, param_names=None):
        """
        Returns a dictionary with keys (ABL, ILD) and values as dicts of mean parameter values.
        """
        if param_names is None:
            param_names = ['gamma', 'omega']
        
        param_dict = {}
        for ABL in ABLs_to_fit:
            for ILD in ILDs_to_fit:
                pkl_folder = '/home/rlab/raghavendra/ddm_data/fit_each_condn/each_animal_cond_fit_gama_omega_pkl_files_LAPSES'
                pkl_file = os.path.join(pkl_folder, f'vbmc_cond_by_cond_{batch_name}_{animal_id}_{ABL}_ILD_{ILD}_FIX_t_E_w_del_go_same_as_parametric_LAPSES.pkl')
                if not os.path.exists(pkl_file):
                    continue
                with open(pkl_file, 'rb') as f:
                    vp = pickle.load(f)
                vp = vp.vp
                vp_samples = vp.sample(int(1e5))[0]
                means = {name: float(np.mean(vp_samples[:, i])) for i, name in enumerate(param_names)}
                param_dict[(ABL, ILD)] = means
        return param_dict
    
    # Extract gamma for both groups
    all_ABL = [20, 40, 60]
    all_ILD_sorted = np.sort([1, -1, 2, -2, 4, -4, 8, -8, 16, -16])
    
    # Initialize storage for low lapse group (below median)
    gamma_low_lapse = {
        '20': np.full((len(low_lapse_animals), len(all_ILD_sorted)), np.nan), 
        '40': np.full((len(low_lapse_animals), len(all_ILD_sorted)), np.nan), 
        '60': np.full((len(low_lapse_animals), len(all_ILD_sorted)), np.nan)
    }
    
    # Initialize storage for high lapse group (at or above median)
    gamma_high_lapse = {
        '20': np.full((len(high_lapse_animals), len(all_ILD_sorted)), np.nan), 
        '40': np.full((len(high_lapse_animals), len(all_ILD_sorted)), np.nan), 
        '60': np.full((len(high_lapse_animals), len(all_ILD_sorted)), np.nan)
    }
    
    # Fill in gamma for low lapse animals (below median)
    print("\n--- Processing Below Median Animals ---")
    for animal_idx, (batch_name, animal_id) in enumerate(low_lapse_animals):
        print(f'Processing {batch_name}_{animal_id} (lapse: {animal_avg_lapse[(batch_name, animal_id)]*100:.2f}%)')
        param_dict = get_param_means_by_ABL_ILD(batch_name, animal_id, all_ABL, all_ILD_sorted)
        for ABL in all_ABL:
            for ild_idx, ILD in enumerate(all_ILD_sorted):
                if (ABL, ILD) in param_dict:
                    gamma_low_lapse[str(ABL)][animal_idx, ild_idx] = param_dict[(ABL, ILD)]['gamma']
    
    # Fill in gamma for high lapse animals (at or above median)
    print("\n--- Processing At or Above Median Animals ---")
    for animal_idx, (batch_name, animal_id) in enumerate(high_lapse_animals):
        print(f'Processing {batch_name}_{animal_id} (lapse: {animal_avg_lapse[(batch_name, animal_id)]*100:.2f}%)')
        param_dict = get_param_means_by_ABL_ILD(batch_name, animal_id, all_ABL, all_ILD_sorted)
        for ABL in all_ABL:
            for ild_idx, ILD in enumerate(all_ILD_sorted):
                if (ABL, ILD) in param_dict:
                    gamma_high_lapse[str(ABL)][animal_idx, ild_idx] = param_dict[(ABL, ILD)]['gamma']
    
    # Return all data needed for plotting
    data = {
        'median_lapse_prob': median_lapse_prob,
        'median_lapse_rate_pct': median_lapse_rate_pct,
        'low_lapse_animals': low_lapse_animals,
        'high_lapse_animals': high_lapse_animals,
        'animal_avg_lapse': animal_avg_lapse,
        'all_ABL': all_ABL,
        'all_ILD_sorted': all_ILD_sorted,
        'gamma_low_lapse': gamma_low_lapse,
        'gamma_high_lapse': gamma_high_lapse,
    }
    
    return data

# Load data
gamma_data = load_gamma_by_median_lapse_data()


# %%
# Step 2: Plot gamma comparison
# def plot_gamma_by_median_lapse(data):
"""
Plot gamma separated by median lapse rate.
Blue: animals below median lapse rate
Red: animals above or equal to median lapse rate

Parameters:
-----------
data : dict
    Dictionary containing all data from load_gamma_by_median_lapse_data()
"""
data  = gamma_data
base_dir = '/home/rlab/raghavendra/ddm_data/fit_animal_by_animal'

# Unpack data
median_lapse_rate_pct = data['median_lapse_rate_pct']
low_lapse_animals = data['low_lapse_animals']
high_lapse_animals = data['high_lapse_animals']
all_ABL = data['all_ABL']
all_ILD_sorted = data['all_ILD_sorted']
gamma_low_lapse = data['gamma_low_lapse']
gamma_high_lapse = data['gamma_high_lapse']

# Plot gamma comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for abl_idx, ABL in enumerate(all_ABL):
    ax_gamma = axes[abl_idx]
    
    # Low lapse group (below median) - BLUE
    mean_gamma_low = np.nanmean(gamma_low_lapse[str(ABL)], axis=0)
    sem_gamma_low = np.nanstd(gamma_low_lapse[str(ABL)], axis=0) / np.sqrt(np.sum(~np.isnan(gamma_low_lapse[str(ABL)]), axis=0))
    
    # High lapse group (at or above median) - RED
    mean_gamma_high = np.nanmean(gamma_high_lapse[str(ABL)], axis=0)
    sem_gamma_high = np.nanstd(gamma_high_lapse[str(ABL)], axis=0) / np.sqrt(np.sum(~np.isnan(gamma_high_lapse[str(ABL)]), axis=0))
    
    ax_gamma.errorbar(all_ILD_sorted, mean_gamma_low, yerr=sem_gamma_low, fmt='o', 
                        color='k', label=f'Below median (n={len(low_lapse_animals)})', capsize=0, alpha=0.7)
    ax_gamma.errorbar(all_ILD_sorted, mean_gamma_high, yerr=sem_gamma_high, fmt='o', 
                        color='red', label=f'Above median (n={len(high_lapse_animals)})', capsize=0, alpha=0.7)
    
    # Publication-grade formatting
    ax_gamma.set_xlabel('ILD', fontsize=20)
    if abl_idx == 0:
        ax_gamma.set_ylabel('Gamma', fontsize=20)
    
    # Set specific ticks
    ax_gamma.set_xticks([-15, -5, 5, 15])
    ax_gamma.set_yticks([-2, 0, 2])
    ax_gamma.tick_params(axis='both', labelsize=18)
    
    # Remove top and right spines
    ax_gamma.spines['top'].set_visible(False)
    ax_gamma.spines['right'].set_visible(False)
    
    # Legend only on first panel
    # if abl_idx == 0:
    #     ax_gamma.legend(fontsize=12, frameon=False)

plt.tight_layout()

# Save figure
output_path = os.path.join(base_dir, 'gamma_sep_by_median_lapse_rate.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {output_path}")

# Save plot data to pickle
plot_data = {
    'median_lapse_prob': data['median_lapse_prob'],
    'median_lapse_rate_pct': median_lapse_rate_pct,
    'low_lapse_animals': low_lapse_animals,
    'high_lapse_animals': high_lapse_animals,
    'animal_avg_lapse': data['animal_avg_lapse'],
    'all_ABL': all_ABL,
    'all_ILD_sorted': all_ILD_sorted,
    'gamma_low_lapse': gamma_low_lapse,
    'gamma_high_lapse': gamma_high_lapse,
    'mean_gamma_by_abl': {
        str(ABL): {
            'below_median': np.nanmean(gamma_low_lapse[str(ABL)], axis=0),
            'above_median': np.nanmean(gamma_high_lapse[str(ABL)], axis=0),
            'sem_below_median': np.nanstd(gamma_low_lapse[str(ABL)], axis=0) / np.sqrt(np.sum(~np.isnan(gamma_low_lapse[str(ABL)]), axis=0)),
            'sem_above_median': np.nanstd(gamma_high_lapse[str(ABL)], axis=0) / np.sqrt(np.sum(~np.isnan(gamma_high_lapse[str(ABL)]), axis=0))
        } for ABL in all_ABL
    },
    'n_below_median': len(low_lapse_animals),
    'n_above_median': len(high_lapse_animals)
}

pkl_output_path = os.path.join(base_dir, 'gamma_sep_by_median_lapse_rate_data.pkl')
with open(pkl_output_path, 'wb') as f:
    pickle.dump(plot_data, f)
print(f"Plot data saved to: {pkl_output_path}")

plt.show()
    


# %%
# 3. rate_norm_l vs lapse rate
# each animal's samples are ellipse fit
# corr calculated

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


# Load and process data
def load_rate_norm_l_vs_lapse_data():
    """Load rate_norm_l and lapse_prob samples from all animals. Returns data dict."""
    import glob
    
    base_dir = '/home/rlab/raghavendra/ddm_data/fit_animal_by_animal'
    norm_lapse_dir = os.path.join(base_dir, 'oct_9_10_norm_lapse_model_fit_files')
    
    # Find all pickle files in norm+lapse directory
    norm_lapse_files = glob.glob(os.path.join(norm_lapse_dir, '*.pkl'))
    
    print(f"Found {len(norm_lapse_files)} norm+lapse pickle files")
    
    # Extract rate_norm_l and lapse_prob samples for all animals
    animal_data = []
    n_samples_per_animal = 5000
    
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
    
    return animal_data


# Load data
print("\n" + "="*60)
print("LOADING RATE_NORM_L VS LAPSE RATE DATA")
print("="*60)
animal_data = load_rate_norm_l_vs_lapse_data()


# %%
# Plot rate_norm_l vs lapse rate
def plot_rate_norm_l_vs_lapse(animal_data, ellipse_quantile=0.95, animals_by_color=False):
    """
    Create scatter plot with samples from each animal's posterior.
    Shows rate_norm_l vs lapse_prob with linear fit, correlation, and covariance ellipses.
    
    Args:
        animal_data: list of dicts with 'rate_norm_l_samples' and 'lapse_prob_samples'
        ellipse_quantile: confidence level for ellipses (default 0.95)
        animals_by_color: if True, each animal gets unique color; if False, all same color (default False)
    """
    if len(animal_data) == 0:
        print("No data to plot!")
        return None
    
    from matplotlib.patches import Ellipse as MplEllipse
    import matplotlib.cm as cm
    
    base_dir = '/home/rlab/raghavendra/ddm_data/fit_animal_by_animal'
    
    # Compute mean per animal for statistics
    rate_norm_l_means = [np.mean(d['rate_norm_l_samples']) for d in animal_data]
    lapse_prob_means = [np.mean(d['lapse_prob_samples']) * 100 for d in animal_data]  # Convert to %
    
    # Generate colors based on flag
    n_animals = len(animal_data)
    if animals_by_color:
        # Each animal gets unique color
        animal_colors = cm.tab20(np.linspace(0, 1, n_animals))
        sample_color = None  # Will use per-animal colors
    else:
        # All same color
        animal_colors = None
        sample_color = 'steelblue'
        ellipse_color = '#2b6cb0'  # Blue color
    
    # Flatten all samples across all animals (with animal indices if coloring by animal)
    all_rate_norm_l = []
    all_lapse_prob_pct = []
    all_animal_indices = []  # Track which animal each sample belongs to
    
    for idx, d in enumerate(animal_data):
        n_samples = len(d['rate_norm_l_samples'])
        all_rate_norm_l.extend(d['rate_norm_l_samples'])
        all_lapse_prob_pct.extend(np.array(d['lapse_prob_samples']) * 100)
        all_animal_indices.extend([idx] * n_samples)
    
    all_rate_norm_l = np.array(all_rate_norm_l)
    all_lapse_prob_pct = np.array(all_lapse_prob_pct)
    all_animal_indices = np.array(all_animal_indices)
    
    # Compute correlation using all samples (use percentage values)
    correlation = np.corrcoef(all_rate_norm_l, all_lapse_prob_pct)[0, 1]
    
    # Fit a linear regression line using all samples (use percentage values)
    z = np.polyfit(all_rate_norm_l, all_lapse_prob_pct, 1)
    p = np.poly1d(z)
    x_line = np.linspace(np.min(all_rate_norm_l), np.max(all_rate_norm_l), 100)
    y_line = p(x_line)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot samples based on flag
    if animals_by_color:
        # Plot with unique colors per animal
        for idx in range(n_animals):
            mask = all_animal_indices == idx
            ax.scatter(all_rate_norm_l[mask], all_lapse_prob_pct[mask], 
                      alpha=0.15, s=20, c=[animal_colors[idx]], edgecolors='none')
    else:
        # Plot all samples with same color (currently commented out)
        ax.scatter(all_rate_norm_l, all_lapse_prob_pct, alpha=0.15, s=20, 
                   c=sample_color, edgecolors='none')
        # pass
    
    # Fit and plot covariance ellipses for each animal
    # chi2 quantile for df=2 has closed form: s = -2 ln(1 - q)
    q = float(ellipse_quantile)
    if not (0.0 < q < 1.0):
        q = 0.95
    s_chi2 = -2.0 * np.log(max(1e-12, 1.0 - q))
    
    ellipse_data = []
    
    for idx, d in enumerate(animal_data):
        x = np.array(d['rate_norm_l_samples'])
        y = np.array(d['lapse_prob_samples']) * 100  # Convert to percentage
        
        if x.size < 2 or y.size < 2:
            continue
        
        # Remove non-finite values
        valid_mask = np.isfinite(x) & np.isfinite(y)
        x = x[valid_mask]
        y = y[valid_mask]
        
        if x.size < 2:
            continue
        
        m_x = float(np.mean(x))
        m_y = float(np.mean(y))
        
        # Compute 2x2 covariance matrix
        cov = np.cov(np.vstack([x, y]))
        
        if not np.all(np.isfinite(cov)):
            continue
        
        try:
            # Eigendecomposition
            evals, evecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            continue
        
        # Sort eigenvalues in descending order
        order = np.argsort(evals)[::-1]
        evals = np.maximum(evals[order], 0.0)
        evecs = evecs[:, order]
        
        # Ellipse axes (width/height) are 2*sqrt(s*lambda)
        width = 2.0 * float(np.sqrt(s_chi2 * evals[0])) if evals.size > 0 else 0.0
        height = 2.0 * float(np.sqrt(s_chi2 * evals[1])) if evals.size > 1 else 0.0
        
        if width == 0.0 or height == 0.0:
            continue
        
        # Rotation angle from eigenvectors
        angle = float(np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0])))
        
        # Select color based on flag
        if animals_by_color:
            current_ellipse_color = animal_colors[idx]
        else:
            current_ellipse_color = ellipse_color
        
        # Create ellipse patch
        ellipse = MplEllipse(
            (m_x, m_y), width=width, height=height, angle=angle,
            facecolor='none', edgecolor=current_ellipse_color, linewidth=1.5,
            alpha=0.8, zorder=4,
        )
        ax.add_patch(ellipse)
        
        # Store ellipse parameters
        ellipse_data.append({
            'batch': d['batch'],
            'animal': d['animal'],
            'mean_x': m_x,
            'mean_y': m_y,
            'width': width,
            'height': height,
            'angle': angle,
            'cov': cov,
            'evals': evals,
            'evecs': evecs
        })
    
    # Plot linear fit (no label for publication-grade)
    ax.plot(x_line, y_line, 'r--', linewidth=2, alpha=0.8)
    
    # Publication-grade formatting
    ax.set_xlabel(r'$\ell$', fontsize=20)
    ax.set_ylabel('Lapse rate (%)', fontsize=20)
    
    # Set specific ticks
    ax.set_xticks([0.8, 1.0])
    ax.set_yticks([0, 6])
    ax.tick_params(axis='both', labelsize=20)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(base_dir, 'rate_norm_l_vs_lapse_prob_scatter_with_ellipses.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    
    plt.show()
    
    # Print summary statistics (based on all samples)
    print("\n" + "="*60)
    print("SUMMARY STATISTICS (all posterior samples)")
    print("="*60)
    print(f"rate_norm_l:   mean={np.mean(all_rate_norm_l):.4f}, "
          f"std={np.std(all_rate_norm_l):.4f}, "
          f"min={np.min(all_rate_norm_l):.4f}, "
          f"max={np.max(all_rate_norm_l):.4f}")
    print(f"lapse_rate(%): mean={np.mean(all_lapse_prob_pct):.4f}, "
          f"std={np.std(all_lapse_prob_pct):.4f}, "
          f"min={np.min(all_lapse_prob_pct):.4f}, "
          f"max={np.max(all_lapse_prob_pct):.4f}")
    print(f"\nPearson correlation: {correlation:.4f}")
    
    # Print per-animal mean statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS (per-animal means)")
    print("="*60)
    print(f"rate_norm_l:   mean={np.mean(rate_norm_l_means):.4f}, "
          f"std={np.std(rate_norm_l_means):.4f}, "
          f"min={np.min(rate_norm_l_means):.4f}, "
          f"max={np.max(rate_norm_l_means):.4f}")
    print(f"lapse_rate(%): mean={np.mean(lapse_prob_means):.4f}, "
          f"std={np.std(lapse_prob_means):.4f}, "
          f"min={np.min(lapse_prob_means):.4f}, "
          f"max={np.max(lapse_prob_means):.4f}")
    correlation_means = np.corrcoef(rate_norm_l_means, lapse_prob_means)[0, 1]
    print(f"\nPearson correlation (means): {correlation_means:.4f}")
    
    # Print ellipse statistics
    print("\n" + "="*60)
    print(f"ELLIPSE FIT SUMMARY (quantile={ellipse_quantile})")
    print("="*60)
    print(f"Number of animals with ellipses fitted: {len(ellipse_data)}")
    print("="*60)
    
    # Prepare plot data for pickle
    plot_data = {
        'animal_data': animal_data,
        'all_rate_norm_l': all_rate_norm_l,
        'all_lapse_prob_pct': all_lapse_prob_pct,  # Percentage values
        'all_animal_indices': all_animal_indices,  # Track which animal each sample belongs to
        'rate_norm_l_means': rate_norm_l_means,
        'lapse_prob_means_pct': lapse_prob_means,  # Percentage values
        'correlation_all_samples': correlation,
        'correlation_means': correlation_means,
        'linear_fit': {
            'slope': z[0],
            'intercept': z[1],
            'x_line': x_line,
            'y_line': y_line  # This is already in percentage
        },
        'ellipses': ellipse_data,
        'ellipse_quantile': ellipse_quantile,
        'animals_by_color': animals_by_color,
        'ellipse_color': ellipse_color if not animals_by_color else None,
        'animal_colors': animal_colors.tolist() if animals_by_color and animal_colors is not None else None,
        'statistics': {
            'all_samples': {
                'rate_norm_l': {
                    'mean': np.mean(all_rate_norm_l),
                    'std': np.std(all_rate_norm_l),
                    'min': np.min(all_rate_norm_l),
                    'max': np.max(all_rate_norm_l)
                },
                'lapse_rate_pct': {
                    'mean': np.mean(all_lapse_prob_pct),
                    'std': np.std(all_lapse_prob_pct),
                    'min': np.min(all_lapse_prob_pct),
                    'max': np.max(all_lapse_prob_pct)
                }
            },
            'animal_means': {
                'rate_norm_l': {
                    'mean': np.mean(rate_norm_l_means),
                    'std': np.std(rate_norm_l_means),
                    'min': np.min(rate_norm_l_means),
                    'max': np.max(rate_norm_l_means)
                },
                'lapse_rate_pct': {
                    'mean': np.mean(lapse_prob_means),
                    'std': np.std(lapse_prob_means),
                    'min': np.min(lapse_prob_means),
                    'max': np.max(lapse_prob_means)
                }
            }
        },
        'n_animals': len(animal_data),
        'n_ellipses_fitted': len(ellipse_data)
    }
    
    # Save plot data to pickle
    pkl_output_path = os.path.join(base_dir, 'rate_norm_l_vs_lapse_prob_data.pkl')
    with open(pkl_output_path, 'wb') as f:
        pickle.dump(plot_data, f)
    print(f"\nPlot data saved to: {pkl_output_path}")
    
    return plot_data


# Run the plot
plot_data = plot_rate_norm_l_vs_lapse(animal_data,animals_by_color=True)
# %%
