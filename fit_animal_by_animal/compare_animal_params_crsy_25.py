# %%
# param vs animal wise pdfs

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

# Directory containing the results
RESULTS_DIR = os.path.dirname(__file__)
DESIRED_BATCHES = ['SD', 'LED34', 'LED6', 'LED8', 'LED7', 'LED34_even']


# Define simple, high-contrast colors for each batch
BATCH_COLORS = {
    'Comparable': 'red',
    'SD': '#87CEEB',  # sky blue
    'LED2': 'green',
    'LED1': 'orange',
    'LED34': 'purple',
    'LED7': 'black',
    'LED34_even': 'blue',
}

# %%
# Build animal list from CSVs for DESIRED_BATCHES
batch_dir = os.path.join(RESULTS_DIR, 'batch_csvs')
batch_files = [f'batch_{batch_name}_valid_and_aborts.csv' for batch_name in DESIRED_BATCHES]
dfs = []
for fname in batch_files:
    fpath = os.path.join(batch_dir, fname)
    if os.path.exists(fpath):
        dfs.append(pd.read_csv(fpath))
if len(dfs) > 0:
    merged_data = pd.concat(dfs, ignore_index=True)
    merged_valid = merged_data[merged_data['success'].isin([1, -1])].copy()
    batch_animal_pairs = sorted(list(map(tuple, merged_valid[['batch_name', 'animal']].drop_duplicates().values)))
    print(f"Found {len(batch_animal_pairs)} batch-animal pairs from CSVs.")
else:
    print('Warning: No batch CSVs found for DESIRED_BATCHES. Falling back to scanning PKL files in RESULTS_DIR.')
    batch_animal_pairs = []

# Build animal tuples from CSV-derived set if available; otherwise fallback to directory scan
animal_batch_tuples = []  # List of (batch, animal_number)
pkl_files = []  # List of (batch, animal_number, filename)
if batch_animal_pairs:
    for (batch, animal) in batch_animal_pairs:
        try:
            animal_id = int(animal)
        except Exception:
            # Skip non-integer animal identifiers as PKL files use integer IDs
            continue
        fname = f'results_{batch}_animal_{animal_id}.pkl'
        pkl_path = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(pkl_path):
            animal_batch_tuples.append((batch, animal_id))
            pkl_files.append((batch, animal_id, fname))
else:
    for fname in os.listdir(RESULTS_DIR):
        if fname.startswith('results_') and fname.endswith('.pkl'):
            for batch in DESIRED_BATCHES:
                prefix = f'results_{batch}_animal_'
                if fname.startswith(prefix):
                    try:
                        animal_id = int(fname.split('_')[-1].replace('.pkl', ''))
                        animal_batch_tuples.append((batch, animal_id))
                        pkl_files.append((batch, animal_id, fname))
                    except Exception:
                        continue

# Sort by batch then animal number
animal_batch_tuples = sorted(animal_batch_tuples, key=lambda x: (x[0], x[1]))
# %%
# Plot only rate_norm_l from vbmc_norm_tied_results

model_key = 'vbmc_norm_tied_results'
param_key = 'rate_norm_l_samples'
param_label = 'rate_norm_l'
plot_title = 'Norm TIED Model'

means = []
ci_lows = []   # 2.5th percentile
ci_highs = []  # 97.5th percentile
valid_animals = []  # Will store (batch, animal_id)
valid_labels = []   # Will store strings like 'LED7-92'
batch_colors = []   # Color for each animal

# Gather means and 95% CI (nonparametric)
for batch, animal_id in animal_batch_tuples:
    pkl_fname = f'results_{batch}_animal_{animal_id}.pkl'
    pkl_path = os.path.join(RESULTS_DIR, pkl_fname)
    if not os.path.exists(pkl_path):
        continue
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)
    if model_key not in results:
        continue
    if param_key not in results[model_key]:
        continue
    
    valid_animals.append((batch, animal_id))
    valid_labels.append(f'{batch}-{animal_id}')
    batch_colors.append(BATCH_COLORS.get(batch, 'gray'))
    
    samples = np.asarray(results[model_key][param_key])
    mean = np.mean(samples)
    ci_lower, ci_upper = np.percentile(samples, [2.5, 97.5])
    means.append(mean)
    ci_lows.append(ci_lower)
    ci_highs.append(ci_upper)

# Sort by mean value (descending)
if len(means) > 0:
    zipped = list(zip(means, ci_lows, ci_highs, valid_labels, batch_colors, valid_animals))
    zipped.sort(key=lambda x: x[0], reverse=True)
    sorted_means, sorted_ci_lows, sorted_ci_highs, sorted_labels, sorted_colors, sorted_animals = zip(*zipped)
    
    y_pos = np.arange(len(sorted_labels))
    fig, ax = plt.subplots(figsize=(7, 6))
    
    for idx in range(len(sorted_labels)):
        # Plot CI as a horizontal line
        ax.hlines(y=y_pos[idx], xmin=sorted_ci_lows[idx], xmax=sorted_ci_highs[idx], 
                  color=sorted_colors[idx], linewidth=3, alpha=0.7)
        # Plot mean as a point
        ax.plot(sorted_means[idx], y_pos[idx], 'o', color=sorted_colors[idx])
    
    # Set xlim to [0, 1] and xticks at 0 and 1
    ax.set_xlim([0, 1])
    ax.set_xticks([0, 1])
    
    # Set x-label with LaTeX ell symbol - larger font
    ax.set_xlabel(r'$\ell$', fontsize=48)
    
    # Increase tick font size
    ax.tick_params(axis='x', labelsize=40)
    
    # Remove y-ticks and y-label completely
    ax.set_yticks([])
    ax.tick_params(axis='y', which='both', left=False, right=False)
    
    # Remove borders - keep only x-axis
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Set margins - no x-margin, but y-margin to prevent cutting
    ax.margins(x=0, y=0.02)
    
    plt.tight_layout()
    
    # Save as SVG
    outname = f'rate_norm_l_vs_animals.svg'
    plt.savefig(os.path.join(RESULTS_DIR, outname), format='svg')
    print(f'Saved: {outname}')
    # plt.close(fig)
else:
    print(f'No data found for {param_label} in {model_key}')

# %%
print(f'Mean: {np.mean(sorted_means)}')
print(f'Min: {np.min(sorted_means)}')
print(f'Max: {np.max(sorted_means)}')   