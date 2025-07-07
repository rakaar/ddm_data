# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import glob
import os
from joblib import Parallel, delayed
from tqdm import tqdm
import pickle
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.neighbors import KernelDensity

# %%
DESIRED_BATCHES = ['SD', 'LED2', 'LED1', 'LED34', 'LED6', 'LED8', 'LED7']

# Base directory paths
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_dir = os.path.join(base_dir, 'batch_csvs')
results_dir = base_dir  # Directory containing the pickle files

def find_batch_animal_pairs():
    pairs = []
    pattern = os.path.join(results_dir, '../fit_animal_by_animal/results_*_animal_*.pkl')
    pickle_files = glob.glob(pattern)
    for pickle_file in pickle_files:
        filename = os.path.basename(pickle_file)
        parts = filename.split('_')
        if len(parts) >= 4:
            batch_index = parts.index('animal') - 1 if 'animal' in parts else 1
            animal_index = parts.index('animal') + 1 if 'animal' in parts else 2
            batch_name = parts[batch_index]
            animal_id = parts[animal_index].split('.')[0]
            if batch_name in DESIRED_BATCHES:
                if not ((batch_name == 'LED2' and animal_id in ['40', '41', '43']) or batch_name == 'LED1'):
                    pairs.append((batch_name, animal_id))
        else:
            print(f"Warning: Invalid filename format: {filename}")
    return pairs

batch_animal_pairs = find_batch_animal_pairs()
print(f"Found {len(batch_animal_pairs)} batch-animal pairs: {batch_animal_pairs}")

# %%
def get_animal_chronometric_data(batch_name, animal_id):
    """
    Calculates mean reaction time and standard error for a given animal from a batch.
    """
    file_name = os.path.join(csv_dir, f'batch_{batch_name}_valid_and_aborts.csv')
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"File not found: {file_name}")
        return None

    df_animal = df[df['animal'] == animal_id].copy()
    
    # Ensure 'abs_ILD' column exists
    if 'ILD' in df_animal.columns:
        df_animal['abs_ILD'] = df_animal['ILD'].abs()
    else:
        print(f"Warning: 'ILD' column not found for animal {animal_id} in {file_name}")
        return None

    # Filter for valid trials (responded after sound onset)
    # As per memory, this is success == 1 or -1
    df_valid = df_animal[df_animal['success'].isin([1, -1])].copy()
    # RTs >=0, <= 1
    df_valid = df_valid[(df_valid['RTwrtStim'] >= 0) & (df_valid['RTwrtStim'] <= 1)].copy()

    if df_valid.empty:
        print(f"No valid trials found for animal {animal_id} in {file_name}")
        return None

    # Calculate mean and SEM for RTwrtStim, grouped by ABL and abs_ILD
    chrono_data = df_valid.groupby(['ABL', 'abs_ILD'])['RTwrtStim'].agg(['mean', 'sem']).reset_index()
    
    return chrono_data

# %%
def process_batch_animal(batch_animal_pair):
    """
    Wrapper function to process a single batch-animal pair.
    """
    batch_name, animal_id = batch_animal_pair
    try:
        # animal_id from the pairs can be string, convert to int for comparison with df column
        chrono_data = get_animal_chronometric_data(batch_name, int(animal_id))
        if chrono_data is not None and not chrono_data.empty:
            return (batch_animal_pair, chrono_data)
    except Exception as e:
        print(f"Error processing batch {batch_name}, animal {animal_id}: {str(e)}")
    return None

# %%
# Run processing in parallel
n_jobs = max(1, os.cpu_count() - 4) # Leave some cores free
print(f"Processing {len(batch_animal_pairs)} animal-batch pairs on {n_jobs} cores...")
results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(process_batch_animal)(pair) for pair in batch_animal_pairs
)

# Filter out None results from processing
valid_results = [r for r in results if r is not None]
print(f"Completed processing. Found data for {len(valid_results)} batch-animal pairs.")

# %% 
# Plotting
output_dir = 'animal_specific_chronometric_plots'
os.makedirs(output_dir, exist_ok=True)
print(f"Saving plots to '{output_dir}/'")

abl_colors = {20: 'tab:blue', 40: 'tab:orange', 60: 'tab:green'}
abs_ild_ticks = [1, 2, 4, 8, 16]

# Prepare figure for subplots
n_animals = len(valid_results)
n_cols = 5
n_rows = int(np.ceil(n_animals / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5), squeeze=False)
axes = axes.flatten() # Flatten to make it easier to iterate

for i, result in enumerate(valid_results):
    ax = axes[i]
    batch_animal_pair, chrono_data = result
    batch_name, animal_id = batch_animal_pair

    # Sort by ABL to ensure consistent legend order
    for abl in sorted(chrono_data['ABL'].unique()):
        if abl not in abl_colors:
            continue
        
        abl_data = chrono_data[chrono_data['ABL'] == abl].sort_values('abs_ILD')
        
        ax.errorbar(
            x=abl_data['abs_ILD'], 
            y=abl_data['mean'], 
            yerr=abl_data['sem'],
            fmt='o-',
            color=abl_colors[abl],
            label=f'{int(abl)} dB',
            capsize=0, 
            markersize=4,
            linewidth=1.5
        )
        
    ax.set_xlabel('Absolute ILD (dB)')
    if i % n_cols == 0: # Only show y-label on the first column
        ax.set_ylabel('Mean RT (s)')

    ax.set_title(f'Animal {animal_id} ({batch_name})', fontsize=10)
    ax.legend(title='ABL', fontsize='x-small')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    ax.set_xscale('log')
    ax.set_xticks(abs_ild_ticks)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.tick_params(axis='x', labelsize=8)

    if not chrono_data.empty:
        max_rt = chrono_data['mean'].max()
        ax.set_ylim(bottom=0, top=max_rt * 1.15)

# Hide unused subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.suptitle('Chronometric Curves for All Animals', fontsize=16)

# Save the single figure
output_filename = os.path.join(output_dir, 'all_animals_chronometric_grid.png')
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"All chronometric plots saved in a single file: '{output_filename}'.")