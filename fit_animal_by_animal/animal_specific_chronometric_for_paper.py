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
# DESIRED_BATCHES = ['LED2', 'LED1', 'LED34', 'LED6', 'LED8', 'LED7']

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
                if batch_name not in ['LED1', 'LED2']:
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

for i, result in enumerate(valid_results):
    r, c = divmod(i, n_cols)
    ax = axes[r, c]

    batch_animal_pair, chrono_data = result
    batch_name, animal_id = batch_animal_pair

    # Filter for selected absolute ILDs
    chrono_data = chrono_data[chrono_data['abs_ILD'].isin(abs_ild_ticks)]

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
    if c == 0: # Only show y-label on the first column of each row
        ax.set_ylabel('Mean RT (s)')

    ax.set_title(f'Animal {animal_id} ({batch_name})', fontsize=10)
    ax.legend(title='ABL', fontsize='x-small')
    
    ax.set_xscale('log')
    ax.set_xticks(abs_ild_ticks)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.tick_params(axis='x', labelsize=8)

    ax.set_ylim(0, 0.5)
    ax.set_yticks([0, 0.2, 0.4])

# Hide any remaining unused subplots
for i in range(len(valid_results), n_rows * n_cols):
    r, c = divmod(i, n_cols)
    axes[r, c].set_visible(False)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.suptitle('Chronometric Curves for All Animals', fontsize=16)

# Save the single figure
output_filename = os.path.join(output_dir, 'all_animals_chronometric_grid.png')
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"All chronometric plots saved in a single file: '{output_filename}'.")
# %%
# New section for 1x4 summary plot

# 1. Data Preparation
# Consolidate all chronometric data into a single DataFrame
all_chrono_data_list = []
for result in valid_results:
    (batch_animal_pair, chrono_data) = result
    batch_name, animal_id = batch_animal_pair
    
    # Add identifiers to each animal's data
    chrono_data_copy = chrono_data.copy()
    chrono_data_copy['animal_id'] = animal_id
    chrono_data_copy['batch_name'] = batch_name
    all_chrono_data_list.append(chrono_data_copy)

all_chrono_data_df = pd.concat(all_chrono_data_list, ignore_index=True)

# Filter for the desired absolute ILDs
all_chrono_data_df = all_chrono_data_df[all_chrono_data_df['abs_ILD'].isin(abs_ild_ticks)]


# %%
# 2. Create the 1x4 subplot figure
fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), sharey=True)
plot_abls = [20, 40, 60]
grand_means_data = {}  # To store the mean data for the last plot
keep_grid = False

# 3. Generate first 3 subplots
for i, abl in enumerate(plot_abls):
    ax = axes[i]
    
    # Filter data for the current ABL
    abl_df = all_chrono_data_df[all_chrono_data_df['ABL'] == abl]
    
    # Plot individual animal lines in a light color
    for animal_id in abl_df['animal_id'].unique():
        animal_df = abl_df[abl_df['animal_id'] == animal_id].sort_values('abs_ILD')
        ax.plot(animal_df['abs_ILD'], animal_df['mean'], color='gray', alpha=0.4, linewidth=1.5)
            
    # Calculate and plot the grand mean with SEM
    grand_mean_stats = abl_df.groupby('abs_ILD')['mean'].agg(['mean', 'sem']).reset_index().sort_values('abs_ILD')


    # Store grand mean for the final plot
    grand_means_data[abl] = grand_mean_stats
    

    # Plot mean line with error bars
    ax.errorbar(
        x=grand_mean_stats['abs_ILD'],
        y=grand_mean_stats['mean'],
        yerr=grand_mean_stats['sem'],
        fmt='o-',
        color='black',
        linewidth=2.5,
        markersize=6,
        capsize=3, # Add caps to the error bars for clarity
        label='Population Mean'
    )

    # Formatting for each subplot
    ax.set_title(f'ABL = {int(abl)} dB', fontsize=14)
    ax.set_xlabel('Absolute ILD (dB)', fontsize=12)
    if i == 0:
        ax.set_ylabel('Mean RT (s)', fontsize=12)
    
    ax.set_xscale('log')
    ax.set_xticks(abs_ild_ticks)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    if keep_grid:
        ax.grid(True, which="both", ls="--", linewidth=1)
    ax.set_ylim(0.1, 0.5) # Consistent Y-axis limits
    ax.set_yticks(np.arange(0.1, 0.51, 0.1))


# 4. Generate the 4th subplot (aggregation)
ax4 = axes[3]
for abl, stats_data in grand_means_data.items():

    # Plot mean line with error bars
    ax4.errorbar(
        x=stats_data['abs_ILD'],
        y=stats_data['mean'],
        yerr=stats_data['sem'],
        fmt='o-',
        color=abl_colors[abl],
        label=f'{int(abl)} dB',
        linewidth=2,
        markersize=5,
        capsize=3
    )

# Formatting for the 4th subplot
ax4.set_title('Mean of means', fontsize=14)
ax4.set_xlabel('Absolute ILD (dB)', fontsize=12)
ax4.legend(title='ABL', fontsize=10)
ax4.set_xscale('log')
ax4.set_xticks(abs_ild_ticks)
ax4.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax4.tick_params(axis='x', labelsize=10)
if keep_grid:
    ax4.grid(True, which="both", ls="--", linewidth=1)


# 5. Final figure adjustments and saving
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.suptitle('Chronometric Curves: Individual Animals vs. Population Mean', fontsize=18, y=1.03)

summary_plot_filename = os.path.join(output_dir, 'summary_chronometric_plot_by_abl.png')
plt.savefig(summary_plot_filename, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"Summary plot saved to '{summary_plot_filename}'")