# %%
import pandas as pd
from collections import defaultdict
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
# DESIRED_BATCHES = ['SD', 'LED2', 'LED1', 'LED34', 'LED6', 'LED8', 'LED7', 'LED34_even']
DESIRED_BATCHES = ['SD','LED34', 'LED6', 'LED8', 'LED7', 'LED34_even']
# DESIRED_BATCHES = ['SD','LED34', 'LED6', 'LED8', 'LED7']


# --- Data loading --- 
csv_dir = os.path.join(os.path.dirname(__file__), 'batch_csvs')
batch_files = [f'batch_{batch_name}_valid_and_aborts.csv' for batch_name in DESIRED_BATCHES]

all_data_list = []
for fname in batch_files:
    fpath = os.path.join(csv_dir, fname)
    if os.path.exists(fpath):
        print(f"Loading {fpath}...")
        all_data_list.append(pd.read_csv(fpath))

if not all_data_list:
    raise FileNotFoundError(f"No batch CSV files found for {DESIRED_BATCHES} in '{csv_dir}'")

merged_data = pd.concat(all_data_list, ignore_index=True)

# --- Identify valid trials and batch-animal pairs ---
merged_valid = merged_data[merged_data['success'].isin([1, -1])].copy()
batch_animal_pairs = sorted(list(map(tuple, merged_valid[['batch_name', 'animal']].drop_duplicates().values)))

# --- Print animal table ---
print(f"Found {len(batch_animal_pairs)} batch-animal pairs from {len(set(p[0] for p in batch_animal_pairs))} batches:")
if batch_animal_pairs:
    batch_to_animals = defaultdict(list)
    for batch, animal in sorted(batch_animal_pairs):
        batch_to_animals[batch].append(str(animal))

    max_batch_len = max(len(b) for b in batch_to_animals.keys()) if batch_to_animals else 0
    animal_strings = {b: ', '.join(sorted(a)) for b, a in batch_to_animals.items()}
    max_animals_len = max(len(s) for s in animal_strings.values()) if animal_strings else 0

    print(f"{'Batch':<{max_batch_len}}  {'Animals'}")
    print(f"{'=' * max_batch_len}  {'=' * max_animals_len}")
    for batch, animals_str in sorted(animal_strings.items()):
        print(f"{batch:<{max_batch_len}}  {animals_str}")

# %%
def get_animal_chronometric_data(animal_df, batch_name, animal_id):
    """
    Calculates mean reaction time and standard error for a given animal from its dataframe.
    """
    # Ensure 'abs_ILD' column exists
    if 'ILD' in animal_df.columns:
        animal_df['abs_ILD'] = animal_df['ILD'].abs()
    else:
        print(f"Warning: 'ILD' column not found for animal {animal_id} in {batch_name}")
        return None

    # Filter for valid trials (responded after sound onset) and valid RTs
    df_valid = animal_df[animal_df['success'].isin([1, -1])].copy()
    df_valid = df_valid[(df_valid['RTwrtStim'] >= 0) & (df_valid['RTwrtStim'] <= 1)].copy()

    if df_valid.empty:
        # This check is now mostly redundant if pairs are derived from valid data, but good practice
        # print(f"No valid trials found for animal {animal_id} in {batch_name}")
        return None

    # Calculate mean and SEM for RTwrtStim, grouped by ABL and abs_ILD
    chrono_data = df_valid.groupby(['ABL', 'abs_ILD'])['RTwrtStim'].agg(['mean', 'sem']).reset_index()
    
    return chrono_data

# %%
def process_batch_animal(batch_animal_pair, animal_data_df):
    """
    Wrapper function to process a single batch-animal pair using pre-loaded data.
    """
    batch_name, animal_id = batch_animal_pair
    try:
        chrono_data = get_animal_chronometric_data(animal_data_df, batch_name, animal_id)
        if chrono_data is not None and not chrono_data.empty:
            return (batch_animal_pair, chrono_data)
    except Exception as e:
        print(f"Error processing batch {batch_name}, animal {animal_id}: {str(e)}")
    return None

# %%
# Group data by animal for parallel processing
animal_groups = merged_data.groupby(['batch_name', 'animal'])

# Run processing in parallel
n_jobs = max(1, os.cpu_count() - 4) # Leave some cores free
print(f"Processing {len(animal_groups)} animal-batch groups on {n_jobs} cores...")
results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(process_batch_animal)(name, group) for name, group in animal_groups
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
    # ax.legend(title='ABL', fontsize='x-small')
    
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
print(f'After filtering for ILDs, shape: {all_chrono_data_df.shape}')

# --- Confirm total number of unique animals ---
total_unique_animals = all_chrono_data_df[['batch_name', 'animal_id']].drop_duplicates().shape[0]
print(f"\nTotal unique animals (batch + ID) being plotted: {total_unique_animals}\n")

# %%
# 2. Create the 1x4 subplot figure



fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
plot_abls = [20, 40, 60]
grand_means_data = {}  # To store the mean data for the last plot
keep_grid = False

# 3. Generate first 3 subplots
for i, abl in enumerate(plot_abls):
    ax = axes[i]
    
    # Filter data for the current ABL
    abl_df = all_chrono_data_df[all_chrono_data_df['ABL'] == abl]
    unique_animals_in_abl = abl_df[['batch_name', 'animal_id']].drop_duplicates().shape[0]
    print(f'For ABL = {abl}: shape = {abl_df.shape}, unique animals = {unique_animals_in_abl}')
    
    # Plot individual animal lines in a light color
    for (batch_name, animal_id), animal_df in abl_df.groupby(['batch_name', 'animal_id']):
        animal_df = animal_df.sort_values('abs_ILD')
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
        markersize=8.5,
        capsize=0,
        label='Population Mean'
    )

    # Formatting for each subplot
    ax.set_xlabel('|ILD| (dB)', fontsize=18)
    if i == 0:
        ax.set_ylabel('Mean RT (s)', fontsize=18)
        ax.spines['left'].set_color('black')
        ax.yaxis.label.set_color('black')
        ax.tick_params(axis='y', colors='black')
    else:
        ax.spines['left'].set_color('#bbbbbb')
        ax.tick_params(axis='y', colors='#bbbbbb')

    ax.set_xscale('log')
    ax.set_xticks(abs_ild_ticks)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator()) # Most forceful way to remove minor ticks
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_ylim(0.1, 0.45)
    ax.set_yticks([0.1, 0.2, 0.3, 0.4])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


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
        linewidth=2.5,
        markersize=8.5,
        capsize=0
    )

# Formatting for the 4th subplot
ax4.set_xlabel('|ILD| (dB)', fontsize=18)
# ax4.legend(title='ABL', fontsize=14)
ax4.set_xscale('log')
ax4.set_xticks(abs_ild_ticks)
ax4.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax4.xaxis.set_minor_locator(matplotlib.ticker.NullLocator()) # Most forceful way to remove minor ticks
ax4.tick_params(axis='both', which='major', labelsize=12)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['left'].set_color('#bbbbbb')
ax4.tick_params(axis='y', colors='#bbbbbb')


# 5. Final figure adjustments and saving
plt.tight_layout()
plt.subplots_adjust(bottom=0.15, left=0.07, right=0.97, top=0.95)

summary_plot_filename = os.path.join(output_dir, 'summary_chronometric_plot_by_abl.png')
plt.savefig(summary_plot_filename, dpi=300, bbox_inches='tight')
plt.show(fig)

print(f"Summary plot saved to '{summary_plot_filename}'")

# %%
# --- New Plots: Mean RT vs ABL and Mean RT vs |ILD| ---

# Plot 1: Mean RT vs ABL (collapsing across |ILD|)
fig_abl, ax_abl = plt.subplots(figsize=(5, 5))

# Aggregate data for specific ABLs
plot_abls = [20, 40, 60]
rt_vs_abl = all_chrono_data_df[all_chrono_data_df['ABL'].isin(plot_abls)].groupby('ABL')['mean'].agg(['mean', 'sem']).reset_index()

ax_abl.errorbar(
    x=range(len(rt_vs_abl)),  # Plot against indices for discrete points
    y=rt_vs_abl['mean'],
    yerr=rt_vs_abl['sem'],
    fmt='o',  # Use 'o' for scatter plot markers
    linestyle='None',  # Do not connect markers with a line
    color='k',
    capsize=5,
    markersize=8
)

ax_abl.set_xticks(range(len(rt_vs_abl)))
ax_abl.set_xticklabels(rt_vs_abl['ABL'].astype(int))

ax_abl.set_xlabel('ABL (dB)', fontsize=18)
# ax_abl.set_ylabel('Mean RT (s)', fontsize=18)
# ax_abl.set_title('Mean Reaction Time vs. ABL', fontsize=16)
ax_abl.set_yticks([0.18, 0.28])
ax_abl.spines['top'].set_visible(False)
ax_abl.spines['right'].set_visible(False)
ax_abl.tick_params(axis='both', which='major', labelsize=18)

plt.tight_layout()
rt_vs_abl_filename = 'summary_rt_vs_abl.png'
plt.savefig(rt_vs_abl_filename, dpi=300)
plt.show()

print(f"RT vs ABL plot saved to '{rt_vs_abl_filename}'")

# %%
# Plot 2: Mean RT vs |ILD| (collapsing across ABL)
fig_ild, ax_ild = plt.subplots(figsize=(5, 4))

# Aggregate data by abs_ILD
rt_vs_ild = all_chrono_data_df.groupby('abs_ILD')['mean'].agg(['mean', 'sem']).reset_index()

ax_ild.errorbar(
    x=rt_vs_ild['abs_ILD'],
    y=rt_vs_ild['mean'],
    yerr=rt_vs_ild['sem'],
    fmt='o',
    color='k',
    capsize=0, # No caps as requested in memory
    markersize=8,
    linewidth=2
)

ax_ild.set_xlabel('|ILD| (dB)', fontsize=18)
ax_ild.set_ylabel('Mean RT (s)', fontsize=18)
# ax_ild.set_title('Mean Reaction Time vs. |ILD|', fontsize=16)
ax_ild.set_xscale('log')
ax_ild.set_xticks(abs_ild_ticks)
ax_ild.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax_ild.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax_ild.spines['top'].set_visible(False)
ax_ild.spines['right'].set_visible(False)
ax_ild.tick_params(axis='both', which='major', labelsize=18)
ax_ild.set_yticks([0.15, 0.28])

plt.tight_layout()
rt_vs_ild_filename = 'summary_rt_vs_ild.png'
plt.savefig(rt_vs_ild_filename, dpi=300)
plt.show()

print(f"RT vs |ILD| plot saved to '{rt_vs_ild_filename}'")

# %%
