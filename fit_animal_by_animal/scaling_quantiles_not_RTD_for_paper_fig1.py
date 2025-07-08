# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from joblib import Parallel, delayed
from tqdm import tqdm
import pickle
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.neighbors import KernelDensity

# Flag to include abort_event == 4. If True, data with these aborts is loaded
# and filenames are updated accordingly.
INCLUDE_ABORT_EVENT_4 = False

if INCLUDE_ABORT_EVENT_4:
    CSV_SUFFIX = '_and_4'
    ABORT_EVENTS = [3, 4]
    FILENAME_SUFFIX = '_with_abort4'
else:
    CSV_SUFFIX = ''
    ABORT_EVENTS = [3]
    FILENAME_SUFFIX = ''

# %%
from collections import defaultdict

# Define desired batches and paths
DESIRED_BATCHES = ['SD', 'LED34', 'LED6', 'LED8', 'LED7', 'LED34_even'] # Excluded LED1 as per original logic
csv_dir = os.path.join(os.path.dirname(__file__), 'batch_csvs')

# --- Data loading ---
batch_files = [f'batch_{batch_name}_valid_and_aborts{CSV_SUFFIX}.csv' for batch_name in DESIRED_BATCHES]
all_data_list = []
for fname in batch_files:
    fpath = os.path.join(csv_dir, fname)
    if os.path.exists(fpath):
        print(f"Loading {fpath}...")
        all_data_list.append(pd.read_csv(fpath))

if not all_data_list:
    raise FileNotFoundError(f"No batch CSV files found for {DESIRED_BATCHES} in '{csv_dir}' with suffix '{CSV_SUFFIX}'")

merged_data = pd.concat(all_data_list, ignore_index=True)

# --- Identify valid trials and batch-animal pairs ---
# Note: We use all data for RTD, not just 'success' trials, so we derive pairs from the full dataset.

# Get initial pairs from CSV data
base_pairs = set(map(tuple, merged_data[['batch_name', 'animal']].drop_duplicates().values))

# Apply specific exclusion logic from the original script
excluded_animals_led2 = {40, 41, 43}

batch_animal_pairs = sorted([
    (batch, animal) for batch, animal in base_pairs 
    if not (batch == 'LED2' and animal in excluded_animals_led2)
])

# --- Print animal table for verification ---
print(f"Found {len(batch_animal_pairs)} batch-animal pairs from {len(set(p[0] for p in batch_animal_pairs))} batches:")
if batch_animal_pairs:
    batch_to_animals = defaultdict(list)
    for batch, animal in sorted(batch_animal_pairs):
        batch_to_animals[batch].append(str(animal))

    max_batch_len = max(len(b) for b in batch_to_animals.keys()) if batch_to_animals else 0
    animal_strings = {b: ', '.join(sorted(a, key=int)) for b, a in batch_to_animals.items()}
    max_animals_len = max(len(s) for s in animal_strings.values()) if animal_strings else 0

    print(f"{'Batch':<{max_batch_len}}  {'Animals'}")
    print(f"{'=' * max_batch_len}  {'=' * max_animals_len}")
    for batch, animals_str in sorted(animal_strings.items()):
        print(f"{batch:<{max_batch_len}}  {animals_str}")


# %%
def get_animal_quantile_data(df, ABL, abs_ILD, quantile_levels):
    """Calculates RT quantiles from a pre-filtered DataFrame for a specific condition."""
    df['abs_ILD'] = df['ILD'].abs()
    
    # Filter for the specific condition and valid RTs
    condition_df = df[(df['ABL'] == ABL) & (df['abs_ILD'] == abs_ILD) & (df['RTwrtStim'] >= 0) & (df['RTwrtStim'] <= 1)]
    
    n_trials = len(condition_df)

    if n_trials < len(quantile_levels): # Not enough data to compute quantiles
        quantiles = np.full(len(quantile_levels), np.nan)
    else:
        quantiles = condition_df['RTwrtStim'].quantile(quantile_levels).values
        
    return quantiles, n_trials

# %%
# params
ABL_arr = [20, 40, 60]
abs_ILD_arr = [1, 2, 4, 8, 16]
quantile_levels = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
min_RT_cut = 0.09 # for slope fitting
max_RT_cut = 0.3 # for slope fitting
# %%
def process_batch_animal(batch_animal_pair, animal_df):
    """Wrapper function to process a single batch-animal pair using pre-loaded data."""
    batch_name, animal_id = batch_animal_pair
    animal_quantile_data = {}
    try:
        for abl in ABL_arr:
            for abs_ild in abs_ILD_arr:
                stim_key = (abl, abs_ild)
                quantiles, n_trials = get_animal_quantile_data(animal_df, abl, abs_ild, quantile_levels)
                animal_quantile_data[stim_key] = {
                    'quantiles': quantiles,
                    'n_trials': n_trials
                }
    except Exception as e:
        print(f"Error processing batch {batch_name}, animal {animal_id}: {str(e)}")
    return batch_animal_pair, animal_quantile_data

# %%
# Group data by animal for efficient parallel processing
animal_groups = merged_data.groupby(['batch_name', 'animal'])

n_jobs = max(1, os.cpu_count() - 4) # Leave some cores free
print(f"Processing {len(animal_groups)} animal-batch groups on {n_jobs} cores...")

results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(process_batch_animal)(name, group) for name, group in animal_groups if name in batch_animal_pairs
)
quantile_data = {pair: data for pair, data in results if data}
print(f"Completed processing {len(quantile_data)} batch-animal pairs")

# %% 
abl_colors = {20: 'purple', 40: 'mediumseagreen', 60: 'darkorange'}
quantile_colors = plt.cm.viridis(np.linspace(0, 1, len(quantile_levels)))

output_filename = f'animal_specific_quantile_scaling_plots{FILENAME_SUFFIX}.pdf'

with PdfPages(output_filename) as pdf:
    for batch_animal_pair, animal_data in quantile_data.items():
        batch_name, animal_id = batch_animal_pair

        # --- Data Preparation for this animal ---
        # For each ABL, get a 2D array of quantiles vs |ILD|
        # Shape: (num_quantiles, num_ilds)
        unscaled_quantiles = {}
        for abl in ABL_arr:
            # Get quantiles for each ILD, resulting in a list of arrays
            q_vs_ild_list = [animal_data.get((abl, ild), {}).get('quantiles', np.full(len(quantile_levels), np.nan)) for ild in abs_ILD_arr]
            # Stack them into a 2D array and transpose
            unscaled_quantiles[abl] = np.array(q_vs_ild_list).T

        # --- Plot 1: Unscaled Quantiles ---
        fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
        fig1.suptitle(f'Unscaled RT Quantiles vs. |ILD| - Animal: {animal_id} (Batch: {batch_name})', fontsize=16)

        for i, abl in enumerate(ABL_arr):
            ax = axes1[i]
            q_matrix = unscaled_quantiles[abl]
            for j, q_level in enumerate(quantile_levels):
                ax.plot(abs_ILD_arr, q_matrix[j, :], marker='o', linestyle='-', color=quantile_colors[j], label=f'{int(q_level*100)}th')
            ax.set_title(f'ABL = {abl} dB')
            ax.set_xlabel('|ILD|')
            if i == 0:
                ax.set_ylabel('Reaction Time (s)')
                ax.legend(title='Quantile')
            ax.set_xscale('log')
            ax.set_xticks(abs_ILD_arr)
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig1)
        plt.close(fig1)

        # --- Scaling Calculation (Corrected based on original script) ---
        scaled_quantiles = {60: unscaled_quantiles[60]} # ABL 60 is the baseline
        slopes = {abl: [] for abl in [20, 40]}

        q_60_all_ilds = unscaled_quantiles[60]

        for abl in [20, 40]:
            q_other_all_ilds = unscaled_quantiles[abl]
            scaled_q_abl_per_ild = []
            
            for ild_idx, abs_ild in enumerate(abs_ILD_arr):
                q_60 = q_60_all_ilds[:, ild_idx]
                q_other = q_other_all_ilds[:, ild_idx]

                # --- Slope calculation using the original method ---
                slope = np.nan # Default slope
                if not np.any(np.isnan(q_60)) and not np.any(np.isnan(q_other)):
                    mask = (q_60 >= min_RT_cut) & (q_60 <= max_RT_cut)
                    if np.sum(mask) >= 2:
                        q_other_minus_60 = q_other - q_60
                        
                        x_fit_calc = q_60[mask] - min_RT_cut
                        y_fit_calc = q_other_minus_60[mask]
                        y_intercept = y_fit_calc[0]
                        y_fit_calc_shifted = y_fit_calc - y_intercept
                        
                        # Calculate slope for (q_other - q_60) vs q_60
                        if np.sum(x_fit_calc**2) > 0:
                            slope = np.sum(x_fit_calc * y_fit_calc_shifted) / np.sum(x_fit_calc**2)
                
                slopes[abl].append(slope)

                # --- Apply scaling transformation ---
                if not np.isnan(slope) and (1 + slope) != 0:
                    # Apply scaling: RT_scaled = ((RT_unscaled - min_RT_cut) / (1 + slope)) + min_RT_cut
                    # This transformation is only applied where the original quantile is > min_RT_cut
                    scaled_values = np.where(
                        q_other > min_RT_cut,
                        ((q_other - min_RT_cut) / (1 + slope)) + min_RT_cut,
                        q_other # Keep original value if below cut-off
                    )
                    scaled_q_abl_per_ild.append(scaled_values)
                else:
                    # If slope is invalid, use original, unscaled quantiles
                    scaled_q_abl_per_ild.append(q_other)
            
            # Transpose at the end to get (num_quantiles, num_ilds)
            scaled_quantiles[abl] = np.array(scaled_q_abl_per_ild).T

        # --- Plot 2: Scaled Quantiles ---
        fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
        fig2.suptitle(f'Scaled RT Quantiles vs. |ILD| - Animal: {animal_id} (Batch: {batch_name})', fontsize=16)

        for i, abl in enumerate(ABL_arr):
            ax = axes2[i]
            q_matrix = scaled_quantiles[abl]
            for j, q_level in enumerate(quantile_levels):
                ax.plot(abs_ILD_arr, q_matrix[j, :], marker='o', linestyle='-', color=quantile_colors[j], label=f'{int(q_level*100)}th')
            
            title = f'ABL = {abl} dB'
            if abl != 60:
                # Create a summary of slopes for the title
                slope_str = ', '.join([f'{s:.2f}' for s in slopes[abl]])
                title += f'\n(Slopes: {slope_str})'
            ax.set_title(title)
            
            ax.set_xlabel('|ILD|')
            if i == 0:
                ax.set_ylabel('Reaction Time (s) (Scaled to ABL 60)')
                ax.legend(title='Quantile')
            ax.set_xscale('log')
            ax.set_xticks(abs_ILD_arr)
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

        # Sync Y-axis across all scaled plots for better comparison
        all_y_lims = [ax.get_ylim() for ax in axes2]
        min_y = min(lim[0] for lim in all_y_lims)
        max_y = max(lim[1] for lim in all_y_lims)
        for ax in axes2:
            ax.set_ylim(min_y, max_y)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig2)
        plt.close(fig2)

print(f'PDF saved to {output_filename}')
