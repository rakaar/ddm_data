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
from matplotlib.ticker import NullFormatter, NullLocator, FormatStrFormatter
from sklearn.neighbors import KernelDensity
from collections import defaultdict

plotting_quantiles = np.arange(0.05, 1, 0.1)


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

min_RT_cut_by_ILD = {1: 0.0865, 2: 0.0865, 4: 0.0885, 8: 0.0785, 16: 0.0615}
does_min_RT_depend_on_ILD = True
# %%
from scipy.stats import wilcoxon

def r_squared(y_true, y_pred):
    """Calculate the R-squared value."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    if ss_tot == 0:
        return 1.0  # Perfect fit if there is no variance in y_true
    return 1 - (ss_res / ss_tot)

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
def get_animal_quantile_data(df, ABL, abs_ILD, plot_q_levels, fit_q_levels):
    """Calculates RT quantiles from a pre-filtered DataFrame for a specific condition."""
    df['abs_ILD'] = df['ILD'].abs()
    
    # valid trials for a stim btn 0 and 1 RT
    condition_df = df[(df['ABL'] == ABL) & (df['abs_ILD'] == abs_ILD) & (df['RTwrtStim'] >= 0) & (df['RTwrtStim'] <= 1) & (df['success'].isin([1, -1]))]
    
    n_trials = len(condition_df)

    if n_trials < 5: # Not enough data
        plot_quantiles = np.full(len(plot_q_levels), np.nan)
        fit_quantiles = np.full(len(fit_q_levels), np.nan)
    else:
        rt_series = condition_df['RTwrtStim']
        plot_quantiles = rt_series.quantile(plot_q_levels).values
        fit_quantiles = rt_series.quantile(fit_q_levels).values
        
    return plot_quantiles, fit_quantiles, n_trials

# %%
# params
ABL_arr = [20, 40, 60]
abs_ILD_arr = [1, 2, 4, 8, 16]


# doesn't matter
fitting_quantiles = np.array([0.5])

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
                plot_q, fit_q, n_trials = get_animal_quantile_data(animal_df, abl, abs_ild, plotting_quantiles, fitting_quantiles)
                animal_quantile_data[stim_key] = {
                    'plotting_quantiles': plot_q,
                    'fitting_quantiles': fit_q,
                    'n_trials': n_trials
                }
    except Exception as e:
        print(f"Error processing batch {batch_name}, animal {animal_id}: {str(e)}")
    return batch_animal_pair, animal_quantile_data

# %%
# Group data by animal for efficient parallel processing
animal_groups = merged_data.groupby(['batch_name', 'animal'])

n_jobs = max(1, os.cpu_count() - 4)
print(f"Processing {len(animal_groups)} animal-batch groups on {n_jobs} cores...")

results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(process_batch_animal)(name, group) for name, group in animal_groups if name in batch_animal_pairs
)
quantile_data = {pair: data for pair, data in results if data}
print(f"Completed processing {len(quantile_data)} batch-animal pairs")

# %% 
abl_colors = {20: 'tab:blue', 40: 'tab:orange', 60: 'tab:green'}
quantile_colors = plt.cm.viridis(np.linspace(0, 1, len(plotting_quantiles)))

# --- Aggregators for average (across-animal) plot ---
avg_unscaled_collect = {abl: [] for abl in ABL_arr}

output_filename = f'animal_specific_qq_plots{FILENAME_SUFFIX}.pdf'

with PdfPages(output_filename) as pdf:
    for batch_animal_pair, animal_data in quantile_data.items():
        batch_name, animal_id = batch_animal_pair

        # --- Data Preparation for this animal ---
        unscaled_plot_quantiles = {}
        for abl in ABL_arr:
            plot_q_list = [animal_data.get((abl, ild), {}).get('plotting_quantiles', np.full(len(plotting_quantiles), np.nan)) for ild in abs_ILD_arr]
            unscaled_plot_quantiles[abl] = np.array(plot_q_list).T

        # --- Q-Q Plot: ABL 20/40 vs ABL 60 for each |ILD| ---
        fig, axes = plt.subplots(1, 5, figsize=(25, 5), sharex=True, sharey=True)
        fig.suptitle(f'RT Q-Q Plots - Animal: {animal_id} (Batch: {batch_name})', fontsize=16)

        # Get quantiles for the baseline ABL 60
        q_60_matrix = unscaled_plot_quantiles[60]

        for i, abs_ild in enumerate(abs_ILD_arr):
            ax = axes[i]
            
            # Get quantiles for this specific ILD for all ABLs
            q_60 = q_60_matrix[:, i]
            q_40 = unscaled_plot_quantiles[40][:, i]
            q_20 = unscaled_plot_quantiles[20][:, i]

            # Plot ABL 20 vs 60 and ABL 40 vs 60
            ax.plot(q_60, q_20, marker='o', linestyle='-', color='tab:blue', label='ABL 20 vs 60')
            ax.plot(q_60, q_40, marker='o', linestyle='-', color='tab:orange', label='ABL 40 vs 60')

            # Add a 1:1 reference line
            all_q = np.concatenate([q_60, q_40, q_20])
            min_val = np.nanmin(all_q)
            max_val = np.nanmax(all_q)
            if np.isfinite(min_val) and np.isfinite(max_val):
                ax.plot([min_val, max_val], [min_val, max_val], color='k', linestyle='--', alpha=0.7)

            ax.set_title(f'|ILD| = {abs_ild}')
            ax.set_xlabel('RT Quantiles (ABL 60)')
            if i == 0:
                ax.set_ylabel('RT Quantiles (ABL 20 & 40)')
                ax.legend()
            
            ax.grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

        # Store this animal's matrices for aggregation
        for abl_key in ABL_arr:
            avg_unscaled_collect[abl_key].append(unscaled_plot_quantiles[abl_key])

print(f'PDF saved to {output_filename}')
#%%
# --- Create and Save Average Plot ---

# 1. Calculate average quantiles and SEM across all animals
avg_quantiles = {}
sem_quantiles = {}
for abl in ABL_arr:
    # Stack the list of 2D arrays into a 3D array (animals, quantiles, ilds)
    stacked_arrays = np.stack(avg_unscaled_collect[abl], axis=0)
    # Calculate the mean over the 'animals' axis, ignoring NaNs
    avg_quantiles[abl] = np.nanmean(stacked_arrays, axis=0)
    # Calculate SEM
    n_animals = np.sum(~np.isnan(stacked_arrays), axis=0)
    std_dev = np.nanstd(stacked_arrays, axis=0)
    sem_quantiles[abl] = np.divide(std_dev, np.sqrt(n_animals), where=n_animals > 0)

# 2. Create the average Q-Q plot
fig_avg, axes_avg = plt.subplots(1, 5, figsize=(25, 5), sharex=False, sharey=True)
# fig_avg.suptitle('Average RT Q-Q Plots Across All Animals', fontsize=16)

# Get average quantiles for the baseline ABL 60
q_60_avg_matrix = avg_quantiles[60]
global_min_val = min(min_RT_cut_by_ILD.values())

for i, abs_ild in enumerate(abs_ILD_arr):
    ax = axes_avg[i]
    
    # Get average quantiles for this specific ILD
    q_60_avg = q_60_avg_matrix[:, i]
    q_40_avg = avg_quantiles[40][:, i]
    q_20_avg = avg_quantiles[20][:, i]

    # Plot ABL 20 vs 60 and ABL 40 vs 60
    # Get SEM for this specific ILD
    sem_20 = sem_quantiles[20][:, i]
    sem_40 = sem_quantiles[40][:, i]
    sem_60 = sem_quantiles[60][:, i]

    # Plot ABL 20 vs 60 and ABL 40 vs 60 with error bars
    ax.errorbar(q_60_avg, q_20_avg, xerr=sem_60, yerr=sem_20, marker='o', linestyle='none', color='tab:blue', label='ABL 20 vs 60', capsize=2)
    ax.errorbar(q_60_avg, q_40_avg, xerr=sem_60, yerr=sem_40, marker='o', linestyle='none', color='tab:orange', label='ABL 40 vs 60', capsize=2)

    # Add a consistent 1:1 reference line
    ax.plot([global_min_val, global_max_val], [global_min_val, global_max_val], color='k', linestyle='--', alpha=0.7, zorder=0)

    # --- Publication Grade Styling ---
    ax.set_title('')
    ax.set_xlabel('')
    if i == 0:
        ax.set_ylabel('')
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(axis='both', which='major', labelsize=18)

    lower_lim = min_RT_cut_by_ILD[abs_ild]
    upper_lim = 0.7

    ax.set_xlim(lower_lim, upper_lim)
    ax.set_xticks([lower_lim, upper_lim])
    
    ax.set_ylim(lower_lim, upper_lim)
    ax.set_yticks([lower_lim, upper_lim])

    # Format ticks to 2 decimal places
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax.grid(False)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# 3. Save the plot as a PNG file
avg_output_filename = f'average_qq_plot{FILENAME_SUFFIX}.png'
fig_avg.savefig(avg_output_filename, dpi=300)
plt.show(fig_avg)

print(f'Average plot saved to {avg_output_filename}')

# %% 
# --- Create and Save Average Plot (ABL 40 as baseline) ---

# 1. Create the average Q-Q plot with ABL 40 as baseline
fig_avg_v2, axes_avg_v2 = plt.subplots(1, 5, figsize=(25, 5), sharex=False, sharey=True)

# Determine the global min/max for the 1:1 line to ensure consistency
global_min_val = min(min_RT_cut_by_ILD.values())
global_max_val = 0.5

for i, abs_ild in enumerate(abs_ILD_arr):
    ax = axes_avg_v2[i]
    
    # Get average quantiles for this specific ILD
    q_20_avg = avg_quantiles[20][:, i]
    q_40_avg = avg_quantiles[40][:, i]
    q_60_avg = avg_quantiles[60][:, i]

    lower_lim = min_RT_cut_by_ILD[abs_ild]

    # --- Process and plot for ABL 40 vs 20 ---
    valid_40_20 = ~np.isnan(q_40_avg) & ~np.isnan(q_20_avg)
    if np.any(valid_40_20):
        x_data, y_data = q_40_avg[valid_40_20], q_20_avg[valid_40_20]
        x_sem = sem_quantiles[40][:, i][valid_40_20]
        y_sem = sem_quantiles[20][:, i][valid_40_20]
        mask = (x_data >= lower_lim) & (y_data >= lower_lim)
        ax.errorbar(x_data[mask], y_data[mask], xerr=x_sem[mask], yerr=y_sem[mask], marker='o', linestyle='none', color='tab:blue', capsize=2)
        if np.sum(mask) > 1:
            x_fit, y_fit = x_data[mask], y_data[mask]
            m, c = np.polyfit(x_fit, y_fit, 1)
            y_pred = m * x_fit + c
            r2 = r_squared(y_fit, y_pred)
            fit_x = np.array([lower_lim, 0.5])
            ax.plot(fit_x, m*fit_x + c, color='tab:blue')
            # ax.text(0.05, 0.9, f'$R^2={r2:.2f}$', transform=ax.transAxes, color='tab:blue', fontsize=14)

    # --- Process and plot for ABL 60 vs 20 ---
    valid_40_60 = ~np.isnan(q_40_avg) & ~np.isnan(q_60_avg)
    if np.any(valid_40_60):
        x_data, y_data = q_40_avg[valid_40_60], q_60_avg[valid_40_60]
        x_sem = sem_quantiles[40][:, i][valid_40_60]
        y_sem = sem_quantiles[60][:, i][valid_40_60]
        mask = (x_data >= lower_lim) & (y_data >= lower_lim)
        ax.errorbar(x_data[mask], y_data[mask], xerr=x_sem[mask], yerr=y_sem[mask], marker='o', linestyle='none', color='tab:green', capsize=2)
        if np.sum(mask) > 1:
            x_fit, y_fit = x_data[mask], y_data[mask]
            m, c = np.polyfit(x_fit, y_fit, 1)
            y_pred = m * x_fit + c
            r2 = r_squared(y_fit, y_pred)
            fit_x = np.array([lower_lim, 0.5])
            ax.plot(fit_x, m*fit_x + c, color='tab:green')
            # ax.text(0.05, 0.8, f'$R^2={r2:.2f}$', transform=ax.transAxes, color='tab:green', fontsize=14)

    # Add a consistent 1:1 reference line
    ax.plot([global_min_val, global_max_val], [global_min_val, global_max_val], color='k', linestyle='--', alpha=0.7, zorder=0)

    # --- Publication Grade Styling ---
    ax.set_title('')
    ax.set_xlabel('')
    if i == 0:
        ax.set_ylabel('')
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(axis='both', which='major', labelsize=18)

    upper_lim = 0.5

    ax.set_xlim(lower_lim, upper_lim)
    ax.set_xticks([lower_lim, upper_lim])
    
    ax.set_ylim(lower_lim, upper_lim)
    ax.set_yticks([lower_lim, upper_lim])

    # xlabel
    ax.set_xlabel('RT Quantiles (ABL 40)', fontsize=18)
    

    # Format ticks to 2 decimal places
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax.grid(False)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# 2. Save the new plot as a PNG file
avg_output_filename_v2 = f'average_qq_plot_abl20_baseline{FILENAME_SUFFIX}.png'
fig_avg_v2.savefig(avg_output_filename_v2, dpi=300)
plt.show(fig_avg_v2)

print(f'Average plot (ABL40 on x-axis) saved to {avg_output_filename_v2}')

# %%