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
plotting_quantiles = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
fitting_quantiles = np.arange(0.01, 1.0, 0.01)
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
avg_scaled_collect = {abl: [] for abl in ABL_arr}

output_filename = f'animal_specific_quantile_scaling_plots{FILENAME_SUFFIX}.pdf'

with PdfPages(output_filename) as pdf:
    for batch_animal_pair, animal_data in quantile_data.items():
        batch_name, animal_id = batch_animal_pair

        # --- Data Preparation for this animal ---
        # For each ABL, get a 2D array of quantiles vs |ILD|
        # Shape: (num_quantiles, num_ilds)
        unscaled_plot_quantiles = {}
        fitting_quantiles_all = {}
        for abl in ABL_arr:
            # Get quantiles for each ILD, resulting in a list of arrays
            plot_q_list = [animal_data.get((abl, ild), {}).get('plotting_quantiles', np.full(len(plotting_quantiles), np.nan)) for ild in abs_ILD_arr]
            fit_q_list = [animal_data.get((abl, ild), {}).get('fitting_quantiles', np.full(len(fitting_quantiles), np.nan)) for ild in abs_ILD_arr]
            # Stack them into a 2D array and transpose
            unscaled_plot_quantiles[abl] = np.array(plot_q_list).T
            fitting_quantiles_all[abl] = np.array(fit_q_list).T

        # --- Plot 1: Unscaled Quantiles ---
        fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
        fig1.suptitle(f'Original RT Quantiles vs. |ILD| - Animal: {animal_id} (Batch: {batch_name})', fontsize=16)

        for i, abl in enumerate(ABL_arr):
            ax = axes1[i]
            q_matrix = unscaled_plot_quantiles[abl]
            for j, q_level in enumerate(plotting_quantiles):
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
        scaled_plot_quantiles = {60: unscaled_plot_quantiles[60]} # ABL 60 is the baseline
        slopes = {abl: [] for abl in [20, 40]}

        # Use the fine-grained quantiles for fitting
        q_60_fit_all_ilds = fitting_quantiles_all[60]

        for abl in [20, 40]:
            q_other_fit_all_ilds = fitting_quantiles_all[abl]
            q_other_plot_all_ilds = unscaled_plot_quantiles[abl]
            scaled_q_abl_per_ild = []
            
            for ild_idx, abs_ild in enumerate(abs_ILD_arr):
                # Use fitting quantiles for slope calculation
                q_60_fit = q_60_fit_all_ilds[:, ild_idx]
                q_other_fit = q_other_fit_all_ilds[:, ild_idx]

                # --- Slope calculation using the original method with fine quantiles ---
                slope = np.nan # Default slope
                if not np.any(np.isnan(q_60_fit)) and not np.any(np.isnan(q_other_fit)):
                    mask = (q_60_fit >= min_RT_cut) & (q_60_fit <= max_RT_cut)
                    if np.sum(mask) >= 2:
                        q_other_minus_60 = q_other_fit - q_60_fit
                        
                        x_fit_calc = q_60_fit[mask] - min_RT_cut
                        y_fit_calc = q_other_minus_60[mask]
                        y_intercept = y_fit_calc[0]
                        y_fit_calc_shifted = y_fit_calc - y_intercept
                        
                        if np.sum(x_fit_calc**2) > 0:
                            slope = np.sum(x_fit_calc * y_fit_calc_shifted) / np.sum(x_fit_calc**2)
                
                slopes[abl].append(slope)

                # --- Apply scaling transformation to the coarse plotting quantiles ---
                q_other_plot = q_other_plot_all_ilds[:, ild_idx]
                if not np.isnan(slope) and (1 + slope) != 0:
                    scaled_values = np.where(
                        q_other_plot > min_RT_cut,
                        ((q_other_plot - min_RT_cut) / (1 + slope)) + min_RT_cut,
                        q_other_plot
                    )
                    scaled_q_abl_per_ild.append(scaled_values)
                else:
                    scaled_q_abl_per_ild.append(q_other_plot)
            
            scaled_plot_quantiles[abl] = np.array(scaled_q_abl_per_ild).T

        # Store this animal's matrices for aggregation
        for abl_key in ABL_arr:
            avg_unscaled_collect[abl_key].append(unscaled_plot_quantiles[abl_key])
            avg_scaled_collect[abl_key].append(scaled_plot_quantiles[abl_key])

        # --- Plot 2: Scaled Quantiles ---
        fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
        fig2.suptitle(f'Scaled RT Quantiles vs. |ILD| - Animal: {animal_id} (Batch: {batch_name})', fontsize=16)

        for i, abl in enumerate(ABL_arr):
            ax = axes2[i]
            q_matrix = scaled_plot_quantiles[abl]
            for j, q_level in enumerate(plotting_quantiles):
                ax.plot(abs_ILD_arr, q_matrix[j, :], marker='o', linestyle='-', color=quantile_colors[j], label=f'{int(q_level*100)}th')
            
            title = f'ABL = {abl} dB'
            if abl != 60:
                # Create a summary of slopes for the title
                slope_str = ', '.join([f'{s:.3f}' for s in slopes[abl]])
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

# ================= Average-Across-Animals Plot =================
print("Generating average (across animals) quantile plot…")

avg_output_filename = f'average_quantile_scaling_plots{FILENAME_SUFFIX}.pdf'

# Compute mean and SEM per ABL, |ILD|, quantile
mean_unscaled, sem_unscaled = {}, {}
mean_scaled, sem_scaled = {}, {}

for abl in ABL_arr:
    data_unscaled = np.array(avg_unscaled_collect[abl])  # (num_animals, Q, num_ilds)
    data_scaled   = np.array(avg_scaled_collect[abl])

    mean_unscaled[abl] = np.nanmean(data_unscaled, axis=0)
    mean_scaled[abl]   = np.nanmean(data_scaled,   axis=0)

    n_unscaled = np.sum(~np.isnan(data_unscaled), axis=0)
    n_scaled   = np.sum(~np.isnan(data_scaled),   axis=0)

    sem_unscaled[abl] = np.nanstd(data_unscaled, axis=0, ddof=0) / np.sqrt(np.where(n_unscaled==0, np.nan, n_unscaled))
    sem_scaled[abl]   = np.nanstd(data_scaled,   axis=0, ddof=0) / np.sqrt(np.where(n_scaled==0,   np.nan, n_scaled))

with PdfPages(avg_output_filename) as pdf_avg:
    fig_avg, axes_avg = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
    fig_avg.suptitle('Average RT Quantiles Across Animals', fontsize=18)

    # Row 0: original
    for col, abl in enumerate(ABL_arr):
        ax = axes_avg[0, col]
        q_mat = mean_unscaled[abl]
        sem_mat = sem_unscaled[abl]
        for q_idx, q_level in enumerate(plotting_quantiles):
            ax.errorbar(abs_ILD_arr, q_mat[q_idx, :], yerr=sem_mat[q_idx, :], marker='o', linestyle='-', color=quantile_colors[q_idx], label=f'{int(q_level*100)}th')
        ax.set_title(f'ABL = {abl} dB (Original)')
        ax.set_xscale('log')
        ax.set_xticks(abs_ILD_arr)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        if col == 0:
            ax.set_ylabel('Reaction Time (s)')
            ax.legend(title='Quantile')

    # Row 1: scaled
    for col, abl in enumerate(ABL_arr):
        ax = axes_avg[1, col]
        q_mat = mean_scaled[abl]
        sem_mat = sem_scaled[abl]
        for q_idx, q_level in enumerate(plotting_quantiles):
            ax.errorbar(abs_ILD_arr, q_mat[q_idx, :], yerr=sem_mat[q_idx, :], marker='o', linestyle='-', color=quantile_colors[q_idx], label=f'{int(q_level*100)}th')
        ax.set_title(f'ABL = {abl} dB (Scaled)')
        ax.set_xscale('log')
        ax.set_xticks(abs_ILD_arr)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        if col == 0:
            ax.set_ylabel('Reaction Time (s) (Scaled)')

    for ax in axes_avg[-1, :]:
        ax.set_xlabel('|ILD|')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf_avg.savefig(fig_avg)
    plt.close(fig_avg)

print(f'Average-across-animals PDF saved to {avg_output_filename}')

# %%
# --------- Overlay figure of scaled quantiles (all ABLs on one axes) ---------
overlay_filename = f'average_scaled_quantiles_overlay{FILENAME_SUFFIX}.pdf'
abl_markers = {20: 'o', 40: 's', 60: '^'}

with PdfPages(overlay_filename) as pdf_overlay:
    fig_overlay, ax_overlay = plt.subplots(figsize=(6, 4))
    fig_overlay.suptitle('Scaled RT Quantiles – Overlay Across ABLs', fontsize=16)

    for abl in ABL_arr:
        q_mat = mean_scaled[abl]
        sem_mat = sem_scaled[abl]
        for q_idx, q_level in enumerate(plotting_quantiles):
            ax_overlay.errorbar(
                abs_ILD_arr,
                q_mat[q_idx, :],
                yerr=sem_mat[q_idx, :],
                marker=abl_markers[abl],
                linestyle='-',
                color=quantile_colors[q_idx],
                alpha=0.8
            )

    ax_overlay.set_xscale('log')
    ax_overlay.set_xticks(abs_ILD_arr)
    ax_overlay.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax_overlay.set_xlabel('|ILD|')
    ax_overlay.set_ylabel('Reaction Time (s) (Scaled)')

    # Legends
    from matplotlib.lines import Line2D
    quantile_handles = [Line2D([0], [0], color=quantile_colors[i], marker='o', linestyle='-', label=f'{int(q*100)}th') for i, q in enumerate(plotting_quantiles)]
    abl_handles = [Line2D([0], [0], color='k', marker=abl_markers[abl], linestyle='None', label=f'ABL {abl} dB') for abl in ABL_arr]
    first_legend = ax_overlay.legend(handles=quantile_handles, title='Quantile', loc='upper left', bbox_to_anchor=(1.02, 1))
    ax_overlay.add_artist(first_legend)
    ax_overlay.legend(handles=abl_handles, title='ABL', loc='lower left', bbox_to_anchor=(1.02, 0))

    plt.tight_layout()
    pdf_overlay.savefig(fig_overlay)
    plt.close(fig_overlay)

print(f'Scaled-overlay quantiles PDF saved to {overlay_filename}')
