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
def get_animal_RTD_data(df, ABL, abs_ILD, bins):
    """Calculates RTD from a pre-filtered DataFrame for a specific condition."""
    df['abs_ILD'] = df['ILD'].abs()
    
    # Filter for the specific condition
    condition_df = df[(df['ABL'] == ABL) & (df['abs_ILD'] == abs_ILD) & (df['RTwrtStim'] >= 0) & (df['RTwrtStim'] <= 1)]
    
    n_trials = len(condition_df)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    if n_trials == 0:
        rtd_hist = np.full_like(bin_centers, np.nan)
    else:
        rtd_hist, _ = np.histogram(condition_df['RTwrtStim'], bins=bins, density=True)
        
    return bin_centers, rtd_hist, n_trials

# %%
# params
ABL_arr = [20, 40, 60]
abs_ILD_arr = [1, 2, 4, 8, 16]
rt_bin_size = 0.02
rt_bins = np.arange(0, 1 + rt_bin_size, rt_bin_size)
min_RT_cut_by_ILD = {1: 0.0865, 2: 0.0865, 4: 0.0885, 8: 0.0785, 16: 0.0615}
does_min_RT_depend_on_ILD = True
min_RT_cut = 0.09 # Fallback for plots/logic not updated to be ILD-specific
max_RT_cut = 0.3
# %%
def process_batch_animal(batch_animal_pair, animal_df):
    """Wrapper function to process a single batch-animal pair using pre-loaded data."""
    batch_name, animal_id = batch_animal_pair
    animal_rtd_data = {}
    try:
        for abl in ABL_arr:
            for abs_ild in abs_ILD_arr:
                stim_key = (abl, abs_ild)
                # Compute RTD histogram
                bin_centers, rtd_hist, n_trials = get_animal_RTD_data(animal_df, abl, abs_ild, rt_bins)

                # Compute quantiles directly from raw RTs (no binning)
                condition_df = animal_df[(animal_df['ABL'] == abl) & (animal_df['ILD'].abs() == abs_ild) &
                                          (animal_df['RTwrtStim'] >= 0) & (animal_df['RTwrtStim'] <= 1) & (animal_df['success'].isin([1, -1]))]
                quantile_levels = np.arange(0.05, 1, 0.01)
                if len(condition_df) > 0:
                    quantiles = condition_df['RTwrtStim'].quantile(quantile_levels).values
                else:
                    quantiles = np.full(len(quantile_levels), np.nan)

                # Store empirical histogram and quantiles together
                animal_rtd_data[stim_key] = {
                    'empirical': {
                        'bin_centers': bin_centers,
                        'rtd_hist': rtd_hist,
                        'n_trials': n_trials
                    },
                    'quantiles': quantiles
                }
    except Exception as e:
        print(f"Error processing batch {batch_name}, animal {animal_id}: {str(e)}")
    return batch_animal_pair, animal_rtd_data

# %%
# Group data by animal for efficient parallel processing
animal_groups = merged_data.groupby(['batch_name', 'animal'])

n_jobs = max(1, os.cpu_count() - 4) # Leave some cores free
print(f"Processing {len(animal_groups)} animal-batch groups on {n_jobs} cores...")

results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(process_batch_animal)(name, group) for name, group in animal_groups if name in batch_animal_pairs
)
rtd_data = {pair: data for pair, data in results if data}
print(f"Completed processing {len(rtd_data)} batch-animal pairs")

# %% 
abl_colors = {20: 'tab:blue', 40: 'tab:orange', 60: 'tab:green'}
output_filename = f'animal_specific_rtd_plots{FILENAME_SUFFIX}_min_RT_{min_RT_cut}_max_RT_{max_RT_cut}_bin_size_{rt_bin_size}.pdf'
all_fit_results = {}
all_scatter_data = {ild: {abl: {'x': [], 'y': []} for abl in [20, 40]} for ild in abs_ILD_arr}

with PdfPages(output_filename) as pdf:
    for batch_animal_pair, animal_data in rtd_data.items():
        batch_name, animal_id = batch_animal_pair
        fig, axes = plt.subplots(1, len(abs_ILD_arr), figsize=(15, 5), sharex=False)
        fig.suptitle(f'Animal: {animal_id} (Batch: {batch_name})', fontsize=16)

        quantile_levels = np.arange(0.01, 1.0, 0.01)
        quantiles_by_abl_ild = {abl: {ild: None for ild in abs_ILD_arr} for abl in ABL_arr}
        fit_results = {ild: {} for ild in abs_ILD_arr}

        for j, abs_ild in enumerate(abs_ILD_arr):
            for abl in ABL_arr:
                stim_key = (abl, abs_ild)
                # Retrieve pre-computed quantiles (calculated in process_batch_animal)
                quant_arr = animal_data.get(stim_key, {}).get('quantiles', np.full_like(quantile_levels, np.nan))
                quantiles_by_abl_ild[abl][abs_ild] = quant_arr

            # Q-Q analysis plots
            ax = axes[j]
            q_60 = quantiles_by_abl_ild[60][abs_ild]
            title_slopes = {}

            # Determine min_RT_cut for this specific ILD
            if does_min_RT_depend_on_ILD:
                current_min_RT_cut = min_RT_cut_by_ILD.get(abs_ild, np.nan)
            else:
                current_min_RT_cut = min_RT_cut
            
            if np.isnan(current_min_RT_cut):
                raise ValueError(f"current_min_RT_cut is nan for abs_ild={abs_ild}")
            
            for abl in [20, 40]:
                q_other = quantiles_by_abl_ild[abl][abs_ild]
                if not np.any(np.isnan(q_60)) and not np.any(np.isnan(q_other)):
                    mask = (q_60 >= current_min_RT_cut) & (q_60 <= max_RT_cut)
                    if np.sum(mask) >= 2:
                        # scatter plot
                        q_other_minus_60 = q_other - q_60
                        # ax.plot(q_60, q_other - q_60, 'o' if abl==20 else 's', color=abl_colors[abl], alpha=0.2)
                        # fit and find slope
                        x_fit_calc = q_60[mask] - current_min_RT_cut
                        y_fit_calc = q_other_minus_60[mask]
                        y_intercept = y_fit_calc[0]
                        y_fit_calc_shifted = y_fit_calc - y_intercept
                        ax.plot(x_fit_calc, y_fit_calc_shifted, 'o' if abl==20 else 's', color=abl_colors[abl], alpha=0.2, lw=1.5)
                        all_scatter_data[abs_ild][abl]['x'].append(x_fit_calc)
                        all_scatter_data[abs_ild][abl]['y'].append(y_fit_calc_shifted)
                        
                        slope = np.sum(x_fit_calc * y_fit_calc_shifted) / np.sum(x_fit_calc**2) if np.sum(x_fit_calc**2) > 0 else np.nan
                        fit_results[abs_ild][abl] = {'slope': slope}
                        title_slopes[abl] = slope

                        # plot fitted line
                        if not np.isnan(slope):
                            x_line_shifted = np.array([0, np.nanmax(x_fit_calc)])
                            y_line_shifted = slope * x_line_shifted
                            ax.plot(x_line_shifted, y_line_shifted, color=abl_colors[abl], label=f'Fit {abl} (m={slope:.3f})', lw=3)

                    else:
                        fit_results[abs_ild][abl] = {'slope': np.nan}
                else:
                    fit_results[abs_ild][abl] = {'slope': np.nan}
            
            # Set title for the Q-Q plot with slopes
            s20 = title_slopes.get(20)
            s40 = title_slopes.get(40)
            title_parts = []
            if s20 is not None and not np.isnan(s20):
                title_parts.append(f'm₂₀={s20:.3f}')
            if s40 is not None and not np.isnan(s40):
                title_parts.append(f'm₄₀={s40:.3f}')
            if title_parts:
                ax.set_title(', '.join(title_parts))

            ax.set_xlim(0, max_RT_cut)
            ax.axhline(0, color='k', linestyle='--')
            if j == 0: ax.set_ylabel('RT Diff (s)')

            ax.set_xlabel('RT (s)')

        # Sync y-axes
        y_lims = [ax.get_ylim() for ax in axes]
        min_y = min(lim[0] for lim in y_lims)
        max_y = max(lim[1] for lim in y_lims)
        for ax in axes:
            ax.set_ylim(min_y, max_y)

        # Store the fit results for this animal before plotting
        all_fit_results[batch_animal_pair] = fit_results

        axes[0].legend()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

print(f'PDF saved to {output_filename}')
# %%
# --- Section for Averaged Plot Across Animals ---

# Create a figure for the averaged plot
fig_avg, axes_avg = plt.subplots(1, len(abs_ILD_arr), figsize=(15, 5), sharex=True, sharey=True)
fig_avg.suptitle('Average Q-Q Across Animals(for Fitting)', fontsize=16)

# Define a common x-axis for interpolation
common_x = np.linspace(0, max_RT_cut, 100) 

for j, abs_ild in enumerate(abs_ILD_arr):
    ax = axes_avg[j]
    ax.set_title(f'|ILD|={abs_ild}')

    for abl in [20, 40]:
        # Collect all slopes for this condition
        slopes = [
            all_fit_results[animal][abs_ild][abl]['slope']
            for animal in all_fit_results
            if abs_ild in all_fit_results[animal] and abl in all_fit_results[animal][abs_ild] and not np.isnan(all_fit_results[animal][abs_ild][abl]['slope'])
        ]
        
        if not slopes:
            continue

        # Average the slopes
        avg_slope = np.mean(slopes)

        # Interpolate and average the scatter data
        all_y_interp = []
        for x_vals, y_vals in zip(all_scatter_data[abs_ild][abl]['x'], all_scatter_data[abs_ild][abl]['y']):
            if len(x_vals) > 1:
                # Use numpy's interpolation function
                y_interp = np.interp(common_x, x_vals, y_vals, left=np.nan, right=np.nan)
                all_y_interp.append(y_interp)
        
        if not all_y_interp:
            continue
            
        # Average the interpolated y-values, ignoring NaNs
        avg_y = np.nanmean(np.array(all_y_interp), axis=0)
        
        # Plot the averaged scatter data
        ax.plot(common_x, avg_y, 'o' if abl == 20 else 's', color=abl_colors[abl], alpha=0.5, markersize=4, label=f'Avg Data {abl}')

        # Plot the line based on the average slope
        x_line = np.array([0, np.nanmax(common_x[~np.isnan(avg_y)])])
        y_line = avg_slope * x_line
        ax.plot(x_line, y_line, color=abl_colors[abl], lw=3, label=f'Avg Fit {abl} (m={avg_slope:.3f})')

    ax.axhline(0, color='k', linestyle='--')
    ax.set_xlabel('RT (s)')
    if j == 0:
        ax.set_ylabel('RT Diff (s)')
    # ax.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save the averaged plot as a PNG
average_plot_filename = 'average_qq_for_fitting_plot.png'
fig_avg.savefig(average_plot_filename)
print(f'Average plot saved to {average_plot_filename}')

plt.show(fig_avg)
