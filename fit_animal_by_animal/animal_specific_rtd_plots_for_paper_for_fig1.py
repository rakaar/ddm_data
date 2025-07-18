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
                quantile_levels = np.arange(0.01, 1.0, 0.01)
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

with PdfPages(output_filename) as pdf:
    for batch_animal_pair, animal_data in rtd_data.items():
        batch_name, animal_id = batch_animal_pair
        fig, axes = plt.subplots(3, len(abs_ILD_arr), figsize=(15, 12), sharex=False)
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

            # Row 1: Original RTDs
            ax1 = axes[0, j]
            for abl in ABL_arr:
                emp_data = animal_data.get((abl, abs_ild), {}).get('empirical', {})
                if emp_data.get('n_trials', 0) > 0:
                    ax1.plot(emp_data['bin_centers'], emp_data['rtd_hist'], color=abl_colors[abl], lw=1.5, label=f'ABL={abl}')
            ax1.set_title(f'|ILD|={abs_ild}')
            if j == 0: ax1.set_ylabel('Density')
            ax1.set_xlim(0, 0.7)
            ax1.tick_params(axis='x', labelbottom=False)

            # Row 2: Q-Q analysis plots
            ax2 = axes[1, j]
            q_60 = quantiles_by_abl_ild[60][abs_ild]
            title_slopes = {}

            # Determine min_RT_cut for this specific ILD
            if does_min_RT_depend_on_ILD:
                current_min_RT_cut = min_RT_cut_by_ILD.get(abs_ild, np.nan)
            else:
                current_min_RT_cut = min_RT_cut
            for abl in [20, 40]:
                q_other = quantiles_by_abl_ild[abl][abs_ild]
                if not np.any(np.isnan(q_60)) and not np.any(np.isnan(q_other)):
                    mask = (q_60 >= current_min_RT_cut) & (q_60 <= max_RT_cut)
                    if np.sum(mask) >= 2:
                        # scatter plot
                        q_other_minus_60 = q_other - q_60
                        # ax2.plot(q_60, q_other - q_60, 'o' if abl==20 else 's', color=abl_colors[abl], alpha=0.2)
                        ax2.plot(q_60[mask] - current_min_RT_cut, q_other_minus_60[mask] - q_other_minus_60[mask][0], 'o' if abl==20 else 's', color=abl_colors[abl], alpha=0.2, lw=1.5)
                        # fit and find slope
                        x_fit_calc = q_60[mask] - current_min_RT_cut
                        y_fit_calc = q_other_minus_60[mask]
                        y_intercept = y_fit_calc[0]
                        y_fit_calc_shifted = y_fit_calc - y_intercept
                        
                        slope = np.sum(x_fit_calc * y_fit_calc_shifted) / np.sum(x_fit_calc**2) if np.sum(x_fit_calc**2) > 0 else np.nan
                        fit_results[abs_ild][abl] = {'slope': slope}
                        title_slopes[abl] = slope

                        # plot fitted line
                        if not np.isnan(slope):
                            x_line_shifted = np.array([0, np.nanmax(x_fit_calc)])
                            y_line_shifted = slope * x_line_shifted
                            ax2.plot(x_line_shifted, y_line_shifted, color=abl_colors[abl], label=f'Fit {abl} (m={slope:.3f})', lw=3)

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
                ax2.set_title(', '.join(title_parts))

            ax2.set_xlim(0, max_RT_cut)
            ax2.axhline(0, color='k', linestyle='--')
            if j == 0: ax2.set_ylabel('RT Diff (s)')

            # Row 3: Rescaled RTDs
            ax3 = axes[2, j]
            for abl in ABL_arr:
                emp_data = animal_data.get((abl, abs_ild), {}).get('empirical', {})
                if emp_data.get('n_trials', 0) > 0:
                    bin_centers = emp_data['bin_centers']
                    rtd_hist = emp_data['rtd_hist']
                    if abl == 60:
                        ax3.plot(bin_centers, rtd_hist, color=abl_colors[abl], lw=1.5)
                    else:
                        slope = fit_results[abs_ild].get(abl, {}).get('slope')
                        if slope is not None and not np.isnan(slope) and (slope + 1) != 0:
                            xvals = np.where(bin_centers > current_min_RT_cut, ((bin_centers - current_min_RT_cut) / (1 + slope)) + current_min_RT_cut, bin_centers)
                            multiplier = np.ones_like(rtd_hist)
                            multiplier[bin_centers > current_min_RT_cut] = slope + 1
                            rescaled_rtd = rtd_hist * multiplier
                            ax3.plot(xvals, rescaled_rtd, color=abl_colors[abl], lw=1.5)
                        else:
                            ax3.plot(bin_centers, rtd_hist, color=abl_colors[abl], lw=1.5, linestyle=':') # Plot original as dotted if no fit
            ax3.set_xlim(0, 0.7)
            ax3.set_xlabel('RT (s)')
            if j == 0: ax3.set_ylabel('Density (Rescaled)')

        # Sync y-axes for each row
        for i in range(3): # For each row
            y_lims = [ax.get_ylim() for ax in axes[i, :]]
            min_y = min(lim[0] for lim in y_lims)
            max_y = max(lim[1] for lim in y_lims)
            for ax in axes[i, :]:
                ax.set_ylim(min_y, max_y)

        # Store the fit results for this animal before plotting
        all_fit_results[batch_animal_pair] = fit_results

        axes[0, 0].legend()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

print(f'PDF saved to {output_filename}')

# %%

# %% Code for average animal plot

print("Generating average animal plot...")

# --- Data Aggregation ---
aggregated_rtds = {(abl, ild): [] for abl in ABL_arr for ild in abs_ILD_arr}
aggregated_rescaled_data = {(abl, ild): [] for abl in ABL_arr for ild in abs_ILD_arr}
quantile_levels = np.arange(0.01, 1.0, 0.01)

for batch_animal_pair, animal_data in tqdm(rtd_data.items(), desc="Aggregating data for average plot"):
    # Recalculate slopes for this animal (same logic as in the per-animal plot)
    quantiles_by_abl_ild = {abl: {ild: None for ild in abs_ILD_arr} for abl in ABL_arr}
    fit_results = {ild: {} for ild in abs_ILD_arr}

    for j, abs_ild in enumerate(abs_ILD_arr):
        for abl in ABL_arr:
            stim_key = (abl, abs_ild)
            # Retrieve pre-computed quantiles (calculated in process_batch_animal)
            quant_arr = animal_data.get(stim_key, {}).get('quantiles', np.full_like(quantile_levels, np.nan))
            quantiles_by_abl_ild[abl][abs_ild] = quant_arr

        # Determine min_RT_cut for this specific ILD
        if does_min_RT_depend_on_ILD:
            current_min_RT_cut = min_RT_cut_by_ILD.get(abs_ild, np.nan)
        else:
            current_min_RT_cut = min_RT_cut

        q_60 = quantiles_by_abl_ild[60][abs_ild]
        for abl in [20, 40]:
            q_other = quantiles_by_abl_ild[abl][abs_ild]
            if not np.any(np.isnan(q_60)) and not np.any(np.isnan(q_other)):
                mask = (q_60 >= current_min_RT_cut) & (q_60 <= max_RT_cut)
                if np.sum(mask) >= 2:
                    q_other_minus_60 = q_other - q_60
                    x_fit_calc = q_60[mask] - current_min_RT_cut
                    y_fit_calc = q_other_minus_60[mask]
                    y_intercept = y_fit_calc[0]
                    y_fit_calc_shifted = y_fit_calc - y_intercept
                    slope = np.sum(x_fit_calc * y_fit_calc_shifted) / np.sum(x_fit_calc**2) if np.sum(x_fit_calc**2) > 0 else np.nan
                    fit_results[abs_ild][abl] = {'slope': slope}
                else:
                    fit_results[abs_ild][abl] = {'slope': np.nan}
            else:
                fit_results[abs_ild][abl] = {'slope': np.nan}

    # Aggregate original and rescaled RTDs
    for abs_ild in abs_ILD_arr:
        for abl in ABL_arr:
            stim_key = (abl, abs_ild)
            emp_data = animal_data.get(stim_key, {}).get('empirical', {})
            
            if emp_data and emp_data.get('n_trials', 0) > 0:
                bin_centers = emp_data['bin_centers']
                rtd_hist = emp_data['rtd_hist']
                
                # Store original RTD
                aggregated_rtds[stim_key].append(rtd_hist)

                # Calculate and store rescaled RTD
                if abl == 60:
                    rescaled_rtd = rtd_hist
                    xvals = bin_centers
                else:
                    slope = fit_results[abs_ild].get(abl, {}).get('slope')
                    if slope is not None and not np.isnan(slope) and (slope + 1) != 0:
                        xvals = np.where(bin_centers > current_min_RT_cut, ((bin_centers - current_min_RT_cut) / (1 + slope)) + current_min_RT_cut, bin_centers)
                        multiplier = np.ones_like(rtd_hist)
                        multiplier[bin_centers > current_min_RT_cut] = slope + 1
                        rescaled_rtd = rtd_hist * multiplier
                    else: # no fit, use original
                        rescaled_rtd = rtd_hist
                        xvals = bin_centers
                
                # Interpolate rescaled RTD back to the common grid (bin_centers) before aggregating
                interp_rescaled_rtd = np.interp(bin_centers, xvals, rescaled_rtd, left=0, right=0)
                aggregated_rescaled_data[stim_key].append(interp_rescaled_rtd)
            else:
                nan_hist = np.full(len(rt_bins) - 1, np.nan)
                aggregated_rtds[stim_key].append(nan_hist)
                aggregated_rescaled_data[stim_key].append(nan_hist)

# --- Plotting ---
avg_output_filename = f'average_animal_rtd_plots{FILENAME_SUFFIX}_min_RT_{min_RT_cut}_max_RT_{max_RT_cut}_bin_size_{rt_bin_size}.pdf'
with PdfPages(avg_output_filename) as pdf:
    fig, axes = plt.subplots(2, len(abs_ILD_arr), figsize=(15, 8), sharex='col', sharey='row')
    fig.suptitle('Average Animal RTD Analysis', fontsize=16)

    for j, abs_ild in enumerate(abs_ILD_arr):
        ax1 = axes[0, j]
        ax2 = axes[1, j]

        for abl in ABL_arr:
            stim_key = (abl, abs_ild)
            
            # Plot average original RTD
            avg_rtd = np.nanmean(np.array(aggregated_rtds[stim_key]), axis=0)
            ax1.plot(rt_bins[:-1] + np.diff(rt_bins)/2, avg_rtd, color=abl_colors[abl], lw=1.5, label=f'ABL={abl}')

            # Plot average rescaled RTD
            avg_rescaled_rtd = np.nanmean(np.array(aggregated_rescaled_data[stim_key]), axis=0)
            ax2.plot(rt_bins[:-1] + np.diff(rt_bins)/2, avg_rescaled_rtd, color=abl_colors[abl], lw=1.5)

        ax1.set_title(f'|ILD|={abs_ild}')
        ax1.set_xlim(0, 0.7)
        if j == 0: ax1.set_ylabel('Density')
        
        ax2.set_xlim(0, 0.7)
        ax2.set_xlabel('RT (s)')
        if j == 0: ax2.set_ylabel('Density (Rescaled)')

    axes[0, 0].legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)

print(f'Average plot PDF saved to {avg_output_filename}')


# %%
print("Generating average animal Q-Q plot...")

# --- 1. Aggregate Q-Q data from all animals ---
aggregated_qq_data = {(abl, ild): [] for abl in [20, 40] for ild in abs_ILD_arr}
quantile_levels = np.arange(0.01, 1.0, 0.01)

for batch_animal_pair, animal_data in tqdm(rtd_data.items(), desc="Aggregating Q-Q data"):
    # This part recalculates the Q-Q data exactly as in the per-animal plot section
    quantiles_by_abl_ild = {abl: {ild: None for ild in abs_ILD_arr} for abl in ABL_arr}
    for j, abs_ild in enumerate(abs_ILD_arr):
        for abl in ABL_arr:
            stim_key = (abl, abs_ild)
            quant_arr = animal_data.get(stim_key, {}).get('quantiles', np.full_like(quantile_levels, np.nan))
            quantiles_by_abl_ild[abl][abs_ild] = quant_arr

        if does_min_RT_depend_on_ILD:
            current_min_RT_cut = min_RT_cut_by_ILD.get(abs_ild, np.nan)
        else:
            current_min_RT_cut = min_RT_cut

        q_60 = quantiles_by_abl_ild[60][abs_ild]
        for abl in [20, 40]:
            q_other = quantiles_by_abl_ild[abl][abs_ild]
            if not np.any(np.isnan(q_60)) and not np.any(np.isnan(q_other)):
                mask = (q_60 >= current_min_RT_cut) & (q_60 <= max_RT_cut)
                if np.sum(mask) >= 2:
                    q_other_minus_60 = q_other - q_60
                    x_qq = q_60[mask] - current_min_RT_cut
                    y_qq_raw = q_other_minus_60[mask]
                    y_qq_shifted = y_qq_raw - y_qq_raw[0] # Shift to start at y=0
                    
                    # Store the (x, y) pairs for this animal
                    aggregated_qq_data[(abl, abs_ild)].append({'x': x_qq, 'y': y_qq_shifted})

# --- 2. Average the Q-Q data and the individual slopes ---

# -- Part A: Interpolate and Average Q-Q data for plotting the average data curve --
common_x_qq = np.linspace(0, max_RT_cut - min_RT_cut, 100)
avg_qq_plots = {}

for stim_key, all_animal_qqs in aggregated_qq_data.items():
    interpolated_ys = []
    for animal_qq in all_animal_qqs:
        # np.interp requires at least 2 points for interpolation
        if len(animal_qq['x']) > 1:
            interp_y = np.interp(common_x_qq, animal_qq['x'], animal_qq['y'], left=np.nan, right=np.nan)
            interpolated_ys.append(interp_y)
    
    if interpolated_ys:
        y_matrix = np.vstack(interpolated_ys)
        avg_y = np.nanmean(y_matrix, axis=0)
        sem_y = np.nanstd(y_matrix, axis=0) / np.sqrt(np.sum(~np.isnan(y_matrix), axis=0))
        avg_qq_plots[stim_key] = {'x': common_x_qq, 'avg_y': avg_y, 'sem_y': sem_y}

# -- Part B: Average the slopes from individual animal fits for plotting the average fit line --
avg_slope_fits = {}
slopes_by_stim = defaultdict(list)

# Collect all individual slopes from the per-animal fits (`all_fit_results`)
for batch_animal_pair, fit_data in all_fit_results.items():
    for abs_ild, abl_fits in fit_data.items():
        for abl, results in abl_fits.items():
            if 'slope' in results and not np.isnan(results['slope']):
                stim_key = (abl, abs_ild)
                slopes_by_stim[stim_key].append(results['slope'])

# Calculate average and SEM of the collected slopes
for stim_key, slopes in slopes_by_stim.items():
    if slopes:
        avg_slope = np.mean(slopes)
        sem_slope = np.std(slopes) / np.sqrt(len(slopes))
        avg_slope_fits[stim_key] = {'avg_slope': avg_slope, 'sem_slope': sem_slope, 'n_animals': len(slopes)}

# --- 3. Plotting Average Q-Q Plot ---
# --- 3. Plotting Average Q-Q Data and Average Slope Line ---
fig, axes = plt.subplots(1, len(abs_ILD_arr), figsize=(15, 5), sharex=True, sharey=True)
fig.suptitle('Average Q-Q Analysis', fontsize=16)

for j, abs_ild in enumerate(abs_ILD_arr):
    ax = axes[j]
    ax.set_title(f'|ILD| = {abs_ild}')

    for abl in [20, 40]:
        stim_key = (abl, abs_ild)

        # Plot 1: The averaged Q-Q data curve with SEM shading
        if stim_key in avg_qq_plots:
            plot_data = avg_qq_plots[stim_key]
            x = plot_data['x']
            y = plot_data['avg_y']
            sem = plot_data['sem_y']
            ax.plot(x, y, color=abl_colors[abl], lw=1, alpha=0.8, label=f'Data (ABL={abl})')
            ax.fill_between(x, y - sem, y + sem, color=abl_colors[abl], alpha=0.2)

        # Plot 2: The line based on the average of individual slopes
        if stim_key in avg_slope_fits:
            fit = avg_slope_fits[stim_key]
            avg_slope = fit['avg_slope']
            x_line = np.array([0, max_RT_cut - min_RT_cut])
            y_line = avg_slope * x_line
            ax.plot(x_line, y_line, color=abl_colors[abl], lw=2.5, linestyle='--', label=f'Fit (m={avg_slope:.3f})')

    ax.axhline(0, color='k', linestyle='--', lw=1)
    ax.set_xlabel('RT(q) - min_RT (s)')
    if j == 0:
        ax.set_ylabel('RT_other(q) - RT_60(q) (s)')
    ax.legend()

# Set common y-limits after all plotting is done
y_lims = [ax.get_ylim() for ax in axes]
min_y = min(lim[0] for lim in y_lims)
max_y = max(lim[1] for lim in y_lims)
for ax in axes:
    ax.set_ylim(min_y, max_y)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
avg_plot_filename = f'average_qq_plot_data_and_fit{FILENAME_SUFFIX}.pdf'
fig.savefig(avg_plot_filename, dpi=300)
plt.close(fig)

print(f'Average Q-Q plot saved to {avg_plot_filename}')

# Save all aggregated results
output_data_filename = f'aggregated_qq_results{FILENAME_SUFFIX}.pkl'
with open(output_data_filename, 'wb') as f:
    pickle.dump({'aggregated_qq_data': aggregated_qq_data, 'avg_slope_fits': avg_slope_fits, 'avg_qq_plots': avg_qq_plots}, f)
print(f"Saved all aggregated results to {output_data_filename}")
# %% 

# =============================================================================
# NEW: Aggregate RAW data across all animals and run KDE on the raw data
# This is the methodologically correct way to perform KDE, as it avoids
# artifacts from binning and uses the true underlying data.
# =============================================================================

import pandas as pd
from collections import defaultdict

# Dictionaries to hold the raw data points for each condition
aggregated_raw_rts = defaultdict(list)
aggregated_raw_rescaled_rts = defaultdict(list)

print("Aggregating raw RT data from all animals...")
# Loop through all animal/batch pairs that were processed
for batch_animal_pair in tqdm(rtd_data.keys(), desc="Loading Raw RTs"):
    batch_name, animal_id_str = batch_animal_pair
    animal_id = int(animal_id_str)

    # Load the corresponding CSV
    csv_path = f'batch_csvs/batch_{batch_name}_valid_and_aborts{CSV_SUFFIX}.csv'
    try:
        df_animal = pd.read_csv(csv_path)
        # Filter for the specific animal and for valid trials & aborts
        df_animal = df_animal[(df_animal['animal'] == animal_id) & ((df_animal['abort_event'].isin(ABORT_EVENTS)) | (df_animal['success'].isin([1,-1])))]

    except FileNotFoundError:
        print(f"Warning: Could not find {csv_path}. Skipping.")
        continue

    # Get the fit results (slopes) for this animal
    animal_fit_results = all_fit_results.get(batch_animal_pair, {})

    for abs_ild in abs_ILD_arr:
        # Determine min_RT_cut for this specific ILD
        if does_min_RT_depend_on_ILD:
            current_min_RT_cut = min_RT_cut_by_ILD.get(abs_ild, np.nan)
        else:
            current_min_RT_cut = min_RT_cut

        # Create a copy for filtering
        df_animal['abs_ILD'] = np.abs(df_animal['ILD'])
        df_animal_ild = df_animal[df_animal['abs_ILD'] == abs_ild].copy()
    
        
        for abl in ABL_arr:
            stim_key = (abl, abs_ild)
            
            # Filter for ABL
            df_stim = df_animal_ild[df_animal_ild['ABL'] == abl].copy()
            raw_rts = df_stim['RTwrtStim'].dropna().values
            valid_rts = raw_rts[(raw_rts >= -0.1) & (raw_rts <= 1)]
            
            if len(valid_rts) > 0:
                # --- Aggregate Original RTs ---
                aggregated_raw_rts[stim_key].extend(valid_rts)

                # --- Aggregate Rescaled RTs ---
                slope = 0  # Default for ABL 60
                if abl != 60:
                    try:
                        slope = animal_fit_results[abs_ild][abl]['slope']
                        if np.isnan(slope): slope = 0
                    except (KeyError, TypeError):
                        raise Exception(f"Slope not found for ABL={abl}, abs_ILD={abs_ild}")
                
                # Rescale the valid RTs and add them to the list, using the correct formula
                if (1 + slope) != 0:
                    rescaled_rts = np.where(valid_rts > current_min_RT_cut, ((valid_rts - current_min_RT_cut) / (1 + slope)) + current_min_RT_cut, valid_rts)
                    aggregated_raw_rescaled_rts[stim_key].extend(rescaled_rts)
                else:
                    # If slope is -1, rescaling is undefined
                    raise Exception(f"Slope is -1 for ABL={abl}, abs_ILD={abs_ild}")
                    # aggregated_raw_rescaled_rts[stim_key].extend(valid_rts)

# --- Plotting with KDE on RAW DATA ---
kde_output_filename_raw = f'average_animal_rtd_plots_KDE_RAW_DATA{FILENAME_SUFFIX}_min_RT_{min_RT_cut}_max_RT_{max_RT_cut}_bin_size_{rt_bin_size}.pdf'
with PdfPages(kde_output_filename_raw) as pdf:
    fig, axes = plt.subplots(2, len(abs_ILD_arr), figsize=(15, 8), sharex='col', sharey='row')
    fig.suptitle('Average Animal RTDs (KDE on Raw Data)', fontsize=16)

    x_grid = np.arange(-0.1, 1, 0.001).reshape(-1, 1)
    bandwidth = 0.01 # Resetting to a more reasonable default after changing method

    for j, abs_ild in enumerate(abs_ILD_arr):
        ax1 = axes[0, j]
        ax2 = axes[1, j]

        for abl in ABL_arr:
            stim_key = (abl, abs_ild)
            
            # --- Original RTD KDE from Raw Data ---
            all_raw_rts = np.array(aggregated_raw_rts.get(stim_key, [])).reshape(-1, 1)
            if all_raw_rts.shape[0] > 1:
                try:
                    kde = KernelDensity(kernel='epanechnikov', bandwidth=bandwidth)
                    # giving it raw data, so that it learns distribution from raw data
                    # it learns the cont basis functions params
                    kde.fit(all_raw_rts)
                    # convert the continous basis on x_grid to plot
                    log_dens = kde.score_samples(x_grid)
                    kde_y = np.exp(log_dens)
                    kde_y /= np.trapz(kde_y, x_grid.ravel())
                    ax1.plot(x_grid.ravel(), kde_y, color=abl_colors[abl], lw=1.5, label=f'ABL={abl}')
                except Exception as e:
                    print(f"KDE failed for original {stim_key}: {e}")

            # --- Rescaled RTD KDE from Raw Data ---
            all_rescaled_rts = np.array(aggregated_raw_rescaled_rts.get(stim_key, [])).reshape(-1, 1)
            if all_rescaled_rts.shape[0] > 1:
                try:
                    kde_rescaled = KernelDensity(kernel='epanechnikov', bandwidth=bandwidth)
                    kde_rescaled.fit(all_rescaled_rts)
                    log_dens_rescaled = kde_rescaled.score_samples(x_grid)
                    kde_y_rescaled = np.exp(log_dens_rescaled)
                    kde_y_rescaled /= np.trapz(kde_y_rescaled, x_grid.ravel())
                    ax2.plot(x_grid.ravel(), kde_y_rescaled, color=abl_colors[abl], lw=1.5)
                except Exception as e:
                    print(f"KDE failed for rescaled {stim_key}: {e}")

        ax1.set_title(f'|ILD|={abs_ild}')
        ax1.set_xlim(-0.1, 0.7)
        if j == 0: ax1.set_ylabel('Density (KDE)')
        
        ax2.set_xlim(-0.1, 0.7)
        ax2.set_xlabel('RT (s)')
        if j == 0: ax2.set_ylabel('Density (Rescaled, KDE)')

    # Sync y-axes for each row
    for i in range(2):
        max_ylim = 0
        for ax in axes[i, :]:
            if ax.get_ylim()[1] > max_ylim:
                max_ylim = ax.get_ylim()[1]
        for ax in axes[i, :]:
            ax.set_ylim(0, max_ylim * 1.1)
    
    axes[0, 0].legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)

print(f'KDE plot PDF saved to {kde_output_filename_raw}')
