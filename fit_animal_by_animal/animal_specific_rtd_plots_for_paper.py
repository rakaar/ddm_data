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
INCLUDE_ABORT_EVENT_4 = True

if INCLUDE_ABORT_EVENT_4:
    CSV_SUFFIX = '_and_4'
    ABORT_EVENTS = [3, 4]
    FILENAME_SUFFIX = '_with_abort4'
else:
    CSV_SUFFIX = ''
    ABORT_EVENTS = [3]
    FILENAME_SUFFIX = ''

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
def get_animal_RTD_data(batch_name, animal_id, ABL, abs_ILD, bins):
    file_name = os.path.join(csv_dir, f'batch_{batch_name}_valid_and_aborts{CSV_SUFFIX}.csv')
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"File not found: {file_name}")
        bin_centers = (bins[:-1] + bins[1:]) / 2
        return bin_centers, np.full_like(bin_centers, np.nan), 0

    df['abs_ILD'] = df['ILD'].abs()
    df_filtered = df[(df['animal'] == animal_id) & (df['ABL'] == ABL) & (df['abs_ILD'] == abs_ILD) \
                     & (df['RTwrtStim'] <= 1) & (df['RTwrtStim'] >= 0)]
    
    n_trials = len(df_filtered)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    if n_trials == 0:
        rtd_hist = np.full_like(bin_centers, np.nan)
    else:
        rtd_hist, _ = np.histogram(df_filtered['RTwrtStim'], bins=bins, density=True)
        
    return bin_centers, rtd_hist, n_trials

# %%
# params
ABL_arr = [20, 40, 60]
abs_ILD_arr = [1, 2, 4, 8, 16]
rt_bin_size = 0.02
rt_bins = np.arange(0, 1 + rt_bin_size, rt_bin_size)
min_RT_cut = 0.09
max_RT_cut = 0.3
# %%
def process_batch_animal(batch_animal_pair):
    batch_name, animal_id = batch_animal_pair
    animal_rtd_data = {}
    try:
        for abl in ABL_arr:
            for abs_ild in abs_ILD_arr:
                stim_key = (abl, abs_ild)
                bin_centers, rtd_hist, n_trials = get_animal_RTD_data(batch_name, int(animal_id), abl, abs_ild, rt_bins)
                animal_rtd_data[stim_key] = {
                    'empirical': {
                        'bin_centers': bin_centers,
                        'rtd_hist': rtd_hist,
                        'n_trials': n_trials
                    }
                }
    except Exception as e:
        print(f"Error processing batch {batch_name}, animal {animal_id}: {str(e)}")
    return batch_animal_pair, animal_rtd_data

# %%
n_jobs = max(1, os.cpu_count() - 1)
results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(process_batch_animal)(pair) for pair in batch_animal_pairs
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
                emp_data = animal_data.get(stim_key, {}).get('empirical', {})
                if emp_data and emp_data.get('n_trials', 0) > 0 and not np.all(np.isnan(emp_data['rtd_hist'])):
                    bin_widths = np.diff(rt_bins)
                    cdf = np.cumsum(emp_data['rtd_hist'] * bin_widths)
                    cdf = cdf / cdf[-1] if cdf[-1] > 0 else cdf
                    quantiles_by_abl_ild[abl][abs_ild] = np.interp(quantile_levels, cdf, emp_data['bin_centers'])
                else:
                    quantiles_by_abl_ild[abl][abs_ild] = np.full_like(quantile_levels, np.nan)

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
            for abl in [20, 40]:
                q_other = quantiles_by_abl_ild[abl][abs_ild]
                if not np.any(np.isnan(q_60)) and not np.any(np.isnan(q_other)):
                    mask = (q_60 >= min_RT_cut) & (q_60 <= max_RT_cut)
                    if np.sum(mask) >= 2:
                        # scatter plot
                        q_other_minus_60 = q_other - q_60
                        # ax2.plot(q_60, q_other - q_60, 'o' if abl==20 else 's', color=abl_colors[abl], alpha=0.2)
                        ax2.plot(q_60[mask] - min_RT_cut, q_other_minus_60[mask] - q_other_minus_60[mask][0], 'o' if abl==20 else 's', color=abl_colors[abl], alpha=0.2, lw=1.5)
                        # fit and find slope
                        x_fit_calc = q_60[mask] - min_RT_cut
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
                            xvals = np.where(bin_centers > min_RT_cut, ((bin_centers - min_RT_cut) / (1 + slope)) + min_RT_cut, bin_centers)
                            multiplier = np.ones_like(rtd_hist)
                            multiplier[bin_centers > min_RT_cut] = slope + 1
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
            emp_data = animal_data.get(stim_key, {}).get('empirical', {})
            if emp_data and emp_data.get('n_trials', 0) > 0 and not np.all(np.isnan(emp_data['rtd_hist'])):
                bin_widths = np.diff(rt_bins)
                cdf = np.cumsum(emp_data['rtd_hist'] * bin_widths)
                cdf = cdf / cdf[-1] if cdf[-1] > 0 else cdf
                quantiles_by_abl_ild[abl][abs_ild] = np.interp(quantile_levels, cdf, emp_data['bin_centers'])
            else:
                quantiles_by_abl_ild[abl][abs_ild] = np.full_like(quantile_levels, np.nan)

        q_60 = quantiles_by_abl_ild[60][abs_ild]
        for abl in [20, 40]:
            q_other = quantiles_by_abl_ild[abl][abs_ild]
            if not np.any(np.isnan(q_60)) and not np.any(np.isnan(q_other)):
                mask = (q_60 >= min_RT_cut) & (q_60 <= max_RT_cut)
                if np.sum(mask) >= 2:
                    q_other_minus_60 = q_other - q_60
                    x_fit_calc = q_60[mask] - min_RT_cut
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
                        xvals = np.where(bin_centers > min_RT_cut, ((bin_centers - min_RT_cut) / (1 + slope)) + min_RT_cut, bin_centers)
                        multiplier = np.ones_like(rtd_hist)
                        multiplier[bin_centers > min_RT_cut] = slope + 1
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

# --- Plotting with KDE (Epanechnikov) ---



kde_output_filename = f'average_animal_rtd_plots_KDE_Epanechnikov{FILENAME_SUFFIX}_min_RT_{min_RT_cut}_max_RT_{max_RT_cut}_bin_size_{rt_bin_size}.pdf'

with PdfPages(kde_output_filename) as pdf:
    fig, axes = plt.subplots(2, len(abs_ILD_arr), figsize=(15, 8), sharex='col', sharey='row')
    fig.suptitle('Average Animal RTDs (Epanechnikov KDE)', fontsize=16)

    x_grid = np.arange(0, 0.7, 0.001).reshape(-1, 1)
    bandwidth = 0.09

    for j, abs_ild in enumerate(abs_ILD_arr):
        ax1 = axes[0, j]
        ax2 = axes[1, j]

        for abl in ABL_arr:
            stim_key = (abl, abs_ild)
            
            # --- Original RTD KDE ---
            avg_rtd = np.nanmean(np.array(aggregated_rtds[stim_key]), axis=0)
            valid_indices = ~np.isnan(avg_rtd)
            if np.count_nonzero(valid_indices) > 1:
                filtered_centers = bin_centers[valid_indices].reshape(-1, 1)
                filtered_weights = avg_rtd[valid_indices]
                if np.sum(filtered_weights) > 1e-6:
                    try:
                        kde = KernelDensity(kernel='epanechnikov', bandwidth=bandwidth)
                        kde.fit(filtered_centers, sample_weight=filtered_weights)
                        log_dens = kde.score_samples(x_grid)
                        kde_y = np.exp(log_dens)
                        # Normalize the density to ensure area is 1
                        kde_y /= np.trapz(kde_y, x_grid.ravel())
                        ax1.plot(x_grid.ravel(), kde_y, color=abl_colors[abl], lw=1.5, label=f'ABL={abl}')
                    except Exception as e:
                        print(f"KDE failed for original {stim_key}: {e}. Plotting original.")
                        ax1.plot(bin_centers, avg_rtd, color=abl_colors[abl], lw=1.5, label=f'ABL={abl}')
            
            # --- Rescaled RTD KDE ---
            avg_rescaled_rtd = np.nanmean(np.array(aggregated_rescaled_data[stim_key]), axis=0)
            valid_indices_rescaled = ~np.isnan(avg_rescaled_rtd)
            if np.count_nonzero(valid_indices_rescaled) > 1:
                filtered_centers_rescaled = bin_centers[valid_indices_rescaled].reshape(-1, 1)
                filtered_weights_rescaled = avg_rescaled_rtd[valid_indices_rescaled]
                if np.sum(filtered_weights_rescaled) > 1e-6:
                    try:
                        kde_rescaled = KernelDensity(kernel='epanechnikov', bandwidth=bandwidth)
                        kde_rescaled.fit(filtered_centers_rescaled, sample_weight=filtered_weights_rescaled)
                        log_dens_rescaled = kde_rescaled.score_samples(x_grid)
                        kde_y_rescaled = np.exp(log_dens_rescaled)
                        # Normalize the density
                        kde_y_rescaled /= np.trapz(kde_y_rescaled, x_grid.ravel())
                        ax2.plot(x_grid.ravel(), kde_y_rescaled, color=abl_colors[abl], lw=1.5)
                    except Exception as e:
                        print(f"KDE failed for rescaled {stim_key}: {e}. Plotting original.")
                        ax2.plot(bin_centers, avg_rescaled_rtd, color=abl_colors[abl], lw=1.5)

        ax1.set_title(f'|ILD|={abs_ild}')
        ax1.set_xlim(0, 0.7)
        ax1.set_ylim(0,6)
        if j == 0: ax1.set_ylabel('Density (KDE)')
        
        ax2.set_xlim(0, 0.7)
        ax2.set_ylim(0,6)
        ax2.set_xlabel('RT (s)')
        if j == 0: ax2.set_ylabel('Density (Rescaled, KDE)')

    # Sync y-axes for each row
    for i in range(2):
        max_ylim = 0
        for j in range(len(abs_ILD_arr)):
            max_ylim = max(max_ylim, axes[i, j].get_ylim()[1])
        for j in range(len(abs_ILD_arr)):
            axes[i, j].set_ylim(0, max_ylim)

    axes[0, 0].legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)

print(f'KDE plot PDF saved to {kde_output_filename}')

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
                    rescaled_rts = np.where(valid_rts > min_RT_cut, ((valid_rts - min_RT_cut) / (1 + slope)) + min_RT_cut, valid_rts)
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


#%%
# =============================================================================
# NEW: Aggregate RAW data across all animals and plot HISTOGRAM on the raw data
# =============================================================================

# --- Plotting with HISTOGRAM on RAW DATA ---
fig_hist, axes_hist = plt.subplots(2, len(abs_ILD_arr), figsize=(15, 8), sharex='col', sharey='row')
fig_hist.suptitle('Average Animal RTDs (Histogram on Raw Data)', fontsize=16)

# Define bins for the histogram.
bins = np.arange(-0.1, 1.0 + 0.02, 0.02)

for j, abs_ild in enumerate(abs_ILD_arr):
    ax1 = axes_hist[0, j]
    ax2 = axes_hist[1, j]

    for abl in ABL_arr:
        stim_key = (abl, abs_ild)
        
        # --- Original RTD Histogram from Raw Data ---
        all_raw_rts = np.array(aggregated_raw_rts.get(stim_key, []))
        if all_raw_rts.shape[0] > 1:
            ax1.hist(all_raw_rts, bins=bins, density=True, histtype='step', lw=1.5, label=f'ABL={abl}', color=abl_colors[abl])

        # --- Rescaled RTD Histogram from Raw Data ---
        all_rescaled_rts = np.array(aggregated_raw_rescaled_rts.get(stim_key, []))
        if all_rescaled_rts.shape[0] > 1:
            ax2.hist(all_rescaled_rts, bins=bins, density=True, histtype='step', lw=1.5, color=abl_colors[abl])

    ax1.set_title(f'|ILD|={abs_ild}')
    ax1.set_xlim(-0.1, 1)
    if j == 0: ax1.set_ylabel('Density (Histogram)')
    
    ax2.set_xlim(-0.1, 1)
    ax2.set_xlabel('RT (s)')
    if j == 0: ax2.set_ylabel('Density (Rescaled, Histogram)')

# Sync y-axes for each row
for i in range(2):
    max_ylim = 0
    for ax in axes_hist[i, :]:
        if ax.get_ylim()[1] > max_ylim:
            max_ylim = ax.get_ylim()[1]
    for ax in axes_hist[i, :]:
        ax.set_ylim(0, max_ylim * 1.1)

axes_hist[0, 0].legend()
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# %% 
# =============================================================================
# NEW: Plot individual animal RTDs to check for the dip around 0ms
# =============================================================================

print("\nPlotting individual animal RTDs to see if the dip is a consistent feature...")

# Use the same animal pairs as before
n_animals = len(batch_animal_pairs)
n_ilds = len(abs_ILD_arr)

# Create a figure with a row for each animal and a column for each |ILD|
fig, axes = plt.subplots(n_animals, n_ilds, figsize=(15, 3 * n_animals), sharex=True, sharey='row', squeeze=False)
fig.suptitle('Individual Animal RTDs (including aborts)', fontsize=16)

# Define bins from -0.5s to 1s to capture pre-stimulus responses
individual_bins = np.arange(-0.5, 1.02, 0.01)

# Loop through each animal
for i, (batch_name, animal_id_str) in enumerate(tqdm(batch_animal_pairs, desc="Plotting Individual RTDs")):
    animal_id = int(animal_id_str)
    
    # Set the y-label for the row to identify the animal
    axes[i, 0].set_ylabel(f"Animal {animal_id}\n({batch_name})", fontsize=9)

    # Load the corresponding CSV data for the animal's batch
    csv_path = f'batch_csvs/batch_{batch_name}_valid_and_aborts{CSV_SUFFIX}.csv'
    try:
        df_full = pd.read_csv(csv_path)
        # Filter for the specific animal and for both valid and abort trials
        df_animal = df_full[(df_full['animal'] == animal_id) & ((df_full['abort_event'].isin(ABORT_EVENTS)) | (df_full['success'].isin([1, -1])))]
        df_animal['abs_ILD'] = np.abs(df_animal['ILD'])
    except FileNotFoundError:
        print(f"Warning: Could not find {csv_path} for animal {animal_id}. Skipping row.")
        # If file not found, disable axes for this row
        for j in range(n_ilds):
            axes[i, j].set_visible(False)
        continue

    # Loop through each |ILD| condition
    for j, abs_ild in enumerate(abs_ILD_arr):
        ax = axes[i, j]
        
        # Set the title for the column (only for the first row)
        if i == 0:
            ax.set_title(f'|ILD|={abs_ild}')

        # Filter data for the specific |ILD|
        df_animal_ild = df_animal[df_animal['abs_ILD'] == abs_ild]

        # Plot a histogram for each ABL level
        for abl in ABL_arr:
            df_stim = df_animal_ild[df_animal_ild['ABL'] == abl]
            rts = df_stim['RTwrtStim'].dropna().values
            
            if len(rts) > 0:
                ax.hist(rts, bins=individual_bins, density=True, histtype='step', lw=1.5, color=abl_colors.get(abl, 'k'))

        ax.set_xlim(-0.5, 0.5)
        # Add a vertical line at 0 to mark stimulus onset
        ax.axvline(0, color='r', linestyle='--', lw=0.8, alpha=0.3)

# Add legend for ABL colors to the figure
legend_labels = [f'ABL={abl}' for abl in ABL_arr]
legend_handles = [plt.Line2D([0], [0], color=abl_colors[abl], lw=2) for abl in ABL_arr]
fig.legend(legend_handles, legend_labels, loc='upper right')

# Set common x-label for the last row of plots
for j in range(n_ilds):
    axes[-1, j].set_xlabel('RT (s)')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# %%
# %%
# =============================================================================
# NEW: Analyze and plot the density of RTs around zero for each animal
# =============================================================================
import re
from collections import defaultdict

print("\nAnalyzing density of RTs around t=0 for each animal (ABL aggregated)...")

# Define the window around zero to check for RTs
# Define the time windows to analyze, based on 20ms bins
windows = {
    'past':    {'range': [-0.03, -0.01], 'color': 'tab:red',   'label': 'Past [-30, -10) ms'},
    'present': {'range': [-0.01,  0.01], 'color': 'tab:blue',  'label': 'Present [-10, 10) ms'},
    'future':  {'range': [ 0.01,  0.03], 'color': 'tab:green', 'label': 'Future [10, 30) ms'}
}

# --- Data Collection (Aggregated over ABL) ---
# Dictionary to store the density results: { window_name: {abs_ild: {animal_id: density}} }
density_by_window = defaultdict(lambda: defaultdict(dict))
animal_labels = []
animal_ids_in_order = []

# Create a unique, ordered list of animal labels and IDs
animal_id_set = set()
for batch_name, animal_id_str in batch_animal_pairs:
    animal_id = int(animal_id_str)
    if animal_id not in animal_id_set:
        animal_id_set.add(animal_id)
        clean_batch_name = re.sub(r'_.*', '', batch_name)
        animal_labels.append(f"{animal_id}\n({clean_batch_name})")
        animal_ids_in_order.append(animal_id)

# Loop through each animal to calculate the proportion of trials in the defined windows
for animal_id in tqdm(animal_ids_in_order, desc="Calculating Window Densities (ABL Aggregated)"):
    batch_name = next((b for b, a in batch_animal_pairs if int(a) == animal_id), None)
    if not batch_name:
        continue

    # Load the corresponding CSV data
    csv_path = f'batch_csvs/batch_{batch_name}_valid_and_aborts{CSV_SUFFIX}.csv'
    try:
        df_full = pd.read_csv(csv_path)
        df_animal = df_full[(df_full['animal'] == animal_id) & ((df_full['abort_event'].isin(ABORT_EVENTS)) | (df_full['success'].isin([1, -1])))]
        df_animal['abs_ILD'] = np.abs(df_animal['ILD'])
    except FileNotFoundError:
        continue

    # Loop through each stimulus condition, aggregating over ABL
    for abs_ild in abs_ILD_arr:
        df_ild = df_animal[(df_animal['abs_ILD'] == abs_ild) & (df_animal['ABL'].isin(ABL_arr))]
        rts = df_ild['RTwrtStim'].dropna().values

        # Calculate density for each time window
        for window_name, params in windows.items():
            window_range = params['range']
            if len(rts) > 0:
                # Use >= and < to define non-overlapping bins
                rts_in_window = np.sum((rts >= window_range[0]) & (rts < window_range[1]))
                density = rts_in_window / len(rts)
            else:
                density = 0.0
            
            density_by_window[window_name][abs_ild][animal_id] = density

# --- Plotting (Aggregated over ABL, showing multiple windows) ---
n_ilds = len(abs_ILD_arr)
fig, axes = plt.subplots(1, n_ilds, figsize=(5 * n_ilds, 7), sharey=True, squeeze=False)
axes = axes.flatten()
fig.suptitle('Proportion of Trials in Adjacent 20ms Time Windows', fontsize=16)

x_indices = np.arange(len(animal_ids_in_order))

for j, abs_ild in enumerate(abs_ILD_arr):
    ax = axes[j]
    ax.set_title(f'|ILD|={abs_ild}')
    
    # For each window, plot the densities per animal and the mean
    for window_name, params in windows.items():
        color = params['color']
        label = params['label']
        
        # Get densities for this condition and window, ordered by animal
        densities = [density_by_window[window_name][abs_ild].get(animal_id, 0) for animal_id in animal_ids_in_order]
        
        # Plot a single line connecting each animal's data point
        ax.plot(x_indices, densities, 'o-', color=color, label=label if j==0 else "")

        # Calculate and plot the mean across all animals
        if densities:
            mean_density = np.mean(densities)
            # Add mean line with same color, but dashed
            ax.axhline(y=mean_density, color=color, linestyle='--', linewidth=2, alpha=0.9)

    ax.set_xticks(x_indices)
    ax.set_xticklabels(animal_labels, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Animal ID (Batch)')
    if j == 0:
        ax.set_ylabel('Proportion of Trials')
        ax.legend(title='Time Window')

    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(bottom=0)

plt.tight_layout(rect=[0.02, 0.03, 1, 0.95])
plt.show()

# %%
##### Check abort types #####
df_7 = pd.read_csv('../out_LED.csv')
df_7['abort_event'].unique()
df_7_abort_4 = df_7[df_7['abort_event'] == 4]
# RTs of aborts
df_7_abort_4.loc[:, 'RTwrtStim'] = (df_7_abort_4['timed_fix'] - df_7_abort_4['intended_fix']).copy()
print(f"min RTwrtStim: {1000*df_7_abort_4['RTwrtStim'].min():.2f} ms, max RTwrtStim: {1000*df_7_abort_4['RTwrtStim'].max():.2f} ms")
# min RTwrtStim: 0.00 ms, max RTwrtStim: 9.76 ms
