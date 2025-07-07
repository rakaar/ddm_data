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
    file_name = os.path.join(csv_dir, f'batch_{batch_name}_valid_and_aborts.csv')
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
output_filename = f'animal_specific_rtd_plots_min_RT_{min_RT_cut}_max_RT_{max_RT_cut}_bin_size_{rt_bin_size}.pdf'
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
                            xvals = ((bin_centers - min_RT_cut) / (1 + slope)) + min_RT_cut
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
                        xvals = ((bin_centers - min_RT_cut) / (1 + slope)) + min_RT_cut
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
avg_output_filename = f'average_animal_rtd_plots_min_RT_{min_RT_cut}_max_RT_{max_RT_cut}_bin_size_{rt_bin_size}.pdf'
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

# --- Plotting with KDE (Epanechnikov) ---

kde_output_filename = f'average_animal_rtd_plots_KDE_Epanechnikov_min_RT_{min_RT_cut}_max_RT_{max_RT_cut}_bin_size_{rt_bin_size}.pdf'
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
            bin_centers = rt_bins[:-1] + np.diff(rt_bins)/2
            
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
    csv_path = f'batch_csvs/batch_{batch_name}_valid_and_aborts.csv'
    try:
        df_animal = pd.read_csv(csv_path)
        # Filter for the specific animal and for valid trials
        # df_animal = df_animal[(df_animal['animal'] == animal_id) & (df_animal['success'].isin([1, -1]))]
        df_animal = df_animal[(df_animal['animal'] == animal_id) & ((df_animal['abort_event'] == 3) | (df_animal['success'].isin([1,-1])))]

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
            df_stim = df_animal_ild[df_animal_ild['ABL'] == abl]
            
            # Get raw RTs and filter them consistent with the original analysis.
            # The original code uses all RTs in the [0, 1] range for histograms.
            # min_RT_cut is for the scaling transformation, not for pre-filtering the data.
            raw_rts = df_stim['RTwrtStim'].dropna().values
            valid_rts = raw_rts[(raw_rts >= -0.1) & (raw_rts <= 1)]
            
            if len(valid_rts) > 0:
                # --- Aggregate Original RTs ---
                aggregated_raw_rts[stim_key].extend(valid_rts)

                # --- Aggregate Rescaled RTs ---
                # Get the slope for this condition
                slope = 0  # Default for ABL 60
                if abl != 60:
                    try:
                        slope = animal_fit_results[abs_ild][abl]['slope']
                        if np.isnan(slope): slope = 0
                    except (KeyError, TypeError):
                        raise Exception(f"Slope not found for ABL={abl}, abs_ILD={abs_ild}")
                        slope = 0 # If slope wasn't calculated, treat as 0
                
                # Rescale the valid RTs and add them to the list, using the correct formula
                if (1 + slope) != 0:
                    rescaled_rts = ((valid_rts - min_RT_cut) / (1 + slope)) + min_RT_cut
                    aggregated_raw_rescaled_rts[stim_key].extend(rescaled_rts)
                else:
                    # If slope is -1, rescaling is undefined; add the original rts
                    aggregated_raw_rescaled_rts[stim_key].extend(valid_rts)

# --- Plotting with KDE on RAW DATA ---
kde_output_filename_raw = f'average_animal_rtd_plots_KDE_RAW_DATA_min_RT_{min_RT_cut}_max_RT_{max_RT_cut}_bin_size_{rt_bin_size}.pdf'
with PdfPages(kde_output_filename_raw) as pdf:
    fig, axes = plt.subplots(2, len(abs_ILD_arr), figsize=(15, 8), sharex='col', sharey='row')
    fig.suptitle('Average Animal RTDs (KDE on Raw Data)', fontsize=16)

    x_grid = np.arange(-0.1, 1, 0.001).reshape(-1, 1)
    bandwidth = 0.001 # Resetting to a more reasonable default after changing method

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


# %%
