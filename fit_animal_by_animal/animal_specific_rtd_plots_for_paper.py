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
rt_bins = np.arange(0, 1.02, 0.02)
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
output_filename = 'animal_specific_rtd_plots.pdf'

with PdfPages(output_filename) as pdf:
    for batch_animal_pair, animal_data in rtd_data.items():
        batch_name, animal_id = batch_animal_pair
        fig, axes = plt.subplots(3, len(abs_ILD_arr), figsize=(15, 12), sharex='col')
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

            # Row 2: Q-Q analysis plots
            ax2 = axes[1, j]
            q_60 = quantiles_by_abl_ild[60][abs_ild]
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

                        # plot fitted line
                        if not np.isnan(slope):
                            # OLD
                            # x_line = np.array([min_RT_cut, np.nanmax(q_60[mask])])
                            # y_line = y_intercept + slope * (x_line - min_RT_cut)
                            # ax2.plot(x_line, y_line, color=abl_colors[abl], linestyle='-' if abl==20 else '--', label=f'Fit {abl} (m={slope:.2f})', lw=3)
                            x_line_shifted = np.array([0, np.nanmax(x_fit_calc)])
                            y_line_shifted = slope * x_line_shifted

                            ax2.plot(x_line_shifted, y_line_shifted, color=abl_colors[abl], label=f'Fit {abl} (m={slope:.2f})', lw=3)

                    else:
                        fit_results[abs_ild][abl] = {'slope': np.nan}
                else:
                    fit_results[abs_ild][abl] = {'slope': np.nan}
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
            ax3.set_xlabel('RT (s)')
            if j == 0: ax3.set_ylabel('Density (Rescaled)')

        # Sync y-axes for each row
        for i in range(3): # For each row
            y_lims = [ax.get_ylim() for ax in axes[i, :]]
            min_y = min(lim[0] for lim in y_lims)
            max_y = max(lim[1] for lim in y_lims)
            for ax in axes[i, :]:
                ax.set_ylim(min_y, max_y)

        axes[0, 0].legend()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

print(f'PDF saved to {output_filename}')

# %%
