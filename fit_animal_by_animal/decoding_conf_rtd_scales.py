# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from joblib import Parallel, delayed
from tqdm import tqdm
from time_vary_and_norm_simulators import psiam_tied_data_gen_wrapper_rate_norm_fn
import pickle
from collections import defaultdict
import warnings
from types import SimpleNamespace
from animal_wise_plotting_utils import calculate_theoretical_curves
from time_vary_norm_utils import (
    up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn, 
    cum_pro_and_reactive_time_vary_fn, 
    rho_A_t_fn, 
    cum_A_t_fn
)
from collections import defaultdict
import random
from scipy.stats import gaussian_kde
from scipy.integrate import trapezoid

# %%
DESIRED_BATCHES = ['Comparable', 'SD', 'LED2', 'LED1', 'LED34', 'LED6']
# DESIRED_BATCHES = ['LED7']

# Base directory paths
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_dir = os.path.join(base_dir, 'batch_csvs')
results_dir = base_dir  # Directory containing the pickle files

def find_batch_animal_pairs():
    pairs = []
    pattern = os.path.join(results_dir, 'results_*_animal_*.pkl')
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
                pairs.append((batch_name, animal_id))
        else:
            print(f"Warning: Invalid filename format: {filename}")
    return pairs

batch_animal_pairs = find_batch_animal_pairs()
# with open('high_slope_animals.pkl', 'rb') as f:
#     batch_animal_pairs = pickle.load(f)

print(f"Found {len(batch_animal_pairs)} batch-animal pairs from {len(set(p[0] for p in batch_animal_pairs))} batches:")

# Group animals by batch and print table
if batch_animal_pairs:
    batch_to_animals = defaultdict(list)
    for batch, animal in sorted(batch_animal_pairs):
        batch_to_animals[batch].append(animal)

    # Determine column widths for formatting
    max_batch_len = max(len(b) for b in batch_to_animals.keys()) if batch_to_animals else 0
    animal_strings = {b: ', '.join(a) for b, a in batch_to_animals.items()}
    max_animals_len = max(len(s) for s in animal_strings.values()) if animal_strings else 0

    # Header
    print(f"{'Batch':<{max_batch_len}}  {'Animals'}")
    print(f"{'=' * max_batch_len}  {'=' * max_animals_len}")

    # Rows
    for batch, animals_str in sorted(animal_strings.items()):
        print(f"{batch:<{max_batch_len}}  {animals_str}")

# %%
def get_animal_RTD_data(batch_name, animal_id, ABL, abs_ILD, bins):
    file_name = f'batch_csvs/batch_{batch_name}_valid_and_aborts.csv'
    df = pd.read_csv(file_name)
    df['abs_ILD'] = df['ILD'].abs()
     # J1: keep all RTs
    # df = df[(df['animal'] == animal_id) & (df['ABL'] == ABL) & (df['ILD'] == ILD) & (df['success'].isin([1, -1]))]
    df = df[(df['animal'] == animal_id) & (df['ABL'] == ABL) & (df['abs_ILD'] == abs_ILD) \
        & ((df['RTwrtStim'] <= 1) & (df['RTwrtStim'] >= 0))]
    
    n_trials = len(df)

    bin_centers = (bins[:-1] + bins[1:]) / 2
    if df.empty:
        print(f"No data found for batch {batch_name}, animal {animal_id}, ABL {ABL}, abs_ILD {abs_ILD}. Returning NaNs.")
        rtd_hist = np.full_like(bin_centers, np.nan)
        return bin_centers, rtd_hist
    df = df[df['RTwrtStim'] <= 1]
    if len(df) == 0:
        print(f"No trials with RTwrtStim <= 1 for batch {batch_name}, animal {animal_id}, ABL {ABL}, abs_ILD {abs_ILD}. Returning NaNs.")
        rtd_hist = np.full_like(bin_centers, np.nan)
        return bin_centers, rtd_hist
    rtd_hist, _ = np.histogram(df['RTwrtStim'], bins=bins, density=True)
    return bin_centers, rtd_hist, n_trials
# %%
ABL_arr = [20, 40, 60]
abs_ILD_arr = [1, 2, 4, 8, 16]
rt_bins = np.arange(0, 1.02, 0.02)  # 0 to 1 second in 0.02s bins

def process_batch_animal(batch_animal_pair):
    batch_name, animal_id = batch_animal_pair
    print(f"Processing batch {batch_name}, animal {animal_id}")
    animal_rtd_data = {}
    try:
        for abl in ABL_arr:
            print(f"Animal = {batch_name},{animal_id}, Processing ABL {abl}")
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
                print(f"  Processed stimulus ABL={abl}, ILD={abs_ild}")
    except Exception as e:
        print(f"Error processing batch {batch_name}, animal {animal_id}: {str(e)}")
    return batch_animal_pair, animal_rtd_data

# %%
n_jobs = max(1, os.cpu_count() - 1)
results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(process_batch_animal)(batch_animal_pair) for batch_animal_pair in batch_animal_pairs
)
rtd_data = {}
for batch_animal_pair, animal_rtd_data in results:
    if animal_rtd_data:
        rtd_data[batch_animal_pair] = animal_rtd_data
print(f"Completed processing {len(rtd_data)} batch-animal pairs")
# %%
## Plot plane RTDs
max_xlim_RT = 1
# Create a single row of subplots, one for each abs_ILD
fig, axes = plt.subplots(1, len(abs_ILD_arr), figsize=(12, 3), sharex=True, sharey=True)

# Make sure axes is always a 1D array even if there's only one subplot
if len(abs_ILD_arr) == 1:
    axes = np.array([axes])

# Set up the aesthetics for all axes
for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Define colors for different ABL values
abl_colors = {20: 'tab:blue', 40: 'tab:orange', 60: 'tab:green'}

# Plot for each abs_ILD (columns)
for j, abs_ild in enumerate(abs_ILD_arr):
    ax = axes[j]
    
    # Plot for each ABL in the same subplot
    for abl in ABL_arr:
        # For each (ABL, abs(ILD)), average over (ABL, ILD) and (ABL, -ILD)
        empirical_rtds = []
        weights = []
        bin_centers = None
        
        for ild in [abs_ild, -abs_ild]:
            stim_key = (abl, ild)
            for batch_animal_pair, animal_data in rtd_data.items():
                if stim_key in animal_data:
                    emp_data = animal_data[stim_key]['empirical']
                    if not np.all(np.isnan(emp_data['rtd_hist'])):
                        empirical_rtds.append(emp_data['rtd_hist'])
                        bin_centers = emp_data['bin_centers']
                        weights.append(emp_data['n_trials'])

        # if ild in [1,-1]:
        #     print(f'ABL = {abl}, empirical_rtds.shape = {np.shape(empirical_rtds)}')
        
        # Plot empirical RTDs only
        if empirical_rtds and bin_centers is not None:
            empirical_rtds = np.array(empirical_rtds)
            weights = np.array(weights)
            if np.sum(weights) > 0:
                avg_empirical_rtd = np.nansum(empirical_rtds.T * weights, axis=1) / np.sum(weights)
            else:
                avg_empirical_rtd = np.full(empirical_rtds.shape[1], np.nan)

            ax.plot(bin_centers, avg_empirical_rtd, color=abl_colors[abl], linewidth=1.5, label=f'ABL={abl}')
            # print(f'area = {trapezoid(avg_empirical_rtd, bin_centers)}')
            edges = rt_bins                               # the array you used to build the histogram
            step_vals = np.repeat(avg_empirical_rtd, 2)   # turn it into a step function
            edge_pts  = np.repeat(edges, 2)[1:-1]
            area = np.trapz(step_vals, x=edge_pts)        # ~1.0
            # print(f'area = {area}')
    
    # Set up the axes
    ax.set_xlabel('RT (s)', fontsize=12)
    max_xlim_RT = 0.6
    max_ylim = 12
    ax.set_xticks([0, max_xlim_RT])
    ax.set_xticklabels(['0', max_xlim_RT], fontsize=12)
    ax.set_xlim(0, max_xlim_RT)
    ax.set_yticks([0, max_ylim])
    ax.set_ylim(0, max_ylim)
    ax.set_title(f'|ILD|={abs_ild}', fontsize=12)
    
    # Add legend only to the first subplot
    if j == 0:
        # ax.legend(fontsize=10)
        ax.set_ylabel('Density', fontsize=12)

# Set tick parameters for all axes
for ax in axes:
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
plt.tight_layout()

plt.savefig('og_rtds.png')
# save pdf
plt.savefig('og_rtds.pdf')
# plt.

# %%
# --- Plot all individual RTDs (not mean) for each abs_ILD and ABL=20 only ---
# fig_all_rtds, axes_all_rtds = plt.subplots(1, len(abs_ILD_arr), figsize=(12, 4), sharex=True, sharey=True)
# if len(abs_ILD_arr) == 1:
#     axes_all_rtds = np.array([axes_all_rtds])

# for ax in axes_all_rtds:
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)

# for j, abs_ild in enumerate(abs_ILD_arr):
#     ax = axes_all_rtds[j]
#     for abl in [20]:   # Only plot for ABL 20
#         for ild in [abs_ild, -abs_ild]:
#             stim_key = (abl, ild)
#             for batch_animal_pair, animal_data in rtd_data.items():
#                 if stim_key in animal_data:
#                     emp_data = animal_data[stim_key]['empirical']
#                     rtd_hist = emp_data['rtd_hist']
#                     bin_centers = emp_data['bin_centers']
#                     if not np.all(np.isnan(rtd_hist)):
#                         ax.plot(bin_centers, rtd_hist, color=abl_colors[abl], alpha=0.25, linewidth=1)
#     ax.set_xlabel('RT (s)', fontsize=12)
#     ax.set_xticks([0, max_xlim_RT])
#     ax.set_xticklabels(['0', max_xlim_RT], fontsize=12)
#     ax.set_xlim(0, max_xlim_RT)
#     ax.set_yticks([0, 15])
#     ax.set_ylim(0, 16)
#     ax.set_title(f'|ILD|={abs_ild}', fontsize=12)
#     if j == 0:
#         from matplotlib.lines import Line2D
#         custom_lines = [Line2D([0], [0], color=abl_colors[20], lw=2)]
#         ax.legend(custom_lines, ['ABL=20'], fontsize=10)
#         ax.set_ylabel('Probability Density', fontsize=12)

# for ax in axes_all_rtds:
#     ax.tick_params(axis='x', labelsize=12)
#     ax.tick_params(axis='y', labelsize=12)

# fig_all_rtds.suptitle('All Individual RTDs for ABL=20 (no averaging)', fontsize=14)
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# %%
# --- CDF Plot: 1x5 subplots for each abs_ILD, CDF for each ABL ---
fig_cdf, axes_cdf = plt.subplots(1, len(abs_ILD_arr), figsize=(12, 4), sharex=True, sharey=True)
if len(abs_ILD_arr) == 1:
    axes_cdf = np.array([axes_cdf])

for ax in axes_cdf:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

for j, abs_ild in enumerate(abs_ILD_arr):
    ax = axes_cdf[j]
    for abl in ABL_arr:
        empirical_rtds = []
        weights = []
        bin_centers = None
        for ild in [abs_ild, -abs_ild]:
            stim_key = (abl, ild)
            for batch_animal_pair, animal_data in rtd_data.items():
                if stim_key in animal_data:
                    emp_data = animal_data[stim_key]['empirical']
                    if not np.all(np.isnan(emp_data['rtd_hist'])):
                        empirical_rtds.append(emp_data['rtd_hist'])
                        bin_centers = emp_data['bin_centers']
                        weights.append(emp_data['n_trials'])
        if empirical_rtds and bin_centers is not None:
            empirical_rtds = np.array(empirical_rtds)
            weights = np.array(weights)
            if np.sum(weights) > 0:
                avg_empirical_rtd = np.nansum(empirical_rtds.T * weights, axis=1) / np.sum(weights)
            else:
                avg_empirical_rtd = np.full(empirical_rtds.shape[1], np.nan)
            # Compute CDF: sum of PDF * bin width, normalized to 1
            bin_widths = np.diff(rt_bins)
            # CDF at bin centers: cumulative sum of PDF * bin width
            cdf = np.cumsum(avg_empirical_rtd * bin_widths)
            # Ensure CDF ends at 1 (or close)
            cdf = cdf / cdf[-1] if cdf[-1] > 0 else cdf
            ax.plot(bin_centers, cdf, color=abl_colors[abl], linewidth=1.5, label=f'ABL={abl}')
    ax.set_xlabel('RT (s)', fontsize=12)
    ax.set_xticks([0, max_xlim_RT])
    ax.set_xticklabels(['0', max_xlim_RT], fontsize=12)
    ax.set_xlim(0, max_xlim_RT)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylim(0, 1.05)
    ax.set_title(f'|ILD|={abs_ild}', fontsize=12)
    if j == 0:
        ax.legend(fontsize=10)
        ax.set_ylabel('Cumulative Probability', fontsize=12)
for ax in axes_cdf:
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
fig_cdf.suptitle('CDF of RTD for each |ILD| and ABL', fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# %%


# %%
# --- QQ plots: Compare quantiles of ABL=60 vs ABL=20 and 40 for each abs_ILD ---
quantile_levels = np.arange(0.01, 0.9, 0.01)
quantiles_by_abl_ild = {abl: {abs_ild: None for abs_ild in abs_ILD_arr} for abl in ABL_arr}

for abs_ild in abs_ILD_arr:
    for abl in ABL_arr:
        empirical_rtds = []
        weights = []
        bin_centers = None
        for ild in [abs_ild, -abs_ild]:
            stim_key = (abl, ild)
            for batch_animal_pair, animal_data in rtd_data.items():
                if stim_key in animal_data:
                    emp_data = animal_data[stim_key]['empirical']
                    if not np.all(np.isnan(emp_data['rtd_hist'])):
                        empirical_rtds.append(emp_data['rtd_hist'])
                        bin_centers = emp_data['bin_centers']
                        weights.append(emp_data['n_trials'])
        if empirical_rtds and bin_centers is not None:
            empirical_rtds = np.array(empirical_rtds)
            weights = np.array(weights)
            if np.sum(weights) > 0:
                avg_empirical_rtd = np.nansum(empirical_rtds.T * weights, axis=1) / np.sum(weights)
            else:
                avg_empirical_rtd = np.full(empirical_rtds.shape[1], np.nan)
            bin_widths = np.diff(rt_bins)
            cdf = np.cumsum(avg_empirical_rtd * bin_widths)
            cdf = cdf / cdf[-1] if cdf[-1] > 0 else cdf
            # Interpolate to get RTs at desired quantiles
            quantile_rts = np.interp(quantile_levels, cdf, bin_centers)
            quantiles_by_abl_ild[abl][abs_ild] = quantile_rts
        else:
            quantiles_by_abl_ild[abl][abs_ild] = np.full_like(quantile_levels, np.nan)

# Now plot QQ plots for each abs_ILD
fit_results = {abs_ild: {} for abs_ild in abs_ILD_arr}
min_RT_cut = 0.07
min_RT_cut_ild_16 = 0.06
# min_cuts_for_cases = {
#     (20, 1): 
# }

for abs_ild in abs_ILD_arr:
    q_60 = quantiles_by_abl_ild[60][abs_ild]
    q_20 = quantiles_by_abl_ild[20][abs_ild]
    q_40 = quantiles_by_abl_ild[40][abs_ild]
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))

    # ABL 20 vs 60
    if abs_ild != 16:
        # mask 20 and 40 from min_RT_cut
        mask_20 = q_20 >= min_RT_cut
        mask_40 = q_40 >= min_RT_cut
        min_x_cut = min_RT_cut
    else:
        mask_20 = q_20 >= min_RT_cut_ild_16
        mask_40 = q_40 >= min_RT_cut_ild_16
        min_x_cut = min_RT_cut_ild_16

    
    if np.sum(mask_20) >= 2:
        # Fit only slope (no intercept): solve q_20 = slope * q_60
        # slope_20 = np.linalg.lstsq(q_60[mask_20] - min_x_cut , q_20[mask_20] - q_60[mask_20]  , rcond=None)[0][0]
        x = q_60[mask_20] - min_x_cut
        y = q_20[mask_20] - q_60[mask_20]
        y_first = y[0]
        y = y - y_first
        slope_20 = np.sum(x * y) / np.sum(x ** 2)
        fit_results[abs_ild][20] = {'slope': slope_20, 'intercept': 0.0}
        x_fit = [min_x_cut, np.nanmax(q_60[mask_20])]
        y_fit = [0, slope_20 * (x_fit[1] - min_x_cut)]
        if abs_ild == 16:
            min_max_mask_ild_16_abl_20 = (q_20 >= min_x_cut) & (q_20 <= 0.13)
            x = q_60[min_max_mask_ild_16_abl_20] - min_x_cut
            y = q_20[min_max_mask_ild_16_abl_20] - q_60[min_max_mask_ild_16_abl_20]
            y_first = y[0]
            # print(f'y[0] in 16, 20 = {y_first}')
            q_diff = q_20 - q_60
            print(q_diff[:10])
            print(f' min x cut = {min_x_cut}')
            y = y - y_first
            slope_20 = np.sum(x * y) / np.sum(x ** 2)
            fit_results[abs_ild][20] = {'slope': slope_20, 'intercept': 0.0}
            x_fit = [min_x_cut, np.nanmax(q_60[min_max_mask_ild_16_abl_20])]
            y_fit = [0, slope_20 * (x_fit[1] - min_x_cut)]
            # reupdate abs ILD 8 with what we got at ILD 16
            # fit_results[8][20] = {'slope': slope_20, 'intercept': 0.0}
            fit_results[8][20] = {'slope': 0.45, 'intercept': 0.0}
            fit_results[16][20] = {'slope': 0.3, 'intercept': 0.0}
            fit_results[16][40] = {'slope': 0.3, 'intercept': 0.0}


        ax.plot(x_fit, y_fit, color=abl_colors[20], linestyle='-', label=f'Fit 20 (m={slope_20:.2f})')
    else:
        fit_results[abs_ild][20] = {'slope': np.nan, 'intercept': np.nan}
    # ABL 40 vs 60
    if np.sum(mask_40) >= 2:
        # slope_40 = np.linalg.lstsq(q_60[mask_40] - min_x_cut, q_40[mask_40] - q_60[mask_40] , rcond=None)[0][0]
        x = q_60[mask_40] - min_x_cut
        y = q_40[mask_40] - q_60[mask_40]
        y_first = y[0]
        y = y - y_first
        
        if abs_ild == 16:
            slope_40 = 0
        else:
            slope_40 = np.sum(x * y) / np.sum(x ** 2)
        fit_results[abs_ild][40] = {'slope': slope_40, 'intercept': 0.0}
        x_fit = [min_x_cut, np.nanmax(q_60[mask_40])]
        y_fit = [0, slope_40 * (x_fit[1] - min_x_cut)]
        ax.plot(x_fit, y_fit, color=abl_colors[40], linestyle='--', label=f'Fit 40 (m={slope_40:.2f})')
    else:
        fit_results[abs_ild][40] = {'slope': np.nan, 'intercept': np.nan}

    # Plot points and identity
    ax.plot(q_60, q_20 - q_60, 'o-', label='ABL 20 vs 60', color=abl_colors[20])
    ax.plot(q_60, q_40 - q_60, 's--', label='ABL 40 vs 60', color=abl_colors[40])
    # ax.plot([np.nanmin(q_60), np.nanmax(q_60)], [np.nanmin(q_60), np.nanmax(q_60)], 'k:', lw=1)
    # for i, q in enumerate(quantile_levels):
    #     if not np.isnan(q_60[i]) and not np.isnan(q_20[i]):
    #         ax.annotate(f'{int(q*100)}%', (q_60[i], q_20[i]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=9, color=abl_colors[20])
    #     if not np.isnan(q_60[i]) and not np.isnan(q_40[i]):
    #         ax.annotate(f'{int(q*100)}%', (q_60[i], q_40[i]), textcoords="offset points", xytext=(0,-12), ha='center', fontsize=9, color=abl_colors[40])
    ax.set_xlabel('ABL 60 RT (s)', fontsize=12)
    ax.set_ylabel('ABL 20/40 RT (s)', fontsize=12)
    ax.set_title(f'QQ plot |ILD|={abs_ild}')
    ax.legend()
    ax.set_xticks(np.arange(0, 0.3, 0.01))
    ax.grid(True)
    ax.set_xlim(0, 0.3)
    plt.tight_layout()

# Print out the fit results
print('QQ plot fit results (slope, intercept):')
for abs_ild in abs_ILD_arr:
    for abl in [20, 40]:
        res = fit_results[abs_ild][abl]
        print(f'abs_ILD={abs_ild}, ABL={abl}: slope={res["slope"]:.3f}, intercept={res["intercept"]:.3f}')

# %%
# --- Rescaled RTD plot: x-axis of ABL 20/40 rescaled using fit_results ---
min_RT_cut = 0.07
min_RT_cut_ild_16 = 0.055
fig_rescaled, axes_rescaled = plt.subplots(1, len(abs_ILD_arr), figsize=(12, 4), sharex=True, sharey=True)
if len(abs_ILD_arr) == 1:
    axes_rescaled = np.array([axes_rescaled])

for ax in axes_rescaled:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

for j, abs_ild in enumerate(abs_ILD_arr):
    if abs_ild != 16:
        min_x_cut = min_RT_cut
    else:
        min_x_cut = min_RT_cut_ild_16
    ax = axes_rescaled[j]
    for abl in ABL_arr:
        empirical_rtds = []
        weights = []
        bin_centers = None
        for ild in [abs_ild, -abs_ild]:
            stim_key = (abl, ild)
            for batch_animal_pair, animal_data in rtd_data.items():
                if stim_key in animal_data:
                    emp_data = animal_data[stim_key]['empirical']
                    if not np.all(np.isnan(emp_data['rtd_hist'])):
                        empirical_rtds.append(emp_data['rtd_hist'])
                        bin_centers = emp_data['bin_centers']
                        weights.append(emp_data['n_trials'])

        bin_centers_mask = bin_centers > min_x_cut
        
        if empirical_rtds and bin_centers is not None:
            empirical_rtds = np.array(empirical_rtds)
            weights = np.array(weights)
            if np.sum(weights) > 0:
                avg_empirical_rtd = np.nansum(empirical_rtds.T * weights, axis=1) / np.sum(weights)
            else:
                avg_empirical_rtd = np.full(empirical_rtds.shape[1], np.nan)
            # Rescale x-axis for ABL 20 and 40
            if abl == 60:
                xvals = bin_centers
                ax.plot(xvals, avg_empirical_rtd, color=abl_colors[abl], linewidth=1.5, label=f'ABL={abl}')

            else:
                slope = fit_results[abs_ild][abl]['slope']
                if slope + 1 != 0 and not np.isnan(slope):
                    xvals =( (bin_centers - min_x_cut) / (1 + slope) ) + min_x_cut
                else:
                    print(f'slope is {slope}')
                    raise ValueError(f"Invalid slope for abs_ILD={abs_ild}, ABL={abl}")
                multiplier = np.ones_like(avg_empirical_rtd)
                multiplier[bin_centers_mask] = slope + 1
                rescaled_rtd = avg_empirical_rtd * multiplier
                print(f'rescaled_rtd shape = {rescaled_rtd.shape}')
                ax.plot(xvals, rescaled_rtd, color=abl_colors[abl], linewidth=1.5, label=f'ABL={abl}')
    ax.set_xlabel('RT (s)', fontsize=12)
    max_xlim_RT = 0.6
    max_ylim = 12
    ax.set_xticks([0, max_xlim_RT])
    ax.set_xticklabels(['0', max_xlim_RT], fontsize=12)
    ax.set_xlim(0, max_xlim_RT)
    ax.set_yticks([0, max_ylim])
    ax.set_ylim(0, max_ylim)
    ax.set_title(f'|ILD|={abs_ild}', fontsize=12)
    if j == 0:
        # ax.legend(fontsize=10)
        ax.set_ylabel('Density', fontsize=12)
for ax in axes_rescaled:
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
plt.tight_layout()

plt.savefig('rescaled_rtds.png')
plt.savefig('rescaled_rtds_4.pdf')
# fig_rescaled.suptitle('RTD rescaled', fontsize=14)
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# %%
