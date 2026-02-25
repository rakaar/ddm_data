# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from joblib import Parallel, delayed

# %%
# Load ASD WT dataset (already filtered upstream)
INPUT_CSV = os.path.join(os.path.dirname(__file__), '..', 'asd_wt.csv')
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

print(f"Loading {INPUT_CSV}...")
merged_data = pd.read_csv(INPUT_CSV)
merged_valid = merged_data[merged_data['success'].isin([1, -1])].copy()

animals = sorted(merged_valid['animal'].dropna().astype(str).unique())
print(f"Found {len(animals)} animals: {', '.join(animals)}")

# %%
def get_animal_RTD_data(df, animal_id, ABL, abs_ILD, bins, fit_q_levels):
    """Calculates RTD histogram + fitting quantiles for one animal and condition."""
    df['abs_ILD'] = df['ILD'].abs()
    
    # Filter for the specific condition
    condition_df = df[(df['ABL'] == ABL) & (df['abs_ILD'] == abs_ILD) & (df['timed_rt'] >= 0) & (df['timed_rt'] <= 1)]
    
    n_trials = len(condition_df)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    if condition_df.empty or n_trials == 0:
        # No print statement here to avoid clutter during parallel execution
        rtd_hist = np.full_like(bin_centers, np.nan)
        fit_quantiles = np.full(len(fit_q_levels), np.nan)
        return bin_centers, rtd_hist, fit_quantiles, n_trials

    rtd_hist, _ = np.histogram(condition_df['timed_rt'], bins=bins, density=True)
    if n_trials < 5:
        fit_quantiles = np.full(len(fit_q_levels), np.nan)
    else:
        fit_quantiles = condition_df['timed_rt'].quantile(fit_q_levels).values
    return bin_centers, rtd_hist, fit_quantiles, n_trials
# %%
ABL_arr = [20, 40, 60]
abs_ILD_arr = [1, 2, 4, 8, 16]
rt_bins = np.arange(0, 1.02, 0.02)  # 0 to 1 second in 0.02s bins
fitting_quantile_levels = np.arange(0.01, 1.0, 0.01)

def process_animal(animal_id, animal_df):
    """Wrapper function to process a single animal using pre-loaded data."""
    animal_rtd_data = {}
    try:
        for abl in ABL_arr:
            for abs_ild in abs_ILD_arr:
                stim_key = (abl, abs_ild)
                # Pass the animal-specific dataframe to the processing function
                bin_centers, rtd_hist, fit_quantiles, n_trials = get_animal_RTD_data(
                    animal_df, str(animal_id), abl, abs_ild, rt_bins, fitting_quantile_levels
                )
                animal_rtd_data[stim_key] = {
                    'empirical': {
                        'bin_centers': bin_centers,
                        'rtd_hist': rtd_hist,
                        'fitting_quantiles': fit_quantiles,
                        'n_trials': n_trials
                    }
                }
    except Exception as e:
        print(f"Error processing animal {animal_id}: {str(e)}")
    return str(animal_id), animal_rtd_data

# %%
# Group data by animal for efficient parallel processing
animal_groups = merged_valid.groupby(['animal'])

n_jobs = max(1, os.cpu_count() - 4) # Leave some cores free
print(f"Processing {len(animal_groups)} animals on {n_jobs} cores...")

results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(process_animal)(name, group) for name, group in animal_groups
)
rtd_data = {}
for animal_id, animal_rtd_data in results:
    if animal_rtd_data:
        rtd_data[animal_id] = animal_rtd_data
print(f"Completed processing {len(rtd_data)} animals")
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
        # Per-animal average RTD for this (ABL, |ILD|)
        empirical_rtds = []
        bin_centers = None
        stim_key = (abl, abs_ild)
        for animal_id, animal_data in rtd_data.items():
            if stim_key in animal_data:
                emp_data = animal_data[stim_key]['empirical']
                if not np.all(np.isnan(emp_data['rtd_hist'])):
                    empirical_rtds.append(emp_data['rtd_hist'])
                    bin_centers = emp_data['bin_centers']

        # if ild in [1,-1]:
        #     print(f'ABL = {abl}, empirical_rtds.shape = {np.shape(empirical_rtds)}')
        
        # Plot empirical RTDs only
        if empirical_rtds and bin_centers is not None:
            empirical_rtds = np.array(empirical_rtds)
            avg_empirical_rtd = np.nanmean(empirical_rtds, axis=0)

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
    max_ylim = 18
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

plt.savefig('asd_wt_og_rtds_quantile_avg_match.png')
# save pdf
plt.savefig('asd_wt_og_rtds_quantile_avg_match.pdf')
# plt.

# %%
# --- QQ plots: compare mean per-animal fitting quantiles (quantile-method style) ---
quantile_levels = fitting_quantile_levels
quantiles_by_abl_ild = {abl: {abs_ild: None for abs_ild in abs_ILD_arr} for abl in ABL_arr}

for abs_ild in abs_ILD_arr:
    for abl in ABL_arr:
        animal_quantiles = []
        stim_key = (abl, abs_ild)
        for animal_id, animal_data in rtd_data.items():
            if stim_key in animal_data:
                q = animal_data[stim_key]['empirical']['fitting_quantiles']
                if not np.all(np.isnan(q)):
                    animal_quantiles.append(q)
        if animal_quantiles:
            quantiles_by_abl_ild[abl][abs_ild] = np.nanmean(np.array(animal_quantiles), axis=0)
        else:
            quantiles_by_abl_ild[abl][abs_ild] = np.full_like(quantile_levels, np.nan)

# Now plot QQ plots for each abs_ILD
fit_results = {abs_ild: {} for abs_ild in abs_ILD_arr}
min_RT_cut_by_ILD = {1: 0.0625, 2: 0.0655, 4: 0.0595, 8: 0.0615, 16: 0.0505}
max_RT_cut = 0.3

for abs_ild in abs_ILD_arr:
    q_60 = quantiles_by_abl_ild[60][abs_ild]
    q_20 = quantiles_by_abl_ild[20][abs_ild]
    q_40 = quantiles_by_abl_ild[40][abs_ild]
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))

    min_RT_cut = min_RT_cut_by_ILD[abs_ild]
    min_x_cut = min_RT_cut

    mask_20 = (q_60 >= min_RT_cut) & (q_60 <= max_RT_cut)
    mask_40 = (q_60 >= min_RT_cut) & (q_60 <= max_RT_cut)

    if not np.any(np.isnan(q_60)) and not np.any(np.isnan(q_20)) and np.sum(mask_20) >= 2:
        x = q_60[mask_20] - min_x_cut
        y = (q_20 - q_60)[mask_20]
        y_intercept = y[0]
        y = y - y_intercept
        slope_20 = np.sum(x * y) / np.sum(x ** 2) if np.sum(x ** 2) > 0 else np.nan
        x_fit = [min_x_cut, np.nanmax(q_60[mask_20])]
        y_fit = [0, slope_20 * (x_fit[1] - min_x_cut)] if not np.isnan(slope_20) else [np.nan, np.nan]
        ax.plot(x_fit, y_fit, color=abl_colors[20], linestyle='-', label=f'Fit 20 (m={slope_20:.2f})')
        fit_results[abs_ild][20] = {'slope': slope_20, 'intercept': 0.0}
    else:
        fit_results[abs_ild][20] = {'slope': np.nan, 'intercept': np.nan}

    # ABL 40 vs 60
    if not np.any(np.isnan(q_60)) and not np.any(np.isnan(q_40)) and np.sum(mask_40) >= 2:
        x = q_60[mask_40] - min_x_cut
        y = (q_40 - q_60)[mask_40]
        y_intercept = y[0]
        y = y - y_intercept
        slope_40 = np.sum(x * y) / np.sum(x ** 2) if np.sum(x ** 2) > 0 else np.nan
        x_fit = [min_x_cut, np.nanmax(q_60[mask_40])]
        y_fit = [0, slope_40 * (x_fit[1] - min_x_cut)] if not np.isnan(slope_40) else [np.nan, np.nan]
        ax.plot(x_fit, y_fit, color=abl_colors[40], linestyle='--', label=f'Fit 40 (m={slope_40:.2f})')
        fit_results[abs_ild][40] = {'slope': slope_40, 'intercept': 0.0}
    else:
        fit_results[abs_ild][40] = {'slope': np.nan, 'intercept': np.nan}

    # Plot points and identity
    ax.plot(q_60, q_20 - q_60, 'o-', label='ABL 20 vs 60', color=abl_colors[20])
    ax.plot(q_60, q_40 - q_60, 's--', label='ABL 40 vs 60', color=abl_colors[40])
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
fig_rescaled, axes_rescaled = plt.subplots(1, len(abs_ILD_arr), figsize=(12, 4), sharex=True, sharey=True)
if len(abs_ILD_arr) == 1:
    axes_rescaled = np.array([axes_rescaled])

for ax in axes_rescaled:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

for j, abs_ild in enumerate(abs_ILD_arr):
    min_x_cut = min_RT_cut_by_ILD[abs_ild]
    ax = axes_rescaled[j]
    for abl in ABL_arr:
        empirical_rtds = []
        bin_centers = None
        stim_key = (abl, abs_ild)
        for animal_id, animal_data in rtd_data.items():
            if stim_key in animal_data:
                emp_data = animal_data[stim_key]['empirical']
                if not np.all(np.isnan(emp_data['rtd_hist'])):
                    empirical_rtds.append(emp_data['rtd_hist'])
                    bin_centers = emp_data['bin_centers']

        bin_centers_mask = bin_centers > min_x_cut
        
        if empirical_rtds and bin_centers is not None:
            empirical_rtds = np.array(empirical_rtds)
            avg_empirical_rtd = np.nanmean(empirical_rtds, axis=0)
            # Rescale x-axis for ABL 20 and 40
            if abl == 60:
                xvals = bin_centers
                ax.plot(xvals, avg_empirical_rtd, color=abl_colors[abl], linewidth=1.5, label=f'ABL={abl}')

            else:
                # print(fit_results)
                slope = fit_results[abs_ild][abl]['slope']
                if slope + 1 != 0 and not np.isnan(slope):
                    xvals =( (bin_centers - min_x_cut) / (1 + slope) ) + min_x_cut
                else:
                    print(f'slope is {slope}')
                    raise ValueError(f"Invalid slope for abs_ILD={abs_ild}, ABL={abl}")
                multiplier = np.ones_like(avg_empirical_rtd)
                multiplier[bin_centers_mask] = slope + 1
                rescaled_rtd = avg_empirical_rtd * multiplier
                ax.plot(xvals, rescaled_rtd, color=abl_colors[abl], linewidth=1.5, label=f'ABL={abl}')
    ax.set_xlabel('RT (s)', fontsize=12)
    max_xlim_RT = 0.6
    max_ylim = 18
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

plt.savefig('asd_wt_rescaled_rtds_quantile_avg_match.png')
plt.savefig('asd_wt_rescaled_rtds_quantile_avg_match.pdf')
# fig_rescaled.suptitle('RTD rescaled', fontsize=14)
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# %%
