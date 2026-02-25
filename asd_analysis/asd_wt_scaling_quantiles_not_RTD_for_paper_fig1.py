# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from joblib import Parallel, delayed
import pickle
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import NullFormatter, NullLocator
FILENAME_SUFFIX = "_asd_wt"
INPUT_CSV = os.path.join(os.path.dirname(__file__), "..", "asd_wt.csv")

min_RT_cut_by_ILD = {1: 0.0625, 2: 0.0655, 4: 0.0595, 8: 0.0615, 16: 0.0505}
does_min_RT_depend_on_ILD = True

if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

print(f"Loading {INPUT_CSV}...")
merged_data = pd.read_csv(INPUT_CSV)

animal_ids = sorted(merged_data["animal"].dropna().astype(str).unique())
print(f"Found {len(animal_ids)} animals")
print("Animals:", ", ".join(animal_ids))


# %%
def get_animal_quantile_data(df, ABL, abs_ILD, plot_q_levels, fit_q_levels):
    """Calculates RT quantiles from a pre-filtered DataFrame for a specific condition."""
    df['abs_ILD'] = df['ILD'].abs()
    
    # valid trials for a stim btn 0 and 1 RT
    condition_df = df[
        (df['ABL'] == ABL)
        & (df['abs_ILD'] == abs_ILD)
        & (df['timed_rt'] >= 0)
        & (df['timed_rt'] <= 1)
        & (df['success'].isin([1, -1]))
    ]
    
    n_trials = len(condition_df)

    if n_trials < 5: # Not enough data
        plot_quantiles = np.full(len(plot_q_levels), np.nan)
        fit_quantiles = np.full(len(fit_q_levels), np.nan)
    else:
        rt_series = condition_df['timed_rt']
        plot_quantiles = rt_series.quantile(plot_q_levels).values
        fit_quantiles = rt_series.quantile(fit_q_levels).values
        
    return plot_quantiles, fit_quantiles, n_trials

# %%
# params
ABL_arr = [20, 40, 60]
abs_ILD_arr = [1, 2, 4, 8, 16]
# plotting_quantiles = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
# plotting_quantiles = np.arange(0.05,0.95,0.01)
plotting_quantiles = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
fitting_quantiles = np.arange(0.01, 1.0, 0.01)
min_RT_cut = 0.09 # for slope fitting
max_RT_cut = 0.3 # for slope fitting
# %%
def process_animal(animal_id, animal_df):
    """Wrapper function to process a single animal using pre-loaded data."""
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
        print(f"Error processing animal {animal_id}: {str(e)}")
    return animal_id, animal_quantile_data

# %%
# Group data by animal for efficient parallel processing
animal_groups = merged_data.groupby(['animal'])

n_jobs = max(1, os.cpu_count() - 4)
print(f"Processing {len(animal_groups)} animals on {n_jobs} cores...")

results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(process_animal)(str(name), group) for name, group in animal_groups
)
quantile_data = {ani: data for ani, data in results if data}
print(f"Completed processing {len(quantile_data)} animals")

# %% 
abl_colors = {20: 'tab:blue', 40: 'tab:orange', 60: 'tab:green'}
quantile_colors = plt.cm.viridis(np.linspace(0, 1, len(plotting_quantiles)))

# --- Aggregators for average (across-animal) plot ---
avg_unscaled_collect = {abl: [] for abl in ABL_arr}
avg_scaled_collect = {abl: [] for abl in ABL_arr}
scaling_factor_rows = []

output_filename = f'animal_specific_quantile_scaling_plots{FILENAME_SUFFIX}.pdf'

with PdfPages(output_filename) as pdf:
    for animal_id, animal_data in quantile_data.items():

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
        fig1.suptitle(f'Original RT Quantiles vs. |ILD| - Animal: {animal_id}', fontsize=16)

        for i, abl in enumerate(ABL_arr):
            ax = axes1[i]
            q_matrix = unscaled_plot_quantiles[abl]
            for j, q_level in enumerate(plotting_quantiles):
                ax.plot(abs_ILD_arr, q_matrix[j, :], marker='o', linestyle='-', color=quantile_colors[j], label=f'{int(q_level*100)}th')
            ax.set_title(f'ABL = {abl} dB')
            ax.set_xlabel('|ILD|')
            ax.set_ylim(0.05, 0.45)
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
                # Determine min RT cut dynamically based on flag
                if does_min_RT_depend_on_ILD:
                    min_rt_cut = min_RT_cut_by_ILD.get(abs_ild, None)
                else:
                    min_rt_cut = min_RT_cut
                # Use fitting quantiles for slope calculation
                q_60_fit = q_60_fit_all_ilds[:, ild_idx]
                q_other_fit = q_other_fit_all_ilds[:, ild_idx]

                # --- Slope calculation using the original method with fine quantiles ---
                slope = np.nan # Default slope
                if not np.any(np.isnan(q_60_fit)) and not np.any(np.isnan(q_other_fit)):
                    mask = (q_60_fit >= min_rt_cut) & (q_60_fit <= max_RT_cut)
                    if np.sum(mask) >= 2:
                        q_other_minus_60 = q_other_fit - q_60_fit
                        
                        x_fit_calc = q_60_fit[mask] - min_rt_cut
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
                        q_other_plot > min_rt_cut,
                        ((q_other_plot - min_rt_cut) / (1 + slope)) + min_rt_cut,
                        q_other_plot
                    )
                    scaled_q_abl_per_ild.append(scaled_values)
                else:
                    scaled_q_abl_per_ild.append(q_other_plot)
            
            scaled_plot_quantiles[abl] = np.array(scaled_q_abl_per_ild).T

        # Save per-animal scaling factors in a comparable table format
        for ild_idx, abs_ild in enumerate(abs_ILD_arr):
            slope_20 = slopes[20][ild_idx]
            slope_40 = slopes[40][ild_idx]
            scaling_factor_rows.append({
                "animal": str(animal_id),
                "abs_ILD": abs_ild,
                "slope_20": slope_20,
                "sf_20": (1.0 + slope_20) if not np.isnan(slope_20) else np.nan,
                "slope_40": slope_40,
                "sf_40": (1.0 + slope_40) if not np.isnan(slope_40) else np.nan,
            })

        # Store this animal's matrices for aggregation
        for abl_key in ABL_arr:
            avg_unscaled_collect[abl_key].append(unscaled_plot_quantiles[abl_key])
            avg_scaled_collect[abl_key].append(scaled_plot_quantiles[abl_key])

        # --- Plot 2: Scaled Quantiles ---
        fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
        fig2.suptitle(f'Scaled RT Quantiles vs. |ILD| - Animal: {animal_id}', fontsize=16)

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
            ax.set_ylim(0.05, 0.3)
            if i == 0:
                ax.set_ylabel('Reaction Time (s) (Scaled to ABL 60)')
                ax.legend(title='Quantile')
            ax.set_xscale('log')
            ax.set_xticks(abs_ILD_arr)
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig2)
        plt.close(fig2)

print(f'PDF saved to {output_filename}')

# Print scaling-factor table for cross-script comparison
if scaling_factor_rows:
    sf_df = pd.DataFrame(scaling_factor_rows).sort_values(["animal", "abs_ILD"]).reset_index(drop=True)
    print("\n=== Quantile Script Scaling Factors (per animal, per |ILD|) ===")
    print(
        sf_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
# %%
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

# --- Average-across-animals plots saved as separate PNGs (unscaled & scaled) ---
orig_output_filename = f'average_quantile_unscaled{FILENAME_SUFFIX}_for_paper_fig1.png'
scaled_output_filename = f'average_quantile_scaled{FILENAME_SUFFIX}.png'
# %%
# -------- Original (unscaled) --------
fig_orig, axes_orig = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

abl_colors = ['tab:blue', 'tab:orange', 'tab:green']

for col, abl in enumerate(ABL_arr):
    ax = axes_orig[col]
    q_mat = mean_unscaled[abl]
    sem_mat = sem_unscaled[abl]
    for q_idx, q_level in enumerate(plotting_quantiles):
        ax.errorbar(abs_ILD_arr, q_mat[q_idx, :], yerr=sem_mat[q_idx, :], marker='o',
                    linestyle='-', color=abl_colors[col])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xscale('log')
    ax.set_xticks(abs_ILD_arr)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis='both', which='major', labelsize=18)

    ax.set_ylim(0.05, 0.45)
    ax.set_yticks([0.05, 0.25, 0.45])

    if col == 0:
        ax.set_ylabel('Mean RT(s)', fontsize=18)

for ax in axes_orig:
    ax.set_xlabel('|ILD| (dB)', fontsize=18)

# --- Save data for external plotting ---
output_dir = os.path.dirname(orig_output_filename)
if not output_dir:
    output_dir = '.'

quantile_plot_data = {
    'ABL_arr': ABL_arr,
    'abs_ILD_arr': abs_ILD_arr,
    'plotting_quantiles': plotting_quantiles,
    'mean_unscaled': mean_unscaled,
    'sem_unscaled': sem_unscaled,
    'mean_scaled': mean_scaled,      # Add scaled data
    'sem_scaled': sem_scaled,        # Add scaled data
    'abl_colors': ['tab:blue', 'tab:orange', 'tab:green'] # Hardcoded from plot
}

output_pickle_path = os.path.join(output_dir, 'asd_wt_fig1_quantiles_plot_data.pkl')
with open(output_pickle_path, 'wb') as f:
    pickle.dump(quantile_plot_data, f)
print(f"\nQuantile plot data saved to '{output_pickle_path}'")


plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig_orig.savefig(orig_output_filename, dpi=300, bbox_inches='tight')
plt.show(fig_orig)
print(f'Average-across-animals unscaled PNG saved to {orig_output_filename}')
# %%
# -------- Scaled --------
fig_scaled, axes_scaled = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
fig_scaled.suptitle('Average RT Quantiles Across Animals (Scaled)', fontsize=18)

for col, abl in enumerate(ABL_arr):
    ax = axes_scaled[col]
    q_mat = mean_scaled[abl]
    sem_mat = sem_scaled[abl]
    for q_idx, q_level in enumerate(plotting_quantiles):
        ax.errorbar(abs_ILD_arr, q_mat[q_idx, :], yerr=sem_mat[q_idx, :], marker='o',
                    linestyle='-', color=quantile_colors[q_idx], label=f'{int(q_level*100)}th')
    ax.set_title(f'ABL = {abl} dB')
    ax.set_xscale('log')
    ax.set_xticks(abs_ILD_arr)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    if col == 0:
        ax.set_ylabel('Reaction Time (s) (Scaled)')
        ax.legend(title='Quantile')
for ax in axes_scaled:
    ax.set_xlabel('|ILD|')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig_scaled.savefig(scaled_output_filename, dpi=300, bbox_inches='tight')
plt.show(fig_scaled)
print(f'Average-across-animals scaled PNG saved to {scaled_output_filename}')

# %%
# --------- Overlay figure of scaled quantiles (all ABLs on one axes) ---------
overlay_filename = f'average_scaled_quantiles_overlay{FILENAME_SUFFIX}.png'

fig_overlay, ax_overlay = plt.subplots(figsize=(6, 4))

abl_colors = ['tab:blue', 'tab:orange', 'tab:green']

for col, abl in enumerate(ABL_arr):
    q_mat = mean_scaled[abl]
    sem_mat = sem_scaled[abl]
    for q_idx, q_level in enumerate(plotting_quantiles):
        ax_overlay.errorbar(
            abs_ILD_arr,
            q_mat[q_idx, :],
            yerr=sem_mat[q_idx, :],
            marker='o',
            linestyle='-',
            color=abl_colors[col]
        )

ax_overlay.spines['right'].set_visible(False)
ax_overlay.spines['top'].set_visible(False)

ax_overlay.set_xscale('log')
ax_overlay.set_xticks(abs_ILD_arr)
ax_overlay.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax_overlay.xaxis.set_minor_formatter(NullFormatter())
ax_overlay.xaxis.set_minor_locator(NullLocator())
ax_overlay.tick_params(axis='both', which='major', labelsize=18)

ax_overlay.set_ylim(0.05, 0.3)
ax_overlay.set_yticks([0.05, 0.15, 0.25])

ax_overlay.set_xlabel('|ILD| (dB)', fontsize=18)
ax_overlay.set_ylabel('Scaled RT (s)', fontsize=18)

plt.tight_layout()
fig_overlay.savefig(overlay_filename, dpi=300, bbox_inches='tight')
plt.show(fig_overlay)

print(f'Scaled-overlay quantiles PNG saved to {overlay_filename}')

# %%
# --------- Quantiles vs ABL grid (2 rows × 5 ILDs) ---------
abl_grid_filename = f'quantiles_vs_ABL_grid{FILENAME_SUFFIX}.pdf'

with PdfPages(abl_grid_filename) as pdf_grid:
    fig_grid, axes_grid = plt.subplots(2, len(abs_ILD_arr), figsize=(20, 6), sharey='row')
    fig_grid.suptitle('Average log RT Quantiles Across ABLs for each |ILD|', fontsize=18)

    for col_idx, abs_ild in enumerate(abs_ILD_arr):
        # --- Original row (0) ---
        ax_orig = axes_grid[0, col_idx]
        for q_idx, q_level in enumerate(plotting_quantiles):
            y_raw = np.array([mean_unscaled[abl][q_idx, col_idx] for abl in ABL_arr])
            y_err_raw = np.array([sem_unscaled[abl][q_idx, col_idx] for abl in ABL_arr])
            y_vals = np.log(y_raw)
            y_errs = y_err_raw / y_raw  # propagate SEM under log transform
            ax_orig.errorbar(ABL_arr, y_vals, yerr=y_errs, marker='o', linestyle='-', color=quantile_colors[q_idx], alpha=0.9)
        ax_orig.set_title(f'|ILD| = {abs_ild}')
        ax_orig.set_xticks(ABL_arr)
        if col_idx == 0:
            ax_orig.set_ylabel('log RT (s)')
        ax_orig.grid(True, axis='y', ls=':', alpha=0.3)

        # --- Scaled row (1) ---
        ax_scaled = axes_grid[1, col_idx]
        for q_idx, q_level in enumerate(plotting_quantiles):
            y_raw = np.array([mean_scaled[abl][q_idx, col_idx] for abl in ABL_arr])
            y_err_raw = np.array([sem_scaled[abl][q_idx, col_idx] for abl in ABL_arr])
            y_vals = np.log(y_raw)
            y_errs = y_err_raw / y_raw  # propagate SEM under log transform
            ax_scaled.errorbar(ABL_arr, y_vals, yerr=y_errs, marker='o', linestyle='-', color=quantile_colors[q_idx], alpha=0.9)
        ax_scaled.set_xticks(ABL_arr)
        if col_idx == 0:
            ax_scaled.set_ylabel('log RT (s) (Scaled)')
        ax_scaled.set_xlabel('ABL (dB)')
        ax_scaled.grid(True, axis='y', ls=':', alpha=0.3)

    # Legends: add to the far right outside first row
    from matplotlib.lines import Line2D
    quantile_handles = [Line2D([0], [0], color=quantile_colors[i], marker='o', linestyle='-', label=f'{int(q*100)}th') for i, q in enumerate(plotting_quantiles)]
    fig_grid.legend(handles=quantile_handles, title='Quantile', loc='center left', bbox_to_anchor=(1.01, 0.5))

    plt.tight_layout(rect=[0, 0, 0.97, 0.95])
    pdf_grid.savefig(fig_grid)
    plt.close(fig_grid)

print(f'Quantiles-vs-ABL grid PDF saved to {abl_grid_filename}')

# %%
# Scatter of Quantiles between ABL pairs (saved as PNG)
print("Generating scatter figures comparing ABL pairs…")

abl_pairs = [(20, 60), (20, 40), (40, 60)]  # (x-axis ABL, y-axis ABL)
row_labels = ["Original", "Scaled"]

for abl_x, abl_y in abl_pairs:
    # File name e.g. quantile_scatter_ABL20_vs60.png
    scatter_filename = f'quantile_scatter_ABL{abl_x}_vs{abl_y}{FILENAME_SUFFIX}.png'

    fig, axes = plt.subplots(2, len(abs_ILD_arr), figsize=(20, 8), sharex=False, sharey=False)
    fig.suptitle(f'Across-Animal Quantile Comparison: ABL {abl_x} dB vs {abl_y} dB', fontsize=18)

    for row_idx, (dataset, label) in enumerate(zip([mean_unscaled, mean_scaled], row_labels)):
        for col_idx, abs_ild in enumerate(abs_ILD_arr):
            ax = axes[row_idx, col_idx]

            # Extract quantile vectors (length = len(plotting_quantiles))
            x_vals = dataset[abl_x][:, col_idx]
            y_vals = dataset[abl_y][:, col_idx]

            # Filter out NaNs to avoid warnings
            valid_mask = ~np.isnan(x_vals) & ~np.isnan(y_vals)
            x_vals_plot = x_vals[valid_mask]
            y_vals_plot = y_vals[valid_mask]
            q_colors_plot = quantile_colors[valid_mask]

            # Scatter plot
            ax.scatter(x_vals_plot, y_vals_plot, c=q_colors_plot, s=40)

            # Identity line
            min_lim = np.nanmin([x_vals_plot.min() if x_vals_plot.size else np.nan,
                                 y_vals_plot.min() if y_vals_plot.size else np.nan])
            max_lim = np.nanmax([x_vals_plot.max() if x_vals_plot.size else np.nan,
                                 y_vals_plot.max() if y_vals_plot.size else np.nan])
            if not np.isnan(min_lim) and not np.isnan(max_lim):
                padding = 0.02 * (max_lim - min_lim)
                ax.plot([min_lim - padding, max_lim + padding], [min_lim - padding, max_lim + padding], 'k--', linewidth=1)
                ax.set_xlim(min_lim - padding, max_lim + padding)
                ax.set_ylim(min_lim - padding, max_lim + padding)
            ax.set_aspect('equal', adjustable='box')

            # Titles & labels
            if row_idx == 0:
                ax.set_title(f'|ILD| = {abs_ild}')
            if col_idx == 0:
                ax.set_ylabel(label)
            if row_idx == len(row_labels) - 1:
                ax.set_xlabel(f'Quantiles at ABL {abl_x} (s)')
            if col_idx == len(abs_ILD_arr) - 1:
                ax.annotate(f'ABL {abl_y}', xy=(1.05, 0.5), xycoords='axes fraction', rotation=-90,
                             va='center', ha='left')

    # Global colour legend (quantile levels) on the right
    from matplotlib.lines import Line2D
    quantile_handles = [Line2D([0], [0], marker='o', linestyle='None', color=quantile_colors[i], label=f'{int(q*100)}th')
                        for i, q in enumerate(plotting_quantiles)]
    fig.legend(handles=quantile_handles, title='Quantile', loc='center left', bbox_to_anchor=(1.02, 0.5))

    plt.tight_layout(rect=[0, 0, 0.97, 0.94])
    fig.savefig(scatter_filename, dpi=300)
    plt.close(fig)
    print(f'Saved {scatter_filename}')

# %%
