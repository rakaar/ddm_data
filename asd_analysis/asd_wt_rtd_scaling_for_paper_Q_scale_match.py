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
    """Calculates RTD and fitting quantiles from a pre-filtered DataFrame for a specific animal."""
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
fitting_quantile_levels = np.arange(0.01, 1.0, 0.01)
rt_bins = np.arange(0, 1.01, 0.01)  # 0 to 1 second in 0.01s bins
BIN_CENTERS = (rt_bins[:-1] + rt_bins[1:]) / 2

# Cutoff mode for scaling:
# True  -> fixed 100 ms cutoff for all ILDs
# False -> ILD-wise cutoffs from MIN_RT_CUT_BY_ILD
USE_FIXED_100MS_CUTOFF = False
FIXED_MIN_RT_CUT = 0.1
MIN_RT_CUT_BY_ILD = {1: 0.0625, 2: 0.0655, 4: 0.0595, 8: 0.0615, 16: 0.0505}

def get_min_rt_cut(abs_ild):
    if USE_FIXED_100MS_CUTOFF:
        return FIXED_MIN_RT_CUT
    return MIN_RT_CUT_BY_ILD[abs_ild]

print(
    "Cutoff mode:",
    "fixed_100ms" if USE_FIXED_100MS_CUTOFF else "ild_wise",
    f"(fixed={FIXED_MIN_RT_CUT:.4f}s)",
)

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

plt.savefig('asd_wt_og_rtds_q_scale_match.png')
# save pdf
plt.savefig('asd_wt_og_rtds_q_scale_match.pdf')
# plt.

# %%
# --- QQ plots: use per-animal fitting quantiles and quantile-style slope fitting ---
quantile_levels = fitting_quantile_levels
quantiles_by_abl_ild = {abl: {abs_ild: None for abs_ild in abs_ILD_arr} for abl in ABL_arr}
max_RT_cut = 0.3

# Mean quantiles across animals (for QQ display only)
for abs_ild in abs_ILD_arr:
    for abl in ABL_arr:
        stim_key = (abl, abs_ild)
        animal_quantiles = []
        for animal_id, animal_data in rtd_data.items():
            if stim_key in animal_data:
                q = animal_data[stim_key]['empirical']['fitting_quantiles']
                if not np.all(np.isnan(q)):
                    animal_quantiles.append(q)
        if animal_quantiles:
            quantiles_by_abl_ild[abl][abs_ild] = np.nanmean(np.array(animal_quantiles), axis=0)
        else:
            quantiles_by_abl_ild[abl][abs_ild] = np.full_like(quantile_levels, np.nan)

# Per-animal slopes (same method as quantile scaling script)
slopes_by_animal = {animal_id: {abs_ild: {20: np.nan, 40: np.nan} for abs_ild in abs_ILD_arr} for animal_id in rtd_data.keys()}
for animal_id, animal_data in rtd_data.items():
    for abs_ild in abs_ILD_arr:
        min_rt_cut = get_min_rt_cut(abs_ild)
        q_60 = animal_data.get((60, abs_ild), {}).get('empirical', {}).get('fitting_quantiles', np.full_like(quantile_levels, np.nan))
        for abl in [20, 40]:
            q_other = animal_data.get((abl, abs_ild), {}).get('empirical', {}).get('fitting_quantiles', np.full_like(quantile_levels, np.nan))
            slope = np.nan
            if not np.any(np.isnan(q_60)) and not np.any(np.isnan(q_other)):
                mask = (q_60 >= min_rt_cut) & (q_60 <= max_RT_cut)
                if np.sum(mask) >= 2:
                    x_fit_calc = q_60[mask] - min_rt_cut
                    y_fit_calc = (q_other - q_60)[mask]
                    y_intercept = y_fit_calc[0]
                    y_fit_calc_shifted = y_fit_calc - y_intercept
                    if np.sum(x_fit_calc**2) > 0:
                        slope = np.sum(x_fit_calc * y_fit_calc_shifted) / np.sum(x_fit_calc**2)
            slopes_by_animal[animal_id][abs_ild][abl] = slope

# Average slopes for QQ summary lines
fit_results = {abs_ild: {} for abs_ild in abs_ILD_arr}
for abs_ild in abs_ILD_arr:
    for abl in [20, 40]:
        slope_vals = [slopes_by_animal[animal_id][abs_ild][abl] for animal_id in slopes_by_animal]
        slope_vals = np.array(slope_vals, dtype=float)
        slope_vals = slope_vals[~np.isnan(slope_vals)]
        fit_results[abs_ild][abl] = {
            'slope': (np.nanmean(slope_vals) if len(slope_vals) else np.nan),
            'intercept': 0.0
        }

# Print scaling-factor table for cross-script comparison
scaling_factor_rows = []
for animal_id in sorted(slopes_by_animal.keys()):
    for abs_ild in abs_ILD_arr:
        slope_20 = slopes_by_animal[animal_id][abs_ild][20]
        slope_40 = slopes_by_animal[animal_id][abs_ild][40]
        scaling_factor_rows.append({
            "animal": str(animal_id),
            "abs_ILD": abs_ild,
            "slope_20": slope_20,
            "sf_20": (1.0 + slope_20) if not np.isnan(slope_20) else np.nan,
            "slope_40": slope_40,
            "sf_40": (1.0 + slope_40) if not np.isnan(slope_40) else np.nan,
        })

if scaling_factor_rows:
    sf_df = pd.DataFrame(scaling_factor_rows).sort_values(["animal", "abs_ILD"]).reset_index(drop=True)
    print("\n=== RTD Q-Scale-Match Script Scaling Factors (per animal, per |ILD|) ===")
    print(
        sf_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

for abs_ild in abs_ILD_arr:
    q_60 = quantiles_by_abl_ild[60][abs_ild]
    q_20 = quantiles_by_abl_ild[20][abs_ild]
    q_40 = quantiles_by_abl_ild[40][abs_ild]
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))

    min_RT_cut = get_min_rt_cut(abs_ild)
    min_x_cut = min_RT_cut

    mask_20 = (q_60 >= min_RT_cut) & (q_60 <= max_RT_cut)
    mask_40 = (q_60 >= min_RT_cut) & (q_60 <= max_RT_cut)

    if not np.any(np.isnan(q_60)) and not np.any(np.isnan(q_20)) and np.sum(mask_20) >= 2:
        x = q_60[mask_20] - min_x_cut
        y = (q_20 - q_60)[mask_20]
        y_intercept = y[0]
        y = y - y_intercept
        slope_20 = fit_results[abs_ild][20]['slope']
        x_fit = [min_x_cut, np.nanmax(q_60[mask_20])]
        y_fit = [0, slope_20 * (x_fit[1] - min_x_cut)] if not np.isnan(slope_20) else [np.nan, np.nan]
        ax.plot(x_fit, y_fit, color=abl_colors[20], linestyle='-', label=f'Fit 20 (m={slope_20:.2f})')
    else:
        slope_20 = np.nan

    # ABL 40 vs 60
    if not np.any(np.isnan(q_60)) and not np.any(np.isnan(q_40)) and np.sum(mask_40) >= 2:
        x = q_60[mask_40] - min_x_cut
        y = (q_40 - q_60)[mask_40]
        y_intercept = y[0]
        y = y - y_intercept
        slope_40 = fit_results[abs_ild][40]['slope']
        x_fit = [min_x_cut, np.nanmax(q_60[mask_40])]
        y_fit = [0, slope_40 * (x_fit[1] - min_x_cut)] if not np.isnan(slope_40) else [np.nan, np.nan]
        ax.plot(x_fit, y_fit, color=abl_colors[40], linestyle='--', label=f'Fit 40 (m={slope_40:.2f})')
    else:
        slope_40 = np.nan

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
    min_x_cut = get_min_rt_cut(abs_ild)
    ax = axes_rescaled[j]
    for abl in ABL_arr:
        per_animal_curves = []
        stim_key = (abl, abs_ild)
        for animal_id, animal_data in rtd_data.items():
            if stim_key not in animal_data:
                continue
            emp_data = animal_data[stim_key]['empirical']
            if np.all(np.isnan(emp_data['rtd_hist'])):
                continue
            bin_centers = emp_data['bin_centers']
            rtd_hist = emp_data['rtd_hist']

            if abl == 60:
                curve_on_common_grid = np.interp(BIN_CENTERS, bin_centers, rtd_hist, left=np.nan, right=np.nan)
                per_animal_curves.append(curve_on_common_grid)
            else:
                slope = slopes_by_animal.get(animal_id, {}).get(abs_ild, {}).get(abl, np.nan)
                if np.isnan(slope) or (1 + slope) == 0:
                    continue
                bin_centers_mask = bin_centers > min_x_cut
                xvals = ((bin_centers - min_x_cut) / (1 + slope)) + min_x_cut
                multiplier = np.ones_like(rtd_hist)
                multiplier[bin_centers_mask] = slope + 1
                rescaled_rtd = rtd_hist * multiplier
                curve_on_common_grid = np.interp(BIN_CENTERS, xvals, rescaled_rtd, left=np.nan, right=np.nan)
                per_animal_curves.append(curve_on_common_grid)

        if per_animal_curves:
            avg_curve = np.nanmean(np.array(per_animal_curves), axis=0)
            ax.plot(BIN_CENTERS, avg_curve, color=abl_colors[abl], linewidth=1.5, label=f'ABL={abl}')
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

plt.savefig('asd_wt_rescaled_rtds_q_scale_match.png')
plt.savefig('asd_wt_rescaled_rtds_q_scale_match.pdf')
# fig_rescaled.suptitle('RTD rescaled', fontsize=14)
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# %%
# Plot 1: slope summary (mean ± SD) across animals for each |ILD|
slope_summary_file = "asd_wt_slope_summary_q_scale_match.png"
fig_slope, ax_slope = plt.subplots(figsize=(6, 4))
abl_plot_style = {20: ('tab:blue', 'o-'), 40: ('tab:orange', 's--')}

for abl in [20, 40]:
    means, sds = [], []
    for abs_ild in abs_ILD_arr:
        vals = np.array(
            [slopes_by_animal[animal_id][abs_ild][abl] for animal_id in slopes_by_animal],
            dtype=float,
        )
        vals = vals[~np.isnan(vals)]
        means.append(np.nanmean(vals) if len(vals) else np.nan)
        sds.append(np.nanstd(vals) if len(vals) else np.nan)
    color, line_style = abl_plot_style[abl]
    ax_slope.errorbar(
        abs_ILD_arr,
        means,
        yerr=sds,
        fmt=line_style,
        color=color,
        linewidth=1.5,
        capsize=3,
        label=f"ABL {abl}",
    )

ax_slope.set_xscale("log")
ax_slope.set_xticks(abs_ILD_arr)
ax_slope.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax_slope.set_xlabel("|ILD|")
ax_slope.set_ylabel("Slope (mean ± SD)")
ax_slope.set_title("Per-ILD slope summary")
ax_slope.grid(True, alpha=0.3)
ax_slope.legend(frameon=False)
plt.tight_layout()
fig_slope.savefig(slope_summary_file, dpi=300, bbox_inches='tight')
plt.close(fig_slope)

# Plot 2: |ILD|=16 raw vs scaled mean quantiles across ABLs
ild16_quant_file = "asd_wt_ild16_quantiles_q_scale_match.png"
ild16_raw_scaled_file = "asd_wt_ild16_quantiles_raw_vs_scaled_q_scale_match.png"
ild_check = 16
q_check_levels = np.arange(0.1, 1.0, 0.05)
idx = [np.argmin(np.abs(quantile_levels - q)) for q in q_check_levels]
min_cut_ild16 = get_min_rt_cut(ild_check)

# Keep the original raw-only plot output
fig_q16, ax_q16 = plt.subplots(figsize=(6, 4))
for abl in ABL_arr:
    qvals = quantiles_by_abl_ild[abl][ild_check][idx]
    ax_q16.plot(q_check_levels, qvals, marker='o', linewidth=1.5, label=f"ABL {abl}")
ax_q16.set_xlabel("Quantile level")
ax_q16.set_ylabel("RT (s)")
ax_q16.set_title("|ILD|=16 mean quantiles across ABL (raw)")
ax_q16.grid(True, alpha=0.3)
ax_q16.legend(frameon=False)
plt.tight_layout()
fig_q16.savefig(ild16_quant_file, dpi=300, bbox_inches='tight')
plt.close(fig_q16)

# Build scaled mean quantiles per ABL using per-animal slopes (same method as quantile script)
scaled_quantiles_by_abl_ild16 = {}
for abl in ABL_arr:
    stim_key = (abl, ild_check)
    per_animal_q = []
    for animal_id, animal_data in rtd_data.items():
        if stim_key not in animal_data:
            continue
        q_raw = animal_data[stim_key]['empirical']['fitting_quantiles']
        if np.all(np.isnan(q_raw)):
            continue
        if abl == 60:
            q_scaled = q_raw.copy()
        else:
            slope = slopes_by_animal.get(animal_id, {}).get(ild_check, {}).get(abl, np.nan)
            if np.isnan(slope) or (1 + slope) == 0:
                q_scaled = q_raw.copy()
            else:
                q_scaled = np.where(
                    q_raw > min_cut_ild16,
                    ((q_raw - min_cut_ild16) / (1 + slope)) + min_cut_ild16,
                    q_raw,
                )
        per_animal_q.append(q_scaled)
    if per_animal_q:
        scaled_quantiles_by_abl_ild16[abl] = np.nanmean(np.array(per_animal_q), axis=0)
    else:
        scaled_quantiles_by_abl_ild16[abl] = np.full_like(quantile_levels, np.nan)

# Raw vs scaled side-by-side comparison
fig_cmp, axes_cmp = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

for abl in ABL_arr:
    raw_vals = quantiles_by_abl_ild[abl][ild_check][idx]
    axes_cmp[0].plot(q_check_levels, raw_vals, marker='o', linewidth=1.5, label=f"ABL {abl}")
axes_cmp[0].set_title("|ILD|=16 raw quantiles")
axes_cmp[0].set_xlabel("Quantile level")
axes_cmp[0].set_ylabel("RT (s)")
axes_cmp[0].grid(True, alpha=0.3)
axes_cmp[0].legend(frameon=False)

for abl in ABL_arr:
    scaled_vals = scaled_quantiles_by_abl_ild16[abl][idx]
    axes_cmp[1].plot(q_check_levels, scaled_vals, marker='o', linewidth=1.5, label=f"ABL {abl}")
axes_cmp[1].set_title("|ILD|=16 scaled quantiles")
axes_cmp[1].set_xlabel("Quantile level")
axes_cmp[1].grid(True, alpha=0.3)
axes_cmp[1].legend(frameon=False)

plt.tight_layout()
fig_cmp.savefig(ild16_raw_scaled_file, dpi=300, bbox_inches='tight')
# plt.close(fig_cmp)

# %%
# CDF comparison at |ILD|=16: raw vs scaled (side by side)
ild16_cdf_file = "asd_wt_ild16_cdf_raw_vs_scaled_q_scale_match.png"
bin_width = rt_bins[1] - rt_bins[0]

# Build mean raw and mean scaled RTD curves for |ILD|=16 at each ABL
raw_rtd_ild16 = {}
scaled_rtd_ild16 = {}

for abl in ABL_arr:
    stim_key = (abl, ild_check)
    raw_curves = []
    scaled_curves = []

    for animal_id, animal_data in rtd_data.items():
        if stim_key not in animal_data:
            continue
        emp_data = animal_data[stim_key]['empirical']
        rtd_hist = emp_data['rtd_hist']
        x_raw = emp_data['bin_centers']
        if np.all(np.isnan(rtd_hist)):
            continue

        # Raw curve on common grid
        raw_curve = np.interp(BIN_CENTERS, x_raw, rtd_hist, left=np.nan, right=np.nan)
        raw_curves.append(raw_curve)

        # Scaled curve on common grid
        if abl == 60:
            scaled_curve = raw_curve.copy()
        else:
            slope = slopes_by_animal.get(animal_id, {}).get(ild_check, {}).get(abl, np.nan)
            if np.isnan(slope) or (1 + slope) == 0:
                continue
            mask = x_raw > min_cut_ild16
            x_scaled = ((x_raw - min_cut_ild16) / (1 + slope)) + min_cut_ild16
            multiplier = np.ones_like(rtd_hist)
            multiplier[mask] = slope + 1
            rtd_scaled = rtd_hist * multiplier
            scaled_curve = np.interp(BIN_CENTERS, x_scaled, rtd_scaled, left=np.nan, right=np.nan)
        scaled_curves.append(scaled_curve)

    if raw_curves:
        raw_rtd_ild16[abl] = np.nanmean(np.array(raw_curves), axis=0)
    else:
        raw_rtd_ild16[abl] = np.full_like(BIN_CENTERS, np.nan)

    if scaled_curves:
        scaled_rtd_ild16[abl] = np.nanmean(np.array(scaled_curves), axis=0)
    else:
        scaled_rtd_ild16[abl] = np.full_like(BIN_CENTERS, np.nan)

# Convert to CDFs
raw_cdf_ild16 = {}
scaled_cdf_ild16 = {}
for abl in ABL_arr:
    raw_cdf = np.nancumsum(np.nan_to_num(raw_rtd_ild16[abl], nan=0.0)) * bin_width
    if raw_cdf[-1] > 0:
        raw_cdf = raw_cdf / raw_cdf[-1]
    raw_cdf_ild16[abl] = raw_cdf

    scaled_cdf = np.nancumsum(np.nan_to_num(scaled_rtd_ild16[abl], nan=0.0)) * bin_width
    if scaled_cdf[-1] > 0:
        scaled_cdf = scaled_cdf / scaled_cdf[-1]
    scaled_cdf_ild16[abl] = scaled_cdf

# Plot side-by-side CDFs
fig_cdf, axes_cdf = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

for abl in ABL_arr:
    axes_cdf[0].plot(BIN_CENTERS, raw_cdf_ild16[abl], linewidth=1.5, label=f"ABL {abl}")
axes_cdf[0].set_title("|ILD|=16 raw CDF")
axes_cdf[0].set_xlabel("RT (s)")
axes_cdf[0].set_ylabel("CDF")
axes_cdf[0].set_xlim(0, 0.2)
axes_cdf[0].set_ylim(0, 0.8)
axes_cdf[0].grid(True, alpha=0.3)
axes_cdf[0].legend(frameon=False)

for abl in ABL_arr:
    axes_cdf[1].plot(BIN_CENTERS, scaled_cdf_ild16[abl], linewidth=1.5, label=f"ABL {abl}")
axes_cdf[1].set_title("|ILD|=16 scaled CDF")
axes_cdf[1].set_xlabel("RT (s)")
axes_cdf[1].set_xlim(0, 0.2)
axes_cdf[1].set_ylim(0, 0.8)
axes_cdf[1].grid(True, alpha=0.3)
axes_cdf[1].legend(frameon=False)

plt.tight_layout()
fig_cdf.savefig(ild16_cdf_file, dpi=300, bbox_inches='tight')
# plt.close(fig_cdf)

# %%
