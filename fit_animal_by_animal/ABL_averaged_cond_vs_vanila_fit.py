# %%
"""
Plot ABL-averaged quantiles vs |ILD| comparing cond-fit and vanilla models.

For each |ILD| (1, 2, 4, 8, 16):
- Average quantiles across ABLs (20, 40, 60)
- Plot 5 representative quantile levels
- Compare: Data (markers), Cond fit (dotted), Vanilla (solid)

Uses vanilla_quant_fig2_data.pkl for vanilla model data (per-animal values).
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from collections import defaultdict

# Helper functions for unpickling the quantile data (must match original definitions)
def _create_innermost_dict():
    return {'empirical': [], 'theoretical': []}

def _create_inner_defaultdict():
    return defaultdict(_create_innermost_dict)

# %%
# =============================================================================
# Configuration
# =============================================================================
ILD_VALUES = [1, 2, 4, 8, 16]
ABL_arr = [20, 40, 60]

# %%
# =============================================================================
# Load vanilla model data (has per-animal empirical and theoretical)
# =============================================================================
with open('vanilla_quant_fig2_data.pkl', 'rb') as f:
    vanilla_data = pickle.load(f)

plot_data = vanilla_data['plot_data']  # empirical data
continuous_plot_data = vanilla_data['continuous_plot_data']  # vanilla theoretical
QUANTILES_TO_PLOT = vanilla_data['QUANTILES_TO_PLOT']
abs_ild_sorted = vanilla_data['abs_ild_sorted']

print(f"Loaded vanilla_quant_fig2_data.pkl")
print(f"ILDs: {abs_ild_sorted}")
print(f"Quantiles: {QUANTILES_TO_PLOT}")

# %%
# =============================================================================
# Compute ABL-averaged quantiles for each ILD (aggregate per-animal values)
# =============================================================================
n_quantiles = len(QUANTILES_TO_PLOT)

# Storage: {ILD: (means array, sems array)} for each quantile
data_avg = {}
data_sem = {}
vanilla_avg = {}
vanilla_sem = {}

for abs_ild in abs_ild_sorted:
    # Collect all per-animal values across all ABLs
    all_emp_quantiles = []  # Will be (n_animals_total, n_quantiles)
    all_theo_quantiles = []
    
    for abl in ABL_arr:
        # Empirical data from plot_data
        if abs_ild in plot_data[abl]:
            emp_list = plot_data[abl][abs_ild]['empirical']
            if len(emp_list) > 0:
                all_emp_quantiles.extend(emp_list)
        
        # Vanilla theoretical from continuous_plot_data
        if abs_ild in continuous_plot_data[abl]:
            theo_list = continuous_plot_data[abl][abs_ild]['theoretical']
            if len(theo_list) > 0:
                all_theo_quantiles.extend(theo_list)
    
    # Convert to arrays and compute mean/SEM across animals
    if len(all_emp_quantiles) > 0:
        emp_arr = np.array(all_emp_quantiles)  # (n_animals, n_quantiles)
        data_avg[abs_ild] = np.nanmean(emp_arr, axis=0)
        data_sem[abs_ild] = sem(emp_arr, axis=0, nan_policy='omit')
    
    if len(all_theo_quantiles) > 0:
        theo_arr = np.array(all_theo_quantiles)
        vanilla_avg[abs_ild] = np.nanmean(theo_arr, axis=0)
        vanilla_sem[abs_ild] = sem(theo_arr, axis=0, nan_policy='omit')

print(f"\nUsing {n_quantiles} quantile levels: {QUANTILES_TO_PLOT}")

# %%
# =============================================================================
# Load cond-fit data from quantiles_gof_ILD_*.pkl files
# =============================================================================
cond_avg = {}
cond_sem = {}

for ILD in ILD_VALUES:
    pkl_file = f'quantiles_gof_ILD_{ILD}.pkl'
    try:
        with open(pkl_file, 'rb') as f:
            cond_data = pickle.load(f)
        
        theory_q_levels = cond_data['quantile_levels']
        emp_q_levels = cond_data['empirical']['plotting_quantiles']
        
        # Collect cond-fit across ABLs
        cond_means_by_abl = []
        cond_sems_by_abl = []
        
        for ABL in ABL_arr:
            if ABL in cond_data['theory']['cond']['mean_quantiles']:
                theory_vals = cond_data['theory']['cond']['mean_quantiles'][ABL]
                theory_sem_vals = cond_data['theory']['cond']['sem_quantiles'][ABL]
                # Interpolate to match empirical quantile levels
                interp_vals = np.interp(emp_q_levels, theory_q_levels, theory_vals)
                interp_sem = np.interp(emp_q_levels, theory_q_levels, theory_sem_vals)
                cond_means_by_abl.append(interp_vals)
                cond_sems_by_abl.append(interp_sem)
        
        if cond_means_by_abl:
            cond_avg[ILD] = np.mean(cond_means_by_abl, axis=0)
            cond_sem[ILD] = np.mean(cond_sems_by_abl, axis=0)
            
    except FileNotFoundError:
        print(f"WARNING: {pkl_file} not found for cond-fit")

# %%
# =============================================================================
# Plot: Quantiles vs |ILD|
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 6))

x_positions = np.arange(len(ILD_VALUES))
x_labels = [str(ild) for ild in ILD_VALUES]

for q_idx, q in enumerate(QUANTILES_TO_PLOT):
    # Extract values and SEMs for this quantile across ILDs
    # Note: vanilla data uses float keys, cond uses int keys
    data_vals = [data_avg[float(ILD)][q_idx] if float(ILD) in data_avg else np.nan for ILD in ILD_VALUES]
    data_errs = [data_sem[float(ILD)][q_idx] if float(ILD) in data_sem else np.nan for ILD in ILD_VALUES]
    cond_vals = [cond_avg[ILD][q_idx] if ILD in cond_avg else np.nan for ILD in ILD_VALUES]
    cond_errs = [cond_sem[ILD][q_idx] if ILD in cond_sem else np.nan for ILD in ILD_VALUES]
    vanilla_vals = [vanilla_avg[float(ILD)][q_idx] if float(ILD) in vanilla_avg else np.nan for ILD in ILD_VALUES]
    vanilla_errs = [vanilla_sem[float(ILD)][q_idx] if float(ILD) in vanilla_sem else np.nan for ILD in ILD_VALUES]
    
    # Data points (black markers with error bars)
    ax.errorbar(x_positions, data_vals, yerr=data_errs, color='black', fmt='o', 
                markersize=8, capsize=0, zorder=3, label='Data' if q_idx == 0 else None)
    
    # Cond fit (red dotted line, no error bars)
    ax.plot(x_positions, cond_vals, color='red', linewidth=2, linestyle=':', zorder=2,
            label='Cond fit' if q_idx == 0 else None)
    
    # Vanilla fit (red solid line, no error bars)
    ax.plot(x_positions, vanilla_vals, color='red', linewidth=2, linestyle='-', zorder=2,
            label='Vanilla' if q_idx == 0 else None)

# Formatting
ax.set_xlabel('|ILD| (dB)', fontsize=14)
ax.set_ylabel('RT (s)', fontsize=14)
ax.set_title('ABL-Averaged Quantiles: Cond Fit vs Vanilla vs Data', fontsize=12)
ax.set_xticks(x_positions)
ax.set_xticklabels(x_labels)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Simple legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='black', marker='o', linestyle='none', markersize=8, label='Data'),
    Line2D([0], [0], color='red', linestyle='-', linewidth=2, label='Vanilla'),
    Line2D([0], [0], color='red', linestyle=':', linewidth=2, label='Cond fit'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('ABL_averaged_quantiles_vs_ILD.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: ABL_averaged_quantiles_vs_ILD.png")

# %%
# =============================================================================
# Print summary table
# =============================================================================
print("\n" + "="*70)
print("ABL-Averaged Quantiles by |ILD| (50th percentile / median)")
print("="*70)
median_idx = 2  # Middle of 5 quantiles (index 2 = 50th percentile)
print(f"{'|ILD|':<8} {'Data':<12} {'Cond':<12} {'Vanilla':<12}")
print("-"*44)
for ILD in ILD_VALUES:
    d_val = data_avg.get(ILD, [np.nan]*5)[median_idx]
    c_val = cond_avg.get(ILD, [np.nan]*5)[median_idx]
    v_val = vanilla_avg.get(ILD, [np.nan]*5)[median_idx]
    print(f"{ILD:<8} {d_val:<12.4f} {c_val:<12.4f} {v_val:<12.4f}")

# %%
