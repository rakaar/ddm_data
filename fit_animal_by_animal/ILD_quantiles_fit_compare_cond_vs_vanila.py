# %%
"""
Compare cond-fit vs vanilla quantiles for any ILD.
Uses quantiles_gof_ILD_{ILD}.pkl files.
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt

# %%
# =============================================================================
# Configuration - CHANGE THIS
# =============================================================================
ILD_TARGET = 1  # Change to 1, 2, 4, 8, or 16

# %%
# =============================================================================
# Load data
# =============================================================================
pkl_file = f'quantiles_gof_ILD_{ILD_TARGET}.pkl'
with open(pkl_file, 'rb') as f:
    data = pickle.load(f)

ABL_arr = data['ABL_arr']
quantile_levels = data['quantile_levels']

# Theory data
cond_mean = data['theory']['cond']['mean_quantiles']
cond_sem = data['theory']['cond']['sem_quantiles']
vanilla_mean = data['theory']['vanilla']['mean_quantiles']
vanilla_sem = data['theory']['vanilla']['sem_quantiles']

# Empirical data
plotting_quantiles = data['empirical']['plotting_quantiles']
mean_unscaled = data['empirical']['mean_unscaled']
sem_unscaled = data['empirical']['sem_unscaled']

# Colors
abl_colors = {20: 'blue', 40: 'green', 60: 'red'}

print(f"Loaded {pkl_file}")
print(f"ABLs: {ABL_arr}")
print(f"R² (cond): {data['mean_R2']['cond']:.4f}")
print(f"R² (vanilla): {data['mean_R2']['vanilla']:.4f}")

# %%
# =============================================================================
# Plot: Compare cond fit vs vanilla fit
# =============================================================================
fig, ax = plt.subplots(figsize=(7, 6))

for i, ABL in enumerate(ABL_arr):
    color = abl_colors[ABL]
    
    # Condition fit - solid line
    if ABL in cond_mean:
        ax.plot(quantile_levels, cond_mean[ABL], color=color, 
                linewidth=2, linestyle='-', label='Cond fit' if i == 0 else None)
    
    # Vanilla fit - dotted line
    if ABL in vanilla_mean:
        ax.plot(quantile_levels, vanilla_mean[ABL], color=color, 
                linewidth=2, linestyle=':', label='Vanilla' if i == 0 else None)

# Overlay empirical data points
for i, ABL in enumerate(ABL_arr):
    if ABL in mean_unscaled:
        ax.errorbar(
            plotting_quantiles,
            mean_unscaled[ABL],
            yerr=sem_unscaled[ABL],
            marker='o',
            linestyle='none',
            color=abl_colors[ABL],
            markersize=8,
            capsize=0,
            label='Data' if i == 0 else None
        )

ax.set_xlabel('Quantile', fontsize=14)
ax.set_ylabel('RT (s)', fontsize=14)
ax.set_title(f'Cond Fit (solid) vs Vanilla (dotted) at |ILD| = {ILD_TARGET} dB', fontsize=14)
ax.legend(loc='upper left', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_xlim(0.05, 0.95)
ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
ax.set_xticklabels(['10', '30', '50', '70', '90'])

plt.tight_layout()
out_file = f'ILD_{ILD_TARGET}_quantiles_cond_vs_vanilla_compare.png'
plt.savefig(out_file, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved: {out_file}")

# %%
# =============================================================================
# Print R² summary
# =============================================================================
print("\n" + "="*50)
print(f"Goodness of Fit (R²) for |ILD| = {ILD_TARGET} dB")
print("="*50)
print(f"\n{'ABL':<8} {'Cond R²':<12} {'Vanilla R²':<12}")
print("-"*32)
for ABL in ABL_arr:
    cond_r2 = data['R2_per_ABL']['cond'].get(ABL, np.nan)
    van_r2 = data['R2_per_ABL']['vanilla'].get(ABL, np.nan)
    print(f"{ABL:<8} {cond_r2:<12.4f} {van_r2:<12.4f}")
print("-"*32)
print(f"{'Mean':<8} {data['mean_R2']['cond']:<12.4f} {data['mean_R2']['vanilla']:<12.4f}")

# %%
