# %%
"""
Plot R² vs ILD for cond-fit and vanilla models.
1. R² per ABL vs ILD (colored by ABL)
2. R² averaged across ABL vs ILD
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt

# %%
# =============================================================================
# Load R² from all ILD pkl files
# =============================================================================
ILD_VALUES = [1, 2, 4, 8, 16]
ABL_arr = [20, 40, 60]

# Colors for ABL
abl_colors = {20: 'tab:blue', 40: 'tab:orange', 60: 'tab:green'}

# Storage
R2_cond_per_ABL = {abl: [] for abl in ABL_arr}
R2_vanilla_per_ABL = {abl: [] for abl in ABL_arr}
R2_cond_mean = []
R2_vanilla_mean = []

for ild in ILD_VALUES:
    pkl_file = f'quantiles_gof_ILD_{ild}.pkl'
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    # Per ABL R²
    for abl in ABL_arr:
        R2_cond_per_ABL[abl].append(data['R2_per_ABL']['cond'].get(abl, np.nan))
        R2_vanilla_per_ABL[abl].append(data['R2_per_ABL']['vanilla'].get(abl, np.nan))
    
    # Mean R²
    R2_cond_mean.append(data['mean_R2']['cond'])
    R2_vanilla_mean.append(data['mean_R2']['vanilla'])

print("Loaded R² from all ILD pkl files")

# %%
# =============================================================================
# Plot 1: R² per ABL vs ILD
# =============================================================================
fig, ax = plt.subplots(figsize=(6, 5))

for abl in ABL_arr:
    color = abl_colors[abl]
    # Cond fit - dot marker
    ax.plot(ILD_VALUES, R2_cond_per_ABL[abl], 'o', color=color, 
            markersize=10, label=f'ABL {abl} Cond')
    # Vanilla - cross marker
    ax.plot(ILD_VALUES, R2_vanilla_per_ABL[abl], 'x', color=color, 
            markersize=10, markeredgewidth=2, label=f'ABL {abl} Vanilla')

ax.set_xlabel('|ILD| (dB)', fontsize=14)
ax.set_ylabel('R²', fontsize=14)
ax.set_title('R² per ABL vs ILD', fontsize=14)
ax.set_xscale('log', base=2)
ax.set_xticks(ILD_VALUES)
ax.set_xticklabels(ILD_VALUES)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=12)

# Custom legend: ABL colors + marker meaning
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:blue', markersize=10, label='ABL 20'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:orange', markersize=10, label='ABL 40'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:green', markersize=10, label='ABL 60'),
    Line2D([0], [0], marker='o', color='gray', linestyle='None', markersize=10, label='Cond fit'),
    Line2D([0], [0], marker='x', color='gray', linestyle='None', markersize=10, markeredgewidth=2, label='Vanilla'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig('R2_per_ABL_vs_ILD.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: R2_per_ABL_vs_ILD.png")

# %%
# =============================================================================
# Plot 2: R² averaged across ABL vs ILD
# =============================================================================
fig, ax = plt.subplots(figsize=(6, 5))

# Cond fit - dot
ax.plot(ILD_VALUES, R2_cond_mean, 'o', color='black', markersize=12, label='Cond fit')
# Vanilla - cross
ax.plot(ILD_VALUES, R2_vanilla_mean, 'x', color='black', markersize=12, markeredgewidth=2, label='Vanilla')

ax.set_xlabel('|ILD| (dB)', fontsize=14)
ax.set_ylabel('R² (mean across ABL)', fontsize=14)
ax.set_title('R² averaged across ABL vs ILD', fontsize=14)
ax.set_xscale('log', base=2)
ax.set_xticks(ILD_VALUES)
ax.set_xticklabels(ILD_VALUES)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend(loc='lower right', fontsize=12)

plt.tight_layout()
plt.savefig('R2_mean_vs_ILD.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: R2_mean_vs_ILD.png")

# %%
# =============================================================================
# Plot 3: R² color-coded on stimulus set (Left SL vs Right SL)
# =============================================================================
# Convert (ABL, ILD) to (Left SL, Right SL)
# ABL = (Left SL + Right SL) / 2
# ILD = Right SL - Left SL
# => Right SL = ABL + ILD/2, Left SL = ABL - ILD/2

# We need both positive and negative ILDs for the full stimulus set
all_ILDs = [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]

# Build arrays for plotting
right_sl_all = []
left_sl_all = []
r2_vanilla_all = []
r2_cond_all = []

for abl in ABL_arr:
    for ild in all_ILDs:
        right_sl = abl + ild / 2
        left_sl = abl - ild / 2
        right_sl_all.append(right_sl)
        left_sl_all.append(left_sl)
        
        # Get R² for |ILD| (since we only have data for positive ILDs)
        abs_ild = abs(ild)
        ild_idx = ILD_VALUES.index(abs_ild)
        r2_vanilla_all.append(R2_vanilla_per_ABL[abl][ild_idx])
        r2_cond_all.append(R2_cond_per_ABL[abl][ild_idx])

right_sl_all = np.array(right_sl_all)
left_sl_all = np.array(left_sl_all)
r2_vanilla_all = np.array(r2_vanilla_all)
r2_cond_all = np.array(r2_cond_all)

# Find common color range for R²
# vmin = min(r2_vanilla_all.min(), r2_cond_all.min())
vmin = 0.75
vmax = max(r2_vanilla_all.max(), r2_cond_all.max())

# Percentage increase: 100 * (cond - vanilla) / vanilla
r2_pct_increase = 100 * (r2_cond_all - r2_vanilla_all) / r2_vanilla_all

# Colormap options:
# Sequential: 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'hot', 'cool', 'YlOrRd', 'Blues'
# Diverging: 'RdYlGn', 'RdBu', 'coolwarm', 'seismic', 'PiYG', 'PRGn', 'BrBG'
cmap_type = 'Blues'
CMAP_R2 = cmap_type        # for R² panels
CMAP_PCT = cmap_type        # for % increase panel

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Left panel: Vanilla R²
ax = axes[0]
sc1 = ax.scatter(right_sl_all, left_sl_all, c=r2_vanilla_all, cmap=CMAP_R2, 
                  s=100, vmin=vmin, vmax=vmax, edgecolors='k', linewidths=0.5)
ax.plot([10, 70], [10, 70], 'k--', alpha=0.5, linewidth=1)  # diagonal ABL line
ax.set_xlabel('Right SL (dB SPL)', fontsize=12)
ax.set_ylabel('Left SL (dB SPL)', fontsize=12)
ax.set_title(f'Vanilla R² ({CMAP_R2})', fontsize=14)
ax.set_xlim(10, 70)
ax.set_ylim(10, 70)
ax.set_aspect('equal')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Middle panel: Cond-fit R²
ax = axes[1]
sc2 = ax.scatter(right_sl_all, left_sl_all, c=r2_cond_all, cmap=CMAP_R2, 
                  s=100, vmin=vmin, vmax=vmax, edgecolors='k', linewidths=0.5)
ax.plot([10, 70], [10, 70], 'k--', alpha=0.5, linewidth=1)  # diagonal ABL line
ax.set_xlabel('Right SL (dB SPL)', fontsize=12)
ax.set_ylabel('Left SL (dB SPL)', fontsize=12)
ax.set_title(f'Cond-fit R² ({CMAP_R2})', fontsize=14)
ax.set_xlim(10, 70)
ax.set_ylim(10, 70)
ax.set_aspect('equal')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Shared colorbar for R² panels
cbar1 = fig.colorbar(sc2, ax=axes[:2], shrink=0.8, pad=0.02)
cbar1.set_label('R²', fontsize=12)

# Right panel: Percentage increase
ax = axes[2]
sc3 = ax.scatter(right_sl_all, left_sl_all, c=r2_pct_increase, cmap=CMAP_PCT, 
                  s=100, edgecolors='k', linewidths=0.5)
ax.plot([10, 70], [10, 70], 'k--', alpha=0.5, linewidth=1)  # diagonal ABL line
ax.set_xlabel('Right SL (dB SPL)', fontsize=12)
ax.set_ylabel('Left SL (dB SPL)', fontsize=12)
ax.set_title(f'% increase ({CMAP_PCT})', fontsize=14)
ax.set_xlim(10, 70)
ax.set_ylim(10, 70)
ax.set_aspect('equal')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Colorbar for percentage increase
cbar2 = fig.colorbar(sc3, ax=axes[2], shrink=0.8)
cbar2.set_label('% increase', fontsize=12)

plt.savefig('R2_stimulus_set_colormap.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: R2_stimulus_set_colormap.png")

# %%
# =============================================================================
# Print summary table
# =============================================================================
print("\n" + "="*60)
print("R² Summary Table")
print("="*60)
print(f"\n{'ILD':<6} {'Cond (mean)':<14} {'Vanilla (mean)':<14}")
print("-"*34)
for i, ild in enumerate(ILD_VALUES):
    print(f"{ild:<6} {R2_cond_mean[i]:<14.4f} {R2_vanilla_mean[i]:<14.4f}")

print("\n" + "="*60)
print("R² per ABL")
print("="*60)
print(f"\n{'ILD':<6} {'ABL 20 C':<10} {'ABL 20 V':<10} {'ABL 40 C':<10} {'ABL 40 V':<10} {'ABL 60 C':<10} {'ABL 60 V':<10}")
print("-"*66)
for i, ild in enumerate(ILD_VALUES):
    print(f"{ild:<6} {R2_cond_per_ABL[20][i]:<10.4f} {R2_vanilla_per_ABL[20][i]:<10.4f} "
          f"{R2_cond_per_ABL[40][i]:<10.4f} {R2_vanilla_per_ABL[40][i]:<10.4f} "
          f"{R2_cond_per_ABL[60][i]:<10.4f} {R2_vanilla_per_ABL[60][i]:<10.4f}")

# %%
# TODO