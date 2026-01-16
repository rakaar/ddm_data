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
