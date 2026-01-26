# %%
"""
Plot R² vs ILD for cond-fit, vanilla, and normalized models.
1. R² per ABL vs ILD (colored by ABL)
2. R² averaged across ABL vs ILD

Normalized R² is computed on-the-fly from norm_quant_fig2_data.pkl.
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Helper functions for unpickling the quantile data (must be defined before pickle.load)
def _create_innermost_dict():
    return {'empirical': [], 'theoretical': []}

def _create_inner_defaultdict():
    return defaultdict(_create_innermost_dict)

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
R2_norm_per_ABL = {abl: [] for abl in ABL_arr}
R2_cond_more_per_ABL = {abl: [] for abl in ABL_arr}  # 5-param cond fit
R2_cond_mean = []
R2_vanilla_mean = []
R2_norm_mean = []
R2_cond_more_mean = []  # 5-param cond fit

def compute_gof_metrics(theory_quantiles, theory_q_levels, emp_quantiles, emp_q_levels, emp_sem):
    """Compute goodness of fit metrics between theoretical and empirical quantiles."""
    theory_interp = np.interp(emp_q_levels, theory_q_levels, theory_quantiles)
    residuals = theory_interp - emp_quantiles
    sse = np.sum(residuals**2)
    weights = 1.0 / (emp_sem**2 + 1e-10)
    weighted_sse = np.sum(weights * residuals**2)
    rmse = np.sqrt(np.mean(residuals**2))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((emp_quantiles - np.mean(emp_quantiles))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    return {'SSE': sse, 'weighted_SSE': weighted_sse, 'RMSE': rmse, 'R2': r_squared}

try:
    with open('norm_quant_fig2_data.pkl', 'rb') as f:
        norm_data = pickle.load(f)
    norm_plot_data = norm_data.get('plot_data', {})
    norm_continuous_plot_data = norm_data.get('continuous_plot_data', {})
    norm_quantile_levels = np.array(norm_data['QUANTILES_TO_PLOT'])
    print("Loaded norm_quant_fig2_data.pkl for norm R²")
except (FileNotFoundError, AttributeError) as e:
    norm_data = None
    norm_plot_data = {}
    norm_continuous_plot_data = {}
    norm_quantile_levels = None
    print(f"WARNING: Could not load norm_quant_fig2_data.pkl ({e}). Norm R² will be NaN.")

def _get_norm_theoretical(plot_data, continuous_plot_data, abl, abs_ild):
    for key in (abs_ild, float(abs_ild)):
        if key in plot_data.get(abl, {}):
            theo = plot_data[abl][key].get('theoretical', [])
            if len(theo) > 0:
                return theo
        if key in continuous_plot_data.get(abl, {}):
            theo = continuous_plot_data[abl][key].get('theoretical', [])
            if len(theo) > 0:
                return theo
    return []

for ild in ILD_VALUES:
    pkl_file = f'quantiles_gof_ILD_{ild}.pkl'
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    abs_ild = abs(ild)
    norm_r2_values = []
    
    # Per ABL R²
    for abl in ABL_arr:
        R2_cond_per_ABL[abl].append(data['R2_per_ABL']['cond'].get(abl, np.nan))
        R2_vanilla_per_ABL[abl].append(data['R2_per_ABL']['vanilla'].get(abl, np.nan))
        r2_norm = np.nan
        if (norm_plot_data or norm_continuous_plot_data) and norm_quantile_levels is not None:
            norm_theo = _get_norm_theoretical(norm_plot_data, norm_continuous_plot_data, abl, abs_ild)
            if len(norm_theo) > 0:
                norm_theo_mean = np.nanmean(np.array(norm_theo), axis=0)
                emp_q = np.array(data['empirical']['mean_unscaled'][abl])
                emp_sem = np.array(data['empirical']['sem_unscaled'][abl])
                emp_q_levels = np.array(data['empirical']['plotting_quantiles'])
                gof = compute_gof_metrics(
                    norm_theo_mean, norm_quantile_levels, emp_q, emp_q_levels, emp_sem
                )
                r2_norm = gof['R2']
        R2_norm_per_ABL[abl].append(r2_norm)
        norm_r2_values.append(r2_norm)
    
    # Mean R²
    R2_cond_mean.append(data['mean_R2']['cond'])
    R2_vanilla_mean.append(data['mean_R2']['vanilla'])
    R2_norm_mean.append(np.nanmean(norm_r2_values) if norm_r2_values else np.nan)
    
    # Load 5-param cond fit (more params)
    more_params_file = f'quantiles_gof_ILD_{ild}_more_params.pkl'
    try:
        with open(more_params_file, 'rb') as f:
            data_more = pickle.load(f)
        for abl in ABL_arr:
            R2_cond_more_per_ABL[abl].append(data_more['R2_per_ABL']['cond_more'].get(abl, np.nan))
        R2_cond_more_mean.append(data_more['mean_R2']['cond_more'])
    except FileNotFoundError:
        for abl in ABL_arr:
            R2_cond_more_per_ABL[abl].append(np.nan)
        R2_cond_more_mean.append(np.nan)
        print(f"WARNING: {more_params_file} not found")

print("Loaded R² from all ILD pkl files (including 5-param)")

# %%
# =============================================================================
# Plot 1: R² per ABL vs ILD
# =============================================================================
fig, ax = plt.subplots(figsize=(6, 5))

for abl in ABL_arr:
    color = abl_colors[abl]
    # Cond fit (2-param) - dot marker
    ax.plot(ILD_VALUES, R2_cond_per_ABL[abl], 'o', color=color, 
            markersize=10, label=f'ABL {abl} Cond')
    # Cond fit (5-param) - star marker
    ax.plot(ILD_VALUES, R2_cond_more_per_ABL[abl], '*', color=color, 
            markersize=12, label=f'ABL {abl} Cond-5p')
    # Vanilla - cross marker
    ax.plot(ILD_VALUES, R2_vanilla_per_ABL[abl], 'x', color=color, 
            markersize=10, markeredgewidth=2, label=f'ABL {abl} Vanilla')
    # Norm - triangle marker
    ax.plot(ILD_VALUES, R2_norm_per_ABL[abl], '^', color=color,
            markersize=10, markeredgewidth=1.5, label=f'ABL {abl} Norm')

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
    Line2D([0], [0], marker='o', color='gray', linestyle='None', markersize=10, label='Cond (2p)'),
    Line2D([0], [0], marker='*', color='gray', linestyle='None', markersize=12, label='Cond (5p)'),
    Line2D([0], [0], marker='x', color='gray', linestyle='None', markersize=10, markeredgewidth=2, label='Vanilla'),
    Line2D([0], [0], marker='^', color='gray', linestyle='None', markersize=10, markeredgewidth=1.5, label='Norm'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig('R2_per_ABL_vs_ILD.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: R2_per_ABL_vs_ILD.png")

# %%
# =============================================================================
# Plot 2: R² averaged across ABL vs ILD
# =============================================================================
fig, ax = plt.subplots(figsize=(6, 5))

# Cond fit (2-param) - dot
ax.plot(ILD_VALUES, R2_cond_mean, 'o', color='black', markersize=12, label='Cond (2p)')
# Cond fit (5-param) - star
ax.plot(ILD_VALUES, R2_cond_more_mean, '*', color='black', markersize=14, label='Cond (5p)')
# Vanilla - cross
ax.plot(ILD_VALUES, R2_vanilla_mean, 'x', color='black', markersize=12, markeredgewidth=2, label='Vanilla')
# Norm - triangle
ax.plot(ILD_VALUES, R2_norm_mean, '^', color='black', markersize=12, markeredgewidth=1.5, label='Norm')

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
print("\n" + "="*70)
print("R² Summary Table")
print("="*70)
print(f"\n{'ILD':<6} {'Cond-2p':<12} {'Cond-5p':<12} {'Vanilla':<12} {'Norm':<12}")
print("-"*54)
for i, ild in enumerate(ILD_VALUES):
    print(f"{ild:<6} {R2_cond_mean[i]:<12.4f} {R2_cond_more_mean[i]:<12.4f} {R2_vanilla_mean[i]:<12.4f} {R2_norm_mean[i]:<12.4f}")

print("\n" + "="*60)
print("R² per ABL")
print("="*60)
print(f"\n{'ILD':<6} {'ABL 20 C':<10} {'ABL 20 V':<10} {'ABL 20 N':<10} {'ABL 40 C':<10} {'ABL 40 V':<10} {'ABL 40 N':<10} {'ABL 60 C':<10} {'ABL 60 V':<10} {'ABL 60 N':<10}")
print("-"*96)
for i, ild in enumerate(ILD_VALUES):
    print(f"{ild:<6} {R2_cond_per_ABL[20][i]:<10.4f} {R2_vanilla_per_ABL[20][i]:<10.4f} {R2_norm_per_ABL[20][i]:<10.4f} "
          f"{R2_cond_per_ABL[40][i]:<10.4f} {R2_vanilla_per_ABL[40][i]:<10.4f} {R2_norm_per_ABL[40][i]:<10.4f} "
          f"{R2_cond_per_ABL[60][i]:<10.4f} {R2_vanilla_per_ABL[60][i]:<10.4f} {R2_norm_per_ABL[60][i]:<10.4f}")

# %%
# =============================================================================
# Comparison Table: Cond (5p) vs Norm
# =============================================================================
print("\n" + "="*70)
print("Comparison: Cond (5p) vs Norm Model")
print("="*70)

# Per ILD (averaged across ABL)
print("\n--- Averaged across ABL ---")
print(f"{'ILD':<6} {'Cond-5p':<12} {'Norm':<12} {'Diff (5p-N)':<14} {'Winner':<10}")
print("-"*54)
for i, ild in enumerate(ILD_VALUES):
    diff = R2_cond_more_mean[i] - R2_norm_mean[i]
    winner = "Cond-5p" if diff > 0 else "Norm"
    print(f"{ild:<6} {R2_cond_more_mean[i]:<12.4f} {R2_norm_mean[i]:<12.4f} {diff:<+14.4f} {winner:<10}")

# Per ABL per ILD
print("\n--- Per ABL ---")
for abl in ABL_arr:
    print(f"\nABL {abl}:")
    print(f"  {'ILD':<6} {'Cond-5p':<12} {'Norm':<12} {'Diff (5p-N)':<14} {'Winner':<10}")
    print(f"  " + "-"*52)
    for i, ild in enumerate(ILD_VALUES):
        c5p = R2_cond_more_per_ABL[abl][i]
        norm = R2_norm_per_ABL[abl][i]
        diff = c5p - norm
        winner = "Cond-5p" if diff > 0 else "Norm"
        print(f"  {ild:<6} {c5p:<12.4f} {norm:<12.4f} {diff:<+14.4f} {winner:<10}")

# Overall summary
print("\n--- Overall Summary ---")
cond5p_wins = sum(1 for i in range(len(ILD_VALUES)) for abl in ABL_arr 
                  if R2_cond_more_per_ABL[abl][i] > R2_norm_per_ABL[abl][i])
norm_wins = len(ILD_VALUES) * len(ABL_arr) - cond5p_wins
print(f"Cond-5p wins: {cond5p_wins}/{len(ILD_VALUES)*len(ABL_arr)} conditions")
print(f"Norm wins:    {norm_wins}/{len(ILD_VALUES)*len(ABL_arr)} conditions")
print(f"Mean R² Cond-5p: {np.mean(R2_cond_more_mean):.4f}")
print(f"Mean R² Norm:    {np.mean(R2_norm_mean):.4f}")

# %%
# =============================================================================
# Plot for paper: R² vs ILD (Cond-5p, Norm, Vanilla)
# =============================================================================
fig, ax = plt.subplots(figsize=(5, 4))

# Plot mean R² across ABL for each model
ax.plot(ILD_VALUES, R2_cond_more_mean, 'o', color='black', markersize=8, 
        linewidth=1.5, label='Cond-5p')
ax.plot(ILD_VALUES, R2_norm_mean, 'o', color='tab:green', markersize=8, 
        linewidth=1.5, label='Norm')
ax.plot(ILD_VALUES, R2_vanilla_mean, 'o', color='tab:red', markersize=8, 
        linewidth=1.5, label='Vanilla')

ax.set_xlabel('|ILD| (dB)', fontsize=12)
ax.set_ylabel('R²', fontsize=12)
ax.set_xscale('log', base=2)
ax.set_xticks(ILD_VALUES)
ax.set_xticklabels(ILD_VALUES)
ax.set_ylim(0.94, 1)
ax.set_yticks([0.95, 1])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=10)
# ax.legend(loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig('R2_vs_ILD_for_paper.png', dpi=300, bbox_inches='tight')
plt.savefig('R2_vs_ILD_for_paper.pdf', bbox_inches='tight')
plt.show()
print("Saved: R2_vs_ILD_for_paper.png and .pdf")
# %%
