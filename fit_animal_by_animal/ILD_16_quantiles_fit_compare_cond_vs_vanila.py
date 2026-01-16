# %%
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load v2 (condition-by-condition fit) data
with open('ILD_16_cond_fit_quantiles.pkl', 'rb') as f:
    cond_data = pickle.load(f)

# Load v3 (vanilla model) data
with open('ILD_16_vanila_model_quantiles.pkl', 'rb') as f:
    vanilla_data = pickle.load(f)

# Extract data
ABL_arr = cond_data['ABL_arr']
abl_colors = cond_data['abl_colors']

# Theory data
cond_quantile_levels = cond_data['theory']['quantile_levels']
cond_mean = cond_data['theory']['mean_quantiles']
cond_sem = cond_data['theory']['sem_quantiles']

vanilla_quantile_levels = vanilla_data['theory']['quantile_levels']
vanilla_mean = vanilla_data['theory']['mean_quantiles']
vanilla_sem = vanilla_data['theory']['sem_quantiles']

# Empirical data (same for both)
plotting_quantiles = cond_data['empirical']['plotting_quantiles']
mean_unscaled = cond_data['empirical']['mean_unscaled']
sem_unscaled = cond_data['empirical']['sem_unscaled']
ild_idx = cond_data['empirical']['ild_idx']

# %%
# --- Plot: Compare cond fit vs vanilla fit ---
fig, ax = plt.subplots(figsize=(7, 6))

for i, ABL in enumerate(ABL_arr):
    color = abl_colors[ABL]
    
    # Condition fit (v2) - solid line
    if ABL in cond_mean:
        ax.plot(cond_quantile_levels, cond_mean[ABL], color=color, 
                linewidth=2, linestyle='-', label='Cond fit' if i == 0 else None)
        # ax.fill_between(cond_quantile_levels, 
        #                 cond_mean[ABL] - cond_sem[ABL], 
        #                 cond_mean[ABL] + cond_sem[ABL],
        #                 color=color, alpha=0.2)
    
    # Vanilla fit (v3) - dotted line
    if ABL in vanilla_mean:
        ax.plot(vanilla_quantile_levels, vanilla_mean[ABL], color=color, 
                linewidth=2, linestyle=':', label='Vanilla' if i == 0 else None)
        # ax.fill_between(vanilla_quantile_levels, 
        #                 vanilla_mean[ABL] - vanilla_sem[ABL], 
        #                 vanilla_mean[ABL] + vanilla_sem[ABL],
        #                 color=color, alpha=0.1)

# Overlay empirical data points
for i, abl in enumerate(ABL_arr):
    q_vals = mean_unscaled[abl][:, ild_idx]
    q_sem = sem_unscaled[abl][:, ild_idx]
    ax.errorbar(
        plotting_quantiles,
        q_vals,
        yerr=q_sem,
        marker='o',
        linestyle='none',
        color=abl_colors[abl],
        markersize=8,
        capsize=3,
        label='Data' if i == 0 else None
    )

ax.set_xlabel('Quantile', fontsize=14)
ax.set_ylabel('RT (s)', fontsize=14)
ax.set_title('Cond Fit (solid) vs Vanilla (dotted) at |ILD| = 16 dB', fontsize=14)
ax.legend(loc='upper left', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_xlim(0.05, 0.95)
ax.set_ylim(0, 0.35)
ax.set_yticks([0, 0.35])
ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
ax.set_xticklabels(['10', '30', '50', '70', '90'])

plt.tight_layout()
plt.savefig('ILD_16_quantiles_cond_vs_vanilla_compare.png', dpi=300, bbox_inches='tight')
plt.show()

print("Saved comparison plot to ILD_16_quantiles_cond_vs_vanilla_compare.png")

# %%
# =============================================================================
# Quantify Goodness of Fit: Compare theoretical vs empirical quantiles
# =============================================================================

def compute_gof_metrics(theory_quantiles, theory_q_levels, emp_quantiles, emp_q_levels, emp_sem):
    """
    Compute goodness of fit metrics between theoretical and empirical quantiles.
    Interpolates theory to match empirical quantile levels.
    
    Returns: dict with SSE, weighted_SSE, RMSE, R²
    """
    # Interpolate theoretical quantiles to empirical quantile levels
    theory_interp = np.interp(emp_q_levels, theory_q_levels, theory_quantiles)
    
    # Residuals
    residuals = theory_interp - emp_quantiles
    
    # SSE (Sum of Squared Errors)
    sse = np.sum(residuals**2)
    
    # Weighted SSE (inverse variance weighting)
    # Avoid division by zero
    weights = 1.0 / (emp_sem**2 + 1e-10)
    weighted_sse = np.sum(weights * residuals**2)
    
    # RMSE
    rmse = np.sqrt(np.mean(residuals**2))
    
    # R² (coefficient of determination)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((emp_quantiles - np.mean(emp_quantiles))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    
    return {
        'SSE': sse,
        'weighted_SSE': weighted_sse,
        'RMSE': rmse,
        'R2': r_squared,
        'residuals': residuals,
        'theory_interp': theory_interp
    }

# Compute metrics for each ABL and model
print("\n" + "="*60)
print("Goodness of Fit Metrics: Theory vs Empirical Quantiles")
print("="*60)

gof_results = {'cond': {}, 'vanilla': {}}

for ABL in ABL_arr:
    emp_q = mean_unscaled[ABL][:, ild_idx]
    emp_sem = sem_unscaled[ABL][:, ild_idx]
    
    # Cond fit
    if ABL in cond_mean:
        gof_cond = compute_gof_metrics(cond_mean[ABL], cond_quantile_levels, 
                                        emp_q, plotting_quantiles, emp_sem)
        gof_results['cond'][ABL] = gof_cond
    
    # Vanilla fit
    if ABL in vanilla_mean:
        gof_vanilla = compute_gof_metrics(vanilla_mean[ABL], vanilla_quantile_levels,
                                           emp_q, plotting_quantiles, emp_sem)
        gof_results['vanilla'][ABL] = gof_vanilla

# Print table
print(f"\n{'ABL':<8} {'Model':<10} {'SSE':<12} {'RMSE':<12} {'R²':<10}")
print("-"*52)
for ABL in ABL_arr:
    if ABL in gof_results['cond']:
        g = gof_results['cond'][ABL]
        print(f"{ABL:<8} {'Cond':<10} {g['SSE']:<12.6f} {g['RMSE']:<12.6f} {g['R2']:<10.4f}")
    if ABL in gof_results['vanilla']:
        g = gof_results['vanilla'][ABL]
        print(f"{ABL:<8} {'Vanilla':<10} {g['SSE']:<12.6f} {g['RMSE']:<12.6f} {g['R2']:<10.4f}")

# Aggregate across ABLs
print("\n" + "-"*52)
print(f"{'ABL':<8} {'Model':<10} {'SSE (sum)':<12} {'RMSE (mean)':<12} {'R² (mean)':<10}")
print("-"*52)
for model in ['cond', 'vanilla']:
    total_sse = sum(gof_results[model][ABL]['SSE'] for ABL in ABL_arr if ABL in gof_results[model])
    mean_rmse = np.mean([gof_results[model][ABL]['RMSE'] for ABL in ABL_arr if ABL in gof_results[model]])
    mean_r2 = np.mean([gof_results[model][ABL]['R2'] for ABL in ABL_arr if ABL in gof_results[model]])
    print(f"{'All':<8} {model.capitalize():<10} {total_sse:<12.6f} {mean_rmse:<12.6f} {mean_r2:<10.4f}")

# %%
