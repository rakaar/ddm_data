# VBMC fit on log-odds lapse biased model
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyvbmc import VBMC
import pickle
import sys
sys.path.append('.')
from vbmc_animal_wise_fit_utils import trapezoidal_logpdf

# %%
# Data loading
T_trunc = 0.3
batch_name = 'LED8'
animal_ids = [105]

csv_filename = f'batch_csvs/batch_{batch_name}_valid_and_aborts.csv'
exp_df = pd.read_csv(csv_filename)
df_valid_and_aborts = exp_df[
    (exp_df['success'].isin([1,-1])) |
    (exp_df['abort_event'] == 3)
].copy()

df_aborts = df_valid_and_aborts[df_valid_and_aborts['abort_event'] == 3]
animal_idx = 0
animal = animal_ids[animal_idx]
df_all_trials_animal = df_valid_and_aborts[df_valid_and_aborts['animal'] == animal]
df_aborts_animal = df_aborts[df_aborts['animal'] == animal]
df_valid_animal = df_all_trials_animal[df_all_trials_animal['success'].isin([1,-1])]
max_rt = df_valid_animal['RTwrtStim'].max()
df_valid_animal_filtered = df_valid_animal[df_valid_animal['RTwrtStim'] > 0].copy()
df_valid_animal_filtered['abs_ILD'] = np.abs(df_valid_animal_filtered['ILD'])

ABL_vals = df_valid_animal_filtered['ABL'].unique()
ILD_vals = sorted(df_valid_animal_filtered['ILD'].unique())

# %%
# Collect all log odds data across all 3 ABLs for fitting
all_x_data = []
all_y_data = []

for idx, abl in enumerate(ABL_vals[:3]):
    for ild in ILD_vals:
        empirical_subset = df_valid_animal_filtered[
            (df_valid_animal_filtered['ABL'] == abl) & 
            (df_valid_animal_filtered['ILD'] == ild)
        ]
        if len(empirical_subset) > 0:
            p_right = np.mean(empirical_subset['choice'] == 1)
            p_left = np.mean(empirical_subset['choice'] == -1)
            if p_left > 0 and p_right > 0:
                log_odds_empirical = np.log(p_right / p_left)
                all_x_data.append(ild)
                all_y_data.append(log_odds_empirical)

all_x_data = np.array(all_x_data)
all_y_data = np.array(all_y_data)

# Remove any NaN values
valid_mask = ~np.isnan(all_y_data)
all_x_data = all_x_data[valid_mask]
all_y_data = all_y_data[valid_mask]

print(f"Number of data points for log-odds fitting: {len(all_x_data)}")

# %%
# Model definition
def log_sigmoid_lapse_biased(x, a, d, th, lapse_pR):
    """Log-odds model with biased lapse"""
    f = th * np.tanh(d * x)
    p0 = 1.0 / (1.0 + np.exp(-2*f))
    p_plus = a*lapse_pR + (1.0 - a) * p0
    p_minus = a*(1-lapse_pR) + (1.0 - a) * (1.0 - p0)
    eps = 1e-50  # Small epsilon to avoid log(0)
    return np.log((p_plus + eps) / (p_minus + eps))

# %%
# VBMC likelihood function
def vbmc_logodds_loglike_fn(params):
    """Compute log-likelihood for log-odds data"""
    a, d, th, lapse_pR = params
    
    # Predict log-odds
    y_pred = np.array([log_sigmoid_lapse_biased(x, a, d, th, lapse_pR) for x in all_x_data])
    
    # Gaussian likelihood: using empirical noise std from data
    sigma = 2.005031  # Empirical std of log-odds data
    log_likelihood = -0.5 * np.sum(((all_y_data - y_pred) / sigma)**2) - len(all_x_data) * np.log(sigma * np.sqrt(2 * np.pi))
    
    return log_likelihood

# %%
# VBMC prior function
def vbmc_logodds_prior_fn(params):
    """Compute log-prior for parameters"""
    a, d, th, lapse_pR = params
    
    a_logpdf = trapezoidal_logpdf(
        a,
        logodds_a_bounds[0],
        logodds_a_plausible_bounds[0],
        logodds_a_plausible_bounds[1],
        logodds_a_bounds[1]
    )
    
    d_logpdf = trapezoidal_logpdf(
        d,
        logodds_d_bounds[0],
        logodds_d_plausible_bounds[0],
        logodds_d_plausible_bounds[1],
        logodds_d_bounds[1]
    )
    
    th_logpdf = trapezoidal_logpdf(
        th,
        logodds_th_bounds[0],
        logodds_th_plausible_bounds[0],
        logodds_th_plausible_bounds[1],
        logodds_th_bounds[1]
    )
    
    lapse_pR_logpdf = trapezoidal_logpdf(
        lapse_pR,
        logodds_lapse_pR_bounds[0],
        logodds_lapse_pR_plausible_bounds[0],
        logodds_lapse_pR_plausible_bounds[1],
        logodds_lapse_pR_bounds[1]
    )
    
    return a_logpdf + d_logpdf + th_logpdf + lapse_pR_logpdf

# %%
# VBMC joint function
def vbmc_logodds_joint_fn(params):
    """Compute joint log-probability (prior + likelihood)"""
    priors = vbmc_logodds_prior_fn(params)
    loglike = vbmc_logodds_loglike_fn(params)
    return priors + loglike

# %%
# Define bounds
# Parameters: [a, d, th, lapse_pR]
logodds_a_bounds = [1e-6, 1-1e-6]
logodds_d_bounds = [0.01, 10]
logodds_th_bounds = [0.01, 50]
logodds_lapse_pR_bounds = [0.0, 1.0]

# Plausible bounds (narrower, for initialization region)
logodds_a_plausible_bounds = [0.01, 0.2]
logodds_d_plausible_bounds = [0.05, 2.0]
logodds_th_plausible_bounds = [0.5, 10.0]
logodds_lapse_pR_plausible_bounds = [0.3, 0.7]

# Lower and upper bounds for VBMC
lb = np.array([
    logodds_a_bounds[0],
    logodds_d_bounds[0],
    logodds_th_bounds[0],
    logodds_lapse_pR_bounds[0]
])

ub = np.array([
    logodds_a_bounds[1],
    logodds_d_bounds[1],
    logodds_th_bounds[1],
    logodds_lapse_pR_bounds[1]
])

plb = np.array([
    logodds_a_plausible_bounds[0],
    logodds_d_plausible_bounds[0],
    logodds_th_plausible_bounds[0],
    logodds_lapse_pR_plausible_bounds[0]
])

pub = np.array([
    logodds_a_plausible_bounds[1],
    logodds_d_plausible_bounds[1],
    logodds_th_plausible_bounds[1],
    logodds_lapse_pR_plausible_bounds[1]
])

# %%
# Random initialization within plausible bounds
np.random.seed(42)
x_0 = np.array([
    np.random.uniform(plb[0], pub[0]),  # a
    np.random.uniform(plb[1], pub[1]),  # d
    np.random.uniform(plb[2], pub[2]),  # th
    np.random.uniform(plb[3], pub[3])   # lapse_pR
])

print("Initial parameters:")
print(f"  a        = {x_0[0]:.6f}")
print(f"  d        = {x_0[1]:.6f}")
print(f"  th       = {x_0[2]:.6f}")
print(f"  lapse_pR = {x_0[3]:.6f}")

# %%
# Run VBMC
print("\n" + "="*60)
print("Running VBMC for log-odds lapse biased model...")
print("="*60)

vbmc = VBMC(
    vbmc_logodds_joint_fn, 
    x_0, 
    lb, 
    ub, 
    plb, 
    pub, 
    options={
        'display': 'on', 
        'max_fun_evals': 200 * (2 + 4)  # 2 + number of parameters
    }
)

vp, results = vbmc.optimize()

# Save results
output_file = f'vbmc_logodds_lapse_biased_batch_{batch_name}_animal_{animal}.pkl'
vbmc.save(output_file, overwrite=True)
print(f"\nResults saved to: {output_file}")

# %%
# Sample from posterior
print("\n" + "="*60)
print("Sampling from posterior...")
print("="*60)

vp_samples = vp.sample(int(1e6))[0]

# Extract parameter samples
a_samples = vp_samples[:, 0]
d_samples = vp_samples[:, 1]
th_samples = vp_samples[:, 2]
lapse_pR_samples = vp_samples[:, 3]

# Compute statistics
print("\nPosterior statistics:")
print(f"{'Parameter':<15} {'Mean':<15} {'Std':<15} {'Median':<15}")
print("-"*60)
print(f"{'a':<15} {np.mean(a_samples):<15.6f} {np.std(a_samples):<15.6f} {np.median(a_samples):<15.6f}")
print(f"{'d':<15} {np.mean(d_samples):<15.6f} {np.std(d_samples):<15.6f} {np.median(d_samples):<15.6f}")
print(f"{'th':<15} {np.mean(th_samples):<15.6f} {np.std(th_samples):<15.6f} {np.median(th_samples):<15.6f}")
print(f"{'lapse_pR':<15} {np.mean(lapse_pR_samples):<15.6f} {np.std(lapse_pR_samples):<15.6f} {np.median(lapse_pR_samples):<15.6f}")

# %%
# Plot posterior distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle(f'Posterior Distributions - Log-Odds Fit\nBatch {batch_name}, Animal {animal}', fontsize=14, fontweight='bold')

# Plot a
ax = axes[0, 0]
ax.hist(a_samples, bins=50, density=True, histtype='step', color='blue', linewidth=2)
ax.axvline(np.mean(a_samples), color='red', linestyle='--', linewidth=2, label=f'Mean={np.mean(a_samples):.4f}')
ax.set_xlabel('a (lapse rate)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.legend()
ax.grid(True, alpha=0.3)

# Plot d
ax = axes[0, 1]
ax.hist(d_samples, bins=50, density=True, histtype='step', color='blue', linewidth=2)
ax.axvline(np.mean(d_samples), color='red', linestyle='--', linewidth=2, label=f'Mean={np.mean(d_samples):.4f}')
ax.set_xlabel('lambda/chi (slope)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.legend()
ax.grid(True, alpha=0.3)

# Plot th
ax = axes[1, 0]
ax.hist(th_samples, bins=50, density=True, histtype='step', color='blue', linewidth=2)
ax.axvline(np.mean(th_samples), color='red', linestyle='--', linewidth=2, label=f'Mean={np.mean(th_samples):.4f}')
ax.set_xlabel('th (threshold)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.legend()
ax.grid(True, alpha=0.3)

# Plot lapse_pR
ax = axes[1, 1]
ax.hist(lapse_pR_samples, bins=50, density=True, histtype='step', color='blue', linewidth=2)
ax.axvline(np.mean(lapse_pR_samples), color='red', linestyle='--', linewidth=2, label=f'Mean={np.mean(lapse_pR_samples):.4f}')
ax.set_xlabel('lapse_pR (lapse prob right)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'vbmc_logodds_posterior_batch_{batch_name}_animal_{animal}.pdf')
plt.show()

print("\nVBMC fitting complete!")

# %%
# Plot log-odds data with fitted theoretical curve using mean parameters
print("\n" + "="*60)
print("Plotting fitted model with empirical data...")
print("="*60)

# Compute mode (MAP) parameters from posterior samples using fine-grained histogram
def find_mode(samples, bin_size_factor=1e-3):
    """Find mode of distribution using histogram with fine bins"""
    # Bin size is factor times the mean of the parameter
    param_mean = np.mean(samples)
    bin_size = max(abs(param_mean) * bin_size_factor, 1e-6)  # Avoid too small bins
    
    # Create histogram
    hist, bin_edges = np.histogram(samples, bins=np.arange(samples.min(), samples.max() + bin_size, bin_size))
    
    # Find bin with highest count
    max_idx = np.argmax(hist)
    mode_value = (bin_edges[max_idx] + bin_edges[max_idx + 1]) / 2  # Center of bin
    
    return mode_value

a_mode = find_mode(a_samples)
d_mode = find_mode(d_samples)
th_mode = find_mode(th_samples)
lapse_pR_mode = find_mode(lapse_pR_samples)

print(f"\nUsing posterior MODE (highest density) parameters:")
print(f"  a        = {a_mode:.6f}  (mean = {np.mean(a_samples):.6f})")
print(f"  d        = {d_mode:.6f}  (mean = {np.mean(d_samples):.6f})")
print(f"  th       = {th_mode:.6f}  (mean = {np.mean(th_samples):.6f})")
print(f"  lapse_pR = {lapse_pR_mode:.6f}  (mean = {np.mean(lapse_pR_samples):.6f})")

# Create plot
fig_fit, axes_fit = plt.subplots(1, 3, figsize=(15, 5))
fig_fit.suptitle(f'Log-Odds with VBMC Fit - Batch {batch_name}, Animal {animal}', 
                 fontsize=14, fontweight='bold')

# Generate smooth curve for model using mode parameters
x_model = np.linspace(-16, 16, 300)
y_fitted = log_sigmoid_lapse_biased(x_model, a_mode, d_mode, th_mode, lapse_pR_mode)

# Plot for each ABL
for idx, abl in enumerate(ABL_vals[:3]):
    ax = axes_fit[idx]
    
    # Collect empirical log-odds for this ABL
    empirical_log_odds = []
    empirical_ild_vals = []
    
    for ild in ILD_vals:
        empirical_subset = df_valid_animal_filtered[
            (df_valid_animal_filtered['ABL'] == abl) & 
            (df_valid_animal_filtered['ILD'] == ild)
        ]
        if len(empirical_subset) > 0:
            p_right = np.mean(empirical_subset['choice'] == 1)
            p_left = np.mean(empirical_subset['choice'] == -1)
            if p_left > 0 and p_right > 0:
                log_odds_empirical = np.log(p_right / p_left)
                empirical_log_odds.append(log_odds_empirical)
                empirical_ild_vals.append(ild)
    
    # Plot empirical data
    ax.plot(empirical_ild_vals, empirical_log_odds, 's', color='green', markersize=8, 
            label='Data', markerfacecolor='green', markeredgecolor='green', alpha=0.6)
    
    # Plot fitted curve
    ax.plot(x_model, y_fitted, '-', color='blue', linewidth=2.5, 
            label=f'VBMC (MAP): a={a_mode:.3f}, d={d_mode:.2f}, th={th_mode:.2f}, pR={lapse_pR_mode:.3f}')
    
    # Formatting
    ax.set_title(f'ABL = {abl} dB', fontsize=12)
    ax.set_xlabel('ILD (dB)', fontsize=11)
    if idx == 0:
        ax.set_ylabel('log(P(right) / P(left))', fontsize=11)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_ylim(-5, 5)

plt.tight_layout()
plt.savefig(f'vbmc_logodds_fit_batch_{batch_name}_animal_{animal}.pdf')
plt.show()

print("\nPlotting complete!") 