"""
VBMC Fitting of Real Animal Data with Proactive LED Model
==========================================================

This script fits the proactive process model to real experimental data aggregated across
ALL animals using Variational Bayesian Monte Carlo (VBMC).

SCRIPT STRUCTURE (cell-by-cell):
--------------------------------
1. **Imports** - numpy, matplotlib, joblib, pandas, pyvbmc, corner
2. **Parameters** - T_trunc truncation threshold
3. **Load & Filter Data** - Load real data, filter by session/training, aggregate all animals
4. **Build Fitting DataFrame** - RT (timed_fix), t_stim (intended_fix), t_LED, LED_trial
5. **Likelihood Functions** - Same as simulated version (truncation + censoring)
6. **Prior Functions** - Trapezoidal priors for VBMC
7. **VBMC Setup** - Define bounds, plausible bounds, and joint function
8. **Run VBMC** - Optimize and get posterior samples
9. **Posterior Visualization** - Corner plot of posterior
10. **Diagnostic Plots**:
    - RTD wrt fixation: data hist, theory (fitted), sim (fitted) — for LED ON/OFF
    - RTD wrt LED: data hist, sim (fitted) — for LED ON/OFF


"""

# %%
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
import os
import sys
sys.path.append('../fit_each_condn')
from psiam_tied_dv_map_utils_with_PDFs import stupid_f_integral, d_A_RT
from post_LED_censor_utils import cum_A_t_fn
from pyvbmc import VBMC
import corner
import pickle

# %%
# =============================================================================
# PARAMETERS
# =============================================================================
# T_trunc removed (no truncation)
ANIMAL_ID = 0  # None for all animals, or integer index (0, 1, 2...)

# %%
# =============================================================================
# Load and filter data
# =============================================================================
og_df = pd.read_csv('../out_LED.csv')

df = og_df[og_df['repeat_trial'].isin([0, 2]) | og_df['repeat_trial'].isna()]
session_type = 7
df = df[df['session_type'].isin([session_type])]
training_level = 16
df = df[df['training_level'].isin([training_level])]

df = df.dropna(subset=['intended_fix', 'LED_onset_time', 'timed_fix'])
df = df[(df['abort_event'] == 3) | (df['success'].isin([1, -1]))]

# Filter out aborts < 300ms (T_trunc)
# df = df[~((df['abort_event'] == 3) & (df['timed_fix'] < 0.3))] # REMOVED

# Get unique animals (for info)
unique_animals = df['animal'].unique()

if ANIMAL_ID is not None:
    animal_name = unique_animals[ANIMAL_ID]
    print(f"Selected ANIMAL_ID={ANIMAL_ID} -> Animal: {animal_name}")
    df_all = df[df['animal'] == animal_name]
    animal_label = f"Animal {animal_name}"
    file_tag = f"animal_{animal_name}"
else:
    print(f"Aggregating all {len(unique_animals)} animals: {unique_animals}")
    df_all = df
    animal_label = "All Animals Aggregated"
    file_tag = "all_animals"

# Separate LED ON and OFF
df_on = df_all[df_all['LED_trial'] == 1]
df_off = df_all[df_all['LED_trial'].isin([0, np.nan])]

print(f"\n{animal_label} data summary:")
print(f"  Total trials: {len(df_all)}")
print(f"  LED ON trials: {len(df_on)}")
print(f"  LED OFF trials: {len(df_off)}")

# %%
# =============================================================================
# Build fitting DataFrame
# =============================================================================
# For LED ON trials
df_on_fit = pd.DataFrame({
    'RT': df_on['timed_fix'].values,
    't_stim': df_on['intended_fix'].values,
    't_LED': (df_on['intended_fix'] - df_on['LED_onset_time']).values,
    'LED_trial': 1
})

# For LED OFF trials (keep t_LED for plotting, even though LED was off)
df_off_fit = pd.DataFrame({
    'RT': df_off['timed_fix'].values,
    't_stim': df_off['intended_fix'].values,
    't_LED': (df_off['intended_fix'] - df_off['LED_onset_time']).values,  # Keep for RT wrt LED plotting
    'LED_trial': 0
})

# Combine
fit_df = pd.concat([df_on_fit, df_off_fit], ignore_index=True)

# Apply truncation only to abort trials (RT < t_stim)
# Keep all censored trials (RT >= t_stim) regardless of RT
# fit_df = fit_df[~((fit_df['RT'] < fit_df['t_stim']) & (fit_df['RT'] <= T_trunc))] # REMOVED

print(f"\nFitting DataFrame summary (no truncation):")
print(f"  Total trials: {len(fit_df)}")
print(f"  LED ON trials: {len(fit_df[fit_df['LED_trial'] == 1])}")
print(f"  LED OFF trials: {len(fit_df[fit_df['LED_trial'] == 0])}")

# Check abort vs censored
n_aborts = len(fit_df[fit_df['RT'] < fit_df['t_stim']])
n_censored = len(fit_df[fit_df['RT'] >= fit_df['t_stim']])
print(f"  Abort trials (RT < t_stim): {n_aborts}")
print(f"  Censored trials (RT >= t_stim): {n_censored}")

# Get LED and stimulus timing distributions from ALL trials (for simulation and theoretical calculations)
# IMPORTANT: Keep as paired arrays to preserve trial-level correlation (t_LED = t_stim - LED_onset_time)
stim_times = df_all['intended_fix'].values
LED_times = (df_all['intended_fix'] - df_all['LED_onset_time']).values
n_trials_data = len(stim_times)  # For sampling by trial index

# %%
bins= np.arange(0,3,0.02)
plt.hist(stim_times, bins=bins, density=True, histtype='step')
plt.hist(LED_times, bins=bins, density=True, histtype='step', ls='--')
plt.legend()
plt.show()
# %%
# =============================================================================
# Simulation function
# =============================================================================
def simulate_proactive_single_bound(V_A_base, V_A_post_LED, theta_A, t_LED, t_stim, del_a_minus_del_LED, del_m_plus_del_LED, is_led_trial, dt=1e-4):
    """
    Simulate proactive process with single bound accumulator.
    Drift changes from V_A_base to V_A_post_LED at t_LED - del_a_minus_del_LED (only for LED ON trials).
    Proactive process starts at t = 0 (no noise before this).
    Returns RT when accumulator hits theta_A.
    """
    AI = 0
    t = 0
    dB = np.sqrt(dt)

    while True:
        if is_led_trial and t >= t_LED - del_a_minus_del_LED:
            V_A = V_A_post_LED
        else:
            V_A = V_A_base

        AI += V_A * dt + np.random.normal(0, dB)
        t += dt

        if AI >= theta_A:
            RT = t + (del_m_plus_del_LED + del_a_minus_del_LED)
            return RT

# %%
# =============================================================================
# Likelihood functions
# =============================================================================
def PA_with_LEDON_2_adapted(t, v, vON, a, del_a_minus_del_LED, del_m_plus_del_LED, tled):
    """
    Compute the PA pdf by combining contributions before and after LED onset.
    """
    tp = tled - del_a_minus_del_LED  # t_LED + del_LED - del_a
    t_post_led = t - tled - del_m_plus_del_LED  # RT - t_LED - del_m_plus_del_LED
    
    t_shift_off = t - (del_m_plus_del_LED + del_a_minus_del_LED) # In LED OFF, removed del_a, del_m
    t_shift_on = t - tled - del_m_plus_del_LED

    if tp > 0 and t_post_led <= 0: # only 1st, NOT 2nd
        pdf = d_A_RT(v * a, t_shift_off / (a**2)) / (a**2)
    else:
        if tp <= 0: # only 2nd, NOT 1st
            pdf = d_A_RT(vON * a, t_shift_on / (a**2)) / (a**2)
        else: # both 1st and 2nd
            pdf = stupid_f_integral(v, vON, a, t_post_led, tp)

    return pdf


def led_off_cdf(t, v, a, del_a_minus_del_LED, del_m_plus_del_LED):
    if t <= del_m_plus_del_LED + del_a_minus_del_LED:
        return 0
    return cum_A_t_fn(t - (del_m_plus_del_LED + del_a_minus_del_LED), v, a)


def led_off_pdf(t, v, a, del_a_minus_del_LED, del_m_plus_del_LED):
    if t <= del_m_plus_del_LED + del_a_minus_del_LED:
        return 0

    pdf = d_A_RT(v * a, (t - (del_m_plus_del_LED + del_a_minus_del_LED)) / (a**2)) / (a**2)
    return pdf


def led_off_survival(t_stim, v, a, del_a_minus_del_LED, del_m_plus_del_LED):
    return 1 - led_off_cdf(t_stim, v, a, del_a_minus_del_LED, del_m_plus_del_LED)


def led_on_survival(t_stim, t_led, v, vON, a, del_a_minus_del_LED, del_m_plus_del_LED):
    t_pts_cdf = np.arange(0, t_stim + 0.001, 0.001)
    pdf_vals_cdf = np.array([
        PA_with_LEDON_2_adapted(ti, v, vON, a, del_a_minus_del_LED, del_m_plus_del_LED, t_led)
        for ti in t_pts_cdf
    ])
    cdf_t_stim = np.trapz(pdf_vals_cdf, t_pts_cdf)
    return 1 - cdf_t_stim


def lapse_pdf(t, beta):
    return beta * np.exp(-beta * t)


def lapse_survival(t_stim, beta):
    return np.exp(-beta * t_stim)


def compute_trial_loglike(row, V_A_base, V_A_post_LED, theta_A, del_a_minus_del_LED, del_m_plus_del_LED, lapse_prob, beta_lapse):
    t = row['RT']
    t_stim = row['t_stim']
    is_led = row['LED_trial'] == 1
    t_led = row['t_LED']

    if is_led and (t_led is None or (isinstance(t_led, float) and np.isnan(t_led))):
        raise ValueError("LED trial has invalid t_LED (None/NaN).")

    if t < t_stim:
        if is_led:
            proactive_ll = PA_with_LEDON_2_adapted(
                t, V_A_base, V_A_post_LED, theta_A, del_a_minus_del_LED, del_m_plus_del_LED, t_led
            )
        else:
            proactive_ll = led_off_pdf(t, V_A_base, theta_A, del_a_minus_del_LED, del_m_plus_del_LED)
        likelihood = (1 - lapse_prob) * proactive_ll + lapse_prob * lapse_pdf(t, beta_lapse)
    else:
        if is_led:
            proactive_surv = led_on_survival(
                t_stim, t_led, V_A_base, V_A_post_LED, theta_A, del_a_minus_del_LED, del_m_plus_del_LED
            )
        else:
            proactive_surv = led_off_survival(t_stim, V_A_base, theta_A, del_a_minus_del_LED, del_m_plus_del_LED)
        likelihood = (1 - lapse_prob) * proactive_surv + lapse_prob * lapse_survival(t_stim, beta_lapse)

    if likelihood <= 0 or np.isnan(likelihood):
        likelihood = 1e-50

    return np.log(likelihood)


def proactive_led_loglike(params):
    V_A_base, V_A_post_LED, theta_A, del_a_minus_del_LED, del_m_plus_del_LED, lapse_prob, beta_lapse = params
    all_loglike = Parallel(n_jobs=30)(
        delayed(compute_trial_loglike)(
            row, V_A_base, V_A_post_LED, theta_A, del_a_minus_del_LED, del_m_plus_del_LED, lapse_prob, beta_lapse
        ) for _, row in fit_df.iterrows()
    )
    return np.sum(all_loglike)

# %%
# =============================================================================
# Bounds and priors
# =============================================================================
V_A_base_bounds = [0.1, 8]
V_A_post_LED_bounds = [0.1, 8]
theta_A_bounds = [0.1, 8]
del_a_minus_del_LED_bounds = [-1.1, 1.1]
del_m_plus_del_LED_bounds = [0.001, 0.2]
lapse_prob_bounds = [0.0, 1.0]
beta_lapse_bounds = [0.001, 20.0]

V_A_base_plausible = [0.5, 3]
V_A_post_LED_plausible = [0.5, 3]
theta_A_plausible = [0.5, 3]
del_a_minus_del_LED_plausible = [0.01, 0.07]
del_m_plus_del_LED_plausible = [0.01, 0.07]
lapse_prob_plausible = [0.01, 0.3]
beta_lapse_plausible = [0.5, 5.0]


def trapezoidal_logpdf(x, a, b, c, d):
    if x < a or x > d:
        return -np.inf
    area = ((b - a) + (d - c)) / 2 + (c - b)
    h_max = 1.0 / area
    
    if a <= x <= b:
        pdf_value = ((x - a) / (b - a)) * h_max
    elif b < x < c:
        pdf_value = h_max
    elif c <= x <= d:
        pdf_value = ((d - x) / (d - c)) * h_max
    else:
        pdf_value = 0.0

    if pdf_value <= 0.0:
        return -np.inf
    else:
        return np.log(pdf_value)


def vbmc_prior_fn(params):
    V_A_base, V_A_post_LED, theta_A, del_a_minus_del_LED, del_m_plus_del_LED, lapse_prob, beta_lapse = params

    log_prior = 0
    log_prior += trapezoidal_logpdf(V_A_base, V_A_base_bounds[0], V_A_base_plausible[0], 
                                     V_A_base_plausible[1], V_A_base_bounds[1])
    log_prior += trapezoidal_logpdf(V_A_post_LED, V_A_post_LED_bounds[0], V_A_post_LED_plausible[0], 
                                     V_A_post_LED_plausible[1], V_A_post_LED_bounds[1])
    log_prior += trapezoidal_logpdf(theta_A, theta_A_bounds[0], theta_A_plausible[0], 
                                     theta_A_plausible[1], theta_A_bounds[1])
    log_prior += trapezoidal_logpdf(del_a_minus_del_LED, del_a_minus_del_LED_bounds[0], del_a_minus_del_LED_plausible[0], 
                                     del_a_minus_del_LED_plausible[1], del_a_minus_del_LED_bounds[1])
    log_prior += trapezoidal_logpdf(del_m_plus_del_LED, del_m_plus_del_LED_bounds[0], del_m_plus_del_LED_plausible[0], 
                                     del_m_plus_del_LED_plausible[1], del_m_plus_del_LED_bounds[1])
    log_prior += trapezoidal_logpdf(lapse_prob, lapse_prob_bounds[0], lapse_prob_plausible[0],
                                     lapse_prob_plausible[1], lapse_prob_bounds[1])
    log_prior += trapezoidal_logpdf(beta_lapse, beta_lapse_bounds[0], beta_lapse_plausible[0],
                                     beta_lapse_plausible[1], beta_lapse_bounds[1])
    return log_prior


def vbmc_joint(params):
    log_prior = vbmc_prior_fn(params)
    if not np.isfinite(log_prior):
        return -np.inf
    return log_prior + proactive_led_loglike(params)

# %%
# =============================================================================
# Set up VBMC
# =============================================================================
lb = np.array([V_A_base_bounds[0], V_A_post_LED_bounds[0], theta_A_bounds[0], 
               del_a_minus_del_LED_bounds[0], del_m_plus_del_LED_bounds[0],
               lapse_prob_bounds[0], beta_lapse_bounds[0]])
ub = np.array([V_A_base_bounds[1], V_A_post_LED_bounds[1], theta_A_bounds[1], 
               del_a_minus_del_LED_bounds[1], del_m_plus_del_LED_bounds[1],
               lapse_prob_bounds[1], beta_lapse_bounds[1]])
plb = np.array([V_A_base_plausible[0], V_A_post_LED_plausible[0], theta_A_plausible[0], 
                del_a_minus_del_LED_plausible[0], del_m_plus_del_LED_plausible[0],
                lapse_prob_plausible[0], beta_lapse_plausible[0]])
pub = np.array([V_A_base_plausible[1], V_A_post_LED_plausible[1], theta_A_plausible[1], 
                del_a_minus_del_LED_plausible[1], del_m_plus_del_LED_plausible[1],
                lapse_prob_plausible[1], beta_lapse_plausible[1]])

# Initial point (use values similar to simulated ground truth)
np.random.seed(42)
x_0 = np.array([
    1.6 ,   # V_A_base
    3.4 ,   # V_A_post_LED
    2.5 ,   # theta_A
    0.04 ,  # del_a_minus_del_LED
    0.05,   # del_m_plus_del_LED
    0.05,    # lapse_prob
    5     # beta_lapse
])

# Ensure x_0 is within plausible bounds
x_0 = np.clip(x_0, plb, pub)

print("\nVBMC setup:")
print(f"  Initial point: {x_0}")
print(f"  Lower bounds: {lb}")
print(f"  Upper bounds: {ub}")
print(f"  Plausible lower: {plb}")
print(f"  Plausible upper: {pub}")

# %%
# =============================================================================
# Run VBMC (or load saved results)
# =============================================================================
LOAD_SAVED_RESULTS = True
VP_PKL_PATH = f'vbmc_real_{file_tag}_fit_NO_TRUNC_with_lapse.pkl'
RESULTS_PKL_PATH = f'vbmc_real_{file_tag}_results_NO_TRUNC_with_lapse.pkl'

if LOAD_SAVED_RESULTS:
    print(f"\nLoading saved VBMC results from {VP_PKL_PATH}...")
    with open(VP_PKL_PATH, 'rb') as f:
        vp = pickle.load(f)
    results_summary = {}
    if os.path.exists(RESULTS_PKL_PATH) and os.path.getsize(RESULTS_PKL_PATH) > 0:
        with open(RESULTS_PKL_PATH, 'rb') as f:
            saved_results = pickle.load(f)
        results_summary = saved_results.get('results_summary', {})
    else:
        print(f"Warning: {RESULTS_PKL_PATH} is missing or empty; proceeding without saved summary.")
    print("Loaded saved VBMC results.")
else:
    print(f"\nRunning VBMC optimization for {animal_label}...")
    vbmc = VBMC(vbmc_joint, x_0, lb, ub, plb, pub, options={'display': 'iter'})
    vp, results = vbmc.optimize()

    print("\nVBMC optimization complete!")

    # Save results
    vp.save(VP_PKL_PATH, overwrite=True)

    results_summary = {
        'elbo': results.get('elbo', None),
        'elbo_sd': results.get('elbo_sd', None),
        'convergence_status': results.get('convergence_status', None)
    }
    with open(RESULTS_PKL_PATH, 'wb') as f:
        pickle.dump({
            'animals': unique_animals.tolist(),
            'animal_id': ANIMAL_ID,
            'results_summary': results_summary,
            'fit_df': fit_df,
            'bounds': {
                'lb': lb, 'ub': ub, 'plb': plb, 'pub': pub
            }
        }, f)
    print(f"Results saved for {animal_label}!")

# %%
# =============================================================================
# Sample from posterior
# =============================================================================
vp_samples = vp.sample(int(1e5))[0]

param_labels = ['V_A_base', 'V_A_post_LED', 'theta_A', 'del_a_minus_del_LED', 'del_m_plus_del_LED', 'lapse_prob', 'beta_lapse']
param_means = np.mean(vp_samples, axis=0)
param_stds = np.std(vp_samples, axis=0)

print(f"\nPosterior summary ({animal_label}):")
print(f"{'Parameter':<15} {'Mean':<12} {'Std':<12}")
print("-" * 40)
for i, label in enumerate(param_labels):
    print(f"{label:<15} {param_means[i]:<12.4f} {param_stds[i]:<12.4f}")

# %%
# =============================================================================
# Corner plot
# =============================================================================
fig = corner.corner(
    vp_samples, 
    labels=param_labels, 
    show_titles=True, 
    title_fmt=".4f",
    quantiles=[0.025, 0.5, 0.975]
)
plt.suptitle(f'{animal_label} Posterior', y=1.02)
plt.savefig(f'vbmc_real_{file_tag}_corner.pdf', bbox_inches='tight')
print(f"Corner plot saved as 'vbmc_real_{file_tag}_corner.pdf'")
plt.show()

# %%
# =============================================================================
# Distribution of del_a_minus_del_LED + del_m_plus_del_LED
# =============================================================================
sum_delay_samples = vp_samples[:, 3] + vp_samples[:, 4]
sum_delay_quantiles = np.quantile(sum_delay_samples, [0.025, 0.5, 0.975])

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(sum_delay_samples, bins=50, density=True, alpha=0.7, color='steelblue')
ax.axvline(sum_delay_quantiles[0], color='k', linestyle='--', label=f'2.5%={sum_delay_quantiles[0]:.4f}')
ax.axvline(sum_delay_quantiles[1], color='k', linestyle='-', label=f'50%={sum_delay_quantiles[1]:.4f}')
ax.axvline(sum_delay_quantiles[2], color='k', linestyle='--', label=f'97.5%={sum_delay_quantiles[2]:.4f}')
ax.set_xlabel('del_a_minus_del_LED + del_m_plus_del_LED (s)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title(f'Posterior: del_a_minus_del_LED + del_m_plus_del_LED - {animal_label}', fontsize=13)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f'vbmc_real_{file_tag}_sum_delay_posterior.pdf', bbox_inches='tight')
print(f"Sum-delay posterior saved as 'vbmc_real_{file_tag}_sum_delay_posterior.pdf'")
plt.show()

# %%
# =============================================================================
# Theoretical RTD calculation (conditioned on RT < t_stim, i.e., aborts only)
# P(RT=t | abort) = P(RT=t) * P(t_stim > t) / P(abort)
# =============================================================================
def compute_theoretical_rtd_on(t_pts, V_A_base, V_A_post_LED, theta_A, del_a_minus_del_LED, del_m_plus_del_LED, lapse_prob, beta_lapse,
                               N_mc=5000):
    """Compute theoretical RTD for LED ON, conditioned on RT < t_stim (aborts).
    
    Monte Carlo loop over t_led and t_stim samples, applying same conditions as likelihood.
    """
    pdf_samples = np.zeros((N_mc, len(t_pts)))
    
    for i in range(N_mc):
        # Sample trial index to preserve (t_LED, t_stim) correlation
        trial_idx = np.random.randint(n_trials_data)
        t_led = LED_times[trial_idx]
        t_stim = stim_times[trial_idx]
        
        # Compute PDF for each t, applying conditions from likelihood
        proactive_pdf = np.zeros(len(t_pts))
        for j, t in enumerate(t_pts):
            # Condition 2: t >= t_stim -> 0 (not an abort)
            if t >= t_stim:
                proactive_pdf[j] = 0
            # Valid abort: t < t_stim
            else:
                proactive_pdf[j] = PA_with_LEDON_2_adapted(
                    t, V_A_base, V_A_post_LED, theta_A, del_a_minus_del_LED, del_m_plus_del_LED, t_led
                )
        
        # Lapse PDF
        lapse_vals = lapse_pdf(t_pts, beta_lapse)
        # Apply censoring to lapse as well (t < t_stim)
        lapse_vals[t_pts >= t_stim] = 0
        
        # Mixture
        sample_pdf = (1 - lapse_prob) * proactive_pdf + lapse_prob * lapse_vals

        # Normalize this sample's PDF to integrate to 1 over valid region
        norm = np.trapz(sample_pdf, t_pts)
        if norm > 0:
            # pdf_samples[i, :] = sample_pdf / norm
            pdf_samples[i, :] = sample_pdf

    
    # Average over all MC samples
    return np.mean(pdf_samples, axis=0)


def compute_theoretical_rtd_off(t_pts, V_A_base, theta_A, del_a_minus_del_LED, del_m_plus_del_LED, lapse_prob, beta_lapse, N_mc=5000):
    """Compute theoretical RTD for LED OFF, conditioned on RT < t_stim (aborts).
    
    Monte Carlo loop over t_stim samples, applying same conditions as likelihood.
    """
    pdf_samples = np.zeros((N_mc, len(t_pts)))
    
    for i in range(N_mc):
        # Sample trial index (for consistency, even though LED OFF doesn't use t_led)
        trial_idx = np.random.randint(n_trials_data)
        t_stim = stim_times[trial_idx]
        
        # Compute PDF for each t, applying conditions from likelihood
        proactive_pdf = np.zeros(len(t_pts))
        for j, t in enumerate(t_pts):
            # Condition 2: t >= t_stim -> 0 (not an abort)
            if t >= t_stim:
                proactive_pdf[j] = 0
            # Valid abort: t < t_stim
            else:
                proactive_pdf[j] = led_off_pdf(t, V_A_base, theta_A, del_a_minus_del_LED, del_m_plus_del_LED)
        
        # Lapse PDF
        lapse_vals = lapse_pdf(t_pts, beta_lapse)
        # Apply censoring to lapse as well (t < t_stim)
        lapse_vals[t_pts >= t_stim] = 0

        # Mixture
        sample_pdf = (1 - lapse_prob) * proactive_pdf + lapse_prob * lapse_vals
        
        # Normalize this sample's PDF to integrate to 1 over valid region
        norm = np.trapz(sample_pdf, t_pts)
        if norm > 0:
            # pdf_samples[i, :] = sample_pdf / norm
            pdf_samples[i, :] = sample_pdf

    
    # Average over all MC samples
    return np.mean(pdf_samples, axis=0)


def compute_theoretical_rtd_on_components(t_pts, V_A_base, V_A_post_LED, theta_A, del_a_minus_del_LED, del_m_plus_del_LED, lapse_prob, beta_lapse,
                               N_mc=5000):
    """Compute theoretical RTD for LED ON, returning separate proactive and lapse components.
    
    Monte Carlo loop over t_led and t_stim samples, applying same conditions as likelihood.
    Returns: (proactive_component, lapse_component)
    """
    proactive_samples = np.zeros((N_mc, len(t_pts)))
    lapse_samples = np.zeros((N_mc, len(t_pts)))
    
    for i in range(N_mc):
        # Sample trial index to preserve (t_LED, t_stim) correlation
        trial_idx = np.random.randint(n_trials_data)
        t_led = LED_times[trial_idx]
        t_stim = stim_times[trial_idx]
        
        # Compute PDF for each t, applying conditions from likelihood
        proactive_pdf = np.zeros(len(t_pts))
        for j, t in enumerate(t_pts):
            # Condition 2: t >= t_stim -> 0 (not an abort)
            if t >= t_stim:
                proactive_pdf[j] = 0
            # Valid abort: t < t_stim
            else:
                proactive_pdf[j] = PA_with_LEDON_2_adapted(
                    t, V_A_base, V_A_post_LED, theta_A, del_a_minus_del_LED, del_m_plus_del_LED, t_led
                )
        
        # Lapse PDF
        lapse_vals = lapse_pdf(t_pts, beta_lapse)
        # Apply censoring to lapse as well (t < t_stim)
        lapse_vals[t_pts >= t_stim] = 0
        
        # Separate components (unnormalized per sample, will average then normalize)
        sample_proactive = (1 - lapse_prob) * proactive_pdf
        sample_lapse = lapse_prob * lapse_vals
        
        proactive_samples[i, :] = sample_proactive
        lapse_samples[i, :] = sample_lapse

    # Average over all MC samples
    return np.mean(proactive_samples, axis=0), np.mean(lapse_samples, axis=0)


def compute_theoretical_rtd_off_components(t_pts, V_A_base, theta_A, del_a_minus_del_LED, del_m_plus_del_LED, lapse_prob, beta_lapse, N_mc=5000):
    """Compute theoretical RTD for LED OFF, returning separate proactive and lapse components.
    
    Monte Carlo loop over t_stim samples, applying same conditions as likelihood.
    Returns: (proactive_component, lapse_component)
    """
    proactive_samples = np.zeros((N_mc, len(t_pts)))
    lapse_samples = np.zeros((N_mc, len(t_pts)))
    
    for i in range(N_mc):
        # Sample trial index
        trial_idx = np.random.randint(n_trials_data)
        t_stim = stim_times[trial_idx]
        
        # Compute PDF for each t
        proactive_pdf = np.zeros(len(t_pts))
        for j, t in enumerate(t_pts):
            if t >= t_stim:
                proactive_pdf[j] = 0
            else:
                proactive_pdf[j] = led_off_pdf(t, V_A_base, theta_A, del_a_minus_del_LED, del_m_plus_del_LED)
        
        # Lapse PDF
        lapse_vals = lapse_pdf(t_pts, beta_lapse)
        lapse_vals[t_pts >= t_stim] = 0

        # Separate components
        sample_proactive = (1 - lapse_prob) * proactive_pdf
        sample_lapse = lapse_prob * lapse_vals
        
        proactive_samples[i, :] = sample_proactive
        lapse_samples[i, :] = sample_lapse

    # Average over all MC samples
    return np.mean(proactive_samples, axis=0), np.mean(lapse_samples, axis=0)


# Compute theoretical distributions with fitted parameters
dt = 0.01
t_max = 3
t_pts = np.arange(0, t_max, dt)

print("\nComputing theoretical RT distributions with fitted parameters...")
pdf_theory_on_fit = compute_theoretical_rtd_on(
    t_pts, param_means[0], param_means[1], param_means[2], param_means[3], param_means[4], param_means[5], param_means[6],
    N_mc=5000
)
pdf_theory_off_fit = compute_theoretical_rtd_off(
    t_pts, param_means[0], param_means[2], param_means[3], param_means[4], param_means[5], param_means[6]
)

# Normalize
pdf_theory_on_fit /= np.trapz(pdf_theory_on_fit, t_pts)
pdf_theory_off_fit /= np.trapz(pdf_theory_off_fit, t_pts)

# %%
# =============================================================================
# Simulate with fitted parameters for comparison
# =============================================================================
N_trials_sim = int(300e3)
print(f"\nSimulating {N_trials_sim} trials with fitted parameters...")

def simulate_single_trial_fit():
    is_led_trial = np.random.random() < 1/3
    # Sample trial index to preserve (t_LED, t_stim) correlation
    trial_idx = np.random.randint(n_trials_data)
    t_LED = LED_times[trial_idx]
    t_stim = stim_times[trial_idx]

    lapse_prob = param_means[5]
    beta_lapse = param_means[6]
    is_lapse = np.random.random() < lapse_prob

    if is_lapse:
        rt = np.random.exponential(1.0 / beta_lapse)
    else:
        rt = simulate_proactive_single_bound(
            param_means[0], param_means[1], param_means[2],  # V_A_base, V_A_post_LED, theta_A
            t_LED if is_led_trial else None, t_stim,
            param_means[3], param_means[4],  # del_a_minus_del_LED, del_m_plus_del_LED
            is_led_trial
        )
    return rt, is_led_trial, t_LED, t_stim

sim_results = Parallel(n_jobs=30)(
    delayed(simulate_single_trial_fit)() for _ in range(N_trials_sim)
)

sim_rts = [r[0] for r in sim_results]
sim_is_led = [r[1] for r in sim_results]
sim_t_LEDs = [r[2] for r in sim_results]
sim_t_stims = [r[3] for r in sim_results]

# Separate into LED ON/OFF, apply truncation AND filter aborts only (RT < t_stim) to match data
sim_rts_on = [rt for rt, is_led, t_stim in zip(sim_rts, sim_is_led, sim_t_stims) if is_led and rt < t_stim]
sim_rts_off = [rt for rt, is_led, t_stim in zip(sim_rts, sim_is_led, sim_t_stims) if not is_led and rt < t_stim]

# For RT wrt LED plots (abort rate), same filtering
sim_rts_wrt_led_on = [rt - t_led for rt, is_led, t_led, t_stim in zip(sim_rts, sim_is_led, sim_t_LEDs, sim_t_stims) if is_led and rt < t_stim]
sim_rts_wrt_led_off = [rt - t_led for rt, is_led, t_led, t_stim in zip(sim_rts, sim_is_led, sim_t_LEDs, sim_t_stims) if not is_led and rt < t_stim]

print(f"Simulation (aborts only): {len(sim_rts_on)} LED ON, {len(sim_rts_off)} LED OFF")

# %%
# =============================================================================
# Get real data RTs for plotting
# =============================================================================
# Aborts only (RT < t_stim) with truncation (RT > T_trunc) for RTD plot
data_rts_on = fit_df[(fit_df['LED_trial'] == 1) & (fit_df['RT'] < fit_df['t_stim'])]['RT'].values
data_rts_off = fit_df[(fit_df['LED_trial'] == 0) & (fit_df['RT'] < fit_df['t_stim'])]['RT'].values

# RT wrt LED for data (same filtering)
df_on_aborts = fit_df[(fit_df['LED_trial'] == 1) & (fit_df['RT'] < fit_df['t_stim'])]
df_off_aborts = fit_df[(fit_df['LED_trial'] == 0) & (fit_df['RT'] < fit_df['t_stim'])]

data_rts_wrt_led_on = (df_on_aborts['RT'] - df_on_aborts['t_LED']).values
# For LED OFF, use actual t_LED from data (same as aborts_animal_wise_explore.py)
data_rts_wrt_led_off = (df_off_aborts['RT'] - df_off_aborts['t_LED']).values

print(f"\nReal data (aborts only): {len(data_rts_on)} LED ON, {len(data_rts_off)} LED OFF")

# %%
# =============================================================================
# Plot 1: RTD wrt fixation (data vs theory vs sim)
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

bins = np.arange(0, t_max, 0.05)
bin_centers = (bins[1:] + bins[:-1]) / 2

# LED ON
data_hist_on, _ = np.histogram(data_rts_on, bins=bins, density=True)
sim_hist_on, _ = np.histogram(sim_rts_on, bins=bins, density=True)

axes[0].plot(bin_centers, data_hist_on, label='Data (aborts)', lw=2, alpha=0.7, color='b')
axes[0].plot(bin_centers, sim_hist_on, label='Sim (fitted)', lw=2, alpha=0.7, color='g')
axes[0].plot(t_pts, pdf_theory_on_fit, label='Theory (fitted)', lw=2, ls='--', color='k')
# axes[0].axvline(x=T_trunc, color='r', linestyle='--', alpha=0.5, label='Truncation')
axes[0].set_xlabel('RT (s)', fontsize=12)
axes[0].set_ylabel('Density', fontsize=12)
axes[0].set_title(f'LED ON Trials - {animal_label}', fontsize=14)
axes[0].legend(fontsize=10)

# LED OFF
data_hist_off, _ = np.histogram(data_rts_off, bins=bins, density=True)
sim_hist_off, _ = np.histogram(sim_rts_off, bins=bins, density=True)

axes[1].plot(bin_centers, data_hist_off, label='Data (aborts)', lw=2, alpha=0.7, color='b')
axes[1].plot(bin_centers, sim_hist_off, label='Sim (fitted)', lw=2, alpha=0.7, color='g')
axes[1].plot(t_pts, pdf_theory_off_fit, label='Theory (fitted)', lw=2, ls='--', color='k')
# axes[1].axvline(x=T_trunc, color='r', linestyle='--', alpha=0.5, label='Truncation')
axes[1].set_xlabel('RT (s)', fontsize=12)
axes[1].set_ylabel('Density', fontsize=12)
axes[1].set_title(f'LED OFF Trials - {animal_label}', fontsize=14)
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig(f'vbmc_real_{file_tag}_rtd_comparison.pdf', bbox_inches='tight')
print(f"RTD comparison saved as 'vbmc_real_{file_tag}_rtd_comparison.pdf'")
plt.show()

# %% 
# =============================================================================
# Plot 1b: RTD wrt fixation (Decomposed components)
# =============================================================================
print("\nComputing theoretical RT components...")
pdf_proactive_on, pdf_lapse_on = compute_theoretical_rtd_on_components(
    t_pts, param_means[0], param_means[1], param_means[2], param_means[3], param_means[4], param_means[5], param_means[6],
    N_mc=5000
)
pdf_proactive_off, pdf_lapse_off = compute_theoretical_rtd_off_components(
    t_pts, param_means[0], param_means[2], param_means[3], param_means[4], param_means[5], param_means[6]
)
# %%
# Normalize total theory to match previous plot normalization logic
# The previous plot did: pdf_theory_on_fit /= np.trapz(pdf_theory_on_fit, t_pts)
# So we calculate total first
pdf_total_on = pdf_proactive_on + pdf_lapse_on
print(f"ON Areas: Proactive={np.trapz(pdf_proactive_on, t_pts):.4f}, Lapse={np.trapz(pdf_lapse_on, t_pts):.4f}, Total={np.trapz(pdf_total_on, t_pts):.4f}")

norm_on = np.trapz(pdf_total_on, t_pts)
if norm_on > 0:
    pdf_proactive_on /= norm_on
    pdf_lapse_on /= norm_on
    pdf_total_on /= norm_on
    print(f"ON Areas (after norm): Proactive={np.trapz(pdf_proactive_on, t_pts):.4f}, Lapse={np.trapz(pdf_lapse_on, t_pts):.4f}, Total={np.trapz(pdf_total_on, t_pts):.4f}")

pdf_total_off = pdf_proactive_off + pdf_lapse_off
print(f"OFF Areas: Proactive={np.trapz(pdf_proactive_off, t_pts):.4f}, Lapse={np.trapz(pdf_lapse_off, t_pts):.4f}, Total={np.trapz(pdf_total_off, t_pts):.4f}")

norm_off = np.trapz(pdf_total_off, t_pts)
if norm_off > 0:
    pdf_proactive_off /= norm_off
    pdf_lapse_off /= norm_off
    pdf_total_off /= norm_off
    print(f"OFF Areas (after norm): Proactive={np.trapz(pdf_proactive_off, t_pts):.4f}, Lapse={np.trapz(pdf_lapse_off, t_pts):.4f}, Total={np.trapz(pdf_total_off, t_pts):.4f}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# LED ON
# axes[0].plot(bin_centers, data_hist_on, label='Data (aborts)', lw=2, alpha=0.3, color='gray')
axes[0].plot(t_pts, pdf_total_on, label='Total Theory', lw=2, color='k')
axes[0].plot(t_pts, pdf_proactive_on, label='Proactive Component', lw=2, ls='--', color='blue')
axes[0].plot(t_pts, pdf_lapse_on, label='Lapse Component', lw=2, ls='--', color='orange')
axes[0].set_xlabel('RT (s)', fontsize=12)
axes[0].set_ylabel('Density', fontsize=12)
axes[0].set_title(f'LED ON Components - {animal_label}', fontsize=14)
axes[0].legend(fontsize=10)

# LED OFF
# axes[1].plot(bin_centers, data_hist_off, label='Data (aborts)', lw=2, alpha=0.3, color='gray')
axes[1].plot(t_pts, pdf_total_off, label='Total Theory', lw=2, color='k')
axes[1].plot(t_pts, pdf_proactive_off, label='Proactive Component', lw=2, ls='--', color='blue')
axes[1].plot(t_pts, pdf_lapse_off, label='Lapse Component', lw=2, ls='--', color='orange')
axes[1].set_xlabel('RT (s)', fontsize=12)
axes[1].set_ylabel('Density', fontsize=12)
axes[1].set_title(f'LED OFF Components - {animal_label}', fontsize=14)
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig(f'vbmc_real_{file_tag}_rtd_components.pdf', bbox_inches='tight')
print(f"RTD components saved as 'vbmc_real_{file_tag}_rtd_components.pdf'")
plt.show()

# %%
# =============================================================================
# Plot 2: RT wrt LED (data vs sim)
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

bins_wrt_led = np.arange(-3, 3, 0.05)
bin_centers_wrt_led = (bins_wrt_led[1:] + bins_wrt_led[:-1]) / 2

# LED ON
data_hist_on_wrt_led, _ = np.histogram(data_rts_wrt_led_on, bins=bins_wrt_led, density=True)
sim_hist_on_wrt_led, _ = np.histogram(sim_rts_wrt_led_on, bins=bins_wrt_led, density=True)

axes[0].plot(bin_centers_wrt_led, data_hist_on_wrt_led, label='Data (aborts)', lw=2, alpha=0.7, color='b')
axes[0].plot(bin_centers_wrt_led, sim_hist_on_wrt_led, label='Sim (fitted)', lw=2, alpha=0.7, color='r')
axes[0].axvline(x=0, color='k', linestyle='--', alpha=0.5, label='LED onset')
axes[0].axvline(x=param_means[4], color='r', linestyle=':', alpha=0.5, label=f'del_m_plus_del_LED={param_means[4]:.2f}')
axes[0].set_xlabel('RT - t_LED (s)', fontsize=12)
axes[0].set_ylabel('Density', fontsize=12)
axes[0].set_title(f'LED ON Trials - {animal_label}', fontsize=14)
axes[0].legend(fontsize=10)

# LED OFF
data_hist_off_wrt_led, _ = np.histogram(data_rts_wrt_led_off, bins=bins_wrt_led, density=True)
sim_hist_off_wrt_led, _ = np.histogram(sim_rts_wrt_led_off, bins=bins_wrt_led, density=True)

axes[1].plot(bin_centers_wrt_led, data_hist_off_wrt_led, label='Data (aborts)', lw=2, alpha=0.7, color='b')
axes[1].plot(bin_centers_wrt_led, sim_hist_off_wrt_led, label='Sim (fitted)', lw=2, alpha=0.7, color='r')
axes[1].axvline(x=0, color='k', linestyle='--', alpha=0.5, label='LED onset')
axes[1].set_xlabel('RT - t_LED (s)', fontsize=12)
axes[1].set_ylabel('Density', fontsize=12)
axes[1].set_title(f'LED OFF Trials - {animal_label}', fontsize=14)
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig(f'vbmc_real_{file_tag}_rt_wrt_led_comparison.pdf', bbox_inches='tight')
print(f"RT wrt LED comparison saved as 'vbmc_real_{file_tag}_rt_wrt_led_comparison.pdf'")
plt.show()

# %%
# =============================================================================
# Theoretical RTD wrt LED (Monte Carlo over trial timings)
# =============================================================================
N_mc_theory_wrt_led = 5000
t_pts_wrt_led_theory = np.arange(-2, 2, 0.005)

print("\nComputing theoretical RT distributions wrt LED...")
rtd_theory_on_wrt_led = np.zeros(len(t_pts_wrt_led_theory))
rtd_theory_off_wrt_led = np.zeros(len(t_pts_wrt_led_theory))

for i in tqdm(range(N_mc_theory_wrt_led)):
    trial_idx = np.random.randint(n_trials_data)
    t_led = LED_times[trial_idx]
    t_stim = stim_times[trial_idx]

    if t_stim <= 0: # minimal check
        continue

    t_pts_wrt_fix = t_pts_wrt_led_theory + t_led

    # Pre-compute LED ON truncation factor for this t_led
    # cdf_pts = np.arange(0, T_trunc + 0.001, 0.001)
    # cdf_vals = np.array([
    #     PA_with_LEDON_2_adapted(ti, param_means[0], param_means[1], param_means[2],
    #                             param_means[3], param_means[4], t_led, None)
    #     for ti in cdf_pts
    # ])
    # cdf_trunc_on = np.trapz(cdf_vals, cdf_pts)
    # trunc_factor_on = 1 - cdf_trunc_on

    mask_on = (t_pts_wrt_fix > 0) & (t_pts_wrt_fix < t_stim)
    if np.any(mask_on):
        proactive_vals = np.array(
            [
                PA_with_LEDON_2_adapted(
                    t_wrt_fix,
                    param_means[0],
                    param_means[1],
                    param_means[2],
                    param_means[3],
                    param_means[4],
                    t_led
                )
                for t_wrt_fix in t_pts_wrt_fix[mask_on]
            ]
        )
        lapse_vals = lapse_pdf(t_pts_wrt_fix[mask_on], param_means[6])
        rtd_theory_on_wrt_led[mask_on] += (1 - param_means[5]) * proactive_vals + param_means[5] * lapse_vals

    mask_off = (t_pts_wrt_fix > 0) & (t_pts_wrt_fix < t_stim)
    if np.any(mask_off):
        proactive_vals = np.array(
            [
                led_off_pdf(
                    t_wrt_fix, param_means[0], param_means[2], param_means[3], param_means[4]
                )
                for t_wrt_fix in t_pts_wrt_fix[mask_off]
            ]
        )
        lapse_vals = lapse_pdf(t_pts_wrt_fix[mask_off], param_means[6])
        rtd_theory_off_wrt_led[mask_off] += (1 - param_means[5]) * proactive_vals + param_means[5] * lapse_vals

rtd_theory_on_wrt_led /= N_mc_theory_wrt_led
rtd_theory_off_wrt_led /= N_mc_theory_wrt_led

print(f"  Theory wrt LED area ON={np.trapz(rtd_theory_on_wrt_led, t_pts_wrt_led_theory):.4f}, "
      f"OFF={np.trapz(rtd_theory_off_wrt_led, t_pts_wrt_led_theory):.4f}")

# %%
# =============================================================================
# Plot 3: RT wrt LED - Abort rate (like aborts_animal_wise_explore.py)
# =============================================================================
# Compute abort rate = n_aborts / n_all_trials for each LED condition (from data)
bins_wrt_led = np.arange(-3, 3, 0.005)
bin_centers_wrt_led = (bins_wrt_led[1:] + bins_wrt_led[:-1]) / 2

n_all_data_on = len(fit_df[fit_df['LED_trial'] == 1])
n_all_data_off = len(fit_df[fit_df['LED_trial'] == 0])
n_aborts_data_on = len(data_rts_wrt_led_on)
n_aborts_data_off = len(data_rts_wrt_led_off)
frac_data_on = n_aborts_data_on / n_all_data_on if n_all_data_on > 0 else 0
frac_data_off = n_aborts_data_off / n_all_data_off if n_all_data_off > 0 else 0

# Compute abort rate for sim (aborts = RT < t_stim, after truncation)
n_all_sim_on = sum(1 for is_led in sim_is_led if is_led)
n_all_sim_off = sum(1 for is_led in sim_is_led if not is_led)
n_aborts_sim_on = sum(1 for rt, is_led, t_stim in zip(sim_rts, sim_is_led, sim_t_stims) if is_led and rt < t_stim)
n_aborts_sim_off = sum(1 for rt, is_led, t_stim in zip(sim_rts, sim_is_led, sim_t_stims) if not is_led and rt < t_stim)
frac_sim_on = n_aborts_sim_on / n_all_sim_on if n_all_sim_on > 0 else 0
frac_sim_off = n_aborts_sim_off / n_all_sim_off if n_all_sim_off > 0 else 0

# Create histograms with density=True then scale by fraction
fig, ax = plt.subplots(figsize=(15, 6))

# Data histograms (area = fraction)
data_hist_on_wrt_led_dens, _ = np.histogram(data_rts_wrt_led_on, bins=bins_wrt_led, density=True)
data_hist_off_wrt_led_dens, _ = np.histogram(data_rts_wrt_led_off, bins=bins_wrt_led, density=True)
data_hist_on_scaled = data_hist_on_wrt_led_dens * frac_data_on
data_hist_off_scaled = data_hist_off_wrt_led_dens * frac_data_off

# Sim histograms (area = fraction)
sim_hist_on_wrt_led_dens, _ = np.histogram(sim_rts_wrt_led_on, bins=bins_wrt_led, density=True)
sim_hist_off_wrt_led_dens, _ = np.histogram(sim_rts_wrt_led_off, bins=bins_wrt_led, density=True)
sim_hist_on_scaled = sim_hist_on_wrt_led_dens * frac_sim_on
sim_hist_off_scaled = sim_hist_off_wrt_led_dens * frac_sim_off

# Plot all on same axes
ax.plot(bin_centers_wrt_led, data_hist_on_scaled, label=f'Data LED ON (frac={frac_data_on:.2f})', lw=2, alpha=0.7, color='r', linestyle='-')
ax.plot(bin_centers_wrt_led, data_hist_off_scaled, label=f'Data LED OFF (frac={frac_data_off:.2f})', lw=2, alpha=0.7, color='b', linestyle='-')
ax.plot(bin_centers_wrt_led, sim_hist_on_scaled, label=f'Sim LED ON (frac={frac_sim_on:.2f})', lw=2, alpha=0.7, color='r', linestyle='--')
ax.plot(bin_centers_wrt_led, sim_hist_off_scaled, label=f'Sim LED OFF (frac={frac_sim_off:.2f})', lw=2, alpha=0.7, color='b', linestyle='--')
ax.plot(t_pts_wrt_led_theory, rtd_theory_on_wrt_led, label='Theory LED ON', lw=2, alpha=0.7, color='r', linestyle=':')
ax.plot(t_pts_wrt_led_theory, rtd_theory_off_wrt_led, label='Theory LED OFF', lw=2, alpha=0.7, color='b', linestyle=':')

ax.axvline(x=0, color='k', linestyle='--', alpha=0.5, label='LED onset')
ax.axvline(x=param_means[4], color='g', linestyle=':', alpha=0.5, label=f'del_m_plus_del_LED={param_means[4]:.2f}')
ax.set_xlabel('RT - t_LED (s)', fontsize=12)
ax.set_ylabel('Rate (area = fraction)', fontsize=12)
ax.set_title(f'RT wrt LED (area-weighted) - {animal_label}', fontsize=14)
ax.legend(fontsize=9)

ax.set_xlim(-0.5,0.4)
# ax.set_xlim(-0.5, 0.4)

plt.tight_layout()
plt.savefig(f'vbmc_real_{file_tag}_rt_wrt_led_rate.pdf', bbox_inches='tight')
print(f"RT wrt LED rate plot saved as 'vbmc_real_{file_tag}_rt_wrt_led_rate.pdf'")
plt.show()

print(f"\nScript complete for {animal_label}!")

# %%
