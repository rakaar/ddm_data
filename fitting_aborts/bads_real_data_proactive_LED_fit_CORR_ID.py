"""
BADS Fitting of Real Animal Data with Proactive LED Model (Correctly Identified Parameters)
============================================================================================

This script fits the proactive process model to real experimental data from a single animal
using Bayesian Adaptive Direct Search (BADS).

PARAMETERS (5 identifiable parameters):
- V_A_base: Base drift rate of proactive accumulator
- V_A_post_LED: Drift rate after LED effect kicks in
- theta_A: Threshold for proactive accumulator
- del_a_minus_del_LED: Composite delay = del_a - del_LED (afferent delay minus LED effect delay)
- del_m_plus_del_LED: Composite delay = del_m + del_LED (motor delay plus LED effect delay)

SCRIPT STRUCTURE (cell-by-cell):
--------------------------------
1. **Imports** - numpy, matplotlib, joblib, pandas, pybads
2. **Parameters** - ANIMAL_IDX to select which animal to fit
3. **Load & Filter Data** - Load real data, filter by session/training, select animal
4. **Build Fitting DataFrame** - RT (timed_fix), t_stim (intended_fix), t_LED, LED_trial
5. **Likelihood Functions** - Same as VBMC version (truncation + censoring)
6. **BADS Setup** - Define bounds, plausible bounds, neg-loglikelihood objective
7. **Run BADS** - Optimize and get best-fit parameters
8. **Diagnostic Plots**:
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
from pybads import BADS
import pickle

# %%
# =============================================================================
# PARAMETERS - Change ANIMAL_IDX to fit different animals
# =============================================================================
ANIMAL_IDX = 5  # Index into unique_animals array (0, 1, 2, ...)
T_trunc = 0.3   # Left truncation threshold (exclude RT <= T_trunc)

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
df = df[~((df['abort_event'] == 3) & (df['timed_fix'] < T_trunc))]

# Get unique animals
unique_animals = df['animal'].unique()
print(f"Available animals: {unique_animals}")
print(f"Selected ANIMAL_IDX = {ANIMAL_IDX} -> Animal: {unique_animals[ANIMAL_IDX]}")

# Select animal
animal = unique_animals[ANIMAL_IDX]
df_animal = df[df['animal'] == animal]

# Separate LED ON and OFF
df_on = df_animal[df_animal['LED_trial'] == 1]
df_off = df_animal[df_animal['LED_trial'].isin([0, np.nan])]

print(f"\nAnimal {animal} data summary:")
print(f"  Total trials: {len(df_animal)}")
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
fit_df = fit_df[~((fit_df['RT'] < fit_df['t_stim']) & (fit_df['RT'] <= T_trunc))]

print(f"\nFitting DataFrame summary (after truncation):")
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
stim_times = df_animal['intended_fix'].values
LED_times = (df_animal['intended_fix'] - df_animal['LED_onset_time']).values
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
def PA_with_LEDON_2_adapted(t, v, vON, a, del_a_minus_del_LED, del_m_plus_del_LED, tled, T_trunc=None):
    """
    Compute the PA pdf by combining contributions before and after LED onset.
    """
    if T_trunc is not None and t <= T_trunc:
        return 0

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

    if T_trunc is not None:
        t_pts = np.arange(0, T_trunc + 0.001, 0.001)
        pdf_vals = np.array([
            PA_with_LEDON_2_adapted(ti, v, vON, a, del_a_minus_del_LED, del_m_plus_del_LED, tled, None)
            for ti in t_pts
        ])
        cdf_trunc = np.trapz(pdf_vals, t_pts)
        pdf = pdf / (1 - cdf_trunc)

    return pdf


def led_off_cdf(t, v, a, del_a_minus_del_LED, del_m_plus_del_LED):
    if t <= del_m_plus_del_LED + del_a_minus_del_LED:
        return 0
    return cum_A_t_fn(t - (del_m_plus_del_LED + del_a_minus_del_LED), v, a)


def led_off_pdf_truncated(t, v, a, del_a_minus_del_LED, del_m_plus_del_LED, T_trunc):
    if t <= T_trunc:
        return 0

    if t <= del_m_plus_del_LED + del_a_minus_del_LED:
        return 0

    pdf = d_A_RT(v * a, (t - (del_m_plus_del_LED + del_a_minus_del_LED)) / (a**2)) / (a**2)

    if T_trunc is not None:
        cdf_trunc = led_off_cdf(T_trunc, v, a, del_a_minus_del_LED, del_m_plus_del_LED)
        trunc_factor = 1 - cdf_trunc
        if trunc_factor <= 0:
            return 0
        pdf = pdf / trunc_factor

    return pdf


def led_off_survival_truncated(t_stim, v, a, del_a_minus_del_LED, del_m_plus_del_LED, T_trunc):
    if t_stim <= T_trunc:
        return 1.0

    cdf_t_stim = led_off_cdf(t_stim, v, a, del_a_minus_del_LED, del_m_plus_del_LED)
    cdf_T_trunc = led_off_cdf(T_trunc, v, a, del_a_minus_del_LED, del_m_plus_del_LED)
    trunc_factor = 1 - cdf_T_trunc
    if trunc_factor <= 0:
        return 1.0

    return (1 - cdf_t_stim) / trunc_factor


def led_on_survival_truncated(t_stim, t_led, v, vON, a, del_a_minus_del_LED, del_m_plus_del_LED, T_trunc):
    if t_stim <= T_trunc:
        return 1.0

    t_pts_cdf = np.arange(0, t_stim + 0.001, 0.001)
    pdf_vals_cdf = np.array([
        PA_with_LEDON_2_adapted(ti, v, vON, a, del_a_minus_del_LED, del_m_plus_del_LED, t_led, None)
        for ti in t_pts_cdf
    ])
    cdf_t_stim = np.trapz(pdf_vals_cdf, t_pts_cdf)

    t_pts_trunc = np.arange(0, T_trunc + 0.001, 0.001)
    pdf_vals_trunc = np.array([
        PA_with_LEDON_2_adapted(ti, v, vON, a, del_a_minus_del_LED, del_m_plus_del_LED, t_led, None)
        for ti in t_pts_trunc
    ])
    cdf_T_trunc = np.trapz(pdf_vals_trunc, t_pts_trunc)

    return (1 - cdf_t_stim) / (1 - cdf_T_trunc)


def compute_trial_loglike(row, V_A_base, V_A_post_LED, theta_A, del_a_minus_del_LED, del_m_plus_del_LED, T_trunc):
    t = row['RT']
    t_stim = row['t_stim']
    is_led = row['LED_trial'] == 1
    t_led = row['t_LED']

    if is_led and (t_led is None or (isinstance(t_led, float) and np.isnan(t_led))):
        raise ValueError("LED trial has invalid t_LED (None/NaN).")

    if (t <= T_trunc) and (t < t_stim):
        return np.log(1e-50)

    if (t_stim <= T_trunc):
        if t < t_stim:
            return np.log(1e-50)
        else:
            return np.log(1.0)

    if t < t_stim:
        if is_led:
            likelihood = PA_with_LEDON_2_adapted(
                t, V_A_base, V_A_post_LED, theta_A, del_a_minus_del_LED, del_m_plus_del_LED, t_led, T_trunc
            )
        else:
            likelihood = led_off_pdf_truncated(t, V_A_base, theta_A, del_a_minus_del_LED, del_m_plus_del_LED, T_trunc)
    else:
        if is_led:
            likelihood = led_on_survival_truncated(
                t_stim, t_led, V_A_base, V_A_post_LED, theta_A, del_a_minus_del_LED, del_m_plus_del_LED, T_trunc
            )
        else:
            likelihood = led_off_survival_truncated(t_stim, V_A_base, theta_A, del_a_minus_del_LED, del_m_plus_del_LED, T_trunc)

    if likelihood <= 0 or np.isnan(likelihood):
        likelihood = 1e-50

    return np.log(likelihood)


def proactive_led_loglike(params):
    V_A_base, V_A_post_LED, theta_A, del_a_minus_del_LED, del_m_plus_del_LED = params
    all_loglike = Parallel(n_jobs=30)(
        delayed(compute_trial_loglike)(
            row, V_A_base, V_A_post_LED, theta_A, del_a_minus_del_LED, del_m_plus_del_LED, T_trunc
        ) for _, row in fit_df.iterrows()
    )
    return np.sum(all_loglike)

# %%
# =============================================================================
# Objective function for BADS (minimize negative log-likelihood)
# =============================================================================
def neg_loglike(params):
    return -proactive_led_loglike(params)

# %%
# =============================================================================
# Bounds
# =============================================================================
V_A_base_bounds = [0.1, 8]
V_A_post_LED_bounds = [0.1, 8]
theta_A_bounds = [0.1, 8]
del_a_minus_del_LED_bounds = [-1.1, 1.1]
del_m_plus_del_LED_bounds = [0.001, 0.2]

V_A_base_plausible = [0.5, 3]
V_A_post_LED_plausible = [0.5, 3]
theta_A_plausible = [0.5, 3]
del_a_minus_del_LED_plausible = [0.01, 0.07]
del_m_plus_del_LED_plausible = [0.01, 0.07]

# %%
# =============================================================================
# Set up BADS
# =============================================================================
lb = np.array([V_A_base_bounds[0], V_A_post_LED_bounds[0], theta_A_bounds[0], 
               del_a_minus_del_LED_bounds[0], del_m_plus_del_LED_bounds[0]])
ub = np.array([V_A_base_bounds[1], V_A_post_LED_bounds[1], theta_A_bounds[1], 
               del_a_minus_del_LED_bounds[1], del_m_plus_del_LED_bounds[1]])
plb = np.array([V_A_base_plausible[0], V_A_post_LED_plausible[0], theta_A_plausible[0], 
                del_a_minus_del_LED_plausible[0], del_m_plus_del_LED_plausible[0]])
pub = np.array([V_A_base_plausible[1], V_A_post_LED_plausible[1], theta_A_plausible[1], 
                del_a_minus_del_LED_plausible[1], del_m_plus_del_LED_plausible[1]])

# Initial point (use values similar to simulated ground truth)
np.random.seed(55)
x_0 = np.array([
    np.random.uniform(V_A_base_plausible[0], V_A_base_plausible[1]),   # V_A_base
    np.random.uniform(V_A_post_LED_plausible[0], V_A_post_LED_plausible[1]),   # V_A_post_LED
    np.random.uniform(theta_A_plausible[0], theta_A_plausible[1]),   # theta_A
   np.random.uniform(del_a_minus_del_LED_plausible[0], del_a_minus_del_LED_plausible[1]),
   np.random.uniform(del_m_plus_del_LED_plausible[0], del_m_plus_del_LED_plausible[1])
])

print("\nInitial parameter values:")
print(f"  V_A_base: {x_0[0]:.3f}")
print(f"  V_A_post_LED: {x_0[1]:.3f}")
print(f"  theta_A: {x_0[2]:.3f}")
print(f"  del_a_minus_del_LED: {x_0[3]:.3f}")
print(f"  del_m_plus_del_LED: {x_0[4]:.3f}")

# Ensure x_0 is within plausible bounds
x_0 = np.clip(x_0, plb, pub)

print("\nBADS setup:")
print(f"  Initial point: {x_0}")
print(f"  Lower bounds: {lb}")
print(f"  Upper bounds: {ub}")
print(f"  Plausible lower: {plb}")
print(f"  Plausible upper: {pub}")

# %%
# =============================================================================
# Run BADS (or load saved results)
# =============================================================================
LOAD_SAVED_RESULTS = False
RESULTS_PKL_PATH = f'bads_real_{animal}_CORR_ID_results.pkl'

if LOAD_SAVED_RESULTS and os.path.exists(RESULTS_PKL_PATH):
    print(f"\nLoading saved BADS results from {RESULTS_PKL_PATH}...")
    with open(RESULTS_PKL_PATH, 'rb') as f:
        saved_results = pickle.load(f)
    best_params = saved_results['best_params']
    best_fval = saved_results['best_fval']
    print("Loaded saved BADS results.")
else:
    print(f"\nRunning BADS optimization for animal {animal}...")
    bads = BADS(neg_loglike, x_0, lb, ub, plb, pub)
    optimize_result = bads.optimize()

    best_params = optimize_result['x']
    best_fval = optimize_result['fval']

    print("\nBADS optimization complete!")

    # =============================================================================
    # Save results
    # =============================================================================
    with open(RESULTS_PKL_PATH, 'wb') as f:
        pickle.dump({
            'animal': animal,
            'best_params': best_params,
            'best_fval': best_fval,
            'fit_df': fit_df,
            'bounds': {
                'lb': lb, 'ub': ub, 'plb': plb, 'pub': pub
            }
        }, f)
    print(f"Results saved to {RESULTS_PKL_PATH}")

# %%
# =============================================================================
# Print best-fit parameters
# =============================================================================
param_labels = ['V_A_base', 'V_A_post_LED', 'theta_A', 'del_a_minus_del_LED', 'del_m_plus_del_LED']

print(f"\nBest-fit parameters for animal {animal}:")
print(f"{'Parameter':<25} {'Value':<12}")
print("-" * 37)
for i, label in enumerate(param_labels):
    print(f"{label:<25} {best_params[i]:<12.4f}")
print(f"\nNeg log-likelihood: {best_fval:.4f}")
print(f"Log-likelihood: {-best_fval:.4f}")

# %%
# =============================================================================
# Theoretical RTD calculation (conditioned on RT < t_stim, i.e., aborts only)
# P(RT=t | abort) = P(RT=t) * P(t_stim > t) / P(abort)
# =============================================================================
def compute_theoretical_rtd_on(t_pts, V_A_base, V_A_post_LED, theta_A, del_a_minus_del_LED, del_m_plus_del_LED,
                               N_mc=5000, T_trunc=None):
    """Compute theoretical RTD for LED ON, conditioned on RT < t_stim (aborts).
    
    Monte Carlo loop over t_led and t_stim samples, applying same conditions as likelihood.
    """
    pdf_samples = np.zeros((N_mc, len(t_pts)))
    
    for i in range(N_mc):
        # Sample trial index to preserve (t_LED, t_stim) correlation
        trial_idx = np.random.randint(n_trials_data)
        t_led = LED_times[trial_idx]
        t_stim = stim_times[trial_idx]
        
        # Skip if t_stim <= T_trunc (no valid abort region)
        if t_stim <= T_trunc:
            continue
        
        # Compute PDF for each t, applying conditions from likelihood
        sample_pdf = np.zeros(len(t_pts))
        for j, t in enumerate(t_pts):
            # Condition 1: t <= T_trunc and t < t_stim -> 0
            if (t <= T_trunc) and (t < t_stim):
                sample_pdf[j] = 0
            # Condition 2: t >= t_stim -> 0 (not an abort)
            elif t >= t_stim:
                sample_pdf[j] = 0
            # Valid abort: T_trunc < t < t_stim
            else:
                sample_pdf[j] = PA_with_LEDON_2_adapted(
                    t, V_A_base, V_A_post_LED, theta_A, del_a_minus_del_LED, del_m_plus_del_LED, t_led, T_trunc
                )
        
        # Normalize this sample's PDF to integrate to 1 over valid region
        norm = np.trapz(sample_pdf, t_pts)
        if norm > 0:
            # pdf_samples[i, :] = sample_pdf / norm
            pdf_samples[i, :] = sample_pdf

    
    # Average over all MC samples
    return np.mean(pdf_samples, axis=0)


def compute_theoretical_rtd_off(t_pts, V_A_base, theta_A, del_a_minus_del_LED, del_m_plus_del_LED, T_trunc=None, N_mc=5000):
    """Compute theoretical RTD for LED OFF, conditioned on RT < t_stim (aborts).
    
    Monte Carlo loop over t_stim samples, applying same conditions as likelihood.
    """
    pdf_samples = np.zeros((N_mc, len(t_pts)))
    
    for i in range(N_mc):
        # Sample trial index (for consistency, even though LED OFF doesn't use t_led)
        trial_idx = np.random.randint(n_trials_data)
        t_stim = stim_times[trial_idx]
        
        # Skip if t_stim <= T_trunc (no valid abort region)
        if t_stim <= T_trunc:
            continue
        
        # Compute PDF for each t, applying conditions from likelihood
        sample_pdf = np.zeros(len(t_pts))
        for j, t in enumerate(t_pts):
            # Condition 1: t <= T_trunc and t < t_stim -> 0
            if (t <= T_trunc) and (t < t_stim):
                sample_pdf[j] = 0
            # Condition 2: t >= t_stim -> 0 (not an abort)
            elif t >= t_stim:
                sample_pdf[j] = 0
            # Valid abort: T_trunc < t < t_stim
            else:
                sample_pdf[j] = led_off_pdf_truncated(t, V_A_base, theta_A, del_a_minus_del_LED, del_m_plus_del_LED, T_trunc)
        
        # Normalize this sample's PDF to integrate to 1 over valid region
        norm = np.trapz(sample_pdf, t_pts)
        if norm > 0:
            # pdf_samples[i, :] = sample_pdf / norm
            pdf_samples[i, :] = sample_pdf

    
    # Average over all MC samples
    return np.mean(pdf_samples, axis=0)

# Compute theoretical distributions with fitted parameters
dt = 0.01
t_max = 3
t_pts = np.arange(T_trunc, t_max, dt)

print("\nComputing theoretical RT distributions with fitted parameters...")
pdf_theory_on_fit = compute_theoretical_rtd_on(
    t_pts, best_params[0], best_params[1], best_params[2], best_params[3], best_params[4],
    N_mc=5000, T_trunc=T_trunc
)
pdf_theory_off_fit = compute_theoretical_rtd_off(
    t_pts, best_params[0], best_params[2], best_params[3], best_params[4], T_trunc=T_trunc
)

# Normalize
pdf_theory_on_fit /= np.trapz(pdf_theory_on_fit, t_pts)
pdf_theory_off_fit /= np.trapz(pdf_theory_off_fit, t_pts)

# %%
# =============================================================================
# Simulate with fitted parameters for comparison
# =============================================================================
N_trials_sim = int(200e3)
print(f"\nSimulating {N_trials_sim} trials with fitted parameters...")

def simulate_single_trial_fit():
    is_led_trial = np.random.random() < 1/3
    # Sample trial index to preserve (t_LED, t_stim) correlation
    trial_idx = np.random.randint(n_trials_data)
    t_LED = LED_times[trial_idx]
    t_stim = stim_times[trial_idx]
    rt = simulate_proactive_single_bound(
        best_params[0], best_params[1], best_params[2],  # V_A_base, V_A_post_LED, theta_A
        t_LED if is_led_trial else None, t_stim,
        best_params[3], best_params[4],  # del_a_minus_del_LED, del_m_plus_del_LED
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
sim_rts_on = [rt for rt, is_led, t_stim in zip(sim_rts, sim_is_led, sim_t_stims) if is_led and rt > T_trunc and rt < t_stim]
sim_rts_off = [rt for rt, is_led, t_stim in zip(sim_rts, sim_is_led, sim_t_stims) if not is_led and rt > T_trunc and rt < t_stim]

# For RT wrt LED plots (abort rate), same filtering
sim_rts_wrt_led_on = [rt - t_led for rt, is_led, t_led, t_stim in zip(sim_rts, sim_is_led, sim_t_LEDs, sim_t_stims) if is_led and rt > T_trunc and rt < t_stim]
sim_rts_wrt_led_off = [rt - t_led for rt, is_led, t_led, t_stim in zip(sim_rts, sim_is_led, sim_t_LEDs, sim_t_stims) if not is_led and rt > T_trunc and rt < t_stim]

print(f"Simulation (aborts only): {len(sim_rts_on)} LED ON, {len(sim_rts_off)} LED OFF")

# %%
# =============================================================================
# Get real data RTs for plotting
# =============================================================================
# Aborts only (RT < t_stim) with truncation (RT > T_trunc) for RTD plot
data_rts_on = fit_df[(fit_df['LED_trial'] == 1) & (fit_df['RT'] < fit_df['t_stim']) & (fit_df['RT'] > T_trunc)]['RT'].values
data_rts_off = fit_df[(fit_df['LED_trial'] == 0) & (fit_df['RT'] < fit_df['t_stim']) & (fit_df['RT'] > T_trunc)]['RT'].values

# RT wrt LED for data (same filtering)
df_on_aborts = fit_df[(fit_df['LED_trial'] == 1) & (fit_df['RT'] < fit_df['t_stim']) & (fit_df['RT'] > T_trunc)]
df_off_aborts = fit_df[(fit_df['LED_trial'] == 0) & (fit_df['RT'] < fit_df['t_stim']) & (fit_df['RT'] > T_trunc)]

data_rts_wrt_led_on = (df_on_aborts['RT'] - df_on_aborts['t_LED']).values
# For LED OFF, use actual t_LED from data (same as aborts_animal_wise_explore.py)
data_rts_wrt_led_off = (df_off_aborts['RT'] - df_off_aborts['t_LED']).values

print(f"\nReal data (aborts only): {len(data_rts_on)} LED ON, {len(data_rts_off)} LED OFF")

# %%
# =============================================================================
# Plot 1: RTD wrt fixation (data vs theory vs sim)
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

bins = np.arange(T_trunc, t_max, 0.05)
bin_centers = (bins[1:] + bins[:-1]) / 2

# LED ON
data_hist_on, _ = np.histogram(data_rts_on, bins=bins, density=True)
sim_hist_on, _ = np.histogram(sim_rts_on, bins=bins, density=True)

axes[0].plot(bin_centers, data_hist_on, label='Data (aborts)', lw=2, alpha=0.7, color='b')
axes[0].plot(bin_centers, sim_hist_on, label='Sim (fitted)', lw=2, alpha=0.7, color='g')
axes[0].plot(t_pts, pdf_theory_on_fit, label='Theory (fitted)', lw=2, ls='--', color='k')
axes[0].axvline(x=T_trunc, color='r', linestyle='--', alpha=0.5, label='Truncation')
axes[0].set_xlabel('RT (s)', fontsize=12)
axes[0].set_ylabel('Density', fontsize=12)
axes[0].set_title(f'LED ON Trials - Animal {animal}', fontsize=14)
axes[0].legend(fontsize=10)

# LED OFF
data_hist_off, _ = np.histogram(data_rts_off, bins=bins, density=True)
sim_hist_off, _ = np.histogram(sim_rts_off, bins=bins, density=True)

axes[1].plot(bin_centers, data_hist_off, label='Data (aborts)', lw=2, alpha=0.7, color='b')
axes[1].plot(bin_centers, sim_hist_off, label='Sim (fitted)', lw=2, alpha=0.7, color='g')
axes[1].plot(t_pts, pdf_theory_off_fit, label='Theory (fitted)', lw=2, ls='--', color='k')
axes[1].axvline(x=T_trunc, color='r', linestyle='--', alpha=0.5, label='Truncation')
axes[1].set_xlabel('RT (s)', fontsize=12)
axes[1].set_ylabel('Density', fontsize=12)
axes[1].set_title(f'LED OFF Trials - Animal {animal}', fontsize=14)
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig(f'bads_real_{animal}_CORR_ID_rtd_comparison.pdf', bbox_inches='tight')
print(f"RTD comparison saved as 'bads_real_{animal}_CORR_ID_rtd_comparison.pdf'")
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
axes[0].axvline(x=best_params[4], color='r', linestyle=':', alpha=0.5, label=f'del_m_plus_del_LED={best_params[4]:.2f}')
axes[0].set_xlabel('RT - t_LED (s)', fontsize=12)
axes[0].set_ylabel('Density', fontsize=12)
axes[0].set_title(f'LED ON Trials - Animal {animal}', fontsize=14)
axes[0].legend(fontsize=10)

# LED OFF
data_hist_off_wrt_led, _ = np.histogram(data_rts_wrt_led_off, bins=bins_wrt_led, density=True)
sim_hist_off_wrt_led, _ = np.histogram(sim_rts_wrt_led_off, bins=bins_wrt_led, density=True)

axes[1].plot(bin_centers_wrt_led, data_hist_off_wrt_led, label='Data (aborts)', lw=2, alpha=0.7, color='b')
axes[1].plot(bin_centers_wrt_led, sim_hist_off_wrt_led, label='Sim (fitted)', lw=2, alpha=0.7, color='r')
axes[1].axvline(x=0, color='k', linestyle='--', alpha=0.5, label='LED onset')
axes[1].set_xlabel('RT - t_LED (s)', fontsize=12)
axes[1].set_ylabel('Density', fontsize=12)
axes[1].set_title(f'LED OFF Trials - Animal {animal}', fontsize=14)
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig(f'bads_real_{animal}_CORR_ID_rt_wrt_led_comparison.pdf', bbox_inches='tight')
print(f"RT wrt LED comparison saved as 'bads_real_{animal}_CORR_ID_rt_wrt_led_comparison.pdf'")
plt.show()

# %%
# =============================================================================
# Plot 3: RT wrt LED - Abort rate (like aborts_animal_wise_explore.py)
# =============================================================================
# Compute abort rate = n_aborts / n_all_trials for each LED condition (from data)
bins_wrt_led = np.arange(-3, 3, 0.05)
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
n_aborts_sim_on = sum(1 for rt, is_led, t_stim in zip(sim_rts, sim_is_led, sim_t_stims) if is_led and rt < t_stim and rt > T_trunc)
n_aborts_sim_off = sum(1 for rt, is_led, t_stim in zip(sim_rts, sim_is_led, sim_t_stims) if not is_led and rt < t_stim and rt > T_trunc)
frac_sim_on = n_aborts_sim_on / n_all_sim_on if n_all_sim_on > 0 else 0
frac_sim_off = n_aborts_sim_off / n_all_sim_off if n_all_sim_off > 0 else 0

# Create histograms with density=True then scale by fraction
fig, ax = plt.subplots(figsize=(10, 6))

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
ax.plot(bin_centers_wrt_led, data_hist_on_scaled, label=f'Data LED ON (frac={frac_data_on:.4f})', lw=2, alpha=0.4, color='r', linestyle='-')
ax.plot(bin_centers_wrt_led, data_hist_off_scaled, label=f'Data LED OFF (frac={frac_data_off:.4f})', lw=2, alpha=0.4, color='b', linestyle='-')
ax.plot(bin_centers_wrt_led, sim_hist_on_scaled, label=f'Sim LED ON (frac={frac_sim_on:.4f})', lw=2, alpha=0.7, color='r', linestyle='--')
ax.plot(bin_centers_wrt_led, sim_hist_off_scaled, label=f'Sim LED OFF (frac={frac_sim_off:.4f})', lw=2, alpha=0.7, color='b', linestyle='--')
ax.set_xlim(-1,1)
ax.axvline(x=0, color='k', linestyle='--', alpha=0.5, label='LED onset')
ax.axvline(x=best_params[4], color='g', linestyle=':', alpha=0.5, label=f'del_m_plus_del_LED={best_params[4]:.2f}')
ax.set_xlabel('RT - t_LED (s)', fontsize=12)
ax.set_ylabel('Rate (area = fraction)', fontsize=12)
ax.set_title(f'RT wrt LED (area-weighted) - Animal {animal}', fontsize=14)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f'bads_real_{animal}_CORR_ID_rt_wrt_led_rate.pdf', bbox_inches='tight')
print(f"RT wrt LED rate plot saved as 'bads_real_{animal}_CORR_ID_rt_wrt_led_rate.pdf'")
plt.show()

print(f"\nScript complete for animal {animal}!")
# %%
print(f"Data abort fraction ON: {frac_data_on:.4f}")
print(f"Sim abort fraction ON: {frac_sim_on:.4f}")
# %%
