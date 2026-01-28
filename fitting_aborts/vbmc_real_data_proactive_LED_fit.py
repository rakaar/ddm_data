"""
VBMC Fitting of Real Animal Data with Proactive LED Model
==========================================================

This script fits the proactive process model to real experimental data from a single animal
using Variational Bayesian Monte Carlo (VBMC).

SCRIPT STRUCTURE (cell-by-cell):
--------------------------------
1. **Imports** - numpy, matplotlib, joblib, pandas, pyvbmc, corner
2. **Parameters** - ANIMAL_IDX to select which animal to fit
3. **Load & Filter Data** - Load real data, filter by session/training, select animal
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
import sys
sys.path.append('../fit_each_condn')
from psiam_tied_dv_map_utils_with_PDFs import stupid_f_integral, d_A_RT
from post_LED_censor_utils import cum_A_t_fn
from pyvbmc import VBMC
import corner
import pickle

# %%
# =============================================================================
# PARAMETERS - Change ANIMAL_IDX to fit different animals
# =============================================================================
ANIMAL_IDX = 2  # Index into unique_animals array (0, 1, 2, ...)
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

# Get LED and stimulus timing distributions (for simulation and theoretical calculations)
LED_times = df_on_fit['t_LED'].values
stim_times = df_on_fit['t_stim'].values
stim_times_off = df_off_fit['t_stim'].values

# %%
# =============================================================================
# Simulation function
# =============================================================================
def simulate_proactive_single_bound(V_A_base, V_A_post_LED, theta_A, t_LED, t_stim, t_aff, t_effect, motor_delay, is_led_trial, dt=1e-4):
    """
    Simulate proactive process with single bound accumulator.
    Drift changes from V_A_base to V_A_post_LED at t_LED + t_effect (only for LED ON trials).
    Proactive process starts at t = t_aff (no noise before this).
    Returns RT when accumulator hits theta_A.
    """
    AI = 0
    t = t_aff
    dB = np.sqrt(dt)

    while True:
        if is_led_trial and t >= t_LED + t_effect:
            V_A = V_A_post_LED
        else:
            V_A = V_A_base

        AI += V_A * dt + np.random.normal(0, dB)
        t += dt

        if AI >= theta_A:
            RT = t + motor_delay
            return RT

# %%
# =============================================================================
# Likelihood functions
# =============================================================================
def PA_with_LEDON_2_adapted(t, v, vON, a, t_aff, motor_delay, tled, t_effect, T_trunc=None):
    """
    Compute the PA pdf by combining contributions before and after LED onset.
    """
    if T_trunc is not None and t <= T_trunc:
        return 0

    if (t - motor_delay) <= (tled + t_effect):
        pdf = d_A_RT(v * a, (t - motor_delay - t_aff) / (a**2)) / (a**2)
    else:
        t_post_led = t - motor_delay - tled - t_effect
        tp = tled + t_effect - t_aff

        if tp <= 0:
            pdf = d_A_RT(vON * a, (t - motor_delay - t_aff) / (a**2)) / (a**2)
        else:
            pdf = stupid_f_integral(v, vON, a, t_post_led, tp)

    if T_trunc is not None:
        t_pts = np.arange(0, T_trunc + 0.001, 0.001)
        pdf_vals = np.array([
            PA_with_LEDON_2_adapted(ti, v, vON, a, t_aff, motor_delay, tled, t_effect, None)
            for ti in t_pts
        ])
        cdf_trunc = np.trapz(pdf_vals, t_pts)
        pdf = pdf / (1 - cdf_trunc)

    return pdf


def led_off_cdf(t, v, a, t_aff, motor_delay):
    if t <= motor_delay + t_aff:
        return 0
    return cum_A_t_fn(t - motor_delay - t_aff, v, a)


def led_off_pdf_truncated(t, v, a, t_aff, motor_delay, T_trunc):
    if t <= T_trunc:
        return 0

    if t <= motor_delay + t_aff:
        return 0

    pdf = d_A_RT(v * a, (t - motor_delay - t_aff) / (a**2)) / (a**2)

    if T_trunc is not None:
        cdf_trunc = led_off_cdf(T_trunc, v, a, t_aff, motor_delay)
        trunc_factor = 1 - cdf_trunc
        if trunc_factor <= 0:
            return 0
        pdf = pdf / trunc_factor

    return pdf


def led_off_survival_truncated(t_stim, v, a, t_aff, motor_delay, T_trunc):
    if t_stim <= T_trunc:
        return 1.0

    cdf_t_stim = led_off_cdf(t_stim, v, a, t_aff, motor_delay)
    cdf_T_trunc = led_off_cdf(T_trunc, v, a, t_aff, motor_delay)
    trunc_factor = 1 - cdf_T_trunc
    if trunc_factor <= 0:
        return 1.0

    return (1 - cdf_t_stim) / trunc_factor


def led_on_survival_truncated(t_stim, t_led, v, vON, a, t_aff, motor_delay, t_effect, T_trunc):
    if t_stim <= T_trunc:
        return 1.0

    t_pts_cdf = np.arange(0, t_stim + 0.001, 0.001)
    pdf_vals_cdf = np.array([
        PA_with_LEDON_2_adapted(ti, v, vON, a, t_aff, motor_delay, t_led, t_effect, None)
        for ti in t_pts_cdf
    ])
    cdf_t_stim = np.trapz(pdf_vals_cdf, t_pts_cdf)

    t_pts_trunc = np.arange(0, T_trunc + 0.001, 0.001)
    pdf_vals_trunc = np.array([
        PA_with_LEDON_2_adapted(ti, v, vON, a, t_aff, motor_delay, t_led, t_effect, None)
        for ti in t_pts_trunc
    ])
    cdf_T_trunc = np.trapz(pdf_vals_trunc, t_pts_trunc)

    return (1 - cdf_t_stim) / (1 - cdf_T_trunc)


def compute_trial_loglike(row, V_A_base, V_A_post_LED, theta_A, t_aff, t_effect, motor_delay, T_trunc):
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
                t, V_A_base, V_A_post_LED, theta_A, t_aff, motor_delay, t_led, t_effect, T_trunc
            )
        else:
            likelihood = led_off_pdf_truncated(t, V_A_base, theta_A, t_aff, motor_delay, T_trunc)
    else:
        if is_led:
            likelihood = led_on_survival_truncated(
                t_stim, t_led, V_A_base, V_A_post_LED, theta_A, t_aff, motor_delay, t_effect, T_trunc
            )
        else:
            likelihood = led_off_survival_truncated(t_stim, V_A_base, theta_A, t_aff, motor_delay, T_trunc)

    if likelihood <= 0 or np.isnan(likelihood):
        likelihood = 1e-50

    return np.log(likelihood)


def proactive_led_loglike(params):
    V_A_base, V_A_post_LED, theta_A, t_aff, t_effect, motor_delay = params
    all_loglike = Parallel(n_jobs=30)(
        delayed(compute_trial_loglike)(
            row, V_A_base, V_A_post_LED, theta_A, t_aff, t_effect, motor_delay, T_trunc
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
t_aff_bounds = [-1, 0.1]
t_effect_bounds = [0.001, 0.1]
motor_delay_bounds = [-0.1, 0.1]

V_A_base_plausible = [0.5, 3]
V_A_post_LED_plausible = [0.5, 3]
theta_A_plausible = [0.5, 3]
t_aff_plausible = [0.01, 0.07]
t_effect_plausible = [0.01, 0.05]
motor_delay_plausible = [0.01, 0.07]


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
    V_A_base, V_A_post_LED, theta_A, t_aff, t_effect, motor_delay = params

    log_prior = 0
    log_prior += trapezoidal_logpdf(V_A_base, V_A_base_bounds[0], V_A_base_plausible[0], 
                                     V_A_base_plausible[1], V_A_base_bounds[1])
    log_prior += trapezoidal_logpdf(V_A_post_LED, V_A_post_LED_bounds[0], V_A_post_LED_plausible[0], 
                                     V_A_post_LED_plausible[1], V_A_post_LED_bounds[1])
    log_prior += trapezoidal_logpdf(theta_A, theta_A_bounds[0], theta_A_plausible[0], 
                                     theta_A_plausible[1], theta_A_bounds[1])
    log_prior += trapezoidal_logpdf(t_aff, t_aff_bounds[0], t_aff_plausible[0], 
                                     t_aff_plausible[1], t_aff_bounds[1])
    log_prior += trapezoidal_logpdf(t_effect, t_effect_bounds[0], t_effect_plausible[0], 
                                     t_effect_plausible[1], t_effect_bounds[1])
    log_prior += trapezoidal_logpdf(motor_delay, motor_delay_bounds[0], motor_delay_plausible[0], 
                                     motor_delay_plausible[1], motor_delay_bounds[1])
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
               t_aff_bounds[0], t_effect_bounds[0], motor_delay_bounds[0]])
ub = np.array([V_A_base_bounds[1], V_A_post_LED_bounds[1], theta_A_bounds[1], 
               t_aff_bounds[1], t_effect_bounds[1], motor_delay_bounds[1]])
plb = np.array([V_A_base_plausible[0], V_A_post_LED_plausible[0], theta_A_plausible[0], 
                t_aff_plausible[0], t_effect_plausible[0], motor_delay_plausible[0]])
pub = np.array([V_A_base_plausible[1], V_A_post_LED_plausible[1], theta_A_plausible[1], 
                t_aff_plausible[1], t_effect_plausible[1], motor_delay_plausible[1]])

# Initial point (use values similar to simulated ground truth)
np.random.seed(42)
x_0 = np.array([
    1.8 ,   # V_A_base
    2.4 ,   # V_A_post_LED
    1.5 ,   # theta_A
    0.04 ,  # t_aff
    0.035 , # t_effect
    0.05   # motor_delay
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
# Run VBMC
# =============================================================================
print(f"\nRunning VBMC optimization for animal {animal}...")
vbmc = VBMC(vbmc_joint, x_0, lb, ub, plb, pub, options={'display': 'iter'})
vp, results = vbmc.optimize()

print("\nVBMC optimization complete!")

# %%
# =============================================================================
# Save results
# =============================================================================
vp.save(f'vbmc_real_{animal}_fit.pkl', overwrite=True)

results_summary = {
    'elbo': results.get('elbo', None),
    'elbo_sd': results.get('elbo_sd', None),
    'convergence_status': results.get('convergence_status', None)
}

with open(f'vbmc_real_{animal}_results.pkl', 'wb') as f:
    pickle.dump({
        'animal': animal,
        'results_summary': results_summary,
        'fit_df': fit_df,
        'bounds': {
            'lb': lb, 'ub': ub, 'plb': plb, 'pub': pub
        }
    }, f)
print(f"Results saved for animal {animal}!")

# %%
# =============================================================================
# Sample from posterior
# =============================================================================
vp_samples = vp.sample(int(1e5))[0]

param_labels = ['V_A_base', 'V_A_post_LED', 'theta_A', 't_aff', 't_effect', 'motor_delay']
param_means = np.mean(vp_samples, axis=0)
param_stds = np.std(vp_samples, axis=0)

print(f"\nPosterior summary for animal {animal}:")
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
    quantiles=[0.16, 0.5, 0.84]
)
plt.suptitle(f'Animal {animal} Posterior', y=1.02)
plt.savefig(f'vbmc_real_{animal}_corner.pdf', bbox_inches='tight')
print(f"Corner plot saved as 'vbmc_real_{animal}_corner.pdf'")
plt.show()

# %%
# =============================================================================
# Theoretical RTD calculation (conditioned on RT < t_stim, i.e., aborts only)
# P(RT=t | abort) = P(RT=t) * P(t_stim > t) / P(abort)
# =============================================================================
def compute_theoretical_rtd_on(t_pts, V_A_base, V_A_post_LED, theta_A, t_aff, t_effect, motor_delay,
                               N_mc=5000, T_trunc=None):
    """Compute theoretical RTD for LED ON, conditioned on RT < t_stim (aborts).
    
    Monte Carlo loop over t_led and t_stim samples, applying same conditions as likelihood.
    """
    pdf_samples = np.zeros((N_mc, len(t_pts)))
    
    for i in range(N_mc):
        t_led = np.random.choice(LED_times)
        t_stim = np.random.choice(stim_times)
        
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
                    t, V_A_base, V_A_post_LED, theta_A, t_aff, motor_delay, t_led, t_effect, T_trunc
                )
        
        # Normalize this sample's PDF to integrate to 1 over valid region
        norm = np.trapz(sample_pdf, t_pts)
        if norm > 0:
            # pdf_samples[i, :] = sample_pdf / norm
            pdf_samples[i, :] = sample_pdf

    
    # Average over all MC samples
    return np.mean(pdf_samples, axis=0)


def compute_theoretical_rtd_off(t_pts, V_A_base, theta_A, t_aff, motor_delay, T_trunc=None, N_mc=5000):
    """Compute theoretical RTD for LED OFF, conditioned on RT < t_stim (aborts).
    
    Monte Carlo loop over t_stim samples, applying same conditions as likelihood.
    """
    pdf_samples = np.zeros((N_mc, len(t_pts)))
    
    for i in range(N_mc):
        t_stim = np.random.choice(stim_times_off)
        
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
                sample_pdf[j] = led_off_pdf_truncated(t, V_A_base, theta_A, t_aff, motor_delay, T_trunc)
        
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
    t_pts, param_means[0], param_means[1], param_means[2], param_means[3], param_means[4], 
    param_means[5], N_mc=5000, T_trunc=T_trunc
)
pdf_theory_off_fit = compute_theoretical_rtd_off(
    t_pts, param_means[0], param_means[2], param_means[3], param_means[5], T_trunc=T_trunc
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

# ##############################################
########### NOTE ###########################
# param_means[3] = -1.2
#############################
def simulate_single_trial_fit():
    is_led_trial = np.random.random() < 1/3
    t_LED = np.random.choice(LED_times)  # Always sample t_LED for RT wrt LED calculation
    if is_led_trial:
        t_stim = np.random.choice(stim_times)
    else:
        t_stim = np.random.choice(stim_times_off)
    rt = simulate_proactive_single_bound(
        param_means[0], param_means[1], param_means[2],  # V_A_base, V_A_post_LED, theta_A
        t_LED if is_led_trial else None, t_stim,
        param_means[3], param_means[4], param_means[5],  # t_aff, t_effect, motor_delay
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
plt.savefig(f'vbmc_real_{animal}_rtd_comparison.pdf', bbox_inches='tight')
print(f"RTD comparison saved as 'vbmc_real_{animal}_rtd_comparison.pdf'")
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
axes[0].axvline(x=param_means[4], color='r', linestyle=':', alpha=0.5, label=f'LED effect={param_means[4]:.2f}')
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
plt.savefig(f'vbmc_real_{animal}_rt_wrt_led_comparison.pdf', bbox_inches='tight')
print(f"RT wrt LED comparison saved as 'vbmc_real_{animal}_rt_wrt_led_comparison.pdf'")
plt.show()

# %%
# =============================================================================
# Plot 3: RT wrt LED - Abort rate (like aborts_animal_wise_explore.py)
# =============================================================================
# Compute abort rate = n_aborts / n_all_trials for each LED condition (from data)
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
ax.plot(bin_centers_wrt_led, data_hist_on_scaled, label=f'Data LED ON (frac={frac_data_on:.2f})', lw=2, alpha=0.7, color='r', linestyle='-')
ax.plot(bin_centers_wrt_led, data_hist_off_scaled, label=f'Data LED OFF (frac={frac_data_off:.2f})', lw=2, alpha=0.7, color='b', linestyle='-')
ax.plot(bin_centers_wrt_led, sim_hist_on_scaled, label=f'Sim LED ON (frac={frac_sim_on:.2f})', lw=2, alpha=0.7, color='r', linestyle='--')
ax.plot(bin_centers_wrt_led, sim_hist_off_scaled, label=f'Sim LED OFF (frac={frac_sim_off:.2f})', lw=2, alpha=0.7, color='b', linestyle='--')

ax.axvline(x=0, color='k', linestyle='--', alpha=0.5, label='LED onset')
ax.axvline(x=param_means[4], color='g', linestyle=':', alpha=0.5, label=f't_effect={param_means[4]:.2f}')
ax.set_xlabel('RT - t_LED (s)', fontsize=12)
ax.set_ylabel('Rate (area = fraction)', fontsize=12)
ax.set_title(f'RT wrt LED (area-weighted) - Animal {animal}', fontsize=14)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f'vbmc_real_{animal}_rt_wrt_led_rate.pdf', bbox_inches='tight')
print(f"RT wrt LED rate plot saved as 'vbmc_real_{animal}_rt_wrt_led_rate.pdf'")
plt.show()

print(f"\nScript complete for animal {animal}!")
# %%
