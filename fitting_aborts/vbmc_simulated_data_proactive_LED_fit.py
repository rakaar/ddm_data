"""
VBMC Fitting of Simulated Proactive Process Data with LED ON/OFF Conditions
============================================================================

This script performs parameter recovery on simulated data from a proactive process model
using Variational Bayesian Monte Carlo (VBMC). It validates the fitting pipeline by:
1. Simulating data with known ground truth parameters
2. Fitting the model using VBMC
3. Comparing fitted vs true parameters

SCRIPT STRUCTURE (cell-by-cell):
--------------------------------
1. **Imports** - numpy, matplotlib, joblib, pandas, pyvbmc, corner
2. **Load Data** - Load real experimental data to get LED/stimulus timing distributions
3. **Ground Truth Parameters** - Define true parameters for simulation
4. **Simulation Function** - `simulate_proactive_single_bound()`: Single-bound accumulator with drift change at LED onset
5. **Generate Simulated Data** - Simulate N_sim trials (3000 default) with ground truth params
6. **Likelihood Functions**:
   - `PA_with_LEDON_2_adapted()`: PDF for LED ON trials (handles drift change)
   - `led_off_pdf_truncated()`: PDF for LED OFF trials
   - `led_on_survival_truncated()`: Survival probability for censored LED ON trials
   - `led_off_survival_truncated()`: Survival probability for censored LED OFF trials
   - `compute_trial_loglike()`: Single trial log-likelihood (handles truncation & censoring)
   - `proactive_led_loglike()`: Total log-likelihood across all trials
7. **Prior Functions** - Trapezoidal priors for VBMC
8. **VBMC Setup** - Define bounds, plausible bounds, and joint function
9. **Run VBMC** - Optimize and get posterior samples
10. **Posterior Visualization** - Corner plot of posterior
11. **Theoretical RTD Calculation** - Monte Carlo computation of theoretical RT distributions
12. **Large-scale Simulation** - 200K trials with ground truth params for plotting
13. **Comparison Plots**:
    - RTD comparison: simulated vs theoretical (ground truth & fitted)
    - RT wrt LED: histograms of RT - t_LED for ground truth vs fitted simulations

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
# Load data to get LED and stimulus timing distributions
og_df = pd.read_csv('../out_LED.csv')

df = og_df[ og_df['repeat_trial'].isin([0,2]) | og_df['repeat_trial'].isna() ]
session_type = 7    
df = df[ df['session_type'].isin([session_type]) ]
training_level = 16
df = df[ df['training_level'].isin([training_level]) ]

df = df.dropna(subset=['intended_fix', 'LED_onset_time', 'timed_fix'])
df = df[(df['abort_event'] == 3) | (df['success'].isin([1,-1]))]

# Filter out aborts < 300ms
FILTER_300 = True
if FILTER_300:
    df = df[~( (df['abort_event'] == 3) & (df['timed_fix'] < 0.3) )]

# Get LED ON and OFF trials
df_on = df[df['LED_trial'] == 1]
df_on_1 = df_on.copy()
df_on_1['LED_wrt_fix'] = df_on_1['intended_fix'] - df_on_1['LED_onset_time']

df_off = df[df['LED_trial'] == 0]

# Get LED times from data
LED_times = df_on_1['LED_wrt_fix'].values
stim_times = df_on_1['intended_fix'].values
stim_times_off = df_off['intended_fix'].values

print(f"Number of LED ON trials in data: {len(df_on_1)}")
print(f"Number of LED OFF trials in data: {len(df_off)}")

# %%
N_sim = int(3e3)

# Ground truth parameters for simulation
V_A_base_true = 1.8
V_A_post_LED_true = 2.4
theta_A_true = 1.5
t_aff_true = -1 * 30*1e-3
t_effect_true = 35*1e-3
motor_delay_true = 30*1e-3
T_trunc = 0.3

print("Ground truth parameters:")
print(f"  V_A_base: {V_A_base_true}")
print(f"  V_A_post_LED: {V_A_post_LED_true}")
print(f"  theta_A: {theta_A_true}")
print(f"  t_aff: {t_aff_true}")
print(f"  t_effect: {t_effect_true}")
print(f"  motor_delay: {motor_delay_true}")
print(f"  T_trunc: {T_trunc}")

# %%
# Simulate proactive process with single bound accumulator
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
# Generate simulated data (3000 trials)
print(f"Simulating {N_sim} trials...")

def simulate_single_trial():
    is_led_trial = np.random.random() < 1/3

    if is_led_trial:
        t_LED = np.random.choice(LED_times)
        t_stim = np.random.choice(stim_times)
    else:
        t_LED = None
        t_stim = np.random.choice(stim_times_off)

    rt = simulate_proactive_single_bound(
        V_A_base_true, V_A_post_LED_true, theta_A_true,
        t_LED,
        t_stim,
        t_aff_true,
        t_effect_true,
        motor_delay_true,
        is_led_trial
    )
    return rt, is_led_trial, t_stim, t_LED

sim_results = Parallel(n_jobs=30)(
    delayed(simulate_single_trial)() for _ in tqdm(range(N_sim))
)
sim_rts = [r[0] for r in sim_results]
sim_is_led_trials = [r[1] for r in sim_results]
sim_t_stims = [r[2] for r in sim_results]
sim_t_LEDs = [r[3] for r in sim_results]

# Separate into LED ON and OFF, apply truncation
sim_rts_on = [rt for rt, is_led in zip(sim_rts, sim_is_led_trials) if is_led and rt > T_trunc]
sim_rts_off = [rt for rt, is_led in zip(sim_rts, sim_is_led_trials) if not is_led and rt > T_trunc]

sim_t_stims_on = [t_stim for rt, is_led, t_stim in zip(sim_rts, sim_is_led_trials, sim_t_stims)
                  if is_led and rt > T_trunc]
sim_t_stims_off = [t_stim for rt, is_led, t_stim in zip(sim_rts, sim_is_led_trials, sim_t_stims)
                   if (not is_led) and rt > T_trunc]
sim_t_LEDs_on = [t_led for rt, is_led, t_led in zip(sim_rts, sim_is_led_trials, sim_t_LEDs)
                 if is_led and rt > T_trunc]

sim_df = pd.DataFrame({
    'RT': sim_rts_on + sim_rts_off,
    't_stim': sim_t_stims_on + sim_t_stims_off,
    't_LED': sim_t_LEDs_on + [np.nan] * len(sim_rts_off),
    'LED_trial': [1] * len(sim_rts_on) + [0] * len(sim_rts_off)
})

print(f"Simulated data summary:")
print(f"  Total trials after truncation: {len(sim_df)}")
print(f"  LED ON trials: {len(sim_rts_on)}")
print(f"  LED OFF trials: {len(sim_rts_off)}")

# %%
# Likelihood functions
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
        ) for _, row in sim_df.iterrows()
    )
    return np.sum(all_loglike)

# %%
# Bounds and priors
# Ground truth: V_A_base=1.8, V_A_post_LED=2.4, theta_A=1.5, t_aff=0.04, t_effect=0.035, motor_delay=0.05


V_A_base_bounds = [0.1, 5]
V_A_post_LED_bounds = [0.1, 5]
theta_A_bounds = [0.1, 5]
t_aff_bounds = [-1, 0.1]
t_effect_bounds = [0.001, 0.10]
motor_delay_bounds = [-0.1, 0.1]


V_A_base_plausible = [0.5, 3]
V_A_post_LED_plausible = [0.5, 3]
theta_A_plausible = [0.5, 3]
t_aff_plausible = [0.01, 0.07]
t_effect_plausible = [0.01, 0.05]
motor_delay_plausible = [0.01, 0.07]

# Narrower bounds for better recovery on simulated data
# V_A_base_bounds = [0.5, 3.5]
# V_A_post_LED_bounds = [1.0, 4.0]
# theta_A_bounds = [0.8, 3.0]
# t_aff_bounds = [0.02, 0.08]
# t_effect_bounds = [0.015, 0.06]
# motor_delay_bounds = [0.03, 0.08]

# V_A_base_plausible = [1.0, 2.5]
# V_A_post_LED_plausible = [1.5, 3.5]
# theta_A_plausible = [1.0, 2.5]
# t_aff_plausible = [0.03, 0.055]
# t_effect_plausible = [0.025, 0.045]
# motor_delay_plausible = [0.04, 0.065]

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
# Set up VBMC bounds
lb = np.array([V_A_base_bounds[0], V_A_post_LED_bounds[0], theta_A_bounds[0], 
               t_aff_bounds[0], t_effect_bounds[0], motor_delay_bounds[0]])
ub = np.array([V_A_base_bounds[1], V_A_post_LED_bounds[1], theta_A_bounds[1], 
               t_aff_bounds[1], t_effect_bounds[1], motor_delay_bounds[1]])
plb = np.array([V_A_base_plausible[0], V_A_post_LED_plausible[0], theta_A_plausible[0], 
                t_aff_plausible[0], t_effect_plausible[0], motor_delay_plausible[0]])
pub = np.array([V_A_base_plausible[1], V_A_post_LED_plausible[1], theta_A_plausible[1], 
                t_aff_plausible[1], t_effect_plausible[1], motor_delay_plausible[1]])

# Initial point (use ground truth with small perturbation)
np.random.seed(42)
x_0 = np.array([
    V_A_base_true + np.random.normal(0, 0.1),
    V_A_post_LED_true + np.random.normal(0, 0.1),
    theta_A_true + np.random.normal(0, 0.1),
    t_aff_true + np.random.normal(0, 0.005),
    t_effect_true + np.random.normal(0, 0.005),
    motor_delay_true + np.random.normal(0, 0.005)
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
# Run VBMC
print("\nRunning VBMC optimization...")
vbmc = VBMC(vbmc_joint, x_0, lb, ub, plb, pub, options={'display': 'iter'})
vp, results = vbmc.optimize()

print("\nVBMC optimization complete!")

# %%
# Save results
vp.save('vbmc_simulated_proactive_LED_fit.pkl', overwrite=True)

# Extract picklable data from results (avoid pickling the full results object)
results_summary = {
    'elbo': results.get('elbo', None),
    'elbo_sd': results.get('elbo_sd', None),
    'convergence_status': results.get('convergence_status', None)
}

with open('vbmc_simulated_proactive_LED_results.pkl', 'wb') as f:
    pickle.dump({
        'results_summary': results_summary,
        'ground_truth': {
            'V_A_base': V_A_base_true,
            'V_A_post_LED': V_A_post_LED_true,
            'theta_A': theta_A_true,
            't_aff': t_aff_true,
            't_effect': t_effect_true,
            'motor_delay': motor_delay_true
        },
        'sim_df': sim_df,
        'bounds': {
            'lb': lb, 'ub': ub, 'plb': plb, 'pub': pub
        }
    }, f)
print("Results saved!")

# %%
# Sample from posterior
vp_samples = vp.sample(int(1e5))[0]

param_labels = ['V_A_base', 'V_A_post_LED', 'theta_A', 't_aff', 't_effect', 'motor_delay']
param_means = np.mean(vp_samples, axis=0)
param_stds = np.std(vp_samples, axis=0)

print("\nPosterior summary:")
print(f"{'Parameter':<15} {'True':<10} {'Mean':<10} {'Std':<10} {'Pct Error':<10}")
print("-" * 65)
ground_truth = [V_A_base_true, V_A_post_LED_true, theta_A_true, t_aff_true, t_effect_true, motor_delay_true]
for i, label in enumerate(param_labels):
    pct_error = 100 * (param_means[i] - ground_truth[i]) / ground_truth[i]
    print(f"{label:<15} {ground_truth[i]:<10.4f} {param_means[i]:<10.4f} {param_stds[i]:<10.4f} {pct_error:<10.2f}%")

# %%
# Corner plot
fig = corner.corner(
    vp_samples, 
    labels=param_labels, 
    truths=ground_truth,
    show_titles=True, 
    title_fmt=".4f",
    quantiles=[0.16, 0.5, 0.84]
)
plt.savefig('vbmc_simulated_proactive_LED_corner.pdf', bbox_inches='tight')
print("Corner plot saved as 'vbmc_simulated_proactive_LED_corner.pdf'")
plt.show()

# %%
# Verification: Compare theoretical RT distributions
def compute_theoretical_rtd_on(t_pts, V_A_base, V_A_post_LED, theta_A, t_aff, t_effect, motor_delay,
                               N_mc=1000, T_trunc=None):
    pdf_samples = np.zeros((N_mc, len(t_pts)))
    for i in range(N_mc):
        t_led = np.random.choice(LED_times)
        for j, t in enumerate(t_pts):
            pdf_samples[i, j] = PA_with_LEDON_2_adapted(
                t, V_A_base, V_A_post_LED, theta_A, t_aff, motor_delay, t_led, t_effect, T_trunc
            )
    return np.mean(pdf_samples, axis=0)


def compute_theoretical_rtd_off(t_pts, V_A_base, theta_A, t_aff, motor_delay, T_trunc=None):
    return np.array([
        led_off_pdf_truncated(t, V_A_base, theta_A, t_aff, motor_delay, T_trunc)
        for t in t_pts
    ])

# Compute theoretical distributions with ground truth parameters
dt = 0.01
t_max = 3
t_pts = np.arange(T_trunc, t_max, dt)

print("\nComputing theoretical RT distributions...")
pdf_theory_on_true = compute_theoretical_rtd_on(
    t_pts, V_A_base_true, V_A_post_LED_true, theta_A_true, t_aff_true, t_effect_true, 
    motor_delay_true, N_mc=500, T_trunc=T_trunc
)
pdf_theory_off_true = compute_theoretical_rtd_off(
    t_pts, V_A_base_true, theta_A_true, t_aff_true, motor_delay_true, T_trunc=T_trunc
)

# Compute theoretical distributions with fitted parameters
pdf_theory_on_fit = compute_theoretical_rtd_on(
    t_pts, param_means[0], param_means[1], param_means[2], param_means[3], param_means[4], 
    param_means[5], N_mc=500, T_trunc=T_trunc
)
pdf_theory_off_fit = compute_theoretical_rtd_off(
    t_pts, param_means[0], param_means[2], param_means[3], param_means[5], T_trunc=T_trunc
)

# Normalize
pdf_theory_on_true /= np.trapz(pdf_theory_on_true, t_pts)
pdf_theory_off_true /= np.trapz(pdf_theory_off_true, t_pts)
pdf_theory_on_fit /= np.trapz(pdf_theory_on_fit, t_pts)
pdf_theory_off_fit /= np.trapz(pdf_theory_off_fit, t_pts)

# %%
# Simulate  with ground truth parameters for RTD comparison
N_trials_plot = int(200e3)
print(f"Simulating {N_trials_plot} trials with ground truth parameters for RTD comparison...")

def simulate_single_trial_plot():
    is_led_trial = np.random.random() < 1/3
    t_LED = np.random.choice(LED_times)  # Always sample t_LED for RT wrt LED calculation
    if is_led_trial:
        t_stim = np.random.choice(stim_times)
    else:
        t_stim = np.random.choice(stim_times_off)
    rt = simulate_proactive_single_bound(
        V_A_base_true, V_A_post_LED_true, theta_A_true,
        t_LED if is_led_trial else None, t_stim, t_aff_true, t_effect_true, motor_delay_true, is_led_trial
    )
    return rt, is_led_trial, t_LED, t_stim

plot_sim_results = Parallel(n_jobs=30)(
    delayed(simulate_single_trial_plot)() for _ in range(N_trials_plot)
)

plot_rts = [r[0] for r in plot_sim_results]
plot_is_led = [r[1] for r in plot_sim_results]
plot_t_LEDs = [r[2] for r in plot_sim_results]
plot_t_stims = [r[3] for r in plot_sim_results]

# Separate into LED ON/OFF and apply truncation, also get RT wrt LED
plot_rts_on = [rt for rt, is_led in zip(plot_rts, plot_is_led) if is_led and rt > T_trunc]
plot_rts_off = [rt for rt, is_led in zip(plot_rts, plot_is_led) if not is_led and rt > T_trunc]
plot_rts_wrt_led_on = [rt - t_led for rt, is_led, t_led in zip(plot_rts, plot_is_led, plot_t_LEDs) if is_led and rt > T_trunc]
plot_rts_wrt_led_off = [rt - t_led for rt, is_led, t_led in zip(plot_rts, plot_is_led, plot_t_LEDs) if not is_led and rt > T_trunc]

print(f"Plot simulation: {len(plot_rts_on)} LED ON trials, {len(plot_rts_off)} LED OFF trials (after truncation)")

# %%
# Plot LED ON comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

bins = np.arange(T_trunc, t_max, 0.05)
sim_hist_on, _ = np.histogram(plot_rts_on, bins=bins, density=True)
bin_centers = (bins[1:] + bins[:-1]) / 2

axes[0].plot(bin_centers, sim_hist_on, label='Simulated data', lw=2, alpha=0.5, color='b')
axes[0].plot(t_pts, pdf_theory_on_fit, label='Theory (fitted)', lw=2, ls='--', color='k')
axes[0].plot(t_pts, pdf_theory_on_true, label='Theory (ground truth)', lw=2, ls=':', color='red')
axes[0].axvline(x=T_trunc, color='r', linestyle='--', alpha=0.5, label='Truncation')
axes[0].set_xlabel('RT (s)', fontsize=12)
axes[0].set_ylabel('Density', fontsize=12)
axes[0].set_title('LED ON Trials', fontsize=14)
axes[0].legend(fontsize=10)


# Plot LED OFF comparison
sim_hist_off, _ = np.histogram(plot_rts_off, bins=bins, density=True)

axes[1].plot(bin_centers, sim_hist_off, label='Simulated data', lw=2, alpha=0.5, color='b')
axes[1].plot(t_pts, pdf_theory_off_fit, label='Theory (fitted)', lw=2, ls='--', color='k')
axes[1].plot(t_pts, pdf_theory_off_true, label='Theory (ground truth)', lw=2, ls=':', color='red')
axes[1].axvline(x=T_trunc, color='r', linestyle='--', alpha=0.5, label='Truncation')
axes[1].set_xlabel('RT (s)', fontsize=12)
axes[1].set_ylabel('Density', fontsize=12)
axes[1].set_title('LED OFF Trials', fontsize=14)
axes[1].legend(fontsize=10)


plt.tight_layout()
plt.savefig('vbmc_simulated_proactive_LED_rtd_comparison.pdf', bbox_inches='tight')
print("RT distribution comparison saved as 'vbmc_simulated_proactive_LED_rtd_comparison.pdf'")
plt.show()

print("\nScript complete!")

# %%
# Simulate with fitted parameters (VP mean) for validation
print("\nSimulating trials with fitted parameters (VP mean)...")
N_trials_fit = int(200e3)

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

fit_sim_results = Parallel(n_jobs=30)(
    delayed(simulate_single_trial_fit)() for _ in range(N_trials_fit)
)

fit_rts = [r[0] for r in fit_sim_results]
fit_is_led = [r[1] for r in fit_sim_results]
fit_t_LEDs = [r[2] for r in fit_sim_results]
fit_t_stims = [r[3] for r in fit_sim_results]

# Separate into LED ON/OFF and apply truncation, also get RT wrt LED
fit_rts_on = [rt for rt, is_led in zip(fit_rts, fit_is_led) if is_led and rt > T_trunc]
fit_rts_off = [rt for rt, is_led in zip(fit_rts, fit_is_led) if not is_led and rt > T_trunc]
fit_rts_wrt_led_on = [rt - t_led for rt, is_led, t_led in zip(fit_rts, fit_is_led, fit_t_LEDs) if is_led and rt > T_trunc]
fit_rts_wrt_led_off = [rt - t_led for rt, is_led, t_led in zip(fit_rts, fit_is_led, fit_t_LEDs) if not is_led and rt > T_trunc]

print(f"Fitted params simulation: {len(fit_rts_on)} LED ON trials, {len(fit_rts_off)} LED OFF trials (after truncation)")

# %%
# Plot RT wrt LED: ground truth sim vs fitted sim
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

bins_wrt_led = np.arange(-3, 3, 0.05)
bin_centers_wrt_led = (bins_wrt_led[1:] + bins_wrt_led[:-1]) / 2

# LED ON
sim_hist_on_gt_wrt_led, _ = np.histogram(plot_rts_wrt_led_on, bins=bins_wrt_led, density=True)
sim_hist_on_fit_wrt_led, _ = np.histogram(fit_rts_wrt_led_on, bins=bins_wrt_led, density=True)

axes[0].plot(bin_centers_wrt_led, sim_hist_on_gt_wrt_led, label='Sim (ground truth)', lw=2, alpha=0.7, color='b')
axes[0].plot(bin_centers_wrt_led, sim_hist_on_fit_wrt_led, label='Sim (fitted)', lw=2, alpha=0.7, color='r')
axes[0].axvline(x=0, color='k', linestyle='--', alpha=0.5, label='LED onset')
axes[0].set_xlabel('RT - t_LED (s)', fontsize=12)
axes[0].set_ylabel('Density', fontsize=12)
axes[0].set_title('LED ON Trials', fontsize=14)
axes[0].legend(fontsize=10)

# LED OFF
sim_hist_off_gt_wrt_led, _ = np.histogram(plot_rts_wrt_led_off, bins=bins_wrt_led, density=True)
sim_hist_off_fit_wrt_led, _ = np.histogram(fit_rts_wrt_led_off, bins=bins_wrt_led, density=True)

axes[1].plot(bin_centers_wrt_led, sim_hist_off_gt_wrt_led, label='Sim (ground truth)', lw=2, alpha=0.7, color='b')
axes[1].plot(bin_centers_wrt_led, sim_hist_off_fit_wrt_led, label='Sim (fitted)', lw=2, alpha=0.7, color='r')
axes[1].axvline(x=0, color='k', linestyle='--', alpha=0.5, label='LED onset')
axes[1].set_xlabel('RT - t_LED (s)', fontsize=12)
axes[1].set_ylabel('Density', fontsize=12)
axes[1].set_title('LED OFF Trials', fontsize=14)
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig('vbmc_simulated_proactive_LED_rt_wrt_led_comparison.pdf', bbox_inches='tight')
print("RT wrt LED comparison saved as 'vbmc_simulated_proactive_LED_rt_wrt_led_comparison.pdf'")
plt.show()

# %%
# Plot RT wrt LED rate (area-weighted by abort rate)
# Compute abort rate for ground truth simulation (aborts = RT < t_stim, after truncation)
n_all_gt_on = sum(1 for is_led in plot_is_led if is_led)
n_all_gt_off = sum(1 for is_led in plot_is_led if not is_led)
n_aborts_gt_on = sum(1 for rt, is_led, t_stim in zip(plot_rts, plot_is_led, plot_t_stims) if is_led and rt < t_stim and rt > T_trunc)
n_aborts_gt_off = sum(1 for rt, is_led, t_stim in zip(plot_rts, plot_is_led, plot_t_stims) if not is_led and rt < t_stim and rt > T_trunc)
frac_gt_on = n_aborts_gt_on / n_all_gt_on if n_all_gt_on > 0 else 0
frac_gt_off = n_aborts_gt_off / n_all_gt_off if n_all_gt_off > 0 else 0

# Compute abort rate for fitted simulation
n_all_fit_on = sum(1 for is_led in fit_is_led if is_led)
n_all_fit_off = sum(1 for is_led in fit_is_led if not is_led)
n_aborts_fit_on = sum(1 for rt, is_led, t_stim in zip(fit_rts, fit_is_led, fit_t_stims) if is_led and rt < t_stim and rt > T_trunc)
n_aborts_fit_off = sum(1 for rt, is_led, t_stim in zip(fit_rts, fit_is_led, fit_t_stims) if not is_led and rt < t_stim and rt > T_trunc)
frac_fit_on = n_aborts_fit_on / n_all_fit_on if n_all_fit_on > 0 else 0
frac_fit_off = n_aborts_fit_off / n_all_fit_off if n_all_fit_off > 0 else 0

# Create histograms with density=True then scale by fraction
fig, ax = plt.subplots(figsize=(10, 6))

# Ground truth histograms (area = fraction)
gt_hist_on_wrt_led_dens, _ = np.histogram(plot_rts_wrt_led_on, bins=bins_wrt_led, density=True)
gt_hist_off_wrt_led_dens, _ = np.histogram(plot_rts_wrt_led_off, bins=bins_wrt_led, density=True)
gt_hist_on_scaled = gt_hist_on_wrt_led_dens * frac_gt_on
gt_hist_off_scaled = gt_hist_off_wrt_led_dens * frac_gt_off

# Fitted histograms (area = fraction)
fit_hist_on_wrt_led_dens, _ = np.histogram(fit_rts_wrt_led_on, bins=bins_wrt_led, density=True)
fit_hist_off_wrt_led_dens, _ = np.histogram(fit_rts_wrt_led_off, bins=bins_wrt_led, density=True)
fit_hist_on_scaled = fit_hist_on_wrt_led_dens * frac_fit_on
fit_hist_off_scaled = fit_hist_off_wrt_led_dens * frac_fit_off

# Plot all on same axes
ax.plot(bin_centers_wrt_led, gt_hist_on_scaled, label=f'GT LED ON (frac={frac_gt_on:.2f})', lw=2, alpha=0.7, color='r')
ax.plot(bin_centers_wrt_led, gt_hist_off_scaled, label=f'GT LED OFF (frac={frac_gt_off:.2f})', lw=2, alpha=0.7, color='b')
ax.plot(bin_centers_wrt_led, fit_hist_on_scaled, label=f'Fit LED ON (frac={frac_fit_on:.2f})', lw=2, alpha=0.7, color='r', linestyle='--')
ax.plot(bin_centers_wrt_led, fit_hist_off_scaled, label=f'Fit LED OFF (frac={frac_fit_off:.2f})', lw=2, alpha=0.7, color='b', linestyle='--')

ax.axvline(x=0, color='k', linestyle='--', alpha=0.5, label='LED onset')
ax.axvline(x=t_effect_true, color='g', linestyle=':', alpha=0.5, label=f't_effect (GT={t_effect_true:.2f})')
ax.set_xlabel('RT - t_LED (s)', fontsize=12)
ax.set_ylabel('Rate (area = fraction)', fontsize=12)
ax.set_title('RT wrt LED (area-weighted)', fontsize=14)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('vbmc_simulated_proactive_LED_rt_wrt_led_rate.pdf', bbox_inches='tight')
print("RT wrt LED rate plot saved as 'vbmc_simulated_proactive_LED_rt_wrt_led_rate.pdf'")
plt.show()
# %%
# TODO