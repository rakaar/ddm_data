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
# Ground truth parameters for simulation
V_A_base_true = 1.8
V_A_post_LED_true = 2.4
theta_A_true = 1.5
t_aff_true = 40*1e-3
t_effect_true = 35*1e-3
motor_delay_true = 50*1e-3
T_trunc = 0.6

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
N_sim = 3000
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

    if t <= T_trunc:
        return np.log(1e-50)

    if t_stim <= T_trunc:
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
# Bounds and priors (from V_A_step_jump_fit_censor_post_LED_real_data.py)
V_A_base_bounds = [0.1, 5]
V_A_post_LED_bounds = [0.1, 6]
theta_A_bounds = [0.5, 5]
t_aff_bounds = [0.01, 0.15]
t_effect_bounds = [0.01, 0.10]
motor_delay_bounds = [0.02, 0.12]

V_A_base_plausible = [0.5, 3]
V_A_post_LED_plausible = [1.0, 4]
theta_A_plausible = [1, 3]
t_aff_plausible = [0.03, 0.06]
t_effect_plausible = [0.02, 0.05]
motor_delay_plausible = [0.04, 0.07]

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
vp.save('vbmc_simulated_proactive_LED_fit.pkl')
with open('vbmc_simulated_proactive_LED_results.pkl', 'wb') as f:
    pickle.dump({
        'vp': vp,
        'results': results,
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
vp_samples = vp.sample(int(1e6))[0]

param_labels = ['V_A_base', 'V_A_post_LED', 'theta_A', 't_aff', 't_effect', 'motor_delay']
param_means = np.mean(vp_samples, axis=0)
param_stds = np.std(vp_samples, axis=0)

print("\nPosterior summary:")
print(f"{'Parameter':<15} {'True':<10} {'Mean':<10} {'Std':<10} {'Error':<10}")
print("-" * 60)
ground_truth = [V_A_base_true, V_A_post_LED_true, theta_A_true, t_aff_true, t_effect_true, motor_delay_true]
for i, label in enumerate(param_labels):
    error = param_means[i] - ground_truth[i]
    print(f"{label:<15} {ground_truth[i]:<10.4f} {param_means[i]:<10.4f} {param_stds[i]:<10.4f} {error:<10.4f}")

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
t_max = 2.0
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
# Plot LED ON comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

bins = np.arange(T_trunc, t_max, dt)
sim_hist_on, _ = np.histogram(sim_rts_on, bins=bins, density=True)
bin_centers = (bins[1:] + bins[:-1]) / 2

axes[0].plot(bin_centers, sim_hist_on, label='Simulated data', lw=2, alpha=0.7)
axes[0].plot(t_pts, pdf_theory_on_true, label='Theory (ground truth)', lw=2, ls='--')
axes[0].plot(t_pts, pdf_theory_on_fit, label='Theory (fitted)', lw=2, ls=':')
axes[0].axvline(x=T_trunc, color='r', linestyle='--', alpha=0.5, label='Truncation')
axes[0].set_xlabel('RT (s)', fontsize=12)
axes[0].set_ylabel('Density', fontsize=12)
axes[0].set_title('LED ON Trials', fontsize=14)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Plot LED OFF comparison
sim_hist_off, _ = np.histogram(sim_rts_off, bins=bins, density=True)

axes[1].plot(bin_centers, sim_hist_off, label='Simulated data', lw=2, alpha=0.7)
axes[1].plot(t_pts, pdf_theory_off_true, label='Theory (ground truth)', lw=2, ls='--')
axes[1].plot(t_pts, pdf_theory_off_fit, label='Theory (fitted)', lw=2, ls=':')
axes[1].axvline(x=T_trunc, color='r', linestyle='--', alpha=0.5, label='Truncation')
axes[1].set_xlabel('RT (s)', fontsize=12)
axes[1].set_ylabel('Density', fontsize=12)
axes[1].set_title('LED OFF Trials', fontsize=14)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('vbmc_simulated_proactive_LED_rtd_comparison.pdf', bbox_inches='tight')
print("RT distribution comparison saved as 'vbmc_simulated_proactive_LED_rtd_comparison.pdf'")
plt.show()

print("\nScript complete!")
