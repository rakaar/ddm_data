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

# %%
# Load data to get LED and stimulus timing distributions
og_df = pd.read_csv('../out_LED.csv')
df = og_df[ og_df['repeat_trial'].isin([0,2]) | og_df['repeat_trial'].isna() ]
session_type = 7    
df = df[ df['session_type'].isin([session_type]) ]
training_level = 16
df = df[ df['training_level'].isin([training_level]) ]

# drop rows from df where intended_fix, LED_onset_time and timed_fix are nan
df = df.dropna(subset=['intended_fix', 'LED_onset_time', 'timed_fix'])
df = df[(df['abort_event'] == 3) | (df['success'].isin([1,-1]))]

# %%
# Filter out aborts < 300ms
FILTER_300 = True
if FILTER_300:
    df = df[~( (df['abort_event'] == 3) & (df['timed_fix'] < 0.3) )]

# %%
# Get LED ON and OFF trials
df_on = df[df['LED_trial'] == 1]
df_on_1 = df_on.copy()
df_on_1['LED_wrt_fix'] = df_on_1['intended_fix'] - df_on_1['LED_onset_time']

df_off = df[df['LED_trial'] == 0]

# %%
# Plot LED timing distributions
bins = np.arange(0,2,0.05)
plt.hist(df_on_1['LED_wrt_fix'], bins=bins, histtype='step', density=True, label='LED time wrt fix')
plt.hist(df_on_1['intended_fix'], bins=bins, histtype='step', density=True, label='stim time wrt fix')
plt.xlabel('Time (s)')
plt.ylabel('Density')
plt.legend()
plt.title('LED and Stimulus Timing Distributions from Data')
plt.show()

# %%
# Simulate only proactive process (single bound)
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
# Parameters for simulation
V_A_base = 1.8
V_A_post_LED = 2.4
theta_A = 1.5
t_aff = 40*1e-3
t_effect = 35*1e-3
motor_delay = 50*1e-3
T_trunc = 0.6
N_sim = int(50e3)

# Get LED times from data
LED_times = df_on_1['LED_wrt_fix'].values
stim_times = df_on_1['intended_fix'].values
stim_times_off = df_off['intended_fix'].values

print(f"Number of LED ON trials in data: {len(df_on_1)}")
print(f"Number of LED OFF trials in data: {len(df_off)}")
print(f"Simulating {N_sim} trials...")

# %%
# Run simulation
def simulate_single_trial():
    # 1/3 chance of LED trial
    is_led_trial = np.random.random() < 1/3

    if is_led_trial:
        t_LED = np.random.choice(LED_times)
        t_stim = np.random.choice(stim_times)
    else:
        t_LED = None
        t_stim = np.random.choice(stim_times_off)

    rt = simulate_proactive_single_bound(
        V_A_base, V_A_post_LED, theta_A,
        t_LED,
        t_stim,
        t_aff,
        t_effect,
        motor_delay,
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

# %%
# VBMC part
# 1. loglike
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
# Test the likelihood
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


# Separate simulated RTs into LED ON and OFF, and apply truncation
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

# Compute theoretical distributions
dt = 0.01
t_max = 5.0
t_pts = np.arange(T_trunc, t_max, dt)

pdf_theory_on = compute_theoretical_rtd_on(
    t_pts, V_A_base, V_A_post_LED, theta_A, t_aff, t_effect, motor_delay, N_mc=1000, T_trunc=T_trunc
)
pdf_theory_off = compute_theoretical_rtd_off(
    t_pts, V_A_base, theta_A, t_aff, motor_delay, T_trunc=T_trunc
)

pdf_theory_on_norm = pdf_theory_on / np.trapz(pdf_theory_on, t_pts)
pdf_theory_off_norm = pdf_theory_off / np.trapz(pdf_theory_off, t_pts)

# Plot theoretical vs simulated histogram for LED ON
plt.figure(figsize=(12, 5))
bins = np.arange(T_trunc, t_max, dt)
sim_hist_on, _ = np.histogram(sim_rts_on, bins=bins, density=True)
bin_centers = (bins[1:] + bins[:-1]) / 2
plt.plot(bin_centers, sim_hist_on, label='simulated (LED ON)', lw=2, alpha=0.7)
plt.plot(t_pts, pdf_theory_on_norm, label='theoretical (LED ON)', lw=2, ls='--')
area_theory_on = np.trapz(pdf_theory_on_norm, t_pts)
print(f"Theoretical area (LED ON): {area_theory_on:.6f}")
plt.axvline(x=T_trunc, color='r', linestyle='--')
plt.xlabel('RT (s)')
plt.ylabel('Density')
plt.title('Theoretical vs Simulated RT Distribution (LED ON)')
plt.legend()
plt.show()

# Plot theoretical vs simulated histogram for LED OFF
plt.figure(figsize=(12, 5))
sim_hist_off, _ = np.histogram(sim_rts_off, bins=bins, density=True)
plt.plot(bin_centers, sim_hist_off, label='simulated (LED OFF)', lw=2, alpha=0.7)
plt.plot(t_pts, pdf_theory_off_norm, label='theoretical (LED OFF)', lw=2, ls='--')
area_theory_off = np.trapz(pdf_theory_off_norm, t_pts)
print(f"Theoretical area (LED OFF): {area_theory_off:.6f}")
plt.axvline(x=T_trunc, color='r', linestyle='--')
plt.xlabel('RT (s)')
plt.ylabel('Density')
plt.title('Theoretical vs Simulated RT Distribution (LED OFF)')
plt.legend()
plt.show()

# Censoring: fraction of trials after t_stim (simulated)
sim_after_t_stim_off = sum(1 for rt, t_stim in zip(sim_rts_off, sim_t_stims_off) if rt > t_stim)
frac_after_t_stim_off = sim_after_t_stim_off / len(sim_rts_off)
print(f"LED OFF - Fraction of trials after t_stim (simulated): {frac_after_t_stim_off:.6f}")
print(f"LED OFF - Total trials (after truncation): {len(sim_rts_off)}")

sim_after_t_stim_on = sum(1 for rt, t_stim in zip(sim_rts_on, sim_t_stims_on) if rt > t_stim)
frac_after_t_stim_on = sim_after_t_stim_on / len(sim_rts_on)
print(f"LED ON - Fraction of trials after t_stim (simulated): {frac_after_t_stim_on:.6f}")
print(f"LED ON - Total trials (after truncation): {len(sim_rts_on)}")

# Censoring: theoretical survival probability via Monte Carlo
N_mc = 1000

survival_off_samples = []
for _ in range(N_mc):
    t_stim = np.random.choice(stim_times_off)
    survival_off_samples.append(
        led_off_survival_truncated(t_stim, V_A_base, theta_A, t_aff, motor_delay, T_trunc)
    )

theoretical_survival_off = np.mean(survival_off_samples)
print(f"LED OFF - Fraction of trials after t_stim (theoretical): {theoretical_survival_off:.6f}")
print(f"LED OFF - Difference (sim - theory): {frac_after_t_stim_off - theoretical_survival_off:.6f}")

survival_on_samples = []
for _ in range(N_mc):
    t_led = np.random.choice(LED_times)
    t_stim = np.random.choice(stim_times)
    survival_on_samples.append(
        led_on_survival_truncated(t_stim, t_led, V_A_base, V_A_post_LED, theta_A, t_aff,
                                  motor_delay, t_effect, T_trunc)
    )

theoretical_survival_on = np.mean(survival_on_samples)
print(f"LED ON - Fraction of trials after t_stim (theoretical): {theoretical_survival_on:.6f}")
print(f"LED ON - Difference (sim - theory): {frac_after_t_stim_on - theoretical_survival_on:.6f}")

#%% 