# %%
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
import sys
sys.path.append('../fit_each_condn')
from psiam_tied_dv_map_utils_with_PDFs import stupid_f_integral, d_A_RT
# %%
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
FILTER_300 = True
if FILTER_300:
    df = df[~( (df['abort_event'] == 3) & (df['timed_fix'] < 0.3) )]
# %%
# aggregate
df_on = df[df['LED_trial'] == 1]
# %%
df_on_1 = df_on.copy()
df_on_1['LED_wrt_fix'] = df_on_1['intended_fix'] - df_on_1['LED_onset_time']
# %%
bins = np.arange(0,2,0.05)
plt.hist(df_on_1['LED_wrt_fix'], bins=bins, histtype='step', density=True)
plt.hist(df_on_1['intended_fix'], bins=bins, histtype='step', density=True)
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
N_sim = int(50e3)

# Get LED times from data
LED_times = df_on_1['LED_wrt_fix'].values
stim_times = df_on_1['intended_fix'].values

# Get LED OFF trials
df_off = df[df['LED_trial'] == 0]
stim_times_off = df_off['intended_fix'].values

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
    return rt, is_led_trial

sim_results = Parallel(n_jobs=30)(
    delayed(simulate_single_trial)() for _ in tqdm(range(N_sim))
)
sim_rts = [r[0] for r in sim_results]
sim_is_led_trials = [r[1] for r in sim_results]

# %%
# Plot results
plt.figure(figsize=(12, 5))

plt.hist(sim_rts, bins=100, density=True, alpha=0.7, label='simulated')
plt.xlabel('RT (s)')
plt.ylabel('Density')
plt.title('Proactive Process RT Distribution')
plt.legend()
plt.show()

# %%
# Adapted theoretical PA function with separate t_aff and motor_delay, and t_effect
def PA_with_LEDON_2_adapted(t, v, vON, a, t_aff, motor_delay, tled, t_effect):
    """
    Compute the PA pdf by combining contributions before and after LED onset.

    Parameters:
        t (float): Time value (RT - motor_delay).
        v (float): Drift parameter before LED (V_A_base).
        vON (float): Drift parameter after LED onset (V_A_post_LED).
        a (float): Decision bound (theta_A).
        t_aff (float): Proactive afferent delay.
        motor_delay (float): Motor delay.
        tled (float): LED time relative to fixation start.
        t_effect (float): Time after LED when drift changes.

    Returns:
        float: The combined PA pdf value.
    """
    # Theoretical time starts at 0, simulation starts at t_aff
    # So we need to add t_aff to t when comparing to tled
    if (t - motor_delay) <= (tled + t_effect):
        # Before LED drift change:
        return d_A_RT(v * a, (t - motor_delay - t_aff) / (a**2)) / (a**2)
    else:
        # After LED drift change:
        t_post_led = t - motor_delay - tled - t_effect
        tp = tled + t_effect - t_aff

        if tp <= 0:
            # tled + t_effect < t_aff
            # delay is so large that only post LED process started
            return d_A_RT(vON * a, (t - motor_delay - t_aff) / (a**2)) / (a**2)
        
        # if t_post_led <= 0:
            # the case is already covered above
            # its basically, pre-LED process only

        try:
            return stupid_f_integral(v, vON, a, t_post_led, tp)
        except ValueError as e:
            print(f"ERROR in stupid_f_integral:")
            print(f"  t (RT): {t}")
            print(f"  v (V_A_base): {v}")
            print(f"  vON (V_A_post_LED): {vON}")
            print(f"  a (theta_A): {a}")
            print(f"  t_aff: {t_aff}")
            print(f"  motor_delay: {motor_delay}")
            print(f"  tled (t_LED): {tled}")
            print(f"  t_effect: {t_effect}")
            print(f"  t_post_led: {t_post_led}")
            print(f"  tp: {tp}")
            print(f"  t_post_led <= 0: {t_post_led <= 0}")
            print(f"  tp <= 0: {tp <= 0}")
            raise

# %%
# Monte Carlo function to compute theoretical RT distribution
def compute_theoretical_RT_distribution(V_A_base, V_A_post_LED, theta_A, t_aff, t_effect, motor_delay,
                                        N_mc=1000, t_max=2.0, dt=0.001):
    """
    Compute theoretical RT distribution by averaging over Monte Carlo samples of (t_stim, t_LED).

    Parameters:
        V_A_base, V_A_post_LED, theta_A, t_aff, t_effect, motor_delay: Model parameters
        N_mc: Number of Monte Carlo samples
        t_max: Maximum RT to consider
        dt: Time step for PDF evaluation

    Returns:
        t_pts: Time points
        pdf_mean_on: Mean PDF for LED ON trials
        pdf_mean_off: Mean PDF for LED OFF trials
    """
    t_pts = np.arange(0, t_max, dt)
    pdf_samples_on = np.zeros((N_mc, len(t_pts)))
    pdf_samples_off = np.zeros((N_mc, len(t_pts)))

    # Sample (t_stim, t_LED) pairs
    for i in range(N_mc):
        is_led_trial = np.random.random() < 1/3
        if is_led_trial:
            t_LED = np.random.choice(LED_times)
            t_stim = np.random.choice(stim_times)

            # Compute PDF for LED ON trial
            for j, t in enumerate(t_pts):
                pdf_samples_on[i, j] = PA_with_LEDON_2_adapted(t, V_A_base, V_A_post_LED, theta_A,
                                                                t_aff, motor_delay, t_LED, t_effect)
        else:
            t_LED = None
            t_stim = np.random.choice(stim_times_off)

            # Compute PDF for LED OFF trial (just base drift)
            for j, t in enumerate(t_pts):
                pdf_samples_off[i, j] = d_A_RT(V_A_base * theta_A, (t - motor_delay - t_aff) / (theta_A**2)) / (theta_A**2)

    pdf_mean_on = np.mean(pdf_samples_on, axis=0)
    pdf_mean_off = np.mean(pdf_samples_off, axis=0)
    return t_pts, pdf_mean_on, pdf_mean_off

# %%
# Compute theoretical distribution
t_pts, pdf_theory_on, pdf_theory_off = compute_theoretical_RT_distribution(
    V_A_base, V_A_post_LED, theta_A, t_aff, t_effect, motor_delay,
    N_mc=1000, t_max=5.0, dt=0.01
)

# %%
# Separate simulated RTs into LED ON and OFF
sim_rts_on = [rt for rt, is_led in zip(sim_rts, sim_is_led_trials) if is_led]
sim_rts_off = [rt for rt, is_led in zip(sim_rts, sim_is_led_trials) if not is_led]

# %%
# Normalize theoretical PDFs
pdf_theory_on_norm = pdf_theory_on / np.trapz(pdf_theory_on, t_pts)
pdf_theory_off_norm = pdf_theory_off / np.trapz(pdf_theory_off, t_pts)

# %%
# Plot theoretical vs simulated histogram for LED ON
plt.figure(figsize=(12, 5))

# Simulated histogram
bins = np.arange(0, 5, 0.01)
sim_hist_on, _ = np.histogram(sim_rts_on, bins=bins, density=True)
bin_centers = (bins[1:] + bins[:-1]) / 2
plt.plot(bin_centers, sim_hist_on, label='simulated (LED ON)', lw=2, alpha=0.7)

# Theoretical PDF
plt.plot(t_pts, pdf_theory_on_norm, label='theoretical (LED ON)', lw=2, ls='--')

# Compute and print theoretical area
area_theory_on = np.trapz(pdf_theory_on_norm, t_pts)
print(f"Theoretical area (LED ON): {area_theory_on:.6f}")

plt.xlabel('RT (s)')
plt.ylabel('Density')
plt.title('Theoretical vs Simulated RT Distribution (LED ON)')
plt.legend()
plt.show()

# %%
# Plot theoretical vs simulated histogram for LED OFF
plt.figure(figsize=(12, 5))

# Simulated histogram
sim_hist_off, _ = np.histogram(sim_rts_off, bins=bins, density=True)
plt.plot(bin_centers, sim_hist_off, label='simulated (LED OFF)', lw=2, alpha=0.7)

# Theoretical PDF
plt.plot(t_pts, pdf_theory_off_norm, label='theoretical (LED OFF)', lw=2, ls='--')

# Compute and print theoretical area
area_theory_off = np.trapz(pdf_theory_off_norm, t_pts)
print(f"Theoretical area (LED OFF): {area_theory_off:.6f}")

plt.xlabel('RT (s)')
plt.ylabel('Density')
plt.title('Theoretical vs Simulated RT Distribution (LED OFF)')
plt.legend()
plt.show()

# %%
