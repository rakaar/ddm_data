
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
# PARAMETERS
# =============================================================================
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

# Get unique animals (for info)
unique_animals = df['animal'].unique()
print(f"Aggregating all {len(unique_animals)} animals: {unique_animals}")

# Use all animals (no filtering by animal)
df_all = df

# Separate LED ON and OFF
df_on = df_all[df_all['LED_trial'] == 1]
df_off = df_all[df_all['LED_trial'].isin([0, np.nan])]

print(f"\nAll animals aggregated data summary:")
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
def simulate_proactive_single_bound_v1(V_A_base, V_A_post_LED, theta_A, t_LED, t_stim, del_a, del_LED, del_m, is_led_trial, dt=1e-4):
    """
    Simulate proactive process with single bound accumulator.
    Drift changes from V_A_base to V_A_post_LED at t_LED + del_LED (only for LED ON trials).
    Proactive process starts at t = del_a (no noise before this).
    Returns RT when accumulator hits theta_A.
    """
    AI = 0
    t = del_a
    dB = np.sqrt(dt)

    while True:
        if is_led_trial and t >= t_LED + del_LED:
            V_A = V_A_post_LED
        else:
            V_A = V_A_base

        AI += V_A * dt + np.random.normal(0, dB)
        t += dt

        if AI >= theta_A:
            RT = t + del_m
            return RT

# %%
def simulate_proactive_single_bound_v2(V_A_base, V_A_post_LED, theta_A, t_LED, t_stim, del_a_minus_del_LED, del_m_plus_del_LED, is_led_trial, dt=1e-4):
    """
    Simulate proactive process with single bound accumulator.
    Drift changes from V_A_base to V_A_post_LED at t_LED + t_effect (only for LED ON trials).
    Proactive process starts at t = t_aff (no noise before this).
    Returns RT when accumulator hits theta_A.
    """
    AI = 0
    t = 0 # counting from zero instead of delta a
    dB = np.sqrt(dt)

    while True:
        if is_led_trial and t >= t_LED - del_a_minus_del_LED:
            V_A = V_A_post_LED
        else:
            V_A = V_A_base

        AI += V_A * dt + np.random.normal(0, dB)
        t += dt

        if AI >= theta_A:
            RT = t + (del_m_plus_del_LED + del_a_minus_del_LED) # after hitting bound adding the del a and del LED
            return RT

# %%
# sample params 

V_A_base_true = 1.8
V_A_post_LED_true = 2.4
theta_A_true = 1.5
del_a_true = 100*1e-3
del_LED_true = 80*1e-3
del_m_true = 50*1e-3
T_trunc = 0.7
# V2 parameterization assumes del_a_minus_del_LED = del_a - del_LED and del_m_plus_del_LED = del_m + del_LED
del_a_minus_del_LED_true = del_a_true - del_LED_true
del_m_plus_del_LED_true = del_m_true + del_LED_true
N_trials_sim = int(200e3)
# %%
def simulate_single_trial_fit():
    is_led_trial = np.random.random() < 1/2
    # Sample trial index to preserve (t_LED, t_stim) correlation
    trial_idx = np.random.randint(n_trials_data)
    t_LED = LED_times[trial_idx]
    t_stim = stim_times[trial_idx]
    rt_v1 = simulate_proactive_single_bound_v1(
        V_A_base_true, V_A_post_LED_true, theta_A_true,
        t_LED, t_stim,
        del_a_true, del_LED_true, del_m_true,
        is_led_trial
    )
    rt_v2 = simulate_proactive_single_bound_v2(
        V_A_base_true, V_A_post_LED_true, theta_A_true,
        t_LED, t_stim,
        del_a_minus_del_LED_true,
        del_m_plus_del_LED_true,
        is_led_trial
    )
    return rt_v1, rt_v2, is_led_trial, t_LED, t_stim

sim_results = Parallel(n_jobs=30)(
    delayed(simulate_single_trial_fit)() for _ in range(N_trials_sim)
)

sim_rts_v1 = [r[0] for r in sim_results]
sim_rts_v2 = [r[1] for r in sim_results]
sim_is_led = [r[2] for r in sim_results]
sim_t_LEDs = [r[3] for r in sim_results]
sim_t_stims = [r[4] for r in sim_results]

# Separate into LED ON/OFF, apply truncation AND filter aborts only (RT < t_stim) to match data
sim_rts_on_v1 = [rt for rt, is_led, t_stim in zip(sim_rts_v1, sim_is_led, sim_t_stims) if is_led and rt > T_trunc and rt < t_stim]
sim_rts_off_v1 = [rt for rt, is_led, t_stim in zip(sim_rts_v1, sim_is_led, sim_t_stims) if not is_led and rt > T_trunc and rt < t_stim]
sim_rts_on_v2 = [rt for rt, is_led, t_stim in zip(sim_rts_v2, sim_is_led, sim_t_stims) if is_led and rt > T_trunc and rt < t_stim]
sim_rts_off_v2 = [rt for rt, is_led, t_stim in zip(sim_rts_v2, sim_is_led, sim_t_stims) if not is_led and rt > T_trunc and rt < t_stim]

# For RT wrt LED plots (abort rate), same filtering
sim_rts_wrt_led_on_v1 = [rt - t_led for rt, is_led, t_led, t_stim in zip(sim_rts_v1, sim_is_led, sim_t_LEDs, sim_t_stims) if is_led and rt > T_trunc and rt < t_stim]
sim_rts_wrt_led_off_v1 = [rt - t_led for rt, is_led, t_led, t_stim in zip(sim_rts_v1, sim_is_led, sim_t_LEDs, sim_t_stims) if not is_led and rt > T_trunc and rt < t_stim]
sim_rts_wrt_led_on_v2 = [rt - t_led for rt, is_led, t_led, t_stim in zip(sim_rts_v2, sim_is_led, sim_t_LEDs, sim_t_stims) if is_led and rt > T_trunc and rt < t_stim]
sim_rts_wrt_led_off_v2 = [rt - t_led for rt, is_led, t_led, t_stim in zip(sim_rts_v2, sim_is_led, sim_t_LEDs, sim_t_stims) if not is_led and rt > T_trunc and rt < t_stim]

print(f"Simulation v1 (aborts only): {len(sim_rts_on_v1)} LED ON, {len(sim_rts_off_v1)} LED OFF")
print(f"Simulation v2 (aborts only): {len(sim_rts_on_v2)} LED ON, {len(sim_rts_off_v2)} LED OFF")

# %%
# =============================================================================
# Plot: RT wrt LED - Abort rate (area-weighted), v1 vs v2
# =============================================================================
n_all_sim_on = sum(1 for is_led in sim_is_led if is_led)
n_all_sim_off = sum(1 for is_led in sim_is_led if not is_led)
n_aborts_sim_on_v1 = sum(1 for rt, is_led, t_stim in zip(sim_rts_v1, sim_is_led, sim_t_stims) if is_led and rt < t_stim and rt > T_trunc)
n_aborts_sim_off_v1 = sum(1 for rt, is_led, t_stim in zip(sim_rts_v1, sim_is_led, sim_t_stims) if not is_led and rt < t_stim and rt > T_trunc)
n_aborts_sim_on_v2 = sum(1 for rt, is_led, t_stim in zip(sim_rts_v2, sim_is_led, sim_t_stims) if is_led and rt < t_stim and rt > T_trunc)
n_aborts_sim_off_v2 = sum(1 for rt, is_led, t_stim in zip(sim_rts_v2, sim_is_led, sim_t_stims) if not is_led and rt < t_stim and rt > T_trunc)
frac_sim_on_v1 = n_aborts_sim_on_v1 / n_all_sim_on if n_all_sim_on > 0 else 0
frac_sim_off_v1 = n_aborts_sim_off_v1 / n_all_sim_off if n_all_sim_off > 0 else 0
frac_sim_on_v2 = n_aborts_sim_on_v2 / n_all_sim_on if n_all_sim_on > 0 else 0
frac_sim_off_v2 = n_aborts_sim_off_v2 / n_all_sim_off if n_all_sim_off > 0 else 0
# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
bins_wrt_led = np.arange(-3, 3, 0.05)
bin_centers_wrt_led = (bins_wrt_led[1:] + bins_wrt_led[:-1]) / 2

sim_hist_on_wrt_led_dens_v1, _ = np.histogram(sim_rts_wrt_led_on_v1, bins=bins_wrt_led, density=True)
sim_hist_off_wrt_led_dens_v1, _ = np.histogram(sim_rts_wrt_led_off_v1, bins=bins_wrt_led, density=True)
sim_hist_on_scaled_v1 = sim_hist_on_wrt_led_dens_v1 * frac_sim_on_v1
sim_hist_off_scaled_v1 = sim_hist_off_wrt_led_dens_v1 * frac_sim_off_v1

sim_hist_on_wrt_led_dens_v2, _ = np.histogram(sim_rts_wrt_led_on_v2, bins=bins_wrt_led, density=True)
sim_hist_off_wrt_led_dens_v2, _ = np.histogram(sim_rts_wrt_led_off_v2, bins=bins_wrt_led, density=True)
sim_hist_on_scaled_v2 = sim_hist_on_wrt_led_dens_v2 * frac_sim_on_v2
sim_hist_off_scaled_v2 = sim_hist_off_wrt_led_dens_v2 * frac_sim_off_v2

axes[0].plot(bin_centers_wrt_led, sim_hist_on_scaled_v1, label=f'V1 (frac={frac_sim_on_v1:.2f})', lw=2, alpha=0.7, color='r', linestyle='-')
axes[0].plot(bin_centers_wrt_led, sim_hist_on_scaled_v2, label=f'V2 (frac={frac_sim_on_v2:.2f})', lw=2, alpha=0.7, color='k', linestyle='--')
axes[0].axvline(x=0, color='k', linestyle='--', alpha=0.5, label='LED onset')
axes[0].axvline(x=del_LED_true, color='g', linestyle=':', alpha=0.5, label=f'del_LED={del_LED_true:.2f}')
axes[0].set_xlabel('RT - t_LED (s)', fontsize=12)
axes[0].set_ylabel('Rate (area = fraction)', fontsize=12)
axes[0].set_title('LED ON', fontsize=14)
axes[0].legend(fontsize=9)

axes[1].plot(bin_centers_wrt_led, sim_hist_off_scaled_v1, label=f'V1 (frac={frac_sim_off_v1:.2f})', lw=2, alpha=0.7, color='b', linestyle='-')
axes[1].plot(bin_centers_wrt_led, sim_hist_off_scaled_v2, label=f'V2 (frac={frac_sim_off_v2:.2f})', lw=2, alpha=0.7, color='k', linestyle='--')
axes[1].axvline(x=0, color='k', linestyle='--', alpha=0.5, label='LED onset')
axes[1].axvline(x=del_LED_true, color='g', linestyle=':', alpha=0.5, label=f'del_LED={del_LED_true:.2f}')
axes[1].set_xlabel('RT - t_LED (s)', fontsize=12)
axes[1].set_title('LED OFF', fontsize=14)
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.show()

# %%