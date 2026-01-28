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

data_rts_on = fit_df[(fit_df['LED_trial'] == 1) & (fit_df['RT'] < fit_df['t_stim']) & (fit_df['RT'] > T_trunc)]['RT'].values
data_rts_off = fit_df[(fit_df['LED_trial'] == 0) & (fit_df['RT'] < fit_df['t_stim']) & (fit_df['RT'] > T_trunc)]['RT'].values

# RT wrt LED for data (same filtering)
df_on_aborts = fit_df[(fit_df['LED_trial'] == 1) & (fit_df['RT'] < fit_df['t_stim']) & (fit_df['RT'] > T_trunc)]
df_off_aborts = fit_df[(fit_df['LED_trial'] == 0) & (fit_df['RT'] < fit_df['t_stim']) & (fit_df['RT'] > T_trunc)]

data_rts_wrt_led_on = (df_on_aborts['RT'] - df_on_aborts['t_LED']).values
# For LED OFF, use actual t_LED from data (same as aborts_animal_wise_explore.py)
data_rts_wrt_led_off = (df_off_aborts['RT'] - df_off_aborts['t_LED']).values


# %%
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
param_means = [0,0,0,0,0,0]
param_means[0] = 1.6
param_means[1] = 3.4
param_means[2] = 2.5

param_means[3] = -0.187 - 0.0336
# param_means[3] = -0.3

param_means[4] = 0.0
param_means[5] = 0.04
N_trials_sim = int(100e3)
print(f"\nSimulating {N_trials_sim} trials with fitted parameters...")

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

# %%

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
bins_wrt_led = np.arange(-2,2 , 0.05)
bin_centers_wrt_led = (bins_wrt_led[1:] + bins_wrt_led[:-1]) / 2
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
ax.set_title('RT wrt LED (area-weighted) - All Animals', fontsize=14)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('vbmc_real_all_animals_rt_wrt_led_rate.pdf', bbox_inches='tight')
print("RT wrt LED rate plot saved as 'vbmc_real_all_animals_rt_wrt_led_rate.pdf'")
plt.show()
# %%

# %%
# OFF
## data
data_abort_wrt_fix = df_off_aborts['RT']
data_N_aborts_off = len(df_off_aborts)/ len(df_off)

# sim   
# sim_rts_off
bins = np.arange(0,2,0.05)
bin_centers = (bins[1:] + bins[:-1]) / 2

# data aborts wrt fix
data_hist_off_wrt_fix_dens, _ = np.histogram(data_abort_wrt_fix, bins=bins, density=True)
data_hist_off_wrt_fix_scaled = data_hist_off_wrt_fix_dens * data_N_aborts_off

# sim aborts wrt fix
sim_hist_off_wrt_fix_dens, _ = np.histogram(sim_rts_off, bins=bins, density=True)
sim_hist_off_wrt_fix_scaled = sim_hist_off_wrt_fix_dens * frac_sim_off
plt.plot(bin_centers, sim_hist_off_wrt_fix_scaled, label='Sim LED OFF (frac=1/3)', lw=2, alpha=0.7, color='b', linestyle='--')
plt.plot(bin_centers, data_hist_off_wrt_fix_scaled, label='Data LED OFF (frac=1/3)', lw=2, alpha=0.7, color='r', linestyle='-')
plt.legend()
plt.show()

print(data_N_aborts_off)
print((len(sim_rts_off)/ (len(sim_rts)*1/3)))

print(frac_sim_off)
# %%
print(frac_data_off)

# %%
y = df_off_aborts['t_LED']
x = df_on_fit['t_LED'].values

bins=np.arange(0,3,0.02)
bin_centers = (bins[1:] + bins[:-1]) / 2

plt.hist(x,bins=bins,density=True, histtype='step', label='y')
plt.hist(y,bins=bins,density=True, histtype='step', label='x')

# remove zeros from y
y = y[y > 0]
plt.hist(y,bins=bins,density=True, histtype='step', label='y-0', ls='--')  
plt.legend()
plt.show()