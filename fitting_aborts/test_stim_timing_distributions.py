# %%
"""
Test script to verify if LED ON and LED OFF trials have different t_stim and t_LED distributions.
This will help diagnose why RT wrt LED doesn't match even when RT wrt fixation matches.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
# Load data (same as main script)
T_trunc = 0.3
og_df = pd.read_csv('../out_LED.csv')

df = og_df[og_df['repeat_trial'].isin([0, 2]) | og_df['repeat_trial'].isna()]
session_type = 7
df = df[df['session_type'].isin([session_type])]
training_level = 16
df = df[df['training_level'].isin([training_level])]

df = df.dropna(subset=['intended_fix', 'LED_onset_time', 'timed_fix'])
df = df[(df['abort_event'] == 3) | (df['success'].isin([1, -1]))]
df = df[~((df['abort_event'] == 3) & (df['timed_fix'] < T_trunc))]

df_all = df

# Separate LED ON and OFF
df_on = df_all[df_all['LED_trial'] == 1]
df_off = df_all[df_all['LED_trial'].isin([0, np.nan])]

print(f"LED ON trials: {len(df_on)}")
print(f"LED OFF trials: {len(df_off)}")

# %%
# Extract timing distributions for each condition
stim_times_on = df_on['intended_fix'].values
stim_times_off = df_off['intended_fix'].values
LED_times_on = (df_on['intended_fix'] - df_on['LED_onset_time']).values
LED_times_off = (df_off['intended_fix'] - df_off['LED_onset_time']).values

# Combined (what we're currently using)
stim_times_all = df_all['intended_fix'].values
LED_times_all = (df_all['intended_fix'] - df_all['LED_onset_time']).values

# %%
# Plot 1: Compare t_stim distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

bins_stim = np.arange(0, 3, 0.02)
axes[0].hist(stim_times_on, bins=bins_stim, density=True, histtype='step', lw=2, label=f'LED ON (n={len(stim_times_on)})')
axes[0].hist(stim_times_off, bins=bins_stim, density=True, histtype='step', lw=2, label=f'LED OFF (n={len(stim_times_off)})')
axes[0].hist(stim_times_all, bins=bins_stim, density=True, histtype='step', lw=2, ls='--', label=f'ALL (n={len(stim_times_all)})')
axes[0].set_xlabel('t_stim (intended_fix) [s]')
axes[0].set_ylabel('Density')
axes[0].set_title('t_stim distribution: LED ON vs OFF')
axes[0].legend()

# Plot 2: Compare t_LED distributions
axes[1].hist(LED_times_on, bins=bins_stim, density=True, histtype='step', lw=2, label=f'LED ON (n={len(LED_times_on)})')
axes[1].hist(LED_times_off, bins=bins_stim, density=True, histtype='step', lw=2, label=f'LED OFF (n={len(LED_times_off)})')
axes[1].hist(LED_times_all, bins=bins_stim, density=True, histtype='step', lw=2, ls='--', label=f'ALL (n={len(LED_times_all)})')
axes[1].set_xlabel('t_LED (intended_fix - LED_onset_time) [s]')
axes[1].set_ylabel('Density')
axes[1].set_title('t_LED distribution: LED ON vs OFF')
axes[1].legend()

plt.tight_layout()
plt.savefig('test_stim_timing_distributions.pdf')
plt.show()

# %%
# Print summary statistics
print("\n=== t_stim (intended_fix) ===")
print(f"LED ON:  mean={np.mean(stim_times_on):.3f}, std={np.std(stim_times_on):.3f}, median={np.median(stim_times_on):.3f}")
print(f"LED OFF: mean={np.mean(stim_times_off):.3f}, std={np.std(stim_times_off):.3f}, median={np.median(stim_times_off):.3f}")
print(f"ALL:     mean={np.mean(stim_times_all):.3f}, std={np.std(stim_times_all):.3f}, median={np.median(stim_times_all):.3f}")

print("\n=== t_LED (intended_fix - LED_onset_time) ===")
print(f"LED ON:  mean={np.mean(LED_times_on):.3f}, std={np.std(LED_times_on):.3f}, median={np.median(LED_times_on):.3f}")
print(f"LED OFF: mean={np.mean(LED_times_off):.3f}, std={np.std(LED_times_off):.3f}, median={np.median(LED_times_off):.3f}")
print(f"ALL:     mean={np.mean(LED_times_all):.3f}, std={np.std(LED_times_all):.3f}, median={np.median(LED_times_all):.3f}")

# %%
# Check if distributions are significantly different
from scipy import stats

ks_stim = stats.ks_2samp(stim_times_on, stim_times_off)
ks_led = stats.ks_2samp(LED_times_on, LED_times_off)

print("\n=== Kolmogorov-Smirnov test (LED ON vs OFF) ===")
print(f"t_stim: KS statistic={ks_stim.statistic:.4f}, p-value={ks_stim.pvalue:.2e}")
print(f"t_LED:  KS statistic={ks_led.statistic:.4f}, p-value={ks_led.pvalue:.2e}")

if ks_stim.pvalue < 0.05:
    print("\n*** t_stim distributions ARE significantly different! ***")
else:
    print("\n t_stim distributions are NOT significantly different")

if ks_led.pvalue < 0.05:
    print("*** t_LED distributions ARE significantly different! ***")
else:
    print(" t_LED distributions are NOT significantly different")

# %%
# Check joint distribution: scatter plot of (t_LED, RT) for aborts only
T_trunc = 0.3

# Get abort trials (RT < t_stim) with RT > T_trunc
df_on_aborts = df_on[(df_on['timed_fix'] < df_on['intended_fix']) & (df_on['timed_fix'] > T_trunc)]

rt_on = df_on_aborts['timed_fix'].values
t_led_on = (df_on_aborts['intended_fix'] - df_on_aborts['LED_onset_time']).values
rt_wrt_led_on = rt_on - t_led_on

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter: RT vs t_LED
axes[0].scatter(t_led_on, rt_on, alpha=0.1, s=1)
axes[0].plot([0, 2], [0, 2], 'r--', label='RT = t_LED')
axes[0].set_xlabel('t_LED (s)')
axes[0].set_ylabel('RT (s)')
axes[0].set_title('LED ON aborts: RT vs t_LED')
axes[0].legend()

# Scatter: RT wrt LED vs t_LED (to see if there's correlation)
axes[1].scatter(t_led_on, rt_wrt_led_on, alpha=0.1, s=1)
axes[1].axhline(y=0, color='r', ls='--', label='LED onset')
axes[1].set_xlabel('t_LED (s)')
axes[1].set_ylabel('RT - t_LED (s)')
axes[1].set_title('LED ON aborts: (RT - t_LED) vs t_LED')
axes[1].legend()

plt.tight_layout()
plt.savefig('test_rt_vs_tled_scatter.pdf')
plt.show()

# Check correlation
corr = np.corrcoef(t_led_on, rt_wrt_led_on)[0, 1]
print(f"\nCorrelation between t_LED and (RT - t_LED): {corr:.3f}")
if abs(corr) > 0.1:
    print("*** There IS correlation between t_LED and RT wrt LED! ***")
    print("This means the model needs to account for this dependency.")
# %%
