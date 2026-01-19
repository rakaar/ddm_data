# %%
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
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
df_off = df[df['LED_trial'].isin([0, np.nan])]

df_on_aborts = df_on[df_on['abort_event'] == 3]
df_off_aborts = df_off[df_off['abort_event'] == 3]

df_on_abort_times = df_on_aborts['timed_fix'] - (df_on_aborts['intended_fix'] - df_on_aborts['LED_onset_time'])
df_off_abort_times = df_off_aborts['timed_fix'] - (df_off_aborts['intended_fix'] - df_off_aborts['LED_onset_time'])

frac_on_aborts = len(df_on_aborts) / len(df_on) 
frac_off_aborts = len(df_off_aborts) / len(df_off)

bins = np.arange(-2, 2, 0.05)
on_aborts_time_hist, _ = np.histogram(df_on_abort_times, bins=bins, density=True)
off_aborts_time_hist, _ = np.histogram(df_off_abort_times, bins=bins, density=True)

on_aborts_rate = on_aborts_time_hist * frac_on_aborts
off_aborts_rate = off_aborts_time_hist * frac_off_aborts 

plt.plot(bins[:-1], on_aborts_rate, color='red')
plt.plot(bins[:-1], off_aborts_rate, color='blue')
plt.axvline(x=0, color='black', ls='--', alpha=0.3)
plt.title('Abort rate wrt LED onset')
plt.xlabel('Abort Time wrt LED onset')
plt.ylabel('Abort rate')
plt.show()


# %%
# unique_animals = df['animal'].unique()[::-1]
unique_animals = df['animal'].unique()

print(f'unique_animals = {unique_animals}')

fig, axes = plt.subplots(1, len(unique_animals), figsize=(5 * len(unique_animals), 5), sharey=True)

all_on_aborts_rates = []
all_off_aborts_rates = []

for i, animal in enumerate(unique_animals):
    df_animal = df[df['animal'] == animal]
    df_on = df_animal[df_animal['LED_trial'] == 1]
    df_off = df_animal[df_animal['LED_trial'].isin([0, np.nan])]

    df_on_aborts = df_on[df_on['abort_event'] == 3]
    df_off_aborts = df_off[df_off['abort_event'] == 3]

    df_on_abort_times = df_on_aborts['timed_fix'] - (df_on_aborts['intended_fix'] - df_on_aborts['LED_onset_time'])
    df_off_abort_times = df_off_aborts['timed_fix'] - (df_off_aborts['intended_fix'] - df_off_aborts['LED_onset_time'])

    frac_on_aborts = len(df_on_aborts) / len(df_on) if len(df_on) > 0 else 0
    frac_off_aborts = len(df_off_aborts) / len(df_off) if len(df_off) > 0 else 0

    bins = np.arange(-2, 2, 0.05)
    on_aborts_time_hist, _ = np.histogram(df_on_abort_times, bins=bins, density=True)
    off_aborts_time_hist, _ = np.histogram(df_off_abort_times, bins=bins, density=True)

    on_aborts_rate = on_aborts_time_hist * frac_on_aborts
    off_aborts_rate = off_aborts_time_hist * frac_off_aborts

    all_on_aborts_rates.append(on_aborts_rate)
    all_off_aborts_rates.append(off_aborts_rate)

    ax = axes[i]
    ax.plot(bins[:-1], on_aborts_rate, color='red')
    ax.plot(bins[:-1], off_aborts_rate, color='blue')
    ax.axvline(x=0, color='black', ls='--', alpha=0.3)
    
    peak_idx = np.argmax(on_aborts_rate)
    peak_time = bins[:-1][peak_idx]
    # ax.axvline(x=peak_time, color='green', ls=':', linewidth=1, alpha=0.4)
    
    A1 = np.trapz(off_aborts_rate[bins[:-1] > 0], bins[:-1][bins[:-1] > 0])
    A2 = np.trapz(on_aborts_rate[bins[:-1] > 0], bins[:-1][bins[:-1] > 0])
    pct_increase = (A2 - A1) / A1 * 100 if A1 > 0 else np.nan
    
    # ax.set_title(f'Animal {animal} (peak: {peak_time * 1000:.1f} ms)')
    ax.set_title(f'Animal {animal}\nA1={A1:.2f}, A2={A2:.2f}, +{pct_increase:.2f}%')

    ax.set_xlabel('Abort Time wrt LED onset')
    if i == 0:
        ax.set_ylabel('Abort rate')

plt.suptitle('Abort rate wrt LED onset')
plt.tight_layout()
plt.show()

# %%
avg_on_aborts_rate = np.mean(all_on_aborts_rates, axis=0)
avg_off_aborts_rate = np.mean(all_off_aborts_rates, axis=0)

plt.figure()
plt.plot(bins[:-1], avg_on_aborts_rate, color='red', label='LED ON')
plt.plot(bins[:-1], avg_off_aborts_rate, color='blue', label='LED OFF')
plt.axvline(x=0, color='black', ls='--', alpha=0.3)
plt.title('Average abort rate wrt LED onset (across animals)')
plt.xlabel('Abort Time wrt LED onset')
plt.ylabel('Abort rate')
plt.legend()
plt.show()
