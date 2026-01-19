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


df_led_on = df[df['LED_trial'] == 1]
df_led_on_valid_and_aborts = df_led_on[(df_led_on['abort_event'] == 3) | (df_led_on['success'].isin([1,-1]))]
df_led_on_aborts = df_led_on_valid_and_aborts[df_led_on_valid_and_aborts['abort_event'] == 3]


df_led_off = df[df['LED_trial'].isin([0, np.nan])]
df_led_off_valid_and_aborts = df_led_off[(df_led_off['abort_event'] == 3) | (df_led_off['success'].isin([1,-1]))]
df_led_off_aborts = df_led_off_valid_and_aborts[df_led_off_valid_and_aborts['abort_event'] == 3]





# %%
unique_animals = df_led_on_aborts['animal'].unique()
print(f'unique_animals = {unique_animals}')
# %%
fig, axes = plt.subplots(1, len(unique_animals), figsize=(5 * len(unique_animals), 5), sharey=True)

for i, animal in enumerate(unique_animals):
    # on
    df_on_valid_and_aborts_animal = df_led_on_valid_and_aborts[df_led_on_valid_and_aborts['animal'] == animal]
    df_on_aborts_animal = df_led_on_aborts[df_led_on_aborts['animal'] == animal]

    df_off_valid_and_aborts_animal = df_led_off_valid_and_aborts[df_led_off_valid_and_aborts['animal'] == animal]
    df_off_aborts_animal = df_led_off_aborts[df_led_off_aborts['animal'] == animal]

    bins = np.arange(0, 2, 0.05)
    on_aborts_hist, _ = np.histogram(df_on_aborts_animal['timed_fix'], bins=bins, density=True)
    off_aborts_hist, _ = np.histogram(df_off_aborts_animal['timed_fix'], bins=bins, density=True)

    # frac aborts animal
    frac_on_aborts_animal = len(df_on_aborts_animal) / len(df_on_valid_and_aborts_animal)
    frac_off_aborts_animal = len(df_off_aborts_animal) / len(df_off_valid_and_aborts_animal)

    ax = axes[i]
    ax.plot(bins[:-1], on_aborts_hist * frac_on_aborts_animal, color='red')
    ax.plot(bins[:-1], off_aborts_hist * frac_off_aborts_animal, color='blue')
    ax.set_title(f'Animal {animal}')
    ax.set_xlabel('timed_fix')
    if i == 0:
        ax.set_ylabel('Density')

plt.suptitle('No filter < 300ms aborts')
plt.tight_layout()
plt.show()

# %%
# remove aborts that are < 300ms
df_led_on_valid_and_aborts_filter = df_led_on_valid_and_aborts[~((df_led_on_valid_and_aborts['timed_fix'] < 0.3) & (df_led_on_valid_and_aborts['abort_event'] == 3))]
df_led_off_valid_and_aborts_filter = df_led_off_valid_and_aborts[~((df_led_off_valid_and_aborts['timed_fix'] < 0.3) & (df_led_off_valid_and_aborts['abort_event'] == 3))]

df_led_on_aborts_filter = df_led_on_valid_and_aborts_filter[df_led_on_valid_and_aborts_filter['abort_event'] == 3]
df_led_off_aborts_filter = df_led_off_valid_and_aborts_filter[df_led_off_valid_and_aborts_filter['abort_event'] == 3]

# %%
fig, axes = plt.subplots(1, len(unique_animals), figsize=(5 * len(unique_animals), 5), sharey=True)

for i, animal in enumerate(unique_animals):
    # on
    df_on_valid_and_aborts_animal = df_led_on_valid_and_aborts_filter[df_led_on_valid_and_aborts_filter['animal'] == animal]
    df_on_aborts_animal = df_led_on_aborts_filter[df_led_on_aborts_filter['animal'] == animal]

    df_off_valid_and_aborts_animal = df_led_off_valid_and_aborts_filter[df_led_off_valid_and_aborts_filter['animal'] == animal]
    df_off_aborts_animal = df_led_off_aborts_filter[df_led_off_aborts_filter['animal'] == animal]

    bins = np.arange(0, 2, 0.05)
    on_aborts_hist, _ = np.histogram(df_on_aborts_animal['timed_fix'], bins=bins, density=True)
    off_aborts_hist, _ = np.histogram(df_off_aborts_animal['timed_fix'], bins=bins, density=True)

    # frac aborts animal
    frac_on_aborts_animal = len(df_on_aborts_animal) / len(df_on_valid_and_aborts_animal)
    frac_off_aborts_animal = len(df_off_aborts_animal) / len(df_off_valid_and_aborts_animal)

    ax = axes[i]
    ax.plot(bins[:-1], on_aborts_hist * frac_on_aborts_animal, color='red')
    ax.plot(bins[:-1], off_aborts_hist * frac_off_aborts_animal, color='blue')
    ax.set_title(f'Animal {animal} (filtered)')
    ax.set_xlabel('timed_fix')
    if i == 0:
        ax.set_ylabel('Density')

plt.suptitle('Filter < 300ms aborts')
plt.tight_layout()
plt.show()
# %%
# now plot wrt LED_onset_time, bins = np.arange(-2,2,0.05)
fig, axes = plt.subplots(1, len(unique_animals), figsize=(5 * len(unique_animals), 5), sharey=True)

for i, animal in enumerate(unique_animals):
    # on
    df_on_valid_and_aborts_animal = df_led_on_valid_and_aborts[df_led_on_valid_and_aborts['animal'] == animal]
    df_on_aborts_animal = df_led_on_aborts[df_led_on_aborts['animal'] == animal]

    df_off_valid_and_aborts_animal = df_led_off_valid_and_aborts[df_led_off_valid_and_aborts['animal'] == animal]
    df_off_aborts_animal = df_led_off_aborts[df_led_off_aborts['animal'] == animal]

    bins = np.arange(-2, 2, 0.05)
    on_aborts_hist, _ = np.histogram(df_on_aborts_animal['timed_fix'] + df_on_aborts_animal['LED_onset_time'] - df_on_aborts_animal['intended_fix'], bins=bins, density=True)
    off_aborts_hist, _ = np.histogram(df_off_aborts_animal['timed_fix'] + df_off_aborts_animal['LED_onset_time'] - df_off_aborts_animal['intended_fix'], bins=bins, density=True)

    # frac aborts animal
    frac_on_aborts_animal = len(df_on_aborts_animal) / len(df_on_valid_and_aborts_animal)
    frac_off_aborts_animal = len(df_off_aborts_animal) / len(df_off_valid_and_aborts_animal)

    ax = axes[i]
    ax.plot(bins[:-1], on_aborts_hist * frac_on_aborts_animal, color='red')
    ax.plot(bins[:-1], off_aborts_hist * frac_off_aborts_animal, color='blue')
    ax.set_title(f'Animal {animal}')
    ax.set_xlabel('timed_fix - LED_onset_time - intended_fix')
    if i == 0:
        ax.set_ylabel('Density')

plt.suptitle('No filter < 300ms aborts (LED onset aligned)')
plt.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(1, len(unique_animals), figsize=(5 * len(unique_animals), 5), sharey=True)

for i, animal in enumerate(unique_animals):
    # on
    df_on_valid_and_aborts_animal = df_led_on_valid_and_aborts_filter[df_led_on_valid_and_aborts_filter['animal'] == animal]
    df_on_aborts_animal = df_led_on_aborts_filter[df_led_on_aborts_filter['animal'] == animal]

    df_off_valid_and_aborts_animal = df_led_off_valid_and_aborts_filter[df_led_off_valid_and_aborts_filter['animal'] == animal]
    df_off_aborts_animal = df_led_off_aborts_filter[df_led_off_aborts_filter['animal'] == animal]

    bins = np.arange(-2, 2, 0.05)
    on_aborts_hist, _ = np.histogram(df_on_aborts_animal['timed_fix'] + df_on_aborts_animal['LED_onset_time'] - df_on_aborts_animal['intended_fix'], bins=bins, density=True)
    off_aborts_hist, _ = np.histogram(df_off_aborts_animal['timed_fix'] + df_off_aborts_animal['LED_onset_time'] - df_off_aborts_animal['intended_fix'], bins=bins, density=True)

    # frac aborts animal
    frac_on_aborts_animal = len(df_on_aborts_animal) / len(df_on_valid_and_aborts_animal)
    frac_off_aborts_animal = len(df_off_aborts_animal) / len(df_off_valid_and_aborts_animal)

    ax = axes[i]
    ax.plot(bins[:-1], on_aborts_hist * frac_on_aborts_animal, color='red')
    ax.plot(bins[:-1], off_aborts_hist * frac_off_aborts_animal, color='blue')
    ax.set_title(f'Animal {animal} (filtered)')
    ax.set_xlabel('timed_fix - LED_onset_time - intended_fix')
    if i == 0:
        ax.set_ylabel('Density')

plt.suptitle('Filter < 300ms aborts (LED onset aligned)')
plt.tight_layout()
plt.show()
# %%
# aggregate, no animal wise seperat
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

bins = np.arange(-2, 2, 0.05)
on_aborts_hist, _ = np.histogram(df_led_on_aborts['timed_fix'] + df_led_on_aborts['LED_onset_time'] - df_led_on_aborts['intended_fix'], bins=bins, density=True)
off_aborts_hist, _ = np.histogram(df_led_off_aborts['timed_fix'] + df_led_off_aborts['LED_onset_time'] - df_led_off_aborts['intended_fix'], bins=bins, density=True)

frac_on_aborts = len(df_led_on_aborts) / len(df_led_on_valid_and_aborts)
frac_off_aborts = len(df_led_off_aborts) / len(df_led_off_valid_and_aborts)

ax.plot(bins[:-1], on_aborts_hist * frac_on_aborts, color='red', label='LED on')
ax.plot(bins[:-1], off_aborts_hist * frac_off_aborts, color='blue', label='LED off')
ax.set_xlabel('timed_fix + LED_onset_time - intended_fix')
ax.set_ylabel('Density')
ax.legend()
plt.suptitle('No filter < 300ms aborts (LED onset aligned, aggregate)')
plt.tight_layout()
plt.show()

# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

bins = np.arange(-2, 2, 0.05)
on_aborts_hist, _ = np.histogram(df_led_on_aborts_filter['timed_fix'] + df_led_on_aborts_filter['LED_onset_time'] - df_led_on_aborts_filter['intended_fix'], bins=bins, density=True)
off_aborts_hist, _ = np.histogram(df_led_off_aborts_filter['timed_fix'] + df_led_off_aborts_filter['LED_onset_time'] - df_led_off_aborts_filter['intended_fix'], bins=bins, density=True)

frac_on_aborts = len(df_led_on_aborts_filter) / len(df_led_on_valid_and_aborts_filter)
frac_off_aborts = len(df_led_off_aborts_filter) / len(df_led_off_valid_and_aborts_filter)

ax.plot(bins[:-1], on_aborts_hist * frac_on_aborts, color='red', label='LED on')
ax.plot(bins[:-1], off_aborts_hist * frac_off_aborts, color='blue', label='LED off')
ax.set_xlabel('timed_fix + LED_onset_time - intended_fix')
ax.set_ylabel('Density')
ax.legend()
plt.suptitle('Filter < 300ms aborts (LED onset aligned, aggregate)')
plt.tight_layout()
plt.show()


# %%
