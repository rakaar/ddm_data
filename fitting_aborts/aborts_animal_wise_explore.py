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
# we look at only valid and aborts
df = df[(df['abort_event'] == 3) | (df['success'].isin([1,-1]))]

# %%
FILTER_300 = False
if FILTER_300:
    df = df[~( (df['abort_event'] == 3) & (df['timed_fix'] < 0.3) )]


# %%
# Compare LED ON abort fractions: bilateral-only vs all LED ON
led_on_abort_fraction_rows = []
animals_with_all = ['ALL'] + list(df['animal'].unique())

for animal in animals_with_all:
    if animal == 'ALL':
        df_scope = df
    else:
        df_scope = df[df['animal'] == animal]

    df_led_on_all = df_scope[df_scope['LED_trial'] == 1]
    df_led_on_bilateral = df_scope[
        (df_scope['LED_trial'] == 1)
        & (df_scope['LED_powerR'] != 0)
        & (df_scope['LED_powerL'] != 0)
    ]
    df_led_on_unilateral_left = df_scope[
        (df_scope['LED_trial'] == 1)
        & (df_scope['LED_powerR'] == 0)
        & (df_scope['LED_powerL'] != 0)
    ]
    df_led_on_unilateral_right = df_scope[
        (df_scope['LED_trial'] == 1)
        & (df_scope['LED_powerR'] != 0)
        & (df_scope['LED_powerL'] == 0)
    ]

    n_all_led_on = len(df_led_on_all)
    n_all_led_on_aborts = int((df_led_on_all['abort_event'] == 3).sum())
    frac_all_led_on_aborts = (
        n_all_led_on_aborts / n_all_led_on if n_all_led_on > 0 else np.nan
    )

    n_bilateral_led_on = len(df_led_on_bilateral)
    n_bilateral_led_on_aborts = int((df_led_on_bilateral['abort_event'] == 3).sum())
    frac_bilateral_led_on_aborts = (
        n_bilateral_led_on_aborts / n_bilateral_led_on if n_bilateral_led_on > 0 else np.nan
    )

    n_unilateral_left_led_on = len(df_led_on_unilateral_left)
    n_unilateral_left_led_on_aborts = int((df_led_on_unilateral_left['abort_event'] == 3).sum())
    frac_unilateral_left_led_on_aborts = (
        n_unilateral_left_led_on_aborts / n_unilateral_left_led_on
        if n_unilateral_left_led_on > 0 else np.nan
    )

    n_unilateral_right_led_on = len(df_led_on_unilateral_right)
    n_unilateral_right_led_on_aborts = int((df_led_on_unilateral_right['abort_event'] == 3).sum())
    frac_unilateral_right_led_on_aborts = (
        n_unilateral_right_led_on_aborts / n_unilateral_right_led_on
        if n_unilateral_right_led_on > 0 else np.nan
    )

    first_three_fracs = np.array([
        frac_bilateral_led_on_aborts,
        frac_unilateral_left_led_on_aborts,
        frac_unilateral_right_led_on_aborts,
    ], dtype=float)
    if np.all(np.isnan(first_three_fracs)) or np.isnan(frac_all_led_on_aborts):
        abs_diff_avg_first3_vs_all = np.nan
    else:
        avg_first_three = np.nanmean(first_three_fracs)
        abs_diff_avg_first3_vs_all = np.abs(avg_first_three - frac_all_led_on_aborts)

    led_on_abort_fraction_rows.append({
        'animal': animal,
        'frac_led_on_bilateral_aborts': frac_bilateral_led_on_aborts,
        'frac_led_on_unilateral_left_aborts': frac_unilateral_left_led_on_aborts,
        'frac_led_on_unilateral_right_aborts': frac_unilateral_right_led_on_aborts,
        'frac_led_on_all_aborts': frac_all_led_on_aborts,
        'abs_diff_avg_first3_vs_all': abs_diff_avg_first3_vs_all,
    })

df_led_on_abort_fraction_compare = pd.DataFrame(led_on_abort_fraction_rows).set_index('animal')
print('\nLED ON abort-fraction table (bilateral, unilateral, and all LED ON):')
table_str = (
    df_led_on_abort_fraction_compare
    .reset_index()
    .to_string(index=False, float_format=lambda x: f'{x:.4f}')
)
table_lines = table_str.splitlines()
print(table_lines[0])
print('-' * len(table_lines[0]))
for row_line in table_lines[1:]:
    print(row_line)
    print('-' * len(table_lines[0]))

# %%
# Permutation test for abort-fraction differences across LED ON conditions
# Shuffle labels (left/right/bilateral) within each animal while preserving condition counts.
# LED OFF trials are excluded.

N_SHUFFLES = int(1e5)
PERM_SEED = 1234

if FILTER_300:
    raise ValueError('Set FILTER_300 = False for this permutation test (raw data requested).')

cond_names = ['Left inhibition', 'Right inhibition', 'Bilateral inhibition']

unique_animals_perm = df['animal'].unique()
fig, axes = plt.subplots(
    len(unique_animals_perm), len(cond_names),
    figsize=(5.2 * len(cond_names), 3.4 * len(unique_animals_perm)),
    sharex=False, sharey=False
)

if len(unique_animals_perm) == 1:
    axes = np.expand_dims(axes, axis=0)

perm_summary_rows = []

for i, animal in enumerate(unique_animals_perm):
    print(f'processng animal = {animal}')
    df_animal = df[df['animal'] == animal]
    df_led_on = df_animal[df_animal['LED_trial'] == 1].copy()

    cond_code = np.full(len(df_led_on), -1, dtype=int)
    cond_code[(df_led_on['LED_powerR'] == 0) & (df_led_on['LED_powerL'] != 0)] = 0  # left inhibition
    cond_code[(df_led_on['LED_powerR'] != 0) & (df_led_on['LED_powerL'] == 0)] = 1  # right inhibition
    cond_code[(df_led_on['LED_powerR'] != 0) & (df_led_on['LED_powerL'] != 0)] = 2  # bilateral inhibition

    valid_mask = cond_code >= 0
    df_led_on = df_led_on.loc[valid_mask]
    cond_code = cond_code[valid_mask]

    abort_mask = (df_led_on['abort_event'] == 3).to_numpy(dtype=float)
    counts = np.bincount(cond_code, minlength=3)
    observed_abort_counts = np.bincount(cond_code, weights=abort_mask, minlength=3)

    observed_fracs = np.full(3, np.nan, dtype=float)
    valid_condn = counts > 0
    observed_fracs[valid_condn] = observed_abort_counts[valid_condn] / counts[valid_condn]

    rng = np.random.default_rng(PERM_SEED + int(animal))
    perm_fracs = np.full((N_SHUFFLES, 3), np.nan, dtype=float)
    for k in range(N_SHUFFLES):
        shuffled_cond = rng.permutation(cond_code)
        shuffled_abort_counts = np.bincount(shuffled_cond, weights=abort_mask, minlength=3)
        perm_fracs[k, valid_condn] = shuffled_abort_counts[valid_condn] / counts[valid_condn]

    for j, cond_name in enumerate(cond_names):
        ax = axes[i, j]

        if counts[j] <= 0:
            ax.text(0.5, 0.5, 'No trials', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Animal {animal} | {cond_name}\nNo data')
            if i == len(unique_animals_perm) - 1:
                ax.set_xlabel('Shuffled abort fraction')
            if j == 0:
                ax.set_ylabel('Count')
            continue

        perm_vals = perm_fracs[:, j]
        obs_val = observed_fracs[j]
        q2p5, q50, q97p5 = np.percentile(perm_vals, [2.5, 50, 97.5])

        p_ge = (np.sum(perm_vals >= obs_val) + 1) / (N_SHUFFLES + 1)
        p_le = (np.sum(perm_vals <= obs_val) + 1) / (N_SHUFFLES + 1)
        p_two = min(1.0, 2 * min(p_ge, p_le))

        # ax.hist(perm_vals, bins=np.arange(0.125, 0.250, 0.001), color='lightgray', edgecolor='black')
        ax.hist(perm_vals, bins=40, color='lightgray', edgecolor='black')

        ax.axvline(q2p5, color='blue', ls=':', linewidth=1.5)
        ax.axvline(q50, color='blue', ls=':', linewidth=1.5)
        ax.axvline(q97p5, color='blue', ls=':', linewidth=1.5)
        ax.axvline(obs_val, color='red', linewidth=2)
        ax.set_title(
            f'Animal {animal} | {cond_name}\n'
            f'obs={obs_val:.3f}, p>= {p_ge:.3f}, p<= {p_le:.3f}, p2={p_two:.3f}'
        )

        if i == len(unique_animals_perm) - 1:
            ax.set_xlabel('Shuffled abort fraction')
        if j == 0:
            ax.set_ylabel('Count')

        perm_summary_rows.append({
            'animal': animal,
            'condition': cond_name,
            'n_trials': int(counts[j]),
            'obs_frac': obs_val,
            'p_ge_obs': p_ge,
            'p_le_obs': p_le,
            'p_two_sided': p_two
        })

plt.suptitle(
    f'Permutation test (N={N_SHUFFLES}) of abort fractions across LED ON conditions',
    y=1.02
)
plt.tight_layout()
plt.show()

perm_summary_df = pd.DataFrame(perm_summary_rows)
print('\nPermutation test summary:')
print(perm_summary_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
# %%
# 6 x 3 plot (N animals x 3 conditions)
# Conditions: left inhibition, right inhibition, bilateral inhibition
# Plot RT wrt LED for LED ON (condition-specific) and LED OFF (same in all 3 condition panels).
# Histogram area is abort fraction.

def scaled_hist(values, n_total, bins):
    if n_total <= 0 or len(values) == 0:
        return np.zeros(len(bins) - 1)
    frac = len(values) / n_total
    hist, _ = np.histogram(values, bins=bins, density=True)
    return hist * frac


def rt_wrt_led(df_subset):
    return (df_subset['timed_fix'] - (df_subset['intended_fix'] - df_subset['LED_onset_time'])).values


bins_condn = np.arange(-2, 2, 0.005)
bin_width = bins_condn[1] - bins_condn[0]
bin_centers = bins_condn[:-1]

condition_defs = [
    ('Left inhibition', lambda d: d[(d['LED_trial'] == 1) & (d['LED_powerR'] == 0) & (d['LED_powerL'] != 0)]),
    ('Right inhibition', lambda d: d[(d['LED_trial'] == 1) & (d['LED_powerR'] != 0) & (d['LED_powerL'] == 0)]),
    ('Bilateral inhibition', lambda d: d[(d['LED_trial'] == 1) & (d['LED_powerR'] != 0) & (d['LED_powerL'] != 0)]),
]

unique_animals = df['animal'].unique()
fig, axes = plt.subplots(
    len(unique_animals), len(condition_defs),
    figsize=(5.0 * len(condition_defs), 3.4 * len(unique_animals)),
    sharex=True, sharey=True
)

if len(unique_animals) == 1:
    axes = np.expand_dims(axes, axis=0)

for i, animal in enumerate(unique_animals):
    df_animal = df[df['animal'] == animal]

    # Same LED OFF curve for all 3 condition panels of this animal.
    df_off = df_animal[df_animal['LED_trial'].isin([0, np.nan])]
    df_off_aborts = df_off[df_off['abort_event'] == 3]
    off_rate = scaled_hist(rt_wrt_led(df_off_aborts), len(df_off), bins_condn)
    off_area = np.sum(off_rate) * bin_width

    for j, (cond_name, cond_filter_fn) in enumerate(condition_defs):
        df_on_cond = cond_filter_fn(df_animal)
        df_on_cond_aborts = df_on_cond[df_on_cond['abort_event'] == 3]
        on_rate = scaled_hist(rt_wrt_led(df_on_cond_aborts), len(df_on_cond), bins_condn)
        on_area = np.sum(on_rate) * bin_width

        ax = axes[i, j]
        ax.plot(bin_centers, on_rate, color='red', label='LED ON')
        ax.plot(bin_centers, off_rate, color='blue', label='LED OFF')
        ax.axvline(x=0, color='black', ls='--', alpha=0.3)
        ax.set_title(f'Animal {animal} | {cond_name}\nAon={on_area:.3f}, Aoff={off_area:.3f}')
        ax.set_xlim(-0.05,0.15)
        if i == len(unique_animals) - 1:
            ax.set_xlabel('Abort Time wrt LED onset')
        if j == 0:
            ax.set_ylabel('Abort rate (area = frac aborts)')

        if i == 0 and j == 0:
            ax.legend(loc='upper right')

# plt.suptitle('RT wrt LED by condition (area-scaled abort fraction)')
plt.tight_layout()
plt.show()

# %%
# Aggregate permutation test (all animals pooled): abort fractions across 3 LED-ON conditions
N_SHUFFLES = int(1e4)

df_led_on_all_animals = df[df['LED_trial'] == 1].copy()
cond_code_all = np.full(len(df_led_on_all_animals), -1, dtype=int)
cond_code_all[(df_led_on_all_animals['LED_powerR'] == 0) & (df_led_on_all_animals['LED_powerL'] != 0)] = 0
cond_code_all[(df_led_on_all_animals['LED_powerR'] != 0) & (df_led_on_all_animals['LED_powerL'] == 0)] = 1
cond_code_all[(df_led_on_all_animals['LED_powerR'] != 0) & (df_led_on_all_animals['LED_powerL'] != 0)] = 2

valid_led_on_mask = cond_code_all >= 0
df_led_on_all_animals = df_led_on_all_animals.loc[valid_led_on_mask].reset_index(drop=True)
cond_code_all = cond_code_all[valid_led_on_mask]

abort_mask_all = (df_led_on_all_animals['abort_event'] == 3).to_numpy(dtype=float)
counts_all = np.bincount(cond_code_all, minlength=3)
obs_abort_counts_all = np.bincount(cond_code_all, weights=abort_mask_all, minlength=3)
obs_fracs_all = np.full(3, np.nan, dtype=float)
valid_counts_all = counts_all > 0
obs_fracs_all[valid_counts_all] = obs_abort_counts_all[valid_counts_all] / counts_all[valid_counts_all]

rng_agg = np.random.default_rng(PERM_SEED + 999)
perm_fracs_all = np.full((N_SHUFFLES, 3), np.nan, dtype=float)
for k in range(N_SHUFFLES):
    shuffled_cond_all = rng_agg.permutation(cond_code_all)
    shuffled_abort_counts_all = np.bincount(shuffled_cond_all, weights=abort_mask_all, minlength=3)
    perm_fracs_all[k, valid_counts_all] = shuffled_abort_counts_all[valid_counts_all] / counts_all[valid_counts_all]

cond_labels = ['Left inhibition', 'Right inhibition', 'Bilateral inhibition']
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=False, sharey=False)
agg_perm_summary_rows = []

for cond_idx, cond_name in enumerate(cond_labels):
    ax = axes[cond_idx]
    perm_vals = perm_fracs_all[:, cond_idx]
    obs_val = obs_fracs_all[cond_idx]
    q2p5, q50, q97p5 = np.percentile(perm_vals, [2.5, 50, 97.5])

    p_ge = (np.sum(perm_vals >= obs_val) + 1) / (N_SHUFFLES + 1)
    p_le = (np.sum(perm_vals <= obs_val) + 1) / (N_SHUFFLES + 1)
    p_two = min(1.0, 2 * min(p_ge, p_le))

    ax.hist(perm_vals, bins=40, color='lightgray', edgecolor='black')
    ax.axvline(q2p5, color='blue', ls=':', linewidth=1.5)
    ax.axvline(q50, color='blue', ls=':', linewidth=1.5)
    ax.axvline(q97p5, color='blue', ls=':', linewidth=1.5)
    ax.axvline(obs_val, color='red', linewidth=2)
    ax.set_title(
        f'Aggregate | {cond_name}\n'
        f'obs={obs_val:.3f}, p>= {p_ge:.3f}, p<= {p_le:.3f}, p2={p_two:.3f}'
    )
    ax.set_xlabel('Shuffled abort fraction')
    if cond_idx == 0:
        ax.set_ylabel('Count')

    agg_perm_summary_rows.append({
        'condition': cond_name,
        'n_trials': int(counts_all[cond_idx]),
        'obs_frac': obs_val,
        'p_ge_obs': p_ge,
        'p_le_obs': p_le,
        'p_two_sided': p_two
    })

plt.suptitle(f'Aggregate permutation test (N={N_SHUFFLES})')
plt.tight_layout()
plt.show()

agg_perm_summary_df = pd.DataFrame(agg_perm_summary_rows)
print('\nAggregate permutation summary:')
print(agg_perm_summary_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))


# %%
# Aggregate RTD wrt LED across 3 conditions (all animals pooled, no RTD shuffling)
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
df_off_all = df[df['LED_trial'].isin([0, np.nan])]
df_off_all_aborts = df_off_all[df_off_all['abort_event'] == 3]
off_rate_all = scaled_hist(rt_wrt_led(df_off_all_aborts), len(df_off_all), bins_condn)
off_area_all = np.sum(off_rate_all) * bin_width

for cond_idx, (cond_name, cond_filter_fn) in enumerate(condition_defs):
    df_on_cond_all = cond_filter_fn(df)
    df_on_cond_all_aborts = df_on_cond_all[df_on_cond_all['abort_event'] == 3]
    on_rate_all = scaled_hist(rt_wrt_led(df_on_cond_all_aborts), len(df_on_cond_all), bins_condn)
    on_area_all = np.sum(on_rate_all) * bin_width

    ax = axes[cond_idx]
    ax.plot(bin_centers, on_rate_all, color='red', label='LED ON')
    ax.plot(bin_centers, off_rate_all, color='blue', label='LED OFF')
    ax.axvline(x=0, color='black', ls='--', alpha=0.3)
    ax.set_xlim(-0.05, 0.15)
    ax.set_title(f'Aggregate | {cond_name}\nAon={on_area_all:.3f}, Aoff={off_area_all:.3f}')
    ax.set_xlabel('Abort Time wrt LED onset')
    if cond_idx == 0:
        ax.set_ylabel('Abort rate (area = frac aborts)')
        ax.legend(loc='upper right')

plt.suptitle('Aggregate RT wrt LED by condition (all animals pooled)')
plt.tight_layout()
plt.show()


# %%
# aggregate
df_on = df[(df['LED_trial'] == 1) & (df['LED_powerR'] != 0) & (df['LED_powerL'] != 0)]
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
    df_on = df_animal[
        (df_animal['LED_trial'] == 1)
        & (df_animal['LED_powerR'] != 0)
        & (df_animal['LED_powerL'] != 0)
    ]
    df_off = df_animal[df_animal['LED_trial'].isin([0, np.nan])]

    df_on_aborts = df_on[df_on['abort_event'] == 3]
    print(f'For animal {animal}, Num of LED ON Aborts = {len(df_on_aborts)}')
    df_off_aborts = df_off[df_off['abort_event'] == 3]
    print(f'For animal {animal}, Num of LED OFF Aborts = {len(df_off_aborts)}')
    print('-------------------------------------')
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
