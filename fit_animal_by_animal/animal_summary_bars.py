import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read experiment data
exp_df = pd.read_csv('../outExp.csv')

# Remove trials where RTwrtStim is nan and abort_event == 3
exp_df = exp_df[~((exp_df['RTwrtStim'].isna()) & (exp_df['abort_event'] == 3))].copy()

# Use only comparable batch
batch_name = 'Comparable'
exp_df_batch = exp_df[(exp_df['batch_name'] == batch_name) & (exp_df['LED_trial'].isin([np.nan, 0]))].copy()

# Compute choice and accuracy columns if needed
import random
exp_df_batch['choice'] = exp_df_batch['response_poke'].apply(lambda x: 1 if x == 3 else (-1 if x == 2 else random.choice([1, -1])))
exp_df_batch['accuracy'] = (exp_df_batch['ILD'] * exp_df_batch['choice']).apply(lambda x: 1 if x > 0 else 0)

# Valid and aborts
df_valid_and_aborts = exp_df_batch[(exp_df_batch['success'].isin([1,-1])) | (exp_df_batch['abort_event'] == 3)].copy()
df_aborts = df_valid_and_aborts[df_valid_and_aborts['abort_event'] == 3]

animal_ids = df_valid_and_aborts['animal'].unique()

# Prepare figure (3 rows, N columns)
n_animals = len(animal_ids)
fig, axes = plt.subplots(nrows=3, ncols=n_animals, figsize=(5 * n_animals, 12), sharey='row')
if n_animals == 1:
    axes = axes[:, np.newaxis]  # ensure axes is always 2D (3,1)

# Precompute all y data for global y-limits
aborts_valid_counts = []
ild_trial_counts = []
abl_trial_counts = []
ild_bin_set = set()
abl_bin_set = set()

# First pass to gather data for y-limits and consistent bins
for animal in animal_ids:
    df_all_trials_animal = df_valid_and_aborts[df_valid_and_aborts['animal'] == animal]
    df_aborts_animal = df_aborts[df_aborts['animal'] == animal]
    df_valid_animal = df_all_trials_animal[df_all_trials_animal['success'].isin([1,-1])]
    df_valid_animal_less_than_1 = df_valid_animal[df_valid_animal['RTwrtStim'] < 1]
    # 1. aborts/valid
    num_aborts = len(df_aborts_animal)
    num_valid = len(df_valid_animal_less_than_1)
    aborts_valid_counts.append([num_aborts, num_valid])
    # 2. |ILD|
    abs_ilds = np.abs(df_valid_animal_less_than_1['ILD'])
    ild_bins = np.sort(np.unique(abs_ilds))
    ild_bin_set.update(ild_bins)
    ild_trial_counts.append(abs_ilds)
    # 3. ABL
    abls = df_valid_animal_less_than_1['ABL']
    abl_bins = np.sort(np.unique(abls))
    abl_bin_set.update(abl_bins)
    abl_trial_counts.append(abls)

# Global y-limits for each row
aborts_valid_max = np.max(aborts_valid_counts) if len(aborts_valid_counts) else 1
ild_global_bins = np.array(sorted(ild_bin_set))
ild_max = 1
if len(ild_trial_counts):
    ild_max = max([np.sum(x == b) for x in ild_trial_counts for b in ild_global_bins])
abl_global_bins = np.array(sorted(abl_bin_set))
abl_max = 1
if len(abl_trial_counts):
    abl_max = max([np.sum(x == b) for x in abl_trial_counts for b in abl_global_bins])

# Second pass: plotting
for col, animal in enumerate(animal_ids):
    df_all_trials_animal = df_valid_and_aborts[df_valid_and_aborts['animal'] == animal]
    df_aborts_animal = df_aborts[df_aborts['animal'] == animal]
    df_valid_animal = df_all_trials_animal[df_all_trials_animal['success'].isin([1,-1])]
    df_valid_animal_less_than_1 = df_valid_animal[df_valid_animal['RTwrtStim'] < 1]

    # 1. Bar: num aborts, num valid (<1s)
    num_aborts = len(df_aborts_animal)
    num_valid = len(df_valid_animal_less_than_1)
    axes[0, col].bar(['aborts', 'valid'], [num_aborts, num_valid], color=['#DD8452', '#4C72B0'])
    axes[0, col].set_ylabel('Count')
    axes[0, col].set_title(f'Animal {animal}: Aborts/Valid (<1s)')
    axes[0, col].set_ylim(0, aborts_valid_max * 1.1)

    # 2. Bar: num trials vs |ILD|
    abs_ilds = np.abs(df_valid_animal_less_than_1['ILD'])
    ild_counts = [np.sum(abs_ilds == b) for b in ild_global_bins]
    axes[1, col].bar(ild_global_bins, ild_counts, color='#4C72B0', width=0.15)
    axes[1, col].set_xlabel('|ILD|')
    axes[1, col].set_ylabel('Num Trials')
    axes[1, col].set_title(f'Animal {animal}: Trials vs |ILD|')
    axes[1, col].set_ylim(0, ild_max * 1.1)

    # 3. Bar: num trials vs ABL
    abls = df_valid_animal_less_than_1['ABL']
    abl_counts = [np.sum(abls == b) for b in abl_global_bins]
    axes[2, col].bar(abl_global_bins, abl_counts, color='#4C72B0', width=0.15)
    axes[2, col].set_xlabel('ABL')
    axes[2, col].set_ylabel('Num Trials')
    axes[2, col].set_title(f'Animal {animal}: Trials vs ABL')
    axes[2, col].set_ylim(0, abl_max * 1.1)

plt.tight_layout()
plt.savefig('animal_summary_bars.pdf')
print('Saved: animal_summary_bars.pdf')
