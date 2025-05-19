# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Directory containing batch CSVs
batch_dir = os.path.join(os.path.dirname(__file__), 'batch_csvs')
batch_files = [f for f in os.listdir(batch_dir) if f.endswith('_valid_and_aborts.csv')]

# Load and concatenate all batch data
all_data = pd.concat([
    pd.read_csv(os.path.join(batch_dir, fname)) for fname in batch_files
], ignore_index=True)

# Only valid trials (success in {1, -1})
valid = all_data[all_data['success'].isin([1, -1])].copy()
valid = valid[valid['RTwrtStim'] <= 1]

# Filter for ABLs 20, 40, 60
valid = valid[valid['ABL'].isin([20, 40, 60])]

# Add abs_ILD if missing
if 'abs_ILD' not in valid.columns:
    valid['abs_ILD'] = valid['ILD'].abs()

# Only keep abs_ILD in [1,2,4,8,16]
# convert  abs_ILD into float in all columns
valid['abs_ILD'] = valid['abs_ILD'].astype(float)
valid = valid[valid['abs_ILD'].isin([1.0,2.0,4.0,8.0,16.0])]

# batch name is nan fill LED7
valid['batch_name'] = valid['batch_name'].fillna('LED7')
#### TEMP: REMOVE LED7 FROM VALID############
# print(valid['batch_name'].unique())
# valid = valid[valid['batch_name'] != 'LED7']
# valid = valid[valid['batch_name'] != 'LED2']
print(valid['batch_name'].unique())
##########################################
# Sort abs_ILD for plotting
abs_ilds = [1.0,2.0,4.0,8.0,16.0]

# Plot setup
fig, ax = plt.subplots(figsize=(8,6))
abl_colors = {20:'#1f77b4', 40:'#ff7f0e', 60:'#2ca02c'}
PLOT_MODE = 'mean+std'
# Get all unique (batch_name, animal) pairs
animal_keys = valid[['batch_name', 'animal']].drop_duplicates()
# For each ABL, collect per-animal mean RTwrtStim for each abs_ILD
abs_ilds = [1.0,2.0,4.0,8.0,16.0]

for abl in [20, 40, 60]:
    # Store, for each abs_ILD, a list of per-animal means
    per_animal_means = {abs_ild: [] for abs_ild in abs_ilds}
    for _, row in animal_keys.iterrows():
        batch = row['batch_name']
        animal = row['animal']
        animal_df = valid[(valid['batch_name'] == batch) & (valid['animal'] == animal) & (valid['ABL'] == abl)]
        for abs_ild in abs_ilds:
            subset = animal_df[animal_df['abs_ILD'] == abs_ild]
            if len(subset) > 0:
                mean_rt = subset['RTwrtStim'].mean()
                per_animal_means[abs_ild].append(mean_rt)
            # If no trials for this abs_ILD, skip (do not append NaN)
    # For each abs_ILD, compute mean and std across animals
    if PLOT_MODE == 'median+empirical_CI':
        yvals = []
        lower_errs = []
        upper_errs = []
        for abs_ild in abs_ilds:
            vals = per_animal_means[abs_ild]
            if len(vals) > 0:
                median = np.median(vals)
                ci_low = np.percentile(vals, 2.5)
                ci_high = np.percentile(vals, 97.5)
                yvals.append(median)
                lower_errs.append(median - ci_low)
                upper_errs.append(ci_high - median)
            else:
                yvals.append(np.nan)
                lower_errs.append(np.nan)
                upper_errs.append(np.nan)
        yerr = np.array([lower_errs, upper_errs])
        ax.errorbar(abs_ilds, yvals, yerr=yerr, marker='o', label=f'ABL {abl}', color=abl_colors[abl], capsize=0, linestyle='-', linewidth=2)
    else:
        yvals = []
        errvals = []
        for abs_ild in abs_ilds:
            vals = per_animal_means[abs_ild]
            if len(vals) > 0:
                if PLOT_MODE == 'mean+std':
                    yvals.append(np.mean(vals))
                    errvals.append(np.std(vals))
                elif PLOT_MODE == 'median':
                    yvals.append(np.median(vals))
                    errvals.append(None)
                else:
                    raise ValueError('PLOT_MODE must be mean+std, median, or median+empirical_CI')
            else:
                yvals.append(np.nan)
                errvals.append(np.nan)
        ax.errorbar(abs_ilds, yvals, yerr=errvals, marker='o', label=f'ABL {abl}', color=abl_colors[abl], capsize=0, linestyle='-', linewidth=2)


ax.set_xlabel('|ILD| (dB)', fontsize=16)
ax.set_ylabel(f'{PLOT_MODE} RTwrtStim (s)', fontsize=16)
ax.set_title('Chronometric (Averaged Across Animals)', fontsize=18)
ax.set_xticks(abs_ilds)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
# ax.axhline(0.29)
ax.legend(title='ABL', fontsize=12)
ax.set_ylim(0, 0.4)
plt.tight_layout()


plt.show()

# %%
for abl in [20, 40, 60]:
    print(f"ABL {abl}: ")
    # Store, for each abs_ILD, a list of per-animal means
    per_animal_means = {abs_ild: [] for abs_ild in abs_ilds}
    for _, row in animal_keys.iterrows():
        batch = row['batch_name']
        animal = row['animal']
        animal_df = valid[(valid['batch_name'] == batch) & (valid['animal'] == animal) & (valid['ABL'] == abl)]
        for abs_ild in abs_ilds:
            subset = animal_df[animal_df['abs_ILD'] == abs_ild]
            if len(subset) > 0:
                mean_rt = subset['RTwrtStim'].mean()
                per_animal_means[abs_ild].append(mean_rt)
            # If no trials for this abs_ILD, skip (do not append NaN)
    # For each abs_ILD, compute mean and std across animals
    yvals = []
    errvals = []
    for abs_ild in abs_ilds:
        vals = per_animal_means[abs_ild]
        print(f'ILDs: {abs_ild}')
        print(f'std: {np.round(np.std(vals), 2)}, mean = {np.round(np.mean(vals), 2)}, mean+std={np.round(np.mean(vals)+np.std(vals), 2)}, mean-std={np.round(np.mean(vals)-np.std(vals), 2)}')


# %%
print(len(animal_keys))
