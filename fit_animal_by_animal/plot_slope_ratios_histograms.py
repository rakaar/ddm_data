# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

# --- Sigmoid function ---
def sigmoid(x, L, x0, k, b):
    return L / (1 + np.exp(-k*(x-x0))) + b

# --- Load merged_valid as in make_all_animals_psycho_single_figure.py ---
batch_dir = os.path.join(os.path.dirname(__file__), 'batch_csvs')
batch_files = [f for f in os.listdir(batch_dir) if f.endswith('_valid_and_aborts.csv')]
merged_data = pd.concat([
    pd.read_csv(os.path.join(batch_dir, fname)) for fname in batch_files
], ignore_index=True)
merged_valid = merged_data[merged_data['success'].isin([1, -1])].copy()

#### To filter out specific batches, uncomment the following lines ######
# DESIRED_BATCHES = ['Comparable', 'SD', 'LED2', 'LED1', 'LED34', 'LED6']
# merged_valid = merged_valid[merged_valid['batch_name'].isin(DESIRED_BATCHES)].copy()

ABLS = [20, 40, 60]

# --- 1. Fit sigmoid for each rat and ABL, store slope (k) ---
slopes = {abl: {} for abl in ABLS}  # slopes[abl][(batch_name, animal)] = k
mean_psycho_slope = {}  # mean_psycho_slope[(batch_name, animal)] = k0
# Create unique identifiers by combining batch_name and animal
merged_valid['animal_id'] = list(zip(merged_valid['batch_name'], merged_valid['animal']))
all_animals = merged_valid['animal_id'].unique()
allowed_ilds = np.sort(np.array([1., 2., 4., 8., 16., -1., -2., -4., -8., -16.]))

for animal_id in all_animals:
    batch_name, animal = animal_id  # Unpack the tuple
    # 1. Fit sigmoid for each ABL
    for abl in ABLS:
        animal_df = merged_valid[(merged_valid['batch_name'] == batch_name) & 
                                  (merged_valid['animal'] == animal) & 
                                  (merged_valid['ABL'] == abl)]
        if animal_df.empty:
            continue
        animal_ilds = np.sort(animal_df['ILD'].unique())
        psycho = []
        for ild in animal_ilds:
            sub = animal_df[animal_df['ILD'] == ild]
            if len(sub) > 0:
                psycho.append(np.mean(sub['choice'] == 1))
            else:
                psycho.append(np.nan)
        psycho = np.array(psycho)
        mask = ~np.isnan(psycho)
        if np.sum(mask) > 3:
            try:
                popt, _ = curve_fit(sigmoid, animal_ilds[mask], psycho[mask], p0=[1, 0, 1, 0], maxfev=5000)
                k = popt[2]
                slopes[abl][animal_id] = k
            except Exception as e:
                continue
    # 2. Fit sigmoid to mean psychometric (across all ABLs)
    animal_df_all_abl = merged_valid[(merged_valid['batch_name'] == batch_name) & 
                                   (merged_valid['animal'] == animal) & 
                                   (merged_valid['ABL'].isin(ABLS))]
    if animal_df_all_abl.empty:
        continue
    # Compute mean P(Right) at each ILD (across ABLs)
    ilds = np.sort(animal_df_all_abl['ILD'].unique())
    psycho = []
    for ild in ilds:
        sub = animal_df_all_abl[animal_df_all_abl['ILD'] == ild]
        if len(sub) > 0:
            psycho.append(np.mean(sub['choice'] == 1))
        else:
            psycho.append(np.nan)
    psycho = np.array(psycho)
    mask = ~np.isnan(psycho)
    if np.sum(mask) > 3:
        try:
            popt, _ = curve_fit(sigmoid, ilds[mask], psycho[mask], p0=[1, 0, 1, 0], maxfev=5000)
            k0 = popt[2]
            mean_psycho_slope[animal_id] = k0
        except Exception as e:
            continue

# --- 2. log-ratio of slope at each ABL to mean psychometric slope for that rat ---
log_ratios_within = []
for animal_id in all_animals:
    if animal_id not in mean_psycho_slope:
        continue
    k0 = mean_psycho_slope[animal_id]
    for abl in ABLS:
        if animal_id in slopes[abl]:
            ratio = slopes[abl][animal_id] / k0
            log_ratios_within.append(np.log(ratio))

# --- 4. log-ratio of mean slope for each rat to grand mean ---
# Only use animals present in mean_psycho_slope
animals_with_mean = list(mean_psycho_slope.keys())
grand_mean_k = np.mean([mean_psycho_slope[animal_id] for animal_id in animals_with_mean])
log_ratios_across = [np.log(mean_psycho_slope[animal_id] / grand_mean_k) for animal_id in animals_with_mean]

# --- 5. Plot histograms ---

ratios_within = np.exp(log_ratios_within)
ratios_across = np.exp(log_ratios_across)

diff_within = []  # slope_ABL - mean_slope_rat
for animal_id in all_animals:
    if animal_id not in mean_psycho_slope:
        continue
    k0 = mean_psycho_slope[animal_id]
    for abl in ABLS:
        if animal_id in slopes[abl]:
            diff_within.append(slopes[abl][animal_id] - k0)

mean_slopes = np.array([mean_psycho_slope[animal_id] for animal_id in animals_with_mean])
diff_across = mean_slopes - grand_mean_k  # mean_slope_rat - grand_mean

bins_within = np.arange(0, 2, 0.05)
bins_across = np.arange(0, 2, 0.1)
bins_diff = np.arange(-1, 1, 0.05)
bins_absdiff = np.arange(-1, 1, 0.05)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.hist(ratios_within, bins=bins_within, color='tab:blue', alpha=0.7, density=True)
plt.axvline(1, color='k', linestyle='--')
plt.title('Within-rat (slope_ABL / mean_slope_rat)')
plt.xlabel('ratio')
plt.ylabel('Density')
plt.ylim(0, 4)
ax1 = plt.gca()
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

plt.subplot(1,2,2)
plt.hist(ratios_across, bins=bins_across, color='tab:orange', alpha=0.7, density=True)
plt.axvline(1, color='k', linestyle='--')
plt.title('Across-rat (mean_slope_rat / grand_mean)')
plt.xlabel('ratio')
ax2 = plt.gca()
ax2.set_ylabel("")
ax2.set_yticklabels([])
ax2.set_yticks([])
plt.ylim(0, 4)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

# --- NEW: Plot histogram of absolute differences ---
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.hist(diff_within, bins=bins_absdiff, color='tab:blue', alpha=0.7, density=True)
plt.axvline(0, color='k', linestyle='--')
plt.title('Within-rat (slope_ABL - mean_slope_rat)')
plt.xlabel('difference')
plt.ylabel('Density')
plt.ylim(0, 4)
ax1 = plt.gca()
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

plt.subplot(1,2,2)
plt.hist(diff_across, bins=bins_absdiff, color='tab:orange', alpha=0.7, density=True)
plt.axvline(0, color='k', linestyle='--')
plt.title('Across-rat (mean_slope_rat - grand_mean)')
plt.xlabel('difference')
ax2 = plt.gca()
ax2.set_ylabel("")
ax2.set_yticklabels([])
ax2.set_yticks([])
plt.ylim(0, 4)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

# %%
# std of diff_with and diff_across
print(f'std(diff_within animal): {np.std(diff_within):.3f}')
print(f'std(diff_across animals): {np.std(diff_across):.3f}')

print(f'std(ratio_within animal) {np.std(ratios_within):.3f}')
print(f'std(ratio_across animal) {np.std(ratios_across):.3f}')
# %%

# %%
# --- 6. Per-ABL slope plot for each animal ---
COLORS = ['tab:blue', 'tab:orange', 'tab:green']
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for idx, abl in enumerate(ABLS):
    ax = axes[idx]
    color = COLORS[idx]
    # Get animals with slope for this ABL
    animals_ids = sorted([animal_id for animal_id in slopes[abl].keys()])
    slope_vals = [slopes[abl][animal_id] for animal_id in animals_ids]
    ax.scatter(range(len(animals_ids)), slope_vals, color=color, s=40)
    ax.set_title(f'ABL = {abl}')
    ax.set_xlabel('Animal')
    if idx == 0:
        ax.set_ylabel('Slope (k)')
    ax.set_xticks(range(len(animals_ids)))
    # Create readable labels by combining batch_name and animal
    animal_labels = [f"{batch}-{animal}" for batch, animal in animals_ids]
    ax.set_xticklabels(animal_labels, rotation=90, fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()


# %%
# --- 7. Overlayed animal slopes for all ABLs ---
COLORS = ['tab:blue', 'tab:orange', 'tab:green']
animals = sorted(set().union(*[slopes[abl].keys() for abl in ABLS]))
fig, ax = plt.subplots(figsize=(6, 3))  # Compressed x-axis
for idx, abl in enumerate(ABLS):
    color = COLORS[idx]
    y = [slopes[abl].get(animal, np.nan) for animal in animals]
    ax.scatter(range(len(animals)), y, color=color, s=40)
# Remove all x-ticks and labels
ax.set_xticks([])
ax.set_xticklabels([])
# Draw a horizontal line at y=0 (or at the bottom of the plot)
ax.axhline(0, color='k', linewidth=1)
ax.set_xlabel('')
ax.set_ylabel('Slope (k)', fontsize=13)
# Set y-ticks to 0.5 and 1
ax.set_yticks([0.5, 1])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()# %%

# %%
# --- 7b. Overlayed animal slopes for all ABLs, excluding LED2, 41 ---
COLORS = ['tab:blue', 'tab:orange', 'tab:green']
# animals should be a list of (batch_name, animal) tuples
print((animals))
# Remove only ('LED2', 41) from the animals list
animals_filtered = [a for a in animals if a != ('LED2', 41)]
print(len(animals_filtered))
fig, ax = plt.subplots(figsize=(5, 3))  # Compressed x-axis
for idx, abl in enumerate(ABLS):
    color = COLORS[idx]
    y = [slopes[abl].get(animal, np.nan) for animal in animals_filtered]
    ax.scatter(range(len(animals_filtered)), y, color=color, s=40)
# Set x-ticks and labels to batch-animal
ax.set_xticks(range(len(animals_filtered)))
animal_labels = ["" for batch, animal in animals_filtered]

ax.set_xticklabels(animal_labels, rotation=90, fontsize=8)
ax.set_xlabel('Rat #', fontsize=13)
ax.set_ylabel('Slope (k)', fontsize=13)
# Set y-ticks to 0.5 and 1
ax.set_yticks([0, 0.5, 1])
ax.tick_params(axis='y', labelsize=13)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
# save as pdf
plt.savefig('slope_ratios_across_rats.pdf', dpi=300, bbox_inches='tight')
plt.show()# %%
# %%

# %%
