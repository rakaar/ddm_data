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
merged_valid['batch_name'] = merged_valid['batch_name'].fillna('LED7')

ABLS = [20, 40, 60]

# --- 1. Fit sigmoid for each rat and ABL, store slope (k) ---
slopes = {abl: {} for abl in ABLS}  # slopes[abl][rat] = k
mean_psycho_slope = {}  # mean_psycho_slope[rat] = k0
all_animals = merged_valid['animal'].unique()
allowed_ilds = np.sort(np.array([1., 2., 4., 8., 16., -1., -2., -4., -8., -16.]))

for animal in all_animals:
    # 1. Fit sigmoid for each ABL
    for abl in ABLS:
        animal_df = merged_valid[(merged_valid['animal'] == animal) & (merged_valid['ABL'] == abl)]
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
                slopes[abl][animal] = k
            except Exception as e:
                continue
    # 2. Fit sigmoid to mean psychometric (across all ABLs)
    animal_df_all_abl = merged_valid[(merged_valid['animal'] == animal) & (merged_valid['ABL'].isin(ABLS))]
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
            mean_psycho_slope[animal] = k0
        except Exception as e:
            continue

# --- 2. log-ratio of slope at each ABL to mean psychometric slope for that rat ---
log_ratios_within = []
for animal in all_animals:
    if animal not in mean_psycho_slope:
        continue
    k0 = mean_psycho_slope[animal]
    for abl in ABLS:
        if animal in slopes[abl]:
            ratio = slopes[abl][animal] / k0
            log_ratios_within.append(np.log(ratio))

# --- 4. log-ratio of mean slope for each rat to grand mean ---
# Only use animals present in mean_psycho_slope
animals_with_mean = list(mean_psycho_slope.keys())
grand_mean_k = np.mean([mean_psycho_slope[animal] for animal in animals_with_mean])
log_ratios_across = [np.log(mean_psycho_slope[animal] / grand_mean_k) for animal in animals_with_mean]

# --- 5. Plot histograms ---

ratios_within = np.exp(log_ratios_within)
ratios_across = np.exp(log_ratios_across)

diff_within = []  # slope_ABL - mean_slope_rat
for animal in all_animals:
    if animal not in mean_psycho_slope:
        continue
    k0 = mean_psycho_slope[animal]
    for abl in ABLS:
        if animal in slopes[abl]:
            diff_within.append(slopes[abl][animal] - k0)

mean_slopes = np.array([mean_psycho_slope[animal] for animal in animals_with_mean])
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
    animals = sorted([animal for animal in slopes[abl].keys()])
    slope_vals = [slopes[abl][animal] for animal in animals]
    ax.scatter(range(len(animals)), slope_vals, color=color, s=40)
    ax.set_title(f'ABL = {abl}')
    ax.set_xlabel('Animal')
    if idx == 0:
        ax.set_ylabel('Slope (k)')
    ax.set_xticks(range(len(animals)))
    ax.set_xticklabels(animals, rotation=90, fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()


# %%
# --- 7. Overlayed animal slopes for all ABLs ---
COLORS = ['tab:blue', 'tab:orange', 'tab:green']
animals = sorted(set().union(*[slopes[abl].keys() for abl in ABLS]))
fig, ax = plt.subplots(figsize=(13, 4))
for idx, abl in enumerate(ABLS):
    color = COLORS[idx]
    y = [slopes[abl].get(animal, np.nan) for animal in animals]
    ax.scatter(range(len(animals)), y, color=color, s=40, label=f'ABL={abl}')
ax.set_xlabel('Animal')
ax.set_ylabel('Slope (k)')
ax.set_xticks(range(len(animals)))
ax.set_xticklabels(animals, rotation=90, fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend()
plt.tight_layout()
plt.show()# %%
# %%
plt.plot(ratios_within)