# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

# --- Sigmoid function ---
def sigmoid(x, lambda_L, lambda_R, k, x0):
    return lambda_L + (1 - lambda_L - lambda_R) / (1 + np.exp(-k * (x - x0)))

# --- Data loading ---
# Define the batches you want to load
# DESIRED_BATCHES = ['SD', 'LED34', 'LED6', 'LED8', 'LED7', 'LED34_even',  'LED1', 'Comparable'] # Excluded LED1 as per original logic
DESIRED_BATCHES = ['SD', 'LED34', 'LED6', 'LED8', 'LED7', 'LED34_even'] # Excluded LED1 as per original logic

csv_dir = os.path.join(os.path.dirname(__file__), 'batch_csvs')

# Construct file paths and load data
batch_files = [f'batch_{batch_name}_valid_and_aborts.csv' for batch_name in DESIRED_BATCHES]
all_data_list = []
for fname in batch_files:
    fpath = os.path.join(csv_dir, fname)
    if os.path.exists(fpath):
        print(f"Loading {fpath}...")
        all_data_list.append(pd.read_csv(fpath))

if not all_data_list:
    raise FileNotFoundError(f"No batch CSV files found for {DESIRED_BATCHES} in '{csv_dir}'")

merged_data = pd.concat(all_data_list, ignore_index=True)

# --- Filter for valid trials ---
merged_valid = merged_data[merged_data['success'].isin([1, -1])].copy()

ABLS = [20, 40, 60]

# --- 1. Fit sigmoid for each rat and ABL, store slope (k) ---
slopes = {abl: {} for abl in ABLS}  # slopes[abl][(batch_name, animal)] = k
mean_psycho_slope = {}  # mean_psycho_slope[(batch_name, animal)] = k0
# Create unique identifiers by combining batch_name and animal
merged_valid['animal_id'] = list(zip(merged_valid['batch_name'], merged_valid['animal']))
all_animals = merged_valid['animal_id'].unique()
allowed_ilds = np.sort(np.array([1., 2., 4., 8., 16., -1., -2., -4., -8., -16.]))
print(f'Number of animals: {len(all_animals)}')
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
                p0 = [0.05, 0.05, 1, 0]
                bounds = ([0, 0, -np.inf, -np.inf], [1, 1, np.inf, np.inf])
                popt, _ = curve_fit(sigmoid, animal_ilds[mask], psycho[mask], p0=p0, bounds=bounds, maxfev=5000)
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
# %%
# --- NEW: Plot histogram of absolute differences ---
font = {'size': 18}
plt.rc('font', **font)
max_ylim = 8
plt.figure(figsize=(12, 6))
max_xlim = 0.4

# Subplot 1: Within-rat differences
ax1 = plt.subplot(1, 2, 1)
plt.hist(diff_within, bins=bins_absdiff, color='grey', alpha=0.7, density=True)
plt.axvline(0, color='k', linestyle=':')
# plt.title('Within-rat', fontsize=18)
plt.xlabel(r'$\mu_{ABL} - \mu_{rat}$', fontsize=18)
plt.ylabel('Density', fontsize=18)
plt.xlim(-max_xlim, max_xlim)
plt.ylim(0, max_ylim)
plt.xticks([-max_xlim, 0, max_xlim])
plt.yticks([0, max_ylim])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params(axis='both', which='major', labelsize=18)


# Subplot 2: Across-rat differences
ax2 = plt.subplot(1, 2, 2)
plt.hist(diff_across, bins=bins_absdiff, color='grey', alpha=0.7, density=True)
plt.axvline(0, color='k', linestyle=':')
# plt.title('Across-rat', fontsize=18)
plt.xlabel(r'$\mu_{rat} - \mu_{grand}$', fontsize=18)
plt.xlim(-max_xlim, max_xlim)
plt.ylim(0, max_ylim)
plt.xticks([-max_xlim, 0, max_xlim])
plt.yticks([0, max_ylim])
ax2.set_yticklabels([])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.tick_params(axis='both', which='major', labelsize=18)

plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to prevent title overlap
# plt.suptitle('Slope Differences', fontsize=18, y=1.02)
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
print(f'len of animals: {len(animals)}')
print(animals)
fig, ax = plt.subplots(figsize=(6, 3))  # Compressed x-axis
for idx, abl in enumerate(ABLS):
    color = COLORS[idx]
    y = [slopes[abl].get(animal, np.nan) for animal in animals]
    ax.scatter(range(len(animals)), y, color=color, s=40)
# Set x-ticks to represent each animal, but without labels
ax.set_xticks([])
ax.set_xlabel('Rat', fontsize=18)
ax.set_ylabel('Slope (k)', fontsize=18)
# Set y-ticks to 0.5 and 1
ax.set_yticks([0, 2])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# --- Save data for external plotting ---
slope_hist_data = {
    'slopes': slopes,
    'ABLS': ABLS,
    'animals': animals,
    'diff_within': diff_within,
    'diff_across': diff_across,
    'bins_absdiff': bins_absdiff,
    'hist_xlim': 0.4, # from max_xlim
    'hist_ylim': 8,   # from max_ylim
    'plot_colors': ['tab:blue', 'tab:orange', 'tab:green']
}

output_pickle_path = 'fig1_slopes_hists_data.pkl'
with open(output_pickle_path, 'wb') as f:
    pickle.dump(slope_hist_data, f)
print(f"\nSlope and histogram data saved to '{output_pickle_path}'")

# %%

# %% 
# overlayed abs of bias for each ABL 
COLORS = ['tab:blue', 'tab:orange', 'tab:green']
# For this, we need to extract and store x0 (bias) for each animal and ABL during the sigmoid fit.
# If not already done, re-fit or re-extract x0 for each animal and ABL.
# We'll re-fit here to guarantee x0 is available.
biases = {abl: {} for abl in ABLS}  # biases[abl][(batch_name, animal)] = x0
print(f'Number of animals: {len(animals)}')
for animal_id in animals:
    batch_name, animal = animal_id
    for idx, abl in enumerate(ABLS):
        animal_df = merged_valid[(merged_valid['batch_name'] == batch_name) & \
                                  (merged_valid['animal'] == animal) & \
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
                p0 = [0.05, 0.05, 1, 0]
                bounds = ([0, 0, -np.inf, -np.inf], [1, 1, np.inf, np.inf])
                popt, _ = curve_fit(sigmoid, animal_ilds[mask], psycho[mask], p0=p0, bounds=bounds, maxfev=5000)
                x0 = popt[3]
                biases[abl][animal_id] = x0
            except Exception as e:
                continue

# %%
# Now plot overlayed |bias| for each ABL
fig, ax = plt.subplots(figsize=(max(6, len(animals)//4), 3))  # wider if many animals
for idx, abl in enumerate(ABLS):
    color = COLORS[idx]
    y = [np.abs(biases[abl].get(animal, np.nan)) for animal in animals]
    ax.scatter(range(len(animals)), y, color=color, s=40)

# Set x-ticks and labels to batch-animal, vertical
animal_labels = [f"{batch}_{animal}" for batch, animal in animals]
ax.set_xticks(range(len(animals)))
ax.set_xticklabels(animal_labels, rotation=90, fontsize=8)

ax.axhline(0, color='k', linewidth=1)
ax.set_xlabel('')
ax.set_ylabel('|Bias| (|x0|)', fontsize=13)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# %% 
# average 
# Compute mean |bias| per animal across all ABLs
avg_abs_bias_per_animal = []
for animal in animals:
    abs_biases = [np.abs(biases[abl].get(animal, np.nan)) for abl in ABLS]
    # Compute mean, ignoring NaNs
    mean_abs_bias = np.nanmean(abs_biases)
    avg_abs_bias_per_animal.append(mean_abs_bias)

# Plot one point per animal (the mean |bias| across ABLs)
fig, ax = plt.subplots(figsize=(max(6, len(animals)//4), 3))
ax.scatter(range(len(animals)), avg_abs_bias_per_animal, color='k', s=60)  # single color, larger marker

# Set x-ticks and labels to batch-animal, vertical
animal_labels = [f"{batch}_{animal}" for batch, animal in animals]
ax.set_xticks(range(len(animals)))
ax.set_xticklabels(animal_labels, rotation=90, fontsize=8)

ax.axhline(0, color='k', linewidth=1)
ax.set_xlabel('')
ax.set_ylabel('Mean |Bias| (|x0|)', fontsize=13)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

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
# --- Compute and plot average lapse rates (mean of lambda_L and lambda_R) ---

# 1. Extract average lapse rates for each animal and ABL
lapse_rates = {abl: {} for abl in ABLS}  # lapse_rates[abl][(batch_name, animal)] = mean(lambda_L, lambda_R)
for animal_id in animals:
    batch_name, animal = animal_id
    for idx, abl in enumerate(ABLS):
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
                p0 = [0.05, 0.05, 1, 0]
                bounds = ([0, 0, -np.inf, -np.inf], [1, 1, np.inf, np.inf])
                popt, _ = curve_fit(sigmoid, animal_ilds[mask], psycho[mask], p0=p0, bounds=bounds, maxfev=5000)
                lambda_L, lambda_R = popt[0], popt[1]
                lapse_rates[abl][animal_id] = 0.5 * (lambda_L + lambda_R)
            except Exception as e:
                continue

# %%
# 2. Plot overlayed average lapse rates for each ABL
fig, ax = plt.subplots(figsize=(max(6, len(animals)//4), 3))
for idx, abl in enumerate(ABLS):
    color = COLORS[idx]
    y = [lapse_rates[abl].get(animal, np.nan) for animal in animals]
    ax.scatter(range(len(animals)), y, color=color, s=40)

animal_labels = [f"{batch}_{animal}" for batch, animal in animals]
ax.set_xticks(range(len(animals)))
ax.set_xticklabels(animal_labels, rotation=90, fontsize=8)

ax.axhline(0, color='k', linewidth=1)
ax.set_xlabel('')
ax.set_ylabel('mean of λ_L, λ_R', fontsize=13)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# 3. Plot mean average lapse rate per animal (across ABLs)
avg_lapse_per_animal = []
for animal in animals:
    lapses = [lapse_rates[abl].get(animal, np.nan) for abl in ABLS]
    mean_lapse = np.nanmean(lapses)
    avg_lapse_per_animal.append(mean_lapse)

fig, ax = plt.subplots(figsize=(max(6, len(animals)//4), 3))
ax.scatter(range(len(animals)), avg_lapse_per_animal, color='k', s=60)

ax.set_xticks(range(len(animals)))
ax.set_xticklabels(animal_labels, rotation=90, fontsize=8)

ax.axhline(0, color='k', linewidth=1)
ax.set_xlabel('')
ax.set_ylabel('ABL mean-λ_L,λ_R', fontsize=13)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# %%
# --- Compute and plot abs(slope differences) between ABLs for each animal ---

# 1. Extract slopes (k) for each animal and ABL
slopes = {abl: {} for abl in ABLS}  # slopes[abl][(batch_name, animal)] = k
for animal_id in animals:
    batch_name, animal = animal_id
    for idx, abl in enumerate(ABLS):
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
                p0 = [0.05, 0.05, 1, 0]
                bounds = ([0, 0, -np.inf, -np.inf], [1, 1, np.inf, np.inf])
                popt, _ = curve_fit(sigmoid, animal_ilds[mask], psycho[mask], p0=p0, bounds=bounds, maxfev=5000)
                k = popt[2]
                slopes[abl][animal_id] = k
            except Exception as e:
                continue

# 2. Compute absolute differences for each animal
abs_slope_20_40 = []
abs_slope_40_60 = []
abs_slope_20_60 = []

for animal in animals:
    k20 = slopes.get(20, {}).get(animal, np.nan)
    k40 = slopes.get(40, {}).get(animal, np.nan)
    k60 = slopes.get(60, {}).get(animal, np.nan)
    abs_slope_20_40.append(np.abs(k20 - k40))
    abs_slope_40_60.append(np.abs(k40 - k60))
    abs_slope_20_60.append(np.abs(k20 - k60))

animal_labels = [f"{batch}_{animal}" for batch, animal in animals]

# 3. Plot the absolute slope differences
fig, ax = plt.subplots(figsize=(max(6, len(animals)//4), 3))
ax.scatter(range(len(animals)), abs_slope_20_40, color='tab:blue', s=40, label='|slope_20 - slope_40|')
ax.scatter(range(len(animals)), abs_slope_40_60, color='tab:orange', s=40, label='|slope_40 - slope_60|')
ax.scatter(range(len(animals)), abs_slope_20_60, color='tab:green', s=40, label='|slope_20 - slope_60|')

ax.set_xticks(range(len(animals)))
ax.set_xticklabels(animal_labels, rotation=90, fontsize=8)
ax.axhline(0, color='k', linewidth=1)
ax.set_xlabel('')
ax.set_ylabel('|k_i - k_j|', fontsize=13)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend()
plt.tight_layout()
plt.show()

# %%
# --- Plot number of trials per ABL in each animal ---

# Prepare data: count trials per animal and ABL
trial_counts = {abl: [] for abl in ABLS}
for animal_id in animals:
    batch_name, animal = animal_id
    for abl in ABLS:
        count = merged_valid[
            (merged_valid['batch_name'] == batch_name) &
            (merged_valid['animal'] == animal) &
            (merged_valid['ABL'] == abl)
        ].shape[0]
        trial_counts[abl].append(count)

animal_labels = [f"{batch}_{animal}" for batch, animal in animals]
x = np.arange(len(animals))
print(f'len of animals: {len(animals)}')
fig, ax = plt.subplots(figsize=(max(6, len(animals)//4), 3))
for idx, abl in enumerate(ABLS):
    # Offset a bit for visibility, or just plot all at the same x
    ax.scatter(x + (idx - 1) * 0.08, np.log10(trial_counts[abl]), color=COLORS[idx], s=40, label=f'ABL {abl}')

ax.set_xticks(x)
ax.set_xticklabels(animal_labels, rotation=90, fontsize=8)
ax.set_ylabel('log10(Number of Trials)', fontsize=13)
ax.set_xlabel('')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.legend()
plt.tight_layout()
plt.show()