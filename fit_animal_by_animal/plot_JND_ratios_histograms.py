# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from scipy.optimize import curve_fit

# --- Sigmoid function and JND calculation ---
def sigmoid(x, lambda_L, lambda_R, k, x0):
    return lambda_L + (1 - lambda_L - lambda_R) / (1 + np.exp(-k * (x - x0)))

def inverse_sigmoid(y, lambda_L, lambda_R, k, x0):
    """Calculates the x-value for a given y-value of the sigmoid function."""
    # The term inside the log can be negative if y is not between lambda_L and 1-lambda_R
    log_arg = ((1 - lambda_L - lambda_R) / (y - lambda_L)) - 1
    if log_arg <= 0:
        return np.nan
    return x0 - (1/k) * np.log(log_arg)

def calculate_jnd(popt):
    """
    Calculates the Just Noticeable Difference (JND), accounting for lapses.
    Instead of fixed 0.75/0.25, uses 0.75 * max_prob_right and 0.25 * (1 - min_prob_right),
    where max_prob_right = sigmoid(+inf), min_prob_right = sigmoid(-inf)
    popt: array of fitted parameters [lambda_L, lambda_R, k, x0]
    """
    lambda_L, lambda_R, k, x0 = popt
    if k == 0:
        return np.nan
    # max and min probability of choosing right
    max_prob_right = sigmoid(np.inf, lambda_L, lambda_R, k, x0)  # as x -> +inf
    min_prob_right = sigmoid(-np.inf, lambda_L, lambda_R, k, x0) # as x -> -inf
    # Use new thresholds
    y_75 = 0.75 * max_prob_right
    y_25 = 0.25 * (1 - min_prob_right)
    ild_75 = inverse_sigmoid(y_75, lambda_L, lambda_R, k, x0)
    ild_25 = inverse_sigmoid(y_25, lambda_L, lambda_R, k, x0)
    if np.isnan(ild_75) or np.isnan(ild_25):
        return np.nan
    return (ild_75 - ild_25) / 2
    # return np.log(3) / k


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

# flag to tell if JND is averaged or pooled
animal_jnd = 'averaged' # other option is pooled
# --- 1. Fit sigmoid for each rat and ABL, store JND ---
jnds = {abl: {} for abl in ABLS}  # jnds[abl][(batch_name, animal)] = jnd
mean_jnd = {}  # mean_jnd[(batch_name, animal)] = jnd0
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
                jnd = calculate_jnd(popt)
                if not np.isnan(jnd):
                    jnds[abl][animal_id] = jnd
            except Exception as e:
                continue
    # 2. Compute mean JND for this animal depending on definition
    if animal_jnd == 'pooled':
        # Fit sigmoid to mean psychometric (across all ABLs) to get mean JND
        animal_df_all_abl = merged_valid[(merged_valid['batch_name'] == batch_name) &
                                         (merged_valid['animal'] == animal) &
                                         (merged_valid['ABL'].isin(ABLS))]
        if animal_df_all_abl.empty:
            continue
        # Compute mean P(Right) at each ILD - collapsed across ABLs
        ilds = np.sort(animal_df_all_abl['ILD'].unique())
        psycho = []
        for ild in ilds:
            sub = animal_df_all_abl[animal_df_all_abl['ILD'] == ild]
            psycho.append(np.mean(sub['choice'] == 1)) if len(sub) > 0 else psycho.append(np.nan)
        psycho = np.array(psycho)
        mask = ~np.isnan(psycho)
        if np.sum(mask) > 3:
            try:
                popt, _ = curve_fit(sigmoid, ilds[mask], psycho[mask],
                                    p0=[0.05, 0.05, 1, 0],
                                    bounds=([0, 0, -np.inf, -np.inf], [1, 1, np.inf, np.inf]),
                                    maxfev=5000)
                jnd0 = calculate_jnd(popt)
                if not np.isnan(jnd0):
                    mean_jnd[animal_id] = jnd0
            except Exception:
                pass
    elif animal_jnd == 'averaged':
        # Average of per-ABL JNDs (take mean over ABLs where JND was successfully fit)
        per_abl_jnds = [jnds[abl][animal_id] for abl in ABLS if animal_id in jnds[abl]]
        if len(per_abl_jnds) > 0:
            mean_jnd[animal_id] = np.mean(per_abl_jnds)

# --- 2. log-ratio of JND at each ABL to mean JND for that rat ---
log_ratios_within = []
for animal_id in all_animals:
    if animal_id not in mean_jnd:
        continue
    jnd0 = mean_jnd[animal_id]
    for abl in ABLS:
        if animal_id in jnds[abl]:
            # We want smaller JND to be better, so ratio should be JND0/JND
            ratio = jnd0 / jnds[abl][animal_id]
            log_ratios_within.append(np.log(ratio))

# --- 4. log-ratio of mean JND for each rat to grand mean ---
# Only use animals present in mean_jnd
animals_with_mean = list(mean_jnd.keys())
grand_mean_jnd = np.mean([mean_jnd[animal_id] for animal_id in animals_with_mean])
# We want smaller JND to be better, so ratio should be grand_mean/mean_jnd
log_ratios_across = [np.log(grand_mean_jnd / mean_jnd[animal_id]) for animal_id in animals_with_mean]

# --- 5. Plot histograms ---

ratios_within = np.exp(log_ratios_within)
ratios_across = np.exp(log_ratios_across)

diff_within = []  # jnd_ABL - mean_jnd_rat
for animal_id in all_animals:
    if animal_id not in mean_jnd:
        continue
    jnd0 = mean_jnd[animal_id]
    for abl in ABLS:
        if animal_id in jnds[abl]:
            diff_within.append(jnds[abl][animal_id] - jnd0)

mean_jnds = np.array([mean_jnd[animal_id] for animal_id in animals_with_mean])
diff_across = mean_jnds - grand_mean_jnd  # mean_jnd_rat - grand_mean

# --- Save data for external plotting ---
jnd_data_for_plotting = {
    'jnds': jnds,
    'mean_jnd': mean_jnd,
    'grand_mean_jnd': grand_mean_jnd,
    'diff_within': diff_within,
    'diff_across': diff_across,
    'ratios_within': ratios_within,
    'ratios_across': ratios_across,
    'animals_with_mean': animals_with_mean,
    'mean_jnds': mean_jnds,
    'ABLS': ABLS,
}

output_dir = os.path.dirname(__file__)
pickle_path = os.path.join(output_dir, 'jnd_analysis_data.pkl')

print(f"Saving JND analysis data to {pickle_path}...")
with open(pickle_path, 'wb') as f:
    pickle.dump(jnd_data_for_plotting, f)

print("Data saved successfully.")

bins_within = np.arange(0, 4, 0.1)
bins_across = np.arange(0, 4, 0.1)
bins_diff = np.arange(-5, 5, 0.2)
bins_absdiff = np.arange(-5, 5, 0.2)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.hist(ratios_within, bins=bins_within, color='tab:blue', alpha=0.7, density=True)
plt.axvline(1, color='k', linestyle='--')
plt.title('Within-rat (mean_JND_rat / JND_ABL)')
plt.xlabel('JND Ratio (smaller is worse)')
plt.ylabel('Density')
plt.ylim(0, 2)
ax1 = plt.gca()
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

plt.subplot(1,2,2)
plt.hist(ratios_across, bins=bins_across, color='tab:orange', alpha=0.7, density=True)
plt.axvline(1, color='k', linestyle='--')
plt.title('Across-rat (grand_mean_JND / mean_JND_rat)')
plt.xlabel('JND Ratio (smaller is worse)')
ax2 = plt.gca()
ax2.set_ylabel("")
ax2.set_yticklabels([])
ax2.set_yticks([])
plt.ylim(0, 2)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
# %%
# --- NEW: Plot histogram of absolute differences ---
font = {'size': 18}
plt.rc('font', **font)
max_ylim = 1.5
plt.figure(figsize=(12, 6))
max_xlim = 4

# Subplot 1: Within-rat differences
ax1 = plt.subplot(1, 2, 1)
plt.hist(diff_within, bins=bins_absdiff, color='grey', alpha=0.7, density=True)
plt.axvline(0, color='k', linestyle=':')
# plt.title('Within-rat', fontsize=18)
plt.xlabel(r'$JND_{ABL} - JND_{rat}$', fontsize=18)
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
plt.xlabel(r'$JND_{rat} - JND_{grand}$', fontsize=18)
plt.xlim(-max_xlim, max_xlim)
plt.ylim(0, max_ylim)
plt.xticks([-max_xlim, 0, max_xlim])
plt.yticks([0, max_ylim])
ax2.set_yticklabels([])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.tick_params(axis='both', which='major', labelsize=18)

plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to prevent title overlap
# plt.suptitle('JND Differences', fontsize=18, y=1.02)
plt.show()

# %%
# std of diff_with and diff_across
print(f'std(JND diff_within animal): {np.std(diff_within):.3f}')
print(f'std(JND diff_across animals): {np.std(diff_across):.3f}')

print(f'std(JND ratio_within animal) {np.std(ratios_within):.3f}')
print(f'std(JND ratio_across animal) {np.std(ratios_across):.3f}')

# %%
##### JND RAT - GRAND MEAN: CENTERED #########################
######################################################
# Create a figure with two subplots, sharing the y-axis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6), sharey=True, gridspec_kw={'width_ratios': [2, 1]})

# Left plot: Deviation of each animal's mean and per-ABL JND from the grand mean
COLORS = ['tab:blue', 'tab:orange', 'tab:green']
# Sort animals based on their mean JND deviation for consistent plotting
sorted_animal_indices = np.argsort(diff_across)
sorted_animals = [animals_with_mean[i] for i in sorted_animal_indices]

for i, animal_id in enumerate(sorted_animals):
    # Plot mean JND deviation (sorted)
    mean_deviation = mean_jnd[animal_id] - grand_mean_jnd
    ax1.plot(i, mean_deviation, 'ko', markersize=8, label='Mean JND' if i == 0 else "")

    # Plot per-ABL JND deviations
    for j, abl in enumerate(ABLS):
        if animal_id in jnds[abl]:
            abl_deviation = jnds[abl][animal_id] - grand_mean_jnd
            ax1.plot(i, abl_deviation, 'o', color=COLORS[j], markersize=5, alpha=0.8, label=f'ABL {abl}' if i == 0 else "")

ax1.axhline(0, color='k', linestyle=':', linewidth=2)  # Center line at zero
ax1.get_xaxis().set_visible(False)  # Hide x-axis ticks and labels
ax1.set_ylabel('JND - Grand Mean JND')
ax1.set_title('Deviation from Grand Mean')
ax1.legend()
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# --- Publication Grade Adjustments ---
# Remove top and right spines from the left plot
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)


# Set specific y-ticks for the left plot
ax1.set_yticks([-2, 0, 2])
ax1.set_ylim([-3, 3])


# Right plot: Horizontal histogram of the differences
ax2.hist(diff_across, bins=bins_absdiff, color='gray', edgecolor='black', orientation='horizontal', density=True)

# Remove all axes, labels, and ticks from the histogram plot
ax2.axis('off')

# Adjust layout
plt.subplots_adjust(wspace=0.05)

plt.tight_layout()
plt.show()
#

# %%
##### PLOT OF MEAN JNDS WITH GRAND MEAN & HISTOGRAM ##########
################################################################
# Create a figure with two subplots, sharing the y-axis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6), sharey=True, gridspec_kw={'width_ratios': [2, 1]})

sorted_animal_indices = np.argsort(mean_jnds)
sorted_animals = [animals_with_mean[i] for i in sorted_animal_indices]

for i, animal_id in enumerate(sorted_animals):
    ax1.plot(i, mean_jnd[animal_id], 'k_', markersize=12, mew=2) # 'k_' for black hline marker, mew for thickness

    for j, abl in enumerate(ABLS):
        if animal_id in jnds[abl]:
            ax1.plot(i, jnds[abl][animal_id], 'o', color=COLORS[j], markersize=5, alpha=0.6)

ax1.axhline(grand_mean_jnd, color='k', linestyle=':', linewidth=2, label=f'Grand Mean = {grand_mean_jnd:.2f}')  # Grand mean line
ax1.get_xaxis().set_visible(False)  # Hide x-axis ticks and labels
ax1.set_ylabel('JND')
ax1.set_title('JNDs per Animal')
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.set_yticks([0, 6])
ax1.set_ylim([0, 6])

# --- Publication Grade Adjustments ---
# Remove top and right spines from the left plot
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)

# Right plot: Horizontal histogram of the mean JNDs
bins_mean_jnds = np.arange(0, 6, 0.5)
ax2.hist(mean_jnds, bins=bins_mean_jnds, color='gray', edgecolor='black', orientation='horizontal', density=True)

# Remove all axes, labels, and ticks from the histogram plot
ax2.axis('off')
# Adjust layout
plt.subplots_adjust(wspace=0.05)

plt.tight_layout()
plt.show()
# %%
sorted_animals

#
# --- Plot psychometric curves (p(choose right) vs ILD) for each animal, ABL wise and mean ---
ild_grid = np.arange(-16, 16.01, 0.1)
COLORS = ['tab:blue', 'tab:orange', 'tab:green']

n_animals = len(sorted_animals)
ncols = 4
nrows = int(np.ceil(n_animals / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3), sharex=True, sharey=True)
axes = axes.flatten()

for idx, animal_id in enumerate(sorted_animals):
    ax = axes[idx]
    for j, abl in enumerate(ABLS):
        if animal_id in jnds[abl]:
            animal_df = merged_valid[(merged_valid['batch_name'] == animal_id[0]) &
                                     (merged_valid['animal'] == animal_id[1]) &
                                     (merged_valid['ABL'] == abl)]
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
                    y = sigmoid(ild_grid, *popt)
                    ax.plot(ild_grid, y, color=COLORS[j], label=f'ABL {abl}')
                    # Draw vertical line at JND
                    jnd = calculate_jnd(popt)
                    if not np.isnan(jnd):
                        ax.axvline(jnd, color=COLORS[j], linewidth=1)
                except Exception as e:
                    pass
    # Mean psychometric (across ABLs)
    animal_df_all_abl = merged_valid[(merged_valid['batch_name'] == animal_id[0]) &
                                    (merged_valid['animal'] == animal_id[1]) &
                                    (merged_valid['ABL'].isin(ABLS))]
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
            p0 = [0.05, 0.05, 1, 0]
            bounds = ([0, 0, -np.inf, -np.inf], [1, 1, np.inf, np.inf])
            popt, _ = curve_fit(sigmoid, ilds[mask], psycho[mask], p0=p0, bounds=bounds, maxfev=5000)
            y = sigmoid(ild_grid, *popt)
            ax.plot(ild_grid, y, 'k--', label='Mean (all ABLs)', linewidth=2)
            # Draw vertical line at JND for mean
            jnd = calculate_jnd(popt)
            if not np.isnan(jnd):
                ax.axvline(jnd, color='k', linewidth=1)
        except Exception as e:
            pass
    ax.set_title(f'{animal_id}')
    ax.set_xlim(2, 5)
    # ax.set_xlim(-16, 16)

    ax.set_ylim(-0.05, 1.05)
    if idx % ncols == 0:
        ax.set_ylabel('P(choose right)')
    if idx >= (nrows-1)*ncols:
        ax.set_xlabel('ILD (dB)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.axhline(0.5, alpha=0.5)
    ax.axvline(0, alpha=0.5)
    # if idx == 0:
    #     ax.legend()


# Hide unused axes
for idx in range(n_animals, nrows * ncols):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.show()

#

# %%
##### WITHIN-ANIMAL JND VARIABILITY ##########################
################################################################
# Create a figure with two subplots, sharing the y-axis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6), sharey=True, gridspec_kw={'width_ratios': [2, 1]})

# Left plot: Deviation of each ABL JND from the animal's mean JND
COLORS = ['tab:blue', 'tab:orange', 'tab:green']
animals_with_mean = list(mean_jnd.keys())

for i, animal_id in enumerate(animals_with_mean):
    jnd0 = mean_jnd[animal_id]
    for j, abl in enumerate(ABLS):
        if animal_id in jnds[abl]:
            diff = jnds[abl][animal_id] - jnd0
            ax1.plot(i, diff, 'o', color=COLORS[j], markersize=8, alpha=0.7)

ax1.axhline(0, color='k', linestyle=':', linewidth=2)  # Center line at zero
ax1.get_xaxis().set_visible(False)  # Hide x-axis ticks and labels
ax1.set_ylabel('JND(ABL) - Mean JND(Animal)')
ax1.set_title('Within-Animal JND Deviation by ABL')
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.set_yticks([-3, 0, 3])
ax1.set_ylim([-3, 3])
# --- Publication Grade Adjustments ---
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)

# Right plot: Horizontal histogram of the differences
ax2.hist(diff_within, bins=bins_absdiff, color='gray', edgecolor='black', orientation='horizontal', density=True)
ax2.axis('off')

# Adjust layout
plt.subplots_adjust(wspace=0.05)
plt.tight_layout()
plt.show()


# %%
# --- 6. Per-ABL JND plot for each animal ---
COLORS = ['tab:blue', 'tab:orange', 'tab:green']
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for idx, abl in enumerate(ABLS):
    ax = axes[idx]
    color = COLORS[idx]
    # Get animals with JND for this ABL
    animals_ids = sorted([animal_id for animal_id in jnds[abl].keys()])
    jnd_vals = [jnds[abl][animal_id] for animal_id in animals_ids]
    ax.scatter(range(len(animals_ids)), jnd_vals, color=color, s=40)
    ax.set_title(f'ABL = {abl}')
    ax.set_xlabel('Animal')
    if idx == 0:
        ax.set_ylabel('JND')
    ax.set_xticks(range(len(animals_ids)))
    # Create readable labels by combining batch_name and animal
    animal_labels = [f"{batch}-{animal}" for batch, animal in animals_ids]
    ax.set_xticklabels(animal_labels, rotation=90, fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()


# %%
# --- 7. Overlayed animal JNDs for all ABLs ---
COLORS = ['tab:blue', 'tab:orange', 'tab:green']
animals = sorted(set().union(*[jnds[abl].keys() for abl in ABLS]))
print(f'len of animals: {len(animals)}')
print(animals)
fig, ax = plt.subplots(figsize=(6, 3))  # Compressed x-axis
for idx, abl in enumerate(ABLS):
    color = COLORS[idx]
    y = [jnds[abl].get(animal, np.nan) for animal in animals]
    ax.scatter(range(len(animals)), y, color=color, s=40)
# Set x-ticks to represent each animal, but without labels
ax.set_xticks([])
ax.set_xlabel('Rat', fontsize=18)
ax.set_ylabel('JND', fontsize=18)
# Set y-ticks
ax.set_yticks([0, 5, 10])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()


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