# %%
########## Make a RTD for all animals #######
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# %%

# Read and merge the three batch CSVs for LED7, LED6, and Comparable
batch_dir = os.path.join(os.path.dirname(__file__), 'batch_csvs')
batch_files = [f for f in os.listdir(batch_dir) if f.endswith('_valid_and_aborts.csv')]
print(batch_files)
merged_data = pd.concat([
    pd.read_csv(os.path.join(batch_dir, fname)) for fname in batch_files
], ignore_index=True)
merged_valid = merged_data[merged_data['success'].isin([1, -1])].copy()

# %%
print(merged_valid['batch_name'].unique())
# batch_name is nan, fill it with LED7


# %%
print(merged_valid['ABL'].unique())
print(merged_valid['ILD'].unique())

# %%
# add abs_ILD column
merged_valid['abs_ILD'] = merged_valid['ILD'].abs()

# %%
check_ILD_10 = merged_valid[merged_valid['abs_ILD'] == 10]
print(check_ILD_10['ABL'].unique())
print(f'len(check_ILD_10): {len(check_ILD_10)}')

check_ILD_6 = merged_valid[merged_valid['abs_ILD'] == 6]
print(check_ILD_6['ABL'].unique())
print(f'len(check_ILD_6): {len(check_ILD_6)}')

check_ABL_50 = merged_valid[merged_valid['ABL'] == 50]
print(check_ABL_50['abs_ILD'].unique())
print(f'len(check_ABL_50): {len(check_ABL_50)}')


# abs ILD 10,6 are very low, just remove them
merged_valid = merged_valid[merged_valid['abs_ILD'] != 10]
merged_valid = merged_valid[merged_valid['abs_ILD'] != 6]




# %%
# Get all unique batch names and count total animals for subplot layout
batch_names = merged_valid['batch_name'].unique()
print(batch_names)

# Count total number of animals to determine subplot layout
total_animals = 0
animal_data = []
for batch_name in batch_names:
    batch_df = merged_valid[merged_valid['batch_name'] == batch_name]
    batch_animals = batch_df['animal'].unique()
    total_animals += len(batch_animals)
    
    # Store data for each animal
    for animal in batch_animals:
        animal_df = batch_df[batch_df['animal'] == animal]
        animal_data.append({
            'batch_name': batch_name,
            'animal': animal,
            'df': animal_df
        })

# Calculate number of rows needed (with 5 columns)
num_cols = 5
num_rows = (total_animals + num_cols - 1) // num_cols  # Ceiling division

# Make each subplot square
subplot_size = 5
fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * subplot_size, num_rows * subplot_size))
# Flatten axes array for easier indexing
axes = axes.flatten() if num_rows > 1 else [axes] if num_cols == 1 else axes

# Dynamically collect all unique ABLs from all animal dataframes
all_abls = set()
for animal_info in animal_data:
    animal_df = animal_info['df']
    all_abls.update(animal_df['ABL'].unique())
all_abls = sorted(list(all_abls))

# Assign colors using matplotlib's tab10 palette (up to 10 ABLs)
import matplotlib
palette = matplotlib.cm.get_cmap('tab10', max(10, len(all_abls)))
abl_color_map = {abl: palette(i) for i, abl in enumerate(all_abls)}
from matplotlib.lines import Line2D

# Compute global min/max ILD across all animals for consistent x-axis
all_ILDs = merged_valid['ILD'].unique()
global_min_ILD = np.min(all_ILDs)
global_max_ILD = np.max(all_ILDs)

# Plot each animal's Reaction Time Distribution (RTD) in its own subplot
for i, animal_info in enumerate(animal_data):
    batch_name = animal_info['batch_name']
    animal = animal_info['animal']
    animal_df = animal_info['df']
    
    ax = axes[i]
    # Plot RTD for each ABL
    for abl in all_abls:
        color = abl_color_map.get(abl, None)
        # Filter for this ABL and valid RTwrtStim
        rtd_df = animal_df[(animal_df['ABL'] == abl) & (animal_df['RTwrtStim'] > 0) & (animal_df['RTwrtStim'] < 1)]
        rt_values = rtd_df['RTwrtStim'].dropna().values
        if len(rt_values) == 0:
            continue
        bins = np.arange(0, 1.02, 0.02)  # 0 to 1 s in 20 ms bins
        ax.hist(rt_values, bins=bins, color=color, label=f'ABL {abl}', histtype='step', density=True, linewidth=2)
    # Formatting
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 12)
    ax.set_yticks([0, 6, 12])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    row_idx = i // num_cols
    col_idx = i % num_cols
    # Only show x-ticks on bottom row
    if row_idx != num_rows - 1:
        ax.set_xticklabels([])
        ax.set_xlabel("")
    else:
        ax.set_xlabel('Reaction Time (s)', fontsize=24)
    # Only show y-ticks on leftmost column
    if col_idx != 0:
        ax.set_yticklabels([])
        ax.set_ylabel("")
    else:
        ax.set_ylabel('Density', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=22, length=8, width=2.5)
    ax.set_title(f'Batch: {batch_name}, Animal: {animal}', fontsize=20)
    # No per-plot legend

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(wspace=0.4, hspace=0.5)  # Increase spacing between subplots

# Add a single legend for ABLs (figure-level)
legend_handles = [Line2D([0], [0], marker='o', color=color, linestyle='-', label=f'ABL {abl}')
                  for abl, color in abl_color_map.items()]
fig.legend(handles=legend_handles, title='ABL', loc='lower right', bbox_to_anchor=(1, 0), fontsize=24, title_fontsize=26)

# Save figure
plt.savefig(f'all_animals_RTD_all_batches_with_LED8.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# === RTD Aggregate Plot (1x4) ===
import matplotlib.pyplot as plt
import numpy as np

ABLS = [20, 40, 60]
ILDS = [1, -1, 2, -2, 4, -4, 8, -8, 16, -16]
bins = np.arange(0, 1.02, 0.02)
from matplotlib.cm import tab10
colors = [tab10(0), tab10(1), tab10(2)]  # blue, orange, green

# Collect RTDs per animal, per ABL
animal_rtds = {abl: [] for abl in ABLS}
for animal_info in animal_data:
    animal_df = animal_info['df']
    for abl in ABLS:
        # Filter for ABL and ILDs of interest, valid RTwrtStim
        df = animal_df[(animal_df['ABL'] == abl) &
                       (animal_df['ILD'].isin(ILDS)) &
                       (animal_df['RTwrtStim'] > 0) & (animal_df['RTwrtStim'] < 1)]
        rts = df['RTwrtStim'].dropna().values
        if len(rts) > 0:
            animal_rtds[abl].append(rts)

# Compute average RTD per ABL
avg_rtds = {}
for i, abl in enumerate(ABLS):
    # Histogram each animal, then average
    all_hist = []
    for rts in animal_rtds[abl]:
        hist, _ = np.histogram(rts, bins=bins, density=True)
        all_hist.append(hist)
    if all_hist:
        avg_rtds[abl] = np.mean(np.stack(all_hist), axis=0)
    else:
        avg_rtds[abl] = np.zeros(len(bins)-1)

# Plot
fig, axes = plt.subplots(1, 4, figsize=(28, 6), sharey=True)
for idx, abl in enumerate(ABLS):
    ax = axes[idx]
    # Plot all animals (light)
    for rts in animal_rtds[abl]:
        hist, _ = np.histogram(rts, bins=bins, density=True)
        ax.plot(bins[:-1], hist, color=colors[idx], alpha=0.3, linewidth=1)
    # Plot average (dark)
    ax.plot(bins[:-1], avg_rtds[abl], color=colors[idx], alpha=1, linewidth=3, label=f'ABL {abl} avg')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Reaction Time (s)', fontsize=18)
    if idx == 0:
        ax.set_ylabel('Density', fontsize=18)
    ax.set_title(f'ABL {abl}', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.legend(fontsize=15)

# Fourth subplot: overlay all averages
ax = axes[3]
for idx, abl in enumerate(ABLS):
    ax.plot(bins[:-1], avg_rtds[abl], color=colors[idx], alpha=1, linewidth=3, label=f'ABL {abl}')
ax.set_xlim(0, 1)
ax.set_xlabel('Reaction Time (s)', fontsize=18)
ax.set_title('All ABL Averages', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.legend(fontsize=15)
plt.tight_layout()
plt.savefig('all_animals_rtd_aggregate_20_40_60_reg_ILDs.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# === Per-Animal x abs_ILD RTD Grid Plot ===
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import tab10

ABLS = [20, 40, 60]
ABS_ILDS = [1, 2, 4, 8, 16]
bins = np.arange(0, 1.02, 0.02)
colors = [tab10(0), tab10(1), tab10(2)]  # blue, orange, green

# Build a unique list of (batch_name, animal) pairs
animal_keys = [(info['batch_name'], info['animal']) for info in animal_data]
unique_keys = [(b, a) for (b, a) in animal_keys]
n_rows = len(unique_keys)
fig, axes = plt.subplots(n_rows, len(ABS_ILDS), figsize=(5*len(ABS_ILDS), 3*n_rows), sharex=True, sharey=True)
if n_rows == 1:
    axes = axes[np.newaxis, :]
if len(ABS_ILDS) == 1:
    axes = axes[:, np.newaxis]

for row, animal_info in enumerate(animal_data):
    batch_name = animal_info['batch_name']
    animal = animal_info['animal']
    animal_df = animal_info['df'].copy()
    animal_df['abs_ILD'] = animal_df['ILD'].abs()
    key = f'{batch_name}_{animal}'
    for col, abs_ild in enumerate(ABS_ILDS):
        ax = axes[row, col] if n_rows > 1 else axes[0, col]
        for idx, abl in enumerate(ABLS):
            df = animal_df[(animal_df['ABL'] == abl) & (animal_df['abs_ILD'] == abs_ild) & (animal_df['RTwrtStim'] > 0) & (animal_df['RTwrtStim'] < 1)]
            rts = df['RTwrtStim'].dropna().values
            if len(rts) > 0:
                hist, _ = np.histogram(rts, bins=bins, density=True)
                ax.plot(bins[:-1], hist, color=colors[idx], alpha=1, linewidth=2, label=f'ABL {abl}')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 15)
        ax.set_yticks([0, 15])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if row == n_rows - 1:
            ax.set_xlabel('RT (s)', fontsize=14)
        # if col == 0:
        #     ax.set_ylabel('Density', fontsize=14)
        if row == 0:
            ax.set_title(f'abs_ILD={abs_ild}', fontsize=16)
        if col == len(ABS_ILDS) - 1:
            ax.legend(fontsize=11)
        if col == 0:
            ax.annotate(f'{batch_name}_{animal}', xy=(0, 0.5), xycoords='axes fraction', fontsize=14, ha='right', va='center', rotation=90)
plt.tight_layout()
plt.savefig('per_animal_rtd_grid.png', dpi=300, bbox_inches='tight')

# %%
# check LED 2
batch_name = 'LED2'
print(f'Batch: {batch_name}')
led2_data = [animal_info for animal_info in animal_data if animal_info['batch_name'] == batch_name]
for animal_led2 in led2_data:
    print(f'animal: {animal_led2["animal"]}')
    led_animal_df = animal_led2['df']
    led_animal_df['abs_ILD'] = led_animal_df['ILD'].abs()
    # RTwrtStim > 0 < 1
    led_animal_df = led_animal_df[(led_animal_df['RTwrtStim'] > 0) & (led_animal_df['RTwrtStim'] < 1)]
    for abs_ild in led_animal_df['abs_ILD'].unique():
        led_animal_df_abs_ild = led_animal_df[led_animal_df['abs_ILD'] == abs_ild]
        for abl in [20, 40, 60]:
            count = len(led_animal_df_abs_ild[led_animal_df_abs_ild['ABL'] == abl])
            print(f'animal: {animal_led2["animal"]}, abs_ILD={abs_ild}, ABL={abl}, count={count}')
    