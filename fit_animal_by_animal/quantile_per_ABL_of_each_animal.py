# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Load merged_valid as in reference scripts ---
batch_dir = os.path.join(os.path.dirname(__file__), 'batch_csvs')
batch_files = [f for f in os.listdir(batch_dir) if f.endswith('_valid_and_aborts.csv')]
merged_data = pd.concat([
    pd.read_csv(os.path.join(batch_dir, fname)) for fname in batch_files
], ignore_index=True)
merged_valid = merged_data[merged_data['success'].isin([1, -1])].copy()
merged_valid['batch_name'] = merged_valid['batch_name'].fillna('LED7')

# Remove RTs > 1 in valid trials
merged_valid = merged_valid[merged_valid['RTwrtStim'] <= 1]

# Add abs_ILD column
merged_valid['abs_ILD'] = merged_valid['ILD'].abs()

# Remove ILD 10 and ILD 6, very few rows
merged_valid = merged_valid[~merged_valid['abs_ILD'].isin([6, 10])]

# Quantiles to compute
quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
quantile_labels = ['Q10', 'Q30', 'Q50', 'Q70', 'Q90']
quantile_linestyles = ['dotted', 'dashdot', 'solid', 'dashdot', 'dotted']

# Get all unique batch names and count total animals for subplot layout
batch_names = merged_valid['batch_name'].unique()
total_animals = 0
animal_data = []
for batch_name in batch_names:
    batch_df = merged_valid[merged_valid['batch_name'] == batch_name]
    batch_animals = batch_df['animal'].unique()
    total_animals += len(batch_animals)
    for animal in batch_animals:
        animal_df = batch_df[batch_df['animal'] == animal]
        animal_data.append({
            'batch_name': batch_name,
            'animal': animal,
            'df': animal_df
        })

# Build a list of all (batch, animal, ABL) combos
combos = []
for animal_info in animal_data:
    batch_name = animal_info['batch_name']
    animal = animal_info['animal']
    df = animal_info['df']
    for abl in sorted(df['ABL'].unique()):
        combos.append({'batch_name': batch_name, 'animal': animal, 'ABL': abl, 'df': df[df['ABL'] == abl]})

num_cols = 5
num_rows = (len(combos) + num_cols - 1) // num_cols
subplot_size = 5
fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * subplot_size, num_rows * subplot_size))
axes = axes.flatten() if num_rows > 1 else [axes] if num_cols == 1 else axes

xticks = [0, 4, 8, 12, 16]

# Set up ABL color map (same as mean_chrono_plot.py)
fixed_abl_colors = {20: '#1f77b4', 40: '#ff7f0e', 60: '#2ca02c'}
all_abls = sorted(set(merged_valid['ABL'].unique()))
tab10 = plt.get_cmap('tab10').colors
palette_indices = [i for i in range(len(tab10)) if i not in [0, 1, 2]]
palette = [tab10[i] for i in palette_indices]
other_abls = [abl for abl in all_abls if abl not in fixed_abl_colors]
other_abl_colors = {abl: palette[i % len(palette)] for i, abl in enumerate(other_abls)}
abl_color_map = {**fixed_abl_colors, **other_abl_colors}
def get_abl_color(abl):
    return abl_color_map.get(abl, '#888888')

for idx, combo in enumerate(combos):
    ax = axes[idx]
    batch_name = combo['batch_name']
    animal = combo['animal']
    abl = combo['ABL']
    df = combo['df']
    abs_ilds = sorted(df['abs_ILD'].unique())
    color = get_abl_color(abl)
    for qidx, q in enumerate(quantiles):
        yvals = []
        for abs_ild in abs_ilds:
            subset = df[df['abs_ILD'] == abs_ild]
            if len(subset) > 0:
                yvals.append(np.quantile(subset['RTwrtStim'], q))
            else:
                yvals.append(np.nan)
        ax.plot(abs_ilds, yvals, marker='o', color=color, linestyle=quantile_linestyles[qidx], linewidth=2, alpha=0.8)
    ax.set_title(f'{batch_name} | {animal} | ABL {abl}', fontsize=18)

    ax.set_xlim([0, 16])
    ax.set_xticks(xticks)
    ax.set_ylim([0, 1])
    # Only show y-ticks for first column
    if idx % num_cols != 0:
        ax.set_yticklabels([])
        ax.set_ylabel("")
        ax.spines['left'].set_color('gray')
    else:
        ax.set_ylabel('RTwrtStim', fontsize=18)
        ax.spines['left'].set_color('black')
    # Only show x-ticks for last row
    if idx < (num_rows - 1) * num_cols:
        ax.set_xticklabels([])
        ax.set_xlabel("")
    else:
        ax.set_xlabel('| ILD |', fontsize=18)
    ax.tick_params(axis='both', labelsize=18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Hide unused axes
for i in range(idx + 1, len(axes)):
    axes[i].axis('off')

plt.tight_layout()

plt.show()
