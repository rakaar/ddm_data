# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Load merged_valid as in make_all_animals_psycho_single_figure.py ---
batch_dir = os.path.join(os.path.dirname(__file__), 'batch_csvs')
batch_files = [f for f in os.listdir(batch_dir) if f.endswith('_valid_and_aborts.csv')]
merged_data = pd.concat([
    pd.read_csv(os.path.join(batch_dir, fname)) for fname in batch_files
], ignore_index=True)
merged_valid = merged_data[merged_data['success'].isin([1, -1])].copy()
merged_valid['batch_name'] = merged_valid['batch_name'].fillna('LED7')

# remove RTs > 1 in valid trials
merged_valid = merged_valid[merged_valid['RTwrtStim'] <= 1]

# add abs_ILD column
merged_valid['abs_ILD'] = merged_valid['ILD'].abs()

# Remove ILD 10 and ILD 6, very few rows
merged_valid = merged_valid[~merged_valid['abs_ILD'].isin([6, 10])]

# Set to either 'mean' or 'median'
mean_or_median = "median"  # or "median"

# Get all unique batch names and count total animals for subplot layout
batch_names = merged_valid['batch_name'].unique()

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

subplot_size = 5
fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * subplot_size, num_rows * subplot_size))
axes = axes.flatten() if num_rows > 1 else [axes] if num_cols == 1 else axes

# Use fixed color scheme for ABL 20, 40, 60, else default to gray
abl_color_map = {
    20: '#1f77b4',   # blue
    40: '#ff7f0e',   # orange
    60: '#2ca02c',   # green
}
def get_abl_color(abl):
    return abl_color_map.get(abl, '#888888')  # gray for others

# --- Compute global x/y limits ---
all_plot_dicts = []
for animal_info in animal_data:
    df = animal_info['df']
    plot_dict = {}
    for abl in df['ABL'].unique():
        abl_df = df[df['ABL'] == abl]
        values = []
        xvals = []
        stds = []
        for abs_ild in sorted(abl_df['abs_ILD'].unique()):
            subset = abl_df[abl_df['abs_ILD'] == abs_ild]
            if mean_or_median == "mean":
                val = subset['RTwrtStim'].mean()
                std = subset['RTwrtStim'].std()
                stds.append(std)
            else:
                val = subset['RTwrtStim'].median()
            values.append(val)
            xvals.append(abs_ild)
        if mean_or_median == "mean":
            plot_dict[abl] = (xvals, values, stds)
        else:
            plot_dict[abl] = (xvals, values)
    all_plot_dicts.append(plot_dict)

# --- Set fixed ticks and axis limits as requested ---
xticks = [0, 4, 8, 12, 16]
yticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

# --- Option to plot standard deviation error bars ---
is_std_plot = False  # Set to False to disable std error bars

# --- Assign unique, non-overlapping colors to all ABLs ---
# Fixed colors for 20, 40, 60
fixed_abl_colors = {
    20: '#1f77b4',   # blue
    40: '#ff7f0e',   # orange
    60: '#2ca02c',   # green
}
# All ABLs present in the data
all_abls = sorted(set(merged_valid['ABL'].unique()))
# Build color list from tab10, skipping indices 0, 1, 2 (used above)
tab10 = plt.get_cmap('tab10').colors
palette_indices = [i for i in range(len(tab10)) if i not in [0, 1, 2]]
palette = [tab10[i] for i in palette_indices]
# Assign palette colors to other ABLs
other_abls = [abl for abl in all_abls if abl not in fixed_abl_colors]
other_abl_colors = {abl: palette[i % len(palette)] for i, abl in enumerate(other_abls)}
# Merge color maps
abl_color_map = {**fixed_abl_colors, **other_abl_colors}

def get_abl_color(abl):
    return abl_color_map.get(abl, '#888888')  # fallback gray (should not happen)

# --- Plot for each animal ---
for idx, animal_info in enumerate(animal_data):
    ax = axes[idx]
    batch_name = animal_info['batch_name']
    animal = animal_info['animal']
    plot_dict = all_plot_dicts[idx]
    for abl in sorted(plot_dict):
        vals = plot_dict[abl]
        color = get_abl_color(abl)
        if mean_or_median == "mean":
            xvals, yvals, stds = vals
            if is_std_plot:
                ax.errorbar(xvals, yvals, yerr=stds, marker='o', label=f'ABL {abl}', color=color, capsize=0, linestyle='-', linewidth=2)
            else:
                ax.plot(xvals, yvals, marker='o', label=f'ABL {abl}', color=color, linewidth=2)
        else:
            xvals, yvals = vals
            ax.plot(xvals, yvals, marker='o', label=f'ABL {abl}', color=color)
    ax.set_title(f'{batch_name} | {animal}', fontsize=18)
    ax.set_xlim([0, 16])
    ax.set_xticks(xticks)
    ax.set_ylim([0, 0.5])
    ax.set_yticks(yticks)
    # Only show y-ticks for first column
    if idx % num_cols != 0:
        ax.set_yticklabels([])
        ax.set_ylabel("")
        ax.spines['left'].set_color('gray')
    else:
        ax.set_ylabel(f'{mean_or_median} RT', fontsize=18)
        ax.spines['left'].set_color('black')
    # Only show x-ticks for last row
    if idx < (num_rows - 1) * num_cols:
        ax.set_xticklabels([])
        ax.set_xlabel("")
    else:
        ax.set_xlabel('| ILD |', fontsize=18)
    ax.tick_params(axis='both', labelsize=18)
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Hide unused axes
for i in range(idx + 1, len(axes)):
    axes[i].axis('off')

plt.tight_layout()

# --- Unified legend for all ABLs at bottom right ---
from matplotlib.lines import Line2D
handles = []
for abl in sorted(abl_color_map):
    handles.append(Line2D([0], [0], color=get_abl_color(abl), marker='o', label=f'ABL {abl}', linestyle='-', linewidth=2))
fig.legend(handles=handles, loc='lower right', fontsize=20, title='ABL', title_fontsize=15, frameon=True)

plt.show()

# %%
print(merged_valid['choice'].unique())