# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

# --- Load merged_valid as in mean_chrono_plot.py ---
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

mean_or_median = "mean"  # or "median"

batch_names = merged_valid['batch_name'].unique()

# Count total number of animals to determine subplot layout
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

# Subplot layout
num_cols = 5
num_rows = (total_animals + num_cols - 1) // num_cols
subplot_size = 5
fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * subplot_size, num_rows * subplot_size))
axes = axes.flatten() if num_rows > 1 else [axes] if num_cols == 1 else axes

# Color map for ABL
fixed_abl_colors = {
    20: '#1f77b4',   # blue
    40: '#ff7f0e',   # orange
    60: '#2ca02c',   # green
}
all_abls = sorted(set(merged_valid['ABL'].unique()))
tab10 = plt.get_cmap('tab10').colors
palette_indices = [i for i in range(len(tab10)) if i not in [0, 1, 2]]
palette = [tab10[i] for i in palette_indices]
other_abls = [abl for abl in all_abls if abl not in fixed_abl_colors]
other_abl_colors = {abl: palette[i % len(palette)] for i, abl in enumerate(other_abls)}
abl_color_map = {**fixed_abl_colors, **other_abl_colors}
def get_abl_color(abl):
    return abl_color_map.get(abl, '#888888')

# Chronometric function to fit, with c fixed
def chrono_func_fixed_c(ILD, a, b, c):
    ILD = np.array(ILD)
    safe_ILD = np.where(ILD == 0, 1e-6, ILD)
    return a * np.tanh(b * safe_ILD) / safe_ILD + c

def chrono_func_ab(ILD, a, b, c):
    # c is fixed, only a and b are fit
    ILD = np.array(ILD)
    safe_ILD = np.where(ILD == 0, 1e-6, ILD)
    return a * np.tanh(b * safe_ILD) / safe_ILD + c

xticks = [0, 4, 8, 12, 16]
yticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

for idx, animal_info in enumerate(animal_data):
    ax = axes[idx]
    batch_name = animal_info['batch_name']
    animal = animal_info['animal']
    df = animal_info['df']
    legend_entries = []
    # --- Estimate c using all ABLs for this animal ---
    x_c = []
    y_c = []
    for abl in sorted(df['ABL'].unique()):
        abl_df = df[df['ABL'] == abl]
        xvals = sorted(abl_df['abs_ILD'].unique())
        for abs_ild in xvals:
            subset = abl_df[abl_df['abs_ILD'] == abs_ild]
            if mean_or_median == "mean":
                val = subset['RTwrtStim'].mean()
            else:
                val = subset['RTwrtStim'].median()
            x_c.append(abs_ild)
            y_c.append(val)
    # Fit c using all data for this animal (fit a, b, c, but only use c)
    try:
        popt_c, _ = curve_fit(chrono_func_fixed_c, x_c, y_c, p0=[0.2, 0.2, 0.2], maxfev=10000)
        c_fixed = popt_c[2]
    except Exception as e:
        c_fixed = 0.2  # fallback
    # --- Now fit a, b for each ABL with c fixed ---
    for abl in sorted(df['ABL'].unique()):
        abl_df = df[df['ABL'] == abl]
        xvals = sorted(abl_df['abs_ILD'].unique())
        yvals = []
        stds = []
        for abs_ild in xvals:
            subset = abl_df[abl_df['abs_ILD'] == abs_ild]
            if mean_or_median == "mean":
                val = subset['RTwrtStim'].mean()
                # std = subset['RTwrtStim'].std()
                std = np.nan
                stds.append(std)
            else:
                val = subset['RTwrtStim'].median()
            yvals.append(val)
        color = get_abl_color(abl)
        # Plot data points
        ax.errorbar(xvals, yvals, yerr=stds if mean_or_median == "mean" else None, marker='o', label=f'ABL {abl}', color=color, capsize=0, linestyle='-', linewidth=2)
        # Fit a, b with c fixed
        def fit_func(ILD, a, b):
            return chrono_func_ab(ILD, a, b, c_fixed)
        try:
            popt, _ = curve_fit(fit_func, xvals, yvals, p0=[0.2, 0.2], maxfev=10000)
            a, b = popt
            xfit = np.linspace(min(xvals), max(xvals), 200)
            yfit = chrono_func_ab(xfit, a, b, c_fixed)
            ax.plot(xfit, yfit, linestyle=':', color=color, linewidth=2)
            legend_entries.append(f'ABL {abl}: a={a:.2f}, b={b:.2f}, c={c_fixed:.2f}')
        except Exception as e:
            legend_entries.append(f'ABL {abl}: fit failed, c={c_fixed:.2f}')
    ax.set_title(f'{batch_name} | {animal}', fontsize=14)
    ax.set_xlim([0, 16])
    ax.set_xticks(xticks)
    ax.set_ylim([0, 0.5])
    ax.set_yticks(yticks)
    if idx % num_cols != 0:
        ax.set_yticklabels([])
        ax.set_ylabel("")
        ax.spines['left'].set_color('gray')
    else:
        ax.set_ylabel(f'{mean_or_median} RT', fontsize=12)
        ax.spines['left'].set_color('black')
    if idx < (num_rows - 1) * num_cols:
        ax.set_xticklabels([])
        ax.set_xlabel("")
    else:
        ax.set_xlabel('| ILD |', fontsize=12)
    ax.tick_params(axis='both', labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Add legend to each subplot
    ax.legend(legend_entries, loc='upper right', fontsize=8, frameon=True)

# Hide unused axes
for i in range(idx + 1, len(axes)):
    axes[i].axis('off')

plt.tight_layout()
plt.show()
