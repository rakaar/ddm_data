# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D

# --- Load merged_valid as in mean_chrono_plot.py ---
batch_dir = os.path.join(os.path.dirname(__file__), 'batch_csvs')
batch_files = [f for f in os.listdir(batch_dir) if f.endswith('_valid_and_aborts.csv')]
merged_data = pd.concat([
    pd.read_csv(os.path.join(batch_dir, fname)) for fname in batch_files
], ignore_index=True)

# Keep only valid trials (success 1 or -1)
merged_valid = merged_data[merged_data['success'].isin([1, -1])].copy()
merged_valid['batch_name'] = merged_valid['batch_name'].fillna('LED7')

# Remove RTwrtStim <> 0.1 and < 1
RT_max = 1
merged_valid = merged_valid[(merged_valid['RTwrtStim'] >= 0.1) & (merged_valid['RTwrtStim'] <= RT_max)]

# Add abs_ILD column
merged_valid['abs_ILD'] = merged_valid['ILD'].abs()

# Remove ILD 6 and 10
merged_valid = merged_valid[~merged_valid['abs_ILD'].isin([6, 10])]

# --- Parameters ---
merged_valid['abs_ILD'] = merged_valid['abs_ILD'].astype(float)
abs_ILDs = [1., 2., 4., 8., 16.]
ABLs = [20, 40, 60]
percentiles = np.arange(5, 100, 10)

# --- Prepare data structure ---
animals = np.sort(merged_valid['animal'].unique())

# For each animal, each abs_ILD, store Q_20, Q_40, Q_60
Q_dict = {animal: {abs_ILD: {} for abs_ILD in abs_ILDs} for animal in animals}

for animal in animals:
    df_animal = merged_valid[merged_valid['animal'] == animal]
    for abs_ILD in abs_ILDs:
        df_ild = df_animal[df_animal['abs_ILD'] == abs_ILD]
        for abl in ABLs:
            vals = df_ild[df_ild['ABL'] == abl]['RTwrtStim']
            if len(vals) > 0:
                Q_dict[animal][abs_ILD][abl] = np.percentile(vals, percentiles)
            else:
                Q_dict[animal][abs_ILD][abl] = np.full(percentiles.shape, np.nan)

# --- Compute Q_ABL - Q_60 for each animal, each abs_ILD ---
diff_dict = {20: {}, 40: {}}
for abl in [20, 40]:
    for abs_ILD in abs_ILDs:
        diffs = []
        for animal in animals:
            Q_abl = Q_dict[animal][abs_ILD][abl]
            Q_60 = Q_dict[animal][abs_ILD][60]
            if np.isnan(Q_abl).all() or np.isnan(Q_60).all():
                continue
            diff = Q_abl - Q_60
            diffs.append(diff)
        if diffs:
            diff_dict[abl][abs_ILD] = np.vstack(diffs)
        else:
            diff_dict[abl][abs_ILD] = np.full((0, len(percentiles)), np.nan)

# --- Plotting ---
fig, axes = plt.subplots(1, 5, figsize=(25, 6), sharey=True)
colors = {20: '#8ecae6', 40: '#ffb703'}  # light blue, light orange
mean_colors = {20: '#219ebc', 40: '#fb8500'}  # dark blue, dark orange
labels = {20: 'ABL 20 - 60', 40: 'ABL 40 - 60'}

x_fit_max = 0.3  # Only fit and plot up to 0.3 s

for i, abs_ILD in enumerate(abs_ILDs):
    ax = axes[i]
    fit_texts = []
    for abl in [20, 40]:
        diffs = diff_dict[abl][abs_ILD]
        if diffs.shape[0] == 0:
            continue
        # Plot individual animals (light color)
        for idx, row in enumerate(diffs):
            Q_60 = Q_dict[animals[idx]][abs_ILD][60]
            mask = ~np.isnan(Q_60) & (Q_60 <= x_fit_max)
            ax.plot(Q_60[mask], row[mask], color=colors[abl], alpha=0.5, linewidth=1)
        # Plot mean curve (bold)
        Q_60_mat = np.array([Q_dict[animal][abs_ILD][60] for animal in animals])
        mean_Q_60 = np.nanmean(Q_60_mat, axis=0)
        mean_diff = np.nanmean(diffs, axis=0)
        std_diff = np.nanstd(diffs, axis=0)
        epsilon = 1e-6
        weights = 1.0 / (std_diff + epsilon)
        mask = ~np.isnan(mean_Q_60) & (mean_Q_60 <= x_fit_max) & (weights > 0)
        ax.plot(mean_Q_60[mask], mean_diff[mask], color=mean_colors[abl], label=labels[abl], linewidth=2)
        # Weighted linear regression only up to x_fit_max
        if np.sum(mask) > 1:
            fit = np.polyfit(mean_Q_60[mask], mean_diff[mask], 1, w=weights[mask])
            slope, intercept = fit
            y_fit = slope * mean_Q_60[mask] + intercept
            ax.plot(mean_Q_60[mask], y_fit, linestyle=':', color=mean_colors[abl], linewidth=2)
            # Weighted R^2 computation
            y_mean = np.average(mean_diff[mask], weights=weights[mask])
            ss_tot = np.sum(weights[mask] * (mean_diff[mask] - y_mean) ** 2)
            ss_res = np.sum(weights[mask] * (mean_diff[mask] - y_fit) ** 2)
            r2_weighted = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            fit_texts.append(f"{labels[abl]}: slope={slope:.3f}, int={intercept:.3f}, wR²={r2_weighted:.3f}")
    # Annotate fit results
    if fit_texts:
        ax.text(0.98, 0.02, '\n'.join(fit_texts), fontsize=9, color='k', ha='right', va='bottom', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.set_xlabel('Q_60 (s)')
    if i == 0:
        ax.set_ylabel('Q_ABL - Q_60 (s)')
    ax.set_title(f'abs(ILD) = {abs_ILD}')
    ax.set_xlim(0.1, x_fit_max)
    ax.set_xticks([0.1, 0.2, 0.3])
    ax.set_ylim(-0.05, 0.4)

# plt.tight_layout(rect=[0, 0.05, 1, 1])
# plt.show()
