# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

# --- Data loading (same as your other scripts) ---
batch_dir = os.path.join(os.path.dirname(__file__), 'batch_csvs')
batch_files = [f for f in os.listdir(batch_dir) if f.endswith('_valid_and_aborts.csv')]
merged_data = pd.concat([
    pd.read_csv(os.path.join(batch_dir, fname)) for fname in batch_files
], ignore_index=True)
merged_valid = merged_data[merged_data['success'].isin([1, -1])].copy()
merged_valid['batch_name'] = merged_valid['batch_name'].fillna('LED7')

# --- Parameters ---
ABLS = [20, 40, 60]
COLORS = ['tab:blue', 'tab:orange', 'tab:green']

# --- Sigmoid function ---
def sigmoid(x, L, x0, k, b):
    return L / (1 + np.exp(-k*(x-x0))) + b

# --- Prepare figure ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for idx, (abl, color) in enumerate(zip(ABLS, COLORS)):
    ax = axes[idx]
    all_animals = merged_valid['animal'].unique()
    allowed_ilds = np.sort(np.array([1., 2., 4., 8., 16., -1., -2., -4., -8., -16.]))
    ilds = np.sort(np.intersect1d(merged_valid[merged_valid['ABL'] == abl]['ILD'].unique(), allowed_ilds))
    # Store all sigmoid fits and all psychometric data
    all_sigmoid_curves = []
    all_psycho_points = []
    for animal in all_animals:
        animal_df = merged_valid[(merged_valid['animal'] == animal) & (merged_valid['ABL'] == abl)]
        if animal_df.empty:
            continue
        # Compute psychometric data
        psycho = []
        for ild in ilds:
            sub = animal_df[animal_df['ILD'] == ild]
            if len(sub) > 0:
                psycho.append(np.mean(sub['choice'] == 1))
            else:
                psycho.append(np.nan)
        psycho = np.array(psycho)
        # Fit sigmoid if enough points
        mask = ~np.isnan(psycho)
        if np.sum(mask) > 3:
            try:
                popt, _ = curve_fit(sigmoid, ilds[mask], psycho[mask], p0=[1, 0, 1, 0], maxfev=5000)
                x_smooth = np.linspace(np.min(ilds), np.max(ilds), 200)
                y_fit = sigmoid(x_smooth, *popt)
                ax.plot(x_smooth, y_fit, color=color, alpha=0.3, linewidth=1)
                all_sigmoid_curves.append(sigmoid(ilds, *popt))
            except Exception as e:
                pass
        all_psycho_points.append(psycho)
    # Average sigmoid fits (average y-values at each ILD)
    if all_sigmoid_curves:
        mean_sigmoid = np.nanmean(np.vstack(all_sigmoid_curves), axis=0)
        ax.plot(ilds, mean_sigmoid, color='black', linewidth=3, label='Avg sigmoid fit')
    # Average data points and std
    all_psycho_points = np.array(all_psycho_points)
    mean_psycho = np.nanmean(all_psycho_points, axis=0)
    std_psycho = np.nanstd(all_psycho_points, axis=0)
    ax.errorbar(ilds, mean_psycho, yerr=std_psycho, fmt='o', color=color, capsize=4, label='Mean ± std')
    ax.set_title(f'ABL = {abl}', fontsize=18)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.7)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1)
    ax.set_xticks([-10, -5, 5, 10])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xlabel('ILD', fontsize=18)
    ax.set_ylabel('P(Right)', fontsize=18)
    # Remove top and right spines (margins)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Set global font size for the legend and tight layout
for ax in axes:
    legend = ax.get_legend()
    if legend:
        legend.prop.set_size(16)
for ax in axes:
    legend = ax.get_legend()
    if legend:
        legend.prop.set_size(16)
plt.tight_layout()
plt.subplots_adjust(bottom=0.15, left=0.07, right=0.97, top=0.88)
plt.savefig('aggregate_psychometric_by_abl.png', dpi=300, bbox_inches='tight')
plt.show()
# %%
all_psycho_points.shape