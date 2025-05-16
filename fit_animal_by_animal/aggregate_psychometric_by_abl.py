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

# --- Black curve mode flag ---
# black_plot_as = "mean_of_params"  # or "mean_of_sigmoids"
black_plot_as = "mean_of_sigmoids"

# --- Prepare figure ---
fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)

mean_params_dict = {}  # Store mean sigmoid parameters for each ABL
ilds_dict = {}         # Store ILDs for each ABL (for x axis in 4th plot)
mean_sigmoid_dict = {} # Store mean-of-sigmoids y values for each ABL (for black_plot_as)
x_smooth_dict = {}    # Store x_smooth for each ABL

for idx, (abl, color) in enumerate(zip(ABLS, COLORS)):
    ax = axes[idx]
    all_animals = merged_valid['animal'].unique()
    allowed_ilds = np.sort(np.array([1., 2., 4., 8., 16., -1., -2., -4., -8., -16.]))
    ilds = np.sort(np.intersect1d(merged_valid[merged_valid['ABL'] == abl]['ILD'].unique(), allowed_ilds))
    ilds_dict[abl] = ilds
    # Store all sigmoid fits and all psychometric data
    all_sigmoid_curves = []
    all_psycho_points = []
    all_sigmoid_params = []  # Store all sigmoid fit parameters
    for animal in all_animals:
        animal_df = merged_valid[(merged_valid['animal'] == animal) & (merged_valid['ABL'] == abl)]
        if animal_df.empty:
            continue
        # Use all ILDs present for this animal
        animal_ilds = np.sort(animal_df['ILD'].unique())
        # Compute psychometric data for all animal ILDs
        psycho = []
        for ild in animal_ilds:
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
                popt, _ = curve_fit(sigmoid, animal_ilds[mask], psycho[mask], p0=[1, 0, 1, 0], maxfev=5000)
                x_smooth = np.linspace(np.min(ilds), np.max(ilds), 200)
                y_fit = sigmoid(x_smooth, *popt)
                ax.plot(x_smooth, y_fit, color=color, alpha=0.3, linewidth=1)
                all_sigmoid_curves.append(sigmoid(x_smooth, *popt))  # Store smooth curve for mean_of_sigmoids
                all_sigmoid_params.append(popt)
                # Store x_smooth for this ABL (once)
                x_smooth_dict[abl] = x_smooth
            except Exception as e:
                pass
        # For group mean/error bars, keep psycho at allowed_ilds
        # (Recompute for allowed_ilds)
        psycho_allowed = []
        for ild in ilds:
            sub = animal_df[animal_df['ILD'] == ild]
            if len(sub) > 0:
                psycho_allowed.append(np.mean(sub['choice'] == 1))
            else:
                psycho_allowed.append(np.nan)
        all_psycho_points.append(np.array(psycho_allowed))
    # Black curve: mean of params or mean of sigmoids
    if black_plot_as == "mean_of_params":
        if all_sigmoid_params:
            mean_params = np.mean(np.vstack(all_sigmoid_params), axis=0)
            mean_params_dict[abl] = mean_params  # Store for 4th plot
            x_smooth = np.linspace(np.min(ilds), np.max(ilds), 200)
            y_mean_sigmoid = sigmoid(x_smooth, *mean_params)
            ax.plot(x_smooth, y_mean_sigmoid, color='black', linewidth=3, label='Avg sigmoid fit')
    elif black_plot_as == "mean_of_sigmoids":
        if all_sigmoid_curves:
            mean_sigmoid = np.nanmean(np.vstack(all_sigmoid_curves), axis=0)
            mean_sigmoid_dict[abl] = mean_sigmoid
            # ax.plot(ilds, mean_sigmoid, color='black', linewidth=3, label='Avg sigmoid fit')
            ax.plot(x_smooth, mean_sigmoid, color='black', linewidth=3, label='Avg sigmoid fit')

    # Average data points and std
    all_psycho_points = np.array(all_psycho_points)
    mean_psycho = np.nanmean(all_psycho_points, axis=0)
    std_psycho = np.nanstd(all_psycho_points, axis=0)
    ax.errorbar(ilds, mean_psycho, yerr=std_psycho, fmt='o', color=color, capsize=0, markersize=8.5, label='Mean ± std')
    ax.set_title(f'ABL = {abl}', fontsize=18)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.7)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1)
    ax.set_xticks([-15, -5, 5, 15])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xlabel('ILD', fontsize=18)
    if idx == 0:
        ax.set_ylabel('P(Right)', fontsize=18)
        # Make y-axis spine and ticks black
        ax.spines['left'].set_color('black')
        ax.yaxis.label.set_color('black')
        ax.tick_params(axis='y', colors='black')
    else:
        # Make y-axis spine and ticks light gray
        ax.spines['left'].set_color('#bbbbbb')
        ax.yaxis.label.set_color('#bbbbbb')
        ax.tick_params(axis='y', colors='#bbbbbb')
    # Remove top and right spines (margins)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# --- 4th plot: All ABLs together ---
ax4 = axes[3]
for abl, color in zip(ABLS, COLORS):
    ilds = ilds_dict[abl]
    mean_params = mean_params_dict.get(abl, None)
    # Compute mean and std for error bars (same as before)
    all_animals = merged_valid['animal'].unique()
    all_psycho_points = []
    for animal in all_animals:
        animal_df = merged_valid[(merged_valid['animal'] == animal) & (merged_valid['ABL'] == abl)]
        if animal_df.empty:
            continue
        psycho = []
        for ild in ilds:
            sub = animal_df[animal_df['ILD'] == ild]
            if len(sub) > 0:
                psycho.append(np.mean(sub['choice'] == 1))
            else:
                psycho.append(np.nan)
        psycho = np.array(psycho)
        all_psycho_points.append(psycho)
    all_psycho_points = np.array(all_psycho_points)
    mean_psycho = np.nanmean(all_psycho_points, axis=0)
    std_psycho = np.nanstd(all_psycho_points, axis=0)
    ax4.errorbar(ilds, mean_psycho, yerr=std_psycho, fmt='o', color=color, capsize=0, markersize=8.5, label=f'ABL={abl} mean')
    # Plot sigmoid using mean parameters or mean-of-sigmoids (as black curve)
    if black_plot_as == "mean_of_params":
        if mean_params is not None:
            x_smooth = np.linspace(np.min(ilds), np.max(ilds), 200)
            y_mean_sigmoid = sigmoid(x_smooth, *mean_params)
            ax4.plot(x_smooth, y_mean_sigmoid, color=color, linewidth=2, label=f'ABL={abl} curve')
    elif black_plot_as == "mean_of_sigmoids":
        mean_sigmoid = mean_sigmoid_dict.get(abl, None)
        x_smooth = x_smooth_dict.get(abl, None)
        if mean_sigmoid is not None and x_smooth is not None:
            ax4.plot(x_smooth, mean_sigmoid, color=color, linewidth=2, label=f'ABL={abl} curve')
ax4.set_title('All ABLs', fontsize=18)
ax4.axvline(0, color='gray', linestyle='--', alpha=0.7)
ax4.axhline(0.5, color='gray', linestyle='--', alpha=0.7)
ax4.set_ylim(0, 1)
ax4.set_xticks([-15, -5, 5, 15])
ax4.set_yticks([0, 0.5, 1])
ax4.tick_params(axis='both', which='major', labelsize=16)
ax4.set_xlabel('ILD', fontsize=18)
# ax4.set_ylabel('P(Right)', fontsize=18)
# Make y-axis spine and ticks light gray for 4th plot
ax4.spines['left'].set_color('#bbbbbb')
ax4.yaxis.label.set_color('#bbbbbb')
ax4.tick_params(axis='y', colors='#bbbbbb')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
# ax4.legend(fontsize=14)

# Set global font size for the legend and tight layout
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