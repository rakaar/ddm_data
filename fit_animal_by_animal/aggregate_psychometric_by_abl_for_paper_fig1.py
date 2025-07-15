# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from collections import defaultdict
import pickle

# --- Data loading (same as your other scripts) ---
DESIRED_BATCHES = ['SD', 'LED34', 'LED6', 'LED8', 'LED7', 'LED34_even']
batch_dir = os.path.join(os.path.dirname(__file__), 'batch_csvs')
batch_files = [f'batch_{batch_name}_valid_and_aborts.csv' for batch_name in DESIRED_BATCHES]

merged_data = pd.concat([
    pd.read_csv(os.path.join(batch_dir, fname)) for fname in batch_files if os.path.exists(os.path.join(batch_dir, fname))
], ignore_index=True)

merged_valid = merged_data[merged_data['success'].isin([1, -1])].copy()

# --- Print animal table ---
batch_animal_pairs = sorted(list(map(tuple, merged_valid[['batch_name', 'animal']].drop_duplicates().values)))

print(f"Found {len(batch_animal_pairs)} batch-animal pairs from {len(set(p[0] for p in batch_animal_pairs))} batches:")

if batch_animal_pairs:
    batch_to_animals = defaultdict(list)
    for batch, animal in batch_animal_pairs:
        # Ensure animal is a string and we don't add duplicates
        animal_str = str(animal)
        if animal_str not in batch_to_animals[batch]:
            batch_to_animals[batch].append(animal_str)

    # Determine column widths for formatting
    max_batch_len = max(len(b) for b in batch_to_animals.keys()) if batch_to_animals else 0
    animal_strings = {b: ', '.join(sorted(a)) for b, a in batch_to_animals.items()}
    max_animals_len = max(len(s) for s in animal_strings.values()) if animal_strings else 0

    # Header
    print(f"{'Batch':<{max_batch_len}}  {'Animals'}")
    print(f"{'=' * max_batch_len}  {'=' * max_animals_len}")

    # Rows
    for batch in sorted(animal_strings.keys()):
        animals_str = animal_strings[batch]
        print(f"{batch:<{max_batch_len}}  {animals_str}")

# %%
# --- Parameters ---
ABLS = [20, 40, 60]
COLORS = ['tab:blue', 'tab:orange', 'tab:green']

# --- Sigmoid function ---
def sigmoid(x, upper, lower, x0, k):
    """Sigmoid function with explicit upper and lower asymptotes."""
    return lower + (upper - lower) / (1 + np.exp(-k*(x-x0)))

# --- Black curve mode flag ---
# black_plot_as = "mean_of_params"  # or "mean_of_sigmoids"
black_plot_as = "mean_of_sigmoids"

# --- Prepare figure ---
fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)

mean_params_dict = {}  # Store mean sigmoid parameters for each ABL
ilds_dict = {}         # Store ILDs for each ABL (for x axis in 4th plot)
mean_sigmoid_dict = {} # Store mean-of-sigmoids y values for each ABL (for black_plot_as)
x_smooth_dict = {}    # Store x_smooth for each ABL
all_sigmoid_curves_dict = {} # Store all individual sigmoid curves for each ABL

unique_animal_identifiers = sorted(list(map(tuple, merged_valid[['batch_name', 'animal']].drop_duplicates().values)))
print(f'\nFound {len(unique_animal_identifiers)} unique animal-batch pairs to analyze.\n')

for idx, (abl, color) in enumerate(zip(ABLS, COLORS)):
    ax = axes[idx]
    allowed_ilds = np.sort(np.array([1., 2., 4., 8., 16., -1., -2., -4., -8., -16.]))
    ilds = np.sort(np.intersect1d(merged_valid[merged_valid['ABL'] == abl]['ILD'].unique(), allowed_ilds))
    ilds_dict[abl] = ilds
    # Store all sigmoid fits and all psychometric data
    all_sigmoid_curves = []
    all_psycho_points = []
    all_sigmoid_params = []  # Store all sigmoid fit parameters
    for batch, animal in unique_animal_identifiers:
        animal_df = merged_valid[(merged_valid['batch_name'] == batch) & (merged_valid['animal'] == animal) & (merged_valid['ABL'] == abl)]
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
                # Bounds for [upper, lower, x0, k]
                bounds = ([0, 0, -np.inf, 0], [1, 1, np.inf, np.inf])
                min_psycho = np.min(psycho[mask])
                max_psycho = np.max(psycho[mask])
                # Initial guess for [upper, lower, x0, k]
                p0 = [max_psycho, min_psycho, np.median(animal_ilds[mask]), 0.1]
                popt, _ = curve_fit(sigmoid, animal_ilds[mask], psycho[mask], p0=p0, bounds=bounds, maxfev=10000)
                
                # Ensure upper > lower, otherwise the fit is invalid
                if popt[0] < popt[1]:
                    continue

                x_smooth = np.linspace(np.min(ilds), np.max(ilds), 200)
                y_fit = sigmoid(x_smooth, *popt)
                ax.plot(x_smooth, y_fit, color=color, alpha=0.3, linewidth=1)
                all_sigmoid_curves.append(y_fit)
                all_sigmoid_params.append(popt)
                x_smooth_dict[abl] = x_smooth
            except RuntimeError:
                # This can happen if the fit is poor or doesn't converge
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
    all_sigmoid_curves_dict[abl] = all_sigmoid_curves

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
    all_psycho_points = []
    for batch, animal in unique_animal_identifiers:
        animal_df = merged_valid[(merged_valid['batch_name'] == batch) & (merged_valid['animal'] == animal) & (merged_valid['ABL'] == abl)]
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
    print(f'ABL {abl}: {all_psycho_points.shape}')
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
plt.savefig('aggregate_psychometric_by_abl_for_paper_fig1.png', dpi=300, bbox_inches='tight')
# --- Save data for external plotting ---
plot_data = {
    'ABLS': ABLS,
    'COLORS': COLORS,
    'black_plot_as': black_plot_as,
    'ilds_dict': ilds_dict,
    'mean_params_dict': mean_params_dict,
    'mean_sigmoid_dict': mean_sigmoid_dict,
    'x_smooth_dict': x_smooth_dict,
    'all_sigmoid_curves_dict': all_sigmoid_curves_dict,
    'unique_animal_identifiers': unique_animal_identifiers,
    'merged_valid': merged_valid, # Pass the dataframe for the 4th plot recreation
}

with open('fig1_plot_data.pkl', 'wb') as f:
    pickle.dump(plot_data, f)

print("\nPlotting data saved to fig1_plot_data.pkl")

plt.show()

# %%
# --- New Section: Plot with Standard Error of the Mean (SEM) ---
print("\n" + "="*50)
print("Generating plot with Standard Error of the Mean (SEM)")
print("="*50 + "\n")

from scipy.stats import sem

# --- Prepare figure for SEM plot ---
fig_sem, axes_sem = plt.subplots(1, 4, figsize=(14, 4), sharey=True)

for idx, (abl, color) in enumerate(zip(ABLS, COLORS)):
    ax = axes_sem[idx]
    ilds = ilds_dict[abl]
    
    # We can reuse the previously calculated sigmoid curves and psycho points
    all_sigmoid_curves = all_sigmoid_curves_dict[abl]
    
    # Re-calculate all_psycho_points for the current ABL to be safe
    all_psycho_points_sem = []
    for batch, animal in unique_animal_identifiers:
        animal_df = merged_valid[(merged_valid['batch_name'] == batch) & (merged_valid['animal'] == animal) & (merged_valid['ABL'] == abl)]
        if animal_df.empty:
            continue
        psycho_allowed = []
        for ild in ilds:
            sub = animal_df[animal_df['ILD'] == ild]
            if len(sub) > 0:
                psycho_allowed.append(np.mean(sub['choice'] == 1))
            else:
                psycho_allowed.append(np.nan)
        all_psycho_points_sem.append(np.array(psycho_allowed))

    all_psycho_points_sem = np.array(all_psycho_points_sem)
    
    # Plot individual sigmoid fits (thin lines)
    x_smooth = x_smooth_dict.get(abl)
    if x_smooth is not None and all_sigmoid_curves:
        for y_fit in all_sigmoid_curves:
            ax.plot(x_smooth, y_fit, color=color, alpha=0.3, linewidth=1)

    # Black curve: mean of sigmoids
    if black_plot_as == "mean_of_sigmoids":
        if all_sigmoid_curves:
            mean_sigmoid = np.nanmean(np.vstack(all_sigmoid_curves), axis=0)
            ax.plot(x_smooth, mean_sigmoid, color='black', linewidth=3, label='Avg sigmoid fit')

    # Average data points and SEM
    mean_psycho = np.nanmean(all_psycho_points_sem, axis=0)
    sem_psycho = sem(all_psycho_points_sem, axis=0, nan_policy='omit') # Calculate SEM
    ax.errorbar(ilds, mean_psycho, yerr=sem_psycho, fmt='o', color=color, capsize=0, markersize=8.5, label='Mean ± SEM')
    
    # --- Formatting (same as before) ---
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
        ax.spines['left'].set_color('black')
        ax.yaxis.label.set_color('black')
        ax.tick_params(axis='y', colors='black')
    else:
        ax.spines['left'].set_color('#bbbbbb')
        ax.yaxis.label.set_color('#bbbbbb')
        ax.tick_params(axis='y', colors='#bbbbbb')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# --- 4th plot: All ABLs together (with SEM) ---
ax4_sem = axes_sem[3]
for abl, color in zip(ABLS, COLORS):
    ilds = ilds_dict[abl]
    
    # Re-calculate psycho points for this ABL
    all_psycho_points_sem = []
    for batch, animal in unique_animal_identifiers:
        animal_df = merged_valid[(merged_valid['batch_name'] == batch) & (merged_valid['animal'] == animal) & (merged_valid['ABL'] == abl)]
        if animal_df.empty:
            continue
        psycho = []
        for ild in ilds:
            sub = animal_df[animal_df['ILD'] == ild]
            if len(sub) > 0:
                psycho.append(np.mean(sub['choice'] == 1))
            else:
                psycho.append(np.nan)
        all_psycho_points_sem.append(np.array(psycho))
    
    all_psycho_points_sem = np.array(all_psycho_points_sem)
    mean_psycho = np.nanmean(all_psycho_points_sem, axis=0)
    sem_psycho = sem(all_psycho_points_sem, axis=0, nan_policy='omit') # Calculate SEM
    
    ax4_sem.errorbar(ilds, mean_psycho, yerr=sem_psycho, fmt='o', color=color, capsize=0, markersize=8.5, label=f'ABL={abl} mean')
    
    # Plot sigmoid curve (reusing from original plot)
    if black_plot_as == "mean_of_sigmoids":
        mean_sigmoid = mean_sigmoid_dict.get(abl, None)
        x_smooth = x_smooth_dict.get(abl, None)
        if mean_sigmoid is not None and x_smooth is not None:
            ax4_sem.plot(x_smooth, mean_sigmoid, color=color, linewidth=2, label=f'ABL={abl} curve')

# --- Formatting for 4th plot (SEM) ---
ax4_sem.set_title('All ABLs (SEM)', fontsize=18)
ax4_sem.axvline(0, color='gray', linestyle='--', alpha=0.7)
ax4_sem.axhline(0.5, color='gray', linestyle='--', alpha=0.7)
ax4_sem.set_ylim(0, 1)
ax4_sem.set_xticks([-15, -5, 5, 15])
ax4_sem.set_yticks([0, 0.5, 1])
ax4_sem.tick_params(axis='both', which='major', labelsize=16)
ax4_sem.set_xlabel('ILD', fontsize=18)
ax4_sem.spines['left'].set_color('#bbbbbb')
ax4_sem.yaxis.label.set_color('#bbbbbb')
ax4_sem.tick_params(axis='y', colors='#bbbbbb')
ax4_sem.spines['top'].set_visible(False)
ax4_sem.spines['right'].set_visible(False)

# Set global font size for the legend and tight layout
for ax in axes_sem:
    legend = ax.get_legend()
    if legend:
        legend.prop.set_size(16)
plt.tight_layout()
plt.subplots_adjust(bottom=0.15, left=0.07, right=0.97, top=0.88)
plt.savefig('aggregate_psychometric_by_abl_for_paper_fig1_sem.png', dpi=300, bbox_inches='tight')
print("\nPlot with SEM saved to aggregate_psychometric_by_abl_for_paper_fig1_sem.png")

plt.show()
# %%
sem_psycho