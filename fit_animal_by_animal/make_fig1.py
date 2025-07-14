# %%

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
# --- Sigmoid function (must match the one in the aggregation script) ---
def sigmoid(x, upper, lower, x0, k):
    """Sigmoid function with explicit upper and lower asymptotes."""
    return lower + (upper - lower) / (1 + np.exp(-k*(x-x0)))

# --- Load data ---
with open('fig1_plot_data.pkl', 'rb') as f:
    plot_data = pickle.load(f)

# --- Extract data ---
ABLS = plot_data['ABLS']
COLORS = plot_data['COLORS']
black_plot_as = plot_data['black_plot_as']
ilds_dict = plot_data['ilds_dict']
mean_params_dict = plot_data['mean_params_dict']
mean_sigmoid_dict = plot_data['mean_sigmoid_dict']
x_smooth_dict = plot_data['x_smooth_dict']
unique_animal_identifiers = plot_data['unique_animal_identifiers']
merged_valid = plot_data['merged_valid']
all_sigmoid_curves_dict = plot_data['all_sigmoid_curves_dict']

# --- Prepare figure using GridSpec for complex layout ---
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.3)

# Create axes for the psychometric plots in the second row (index 1)
ax_psych_1 = fig.add_subplot(gs[1, 0])
ax_psych_2 = fig.add_subplot(gs[1, 1], sharey=ax_psych_1)
ax_psych_3 = fig.add_subplot(gs[1, 2], sharey=ax_psych_1)
ax_psych_4 = fig.add_subplot(gs[1, 3], sharey=ax_psych_1)

# Group axes for easy iteration in the existing plotting loop
axes = [ax_psych_1, ax_psych_2, ax_psych_3, ax_psych_4]

# Hide y-tick labels for shared axes to avoid clutter
for ax in [ax_psych_2, ax_psych_3, ax_psych_4]:
    plt.setp(ax.get_yticklabels(), visible=False)

# --- Recreate first 3 plots ---
for idx, (abl, color) in enumerate(zip(ABLS, COLORS)):
    ax = axes[idx]
    ilds = ilds_dict[abl]
    x_smooth = x_smooth_dict[abl]

    # Plot individual animal fits from the loaded data
    if abl in all_sigmoid_curves_dict:
        for y_fit in all_sigmoid_curves_dict[abl]:
            ax.plot(x_smooth, y_fit, color=color, alpha=0.3, linewidth=1)

    # Plot the main average sigmoid curve (black one)
    if black_plot_as == "mean_of_params" and abl in mean_params_dict:
        mean_params = mean_params_dict[abl]
        y_mean_sigmoid = sigmoid(x_smooth, *mean_params)
        ax.plot(x_smooth, y_mean_sigmoid, color='black', linewidth=3, label='Avg sigmoid fit')
    elif black_plot_as == "mean_of_sigmoids" and abl in mean_sigmoid_dict:
        mean_sigmoid = mean_sigmoid_dict[abl]
        ax.plot(x_smooth, mean_sigmoid, color='black', linewidth=3, label='Avg sigmoid fit')

    # Re-calculate and plot average data points and std
    all_psycho_points = []
    for batch, animal in unique_animal_identifiers:
        animal_df = merged_valid[(merged_valid['batch_name'] == batch) & (merged_valid['animal'] == animal) & (merged_valid['ABL'] == abl)]
        psycho_allowed = []
        for ild in ilds:
            sub = animal_df[animal_df['ILD'] == ild]
            psycho_allowed.append(np.mean(sub['choice'] == 1) if len(sub) > 0 else np.nan)
        all_psycho_points.append(np.array(psycho_allowed))
    
    mean_psycho = np.nanmean(all_psycho_points, axis=0)
    std_psycho = np.nanstd(all_psycho_points, axis=0)
    ax.errorbar(ilds, mean_psycho, yerr=std_psycho, fmt='o', color=color, capsize=0, markersize=8.5, label='Mean ± std')

    # --- Formatting for first 3 plots ---
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

# --- 4th plot: All ABLs together ---
ax4 = axes[3]
for abl, color in zip(ABLS, COLORS):
    ilds = ilds_dict[abl]
    
    # Re-calculate mean and std for error bars
    all_psycho_points = []
    for batch, animal in unique_animal_identifiers:
        animal_df = merged_valid[(merged_valid['batch_name'] == batch) & (merged_valid['animal'] == animal) & (merged_valid['ABL'] == abl)]
        psycho = [np.mean(animal_df[animal_df['ILD'] == ild]['choice'] == 1) if len(animal_df[animal_df['ILD'] == ild]) > 0 else np.nan for ild in ilds]
        all_psycho_points.append(psycho)
    
    mean_psycho = np.nanmean(all_psycho_points, axis=0)
    std_psycho = np.nanstd(all_psycho_points, axis=0)
    ax4.errorbar(ilds, mean_psycho, yerr=std_psycho, fmt='o', color=color, capsize=0, markersize=8.5, label=f'ABL={abl} mean')

    # Plot sigmoid using mean parameters or mean-of-sigmoids
    if black_plot_as == "mean_of_params" and abl in mean_params_dict:
        mean_params = mean_params_dict[abl]
        x_smooth = x_smooth_dict[abl]
        y_mean_sigmoid = sigmoid(x_smooth, *mean_params)
        ax4.plot(x_smooth, y_mean_sigmoid, color=color, linewidth=2, label=f'ABL={abl} curve')
    elif black_plot_as == "mean_of_sigmoids" and abl in mean_sigmoid_dict:
        mean_sigmoid = mean_sigmoid_dict[abl]
        x_smooth = x_smooth_dict[abl]
        if mean_sigmoid is not None and x_smooth is not None:
            ax4.plot(x_smooth, mean_sigmoid, color=color, linewidth=2, label=f'ABL={abl} curve')

# --- Formatting for 4th plot ---
ax4.set_title('All ABLs', fontsize=18)
ax4.axvline(0, color='gray', linestyle='--', alpha=0.7)
ax4.axhline(0.5, color='gray', linestyle='--', alpha=0.7)
ax4.set_ylim(0, 1)
ax4.set_xticks([-15, -5, 5, 15])
ax4.set_yticks([0, 0.5, 1])
ax4.tick_params(axis='both', which='major', labelsize=16)
ax4.set_xlabel('ILD', fontsize=18)
ax4.spines['left'].set_color('#bbbbbb')
ax4.yaxis.label.set_color('#bbbbbb')
ax4.tick_params(axis='y', colors='#bbbbbb')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

# --- Final layout adjustments ---
for ax in axes:
    legend = ax.get_legend()
    if legend:
        legend.prop.set_size(16)
plt.tight_layout()
plt.subplots_adjust(bottom=0.15, left=0.07, right=0.97, top=0.88)
plt.savefig('fig1_from_pickle.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nFigure saved as fig1_from_pickle.png")
