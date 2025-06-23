# %%
########## Make a psychometric for all animals #######
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
def plot_psycho(df_1):
    prob_choice_dict = {}
    ILD_arr = np.sort(df_1['ILD'].unique())
    all_ABL = np.sort(df_1['ABL'].unique())

    for abl in all_ABL:
        filtered_df = df_1[df_1['ABL'] == abl]
        prob_choice_dict[abl] = [
            sum(filtered_df[filtered_df['ILD'] == ild]['choice'] == 1) / len(filtered_df[filtered_df['ILD'] == ild])
            if len(filtered_df[filtered_df['ILD'] == ild]) > 0 else np.nan
            for ild in ILD_arr
        ]
    
    return prob_choice_dict

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

# Plot each animal's data in its own subplot
for i, animal_info in enumerate(animal_data):
    batch_name = animal_info['batch_name']
    animal = animal_info['animal']
    animal_df = animal_info['df']
    
    # Get current axis
    ax = axes[i]
    
    # Plot psychometric function
    # Confirm: ILD and psychometric are computed per animal
    prob_choice_dict = plot_psycho(animal_df)
    all_ILD = np.sort(animal_df['ILD'].unique())  # <- ILDs for this animal only
    
    from scipy.optimize import curve_fit
    # Sigmoid function definition
    def sigmoid(x, L ,x0, k, b):
        return L / (1 + np.exp(-k*(x-x0))) + b

    for abl in prob_choice_dict.keys():
        color = abl_color_map.get(abl, None)
        ydata = np.array(prob_choice_dict[abl])
        xdata = np.array(all_ILD)
        # Remove NaNs for fitting
        mask = ~np.isnan(ydata)
        x_fit = xdata[mask]
        y_fit = ydata[mask]
        # Scatter plot for data points
        ax.scatter(x_fit, y_fit, label=None, alpha=0.8, color=color)
        # Fit sigmoid if enough points
        if len(x_fit) > 3:
            try:
                popt, _ = curve_fit(sigmoid, x_fit, y_fit, p0=[1, 0, 1, 0], maxfev=5000)
                x_smooth = np.linspace(np.min(x_fit), np.max(x_fit), 200)
                y_smooth = sigmoid(x_smooth, *popt)
                ax.plot(x_smooth, y_smooth, label=None, color=color)
            except Exception as e:
                print(f'Could not fit sigmoid for ABL {abl}:', e)
        else:
            print(f'Not enough points to fit sigmoid for ABL {abl}')

    
    # Only keep vertical at 0 and horizontal at 0.5
    ax.axvline(0, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.set_ylim(0, 1)
    # Remove grid
    ax.grid(False)
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set x/y ticks for all axes
    ax.set_xlim(global_min_ILD, global_max_ILD)
    ax.set_xticks([-15, -5, 5, 15])
    ax.set_yticks([0, 0.5, 1])
    # Increase font size for ticks (journal style)
    ax.tick_params(axis='both', which='major', labelsize=22, length=8, width=2.5)
    # Increase font size for axis labels and title
    row_idx = i // num_cols
    col_idx = i % num_cols
    # Only show x-ticks on bottom row
    if row_idx != num_rows - 1:
        ax.set_xticklabels([])
        ax.set_xlabel("")
    else:
        ax.set_xlabel('ILD', fontsize=24)
    # Only show y-ticks on leftmost column
    if col_idx != 0:
        ax.set_yticklabels([])
        ax.set_ylabel("")
    else:
        ax.set_ylabel('P(right)', fontsize=24)
    ax.set_title(f'Batch: {batch_name}, Animal: {animal}', fontsize=20)
    # Suppress per-plot legend
    # ax.legend(title='ABL', fontsize='small')

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
plt.savefig(f'all_animals_psychometric_all_batches_with_LED8.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Aggregate Psychometric
