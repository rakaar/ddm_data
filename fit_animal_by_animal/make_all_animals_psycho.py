# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# %%
batch_types = 'bad' # good or bad
if batch_types == 'good':
    merged_data = pd.read_csv('batch_csvs/merged_batches.csv')
    merged_valid = merged_data[merged_data['success'].isin([1, -1])].copy()
else:
    # Read and merge the three batch CSVs for LED7, LED6, and Comparable
    batch_dir = os.path.join(os.path.dirname(__file__), 'batch_csvs')
    batch_files = [
        'batch_LED7_valid_and_aborts.csv',
        'batch_LED6_valid_and_aborts.csv',
        'batch_Comparable_valid_and_aborts.csv',
    ]
    merged_data = pd.concat([
        pd.read_csv(os.path.join(batch_dir, fname)) for fname in batch_files
    ], ignore_index=True)
    merged_valid = merged_data[merged_data['success'].isin([1, -1])].copy()

# %%
print(merged_valid['batch_name'].unique())
# batch_name is nan, fill it with LED7
merged_valid['batch_name'] = merged_valid['batch_name'].fillna('LED7')
print(merged_valid['batch_name'].unique())


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

# Calculate number of rows needed (with 4 columns)
num_cols = 4
num_rows = (total_animals + num_cols - 1) // num_cols  # Ceiling division

# Create figure with subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5 * num_rows))
# Flatten axes array for easier indexing
axes = axes.flatten() if num_rows > 1 else [axes] if num_cols == 1 else axes

# Plot each animal's data in its own subplot
for i, animal_info in enumerate(animal_data):
    batch_name = animal_info['batch_name']
    animal = animal_info['animal']
    animal_df = animal_info['df']
    
    # Get current axis
    ax = axes[i]
    
    # Plot psychometric function
    prob_choice_dict = plot_psycho(animal_df)
    all_ILD = np.sort(animal_df['ILD'].unique())
    
    for abl in prob_choice_dict.keys():
        ax.plot(all_ILD, prob_choice_dict[abl], '-o', label=f'ABL {abl}')
    
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(f'Batch: {batch_name}, Animal: {animal}')
    ax.set_xlabel('ILD')
    ax.set_ylabel('P(right)')
    ax.legend(title='ABL', fontsize='small')

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

# Adjust layout
plt.tight_layout()

# Save figure
plt.savefig(f'all_animals_psychometric_{batch_types}.png', dpi=300, bbox_inches='tight')
plt.show()
        
# %%
batches = ['LED6', 'LED7']
ref_stim_1 = np.random.exponential(0.2, 10000) + 0.2
ref_stim_2 = np.random.exponential(0.4, 10000) + 0.2

# for these batches in merged_data plot intended_fix histogram for each animal from merged_valid
for batch in batches:
    batch_df = merged_valid[merged_valid['batch_name'] == batch]
    animal_data = np.sort(batch_df['animal'].unique())
    n_animals = len(animal_data)
    n_cols = min(4, n_animals)
    n_rows = int(np.ceil(n_animals / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=True, sharey=True)
    if n_animals == 1:
        axes = np.array([[axes]])
    axes = axes.flatten()
    for idx, animal in enumerate(animal_data):
        ax = axes[idx]
        animal_df = batch_df[batch_df['animal'] == animal]
        session_types = np.sort(animal_df['session_type'].unique())
        for session_type in session_types:
            session_df = animal_df[animal_df['session_type'] == session_type]
            ax.hist(session_df['intended_fix'], bins=np.arange(0,2,0.02), label=f'Session {session_type}', histtype='step', density=True)
        # Overlay reference histograms
        ax.hist(ref_stim_1, bins=np.arange(0,2,0.02), density=True, histtype='step', color='k', linestyle=':', linewidth=4, label='tau=0.2', alpha=0.4)
        ax.hist(ref_stim_2, bins=np.arange(0,2,0.02), density=True, histtype='step', color='k', linestyle='--', linewidth=4, label='tau=0.4', alpha=0.4)
        ax.set_title(f'Animal: {animal}')
        ax.set_xlabel('Intended Fix')
        ax.set_ylabel('Density')
        ax.legend(title='Session Type / Ref')
    # Hide unused subplots
    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f'Batch: {batch} — Intended Fix by Session Type per Animal', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


    
