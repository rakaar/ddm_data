# %%
"""
Plot w vs ILD for all animals from 5-param condition fit.
- One figure per abs_ILD
- Y-axis: animal
- X-axis: w (urgency weight)
- Average +ILD and -ILD for each abs_ILD
"""
import os
import sys
sys.path.insert(0, '/home/rlab/raghavendra/ddm_data/fit_each_condn')
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

# %%
# =============================================================================
# Configuration
# =============================================================================
COND_FIT_MORE_PARAMS_FOLDER = '/home/rlab/raghavendra/ddm_data/fit_each_condn/each_animal_cond_fit_5_params_pkl_files'
DESIRED_BATCHES = ['SD', 'LED34', 'LED6', 'LED8', 'LED7', 'LED34_even']
batch_dir = '/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/batch_csvs'

ILD_VALUES = [1, 2, 4, 8, 16]
ABL_arr = [20, 40, 60]

# %%
# =============================================================================
# Load batch-animal pairs
# =============================================================================
batch_files = [f'batch_{batch_name}_valid_and_aborts.csv' for batch_name in DESIRED_BATCHES]
merged_data = pd.concat([
    pd.read_csv(os.path.join(batch_dir, fname)) 
    for fname in batch_files if os.path.exists(os.path.join(batch_dir, fname))
], ignore_index=True)

merged_valid = merged_data[merged_data['success'].isin([1, -1])].copy()
batch_animal_pairs = sorted(list(map(tuple, merged_valid[['batch_name', 'animal']].drop_duplicates().values)))
print(f"Found {len(batch_animal_pairs)} batch-animal pairs")

# %%
# =============================================================================
# Function to load del_go from 5-param condition fit
# =============================================================================
def get_w_from_pkl(batch_name, animal_id, ABL, ILD):
    """
    Load w from 5-param condition-by-condition fit pkl file.
    Returns mean of vp samples for w (index 3).
    """
    pkl_file = os.path.join(COND_FIT_MORE_PARAMS_FOLDER, 
                            f'vbmc_cond_by_cond_{batch_name}_{animal_id}_{ABL}_ILD_{ILD}_5_params.pkl')
    if not os.path.exists(pkl_file):
        return None
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        vp = data.vp
        vp_samples = vp.sample(int(1e4))[0]
        # 5 params: gamma, omega, t_E_aff, w, del_go
        w = float(np.mean(vp_samples[:, 3]))
        return w
    except Exception as e:
        print(f"Error loading {pkl_file}: {e}")
        return None

# %%
# =============================================================================
# Collect w for all animals, all ILDs, all ABLs
# =============================================================================
# Structure: {abs_ild: {abl: [(animal_label, w), ...]}}
delay_data = {abs_ild: {abl: [] for abl in ABL_arr} for abs_ild in ILD_VALUES}

for batch_name, animal_id in batch_animal_pairs:
    animal_label = f"{batch_name}_{animal_id}"
    
    for abs_ild in ILD_VALUES:
        for abl in ABL_arr:
            # Get w for +ILD and -ILD, then average
            w_pos = get_w_from_pkl(batch_name, animal_id, abl, abs_ild)
            w_neg = get_w_from_pkl(batch_name, animal_id, abl, -abs_ild)
            
            if w_pos is not None and w_neg is not None:
                w_avg = (w_pos + w_neg) / 2
            elif w_pos is not None:
                w_avg = w_pos
            elif w_neg is not None:
                w_avg = w_neg
            else:
                w_avg = None
            
            if w_avg is not None:
                delay_data[abs_ild][abl].append((animal_label, w_avg))

print("Data collection complete.")

# %%
# =============================================================================
# Plot: One figure per abs_ILD, animals on y-axis, del_go on x-axis
# =============================================================================
abl_colors = {20: 'tab:blue', 40: 'tab:orange', 60: 'tab:green'}

for abs_ild in ILD_VALUES:
    fig, ax = plt.subplots(figsize=(8, 10))
    
    # Collect all animals that have data for this ILD
    all_animals = set()
    for abl in ABL_arr:
        for animal_label, _ in delay_data[abs_ild][abl]:
            all_animals.add(animal_label)
    
    # Sort animals
    all_animals = sorted(list(all_animals))
    animal_to_y = {animal: i for i, animal in enumerate(all_animals)}
    
    # Plot each ABL and collect all delays for mean calculation
    all_delays = []
    for abl in ABL_arr:
        animals = [a for a, _ in delay_data[abs_ild][abl]]
        delays = [d for _, d in delay_data[abs_ild][abl]]
        all_delays.extend(delays)
        y_positions = [animal_to_y[a] for a in animals]
        
        ax.scatter(delays, y_positions, c=abl_colors[abl], label=f'ABL {abl}', 
                   alpha=0.7, s=50, edgecolors='black', linewidths=0.5)
    
    # Add vertical line at mean of all means
    if all_delays:
        grand_mean = np.mean(all_delays)
        ax.axvline(x=grand_mean, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean = {grand_mean:.3f}')
    
    ax.set_yticks(range(len(all_animals)))
    ax.set_yticklabels(all_animals, fontsize=8)
    ax.set_xlabel('w (s)', fontsize=12)
    ax.set_ylabel('Animal', fontsize=12)
    ax.set_title(f'w for |ILD| = {abs_ild} dB', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'w_vs_animal_ILD_{abs_ild}.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: w_vs_animal_ILD_{abs_ild}.png")

print("\nDone!")
