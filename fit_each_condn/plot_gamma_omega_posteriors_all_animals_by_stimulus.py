# %%
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import glob

# %%
def get_posterior_samples_by_ABL_ILD(batch_name, animal_id, ABLs_to_fit, ILDs_to_fit):
    """
    Returns a dictionary with keys (ABL, ILD) and values as dicts of posterior samples.
    Only includes (ABL, ILD) combinations for which the corresponding pickle file exists.
    """
    pkl_folder = '/home/rlab/raghavendra/ddm_data/fit_each_condn/each_animal_cond_fit_gama_omega_pkl_files'
    param_dict = {}
    
    for ABL in ABLs_to_fit:
        for ILD in ILDs_to_fit:
            pkl_file = os.path.join(pkl_folder, f'vbmc_cond_by_cond_{batch_name}_{animal_id}_{ABL}_ILD_{ILD}_FIX_t_E_w_del_go_same_as_parametric.pkl')
            if not os.path.exists(pkl_file):
                continue
            with open(pkl_file, 'rb') as f:
                vp = pickle.load(f)
            vp = vp.vp
            vp_samples = vp.sample(int(1e5))[0]
            # Each column: gamma, omega, t_E_aff, w, del_go
            param_dict[(ABL, ILD)] = {
                'gamma_samples': vp_samples[:, 0],
                'omega_samples': vp_samples[:, 1]
            }
    return param_dict

# %%
DESIRED_BATCHES = ['SD', 'LED2', 'LED1', 'LED34', 'LED6', 'LED8', 'LED7']

base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = base_dir

def find_batch_animal_pairs():
    pairs = []
    pattern = os.path.join(results_dir, '../fit_animal_by_animal/results_*_animal_*.pkl')
    pickle_files = glob.glob(pattern)
    for pickle_file in pickle_files:
        filename = os.path.basename(pickle_file)
        parts = filename.split('_')
        if len(parts) >= 4:
            batch_index = parts.index('animal') - 1 if 'animal' in parts else 1
            animal_index = parts.index('animal') + 1 if 'animal' in parts else 2
            batch_name = parts[batch_index]
            animal_id = parts[animal_index].split('.')[0]
            if batch_name in DESIRED_BATCHES:
                if not (batch_name == 'LED2' and animal_id in ['40', '41', '43']) and not batch_name == 'LED1':
                    pairs.append((batch_name, animal_id))
        else:
            print(f"Warning: Invalid filename format: {filename}")
    return pairs

batch_animal_pairs = find_batch_animal_pairs()

print(f"Found {len(batch_animal_pairs)} batch-animal pairs")

# %%
all_ABLs_cond = [20, 40, 60]
all_ILDs_cond = [1, -1, 2, -2, 4, -4, 8, -8, 16, -16]

# Collect all posterior samples for each stimulus
all_stimuli_data = {}  # (ABL, ILD): {'gamma_samples': [], 'omega_samples': []}

for batch_name, animal_id in batch_animal_pairs:
    print(f'Loading {batch_name}, Animal {animal_id}')
    param_dict = get_posterior_samples_by_ABL_ILD(batch_name, animal_id, all_ABLs_cond, all_ILDs_cond)
    
    for (ABL, ILD), samples in param_dict.items():
        if (ABL, ILD) not in all_stimuli_data:
            all_stimuli_data[(ABL, ILD)] = {'gamma_samples': [], 'omega_samples': []}
        all_stimuli_data[(ABL, ILD)]['gamma_samples'].append(samples['gamma_samples'])
        all_stimuli_data[(ABL, ILD)]['omega_samples'].append(samples['omega_samples'])

print(f"Found {len(all_stimuli_data)} stimuli with data")

# %%
# Plot gamma posteriors for all stimuli
fig_gamma, axes_gamma = plt.subplots(10, 3, figsize=(12, 30))
fig_gamma.suptitle('Gamma Posterior Distributions', fontsize=14)

for i, ABL in enumerate(all_ABLs_cond):
    for j, ILD in enumerate(all_ILDs_cond):
        ax = axes_gamma[j, i]  # ILD on y-axis (rows), ABL on x-axis (cols)
        
        if (ABL, ILD) not in all_stimuli_data:
            ax.set_visible(False)
            continue
        
        all_gamma_samples = all_stimuli_data[(ABL, ILD)]['gamma_samples']
        
        # Set bounds based on ILD sign
        if ILD > 0:
            gamma_bounds = [-1, 5]
        elif ILD < 0:
            gamma_bounds = [-5, 1]
        else:
            gamma_bounds = [-1, 5]
        
        # Plot histograms for all animals with histtype='step'
        for idx, gamma_samples in enumerate(all_gamma_samples):
            ax.hist(gamma_samples, bins=50, density=True, histtype='step', 
                    alpha=0.3, linewidth=1, color='blue')
        
        ax.set_xlim(gamma_bounds)
        ax.set_xlabel('gamma')
        ax.set_ylabel('Density')
        ax.set_title(f'ABL={ABL}, ILD={ILD}')
        ax.grid(True, alpha=0.3)

plt.tight_layout()
gamma_output_path = '/home/rlab/raghavendra/ddm_data/fit_each_condn/gamma_omega_posterior_pdfs/gamma_all_animals_by_stimulus.png'
plt.savefig(gamma_output_path, dpi=150, bbox_inches='tight')
print(f'Saved {gamma_output_path}')
# plt.close(fig_gamma)

# %%
# Plot omega posteriors for all stimuli
fig_omega, axes_omega = plt.subplots(10, 3, figsize=(12, 30))
fig_omega.suptitle('Omega Posterior Distributions', fontsize=14)

for i, ABL in enumerate(all_ABLs_cond):
    for j, ILD in enumerate(all_ILDs_cond):
        ax = axes_omega[j, i]  # ILD on y-axis (rows), ABL on x-axis (cols)
        
        if (ABL, ILD) not in all_stimuli_data:
            ax.set_visible(False)
            continue
        
        all_omega_samples = all_stimuli_data[(ABL, ILD)]['omega_samples']
        omega_bounds = [0.1, 15]
        
        # Plot histograms for all animals with histtype='step'
        for idx, omega_samples in enumerate(all_omega_samples):
            ax.hist(omega_samples, bins=50, density=True, histtype='step',
                    alpha=0.3, linewidth=1, color='red')
        
        ax.set_xlim(omega_bounds)
        ax.set_xlabel('omega')
        ax.set_ylabel('Density')
        ax.set_title(f'ABL={ABL}, ILD={ILD}')
        ax.grid(True, alpha=0.3)

plt.tight_layout()
omega_output_path = '/home/rlab/raghavendra/ddm_data/fit_each_condn/gamma_omega_posterior_pdfs/omega_all_animals_by_stimulus.png'
plt.savefig(omega_output_path, dpi=150, bbox_inches='tight')
print(f'Saved {omega_output_path}')
# plt.close(fig_omega)

print('Done!')

# %%
