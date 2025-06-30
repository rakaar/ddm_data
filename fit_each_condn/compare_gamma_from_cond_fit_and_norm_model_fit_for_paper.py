# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import scipy.stats

# %%
DESIRED_BATCHES = ['SD', 'LED2', 'LED1', 'LED34', 'LED6', 'LED8', 'LED7']

# Base directory paths
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_dir = os.path.join(base_dir, 'batch_csvs')
results_dir = base_dir  # Directory containing the pickle files

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
                # Exclude animals 40, 41, 43 from LED2 batch and the entire LED1 batch
                if not ((batch_name == 'LED2' and animal_id in ['40', '41', '43']) or batch_name == 'LED1'):
                    pairs.append((batch_name, animal_id))
        else:
            print(f"Warning: Invalid filename format: {filename}")
    return pairs

batch_animal_pairs = find_batch_animal_pairs()

print(f"Found {len(batch_animal_pairs)} batch-animal pairs: {batch_animal_pairs}")


# %%


# Cond by Cond fit
# Average Gamma vs ILD for each ABL
def get_param_means_by_ABL_ILD(batch_name, animal_id, ABLs_to_fit, ILDs_to_fit, param_names=None):
    """
    Returns a dictionary with keys (ABL, ILD) and values as dicts of mean parameter values.
    Only includes (ABL, ILD) combinations for which the corresponding pickle file exists.
    param_names: list of parameter names in the order of columns in vp_samples, default uses [gamma, omega, t_E_aff, w, del_go]
    """
    import os
    import pickle
    import numpy as np
    
    if param_names is None:
        param_names = ['gamma', 'omega']
    
    param_dict = {}
    for ABL in ABLs_to_fit:
        for ILD in ILDs_to_fit:
            pkl_folder = '/home/rlab/raghavendra/ddm_data/fit_each_condn/each_animal_cond_fit_gama_omega_pkl_files'
            # vbmc_cond_by_cond_LED1_33_20_ILD_1_FIX_t_E_w_del_go_same_as_parametric
            pkl_file = os.path.join(pkl_folder, f'vbmc_cond_by_cond_{batch_name}_{animal_id}_{ABL}_ILD_{ILD}_FIX_t_E_w_del_go_same_as_parametric.pkl')
            if not os.path.exists(pkl_file):
                print(f'{pkl_file} does not exist')
                continue
            with open(pkl_file, 'rb') as f:
                vp = pickle.load(f)
            vp = vp.vp
            vp_samples = vp.sample(int(1e5))[0]
            # Each column: gamma, omega, t_E_aff, w, del_go
            means = {name: float(np.mean(vp_samples[:, i])) for i, name in enumerate(param_names)}
            param_dict[(ABL, ILD)] = means
    return param_dict


#%%
all_ABL = [20, 40, 60]
all_ILD_sorted = np.sort([1, -1, 2, -2, 4, -4, 8, -8, 16, -16])
gamma_cond_by_cond_fit_all_animals = {
    '20': np.full((len(batch_animal_pairs), len(all_ILD_sorted)), np.nan), 
    '40': np.full((len(batch_animal_pairs), len(all_ILD_sorted)), np.nan), 
    '60': np.full((len(batch_animal_pairs), len(all_ILD_sorted)), np.nan)
}

omega_cond_by_cond_fit_all_animals = {
    '20': np.full((len(batch_animal_pairs), len(all_ILD_sorted)), np.nan), 
    '40': np.full((len(batch_animal_pairs), len(all_ILD_sorted)), np.nan), 
    '60': np.full((len(batch_animal_pairs), len(all_ILD_sorted)), np.nan)
}
    
for animal_idx, (batch_name, animal_id) in enumerate(batch_animal_pairs):
# for batch_name, animal_id in [('LED7', '103')]:

    print('##########################################')
    print(f'Batch: {batch_name}, Animal: {animal_id}')
    print('##########################################')

    param_dict = get_param_means_by_ABL_ILD(batch_name, animal_id, all_ABL, all_ILD_sorted)
    # print(param_dict)
    for ABL in all_ABL:
        for ild_idx, ILD in enumerate(all_ILD_sorted):
            if (ABL,ILD) in param_dict:
                gamma_cond_by_cond_fit_all_animals[str(ABL)][animal_idx, ild_idx] = param_dict[(ABL,ILD)]['gamma']
                omega_cond_by_cond_fit_all_animals[str(ABL)][animal_idx, ild_idx] = param_dict[(ABL,ILD)]['omega']

# %%
# Cond by Cond fit, average of all animals
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
for ABL in all_ABL:
    # Calculate mean and standard error of mean
    mean_gamma = np.nanmean(gamma_cond_by_cond_fit_all_animals[str(ABL)], axis=0)
    sem_gamma = np.nanstd(gamma_cond_by_cond_fit_all_animals[str(ABL)], axis=0) / np.sqrt(np.sum(~np.isnan(gamma_cond_by_cond_fit_all_animals[str(ABL)]), axis=0))
    
    mean_omega = np.nanmean(omega_cond_by_cond_fit_all_animals[str(ABL)], axis=0)
    sem_omega = np.nanstd(omega_cond_by_cond_fit_all_animals[str(ABL)], axis=0) / np.sqrt(np.sum(~np.isnan(omega_cond_by_cond_fit_all_animals[str(ABL)]), axis=0))
    
    # Create scatter plots with error bars
    ax[0].errorbar(all_ILD_sorted, mean_gamma, yerr=sem_gamma, fmt='o', color=f'tab:{["blue", "orange", "green"][ABL//20-1]}', 
                  label=f'ABL={ABL}', capsize=0)
    ax[1].errorbar(all_ILD_sorted, mean_omega, yerr=sem_omega, fmt='o', color=f'tab:{["blue", "orange", "green"][ABL//20-1]}', 
                  label=f'ABL={ABL}', capsize=0)

ax[0].set_title('Gamma')
ax[1].set_title('Omega')
ax[0].set_xlabel('ILD')
ax[1].set_xlabel('ILD')
ax[0].set_ylabel('Gamma')
ax[1].set_ylabel('Omega')
ax[0].legend()
ax[1].legend()
plt.tight_layout()
plt.show()

# %%
# Helper Funcs to get params from model fit, plot gamma
def get_params_from_animal_pkl_file(batch_name, animal_id, MODEL_TYPE):
    pkl_file = f'/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/results_{batch_name}_animal_{animal_id}.pkl'
    with open(pkl_file, 'rb') as f:
        fit_results_data = pickle.load(f)
    vbmc_aborts_param_keys_map = {
        'V_A_samples': 'V_A',
        'theta_A_samples': 'theta_A',
        't_A_aff_samp': 't_A_aff'
    }
    vbmc_vanilla_tied_param_keys_map = {
        'rate_lambda_samples': 'rate_lambda',
        'T_0_samples': 'T_0',
        'theta_E_samples': 'theta_E',
        'w_samples': 'w',
        't_E_aff_samples': 't_E_aff',
        'del_go_samples': 'del_go'
    }
    vbmc_norm_tied_param_keys_map = {
        **vbmc_vanilla_tied_param_keys_map,
        'rate_norm_l_samples': 'rate_norm_l'
    }
    abort_keyname = "vbmc_aborts_results"
    if MODEL_TYPE == 'vanilla':
        tied_keyname = "vbmc_vanilla_tied_results"
        tied_param_keys_map = vbmc_vanilla_tied_param_keys_map
        is_norm = False
    elif MODEL_TYPE == 'norm':
        tied_keyname = "vbmc_norm_tied_results"
        tied_param_keys_map = vbmc_norm_tied_param_keys_map
        is_norm = True
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")
    abort_params = {}
    tied_params = {}
    rate_norm_l = 0
    if abort_keyname in fit_results_data:
        abort_samples = fit_results_data[abort_keyname]
        for param_samples_name, param_label in vbmc_aborts_param_keys_map.items():
            abort_params[param_label] = np.mean(abort_samples[param_samples_name])
    if tied_keyname in fit_results_data:
        tied_samples = fit_results_data[tied_keyname]
        for param_samples_name, param_label in tied_param_keys_map.items():
            tied_params[param_label] = np.mean(tied_samples[param_samples_name])
        if is_norm:
            rate_norm_l = tied_params.get('rate_norm_l', np.nan)
        else:
            rate_norm_l = 0
    else:
        print(f"Warning: {tied_keyname} not found in pickle for {batch_name}, {animal_id}")
    return abort_params, tied_params, rate_norm_l, is_norm

def gamma_from_params(tied_params, ild_theory):
    rate_lambda = tied_params['rate_lambda']
    theta_E = tied_params['theta_E']
    return  theta_E* np.tanh(rate_lambda * ild_theory / 17.37)

def calc_omega_vs_ABL_from_params_norm(tied_params, ABL, ILD_pts):
    T0 = tied_params['T_0']
    theta_E = tied_params['theta_E']
    rate_lambda = tied_params['rate_lambda']
    rate_norm_l = tied_params.get('rate_norm_l', np.nan)
    if np.isnan(rate_norm_l):
        print(tied_params)
        raise ValueError("rate_norm_l is NaN")
    return (1 / (T0 * (theta_E**2))) * (10**(rate_lambda*(1-rate_norm_l)*ABL/20)) \
        * (np.cosh(rate_lambda * ILD_pts / 17.37) / np.cosh(rate_lambda * ILD_pts * rate_norm_l / 17.37))

def calc_omega_vs_ABL_from_params_vanilla(tied_params, ABL, ILD_pts):
    T0 = tied_params['T_0']
    theta_E = tied_params['theta_E']
    rate_lambda = tied_params['rate_lambda']
    omega =  (1 / (T0 * (theta_E**2))) * (10**(rate_lambda*ABL/20)) 
    return np.ones_like(ILD_pts) * omega

# %%
# Model fit, Theoretical Gamma from parameters
ILD_pts = np.linspace(-16, 16, 100)
gamma_vanilla_model_fit_theoretical_all_animals = np.full((len(batch_animal_pairs), len(ILD_pts)), np.nan)
gamma_norm_model_fit_theoretical_all_animals = np.full((len(batch_animal_pairs), len(ILD_pts)), np.nan)

 
omega_vanilla_model_fit_theoretical_all_animals = {
    '20': np.full((len(batch_animal_pairs), len(ILD_pts)), np.nan), 
    '40': np.full((len(batch_animal_pairs), len(ILD_pts)), np.nan), 
    '60': np.full((len(batch_animal_pairs), len(ILD_pts)), np.nan)       
}
omega_norm_model_fit_theoretical_all_animals = {
    '20': np.full((len(batch_animal_pairs), len(ILD_pts)), np.nan), 
    '40': np.full((len(batch_animal_pairs), len(ILD_pts)), np.nan), 
    '60': np.full((len(batch_animal_pairs), len(ILD_pts)), np.nan)       
}

for animal_idx, (batch_name, animal_id) in enumerate(batch_animal_pairs):
    _, tied_params_vanilla, _, _ = get_params_from_animal_pkl_file(batch_name, animal_id, 'vanilla')
    _, tied_params_norm, _, _ = get_params_from_animal_pkl_file(batch_name, animal_id, 'norm')

    # gamma
    gamma_vanilla_model_fit_theoretical_all_animals[animal_idx] = gamma_from_params(tied_params_vanilla, ILD_pts)
    gamma_norm_model_fit_theoretical_all_animals[animal_idx] = gamma_from_params(tied_params_norm, ILD_pts)

    # omega
    for ABL in all_ABL:
        omega_vanilla_model_fit_theoretical_all_animals[str(ABL)][animal_idx] = calc_omega_vs_ABL_from_params_vanilla(tied_params_vanilla, ABL, ILD_pts)
        omega_norm_model_fit_theoretical_all_animals[str(ABL)][animal_idx] = calc_omega_vs_ABL_from_params_norm(tied_params_norm, ABL, ILD_pts)


# %%
# Theoretical gamma, omega from the model fit

# Vanilla model fit
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Plot gamma for each ABL - Vanilla model
for ABL in all_ABL:
    # Get gamma values for this ABL
    gamma_for_ABL = np.full((len(batch_animal_pairs), len(ILD_pts)), np.nan)
    for animal_idx in range(len(batch_animal_pairs)):
        gamma_for_ABL[animal_idx] = gamma_vanilla_model_fit_theoretical_all_animals[animal_idx]
    
    mean_gamma = np.nanmean(gamma_for_ABL, axis=0)
    sem_gamma = np.nanstd(gamma_for_ABL, axis=0) / np.sqrt(np.sum(~np.isnan(gamma_for_ABL), axis=0))
    
    ax[0].plot(ILD_pts, mean_gamma, color=f'tab:{["blue", "orange", "green"][ABL//20-1]}', label=f'ABL={ABL}')
    ax[0].fill_between(ILD_pts, mean_gamma - sem_gamma, mean_gamma + sem_gamma, 
                      color=f'tab:{["blue", "orange", "green"][ABL//20-1]}', alpha=0.3)

# Plot omega for each ABL - Vanilla model
for ABL in all_ABL:
    mean_omega = np.nanmean(omega_vanilla_model_fit_theoretical_all_animals[str(ABL)], axis=0)
    sem_omega = np.nanstd(omega_vanilla_model_fit_theoretical_all_animals[str(ABL)], axis=0) / np.sqrt(np.sum(~np.isnan(omega_vanilla_model_fit_theoretical_all_animals[str(ABL)]), axis=0))
    
    ax[1].plot(ILD_pts, mean_omega, color=f'tab:{["blue", "orange", "green"][ABL//20-1]}', label=f'ABL={ABL}')
    ax[1].fill_between(ILD_pts, mean_omega - sem_omega, mean_omega + sem_omega, 
                      color=f'tab:{["blue", "orange", "green"][ABL//20-1]}', alpha=0.3)

ax[0].set_title('Gamma - Vanilla Model')
ax[1].set_title('Omega - Vanilla Model')
ax[0].set_xlabel('ILD')
ax[1].set_xlabel('ILD')
ax[0].set_ylabel('Gamma')
ax[1].set_ylabel('Omega')
ax[0].legend()
ax[1].legend()
plt.tight_layout()
plt.show()

# Norm model fit
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Plot gamma for each ABL - Norm model
for ABL in all_ABL:
    # Get gamma values for this ABL
    gamma_for_ABL = np.full((len(batch_animal_pairs), len(ILD_pts)), np.nan)
    for animal_idx in range(len(batch_animal_pairs)):
        gamma_for_ABL[animal_idx] = gamma_norm_model_fit_theoretical_all_animals[animal_idx]
    
    mean_gamma = np.nanmean(gamma_for_ABL, axis=0)
    sem_gamma = np.nanstd(gamma_for_ABL, axis=0) / np.sqrt(np.sum(~np.isnan(gamma_for_ABL), axis=0))
    
    ax[0].plot(ILD_pts, mean_gamma, color=f'tab:{["blue", "orange", "green"][ABL//20-1]}', label=f'ABL={ABL}')
    ax[0].fill_between(ILD_pts, mean_gamma - sem_gamma, mean_gamma + sem_gamma, 
                      color=f'tab:{["blue", "orange", "green"][ABL//20-1]}', alpha=0.3)

# Plot omega for each ABL - Norm model
for ABL in all_ABL:
    mean_omega = np.nanmean(omega_norm_model_fit_theoretical_all_animals[str(ABL)], axis=0)
    sem_omega = np.nanstd(omega_norm_model_fit_theoretical_all_animals[str(ABL)], axis=0) / np.sqrt(np.sum(~np.isnan(omega_norm_model_fit_theoretical_all_animals[str(ABL)]), axis=0))
    
    ax[1].plot(ILD_pts, mean_omega, color=f'tab:{["blue", "orange", "green"][ABL//20-1]}', label=f'ABL={ABL}')
    ax[1].fill_between(ILD_pts, mean_omega - sem_omega, mean_omega + sem_omega, 
                      color=f'tab:{["blue", "orange", "green"][ABL//20-1]}', alpha=0.3)

ax[0].set_title('Gamma - Norm Model')
ax[1].set_title('Omega - Norm Model')
ax[0].set_xlabel('ILD')
ax[1].set_xlabel('ILD')
ax[0].set_ylabel('Gamma')
ax[1].set_ylabel('Omega')
ax[0].legend()
ax[1].legend()
plt.tight_layout()
plt.show()

# %%
# Plot cond by cond fit, model fit theoretical gamma, omega

# 1. Vanilla model comparison
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Plot condition by condition fit gamma
for ABL in all_ABL:
    # Calculate mean and standard error of mean for condition fit
    mean_gamma = np.nanmean(gamma_cond_by_cond_fit_all_animals[str(ABL)], axis=0)
    sem_gamma = np.nanstd(gamma_cond_by_cond_fit_all_animals[str(ABL)], axis=0) / np.sqrt(np.sum(~np.isnan(gamma_cond_by_cond_fit_all_animals[str(ABL)]), axis=0))
    
    # Plot condition fit as scatter points with error bars
    ax[0].errorbar(all_ILD_sorted, mean_gamma, yerr=sem_gamma, fmt='o', color=f'tab:{["blue", "orange", "green"][ABL//20-1]}', 
                  label=f'ABL={ABL} (cond fit)', capsize=0)

# Plot theoretical vanilla model gamma
for ABL in all_ABL:
    # Get gamma values for this ABL
    gamma_for_ABL = np.full((len(batch_animal_pairs), len(ILD_pts)), np.nan)
    for animal_idx in range(len(batch_animal_pairs)):
        gamma_for_ABL[animal_idx] = gamma_vanilla_model_fit_theoretical_all_animals[animal_idx]
    
    mean_gamma = np.nanmean(gamma_for_ABL, axis=0)
    sem_gamma = np.nanstd(gamma_for_ABL, axis=0) / np.sqrt(np.sum(~np.isnan(gamma_for_ABL), axis=0))
    
    ax[0].plot(ILD_pts, mean_gamma, color=f'tab:{["blue", "orange", "green"][ABL//20-1]}', 
              label=f'ABL={ABL} (vanilla)', linestyle='--')
    ax[0].fill_between(ILD_pts, mean_gamma - sem_gamma, mean_gamma + sem_gamma, 
                      color=f'tab:{["blue", "orange", "green"][ABL//20-1]}', alpha=0.2)

# Plot condition by condition fit omega
for ABL in all_ABL:
    # Calculate mean and standard error of mean for condition fit
    mean_omega = np.nanmean(omega_cond_by_cond_fit_all_animals[str(ABL)], axis=0)
    sem_omega = np.nanstd(omega_cond_by_cond_fit_all_animals[str(ABL)], axis=0) / np.sqrt(np.sum(~np.isnan(omega_cond_by_cond_fit_all_animals[str(ABL)]), axis=0))
    
    # Plot condition fit as scatter points with error bars
    ax[1].errorbar(all_ILD_sorted, mean_omega, yerr=sem_omega, fmt='o', color=f'tab:{["blue", "orange", "green"][ABL//20-1]}', 
                  label=f'ABL={ABL} (cond fit)', capsize=0)

# Plot theoretical vanilla model omega
for ABL in all_ABL:
    mean_omega = np.nanmean(omega_vanilla_model_fit_theoretical_all_animals[str(ABL)], axis=0)
    sem_omega = np.nanstd(omega_vanilla_model_fit_theoretical_all_animals[str(ABL)], axis=0) / np.sqrt(np.sum(~np.isnan(omega_vanilla_model_fit_theoretical_all_animals[str(ABL)]), axis=0))
    
    ax[1].plot(ILD_pts, mean_omega, color=f'tab:{["blue", "orange", "green"][ABL//20-1]}', 
              label=f'ABL={ABL} (vanilla)', linestyle='--')
    ax[1].fill_between(ILD_pts, mean_omega - sem_omega, mean_omega + sem_omega, 
                      color=f'tab:{["blue", "orange", "green"][ABL//20-1]}', alpha=0.2)

ax[0].set_title('Gamma: Condition Fit vs Vanilla Model')
ax[1].set_title('Omega: Condition Fit vs Vanilla Model')
ax[0].set_xlabel('ILD')
ax[1].set_xlabel('ILD')
ax[0].set_ylabel('Gamma')
ax[1].set_ylabel('Omega')
# ax[0].legend()
# ax[1].legend()
plt.tight_layout()
plt.savefig('gamma_omega_cond_fit_vs_vanilla_model.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Norm model comparison
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Plot condition by condition fit gamma
for ABL in all_ABL:
    # Calculate mean and standard error of mean for condition fit
    mean_gamma = np.nanmean(gamma_cond_by_cond_fit_all_animals[str(ABL)], axis=0)
    sem_gamma = np.nanstd(gamma_cond_by_cond_fit_all_animals[str(ABL)], axis=0) / np.sqrt(np.sum(~np.isnan(gamma_cond_by_cond_fit_all_animals[str(ABL)]), axis=0))
    
    # Plot condition fit as scatter points with error bars
    ax[0].errorbar(all_ILD_sorted, mean_gamma, yerr=sem_gamma, fmt='o', color=f'tab:{["blue", "orange", "green"][ABL//20-1]}', 
                  label=f'ABL={ABL} (cond fit)', capsize=0)

# Plot theoretical norm model gamma
for ABL in all_ABL:
    # Get gamma values for this ABL
    gamma_for_ABL = np.full((len(batch_animal_pairs), len(ILD_pts)), np.nan)
    for animal_idx in range(len(batch_animal_pairs)):
        gamma_for_ABL[animal_idx] = gamma_norm_model_fit_theoretical_all_animals[animal_idx]
    
    mean_gamma = np.nanmean(gamma_for_ABL, axis=0)
    sem_gamma = np.nanstd(gamma_for_ABL, axis=0) / np.sqrt(np.sum(~np.isnan(gamma_for_ABL), axis=0))
    
    ax[0].plot(ILD_pts, mean_gamma, color=f'tab:{["blue", "orange", "green"][ABL//20-1]}', 
              label=f'ABL={ABL} (norm)', linestyle='--')
    ax[0].fill_between(ILD_pts, mean_gamma - sem_gamma, mean_gamma + sem_gamma, 
                      color=f'tab:{["blue", "orange", "green"][ABL//20-1]}', alpha=0.2)

# Plot condition by condition fit omega
for ABL in all_ABL:
    # Calculate mean and standard error of mean for condition fit
    mean_omega = np.nanmean(omega_cond_by_cond_fit_all_animals[str(ABL)], axis=0)
    sem_omega = np.nanstd(omega_cond_by_cond_fit_all_animals[str(ABL)], axis=0) / np.sqrt(np.sum(~np.isnan(omega_cond_by_cond_fit_all_animals[str(ABL)]), axis=0))
    
    # Plot condition fit as scatter points with error bars
    ax[1].errorbar(all_ILD_sorted, mean_omega, yerr=sem_omega, fmt='o', color=f'tab:{["blue", "orange", "green"][ABL//20-1]}', 
                  label=f'ABL={ABL} (cond fit)', capsize=0)

# Plot theoretical norm model omega
for ABL in all_ABL:
    mean_omega = np.nanmean(omega_norm_model_fit_theoretical_all_animals[str(ABL)], axis=0)
    sem_omega = np.nanstd(omega_norm_model_fit_theoretical_all_animals[str(ABL)], axis=0) / np.sqrt(np.sum(~np.isnan(omega_norm_model_fit_theoretical_all_animals[str(ABL)]), axis=0))
    
    ax[1].plot(ILD_pts, mean_omega, color=f'tab:{["blue", "orange", "green"][ABL//20-1]}', 
              label=f'ABL={ABL} (norm)', linestyle='--')
    ax[1].fill_between(ILD_pts, mean_omega - sem_omega, mean_omega + sem_omega, 
                      color=f'tab:{["blue", "orange", "green"][ABL//20-1]}', alpha=0.2)

ax[0].set_title('Gamma: Condition Fit vs Norm Model')
ax[1].set_title('Omega: Condition Fit vs Norm Model')
ax[0].set_xlabel('ILD')
ax[1].set_xlabel('ILD')
ax[0].set_ylabel('Gamma')
ax[1].set_ylabel('Omega')
# ax[0].legend()
# ax[1].legend()
plt.tight_layout()
plt.savefig('gamma_omega_cond_fit_vs_norm_model.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# why is omega inverted in condition by condition

# %%
# Find the ILD with lowest omega for each animal at all ABLs and plot histograms

# Create a figure with 3 subplots for the 3 ABLs
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Process each ABL
for i, ABL in enumerate(all_ABL):
    # Extract omega values for this ABL
    omega_ABL = omega_cond_by_cond_fit_all_animals[str(ABL)]
    
    # Find the ILD with the lowest omega for each animal
    lowest_omega_ILD_indices = np.nanargmin(omega_ABL, axis=1)
    
    # Convert indices to actual ILD values
    lowest_omega_ILDs = [all_ILD_sorted[idx] if not np.isnan(idx) else np.nan for idx in lowest_omega_ILD_indices]
    
    # Filter out NaN values (animals with no data)
    valid_ILDs = [ild for ild in lowest_omega_ILDs if not np.isnan(ild)]
    
    # Create the histogram in the corresponding subplot
    axes[i].hist(valid_ILDs, bins=np.arange(min(all_ILD_sorted)-0.5, max(all_ILD_sorted)+1.5, 1), 
             edgecolor='black', alpha=0.7)
    axes[i].set_xlabel('ILD with lowest omega', fontsize=14)
    if i == 0:
        axes[i].set_ylabel('Number of animals', fontsize=14)
    axes[i].set_title(f'ILDs with lowest omega at ABL {ABL}', fontsize=16)
    
    # Add vertical line at ILD=0
    axes[i].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # Customize x-ticks to match the actual ILD values
    axes[i].set_xticks(all_ILD_sorted)
    
    # Remove top and right spines
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)
    
    # Print the ILD with lowest omega for each animal
    print(f"\nILD with lowest omega for each animal at ABL {ABL}:")
    for (batch_name, animal_id), ild in zip(batch_animal_pairs, lowest_omega_ILDs):
        if not np.isnan(ild):
            print(f"Batch: {batch_name}, Animal: {animal_id}, Lowest omega ILD: {ild}")

plt.tight_layout()
plt.savefig('lowest_omega_ILD_histogram_all_ABLs.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Plot omega vs ILD for all animals with different colors for each animal
# Using a 3x3 grid: 3 columns for ABLs, 3 rows with ~10 animals per row

# Determine how many animals to show per row
num_animals = len(batch_animal_pairs)
animals_per_row = 10
num_rows = 3  # Fixed at 3 rows as requested

# Create a figure with 3x3 grid of subplots
fig, axes = plt.subplots(num_rows, 3, figsize=(18, 15), sharex=True)

# Create a colormap with distinct colors
cmap = plt.cm.tab20  # tab20 gives 20 distinct colors
cmap2 = plt.cm.tab20b  # Additional colors if needed
colors = []
for i in range(num_animals):
    if i < 20:
        colors.append(cmap(i % 20))
    else:
        colors.append(cmap2((i - 20) % 20))

# Process each row
for row in range(num_rows):
    # Determine which animals to plot in this row
    start_idx = row * animals_per_row
    end_idx = min(start_idx + animals_per_row, num_animals)
    
    # Process each ABL (column)
    for col, ABL in enumerate(all_ABL):
        # Get the omega data for this ABL
        omega_data = omega_cond_by_cond_fit_all_animals[str(ABL)]
        
        # Plot each animal's data for this row
        for animal_idx in range(start_idx, end_idx):
            if animal_idx < num_animals:  # Check if we still have animals to plot
                batch_name, animal_id = batch_animal_pairs[animal_idx]
                
                # Get this animal's omega values
                animal_omega = omega_data[animal_idx, :]
                
                # Plot only if there are valid data points
                if not np.all(np.isnan(animal_omega)):
                    axes[row, col].scatter(all_ILD_sorted, animal_omega, color=colors[animal_idx], 
                                       s=30, alpha=0.7, label=f'{batch_name}_{animal_id}')
                    
                    # Connect points with lines for better visibility of trends
                    axes[row, col].plot(all_ILD_sorted, animal_omega, color=colors[animal_idx], 
                                    alpha=0.7, linewidth=1)
        
        # Set subplot title and labels
        if row == 0:  # Only add title to the top row
            axes[row, col].set_title(f'Omega vs ILD at ABL {ABL}', fontsize=14)

        

        if row == num_rows - 1:  # Only add x-label to the bottom row
            axes[row, col].set_xlabel('ILD', fontsize=12)
        
        if col == 0:  # Only add y-label to the first column
            axes[row, col].set_ylabel(f'Omega (Animals {start_idx+1}-{end_idx})', fontsize=12)
        
        # Add legend within each subplot
        # if end_idx - start_idx > 0:  # Only add legend if there are animals to show
        #     axes[row, col].legend(fontsize=8, loc='upper right')

        
        
        # Remove top and right spines
        axes[row, col].spines['top'].set_visible(False)
        axes[row, col].spines['right'].set_visible(False)
        
        # Set the same y-axis limits for all plots in the same row for better comparison
        all_valid_omega = []
        for animal_idx in range(start_idx, end_idx):
            if animal_idx < num_animals:
                all_valid_omega.extend(omega_data[animal_idx, :][~np.isnan(omega_data[animal_idx, :])])
        
        if all_valid_omega:  # Only set limits if we have valid data
            min_omega = np.min(all_valid_omega)
            max_omega = np.max(all_valid_omega)
            padding = (max_omega - min_omega) * 0.1  # Add 10% padding
            axes[row, col].set_ylim(min_omega - padding, max_omega + padding)

# Adjust layout and save
plt.tight_layout()
fig.savefig('omega_vs_ILD_all_animals_grid.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

# %%
# Create simple heatmap visualizations of omega values
# Sort animals by ILD with lowest omega value

# Create a figure with 3 subplots (one for each ABL)
fig, axes = plt.subplots(1, 3, figsize=(18, 8))

# Process each ABL
for i, ABL in enumerate(all_ABL):
    # Get omega values for this ABL
    omega_data = omega_cond_by_cond_fit_all_animals[str(ABL)]
    
    # Find the ILD with lowest omega for each animal
    lowest_omega_ILD_indices = np.nanargmin(omega_data, axis=1)
    
    # Convert indices to actual ILD values
    lowest_omega_ILDs = [all_ILD_sorted[idx] if not np.isnan(idx) and idx < len(all_ILD_sorted) 
                         else np.nan for animal_idx, idx in enumerate(lowest_omega_ILD_indices)]
    
    # Create a list of (animal_index, lowest_omega_ILD) tuples for sorting
    animal_ILD_pairs = [(idx, ild) for idx, ild in enumerate(lowest_omega_ILDs) if not np.isnan(ild)]
    
    # Sort by lowest_omega_ILD (from lowest to highest)
    sorted_pairs = sorted(animal_ILD_pairs, key=lambda x: x[1])
    
    # Extract the sorted animal indices
    sorted_indices = [pair[0] for pair in sorted_pairs]
    
    # Create a new array with sorted animal data
    sorted_omega_data = omega_data[sorted_indices, :]
    
    # Normalize each row (animal) independently between 0 and 1
    normalized_data = np.zeros_like(sorted_omega_data)
    
    # Process each animal (row) separately
    for row_idx in range(sorted_omega_data.shape[0]):
        row_data = sorted_omega_data[row_idx, :]
        valid_data = row_data[~np.isnan(row_data)]
        
        if len(valid_data) > 0:  # Check if there's valid data in this row
            row_min = np.min(valid_data)
            row_max = np.max(valid_data)
            
            # Avoid division by zero if min and max are the same
            if row_min != row_max:
                normalized_data[row_idx, :] = (row_data - row_min) / (row_max - row_min)
            else:
                # If all values are the same, set to 0.5 (middle of range)
                normalized_data[row_idx, :] = 0.5 * np.ones_like(row_data)
        
        # NaN values remain as NaN
        normalized_data[row_idx, np.isnan(row_data)] = np.nan
    
    # Create the heatmap with normalized data
    im = axes[i].imshow(normalized_data, aspect='auto', cmap='viridis', vmin=0, vmax=1,
                      extent=[min(all_ILD_sorted), max(all_ILD_sorted), len(sorted_indices)-0.5, -0.5])
    
    # Add colorbar
    plt.colorbar(im, ax=axes[i])
    
    # Set title and labels
    axes[i].set_title(f'Omega vs ILD at ABL {ABL}')
    axes[i].set_xlabel('ILD')
    if i == 0:
        axes[i].set_ylabel('Animal Index (sorted by lowest omega ILD)')
    
    # Set x-ticks to match ILD values
    axes[i].set_xticks(all_ILD_sorted)
    
    # Print the sorted animals and their lowest omega ILDs
    # print(f"\nAnimals sorted by ILD with lowest omega at ABL {ABL}:")
    # for rank, animal_idx in enumerate(sorted_indices):
    #     batch_name, animal_id = batch_animal_pairs[animal_idx]
    #     ild = lowest_omega_ILDs[animal_idx]
    #     print(f"Rank {rank+1}: Batch: {batch_name}, Animal: {animal_id}, Lowest omega ILD: {ild}")

# Adjust layout and save
plt.tight_layout()
fig.savefig('omega_heatmap_sorted_by_lowest_ILD.png', dpi=300)
plt.show()

# %%
# Create simple heatmap visualizations of absolute gamma values
# Use the same sorting order as in the omega plot (by ILD with lowest omega value)

# Create a figure with 3 subplots (one for each ABL)
fig, axes = plt.subplots(1, 3, figsize=(18, 8))

# Process each ABL
for i, ABL in enumerate(all_ABL):
    # Get omega values for this ABL (for sorting)
    omega_data = omega_cond_by_cond_fit_all_animals[str(ABL)]
    
    # Get gamma values for this ABL
    gamma_data = gamma_cond_by_cond_fit_all_animals[str(ABL)]
    
    # Find the ILD with lowest omega for each animal
    lowest_omega_ILD_indices = np.nanargmin(omega_data, axis=1)
    
    # Convert indices to actual ILD values
    lowest_omega_ILDs = [all_ILD_sorted[idx] if not np.isnan(idx) and idx < len(all_ILD_sorted) 
                         else np.nan for animal_idx, idx in enumerate(lowest_omega_ILD_indices)]
    
    # Create a list of (animal_index, lowest_omega_ILD) tuples for sorting
    animal_ILD_pairs = [(idx, ild) for idx, ild in enumerate(lowest_omega_ILDs) if not np.isnan(ild)]
    
    # Sort by lowest_omega_ILD (from lowest to highest)
    sorted_pairs = sorted(animal_ILD_pairs, key=lambda x: x[1])
    
    # Extract the sorted animal indices
    sorted_indices = [pair[0] for pair in sorted_pairs]
    
    # Create a new array with sorted animal data and take absolute value of gamma
    sorted_gamma_data = np.abs(gamma_data[sorted_indices, :])
    
    # Normalize each row (animal) independently between 0 and 1
    normalized_data = np.zeros_like(sorted_gamma_data)
    
    # Process each animal (row) separately
    for row_idx in range(sorted_gamma_data.shape[0]):
        row_data = sorted_gamma_data[row_idx, :]
        valid_data = row_data[~np.isnan(row_data)]
        
        if len(valid_data) > 0:  # Check if there's valid data in this row
            row_min = np.min(valid_data)
            row_max = np.max(valid_data)
            
            # Avoid division by zero if min and max are the same
            if row_min != row_max:
                normalized_data[row_idx, :] = (row_data - row_min) / (row_max - row_min)
            # else:
            #     # If all values are the same, set to 0.5 (middle of range)
            #     normalized_data[row_idx, :] = 0.5 * np.ones_like(row_data)
        
        # NaN values remain as NaN
        normalized_data[row_idx, np.isnan(row_data)] = np.nan
    
    # Create the heatmap with normalized data
    im = axes[i].imshow(normalized_data, aspect='auto', cmap='viridis', vmin=0, vmax=1,
                      extent=[min(all_ILD_sorted), max(all_ILD_sorted), len(sorted_indices)-0.5, -0.5])
    
    # Add colorbar
    plt.colorbar(im, ax=axes[i])
    
    # Set title and labels
    axes[i].set_title(f'Absolute Gamma vs ILD at ABL {ABL}')
    axes[i].set_xlabel('ILD')
    if i == 0:
        axes[i].set_ylabel('Animal Index (sorted by lowest omega ILD)')
    
    # Set x-ticks to match ILD values
    axes[i].set_xticks(all_ILD_sorted)

# Adjust layout and save
plt.tight_layout()
fig.savefig('abs_gamma_heatmap_sorted_by_lowest_omega_ILD.png', dpi=300)
plt.show()

# %%
# Create a scatter plot comparing ILDs with minimum absolute gamma vs minimum omega for each animal at each ABL

# Create the figure
plt.figure(figsize=(10, 8))

# Define colors for each ABL
abl_colors = {'20': 'tab:blue', '40': 'tab:orange', '60': 'tab:green'}

# Store all points for correlation calculation
all_min_gamma_ilds = []
all_min_omega_ilds = []

# Process each ABL
for ABL in all_ABL:
    # Get data for this ABL
    gamma_data = gamma_cond_by_cond_fit_all_animals[str(ABL)]
    omega_data = omega_cond_by_cond_fit_all_animals[str(ABL)]
    
    # Take absolute value of gamma
    abs_gamma_data = np.abs(gamma_data)
    
    # Find the ILD with minimum absolute gamma and minimum omega for each animal
    min_abs_gamma_ild_indices = np.nanargmin(abs_gamma_data, axis=1)
    min_omega_ild_indices = np.nanargmin(omega_data, axis=1)
    
    # Convert indices to actual ILD values
    min_abs_gamma_ilds = [all_ILD_sorted[idx] if not np.isnan(idx) and idx < len(all_ILD_sorted) 
                         else np.nan for idx in min_abs_gamma_ild_indices]
    min_omega_ilds = [all_ILD_sorted[idx] if not np.isnan(idx) and idx < len(all_ILD_sorted) 
                     else np.nan for idx in min_omega_ild_indices]
    
    # Store valid pairs for correlation calculation
    for g_ild, o_ild in zip(min_abs_gamma_ilds, min_omega_ilds):
        if not np.isnan(g_ild) and not np.isnan(o_ild):
            all_min_gamma_ilds.append(g_ild)
            all_min_omega_ilds.append(o_ild)
    
    # Plot scatter points for this ABL
    plt.scatter(min_abs_gamma_ilds, min_omega_ilds, color=abl_colors[str(ABL)], 
                alpha=0.7, label=f'ABL {ABL}', s=80)

# Calculate correlation if there are enough data points
if len(all_min_gamma_ilds) > 1:
    correlation = np.corrcoef(all_min_gamma_ilds, all_min_omega_ilds)[0, 1]
    plt.title(f'ILD with Minimum Absolute Gamma vs ILD with Minimum Omega\nCorrelation: {correlation:.3f}')
else:
    plt.title('ILD with Minimum Absolute Gamma vs ILD with Minimum Omega')

# Add identity line
min_val = min(plt.xlim()[0], plt.ylim()[0])
max_val = max(plt.xlim()[1], plt.ylim()[1])
plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

# Set labels and legend
plt.xlabel('ILD with Minimum Absolute Gamma')
plt.ylabel('ILD with Minimum Omega')
plt.legend()
plt.grid(True, alpha=0.3)
# Make the plot square
plt.axis('equal')

# Save the figure
plt.tight_layout()
plt.xlim(-2,2)
plt.savefig('min_abs_gamma_vs_min_omega_ILD_scatter.png', dpi=300)
plt.show()

# %%
# ILD with min omega vs w

# Create the figure
plt.figure(figsize=(10, 8))

# Define colors for each ABL
abl_colors = {'20': 'tab:blue', '40': 'tab:orange', '60': 'tab:green'}

# Process each ABL
for ABL in all_ABL:
    # Get omega values for this ABL
    omega_data = omega_cond_by_cond_fit_all_animals[str(ABL)]
    
    # Find the ILD with minimum omega for each animal
    min_omega_ild_indices = np.nanargmin(omega_data, axis=1)
    
    # Convert indices to actual ILD values
    min_omega_ilds = [all_ILD_sorted[idx] if not np.isnan(idx) and idx < len(all_ILD_sorted) 
                     else np.nan for idx in min_omega_ild_indices]
    
    # Get w values for each animal
    w_values = []
    valid_ilds = []
    valid_animal_indices = []
    
    for animal_idx, (batch_name, animal_id) in enumerate(batch_animal_pairs):
        if not np.isnan(min_omega_ilds[animal_idx]):
            # Get w parameter for this animal
            MODEL_TYPE = 'vanilla'
            abort_params, vanilla_tied_params, rate_norm_l, is_norm = get_params_from_animal_pkl_file(batch_name, animal_id)
            MODEL_TYPE = 'norm'
            abort_params, norm_tied_params, rate_norm_l, is_norm = get_params_from_animal_pkl_file(batch_name, animal_id)
            
            # Calculate average w from vanilla and norm models
            w = (vanilla_tied_params['w'] + norm_tied_params['w']) / 2
            
            w_values.append(w)
            valid_ilds.append(min_omega_ilds[animal_idx])
            valid_animal_indices.append(animal_idx)
    
    # Plot scatter points for this ABL
    plt.scatter(valid_ilds, w_values, color=abl_colors[str(ABL)], 
                alpha=0.7, label=f'ABL {ABL}', s=80)

# Set title and labels
plt.title('ILD with Minimum Omega vs w')
plt.xlabel('ILD with Minimum Omega')
plt.ylabel('w')
plt.legend()
plt.grid(True, alpha=0.3)

# Save the figure
plt.tight_layout()
plt.xlim(-17, 17)
plt.savefig('min_omega_ILD_vs_w_scatter.png', dpi=300)
plt.show()