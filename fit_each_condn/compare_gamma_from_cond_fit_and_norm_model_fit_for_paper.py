# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import glob
import scipy.stats
from collections import defaultdict

# %%
# --- Get Batch-Animal Pairs ---
DESIRED_BATCHES = ['SD', 'LED34', 'LED6', 'LED8', 'LED7', 'LED34_even']
batch_dir = '/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/batch_csvs'
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
# FOR FIG 2 paper - Gamma vanilla and omega seperate

# 3. Gamma plot for vanilla model
gamma_plot_data = {
    'all_ABL': all_ABL,
    'gamma_cond_by_cond_fit_all_animals': gamma_cond_by_cond_fit_all_animals,
    'all_ILD_sorted': all_ILD_sorted,
    'batch_animal_pairs': batch_animal_pairs,
    'ILD_pts': ILD_pts,
    'gamma_vanilla_model_fit_theoretical_all_animals': gamma_vanilla_model_fit_theoretical_all_animals
}
with open('vanilla_gamma_fig2_data.pkl', 'wb') as f:
    pickle.dump(gamma_plot_data, f)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))

# Plot condition by condition fit gamma
for ABL in all_ABL:
    # Calculate mean and standard error of mean for condition fit
    mean_gamma = np.nanmean(gamma_cond_by_cond_fit_all_animals[str(ABL)], axis=0)
    sem_gamma = np.nanstd(gamma_cond_by_cond_fit_all_animals[str(ABL)], axis=0) / np.sqrt(np.sum(~np.isnan(gamma_cond_by_cond_fit_all_animals[str(ABL)]), axis=0))
    
    # Plot condition fit as scatter points with error bars
    ax.errorbar(all_ILD_sorted, mean_gamma, yerr=sem_gamma, fmt='o', color=f'tab:{["blue", "orange", "green"][ABL//20-1]}', 
                  label=f'ABL={ABL} (cond fit)', capsize=0)

# Plot theoretical vanilla model gamma
for ABL in all_ABL:
    # Get gamma values for this ABL
    gamma_for_ABL = np.full((len(batch_animal_pairs), len(ILD_pts)), np.nan)
    for animal_idx in range(len(batch_animal_pairs)):
        gamma_for_ABL[animal_idx] = gamma_vanilla_model_fit_theoretical_all_animals[animal_idx]
    
    mean_gamma = np.nanmean(gamma_for_ABL, axis=0)
    sem_gamma = np.nanstd(gamma_for_ABL, axis=0) / np.sqrt(np.sum(~np.isnan(gamma_for_ABL), axis=0))
    
    ax.plot(ILD_pts, mean_gamma, color=f'tab:{["blue", "orange", "green"][ABL//20-1]}', 
              label=f'ABL={ABL} (vanilla)', linestyle='--')
    ax.fill_between(ILD_pts, mean_gamma - sem_gamma, mean_gamma + sem_gamma, 
                      color=f'tab:{["blue", "orange", "green"][ABL//20-1]}', alpha=0.2)

# ax.set_title('Vanilla', fontsize=24)
ax.set_xlabel('ILD', fontsize=25)
ax.set_ylabel('Gamma', fontsize=25)
ax.tick_params(axis='both', which='major', labelsize=24)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xticks([-15, -5, 5, 15])
ax.set_yticks([-2, 0, 2])
ax.set_ylim(-3, 3)

# ax.legend()
plt.tight_layout()
plt.savefig('gamma_cond_fit_vs_vanilla_model.png', dpi=300, bbox_inches='tight')
plt.show()
# %%

gamma_plot_data = {
    'all_ABL': all_ABL,
    'gamma_cond_by_cond_fit_all_animals': gamma_cond_by_cond_fit_all_animals,
    'all_ILD_sorted': all_ILD_sorted,
    'batch_animal_pairs': batch_animal_pairs,
    'ILD_pts': ILD_pts,
    'gamma_norm_model_fit_theoretical_all_animals': gamma_norm_model_fit_theoretical_all_animals
}
with open('norm_gamma_fig2_data.pkl', 'wb') as f:
    pickle.dump(gamma_plot_data, f)

print(f'saved to norm_gamma_fig2_data.pkl')
# 4. Gamma plot for norm model
fig, ax = plt.subplots(1, 1, figsize=(5, 5))

# Plot condition by condition fit gamma
for ABL in all_ABL:
    # Calculate mean and standard error of mean for condition fit
    mean_gamma = np.nanmean(gamma_cond_by_cond_fit_all_animals[str(ABL)], axis=0)
    sem_gamma = np.nanstd(gamma_cond_by_cond_fit_all_animals[str(ABL)], axis=0) / np.sqrt(np.sum(~np.isnan(gamma_cond_by_cond_fit_all_animals[str(ABL)]), axis=0))
    
    # Plot condition fit as scatter points with error bars
    ax.errorbar(all_ILD_sorted, mean_gamma, yerr=sem_gamma, fmt='o', color=f'tab:{["blue", "orange", "green"][ABL//20-1]}', 
                  label=f'ABL={ABL} (cond fit)', capsize=0)

# Plot theoretical norm model gamma
for ABL in all_ABL:
    # Get gamma values for this ABL
    gamma_for_ABL = np.full((len(batch_animal_pairs), len(ILD_pts)), np.nan)
    for animal_idx in range(len(batch_animal_pairs)):
        gamma_for_ABL[animal_idx] = gamma_norm_model_fit_theoretical_all_animals[animal_idx]
    
    mean_gamma = np.nanmean(gamma_for_ABL, axis=0)
    sem_gamma = np.nanstd(gamma_for_ABL, axis=0) / np.sqrt(np.sum(~np.isnan(gamma_for_ABL), axis=0))
    
    ax.plot(ILD_pts, mean_gamma, color=f'tab:{["blue", "orange", "green"][ABL//20-1]}', 
              label=f'ABL={ABL} (norm)', linestyle='--')
    ax.fill_between(ILD_pts, mean_gamma - sem_gamma, mean_gamma + sem_gamma, 
                      color=f'tab:{["blue", "orange", "green"][ABL//20-1]}', alpha=0.2)

# ax.set_title('Norm', fontsize=24)
ax.set_xlabel('ILD', fontsize=25)
ax.set_ylabel('Gamma', fontsize=25)
ax.tick_params(axis='both', which='major', labelsize=24)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xticks([-15, -5, 5, 15])
ax.set_yticks([-2, 0, 2])
ax.set_ylim(-3, 3)
# ax.legend()
plt.tight_layout()
plt.savefig('gamma_cond_fit_vs_norm_model.png', dpi=300, bbox_inches='tight')
plt.show()

# %% 
# check gamma linearity
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

for i, ABL in enumerate(all_ABL):
    # Calculate mean and standard error of mean for condition fit
    mean_gamma = np.nanmean(gamma_cond_by_cond_fit_all_animals[str(ABL)], axis=0)
    sem_gamma = np.nanstd(gamma_cond_by_cond_fit_all_animals[str(ABL)], axis=0) / np.sqrt(np.sum(~np.isnan(gamma_cond_by_cond_fit_all_animals[str(ABL)]), axis=0))
    
    # Plot condition fit as scatter points with error bars
    axes[i].errorbar(all_ILD_sorted, mean_gamma, yerr=sem_gamma, fmt='o', color=f'tab:{["blue", "orange", "green"][ABL//20-1]}', 
                  capsize=0)

    # Fit a straight line to mean gamma vs ILD restricted to ILDs in [-8, 8]
    mask_range = (np.abs(all_ILD_sorted) <= 8)
    x = all_ILD_sorted[mask_range]
    y = mean_gamma[mask_range]
    valid = ~np.isnan(y)
    if np.sum(valid) >= 2:
        slope, intercept = np.polyfit(x[valid], y[valid], 1)
        x_fit = np.linspace(-8, 8, 100)
        y_fit = slope * x_fit + intercept
        axes[i].plot(x_fit, y_fit, color=f'tab:{["blue", "orange", "green"][ABL//20-1]}', linewidth=2)

    axes[i].set_title(f'ABL={ABL}')

axes[0].set_ylabel('Gamma')
for ax_i in axes:
    ax_i.set_xlabel('ILD')

plt.tight_layout()
plt.show()

# %%
# Compare two sigmoid models with original data (with fitted parameters)

from scipy.optimize import curve_fit

# Define the functions
def log_sigmoid(x, a, d):
    exp_term = np.exp(-d * x)
    
    numerator = (1 - a/2) + (a/2) * exp_term
    denominator = (a/2) + (1 - a/2) * exp_term
    
    # Add epsilon for numerical stability
    epsilon = 1e-12
    ratio = numerator / (denominator + epsilon)
    ratio = np.clip(ratio, epsilon, None)
    
    return np.log(ratio)


def scaled_tanh(x, b, c):
    return c * np.tanh(b * x)

# Create 2x3 subplot (2 models, 3 ABLs)
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Plot for each ABL
fitted_params = {}  # To store fitted parameters
for col, ABL in enumerate(all_ABL):
    # Calculate mean and standard error of mean for condition fit
    mean_gamma = np.nanmean(gamma_cond_by_cond_fit_all_animals[str(ABL)], axis=0)
    sem_gamma = np.nanstd(gamma_cond_by_cond_fit_all_animals[str(ABL)], axis=0) / np.sqrt(np.sum(~np.isnan(gamma_cond_by_cond_fit_all_animals[str(ABL)]), axis=0))
    
    # Filter out NaN values for fitting
    valid_indices = ~np.isnan(mean_gamma)
    x_data = all_ILD_sorted[valid_indices]
    y_data = mean_gamma[valid_indices]
    y_err = sem_gamma[valid_indices]
    
    # Fit log-sigmoid model
    try:
        # Fit with bounds to ensure parameters are reasonable
        # Initial guess for [a, d]
        p0 = [0.001, 1.0]
        # Bounds for [a, d]
        # a: (0, 1), d: (0.01, 10)
        bounds = ([1e-6, 0.01], [1-1e-6, 10])
        popt_log, _ = curve_fit(log_sigmoid, x_data, y_data, p0=p0, 
                               bounds=bounds, sigma=y_err, absolute_sigma=True)
        a_fitted, d_fitted = popt_log
        fitted_params[f'log_sigmoid_ABL{ABL}'] = (a_fitted, d_fitted)
        
        # Generate fitted curve
        x_model = np.linspace(-16, 16, 300)
        y_log_sigmoid_fitted = log_sigmoid(x_model, a_fitted, d_fitted)
        
        # Top row: Log-sigmoid model
        axes[0, col].errorbar(all_ILD_sorted, mean_gamma, yerr=sem_gamma, fmt='o', 
                             color=f'tab:{["blue", "orange", "green"][ABL//20-1]}', 
                             capsize=0, label='Data')
        axes[0, col].plot(x_model, y_log_sigmoid_fitted, 'k-', linewidth=2, 
                         label=f'Log-sigmoid fit')
        axes[0, col].set_title(f'ABL={ABL} - Log-sigmoid (a={a_fitted:.5f}, d={d_fitted:.3f})')
        axes[0, col].set_xlabel('ILD')
        axes[0, col].set_ylabel('Gamma')
        axes[0, col].grid(True, alpha=0.3)
    except Exception as e:
        print(f"Could not fit log-sigmoid for ABL={ABL}: {e}")
        axes[0, col].set_title(f'ABL={ABL} - Log-sigmoid (fit failed)')
    
    # Fit scaled tanh model
    try:
        # Fit with bounds to ensure reasonable parameter values
        popt_tanh, _ = curve_fit(scaled_tanh, x_data, y_data, p0=[0.14, 3.3], 
                                bounds=([1e-6, 0.1], [5, 10]), sigma=y_err, absolute_sigma=True)
        b_fitted, c_fitted = popt_tanh
        fitted_params[f'scaled_tanh_ABL{ABL}'] = (b_fitted, c_fitted)
        
        # Generate fitted curve
        x_model = np.linspace(-16, 16, 300)
        y_scaled_tanh_fitted = scaled_tanh(x_model, b_fitted, c_fitted)
        
        # Bottom row: Scaled tanh model
        axes[1, col].errorbar(all_ILD_sorted, mean_gamma, yerr=sem_gamma, fmt='o', 
                             color=f'tab:{["blue", "orange", "green"][ABL//20-1]}', 
                             capsize=0, label='Data')
        axes[1, col].plot(x_model, y_scaled_tanh_fitted, 'k-', linewidth=2, 
                         label=f'Scaled tanh fit')
        axes[1, col].set_title(f'ABL={ABL} - Scaled tanh (b={b_fitted:.3f}, c={c_fitted:.3f})')
        axes[1, col].set_xlabel('ILD')
        axes[1, col].set_ylabel('Gamma')
        axes[1, col].grid(True, alpha=0.3)
    except Exception as e:
        print(f"Could not fit scaled tanh for ABL={ABL}: {e}")
        axes[1, col].set_title(f'ABL={ABL} - Scaled tanh (fit failed)')

# Add legends
axes[0, 0].legend()
axes[1, 0].legend()

plt.tight_layout()
plt.show()

# Print fitted parameters
print("Fitted Parameters:")
for key, value in fitted_params.items():
    if 'log_sigmoid' in key:
        a, d = value
        print(f"{key}: a = {a:.6f}, d = {d:.6f}")
    else:  # scaled_tanh
        b, c = value
        print(f"{key}: b = {b:.6f}, c = {c:.6f}")

# %%
x = np.arange(-16, 16, 0.1)
a = 0.4
numerator = (1 - a/2) + (a/2) * np.exp(-x)
denominator = (a/2) + (1 - a/2) * np.exp(-x)
plt.plot(x, numerator/denominator, label='arg of log')
plt.plot(x, np.log(numerator/denominator), label='log')
plt.legend()
