# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

all_ABL = [20, 40, 60]
all_ILD_sorted = np.sort([1, -1, 2, -2, 4, -4, 8, -8, 16, -16])
gamma_vs_ILD_for_each_ABL = np.zeros((len(all_ABL),len(all_ILD_sorted)))
omega_vs_ILD_for_each_ABL = np.zeros((len(all_ABL),len(all_ILD_sorted)))
t_E_aff_vs_ILD_for_each_ABL = np.zeros((len(all_ABL),len(all_ILD_sorted)))

for a_idx, abl in enumerate(all_ABL):
    for i_idx, ILD in enumerate(all_ILD_sorted):
        pkl_file = os.path.join('each_cond_data_apr', f"vbmc_single_condn_ABL_{abl}_ILD_{ILD}.pkl")
        with open(pkl_file, 'rb') as f:
            vp = pickle.load(f)
        vp = vp.vp
        vp_samples = vp.sample(int(1e5))[0]
        
        gamma_vs_ILD_for_each_ABL[a_idx, i_idx] = vp_samples[:, 0].mean()
        omega_vs_ILD_for_each_ABL[a_idx, i_idx] = vp_samples[:, 1].mean()
        t_E_aff_vs_ILD_for_each_ABL[a_idx, i_idx] = vp_samples[:, 2].mean() * 1000


def calc_gamma_vs_ILD(params):
    rate_lambda, T0, theta_E = params
    return rate_lambda * theta_E * all_ILD_sorted / 17.37

def calc_omega_vs_ABL(params, ABL):
    rate_lambda, T0, theta_E = params
    return (2 / (T0 * (theta_E**2))) * (10**(rate_lambda*ABL/20))

juan_eye_fit_params = [0.118, 1/2220, 45]
vbmc_fit_params = [0.1310, 0.8378*1e-3, 33.7890]

# rate_lambda = 0.118
# T_0 = 1/2220
# theta_E = 45

plt.figure(figsize=(10, 10))
plt.subplot(3, 1, 1)
# plt.plot(all_ILD_sorted, calc_gamma_vs_ILD(juan_eye_fit_params), label='juan eye fit', ls='--', color='k', alpha=0.4)
# plt.plot(all_ILD_sorted, calc_gamma_vs_ILD(vbmc_fit_params), label='vbmc fit', ls='--', color='r', alpha=0.4)
for a_idx, ABL in enumerate(all_ABL):
    plt.plot(all_ILD_sorted, gamma_vs_ILD_for_each_ABL[a_idx, :], label=f'ABL={ABL}')

    plt.title('Gamma vs ILD')
    plt.legend()


plt.subplot(3, 1, 2)
for a_idx, ABL in enumerate(all_ABL):
    plt.plot(all_ILD_sorted, omega_vs_ILD_for_each_ABL[a_idx, :], label=f'ABL={ABL}')
    
    plt.axhline(calc_omega_vs_ABL(juan_eye_fit_params, ABL), color='k', ls='--', alpha=0.4)
    plt.axhline(calc_omega_vs_ABL(vbmc_fit_params, ABL), color='r', ls='--', alpha=0.4)

    plt.title('Omega vs ILD')
    plt.legend()


plt.subplot(3, 1, 3)
plt.axhline(75, color='k', ls='--', alpha=0.4)
plt.axhline(68.7, color='r', ls='--', alpha=0.4)
for a_idx, ABL in enumerate(all_ABL):
    plt.plot(all_ILD_sorted, t_E_aff_vs_ILD_for_each_ABL[a_idx, :], label=f'ABL={ABL}')
    

    plt.title('t_E_aff vs ILD')
    plt.legend()

# %%


import os
import glob

DESIRED_BATCHES = ['LED7']
# DESIRED_BATCHES = ['LED1']

# Directory containing the pickle files
results_dir = '/home/rlab/raghavendra/ddm_data/fit_animal_by_animal'

def find_batch_animal_pairs():
    pairs = []
    pattern = os.path.join(results_dir, 'results_*_animal_*.pkl')
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
                pairs.append((batch_name, animal_id))
        else:
            print(f"Warning: Invalid filename format: {filename}")
    return pairs

batch_animal_pairs = find_batch_animal_pairs()
print(f"Found {len(batch_animal_pairs)} batch-animal pairs: {batch_animal_pairs}")

# %%
def get_params_from_animal_pkl_file(batch_name, animal_id):
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

# %%
MODEL_TYPE = 'vanilla'
ild_theory = np.arange(-16, 16, 0.05)
def gamma_from_params(tied_params):
    rate_lambda = tied_params['rate_lambda']
    theta_E = tied_params['theta_E']
    return  theta_E* np.tanh(rate_lambda * ild_theory / 17.37)

import pickle

gamma_results = {}

for batch_name, animal_id in batch_animal_pairs:
    gamma_results[(batch_name, animal_id)] = {}
    # Vanilla
    MODEL_TYPE = 'vanilla'
    try:
        _, tied_params_vanilla, _, _ = get_params_from_animal_pkl_file(batch_name, animal_id)
        gamma_vanilla = gamma_from_params(tied_params_vanilla)
        gamma_results[(batch_name, animal_id)]['vanilla'] = gamma_vanilla
    except Exception as e:
        print(f"Error processing vanilla for {batch_name}, {animal_id}: {e}")
        gamma_results[(batch_name, animal_id)]['vanilla'] = None
    # Norm
    MODEL_TYPE = 'norm'
    try:
        _, tied_params_norm, _, _ = get_params_from_animal_pkl_file(batch_name, animal_id)
        gamma_norm = gamma_from_params(tied_params_norm)
        gamma_results[(batch_name, animal_id)]['norm'] = gamma_norm
    except Exception as e:
        print(f"Error processing norm for {batch_name}, {animal_id}: {e}")
        gamma_results[(batch_name, animal_id)]['norm'] = None

# Save to pickle
with open('gamma_results_by_animal.pkl', 'wb') as f:
    pickle.dump(gamma_results, f)
print("Saved gamma results to gamma_results_by_animal.pkl")

# %%
vanilla_all = []
norm_all = []
for batch_name, animal_id in batch_animal_pairs:
    vanilla_all.append(gamma_results[(batch_name, animal_id)]['vanilla'])
    norm_all.append(gamma_results[(batch_name, animal_id)]['norm'])
vanilla_all = np.array(vanilla_all)
norm_all = np.array(norm_all)



# %%
plt.figure(figsize=(4, 3))
for a_idx, ABL in enumerate(all_ABL):
    plt.scatter(all_ILD_sorted, gamma_vs_ILD_for_each_ABL[a_idx, :], label=f'ABL={ABL}')
plt.plot(ild_theory, vanilla_all.mean(axis=0), label='Vanilla', color='k', alpha=0.4)
plt.plot(ild_theory, norm_all.mean(axis=0), label='Norm', color='r', alpha=0.4)
plt.xlabel('ILD', fontsize=14)
plt.ylabel('Gamma', fontsize=14)
plt.xticks([-15, -5, 5, 15], fontsize=12)
plt.yticks([-2, 0, 2], fontsize=12)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# Legend removed as requested
plt.tight_layout(pad=0.2)
plt.show()


#%%

# %%
plt.figure(figsize=(4, 3))
for a_idx, ABL in enumerate(all_ABL):
    plt.scatter(all_ILD_sorted, gamma_vs_ILD_for_each_ABL[a_idx, :], label=f'ABL={ABL}')

# Plot mean lines
plt.plot(ild_theory, vanilla_all.T, label='Vanilla', color='k', alpha=0.4)
plt.plot(ild_theory, norm_all.T, label='Norm', color='r', alpha=0.4)





plt.xlabel('ILD', fontsize=14)
plt.ylabel('Gamma', fontsize=14)
plt.xticks([-15, -5, 5, 15], fontsize=12)
plt.yticks([-2, 0, 2], fontsize=12)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# Legend removed as requested
plt.tight_layout(pad=0.2)
plt.show()

# %%
### TEMP ###
def calc_omega_vs_ABL_from_params(tied_params, ABL, ILD):
    T0 = tied_params['T_0']
    theta_E = tied_params['theta_E']
    rate_lambda = tied_params['rate_lambda']
    rate_norm_l = tied_params.get('rate_norm_l', np.nan)
    return (1 / (T0 * (theta_E**2))) * (10**(rate_lambda*(1-rate_norm_l)*ABL/20)) \
        * (np.cosh(rate_lambda * ILD / 17.37) / np.cosh(rate_lambda * ILD * rate_norm_l / 17.37))

def calc_omega_vs_ABL_from_params_vanilla(tied_params, ABL):
    T0 = tied_params['T_0']
    theta_E = tied_params['theta_E']
    rate_lambda = tied_params['rate_lambda']
    return (1 / (T0 * (theta_E**2))) * (10**(rate_lambda*ABL/20)) \


# ABL_arr = [20, 40, 60]
# ILD_val = 16
# omega_results = {}
# for batch_name, animal_id in batch_animal_pairs:
#     gamma_results[(batch_name, animal_id)] = {}
#     # Vanilla
#     MODEL_TYPE = 'norm'
#     _, tied_params_norm, _, _ = get_params_from_animal_pkl_file(batch_name, animal_id)
#     for ABL in ABL_arr:
        
print('ILD  16, omega')
for a_idx, ABL in enumerate(all_ABL):
    # plt.bar('ILD', omega_vs_ILD_for_each_ABL[a_idx, -1], label=f'ABL={ABL}')
    print(f'Data ABL={ABL}: {omega_vs_ILD_for_each_ABL[a_idx, -1]:.2f}')
    omega_of_all_animals_per_ABL = []
    for batch_name, animal_id in batch_animal_pairs:
        MODEL_TYPE = 'norm'
        _, tied_params_norm, _, _ = get_params_from_animal_pkl_file(batch_name, animal_id)
        omega_of_all_animals_per_ABL.append(calc_omega_vs_ABL_from_params(tied_params_norm, ABL, 16))
    print(f'Norm model: ABL={ABL}: {np.mean(omega_of_all_animals_per_ABL):.2f}')
        
# %%

# --- Omega vs ILD plot for positive ILDs (1,2,4,8,16) ---
plt.figure(figsize=(10, 6))
# Find indices of positive ILDs in all_ILD_sorted
# pos_ILD_indices = [np.where(all_ILD_sorted == ild)[0][0] for ild in all_ILD_sorted]
pos_ILD_indices = np.arange(len(all_ILD_sorted))
colors = ['C0', 'C1', 'C2']
markers = ['o', 's', 'D']

for a_idx, ABL in enumerate(all_ABL):
    # Data omega
    omega_data = omega_vs_ILD_for_each_ABL[a_idx, pos_ILD_indices]
    plt.plot(all_ILD_sorted, omega_data, marker=markers[a_idx], color=colors[a_idx], label=f'Data ABL={ABL}', ls='--')


    
    # Model omega (mean over animals)
    MODEL_TYPE = 'norm'
    omega_model_all_animals = []
    for ILD in all_ILD_sorted:
        omega_animals = []
        for batch_name, animal_id in batch_animal_pairs:
            _, tied_params_norm, _, _ = get_params_from_animal_pkl_file(batch_name, animal_id)
            omega_animals.append(calc_omega_vs_ABL_from_params(tied_params_norm, ABL, ILD))
        omega_model_all_animals.append(np.mean(omega_animals))
    plt.plot(all_ILD_sorted, omega_model_all_animals, marker=markers[a_idx], color=colors[a_idx], label=f'Model ABL={ABL}')

    MODEL_TYPE = 'vanilla'
    omega_model_all_animals_vanilla = []
    for batch_name, animal_id in batch_animal_pairs:
        _, tied_params_vanilla, _, _ = get_params_from_animal_pkl_file(batch_name, animal_id)
        omega_model_all_animals_vanilla.append(calc_omega_vs_ABL_from_params_vanilla(tied_params_vanilla, ABL))
    plt.axhline(y=np.mean(omega_model_all_animals_vanilla), color='k', linestyle='--', label=f'Vanilla ABL={ABL}') 

plt.xlabel('ILD', fontsize=14)
plt.ylabel('Omega', fontsize=14)
plt.title('Omega vs ILD (positive ILDs)')
plt.xticks(all_ILD_sorted, fontsize=12)
plt.yticks(fontsize=12)
# plt.legend()
plt.tight_layout()
plt.show()

    
# %%
all_ILD_sorted