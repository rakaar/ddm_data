# %%
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import pandas as pd
import random
from scipy.integrate import trapezoid as trapz
from pyvbmc import VBMC
import corner
from scipy.integrate import cumulative_trapezoid as cumtrapz
import pickle
from led_off_gamma_omega_pdf_utils import cum_pro_and_reactive_trunc_fn, up_or_down_RTs_fit_OPTIM_V_A_change_gamma_omega_with_w_fn
from led_off_gamma_omega_pdf_utils import cum_pro_and_reactive, up_or_down_RTs_fit_OPTIM_V_A_change_gamma_omega_fn,\
         rho_A_t_VEC_fn, up_or_down_RTs_fit_OPTIM_V_A_change_gamma_omega_P_A_C_A_wrt_stim_fn
from led_off_gamma_omega_pdf_utils import up_or_down_RTs_fit_OPTIM_V_A_change_gamma_omega_with_w_PA_CA_fn
from matplotlib.backends.backend_pdf import PdfPages
import os
import glob

# %%
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
                # Exclude animals 40, 41, 43 from LED2 batch
                if not (batch_name == 'LED2' and animal_id in ['40', '41', '43']):
                    pairs.append((batch_name, animal_id))
        else:
            print(f"Warning: Invalid filename format: {filename}")
    return pairs

batch_animal_pairs = find_batch_animal_pairs()

print(f"Found {len(batch_animal_pairs)} batch-animal pairs: {batch_animal_pairs}")

# %%
def get_params_from_animal_pkl_file(batch_name, animal_id):
    pkl_file = f'../fit_animal_by_animal/results_{batch_name}_animal_{animal_id}.pkl'
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
all_ABLs_cond = [20, 40, 60]
all_ILDs_cond = [1, -1, 2, -2, 4, -4, 8, -8, 16, -16]
vbmc_fit_saving_path = '/home/rlab/raghavendra/ddm_data/fit_each_condn/each_animal_cond_fit_gama_omega_pkl_files'
K_max = 10

for batch_name, animal_id in batch_animal_pairs:
# for batch_name, animal_id in [('LED7', '103')]:

    print('##########################################')
    print(f'Batch: {batch_name}, Animal: {animal_id}')
    print('##########################################')

    MODEL_TYPE = 'vanilla'
    abort_params, vanilla_tied_params, rate_norm_l, is_norm = get_params_from_animal_pkl_file(batch_name, animal_id)
    MODEL_TYPE = 'norm'
    abort_params, norm_tied_params, rate_norm_l, is_norm = get_params_from_animal_pkl_file(batch_name, animal_id)
    
    # take w, t_E_aff, del_go avg from both vanilla and norm tied params
    w = (vanilla_tied_params['w'] + norm_tied_params['w']) / 2
    t_E_aff = (vanilla_tied_params['t_E_aff'] + norm_tied_params['t_E_aff']) / 2
    del_go = (vanilla_tied_params['del_go'] + norm_tied_params['del_go']) / 2
    
    print(f"Batch: {batch_name}, Animal: {animal_id}")
    print(f"w: {w}")
    print(f"t_E_aff: {t_E_aff}")
    print(f"del_go: {del_go}")
    print("\n")

    # abort params
    V_A = abort_params['V_A']
    theta_A = abort_params['theta_A']
    t_A_aff = abort_params['t_A_aff']

    # get the database from batch_csvs
    file_name = f'../fit_animal_by_animal/batch_csvs/batch_{batch_name}_valid_and_aborts.csv'
    df = pd.read_csv(file_name)
    df_animal = df[df['animal'] == int(animal_id)]
    df_animal_success = df_animal[df_animal['success'].isin([1, -1])]
    df_animal_success_rt_filter = df_animal_success[(df_animal_success['RTwrtStim'] <= 1) & (df_animal_success['RTwrtStim'] > 0)]

    # --- PDF output block ---
    plt.close('all')  # Close any existing figures to ensure clean state
    
    # Setup PDF folder and file
    pdf_folder = '/home/rlab/raghavendra/ddm_data/fit_each_condn/each_animal_cond_fit_diagnostic_files'
    os.makedirs(pdf_folder, exist_ok=True)
    pdf_filename = f'{batch_name}_{animal_id}_diagnostics_cond_by_cond_FIX_t_E_aff_w_del_go.pdf'
    pdf_path = os.path.join(pdf_folder, pdf_filename)
    
    # --- PDF output in a single context manager ---
    with PdfPages(pdf_path) as pdf:
        # Create cover page right away
        fig_cover = plt.figure(figsize=(8, 4))
        plt.axis('off')
        plt.title('Diagnostics Summary', fontsize=18, pad=30)
        plt.text(0.5, 0.7, f'Animal: {animal_id}', ha='center', va='center', fontsize=16)
        plt.text(0.5, 0.5, f'Batch: {batch_name}', ha='center', va='center', fontsize=16)
        pdf.savefig(fig_cover)
        plt.close(fig_cover)
        
        # Diagnostics - RTD choice
        N_theory = int(1e3)
        t_pts = np.arange(-1, 2, 0.001)
        t_stim_samples = df_animal['intended_fix'].sample(N_theory, replace=True).values
        P_A_samples = np.zeros((N_theory, len(t_pts)))
        t_trunc = 0.3 # wrt fix
        for idx, t_stim in enumerate(t_stim_samples):
            # t is wrt t_stim, t + t_stim is wrt fix
            # Vectorized version using rho_A_t_VEC_fn
            t_shifted = t_pts + t_stim
            mask = t_shifted > t_trunc
            vals = np.zeros_like(t_pts)
            if np.any(mask):
                vals[mask] = rho_A_t_VEC_fn(t_shifted[mask] - t_A_aff, V_A, theta_A)
            P_A_samples[idx, :] = vals


        from scipy.integrate import trapezoid
        P_A_mean = np.mean(P_A_samples, axis=0)
        area = trapezoid(P_A_mean, t_pts)

        if area != 0:
            P_A_mean = P_A_mean / area
        C_A_mean = cumtrapz(P_A_mean, t_pts, initial=0)


        # --- Compute and store theory/data for all ABL/ILD combinations ---
        theory_curves = {}  # (ABL, ILD): dict with up_mean, down_mean, up_plus_down, t_pts_0_1, up_plus_down_mean
        rt_data = {}        # (ABL, ILD): data_a_i_rt
        ABLs_to_fit = [20, 40, 60]
        ILDs_to_fit = [1, -1, 2, -2, 4, -4, 8, -8, 16, -16]
        param_dict = get_param_means_by_ABL_ILD(batch_name, animal_id, ABLs_to_fit, ILDs_to_fit)

        for ABL in ABLs_to_fit:
            for ILD in ILDs_to_fit:
                # get gamma, omega, t_E_aff, w, del_go from pkl file
                if (ABL, ILD) not in param_dict:
                    print(f'ABL={ABL}, ILD={ILD} not in param_dict')
                    continue
                gamma = param_dict[(ABL, ILD)].get('gamma')
                omega = param_dict[(ABL, ILD)].get('omega')
                if gamma is None or omega is None:
                    continue
        

                bound = 1
                up_mean = np.array([
                    up_or_down_RTs_fit_OPTIM_V_A_change_gamma_omega_with_w_PA_CA_fn(
                        t, P_A_mean[idx], C_A_mean[idx],
                        gamma, omega, 0, t_E_aff, del_go, bound, w, K_max
                    ) for idx, t in enumerate(t_pts)
                ])
                down_mean = np.array([
                    up_or_down_RTs_fit_OPTIM_V_A_change_gamma_omega_with_w_PA_CA_fn(
                        t, P_A_mean[idx], C_A_mean[idx],
                        gamma, omega, 0, t_E_aff, del_go, -bound, w, K_max
                    ) for idx, t in enumerate(t_pts)
                ])
                mask_0_1 = (t_pts >= 0) & (t_pts <= 1)
                t_pts_0_1 = t_pts[mask_0_1]
                up_plus_down = up_mean + down_mean
                # area up and down
                print(f'ABL={ABL}, ILD={ILD}')
                print(f'area up {trapezoid(up_mean, t_pts) :.2f}')
                print(f'area down {trapezoid(down_mean, t_pts) :.2f}')
                up_plus_down_masked = up_plus_down[mask_0_1]
                area_masked = trapezoid(up_plus_down_masked, t_pts_0_1)

                if area_masked != 0:
                    up_plus_down_mean = up_plus_down_masked / area_masked
                else:
                    up_plus_down_mean = up_plus_down_masked
                theory_curves[(ABL, ILD)] = {
                    'up_mean': up_mean,
                    'down_mean': down_mean,
                    't_pts': t_pts,

                    'up_mean_mask': up_mean[mask_0_1],
                    'down_mean_mask': down_mean[mask_0_1],
                    't_pts_0_1': t_pts_0_1,
                    'up_plus_down_mean': up_plus_down_mean
                }
                # Data
                data_a_i = df_animal_success_rt_filter[
                    (df_animal_success_rt_filter['ABL'] == ABL) &
                    (df_animal_success_rt_filter['ILD'] == ILD)
                ]
                if 'timed_fix' in data_a_i.columns:
                    data_a_i_rt = data_a_i['timed_fix'] - data_a_i['intended_fix']
                else:
                    data_a_i_rt = data_a_i['TotalFixTime'] - data_a_i['intended_fix']
                rt_data[(ABL, ILD)] = data_a_i_rt

        # --- Plot from stored arrays ---
        n_ABLs = len(ABLs_to_fit)
        n_ILDs = len(ILDs_to_fit)
        fig, axes = plt.subplots(n_ABLs, n_ILDs, figsize=(4*n_ILDs, 3*n_ABLs), sharex=True, sharey=True)
        for i_ABL, ABL in enumerate(ABLs_to_fit):
            for i_ILD, ILD in enumerate(np.sort(ILDs_to_fit)):
                ax = axes[i_ABL, i_ILD] if n_ABLs > 1 and n_ILDs > 1 else (
                    axes[i_ILD] if n_ABLs == 1 else axes[i_ABL]
                )
                if (ABL, ILD) not in theory_curves:
                    ax.set_visible(False)
                    continue
                tc = theory_curves[(ABL, ILD)]
                ax.plot(tc['t_pts_0_1'], tc['up_plus_down_mean'], label="theory")
                ax.hist(rt_data[(ABL, ILD)], bins=np.arange(0,1,0.02), density=True, histtype='step', label="data")
                ax.set_title(f"ABL={ABL}, ILD={ILD}")
                if i_ABL == n_ABLs-1:
                    ax.set_xlabel("RT (s)")
                if i_ILD == 0:
                    ax.set_ylabel("Density")
                ax.legend(fontsize=8)
        plt.tight_layout()
        pdf.savefig(plt.gcf())
        plt.close()

        # psychometrics:
        # Plot psychometric function: prob(choice==1) vs ILD for each ABL in data
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        for ABL in ABLs_to_fit:
            # --- Data psychometric ---
            df_abl = df_animal_success_rt_filter[df_animal_success_rt_filter['ABL'] == ABL]
            prob_right_data = df_abl.groupby('ILD')['choice'].apply(lambda x: (x == 1).mean())
            ILDs_data = prob_right_data.index.values
            plt.scatter(ILDs_data, prob_right_data.values, label=f'Data ABL={ABL}', alpha=0.7, marker='o')
            # --- Theory psychometric ---
            ILDs_theory = []
            prob_right_theory = []
            for ILD in np.sort(ILDs_to_fit):
                key = (ABL, ILD)
                if key not in theory_curves:
                    continue
                tc = theory_curves[key]
                area_up = trapezoid(tc['up_mean_mask'], tc['t_pts_0_1'])
                area_down = trapezoid(tc['down_mean_mask'], tc['t_pts_0_1'])
                p_right = area_up / (area_up + area_down) if (area_up + area_down) > 0 else np.nan
                print(f'ILD = {ILD}, p_right = {p_right}')
                ILDs_theory.append(ILD)
                prob_right_theory.append(p_right)
            # Sort for line plot
            ILDs_theory = np.array(ILDs_theory)
            prob_right_theory = np.array(prob_right_theory)
            idx_sort = np.argsort(ILDs_theory)
            plt.plot(ILDs_theory[idx_sort], prob_right_theory[idx_sort], label=f'Theory ABL={ABL}', marker='x')
        plt.axhline(0.5, color='gray', ls='--', lw=1)
        plt.xlabel('ILD (dB)')
        plt.ylabel('P(choice=right)')
        plt.title(f'Psychometric curve: Data vs Theory\nAnimal: {animal_id}, Batch: {batch_name}')
        plt.legend(title='Curve')
        plt.tight_layout()
        pdf.savefig(plt.gcf())
        plt.close()

  
    


        # --- gamma plot ---
        plt.figure(figsize=(4, 3))
        all_ILDs = sorted(set(ILD for ILD in ILDs_to_fit))
        for ABL in ABLs_to_fit:
            gammas = []
            ILDs_plot = []
            for ILD in ILDs_to_fit:
                if (ABL, ILD) in param_dict:
                    gamma = param_dict[(ABL, ILD)].get('gamma')
                else:
                    continue
                if gamma is not None:
                    gammas.append(gamma)
                ILDs_plot.append(ILD)
            plt.scatter(ILDs_plot, gammas, marker='o', label=f'ABL={ABL}')
        plt.xlabel('ILD (dB)')
        plt.ylabel('gamma')
        plt.title(f'gamma vs ILD for each ABL\nAnimal: {animal_id}, Batch: {batch_name}')
        plt.xticks(all_ILDs)
        plt.tight_layout()
        pdf.savefig(plt.gcf())
        plt.close()

        # --- omega plot ---
        plt.figure(figsize=(4, 3))
        all_ILDs = sorted(set(ILD for ILD in ILDs_to_fit))
        for ABL in ABLs_to_fit:
            omegas = []
            ILDs_plot = []
            for ILD in ILDs_to_fit:
                if (ABL, ILD) in param_dict:
                    omega = param_dict[(ABL, ILD)].get('omega')
                else:
                    continue
                if omega is not None:
                    omegas.append(omega)
                ILDs_plot.append(ILD)
            plt.scatter(ILDs_plot, omegas, marker='o', label=f'ABL={ABL}')
        plt.xlabel('ILD (dB)')
        plt.ylabel('omega')
        plt.title(f'omega vs ILD for each ABL\nAnimal: {animal_id}, Batch: {batch_name}')
        plt.xticks(all_ILDs)
        plt.tight_layout()
        pdf.savefig(plt.gcf())
        plt.close()

    
    # PDF is automatically closed when exiting the 'with' block

    # break


# %%
