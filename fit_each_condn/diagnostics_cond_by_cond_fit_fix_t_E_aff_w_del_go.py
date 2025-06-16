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
            pkl_file = f'vbmc_cond_by_cond_{batch_name}_{animal_id}_{ABL}_ILD_{ILD}_FIX_t_E_w_del_go.pkl'
            if not os.path.exists(pkl_file):
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
# vbmc.save(f'{batch_name}_{animal_id}_vbmc_mutiple_gama_omega_at_once_ILDs_1_2_4_8_16.pkl', overwrite=True)
batch_name = 'LED7'
og_df = pd.read_csv('../out_LED.csv')
all_animals = og_df['animal'].unique()
ABLs_to_fit = [20, 40, 60]
ILDs_to_fit = [1,2,4,8,16,-1,-2,-4,-8,-16]
# ILDs_to_fit = [1,4,16, -1, -4, -16]
K_max = 10
animal_id = 103
# %%
param_dict = get_param_means_by_ABL_ILD(batch_name, animal_id, ABLs_to_fit, ILDs_to_fit)
print(param_dict)

# %%
t_E_aff = 0.08411045805617333
w = 0.4594308751578441
del_go = 0.12118261009394682
# %%
# bounds
# omega_bounds = [0.1, 15]
# omega_plausible_bounds = [2, 12]

# t_E_aff_bounds = [0, 1]
# t_E_aff_plausible_bounds = [0.01, 0.2]

# w_bounds = [0.1, 0.9]
# w_plausible_bounds = [0.3, 0.7]

# del_go_bounds = [0.001, 0.2]
# del_go_plausible_bounds = [0.11, 0.15]

# if ILD > 0:
#     gamma_bounds = [-1, 5]
#     gamma_plausible_bounds = [0, 3]
# elif ILD < 0:
#     gamma_bounds = [-5, 1]
#     gamma_plausible_bounds = [-3, 0]

# %%
# for animal_id in all_animals:
for animal_id in [103]:

    animal_id = str(animal_id)
    
    # get the df
    df = og_df[ og_df['repeat_trial'].isin([0,2]) | og_df['repeat_trial'].isna() ]
    session_type = 7    
    df = df[ df['session_type'].isin([session_type]) ]
    training_level = 16
    df = df[ df['training_level'].isin([training_level]) ]
    t_stim_and_led_tuple = [(row['intended_fix'], row['intended_fix'] - row['LED_onset_time']) for _, row in df.iterrows()]

    df['choice'] = df['response_poke'].apply(lambda x: 1 if x == 3 else (-1 if x == 2 else random.choice([1, -1])))
    df['correct'] = (df['ILD'] * df['choice']).apply(lambda x: 1 if x > 0 else 0)

    print(f'animal filter: {animal_id}')
    df = df[df['animal'] == int(animal_id)]

    df_led_off = df[df['LED_trial'] == 0]
    df_led_off_valid_trials = df_led_off[df_led_off['success'].isin([1,-1])]
    df_led_off_valid_trials = df_led_off_valid_trials[df_led_off_valid_trials['timed_fix'] - df_led_off_valid_trials['intended_fix'] < 1]

    df_led_off_valid_trials_cond_filtered = df_led_off_valid_trials[
        (df_led_off_valid_trials['ABL'].isin(ABLs_to_fit)) & 
        (df_led_off_valid_trials['ILD'].isin(ILDs_to_fit))
    ]

    
    # --- PDF output block ---
    pdf_filename = f'{batch_name}_{animal_id}_diagnostics_cond_by_cond_FIX_t_E_aff_w_del_go.pdf'
    with PdfPages(pdf_filename) as pdf:
        fig_cover = plt.figure(figsize=(8, 4))
        plt.axis('off')
        plt.title('Diagnostics Summary', fontsize=18, pad=30)
        plt.text(0.5, 0.7, f'Animal: {animal_id}', ha='center', va='center', fontsize=16)
        plt.text(0.5, 0.5, f'Batch: {batch_name}', ha='center', va='center', fontsize=16)
        pdf.savefig(fig_cover)
        plt.close(fig_cover)

        

    # proactive params
    pkl_file = f'/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/results_{batch_name}_animal_{animal_id}.pkl'
    with open(pkl_file, 'rb') as f:
        fit_results_data = pickle.load(f)

    vbmc_aborts_param_keys_map = {
        'V_A_samples': 'V_A',
        'theta_A_samples': 'theta_A',
        't_A_aff_samp': 't_A_aff'
    }

    abort_keyname = "vbmc_aborts_results"
    if abort_keyname not in fit_results_data:
        raise Exception(f"No abort parameters found for batch {batch_name}, animal {animal_id}. Skipping.")
        
    abort_samples = fit_results_data[abort_keyname]
    abort_params = {}
    for param_samples_name, param_label in vbmc_aborts_param_keys_map.items():
        abort_params[param_label] = np.mean(abort_samples[param_samples_name])

    V_A = abort_params['V_A']
    theta_A = abort_params['theta_A']
    t_A_aff = abort_params['t_A_aff']

    # Diagnostics - RTD choice
    N_theory = int(1e3)
    t_pts = np.arange(-1, 2, 0.001)
    t_stim_samples = df_led_off_valid_trials_cond_filtered['intended_fix'].sample(N_theory, replace=True).values
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

    for ABL in ABLs_to_fit:
        for ILD in ILDs_to_fit:
            # get gamma, omega, t_E_aff, w, del_go from pkl file
            gamma = param_dict[(ABL, ILD)]['gamma']
            omega = param_dict[(ABL, ILD)]['omega']
        

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
            data_a_i = df_led_off_valid_trials_cond_filtered[
                (df_led_off_valid_trials_cond_filtered['ABL'] == ABL) &
                (df_led_off_valid_trials_cond_filtered['ILD'] == ILD)
            ]
            data_a_i_rt = data_a_i['timed_fix'] - data_a_i['intended_fix']
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
        df_abl = df_led_off_valid_trials_cond_filtered[df_led_off_valid_trials_cond_filtered['ABL'] == ABL]
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
            gamma = param_dict[(ABL, ILD)]['gamma']
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
            omega = param_dict[(ABL, ILD)]['omega']
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

    
    pdf.close()


# %%
gamma_vs_ILD_for_each_ABL = np.zeros((len(ABLs_to_fit), len(ILDs_to_fit)))
omega_vs_ILD_for_each_ABL = np.zeros((len(ABLs_to_fit), len(ILDs_to_fit)))

for a_idx, ABL in enumerate(ABLs_to_fit):
    for i_idx, ILD in enumerate(ILDs_to_fit):
        gamma_vs_ILD_for_each_ABL[a_idx, i_idx] = param_dict[(ABL, ILD)]['gamma']
        omega_vs_ILD_for_each_ABL[a_idx, i_idx] = param_dict[(ABL, ILD)]['omega']

gamma_vs_ILD_mean = np.mean(gamma_vs_ILD_for_each_ABL, axis=0)

# %%
plt.scatter(ILDs_to_fit, gamma_vs_ILD_mean)

# Fit gamma = theta * tanh(lambda * ILD / 17.37)
from scipy.optimize import curve_fit

def gamma_tanh(ILD, lambd, theta):
    return theta * np.tanh(lambd * ILD / 17.37)

ILDs_arr = np.array(ILDs_to_fit)
gamma_arr = np.array(gamma_vs_ILD_mean)

popt, pcov = curve_fit(gamma_tanh, ILDs_arr, gamma_arr, p0=[1, 1])
lambd_fit, theta_fit = popt

ILDs_fine = np.linspace(min(ILDs_to_fit), max(ILDs_to_fit), 200)
plt.plot(ILDs_fine, gamma_tanh(ILDs_fine, lambd_fit, theta_fit), 'r-', label=f'Fit: theta={theta_fit:.2f}, lambda={lambd_fit:.2f}')
plt.legend()
plt.xlabel('ILD (dB)')
plt.ylabel('gamma')
plt.title('gamma vs ILD (mean across ABLs) with fit')
plt.show()
print(f'Fitted parameters: theta = {theta_fit:.4f}, lambda = {lambd_fit:.4f}')

# %%
# %%
# --- Omega fit using global R0 and l ---
from scipy.optimize import curve_fit

def omega_formula(X, R0, ell):
    # X: stacked array with shape (2, N), X[0]=ABL, X[1]=ILD
    ABL = X[0]
    ILD = X[1]
    chi = 17.37
    theta = theta_fit  # from previous fit
    lambd = lambd_fit  # from previous fit
    num = np.cosh(lambd * ILD / chi)
    denom = np.cosh(lambd * ell * ILD / chi)
    return (R0 / theta**2) * 10**((1-ell)*ABL/20) * num / denom

# Prepare data for fitting
ABLs_grid, ILDs_grid = np.meshgrid(ABLs_to_fit, ILDs_to_fit, indexing='ij')
ABLs_flat = ABLs_grid.flatten()
ILDs_flat = ILDs_grid.flatten()
omega_flat = omega_vs_ILD_for_each_ABL.flatten()

Xdata = np.vstack([ABLs_flat, ILDs_flat])

# Initial guess: R0=1, ell=0.5
popt_omega, pcov_omega = curve_fit(omega_formula, Xdata, omega_flat, p0=[1, 0.5])
R0_fit, ell_fit = popt_omega
print(f'Fitted omega params: R0 = {R0_fit:.4f}, ell = {ell_fit:.4f}')

# Plot fit for each ABL
plt.figure(figsize=(6, 4))
colors = ['b', 'g', 'r']
for i, ABL in enumerate(ABLs_to_fit):
    idx = np.where(ABLs_flat == ABL)[0]
    ILDs = ILDs_flat[idx]
    omega_obs = omega_flat[idx]
    plt.scatter(ILDs, omega_obs, color=colors[i%3], label=f'ABL={ABL} data')
    ILDs_fine = np.linspace(min(ILDs_to_fit), max(ILDs_to_fit), 200)
    omega_fit_curve = omega_formula(np.vstack([np.full_like(ILDs_fine, ABL), ILDs_fine]), R0_fit, ell_fit)
    plt.plot(ILDs_fine, omega_fit_curve, color=colors[i%3], linestyle='--', label=f'ABL={ABL} fit')
plt.xlabel('ILD (dB)')
plt.ylabel('omega')
plt.title('omega vs ILD for each ABL with fit')
# plt.legend()
plt.tight_layout()
plt.show()