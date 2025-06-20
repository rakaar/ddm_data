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
# %%
# vbmc.save(f'{batch_name}_{animal_id}_vbmc_mutiple_gama_omega_at_once_ILDs_1_2_4_8_16.pkl', overwrite=True)
batch_name = 'LED7'
og_df = pd.read_csv('../out_LED.csv')
all_animals = og_df['animal'].unique()
ABLs_to_fit = [20, 40, 60]
ILDs_to_fit = [1,2,4,8,16,-1,-2,-4,-8,-16]
K_max = 10
gamma_all_animals = {}
omega_all_animals = {}

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

    # saved_vbmc_file = f'{batch_name}_{animal_id}_vbmc_mutiple_gama_omega_at_once_ILDs_1_2_4_8_16.pkl'

    # saved_vbmc_file = 'vbmc_mutiple_gama_omega_at_once_but_parametric.pkl'
    saved_vbmc_file = f'vbmc_mutiple_gama_omega_at_once_but_parametric_batch_{batch_name}_animal_{animal_id}_BETTER_BOUNDS_V2.pkl'

    if not os.path.exists(saved_vbmc_file):
        print(f"Skipping {animal_id} as VBMC file {saved_vbmc_file} does not exist.")
        continue
    with open(saved_vbmc_file, 'rb') as f:
        vp = pickle.load(f)

    vp = vp.vp

    # --- PDF output block ---
    pdf_filename = f'{batch_name}_{animal_id}_diagnostics_multiple_gama_omega_at_once_but_parametric_batch_{batch_name}_animal_{animal_id}_BETTER_BOUNDS_V2.pdf'
    with PdfPages(pdf_filename) as pdf:
        # Title/cover page
        fig_cover = plt.figure(figsize=(8, 4))
        plt.axis('off')
        plt.title('Diagnostics Summary', fontsize=18, pad=30)
        plt.text(0.5, 0.7, f'Animal: {animal_id}', ha='center', va='center', fontsize=16)
        plt.text(0.5, 0.5, f'Batch: {batch_name}', ha='center', va='center', fontsize=16)
        pdf.savefig(fig_cover)
        plt.close(fig_cover)

        # --- All plotting code for this animal goes below. After each plot, use pdf.savefig(plt.gcf()) and plt.close(). ---


    vp_samples = vp.sample(int(1e5))[0]

    # Extract new parameter samples according to the new order
    g_tanh_scale_20_samples = vp_samples[:, 0]
    g_ild_scale_20_samples = vp_samples[:, 1]
    g_ild_offset_20_samples = vp_samples[:, 2]
    o_ratio_scale_20_samples = vp_samples[:, 3]
    o_ild_scale_20_samples = vp_samples[:, 4]
    o_ild_offset_20_samples = vp_samples[:, 5]
    norm_factor_20_samples = vp_samples[:, 6]

    g_tanh_scale_40_samples = vp_samples[:, 7]
    g_ild_scale_40_samples = vp_samples[:, 8]
    g_ild_offset_40_samples = vp_samples[:, 9]
    o_ratio_scale_40_samples = vp_samples[:, 10]
    o_ild_scale_40_samples = vp_samples[:, 11]
    o_ild_offset_40_samples = vp_samples[:, 12]
    norm_factor_40_samples = vp_samples[:, 13]

    g_tanh_scale_60_samples = vp_samples[:, 14]
    g_ild_scale_60_samples = vp_samples[:, 15]
    g_ild_offset_60_samples = vp_samples[:, 16]
    o_ratio_scale_60_samples = vp_samples[:, 17]
    o_ild_scale_60_samples = vp_samples[:, 18]
    o_ild_offset_60_samples = vp_samples[:, 19]
    norm_factor_60_samples = vp_samples[:, 20]

    w_20_samples = vp_samples[:, 21]
    w_40_samples = vp_samples[:, 22]
    w_60_samples = vp_samples[:, 23]
    t_E_aff_samples = vp_samples[:, 24]
    del_go_samples = vp_samples[:, 25]

    # Means (if needed elsewhere)
    g_tanh_scale_20 = g_tanh_scale_20_samples.mean()
    g_ild_scale_20 = g_ild_scale_20_samples.mean()
    g_ild_offset_20 = g_ild_offset_20_samples.mean()
    o_ratio_scale_20 = o_ratio_scale_20_samples.mean()
    o_ild_scale_20 = o_ild_scale_20_samples.mean()
    o_ild_offset_20 = o_ild_offset_20_samples.mean()
    norm_factor_20 = norm_factor_20_samples.mean()

    g_tanh_scale_40 = g_tanh_scale_40_samples.mean()
    g_ild_scale_40 = g_ild_scale_40_samples.mean()
    g_ild_offset_40 = g_ild_offset_40_samples.mean()
    o_ratio_scale_40 = o_ratio_scale_40_samples.mean()
    o_ild_scale_40 = o_ild_scale_40_samples.mean()
    o_ild_offset_40 = o_ild_offset_40_samples.mean()
    norm_factor_40 = norm_factor_40_samples.mean()

    g_tanh_scale_60 = g_tanh_scale_60_samples.mean()
    g_ild_scale_60 = g_ild_scale_60_samples.mean()
    g_ild_offset_60 = g_ild_offset_60_samples.mean()
    o_ratio_scale_60 = o_ratio_scale_60_samples.mean()
    o_ild_scale_60 = o_ild_scale_60_samples.mean()
    o_ild_offset_60 = o_ild_offset_60_samples.mean()
    norm_factor_60 = norm_factor_60_samples.mean()

    w_20 = w_20_samples.mean()
    w_40 = w_40_samples.mean()
    w_60 = w_60_samples.mean()
    t_E_aff = t_E_aff_samples.mean()
    del_go = del_go_samples.mean()

    # plot the corner plot
    corner_samples = np.vstack([
        g_tanh_scale_20_samples, g_ild_scale_20_samples, g_ild_offset_20_samples,
        o_ratio_scale_20_samples, o_ild_scale_20_samples, o_ild_offset_20_samples, norm_factor_20_samples,
        g_tanh_scale_40_samples, g_ild_scale_40_samples, g_ild_offset_40_samples,
        o_ratio_scale_40_samples, o_ild_scale_40_samples, o_ild_offset_40_samples, norm_factor_40_samples,
        g_tanh_scale_60_samples, g_ild_scale_60_samples, g_ild_offset_60_samples,
        o_ratio_scale_60_samples, o_ild_scale_60_samples, o_ild_offset_60_samples, norm_factor_60_samples,
        w_20_samples, w_40_samples, w_60_samples, t_E_aff_samples, del_go_samples
    ]).T
    percentiles = np.percentile(corner_samples, [0, 100], axis=0)
    _ranges = [(percentiles[0, i], percentiles[1, i]) for i in np.arange(corner_samples.shape[1])]
    param_labels = [
        'g_tanh_scale_20', 'g_ild_scale_20', 'g_ild_offset_20',
        'o_ratio_scale_20', 'o_ild_scale_20', 'o_ild_offset_20', 'norm_factor_20',
        'g_tanh_scale_40', 'g_ild_scale_40', 'g_ild_offset_40',
        'o_ratio_scale_40', 'o_ild_scale_40', 'o_ild_offset_40', 'norm_factor_40',
        'g_tanh_scale_60', 'g_ild_scale_60', 'g_ild_offset_60',
        'o_ratio_scale_60', 'o_ild_scale_60', 'o_ild_offset_60', 'norm_factor_60',
        'w_20', 'w_40', 'w_60', 't_E_aff', 'del_go'
    ]

    fig_corner = corner.corner(
        corner_samples,
        labels=param_labels,
        show_titles=True,
        quantiles=[0.025, 0.5, 0.975],
        range=_ranges,
        title_fmt=".4f"
    )
    plt.suptitle(f'Corner Plot\nAnimal: {animal_id}, Batch: {batch_name}', fontsize=14)
    pdf.savefig(plt.gcf())
    plt.close()

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
            gamma = None
            omega = None
            if ABL == 20:
                gamma = g_tanh_scale_20 * np.tanh(g_ild_scale_20 * (ILD - g_ild_offset_20))
                omega = o_ratio_scale_20 * np.cosh(o_ild_scale_20 * (ILD - o_ild_offset_20)) / np.cosh(o_ild_scale_20 * norm_factor_20 * (ILD - o_ild_offset_20))
                w = w_20
            elif ABL == 40:
                gamma = g_tanh_scale_40 * np.tanh(g_ild_scale_40 * (ILD - g_ild_offset_40))
                omega = o_ratio_scale_40 * np.cosh(o_ild_scale_40 * (ILD - o_ild_offset_40)) / np.cosh(o_ild_scale_40 * norm_factor_40 * (ILD - o_ild_offset_40))
                w = w_40
            elif ABL == 60:
                gamma = g_tanh_scale_60 * np.tanh(g_ild_scale_60 * (ILD - g_ild_offset_60))
                omega = o_ratio_scale_60 * np.cosh(o_ild_scale_60 * (ILD - o_ild_offset_60)) / np.cosh(o_ild_scale_60 * norm_factor_60 * (ILD - o_ild_offset_60))
                w = w_60

            if gamma is None or omega is None:
                print(f"Skipping ABL={ABL}, ILD={ILD} (no gamma/omega)")
                continue
            bound = 1
            # print params and break
            print(f'ABL={ABL}, ILD={ILD}')
            print(f'gamma={gamma}, omega={omega}')
            print(f'P_A_mean shape: {P_A_mean.shape}')
            print(f'C_A_mean shape: {C_A_mean.shape}')
            print(f't_pts shape: {t_pts.shape}')
            print(f't_E_aff={t_E_aff}, del_go={del_go}, bound={bound}, w={w}, K_max={K_max}')
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
        for i_ILD, ILD in enumerate(ILDs_to_fit):
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

  
    def get_gamma(ABL, ILD):
        if ABL == 20:
            return g_tanh_scale_20 * np.tanh(g_ild_scale_20 * (ILD - g_ild_offset_20))
        elif ABL == 40:
            return g_tanh_scale_40 * np.tanh(g_ild_scale_40 * (ILD - g_ild_offset_40))
        elif ABL == 60:
            return g_tanh_scale_60 * np.tanh(g_ild_scale_60 * (ILD - g_ild_offset_60))
        else:
            return None

    def get_omega(ABL, ILD):
        if ABL == 20:
            return o_ratio_scale_20 * np.cosh(o_ild_scale_20 * (ILD - o_ild_offset_20)) / np.cosh(o_ild_scale_20 * norm_factor_20 * (ILD - o_ild_offset_20))
        elif ABL == 40:
            return o_ratio_scale_40 * np.cosh(o_ild_scale_40 * (ILD - o_ild_offset_40)) / np.cosh(o_ild_scale_40 * norm_factor_40 * (ILD - o_ild_offset_40))
        elif ABL == 60:
            return o_ratio_scale_60 * np.cosh(o_ild_scale_60 * (ILD - o_ild_offset_60)) / np.cosh(o_ild_scale_60 * norm_factor_60 * (ILD - o_ild_offset_60))
        else:
            return None


    # --- gamma plot ---
    plt.figure(figsize=(4, 3))
    all_ILDs = sorted(set(ILD for ILD in ILDs_to_fit))
    for ABL in ABLs_to_fit:
        gammas = []
        ILDs_plot = []
        for ILD in ILDs_to_fit:
            gamma = get_gamma(ABL, ILD)
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
            omega = get_omega(ABL, ILD)
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

#     # -- w plot ---

