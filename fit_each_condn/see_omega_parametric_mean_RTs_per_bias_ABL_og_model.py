# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.integrate import trapezoid
import random

# --- User config ---
batch_name = 'LED7'
# Use out_LED.csv as in diagnostics_multiple_gamma_omega_fit_at_once_but_parametric.py
og_df = pd.read_csv('../out_LED.csv')
ABLs_to_fit = [20, 40, 60]
ILDs_to_fit = [1, 2, 4, 8, 16, -1, -2, -4, -8, -16]
K_max = 10

gamma_all_animals = {}
omega_all_animals = {}

# Use the same animal list as in the reference script
all_animals = og_df['animal'].unique()

for animal_id in [103]:
# for animal_id in all_animals:
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

    # --- Load parametric fit (NEW FILE FORMAT) ---
    saved_vbmc_file = f'vbmc_mutiple_gama_omega_at_once_but_parametric_batch_{batch_name}_animal_{animal_id}_params_like_og_MODEL_bias_per_ABL.pkl'

    if not os.path.exists(saved_vbmc_file):
        print(f"Skipping {animal_id} as VBMC file {saved_vbmc_file} does not exist.")
        continue
    with open(saved_vbmc_file, 'rb') as f:
        vp = pickle.load(f)

    vp = vp.vp

    # --- PDF output block (if needed, insert here) ---
    # pdf_filename = f'{batch_name}_{animal_id}_diagnostics_multiple_gama_omega_at_once_but_parametric_batch_{batch_name}_animal_{animal_id}_params_like_og_MODEL_bias_per_ABL.pdf'
    # with PdfPages(pdf_filename) as pdf:
    #     ...

    # --- Extract parameter means (NEW ORDER) ---
    vp_samples = vp.sample(int(1e5))[0]

    theta_E_samples = vp_samples[:, 0]
    rate_lambda_samples = vp_samples[:, 1]
    ILD_bias_20_samples = vp_samples[:, 2]
    ILD_bias_40_samples = vp_samples[:, 3]
    ILD_bias_60_samples = vp_samples[:, 4]
    o_ratio_scale_20_samples = vp_samples[:, 5]
    o_ratio_scale_40_samples = vp_samples[:, 6]
    o_ratio_scale_60_samples = vp_samples[:, 7]
    norm_factor_samples = vp_samples[:, 8]

    w_samples = vp_samples[:, 9]
    t_E_aff_samples = vp_samples[:, 10]
    del_go_samples = vp_samples[:, 11]

    # Means (if needed elsewhere)
    theta_E = theta_E_samples.mean()
    rate_lambda = rate_lambda_samples.mean()
    ILD_bias_20 = ILD_bias_20_samples.mean()
    ILD_bias_40 = ILD_bias_40_samples.mean()
    ILD_bias_60 = ILD_bias_60_samples.mean()
    o_ratio_scale_20 = o_ratio_scale_20_samples.mean()
    o_ratio_scale_40 = o_ratio_scale_40_samples.mean()
    o_ratio_scale_60 = o_ratio_scale_60_samples.mean()
    norm_factor = norm_factor_samples.mean()
    w = w_samples.mean()
    t_E_aff = t_E_aff_samples.mean()
    del_go = del_go_samples.mean()


    # TEMP: What if bias was zero
    # ILD_bias_20 = 0
    # ILD_bias_40 = 0
    # ILD_bias_60 = 0


    # --- Define omega function ---
    def get_omega(ABL, ILD):
        if ABL == 20:
            return o_ratio_scale_20 * np.cosh(rate_lambda * (ILD - ILD_bias_20)) / np.cosh(rate_lambda * norm_factor * (ILD - ILD_bias_20))
        elif ABL == 40:
            return o_ratio_scale_40 * np.cosh(rate_lambda * (ILD - ILD_bias_40)) / np.cosh(rate_lambda * norm_factor * (ILD - ILD_bias_40))
        elif ABL == 60:
            return o_ratio_scale_60 * np.cosh(rate_lambda * (ILD - ILD_bias_60)) / np.cosh(rate_lambda * norm_factor * (ILD - ILD_bias_60))
        else:
            return None

    # --- Mean RT (data) ---
    mean_rts = {ABL: [] for ABL in ABLs_to_fit}
    for ABL in ABLs_to_fit:
        for ILD in ILDs_to_fit:
            data_a_i = df_led_off_valid_trials_cond_filtered[
                (df_led_off_valid_trials_cond_filtered['ABL'] == ABL) &
                (df_led_off_valid_trials_cond_filtered['ILD'] == ILD)
            ]
            data_a_i_rt = data_a_i['timed_fix'] - data_a_i['intended_fix']
            mean_rts[ABL].append(data_a_i_rt.mean() if len(data_a_i_rt) > 0 else np.nan)

    # --- Mean RT (model/theory) ---
    # Load proactive params from animal fit
    pkl_animal = f'/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/results_{batch_name}_animal_{animal_id}.pkl'
    if not os.path.exists(pkl_animal):
        print(f"Skipping {animal_id} as animal fit file {pkl_animal} does not exist.")
        continue
    with open(pkl_animal, 'rb') as f:
        fit_results_data = pickle.load(f)
    abort_samples = fit_results_data['vbmc_aborts_results']
    V_A = np.mean(abort_samples['V_A_samples'])
    theta_A = np.mean(abort_samples['theta_A_samples'])
    t_A_aff = np.mean(abort_samples['t_A_aff_samp'])

    # Prepare for theory RTD calculation
    N_theory = int(1e3)
    t_pts = np.arange(-1, 2, 0.001)
    t_stim_samples = df_led_off_valid_trials_cond_filtered['intended_fix'].sample(N_theory, replace=True).values
    P_A_samples = np.zeros((N_theory, len(t_pts)))
    t_trunc = 0.3
    for idx, t_stim in enumerate(t_stim_samples):
        t_shifted = t_pts + t_stim
        mask = t_shifted > t_trunc
        vals = np.zeros_like(t_pts)
        if np.any(mask):
            # rho_A_t_VEC_fn must be imported from led_off_gamma_omega_pdf_utils.py
            from led_off_gamma_omega_pdf_utils import rho_A_t_VEC_fn
            vals[mask] = rho_A_t_VEC_fn(t_shifted[mask] - t_A_aff, V_A, theta_A)
        P_A_samples[idx, :] = vals
    P_A_mean = np.mean(P_A_samples, axis=0)
    area = trapezoid(P_A_mean, t_pts)
    if area != 0:
        P_A_mean = P_A_mean / area
    C_A_mean = cumtrapz(P_A_mean, t_pts, initial=0)

    # --- Model RT calculation ---
    def model_mean_rt(ABL, ILD):
        # Get gamma, omega, w for this ABL
        if ABL == 20:
            gamma = theta_E * np.tanh(rate_lambda * (ILD - ILD_bias_20))
            omega = o_ratio_scale_20 * ( np.cosh(rate_lambda * (ILD - ILD_bias_20))) / ( np.cosh(rate_lambda * norm_factor * (ILD - ILD_bias_20)))
        elif ABL == 40:
            gamma = theta_E * np.tanh(rate_lambda * (ILD - ILD_bias_40))
            omega = o_ratio_scale_40 * ( np.cosh(rate_lambda * (ILD - ILD_bias_40))) / ( np.cosh(rate_lambda * norm_factor * (ILD - ILD_bias_40)))
        elif ABL == 60:
            gamma = theta_E * np.tanh(rate_lambda * (ILD - ILD_bias_60))
            omega = o_ratio_scale_60 * ( np.cosh(rate_lambda * (ILD - ILD_bias_60))) / ( np.cosh(rate_lambda * norm_factor * (ILD - ILD_bias_60)))
        else:
            return np.nan
        bound = 1
        # up_or_down_RTs_fit_OPTIM_V_A_change_gamma_omega_with_w_PA_CA_fn must be imported
        from led_off_gamma_omega_pdf_utils import up_or_down_RTs_fit_OPTIM_V_A_change_gamma_omega_with_w_PA_CA_fn
        up_mean = np.array([
            up_or_down_RTs_fit_OPTIM_V_A_change_gamma_omega_with_w_PA_CA_fn(
                t, P_A_mean[idx], C_A_mean[idx],
                gamma, omega, 0, t_E_aff, del_go, bound, w, 10
            ) for idx, t in enumerate(t_pts)
        ])
        down_mean = np.array([
            up_or_down_RTs_fit_OPTIM_V_A_change_gamma_omega_with_w_PA_CA_fn(
                t, P_A_mean[idx], C_A_mean[idx],
                gamma, omega, 0, t_E_aff, del_go, -bound, w, 10
            ) for idx, t in enumerate(t_pts)
        ])
        mask_0_1 = (t_pts >= 0) & (t_pts <= 1)
        t_pts_0_1 = t_pts[mask_0_1]
        up_plus_down = up_mean + down_mean
        up_plus_down_masked = up_plus_down[mask_0_1]
        area_masked = trapezoid(up_plus_down_masked, t_pts_0_1)
        if area_masked != 0:
            up_plus_down_mean = up_plus_down_masked / area_masked
        else:
            up_plus_down_mean = up_plus_down_masked
        # mean RT from model PDF
        mean_rt = np.sum(t_pts_0_1 * up_plus_down_mean) / np.sum(up_plus_down_mean)
        return mean_rt

    model_rts = {ABL: [] for ABL in ABLs_to_fit}
    for ABL in ABLs_to_fit:
        for ILD in ILDs_to_fit:
            model_rts[ABL].append(model_mean_rt(ABL, ILD))

    # --- Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False)

# --- Left: mean RT (data & model) vs ILD, color per ABL ---
handles = []
labels = []
for i, ABL in enumerate(ABLs_to_fit):
    color = f'C{i}'
    # Data: dot marker
    h1 = axes[0].scatter(ILDs_to_fit, mean_rts[ABL], marker='o', color=color, alpha=0.8, label=None)
    # Model: cross marker
    h2 = axes[0].scatter(ILDs_to_fit, model_rts[ABL], marker='x', color=color, alpha=0.8, label=None)
    if i == 0:
        handles.extend([h1, h2])
        labels.extend(['Data', 'Model'])
axes[0].set_title('Mean RT vs ILD (Data & Model)')
axes[0].set_xlabel('ILD (dB)')
axes[0].set_ylabel('Mean RT (s)')
axes[0].legend(handles, labels)

# --- Right: omega vs ILD (continuous) ---
ILD_continuous = np.linspace(-16, 16, 100)
for ABL in ABLs_to_fit:
    omega_curve = [get_omega(ABL, ild) for ild in ILD_continuous]
    axes[1].plot(ILD_continuous, omega_curve, label=f'ABL={ABL}')
# Overlay the discrete points as scatter
for ABL in ABLs_to_fit:
    omegas = [get_omega(ABL, ILD) for ILD in ILDs_to_fit]
    axes[1].scatter(ILDs_to_fit, omegas, marker='o', s=40, alpha=0.6)
axes[1].set_title('Omega vs ILD')
axes[1].set_xlabel('ILD (dB)')
axes[1].set_ylabel('Omega')
axes[1].legend()

plt.suptitle(f'Animal: {animal_id}, Batch: {batch_name}')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f'see_omega_parametric_mean_RTs_{batch_name}_animal_{animal_id}_og_model_bias_per_ABL.png')
# plt.savefig(f'see_omega_parametric_mean_RTs_{batch_name}_animal_{animal_id}_og_model_bias_per_ABL_bias_zero.png')
plt.close()
print('Done.')
