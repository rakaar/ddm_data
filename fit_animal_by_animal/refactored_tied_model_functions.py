import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Copied from time_vary_and_norm_simulators.py
def simulate_psiam_tied_rate_norm(V_A, theta_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_stim, \
                                  t_A_aff, t_E_aff, del_go, rate_norm_l, dt):
    AI = 0; DV = Z_E; t = t_A_aff; dB = dt**0.5
    
    chi = 17.37; q_e = 1
    theta = theta_E * q_e
    lambda_ABL_term = (10 ** (rate_lambda * (1 - rate_norm_l) * ABL / 20))
    lambda_ILD_arg = rate_lambda * ILD / chi
    lambda_ILD_L_arg = rate_lambda * rate_norm_l * ILD / chi
    mu = (1/T_0) * lambda_ABL_term * (np.sinh(lambda_ILD_arg) / np.cosh(lambda_ILD_L_arg)) 
    sigma = np.sqrt( (1/T_0) * lambda_ABL_term * ( np.cosh(lambda_ILD_arg) / np.cosh(lambda_ILD_L_arg) ) )

    is_act = 0
    while True:
        AI += V_A*dt + np.random.normal(0, dB)

        if t > t_stim + t_E_aff:
            DV += mu*dt + sigma*np.random.normal(0, dB)
        
        
        t += dt
        
        if DV >= theta:
            choice = +1; RT = t
            break
        elif DV <= -theta:
            choice = -1; RT = t
            break
        
        if AI >= theta_A:
            both_AI_hit_and_EA_hit = 0 # see if both AI and EA hit 
            is_act = 1
            AI_hit_time = t
            while t <= (AI_hit_time + del_go):
                if t > t_stim + t_E_aff: 
                    DV += mu*dt + sigma*np.random.normal(0, dB)
                    if DV >= theta:
                        DV = theta
                        both_AI_hit_and_EA_hit = 1
                        break
                    elif DV <= -theta:
                        DV = -theta
                        both_AI_hit_and_EA_hit = -1
                        break
                t += dt
            
            break
        
        
    if is_act == 1:
        RT = AI_hit_time
        if both_AI_hit_and_EA_hit != 0:
            choice = both_AI_hit_and_EA_hit
        else:
            randomly_choose_up = np.random.rand() >= 0.5
            if randomly_choose_up:
                choice = 1
            else:
                choice = -1
    
    return choice, RT, is_act

# Refactored function
def psiam_tied_data_gen_wrapper_rate_norm_fn_refactored(V_A, theta_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_A_aff, t_E_aff_slow, t_E_aff_fast, del_go, \
                                t_stim, rate_norm_l,iter_num, N_print, dt):

    if abs(ILD) < 7:
        t_E_aff = t_E_aff_slow
    else:
        t_E_aff = t_E_aff_fast

    if iter_num % N_print == 0:
        print(f'os id: {os.getpid()}, In iter_num: {iter_num}, ABL: {ABL}, ILD: {ILD}, t_stim: {t_stim}')

    choice, rt, is_act = simulate_psiam_tied_rate_norm(V_A, theta_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, \
                                                       t_stim, t_A_aff, t_E_aff, del_go, rate_norm_l, dt)
    return {'choice': choice, 'rt': rt, 'is_act': is_act ,'ABL': ABL, 'ILD': ILD, 't_stim': t_stim}


# Refactored function
def plot_rt_distributions_refactored(sim_df_1, data_df_1, ILD_arr, ABL_arr, t_pts, P_A_mean, C_A_mean, 
                          t_stim_samples, V_A, theta_A, t_A_aff, rate_lambda, T_0, theta_E, Z_E, t_E_aff_slow, t_E_aff_fast, del_go, 
                          phi_params_obj, rate_norm_l, is_norm, is_time_vary, K_max, T_trunc,
                          cum_pro_and_reactive_time_vary_fn, up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn,
                          animal, pdf, model_name="Vanilla Tied"):
    """
    Plot RT distributions for all ABL-ILD combinations.
    """
    bw = 0.02
    bins = np.arange(0, 1, bw)
    bin_centers = bins[:-1] + (0.5 * bw)
    
    n_rows = len(ILD_arr)
    n_cols = len(ABL_arr)
    theory_results_up_and_down = {}  # Dictionary to store the theory results
    theory_time_axis = None
    
    fig_rtd, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharey='row')
    
    for i_idx, ILD in enumerate(ILD_arr):
        for a_idx, ABL in enumerate(ABL_arr):
            ax = axs[i_idx, a_idx] if n_rows > 1 else axs[a_idx]

            if abs(ILD) < 7:
                t_E_aff = t_E_aff_slow
            else:
                t_E_aff = t_E_aff_fast
            
            # Filter data for current ABL-ILD combination
            sim_df_1_ABL_ILD = sim_df_1[(sim_df_1['ABL'] == ABL) & (sim_df_1['ILD'] == ILD)]
            data_df_1_ABL_ILD = data_df_1[(data_df_1['ABL'] == ABL) & (data_df_1['ILD'] == ILD)]
            
            # Split by choice direction
            sim_up = sim_df_1_ABL_ILD[sim_df_1_ABL_ILD['choice'] == 1]
            sim_down = sim_df_1_ABL_ILD[sim_df_1_ABL_ILD['choice'] == -1]
            data_up = data_df_1_ABL_ILD[data_df_1_ABL_ILD['choice'] == 1]
            data_down = data_df_1_ABL_ILD[data_df_1_ABL_ILD['choice'] == -1]
            
            # Calculate RTs relative to stimulus onset
            sim_up_rt = sim_up['rt'] - sim_up['t_stim']
            sim_down_rt = sim_down['rt'] - sim_down['t_stim']
            data_up_rt = data_up['rt'] - data_up['t_stim']
            data_down_rt = data_down['rt'] - data_down['t_stim']
            
            # Create histograms
            sim_up_hist, _ = np.histogram(sim_up_rt, bins=bins, density=True)
            sim_down_hist, _ = np.histogram(sim_down_rt, bins=bins, density=True)
            data_up_hist, _ = np.histogram(data_up_rt, bins=bins, density=True)
            data_down_hist, _ = np.histogram(data_down_rt, bins=bins, density=True)
            
            # Normalize histograms by proportion of trials
            sim_up_hist *= len(sim_up) / len(sim_df_1_ABL_ILD) if len(sim_df_1_ABL_ILD) else 0
            sim_down_hist *= len(sim_down) / len(sim_df_1_ABL_ILD) if len(sim_df_1_ABL_ILD) else 0
            data_up_hist *= len(data_up) / len(data_df_1_ABL_ILD) if len(data_df_1_ABL_ILD) else 0
            data_down_hist *= len(data_down) / len(data_df_1_ABL_ILD) if len(data_df_1_ABL_ILD) else 0
            
            # Calculate theory curves with truncation factor
            trunc_fac_samples = np.zeros((len(t_stim_samples)))
            for idx, t_stim in enumerate(t_stim_samples):
                trunc_fac_samples[idx] = cum_pro_and_reactive_time_vary_fn(
                                t_stim + 1, T_trunc,
                                V_A, theta_A, t_A_aff,
                                t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_E_aff,
                                phi_params_obj, rate_norm_l, 
                                is_norm, is_time_vary, K_max) \
                                - \
                                cum_pro_and_reactive_time_vary_fn(
                                t_stim, T_trunc,
                                V_A, theta_A, t_A_aff,
                                t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_E_aff,
                                phi_params_obj, rate_norm_l, 
                                is_norm, is_time_vary, K_max) + 1e-10
            trunc_factor = np.mean(trunc_fac_samples)
            
            up_mean = np.array([up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn(
                        t, 1,
                        P_A_mean[i], C_A_mean[i],
                        ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go,
                        phi_params_obj, rate_norm_l, 
                        is_norm, is_time_vary, K_max) for i, t in enumerate(t_pts)])
            down_mean = np.array([up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn(
                    t, -1,
                    P_A_mean[i], C_A_mean[i],
                    ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go,
                    phi_params_obj, rate_norm_l, 
                    is_norm, is_time_vary, K_max) for i, t in enumerate(t_pts)])
            
            # Filter to relevant time window
            mask_0_1 = (t_pts >= 0) & (t_pts <= 1)
            t_pts_0_1 = t_pts[mask_0_1]
            up_mean_0_1 = up_mean[mask_0_1]
            down_mean_0_1 = down_mean[mask_0_1]
            
            # Normalize theory curves
            up_theory_mean_norm = up_mean_0_1 / trunc_factor
            down_theory_mean_norm = down_mean_0_1 / trunc_factor
            
            # Store for later use in tachometric curves
            theory_results_up_and_down[(ABL, ILD)] = {
                'up': up_theory_mean_norm.copy(),
                'down': down_theory_mean_norm.copy()
            }
            if theory_time_axis is None:
                theory_time_axis = t_pts_0_1.copy()
            
            # Plot
            ax.plot(bin_centers, data_up_hist, color='b', label='Data' if (i_idx == 0 and a_idx == 0) else "")
            ax.plot(bin_centers, -data_down_hist, color='b')
            ax.plot(bin_centers, sim_up_hist, color='r', label='Sim' if (i_idx == 0 and a_idx == 0) else "")
            ax.plot(bin_centers, -sim_down_hist, color='r')
            
            ax.plot(t_pts_0_1, up_theory_mean_norm, lw=3, alpha=0.2, color='g')
            ax.plot(t_pts_0_1, -down_theory_mean_norm, lw=3, alpha=0.2, color='g')
            
            # Compute fractions
            data_total = len(data_df_1_ABL_ILD)
            sim_total = len(sim_df_1_ABL_ILD)
            data_up_frac = len(data_up) / data_total if data_total else 0
            data_down_frac = len(data_down) / data_total if data_total else 0
            sim_up_frac = len(sim_up) / sim_total if sim_total else 0
            sim_down_frac = len(sim_down) / sim_total if sim_total else 0
            
            ax.set_title(
                f"ABL: {ABL}, ILD: {ILD}\n"
                f"Data,Sim: (+{data_up_frac:.2f},+{sim_up_frac:.2f}), "
                f"(-{data_down_frac:.2f},-{sim_down_frac:.2f})"
            )
            
            ax.axhline(0, color='k', linewidth=0.5)
            ax.set_xlim([0, 0.7])
            if a_idx == 0:
                ax.set_ylabel("Density (Up / Down flipped)")
            if i_idx == n_rows - 1:
                ax.set_xlabel("RT (s)")
    
    fig_rtd.tight_layout()
    fig_rtd.legend(loc='upper right')
    fig_rtd.suptitle(f'RT Distributions - {model_name} (Animal: {animal})', y=1.01)
    pdf.savefig(fig_rtd, bbox_inches='tight')
    plt.close(fig_rtd)
    
    return theory_results_up_and_down, theory_time_axis, bins, bin_centers
