import matplotlib.pyplot as plt
import numpy as np
import corner
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.integrate import cumulative_trapezoid as cumtrapz

def save_posterior_summary_page(pdf_pages: PdfPages, title: str, posterior_means: pd.Series, 
                                param_labels: dict, vbmc_results: dict, extra_text: str = '') -> None:
    """Creates and saves a text-based summary page for a model's fit results to a PDF."""
    fig = plt.figure(figsize=(8.27, 11.69)) # A4 size
    plt.suptitle(title, fontsize=14, fontweight='bold')
    
    # Format parameter means
    param_text = "Posterior Means:\n"
    for name, value in posterior_means.items():
        label = param_labels.get(name, name) # Use label if available, else use name
        param_text += f"{label}: {value:.4f}\n"
    
    # Format VBMC results
    vbmc_text = "\nVBMC Results:\n"
    vbmc_text += f"ELBO: {vbmc_results['elbo']:.4f} +/- {vbmc_results['elbo_sd']:.4f}\n"
    vbmc_text += f"Log Likelihood (at mean): {vbmc_results['loglike']:.4f}\n"
    vbmc_text += f"Message: {vbmc_results['message']}\n"

    full_text = param_text + vbmc_text
    if extra_text:
        full_text += "\n" + extra_text
        
    plt.text(0.05, 0.95, full_text, va='top', ha='left', fontsize=10, family='monospace', wrap=True)
    plt.axis('off')
    pdf_pages.savefig(fig)
    plt.close(fig) # Close the figure to free memory


def save_corner_plot(pdf_pages: PdfPages, samples: np.ndarray, param_labels: list, 
                     title: str, truths: list = None, quantiles: list = [0.16, 0.5, 0.84],
                     show_titles: bool = True, **corner_kwargs) -> None:
    """Generates and saves a corner plot of posterior samples to a PDF."""
    # Check if samples array is not empty or None
    if samples is None or samples.shape[0] == 0:
        print(f"Warning: Skipping corner plot '{title}' due to empty samples.")
        # Optionally, add a placeholder page to the PDF?
        fig = plt.figure(figsize=(8.27, 11.69))
        plt.text(0.5, 0.5, f"Corner plot for '{title}' skipped:\nNo samples provided.", 
                 ha='center', va='center', fontsize=12, color='red')
        plt.axis('off')
        pdf_pages.savefig(fig)
        plt.close(fig)
        return

    fig = corner.corner(samples, 
                        labels=param_labels, 
                        truths=truths, 
                        quantiles=quantiles,
                        show_titles=show_titles,
                        **corner_kwargs)
    fig.suptitle(title, fontsize=16)
    pdf_pages.savefig(fig)
    plt.close(fig) # Close the figure to free memory


def plot_abort_diagnostic(pdf_pages: PdfPages, df_aborts_animal: pd.DataFrame, 
                          df_valid_and_aborts: pd.DataFrame, N_theory: int,
                          V_A: float, theta_A: float, t_A_aff: float, T_trunc: float, 
                          rho_A_t_VEC_fn, cum_A_t_fn, title: str) -> None:
    """Creates and saves the specific diagnostic plot comparing empirical and theoretical abort RTs."""
    t_pts = np.arange(0, 2, 0.001)
    pdf_samples = np.zeros((N_theory, len(t_pts)))

    t_stim_samples_df = df_valid_and_aborts.sample(n=N_theory, replace=True).copy()
    t_stim_samples = t_stim_samples_df['intended_fix'].values

    for i, t_stim in enumerate(t_stim_samples):
        t_stim_idx = np.searchsorted(t_pts, t_stim)
        proactive_trunc_idx = np.searchsorted(t_pts, T_trunc)
        pdf_samples[i, :proactive_trunc_idx] = 0
        pdf_samples[i, t_stim_idx:] = 0
        t_btn = t_pts[proactive_trunc_idx:t_stim_idx-1]
        pdf_samples[i, proactive_trunc_idx:t_stim_idx-1] = rho_A_t_VEC_fn(t_btn - t_A_aff, V_A, theta_A) / (1 - cum_A_t_fn(T_trunc - t_A_aff, V_A, theta_A))
    avg_pdf = np.mean(pdf_samples, axis=0)

    # Plotting
    fig_aborts_diag = plt.figure(figsize=(10, 5))
    bins = np.arange(0, 2, 0.01)

    # Get empirical abort RTs and filter by truncation time
    animal_abort_RT = df_aborts_animal['TotalFixTime'].dropna().values
    animal_abort_RT_trunc = animal_abort_RT[animal_abort_RT > T_trunc]

    # Plot empirical histogram (scaled by fraction of aborts after truncation)
    if len(animal_abort_RT_trunc) > 0:
        # Compute N_valid_and_trunc_aborts as in the reference code
        # Need df_all_trials_animal and df_before_trunc_animal
        if 'animal' in df_valid_and_aborts.columns:
            animal_id = df_aborts_animal['animal'].iloc[0] if len(df_aborts_animal) > 0 else None
            df_all_trials_animal = df_valid_and_aborts[df_valid_and_aborts['animal'] == animal_id]
        else:
            df_all_trials_animal = df_valid_and_aborts
        df_before_trunc_animal = df_aborts_animal[df_aborts_animal['TotalFixTime'] < T_trunc]
        N_valid_and_trunc_aborts = len(df_all_trials_animal) - len(df_before_trunc_animal)
        frac_aborts = len(animal_abort_RT_trunc) / N_valid_and_trunc_aborts if N_valid_and_trunc_aborts > 0 else 0
        aborts_hist, _ = np.histogram(animal_abort_RT_trunc, bins=bins, density=True)
        # Scale the histogram by frac_aborts
        plt.plot(bins[:-1], aborts_hist * frac_aborts, label='Data (Aborts > T_trunc)')
    else:
        # Add a note if no data to plot
        plt.text(0.5, 0.5, 'No empirical abort data > T_trunc', 
                 horizontalalignment='center', verticalalignment='center', 
                 transform=plt.gca().transAxes)

    # Plot theoretical PDF
    plt.plot(t_pts, avg_pdf, 'r-', lw=2, label='Theory (Abort Model)')

    plt.title(title)
    plt.xlabel('Reaction Time (s)')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.xlim([0, np.max(bins)]) # Limit x-axis to the bin range
    # Add a reasonable upper limit to y-axis if needed, e.g., based on max density
    if len(animal_abort_RT_trunc) > 0:
        max_density = np.max(aborts_hist * frac_aborts) if len(aborts_hist) > 0 else 1
        plt.ylim([0, max(np.max(avg_pdf), max_density) * 1.1])
    elif np.any(avg_pdf > 0):
         plt.ylim([0, np.max(avg_pdf) * 1.1])
    else:
        plt.ylim([0, 1]) # Default ylim if no data and no theory

    pdf_pages.savefig(fig_aborts_diag)
    plt.close(fig_aborts_diag)


def prepare_simulation_data(sim_results_df, df_valid_animal_less_than_1):
    """
    Prepares simulation and data dataframes for comparison and plotting.
    
    Args:
        sim_results_df: Raw simulation results dataframe
        df_valid_animal_less_than_1: Original animal data
        
    Returns:
        sim_df_1: Processed simulation dataframe
        data_df_1: Processed data dataframe 
    """
    # Helper function for determining correct trials
    def correct_func(row):
        if row['ILD'] == 0:
            return np.random.choice([0, 1])
        return 1 if row['ILD'] * row['choice'] > 0 else 0
    
    # Process simulation data
    sim_results_df_valid = sim_results_df[
        (sim_results_df['rt'] > sim_results_df['t_stim']) &
        (sim_results_df['rt'] - sim_results_df['t_stim'] < 1)
    ].copy()
    
    sim_results_df_valid.loc[:, 'correct'] = sim_results_df_valid.apply(correct_func, axis=1)
    
    # Process data dataframe
    if 'correct' in df_valid_animal_less_than_1.columns:
        df_valid_animal_less_than_1_drop = df_valid_animal_less_than_1.copy().drop(columns=['correct']).copy()
    else:
        df_valid_animal_less_than_1_drop = df_valid_animal_less_than_1.copy()
    
    df_valid_animal_less_than_1_drop.loc[:,'correct'] = df_valid_animal_less_than_1_drop.apply(correct_func, axis=1)
    df_valid_animal_less_than_1_renamed = df_valid_animal_less_than_1_drop.rename(columns = {
        'TotalFixTime': 'rt',
        'intended_fix': 't_stim',
    }).copy()
    
    sim_df_1 = sim_results_df_valid.copy()
    data_df_1 = df_valid_animal_less_than_1_renamed.copy()
    
    return sim_df_1, data_df_1


def calculate_theoretical_curves(df_valid_and_aborts, N_theory, t_pts, t_A_aff, V_A, theta_A, rho_A_t_fn):
    """
    Calculate theoretical P_A_mean and C_A_mean curves.
    
    Args:
        df_valid_and_aborts: Dataframe with valid trials and aborts
        N_theory: Number of samples for theoretical calculation
        t_pts: Time points for evaluation
        t_A_aff: Afferent time
        V_A: V_A parameter
        theta_A: theta_A parameter
        rho_A_t_fn: Function to compute rho_A_t
        
    Returns:
        P_A_mean: Mean probability
        C_A_mean: Cumulative mean probability
        t_stim_samples: Samples used for calculation
    """
    t_stim_samples = df_valid_and_aborts['intended_fix'].sample(N_theory, replace=True).values
    
    P_A_samples = np.zeros((N_theory, len(t_pts)))
    for idx, t_stim in enumerate(t_stim_samples):
        P_A_samples[idx, :] = [rho_A_t_fn(t + t_stim - t_A_aff, V_A, theta_A) for t in t_pts]
    
    P_A_mean = np.mean(P_A_samples, axis=0)
    C_A_mean = cumtrapz(P_A_mean, t_pts, initial=0)
    
    return P_A_mean, C_A_mean, t_stim_samples

# def calculate_theoretical_PA_CA_truncated_curves(df_valid_and_aborts, N_theory, t_pts, t_A_aff, V_A, theta_A, rho_A_t_fn):
#     """
#     Calculate theoretical P_A_mean and C_A_mean curves.
    
#     Args:
#         df_valid_and_aborts: Dataframe with valid trials and aborts
#         N_theory: Number of samples for theoretical calculation
#         t_pts: Time points for evaluation
#         t_A_aff: Afferent time
#         V_A: V_A parameter
#         theta_A: theta_A parameter
#         rho_A_t_fn: Function to compute rho_A_t
        
#     Returns:
#         P_A_mean: Mean probability
#         C_A_mean: Cumulative mean probability
#         t_stim_samples: Samples used for calculation
#     """
#     t_stim_samples = df_valid_and_aborts['intended_fix'].sample(N_theory, replace=True).values
    
#     P_A_samples = np.zeros((N_theory, len(t_pts)))
#     for idx, t_stim in enumerate(t_stim_samples):
#         P_A_samples[idx, :] = [rho_A_t_fn(t + t_stim - t_A_aff, V_A, theta_A) for t in t_pts]
    
#     P_A_mean = np.mean(P_A_samples, axis=0)
#     C_A_mean = cumtrapz(P_A_mean, t_pts, initial=0)
    
#     return P_A_mean, C_A_mean, t_stim_samples

def plot_rt_distributions(sim_df_1, data_df_1, ILD_arr, ABL_arr, t_pts, P_A_mean, C_A_mean, 
                          t_stim_samples, V_A, theta_A, t_A_aff, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go, 
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


def plot_tachometric_curves(sim_df_1, data_df_1, ILD_arr, ABL_arr, theory_results_up_and_down, 
                            theory_time_axis, bins, animal, pdf, model_name="Vanilla Tied"):
    """
    Plot tachometric curves for all ABL-ILD combinations.
    """
    def plot_tacho(df_1):
        df_1 = df_1.copy()
        df_1['RT_bin'] = pd.cut(df_1['rt'] - df_1['t_stim'], bins=bins, include_lowest=True)
        grouped = df_1.groupby('RT_bin', observed=False)['correct'].agg(['mean', 'count'])
        grouped['bin_mid'] = grouped.index.map(lambda x: x.mid)
        return grouped['bin_mid'], grouped['mean']
    
    n_rows = len(ILD_arr)
    n_cols = len(ABL_arr)
    
    fig_tacho, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharey='row')
    
    for i_idx, ILD in enumerate(ILD_arr):
        for a_idx, ABL in enumerate(ABL_arr):
            ax = axs[i_idx, a_idx] if n_rows > 1 else axs[a_idx]
            
            sim_df_1_ABL_ILD = sim_df_1[(sim_df_1['ABL'] == ABL) & (sim_df_1['ILD'] == ILD)]
            data_df_1_ABL_ILD = data_df_1[(data_df_1['ABL'] == ABL) & (data_df_1['ILD'] == ILD)]
            
            sim_tacho_x, sim_tacho_y = plot_tacho(sim_df_1_ABL_ILD)
            data_tacho_x, data_tacho_y = plot_tacho(data_df_1_ABL_ILD)
            
            # Plotting
            ax.plot(data_tacho_x, data_tacho_y, color='b', label='Data' if (i_idx == 0 and a_idx == 0) else "")
            ax.plot(sim_tacho_x, sim_tacho_y, color='r', label='Sim' if (i_idx == 0 and a_idx == 0) else "")
            
            up_theory_mean_norm = theory_results_up_and_down[(ABL, ILD)]['up']
            down_theory_mean_norm = theory_results_up_and_down[(ABL, ILD)]['down']
            t_pts_0_1 = theory_time_axis
            
            if ILD > 0:
                ax.plot(t_pts_0_1, up_theory_mean_norm/(up_theory_mean_norm + down_theory_mean_norm), 
                        color='g', lw=3, alpha=0.2)
            elif ILD < 0:
                ax.plot(t_pts_0_1, down_theory_mean_norm/(up_theory_mean_norm + down_theory_mean_norm), 
                        color='g', lw=3, alpha=0.2)
            else:
                ax.plot(t_pts_0_1, 0.5*np.ones_like(t_pts_0_1), color='g', lw=3, alpha=0.2)
            
            ax.set_ylim([0, 1])
            ax.set_xlim([0, 0.7])
            ax.set_title(f"ABL: {ABL}, ILD: {ILD}")
            if a_idx == 0:
                ax.set_ylabel("P(correct)")
            if i_idx == n_rows - 1:
                ax.set_xlabel("RT (s)")
    
    fig_tacho.tight_layout()
    fig_tacho.legend(loc='upper right')
    fig_tacho.suptitle(f'Tachometric Curves - {model_name} (Animal: {animal})', y=1.01)
    pdf.savefig(fig_tacho, bbox_inches='tight')
    plt.close(fig_tacho)


def plot_grand_summary(sim_df_1, data_df_1, ILD_arr, ABL_arr, bins, bin_centers, animal, pdf, model_name="Vanilla Tied"):
    """
    Plot grand summary (RTD, Psychometric, Tachometric).
    """
    def grand_rtd(df_1):
        df_1_rt = df_1['rt'] - df_1['t_stim']
        rt_hist, _ = np.histogram(df_1_rt, bins=bins, density=True)
        return rt_hist

    def plot_psycho(df_1):
        prob_choice_dict = {}
        
        all_ABL = np.sort(df_1['ABL'].unique())
        all_ILD = np.sort(ILD_arr)
        
        for abl in all_ABL:
            filtered_df = df_1[df_1['ABL'] == abl]
            prob_choice_dict[abl] = [
                sum(filtered_df[filtered_df['ILD'] == ild]['choice'] == 1) / len(filtered_df[filtered_df['ILD'] == ild])
                if len(filtered_df[filtered_df['ILD'] == ild]) > 0 else np.nan
                for ild in all_ILD
            ]
        
        return prob_choice_dict
    
    def plot_tacho(df_1):
        df_1 = df_1.copy()
        df_1['RT_bin'] = pd.cut(df_1['rt'] - df_1['t_stim'], bins=bins, include_lowest=True)
        grouped = df_1.groupby('RT_bin', observed=False)['correct'].agg(['mean', 'count'])
        grouped['bin_mid'] = grouped.index.map(lambda x: x.mid)
        return grouped['bin_mid'], grouped['mean']
    
    # Create grand summary plots
    fig_grand, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # === Grand RTD ===
    axes[0].plot(bin_centers, grand_rtd(data_df_1), color='b', label='data')
    axes[0].plot(bin_centers, grand_rtd(sim_df_1), color='r', label='sim')
    axes[0].legend()
    axes[0].set_xlabel('rt wrt stim')
    axes[0].set_ylabel('density')
    axes[0].set_title('Grand RTD')
    
    # === Grand Psychometric ===
    data_psycho = plot_psycho(data_df_1)
    sim_psycho = plot_psycho(sim_df_1)
    
    colors = [
        '#1f77b4',  # muted blue
        '#ff7f0e',  # safety orange
        '#2ca02c',  # muted green
        '#d62728',  # brick red
        '#9467bd',  # muted purple
        '#8dd3c7',  # pale teal
        '#fdb462',  # burnt orange
        '#bcbd22',  # golden yellow
        '#17becf',  # bright blue
        '#9edae5',  # pale blue-green
    ]  # Define colors for each ABL
    
    for i, ABL in enumerate(ABL_arr):
        axes[1].plot(ILD_arr, data_psycho[ABL], color=colors[i], label=f'data ABL={ABL}', marker='o', linestyle='None')
        axes[1].plot(ILD_arr, sim_psycho[ABL], color=colors[i], linestyle='-')
    
    axes[1].legend()
    axes[1].set_xlabel('ILD')
    axes[1].set_ylabel('P(right)')
    axes[1].set_title('Grand Psychometric')
    
    # === Grand Tacho ===
    data_tacho_x, data_tacho_y = plot_tacho(data_df_1)
    sim_tacho_x, sim_tacho_y = plot_tacho(sim_df_1)
    
    axes[2].plot(data_tacho_x, data_tacho_y, color='b', label='data')
    axes[2].plot(sim_tacho_x, sim_tacho_y, color='r', label='sim')
    axes[2].legend()
    axes[2].set_xlabel('rt wrt stim')
    axes[2].set_ylabel('acc')
    axes[2].set_title('Grand Tacho')
    axes[2].set_ylim(0.5, 1)
    
    fig_grand.tight_layout()
    fig_grand.suptitle(f'Grand Summary Plots - {model_name} (Animal: {animal})', y=1.03)
    pdf.savefig(fig_grand, bbox_inches='tight')
    plt.close(fig_grand)


def render_df_to_pdf(df, title, pdf):
    """
    Renders a DataFrame as a formatted table in the PDF document.
    
    Args:
        df: DataFrame to display
        title: Title for the table page
        pdf: PdfPages object to save to
    """
    fig, ax = plt.subplots(figsize=(min(20, 2 + 0.7 * len(df.columns)), 1.5 + 0.4 * len(df)))
    ax.axis('off')
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     loc='center',
                     cellLoc='center',
                     colLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df.columns))))
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def create_abort_table(abort_results):
    """
    Creates a DataFrame table summarizing abort model results.
    
    Args:
        abort_results: Dictionary containing abort model results with parameter samples
        
    Returns:
        DataFrame with formatted abort results or None if no valid data
    """
    if not abort_results or not isinstance(abort_results, dict):
        print("Abort results data not found or invalid.")
        return None

    data = {'Parameter': [], 'Mean Value': []}
    scalar_data = {}

    for key, value in abort_results.items():
        if key.endswith('_samples') or key == 't_A_aff_samp': # Include potential typo
            param_name = key.replace('_samples', '').replace('_samp', '')
            if isinstance(value, np.ndarray) and value.size > 0:
                data['Parameter'].append(param_name)
                data['Mean Value'].append(np.mean(value))
            else:
                data['Parameter'].append(param_name)
                data['Mean Value'].append('N/A') # Handle non-array or empty
        elif key in ['elbo', 'elbo_sd', 'loglike']:
            scalar_data[key] = value

    df = pd.DataFrame(data)
    # Add scalar values as separate rows
    for key, value in scalar_data.items():
         df.loc[len(df)] = [key, value]

    # Add message if exists
    if 'message' in abort_results:
         df.loc[len(df)] = ['message', abort_results['message']]

    print("--- Abort Model Results ---")
    print(df.to_string(index=False))
    print("\n")
    return df

def create_tied_table(all_results):
    """
    Creates a DataFrame table comparing all tied model results.
    
    Args:
        all_results: Dictionary containing all model results
        
    Returns:
        DataFrame with formatted comparison or None if no valid data
    """
    tied_keys = [k for k in all_results.keys() if k.startswith('vbmc_') and k.endswith('_tied_results')]
    if not tied_keys:
        print("No tied parameter results found.")
        return None

    # Collect all unique parameter names (base name without _samples)
    all_params = set()
    for key in tied_keys:
        for param_key in all_results[key].keys():
            if param_key.endswith('_samples'):
                all_params.add(param_key.replace('_samples', ''))

    # Use user-specified order
    desired_order = [
        'rate_lambda', 'T_0', 'theta_E', 'w', 't_E_aff', 'del_go',
        'rate_norm_l', 'bump_height', 'bump_width', 'dip_height', 'dip_width'
    ]
    # Only keep params that are present
    present_params = [p for p in desired_order if p in all_params]
    param_headers = []
    for p in present_params:
        if p == 'T_0':
            param_headers.append('T_0 (ms)')
        else:
            param_headers.append(p)

    table_data = {'Model': []}
    scalar_cols = ['ELBO', 'ELBO SD', 'Log Likelihood']
    for col in scalar_cols + param_headers:
        table_data[col] = []

    # Populate table data
    for key in tied_keys:
        model_name = key.replace('vbmc_', '').replace('_tied_results', '').replace('_', ' ').title()
        table_data['Model'].append(model_name)
        model_results = all_results[key]

        # Add scalar values, rounded
        for col in ['elbo', 'elbo_sd', 'loglike']:
            val = model_results.get(col, '-')
            if isinstance(val, float):
                table_data[{'elbo': 'ELBO', 'elbo_sd': 'ELBO SD', 'loglike': 'Log Likelihood'}[col]].append(f"{val:.3f}")
            else:
                table_data[{'elbo': 'ELBO', 'elbo_sd': 'ELBO SD', 'loglike': 'Log Likelihood'}[col]].append(val)

        # Add parameter means, rounded, and T_0 scaled
        for i, param in enumerate(present_params):
            param_key_samples = f"{param}_samples"
            header = param_headers[i]
            if param_key_samples in model_results and isinstance(model_results[param_key_samples], np.ndarray):
                mean_val = np.mean(model_results[param_key_samples])
                if param == 'T_0':
                    mean_val = mean_val * 1000  # convert to ms
                table_data[header].append(f"{mean_val:.3f}")
            else:
                table_data[header].append('-')

    df = pd.DataFrame(table_data)
    print("--- Tied Models Comparison ---")
    print(df.to_string(index=False))
    print("\n")
    return df
