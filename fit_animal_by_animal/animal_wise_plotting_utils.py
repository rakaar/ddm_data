import matplotlib.pyplot as plt
import numpy as np
import corner
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


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
        
    plt.text(0.05, 0.95, full_text, va='top', ha='left', fontsize=10, family='monospace')
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
    t_pts = np.arange(0, 1.25, 0.001)
    
    # Sample intended fixation times from all trials (valid and aborts)
    # Use .copy() to avoid SettingWithCopyWarning if df_valid_and_aborts is a slice
    t_stim_samples_df = df_valid_and_aborts.sample(n=N_theory, replace=True).copy()
    t_stim_samples = t_stim_samples_df['intended_fix'].values 

    pdf_samples = np.zeros((N_theory, len(t_pts)))

    for i, t_stim in enumerate(t_stim_samples):
        t_stim_idx = np.searchsorted(t_pts, t_stim)
        proactive_trunc_idx = np.searchsorted(t_pts, T_trunc)
        
        # Ensure indices are valid and make sense
        if proactive_trunc_idx >= t_stim_idx:
            # If truncation happens at or after intended stim time, PDF is 0
            continue 
            
        pdf_samples[i, :proactive_trunc_idx] = 0 # PDF is zero before truncation
        pdf_samples[i, t_stim_idx:] = 0 # PDF is zero after intended stim time
        
        t_btn = t_pts[proactive_trunc_idx:t_stim_idx] # Time between truncation and stim end
        if len(t_btn) == 0:
            continue # Skip if no time points in the interval
            
        # Calculate time relative to afferent delay
        t_rel_aff = t_btn - t_A_aff
        # Ensure we only calculate for non-negative relative times
        valid_time_mask = t_rel_aff >= 0
        if not np.any(valid_time_mask):
             continue # Skip if no valid times after afferent delay
        
        # Calculate the normalization factor (denominator)
        # Calculate CDF at T_trunc relative to afferent delay
        t_trunc_rel_aff = T_trunc - t_A_aff
        if t_trunc_rel_aff < 0:
             cum_A_at_trunc = 0.0 # CDF is 0 before afferent delay starts
        else:
             cum_A_at_trunc = cum_A_t_fn(t_trunc_rel_aff, V_A, theta_A)
        
        norm_factor = 1.0 - cum_A_at_trunc
        if norm_factor <= 1e-9: # Avoid division by zero or very small numbers
            # If probability of not aborting by T_trunc is near zero, PDF is effectively zero
            continue 

        # Calculate the PDF for valid relative times
        rho_vals = rho_A_t_VEC_fn(t_rel_aff[valid_time_mask], V_A, theta_A)
        
        # Assign calculated PDF values, dividing by the normalization factor
        pdf_slice = np.zeros_like(t_btn, dtype=float)
        pdf_slice[valid_time_mask] = rho_vals / norm_factor
        
        # Place the calculated PDF slice into the correct indices
        pdf_samples[i, proactive_trunc_idx:t_stim_idx] = pdf_slice

    # Average PDF across samples
    avg_pdf = np.mean(pdf_samples, axis=0)
    # Normalize the average PDF so its integral is 1
    if np.sum(avg_pdf) > 1e-9:
         avg_pdf = avg_pdf / (np.sum(avg_pdf) * (t_pts[1]-t_pts[0])) # Normalize by integral
    else:
         avg_pdf.fill(0)

    # Plotting
    fig_aborts_diag = plt.figure(figsize=(10, 5))
    bins = np.arange(0, 2, 0.01)

    # Get empirical abort RTs and filter by truncation time
    animal_abort_RT = df_aborts_animal['TotalFixTime'].dropna().values
    animal_abort_RT_trunc = animal_abort_RT[animal_abort_RT > T_trunc]

    # Plot empirical histogram
    if len(animal_abort_RT_trunc) > 0:
        plt.hist(animal_abort_RT_trunc, bins=bins, density=True, alpha=0.6, label='Data (Aborts > T_trunc)')
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
        max_density = np.max(np.histogram(animal_abort_RT_trunc, bins=bins, density=True)[0])
        plt.ylim([0, max(np.max(avg_pdf), max_density) * 1.1])
    elif np.any(avg_pdf > 0):
         plt.ylim([0, np.max(avg_pdf) * 1.1])
    else:
        plt.ylim([0, 1]) # Default ylim if no data and no theory

    pdf_pages.savefig(fig_aborts_diag)
    plt.close(fig_aborts_diag)
