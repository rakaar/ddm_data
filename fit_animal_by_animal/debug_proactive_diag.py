# %% 
import numpy as np
import pickle
import pandas as pd

filename = 'results_Comparable_animal_37.pkl'
with open(filename, 'rb') as f:
    results = pickle.load(f)

# %%
print(results['vbmc_aborts_results'])

# %%
abort_results = results['vbmc_aborts_results']
V_A = abort_results['V_A_samples'].mean()
theta_A = abort_results['theta_A_samples'].mean()
t_A_aff = abort_results['t_A_aff_samp'].mean()

print(f'V_A ={V_A :.2f}, theta_A={theta_A :.2f}, t_A_aff={t_A_aff :.2f}')

# %%
# check old code

import random
animal = 37
batch_name = 'Comparable'
exp_df = pd.read_csv('../outExp.csv')

exp_df = exp_df[~((exp_df['RTwrtStim'].isna()) & (exp_df['abort_event'] == 3))].copy()

exp_df_batch = exp_df[
    (exp_df['batch_name'] == batch_name) &
    (exp_df['LED_trial'].isin([np.nan, 0]))
].copy()

exp_df_batch['choice'] = exp_df_batch['response_poke'].apply(lambda x: 1 if x == 3 else (-1 if x == 2 else random.choice([1, -1])))
# 1 or 0 if the choice was correct or not
exp_df_batch['accuracy'] = (exp_df_batch['ILD'] * exp_df_batch['choice']).apply(lambda x: 1 if x > 0 else 0)

### DF - valid and aborts ###
df_valid_and_aborts = exp_df_batch[
    (exp_df_batch['success'].isin([1,-1])) |
    (exp_df_batch['abort_event'] == 3)
].copy()

df_aborts = df_valid_and_aborts[df_valid_and_aborts['abort_event'] == 3]
df_all_trials_animal = df_valid_and_aborts[df_valid_and_aborts['animal'] == animal]
df_aborts_animal = df_aborts[df_aborts['animal'] == animal]

# %%
from time_vary_norm_utils import (
    up_or_down_RTs_fit_fn, cum_pro_and_reactive_time_vary_fn,
    rho_A_t_VEC_fn, up_or_down_RTs_fit_wrt_stim_fn, rho_A_t_fn, cum_A_t_fn)

N_theory = int(1e3)
T_trunc = 0.3
import matplotlib.pyplot as plt

t_pts = np.arange(0, 1.25, 0.001)

# Sample intended fixation times from all trials (valid and aborts)
# Use .copy() to avoid SettingWithCopyWarning if df_valid_and_aborts is a slice
t_stim_samples_df = df_valid_and_aborts.sample(N_theory, replace=True).copy()
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
    
    t_btn = t_pts[proactive_trunc_idx:t_stim_idx-1] # Time between truncation and stim end
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
    pdf_samples[i, proactive_trunc_idx:t_stim_idx-1] = pdf_slice

# Average PDF across samples
avg_pdf = np.mean(pdf_samples, axis=0)

# Plotting
fig_aborts_diag = plt.figure(figsize=(10, 5))
bins = np.arange(0, 2, 0.02)

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

plt.title('Abort Diagnostic (Refactored)')
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
plt.show()


# %%
# check refactored code
import random
animal = 37
batch_name = 'Comparable'
exp_df = pd.read_csv('../outExp.csv')

exp_df = exp_df[~((exp_df['RTwrtStim'].isna()) & (exp_df['abort_event'] == 3))].copy()

exp_df_batch = exp_df[
    (exp_df['batch_name'] == batch_name) &
    (exp_df['LED_trial'].isin([np.nan, 0]))
].copy()

exp_df_batch['choice'] = exp_df_batch['response_poke'].apply(lambda x: 1 if x == 3 else (-1 if x == 2 else random.choice([1, -1])))
# 1 or 0 if the choice was correct or not
exp_df_batch['accuracy'] = (exp_df_batch['ILD'] * exp_df_batch['choice']).apply(lambda x: 1 if x > 0 else 0)

### DF - valid and aborts ###
df_valid_and_aborts = exp_df_batch[
    (exp_df_batch['success'].isin([1,-1])) |
    (exp_df_batch['abort_event'] == 3)
].copy()

df_aborts = df_valid_and_aborts[df_valid_and_aborts['abort_event'] == 3]
df_all_trials_animal = df_valid_and_aborts[df_valid_and_aborts['animal'] == animal]
df_aborts_animal = df_aborts[df_aborts['animal'] == animal]

# %%
from time_vary_norm_utils import (
    up_or_down_RTs_fit_fn, cum_pro_and_reactive_time_vary_fn,
    rho_A_t_VEC_fn, up_or_down_RTs_fit_wrt_stim_fn, rho_A_t_fn, cum_A_t_fn)

N_theory = int(1e3)
T_trunc = 0.3
import matplotlib.pyplot as plt

t_pts = np.arange(0, 1.25, 0.001)

# Sample intended fixation times from all trials (valid and aborts)
# Use .copy() to avoid SettingWithCopyWarning if df_valid_and_aborts is a slice
t_stim_samples_df = df_valid_and_aborts.sample(N_theory, replace=True).copy()
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
    
    t_btn = t_pts[proactive_trunc_idx:t_stim_idx-1] # Time between truncation and stim end
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
    pdf_samples[i, proactive_trunc_idx:t_stim_idx-1] = pdf_slice

# Average PDF across samples
avg_pdf = np.mean(pdf_samples, axis=0)

# Plotting
fig_aborts_diag = plt.figure(figsize=(10, 5))
bins = np.arange(0, 2, 0.02)

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

plt.title('Abort Diagnostic (Refactored)')
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
plt.show()

# %%
# Fix the PDFs

from animal_wise_plotting_utils import plot_abort_diagnostic
from matplotlib.backends.backend_pdf import PdfPages

# Loop through all animals and generate a new PDF with the correct diagnostic page
for animal in df_valid_and_aborts['animal'].unique():
    df_all_trials_animal = df_valid_and_aborts[df_valid_and_aborts['animal'] == animal]
    df_aborts_animal = df_aborts[df_aborts['animal'] == animal]
    pdf_filename = f'results_{batch_name}_animal_{animal}_diagnostic_fixed.pdf'
    pdf = PdfPages(pdf_filename)
    print(f'Writing fixed diagnostic for animal {animal} to {pdf_filename}')
    plot_abort_diagnostic(
        pdf_pages=pdf,
        df_aborts_animal=df_aborts_animal,
        df_valid_and_aborts=df_valid_and_aborts,
        N_theory=N_theory,
        V_A=V_A,
        theta_A=theta_A,
        t_A_aff=t_A_aff,
        T_trunc=T_trunc,
        rho_A_t_VEC_fn=rho_A_t_VEC_fn,
        cum_A_t_fn=cum_A_t_fn,
        title=f'Abort Model RTD Diagnostic (fixed, animal {animal})'
    )
    pdf.close()

