import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time_vary_norm_utils import rho_A_t_VEC_fn, cum_A_t_fn

# --- Parameters ---
params2 = dict(V_A=2.6762, theta_A=3.8446, t_A_aff=-0.3270)
N_theory = 1000
T_trunc = 0.300
animal = 40
batch_name = "Comparable"  # Change as needed

# --- Load Data ---
exp_df = pd.read_csv('../outExp.csv')
exp_df = exp_df[~((exp_df['RTwrtStim'].isna()) & (exp_df['abort_event'] == 3))].copy()
exp_df_batch = exp_df[
    (exp_df['batch_name'] == batch_name) &
    (exp_df['LED_trial'].isin([np.nan, 0]))
].copy()
exp_df_batch['choice'] = exp_df_batch['response_poke'].apply(lambda x: 1 if x == 3 else (-1 if x == 2 else np.random.choice([1, -1])))
exp_df_batch['accuracy'] = (exp_df_batch['ILD'] * exp_df_batch['choice']).apply(lambda x: 1 if x > 0 else 0)

df_valid_and_aborts = exp_df_batch[
    (exp_df_batch['success'].isin([1,-1])) |
    (exp_df_batch['abort_event'] == 3)
].copy()
df_aborts = df_valid_and_aborts[df_valid_and_aborts['abort_event'] == 3]
df_all_trials_animal = df_valid_and_aborts[df_valid_and_aborts['animal'] == animal]
df_aborts_animal = df_aborts[df_aborts['animal'] == animal]

# --- Prepare for plotting ---
t_pts = np.arange(0, 1.25, 0.001)
t_stim_samples = df_all_trials_animal['intended_fix'].sample(N_theory, replace=True).values

# --- Calculate theory PDFs ---
pdf_samples_diag = np.zeros((N_theory, len(t_pts)))
pdf_samples_mean = np.zeros((N_theory, len(t_pts)))
for i, t_stim in enumerate(t_stim_samples):
    t_stim_idx = np.searchsorted(t_pts, t_stim)
    proactive_trunc_idx = np.searchsorted(t_pts, T_trunc)
    # Diagnostic style (same as mean, but kept for clarity)
    pdf_samples_diag[i, :proactive_trunc_idx] = 0
    pdf_samples_diag[i, t_stim_idx:] = 0
    t_btn = t_pts[proactive_trunc_idx:t_stim_idx-1]
    pdf_samples_diag[i, proactive_trunc_idx:t_stim_idx-1] = rho_A_t_VEC_fn(t_btn - params2['t_A_aff'], params2['V_A'], params2['theta_A']) / (1 - cum_A_t_fn(T_trunc - params2['t_A_aff'], params2['V_A'], params2['theta_A']))
    # Mean PDF style (identical calculation in this case)
    pdf_samples_mean[i, :proactive_trunc_idx] = 0
    pdf_samples_mean[i, t_stim_idx:] = 0
    pdf_samples_mean[i, proactive_trunc_idx:t_stim_idx-1] = rho_A_t_VEC_fn(t_btn - params2['t_A_aff'], params2['V_A'], params2['theta_A']) / (1 - cum_A_t_fn(T_trunc - params2['t_A_aff'], params2['V_A'], params2['theta_A']))

avg_pdf_diag = np.mean(pdf_samples_diag, axis=0)
avg_pdf_mean = np.mean(pdf_samples_mean, axis=0)

# --- Empirical histogram (as in plot_abort_diagnostic) ---
bins = np.arange(0, 2, 0.01)
animal_abort_RT = df_aborts_animal['TotalFixTime'].dropna().values
animal_abort_RT_trunc = animal_abort_RT[animal_abort_RT > T_trunc]

plt.figure(figsize=(10,5))
if len(animal_abort_RT_trunc) > 0:
    if 'animal' in df_valid_and_aborts.columns:
        animal_id = df_aborts_animal['animal'].iloc[0] if len(df_aborts_animal) > 0 else None
        df_all_trials_animal = df_valid_and_aborts[df_valid_and_aborts['animal'] == animal_id]
    else:
        df_all_trials_animal = df_valid_and_aborts
    df_before_trunc_animal = df_aborts_animal[df_aborts_animal['TotalFixTime'] < T_trunc]
    N_valid_and_trunc_aborts = len(df_all_trials_animal) - len(df_before_trunc_animal)
    frac_aborts = len(animal_abort_RT_trunc) / N_valid_and_trunc_aborts if N_valid_and_trunc_aborts > 0 else 0
    aborts_hist, _ = np.histogram(animal_abort_RT_trunc, bins=bins, density=True)
    plt.plot(bins[:-1], aborts_hist * frac_aborts, label='Data (Aborts > T_trunc)', color='black')
else:
    plt.text(0.5, 0.5, 'No empirical abort data > T_trunc', 
             horizontalalignment='center', verticalalignment='center', 
             transform=plt.gca().transAxes)

# --- Plot both theory curves on the same axes ---
plt.plot(t_pts, avg_pdf_diag, 'r-', lw=2, label='Theory (Abort Model Diagnostic)')
plt.plot(t_pts, avg_pdf_mean, 'b--', lw=2, label='Theory (Mean PDF)')

plt.title("Theory Aborts for Animal 40 (Empirical + Both Theory Curves)")
plt.xlabel("Time (s)")
plt.ylabel("Probability Density")
plt.legend()
plt.xlim([0, np.max(bins)])
# Set y-limits sensibly
if len(animal_abort_RT_trunc) > 0:
    max_density = np.max(aborts_hist * frac_aborts) if len(aborts_hist) > 0 else 1
    plt.ylim([0, max(np.max(avg_pdf_diag), np.max(avg_pdf_mean), max_density) * 1.1])
elif np.any(avg_pdf_diag > 0) or np.any(avg_pdf_mean > 0):
    plt.ylim([0, max(np.max(avg_pdf_diag), np.max(avg_pdf_mean)) * 1.1])
else:
    plt.ylim([0, 1])
plt.show()
