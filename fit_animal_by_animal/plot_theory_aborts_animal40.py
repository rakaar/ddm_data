import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from animal_wise_plotting_utils import plot_abort_diagnostic
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

# --- Plot using plot_abort_diagnostic (empirical + theory) ---
print("Showing plot_abort_diagnostic (empirical + theory, animal 40)...")
plot_abort_diagnostic(
    pdf_pages=None,  # Shows the plot instead of saving
    df_aborts_animal=df_aborts_animal,
    df_valid_and_aborts=df_all_trials_animal,
    N_theory=N_theory,
    V_A=params2['V_A'],
    theta_A=params2['theta_A'],
    t_A_aff=params2['t_A_aff'],
    T_trunc=T_trunc,
    rho_A_t_VEC_fn=rho_A_t_VEC_fn,
    cum_A_t_fn=cum_A_t_fn,
    title="Abort Model RTD Diagnostic (animal 40)"
)

# --- Plot using mean PDF style (theory only) ---
t_pts = np.arange(0, 1.25, 0.001)
t_stim_samples = df_all_trials_animal['intended_fix'].sample(N_theory, replace=True).values

def plot_mean_pdf_for_params(V_A, theta_A, t_A_aff, label, color=None):
    pdf_samples = np.zeros((N_theory, len(t_pts)))
    for i, t_stim in enumerate(t_stim_samples):
        t_stim_idx = np.searchsorted(t_pts, t_stim)
        proactive_trunc_idx = np.searchsorted(t_pts, T_trunc)
        pdf_samples[i, :proactive_trunc_idx] = 0
        pdf_samples[i, t_stim_idx:] = 0
        t_btn = t_pts[proactive_trunc_idx:t_stim_idx-1]
        pdf_samples[i, proactive_trunc_idx:t_stim_idx-1] = rho_A_t_VEC_fn(t_btn - t_A_aff, V_A, theta_A) / (1 - cum_A_t_fn(T_trunc - t_A_aff, V_A, theta_A))
    plt.plot(t_pts, np.mean(pdf_samples, axis=0), label=label, color=color)

print("Showing plot_mean_pdf_for_params (theory only, animal 40)...")
plt.figure(figsize=(10,5))
plot_mean_pdf_for_params(params2['V_A'], params2['theta_A'], params2['t_A_aff'], label="Mean PDF (animal 40)", color="orange")
plt.title("Theory Aborts for Animal 40 (Mean PDF Style)")
plt.xlabel("Time (s)")
plt.ylabel("PDF")
plt.legend()
plt.show()
