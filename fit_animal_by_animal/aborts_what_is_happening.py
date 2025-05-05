from time_vary_norm_utils import (
    up_or_down_RTs_fit_fn, cum_pro_and_reactive_time_vary_fn,
    rho_A_t_VEC_fn, up_or_down_RTs_fit_wrt_stim_fn, rho_A_t_fn, cum_A_t_fn)

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

batch_name = "Comparable"
N_theory = 1000
T_trunc = 0.300

# read data
exp_df = pd.read_csv('../outExp.csv')
exp_df = exp_df[~((exp_df['RTwrtStim'].isna()) & (exp_df['abort_event'] == 3))].copy()
exp_df_batch = exp_df[
    (exp_df['batch_name'] == batch_name) &
    (exp_df['LED_trial'].isin([np.nan, 0]))
].copy()
exp_df_batch['choice'] = exp_df_batch['response_poke'].apply(lambda x: 1 if x == 3 else (-1 if x == 2 else random.choice([1, -1])))
exp_df_batch['accuracy'] = (exp_df_batch['ILD'] * exp_df_batch['choice']).apply(lambda x: 1 if x > 0 else 0)
df_valid_and_aborts = exp_df_batch[
    (exp_df_batch['success'].isin([1,-1])) |
    (exp_df_batch['abort_event'] == 3)
].copy()


### Animal selection ###

t_pts = np.arange(0,1.25, 0.001)
t_stim_samples = df_valid_and_aborts.sample(N_theory)['intended_fix']
pdf_samples = np.zeros((N_theory, len(t_pts)))


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

# Set 1 parameters
params1 = dict(V_A=2.61106, theta_A=3.67686, t_A_aff=-0.29517)
# Set 2 parameters
params2 = dict(V_A=2.6762, theta_A=3.8446, t_A_aff=-0.3270)

plot_mean_pdf_for_params(**params1, label='Set 1', color='blue')
plot_mean_pdf_for_params(**params2, label='Set 2', color='orange')

plt.legend()
plt.title('Comparison of Mean pdf_samples for Two Parameter Sets')
plt.xlabel('Time (s)')
plt.ylabel('Mean PDF')
plt.show()