# %%
from time_vary_and_norm_simulators import psiam_tied_data_gen_wrapper_rate_norm_time_vary_fn
from joblib import Parallel, delayed
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd


ABL_samples = np.random.choice([20, 40, 60], size=int(1e6))
ILD_samples = np.random.choice([1, 2, 4, 8, 16, -1, -2, -4, -8, -16], size=int(1e6))
t_stim_samples = np.random.exponential(0.4, int(1e6)) + 0.2

dt = 1e-3
N_sim = int(1e6)

from types import SimpleNamespace

V_A = 1.6
theta_A = 2.5
t_A_aff = -0.22
N_print = int(N_sim / 5)
rate_lambda = 2.2391
T_0 = 0.101692  # seconds (converted from ms)
theta_E = 2.3770
w = 0.5458
Z_E = 0.2177
t_E_aff = 0.0390103  # seconds (converted from ms)
del_go = 0.1963
rate_norm_l = 0.9652

bump_height = 0.1124
bump_width = 0.2555
dip_height = 0.4427
dip_width = 0.0235
bump_offset = 0.0  # Not in image, set to 0 by default

phi_params = {
    'h1': bump_width,
    'a1': bump_height,
    'h2': dip_width,
    'a2': dip_height,
    'b1': bump_offset
}
phi_params_obj = SimpleNamespace(**phi_params)



sim_results = Parallel(n_jobs=30)(
        delayed(psiam_tied_data_gen_wrapper_rate_norm_fn)(
            V_A, theta_A, ABL_samples[iter_num], ILD_samples[iter_num], rate_lambda, T_0, theta_E, Z_E, t_A_aff, t_E_aff, del_go, 
            t_stim_samples[iter_num], rate_norm_l, iter_num, N_print, dt
        ) for iter_num in tqdm(range(N_sim))
    )

sim_results_df = pd.DataFrame(sim_results)
sim_results_df_valid = sim_results_df[
    (sim_results_df['rt'] > sim_results_df['t_stim']) 
].copy()

# %%
# Q Q plots

import matplotlib.pyplot as plt
import numpy as np

# Prepare
sim_results_df_valid['abs_ILD'] = sim_results_df_valid['ILD'].abs()
ABLs = [20, 40, 60]
abs_ILDs = [1., 2., 4., 8., 16.]
percentiles = np.arange(5, 100, 10)

# Compute quantiles for each (ABL, abs_ILD)
Q_dict = {abs_ILD: {} for abs_ILD in abs_ILDs}
for abs_ILD in abs_ILDs:
    df_ild = sim_results_df_valid[sim_results_df_valid['abs_ILD'] == abs_ILD]
    for abl in ABLs:
        vals = df_ild[df_ild['ABL'] == abl]['rt'] - df_ild[df_ild['ABL'] == abl]['t_stim']
        if len(vals) > 0:
            Q_dict[abs_ILD][abl] = np.percentile(vals, percentiles)
        else:
            Q_dict[abs_ILD][abl] = np.full(percentiles.shape, np.nan)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
labels = {20: 'ABL 20 - 60', 40: 'ABL 40 - 60'}
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for j, abl in enumerate([20, 40]):
    ax = axes[j]
    for i, abs_ILD in enumerate(abs_ILDs):
        Q_abl = Q_dict[abs_ILD][abl]
        Q_60 = Q_dict[abs_ILD][60]
        mask = ~np.isnan(Q_abl) & ~np.isnan(Q_60) & (Q_60 > 0)
        if not np.any(mask):
            continue
        x = Q_60[mask]
        y = Q_abl[mask] - Q_60[mask]
        ax.plot(x, y, color=colors[i], label=f'|ILD|={abs_ILD}', linewidth=2)
    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.set_xlabel('Q_60 (s)')
    if j == 0:
        ax.set_ylabel('Q_ABL - Q_60')
    ax.set_title(labels[abl])
    # ax.set_xlim(0.1, 0.3)
    # ax.set_ylim(-0.05, 0.4)
    ax.legend(title='abs(ILD)', fontsize=10)

plt.tight_layout()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

# Parameters
_lambda = 2
chi = 17

# ILD values (use your simulation values)
ILDs = [1., 2., 4., 8., 16.]
ILDs = np.array(ILDs)
l = 0.9  # set l=1 unless you want to plot for other l values

# Compute numerator and denominator
numerator = np.cosh(_lambda * ILDs / chi)
denominator = np.cosh(_lambda * l * ILDs / chi)
ratio = numerator / denominator

plt.figure(figsize=(7,4))
plt.plot(ILDs, ratio, 'o-', label=r'$\cosh(\lambda\, \mathrm{ILD}/\chi)/\cosh(\lambda\, l\, \mathrm{ILD}/\chi)$')
plt.xlabel('ILD')
plt.ylabel('cosh ratio')
plt.title(r'$\frac{\cosh(\lambda\, \mathrm{ILD}/\chi)}{\cosh(\lambda\, l\, \mathrm{ILD}/\chi)}$' + f' ($\lambda$={_lambda}, $\chi$={chi}, $l$={l})')
plt.axhline(1, color='gray', linestyle='--', linewidth=1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()