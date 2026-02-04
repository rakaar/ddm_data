# %%
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = '/home/rlab/raghavendra/ddm_data/fit_animal_by_animal'
GAMMA_DATA_PATH = os.path.join(BASE_DIR, 'gamma_sep_by_median_lapse_rate_data.pkl')

# %%
with open(GAMMA_DATA_PATH, 'rb') as handle:
    gamma_data = pickle.load(handle)

# %%
all_ABL = gamma_data['all_ABL']
all_ILD_sorted = gamma_data['all_ILD_sorted']
gamma_low_lapse = gamma_data['gamma_low_lapse']
gamma_high_lapse = gamma_data['gamma_high_lapse']

colors = {20: 'tab:blue', 40: 'tab:orange', 60: 'tab:green'}

fig, ax = plt.subplots(figsize=(6, 4))

for ABL in all_ABL:
    low_values = gamma_low_lapse[str(ABL)]
    high_values = gamma_high_lapse[str(ABL)]
    if low_values.size == 0:
        combined_values = high_values
    elif high_values.size == 0:
        combined_values = low_values
    else:
        combined_values = np.vstack([low_values, high_values])

    mean_gamma = np.nanmean(combined_values, axis=0)
    sem_gamma = np.nanstd(combined_values, axis=0) / np.sqrt(
        np.sum(~np.isnan(combined_values), axis=0)
    )

    color = colors.get(ABL, 'tab:blue')
    ax.errorbar(
        all_ILD_sorted,
        mean_gamma,
        yerr=sem_gamma,
        fmt='o',
        color=color,
        capsize=0,
        markersize=8,
    )

ax.set_xlabel('ILD', fontsize=20)
ax.set_ylabel(r'$\Gamma$', fontsize=20)
ax.set_xticks([-15, -5, 5, 15])
ax.set_yticks([-2, 0, 2])
ax.tick_params(axis='both', labelsize=18)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
output_prefix = os.path.join(BASE_DIR, 'gamma_norm_lapse_all_ABL')
plt.tight_layout()
plt.savefig(f'{output_prefix}.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_prefix}.pdf', bbox_inches='tight')
plt.show()

print(f'Saved: {output_prefix}.png')
print(f'Saved: {output_prefix}.pdf')

# %%
