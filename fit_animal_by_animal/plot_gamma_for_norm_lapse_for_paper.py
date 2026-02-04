# %%
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = '/home/rlab/raghavendra/ddm_data/fit_animal_by_animal'
GAMMA_DATA_PATH = os.path.join(BASE_DIR, 'gamma_sep_by_median_lapse_rate_data.pkl')
NORM_GAMMA_DATA_PATH = os.path.join(
    os.path.dirname(BASE_DIR),
    'fit_each_condn',
    'norm_gamma_fig2_data.pkl',
)

# %%
with open(GAMMA_DATA_PATH, 'rb') as handle:
    gamma_data = pickle.load(handle)

with open(NORM_GAMMA_DATA_PATH, 'rb') as handle:
    norm_gamma_data = pickle.load(handle)

# %%
all_ABL = gamma_data['all_ABL']
all_ILD_sorted = gamma_data['all_ILD_sorted']
gamma_low_lapse = gamma_data['gamma_low_lapse']
gamma_high_lapse = gamma_data['gamma_high_lapse']
gamma_norm_no_lapse = norm_gamma_data['gamma_norm_model_fit_theoretical_all_animals']
norm_no_lapse_ild_pts = norm_gamma_data['ILD_pts']

fig, ax = plt.subplots(figsize=(6, 4))

gamma_lapse_by_abl = []
for ABL in all_ABL:
    low_values = gamma_low_lapse[str(ABL)]
    high_values = gamma_high_lapse[str(ABL)]
    if low_values.size == 0:
        combined_values = high_values
    elif high_values.size == 0:
        combined_values = low_values
    else:
        combined_values = np.vstack([low_values, high_values])
    gamma_lapse_by_abl.append(combined_values)

if gamma_lapse_by_abl:
    gamma_lapse_all = np.vstack(gamma_lapse_by_abl)
    mean_gamma_lapse = np.nanmean(gamma_lapse_all, axis=0)
    sem_gamma_lapse = np.nanstd(gamma_lapse_all, axis=0) / np.sqrt(
        np.sum(~np.isnan(gamma_lapse_all), axis=0)
    )

    ax.errorbar(
        all_ILD_sorted,
        mean_gamma_lapse,
        yerr=sem_gamma_lapse,
        fmt='o',
        color='red',
        capsize=0,
        markersize=8,
    )

if gamma_norm_no_lapse is not None and len(gamma_norm_no_lapse) > 0:
    mean_gamma_no_lapse = np.nanmean(gamma_norm_no_lapse, axis=0)
    sem_gamma_no_lapse = np.nanstd(gamma_norm_no_lapse, axis=0) / np.sqrt(
        np.sum(~np.isnan(gamma_norm_no_lapse), axis=0)
    )

    mean_gamma_no_lapse_interp = np.interp(
        all_ILD_sorted,
        norm_no_lapse_ild_pts,
        mean_gamma_no_lapse,
    )
    sem_gamma_no_lapse_interp = np.interp(
        all_ILD_sorted,
        norm_no_lapse_ild_pts,
        sem_gamma_no_lapse,
    )

    ax.errorbar(
        all_ILD_sorted,
        mean_gamma_no_lapse_interp,
        yerr=sem_gamma_no_lapse_interp,
        fmt='o',
        color='black',
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
