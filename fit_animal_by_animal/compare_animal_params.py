# %%
# param vs animal wise pdfs

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Directory containing the results
RESULTS_DIR = os.path.dirname(__file__)
PKL_PATTERN = 'results_Comparable_animal_{}.pkl'

# Find all animal pickle files
animal_ids = []
for fname in os.listdir(RESULTS_DIR):
    if fname.startswith('results_Comparable_animal_') and fname.endswith('.pkl'):
        try:
            animal_id = int(fname.split('_')[-1].replace('.pkl', ''))
            animal_ids.append(animal_id)
        except Exception:
            continue
animal_ids = sorted(animal_ids)

# Model configs: (model_key, param_keys, param_labels, plot_title)
model_configs = [
    ('vbmc_aborts_results',
        ['V_A_samples', 'theta_A_samples', 't_A_aff_samp'],
        ['V_A', 'theta_A', 't_A_aff'],
        'Aborts Model'),
    ('vbmc_vanilla_tied_results',
        ['rate_lambda_samples', 'T_0_samples', 'theta_E_samples', 'w_samples', 't_E_aff_samples', 'del_go_samples'],
        ['rate_lambda', 'T_0', 'theta_E', 'w', 't_E_aff', 'del_go'],
        'Vanilla TIED Model'),
    ('vbmc_norm_tied_results',
        ['rate_lambda_samples', 'T_0_samples', 'theta_E_samples', 'w_samples', 't_E_aff_samples', 'del_go_samples', 'rate_norm_l_samples'],
        ['rate_lambda', 'T_0', 'theta_E', 'w', 't_E_aff', 'del_go', 'rate_norm_l'],
        'Norm TIED Model'),
    ('vbmc_time_vary_norm_tied_results',
        ['rate_lambda_samples', 'T_0_samples', 'theta_E_samples', 'w_samples', 't_E_aff_samples', 'del_go_samples', 'rate_norm_l_samples', 'bump_height_samples', 'bump_width_samples', 'dip_height_samples', 'dip_width_samples'],
        ['rate_lambda', 'T_0', 'theta_E', 'w', 't_E_aff', 'del_go', 'rate_norm_l', 'bump_height', 'bump_width', 'dip_height', 'dip_width'],
        'Time-Varying Norm TIED Model'),
]

for model_key, param_keys, param_labels, plot_title in model_configs:
    means = {param: [] for param in param_keys}
    stds = {param: [] for param in param_keys}
    valid_animals = []
    for animal_id in animal_ids:
        pkl_path = os.path.join(RESULTS_DIR, PKL_PATTERN.format(animal_id))
        if not os.path.exists(pkl_path):
            continue
        with open(pkl_path, 'rb') as f:
            results = pickle.load(f)
        if model_key not in results:
            continue
        valid_animals.append(animal_id)
        for param in param_keys:
            samples = np.asarray(results[model_key][param])
            means[param].append(np.mean(samples))
            stds[param].append(np.std(samples))

    # Plot
    fig, axes = plt.subplots(len(param_keys), 1, figsize=(7, 1.5*len(param_keys)), sharex=False)
    if len(param_keys) == 1:
        axes = [axes]
    for i, param in enumerate(param_keys):
        ax = axes[i]
        y_pos = np.arange(len(valid_animals))
        ax.errorbar(means[param], y_pos, xerr=stds[param], fmt='o', color='k', ecolor='gray', capsize=4)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([str(a) for a in valid_animals])
        ax.set_xlabel(param_labels[i])
        ax.set_ylabel('Animal')
        ax.set_title(f'{plot_title}: {param_labels[i]}')
        ax.axvspan(0, np.max(means[param]+np.array(stds[param])), color='#b7e4c7', alpha=0.2, zorder=-1)
    plt.tight_layout()
    outname = f'compare_animals_{model_key}.pdf'
    plt.savefig(os.path.join(RESULTS_DIR, outname))
    print(f'Saved: {outname}')
    plt.close(fig)

# %%
# elbo, loglike

fig, axes = plt.subplots(4, 2, figsize=(12, 12), sharex=True)

# Model order for this plot (to match rows)
model_order = [
    ('vbmc_aborts_results', 'Aborts Model'),
    ('vbmc_vanilla_tied_results', 'Vanilla TIED Model'),
    ('vbmc_norm_tied_results', 'Norm TIED Model'),
    ('vbmc_time_vary_norm_tied_results', 'Time-Varying Norm TIED Model'),
]

# First, collect all TIED model values to compute global y-limits
all_tied_elbos = []
all_tied_elbo_sds = []
all_tied_loglikes = []
for row, (model_key, _) in enumerate(model_order[1:], 1):
    for animal_id in animal_ids:
        pkl_path = os.path.join(RESULTS_DIR, PKL_PATTERN.format(animal_id))
        if not os.path.exists(pkl_path):
            continue
        with open(pkl_path, 'rb') as f:
            results = pickle.load(f)
        if model_key not in results:
            continue
        all_tied_elbos.append(results[model_key].get('elbo', np.nan))
        all_tied_elbo_sds.append(results[model_key].get('elbo_sd', 0.0))
        all_tied_loglikes.append(results[model_key].get('loglike', np.nan))

tied_vals = np.array(all_tied_elbos + all_tied_loglikes)
tied_finite = tied_vals[np.isfinite(tied_vals)]
if len(tied_finite) > 0:
    tied_min, tied_max = np.min(tied_finite), np.max(tied_finite)
    tied_pad = 0.05 * (tied_max - tied_min)
    tied_min -= tied_pad
    tied_max += tied_pad
else:
    tied_min, tied_max = None, None

for row, (model_key, model_title) in enumerate(model_order):
    elbos = []
    elbo_sds = []
    loglikes = []
    valid_animals = []
    for animal_id in animal_ids:
        pkl_path = os.path.join(RESULTS_DIR, PKL_PATTERN.format(animal_id))
        if not os.path.exists(pkl_path):
            continue
        with open(pkl_path, 'rb') as f:
            results = pickle.load(f)
        if model_key not in results:
            continue
        elbos.append(results[model_key].get('elbo', np.nan))
        elbo_sds.append(results[model_key].get('elbo_sd', 0.0))
        loglikes.append(results[model_key].get('loglike', np.nan))
        valid_animals.append(animal_id)
    x = np.arange(len(valid_animals))
    # Get y-limits for aborts (row 0) and use tied_min/tied_max for others
    if row == 0:
        all_vals = np.array(elbos + loglikes)
        finite_vals = all_vals[np.isfinite(all_vals)]
        if len(finite_vals) > 0:
            min_y, max_y = np.min(finite_vals), np.max(finite_vals)
            pad = 0.05 * (max_y - min_y)
            min_y -= pad
            max_y += pad
        else:
            min_y, max_y = None, None
    else:
        min_y, max_y = tied_min, tied_max
    # ELBO bar plot with error bars
    ax_elbo = axes[row, 0]
    ax_elbo.bar(x, elbos, yerr=elbo_sds, color='royalblue', alpha=0.8, capsize=6, edgecolor='black')
    ax_elbo.set_ylabel(model_title)
    ax_elbo.set_title('ELBO vs Animal')
    ax_elbo.set_xticks(x)
    ax_elbo.set_xticklabels([str(a) for a in valid_animals])
    if row == 3:
        ax_elbo.set_xlabel('Animal')
    if min_y is not None and max_y is not None:
        ax_elbo.set_ylim([min_y, max_y])
    # loglike bar plot
    ax_loglike = axes[row, 1]
    ax_loglike.bar(x, loglikes, color='firebrick', alpha=0.8, edgecolor='black')
    ax_loglike.set_title('Loglike vs Animal')
    ax_loglike.set_xticks(x)
    ax_loglike.set_xticklabels([str(a) for a in valid_animals])
    if row == 3:
        ax_loglike.set_xlabel('Animal')
    if min_y is not None and max_y is not None:
        ax_loglike.set_ylim([min_y, max_y])
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'compare_animals_elbo_loglike.pdf'))
print('Saved: compare_animals_elbo_loglike.pdf')
plt.close(fig)
