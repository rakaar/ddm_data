# %%
# param vs animal wise pdfs

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Directory containing the results
RESULTS_DIR = os.path.dirname(__file__)
BATCHES = ['Comparable', 'SD', 'LED2', 'LED1', 'LED34', 'LED7']

# Define simple, high-contrast colors for each batch
BATCH_COLORS = {
    'Comparable': 'red',
    'SD': '#87CEEB',  # sky blue
    'LED2': 'green',
    'LED1': 'orange',
    'LED34': 'purple',
    'LED7': 'black',
}

# Find all animal pickle files from all batches
animal_batch_tuples = []  # List of (batch, animal_number)
pkl_files = []  # List of (batch, animal_number, filename)
for fname in os.listdir(RESULTS_DIR):
    if fname.startswith('results_') and fname.endswith('.pkl'):
        for batch in BATCHES:
            prefix = f'results_{batch}_animal_'
            if fname.startswith(prefix):
                try:
                    animal_id = int(fname.split('_')[-1].replace('.pkl', ''))
                    animal_batch_tuples.append((batch, animal_id))
                    pkl_files.append((batch, animal_id, fname))
                except Exception:
                    continue
# Sort by batch then animal number
animal_batch_tuples = sorted(animal_batch_tuples, key=lambda x: (x[0], x[1]))

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
    ci_lows = {param: [] for param in param_keys}   # 2.5th percentile
    ci_highs = {param: [] for param in param_keys}  # 97.5th percentile
    valid_animals = []  # Will store (batch, animal_id)
    valid_labels = []   # Will store strings like 'LED7-92'
    batch_colors = []   # Color for each animal
    # First pass: gather means and 95% CI (nonparametric)
    for batch, animal_id in animal_batch_tuples:
        pkl_fname = f'results_{batch}_animal_{animal_id}.pkl'
        pkl_path = os.path.join(RESULTS_DIR, pkl_fname)
        if not os.path.exists(pkl_path):
            continue
        with open(pkl_path, 'rb') as f:
            results = pickle.load(f)
        if model_key not in results:
            continue
        valid_animals.append((batch, animal_id))
        valid_labels.append(f'{batch}-{animal_id}')
        batch_colors.append(BATCH_COLORS.get(batch, 'gray'))
        for param in param_keys:
            samples = np.asarray(results[model_key][param])
            mean = np.mean(samples)
            ci_lower, ci_upper = np.percentile(samples, [2.5, 97.5])
            means[param].append(mean)
            ci_lows[param].append(ci_lower)
            ci_highs[param].append(ci_upper)

    from matplotlib.backends.backend_pdf import PdfPages
    outname = f'compare_animals_all_batches_{model_key}.pdf'
    with PdfPages(os.path.join(RESULTS_DIR, outname)) as pdf:
        for i, param in enumerate(param_keys):
            # Sort by mean value (descending)
            zipped = list(zip(means[param], ci_lows[param], ci_highs[param], valid_labels, batch_colors, valid_animals))
            zipped.sort(key=lambda x: x[0], reverse=True)
            sorted_means, sorted_ci_lows, sorted_ci_highs, sorted_labels, sorted_colors, sorted_animals = zip(*zipped)
            y_pos = np.arange(len(sorted_labels))
            fig, ax = plt.subplots(figsize=(7, 6))
            for idx in range(len(sorted_labels)):
                # Plot CI as a horizontal line
                ax.hlines(y=y_pos[idx], xmin=sorted_ci_lows[idx], xmax=sorted_ci_highs[idx], color=sorted_colors[idx], linewidth=3, alpha=0.7)
                # Plot mean as a point
                ax.plot(sorted_means[idx], y_pos[idx], 'o', color=sorted_colors[idx])
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sorted_labels)
            # Color y-tick labels by batch
            for ticklabel, (batch, _) in zip(ax.get_yticklabels(), sorted_animals):
                ticklabel.set_color(BATCH_COLORS.get(batch, 'gray'))
            ax.set_xlabel(param_labels[i])
            ax.set_ylabel('Batch-Animal')
            ax.set_title(f'{plot_title}: {param_labels[i]} (mean, 95% CI)')
            ax.axvspan(0, np.max(np.array(sorted_ci_highs)), color='#b7e4c7', alpha=0.2, zorder=-1)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(f'Saved: {outname}')

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
    for batch, animal_id in animal_batch_tuples:
        pkl_fname = f'results_{batch}_animal_{animal_id}.pkl'
        pkl_path = os.path.join(RESULTS_DIR, pkl_fname)
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
    valid_animals = []  # (batch, animal_id)
    valid_labels = []   # e.g. LED7-92
    for batch, animal_id in animal_batch_tuples:
        pkl_fname = f'results_{batch}_animal_{animal_id}.pkl'
        pkl_path = os.path.join(RESULTS_DIR, pkl_fname)
        if not os.path.exists(pkl_path):
            continue
        with open(pkl_path, 'rb') as f:
            results = pickle.load(f)
        if model_key not in results:
            continue
        elbos.append(results[model_key].get('elbo', np.nan))
        elbo_sds.append(results[model_key].get('elbo_sd', 0.0))
        loglikes.append(results[model_key].get('loglike', np.nan))
        valid_animals.append((batch, animal_id))
        valid_labels.append(f'{batch}-{animal_id}')
    x = np.arange(len(valid_labels))
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
    ax_elbo.set_title('ELBO vs Batch-Animal')
    ax_elbo.set_xticks(x)
    ax_elbo.set_xticklabels(valid_labels, rotation=45, ha='right')
    if row == 3:
        ax_elbo.set_xlabel('Batch-Animal')
    if min_y is not None and max_y is not None:
        ax_elbo.set_ylim([min_y, max_y])
    # loglike bar plot
    ax_loglike = axes[row, 1]
    ax_loglike.bar(x, loglikes, color='firebrick', alpha=0.8, edgecolor='black')
    ax_loglike.set_title('Loglike vs Batch-Animal')
    ax_loglike.set_xticks(x)
    ax_loglike.set_xticklabels(valid_labels, rotation=45, ha='right')
    if row == 3:
        ax_loglike.set_xlabel('Batch-Animal')
    if min_y is not None and max_y is not None:
        ax_loglike.set_ylim([min_y, max_y])
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'compare_animals_elbo_loglike_all_batches.pdf'))
print('Saved: compare_animals_elbo_loglike_all_batches.pdf')
plt.close(fig)
