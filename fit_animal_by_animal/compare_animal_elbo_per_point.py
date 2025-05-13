# %%
# param vs animal wise pdfs (ordered by ELBO per datapoint)

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

# Load animal stats dictionaries from pickle files
animal_stats = {}
ANIMAL_STATS_DIR = os.path.join(RESULTS_DIR, 'animal_stats_pickles')

# Load stats for each batch from the corresponding pickle file
for batch in BATCHES:
    stats_file = os.path.join(ANIMAL_STATS_DIR, f'animal_stats_{batch}.pkl')
    if os.path.exists(stats_file):
        with open(stats_file, 'rb') as f:
            batch_stats = pickle.load(f)
            animal_stats[batch] = batch_stats
    else:
        print(f"Warning: Stats file not found for batch {batch}: {stats_file}")
        animal_stats[batch] = {}


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

def format_sci(val):
    # Format a float in scientific notation as 2.23 x 10^5
    if not np.isfinite(val):
        return 'NaN'
    exp = int(np.floor(np.log10(abs(val)))) if val != 0 else 0
    mant = val / (10 ** exp) if val != 0 else 0
    return f"{mant:.2f} x 10^{exp}"

def format_label(label, value, norm_value, is_elbo_per_point=False):
    # Compose label with sci notation and normalized value (3 decimal places)
    if is_elbo_per_point:
        return f"{label} ({value:.3f})" if np.isfinite(value) else f"{label} (NaN)"
    else:
        return f"{label} ({format_sci(value)} = {norm_value:.3f})" if np.isfinite(norm_value) else f"{label} ({format_sci(value)})"

for model_key, param_keys, param_labels, plot_title in model_configs:
    means = {param: [] for param in param_keys}
    ci_lows = {param: [] for param in param_keys}   # 2.5th percentile
    ci_highs = {param: [] for param in param_keys}  # 97.5th percentile
    valid_animals = []  # Will store (batch, animal_id)
    valid_labels = []   # Will store strings like 'LED7-92'
    batch_colors = []   # Color for each animal
    loglikes = []
    elbos = []
    elbo_per_points = []  # Will store ELBO per datapoint
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
        loglikes.append(results[model_key].get('loglike', np.nan))
        elbo = results[model_key].get('elbo', np.nan)
        elbos.append(elbo)
        
        # Calculate ELBO per datapoint
        if model_key == 'vbmc_aborts_results':
            # For aborts model, use aborts_len
            n_datapoints = animal_stats.get(batch, {}).get(animal_id, {}).get('aborts_len', 0)
        else:
            # For TIED models, use valid_len
            n_datapoints = animal_stats.get(batch, {}).get(animal_id, {}).get('valid_len', 0)
        
        elbo_per_point = elbo / n_datapoints if n_datapoints > 0 and np.isfinite(elbo) else np.nan
        elbo_per_points.append(elbo_per_point)
        for param in param_keys:
            samples = np.asarray(results[model_key][param])
            mean = np.mean(samples)
            ci_lower, ci_upper = np.percentile(samples, [2.5, 97.5])
            means[param].append(mean)
            ci_lows[param].append(ci_lower)
            ci_highs[param].append(ci_upper)

    from matplotlib.backends.backend_pdf import PdfPages
    outname = f'compare_animals_elbo_per_point_order_{model_key}.pdf'
    with PdfPages(os.path.join(RESULTS_DIR, outname)) as pdf:
        for i, param in enumerate(param_keys):
            # Sort by ELBO per datapoint value (descending)
            zipped = list(zip(elbo_per_points, means[param], ci_lows[param], ci_highs[param], valid_labels, batch_colors, valid_animals, loglikes))
            zipped.sort(key=lambda x: x[0], reverse=True)
            sorted_elbo_per_points, sorted_means, sorted_ci_lows, sorted_ci_highs, sorted_labels, sorted_colors, sorted_animals, sorted_loglikes = zip(*zipped)
            # Reverse so highest ELBO per datapoint is at the top
            sorted_elbo_per_points = list(sorted_elbo_per_points)[::-1]
            sorted_loglikes = list(sorted_loglikes)[::-1]
            sorted_means = list(sorted_means)[::-1]
            sorted_ci_lows = list(sorted_ci_lows)[::-1]
            sorted_ci_highs = list(sorted_ci_highs)[::-1]
            sorted_labels = list(sorted_labels)[::-1]
            sorted_colors = list(sorted_colors)[::-1]
            sorted_animals = list(sorted_animals)[::-1]
            y_pos = np.arange(len(sorted_labels))
            fig, ax = plt.subplots(figsize=(7, 6))
            for idx in range(len(sorted_labels)):
                # Plot CI as a horizontal line
                ax.hlines(y=y_pos[idx], xmin=sorted_ci_lows[idx], xmax=sorted_ci_highs[idx], color=sorted_colors[idx], linewidth=3, alpha=0.7)
                # Plot mean as a point
                ax.plot(sorted_means[idx], y_pos[idx], 'o', color=sorted_colors[idx])
            # Compose y-tick labels with ELBO per datapoint
            yticks_with_elbo_per_point = [format_label(label, elbo_per_point, 0, is_elbo_per_point=True) for label, elbo_per_point in zip(sorted_labels, sorted_elbo_per_points)]
            ax.set_yticks(y_pos)
            ax.set_yticklabels(yticks_with_elbo_per_point)
            # Color y-tick labels by batch
            for ticklabel, (batch, _) in zip(ax.get_yticklabels(), sorted_animals):
                ticklabel.set_color(BATCH_COLORS.get(batch, 'gray'))
            ax.set_xlabel(param_labels[i])
            ax.set_ylabel('Batch-Animal (ELBO per datapoint)')
            ax.set_title(f'{plot_title}: {param_labels[i]} (mean, 95% CI)')
            ax.axvspan(0, np.max(np.array(sorted_ci_highs)), color='#b7e4c7', alpha=0.2, zorder=-1)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(f'Saved: {outname}')

# %%
# elbo, loglike (ordered by ELBO per datapoint)

fig, axes = plt.subplots(4, 2, figsize=(12, 12), sharex=True)

# Model order for this plot (to match rows)
model_order = [
    ('vbmc_aborts_results', 'Aborts Model'),
    ('vbmc_vanilla_tied_results', 'Vanilla TIED Model'),
    ('vbmc_norm_tied_results', 'Norm TIED Model'),
    ('vbmc_time_vary_norm_tied_results', 'Time-Varying Norm TIED Model'),
]

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
    # Calculate ELBO per datapoint for each animal
    elbo_per_points = []
    for i, (batch, animal_id) in enumerate(valid_animals):
        if model_key == 'vbmc_aborts_results':
            # For aborts model, use aborts_len
            n_datapoints = animal_stats.get(batch, {}).get(animal_id, {}).get('aborts_len', 0)
        else:
            # For TIED models, use valid_len
            n_datapoints = animal_stats.get(batch, {}).get(animal_id, {}).get('valid_len', 0)
        
        elbo_per_point = elbos[i] / n_datapoints if n_datapoints > 0 and np.isfinite(elbos[i]) else np.nan
        elbo_per_points.append(elbo_per_point)
    
    # Sort by ELBO per datapoint
    zipped = list(zip(elbo_per_points, elbos, elbo_sds, loglikes, valid_labels, valid_animals))
    zipped.sort(key=lambda x: x[0], reverse=True)
    sorted_elbo_per_points, sorted_elbos, sorted_elbo_sds, sorted_loglikes, sorted_labels, sorted_animals = zip(*zipped) if zipped else ([],[],[],[],[],[])
    # Reverse so highest ELBO per datapoint is at the top/left
    sorted_elbo_per_points = list(sorted_elbo_per_points)[::-1]
    sorted_loglikes = list(sorted_loglikes)[::-1]
    sorted_elbos = list(sorted_elbos)[::-1]
    sorted_elbo_sds = list(sorted_elbo_sds)[::-1]
    sorted_labels = list(sorted_labels)[::-1]
    sorted_animals = list(sorted_animals)[::-1]
    x = np.arange(len(sorted_labels))
    # ELBO bar plot with error bars
    ax_elbo = axes[row, 0]
    ax_elbo.bar(x, sorted_elbos, yerr=sorted_elbo_sds, color='royalblue', alpha=0.8, capsize=6, edgecolor='black')
    ax_elbo.set_ylabel(model_title)
    ax_elbo.set_title('ELBO vs Batch-Animal (ELBO per datapoint order)')
    ax_elbo.set_xticks(x)
    # Compose x-tick labels with ELBO per datapoint
    xticks_with_elbo_per_point = [format_label(label, elbo_per_point, 0, is_elbo_per_point=True) for label, elbo_per_point in zip(sorted_labels, sorted_elbo_per_points)]
    ax_elbo.set_xticklabels(xticks_with_elbo_per_point, rotation=45, ha='right')

    if row == 3:
        ax_elbo.set_xlabel('Batch-Animal (ELBO per datapoint)')
    # loglike bar plot
    ax_loglike = axes[row, 1]
    ax_loglike.bar(x, sorted_loglikes, color='firebrick', alpha=0.8, edgecolor='black')
    ax_loglike.set_title('Loglike vs Batch-Animal (ELBO per datapoint order)')
    ax_loglike.set_xticks(x)
    ax_loglike.set_xticklabels(xticks_with_elbo_per_point, rotation=45, ha='right')
    if row == 3:
        ax_loglike.set_xlabel('Batch-Animal (ELBO per datapoint)')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'compare_animals_elbo_loglike_elbo_per_point_order.pdf'))
print('Saved: compare_animals_elbo_loglike_elbo_per_point_order.pdf')
plt.close(fig)
