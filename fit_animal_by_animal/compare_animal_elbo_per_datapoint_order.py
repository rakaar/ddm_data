# %%
# param vs animal wise pdfs (ordered by ELBO per data point)

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

# Model configs: (model_key, param_keys, param_labels, plot_title, fit_type)
model_configs = [
    ('vbmc_aborts_results',
        ['V_A_samples', 'theta_A_samples', 't_A_aff_samp'],
        ['V_A', 'theta_A', 't_A_aff'],
        'Aborts Model',
        'aborts_len'),
    ('vbmc_vanilla_tied_results',
        ['rate_lambda_samples', 'T_0_samples', 'theta_E_samples', 'w_samples', 't_E_aff_samples', 'del_go_samples'],
        ['rate_lambda', 'T_0', 'theta_E', 'w', 't_E_aff', 'del_go'],
        'Vanilla TIED Model',
        'valid_len'),
    ('vbmc_norm_tied_results',
        ['rate_lambda_samples', 'T_0_samples', 'theta_E_samples', 'w_samples', 't_E_aff_samples', 'del_go_samples', 'rate_norm_l_samples'],
        ['rate_lambda', 'T_0', 'theta_E', 'w', 't_E_aff', 'del_go', 'rate_norm_l'],
        'Norm TIED Model',
        'valid_len'),
    ('vbmc_time_vary_norm_tied_results',
        ['rate_lambda_samples', 'T_0_samples', 'theta_E_samples', 'w_samples', 't_E_aff_samples', 'del_go_samples', 'rate_norm_l_samples', 'bump_height_samples', 'bump_width_samples', 'dip_height_samples', 'dip_width_samples'],
        ['rate_lambda', 'T_0', 'theta_E', 'w', 't_E_aff', 'del_go', 'rate_norm_l', 'bump_height', 'bump_width', 'dip_height', 'dip_width'],
        'Time-Varying Norm TIED Model',
        'valid_len'),
]

def format_sci(val):
    # Format a float in scientific notation as 2.23 x 10^5
    if not np.isfinite(val):
        return 'NaN'
    exp = int(np.floor(np.log10(abs(val)))) if val != 0 else 0
    mant = val / (10 ** exp) if val != 0 else 0
    return f"{mant:.2f} x 10^{exp}"

def format_label(label, elbo, norm_elbo):
    # Compose label with sci notation and normalized value (3 decimal places)
    return f"{label} ({format_sci(elbo)} = {norm_elbo:.3f})" if np.isfinite(norm_elbo) else f"{label} ({format_sci(elbo)})"

def load_animal_stats(batch):
    stats_path = os.path.join(RESULTS_DIR, 'animal_stats_pickles', f'animal_stats_{batch}.pkl')
    if not os.path.exists(stats_path):
        return None
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    return stats

for model_key, param_keys, param_labels, plot_title, fit_len_key in model_configs:
    elbos = []
    elbo_per_datapoint = []
    datapoint_counts = []
    valid_animals = []  # Will store (batch, animal_id)
    valid_labels = []   # Will store strings like 'LED7-92'
    batch_colors = []   # Color for each animal
    # First pass: gather ELBO values and 95% CI (nonparametric)
    for batch, animal_id in animal_batch_tuples:
        pkl_fname = f'results_{batch}_animal_{animal_id}.pkl'
        pkl_path = os.path.join(RESULTS_DIR, pkl_fname)
        if not os.path.exists(pkl_path):
            continue
        with open(pkl_path, 'rb') as f:
            results = pickle.load(f)
        if model_key not in results:
            continue
        # Load stats for this batch
        stats = load_animal_stats(batch)
        if stats is None or animal_id not in stats or fit_len_key not in stats[animal_id]:
            continue
        n_data = stats[animal_id][fit_len_key]
        elbo_val = results[model_key].get('elbo', np.nan)
        valid_animals.append((batch, animal_id))
        valid_labels.append(f'{batch}-{animal_id}')
        batch_colors.append(BATCH_COLORS.get(batch, 'gray'))
        elbos.append(elbo_val)
        datapoint_counts.append(n_data)
        if n_data > 0:
            elbo_per_datapoint.append(elbo_val / n_data)
        else:
            elbo_per_datapoint.append(np.nan)
    # Sort by ELBO per data point (descending)
    zipped = list(zip(elbo_per_datapoint, elbos, datapoint_counts, valid_labels, batch_colors, valid_animals))
    zipped.sort(key=lambda x: x[0], reverse=True)
    if zipped:
        columns = list(zip(*zipped))
        sorted_elbo_per_datapoint = list(columns[0])[::-1]
        sorted_elbos = list(columns[1])[::-1]
        sorted_counts = list(columns[2])[::-1]
        sorted_labels = list(columns[3])[::-1]
        sorted_colors = list(columns[4])[::-1]
        sorted_animals = list(columns[5])[::-1]
    else:
        sorted_elbo_per_datapoint = []
        sorted_elbos = []
        sorted_counts = []
        sorted_labels = []
        sorted_colors = []
        sorted_animals = []
    y_pos = np.arange(len(sorted_labels))[::-1]
    outname = f'compare_animals_elbo_per_datapoint_order_{model_key}.pdf'
    with PdfPages(os.path.join(RESULTS_DIR, outname)) as pdf:
        fig, ax = plt.subplots(figsize=(7, 6))
        for idx in range(len(sorted_labels)):
            ax.hlines(y=y_pos[idx], xmin=0, xmax=sorted_elbo_per_datapoint[idx], color=sorted_colors[idx], linewidth=3, alpha=0.7)
        # Normalize by the true maximum ELBO per data point (before reversal)
        if elbo_per_datapoint:
            max_elbo_per_datapoint = np.nanmax(elbo_per_datapoint)
        else:
            max_elbo_per_datapoint = 1
        norm_elbos = [elbo/max_elbo_per_datapoint if np.isfinite(elbo) and max_elbo_per_datapoint != 0 else np.nan for elbo in sorted_elbo_per_datapoint]
        yticks_with_elbo = [format_label(label, elbo, norm) for label, elbo, norm in zip(sorted_labels, sorted_elbo_per_datapoint, norm_elbos)]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(yticks_with_elbo)
        for ticklabel, (batch, _) in zip(ax.get_yticklabels(), sorted_animals):
            ticklabel.set_color(BATCH_COLORS.get(batch, 'gray'))
        ax.set_xlabel('ELBO per data point')
        ax.set_ylabel('Batch-Animal')
        ax.set_title(f'{plot_title}: ELBO per data point')
        ax.axvspan(0, np.max(np.array(sorted_elbo_per_datapoint)), color='#b7e4c7', alpha=0.2, zorder=-1)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    print(f'Saved: {outname}')
    for param in param_keys:
        means = []
        ci_lows = []   # 2.5th percentile
        ci_highs = []  # 97.5th percentile
        for batch, animal_id in sorted_animals:
            pkl_fname = f'results_{batch}_animal_{animal_id}.pkl'
            pkl_path = os.path.join(RESULTS_DIR, pkl_fname)
            with open(pkl_path, 'rb') as f:
                results = pickle.load(f)
            if model_key not in results:
                continue
            samples = np.asarray(results[model_key][param])
            mean = np.mean(samples)
            ci_lower, ci_upper = np.percentile(samples, [2.5, 97.5])
            means.append(mean)
            ci_lows.append(ci_lower)
            ci_highs.append(ci_upper)
        y_pos = np.arange(len(sorted_labels))
        outname = f'compare_animals_param_vs_animal_elbo_per_datapoint_order_{model_key}_{param}.pdf'
        with PdfPages(os.path.join(RESULTS_DIR, outname)) as pdf:
            fig, ax = plt.subplots(figsize=(7, 6))
            for idx in range(len(sorted_labels)):
                # Plot CI as a horizontal line
                ax.hlines(y=y_pos[idx], xmin=ci_lows[idx], xmax=ci_highs[idx], color=sorted_colors[idx], linewidth=3, alpha=0.7)
                # Plot mean as a point
                ax.plot(means[idx], y_pos[idx], 'o', color=sorted_colors[idx])
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sorted_labels)
            for ticklabel, (batch, _) in zip(ax.get_yticklabels(), sorted_animals):
                ticklabel.set_color(BATCH_COLORS.get(batch, 'gray'))
            ax.set_xlabel(param_labels[param_keys.index(param)])
            ax.set_ylabel('Batch-Animal (ELBO per data point)')
            ax.set_title(f'{plot_title}: {param_labels[param_keys.index(param)]} (mean, 95% CI)')
            ax.axvspan(0, np.max(np.array(ci_highs)), color='#b7e4c7', alpha=0.2, zorder=-1)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        print(f'Saved: {outname}')
