import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import corner

# Directory containing the results
RESULTS_DIR = os.path.dirname(__file__)
BATCHES = ['Comparable', 'SD', 'LED2', 'LED1', 'LED34', 'LED7']

# Define simple, high-contrast colors for each batch (same as compare_animal_params.py)
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

# Refactored script to generate corner plots for all models (aborts, vanilla tied, norm tied, time-varying norm tied)
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import corner
import matplotlib.patches as mpatches

# Directory containing the results
RESULTS_DIR = os.path.dirname(__file__)
BATCHES = ['Comparable', 'SD', 'LED2', 'LED1', 'LED34', 'LED7']
BATCH_COLORS = {
    'Comparable': 'red',
    'SD': '#87CEEB',  # sky blue
    'LED2': 'green',
    'LED1': 'orange',
    'LED34': 'purple',
    'LED7': 'black',
}

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

# Find all animal pickle files from all batches
animal_batch_tuples = []  # List of (batch, animal_number)
for fname in os.listdir(RESULTS_DIR):
    if fname.startswith('results_') and fname.endswith('.pkl'):
        for batch in BATCHES:
            prefix = f'results_{batch}_animal_'
            if fname.startswith(prefix):
                try:
                    animal_id = int(fname.split('_')[-1].replace('.pkl', ''))
                    animal_batch_tuples.append((batch, animal_id))
                except Exception:
                    continue
animal_batch_tuples = sorted(animal_batch_tuples, key=lambda x: (x[0], x[1]))

# Helper to get message for a given model/animal
def get_model_message(results, model_key):
    if model_key in results and isinstance(results[model_key], dict):
        return results[model_key].get('message', None)
    return None

def gather_samples_for_model(model_key, param_keys):
    """Return list of (samples, batch, animal_label, message) for all animals for a given model."""
    animal_samples = []
    for batch, animal_id in animal_batch_tuples:
        pkl_fname = f'results_{batch}_animal_{animal_id}.pkl'
        pkl_path = os.path.join(RESULTS_DIR, pkl_fname)
        if not os.path.exists(pkl_path):
            continue
        with open(pkl_path, 'rb') as f:
            results = pickle.load(f)
        if model_key not in results:
            continue
        try:
            samples = np.column_stack([
                np.asarray(results[model_key][k]) for k in param_keys
            ])
            if samples.shape[0] > 2500:
                idxs = np.random.choice(samples.shape[0], 2500, replace=False)
                samples = samples[idxs]
        except Exception as e:
            print(f"Skipping {pkl_fname} for model {model_key}: {e}")
            continue
        message = get_model_message(results, model_key)
        animal_samples.append((samples, batch, f'{batch}-{animal_id}', message))
    return animal_samples

# Helper to plot and save corner plots for a model
def plot_corner_for_model(animal_samples, param_labels, model_title, pdf_pages, converged_only=False):
    corner_fig = None
    for samples, batch, animal_label, message in animal_samples:
        if converged_only and message is not None and "reached maximum number of function evaluations options.max_fun_evals" in message:
            continue
        if samples.shape[0] < 50:
            continue
        corner_fig = corner.corner(
            samples,
            labels=param_labels,
            color=BATCH_COLORS[batch],
            label_kwargs=dict(fontsize=12),
            show_titles=False,
            title_fmt='.3f',
            plot_datapoints=False,
            fill_contours=False,
            plot_density=True,
            hist_kwargs=dict(color=BATCH_COLORS[batch], alpha=0.3, linewidth=1),
            contour_kwargs=dict(colors=[BATCH_COLORS[batch]], linewidths=1, alpha=0.5),
            fig=corner_fig,
            max_n_ticks=4,
            smooth=1.0,
        )
    if corner_fig is not None:
        corner_fig.suptitle(f'Corner Plot: Per-Animal Densities ({model_title})', fontsize=16, y=1.01)
        pdf_pages.savefig(corner_fig, bbox_inches='tight')
        plt.close(corner_fig)

# Save color legend as PNG
legend_fig = plt.figure(figsize=(6, 2))
legend_handles = [mpatches.Patch(color=color, label=batch) for batch, color in BATCH_COLORS.items()]
plt.legend(handles=legend_handles, loc='center', ncol=3, fontsize=12)
plt.axis('off')
legend_fig.tight_layout()
legend_png_path = os.path.join(RESULTS_DIR, 'cornerplot_batch_legend.png')
plt.savefig(legend_png_path, bbox_inches='tight', dpi=200)
plt.close(legend_fig)
print(f"Saved legend PNG: {legend_png_path}")

# Generate PDFs for all models
for converged_only, pdf_name in zip([False, True], ['all_animals.pdf', 'converged_animals.pdf']):
    pdf_path = os.path.join(RESULTS_DIR, pdf_name)
    with PdfPages(pdf_path) as pdf:
        for model_key, param_keys, param_labels, model_title in model_configs:
            print(f"Processing {model_title} ({'Converged only' if converged_only else 'All animals'})...")
            animal_samples = gather_samples_for_model(model_key, param_keys)
            plot_corner_for_model(animal_samples, param_labels, model_title, pdf, converged_only=converged_only)
    print(f"Saved: {pdf_path}")
