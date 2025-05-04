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

# Define TIED models
TIED_MODELS = [
    ('vbmc_vanilla_tied_results', 'Vanilla TIED'),
    ('vbmc_norm_tied_results', 'Norm TIED'),
    ('vbmc_time_vary_norm_tied_results', 'Time-Varying Norm TIED'),
]

# Collect loglike and elbo for each animal and each TIED model
loglike_dict = {label: [] for _, label in TIED_MODELS}
elbo_dict = {label: [] for _, label in TIED_MODELS}
valid_animals = []

for animal_id in animal_ids:
    has_all = True
    vals_loglike = []
    vals_elbo = []
    for model_key, label in TIED_MODELS:
        pkl_path = os.path.join(RESULTS_DIR, PKL_PATTERN.format(animal_id))
        if not os.path.exists(pkl_path):
            has_all = False
            break
        with open(pkl_path, 'rb') as f:
            results = pickle.load(f)
        if model_key not in results:
            has_all = False
            break
        vals_loglike.append(results[model_key].get('loglike', np.nan))
        vals_elbo.append(results[model_key].get('elbo', np.nan))
    if has_all:
        valid_animals.append(animal_id)
        for i, (_, label) in enumerate(TIED_MODELS):
            loglike_dict[label].append(vals_loglike[i])
            elbo_dict[label].append(vals_elbo[i])

x = np.arange(len(valid_animals))
bar_width = 0.25
colors = ['royalblue', 'seagreen', 'darkorange']
labels = [label for _, label in TIED_MODELS]

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Loglike grouped bar plot
for i, label in enumerate(labels):
    axes[0].bar(x + i*bar_width - bar_width, loglike_dict[label], width=bar_width, label=label, color=colors[i], alpha=0.85, edgecolor='black')
axes[0].set_ylabel('Loglike')
axes[0].set_title('Loglike by Animal and Model')
axes[0].legend()
axes[0].set_xticks(x)
axes[0].set_xticklabels([str(a) for a in valid_animals])
axes[0].grid(True, axis='y', which='both', linestyle='--', alpha=0.7)

# ELBO grouped bar plot
for i, label in enumerate(labels):
    axes[1].bar(x + i*bar_width - bar_width, elbo_dict[label], width=bar_width, label=label, color=colors[i], alpha=0.85, edgecolor='black')
axes[1].set_ylabel('ELBO')
axes[1].set_title('ELBO by Animal and Model')
axes[1].legend()
axes[1].set_xticks(x)
axes[1].set_xticklabels([str(a) for a in valid_animals])
axes[1].set_xlabel('Animal')
axes[1].grid(True, axis='y', which='both', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'grouped_barplot_tied_models.pdf'))
print('Saved: grouped_barplot_tied_models.pdf')
plt.close(fig)
