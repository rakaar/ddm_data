import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.dirname(__file__)
PKL_PATTERN = 'results_Comparable_animal_{}.pkl'

# Choose the animal of interest
animal_id = 39

TIED_MODELS = [
    ('vbmc_vanilla_tied_results', 'Vanilla TIED'),
    ('vbmc_norm_tied_results', 'Norm TIED'),
    ('vbmc_time_vary_norm_tied_results', 'Time-Varying Norm TIED'),
]

loglikes = []
elbos = []
labels = []

pkl_path = os.path.join(RESULTS_DIR, PKL_PATTERN.format(animal_id))
if not os.path.exists(pkl_path):
    raise FileNotFoundError(f"Pickle file not found for animal {animal_id}")
with open(pkl_path, 'rb') as f:
    results = pickle.load(f)

print(f"Animal {animal_id} TIED model loglike/elbo values:")
for model_key, label in TIED_MODELS:
    if model_key in results:
        loglike = results[model_key].get('loglike', np.nan)
        elbo = results[model_key].get('elbo', np.nan)
        loglikes.append(loglike)
        elbos.append(elbo)
        labels.append(label)
        print(f"  {label}: loglike = {loglike}, elbo = {elbo}")
    else:
        loglikes.append(np.nan)
        elbos.append(np.nan)
        labels.append(label)
        print(f"  {label}: loglike = NaN, elbo = NaN (not found)")

# Plot
fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
x = np.arange(len(labels))

# Loglike
axes[0].bar(x, loglikes, color=['royalblue', 'seagreen', 'darkorange'], alpha=0.85, edgecolor='black')
axes[0].set_ylabel('Loglike')
axes[0].set_title(f'Loglike for Animal {animal_id}')
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels)
axes[0].axhline(4000, color='red', linestyle='--', label='Reference: 4000')
axes[0].legend()

# ELBO
axes[1].bar(x, elbos, color=['royalblue', 'seagreen', 'darkorange'], alpha=0.85, edgecolor='black')
axes[1].set_ylabel('ELBO')
axes[1].set_title(f'ELBO for Animal {animal_id}')
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels)
axes[1].axhline(4000, color='red', linestyle='--', label='Reference: 4000')
axes[1].legend()
axes[1].set_xlabel('Model')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, f'single_animal_{animal_id}_tied_loglike_elbo.pdf'))
print(f'Saved: single_animal_{animal_id}_tied_loglike_elbo.pdf')
plt.close(fig)
