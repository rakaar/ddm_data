import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.dirname(__file__)
ANIMAL_IDS = []

# Get animal IDs present in both non-shabby and shabby files
for fname in os.listdir(RESULTS_DIR):
    if fname.startswith('results_Comparable_animal_') and fname.endswith('.pkl') and not fname.startswith('shabby'):
        try:
            animal_id = int(fname.split('_')[-1].replace('.pkl', ''))
            shabby = f'shabby_code_results_Comparable_animal_{animal_id}.pkl'
            if os.path.exists(os.path.join(RESULTS_DIR, shabby)):
                ANIMAL_IDS.append(animal_id)
        except Exception:
            continue
ANIMAL_IDS = sorted(ANIMAL_IDS)

# Model order and titles
MODEL_ORDER = [
    ('vbmc_aborts_results', 'Aborts Model'),
    ('vbmc_vanilla_tied_results', 'Vanilla TIED Model'),
    ('vbmc_norm_tied_results', 'Norm TIED Model'),
    ('vbmc_time_vary_norm_tied_results', 'Time-Varying Norm TIED Model'),
]

# Data storage: model x animal x [nonshabby, shabby] x [elbo, elbo_sd, loglike]
data = {}
for model_key, model_title in MODEL_ORDER:
    data[model_key] = {'elbo': [], 'elbo_sd': [], 'loglike': [], 'elbo_shabby': [], 'elbo_sd_shabby': [], 'loglike_shabby': []}
    for animal_id in ANIMAL_IDS:
        # Non-shabby
        pkl_path = os.path.join(RESULTS_DIR, f'results_Comparable_animal_{animal_id}.pkl')
        with open(pkl_path, 'rb') as f:
            results = pickle.load(f)
        if model_key in results:
            data[model_key]['elbo'].append(results[model_key].get('elbo', np.nan))
            data[model_key]['elbo_sd'].append(results[model_key].get('elbo_sd', 0.0))
            data[model_key]['loglike'].append(results[model_key].get('loglike', np.nan))
        else:
            data[model_key]['elbo'].append(np.nan)
            data[model_key]['elbo_sd'].append(0.0)
            data[model_key]['loglike'].append(np.nan)
        # Shabby
        pkl_path_shabby = os.path.join(RESULTS_DIR, f'shabby_code_results_Comparable_animal_{animal_id}.pkl')
        with open(pkl_path_shabby, 'rb') as f:
            results_shabby = pickle.load(f)
        if model_key in results_shabby:
            data[model_key]['elbo_shabby'].append(results_shabby[model_key].get('elbo', np.nan))
            data[model_key]['elbo_sd_shabby'].append(results_shabby[model_key].get('elbo_sd', 0.0))
            data[model_key]['loglike_shabby'].append(results_shabby[model_key].get('loglike', np.nan))
        else:
            data[model_key]['elbo_shabby'].append(np.nan)
            data[model_key]['elbo_sd_shabby'].append(0.0)
            data[model_key]['loglike_shabby'].append(np.nan)

# Plotting
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 16), sharex=True)
bar_width = 0.35
x = np.arange(len(ANIMAL_IDS))

for row, (model_key, model_title) in enumerate(MODEL_ORDER):
    # ELBO plot
    ax_elbo = axes[row, 0]
    elbo = np.array(data[model_key]['elbo'])
    elbo_sd = np.array(data[model_key]['elbo_sd'])
    elbo_shabby = np.array(data[model_key]['elbo_shabby'])
    elbo_sd_shabby = np.array(data[model_key]['elbo_sd_shabby'])
    ax_elbo.bar(x - bar_width/2, elbo, bar_width, yerr=elbo_sd, label='run1', color='#4C72B0', alpha=0.8, capsize=6, edgecolor='black')
    ax_elbo.bar(x + bar_width/2, elbo_shabby, bar_width, yerr=elbo_sd_shabby, label='run2', color='#DD8452', alpha=0.8, capsize=6, edgecolor='black')
    if row == 0:
        ax_elbo.legend()
    ax_elbo.set_ylabel(model_title)
    if row == 0:
        ax_elbo.set_title('ELBO')
    ax_elbo.set_xticks(x)
    if row == 3:
        ax_elbo.set_xticklabels([str(a) for a in ANIMAL_IDS])
    else:
        ax_elbo.set_xticklabels([])
    # Loglike plot
    ax_loglike = axes[row, 1]
    loglike = np.array(data[model_key]['loglike'])
    loglike_shabby = np.array(data[model_key]['loglike_shabby'])
    ax_loglike.bar(x - bar_width/2, loglike, bar_width, label='run1', color='#4C72B0', alpha=0.8, edgecolor='black')
    ax_loglike.bar(x + bar_width/2, loglike_shabby, bar_width, label='run2', color='#DD8452', alpha=0.8, edgecolor='black')
    if row == 0:
        ax_loglike.legend()
    if row == 0:
        ax_loglike.set_title('Loglike')
    ax_loglike.set_xticks(x)
    if row == 3:
        ax_loglike.set_xticklabels([str(a) for a in ANIMAL_IDS])
    else:
        ax_loglike.set_xticklabels([])

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'compare_shabby_vs_nonshabby.pdf'))
print('Saved: compare_shabby_vs_nonshabby.pdf')
