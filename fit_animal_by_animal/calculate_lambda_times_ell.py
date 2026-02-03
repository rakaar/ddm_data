# %%
import os
import pickle

import numpy as np
import pandas as pd


RESULTS_DIR = os.path.dirname(__file__)
DESIRED_BATCHES = ['SD', 'LED34', 'LED6', 'LED8', 'LED7', 'LED34_even']
MODEL_KEY = 'vbmc_norm_tied_results'


def build_animal_batch_tuples():
    batch_dir = os.path.join(RESULTS_DIR, 'batch_csvs')
    batch_files = [f'batch_{batch_name}_valid_and_aborts.csv' for batch_name in DESIRED_BATCHES]
    dfs = []
    for fname in batch_files:
        fpath = os.path.join(batch_dir, fname)
        if os.path.exists(fpath):
            dfs.append(pd.read_csv(fpath))
    if len(dfs) > 0:
        merged_data = pd.concat(dfs, ignore_index=True)
        merged_valid = merged_data[merged_data['success'].isin([1, -1])].copy()
        batch_animal_pairs = sorted(list(map(tuple, merged_valid[['batch_name', 'animal']].drop_duplicates().values)))
    else:
        batch_animal_pairs = []

    animal_batch_tuples = []
    if batch_animal_pairs:
        for batch, animal in batch_animal_pairs:
            try:
                animal_id = int(animal)
            except Exception:
                continue
            fname = f'results_{batch}_animal_{animal_id}.pkl'
            pkl_path = os.path.join(RESULTS_DIR, fname)
            if os.path.exists(pkl_path):
                animal_batch_tuples.append((batch, animal_id))
    else:
        for fname in os.listdir(RESULTS_DIR):
            if fname.startswith('results_') and fname.endswith('.pkl'):
                for batch in DESIRED_BATCHES:
                    prefix = f'results_{batch}_animal_'
                    if fname.startswith(prefix):
                        try:
                            animal_id = int(fname.split('_')[-1].replace('.pkl', ''))
                            animal_batch_tuples.append((batch, animal_id))
                        except Exception:
                            continue

    return sorted(animal_batch_tuples, key=lambda x: (x[0], x[1]))



animal_batch_tuples = build_animal_batch_tuples()
per_animal_means = []

for batch, animal_id in animal_batch_tuples:
    pkl_fname = f'results_{batch}_animal_{animal_id}.pkl'
    pkl_path = os.path.join(RESULTS_DIR, pkl_fname)
    if not os.path.exists(pkl_path):
        continue
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)
    if MODEL_KEY not in results:
        continue

    rate_lambda = np.asarray(results[MODEL_KEY].get('rate_lambda_samples', []), dtype=float).ravel()
    rate_norm_l = np.asarray(results[MODEL_KEY].get('rate_norm_l_samples', []), dtype=float).ravel()
    n = min(rate_lambda.size, rate_norm_l.size)
    if n == 0:
        continue

    product = rate_lambda[:n] * (1.0 - rate_norm_l[:n])
    product = product[np.isfinite(product)]
    if product.size == 0:
        continue

    per_animal_means.append(float(np.mean(product)))

if len(per_animal_means) == 0:
    print('No animals found with norm tied results and valid samples.')

avg_value = float(np.mean(per_animal_means))
print(f'Average lambda*(1 - rate_norm_l) across animals: {avg_value:.6g}')

# %%
import matplotlib.pyplot as plt
plt.hist(per_animal_means, bins=np.arange(0.05, 0.20, 0.01), label=f'mean={avg_value:.6f}');
plt.xlabel('lambda*(1 - rate_norm_l)');
plt.ylabel('Count');
plt.title('Distribution of lambda*(1 - rate_norm_l) across animals');
plt.legend()
plt.show();

