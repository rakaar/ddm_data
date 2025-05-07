# %%
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from time_vary_norm_utils import phi_t_fn

# Settings
BATCHES = ["Comparable", "SD"]
RESULTS_DIR = os.path.dirname(__file__)
MODEL_KEY = "vbmc_time_vary_norm_tied_results"
PARAM_KEYS = [
    "bump_height_samples", "bump_width_samples", "dip_height_samples", "dip_width_samples"
]

# Map to phi_t_fn args: h1, a1, b1, h2, a2
PARAM_MAP = {
    "h1": "bump_width_samples",
    "a1": "bump_height_samples",
    "h2": "dip_width_samples",
    "a2": "dip_height_samples",
    "b1": 0.0  # Offset is 0
}

T_RANGE = np.linspace(0, 1, 200)

for batch in BATCHES:
    # Find all animal pickle files for this batch
    pkl_files = [
        f for f in os.listdir(RESULTS_DIR)
        if f.startswith(f"results_{batch}_animal_") and f.endswith(".pkl")
    ]
    animal_ids = [
        int(f.split("_")[-1].replace(".pkl", "")) for f in pkl_files
    ]
    animal_ids, pkl_files = zip(*sorted(zip(animal_ids, pkl_files))) if animal_ids else ([],[])
    n_animals = len(animal_ids)
    if n_animals == 0:
        print(f"No animals found for batch {batch}")
        continue

    # Setup subplots
    ncols = min(4, n_animals)
    nrows = int(np.ceil(n_animals / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False)
    fig.suptitle(f"phi(t) for {batch} batch")

    for idx, (animal_id, pkl_file) in enumerate(zip(animal_ids, pkl_files)):
        ax = axs[idx // ncols][idx % ncols]
        pkl_path = os.path.join(RESULTS_DIR, pkl_file)
        with open(pkl_path, 'rb') as f:
            results = pickle.load(f)
        if MODEL_KEY not in results:
            ax.set_title(f"Animal {animal_id}\n(no model)")
            continue
        model_res = results[MODEL_KEY]
        # Get means of relevant params
        try:
            phi_args = {
                k: np.mean(model_res[v]) if v in model_res else 0.0
                for k, v in PARAM_MAP.items() if k != 'b1'
            }
            phi_args['b1'] = PARAM_MAP['b1']
            phi_vals = phi_t_fn(T_RANGE, **phi_args)
            ax.plot(T_RANGE, phi_vals, label=f"Animal {animal_id}")
            ax.set_title(f"Animal {animal_id}")
            ax.set_xlabel('t')
            ax.set_ylabel('phi(t)')
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", ha='center')
    # Hide unused axes
    for j in range(idx+1, nrows*ncols):
        axs[j // ncols][j % ncols].axis('off')
    plt.tight_layout(rect=[0,0,1,0.96])
    out_path = os.path.join(RESULTS_DIR, f"phi_vs_t_{batch}.png")
    plt.savefig(out_path)
    print(f"Saved: {out_path}")
    plt.close(fig)
