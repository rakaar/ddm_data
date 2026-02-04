# %%
import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = '/home/rlab/raghavendra/ddm_data/fit_animal_by_animal'
NORM_LAPSE_DIR = os.path.join(BASE_DIR, 'oct_9_10_norm_lapse_model_fit_files')
RESULTS_DIR = BASE_DIR
SAMPLE_SIZE = int(1e5)


# %%
def parse_filename_norm_lapse(filename):
    """
    Parse norm+lapse pickle filename to extract batch and animal.
    Expected format: vbmc_norm_tied_results_batch_{batch}_animal_{animal}_lapses_truncate_1s_norm.pkl
    """
    name = filename.replace('.pkl', '')
    parts = name.split('_')

    batch_idx = parts.index('batch') + 1
    batch_parts = []
    animal_idx = None

    for i in range(batch_idx, len(parts)):
        if parts[i] == 'animal':
            animal_idx = i + 1
            break
        batch_parts.append(parts[i])

    batch = '_'.join(batch_parts)
    animal = int(parts[animal_idx])
    return batch, animal


def get_norm_lapse_params(pkl_path):
    """Extract mean and 95% CI for norm+lapse parameters from VBMC pickle."""
    try:
        with open(pkl_path, 'rb') as handle:
            vbmc = pickle.load(handle)

        if not hasattr(vbmc, 'iteration_history'):
            return {}

        iter_hist = vbmc.iteration_history
        if 'vp' not in iter_hist:
            return {}

        last_vp = iter_hist['vp'][-1]
        vp_samples, _ = last_vp.sample(SAMPLE_SIZE)

        param_names = [
            'rate_lambda',
            'T_0',
            'theta_E',
            'w',
            't_E_aff',
            'del_go',
            'rate_norm_l',
            'lapse_prob',
            'lapse_prob_right',
        ]

        params = {}
        for idx, name in enumerate(param_names):
            samples = vp_samples[:, idx]
            params[name] = {
                'mean': float(np.mean(samples)),
                'percentile_2_5': float(np.percentile(samples, 2.5)),
                'percentile_97_5': float(np.percentile(samples, 97.5)),
            }

        return params
    except Exception as exc:
        print(f"Warning: could not load norm+lapse params from {pkl_path}: {exc}")
        return {}


def get_original_norm_params(batch, animal_id):
    """Extract mean and 95% CI for original norm parameters from results pickle."""
    pkl_fname = f'results_{batch}_animal_{animal_id}.pkl'
    pkl_path = os.path.join(RESULTS_DIR, pkl_fname)

    if not os.path.exists(pkl_path):
        return {}

    try:
        with open(pkl_path, 'rb') as handle:
            results = pickle.load(handle)

        if 'vbmc_norm_tied_results' not in results:
            return {}

        norm_data = results['vbmc_norm_tied_results']
        param_keys = ['rate_lambda', 'T_0', 'theta_E', 'rate_norm_l']
        sample_keys = ['rate_lambda_samples', 'T_0_samples', 'theta_E_samples', 'rate_norm_l_samples']

        params = {}
        for param_key, sample_key in zip(param_keys, sample_keys):
            if sample_key not in norm_data:
                continue
            samples = np.asarray(norm_data[sample_key])
            params[param_key] = {
                'mean': float(np.mean(samples)),
                'percentile_2_5': float(np.percentile(samples, 2.5)),
                'percentile_97_5': float(np.percentile(samples, 97.5)),
            }

        return params
    except Exception as exc:
        print(f"Warning: could not load norm params from {pkl_path}: {exc}")
        return {}


# %%
entries = []

for pkl_path in glob.glob(os.path.join(NORM_LAPSE_DIR, '*.pkl')):
    filename = os.path.basename(pkl_path)
    try:
        batch, animal = parse_filename_norm_lapse(filename)
    except Exception as exc:
        print(f"Warning: could not parse {filename}: {exc}")
        continue

    norm_lapse_params = get_norm_lapse_params(pkl_path)
    lapse_param = norm_lapse_params.get('lapse_prob')
    if not lapse_param:
        continue

    norm_params = get_original_norm_params(batch, animal)
    if not norm_params:
        continue

    required = ['rate_norm_l', 'rate_lambda', 'theta_E', 'T_0']
    if not all(param in norm_params for param in required):
        continue
    if not all(param in norm_lapse_params for param in required):
        continue

    entries.append({
        'batch': batch,
        'animal': animal,
        'label': f"{batch}-{animal}",
        'lapse_prob': lapse_param['mean'],
        'norm_params': norm_params,
        'norm_lapse_params': norm_lapse_params,
    })

entries.sort(key=lambda item: item['lapse_prob'])
print(f"Loaded {len(entries)} animals with norm + norm+lapse params.")


# %%
param_configs = [
    {'name': 'rate_norm_l', 'label': r'$\ell$', 'convert_ms': False},
    {'name': 'rate_lambda', 'label': r"$\lambda'$", 'convert_ms': False},
    {'name': 'theta_E', 'label': r'$\theta_E$', 'convert_ms': False},
    {'name': 'T_0', 'label': r'$T_0$ (s)', 'convert_ms': False},
]

x_pos = np.arange(len(entries))
tick_map = {
    'rate_norm_l': [0.8, 0.9, 1.0],
    'rate_lambda': [0.1, 0.2],
    'theta_E': [2.0, 3.0],
    'T_0': [0.1, 0.2],
}

for config in param_configs:
    param = config['name']
    convert_ms = config['convert_ms']

    norm_means = np.array([entry['norm_params'][param]['mean'] for entry in entries], dtype=float)
    norm_low = np.array([entry['norm_params'][param]['percentile_2_5'] for entry in entries], dtype=float)
    norm_high = np.array([entry['norm_params'][param]['percentile_97_5'] for entry in entries], dtype=float)

    norm_lapse_means = np.array([entry['norm_lapse_params'][param]['mean'] for entry in entries], dtype=float)
    norm_lapse_low = np.array([entry['norm_lapse_params'][param]['percentile_2_5'] for entry in entries], dtype=float)
    norm_lapse_high = np.array([entry['norm_lapse_params'][param]['percentile_97_5'] for entry in entries], dtype=float)

    if convert_ms:
        norm_means *= 1000
        norm_low *= 1000
        norm_high *= 1000
        norm_lapse_means *= 1000
        norm_lapse_low *= 1000
        norm_lapse_high *= 1000

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.errorbar(
        x_pos,
        norm_means,
        yerr=[norm_means - norm_low, norm_high - norm_means],
        fmt='o',
        color='green',
        alpha=0.7,
        capsize=0,
        label='Norm',
        markersize=6,
        linewidth=1.5,
    )
    ax.errorbar(
        x_pos,
        norm_lapse_means,
        yerr=[norm_lapse_means - norm_lapse_low, norm_lapse_high - norm_lapse_means],
        fmt='s',
        color='red',
        alpha=0.7,
        capsize=0,
        label='Norm + lapse',
        markersize=6,
        linewidth=1.5,
    )

    ax.set_xlabel('Rat', fontsize=12)
    ax.set_ylabel(config['label'], fontsize=12)
    ax.set_title(config['label'], fontsize=14)
    ax.set_xticks([])
    if param in tick_map:
        ax.set_yticks(tick_map[param])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, fontsize=10)

    output_prefix = os.path.join(BASE_DIR, f"param_{param}_ordered_by_npl_lapse")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_prefix}.pdf", bbox_inches='tight')
    plt.show()

    print(f"Saved: {output_prefix}.png")
    print(f"Saved: {output_prefix}.pdf")

# %%
