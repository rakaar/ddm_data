# %%
import glob
import os
import pickle

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse as MplEllipse


BASE_DIR = '/home/rlab/raghavendra/ddm_data/fit_animal_by_animal'
NORM_LAPSE_DIR = os.path.join(BASE_DIR, 'oct_9_10_norm_lapse_model_fit_files')
DESIRED_BATCHES = ['SD', 'LED34', 'LED6', 'LED8', 'LED7', 'LED34_even']


def get_valid_batch_animal_pairs():
    import csv

    batch_dir = os.path.join(BASE_DIR, 'batch_csvs')
    batch_files = [f'batch_{batch_name}_valid_and_aborts.csv' for batch_name in DESIRED_BATCHES]

    pairs = set()
    for fname in batch_files:
        path = os.path.join(batch_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path, 'r', newline='') as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    success = int(float(row.get('success', 'nan')))
                except (TypeError, ValueError):
                    continue
                if success not in (1, -1):
                    continue
                batch_name = row.get('batch_name') or fname.replace('batch_', '').replace('_valid_and_aborts.csv', '')
                animal = row.get('animal')
                if animal in (None, ''):
                    continue
                try:
                    animal_id = int(float(animal))
                except (TypeError, ValueError):
                    continue
                pairs.add((batch_name, animal_id))
    return pairs


def is_preferred_norm_lapse_file(filename):
    return filename.endswith('_lapses_truncate_1s_norm.pkl')


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


def extract_params_from_norm_lapse_pkl(pkl_path, n_samples):
    """
    Extract rate_lambda and lapse_prob samples from norm+lapse VBMC pickle file.

    Args:
        pkl_path: Path to pickle file
        n_samples: Number of samples to draw from the VP

    Returns:
        dict with 'rate_lambda_samples' and 'lapse_prob_samples' arrays (or None if extraction fails)
    """
    try:
        with open(pkl_path, 'rb') as f:
            vbmc = pickle.load(f)

        if not hasattr(vbmc, 'iteration_history'):
            print(f"No iteration_history in VBMC object from {pkl_path}")
            return None

        iter_hist = vbmc.iteration_history
        if 'vp' not in iter_hist:
            print(f"No 'vp' in iteration_history from {pkl_path}")
            return None

        vp_arr = iter_hist['vp']
        last_vp = vp_arr[-1]
        vp_samples, _ = last_vp.sample(n_samples)

        # Norm+lapse parameter order:
        # rate_lambda, T_0, theta_E, w, t_E_aff, del_go, rate_norm_l, lapse_prob, lapse_prob_right
        rate_lambda_samples = vp_samples[:, 0]
        lapse_prob_samples = vp_samples[:, 7]

        return {
            'rate_lambda_samples': rate_lambda_samples,
            'lapse_prob_samples': lapse_prob_samples,
        }
    except Exception as exc:
        print(f"Error extracting parameters from {pkl_path}: {exc}")
        import traceback

        traceback.print_exc()
        return None


def load_rate_lambda_vs_lapse_data(n_samples_per_animal=5000):
    """Load rate_lambda and lapse_prob samples from all animals."""
    norm_lapse_files = glob.glob(os.path.join(NORM_LAPSE_DIR, '*.pkl'))
    print(f"Found {len(norm_lapse_files)} norm+lapse pickle files")

    valid_pairs = get_valid_batch_animal_pairs()
    seen_pairs = set()

    animal_data = []
    for pkl_path in norm_lapse_files:
        filename = os.path.basename(pkl_path)
        if not is_preferred_norm_lapse_file(filename):
            continue
        try:
            batch, animal = parse_filename_norm_lapse(filename)
            pair = (batch, int(animal))
            if pair not in valid_pairs:
                continue
            if pair in seen_pairs:
                print(f"Skipping duplicate entry for {batch} animal {animal}")
                continue
            seen_pairs.add(pair)
            print(f"Processing {batch} animal {animal}...")
            params = extract_params_from_norm_lapse_pkl(pkl_path, n_samples_per_animal)
            if params is not None:
                animal_data.append({
                    'batch': batch,
                    'animal': animal,
                    'rate_lambda_samples': params['rate_lambda_samples'],
                    'lapse_prob_samples': params['lapse_prob_samples'],
                })
        except Exception as exc:
            print(f"Error processing {filename}: {exc}")

    print(f"\nSuccessfully processed {len(animal_data)} animals")
    print(f"Total number of points to plot: {len(animal_data) * n_samples_per_animal}")
    return animal_data


def plot_rate_lambda_vs_lapse(animal_data, ellipse_quantile=0.95, animals_by_color=False):
    """
    Create scatter plot with samples from each animal's posterior.
    Shows rate_lambda vs lapse_prob with linear fit, correlation, and covariance ellipses.
    """
    if len(animal_data) == 0:
        print("No data to plot!")
        return None

    # Compute mean per animal for statistics
    rate_lambda_means = [np.mean(d['rate_lambda_samples']) for d in animal_data]
    lapse_prob_means = [np.mean(d['lapse_prob_samples']) * 100 for d in animal_data]

    # Generate colors based on flag
    n_animals = len(animal_data)
    if animals_by_color:
        animal_colors = cm.tab20(np.linspace(0, 1, n_animals))
        sample_color = None
    else:
        animal_colors = None
        sample_color = 'steelblue'
        ellipse_color = '#2b6cb0'

    # Flatten all samples across all animals (with animal indices if coloring by animal)
    all_rate_lambda = []
    all_lapse_prob_pct = []
    all_animal_indices = []

    for idx, d in enumerate(animal_data):
        n_samples = len(d['rate_lambda_samples'])
        all_rate_lambda.extend(d['rate_lambda_samples'])
        all_lapse_prob_pct.extend(np.array(d['lapse_prob_samples']) * 100)
        all_animal_indices.extend([idx] * n_samples)

    all_rate_lambda = np.array(all_rate_lambda)
    all_lapse_prob_pct = np.array(all_lapse_prob_pct)
    all_animal_indices = np.array(all_animal_indices)

    # Compute correlation using all samples (use percentage values)
    correlation = np.corrcoef(all_rate_lambda, all_lapse_prob_pct)[0, 1]

    # Fit a linear regression line using all samples (use percentage values)
    z = np.polyfit(all_rate_lambda, all_lapse_prob_pct, 1)
    p = np.poly1d(z)
    x_line = np.linspace(np.min(all_rate_lambda), np.max(all_rate_lambda), 100)
    y_line = p(x_line)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot samples based on flag
    if animals_by_color:
        for idx in range(n_animals):
            mask = all_animal_indices == idx
            ax.scatter(all_rate_lambda[mask], all_lapse_prob_pct[mask],
                       alpha=0.15, s=20, c=[animal_colors[idx]], edgecolors='none')
    else:
        ax.scatter(all_rate_lambda, all_lapse_prob_pct, alpha=0.15, s=20,
                   c=sample_color, edgecolors='none')

    # Fit and plot covariance ellipses for each animal
    q = float(ellipse_quantile)
    if not (0.0 < q < 1.0):
        q = 0.95
    s_chi2 = -2.0 * np.log(max(1e-12, 1.0 - q))

    ellipse_data = []

    for idx, d in enumerate(animal_data):
        x = np.array(d['rate_lambda_samples'])
        y = np.array(d['lapse_prob_samples']) * 100

        if x.size < 2 or y.size < 2:
            continue

        valid_mask = np.isfinite(x) & np.isfinite(y)
        x = x[valid_mask]
        y = y[valid_mask]

        if x.size < 2:
            continue

        m_x = float(np.mean(x))
        m_y = float(np.mean(y))

        cov = np.cov(np.vstack([x, y]))
        if not np.all(np.isfinite(cov)):
            continue

        try:
            evals, evecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            continue

        order = np.argsort(evals)[::-1]
        evals = np.maximum(evals[order], 0.0)
        evecs = evecs[:, order]

        width = 2.0 * float(np.sqrt(s_chi2 * evals[0])) if evals.size > 0 else 0.0
        height = 2.0 * float(np.sqrt(s_chi2 * evals[1])) if evals.size > 1 else 0.0
        if width == 0.0 or height == 0.0:
            continue

        angle = float(np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0])))

        if animals_by_color:
            current_ellipse_color = animal_colors[idx]
        else:
            current_ellipse_color = ellipse_color

        ellipse = MplEllipse(
            (m_x, m_y), width=width, height=height, angle=angle,
            facecolor='none', edgecolor=current_ellipse_color, linewidth=1.5,
            alpha=0.8, zorder=4,
        )
        ax.add_patch(ellipse)

        ellipse_data.append({
            'batch': d['batch'],
            'animal': d['animal'],
            'mean_x': m_x,
            'mean_y': m_y,
            'width': width,
            'height': height,
            'angle': angle,
            'cov': cov,
            'evals': evals,
            'evecs': evecs,
        })

    ax.plot(x_line, y_line, 'r--', linewidth=2, alpha=0.8)

    ax.set_xlabel(r'$\lambda^{\prime}$', fontsize=20)
    ax.set_ylabel('Lapse rate (%)', fontsize=20)

    x_min = float(np.nanmin(all_rate_lambda))
    x_max = float(np.nanmax(all_rate_lambda))
    if np.isfinite(x_min) and np.isfinite(x_max) and x_min != x_max:
        ax.set_xticks(np.round(np.linspace(x_min, x_max, 2), 2))

    ax.set_yticks([0, 6])
    ax.tick_params(axis='both', labelsize=20)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    output_path = os.path.join(BASE_DIR, 'rate_lambda_vs_lapse_prob_scatter_with_ellipses.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")

    plt.show()

    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS (all posterior samples)")
    print("=" * 60)
    print(f"rate_lambda:  mean={np.mean(all_rate_lambda):.4f}, "
          f"std={np.std(all_rate_lambda):.4f}, "
          f"min={np.min(all_rate_lambda):.4f}, "
          f"max={np.max(all_rate_lambda):.4f}")
    print(f"lapse_rate(%): mean={np.mean(all_lapse_prob_pct):.4f}, "
          f"std={np.std(all_lapse_prob_pct):.4f}, "
          f"min={np.min(all_lapse_prob_pct):.4f}, "
          f"max={np.max(all_lapse_prob_pct):.4f}")
    print(f"\nPearson correlation: {correlation:.4f}")

    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS (per-animal means)")
    print("=" * 60)
    print(f"rate_lambda:  mean={np.mean(rate_lambda_means):.4f}, "
          f"std={np.std(rate_lambda_means):.4f}, "
          f"min={np.min(rate_lambda_means):.4f}, "
          f"max={np.max(rate_lambda_means):.4f}")
    print(f"lapse_rate(%): mean={np.mean(lapse_prob_means):.4f}, "
          f"std={np.std(lapse_prob_means):.4f}, "
          f"min={np.min(lapse_prob_means):.4f}, "
          f"max={np.max(lapse_prob_means):.4f}")
    correlation_means = np.corrcoef(rate_lambda_means, lapse_prob_means)[0, 1]
    print(f"\nPearson correlation (means): {correlation_means:.4f}")

    print("\n" + "=" * 60)
    print(f"ELLIPSE FIT SUMMARY (quantile={ellipse_quantile})")
    print("=" * 60)
    print(f"Number of animals with ellipses fitted: {len(ellipse_data)}")
    print("=" * 60)

    plot_data = {
        'animal_data': animal_data,
        'all_rate_lambda': all_rate_lambda,
        'all_lapse_prob_pct': all_lapse_prob_pct,
        'all_animal_indices': all_animal_indices,
        'rate_lambda_means': rate_lambda_means,
        'lapse_prob_means_pct': lapse_prob_means,
        'correlation_all_samples': correlation,
        'correlation_means': correlation_means,
        'linear_fit': {
            'slope': z[0],
            'intercept': z[1],
            'x_line': x_line,
            'y_line': y_line,
        },
        'ellipses': ellipse_data,
        'ellipse_quantile': ellipse_quantile,
        'animals_by_color': animals_by_color,
        'ellipse_color': ellipse_color if not animals_by_color else None,
        'animal_colors': animal_colors.tolist() if animals_by_color and animal_colors is not None else None,
        'statistics': {
            'all_samples': {
                'rate_lambda': {
                    'mean': np.mean(all_rate_lambda),
                    'std': np.std(all_rate_lambda),
                    'min': np.min(all_rate_lambda),
                    'max': np.max(all_rate_lambda),
                },
                'lapse_rate_pct': {
                    'mean': np.mean(all_lapse_prob_pct),
                    'std': np.std(all_lapse_prob_pct),
                    'min': np.min(all_lapse_prob_pct),
                    'max': np.max(all_lapse_prob_pct),
                },
            },
            'animal_means': {
                'rate_lambda': {
                    'mean': np.mean(rate_lambda_means),
                    'std': np.std(rate_lambda_means),
                    'min': np.min(rate_lambda_means),
                    'max': np.max(rate_lambda_means),
                },
                'lapse_rate_pct': {
                    'mean': np.mean(lapse_prob_means),
                    'std': np.std(lapse_prob_means),
                    'min': np.min(lapse_prob_means),
                    'max': np.max(lapse_prob_means),
                },
            },
        },
        'n_animals': len(animal_data),
        'n_ellipses_fitted': len(ellipse_data),
    }

    pkl_output_path = os.path.join(BASE_DIR, 'rate_lambda_vs_lapse_prob_data.pkl')
    with open(pkl_output_path, 'wb') as f:
        pickle.dump(plot_data, f)
    print(f"\nPlot data saved to: {pkl_output_path}")

    return plot_data


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("LOADING RATE_LAMBDA VS LAPSE RATE DATA")
    print("=" * 60)
    data = load_rate_lambda_vs_lapse_data()
    plot_rate_lambda_vs_lapse(data, animals_by_color=True)
