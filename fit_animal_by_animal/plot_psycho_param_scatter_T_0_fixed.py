# %%
"""
Create 3 scatter plots (one per parameter: rate_lambda, theta_E, w).
- x-axis: parameter value
- y-axis: "batch, animal" (categorical), arranged in ASCENDING ORDER BY THE PARAMETER shown in that subplot
- vertical lines per parameter:
  - green (alpha=0.5): plausible bounds
  - red (alpha=0.5): hard bounds

Reads 3-parameter VBMC psycho-fit pickles (T_0 fixed from vanilla) saved by
  decoding_conf_NEW_psychometric_fit_vbmc_all_animals_pot_supp_for_paper.py
from directory:
  /home/rlab/raghavendra/ddm_data/fit_valid_trials/psycho_fits_T_0_fixed_from_vanilla

Saves figure to:
  fit_animal_by_animal/psycho_param_scatter.png

Usage:
  python plot_psycho_param_scatter_T_0_fixed.py [--n-samples 50000] [--out psycho_param_scatter.png]
"""

import os
import re
import glob
import sys
import pickle
import argparse
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt

# Directory with psycho-fit pickles (3-parameter, T_0 fixed from vanilla)
PSYCHO_DIR = \
    '/home/rlab/raghavendra/ddm_data/fit_valid_trials/psycho_fits_T_0_fixed_from_vanilla'

FILE_PREFIX = 'psycho_fit_3-params-T_0_fixed_from_vanilla_'
FILE_REGEX = re.compile(r'^psycho_fit_3-params-T_0_fixed_from_vanilla_(?P<batch>.+)_(?P<animal>\d+)\.pkl$')
RESULTS_FILE_REGEX = re.compile(r'^results_(?P<batch>.+)_animal_(?P<animal>\d+)\.pkl$')

# Parameter bounds copied from
#   fit_animal_by_animal/decoding_conf_NEW_psychometric_fit_vbmc_all_animals_pot_supp_for_paper.py
# lines 291-299
BOUNDS = {
    'rate_lambda': {
        'bounds': (0.01, 0.2),
        'plausible': (0.05, 0.15),
    },
    'theta_E': {
        'bounds': (10.0, 80.0),
        'plausible': (30.0, 60.0),
    },
    'w': {
        'bounds': (0.2, 0.8),
        'plausible': (0.4, 0.6),
    },
}

VANILLA_BOUNDS = {
    'rate_lambda': (0.01, 1.0),
    'theta_E': (5.0, 65.0),
    'w': (0.3, 0.7),
    # The following are provided but not plotted in current grids
    't_E_aff': (0.01, 0.2),
    'del_go': (0.0, 0.2),
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Plot psycho-fit param means per animal (3 scatter plots)')
    ap.add_argument('--n-samples', type=int, default=50000,
                    help='VP samples to estimate means (default: 50000)')
    ap.add_argument('--dir', type=str, default=PSYCHO_DIR,
                    help='Directory of psycho-fit pickles (default: %(default)s)')
    ap.add_argument('--out', type=str, default='psycho_param_scatter.png',
                    help='Output PNG filename (default: %(default)s)')
    return ap.parse_args()


def find_pickles(directory: str) -> List[str]:
    pattern = os.path.join(directory, f'{FILE_PREFIX}*.pkl')
    return sorted(glob.glob(pattern))


def parse_batch_animal(filename: str) -> Dict[str, Any]:
    base = os.path.basename(filename)
    m = FILE_REGEX.match(base)
    if not m:
        raise ValueError(f'Filename does not match expected pattern: {base}')
    return {'batch': m.group('batch'), 'animal': int(m.group('animal'))}


def load_mean_params(pkl_path: str, n_samples: int) -> Dict[str, float]:
    with open(pkl_path, 'rb') as f:
        vbmc_obj = pickle.load(f)
    vp = getattr(vbmc_obj, 'vp', None)
    if vp is None:
        raise AttributeError(f'Loaded object has no attribute vp: {pkl_path}')
    samples = vp.sample(int(n_samples))[0]
    # 3-parameter order: [rate_lambda, theta_E, w]
    return {
        'rate_lambda': float(np.mean(samples[:, 0])),
        'theta_E': float(np.mean(samples[:, 1])),
        'w': float(np.mean(samples[:, 2])),
    }


def build_df(directory: str, n_samples: int) -> pd.DataFrame:
    files = find_pickles(directory)
    if not files:
        raise FileNotFoundError(f'No psycho-fit pickles found in {directory}')

    rows: List[Dict[str, Any]] = []
    for pkl in files:
        meta = parse_batch_animal(pkl)
        means = load_mean_params(pkl, n_samples)
        rows.append({
            'batch': meta['batch'],
            'animal': meta['animal'],
            **means,
        })
    df = pd.DataFrame(rows)
    # Arrange animals in ascending order (by batch then animal id)
    df = df.sort_values(by=['batch', 'animal']).reset_index(drop=True)
    # y labels "batch, animal"
    df['label'] = df.apply(lambda r: f"{r['batch']}, {r['animal']}", axis=1)
    return df


def find_results_pickles(directory: str) -> List[str]:
    """Find result pickles strictly matching results_{batch}_animal_{id}.pkl (no suffixes)."""
    pattern = os.path.join(directory, 'results_*_animal_*.pkl')
    files = sorted(glob.glob(pattern))
    return [p for p in files if RESULTS_FILE_REGEX.match(os.path.basename(p))]


def parse_results_batch_animal(filename: str) -> Dict[str, Any]:
    base = os.path.basename(filename)
    m = RESULTS_FILE_REGEX.match(base)
    if not m:
        raise ValueError(f'Filename does not match expected results pattern: {base}')
    return {'batch': m.group('batch'), 'animal': int(m.group('animal'))}


def load_mean_tied_params_from_results(pkl_path: str) -> Dict[str, float]:
    """Load vanilla tied parameter means from a results_{batch}_animal_{id}.pkl file.
    Uses only 'vbmc_vanilla_tied_results'; skips files without it.
    """
    with open(pkl_path, 'rb') as f:
        fit_results_data = pickle.load(f)

    if not isinstance(fit_results_data, dict) or 'vbmc_vanilla_tied_results' not in fit_results_data:
        raise KeyError(f"Missing 'vbmc_vanilla_tied_results' in {os.path.basename(pkl_path)}")
    model_samples = fit_results_data['vbmc_vanilla_tied_results']

    tied_map = {
        'rate_lambda_samples': 'rate_lambda',
        'theta_E_samples': 'theta_E',
        'w_samples': 'w',
    }
    out: Dict[str, float] = {}
    for s_key, label in tied_map.items():
        if s_key not in model_samples:
            raise KeyError(f"Key '{s_key}' not found in tied results for {os.path.basename(pkl_path)}")
        out[label] = float(np.mean(np.asarray(model_samples[s_key])))
    return out


def build_results_df(directory: str) -> pd.DataFrame:
    files = find_results_pickles(directory)
    if not files:
        raise FileNotFoundError(f'No results_*_animal_*.pkl files found in {directory}')

    rows: List[Dict[str, Any]] = []
    for pkl in files:
        try:
            meta = parse_results_batch_animal(pkl)
            means = load_mean_tied_params_from_results(pkl)
            rows.append({
                'batch': meta['batch'],
                'animal': meta['animal'],
                **means,
            })
        except Exception as e:
            # Skip files that don't conform; continue with others
            print(f"[WARN] Skipping {os.path.basename(pkl)}: {e}")
            continue

    if not rows:
        raise FileNotFoundError(f'No valid tied-params found in results pickles under {directory}')

    df = pd.DataFrame(rows)
    df = df.sort_values(by=['batch', 'animal']).reset_index(drop=True)
    return df


def plot_scatter(df: pd.DataFrame, out_path: str):
    params = ['rate_lambda', 'theta_E', 'w']

    fig, axes = plt.subplots(1, 3, figsize=(14, max(8, len(df) * 0.35)))
    # Ensure axes is a flat numpy array
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, p in zip(axes, params):
        # Sort animals ascending by this parameter (NaNs last)
        df_sorted = df.sort_values(by=p, na_position='last').reset_index(drop=True)
        y = np.arange(len(df_sorted))

        # Vertical lines: plausible (green) and hard bounds (red)
        b = BOUNDS[p]['bounds']
        pb = BOUNDS[p]['plausible']
        pb_vals = list(pb)
        b_vals = list(b)
        for x in pb_vals:
            ax.axvline(x, color='green', alpha=0.5, linestyle='-', zorder=0)
        for x in b_vals:
            ax.axvline(x, color='red', alpha=0.5, linestyle='-', zorder=0)

        # X values
        x_vals = df_sorted[p].values
        ax.scatter(x_vals, y, s=35, color='#1f77b4', alpha=0.9, edgecolor='none', zorder=2)
        # xticks at bounds and plausible bounds
        # tick_vals = sorted(set(pb_vals + b_vals))
        # ax.set_xticks(tick_vals)
        ax.set_title(p)
        ax.grid(axis='x', linestyle='--', alpha=0.4)
        ax.set_xlabel(p)
        ax.set_ylim(-1, len(df_sorted))
        ax.set_yticks(y)
        ax.set_yticklabels(df_sorted['label'].tolist())

    fig.suptitle('Psycho-fit parameter means per animal (x=value, y=batch, animal)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

    # Ensure output in same folder as this script if relative path
    if not os.path.isabs(out_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.join(script_dir, out_path)

    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_param_vs_param(df: pd.DataFrame, out_path: str, df_overlay: pd.DataFrame = None):
    """
    Create a 3x3 grid of parameter vs parameter scatter plots.
    Each subplot shows one animal's mean param1 vs mean param2 values.
    """
    params = ['rate_lambda', 'theta_E', 'w']

    df_plot = df.copy()
    df_overlay_plot = None
    if df_overlay is not None and not df_overlay.empty:
        df_overlay_plot = df_overlay.copy()

    n = len(params)
    fig, axes = plt.subplots(n, n, figsize=(12, 12))
    axes = axes.ravel()

    for i, p1 in enumerate(params):
        for j, p2 in enumerate(params):
            ax = axes[i * n + j]
            # Scatter plot of mean parameter values per animal (psychometric fits)
            ax.scatter(df_plot[p1], df_plot[p2], s=35, alpha=0.7, color='#1f77b4', edgecolor='none')
            
            # Add vertical and horizontal lines for parameter bounds (red, alpha=0.5)
            p1_bounds = BOUNDS[p1]['bounds']
            p2_bounds = BOUNDS[p2]['bounds']
            
            for x in p1_bounds:
                ax.axvline(x, color='yellow', alpha=0.5, linestyle='-', zorder=0)
            for y in p2_bounds:
                ax.axhline(y, color='yellow', alpha=0.5, linestyle='-', zorder=0)
            
            # Add vanilla bounds as dotted black lines
            v1_bounds = VANILLA_BOUNDS.get(p1)
            v2_bounds = VANILLA_BOUNDS.get(p2)
            if v1_bounds is not None:
                for x in v1_bounds:
                    ax.axvline(x, color='black', alpha=0.8, linestyle=':', zorder=0)
            if v2_bounds is not None:
                for y in v2_bounds:
                    ax.axhline(y, color='black', alpha=0.8, linestyle=':', zorder=0)
            
            # Overlay tied-params from vanilla results pickles (red 'x')
            if df_overlay_plot is not None:
                ax.scatter(df_overlay_plot[p1], df_overlay_plot[p2], s=30, alpha=0.7, color='red', marker='x')
            
            ax.set_xlabel(p1)
            ax.set_ylabel(p2)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if not os.path.isabs(out_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.join(script_dir, out_path)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved param vs param plot: {out_path}')


def main():
    args = parse_args()
    df = build_df(args.dir, args.n_samples)
    plot_scatter(df, args.out)
    # Generate param vs param plot
    base_out = os.path.splitext(args.out)[0]
    param_vs_param_out = f"{base_out}_param_vs_param.png"
    # Try to overlay results from results_{batch}_animal_{id}.pkl in this folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        df_results = build_results_df(script_dir)
        print(f"Overlaying {len(df_results)} vanilla result-pickle points (red 'x')")
    except FileNotFoundError:
        df_results = None
        print("No results_*_animal_*.pkl found for overlay; proceeding without overlay")
    plot_param_vs_param(df, param_vs_param_out, df_overlay=df_results)


if __name__ == '__main__':
    main()
