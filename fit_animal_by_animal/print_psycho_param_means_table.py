#!/usr/bin/env python3
"""
Scan VBMC psycho-fit pickles and print mean parameters per animal as a table.

- Reads files saved by decoding_conf_NEW_psychometric_fit_vbmc_all_animals_pot_supp_for_paper.py
  at: /home/rlab/raghavendra/ddm_data/fit_valid_trials/psycho_fits_4-params-del_E_go_fixed_as_avg
  with pattern: psycho_fit_4-params-del_E_go_fixed_as_avg_{batch_name}_{animal_id}.pkl
- For each file, loads the VBMC object, samples from vp, computes means for
  [rate_lambda, T_0, theta_E, w]
- Prints a table (optionally CSV) to stdout

Usage:
  python print_psycho_param_means_table.py [--n-samples 50000] [--format plain|csv|md]
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

# Default directory where psycho-fit pickles are saved
PSYCHO_DIR = \
    '/home/rlab/raghavendra/ddm_data/fit_valid_trials/psycho_fits_4-params-del_E_go_fixed_as_avg'

# File pattern components
FILE_PREFIX = 'psycho_fit_4-params-del_E_go_fixed_as_avg_'
FILE_REGEX = re.compile(r'^psycho_fit_4-params-del_E_go_fixed_as_avg_(?P<batch>.+)_(?P<animal>\d+)\.pkl$')


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Print mean psycho-fit params per animal')
    ap.add_argument('--n-samples', type=int, default=50000,
                    help='Number of VP samples to estimate means (default: 50000)')
    ap.add_argument('--format', type=str, default='plain', choices=['plain', 'csv', 'md'],
                    help='Output format: plain = aligned text, csv = CSV, md = Markdown table')
    ap.add_argument('--dir', type=str, default=PSYCHO_DIR,
                    help='Directory containing psycho-fit pickles (default: %(default)s)')
    return ap.parse_args()


def find_pickles(directory: str) -> List[str]:
    pattern = os.path.join(directory, f'{FILE_PREFIX}*.pkl')
    return sorted(glob.glob(pattern))


def parse_batch_animal(filename: str) -> Dict[str, Any]:
    base = os.path.basename(filename)
    m = FILE_REGEX.match(base)
    if not m:
        raise ValueError(f'Filename does not match expected pattern: {base}')
    return {
        'batch': m.group('batch'),
        'animal': int(m.group('animal')),
    }


def load_mean_params(pkl_path: str, n_samples: int) -> Dict[str, float]:
    with open(pkl_path, 'rb') as f:
        vbmc_obj = pickle.load(f)
    # Following get_psycho_params() usage: vbmc_obj.vp.sample(N)[0]
    vp = getattr(vbmc_obj, 'vp', None)
    if vp is None:
        raise AttributeError(f'Loaded object has no attribute vp: {pkl_path}')
    samples = vp.sample(int(n_samples))[0]
    return {
        'rate_lambda': float(np.mean(samples[:, 0])),
        'T_0': float(np.mean(samples[:, 1])),
        'theta_E': float(np.mean(samples[:, 2])),
        'w': float(np.mean(samples[:, 3])),
    }


def to_markdown_table(df: pd.DataFrame) -> str:
    # Build a simple markdown table without extra deps
    headers = list(df.columns)
    # Header row
    lines = ['| ' + ' | '.join(str(h) for h in headers) + ' |']
    # Separator
    lines.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
    # Data rows
    for _, row in df.iterrows():
        lines.append('| ' + ' | '.join(str(v) for v in row.values) + ' |')
    return '\n'.join(lines)


def main():
    args = parse_args()
    directory = os.path.abspath(args.dir)
    if not os.path.isdir(directory):
        print(f"Error: directory not found: {directory}", file=sys.stderr)
        sys.exit(1)

    files = find_pickles(directory)
    if not files:
        print(f"No psycho-fit pickles found in {directory}")
        sys.exit(0)

    rows: List[Dict[str, Any]] = []
    failed: List[str] = []

    for pkl in files:
        try:
            meta = parse_batch_animal(pkl)
            means = load_mean_params(pkl, args.n_samples)
            row = {
                'batch': meta['batch'],
                'animal': meta['animal'],
                'rate_lambda': means['rate_lambda'],
                'T_0': means['T_0'],
                'theta_E': means['theta_E'],
                'w': means['w'],
            }
            rows.append(row)
        except Exception as e:
            failed.append(f"{os.path.basename(pkl)}: {e}")

    if not rows:
        print("No valid results to display.")
        if failed:
            print("Failures:")
            for msg in failed:
                print(f"  - {msg}")
        sys.exit(0)

    df = pd.DataFrame(rows)
    df = df.sort_values(by=['batch', 'animal']).reset_index(drop=True)

    # Format floats for prettier printing
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.options.display.float_format = '{:0.6f}'.format

    if args.format == 'csv':
        print(df.to_csv(index=False))
    elif args.format == 'md':
        # Avoid scientific notation in text
        df_fmt = df.copy()
        for c in ['rate_lambda', 'T_0', 'theta_E', 'w']:
            df_fmt[c] = df_fmt[c].map(lambda x: f"{x:0.6f}")
        print(to_markdown_table(df_fmt))
    else:
        print(df.to_string(index=False))

    if failed:
        print("\nWarnings (failed files):", file=sys.stderr)
        for msg in failed:
            print(f"  - {msg}", file=sys.stderr)


if __name__ == '__main__':
    main()
