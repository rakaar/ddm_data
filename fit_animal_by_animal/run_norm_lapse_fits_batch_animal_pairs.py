#!/usr/bin/env python3
import argparse
import subprocess
import os
import sys
from typing import List, Tuple
import pandas as pd
from collections import defaultdict

# Default batches to include (same list as vanilla runner)
DEFAULT_BATCHES = ['SD', 'LED34', 'LED6', 'LED8', 'LED7', 'LED34_even']


def get_batch_animal_pairs_from_csvs(batch_names: List[str]) -> List[Tuple[str, int]]:
    """Load batch CSVs and extract unique (batch, animal) pairs for valid trials."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    batch_dir = os.path.join(script_dir, 'batch_csvs')
    batch_files = [f'batch_{batch_name}_valid_and_aborts.csv' for batch_name in batch_names]

    dfs = []
    for fname in batch_files:
        fpath = os.path.join(batch_dir, fname)
        if os.path.exists(fpath):
            dfs.append(pd.read_csv(fpath))
        else:
            print(f"Warning: CSV not found: {fpath}", file=sys.stderr)

    if not dfs:
        print("Error: No CSV files found for the requested batches.", file=sys.stderr)
        sys.exit(1)

    merged = pd.concat(dfs, ignore_index=True)
    merged_valid = merged[merged['success'].isin([1, -1])].copy()
    pairs = sorted(list(map(tuple, merged_valid[['batch_name', 'animal']].drop_duplicates().values)))
    return pairs

def print_batch_animal_table(pairs: List[Tuple[str, int]]):
    if not pairs:
        print("No batch-animal pairs found.")
        return
    batch_to_animals = defaultdict(list)
    for batch, animal in pairs:
        s = str(animal)
        if s not in batch_to_animals[batch]:
            batch_to_animals[batch].append(s)
    max_batch_len = max(len(b) for b in batch_to_animals.keys()) if batch_to_animals else 0
    animal_strings = {b: ', '.join(sorted(a)) for b, a in batch_to_animals.items()}
    max_animals_len = max(len(s) for s in animal_strings.values()) if animal_strings else 0
    print(f"{'Batch':<{max_batch_len}}  {'Animals'}")
    print(f"{'=' * max_batch_len}  {'=' * max_animals_len}")
    for batch in sorted(animal_strings.keys()):
        animals_str = animal_strings[batch]
        print(f"{batch:<{max_batch_len}}  {animals_str}")


def main():
    parser = argparse.ArgumentParser(description='Run norm+lapse fits (init-type=norm) for all (batch, animal) pairs discovered from CSVs')
    parser.add_argument('--batches', nargs='+', default=DEFAULT_BATCHES,
                        help=f"Batches to include. Default: {' '.join(DEFAULT_BATCHES)}")
    parser.add_argument('--output-dir', default='oct_9_10_norm_lapse_model_fit_files',
                        help='Directory to save results for all runs')
    parser.add_argument('--is-stim-filtered', action='store_true',
                        help='Filter to specific ABLs and ILDs')
    parser.add_argument('--python', default=sys.executable,
                        help='Python interpreter to use for running the single-animal script')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be run without actually running')
    parser.add_argument('--start-from', type=int, default=1,
                        help='Start from this run number (useful for resuming)')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    single_script = os.path.join(script_dir, 'lapses_fit_single_animal_norm_model.py')

    if not os.path.exists(single_script):
        print(f"Could not find single-animal script at {single_script}", file=sys.stderr)
        sys.exit(1)

    # Ensure output directory exists upfront
    os.makedirs(args.output_dir, exist_ok=True)

    # Discover pairs
    print("Discovering batch-animal pairs from CSV files...")
    pairs = get_batch_animal_pairs_from_csvs(args.batches)
    print(f"\nFound {len(pairs)} batch-animal pairs from {len(set(p[0] for p in pairs))} batches:\n")
    print_batch_animal_table(pairs)
    print()

    total_runs = len(pairs)
    stim_filter_msg = " (with stimulus filtering)" if args.is_stim_filtered else ""
    print(f"Will run {total_runs} norm+lapse fits (init-type=norm){stim_filter_msg}. Output dir: {args.output_dir}\n")

    if args.dry_run:
        print("DRY RUN - commands that would be executed:")
        for idx, (batch, animal) in enumerate(pairs, start=1):
            if idx < args.start_from:
                continue
            cmd = [
                args.python,
                single_script,
                '--batch', batch,
                '--animal', str(animal),
                '--init-type', 'norm',
                '--output-dir', args.output_dir,
            ]
            if args.is_stim_filtered:
                cmd.append('--is-stim-filtered')
            print(f"[{idx}/{total_runs}] {' '.join(cmd)}")
        sys.exit(0)

    failures = []
    skipped = []
    for run_idx, (batch, animal) in enumerate(pairs, start=1):
        if run_idx < args.start_from:
            print(f"[{run_idx}/{total_runs}] Skipping batch={batch}, animal={animal} (before start-from)")
            continue
        
        # Check if output pickle file already exists
        stim_filter_suffix = '_stim_filtered' if args.is_stim_filtered else ''
        pkl_filename = f'vbmc_norm_tied_results_batch_{batch}_animal_{animal}_lapses_truncate_1s_norm{stim_filter_suffix}.pkl'
        pkl_path = os.path.join(args.output_dir, pkl_filename)
        if os.path.exists(pkl_path):
            print(f"[{run_idx}/{total_runs}] Skipping batch={batch}, animal={animal} (pkl file already exists)")
            skipped.append((batch, animal))
            continue
        
        print(f"[{run_idx}/{total_runs}] Running batch={batch}, animal={animal}, init_type=norm ...")
        cmd = [
            args.python,
            single_script,
            '--batch', batch,
            '--animal', str(animal),
            '--init-type', 'norm',
            '--output-dir', args.output_dir,
        ]
        if args.is_stim_filtered:
            cmd.append('--is-stim-filtered')
        try:
            result = subprocess.run(cmd, cwd=script_dir, check=True)
            print(f"[{run_idx}/{total_runs}] Done: batch={batch}, animal={animal}, init_type=norm")
        except subprocess.CalledProcessError as e:
            print(f"[{run_idx}/{total_runs}] FAILED: batch={batch}, animal={animal}, init_type=norm (returncode={e.returncode})", file=sys.stderr)
            failures.append((batch, animal))

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if skipped:
        print(f"\nSkipped {len(skipped)} animals (pkl file already exists):")
        for batch, animal in skipped:
            print(f" - {batch}:{animal}")
    
    if failures:
        print(f"\n{len(failures)} runs FAILED:")
        for batch, animal in failures:
            print(f" - {batch}:{animal} (init_type=norm)")
        sys.exit(2)
    else:
        completed = len(pairs) - len(skipped)
        print(f"\nAll {completed} new runs completed successfully.")
        if skipped:
            print(f"({len(skipped)} animals were skipped because pkl files already existed)")


if __name__ == '__main__':
    main()
