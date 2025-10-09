#!/usr/bin/env python3
"""
Run vanilla+lapse model fits for all batch-animal pairs found in CSV files.
Automatically discovers pairs from batch CSV files, no manual specification needed.
"""
import argparse
import subprocess
import os
import sys
import pandas as pd
from collections import defaultdict
from typing import List, Tuple


def get_batch_animal_pairs_from_csvs(batch_names: List[str]) -> List[Tuple[str, int]]:
    """
    Load batch CSV files and extract unique (batch, animal) pairs.
    
    Args:
        batch_names: List of batch names to include
        
    Returns:
        Sorted list of (batch_name, animal_id) tuples
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    batch_dir = os.path.join(script_dir, 'batch_csvs')
    
    batch_files = [f'batch_{batch_name}_valid_and_aborts.csv' for batch_name in batch_names]
    
    # Load and merge data
    dfs = []
    for fname in batch_files:
        fpath = os.path.join(batch_dir, fname)
        if os.path.exists(fpath):
            dfs.append(pd.read_csv(fpath))
        else:
            print(f"Warning: Could not find {fpath}", file=sys.stderr)
    
    if not dfs:
        print("Error: No CSV files found", file=sys.stderr)
        sys.exit(1)
    
    merged_data = pd.concat(dfs, ignore_index=True)
    merged_valid = merged_data[merged_data['success'].isin([1, -1])].copy()
    
    # Extract unique (batch, animal) pairs
    batch_animal_pairs = sorted(list(map(tuple, merged_valid[['batch_name', 'animal']].drop_duplicates().values)))
    
    return batch_animal_pairs


def print_batch_animal_table(pairs: List[Tuple[str, int]]):
    """Pretty-print the batch-animal pairs as a table."""
    if not pairs:
        return
    
    batch_to_animals = defaultdict(list)
    for batch, animal in pairs:
        animal_str = str(animal)
        if animal_str not in batch_to_animals[batch]:
            batch_to_animals[batch].append(animal_str)
    
    # Format table
    max_batch_len = max(len(b) for b in batch_to_animals.keys())
    animal_strings = {b: ', '.join(sorted(a)) for b, a in batch_to_animals.items()}
    max_animals_len = max(len(s) for s in animal_strings.values()) if animal_strings else 0
    
    print(f"{'Batch':<{max_batch_len}}  {'Animals'}")
    print(f"{'=' * max_batch_len}  {'=' * max_animals_len}")
    
    for batch in sorted(animal_strings.keys()):
        animals_str = animal_strings[batch]
        print(f"{batch:<{max_batch_len}}  {animals_str}")


def main():
    # Default batches to include
    DEFAULT_BATCHES = ['SD', 'LED34', 'LED6', 'LED8', 'LED7', 'LED34_even']
    
    parser = argparse.ArgumentParser(
        description='Run vanilla+lapse model fits for all batch-animal pairs discovered from CSV files'
    )
    parser.add_argument('--batches', nargs='+', default=DEFAULT_BATCHES,
                        help=f'Batches to include. Default: {" ".join(DEFAULT_BATCHES)}')
    parser.add_argument('--output-dir', default='oct_9_10_vanila_lapse_model_fit_files',
                        help='Directory to save results for all runs')
    parser.add_argument('--python', default=sys.executable,
                        help='Python interpreter to use for running the single-animal script')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be run without actually running')
    parser.add_argument('--start-from', type=int, default=1,
                        help='Start from this run number (useful for resuming)')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    single_script = os.path.join(script_dir, 'lapses_fit_single_animal.py')

    if not os.path.exists(single_script):
        print(f"Could not find single-animal script at {single_script}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover all batch-animal pairs
    print("Discovering batch-animal pairs from CSV files...")
    pairs = get_batch_animal_pairs_from_csvs(args.batches)
    
    print(f"\nFound {len(pairs)} batch-animal pairs from {len(set(p[0] for p in pairs))} batches:\n")
    print_batch_animal_table(pairs)
    print()

    total_runs = len(pairs)
    print(f"Will run {total_runs} vanilla+lapse fits. Output dir: {args.output_dir}\n")
    
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
                '--output-dir', args.output_dir,
            ]
            print(f"[{idx}/{total_runs}] {' '.join(cmd)}")
        sys.exit(0)

    failures = []
    for run_idx, (batch, animal) in enumerate(pairs, start=1):
        if run_idx < args.start_from:
            print(f"[{run_idx}/{total_runs}] Skipping batch={batch}, animal={animal} (before start-from)")
            continue
            
        print(f"[{run_idx}/{total_runs}] Running batch={batch}, animal={animal} ...")
        cmd = [
            args.python,
            single_script,
            '--batch', batch,
            '--animal', str(animal),
            '--output-dir', args.output_dir,
        ]
        try:
            result = subprocess.run(cmd, cwd=script_dir, check=True)
            print(f"[{run_idx}/{total_runs}] ✓ Done: batch={batch}, animal={animal}\n")
        except subprocess.CalledProcessError as e:
            print(f"[{run_idx}/{total_runs}] ✗ FAILED: batch={batch}, animal={animal} (returncode={e.returncode})\n", 
                  file=sys.stderr)
            failures.append((batch, animal))

    print("\n" + "="*80)
    if failures:
        print(f"⚠ {len(failures)}/{total_runs} runs failed:")
        for batch, animal in failures:
            print(f"   - {batch}:{animal}")
        sys.exit(2)
    else:
        print(f"✓ All {total_runs} runs completed successfully.")
        print("="*80)


if __name__ == '__main__':
    main()
