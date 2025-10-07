#!/usr/bin/env python3
import argparse
import subprocess
import os
import sys
from typing import List, Tuple

# Default (batch, animal) pairs requested by user
DEFAULT_PAIRS = [
    # 'LED34_even:48',
    # 'LED7:103',
    # 'LED7:93',
    # 'LED6:82',
    'LED34_even:52',
    # 'LED34:45',
    'LED34_even:60',
    # 'LED34:57',
    # 'LED34:63',
    'LED34_even:56',
    # 'LED6:81',
    # 'LED8:105',
    # 'LED6:86',
    # 'LED7:98',
    # 'LED8:112',
    # 'LED8:109',
    # 'LED6:84',
    # 'LED34:61',
    # 'LED34:59',
    # 'LED7:100',
    # 'LED7:92',
    # 'LED8:107',
    # 'LED7:99',
    # 'LED8:108',
]

# Init types to run for each pair
DEFAULT_INIT_TYPES = ['vanilla', 'norm']


def parse_pairs(pairs_list: List[str]) -> List[Tuple[str, int]]:
    out = []
    for item in pairs_list:
        # Expected format: BATCH:ANIMAL (e.g., LED8:105)
        if ':' not in item:
            raise ValueError(f"Invalid pair '{item}'. Expected format 'BATCH:ANIMAL' (e.g., LED8:105)")
        batch, animal_str = item.split(':', 1)
        try:
            animal = int(animal_str)
        except ValueError:
            raise ValueError(f"Animal ID must be an integer in pair '{item}'")
        out.append((batch, animal))
    return out


def main():
    parser = argparse.ArgumentParser(description='Run norm+lapse fits for multiple (batch, animal) pairs')
    parser.add_argument('--pairs', nargs='+', default=DEFAULT_PAIRS,
                        help="List of pairs in the form BATCH:ANIMAL, e.g., --pairs LED8:105 LED34_even:207. Default runs: "
                             + ", ".join(DEFAULT_PAIRS))
    parser.add_argument('--init-types', nargs='+', default=DEFAULT_INIT_TYPES, choices=['vanilla', 'norm'],
                        help='Init types to run for each pair. Default: both vanilla and norm')
    parser.add_argument('--output-dir', default='oct_6_7_large_bounds_diff_init_lapse_fit',
                        help='Directory to save results for all runs')
    parser.add_argument('--python', default=sys.executable,
                        help='Python interpreter to use for running the single-animal script')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    single_script = os.path.join(script_dir, 'lapses_fit_single_animal_norm_model.py')

    if not os.path.exists(single_script):
        print(f"Could not find single-animal script at {single_script}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    pairs = parse_pairs(args.pairs)
    init_types = args.init_types

    total_runs = len(pairs) * len(init_types)
    print(f"Running {total_runs} fits ({len(pairs)} pairs × {len(init_types)} init types). Output dir: {args.output_dir}")
    failures = []
    run_idx = 0
    for batch, animal in pairs:
        for init_type in init_types:
            run_idx += 1
            print(f"[{run_idx}/{total_runs}] Running batch={batch}, animal={animal}, init_type={init_type} ...")
            cmd = [
                args.python,
                single_script,
                '--batch', batch,
                '--animal', str(animal),
                '--init-type', init_type,
                '--output-dir', args.output_dir,
            ]
            try:
                result = subprocess.run(cmd, cwd=script_dir, check=True)
                print(f"[{run_idx}/{total_runs}] Done: batch={batch}, animal={animal}, init_type={init_type}")
            except subprocess.CalledProcessError as e:
                print(f"[{run_idx}/{total_runs}] FAILED: batch={batch}, animal={animal}, init_type={init_type} (returncode={e.returncode})", file=sys.stderr)
                failures.append((batch, animal, init_type))

    if failures:
        print("\nSome runs failed:")
        for batch, animal, init_type in failures:
            print(f" - {batch}:{animal} (init_type={init_type})")
        sys.exit(2)
    else:
        print("\nAll runs completed successfully.")


if __name__ == '__main__':
    main()
