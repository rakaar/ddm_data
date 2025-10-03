#!/usr/bin/env python3
import argparse
import subprocess
import os
import sys
from typing import List, Tuple

# Default (batch, animal) pairs requested by user
DEFAULT_PAIRS = [
    'SD:52',
    'LED6:84',
    'LED6:86',
    'LED34_even:52',
]


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
    parser.add_argument('--output-dir', default='oct_3_norm_lapse_fit_results',
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

    print(f"Running {len(pairs)} fits. Output dir: {args.output_dir}")
    failures = []
    for i, (batch, animal) in enumerate(pairs, start=1):
        print(f"[{i}/{len(pairs)}] Running batch={batch}, animal={animal} ...")
        cmd = [
            args.python,
            single_script,
            '--batch', batch,
            '--animal', str(animal),
            '--output-dir', args.output_dir,
        ]
        try:
            result = subprocess.run(cmd, cwd=script_dir, check=True)
            print(f"[{i}/{len(pairs)}] Done: batch={batch}, animal={animal}")
        except subprocess.CalledProcessError as e:
            print(f"[{i}/{len(pairs)}] FAILED: batch={batch}, animal={animal} (returncode={e.returncode})", file=sys.stderr)
            failures.append((batch, animal))

    if failures:
        print("\nSome runs failed:")
        for batch, animal in failures:
            print(f" - {batch}:{animal}")
        sys.exit(2)
    else:
        print("\nAll runs completed successfully.")


if __name__ == '__main__':
    main()
