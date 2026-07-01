#!/usr/bin/env python3
# %%
"""
Run vanilla/IPL condition-delay NumPyro SVI for all 30 animals.

This runner discovers animals from `aborts_ipl_npl_time_fit_results/`, runs the
single-animal SVI script with environment variables, and maintains a batch
ledger under the output root.
"""

# %%
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# %%
# =============================================================================
# Paths and defaults
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

FIT_SCRIPT = SCRIPT_DIR / "numpyro_svi_vanilla_condition_delay_single_animal.py"
ABORT_RESULTS_DIR = REPO_DIR / "aborts_ipl_npl_time_fit_results"
OUTPUT_ROOT = SCRIPT_DIR / "numpyro_svi_vanilla_condition_delay_patience12_restore_best_outputs"

DEFAULT_BATCH_ORDER = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]
EXPECTED_N_ANIMALS = 30


# %%
# =============================================================================
# Helpers
# =============================================================================
def rel(path: Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_DIR.resolve()))
    except ValueError:
        return str(path)


def parse_only_items(items):
    if not items:
        return None
    parsed = set()
    for item in items:
        for part in str(item).split(","):
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                raise ValueError(f"--only entries must look like BATCH:ANIMAL, got {part!r}")
            batch, animal_text = part.split(":", 1)
            parsed.add((batch.strip(), int(animal_text)))
    return parsed


def discover_animals():
    pattern = re.compile(r"^results_(?P<batch>.+)_animal_(?P<animal>\d+)\.pkl$")
    pairs = []
    for result_path in ABORT_RESULTS_DIR.glob("results_*_animal_*.pkl"):
        match = pattern.match(result_path.name)
        if match is None:
            continue
        batch_name = match.group("batch")
        if batch_name not in DEFAULT_BATCH_ORDER:
            continue
        pairs.append((batch_name, int(match.group("animal"))))

    batch_order = {batch: idx for idx, batch in enumerate(DEFAULT_BATCH_ORDER)}
    return sorted(pairs, key=lambda pair: (batch_order.get(pair[0], 999), pair[0], pair[1]))


def animal_paths(output_root: Path, batch: str, animal: int, label: str):
    output_dir = output_root / f"{batch}_{animal}"
    return {
        "output_dir": output_dir,
        "posterior_npz": output_dir / f"{label}_posterior_samples.npz",
        "guide_params_pkl": output_dir / f"{label}_guide_params.pkl",
        "posterior_summary_csv": output_dir / f"{label}_posterior_summary.csv",
        "finite_report_csv": output_dir / f"{label}_posterior_finite_report.csv",
        "loss_csv": output_dir / f"{label}_loss.csv",
        "convergence_csv": output_dir / f"{label}_convergence_checks.csv",
        "condition_table_csv": output_dir / "condition_table.csv",
        "loss_png": output_dir / f"{label}_loss.png",
        "delay_intervals_png": output_dir / f"{label}_condition_delay_intervals.png",
        "bundle_pkl": output_dir / f"{label}_variational_posterior_bundle.pkl",
    }


def required_outputs(paths):
    return [
        paths["posterior_npz"],
        paths["guide_params_pkl"],
        paths["posterior_summary_csv"],
        paths["finite_report_csv"],
        paths["loss_csv"],
        paths["convergence_csv"],
        paths["condition_table_csv"],
        paths["loss_png"],
        paths["delay_intervals_png"],
        paths["bundle_pkl"],
    ]


def outputs_complete(paths):
    return all(path.exists() and path.stat().st_size > 0 for path in required_outputs(paths))


def abort_result_path(batch: str, animal: int) -> Path:
    return ABORT_RESULTS_DIR / f"results_{batch}_animal_{animal}.pkl"


def batch_csv_path(batch: str) -> Path:
    return REPO_DIR / "raw_data" / "batch_csvs" / f"batch_{batch}_valid_and_aborts.csv"


def condition_delay_cache_path() -> Path:
    return (
        REPO_DIR
        / "fit_each_condn"
        / "abl_specific_ild2_delay_agreement"
        / "condition_t_E_aff_extraction_cache.csv"
    )


def run_and_log(cmd, env, log_path: Path, cwd: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"Command: {' '.join(str(part) for part in cmd)}\n")
        log.write(f"CWD: {cwd}\n")
        log.write(f"Started: {datetime.now().isoformat(timespec='seconds')}\n\n")
        log.flush()

        process = subprocess.Popen(
            [str(part) for part in cmd],
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()

        elapsed = time.monotonic() - start
        log.write(f"\nFinished: {datetime.now().isoformat(timespec='seconds')}\n")
        log.write(f"Return code: {return_code}\n")
        log.write(f"Elapsed seconds: {elapsed:.3f}\n")
    return return_code, elapsed


def write_ledger(rows, ledger_path: Path):
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "run_index",
        "n_runs",
        "batch_name",
        "animal",
        "status",
        "elapsed_seconds",
        "fit_return_code",
        "bundle_path",
        "output_dir",
        "fit_log",
        "error",
    ]
    with ledger_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def print_pair_table(pairs):
    current_batch = None
    for batch, _animal in pairs:
        if batch != current_batch:
            current_batch = batch
            animals = [str(a) for b, a in pairs if b == batch]
            print(f"  {batch}: {', '.join(animals)}")


# %%
# =============================================================================
# CLI and run loop
# =============================================================================
parser = argparse.ArgumentParser(
    description="Run NumPyro SVI vanilla/IPL condition-delay fits for all 30 animals."
)
parser.add_argument("--dry-run", action="store_true", help="Print planned runs without executing them.")
parser.add_argument("--force", action="store_true", help="Run even if all expected outputs already exist.")
parser.add_argument("--only", nargs="*", help="Restrict to BATCH:ANIMAL entries, e.g. --only LED8:105 LED7:92.")
parser.add_argument("--python", default=str(REPO_DIR / ".venv" / "bin" / "python"), help="Python interpreter to use.")
parser.add_argument("--guide", default="fullrank", help="NumPyro guide kind. Default: fullrank.")
parser.add_argument("--guide-init-scale", type=float, default=0.1, help="Initial latent scale for full-rank guide.")
parser.add_argument("--main-steps", type=int, default=100000, help="Maximum main SVI steps.")
parser.add_argument("--check-every", type=int, default=1000, help="SVI convergence check interval.")
parser.add_argument("--output-root", default=str(OUTPUT_ROOT), help="Output root for all animal fit folders.")
parser.add_argument(
    "--stop-mode",
    default="patience_restore_best",
    choices=["legacy", "stable_or_no_improve", "patience_restore_best"],
    help="SVI stopping rule mode passed to the single-animal script.",
)
parser.add_argument("--rel-tol", type=float, default=0.001, help="Relative change tolerance for stable-window stopping.")
parser.add_argument("--patience-windows", type=int, default=12, help="Stable-window patience.")
parser.add_argument(
    "--no-improve-patience-windows",
    type=int,
    default=12,
    help="No-best-window-improvement patience, measured in check windows.",
)
parser.add_argument(
    "--min-improvement-rel",
    type=float,
    default=0.001,
    help="Relative best-window improvement required to reset no-improve patience.",
)
parser.add_argument("--min-steps", type=int, default=0, help="Minimum SVI steps before early stopping can trigger.")
parser.add_argument("--seed", type=int, default=0, help="NumPyro/JAX random seed.")
parser.add_argument("--posterior-samples", type=int, default=10000, help="Number of posterior samples to save.")
parser.add_argument("--expected-n-animals", type=int, default=EXPECTED_N_ANIMALS, help="Expected discovered animal count.")
parser.add_argument("--stop-on-failure", action="store_true", help="Stop immediately if any fit fails.")
parser.add_argument(
    "--ledger-name",
    default="batch_run_status.csv",
    help="Ledger CSV filename under the output root's _batch_logs folder.",
)
args = parser.parse_args()

output_root = Path(args.output_root).expanduser()
if not output_root.is_absolute():
    output_root = (REPO_DIR / output_root).resolve()
log_dir = output_root / "_batch_logs"
ledger_path = log_dir / args.ledger_name

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
label = f"main_{args.guide}"

if not FIT_SCRIPT.exists():
    raise FileNotFoundError(FIT_SCRIPT)
if not ABORT_RESULTS_DIR.exists():
    raise FileNotFoundError(ABORT_RESULTS_DIR)
if not condition_delay_cache_path().exists():
    raise FileNotFoundError(condition_delay_cache_path())

selected_pairs = parse_only_items(args.only)
all_pairs = discover_animals()
if selected_pairs is None and len(all_pairs) != args.expected_n_animals:
    raise RuntimeError(f"Expected {args.expected_n_animals} animals, discovered {len(all_pairs)}")

pairs = [pair for pair in all_pairs if selected_pairs is None or pair in selected_pairs]
missing_requested = sorted(selected_pairs - set(pairs)) if selected_pairs is not None else []
if missing_requested:
    raise RuntimeError(f"Requested --only animals were not discovered from abort results: {missing_requested}")
if not pairs:
    raise RuntimeError("No animals selected.")

print(f"Run id: {run_id}")
print(f"Repository: {REPO_DIR}")
print(f"Single-animal fit script: {FIT_SCRIPT}")
print(f"Output root: {output_root}")
print(f"Guide label: {label}")
print(
    "Stopping controls: "
    f"steps={args.main_steps}, check_every={args.check_every}, stop_mode={args.stop_mode}, "
    f"rel_tol={args.rel_tol:g}, patience_windows={args.patience_windows}, "
    f"no_improve_patience={args.no_improve_patience_windows}, "
    f"min_improvement_rel={args.min_improvement_rel:g}, min_steps={args.min_steps}"
)
print(f"Selected animals: {len(pairs)}")
print_pair_table(pairs)

preflight_errors = []
for batch, animal in pairs:
    for required_path in [batch_csv_path(batch), abort_result_path(batch, animal), condition_delay_cache_path()]:
        if not required_path.exists():
            preflight_errors.append(f"{batch}/{animal} missing {required_path}")
if preflight_errors:
    raise RuntimeError("Preflight failed:\n" + "\n".join(preflight_errors))

python_path = Path(args.python)
fit_cmd = [python_path, "-u", FIT_SCRIPT]

if args.dry_run:
    print("\nDRY RUN")
    for run_index, (batch, animal) in enumerate(pairs, start=1):
        paths = animal_paths(output_root, batch, animal, label)
        complete = outputs_complete(paths)
        action = "skip existing" if complete and not args.force else "run fit"
        print(f"[{run_index}/{len(pairs)}] {batch}/{animal}: {action}")
        print(f"  fit: NUMPYRO_SVI_BATCH={batch} NUMPYRO_SVI_ANIMAL={animal} {' '.join(str(x) for x in fit_cmd)}")
    raise SystemExit(0)

log_dir.mkdir(parents=True, exist_ok=True)
rows = []
for run_index, (batch, animal) in enumerate(pairs, start=1):
    paths = animal_paths(output_root, batch, animal, label)
    rows.append(
        {
            "run_id": run_id,
            "run_index": run_index,
            "n_runs": len(pairs),
            "batch_name": batch,
            "animal": animal,
            "status": "pending",
            "bundle_path": rel(paths["bundle_pkl"]),
            "output_dir": rel(paths["output_dir"]),
        }
    )
write_ledger(rows, ledger_path)

failures = []
for row in rows:
    batch = row["batch_name"]
    animal = int(row["animal"])
    run_index = int(row["run_index"])
    paths = animal_paths(output_root, batch, animal, label)
    fit_log = log_dir / f"{run_id}_{run_index:02d}_{batch}_{animal}_fit.log"
    row["fit_log"] = rel(fit_log)

    if outputs_complete(paths) and not args.force:
        row["status"] = "skipped_existing"
        row["elapsed_seconds"] = 0.0
        write_ledger(rows, ledger_path)
        print(f"\n[{run_index}/{len(rows)}] Skipping {batch}/{animal}; expected outputs already exist.")
        continue

    print("\n" + "=" * 80)
    print(f"[{run_index}/{len(rows)}] Running {batch}/{animal}")
    print("=" * 80)

    env = os.environ.copy()
    env.update(
        {
            "NUMPYRO_SVI_BATCH": str(batch),
            "NUMPYRO_SVI_ANIMAL": str(animal),
            "NUMPYRO_SVI_OUTPUT_ROOT": str(output_root),
            "NUMPYRO_SVI_GUIDE": str(args.guide),
            "NUMPYRO_SVI_GUIDE_INIT_SCALE": str(args.guide_init_scale),
            "RUN_MAIN_SVI": "1",
            "MAIN_STEPS": str(args.main_steps),
            "SVI_CHECK_EVERY": str(args.check_every),
            "SVI_STOP_MODE": str(args.stop_mode),
            "SVI_REL_TOL": str(args.rel_tol),
            "SVI_PATIENCE_WINDOWS": str(args.patience_windows),
            "SVI_NO_IMPROVE_PATIENCE_WINDOWS": str(args.no_improve_patience_windows),
            "SVI_MIN_IMPROVEMENT_REL": str(args.min_improvement_rel),
            "SVI_MIN_STEPS": str(args.min_steps),
            "NUMPYRO_SVI_SEED": str(args.seed),
            "POSTERIOR_N_SAMPLES": str(args.posterior_samples),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )

    start = time.monotonic()
    fit_return_code, fit_elapsed = run_and_log(fit_cmd, env, fit_log, REPO_DIR)
    row["fit_return_code"] = fit_return_code
    row["elapsed_seconds"] = f"{time.monotonic() - start:.3f}"

    if fit_return_code != 0:
        row["status"] = "fit_failed"
        row["error"] = f"fit returned {fit_return_code}"
        failures.append((batch, animal, row["status"]))
        write_ledger(rows, ledger_path)
        if args.stop_on_failure:
            break
        continue

    missing_outputs = [rel(path) for path in required_outputs(paths) if not path.exists() or path.stat().st_size == 0]
    if missing_outputs:
        row["status"] = "missing_outputs"
        row["error"] = "; ".join(missing_outputs)
        failures.append((batch, animal, row["status"]))
        write_ledger(rows, ledger_path)
        if args.stop_on_failure:
            break
        continue

    row["status"] = "completed"
    row["elapsed_seconds"] = f"{fit_elapsed:.3f}"
    row["bundle_path"] = rel(paths["bundle_pkl"])
    write_ledger(rows, ledger_path)
    print(f"[{run_index}/{len(rows)}] Completed {batch}/{animal}; bundle: {paths['bundle_pkl']}")

print("\n" + "=" * 80)
print("Batch run summary")
print("=" * 80)
status_counts = {}
for row in rows:
    status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
for status, count in sorted(status_counts.items()):
    print(f"  {status}: {count}")
print(f"Ledger: {ledger_path}")
print(f"Logs: {log_dir}")

if failures:
    print("\nFailures:")
    for batch, animal, status in failures:
        print(f"  {batch}/{animal}: {status}")
    raise SystemExit(2)

print("Done.")

# %%
