#!/usr/bin/env python3
# %%
"""
Sequential tmux-friendly runner for the six LED7 reactive-only NPL+alpha fits.

Each fit uses valid trials with 0.100 <= RTwrtStim < 1.000 s and the pure
reactive RT+choice likelihood. The runner writes per-animal logs and updates a
batch ledger after every status change.
"""

# %%
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pandas as pd


# %%
# =============================================================================
# Paths and fixed animal set
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

FIT_SCRIPT = (
    SCRIPT_DIR
    / "numpyro_svi_npl_alpha_reactive_condition_delay_single_animal.py"
)
REFERENCE_ROOT = (
    SCRIPT_DIR
    / "numpyro_svi_npl_alpha_condition_delay_patience12_restore_best_outputs"
)
DEFAULT_OUTPUT_ROOT = (
    SCRIPT_DIR
    / "numpyro_svi_npl_alpha_reactive_only_rt_ge_100ms_condition_delay_"
    "patience12_min50k_restore_best_outputs"
)

BATCH_NAME = "LED7"
LED7_ANIMALS = [92, 93, 98, 99, 100, 103]
RT_LOWER = 0.100
RT_UPPER = 1.000
LABEL = "main_fullrank"


# %%
# =============================================================================
# Small helpers
# =============================================================================
def rel(path):
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_DIR.resolve()))
    except ValueError:
        return str(path)


def output_paths(output_root, animal):
    output_dir = output_root / f"{BATCH_NAME}_{animal}"
    return {
        "output_dir": output_dir,
        "posterior_npz": output_dir / f"{LABEL}_posterior_samples.npz",
        "guide_params_pkl": output_dir / f"{LABEL}_guide_params.pkl",
        "posterior_summary_csv": output_dir / f"{LABEL}_posterior_summary.csv",
        "finite_report_csv": output_dir / f"{LABEL}_posterior_finite_report.csv",
        "condition_table_csv": output_dir / "condition_table.csv",
        "loss_csv": output_dir / f"{LABEL}_loss.csv",
        "convergence_csv": output_dir / f"{LABEL}_convergence_checks.csv",
        "metadata_json": output_dir / f"{LABEL}_run_metadata.json",
        "loss_png": output_dir / f"{LABEL}_loss.png",
        "global_corner_png": output_dir / f"{LABEL}_global_corner.png",
        "selected_corner_png": (
            output_dir / f"{LABEL}_global_selected_delay_corner.png"
        ),
        "delay_png": output_dir / f"{LABEL}_condition_delay_intervals.png",
        "bundle_pkl": (
            output_dir / f"{LABEL}_variational_posterior_bundle.pkl"
        ),
    }


def required_outputs(paths):
    return [
        path
        for key, path in paths.items()
        if key != "output_dir"
    ]


def outputs_complete(paths):
    return all(
        path.exists() and path.stat().st_size > 0
        for path in required_outputs(paths)
    )


def write_ledger(rows, ledger_path):
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "run_index",
        "n_runs",
        "batch_name",
        "animal",
        "status",
        "n_valid_rt_0_to_1",
        "n_retained_rt_100ms_to_1s",
        "n_removed_below_100ms",
        "n_conditions",
        "elapsed_seconds",
        "return_code",
        "output_dir",
        "fit_log",
        "error",
    ]
    with ledger_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: row.get(key, "") for key in fieldnames}
            )


def run_and_log(command, env, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"Command: {' '.join(str(part) for part in command)}\n")
        log.write(f"CWD: {REPO_DIR}\n")
        log.write(f"Started: {datetime.now().isoformat(timespec='seconds')}\n\n")
        log.flush()

        process = subprocess.Popen(
            [str(part) for part in command],
            cwd=str(REPO_DIR),
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
            log.flush()
        return_code = process.wait()
        elapsed = time.monotonic() - start
        log.write(f"\nFinished: {datetime.now().isoformat(timespec='seconds')}\n")
        log.write(f"Return code: {return_code}\n")
        log.write(f"Elapsed seconds: {elapsed:.3f}\n")
    return return_code, elapsed


# %%
# =============================================================================
# CLI and preflight
# =============================================================================
parser = argparse.ArgumentParser(
    description=(
        "Run six LED7 reactive-only NPL+alpha condition-delay SVI fits "
        "sequentially."
    )
)
parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Print the selected animals and preflight counts without fitting.",
)
parser.add_argument(
    "--force",
    action="store_true",
    help="Overwrite complete fit folders.",
)
parser.add_argument(
    "--only",
    nargs="*",
    type=int,
    help="Restrict the run to one or more LED7 animal IDs.",
)
parser.add_argument(
    "--python",
    default=str(REPO_DIR / ".venv" / "bin" / "python"),
)
parser.add_argument(
    "--output-root",
    default=str(DEFAULT_OUTPUT_ROOT),
)
parser.add_argument("--main-steps", type=int, default=150000)
parser.add_argument("--check-every", type=int, default=1000)
parser.add_argument("--min-steps", type=int, default=50000)
parser.add_argument("--no-improve-patience-windows", type=int, default=12)
parser.add_argument("--min-improvement-rel", type=float, default=0.001)
parser.add_argument("--posterior-samples", type=int, default=10000)
parser.add_argument("--learning-rate", type=float, default=0.0002)
parser.add_argument("--clip-norm", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--stop-on-failure",
    action="store_true",
    help="Do not continue to later animals after a failed fit.",
)
args = parser.parse_args()

output_root = Path(args.output_root).expanduser()
if not output_root.is_absolute():
    output_root = (REPO_DIR / output_root).resolve()
log_dir = output_root / "_batch_logs"
ledger_path = log_dir / "batch_run_status.csv"
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

if not FIT_SCRIPT.exists():
    raise FileNotFoundError(FIT_SCRIPT)
python_path = Path(args.python)
if not python_path.exists():
    raise FileNotFoundError(python_path)

selected_animals = LED7_ANIMALS
if args.only:
    unknown = sorted(set(args.only) - set(LED7_ANIMALS))
    if unknown:
        raise ValueError(f"Unknown LED7 animal IDs in --only: {unknown}")
    selected_animals = [
        animal for animal in LED7_ANIMALS if animal in set(args.only)
    ]
if not selected_animals:
    raise RuntimeError("No animals selected.")

batch_csv = (
    REPO_DIR
    / "raw_data"
    / "batch_csvs"
    / f"batch_{BATCH_NAME}_valid_and_aborts.csv"
)
if not batch_csv.exists():
    raise FileNotFoundError(batch_csv)
raw_df = pd.read_csv(batch_csv)
if "choice" not in raw_df.columns:
    raw_df["choice"] = raw_df["response_poke"].map({3: 1, 2: -1})

rows = []
preflight_errors = []
for run_index, animal in enumerate(selected_animals, start=1):
    reference_dir = REFERENCE_ROOT / f"{BATCH_NAME}_{animal}"
    reference_npz = (
        reference_dir / "main_fullrank_posterior_samples.npz"
    )
    reference_conditions = reference_dir / "condition_table.csv"
    for required_path in [reference_npz, reference_conditions]:
        if not required_path.exists():
            preflight_errors.append(
                f"{BATCH_NAME}/{animal} missing {required_path}"
            )

    animal_df = raw_df[
        (raw_df["animal"].astype(int) == animal)
        & raw_df["success"].isin([1, -1])
        & (raw_df["RTwrtStim"] >= 0)
        & (raw_df["RTwrtStim"] < RT_UPPER)
        & raw_df["ABL"].isin([20, 40, 60])
    ].dropna(subset=["RTwrtStim", "ABL", "ILD", "choice"])
    retained_df = animal_df[
        (animal_df["RTwrtStim"] >= RT_LOWER)
        & (animal_df["RTwrtStim"] < RT_UPPER)
    ]
    n_conditions = len(
        retained_df[["ABL", "ILD"]].drop_duplicates()
    )
    if n_conditions != 30:
        preflight_errors.append(
            f"{BATCH_NAME}/{animal} has {n_conditions} retained conditions, "
            "expected 30"
        )

    paths = output_paths(output_root, animal)
    rows.append(
        {
            "run_id": run_id,
            "run_index": run_index,
            "n_runs": len(selected_animals),
            "batch_name": BATCH_NAME,
            "animal": animal,
            "status": "pending",
            "n_valid_rt_0_to_1": len(animal_df),
            "n_retained_rt_100ms_to_1s": len(retained_df),
            "n_removed_below_100ms": len(animal_df) - len(retained_df),
            "n_conditions": n_conditions,
            "output_dir": rel(paths["output_dir"]),
        }
    )

if preflight_errors:
    raise RuntimeError("Preflight failed:\n" + "\n".join(preflight_errors))

print(f"Run id: {run_id}")
print(f"Fit script: {FIT_SCRIPT}")
print(f"Initialization root: {REFERENCE_ROOT}")
print(f"Output root: {output_root}")
print(f"Animals: {selected_animals}")
print(
    "Fit: pure reactive RT+choice, 0.100 <= RTwrtStim < 1.000 s, "
    "six shared NPL+alpha parameters + 30 condition delays"
)
print(
    "Stopping: "
    f"max_steps={args.main_steps}, check_every={args.check_every}, "
    f"min_steps={args.min_steps}, min_improvement_rel="
    f"{args.min_improvement_rel:g}, no_improve_patience="
    f"{args.no_improve_patience_windows}, restore_best=True"
)
print("\nPreflight trial counts:")
print(
    pd.DataFrame(rows)[
        [
            "animal",
            "n_valid_rt_0_to_1",
            "n_retained_rt_100ms_to_1s",
            "n_removed_below_100ms",
            "n_conditions",
        ]
    ].to_string(index=False)
)

if args.dry_run:
    print("\nDRY RUN: no fits started.")
    for row in rows:
        paths = output_paths(output_root, int(row["animal"]))
        action = (
            "skip complete"
            if outputs_complete(paths) and not args.force
            else "run fit"
        )
        print(f"  {BATCH_NAME}/{row['animal']}: {action}")
    raise SystemExit(0)


# %%
# =============================================================================
# Sequential fit loop
# =============================================================================
log_dir.mkdir(parents=True, exist_ok=True)
write_ledger(rows, ledger_path)
fit_command = [python_path, "-u", FIT_SCRIPT]
failures = []

for row in rows:
    animal = int(row["animal"])
    run_index = int(row["run_index"])
    paths = output_paths(output_root, animal)
    fit_log = (
        log_dir
        / f"{run_id}_{run_index:02d}_{BATCH_NAME}_{animal}_fit.log"
    )
    row["fit_log"] = rel(fit_log)

    if outputs_complete(paths) and not args.force:
        row["status"] = "skipped_existing"
        row["elapsed_seconds"] = "0.000"
        write_ledger(rows, ledger_path)
        print(
            f"\n[{run_index}/{len(rows)}] Skipping {BATCH_NAME}/{animal}; "
            "all expected outputs exist."
        )
        continue

    print("\n" + "=" * 80)
    print(f"[{run_index}/{len(rows)}] Running {BATCH_NAME}/{animal}")
    print("=" * 80)
    row["status"] = "running"
    write_ledger(rows, ledger_path)

    env = os.environ.copy()
    env.update(
        {
            "NUMPYRO_SVI_BATCH": BATCH_NAME,
            "NUMPYRO_SVI_ANIMAL": str(animal),
            "NUMPYRO_SVI_OUTPUT_ROOT": str(output_root),
            "NPL_REACTIVE_INIT_ROOT": str(REFERENCE_ROOT),
            "NPL_REACTIVE_RT_LOWER": str(RT_LOWER),
            "NPL_REACTIVE_RT_UPPER": str(RT_UPPER),
            "NUMPYRO_SVI_GUIDE": "fullrank",
            "RUN_MAIN_SVI": "1",
            "MAIN_STEPS": str(args.main_steps),
            "SVI_CHECK_EVERY": str(args.check_every),
            "SVI_EARLY_STOP": "1",
            "SVI_STOP_MODE": "patience_restore_best",
            "SVI_REL_TOL": "0.001",
            "SVI_PATIENCE_WINDOWS": "12",
            "SVI_MIN_IMPROVEMENT_REL": str(
                args.min_improvement_rel
            ),
            "SVI_NO_IMPROVE_PATIENCE_WINDOWS": str(
                args.no_improve_patience_windows
            ),
            "SVI_MIN_STEPS": str(args.min_steps),
            "NUMPYRO_SVI_LR": str(args.learning_rate),
            "NUMPYRO_SVI_OPTIMIZER": "clipped_adam",
            "NUMPYRO_SVI_CLIP_NORM": str(args.clip_norm),
            "POSTERIOR_N_SAMPLES": str(args.posterior_samples),
            "NUMPYRO_SVI_SEED": str(args.seed),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )

    return_code, elapsed = run_and_log(
        fit_command,
        env,
        fit_log,
    )
    row["return_code"] = return_code
    row["elapsed_seconds"] = f"{elapsed:.3f}"
    if return_code != 0:
        row["status"] = "failed"
        row["error"] = f"fit returned {return_code}"
        failures.append((BATCH_NAME, animal))
        write_ledger(rows, ledger_path)
        if args.stop_on_failure:
            break
        continue

    missing_outputs = [
        rel(path)
        for path in required_outputs(paths)
        if not path.exists() or path.stat().st_size == 0
    ]
    if missing_outputs:
        row["status"] = "incomplete_outputs"
        row["error"] = "; ".join(missing_outputs)
        failures.append((BATCH_NAME, animal))
        write_ledger(rows, ledger_path)
        if args.stop_on_failure:
            break
        continue

    row["status"] = "completed"
    write_ledger(rows, ledger_path)
    print(
        f"[{run_index}/{len(rows)}] Completed {BATCH_NAME}/{animal} "
        f"in {elapsed / 60.0:.2f} min."
    )

print("\n" + "=" * 80)
print("LED7 reactive-only batch summary")
print("=" * 80)
status_counts = {}
for row in rows:
    status = row["status"]
    status_counts[status] = status_counts.get(status, 0) + 1
for status, count in sorted(status_counts.items()):
    print(f"  {status}: {count}")
print(f"Ledger: {ledger_path}")
print(f"Logs: {log_dir}")

if failures:
    print("Failures:")
    for batch, animal in failures:
        print(f"  {batch}/{animal}")
    raise SystemExit(2)

print("Done.")

# %%
