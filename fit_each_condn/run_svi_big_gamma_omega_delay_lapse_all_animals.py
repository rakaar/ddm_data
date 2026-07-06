#!/usr/bin/env python3
# %%
"""
Run the big condition-wise Gamma/Omega/delay+lapse SVI fit for all animals.

This wraps svi_big_gamma_omega_delay_lapse_single_animal.py so the likelihood and
model stay in one place. Each animal gets condition-wise gamma, omega, and
t_E_aff plus global w, del_go, lapse_prob, and lapse_prob_right.
"""

# %%
from __future__ import annotations

import argparse
import csv
import os
import pickle
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# %%
# =============================================================================
# Paths and defaults
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

FIT_SCRIPT = SCRIPT_DIR / "svi_big_gamma_omega_delay_lapse_single_animal.py"
ANIMAL_SVI_OUTPUT_ROOT = (
    REPO_DIR
    / "fit_animal_by_animal"
    / "numpyro_svi_npl_alpha_condition_delay_single_animal_outputs"
)
OUTPUT_ROOT = SCRIPT_DIR / "svi_big_gamma_omega_delay_lapse_all_animals_outputs"
LOG_DIR = OUTPUT_ROOT / "_batch_logs"
COND_SVI_CACHE = (
    SCRIPT_DIR
    / "svi_gamma_omega_fixed_from_animal_svi_condition_delay_results"
    / "all_observed_with_30k_reruns"
    / "condition_gamma_omega_extraction_cache.csv"
)

DEFAULT_BATCH_ORDER = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]
EXPECTED_N_ANIMALS = 30
POSTERIOR_KEYS = ["gamma", "omega", "t_E_aff", "w", "del_go", "lapse_prob", "lapse_prob_right"]


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
    pairs = []
    for output_dir in ANIMAL_SVI_OUTPUT_ROOT.iterdir():
        if not output_dir.is_dir() or output_dir.name.startswith("_") or "_" not in output_dir.name:
            continue
        batch, animal_text = output_dir.name.rsplit("_", 1)
        if not animal_text.isdigit():
            continue
        posterior_npz = output_dir / "main_fullrank_posterior_samples.npz"
        condition_table = output_dir / "condition_table.csv"
        if posterior_npz.exists() and condition_table.exists():
            pairs.append((batch, int(animal_text)))

    batch_order = {batch: idx for idx, batch in enumerate(DEFAULT_BATCH_ORDER)}
    return sorted(set(pairs), key=lambda pair: (batch_order.get(pair[0], 999), pair[0], pair[1]))


def animal_output_prefix(batch: str, animal: int) -> str:
    return f"{batch}_{animal}_big_gamma_omega_delay_lapse"


def animal_paths(batch: str, animal: int):
    output_dir = OUTPUT_ROOT / f"{batch}_{animal}"
    prefix = animal_output_prefix(batch, animal)
    return {
        "output_dir": output_dir,
        "condition_summary_csv": output_dir / f"{prefix}_condition_summary.csv",
        "posterior_summary_csv": output_dir / f"{prefix}_posterior_summary.csv",
        "loss_csv": output_dir / f"{prefix}_loss.csv",
        "convergence_csv": output_dir / f"{prefix}_convergence_checks.csv",
        "posterior_npz": output_dir / f"{prefix}_posterior_samples.npz",
        "bundle_pkl": output_dir / f"{prefix}_fit_bundle.pkl",
        "condition_table_csv": output_dir / f"{prefix}_condition_table.csv",
        "loss_png": output_dir / f"{prefix}_loss.png",
        "condition_params_png": output_dir / f"{prefix}_condition_params.png",
    }


def source_animal_paths(batch: str, animal: int):
    output_dir = ANIMAL_SVI_OUTPUT_ROOT / f"{batch}_{animal}"
    return {
        "output_dir": output_dir,
        "posterior_npz": output_dir / "main_fullrank_posterior_samples.npz",
        "condition_table_csv": output_dir / "condition_table.csv",
    }


def batch_csv_path(batch: str) -> Path:
    return REPO_DIR / "raw_data" / "batch_csvs" / f"batch_{batch}_valid_and_aborts.csv"


def abort_result_path(batch: str, animal: int) -> Path:
    return REPO_DIR / "aborts_ipl_npl_time_fit_results" / f"results_{batch}_animal_{animal}.pkl"


def required_outputs(paths):
    return [
        paths["condition_summary_csv"],
        paths["posterior_summary_csv"],
        paths["loss_csv"],
        paths["convergence_csv"],
        paths["posterior_npz"],
        paths["bundle_pkl"],
        paths["condition_table_csv"],
        paths["loss_png"],
        paths["condition_params_png"],
    ]


def outputs_exist(paths):
    return all(path.exists() and path.stat().st_size > 0 for path in required_outputs(paths))


def verify_outputs(paths):
    missing = [rel(path) for path in required_outputs(paths) if not path.exists() or path.stat().st_size == 0]
    if missing:
        return False, "missing outputs: " + "; ".join(missing), ""

    try:
        with paths["bundle_pkl"].open("rb") as handle:
            bundle = pickle.load(handle)
    except Exception as exc:
        return False, f"could not read bundle: {exc!r}", ""

    posterior_samples = bundle.get("posterior_samples", None)
    if not isinstance(posterior_samples, dict):
        return False, "bundle missing posterior_samples dict", str(bundle.get("stop_reason", ""))
    for key in POSTERIOR_KEYS:
        if key not in posterior_samples:
            return False, f"bundle missing posterior sample key {key!r}", str(bundle.get("stop_reason", ""))
        values = np.asarray(posterior_samples[key])
        if values.size == 0 or not np.all(np.isfinite(values)):
            return False, f"posterior sample key {key!r} is empty or non-finite", str(bundle.get("stop_reason", ""))

    try:
        convergence_df = pd.read_csv(paths["convergence_csv"])
    except Exception as exc:
        return False, f"could not read convergence csv: {exc!r}", str(bundle.get("stop_reason", ""))
    if "n_nonfinite" in convergence_df.columns and int(convergence_df["n_nonfinite"].sum()) != 0:
        return False, "convergence csv reports non-finite losses", str(bundle.get("stop_reason", ""))

    return True, "", str(bundle.get("stop_reason", ""))


def print_pair_table(pairs):
    for batch in DEFAULT_BATCH_ORDER:
        animals = [str(animal) for pair_batch, animal in pairs if pair_batch == batch]
        if animals:
            print(f"  {batch}: {', '.join(animals)}")
    other_batches = sorted(set(batch for batch, _ in pairs) - set(DEFAULT_BATCH_ORDER))
    for batch in other_batches:
        animals = [str(animal) for pair_batch, animal in pairs if pair_batch == batch]
        print(f"  {batch}: {', '.join(animals)}")


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
        "return_code",
        "stop_reason",
        "stop_mode",
        "steps",
        "check_every",
        "min_steps",
        "no_improve_patience_windows",
        "min_improvement_rel",
        "bundle_path",
        "output_dir",
        "log_path",
        "error",
    ]
    with ledger_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


# %%
# =============================================================================
# CLI and preflight
# =============================================================================
parser = argparse.ArgumentParser(description="Run big Gamma/Omega/delay+lapse SVI fits for all animals.")
parser.add_argument("--dry-run", action="store_true", help="Print selected animals and actions without running fits.")
parser.add_argument("--force", action="store_true", help="Run even when all expected outputs already exist.")
parser.add_argument("--only", nargs="*", help="Restrict to BATCH:ANIMAL entries, e.g. --only LED8:105 LED7:92.")
parser.add_argument("--python", default=str(REPO_DIR / ".venv" / "bin" / "python"), help="Python interpreter.")
parser.add_argument("--guide", default="fullrank", help="NumPyro guide kind. Default: fullrank.")
parser.add_argument("--steps", type=int, default=20000, help="Maximum SVI steps per animal.")
parser.add_argument("--check-every", type=int, default=1000, help="SVI convergence check interval.")
parser.add_argument("--min-steps", type=int, default=0, help="Minimum SVI steps before early stopping can trigger.")
parser.add_argument(
    "--stop-mode",
    default="legacy",
    choices=["legacy", "stable_or_no_improve", "patience_restore_best"],
    help="SVI stopping rule mode passed to the single-animal script.",
)
parser.add_argument(
    "--no-improve-patience-windows",
    type=int,
    default=5,
    help="No-best-window-improvement patience, measured in check windows.",
)
parser.add_argument(
    "--min-improvement-rel",
    type=float,
    default=0.001,
    help="Relative best-window improvement required to reset no-improve patience.",
)
parser.add_argument("--posterior-samples", type=int, default=10000, help="Posterior samples to save per animal.")
parser.add_argument("--seed", type=int, default=0, help="Base NumPyro/JAX random seed.")
parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate.")
parser.add_argument("--clip-norm", type=float, default=2.0, help="ClippedAdam clip norm.")
parser.add_argument("--output-root", default=str(OUTPUT_ROOT), help="Output root for all animal fit folders.")
parser.add_argument("--expected-n-animals", type=int, default=EXPECTED_N_ANIMALS, help="Expected discovered animal count.")
parser.add_argument("--stop-on-failure", action="store_true", help="Stop immediately after any failed animal.")
args = parser.parse_args()

OUTPUT_ROOT = Path(args.output_root).expanduser()
if not OUTPUT_ROOT.is_absolute():
    OUTPUT_ROOT = (REPO_DIR / OUTPUT_ROOT).resolve()
LOG_DIR = OUTPUT_ROOT / "_batch_logs"

if not FIT_SCRIPT.exists():
    raise FileNotFoundError(FIT_SCRIPT)
if not ANIMAL_SVI_OUTPUT_ROOT.exists():
    raise FileNotFoundError(ANIMAL_SVI_OUTPUT_ROOT)
if not COND_SVI_CACHE.exists():
    raise FileNotFoundError(COND_SVI_CACHE)

selected_pairs = parse_only_items(args.only)
all_pairs = discover_animals()
if selected_pairs is None and len(all_pairs) != args.expected_n_animals:
    raise RuntimeError(f"Expected {args.expected_n_animals} animals, discovered {len(all_pairs)}")

pairs = [pair for pair in all_pairs if selected_pairs is None or pair in selected_pairs]
missing_requested = sorted(selected_pairs - set(pairs)) if selected_pairs is not None else []
if missing_requested:
    raise RuntimeError(f"Requested --only animals were not discovered from source SVI outputs: {missing_requested}")
if not pairs:
    raise RuntimeError("No animals selected.")

preflight_errors = []
for batch, animal in pairs:
    source_paths = source_animal_paths(batch, animal)
    for required_path in [
        source_paths["posterior_npz"],
        source_paths["condition_table_csv"],
        batch_csv_path(batch),
        abort_result_path(batch, animal),
    ]:
        if not required_path.exists():
            preflight_errors.append(f"{batch}/{animal} missing {required_path}")
if preflight_errors:
    raise RuntimeError("Preflight failed:\n" + "\n".join(preflight_errors))

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
python_path = Path(args.python)
fit_cmd = [python_path, "-u", FIT_SCRIPT]

print(f"Run id: {RUN_ID}")
print(f"Repository: {REPO_DIR}")
print(f"Single-animal fit script: {FIT_SCRIPT}")
print(f"Output root: {OUTPUT_ROOT}")
print(f"Guide: {args.guide}")
print(f"Steps/check_every/min_steps/posterior_samples: {args.steps}/{args.check_every}/{args.min_steps}/{args.posterior_samples}")
print(
    "Stop mode: "
    f"{args.stop_mode}; no_improve_patience={args.no_improve_patience_windows}; "
    f"min_improvement_rel={args.min_improvement_rel:g}"
)
print(f"Selected animals: {len(pairs)}")
print_pair_table(pairs)

if args.dry_run:
    print("\nDRY RUN")
    for run_index, (batch, animal) in enumerate(pairs, start=1):
        paths = animal_paths(batch, animal)
        action = "skip existing" if outputs_exist(paths) and not args.force else "run fit"
        print(f"[{run_index}/{len(pairs)}] {batch}/{animal}: {action}")
        print(
            "  env "
            f"NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_BATCH={batch} "
            f"NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_ANIMAL={animal} "
            f"NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_OUTPUT_ROOT={OUTPUT_ROOT} "
            f"NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_STOP_MODE={args.stop_mode} "
            f"NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_MIN_STEPS={args.min_steps} "
            f"NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_NO_IMPROVE_PATIENCE_WINDOWS={args.no_improve_patience_windows} "
            f"NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_MIN_IMPROVEMENT_REL={args.min_improvement_rel:g} "
            f"{' '.join(str(part) for part in fit_cmd)}"
        )
    raise SystemExit(0)


# %%
# =============================================================================
# Run loop
# =============================================================================
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
ledger_path = LOG_DIR / "batch_run_status.csv"

rows = []
for run_index, (batch, animal) in enumerate(pairs, start=1):
    paths = animal_paths(batch, animal)
    rows.append(
        {
            "run_id": RUN_ID,
            "run_index": run_index,
            "n_runs": len(pairs),
            "batch_name": batch,
            "animal": animal,
            "status": "pending",
            "stop_mode": args.stop_mode,
            "steps": args.steps,
            "check_every": args.check_every,
            "min_steps": args.min_steps,
            "no_improve_patience_windows": args.no_improve_patience_windows,
            "min_improvement_rel": args.min_improvement_rel,
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
    paths = animal_paths(batch, animal)
    log_path = LOG_DIR / f"{RUN_ID}_{run_index:02d}_{batch}_{animal}.log"
    row["log_path"] = rel(log_path)

    if outputs_exist(paths) and not args.force:
        ok, error, stop_reason = verify_outputs(paths)
        if ok:
            row["status"] = "skipped_existing"
            row["elapsed_seconds"] = "0.000"
            row["return_code"] = 0
            row["stop_reason"] = stop_reason
            write_ledger(rows, ledger_path)
            print(f"\n[{run_index}/{len(rows)}] Skipping {batch}/{animal}; verified existing outputs.")
            continue
        row["status"] = "existing_incomplete"
        row["error"] = error
        failures.append((batch, animal, row["status"]))
        write_ledger(rows, ledger_path)
        if args.stop_on_failure:
            break

    print("\n" + "=" * 80)
    print(f"[{run_index}/{len(rows)}] Running {batch}/{animal}")
    print("=" * 80)

    env = os.environ.copy()
    env.update(
        {
            "NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_BATCH": str(batch),
            "NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_ANIMAL": str(animal),
            "NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_OUTPUT_ROOT": str(OUTPUT_ROOT),
            "NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_GUIDE": str(args.guide),
            "NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_STEPS": str(args.steps),
            "NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_CHECK_EVERY": str(args.check_every),
            "NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_STOP_MODE": str(args.stop_mode),
            "NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_MIN_STEPS": str(args.min_steps),
            "NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_NO_IMPROVE_PATIENCE_WINDOWS": str(
                args.no_improve_patience_windows
            ),
            "NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_MIN_IMPROVEMENT_REL": str(args.min_improvement_rel),
            "NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_POSTERIOR_N_SAMPLES": str(args.posterior_samples),
            "NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_SEED": str(args.seed),
            "NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_LR": str(args.lr),
            "NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_CLIP_NORM": str(args.clip_norm),
            "NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_OPTIMIZER": "clipped_adam",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )

    total_start = time.monotonic()
    return_code, elapsed = run_and_log(fit_cmd, env, log_path, REPO_DIR)
    row["return_code"] = return_code
    row["elapsed_seconds"] = f"{elapsed:.3f}"
    if return_code != 0:
        row["status"] = "fit_failed"
        row["error"] = f"fit returned {return_code}"
        failures.append((batch, animal, row["status"]))
        write_ledger(rows, ledger_path)
        if args.stop_on_failure:
            break
        continue

    ok, error, stop_reason = verify_outputs(paths)
    row["stop_reason"] = stop_reason
    row["elapsed_seconds"] = f"{time.monotonic() - total_start:.3f}"
    if not ok:
        row["status"] = "output_verify_failed"
        row["error"] = error
        failures.append((batch, animal, row["status"]))
        write_ledger(rows, ledger_path)
        if args.stop_on_failure:
            break
        continue

    row["status"] = "completed"
    write_ledger(rows, ledger_path)
    print(f"[{run_index}/{len(rows)}] Completed {batch}/{animal}; stop_reason={stop_reason}")

print("\n" + "=" * 80)
print("Batch run summary")
print("=" * 80)
status_counts = {}
for row in rows:
    status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
for status, count in sorted(status_counts.items()):
    print(f"  {status}: {count}")
print(f"Ledger: {ledger_path}")
print(f"Logs: {LOG_DIR}")

if failures:
    print("\nFailures:")
    for batch, animal, status in failures:
        print(f"  {batch}/{animal}: {status}")
    raise SystemExit(1)
