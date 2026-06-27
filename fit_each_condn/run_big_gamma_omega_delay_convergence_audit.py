#!/usr/bin/env python3
# %%
"""
Run no-early-stop convergence audits for selected big Gamma/Omega/delay SVI fits.

This keeps the original single-animal model script unchanged and reruns selected
animals from the same initialization with early stopping disabled. The goal is
to test whether posterior means would have moved materially if the original
`stable_3_windows` criterion had not stopped the fit.
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
# Parameters
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
FIT_SCRIPT = SCRIPT_DIR / "svi_big_gamma_omega_delay_single_animal.py"

ORIGINAL_OUTPUT_ROOT = SCRIPT_DIR / "svi_big_gamma_omega_delay_all_animals_outputs"
AUDIT_OUTPUT_ROOT = (
    SCRIPT_DIR
    / "svi_big_gamma_omega_delay_convergence_audit_outputs"
    / "no_early_stop_50k"
)
LOG_DIR = AUDIT_OUTPUT_ROOT / "_audit_logs"

DEFAULT_SELECTED_PAIRS = [
    ("SD", 49),
    ("LED8", 105),
    ("SD", 52),
    ("LED6", 81),
    ("LED34", 63),
    ("LED7", 103),
]
POSTERIOR_KEYS = ["gamma", "omega", "t_E_aff", "w", "del_go"]


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


def parse_pair_items(items):
    if not items:
        return DEFAULT_SELECTED_PAIRS
    pairs = []
    for item in items:
        for part in str(item).split(","):
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                raise ValueError(f"Animal entries must look like BATCH:ANIMAL, got {part!r}")
            batch, animal_text = part.split(":", 1)
            pairs.append((batch.strip(), int(animal_text)))
    return pairs


def animal_output_prefix(batch: str, animal: int) -> str:
    return f"{batch}_{animal}_big_gamma_omega_delay"


def animal_paths(root: Path, batch: str, animal: int):
    output_dir = root / f"{batch}_{animal}"
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
        "original_bundle_path",
        "audit_bundle_path",
        "audit_output_dir",
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
parser = argparse.ArgumentParser(description="Run no-early-stop convergence audits for big SVI fits.")
parser.add_argument("--dry-run", action="store_true", help="Print selected animals without running fits.")
parser.add_argument("--force", action="store_true", help="Run even when audit outputs already exist.")
parser.add_argument("--only", nargs="*", help="Restrict to BATCH:ANIMAL entries. Defaults to the six audit animals.")
parser.add_argument("--python", default=str(REPO_DIR / ".venv" / "bin" / "python"), help="Python interpreter.")
parser.add_argument("--steps", type=int, default=50000, help="Maximum SVI steps per animal.")
parser.add_argument("--check-every", type=int, default=1000, help="SVI convergence check interval.")
parser.add_argument("--posterior-samples", type=int, default=10000, help="Posterior samples to save per animal.")
parser.add_argument("--guide", default="fullrank", help="NumPyro guide kind.")
parser.add_argument("--seed", type=int, default=0, help="Base NumPyro/JAX random seed.")
parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate.")
parser.add_argument("--clip-norm", type=float, default=2.0, help="ClippedAdam clip norm.")
parser.add_argument("--output-root", default=str(AUDIT_OUTPUT_ROOT), help="Audit output root.")
parser.add_argument("--stop-on-failure", action="store_true", help="Stop immediately after any failed animal.")
args = parser.parse_args()

audit_output_root = Path(args.output_root).expanduser()
if not audit_output_root.is_absolute():
    audit_output_root = (REPO_DIR / audit_output_root).resolve()
log_dir = audit_output_root / "_audit_logs"

if not FIT_SCRIPT.exists():
    raise FileNotFoundError(FIT_SCRIPT)

pairs = parse_pair_items(args.only)
if not pairs:
    raise RuntimeError("No animals selected.")

for batch, animal in pairs:
    original_paths = animal_paths(ORIGINAL_OUTPUT_ROOT, batch, animal)
    missing = [path for path in required_outputs(original_paths) if not path.exists()]
    if missing:
        raise RuntimeError(f"{batch}/{animal} missing original outputs:\n" + "\n".join(str(path) for path in missing))

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
fit_cmd = [Path(args.python), "-u", FIT_SCRIPT]

print(f"Run id: {run_id}")
print(f"Repository: {REPO_DIR}")
print(f"Single-animal fit script: {FIT_SCRIPT}")
print(f"Original output root: {ORIGINAL_OUTPUT_ROOT}")
print(f"Audit output root: {audit_output_root}")
print(f"Steps/check_every/posterior_samples: {args.steps}/{args.check_every}/{args.posterior_samples}")
print(f"Early stopping: disabled")
print("Selected animals:")
for batch, animal in pairs:
    print(f"  {batch}/{animal}")

if args.dry_run:
    print("\nDRY RUN")
    for run_index, (batch, animal) in enumerate(pairs, start=1):
        paths = animal_paths(audit_output_root, batch, animal)
        action = "skip existing" if outputs_exist(paths) and not args.force else "run audit"
        print(f"[{run_index}/{len(pairs)}] {batch}/{animal}: {action}")
        print(f"  output: {paths['output_dir']}")
    raise SystemExit(0)


# %%
# =============================================================================
# Run loop
# =============================================================================
audit_output_root.mkdir(parents=True, exist_ok=True)
log_dir.mkdir(parents=True, exist_ok=True)
ledger_path = log_dir / "convergence_audit_run_status.csv"

rows = []
for run_index, (batch, animal) in enumerate(pairs, start=1):
    original_paths = animal_paths(ORIGINAL_OUTPUT_ROOT, batch, animal)
    audit_paths = animal_paths(audit_output_root, batch, animal)
    rows.append(
        {
            "run_id": run_id,
            "run_index": run_index,
            "n_runs": len(pairs),
            "batch_name": batch,
            "animal": animal,
            "status": "pending",
            "original_bundle_path": rel(original_paths["bundle_pkl"]),
            "audit_bundle_path": rel(audit_paths["bundle_pkl"]),
            "audit_output_dir": rel(audit_paths["output_dir"]),
        }
    )
write_ledger(rows, ledger_path)

failures = []
for row in rows:
    batch = row["batch_name"]
    animal = int(row["animal"])
    run_index = int(row["run_index"])
    audit_paths = animal_paths(audit_output_root, batch, animal)
    log_path = log_dir / f"{run_id}_{run_index:02d}_{batch}_{animal}_no_early_stop.log"
    row["log_path"] = rel(log_path)

    if outputs_exist(audit_paths) and not args.force:
        ok, error, stop_reason = verify_outputs(audit_paths)
        if ok:
            row["status"] = "skipped_existing"
            row["elapsed_seconds"] = "0.000"
            row["return_code"] = 0
            row["stop_reason"] = stop_reason
            write_ledger(rows, ledger_path)
            print(f"\n[{run_index}/{len(rows)}] Skipping {batch}/{animal}; verified existing audit outputs.")
            continue
        row["status"] = "existing_incomplete"
        row["error"] = error
        failures.append((batch, animal, row["status"]))
        write_ledger(rows, ledger_path)
        if args.stop_on_failure:
            break

    print("\n" + "=" * 80)
    print(f"[{run_index}/{len(rows)}] Running no-early-stop audit for {batch}/{animal}")
    print("=" * 80)

    env = os.environ.copy()
    env.update(
        {
            "NUMPYRO_BIG_GAMMA_OMEGA_DELAY_BATCH": str(batch),
            "NUMPYRO_BIG_GAMMA_OMEGA_DELAY_ANIMAL": str(animal),
            "NUMPYRO_BIG_GAMMA_OMEGA_DELAY_OUTPUT_ROOT": str(audit_output_root),
            "NUMPYRO_BIG_GAMMA_OMEGA_DELAY_GUIDE": str(args.guide),
            "NUMPYRO_BIG_GAMMA_OMEGA_DELAY_STEPS": str(args.steps),
            "NUMPYRO_BIG_GAMMA_OMEGA_DELAY_CHECK_EVERY": str(args.check_every),
            "NUMPYRO_BIG_GAMMA_OMEGA_DELAY_POSTERIOR_N_SAMPLES": str(args.posterior_samples),
            "NUMPYRO_BIG_GAMMA_OMEGA_DELAY_SEED": str(args.seed),
            "NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LR": str(args.lr),
            "NUMPYRO_BIG_GAMMA_OMEGA_DELAY_CLIP_NORM": str(args.clip_norm),
            "NUMPYRO_BIG_GAMMA_OMEGA_DELAY_OPTIMIZER": "clipped_adam",
            "NUMPYRO_BIG_GAMMA_OMEGA_DELAY_EARLY_STOP": "0",
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

    ok, error, stop_reason = verify_outputs(audit_paths)
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
print("Convergence audit summary")
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
    raise SystemExit(1)

# %%
