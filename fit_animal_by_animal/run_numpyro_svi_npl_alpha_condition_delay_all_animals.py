#!/usr/bin/env python3
# %%
"""
Run the single-animal NumPyro SVI prototype and diagnostics for all animals.

This script intentionally does not modify the single-animal SVI script. It runs
that script with environment variables, asks it to save its CSV tables, runs the
RTD/psychometric diagnostic, and then bundles the recoverable posterior samples
and metadata into one pickle per animal.
"""

# %%
from __future__ import annotations

import argparse
import csv
import os
import pickle
import re
import subprocess
import sys
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

FIT_SCRIPT = SCRIPT_DIR / "numpyro_svi_npl_alpha_condition_delay_single_animal.py"
DIAGNOSTIC_SCRIPT = SCRIPT_DIR / "diagnose_numpyro_svi_npl_alpha_condition_delay_single_animal.py"
FIXED_DELAY_DIR = SCRIPT_DIR / "NPL_alpha_condition_t_E_aff_fixed_delay_fit_results_all_30"
OUTPUT_ROOT = SCRIPT_DIR / "numpyro_svi_npl_alpha_condition_delay_single_animal_outputs"
LOG_DIR = OUTPUT_ROOT / "_batch_logs"

FIXED_DELAY_RESULT_KEY = "vbmc_norm_alpha_condition_t_E_aff_fixed_delay_tied_results"
ABORT_RESULT_KEY = "vbmc_aborts_results"

DEFAULT_BATCH_ORDER = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]
EXPECTED_N_ANIMALS = 30

GLOBAL_SAMPLE_KEYS = {
    "rate_lambda": "rate_lambda_samples",
    "T_0": "T_0_samples",
    "theta_E": "theta_E_samples",
    "w": "w_samples",
    "del_go": "del_go_samples",
    "rate_norm_l": "rate_norm_l_samples",
    "alpha": "alpha_samples",
}


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
    pattern = re.compile(
        r"^results_(?P<batch>.+)_animal_(?P<animal>\d+)_NORM_ALPHA_CONDITION_T_E_AFF_FIXED_DELAY_FROM_ABORTS\.pkl$"
    )
    pairs = []
    for result_path in FIXED_DELAY_DIR.glob("results_*_NORM_ALPHA_CONDITION_T_E_AFF_FIXED_DELAY_FROM_ABORTS.pkl"):
        match = pattern.match(result_path.name)
        if match is None:
            continue
        pairs.append((match.group("batch"), int(match.group("animal"))))

    batch_order = {batch: idx for idx, batch in enumerate(DEFAULT_BATCH_ORDER)}
    return sorted(pairs, key=lambda pair: (batch_order.get(pair[0], 999), pair[0], pair[1]))


def animal_paths(batch: str, animal: int, label: str):
    output_dir = OUTPUT_ROOT / f"{batch}_{animal}"
    diagnostics_dir = output_dir / "diagnostics"
    return {
        "output_dir": output_dir,
        "diagnostics_dir": diagnostics_dir,
        "posterior_npz": output_dir / f"{label}_posterior_samples.npz",
        "posterior_summary_csv": output_dir / f"{label}_posterior_summary.csv",
        "finite_report_csv": output_dir / f"{label}_posterior_finite_report.csv",
        "loss_csv": output_dir / f"{label}_loss.csv",
        "convergence_csv": output_dir / f"{label}_convergence_checks.csv",
        "condition_table_csv": output_dir / "condition_table.csv",
        "loss_png": output_dir / f"{label}_loss.png",
        "global_corner_png": output_dir / f"{label}_global_corner.png",
        "selected_delay_corner_png": output_dir / f"{label}_global_selected_delay_corner.png",
        "delay_intervals_png": output_dir / f"{label}_condition_delay_intervals.png",
        "rtd_png": diagnostics_dir / f"{label}_rtd_by_abl_abs_ild.png",
        "rtd_zoom_png": diagnostics_dir / f"{label}_rtd_by_abl_abs_ild_zoom.png",
        "psychometric_png": diagnostics_dir / f"{label}_psychometric_by_abl.png",
        "bundle_pkl": output_dir / f"{label}_variational_posterior_bundle.pkl",
    }


def fixed_delay_result_path(batch: str, animal: int) -> Path:
    return (
        FIXED_DELAY_DIR
        / f"results_{batch}_animal_{animal}_NORM_ALPHA_CONDITION_T_E_AFF_FIXED_DELAY_FROM_ABORTS.pkl"
    )


def abort_result_path(batch: str, animal: int) -> Path:
    return REPO_DIR / "aborts_ipl_npl_time_fit_results" / f"results_{batch}_animal_{animal}.pkl"


def batch_csv_path(batch: str) -> Path:
    return REPO_DIR / "raw_data" / "batch_csvs" / f"batch_{batch}_valid_and_aborts.csv"


def required_outputs(paths):
    return [
        paths["posterior_npz"],
        paths["posterior_summary_csv"],
        paths["finite_report_csv"],
        paths["loss_csv"],
        paths["convergence_csv"],
        paths["condition_table_csv"],
        paths["loss_png"],
        paths["global_corner_png"],
        paths["selected_delay_corner_png"],
        paths["delay_intervals_png"],
        paths["rtd_png"],
        paths["rtd_zoom_png"],
        paths["psychometric_png"],
        paths["bundle_pkl"],
    ]


def required_source_outputs(paths):
    return [
        paths["posterior_npz"],
        paths["loss_png"],
        paths["global_corner_png"],
        paths["selected_delay_corner_png"],
        paths["delay_intervals_png"],
        paths["rtd_png"],
        paths["rtd_zoom_png"],
        paths["psychometric_png"],
    ]


def outputs_complete(paths):
    return all(path.exists() and path.stat().st_size > 0 for path in required_outputs(paths))


def source_outputs_complete(paths):
    return all(path.exists() and path.stat().st_size > 0 for path in required_source_outputs(paths))


def read_csv_if_exists(path: Path):
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path)
    return None


def load_npz_if_exists(path: Path):
    if not path.exists():
        return None
    with np.load(path) as saved:
        return {key: np.asarray(saved[key]) for key in saved.files}


def posterior_sample_summary(posterior_samples):
    rows = []
    if not posterior_samples:
        return pd.DataFrame(rows)
    for key, values in posterior_samples.items():
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 1:
            flat = arr[np.isfinite(arr)]
            rows.append(
                {
                    "parameter": key,
                    "shape": str(arr.shape),
                    "mean": float(np.mean(flat)) if flat.size else np.nan,
                    "sd": float(np.std(flat)) if flat.size else np.nan,
                    "q025": float(np.quantile(flat, 0.025)) if flat.size else np.nan,
                    "q500": float(np.quantile(flat, 0.5)) if flat.size else np.nan,
                    "q975": float(np.quantile(flat, 0.975)) if flat.size else np.nan,
                    "n_finite": int(flat.size),
                    "n_total": int(arr.size),
                }
            )
    return pd.DataFrame(rows)


def upstream_fixed_delay_means(batch: str, animal: int):
    path = fixed_delay_result_path(batch, animal)
    if not path.exists():
        return None
    with path.open("rb") as handle:
        saved = pickle.load(handle)
    fit = saved.get(FIXED_DELAY_RESULT_KEY, {})
    means = {}
    for param_name, sample_key in GLOBAL_SAMPLE_KEYS.items():
        if sample_key in fit:
            means[param_name] = float(np.mean(np.asarray(fit[sample_key], dtype=float)))
    return {
        "path": rel(path),
        "message": str(fit.get("message", "")),
        "means": means,
    }


def abort_means(batch: str, animal: int):
    path = abort_result_path(batch, animal)
    if not path.exists():
        return None
    with path.open("rb") as handle:
        saved = pickle.load(handle)
    abort = saved.get(ABORT_RESULT_KEY, {})
    means = {}
    sample_key_map = {
        "V_A": "V_A_samples",
        "theta_A": "theta_A_samples",
        "t_A_aff": "t_A_aff_samp",
    }
    for param_name, sample_key in sample_key_map.items():
        if sample_key in abort:
            means[param_name] = float(np.mean(np.asarray(abort[sample_key], dtype=float)))
    return {
        "path": rel(path),
        "means": means,
    }


def make_bundle(batch: str, animal: int, label: str, guide_kind: str, commands, return_codes, elapsed_seconds):
    paths = animal_paths(batch, animal, label)
    posterior_samples = load_npz_if_exists(paths["posterior_npz"])
    posterior_summary = read_csv_if_exists(paths["posterior_summary_csv"])
    if posterior_summary is None:
        posterior_summary = posterior_sample_summary(posterior_samples)

    bundle = {
        "schema_version": 1,
        "note": (
            "This bundle is built without modifying the single-animal SVI script. "
            "It contains sampled variational posterior arrays and saved summaries, "
            "not the raw internal NumPyro guide parameter state."
        ),
        "batch_name": batch,
        "animal": int(animal),
        "label": label,
        "guide_kind": guide_kind,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "commands": commands,
        "return_codes": return_codes,
        "elapsed_seconds": elapsed_seconds,
        "paths": {key: rel(path) for key, path in paths.items()},
        "input_paths": {
            "batch_csv": rel(batch_csv_path(batch)),
            "abort_result_pkl": rel(abort_result_path(batch, animal)),
            "fixed_delay_result_pkl": rel(fixed_delay_result_path(batch, animal)),
        },
        "posterior_samples": posterior_samples,
        "posterior_summary": posterior_summary,
        "finite_report": read_csv_if_exists(paths["finite_report_csv"]),
        "condition_table": read_csv_if_exists(paths["condition_table_csv"]),
        "loss_trace": read_csv_if_exists(paths["loss_csv"]),
        "convergence_checks": read_csv_if_exists(paths["convergence_csv"]),
        "upstream_fixed_delay": upstream_fixed_delay_means(batch, animal),
        "abort_means": abort_means(batch, animal),
        "plots": {
            "loss": rel(paths["loss_png"]),
            "global_corner": rel(paths["global_corner_png"]),
            "selected_delay_corner": rel(paths["selected_delay_corner_png"]),
            "delay_intervals": rel(paths["delay_intervals_png"]),
            "rtd": rel(paths["rtd_png"]),
            "rtd_zoom": rel(paths["rtd_zoom_png"]),
            "psychometric": rel(paths["psychometric_png"]),
        },
    }
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    with paths["bundle_pkl"].open("wb") as handle:
        pickle.dump(bundle, handle)
    return paths["bundle_pkl"]


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
        "diagnostic_return_code",
        "bundle_path",
        "output_dir",
        "fit_log",
        "diagnostic_log",
        "error",
    ]
    with ledger_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def print_pair_table(pairs):
    current_batch = None
    for batch, animal in pairs:
        if batch != current_batch:
            current_batch = batch
            animals = [str(a) for b, a in pairs if b == batch]
            print(f"  {batch}: {', '.join(animals)}")


# %%
# =============================================================================
# CLI and run loop
# =============================================================================
parser = argparse.ArgumentParser(
    description="Run NumPyro SVI NPL+alpha condition-delay fits and diagnostics for all 30 animals."
)
parser.add_argument("--dry-run", action="store_true", help="Print planned runs without executing them.")
parser.add_argument("--force", action="store_true", help="Run even if all expected outputs already exist.")
parser.add_argument(
    "--bundle-existing-only",
    action="store_true",
    help="Do not run fits; create bundle pickles from existing posterior/plot artifacts.",
)
parser.add_argument("--only", nargs="*", help="Restrict to BATCH:ANIMAL entries, e.g. --only LED8:105 LED7:92.")
parser.add_argument("--python", default=str(REPO_DIR / ".venv" / "bin" / "python"), help="Python interpreter to use.")
parser.add_argument("--guide", default="fullrank", help="NumPyro guide kind. Default: fullrank.")
parser.add_argument("--main-steps", type=int, default=100000, help="Maximum main SVI steps.")
parser.add_argument("--check-every", type=int, default=10000, help="SVI convergence check interval.")
parser.add_argument("--output-root", default=str(OUTPUT_ROOT), help="Output root for all animal fit folders.")
parser.add_argument(
    "--stop-mode",
    default="stable_or_no_improve",
    choices=["legacy", "stable_or_no_improve", "patience_restore_best"],
    help="SVI stopping rule mode passed to the single-animal script.",
)
parser.add_argument("--rel-tol", type=float, default=0.01, help="Relative change tolerance for stable-window stopping.")
parser.add_argument(
    "--patience-windows",
    type=int,
    default=3,
    help="Stable-window patience for legacy/stable stopping.",
)
parser.add_argument(
    "--no-improve-patience-windows",
    type=int,
    default=2,
    help="No-best-window-improvement patience, measured in check windows.",
)
parser.add_argument(
    "--min-improvement-rel",
    type=float,
    default=0.005,
    help="Relative best-window improvement required to reset no-improve patience.",
)
parser.add_argument("--min-steps", type=int, default=0, help="Minimum SVI steps before early stopping can trigger.")
parser.add_argument("--seed", type=int, default=0, help="NumPyro/JAX random seed.")
parser.add_argument("--posterior-samples", type=int, default=10000, help="Number of posterior samples to save.")
parser.add_argument("--expected-n-animals", type=int, default=EXPECTED_N_ANIMALS, help="Expected discovered animal count.")
parser.add_argument("--stop-on-failure", action="store_true", help="Stop immediately if any fit or diagnostic fails.")
parser.add_argument(
    "--ledger-name",
    default="batch_run_status.csv",
    help="Ledger CSV filename under the output root's _batch_logs folder.",
)
args = parser.parse_args()

OUTPUT_ROOT = Path(args.output_root).expanduser()
if not OUTPUT_ROOT.is_absolute():
    OUTPUT_ROOT = (REPO_DIR / OUTPUT_ROOT).resolve()
LOG_DIR = OUTPUT_ROOT / "_batch_logs"
LEDGER_PATH = LOG_DIR / args.ledger_name

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
label = f"main_{args.guide}"

if not FIT_SCRIPT.exists():
    raise FileNotFoundError(FIT_SCRIPT)
if not DIAGNOSTIC_SCRIPT.exists():
    raise FileNotFoundError(DIAGNOSTIC_SCRIPT)
if not FIXED_DELAY_DIR.exists():
    raise FileNotFoundError(FIXED_DELAY_DIR)

selected_pairs = parse_only_items(args.only)
all_pairs = discover_animals()
if selected_pairs is None and len(all_pairs) != args.expected_n_animals:
    raise RuntimeError(f"Expected {args.expected_n_animals} animals, discovered {len(all_pairs)}")

pairs = [pair for pair in all_pairs if selected_pairs is None or pair in selected_pairs]
missing_requested = sorted(selected_pairs - set(pairs)) if selected_pairs is not None else []
if missing_requested:
    raise RuntimeError(f"Requested --only animals were not discovered from fixed-delay results: {missing_requested}")
if not pairs:
    raise RuntimeError("No animals selected.")

print(f"Run id: {RUN_ID}")
print(f"Repository: {REPO_DIR}")
print(f"Single-animal fit script: {FIT_SCRIPT}")
print(f"Diagnostic script: {DIAGNOSTIC_SCRIPT}")
print(f"Output root: {OUTPUT_ROOT}")
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
    for required_path in [batch_csv_path(batch), abort_result_path(batch, animal), fixed_delay_result_path(batch, animal)]:
        if not required_path.exists():
            preflight_errors.append(f"{batch}/{animal} missing {required_path}")
if preflight_errors:
    raise RuntimeError("Preflight failed:\n" + "\n".join(preflight_errors))

python_path = Path(args.python)
fit_cmd = [python_path, "-u", FIT_SCRIPT]
diag_cmd = [python_path, "-u", DIAGNOSTIC_SCRIPT]

if args.dry_run:
    print("\nDRY RUN")
    for run_index, (batch, animal) in enumerate(pairs, start=1):
        paths = animal_paths(batch, animal, label)
        complete = outputs_complete(paths)
        if args.bundle_existing_only:
            action = "bundle existing artifacts" if source_outputs_complete(paths) else "missing existing artifacts"
        else:
            action = "skip existing" if complete and not args.force else "run fit + diagnostics + bundle"
        print(f"[{run_index}/{len(pairs)}] {batch}/{animal}: {action}")
        print(f"  fit:  NUMPYRO_SVI_BATCH={batch} NUMPYRO_SVI_ANIMAL={animal} {' '.join(str(x) for x in fit_cmd)}")
        print(f"  diag: NUMPYRO_SVI_BATCH={batch} NUMPYRO_SVI_ANIMAL={animal} {' '.join(str(x) for x in diag_cmd)}")
    raise SystemExit(0)

LOG_DIR.mkdir(parents=True, exist_ok=True)
rows = []
for run_index, (batch, animal) in enumerate(pairs, start=1):
    paths = animal_paths(batch, animal, label)
    rows.append(
        {
            "run_id": RUN_ID,
            "run_index": run_index,
            "n_runs": len(pairs),
            "batch_name": batch,
            "animal": animal,
            "status": "pending",
            "bundle_path": rel(paths["bundle_pkl"]),
            "output_dir": rel(paths["output_dir"]),
        }
    )
write_ledger(rows, LEDGER_PATH)

failures = []
for row in rows:
    batch = row["batch_name"]
    animal = int(row["animal"])
    run_index = int(row["run_index"])
    paths = animal_paths(batch, animal, label)
    fit_log = LOG_DIR / f"{RUN_ID}_{run_index:02d}_{batch}_{animal}_fit.log"
    diag_log = LOG_DIR / f"{RUN_ID}_{run_index:02d}_{batch}_{animal}_diagnostic.log"
    row["fit_log"] = rel(fit_log)
    row["diagnostic_log"] = rel(diag_log)

    if args.bundle_existing_only:
        if not source_outputs_complete(paths):
            missing = [rel(path) for path in required_source_outputs(paths) if not path.exists() or path.stat().st_size == 0]
            row["status"] = "missing_existing_artifacts"
            row["error"] = "; ".join(missing)
            failures.append((batch, animal, row["status"]))
            write_ledger(rows, LEDGER_PATH)
            if args.stop_on_failure:
                break
            continue
        try:
            bundle_path = make_bundle(
                batch,
                animal,
                label,
                args.guide,
                commands={"fit": "not run; bundled existing artifacts", "diagnostic": "not run; bundled existing artifacts"},
                return_codes={"fit": None, "diagnostic": None},
                elapsed_seconds={"fit": 0.0, "diagnostic": 0.0, "total": 0.0},
            )
        except Exception as exc:
            row["status"] = "bundle_failed"
            row["error"] = repr(exc)
            failures.append((batch, animal, row["status"]))
            write_ledger(rows, LEDGER_PATH)
            if args.stop_on_failure:
                break
            continue
        row["status"] = "bundled_existing"
        row["bundle_path"] = rel(bundle_path)
        row["elapsed_seconds"] = 0.0
        write_ledger(rows, LEDGER_PATH)
        print(f"[{run_index}/{len(rows)}] Bundled existing artifacts for {batch}/{animal}: {bundle_path}")
        continue

    if outputs_complete(paths) and not args.force:
        row["status"] = "skipped_existing"
        row["elapsed_seconds"] = 0.0
        write_ledger(rows, LEDGER_PATH)
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
            "NUMPYRO_SVI_OUTPUT_ROOT": str(OUTPUT_ROOT),
            "NUMPYRO_SVI_GUIDE": str(args.guide),
            "NUMPYRO_SVI_SAVE_TABLE_CSVS": "1",
            "RUN_SMOKE_SVI": "0",
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
            "NUMPYRO_SVI_DIAG_LABEL": str(label),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )

    total_start = time.monotonic()
    fit_return_code, fit_elapsed = run_and_log(fit_cmd, env, fit_log, REPO_DIR)
    row["fit_return_code"] = fit_return_code
    if fit_return_code != 0:
        row["status"] = "fit_failed"
        row["elapsed_seconds"] = f"{time.monotonic() - total_start:.3f}"
        row["error"] = f"fit returned {fit_return_code}"
        failures.append((batch, animal, row["status"]))
        write_ledger(rows, LEDGER_PATH)
        if args.stop_on_failure:
            break
        continue

    diagnostic_return_code, diag_elapsed = run_and_log(diag_cmd, env, diag_log, REPO_DIR)
    row["diagnostic_return_code"] = diagnostic_return_code
    if diagnostic_return_code != 0:
        row["status"] = "diagnostic_failed"
        row["elapsed_seconds"] = f"{time.monotonic() - total_start:.3f}"
        row["error"] = f"diagnostic returned {diagnostic_return_code}"
        failures.append((batch, animal, row["status"]))
        write_ledger(rows, LEDGER_PATH)
        if args.stop_on_failure:
            break
        continue

    try:
        bundle_path = make_bundle(
            batch,
            animal,
            label,
            args.guide,
            commands={
                "fit": " ".join(str(part) for part in fit_cmd),
                "diagnostic": " ".join(str(part) for part in diag_cmd),
            },
            return_codes={
                "fit": fit_return_code,
                "diagnostic": diagnostic_return_code,
            },
            elapsed_seconds={
                "fit": fit_elapsed,
                "diagnostic": diag_elapsed,
                "total": time.monotonic() - total_start,
            },
        )
    except Exception as exc:
        row["status"] = "bundle_failed"
        row["elapsed_seconds"] = f"{time.monotonic() - total_start:.3f}"
        row["error"] = repr(exc)
        failures.append((batch, animal, row["status"]))
        write_ledger(rows, LEDGER_PATH)
        if args.stop_on_failure:
            break
        continue

    row["status"] = "completed"
    row["elapsed_seconds"] = f"{time.monotonic() - total_start:.3f}"
    row["bundle_path"] = rel(bundle_path)
    write_ledger(rows, LEDGER_PATH)
    print(f"[{run_index}/{len(rows)}] Completed {batch}/{animal}; bundle: {bundle_path}")

print("\n" + "=" * 80)
print("Batch run summary")
print("=" * 80)
status_counts = {}
for row in rows:
    status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
for status, count in sorted(status_counts.items()):
    print(f"  {status}: {count}")
print(f"Ledger: {LEDGER_PATH}")
print(f"Logs: {LOG_DIR}")

if failures:
    print("\nFailures:")
    for batch, animal, status in failures:
        print(f"  {batch}/{animal}: {status}")
    raise SystemExit(2)

print("Done.")

# %%
