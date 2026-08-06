#!/usr/bin/env python3
# %%
"""Sequential runner for the remaining LED7 NPL+alpha uniform-delay fits.

Every animal is fit with the same 67-parameter RT+choice model used for
LED7/92.  Successful fits are followed by 10/20 ms RTD diagnostics and the
global-parameter/uniform-delay figures.  A CSV ledger is updated after every
state change so a partial batch can be resumed without refitting completed
animals.
"""

# %%
# =============================================================================
# Editable parameters and paths
# =============================================================================
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ANIMAL_FIT_DIR = SCRIPT_DIR.parent
REPO_DIR = ANIMAL_FIT_DIR.parent

BATCH_NAME = "LED7"
ALL_LED7_ANIMALS = (92, 93, 98, 99, 100, 103)
DEFAULT_ANIMALS = (93, 98, 99, 100, 103)
ABLS = (20, 40, 60)

FIT_SCRIPT = SCRIPT_DIR / "numpyro_svi_npl_alpha_uniform_delay_single_animal.py"
RTD_SCRIPT = SCRIPT_DIR / "plot_npl_alpha_uniform_delay_single_animal_rtds.py"
PARAMETER_DIAGNOSTIC_SCRIPT = (
    SCRIPT_DIR
    / "plot_npl_alpha_uniform_delay_single_animal_parameter_diagnostics.py"
)
REFERENCE_ROOT = (
    ANIMAL_FIT_DIR
    / "numpyro_svi_npl_alpha_condition_delay_patience12_restore_best_outputs"
)
ABORT_ROOT = REPO_DIR / "aborts_ipl_npl_time_fit_results"
DEFAULT_OUTPUT_ROOT = (
    SCRIPT_DIR
    / (
        "numpyro_svi_npl_alpha_uniform_delay_rt_choice_"
        "patience12_min50k_restore_best_outputs"
    )
)


# %%
# =============================================================================
# Small reused operations
# =============================================================================
def relative_path(path):
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_DIR.resolve()))
    except ValueError:
        return str(path)


def fit_output_paths(output_root, animal):
    output_dir = output_root / f"{BATCH_NAME}_{animal}"
    return {
        "output_dir": output_dir,
        "posterior_npz": output_dir / "main_fullrank_posterior_samples.npz",
        "bundle_pkl": (
            output_dir / "main_fullrank_variational_posterior_bundle.pkl"
        ),
        "global_summary_csv": output_dir / "main_fullrank_global_summary.csv",
        "condition_summary_csv": output_dir / "condition_delay_summary.csv",
        "condition_table_csv": output_dir / "condition_table.csv",
        "loss_csv": output_dir / "main_fullrank_loss.csv",
        "convergence_csv": output_dir / "main_fullrank_convergence_checks.csv",
        "metadata_json": output_dir / "main_fullrank_run_metadata.json",
        "loss_png": output_dir / "main_fullrank_loss.png",
    }


def nonempty_files(paths):
    return all(path.exists() and path.stat().st_size > 0 for path in paths)


def run_and_log(command, environment, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"Command: {' '.join(str(part) for part in command)}\n")
        log_file.write(f"CWD: {REPO_DIR}\n")
        log_file.write(
            f"Started: {datetime.now().isoformat(timespec='seconds')}\n\n"
        )
        log_file.flush()
        process = subprocess.Popen(
            [str(part) for part in command],
            cwd=str(REPO_DIR),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
        elapsed_seconds = time.monotonic() - started
        log_file.write(
            f"\nFinished: {datetime.now().isoformat(timespec='seconds')}\n"
        )
        log_file.write(f"Return code: {return_code}\n")
        log_file.write(f"Elapsed seconds: {elapsed_seconds:.3f}\n")
    return return_code, elapsed_seconds


def write_ledger(rows, ledger_path):
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "run_index",
        "n_runs",
        "batch_name",
        "animal",
        "status",
        "n_successful_rt_0_to_1",
        "n_conditions",
        "fit_elapsed_seconds",
        "diagnostic_elapsed_seconds",
        "fit_return_code",
        "diagnostic_return_code",
        "output_dir",
        "fit_log",
        "diagnostic_logs",
        "error",
    ]
    with ledger_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


# %%
# =============================================================================
# Command-line settings and preflight
# =============================================================================
parser = argparse.ArgumentParser(
    description="Run the remaining LED7 67-parameter uniform-delay SVI fits."
)
parser.add_argument("--dry-run", action="store_true")
parser.add_argument(
    "--force",
    action="store_true",
    help="Refit animals even when all fit artifacts already exist.",
)
parser.add_argument(
    "--force-diagnostics",
    action="store_true",
    help="Regenerate diagnostic artifacts without forcing the SVI fit.",
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
parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
parser.add_argument("--main-steps", type=int, default=150000)
parser.add_argument("--check-every", type=int, default=1000)
parser.add_argument("--min-steps", type=int, default=50000)
parser.add_argument("--patience-windows", type=int, default=12)
parser.add_argument("--min-improvement-rel", type=float, default=0.001)
parser.add_argument("--posterior-samples", type=int, default=10000)
parser.add_argument("--learning-rate", type=float, default=0.0002)
parser.add_argument("--clip-norm", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--rtd-bin-ms",
    nargs="*",
    type=float,
    default=[10.0, 20.0],
    help="Empirical RTD bin widths generated after each fit.",
)
parser.add_argument("--stop-on-failure", action="store_true")
args = parser.parse_args()

python_path = Path(args.python).expanduser()
if not python_path.is_absolute():
    python_path = (REPO_DIR / python_path).absolute()
output_root = Path(args.output_root).expanduser()
if not output_root.is_absolute():
    output_root = (REPO_DIR / output_root).resolve()
log_dir = output_root / "_batch_logs"
ledger_path = log_dir / "batch_run_status.csv"
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

for required_path in (
    python_path,
    FIT_SCRIPT,
    RTD_SCRIPT,
    PARAMETER_DIAGNOSTIC_SCRIPT,
):
    if not required_path.exists():
        raise FileNotFoundError(required_path)

selected_animals = list(DEFAULT_ANIMALS)
if args.only:
    unknown_animals = sorted(set(args.only) - set(ALL_LED7_ANIMALS))
    if unknown_animals:
        raise ValueError(f"Unknown LED7 animal IDs: {unknown_animals}")
    selected_animals = [
        animal for animal in ALL_LED7_ANIMALS if animal in set(args.only)
    ]
if not selected_animals:
    raise RuntimeError("No LED7 animals selected.")
if not args.rtd_bin_ms or any(value <= 0.0 for value in args.rtd_bin_ms):
    raise ValueError("--rtd-bin-ms requires one or more positive widths.")

batch_csv = (
    REPO_DIR
    / "raw_data"
    / "batch_csvs"
    / f"batch_{BATCH_NAME}_valid_and_aborts.csv"
)
if not batch_csv.exists():
    raise FileNotFoundError(batch_csv)
raw_df = pd.read_csv(batch_csv)

rows = []
preflight_errors = []
for run_index, animal in enumerate(selected_animals, start=1):
    reference_dir = REFERENCE_ROOT / f"{BATCH_NAME}_{animal}"
    required_inputs = (
        reference_dir / "main_fullrank_posterior_samples.npz",
        reference_dir / "condition_table.csv",
        ABORT_ROOT / f"results_{BATCH_NAME}_animal_{animal}.pkl",
    )
    for required_path in required_inputs:
        if not required_path.exists():
            preflight_errors.append(
                f"{BATCH_NAME}/{animal} missing {required_path}"
            )

    animal_df = raw_df[
        raw_df["animal"].eq(animal)
        & raw_df["success"].isin([1, -1])
        & raw_df["RTwrtStim"].ge(0.0)
        & raw_df["RTwrtStim"].lt(1.0)
        & raw_df["ABL"].isin(ABLS)
    ].dropna(
        subset=["RTwrtStim", "TotalFixTime", "intended_fix", "ABL", "ILD"]
    )
    n_conditions = len(animal_df[["ABL", "ILD"]].drop_duplicates())
    if n_conditions != 30:
        preflight_errors.append(
            f"{BATCH_NAME}/{animal} has {n_conditions} conditions; expected 30"
        )

    paths = fit_output_paths(output_root, animal)
    rows.append(
        {
            "run_id": run_id,
            "run_index": run_index,
            "n_runs": len(selected_animals),
            "batch_name": BATCH_NAME,
            "animal": animal,
            "status": "pending",
            "n_successful_rt_0_to_1": len(animal_df),
            "n_conditions": n_conditions,
            "output_dir": relative_path(paths["output_dir"]),
        }
    )

if preflight_errors:
    raise RuntimeError("Preflight failed:\n" + "\n".join(preflight_errors))

print(f"Run id: {run_id}")
print(f"Python: {python_path}")
print(f"Fit script: {FIT_SCRIPT}")
print(f"Point-delay initialization root: {REFERENCE_ROOT}")
print(f"Output root: {output_root}")
print(f"Animals: {selected_animals}")
print(
    "Model: NPL+alpha RT+choice, seven global parameters + 30 Uniform "
    "delay centers + 30 Uniform delay widths = 67 parameters"
)
print(
    "Stopping: "
    f"max={args.main_steps}, minimum={args.min_steps}, "
    f"window={args.check_every}, patience={args.patience_windows}, "
    f"minimum relative improvement={args.min_improvement_rel:g}"
)
print(f"Post-fit RTD data bins: {args.rtd_bin_ms} ms")
print("\nPreflight trial counts:")
print(
    pd.DataFrame(rows)[
        ["animal", "n_successful_rt_0_to_1", "n_conditions"]
    ].to_string(index=False)
)


# %%
# =============================================================================
# Expected post-fit diagnostics
# =============================================================================
def diagnostic_paths(output_dir, animal):
    diagnostic_dir = output_dir / "diagnostics"
    paths = []
    for bin_ms in args.rtd_bin_ms:
        bin_text = f"{bin_ms:g}".replace(".", "p")
        filename_part = (
            "" if abs(bin_ms - 5.0) < 1e-12 else f"_data_bin_{bin_text}ms"
        )
        stem = (
            f"{BATCH_NAME.lower()}_{animal}_npl_alpha_uniform_delay_"
            f"fit_aligned_rtds_by_abl_abs_ild{filename_part}"
        )
        paths.extend(
            [
                diagnostic_dir / f"{stem}_0_600ms_xlim.png",
                diagnostic_dir / f"{stem}.pkl",
            ]
        )
    paths.extend(
        [
            diagnostic_dir
            / f"{BATCH_NAME.lower()}_{animal}_npl_alpha_uniform_delay_global_corner.png",
            diagnostic_dir
            / (
                f"{BATCH_NAME.lower()}_{animal}_npl_alpha_uniform_delay_"
                "condition_distributions_by_ild.png"
            ),
            diagnostic_dir
            / (
                f"{BATCH_NAME.lower()}_{animal}_npl_alpha_uniform_delay_"
                "center_and_support_vs_ild_by_abl.png"
            ),
        ]
    )
    return paths


if args.dry_run:
    print("\nDRY RUN: no fits or diagnostics started.")
    for row in rows:
        animal = int(row["animal"])
        paths = fit_output_paths(output_root, animal)
        fit_files = [
            path for key, path in paths.items() if key != "output_dir"
        ]
        fit_action = (
            "skip existing fit"
            if nonempty_files(fit_files) and not args.force
            else "run fit"
        )
        diagnostics_action = (
            "skip existing diagnostics"
            if nonempty_files(diagnostic_paths(paths["output_dir"], animal))
            and not args.force_diagnostics
            else "run diagnostics"
        )
        print(
            f"  {BATCH_NAME}/{animal}: {fit_action}; {diagnostics_action}"
        )
    raise SystemExit(0)


# %%
# =============================================================================
# Sequential fit and diagnostic loop
# =============================================================================
log_dir.mkdir(parents=True, exist_ok=True)
write_ledger(rows, ledger_path)
failures = []

for row in rows:
    animal = int(row["animal"])
    run_index = int(row["run_index"])
    paths = fit_output_paths(output_root, animal)
    output_dir = paths["output_dir"]
    fit_files = [path for key, path in paths.items() if key != "output_dir"]
    fit_log = log_dir / f"{run_id}_{run_index:02d}_LED7_{animal}_fit.log"
    row["fit_log"] = relative_path(fit_log)

    base_environment = os.environ.copy()
    base_environment.update(
        {
            "NUMPYRO_SVI_BATCH": BATCH_NAME,
            "NUMPYRO_SVI_ANIMAL": str(animal),
            "NUMPYRO_SVI_OUTPUT_ROOT": str(output_root),
            "UNIFORM_DELAY_REFERENCE_FIT_ROOT": str(REFERENCE_ROOT),
            "NUMPYRO_SVI_GUIDE": "fullrank",
            "RUN_MAIN_SVI": "1",
            "MAIN_STEPS": str(args.main_steps),
            "SVI_CHECK_EVERY": str(args.check_every),
            "SVI_MIN_STEPS": str(args.min_steps),
            "SVI_NO_IMPROVE_PATIENCE_WINDOWS": str(args.patience_windows),
            "SVI_MIN_IMPROVEMENT_REL": str(args.min_improvement_rel),
            "SVI_EARLY_STOP": "1",
            "POSTERIOR_N_SAMPLES": str(args.posterior_samples),
            "NUMPYRO_SVI_LR": str(args.learning_rate),
            "NUMPYRO_SVI_CLIP_NORM": str(args.clip_norm),
            "NUMPYRO_SVI_SEED": str(args.seed),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )

    if nonempty_files(fit_files) and not args.force:
        print(
            f"\n[{run_index}/{len(rows)}] {BATCH_NAME}/{animal}: "
            "fit artifacts already complete; skipping SVI."
        )
        row["fit_return_code"] = 0
        row["fit_elapsed_seconds"] = "0.000"
    else:
        print("\n" + "=" * 80)
        print(f"[{run_index}/{len(rows)}] Fitting {BATCH_NAME}/{animal}")
        print("=" * 80)
        row["status"] = "fitting"
        write_ledger(rows, ledger_path)
        fit_return_code, fit_elapsed = run_and_log(
            [python_path, "-u", FIT_SCRIPT],
            base_environment,
            fit_log,
        )
        row["fit_return_code"] = fit_return_code
        row["fit_elapsed_seconds"] = f"{fit_elapsed:.3f}"
        if fit_return_code != 0 or not nonempty_files(fit_files):
            row["status"] = "fit_failed"
            row["error"] = (
                f"fit returned {fit_return_code}; "
                f"fit_outputs_complete={nonempty_files(fit_files)}"
            )
            failures.append((animal, row["status"]))
            write_ledger(rows, ledger_path)
            if args.stop_on_failure:
                break
            continue
        print(
            f"[{run_index}/{len(rows)}] Fit completed in "
            f"{fit_elapsed / 60.0:.2f} min."
        )

    expected_diagnostics = diagnostic_paths(output_dir, animal)
    if nonempty_files(expected_diagnostics) and not args.force_diagnostics:
        row["status"] = "completed"
        row["diagnostic_return_code"] = 0
        row["diagnostic_elapsed_seconds"] = "0.000"
        write_ledger(rows, ledger_path)
        print(f"[{run_index}/{len(rows)}] Diagnostics already complete.")
        continue

    row["status"] = "diagnosing"
    write_ledger(rows, ledger_path)
    diagnostic_started = time.monotonic()
    diagnostic_logs = []
    diagnostic_return_code = 0

    for bin_ms in args.rtd_bin_ms:
        bin_text = f"{bin_ms:g}".replace(".", "p")
        diagnostic_log = (
            log_dir
            / f"{run_id}_{run_index:02d}_LED7_{animal}_rtd_{bin_text}ms.log"
        )
        diagnostic_logs.append(relative_path(diagnostic_log))
        diagnostic_environment = base_environment.copy()
        diagnostic_environment["NUMPYRO_SVI_DIAG_DATA_BIN_S"] = str(
            bin_ms / 1000.0
        )
        return_code, _ = run_and_log(
            [python_path, "-u", RTD_SCRIPT],
            diagnostic_environment,
            diagnostic_log,
        )
        if return_code != 0:
            diagnostic_return_code = return_code
            break

    if diagnostic_return_code == 0:
        diagnostic_log = (
            log_dir
            / f"{run_id}_{run_index:02d}_LED7_{animal}_parameters.log"
        )
        diagnostic_logs.append(relative_path(diagnostic_log))
        diagnostic_return_code, _ = run_and_log(
            [python_path, "-u", PARAMETER_DIAGNOSTIC_SCRIPT],
            base_environment,
            diagnostic_log,
        )

    diagnostic_elapsed = time.monotonic() - diagnostic_started
    row["diagnostic_logs"] = ";".join(diagnostic_logs)
    row["diagnostic_return_code"] = diagnostic_return_code
    row["diagnostic_elapsed_seconds"] = f"{diagnostic_elapsed:.3f}"
    if diagnostic_return_code != 0 or not nonempty_files(expected_diagnostics):
        row["status"] = "diagnostic_failed"
        row["error"] = (
            f"diagnostics returned {diagnostic_return_code}; "
            f"diagnostic_outputs_complete={nonempty_files(expected_diagnostics)}"
        )
        failures.append((animal, row["status"]))
        write_ledger(rows, ledger_path)
        if args.stop_on_failure:
            break
        continue

    row["status"] = "completed"
    write_ledger(rows, ledger_path)
    print(
        f"[{run_index}/{len(rows)}] {BATCH_NAME}/{animal} complete; "
        f"diagnostics took {diagnostic_elapsed / 60.0:.2f} min."
    )


# %%
# =============================================================================
# Final ledger summary
# =============================================================================
print("\n" + "=" * 80)
print("LED7 NPL+alpha uniform-delay batch summary")
print("=" * 80)
status_counts = pd.Series([row["status"] for row in rows]).value_counts()
for status, count in status_counts.items():
    print(f"  {status}: {count}")
print(f"Ledger: {ledger_path}")
print(f"Logs: {log_dir}")

if failures:
    print("Failures:")
    for animal, status in failures:
        print(f"  {BATCH_NAME}/{animal}: {status}")
    raise SystemExit(2)

print("Done.")

# %%
