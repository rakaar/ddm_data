# %%
"""Run the unweighted proactive LED step-jump SVI fit for all six LED7 animals."""

# %%
from pathlib import Path
from datetime import datetime
from time import perf_counter
import argparse
import json
import os
import subprocess
import sys

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
FIT_SCRIPT = SCRIPT_DIR / "numpyro_svi_proactive_led_step_jump_single_animal.py"
VALIDATION_SUMMARY = (
    SCRIPT_DIR
    / "proactive_led_step_jump_svi_validation"
    / "proactive_led_step_jump_validation_summary.json"
)
DEFAULT_OUTPUT_ROOT = (
    SCRIPT_DIR
    / "numpyro_svi_proactive_led_step_jump_bilateral_unweighted_patience12_min50k_restore_best_outputs"
)
ANIMALS = [92, 93, 98, 99, 100, 103]


# %%
# =============================================================================
# Command-line controls
# =============================================================================
parser = argparse.ArgumentParser(
    description="Run bilateral LED-ON plus LED-OFF proactive step-jump SVI fits for LED7."
)
parser.add_argument("--dry-run", action="store_true", help="Print the planned runs without fitting.")
parser.add_argument("--force", action="store_true", help="Refit animals with completed matching outputs.")
parser.add_argument("--only", nargs="*", type=int, help="Restrict the run to selected LED7 animal IDs.")
parser.add_argument("--stop-on-failure", action="store_true", help="Stop after the first failed animal.")
parser.add_argument("--python", default=str(REPO_DIR / ".venv" / "bin" / "python"))
parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
parser.add_argument("--quadrature-nodes", type=int, default=64)
parser.add_argument("--main-steps", type=int, default=150000)
parser.add_argument("--min-steps", type=int, default=50000)
parser.add_argument("--check-every", type=int, default=1000)
parser.add_argument("--patience-windows", type=int, default=12)
parser.add_argument("--min-improvement-rel", type=float, default=0.001)
parser.add_argument("--learning-rate", type=float, default=0.0002)
parser.add_argument("--posterior-samples", type=int, default=10000)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

OUTPUT_ROOT = Path(args.output_root).expanduser().resolve()
LOG_DIR = OUTPUT_ROOT / "_batch_logs"
LEDGER_PATH = LOG_DIR / "batch_run_status.csv"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

selected_animals = ANIMALS if args.only is None else args.only
unknown_animals = sorted(set(selected_animals) - set(ANIMALS))
if unknown_animals:
    raise ValueError(f"Unknown LED7 animals requested: {unknown_animals}; expected {ANIMALS}.")

if not FIT_SCRIPT.exists():
    raise FileNotFoundError(FIT_SCRIPT)
if not Path(args.python).exists():
    raise FileNotFoundError(args.python)
if not VALIDATION_SUMMARY.exists():
    raise FileNotFoundError(
        f"Run validate_numpyro_proactive_led_step_jump_likelihood.py first: {VALIDATION_SUMMARY}"
    )
validation = json.loads(VALIDATION_SUMMARY.read_text())
if validation.get("status") != "passed":
    raise RuntimeError(f"Likelihood validation did not pass: {VALIDATION_SUMMARY}")
if int(validation.get("recommended_quadrature_nodes")) != args.quadrature_nodes:
    raise RuntimeError(
        "Requested quadrature nodes do not match the validated recommendation: "
        f"requested={args.quadrature_nodes}, recommended={validation.get('recommended_quadrature_nodes')}"
    )


# %%
# =============================================================================
# Resumable ledger helpers
# =============================================================================
ledger_columns = [
    "run_id",
    "animal",
    "status",
    "started_at",
    "ended_at",
    "elapsed_minutes",
    "return_code",
    "completed_steps",
    "restored_best_step",
    "fit_elapsed_minutes",
    "output_dir",
    "animal_log",
    "message",
]


def read_ledger():
    if LEDGER_PATH.exists():
        return pd.read_csv(LEDGER_PATH)
    return pd.DataFrame(columns=ledger_columns)


def update_ledger(row):
    ledger_df = read_ledger()
    animal = int(row["animal"])
    if len(ledger_df):
        ledger_df = ledger_df[ledger_df["animal"].astype(int) != animal].copy()
    ledger_df = pd.concat([ledger_df, pd.DataFrame([row])], ignore_index=True)
    ledger_df = ledger_df.reindex(columns=ledger_columns).sort_values("animal")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    temporary_path = LEDGER_PATH.with_suffix(".tmp")
    ledger_df.to_csv(temporary_path, index=False)
    temporary_path.replace(LEDGER_PATH)


def matching_complete_summary(animal):
    summary_path = OUTPUT_ROOT / f"LED7_{animal}" / "run_summary.json"
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    config = summary.get("config", {})
    matches = (
        summary.get("status") == "complete"
        and int(config.get("quadrature_nodes", -1)) == args.quadrature_nodes
        and int(config.get("main_steps_max", -1)) == args.main_steps
        and int(config.get("min_steps", -1)) == args.min_steps
        and int(config.get("check_every", -1)) == args.check_every
        and int(config.get("patience_windows", -1)) == args.patience_windows
        and float(config.get("min_improvement_rel", -1.0)) == args.min_improvement_rel
        and float(config.get("learning_rate", -1.0)) == args.learning_rate
        and int(config.get("posterior_n_samples", -1)) == args.posterior_samples
        and int(config.get("rng_seed", -1)) == args.seed
        and float(config.get("led_on_weight", -1.0)) == 1.0
    )
    return summary if matches else None


# %%
# =============================================================================
# Print planned run
# =============================================================================
print("Proactive LED step-jump SVI batch")
print(f"  run ID: {RUN_ID}")
print(f"  animals: {selected_animals}")
print(f"  output root: {OUTPUT_ROOT}")
print(f"  validation: {VALIDATION_SUMMARY}")
print(f"  quadrature nodes: {args.quadrature_nodes}")
print(
    f"  stopping: min={args.min_steps}, max={args.main_steps}, "
    f"check={args.check_every}, patience={args.patience_windows}, "
    f"minimum improvement={100.0 * args.min_improvement_rel:.3g}%"
)
print(f"  optimizer: clipped Adam, lr={args.learning_rate:g}")
print("  execution: sequential")

if args.dry_run:
    print("\nDry-run status:")
    for animal in selected_animals:
        action = "refit" if args.force else (
            "skip completed" if matching_complete_summary(animal) is not None else "fit"
        )
        print(f"  LED7/{animal}: {action}")
    raise SystemExit(0)


# %%
# =============================================================================
# Sequential fit loop
# =============================================================================
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
failed_animals = []

for animal in selected_animals:
    output_dir = OUTPUT_ROOT / f"LED7_{animal}"
    animal_log = LOG_DIR / f"{RUN_ID}_LED7_{animal}.log"
    completed_summary = None if args.force else matching_complete_summary(animal)
    if completed_summary is not None:
        print(
            f"\nLED7/{animal}: skipping completed matching fit "
            f"(best={completed_summary['restored_best_step']}, "
            f"checked={completed_summary['completed_steps']})."
        )
        update_ledger(
            {
                "run_id": RUN_ID,
                "animal": animal,
                "status": "complete_existing",
                "started_at": "",
                "ended_at": datetime.now().isoformat(timespec="seconds"),
                "elapsed_minutes": 0.0,
                "return_code": 0,
                "completed_steps": completed_summary.get("completed_steps", ""),
                "restored_best_step": completed_summary.get("restored_best_step", ""),
                "fit_elapsed_minutes": completed_summary.get("fit_elapsed_minutes", ""),
                "output_dir": str(output_dir),
                "animal_log": "",
                "message": "Reused completed matching fit.",
            }
        )
        continue

    started_at = datetime.now().isoformat(timespec="seconds")
    update_ledger(
        {
            "run_id": RUN_ID,
            "animal": animal,
            "status": "running",
            "started_at": started_at,
            "ended_at": "",
            "elapsed_minutes": "",
            "return_code": "",
            "completed_steps": "",
            "restored_best_step": "",
            "fit_elapsed_minutes": "",
            "output_dir": str(output_dir),
            "animal_log": str(animal_log),
            "message": "",
        }
    )

    environment = os.environ.copy()
    environment.update(
        {
            "PROACTIVE_LED_SVI_ANIMAL": str(animal),
            "PROACTIVE_LED_SVI_OUTPUT_ROOT": str(OUTPUT_ROOT),
            "PROACTIVE_LED_SVI_QUADRATURE_NODES": str(args.quadrature_nodes),
            "PROACTIVE_LED_SVI_MAIN_STEPS": str(args.main_steps),
            "PROACTIVE_LED_SVI_MIN_STEPS": str(args.min_steps),
            "PROACTIVE_LED_SVI_CHECK_EVERY": str(args.check_every),
            "PROACTIVE_LED_SVI_PATIENCE_WINDOWS": str(args.patience_windows),
            "PROACTIVE_LED_SVI_MIN_IMPROVEMENT_REL": str(args.min_improvement_rel),
            "PROACTIVE_LED_SVI_LR": str(args.learning_rate),
            "PROACTIVE_LED_SVI_POSTERIOR_SAMPLES": str(args.posterior_samples),
            "PROACTIVE_LED_SVI_SEED": str(args.seed),
        }
    )
    command = [args.python, "-u", str(FIT_SCRIPT)]
    print(f"\nLED7/{animal}: starting at {started_at}")
    print(f"  log: {animal_log}")
    run_start = perf_counter()
    with animal_log.open("w") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=REPO_DIR,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
            log_handle.flush()
        return_code = process.wait()
    elapsed_minutes = (perf_counter() - run_start) / 60.0
    ended_at = datetime.now().isoformat(timespec="seconds")

    summary_path = output_dir / "run_summary.json"
    summary = None
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except (OSError, json.JSONDecodeError):
            summary = None
    succeeded = return_code == 0 and summary is not None and summary.get("status") == "complete"
    status = "complete" if succeeded else "failed"
    message = "" if succeeded else "Fit process or saved finite-value validation failed."
    update_ledger(
        {
            "run_id": RUN_ID,
            "animal": animal,
            "status": status,
            "started_at": started_at,
            "ended_at": ended_at,
            "elapsed_minutes": elapsed_minutes,
            "return_code": return_code,
            "completed_steps": summary.get("completed_steps", "") if summary else "",
            "restored_best_step": summary.get("restored_best_step", "") if summary else "",
            "fit_elapsed_minutes": summary.get("fit_elapsed_minutes", "") if summary else "",
            "output_dir": str(output_dir),
            "animal_log": str(animal_log),
            "message": message,
        }
    )
    print(f"LED7/{animal}: {status} in {elapsed_minutes:.2f} minutes")
    if not succeeded:
        failed_animals.append(animal)
        if args.stop_on_failure:
            break


# %%
# =============================================================================
# Final batch status
# =============================================================================
final_ledger = read_ledger().sort_values("animal")
print("\nBatch ledger:")
print(
    final_ledger[
        [
            "animal",
            "status",
            "fit_elapsed_minutes",
            "restored_best_step",
            "completed_steps",
        ]
    ].to_string(index=False)
)
print(f"Ledger: {LEDGER_PATH}")
if failed_animals:
    raise SystemExit(f"Failed LED7 animals: {failed_animals}")

# %%
