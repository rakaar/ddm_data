# %%
"""Run the FCT-matched no-truncation lapse SVI fit for six LED7 animals."""

# %%
from pathlib import Path
from datetime import datetime
from time import perf_counter
import argparse
import json
import os
import subprocess

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
FIT_SCRIPT = (
    SCRIPT_DIR
    / "numpyro_svi_proactive_led_step_jump_no_trunc_exp_lapse_single_animal.py"
)
VALIDATION_SUMMARY = (
    SCRIPT_DIR
    / "proactive_led_step_jump_no_trunc_exp_lapse_svi_validation"
    / "no_trunc_exp_lapse_validation_summary.json"
)
DEFAULT_OUTPUT_ROOT = (
    SCRIPT_DIR
    / "numpyro_svi_proactive_led_step_jump_all_on_no_trunc_exp_lapse_"
    "patience12_min50k_restore_best_outputs"
)
ANIMALS = [92, 93, 98, 99, 100, 103]
EXPECTED_TOTAL_TRIALS = {
    92: 21608,
    93: 16486,
    98: 14055,
    99: 14570,
    100: 8727,
    103: 18987,
}
EXPECTED_EARLY_ABORTS = {
    92: 543,
    93: 664,
    98: 453,
    99: 787,
    100: 148,
    103: 486,
}


# %%
# =============================================================================
# Command-line controls
# =============================================================================
parser = argparse.ArgumentParser(
    description="Run all-ON plus OFF, no-truncation, exponential-lapse SVI fits."
)
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--force", action="store_true")
parser.add_argument("--only", nargs="*", type=int)
parser.add_argument("--stop-on-failure", action="store_true")
parser.add_argument("--python", default=str(REPO_DIR / ".venv" / "bin" / "python"))
parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
parser.add_argument("--quadrature-nodes", type=int, default=64)
parser.add_argument("--main-steps", type=int, default=150000)
parser.add_argument("--extended-max-steps", type=int, default=250000)
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
    raise ValueError(f"Unknown LED7 animals {unknown_animals}; expected {ANIMALS}.")
if not FIT_SCRIPT.exists():
    raise FileNotFoundError(FIT_SCRIPT)
if not Path(args.python).exists():
    raise FileNotFoundError(args.python)
if not VALIDATION_SUMMARY.exists():
    raise FileNotFoundError(
        "Run validate_numpyro_proactive_led_step_jump_no_trunc_exp_lapse_"
        f"likelihood.py first: {VALIDATION_SUMMARY}"
    )
validation = json.loads(VALIDATION_SUMMARY.read_text())
if validation.get("status") != "passed":
    raise RuntimeError(f"Likelihood validation did not pass: {VALIDATION_SUMMARY}")
if int(validation.get("recommended_quadrature_nodes", -1)) != args.quadrature_nodes:
    raise RuntimeError(
        "Requested quadrature nodes differ from validation: "
        f"requested={args.quadrature_nodes}, "
        f"validated={validation.get('recommended_quadrature_nodes')}."
    )


# %%
# =============================================================================
# Resumable ledger
# =============================================================================
ledger_columns = [
    "run_id",
    "animal",
    "status",
    "stop_reason",
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
    accepted_status = summary.get("status") in {"complete", "max_steps_restore_best"}
    matches = (
        accepted_status
        and config.get("model_name")
        == "proactive_led_step_jump_all_on_no_trunc_exp_lapse_svi"
        and config.get("truncation") == "none"
        and config.get("led_on_scope") == "all LED_trial == 1 rows"
        and int(config.get("quadrature_nodes", -1)) == args.quadrature_nodes
        and int(config.get("main_steps_initial_max", -1)) == args.main_steps
        and int(config.get("extended_max_steps", -1)) == args.extended_max_steps
        and int(config.get("min_steps", -1)) == args.min_steps
        and int(config.get("check_every", -1)) == args.check_every
        and int(config.get("patience_windows", -1)) == args.patience_windows
        and float(config.get("min_improvement_rel", -1.0))
        == args.min_improvement_rel
        and float(config.get("learning_rate", -1.0)) == args.learning_rate
        and int(config.get("posterior_n_samples", -1)) == args.posterior_samples
        and int(config.get("rng_seed", -1)) == args.seed
        and bool(summary.get("all_posterior_samples_finite", False))
        and int(summary.get("n_nonfinite_losses", -1)) == 0
        and int(summary.get("trial_counts", {}).get("total", -1))
        == EXPECTED_TOTAL_TRIALS[animal]
        and int(
            summary.get("trial_counts", {}).get(
                "early_aborts_below_300ms_retained", -1
            )
        )
        == EXPECTED_EARLY_ABORTS[animal]
    )
    return summary if matches else None


# %%
# =============================================================================
# Planned run
# =============================================================================
print("Proactive LED step-jump no-truncation lapse SVI batch")
print(f"  run ID: {RUN_ID}")
print(f"  animals: {selected_animals}")
print(f"  output root: {OUTPUT_ROOT}")
print(f"  validation: {VALIDATION_SUMMARY}")
print("  data: every LED-ON row plus LED-OFF; no 300 ms removal")
print(f"  quadrature nodes: {args.quadrature_nodes}")
print(
    f"  stopping: min={args.min_steps}, initial max={args.main_steps}, "
    f"extended max={args.extended_max_steps}, check={args.check_every}, "
    f"patience={args.patience_windows}, "
    f"minimum improvement={100.0 * args.min_improvement_rel:.3g}%"
)
print(f"  optimizer: clipped Adam, lr={args.learning_rate:g}")
print("  execution: sequential")

if args.dry_run:
    print("\nDry-run status:")
    for animal in selected_animals:
        action = (
            "refit"
            if args.force
            else (
                "skip completed"
                if matching_complete_summary(animal) is not None
                else "fit"
            )
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
            f"\nLED7/{animal}: reusing completed matching fit "
            f"(best={completed_summary['restored_best_step']}, "
            f"checked={completed_summary['completed_steps']})."
        )
        update_ledger(
            {
                "run_id": RUN_ID,
                "animal": animal,
                "status": "complete_existing",
                "stop_reason": completed_summary.get("stop_reason", ""),
                "started_at": "",
                "ended_at": datetime.now().isoformat(timespec="seconds"),
                "elapsed_minutes": 0.0,
                "return_code": 0,
                "completed_steps": completed_summary.get("completed_steps", ""),
                "restored_best_step": completed_summary.get(
                    "restored_best_step", ""
                ),
                "fit_elapsed_minutes": completed_summary.get(
                    "fit_elapsed_minutes", ""
                ),
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
            "stop_reason": "",
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
            "PROACTIVE_LED_LAPSE_SVI_ANIMAL": str(animal),
            "PROACTIVE_LED_LAPSE_SVI_OUTPUT_ROOT": str(OUTPUT_ROOT),
            "PROACTIVE_LED_LAPSE_SVI_QUADRATURE_NODES": str(
                args.quadrature_nodes
            ),
            "PROACTIVE_LED_LAPSE_SVI_MAIN_STEPS": str(args.main_steps),
            "PROACTIVE_LED_LAPSE_SVI_EXTENDED_MAX_STEPS": str(
                args.extended_max_steps
            ),
            "PROACTIVE_LED_LAPSE_SVI_MIN_STEPS": str(args.min_steps),
            "PROACTIVE_LED_LAPSE_SVI_CHECK_EVERY": str(args.check_every),
            "PROACTIVE_LED_LAPSE_SVI_PATIENCE_WINDOWS": str(
                args.patience_windows
            ),
            "PROACTIVE_LED_LAPSE_SVI_MIN_IMPROVEMENT_REL": str(
                args.min_improvement_rel
            ),
            "PROACTIVE_LED_LAPSE_SVI_LR": str(args.learning_rate),
            "PROACTIVE_LED_LAPSE_SVI_POSTERIOR_SAMPLES": str(
                args.posterior_samples
            ),
            "PROACTIVE_LED_LAPSE_SVI_SEED": str(args.seed),
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
    succeeded = (
        return_code == 0
        and summary is not None
        and summary.get("status") in {"complete", "max_steps_restore_best"}
    )
    status = summary.get("status") if succeeded else "failed"
    message = "" if succeeded else "Fit process or finite-value validation failed."
    update_ledger(
        {
            "run_id": RUN_ID,
            "animal": animal,
            "status": status,
            "stop_reason": "" if summary is None else summary.get("stop_reason", ""),
            "started_at": started_at,
            "ended_at": ended_at,
            "elapsed_minutes": elapsed_minutes,
            "return_code": return_code,
            "completed_steps": "" if summary is None else summary.get("completed_steps", ""),
            "restored_best_step": ""
            if summary is None
            else summary.get("restored_best_step", ""),
            "fit_elapsed_minutes": ""
            if summary is None
            else summary.get("fit_elapsed_minutes", ""),
            "output_dir": str(output_dir),
            "animal_log": str(animal_log),
            "message": message,
        }
    )
    if not succeeded:
        failed_animals.append(animal)
        if args.stop_on_failure:
            break

print(f"\nBatch ledger: {LEDGER_PATH}")
if failed_animals:
    raise SystemExit(f"Failed LED7 animals: {failed_animals}")

if set(selected_animals) == set(ANIMALS):
    completed_summaries = [matching_complete_summary(animal) for animal in ANIMALS]
    if any(summary is None for summary in completed_summaries):
        raise RuntimeError("Six-animal completion audit could not reload every summary.")
    total_trials = sum(summary["trial_counts"]["total"] for summary in completed_summaries)
    total_early_aborts = sum(
        summary["trial_counts"]["early_aborts_below_300ms_retained"]
        for summary in completed_summaries
    )
    if total_trials != 94433 or total_early_aborts != 3081:
        raise RuntimeError(
            "Six-animal row audit failed: "
            f"trials={total_trials}, early aborts={total_early_aborts}."
        )
    print(
        "Six-animal row audit: 94,433 fitting rows and 3,081 aborts below "
        "300 ms retained."
    )
print("All requested LED7 fits are complete or max-steps restore-best outputs.")
