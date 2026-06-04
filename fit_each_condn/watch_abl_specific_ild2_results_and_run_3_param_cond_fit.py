# %%
"""
Watch the ABL-specific NPL+alpha+ILD2-delay fits and launch the 3-param
condition-by-condition fit once each animal's upstream result is ready.

Run this in tmux on each machine. The watcher launches one animal at a time,
then rescans outputs so it does not duplicate completed condition fits.
"""

import datetime as dt
import os
import pickle
import re
import socket
import subprocess
import sys
import time

import pandas as pd

from gamma_omega_alpha_utils import load_batch_animal_pairs, print_batch_animal_table


# %%
# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR = os.path.dirname(__file__)
REPO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CHILD_SCRIPT = os.path.join(
    SCRIPT_DIR,
    "fit_single_rat_condn_by_condn_3_params_fix_w_del_go_from_abl_specific_ild2_all_animals_loop.py",
)

DESIRED_BATCHES = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]
desired_batches_override = os.environ.get("DESIRED_BATCHES_OVERRIDE")
if desired_batches_override:
    DESIRED_BATCHES = [
        batch.strip() for batch in desired_batches_override.split(",") if batch.strip()
    ]

default_batch_dir = os.path.join(REPO_DIR, "fit_animal_by_animal", "batch_csvs")
if not any(
    os.path.exists(os.path.join(default_batch_dir, f"batch_{batch}_valid_and_aborts.csv"))
    for batch in DESIRED_BATCHES
):
    raw_batch_dir = os.path.join(REPO_DIR, "raw_data", "batch_csvs")
    if any(
        os.path.exists(os.path.join(raw_batch_dir, f"batch_{batch}_valid_and_aborts.csv"))
        for batch in DESIRED_BATCHES
    ):
        default_batch_dir = raw_batch_dir

batch_dir = os.environ.get("BATCH_CSV_DIR_OVERRIDE", default_batch_dir)
abort_params_dir = os.environ.get(
    "ABORT_PARAMS_DIR_OVERRIDE",
    os.path.join(REPO_DIR, "aborts_ipl_npl_time_fit_results"),
)
abl_specific_result_dir = os.environ.get(
    "ABL_SPECIFIC_RESULT_DIR_OVERRIDE",
    os.path.join(REPO_DIR, "fit_animal_by_animal", "NPL_alpha_ABL_specific_ILD2_delay_fit_results"),
)

OUTPUT_FOLDER = os.path.join(
    SCRIPT_DIR,
    "each_animal_cond_fit_3_params_fix_w_del_go_from_abl_specific_ild2_pkl_files",
)
CORNER_PLOT_FOLDER = os.path.join(
    SCRIPT_DIR,
    "each_animal_cond_fit_3_params_fix_w_del_go_from_abl_specific_ild2_corner_plots",
)
WATCHER_LOG_DIR = os.path.join(SCRIPT_DIR, "watcher_logs")
WATCHER_LOCK_DIR = os.path.join(SCRIPT_DIR, "watcher_locks")
WATCHER_LOCK_FILE = os.path.join(WATCHER_LOCK_DIR, "watcher.lock")
FAILURE_MARKER_FOLDER = os.path.join(
    SCRIPT_DIR,
    "each_animal_cond_fit_3_params_fix_w_del_go_from_abl_specific_ild2_failed_conditions",
)

ABL_SPECIFIC_RESULT_KEY = "vbmc_norm_alpha_abl_specific_ild2_delay_tied_results"
UPSTREAM_SUFFIX = "NORM_ALPHA_ABL_SPECIFIC_ILD2_DELAY_FROM_ABORTS"
FILENAME_SUFFIX = "_FIX_w_del_go_FROM_ABL_SPECIFIC_ILD2_3_params"

all_ABLs_cond = [20, 40, 60]
all_ILDs_cond = [1, -1, 2, -2, 4, -4, 8, -8, 16, -16]
MIN_TRIALS_PER_CONDITION = 10

COND_FIT_N_JOBS = int(os.environ.get("COND_FIT_N_JOBS", "4"))
COND_FIT_MAX_FAILED_ATTEMPTS = int(os.environ.get("COND_FIT_MAX_FAILED_ATTEMPTS", "3"))
WATCHER_POLL_SECONDS = int(float(os.environ.get("WATCHER_POLL_SECONDS", "300")))
WATCHER_NICE = int(os.environ.get("WATCHER_NICE", "10"))
WATCHER_DRY_RUN = os.environ.get("WATCHER_DRY_RUN", "0").lower() in {
    "1",
    "true",
    "yes",
    "y",
}
WATCHER_ONCE = os.environ.get("WATCHER_ONCE", "0").lower() in {
    "1",
    "true",
    "yes",
    "y",
}
BATCH_ANIMAL_PAIRS_OVERRIDE = os.environ.get("BATCH_ANIMAL_PAIRS_OVERRIDE")


# %%
# =============================================================================
# Helper functions
# =============================================================================
def parse_batch_animal_pairs_override(override_text):
    batch_animal_pairs = []
    for pair_text in override_text.split(","):
        pair_text = pair_text.strip()
        if not pair_text:
            continue
        batch_name, animal_id = pair_text.replace("/", ":").split(":")
        batch_animal_pairs.append((batch_name.strip(), int(animal_id)))
    return batch_animal_pairs


def read_lock_file(lock_file):
    lock_info = {}
    if not os.path.exists(lock_file):
        return lock_info

    with open(lock_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            lock_info[key.strip()] = value.strip()
    return lock_info


def pid_is_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def acquire_watcher_lock():
    if WATCHER_DRY_RUN:
        print("WATCHER_DRY_RUN=1, skipping lock acquisition.")
        return False

    os.makedirs(WATCHER_LOCK_DIR, exist_ok=True)
    hostname = socket.gethostname()

    if os.path.exists(WATCHER_LOCK_FILE):
        lock_info = read_lock_file(WATCHER_LOCK_FILE)
        lock_host = lock_info.get("hostname")
        lock_pid_text = lock_info.get("pid")
        lock_pid = int(lock_pid_text) if lock_pid_text and lock_pid_text.isdigit() else None

        if lock_host and lock_host != hostname:
            raise RuntimeError(
                f"Watcher lock belongs to a different host ({lock_host}): "
                f"{WATCHER_LOCK_FILE}. Remove it manually only if that watcher is gone."
            )

        if lock_host == hostname and lock_pid is not None and pid_is_alive(lock_pid):
            raise RuntimeError(
                f"Watcher lock is active on this host: {WATCHER_LOCK_FILE} "
                f"(pid {lock_pid}). Stop that watcher first."
            )

        print(f"Reclaiming stale watcher lock: {WATCHER_LOCK_FILE}")

    with open(WATCHER_LOCK_FILE, "w", encoding="utf-8") as f:
        f.write(f"hostname={hostname}\n")
        f.write(f"pid={os.getpid()}\n")
        f.write(f"started_at={dt.datetime.now().isoformat(timespec='seconds')}\n")
    return True


def release_watcher_lock(acquired_lock):
    if not acquired_lock:
        return

    try:
        lock_info = read_lock_file(WATCHER_LOCK_FILE)
        if lock_info.get("hostname") == socket.gethostname() and lock_info.get("pid") == str(os.getpid()):
            os.remove(WATCHER_LOCK_FILE)
    except OSError as e:
        print(f"WARNING: Could not release watcher lock {WATCHER_LOCK_FILE}: {e}")


def upstream_result_path(batch_name, animal_id):
    return os.path.join(
        abl_specific_result_dir,
        f"results_{batch_name}_animal_{animal_id}_{UPSTREAM_SUFFIX}.pkl",
    )


def upstream_result_ready(batch_name, animal_id):
    result_pkl = upstream_result_path(batch_name, animal_id)
    if not os.path.exists(result_pkl):
        return False, "missing pkl"

    try:
        with open(result_pkl, "rb") as f:
            fit_results_data = pickle.load(f)
    except Exception as e:
        return False, f"could not load pkl: {e}"

    if ABL_SPECIFIC_RESULT_KEY not in fit_results_data:
        return False, f"missing key {ABL_SPECIFIC_RESULT_KEY}"

    result_samples = fit_results_data[ABL_SPECIFIC_RESULT_KEY]
    missing_keys = [
        key for key in ["w_samples", "del_go_samples"] if key not in result_samples
    ]
    if missing_keys:
        return False, f"missing samples {missing_keys}"

    vbmc_message = str(result_samples.get("message", ""))
    if "stable" not in vbmc_message.lower():
        short_message = vbmc_message.replace("\n", " ")[:120] or "missing VBMC message"
        return False, f"upstream not stable: {short_message}"

    return True, "ready"


def condition_output_paths(batch_name, animal_id, cond_ABL, cond_ILD):
    pkl_file = os.path.join(
        OUTPUT_FOLDER,
        f"vbmc_cond_by_cond_{batch_name}_{animal_id}_{cond_ABL}_ILD_{cond_ILD}{FILENAME_SUFFIX}.pkl",
    )
    corner_plot_file = os.path.join(
        CORNER_PLOT_FOLDER,
        f"corner_cond_by_cond_{batch_name}_{animal_id}_{cond_ABL}_ILD_{cond_ILD}{FILENAME_SUFFIX}.png",
    )
    return pkl_file, corner_plot_file


def condition_failure_marker_file(batch_name, animal_id, cond_ABL, cond_ILD):
    return os.path.join(
        FAILURE_MARKER_FOLDER,
        f"failed_cond_by_cond_{batch_name}_{animal_id}_{cond_ABL}_ILD_{cond_ILD}{FILENAME_SUFFIX}.log",
    )


def failure_marker_attempt_count(batch_name, animal_id, cond_ABL, cond_ILD):
    marker_file = condition_failure_marker_file(batch_name, animal_id, cond_ABL, cond_ILD)
    if not os.path.exists(marker_file):
        return 0

    with open(marker_file, "r", encoding="utf-8", errors="replace") as f:
        return sum(1 for line in f if line.startswith("attempt_at="))


def watcher_log_failure_attempt_count(batch_name, animal_id, cond_ABL, cond_ILD):
    if not os.path.exists(WATCHER_LOG_DIR):
        return 0

    log_prefix = f"cond_fit_{batch_name}_{animal_id}_"
    condition_line_re = re.compile(r"\[\s*(\d+),\s*([+-]?\d+)\]\s+")
    error_count = 0

    for fname in os.listdir(WATCHER_LOG_DIR):
        if not fname.startswith(log_prefix) or not fname.endswith(".log"):
            continue

        log_file = os.path.join(WATCHER_LOG_DIR, fname)
        current_condition = None
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                match = condition_line_re.search(line)
                if match:
                    current_condition = (int(match.group(1)), int(match.group(2)))
                if "-> ERROR" in line and current_condition == (cond_ABL, cond_ILD):
                    error_count += 1
                    current_condition = None

    return error_count


def condition_failure_attempt_count(batch_name, animal_id, cond_ABL, cond_ILD):
    return max(
        failure_marker_attempt_count(batch_name, animal_id, cond_ABL, cond_ILD),
        watcher_log_failure_attempt_count(batch_name, animal_id, cond_ABL, cond_ILD),
    )


def expected_conditions_for_animal(batch_name, animal_id):
    batch_file = os.path.join(batch_dir, f"batch_{batch_name}_valid_and_aborts.csv")
    if not os.path.exists(batch_file):
        return None, f"Batch CSV not found: {batch_file}"

    df = pd.read_csv(batch_file)
    df_animal = df[df["animal"] == int(animal_id)]
    df_animal_success = df_animal[df_animal["success"].isin([1, -1])]
    df_animal_success_rt_filter = df_animal_success[
        (df_animal_success["RTwrtStim"] <= 1) & (df_animal_success["RTwrtStim"] > 0)
    ]

    expected_conditions = []
    low_trial_conditions = []
    for cond_ABL in all_ABLs_cond:
        for cond_ILD in all_ILDs_cond:
            df_cond = df_animal_success_rt_filter[
                (df_animal_success_rt_filter["ABL"] == cond_ABL)
                & (df_animal_success_rt_filter["ILD"] == cond_ILD)
            ]
            n_trials = len(df_cond)
            if n_trials >= MIN_TRIALS_PER_CONDITION:
                expected_conditions.append((cond_ABL, cond_ILD, n_trials))
            else:
                low_trial_conditions.append((cond_ABL, cond_ILD, n_trials))

    return {
        "valid_trials": len(df_animal_success_rt_filter),
        "expected_conditions": expected_conditions,
        "low_trial_conditions": low_trial_conditions,
    }, None


def condition_fit_status(batch_name, animal_id):
    condition_info, error = expected_conditions_for_animal(batch_name, animal_id)
    if error is not None:
        return {"complete": False, "error": error}

    missing_pickles = []
    missing_corners = []
    failed_skipped_conditions = []
    for cond_ABL, cond_ILD, n_trials in condition_info["expected_conditions"]:
        pkl_file, corner_plot_file = condition_output_paths(
            batch_name,
            animal_id,
            cond_ABL,
            cond_ILD,
        )
        condition_label = (cond_ABL, cond_ILD, n_trials)
        pkl_exists = os.path.exists(pkl_file)
        corner_exists = os.path.exists(corner_plot_file)
        failure_attempts = condition_failure_attempt_count(
            batch_name,
            animal_id,
            cond_ABL,
            cond_ILD,
        )

        if (
            (not pkl_exists or not corner_exists)
            and COND_FIT_MAX_FAILED_ATTEMPTS > 0
            and failure_attempts >= COND_FIT_MAX_FAILED_ATTEMPTS
        ):
            failed_skipped_conditions.append((cond_ABL, cond_ILD, n_trials, failure_attempts))
            continue

        if not pkl_exists:
            missing_pickles.append(condition_label)
        if not corner_exists:
            missing_corners.append(condition_label)

    return {
        "complete": len(missing_pickles) == 0 and len(missing_corners) == 0,
        "expected_count": len(condition_info["expected_conditions"]),
        "low_trial_skip_count": len(condition_info["low_trial_conditions"]),
        "valid_trials": condition_info["valid_trials"],
        "missing_pickles": missing_pickles,
        "missing_corners": missing_corners,
        "failed_skipped_conditions": failed_skipped_conditions,
        "error": None,
    }


def run_condition_fit_child(batch_name, animal_id):
    os.makedirs(WATCHER_LOG_DIR, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(
        WATCHER_LOG_DIR,
        f"cond_fit_{batch_name}_{animal_id}_{timestamp}.log",
    )

    child_env = os.environ.copy()
    child_env["BATCH_ANIMAL_PAIRS_OVERRIDE"] = f"{batch_name}:{animal_id}"
    child_env["N_JOBS"] = str(COND_FIT_N_JOBS)
    child_env["DRY_RUN"] = "0"
    child_env.setdefault("DESIRED_BATCHES_OVERRIDE", ",".join(DESIRED_BATCHES))
    child_env.setdefault("BATCH_CSV_DIR_OVERRIDE", batch_dir)
    child_env.setdefault("ABORT_PARAMS_DIR_OVERRIDE", abort_params_dir)
    child_env.setdefault("ABL_SPECIFIC_RESULT_DIR_OVERRIDE", abl_specific_result_dir)

    cmd = [
        "nice",
        "-n",
        str(WATCHER_NICE),
        sys.executable,
        "-u",
        CHILD_SCRIPT,
    ]

    print(f"Launching condition fit for {batch_name}/{animal_id}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Log file: {log_file}")

    with open(log_file, "w", encoding="utf-8") as log_handle:
        log_handle.write(f"Started at {dt.datetime.now().isoformat(timespec='seconds')}\n")
        log_handle.write(f"Batch/animal: {batch_name}/{animal_id}\n")
        log_handle.write(f"Command: {' '.join(cmd)}\n\n")
        log_handle.flush()

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=child_env,
            cwd=SCRIPT_DIR,
        )

        for line in process.stdout:
            print(line, end="")
            log_handle.write(line)
            log_handle.flush()

        return_code = process.wait()
        log_handle.write(f"\nFinished at {dt.datetime.now().isoformat(timespec='seconds')}\n")
        log_handle.write(f"Return code: {return_code}\n")

    print(f"Child finished for {batch_name}/{animal_id} with return code {return_code}")
    return return_code


def print_scan_summary(scan_rows):
    print("\nScan summary")
    print("=" * 90)
    for row in scan_rows:
        batch_name = row["batch"]
        animal_id = row["animal"]
        upstream_text = row["upstream"]
        cond_text = row["condition"]
        print(f"{batch_name}/{animal_id}: upstream={upstream_text}; condition={cond_text}")
    print("=" * 90)


# %%
# =============================================================================
# Load animal list
# =============================================================================
if BATCH_ANIMAL_PAIRS_OVERRIDE:
    batch_animal_pairs = parse_batch_animal_pairs_override(BATCH_ANIMAL_PAIRS_OVERRIDE)
else:
    batch_animal_pairs = load_batch_animal_pairs(batch_dir, DESIRED_BATCHES)

print_batch_animal_table(batch_animal_pairs)
print(f"Batch CSV dir: {batch_dir}")
print(f"Abort params dir passed to child: {abort_params_dir}")
print(f"ABL-specific result dir: {abl_specific_result_dir}")
print(f"Condition output folder: {OUTPUT_FOLDER}")
print(f"Condition corner folder: {CORNER_PLOT_FOLDER}")
print(f"Condition failure marker folder: {FAILURE_MARKER_FOLDER}")
print(f"Child script: {CHILD_SCRIPT}")
print(
    f"WATCHER_DRY_RUN={WATCHER_DRY_RUN}, WATCHER_ONCE={WATCHER_ONCE}, "
    f"COND_FIT_N_JOBS={COND_FIT_N_JOBS}, "
    f"COND_FIT_MAX_FAILED_ATTEMPTS={COND_FIT_MAX_FAILED_ATTEMPTS}, "
    f"WATCHER_POLL_SECONDS={WATCHER_POLL_SECONDS}"
)


# %%
# =============================================================================
# Watch loop
# =============================================================================
acquired_lock = acquire_watcher_lock()

try:
    while True:
        scan_rows = []
        ready_incomplete_animals = []
        complete_animals = []

        for batch_name, animal_id in batch_animal_pairs:
            upstream_ready, upstream_message = upstream_result_ready(batch_name, animal_id)
            if not upstream_ready:
                scan_rows.append(
                    {
                        "batch": batch_name,
                        "animal": animal_id,
                        "upstream": upstream_message,
                        "condition": "waiting",
                    }
                )
                continue

            status = condition_fit_status(batch_name, animal_id)
            if status["error"] is not None:
                scan_rows.append(
                    {
                        "batch": batch_name,
                        "animal": animal_id,
                        "upstream": "ready",
                        "condition": status["error"],
                    }
                )
                continue

            if status["complete"]:
                complete_animals.append((batch_name, animal_id))
                condition_text = (
                    f"complete {status['expected_count']} expected conditions "
                    f"({status['low_trial_skip_count']} low-trial skips)"
                )
                if status["failed_skipped_conditions"]:
                    condition_text += (
                        f", failed_skips={len(status['failed_skipped_conditions'])}"
                    )
            else:
                ready_incomplete_animals.append((batch_name, animal_id, status))
                condition_text = (
                    f"incomplete, expected={status['expected_count']}, "
                    f"missing_pickles={len(status['missing_pickles'])}, "
                    f"missing_corners={len(status['missing_corners'])}, "
                    f"failed_skips={len(status['failed_skipped_conditions'])}, "
                    f"low_trial_skips={status['low_trial_skip_count']}"
                )

            scan_rows.append(
                {
                    "batch": batch_name,
                    "animal": animal_id,
                    "upstream": "ready",
                    "condition": condition_text,
                }
            )

        print(f"\n{dt.datetime.now().isoformat(timespec='seconds')}")
        print_scan_summary(scan_rows)

        if len(complete_animals) == len(batch_animal_pairs):
            print(
                "All configured animals have complete condition fits, or only "
                "failed-skipped conditions remain. Watcher is done."
            )
            break

        if ready_incomplete_animals:
            batch_name, animal_id, status = ready_incomplete_animals[0]
            print(
                f"Next launch candidate: {batch_name}/{animal_id}; "
                f"missing_pickles={len(status['missing_pickles'])}, "
                f"missing_corners={len(status['missing_corners'])}, "
                f"failed_skips={len(status['failed_skipped_conditions'])}"
            )
            if status["missing_pickles"][:5]:
                print(f"First missing pickles: {status['missing_pickles'][:5]}")
            if status["missing_corners"][:5]:
                print(f"First missing corners: {status['missing_corners'][:5]}")
            if status["failed_skipped_conditions"][:5]:
                print(f"First failed-skipped conditions: {status['failed_skipped_conditions'][:5]}")

            if WATCHER_DRY_RUN:
                print(
                    "WATCHER_DRY_RUN=1, would launch child for "
                    f"{batch_name}/{animal_id} and then rescan."
                )
                if WATCHER_ONCE:
                    break
            else:
                return_code = run_condition_fit_child(batch_name, animal_id)
                if return_code != 0:
                    print(
                        f"WARNING: child returned {return_code}. Sleeping before retry/rescan."
                    )
                    time.sleep(WATCHER_POLL_SECONDS)
                continue
        else:
            print("No upstream-ready incomplete animals available right now.")
            if WATCHER_ONCE:
                break

        print(f"Sleeping {WATCHER_POLL_SECONDS} seconds before next scan...")
        time.sleep(WATCHER_POLL_SECONDS)
finally:
    release_watcher_lock(acquired_lock)

# %%
