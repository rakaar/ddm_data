# %%
"""
Run refreshed VBMC NPL+alpha+lapse scalar-delay fits for four suspicious animals.

This runner is deliberately narrow: it defaults to the same four animals used in
the SVI early-best versus random-init comparison. Existing posterior summaries
are skipped unless --force is passed.
"""

# %%
# =============================================================================
# Parameters
# =============================================================================
import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "vbmc_npl_alpha_lapse_scalar_delay_rerun"
DEFAULT_ANIMALS = ["LED7:98", "LED7:100", "LED34_even:52", "LED34_even:60"]


# %%
def parse_pair(pair_text):
    batch, animal = pair_text.split(":", 1)
    return batch, int(animal)


parser = argparse.ArgumentParser(description="Run VBMC NPL+alpha+lapse scalar-delay fits for selected animals.")
parser.add_argument("--only", nargs="+", default=DEFAULT_ANIMALS, help="Animals as batch:animal")
parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Output root")
parser.add_argument("--python", default=sys.executable, help="Python interpreter")
parser.add_argument("--posterior-samples", type=int, default=100000, help="Posterior samples per fit summary")
parser.add_argument("--max-fun-evals", type=int, default=200 * (2 + 8), help="VBMC max_fun_evals")
parser.add_argument("--force", action="store_true", help="Re-run even when posterior summary exists")
args = parser.parse_args()

output_root = Path(args.output_root).expanduser().resolve()
log_dir = output_root / "_batch_logs"
log_dir.mkdir(parents=True, exist_ok=True)
python_path = Path(args.python).expanduser()
if not python_path.is_absolute():
    python_path = Path.cwd() / python_path
if not python_path.exists():
    raise FileNotFoundError(python_path)

animals = [parse_pair(pair_text) for pair_text in args.only]
single_script = SCRIPT_DIR / "vbmc_npl_alpha_lapse_scalar_delay_refreshed_single_animal.py"
if not single_script.exists():
    raise FileNotFoundError(single_script)

run_id = time.strftime("%Y%m%d_%H%M%S")
ledger_path = log_dir / f"vbmc_npl_alpha_lapse_scalar_delay_four_animal_status_{run_id}.csv"
latest_ledger_path = log_dir / "vbmc_npl_alpha_lapse_scalar_delay_four_animal_status_latest.csv"

fieldnames = [
    "run_id",
    "run_index",
    "n_runs",
    "batch_name",
    "animal",
    "status",
    "elapsed_seconds",
    "return_code",
    "output_dir",
    "fit_log",
    "posterior_summary_csv",
    "run_summary_csv",
    "error",
]

rows = []
for run_index, (batch_name, animal) in enumerate(animals, start=1):
    output_dir = output_root / f"{batch_name}_{animal}"
    rows.append(
        {
            "run_id": run_id,
            "run_index": run_index,
            "n_runs": len(animals),
            "batch_name": batch_name,
            "animal": animal,
            "status": "pending",
            "elapsed_seconds": "",
            "return_code": "",
            "output_dir": str(output_dir),
            "fit_log": "",
            "posterior_summary_csv": str(output_dir / "vbmc_norm_alpha_lapse_scalar_t_E_aff_posterior_summary.csv"),
            "run_summary_csv": str(output_dir / "vbmc_norm_alpha_lapse_scalar_t_E_aff_run_summary.csv"),
            "error": "",
        }
    )


def write_ledgers():
    for path in [ledger_path, latest_ledger_path]:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


write_ledgers()
print(f"Ledger: {ledger_path}")
print(f"Latest ledger: {latest_ledger_path}")


# %%
# =============================================================================
# Run selected animals sequentially
# =============================================================================
for row in rows:
    batch_name = row["batch_name"]
    animal = int(row["animal"])
    output_dir = Path(row["output_dir"])
    summary_csv = Path(row["posterior_summary_csv"])
    log_path = log_dir / f"{run_id}_{int(row['run_index']):02d}_{batch_name}_{animal}_fit.log"
    row["fit_log"] = str(log_path)

    if summary_csv.exists() and not args.force:
        row["status"] = "skipped_existing"
        row["elapsed_seconds"] = "0"
        row["return_code"] = "0"
        write_ledgers()
        print(f"[{row['run_index']}/{len(rows)}] Skipping {batch_name}/{animal}: {summary_csv} exists")
        continue

    cmd = [
        str(python_path),
        str(single_script),
        "--batch",
        batch_name,
        "--animal",
        str(animal),
        "--output-root",
        str(output_root),
        "--init-type",
        "norm",
        "--posterior-samples",
        str(args.posterior_samples),
        "--max-fun-evals",
        str(args.max_fun_evals),
    ]

    print("=" * 80)
    print(f"[{row['run_index']}/{len(rows)}] Running {batch_name}/{animal}")
    print(" ".join(cmd))
    print(f"Log: {log_path}")
    print("=" * 80)
    row["status"] = "running"
    write_ledgers()

    t0 = time.time()
    try:
        with open(log_path, "w") as log_f:
            proc = subprocess.Popen(
                cmd,
                cwd=str(SCRIPT_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                log_f.write(line)
                log_f.flush()
            return_code = proc.wait()

        elapsed = time.time() - t0
        row["elapsed_seconds"] = f"{elapsed:.3f}"
        row["return_code"] = str(return_code)
        if return_code == 0:
            row["status"] = "completed"
        else:
            row["status"] = "failed"
            row["error"] = f"return_code={return_code}"
    except Exception as exc:
        elapsed = time.time() - t0
        row["elapsed_seconds"] = f"{elapsed:.3f}"
        row["return_code"] = "999"
        row["status"] = "failed"
        row["error"] = repr(exc)

    write_ledgers()
    if row["status"] == "failed":
        print(f"Failed {batch_name}/{animal}: {row['error']}")
        raise SystemExit(2)

print("\nRun summary:")
print(pd.DataFrame(rows).to_string(index=False))
print(f"Ledger: {ledger_path}")

# %%
