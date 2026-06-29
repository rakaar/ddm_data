# %%
"""
Run per-animal MSE alpha-model fits for patience12 big SVI Gamma/Omega variants.

Each objective writes into its own folder so Gamma+Omega, Gamma-only, and
Omega-only fits can be compared without overwriting the original default output.
"""

# %%
# =============================================================================
# Parameters
# =============================================================================
from pathlib import Path
import os
import subprocess
import sys

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

FIT_SCRIPT = SCRIPT_DIR / "compare_patience12_big_svi_gamma_omega_with_mse_alpha_model.py"
OUTPUT_ROOT = SCRIPT_DIR / "svi_big_gamma_omega_delay_patience12_restore_best_all_animals_outputs"
VARIANT_ROOT = OUTPUT_ROOT / "mse_alpha_model_comparison" / "objective_variants"
SUMMARY_CSV = VARIANT_ROOT / "mse_objective_variant_metrics_summary.csv"

OBJECTIVES = [
    ("gamma_omega", "fit Gamma + Omega"),
    ("gamma_only", "fit Gamma only"),
    ("omega_only", "fit Omega only"),
]


# %%
# =============================================================================
# Run variants
# =============================================================================
VARIANT_ROOT.mkdir(parents=True, exist_ok=True)
metric_frames = []

for objective, objective_label in OBJECTIVES:
    output_dir = VARIANT_ROOT / objective
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_basename = f"patience12_mse_objective_{objective}_gamma_omega.png"

    env = os.environ.copy()
    env["BIG_SVI_GAMMA_OMEGA_MSE_OBJECTIVE"] = objective
    env["BIG_SVI_GAMMA_OMEGA_OUTPUT_ROOT"] = str(OUTPUT_ROOT)
    env["BIG_SVI_GAMMA_OMEGA_OUTPUT_DIR"] = str(output_dir)
    env["BIG_SVI_GAMMA_OMEGA_SOURCE_LABEL"] = "Patience12 big Gamma/Omega/delay SVI"
    env["BIG_SVI_GAMMA_OMEGA_FIG_BASENAME"] = fig_basename

    print("\n" + "=" * 80)
    print(f"Running objective: {objective} ({objective_label})")
    print(f"Output folder: {output_dir}")
    print("=" * 80)
    subprocess.run(
        [sys.executable, "-u", str(FIT_SCRIPT)],
        cwd=REPO_DIR,
        env=env,
        check=True,
    )

    metrics_csv = output_dir / "per_animal_mse_gamma_omega_metrics.csv"
    if metrics_csv.exists():
        metrics_df = pd.read_csv(metrics_csv)
        metrics_df.insert(0, "variant_output_dir", str(output_dir))
        metric_frames.append(metrics_df)

if metric_frames:
    summary_df = pd.concat(metric_frames, ignore_index=True)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSaved objective variant metrics summary: {SUMMARY_CSV}")

print("\nCompleted patience12 MSE objective variants.")

# %%
