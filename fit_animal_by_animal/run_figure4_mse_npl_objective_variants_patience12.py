# %%
"""
Run Figure 4-style diagnostics for patience12 MSE objective variants.

The paired Gamma/Omega comparison script writes one MSE parameter CSV per
objective. This runner points the Figure 4 diagnostic at each CSV and keeps the
outputs in the same per-objective folders.
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

FIGURE4_SCRIPT = SCRIPT_DIR / "figure4_mse_npl_params_patience12_big_svi.py"
BIG_SVI_ROOT = REPO_DIR / "fit_each_condn" / "svi_big_gamma_omega_delay_patience12_restore_best_all_animals_outputs"
VARIANT_ROOT = BIG_SVI_ROOT / "mse_alpha_model_comparison" / "objective_variants"
SUMMARY_CSV = VARIANT_ROOT / "figure4_objective_variant_outputs.csv"

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
output_rows = []

for objective, objective_label in OBJECTIVES:
    output_dir = VARIANT_ROOT / objective
    mse_param_csv = output_dir / "per_animal_mse_gamma_omega_alpha_params.csv"
    if not mse_param_csv.exists():
        raise FileNotFoundError(
            f"Missing MSE params for objective {objective}: {mse_param_csv}. "
            "Run fit_each_condn/run_patience12_big_svi_gamma_omega_mse_objective_variants.py first."
        )

    fig_dir = output_dir / "figure4_mse_params_diagnostics"
    fig_basename = f"patience12_mse_objective_{objective}_figure4_2x3.png"
    pkl_basename = f"patience12_mse_objective_{objective}_figure4_2x3.pkl"

    env = os.environ.copy()
    env["MSE_FIG4_OBJECTIVE"] = objective
    env["MSE_FIG4_OBJECTIVE_LABEL"] = objective_label
    env["MSE_FIG4_BIG_SVI_ROOT"] = str(BIG_SVI_ROOT)
    env["MSE_FIG4_MSE_PARAM_CSV"] = str(mse_param_csv)
    env["MSE_FIG4_OUTPUT_DIR"] = str(fig_dir)
    env["MSE_FIG4_FIG_BASENAME"] = fig_basename
    env["MSE_FIG4_PKL_BASENAME"] = pkl_basename

    print("\n" + "=" * 80)
    print(f"Running Figure 4 diagnostics for objective: {objective} ({objective_label})")
    print(f"MSE params: {mse_param_csv}")
    print(f"Output folder: {fig_dir}")
    print("=" * 80)
    subprocess.run(
        [sys.executable, "-u", str(FIGURE4_SCRIPT)],
        cwd=REPO_DIR,
        env=env,
        check=True,
    )

    output_rows.append(
        {
            "objective": objective,
            "objective_label": objective_label,
            "mse_param_csv": str(mse_param_csv),
            "figure_png": str(fig_dir / fig_basename),
            "data_pkl": str(fig_dir / pkl_basename),
        }
    )

pd.DataFrame(output_rows).to_csv(SUMMARY_CSV, index=False)
print(f"\nSaved Figure 4 objective variant output manifest: {SUMMARY_CSV}")
print("Completed patience12 Figure 4 MSE objective variants.")

# %%
