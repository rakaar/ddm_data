# %%
"""
Fit the Gamma/Omega alpha model to the patience12 big SVI condition parameters.

This keeps the original stable3 big-SVI comparison script reusable while pointing
the analysis at the all-animal patience12 restore-best outputs.
"""

# %%
# =============================================================================
# Parameters
# =============================================================================
from pathlib import Path
import os
import runpy


SCRIPT_DIR = Path(__file__).resolve().parent
OBJECTIVE = os.environ.get("BIG_SVI_GAMMA_OMEGA_MSE_OBJECTIVE", "gamma_omega").strip().lower()

os.environ.setdefault("BIG_SVI_GAMMA_OMEGA_OUTPUT_ROOT", str(
    SCRIPT_DIR / "svi_big_gamma_omega_delay_patience12_restore_best_all_animals_outputs"
))
os.environ.setdefault("BIG_SVI_GAMMA_OMEGA_SOURCE_LABEL", "Patience12 big Gamma/Omega/delay SVI")
os.environ.setdefault(
    "BIG_SVI_GAMMA_OMEGA_FIG_BASENAME",
    f"patience12_big_svi_gamma_omega_with_per_animal_mse_alpha_model_{OBJECTIVE}.png"
    if OBJECTIVE != "gamma_omega"
    else "patience12_big_svi_gamma_omega_with_per_animal_mse_alpha_model.png",
)

# %%
# =============================================================================
# Run shared analysis
# =============================================================================
runpy.run_path(
    str(SCRIPT_DIR / "compare_big_svi_gamma_omega_with_mse_alpha_model.py"),
    run_name="__main__",
)
