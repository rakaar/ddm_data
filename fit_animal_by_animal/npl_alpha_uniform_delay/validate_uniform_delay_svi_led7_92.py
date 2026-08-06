# %%
"""Validate the uniform-delay evidence terms for all LED7/92 conditions.

The analytic uniform-delay CDF uses the integrated spectral expression, while
the reference averages the existing point-delay CDF/PDF over 64
Gauss-Legendre nodes.  This checks the exact parameter values used to
initialize the 67-parameter LED7/92 fit on a 1 ms grid.
"""

# %%
# =============================================================================
# Editable parameters
# =============================================================================
from pathlib import Path
import sys

REPO_DIR = Path(__file__).resolve().parents[2]
ANIMAL_FIT_DIR = REPO_DIR / "fit_animal_by_animal"
SCRIPT_DIR = Path(__file__).resolve().parent

BATCH_NAME = "LED7"
ANIMAL = 92
DELAY_WIDTH_S = 0.005
TIME_STEP_S = 0.001
TIME_MAX_S = 1.0
K_MAX = 10
INTEGRATED_CDF_TERMS = 200
QUADRATURE_NODES = 64

REFERENCE_DIR = (
    ANIMAL_FIT_DIR
    / "numpyro_svi_npl_alpha_condition_delay_patience12_restore_best_outputs"
    / f"{BATCH_NAME}_{ANIMAL}"
)
POSTERIOR_NPZ = REFERENCE_DIR / "main_fullrank_posterior_samples.npz"
CONDITION_CSV = REFERENCE_DIR / "condition_table.csv"


# %%
# =============================================================================
# Imports and reference parameters
# =============================================================================
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pandas as pd

for import_path in (ANIMAL_FIT_DIR, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import numpyro_npl_alpha_svi_utils as point_utils
import uniform_delay_likelihood_utils as uniform_utils

for required_path in (POSTERIOR_NPZ, CONDITION_CSV):
    if not required_path.exists():
        raise FileNotFoundError(required_path)

posterior = np.load(POSTERIOR_NPZ)
condition_table = (
    pd.read_csv(CONDITION_CSV)
    .sort_values("condition_id")
    .reset_index(drop=True)
)
if len(condition_table) != 30:
    raise RuntimeError(f"Expected 30 conditions, found {len(condition_table)}.")

params = {
    name: float(np.mean(np.asarray(posterior[name], dtype=float)))
    for name in point_utils.GLOBAL_PARAM_NAMES
}
delay_centers = np.mean(
    np.asarray(posterior["t_E_aff"], dtype=float),
    axis=0,
)
if delay_centers.shape != (30,):
    raise RuntimeError(f"Unexpected delay-center shape: {delay_centers.shape}.")

Z_E = (params["w"] - 0.5) * 2.0 * params["theta_E"]
elapsed_grid_s = np.arange(
    0.0,
    TIME_MAX_S + 0.5 * TIME_STEP_S,
    TIME_STEP_S,
)
nodes, weights = np.polynomial.legendre.leggauss(QUADRATURE_NODES)
quadrature_weights = weights / 2.0


# %%
# =============================================================================
# Compare analytic and numerical delay averages
# =============================================================================
rows = []
for condition, center in zip(
    condition_table.itertuples(index=False),
    delay_centers,
):
    delay_low = float(center - 0.5 * DELAY_WIDTH_S)
    delay_high = float(center + 0.5 * DELAY_WIDTH_S)
    delay_nodes = center + 0.5 * DELAY_WIDTH_S * nodes

    if not (0.0 <= delay_low < delay_high <= 1.0):
        raise RuntimeError(
            f"Invalid validation delay interval [{delay_low}, {delay_high}]."
        )

    for bound in (-1, 1):
        analytic_cdf = np.asarray(
            uniform_utils.uniform_delay_bound_cdf_alpha_jax(
                jnp.asarray(elapsed_grid_s),
                bound,
                delay_low,
                delay_high,
                float(condition.ABL),
                float(condition.ILD),
                params["rate_lambda"],
                params["T_0"],
                params["theta_E"],
                Z_E,
                params["rate_norm_l"],
                params["alpha"],
                INTEGRATED_CDF_TERMS,
            )
        )
        quadrature_cdf = np.asarray(
            point_utils.CDF_E_alpha_jax(
                jnp.asarray(
                    elapsed_grid_s[:, None] - delay_nodes[None, :]
                ),
                bound,
                float(condition.ABL),
                float(condition.ILD),
                params["rate_lambda"],
                params["T_0"],
                params["theta_E"],
                Z_E,
                params["rate_norm_l"],
                params["alpha"],
                K_MAX,
            )
        ) @ quadrature_weights

        analytic_pdf = np.asarray(
            uniform_utils.uniform_delay_bound_pdf_alpha_jax(
                jnp.asarray(elapsed_grid_s),
                bound,
                delay_low,
                delay_high,
                float(condition.ABL),
                float(condition.ILD),
                params["rate_lambda"],
                params["T_0"],
                params["theta_E"],
                Z_E,
                params["rate_norm_l"],
                params["alpha"],
                K_MAX,
            )
        )
        quadrature_pdf = np.asarray(
            point_utils.rho_E_alpha_jax(
                jnp.asarray(
                    elapsed_grid_s[:, None] - delay_nodes[None, :]
                ),
                bound,
                float(condition.ABL),
                float(condition.ILD),
                params["rate_lambda"],
                params["T_0"],
                params["theta_E"],
                Z_E,
                params["rate_norm_l"],
                params["alpha"],
                K_MAX,
            )
        ) @ quadrature_weights

        cdf_difference = np.abs(analytic_cdf - quadrature_cdf)
        pdf_difference = np.abs(analytic_pdf - quadrature_pdf)
        rows.append(
            {
                "ABL": float(condition.ABL),
                "ILD": float(condition.ILD),
                "bound": bound,
                "center_ms": 1e3 * center,
                "width_ms": 1e3 * DELAY_WIDTH_S,
                "max_abs_cdf_error": float(np.max(cdf_difference)),
                "cdf_error_time_ms": float(
                    1e3 * elapsed_grid_s[np.argmax(cdf_difference)]
                ),
                "max_abs_pdf_error": float(np.max(pdf_difference)),
                "pdf_error_time_ms": float(
                    1e3 * elapsed_grid_s[np.argmax(pdf_difference)]
                ),
            }
        )

validation_df = pd.DataFrame(rows)
if len(validation_df) != 60:
    raise RuntimeError(f"Expected 60 condition/bound checks, found {len(validation_df)}.")

worst_cdf = validation_df.loc[validation_df["max_abs_cdf_error"].idxmax()]
worst_pdf = validation_df.loc[validation_df["max_abs_pdf_error"].idxmax()]
print(
    f"Validated {len(condition_table)} conditions x 2 bounds on a "
    f"{1e3 * TIME_STEP_S:.0f} ms grid."
)
print(
    "Worst CDF error: "
    f"{worst_cdf['max_abs_cdf_error']:.6g} at "
    f"ABL={worst_cdf['ABL']:g}, ILD={worst_cdf['ILD']:+g}, "
    f"bound={int(worst_cdf['bound']):+d}, "
    f"t={worst_cdf['cdf_error_time_ms']:.0f} ms"
)
print(
    "Worst PDF error: "
    f"{worst_pdf['max_abs_pdf_error']:.6g} at "
    f"ABL={worst_pdf['ABL']:g}, ILD={worst_pdf['ILD']:+g}, "
    f"bound={int(worst_pdf['bound']):+d}, "
    f"t={worst_pdf['pdf_error_time_ms']:.0f} ms"
)

if validation_df["max_abs_cdf_error"].max() > 1e-4:
    raise AssertionError("K=200 integrated-CDF error exceeded 1e-4.")
if validation_df["max_abs_pdf_error"].max() > 1e-8:
    raise AssertionError("Analytic uniform-delay PDF error exceeded 1e-8.")
print("PASS: analytic uniform-delay evidence terms match 64-node quadrature.")
