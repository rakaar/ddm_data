# %%
"""
Validate the two choice-collapsed NPL+alpha RT likelihoods on LED7/92.

Checks:
1. reactive-only density is the sum over evidence bounds;
2. proactive+reactive density is the sum over existing bound-specific races;
3. the proactive+reactive sum is invariant to del_go;
4. the algebraically collapsed race density matches the bound sum;
5. retained-window likelihoods and gradients are finite;
6. numerical integration over 0-1 s agrees with the retained-window mass.
"""

# %%
# =============================================================================
# Paths and tolerances
# =============================================================================
from pathlib import Path
import pickle
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

BATCH_NAME = "LED7"
ANIMAL = 92
N_VECTOR_TRIALS = 1000
N_GRADIENT_TRIALS = 256
INTEGRATION_STEP_S = 0.0001
K_MAX = 10

COLLAPSE_ATOL = 2e-12
DEL_GO_ATOL = 2e-12
INTEGRAL_ATOL = 5e-3

BATCH_CSV = (
    REPO_DIR
    / "raw_data"
    / "batch_csvs"
    / f"batch_{BATCH_NAME}_valid_and_aborts.csv"
)
REFERENCE_DIR = (
    SCRIPT_DIR
    / "numpyro_svi_npl_alpha_condition_delay_patience12_restore_best_outputs"
    / f"{BATCH_NAME}_{ANIMAL}"
)
REFERENCE_NPZ = REFERENCE_DIR / "main_fullrank_posterior_samples.npz"
REFERENCE_CONDITIONS = REFERENCE_DIR / "condition_table.csv"
ABORT_PKL = (
    REPO_DIR
    / "aborts_ipl_npl_time_fit_results"
    / f"results_{BATCH_NAME}_animal_{ANIMAL}.pkl"
)


# %%
# =============================================================================
# Imports and real-data inputs
# =============================================================================
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pandas as pd

sys.path.insert(0, str(SCRIPT_DIR))
import numpyro_npl_alpha_rt_only_svi_utils as rt_utils
import numpyro_npl_alpha_svi_utils as base_utils

for required_path in (
    BATCH_CSV,
    REFERENCE_NPZ,
    REFERENCE_CONDITIONS,
    ABORT_PKL,
):
    if not required_path.exists():
        raise FileNotFoundError(required_path)

condition_table = (
    pd.read_csv(REFERENCE_CONDITIONS)
    .sort_values("condition_id")
    .reset_index(drop=True)
)

with np.load(REFERENCE_NPZ) as posterior:
    params = {
        name: jnp.asarray(
            float(np.mean(np.asarray(posterior[name], dtype=float))),
            dtype=jnp.float64,
        )
        for name in rt_utils.GLOBAL_PARAM_NAMES
    }
    params["t_E_aff"] = jnp.asarray(
        np.mean(np.asarray(posterior["t_E_aff"], dtype=float), axis=0),
        dtype=jnp.float64,
    )

rt_utils.assert_no_del_go(params)

with ABORT_PKL.open("rb") as handle:
    abort_results = pickle.load(handle)["vbmc_aborts_results"]
V_A = float(np.mean(abort_results["V_A_samples"]))
theta_A = float(np.mean(abort_results["theta_A_samples"]))
t_A_aff = float(np.mean(abort_results["t_A_aff_samp"]))

valid_df = pd.read_csv(BATCH_CSV)
valid_df = valid_df[
    (valid_df["animal"].astype(int) == ANIMAL)
    & valid_df["success"].isin([1, -1])
    & (valid_df["RTwrtStim"] >= 0)
    & (valid_df["RTwrtStim"] < 1)
    & valid_df["ABL"].isin([20, 40, 60])
].dropna(
    subset=[
        "RTwrtStim",
        "TotalFixTime",
        "intended_fix",
        "ABL",
        "ILD",
    ]
).copy()
valid_df = valid_df.merge(
    condition_table[["ABL", "ILD", "condition_id"]],
    on=["ABL", "ILD"],
    how="left",
    validate="many_to_one",
)
if valid_df["condition_id"].isna().any():
    raise RuntimeError("Condition mapping failed.")


def make_data(df):
    return {
        "rt_wrt_stim": jnp.asarray(
            df["RTwrtStim"].to_numpy(dtype=float)
        ),
        "total_fix": jnp.asarray(
            df["TotalFixTime"].to_numpy(dtype=float)
        ),
        "t_stim": jnp.asarray(
            df["intended_fix"].to_numpy(dtype=float)
        ),
        "ABL": jnp.asarray(df["ABL"].to_numpy(dtype=float)),
        "ILD": jnp.asarray(df["ILD"].to_numpy(dtype=float)),
        "condition_id": jnp.asarray(
            df["condition_id"].to_numpy(dtype=int)
        ),
        "V_A": jnp.asarray(V_A, dtype=jnp.float64),
        "theta_A": jnp.asarray(theta_A, dtype=jnp.float64),
        "t_A_aff": jnp.asarray(t_A_aff, dtype=jnp.float64),
        "T_trunc": jnp.asarray(0.300, dtype=jnp.float64),
        "rt_lower": jnp.asarray(0.0, dtype=jnp.float64),
        "rt_upper": jnp.asarray(1.0, dtype=jnp.float64),
    }


vector_df = valid_df.head(N_VECTOR_TRIALS).copy()
vector_data = make_data(vector_df)

print(f"Validation trials: {len(vector_df)}")
print(f"Global parameters: {rt_utils.GLOBAL_PARAM_NAMES}")
print(f"Parameter count: {rt_utils.parameter_count(len(condition_table))}")
print(
    "Fixed proactive parameters: "
    f"V_A={V_A:.6g}, theta_A={theta_A:.6g}, "
    f"t_A_aff={1e3 * t_A_aff:.3f} ms"
)


# %%
# =============================================================================
# Density identities and del_go cancellation
# =============================================================================
t_E_aff = params["t_E_aff"][vector_data["condition_id"]]
Z_E = (params["w"] - 0.5) * 2.0 * params["theta_E"]
reactive_time = vector_data["rt_wrt_stim"] - t_E_aff

reactive_direct = (
    base_utils.rho_E_alpha_jax(
        reactive_time,
        1,
        vector_data["ABL"],
        vector_data["ILD"],
        params["rate_lambda"],
        params["T_0"],
        params["theta_E"],
        Z_E,
        params["rate_norm_l"],
        params["alpha"],
        K_MAX,
    )
    + base_utils.rho_E_alpha_jax(
        reactive_time,
        -1,
        vector_data["ABL"],
        vector_data["ILD"],
        params["rate_lambda"],
        params["T_0"],
        params["theta_E"],
        Z_E,
        params["rate_norm_l"],
        params["alpha"],
        K_MAX,
    )
)
reactive_collapsed = rt_utils.choice_collapsed_reactive_pdf(
    params,
    vector_data,
    K_max=K_MAX,
)
reactive_max_abs_diff = float(
    np.max(
        np.abs(
            np.asarray(reactive_direct)
            - np.asarray(reactive_collapsed)
        )
    )
)

proactive_simplified = (
    rt_utils.choice_collapsed_proactive_reactive_pdf_simplified(
        params,
        vector_data,
        K_max=K_MAX,
    )
)
del_go_rows = []
for del_go_s in (0.0, 0.05, 0.10, 0.199):
    bound_sum = rt_utils.choice_collapsed_proactive_reactive_pdf(
        params,
        vector_data,
        K_max=K_MAX,
        del_go_s=del_go_s,
    )
    max_abs_diff = float(
        np.max(
            np.abs(
                np.asarray(bound_sum)
                - np.asarray(proactive_simplified)
            )
        )
    )
    del_go_rows.append(
        {
            "del_go_s": del_go_s,
            "max_abs_diff_from_simplified": max_abs_diff,
        }
    )

del_go_check_df = pd.DataFrame(del_go_rows)
proactive_max_abs_diff = float(
    del_go_check_df["max_abs_diff_from_simplified"].max()
)

print("\nChoice-collapse checks:")
print(f"  reactive max abs difference: {reactive_max_abs_diff:.3e}")
print(del_go_check_df.to_string(index=False))

if reactive_max_abs_diff > COLLAPSE_ATOL:
    raise RuntimeError("Reactive choice-collapse identity failed.")
if proactive_max_abs_diff > DEL_GO_ATOL:
    raise RuntimeError(
        "Proactive choice-collapse/del_go cancellation check failed."
    )


# %%
# =============================================================================
# Finite likelihood and gradient checks
# =============================================================================
gradient_data = make_data(valid_df.head(N_GRADIENT_TRIALS))

for process_mode in rt_utils.PROCESS_MODES:
    terms = rt_utils.npl_alpha_rt_only_condition_delay_loglike_terms(
        params,
        vector_data,
        process_mode,
        K_max=K_MAX,
    )
    arrays = {
        key: np.asarray(value)
        for key, value in terms.items()
    }
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise RuntimeError(f"Non-finite {process_mode} likelihood terms.")
    if np.any(arrays["retained_window_mass"] <= 0):
        raise RuntimeError(f"Non-positive {process_mode} retained mass.")

    loglike = rt_utils.npl_alpha_rt_only_condition_delay_loglike(
        params,
        gradient_data,
        process_mode,
        K_max=K_MAX,
    )
    gradient = jax.grad(
        lambda values: (
            rt_utils.npl_alpha_rt_only_condition_delay_loglike(
                values,
                gradient_data,
                process_mode,
                K_max=K_MAX,
            )
        )
    )(params)

    if not np.isfinite(float(loglike)):
        raise RuntimeError(f"Non-finite {process_mode} log likelihood.")
    if not rt_utils.tree_all_finite(gradient):
        raise RuntimeError(f"Non-finite {process_mode} gradient.")

    print(
        f"{process_mode}: loglike={float(loglike):.6f}; "
        f"normalized density range="
        f"[{arrays['normalized_pdf'].min():.3g}, "
        f"{arrays['normalized_pdf'].max():.3g}]; "
        "gradient finite=True"
    )


# %%
# =============================================================================
# Numerical retained-window area checks
# =============================================================================
rt_grid_s = np.arange(
    0.0,
    1.0 + 0.5 * INTEGRATION_STEP_S,
    INTEGRATION_STEP_S,
)
target_t_stim = (0.20, 0.50, 1.00)
integration_rows = []

for target in target_t_stim:
    row_idx = int(
        np.argmin(
            np.abs(
                valid_df["intended_fix"].to_numpy(dtype=float)
                - target
            )
        )
    )
    row = valid_df.iloc[row_idx]
    grid_df = pd.DataFrame(
        {
            "RTwrtStim": rt_grid_s,
            "TotalFixTime": float(row["intended_fix"]) + rt_grid_s,
            "intended_fix": float(row["intended_fix"]),
            "ABL": float(row["ABL"]),
            "ILD": float(row["ILD"]),
            "condition_id": int(row["condition_id"]),
        }
    )
    grid_data = make_data(grid_df)

    for process_mode in rt_utils.PROCESS_MODES:
        terms = (
            rt_utils.npl_alpha_rt_only_condition_delay_loglike_terms(
                params,
                grid_data,
                process_mode,
                K_max=K_MAX,
            )
        )
        normalized_pdf = np.asarray(terms["normalized_pdf"])
        area = float(np.trapz(normalized_pdf, rt_grid_s))
        area_error = abs(area - 1.0)
        integration_rows.append(
            {
                "process_mode": process_mode,
                "t_stim_s": float(row["intended_fix"]),
                "ABL": float(row["ABL"]),
                "ILD": float(row["ILD"]),
                "normalized_area_0_to_1": area,
                "abs_error": area_error,
            }
        )

integration_df = pd.DataFrame(integration_rows)
print("\nNumerical 0-1 s normalization checks:")
print(integration_df.to_string(index=False))

if float(integration_df["abs_error"].max()) > INTEGRAL_ATOL:
    raise RuntimeError(
        "Numerical retained-window area check exceeded tolerance."
    )

print("\nAll RT-only likelihood validations passed.")

# %%
