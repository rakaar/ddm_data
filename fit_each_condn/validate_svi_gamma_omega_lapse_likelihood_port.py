# %%
"""
Check the JAX Gamma/Omega/delay+lapse likelihood against the legacy VBMC formula.

This is a formula-level validation. It does not require the old lapse condition
fit pickle folder; it uses real trial rows and fixed parameter values, then
compares the old NumPy/Python likelihood calls to the new vectorized JAX helper.
"""

# %%
from pathlib import Path
import pickle
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from led_off_gamma_omega_pdf_utils import (  # noqa: E402
    cum_pro_and_reactive_trunc_fn,
    up_or_down_RTs_fit_OPTIM_V_A_change_gamma_omega_with_w_fn,
)
import svi_gamma_omega_likelihood_utils as likelihood_utils  # noqa: E402


# %%
# =============================================================================
# Editable parameters
# =============================================================================
TEST_CASES = [
    {"batch_name": "LED8", "animal": 105, "ABL": 40, "ILD": 8},
    {"batch_name": "LED34_even", "animal": 52, "ABL": 40, "ILD": -4},
]
N_ROWS_PER_CASE = 250
K_MAX = 10

BATCH_T_TRUNC = {"LED34_even": 0.15}
DEFAULT_T_TRUNC = 0.3


# %%
def old_lapse_loglike(row, params, abort_params, batch_name, total_fix_col):
    c_A_trunc_time = BATCH_T_TRUNC.get(batch_name, DEFAULT_T_TRUNC)
    rt = float(row[total_fix_col])
    t_stim = float(row["intended_fix"])
    choice = int(row["choice"])

    trunc_factor = cum_pro_and_reactive_trunc_fn(
        t_stim + 1.0,
        c_A_trunc_time,
        abort_params["V_A"],
        abort_params["theta_A"],
        abort_params["t_A_aff"],
        t_stim,
        params["t_E_aff"],
        params["gamma"],
        params["omega"],
        params["w"],
        K_MAX,
    ) - cum_pro_and_reactive_trunc_fn(
        t_stim,
        c_A_trunc_time,
        abort_params["V_A"],
        abort_params["theta_A"],
        abort_params["t_A_aff"],
        t_stim,
        params["t_E_aff"],
        params["gamma"],
        params["omega"],
        params["w"],
        K_MAX,
    )

    joint_pdf = up_or_down_RTs_fit_OPTIM_V_A_change_gamma_omega_with_w_fn(
        rt,
        abort_params["V_A"],
        abort_params["theta_A"],
        params["gamma"],
        params["omega"],
        t_stim,
        abort_params["t_A_aff"],
        params["t_E_aff"],
        params["del_go"],
        choice,
        params["w"],
        K_MAX,
    )
    ddm_pdf = max(joint_pdf / (trunc_factor + 1e-10), 1e-10)
    lapse_choice_pdf = params["lapse_prob_right"] if choice == 1 else 1.0 - params["lapse_prob_right"]
    mixed_pdf = (1.0 - params["lapse_prob"]) * ddm_pdf + params["lapse_prob"] * lapse_choice_pdf
    return np.log(max(mixed_pdf, 1e-50))


# %%
rows = []
max_abs_diff = 0.0
for test_case in TEST_CASES:
    batch_name = test_case["batch_name"]
    animal = int(test_case["animal"])
    batch_csv = REPO_DIR / "raw_data" / "batch_csvs" / f"batch_{batch_name}_valid_and_aborts.csv"
    abort_pkl = REPO_DIR / "aborts_ipl_npl_time_fit_results" / f"results_{batch_name}_animal_{animal}.pkl"
    no_lapse_dir = (
        SCRIPT_DIR
        / "svi_big_gamma_omega_delay_patience12_restore_best_all_animals_outputs"
        / f"{batch_name}_{animal}"
    )
    no_lapse_prefix = f"{batch_name}_{animal}_big_gamma_omega_delay"
    no_lapse_summary = no_lapse_dir / f"{no_lapse_prefix}_condition_summary.csv"
    no_lapse_posterior = no_lapse_dir / f"{no_lapse_prefix}_posterior_summary.csv"

    raw_df = pd.read_csv(batch_csv)
    if "choice" not in raw_df.columns:
        raw_df["choice"] = raw_df["response_poke"].map({3: 1, 2: -1})
    total_fix_col = "timed_fix" if "timed_fix" in raw_df.columns else "TotalFixTime"
    valid_df = raw_df[
        (raw_df["animal"].astype(int) == animal)
        & (raw_df["success"].isin([1, -1]))
        & (raw_df["RTwrtStim"] > 0)
        & (raw_df["RTwrtStim"] <= 1)
        & (raw_df["ABL"].astype(int) == int(test_case["ABL"]))
        & (raw_df["ILD"].astype(int) == int(test_case["ILD"]))
    ].copy()
    valid_df = valid_df.dropna(subset=[total_fix_col, "intended_fix", "choice"]).head(N_ROWS_PER_CASE)
    valid_df["choice"] = valid_df["choice"].astype(int)
    if len(valid_df) == 0:
        raise RuntimeError(f"No validation rows for {test_case}")

    with abort_pkl.open("rb") as handle:
        abort_result = pickle.load(handle)
    abort_samples = abort_result["vbmc_aborts_results"]
    abort_params = {
        "V_A": float(np.mean(abort_samples["V_A_samples"])),
        "theta_A": float(np.mean(abort_samples["theta_A_samples"])),
        "t_A_aff": float(np.mean(abort_samples["t_A_aff_samp"])),
    }

    condition_summary = pd.read_csv(no_lapse_summary)
    condition_match = condition_summary[
        (condition_summary["ABL"].astype(int) == int(test_case["ABL"]))
        & (condition_summary["ILD"].astype(int) == int(test_case["ILD"]))
    ]
    if len(condition_match) != 1:
        raise RuntimeError(f"Expected one no-lapse condition row for {test_case}, got {len(condition_match)}")

    posterior_summary = pd.read_csv(no_lapse_posterior)
    scalar_means = {
        parameter: float(posterior_summary.loc[posterior_summary["parameter"] == parameter, "mean"].iloc[0])
        for parameter in ["w", "del_go"]
    }
    params = {
        "gamma": float(condition_match.iloc[0]["gamma_mean"]),
        "omega": float(condition_match.iloc[0]["omega_mean"]),
        "t_E_aff": float(condition_match.iloc[0]["t_E_aff_mean"]),
        "w": scalar_means["w"],
        "del_go": scalar_means["del_go"],
        "lapse_prob": 0.025,
        "lapse_prob_right": 0.52,
    }

    old_total = float(
        np.sum(
            [
                old_lapse_loglike(row, params, abort_params, batch_name, total_fix_col)
                for _, row in valid_df.iterrows()
            ]
        )
    )
    jax_data = {
        "total_fix": jnp.asarray(valid_df[total_fix_col].to_numpy(dtype=float)),
        "t_stim": jnp.asarray(valid_df["intended_fix"].to_numpy(dtype=float)),
        "choice": jnp.asarray(valid_df["choice"].to_numpy(dtype=int)),
        "condition_id": jnp.zeros(len(valid_df), dtype=int),
        "mask": jnp.ones(len(valid_df), dtype=bool),
        "V_A": jnp.asarray(abort_params["V_A"], dtype=jnp.float64),
        "theta_A": jnp.asarray(abort_params["theta_A"], dtype=jnp.float64),
        "t_A_aff": jnp.asarray(abort_params["t_A_aff"], dtype=jnp.float64),
        "T_trunc": jnp.asarray(BATCH_T_TRUNC.get(batch_name, DEFAULT_T_TRUNC), dtype=jnp.float64),
    }
    jax_params = {
        "gamma": jnp.asarray([params["gamma"]], dtype=jnp.float64),
        "omega": jnp.asarray([params["omega"]], dtype=jnp.float64),
        "t_E_aff": jnp.asarray([params["t_E_aff"]], dtype=jnp.float64),
        "w": jnp.asarray(params["w"], dtype=jnp.float64),
        "del_go": jnp.asarray(params["del_go"], dtype=jnp.float64),
        "lapse_prob": jnp.asarray(params["lapse_prob"], dtype=jnp.float64),
        "lapse_prob_right": jnp.asarray(params["lapse_prob_right"], dtype=jnp.float64),
    }
    jax_total = float(likelihood_utils.gamma_omega_delay_lapse_loglike(jax_params, jax_data, K_MAX))
    abs_diff = abs(old_total - jax_total)
    max_abs_diff = max(max_abs_diff, abs_diff)
    rows.append(
        {
            **test_case,
            "n_rows": len(valid_df),
            "old_loglike": old_total,
            "jax_loglike": jax_total,
            "abs_diff": abs_diff,
        }
    )

summary_df = pd.DataFrame(rows)
print(summary_df.to_string(index=False))
print(f"\nMax abs diff: {max_abs_diff:.6g}")
if max_abs_diff > 1e-5:
    raise RuntimeError(f"JAX lapse likelihood port differs from old formula; max abs diff={max_abs_diff:.6g}")

# %%
