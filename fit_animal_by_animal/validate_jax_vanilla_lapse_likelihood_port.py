# %%
"""
Validate the JAX vanilla/IPL+lapse likelihood port against the old NumPy functions.

This script compares per-trial normalized log likelihoods from:

    lapses_fit_single_animal.py's vanilla+lapse mixture around
    time_vary_norm_utils.up_or_down_RTs_fit_fn and
    time_vary_norm_utils.cum_pro_and_reactive_time_vary_fn

against the JAX port in `numpyro_vanilla_lapse_condition_delay_svi_utils.py`.
"""

# %%
# =============================================================================
# Editable parameters
# =============================================================================
from pathlib import Path
import os
import pickle
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

TEST_ANIMALS = [
    ("LED8", 105),
    ("LED34_even", 52),
]
MAX_ROWS_PER_ANIMAL = int(os.environ.get("VANILLA_LAPSE_JAX_VALIDATE_MAX_ROWS", "300"))
RNG_SEED = int(os.environ.get("VANILLA_LAPSE_JAX_VALIDATE_SEED", "0"))
K_MAX = int(os.environ.get("K_MAX", "10"))
LOG_TOL = float(os.environ.get("VANILLA_LAPSE_JAX_VALIDATE_LOG_TOL", "1e-6"))
TOTAL_LOG_TOL = float(os.environ.get("VANILLA_LAPSE_JAX_VALIDATE_TOTAL_LOG_TOL", "1e-4"))

ABLS = [20.0, 40.0, 60.0]
BATCH_T_TRUNC = {"LED34_even": 0.15}
DEFAULT_T_TRUNC = 0.3

CONDITION_T_E_AFF_CACHE = (
    REPO_DIR
    / "fit_each_condn"
    / "abl_specific_ild2_delay_agreement"
    / "condition_t_E_aff_extraction_cache.csv"
)


# %%
# =============================================================================
# Imports
# =============================================================================
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pandas as pd

sys.path.insert(0, str(SCRIPT_DIR))
from time_vary_norm_utils import cum_pro_and_reactive_time_vary_fn, up_or_down_RTs_fit_fn
import numpyro_vanilla_lapse_condition_delay_svi_utils as vanilla_lapse_utils


# %%
# =============================================================================
# Helpers
# =============================================================================
def ensure_choice_column(df):
    if "choice" not in df.columns:
        if "response_poke" not in df.columns:
            raise KeyError("Need either `choice` or `response_poke` in the batch CSV.")
        df = df.copy()
        df["choice"] = df["response_poke"].map({3: 1, 2: -1})
    return df


def load_abort_means(batch_name, animal):
    abort_pkl = REPO_DIR / "aborts_ipl_npl_time_fit_results" / f"results_{batch_name}_animal_{animal}.pkl"
    if not abort_pkl.exists():
        raise FileNotFoundError(abort_pkl)
    with abort_pkl.open("rb") as handle:
        saved = pickle.load(handle)
    abort = saved["vbmc_aborts_results"]
    return {
        "V_A": float(np.mean(abort["V_A_samples"])),
        "theta_A": float(np.mean(abort["theta_A_samples"])),
        "t_A_aff": float(np.mean(abort["t_A_aff_samp"])),
    }


def load_validation_rows(batch_name, animal):
    batch_csv = REPO_DIR / "raw_data" / "batch_csvs" / f"batch_{batch_name}_valid_and_aborts.csv"
    if not batch_csv.exists():
        raise FileNotFoundError(batch_csv)
    df = ensure_choice_column(pd.read_csv(batch_csv))
    valid_df = df[
        (df["animal"].astype(int) == int(animal))
        & df["success"].isin([1, -1])
        & (df["RTwrtStim"] < 1)
        & (df["ABL"].isin(ABLS))
    ].copy()
    valid_df = valid_df.dropna(subset=["TotalFixTime", "intended_fix", "ABL", "ILD", "choice", "RTwrtStim"])
    if len(valid_df) == 0:
        raise RuntimeError(f"No validation rows for {batch_name}/{animal}.")

    valid_df["ABL"] = valid_df["ABL"].astype(float)
    valid_df["ILD"] = valid_df["ILD"].astype(float)
    valid_df["choice"] = valid_df["choice"].astype(int)

    if len(valid_df) > MAX_ROWS_PER_ANIMAL:
        # Stratify lightly by condition before sampling so high-|ILD| rows are represented.
        valid_df = (
            valid_df.groupby(["ABL", "ILD"], group_keys=False)
            .sample(
                n=max(1, int(np.ceil(MAX_ROWS_PER_ANIMAL / valid_df[["ABL", "ILD"]].drop_duplicates().shape[0]))),
                replace=True,
                random_state=RNG_SEED,
            )
            .drop_duplicates()
            .head(MAX_ROWS_PER_ANIMAL)
            .sort_index()
            .copy()
        )

    condition_table = (
        valid_df[["ABL", "ILD"]]
        .drop_duplicates()
        .sort_values(["ABL", "ILD"])
        .reset_index(drop=True)
    )
    condition_table["condition_id"] = np.arange(len(condition_table), dtype=int)

    delay_cache = pd.read_csv(CONDITION_T_E_AFF_CACHE)
    animal_delay_df = delay_cache[
        (delay_cache["batch_name"].astype(str) == str(batch_name))
        & (delay_cache["animal"].astype(int) == int(animal))
    ][["ABL", "ILD", "t_E_aff_s"]].copy()
    animal_delay_df["ABL"] = animal_delay_df["ABL"].astype(float)
    animal_delay_df["ILD"] = animal_delay_df["ILD"].astype(float)

    condition_table = condition_table.merge(
        animal_delay_df,
        on=["ABL", "ILD"],
        how="left",
        validate="one_to_one",
    )
    if condition_table["t_E_aff_s"].isna().any():
        raise RuntimeError(f"Missing delay cache rows:\n{condition_table[condition_table['t_E_aff_s'].isna()]}")

    valid_df = valid_df.merge(condition_table[["ABL", "ILD", "condition_id"]], on=["ABL", "ILD"], how="left")
    return valid_df.reset_index(drop=True), condition_table


def make_jax_data(df, abort_params, T_trunc):
    return {
        "total_fix": jnp.asarray(df["TotalFixTime"].to_numpy(dtype=float)),
        "t_stim": jnp.asarray(df["intended_fix"].to_numpy(dtype=float)),
        "ABL": jnp.asarray(df["ABL"].to_numpy(dtype=float)),
        "ILD": jnp.asarray(df["ILD"].to_numpy(dtype=float)),
        "choice": jnp.asarray(df["choice"].to_numpy(dtype=int)),
        "condition_id": jnp.asarray(df["condition_id"].to_numpy(dtype=int)),
        "V_A": jnp.asarray(abort_params["V_A"], dtype=jnp.float64),
        "theta_A": jnp.asarray(abort_params["theta_A"], dtype=jnp.float64),
        "t_A_aff": jnp.asarray(abort_params["t_A_aff"], dtype=jnp.float64),
        "T_trunc": jnp.asarray(T_trunc, dtype=jnp.float64),
        "lapse_rt_window": jnp.asarray(1.0, dtype=jnp.float64),
    }


def old_numpy_log_terms(df, condition_table, abort_params, T_trunc, params):
    t_e_by_condition = condition_table.set_index("condition_id")["t_E_aff_s"].to_dict()
    z_e = (params["w"] - 0.5) * 2.0 * params["theta_E"]
    rows = []
    for row_idx, row in df.iterrows():
        t_E_aff = float(t_e_by_condition[int(row["condition_id"])])
        pdf = up_or_down_RTs_fit_fn(
            float(row["TotalFixTime"]),
            int(row["choice"]),
            abort_params["V_A"],
            abort_params["theta_A"],
            abort_params["t_A_aff"],
            float(row["intended_fix"]),
            float(row["ABL"]),
            float(row["ILD"]),
            params["rate_lambda"],
            params["T_0"],
            params["theta_E"],
            z_e,
            t_E_aff,
            params["del_go"],
            None,
            np.nan,
            False,
            False,
            K_MAX,
        )
        trunc_factor = cum_pro_and_reactive_time_vary_fn(
            float(row["intended_fix"]) + 1.0,
            T_trunc,
            abort_params["V_A"],
            abort_params["theta_A"],
            abort_params["t_A_aff"],
            float(row["intended_fix"]),
            float(row["ABL"]),
            float(row["ILD"]),
            params["rate_lambda"],
            params["T_0"],
            params["theta_E"],
            z_e,
            t_E_aff,
            None,
            np.nan,
            False,
            False,
            K_MAX,
        ) - cum_pro_and_reactive_time_vary_fn(
            float(row["intended_fix"]),
            T_trunc,
            abort_params["V_A"],
            abort_params["theta_A"],
            abort_params["t_A_aff"],
            float(row["intended_fix"]),
            float(row["ABL"]),
            float(row["ILD"]),
            params["rate_lambda"],
            params["T_0"],
            params["theta_E"],
            z_e,
            t_E_aff,
            None,
            np.nan,
            False,
            False,
            K_MAX,
        )
        normalized_pdf = max(pdf / (trunc_factor + 1e-20), 1e-50)
        if int(row["choice"]) == 1:
            lapse_choice_pdf = params["lapse_prob_right"] / 1.0
        else:
            lapse_choice_pdf = (1.0 - params["lapse_prob_right"]) / 1.0
        mixture_pdf = (1.0 - params["lapse_prob"]) * normalized_pdf + params["lapse_prob"] * lapse_choice_pdf
        log_pdf = np.log(max(mixture_pdf, 1e-50))
        rows.append(
            {
                "row_idx": int(row_idx),
                "pdf_numpy": float(pdf),
                "trunc_numpy": float(trunc_factor),
                "log_pdf_numpy": float(log_pdf),
            }
        )
    return pd.DataFrame(rows)


# %%
# =============================================================================
# Validation run
# =============================================================================
summary_rows = []

for batch_name, animal in TEST_ANIMALS:
    print("\n" + "=" * 80)
    print(f"Validating {batch_name}/{animal}")
    print("=" * 80)

    valid_df, condition_table = load_validation_rows(batch_name, animal)
    abort_params = load_abort_means(batch_name, animal)
    T_trunc = BATCH_T_TRUNC.get(batch_name, DEFAULT_T_TRUNC)

    init_values = dict(vanilla_lapse_utils.DEFAULT_INIT_VALUES)
    init_values["t_E_aff"] = condition_table["t_E_aff_s"].to_numpy(dtype=float)
    init_values = vanilla_lapse_utils.clip_init_to_hard_bounds(init_values)

    data = make_jax_data(valid_df, abort_params, T_trunc)
    old_df = old_numpy_log_terms(valid_df, condition_table, abort_params, T_trunc, init_values)
    pdf_jax, trunc_jax, log_pdf_jax = vanilla_lapse_utils.vanilla_condition_delay_loglike_terms(
        init_values,
        data,
        K_max=K_MAX,
    )

    compare_df = old_df.copy()
    compare_df["pdf_jax"] = np.asarray(jax.device_get(pdf_jax), dtype=float)
    compare_df["trunc_jax"] = np.asarray(jax.device_get(trunc_jax), dtype=float)
    compare_df["log_pdf_jax"] = np.asarray(jax.device_get(log_pdf_jax), dtype=float)
    compare_df["abs_log_diff"] = np.abs(compare_df["log_pdf_numpy"] - compare_df["log_pdf_jax"])
    compare_df["abs_pdf_diff"] = np.abs(compare_df["pdf_numpy"] - compare_df["pdf_jax"])
    compare_df["abs_trunc_diff"] = np.abs(compare_df["trunc_numpy"] - compare_df["trunc_jax"])

    max_abs_log = float(compare_df["abs_log_diff"].max())
    mean_abs_log = float(compare_df["abs_log_diff"].mean())
    total_numpy = float(compare_df["log_pdf_numpy"].sum())
    total_jax = float(compare_df["log_pdf_jax"].sum())
    total_abs_diff = abs(total_numpy - total_jax)

    log_joint = vanilla_lapse_utils.vanilla_condition_delay_loglike(init_values, data, K_max=K_MAX)
    grad = jax.grad(lambda values: vanilla_lapse_utils.vanilla_condition_delay_loglike(values, data, K_max=K_MAX))(
        init_values
    )
    gradients_finite = vanilla_lapse_utils.tree_all_finite(grad)

    print(f"Rows compared: {len(compare_df)}")
    print(f"Conditions compared: {len(condition_table)}")
    print(f"T_trunc: {T_trunc:.3f} s")
    print(f"Total loglike NumPy: {total_numpy:.9f}")
    print(f"Total loglike JAX:   {total_jax:.9f}")
    print(f"Total abs diff:      {total_abs_diff:.3e}")
    print(f"Max abs log diff:    {max_abs_log:.3e}")
    print(f"Mean abs log diff:   {mean_abs_log:.3e}")
    print(f"JAX gradients finite: {gradients_finite}")

    if not np.isfinite(float(log_joint)):
        raise RuntimeError(f"{batch_name}/{animal}: non-finite JAX loglike.")
    if not gradients_finite:
        raise RuntimeError(f"{batch_name}/{animal}: non-finite JAX gradients.")
    if max_abs_log > LOG_TOL:
        worst = compare_df.sort_values("abs_log_diff", ascending=False).head(5)
        raise RuntimeError(
            f"{batch_name}/{animal}: max per-row log diff {max_abs_log:.3e} > {LOG_TOL:.3e}\n"
            + worst.to_string(index=False)
        )
    if total_abs_diff > TOTAL_LOG_TOL:
        raise RuntimeError(
            f"{batch_name}/{animal}: total loglike diff {total_abs_diff:.3e} > {TOTAL_LOG_TOL:.3e}"
        )

    summary_rows.append(
        {
            "batch_name": batch_name,
            "animal": int(animal),
            "n_rows": int(len(compare_df)),
            "n_conditions": int(len(condition_table)),
            "T_trunc": float(T_trunc),
            "total_loglike_numpy": total_numpy,
            "total_loglike_jax": total_jax,
            "total_abs_diff": total_abs_diff,
            "max_abs_log_diff": max_abs_log,
            "mean_abs_log_diff": mean_abs_log,
            "gradients_finite": bool(gradients_finite),
        }
    )

summary_df = pd.DataFrame(summary_rows)
print("\nValidation passed:")
print(summary_df.to_string(index=False))

# %%
