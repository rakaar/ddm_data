# %%
"""
Fit-aligned LED7 valid-trial RTDs from the patience-12 NPL+alpha SVI fit.

This regenerates ABL-collapsed views from the exact valid trials used by the
fit:
1. The complete 0--1 s RTD.
2. Conditional RTDs truncated at 115, 130, 150, and 170 ms.

Unlike the older VBMC diagnostic, the theoretical curve uses every fitting
trial's own intended-fix time, animal-specific proactive parameters, signed
ABL/ILD condition delay, and SVI truncation denominator before averaging.
"""

# %%
# =============================================================================
# Editable parameters
# =============================================================================
from pathlib import Path
import os
import pickle
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
FIT_UTILS_DIR = REPO_ROOT / "fit_animal_by_animal"

BATCH_NAME = "LED7"
ANIMALS = (92, 93, 98, 99, 100, 103)
ABLS = (20, 40, 60)
FIT_RT_MAX_S = 1.0
TRUNCATION_MS = (115, 130, 150, 170)
TRUNCATION_S = {cutoff_ms: cutoff_ms / 1e3 for cutoff_ms in TRUNCATION_MS}
MODEL_STEP_S = 0.001
FULL_DATA_BIN_S = 0.020
TRUNCATED_DATA_BIN_S = 0.005
T_TRUNC_S = 0.300
K_MAX = 10
TRIAL_CHUNK_SIZE = 512
PLOT_DPI = 300
THEORY_ALPHA = 0.5

DATA_CSV = REPO_ROOT / "raw_data" / "batch_csvs" / "batch_LED7_valid_and_aborts.csv"
OUT_LED_CSV = REPO_ROOT / "out_LED.csv"
FIT_ROOT = (
    REPO_ROOT
    / "fit_animal_by_animal"
    / "numpyro_svi_npl_alpha_condition_delay_patience12_restore_best_outputs"
)
ABORT_ROOT = REPO_ROOT / "aborts_ipl_npl_time_fit_results"
OUTPUT_DIR = SCRIPT_DIR / "led7_npl_alpha_patience12_fit_aligned_valid_rtds"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COMBINED_PNG = OUTPUT_DIR / "led7_npl_alpha_patience12_fit_aligned_valid_rtds_complete_and_115ms.png"
COMBINED_PDF = COMBINED_PNG.with_suffix(".pdf")
FULL_PNG = OUTPUT_DIR / "led7_npl_alpha_patience12_fit_aligned_valid_rtd_complete.png"
FULL_PDF = FULL_PNG.with_suffix(".pdf")
ADDITIONAL_TRUNCATIONS_PNG = (
    OUTPUT_DIR / "led7_npl_alpha_patience12_fit_aligned_valid_rtds_conditional_130_150_170ms.png"
)
ADDITIONAL_TRUNCATIONS_PDF = ADDITIONAL_TRUNCATIONS_PNG.with_suffix(".pdf")
TRUNCATED_PNG_BY_MS = {
    cutoff_ms: OUTPUT_DIR
    / f"led7_npl_alpha_patience12_fit_aligned_valid_rtd_{cutoff_ms}ms.png"
    for cutoff_ms in TRUNCATION_MS
}
TRUNCATED_PDF_BY_MS = {
    cutoff_ms: path.with_suffix(".pdf") for cutoff_ms, path in TRUNCATED_PNG_BY_MS.items()
}
EQUAL_ANIMAL_COMBINED_PNG = (
    OUTPUT_DIR
    / "led7_npl_alpha_patience12_fit_aligned_valid_rtds_complete_and_115ms_equal_animal_average.png"
)
EQUAL_ANIMAL_COMBINED_PDF = EQUAL_ANIMAL_COMBINED_PNG.with_suffix(".pdf")
EQUAL_ANIMAL_FULL_PNG = (
    OUTPUT_DIR / "led7_npl_alpha_patience12_fit_aligned_valid_rtd_complete_equal_animal_average.png"
)
EQUAL_ANIMAL_FULL_PDF = EQUAL_ANIMAL_FULL_PNG.with_suffix(".pdf")
EQUAL_ANIMAL_ADDITIONAL_TRUNCATIONS_PNG = (
    OUTPUT_DIR
    / "led7_npl_alpha_patience12_fit_aligned_valid_rtds_conditional_130_150_170ms_equal_animal_average.png"
)
EQUAL_ANIMAL_ADDITIONAL_TRUNCATIONS_PDF = (
    EQUAL_ANIMAL_ADDITIONAL_TRUNCATIONS_PNG.with_suffix(".pdf")
)
EQUAL_ANIMAL_TRUNCATED_PNG_BY_MS = {
    cutoff_ms: OUTPUT_DIR
    / f"led7_npl_alpha_patience12_fit_aligned_valid_rtd_{cutoff_ms}ms_equal_animal_average.png"
    for cutoff_ms in TRUNCATION_MS
}
EQUAL_ANIMAL_TRUNCATED_PDF_BY_MS = {
    cutoff_ms: path.with_suffix(".pdf")
    for cutoff_ms, path in EQUAL_ANIMAL_TRUNCATED_PNG_BY_MS.items()
}
SUMMARY_CSV = OUTPUT_DIR / "led7_npl_alpha_patience12_fit_aligned_valid_rtd_summary.csv"
DATA_AUDIT_CSV = OUTPUT_DIR / "led7_npl_alpha_patience12_fit_aligned_data_audit.csv"
ILD_COUNT_AUDIT_CSV = OUTPUT_DIR / "led7_npl_alpha_patience12_signed_ild_count_audit.csv"
PAYLOAD_NPZ = OUTPUT_DIR / "led7_npl_alpha_patience12_fit_aligned_valid_rtd_payload.npz"

ABL_COLORS = {20: "#0072B2", 40: "#E69F00", 60: "#009E73"}


# %%
# =============================================================================
# Imports and exact likelihood helpers
# =============================================================================
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

if str(FIT_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(FIT_UTILS_DIR))
import numpyro_npl_alpha_svi_utils as svi_utils


# %%
# =============================================================================
# Load the exact SVI fitting rows and verify them against the newly added CSV
# =============================================================================
print(f"Data CSV: {DATA_CSV}")
print(f"New out_LED CSV: {OUT_LED_CSV}")
print(f"SVI fit root: {FIT_ROOT}")
print(f"Abort/proactive root: {ABORT_ROOT}")
print(f"Output directory: {OUTPUT_DIR}")

for required_path in [DATA_CSV, OUT_LED_CSV, FIT_ROOT, ABORT_ROOT]:
    if not required_path.exists():
        raise FileNotFoundError(required_path)

batch_df = pd.read_csv(DATA_CSV)
required_columns = [
    "animal",
    "success",
    "RTwrtStim",
    "TotalFixTime",
    "intended_fix",
    "ABL",
    "ILD",
    "choice",
]
missing_columns = [column for column in required_columns if column not in batch_df.columns]
if missing_columns:
    raise KeyError(f"Missing columns in {DATA_CSV}: {missing_columns}")

valid_df = batch_df[
    batch_df["animal"].astype(int).isin(ANIMALS)
    & batch_df["success"].isin([1, -1])
    & (batch_df["RTwrtStim"] < FIT_RT_MAX_S)
    & batch_df["ABL"].isin(ABLS)
].copy()
valid_df = valid_df.dropna(subset=required_columns)
valid_df["animal"] = valid_df["animal"].astype(int)
valid_df["ABL"] = valid_df["ABL"].astype(float)
valid_df["ILD"] = valid_df["ILD"].astype(float)
valid_df["choice"] = valid_df["choice"].astype(int)

if not np.allclose(
    valid_df["RTwrtStim"].to_numpy(dtype=float),
    valid_df["TotalFixTime"].to_numpy(dtype=float)
    - valid_df["intended_fix"].to_numpy(dtype=float),
    atol=2e-6,
    rtol=0,
):
    raise RuntimeError("RTwrtStim is not consistent with TotalFixTime - intended_fix.")
if (valid_df["RTwrtStim"] < 0).any():
    raise RuntimeError("The current LED7 fit rows unexpectedly include negative RTwrtStim values.")

# Rebuild the fitting-row multiset directly from raw out_LED.csv using the same
# preprocessing as fit_animal_by_animal/save_valid_and_aborts_batches.py.
raw_columns = [
    "animal",
    "success",
    "timed_fix",
    "intended_fix",
    "ABL",
    "ILD",
    "response_poke",
    "abort_event",
    "LED_trial",
    "session_type",
    "training_level",
    "repeat_trial",
]
out_led_df = pd.read_csv(OUT_LED_CSV, usecols=raw_columns)
out_led_df["RTwrtStim"] = out_led_df["timed_fix"] - out_led_df["intended_fix"]
out_led_df = out_led_df.rename(columns={"timed_fix": "TotalFixTime"})
missing_response = out_led_df["response_poke"].isna()
out_led_df.loc[
    missing_response & (out_led_df["success"] == 1) & (out_led_df["ILD"] > 0),
    "response_poke",
] = 3
out_led_df.loc[
    missing_response & (out_led_df["success"] == 1) & (out_led_df["ILD"] < 0),
    "response_poke",
] = 2
out_led_df.loc[
    missing_response & (out_led_df["success"] == -1) & (out_led_df["ILD"] > 0),
    "response_poke",
] = 2
out_led_df.loc[
    missing_response & (out_led_df["success"] == -1) & (out_led_df["ILD"] < 0),
    "response_poke",
] = 3
out_led_df["choice"] = out_led_df["response_poke"].map({3: 1, 2: -1})
out_led_df = out_led_df[
    (out_led_df["LED_trial"].isna() | (out_led_df["LED_trial"] == 0))
    & (out_led_df["session_type"] == 7)
    & (out_led_df["training_level"] == 16)
    & (out_led_df["repeat_trial"].isna() | out_led_df["repeat_trial"].isin([0, 2]))
].copy()
out_led_valid = out_led_df[
    out_led_df["animal"].astype(int).isin(ANIMALS)
    & out_led_df["success"].isin([1, -1])
    & (out_led_df["RTwrtStim"] < FIT_RT_MAX_S)
    & out_led_df["ABL"].isin(ABLS)
].copy()
out_led_valid = out_led_valid.dropna(subset=required_columns)

comparison_columns = [
    "animal",
    "success",
    "RTwrtStim",
    "TotalFixTime",
    "intended_fix",
    "ABL",
    "ILD",
    "choice",
]
batch_rows = valid_df[comparison_columns].sort_values(comparison_columns).reset_index(drop=True)
out_led_rows = out_led_valid[comparison_columns].sort_values(comparison_columns).reset_index(drop=True)
if len(batch_rows) != len(out_led_rows) or not np.allclose(
    batch_rows.to_numpy(dtype=float),
    out_led_rows.to_numpy(dtype=float),
    equal_nan=True,
    atol=1e-12,
    rtol=0,
):
    raise RuntimeError(
        "The fitting rows reconstructed from out_LED.csv do not exactly match "
        "raw_data/batch_csvs/batch_LED7_valid_and_aborts.csv."
    )

expected_total_rows = 52799
expected_early_rows = 7222
truncated_df_by_ms = {
    cutoff_ms: valid_df[valid_df["RTwrtStim"] <= cutoff_s].copy()
    for cutoff_ms, cutoff_s in TRUNCATION_S.items()
}
if len(valid_df) != expected_total_rows or len(truncated_df_by_ms[115]) != expected_early_rows:
    raise RuntimeError(
        "LED7 fitting-row counts changed: "
        f"full={len(valid_df)} (expected {expected_total_rows}), "
        f"0--115 ms={len(truncated_df_by_ms[115])} (expected {expected_early_rows})."
    )

data_audit_rows = []
for animal in ANIMALS:
    animal_df = valid_df[valid_df["animal"] == animal]
    data_audit_rows.append(
        {
            "animal": animal,
            "n_fit_rows": len(animal_df),
            "min_rtwrtstim_s": animal_df["RTwrtStim"].min(),
            "max_rtwrtstim_s": animal_df["RTwrtStim"].max(),
            **{
                f"n_0_to_{cutoff_ms}ms": int(
                    (animal_df["RTwrtStim"] <= cutoff_s).sum()
                )
                for cutoff_ms, cutoff_s in TRUNCATION_S.items()
            },
        }
    )
pd.DataFrame(data_audit_rows).to_csv(DATA_AUDIT_CSV, index=False)

ild_count_audit = (
    valid_df.groupby(["animal", "ABL", "ILD"])
    .size()
    .rename("n_trials")
    .reset_index()
)
pooled_ild_count_audit = (
    valid_df.groupby(["ABL", "ILD"])
    .size()
    .rename("n_trials")
    .reset_index()
)
pooled_ild_count_audit.insert(0, "animal", "pooled")
ild_count_audit["animal"] = ild_count_audit["animal"].astype(str)
pd.concat([pooled_ild_count_audit, ild_count_audit], ignore_index=True).to_csv(
    ILD_COUNT_AUDIT_CSV,
    index=False,
)

print(f"Exact fitting rows: {len(valid_df):,}")
print("Counts by ABL:")
print(valid_df.groupby("ABL").size().astype(int).to_string())
for cutoff_ms in TRUNCATION_MS:
    print(f"0--{cutoff_ms} ms rows: {len(truncated_df_by_ms[cutoff_ms]):,}")
    print(truncated_df_by_ms[cutoff_ms].groupby("ABL").size().astype(int).to_string())
print("out_LED.csv and the processed SVI batch CSV yield the same fitting-row multiset.")


# %%
# =============================================================================
# Exact fit-aligned theoretical RTDs
# =============================================================================
rt_grid_s = np.arange(0.0, FIT_RT_MAX_S + MODEL_STEP_S / 2.0, MODEL_STEP_S)
truncation_grid_masks = {
    cutoff_ms: rt_grid_s <= cutoff_s + 1e-12
    for cutoff_ms, cutoff_s in TRUNCATION_S.items()
}
model_sum_by_abl = {abl: np.zeros_like(rt_grid_s) for abl in ABLS}
model_n_by_abl = {abl: 0 for abl in ABLS}
model_sum_by_animal_abl = {
    (animal, abl): np.zeros_like(rt_grid_s) for animal in ANIMALS for abl in ABLS
}
model_n_by_animal_abl = {(animal, abl): 0 for animal in ANIMALS for abl in ABLS}
normalization_denominators = []
direct_formula_checks = []

for animal in ANIMALS:
    animal_df = valid_df[valid_df["animal"] == animal].copy()
    fit_dir = FIT_ROOT / f"{BATCH_NAME}_{animal}"
    posterior_path = fit_dir / "main_fullrank_posterior_samples.npz"
    condition_path = fit_dir / "condition_table.csv"
    abort_path = ABORT_ROOT / f"results_{BATCH_NAME}_animal_{animal}.pkl"
    for required_path in [posterior_path, condition_path, abort_path]:
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    posterior = np.load(posterior_path)
    required_posterior_keys = [
        "rate_lambda",
        "T_0",
        "theta_E",
        "w",
        "del_go",
        "rate_norm_l",
        "alpha",
        "t_E_aff",
    ]
    missing_posterior_keys = [key for key in required_posterior_keys if key not in posterior.files]
    if missing_posterior_keys:
        raise KeyError(f"Missing posterior keys for LED7/{animal}: {missing_posterior_keys}")
    if any(not np.isfinite(np.asarray(posterior[key], dtype=float)).all() for key in required_posterior_keys):
        raise RuntimeError(f"Non-finite posterior samples for LED7/{animal}.")

    params = {
        key: float(np.mean(np.asarray(posterior[key], dtype=float)))
        for key in required_posterior_keys
        if key != "t_E_aff"
    }
    delay_means = np.mean(np.asarray(posterior["t_E_aff"], dtype=float), axis=0)

    saved_conditions = pd.read_csv(condition_path).sort_values("condition_id").reset_index(drop=True)
    reconstructed_conditions = (
        animal_df[["ABL", "ILD"]]
        .drop_duplicates()
        .sort_values(["ABL", "ILD"])
        .reset_index(drop=True)
    )
    reconstructed_conditions["condition_id"] = np.arange(len(reconstructed_conditions), dtype=int)
    if len(saved_conditions) != len(reconstructed_conditions) or not np.allclose(
        saved_conditions[["ABL", "ILD", "condition_id"]].to_numpy(dtype=float),
        reconstructed_conditions[["ABL", "ILD", "condition_id"]].to_numpy(dtype=float),
        atol=1e-12,
        rtol=0,
    ):
        raise RuntimeError(f"Saved and reconstructed condition tables differ for LED7/{animal}.")
    if len(delay_means) != len(saved_conditions):
        raise RuntimeError(f"Delay vector length does not match condition table for LED7/{animal}.")

    with abort_path.open("rb") as f:
        abort_fit = pickle.load(f)["vbmc_aborts_results"]
    V_A = float(np.mean(np.asarray(abort_fit["V_A_samples"], dtype=float)))
    theta_A = float(np.mean(np.asarray(abort_fit["theta_A_samples"], dtype=float)))
    t_A_aff = float(np.mean(np.asarray(abort_fit["t_A_aff_samp"], dtype=float)))

    animal_df = animal_df.merge(
        reconstructed_conditions,
        on=["ABL", "ILD"],
        how="left",
        validate="many_to_one",
    )
    if animal_df["condition_id"].isna().any():
        raise RuntimeError(f"Failed to assign condition IDs for LED7/{animal}.")

    condition_abls = saved_conditions["ABL"].to_numpy(dtype=float)
    condition_ilds = saved_conditions["ILD"].to_numpy(dtype=float)
    Z_E = (params["w"] - 0.5) * 2.0 * params["theta_E"]

    # Evidence terms depend only on animal, signed ABL/ILD condition, and RT
    # relative to stimulus. They are evaluated from the exact SVI utilities.
    relative_evidence_time = jnp.asarray(
        rt_grid_s[None, :] - delay_means[:, None], dtype=jnp.float64
    )
    relative_go_time = relative_evidence_time + params["del_go"]
    abl_jax = jnp.asarray(condition_abls[:, None], dtype=jnp.float64)
    ild_jax = jnp.asarray(condition_ilds[:, None], dtype=jnp.float64)

    c_e_t1 = np.asarray(
        svi_utils.CDF_E_alpha_jax(
            relative_evidence_time,
            1,
            abl_jax,
            ild_jax,
            params["rate_lambda"],
            params["T_0"],
            params["theta_E"],
            Z_E,
            params["rate_norm_l"],
            params["alpha"],
            K_MAX,
        )
        + svi_utils.CDF_E_alpha_jax(
            relative_evidence_time,
            -1,
            abl_jax,
            ild_jax,
            params["rate_lambda"],
            params["T_0"],
            params["theta_E"],
            Z_E,
            params["rate_norm_l"],
            params["alpha"],
            K_MAX,
        )
    )
    c_e_t2 = np.asarray(
        svi_utils.CDF_E_alpha_jax(
            relative_go_time,
            1,
            abl_jax,
            ild_jax,
            params["rate_lambda"],
            params["T_0"],
            params["theta_E"],
            Z_E,
            params["rate_norm_l"],
            params["alpha"],
            K_MAX,
        )
        + svi_utils.CDF_E_alpha_jax(
            relative_go_time,
            -1,
            abl_jax,
            ild_jax,
            params["rate_lambda"],
            params["T_0"],
            params["theta_E"],
            Z_E,
            params["rate_norm_l"],
            params["alpha"],
            K_MAX,
        )
    )
    rho_e = np.asarray(
        svi_utils.rho_E_alpha_jax(
            relative_evidence_time,
            1,
            abl_jax,
            ild_jax,
            params["rate_lambda"],
            params["T_0"],
            params["theta_E"],
            Z_E,
            params["rate_norm_l"],
            params["alpha"],
            K_MAX,
        )
        + svi_utils.rho_E_alpha_jax(
            relative_evidence_time,
            -1,
            abl_jax,
            ild_jax,
            params["rate_lambda"],
            params["T_0"],
            params["theta_E"],
            Z_E,
            params["rate_norm_l"],
            params["alpha"],
            K_MAX,
        )
    )
    proactive_multiplier = (1.0 - c_e_t2) + (c_e_t2 - c_e_t1)

    # The SVI likelihood denominator uses the combined proactive/evidence CDF
    # at RT=0 and RT=1 for each trial's intended-fix time.
    c_e_lower = c_e_t1[:, 0]
    c_e_upper = c_e_t1[:, -1]

    for start in range(0, len(animal_df), TRIAL_CHUNK_SIZE):
        chunk = animal_df.iloc[start : start + TRIAL_CHUNK_SIZE]
        condition_ids = chunk["condition_id"].to_numpy(dtype=int)
        t_stim = chunk["intended_fix"].to_numpy(dtype=float)
        trial_abls = chunk["ABL"].to_numpy(dtype=int)

        absolute_t = t_stim[:, None] + rt_grid_s[None, :]
        p_a = np.asarray(
            svi_utils.rho_A_t_jax(jnp.asarray(absolute_t - t_A_aff), V_A, theta_A)
        )
        c_a = np.asarray(
            svi_utils.cum_A_t_jax(jnp.asarray(absolute_t - t_A_aff), V_A, theta_A)
        )

        c_a_lower_raw = np.asarray(
            svi_utils.cum_A_t_jax(jnp.asarray(t_stim - t_A_aff), V_A, theta_A)
        )
        c_a_upper_raw = np.asarray(
            svi_utils.cum_A_t_jax(
                jnp.asarray(t_stim + FIT_RT_MAX_S - t_A_aff), V_A, theta_A
            )
        )
        proactive_trunc_denom = float(
            1.0 - svi_utils.cum_A_t_jax(T_TRUNC_S - t_A_aff, V_A, theta_A)
        )
        c_a_lower = np.where(
            t_stim < T_TRUNC_S,
            0.0,
            c_a_lower_raw / proactive_trunc_denom,
        )
        c_a_upper = np.where(
            t_stim + FIT_RT_MAX_S < T_TRUNC_S,
            0.0,
            c_a_upper_raw / proactive_trunc_denom,
        )

        c_e_lo = c_e_lower[condition_ids]
        c_e_hi = c_e_upper[condition_ids]
        combined_lower = c_a_lower + c_e_lo - c_a_lower * c_e_lo
        combined_upper = c_a_upper + c_e_hi - c_a_upper * c_e_hi
        trunc_denom = combined_upper - combined_lower
        if not np.isfinite(trunc_denom).all() or np.any(trunc_denom <= 0):
            raise RuntimeError(f"Invalid SVI truncation denominator for LED7/{animal}.")
        normalization_denominators.extend(trunc_denom.tolist())

        numerator = (
            p_a * proactive_multiplier[condition_ids]
            + rho_e[condition_ids] * (1.0 - c_a)
        )
        normalized_density = numerator / trunc_denom[:, None]
        if not np.isfinite(normalized_density).all():
            raise RuntimeError(f"Non-finite model density for LED7/{animal}.")

        for abl in ABLS:
            abl_mask = trial_abls == abl
            if np.any(abl_mask):
                abl_density_sum = normalized_density[abl_mask].sum(axis=0)
                abl_trial_count = int(abl_mask.sum())
                model_sum_by_abl[abl] += abl_density_sum
                model_n_by_abl[abl] += abl_trial_count
                model_sum_by_animal_abl[(animal, abl)] += abl_density_sum
                model_n_by_animal_abl[(animal, abl)] += abl_trial_count

    # One full-grid check per animal verifies the algebraic choice collapse
    # against the exact high-level SVI helpers.
    check_trial = animal_df.iloc[0]
    check_condition_id = int(check_trial["condition_id"])
    check_t_stim = float(check_trial["intended_fix"])
    direct_up = np.asarray(
        svi_utils.up_or_down_alpha_jax(
            jnp.asarray(check_t_stim + rt_grid_s),
            1,
            V_A,
            theta_A,
            t_A_aff,
            check_t_stim,
            float(check_trial["ABL"]),
            float(check_trial["ILD"]),
            params["rate_lambda"],
            params["T_0"],
            params["theta_E"],
            Z_E,
            delay_means[check_condition_id],
            params["del_go"],
            params["rate_norm_l"],
            params["alpha"],
            K_MAX,
        )
    )
    direct_down = np.asarray(
        svi_utils.up_or_down_alpha_jax(
            jnp.asarray(check_t_stim + rt_grid_s),
            -1,
            V_A,
            theta_A,
            t_A_aff,
            check_t_stim,
            float(check_trial["ABL"]),
            float(check_trial["ILD"]),
            params["rate_lambda"],
            params["T_0"],
            params["theta_E"],
            Z_E,
            delay_means[check_condition_id],
            params["del_go"],
            params["rate_norm_l"],
            params["alpha"],
            K_MAX,
        )
    )
    check_proactive_time = check_t_stim + rt_grid_s - t_A_aff
    check_p_a_jax = np.asarray(svi_utils.rho_A_t_jax(check_proactive_time, V_A, theta_A))
    check_c_a_jax = np.asarray(svi_utils.cum_A_t_jax(check_proactive_time, V_A, theta_A))
    collapsed_numerator = (
        check_p_a_jax * proactive_multiplier[check_condition_id]
        + rho_e[check_condition_id] * (1.0 - check_c_a_jax)
    )
    max_abs_difference = float(np.max(np.abs(direct_up + direct_down - collapsed_numerator)))
    scale = max(1.0, float(np.max(np.abs(direct_up + direct_down))))
    direct_formula_checks.append(max_abs_difference / scale)
    if direct_formula_checks[-1] > 1e-10:
        raise RuntimeError(
            f"Choice-collapsed formula check failed for LED7/{animal}: "
            f"relative max difference={direct_formula_checks[-1]:.3g}."
        )

    print(
        f"LED7/{animal}: {len(animal_df):,} trials, "
        f"{len(saved_conditions)} signed conditions, formula check "
        f"{direct_formula_checks[-1]:.2e}"
    )

if any(model_n_by_abl[abl] != int((valid_df["ABL"] == abl).sum()) for abl in ABLS):
    raise RuntimeError("Model aggregation counts do not match fitting-row counts by ABL.")
if any(
    model_n_by_animal_abl[(animal, abl)]
    != int(((valid_df["animal"] == animal) & (valid_df["ABL"] == abl)).sum())
    for animal in ANIMALS
    for abl in ABLS
):
    raise RuntimeError("Model aggregation counts do not match fitting rows by animal and ABL.")


# %%
# =============================================================================
# Normalize for display, build empirical histograms, and save payloads
# =============================================================================
full_bins_s = np.arange(0.0, FIT_RT_MAX_S + FULL_DATA_BIN_S / 2.0, FULL_DATA_BIN_S)
full_bin_centers_s = 0.5 * (full_bins_s[:-1] + full_bins_s[1:])
truncated_bins_by_ms = {
    cutoff_ms: np.arange(
        0.0,
        cutoff_s + TRUNCATED_DATA_BIN_S / 2.0,
        TRUNCATED_DATA_BIN_S,
    )
    for cutoff_ms, cutoff_s in TRUNCATION_S.items()
}
truncated_bin_centers_by_ms = {
    cutoff_ms: 0.5 * (bins_s[:-1] + bins_s[1:])
    for cutoff_ms, bins_s in truncated_bins_by_ms.items()
}

model_raw_by_abl = {}
model_full_by_abl = {}
data_full_by_abl = {}
model_truncated_by_ms_abl = {cutoff_ms: {} for cutoff_ms in TRUNCATION_MS}
data_truncated_by_ms_abl = {cutoff_ms: {} for cutoff_ms in TRUNCATION_MS}
summary_rows = []

for abl in ABLS:
    abl_df = valid_df[valid_df["ABL"] == abl]
    data_full, _ = np.histogram(abl_df["RTwrtStim"], bins=full_bins_s, density=True)

    model_raw = model_sum_by_abl[abl] / model_n_by_abl[abl]
    model_raw_full_area = float(np.trapz(model_raw, rt_grid_s))
    if model_raw_full_area <= 0:
        raise RuntimeError(f"Non-positive model area for ABL {abl}.")

    model_full = model_raw / model_raw_full_area
    model_raw_by_abl[abl] = model_raw
    model_full_by_abl[abl] = model_full
    data_full_by_abl[abl] = data_full

    for cutoff_ms, cutoff_s in TRUNCATION_S.items():
        grid_mask = truncation_grid_masks[cutoff_ms]
        abl_truncated_df = abl_df[abl_df["RTwrtStim"] <= cutoff_s]
        data_truncated, _ = np.histogram(
            abl_truncated_df["RTwrtStim"],
            bins=truncated_bins_by_ms[cutoff_ms],
            density=True,
        )
        model_raw_truncated_area = float(
            np.trapz(model_raw[grid_mask], rt_grid_s[grid_mask])
        )
        if model_raw_truncated_area <= 0:
            raise RuntimeError(f"Non-positive model area for ABL {abl} at {cutoff_ms} ms.")

        model_truncated_by_ms_abl[cutoff_ms][abl] = (
            model_raw[grid_mask] / model_raw_truncated_area
        )
        data_truncated_by_ms_abl[cutoff_ms][abl] = data_truncated
        summary_rows.append(
            {
                "truncation_ms": cutoff_ms,
                "ABL_dB": abl,
                "n_full_fit_rows": len(abl_df),
                "n_truncated_rows": len(abl_truncated_df),
                "data_truncated_fraction": len(abl_truncated_df) / len(abl_df),
                "model_raw_0_to_1s_area": model_raw_full_area,
                "model_raw_truncated_area": model_raw_truncated_area,
                "model_truncated_fraction_of_full": (
                    model_raw_truncated_area / model_raw_full_area
                ),
                "n_model_trials": model_n_by_abl[abl],
            }
        )

equal_animal_data_full_by_abl = {}
equal_animal_model_full_by_abl = {}
equal_animal_data_truncated_by_ms_abl = {
    cutoff_ms: {} for cutoff_ms in TRUNCATION_MS
}
equal_animal_model_truncated_by_ms_abl = {
    cutoff_ms: {} for cutoff_ms in TRUNCATION_MS
}

for abl in ABLS:
    animal_data_full_curves = []
    animal_model_full_curves = []
    animal_data_truncated_curves = {cutoff_ms: [] for cutoff_ms in TRUNCATION_MS}
    animal_model_truncated_curves = {cutoff_ms: [] for cutoff_ms in TRUNCATION_MS}

    for animal in ANIMALS:
        animal_abl_df = valid_df[
            (valid_df["animal"] == animal) & (valid_df["ABL"] == abl)
        ]
        if len(animal_abl_df) == 0:
            raise RuntimeError(f"No fitting rows for LED7/{animal}, ABL {abl}.")

        animal_data_full, _ = np.histogram(
            animal_abl_df["RTwrtStim"],
            bins=full_bins_s,
            density=True,
        )
        animal_model_raw = (
            model_sum_by_animal_abl[(animal, abl)]
            / model_n_by_animal_abl[(animal, abl)]
        )
        animal_model_full_area = float(np.trapz(animal_model_raw, rt_grid_s))
        if animal_model_full_area <= 0:
            raise RuntimeError(f"Non-positive model area for LED7/{animal}, ABL {abl}.")

        animal_data_full_curves.append(animal_data_full)
        animal_model_full_curves.append(animal_model_raw / animal_model_full_area)

        for cutoff_ms, cutoff_s in TRUNCATION_S.items():
            grid_mask = truncation_grid_masks[cutoff_ms]
            animal_abl_truncated_df = animal_abl_df[
                animal_abl_df["RTwrtStim"] <= cutoff_s
            ]
            if len(animal_abl_truncated_df) == 0:
                raise RuntimeError(
                    f"No 0--{cutoff_ms} ms rows for LED7/{animal}, ABL {abl}."
                )
            animal_data_truncated, _ = np.histogram(
                animal_abl_truncated_df["RTwrtStim"],
                bins=truncated_bins_by_ms[cutoff_ms],
                density=True,
            )
            animal_model_truncated_area = float(
                np.trapz(animal_model_raw[grid_mask], rt_grid_s[grid_mask])
            )
            if animal_model_truncated_area <= 0:
                raise RuntimeError(
                    f"Non-positive 0--{cutoff_ms} ms model area for "
                    f"LED7/{animal}, ABL {abl}."
                )
            animal_data_truncated_curves[cutoff_ms].append(animal_data_truncated)
            animal_model_truncated_curves[cutoff_ms].append(
                animal_model_raw[grid_mask] / animal_model_truncated_area
            )

    equal_animal_data_full_by_abl[abl] = np.mean(animal_data_full_curves, axis=0)
    equal_animal_model_full_by_abl[abl] = np.mean(animal_model_full_curves, axis=0)
    for cutoff_ms in TRUNCATION_MS:
        equal_animal_data_truncated_by_ms_abl[cutoff_ms][abl] = np.mean(
            animal_data_truncated_curves[cutoff_ms],
            axis=0,
        )
        equal_animal_model_truncated_by_ms_abl[cutoff_ms][abl] = np.mean(
            animal_model_truncated_curves[cutoff_ms],
            axis=0,
        )

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_CSV, index=False)

payload = {
    "rt_grid_s": rt_grid_s,
    "full_bins_s": full_bins_s,
    "full_bin_centers_s": full_bin_centers_s,
    "animals": np.asarray(ANIMALS, dtype=int),
    "abls": np.asarray(ABLS, dtype=int),
    "truncation_ms": np.asarray(TRUNCATION_MS, dtype=int),
    "n_total_fit_rows": np.asarray([len(valid_df)], dtype=int),
    "data_csv": np.asarray([str(DATA_CSV.relative_to(REPO_ROOT))]),
    "out_led_csv": np.asarray([str(OUT_LED_CSV.relative_to(REPO_ROOT))]),
    "fit_root": np.asarray([str(FIT_ROOT.relative_to(REPO_ROOT))]),
    "formula_check_relative_max": np.asarray(direct_formula_checks, dtype=float),
    "truncation_denominator_min": np.asarray([np.min(normalization_denominators)]),
    "truncation_denominator_max": np.asarray([np.max(normalization_denominators)]),
}
for abl in ABLS:
    payload[f"data_full_density_ABL{abl}"] = data_full_by_abl[abl]
    payload[f"model_raw_density_ABL{abl}"] = model_raw_by_abl[abl]
    payload[f"model_full_density_ABL{abl}"] = model_full_by_abl[abl]
    payload[f"equal_animal_data_full_density_ABL{abl}"] = (
        equal_animal_data_full_by_abl[abl]
    )
    payload[f"equal_animal_model_full_density_ABL{abl}"] = (
        equal_animal_model_full_by_abl[abl]
    )
for cutoff_ms in TRUNCATION_MS:
    grid_mask = truncation_grid_masks[cutoff_ms]
    payload[f"rt_grid_{cutoff_ms}ms_s"] = rt_grid_s[grid_mask]
    payload[f"bins_{cutoff_ms}ms_s"] = truncated_bins_by_ms[cutoff_ms]
    payload[f"bin_centers_{cutoff_ms}ms_s"] = truncated_bin_centers_by_ms[cutoff_ms]
    payload[f"n_0_to_{cutoff_ms}ms"] = np.asarray(
        [len(truncated_df_by_ms[cutoff_ms])], dtype=int
    )
    for abl in ABLS:
        payload[f"data_{cutoff_ms}ms_conditional_density_ABL{abl}"] = (
            data_truncated_by_ms_abl[cutoff_ms][abl]
        )
        payload[f"model_{cutoff_ms}ms_conditional_density_ABL{abl}"] = (
            model_truncated_by_ms_abl[cutoff_ms][abl]
        )
        payload[f"equal_animal_data_{cutoff_ms}ms_conditional_density_ABL{abl}"] = (
            equal_animal_data_truncated_by_ms_abl[cutoff_ms][abl]
        )
        payload[f"equal_animal_model_{cutoff_ms}ms_conditional_density_ABL{abl}"] = (
            equal_animal_model_truncated_by_ms_abl[cutoff_ms][abl]
        )
np.savez_compressed(PAYLOAD_NPZ, **payload)

print("\nFit-aligned retained-mass summary:")
print(summary_df.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
print(
    "SVI truncation denominators: "
    f"min={np.min(normalization_denominators):.6g}, "
    f"max={np.max(normalization_denominators):.6g}"
)


# %%
# =============================================================================
# Plot complete and conditionally truncated RTDs
# =============================================================================
def add_style_legend(ax, location="upper right"):
    handles = [
        Line2D([0], [0], color="0.25", linestyle="-", linewidth=1.5, label="Data"),
        Line2D(
            [0],
            [0],
            color="0.25",
            linestyle="-",
            linewidth=3,
            alpha=THEORY_ALPHA,
            label="NPL+alpha SVI",
        ),
    ]
    handles.extend(
        Line2D([0], [0], color=ABL_COLORS[abl], linewidth=3, label=f"ABL {abl} dB")
        for abl in ABLS
    )
    ax.legend(handles=handles, loc=location, frameon=False, fontsize=8.5)


def draw_full_panel(ax, data_curves_by_abl, model_curves_by_abl, title):
    for abl in ABLS:
        ax.plot(
            rt_grid_s * 1e3,
            model_curves_by_abl[abl],
            color=ABL_COLORS[abl],
            linestyle="-",
            linewidth=3,
            alpha=THEORY_ALPHA,
            zorder=2,
        )
        ax.stairs(
            data_curves_by_abl[abl],
            full_bins_s * 1e3,
            color=ABL_COLORS[abl],
            linestyle="-",
            linewidth=1.5,
            alpha=1.0,
            zorder=3,
        )
    ax.set_xlim(0, 1000)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"RT - $t_{stim}$ (ms)")
    ax.set_ylabel(r"Density (s$^{-1}$)")
    ax.set_title(title)
    add_style_legend(ax)


def draw_truncated_panel(
    ax,
    cutoff_ms,
    data_curves_by_ms_abl,
    model_curves_by_ms_abl,
    title,
):
    grid_mask = truncation_grid_masks[cutoff_ms]
    for abl in ABLS:
        ax.plot(
            rt_grid_s[grid_mask] * 1e3,
            model_curves_by_ms_abl[cutoff_ms][abl],
            color=ABL_COLORS[abl],
            linestyle="-",
            linewidth=3,
            alpha=THEORY_ALPHA,
            zorder=2,
        )
        ax.stairs(
            data_curves_by_ms_abl[cutoff_ms][abl],
            truncated_bins_by_ms[cutoff_ms] * 1e3,
            color=ABL_COLORS[abl],
            linestyle="-",
            linewidth=1.5,
            alpha=1.0,
            zorder=3,
        )
    ax.set_xlim(0, cutoff_ms)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"RT - $t_{stim}$ (ms)")
    ax.set_ylabel(r"Conditional density (s$^{-1}$)")
    ax.set_title(title)


fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.3), constrained_layout=True)
draw_full_panel(
    axes[0],
    data_full_by_abl,
    model_full_by_abl,
    "Complete valid-trial RTD (0--1 s)",
)
draw_truncated_panel(
    axes[1],
    115,
    data_truncated_by_ms_abl,
    model_truncated_by_ms_abl,
    "Valid-trial RTD (0--115 ms, conditional)",
)
axes[0].text(-0.13, 1.04, "A", transform=axes[0].transAxes, fontsize=14, fontweight="bold")
axes[1].text(-0.13, 1.04, "B", transform=axes[1].transAxes, fontsize=14, fontweight="bold")
fig.suptitle(
    "LED7 valid trials: exact patience-12 NPL+alpha SVI fit rows and normalization",
    fontsize=12,
)
fig.savefig(COMBINED_PNG, dpi=PLOT_DPI, bbox_inches="tight")
fig.savefig(COMBINED_PDF, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
draw_full_panel(
    ax,
    data_full_by_abl,
    model_full_by_abl,
    "Complete valid-trial RTD (0--1 s)",
)
fig.savefig(FULL_PNG, dpi=PLOT_DPI, bbox_inches="tight")
fig.savefig(FULL_PDF, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(18.0, 4.5), constrained_layout=True)
for panel_label, ax, cutoff_ms in zip("ABC", axes, (130, 150, 170)):
    draw_truncated_panel(
        ax,
        cutoff_ms,
        data_truncated_by_ms_abl,
        model_truncated_by_ms_abl,
        f"Valid-trial RTD (0--{cutoff_ms} ms, conditional)",
    )
    ax.text(-0.13, 1.04, panel_label, transform=ax.transAxes, fontsize=14, fontweight="bold")
fig.suptitle(
    "LED7 conditionally truncated valid-trial RTDs: exact patience-12 NPL+alpha SVI",
    fontsize=12,
)
fig.savefig(ADDITIONAL_TRUNCATIONS_PNG, dpi=PLOT_DPI, bbox_inches="tight")
fig.savefig(ADDITIONAL_TRUNCATIONS_PDF, bbox_inches="tight")
plt.close(fig)

for cutoff_ms in TRUNCATION_MS:
    fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    draw_truncated_panel(
        ax,
        cutoff_ms,
        data_truncated_by_ms_abl,
        model_truncated_by_ms_abl,
        f"Valid-trial RTD (0--{cutoff_ms} ms, conditional)",
    )
    fig.savefig(TRUNCATED_PNG_BY_MS[cutoff_ms], dpi=PLOT_DPI, bbox_inches="tight")
    fig.savefig(TRUNCATED_PDF_BY_MS[cutoff_ms], bbox_inches="tight")
    plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.3), constrained_layout=True)
draw_full_panel(
    axes[0],
    equal_animal_data_full_by_abl,
    equal_animal_model_full_by_abl,
    "Complete valid-trial RTD (0--1 s)",
)
draw_truncated_panel(
    axes[1],
    115,
    equal_animal_data_truncated_by_ms_abl,
    equal_animal_model_truncated_by_ms_abl,
    "Valid-trial RTD truncated at 115 ms",
)
axes[0].text(-0.13, 1.04, "A", transform=axes[0].transAxes, fontsize=14, fontweight="bold")
axes[1].text(-0.13, 1.04, "B", transform=axes[1].transAxes, fontsize=14, fontweight="bold")
fig.suptitle(
    "LED7 valid trials: equal mean of six animal-specific RTDs",
    fontsize=12,
)
fig.savefig(EQUAL_ANIMAL_COMBINED_PNG, dpi=PLOT_DPI, bbox_inches="tight")
fig.savefig(EQUAL_ANIMAL_COMBINED_PDF, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
draw_full_panel(
    ax,
    equal_animal_data_full_by_abl,
    equal_animal_model_full_by_abl,
    "Complete valid-trial RTD (0--1 s)",
)
fig.savefig(EQUAL_ANIMAL_FULL_PNG, dpi=PLOT_DPI, bbox_inches="tight")
fig.savefig(EQUAL_ANIMAL_FULL_PDF, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(18.0, 4.5), constrained_layout=True)
for panel_label, ax, cutoff_ms in zip("ABC", axes, (130, 150, 170)):
    draw_truncated_panel(
        ax,
        cutoff_ms,
        equal_animal_data_truncated_by_ms_abl,
        equal_animal_model_truncated_by_ms_abl,
        f"Valid-trial RTD truncated at {cutoff_ms} ms",
    )
    ax.text(-0.13, 1.04, panel_label, transform=ax.transAxes, fontsize=14, fontweight="bold")
fig.suptitle(
    "LED7 conditionally truncated RTDs: equal mean of six animals",
    fontsize=12,
)
fig.savefig(EQUAL_ANIMAL_ADDITIONAL_TRUNCATIONS_PNG, dpi=PLOT_DPI, bbox_inches="tight")
fig.savefig(EQUAL_ANIMAL_ADDITIONAL_TRUNCATIONS_PDF, bbox_inches="tight")
plt.close(fig)

for cutoff_ms in TRUNCATION_MS:
    fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    draw_truncated_panel(
        ax,
        cutoff_ms,
        equal_animal_data_truncated_by_ms_abl,
        equal_animal_model_truncated_by_ms_abl,
        f"Valid-trial RTD truncated at {cutoff_ms} ms",
    )
    fig.savefig(
        EQUAL_ANIMAL_TRUNCATED_PNG_BY_MS[cutoff_ms],
        dpi=PLOT_DPI,
        bbox_inches="tight",
    )
    fig.savefig(EQUAL_ANIMAL_TRUNCATED_PDF_BY_MS[cutoff_ms], bbox_inches="tight")
    plt.close(fig)

print("\nSaved figures:")
figure_paths = [
    COMBINED_PNG,
    COMBINED_PDF,
    FULL_PNG,
    FULL_PDF,
    ADDITIONAL_TRUNCATIONS_PNG,
    ADDITIONAL_TRUNCATIONS_PDF,
    EQUAL_ANIMAL_COMBINED_PNG,
    EQUAL_ANIMAL_COMBINED_PDF,
    EQUAL_ANIMAL_FULL_PNG,
    EQUAL_ANIMAL_FULL_PDF,
    EQUAL_ANIMAL_ADDITIONAL_TRUNCATIONS_PNG,
    EQUAL_ANIMAL_ADDITIONAL_TRUNCATIONS_PDF,
]
for cutoff_ms in TRUNCATION_MS:
    figure_paths.extend([TRUNCATED_PNG_BY_MS[cutoff_ms], TRUNCATED_PDF_BY_MS[cutoff_ms]])
    figure_paths.extend(
        [
            EQUAL_ANIMAL_TRUNCATED_PNG_BY_MS[cutoff_ms],
            EQUAL_ANIMAL_TRUNCATED_PDF_BY_MS[cutoff_ms],
        ]
    )
for path in figure_paths:
    print(f"  {path}")
print("Saved audit artifacts:")
for path in [SUMMARY_CSV, DATA_AUDIT_CSV, ILD_COUNT_AUDIT_CSV, PAYLOAD_NPZ]:
    print(f"  {path}")
