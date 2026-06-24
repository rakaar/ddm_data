# %%
"""
Exploratory NumPyro SVI fit for one NPL+alpha animal.

Goal: fit shared NPL+alpha parameters plus one t_E_aff per observed
ABL/signed-ILD condition, then sample an approximate posterior for corner
plots. The long JAX likelihood lives in numpyro_npl_alpha_svi_utils.py so this
script can be run cell-by-cell and inspected.
"""

# %%
# =============================================================================
# Editable parameters
# =============================================================================
from pathlib import Path
import importlib.util
import os
import pickle
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

BATCH_NAME = os.environ.get("NUMPYRO_SVI_BATCH", "LED7")
ANIMAL = int(os.environ.get("NUMPYRO_SVI_ANIMAL", "92"))

GUIDE_KIND = os.environ.get("NUMPYRO_SVI_GUIDE", "fullrank")  # fullrank, iaf, meanfield, lowrank
IAF_HIDDEN_DIMS = tuple(
    int(part)
    for part in os.environ.get("NUMPYRO_SVI_IAF_HIDDEN_DIMS", "64,64").replace("x", ",").split(",")
    if part.strip()
)
IAF_NUM_FLOWS = int(os.environ.get("NUMPYRO_SVI_IAF_NUM_FLOWS", "1"))  # 1 is the stable default here; 2+ can destabilize early.
IAF_SKIP_CONNECTIONS = os.environ.get("NUMPYRO_SVI_IAF_SKIP_CONNECTIONS", "0").strip().lower() in {"1", "true", "yes"}
IAF_BASE_SCALE = float(os.environ.get("NUMPYRO_SVI_IAF_BASE_SCALE", "0.05"))
LOWRANK_RANK = int(os.environ.get("NUMPYRO_SVI_LOWRANK_RANK", "10"))
USE_VBMC_GLOBAL_COV_INIT = os.environ.get("NUMPYRO_SVI_USE_VBMC_GLOBAL_COV_INIT", "1").strip().lower() in {"1", "true", "yes"}
DELAY_INIT_LATENT_SCALE = float(os.environ.get("NUMPYRO_SVI_DELAY_INIT_LATENT_SCALE", "0.05"))

RUN_SMOKE_SVI = os.environ.get("RUN_SMOKE_SVI", "0").strip().lower() in {"1", "true", "yes"}
RUN_MAIN_SVI = os.environ.get("RUN_MAIN_SVI", "1").strip().lower() in {"1", "true", "yes"}
SMOKE_N_TRIALS = int(os.environ.get("SMOKE_N_TRIALS", "2000"))
MAIN_N_TRIALS_OVERRIDE = int(os.environ.get("MAIN_N_TRIALS_OVERRIDE", "0"))  # 0 means all trials
SMOKE_STEPS = int(os.environ.get("SMOKE_STEPS", "300"))
MAIN_STEPS = int(os.environ.get("MAIN_STEPS", "100000"))
SVI_CHECK_EVERY = int(os.environ.get("SVI_CHECK_EVERY", "10000"))
SVI_EARLY_STOP = os.environ.get("SVI_EARLY_STOP", "1").strip().lower() in {"1", "true", "yes"}
SVI_REL_TOL = float(os.environ.get("SVI_REL_TOL", "0.01"))
SVI_PATIENCE_WINDOWS = int(os.environ.get("SVI_PATIENCE_WINDOWS", "3"))
SVI_MIN_IMPROVEMENT_REL = float(os.environ.get("SVI_MIN_IMPROVEMENT_REL", "0.005"))
SVI_NO_IMPROVE_PATIENCE_WINDOWS = int(os.environ.get("SVI_NO_IMPROVE_PATIENCE_WINDOWS", "2"))
SVI_STABLE_UPDATE = os.environ.get("SVI_STABLE_UPDATE", "1").strip().lower() in {"1", "true", "yes"}
LEARNING_RATE = float(os.environ.get("NUMPYRO_SVI_LR", "0.0002"))
OPTIMIZER_KIND = os.environ.get("NUMPYRO_SVI_OPTIMIZER", "clipped_adam")
CLIP_NORM = float(os.environ.get("NUMPYRO_SVI_CLIP_NORM", "1.0"))
POSTERIOR_N_SAMPLES = int(os.environ.get("POSTERIOR_N_SAMPLES", "10000"))
RNG_SEED = int(os.environ.get("NUMPYRO_SVI_SEED", "0"))
K_MAX = int(os.environ.get("K_MAX", "10"))
MIN_CORNER_SAMPLES = int(os.environ.get("MIN_CORNER_SAMPLES", "20"))
SAVE_TABLE_CSVS = os.environ.get("NUMPYRO_SVI_SAVE_TABLE_CSVS", "0").strip().lower() in {"1", "true", "yes"}

HARD_CODED_DELAY_ABL_LEVELS = [20.0, 40.0, 60.0]
BATCH_T_TRUNC = {"LED34_even": 0.15}
DEFAULT_T_TRUNC = 0.3

BATCH_CSV = REPO_DIR / "raw_data" / "batch_csvs" / f"batch_{BATCH_NAME}_valid_and_aborts.csv"
ABORT_RESULT_PKL = REPO_DIR / "aborts_ipl_npl_time_fit_results" / f"results_{BATCH_NAME}_animal_{ANIMAL}.pkl"
FIXED_DELAY_RESULT_PKL = (
    SCRIPT_DIR
    / "NPL_alpha_condition_t_E_aff_fixed_delay_fit_results_all_30"
    / f"results_{BATCH_NAME}_animal_{ANIMAL}_NORM_ALPHA_CONDITION_T_E_AFF_FIXED_DELAY_FROM_ABORTS.pkl"
)
CONDITION_T_E_AFF_CACHE = (
    REPO_DIR
    / "fit_each_condn"
    / "abl_specific_ild2_delay_agreement"
    / "condition_t_E_aff_extraction_cache.csv"
)

OUTPUT_DIR = (
    SCRIPT_DIR
    / "numpyro_svi_npl_alpha_condition_delay_single_animal_outputs"
    / f"{BATCH_NAME}_{ANIMAL}"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# %%
# =============================================================================
# Dependency preflight
# =============================================================================
required_modules = ["jax", "jaxlib", "numpyro"]
missing_modules = [module for module in required_modules if importlib.util.find_spec(module) is None]
if missing_modules:
    print("Missing dependencies for NumPyro SVI prototype:")
    for module in missing_modules:
        print(f"  - {module}")
    print("\nInstall the GPU-enabled JAX/NumPyro stack used for this prototype with:")
    print(
        '  .venv/bin/python -m pip install '
        '"numpy==1.26.4" "scipy==1.12.0" '
        '"jax[cuda12]==0.4.38" "jaxlib==0.4.38" "numpyro==0.15.3"'
    )
    raise SystemExit(1)


# %%
# =============================================================================
# Imports after dependency preflight
# =============================================================================
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import corner
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import numpyro
import pandas as pd
from jax import random
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.svi import SVIRunResult
from numpyro.infer.util import log_density

sys.path.insert(0, str(SCRIPT_DIR))
import numpyro_npl_alpha_svi_utils as svi_utils


# %%
# =============================================================================
# Load animal data and upstream initial values
# =============================================================================
print(f"Batch/animal: {BATCH_NAME}/{ANIMAL}")
print(f"Batch CSV: {BATCH_CSV}")
print(f"Abort params: {ABORT_RESULT_PKL}")
print(f"Fixed-delay initialization: {FIXED_DELAY_RESULT_PKL}")
print(f"Condition delay cache: {CONDITION_T_E_AFF_CACHE}")
print(f"Output folder: {OUTPUT_DIR}")

for required_path in [BATCH_CSV, ABORT_RESULT_PKL, FIXED_DELAY_RESULT_PKL, CONDITION_T_E_AFF_CACHE]:
    if not required_path.exists():
        raise FileNotFoundError(required_path)

raw_df = pd.read_csv(BATCH_CSV)
if "choice" not in raw_df.columns:
    if "response_poke" not in raw_df.columns:
        raise KeyError("Need either `choice` or `response_poke` in the batch CSV.")
    raw_df["choice"] = raw_df["response_poke"].map({3: 1, 2: -1})

valid_df = raw_df[
    (raw_df["animal"].astype(int) == ANIMAL)
    & (raw_df["success"].isin([1, -1]))
    & (raw_df["RTwrtStim"] < 1)
    & (raw_df["ABL"].isin(HARD_CODED_DELAY_ABL_LEVELS))
].copy()
valid_df = valid_df.dropna(subset=["TotalFixTime", "intended_fix", "ABL", "ILD", "choice", "RTwrtStim"])

if len(valid_df) == 0:
    raise RuntimeError(f"No valid RT<1 trials for {BATCH_NAME}/{ANIMAL}.")

valid_df["ABL"] = valid_df["ABL"].astype(float)
valid_df["ILD"] = valid_df["ILD"].astype(float)
valid_df["choice"] = valid_df["choice"].astype(int)

condition_table = (
    valid_df[["ABL", "ILD"]]
    .drop_duplicates()
    .sort_values(["ABL", "ILD"])
    .reset_index(drop=True)
)
condition_table["condition_id"] = np.arange(len(condition_table), dtype=int)
valid_df = valid_df.merge(condition_table, on=["ABL", "ILD"], how="left", validate="many_to_one")
if valid_df["condition_id"].isna().any():
    raise RuntimeError("Failed to assign condition IDs to all trials.")

print(f"Valid fitting trials: {len(valid_df)}")
print(f"Observed conditions: {len(condition_table)}")
print(condition_table.to_string(index=False))

with ABORT_RESULT_PKL.open("rb") as f:
    abort_saved = pickle.load(f)
abort_results = abort_saved["vbmc_aborts_results"]
V_A = float(np.mean(abort_results["V_A_samples"]))
theta_A = float(np.mean(abort_results["theta_A_samples"]))
t_A_aff = float(np.mean(abort_results["t_A_aff_samp"]))
print("\nAbort/proactive posterior means:")
print(f"  V_A      = {V_A:.6g}")
print(f"  theta_A  = {theta_A:.6g}")
print(f"  t_A_aff  = {1e3 * t_A_aff:.3f} ms")

with FIXED_DELAY_RESULT_PKL.open("rb") as f:
    fixed_delay_saved = pickle.load(f)
fixed_fit = fixed_delay_saved["vbmc_norm_alpha_condition_t_E_aff_fixed_delay_tied_results"]

condition_cache_df = pd.read_csv(CONDITION_T_E_AFF_CACHE)
animal_delay_df = condition_cache_df[
    (condition_cache_df["batch_name"].astype(str) == str(BATCH_NAME))
    & (condition_cache_df["animal"].astype(int) == int(ANIMAL))
][["ABL", "ILD", "t_E_aff_s", "t_E_aff_ms"]].copy()
animal_delay_df["ABL"] = animal_delay_df["ABL"].astype(float)
animal_delay_df["ILD"] = animal_delay_df["ILD"].astype(float)

condition_table = condition_table.merge(
    animal_delay_df,
    on=["ABL", "ILD"],
    how="left",
    validate="one_to_one",
)
if condition_table["t_E_aff_s"].isna().any():
    missing = condition_table[condition_table["t_E_aff_s"].isna()]
    raise RuntimeError(f"Missing condition-cache t_E_aff rows:\n{missing}")

condition_table = condition_table.sort_values("condition_id").reset_index(drop=True)
print("\nCondition delay initialization:")
print(condition_table[["condition_id", "ABL", "ILD", "t_E_aff_ms"]].to_string(index=False))

init_values = {
    "rate_lambda": float(np.mean(fixed_fit["rate_lambda_samples"])),
    "T_0": float(np.mean(fixed_fit["T_0_samples"])),
    "theta_E": float(np.mean(fixed_fit["theta_E_samples"])),
    "w": float(np.mean(fixed_fit["w_samples"])),
    "del_go": float(np.mean(fixed_fit["del_go_samples"])),
    "rate_norm_l": float(np.mean(fixed_fit["rate_norm_l_samples"])),
    "alpha": float(np.mean(fixed_fit["alpha_samples"])),
    "t_E_aff": condition_table["t_E_aff_s"].to_numpy(dtype=float),
}
init_values = svi_utils.clip_init_to_hard_bounds(init_values)

print("\nInitial global values from fixed condition-delay fit:")
for param_name in svi_utils.GLOBAL_PARAM_NAMES:
    value = init_values[param_name]
    if param_name in {"T_0", "del_go"}:
        print(f"  {param_name:<12} = {1e3 * value:.5f} ms")
    else:
        print(f"  {param_name:<12} = {value:.6g}")

fullrank_init_scale_tril = None
if GUIDE_KIND.strip().lower() in {"fullrank", "multivariate", "automultivariate"} and USE_VBMC_GLOBAL_COV_INIT:
    fixed_sample_keys = [
        "rate_lambda_samples",
        "T_0_samples",
        "theta_E_samples",
        "w_samples",
        "del_go_samples",
        "rate_norm_l_samples",
        "alpha_samples",
    ]
    latent_global_samples = []
    for param_name, sample_key in zip(svi_utils.GLOBAL_PARAM_NAMES, fixed_sample_keys):
        hard_low, hard_high = svi_utils.GLOBAL_BOUNDS[param_name]["hard"]
        values = np.asarray(fixed_fit[sample_key], dtype=float)
        eps = 1e-6 * (hard_high - hard_low)
        values = np.clip(values, hard_low + eps, hard_high - eps)
        unit_values = (values - hard_low) / (hard_high - hard_low)
        latent_global_samples.append(np.log(unit_values / (1.0 - unit_values)))

    latent_global_samples = np.column_stack(latent_global_samples)
    global_cov = np.cov(latent_global_samples, rowvar=False)
    global_corr = np.corrcoef(latent_global_samples, rowvar=False)

    latent_dim = len(svi_utils.GLOBAL_PARAM_NAMES) + int(len(condition_table))
    init_cov = np.eye(latent_dim) * (DELAY_INIT_LATENT_SCALE**2)
    init_cov[: len(svi_utils.GLOBAL_PARAM_NAMES), : len(svi_utils.GLOBAL_PARAM_NAMES)] = global_cov
    init_cov += np.eye(latent_dim) * 1e-8
    fullrank_init_scale_tril = np.linalg.cholesky(init_cov)

    print("\nFull-rank guide covariance initialized from fixed-delay VBMC global posterior:")
    print(f"  delay latent diagonal scale = {DELAY_INIT_LATENT_SCALE:g}")
    print(f"  VBMC latent corr(T_0, theta_E) = {global_corr[1, 2]:.3f}")
    print(f"  VBMC latent corr(rate_norm_l, alpha) = {global_corr[5, 6]:.3f}")


# %%
# =============================================================================
# Build JAX trial dictionaries
# =============================================================================
T_trunc = BATCH_T_TRUNC.get(BATCH_NAME, DEFAULT_T_TRUNC)


def make_jax_data(df):
    return {
        "total_fix": jnp.asarray(df["TotalFixTime"].to_numpy(dtype=float)),
        "t_stim": jnp.asarray(df["intended_fix"].to_numpy(dtype=float)),
        "ABL": jnp.asarray(df["ABL"].to_numpy(dtype=float)),
        "ILD": jnp.asarray(df["ILD"].to_numpy(dtype=float)),
        "choice": jnp.asarray(df["choice"].to_numpy(dtype=int)),
        "condition_id": jnp.asarray(df["condition_id"].to_numpy(dtype=int)),
        "V_A": jnp.asarray(V_A, dtype=jnp.float64),
        "theta_A": jnp.asarray(theta_A, dtype=jnp.float64),
        "t_A_aff": jnp.asarray(t_A_aff, dtype=jnp.float64),
        "T_trunc": jnp.asarray(T_trunc, dtype=jnp.float64),
    }


full_data = make_jax_data(valid_df)
if len(valid_df) > SMOKE_N_TRIALS:
    smoke_df = valid_df.sample(SMOKE_N_TRIALS, random_state=RNG_SEED).sort_index().copy()
else:
    smoke_df = valid_df.copy()
smoke_data = make_jax_data(smoke_df)
if MAIN_N_TRIALS_OVERRIDE > 0 and len(valid_df) > MAIN_N_TRIALS_OVERRIDE:
    main_df = valid_df.sample(MAIN_N_TRIALS_OVERRIDE, random_state=RNG_SEED + 1).sort_index().copy()
else:
    main_df = valid_df.copy()
main_data = make_jax_data(main_df)
n_conditions = int(len(condition_table))

print(f"\nT_trunc = {T_trunc:.3f} s")
print(f"Smoke trials: {len(smoke_df)}")
print(f"Main trials: {len(main_df)}")
print(f"Full available trials: {len(valid_df)}")
print(f"Guide: {GUIDE_KIND}")
if GUIDE_KIND.strip().lower() in {"iaf", "centered_iaf", "centered-iaf", "flow", "raw_iaf", "raw-iaf", "autoiaf"}:
    print(
        "IAF config: "
        f"hidden_dims={IAF_HIDDEN_DIMS}, "
        f"num_flows={IAF_NUM_FLOWS}, "
        f"skip_connections={IAF_SKIP_CONNECTIONS}, "
        f"base_scale={IAF_BASE_SCALE:g}"
    )
if GUIDE_KIND.strip().lower() in {"fullrank", "multivariate", "automultivariate"}:
    print(
        "Full-rank config: "
        f"use_vbmc_global_cov_init={USE_VBMC_GLOBAL_COV_INIT}, "
        f"delay_init_latent_scale={DELAY_INIT_LATENT_SCALE:g}"
    )
print(f"Optimizer: {OPTIMIZER_KIND}, learning_rate={LEARNING_RATE:g}, clip_norm={CLIP_NORM:g}")
print(
    "SVI convergence checks: "
    f"every {SVI_CHECK_EVERY} steps, "
    f"early_stop={SVI_EARLY_STOP}, "
    f"rel_tol={SVI_REL_TOL:g}, "
    f"patience_windows={SVI_PATIENCE_WINDOWS}, "
    f"min_improvement_rel={SVI_MIN_IMPROVEMENT_REL:g}, "
    f"no_improve_patience={SVI_NO_IMPROVE_PATIENCE_WINDOWS}, "
    f"stable_update={SVI_STABLE_UPDATE}"
)


def finite_sample_report(samples):
    rows = []
    all_finite = True
    for key, value in samples.items():
        arr = np.asarray(value)
        finite = np.isfinite(arr)
        all_finite = all_finite and bool(np.all(finite))
        rows.append(
            {
                "parameter": key,
                "shape": str(arr.shape),
                "n_total": int(arr.size),
                "n_finite": int(np.sum(finite)),
                "n_nan": int(np.sum(np.isnan(arr))),
                "n_inf": int(np.sum(np.isinf(arr))),
            }
        )
    return pd.DataFrame(rows), all_finite


def finite_corner_input(samples, labels, plot_name):
    samples = np.asarray(samples, dtype=float)
    finite_rows = np.all(np.isfinite(samples), axis=1)
    n_drop = int(np.sum(~finite_rows))
    if n_drop:
        print(f"{plot_name}: dropping {n_drop}/{len(samples)} posterior rows with NaN/Inf before corner plot.")
    clean = samples[finite_rows]
    if clean.shape[0] < MIN_CORNER_SAMPLES:
        print(
            f"{plot_name}: only {clean.shape[0]} finite posterior rows remain; "
            "skipping corner plot."
        )
        return None, None

    ranges = []
    for col_idx in range(clean.shape[1]):
        values = clean[:, col_idx]
        low, high = np.quantile(values, [0.01, 0.99])
        if not np.isfinite(low) or not np.isfinite(high):
            print(f"{plot_name}: non-finite range for {labels[col_idx]}; skipping corner plot.")
            return None, None
        if np.isclose(low, high, rtol=0, atol=1e-12):
            pad = max(1e-6, abs(low) * 1e-3)
            low -= pad
            high += pad
        ranges.append((float(low), float(high)))
    return clean, ranges


def make_optimizer():
    optimizer_kind = OPTIMIZER_KIND.strip().lower()
    if optimizer_kind in {"adam", "plain_adam"}:
        return numpyro.optim.Adam(LEARNING_RATE)
    if optimizer_kind in {"clipped_adam", "clipped-adam", "clip_adam"}:
        return numpyro.optim.ClippedAdam(LEARNING_RATE, clip_norm=CLIP_NORM)
    raise ValueError(f"Unknown NUMPYRO_SVI_OPTIMIZER={OPTIMIZER_KIND!r}")


def run_svi_with_convergence_checks(svi, rng_key, n_steps, data, n_conditions, run_label):
    if n_steps < 1:
        raise ValueError("n_steps must be positive.")
    if SVI_CHECK_EVERY < 1:
        raise ValueError("SVI_CHECK_EVERY must be positive.")

    all_losses = []
    convergence_rows = []
    state = None
    best_state = None
    best_params = None
    best_window_mean = np.inf
    best_window_chunk = 0
    best_window_end_step = 0
    prev_window_mean = np.nan
    stable_window_count = 0
    no_improve_window_count = 0
    completed_steps = 0

    print(
        f"\nRunning {run_label} for up to {n_steps} steps "
        f"with checks every {SVI_CHECK_EVERY} steps..."
    )
    while completed_steps < n_steps:
        chunk_index = len(convergence_rows) + 1
        chunk_steps = min(SVI_CHECK_EVERY, n_steps - completed_steps)
        start_step = completed_steps + 1
        end_step = completed_steps + chunk_steps

        chunk_result = svi.run(
            random.fold_in(rng_key, chunk_index),
            chunk_steps,
            data,
            n_conditions,
            progress_bar=False,
            init_state=state,
            stable_update=SVI_STABLE_UPDATE,
        )
        state = chunk_result.state
        window_losses = np.asarray(jax.device_get(chunk_result.losses), dtype=float)
        all_losses.append(window_losses)
        completed_steps = end_step

        finite_mask = np.isfinite(window_losses)
        finite_losses = window_losses[finite_mask]
        n_nonfinite = int(np.sum(~finite_mask))
        if finite_losses.size:
            window_mean = float(np.mean(finite_losses))
            window_median = float(np.median(finite_losses))
            window_last = float(window_losses[-1]) if np.isfinite(window_losses[-1]) else np.nan
            window_min = float(np.min(finite_losses))
            window_max = float(np.max(finite_losses))
        else:
            window_mean = np.nan
            window_median = np.nan
            window_last = np.nan
            window_min = np.nan
            window_max = np.nan

        if finite_losses.size > 1:
            finite_x = np.flatnonzero(finite_mask).astype(float)
            slope_per_1000 = float(np.polyfit(finite_x, finite_losses, 1)[0] * 1000.0)
        else:
            slope_per_1000 = np.nan

        if np.isfinite(prev_window_mean) and np.isfinite(window_mean):
            delta_from_prev = window_mean - prev_window_mean
            rel_change = abs(delta_from_prev) / max(1.0, abs(prev_window_mean))
            is_stable = bool(rel_change <= SVI_REL_TOL)
        else:
            delta_from_prev = np.nan
            rel_change = np.nan
            is_stable = False

        improvement_from_best = np.nan
        relative_improvement_from_best = np.nan
        improved_best = False
        significant_improvement = False
        if np.isfinite(window_mean):
            if np.isfinite(best_window_mean):
                improvement_from_best = best_window_mean - window_mean
                relative_improvement_from_best = improvement_from_best / max(1.0, abs(best_window_mean))
            if (not np.isfinite(best_window_mean)) or (window_mean < best_window_mean):
                improved_best = True
            if (not np.isfinite(relative_improvement_from_best)) or (
                relative_improvement_from_best >= SVI_MIN_IMPROVEMENT_REL
            ):
                significant_improvement = improved_best

        if improved_best and n_nonfinite == 0:
            best_state = state
            best_params = chunk_result.params
            best_window_mean = window_mean
            best_window_chunk = chunk_index
            best_window_end_step = end_step
            if significant_improvement:
                no_improve_window_count = 0
            else:
                no_improve_window_count += 1
        else:
            no_improve_window_count += 1

        if is_stable:
            stable_window_count += 1
        else:
            stable_window_count = 0

        convergence_rows.append(
            {
                "chunk": chunk_index,
                "start_step": start_step,
                "end_step": end_step,
                "n_steps": chunk_steps,
                "mean_loss": window_mean,
                "median_loss": window_median,
                "last_loss": window_last,
                "min_loss": window_min,
                "max_loss": window_max,
                "delta_mean_from_prev": delta_from_prev,
                "relative_mean_change": rel_change,
                "improvement_from_best": improvement_from_best,
                "relative_improvement_from_best": relative_improvement_from_best,
                "best_mean_loss_so_far": best_window_mean,
                "best_chunk_so_far": best_window_chunk,
                "best_end_step_so_far": best_window_end_step,
                "no_improve_window_count": no_improve_window_count,
                "updated_best_state": bool(improved_best and n_nonfinite == 0),
                "significant_best_improvement": bool(significant_improvement and n_nonfinite == 0),
                "slope_per_1000_steps": slope_per_1000,
                "stable_window_count": stable_window_count,
                "n_nonfinite": n_nonfinite,
                "early_stop_candidate": bool(stable_window_count >= SVI_PATIENCE_WINDOWS),
                "no_improve_stop_candidate": bool(no_improve_window_count >= SVI_NO_IMPROVE_PATIENCE_WINDOWS),
            }
        )

        rel_text = "NA" if not np.isfinite(rel_change) else f"{100.0 * rel_change:.3f}%"
        best_rel_text = (
            "NA"
            if not np.isfinite(relative_improvement_from_best)
            else f"{100.0 * relative_improvement_from_best:.3f}%"
        )
        slope_text = "NA" if not np.isfinite(slope_per_1000) else f"{slope_per_1000:.3g}"
        print(
            f"{run_label} chunk {chunk_index:02d} steps {start_step}-{end_step}: "
            f"mean={window_mean:.6g}, last={window_last:.6g}, "
            f"rel_change={rel_text}, best_delta={best_rel_text}, slope/1k={slope_text}, "
            f"stable={stable_window_count}/{SVI_PATIENCE_WINDOWS}, "
            f"no_improve={no_improve_window_count}/{SVI_NO_IMPROVE_PATIENCE_WINDOWS}, "
            f"best_chunk={best_window_chunk}, "
            f"nonfinite={n_nonfinite}"
        )

        if n_nonfinite:
            print(
                f"WARNING: stopping {run_label} because this window had non-finite losses. "
                f"Returning best state from chunk {best_window_chunk} ending at step {best_window_end_step}."
            )
            break
        if SVI_EARLY_STOP and stable_window_count >= SVI_PATIENCE_WINDOWS:
            print(
                f"Stopping {run_label} early at step {completed_steps}: "
                f"{SVI_PATIENCE_WINDOWS} consecutive windows changed by <= {100.0 * SVI_REL_TOL:.3g}%."
            )
            break
        if SVI_EARLY_STOP and no_improve_window_count >= SVI_NO_IMPROVE_PATIENCE_WINDOWS:
            print(
                f"Stopping {run_label} early at step {completed_steps}: "
                f"no best-window improvement for {SVI_NO_IMPROVE_PATIENCE_WINDOWS} consecutive windows. "
                f"Returning best state from chunk {best_window_chunk} ending at step {best_window_end_step}."
            )
            break

        prev_window_mean = window_mean

    losses = np.concatenate(all_losses) if all_losses else np.array([], dtype=float)
    if best_state is None:
        best_state = state
        best_params = svi.get_params(state)
    if np.isfinite(best_window_mean):
        print(
            f"{run_label} best returned state: chunk {best_window_chunk}, "
            f"step {best_window_end_step}, mean_loss={best_window_mean:.6g}"
        )
    result = SVIRunResult(best_params, best_state, jnp.asarray(losses))
    convergence_df = pd.DataFrame(convergence_rows)
    return result, convergence_df


# %%
# =============================================================================
# Initial log joint and gradients
# =============================================================================
model = lambda data, n_conditions: svi_utils.npl_alpha_condition_delay_model(
    data,
    n_conditions,
    K_max=K_MAX,
)


def log_joint_from_values(values, data):
    log_joint, _ = log_density(model, (data, n_conditions), {}, values)
    return log_joint


initial_smoke_log_joint = log_joint_from_values(init_values, smoke_data)
initial_full_log_joint = log_joint_from_values(init_values, full_data)
initial_grad = jax.grad(lambda values: log_joint_from_values(values, smoke_data))(init_values)

print("\nInitial log joint:")
print(f"  smoke = {float(initial_smoke_log_joint):.6f}")
print(f"  full  = {float(initial_full_log_joint):.6f}")
print(f"Initial smoke gradient finite: {svi_utils.tree_all_finite(initial_grad)}")
if not np.isfinite(float(initial_smoke_log_joint)) or not svi_utils.tree_all_finite(initial_grad):
    raise RuntimeError("Initial log joint or gradients are non-finite.")


# %%
# =============================================================================
# Short AutoNormal smoke SVI
# =============================================================================
active_result = None
active_guide = None
active_label = None
active_data = None
active_convergence_df = None

if RUN_SMOKE_SVI:
    print(f"\nRunning AutoNormal smoke SVI for {SMOKE_STEPS} steps on {len(smoke_df)} trials...")
    smoke_guide = svi_utils.make_guide(model, "meanfield", init_values)
    smoke_svi = SVI(model, smoke_guide, make_optimizer(), Trace_ELBO())
    smoke_result, smoke_convergence_df = run_svi_with_convergence_checks(
        smoke_svi,
        random.PRNGKey(RNG_SEED),
        SMOKE_STEPS,
        smoke_data,
        n_conditions,
        "smoke_autonormal",
    )
    print(f"Smoke loss: first={float(smoke_result.losses[0]):.6f}, last={float(smoke_result.losses[-1]):.6f}")
    active_result = smoke_result
    active_guide = smoke_guide
    active_label = "smoke_autonormal"
    active_data = smoke_data
    active_convergence_df = smoke_convergence_df
else:
    print("\nSkipping smoke SVI because RUN_SMOKE_SVI=False.")


# %%
# =============================================================================
# Main SVI, default centered one-flow IAF
# =============================================================================
if RUN_MAIN_SVI:
    print(f"\nRunning {GUIDE_KIND} main SVI for {MAIN_STEPS} steps on {len(main_df)} trials...")
    main_guide = svi_utils.make_guide(
        model,
        GUIDE_KIND,
        init_values,
        iaf_hidden_dims=IAF_HIDDEN_DIMS,
        iaf_num_flows=IAF_NUM_FLOWS,
        iaf_skip_connections=IAF_SKIP_CONNECTIONS,
        iaf_base_scale=IAF_BASE_SCALE,
        fullrank_init_scale_tril=fullrank_init_scale_tril,
        lowrank_rank=LOWRANK_RANK,
    )
    main_svi = SVI(model, main_guide, make_optimizer(), Trace_ELBO())
    main_result, main_convergence_df = run_svi_with_convergence_checks(
        main_svi,
        random.PRNGKey(RNG_SEED + 1),
        MAIN_STEPS,
        main_data,
        n_conditions,
        f"main_{GUIDE_KIND}",
    )
    print(f"Main loss: first={float(main_result.losses[0]):.6f}, last={float(main_result.losses[-1]):.6f}")
    nonfinite_loss = np.flatnonzero(~np.isfinite(np.asarray(main_result.losses)))
    if len(nonfinite_loss):
        print(
            "WARNING: main SVI produced non-finite losses at steps "
            f"{nonfinite_loss[:10].tolist()}"
            f"{'...' if len(nonfinite_loss) > 10 else ''}."
        )
    active_result = main_result
    active_guide = main_guide
    active_label = f"main_{GUIDE_KIND}"
    active_data = main_data
    active_convergence_df = main_convergence_df
else:
    print("\nSkipping main SVI because RUN_MAIN_SVI=False.")


# %%
# =============================================================================
# Posterior samples and diagnostics
# =============================================================================
if active_result is None:
    print("\nNo SVI result is available; set RUN_SMOKE_SVI or RUN_MAIN_SVI to save diagnostics.")
else:
    posterior_samples = active_guide.sample_posterior(
        random.PRNGKey(RNG_SEED + 2),
        active_result.params,
        sample_shape=(POSTERIOR_N_SAMPLES,),
    )
    posterior_np = {key: np.asarray(value) for key, value in posterior_samples.items()}
    finite_report_df, all_posterior_finite = finite_sample_report(posterior_np)
    finite_report_csv = OUTPUT_DIR / f"{active_label}_posterior_finite_report.csv" if SAVE_TABLE_CSVS else None
    if SAVE_TABLE_CSVS:
        finite_report_df.to_csv(finite_report_csv, index=False)
    print("\nPosterior finite-sample report:")
    print(finite_report_df.to_string(index=False))
    if not all_posterior_finite:
        print(
            "WARNING: posterior samples contain NaN/Inf. "
            "Plots will use finite rows only, and fully invalid plots will be skipped."
        )
    posterior_summary_df = svi_utils.posterior_samples_to_frame(posterior_np, condition_table)

    sample_npz = OUTPUT_DIR / f"{active_label}_posterior_samples.npz"
    np.savez(sample_npz, **posterior_np)
    summary_csv = OUTPUT_DIR / f"{active_label}_posterior_summary.csv" if SAVE_TABLE_CSVS else None
    condition_csv = OUTPUT_DIR / "condition_table.csv" if SAVE_TABLE_CSVS else None
    loss_csv = OUTPUT_DIR / f"{active_label}_loss.csv" if SAVE_TABLE_CSVS else None
    convergence_csv = (
        OUTPUT_DIR / f"{active_label}_convergence_checks.csv"
        if SAVE_TABLE_CSVS and active_convergence_df is not None
        else None
    )
    if SAVE_TABLE_CSVS:
        posterior_summary_df.to_csv(summary_csv, index=False)
        condition_table.to_csv(condition_csv, index=False)
        pd.DataFrame({"step": np.arange(len(active_result.losses)), "loss": np.asarray(active_result.losses)}).to_csv(
            loss_csv,
            index=False,
        )
        if active_convergence_df is not None:
            active_convergence_df.to_csv(convergence_csv, index=False)

    loss_png = OUTPUT_DIR / f"{active_label}_loss.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.asarray(active_result.losses), lw=1.0)
    if active_convergence_df is not None and len(active_convergence_df):
        ax.plot(
            active_convergence_df["end_step"].to_numpy(dtype=float) - 1,
            active_convergence_df["mean_loss"].to_numpy(dtype=float),
            marker="o",
            ms=3,
            lw=1.2,
            label="window mean",
        )
        ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("SVI step")
    ax.set_ylabel("negative ELBO")
    ax.set_title(f"{active_label} loss")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(loss_png, dpi=200)

    global_corner_samples = np.column_stack(
        [
            posterior_np["rate_lambda"],
            1e3 * posterior_np["T_0"],
            posterior_np["theta_E"],
            posterior_np["w"],
            1e3 * posterior_np["del_go"],
            posterior_np["rate_norm_l"],
            posterior_np["alpha"],
        ]
    )
    global_labels = [
        "lambda",
        "T_0 (ms)",
        "theta_E",
        "w",
        "del_go (ms)",
        "rate_norm_l",
        "alpha",
    ]
    global_corner_png = OUTPUT_DIR / f"{active_label}_global_corner.png"
    clean_global_corner_samples, global_corner_ranges = finite_corner_input(
        global_corner_samples,
        global_labels,
        "global corner",
    )
    if clean_global_corner_samples is not None:
        fig = corner.corner(
            clean_global_corner_samples,
            labels=global_labels,
            show_titles=True,
            quantiles=[0.025, 0.5, 0.975],
            title_fmt=".3f",
            range=global_corner_ranges,
        )
        fig.suptitle(f"{BATCH_NAME}/{ANIMAL} {active_label} global posterior", y=1.02)
        fig.savefig(global_corner_png, dpi=200, bbox_inches="tight")
    else:
        global_corner_png = None

    selected_conditions = [(20.0, -1.0), (20.0, 1.0), (60.0, -16.0), (60.0, 1.0), (60.0, 16.0)]
    selected_ids = []
    selected_labels = []
    for abl, ild in selected_conditions:
        match = condition_table[(condition_table["ABL"] == abl) & (condition_table["ILD"] == ild)]
        if len(match) == 1:
            condition_id = int(match["condition_id"].iloc[0])
            selected_ids.append(condition_id)
            selected_labels.append(f"tE ABL{int(abl)} ILD{ild:g} (ms)")
    if selected_ids:
        selected_delay_samples = 1e3 * posterior_np["t_E_aff"][:, selected_ids]
        selected_corner_samples = np.column_stack(
            [
                posterior_np["rate_lambda"],
                1e3 * posterior_np["T_0"],
                posterior_np["theta_E"],
                1e3 * posterior_np["del_go"],
                posterior_np["alpha"],
                selected_delay_samples,
            ]
        )
        selected_corner_labels = ["lambda", "T_0 (ms)", "theta_E", "del_go (ms)", "alpha"] + selected_labels
        selected_corner_png = OUTPUT_DIR / f"{active_label}_global_selected_delay_corner.png"
        clean_selected_corner_samples, selected_corner_ranges = finite_corner_input(
            selected_corner_samples,
            selected_corner_labels,
            "selected-delay corner",
        )
        if clean_selected_corner_samples is not None:
            fig = corner.corner(
                clean_selected_corner_samples,
                labels=selected_corner_labels,
                show_titles=True,
                quantiles=[0.025, 0.5, 0.975],
                title_fmt=".3f",
                range=selected_corner_ranges,
            )
            fig.suptitle(f"{BATCH_NAME}/{ANIMAL} {active_label} selected-delay posterior", y=1.02)
            fig.savefig(selected_corner_png, dpi=200, bbox_inches="tight")
        else:
            selected_corner_png = None
    else:
        selected_corner_png = None

    delay_values_ms = 1e3 * posterior_np["t_E_aff"]
    finite_delay_cols = np.any(np.isfinite(delay_values_ms), axis=0)
    delay_q025 = np.full(delay_values_ms.shape[1], np.nan)
    delay_q500 = np.full(delay_values_ms.shape[1], np.nan)
    delay_q975 = np.full(delay_values_ms.shape[1], np.nan)
    for delay_idx in np.flatnonzero(finite_delay_cols):
        values = delay_values_ms[:, delay_idx]
        values = values[np.isfinite(values)]
        delay_q025[delay_idx], delay_q500[delay_idx], delay_q975[delay_idx] = np.quantile(values, [0.025, 0.5, 0.975])
    delay_init_ms = 1e3 * condition_table["t_E_aff_s"].to_numpy(dtype=float)

    delay_png = OUTPUT_DIR / f"{active_label}_condition_delay_intervals.png"
    if np.any(finite_delay_cols):
        fig, ax = plt.subplots(figsize=(8, 5))
        abl_colors = {20.0: "tab:blue", 40.0: "tab:orange", 60.0: "tab:green"}
        abl_offsets = {20.0: -0.12, 40.0: 0.0, 60.0: 0.12}
        for abl in HARD_CODED_DELAY_ABL_LEVELS:
            mask = condition_table["ABL"].to_numpy(dtype=float) == abl
            x = condition_table.loc[mask, "ILD"].to_numpy(dtype=float)
            order = np.argsort(x)
            ids = condition_table.loc[mask, "condition_id"].to_numpy(dtype=int)[order]
            x = x[order]
            finite_ids = ids[np.isfinite(delay_q500[ids])]
            finite_x = x[np.isfinite(delay_q500[ids])]
            color = abl_colors.get(float(abl), None)
            x_offset = abl_offsets.get(float(abl), 0.0)
            if len(finite_ids):
                ax.errorbar(
                    finite_x + x_offset,
                    delay_q500[finite_ids],
                    yerr=np.vstack(
                        [
                            delay_q500[finite_ids] - delay_q025[finite_ids],
                            delay_q975[finite_ids] - delay_q500[finite_ids],
                        ]
                    ),
                    fmt="o",
                    capsize=2,
                    color=color,
                    linestyle="none",
                    label=f"ABL {int(abl)} SVI 95% CI",
                )
            ax.scatter(
                x + x_offset,
                delay_init_ms[ids],
                marker="x",
                color=color,
                alpha=0.45,
                s=35,
                label=f"ABL {int(abl)} init",
            )
        ax.axhline(0, color="0.85", lw=0.8)
        ax.set_xlabel("ILD")
        ax.set_ylabel("t_E_aff (ms)")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8, ncol=2)
        fig.suptitle(f"{BATCH_NAME}/{ANIMAL} {active_label} condition-delay intervals")
        fig.tight_layout()
        fig.savefig(delay_png, dpi=200)
    else:
        print("Delay interval plot skipped because all t_E_aff posterior samples are non-finite.")
        delay_png = None

    print("\nSaved posterior outputs:")
    print(f"  samples: {sample_npz}")
    if SAVE_TABLE_CSVS:
        print(f"  summary: {summary_csv}")
        print(f"  finite report: {finite_report_csv}")
        print(f"  condition table: {condition_csv}")
        print(f"  loss CSV: {loss_csv}")
    else:
        print("  CSV tables: skipped (set NUMPYRO_SVI_SAVE_TABLE_CSVS=1 to write them)")
    if convergence_csv is not None:
        print(f"  convergence checks: {convergence_csv}")
    print(f"  loss plot: {loss_png}")
    if global_corner_png is not None:
        print(f"  global corner: {global_corner_png}")
    if selected_corner_png is not None:
        print(f"  selected-delay corner: {selected_corner_png}")
    if delay_png is not None:
        print(f"  delay intervals: {delay_png}")

# %%
