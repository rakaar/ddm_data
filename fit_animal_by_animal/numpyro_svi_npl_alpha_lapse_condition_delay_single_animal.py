# %%
"""
NumPyro SVI fit for one NPL+alpha+lapse animal with condition-wise t_E_aff.

The fitted parameters are:

    rate_lambda, T_0, theta_E, alpha, rate_norm_l, w, del_go,
    lapse_prob, lapse_prob_right, t_E_aff[condition]

This script starts from the completed patience-12 NPL+alpha condition-delay SVI
fit and from the completed IPL+lapse SVI lapse estimates, then lets all NPL,
lapse, and condition-delay parameters move together.
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

BATCH_NAME = os.environ.get("NUMPYRO_SVI_BATCH", "LED8")
ANIMAL = int(os.environ.get("NUMPYRO_SVI_ANIMAL", "105"))

GUIDE_KIND = os.environ.get("NUMPYRO_SVI_GUIDE", "fullrank")
IAF_HIDDEN_DIMS = tuple(
    int(part)
    for part in os.environ.get("NUMPYRO_SVI_IAF_HIDDEN_DIMS", "64,64").replace("x", ",").split(",")
    if part.strip()
)
IAF_NUM_FLOWS = int(os.environ.get("NUMPYRO_SVI_IAF_NUM_FLOWS", "1"))
IAF_SKIP_CONNECTIONS = os.environ.get("NUMPYRO_SVI_IAF_SKIP_CONNECTIONS", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}
IAF_BASE_SCALE = float(os.environ.get("NUMPYRO_SVI_IAF_BASE_SCALE", "0.05"))
LOWRANK_RANK = int(os.environ.get("NUMPYRO_SVI_LOWRANK_RANK", "10"))
USE_REFERENCE_POSTERIOR_COV_INIT = os.environ.get(
    "NUMPYRO_SVI_USE_REFERENCE_POSTERIOR_COV_INIT",
    "1",
).strip().lower() in {"1", "true", "yes"}
REFERENCE_LATENT_DIAG_SCALE = float(os.environ.get("NUMPYRO_SVI_REFERENCE_LATENT_DIAG_SCALE", "0.05"))

RUN_MAIN_SVI = os.environ.get("RUN_MAIN_SVI", "1").strip().lower() in {"1", "true", "yes"}
MAIN_N_TRIALS_OVERRIDE = int(os.environ.get("MAIN_N_TRIALS_OVERRIDE", "0"))
MAIN_STEPS = int(os.environ.get("MAIN_STEPS", "100000"))
SVI_CHECK_EVERY = int(os.environ.get("SVI_CHECK_EVERY", "1000"))
SVI_EARLY_STOP = os.environ.get("SVI_EARLY_STOP", "1").strip().lower() in {"1", "true", "yes"}
SVI_REL_TOL = float(os.environ.get("SVI_REL_TOL", "0.001"))
SVI_PATIENCE_WINDOWS = int(os.environ.get("SVI_PATIENCE_WINDOWS", "12"))
SVI_MIN_IMPROVEMENT_REL = float(os.environ.get("SVI_MIN_IMPROVEMENT_REL", "0.001"))
SVI_NO_IMPROVE_PATIENCE_WINDOWS = int(os.environ.get("SVI_NO_IMPROVE_PATIENCE_WINDOWS", "12"))
SVI_MIN_STEPS = int(os.environ.get("SVI_MIN_STEPS", "50000"))
SVI_STOP_MODE = os.environ.get("SVI_STOP_MODE", "patience_restore_best").strip().lower()
if SVI_STOP_MODE not in {"legacy", "stable_or_no_improve", "patience_restore_best"}:
    raise ValueError(
        "SVI_STOP_MODE must be 'legacy', 'stable_or_no_improve', or "
        f"'patience_restore_best', got {SVI_STOP_MODE!r}."
    )
SVI_STABLE_UPDATE = os.environ.get("SVI_STABLE_UPDATE", "1").strip().lower() in {"1", "true", "yes"}
LEARNING_RATE = float(os.environ.get("NUMPYRO_SVI_LR", "0.0002"))
OPTIMIZER_KIND = os.environ.get("NUMPYRO_SVI_OPTIMIZER", "clipped_adam")
CLIP_NORM = float(os.environ.get("NUMPYRO_SVI_CLIP_NORM", "1.0"))
POSTERIOR_N_SAMPLES = int(os.environ.get("POSTERIOR_N_SAMPLES", "10000"))
RNG_SEED = int(os.environ.get("NUMPYRO_SVI_SEED", "0"))
K_MAX = int(os.environ.get("K_MAX", "10"))
INIT_MODE = os.environ.get("NUMPYRO_SVI_INIT_MODE", "reference").strip().lower()
if INIT_MODE not in {"reference", "random_plausible", "random_hard"}:
    raise ValueError(
        "NUMPYRO_SVI_INIT_MODE must be 'reference', 'random_plausible', "
        f"or 'random_hard', got {INIT_MODE!r}."
    )

ABLS = [20.0, 40.0, 60.0]
BATCH_T_TRUNC = {"LED34_even": 0.15}
DEFAULT_T_TRUNC = 0.3

BATCH_CSV = REPO_DIR / "raw_data" / "batch_csvs" / f"batch_{BATCH_NAME}_valid_and_aborts.csv"
ABORT_RESULT_PKL = REPO_DIR / "aborts_ipl_npl_time_fit_results" / f"results_{BATCH_NAME}_animal_{ANIMAL}.pkl"

NPL_REFERENCE_ROOT = Path(
    os.environ.get(
        "NPL_REFERENCE_ROOT",
        str(SCRIPT_DIR / "numpyro_svi_npl_alpha_condition_delay_patience12_restore_best_outputs"),
    )
).expanduser()
NPL_REFERENCE_DIR = NPL_REFERENCE_ROOT / f"{BATCH_NAME}_{ANIMAL}"
NPL_REFERENCE_SUMMARY = Path(
    os.environ.get("NPL_REFERENCE_SUMMARY", str(NPL_REFERENCE_DIR / "main_fullrank_posterior_summary.csv"))
).expanduser()
NPL_REFERENCE_SAMPLES = Path(
    os.environ.get("NPL_REFERENCE_SAMPLES", str(NPL_REFERENCE_DIR / "main_fullrank_posterior_samples.npz"))
).expanduser()
NPL_REFERENCE_CONDITION_TABLE = Path(
    os.environ.get("NPL_REFERENCE_CONDITION_TABLE", str(NPL_REFERENCE_DIR / "condition_table.csv"))
).expanduser()

LAPSE_REFERENCE_ROOT = Path(
    os.environ.get(
        "LAPSE_REFERENCE_ROOT",
        str(SCRIPT_DIR / "numpyro_svi_vanilla_lapse_condition_delay_patience12_min50k_restore_best_outputs"),
    )
).expanduser()
LAPSE_REFERENCE_DIR = LAPSE_REFERENCE_ROOT / f"{BATCH_NAME}_{ANIMAL}"
LAPSE_REFERENCE_SUMMARY = Path(
    os.environ.get("LAPSE_REFERENCE_SUMMARY", str(LAPSE_REFERENCE_DIR / "main_fullrank_posterior_summary.csv"))
).expanduser()
LAPSE_REFERENCE_SAMPLES = Path(
    os.environ.get("LAPSE_REFERENCE_SAMPLES", str(LAPSE_REFERENCE_DIR / "main_fullrank_posterior_samples.npz"))
).expanduser()

OUTPUT_ROOT = Path(
    os.environ.get(
        "NUMPYRO_SVI_OUTPUT_ROOT",
        str(SCRIPT_DIR / "numpyro_svi_npl_alpha_lapse_condition_delay_single_animal_outputs"),
    )
).expanduser()
OUTPUT_DIR = OUTPUT_ROOT / f"{BATCH_NAME}_{ANIMAL}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NPL_SCALAR_PARAM_NAMES = [
    "rate_lambda",
    "T_0",
    "theta_E",
    "w",
    "del_go",
    "rate_norm_l",
    "alpha",
]


# %%
# =============================================================================
# Dependency preflight
# =============================================================================
required_modules = ["jax", "jaxlib", "numpyro"]
missing_modules = [module for module in required_modules if importlib.util.find_spec(module) is None]
if missing_modules:
    print("Missing dependencies for NumPyro NPL+alpha+lapse SVI:")
    for module in missing_modules:
        print(f"  - {module}")
    raise SystemExit(1)


# %%
# =============================================================================
# Imports after dependency preflight
# =============================================================================
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import pandas as pd
from jax import random
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.svi import SVIRunResult
from numpyro.infer.util import log_density

sys.path.insert(0, str(SCRIPT_DIR))
import numpyro_npl_alpha_lapse_svi_utils as lapse_utils


# %%
# =============================================================================
# Small helpers
# =============================================================================
def ensure_choice_column(df):
    if "choice" not in df.columns:
        if "response_poke" not in df.columns:
            raise KeyError("Need either `choice` or `response_poke` in the batch CSV.")
        df = df.copy()
        df["choice"] = df["response_poke"].map({3: 1, 2: -1})
    return df


def read_summary_mean(summary_df, parameter):
    rows = summary_df[summary_df["parameter"].astype(str) == parameter]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one summary row for {parameter!r}, found {len(rows)}.")
    return float(rows.iloc[0]["mean"])


def bounded_latent(values, hard_low, hard_high):
    values = np.asarray(values, dtype=float)
    eps = 1e-6 * (hard_high - hard_low)
    values = np.clip(values, hard_low + eps, hard_high - eps)
    unit_values = (values - hard_low) / (hard_high - hard_low)
    return np.log(unit_values / (1.0 - unit_values))


def make_optimizer():
    optimizer_kind = OPTIMIZER_KIND.strip().lower()
    if optimizer_kind in {"adam", "plain_adam"}:
        return numpyro.optim.Adam(LEARNING_RATE)
    if optimizer_kind in {"clipped_adam", "clipped-adam", "clip_adam"}:
        return numpyro.optim.ClippedAdam(LEARNING_RATE, clip_norm=CLIP_NORM)
    raise ValueError(f"Unknown NUMPYRO_SVI_OPTIMIZER={OPTIMIZER_KIND!r}")


def make_jax_data(df, V_A, theta_A, t_A_aff, T_trunc):
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
        "lapse_rt_window": jnp.asarray(1.0, dtype=jnp.float64),
    }


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

        can_stop_for_patience = completed_steps >= SVI_MIN_STEPS

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
                "can_stop_for_patience": can_stop_for_patience,
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
            f"best_chunk={best_window_chunk}, can_stop={can_stop_for_patience}, nonfinite={n_nonfinite}"
        )

        if n_nonfinite:
            print(
                f"WARNING: stopping {run_label} because this window had non-finite losses. "
                f"Returning best state from chunk {best_window_chunk} ending at step {best_window_end_step}."
            )
            break
        if not SVI_EARLY_STOP or not can_stop_for_patience:
            prev_window_mean = window_mean
            continue

        if SVI_STOP_MODE in {"legacy", "stable_or_no_improve"} and stable_window_count >= SVI_PATIENCE_WINDOWS:
            print(
                f"Stopping {run_label} early at step {completed_steps}: "
                f"{SVI_PATIENCE_WINDOWS} consecutive windows changed by <= {100.0 * SVI_REL_TOL:.3g}%."
            )
            break
        if SVI_STOP_MODE in {"stable_or_no_improve", "patience_restore_best"} and (
            no_improve_window_count >= SVI_NO_IMPROVE_PATIENCE_WINDOWS
        ):
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
# Load data and reference initial values
# =============================================================================
print(f"Batch/animal: {BATCH_NAME}/{ANIMAL}")
print(f"Batch CSV: {BATCH_CSV}")
print(f"Abort params: {ABORT_RESULT_PKL}")
print(f"NPL reference summary: {NPL_REFERENCE_SUMMARY}")
print(f"NPL reference samples: {NPL_REFERENCE_SAMPLES}")
print(f"IPL+lapse reference summary: {LAPSE_REFERENCE_SUMMARY}")
print(f"IPL+lapse reference samples: {LAPSE_REFERENCE_SAMPLES}")
print(f"Output folder: {OUTPUT_DIR}")

for required_path in [
    BATCH_CSV,
    ABORT_RESULT_PKL,
    NPL_REFERENCE_SUMMARY,
    NPL_REFERENCE_CONDITION_TABLE,
    LAPSE_REFERENCE_SUMMARY,
]:
    if not required_path.exists():
        raise FileNotFoundError(required_path)

raw_df = ensure_choice_column(pd.read_csv(BATCH_CSV))
valid_df = raw_df[
    (raw_df["animal"].astype(int) == ANIMAL)
    & (raw_df["success"].isin([1, -1]))
    & (raw_df["RTwrtStim"] < 1)
    & (raw_df["ABL"].isin(ABLS))
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

with ABORT_RESULT_PKL.open("rb") as handle:
    abort_saved = pickle.load(handle)
abort_results = abort_saved["vbmc_aborts_results"]
V_A = float(np.mean(abort_results["V_A_samples"]))
theta_A = float(np.mean(abort_results["theta_A_samples"]))
t_A_aff = float(np.mean(abort_results["t_A_aff_samp"]))

npl_reference_summary_df = pd.read_csv(NPL_REFERENCE_SUMMARY)
lapse_reference_summary_df = pd.read_csv(LAPSE_REFERENCE_SUMMARY)
npl_reference_condition_df = pd.read_csv(NPL_REFERENCE_CONDITION_TABLE)
npl_reference_condition_df["ABL"] = npl_reference_condition_df["ABL"].astype(float)
npl_reference_condition_df["ILD"] = npl_reference_condition_df["ILD"].astype(float)
npl_reference_condition_df = npl_reference_condition_df.rename(
    columns={"condition_id": "npl_reference_condition_id"}
)

npl_delay_summary_df = npl_reference_summary_df[
    npl_reference_summary_df["parameter"].astype(str).str.startswith("t_E_aff_")
][["ABL", "ILD", "mean", "q025", "q500", "q975"]].copy()
npl_delay_summary_df["ABL"] = npl_delay_summary_df["ABL"].astype(float)
npl_delay_summary_df["ILD"] = npl_delay_summary_df["ILD"].astype(float)
npl_delay_summary_df = npl_delay_summary_df.rename(
    columns={
        "mean": "npl_reference_t_E_aff_s",
        "q025": "npl_reference_t_E_aff_q025_s",
        "q500": "npl_reference_t_E_aff_q500_s",
        "q975": "npl_reference_t_E_aff_q975_s",
    }
)

condition_table = condition_table.merge(
    npl_delay_summary_df,
    on=["ABL", "ILD"],
    how="left",
    validate="one_to_one",
)
condition_table = condition_table.merge(
    npl_reference_condition_df[["ABL", "ILD", "npl_reference_condition_id"]],
    on=["ABL", "ILD"],
    how="left",
    validate="one_to_one",
)
if condition_table["npl_reference_t_E_aff_s"].isna().any():
    missing = condition_table[condition_table["npl_reference_t_E_aff_s"].isna()]
    raise RuntimeError(f"Missing NPL reference delay rows:\n{missing}")
if condition_table["npl_reference_condition_id"].isna().any():
    missing = condition_table[condition_table["npl_reference_condition_id"].isna()]
    raise RuntimeError(f"Missing NPL reference condition IDs:\n{missing}")
condition_table = condition_table.sort_values("condition_id").reset_index(drop=True)

init_values = dict(lapse_utils.DEFAULT_INIT_VALUES)
if INIT_MODE == "reference":
    for param_name in NPL_SCALAR_PARAM_NAMES:
        init_values[param_name] = read_summary_mean(npl_reference_summary_df, param_name)
    init_values["lapse_prob"] = read_summary_mean(lapse_reference_summary_df, "lapse_prob")
    init_values["lapse_prob_right"] = read_summary_mean(lapse_reference_summary_df, "lapse_prob_right")
    init_values["t_E_aff"] = condition_table["npl_reference_t_E_aff_s"].to_numpy(dtype=float)
else:
    init_rng = np.random.default_rng(RNG_SEED + 1000)
    bound_key = "plausible" if INIT_MODE == "random_plausible" else "hard"
    for param_name, bounds in lapse_utils.GLOBAL_BOUNDS.items():
        low, high = bounds[bound_key]
        init_values[param_name] = float(init_rng.uniform(low, high))
    delay_low, delay_high = lapse_utils.DELAY_BOUNDS[bound_key]
    init_values["t_E_aff"] = init_rng.uniform(delay_low, delay_high, size=len(condition_table))
init_values = lapse_utils.clip_init_to_hard_bounds(init_values)

fullrank_init_scale_tril = None
effective_use_reference_cov_init = USE_REFERENCE_POSTERIOR_COV_INIT and INIT_MODE == "reference"
if GUIDE_KIND.strip().lower() in {"fullrank", "multivariate", "automultivariate"} and effective_use_reference_cov_init:
    latent_dim = len(lapse_utils.GLOBAL_PARAM_NAMES) + int(len(condition_table))
    init_cov = np.eye(latent_dim) * (REFERENCE_LATENT_DIAG_SCALE**2)

    if NPL_REFERENCE_SAMPLES.exists():
        with np.load(NPL_REFERENCE_SAMPLES) as npl_samples:
            npl_latent_columns = []
            for param_name in NPL_SCALAR_PARAM_NAMES:
                hard_low, hard_high = lapse_utils.GLOBAL_BOUNDS[param_name]["hard"]
                npl_latent_columns.append(bounded_latent(npl_samples[param_name], hard_low, hard_high))

            delay_hard_low, delay_hard_high = lapse_utils.DELAY_BOUNDS["hard"]
            delay_samples = npl_samples["t_E_aff"][
                :,
                condition_table["npl_reference_condition_id"].to_numpy(dtype=int),
            ]
            npl_latent_columns.extend(
                bounded_latent(delay_samples[:, idx], delay_hard_low, delay_hard_high)
                for idx in range(delay_samples.shape[1])
            )
            npl_latent_samples = np.column_stack(npl_latent_columns)
            npl_cov = np.cov(npl_latent_samples, rowvar=False)
            npl_indices = list(range(len(NPL_SCALAR_PARAM_NAMES))) + list(
                range(len(lapse_utils.GLOBAL_PARAM_NAMES), latent_dim)
            )
            init_cov[np.ix_(npl_indices, npl_indices)] = npl_cov
            npl_latent_corr = np.corrcoef(
                npl_latent_samples[:, NPL_SCALAR_PARAM_NAMES.index("T_0")],
                npl_latent_samples[:, NPL_SCALAR_PARAM_NAMES.index("theta_E")],
            )[0, 1]
            print("\nFull-rank guide covariance initialized from NPL reference posterior:")
            print(f"  latent corr(T_0, theta_E) = {npl_latent_corr:.3f}")
    else:
        print("\nNPL reference samples not found; using diagonal full-rank initialization for NPL block.")

    if LAPSE_REFERENCE_SAMPLES.exists():
        with np.load(LAPSE_REFERENCE_SAMPLES) as lapse_samples:
            for param_name in ["lapse_prob", "lapse_prob_right"]:
                hard_low, hard_high = lapse_utils.GLOBAL_BOUNDS[param_name]["hard"]
                latent_values = bounded_latent(lapse_samples[param_name], hard_low, hard_high)
                param_index = lapse_utils.GLOBAL_PARAM_NAMES.index(param_name)
                init_cov[param_index, param_index] = float(np.var(latent_values, ddof=1))
        print("Full-rank guide lapse variances initialized from IPL+lapse posterior.")
    else:
        print("IPL+lapse reference samples not found; using diagonal initialization for lapse block.")

    init_cov += np.eye(latent_dim) * 1e-8
    fullrank_init_scale_tril = np.linalg.cholesky(init_cov)

T_trunc = BATCH_T_TRUNC.get(BATCH_NAME, DEFAULT_T_TRUNC)
full_data = make_jax_data(valid_df, V_A, theta_A, t_A_aff, T_trunc)
if MAIN_N_TRIALS_OVERRIDE > 0 and len(valid_df) > MAIN_N_TRIALS_OVERRIDE:
    main_df = valid_df.sample(MAIN_N_TRIALS_OVERRIDE, random_state=RNG_SEED + 1).sort_index().copy()
else:
    main_df = valid_df.copy()
main_data = make_jax_data(main_df, V_A, theta_A, t_A_aff, T_trunc)
n_conditions = int(len(condition_table))

print(f"\nValid fitting trials: {len(valid_df)}")
print(f"Main fitting trials: {len(main_df)}")
print(f"Observed conditions: {n_conditions}")
print(
    condition_table[
        ["condition_id", "ABL", "ILD", "npl_reference_condition_id", "npl_reference_t_E_aff_s"]
    ].to_string(index=False)
)
print("\nAbort/proactive posterior means:")
print(f"  V_A      = {V_A:.6g}")
print(f"  theta_A  = {theta_A:.6g}")
print(f"  t_A_aff  = {1e3 * t_A_aff:.3f} ms")
print(f"  T_trunc  = {T_trunc:.3f} s")
print("\nInitial global values:")
print(f"  init_mode        = {INIT_MODE}")
for param_name in lapse_utils.GLOBAL_PARAM_NAMES:
    value = init_values[param_name]
    if param_name in {"T_0", "del_go"}:
        print(f"  {param_name:<16} = {1e3 * value:.5f} ms")
    elif param_name == "lapse_prob":
        print(f"  {param_name:<16} = {100.0 * value:.4f}%")
    else:
        print(f"  {param_name:<16} = {value:.6g}")
print(f"Guide: {GUIDE_KIND}")
if GUIDE_KIND.strip().lower() in {"fullrank", "multivariate", "automultivariate"}:
    print(
        "Full-rank config: "
        f"use_reference_cov_init={effective_use_reference_cov_init}, "
        f"reference_latent_diag_scale={REFERENCE_LATENT_DIAG_SCALE:g}"
    )
print(f"Optimizer: {OPTIMIZER_KIND}, learning_rate={LEARNING_RATE:g}, clip_norm={CLIP_NORM:g}")
print(
    "SVI convergence checks: "
    f"every {SVI_CHECK_EVERY} steps, early_stop={SVI_EARLY_STOP}, stop_mode={SVI_STOP_MODE}, "
    f"min_steps={SVI_MIN_STEPS}, rel_tol={SVI_REL_TOL:g}, patience_windows={SVI_PATIENCE_WINDOWS}, "
    f"min_improvement_rel={SVI_MIN_IMPROVEMENT_REL:g}, "
    f"no_improve_patience={SVI_NO_IMPROVE_PATIENCE_WINDOWS}, stable_update={SVI_STABLE_UPDATE}"
)


# %%
# =============================================================================
# Initial log joint and gradients
# =============================================================================
model = lambda data, n_conditions: lapse_utils.npl_alpha_lapse_condition_delay_model(
    data,
    n_conditions,
    K_max=K_MAX,
)


def log_joint_from_values(values, data):
    log_joint, _ = log_density(model, (data, n_conditions), {}, values)
    return log_joint


initial_main_log_joint = log_joint_from_values(init_values, main_data)
initial_full_log_joint = log_joint_from_values(init_values, full_data)
initial_grad = jax.grad(lambda values: log_joint_from_values(values, main_data))(init_values)

print("\nInitial log joint:")
print(f"  main = {float(initial_main_log_joint):.6f}")
print(f"  full = {float(initial_full_log_joint):.6f}")
print(f"Initial main gradient finite: {lapse_utils.tree_all_finite(initial_grad)}")
if not np.isfinite(float(initial_main_log_joint)) or not lapse_utils.tree_all_finite(initial_grad):
    raise RuntimeError("Initial log joint or gradients are non-finite.")


# %%
# =============================================================================
# Main SVI
# =============================================================================
active_result = None
active_guide = None
active_label = None
active_convergence_df = None

if RUN_MAIN_SVI:
    print(f"\nRunning {GUIDE_KIND} main SVI for {MAIN_STEPS} steps on {len(main_df)} trials...")
    main_guide = lapse_utils.make_guide(
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
    active_convergence_df = main_convergence_df
else:
    print("\nSkipping main SVI because RUN_MAIN_SVI=False.")


# %%
# =============================================================================
# Posterior samples, comparison table, and saved outputs
# =============================================================================
if active_result is None:
    print("\nNo SVI result is available; set RUN_MAIN_SVI=1 to save outputs.")
else:
    posterior_samples = active_guide.sample_posterior(
        random.PRNGKey(RNG_SEED + 2),
        active_result.params,
        sample_shape=(POSTERIOR_N_SAMPLES,),
    )
    posterior_np = {key: np.asarray(value) for key, value in posterior_samples.items()}
    guide_params_np = lapse_utils.tree_to_numpy(active_result.params)

    finite_report_df, all_posterior_finite = lapse_utils.finite_sample_report(posterior_np)
    posterior_summary_df = lapse_utils.posterior_samples_to_frame(posterior_np, condition_table)

    comparison_rows = []
    for param_name in NPL_SCALAR_PARAM_NAMES:
        reference_mean = read_summary_mean(npl_reference_summary_df, param_name)
        fitted_mean = read_summary_mean(posterior_summary_df, param_name)
        scale = 1000.0 if param_name in {"T_0", "del_go"} else 1.0
        unit = "ms" if param_name in {"T_0", "del_go"} else ""
        comparison_rows.append(
            {
                "parameter": param_name,
                "parameter_group": "global",
                "reference_source": "npl_alpha_condition_delay_patience12",
                "reference_mean_raw": reference_mean,
                "npl_alpha_lapse_mean_raw": fitted_mean,
                "delta_raw": fitted_mean - reference_mean,
                "display_unit": unit,
                "reference_display": scale * reference_mean,
                "npl_alpha_lapse_display": scale * fitted_mean,
                "delta_display": scale * (fitted_mean - reference_mean),
            }
        )

    for param_name in ["lapse_prob", "lapse_prob_right"]:
        reference_mean = read_summary_mean(lapse_reference_summary_df, param_name)
        fitted_mean = read_summary_mean(posterior_summary_df, param_name)
        scale = 100.0 if param_name == "lapse_prob" else 1.0
        unit = "%" if param_name == "lapse_prob" else ""
        comparison_rows.append(
            {
                "parameter": param_name,
                "parameter_group": "lapse",
                "reference_source": "ipl_lapse_condition_delay_patience12_min50k",
                "reference_mean_raw": reference_mean,
                "npl_alpha_lapse_mean_raw": fitted_mean,
                "delta_raw": fitted_mean - reference_mean,
                "display_unit": unit,
                "reference_display": scale * reference_mean,
                "npl_alpha_lapse_display": scale * fitted_mean,
                "delta_display": scale * (fitted_mean - reference_mean),
            }
        )

    fitted_delay_summary = posterior_summary_df[
        posterior_summary_df["parameter"].astype(str).str.startswith("t_E_aff_")
    ][["ABL", "ILD", "mean"]].copy()
    fitted_delay_summary["ABL"] = fitted_delay_summary["ABL"].astype(float)
    fitted_delay_summary["ILD"] = fitted_delay_summary["ILD"].astype(float)
    fitted_delay_summary = fitted_delay_summary.rename(columns={"mean": "npl_alpha_lapse_t_E_aff_s"})
    delay_comparison_df = condition_table[["condition_id", "ABL", "ILD", "npl_reference_t_E_aff_s"]].merge(
        fitted_delay_summary,
        on=["ABL", "ILD"],
        how="left",
        validate="one_to_one",
    )
    for _, row in delay_comparison_df.iterrows():
        comparison_rows.append(
            {
                "parameter": f"t_E_aff_ABL{int(row['ABL'])}_ILD{row['ILD']:g}",
                "parameter_group": "condition_delay",
                "reference_source": "npl_alpha_condition_delay_patience12",
                "reference_mean_raw": float(row["npl_reference_t_E_aff_s"]),
                "npl_alpha_lapse_mean_raw": float(row["npl_alpha_lapse_t_E_aff_s"]),
                "delta_raw": float(row["npl_alpha_lapse_t_E_aff_s"] - row["npl_reference_t_E_aff_s"]),
                "display_unit": "ms",
                "reference_display": 1000.0 * float(row["npl_reference_t_E_aff_s"]),
                "npl_alpha_lapse_display": 1000.0 * float(row["npl_alpha_lapse_t_E_aff_s"]),
                "delta_display": 1000.0 * float(row["npl_alpha_lapse_t_E_aff_s"] - row["npl_reference_t_E_aff_s"]),
                "ABL": float(row["ABL"]),
                "ILD": float(row["ILD"]),
                "condition_id": int(row["condition_id"]),
            }
        )
    comparison_df = pd.DataFrame(comparison_rows)

    sample_npz = OUTPUT_DIR / f"{active_label}_posterior_samples.npz"
    guide_params_pkl = OUTPUT_DIR / f"{active_label}_guide_params.pkl"
    summary_csv = OUTPUT_DIR / f"{active_label}_posterior_summary.csv"
    finite_report_csv = OUTPUT_DIR / f"{active_label}_posterior_finite_report.csv"
    condition_csv = OUTPUT_DIR / "condition_table.csv"
    comparison_csv = OUTPUT_DIR / f"{active_label}_reference_param_comparison.csv"
    loss_csv = OUTPUT_DIR / f"{active_label}_loss.csv"
    convergence_csv = OUTPUT_DIR / f"{active_label}_convergence_checks.csv"
    bundle_pkl = OUTPUT_DIR / f"{active_label}_variational_posterior_bundle.pkl"

    np.savez(sample_npz, **posterior_np)
    with guide_params_pkl.open("wb") as handle:
        pickle.dump(guide_params_np, handle)
    posterior_summary_df.to_csv(summary_csv, index=False)
    finite_report_df.to_csv(finite_report_csv, index=False)
    condition_table.to_csv(condition_csv, index=False)
    comparison_df.to_csv(comparison_csv, index=False)
    loss_df = pd.DataFrame({"step": np.arange(len(active_result.losses)), "loss": np.asarray(active_result.losses)})
    loss_df.to_csv(loss_csv, index=False)
    active_convergence_df.to_csv(convergence_csv, index=False)

    loss_png = OUTPUT_DIR / f"{active_label}_loss.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(loss_df["step"], loss_df["loss"], lw=1.0, color="0.15", label="step loss")
    if active_convergence_df is not None and len(active_convergence_df):
        ax.plot(
            active_convergence_df["end_step"].to_numpy(dtype=float) - 1,
            active_convergence_df["mean_loss"].to_numpy(dtype=float),
            marker="o",
            ms=3,
            lw=1.2,
            color="tab:blue",
            label="window mean",
        )
        best_step = int(active_convergence_df.iloc[-1]["best_end_step_so_far"])
        checked_step = int(active_convergence_df.iloc[-1]["end_step"])
        ax.axvline(best_step - 1, color="tab:green", lw=1.2, label="restored best")
        ax.axvline(checked_step - 1, color="tab:red", lw=1.2, ls="--", label="final checked")
    ax.set_xlabel("SVI step")
    ax.set_ylabel("negative ELBO")
    ax.set_title(f"{BATCH_NAME}/{ANIMAL} {active_label} NPL+alpha+lapse loss")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(loss_png, dpi=200)

    delay_values_ms = 1000.0 * posterior_np["t_E_aff"]
    delay_q025 = np.nanquantile(delay_values_ms, 0.025, axis=0)
    delay_q500 = np.nanquantile(delay_values_ms, 0.5, axis=0)
    delay_q975 = np.nanquantile(delay_values_ms, 0.975, axis=0)
    delay_ref_ms = 1000.0 * condition_table["npl_reference_t_E_aff_s"].to_numpy(dtype=float)

    delay_png = OUTPUT_DIR / f"{active_label}_condition_delay_intervals.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    abl_colors = {20.0: "tab:blue", 40.0: "tab:orange", 60.0: "tab:green"}
    abl_offsets = {20.0: -0.12, 40.0: 0.0, 60.0: 0.12}
    for abl in ABLS:
        mask = condition_table["ABL"].to_numpy(dtype=float) == abl
        ids = condition_table.loc[mask, "condition_id"].to_numpy(dtype=int)
        x = condition_table.loc[mask, "ILD"].to_numpy(dtype=float)
        order = np.argsort(x)
        ids = ids[order]
        x = x[order]
        color = abl_colors[abl]
        x_offset = abl_offsets[abl]
        ax.errorbar(
            x + x_offset,
            delay_q500[ids],
            yerr=np.vstack([delay_q500[ids] - delay_q025[ids], delay_q975[ids] - delay_q500[ids]]),
            fmt="o",
            capsize=2,
            color=color,
            linestyle="none",
            label=f"ABL {int(abl)} NPL+lapse 95% CI",
        )
        ax.scatter(
            x + x_offset,
            delay_ref_ms[ids],
            marker="x",
            color=color,
            alpha=0.45,
            s=35,
            label=f"ABL {int(abl)} NPL ref",
        )
    ax.axhline(0, color="0.85", lw=0.8)
    ax.set_xlabel("ILD")
    ax.set_ylabel("t_E_aff (ms)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle(f"{BATCH_NAME}/{ANIMAL} {active_label} NPL+alpha+lapse condition-delay intervals")
    fig.tight_layout()
    fig.savefig(delay_png, dpi=200)

    bundle = {
        "schema_version": 1,
        "model_name": "npl_alpha_lapse_condition_delay_svi",
        "note": (
            "NumPyro SVI fit of NPL+alpha+lapse parameters with condition-wise t_E_aff. "
            "Guide params are NumPy arrays from the fitted NumPyro autoguide; "
            "posterior_samples are sampled from that variational posterior."
        ),
        "batch_name": BATCH_NAME,
        "animal": int(ANIMAL),
        "label": active_label,
        "guide_kind": GUIDE_KIND,
        "config": {
            "main_steps": int(MAIN_STEPS),
            "check_every": int(SVI_CHECK_EVERY),
            "stop_mode": SVI_STOP_MODE,
            "rel_tol": float(SVI_REL_TOL),
            "patience_windows": int(SVI_PATIENCE_WINDOWS),
            "no_improve_patience_windows": int(SVI_NO_IMPROVE_PATIENCE_WINDOWS),
            "min_improvement_rel": float(SVI_MIN_IMPROVEMENT_REL),
            "min_steps": int(SVI_MIN_STEPS),
            "T_trunc": float(T_trunc),
            "K_max": int(K_MAX),
            "posterior_n_samples": int(POSTERIOR_N_SAMPLES),
            "rng_seed": int(RNG_SEED),
            "lapse_rt_window": 1.0,
            "init_mode": INIT_MODE,
            "reference_cov_init": bool(effective_use_reference_cov_init),
        },
        "input_paths": {
            "batch_csv": str(BATCH_CSV),
            "abort_result_pkl": str(ABORT_RESULT_PKL),
            "npl_reference_summary": str(NPL_REFERENCE_SUMMARY),
            "npl_reference_samples": str(NPL_REFERENCE_SAMPLES),
            "npl_reference_condition_table": str(NPL_REFERENCE_CONDITION_TABLE),
            "lapse_reference_summary": str(LAPSE_REFERENCE_SUMMARY),
            "lapse_reference_samples": str(LAPSE_REFERENCE_SAMPLES),
        },
        "output_paths": {
            "posterior_samples_npz": str(sample_npz),
            "guide_params_pkl": str(guide_params_pkl),
            "posterior_summary_csv": str(summary_csv),
            "finite_report_csv": str(finite_report_csv),
            "condition_table_csv": str(condition_csv),
            "comparison_csv": str(comparison_csv),
            "loss_csv": str(loss_csv),
            "convergence_csv": str(convergence_csv),
            "loss_png": str(loss_png),
            "delay_intervals_png": str(delay_png),
        },
        "abort_means": {
            "V_A": V_A,
            "theta_A": theta_A,
            "t_A_aff": t_A_aff,
        },
        "init_values": {key: np.asarray(value) for key, value in init_values.items()},
        "guide_params": guide_params_np,
        "posterior_samples": posterior_np,
        "posterior_summary": posterior_summary_df,
        "finite_report": finite_report_df,
        "condition_table": condition_table,
        "reference_param_comparison": comparison_df,
        "loss_trace": loss_df,
        "convergence_checks": active_convergence_df,
    }
    with bundle_pkl.open("wb") as handle:
        pickle.dump(bundle, handle)

    print("\nPosterior finite-sample report:")
    print(finite_report_df.to_string(index=False))
    if not all_posterior_finite:
        print("WARNING: posterior samples contain NaN/Inf.")

    print("\nGlobal parameter comparison:")
    global_display = comparison_df[comparison_df["parameter_group"].isin(["global", "lapse"])][
        [
            "parameter",
            "reference_source",
            "reference_display",
            "npl_alpha_lapse_display",
            "delta_display",
            "display_unit",
        ]
    ].copy()
    print(global_display.to_string(index=False))

    delay_deltas_ms = comparison_df.loc[
        comparison_df["parameter_group"] == "condition_delay",
        "delta_display",
    ].to_numpy(dtype=float)
    print("\nDelay comparison vs NPL reference:")
    print(f"  mean delta = {np.mean(delay_deltas_ms):.4f} ms")
    print(f"  RMSE delta = {np.sqrt(np.mean(delay_deltas_ms**2)):.4f} ms")
    print(f"  max |delta| = {np.max(np.abs(delay_deltas_ms)):.4f} ms")

    print("\nSaved outputs:")
    print(f"  samples: {sample_npz}")
    print(f"  guide params: {guide_params_pkl}")
    print(f"  summary: {summary_csv}")
    print(f"  finite report: {finite_report_csv}")
    print(f"  condition table: {condition_csv}")
    print(f"  comparison: {comparison_csv}")
    print(f"  loss CSV: {loss_csv}")
    print(f"  convergence checks: {convergence_csv}")
    print(f"  loss plot: {loss_png}")
    print(f"  delay intervals: {delay_png}")
    print(f"  VP bundle: {bundle_pkl}")

# %%
