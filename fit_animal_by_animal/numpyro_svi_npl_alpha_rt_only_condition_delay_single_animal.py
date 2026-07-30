# %%
"""
Choice-collapsed NPL+alpha SVI fit for one animal.

The fit uses successful stimulus-relative RTs in [0, 1) seconds and contains:

    rate_lambda, T_0, theta_E, w, rate_norm_l, alpha,
    t_E_aff[ABL, signed ILD]

`NPL_RT_ONLY_PROCESS_MODE` selects a pure reactive density or a fixed
proactive+reactive race. Both likelihoods collapse over choice and therefore
exclude the structurally unidentifiable `del_go`.
"""

# %%
# =============================================================================
# Editable parameters
# =============================================================================
from pathlib import Path
import importlib.util
import json
import os
import pickle
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

BATCH_NAME = os.environ.get("NUMPYRO_SVI_BATCH", "LED7")
ANIMAL = int(os.environ.get("NUMPYRO_SVI_ANIMAL", "92"))

PROCESS_MODE = (
    os.environ.get("NPL_RT_ONLY_PROCESS_MODE", "reactive_only")
    .strip()
    .lower()
    .replace("-", "_")
)
if PROCESS_MODE not in {"reactive_only", "proactive_reactive"}:
    raise ValueError(
        "NPL_RT_ONLY_PROCESS_MODE must be 'reactive_only' or "
        f"'proactive_reactive', got {PROCESS_MODE!r}."
    )
PROCESS_LABEL = PROCESS_MODE.replace("_", "-")

RT_LOWER = float(os.environ.get("NPL_RT_ONLY_RT_LOWER", "0.000"))
RT_UPPER = float(os.environ.get("NPL_RT_ONLY_RT_UPPER", "1.000"))
ABLS = [20.0, 40.0, 60.0]
EXPECTED_N_CONDITIONS = int(
    os.environ.get("NPL_RT_ONLY_EXPECTED_N_CONDITIONS", "30")
)

GUIDE_KIND = os.environ.get("NUMPYRO_SVI_GUIDE", "fullrank")
RUN_MAIN_SVI = os.environ.get("RUN_MAIN_SVI", "1").strip().lower() in {"1", "true", "yes"}
MAIN_N_TRIALS_OVERRIDE = int(os.environ.get("MAIN_N_TRIALS_OVERRIDE", "0"))
MAIN_STEPS = int(os.environ.get("MAIN_STEPS", "150000"))
SVI_CHECK_EVERY = int(os.environ.get("SVI_CHECK_EVERY", "1000"))
SVI_EARLY_STOP = os.environ.get("SVI_EARLY_STOP", "1").strip().lower() in {"1", "true", "yes"}
SVI_REL_TOL = float(os.environ.get("SVI_REL_TOL", "0.001"))
SVI_PATIENCE_WINDOWS = int(os.environ.get("SVI_PATIENCE_WINDOWS", "12"))
SVI_MIN_IMPROVEMENT_REL = float(os.environ.get("SVI_MIN_IMPROVEMENT_REL", "0.001"))
SVI_NO_IMPROVE_PATIENCE_WINDOWS = int(
    os.environ.get("SVI_NO_IMPROVE_PATIENCE_WINDOWS", "12")
)
SVI_MIN_STEPS = int(os.environ.get("SVI_MIN_STEPS", "50000"))
SVI_STOP_MODE = os.environ.get("SVI_STOP_MODE", "patience_restore_best").strip().lower()
if SVI_STOP_MODE not in {"legacy", "stable_or_no_improve", "patience_restore_best"}:
    raise ValueError(
        "SVI_STOP_MODE must be 'legacy', 'stable_or_no_improve', or "
        f"'patience_restore_best', got {SVI_STOP_MODE!r}."
    )
SVI_STABLE_UPDATE = os.environ.get("SVI_STABLE_UPDATE", "1").strip().lower() in {
    "1",
    "true",
    "yes",
}

LEARNING_RATE = float(os.environ.get("NUMPYRO_SVI_LR", "0.0002"))
OPTIMIZER_KIND = os.environ.get("NUMPYRO_SVI_OPTIMIZER", "clipped_adam")
CLIP_NORM = float(os.environ.get("NUMPYRO_SVI_CLIP_NORM", "1.0"))
POSTERIOR_N_SAMPLES = int(os.environ.get("POSTERIOR_N_SAMPLES", "10000"))
RNG_SEED = int(os.environ.get("NUMPYRO_SVI_SEED", "0"))
K_MAX = int(os.environ.get("K_MAX", "10"))
FULLRANK_COV_JITTER = float(
    os.environ.get("NPL_RT_ONLY_FULLRANK_COV_JITTER", "1e-8")
)
CORNER_MAX_SAMPLES = int(
    os.environ.get("NPL_RT_ONLY_CORNER_MAX_SAMPLES", "5000")
)

BATCH_CSV = REPO_DIR / "raw_data" / "batch_csvs" / f"batch_{BATCH_NAME}_valid_and_aborts.csv"
ABORT_RESULT_PKL = (
    REPO_DIR
    / "aborts_ipl_npl_time_fit_results"
    / f"results_{BATCH_NAME}_animal_{ANIMAL}.pkl"
)
REFERENCE_ROOT = Path(
    os.environ.get(
        "NPL_RT_ONLY_INIT_ROOT",
        str(
            SCRIPT_DIR
            / "numpyro_svi_npl_alpha_condition_delay_patience12_restore_best_outputs"
        ),
    )
).expanduser()
REFERENCE_DIR = REFERENCE_ROOT / f"{BATCH_NAME}_{ANIMAL}"
REFERENCE_NPZ = REFERENCE_DIR / "main_fullrank_posterior_samples.npz"
REFERENCE_CONDITION_CSV = REFERENCE_DIR / "condition_table.csv"

DEFAULT_OUTPUT_ROOT = (
    SCRIPT_DIR
    / (
        f"numpyro_svi_npl_alpha_rt_only_{PROCESS_MODE}_0_to_1s_"
        "condition_delay_patience12_min50k_restore_best_outputs"
    )
)
OUTPUT_ROOT = Path(
    os.environ.get(
        "NUMPYRO_SVI_OUTPUT_ROOT",
        str(DEFAULT_OUTPUT_ROOT),
    )
).expanduser()
OUTPUT_DIR = OUTPUT_ROOT / f"{BATCH_NAME}_{ANIMAL}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# %%
# =============================================================================
# Dependency preflight and imports
# =============================================================================
required_modules = ["jax", "jaxlib", "numpyro", "corner"]
missing_modules = [
    module for module in required_modules if importlib.util.find_spec(module) is None
]
if missing_modules:
    raise RuntimeError(f"Missing required Python modules: {missing_modules}")

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import corner
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
import numpyro_npl_alpha_rt_only_svi_utils as rt_utils


# %%
# =============================================================================
# Small reusable helpers
# =============================================================================
def bounded_logit(values, hard_low, hard_high, eps=1e-6):
    values = np.asarray(values, dtype=float)
    width = hard_high - hard_low
    values = np.clip(values, hard_low + eps * width, hard_high - eps * width)
    unit_values = (values - hard_low) / width
    return np.log(unit_values / (1.0 - unit_values))


def make_optimizer():
    optimizer_kind = OPTIMIZER_KIND.strip().lower()
    if optimizer_kind in {"adam", "plain_adam"}:
        return numpyro.optim.Adam(LEARNING_RATE)
    if optimizer_kind in {"clipped_adam", "clipped-adam", "clip_adam"}:
        return numpyro.optim.ClippedAdam(LEARNING_RATE, clip_norm=CLIP_NORM)
    raise ValueError(f"Unknown NUMPYRO_SVI_OPTIMIZER={OPTIMIZER_KIND!r}")


def make_jax_data(df):
    return {
        "rt_wrt_stim": jnp.asarray(df["RTwrtStim"].to_numpy(dtype=float)),
        "total_fix": jnp.asarray(df["TotalFixTime"].to_numpy(dtype=float)),
        "t_stim": jnp.asarray(df["intended_fix"].to_numpy(dtype=float)),
        "ABL": jnp.asarray(df["ABL"].to_numpy(dtype=float)),
        "ILD": jnp.asarray(df["ILD"].to_numpy(dtype=float)),
        "condition_id": jnp.asarray(df["condition_id"].to_numpy(dtype=int)),
        "V_A": jnp.asarray(V_A, dtype=jnp.float64),
        "theta_A": jnp.asarray(theta_A, dtype=jnp.float64),
        "t_A_aff": jnp.asarray(t_A_aff, dtype=jnp.float64),
        "T_trunc": jnp.asarray(T_trunc, dtype=jnp.float64),
        "rt_lower": jnp.asarray(RT_LOWER, dtype=jnp.float64),
        "rt_upper": jnp.asarray(RT_UPPER, dtype=jnp.float64),
    }


def finite_corner_input(samples, labels, plot_name):
    samples = np.asarray(samples, dtype=float)
    finite_rows = np.all(np.isfinite(samples), axis=1)
    clean = samples[finite_rows]
    n_dropped = int(np.sum(~finite_rows))
    if n_dropped:
        print(f"{plot_name}: dropped {n_dropped} posterior rows containing NaN/Inf.")
    if len(clean) < 20:
        print(f"{plot_name}: fewer than 20 finite posterior rows; skipping.")
        return None, None

    if CORNER_MAX_SAMPLES > 0 and len(clean) > CORNER_MAX_SAMPLES:
        rng = np.random.default_rng(RNG_SEED)
        clean = clean[rng.choice(len(clean), CORNER_MAX_SAMPLES, replace=False)]

    ranges = []
    for col_idx, label in enumerate(labels):
        low, high = np.quantile(clean[:, col_idx], [0.01, 0.99])
        if not np.isfinite(low) or not np.isfinite(high):
            print(f"{plot_name}: non-finite range for {label}; skipping.")
            return None, None
        if np.isclose(low, high, rtol=0, atol=1e-12):
            pad = max(1e-6, abs(low) * 1e-3)
            low -= pad
            high += pad
        ranges.append((float(low), float(high)))
    return clean, ranges


def run_svi_with_convergence_checks(
    svi,
    rng_key,
    n_steps,
    data,
    n_conditions,
    run_label,
):
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
            window_last = (
                float(window_losses[-1]) if np.isfinite(window_losses[-1]) else np.nan
            )
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
            slope_per_1000 = float(
                np.polyfit(finite_x, finite_losses, 1)[0] * 1000.0
            )
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
                relative_improvement_from_best = improvement_from_best / max(
                    1.0,
                    abs(best_window_mean),
                )
            improved_best = bool(
                not np.isfinite(best_window_mean) or window_mean < best_window_mean
            )
            significant_improvement = bool(
                improved_best
                and (
                    not np.isfinite(relative_improvement_from_best)
                    or relative_improvement_from_best >= SVI_MIN_IMPROVEMENT_REL
                )
            )

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
                "significant_best_improvement": bool(
                    significant_improvement and n_nonfinite == 0
                ),
                "slope_per_1000_steps": slope_per_1000,
                "stable_window_count": stable_window_count,
                "can_stop_for_patience": can_stop_for_patience,
                "n_nonfinite": n_nonfinite,
                "early_stop_candidate": bool(
                    stable_window_count >= SVI_PATIENCE_WINDOWS
                ),
                "no_improve_stop_candidate": bool(
                    no_improve_window_count
                    >= SVI_NO_IMPROVE_PATIENCE_WINDOWS
                ),
            }
        )

        rel_text = (
            "NA" if not np.isfinite(rel_change) else f"{100.0 * rel_change:.3f}%"
        )
        best_rel_text = (
            "NA"
            if not np.isfinite(relative_improvement_from_best)
            else f"{100.0 * relative_improvement_from_best:.3f}%"
        )
        slope_text = (
            "NA" if not np.isfinite(slope_per_1000) else f"{slope_per_1000:.3g}"
        )
        print(
            f"{run_label} chunk {chunk_index:03d} steps {start_step}-{end_step}: "
            f"mean={window_mean:.6g}, last={window_last:.6g}, "
            f"rel_change={rel_text}, best_delta={best_rel_text}, "
            f"slope/1k={slope_text}, stable={stable_window_count}/"
            f"{SVI_PATIENCE_WINDOWS}, no_improve={no_improve_window_count}/"
            f"{SVI_NO_IMPROVE_PATIENCE_WINDOWS}, best_chunk={best_window_chunk}, "
            f"can_stop={can_stop_for_patience}, nonfinite={n_nonfinite}"
        )

        if n_nonfinite:
            print(
                f"WARNING: stopping {run_label} because this window had "
                f"non-finite losses. Returning best state from chunk "
                f"{best_window_chunk} ending at step {best_window_end_step}."
            )
            break
        if not SVI_EARLY_STOP or not can_stop_for_patience:
            prev_window_mean = window_mean
            continue

        if (
            SVI_STOP_MODE in {"legacy", "stable_or_no_improve"}
            and stable_window_count >= SVI_PATIENCE_WINDOWS
        ):
            print(
                f"Stopping {run_label} at step {completed_steps}: "
                f"{SVI_PATIENCE_WINDOWS} consecutive stable windows."
            )
            break
        if (
            SVI_STOP_MODE in {"stable_or_no_improve", "patience_restore_best"}
            and no_improve_window_count >= SVI_NO_IMPROVE_PATIENCE_WINDOWS
        ):
            print(
                f"Stopping {run_label} at step {completed_steps}: no significant "
                f"best-window improvement for "
                f"{SVI_NO_IMPROVE_PATIENCE_WINDOWS} consecutive windows. "
                f"Returning chunk {best_window_chunk}, step "
                f"{best_window_end_step}."
            )
            break
        prev_window_mean = window_mean

    losses = np.concatenate(all_losses) if all_losses else np.array([], dtype=float)
    if best_state is None:
        best_state = state
        best_params = svi.get_params(state)
    print(
        f"{run_label} best returned state: chunk {best_window_chunk}, "
        f"step {best_window_end_step}, mean_loss={best_window_mean:.6g}"
    )
    result = SVIRunResult(best_params, best_state, jnp.asarray(losses))
    return result, pd.DataFrame(convergence_rows)


# %%
# =============================================================================
# Load retained valid trials
# =============================================================================
print(f"Batch/animal: {BATCH_NAME}/{ANIMAL}")
print(f"Process mode: {PROCESS_MODE}")
print(f"RT-only window: [{RT_LOWER:.3f}, {RT_UPPER:.3f}) s wrt stimulus")
print(f"Batch CSV: {BATCH_CSV}")
print(f"Initialization posterior: {REFERENCE_NPZ}")
print(f"Initialization condition table: {REFERENCE_CONDITION_CSV}")
print(f"Output folder: {OUTPUT_DIR}")
print("Likelihood: successful-trial RT density collapsed over choice")
print("Fitted del_go: no (structurally cancels after choice collapse)")

required_paths = [BATCH_CSV, REFERENCE_NPZ, REFERENCE_CONDITION_CSV]
if PROCESS_MODE == "proactive_reactive":
    required_paths.append(ABORT_RESULT_PKL)
for required_path in required_paths:
    if not required_path.exists():
        raise FileNotFoundError(required_path)

raw_df = pd.read_csv(BATCH_CSV)
animal_valid_df = raw_df[
    (raw_df["animal"].astype(int) == ANIMAL)
    & raw_df["success"].isin([1, -1])
    & (raw_df["RTwrtStim"] >= 0)
    & (raw_df["RTwrtStim"] < RT_UPPER)
    & raw_df["ABL"].isin(ABLS)
].dropna(
    subset=[
        "RTwrtStim",
        "TotalFixTime",
        "intended_fix",
        "ABL",
        "ILD",
    ]
).copy()

valid_df = animal_valid_df[
    (animal_valid_df["RTwrtStim"] >= RT_LOWER)
    & (animal_valid_df["RTwrtStim"] < RT_UPPER)
].copy()
if len(valid_df) == 0:
    raise RuntimeError(
        f"No valid trials in [{RT_LOWER:g}, {RT_UPPER:g}) s for "
        f"{BATCH_NAME}/{ANIMAL}."
    )

valid_df["ABL"] = valid_df["ABL"].astype(float)
valid_df["ILD"] = valid_df["ILD"].astype(float)

condition_table = (
    valid_df[["ABL", "ILD"]]
    .drop_duplicates()
    .sort_values(["ABL", "ILD"])
    .reset_index(drop=True)
)
condition_table["condition_id"] = np.arange(len(condition_table), dtype=int)
if len(condition_table) != EXPECTED_N_CONDITIONS:
    raise RuntimeError(
        f"Expected {EXPECTED_N_CONDITIONS} observed conditions, found "
        f"{len(condition_table)}."
    )

valid_df = valid_df.merge(
    condition_table,
    on=["ABL", "ILD"],
    how="left",
    validate="many_to_one",
)
condition_counts = (
    valid_df.groupby("condition_id", as_index=False)
    .size()
    .rename(columns={"size": "n_retained_trials"})
)
condition_table = condition_table.merge(
    condition_counts,
    on="condition_id",
    how="left",
    validate="one_to_one",
)

print(f"Successful trials in 0-1 s candidate pool: {len(animal_valid_df)}")
print(f"Retained fitting trials: {len(valid_df)}")
print(
    f"Removed below lower bound ({1e3 * RT_LOWER:.0f} ms): "
    f"{len(animal_valid_df) - len(valid_df)}"
)
print(f"Observed conditions: {len(condition_table)}")
print(
    condition_table[
        ["condition_id", "ABL", "ILD", "n_retained_trials"]
    ].to_string(index=False)
)

T_trunc = 0.15 if BATCH_NAME == "LED34_even" else 0.30
V_A = 1.0
theta_A = 1.0
t_A_aff = 0.0
if PROCESS_MODE == "proactive_reactive":
    with ABORT_RESULT_PKL.open("rb") as handle:
        abort_results = pickle.load(handle)["vbmc_aborts_results"]
    V_A = float(np.mean(abort_results["V_A_samples"]))
    theta_A = float(np.mean(abort_results["theta_A_samples"]))
    t_A_aff = float(np.mean(abort_results["t_A_aff_samp"]))
    print("\nFixed proactive posterior means:")
    print(f"  V_A      = {V_A:.6g}")
    print(f"  theta_A  = {theta_A:.6g}")
    print(f"  t_A_aff  = {1e3 * t_A_aff:.3f} ms")
    print(f"  T_trunc  = {T_trunc:.3f} s")
else:
    print("\nNo proactive process in this mode.")


# %%
# =============================================================================
# Initialize all 36 dimensions from the completed 37-parameter SVI posterior
# =============================================================================
reference_conditions = pd.read_csv(REFERENCE_CONDITION_CSV)
required_reference_columns = {"condition_id", "ABL", "ILD"}
if not required_reference_columns.issubset(reference_conditions.columns):
    raise RuntimeError(
        f"Reference condition table lacks columns "
        f"{sorted(required_reference_columns - set(reference_conditions.columns))}."
    )
reference_conditions = reference_conditions[
    ["condition_id", "ABL", "ILD"]
].rename(columns={"condition_id": "reference_condition_id"})
reference_conditions["ABL"] = reference_conditions["ABL"].astype(float)
reference_conditions["ILD"] = reference_conditions["ILD"].astype(float)

condition_table = condition_table.merge(
    reference_conditions,
    on=["ABL", "ILD"],
    how="left",
    validate="one_to_one",
)
if condition_table["reference_condition_id"].isna().any():
    missing = condition_table[condition_table["reference_condition_id"].isna()]
    raise RuntimeError(f"Conditions missing from initialization posterior:\n{missing}")
reference_ids = condition_table["reference_condition_id"].to_numpy(dtype=int)

with np.load(REFERENCE_NPZ) as saved:
    missing_keys = [
        name
        for name in [*rt_utils.GLOBAL_PARAM_NAMES, "t_E_aff"]
        if name not in saved.files
    ]
    if missing_keys:
        raise RuntimeError(f"Initialization posterior lacks keys: {missing_keys}")
    reference_samples = {
        name: np.asarray(saved[name], dtype=float)
        for name in rt_utils.GLOBAL_PARAM_NAMES
    }
    reference_samples["t_E_aff"] = np.asarray(
        saved["t_E_aff"][:, reference_ids],
        dtype=float,
    )

n_reference_samples = len(reference_samples[rt_utils.GLOBAL_PARAM_NAMES[0]])
for name, values in reference_samples.items():
    if len(values) != n_reference_samples:
        raise RuntimeError(
            f"Reference sample count mismatch for {name}: "
            f"{len(values)} vs {n_reference_samples}."
        )

init_values = {
    name: float(np.mean(reference_samples[name]))
    for name in rt_utils.GLOBAL_PARAM_NAMES
}
init_values["t_E_aff"] = np.mean(reference_samples["t_E_aff"], axis=0)
init_values = rt_utils.clip_init_to_hard_bounds(init_values)
rt_utils.assert_no_del_go(init_values)

condition_table["initial_t_E_aff_s"] = init_values["t_E_aff"]
condition_table["initial_t_E_aff_ms"] = 1e3 * init_values["t_E_aff"]

latent_columns = []
for name in rt_utils.GLOBAL_PARAM_NAMES:
    hard_low, hard_high = rt_utils.GLOBAL_BOUNDS[name]["hard"]
    latent_columns.append(
        bounded_logit(reference_samples[name], hard_low, hard_high)
    )
hard_low, hard_high = rt_utils.DELAY_BOUNDS["hard"]
for condition_id in range(len(condition_table)):
    latent_columns.append(
        bounded_logit(
            reference_samples["t_E_aff"][:, condition_id],
            hard_low,
            hard_high,
        )
    )
latent_reference_samples = np.column_stack(latent_columns)
latent_dim = len(rt_utils.GLOBAL_PARAM_NAMES) + len(condition_table)
if latent_reference_samples.shape != (n_reference_samples, latent_dim):
    raise RuntimeError(
        f"Unexpected latent sample shape {latent_reference_samples.shape}; "
        f"expected {(n_reference_samples, latent_dim)}."
    )

init_cov = np.cov(latent_reference_samples, rowvar=False)
init_cov += np.eye(latent_dim) * FULLRANK_COV_JITTER
fullrank_init_scale_tril = np.linalg.cholesky(init_cov)
latent_eigenvalues = np.linalg.eigvalsh(init_cov)

print("\nInitial values from the completed 37-parameter SVI posterior:")
for name in rt_utils.GLOBAL_PARAM_NAMES:
    value = init_values[name]
    if name == "T_0":
        print(f"  {name:<12} = {1e3 * value:.5f} ms")
    else:
        print(f"  {name:<12} = {value:.6g}")
print(
    f"  condition t_E_aff mean/range = "
    f"{1e3 * np.mean(init_values['t_E_aff']):.3f} ms / "
    f"[{1e3 * np.min(init_values['t_E_aff']):.3f}, "
    f"{1e3 * np.max(init_values['t_E_aff']):.3f}] ms"
)
print(
    f"Full-rank latent covariance: shape={init_cov.shape}, "
    f"eigenvalue range=[{latent_eigenvalues[0]:.3g}, "
    f"{latent_eigenvalues[-1]:.3g}]"
)
print(
    "Previous fit's del_go posterior was deliberately excluded from "
    "initialization and from this model."
)


# %%
# =============================================================================
# JAX data, initial joint, and gradient checks
# =============================================================================
full_data = make_jax_data(valid_df)
if MAIN_N_TRIALS_OVERRIDE > 0 and len(valid_df) > MAIN_N_TRIALS_OVERRIDE:
    main_df = valid_df.sample(
        MAIN_N_TRIALS_OVERRIDE,
        random_state=RNG_SEED + 1,
    ).sort_index()
else:
    main_df = valid_df.copy()
main_data = make_jax_data(main_df)
n_conditions = len(condition_table)

print(f"\nMain fitting trials: {len(main_df)}")
print(f"Full available retained trials: {len(valid_df)}")
print(
    f"Parameter count: {len(rt_utils.GLOBAL_PARAM_NAMES)} + "
    f"{n_conditions} = {latent_dim}"
)
print(f"JAX devices: {jax.devices()}")
print(f"Guide: {GUIDE_KIND}")
print(
    f"Optimizer: {OPTIMIZER_KIND}, learning_rate={LEARNING_RATE:g}, "
    f"clip_norm={CLIP_NORM:g}"
)
print(
    "SVI convergence checks: "
    f"max_steps={MAIN_STEPS}, every={SVI_CHECK_EVERY}, "
    f"stop_mode={SVI_STOP_MODE}, min_steps={SVI_MIN_STEPS}, "
    f"min_improvement_rel={SVI_MIN_IMPROVEMENT_REL:g}, "
    f"no_improve_patience={SVI_NO_IMPROVE_PATIENCE_WINDOWS}"
)

model = lambda data, n_conditions: (
    rt_utils.npl_alpha_rt_only_condition_delay_model(
        data,
        n_conditions,
        PROCESS_MODE,
        K_max=K_MAX,
    )
)


def log_joint_from_values(values, data):
    log_joint, _ = log_density(model, (data, n_conditions), {}, values)
    return log_joint


initial_main_log_joint = log_joint_from_values(init_values, main_data)
initial_full_log_joint = log_joint_from_values(init_values, full_data)
initial_grad = jax.grad(
    lambda values: log_joint_from_values(values, main_data)
)(init_values)

print("\nInitial log joint:")
print(f"  main = {float(initial_main_log_joint):.6f}")
print(f"  full = {float(initial_full_log_joint):.6f}")
print(
    f"Initial main gradient finite: "
    f"{rt_utils.tree_all_finite(initial_grad)}"
)
if (
    not np.isfinite(float(initial_main_log_joint))
    or not rt_utils.tree_all_finite(initial_grad)
):
    raise RuntimeError("Initial log joint or gradients are non-finite.")


# %%
# =============================================================================
# Main full-rank SVI
# =============================================================================
active_result = None
active_guide = None
active_label = None
active_convergence_df = None

if RUN_MAIN_SVI:
    print(
        f"\nRunning {GUIDE_KIND} main SVI for up to {MAIN_STEPS} steps "
        f"on {len(main_df)} trials..."
    )
    main_guide = rt_utils.make_guide(
        model,
        GUIDE_KIND,
        init_values,
        fullrank_init_scale_tril=fullrank_init_scale_tril,
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
    print(
        f"Main loss: first={float(main_result.losses[0]):.6f}, "
        f"last={float(main_result.losses[-1]):.6f}"
    )
    active_result = main_result
    active_guide = main_guide
    active_label = f"main_{GUIDE_KIND}"
    active_convergence_df = main_convergence_df
else:
    print("\nSkipping main SVI because RUN_MAIN_SVI=False.")


# %%
# =============================================================================
# Posterior samples and saved artifacts
# =============================================================================
if active_result is None:
    print("\nNo SVI result is available; set RUN_MAIN_SVI=1 to save outputs.")
else:
    posterior_samples = active_guide.sample_posterior(
        random.PRNGKey(RNG_SEED + 2),
        active_result.params,
        sample_shape=(POSTERIOR_N_SAMPLES,),
    )
    posterior_np = {
        key: np.asarray(value)
        for key, value in posterior_samples.items()
    }
    guide_params_np = rt_utils.tree_to_numpy(active_result.params)
    finite_report_df, all_posterior_finite = (
        rt_utils.finite_sample_report(posterior_np)
    )
    posterior_summary_df = rt_utils.posterior_samples_to_frame(
        posterior_np,
        condition_table,
    )

    sample_npz = OUTPUT_DIR / f"{active_label}_posterior_samples.npz"
    guide_params_pkl = OUTPUT_DIR / f"{active_label}_guide_params.pkl"
    summary_csv = OUTPUT_DIR / f"{active_label}_posterior_summary.csv"
    finite_report_csv = OUTPUT_DIR / f"{active_label}_posterior_finite_report.csv"
    condition_csv = OUTPUT_DIR / "condition_table.csv"
    loss_csv = OUTPUT_DIR / f"{active_label}_loss.csv"
    convergence_csv = OUTPUT_DIR / f"{active_label}_convergence_checks.csv"
    bundle_pkl = OUTPUT_DIR / f"{active_label}_variational_posterior_bundle.pkl"
    metadata_json = OUTPUT_DIR / f"{active_label}_run_metadata.json"

    np.savez(sample_npz, **posterior_np)
    with guide_params_pkl.open("wb") as handle:
        pickle.dump(guide_params_np, handle)
    posterior_summary_df.to_csv(summary_csv, index=False)
    finite_report_df.to_csv(finite_report_csv, index=False)
    condition_table.to_csv(condition_csv, index=False)

    loss_df = pd.DataFrame(
        {
            "step": np.arange(1, len(active_result.losses) + 1),
            "loss": np.asarray(active_result.losses),
        }
    )
    loss_df.to_csv(loss_csv, index=False)
    active_convergence_df.to_csv(convergence_csv, index=False)

    best_step = int(active_convergence_df.iloc[-1]["best_end_step_so_far"])
    checked_step = int(active_convergence_df.iloc[-1]["end_step"])
    best_window_mean = float(
        active_convergence_df.iloc[-1]["best_mean_loss_so_far"]
    )

    loss_png = OUTPUT_DIR / f"{active_label}_loss.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        loss_df["step"],
        loss_df["loss"],
        lw=0.7,
        color="0.25",
        alpha=0.55,
        label="step loss",
    )
    ax.plot(
        active_convergence_df["end_step"],
        active_convergence_df["mean_loss"],
        marker="o",
        ms=2.5,
        lw=1.1,
        color="tab:blue",
        label="1k-window mean",
    )
    ax.axvline(
        best_step,
        color="tab:green",
        lw=1.2,
        label="restored best",
    )
    ax.axvline(
        checked_step,
        color="tab:red",
        lw=1.2,
        ls="--",
        label="final checked",
    )
    ax.set_xlabel("SVI step")
    ax.set_ylabel("negative ELBO")
    ax.set_title(
        f"{BATCH_NAME}/{ANIMAL} {PROCESS_LABEL} RT-only NPL+alpha"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(loss_png, dpi=200)

    global_corner_samples = np.column_stack(
        [
            posterior_np["rate_lambda"],
            1e3 * posterior_np["T_0"],
            posterior_np["theta_E"],
            posterior_np["w"],
            posterior_np["rate_norm_l"],
            posterior_np["alpha"],
        ]
    )
    global_labels = [
        "lambda",
        "T_0 (ms)",
        "theta_E",
        "w",
        "rate_norm_l",
        "alpha",
    ]
    global_corner_png = OUTPUT_DIR / f"{active_label}_global_corner.png"
    clean_global, global_ranges = finite_corner_input(
        global_corner_samples,
        global_labels,
        "global corner",
    )
    if clean_global is not None:
        fig = corner.corner(
            clean_global,
            labels=global_labels,
            show_titles=True,
            quantiles=[0.025, 0.5, 0.975],
            title_fmt=".3f",
            range=global_ranges,
        )
        fig.suptitle(
            f"{BATCH_NAME}/{ANIMAL} {PROCESS_LABEL} RT-only global posterior",
            y=1.02,
        )
        fig.savefig(global_corner_png, dpi=200, bbox_inches="tight")
    else:
        global_corner_png = None

    selected_conditions = [
        (20.0, -16.0),
        (40.0, 1.0),
        (60.0, 16.0),
    ]
    selected_ids = []
    selected_labels = []
    for abl, ild in selected_conditions:
        match = condition_table[
            (condition_table["ABL"] == abl)
            & (condition_table["ILD"] == ild)
        ]
        if len(match) == 1:
            selected_ids.append(int(match["condition_id"].iloc[0]))
            selected_labels.append(f"tE {int(abl)}/{ild:g} (ms)")

    selected_corner_png = (
        OUTPUT_DIR / f"{active_label}_global_selected_delay_corner.png"
    )
    if selected_ids:
        selected_corner_samples = np.column_stack(
            [
                global_corner_samples,
                1e3 * posterior_np["t_E_aff"][:, selected_ids],
            ]
        )
        selected_corner_labels = global_labels + selected_labels
        clean_selected, selected_ranges = finite_corner_input(
            selected_corner_samples,
            selected_corner_labels,
            "selected-delay corner",
        )
        if clean_selected is not None:
            fig = corner.corner(
                clean_selected,
                labels=selected_corner_labels,
                show_titles=True,
                quantiles=[0.025, 0.5, 0.975],
                title_fmt=".3f",
                range=selected_ranges,
            )
            fig.suptitle(
                f"{BATCH_NAME}/{ANIMAL} {PROCESS_LABEL} RT-only "
                "selected-delay posterior",
                y=1.02,
            )
            fig.savefig(
                selected_corner_png,
                dpi=200,
                bbox_inches="tight",
            )
        else:
            selected_corner_png = None
    else:
        selected_corner_png = None

    delay_values_ms = 1e3 * posterior_np["t_E_aff"]
    delay_q025 = np.nanquantile(delay_values_ms, 0.025, axis=0)
    delay_q500 = np.nanquantile(delay_values_ms, 0.5, axis=0)
    delay_q975 = np.nanquantile(delay_values_ms, 0.975, axis=0)
    delay_init_ms = condition_table["initial_t_E_aff_ms"].to_numpy(dtype=float)

    delay_png = OUTPUT_DIR / f"{active_label}_condition_delay_intervals.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    abl_colors = {
        20.0: "tab:blue",
        40.0: "tab:orange",
        60.0: "tab:green",
    }
    abl_offsets = {20.0: -0.12, 40.0: 0.0, 60.0: 0.12}
    for abl in ABLS:
        mask = condition_table["ABL"].to_numpy(dtype=float) == abl
        ids = condition_table.loc[mask, "condition_id"].to_numpy(dtype=int)
        x = condition_table.loc[mask, "ILD"].to_numpy(dtype=float)
        order = np.argsort(x)
        ids = ids[order]
        x = x[order]
        ax.errorbar(
            x + abl_offsets[abl],
            delay_q500[ids],
            yerr=np.vstack(
                [
                    delay_q500[ids] - delay_q025[ids],
                    delay_q975[ids] - delay_q500[ids],
                ]
            ),
            fmt="o",
            capsize=2,
            color=abl_colors[abl],
            linestyle="none",
            label=f"ABL {int(abl)} reactive SVI 95% CI",
        )
        ax.scatter(
            x + abl_offsets[abl],
            delay_init_ms[ids],
            marker="x",
            color=abl_colors[abl],
            alpha=0.45,
            s=35,
            label=f"ABL {int(abl)} previous SVI mean",
        )
    ax.set_xlabel("ILD")
    ax.set_ylabel("t_E_aff (ms)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle(
        f"{BATCH_NAME}/{ANIMAL} {PROCESS_LABEL} RT-only "
        "condition-delay posterior"
    )
    fig.tight_layout()
    fig.savefig(delay_png, dpi=200)

    posterior_mean_params = {
        name: jnp.asarray(np.mean(posterior_np[name]), dtype=jnp.float64)
        for name in rt_utils.GLOBAL_PARAM_NAMES
    }
    posterior_mean_params["t_E_aff"] = jnp.asarray(
        np.mean(posterior_np["t_E_aff"], axis=0),
        dtype=jnp.float64,
    )
    posterior_mean_loglike = float(
        rt_utils.npl_alpha_rt_only_condition_delay_loglike(
            posterior_mean_params,
            full_data,
            PROCESS_MODE,
            K_max=K_MAX,
        )
    )

    config = {
        "main_steps": int(MAIN_STEPS),
        "check_every": int(SVI_CHECK_EVERY),
        "stop_mode": SVI_STOP_MODE,
        "rel_tol": float(SVI_REL_TOL),
        "patience_windows": int(SVI_PATIENCE_WINDOWS),
        "no_improve_patience_windows": int(
            SVI_NO_IMPROVE_PATIENCE_WINDOWS
        ),
        "min_improvement_rel": float(SVI_MIN_IMPROVEMENT_REL),
        "min_steps": int(SVI_MIN_STEPS),
        "learning_rate": float(LEARNING_RATE),
        "optimizer": OPTIMIZER_KIND,
        "clip_norm": float(CLIP_NORM),
        "rt_lower_s": float(RT_LOWER),
        "rt_upper_s": float(RT_UPPER),
        "K_max": int(K_MAX),
        "posterior_n_samples": int(POSTERIOR_N_SAMPLES),
        "rng_seed": int(RNG_SEED),
        "n_global_parameters": len(rt_utils.GLOBAL_PARAM_NAMES),
        "n_condition_delays": int(n_conditions),
        "n_parameters": int(latent_dim),
        "process_mode": PROCESS_MODE,
        "choice_collapsed": True,
        "includes_proactive_process": bool(
            PROCESS_MODE == "proactive_reactive"
        ),
        "includes_del_go": False,
        "proactive_parameters_fixed": bool(
            PROCESS_MODE == "proactive_reactive"
        ),
    }
    metadata = {
        "schema_version": 1,
        "model_name": (
            f"npl_alpha_rt_only_{PROCESS_MODE}_condition_delay_svi"
        ),
        "batch_name": BATCH_NAME,
        "animal": int(ANIMAL),
        "n_valid_before_rt_lower": int(len(animal_valid_df)),
        "n_retained_trials": int(len(valid_df)),
        "n_removed_below_rt_lower": int(
            len(animal_valid_df) - len(valid_df)
        ),
        "best_step": best_step,
        "final_checked_step": checked_step,
        "best_window_mean_negative_elbo": best_window_mean,
        "posterior_mean_rt_only_loglike": posterior_mean_loglike,
        "all_posterior_samples_finite": bool(all_posterior_finite),
        "fixed_proactive_parameters": (
            {
                "V_A": float(V_A),
                "theta_A": float(theta_A),
                "t_A_aff_s": float(t_A_aff),
                "T_trunc_s": float(T_trunc),
            }
            if PROCESS_MODE == "proactive_reactive"
            else None
        ),
        "config": config,
        "input_paths": {
            "batch_csv": str(BATCH_CSV),
            "initialization_posterior_npz": str(REFERENCE_NPZ),
            "initialization_condition_table": str(
                REFERENCE_CONDITION_CSV
            ),
            "abort_result_pkl": (
                str(ABORT_RESULT_PKL)
                if PROCESS_MODE == "proactive_reactive"
                else None
            ),
        },
    }
    with metadata_json.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    bundle = {
        **metadata,
        "note": (
            f"NPL+alpha {PROCESS_MODE} RT-only SVI fit on successful trials "
            f"with {RT_LOWER:.3f} <= RTwrtStim < {RT_UPPER:.3f} s. "
            "Observed choice is ignored. The model has no del_go because it "
            "cancels after choice collapse. It uses six shared parameters and "
            "one t_E_aff per observed signed ABL/ILD condition."
        ),
        "label": active_label,
        "guide_kind": GUIDE_KIND,
        "output_paths": {
            "posterior_samples_npz": str(sample_npz),
            "guide_params_pkl": str(guide_params_pkl),
            "posterior_summary_csv": str(summary_csv),
            "finite_report_csv": str(finite_report_csv),
            "condition_table_csv": str(condition_csv),
            "loss_csv": str(loss_csv),
            "convergence_csv": str(convergence_csv),
            "metadata_json": str(metadata_json),
            "loss_png": str(loss_png),
            "global_corner_png": (
                str(global_corner_png) if global_corner_png else None
            ),
            "selected_delay_corner_png": (
                str(selected_corner_png) if selected_corner_png else None
            ),
            "delay_intervals_png": str(delay_png),
        },
        "init_values": {
            key: np.asarray(value)
            for key, value in init_values.items()
        },
        "guide_params": guide_params_np,
        "posterior_samples": posterior_np,
        "posterior_summary": posterior_summary_df,
        "finite_report": finite_report_df,
        "condition_table": condition_table,
        "loss_trace": loss_df,
        "convergence_checks": active_convergence_df,
    }
    with bundle_pkl.open("wb") as handle:
        pickle.dump(bundle, handle)

    print("\nPosterior finite-sample report:")
    print(finite_report_df.to_string(index=False))
    if not all_posterior_finite:
        raise RuntimeError("Posterior samples contain NaN or Inf.")

    print("\nSaved outputs:")
    print(f"  samples: {sample_npz}")
    print(f"  guide params: {guide_params_pkl}")
    print(f"  summary: {summary_csv}")
    print(f"  finite report: {finite_report_csv}")
    print(f"  condition table: {condition_csv}")
    print(f"  loss CSV: {loss_csv}")
    print(f"  convergence checks: {convergence_csv}")
    print(f"  metadata: {metadata_json}")
    print(f"  loss plot: {loss_png}")
    print(f"  global corner: {global_corner_png}")
    print(f"  selected-delay corner: {selected_corner_png}")
    print(f"  delay intervals: {delay_png}")
    print(f"  VP bundle: {bundle_pkl}")
    print(f"  posterior-mean RT-only loglike: {posterior_mean_loglike:.6f}")

# %%
