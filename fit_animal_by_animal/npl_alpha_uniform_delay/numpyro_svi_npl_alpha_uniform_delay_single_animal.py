# %%
"""Fit one animal with NPL+alpha and condition-wise uniform delays.

The fit uses successful RT+choice trials with 0 <= RTwrtStim < 1 s.  Seven
global parameters are shared across stimuli, while every observed ABL/signed-ILD
condition has a fitted delay center and width.
"""

# %%
# =============================================================================
# Editable parameters
# =============================================================================
from pathlib import Path
import json
import os
import pickle
import sys
import time

SCRIPT_DIR = Path(__file__).resolve().parent
ANIMAL_FIT_DIR = SCRIPT_DIR.parent
REPO_DIR = ANIMAL_FIT_DIR.parent

BATCH_NAME = os.environ.get("NUMPYRO_SVI_BATCH", "LED7")
ANIMAL = int(os.environ.get("NUMPYRO_SVI_ANIMAL", "92"))

GUIDE_KIND = os.environ.get("NUMPYRO_SVI_GUIDE", "fullrank")
LEARNING_RATE = float(os.environ.get("NUMPYRO_SVI_LR", "0.0002"))
CLIP_NORM = float(os.environ.get("NUMPYRO_SVI_CLIP_NORM", "1.0"))
RNG_SEED = int(os.environ.get("NUMPYRO_SVI_SEED", "0"))

RUN_SVI = os.environ.get("RUN_MAIN_SVI", "1").strip().lower() in {
    "1",
    "true",
    "yes",
}
MAIN_STEPS = int(os.environ.get("MAIN_STEPS", "150000"))
SVI_CHECK_EVERY = int(os.environ.get("SVI_CHECK_EVERY", "1000"))
SVI_MIN_STEPS = int(os.environ.get("SVI_MIN_STEPS", "50000"))
SVI_NO_IMPROVE_PATIENCE = int(
    os.environ.get("SVI_NO_IMPROVE_PATIENCE_WINDOWS", "12")
)
SVI_MIN_IMPROVEMENT_REL = float(
    os.environ.get("SVI_MIN_IMPROVEMENT_REL", "0.001")
)
SVI_EARLY_STOP = os.environ.get("SVI_EARLY_STOP", "1").strip().lower() in {
    "1",
    "true",
    "yes",
}
POSTERIOR_N_SAMPLES = int(os.environ.get("POSTERIOR_N_SAMPLES", "10000"))
WIDTH_INIT_S = float(os.environ.get("UNIFORM_DELAY_WIDTH_INIT_S", "0.005"))
WIDTH_LATENT_INIT_SCALE = float(
    os.environ.get("UNIFORM_DELAY_WIDTH_LATENT_INIT_SCALE", "0.05")
)

K_MAX = int(os.environ.get("K_MAX", "10"))
INTEGRATED_CDF_TERMS = int(os.environ.get("INTEGRATED_CDF_TERMS", "200"))
ABLS = (20.0, 40.0, 60.0)
DEFAULT_T_TRUNC_S = 0.300
BATCH_T_TRUNC_S = {"LED34_even": 0.150}

REFERENCE_FIT_ROOT = Path(
    os.environ.get(
        "UNIFORM_DELAY_REFERENCE_FIT_ROOT",
        str(
            ANIMAL_FIT_DIR
            / "numpyro_svi_npl_alpha_condition_delay_patience12_restore_best_outputs"
        ),
    )
).expanduser()
OUTPUT_ROOT = Path(
    os.environ.get(
        "NUMPYRO_SVI_OUTPUT_ROOT",
        str(
            SCRIPT_DIR
            / (
                "numpyro_svi_npl_alpha_uniform_delay_rt_choice_"
                "patience12_min50k_restore_best_outputs"
            )
        ),
    )
).expanduser()
OUTPUT_DIR = OUTPUT_ROOT / f"{BATCH_NAME}_{ANIMAL}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_CSV = (
    REPO_DIR
    / "raw_data"
    / "batch_csvs"
    / f"batch_{BATCH_NAME}_valid_and_aborts.csv"
)
ABORT_RESULT_PKL = (
    REPO_DIR
    / "aborts_ipl_npl_time_fit_results"
    / f"results_{BATCH_NAME}_animal_{ANIMAL}.pkl"
)
REFERENCE_DIR = REFERENCE_FIT_ROOT / f"{BATCH_NAME}_{ANIMAL}"
REFERENCE_POSTERIOR_NPZ = REFERENCE_DIR / "main_fullrank_posterior_samples.npz"
REFERENCE_CONDITION_CSV = REFERENCE_DIR / "condition_table.csv"


# %%
# =============================================================================
# Imports
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

for import_path in (ANIMAL_FIT_DIR, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import numpyro_npl_alpha_svi_utils as base_utils
import uniform_delay_svi_utils as svi_utils


# %%
# =============================================================================
# Input preflight and fitting rows
# =============================================================================
print(f"Batch/animal: {BATCH_NAME}/{ANIMAL}")
print(f"Data: {DATA_CSV}")
print(f"Fixed proactive fit: {ABORT_RESULT_PKL}")
print(f"Point-delay initialization: {REFERENCE_DIR}")
print(f"Output: {OUTPUT_DIR}")

for required_path in (
    DATA_CSV,
    ABORT_RESULT_PKL,
    REFERENCE_POSTERIOR_NPZ,
    REFERENCE_CONDITION_CSV,
):
    if not required_path.exists():
        raise FileNotFoundError(required_path)

raw_df = pd.read_csv(DATA_CSV)
if "choice" not in raw_df.columns:
    if "response_poke" not in raw_df.columns:
        raise KeyError("Need either `choice` or `response_poke` in the batch CSV.")
    raw_df["choice"] = raw_df["response_poke"].map({3: 1, 2: -1})

for column in (
    "animal",
    "success",
    "abort_event",
    "TotalFixTime",
    "intended_fix",
    "RTwrtStim",
    "ABL",
    "ILD",
    "choice",
):
    raw_df[column] = pd.to_numeric(raw_df[column], errors="coerce")

animal_df = raw_df[raw_df["animal"].eq(ANIMAL)].copy()
T_trunc_s = BATCH_T_TRUNC_S.get(BATCH_NAME, DEFAULT_T_TRUNC_S)
early_abort_mask = (
    animal_df["abort_event"].eq(3)
    & animal_df["RTwrtStim"].lt(0)
    & animal_df["TotalFixTime"].lt(T_trunc_s)
)
all_pre_stim_abort_mask = (
    animal_df["abort_event"].eq(3) & animal_df["RTwrtStim"].lt(0)
)

valid_df = animal_df[
    animal_df["success"].isin([1, -1])
    & animal_df["RTwrtStim"].ge(0.0)
    & animal_df["RTwrtStim"].lt(1.0)
    & animal_df["ABL"].isin(ABLS)
].copy()
valid_df = valid_df.dropna(
    subset=[
        "TotalFixTime",
        "intended_fix",
        "RTwrtStim",
        "ABL",
        "ILD",
        "choice",
    ]
)
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
valid_df = valid_df.merge(
    condition_table,
    on=["ABL", "ILD"],
    how="left",
    validate="many_to_one",
)
condition_table["n_trials"] = (
    valid_df.groupby("condition_id").size().reindex(condition_table["condition_id"]).to_numpy()
)

if len(condition_table) != 30:
    raise RuntimeError(f"Expected 30 ABL/signed-ILD conditions, found {len(condition_table)}.")
if not valid_df["choice"].isin([-1, 1]).all():
    raise RuntimeError("Fitting rows contain a choice other than -1/+1.")
if BATCH_NAME == "LED7" and ANIMAL == 92 and len(valid_df) != 12137:
    raise RuntimeError(f"Expected 12,137 LED7/92 fitting rows, found {len(valid_df):,}.")

n_early_success = int(
    (
        valid_df["success"].isin([1, -1])
        & valid_df["TotalFixTime"].lt(T_trunc_s)
    ).sum()
)
print("\nData filtering audit:")
print(f"  successful RT+choice rows in [0, 1): {len(valid_df):,}")
print(f"  observed conditions: {len(condition_table)}")
print(f"  pre-stimulus abort_event=3 rows: {int(all_pre_stim_abort_mask.sum()):,}")
print(
    f"  abort_event=3 rows below {1e3 * T_trunc_s:.0f} ms: "
    f"{int(early_abort_mask.sum()):,} (not fitting rows)"
)
print(
    f"  successful fitting rows with TotalFixTime < {1e3 * T_trunc_s:.0f} ms: "
    f"{n_early_success} (retained)"
)
print(condition_table.to_string(index=False))


# %%
# =============================================================================
# Fixed proactive parameters and point-delay initialization
# =============================================================================
with ABORT_RESULT_PKL.open("rb") as file:
    abort_saved = pickle.load(file)
abort_results = abort_saved["vbmc_aborts_results"]
V_A = float(np.mean(abort_results["V_A_samples"]))
theta_A = float(np.mean(abort_results["theta_A_samples"]))
t_A_aff = float(np.mean(abort_results["t_A_aff_samp"]))

reference_samples_file = np.load(REFERENCE_POSTERIOR_NPZ)
reference_samples = {
    key: np.asarray(reference_samples_file[key], dtype=float)
    for key in reference_samples_file.files
}
reference_condition_table = pd.read_csv(REFERENCE_CONDITION_CSV).sort_values(
    "condition_id"
)
reference_condition_table = reference_condition_table.reset_index(drop=True)
condition_table = condition_table.sort_values("condition_id").reset_index(drop=True)

if not np.allclose(
    reference_condition_table[["ABL", "ILD"]].to_numpy(dtype=float),
    condition_table[["ABL", "ILD"]].to_numpy(dtype=float),
):
    raise RuntimeError("Point-delay and new condition tables do not match.")

for parameter in svi_utils.GLOBAL_PARAM_NAMES + ["t_E_aff"]:
    if parameter not in reference_samples:
        raise KeyError(f"Missing `{parameter}` in {REFERENCE_POSTERIOR_NPZ}.")
    if not np.isfinite(reference_samples[parameter]).all():
        raise RuntimeError(f"Reference posterior `{parameter}` contains NaN/Inf.")

center_init = np.mean(reference_samples["t_E_aff"], axis=0)
center_init = np.clip(
    center_init,
    svi_utils.DELAY_CENTER_HARD[0] + 1e-9,
    svi_utils.DELAY_CENTER_HARD[1] - 1e-9,
)
width_cap_init = np.asarray(svi_utils.delay_width_cap_jax(jnp.asarray(center_init)))
if np.any(width_cap_init <= svi_utils.DELAY_WIDTH_HARD[0]):
    raise RuntimeError("A point-delay center cannot support the 1 ms minimum width.")
width_init = np.minimum(WIDTH_INIT_S, width_cap_init - 1e-9)
width_unit_init = (
    width_init - svi_utils.DELAY_WIDTH_HARD[0]
) / (width_cap_init - svi_utils.DELAY_WIDTH_HARD[0])
width_unit_init = np.clip(width_unit_init, 1e-6, 1.0 - 1e-6)

init_values = {
    parameter: float(np.mean(reference_samples[parameter]))
    for parameter in svi_utils.GLOBAL_PARAM_NAMES
}
init_values["t_E_aff_center"] = center_init
init_values["t_E_aff_width_unit"] = width_unit_init

condition_table["center_init_s"] = center_init
condition_table["width_init_s"] = width_init
condition_table["low_init_s"] = center_init - 0.5 * width_init
condition_table["high_init_s"] = center_init + 0.5 * width_init

print("\nFixed proactive posterior means:")
print(f"  V_A = {V_A:.6g}")
print(f"  theta_A = {theta_A:.6g}")
print(f"  t_A_aff = {1e3 * t_A_aff:.3f} ms")
print(f"  abort truncation used upstream = {1e3 * T_trunc_s:.0f} ms")
print("\nInitial global values from the completed point-delay SVI:")
for parameter in svi_utils.GLOBAL_PARAM_NAMES:
    value = init_values[parameter]
    suffix = " ms" if parameter in {"T_0", "del_go"} else ""
    scale = 1e3 if suffix else 1.0
    print(f"  {parameter:<12} = {scale * value:.6g}{suffix}")
print(
    "Initial delay centers: "
    f"{1e3 * center_init.min():.3f}--{1e3 * center_init.max():.3f} ms"
)
print(
    "Initial delay widths: "
    f"{1e3 * width_init.min():.3f}--{1e3 * width_init.max():.3f} ms"
)


# %%
# =============================================================================
# Full-rank covariance initialization
# =============================================================================
def bounded_to_latent(values, hard_bounds):
    hard_low, hard_high = hard_bounds
    epsilon = 1e-6 * (hard_high - hard_low)
    values = np.clip(values, hard_low + epsilon, hard_high - epsilon)
    unit = (values - hard_low) / (hard_high - hard_low)
    return np.log(unit / (1.0 - unit))


reference_latent_columns = []
for parameter in svi_utils.GLOBAL_PARAM_NAMES:
    reference_latent_columns.append(
        bounded_to_latent(
            reference_samples[parameter],
            svi_utils.GLOBAL_BOUNDS[parameter]["hard"],
        )
    )
for condition_id in range(len(condition_table)):
    reference_latent_columns.append(
        bounded_to_latent(
            reference_samples["t_E_aff"][:, condition_id],
            svi_utils.DELAY_CENTER_HARD,
        )
    )

reference_latent_samples = np.column_stack(reference_latent_columns)
reference_covariance = np.cov(reference_latent_samples, rowvar=False)
old_dimension = reference_covariance.shape[0]
latent_dimension = svi_utils.parameter_count(len(condition_table))
if old_dimension != len(svi_utils.GLOBAL_PARAM_NAMES) + len(condition_table):
    raise RuntimeError(f"Unexpected point-delay latent dimension: {old_dimension}.")
if latent_dimension != 67:
    raise RuntimeError(f"Expected 67 fitted parameters, found {latent_dimension}.")

initial_covariance = np.eye(latent_dimension) * WIDTH_LATENT_INIT_SCALE**2
initial_covariance[:old_dimension, :old_dimension] = reference_covariance
initial_covariance += np.eye(latent_dimension) * 1e-8
fullrank_init_scale_tril = np.linalg.cholesky(initial_covariance)

print("\nFull-rank guide initialization:")
print(f"  point-delay covariance block: {old_dimension} x {old_dimension}")
print(f"  new width latent dimensions: {len(condition_table)}")
print(f"  total latent dimension: {latent_dimension}")
print(f"  width latent diagonal scale: {WIDTH_LATENT_INIT_SCALE:g}")


# %%
# =============================================================================
# JAX data and initial likelihood/gradient checks
# =============================================================================
def make_jax_data(frame):
    return {
        "total_fix": jnp.asarray(frame["TotalFixTime"].to_numpy(dtype=float)),
        "t_stim": jnp.asarray(frame["intended_fix"].to_numpy(dtype=float)),
        "ABL": jnp.asarray(frame["ABL"].to_numpy(dtype=float)),
        "ILD": jnp.asarray(frame["ILD"].to_numpy(dtype=float)),
        "choice": jnp.asarray(frame["choice"].to_numpy(dtype=int)),
        "condition_id": jnp.asarray(
            frame["condition_id"].to_numpy(dtype=int)
        ),
        "V_A": jnp.asarray(V_A, dtype=jnp.float64),
        "theta_A": jnp.asarray(theta_A, dtype=jnp.float64),
        "t_A_aff": jnp.asarray(t_A_aff, dtype=jnp.float64),
        "T_trunc": jnp.asarray(T_trunc_s, dtype=jnp.float64),
    }


full_data = make_jax_data(valid_df)
n_conditions = len(condition_table)

model = lambda data, n_conditions: svi_utils.npl_alpha_uniform_delay_model(
    data,
    n_conditions,
    K_max=K_MAX,
    integrated_cdf_terms=INTEGRATED_CDF_TERMS,
)


def log_joint_from_values(values):
    value, _ = log_density(model, (full_data, n_conditions), {}, values)
    return value


initial_check_start = time.perf_counter()
initial_log_joint = log_joint_from_values(init_values)
initial_gradient = jax.grad(log_joint_from_values)(init_values)
initial_check_seconds = time.perf_counter() - initial_check_start
gradient_finite = all(
    np.isfinite(np.asarray(leaf)).all()
    for leaf in jax.tree_util.tree_leaves(initial_gradient)
)
print("\nInitial full-data checks:")
print(f"  log joint = {float(initial_log_joint):.6f}")
print(f"  gradient finite = {gradient_finite}")
print(f"  compile + first gradient = {initial_check_seconds:.3f} s")
if not np.isfinite(float(initial_log_joint)) or not gradient_finite:
    raise RuntimeError("Initial log joint or gradient is non-finite.")


# %%
# =============================================================================
# Patience-12 restore-best SVI
# =============================================================================
def make_optimizer():
    return numpyro.optim.ClippedAdam(LEARNING_RATE, clip_norm=CLIP_NORM)


def run_svi_with_patience(svi, rng_key):
    all_losses = []
    convergence_rows = []
    state = None
    best_state = None
    best_params = None
    best_window_mean = np.inf
    best_window_end_step = 0
    best_window_index = 0
    no_improve_count = 0
    completed_steps = 0
    stop_reason = "max_steps"

    while completed_steps < MAIN_STEPS:
        window_index = len(convergence_rows) + 1
        window_steps = min(SVI_CHECK_EVERY, MAIN_STEPS - completed_steps)
        start_step = completed_steps + 1
        end_step = completed_steps + window_steps
        result = svi.run(
            random.fold_in(rng_key, window_index),
            window_steps,
            full_data,
            n_conditions,
            progress_bar=False,
            init_state=state,
            stable_update=True,
        )
        state = result.state
        window_losses = np.asarray(jax.device_get(result.losses), dtype=float)
        all_losses.append(window_losses)
        completed_steps = end_step

        finite_mask = np.isfinite(window_losses)
        finite_losses = window_losses[finite_mask]
        n_nonfinite = int((~finite_mask).sum())
        window_mean = (
            float(np.mean(finite_losses)) if finite_losses.size else np.nan
        )
        window_last = (
            float(window_losses[-1])
            if window_losses.size and np.isfinite(window_losses[-1])
            else np.nan
        )

        previous_best = best_window_mean
        improved_best = bool(
            np.isfinite(window_mean)
            and (
                not np.isfinite(previous_best)
                or window_mean < previous_best
            )
        )
        if np.isfinite(previous_best) and np.isfinite(window_mean):
            relative_improvement = (
                previous_best - window_mean
            ) / max(1.0, abs(previous_best))
        else:
            relative_improvement = np.nan
        significant_improvement = bool(
            improved_best
            and (
                not np.isfinite(relative_improvement)
                or relative_improvement >= SVI_MIN_IMPROVEMENT_REL
            )
        )

        if improved_best and n_nonfinite == 0:
            best_state = state
            best_params = result.params
            best_window_mean = window_mean
            best_window_end_step = end_step
            best_window_index = window_index

        if significant_improvement:
            no_improve_count = 0
        else:
            no_improve_count += 1

        convergence_rows.append(
            {
                "window": window_index,
                "start_step": start_step,
                "end_step": end_step,
                "mean_loss": window_mean,
                "last_loss": window_last,
                "min_loss": (
                    float(np.min(finite_losses)) if finite_losses.size else np.nan
                ),
                "max_loss": (
                    float(np.max(finite_losses)) if finite_losses.size else np.nan
                ),
                "relative_improvement_from_previous_best": relative_improvement,
                "updated_best_state": improved_best and n_nonfinite == 0,
                "significant_best_improvement": significant_improvement,
                "best_mean_loss_so_far": best_window_mean,
                "best_end_step_so_far": best_window_end_step,
                "no_improve_window_count": no_improve_count,
                "n_nonfinite": n_nonfinite,
            }
        )

        improvement_text = (
            "NA"
            if not np.isfinite(relative_improvement)
            else f"{100.0 * relative_improvement:.4f}%"
        )
        print(
            f"window {window_index:03d} steps {start_step}-{end_step}: "
            f"mean={window_mean:.6g}, last={window_last:.6g}, "
            f"best_delta={improvement_text}, "
            f"no_improve={no_improve_count}/{SVI_NO_IMPROVE_PATIENCE}, "
            f"best_step={best_window_end_step}, nonfinite={n_nonfinite}",
            flush=True,
        )

        if n_nonfinite:
            stop_reason = "nonfinite_loss"
            break
        if (
            SVI_EARLY_STOP
            and completed_steps >= SVI_MIN_STEPS
            and no_improve_count >= SVI_NO_IMPROVE_PATIENCE
        ):
            stop_reason = "patience_no_significant_improvement"
            break

    if best_state is None:
        best_state = state
        best_params = svi.get_params(state)
        best_window_end_step = completed_steps
        best_window_index = len(convergence_rows)

    losses = np.concatenate(all_losses)
    print(
        f"Restoring best window {best_window_index} ending at "
        f"step {best_window_end_step}; checked through step {completed_steps}.",
        flush=True,
    )
    return (
        SVIRunResult(best_params, best_state, jnp.asarray(losses)),
        pd.DataFrame(convergence_rows),
        {
            "stop_reason": stop_reason,
            "best_window": int(best_window_index),
            "best_step": int(best_window_end_step),
            "final_checked_step": int(completed_steps),
            "best_window_mean_loss": float(best_window_mean),
        },
    )


fit_start = time.perf_counter()
fit_result = None
convergence_df = None
convergence_summary = None
guide = None

if RUN_SVI:
    print("\nSVI configuration:")
    print(f"  guide = {GUIDE_KIND}")
    print(f"  learning rate = {LEARNING_RATE:g}")
    print(f"  maximum steps = {MAIN_STEPS:,}")
    print(f"  check window = {SVI_CHECK_EVERY:,}")
    print(f"  minimum steps = {SVI_MIN_STEPS:,}")
    print(f"  patience = {SVI_NO_IMPROVE_PATIENCE} windows")
    print(
        "  significant improvement = "
        f"{100.0 * SVI_MIN_IMPROVEMENT_REL:.3f}%"
    )
    print(f"  reactive CDF/PDF terms = {K_MAX}")
    print(f"  integrated-CDF terms = {INTEGRATED_CDF_TERMS}")

    guide = base_utils.make_guide(
        model,
        GUIDE_KIND,
        init_values,
        fullrank_init_scale_tril=fullrank_init_scale_tril,
    )
    svi = SVI(model, guide, make_optimizer(), Trace_ELBO())
    fit_result, convergence_df, convergence_summary = run_svi_with_patience(
        svi,
        random.PRNGKey(RNG_SEED),
    )
else:
    print("RUN_MAIN_SVI=0: stopping after initial likelihood/gradient checks.")


# %%
# =============================================================================
# Posterior artifacts and convergence plot
# =============================================================================
if fit_result is not None:
    posterior = guide.sample_posterior(
        random.PRNGKey(RNG_SEED + 1),
        fit_result.params,
        sample_shape=(POSTERIOR_N_SAMPLES,),
    )
    posterior_np = {
        key: np.asarray(jax.device_get(value)) for key, value in posterior.items()
    }
    center_samples = posterior_np["t_E_aff_center"]
    width_unit_samples = posterior_np["t_E_aff_width_unit"]
    width_samples, _, _ = svi_utils.delay_width_from_unit_jax(
        jnp.asarray(center_samples),
        jnp.asarray(width_unit_samples),
    )
    width_samples = np.asarray(width_samples)
    low_samples, high_samples = svi_utils.delay_endpoints_jax(
        jnp.asarray(center_samples),
        jnp.asarray(width_samples),
    )
    posterior_np["t_E_aff_width"] = width_samples
    posterior_np["t_E_aff_low"] = np.asarray(low_samples)
    posterior_np["t_E_aff_high"] = np.asarray(high_samples)
    posterior_np["t_E_aff"] = center_samples

    for name, values in posterior_np.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f"Posterior `{name}` contains NaN/Inf.")

    posterior_npz = OUTPUT_DIR / "main_fullrank_posterior_samples.npz"
    np.savez(posterior_npz, **posterior_np)

    global_rows = []
    for parameter in svi_utils.GLOBAL_PARAM_NAMES:
        values = posterior_np[parameter]
        q025, q500, q975 = np.quantile(values, [0.025, 0.5, 0.975])
        global_rows.append(
            {
                "parameter": parameter,
                "mean": float(np.mean(values)),
                "q025": float(q025),
                "median": float(q500),
                "q975": float(q975),
            }
        )
    global_summary_df = pd.DataFrame(global_rows)
    global_summary_csv = OUTPUT_DIR / "main_fullrank_global_summary.csv"
    global_summary_df.to_csv(global_summary_csv, index=False)

    condition_rows = []
    for condition in condition_table.itertuples(index=False):
        condition_id = int(condition.condition_id)
        row = {
            "batch_name": BATCH_NAME,
            "animal": ANIMAL,
            "condition_id": condition_id,
            "ABL": float(condition.ABL),
            "ILD": float(condition.ILD),
            "n_trials": int(condition.n_trials),
        }
        for label, values in (
            ("center", center_samples[:, condition_id]),
            ("width", width_samples[:, condition_id]),
            ("low", posterior_np["t_E_aff_low"][:, condition_id]),
            ("high", posterior_np["t_E_aff_high"][:, condition_id]),
        ):
            q025, q500, q975 = np.quantile(values, [0.025, 0.5, 0.975])
            row[f"{label}_mean_s"] = float(np.mean(values))
            row[f"{label}_median_s"] = float(q500)
            row[f"{label}_q025_s"] = float(q025)
            row[f"{label}_q975_s"] = float(q975)
            row[f"{label}_mean_ms"] = 1e3 * row[f"{label}_mean_s"]
        condition_rows.append(row)
    condition_summary_df = pd.DataFrame(condition_rows)
    condition_summary_csv = OUTPUT_DIR / "condition_delay_summary.csv"
    condition_summary_df.to_csv(condition_summary_csv, index=False)
    condition_table_csv = OUTPUT_DIR / "condition_table.csv"
    condition_table.to_csv(condition_table_csv, index=False)

    loss_values = np.asarray(jax.device_get(fit_result.losses), dtype=float)
    loss_csv = OUTPUT_DIR / "main_fullrank_loss.csv"
    pd.DataFrame(
        {"step": np.arange(1, len(loss_values) + 1), "negative_elbo": loss_values}
    ).to_csv(loss_csv, index=False)
    convergence_csv = OUTPUT_DIR / "main_fullrank_convergence_checks.csv"
    convergence_df.to_csv(convergence_csv, index=False)

    loss_png = OUTPUT_DIR / "main_fullrank_loss.png"
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(
        np.arange(1, len(loss_values) + 1),
        loss_values,
        color="0.65",
        linewidth=0.35,
        alpha=0.30,
        label="step negative ELBO",
    )
    ax.plot(
        convergence_df["end_step"],
        convergence_df["mean_loss"],
        color="#0072B2",
        linewidth=1.4,
        label=f"{SVI_CHECK_EVERY}-step mean",
    )
    ax.axvline(
        convergence_summary["best_step"],
        color="#009E73",
        linewidth=1.4,
        label="restored best",
    )
    ax.axvline(
        convergence_summary["final_checked_step"],
        color="#D55E00",
        linestyle="--",
        linewidth=1.2,
        label="final checked",
    )
    ax.set_xlabel("SVI step")
    ax.set_ylabel("Negative ELBO")
    ax.set_title(f"{BATCH_NAME}/{ANIMAL} NPL+alpha uniform-delay SVI")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(loss_png, dpi=250)

    elapsed_seconds = time.perf_counter() - fit_start
    metadata = {
        "batch_name": BATCH_NAME,
        "animal": ANIMAL,
        "model": "NPL+alpha RT+choice with condition-wise uniform t_E_aff",
        "n_parameters": latent_dimension,
        "n_global_parameters": len(svi_utils.GLOBAL_PARAM_NAMES),
        "n_conditions": n_conditions,
        "n_delay_parameters": 2 * n_conditions,
        "n_fitting_trials": len(valid_df),
        "rt_window_s": [0.0, 1.0],
        "fixed_proactive_parameters": {
            "V_A": V_A,
            "theta_A": theta_A,
            "t_A_aff_s": t_A_aff,
            "upstream_abort_truncation_s": T_trunc_s,
        },
        "abort_filter_audit": {
            "n_pre_stim_abort_event_3": int(all_pre_stim_abort_mask.sum()),
            "n_early_abort_event_3": int(early_abort_mask.sum()),
            "n_successful_total_fix_below_truncation": n_early_success,
            "note": (
                "Abort truncation applies to abort_event=3 rows only; successful "
                "rows are retained. The valid-trial likelihood matches the legacy "
                "point-delay retained-window convention."
            ),
        },
        "delay_center_bounds_s": {
            "hard": list(svi_utils.DELAY_CENTER_HARD),
            "plausible": list(svi_utils.DELAY_CENTER_PLAUSIBLE),
        },
        "delay_width_bounds_s": {
            "hard": list(svi_utils.DELAY_WIDTH_HARD),
            "plausible": list(svi_utils.DELAY_WIDTH_PLAUSIBLE),
            "initial": WIDTH_INIT_S,
        },
        "K_max": K_MAX,
        "integrated_cdf_terms": INTEGRATED_CDF_TERMS,
        "guide": GUIDE_KIND,
        "optimizer": "ClippedAdam",
        "learning_rate": LEARNING_RATE,
        "clip_norm": CLIP_NORM,
        "stopping": {
            "maximum_steps": MAIN_STEPS,
            "window_steps": SVI_CHECK_EVERY,
            "minimum_steps": SVI_MIN_STEPS,
            "patience_windows": SVI_NO_IMPROVE_PATIENCE,
            "minimum_relative_improvement": SVI_MIN_IMPROVEMENT_REL,
            **convergence_summary,
        },
        "posterior_samples": POSTERIOR_N_SAMPLES,
        "rng_seed": RNG_SEED,
        "elapsed_seconds": elapsed_seconds,
        "input_paths": {
            "data_csv": str(DATA_CSV),
            "abort_result_pkl": str(ABORT_RESULT_PKL),
            "reference_posterior_npz": str(REFERENCE_POSTERIOR_NPZ),
        },
    }
    metadata_json = OUTPUT_DIR / "main_fullrank_run_metadata.json"
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    guide_params_np = jax.tree_util.tree_map(
        lambda value: np.asarray(jax.device_get(value)),
        fit_result.params,
    )
    bundle = {
        "schema_version": 1,
        "metadata": metadata,
        "guide_kind": GUIDE_KIND,
        "guide_params": guide_params_np,
        "posterior_samples": posterior_np,
        "condition_table": condition_table,
        "condition_summary": condition_summary_df,
        "global_summary": global_summary_df,
        "convergence_checks": convergence_df,
        "loss_trace": loss_values,
    }
    bundle_pkl = OUTPUT_DIR / "main_fullrank_variational_posterior_bundle.pkl"
    with bundle_pkl.open("wb") as file:
        pickle.dump(bundle, file)

    print("\nSaved fit artifacts:")
    for artifact in (
        posterior_npz,
        bundle_pkl,
        global_summary_csv,
        condition_summary_csv,
        condition_table_csv,
        loss_csv,
        convergence_csv,
        metadata_json,
        loss_png,
    ):
        print(f"  {artifact}")
    print(f"Elapsed fit + artifact time: {elapsed_seconds / 60.0:.2f} min")
