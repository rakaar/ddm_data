# %%
"""Fit LED7/98 valid OFF and bilateral-ON trials with fixed proactive parameters."""

# %%
# =============================================================================
# Editable parameters
# =============================================================================
from pathlib import Path
from time import perf_counter
import json
import os
import pickle
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
ANIMAL_FIT_DIR = REPO_DIR / "fit_animal_by_animal"

ANIMAL = 98
CATEGORIES = [
    item.strip()
    for item in os.environ.get(
        "LED7_VALID_NPL_PILOT_CATEGORIES", "off,on_bilateral"
    ).split(",")
    if item.strip()
]
FORCE_RERUN = os.environ.get("LED7_VALID_NPL_PILOT_FORCE", "0").lower() in {
    "1",
    "true",
    "yes",
}

K_MAX = int(os.environ.get("LED7_VALID_NPL_PILOT_K_MAX", "10"))
QUADRATURE_NODES = int(
    os.environ.get("LED7_VALID_NPL_PILOT_QUADRATURE_NODES", "64")
)
MAIN_STEPS = int(os.environ.get("LED7_VALID_NPL_PILOT_MAIN_STEPS", "150000"))
EXTENDED_MAX_STEPS = int(
    os.environ.get("LED7_VALID_NPL_PILOT_EXTENDED_MAX_STEPS", "250000")
)
SVI_CHECK_EVERY = int(
    os.environ.get("LED7_VALID_NPL_PILOT_CHECK_EVERY", "1000")
)
SVI_MIN_STEPS = int(os.environ.get("LED7_VALID_NPL_PILOT_MIN_STEPS", "50000"))
SVI_MIN_IMPROVEMENT_REL = float(
    os.environ.get("LED7_VALID_NPL_PILOT_MIN_IMPROVEMENT_REL", "0.001")
)
SVI_NO_IMPROVE_PATIENCE_WINDOWS = int(
    os.environ.get("LED7_VALID_NPL_PILOT_PATIENCE_WINDOWS", "12")
)
LEARNING_RATE = float(os.environ.get("LED7_VALID_NPL_PILOT_LR", "0.0002"))
CLIP_NORM = float(os.environ.get("LED7_VALID_NPL_PILOT_CLIP_NORM", "1.0"))
POSTERIOR_N_SAMPLES = int(
    os.environ.get("LED7_VALID_NPL_PILOT_POSTERIOR_SAMPLES", "10000")
)
FULLRANK_COV_JITTER = float(
    os.environ.get("LED7_VALID_NPL_PILOT_COV_JITTER", "1e-6")
)
RNG_SEED = int(os.environ.get("LED7_VALID_NPL_PILOT_SEED", "20260829"))
INIT_MODE = os.environ.get(
    "LED7_VALID_NPL_PILOT_INIT_MODE", "reference_posterior"
).strip().lower()

# Moderate deterministic displacement used to test recovery from a fit that is
# visibly away from the previous posterior while remaining inside sensible bounds.
DISPLACED_INIT_OFFSETS = {
    "rate_lambda": -0.60,
    "T_0": 0.040,
    "theta_E": 0.60,
    "w": 0.040,
    "del_go": -0.040,
    "rate_norm_l": -0.040,
    "alpha": 0.20,
    "t_E_aff": 0.025,
}

DATA_CSV = REPO_DIR / "out_LED.csv"
REFERENCE_ROOT = (
    ANIMAL_FIT_DIR
    / "numpyro_svi_npl_alpha_condition_delay_patience12_restore_best_outputs"
    / f"LED7_{ANIMAL}"
)
REFERENCE_NPZ = REFERENCE_ROOT / "main_fullrank_posterior_samples.npz"
REFERENCE_CONDITION_CSV = REFERENCE_ROOT / "condition_table.csv"

PROACTIVE_ROOT = (
    SCRIPT_DIR
    / "numpyro_svi_proactive_led_step_jump_all_on_no_trunc_exp_lapse_"
    "patience12_min50k_restore_best_outputs"
    / f"LED7_{ANIMAL}"
)
PROACTIVE_NPZ = PROACTIVE_ROOT / "main_fullrank_posterior_samples.npz"
PROACTIVE_RUN_SUMMARY = PROACTIVE_ROOT / "run_summary.json"

OUTPUT_ROOT = Path(
    os.environ.get(
        "LED7_VALID_NPL_PILOT_OUTPUT_ROOT",
        str(
            SCRIPT_DIR
            / "numpyro_svi_proactive_led_step_jump_npl_alpha_valid_"
            "led7_98_pilot_outputs"
        ),
    )
).expanduser()
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

EXPECTED_VALID_COUNTS = {"off": 7791, "on_bilateral": 1170}
EXPECTED_CONDITION_COUNT_RANGE = {
    "off": (226, 292),
    "on_bilateral": (31, 54),
}


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

for import_dir in [SCRIPT_DIR, ANIMAL_FIT_DIR]:
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

import numpyro_proactive_led_step_jump_npl_alpha_valid_svi_utils as svi_utils


# %%
# =============================================================================
# Reusable fit helpers
# =============================================================================
def make_optimizer():
    return numpyro.optim.ClippedAdam(LEARNING_RATE, clip_norm=CLIP_NORM)


def run_svi_with_convergence_checks(svi, rng_key, data):
    all_losses = []
    convergence_rows = []
    state = None
    best_state = None
    best_params = None
    best_window_mean = np.inf
    best_window_chunk = 0
    best_window_end_step = 0
    no_improve_window_count = 0
    completed_steps = 0
    active_max_steps = MAIN_STEPS
    extended_after_150k = False
    stop_reason = "max_steps_restore_best"

    while True:
        chunk_index = len(convergence_rows) + 1
        chunk_steps = min(SVI_CHECK_EVERY, active_max_steps - completed_steps)
        if chunk_steps <= 0:
            break

        start_step = completed_steps + 1
        end_step = completed_steps + chunk_steps
        chunk_start = perf_counter()
        chunk_result = svi.run(
            random.fold_in(rng_key, chunk_index),
            chunk_steps,
            data,
            progress_bar=False,
            init_state=state,
            stable_update=True,
        )
        state = chunk_result.state
        window_losses = np.asarray(jax.device_get(chunk_result.losses), dtype=float)
        all_losses.append(window_losses)
        completed_steps = end_step
        chunk_seconds = perf_counter() - chunk_start

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

        relative_improvement_from_best = np.nan
        improved_best = False
        significant_improvement = False
        if np.isfinite(window_mean):
            if np.isfinite(best_window_mean):
                relative_improvement_from_best = (
                    best_window_mean - window_mean
                ) / max(1.0, abs(best_window_mean))
            improved_best = window_mean < best_window_mean
            significant_improvement = improved_best and (
                not np.isfinite(relative_improvement_from_best)
                or relative_improvement_from_best >= SVI_MIN_IMPROVEMENT_REL
            )

        if improved_best and n_nonfinite == 0:
            best_state = state
            best_params = chunk_result.params
            best_window_mean = window_mean
            best_window_chunk = chunk_index
            best_window_end_step = end_step

        if significant_improvement and n_nonfinite == 0:
            no_improve_window_count = 0
        else:
            no_improve_window_count += 1

        can_stop = completed_steps >= SVI_MIN_STEPS
        convergence_rows.append(
            {
                "chunk": chunk_index,
                "start_step": start_step,
                "end_step": end_step,
                "n_steps": chunk_steps,
                "chunk_seconds": chunk_seconds,
                "steps_per_second": chunk_steps / max(chunk_seconds, 1e-12),
                "mean_loss": window_mean,
                "median_loss": window_median,
                "last_loss": window_last,
                "min_loss": window_min,
                "max_loss": window_max,
                "relative_improvement_from_best": relative_improvement_from_best,
                "best_mean_loss_so_far": best_window_mean,
                "best_chunk_so_far": best_window_chunk,
                "best_end_step_so_far": best_window_end_step,
                "updated_best_state": bool(improved_best and n_nonfinite == 0),
                "significant_best_improvement": bool(
                    significant_improvement and n_nonfinite == 0
                ),
                "no_improve_window_count": no_improve_window_count,
                "slope_per_1000_steps": slope_per_1000,
                "can_stop": can_stop,
                "active_max_steps": active_max_steps,
                "n_nonfinite": n_nonfinite,
            }
        )

        print(
            f"chunk {chunk_index:03d} steps {start_step}-{end_step}: "
            f"mean={window_mean:.6g}, last={window_last:.6g}, "
            f"slope/1k={slope_per_1000:.3g}, "
            f"no_improve={no_improve_window_count}/"
            f"{SVI_NO_IMPROVE_PATIENCE_WINDOWS}, best={best_window_end_step}, "
            f"nonfinite={n_nonfinite}, {chunk_seconds:.1f}s"
        )

        if n_nonfinite:
            stop_reason = "nonfinite_loss"
            break
        if can_stop and no_improve_window_count >= SVI_NO_IMPROVE_PATIENCE_WINDOWS:
            stop_reason = "patience12_restore_best"
            break

        if completed_steps >= active_max_steps:
            if active_max_steps == MAIN_STEPS and EXTENDED_MAX_STEPS > MAIN_STEPS:
                active_max_steps = EXTENDED_MAX_STEPS
                extended_after_150k = True
                print(
                    f"Patience did not trigger by {MAIN_STEPS}; continuing the "
                    f"same optimizer state to at most {EXTENDED_MAX_STEPS} steps."
                )
            else:
                break

    losses = np.concatenate(all_losses) if all_losses else np.array([], dtype=float)
    if best_state is None:
        best_state = state
        best_params = svi.get_params(state)
        best_window_end_step = completed_steps
    return (
        SVIRunResult(best_params, best_state, jnp.asarray(losses)),
        pd.DataFrame(convergence_rows),
        stop_reason,
        extended_after_150k,
    )


def make_jax_data(df, category):
    data = {
        "total_fix": jnp.asarray(df["timed_fix"].to_numpy(dtype=float)),
        "t_stim": jnp.asarray(df["intended_fix"].to_numpy(dtype=float)),
        "ABL": jnp.asarray(df["ABL"].to_numpy(dtype=float)),
        "ILD": jnp.asarray(df["ILD"].to_numpy(dtype=float)),
        "choice": jnp.asarray(df["choice"].to_numpy(dtype=int)),
        "condition_id": jnp.asarray(df["condition_id"].to_numpy(dtype=int)),
        "rt_low": jnp.asarray(0.0, dtype=jnp.float64),
        "rt_high": jnp.asarray(1.0, dtype=jnp.float64),
    }
    if category == "on_bilateral":
        data["t_LED"] = jnp.asarray(df["t_LED"].to_numpy(dtype=float))
    return data


# %%
# =============================================================================
# Load exact upstream posteriors and prepare the shared initialization
# =============================================================================
for required_path in [
    DATA_CSV,
    REFERENCE_NPZ,
    REFERENCE_CONDITION_CSV,
    PROACTIVE_NPZ,
    PROACTIVE_RUN_SUMMARY,
]:
    if not required_path.exists():
        raise FileNotFoundError(required_path)

invalid_categories = sorted(set(CATEGORIES) - svi_utils.VALID_CATEGORIES)
if invalid_categories:
    raise ValueError(f"Unknown categories: {invalid_categories}")

with PROACTIVE_RUN_SUMMARY.open() as handle:
    proactive_run_summary = json.load(handle)
if proactive_run_summary.get("status") != "complete":
    raise RuntimeError(
        f"Accepted proactive source is not complete: {PROACTIVE_RUN_SUMMARY}"
    )

with np.load(PROACTIVE_NPZ) as saved:
    missing = [
        name for name in svi_utils.FIXED_PROACTIVE_PARAM_NAMES if name not in saved.files
    ]
    if missing:
        raise RuntimeError(f"Proactive posterior lacks parameters: {missing}")
    fixed_proactive_params = {
        name: float(np.mean(np.asarray(saved[name], dtype=float)))
        for name in svi_utils.FIXED_PROACTIVE_PARAM_NAMES
    }

reference_condition_df = pd.read_csv(REFERENCE_CONDITION_CSV)
reference_condition_df = reference_condition_df[["condition_id", "ABL", "ILD"]].copy()
reference_condition_df = reference_condition_df.rename(
    columns={"condition_id": "reference_condition_id"}
)
reference_condition_df[["ABL", "ILD"]] = reference_condition_df[
    ["ABL", "ILD"]
].astype(float)

with np.load(REFERENCE_NPZ) as saved:
    required_reference_names = [*svi_utils.GLOBAL_PARAM_NAMES, "t_E_aff"]
    missing = [name for name in required_reference_names if name not in saved.files]
    if missing:
        raise RuntimeError(f"Reference posterior lacks parameters: {missing}")
    raw_reference_samples = {
        name: np.asarray(saved[name], dtype=float)
        for name in required_reference_names
    }

print(f"Data: {DATA_CSV.resolve()}")
print(f"Reference NPL+alpha fit: {REFERENCE_ROOT.resolve()}")
print(f"Fixed proactive+lapse fit: {PROACTIVE_ROOT.resolve()}")
print(f"Output: {OUTPUT_ROOT.resolve()}")
print("\nFixed proactive+lapse posterior means:")
for name, value in fixed_proactive_params.items():
    print(f"  {name:<25} {value:.8g}")


# %%
# =============================================================================
# Rebuild the LED7/98 valid-trial datasets
# =============================================================================
raw_df = pd.read_csv(DATA_CSV)
source_df = raw_df[
    raw_df["repeat_trial"].isin([0, 2]) | raw_df["repeat_trial"].isna()
].copy()
source_df = source_df[
    (source_df["session_type"] == 7)
    & (source_df["training_level"] == 16)
    & (source_df["animal"].astype(int) == ANIMAL)
].copy()
source_df = source_df.dropna(
    subset=[
        "timed_fix",
        "intended_fix",
        "ABL",
        "ILD",
        "response_poke",
        "LED_onset_time",
    ]
)
source_df["RTwrtStim"] = source_df["timed_fix"] - source_df["intended_fix"]
source_df["t_LED"] = source_df["intended_fix"] - source_df["LED_onset_time"]
source_df["choice"] = source_df["response_poke"].map({3: 1, 2: -1})
source_df = source_df[
    source_df["success"].isin([1, -1])
    & source_df["choice"].isin([1, -1])
    & source_df["ABL"].isin([20, 40, 60])
    & source_df["RTwrtStim"].between(0.0, 1.0, inclusive="both")
].copy()

category_frames = {
    "off": source_df[
        (source_df["LED_trial"] == 0) | source_df["LED_trial"].isna()
    ].copy(),
    "on_bilateral": source_df[
        (source_df["LED_trial"] == 1)
        & (source_df["LED_powerL"] > 0)
        & (source_df["LED_powerR"] > 0)
    ].copy(),
}


# %%
# =============================================================================
# Fit each category independently
# =============================================================================
for category_index, category in enumerate(CATEGORIES):
    output_dir = OUTPUT_ROOT / category
    output_dir.mkdir(parents=True, exist_ok=True)
    run_summary_json = output_dir / "run_summary.json"
    if run_summary_json.exists() and not FORCE_RERUN:
        with run_summary_json.open() as handle:
            existing_summary = json.load(handle)
        if existing_summary.get("status") == "complete":
            print(f"\nSkipping completed category {category}: {run_summary_json}")
            continue

    valid_df = category_frames[category].copy()
    if len(valid_df) != EXPECTED_VALID_COUNTS[category]:
        raise RuntimeError(
            f"{category}: expected {EXPECTED_VALID_COUNTS[category]} valid rows, "
            f"found {len(valid_df)}."
        )

    condition_table = (
        valid_df[["ABL", "ILD"]]
        .drop_duplicates()
        .astype(float)
        .sort_values(["ABL", "ILD"])
        .reset_index(drop=True)
    )
    condition_table["condition_id"] = np.arange(len(condition_table), dtype=int)
    if len(condition_table) != 30:
        raise RuntimeError(f"{category}: expected 30 conditions, found {len(condition_table)}.")

    valid_df = valid_df.merge(
        condition_table,
        on=["ABL", "ILD"],
        how="left",
        validate="many_to_one",
    )
    condition_counts = (
        valid_df.groupby("condition_id", observed=True)
        .size()
        .rename("n_valid_trials")
        .reset_index()
    )
    condition_table = condition_table.merge(
        condition_counts,
        on="condition_id",
        validate="one_to_one",
    )
    observed_min = int(condition_table["n_valid_trials"].min())
    observed_max = int(condition_table["n_valid_trials"].max())
    expected_min, expected_max = EXPECTED_CONDITION_COUNT_RANGE[category]
    if (observed_min, observed_max) != (expected_min, expected_max):
        raise RuntimeError(
            f"{category}: expected per-condition count range "
            f"{(expected_min, expected_max)}, found {(observed_min, observed_max)}."
        )

    condition_table = condition_table.merge(
        reference_condition_df,
        on=["ABL", "ILD"],
        how="left",
        validate="one_to_one",
    )
    if condition_table["reference_condition_id"].isna().any():
        raise RuntimeError(f"{category}: reference fit is missing conditions.")
    reference_ids = condition_table["reference_condition_id"].to_numpy(dtype=int)

    reference_samples = {
        name: raw_reference_samples[name]
        for name in svi_utils.GLOBAL_PARAM_NAMES
    }
    reference_samples["t_E_aff"] = raw_reference_samples["t_E_aff"][:, reference_ids]
    n_reference_samples = len(reference_samples[svi_utils.GLOBAL_PARAM_NAMES[0]])
    if any(len(values) != n_reference_samples for values in reference_samples.values()):
        raise RuntimeError(f"{category}: inconsistent reference posterior sample counts.")

    init_values = {
        name: float(np.mean(reference_samples[name]))
        for name in svi_utils.GLOBAL_PARAM_NAMES
    }
    init_values["t_E_aff"] = np.mean(reference_samples["t_E_aff"], axis=0)
    reference_init_values = {
        name: (
            np.asarray(value, dtype=float).copy()
            if name == "t_E_aff"
            else float(value)
        )
        for name, value in init_values.items()
    }
    if INIT_MODE == "displaced":
        for name in svi_utils.GLOBAL_PARAM_NAMES:
            init_values[name] += DISPLACED_INIT_OFFSETS[name]
        init_values["t_E_aff"] = (
            init_values["t_E_aff"] + DISPLACED_INIT_OFFSETS["t_E_aff"]
        )
    elif INIT_MODE != "reference_posterior":
        raise ValueError(
            "LED7_VALID_NPL_PILOT_INIT_MODE must be 'reference_posterior' "
            f"or 'displaced', got {INIT_MODE!r}."
        )
    init_values = svi_utils.clip_init_to_hard_bounds(init_values)

    latent_columns = []
    for name in svi_utils.GLOBAL_PARAM_NAMES:
        hard_low, hard_high = svi_utils.GLOBAL_BOUNDS[name]["hard"]
        latent_columns.append(
            svi_utils.bounded_logit(
                reference_samples[name],
                hard_low,
                hard_high,
            )
        )
    hard_low, hard_high = svi_utils.DELAY_BOUNDS["hard"]
    for condition_id in range(len(condition_table)):
        latent_columns.append(
            svi_utils.bounded_logit(
                reference_samples["t_E_aff"][:, condition_id],
                hard_low,
                hard_high,
            )
        )
    latent_reference_samples = np.column_stack(latent_columns)
    latent_dim = len(svi_utils.GLOBAL_PARAM_NAMES) + len(condition_table)
    if latent_reference_samples.shape != (n_reference_samples, latent_dim):
        raise RuntimeError(
            f"{category}: latent initialization has shape "
            f"{latent_reference_samples.shape}, expected "
            f"{(n_reference_samples, latent_dim)}."
        )
    init_cov = np.cov(latent_reference_samples, rowvar=False)
    init_cov += np.eye(latent_dim) * FULLRANK_COV_JITTER
    fullrank_init_scale_tril = np.linalg.cholesky(init_cov)

    condition_table["initial_t_E_aff_s"] = init_values["t_E_aff"]
    condition_table["initial_t_E_aff_ms"] = 1000.0 * init_values["t_E_aff"]
    condition_table.to_csv(output_dir / "condition_table.csv", index=False)

    data = make_jax_data(valid_df, category)
    model = lambda data: svi_utils.proactive_led_npl_alpha_valid_model(
        data,
        len(condition_table),
        fixed_proactive_params,
        category,
        K_max=K_MAX,
        n_quad=QUADRATURE_NODES,
    )

    def log_joint_from_values(values):
        log_joint, _ = log_density(model, (data,), {}, values)
        return log_joint

    print(f"\n{'=' * 76}")
    print(f"Fitting LED7/{ANIMAL} {category}")
    print(f"Valid trials: {len(valid_df)}")
    print(f"Conditions: {len(condition_table)}")
    print(f"Per-condition count range: {observed_min}-{observed_max}")
    print(f"Parameter count: {latent_dim}")
    print(f"Initialization mode: {INIT_MODE}")
    if INIT_MODE == "displaced":
        print(f"Initialization offsets: {DISPLACED_INIT_OFFSETS}")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Output: {output_dir.resolve()}")

    initial_log_joint = log_joint_from_values(init_values)
    initial_grad = jax.grad(log_joint_from_values)(init_values)
    gradients_finite = svi_utils.tree_all_finite(initial_grad)
    print(f"Initial log joint: {float(initial_log_joint):.8g}")
    print(f"Initial gradients finite: {gradients_finite}")
    if not np.isfinite(float(initial_log_joint)) or not gradients_finite:
        raise RuntimeError(f"{category}: initial log density or gradients are non-finite.")

    started_summary = {
        "status": "running",
        "animal": ANIMAL,
        "category": category,
        "started_at_unix": float(pd.Timestamp.now().timestamp()),
        "valid_trial_count": int(len(valid_df)),
        "condition_count": int(len(condition_table)),
        "output_dir": str(output_dir.resolve()),
    }
    run_summary_json.write_text(json.dumps(started_summary, indent=2) + "\n")

    guide = svi_utils.make_guide(
        model,
        "fullrank",
        init_values,
        fullrank_init_scale_tril=fullrank_init_scale_tril,
    )
    svi = SVI(model, guide, make_optimizer(), Trace_ELBO())
    fit_start = perf_counter()
    fit_result, convergence_df, stop_reason, extended_after_150k = (
        run_svi_with_convergence_checks(
            svi,
            random.PRNGKey(RNG_SEED + 100 * category_index + ANIMAL),
            data,
        )
    )
    fit_elapsed_seconds = perf_counter() - fit_start
    losses = np.asarray(jax.device_get(fit_result.losses), dtype=float)

    posterior_samples = guide.sample_posterior(
        random.PRNGKey(RNG_SEED + 100 * category_index + ANIMAL + 1),
        fit_result.params,
        sample_shape=(POSTERIOR_N_SAMPLES,),
    )
    posterior_np = {
        name: np.asarray(jax.device_get(values))
        for name, values in posterior_samples.items()
    }
    all_posterior_finite = bool(
        all(np.all(np.isfinite(values)) for values in posterior_np.values())
    )
    n_nonfinite_losses = int(np.sum(~np.isfinite(losses)))
    posterior_summary_df = svi_utils.posterior_samples_to_frame(
        posterior_np,
        condition_table,
    )

    best_step = int(convergence_df.iloc[-1]["best_end_step_so_far"])
    checked_step = int(convergence_df.iloc[-1]["end_step"])
    if n_nonfinite_losses or not all_posterior_finite:
        status = "failed_validation"
    elif stop_reason == "patience12_restore_best":
        status = "complete"
    else:
        status = "inconclusive_max_steps"

    label = "main_fullrank"
    sample_npz = output_dir / f"{label}_posterior_samples.npz"
    guide_params_pkl = output_dir / f"{label}_guide_params.pkl"
    posterior_summary_csv = output_dir / f"{label}_posterior_summary.csv"
    loss_csv = output_dir / f"{label}_loss.csv"
    convergence_csv = output_dir / f"{label}_convergence_checks.csv"
    loss_png = output_dir / f"{label}_loss.png"
    bundle_pkl = output_dir / f"{label}_variational_posterior_bundle.pkl"
    fixed_params_json = output_dir / "fixed_proactive_lapse_params.json"

    np.savez_compressed(sample_npz, **posterior_np)
    guide_params_np = svi_utils.tree_to_numpy(fit_result.params)
    with guide_params_pkl.open("wb") as handle:
        pickle.dump(guide_params_np, handle)
    posterior_summary_df.to_csv(posterior_summary_csv, index=False)
    loss_df = pd.DataFrame(
        {"step": np.arange(1, len(losses) + 1), "negative_elbo": losses}
    )
    loss_df.to_csv(loss_csv, index=False)
    convergence_df.to_csv(convergence_csv, index=False)
    fixed_params_json.write_text(
        json.dumps(
            {
                "posterior_means": fixed_proactive_params,
                "source_npz": str(PROACTIVE_NPZ.resolve()),
                "source_run_summary": str(PROACTIVE_RUN_SUMMARY.resolve()),
                "source_led_on_scope": "all LED_trial == 1 rows",
            },
            indent=2,
        )
        + "\n"
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(
        convergence_df["end_step"],
        convergence_df["mean_loss"],
        color="tab:blue",
        linewidth=1.0,
        label="1k-window mean",
    )
    ax.axvline(best_step, color="tab:green", linewidth=1.2, label="restored best")
    ax.axvline(
        checked_step,
        color="tab:red",
        linestyle="--",
        linewidth=1.2,
        label="final checked",
    )
    ax.set_xlabel("SVI step")
    ax.set_ylabel("negative ELBO")
    ax.set_title(f"LED7/{ANIMAL} {category} valid RT+choice fit")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(loss_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    config = {
        "model_name": "fixed_proactive_lapse_plus_npl_alpha_valid_rt_choice",
        "animal": ANIMAL,
        "category": category,
        "jax_backend": jax.default_backend(),
        "valid_rt_window_s": [0.0, 1.0],
        "led_on_scope": "bilateral only" if category == "on_bilateral" else "off",
        "K_max": K_MAX,
        "quadrature_nodes": QUADRATURE_NODES,
        "main_steps_initial_max": MAIN_STEPS,
        "extended_max_steps": EXTENDED_MAX_STEPS,
        "min_steps": SVI_MIN_STEPS,
        "check_every": SVI_CHECK_EVERY,
        "min_improvement_rel": SVI_MIN_IMPROVEMENT_REL,
        "patience_windows": SVI_NO_IMPROVE_PATIENCE_WINDOWS,
        "learning_rate": LEARNING_RATE,
        "clip_norm": CLIP_NORM,
        "guide": "AutoMultivariateNormal",
        "posterior_n_samples": POSTERIOR_N_SAMPLES,
        "rng_seed": RNG_SEED,
        "initialization_mode": INIT_MODE,
        "initialization_offsets": (
            DISPLACED_INIT_OFFSETS if INIT_MODE == "displaced" else {}
        ),
        "fixed_proactive_uncertainty_propagated": False,
        "include_exponential_lapse": True,
    }
    provenance = {
        "input_csv": str(DATA_CSV.resolve()),
        "reference_npl_posterior_npz": str(REFERENCE_NPZ.resolve()),
        "reference_condition_csv": str(REFERENCE_CONDITION_CSV.resolve()),
        "fixed_proactive_posterior_npz": str(PROACTIVE_NPZ.resolve()),
        "fixed_proactive_run_summary": str(PROACTIVE_RUN_SUMMARY.resolve()),
    }
    bundle = {
        "schema_version": 1,
        "config": config,
        "provenance": provenance,
        "fixed_proactive_params": fixed_proactive_params,
        "valid_trial_count": int(len(valid_df)),
        "condition_table": condition_table,
        "reference_init_values": reference_init_values,
        "init_values": init_values,
        "initial_log_joint": float(initial_log_joint),
        "guide_params": guide_params_np,
        "posterior_samples": posterior_np,
        "posterior_summary": posterior_summary_df,
        "loss_trace": loss_df,
        "convergence_checks": convergence_df,
        "stop_reason": stop_reason,
        "fit_elapsed_seconds": fit_elapsed_seconds,
    }
    with bundle_pkl.open("wb") as handle:
        pickle.dump(bundle, handle)

    run_summary = {
        "status": status,
        "stop_reason": stop_reason,
        "animal": ANIMAL,
        "category": category,
        "fit_elapsed_seconds": fit_elapsed_seconds,
        "fit_elapsed_minutes": fit_elapsed_seconds / 60.0,
        "completed_steps": checked_step,
        "restored_best_step": best_step,
        "extended_after_150k": extended_after_150k,
        "n_nonfinite_losses": n_nonfinite_losses,
        "all_posterior_samples_finite": all_posterior_finite,
        "valid_trial_count": int(len(valid_df)),
        "condition_count": int(len(condition_table)),
        "condition_trial_count_min": observed_min,
        "condition_trial_count_max": observed_max,
        "config": config,
        "provenance": provenance,
        "artifacts": {
            "posterior_samples_npz": str(sample_npz.resolve()),
            "guide_params_pkl": str(guide_params_pkl.resolve()),
            "posterior_summary_csv": str(posterior_summary_csv.resolve()),
            "condition_table_csv": str((output_dir / "condition_table.csv").resolve()),
            "loss_csv": str(loss_csv.resolve()),
            "convergence_csv": str(convergence_csv.resolve()),
            "loss_png": str(loss_png.resolve()),
            "variational_posterior_bundle_pkl": str(bundle_pkl.resolve()),
            "fixed_proactive_lapse_params_json": str(fixed_params_json.resolve()),
        },
    }
    run_summary_json.write_text(json.dumps(run_summary, indent=2) + "\n")

    print(f"\n{category} fit elapsed: {fit_elapsed_seconds / 60.0:.2f} minutes")
    print(f"Restored best step: {best_step}")
    print(f"Final checked step: {checked_step}")
    print(f"Stop reason: {stop_reason}")
    print(f"Run status: {status}")
    print(f"Run summary: {run_summary_json.resolve()}")

    if status == "failed_validation":
        raise RuntimeError(f"{category}: SVI fit failed finite-value validation.")
