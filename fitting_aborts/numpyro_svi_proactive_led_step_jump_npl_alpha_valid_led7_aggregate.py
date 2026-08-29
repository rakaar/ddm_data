# %%
"""Fit pooled LED7 valid OFF and bilateral-ON trials with fixed proactive parameters."""

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

ANIMALS = (92, 93, 98, 99, 100, 103)
CATEGORIES = [
    value.strip()
    for value in os.environ.get(
        "LED7_VALID_NPL_AGG_CATEGORIES", "off,on_bilateral"
    ).split(",")
    if value.strip()
]
INIT_MODES = [
    value.strip()
    for value in os.environ.get(
        "LED7_VALID_NPL_AGG_INIT_MODES", "warm,displaced"
    ).split(",")
    if value.strip()
]
FORCE_RERUN = os.environ.get("LED7_VALID_NPL_AGG_FORCE", "0").lower() in {
    "1",
    "true",
    "yes",
}

K_MAX = int(os.environ.get("LED7_VALID_NPL_AGG_K_MAX", "10"))
QUADRATURE_NODES = int(
    os.environ.get("LED7_VALID_NPL_AGG_QUADRATURE_NODES", "64")
)
MAIN_STEPS = int(os.environ.get("LED7_VALID_NPL_AGG_MAIN_STEPS", "150000"))
EXTENDED_MAX_STEPS = int(
    os.environ.get("LED7_VALID_NPL_AGG_EXTENDED_MAX_STEPS", "250000")
)
SVI_CHECK_EVERY = int(
    os.environ.get("LED7_VALID_NPL_AGG_CHECK_EVERY", "1000")
)
SVI_MIN_STEPS = int(os.environ.get("LED7_VALID_NPL_AGG_MIN_STEPS", "50000"))
SVI_MIN_IMPROVEMENT_REL = float(
    os.environ.get("LED7_VALID_NPL_AGG_MIN_IMPROVEMENT_REL", "0.001")
)
SVI_NO_IMPROVE_PATIENCE_WINDOWS = int(
    os.environ.get("LED7_VALID_NPL_AGG_PATIENCE_WINDOWS", "12")
)
LEARNING_RATE = float(os.environ.get("LED7_VALID_NPL_AGG_LR", "0.0002"))
CLIP_NORM = float(os.environ.get("LED7_VALID_NPL_AGG_CLIP_NORM", "1.0"))
POSTERIOR_N_SAMPLES = int(
    os.environ.get("LED7_VALID_NPL_AGG_POSTERIOR_SAMPLES", "10000")
)
FULLRANK_COV_JITTER = float(
    os.environ.get("LED7_VALID_NPL_AGG_COV_JITTER", "1e-6")
)
RNG_SEED = int(os.environ.get("LED7_VALID_NPL_AGG_SEED", "20260830"))

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

EXPECTED_VALID_COUNTS = {"off": 52799, "on_bilateral": 8138}
EXPECTED_CONDITION_COUNT_RANGE = {
    "off": (1665, 1844),
    "on_bilateral": (224, 294),
}

DATA_CSV = REPO_DIR / "out_LED.csv"
REFERENCE_ROOT = (
    ANIMAL_FIT_DIR
    / "numpyro_svi_npl_alpha_condition_delay_patience12_restore_best_outputs"
)
PROACTIVE_ROOT = (
    SCRIPT_DIR
    / "numpyro_svi_proactive_led_step_jump_all_on_no_trunc_exp_lapse_"
    "aggregate_patience12_min150k_restore_best_outputs"
    / "LED7_all_animals"
)
PROACTIVE_NPZ = PROACTIVE_ROOT / "main_fullrank_posterior_samples.npz"
PROACTIVE_RUN_SUMMARY = PROACTIVE_ROOT / "run_summary.json"

OUTPUT_ROOT = Path(
    os.environ.get(
        "LED7_VALID_NPL_AGG_OUTPUT_ROOT",
        str(
            SCRIPT_DIR
            / "numpyro_svi_proactive_led_step_jump_npl_alpha_valid_led7_"
            "aggregate_patience12_min50k_restore_best_outputs"
        ),
    )
).expanduser()
OUTPUT_DIR = OUTPUT_ROOT / "LED7_all_animals"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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

import numpyro_proactive_led_step_jump_npl_alpha_utils as combined_likelihood
import numpyro_proactive_led_step_jump_npl_alpha_valid_svi_utils as svi_utils


# %%
# =============================================================================
# SVI helpers
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
        window_mean = float(np.mean(finite_losses)) if finite_losses.size else np.nan
        window_median = (
            float(np.median(finite_losses)) if finite_losses.size else np.nan
        )
        window_last = (
            float(window_losses[-1])
            if window_losses.size and np.isfinite(window_losses[-1])
            else np.nan
        )
        window_min = float(np.min(finite_losses)) if finite_losses.size else np.nan
        window_max = float(np.max(finite_losses)) if finite_losses.size else np.nan
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

        windows_since_best = int(
            round((completed_steps - best_window_end_step) / SVI_CHECK_EVERY)
        )
        can_stop = (
            completed_steps >= SVI_MIN_STEPS
            and no_improve_window_count >= SVI_NO_IMPROVE_PATIENCE_WINDOWS
            and windows_since_best >= SVI_NO_IMPROVE_PATIENCE_WINDOWS
        )
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
                "windows_since_best": windows_since_best,
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
            f"post_best={windows_since_best}, nonfinite={n_nonfinite}, "
            f"{chunk_seconds:.1f}s"
        )

        if n_nonfinite:
            stop_reason = "nonfinite_loss"
            break
        if can_stop:
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
    print(
        f"Restored best window: chunk {best_window_chunk}, "
        f"step {best_window_end_step}, mean negative ELBO={best_window_mean:.6g}"
    )
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


def posterior_mean_loglike(posterior, data, category, fixed_proactive_params):
    params = {
        name: jnp.asarray(np.mean(np.asarray(posterior[name], dtype=float)))
        for name in svi_utils.GLOBAL_PARAM_NAMES
    }
    params["t_E_aff"] = jnp.asarray(
        np.mean(np.asarray(posterior["t_E_aff"], dtype=float), axis=0)
    )
    for name, value in fixed_proactive_params.items():
        params[name] = jnp.asarray(value, dtype=jnp.float64)

    likelihood_data = dict(data)
    likelihood_data["t_E_aff"] = params["t_E_aff"][data["condition_id"]]
    if category == "off":
        value = combined_likelihood.led_off_npl_alpha_valid_loglike_jax(
            params,
            likelihood_data,
            K_max=K_MAX,
            include_lapse=True,
        )
    else:
        value = combined_likelihood.led_on_npl_alpha_valid_loglike_jax(
            params,
            likelihood_data,
            K_max=K_MAX,
            n_quad=QUADRATURE_NODES,
            include_lapse=True,
        )
    return float(jax.device_get(value))


# %%
# =============================================================================
# Load aggregate proactive posterior and pooled valid-trial data
# =============================================================================
for path in [DATA_CSV, PROACTIVE_NPZ, PROACTIVE_RUN_SUMMARY]:
    if not path.exists():
        raise FileNotFoundError(path)

invalid_categories = sorted(set(CATEGORIES) - svi_utils.VALID_CATEGORIES)
if invalid_categories:
    raise ValueError(f"Unknown categories: {invalid_categories}")
invalid_init_modes = sorted(set(INIT_MODES) - {"warm", "displaced"})
if invalid_init_modes:
    raise ValueError(f"Unknown initialization modes: {invalid_init_modes}")

with PROACTIVE_RUN_SUMMARY.open() as handle:
    proactive_run_summary = json.load(handle)
if proactive_run_summary.get("status") != "complete":
    raise RuntimeError(f"Aggregate proactive source is not complete: {PROACTIVE_ROOT}")

with np.load(PROACTIVE_NPZ) as saved:
    missing = [
        name for name in svi_utils.FIXED_PROACTIVE_PARAM_NAMES if name not in saved.files
    ]
    if missing:
        raise RuntimeError(f"Aggregate proactive posterior lacks parameters: {missing}")
    fixed_proactive_params = {
        name: float(np.mean(np.asarray(saved[name], dtype=float)))
        for name in svi_utils.FIXED_PROACTIVE_PARAM_NAMES
    }

raw_df = pd.read_csv(DATA_CSV)
source_df = raw_df[
    raw_df["repeat_trial"].isin([0, 2]) | raw_df["repeat_trial"].isna()
].copy()
source_df = source_df[
    (source_df["session_type"] == 7)
    & (source_df["training_level"] == 16)
    & source_df["animal"].astype(int).isin(ANIMALS)
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
source_df["animal"] = source_df["animal"].astype(int)
source_df["RTwrtStim"] = source_df["timed_fix"] - source_df["intended_fix"]
source_df["t_LED"] = source_df["intended_fix"] - source_df["LED_onset_time"]
source_df["choice"] = source_df["response_poke"].map({3: 1, 2: -1})
valid_source_df = source_df[
    source_df["success"].isin([1, -1])
    & source_df["choice"].isin([1, -1])
    & source_df["ABL"].isin([20, 40, 60])
    & source_df["RTwrtStim"].between(0.0, 1.0, inclusive="both")
].copy()

category_frames = {
    "off": valid_source_df[
        (valid_source_df["LED_trial"] == 0) | valid_source_df["LED_trial"].isna()
    ].copy(),
    "on_bilateral": valid_source_df[
        (valid_source_df["LED_trial"] == 1)
        & (valid_source_df["LED_powerL"] > 0)
        & (valid_source_df["LED_powerR"] > 0)
    ].copy(),
}

master_condition_table = (
    valid_source_df[["ABL", "ILD"]]
    .drop_duplicates()
    .astype(float)
    .sort_values(["ABL", "ILD"])
    .reset_index(drop=True)
)
master_condition_table["condition_id"] = np.arange(
    len(master_condition_table), dtype=int
)
if len(master_condition_table) != 30:
    raise RuntimeError(f"Expected 30 conditions, found {len(master_condition_table)}.")

for category, frame in category_frames.items():
    if len(frame) != EXPECTED_VALID_COUNTS[category]:
        raise RuntimeError(
            f"{category}: expected {EXPECTED_VALID_COUNTS[category]} valid rows, "
            f"found {len(frame)}."
        )
    counts = frame.groupby(["ABL", "ILD"], observed=True).size()
    observed_range = (int(counts.min()), int(counts.max()))
    if observed_range != EXPECTED_CONDITION_COUNT_RANGE[category]:
        raise RuntimeError(
            f"{category}: expected per-condition range "
            f"{EXPECTED_CONDITION_COUNT_RANGE[category]}, found {observed_range}."
        )
    category_frames[category] = frame.merge(
        master_condition_table,
        on=["ABL", "ILD"],
        how="left",
        validate="many_to_one",
    )

print(f"Data: {DATA_CSV.resolve()}")
print(f"Aggregate proactive+lapse source: {PROACTIVE_ROOT.resolve()}")
print(f"Animal-wise NPL+alpha initialization root: {REFERENCE_ROOT.resolve()}")
print(f"Output: {OUTPUT_DIR.resolve()}")
for category, frame in category_frames.items():
    print(f"{category}: {len(frame)} valid RT+choice trials")


# %%
# =============================================================================
# Load and align the six animal-wise NPL+alpha reference posteriors
# =============================================================================
reference_by_animal = {}
latent_by_animal = {}
required_names = [*svi_utils.GLOBAL_PARAM_NAMES, "t_E_aff"]

for animal in ANIMALS:
    animal_root = REFERENCE_ROOT / f"LED7_{animal}"
    posterior_path = animal_root / "main_fullrank_posterior_samples.npz"
    condition_path = animal_root / "condition_table.csv"
    for path in [posterior_path, condition_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    condition_table = pd.read_csv(condition_path)[["condition_id", "ABL", "ILD"]]
    condition_table[["ABL", "ILD"]] = condition_table[["ABL", "ILD"]].astype(float)
    aligned = master_condition_table.merge(
        condition_table.rename(columns={"condition_id": "reference_condition_id"}),
        on=["ABL", "ILD"],
        how="left",
        validate="one_to_one",
    )
    if aligned["reference_condition_id"].isna().any():
        raise RuntimeError(f"LED7/{animal} reference is missing conditions.")
    reference_ids = aligned["reference_condition_id"].to_numpy(dtype=int)

    with np.load(posterior_path) as saved:
        missing = [name for name in required_names if name not in saved.files]
        if missing:
            raise RuntimeError(f"LED7/{animal} posterior lacks parameters: {missing}")
        samples = {
            name: np.asarray(saved[name], dtype=float)
            for name in svi_utils.GLOBAL_PARAM_NAMES
        }
        samples["t_E_aff"] = np.asarray(saved["t_E_aff"], dtype=float)[:, reference_ids]
    if not all(np.all(np.isfinite(values)) for values in samples.values()):
        raise RuntimeError(f"LED7/{animal} reference contains non-finite samples.")
    reference_by_animal[animal] = samples

    latent_columns = []
    for name in svi_utils.GLOBAL_PARAM_NAMES:
        hard_low, hard_high = svi_utils.GLOBAL_BOUNDS[name]["hard"]
        latent_columns.append(
            svi_utils.bounded_logit(samples[name], hard_low, hard_high)
        )
    delay_low, delay_high = svi_utils.DELAY_BOUNDS["hard"]
    for condition_id in range(len(master_condition_table)):
        latent_columns.append(
            svi_utils.bounded_logit(
                samples["t_E_aff"][:, condition_id], delay_low, delay_high
            )
        )
    latent_by_animal[animal] = np.column_stack(latent_columns)


# %%
# =============================================================================
# Build category-specific pooled initializations
# =============================================================================
category_initialization = {}
latent_dim = len(svi_utils.GLOBAL_PARAM_NAMES) + len(master_condition_table)

for category, frame in category_frames.items():
    animal_trial_counts = (
        frame.groupby("animal", observed=True).size().reindex(ANIMALS).astype(int)
    )
    animal_weights = animal_trial_counts.to_numpy(dtype=float)
    animal_weights /= animal_weights.sum()
    effective_animal_count = float(1.0 / np.sum(animal_weights**2))

    warm_values = {}
    for name in svi_utils.GLOBAL_PARAM_NAMES:
        animal_means = np.array(
            [np.mean(reference_by_animal[a][name]) for a in ANIMALS], dtype=float
        )
        warm_values[name] = float(np.sum(animal_weights * animal_means))

    delay_means = np.zeros(len(master_condition_table), dtype=float)
    condition_animal_counts = np.zeros((len(ANIMALS), len(master_condition_table)))
    for animal_index, animal in enumerate(ANIMALS):
        animal_counts = (
            frame[frame["animal"] == animal]
            .groupby("condition_id", observed=True)
            .size()
            .reindex(range(len(master_condition_table)), fill_value=0)
            .to_numpy(dtype=float)
        )
        condition_animal_counts[animal_index] = animal_counts
    for condition_id in range(len(master_condition_table)):
        weights = condition_animal_counts[:, condition_id]
        if weights.sum() <= 0:
            raise RuntimeError(f"{category}: empty condition {condition_id}.")
        weights /= weights.sum()
        animal_delay_means = np.array(
            [
                np.mean(reference_by_animal[a]["t_E_aff"][:, condition_id])
                for a in ANIMALS
            ]
        )
        delay_means[condition_id] = np.sum(weights * animal_delay_means)
    warm_values["t_E_aff"] = delay_means
    warm_values = svi_utils.clip_init_to_hard_bounds(warm_values)

    pooled_within_cov = np.zeros((latent_dim, latent_dim), dtype=float)
    for weight, animal in zip(animal_weights, ANIMALS):
        latent_samples = latent_by_animal[animal]
        if latent_samples.shape[1] != latent_dim:
            raise RuntimeError(f"LED7/{animal}: unexpected latent dimension.")
        pooled_within_cov += weight * np.cov(latent_samples, rowvar=False)
    aggregate_cov = pooled_within_cov / effective_animal_count
    aggregate_cov += np.eye(latent_dim) * FULLRANK_COV_JITTER
    fullrank_init_scale_tril = np.linalg.cholesky(aggregate_cov)

    category_initialization[category] = {
        "warm_values": warm_values,
        "fullrank_init_scale_tril": fullrank_init_scale_tril,
        "animal_trial_counts": animal_trial_counts.to_dict(),
        "animal_weights": {
            str(animal): float(weight)
            for animal, weight in zip(ANIMALS, animal_weights)
        },
        "effective_animal_count": effective_animal_count,
        "condition_animal_counts": condition_animal_counts,
    }


# %%
# =============================================================================
# Run the four independent aggregate fits
# =============================================================================
for category_index, category in enumerate(CATEGORIES):
    valid_df = category_frames[category]
    condition_counts = (
        valid_df.groupby("condition_id", observed=True)
        .size()
        .rename("n_valid_trials")
        .reset_index()
    )
    base_condition_table = master_condition_table.merge(
        condition_counts,
        on="condition_id",
        validate="one_to_one",
    )
    data = make_jax_data(valid_df, category)
    initialization = category_initialization[category]

    model = lambda data: svi_utils.proactive_led_npl_alpha_valid_model(
        data,
        len(base_condition_table),
        fixed_proactive_params,
        category,
        K_max=K_MAX,
        n_quad=QUADRATURE_NODES,
    )

    def log_joint_from_values(values):
        log_joint, _ = log_density(model, (data,), {}, values)
        return log_joint

    for init_index, init_mode in enumerate(INIT_MODES):
        output_dir = OUTPUT_DIR / category / init_mode
        output_dir.mkdir(parents=True, exist_ok=True)
        run_summary_json = output_dir / "run_summary.json"
        if run_summary_json.exists() and not FORCE_RERUN:
            with run_summary_json.open() as handle:
                existing_summary = json.load(handle)
            if existing_summary.get("status") == "complete":
                print(f"Skipping completed {category}/{init_mode}: {run_summary_json}")
                continue

        warm_values = initialization["warm_values"]
        init_values = {
            name: (
                np.asarray(value, dtype=float).copy()
                if name == "t_E_aff"
                else float(value)
            )
            for name, value in warm_values.items()
        }
        if init_mode == "displaced":
            for name in svi_utils.GLOBAL_PARAM_NAMES:
                init_values[name] += DISPLACED_INIT_OFFSETS[name]
            init_values["t_E_aff"] += DISPLACED_INIT_OFFSETS["t_E_aff"]
        init_values = svi_utils.clip_init_to_hard_bounds(init_values)

        condition_table = base_condition_table.copy()
        condition_table["warm_initial_t_E_aff_s"] = warm_values["t_E_aff"]
        condition_table["initial_t_E_aff_s"] = init_values["t_E_aff"]
        condition_table["initial_t_E_aff_ms"] = 1000.0 * init_values["t_E_aff"]
        condition_table.to_csv(output_dir / "condition_table.csv", index=False)

        initial_log_joint = log_joint_from_values(init_values)
        initial_grad = jax.grad(log_joint_from_values)(init_values)
        gradients_finite = svi_utils.tree_all_finite(initial_grad)
        print(f"\n{'=' * 78}")
        print(f"Fitting LED7 super animal: {category}/{init_mode}")
        print(f"Valid trials: {len(valid_df)}")
        print(f"Conditions: {len(condition_table)}")
        print(f"Parameters: {latent_dim}")
        print(f"Initial log joint: {float(initial_log_joint):.8g}")
        print(f"Initial gradients finite: {gradients_finite}")
        print(f"Output: {output_dir.resolve()}")
        if not np.isfinite(float(initial_log_joint)) or not gradients_finite:
            raise RuntimeError(
                f"{category}/{init_mode}: non-finite initial density or gradients."
            )

        run_summary_json.write_text(
            json.dumps(
                {
                    "status": "running",
                    "category": category,
                    "initialization_mode": init_mode,
                    "valid_trial_count": int(len(valid_df)),
                    "output_dir": str(output_dir.resolve()),
                },
                indent=2,
            )
            + "\n"
        )

        guide = svi_utils.make_guide(
            model,
            "fullrank",
            init_values,
            fullrank_init_scale_tril=initialization["fullrank_init_scale_tril"],
        )
        svi = SVI(model, guide, make_optimizer(), Trace_ELBO())
        fit_start = perf_counter()
        fit_result, convergence_df, stop_reason, extended_after_150k = (
            run_svi_with_convergence_checks(
                svi,
                random.PRNGKey(
                    RNG_SEED + 1000 * category_index + 100 * init_index
                ),
                data,
            )
        )
        fit_elapsed_seconds = perf_counter() - fit_start
        losses = np.asarray(jax.device_get(fit_result.losses), dtype=float)

        posterior_samples = guide.sample_posterior(
            random.PRNGKey(
                RNG_SEED + 1000 * category_index + 100 * init_index + 1
            ),
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
        best_window_negative_elbo = float(convergence_df["mean_loss"].min())

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
        initialization_json = output_dir / "initialization_provenance.json"

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
        initialization_json.write_text(
            json.dumps(
                {
                    "mode": init_mode,
                    "reference_root": str(REFERENCE_ROOT.resolve()),
                    "animals": list(ANIMALS),
                    "animal_trial_counts": {
                        str(key): int(value)
                        for key, value in initialization[
                            "animal_trial_counts"
                        ].items()
                    },
                    "animal_weights": initialization["animal_weights"],
                    "effective_animal_count": initialization[
                        "effective_animal_count"
                    ],
                    "aggregate_covariance_scaling": "weighted within-animal covariance divided by effective animal count",
                    "displacement_offsets": (
                        DISPLACED_INIT_OFFSETS if init_mode == "displaced" else {}
                    ),
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
        ax.axvline(
            best_step, color="tab:green", linewidth=1.2, label="restored best"
        )
        ax.axvline(
            checked_step,
            color="tab:red",
            linestyle="--",
            linewidth=1.2,
            label="final checked",
        )
        ax.set_xlabel("SVI step")
        ax.set_ylabel("negative ELBO")
        ax.set_title(f"LED7 super animal {category}/{init_mode}")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(loss_png, dpi=200, bbox_inches="tight")
        plt.close(fig)

        config = {
            "model_name": "aggregate_fixed_proactive_lapse_plus_npl_alpha_valid_rt_choice",
            "animals": list(ANIMALS),
            "pooling": "trial_weighted",
            "category": category,
            "initialization_mode": init_mode,
            "valid_rt_window_s": [0.0, 1.0],
            "led_on_scope": (
                "bilateral only" if category == "on_bilateral" else "off"
            ),
            "parameter_count": latent_dim,
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
            "rng_seed": RNG_SEED + 1000 * category_index + 100 * init_index,
            "fixed_proactive_uncertainty_propagated": False,
            "include_exponential_lapse": True,
        }
        provenance = {
            "input_csv": str(DATA_CSV.resolve()),
            "reference_npl_root": str(REFERENCE_ROOT.resolve()),
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
            "warm_init_values": warm_values,
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
            "animals": list(ANIMALS),
            "category": category,
            "initialization_mode": init_mode,
            "fit_elapsed_seconds": fit_elapsed_seconds,
            "fit_elapsed_minutes": fit_elapsed_seconds / 60.0,
            "completed_steps": checked_step,
            "restored_best_step": best_step,
            "best_window_negative_elbo": best_window_negative_elbo,
            "extended_after_150k": extended_after_150k,
            "n_nonfinite_losses": n_nonfinite_losses,
            "all_posterior_samples_finite": all_posterior_finite,
            "valid_trial_count": int(len(valid_df)),
            "condition_count": int(len(condition_table)),
            "condition_trial_count_min": int(
                condition_table["n_valid_trials"].min()
            ),
            "condition_trial_count_max": int(
                condition_table["n_valid_trials"].max()
            ),
            "config": config,
            "provenance": provenance,
            "artifacts": {
                "posterior_samples_npz": str(sample_npz.resolve()),
                "guide_params_pkl": str(guide_params_pkl.resolve()),
                "posterior_summary_csv": str(posterior_summary_csv.resolve()),
                "condition_table_csv": str(
                    (output_dir / "condition_table.csv").resolve()
                ),
                "loss_csv": str(loss_csv.resolve()),
                "convergence_csv": str(convergence_csv.resolve()),
                "loss_png": str(loss_png.resolve()),
                "variational_posterior_bundle_pkl": str(bundle_pkl.resolve()),
                "fixed_proactive_lapse_params_json": str(
                    fixed_params_json.resolve()
                ),
                "initialization_provenance_json": str(
                    initialization_json.resolve()
                ),
            },
        }
        run_summary_json.write_text(json.dumps(run_summary, indent=2) + "\n")

        print(f"Fit elapsed: {fit_elapsed_seconds / 60.0:.2f} minutes")
        print(f"Restored best step: {best_step}")
        print(f"Final checked step: {checked_step}")
        print(f"Run status: {status}")
        if status == "failed_validation":
            raise RuntimeError(f"{category}/{init_mode}: finite-value validation failed.")


# %%
# =============================================================================
# Select the accepted branch for each category
# =============================================================================
selection_rows = []
for category in CATEGORIES:
    valid_df = category_frames[category]
    data = make_jax_data(valid_df, category)
    branch_rows = []
    for init_mode in ["warm", "displaced"]:
        branch_dir = OUTPUT_DIR / category / init_mode
        summary_path = branch_dir / "run_summary.json"
        sample_path = branch_dir / "main_fullrank_posterior_samples.npz"
        if not summary_path.exists() or not sample_path.exists():
            continue
        with summary_path.open() as handle:
            branch_summary = json.load(handle)
        with np.load(sample_path) as saved:
            posterior = {name: np.asarray(saved[name]) for name in saved.files}
        mean_loglike = posterior_mean_loglike(
            posterior, data, category, fixed_proactive_params
        )
        branch_rows.append(
            {
                "initialization_mode": init_mode,
                "status": branch_summary["status"],
                "best_window_negative_elbo": float(
                    branch_summary["best_window_negative_elbo"]
                ),
                "posterior_mean_loglike": mean_loglike,
                "restored_best_step": int(branch_summary["restored_best_step"]),
                "completed_steps": int(branch_summary["completed_steps"]),
                "fit_root": str(branch_dir.resolve()),
            }
        )

    complete_rows = [row for row in branch_rows if row["status"] == "complete"]
    selection_path = OUTPUT_DIR / category / "selection_summary.json"
    if len(complete_rows) != 2:
        selection = {
            "status": "inconclusive",
            "category": category,
            "reason": "both warm and displaced branches must patience-converge",
            "branches": branch_rows,
        }
    else:
        warm_row = next(
            row for row in complete_rows if row["initialization_mode"] == "warm"
        )
        displaced_row = next(
            row
            for row in complete_rows
            if row["initialization_mode"] == "displaced"
        )
        relative_elbo_difference = abs(
            warm_row["best_window_negative_elbo"]
            - displaced_row["best_window_negative_elbo"]
        ) / max(
            1.0,
            min(
                abs(warm_row["best_window_negative_elbo"]),
                abs(displaced_row["best_window_negative_elbo"]),
            ),
        )
        if relative_elbo_difference <= 0.001:
            accepted = max(
                complete_rows, key=lambda row: row["posterior_mean_loglike"]
            )
            selection_rule = "higher posterior-mean likelihood within 0.1% ELBO tie"
        else:
            accepted = min(
                complete_rows, key=lambda row: row["best_window_negative_elbo"]
            )
            selection_rule = "lower best-window negative ELBO"
        selection = {
            "status": "complete",
            "category": category,
            "accepted_initialization_mode": accepted["initialization_mode"],
            "accepted_fit_root": accepted["fit_root"],
            "selection_rule": selection_rule,
            "relative_best_window_elbo_difference": relative_elbo_difference,
            "branches": branch_rows,
        }
        selection_rows.append(selection)
    selection_path.write_text(json.dumps(selection, indent=2) + "\n")
    print(f"Selection summary: {selection_path.resolve()}")

aggregate_selection_path = OUTPUT_DIR / "aggregate_selection_summary.json"
aggregate_selection_path.write_text(
    json.dumps(
        {
            "status": (
                "complete" if len(selection_rows) == len(CATEGORIES) else "inconclusive"
            ),
            "animals": list(ANIMALS),
            "pooling": "trial_weighted",
            "categories": selection_rows,
        },
        indent=2,
    )
    + "\n"
)
print(f"Aggregate selection summary: {aggregate_selection_path.resolve()}")
