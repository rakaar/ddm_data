# %%
"""
Prototype one-animal SVI fit with condition-wise Gamma/Omega/delay+lapse parameters.

This is the large direct Gamma/Omega+lapse fit:

    gamma[ABL, ILD], omega[ABL, ILD], t_E_aff[ABL, ILD],
    plus global w, del_go, lapse_prob, and lapse_prob_right

The goal is only to smoke-test whether a 90+ dimensional SVI fit is numerically
viable for one animal before building any all-animal runner.
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
import time

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
ANIMAL_SVI_OUTPUT_ROOT = (
    REPO_DIR
    / "fit_animal_by_animal"
    / "numpyro_svi_npl_alpha_condition_delay_single_animal_outputs"
)

BATCH_NAME = os.environ.get("NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_BATCH", "LED8")
ANIMAL = int(os.environ.get("NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_ANIMAL", "105"))

GUIDE_KIND = os.environ.get("NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_GUIDE", "fullrank")
GUIDE_INIT_SCALE = float(os.environ.get("NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_GUIDE_INIT_SCALE", "0.02"))
LOWRANK_RANK = int(os.environ.get("NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_LOWRANK_RANK", "20"))
LEARNING_RATE = float(os.environ.get("NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_LR", "0.0003"))
OPTIMIZER_KIND = os.environ.get("NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_OPTIMIZER", "clipped_adam")
CLIP_NORM = float(os.environ.get("NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_CLIP_NORM", "2.0"))
SVI_STEPS = int(os.environ.get("NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_STEPS", "5000"))
SVI_CHECK_EVERY = int(os.environ.get("NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_CHECK_EVERY", "500"))
SVI_STOP_MODE = os.environ.get("NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_STOP_MODE", "legacy").strip().lower()
SVI_MIN_STEPS = int(os.environ.get("NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_MIN_STEPS", "0"))
SVI_EARLY_STOP = os.environ.get("NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_EARLY_STOP", "1").strip().lower() in {
    "1",
    "true",
    "yes",
}
SVI_REL_TOL = float(os.environ.get("NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_REL_TOL", "0.001"))
SVI_PATIENCE_WINDOWS = int(os.environ.get("NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_PATIENCE_WINDOWS", "3"))
SVI_MIN_IMPROVEMENT_REL = float(os.environ.get("NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_MIN_IMPROVEMENT_REL", "0.001"))
SVI_NO_IMPROVE_PATIENCE_WINDOWS = int(
    os.environ.get("NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_NO_IMPROVE_PATIENCE_WINDOWS", "5")
)
SVI_STABLE_UPDATE = os.environ.get("NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_STABLE_UPDATE", "1").strip().lower() in {
    "1",
    "true",
    "yes",
}
if SVI_STOP_MODE not in {"legacy", "stable_or_no_improve", "patience_restore_best"}:
    raise ValueError(
        "NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_STOP_MODE must be one of "
        "'legacy', 'stable_or_no_improve', or 'patience_restore_best', "
        f"got {SVI_STOP_MODE!r}."
    )
POSTERIOR_N_SAMPLES = int(os.environ.get("NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_POSTERIOR_N_SAMPLES", "2000"))
RNG_SEED = int(os.environ.get("NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_SEED", "0"))
K_MAX = int(os.environ.get("NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_K_MAX", "10"))
MIN_TRIALS_PER_CONDITION = int(os.environ.get("NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_MIN_TRIALS", "10"))

BATCH_T_TRUNC = {"LED34_even": 0.15}
DEFAULT_T_TRUNC = 0.3
P_0 = 1.0

BATCH_CSV = REPO_DIR / "raw_data" / "batch_csvs" / f"batch_{BATCH_NAME}_valid_and_aborts.csv"
ABORT_RESULT_PKL = REPO_DIR / "aborts_ipl_npl_time_fit_results" / f"results_{BATCH_NAME}_animal_{ANIMAL}.pkl"
ANIMAL_SVI_OUTPUT_DIR = ANIMAL_SVI_OUTPUT_ROOT / f"{BATCH_NAME}_{ANIMAL}"
ANIMAL_SVI_POSTERIOR_NPZ = ANIMAL_SVI_OUTPUT_DIR / "main_fullrank_posterior_samples.npz"
ANIMAL_SVI_CONDITION_TABLE = ANIMAL_SVI_OUTPUT_DIR / "condition_table.csv"
NO_LAPSE_BIG_SVI_OUTPUT_ROOT = (
    SCRIPT_DIR / "svi_big_gamma_omega_delay_patience12_restore_best_all_animals_outputs"
)
NO_LAPSE_BIG_SVI_OUTPUT_DIR = NO_LAPSE_BIG_SVI_OUTPUT_ROOT / f"{BATCH_NAME}_{ANIMAL}"
NO_LAPSE_BIG_SVI_PREFIX = f"{BATCH_NAME}_{ANIMAL}_big_gamma_omega_delay"
NO_LAPSE_BIG_SVI_CONDITION_SUMMARY = (
    NO_LAPSE_BIG_SVI_OUTPUT_DIR / f"{NO_LAPSE_BIG_SVI_PREFIX}_condition_summary.csv"
)
NO_LAPSE_BIG_SVI_POSTERIOR_SUMMARY = (
    NO_LAPSE_BIG_SVI_OUTPUT_DIR / f"{NO_LAPSE_BIG_SVI_PREFIX}_posterior_summary.csv"
)
VANILLA_LAPSE_OUTPUT_ROOT = (
    REPO_DIR
    / "fit_animal_by_animal"
    / "numpyro_svi_vanilla_lapse_condition_delay_patience12_min50k_restore_best_outputs"
)
NPL_ALPHA_LAPSE_OUTPUT_ROOT = (
    REPO_DIR
    / "fit_animal_by_animal"
    / "numpyro_svi_npl_alpha_lapse_condition_delay_patience12_min50k_restore_best_outputs"
)
COND_SVI_CACHE = (
    SCRIPT_DIR
    / "svi_gamma_omega_fixed_from_animal_svi_condition_delay_results"
    / "all_observed_with_30k_reruns"
    / "condition_gamma_omega_extraction_cache.csv"
)

OUTPUT_ROOT = Path(
    os.environ.get(
        "NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_OUTPUT_ROOT",
        str(SCRIPT_DIR / "svi_big_gamma_omega_delay_lapse_single_animal_outputs"),
    )
).expanduser()
if not OUTPUT_ROOT.is_absolute():
    OUTPUT_ROOT = (REPO_DIR / OUTPUT_ROOT).resolve()
OUTPUT_DIR = OUTPUT_ROOT / f"{BATCH_NAME}_{ANIMAL}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PREFIX = f"{BATCH_NAME}_{ANIMAL}_big_gamma_omega_delay_lapse"

SUMMARY_CSV = OUTPUT_DIR / f"{OUTPUT_PREFIX}_condition_summary.csv"
POSTERIOR_SUMMARY_CSV = OUTPUT_DIR / f"{OUTPUT_PREFIX}_posterior_summary.csv"
LOSS_CSV = OUTPUT_DIR / f"{OUTPUT_PREFIX}_loss.csv"
CONVERGENCE_CSV = OUTPUT_DIR / f"{OUTPUT_PREFIX}_convergence_checks.csv"
SAMPLES_NPZ = OUTPUT_DIR / f"{OUTPUT_PREFIX}_posterior_samples.npz"
BUNDLE_PKL = OUTPUT_DIR / f"{OUTPUT_PREFIX}_fit_bundle.pkl"
CONDITION_TABLE_CSV = OUTPUT_DIR / f"{OUTPUT_PREFIX}_condition_table.csv"
LOSS_FIG = OUTPUT_DIR / f"{OUTPUT_PREFIX}_loss.png"
PARAM_FIG = OUTPUT_DIR / f"{OUTPUT_PREFIX}_condition_params.png"


# %%
# =============================================================================
# Dependency preflight
# =============================================================================
required_modules = ["jax", "jaxlib", "numpyro"]
missing_modules = [module for module in required_modules if importlib.util.find_spec(module) is None]
if missing_modules:
    print("Missing dependencies for big Gamma/Omega/delay+lapse SVI:")
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
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from jax import random
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoLowRankMultivariateNormal, AutoMultivariateNormal, AutoNormal
from numpyro.infer.initialization import init_to_value
from numpyro.infer.svi import SVIRunResult
from numpyro.infer.util import log_density

sys.path.insert(0, str(SCRIPT_DIR))
from gamma_omega_alpha_utils import gamma_omega_alpha_model
import svi_gamma_omega_likelihood_utils as likelihood_utils


# %%
# =============================================================================
# Small helpers
# =============================================================================
def finite_tree(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return bool(all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves))


def finite_mean(values):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.nan
    return float(np.mean(values[finite]))


def summarize_samples(values):
    values = np.asarray(values, dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return {"mean": np.nan, "sd": np.nan, "q025": np.nan, "q500": np.nan, "q975": np.nan}
    return {
        "mean": float(np.mean(finite_values)),
        "sd": float(np.std(finite_values)),
        "q025": float(np.quantile(finite_values, 0.025)),
        "q500": float(np.quantile(finite_values, 0.5)),
        "q975": float(np.quantile(finite_values, 0.975)),
    }


def load_scalar_summary(summary_csv, parameter):
    summary_csv = Path(summary_csv)
    if not summary_csv.exists():
        return np.nan
    summary_df = pd.read_csv(summary_csv)
    match = summary_df[summary_df["parameter"].astype(str) == str(parameter)]
    if len(match) != 1:
        return np.nan
    return float(match.iloc[0]["mean"])


def make_optimizer():
    optimizer_kind = OPTIMIZER_KIND.strip().lower()
    if optimizer_kind in {"adam", "plain_adam"}:
        return numpyro.optim.Adam(LEARNING_RATE)
    if optimizer_kind in {"clipped_adam", "clipped-adam", "clip_adam"}:
        return numpyro.optim.ClippedAdam(LEARNING_RATE, clip_norm=CLIP_NORM)
    raise ValueError(f"Unknown NUMPYRO_BIG_GAMMA_OMEGA_DELAY_LAPSE_OPTIMIZER={OPTIMIZER_KIND!r}")


def make_guide(model, init_values):
    init_loc_fn = init_to_value(values=init_values)
    guide_kind = GUIDE_KIND.strip().lower()
    if guide_kind in {"meanfield", "autonormal", "normal"}:
        return AutoNormal(model, init_loc_fn=init_loc_fn)
    if guide_kind in {"fullrank", "multivariate", "automultivariate"}:
        return AutoMultivariateNormal(model, init_loc_fn=init_loc_fn, init_scale=GUIDE_INIT_SCALE)
    if guide_kind in {"lowrank", "autolowrank"}:
        return AutoLowRankMultivariateNormal(model, rank=LOWRANK_RANK, init_loc_fn=init_loc_fn)
    raise ValueError(f"Use fullrank, lowrank, or meanfield guide for this prototype, got {GUIDE_KIND!r}.")


def gamma_bounds_for_ild(ild):
    if ild > 0:
        return {
            "gamma_hard_low": -1.0,
            "gamma_hard_high": 5.0,
            "gamma_plausible_low": 0.0,
            "gamma_plausible_high": 3.0,
        }
    return {
        "gamma_hard_low": -5.0,
        "gamma_hard_high": 1.0,
        "gamma_plausible_low": -3.0,
        "gamma_plausible_high": 0.0,
    }


def trapezoidal_logpdf_jax(x, hard_low, plausible_low, plausible_high, hard_high):
    x = jnp.asarray(x)
    area = ((plausible_low - hard_low) + (hard_high - plausible_high)) / 2.0 + (
        plausible_high - plausible_low
    )
    h_max = 1.0 / area

    rising = ((x - hard_low) / (plausible_low - hard_low)) * h_max
    flat = jnp.full_like(x, h_max, dtype=jnp.result_type(x, jnp.float64))
    falling = ((hard_high - x) / (hard_high - plausible_high)) * h_max

    pdf = jnp.where(
        (hard_low <= x) & (x <= plausible_low),
        rising,
        jnp.where(
            (plausible_low < x) & (x < plausible_high),
            flat,
            jnp.where((plausible_high <= x) & (x <= hard_high), falling, 0.0),
        ),
    )
    return jnp.where(pdf > 0, jnp.log(pdf), -jnp.inf)


def sample_trapezoid(name, hard_low, hard_high, plausible_low, plausible_high):
    value = numpyro.sample(name, dist.Uniform(hard_low, hard_high))
    target_logpdf = trapezoidal_logpdf_jax(value, hard_low, plausible_low, plausible_high, hard_high)
    uniform_logpdf = -jnp.log(hard_high - hard_low)
    numpyro.factor(f"{name}_trapezoid_prior", target_logpdf - uniform_logpdf)
    return value


def sample_trapezoid_vector(name, hard_low, hard_high, plausible_low, plausible_high):
    hard_low = jnp.asarray(hard_low, dtype=jnp.float64)
    hard_high = jnp.asarray(hard_high, dtype=jnp.float64)
    plausible_low = jnp.asarray(plausible_low, dtype=jnp.float64)
    plausible_high = jnp.asarray(plausible_high, dtype=jnp.float64)
    value = numpyro.sample(name, dist.Uniform(hard_low, hard_high).to_event(1))
    target_logpdf = trapezoidal_logpdf_jax(value, hard_low, plausible_low, plausible_high, hard_high)
    uniform_logpdf = -jnp.log(hard_high - hard_low)
    numpyro.factor(f"{name}_trapezoid_prior", jnp.sum(target_logpdf - uniform_logpdf))
    return value


def big_gamma_omega_delay_lapse_model(data, n_conditions):
    condition_shape = (int(n_conditions),)
    params = {
        "gamma": sample_trapezoid_vector(
            "gamma",
            data["gamma_hard_low"],
            data["gamma_hard_high"],
            data["gamma_plausible_low"],
            data["gamma_plausible_high"],
        ),
        "omega": sample_trapezoid_vector(
            "omega",
            jnp.full(condition_shape, 0.1),
            jnp.full(condition_shape, 15.0),
            jnp.full(condition_shape, 2.0),
            jnp.full(condition_shape, 12.0),
        ),
        "t_E_aff": sample_trapezoid_vector(
            "t_E_aff",
            jnp.full(condition_shape, 0.0),
            jnp.full(condition_shape, 1.0),
            jnp.full(condition_shape, 0.01),
            jnp.full(condition_shape, 0.2),
        ),
        "w": sample_trapezoid("w", 0.3, 0.7, 0.4, 0.6),
        "del_go": sample_trapezoid("del_go", 0.0, 0.2, 0.02, 0.199),
        "lapse_prob": sample_trapezoid("lapse_prob", 0.0001, 0.2, 0.001, 0.1),
        "lapse_prob_right": sample_trapezoid("lapse_prob_right", 0.001, 0.999, 0.4, 0.6),
    }
    loglike = likelihood_utils.gamma_omega_delay_lapse_loglike(params, data, K_MAX)
    numpyro.factor("ddm_loglike", loglike)


def run_svi_with_convergence_checks(svi, rng_key, n_steps, data, n_conditions, run_label):
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
    stop_reason = "max_steps"

    print(
        f"\nRunning {run_label} for up to {n_steps} steps "
        f"with checks every {SVI_CHECK_EVERY} steps..."
    )
    print(
        f"Stopping mode: {SVI_STOP_MODE}; min_steps={SVI_MIN_STEPS}; "
        f"stable_patience={SVI_PATIENCE_WINDOWS}; no_improve_patience={SVI_NO_IMPROVE_PATIENCE_WINDOWS}"
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
            f"rel_change={rel_text}, best_delta={best_rel_text}, "
            f"slope/1k={slope_text}, stable={stable_window_count}/{SVI_PATIENCE_WINDOWS}, "
            f"no_improve={no_improve_window_count}/{SVI_NO_IMPROVE_PATIENCE_WINDOWS}, "
            f"best_chunk={best_window_chunk}, can_stop={can_stop_for_patience}, nonfinite={n_nonfinite}"
        )

        if n_nonfinite:
            stop_reason = f"nonfinite_losses_chunk_{chunk_index}"
            print(
                f"WARNING: stopping {run_label}; this window had non-finite losses. "
                f"Returning best state from chunk {best_window_chunk}."
            )
            break

        if not SVI_EARLY_STOP or not can_stop_for_patience:
            prev_window_mean = window_mean
            continue

        if SVI_STOP_MODE in {"legacy", "stable_or_no_improve"} and stable_window_count >= SVI_PATIENCE_WINDOWS:
            stop_reason = f"stable_{SVI_PATIENCE_WINDOWS}_windows"
            print(f"Stopping {run_label} early at step {completed_steps}: stable windows reached.")
            break
        if no_improve_window_count >= SVI_NO_IMPROVE_PATIENCE_WINDOWS:
            if SVI_STOP_MODE == "patience_restore_best":
                stop_reason = f"patience_restore_best_{SVI_NO_IMPROVE_PATIENCE_WINDOWS}_windows"
            else:
                stop_reason = f"no_improve_{SVI_NO_IMPROVE_PATIENCE_WINDOWS}_windows"
            print(
                f"Stopping {run_label} early at step {completed_steps}: no best-window improvement. "
                f"Returning best state from chunk {best_window_chunk}."
            )
            break

        prev_window_mean = window_mean

    losses = np.concatenate(all_losses) if all_losses else np.array([], dtype=float)
    if best_state is None:
        best_state = state
        best_params = svi.get_params(state)
    result = SVIRunResult(best_params, best_state, jnp.asarray(losses))
    convergence_df = pd.DataFrame(convergence_rows)
    return result, convergence_df, stop_reason


def make_trial_data(valid_df, V_A, theta_A, t_A_aff, T_trunc, total_fix_col):
    return {
        "total_fix": jnp.asarray(valid_df[total_fix_col].to_numpy(dtype=float)),
        "t_stim": jnp.asarray(valid_df["intended_fix"].to_numpy(dtype=float)),
        "choice": jnp.asarray(valid_df["choice"].to_numpy(dtype=int)),
        "condition_id": jnp.asarray(valid_df["condition_id"].to_numpy(dtype=int)),
        "mask": jnp.ones(len(valid_df), dtype=bool),
        "V_A": jnp.asarray(V_A, dtype=jnp.float64),
        "theta_A": jnp.asarray(theta_A, dtype=jnp.float64),
        "t_A_aff": jnp.asarray(t_A_aff, dtype=jnp.float64),
        "T_trunc": jnp.asarray(T_trunc, dtype=jnp.float64),
        "gamma_hard_low": jnp.asarray(condition_table["gamma_hard_low"].to_numpy(dtype=float), dtype=jnp.float64),
        "gamma_hard_high": jnp.asarray(condition_table["gamma_hard_high"].to_numpy(dtype=float), dtype=jnp.float64),
        "gamma_plausible_low": jnp.asarray(
            condition_table["gamma_plausible_low"].to_numpy(dtype=float), dtype=jnp.float64
        ),
        "gamma_plausible_high": jnp.asarray(
            condition_table["gamma_plausible_high"].to_numpy(dtype=float), dtype=jnp.float64
        ),
    }


def plot_loss(losses, convergence_df):
    losses = np.asarray(losses, dtype=float)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(np.arange(1, len(losses) + 1), losses, color="tab:blue", linewidth=1.0)
    if len(convergence_df):
        ax.scatter(
            convergence_df["end_step"],
            convergence_df["mean_loss"],
            color="tab:orange",
            s=28,
            label="window mean",
            zorder=3,
        )
        ax.legend(frameon=False)
    ax.set_xlabel("SVI step")
    ax.set_ylabel("negative ELBO")
    ax.set_title(f"{BATCH_NAME}/{ANIMAL} big Gamma/Omega/delay+lapse SVI loss")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(LOSS_FIG, dpi=200, bbox_inches="tight")
    return LOSS_FIG


def plot_condition_params(condition_summary_df):
    condition_summary_df = condition_summary_df.sort_values(["ABL", "ILD"]).reset_index(drop=True)
    abl_colors = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}
    plot_specs = [
        ("Gamma", "gamma_mean", "gamma_q025", "gamma_q975", 1.0),
        ("Omega", "omega_mean", "omega_q025", "omega_q975", 1.0),
        ("t_E_aff (ms)", "t_E_aff_mean", "t_E_aff_q025", "t_E_aff_q975", 1000.0),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
    for ax, (ylabel, mean_col, q025_col, q975_col, scale) in zip(axes, plot_specs):
        for abl, abl_df in condition_summary_df.groupby("ABL", sort=True):
            abl = int(abl)
            ilds = abl_df["ILD"].to_numpy(dtype=float)
            means = abl_df[mean_col].to_numpy(dtype=float) * scale
            q025 = abl_df[q025_col].to_numpy(dtype=float) * scale
            q975 = abl_df[q975_col].to_numpy(dtype=float) * scale
            yerr = np.vstack([means - q025, q975 - means])
            ax.errorbar(
                ilds,
                means,
                yerr=yerr,
                fmt="o-",
                color=abl_colors.get(abl, None),
                ecolor=abl_colors.get(abl, None),
                capsize=2.0,
                markersize=4,
                linewidth=1.2,
                label=f"ABL {abl}",
            )
        ax.set_ylabel(ylabel)
        ax.set_xlabel("ILD")
        ax.grid(True, alpha=0.25)
        if ylabel == "Gamma":
            ax.axhline(0.0, color="0.75", linewidth=0.8, zorder=0)
    unique_ilds = np.sort(condition_summary_df["ILD"].unique())
    for ax in axes:
        ax.set_xticks(unique_ilds)
        ax.set_xticklabels([f"{int(ild):+d}" for ild in unique_ilds], rotation=45, ha="right", fontsize=8)
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle(f"{BATCH_NAME}/{ANIMAL} condition-wise posterior means", y=1.02)
    fig.tight_layout()
    fig.savefig(PARAM_FIG, dpi=200, bbox_inches="tight")
    return PARAM_FIG


# %%
# =============================================================================
# Load inputs
# =============================================================================
print(f"Batch/animal: {BATCH_NAME}/{ANIMAL}")
print(f"Batch CSV: {BATCH_CSV}")
print(f"Abort params: {ABORT_RESULT_PKL}")
print(f"Animal SVI posterior: {ANIMAL_SVI_POSTERIOR_NPZ}")
print(f"Animal SVI condition table: {ANIMAL_SVI_CONDITION_TABLE}")
print(f"No-lapse big SVI condition summary for initialization: {NO_LAPSE_BIG_SVI_CONDITION_SUMMARY}")
print(f"No-lapse big SVI posterior summary for initialization: {NO_LAPSE_BIG_SVI_POSTERIOR_SUMMARY}")
print(f"Vanilla lapse reference root: {VANILLA_LAPSE_OUTPUT_ROOT}")
print(f"NPL+alpha lapse reference root: {NPL_ALPHA_LAPSE_OUTPUT_ROOT}")
print(f"Condition SVI cache for initialization: {COND_SVI_CACHE}")
print(f"Output folder: {OUTPUT_DIR}")
print(f"JAX default backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

for required_path in [BATCH_CSV, ABORT_RESULT_PKL, ANIMAL_SVI_POSTERIOR_NPZ, ANIMAL_SVI_CONDITION_TABLE]:
    if not required_path.exists():
        raise FileNotFoundError(required_path)

raw_df = pd.read_csv(BATCH_CSV)
if "choice" not in raw_df.columns:
    if "response_poke" not in raw_df.columns:
        raise KeyError("Need either `choice` or `response_poke` in the batch CSV.")
    raw_df["choice"] = raw_df["response_poke"].map({3: 1, 2: -1})

total_fix_col = "timed_fix" if "timed_fix" in raw_df.columns else "TotalFixTime"
valid_df = raw_df[
    (raw_df["animal"].astype(int) == int(ANIMAL))
    & (raw_df["success"].isin([1, -1]))
    & (raw_df["RTwrtStim"] > 0)
    & (raw_df["RTwrtStim"] <= 1)
].copy()
valid_df = valid_df.dropna(subset=[total_fix_col, "intended_fix", "ABL", "ILD", "choice", "RTwrtStim"])
valid_df["ABL"] = valid_df["ABL"].astype(int)
valid_df["ILD"] = valid_df["ILD"].astype(int)
valid_df["choice"] = valid_df["choice"].astype(int)

with ABORT_RESULT_PKL.open("rb") as f:
    abort_result = pickle.load(f)
abort_samples = abort_result["vbmc_aborts_results"]
V_A = float(np.mean(abort_samples["V_A_samples"]))
theta_A = float(np.mean(abort_samples["theta_A_samples"]))
t_A_aff = float(np.mean(abort_samples["t_A_aff_samp"]))
T_trunc = BATCH_T_TRUNC.get(BATCH_NAME, DEFAULT_T_TRUNC)

animal_svi_samples = np.load(ANIMAL_SVI_POSTERIOR_NPZ)
animal_svi_means = {key: finite_mean(animal_svi_samples[key]) for key in animal_svi_samples.files if key != "t_E_aff"}

condition_table = pd.read_csv(ANIMAL_SVI_CONDITION_TABLE)
condition_table["ABL"] = condition_table["ABL"].astype(int)
condition_table["ILD"] = condition_table["ILD"].astype(int)
condition_table["condition_id"] = condition_table["condition_id"].astype(int)
condition_table = condition_table.sort_values("condition_id").reset_index(drop=True)
n_conditions = int(len(condition_table))
expected_ids = np.arange(n_conditions, dtype=int)
if not np.array_equal(condition_table["condition_id"].to_numpy(dtype=int), expected_ids):
    raise RuntimeError("condition_table.csv must have contiguous condition_id values from 0.")

for bounds_key in ["gamma_hard_low", "gamma_hard_high", "gamma_plausible_low", "gamma_plausible_high"]:
    condition_table[bounds_key] = np.nan
for idx, row in condition_table.iterrows():
    bounds = gamma_bounds_for_ild(int(row["ILD"]))
    for key, value in bounds.items():
        condition_table.loc[idx, key] = value

condition_table_lookup = condition_table[["ABL", "ILD", "condition_id"]].copy()
valid_df = valid_df.merge(condition_table_lookup, on=["ABL", "ILD"], how="left", validate="many_to_one")
missing_condition_mask = valid_df["condition_id"].isna()
if missing_condition_mask.any():
    missing_conditions = (
        valid_df.loc[missing_condition_mask, ["ABL", "ILD"]]
        .drop_duplicates()
        .sort_values(["ABL", "ILD"])
    )
    print(
        "\nDropping valid trials absent from the saved condition table:\n"
        f"{missing_conditions.to_string(index=False)}\n"
        f"  trials dropped: {int(missing_condition_mask.sum())}"
    )
    valid_df = valid_df.loc[~missing_condition_mask].copy()
if len(valid_df) == 0:
    raise RuntimeError("No fitting trials remain after filtering to the saved condition table.")
valid_df["condition_id"] = valid_df["condition_id"].astype(int)

condition_counts = valid_df.groupby("condition_id").size().rename("n_trials").reset_index()
condition_table = condition_table.merge(condition_counts, on="condition_id", how="left")
condition_table["n_trials"] = condition_table["n_trials"].fillna(0).astype(int)
too_few = condition_table[condition_table["n_trials"] < MIN_TRIALS_PER_CONDITION]
if len(too_few):
    raise RuntimeError(
        "Some saved conditions have too few fitting trials after RT filtering:\n"
        + too_few[["ABL", "ILD", "n_trials"]].to_string(index=False)
    )

cond_cache_df = None
if COND_SVI_CACHE.exists():
    cond_cache_df = pd.read_csv(COND_SVI_CACHE)
    cond_cache_df["animal"] = cond_cache_df["animal"].astype(int)
    cond_cache_df["ABL"] = cond_cache_df["ABL"].astype(int)
    cond_cache_df["ILD"] = cond_cache_df["ILD"].astype(int)

no_lapse_condition_df = None
if NO_LAPSE_BIG_SVI_CONDITION_SUMMARY.exists():
    no_lapse_condition_df = pd.read_csv(NO_LAPSE_BIG_SVI_CONDITION_SUMMARY)
    no_lapse_condition_df["animal"] = no_lapse_condition_df["animal"].astype(int)
    no_lapse_condition_df["ABL"] = no_lapse_condition_df["ABL"].astype(int)
    no_lapse_condition_df["ILD"] = no_lapse_condition_df["ILD"].astype(int)

gamma_init = np.zeros(n_conditions, dtype=float)
omega_init = np.zeros(n_conditions, dtype=float)
t_E_aff_init = np.asarray(animal_svi_samples["t_E_aff"], dtype=float).mean(axis=0)
init_source = []

if t_E_aff_init.shape[0] != n_conditions:
    raise RuntimeError(f"t_E_aff samples have {t_E_aff_init.shape[0]} conditions, expected {n_conditions}.")

for row in condition_table.itertuples(index=False):
    condition_id = int(row.condition_id)
    no_lapse_match = pd.DataFrame()
    if no_lapse_condition_df is not None:
        no_lapse_match = no_lapse_condition_df[
            (no_lapse_condition_df["batch_name"].astype(str) == str(BATCH_NAME))
            & (no_lapse_condition_df["animal"] == int(ANIMAL))
            & (no_lapse_condition_df["ABL"] == int(row.ABL))
            & (no_lapse_condition_df["ILD"] == int(row.ILD))
        ]
    cache_match = pd.DataFrame()
    if cond_cache_df is not None:
        cache_match = cond_cache_df[
            (cond_cache_df["batch_name"].astype(str) == str(BATCH_NAME))
            & (cond_cache_df["animal"] == int(ANIMAL))
            & (cond_cache_df["ABL"] == int(row.ABL))
            & (cond_cache_df["ILD"] == int(row.ILD))
        ]

    if len(no_lapse_match) == 1:
        gamma_value = float(no_lapse_match.iloc[0]["gamma_mean"])
        omega_value = float(no_lapse_match.iloc[0]["omega_mean"])
        t_E_aff_init[condition_id] = float(no_lapse_match.iloc[0]["t_E_aff_mean"])
        source = "big_svi_patience12_no_lapse"
    elif len(cache_match) == 1:
        gamma_value = float(cache_match.iloc[0]["condition_gamma"])
        omega_value = float(cache_match.iloc[0]["condition_omega"])
        source = "condition_svi_cache"
    else:
        gamma_value, omega_value = gamma_omega_alpha_model(
            int(row.ABL),
            int(row.ILD),
            animal_svi_means["rate_lambda"],
            animal_svi_means["rate_norm_l"],
            animal_svi_means["alpha"],
            animal_svi_means["theta_E"],
            animal_svi_means["T_0"],
            P_0,
        )
        gamma_value = float(gamma_value)
        omega_value = float(omega_value)
        source = "npl_alpha_expression"

    gamma_init[condition_id] = np.clip(
        gamma_value,
        float(row.gamma_hard_low) + 1e-5,
        float(row.gamma_hard_high) - 1e-5,
    )
    omega_init[condition_id] = np.clip(omega_value, 0.10001, 14.99999)
    t_E_aff_init[condition_id] = np.clip(t_E_aff_init[condition_id], 1e-5, 0.99999)
    init_source.append(source)

condition_table["init_gamma"] = gamma_init
condition_table["init_omega"] = omega_init
condition_table["init_t_E_aff_s"] = t_E_aff_init
condition_table["init_t_E_aff_ms"] = t_E_aff_init * 1000.0
condition_table["gamma_omega_init_source"] = init_source

no_lapse_w = load_scalar_summary(NO_LAPSE_BIG_SVI_POSTERIOR_SUMMARY, "w")
no_lapse_del_go = load_scalar_summary(NO_LAPSE_BIG_SVI_POSTERIOR_SUMMARY, "del_go")
w_init_source = "big_svi_patience12_no_lapse" if np.isfinite(no_lapse_w) else "npl_alpha_condition_delay_svi"
del_go_init_source = "big_svi_patience12_no_lapse" if np.isfinite(no_lapse_del_go) else "npl_alpha_condition_delay_svi"
w_init = float(np.clip(no_lapse_w if np.isfinite(no_lapse_w) else animal_svi_means["w"], 0.30001, 0.69999))
del_go_init = float(np.clip(no_lapse_del_go if np.isfinite(no_lapse_del_go) else animal_svi_means["del_go"], 1e-5, 0.19999))

lapse_reference_values = []
lapse_right_reference_values = []
for lapse_root in [VANILLA_LAPSE_OUTPUT_ROOT, NPL_ALPHA_LAPSE_OUTPUT_ROOT]:
    lapse_summary_csv = lapse_root / f"{BATCH_NAME}_{ANIMAL}" / "main_fullrank_posterior_summary.csv"
    lapse_value = load_scalar_summary(lapse_summary_csv, "lapse_prob")
    lapse_right_value = load_scalar_summary(lapse_summary_csv, "lapse_prob_right")
    if np.isfinite(lapse_value):
        lapse_reference_values.append(lapse_value)
    if np.isfinite(lapse_right_value):
        lapse_right_reference_values.append(lapse_right_value)
lapse_init_source = "mean_vanilla_and_npl_lapse_svi" if lapse_reference_values else "default"
lapse_prob_init = float(np.clip(np.mean(lapse_reference_values) if lapse_reference_values else 0.02, 0.00011, 0.19999))
lapse_prob_right_init = float(
    np.clip(np.mean(lapse_right_reference_values) if lapse_right_reference_values else 0.5, 0.00101, 0.99899)
)
init_values = {
    "gamma": gamma_init,
    "omega": omega_init,
    "t_E_aff": t_E_aff_init,
    "w": w_init,
    "del_go": del_go_init,
    "lapse_prob": lapse_prob_init,
    "lapse_prob_right": lapse_prob_right_init,
}

data = make_trial_data(valid_df, V_A, theta_A, t_A_aff, T_trunc, total_fix_col)
latent_dim = 3 * n_conditions + 4

print("\nFit inputs:")
print(f"  valid RT-filtered trials: {len(valid_df)}")
print(f"  conditions: {n_conditions}")
print(f"  latent dimension: {latent_dim}")
print(f"  total fixation column: {total_fix_col}")
print(f"  T_trunc: {T_trunc:.3f} s")
print(f"  abort means: V_A={V_A:.6g}, theta_A={theta_A:.6g}, t_A_aff={1e3 * t_A_aff:.3f} ms")
print(f"  init w={w_init:.6f} ({w_init_source}), del_go={1e3 * del_go_init:.3f} ms ({del_go_init_source})")
print(
    f"  init lapse_prob={lapse_prob_init:.6f}, lapse_prob_right={lapse_prob_right_init:.6f} "
    f"({lapse_init_source})"
)
print(f"  gamma/omega init sources:\n{condition_table['gamma_omega_init_source'].value_counts().to_string()}")
print(
    f"  SVI config: guide={GUIDE_KIND}, lr={LEARNING_RATE:g}, clip_norm={CLIP_NORM:g}, "
    f"steps={SVI_STEPS}, check_every={SVI_CHECK_EVERY}, stop_mode={SVI_STOP_MODE}, "
    f"min_steps={SVI_MIN_STEPS}, no_improve_patience={SVI_NO_IMPROVE_PATIENCE_WINDOWS}, "
    f"posterior_samples={POSTERIOR_N_SAMPLES}"
)


# %%
# =============================================================================
# Initial finite check
# =============================================================================
def log_joint_from_values(values):
    log_joint, _ = log_density(big_gamma_omega_delay_lapse_model, (data, n_conditions), {}, values)
    return log_joint


initial_log_joint = float(log_joint_from_values(init_values))
initial_grad = jax.grad(log_joint_from_values)(init_values)
initial_grad_finite = finite_tree(initial_grad)
initial_loglike = float(likelihood_utils.gamma_omega_delay_lapse_loglike(init_values, data, K_MAX))

print("\nInitial check:")
print(f"  log joint: {initial_log_joint:.6f}")
print(f"  data loglike: {initial_loglike:.6f}")
print(f"  gradient finite: {initial_grad_finite}")
if (not np.isfinite(initial_log_joint)) or (not np.isfinite(initial_loglike)) or (not initial_grad_finite):
    raise RuntimeError("Initial log joint, data loglike, or gradient is non-finite.")


# %%
# =============================================================================
# Run SVI
# =============================================================================
guide = make_guide(big_gamma_omega_delay_lapse_model, init_values)
svi = SVI(big_gamma_omega_delay_lapse_model, guide, make_optimizer(), Trace_ELBO())

fit_start = time.perf_counter()
svi_result, convergence_df, stop_reason = run_svi_with_convergence_checks(
    svi,
    random.PRNGKey(RNG_SEED),
    SVI_STEPS,
    data,
    n_conditions,
    f"{BATCH_NAME}/{ANIMAL} big_gamma_omega_delay_lapse",
)
elapsed_s = time.perf_counter() - fit_start
losses_np = np.asarray(jax.device_get(svi_result.losses), dtype=float)
if not np.all(np.isfinite(losses_np)):
    nonfinite_steps = np.where(~np.isfinite(losses_np))[0][:20]
    raise RuntimeError(f"SVI produced non-finite losses at 0-indexed steps {nonfinite_steps}.")

posterior_samples = guide.sample_posterior(
    random.PRNGKey(RNG_SEED + 1),
    svi_result.params,
    sample_shape=(POSTERIOR_N_SAMPLES,),
)
posterior_np = {key: np.asarray(value) for key, value in posterior_samples.items()}
posterior_all_finite = bool(all(np.all(np.isfinite(value)) for value in posterior_np.values()))
if not posterior_all_finite:
    raise RuntimeError("Posterior samples contain NaN/Inf.")

print("\nSVI completed:")
print(f"  elapsed: {elapsed_s:.2f} s ({elapsed_s / 60.0:.2f} min)")
print(f"  stop_reason: {stop_reason}")
print(f"  loss first/last: {losses_np[0]:.6f} / {losses_np[-1]:.6f}")
print(f"  posterior samples finite: {posterior_all_finite}")


# %%
# =============================================================================
# Save outputs
# =============================================================================
condition_summary_rows = []
posterior_summary_rows = []
for row in condition_table.itertuples(index=False):
    condition_id = int(row.condition_id)
    gamma_summary = summarize_samples(posterior_np["gamma"][:, condition_id])
    omega_summary = summarize_samples(posterior_np["omega"][:, condition_id])
    delay_summary = summarize_samples(posterior_np["t_E_aff"][:, condition_id])

    condition_summary_rows.append(
        {
            "batch_name": BATCH_NAME,
            "animal": ANIMAL,
            "condition_id": condition_id,
            "ABL": int(row.ABL),
            "ILD": int(row.ILD),
            "n_trials": int(row.n_trials),
            "init_source": row.gamma_omega_init_source,
            "init_gamma": float(row.init_gamma),
            "init_omega": float(row.init_omega),
            "init_t_E_aff_s": float(row.init_t_E_aff_s),
            "init_t_E_aff_ms": float(row.init_t_E_aff_ms),
            "gamma_mean": gamma_summary["mean"],
            "gamma_sd": gamma_summary["sd"],
            "gamma_q025": gamma_summary["q025"],
            "gamma_q500": gamma_summary["q500"],
            "gamma_q975": gamma_summary["q975"],
            "omega_mean": omega_summary["mean"],
            "omega_sd": omega_summary["sd"],
            "omega_q025": omega_summary["q025"],
            "omega_q500": omega_summary["q500"],
            "omega_q975": omega_summary["q975"],
            "t_E_aff_mean": delay_summary["mean"],
            "t_E_aff_sd": delay_summary["sd"],
            "t_E_aff_q025": delay_summary["q025"],
            "t_E_aff_q500": delay_summary["q500"],
            "t_E_aff_q975": delay_summary["q975"],
            "t_E_aff_ms_mean": delay_summary["mean"] * 1000.0,
            "t_E_aff_ms_sd": delay_summary["sd"] * 1000.0,
            "t_E_aff_ms_q025": delay_summary["q025"] * 1000.0,
            "t_E_aff_ms_q500": delay_summary["q500"] * 1000.0,
            "t_E_aff_ms_q975": delay_summary["q975"] * 1000.0,
        }
    )

    for param_name, summary in [
        ("gamma", gamma_summary),
        ("omega", omega_summary),
        ("t_E_aff", delay_summary),
    ]:
        posterior_summary_rows.append(
            {
                "batch_name": BATCH_NAME,
                "animal": ANIMAL,
                "condition_id": condition_id,
                "ABL": int(row.ABL),
                "ILD": int(row.ILD),
                "parameter": param_name,
                **summary,
                "n_samples": POSTERIOR_N_SAMPLES,
            }
        )

for scalar_name in ["w", "del_go", "lapse_prob", "lapse_prob_right"]:
    scalar_summary = summarize_samples(posterior_np[scalar_name])
    posterior_summary_rows.append(
        {
            "batch_name": BATCH_NAME,
            "animal": ANIMAL,
            "condition_id": np.nan,
            "ABL": np.nan,
            "ILD": np.nan,
            "parameter": scalar_name,
            **scalar_summary,
            "n_samples": POSTERIOR_N_SAMPLES,
        }
    )

condition_summary_df = pd.DataFrame(condition_summary_rows)
posterior_summary_df = pd.DataFrame(posterior_summary_rows)
loss_df = pd.DataFrame({"step": np.arange(1, len(losses_np) + 1), "loss": losses_np})
convergence_df = convergence_df.copy()
convergence_df.insert(0, "batch_name", BATCH_NAME)
convergence_df.insert(1, "animal", ANIMAL)

condition_summary_df.to_csv(SUMMARY_CSV, index=False)
posterior_summary_df.to_csv(POSTERIOR_SUMMARY_CSV, index=False)
loss_df.to_csv(LOSS_CSV, index=False)
convergence_df.to_csv(CONVERGENCE_CSV, index=False)
condition_table.to_csv(CONDITION_TABLE_CSV, index=False)
np.savez(SAMPLES_NPZ, **posterior_np)

bundle = {
    "batch_name": BATCH_NAME,
    "animal": ANIMAL,
    "config": {
        "GUIDE_KIND": GUIDE_KIND,
        "GUIDE_INIT_SCALE": GUIDE_INIT_SCALE,
        "LOWRANK_RANK": LOWRANK_RANK,
        "LEARNING_RATE": LEARNING_RATE,
        "OPTIMIZER_KIND": OPTIMIZER_KIND,
        "CLIP_NORM": CLIP_NORM,
        "SVI_STEPS": SVI_STEPS,
        "SVI_CHECK_EVERY": SVI_CHECK_EVERY,
        "SVI_STOP_MODE": SVI_STOP_MODE,
        "SVI_MIN_STEPS": SVI_MIN_STEPS,
        "SVI_EARLY_STOP": SVI_EARLY_STOP,
        "SVI_REL_TOL": SVI_REL_TOL,
        "SVI_PATIENCE_WINDOWS": SVI_PATIENCE_WINDOWS,
        "SVI_MIN_IMPROVEMENT_REL": SVI_MIN_IMPROVEMENT_REL,
        "SVI_NO_IMPROVE_PATIENCE_WINDOWS": SVI_NO_IMPROVE_PATIENCE_WINDOWS,
        "POSTERIOR_N_SAMPLES": POSTERIOR_N_SAMPLES,
        "K_MAX": K_MAX,
        "latent_dim": latent_dim,
    },
    "condition_table": condition_table.to_dict("records"),
    "condition_summary_rows": condition_summary_rows,
    "posterior_summary_rows": posterior_summary_rows,
    "posterior_samples": posterior_np,
    "loss_trace": loss_df,
    "convergence_checks": convergence_df,
    "initial_values": {key: np.asarray(value) for key, value in init_values.items()},
    "animal_svi_means": animal_svi_means,
    "initialization_sources": {
        "w": w_init_source,
        "del_go": del_go_init_source,
        "lapse": lapse_init_source,
    },
    "abort_means": {"V_A": V_A, "theta_A": theta_A, "t_A_aff": t_A_aff},
    "initial_log_joint": initial_log_joint,
    "initial_loglike": initial_loglike,
    "initial_grad_finite": initial_grad_finite,
    "stop_reason": stop_reason,
    "elapsed_s": elapsed_s,
    "output_files": {
        "summary_csv": str(SUMMARY_CSV),
        "posterior_summary_csv": str(POSTERIOR_SUMMARY_CSV),
        "loss_csv": str(LOSS_CSV),
        "convergence_csv": str(CONVERGENCE_CSV),
        "samples_npz": str(SAMPLES_NPZ),
    },
}
with BUNDLE_PKL.open("wb") as f:
    pickle.dump(bundle, f)

loss_fig = plot_loss(losses_np, convergence_df)
param_fig = plot_condition_params(condition_summary_df)

print("\nSaved outputs:")
for path in [
    SUMMARY_CSV,
    POSTERIOR_SUMMARY_CSV,
    LOSS_CSV,
    CONVERGENCE_CSV,
    SAMPLES_NPZ,
    BUNDLE_PKL,
    CONDITION_TABLE_CSV,
    loss_fig,
    param_fig,
]:
    print(f"  {path}")

print("\nScalar posterior summaries:")
for scalar_name in ["w", "del_go", "lapse_prob", "lapse_prob_right"]:
    summary = summarize_samples(posterior_np[scalar_name])
    if scalar_name == "del_go":
        print(
            f"  {scalar_name}: mean={summary['mean'] * 1000.0:.3f} ms "
            f"[{summary['q025'] * 1000.0:.3f}, {summary['q975'] * 1000.0:.3f}] ms"
        )
    else:
        print(f"  {scalar_name}: mean={summary['mean']:.6f} [{summary['q025']:.6f}, {summary['q975']:.6f}]")

# %%
