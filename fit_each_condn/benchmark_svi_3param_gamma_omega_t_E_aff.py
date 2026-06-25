# %%
"""
Benchmark NumPyro SVI for one condition-by-condition 3-param fit.

This is the fair speed/comparison test against the older VBMC condition fits:
for each selected ABL/ILD condition, fit Gamma, Omega, and t_E_aff while fixing
the animal-level w and del_go to posterior means from the current animal-wise
NPL + alpha + condition-delay SVI fit.

Default benchmark animal:
    LED8/105

Default conditions:
    (ABL, ILD) = (20, -1), (20, +1), (40, -4), (40, +4), (60, -16), (60, +16)
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

BATCH_NAME = os.environ.get("NUMPYRO_COND_SVI_BATCH", "LED8")
ANIMAL = int(os.environ.get("NUMPYRO_COND_SVI_ANIMAL", "105"))

CONDITIONS_TEXT = os.environ.get(
    "NUMPYRO_COND_SVI_CONDITIONS",
    "20:-1,20:1,40:-4,40:4,60:-16,60:16",
)

GUIDE_KIND = os.environ.get("NUMPYRO_COND_SVI_GUIDE", "fullrank")
LEARNING_RATE = float(os.environ.get("NUMPYRO_COND_SVI_LR", "0.001"))
OPTIMIZER_KIND = os.environ.get("NUMPYRO_COND_SVI_OPTIMIZER", "clipped_adam")
CLIP_NORM = float(os.environ.get("NUMPYRO_COND_SVI_CLIP_NORM", "5.0"))
GUIDE_INIT_SCALE = float(os.environ.get("NUMPYRO_COND_SVI_GUIDE_INIT_SCALE", "0.05"))
SVI_STEPS = int(os.environ.get("NUMPYRO_COND_SVI_STEPS", "12000"))
SVI_CHECK_EVERY = int(os.environ.get("NUMPYRO_COND_SVI_CHECK_EVERY", "1000"))
SVI_EARLY_STOP = os.environ.get("NUMPYRO_COND_SVI_EARLY_STOP", "1").strip().lower() in {
    "1",
    "true",
    "yes",
}
SVI_REL_TOL = float(os.environ.get("NUMPYRO_COND_SVI_REL_TOL", "0.001"))
SVI_PATIENCE_WINDOWS = int(os.environ.get("NUMPYRO_COND_SVI_PATIENCE_WINDOWS", "3"))
SVI_MIN_IMPROVEMENT_REL = float(os.environ.get("NUMPYRO_COND_SVI_MIN_IMPROVEMENT_REL", "0.001"))
SVI_NO_IMPROVE_PATIENCE_WINDOWS = int(
    os.environ.get("NUMPYRO_COND_SVI_NO_IMPROVE_PATIENCE_WINDOWS", "5")
)
SVI_STABLE_UPDATE = os.environ.get("NUMPYRO_COND_SVI_STABLE_UPDATE", "1").strip().lower() in {
    "1",
    "true",
    "yes",
}
POSTERIOR_N_SAMPLES = int(os.environ.get("NUMPYRO_COND_SVI_POSTERIOR_N_SAMPLES", "10000"))
RNG_SEED = int(os.environ.get("NUMPYRO_COND_SVI_SEED", "0"))
K_MAX = int(os.environ.get("NUMPYRO_COND_SVI_K_MAX", "10"))

MIN_TRIALS_PER_CONDITION = int(os.environ.get("NUMPYRO_COND_SVI_MIN_TRIALS", "10"))
VBMC_REFERENCE_MINUTES_PER_CONDITION = float(
    os.environ.get("VBMC_REFERENCE_MINUTES_PER_CONDITION", "4.5")
)

BATCH_T_TRUNC = {"LED34_even": 0.15}
DEFAULT_T_TRUNC = 0.3

BATCH_CSV = REPO_DIR / "raw_data" / "batch_csvs" / f"batch_{BATCH_NAME}_valid_and_aborts.csv"
ABORT_RESULT_PKL = (
    REPO_DIR / "aborts_ipl_npl_time_fit_results" / f"results_{BATCH_NAME}_animal_{ANIMAL}.pkl"
)
ANIMAL_SVI_OUTPUT_DIR = (
    REPO_DIR
    / "fit_animal_by_animal"
    / "numpyro_svi_npl_alpha_condition_delay_single_animal_outputs"
    / f"{BATCH_NAME}_{ANIMAL}"
)
ANIMAL_SVI_POSTERIOR_NPZ = ANIMAL_SVI_OUTPUT_DIR / "main_fullrank_posterior_samples.npz"
ANIMAL_SVI_CONDITION_TABLE = ANIMAL_SVI_OUTPUT_DIR / "condition_table.csv"
VBMC_CACHE = (
    SCRIPT_DIR
    / "condition_t_E_aff_fixed_gamma_omega_comparison"
    / "condition_gamma_omega_extraction_cache.csv"
)
if not VBMC_CACHE.exists():
    VBMC_CACHE = (
        SCRIPT_DIR
        / "abl_specific_ild2_gamma_omega_comparison"
        / "condition_gamma_omega_extraction_cache.csv"
    )

OUTPUT_DIR = SCRIPT_DIR / "svi_3param_gamma_omega_t_E_aff_benchmark"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PREFIX = f"{BATCH_NAME}_{ANIMAL}_svi_3param_benchmark"


# %%
# =============================================================================
# Dependency preflight
# =============================================================================
required_modules = ["jax", "jaxlib", "numpyro", "corner"]
missing_modules = [module for module in required_modules if importlib.util.find_spec(module) is None]
if missing_modules:
    print("Missing dependencies for the condition SVI benchmark:")
    for module in missing_modules:
        print(f"  - {module}")
    print("\nUse the repository virtual environment and install the NumPyro/JAX stack there.")
    raise SystemExit(1)


# %%
# =============================================================================
# Imports after dependency preflight
# =============================================================================
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import corner
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from jax import random
from jax.scipy.special import erf, erfc
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoMultivariateNormal, AutoNormal
from numpyro.infer.initialization import init_to_value
from numpyro.infer.svi import SVIRunResult
from numpyro.infer.util import log_density

sys.path.insert(0, str(SCRIPT_DIR))
from gamma_omega_alpha_utils import gamma_omega_alpha_model


# %%
# =============================================================================
# Small configuration helpers
# =============================================================================
def parse_conditions(text):
    conditions = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        abl_text, ild_text = chunk.replace("/", ":").split(":")
        conditions.append((int(float(abl_text)), int(float(ild_text))))
    return conditions


def finite_tree(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return bool(all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves))


def make_optimizer():
    optimizer_kind = OPTIMIZER_KIND.strip().lower()
    if optimizer_kind in {"adam", "plain_adam"}:
        return numpyro.optim.Adam(LEARNING_RATE)
    if optimizer_kind in {"clipped_adam", "clipped-adam", "clip_adam"}:
        return numpyro.optim.ClippedAdam(LEARNING_RATE, clip_norm=CLIP_NORM)
    raise ValueError(f"Unknown NUMPYRO_COND_SVI_OPTIMIZER={OPTIMIZER_KIND!r}")


def make_guide(model, init_values):
    init_loc_fn = init_to_value(values=init_values)
    guide_kind = GUIDE_KIND.strip().lower()
    if guide_kind in {"meanfield", "autonormal", "normal"}:
        return AutoNormal(model, init_loc_fn=init_loc_fn)
    if guide_kind in {"fullrank", "multivariate", "automultivariate"}:
        return AutoMultivariateNormal(
            model,
            init_loc_fn=init_loc_fn,
            init_scale=GUIDE_INIT_SCALE,
        )
    raise ValueError(
        "This benchmark keeps the guide simple for parameter comparison. "
        f"Use fullrank or meanfield, got {GUIDE_KIND!r}."
    )


# %%
# =============================================================================
# JAX likelihood pieces: direct Gamma/Omega version of the existing VBMC model
# =============================================================================
def phi_jax(x):
    return (1.0 / jnp.sqrt(2.0 * jnp.pi)) * jnp.exp(-0.5 * x**2)


def normal_cdf_jax(x):
    return 0.5 * (1.0 + erf(x / jnp.sqrt(2.0)))


def mills_ratio_jax(x):
    x = jnp.clip(x, -37.0, 37.0)
    scaled_x = x / jnp.sqrt(2.0)
    erfcx_approx = jnp.exp(scaled_x**2) * erfc(scaled_x)
    return jnp.sqrt(jnp.pi / 2.0) * erfcx_approx


def rho_A_t_jax(t, V_A, theta_A):
    t = jnp.asarray(t, dtype=jnp.float64)
    safe_t = jnp.maximum(t, 1e-12)
    rho = (
        theta_A
        / jnp.sqrt(2.0 * jnp.pi * safe_t**3)
        * jnp.exp(-0.5 * (V_A**2) * ((safe_t - theta_A / V_A) ** 2) / safe_t)
    )
    return jnp.where(t > 0, rho, 0.0)


def cum_A_t_jax(t, V_A, theta_A):
    t = jnp.asarray(t, dtype=jnp.float64)
    safe_t = jnp.maximum(t, 1e-12)
    term1 = normal_cdf_jax(V_A * (safe_t - theta_A / V_A) / jnp.sqrt(safe_t))
    term2 = jnp.exp(2.0 * V_A * theta_A) * normal_cdf_jax(
        -V_A * (safe_t + theta_A / V_A) / jnp.sqrt(safe_t)
    )
    return jnp.where(t > 0, term1 + term2, 0.0)


def CDF_E_gamma_omega_with_w_jax(t, gamma, omega, bound, w, K_max):
    t_original = jnp.asarray(t, dtype=jnp.float64)
    gamma = jnp.asarray(gamma, dtype=jnp.float64)
    omega = jnp.asarray(omega, dtype=jnp.float64)
    w = jnp.asarray(w, dtype=jnp.float64)
    bound = jnp.asarray(bound)

    a = 2.0
    v = jnp.where(bound == 1, -gamma, gamma)
    w_bound = jnp.where(bound == 1, 1.0 - w, w)

    t_eff = omega * t_original
    shape = jnp.broadcast_shapes(jnp.shape(t_eff), jnp.shape(v), jnp.shape(w_bound))
    valid = jnp.broadcast_to(t_original, shape) > 0
    safe_t = jnp.where(valid, jnp.broadcast_to(t_eff, shape), 1e-12)

    v_full = jnp.broadcast_to(v, shape)
    w_full = jnp.broadcast_to(w_bound, shape)
    result = jnp.exp(-v_full * a * w_full - (v_full**2) * safe_t / 2.0)

    k_arr = jnp.arange(K_max + 1)
    extra_shape = (1,) * len(shape) + (K_max + 1,)
    k_b = k_arr.reshape(extra_shape)
    t_b = safe_t[..., None]
    v_b = v_full[..., None]
    w_b = w_full[..., None]

    r_k = jnp.where(k_b % 2 == 0, k_b * a + a * w_b, k_b * a + a * (1.0 - w_b))
    sqrt_t = jnp.sqrt(t_b)
    term1 = phi_jax(r_k / sqrt_t)
    term2 = mills_ratio_jax((r_k - v_b * t_b) / sqrt_t) + mills_ratio_jax(
        (r_k + v_b * t_b) / sqrt_t
    )
    summation = jnp.sum(((-1.0) ** k_b) * term1 * term2, axis=-1)

    return jnp.where(valid, result * summation, 0.0)


def rho_E_gamma_omega_with_w_jax(t, gamma, omega, bound, w, K_max):
    t_original = jnp.asarray(t, dtype=jnp.float64)
    gamma = jnp.asarray(gamma, dtype=jnp.float64)
    omega = jnp.asarray(omega, dtype=jnp.float64)
    w = jnp.asarray(w, dtype=jnp.float64)
    bound = jnp.asarray(bound)

    a = 2.0
    v = jnp.where(bound == 1, -gamma, gamma)
    w_bound = jnp.where(bound == 1, 1.0 - w, w)

    t_eff = omega * t_original
    shape = jnp.broadcast_shapes(jnp.shape(t_eff), jnp.shape(v), jnp.shape(w_bound))
    valid = jnp.broadcast_to(t_original, shape) > 0
    safe_t = jnp.where(valid, jnp.broadcast_to(t_eff, shape), 1e-12)

    v_full = jnp.broadcast_to(v, shape)
    w_full = jnp.broadcast_to(w_bound, shape)
    non_sum_term = (
        (1.0 / a**2)
        * (a**3 / jnp.sqrt(2.0 * jnp.pi * safe_t**3))
        * jnp.exp(-v_full * a * w_full - (v_full**2) * safe_t / 2.0)
    )

    K_half = int(K_max / 2)
    k_vals = jnp.linspace(-K_half, K_half, 2 * K_half + 1)
    extra_shape = (1,) * len(shape) + (2 * K_half + 1,)
    k_b = k_vals.reshape(extra_shape)
    t_b = safe_t[..., None]
    w_b = w_full[..., None]

    sum_w_term = w_b + 2.0 * k_b
    sum_exp_term = jnp.exp(-(a**2 * (w_b + 2.0 * k_b) ** 2) / (2.0 * t_b))
    sum_result = jnp.sum(sum_w_term * sum_exp_term, axis=-1)

    density = non_sum_term * sum_result
    density = jnp.where(density <= 0, 1e-16, density)
    return jnp.where(valid, density * jnp.broadcast_to(omega, shape), 0.0)


def up_or_down_gamma_omega_with_w_jax(
    t,
    bound,
    V_A,
    theta_A,
    t_A_aff,
    t_stim,
    gamma,
    omega,
    t_E_aff,
    del_go,
    w,
    K_max,
):
    t = jnp.asarray(t, dtype=jnp.float64)
    t_stim = jnp.asarray(t_stim, dtype=jnp.float64)

    t2 = t - t_stim - t_E_aff + del_go
    t1 = t - t_stim - t_E_aff

    p_A = rho_A_t_jax(t - t_A_aff, V_A, theta_A)
    p_EA_hits_either_bound = (
        CDF_E_gamma_omega_with_w_jax(t2, gamma, omega, 1, w, K_max)
        + CDF_E_gamma_omega_with_w_jax(t2, gamma, omega, -1, w, K_max)
    )
    random_readout_if_EA_survives = 0.5 * (1.0 - p_EA_hits_either_bound)

    p_E_bound_cum = CDF_E_gamma_omega_with_w_jax(
        t2, gamma, omega, bound, w, K_max
    ) - CDF_E_gamma_omega_with_w_jax(t1, gamma, omega, bound, w, K_max)
    p_E_bound = rho_E_gamma_omega_with_w_jax(
        t - t_stim - t_E_aff, gamma, omega, bound, w, K_max
    )
    c_A = cum_A_t_jax(t - t_A_aff, V_A, theta_A)

    return p_A * (random_readout_if_EA_survives + p_E_bound_cum) + p_E_bound * (1.0 - c_A)


def cum_pro_and_reactive_gamma_omega_with_w_jax(
    t,
    c_A_trunc_time,
    V_A,
    theta_A,
    t_A_aff,
    t_stim,
    gamma,
    omega,
    t_E_aff,
    w,
    K_max,
):
    t = jnp.asarray(t, dtype=jnp.float64)
    c_A = cum_A_t_jax(t - t_A_aff, V_A, theta_A)
    trunc_denom = 1.0 - cum_A_t_jax(c_A_trunc_time - t_A_aff, V_A, theta_A)
    c_A = jnp.where(t < c_A_trunc_time, 0.0, c_A / trunc_denom)

    c_E = CDF_E_gamma_omega_with_w_jax(
        t - t_stim - t_E_aff, gamma, omega, 1, w, K_max
    ) + CDF_E_gamma_omega_with_w_jax(t - t_stim - t_E_aff, gamma, omega, -1, w, K_max)
    return c_A + c_E - c_A * c_E


def condition_gamma_omega_loglike(params, data, K_max):
    pdf = up_or_down_gamma_omega_with_w_jax(
        data["total_fix"],
        data["choice"],
        data["V_A"],
        data["theta_A"],
        data["t_A_aff"],
        data["t_stim"],
        params["gamma"],
        params["omega"],
        params["t_E_aff"],
        data["del_go"],
        data["w"],
        K_max,
    )
    trunc_factor = (
        cum_pro_and_reactive_gamma_omega_with_w_jax(
            data["t_stim"] + 1.0,
            data["T_trunc"],
            data["V_A"],
            data["theta_A"],
            data["t_A_aff"],
            data["t_stim"],
            params["gamma"],
            params["omega"],
            params["t_E_aff"],
            data["w"],
            K_max,
        )
        - cum_pro_and_reactive_gamma_omega_with_w_jax(
            data["t_stim"],
            data["T_trunc"],
            data["V_A"],
            data["theta_A"],
            data["t_A_aff"],
            data["t_stim"],
            params["gamma"],
            params["omega"],
            params["t_E_aff"],
            data["w"],
            K_max,
        )
    )

    normalized_pdf = jnp.maximum(pdf / (trunc_factor + 1e-20), 1e-50)
    log_pdf = jnp.log(normalized_pdf)
    return jnp.sum(jnp.where(data["mask"], log_pdf, 0.0))


# %%
# =============================================================================
# Priors and model
# =============================================================================
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
    target_logpdf = trapezoidal_logpdf_jax(
        value,
        hard_low,
        plausible_low,
        plausible_high,
        hard_high,
    )
    uniform_logpdf = -jnp.log(hard_high - hard_low)
    numpyro.factor(f"{name}_trapezoid_prior", target_logpdf - uniform_logpdf)
    return value


def condition_gamma_omega_t_E_aff_model(data):
    params = {
        "gamma": sample_trapezoid(
            "gamma",
            data["gamma_hard_low"],
            data["gamma_hard_high"],
            data["gamma_plausible_low"],
            data["gamma_plausible_high"],
        ),
        "omega": sample_trapezoid("omega", 0.1, 15.0, 2.0, 12.0),
        "t_E_aff": sample_trapezoid("t_E_aff", 0.0, 1.0, 0.01, 0.2),
    }
    loglike = condition_gamma_omega_loglike(params, data, K_MAX)
    numpyro.factor("ddm_loglike", loglike)


# %%
# =============================================================================
# SVI runner
# =============================================================================
def run_svi_with_convergence_checks(svi, rng_key, n_steps, data, run_label):
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
    while completed_steps < n_steps:
        chunk_index = len(convergence_rows) + 1
        chunk_steps = min(SVI_CHECK_EVERY, n_steps - completed_steps)
        start_step = completed_steps + 1
        end_step = completed_steps + chunk_steps

        chunk_result = svi.run(
            random.fold_in(rng_key, chunk_index),
            chunk_steps,
            data,
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
            f"best_chunk={best_window_chunk}, nonfinite={n_nonfinite}"
        )

        if n_nonfinite:
            stop_reason = f"nonfinite_losses_chunk_{chunk_index}"
            print(
                f"WARNING: stopping {run_label}; this window had non-finite losses. "
                f"Returning best state from chunk {best_window_chunk}."
            )
            break
        if SVI_EARLY_STOP and stable_window_count >= SVI_PATIENCE_WINDOWS:
            stop_reason = f"stable_{SVI_PATIENCE_WINDOWS}_windows"
            print(f"Stopping {run_label} early at step {completed_steps}: stable windows reached.")
            break
        if SVI_EARLY_STOP and no_improve_window_count >= SVI_NO_IMPROVE_PATIENCE_WINDOWS:
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


# %%
# =============================================================================
# Plot helpers
# =============================================================================
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


def finite_corner_data(samples, labels, plot_name):
    clean = np.asarray(samples, dtype=float)
    clean = clean[np.all(np.isfinite(clean), axis=1)]
    if clean.shape[0] < 20:
        print(f"{plot_name}: only {clean.shape[0]} finite samples; skipping corner plot.")
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


def make_padded_condition_data(df_cond, max_n_trials, fixed_values, bounds):
    n_trials = len(df_cond)
    mask = np.zeros(max_n_trials, dtype=bool)
    mask[:n_trials] = True

    total_fix = np.full(max_n_trials, 0.5, dtype=float)
    t_stim = np.full(max_n_trials, 0.3, dtype=float)
    choice = np.ones(max_n_trials, dtype=float)

    total_fix[:n_trials] = df_cond[fixed_values["total_fix_col"]].to_numpy(dtype=float)
    t_stim[:n_trials] = df_cond["intended_fix"].to_numpy(dtype=float)
    choice[:n_trials] = df_cond["choice"].to_numpy(dtype=float)

    data = {
        "total_fix": jnp.asarray(total_fix),
        "t_stim": jnp.asarray(t_stim),
        "choice": jnp.asarray(choice),
        "mask": jnp.asarray(mask),
        "V_A": jnp.asarray(fixed_values["V_A"], dtype=jnp.float64),
        "theta_A": jnp.asarray(fixed_values["theta_A"], dtype=jnp.float64),
        "t_A_aff": jnp.asarray(fixed_values["t_A_aff"], dtype=jnp.float64),
        "w": jnp.asarray(fixed_values["w"], dtype=jnp.float64),
        "del_go": jnp.asarray(fixed_values["del_go"], dtype=jnp.float64),
        "T_trunc": jnp.asarray(fixed_values["T_trunc"], dtype=jnp.float64),
        "gamma_hard_low": jnp.asarray(bounds["gamma_hard_low"], dtype=jnp.float64),
        "gamma_hard_high": jnp.asarray(bounds["gamma_hard_high"], dtype=jnp.float64),
        "gamma_plausible_low": jnp.asarray(bounds["gamma_plausible_low"], dtype=jnp.float64),
        "gamma_plausible_high": jnp.asarray(bounds["gamma_plausible_high"], dtype=jnp.float64),
    }
    return data


# %%
# =============================================================================
# Load inputs
# =============================================================================
conditions_to_fit = parse_conditions(CONDITIONS_TEXT)

print(f"Benchmark batch/animal: {BATCH_NAME}/{ANIMAL}")
print(f"Conditions: {conditions_to_fit}")
print(f"Batch CSV: {BATCH_CSV}")
print(f"Abort params: {ABORT_RESULT_PKL}")
print(f"Animal SVI posterior: {ANIMAL_SVI_POSTERIOR_NPZ}")
print(f"Animal SVI condition table: {ANIMAL_SVI_CONDITION_TABLE}")
print(f"VBMC comparison cache: {VBMC_CACHE}")
print(f"Output folder: {OUTPUT_DIR}")
print(f"JAX default backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

for required_path in [BATCH_CSV, ABORT_RESULT_PKL, ANIMAL_SVI_POSTERIOR_NPZ, ANIMAL_SVI_CONDITION_TABLE, VBMC_CACHE]:
    if not required_path.exists():
        raise FileNotFoundError(required_path)

raw_df = pd.read_csv(BATCH_CSV)
if "choice" not in raw_df.columns:
    if "response_poke" not in raw_df.columns:
        raise KeyError("Need either `choice` or `response_poke` in the batch CSV.")
    raw_df["choice"] = raw_df["response_poke"].map({3: 1, 2: -1})

total_fix_col = "timed_fix" if "timed_fix" in raw_df.columns else "TotalFixTime"
valid_df = raw_df[
    (raw_df["animal"].astype(int) == ANIMAL)
    & (raw_df["success"].isin([1, -1]))
    & (raw_df["RTwrtStim"] > 0)
    & (raw_df["RTwrtStim"] <= 1)
].copy()
valid_df = valid_df.dropna(subset=[total_fix_col, "intended_fix", "ABL", "ILD", "choice", "RTwrtStim"])
valid_df["ABL"] = valid_df["ABL"].astype(int)
valid_df["ILD"] = valid_df["ILD"].astype(int)
valid_df["choice"] = valid_df["choice"].astype(int)

with open(ABORT_RESULT_PKL, "rb") as f:
    abort_result = pickle.load(f)
abort_samples = abort_result["vbmc_aborts_results"]
V_A = float(np.mean(abort_samples["V_A_samples"]))
theta_A = float(np.mean(abort_samples["theta_A_samples"]))
t_A_aff = float(np.mean(abort_samples["t_A_aff_samp"]))

animal_svi_samples = np.load(ANIMAL_SVI_POSTERIOR_NPZ)
animal_svi_condition_table = pd.read_csv(ANIMAL_SVI_CONDITION_TABLE)
animal_svi_means = {key: float(np.mean(animal_svi_samples[key])) for key in animal_svi_samples.files if key != "t_E_aff"}
fixed_w = animal_svi_means["w"]
fixed_del_go = animal_svi_means["del_go"]

vbmc_cache_df = pd.read_csv(VBMC_CACHE)

print("\nFixed values for this benchmark:")
print(f"  w from current animal SVI: {fixed_w:.6f}")
print(f"  del_go from current animal SVI: {fixed_del_go:.6f}")
print(f"  abort V_A={V_A:.6f}, theta_A={theta_A:.6f}, t_A_aff={t_A_aff:.6f}")
print(f"  total fixation column: {total_fix_col}")
print(f"  T_trunc for {BATCH_NAME}: {BATCH_T_TRUNC.get(BATCH_NAME, DEFAULT_T_TRUNC):.3f} s")
print(f"  valid RT-filtered trials for animal: {len(valid_df)}")


# %%
# =============================================================================
# Prepare selected conditions
# =============================================================================
condition_frames = []
for cond_ABL, cond_ILD in conditions_to_fit:
    df_cond = valid_df[(valid_df["ABL"] == cond_ABL) & (valid_df["ILD"] == cond_ILD)].copy()
    if len(df_cond) < MIN_TRIALS_PER_CONDITION:
        raise RuntimeError(
            f"{BATCH_NAME}/{ANIMAL} ABL={cond_ABL}, ILD={cond_ILD} has only "
            f"{len(df_cond)} trials after RT filtering."
        )
    condition_frames.append({"ABL": cond_ABL, "ILD": cond_ILD, "df": df_cond})
    print(f"  ABL={cond_ABL:2d}, ILD={cond_ILD:+3d}: {len(df_cond)} trials")

max_n_trials = max(len(item["df"]) for item in condition_frames)
print(f"\nPadded trial length for all benchmark conditions: {max_n_trials}")

fixed_values = {
    "V_A": V_A,
    "theta_A": theta_A,
    "t_A_aff": t_A_aff,
    "w": fixed_w,
    "del_go": fixed_del_go,
    "T_trunc": BATCH_T_TRUNC.get(BATCH_NAME, DEFAULT_T_TRUNC),
    "total_fix_col": total_fix_col,
}


# %%
# =============================================================================
# Run SVI condition by condition
# =============================================================================
summary_rows = []
posterior_summary_rows = []
loss_rows = []
convergence_tables = []
posterior_npz_payload = {}
posterior_by_condition = {}

for condition_index, condition_item in enumerate(condition_frames):
    cond_ABL = condition_item["ABL"]
    cond_ILD = condition_item["ILD"]
    df_cond = condition_item["df"]
    condition_label = f"ABL{cond_ABL}_ILD{cond_ILD:+d}"
    condition_key = f"cond{condition_index:02d}_ABL{cond_ABL}_ILD{'m' if cond_ILD < 0 else 'p'}{abs(cond_ILD)}"

    if cond_ILD > 0:
        gamma_bounds = {
            "gamma_hard_low": -1.0,
            "gamma_hard_high": 5.0,
            "gamma_plausible_low": 0.0,
            "gamma_plausible_high": 3.0,
        }
    else:
        gamma_bounds = {
            "gamma_hard_low": -5.0,
            "gamma_hard_high": 1.0,
            "gamma_plausible_low": -3.0,
            "gamma_plausible_high": 0.0,
        }

    condition_table_match = animal_svi_condition_table[
        (animal_svi_condition_table["ABL"].astype(float) == float(cond_ABL))
        & (animal_svi_condition_table["ILD"].astype(float) == float(cond_ILD))
    ]
    if len(condition_table_match) != 1:
        raise RuntimeError(f"Could not find exactly one animal-SVI condition row for {condition_label}.")
    condition_id = int(condition_table_match.iloc[0]["condition_id"])
    init_t_E_aff = float(np.mean(animal_svi_samples["t_E_aff"][:, condition_id]))

    init_gamma, init_omega = gamma_omega_alpha_model(
        cond_ABL,
        cond_ILD,
        animal_svi_means["rate_lambda"],
        animal_svi_means["rate_norm_l"],
        animal_svi_means["alpha"],
        animal_svi_means["theta_E"],
        animal_svi_means["T_0"],
        1.0,
    )
    init_gamma = float(
        np.clip(
            init_gamma,
            gamma_bounds["gamma_hard_low"] + 1e-5,
            gamma_bounds["gamma_hard_high"] - 1e-5,
        )
    )
    init_omega = float(np.clip(init_omega, 0.10001, 14.99999))
    init_t_E_aff = float(np.clip(init_t_E_aff, 1e-5, 0.99999))
    init_values = {"gamma": init_gamma, "omega": init_omega, "t_E_aff": init_t_E_aff}

    data = make_padded_condition_data(df_cond, max_n_trials, fixed_values, gamma_bounds)

    def log_joint_from_values(values):
        log_joint, _ = log_density(condition_gamma_omega_t_E_aff_model, (data,), {}, values)
        return log_joint

    initial_log_joint = float(log_joint_from_values(init_values))
    initial_grad = jax.grad(log_joint_from_values)(init_values)
    initial_grad_finite = finite_tree(initial_grad)

    print("\n" + "=" * 72)
    print(f"{condition_label}: n_trials={len(df_cond)}, condition_id={condition_id}")
    print(
        f"Initial values: gamma={init_gamma:.6g}, omega={init_omega:.6g}, "
        f"t_E_aff={init_t_E_aff * 1000.0:.3f} ms"
    )
    print(f"Initial log joint={initial_log_joint:.6f}, grad finite={initial_grad_finite}")
    if (not np.isfinite(initial_log_joint)) or (not initial_grad_finite):
        raise RuntimeError(f"Initial log joint or gradient is non-finite for {condition_label}.")

    guide = make_guide(condition_gamma_omega_t_E_aff_model, init_values)
    svi = SVI(condition_gamma_omega_t_E_aff_model, guide, make_optimizer(), Trace_ELBO())

    condition_start = time.perf_counter()
    svi_result, convergence_df, stop_reason = run_svi_with_convergence_checks(
        svi,
        random.PRNGKey(RNG_SEED + condition_index),
        SVI_STEPS,
        data,
        condition_label,
    )
    elapsed_s = time.perf_counter() - condition_start

    posterior_samples = guide.sample_posterior(
        random.PRNGKey(RNG_SEED + 1000 + condition_index),
        svi_result.params,
        sample_shape=(POSTERIOR_N_SAMPLES,),
    )
    posterior_np = {key: np.asarray(value) for key, value in posterior_samples.items()}
    posterior_all_finite = bool(all(np.all(np.isfinite(value)) for value in posterior_np.values()))
    posterior_by_condition[condition_key] = posterior_np
    for param_name, values in posterior_np.items():
        posterior_npz_payload[f"{condition_key}_{param_name}"] = values

    gamma_summary = summarize_samples(posterior_np["gamma"])
    omega_summary = summarize_samples(posterior_np["omega"])
    t_E_aff_summary = summarize_samples(posterior_np["t_E_aff"])

    vbmc_row = vbmc_cache_df[
        (vbmc_cache_df["batch_name"] == BATCH_NAME)
        & (vbmc_cache_df["animal"].astype(int) == ANIMAL)
        & (vbmc_cache_df["ABL"].astype(int) == cond_ABL)
        & (vbmc_cache_df["ILD"].astype(int) == cond_ILD)
    ]
    if len(vbmc_row) == 1:
        vbmc_gamma = float(vbmc_row.iloc[0]["condition_gamma"])
        vbmc_omega = float(vbmc_row.iloc[0]["condition_omega"])
        vbmc_t_E_aff_s = float(vbmc_row.iloc[0]["condition_t_E_aff_s"])
        vbmc_source_pkl = str(vbmc_row.iloc[0]["source_pkl"])
    else:
        vbmc_gamma = np.nan
        vbmc_omega = np.nan
        vbmc_t_E_aff_s = np.nan
        vbmc_source_pkl = ""
        print(f"WARNING: missing VBMC cache row for {condition_label}.")

    losses_np = np.asarray(jax.device_get(svi_result.losses), dtype=float)
    for step_idx, loss_value in enumerate(losses_np, start=1):
        loss_rows.append(
            {
                "batch_name": BATCH_NAME,
                "animal": ANIMAL,
                "condition_index": condition_index,
                "ABL": cond_ABL,
                "ILD": cond_ILD,
                "step": step_idx,
                "loss": float(loss_value),
            }
        )

    convergence_df = convergence_df.copy()
    convergence_df.insert(0, "condition_index", condition_index)
    convergence_df.insert(1, "ABL", cond_ABL)
    convergence_df.insert(2, "ILD", cond_ILD)
    convergence_tables.append(convergence_df)

    row = {
        "batch_name": BATCH_NAME,
        "animal": ANIMAL,
        "condition_index": condition_index,
        "condition_id_from_animal_svi": condition_id,
        "ABL": cond_ABL,
        "ILD": cond_ILD,
        "n_trials": len(df_cond),
        "padded_n_trials": max_n_trials,
        "guide_kind": GUIDE_KIND,
        "learning_rate": LEARNING_RATE,
        "clip_norm": CLIP_NORM,
        "requested_steps": SVI_STEPS,
        "completed_steps": int(len(losses_np)),
        "stop_reason": stop_reason,
        "elapsed_s": elapsed_s,
        "elapsed_min": elapsed_s / 60.0,
        "vbmc_reference_min_per_condition": VBMC_REFERENCE_MINUTES_PER_CONDITION,
        "speedup_vs_vbmc_reference": (VBMC_REFERENCE_MINUTES_PER_CONDITION * 60.0) / elapsed_s,
        "loss_first": float(losses_np[0]) if len(losses_np) else np.nan,
        "loss_last": float(losses_np[-1]) if len(losses_np) else np.nan,
        "best_window_mean_loss": float(convergence_df["best_mean_loss_so_far"].iloc[-1])
        if len(convergence_df)
        else np.nan,
        "initial_log_joint": initial_log_joint,
        "initial_grad_finite": initial_grad_finite,
        "posterior_all_finite": posterior_all_finite,
        "fixed_w": fixed_w,
        "fixed_del_go": fixed_del_go,
        "V_A": V_A,
        "theta_A": theta_A,
        "t_A_aff": t_A_aff,
        "T_trunc": fixed_values["T_trunc"],
        "init_gamma_npl_alpha": init_gamma,
        "init_omega_npl_alpha": init_omega,
        "init_t_E_aff_s": init_t_E_aff,
        "init_t_E_aff_ms": init_t_E_aff * 1000.0,
        "svi_gamma_mean": gamma_summary["mean"],
        "svi_gamma_sd": gamma_summary["sd"],
        "svi_gamma_q025": gamma_summary["q025"],
        "svi_gamma_q500": gamma_summary["q500"],
        "svi_gamma_q975": gamma_summary["q975"],
        "svi_omega_mean": omega_summary["mean"],
        "svi_omega_sd": omega_summary["sd"],
        "svi_omega_q025": omega_summary["q025"],
        "svi_omega_q500": omega_summary["q500"],
        "svi_omega_q975": omega_summary["q975"],
        "svi_t_E_aff_s_mean": t_E_aff_summary["mean"],
        "svi_t_E_aff_s_sd": t_E_aff_summary["sd"],
        "svi_t_E_aff_s_q025": t_E_aff_summary["q025"],
        "svi_t_E_aff_s_q500": t_E_aff_summary["q500"],
        "svi_t_E_aff_s_q975": t_E_aff_summary["q975"],
        "svi_t_E_aff_ms_mean": t_E_aff_summary["mean"] * 1000.0,
        "svi_t_E_aff_ms_sd": t_E_aff_summary["sd"] * 1000.0,
        "svi_t_E_aff_ms_q025": t_E_aff_summary["q025"] * 1000.0,
        "svi_t_E_aff_ms_q500": t_E_aff_summary["q500"] * 1000.0,
        "svi_t_E_aff_ms_q975": t_E_aff_summary["q975"] * 1000.0,
        "vbmc_gamma": vbmc_gamma,
        "vbmc_omega": vbmc_omega,
        "vbmc_t_E_aff_s": vbmc_t_E_aff_s,
        "vbmc_t_E_aff_ms": vbmc_t_E_aff_s * 1000.0 if np.isfinite(vbmc_t_E_aff_s) else np.nan,
        "vbmc_source_pkl": vbmc_source_pkl,
        "svi_minus_vbmc_gamma": gamma_summary["mean"] - vbmc_gamma,
        "svi_minus_vbmc_omega": omega_summary["mean"] - vbmc_omega,
        "svi_minus_vbmc_t_E_aff_ms": (t_E_aff_summary["mean"] - vbmc_t_E_aff_s) * 1000.0
        if np.isfinite(vbmc_t_E_aff_s)
        else np.nan,
    }
    summary_rows.append(row)

    for param_name, summary in [
        ("gamma", gamma_summary),
        ("omega", omega_summary),
        ("t_E_aff_s", t_E_aff_summary),
    ]:
        posterior_summary_rows.append(
            {
                "batch_name": BATCH_NAME,
                "animal": ANIMAL,
                "condition_index": condition_index,
                "condition_id_from_animal_svi": condition_id,
                "ABL": cond_ABL,
                "ILD": cond_ILD,
                "parameter": param_name,
                **summary,
                "n_samples": POSTERIOR_N_SAMPLES,
            }
        )

    print(
        f"{condition_label} done in {elapsed_s:.2f} s "
        f"({elapsed_s / 60.0:.2f} min), speedup vs {VBMC_REFERENCE_MINUTES_PER_CONDITION:g} min ref: "
        f"{(VBMC_REFERENCE_MINUTES_PER_CONDITION * 60.0) / elapsed_s:.2f}x"
    )
    print(
        f"  SVI mean: gamma={gamma_summary['mean']:.4g}, omega={omega_summary['mean']:.4g}, "
        f"t_E_aff={t_E_aff_summary['mean'] * 1000.0:.2f} ms"
    )
    print(
        f"  VBMC mean: gamma={vbmc_gamma:.4g}, omega={vbmc_omega:.4g}, "
        f"t_E_aff={vbmc_t_E_aff_s * 1000.0 if np.isfinite(vbmc_t_E_aff_s) else np.nan:.2f} ms"
    )


# %%
# =============================================================================
# Save tabular outputs
# =============================================================================
summary_df = pd.DataFrame(summary_rows)
posterior_summary_df = pd.DataFrame(posterior_summary_rows)
loss_df = pd.DataFrame(loss_rows)
convergence_df = pd.concat(convergence_tables, ignore_index=True) if convergence_tables else pd.DataFrame()

summary_csv = OUTPUT_DIR / f"{OUTPUT_PREFIX}_summary.csv"
posterior_summary_csv = OUTPUT_DIR / f"{OUTPUT_PREFIX}_posterior_summary.csv"
loss_csv = OUTPUT_DIR / f"{OUTPUT_PREFIX}_loss_by_condition.csv"
convergence_csv = OUTPUT_DIR / f"{OUTPUT_PREFIX}_convergence_checks.csv"
samples_npz = OUTPUT_DIR / f"{OUTPUT_PREFIX}_posterior_samples.npz"
bundle_pkl = OUTPUT_DIR / f"{OUTPUT_PREFIX}_svi_params_bundle.pkl"

summary_df.to_csv(summary_csv, index=False)
posterior_summary_df.to_csv(posterior_summary_csv, index=False)
loss_df.to_csv(loss_csv, index=False)
convergence_df.to_csv(convergence_csv, index=False)
np.savez(samples_npz, **posterior_npz_payload)

bundle = {
    "config": {
        "BATCH_NAME": BATCH_NAME,
        "ANIMAL": ANIMAL,
        "conditions_to_fit": conditions_to_fit,
        "GUIDE_KIND": GUIDE_KIND,
        "LEARNING_RATE": LEARNING_RATE,
        "OPTIMIZER_KIND": OPTIMIZER_KIND,
        "CLIP_NORM": CLIP_NORM,
        "SVI_STEPS": SVI_STEPS,
        "SVI_CHECK_EVERY": SVI_CHECK_EVERY,
        "POSTERIOR_N_SAMPLES": POSTERIOR_N_SAMPLES,
        "K_MAX": K_MAX,
    },
    "summary_rows": summary_rows,
    "posterior_summary_rows": posterior_summary_rows,
}
with open(bundle_pkl, "wb") as f:
    pickle.dump(bundle, f)

print("\nSaved benchmark outputs:")
print(f"  {summary_csv}")
print(f"  {posterior_summary_csv}")
print(f"  {loss_csv}")
print(f"  {convergence_csv}")
print(f"  {samples_npz}")
print(f"  {bundle_pkl}")


# %%
# =============================================================================
# Diagnostic plots
# =============================================================================
loss_png = OUTPUT_DIR / f"{OUTPUT_PREFIX}_loss_by_condition.png"
fig, ax = plt.subplots(figsize=(8, 4.5))
for _, row in summary_df.iterrows():
    condition_loss = loss_df[
        (loss_df["condition_index"] == row["condition_index"])
        & (np.isfinite(loss_df["loss"]))
    ]
    if len(condition_loss) == 0:
        continue
    label = f"ABL {int(row['ABL'])}, ILD {int(row['ILD']):+d}"
    ax.plot(condition_loss["step"], condition_loss["loss"], lw=1.0, alpha=0.85, label=label)
ax.set_xlabel("SVI step")
ax.set_ylabel("negative ELBO")
ax.set_title(f"{BATCH_NAME}/{ANIMAL} 3-param condition SVI loss")
ax.grid(True, alpha=0.25)
ax.legend(frameon=False, fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(loss_png, dpi=200)

comparison_png = OUTPUT_DIR / f"{OUTPUT_PREFIX}_param_comparison.png"
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharex=True)
x = np.arange(len(summary_df))
condition_labels = [
    f"{int(row.ABL)}/{int(row.ILD):+d}" for row in summary_df.itertuples(index=False)
]
plot_specs = [
    ("gamma", "Gamma", "svi_gamma", "vbmc_gamma", "init_gamma_npl_alpha"),
    ("omega", "Omega", "svi_omega", "vbmc_omega", "init_omega_npl_alpha"),
    ("t_E_aff_ms", "t_E_aff (ms)", "svi_t_E_aff_ms", "vbmc_t_E_aff_ms", "init_t_E_aff_ms"),
]
for ax, (_, title, svi_prefix, vbmc_col, init_col) in zip(axes, plot_specs):
    mean = summary_df[f"{svi_prefix}_mean"].to_numpy(dtype=float)
    q025 = summary_df[f"{svi_prefix}_q025"].to_numpy(dtype=float)
    q975 = summary_df[f"{svi_prefix}_q975"].to_numpy(dtype=float)
    yerr = np.vstack([mean - q025, q975 - mean])
    ax.errorbar(
        x - 0.08,
        mean,
        yerr=yerr,
        fmt="o",
        ms=4,
        capsize=3,
        lw=1.2,
        color="tab:blue",
        label="SVI mean + 95% CI",
    )
    ax.scatter(
        x + 0.08,
        summary_df[vbmc_col].to_numpy(dtype=float),
        marker="x",
        s=42,
        color="tab:red",
        label="VBMC cache mean",
    )
    ax.scatter(
        x,
        summary_df[init_col].to_numpy(dtype=float),
        marker=".",
        s=42,
        color="0.35",
        label="NPL-SVI init",
    )
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(condition_labels, rotation=45, ha="right")
    ax.grid(True, alpha=0.25)
axes[0].set_ylabel("parameter value")
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle(f"{BATCH_NAME}/{ANIMAL}: SVI 3-param condition fits vs VBMC cache", y=1.02)
fig.tight_layout()
fig.savefig(comparison_png, dpi=200, bbox_inches="tight")

corner_pngs = []
for condition_index, row in summary_df.iterrows():
    cond_ABL = int(row["ABL"])
    cond_ILD = int(row["ILD"])
    condition_key = f"cond{condition_index:02d}_ABL{cond_ABL}_ILD{'m' if cond_ILD < 0 else 'p'}{abs(cond_ILD)}"
    posterior_np = posterior_by_condition[condition_key]
    corner_samples = np.column_stack(
        [
            posterior_np["gamma"],
            posterior_np["omega"],
            posterior_np["t_E_aff"] * 1000.0,
        ]
    )
    labels = ["gamma", "omega", "t_E_aff (ms)"]
    clean_samples, ranges = finite_corner_data(corner_samples, labels, condition_key)
    if clean_samples is None:
        continue
    fig = corner.corner(
        clean_samples,
        labels=labels,
        range=ranges,
        show_titles=True,
        title_fmt=".3g",
        quantiles=[0.025, 0.5, 0.975],
    )
    fig.suptitle(
        f"{BATCH_NAME}/{ANIMAL}, ABL={cond_ABL}, ILD={cond_ILD:+d}, "
        f"fixed w={fixed_w:.4f}, del_go={fixed_del_go:.4f}",
        y=1.03,
        fontsize=12,
    )
    corner_png = OUTPUT_DIR / f"{OUTPUT_PREFIX}_corner_ABL{cond_ABL}_ILD{cond_ILD:+d}.png"
    fig.savefig(corner_png, dpi=200, bbox_inches="tight")
    corner_pngs.append(corner_png)

print("\nSaved figures:")
print(f"  {loss_png}")
print(f"  {comparison_png}")
for corner_png in corner_pngs:
    print(f"  {corner_png}")


# %%
# =============================================================================
# Compact console summary
# =============================================================================
print("\nTiming summary:")
print(
    summary_df[
        [
            "ABL",
            "ILD",
            "n_trials",
            "completed_steps",
            "elapsed_s",
            "speedup_vs_vbmc_reference",
            "stop_reason",
        ]
    ].to_string(index=False)
)

print("\nParameter comparison summary:")
print(
    summary_df[
        [
            "ABL",
            "ILD",
            "svi_gamma_mean",
            "vbmc_gamma",
            "svi_omega_mean",
            "vbmc_omega",
            "svi_t_E_aff_ms_mean",
            "vbmc_t_E_aff_ms",
        ]
    ].to_string(index=False)
)
