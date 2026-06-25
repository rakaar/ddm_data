# %%
"""
Fit condition-by-condition gamma/omega with NumPyro SVI.

All non-gamma/omega parameters are fixed from the completed animal-wise
NPL + alpha + condition-delay SVI fits:

    fit_animal_by_animal/numpyro_svi_npl_alpha_condition_delay_single_animal_outputs/

For each observed ABL/ILD in each animal's condition_table.csv, this script
fits only gamma and omega. The condition-specific t_E_aff, global w, and global
del_go are fixed to posterior means from the animal-wise SVI samples.
"""

# %%
# =============================================================================
# Editable parameters
# =============================================================================
from pathlib import Path
import importlib.util
import os
import pickle
import re
import sys
import time

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

ANIMAL_SVI_OUTPUT_ROOT = (
    REPO_DIR
    / "fit_animal_by_animal"
    / "numpyro_svi_npl_alpha_condition_delay_single_animal_outputs"
)
OUTPUT_ROOT = SCRIPT_DIR / "svi_gamma_omega_fixed_from_animal_svi_condition_delay_results"

BATCH_ANIMAL_PAIRS_OVERRIDE = os.environ.get("BATCH_ANIMAL_PAIRS_OVERRIDE", "").strip()
CONDITIONS_OVERRIDE = os.environ.get("CONDITIONS_OVERRIDE", "").strip()
RUN_LABEL = os.environ.get(
    "NUMPYRO_COND_GAMMA_OMEGA_RUN_LABEL",
    "custom_conditions" if CONDITIONS_OVERRIDE else "all_observed",
)

GUIDE_KIND = os.environ.get("NUMPYRO_COND_GAMMA_OMEGA_GUIDE", "fullrank")
LEARNING_RATE = float(os.environ.get("NUMPYRO_COND_GAMMA_OMEGA_LR", "0.001"))
OPTIMIZER_KIND = os.environ.get("NUMPYRO_COND_GAMMA_OMEGA_OPTIMIZER", "clipped_adam")
CLIP_NORM = float(os.environ.get("NUMPYRO_COND_GAMMA_OMEGA_CLIP_NORM", "5.0"))
GUIDE_INIT_SCALE = float(os.environ.get("NUMPYRO_COND_GAMMA_OMEGA_GUIDE_INIT_SCALE", "0.05"))
SVI_STEPS = int(os.environ.get("NUMPYRO_COND_GAMMA_OMEGA_STEPS", "12000"))
SVI_CHECK_EVERY = int(os.environ.get("NUMPYRO_COND_GAMMA_OMEGA_CHECK_EVERY", "1000"))
SVI_EARLY_STOP = os.environ.get("NUMPYRO_COND_GAMMA_OMEGA_EARLY_STOP", "1").strip().lower() in {
    "1",
    "true",
    "yes",
}
SVI_REL_TOL = float(os.environ.get("NUMPYRO_COND_GAMMA_OMEGA_REL_TOL", "0.001"))
SVI_PATIENCE_WINDOWS = int(os.environ.get("NUMPYRO_COND_GAMMA_OMEGA_PATIENCE_WINDOWS", "3"))
SVI_MIN_IMPROVEMENT_REL = float(os.environ.get("NUMPYRO_COND_GAMMA_OMEGA_MIN_IMPROVEMENT_REL", "0.001"))
SVI_NO_IMPROVE_PATIENCE_WINDOWS = int(
    os.environ.get("NUMPYRO_COND_GAMMA_OMEGA_NO_IMPROVE_PATIENCE_WINDOWS", "5")
)
SVI_STABLE_UPDATE = os.environ.get("NUMPYRO_COND_GAMMA_OMEGA_STABLE_UPDATE", "1").strip().lower() in {
    "1",
    "true",
    "yes",
}
POSTERIOR_N_SAMPLES = int(os.environ.get("NUMPYRO_COND_GAMMA_OMEGA_POSTERIOR_N_SAMPLES", "10000"))
RNG_SEED = int(os.environ.get("NUMPYRO_COND_GAMMA_OMEGA_SEED", "0"))
K_MAX = int(os.environ.get("NUMPYRO_COND_GAMMA_OMEGA_K_MAX", "10"))
MIN_TRIALS_PER_CONDITION = int(os.environ.get("NUMPYRO_COND_GAMMA_OMEGA_MIN_TRIALS", "10"))
OVERWRITE_FITS = os.environ.get("NUMPYRO_COND_GAMMA_OMEGA_OVERWRITE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}
MAX_ANIMALS_ENV = os.environ.get("NUMPYRO_COND_GAMMA_OMEGA_MAX_ANIMALS", "").strip()
MAX_ANIMALS = int(MAX_ANIMALS_ENV) if MAX_ANIMALS_ENV else None

BATCH_T_TRUNC = {"LED34_even": 0.15}
DEFAULT_T_TRUNC = 0.3
DEFAULT_BATCH_ORDER = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]
GLOBAL_SAMPLE_KEYS = ["rate_lambda", "T_0", "theta_E", "w", "del_go", "rate_norm_l", "alpha"]

OUTPUT_DIR = OUTPUT_ROOT / RUN_LABEL
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# %%
# =============================================================================
# Dependency preflight
# =============================================================================
required_modules = ["jax", "jaxlib", "numpyro"]
missing_modules = [module for module in required_modules if importlib.util.find_spec(module) is None]
if missing_modules:
    print("Missing dependencies for NumPyro condition gamma/omega SVI:")
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
# Small helpers
# =============================================================================
def parse_batch_animal_pairs(text):
    pairs = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        batch_text, animal_text = chunk.replace("/", ":").split(":")
        pairs.append((batch_text.strip(), int(animal_text)))
    return pairs


def parse_conditions(text):
    conditions = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        abl_text, ild_text = chunk.replace("/", ":").split(":")
        conditions.append((int(float(abl_text)), int(float(ild_text))))
    return conditions


def discover_animal_pairs():
    pattern = re.compile(r"^(?P<batch>.+)_(?P<animal>\d+)$")
    pairs = []
    for npz_path in ANIMAL_SVI_OUTPUT_ROOT.glob("*/main_fullrank_posterior_samples.npz"):
        match = pattern.match(npz_path.parent.name)
        if match is None:
            continue
        pairs.append((match.group("batch"), int(match.group("animal"))))
    batch_order = {batch: idx for idx, batch in enumerate(DEFAULT_BATCH_ORDER)}
    return sorted(pairs, key=lambda pair: (batch_order.get(pair[0], 999), pair[0], pair[1]))


def finite_tree(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return bool(all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves))


def make_optimizer():
    optimizer_kind = OPTIMIZER_KIND.strip().lower()
    if optimizer_kind in {"adam", "plain_adam"}:
        return numpyro.optim.Adam(LEARNING_RATE)
    if optimizer_kind in {"clipped_adam", "clipped-adam", "clip_adam"}:
        return numpyro.optim.ClippedAdam(LEARNING_RATE, clip_norm=CLIP_NORM)
    raise ValueError(f"Unknown NUMPYRO_COND_GAMMA_OMEGA_OPTIMIZER={OPTIMIZER_KIND!r}")


def make_guide(model, init_values):
    init_loc_fn = init_to_value(values=init_values)
    guide_kind = GUIDE_KIND.strip().lower()
    if guide_kind in {"meanfield", "autonormal", "normal"}:
        return AutoNormal(model, init_loc_fn=init_loc_fn)
    if guide_kind in {"fullrank", "multivariate", "automultivariate"}:
        return AutoMultivariateNormal(model, init_loc_fn=init_loc_fn, init_scale=GUIDE_INIT_SCALE)
    raise ValueError(f"Use fullrank or meanfield guide for this two-param fit, got {GUIDE_KIND!r}.")


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


def safe_condition_key(condition_index, abl, ild):
    sign = "m" if ild < 0 else "p"
    return f"cond{condition_index:02d}_ABL{int(abl)}_ILD{sign}{abs(int(ild))}"


def condition_output_paths(batch_name, animal_id):
    animal_prefix = f"{batch_name}_{animal_id}_gamma_omega_fixed_from_animal_svi"
    return {
        "summary_csv": OUTPUT_DIR / f"{animal_prefix}_summary.csv",
        "posterior_summary_csv": OUTPUT_DIR / f"{animal_prefix}_posterior_summary.csv",
        "loss_csv": OUTPUT_DIR / f"{animal_prefix}_loss_by_condition.csv",
        "convergence_csv": OUTPUT_DIR / f"{animal_prefix}_convergence_checks.csv",
        "samples_npz": OUTPUT_DIR / f"{animal_prefix}_posterior_samples.npz",
        "bundle_pkl": OUTPUT_DIR / f"{animal_prefix}_fit_bundle.pkl",
    }


def animal_complete(paths):
    required = [
        paths["summary_csv"],
        paths["posterior_summary_csv"],
        paths["loss_csv"],
        paths["convergence_csv"],
        paths["samples_npz"],
        paths["bundle_pkl"],
    ]
    return all(path.exists() and path.stat().st_size > 0 for path in required)


# %%
# =============================================================================
# JAX likelihood pieces
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
        data["t_E_aff"],
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
            data["t_E_aff"],
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
            data["t_E_aff"],
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
    target_logpdf = trapezoidal_logpdf_jax(value, hard_low, plausible_low, plausible_high, hard_high)
    uniform_logpdf = -jnp.log(hard_high - hard_low)
    numpyro.factor(f"{name}_trapezoid_prior", target_logpdf - uniform_logpdf)
    return value


def condition_gamma_omega_model(data):
    params = {
        "gamma": sample_trapezoid(
            "gamma",
            data["gamma_hard_low"],
            data["gamma_hard_high"],
            data["gamma_plausible_low"],
            data["gamma_plausible_high"],
        ),
        "omega": sample_trapezoid("omega", 0.1, 15.0, 2.0, 12.0),
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

    return {
        "total_fix": jnp.asarray(total_fix),
        "t_stim": jnp.asarray(t_stim),
        "choice": jnp.asarray(choice),
        "mask": jnp.asarray(mask),
        "V_A": jnp.asarray(fixed_values["V_A"], dtype=jnp.float64),
        "theta_A": jnp.asarray(fixed_values["theta_A"], dtype=jnp.float64),
        "t_A_aff": jnp.asarray(fixed_values["t_A_aff"], dtype=jnp.float64),
        "w": jnp.asarray(fixed_values["w"], dtype=jnp.float64),
        "del_go": jnp.asarray(fixed_values["del_go"], dtype=jnp.float64),
        "t_E_aff": jnp.asarray(fixed_values["t_E_aff"], dtype=jnp.float64),
        "T_trunc": jnp.asarray(fixed_values["T_trunc"], dtype=jnp.float64),
        "gamma_hard_low": jnp.asarray(bounds["gamma_hard_low"], dtype=jnp.float64),
        "gamma_hard_high": jnp.asarray(bounds["gamma_hard_high"], dtype=jnp.float64),
        "gamma_plausible_low": jnp.asarray(bounds["gamma_plausible_low"], dtype=jnp.float64),
        "gamma_plausible_high": jnp.asarray(bounds["gamma_plausible_high"], dtype=jnp.float64),
    }


# %%
# =============================================================================
# Load animal list
# =============================================================================
if BATCH_ANIMAL_PAIRS_OVERRIDE:
    batch_animal_pairs = parse_batch_animal_pairs(BATCH_ANIMAL_PAIRS_OVERRIDE)
else:
    batch_animal_pairs = discover_animal_pairs()

if MAX_ANIMALS is not None:
    batch_animal_pairs = batch_animal_pairs[:MAX_ANIMALS]

if CONDITIONS_OVERRIDE:
    override_conditions = parse_conditions(CONDITIONS_OVERRIDE)
else:
    override_conditions = None

print(f"Run label: {RUN_LABEL}")
print(f"Animal SVI output root: {ANIMAL_SVI_OUTPUT_ROOT}")
print(f"Output dir: {OUTPUT_DIR}")
print(f"Found {len(batch_animal_pairs)} animals to process")
print(f"JAX default backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")
print(
    f"SVI config: guide={GUIDE_KIND}, lr={LEARNING_RATE}, steps={SVI_STEPS}, "
    f"check_every={SVI_CHECK_EVERY}, posterior_samples={POSTERIOR_N_SAMPLES}"
)
if override_conditions is not None:
    print(f"Using condition override: {override_conditions}")


# %%
# =============================================================================
# Main all-animal fitting loop
# =============================================================================
all_summary_rows = []
all_posterior_summary_rows = []
all_condition_cache_rows = []
all_failures = []

for animal_index, (batch_name, animal_id) in enumerate(batch_animal_pairs):
    print("\n" + "=" * 80)
    print(f"Animal {animal_index + 1}/{len(batch_animal_pairs)}: {batch_name}/{animal_id}")
    print("=" * 80)

    paths = condition_output_paths(batch_name, animal_id)
    if animal_complete(paths) and not OVERWRITE_FITS:
        print("Existing per-animal outputs complete; loading summary and skipping.")
        animal_summary_df = pd.read_csv(paths["summary_csv"])
        animal_posterior_summary_df = pd.read_csv(paths["posterior_summary_csv"])
        all_summary_rows.extend(animal_summary_df.to_dict("records"))
        all_posterior_summary_rows.extend(animal_posterior_summary_df.to_dict("records"))
        for _, row in animal_summary_df.iterrows():
            all_condition_cache_rows.append(
                {
                    "batch_name": batch_name,
                    "animal": animal_id,
                    "ABL": int(row["ABL"]),
                    "ILD": int(row["ILD"]),
                    "condition_gamma": float(row["svi_gamma_mean"]),
                    "condition_omega": float(row["svi_omega_mean"]),
                    "fixed_t_E_aff_s": float(row["fixed_t_E_aff_s"]),
                    "fixed_t_E_aff_ms": float(row["fixed_t_E_aff_ms"]),
                    "source_npz": str(paths["samples_npz"]),
                    "source_script": str(Path(__file__).resolve()),
                }
            )
        continue

    animal_output_dir = ANIMAL_SVI_OUTPUT_ROOT / f"{batch_name}_{animal_id}"
    animal_svi_posterior_npz = animal_output_dir / "main_fullrank_posterior_samples.npz"
    animal_svi_condition_table_csv = animal_output_dir / "condition_table.csv"
    batch_csv = REPO_DIR / "raw_data" / "batch_csvs" / f"batch_{batch_name}_valid_and_aborts.csv"
    abort_result_pkl = REPO_DIR / "aborts_ipl_npl_time_fit_results" / f"results_{batch_name}_animal_{animal_id}.pkl"

    for required_path in [animal_svi_posterior_npz, animal_svi_condition_table_csv, batch_csv, abort_result_pkl]:
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    raw_df = pd.read_csv(batch_csv)
    if "choice" not in raw_df.columns:
        if "response_poke" not in raw_df.columns:
            raise KeyError("Need either `choice` or `response_poke` in the batch CSV.")
        raw_df["choice"] = raw_df["response_poke"].map({3: 1, 2: -1})

    total_fix_col = "timed_fix" if "timed_fix" in raw_df.columns else "TotalFixTime"
    valid_df = raw_df[
        (raw_df["animal"].astype(int) == int(animal_id))
        & (raw_df["success"].isin([1, -1]))
        & (raw_df["RTwrtStim"] > 0)
        & (raw_df["RTwrtStim"] <= 1)
    ].copy()
    valid_df = valid_df.dropna(subset=[total_fix_col, "intended_fix", "ABL", "ILD", "choice", "RTwrtStim"])
    valid_df["ABL"] = valid_df["ABL"].astype(int)
    valid_df["ILD"] = valid_df["ILD"].astype(int)
    valid_df["choice"] = valid_df["choice"].astype(int)

    with open(abort_result_pkl, "rb") as f:
        abort_result = pickle.load(f)
    abort_samples = abort_result["vbmc_aborts_results"]
    V_A = float(np.mean(abort_samples["V_A_samples"]))
    theta_A = float(np.mean(abort_samples["theta_A_samples"]))
    t_A_aff = float(np.mean(abort_samples["t_A_aff_samp"]))

    animal_svi_samples = np.load(animal_svi_posterior_npz)
    animal_svi_condition_table = pd.read_csv(animal_svi_condition_table_csv)
    animal_svi_condition_table["ABL"] = animal_svi_condition_table["ABL"].astype(int)
    animal_svi_condition_table["ILD"] = animal_svi_condition_table["ILD"].astype(int)
    animal_svi_means = {
        key: float(np.mean(animal_svi_samples[key]))
        for key in animal_svi_samples.files
        if key != "t_E_aff"
    }

    if override_conditions is None:
        conditions_to_fit = [
            (int(row.ABL), int(row.ILD))
            for row in animal_svi_condition_table.sort_values("condition_id").itertuples(index=False)
        ]
    else:
        available_conditions = {
            (int(row.ABL), int(row.ILD))
            for row in animal_svi_condition_table.itertuples(index=False)
        }
        conditions_to_fit = [condition for condition in override_conditions if condition in available_conditions]
        missing_override = [condition for condition in override_conditions if condition not in available_conditions]
        if missing_override:
            print(f"  Skipping override conditions absent from this animal: {missing_override}")

    condition_frames = []
    for cond_ABL, cond_ILD in conditions_to_fit:
        df_cond = valid_df[(valid_df["ABL"] == cond_ABL) & (valid_df["ILD"] == cond_ILD)].copy()
        if len(df_cond) < MIN_TRIALS_PER_CONDITION:
            print(
                f"  [{cond_ABL}, {cond_ILD:+d}] only {len(df_cond)} trials after RT filter; skipping."
            )
            continue
        condition_frames.append({"ABL": cond_ABL, "ILD": cond_ILD, "df": df_cond})

    if len(condition_frames) == 0:
        raise RuntimeError(f"No conditions to fit for {batch_name}/{animal_id}.")

    max_n_trials = max(len(item["df"]) for item in condition_frames)
    print(
        f"  Fixed from animal SVI: w={animal_svi_means['w']:.6f}, "
        f"del_go={animal_svi_means['del_go']:.6f}; "
        f"{len(condition_frames)} conditions, padded_n={max_n_trials}"
    )

    fixed_common = {
        "V_A": V_A,
        "theta_A": theta_A,
        "t_A_aff": t_A_aff,
        "w": animal_svi_means["w"],
        "del_go": animal_svi_means["del_go"],
        "T_trunc": BATCH_T_TRUNC.get(batch_name, DEFAULT_T_TRUNC),
        "total_fix_col": total_fix_col,
    }

    animal_summary_rows = []
    animal_posterior_summary_rows = []
    animal_loss_rows = []
    animal_convergence_tables = []
    posterior_npz_payload = {}

    for condition_index, condition_item in enumerate(condition_frames):
        cond_ABL = condition_item["ABL"]
        cond_ILD = condition_item["ILD"]
        df_cond = condition_item["df"]
        condition_label = f"{batch_name}/{animal_id} ABL{cond_ABL}_ILD{cond_ILD:+d}"
        condition_key = safe_condition_key(condition_index, cond_ABL, cond_ILD)

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
            (animal_svi_condition_table["ABL"] == cond_ABL)
            & (animal_svi_condition_table["ILD"] == cond_ILD)
        ]
        if len(condition_table_match) != 1:
            raise RuntimeError(f"Could not find exactly one animal-SVI condition row for {condition_label}.")
        condition_id = int(condition_table_match.iloc[0]["condition_id"])
        fixed_t_E_aff_samples = np.asarray(animal_svi_samples["t_E_aff"][:, condition_id], dtype=float)
        fixed_t_E_aff_summary = summarize_samples(fixed_t_E_aff_samples)
        fixed_t_E_aff_s = fixed_t_E_aff_summary["mean"]

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
        init_values = {"gamma": init_gamma, "omega": init_omega}

        fixed_values = dict(fixed_common)
        fixed_values["t_E_aff"] = fixed_t_E_aff_s
        data = make_padded_condition_data(df_cond, max_n_trials, fixed_values, gamma_bounds)

        def log_joint_from_values(values):
            log_joint, _ = log_density(condition_gamma_omega_model, (data,), {}, values)
            return log_joint

        initial_log_joint = float(log_joint_from_values(init_values))
        initial_grad = jax.grad(log_joint_from_values)(init_values)
        initial_grad_finite = finite_tree(initial_grad)

        print(
            f"\n  {condition_label}: n={len(df_cond)}, "
            f"fixed t_E_aff={fixed_t_E_aff_s * 1000.0:.3f} ms, "
            f"init gamma={init_gamma:.4g}, omega={init_omega:.4g}"
        )
        print(f"    initial log joint={initial_log_joint:.6f}, grad finite={initial_grad_finite}")
        if (not np.isfinite(initial_log_joint)) or (not initial_grad_finite):
            raise RuntimeError(f"Initial log joint or gradient is non-finite for {condition_label}.")

        guide = make_guide(condition_gamma_omega_model, init_values)
        svi = SVI(condition_gamma_omega_model, guide, make_optimizer(), Trace_ELBO())

        condition_start = time.perf_counter()
        rng_offset = animal_index * 1000 + condition_index
        svi_result, convergence_df, stop_reason = run_svi_with_convergence_checks(
            svi,
            random.PRNGKey(RNG_SEED + rng_offset),
            SVI_STEPS,
            data,
            condition_label,
        )
        elapsed_s = time.perf_counter() - condition_start

        posterior_samples = guide.sample_posterior(
            random.PRNGKey(RNG_SEED + 100000 + rng_offset),
            svi_result.params,
            sample_shape=(POSTERIOR_N_SAMPLES,),
        )
        posterior_np = {key: np.asarray(value) for key, value in posterior_samples.items()}
        posterior_all_finite = bool(all(np.all(np.isfinite(value)) for value in posterior_np.values()))
        for param_name, values in posterior_np.items():
            posterior_npz_payload[f"{condition_key}_{param_name}"] = values

        gamma_summary = summarize_samples(posterior_np["gamma"])
        omega_summary = summarize_samples(posterior_np["omega"])
        losses_np = np.asarray(jax.device_get(svi_result.losses), dtype=float)

        for step_idx, loss_value in enumerate(losses_np, start=1):
            animal_loss_rows.append(
                {
                    "batch_name": batch_name,
                    "animal": animal_id,
                    "condition_index": condition_index,
                    "condition_id_from_animal_svi": condition_id,
                    "ABL": cond_ABL,
                    "ILD": cond_ILD,
                    "step": step_idx,
                    "loss": float(loss_value),
                }
            )

        convergence_df = convergence_df.copy()
        convergence_df.insert(0, "batch_name", batch_name)
        convergence_df.insert(1, "animal", animal_id)
        convergence_df.insert(2, "condition_index", condition_index)
        convergence_df.insert(3, "condition_id_from_animal_svi", condition_id)
        convergence_df.insert(4, "ABL", cond_ABL)
        convergence_df.insert(5, "ILD", cond_ILD)
        animal_convergence_tables.append(convergence_df)

        row = {
            "batch_name": batch_name,
            "animal": animal_id,
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
            "loss_first": float(losses_np[0]) if len(losses_np) else np.nan,
            "loss_last": float(losses_np[-1]) if len(losses_np) else np.nan,
            "best_window_mean_loss": float(convergence_df["best_mean_loss_so_far"].iloc[-1])
            if len(convergence_df)
            else np.nan,
            "initial_log_joint": initial_log_joint,
            "initial_grad_finite": initial_grad_finite,
            "posterior_all_finite": posterior_all_finite,
            "fixed_w": animal_svi_means["w"],
            "fixed_del_go": animal_svi_means["del_go"],
            "fixed_t_E_aff_s": fixed_t_E_aff_s,
            "fixed_t_E_aff_ms": fixed_t_E_aff_s * 1000.0,
            "fixed_t_E_aff_s_q025": fixed_t_E_aff_summary["q025"],
            "fixed_t_E_aff_s_q975": fixed_t_E_aff_summary["q975"],
            "V_A": V_A,
            "theta_A": theta_A,
            "t_A_aff": t_A_aff,
            "T_trunc": fixed_common["T_trunc"],
            "init_gamma_npl_alpha": init_gamma,
            "init_omega_npl_alpha": init_omega,
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
            "animal_svi_posterior_npz": str(animal_svi_posterior_npz),
            "source_script": str(Path(__file__).resolve()),
        }
        for param_name in GLOBAL_SAMPLE_KEYS:
            row[f"fixed_{param_name}"] = animal_svi_means[param_name]

        animal_summary_rows.append(row)
        all_summary_rows.append(row)

        for param_name, summary in [("gamma", gamma_summary), ("omega", omega_summary)]:
            posterior_row = {
                "batch_name": batch_name,
                "animal": animal_id,
                "condition_index": condition_index,
                "condition_id_from_animal_svi": condition_id,
                "ABL": cond_ABL,
                "ILD": cond_ILD,
                "parameter": param_name,
                **summary,
                "n_samples": POSTERIOR_N_SAMPLES,
            }
            animal_posterior_summary_rows.append(posterior_row)
            all_posterior_summary_rows.append(posterior_row)

        cache_row = {
            "batch_name": batch_name,
            "animal": animal_id,
            "ABL": cond_ABL,
            "ILD": cond_ILD,
            "condition_gamma": gamma_summary["mean"],
            "condition_omega": omega_summary["mean"],
            "fixed_t_E_aff_s": fixed_t_E_aff_s,
            "fixed_t_E_aff_ms": fixed_t_E_aff_s * 1000.0,
            "source_npz": str(paths["samples_npz"]),
            "source_script": str(Path(__file__).resolve()),
        }
        all_condition_cache_rows.append(cache_row)

        print(
            f"    done in {elapsed_s:.2f} s: gamma={gamma_summary['mean']:.4g} "
            f"[{gamma_summary['q025']:.4g}, {gamma_summary['q975']:.4g}], "
            f"omega={omega_summary['mean']:.4g} "
            f"[{omega_summary['q025']:.4g}, {omega_summary['q975']:.4g}]"
        )

    animal_summary_df = pd.DataFrame(animal_summary_rows)
    animal_posterior_summary_df = pd.DataFrame(animal_posterior_summary_rows)
    animal_loss_df = pd.DataFrame(animal_loss_rows)
    animal_convergence_df = (
        pd.concat(animal_convergence_tables, ignore_index=True)
        if animal_convergence_tables
        else pd.DataFrame()
    )

    animal_summary_df.to_csv(paths["summary_csv"], index=False)
    animal_posterior_summary_df.to_csv(paths["posterior_summary_csv"], index=False)
    animal_loss_df.to_csv(paths["loss_csv"], index=False)
    animal_convergence_df.to_csv(paths["convergence_csv"], index=False)
    np.savez(paths["samples_npz"], **posterior_npz_payload)

    bundle = {
        "batch_name": batch_name,
        "animal": animal_id,
        "run_label": RUN_LABEL,
        "config": {
            "GUIDE_KIND": GUIDE_KIND,
            "LEARNING_RATE": LEARNING_RATE,
            "OPTIMIZER_KIND": OPTIMIZER_KIND,
            "CLIP_NORM": CLIP_NORM,
            "SVI_STEPS": SVI_STEPS,
            "SVI_CHECK_EVERY": SVI_CHECK_EVERY,
            "POSTERIOR_N_SAMPLES": POSTERIOR_N_SAMPLES,
            "K_MAX": K_MAX,
            "conditions_override": override_conditions,
        },
        "condition_table": animal_svi_condition_table.to_dict("records"),
        "summary_rows": animal_summary_rows,
        "posterior_summary_rows": animal_posterior_summary_rows,
        "fixed_global_means": animal_svi_means,
        "abort_means": {"V_A": V_A, "theta_A": theta_A, "t_A_aff": t_A_aff},
        "animal_svi_posterior_npz": str(animal_svi_posterior_npz),
    }
    with open(paths["bundle_pkl"], "wb") as f:
        pickle.dump(bundle, f)

    print("  Saved per-animal outputs:")
    for path in paths.values():
        print(f"    {path}")


# %%
# =============================================================================
# Save aggregate outputs
# =============================================================================
aggregate_summary_csv = OUTPUT_DIR / "all_animals_gamma_omega_fixed_from_animal_svi_summary.csv"
aggregate_posterior_summary_csv = OUTPUT_DIR / "all_animals_gamma_omega_fixed_from_animal_svi_posterior_summary.csv"
condition_cache_csv = OUTPUT_DIR / "condition_gamma_omega_extraction_cache.csv"
failure_csv = OUTPUT_DIR / "failures.csv"

pd.DataFrame(all_summary_rows).to_csv(aggregate_summary_csv, index=False)
pd.DataFrame(all_posterior_summary_rows).to_csv(aggregate_posterior_summary_csv, index=False)
pd.DataFrame(all_condition_cache_rows).to_csv(condition_cache_csv, index=False)
pd.DataFrame(all_failures).to_csv(failure_csv, index=False)

print("\nAggregate outputs:")
print(f"  {aggregate_summary_csv}")
print(f"  {aggregate_posterior_summary_csv}")
print(f"  {condition_cache_csv}")
print(f"  {failure_csv}")

if all_summary_rows:
    summary_df = pd.DataFrame(all_summary_rows)
    print("\nFit count summary:")
    print(
        summary_df.groupby(["batch_name", "animal"]).size().rename("n_conditions").reset_index().to_string(index=False)
    )
