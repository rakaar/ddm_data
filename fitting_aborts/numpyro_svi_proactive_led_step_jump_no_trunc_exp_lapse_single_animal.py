# %%
"""Fit one LED7 animal with the FCT-matched no-truncation lapse model."""

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

ANIMAL = int(os.environ.get("PROACTIVE_LED_LAPSE_SVI_ANIMAL", "93"))
QUADRATURE_NODES = int(
    os.environ.get("PROACTIVE_LED_LAPSE_SVI_QUADRATURE_NODES", "64")
)

MAIN_STEPS = int(os.environ.get("PROACTIVE_LED_LAPSE_SVI_MAIN_STEPS", "150000"))
EXTENDED_MAX_STEPS = int(
    os.environ.get("PROACTIVE_LED_LAPSE_SVI_EXTENDED_MAX_STEPS", "250000")
)
SVI_CHECK_EVERY = int(
    os.environ.get("PROACTIVE_LED_LAPSE_SVI_CHECK_EVERY", "1000")
)
SVI_MIN_STEPS = int(
    os.environ.get("PROACTIVE_LED_LAPSE_SVI_MIN_STEPS", "50000")
)
SVI_MIN_IMPROVEMENT_REL = float(
    os.environ.get("PROACTIVE_LED_LAPSE_SVI_MIN_IMPROVEMENT_REL", "0.001")
)
SVI_NO_IMPROVE_PATIENCE_WINDOWS = int(
    os.environ.get("PROACTIVE_LED_LAPSE_SVI_PATIENCE_WINDOWS", "12")
)
SVI_REL_TOL = float(
    os.environ.get("PROACTIVE_LED_LAPSE_SVI_REL_TOL", "0.001")
)
LEARNING_RATE = float(os.environ.get("PROACTIVE_LED_LAPSE_SVI_LR", "0.0002"))
CLIP_NORM = float(os.environ.get("PROACTIVE_LED_LAPSE_SVI_CLIP_NORM", "1.0"))
GUIDE_INIT_SCALE = float(
    os.environ.get("PROACTIVE_LED_LAPSE_SVI_GUIDE_INIT_SCALE", "0.1")
)
POSTERIOR_N_SAMPLES = int(
    os.environ.get("PROACTIVE_LED_LAPSE_SVI_POSTERIOR_SAMPLES", "10000")
)
RNG_SEED = int(os.environ.get("PROACTIVE_LED_LAPSE_SVI_SEED", "0"))

DATA_CSV = REPO_DIR / "out_LED.csv"
OUTPUT_ROOT = Path(
    os.environ.get(
        "PROACTIVE_LED_LAPSE_SVI_OUTPUT_ROOT",
        str(
            SCRIPT_DIR
            / "numpyro_svi_proactive_led_step_jump_all_on_no_trunc_exp_lapse_"
            "patience12_min50k_restore_best_outputs"
        ),
    )
).expanduser()
OUTPUT_DIR = OUTPUT_ROOT / f"LED7_{ANIMAL}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_TOTAL_TRIALS = {
    92: 21608,
    93: 16486,
    98: 14055,
    99: 14570,
    100: 8727,
    103: 18987,
}
EXPECTED_LED_ON_TRIALS = {
    92: 7064,
    93: 5595,
    98: 4553,
    99: 4814,
    100: 2823,
    103: 6061,
}
EXPECTED_EARLY_ABORTS = {
    92: 543,
    93: 664,
    98: 453,
    99: 787,
    100: 148,
    103: 486,
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

sys.path.insert(0, str(SCRIPT_DIR))
import numpyro_proactive_led_step_jump_no_trunc_exp_lapse_svi_utils as svi_utils


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
    prev_window_mean = np.nan
    stable_window_count = 0
    no_improve_window_count = 0
    completed_steps = 0
    active_max_steps = MAIN_STEPS
    extended_after_150k = False
    stop_reason = "max_steps_restore_best"

    print(
        f"\nRunning full-rank SVI for up to {MAIN_STEPS} steps "
        f"with checks every {SVI_CHECK_EVERY} steps..."
    )
    while True:
        chunk_index = len(convergence_rows) + 1
        chunk_steps = min(SVI_CHECK_EVERY, active_max_steps - completed_steps)
        if chunk_steps <= 0:
            stop_reason = "max_steps_restore_best"
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

        if np.isfinite(prev_window_mean) and np.isfinite(window_mean):
            delta_from_prev = window_mean - prev_window_mean
            relative_change = abs(delta_from_prev) / max(1.0, abs(prev_window_mean))
            is_stable = bool(relative_change <= SVI_REL_TOL)
        else:
            delta_from_prev = np.nan
            relative_change = np.nan
            is_stable = False

        relative_improvement_from_best = np.nan
        improved_best = False
        significant_improvement = False
        if np.isfinite(window_mean):
            if np.isfinite(best_window_mean):
                relative_improvement_from_best = (
                    best_window_mean - window_mean
                ) / max(1.0, abs(best_window_mean))
            improved_best = (
                not np.isfinite(best_window_mean) or window_mean < best_window_mean
            )
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
            no_improve_window_count = (
                0 if significant_improvement else no_improve_window_count + 1
            )
        else:
            no_improve_window_count += 1

        stable_window_count = stable_window_count + 1 if is_stable else 0
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
                "delta_mean_from_prev": delta_from_prev,
                "relative_mean_change": relative_change,
                "relative_improvement_from_best": relative_improvement_from_best,
                "best_mean_loss_so_far": best_window_mean,
                "best_chunk_so_far": best_window_chunk,
                "best_end_step_so_far": best_window_end_step,
                "updated_best_state": bool(improved_best and n_nonfinite == 0),
                "significant_best_improvement": bool(
                    significant_improvement and n_nonfinite == 0
                ),
                "no_improve_window_count": no_improve_window_count,
                "stable_window_count": stable_window_count,
                "slope_per_1000_steps": slope_per_1000,
                "can_stop": can_stop,
                "active_max_steps": active_max_steps,
                "n_nonfinite": n_nonfinite,
            }
        )

        rel_text = (
            "NA"
            if not np.isfinite(relative_change)
            else f"{100.0 * relative_change:.3f}%"
        )
        print(
            f"chunk {chunk_index:03d} steps {start_step}-{end_step}: "
            f"mean={window_mean:.6g}, last={window_last:.6g}, "
            f"rel_change={rel_text}, slope/1k={slope_per_1000:.3g}, "
            f"no_improve={no_improve_window_count}/"
            f"{SVI_NO_IMPROVE_PATIENCE_WINDOWS}, best={best_window_end_step}, "
            f"can_stop={can_stop}, nonfinite={n_nonfinite}, {chunk_seconds:.1f}s"
        )

        if n_nonfinite:
            stop_reason = "nonfinite_loss"
            print(
                "WARNING: non-finite losses detected; restoring the best state "
                f"from step {best_window_end_step}."
            )
            break
        if can_stop and no_improve_window_count >= SVI_NO_IMPROVE_PATIENCE_WINDOWS:
            stop_reason = "patience12_restore_best"
            print(
                f"Stopping at step {completed_steps}: no significant best-window "
                f"improvement for {SVI_NO_IMPROVE_PATIENCE_WINDOWS} windows."
            )
            break

        if completed_steps >= active_max_steps:
            if active_max_steps == MAIN_STEPS and EXTENDED_MAX_STEPS > MAIN_STEPS:
                active_max_steps = EXTENDED_MAX_STEPS
                extended_after_150k = True
                print(
                    f"Patience did not trigger by {MAIN_STEPS} steps; continuing "
                    f"the same SVI state to at most {EXTENDED_MAX_STEPS} steps."
                )
            else:
                stop_reason = "max_steps_restore_best"
                break

        prev_window_mean = window_mean

    losses = np.concatenate(all_losses) if all_losses else np.array([], dtype=float)
    if best_state is None:
        best_state = state
        best_params = svi.get_params(state)
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


def to_numpy_tree(tree):
    return jax.tree_util.tree_map(
        lambda value: np.asarray(jax.device_get(value)),
        tree,
    )


# %%
# =============================================================================
# Reproduce the historical all-ON plus OFF, no-truncation FCT dataset
# =============================================================================
print(f"LED7 animal: {ANIMAL}")
print(f"Data: {DATA_CSV}")
print(f"Output: {OUTPUT_DIR}")
print("LED-ON scope: every LED_trial == 1 row")
print("Truncation: none; early aborts remain in the likelihood")
print("LED-ON weighting: none")
if not DATA_CSV.exists():
    raise FileNotFoundError(DATA_CSV)
if ANIMAL not in EXPECTED_TOTAL_TRIALS:
    raise ValueError(f"Unknown LED7 animal {ANIMAL}.")

raw_df = pd.read_csv(DATA_CSV)
source_df = raw_df[
    raw_df["repeat_trial"].isin([0, 2]) | raw_df["repeat_trial"].isna()
].copy()
source_df = source_df[
    (source_df["session_type"] == 7) & (source_df["training_level"] == 16)
].copy()
source_df = source_df.dropna(
    subset=["intended_fix", "LED_onset_time", "timed_fix"]
)
source_df = source_df[
    (source_df["abort_event"] == 3) | source_df["success"].isin([1, -1])
].copy()
animal_df = source_df[source_df["animal"].astype(int) == ANIMAL].copy()
on_df = animal_df[animal_df["LED_trial"] == 1].copy()
off_df = animal_df[
    (animal_df["LED_trial"] == 0) | animal_df["LED_trial"].isna()
].copy()
if on_df.empty or off_df.empty:
    raise RuntimeError(f"Need LED-ON and LED-OFF rows for LED7/{ANIMAL}.")

trial_df = pd.concat(
    [
        pd.DataFrame(
            {
                "RT": on_df["timed_fix"].to_numpy(dtype=float),
                "t_stim": on_df["intended_fix"].to_numpy(dtype=float),
                "t_LED": (
                    on_df["intended_fix"] - on_df["LED_onset_time"]
                ).to_numpy(dtype=float),
                "is_led": True,
            }
        ),
        pd.DataFrame(
            {
                "RT": off_df["timed_fix"].to_numpy(dtype=float),
                "t_stim": off_df["intended_fix"].to_numpy(dtype=float),
                "t_LED": np.zeros(len(off_df), dtype=float),
                "is_led": False,
            }
        ),
    ],
    ignore_index=True,
)

early_abort_count = int(
    np.sum(
        (animal_df["abort_event"] == 3)
        & (animal_df["timed_fix"] < 0.3)
        & (
            (animal_df["LED_trial"] == 1)
            | (animal_df["LED_trial"] == 0)
            | animal_df["LED_trial"].isna()
        )
    )
)
if len(trial_df) != EXPECTED_TOTAL_TRIALS[ANIMAL]:
    raise RuntimeError(
        f"LED7/{ANIMAL} expected {EXPECTED_TOTAL_TRIALS[ANIMAL]} rows, "
        f"found {len(trial_df)}."
    )
if int(trial_df["is_led"].sum()) != EXPECTED_LED_ON_TRIALS[ANIMAL]:
    raise RuntimeError(
        f"LED7/{ANIMAL} expected {EXPECTED_LED_ON_TRIALS[ANIMAL]} ON rows, "
        f"found {int(trial_df['is_led'].sum())}."
    )
if early_abort_count != EXPECTED_EARLY_ABORTS[ANIMAL]:
    raise RuntimeError(
        f"LED7/{ANIMAL} expected {EXPECTED_EARLY_ABORTS[ANIMAL]} early aborts, "
        f"found {early_abort_count}."
    )

on_abort_df = trial_df[trial_df["is_led"] & (trial_df["RT"] < trial_df["t_stim"])]
on_censored_df = trial_df[
    trial_df["is_led"] & (trial_df["RT"] >= trial_df["t_stim"])
]
off_abort_df = trial_df[
    ~trial_df["is_led"] & (trial_df["RT"] < trial_df["t_stim"])
]
off_censored_df = trial_df[
    ~trial_df["is_led"] & (trial_df["RT"] >= trial_df["t_stim"])
]

jax_data = {
    "on_abort_t": jnp.asarray(on_abort_df["RT"].to_numpy(dtype=float)),
    "on_abort_t_led": jnp.asarray(on_abort_df["t_LED"].to_numpy(dtype=float)),
    "on_censored_t_stim": jnp.asarray(
        on_censored_df["t_stim"].to_numpy(dtype=float)
    ),
    "on_censored_t_led": jnp.asarray(
        on_censored_df["t_LED"].to_numpy(dtype=float)
    ),
    "off_abort_t": jnp.asarray(off_abort_df["RT"].to_numpy(dtype=float)),
    "off_censored_t_stim": jnp.asarray(
        off_censored_df["t_stim"].to_numpy(dtype=float)
    ),
}
if "T_trunc" in jax_data:
    raise RuntimeError("No-truncation fit data unexpectedly contain T_trunc.")

trial_counts = {
    "total": int(len(trial_df)),
    "led_on_total": int(trial_df["is_led"].sum()),
    "led_off_total": int((~trial_df["is_led"]).sum()),
    "led_on_abort": int(len(on_abort_df)),
    "led_on_censored": int(len(on_censored_df)),
    "led_off_abort": int(len(off_abort_df)),
    "led_off_censored": int(len(off_censored_df)),
    "early_aborts_below_300ms_retained": early_abort_count,
}
print("\nTrial counts:")
for name, count in trial_counts.items():
    print(f"  {name:<38} {count}")


# %%
# =============================================================================
# Initial joint density and gradients
# =============================================================================
init_values = svi_utils.clip_init_to_hard_bounds(dict(svi_utils.DEFAULT_INIT_VALUES))
model = lambda data: svi_utils.proactive_led_step_jump_no_trunc_lapse_model(
    data,
    n_quad=QUADRATURE_NODES,
)


def log_joint_from_values(values):
    log_joint, _ = log_density(model, (jax_data,), {}, values)
    return log_joint


print("\nInitial values:")
for name in svi_utils.PARAM_NAMES:
    print(f"  {name:<24} {init_values[name]:.8g}")
print(f"Quadrature nodes: {QUADRATURE_NODES}")
print(f"JAX backend: {jax.default_backend()}")
print(
    f"SVI: initial max={MAIN_STEPS}, extended max={EXTENDED_MAX_STEPS}, "
    f"min={SVI_MIN_STEPS}, check={SVI_CHECK_EVERY}, "
    f"patience={SVI_NO_IMPROVE_PATIENCE_WINDOWS}, "
    f"min_improvement={100.0 * SVI_MIN_IMPROVEMENT_REL:.3g}%, "
    f"lr={LEARNING_RATE:g}"
)

initial_log_joint = log_joint_from_values(init_values)
initial_grad = jax.grad(log_joint_from_values)(init_values)
print(f"Initial log joint: {float(initial_log_joint):.8g}")
print(f"Initial gradients finite: {svi_utils.tree_all_finite(initial_grad)}")
if not np.isfinite(float(initial_log_joint)) or not svi_utils.tree_all_finite(
    initial_grad
):
    raise RuntimeError("Initial log joint or gradients are non-finite.")


# %%
# =============================================================================
# Full-rank SVI with patience-12 restore-best stopping
# =============================================================================
fit_start = perf_counter()
guide = svi_utils.make_fullrank_guide(
    model,
    init_values,
    init_scale=GUIDE_INIT_SCALE,
)
svi = SVI(model, guide, make_optimizer(), Trace_ELBO())
fit_result, convergence_df, stop_reason, extended_after_150k = (
    run_svi_with_convergence_checks(
        svi,
        random.PRNGKey(RNG_SEED + ANIMAL),
        jax_data,
    )
)
fit_elapsed_seconds = perf_counter() - fit_start
losses = np.asarray(jax.device_get(fit_result.losses), dtype=float)


# %%
# =============================================================================
# Posterior samples and summaries
# =============================================================================
posterior_samples = guide.sample_posterior(
    random.PRNGKey(RNG_SEED + ANIMAL + 1000),
    fit_result.params,
    sample_shape=(POSTERIOR_N_SAMPLES,),
)
posterior_np = {
    name: np.asarray(jax.device_get(values))
    for name, values in posterior_samples.items()
}
guide_params_np = to_numpy_tree(fit_result.params)

summary_rows = []
all_posterior_finite = True
for name in svi_utils.PARAM_NAMES:
    values = np.asarray(posterior_np[name], dtype=float).reshape(-1)
    finite = np.isfinite(values)
    all_posterior_finite = all_posterior_finite and bool(finite.all())
    finite_values = values[finite]
    bounds = svi_utils.PARAM_BOUNDS[name]
    summary_rows.append(
        {
            "parameter": name,
            "mean": float(np.mean(finite_values)),
            "std": float(np.std(finite_values)),
            "q025": float(np.quantile(finite_values, 0.025)),
            "median": float(np.quantile(finite_values, 0.5)),
            "q975": float(np.quantile(finite_values, 0.975)),
            "n_samples": int(len(values)),
            "n_nonfinite": int(np.sum(~finite)),
            "hard_low": bounds["hard"][0],
            "hard_high": bounds["hard"][1],
            "plausible_low": bounds["plausible"][0],
            "plausible_high": bounds["plausible"][1],
        }
    )
posterior_summary_df = pd.DataFrame(summary_rows)

best_step = int(convergence_df.iloc[-1]["best_end_step_so_far"])
checked_step = int(convergence_df.iloc[-1]["end_step"])
n_nonfinite_losses = int(convergence_df["n_nonfinite"].sum())
if not all_posterior_finite or n_nonfinite_losses:
    status = "failed_validation"
elif stop_reason == "max_steps_restore_best":
    status = "max_steps_restore_best"
else:
    status = "complete"


# %%
# =============================================================================
# Save fit artifacts
# =============================================================================
label = "main_fullrank"
sample_npz = OUTPUT_DIR / f"{label}_posterior_samples.npz"
guide_params_pkl = OUTPUT_DIR / f"{label}_guide_params.pkl"
posterior_summary_csv = OUTPUT_DIR / f"{label}_posterior_summary.csv"
loss_csv = OUTPUT_DIR / f"{label}_loss.csv"
convergence_csv = OUTPUT_DIR / f"{label}_convergence_checks.csv"
loss_png = OUTPUT_DIR / f"{label}_loss.png"
bundle_pkl = OUTPUT_DIR / f"{label}_variational_posterior_bundle.pkl"
run_summary_json = OUTPUT_DIR / "run_summary.json"

np.savez_compressed(sample_npz, **posterior_np)
with guide_params_pkl.open("wb") as handle:
    pickle.dump(guide_params_np, handle)
posterior_summary_df.to_csv(posterior_summary_csv, index=False)
loss_df = pd.DataFrame(
    {"step": np.arange(1, len(losses) + 1), "negative_elbo": losses}
)
loss_df.to_csv(loss_csv, index=False)
convergence_df.to_csv(convergence_csv, index=False)

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
ax.set_title(f"LED7/{ANIMAL} no-truncation exponential-lapse SVI")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(loss_png, dpi=200, bbox_inches="tight")

config = {
    "model_name": "proactive_led_step_jump_all_on_no_trunc_exp_lapse_svi",
    "animal": int(ANIMAL),
    "truncation": "none",
    "quadrature_nodes": QUADRATURE_NODES,
    "main_steps_initial_max": MAIN_STEPS,
    "extended_max_steps": EXTENDED_MAX_STEPS,
    "min_steps": SVI_MIN_STEPS,
    "check_every": SVI_CHECK_EVERY,
    "min_improvement_rel": SVI_MIN_IMPROVEMENT_REL,
    "patience_windows": SVI_NO_IMPROVE_PATIENCE_WINDOWS,
    "relative_stability_tolerance": SVI_REL_TOL,
    "learning_rate": LEARNING_RATE,
    "clip_norm": CLIP_NORM,
    "guide": "AutoMultivariateNormal",
    "guide_init_scale": GUIDE_INIT_SCALE,
    "posterior_n_samples": POSTERIOR_N_SAMPLES,
    "rng_seed": RNG_SEED,
    "led_on_weight": 1.0,
    "led_on_scope": "all LED_trial == 1 rows",
    "lapse_process": "exponential in fixation time",
}
bundle = {
    "schema_version": 1,
    "config": config,
    "trial_counts": trial_counts,
    "init_values": init_values,
    "initial_log_joint": float(initial_log_joint),
    "guide_params": guide_params_np,
    "posterior_samples": posterior_np,
    "posterior_summary": posterior_summary_df,
    "loss_trace": loss_df,
    "convergence_checks": convergence_df,
    "stop_reason": stop_reason,
    "fit_elapsed_seconds": fit_elapsed_seconds,
    "input_path": str(DATA_CSV),
}
with bundle_pkl.open("wb") as handle:
    pickle.dump(bundle, handle)

run_summary = {
    "status": status,
    "stop_reason": stop_reason,
    "animal": int(ANIMAL),
    "fit_elapsed_seconds": fit_elapsed_seconds,
    "fit_elapsed_minutes": fit_elapsed_seconds / 60.0,
    "completed_steps": checked_step,
    "restored_best_step": best_step,
    "extended_after_150k": extended_after_150k,
    "n_nonfinite_losses": n_nonfinite_losses,
    "all_posterior_samples_finite": all_posterior_finite,
    "trial_counts": trial_counts,
    "config": config,
    "artifacts": {
        "posterior_samples_npz": str(sample_npz),
        "guide_params_pkl": str(guide_params_pkl),
        "posterior_summary_csv": str(posterior_summary_csv),
        "loss_csv": str(loss_csv),
        "convergence_csv": str(convergence_csv),
        "loss_png": str(loss_png),
        "variational_posterior_bundle_pkl": str(bundle_pkl),
    },
}
run_summary_json.write_text(json.dumps(run_summary, indent=2) + "\n")

print("\nPosterior summary:")
print(posterior_summary_df.to_string(index=False))
print(f"\nFit elapsed: {fit_elapsed_seconds / 60.0:.2f} minutes")
print(f"Restored best step: {best_step}")
print(f"Final checked step: {checked_step}")
print(f"Stop reason: {stop_reason}")
print(f"Run status: {status}")
print(f"Run summary: {run_summary_json}")
print(f"VP bundle: {bundle_pkl}")
print(f"Loss plot: {loss_png}")

if status == "failed_validation":
    raise SystemExit("SVI fit failed finite-value validation.")
