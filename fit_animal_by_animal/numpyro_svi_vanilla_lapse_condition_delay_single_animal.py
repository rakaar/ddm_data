# %%
"""
NumPyro SVI fit for one vanilla/IPL+lapse animal with condition-wise t_E_aff.

The fitted parameters are:

    rate_lambda, T_0, theta_E, w, del_go, lapse_prob, lapse_prob_right,
    t_E_aff[condition]

Abort/proactive parameters are fixed from the animal-wise abort fit. The
likelihood is the JAX port of the old vanilla+lapse branch in
`lapses_fit_single_animal.py`.
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

BATCH_NAME = os.environ.get("NUMPYRO_SVI_BATCH", os.environ.get("VANILLA_SVI_BATCH", "LED8"))
ANIMAL = int(os.environ.get("NUMPYRO_SVI_ANIMAL", os.environ.get("VANILLA_SVI_ANIMAL", "105")))

GUIDE_KIND = os.environ.get("NUMPYRO_SVI_GUIDE", "fullrank")
GUIDE_INIT_SCALE = float(os.environ.get("NUMPYRO_SVI_GUIDE_INIT_SCALE", "0.1"))
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

ABLS = [20.0, 40.0, 60.0]
BATCH_T_TRUNC = {"LED34_even": 0.15}
DEFAULT_T_TRUNC = 0.3

BATCH_CSV = REPO_DIR / "raw_data" / "batch_csvs" / f"batch_{BATCH_NAME}_valid_and_aborts.csv"
ABORT_RESULT_PKL = REPO_DIR / "aborts_ipl_npl_time_fit_results" / f"results_{BATCH_NAME}_animal_{ANIMAL}.pkl"
CONDITION_T_E_AFF_CACHE = (
    REPO_DIR
    / "fit_each_condn"
    / "abl_specific_ild2_delay_agreement"
    / "condition_t_E_aff_extraction_cache.csv"
)

OUTPUT_ROOT = Path(
    os.environ.get(
        "NUMPYRO_SVI_OUTPUT_ROOT",
        str(SCRIPT_DIR / "numpyro_svi_vanilla_lapse_condition_delay_single_animal_outputs"),
    )
).expanduser()
OUTPUT_DIR = OUTPUT_ROOT / f"{BATCH_NAME}_{ANIMAL}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# %%
# =============================================================================
# Dependency preflight
# =============================================================================
required_modules = ["jax", "jaxlib", "numpyro"]
missing_modules = [module for module in required_modules if importlib.util.find_spec(module) is None]
if missing_modules:
    print("Missing dependencies for NumPyro vanilla+lapse SVI:")
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
import numpyro_vanilla_lapse_condition_delay_svi_utils as vanilla_lapse_utils


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
# Load animal data and initial values
# =============================================================================
print(f"Batch/animal: {BATCH_NAME}/{ANIMAL}")
print(f"Batch CSV: {BATCH_CSV}")
print(f"Abort params: {ABORT_RESULT_PKL}")
print(f"Condition delay cache: {CONDITION_T_E_AFF_CACHE}")
print(f"Output folder: {OUTPUT_DIR}")

for required_path in [BATCH_CSV, ABORT_RESULT_PKL, CONDITION_T_E_AFF_CACHE]:
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

init_values = dict(vanilla_lapse_utils.DEFAULT_INIT_VALUES)
init_values["t_E_aff"] = condition_table["t_E_aff_s"].to_numpy(dtype=float)
init_values = vanilla_lapse_utils.clip_init_to_hard_bounds(init_values)

T_trunc = BATCH_T_TRUNC.get(BATCH_NAME, DEFAULT_T_TRUNC)
full_data = make_jax_data(valid_df, V_A, theta_A, t_A_aff, T_trunc)
if MAIN_N_TRIALS_OVERRIDE > 0 and len(valid_df) > MAIN_N_TRIALS_OVERRIDE:
    main_df = valid_df.sample(MAIN_N_TRIALS_OVERRIDE, random_state=RNG_SEED + 1).sort_index().copy()
else:
    main_df = valid_df.copy()
main_data = make_jax_data(main_df, V_A, theta_A, t_A_aff, T_trunc)
n_conditions = int(len(condition_table))

print(f"Valid fitting trials: {len(valid_df)}")
print(f"Main fitting trials: {len(main_df)}")
print(f"Observed conditions: {n_conditions}")
print(condition_table[["condition_id", "ABL", "ILD", "t_E_aff_ms"]].to_string(index=False))
print("\nAbort/proactive posterior means:")
print(f"  V_A      = {V_A:.6g}")
print(f"  theta_A  = {theta_A:.6g}")
print(f"  t_A_aff  = {1e3 * t_A_aff:.3f} ms")
print(f"  T_trunc  = {T_trunc:.3f} s")
print("\nInitial global values:")
for param_name in vanilla_lapse_utils.GLOBAL_PARAM_NAMES:
    value = init_values[param_name]
    if param_name in {"T_0", "del_go"}:
        print(f"  {param_name:<12} = {1e3 * value:.5f} ms")
    else:
        print(f"  {param_name:<12} = {value:.6g}")
print(f"Guide: {GUIDE_KIND}, guide_init_scale={GUIDE_INIT_SCALE:g}")
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
model = lambda data, n_conditions: vanilla_lapse_utils.vanilla_condition_delay_model(
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
print(f"Initial main gradient finite: {vanilla_lapse_utils.tree_all_finite(initial_grad)}")
if not np.isfinite(float(initial_main_log_joint)) or not vanilla_lapse_utils.tree_all_finite(initial_grad):
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
    main_guide = vanilla_lapse_utils.make_guide(
        model,
        GUIDE_KIND,
        init_values,
        init_scale=GUIDE_INIT_SCALE,
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
    active_result = main_result
    active_guide = main_guide
    active_label = f"main_{GUIDE_KIND}"
    active_convergence_df = main_convergence_df
else:
    print("\nSkipping main SVI because RUN_MAIN_SVI=False.")


# %%
# =============================================================================
# Posterior samples and saved outputs
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
    guide_params_np = vanilla_lapse_utils.tree_to_numpy(active_result.params)

    finite_report_df, all_posterior_finite = vanilla_lapse_utils.finite_sample_report(posterior_np)
    posterior_summary_df = vanilla_lapse_utils.posterior_samples_to_frame(posterior_np, condition_table)

    sample_npz = OUTPUT_DIR / f"{active_label}_posterior_samples.npz"
    guide_params_pkl = OUTPUT_DIR / f"{active_label}_guide_params.pkl"
    summary_csv = OUTPUT_DIR / f"{active_label}_posterior_summary.csv"
    finite_report_csv = OUTPUT_DIR / f"{active_label}_posterior_finite_report.csv"
    condition_csv = OUTPUT_DIR / "condition_table.csv"
    loss_csv = OUTPUT_DIR / f"{active_label}_loss.csv"
    convergence_csv = OUTPUT_DIR / f"{active_label}_convergence_checks.csv"
    bundle_pkl = OUTPUT_DIR / f"{active_label}_variational_posterior_bundle.pkl"

    np.savez(sample_npz, **posterior_np)
    with guide_params_pkl.open("wb") as handle:
        pickle.dump(guide_params_np, handle)
    posterior_summary_df.to_csv(summary_csv, index=False)
    finite_report_df.to_csv(finite_report_csv, index=False)
    condition_table.to_csv(condition_csv, index=False)
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
    ax.set_title(f"{BATCH_NAME}/{ANIMAL} {active_label} loss")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(loss_png, dpi=200)

    delay_values_ms = 1e3 * posterior_np["t_E_aff"]
    delay_q025 = np.nanquantile(delay_values_ms, 0.025, axis=0)
    delay_q500 = np.nanquantile(delay_values_ms, 0.5, axis=0)
    delay_q975 = np.nanquantile(delay_values_ms, 0.975, axis=0)
    delay_init_ms = 1e3 * condition_table["t_E_aff_s"].to_numpy(dtype=float)

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
    fig.suptitle(f"{BATCH_NAME}/{ANIMAL} {active_label} vanilla+lapse condition-delay intervals")
    fig.tight_layout()
    fig.savefig(delay_png, dpi=200)

    bundle = {
        "schema_version": 1,
        "model_name": "vanilla_ipl_lapse_condition_delay_svi",
        "note": (
            "NumPyro SVI fit of vanilla/IPL+lapse parameters with condition-wise t_E_aff. "
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
            "T_trunc": float(T_trunc),
            "K_max": int(K_MAX),
            "posterior_n_samples": int(POSTERIOR_N_SAMPLES),
            "rng_seed": int(RNG_SEED),
            "lapse_rt_window": 1.0,
        },
        "input_paths": {
            "batch_csv": str(BATCH_CSV),
            "abort_result_pkl": str(ABORT_RESULT_PKL),
            "condition_t_E_aff_cache": str(CONDITION_T_E_AFF_CACHE),
        },
        "output_paths": {
            "posterior_samples_npz": str(sample_npz),
            "guide_params_pkl": str(guide_params_pkl),
            "posterior_summary_csv": str(summary_csv),
            "finite_report_csv": str(finite_report_csv),
            "condition_table_csv": str(condition_csv),
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
        "loss_trace": loss_df,
        "convergence_checks": active_convergence_df,
    }
    with bundle_pkl.open("wb") as handle:
        pickle.dump(bundle, handle)

    print("\nPosterior finite-sample report:")
    print(finite_report_df.to_string(index=False))
    if not all_posterior_finite:
        print("WARNING: posterior samples contain NaN/Inf.")

    print("\nSaved outputs:")
    print(f"  samples: {sample_npz}")
    print(f"  guide params: {guide_params_pkl}")
    print(f"  summary: {summary_csv}")
    print(f"  finite report: {finite_report_csv}")
    print(f"  condition table: {condition_csv}")
    print(f"  loss CSV: {loss_csv}")
    print(f"  convergence checks: {convergence_csv}")
    print(f"  loss plot: {loss_png}")
    print(f"  delay intervals: {delay_png}")
    print(f"  VP bundle: {bundle_pkl}")

# %%
