# %%
"""Fit pooled LED7 session-type-8/9 proactive+lapse abort models."""

# %%
# =============================================================================
# Editable parameters
# =============================================================================
from pathlib import Path
from time import perf_counter
import hashlib
import json
import os
import pickle
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
INPUT_MAT_PATH = REPO_DIR / "raw_data" / "outMatrix_LED7_latest.mat"
TABLE_NAME = "totalout_stGtACRII"
EXPECTED_SOURCE_ROWS = 571_060
EXPECTED_SOURCE_SHA256 = (
    "9844c0e940db52f68365612f7baf934f5fbce6d7e96f5ba3570bc73ae593941b"
)

OUTPUT_ROOT = Path(
    os.environ.get(
        "LED7_S8_S9_AGG_SVI_OUTPUT_ROOT",
        str(
            SCRIPT_DIR
            / "numpyro_svi_led7_s8_s9_aggregate_no_trunc_exp_lapse_outputs"
        ),
    )
).expanduser()
SESSION_TYPES = tuple(
    int(value.strip())
    for value in os.environ.get(
        "LED7_S8_S9_AGG_SVI_SESSION_TYPES", "8,9"
    ).split(",")
    if value.strip()
)
ALLOWED_SESSION_TYPES = (8, 9)
ANIMALS = (90, 92, 93, 98, 99, 100, 102, 103)
TRAINING_LEVEL = 16
ALLOWED_REPEAT_TRIALS = {0, 2}
LED_TRIAL_VALUES = (0, 1)

EXPECTED = {
    8: {
        "pre_total": 148_428,
        "sessions": 215,
        "by_led": {
            0: {"pre": 92_784, "abort": 12_570, "censored": 77_310, "missing_event3": 0},
            1: {"pre": 55_644, "abort": 10_354, "censored": 39_313, "missing_event3": 2},
        },
    },
    9: {
        "pre_total": 100_788,
        "sessions": 191,
        "by_led": {
            0: {"pre": 61_408, "abort": 10_107, "censored": 48_599, "missing_event3": 1},
            1: {"pre": 39_380, "abort": 8_747, "censored": 25_928, "missing_event3": 5},
        },
    },
}

QUADRATURE_NODES = int(
    os.environ.get("LED7_S8_S9_AGG_SVI_QUADRATURE_NODES", "64")
)
MAIN_STEPS = int(
    os.environ.get("LED7_S8_S9_AGG_SVI_MAIN_STEPS", "150000")
)
SAFETY_MAX_STEPS = int(
    os.environ.get("LED7_S8_S9_AGG_SVI_SAFETY_MAX_STEPS", "400000")
)
SVI_CHECK_EVERY = int(
    os.environ.get("LED7_S8_S9_AGG_SVI_CHECK_EVERY", "1000")
)
SVI_MIN_STEPS = int(
    os.environ.get("LED7_S8_S9_AGG_SVI_MIN_STEPS", "150000")
)
SVI_MIN_IMPROVEMENT_REL = float(
    os.environ.get("LED7_S8_S9_AGG_SVI_MIN_IMPROVEMENT_REL", "0.001")
)
SVI_NO_IMPROVE_PATIENCE_WINDOWS = int(
    os.environ.get("LED7_S8_S9_AGG_SVI_PATIENCE_WINDOWS", "12")
)
SVI_REL_TOL = float(
    os.environ.get("LED7_S8_S9_AGG_SVI_REL_TOL", "0.001")
)
LEARNING_RATE = float(
    os.environ.get("LED7_S8_S9_AGG_SVI_LR", "0.0002")
)
CLIP_NORM = float(
    os.environ.get("LED7_S8_S9_AGG_SVI_CLIP_NORM", "1.0")
)
GUIDE_INIT_SCALE = float(
    os.environ.get("LED7_S8_S9_AGG_SVI_GUIDE_INIT_SCALE", "0.1")
)
POSTERIOR_N_SAMPLES = int(
    os.environ.get("LED7_S8_S9_AGG_SVI_POSTERIOR_SAMPLES", "10000")
)
RNG_SEED = int(os.environ.get("LED7_S8_S9_AGG_SVI_SEED", "0"))
VALIDATE_ONLY = os.environ.get(
    "LED7_S8_S9_AGG_SVI_VALIDATE_ONLY", "0"
) == "1"
OVERWRITE = os.environ.get("LED7_S8_S9_AGG_SVI_OVERWRITE", "0") == "1"


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
import numpyro
import pandas as pd
from jax import random
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.svi import SVIRunResult
from numpyro.infer.util import log_density

try:
    from matio import load_from_mat
except ImportError as exc:
    raise ImportError(
        "Run with the isolated MAT-table decoder, for example:\n"
        "env PYTHONPATH=/tmp/codex_mat_io_led789 .venv/bin/python "
        "fitting_aborts/numpyro_svi_proactive_led_step_jump_no_trunc_exp_lapse_"
        "led7_s8_s9_aggregate.py"
    ) from exc

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import numpyro_proactive_led_step_jump_no_trunc_exp_lapse_svi_utils as svi_utils


# %%
# =============================================================================
# SVI and serialization helpers
# =============================================================================
def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_optimizer():
    return numpyro.optim.ClippedAdam(LEARNING_RATE, clip_norm=CLIP_NORM)


def to_numpy_tree(tree):
    return jax.tree_util.tree_map(
        lambda value: np.asarray(jax.device_get(value)), tree
    )


def run_svi_with_convergence_checks(svi, rng_key, data, fit_label):
    """Run fixed-size JIT windows, retain the best, and require a patience tail."""

    def full_window_scan(svi_state, scan_data):
        def body_fn(active_state, _):
            return svi.stable_update(active_state, scan_data)

        return jax.lax.scan(
            body_fn, svi_state, None, length=SVI_CHECK_EVERY
        )

    run_full_window = jax.jit(full_window_scan)
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
    extended_after_main = False
    stop_reason = "safety_cap_restore_best"

    print(
        f"\n[{fit_label}] Full-rank SVI: min={SVI_MIN_STEPS:,}, "
        f"check={SVI_CHECK_EVERY:,}, patience="
        f"{SVI_NO_IMPROVE_PATIENCE_WINDOWS}, safety={SAFETY_MAX_STEPS:,}"
    )
    while True:
        chunk_index = len(convergence_rows) + 1
        chunk_steps = min(SVI_CHECK_EVERY, active_max_steps - completed_steps)
        if chunk_steps <= 0:
            if active_max_steps < SAFETY_MAX_STEPS:
                active_max_steps = SAFETY_MAX_STEPS
                extended_after_main = True
                print(
                    f"[{fit_label}] Extending the same optimizer state to "
                    f"{SAFETY_MAX_STEPS:,} steps for a full patience tail."
                )
                continue
            break

        start_step = completed_steps + 1
        end_step = completed_steps + chunk_steps
        chunk_start = perf_counter()
        if state is None:
            state = svi.init(random.fold_in(rng_key, chunk_index), data)
        if chunk_steps == SVI_CHECK_EVERY:
            state, window_losses_jax = run_full_window(state, data)
            chunk_params = svi.get_params(state)
        else:
            chunk_result = svi.run(
                random.fold_in(rng_key, chunk_index),
                chunk_steps,
                data,
                progress_bar=False,
                init_state=state,
                stable_update=True,
            )
            state = chunk_result.state
            window_losses_jax = chunk_result.losses
            chunk_params = chunk_result.params

        window_losses = np.asarray(
            jax.device_get(window_losses_jax), dtype=float
        )
        all_losses.append(window_losses)
        completed_steps = end_step
        chunk_seconds = perf_counter() - chunk_start
        finite_mask = np.isfinite(window_losses)
        finite_losses = window_losses[finite_mask]
        n_nonfinite = int(np.sum(~finite_mask))
        window_mean = (
            float(np.mean(finite_losses)) if finite_losses.size else np.nan
        )
        window_median = (
            float(np.median(finite_losses)) if finite_losses.size else np.nan
        )
        window_last = (
            float(window_losses[-1])
            if window_losses.size and np.isfinite(window_losses[-1])
            else np.nan
        )
        window_min = (
            float(np.min(finite_losses)) if finite_losses.size else np.nan
        )
        window_max = (
            float(np.max(finite_losses)) if finite_losses.size else np.nan
        )
        if finite_losses.size > 1:
            finite_x = np.flatnonzero(finite_mask).astype(float)
            slope_per_1000 = float(
                np.polyfit(finite_x, finite_losses, 1)[0] * 1000.0
            )
        else:
            slope_per_1000 = np.nan

        if np.isfinite(prev_window_mean) and np.isfinite(window_mean):
            delta_from_prev = window_mean - prev_window_mean
            relative_change = abs(delta_from_prev) / max(
                1.0, abs(prev_window_mean)
            )
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
            improved_best = window_mean < best_window_mean
            significant_improvement = improved_best and (
                not np.isfinite(relative_improvement_from_best)
                or relative_improvement_from_best >= SVI_MIN_IMPROVEMENT_REL
            )

        if improved_best and n_nonfinite == 0:
            best_state = state
            best_params = chunk_params
            best_window_mean = window_mean
            best_window_chunk = chunk_index
            best_window_end_step = end_step
            no_improve_window_count = (
                0 if significant_improvement else no_improve_window_count + 1
            )
        else:
            no_improve_window_count += 1

        stable_window_count = stable_window_count + 1 if is_stable else 0
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
                "delta_mean_from_prev": delta_from_prev,
                "relative_mean_change": relative_change,
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
            f"[{fit_label}] {start_step:06d}-{end_step:06d} "
            f"mean={window_mean:.6g}, rel={rel_text}, "
            f"slope/1k={slope_per_1000:.3g}, best={best_window_end_step}, "
            f"post_best={windows_since_best}, nonfinite={n_nonfinite}, "
            f"{chunk_seconds:.1f}s"
        )

        if n_nonfinite:
            stop_reason = "nonfinite_loss_restore_best"
            break
        if can_stop:
            stop_reason = "patience12_restore_best"
            break
        prev_window_mean = window_mean

    losses = (
        np.concatenate(all_losses) if all_losses else np.array([], dtype=float)
    )
    if best_state is None or best_params is None:
        raise RuntimeError(f"{fit_label}: no finite best optimizer state exists.")
    print(
        f"[{fit_label}] Restored step {best_window_end_step:,}; "
        f"final checked step {completed_steps:,}; stop={stop_reason}"
    )
    return (
        SVIRunResult(best_params, best_state, jnp.asarray(losses)),
        pd.DataFrame(convergence_rows),
        stop_reason,
        extended_after_main,
    )


# %%
# =============================================================================
# Load and verify the source once
# =============================================================================
if not INPUT_MAT_PATH.exists():
    raise FileNotFoundError(INPUT_MAT_PATH)
if set(SESSION_TYPES) - set(ALLOWED_SESSION_TYPES) or not SESSION_TYPES:
    raise ValueError(
        f"Session types must be drawn from {ALLOWED_SESSION_TYPES}; "
        f"found {SESSION_TYPES}."
    )

source_sha256 = file_sha256(INPUT_MAT_PATH)
if source_sha256 != EXPECTED_SOURCE_SHA256:
    raise RuntimeError(
        f"Source MAT hash changed: {source_sha256}; expected "
        f"{EXPECTED_SOURCE_SHA256}."
    )
mat_contents = load_from_mat(INPUT_MAT_PATH, variable_names=[TABLE_NAME])
if TABLE_NAME not in mat_contents:
    raise KeyError(f"{TABLE_NAME!r} is absent from {INPUT_MAT_PATH}.")
raw_df = mat_contents[TABLE_NAME]
if not isinstance(raw_df, pd.DataFrame):
    raise TypeError(f"Expected a DataFrame, found {type(raw_df).__name__}.")
if len(raw_df) != EXPECTED_SOURCE_ROWS:
    raise RuntimeError(
        f"Expected {EXPECTED_SOURCE_ROWS:,} source rows, found {len(raw_df):,}."
    )

required_columns = [
    "animal",
    "session",
    "block",
    "trial",
    "training_level",
    "session_type",
    "repeat_trial",
    "LED_trial",
    "abort_event",
    "success",
    "timed_fix",
    "intended_fix",
    "LED_onset_time",
]
missing_columns = [column for column in required_columns if column not in raw_df]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")
if not raw_df.index.is_unique:
    raise RuntimeError("Source table row index is not unique.")

training_df = raw_df.loc[raw_df["training_level"].eq(TRAINING_LEVEL)].copy()
session_df = training_df.loc[
    training_df["session_type"].isin(SESSION_TYPES)
].copy()
repeat_df = session_df.loc[
    session_df["repeat_trial"].isin(ALLOWED_REPEAT_TRIALS)
    | session_df["repeat_trial"].isna()
].copy()
filtered_df = repeat_df.loc[
    repeat_df["LED_trial"].isin(LED_TRIAL_VALUES)
].copy()

print(f"Source: {INPUT_MAT_PATH.resolve()} :: {TABLE_NAME}")
print(f"Source SHA256: {source_sha256}")
print(f"Output root: {OUTPUT_ROOT.resolve()}")
print(f"Session types: {SESSION_TYPES}")
print(f"Animals: {ANIMALS} (trial pooled)")
print("Likelihood: finite event 3 density + finite successful survival")
print("Truncation: none; LED ON scope: every LED_trial == 1 row")


# %%
# =============================================================================
# Construct and validate each session-type likelihood payload
# =============================================================================
fit_payloads = {}
for session_type in SESSION_TYPES:
    expected = EXPECTED[session_type]
    candidate_df = filtered_df.loc[
        filtered_df["session_type"].eq(session_type)
    ].copy()
    candidate_df["effective_scheduled_onset"] = (
        candidate_df["LED_onset_time"]
        if session_type == 9
        else candidate_df["intended_fix"] - candidate_df["LED_onset_time"]
    )
    animals = sorted(
        candidate_df["animal"].dropna().astype(int).unique().tolist()
    )
    if animals != list(ANIMALS):
        raise RuntimeError(
            f"s{session_type}: expected animals {ANIMALS}, found {animals}."
        )
    if len(candidate_df) != expected["pre_total"]:
        raise RuntimeError(
            f"s{session_type}: expected {expected['pre_total']:,} rows, "
            f"found {len(candidate_df):,}."
        )
    n_sessions = candidate_df[["animal", "session"]].drop_duplicates().shape[0]
    if n_sessions != expected["sessions"]:
        raise RuntimeError(
            f"s{session_type}: expected {expected['sessions']} animal-session "
            f"pairs, found {n_sessions}."
        )
    if not candidate_df["training_level"].eq(TRAINING_LEVEL).all():
        raise RuntimeError(f"s{session_type}: another training level survived.")
    if not candidate_df["session_type"].eq(session_type).all():
        raise RuntimeError(f"s{session_type}: another session type survived.")
    repeat_values = set(
        candidate_df["repeat_trial"].dropna().astype(int).unique()
    )
    if not repeat_values.issubset(ALLOWED_REPEAT_TRIALS):
        raise RuntimeError(f"s{session_type}: invalid repeat_trial values.")
    if set(candidate_df["LED_trial"].astype(int).unique()) != {0, 1}:
        raise RuntimeError(f"s{session_type}: expected LED_trial 0 and 1.")
    timing = candidate_df[["intended_fix", "effective_scheduled_onset"]]
    if not np.isfinite(timing).all(axis=None):
        raise RuntimeError(f"s{session_type}: scheduled timing is non-finite.")
    if (candidate_df["effective_scheduled_onset"] < 0).any():
        raise RuntimeError(f"s{session_type}: negative scheduled onset.")
    if (
        candidate_df["effective_scheduled_onset"]
        > candidate_df["intended_fix"] + 1e-12
    ).any():
        raise RuntimeError(f"s{session_type}: onset occurs after intended_fix.")

    manifest = candidate_df[
        [
            "animal",
            "session",
            "block",
            "trial",
            "LED_trial",
            "abort_event",
            "success",
            "timed_fix",
            "intended_fix",
            "LED_onset_time",
            "effective_scheduled_onset",
        ]
    ].copy()
    manifest.insert(0, "source_row_index", candidate_df.index.to_numpy(dtype=int))
    manifest.insert(1, "selection_order", np.arange(len(candidate_df), dtype=int))
    if not manifest["source_row_index"].is_unique:
        raise RuntimeError(f"s{session_type}: source rows repeat.")

    finite_timing = np.isfinite(
        manifest[["timed_fix", "intended_fix", "effective_scheduled_onset"]]
    ).all(axis=1)
    is_event3 = manifest["abort_event"].eq(3)
    is_success = manifest["success"].isin([-1, 1])
    if (is_event3 & is_success).any():
        raise RuntimeError(f"s{session_type}: event and censored roles overlap.")
    manifest["likelihood_role"] = "excluded_other_event_or_outcome"
    manifest.loc[~finite_timing, "likelihood_role"] = "excluded_missing_timing"
    manifest.loc[finite_timing & is_success, "likelihood_role"] = "censored"
    manifest.loc[finite_timing & is_event3, "likelihood_role"] = "abort"
    fit_df = manifest.loc[
        manifest["likelihood_role"].isin(["abort", "censored"])
    ].copy()
    abort_df = fit_df.loc[fit_df["likelihood_role"].eq("abort")]
    censored_df = fit_df.loc[fit_df["likelihood_role"].eq("censored")]
    if not (abort_df["timed_fix"] < abort_df["intended_fix"]).all():
        raise RuntimeError(f"s{session_type}: event-3 time at/after stimulus.")
    if not (censored_df["timed_fix"] >= censored_df["intended_fix"]).all():
        raise RuntimeError(f"s{session_type}: censored row before stimulus.")

    category_frames = {
        led: fit_df.loc[fit_df["LED_trial"].eq(led)].copy()
        for led in LED_TRIAL_VALUES
    }
    audit_rows = []
    for led in LED_TRIAL_VALUES:
        led_manifest = manifest.loc[manifest["LED_trial"].eq(led)]
        led_fit = category_frames[led]
        observed = {
            "pre": int(len(led_manifest)),
            "abort": int(led_fit["likelihood_role"].eq("abort").sum()),
            "censored": int(led_fit["likelihood_role"].eq("censored").sum()),
            "missing_event3": int(
                (
                    led_manifest["abort_event"].eq(3)
                    & ~np.isfinite(led_manifest["timed_fix"])
                ).sum()
            ),
        }
        if observed != expected["by_led"][led]:
            raise RuntimeError(
                f"s{session_type} LED {led}: counts changed: {observed}."
            )
        audit_rows.append(
            {
                "session_type": session_type,
                "LED_trial": led,
                "pre_likelihood_rows": observed["pre"],
                "finite_event3_aborts": observed["abort"],
                "finite_success_censored": observed["censored"],
                "likelihood_rows": observed["abort"] + observed["censored"],
                "missing_event3_timing": observed["missing_event3"],
                "excluded_rows": int(
                    observed["pre"] - observed["abort"] - observed["censored"]
                ),
                "early_aborts_below_300ms": int(
                    (
                        led_fit["likelihood_role"].eq("abort")
                        & (led_fit["timed_fix"] < 0.3)
                    ).sum()
                ),
                "scheduled_onset_min_s": float(
                    led_manifest["effective_scheduled_onset"].min()
                ),
                "scheduled_onset_max_s": float(
                    led_manifest["effective_scheduled_onset"].max()
                ),
                "intended_fix_min_s": float(led_manifest["intended_fix"].min()),
                "intended_fix_max_s": float(led_manifest["intended_fix"].max()),
            }
        )

    on_abort_df = category_frames[1].loc[
        category_frames[1]["likelihood_role"].eq("abort")
    ]
    on_censored_df = category_frames[1].loc[
        category_frames[1]["likelihood_role"].eq("censored")
    ]
    off_abort_df = category_frames[0].loc[
        category_frames[0]["likelihood_role"].eq("abort")
    ]
    off_censored_df = category_frames[0].loc[
        category_frames[0]["likelihood_role"].eq("censored")
    ]
    jax_data = {
        "on_abort_t": jnp.asarray(on_abort_df["timed_fix"].to_numpy(float)),
        "on_abort_t_led": jnp.asarray(
            on_abort_df["effective_scheduled_onset"].to_numpy(float)
        ),
        "on_censored_t_stim": jnp.asarray(
            on_censored_df["intended_fix"].to_numpy(float)
        ),
        "on_censored_t_led": jnp.asarray(
            on_censored_df["effective_scheduled_onset"].to_numpy(float)
        ),
        "off_abort_t": jnp.asarray(off_abort_df["timed_fix"].to_numpy(float)),
        "off_censored_t_stim": jnp.asarray(
            off_censored_df["intended_fix"].to_numpy(float)
        ),
    }
    counts = {
        "pre_likelihood_total": int(len(manifest)),
        "likelihood_total": int(len(fit_df)),
        "animal_session_pairs": int(n_sessions),
        "by_led": {
            str(row["LED_trial"]): {
                key: row[key]
                for key in [
                    "pre_likelihood_rows",
                    "finite_event3_aborts",
                    "finite_success_censored",
                    "likelihood_rows",
                    "missing_event3_timing",
                    "excluded_rows",
                    "early_aborts_below_300ms",
                ]
            }
            for row in audit_rows
        },
        "by_animal_likelihood_rows": {
            str(animal): int(fit_df["animal"].eq(animal).sum())
            for animal in ANIMALS
        },
    }
    fit_payloads[session_type] = {
        "manifest": manifest,
        "fit_df": fit_df,
        "category_frames": category_frames,
        "jax_data": jax_data,
        "audit": pd.DataFrame(audit_rows),
        "counts": counts,
    }
    print(f"\ns{session_type} likelihood audit:")
    print(pd.DataFrame(audit_rows).to_string(index=False))


# %%
# =============================================================================
# Validate initialization, then fit requested session types sequentially
# =============================================================================
inconclusive_sessions = []
for session_type in SESSION_TYPES:
    fit_label = f"LED7 session type {session_type}"
    payload = fit_payloads[session_type]
    jax_data = payload["jax_data"]
    init_values = svi_utils.clip_init_to_hard_bounds(
        dict(svi_utils.DEFAULT_INIT_VALUES)
    )
    expected_init_values = {
        "V_A_base": 1.6,
        "V_A_post_LED": 3.0,
        "theta_A": 2.5,
        "del_a_minus_del_LED": 0.04,
        "del_m_plus_del_LED": 0.05,
        "lapse_prob": 0.05,
        "beta_lapse": 5.0,
    }
    if init_values != expected_init_values:
        raise RuntimeError(
            f"Established initialization changed: {init_values}."
        )
    model = lambda data: svi_utils.proactive_led_step_jump_no_trunc_lapse_model(
        data, n_quad=QUADRATURE_NODES
    )

    def log_joint_from_values(values):
        log_joint, _ = log_density(model, (jax_data,), {}, values)
        return log_joint

    initial_log_joint = log_joint_from_values(init_values)
    initial_grad = jax.grad(log_joint_from_values)(init_values)
    initial_log_likelihood = (
        svi_utils.proactive_led_step_jump_no_trunc_lapse_loglike_jax(
            init_values, jax_data, n_quad=QUADRATURE_NODES
        )
    )
    print(f"\n[{fit_label}] Initial values: {init_values}")
    print(f"[{fit_label}] Initial log joint: {float(initial_log_joint):.8f}")
    print(
        f"[{fit_label}] Initial log likelihood: "
        f"{float(initial_log_likelihood):.8f}"
    )
    if not np.isfinite(float(initial_log_joint)):
        raise RuntimeError(f"{fit_label}: initial log joint is non-finite.")
    if not np.isfinite(float(initial_log_likelihood)):
        raise RuntimeError(f"{fit_label}: initial log likelihood is non-finite.")
    if not svi_utils.tree_all_finite(initial_grad):
        raise RuntimeError(f"{fit_label}: initial gradients are non-finite.")
    if VALIDATE_ONLY:
        print(f"[{fit_label}] Validation-only check passed; fit not started.")
        continue

    fit_dir = OUTPUT_ROOT / f"session_type_{session_type}"
    if fit_dir.exists() and any(fit_dir.iterdir()) and not OVERWRITE:
        raise FileExistsError(
            f"Refusing to overwrite nonempty fit directory: {fit_dir}."
        )

    guide = svi_utils.make_fullrank_guide(
        model, init_values, init_scale=GUIDE_INIT_SCALE
    )
    svi = SVI(model, guide, make_optimizer(), Trace_ELBO())
    fit_start = perf_counter()
    fit_result, convergence_df, stop_reason, extended_after_main = (
        run_svi_with_convergence_checks(
            svi, random.PRNGKey(RNG_SEED), jax_data, fit_label
        )
    )
    fit_elapsed_seconds = perf_counter() - fit_start
    losses = np.asarray(jax.device_get(fit_result.losses), dtype=float)

    posterior_samples = guide.sample_posterior(
        random.PRNGKey(RNG_SEED + 1000),
        fit_result.params,
        sample_shape=(POSTERIOR_N_SAMPLES,),
    )
    posterior_np = {
        name: np.asarray(jax.device_get(values), dtype=float).reshape(-1)
        for name, values in posterior_samples.items()
    }
    guide_params_np = to_numpy_tree(fit_result.params)
    all_posterior_finite = all(
        np.isfinite(posterior_np[name]).all()
        for name in svi_utils.PARAM_NAMES
    )
    guide_state_finite = svi_utils.tree_all_finite(guide_params_np)
    summary_rows = []
    for name in svi_utils.PARAM_NAMES:
        values = posterior_np[name]
        bounds = svi_utils.PARAM_BOUNDS[name]
        hard_low, hard_high = bounds["hard"]
        if not ((values >= hard_low) & (values <= hard_high)).all():
            raise RuntimeError(f"{fit_label}: {name} samples exceed hard bounds.")
        summary_rows.append(
            {
                "session_type": session_type,
                "parameter": name,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "q025": float(np.quantile(values, 0.025)),
                "median": float(np.quantile(values, 0.5)),
                "q975": float(np.quantile(values, 0.975)),
                "n_samples": int(len(values)),
                "n_nonfinite": int(np.sum(~np.isfinite(values))),
                "hard_low": hard_low,
                "hard_high": hard_high,
                "plausible_low": bounds["plausible"][0],
                "plausible_high": bounds["plausible"][1],
            }
        )
    posterior_summary_df = pd.DataFrame(summary_rows)
    derived_t_a_aff = (
        posterior_np["del_a_minus_del_LED"]
        + posterior_np["del_m_plus_del_LED"]
    )
    derived_summary_df = pd.DataFrame(
        [
            {
                "session_type": session_type,
                "parameter": "t_A_aff",
                "definition": "del_a_minus_del_LED + del_m_plus_del_LED",
                "mean": float(np.mean(derived_t_a_aff)),
                "std": float(np.std(derived_t_a_aff)),
                "q025": float(np.quantile(derived_t_a_aff, 0.025)),
                "median": float(np.quantile(derived_t_a_aff, 0.5)),
                "q975": float(np.quantile(derived_t_a_aff, 0.975)),
                "n_samples": int(len(derived_t_a_aff)),
                "n_nonfinite": int(np.sum(~np.isfinite(derived_t_a_aff))),
            }
        ]
    )

    posterior_means = {
        name: float(np.mean(posterior_np[name]))
        for name in svi_utils.PARAM_NAMES
    }
    posterior_mean_log_likelihood = float(
        svi_utils.proactive_led_step_jump_no_trunc_lapse_loglike_jax(
            posterior_means, jax_data, n_quad=QUADRATURE_NODES
        )
    )
    initial_log_likelihood_float = float(initial_log_likelihood)
    log_likelihood_improvement = (
        posterior_mean_log_likelihood - initial_log_likelihood_float
    )
    if not np.isfinite(posterior_mean_log_likelihood):
        raise RuntimeError(f"{fit_label}: posterior likelihood is non-finite.")
    if log_likelihood_improvement <= 1.0:
        raise RuntimeError(
            f"{fit_label}: posterior likelihood did not improve materially "
            f"({log_likelihood_improvement:.6g})."
        )

    best_step = int(convergence_df.iloc[-1]["best_end_step_so_far"])
    checked_step = int(convergence_df.iloc[-1]["end_step"])
    n_nonfinite_losses = int(convergence_df["n_nonfinite"].sum())
    if (
        stop_reason == "patience12_restore_best"
        and n_nonfinite_losses == 0
        and all_posterior_finite
        and guide_state_finite
    ):
        status = "complete"
    elif n_nonfinite_losses or not all_posterior_finite or not guide_state_finite:
        status = "failed_validation"
    else:
        status = "inconclusive_safety_cap"
        inconclusive_sessions.append(session_type)

    fit_dir.mkdir(parents=True, exist_ok=True)
    label = "main_fullrank"
    sample_npz = fit_dir / f"{label}_posterior_samples.npz"
    guide_params_pkl = fit_dir / f"{label}_guide_params.pkl"
    posterior_summary_csv = fit_dir / f"{label}_posterior_summary.csv"
    derived_summary_csv = fit_dir / f"{label}_derived_posterior_summary.csv"
    loss_csv = fit_dir / f"{label}_loss.csv"
    convergence_csv = fit_dir / f"{label}_convergence_checks.csv"
    loss_png = fit_dir / f"{label}_loss.png"
    bundle_pkl = fit_dir / f"{label}_variational_posterior_bundle.pkl"
    manifest_csv = fit_dir / "source_row_manifest.csv"
    likelihood_audit_csv = fit_dir / "likelihood_audit.csv"
    run_summary_json = fit_dir / "run_summary.json"

    np.savez_compressed(sample_npz, **posterior_np)
    with guide_params_pkl.open("wb") as handle:
        pickle.dump(guide_params_np, handle)
    posterior_summary_df.to_csv(posterior_summary_csv, index=False)
    derived_summary_df.to_csv(derived_summary_csv, index=False)
    loss_df = pd.DataFrame(
        {"step": np.arange(1, len(losses) + 1), "negative_elbo": losses}
    )
    loss_df.to_csv(loss_csv, index=False)
    convergence_df.to_csv(convergence_csv, index=False)
    payload["manifest"].to_csv(manifest_csv, index=False)
    payload["audit"].to_csv(likelihood_audit_csv, index=False)

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
    ax.set_title(f"LED7 session type {session_type}: aggregate SVI")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(loss_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    config = {
        "model_name": "proactive_led_step_jump_no_trunc_exp_lapse_svi",
        "source_mat": str(INPUT_MAT_PATH.resolve()),
        "source_table": TABLE_NAME,
        "source_sha256": source_sha256,
        "animals": list(ANIMALS),
        "training_level": TRAINING_LEVEL,
        "session_type": session_type,
        "repeat_trial": [0, 2, "NaN"],
        "led_trial": [0, 1],
        "scheduled_onset_definition": (
            "LED_onset_time"
            if session_type == 9
            else "intended_fix - LED_onset_time"
        ),
        "pooling": "trial_weighted",
        "truncation": "none",
        "quadrature_nodes": QUADRATURE_NODES,
        "main_steps_initial_max": MAIN_STEPS,
        "safety_max_steps": SAFETY_MAX_STEPS,
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
        "led_off_weight": 1.0,
        "led_on_scope": "all LED_trial == 1 rows",
        "lapse_process": "exponential in fixation time",
        "likelihood_population": (
            "finite abort_event == 3 plus finite success in {-1,+1} "
            "censored at intended_fix"
        ),
    }
    bundle = {
        "schema_version": 1,
        "config": config,
        "trial_counts": payload["counts"],
        "init_values": init_values,
        "initial_log_joint": float(initial_log_joint),
        "initial_log_likelihood": initial_log_likelihood_float,
        "posterior_mean_log_likelihood": posterior_mean_log_likelihood,
        "log_likelihood_improvement": log_likelihood_improvement,
        "guide_params": guide_params_np,
        "posterior_samples": posterior_np,
        "posterior_summary": posterior_summary_df,
        "derived_posterior_summary": derived_summary_df,
        "loss_trace": loss_df,
        "convergence_checks": convergence_df,
        "stop_reason": stop_reason,
        "fit_elapsed_seconds": fit_elapsed_seconds,
        "input_path": str(INPUT_MAT_PATH.resolve()),
        "source_row_manifest": str(manifest_csv.resolve()),
    }
    with bundle_pkl.open("wb") as handle:
        pickle.dump(bundle, handle)

    run_summary = {
        "status": status,
        "stop_reason": stop_reason,
        "session_type": session_type,
        "animals": list(ANIMALS),
        "fit_elapsed_seconds": fit_elapsed_seconds,
        "fit_elapsed_minutes": fit_elapsed_seconds / 60.0,
        "completed_steps": checked_step,
        "restored_best_step": best_step,
        "extended_after_main": extended_after_main,
        "n_nonfinite_losses": n_nonfinite_losses,
        "all_posterior_samples_finite": all_posterior_finite,
        "guide_state_finite": guide_state_finite,
        "initial_log_joint": float(initial_log_joint),
        "initial_log_likelihood": initial_log_likelihood_float,
        "posterior_mean_log_likelihood": posterior_mean_log_likelihood,
        "log_likelihood_improvement": log_likelihood_improvement,
        "trial_counts": payload["counts"],
        "config": config,
        "artifacts": {
            "posterior_samples_npz": str(sample_npz.resolve()),
            "guide_params_pkl": str(guide_params_pkl.resolve()),
            "posterior_summary_csv": str(posterior_summary_csv.resolve()),
            "derived_posterior_summary_csv": str(derived_summary_csv.resolve()),
            "loss_csv": str(loss_csv.resolve()),
            "convergence_csv": str(convergence_csv.resolve()),
            "loss_png": str(loss_png.resolve()),
            "variational_posterior_bundle_pkl": str(bundle_pkl.resolve()),
            "source_row_manifest_csv": str(manifest_csv.resolve()),
            "likelihood_audit_csv": str(likelihood_audit_csv.resolve()),
        },
    }
    run_summary_json.write_text(json.dumps(run_summary, indent=2) + "\n")

    print(f"\n[{fit_label}] Posterior summary:")
    print(posterior_summary_df.to_string(index=False))
    print(derived_summary_df.to_string(index=False))
    print(f"[{fit_label}] Fit elapsed: {fit_elapsed_seconds / 60.0:.2f} min")
    print(f"[{fit_label}] Status: {status}; stop: {stop_reason}")
    print(f"[{fit_label}] Run summary: {run_summary_json.resolve()}")

if VALIDATE_ONLY:
    print("\nAll requested validation-only checks passed; no fit outputs were written.")
elif inconclusive_sessions:
    raise SystemExit(
        "Inconclusive safety-cap fits: "
        + ", ".join(f"session type {value}" for value in inconclusive_sessions)
    )
