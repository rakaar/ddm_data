# %%
"""Fit pooled LED7 s7 abort models before and after s9 timing matching."""

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
MATCH_PROFILE = os.environ.get(
    "LED7_S7_TIMING_MATCHED_SVI_MATCH_PROFILE", "strict_equal"
)
MATCH_PROFILE_CONFIGS = {
    "strict_equal": {
        "sample_sizes": {0: 5_500, 1: 5_500},
        "max_ks": 0.015,
        "max_wasserstein_ms": 10.0,
        "selection_hashes": {
            0: "475dbc17f6230a311756c1f9693cf295358f3d323d80710e5ed766a3ecf9592d",
            1: "57eec5c5f2de1c8f195a62c9e729138dbaf8fc5c1bece572be8a96f1ffaa908b",
        },
        "likelihood_counts": {
            0: {"abort": 740, "censored": 4_553},
            1: {"abort": 1_120, "censored": 3_897},
        },
    },
    "relaxed_unequal": {
        "sample_sizes": {0: 13_750, 1: 7_000},
        "max_ks": 0.030,
        "max_wasserstein_ms": 10.0,
        "selection_hashes": {
            0: "0524b9c51119f79fb90f2160972a7e85b56d9cf40928b237ab433e637d7f0371",
            1: "8e40159aec2cfa01cb4eb336a9e18a64a9222af0236ebd89247aabc66b4b4077",
        },
        "likelihood_counts": {
            0: {"abort": 1_914, "censored": 11_324},
            1: {"abort": 1_416, "censored": 4_987},
        },
    },
}
if MATCH_PROFILE not in MATCH_PROFILE_CONFIGS:
    raise ValueError(
        f"Unknown matching profile {MATCH_PROFILE!r}; expected one of "
        f"{tuple(MATCH_PROFILE_CONFIGS)}."
    )
MATCH_CONFIG = MATCH_PROFILE_CONFIGS[MATCH_PROFILE]
MATCH_SAMPLE_SIZES = MATCH_CONFIG["sample_sizes"]
MAX_KS_DISTANCE = MATCH_CONFIG["max_ks"]
MAX_WASSERSTEIN_MS = MATCH_CONFIG["max_wasserstein_ms"]
EXPECTED_SELECTION_HASHES = MATCH_CONFIG["selection_hashes"]

if MATCH_PROFILE == "strict_equal":
    default_output_name = (
        "numpyro_svi_led7_s7_all_vs_s9_timing_matched_"
        "no_trunc_exp_lapse_outputs"
    )
else:
    default_output_name = (
        "numpyro_svi_led7_s7_relaxed_s9_timing_matched_"
        "no_trunc_exp_lapse_outputs"
    )
OUTPUT_ROOT = Path(
    os.environ.get(
        "LED7_S7_TIMING_MATCHED_SVI_OUTPUT_ROOT",
        str(SCRIPT_DIR / default_output_name),
    )
).expanduser()

FIT_LABELS = tuple(
    label.strip()
    for label in os.environ.get(
        "LED7_S7_TIMING_MATCHED_SVI_FITS",
        "s7_all,s7_matched_to_s9",
    ).split(",")
    if label.strip()
)
ALLOWED_FIT_LABELS = ("s7_all", "s7_matched_to_s9")

ANIMALS = (90, 92, 93, 98, 99, 100, 102, 103)
TRAINING_LEVEL = 16
SESSION_TYPE = 7
REFERENCE_SESSION_TYPE = 9
ALLOWED_REPEAT_TRIALS = {0, 2}
LED_TRIAL_VALUES = (0, 1)

MATCH_RANDOM_SEED = 20_260_909
MATCHING_COLUMNS = ["intended_fix", "effective_scheduled_onset"]
MATCH_BIN_WIDTH_S = 0.020
MATCH_INTENDED_BINS = (
    np.arange(200, 2200 + 20, 20, dtype=float) / 1000
)
MATCH_ONSET_BINS = (
    np.arange(100, 1100 + 20, 20, dtype=float) / 1000
)
EXPECTED_PRE_LIKELIHOOD_COUNTS = {
    "s7_all": {0: 92_057, 1: 46_383},
    "s7_matched_to_s9": dict(MATCH_SAMPLE_SIZES),
}
EXPECTED_LIKELIHOOD_COUNTS = {
    "s7_all": {
        0: {"abort": 14_215, "censored": 74_199},
        1: {"abort": 8_276, "censored": 34_209},
    },
    "s7_matched_to_s9": MATCH_CONFIG["likelihood_counts"],
}
EXPECTED_MISSING_EVENT3_TIMING = {
    "s7_all": {0: 3, 1: 0},
    "s7_matched_to_s9": {0: 3, 1: 0},
}

QUADRATURE_NODES = int(
    os.environ.get("LED7_S7_TIMING_MATCHED_SVI_QUADRATURE_NODES", "64")
)
MAIN_STEPS = int(
    os.environ.get("LED7_S7_TIMING_MATCHED_SVI_MAIN_STEPS", "150000")
)
EXTENDED_MAX_STEPS = int(
    os.environ.get("LED7_S7_TIMING_MATCHED_SVI_EXTENDED_MAX_STEPS", "250000")
)
SVI_CHECK_EVERY = int(
    os.environ.get("LED7_S7_TIMING_MATCHED_SVI_CHECK_EVERY", "1000")
)
SVI_MIN_STEPS = int(
    os.environ.get("LED7_S7_TIMING_MATCHED_SVI_MIN_STEPS", "150000")
)
SVI_MIN_IMPROVEMENT_REL = float(
    os.environ.get("LED7_S7_TIMING_MATCHED_SVI_MIN_IMPROVEMENT_REL", "0.001")
)
SVI_NO_IMPROVE_PATIENCE_WINDOWS = int(
    os.environ.get("LED7_S7_TIMING_MATCHED_SVI_PATIENCE_WINDOWS", "12")
)
SVI_REL_TOL = float(
    os.environ.get("LED7_S7_TIMING_MATCHED_SVI_REL_TOL", "0.001")
)
LEARNING_RATE = float(
    os.environ.get("LED7_S7_TIMING_MATCHED_SVI_LR", "0.0002")
)
CLIP_NORM = float(
    os.environ.get("LED7_S7_TIMING_MATCHED_SVI_CLIP_NORM", "1.0")
)
GUIDE_INIT_SCALE = float(
    os.environ.get("LED7_S7_TIMING_MATCHED_SVI_GUIDE_INIT_SCALE", "0.1")
)
POSTERIOR_N_SAMPLES = int(
    os.environ.get("LED7_S7_TIMING_MATCHED_SVI_POSTERIOR_SAMPLES", "10000")
)
RNG_SEED = int(os.environ.get("LED7_S7_TIMING_MATCHED_SVI_SEED", "0"))
VALIDATE_ONLY = os.environ.get(
    "LED7_S7_TIMING_MATCHED_SVI_VALIDATE_ONLY", "0"
) == "1"


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
from scipy.stats import ks_2samp, wasserstein_distance

try:
    from matio import load_from_mat
except ImportError as exc:
    raise ImportError(
        "This script needs the isolated mat-io decoder. Run it as:\n"
        "env PYTHONPATH=/tmp/codex_mat_io .venv/bin/python "
        "fitting_aborts/"
        "numpyro_svi_proactive_led_step_jump_no_trunc_exp_lapse_"
        "led7_s7_all_vs_s9_timing_matched.py"
    ) from exc

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from led7_scheduled_timing_matching_utils import (
    largest_remainder_counts,
    select_rows_matching_two_margins,
)
import numpyro_proactive_led_step_jump_no_trunc_exp_lapse_svi_utils as svi_utils


# %%
# =============================================================================
# SVI helpers
# =============================================================================
def make_optimizer():
    return numpyro.optim.ClippedAdam(LEARNING_RATE, clip_norm=CLIP_NORM)


def run_svi_with_convergence_checks(svi, rng_key, data, fit_label):
    # SVI.run defines its lax.scan body inside every call.  In a chunked fit that
    # can trigger an expensive recompilation at each convergence check.  Keep one
    # jitted full-window runner alive for the entire fit; it performs the same
    # stable_update scan while reusing the compiled executable.
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
    extended_after_150k = False
    stop_reason = "max_steps_restore_best"

    print(
        f"\n[{fit_label}] Running full-rank SVI for at least "
        f"{SVI_MIN_STEPS:,} steps with checks every {SVI_CHECK_EVERY:,} steps..."
    )
    while True:
        chunk_index = len(convergence_rows) + 1
        chunk_steps = min(SVI_CHECK_EVERY, active_max_steps - completed_steps)
        if chunk_steps <= 0:
            if active_max_steps < EXTENDED_MAX_STEPS:
                active_max_steps = EXTENDED_MAX_STEPS
                extended_after_150k = True
                print(
                    f"[{fit_label}] Best checkpoint lacks a full patience tail; "
                    f"continuing to {EXTENDED_MAX_STEPS:,} steps."
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
                "mean_loss": window_mean,
                "median_loss": window_median,
                "last_loss": window_last,
                "min_loss": window_min,
                "max_loss": window_max,
                "slope_per_1000": slope_per_1000,
                "n_nonfinite": n_nonfinite,
                "delta_from_previous_mean": delta_from_prev,
                "relative_change": relative_change,
                "stable_vs_previous": is_stable,
                "stable_window_count": stable_window_count,
                "improved_best": improved_best,
                "significant_improvement": significant_improvement,
                "relative_improvement_from_best": relative_improvement_from_best,
                "best_mean_loss_so_far": best_window_mean,
                "best_chunk_so_far": best_window_chunk,
                "best_end_step_so_far": best_window_end_step,
                "no_improve_window_count": no_improve_window_count,
                "windows_since_best": windows_since_best,
                "chunk_seconds": chunk_seconds,
                "can_stop": can_stop,
            }
        )
        print(
            f"[{fit_label}] {start_step:>6}-{end_step:<6} "
            f"mean={window_mean:.7g} best={best_window_mean:.7g} "
            f"best_step={best_window_end_step:<6} "
            f"no_improve={no_improve_window_count:>2} "
            f"since_best={windows_since_best:>2} ({chunk_seconds:.2f}s)"
        )

        if can_stop:
            stop_reason = f"patience{SVI_NO_IMPROVE_PATIENCE_WINDOWS}_restore_best"
            break

        prev_window_mean = window_mean

    losses = np.concatenate(all_losses) if all_losses else np.array([], dtype=float)
    if best_state is None:
        best_state = state
        best_params = svi.get_params(state)
    print(
        f"[{fit_label}] Restored best window: chunk {best_window_chunk}, "
        f"step {best_window_end_step}, mean negative ELBO={best_window_mean:.7g}"
    )
    return (
        SVIRunResult(best_params, best_state, jnp.asarray(losses)),
        pd.DataFrame(convergence_rows),
        stop_reason,
        extended_after_150k,
    )


def to_numpy_tree(tree):
    return jax.tree_util.tree_map(
        lambda value: np.asarray(jax.device_get(value)), tree
    )


def selected_index_hash(indices):
    payload = "\n".join(str(int(index)) for index in indices).encode()
    return hashlib.sha256(payload).hexdigest()


# %%
# =============================================================================
# Load LED7 s7 and pooled s9 timing reference
# =============================================================================
invalid_fit_labels = sorted(set(FIT_LABELS) - set(ALLOWED_FIT_LABELS))
if invalid_fit_labels or not FIT_LABELS:
    raise ValueError(
        f"Requested fit labels must be drawn from {ALLOWED_FIT_LABELS}; "
        f"found {FIT_LABELS}."
    )
if not INPUT_MAT_PATH.exists():
    raise FileNotFoundError(INPUT_MAT_PATH)

mat_contents = load_from_mat(INPUT_MAT_PATH, variable_names=[TABLE_NAME])
if TABLE_NAME not in mat_contents:
    raise KeyError(f"{TABLE_NAME!r} is absent from {INPUT_MAT_PATH}.")
raw_df = mat_contents[TABLE_NAME]
if not isinstance(raw_df, pd.DataFrame):
    raise TypeError(f"Expected a DataFrame, found {type(raw_df).__name__}.")

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

training_df = raw_df.loc[raw_df["training_level"].eq(TRAINING_LEVEL)].copy()
session_df = training_df.loc[
    training_df["session_type"].isin([SESSION_TYPE, REFERENCE_SESSION_TYPE])
].copy()
repeat_df = session_df.loc[
    session_df["repeat_trial"].isin(ALLOWED_REPEAT_TRIALS)
    | session_df["repeat_trial"].isna()
].copy()
filtered_df = repeat_df.loc[
    repeat_df["LED_trial"].isin(LED_TRIAL_VALUES)
].copy()

if sorted(filtered_df["animal"].dropna().astype(int).unique()) != list(ANIMALS):
    raise RuntimeError("Filtered source does not contain the expected eight animals.")
if not filtered_df["training_level"].eq(TRAINING_LEVEL).all():
    raise RuntimeError("Another training level survived filtering.")
if set(filtered_df["session_type"].astype(int).unique()) != {
    SESSION_TYPE,
    REFERENCE_SESSION_TYPE,
}:
    raise RuntimeError("Unexpected session types survived filtering.")
if not set(filtered_df["repeat_trial"].dropna().astype(int)).issubset(
    ALLOWED_REPEAT_TRIALS
):
    raise RuntimeError("Unexpected repeat_trial value survived filtering.")
if set(filtered_df["LED_trial"].astype(int).unique()) != set(LED_TRIAL_VALUES):
    raise RuntimeError("Expected LED_trial values 0 and 1.")

s7_df = filtered_df.loc[filtered_df["session_type"].eq(SESSION_TYPE)].copy()
s9_df = filtered_df.loc[
    filtered_df["session_type"].eq(REFERENCE_SESSION_TYPE)
].copy()
s7_df["effective_scheduled_onset"] = (
    s7_df["intended_fix"] - s7_df["LED_onset_time"]
)
s9_df["effective_scheduled_onset"] = s9_df["LED_onset_time"]

if len(s9_df) != 100_788:
    raise RuntimeError(f"Expected 100,788 reference rows, found {len(s9_df):,}.")
observed_s7_counts = {
    led: int(s7_df["LED_trial"].eq(led).sum()) for led in LED_TRIAL_VALUES
}
if observed_s7_counts != EXPECTED_PRE_LIKELIHOOD_COUNTS["s7_all"]:
    raise RuntimeError(f"Unexpected s7 LED counts: {observed_s7_counts}")

for label, frame in (("s7", s7_df), ("s9", s9_df)):
    timing = frame[["intended_fix", "effective_scheduled_onset"]]
    if not np.isfinite(timing).all(axis=None):
        raise RuntimeError(f"{label} has non-finite scheduled timing.")
    if (frame["effective_scheduled_onset"] < 0).any():
        raise RuntimeError(f"{label} has negative scheduled onset.")
    if (frame["effective_scheduled_onset"] > frame["intended_fix"] + 1e-12).any():
        raise RuntimeError(f"{label} has onset after intended fixation.")


# %%
# =============================================================================
# Reproduce exact s7 OFF/ON rows matched to pooled s9 timing marginals
# =============================================================================
reference_intended_counts, _ = np.histogram(
    s9_df["intended_fix"], bins=MATCH_INTENDED_BINS
)
reference_onset_counts, _ = np.histogram(
    s9_df["effective_scheduled_onset"], bins=MATCH_ONSET_BINS
)
if int(reference_intended_counts.sum()) != len(s9_df):
    raise RuntimeError("Reference intended_fix exceeds matching support.")
if int(reference_onset_counts.sum()) != len(s9_df):
    raise RuntimeError("Reference scheduled onset exceeds matching support.")

matching_rng = np.random.default_rng(MATCH_RANDOM_SEED)
matched_groups = {}
matching_audit = {}
for led_trial in LED_TRIAL_VALUES:
    sample_size = MATCH_SAMPLE_SIZES[led_trial]
    target_intended_counts = largest_remainder_counts(
        reference_intended_counts, sample_size
    )
    target_onset_counts = largest_remainder_counts(
        reference_onset_counts, sample_size
    )
    candidate_df = s7_df.loc[s7_df["LED_trial"].eq(led_trial)].copy()
    selected_indices, flow_audit = select_rows_matching_two_margins(
        timing_frame=candidate_df[MATCHING_COLUMNS].copy(),
        target_intended_counts=target_intended_counts,
        target_onset_counts=target_onset_counts,
        rng=matching_rng,
        matching_columns=MATCHING_COLUMNS,
        intended_bins=MATCH_INTENDED_BINS,
        onset_bins=MATCH_ONSET_BINS,
        sample_size=sample_size,
        bin_width_s=MATCH_BIN_WIDTH_S,
    )
    selection_hash = selected_index_hash(selected_indices)
    if selection_hash != EXPECTED_SELECTION_HASHES[led_trial]:
        raise RuntimeError(
            f"LED {led_trial} selected-row hash changed: {selection_hash}."
        )
    matched_df = candidate_df.loc[selected_indices].copy()
    intended_ks = float(
        ks_2samp(
            matched_df["intended_fix"],
            s9_df["intended_fix"],
            method="asymp",
        ).statistic
    )
    intended_w_ms = float(
        1000
        * wasserstein_distance(
            matched_df["intended_fix"], s9_df["intended_fix"]
        )
    )
    onset_ks = float(
        ks_2samp(
            matched_df["effective_scheduled_onset"],
            s9_df["effective_scheduled_onset"],
            method="asymp",
        ).statistic
    )
    onset_w_ms = float(
        1000
        * wasserstein_distance(
            matched_df["effective_scheduled_onset"],
            s9_df["effective_scheduled_onset"],
        )
    )
    if max(intended_ks, onset_ks) > MAX_KS_DISTANCE:
        raise RuntimeError(f"LED {led_trial} matching KS threshold failed.")
    if max(intended_w_ms, onset_w_ms) > MAX_WASSERSTEIN_MS:
        raise RuntimeError(f"LED {led_trial} matching Wasserstein threshold failed.")

    matched_groups[led_trial] = matched_df
    matching_audit[str(led_trial)] = {
        **flow_audit,
        "selected_rows": int(len(matched_df)),
        "selected_source_index_sha256": selection_hash,
        "intended_KS": intended_ks,
        "intended_W_ms": intended_w_ms,
        "onset_KS": onset_ks,
        "onset_W_ms": onset_w_ms,
    }

matched_s7_df = pd.concat(
    [matched_groups[0], matched_groups[1]], axis=0, ignore_index=False
)
if len(matched_s7_df) != sum(MATCH_SAMPLE_SIZES.values()):
    raise RuntimeError("Matched s7 table has the wrong total.")
if not matched_s7_df.index.is_unique:
    raise RuntimeError("Matched OFF/ON source rows overlap.")

pre_likelihood_frames = {
    "s7_all": s7_df.copy(),
    "s7_matched_to_s9": matched_s7_df.copy(),
}


# %%
# =============================================================================
# Build exact event-3 plus successful-censored likelihood datasets
# =============================================================================
fit_payloads = {}
for fit_label, candidate_df in pre_likelihood_frames.items():
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

    finite_timing = np.isfinite(
        manifest[
            ["timed_fix", "intended_fix", "effective_scheduled_onset"]
        ]
    ).all(axis=1)
    is_event3 = manifest["abort_event"].eq(3)
    is_success = manifest["success"].isin([-1, 1])
    if (is_event3 & is_success).any():
        raise RuntimeError(f"{fit_label} has rows that are both event and censored.")

    manifest["likelihood_role"] = "excluded_other_event_or_outcome"
    manifest.loc[~finite_timing, "likelihood_role"] = "excluded_missing_timing"
    manifest.loc[finite_timing & is_success, "likelihood_role"] = "censored"
    manifest.loc[finite_timing & is_event3, "likelihood_role"] = "abort"

    fit_df = manifest.loc[
        manifest["likelihood_role"].isin(["abort", "censored"])
    ].copy()
    abort_df = fit_df.loc[fit_df["likelihood_role"].eq("abort")].copy()
    censored_df = fit_df.loc[fit_df["likelihood_role"].eq("censored")].copy()
    if not (abort_df["timed_fix"] < abort_df["intended_fix"]).all():
        raise RuntimeError(f"{fit_label} has event-3 time at/after stimulus.")
    if not (censored_df["timed_fix"] >= censored_df["intended_fix"]).all():
        raise RuntimeError(f"{fit_label} has censored time before stimulus.")

    observed_pre_counts = {
        led: int(manifest["LED_trial"].eq(led).sum()) for led in LED_TRIAL_VALUES
    }
    if observed_pre_counts != EXPECTED_PRE_LIKELIHOOD_COUNTS[fit_label]:
        raise RuntimeError(
            f"{fit_label} pre-likelihood counts changed: {observed_pre_counts}"
        )

    observed_likelihood_counts = {}
    observed_missing_event3 = {}
    for led_trial in LED_TRIAL_VALUES:
        led_fit = fit_df.loc[fit_df["LED_trial"].eq(led_trial)]
        observed_likelihood_counts[led_trial] = {
            "abort": int(led_fit["likelihood_role"].eq("abort").sum()),
            "censored": int(led_fit["likelihood_role"].eq("censored").sum()),
        }
        observed_missing_event3[led_trial] = int(
            (
                manifest["LED_trial"].eq(led_trial)
                & manifest["abort_event"].eq(3)
                & ~np.isfinite(manifest["timed_fix"])
            ).sum()
        )
    if observed_likelihood_counts != EXPECTED_LIKELIHOOD_COUNTS[fit_label]:
        raise RuntimeError(
            f"{fit_label} likelihood counts changed: {observed_likelihood_counts}"
        )
    if observed_missing_event3 != EXPECTED_MISSING_EVENT3_TIMING[fit_label]:
        raise RuntimeError(
            f"{fit_label} missing-event timing changed: {observed_missing_event3}"
        )

    category_frames = {
        led_trial: fit_df.loc[fit_df["LED_trial"].eq(led_trial)].copy()
        for led_trial in LED_TRIAL_VALUES
    }
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
        "on_abort_t": jnp.asarray(on_abort_df["timed_fix"].to_numpy(dtype=float)),
        "on_abort_t_led": jnp.asarray(
            on_abort_df["effective_scheduled_onset"].to_numpy(dtype=float)
        ),
        "on_censored_t_stim": jnp.asarray(
            on_censored_df["intended_fix"].to_numpy(dtype=float)
        ),
        "on_censored_t_led": jnp.asarray(
            on_censored_df["effective_scheduled_onset"].to_numpy(dtype=float)
        ),
        "off_abort_t": jnp.asarray(
            off_abort_df["timed_fix"].to_numpy(dtype=float)
        ),
        "off_censored_t_stim": jnp.asarray(
            off_censored_df["intended_fix"].to_numpy(dtype=float)
        ),
    }
    counts = {
        "pre_likelihood_total": int(len(manifest)),
        "likelihood_total": int(len(fit_df)),
        "by_led": {
            str(led): {
                "pre_likelihood": observed_pre_counts[led],
                **observed_likelihood_counts[led],
                "likelihood_total": int(
                    sum(observed_likelihood_counts[led].values())
                ),
                "missing_event3_timing": observed_missing_event3[led],
                "early_aborts_below_300ms_retained": int(
                    (
                        category_frames[led]["likelihood_role"].eq("abort")
                        & (category_frames[led]["timed_fix"] < 0.3)
                    ).sum()
                ),
            }
            for led in LED_TRIAL_VALUES
        },
        "by_animal": {
            str(animal): int(fit_df["animal"].eq(animal).sum())
            for animal in ANIMALS
        },
    }
    fit_payloads[fit_label] = {
        "manifest": manifest,
        "fit_df": fit_df,
        "category_frames": category_frames,
        "jax_data": jax_data,
        "counts": counts,
    }

print(f"Source: {INPUT_MAT_PATH.resolve()} :: {TABLE_NAME}")
print(f"Output root: {OUTPUT_ROOT.resolve()}")
print(f"Fits requested: {FIT_LABELS}")
print(f"Matching profile: {MATCH_PROFILE} ({MATCH_SAMPLE_SIZES})")
print(f"Animals: {ANIMALS} (trial pooled)")
print("No truncation; all LED_trial == 1 rows; event 3 plus successful censoring")
print("\nMatching audit:")
print(pd.DataFrame(matching_audit).T.to_string())
print("\nLikelihood counts:")
for fit_label, payload in fit_payloads.items():
    print(f"  {fit_label}: {json.dumps(payload['counts'], sort_keys=True)}")


# %%
# =============================================================================
# Independently fit each requested dataset and save complete artifacts
# =============================================================================
for fit_label in FIT_LABELS:
    payload = fit_payloads[fit_label]
    jax_data = payload["jax_data"]
    fit_dir = OUTPUT_ROOT / fit_label
    fit_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = fit_dir / "source_row_manifest.csv"
    payload["manifest"].to_csv(manifest_path, index=False)
    matching_audit_path = fit_dir / "matching_audit.json"
    matching_audit_path.write_text(
        json.dumps(
            {
                "reference_session_type": REFERENCE_SESSION_TYPE,
                "matching_profile": MATCH_PROFILE,
                "match_sample_sizes_by_led": {
                    str(key): int(value)
                    for key, value in MATCH_SAMPLE_SIZES.items()
                },
                "max_ks_distance": MAX_KS_DISTANCE,
                "max_wasserstein_ms": MAX_WASSERSTEIN_MS,
                "matching_seed": MATCH_RANDOM_SEED,
                "matching_columns": MATCHING_COLUMNS,
                "matching_bin_width_s": MATCH_BIN_WIDTH_S,
                "group_audit": matching_audit,
                "applies_to_fit": fit_label == "s7_matched_to_s9",
            },
            indent=2,
        )
        + "\n"
    )

    init_values = svi_utils.clip_init_to_hard_bounds(
        dict(svi_utils.DEFAULT_INIT_VALUES)
    )
    model = lambda data: svi_utils.proactive_led_step_jump_no_trunc_lapse_model(
        data, n_quad=QUADRATURE_NODES
    )

    def log_joint_from_values(values):
        log_joint, _ = log_density(model, (jax_data,), {}, values)
        return log_joint

    initial_log_joint = log_joint_from_values(init_values)
    initial_grad = jax.grad(log_joint_from_values)(init_values)
    print(f"\n[{fit_label}] Initial values:")
    for name in svi_utils.PARAM_NAMES:
        print(f"  {name:<24} {init_values[name]:.8g}")
    print(f"[{fit_label}] Initial log joint: {float(initial_log_joint):.8g}")
    print(
        f"[{fit_label}] Initial gradients finite: "
        f"{svi_utils.tree_all_finite(initial_grad)}"
    )
    if not np.isfinite(float(initial_log_joint)) or not svi_utils.tree_all_finite(
        initial_grad
    ):
        raise RuntimeError(f"{fit_label}: non-finite initial log joint/gradient.")
    if VALIDATE_ONLY:
        print(f"[{fit_label}] Validation-only check passed; fit not started.")
        continue

    guide = svi_utils.make_fullrank_guide(
        model, init_values, init_scale=GUIDE_INIT_SCALE
    )
    svi = SVI(model, guide, make_optimizer(), Trace_ELBO())
    fit_start = perf_counter()
    fit_result, convergence_df, stop_reason, extended_after_150k = (
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
        name: np.asarray(jax.device_get(values))
        for name, values in posterior_samples.items()
    }
    guide_params_np = to_numpy_tree(fit_result.params)

    summary_rows = []
    all_posterior_finite = True
    posterior_means = {}
    for name in svi_utils.PARAM_NAMES:
        values = np.asarray(posterior_np[name], dtype=float).reshape(-1)
        finite = np.isfinite(values)
        all_posterior_finite = all_posterior_finite and bool(finite.all())
        finite_values = values[finite]
        bounds = svi_utils.PARAM_BOUNDS[name]
        posterior_means[name] = float(np.mean(finite_values))
        summary_rows.append(
            {
                "dataset": fit_label,
                "parameter": name,
                "mean": posterior_means[name],
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
    if not all_posterior_finite:
        raise RuntimeError(f"{fit_label}: posterior has non-finite samples.")
    for row in posterior_summary_df.itertuples(index=False):
        if row.q025 < row.hard_low or row.q975 > row.hard_high:
            raise RuntimeError(f"{fit_label}: {row.parameter} interval exceeds bounds.")

    posterior_mean_log_likelihood = float(
        svi_utils.proactive_led_step_jump_no_trunc_lapse_loglike_jax(
            posterior_means, jax_data, n_quad=QUADRATURE_NODES
        )
    )
    best_step = int(convergence_df.iloc[-1]["best_end_step_so_far"])
    checked_step = int(convergence_df.iloc[-1]["end_step"])
    n_nonfinite_losses = int(convergence_df["n_nonfinite"].sum())
    if n_nonfinite_losses:
        status = "failed_validation"
    elif stop_reason == "max_steps_restore_best":
        status = "max_steps_restore_best"
    else:
        status = "complete"

    sample_npz = fit_dir / "main_fullrank_posterior_samples.npz"
    guide_params_pkl = fit_dir / "main_fullrank_guide_params.pkl"
    posterior_summary_csv = fit_dir / "main_fullrank_posterior_summary.csv"
    loss_csv = fit_dir / "main_fullrank_loss.csv"
    convergence_csv = fit_dir / "main_fullrank_convergence_checks.csv"
    loss_png = fit_dir / "main_fullrank_loss.png"
    bundle_pkl = fit_dir / "main_fullrank_variational_posterior_bundle.pkl"
    run_summary_json = fit_dir / "run_summary.json"

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
    ax.set_title(f"LED7 {fit_label}: no-truncation lapse SVI")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(loss_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    config = {
        "model_name": "proactive_led_step_jump_no_trunc_exp_lapse_svi",
        "dataset": fit_label,
        "source_mat": str(INPUT_MAT_PATH.resolve()),
        "source_table": TABLE_NAME,
        "animals": list(ANIMALS),
        "training_level": TRAINING_LEVEL,
        "session_type": SESSION_TYPE,
        "repeat_trial": [0, 2, "NaN"],
        "led_trial": [0, 1],
        "pooling": "trial_weighted",
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
        "matching_profile": MATCH_PROFILE,
        "matching_seed": MATCH_RANDOM_SEED,
        "match_sample_sizes_by_led": {
            str(key): int(value) for key, value in MATCH_SAMPLE_SIZES.items()
        },
        "matching_max_ks_distance": MAX_KS_DISTANCE,
        "matching_max_wasserstein_ms": MAX_WASSERSTEIN_MS,
        "led_on_weight": 1.0,
        "led_off_weight": 1.0,
        "led_on_scope": "all LED_trial == 1 rows",
        "lapse_process": "exponential in fixation time",
        "likelihood_population": (
            "finite abort_event == 3 plus finite success in {-1,+1} censored "
            "at intended_fix"
        ),
    }
    bundle = {
        "schema_version": 1,
        "config": config,
        "trial_counts": payload["counts"],
        "matching_audit": matching_audit,
        "init_values": init_values,
        "initial_log_joint": float(initial_log_joint),
        "posterior_mean_log_likelihood": posterior_mean_log_likelihood,
        "guide_params": guide_params_np,
        "posterior_samples": posterior_np,
        "posterior_summary": posterior_summary_df,
        "loss_trace": loss_df,
        "convergence_checks": convergence_df,
        "stop_reason": stop_reason,
        "fit_elapsed_seconds": fit_elapsed_seconds,
        "input_path": str(INPUT_MAT_PATH.resolve()),
        "source_row_manifest": str(manifest_path.resolve()),
    }
    with bundle_pkl.open("wb") as handle:
        pickle.dump(bundle, handle)

    run_summary = {
        "status": status,
        "stop_reason": stop_reason,
        "dataset": fit_label,
        "animals": list(ANIMALS),
        "fit_elapsed_seconds": fit_elapsed_seconds,
        "fit_elapsed_minutes": fit_elapsed_seconds / 60.0,
        "completed_steps": checked_step,
        "restored_best_step": best_step,
        "extended_after_150k": extended_after_150k,
        "n_nonfinite_losses": n_nonfinite_losses,
        "all_posterior_samples_finite": all_posterior_finite,
        "initial_log_joint": float(initial_log_joint),
        "posterior_mean_log_likelihood": posterior_mean_log_likelihood,
        "trial_counts": payload["counts"],
        "matching_audit": matching_audit,
        "config": config,
        "artifacts": {
            "posterior_samples_npz": str(sample_npz.resolve()),
            "guide_params_pkl": str(guide_params_pkl.resolve()),
            "posterior_summary_csv": str(posterior_summary_csv.resolve()),
            "loss_csv": str(loss_csv.resolve()),
            "convergence_csv": str(convergence_csv.resolve()),
            "loss_png": str(loss_png.resolve()),
            "variational_posterior_bundle_pkl": str(bundle_pkl.resolve()),
            "source_row_manifest_csv": str(manifest_path.resolve()),
            "matching_audit_json": str(matching_audit_path.resolve()),
        },
    }
    run_summary_json.write_text(json.dumps(run_summary, indent=2) + "\n")

    print(f"\n[{fit_label}] Posterior summary:")
    print(posterior_summary_df.to_string(index=False))
    print(f"[{fit_label}] Fit elapsed: {fit_elapsed_seconds / 60.0:.2f} minutes")
    print(f"[{fit_label}] Restored best step: {best_step:,}")
    print(f"[{fit_label}] Final checked step: {checked_step:,}")
    print(f"[{fit_label}] Stop reason: {stop_reason}")
    print(f"[{fit_label}] Run status: {status}")
    print(f"[{fit_label}] Posterior-mean log likelihood: {posterior_mean_log_likelihood:.6f}")
    print(f"[{fit_label}] Run summary: {run_summary_json.resolve()}")

    if status == "failed_validation":
        raise RuntimeError(f"{fit_label}: SVI failed finite-value validation.")

if VALIDATE_ONLY:
    print("\nAll requested validation-only checks passed.")
