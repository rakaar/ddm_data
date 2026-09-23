# %%
"""Compare pooled LED7 session-type-7/8/9 proactive+lapse SVI fits."""

# %%
# =============================================================================
# Editable paths and numerical/plotting settings
# =============================================================================
from pathlib import Path
import hashlib
import json
import os
import pickle
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
SOURCE_MAT = REPO_DIR / "raw_data" / "outMatrix_LED7_latest.mat"
EXPECTED_SOURCE_SHA256 = (
    "9844c0e940db52f68365612f7baf934f5fbce6d7e96f5ba3570bc73ae593941b"
)
S7_FIT_DIR = (
    SCRIPT_DIR
    / "numpyro_svi_led7_s7_all_vs_s9_timing_matched_no_trunc_exp_lapse_outputs"
    / "s7_all"
)
NEW_FIT_ROOT = (
    SCRIPT_DIR / "numpyro_svi_led7_s8_s9_aggregate_no_trunc_exp_lapse_outputs"
)
OUTPUT_DIR = NEW_FIT_ROOT / "summary_figures"

CONVERGENCE_FIGURE_PATH = (
    OUTPUT_DIR / "led7_s7_s8_s9_aggregate_svi_convergence_1x3.png"
)
ONSET_FIGURE_PATH = (
    OUTPUT_DIR
    / "led7_s7_s8_s9_aggregate_svi_abort_rate_scheduled_onset_2x3.png"
)
PARAMETER_FIGURE_PATH = (
    OUTPUT_DIR / "led7_s7_s8_s9_aggregate_svi_parameters.png"
)
CONVERGENCE_AUDIT_PATH = OUTPUT_DIR / "convergence_comparison.csv"
PARAMETER_AUDIT_PATH = OUTPUT_DIR / "posterior_parameter_comparison.csv"
MODEL_FIT_AUDIT_PATH = OUTPUT_DIR / "theory_data_model_fit_audit.csv"
PAYLOAD_PATH = OUTPUT_DIR / "theory_data_diagnostic_payload.pkl"

ANIMALS = (90, 92, 93, 98, 99, 100, 102, 103)
FIT_SPECS = (
    {
        "session_type": 7,
        "label": "Session type 7",
        "short_label": "s7",
        "color": "#4C78A8",
        "fit_dir": S7_FIT_DIR,
        "expected": {
            0: {"pre": 92_057, "abort": 14_215, "censored": 74_199, "missing": 3},
            1: {"pre": 46_383, "abort": 8_276, "censored": 34_209, "missing": 0},
        },
    },
    {
        "session_type": 8,
        "label": "Session type 8",
        "short_label": "s8",
        "color": "#F58518",
        "fit_dir": NEW_FIT_ROOT / "session_type_8",
        "expected": {
            0: {"pre": 92_784, "abort": 12_570, "censored": 77_310, "missing": 0},
            1: {"pre": 55_644, "abort": 10_354, "censored": 39_313, "missing": 2},
        },
    },
    {
        "session_type": 9,
        "label": "Session type 9",
        "short_label": "s9",
        "color": "#54A24B",
        "fit_dir": NEW_FIT_ROOT / "session_type_9",
        "expected": {
            0: {"pre": 61_408, "abort": 10_107, "censored": 48_599, "missing": 1},
            1: {"pre": 39_380, "abort": 8_747, "censored": 25_928, "missing": 5},
        },
    },
)
LED_SPECS = (
    {"value": 0, "label": "LED OFF", "color": "#1F77B4"},
    {"value": 1, "label": "LED ON", "color": "#D62728"},
)

QUADRATURE_NODES = 64
MODEL_DT_S = 0.001
FULL_ONSET_RANGE_S = (-2.2, 2.2)
FIXATION_RANGE_S = (0.0, 2.2)
REGULAR_DISPLAY_RANGE_S = (-0.3, 0.4)
ZOOM_DISPLAY_RANGE_S = (-0.2, 0.2)
DISPLAY_RANGES_BY_SESSION = {
    7: {"regular": REGULAR_DISPLAY_RANGE_S, "zoom": ZOOM_DISPLAY_RANGE_S},
    8: {"regular": (-0.3, 0.7), "zoom": (-0.2, 0.3)},
    9: {"regular": (-0.3, 0.7), "zoom": (-0.2, 0.4)},
}
REGULAR_BIN_S = 0.010
ZOOM_BIN_S = 0.005
MODEL_TRIAL_CHUNK = 1024
MASS_TOLERANCE = 2.0e-3
OUTPUT_DPI = 250
SHOW_PLOT = False


# %%
# =============================================================================
# Imports and stable numerical helpers
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
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import numpyro_proactive_led_step_jump_no_trunc_exp_lapse_svi_utils as svi_utils


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_edges(data_range, bin_width):
    low, high = data_range
    return np.arange(low, high + 0.5 * bin_width, bin_width)


def make_jax_likelihood_data(category_frames):
    on_frame = category_frames[1]
    off_frame = category_frames[0]
    on_abort = on_frame.loc[on_frame["likelihood_role"].eq("abort")]
    on_censored = on_frame.loc[on_frame["likelihood_role"].eq("censored")]
    off_abort = off_frame.loc[off_frame["likelihood_role"].eq("abort")]
    off_censored = off_frame.loc[off_frame["likelihood_role"].eq("censored")]
    return {
        "on_abort_t": jnp.asarray(on_abort["timed_fix"].to_numpy(float)),
        "on_abort_t_led": jnp.asarray(
            on_abort["effective_scheduled_onset"].to_numpy(float)
        ),
        "on_censored_t_stim": jnp.asarray(
            on_censored["intended_fix"].to_numpy(float)
        ),
        "on_censored_t_led": jnp.asarray(
            on_censored["effective_scheduled_onset"].to_numpy(float)
        ),
        "off_abort_t": jnp.asarray(off_abort["timed_fix"].to_numpy(float)),
        "off_censored_t_stim": jnp.asarray(
            off_censored["intended_fix"].to_numpy(float)
        ),
    }


@jax.jit
def on_curve_chunk(physical_t, t_led, t_stim, params):
    risk = (physical_t >= 0.0) & (physical_t < t_stim[:, None])
    proactive = svi_utils.led_on_pdf_jax(
        physical_t,
        t_led[:, None],
        params[0],
        params[1],
        params[2],
        params[3],
        params[4],
    )
    lapse = params[6] * jnp.exp(-params[6] * jnp.maximum(physical_t, 0.0))
    return (
        jnp.sum(jnp.where(risk, proactive, 0.0), axis=0),
        jnp.sum(jnp.where(risk, lapse, 0.0), axis=0),
    )


@jax.jit
def off_curve_chunk(physical_t, t_stim, params):
    risk = (physical_t >= 0.0) & (physical_t < t_stim[:, None])
    proactive = svi_utils.led_off_pdf_jax(
        physical_t,
        params[0],
        params[2],
        params[3],
        params[4],
    )
    lapse = params[6] * jnp.exp(-params[6] * jnp.maximum(physical_t, 0.0))
    return (
        jnp.sum(jnp.where(risk, proactive, 0.0), axis=0),
        jnp.sum(jnp.where(risk, lapse, 0.0), axis=0),
    )


@jax.jit
def on_cdf_chunk(t_stim, t_led, params):
    return svi_utils.led_on_cdf_jax(
        t_stim,
        t_led,
        params[0],
        params[1],
        params[2],
        params[3],
        params[4],
        n_quad=QUADRATURE_NODES,
    )


@jax.jit
def off_cdf_chunk(t_stim, params):
    return svi_utils.led_off_cdf_jax(
        t_stim,
        params[0],
        params[2],
        params[3],
        params[4],
    )


def posterior_mean_params(posterior_samples):
    return {
        name: float(np.mean(posterior_samples[name]))
        for name in svi_utils.PARAM_NAMES
    }


def params_vector(params):
    return jnp.asarray([params[name] for name in svi_utils.PARAM_NAMES])


def analytic_abort_probability(led_value, frame, params):
    t_stim = frame["intended_fix"].to_numpy(float)
    t_led = frame["effective_scheduled_onset"].to_numpy(float)
    vector = params_vector(params)
    proactive_sum = 0.0
    for start in range(0, len(frame), MODEL_TRIAL_CHUNK):
        stop = min(start + MODEL_TRIAL_CHUNK, len(frame))
        chunk_stim = jnp.asarray(t_stim[start:stop])
        if led_value == 1:
            cdf = on_cdf_chunk(
                chunk_stim, jnp.asarray(t_led[start:stop]), vector
            )
        else:
            cdf = off_cdf_chunk(chunk_stim, vector)
        proactive_sum += float(np.asarray(jax.device_get(cdf)).sum())
    proactive_mass = (
        (1.0 - params["lapse_prob"]) * proactive_sum / len(frame)
    )
    lapse_mass = params["lapse_prob"] * float(
        np.mean(1.0 - np.exp(-params["beta_lapse"] * t_stim))
    )
    return {
        "proactive": proactive_mass,
        "lapse": lapse_mass,
        "mixture": proactive_mass + lapse_mass,
    }


def model_rate_curve(led_value, frame, params, coordinate, model_t):
    t_stim = frame["intended_fix"].to_numpy(float)
    t_led = frame["effective_scheduled_onset"].to_numpy(float)
    model_t_jax = jnp.asarray(model_t)
    vector = params_vector(params)
    proactive_sum = np.zeros_like(model_t)
    lapse_sum = np.zeros_like(model_t)
    for start in range(0, len(frame), MODEL_TRIAL_CHUNK):
        stop = min(start + MODEL_TRIAL_CHUNK, len(frame))
        chunk_stim = jnp.asarray(t_stim[start:stop])
        chunk_led = jnp.asarray(t_led[start:stop])
        if coordinate == "scheduled_onset":
            physical_t = model_t_jax[None, :] + chunk_led[:, None]
        elif coordinate == "fixation":
            physical_t = jnp.broadcast_to(
                model_t_jax[None, :], (stop - start, len(model_t_jax))
            )
        else:
            raise ValueError(f"Unknown coordinate: {coordinate}")
        if led_value == 1:
            proactive_chunk, lapse_chunk = on_curve_chunk(
                physical_t, chunk_led, chunk_stim, vector
            )
        else:
            proactive_chunk, lapse_chunk = off_curve_chunk(
                physical_t, chunk_stim, vector
            )
        proactive_sum += np.asarray(jax.device_get(proactive_chunk))
        lapse_sum += np.asarray(jax.device_get(lapse_chunk))
    proactive_rate = (
        (1.0 - params["lapse_prob"]) * proactive_sum / len(frame)
    )
    lapse_rate = params["lapse_prob"] * lapse_sum / len(frame)
    mixture_rate = proactive_rate + lapse_rate
    if not all(
        np.isfinite(values).all()
        for values in (proactive_rate, lapse_rate, mixture_rate)
    ):
        raise RuntimeError(f"Non-finite {coordinate} theory curve.")
    if not np.allclose(
        mixture_rate,
        proactive_rate + lapse_rate,
        atol=1e-12,
        rtol=1e-12,
    ):
        raise RuntimeError("Theory components do not sum to the mixture.")
    return {
        "model_t_s": model_t.copy(),
        "proactive_rate": proactive_rate,
        "lapse_rate": lapse_rate,
        "mixture_rate": mixture_rate,
        "proactive_mass": float(np.sum(proactive_rate) * MODEL_DT_S),
        "lapse_mass": float(np.sum(lapse_rate) * MODEL_DT_S),
        "mixture_mass": float(np.sum(mixture_rate) * MODEL_DT_S),
    }


def empirical_rate(frame, bin_width):
    abort_frame = frame.loc[frame["likelihood_role"].eq("abort")]
    abort_times = (
        abort_frame["timed_fix"]
        - abort_frame["effective_scheduled_onset"]
    ).to_numpy(float)
    edges = make_edges(FULL_ONSET_RANGE_S, bin_width)
    counts, _ = np.histogram(abort_times, bins=edges)
    if int(counts.sum()) != len(abort_times):
        raise RuntimeError(
            f"Full onset histogram contains {counts.sum()} of "
            f"{len(abort_times)} aborts."
        )
    rates = counts / (len(frame) * bin_width)
    mass = float(np.sum(rates) * bin_width)
    expected_mass = len(abort_times) / len(frame)
    if not np.isclose(mass, expected_mass, atol=1e-12, rtol=1e-12):
        raise RuntimeError("Empirical histogram area is not the abort fraction.")
    return {
        "bin_width_s": bin_width,
        "edges_s": edges,
        "centers_s": 0.5 * (edges[:-1] + edges[1:]),
        "rate": rates,
        "mass": mass,
        "n_abort": len(abort_times),
        "n_total": len(frame),
    }


def bin_model_rate(model_t, model_rate, edges):
    mass, _ = np.histogram(
        model_t, bins=edges, weights=model_rate * MODEL_DT_S
    )
    return mass / np.diff(edges)


# %%
# =============================================================================
# Load and validate completed fits and exact source-row manifests
# =============================================================================
if file_sha256(SOURCE_MAT) != EXPECTED_SOURCE_SHA256:
    raise RuntimeError("LED7 source MAT hash changed.")

fit_payloads = {}
parameter_rows = []
convergence_rows = []
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for spec in FIT_SPECS:
    session_type = spec["session_type"]
    fit_dir = spec["fit_dir"]
    run_summary = json.loads((fit_dir / "run_summary.json").read_text())
    if run_summary.get("status") != "complete":
        raise RuntimeError(
            f"s{session_type} status is {run_summary.get('status')!r}."
        )
    if run_summary.get("stop_reason") != "patience12_restore_best":
        raise RuntimeError(f"s{session_type} did not patience-converge.")
    if run_summary.get("n_nonfinite_losses") != 0:
        raise RuntimeError(f"s{session_type} has non-finite losses.")
    if not run_summary.get("all_posterior_samples_finite", False):
        raise RuntimeError(f"s{session_type} has non-finite posterior draws.")
    config = run_summary["config"]
    if config.get("session_type") != session_type:
        raise RuntimeError(f"s{session_type} config session changed.")
    if config.get("training_level") != 16:
        raise RuntimeError(f"s{session_type} training-level filter changed.")
    if config.get("repeat_trial") != [0, 2, "NaN"]:
        raise RuntimeError(f"s{session_type} repeat-trial filter changed.")
    if config.get("led_trial") != [0, 1]:
        raise RuntimeError(f"s{session_type} LED-trial filter changed.")
    if config.get("pooling") != "trial_weighted" or config.get("truncation") != "none":
        raise RuntimeError(f"s{session_type} pooling/truncation changed.")
    if Path(config.get("source_mat", "")).resolve() != SOURCE_MAT.resolve():
        raise RuntimeError(f"s{session_type} source path changed.")
    if int(config.get("quadrature_nodes", -1)) != QUADRATURE_NODES:
        raise RuntimeError(f"s{session_type} quadrature count changed.")
    expected_config = {
        "min_steps": 150_000,
        "check_every": 1_000,
        "min_improvement_rel": 0.001,
        "patience_windows": 12,
        "relative_stability_tolerance": 0.001,
        "learning_rate": 0.0002,
        "clip_norm": 1.0,
        "guide": "AutoMultivariateNormal",
        "guide_init_scale": 0.1,
        "posterior_n_samples": 10_000,
        "rng_seed": 0,
        "led_on_weight": 1.0,
        "led_off_weight": 1.0,
        "led_on_scope": "all LED_trial == 1 rows",
        "lapse_process": "exponential in fixation time",
    }
    for key, expected_value in expected_config.items():
        if config.get(key) != expected_value:
            raise RuntimeError(
                f"s{session_type} config {key!r} changed: {config.get(key)!r}."
            )
    if tuple(config.get("animals", [])) != ANIMALS:
        raise RuntimeError(f"s{session_type} animal set changed.")
    if session_type in (8, 9) and config.get("source_sha256") != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(f"s{session_type} source identity changed.")
    expected_onset = (
        "LED_onset_time"
        if session_type == 9
        else "intended_fix - LED_onset_time"
    )
    if session_type in (8, 9) and config.get("scheduled_onset_definition") != expected_onset:
        raise RuntimeError(f"s{session_type} onset definition changed.")

    with np.load(fit_dir / "main_fullrank_posterior_samples.npz") as saved:
        posterior_samples = {
            name: np.asarray(saved[name], dtype=float)
            for name in svi_utils.PARAM_NAMES
        }
    if any(len(values) != 10_000 for values in posterior_samples.values()):
        raise RuntimeError(f"s{session_type} posterior draw count changed.")
    if not all(np.isfinite(values).all() for values in posterior_samples.values()):
        raise RuntimeError(f"s{session_type} posterior contains non-finite values.")
    for name, values in posterior_samples.items():
        low, high = svi_utils.PARAM_BOUNDS[name]["hard"]
        if not ((values >= low) & (values <= high)).all():
            raise RuntimeError(f"s{session_type} {name} exceeds hard bounds.")
    posterior_samples["del_a_plus_del_m"] = (
        posterior_samples["del_a_minus_del_LED"]
        + posterior_samples["del_m_plus_del_LED"]
    )
    params = posterior_mean_params(posterior_samples)

    manifest = pd.read_csv(fit_dir / "source_row_manifest.csv")
    required_columns = {
        "source_row_index",
        "animal",
        "session",
        "LED_trial",
        "abort_event",
        "success",
        "timed_fix",
        "intended_fix",
        "LED_onset_time",
        "effective_scheduled_onset",
        "likelihood_role",
    }
    if not required_columns.issubset(manifest.columns):
        raise RuntimeError(f"s{session_type} manifest schema changed.")
    if not manifest["source_row_index"].is_unique:
        raise RuntimeError(f"s{session_type} manifest repeats source rows.")
    if sorted(manifest["animal"].dropna().astype(int).unique()) != list(ANIMALS):
        raise RuntimeError(f"s{session_type} manifest animal set changed.")
    if len(manifest) != sum(row["pre"] for row in spec["expected"].values()):
        raise RuntimeError(f"s{session_type} pre-likelihood count changed.")
    expected_onset_values = (
        manifest["LED_onset_time"].to_numpy(float)
        if session_type == 9
        else (
            manifest["intended_fix"] - manifest["LED_onset_time"]
        ).to_numpy(float)
    )
    if not np.allclose(
        manifest["effective_scheduled_onset"].to_numpy(float),
        expected_onset_values,
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError(f"s{session_type} manifest onset definition changed.")
    if (manifest["effective_scheduled_onset"] < 0).any() or (
        manifest["effective_scheduled_onset"] > manifest["intended_fix"] + 1e-12
    ).any():
        raise RuntimeError(f"s{session_type} invalid scheduled onset.")

    fit_df = manifest.loc[
        manifest["likelihood_role"].isin(["abort", "censored"])
    ].copy()
    abort_df = fit_df.loc[fit_df["likelihood_role"].eq("abort")]
    censored_df = fit_df.loc[fit_df["likelihood_role"].eq("censored")]
    if not abort_df["abort_event"].eq(3).all():
        raise RuntimeError(f"s{session_type} abort role contains another event.")
    if not (abort_df["timed_fix"] < abort_df["intended_fix"]).all():
        raise RuntimeError(f"s{session_type} event-3 abort at/after stimulus.")
    if not censored_df["success"].isin([-1, 1]).all():
        raise RuntimeError(f"s{session_type} censored role contains invalid success.")
    if not (censored_df["timed_fix"] >= censored_df["intended_fix"]).all():
        raise RuntimeError(f"s{session_type} censored row before stimulus.")

    category_frames = {}
    for led_value in (0, 1):
        led_manifest = manifest.loc[manifest["LED_trial"].eq(led_value)]
        led_fit = fit_df.loc[fit_df["LED_trial"].eq(led_value)].copy()
        observed = {
            "pre": len(led_manifest),
            "abort": int(led_fit["likelihood_role"].eq("abort").sum()),
            "censored": int(led_fit["likelihood_role"].eq("censored").sum()),
            "missing": int(
                (
                    led_manifest["abort_event"].eq(3)
                    & ~np.isfinite(led_manifest["timed_fix"])
                ).sum()
            ),
        }
        if observed != spec["expected"][led_value]:
            raise RuntimeError(
                f"s{session_type} LED {led_value} counts changed: {observed}."
            )
        category_frames[led_value] = led_fit

    jax_data = make_jax_likelihood_data(category_frames)
    initial_log_likelihood = float(
        svi_utils.proactive_led_step_jump_no_trunc_lapse_loglike_jax(
            svi_utils.DEFAULT_INIT_VALUES,
            jax_data,
            n_quad=QUADRATURE_NODES,
        )
    )
    posterior_log_likelihood = float(
        svi_utils.proactive_led_step_jump_no_trunc_lapse_loglike_jax(
            params, jax_data, n_quad=QUADRATURE_NODES
        )
    )
    posterior_gradient = jax.grad(
        lambda values: svi_utils.proactive_led_step_jump_no_trunc_lapse_loglike_jax(
            values, jax_data, n_quad=QUADRATURE_NODES
        )
    )(params)
    if not np.isfinite(initial_log_likelihood) or not np.isfinite(posterior_log_likelihood):
        raise RuntimeError(f"s{session_type} likelihood audit is non-finite.")
    if posterior_log_likelihood - initial_log_likelihood <= 1.0:
        raise RuntimeError(f"s{session_type} did not improve over initialization.")
    if not svi_utils.tree_all_finite(posterior_gradient):
        raise RuntimeError(f"s{session_type} posterior-mean gradient is non-finite.")

    convergence = pd.read_csv(fit_dir / "main_fullrank_convergence_checks.csv")
    if int(convergence["n_nonfinite"].sum()) != 0:
        raise RuntimeError(f"s{session_type} convergence table has non-finite loss.")
    convergence_rows.append(
        {
            "session_type": session_type,
            "status": run_summary["status"],
            "stop_reason": run_summary["stop_reason"],
            "restored_best_step": int(run_summary["restored_best_step"]),
            "final_checked_step": int(run_summary["completed_steps"]),
            "best_window_mean_negative_elbo": float(convergence["best_mean_loss_so_far"].iloc[-1]),
            "final_window_mean_negative_elbo": float(convergence["mean_loss"].iloc[-1]),
            "n_nonfinite_losses": int(run_summary["n_nonfinite_losses"]),
            "initial_log_likelihood": initial_log_likelihood,
            "posterior_mean_log_likelihood": posterior_log_likelihood,
            "log_likelihood_improvement": posterior_log_likelihood - initial_log_likelihood,
            "posterior_mean_gradient_finite": True,
            "fit_elapsed_minutes": float(run_summary["fit_elapsed_minutes"]),
        }
    )

    for name in svi_utils.PARAM_NAMES + ["del_a_plus_del_m"]:
        values = posterior_samples[name]
        if name in {
            "del_a_minus_del_LED",
            "del_m_plus_del_LED",
            "del_a_plus_del_m",
        }:
            panel = "delays"
            units = "ms"
            scale = 1000.0
        elif name == "lapse_prob":
            panel = "lapse_probability"
            units = "%"
            scale = 100.0
        elif name == "beta_lapse":
            panel = "lapse_rate"
            units = "s^-1"
            scale = 1.0
        else:
            panel = "drift_and_bound"
            units = "model units"
            scale = 1.0
        parameter_rows.append(
            {
                "session_type": session_type,
                "parameter": name,
                "panel": panel,
                "plot_units": units,
                "plot_scale": scale,
                "mean_raw": float(np.mean(values)),
                "std_raw": float(np.std(values)),
                "q025_raw": float(np.quantile(values, 0.025)),
                "median_raw": float(np.quantile(values, 0.5)),
                "q975_raw": float(np.quantile(values, 0.975)),
                "plot_mean": float(np.mean(values) * scale),
                "plot_q025": float(np.quantile(values, 0.025) * scale),
                "plot_median": float(np.quantile(values, 0.5) * scale),
                "plot_q975": float(np.quantile(values, 0.975) * scale),
                "n_samples": len(values),
            }
        )

    fit_payloads[session_type] = {
        "spec": spec,
        "run_summary": run_summary,
        "convergence": convergence,
        "posterior_samples": posterior_samples,
        "params": params,
        "manifest": manifest,
        "fit_df": fit_df,
        "category_frames": category_frames,
        "initial_log_likelihood": initial_log_likelihood,
        "posterior_log_likelihood": posterior_log_likelihood,
    }
    print(
        f"Validated completed s{session_type} fit and exact source manifest.",
        flush=True,
    )

convergence_audit = pd.DataFrame(convergence_rows)
convergence_audit.to_csv(CONVERGENCE_AUDIT_PATH, index=False)
parameter_audit = pd.DataFrame(parameter_rows)
parameter_audit.to_csv(PARAMETER_AUDIT_PATH, index=False)


# %%
# =============================================================================
# 1 x 3 convergence comparison
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.1))
for ax, spec in zip(axes, FIT_SPECS):
    session_type = spec["session_type"]
    payload = fit_payloads[session_type]
    convergence = payload["convergence"]
    run_summary = payload["run_summary"]
    ax.plot(
        convergence["end_step"] / 1000.0,
        convergence["mean_loss"],
        color=spec["color"],
        linewidth=1.35,
    )
    ax.axvline(
        run_summary["restored_best_step"] / 1000.0,
        color="#2E7D32",
        linewidth=1.15,
        label="Restored best",
    )
    ax.axvline(
        run_summary["completed_steps"] / 1000.0,
        color="0.25",
        linestyle="--",
        linewidth=1.15,
        label="Final checked",
    )
    ax.set_title(spec["label"])
    ax.set_xlabel("SVI step (thousands)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.88", alpha=0.65, linewidth=0.55)
axes[0].set_ylabel("1,000-step mean negative ELBO")
fig.legend(
    handles=[
        Line2D([0], [0], color="#2E7D32", lw=1.15, label="Restored best"),
        Line2D(
            [0],
            [0],
            color="0.25",
            lw=1.15,
            ls="--",
            label="Final checked",
        ),
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=2,
    frameon=False,
)
fig.suptitle("LED7 aggregate joint LED-OFF/ON SVI convergence", y=1.10)
fig.tight_layout()
fig.savefig(CONVERGENCE_FIGURE_PATH, dpi=OUTPUT_DPI, bbox_inches="tight")
if SHOW_PLOT:
    plt.show()
else:
    plt.close(fig)


# %%
# =============================================================================
# Four-panel posterior comparison, including derived del_a + del_m
# =============================================================================
parameter_panels = (
    {
        "parameters": ["V_A_base", "V_A_post_LED", "theta_A"],
        "labels": [r"$V_{A,base}$", r"$V_{A,postLED}$", r"$\theta_A$"],
        "ylabel": "Parameter value (model units)",
        "title": "Drift and bound",
    },
    {
        "parameters": [
            "del_a_minus_del_LED",
            "del_m_plus_del_LED",
            "del_a_plus_del_m",
        ],
        "labels": [
            r"$\delta_a-\delta_{LED}$",
            r"$\delta_m+\delta_{LED}$",
            r"$\delta_a+\delta_m$" + "\n(derived)",
        ],
        "ylabel": "Delay (ms)",
        "title": "Delay combinations",
    },
    {
        "parameters": ["lapse_prob"],
        "labels": [r"$p_{lapse}$"],
        "ylabel": "Probability (%)",
        "title": "Lapse probability",
    },
    {
        "parameters": ["beta_lapse"],
        "labels": [r"$\beta_{lapse}$"],
        "ylabel": r"Rate (s$^{-1}$)",
        "title": "Lapse rate",
    },
)
fig, axes = plt.subplots(
    1,
    4,
    figsize=(16.0, 4.7),
    gridspec_kw={"width_ratios": [2.2, 2.2, 1.0, 1.0]},
)
offsets = (-0.16, 0.0, 0.16)
for ax, panel in zip(axes, parameter_panels):
    x = np.arange(len(panel["parameters"]), dtype=float)
    for offset, spec in zip(offsets, FIT_SPECS):
        rows = (
            parameter_audit.loc[
                parameter_audit["session_type"].eq(spec["session_type"])
            ]
            .set_index("parameter")
            .loc[panel["parameters"]]
        )
        means = rows["plot_mean"].to_numpy(float)
        low = rows["plot_q025"].to_numpy(float)
        high = rows["plot_q975"].to_numpy(float)
        ax.errorbar(
            x + offset,
            means,
            yerr=np.vstack((means - low, high - means)),
            fmt="o",
            ms=5.8,
            color=spec["color"],
            ecolor=spec["color"],
            elinewidth=1.25,
            capsize=3.0,
            capthick=1.0,
            label=spec["label"],
            zorder=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(panel["labels"])
    ax.set_ylabel(panel["ylabel"])
    ax.set_title(panel["title"])
    ax.grid(axis="y", color="0.87", alpha=0.65, linewidth=0.55)
    ax.spines[["top", "right"]].set_visible(False)
axes[1].axhline(0.0, color="0.55", linestyle=":", linewidth=0.8, zorder=1)
fig.legend(
    handles=[
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=spec["color"],
            label=spec["label"],
        )
        for spec in FIT_SPECS
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=3,
    frameon=False,
)
fig.suptitle(
    "LED7 aggregate joint LED-OFF/ON proactive-lapse parameters", y=1.09
)
fig.tight_layout()
fig.savefig(PARAMETER_FIGURE_PATH, dpi=OUTPUT_DPI, bbox_inches="tight")
if SHOW_PLOT:
    plt.show()
else:
    plt.close(fig)


# %%
# =============================================================================
# Exact timing-pair-averaged theory and empirical absolute abort rates
# =============================================================================
onset_model_t = np.arange(
    FULL_ONSET_RANGE_S[0] + 0.5 * MODEL_DT_S,
    FULL_ONSET_RANGE_S[1],
    MODEL_DT_S,
)
fixation_model_t = np.arange(
    FIXATION_RANGE_S[0] + 0.5 * MODEL_DT_S,
    FIXATION_RANGE_S[1],
    MODEL_DT_S,
)

diagnostic_payloads = {}
fit_audit_rows = []
for spec in FIT_SPECS:
    session_type = spec["session_type"]
    payload = fit_payloads[session_type]
    params = payload["params"]
    diagnostic_payloads[session_type] = {}
    for led_spec in LED_SPECS:
        led_value = led_spec["value"]
        frame = payload["category_frames"][led_value]
        print(
            f"Computing exact 1-ms s{session_type} {led_spec['label']} "
            f"theory from {len(frame):,} timing pairs...",
            flush=True,
        )
        analytic = analytic_abort_probability(led_value, frame, params)
        onset_model = model_rate_curve(
            led_value,
            frame,
            params,
            "scheduled_onset",
            onset_model_t,
        )
        fixation_model = model_rate_curve(
            led_value,
            frame,
            params,
            "fixation",
            fixation_model_t,
        )
        for coordinate, model in (
            ("scheduled_onset", onset_model),
            ("fixation", fixation_model),
        ):
            if abs(model["mixture_mass"] - analytic["mixture"]) > MASS_TOLERANCE:
                raise RuntimeError(
                    f"s{session_type} LED {led_value} {coordinate} numerical "
                    "mass disagrees with analytic CDF probability."
                )
        if (
            abs(onset_model["mixture_mass"] - fixation_model["mixture_mass"])
            > MASS_TOLERANCE
        ):
            raise RuntimeError(
                f"s{session_type} LED {led_value} coordinate transform changed mass."
            )

        empirical_by_bin = {}
        for view_name, bin_width in (
            ("regular", REGULAR_BIN_S),
            ("zoom", ZOOM_BIN_S),
        ):
            empirical = empirical_rate(frame, bin_width)
            binned_model = bin_model_rate(
                onset_model["model_t_s"],
                onset_model["mixture_rate"],
                empirical["edges_s"],
            )
            widths = np.diff(empirical["edges_s"])
            iae = float(
                np.sum(np.abs(empirical["rate"] - binned_model) * widths)
            )
            rmse = float(
                np.sqrt(np.mean((empirical["rate"] - binned_model) ** 2))
            )
            empirical_by_bin[view_name] = {
                **empirical,
                "binned_model_rate": binned_model,
                "integrated_absolute_error": iae,
                "rmse_rate_per_s": rmse,
            }
            fit_audit_rows.append(
                {
                    "session_type": session_type,
                    "LED_group": led_spec["label"],
                    "view": view_name,
                    "data_bin_ms": 1000.0 * bin_width,
                    "n_total": empirical["n_total"],
                    "n_abort": empirical["n_abort"],
                    "observed_abort_fraction": empirical["mass"],
                    "analytic_model_abort_probability": analytic["mixture"],
                    "model_minus_observed_abort_fraction": analytic["mixture"] - empirical["mass"],
                    "onset_numerical_model_mass": onset_model["mixture_mass"],
                    "fixation_numerical_model_mass": fixation_model["mixture_mass"],
                    "onset_proactive_mass": onset_model["proactive_mass"],
                    "onset_lapse_mass": onset_model["lapse_mass"],
                    "integrated_absolute_error": iae,
                    "binned_rmse_rate_per_s": rmse,
                }
            )
        diagnostic_payloads[session_type][led_value] = {
            "analytic": analytic,
            "onset_model": onset_model,
            "fixation_model": fixation_model,
            "empirical": empirical_by_bin,
        }

model_fit_audit = pd.DataFrame(fit_audit_rows)
model_fit_audit.to_csv(MODEL_FIT_AUDIT_PATH, index=False)


# %%
# =============================================================================
# 2 x 3 scheduled-onset-aligned theory-versus-data comparison
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(15.4, 7.8), sharey=True)
view_specs = (
    ("regular", "10-ms data bins"),
    ("zoom", "5-ms data bins"),
)
for column, spec in enumerate(FIT_SPECS):
    session_type = spec["session_type"]
    for row, (view_name, bin_label) in enumerate(view_specs):
        ax = axes[row, column]
        x_range = DISPLAY_RANGES_BY_SESSION[session_type][view_name]
        annotation_lines = []
        for led_spec in LED_SPECS:
            led_value = led_spec["value"]
            payload = diagnostic_payloads[session_type][led_value]
            empirical = payload["empirical"][view_name]
            model = payload["onset_model"]
            ax.step(
                empirical["centers_s"],
                empirical["rate"],
                where="mid",
                color=led_spec["color"],
                linewidth=0.95,
                alpha=0.62,
            )
            ax.plot(
                model["model_t_s"],
                model["mixture_rate"],
                color=led_spec["color"],
                linewidth=2.0,
            )
            annotation_lines.append(
                f"{led_spec['label'].replace('LED ', '')}: "
                f"n={empirical['n_abort']:,}/{empirical['n_total']:,}, "
                f"obs={empirical['mass']:.3f}, "
                f"pred={payload['analytic']['mixture']:.3f}"
            )
        ax.axvline(0.0, color="0.35", linestyle=":", linewidth=0.9)
        ax.set_xlim(x_range)
        ax.set_title(spec["label"] if row == 0 else bin_label)
        ax.set_xlabel("Time from scheduled LED onset (s)")
        ax.grid(axis="y", color="0.88", alpha=0.6, linewidth=0.55)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(
            0.98,
            0.96,
            "\n".join(annotation_lines),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.7,
        )
axes[0, 0].set_ylabel("Regular view\nAbort rate (s$^{-1}$)")
axes[1, 0].set_ylabel("Zoom\nAbort rate (s$^{-1}$)")
fig.legend(
    handles=[
        Line2D([0], [0], color="#1F77B4", lw=2.0, label="LED OFF"),
        Line2D([0], [0], color="#D62728", lw=2.0, label="LED ON"),
        Line2D([0], [0], color="0.25", lw=0.95, alpha=0.62, drawstyle="steps-mid", label="Data"),
        Line2D([0], [0], color="0.25", lw=2.0, label="Proactive + lapse"),
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, 1.005),
    ncol=4,
    frameon=False,
)
fig.suptitle(
    "LED7 aggregate abort rate aligned to scheduled LED onset\n"
    "OFF zero is scheduled/counterfactual; ON zero is actual onset",
    y=1.07,
)
fig.tight_layout()
fig.savefig(ONSET_FIGURE_PATH, dpi=OUTPUT_DPI, bbox_inches="tight")
if SHOW_PLOT:
    plt.show()
else:
    plt.close(fig)


# %%
# =============================================================================
# Reusable payload and complete numerical audit
# =============================================================================
payload = {
    "schema_version": 2,
    "source_mat": str(SOURCE_MAT.resolve()),
    "source_sha256": EXPECTED_SOURCE_SHA256,
    "fit_dirs": {
        f"session_type_{spec['session_type']}": str(spec["fit_dir"].resolve())
        for spec in FIT_SPECS
    },
    "posterior_point_estimate": "posterior mean",
    "posterior_interval": "central 95% posterior credible interval",
    "model_dt_s": MODEL_DT_S,
    "full_onset_range_s": FULL_ONSET_RANGE_S,
    "display_ranges_by_session_s": DISPLAY_RANGES_BY_SESSION,
    "regular_data_bin_s": REGULAR_BIN_S,
    "zoom_data_bin_s": ZOOM_BIN_S,
    "normalization": (
        "event-3 abort count divided by all event-or-censored likelihood "
        "trials in the LED group and by bin width"
    ),
    "diagnostics": diagnostic_payloads,
}
with PAYLOAD_PATH.open("wb") as handle:
    pickle.dump(payload, handle)

print("\nConvergence audit:")
print(convergence_audit.to_string(index=False))
print("\nPosterior parameter comparison:")
print(
    parameter_audit[
        [
            "session_type",
            "parameter",
            "mean_raw",
            "q025_raw",
            "median_raw",
            "q975_raw",
        ]
    ].to_string(index=False)
)
print("\nTheory-versus-data audit:")
print(model_fit_audit.to_string(index=False))
print(f"\nConvergence figure: {CONVERGENCE_FIGURE_PATH.resolve()}")
print(f"Scheduled-onset figure: {ONSET_FIGURE_PATH.resolve()}")
print(f"Parameter figure: {PARAMETER_FIGURE_PATH.resolve()}")
print(f"Convergence CSV: {CONVERGENCE_AUDIT_PATH.resolve()}")
print(f"Parameter CSV: {PARAMETER_AUDIT_PATH.resolve()}")
print(f"Model-fit CSV: {MODEL_FIT_AUDIT_PATH.resolve()}")
print(f"Reusable payload: {PAYLOAD_PATH.resolve()}")
