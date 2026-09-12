# %%
"""Compare LED7 s7 proactive-lapse SVI fits before/after timing matching."""

# %%
# =============================================================================
# Editable paths and plotting settings
# =============================================================================
from pathlib import Path
import json
import os
import pickle
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
STRICT_FIT_ROOT = (
    SCRIPT_DIR
    / "numpyro_svi_led7_s7_all_vs_s9_timing_matched_"
    "no_trunc_exp_lapse_outputs"
)
RELAXED_FIT_ROOT = (
    SCRIPT_DIR
    / "numpyro_svi_led7_s7_relaxed_s9_timing_matched_"
    "no_trunc_exp_lapse_outputs"
)
DIAGNOSTIC_PROFILE = os.environ.get(
    "LED7_S7_TIMING_MATCHED_DIAGNOSTIC_PROFILE", "strict_equal"
)
if DIAGNOSTIC_PROFILE not in {"strict_equal", "relaxed_unequal"}:
    raise ValueError(
        "LED7_S7_TIMING_MATCHED_DIAGNOSTIC_PROFILE must be "
        "'strict_equal' or 'relaxed_unequal'."
    )

if DIAGNOSTIC_PROFILE == "strict_equal":
    OUTPUT_ROOT = STRICT_FIT_ROOT
    OUTPUT_STEM = "led7_s7_all_vs_s9_timing_matched_svi"
    MATCHED_LABEL = "Session type 7 matched to session type 9"
    MATCHED_SHORT_LABEL = "Timing-matched"
else:
    OUTPUT_ROOT = RELAXED_FIT_ROOT
    OUTPUT_STEM = "led7_s7_all_vs_relaxed_s9_timing_matched_svi"
    MATCHED_LABEL = "Session type 7 relaxed match to session type 9"
    MATCHED_SHORT_LABEL = "Relaxed timing-matched"

OUTPUT_DIR = OUTPUT_ROOT / "summary_figures"
PARAMETER_FIGURE_PATH = OUTPUT_DIR / f"{OUTPUT_STEM}_parameters.png"
FIXATION_FIGURE_PATH = OUTPUT_DIR / f"{OUTPUT_STEM}_abort_rate_fixation.png"
ONSET_FIGURE_PATH = OUTPUT_DIR / f"{OUTPUT_STEM}_abort_rate_scheduled_onset.png"
PARAMETER_SUMMARY_PATH = OUTPUT_DIR / "posterior_parameter_comparison.csv"
FIT_AUDIT_PATH = OUTPUT_DIR / "theory_data_fit_audit.csv"
PAYLOAD_PATH = OUTPUT_DIR / "theory_data_diagnostic_payload.pkl"

FIT_SPECS = (
    {
        "key": "s7_all",
        "label": "Original session type 7",
        "short_label": "Original",
        "color": "#4C78A8",
        "fit_dir": STRICT_FIT_ROOT / "s7_all",
        "expected_match_profile": None,
    },
    {
        "key": "s7_matched_to_s9",
        "label": MATCHED_LABEL,
        "short_label": MATCHED_SHORT_LABEL,
        "color": "#F58518",
        "fit_dir": (
            STRICT_FIT_ROOT / "s7_matched_to_s9"
            if DIAGNOSTIC_PROFILE == "strict_equal"
            else RELAXED_FIT_ROOT / "s7_matched_to_s9"
        ),
        "expected_match_profile": (
            None if DIAGNOSTIC_PROFILE == "strict_equal" else "relaxed_unequal"
        ),
    },
)
LED_SPECS = (
    {"value": 0, "label": "LED OFF"},
    {"value": 1, "label": "LED ON"},
)

DATA_BIN_S = 0.020
MODEL_DT_S = 0.001
FIXATION_RANGE_S = (0.0, 2.2)
ONSET_FULL_RANGE_S = (-2.2, 2.2)
ONSET_DISPLAY_RANGE_S = (-1.0, 1.0)
MODEL_TRIAL_CHUNK = 256
QUADRATURE_NODES = 64
MASS_TOLERANCE = 2.0e-3
OUTPUT_DPI = 250
SHOW_PLOT = False

DATA_COLOR = "black"
MIXTURE_COLOR = "#6A3D9A"
PROACTIVE_COLOR = "#1F77B4"
LAPSE_COLOR = "#E66101"


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
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpyro_proactive_led_step_jump_no_trunc_exp_lapse_svi_utils as svi_utils


# %%
# =============================================================================
# Load completed fits, posterior samples, and exact source-row manifests
# =============================================================================
fit_payloads = {}
parameter_frames = []

for fit_spec in FIT_SPECS:
    fit_key = fit_spec["key"]
    fit_dir = fit_spec["fit_dir"]
    run_summary_path = fit_dir / "run_summary.json"
    posterior_path = fit_dir / "main_fullrank_posterior_samples.npz"
    parameter_path = fit_dir / "main_fullrank_posterior_summary.csv"
    manifest_path = fit_dir / "source_row_manifest.csv"
    for path in (run_summary_path, posterior_path, parameter_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(path)

    run_summary = json.loads(run_summary_path.read_text())
    if run_summary.get("status") != "complete":
        raise RuntimeError(
            f"{fit_key} status is {run_summary.get('status')!r}, not complete."
        )
    config = run_summary.get("config", {})
    if config.get("truncation") != "none":
        raise RuntimeError(f"{fit_key} is not the no-truncation fit.")
    if config.get("pooling") != "trial_weighted":
        raise RuntimeError(f"{fit_key} is not trial pooled.")
    if config.get("led_on_scope") != "all LED_trial == 1 rows":
        raise RuntimeError(f"{fit_key} does not use every LED-ON trial.")
    expected_match_profile = fit_spec["expected_match_profile"]
    if (
        expected_match_profile is not None
        and config.get("matching_profile") != expected_match_profile
    ):
        raise RuntimeError(
            f"{fit_key} matching profile is "
            f"{config.get('matching_profile')!r}, not {expected_match_profile!r}."
        )
    if int(config.get("quadrature_nodes", -1)) != QUADRATURE_NODES:
        raise RuntimeError(f"{fit_key} quadrature-node count changed.")
    if run_summary.get("n_nonfinite_losses") != 0:
        raise RuntimeError(f"{fit_key} has non-finite losses.")
    if not run_summary.get("all_posterior_samples_finite", False):
        raise RuntimeError(f"{fit_key} reports non-finite posterior samples.")

    with np.load(posterior_path) as saved:
        missing = sorted(set(svi_utils.PARAM_NAMES) - set(saved.files))
        if missing:
            raise RuntimeError(f"{fit_key} posterior is missing {missing}.")
        posterior_samples = {
            name: np.asarray(saved[name], dtype=float)
            for name in svi_utils.PARAM_NAMES
        }
    if not all(np.isfinite(values).all() for values in posterior_samples.values()):
        raise RuntimeError(f"{fit_key} posterior contains non-finite values.")
    posterior_means = {
        name: float(np.mean(values)) for name, values in posterior_samples.items()
    }

    parameter_df = pd.read_csv(parameter_path)
    if parameter_df["parameter"].tolist() != svi_utils.PARAM_NAMES:
        raise RuntimeError(f"{fit_key} posterior parameter order changed.")
    if not parameter_df["dataset"].eq(fit_key).all():
        raise RuntimeError(f"{fit_key} posterior summary label changed.")
    parameter_df.insert(1, "fit_root", str(fit_dir.parent.resolve()))
    parameter_frames.append(parameter_df)

    manifest = pd.read_csv(manifest_path)
    required_manifest_columns = [
        "source_row_index",
        "animal",
        "session",
        "trial",
        "LED_trial",
        "abort_event",
        "success",
        "timed_fix",
        "intended_fix",
        "effective_scheduled_onset",
        "likelihood_role",
    ]
    missing_manifest = [
        column for column in required_manifest_columns if column not in manifest
    ]
    if missing_manifest:
        raise RuntimeError(f"{fit_key} manifest is missing {missing_manifest}.")
    if not manifest["source_row_index"].is_unique:
        raise RuntimeError(f"{fit_key} manifest repeats source rows.")

    fit_df = manifest.loc[
        manifest["likelihood_role"].isin(["abort", "censored"])
    ].copy()
    abort_df = fit_df.loc[fit_df["likelihood_role"].eq("abort")]
    censored_df = fit_df.loc[fit_df["likelihood_role"].eq("censored")]
    if not (abort_df["abort_event"].eq(3)).all():
        raise RuntimeError(f"{fit_key} abort role contains another event code.")
    if not (abort_df["timed_fix"] < abort_df["intended_fix"]).all():
        raise RuntimeError(f"{fit_key} has abort at/after intended fixation.")
    if not censored_df["success"].isin([-1, 1]).all():
        raise RuntimeError(f"{fit_key} censored role contains a non-success row.")
    if not (censored_df["timed_fix"] >= censored_df["intended_fix"]).all():
        raise RuntimeError(f"{fit_key} has censored row before intended fixation.")

    expected_counts = run_summary["trial_counts"]["by_led"]
    category_frames = {}
    for led_spec in LED_SPECS:
        led_value = led_spec["value"]
        category = fit_df.loc[fit_df["LED_trial"].eq(led_value)].copy()
        observed_abort = int(category["likelihood_role"].eq("abort").sum())
        observed_censored = int(category["likelihood_role"].eq("censored").sum())
        expected = expected_counts[str(led_value)]
        if observed_abort != int(expected["abort"]):
            raise RuntimeError(f"{fit_key} LED {led_value} abort count changed.")
        if observed_censored != int(expected["censored"]):
            raise RuntimeError(f"{fit_key} LED {led_value} censored count changed.")
        category_frames[led_value] = category

    fit_payloads[fit_key] = {
        "spec": fit_spec,
        "run_summary": run_summary,
        "posterior_samples": posterior_samples,
        "posterior_means": posterior_means,
        "manifest": manifest,
        "fit_df": fit_df,
        "category_frames": category_frames,
    }

parameter_summary = pd.concat(parameter_frames, ignore_index=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
parameter_summary.to_csv(PARAMETER_SUMMARY_PATH, index=False)


# %%
# =============================================================================
# Four-panel posterior mean and central 95% credible-interval comparison
# =============================================================================
parameter_panels = (
    {
        "parameters": ["V_A_base", "V_A_post_LED", "theta_A"],
        "labels": [r"$V_{A,base}$", r"$V_{A,postLED}$", r"$\theta_A$"],
        "scale": 1.0,
        "ylabel": "Parameter value (model units)",
        "title": "Drift and bound",
    },
    {
        "parameters": ["del_a_minus_del_LED", "del_m_plus_del_LED"],
        "labels": [r"$\delta_a-\delta_{LED}$", r"$\delta_m+\delta_{LED}$"],
        "scale": 1000.0,
        "ylabel": "Delay (ms)",
        "title": "Delay parameters",
    },
    {
        "parameters": ["lapse_prob"],
        "labels": [r"$p_{lapse}$"],
        "scale": 100.0,
        "ylabel": "Probability (%)",
        "title": "Lapse probability",
    },
    {
        "parameters": ["beta_lapse"],
        "labels": [r"$\beta_{lapse}$"],
        "scale": 1.0,
        "ylabel": r"Rate (s$^{-1}$)",
        "title": "Lapse rate",
    },
)

fig, axes = plt.subplots(
    1,
    4,
    figsize=(15.2, 4.6),
    gridspec_kw={"width_ratios": [2.2, 1.7, 1.0, 1.0]},
)
offsets = (-0.09, 0.09)
for ax, panel in zip(axes, parameter_panels):
    x = np.arange(len(panel["parameters"]), dtype=float)
    for fit_index, fit_spec in enumerate(FIT_SPECS):
        rows = (
            parameter_summary.loc[
                parameter_summary["dataset"].eq(fit_spec["key"])
            ]
            .set_index("parameter")
            .loc[panel["parameters"]]
        )
        scale = panel["scale"]
        means = rows["mean"].to_numpy(dtype=float) * scale
        low = rows["q025"].to_numpy(dtype=float) * scale
        high = rows["q975"].to_numpy(dtype=float) * scale
        ax.errorbar(
            x + offsets[fit_index],
            means,
            yerr=np.vstack((means - low, high - means)),
            fmt="o",
            ms=6.0,
            color=fit_spec["color"],
            ecolor=fit_spec["color"],
            elinewidth=1.25,
            capsize=3.0,
            capthick=1.1,
            label=fit_spec["short_label"],
            zorder=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(panel["labels"])
    ax.set_ylabel(panel["ylabel"])
    ax.set_title(panel["title"])
    ax.grid(axis="y", color="0.86", alpha=0.65, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)

axes[1].axhline(0.0, color="0.55", linestyle=":", linewidth=0.8, zorder=1)
fig.legend(
    handles=[
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=fit_spec["color"],
            label=fit_spec["short_label"],
        )
        for fit_spec in FIT_SPECS
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=2,
    frameon=False,
)
fig.suptitle(
    "LED7 session type 7: joint LED-OFF/ON proactive-lapse SVI parameters",
    y=1.10,
)
fig.tight_layout()
fig.savefig(PARAMETER_FIGURE_PATH, dpi=OUTPUT_DPI, bbox_inches="tight")
if SHOW_PLOT:
    plt.show()
else:
    plt.close(fig)


# %%
# =============================================================================
# Exact timing-averaged posterior-mean absolute abort-rate curves
# =============================================================================
fixation_model_t = np.arange(
    FIXATION_RANGE_S[0] + 0.5 * MODEL_DT_S,
    FIXATION_RANGE_S[1],
    MODEL_DT_S,
)
onset_model_t = np.arange(
    ONSET_FULL_RANGE_S[0] + 0.5 * MODEL_DT_S,
    ONSET_FULL_RANGE_S[1],
    MODEL_DT_S,
)
fixation_data_edges = np.arange(
    FIXATION_RANGE_S[0],
    FIXATION_RANGE_S[1] + 0.5 * DATA_BIN_S,
    DATA_BIN_S,
)
onset_data_edges = np.arange(
    ONSET_FULL_RANGE_S[0],
    ONSET_FULL_RANGE_S[1] + 0.5 * DATA_BIN_S,
    DATA_BIN_S,
)


def analytic_abort_probability(category, frame, params):
    t_stim = frame["intended_fix"].to_numpy(dtype=float)
    t_led = frame["effective_scheduled_onset"].to_numpy(dtype=float)
    proactive_cdf_sum = 0.0
    for start in range(0, len(frame), MODEL_TRIAL_CHUNK):
        stop = min(start + MODEL_TRIAL_CHUNK, len(frame))
        chunk_t_stim = jnp.asarray(t_stim[start:stop], dtype=jnp.float64)
        if category == 1:
            chunk_t_led = jnp.asarray(t_led[start:stop], dtype=jnp.float64)
            proactive_cdf = svi_utils.led_on_cdf_jax(
                chunk_t_stim,
                chunk_t_led,
                params["V_A_base"],
                params["V_A_post_LED"],
                params["theta_A"],
                params["del_a_minus_del_LED"],
                params["del_m_plus_del_LED"],
                n_quad=QUADRATURE_NODES,
            )
        else:
            proactive_cdf = svi_utils.led_off_cdf_jax(
                chunk_t_stim,
                params["V_A_base"],
                params["theta_A"],
                params["del_a_minus_del_LED"],
                params["del_m_plus_del_LED"],
            )
        proactive_cdf_sum += float(
            np.asarray(jax.device_get(proactive_cdf), dtype=float).sum()
        )

    lapse_cdf_sum = float(
        np.sum(1.0 - np.exp(-params["beta_lapse"] * t_stim))
    )
    proactive_mass = (
        (1.0 - params["lapse_prob"]) * proactive_cdf_sum / len(frame)
    )
    lapse_mass = params["lapse_prob"] * lapse_cdf_sum / len(frame)
    return {
        "proactive": proactive_mass,
        "lapse": lapse_mass,
        "mixture": proactive_mass + lapse_mass,
    }


def model_rate_curves(category, frame, params, coordinate, model_t):
    t_stim = frame["intended_fix"].to_numpy(dtype=float)
    t_led = frame["effective_scheduled_onset"].to_numpy(dtype=float)
    model_t_jax = jnp.asarray(model_t, dtype=jnp.float64)
    proactive_sum = np.zeros_like(model_t)
    lapse_sum = np.zeros_like(model_t)

    for start in range(0, len(frame), MODEL_TRIAL_CHUNK):
        stop = min(start + MODEL_TRIAL_CHUNK, len(frame))
        chunk_t_stim = jnp.asarray(t_stim[start:stop], dtype=jnp.float64)
        chunk_t_led = jnp.asarray(t_led[start:stop], dtype=jnp.float64)
        if coordinate == "fixation":
            physical_t = jnp.broadcast_to(
                model_t_jax[None, :], (stop - start, len(model_t_jax))
            )
        elif coordinate == "scheduled_onset":
            physical_t = model_t_jax[None, :] + chunk_t_led[:, None]
        else:
            raise ValueError(f"Unknown coordinate: {coordinate}")

        risk_mask = (physical_t >= 0.0) & (
            physical_t < chunk_t_stim[:, None]
        )
        if category == 1:
            proactive_pdf = svi_utils.led_on_pdf_jax(
                physical_t,
                chunk_t_led[:, None],
                params["V_A_base"],
                params["V_A_post_LED"],
                params["theta_A"],
                params["del_a_minus_del_LED"],
                params["del_m_plus_del_LED"],
            )
        else:
            proactive_pdf = svi_utils.led_off_pdf_jax(
                physical_t,
                params["V_A_base"],
                params["theta_A"],
                params["del_a_minus_del_LED"],
                params["del_m_plus_del_LED"],
            )
        lapse_pdf = params["beta_lapse"] * jnp.exp(
            -params["beta_lapse"] * jnp.maximum(physical_t, 0.0)
        )
        proactive_sum += np.asarray(
            jax.device_get(
                jnp.sum(jnp.where(risk_mask, proactive_pdf, 0.0), axis=0)
            ),
            dtype=float,
        )
        lapse_sum += np.asarray(
            jax.device_get(
                jnp.sum(jnp.where(risk_mask, lapse_pdf, 0.0), axis=0)
            ),
            dtype=float,
        )

    proactive_rate = (
        (1.0 - params["lapse_prob"]) * proactive_sum / len(frame)
    )
    lapse_rate = params["lapse_prob"] * lapse_sum / len(frame)
    mixture_rate = proactive_rate + lapse_rate
    arrays = (proactive_rate, lapse_rate, mixture_rate)
    if not all(np.isfinite(values).all() for values in arrays):
        raise RuntimeError(f"{coordinate} model curve is non-finite.")
    if any((values < -1e-12).any() for values in arrays):
        raise RuntimeError(f"{coordinate} model curve is negative.")
    if not np.allclose(
        mixture_rate,
        proactive_rate + lapse_rate,
        atol=1e-12,
        rtol=1e-12,
    ):
        raise RuntimeError(f"{coordinate} model components do not sum.")
    return {
        "model_t_s": model_t.copy(),
        "proactive_rate": proactive_rate,
        "lapse_rate": lapse_rate,
        "mixture_rate": mixture_rate,
        "proactive_mass": float(np.sum(proactive_rate) * MODEL_DT_S),
        "lapse_mass": float(np.sum(lapse_rate) * MODEL_DT_S),
        "mixture_mass": float(np.sum(mixture_rate) * MODEL_DT_S),
    }


def empirical_rate_payload(frame, coordinate, data_edges):
    abort_frame = frame.loc[frame["likelihood_role"].eq("abort")]
    if coordinate == "fixation":
        values = abort_frame["timed_fix"].to_numpy(dtype=float)
    elif coordinate == "scheduled_onset":
        values = (
            abort_frame["timed_fix"]
            - abort_frame["effective_scheduled_onset"]
        ).to_numpy(dtype=float)
    else:
        raise ValueError(f"Unknown coordinate: {coordinate}")
    counts, _ = np.histogram(values, bins=data_edges)
    if int(counts.sum()) != len(values):
        raise RuntimeError(
            f"{coordinate} data bins contain {counts.sum()} of {len(values)} aborts."
        )
    rate = counts.astype(float) / (len(frame) * DATA_BIN_S)
    return {
        "abort_times_s": values,
        "data_edges_s": data_edges.copy(),
        "data_centers_s": 0.5 * (data_edges[:-1] + data_edges[1:]),
        "data_rate": rate,
        "data_mass": float(np.sum(rate) * DATA_BIN_S),
        "n_abort": int(len(values)),
        "n_total": int(len(frame)),
    }


def bin_model_rate(model_t, model_rate, data_edges):
    mass, _ = np.histogram(
        model_t,
        bins=data_edges,
        weights=model_rate * MODEL_DT_S,
    )
    return mass / np.diff(data_edges)


coordinate_specs = {
    "fixation": {
        "model_t": fixation_model_t,
        "data_edges": fixation_data_edges,
        "display_range": FIXATION_RANGE_S,
    },
    "scheduled_onset": {
        "model_t": onset_model_t,
        "data_edges": onset_data_edges,
        "display_range": ONSET_DISPLAY_RANGE_S,
    },
}

diagnostic_payloads = {}
audit_rows = []
for fit_spec in FIT_SPECS:
    fit_key = fit_spec["key"]
    diagnostic_payloads[fit_key] = {}
    params = fit_payloads[fit_key]["posterior_means"]
    for led_spec in LED_SPECS:
        led_value = led_spec["value"]
        frame = fit_payloads[fit_key]["category_frames"][led_value]
        analytic_mass = analytic_abort_probability(led_value, frame, params)
        diagnostic_payloads[fit_key][led_value] = {}
        coordinate_masses = {}

        for coordinate, coordinate_spec in coordinate_specs.items():
            empirical = empirical_rate_payload(
                frame, coordinate, coordinate_spec["data_edges"]
            )
            model = model_rate_curves(
                led_value,
                frame,
                params,
                coordinate,
                coordinate_spec["model_t"],
            )
            if abs(model["mixture_mass"] - analytic_mass["mixture"]) > MASS_TOLERANCE:
                raise RuntimeError(
                    f"{fit_key} LED {led_value} {coordinate} numerical/analytic "
                    "model masses disagree."
                )
            coordinate_masses[coordinate] = model["mixture_mass"]

            binned_model_rate = bin_model_rate(
                model["model_t_s"],
                model["mixture_rate"],
                empirical["data_edges_s"],
            )
            if len(binned_model_rate) != len(empirical["data_rate"]):
                raise RuntimeError("Data/model binned curve lengths differ.")
            bin_widths = np.diff(empirical["data_edges_s"])
            integrated_absolute_error = float(
                np.sum(
                    np.abs(empirical["data_rate"] - binned_model_rate)
                    * bin_widths
                )
            )
            rmse_rate = float(
                np.sqrt(
                    np.mean((empirical["data_rate"] - binned_model_rate) ** 2)
                )
            )
            display_low, display_high = coordinate_spec["display_range"]
            data_visible = (
                (empirical["data_centers_s"] >= display_low)
                & (empirical["data_centers_s"] <= display_high)
            )
            model_visible = (
                (model["model_t_s"] >= display_low)
                & (model["model_t_s"] <= display_high)
            )
            visible_data_mass = float(
                np.sum(empirical["data_rate"][data_visible]) * DATA_BIN_S
            )
            visible_model_mass = float(
                np.sum(model["mixture_rate"][model_visible]) * MODEL_DT_S
            )

            diagnostic_payloads[fit_key][led_value][coordinate] = {
                "empirical": empirical,
                "model": model,
                "analytic_mass": dict(analytic_mass),
                "binned_model_rate": binned_model_rate,
                "integrated_absolute_error": integrated_absolute_error,
                "rmse_rate": rmse_rate,
                "visible_data_mass": visible_data_mass,
                "visible_model_mass": visible_model_mass,
            }
            audit_rows.append(
                {
                    "dataset": fit_key,
                    "LED_group": led_spec["label"],
                    "coordinate": coordinate,
                    "n_total": empirical["n_total"],
                    "n_abort": empirical["n_abort"],
                    "observed_abort_fraction": empirical["data_mass"],
                    "analytic_model_abort_probability": analytic_mass["mixture"],
                    "model_minus_observed_abort_fraction": (
                        analytic_mass["mixture"] - empirical["data_mass"]
                    ),
                    "numerical_model_mass": model["mixture_mass"],
                    "numerical_proactive_mass": model["proactive_mass"],
                    "numerical_lapse_mass": model["lapse_mass"],
                    "integrated_absolute_error": integrated_absolute_error,
                    "rmse_rate_per_s": rmse_rate,
                    "visible_data_mass": visible_data_mass,
                    "visible_model_mass": visible_model_mass,
                }
            )

        if (
            abs(
                coordinate_masses["fixation"]
                - coordinate_masses["scheduled_onset"]
            )
            > MASS_TOLERANCE
        ):
            raise RuntimeError(
                f"{fit_key} LED {led_value} coordinate transformation changed mass."
            )

fit_audit = pd.DataFrame(audit_rows)
fit_audit.to_csv(FIT_AUDIT_PATH, index=False)


# %%
# =============================================================================
# Two 2 x 2 absolute-rate theory-versus-data figures
# =============================================================================
def plot_coordinate_figure(coordinate, output_path, suptitle):
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.0), sharex=True, sharey=True)
    for row_index, fit_spec in enumerate(FIT_SPECS):
        for column_index, led_spec in enumerate(LED_SPECS):
            ax = axes[row_index, column_index]
            payload = diagnostic_payloads[fit_spec["key"]][led_spec["value"]][
                coordinate
            ]
            empirical = payload["empirical"]
            model = payload["model"]
            analytic = payload["analytic_mass"]
            ax.step(
                empirical["data_centers_s"],
                empirical["data_rate"],
                where="mid",
                color=DATA_COLOR,
                alpha=0.62,
                linewidth=1.05,
                label="Data",
            )
            ax.plot(
                model["model_t_s"],
                model["mixture_rate"],
                color=MIXTURE_COLOR,
                linewidth=2.0,
                label="Proactive + lapse",
            )
            ax.plot(
                model["model_t_s"],
                model["proactive_rate"],
                color=PROACTIVE_COLOR,
                linestyle="--",
                linewidth=1.45,
                label="Proactive",
            )
            ax.plot(
                model["model_t_s"],
                model["lapse_rate"],
                color=LAPSE_COLOR,
                linestyle="-.",
                linewidth=1.45,
                label="Lapse",
            )
            if coordinate == "scheduled_onset":
                ax.axvline(0.0, color="0.45", linestyle=":", linewidth=0.9)
                ax.set_xlim(ONSET_DISPLAY_RANGE_S)
                ax.set_xlabel("Time from scheduled LED onset (s)")
            else:
                ax.set_xlim(FIXATION_RANGE_S)
                ax.set_xlabel("Time from fixation onset (s)")
            ax.set_title(led_spec["label"])
            ax.grid(axis="y", color="0.88", alpha=0.6, linewidth=0.55)
            ax.spines[["top", "right"]].set_visible(False)
            ax.text(
                0.98,
                0.96,
                f"n = {empirical['n_abort']:,}/{empirical['n_total']:,}\n"
                f"data area = {empirical['data_mass']:.3f}\n"
                f"model area = {analytic['mixture']:.3f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8.2,
            )
        axes[row_index, 0].set_ylabel(
            f"{fit_spec['label']}\nAbort rate (s$^{{-1}}$)"
        )

    fig.legend(
        handles=[
            Line2D([0], [0], color=DATA_COLOR, lw=1.1, label="Data"),
            Line2D([0], [0], color=MIXTURE_COLOR, lw=2.0, label="Proactive + lapse"),
            Line2D(
                [0], [0], color=PROACTIVE_COLOR, lw=1.45, ls="--", label="Proactive"
            ),
            Line2D(
                [0], [0], color=LAPSE_COLOR, lw=1.45, ls="-.", label="Lapse"
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(suptitle, y=1.055)
    fig.tight_layout()
    fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight")
    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)


plot_coordinate_figure(
    "fixation",
    FIXATION_FIGURE_PATH,
    "LED7 session type 7: posterior-mean abort-rate fits from fixation onset",
)
plot_coordinate_figure(
    "scheduled_onset",
    ONSET_FIGURE_PATH,
    (
        "LED7 session type 7: posterior-mean abort-rate fits from scheduled LED onset\n"
        "OFF uses a scheduled/counterfactual onset"
    ),
)


# %%
# =============================================================================
# Reusable payload and printed audit
# =============================================================================
payload = {
    "schema_version": 1,
    "diagnostic_profile": DIAGNOSTIC_PROFILE,
    "output_root": str(OUTPUT_ROOT.resolve()),
    "fit_roots": {
        fit_spec["key"]: str(fit_spec["fit_dir"].parent.resolve())
        for fit_spec in FIT_SPECS
    },
    "posterior_point_estimate": "posterior mean",
    "posterior_interval": "central 95% credible interval",
    "data_bin_s": DATA_BIN_S,
    "model_dt_s": MODEL_DT_S,
    "fixation_range_s": FIXATION_RANGE_S,
    "onset_full_range_s": ONSET_FULL_RANGE_S,
    "onset_display_range_s": ONSET_DISPLAY_RANGE_S,
    "normalization": (
        "absolute abort rate: event-3 bin count divided by all event-or-censored "
        "likelihood trials in the LED group and by bin width"
    ),
    "fit_parameters": {
        fit_key: fit_payloads[fit_key]["posterior_means"]
        for fit_key in fit_payloads
    },
    "diagnostics": diagnostic_payloads,
}
with PAYLOAD_PATH.open("wb") as handle:
    pickle.dump(payload, handle)

print(f"Diagnostic profile: {DIAGNOSTIC_PROFILE}")
print(f"Output root: {OUTPUT_ROOT.resolve()}")
for fit_spec in FIT_SPECS:
    print(f"{fit_spec['short_label']} fit: {fit_spec['fit_dir'].resolve()}")
print("\nPosterior parameter comparison:")
print(
    parameter_summary[
        ["dataset", "parameter", "mean", "q025", "median", "q975"]
    ].to_string(index=False)
)
print("\nTheory-versus-data audit:")
print(fit_audit.to_string(index=False))
print(f"\nParameter figure: {PARAMETER_FIGURE_PATH.resolve()}")
print(f"Fixation-coordinate figure: {FIXATION_FIGURE_PATH.resolve()}")
print(f"Scheduled-onset figure: {ONSET_FIGURE_PATH.resolve()}")
print(f"Parameter CSV: {PARAMETER_SUMMARY_PATH.resolve()}")
print(f"Fit audit CSV: {FIT_AUDIT_PATH.resolve()}")
print(f"Reusable payload: {PAYLOAD_PATH.resolve()}")
