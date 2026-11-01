# %%
"""Compare lapse families; optionally add Gaussian and fixed-location Gamma rows."""

# %%
# Editable source paths and plotting settings.
from functools import partial
from pathlib import Path
import hashlib
import json
import os
import pickle

SCRIPT_DIR = Path(__file__).resolve().parent
OLD_ROOT = SCRIPT_DIR / "numpyro_svi_led7_s8_s9_aggregate_no_trunc_exp_lapse_outputs"
NEW_ROOT = SCRIPT_DIR / "numpyro_svi_led7_s8_s9_aggregate_no_trunc_uniform_lapse_0_2p2s_outputs"
GAUSSIAN_ROOT = SCRIPT_DIR / "numpyro_svi_led7_s8_s9_aggregate_no_trunc_gaussian_lapse_outputs"
GAMMA_ROOT = SCRIPT_DIR / "numpyro_svi_led7_s8_s9_aggregate_no_trunc_gamma_lapse_outputs"
INCLUDE_GAMMA = os.environ.get("LED7_S8_S9_LAPSE_COMPARISON_INCLUDE_GAMMA", "0") == "1"
INCLUDE_GAUSSIAN = INCLUDE_GAMMA or os.environ.get("LED7_S8_S9_LAPSE_COMPARISON_INCLUDE_GAUSSIAN", "0") == "1"
LAPSE_FAMILIES = (("exponential", "uniform", "gaussian", "gamma") if INCLUDE_GAMMA
                  else ("exponential", "uniform", "gaussian") if INCLUDE_GAUSSIAN
                  else ("exponential", "uniform"))
OUTPUT_ROOT = GAMMA_ROOT if INCLUDE_GAMMA else GAUSSIAN_ROOT if INCLUDE_GAUSSIAN else NEW_ROOT
OUTPUT_DIR = OUTPUT_ROOT / "summary_figures"
FIGURE_NAME = ("led7_s8_s9_exponential_uniform_gaussian_gamma_lapse_abort_rate_4x2" if INCLUDE_GAMMA
               else "led7_s8_s9_exponential_uniform_gaussian_lapse_abort_rate_3x2" if INCLUDE_GAUSSIAN
               else "led7_s8_s9_exponential_vs_uniform_lapse_abort_rate_2x2")
SESSION_TYPES = (8, 9)
DISPLAY_RANGE_S = (-0.3, 0.7)
TRIAL_CHUNK = 512
MASS_TOLERANCE = 2e-3

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
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

import numpyro_proactive_led_step_jump_no_trunc_exp_lapse_svi_utils as exponential
import numpyro_proactive_led_step_jump_no_trunc_uniform_lapse_svi_utils as uniform
import numpyro_proactive_led_step_jump_no_trunc_gaussian_lapse_svi_utils as gaussian
import numpyro_proactive_led_step_jump_no_trunc_gamma_lapse_svi_utils as gamma


# %%
@partial(jax.jit, static_argnames=("led", "family"))
def lapse_curve_chunk(grid, t_led, t_stim, params, led, family):
    physical_t = grid[None, :] + t_led[:, None]
    risk = (physical_t >= 0.0) & (physical_t < t_stim[:, None])
    if led:
        proactive = exponential.led_on_pdf_jax(physical_t, t_led[:, None], *params[:5])
    else:
        proactive = exponential.led_off_pdf_jax(physical_t, params[0], params[2], params[3], params[4])
    if family == "gamma":
        # Gamma density is unbounded at zero when shape < 1. Integrate each
        # 1 ms display bin instead of evaluating that boundary as a point.
        dt = grid[1] - grid[0]
        lower = jnp.clip(physical_t - dt / 2.0, 0.0, t_stim[:, None])
        upper = jnp.clip(physical_t + dt / 2.0, 0.0, t_stim[:, None])
        lapse = (gamma.gamma_lapse_cdf_jax(upper, params[6], params[7])
                 - gamma.gamma_lapse_cdf_jax(lower, params[6], params[7])) / dt
    elif family == "gaussian":
        lapse = jnp.where(risk, jnp.exp(gaussian.gaussian_lapse_logpdf_jax(physical_t, params[6], params[7])), 0.0)
    else:
        lapse = jnp.where(risk, jnp.exp(uniform.uniform_lapse_logpdf_jax(physical_t)), 0.0)
    return (
        (1.0 - params[5]) * jnp.sum(jnp.where(risk, proactive, 0.0), axis=0),
        params[5] * jnp.sum(lapse, axis=0),
    )


def likelihood_data(frame):
    data = {}
    for led, prefix in ((0, "off"), (1, "on")):
        group = frame.loc[frame["LED_trial"].eq(led)]
        aborts = group.loc[group["likelihood_role"].eq("abort")]
        censored = group.loc[group["likelihood_role"].eq("censored")]
        data[f"{prefix}_abort_t"] = jnp.asarray(aborts["timed_fix"].to_numpy(float))
        data[f"{prefix}_censored_t_stim"] = jnp.asarray(censored["intended_fix"].to_numpy(float))
        if led:
            data["on_abort_t_led"] = jnp.asarray(aborts["effective_scheduled_onset"].to_numpy(float))
            data["on_censored_t_led"] = jnp.asarray(censored["effective_scheduled_onset"].to_numpy(float))
    return data


# %%
# Verify historical artifacts, exact trial identities, and completed new fits.
baseline_hashes = json.loads((OUTPUT_ROOT / "baseline_artifact_hashes.json").read_text())
for filename, expected in baseline_hashes.items():
    path = Path(filename)
    if not path.is_absolute():
        path = SCRIPT_DIR.parent / path
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"Historical artifact changed: {path}")
with (OLD_ROOT / "summary_figures" / "theory_data_diagnostic_payload.pkl").open("rb") as handle:
    historical = pickle.load(handle)
if INCLUDE_GAUSSIAN:
    cache_root = GAUSSIAN_ROOT if INCLUDE_GAMMA else NEW_ROOT
    with (cache_root / "summary_figures" / "comparison_payload.pkl").open("rb") as handle:
        cached_comparison = pickle.load(handle)["diagnostics"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
diagnostics = {}
fit_rows, condition_rows, bin_rows, provenance = [], [], [], {}
parameter_rows = []
stability_rows = []
boundary_rows = []

for session_type in SESSION_TYPES:
    print(f"Checking session type {session_type}", flush=True)
    old_dir = OLD_ROOT / f"session_type_{session_type}"
    new_dir = NEW_ROOT / f"session_type_{session_type}"
    old_summary = json.loads((old_dir / "run_summary.json").read_text())
    new_summary = json.loads((new_dir / "run_summary.json").read_text())
    fit_specs = [("exponential", exponential, old_dir, old_summary),
                 ("uniform", uniform, new_dir, new_summary)]
    if INCLUDE_GAUSSIAN:
        gaussian_dir = GAUSSIAN_ROOT / f"session_type_{session_type}"
        gaussian_summary = json.loads((gaussian_dir / "run_summary.json").read_text())
        fit_specs.append(("gaussian", gaussian, gaussian_dir, gaussian_summary))
        if gaussian_summary["config"]["gaussian_support_s"] != [0.0, None]:
            raise RuntimeError("Gaussian support changed.")
    if INCLUDE_GAMMA:
        gamma_dir = GAMMA_ROOT / f"session_type_{session_type}"
        gamma_summary = json.loads((gamma_dir / "run_summary.json").read_text())
        fit_specs.append(("gamma", gamma, gamma_dir, gamma_summary))
        if gamma_summary["config"]["gamma_location_s"] != 0.0:
            raise RuntimeError("Gamma location changed.")
    for family, module, fit_dir, summary in fit_specs:
        if summary["status"] != "complete" or summary["stop_reason"] != "patience12_restore_best":
            raise RuntimeError(f"s{session_type}: fit has not patience-converged.")
        if summary["n_nonfinite_losses"] or not summary["all_posterior_samples_finite"]:
            raise RuntimeError(f"s{session_type}: nonfinite fit artifacts.")
    if new_summary["config"]["uniform_interval_s"] != [0.0, 2.2]:
        raise RuntimeError("Uniform support changed.")
    for key in ("animals", "training_level", "session_type", "repeat_trial", "led_trial",
                "scheduled_onset_definition", "pooling", "truncation", "quadrature_nodes",
                "source_sha256", "led_on_scope", "led_on_weight", "led_off_weight"):
        for family, module, fit_dir, summary in fit_specs:
            if summary["config"][key] != old_summary["config"][key]:
                raise RuntimeError(f"s{session_type} {family}: configuration differs in {key}.")
    old_manifest = pd.read_csv(old_dir / "source_row_manifest.csv")
    new_manifest = pd.read_csv(new_dir / "source_row_manifest.csv")
    pd.testing.assert_frame_equal(old_manifest, new_manifest, check_exact=False, rtol=0.0, atol=1e-12)
    if INCLUDE_GAUSSIAN:
        pd.testing.assert_frame_equal(old_manifest, pd.read_csv(gaussian_dir / "source_row_manifest.csv"),
                                      check_exact=False, rtol=0.0, atol=1e-12)
    if INCLUDE_GAMMA:
        pd.testing.assert_frame_equal(old_manifest, pd.read_csv(gamma_dir / "source_row_manifest.csv"),
                                      check_exact=False, rtol=0.0, atol=1e-12)
    frame = new_manifest.loc[new_manifest["likelihood_role"].isin(["abort", "censored"])].copy()
    diagnostics[session_type] = {family: {} for family in LAPSE_FAMILIES}
    provenance[session_type] = {"old_fit_dir": str(old_dir), "new_fit_dir": str(new_dir),
                                "identical_source_rows": True, "n_trials": len(frame)}
    if INCLUDE_GAUSSIAN:
        provenance[session_type]["gaussian_fit_dir"] = str(gaussian_dir)
    if INCLUDE_GAMMA:
        provenance[session_type]["gamma_fit_dir"] = str(gamma_dir)

    for family, module, fit_dir, summary in fit_specs:
        with np.load(fit_dir / "main_fullrank_posterior_samples.npz") as saved:
            params = {name: float(np.mean(saved[name])) for name in module.PARAM_NAMES}
            if not all(np.isfinite(saved[name]).all() for name in module.PARAM_NAMES):
                raise RuntimeError("Nonfinite posterior samples.")
            for name in module.PARAM_NAMES:
                parameter_rows.append({"session_type": session_type, "lapse_family": family,
                    "parameter": name, "mean": params[name], "q025": float(np.quantile(saved[name], 0.025)),
                    "q975": float(np.quantile(saved[name], 0.975))})
            if family in ("uniform", "gaussian", "gamma"):
                convergence = pd.read_csv(fit_dir / "main_fullrank_convergence_checks.csv")
                tail = convergence.tail(12)
                for name in module.PARAM_NAMES:
                    values = tail[f"median_{name}"].to_numpy(float)
                    posterior_sd = float(np.std(saved[name]))
                    shift = float(np.mean(values[6:]) - np.mean(values[:6]))
                    stability_rows.append({"session_type": session_type, "lapse_family": family, "parameter": name,
                        "first_tail_step": int(tail["end_step"].iloc[0]),
                        "last_tail_step": int(tail["end_step"].iloc[-1]),
                        "tail_median_range": float(np.ptp(values)), "tail_half_mean_shift": shift,
                        "posterior_sd": posterior_sd, "shift_in_posterior_sd": shift / posterior_sd})
            if family in ("gaussian", "gamma"):
                lapse_names = ("mu_lapse", "sigma_lapse") if family == "gaussian" else ("shape_lapse", "scale_lapse")
                for name in lapse_names:
                    low, high = module.PARAM_BOUNDS[name]["hard"]
                    values = saved[name]
                    mean, sd = float(np.mean(values)), float(np.std(values))
                    distance_low, distance_high = (mean - low) / sd, (high - mean) / sd
                    boundary_rows.append({"session_type": session_type, "lapse_family": family, "parameter": name,
                        "hard_low": low, "hard_high": high, "q025": float(np.quantile(values, 0.025)),
                        "q975": float(np.quantile(values, 0.975)),
                        "posterior_mean": mean, "posterior_sd": sd,
                        "distance_to_low_in_posterior_sd": distance_low,
                        "distance_to_high_in_posterior_sd": distance_high,
                        "mean_within_two_posterior_sd_of_a_bound": bool(min(distance_low, distance_high) < 2.0)})
        loglike = float(module.proactive_led_step_jump_no_trunc_lapse_loglike_jax(params, likelihood_data(frame)))
        np.testing.assert_allclose(loglike, summary["posterior_mean_log_likelihood"], atol=1e-6, rtol=0.0)
        fit_rows.append({"session_type": session_type, "lapse_family": family,
            "n_parameters": len(module.PARAM_NAMES), "n_trials": len(frame),
            "posterior_mean_log_likelihood": loglike, "mean_log_likelihood_per_trial": loglike / len(frame),
            "fit_quality_scope": "in-sample at posterior means; not held-out prediction",
            "restored_best_step": summary["restored_best_step"], "completed_steps": summary["completed_steps"],
            "lapse_prob": params["lapse_prob"], "uniform_end_s": 2.2 if family == "uniform" else np.nan})
        if family == "gaussian":
            derived = pd.read_csv(fit_dir / "main_fullrank_derived_posterior_summary.csv").set_index("parameter")
            fit_rows[-1].update(mu_lapse=params["mu_lapse"], sigma_lapse=params["sigma_lapse"],
                retained_lapse_mean_s=float(derived.loc["lapse_mean_after_zero_truncation", "mean"]),
                retained_lapse_sd_s=float(derived.loc["lapse_sd_after_zero_truncation", "mean"]))
        if family == "gamma":
            derived = pd.read_csv(fit_dir / "main_fullrank_derived_posterior_summary.csv").set_index("parameter")
            fit_rows[-1].update(shape_lapse=params["shape_lapse"], scale_lapse_s=params["scale_lapse"], gamma_location_s=0.0,
                retained_lapse_mean_s=float(derived.loc["lapse_mean", "mean"]),
                retained_lapse_sd_s=float(derived.loc["lapse_sd", "mean"]))

        for led in (0, 1):
            group = frame.loc[frame["LED_trial"].eq(led)]
            old_payload = historical["diagnostics"][session_type][led]
            empirical = old_payload["empirical"]["regular"]
            grid = old_payload["onset_model"]["model_t_s"]
            dt = float(historical["model_dt_s"])
            if INCLUDE_GAUSSIAN and family in cached_comparison[session_type]:
                cached = cached_comparison[session_type][family][led]
                curve = cached["curve"]
                predicted_fraction = cached["predicted_abort_fraction"]
                stim = group["intended_fix"].to_numpy(float)
                if family == "exponential":
                    lapse_cdf = 1.0 - np.exp(-params["beta_lapse"] * stim)
                elif family == "uniform":
                    lapse_cdf = np.minimum(stim / uniform.UNIFORM_END_S, 1.0)
                else:
                    lapse_cdf = -np.expm1(np.asarray(gaussian.gaussian_lapse_logsurvival_jax(stim, params["mu_lapse"], params["sigma_lapse"])))
                predicted_lapse_fraction = params["lapse_prob"] * float(np.mean(lapse_cdf))
            elif family == "exponential":
                curve = old_payload["onset_model"]
                predicted_fraction = old_payload["analytic"]["mixture"]
                predicted_lapse_fraction = old_payload["analytic"]["lapse"]
            else:
                totals = np.zeros((2, len(grid)))
                vector = jnp.asarray([params[name] for name in module.PARAM_NAMES])
                for start in range(0, len(group), TRIAL_CHUNK):
                    chunk = group.iloc[start:start + TRIAL_CHUNK]
                    values = lapse_curve_chunk(jnp.asarray(grid),
                        jnp.asarray(chunk["effective_scheduled_onset"].to_numpy(float)),
                        jnp.asarray(chunk["intended_fix"].to_numpy(float)), vector, led=led, family=family)
                    totals += np.asarray(jax.device_get(values))
                totals /= len(group)
                curve = {"model_t_s": grid, "proactive_rate": totals[0], "lapse_rate": totals[1],
                         "mixture_rate": totals.sum(axis=0)}
                stim = jnp.asarray(group["intended_fix"].to_numpy(float))
                if led:
                    cdf = module.led_on_cdf_jax(stim,
                        jnp.asarray(group["effective_scheduled_onset"].to_numpy(float)), *vector[:5], n_quad=64)
                else:
                    cdf = module.led_off_cdf_jax(stim, vector[0], vector[2], vector[3], vector[4])
                if family == "gamma":
                    lapse_cdf = gamma.gamma_lapse_cdf_jax(stim, params["shape_lapse"], params["scale_lapse"])
                elif family == "gaussian":
                    lapse_cdf = -jnp.expm1(gaussian.gaussian_lapse_logsurvival_jax(stim, params["mu_lapse"], params["sigma_lapse"]))
                else:
                    lapse_cdf = jnp.minimum(stim / 2.2, 1.0)
                predicted_lapse_fraction = params["lapse_prob"] * float(jnp.mean(lapse_cdf))
                predicted_fraction = (1.0 - params["lapse_prob"]) * float(jnp.mean(cdf)) + predicted_lapse_fraction
            if abs(float(np.sum(curve["mixture_rate"]) * dt) - predicted_fraction) > MASS_TOLERANCE:
                raise RuntimeError("Full curve mass disagrees with analytic abort probability.")

            aborts = group.loc[group["likelihood_role"].eq("abort")]
            counts, _ = np.histogram(aborts["timed_fix"] - aborts["effective_scheduled_onset"], bins=empirical["edges_s"])
            np.testing.assert_allclose(counts / (len(group) * 0.01), empirical["rate"], atol=1e-12, rtol=0.0)
            model_bin_mass, _ = np.histogram(grid, bins=empirical["edges_s"], weights=curve["mixture_rate"] * dt)
            model_bin_rate = model_bin_mass / np.diff(empirical["edges_s"])
            visible = (empirical["centers_s"] >= DISPLAY_RANGE_S[0]) & (empirical["centers_s"] < DISPLAY_RANGE_S[1])
            residual = model_bin_rate - empirical["rate"]
            group_loglike = float(module.proactive_led_step_jump_no_trunc_lapse_loglike_jax(params, likelihood_data(group)))
            condition_rows.append({"session_type": session_type, "lapse_family": family, "LED_trial": led,
                "n_total": len(group), "n_abort": len(aborts), "observed_abort_fraction": len(aborts) / len(group),
                "predicted_abort_fraction": predicted_fraction, "log_likelihood": group_loglike,
                "predicted_lapse_abort_fraction": predicted_lapse_fraction,
                "predicted_proactive_abort_fraction": predicted_fraction - predicted_lapse_fraction,
                "display_rmse_rate_per_s": float(np.sqrt(np.mean(residual[visible] ** 2))),
                "display_integrated_absolute_error": float(np.sum(np.abs(residual[visible])) * 0.01),
                "full_integrated_absolute_error": float(np.sum(np.abs(residual)) * 0.01),
                "numerical_full_curve_mass": float(np.sum(curve["mixture_rate"]) * dt)})
            for i, center in enumerate(empirical["centers_s"]):
                bin_rows.append({"session_type": session_type, "lapse_family": family, "LED_trial": led,
                    "bin_center_s": center, "data_rate_per_s": empirical["rate"][i],
                    "model_rate_per_s": model_bin_rate[i], "in_display": bool(visible[i])})
            diagnostics[session_type][family][led] = {
                "empirical": empirical, "curve": curve, "predicted_abort_fraction": predicted_fraction}
        print(f"s{session_type} {family}: log likelihood {loglike:.3f}", flush=True)

# %%
fit_table = pd.DataFrame(fit_rows)
for session_type in SESSION_TYPES:
    mask = fit_table["session_type"].eq(session_type)
    old_ll = fit_table.loc[mask & fit_table["lapse_family"].eq("exponential"), "posterior_mean_log_likelihood"].iloc[0]
    fit_table.loc[mask, "log_likelihood_change_vs_exponential"] = fit_table.loc[mask, "posterior_mean_log_likelihood"] - old_ll
condition_table = pd.DataFrame(condition_rows)
for session_type in SESSION_TYPES:
    for family in LAPSE_FAMILIES:
        combined = condition_table.loc[condition_table["session_type"].eq(session_type) & condition_table["lapse_family"].eq(family), "log_likelihood"].sum()
        joint = fit_table.loc[fit_table["session_type"].eq(session_type) & fit_table["lapse_family"].eq(family), "posterior_mean_log_likelihood"].iloc[0]
        np.testing.assert_allclose(combined, joint, atol=1e-6, rtol=0.0)
fit_table.to_csv(OUTPUT_DIR / "fit_quality_comparison.csv", index=False)
condition_table.to_csv(OUTPUT_DIR / "condition_fit_quality.csv", index=False)
pd.DataFrame(bin_rows).to_csv(OUTPUT_DIR / "histogram_comparison.csv", index=False)
pd.DataFrame(parameter_rows).to_csv(OUTPUT_DIR / "posterior_parameter_comparison.csv", index=False)
pd.DataFrame(stability_rows).to_csv(OUTPUT_DIR / "parameter_stability_audit.csv", index=False)
if INCLUDE_GAUSSIAN:
    boundary_table = pd.DataFrame(boundary_rows)
    boundary_table.loc[boundary_table["lapse_family"].eq("gaussian")].to_csv(OUTPUT_DIR / "gaussian_parameter_boundary_audit.csv", index=False)
    if INCLUDE_GAMMA:
        boundary_table.loc[boundary_table["lapse_family"].eq("gamma")].to_csv(OUTPUT_DIR / "gamma_parameter_boundary_audit.csv", index=False)

fig, axes = plt.subplots(len(LAPSE_FAMILIES), 2, figsize=(12.2, 15.0 if INCLUDE_GAMMA else 11.8 if INCLUDE_GAUSSIAN else 8.6), sharex=True, sharey=True)
colors = {0: "#1F77B4", 1: "#D62728"}
visible_max = 0.0
for row, family in enumerate(LAPSE_FAMILIES):
    for col, session_type in enumerate(SESSION_TYPES):
        ax = axes[row, col]
        annotations = []
        for led in (0, 1):
            payload = diagnostics[session_type][family][led]
            empirical, curve = payload["empirical"], payload["curve"]
            ax.stairs(empirical["rate"], empirical["edges_s"], color=colors[led], alpha=0.6, linewidth=1.1)
            ax.plot(curve["model_t_s"], curve["mixture_rate"], color=colors[led], linewidth=2.2)
            visible = (empirical["centers_s"] >= DISPLAY_RANGE_S[0]) & (empirical["centers_s"] <= DISPLAY_RANGE_S[1])
            visible_max = max(visible_max, float(empirical["rate"][visible].max()))
            model_visible = (curve["model_t_s"] >= DISPLAY_RANGE_S[0]) & (curve["model_t_s"] <= DISPLAY_RANGE_S[1])
            visible_max = max(visible_max, float(curve["mixture_rate"][model_visible].max()))
            annotations.append(f"{'ON' if led else 'OFF'}: observed {empirical['mass']:.3f}, model {payload['predicted_abort_fraction']:.3f}")
        ax.axvline(0.0, color="0.5", linestyle=":", linewidth=1.0)
        ax.text(0.98, 0.97, "\n".join(annotations), transform=ax.transAxes, ha="right", va="top", fontsize=9)
        ax.set_title(f"Session type {session_type}: {family} lapse", fontsize=12)
        ax.set_xlim(*DISPLAY_RANGE_S)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
        if col == 0:
            ax.set_ylabel("Abort rate (s$^{-1}$)")
        if row == len(LAPSE_FAMILIES) - 1:
            ax.set_xlabel("Time from scheduled LED onset (s)")
for ax in axes.flat:
    ax.set_ylim(0.0, visible_max * 1.18)
title_text = ("LED7 abort RT: exponential, uniform, Gaussian and Gamma lapses\nGamma clock: fitted shape and scale; location fixed at fixation start"
              if INCLUDE_GAMMA else "LED7 abort RT: exponential, uniform and Gaussian lapses\nGaussian clock: fixation time >= 0, with fitted μ and σ"
              if INCLUDE_GAUSSIAN else "LED7 abort RT: exponential versus uniform lapses\nUniform lapse timing: 0–2.2 s from fixation start")
title = fig.suptitle(title_text, fontsize=14, y=0.995)
title.set_in_layout(False)
fig.legend(handles=[Line2D([], [], color=colors[0], label="LED OFF"),
                    Line2D([], [], color=colors[1], label="LED ON"),
                    Line2D([], [], color="0.5", linewidth=1, drawstyle="steps-mid", label="Data (10 ms bins)"),
                    Line2D([], [], color="0.2", linewidth=2.2, label="Proactive + lapse")],
           loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=4, frameon=False)
footer = fig.text(0.5, 0.008, "Trial-pooled rates; denominator includes abort and censored trials. OFF zero is scheduled; ON zero is actual onset.", ha="center", fontsize=9)
fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.88))
for extension in ("png", "pdf"):
    fig.savefig(OUTPUT_DIR / f"{FIGURE_NAME}.{extension}", dpi=220, bbox_inches="tight", bbox_extra_artists=(title, footer))
plt.close(fig)

metadata = {"uniform_interval_s": [0.0, 2.2], "time_reference": "fixation start",
    "posterior_point_estimate": "mean", "model_grid_s": 0.001, "data_bin_s": 0.01,
    "display_range_s": DISPLAY_RANGE_S, "trial_population": "identical baseline event-3 plus censored likelihood rows",
    "pooling": "trial_weighted", "normalization": "all event-or-censored trials in each session-type and LED group",
    "interpretation": "In-sample comparisons at posterior means; prediction not evaluated on held-out trials.",
    "source_script": str(Path(__file__).resolve()), "source_fits": provenance,
    "uniform_interval_decision": "user selected a fixed interval of 0 to 2.2 seconds from fixation start",
    "historical_hashes_verified": len(baseline_hashes)}
metadata.update(lapse_families=LAPSE_FAMILIES, figure_layout=[len(LAPSE_FAMILIES), 2])
if INCLUDE_GAUSSIAN:
    metadata.update(gaussian_support_s=[0.0, None], gaussian_parameter_bounds=gaussian.PARAM_BOUNDS,
        gaussian_parameterization="mu and sigma of the underlying Gaussian; actual retained moments saved separately",
        gaussian_bound_audit="posterior mean distance to each hard bound in posterior SD; flag within two SD",
        gaussian_mu_decision="user allows negative underlying mu with sampled times conditioned on >= 0")
if INCLUDE_GAMMA:
    metadata.update(gamma_location_s=0.0, gamma_parameter_bounds=gamma.PARAM_BOUNDS,
        gamma_parameterization="shape and scale in seconds; location fixed at zero",
        gamma_location_decision="user selected fixed zero location from fixation start",
        gamma_display="exact lapse CDF differences averaged over 1 ms bins, clipped at fixation start and each trial stimulus time",
        historical_curves="exponential, uniform and Gaussian arrays reused from saved Gaussian comparison")
metadata["source_code_sha256"] = {
    str(path): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in (Path(__file__).resolve(), Path(uniform.__file__), Path(exponential.__file__),
                 Path(exponential.step_jump.__file__),
                 SCRIPT_DIR / "numpyro_svi_proactive_led_step_jump_no_trunc_exp_lapse_led7_s8_s9_aggregate.py")
}
if INCLUDE_GAUSSIAN:
    metadata["source_code_sha256"][gaussian.__file__] = hashlib.sha256(Path(gaussian.__file__).read_bytes()).hexdigest()
if INCLUDE_GAMMA:
    metadata["source_code_sha256"][gamma.__file__] = hashlib.sha256(Path(gamma.__file__).read_bytes()).hexdigest()
(OUTPUT_DIR / "comparison_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
with (OUTPUT_DIR / "comparison_payload.pkl").open("wb") as handle:
    pickle.dump({"metadata": metadata, "diagnostics": diagnostics}, handle)
print(fit_table.to_string(index=False))
print(condition_table.to_string(index=False))
print(OUTPUT_DIR / f"{FIGURE_NAME}.png")
