# %%
"""One row: s8 uniform lapse PDF, s8 abort RT, s9 PDF, s9 abort RT."""

# %%
# Editable paths and plotting settings.
from functools import partial
from pathlib import Path
import hashlib
import json
import os
import pickle
import shutil

SCRIPT_DIR = Path(__file__).resolve().parent
FIT_ROOT = SCRIPT_DIR / "numpyro_mcmc_led7_s8_s9_aggregate_no_trunc_uniform_lapse_fitted_width_outputs"
CACHE_ROOT = SCRIPT_DIR / "numpyro_svi_led7_s8_s9_aggregate_no_trunc_gamma_lapse_outputs" / "summary_figures"
OUTPUT_DIR = FIT_ROOT / "summary_figures"
FIGURE_NAME = "led7_s8_s9_uniform_fitted_width_lapse_pdf_abort_rate_1x4"
SESSION_TYPES = (8, 9)
DISPLAY_RANGE_S = (-0.3, 0.7)
LAPSE_RANGE_S = (0.0, 0.5)
TRIAL_CHUNK = 256

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpyro_proactive_led_step_jump_no_trunc_exp_lapse_svi_utils as exponential
import numpyro_proactive_led_step_jump_no_trunc_uniform_lapse_svi_utils as uniform
exponential.step_jump._legendre_nodes_weights(64)


# %%
@partial(jax.jit, static_argnames=("led",))
def curve_chunk(grid, onset, stim, vector, led):
    physical_t = grid[None, :] + onset[:, None]
    risk = (physical_t >= 0.0) & (physical_t < stim[:, None])
    if led:
        proactive = exponential.led_on_pdf_jax(physical_t, onset[:, None], *vector[:5])
    else:
        proactive = exponential.led_off_pdf_jax(physical_t, vector[0], vector[2], vector[3], vector[4])
    # Exact uniform CDF differences within 1 ms display bins avoid grid artifacts
    # when a bin straddles fixation start, the fitted cutoff, or censoring.
    dt = grid[1] - grid[0]
    lower = jnp.clip(physical_t - dt / 2.0, 0.0, stim[:, None])
    upper = jnp.clip(physical_t + dt / 2.0, 0.0, stim[:, None])
    lapse = (jnp.clip(upper / vector[6], 0.0, 1.0) - jnp.clip(lower / vector[6], 0.0, 1.0)) / dt
    return ((1.0 - vector[5]) * jnp.sum(jnp.where(risk, proactive, 0.0), axis=0),
            vector[5] * jnp.sum(lapse, axis=0))


# %%
OUTPUT_DIR.mkdir(exist_ok=True)
cached = pickle.loads((CACHE_ROOT / "comparison_payload.pkl").read_bytes())
fig, axes = plt.subplots(1, 4, figsize=(22, 4.8), gridspec_kw={"width_ratios": [1, 1.5, 1, 1.5]})
colors = {0: "#1F77B4", 1: "#D62728"}
condition_rows, fit_rows, parameter_rows, bin_rows = [], [], [], []
lapse_pdf_rows, lapse_pdf_audit = [], []
diagnostics, lapse_parameters = {}, {}
abort_max, pdf_max = 0.0, 0.0

for column, session_type in enumerate(SESSION_TYPES):
    fit_dir = FIT_ROOT / f"session_type_{session_type}"
    summary = json.loads((fit_dir / "run_summary.json").read_text())
    if summary["status"] != "converged":
        raise RuntimeError(f"s{session_type} requires convergence review: {summary['status']}")
    samples = np.load(fit_dir / "posterior_samples.npz")
    params = {name: float(samples[name].mean()) for name in samples.files}
    for name, value in params.items():
        np.testing.assert_allclose(value, summary["posterior_mean_parameters"][name], atol=1e-12)
    width, p = params["width_lapse"], params["lapse_prob"]
    lapse_parameters[session_type] = params
    vector = jnp.asarray([params[name] for name in (*uniform.PARAM_NAMES, "width_lapse")])
    frame = pd.read_csv(fit_dir / "source_row_manifest.csv")
    frame = frame.loc[frame.likelihood_role.isin(("abort", "censored"))].copy()
    title = f"[Uniform(0, {width:.3f} s); lapse p = {100*p:.2f}%]"
    pdf_ax, abort_ax = axes[2 * column:2 * column + 2]
    # Draw the step exactly, including the fitted cutoff rather than snapping
    # it to a display grid. The conditional PDF has full-support area one.
    right = max(LAPSE_RANGE_S[1], width * 1.15)
    pdf_ax.plot([0, width, width, right], [1 / width, 1 / width, 0, 0], color="#7040A0", linewidth=2.2)
    pdf_ax.fill_between([0, width], 0, 1 / width, color="#7040A0", alpha=0.10)
    pdf_ax.set(xlim=(0, right), xlabel="Time from fixation start (s)", ylabel="Lapse density (s$^{-1}$)")
    pdf_ax.set_title(f"Session type {session_type}: lapse RT\n{title}", fontsize=11)
    pdf_max = max(pdf_max, 1 / width)
    np.testing.assert_allclose(stats.uniform.cdf(width, scale=width), 1.0)
    lapse_grid = np.unique(np.r_[np.arange(0, right + .0005, .001), width])
    density = np.exp(np.asarray(uniform.uniform_lapse_logpdf_jax(lapse_grid, width)))
    reference_pdf = np.where(lapse_grid < width, stats.uniform.pdf(lapse_grid, scale=width), 0.0)
    np.testing.assert_allclose(density, reference_pdf, atol=1e-12)
    lapse_pdf_rows.extend({"session_type": session_type, "fixation_time_s": float(t), "density_per_s": float(f)}
                          for t, f in zip(lapse_grid, density))
    lapse_pdf_audit.append({"session_type": session_type, "width_s": width, "lapse_prob": p,
        "mean_lapse_time_s": width / 2, "sd_lapse_time_s": width / np.sqrt(12),
        "full_support_mass": 1.0, "displayed_mass": float(stats.uniform.cdf(right, scale=width)),
        "max_scipy_pdf_error": float(np.max(np.abs(density - reference_pdf)))})
    parameter_table = pd.read_csv(fit_dir / "posterior_summary.csv")
    trace_fig, trace_axes = plt.subplots(len(samples.files), 2, figsize=(12, 16))
    for row, name in enumerate(samples.files):
        values = samples[name]
        selected = parameter_table.loc[parameter_table.parameter.eq(name)].iloc[0]
        for chain, chain_values in enumerate(values):
            stride = max(1, len(chain_values) // 1500)
            trace_axes[row, 0].plot(np.arange(len(chain_values))[::stride], chain_values[::stride],
                                    linewidth=.6, alpha=.7, label=f"Chain {chain + 1}")
            trace_axes[row, 1].hist(chain_values, bins=60, density=True, histtype="step", alpha=.7)
        trace_axes[row, 0].set_ylabel(name)
        trace_axes[row, 1].set_title(f"R-hat {selected.r_hat:.4f}; ESS {selected.n_eff:.0f}", fontsize=10)
    trace_axes[0, 0].legend(ncol=4, fontsize=8)
    trace_axes[-1, 0].set_xlabel("Retained MCMC draw")
    trace_axes[-1, 1].set_xlabel("Parameter value")
    trace_fig.suptitle(f"LED7 type {session_type}: joint Metropolis posterior diagnostics")
    trace_fig.tight_layout(rect=(0, 0, 1, .98))
    trace_fig.savefig(OUTPUT_DIR / f"posterior_trace_diagnostics_s{session_type}.png", dpi=150)
    plt.close(trace_fig)
    parameter_table.insert(0, "session_type", session_type)
    parameter_rows.extend(parameter_table.to_dict("records"))
    diagnostics[session_type] = {}
    annotations = []
    for led in (0, 1):
        group = frame.loc[frame.LED_trial.eq(led)]
        aborts = group.loc[group.likelihood_role.eq("abort")]
        old = cached["diagnostics"][session_type]["uniform"][led]
        empirical = dict(old["empirical"])
        grid = old["curve"]["model_t_s"]
        dt = float(np.diff(grid)[0])
        totals = np.zeros((2, len(grid)))
        for start in range(0, len(group), TRIAL_CHUNK):
            chunk = group.iloc[start:start + TRIAL_CHUNK]
            values = curve_chunk(jnp.asarray(grid), jnp.asarray(chunk.effective_scheduled_onset.to_numpy(float)),
                                 jnp.asarray(chunk.intended_fix.to_numpy(float)), vector, led=led)
            totals += np.asarray(jax.device_get(values))
        totals /= len(group)
        mixture = totals.sum(axis=0)
        counts, _ = np.histogram(aborts.timed_fix - aborts.effective_scheduled_onset, empirical["edges_s"])
        rates = counts / (len(group) * np.diff(empirical["edges_s"]))
        np.testing.assert_allclose(rates, empirical["rate"], atol=1e-11)
        stim = jnp.asarray(group.intended_fix.to_numpy(float))
        if led:
            cdf = exponential.led_on_cdf_jax(stim, jnp.asarray(group.effective_scheduled_onset.to_numpy(float)), *vector[:5])
        else:
            cdf = exponential.led_off_cdf_jax(stim, vector[0], vector[2], vector[3], vector[4])
        lapse_fraction = p * float(jnp.mean(jnp.minimum(stim / width, 1.0)))
        predicted = (1 - p) * float(jnp.mean(cdf)) + lapse_fraction
        full_mass = float(np.sum(mixture) * dt)
        np.testing.assert_allclose(full_mass, predicted, atol=2e-3, rtol=0)
        model_bin_mass, _ = np.histogram(grid, empirical["edges_s"], weights=mixture * dt)
        model_rates = model_bin_mass / np.diff(empirical["edges_s"])
        empirical.update(binned_model_rate=model_rates,
                         integrated_absolute_error=float(np.sum(np.abs(model_rates - rates)) * .01),
                         rmse_rate_per_s=float(np.sqrt(np.mean((model_rates - rates) ** 2))))
        visible = (empirical["centers_s"] >= DISPLAY_RANGE_S[0]) & (empirical["centers_s"] < DISPLAY_RANGE_S[1])
        visible_model = (grid >= DISPLAY_RANGE_S[0]) & (grid <= DISPLAY_RANGE_S[1])
        abort_max = max(abort_max, rates[visible].max(), mixture[visible_model].max())
        abort_ax.stairs(rates, empirical["edges_s"], color=colors[led], alpha=0.6, linewidth=1.1)
        abort_ax.plot(grid, mixture, color=colors[led], linewidth=2.2)
        observed = len(aborts) / len(group)
        annotations.append(f"{'ON' if led else 'OFF'}: observed {observed:.3f}, model {predicted:.3f}")
        condition_rows.append({"session_type": session_type, "LED_trial": led, "n_total": len(group),
            "n_abort": len(aborts), "observed_abort_fraction": observed, "predicted_abort_fraction": predicted,
            "predicted_lapse_abort_fraction": lapse_fraction, "full_curve_mass": full_mass,
            "display_rmse_rate_per_s": float(np.sqrt(np.mean((model_rates[visible] - rates[visible]) ** 2)))})
        bin_rows.extend({"session_type": session_type, "LED_trial": led, "center_s": center,
                         "data_rate_per_s": rate, "model_rate_per_s": model}
                        for center, rate, model in zip(empirical["centers_s"], rates, model_rates))
        diagnostics[session_type][led] = {"empirical": empirical, "curve": {"model_t_s": grid,
            "proactive_rate": totals[0], "lapse_rate": totals[1], "mixture_rate": mixture}, "predicted_abort_fraction": predicted}
    abort_ax.text(0.98, 0.97, "\n".join(annotations), transform=abort_ax.transAxes, ha="right", va="top", fontsize=9)
    abort_ax.axvline(0, color="0.5", linestyle=":", linewidth=1)
    abort_ax.set(xlim=DISPLAY_RANGE_S, xlabel="Time from scheduled LED onset (s)", ylabel="Abort rate (s$^{-1}$)")
    abort_ax.set_title(f"Session type {session_type}: data + model\n{title}", fontsize=11)
    old_fit = pd.read_csv(CACHE_ROOT / "fit_quality_comparison.csv")
    old_fit = old_fit.loc[old_fit.session_type.eq(session_type)].set_index("lapse_family")
    ll = summary["posterior_mean_log_likelihood"]
    fit_rows.append({"session_type": session_type, "n_trials": len(frame), "width_lapse_s": width,
        "width_q025_s": float(np.quantile(samples["width_lapse"], .025)),
        "width_q975_s": float(np.quantile(samples["width_lapse"], .975)), "lapse_prob": p,
        "posterior_mean_log_likelihood": ll, "change_vs_fixed_uniform": ll - old_fit.loc["uniform", "posterior_mean_log_likelihood"],
        "change_vs_exponential": ll - old_fit.loc["exponential", "posterior_mean_log_likelihood"],
        "inference": summary["inference"], "max_rhat": summary["max_rhat"], "min_ess": summary["min_effective_sample_size"]})

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
for ax in axes[::2]:
    ax.set_ylim(0, pdf_max * 1.18)
for ax in axes[1::2]:
    ax.set_ylim(0, abort_max * 1.18)
fig.suptitle("LED7 uniform lapses: fitted width from fixation start", fontsize=14, y=0.99)
fig.legend(handles=[Line2D([], [], color="#7040A0", linewidth=2.2, label="Conditional lapse PDF"),
    Line2D([], [], color=colors[0], label="LED OFF"), Line2D([], [], color=colors[1], label="LED ON"),
    Line2D([], [], color="0.5", drawstyle="steps-mid", label="Data (10 ms bins)"),
    Line2D([], [], color="0.2", linewidth=2.2, label="Proactive + lapse")],
    loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=5, frameon=False)
fig.text(0.5, 0.012, "Curves use MCMC posterior means. Lapse PDFs have unit full-support area; p is the mixture probability.\n"
         "Abort rates divide by all abort/censored trials. OFF zero is scheduled; ON zero is actual LED onset.", ha="center", fontsize=9)
fig.tight_layout(rect=(0, 0.07, 1, 0.85))
for extension in ("png", "pdf"):
    fig.savefig(OUTPUT_DIR / f"{FIGURE_NAME}.{extension}", dpi=220, bbox_inches="tight")
plt.close(fig)
pd.DataFrame(fit_rows).to_csv(OUTPUT_DIR / "fit_quality_comparison.csv", index=False)
pd.DataFrame(condition_rows).to_csv(OUTPUT_DIR / "condition_fit_quality.csv", index=False)
pd.DataFrame(parameter_rows).to_csv(OUTPUT_DIR / "posterior_parameter_comparison.csv", index=False)
pd.DataFrame(bin_rows).to_csv(OUTPUT_DIR / "histogram_comparison.csv", index=False)
pd.DataFrame(lapse_pdf_rows).to_csv(OUTPUT_DIR / "lapse_pdf_curves.csv", index=False)
pd.DataFrame(lapse_pdf_audit).to_csv(OUTPUT_DIR / "lapse_pdf_audit.csv", index=False)
metadata = {"source_script": str(Path(__file__).resolve()), "source_fit_root": str(FIT_ROOT), "layout": [1, 4],
    "columns": ["s8 lapse PDF", "s8 abort data/model", "s9 lapse PDF", "s9 abort data/model"],
    "lapse_location_s": 0.0, "fitted": "five proactive parameters, lapse probability and uniform width per session type",
    "inference_change": "MCMC replaces earlier SVI because the exact sharp-cutoff likelihood is discontinuous in width",
    "point_estimate": "density at mean parameters, not posterior-average density", "trial_selection": "exact baseline eight-animal manifest rows",
    "pooling": "trial pooled OFF plus all ON; early aborts retained", "abort_normalization": "all event-or-censored rows per LED/session-type group",
    "lapse_pdf_normalization": "conditional on a lapse, full support area 1; not weighted by p", "data_bin_s": .01, "model_bin_s": .001,
    "lapse_bin_evaluation": "exact uniform CDF differences, clipped at fixation start and each trial stimulus time",
    "quality_scope": "in sample at posterior means; different inference algorithms, no held-out comparison"}
hashes = json.loads((FIT_ROOT / "historical_artifact_hashes.json").read_text())
for filename, expected_hash in hashes.items():
    assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected_hash, filename
metadata["historical_artifact_hashes_verified"] = len(hashes)
(OUTPUT_DIR / "comparison_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
with (OUTPUT_DIR / "comparison_payload.pkl").open("wb") as handle:
    pickle.dump({"metadata": metadata, "diagnostics": diagnostics, "lapse_parameters": lapse_parameters}, handle)
(OUTPUT_DIR / "source").mkdir(exist_ok=True)
shutil.copy2(__file__, OUTPUT_DIR / "source" / Path(__file__).name)
print(pd.DataFrame(fit_rows).to_string(index=False))
print(pd.DataFrame(condition_rows).to_string(index=False))
print(OUTPUT_DIR / f"{FIGURE_NAME}.png")
