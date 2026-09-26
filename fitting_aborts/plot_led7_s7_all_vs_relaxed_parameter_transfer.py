# %%
"""Compare original and relaxed-matched LED7 s7 fits and transfer parameters.

Extends plot_led7_s7_relaxed_s9_timing_matched_scheduled_onset_compact.py.
Timing pairs come from the saved fit manifests, so their effective scheduled
onset is already fixation referenced. No MAT/CSV timing conversion is needed.
"""

from pathlib import Path
from time import perf_counter
import json
import os
import pickle
import sys


# %%
############ Editable parameters ############
SCRIPT_DIR = Path(__file__).resolve().parent
ORIGINAL_FIT_ROOT = (
    SCRIPT_DIR / "numpyro_svi_led7_s7_all_vs_s9_timing_matched_no_trunc_exp_lapse_outputs"
)
RELAXED_FIT_ROOT = (
    SCRIPT_DIR / "numpyro_svi_led7_s7_relaxed_s9_timing_matched_no_trunc_exp_lapse_outputs"
)
PAYLOAD_PATH = RELAXED_FIT_ROOT / "summary_figures/theory_data_diagnostic_payload.pkl"
PREVIOUS_AUDIT_PATH = SCRIPT_DIR / (
    "led7_s7_all_vs_relaxed_s9_timing_matched_svi_"
    "abort_rate_scheduled_onset_1x2_audit.csv"
)
OUTPUT_STEM = "led7_s7_all_vs_relaxed_parameter_transfer_1x4"
FIGURE_PATH = SCRIPT_DIR / f"{OUTPUT_STEM}.png"
AUDIT_PATH = SCRIPT_DIR / f"{OUTPUT_STEM}_audit.csv"
CURVES_PATH = SCRIPT_DIR / f"{OUTPUT_STEM}_curves.npz"

DATASETS = {
    "s7_all": {
        "fit_dir": ORIGINAL_FIT_ROOT / "s7_all",
        "counts": {0: (14_215, 88_414), 1: (8_276, 42_485)},
    },
    "s7_matched_to_s9": {
        "fit_dir": RELAXED_FIT_ROOT / "s7_matched_to_s9",
        "counts": {0: (1_914, 13_238), 1: (1_416, 6_403)},
    },
}
# Each entry maps a column to its data/timing source and its parameter source.
COMPARISONS = (
    ("original", "s7_all", "s7_all", "Original s7\nOriginal fit"),
    ("matched", "s7_matched_to_s9", "s7_matched_to_s9", "Timing-matched s7\nMatched fit"),
    ("transfer", "s7_matched_to_s9", "s7_all", "Timing-matched s7\nOriginal parameters"),
)
ANIMALS = (90, 92, 93, 98, 99, 100, 102, 103)
LED_COLORS = {0: "tab:blue", 1: "tab:red"}
LED_LABELS = {0: "LED OFF", 1: "LED ON"}
DISPLAY_RANGE_S = (-1.0, 1.0)
FULL_RANGE_S = (-2.2, 2.2)
DATA_BIN_S = 0.020
MODEL_DT_S = 0.001
MC_SAMPLES = 5_000
MC_SEED = 20_260_912
TRIAL_CHUNK = 256
QUADRATURE_NODES = 64
MASS_TOLERANCE = 0.002
FIGURE_DPI = 250


# %%
############ Load the existing fits and exact empirical histograms ############
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

jax.config.update("jax_enable_x64", True)
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import numpyro_proactive_led_step_jump_no_trunc_exp_lapse_svi_utils as svi_utils

with PAYLOAD_PATH.open("rb") as handle:
    payload = pickle.load(handle)
assert payload["diagnostic_profile"] == "relaxed_unequal"
assert np.isclose(payload["data_bin_s"], DATA_BIN_S)
assert np.isclose(payload["model_dt_s"], MODEL_DT_S)
assert tuple(payload["onset_full_range_s"]) == FULL_RANGE_S
assert tuple(payload["onset_display_range_s"]) == DISPLAY_RANGE_S
previous_audit = pd.read_csv(PREVIOUS_AUDIT_PATH).set_index(["dataset", "LED_group"])
model_t = np.asarray(
    payload["diagnostics"]["s7_all"][0]["scheduled_onset"]["model"]["model_t_s"]
)
assert np.allclose(np.diff(model_t), MODEL_DT_S, atol=1e-12)
assert np.allclose(
    model_t[[0, -1]], [FULL_RANGE_S[0] + MODEL_DT_S / 2, FULL_RANGE_S[1] - MODEL_DT_S / 2]
)
saved_arrays = {
    "model_t_s": model_t,
    "parameter_names": np.array(svi_utils.PARAM_NAMES),
    "timing_column_names": np.array(["intended_fix", "effective_scheduled_onset"]),
    "mc_seed": MC_SEED,
    "mc_samples_per_curve": MC_SAMPLES,
}

for dataset_key, spec in DATASETS.items():
    fit_dir = spec["fit_dir"]
    summary = json.loads((fit_dir / "run_summary.json").read_text())
    assert summary["status"] == "complete"
    assert summary["stop_reason"] == "patience12_restore_best"
    assert summary["n_nonfinite_losses"] == 0
    assert summary["all_posterior_samples_finite"]
    config = summary["config"]
    assert config["training_level"] == 16 and config["session_type"] == 7
    assert config["repeat_trial"] == [0, 2, "NaN"] and config["led_trial"] == [0, 1]
    assert config["pooling"] == "trial_weighted" and config["truncation"] == "none"
    assert config["quadrature_nodes"] == QUADRATURE_NODES
    assert tuple(config["animals"]) == ANIMALS
    if dataset_key == "s7_matched_to_s9":
        assert config["matching_profile"] == "relaxed_unequal"
        assert config["match_sample_sizes_by_led"] == {"0": 13_750, "1": 7_000}

    posterior = pd.read_csv(fit_dir / "main_fullrank_posterior_summary.csv")
    assert posterior["parameter"].tolist() == svi_utils.PARAM_NAMES
    spec["params"] = dict(zip(posterior["parameter"], posterior["mean"]))
    for name, value in spec["params"].items():
        bounds = svi_utils.PARAM_BOUNDS[name]["hard"]
        assert np.isfinite(value) and bounds[0] <= value <= bounds[1]
        assert np.isclose(value, payload["fit_parameters"][dataset_key][name], atol=1e-12)

    manifest = pd.read_csv(fit_dir / "source_row_manifest.csv")
    assert not manifest["source_row_index"].duplicated().any()
    frame = manifest.loc[manifest["likelihood_role"].isin(["abort", "censored"])].copy()
    assert tuple(sorted(frame["animal"].unique())) == ANIMALS
    assert set(frame["LED_trial"].unique()) == {0, 1}
    assert np.isfinite(frame[["intended_fix", "effective_scheduled_onset", "timed_fix"]]).all().all()
    assert np.allclose(
        frame["effective_scheduled_onset"], frame["intended_fix"] - frame["LED_onset_time"],
        atol=1e-12,
    )
    aborts = frame.loc[frame["likelihood_role"].eq("abort")]
    censored = frame.loc[frame["likelihood_role"].eq("censored")]
    assert aborts["abort_event"].eq(3).all()
    assert aborts["timed_fix"].lt(aborts["intended_fix"]).all()
    assert censored["success"].isin([-1, 1]).all()
    assert censored["timed_fix"].ge(censored["intended_fix"]).all()
    assert frame["effective_scheduled_onset"].ge(0).all()
    assert frame["effective_scheduled_onset"].lt(frame["intended_fix"]).all()
    spec["frame"] = frame
    spec["groups"] = {}
    spec["empirical"] = {}
    for led, (n_abort, n_total) in spec["counts"].items():
        group = frame.loc[frame["LED_trial"].eq(led)].copy()
        empirical = payload["diagnostics"][dataset_key][led]["scheduled_onset"]["empirical"]
        assert len(group) == n_total == empirical["n_total"]
        assert int(group["likelihood_role"].eq("abort").sum()) == n_abort == empirical["n_abort"]
        times = group.loc[group["likelihood_role"].eq("abort")]
        counts, edges = np.histogram(
            times["timed_fix"] - times["effective_scheduled_onset"],
            bins=empirical["data_edges_s"],
        )
        assert counts.sum() == n_abort
        assert np.allclose(np.diff(edges), DATA_BIN_S, atol=1e-12)
        assert np.allclose(counts / (n_total * DATA_BIN_S), empirical["data_rate"], atol=1e-12)
        assert np.isclose(np.sum(empirical["data_rate"]) * DATA_BIN_S, n_abort / n_total, atol=1e-12)
        spec["groups"][led] = group
        spec["empirical"][led] = empirical
        key = f"{dataset_key}_led{led}"
        saved_arrays[f"{key}_data_edges_s"] = edges
        saved_arrays[f"{key}_data_centers_s"] = empirical["data_centers_s"]
        saved_arrays[f"{key}_data_rate"] = empirical["data_rate"]
    saved_arrays[f"{dataset_key}_posterior_means"] = posterior["mean"].to_numpy()
    saved_arrays[f"{dataset_key}_fit_dir"] = str(fit_dir)

# Recheck that the saved matched rows are unchanged rows of the original fit.
original_by_id = DATASETS["s7_all"]["frame"].set_index("source_row_index")
matched_by_id = DATASETS["s7_matched_to_s9"]["frame"].set_index("source_row_index")
# Matching changes selection_order, so compare the preserved trial fields.
paired_columns = ["LED_trial", "intended_fix", "effective_scheduled_onset", "timed_fix", "likelihood_role"]
pd.testing.assert_frame_equal(
    matched_by_id[paired_columns], original_by_id.loc[matched_by_id.index, paired_columns],
    check_exact=True,
)


# %%
############ Preserve the original RNG order and intact timing pairs ############
# Four draws, in order: all OFF, all ON, matched OFF, matched ON. The transfer
# uses the already sampled matched frames; it never consumes another RNG draw.
rng = np.random.default_rng(MC_SEED)
for dataset_key, spec in DATASETS.items():
    spec["sampled_timings"] = {}
    for led in (0, 1):
        group = spec["groups"][led]
        positions = rng.integers(0, len(group), size=MC_SAMPLES)
        sample = group.iloc[positions]
        spec["sampled_timings"][led] = sample[
            ["intended_fix", "effective_scheduled_onset"]
        ].copy()
        key = f"{dataset_key}_led{led}"
        saved_arrays[f"{key}_sample_positions"] = positions
        saved_arrays[f"{key}_sample_source_row_index"] = sample["source_row_index"].to_numpy()
        saved_arrays[f"{key}_sample_timing_pairs"] = spec["sampled_timings"][led].to_numpy()


# %%
############ Evaluate proactive + lapse theory and its analytic total mass ############
def scheduled_onset_rate(led, timings, params):
    """Average the density at relative time + onset, stopping at each stimulus."""
    t_stim = timings["intended_fix"].to_numpy(dtype=float)
    t_led = timings["effective_scheduled_onset"].to_numpy(dtype=float)
    relative_t = jnp.asarray(model_t, dtype=jnp.float64)
    proactive_sum = np.zeros_like(model_t)
    lapse_sum = np.zeros_like(model_t)
    for start in range(0, len(timings), TRIAL_CHUNK):
        stop = min(start + TRIAL_CHUNK, len(timings))
        stimulus = jnp.asarray(t_stim[start:stop], dtype=jnp.float64)
        onset = jnp.asarray(t_led[start:stop], dtype=jnp.float64)
        physical_t = relative_t[None, :] + onset[:, None]
        at_risk = (physical_t >= 0) & (physical_t < stimulus[:, None])
        if led == 1:
            proactive = svi_utils.led_on_pdf_jax(
                physical_t, onset[:, None], params["V_A_base"], params["V_A_post_LED"],
                params["theta_A"], params["del_a_minus_del_LED"], params["del_m_plus_del_LED"],
            )
        else:
            proactive = svi_utils.led_off_pdf_jax(
                physical_t, params["V_A_base"], params["theta_A"],
                params["del_a_minus_del_LED"], params["del_m_plus_del_LED"],
            )
        lapse = params["beta_lapse"] * jnp.exp(-params["beta_lapse"] * jnp.maximum(physical_t, 0))
        proactive_sum += np.asarray(jax.device_get(jnp.where(at_risk, proactive, 0).sum(axis=0)))
        lapse_sum += np.asarray(jax.device_get(jnp.where(at_risk, lapse, 0).sum(axis=0)))
    proactive_rate = (1 - params["lapse_prob"]) * proactive_sum / len(timings)
    lapse_rate = params["lapse_prob"] * lapse_sum / len(timings)
    mixture = proactive_rate + lapse_rate
    assert np.isfinite(mixture).all() and (mixture >= 0).all()
    return mixture, proactive_rate, lapse_rate


def analytic_abort_probability(led, timings, params):
    """CDF mass over the same timing population used by a curve or full group."""
    t_stim = timings["intended_fix"].to_numpy(dtype=float)
    t_led = timings["effective_scheduled_onset"].to_numpy(dtype=float)
    proactive_sum = 0.0
    for start in range(0, len(timings), TRIAL_CHUNK):
        stop = min(start + TRIAL_CHUNK, len(timings))
        stimulus = jnp.asarray(t_stim[start:stop], dtype=jnp.float64)
        if led == 1:
            cdf = svi_utils.led_on_cdf_jax(
                stimulus, jnp.asarray(t_led[start:stop], dtype=jnp.float64),
                params["V_A_base"], params["V_A_post_LED"], params["theta_A"],
                params["del_a_minus_del_LED"], params["del_m_plus_del_LED"],
                n_quad=QUADRATURE_NODES,
            )
        else:
            cdf = svi_utils.led_off_cdf_jax(
                stimulus, params["V_A_base"], params["theta_A"],
                params["del_a_minus_del_LED"], params["del_m_plus_del_LED"],
            )
        proactive_sum += float(np.asarray(jax.device_get(cdf)).sum())
    proactive_mass = (1 - params["lapse_prob"]) * proactive_sum / len(timings)
    lapse_mass = params["lapse_prob"] * float(np.mean(1 - np.exp(-params["beta_lapse"] * t_stim)))
    return proactive_mass + lapse_mass


# %%
############ Calculate six unique curves and audit data agreement ############
curves = {}
audit_rows = []
start_time = perf_counter()
for comparison_key, timing_key, parameter_key, title in COMPARISONS:
    timing_spec = DATASETS[timing_key]
    params = DATASETS[parameter_key]["params"]
    for led in (0, 1):
        curve_start = perf_counter()
        empirical = timing_spec["empirical"][led]
        sample = timing_spec["sampled_timings"][led]
        theory, proactive, lapse = scheduled_onset_rate(led, sample, params)
        assert np.allclose(theory, proactive + lapse, atol=1e-12, rtol=1e-12)
        sampled_mass = analytic_abort_probability(led, sample, params)
        full_mass = analytic_abort_probability(led, timing_spec["groups"][led], params)
        curve_mass = float(theory.sum() * MODEL_DT_S)
        assert abs(curve_mass - sampled_mass) <= MASS_TOLERANCE
        assert 0 <= sampled_mass <= 1 and 0 <= full_mass <= 1

        # The first four curves reproduce the original figure, including MC draws.
        if comparison_key != "transfer":
            old = previous_audit.loc[(timing_key, LED_LABELS[led])]
            assert np.isclose(curve_mass, old["mc_theory_mass"], atol=1e-10, rtol=1e-10)
            assert np.isclose(full_mass, old["predicted_abort_fraction"], atol=1e-10, rtol=1e-10)

        binned_mass, _ = np.histogram(
            model_t, bins=empirical["data_edges_s"], weights=theory * MODEL_DT_S
        )
        binned_theory = binned_mass / np.diff(empirical["data_edges_s"])
        error = binned_theory - empirical["data_rate"]
        visible = (empirical["data_centers_s"] >= DISPLAY_RANGE_S[0]) & (
            empirical["data_centers_s"] <= DISPLAY_RANGE_S[1]
        )
        curves[comparison_key, led] = theory
        key = f"{comparison_key}_led{led}"
        saved_arrays[f"{key}_mixture_rate"] = theory
        saved_arrays[f"{key}_proactive_rate"] = proactive
        saved_arrays[f"{key}_lapse_rate"] = lapse
        saved_arrays[f"{key}_binned_theory_rate"] = binned_theory
        saved_arrays[f"{key}_timing_source"] = timing_key
        saved_arrays[f"{key}_parameter_source"] = parameter_key
        audit_rows.append({
            "comparison": comparison_key,
            "timing_dataset": timing_key,
            "parameter_dataset": parameter_key,
            "parameter_fit_dir": str(DATASETS[parameter_key]["fit_dir"]),
            "LED_group": LED_LABELS[led],
            "n_abort": empirical["n_abort"],
            "n_total": empirical["n_total"],
            "observed_abort_fraction": empirical["data_mass"],
            "predicted_abort_fraction_all_timings": full_mass,
            "predicted_abort_fraction_sampled_timings": sampled_mass,
            "mc_curve_integral": curve_mass,
            "integration_minus_sampled_cdf": curve_mass - sampled_mass,
            "mc_sampling_minus_all_timings_cdf": sampled_mass - full_mass,
            "mc_curve_minus_all_timings_cdf": curve_mass - full_mass,
            "prediction_minus_observed_all_timings": full_mass - empirical["data_mass"],
            "binned_rmse_full_per_s": float(np.sqrt(np.mean(error**2))),
            "binned_iae_full": float(np.abs(error).sum() * DATA_BIN_S),
            "binned_rmse_display_per_s": float(np.sqrt(np.mean(error[visible]**2))),
            "binned_iae_display": float(np.abs(error[visible]).sum() * DATA_BIN_S),
            "mc_samples": MC_SAMPLES,
            "mc_seed": MC_SEED,
            "mc_sampling_with_replacement": True,
            "curve_and_audit_seconds": perf_counter() - curve_start,
        })
        print(
            f"{comparison_key} {LED_LABELS[led]}: observed={empirical['data_mass']:.6f}, "
            f"predicted (all timings)={full_mass:.6f}, "
            f"MC integral={curve_mass:.6f}, "
            f"integral minus sampled CDF={curve_mass - sampled_mass:+.2e}",
            flush=True,
        )

audit = pd.DataFrame(audit_rows)
for led in (0, 1):
    matched_row = audit.loc[audit["comparison"].eq("matched") & audit["LED_group"].eq(LED_LABELS[led])].iloc[0]
    transfer_mask = audit["comparison"].eq("transfer") & audit["LED_group"].eq(LED_LABELS[led])
    for metric in ("binned_rmse_display_per_s", "binned_iae_display"):
        audit.loc[transfer_mask, f"transfer_minus_matched_{metric}"] = (
            audit.loc[transfer_mask, metric] - matched_row[metric]
        )
audit.to_csv(AUDIT_PATH, index=False)
np.savez_compressed(CURVES_PATH, **saved_arrays)
print(f"Six curves and mass audits: {perf_counter() - start_time:.2f} seconds", flush=True)


# %%
############ Draw the shared-axis 1 x 4 comparison ############
fig, axes = plt.subplots(1, 4, figsize=(22.4, 5.2), sharex=True, sharey=True)
for ax, (comparison_key, timing_key, parameter_key, title) in zip(axes[:3], COMPARISONS):
    for led in (0, 1):
        empirical = DATASETS[timing_key]["empirical"][led]
        ax.step(
            empirical["data_centers_s"], empirical["data_rate"], where="mid",
            color=LED_COLORS[led], linewidth=1.0, alpha=0.48,
        )
        ax.plot(model_t, curves[comparison_key, led], color=LED_COLORS[led], linewidth=2.2)
    ax.set_title(title, fontsize=12)

# Reuse column 1 and 3 arrays exactly, so column 4 cannot add sampling noise.
for led in (0, 1):
    axes[3].plot(model_t, curves["original", led], color=LED_COLORS[led], linewidth=2.2)
    axes[3].plot(
        model_t, curves["transfer", led], color=LED_COLORS[led],
        linewidth=2.2, linestyle="--",
    )
axes[3].set_title("Original parameters\nEffect of timing distribution", fontsize=12)
axes[3].legend(handles=[
    Line2D([0], [0], color="0.25", linewidth=2, label="Original timings"),
    Line2D([0], [0], color="0.25", linewidth=2, linestyle="--", label="Matched timings"),
], loc="upper left", frameon=False, fontsize=9)

for ax in axes:
    ax.axvline(0, color="0.55", linestyle=":", linewidth=0.9, zorder=0)
    ax.set_xlim(DISPLAY_RANGE_S)
    ax.set_xticks(np.arange(-1, 1.01, 0.5))
    ax.set_xlabel("Time from scheduled LED onset (s)")
    ax.grid(axis="y", color="0.88", linewidth=0.55)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel(r"Abort rate (s$^{-1}$)")
fig.suptitle("LED7 session type 7: original parameters with original and matched timings", y=0.99, fontsize=15)
fig.legend(handles=[
    Line2D([0], [0], color="tab:blue", linewidth=1, alpha=0.48, label="LED OFF data"),
    Line2D([0], [0], color="tab:blue", linewidth=2.2, label="LED OFF theory (5k MC)"),
    Line2D([0], [0], color="tab:red", linewidth=1, alpha=0.48, label="LED ON data"),
    Line2D([0], [0], color="tab:red", linewidth=2.2, label="LED ON theory (5k MC)"),
], loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=4, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.82), w_pad=1.4)
fig.savefig(FIGURE_PATH, dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)
print(audit.to_string(index=False))
print(f"Figure: {FIGURE_PATH}")
print(f"Audit: {AUDIT_PATH}")
print(f"Curves and timing draws: {CURVES_PATH}")
