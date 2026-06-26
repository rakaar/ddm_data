"""
Simple teaching script for a two-bound DDM on rat 2AFC data.

Run from this folder:

    .venv/bin/python juan_fit_teaching_2afc_ddm.py

The model is intentionally standard:

- choices are right vs left
- evidence is right_db - left_db
- drift is a linear function of evidence
- the starting point is fixed halfway between the bounds
- one boundary height and one nondecision time are shared by all trials
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import corner
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import Predictive, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoIAFNormal
from numpyro.infer.util import init_to_value
from numpyro.optim import Adam

from juan_ddm_math import (
    add_2afc_ddm_log_likelihood,
    choice_rt_log_pdf,
    two_bound_survival_probability,
    upper_choice_probability,
)


# ---------------------------------------------------------------------------
# Things students are expected to edit
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_URL = "https://raw.githubusercontent.com/rakaar/workshop/refs/heads/main/workshop_dataset_2AFC.csv"
DATA_FILE = os.path.join(SCRIPT_DIR, "workshop_dataset_2AFC.csv")
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "model_results_2afc")

MIN_RT = 0.08  # Drop faster trials and condition the likelihood on RT >= MIN_RT.
T_MAX = 1.0  # Treat trials at or beyond this deadline as right-censored.
MAX_TRIALS = 4000
PLOT_MAX_TRIALS = 1000
EVIDENCE_SCALE = 10.0

NUM_STEPS = 5000
NUM_POSTERIOR_SAMPLES = 1000
LEARNING_RATE = 0.001
RANDOM_SEED = 20260619

GUIDE_NUM_FLOWS = 2
GUIDE_HIDDEN_DIMS = [16, 16]

PRINT_EVERY = 250
EARLY_STOPPING_WINDOW = 200
EARLY_STOPPING_PATIENCE = 7
MIN_LOSS_IMPROVEMENT = 0.2
RELATIVE_LOSS_IMPROVEMENT = 0.0005

DRIFT_BIAS_RANGE = (-2.0, 2.0)
DRIFT_SENSITIVITY_RANGE = (0.0, 5.0)
BOUND_RANGE = (0.1, 3.0)

PARAMETER_NAMES = [
    "drift_bias",
    "drift_sensitivity",
    "bound",
    "nondecision_time",
]

PARAMETER_LABELS = {
    "drift_bias": "drift bias",
    "drift_sensitivity": "drift sensitivity",
    "bound": "bound",
    "nondecision_time": "nondecision time",
}


# ---------------------------------------------------------------------------
# 1. Load and clean the data
# ---------------------------------------------------------------------------


def load_data(csv_path=DATA_FILE):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        print("Downloading 2AFC teaching data...")
        df = pd.read_csv(DATA_URL)
        df.to_csv(csv_path, index=False)
        print(f"Saved a local copy to {csv_path}")

    complete_trials = (
        df["choice"].isin([-1.0, 1.0])
        & df["RT"].notna()
        & df["right_db"].notna()
        & df["left_db"].notna()
    )
    n_removed_fast = int(np.sum(complete_trials & (df["RT"] < MIN_RT)))

    valid_trials = (
        complete_trials
        & (df["RT"] >= MIN_RT)
    )
    df = df.loc[valid_trials, ["right_db", "left_db", "RT", "choice"]].copy()

    if len(df) == 0:
        raise ValueError("No valid 2AFC trials were found.")

    if len(df) > MAX_TRIALS:
        df = df.sample(n=MAX_TRIALS, random_state=RANDOM_SEED).sort_index()

    evidence = (df["right_db"].to_numpy(dtype=float) - df["left_db"].to_numpy(dtype=float)) / EVIDENCE_SCALE
    choices = df["choice"].to_numpy(dtype=float)
    raw_rts = df["RT"].to_numpy(dtype=float)
    is_censored = raw_rts >= T_MAX
    rts = np.where(is_censored, -1.0, raw_rts)

    return {
        "rts": jnp.asarray(rts),
        "choices": jnp.asarray(choices),
        "evidence": jnp.asarray(evidence),
        "right_db": df["right_db"].to_numpy(dtype=float),
        "left_db": df["left_db"].to_numpy(dtype=float),
        "n_trials": int(len(df)),
        "n_right_choices": int(np.sum(choices > 0)),
        "n_censored": int(np.sum(is_censored)),
        "n_removed_fast": n_removed_fast,
    }


# ---------------------------------------------------------------------------
# 2. Define the model
# ---------------------------------------------------------------------------


def ddm_model(rts, choices, evidence):
    min_rt = jnp.min(jnp.where(rts > 0, rts, jnp.inf))
    max_ndt = jnp.maximum(min_rt - 0.01, 0.001)

    drift_bias = numpyro.sample("drift_bias", dist.Uniform(*DRIFT_BIAS_RANGE))
    drift_sensitivity = numpyro.sample("drift_sensitivity", dist.Uniform(*DRIFT_SENSITIVITY_RANGE))
    bound = numpyro.sample("bound", dist.Uniform(*BOUND_RANGE))
    nondecision_time = numpyro.sample("nondecision_time", dist.Uniform(0.0, max_ndt))

    drift = drift_bias + drift_sensitivity * evidence

    add_2afc_ddm_log_likelihood(rts, choices, drift, bound, nondecision_time, MIN_RT, T_MAX)


# ---------------------------------------------------------------------------
# 3. Fit the model
# ---------------------------------------------------------------------------


def fit_model(data, num_steps=NUM_STEPS, num_samples=NUM_POSTERIOR_SAMPLES):
    initial_values = {
        "drift_bias": 0.0,
        "drift_sensitivity": 2.0,
        "bound": 1.0,
        "nondecision_time": 0.04,
    }
    guide = AutoIAFNormal(
        ddm_model,
        num_flows=GUIDE_NUM_FLOWS,
        hidden_dims=GUIDE_HIDDEN_DIMS,
        init_loc_fn=init_to_value(values=initial_values),
    )
    svi = SVI(ddm_model, guide, Adam(LEARNING_RATE), Trace_ELBO())

    rng_key = jax.random.PRNGKey(RANDOM_SEED)
    state = svi.init(rng_key, data["rts"], data["choices"], data["evidence"])
    update_step = jax.jit(svi.update)

    print(f"Loaded {data['n_trials']} 2AFC trials.")
    print(f"Right choices: {data['n_right_choices']}")
    print(f"Censored at {T_MAX:.2f}s: {data['n_censored']}")
    print("Fitting two-bound DDM...")

    losses = []
    checks_without_progress = 0
    steps_run = 0

    for step in range(int(num_steps)):
        state, loss = update_step(state, data["rts"], data["choices"], data["evidence"])
        steps_run = step + 1
        losses.append(float(loss))

        if (
            steps_run % EARLY_STOPPING_WINDOW == 0
            and len(losses) >= 2 * EARLY_STOPPING_WINDOW
        ):
            recent_loss = np.mean(losses[-EARLY_STOPPING_WINDOW:])
            previous_loss = np.mean(losses[-2 * EARLY_STOPPING_WINDOW : -EARLY_STOPPING_WINDOW])
            improvement = previous_loss - recent_loss
            needed_improvement = max(
                MIN_LOSS_IMPROVEMENT,
                RELATIVE_LOSS_IMPROVEMENT * abs(previous_loss),
            )

            if improvement < needed_improvement:
                checks_without_progress += 1
            else:
                checks_without_progress = 0

            if checks_without_progress >= EARLY_STOPPING_PATIENCE:
                print(f"  early stopping at step {steps_run}: loss has stabilized")
                break

        if step == 0 or steps_run % PRINT_EVERY == 0 or step == int(num_steps) - 1:
            recent_loss = np.mean(losses[-min(len(losses), EARLY_STOPPING_WINDOW) :])
            print(f"  step {steps_run:5d} / {int(num_steps):5d}   recent loss = {recent_loss:.2f}")

    params = svi.get_params(state)
    sample_key = jax.random.PRNGKey(RANDOM_SEED + 1)
    posterior = Predictive(guide, params=params, num_samples=int(num_samples))(
        sample_key,
        data["rts"],
        data["choices"],
        data["evidence"],
    )
    samples = {name: np.asarray(values) for name, values in posterior.items()}
    return samples, {"steps_run": steps_run}


# ---------------------------------------------------------------------------
# 4. Save a small summary
# ---------------------------------------------------------------------------


def make_output_dir():
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(OUTPUT_ROOT, run_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_summary(samples, data, output_dir, fit_info=None):
    fit_info = fit_info or {}
    rows = []
    for name in PARAMETER_NAMES:
        values = samples[name]
        rows.append(
            {
                "parameter": name,
                "mean": float(np.mean(values)),
                "p5": float(np.percentile(values, 5)),
                "p95": float(np.percentile(values, 95)),
            }
        )

    summary = pd.DataFrame(rows)
    summary_path = os.path.join(output_dir, "summary.csv")
    summary.to_csv(summary_path, index=False)

    np.savez(
        os.path.join(output_dir, "posterior_samples.npz"),
        **{name: values for name, values in samples.items()},
    )

    metadata = {
        "data_url": DATA_URL,
        "data_file": DATA_FILE,
        "n_trials": data["n_trials"],
        "n_right_choices": data["n_right_choices"],
        "n_censored": data["n_censored"],
        "min_rt": MIN_RT,
        "t_max": T_MAX,
        "max_trials": MAX_TRIALS,
        "plot_max_trials": PLOT_MAX_TRIALS,
        "evidence_scale": EVIDENCE_SCALE,
        "num_steps": int(fit_info.get("steps_run", NUM_STEPS)),
        "num_posterior_samples": int(len(samples["drift_bias"])),
    }
    with open(os.path.join(output_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved summary to {summary_path}")
    return summary


# ---------------------------------------------------------------------------
# 5. Plot data against the fitted model
# ---------------------------------------------------------------------------


def mean_parameter(samples, name):
    return float(np.mean(samples[name]))


def fitted_choice_probability(evidence, samples):
    drift = mean_parameter(samples, "drift_bias") + mean_parameter(samples, "drift_sensitivity") * evidence
    bound = mean_parameter(samples, "bound")
    return np.asarray(upper_choice_probability(jnp.asarray(drift), bound))


def fitted_rt_density(times, data, samples):
    rts = np.asarray(data["rts"], dtype=float)
    observed = rts > 0
    evidence = np.asarray(data["evidence"], dtype=float)[observed]
    if evidence.size > PLOT_MAX_TRIALS:
        plot_idx = np.linspace(0, evidence.size - 1, PLOT_MAX_TRIALS).astype(int)
        evidence = evidence[plot_idx]

    drift = mean_parameter(samples, "drift_bias") + mean_parameter(samples, "drift_sensitivity") * evidence
    bound = mean_parameter(samples, "bound")
    nondecision_time = mean_parameter(samples, "nondecision_time")

    decision_times = np.asarray(times, dtype=float)[:, None] - nondecision_time
    trunc_after_ndt = max(MIN_RT - nondecision_time, 1e-10)
    survival_after_trunc = np.asarray(
        two_bound_survival_probability(
            trunc_after_ndt,
            jnp.asarray(drift),
            bound,
        )
    )
    upper_log_pdf = choice_rt_log_pdf(
        jnp.asarray(decision_times),
        jnp.ones_like(jnp.asarray(decision_times)),
        jnp.asarray(drift)[None, :],
        bound,
    )
    lower_log_pdf = choice_rt_log_pdf(
        jnp.asarray(decision_times),
        -jnp.ones_like(jnp.asarray(decision_times)),
        jnp.asarray(drift)[None, :],
        bound,
    )
    density = (
        np.exp(np.asarray(upper_log_pdf))
        + np.exp(np.asarray(lower_log_pdf))
    ) / np.maximum(survival_after_trunc[None, :], 1e-12)
    return np.mean(density, axis=1)


def save_plot(samples, data, output_dir):
    rts = np.asarray(data["rts"], dtype=float)
    choices = np.asarray(data["choices"], dtype=float)
    evidence = np.asarray(data["evidence"], dtype=float)
    observed = rts > 0

    observed_evidence = evidence[observed]
    observed_choices = choices[observed]
    observed_rts = rts[observed]

    unique_evidence = np.sort(np.unique(observed_evidence))
    observed_p_right = np.array(
        [np.mean(observed_choices[observed_evidence == value] > 0) for value in unique_evidence]
    )
    trial_counts = np.array([np.sum(observed_evidence == value) for value in unique_evidence])

    evidence_grid = np.linspace(np.min(evidence), np.max(evidence), 200)
    predicted_p_right = fitted_choice_probability(evidence_grid, samples)

    time_grid = np.linspace(MIN_RT, min(T_MAX, np.percentile(observed_rts, 99.5)), 300)
    density = fitted_rt_density(time_grid, data, samples)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), dpi=150)

    axes[0].scatter(
        unique_evidence * EVIDENCE_SCALE,
        observed_p_right,
        s=10.0 + 0.15 * trial_counts,
        color="#35658A",
        alpha=0.75,
        label="Observed data",
    )
    axes[0].plot(
        evidence_grid * EVIDENCE_SCALE,
        predicted_p_right,
        color="#A63D40",
        linewidth=2.0,
        label="Fitted DDM",
    )
    axes[0].axhline(0.5, color="black", linewidth=0.8, alpha=0.4)
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_xlabel("Right dB - left dB")
    axes[0].set_ylabel("P(right choice)")
    axes[0].legend(frameon=False)

    bins = np.linspace(MIN_RT, min(T_MAX, np.percentile(observed_rts, 99.5)), 40)
    axes[1].hist(
        observed_rts,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.5,
        color="#35658A",
        label="Observed RTs",
    )
    axes[1].plot(time_grid, density, color="#A63D40", linewidth=2.0, label="Fitted DDM")
    axes[1].set_xlabel("Reaction time (s)")
    axes[1].set_ylabel("Probability density")
    axes[1].legend(frameon=False)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()

    plot_path = os.path.join(output_dir, "fit_plot_2afc.png")
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Saved plot to {plot_path}")


def prior_ranges_for_plot(data):
    min_rt = float(np.min(np.asarray(data["rts"], dtype=float)))
    max_ndt = max(min_rt - 0.01, 0.001)
    return {
        "drift_bias": DRIFT_BIAS_RANGE,
        "drift_sensitivity": DRIFT_SENSITIVITY_RANGE,
        "bound": BOUND_RANGE,
        "nondecision_time": (0.0, max_ndt),
    }


def corner_bins_from_posterior(stacked, plot_ranges, width_scale=2.5, fallback_bins=30):
    stacked = np.asarray(stacked, dtype=float)
    n_samples = int(stacked.shape[0])
    bins = []
    for idx, plot_range in enumerate(plot_ranges):
        values = stacked[:, idx]
        finite = values[np.isfinite(values)]
        if finite.size <= 1:
            bins.append(int(fallback_bins))
            continue

        sd = float(np.std(finite, ddof=1))
        range_width = float(plot_range[1] - plot_range[0])
        if not np.isfinite(sd) or sd <= 1e-8 or range_width <= 0.0:
            bins.append(int(fallback_bins))
            continue

        bin_width = float(width_scale) * sd * (float(n_samples) ** (-1.0 / 3.0))
        if not np.isfinite(bin_width) or bin_width <= 0.0:
            bins.append(int(fallback_bins))
            continue

        bins.append(max(1, int(np.ceil(range_width / bin_width))))
    return bins


def save_corner_plot(samples, data, output_dir):
    stacked = np.column_stack([np.asarray(samples[name]).reshape(-1) for name in PARAMETER_NAMES])
    if stacked.shape[0] <= stacked.shape[1]:
        print("Skipping corner plot: not enough posterior samples.")
        return None

    prior_ranges = prior_ranges_for_plot(data)
    plot_ranges = [prior_ranges[name] for name in PARAMETER_NAMES]
    plot_bins = corner_bins_from_posterior(stacked, plot_ranges)
    labels = [PARAMETER_LABELS[name] for name in PARAMETER_NAMES]

    side = 8.0
    fig = corner.corner(
        stacked,
        labels=labels,
        range=plot_ranges,
        bins=plot_bins,
        show_titles=True,
        quantiles=[0.16, 0.5, 0.84],
        color="tab:blue",
        alpha=0.6,
        fill_contours=True,
        quiet=True,
        label_kwargs={"fontsize": 9},
        title_kwargs={"fontsize": 9},
        fig=plt.figure(figsize=(side, side)),
    )

    axes = np.asarray(fig.axes, dtype=object).reshape((len(PARAMETER_NAMES), len(PARAMETER_NAMES)))
    for row, y_name in enumerate(PARAMETER_NAMES):
        for col, x_name in enumerate(PARAMETER_NAMES):
            if col > row:
                continue
            ax = axes[row, col]
            x_low, x_high = prior_ranges[x_name]
            ax.axvline(x_low, color="red", linestyle="--", linewidth=1.0, alpha=0.8)
            ax.axvline(x_high, color="red", linestyle="--", linewidth=1.0, alpha=0.8)
            if row != col:
                y_low, y_high = prior_ranges[y_name]
                ax.axhline(y_low, color="red", linestyle="--", linewidth=1.0, alpha=0.8)
                ax.axhline(y_high, color="red", linestyle="--", linewidth=1.0, alpha=0.8)

    fig.suptitle("Two-bound DDM posterior", fontsize=14, y=0.99)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.94)

    corner_path = os.path.join(output_dir, "corner_plot_2afc.png")
    fig.savefig(corner_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved corner plot to {corner_path}")
    return corner_path


def main():
    data = load_data()
    output_dir = make_output_dir()
    samples, fit_info = fit_model(data)
    save_summary(samples, data, output_dir, fit_info)
    save_plot(samples, data, output_dir)
    save_corner_plot(samples, data, output_dir)
    print(f"Done. Outputs are in {output_dir}")


if __name__ == "__main__":
    main()
