# %%
"""
Compare original early-stopped big SVI fits against no-early-stop audit reruns.

The scientific question is whether posterior means would have changed
materially if the original `stable_3_windows` criterion had not stopped the
fit. The comparison therefore focuses on post-stop ELBO gain per trial and
posterior-mean movement for Gamma, Omega, t_E_aff, w, and del_go.
"""

# %%
# =============================================================================
# Parameters
# =============================================================================
from pathlib import Path
import os
import pickle
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import svi_gamma_omega_likelihood_utils as likelihood_utils


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
ORIGINAL_OUTPUT_ROOT = SCRIPT_DIR / "svi_big_gamma_omega_delay_all_animals_outputs"
AUDIT_OUTPUT_ROOT = (
    SCRIPT_DIR
    / "svi_big_gamma_omega_delay_convergence_audit_outputs"
    / "no_early_stop_50k"
)
COMPARISON_DIR = AUDIT_OUTPUT_ROOT / "comparison_summary"
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

SELECTED_PAIRS = [
    ("SD", 49),
    ("LED8", 105),
    ("SD", 52),
    ("LED6", 81),
    ("LED34", 63),
    ("LED7", 103),
]

LOSS_METRICS_CSV = COMPARISON_DIR / "convergence_audit_loss_metrics.csv"
PARAM_SHIFT_CSV = COMPARISON_DIR / "convergence_audit_condition_parameter_shifts.csv"
PARAM_SHIFT_SUMMARY_CSV = COMPARISON_DIR / "convergence_audit_condition_parameter_shift_summary.csv"
SCALAR_SHIFT_CSV = COMPARISON_DIR / "convergence_audit_scalar_parameter_shifts.csv"
ANIMAL_CRITERIA_CSV = COMPARISON_DIR / "convergence_audit_animal_criteria_summary.csv"

LOSS_FIG = COMPARISON_DIR / "big_gamma_omega_delay_convergence_audit_loss_continuation.png"
CURVES_FIG = COMPARISON_DIR / "big_gamma_omega_delay_convergence_audit_gamma_omega_condition_curves.png"

TRACE_MATCH_ATOL = 1e-4
POST_STOP_GAIN_PER_TRIAL_LIMIT = 0.005
LOGLIKE_DELTA_PER_TRIAL_LIMIT = 0.005
MEDIAN_STANDARD_SHIFT_LIMIT = 0.1
P95_STANDARD_SHIFT_LIMIT = 0.5
SCALAR_STANDARD_SHIFT_LIMIT = 0.25

PARAM_SPECS = [
    {
        "parameter": "gamma",
        "label": "Gamma",
        "mean_col": "gamma_mean",
        "sd_col": "gamma_sd",
        "raw_scale": 1.0,
        "raw_unit": "",
    },
    {
        "parameter": "omega",
        "label": "Omega",
        "mean_col": "omega_mean",
        "sd_col": "omega_sd",
        "raw_scale": 1.0,
        "raw_unit": "",
    },
    {
        "parameter": "t_E_aff",
        "label": "t_E_aff",
        "mean_col": "t_E_aff_mean",
        "sd_col": "t_E_aff_sd",
        "raw_scale": 1000.0,
        "raw_unit": "ms",
    },
]
SCALAR_SPECS = [
    {"parameter": "w", "label": "w", "raw_scale": 1.0, "raw_unit": ""},
    {"parameter": "del_go", "label": "del_go", "raw_scale": 1000.0, "raw_unit": "ms"},
]
COLORS = {"gamma": "tab:blue", "omega": "tab:orange", "t_E_aff": "tab:green"}
SCALAR_COLORS = {"w": "tab:purple", "del_go": "tab:red"}
ABL_COLORS = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}
BATCH_T_TRUNC = {"LED34_even": 0.15}
DEFAULT_T_TRUNC = 0.3
K_MAX = 10


# %%
# =============================================================================
# Helpers
# =============================================================================
def animal_prefix(batch: str, animal: int) -> str:
    return f"{batch}_{animal}_big_gamma_omega_delay"


def animal_dir(root: Path, batch: str, animal: int) -> Path:
    return root / f"{batch}_{animal}"


def paths_for(root: Path, batch: str, animal: int):
    base_dir = animal_dir(root, batch, animal)
    prefix = animal_prefix(batch, animal)
    return {
        "dir": base_dir,
        "condition_summary": base_dir / f"{prefix}_condition_summary.csv",
        "posterior_summary": base_dir / f"{prefix}_posterior_summary.csv",
        "loss": base_dir / f"{prefix}_loss.csv",
        "convergence": base_dir / f"{prefix}_convergence_checks.csv",
        "condition_table": base_dir / f"{prefix}_condition_table.csv",
    }


def load_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def finite_or_nan(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return finite if finite.size else np.array([np.nan])


def summarize_shift(values):
    finite = finite_or_nan(values)
    return {
        "median": float(np.nanmedian(finite)),
        "p95": float(np.nanpercentile(finite, 95)),
        "max": float(np.nanmax(finite)),
    }


def batch_csv_path(batch: str) -> Path:
    return REPO_DIR / "raw_data" / "batch_csvs" / f"batch_{batch}_valid_and_aborts.csv"


def load_trial_data(batch: str, animal: int, condition_table: pd.DataFrame, abort_means: dict):
    raw_df = pd.read_csv(batch_csv_path(batch))
    if "choice" not in raw_df.columns:
        if "response_poke" not in raw_df.columns:
            raise KeyError("Need either `choice` or `response_poke` in the batch CSV.")
        raw_df["choice"] = raw_df["response_poke"].map({3: 1, 2: -1})

    total_fix_col = "timed_fix" if "timed_fix" in raw_df.columns else "TotalFixTime"
    valid_df = raw_df[
        (raw_df["animal"].astype(int) == int(animal))
        & (raw_df["success"].isin([1, -1]))
        & (raw_df["RTwrtStim"] > 0)
        & (raw_df["RTwrtStim"] <= 1)
    ].copy()
    valid_df = valid_df.dropna(subset=[total_fix_col, "intended_fix", "ABL", "ILD", "choice", "RTwrtStim"])
    valid_df["ABL"] = valid_df["ABL"].astype(int)
    valid_df["ILD"] = valid_df["ILD"].astype(int)
    valid_df["choice"] = valid_df["choice"].astype(int)

    lookup = condition_table[["ABL", "ILD", "condition_id"]].copy()
    lookup["ABL"] = lookup["ABL"].astype(int)
    lookup["ILD"] = lookup["ILD"].astype(int)
    lookup["condition_id"] = lookup["condition_id"].astype(int)
    valid_df = valid_df.merge(lookup, on=["ABL", "ILD"], how="left", validate="many_to_one")
    valid_df = valid_df.dropna(subset=["condition_id"]).copy()
    valid_df["condition_id"] = valid_df["condition_id"].astype(int)

    return {
        "total_fix": jnp.asarray(valid_df[total_fix_col].to_numpy(dtype=float)),
        "t_stim": jnp.asarray(valid_df["intended_fix"].to_numpy(dtype=float)),
        "choice": jnp.asarray(valid_df["choice"].to_numpy(dtype=int)),
        "condition_id": jnp.asarray(valid_df["condition_id"].to_numpy(dtype=int)),
        "mask": jnp.ones(len(valid_df), dtype=bool),
        "V_A": jnp.asarray(float(abort_means["V_A"]), dtype=jnp.float64),
        "theta_A": jnp.asarray(float(abort_means["theta_A"]), dtype=jnp.float64),
        "t_A_aff": jnp.asarray(float(abort_means["t_A_aff"]), dtype=jnp.float64),
        "T_trunc": jnp.asarray(BATCH_T_TRUNC.get(batch, DEFAULT_T_TRUNC), dtype=jnp.float64),
    }, int(len(valid_df))


def params_from_summaries(condition_df: pd.DataFrame, scalar_df: pd.DataFrame):
    condition_df = condition_df.sort_values("condition_id").reset_index(drop=True)
    scalar_by_name = scalar_df.set_index("parameter")
    return {
        "gamma": jnp.asarray(condition_df["gamma_mean"].to_numpy(dtype=float), dtype=jnp.float64),
        "omega": jnp.asarray(condition_df["omega_mean"].to_numpy(dtype=float), dtype=jnp.float64),
        "t_E_aff": jnp.asarray(condition_df["t_E_aff_mean"].to_numpy(dtype=float), dtype=jnp.float64),
        "w": jnp.asarray(float(scalar_by_name.loc["w", "mean"]), dtype=jnp.float64),
        "del_go": jnp.asarray(float(scalar_by_name.loc["del_go", "mean"]), dtype=jnp.float64),
    }


# %%
# =============================================================================
# Load and compare loss traces
# =============================================================================
loss_metric_rows = []
condition_shift_rows = []
scalar_shift_rows = []
loss_plot_payload = []

for batch, animal in SELECTED_PAIRS:
    label = f"{batch}/{animal}"
    original_paths = paths_for(ORIGINAL_OUTPUT_ROOT, batch, animal)
    audit_paths = paths_for(AUDIT_OUTPUT_ROOT, batch, animal)

    original_loss = load_csv(original_paths["loss"])
    audit_loss = load_csv(audit_paths["loss"])
    original_conv = load_csv(original_paths["convergence"])
    audit_conv = load_csv(audit_paths["convergence"])
    original_condition = load_csv(original_paths["condition_summary"])
    audit_condition = load_csv(audit_paths["condition_summary"])
    original_scalar = load_csv(original_paths["posterior_summary"])
    audit_scalar = load_csv(audit_paths["posterior_summary"])
    condition_table = load_csv(original_paths["condition_table"])

    original_stop_step = int(original_conv["end_step"].max())
    n_original_loss = len(original_loss)
    if n_original_loss != original_stop_step:
        raise RuntimeError(f"{label}: original loss rows ({n_original_loss}) != stop step ({original_stop_step})")
    if len(audit_loss) < n_original_loss:
        raise RuntimeError(f"{label}: audit loss has fewer rows than original.")

    original_loss_values = original_loss["loss"].to_numpy(dtype=float)
    audit_prefix_loss_values = audit_loss["loss"].iloc[:n_original_loss].to_numpy(dtype=float)
    trace_diff = audit_prefix_loss_values - original_loss_values
    trace_max_abs_diff = float(np.max(np.abs(trace_diff)))
    trace_rmse = float(np.sqrt(np.mean(trace_diff**2)))
    trace_matches = bool(trace_max_abs_diff <= TRACE_MATCH_ATOL)

    n_trials = int(original_condition["n_trials"].sum())
    with (ORIGINAL_OUTPUT_ROOT / f"{batch}_{animal}" / f"{animal_prefix(batch, animal)}_fit_bundle.pkl").open("rb") as handle:
        original_bundle = pickle.load(handle)
    data, n_loglike_trials = load_trial_data(batch, animal, condition_table, original_bundle["abort_means"])
    if n_loglike_trials != n_trials:
        raise RuntimeError(f"{label}: loglike trial count {n_loglike_trials} != condition-summary count {n_trials}")

    original_params = params_from_summaries(original_condition, original_scalar)
    audit_params = params_from_summaries(audit_condition, audit_scalar)
    original_posterior_mean_loglike = float(
        likelihood_utils.gamma_omega_delay_loglike(original_params, data, K_MAX)
    )
    audit_posterior_mean_loglike = float(
        likelihood_utils.gamma_omega_delay_loglike(audit_params, data, K_MAX)
    )
    delta_posterior_mean_loglike = audit_posterior_mean_loglike - original_posterior_mean_loglike
    delta_posterior_mean_loglike_per_trial = delta_posterior_mean_loglike / n_trials

    original_final_window_mean = float(original_conv.iloc[-1]["mean_loss"])
    audit_final_window_mean = float(audit_conv.iloc[-1]["mean_loss"])
    audit_best_window_mean = float(audit_conv["mean_loss"].min())
    audit_best_end_step = int(audit_conv.loc[audit_conv["mean_loss"].idxmin(), "end_step"])

    post_stop_conv = audit_conv[audit_conv["end_step"] > original_stop_step].copy()
    if len(post_stop_conv):
        post_stop_best_mean = float(post_stop_conv["mean_loss"].min())
        post_stop_best_end_step = int(post_stop_conv.loc[post_stop_conv["mean_loss"].idxmin(), "end_step"])
        post_stop_gain = original_final_window_mean - post_stop_best_mean
        post_stop_gain_per_trial = post_stop_gain / n_trials
    else:
        post_stop_best_mean = np.nan
        post_stop_best_end_step = np.nan
        post_stop_gain = np.nan
        post_stop_gain_per_trial = np.nan

    final_window = audit_conv.iloc[-1]
    ten_k_prior = audit_conv[audit_conv["end_step"] <= int(final_window["end_step"]) - 10000]
    if len(ten_k_prior):
        ten_k_reference = float(ten_k_prior.iloc[-1]["mean_loss"])
        final_10k_gain = ten_k_reference - audit_best_window_mean
        final_10k_gain_per_trial = final_10k_gain / n_trials
    else:
        final_10k_gain = np.nan
        final_10k_gain_per_trial = np.nan

    loss_metric_rows.append(
        {
            "batch_name": batch,
            "animal": animal,
            "animal_label": label,
            "n_trials": n_trials,
            "original_stop_step": original_stop_step,
            "audit_end_step": int(final_window["end_step"]),
            "trace_matches_to_original_stop": trace_matches,
            "trace_max_abs_diff": trace_max_abs_diff,
            "trace_rmse": trace_rmse,
            "original_final_window_mean_loss": original_final_window_mean,
            "audit_final_window_mean_loss": audit_final_window_mean,
            "audit_best_window_mean_loss": audit_best_window_mean,
            "audit_best_end_step": audit_best_end_step,
            "post_stop_best_mean_loss": post_stop_best_mean,
            "post_stop_best_end_step": post_stop_best_end_step,
            "post_stop_gain": post_stop_gain,
            "post_stop_gain_per_trial": post_stop_gain_per_trial,
            "original_posterior_mean_loglike": original_posterior_mean_loglike,
            "audit_posterior_mean_loglike": audit_posterior_mean_loglike,
            "delta_posterior_mean_loglike": delta_posterior_mean_loglike,
            "delta_posterior_mean_loglike_per_trial": delta_posterior_mean_loglike_per_trial,
            "audit_final_relative_change_pct": float(final_window["relative_mean_change"]) * 100.0,
            "audit_final_slope_per_1000_steps": float(final_window["slope_per_1000_steps"]),
            "audit_final_10k_gain": final_10k_gain,
            "audit_final_10k_gain_per_trial": final_10k_gain_per_trial,
        }
    )

    loss_plot_payload.append(
        {
            "label": label,
            "original_stop_step": original_stop_step,
            "audit_conv": audit_conv.copy(),
            "original_conv": original_conv.copy(),
            "post_stop_gain_per_trial": post_stop_gain_per_trial,
            "delta_loglike_per_trial": delta_posterior_mean_loglike_per_trial,
            "trace_matches": trace_matches,
        }
    )

    merge_cols = ["batch_name", "animal", "condition_id", "ABL", "ILD"]
    merged = original_condition.merge(
        audit_condition,
        on=merge_cols,
        suffixes=("_original", "_audit"),
        validate="one_to_one",
    )
    if len(merged) != len(original_condition):
        raise RuntimeError(f"{label}: condition merge lost rows.")

    for spec in PARAM_SPECS:
        original_mean = merged[f"{spec['mean_col']}_original"].to_numpy(dtype=float)
        audit_mean = merged[f"{spec['mean_col']}_audit"].to_numpy(dtype=float)
        original_sd = merged[f"{spec['sd_col']}_original"].to_numpy(dtype=float)
        raw_shift = (audit_mean - original_mean) * spec["raw_scale"]
        abs_raw_shift = np.abs(raw_shift)
        standardized_shift = np.abs(audit_mean - original_mean) / np.where(original_sd > 0, original_sd, np.nan)

        for idx, row in merged.iterrows():
            condition_shift_rows.append(
                {
                    "batch_name": batch,
                    "animal": animal,
                    "animal_label": label,
                    "condition_id": int(row.condition_id),
                    "ABL": int(row.ABL),
                    "ILD": int(row.ILD),
                    "parameter": spec["parameter"],
                    "original_mean": float(original_mean[idx]),
                    "audit_mean": float(audit_mean[idx]),
                    "raw_shift": float(raw_shift[idx]),
                    "abs_raw_shift": float(abs_raw_shift[idx]),
                    "raw_unit": spec["raw_unit"],
                    "original_sd": float(original_sd[idx]),
                    "standardized_shift": float(standardized_shift[idx]),
                }
            )

    for spec in SCALAR_SPECS:
        original_row = original_scalar[original_scalar["parameter"] == spec["parameter"]]
        audit_row = audit_scalar[audit_scalar["parameter"] == spec["parameter"]]
        if len(original_row) != 1 or len(audit_row) != 1:
            raise RuntimeError(f"{label}: expected one scalar row for {spec['parameter']}")
        original_row = original_row.iloc[0]
        audit_row = audit_row.iloc[0]
        original_mean = float(original_row["mean"])
        audit_mean = float(audit_row["mean"])
        original_sd = float(original_row["sd"])
        raw_shift = (audit_mean - original_mean) * spec["raw_scale"]
        standardized_shift = abs(audit_mean - original_mean) / original_sd if original_sd > 0 else np.nan
        scalar_shift_rows.append(
            {
                "batch_name": batch,
                "animal": animal,
                "animal_label": label,
                "parameter": spec["parameter"],
                "original_mean": original_mean,
                "audit_mean": audit_mean,
                "raw_shift": raw_shift,
                "abs_raw_shift": abs(raw_shift),
                "raw_unit": spec["raw_unit"],
                "original_sd": original_sd,
                "standardized_shift": standardized_shift,
            }
        )


# %%
# =============================================================================
# Summaries and pass/fail criteria
# =============================================================================
loss_metrics_df = pd.DataFrame(loss_metric_rows)
condition_shift_df = pd.DataFrame(condition_shift_rows)
scalar_shift_df = pd.DataFrame(scalar_shift_rows)

summary_rows = []
for (batch, animal, label, parameter), group in condition_shift_df.groupby(
    ["batch_name", "animal", "animal_label", "parameter"],
    sort=False,
):
    standardized_summary = summarize_shift(group["standardized_shift"])
    raw_summary = summarize_shift(group["abs_raw_shift"])
    summary_rows.append(
        {
            "batch_name": batch,
            "animal": int(animal),
            "animal_label": label,
            "parameter": parameter,
            "n_conditions": int(len(group)),
            "median_standardized_shift": standardized_summary["median"],
            "p95_standardized_shift": standardized_summary["p95"],
            "max_standardized_shift": standardized_summary["max"],
            "median_abs_raw_shift": raw_summary["median"],
            "p95_abs_raw_shift": raw_summary["p95"],
            "max_abs_raw_shift": raw_summary["max"],
        }
    )
condition_shift_summary_df = pd.DataFrame(summary_rows)

criteria_rows = []
for batch, animal in SELECTED_PAIRS:
    label = f"{batch}/{animal}"
    loss_row = loss_metrics_df[loss_metrics_df["animal_label"] == label].iloc[0]
    param_group = condition_shift_summary_df[condition_shift_summary_df["animal_label"] == label]
    scalar_group = scalar_shift_df[scalar_shift_df["animal_label"] == label]

    median_ok = bool(np.nanmax(param_group["median_standardized_shift"]) < MEDIAN_STANDARD_SHIFT_LIMIT)
    p95_ok = bool(np.nanmax(param_group["p95_standardized_shift"]) < P95_STANDARD_SHIFT_LIMIT)
    scalar_ok = bool(np.nanmax(scalar_group["standardized_shift"]) < SCALAR_STANDARD_SHIFT_LIMIT)
    loss_ok = bool(loss_row["post_stop_gain_per_trial"] < POST_STOP_GAIN_PER_TRIAL_LIMIT)
    loglike_ok = bool(abs(loss_row["delta_posterior_mean_loglike_per_trial"]) < LOGLIKE_DELTA_PER_TRIAL_LIMIT)
    trace_ok = bool(loss_row["trace_matches_to_original_stop"])
    criteria_rows.append(
        {
            "batch_name": batch,
            "animal": animal,
            "animal_label": label,
            "trace_matches_to_original_stop": trace_ok,
            "post_stop_gain_per_trial": float(loss_row["post_stop_gain_per_trial"]),
            "loss_gain_ok": loss_ok,
            "delta_posterior_mean_loglike_per_trial": float(loss_row["delta_posterior_mean_loglike_per_trial"]),
            "loglike_delta_ok": loglike_ok,
            "max_median_standardized_shift": float(np.nanmax(param_group["median_standardized_shift"])),
            "median_shift_ok": median_ok,
            "max_p95_standardized_shift": float(np.nanmax(param_group["p95_standardized_shift"])),
            "p95_shift_ok": p95_ok,
            "max_scalar_standardized_shift": float(np.nanmax(scalar_group["standardized_shift"])),
            "scalar_shift_ok": scalar_ok,
            "all_criteria_ok": bool(trace_ok and loss_ok and loglike_ok and median_ok and p95_ok and scalar_ok),
        }
    )
criteria_df = pd.DataFrame(criteria_rows)

loss_metrics_df.to_csv(LOSS_METRICS_CSV, index=False)
condition_shift_df.to_csv(PARAM_SHIFT_CSV, index=False)
condition_shift_summary_df.to_csv(PARAM_SHIFT_SUMMARY_CSV, index=False)
scalar_shift_df.to_csv(SCALAR_SHIFT_CSV, index=False)
criteria_df.to_csv(ANIMAL_CRITERIA_CSV, index=False)

print("Saved CSVs:")
for path in [LOSS_METRICS_CSV, PARAM_SHIFT_CSV, PARAM_SHIFT_SUMMARY_CSV, SCALAR_SHIFT_CSV, ANIMAL_CRITERIA_CSV]:
    print(f"  {path}")

print("\nCriteria summary:")
print(criteria_df.to_string(index=False))


# %%
# =============================================================================
# Plot loss continuation
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 7.8), sharex=False, sharey=False)
axes_flat = axes.ravel()
for ax, payload in zip(axes_flat, loss_plot_payload):
    audit_conv = payload["audit_conv"]
    original_conv = payload["original_conv"]
    label = payload["label"]
    ax.plot(
        audit_conv["end_step"],
        audit_conv["mean_loss"],
        color="tab:blue",
        linewidth=1.0,
        linestyle="-",
        label="late stop (50k)",
    )
    ax.plot(
        original_conv["end_step"],
        original_conv["mean_loss"],
        color="tab:red",
        linewidth=3.0,
        linestyle="--",
        alpha=0.4,
        label="early stop",
    )
    ax.axvline(payload["original_stop_step"], color="tab:red", linestyle=":", linewidth=1.2)
    gain_text = f"post-stop gain/trial={payload['post_stop_gain_per_trial']:.4g}"
    loglike_text = f"delta LL/trial={payload['delta_loglike_per_trial']:.4g}"
    match_text = "trace ok" if payload["trace_matches"] else "trace mismatch"
    ax.text(
        0.98,
        0.96,
        f"{gain_text}\n{loglike_text}\n{match_text}",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        ha="right",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "0.85", "alpha": 0.9},
    )
    ax.set_title(label)
    ax.set_xlabel("SVI step")
    ax.set_ylabel("window mean negative ELBO")
    ax.grid(True, alpha=0.25)
loss_legend_handles = [
    Line2D([0], [0], color="tab:blue", lw=1.0, linestyle="-", label="late stop (50k)"),
    Line2D([0], [0], color="tab:red", lw=3.0, linestyle="--", alpha=0.4, label="early stop"),
]
fig.legend(
    handles=loss_legend_handles,
    frameon=False,
    fontsize=9,
    loc="upper right",
    bbox_to_anchor=(0.995, 1.02),
    ncol=2,
)
fig.suptitle("No-early-stop big Gamma/Omega/delay SVI convergence audit", y=1.02)
fig.tight_layout()
fig.savefig(LOSS_FIG, dpi=200, bbox_inches="tight")
print(f"Saved loss figure: {LOSS_FIG}")


# %%
# =============================================================================
# Plot condition-wise Gamma/Omega curves before and after early stopping
# =============================================================================
animal_labels = [f"{batch}/{animal}" for batch, animal in SELECTED_PAIRS]

fig, axes = plt.subplots(2, 6, figsize=(22, 7.0), sharex=False, sharey="row")
curve_specs = [
    ("gamma", "Gamma", "gamma"),
    ("omega", "Omega", "omega"),
]
for col_idx, label in enumerate(animal_labels):
    for row_idx, (parameter, y_label, title_prefix) in enumerate(curve_specs):
        ax = axes[row_idx, col_idx]
        rows = condition_shift_df[
            (condition_shift_df["animal_label"] == label)
            & (condition_shift_df["parameter"] == parameter)
        ].copy()
        for abl in sorted(rows["ABL"].unique()):
            abl_rows = rows[rows["ABL"] == abl].sort_values("ILD")
            color = ABL_COLORS.get(int(abl), "0.3")
            ax.plot(
                abl_rows["ILD"],
                abl_rows["audit_mean"],
                color=color,
                linestyle="-",
                linewidth=1.0,
            )
            ax.plot(
                abl_rows["ILD"],
                abl_rows["original_mean"],
                color=color,
                linestyle="--",
                linewidth=3.0,
                alpha=0.4,
            )
        if row_idx == 0:
            ax.set_title(label)
        if col_idx == 0:
            ax.set_ylabel(y_label)
        if row_idx == 1:
            ax.set_xlabel("ILD")
        ax.grid(True, alpha=0.22)

legend_handles = [
    Line2D([0], [0], color=ABL_COLORS[20], lw=1.0, label="ABL 20"),
    Line2D([0], [0], color=ABL_COLORS[40], lw=1.0, label="ABL 40"),
    Line2D([0], [0], color=ABL_COLORS[60], lw=1.0, label="ABL 60"),
    Line2D([0], [0], color="0.15", lw=1.0, linestyle="-", label="late stop (50k)"),
    Line2D([0], [0], color="0.15", lw=3.0, linestyle="--", alpha=0.4, label="early stop"),
]
fig.legend(
    handles=legend_handles,
    frameon=False,
    fontsize=9,
    loc="upper right",
    bbox_to_anchor=(0.995, 1.02),
    ncol=5,
)
fig.suptitle("Gamma/Omega condition curves before and after disabling early stopping", y=1.04)
fig.tight_layout()
fig.savefig(CURVES_FIG, dpi=200, bbox_inches="tight")
print(f"Saved condition-curve figure: {CURVES_FIG}")

plt.show()

# %%
