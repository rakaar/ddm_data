# %%
"""
Fit normalized TIED parameters on LED-OFF data using proactive+lapse parameters loaded from
proactive VP pickles.

This aggregate-only variant keeps the original right-truncated no-choice likelihood, but when
the target truncation exceeds a reference truncation it subsamples the target trial pool so the
fitted trials match the reference intended-fix distribution within each (ABL, ILD) stratum.
"""

# %%
from pathlib import Path
import os
import pickle
import sys

import matplotlib
import matplotlib.pyplot as plt
import corner
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pyvbmc import VBMC
from scipy.stats import ks_2samp

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ANIMAL_WISE_DIR = REPO_ROOT / "fit_animal_by_animal"
if str(ANIMAL_WISE_DIR) not in sys.path:
    sys.path.insert(0, str(ANIMAL_WISE_DIR))

from proactive_plus_lapse_plus_reactive_uitls import (
    trial_logpdf_proactive_lapse_only_no_trunc_right_truncated_no_choice,
)
from vbmc_animal_wise_fit_utils import trapezoidal_logpdf
from animal_wise_config import T_trunc


# %%
############ Parameters (edit here) ############
batch_name = "LED7"
fit_mode = "aggregate"
save_corner_plot = os.getenv("MATCH_REF_STIM_SAVE_CORNER_PLOT", "1") == "1"
save_stim_match_plot = os.getenv("MATCH_REF_STIM_SAVE_MATCH_PLOT", "1") == "1"

session_type = 7
training_level = 16
allowed_repeat_trials = [0, 2]

max_rtwrtstim_for_fit = 1.0
n_jobs = int(os.getenv("MATCH_REF_STIM_N_JOBS", "30"))
posterior_sample_count = int(
    os.getenv("MATCH_REF_STIM_POSTERIOR_SAMPLE_COUNT", str(int(1e5)))
)
proactive_posterior_sample_count = int(
    os.getenv("MATCH_REF_STIM_PROACTIVE_POSTERIOR_SAMPLE_COUNT", str(int(2e4)))
)
vbmc_max_fun_evals = int(
    os.getenv("MATCH_REF_STIM_VBMC_MAX_FUN_EVALS", str(200 * (2 + 9)))
)

led_data_csv_path = REPO_ROOT / "out_LED.csv"
proactive_vp_aggregate_path = SCRIPT_DIR / "vbmc_real_all_animals_fit_NO_TRUNC_with_lapse.pkl"

output_dir = (
    SCRIPT_DIR
    / "norm_only_led_off_from_loaded_proactive_truncate_NOT_censor_ABL_delay_no_choice_match_ref_stim"
)
output_dir.mkdir(parents=True, exist_ok=True)

# ###### RUN TAG / STIM-MATCH CONFIG ######
K_max = 10
is_norm = True
is_time_vary = False
phi_params_obj = np.nan
reference_truncate_rt_wrt_stim_s = 0.115
truncate_rt_wrt_stim_s = 0.130
supported_abl_values = (20, 40, 60)

match_stim_distribution = True
stim_match_column = "intended_fix"
stim_match_group_cols = ("ABL", "ILD")
stim_match_quantile_bins = 10
stim_match_min_trials_per_bin = 5
stim_match_seed = 12345

if fit_mode != "aggregate":
    raise ValueError(
        "This script supports only fit_mode='aggregate'. "
        f"Current fit_mode={fit_mode!r}."
    )
if reference_truncate_rt_wrt_stim_s <= 0:
    raise ValueError("reference_truncate_rt_wrt_stim_s must be positive.")
if truncate_rt_wrt_stim_s <= 0:
    raise ValueError("truncate_rt_wrt_stim_s must be positive.")
if reference_truncate_rt_wrt_stim_s > max_rtwrtstim_for_fit:
    raise ValueError("reference_truncate_rt_wrt_stim_s cannot exceed max_rtwrtstim_for_fit.")
if truncate_rt_wrt_stim_s > max_rtwrtstim_for_fit:
    raise ValueError("truncate_rt_wrt_stim_s cannot exceed max_rtwrtstim_for_fit.")
if truncate_rt_wrt_stim_s < reference_truncate_rt_wrt_stim_s:
    raise ValueError(
        "truncate_rt_wrt_stim_s must be greater than or equal to "
        "reference_truncate_rt_wrt_stim_s."
    )
if stim_match_quantile_bins <= 0:
    raise ValueError("stim_match_quantile_bins must be positive.")
if stim_match_min_trials_per_bin <= 0:
    raise ValueError("stim_match_min_trials_per_bin must be positive.")


# %%
############ Global vars used by likelihood callbacks ############
V_A = np.nan
theta_A = np.nan
t_A_aff = np.nan
lapse_prob = np.nan
beta_lapse = np.nan
df_valid_animal_truncated = pd.DataFrame()


# %%
############ Helpers ############
def get_t_E_aff_from_abl(abl, t_E_aff_20, t_E_aff_40, t_E_aff_60):
    abl_value = float(abl)
    if np.isclose(abl_value, 20.0):
        return t_E_aff_20
    if np.isclose(abl_value, 40.0):
        return t_E_aff_40
    if np.isclose(abl_value, 60.0):
        return t_E_aff_60
    raise ValueError(
        f"Unsupported ABL value {abl_value}. Expected one of {supported_abl_values}."
    )


def validate_supported_abl_values(df, df_name):
    observed = np.sort(df["ABL"].dropna().astype(float).unique())
    if len(observed) == 0:
        raise ValueError(f"No ABL values found in {df_name}.")

    unexpected = [
        float(abl)
        for abl in observed
        if not any(np.isclose(float(abl), float(supported)) for supported in supported_abl_values)
    ]
    if unexpected:
        raise ValueError(
            f"Unexpected ABL values in {df_name}: {unexpected}. "
            f"Supported values are {supported_abl_values}."
        )
    return observed


def normalize_numeric_key(value):
    value = float(value)
    if np.isfinite(value) and np.isclose(value, np.round(value)):
        return int(np.round(value))
    return float(value)


def format_column_counts(df, column):
    series = df[column].dropna().astype(float)
    counts = series.value_counts().sort_index().to_dict()
    return {normalize_numeric_key(k): int(v) for k, v in counts.items()}


def format_group_key(group_cols, group_key):
    if not isinstance(group_key, tuple):
        group_key = (group_key,)
    key_parts = [
        f"{column}={normalize_numeric_key(value)}"
        for column, value in zip(group_cols, group_key)
    ]
    return "|".join(key_parts)


def format_group_counts(df, group_cols):
    if len(df) == 0:
        return {}
    group_sizes = df.groupby(list(group_cols), dropna=False, sort=True).size()
    return {
        format_group_key(group_cols, group_key): int(count)
        for group_key, count in group_sizes.items()
    }


def build_run_tag(
    truncate_rt_wrt_stim_s,
    reference_truncate_rt_wrt_stim_s,
    match_stim_distribution,
    stim_match_group_cols,
    stim_match_quantile_bins,
):
    truncate_ms = int(round(float(truncate_rt_wrt_stim_s) * 1e3))
    reference_ms = int(round(float(reference_truncate_rt_wrt_stim_s) * 1e3))
    if not match_stim_distribution:
        return f"trunc{truncate_ms}ms_ref{reference_ms}ms_nomatch"

    group_tag = "_".join(str(column) for column in stim_match_group_cols)
    return (
        f"trunc{truncate_ms}ms_ref{reference_ms}ms_matchStim_"
        f"{group_tag}_q{int(stim_match_quantile_bins)}"
    )


def compute_ks_stats(left_values, right_values):
    left_values = np.asarray(left_values, dtype=np.float64)
    right_values = np.asarray(right_values, dtype=np.float64)
    if len(left_values) == 0 or len(right_values) == 0:
        return {
            "n_left": int(len(left_values)),
            "n_right": int(len(right_values)),
            "statistic": None,
            "pvalue": None,
            "statistic_location": None,
            "statistic_sign": None,
        }

    ks_result = ks_2samp(left_values, right_values)
    return {
        "n_left": int(len(left_values)),
        "n_right": int(len(right_values)),
        "statistic": float(ks_result.statistic),
        "pvalue": float(ks_result.pvalue),
        "statistic_location": (
            float(getattr(ks_result, "statistic_location"))
            if getattr(ks_result, "statistic_location", None) is not None
            else None
        ),
        "statistic_sign": (
            int(getattr(ks_result, "statistic_sign"))
            if getattr(ks_result, "statistic_sign", None) is not None
            else None
        ),
    }


def compute_ks_audit(reference_df, other_df, column):
    audit = {
        "overall": compute_ks_stats(reference_df[column].to_numpy(), other_df[column].to_numpy()),
        "by_abl": {},
    }
    for abl in supported_abl_values:
        ref_values = reference_df.loc[np.isclose(reference_df["ABL"], float(abl)), column].to_numpy()
        other_values = other_df.loc[np.isclose(other_df["ABL"], float(abl)), column].to_numpy()
        audit["by_abl"][int(abl)] = compute_ks_stats(ref_values, other_values)
    return audit


def get_candidate_groups(candidate_df, group_cols):
    return {
        group_key: group_df.copy()
        for group_key, group_df in candidate_df.groupby(list(group_cols), sort=True, dropna=False)
    }


def sample_candidate_trials_to_match_reference(
    reference_df,
    candidate_df,
    match_column,
    group_cols,
    quantile_bins,
    min_trials_per_bin,
    seed,
):
    if match_column not in reference_df.columns:
        raise KeyError(f"Missing match column {match_column!r} in reference_df.")
    if match_column not in candidate_df.columns:
        raise KeyError(f"Missing match column {match_column!r} in candidate_df.")
    for column in group_cols:
        if column not in reference_df.columns:
            raise KeyError(f"Missing group column {column!r} in reference_df.")
        if column not in candidate_df.columns:
            raise KeyError(f"Missing group column {column!r} in candidate_df.")
        if reference_df[column].isna().any():
            raise ValueError(f"Reference data contains NaN values in group column {column!r}.")
        if candidate_df[column].isna().any():
            raise ValueError(f"Candidate data contains NaN values in group column {column!r}.")
    if reference_df[match_column].isna().any():
        raise ValueError(f"Reference data contains NaN values in match column {match_column!r}.")
    if candidate_df[match_column].isna().any():
        raise ValueError(f"Candidate data contains NaN values in match column {match_column!r}.")

    rng = np.random.default_rng(seed)
    candidate_groups = get_candidate_groups(candidate_df, group_cols)
    matched_frames = []
    strata_audit = []

    grouped_reference = reference_df.groupby(list(group_cols), sort=True, dropna=False)
    for group_key, reference_group in grouped_reference:
        candidate_group = candidate_groups.get(group_key)
        group_label = format_group_key(group_cols, group_key)
        n_reference = int(len(reference_group))

        if candidate_group is None:
            raise ValueError(
                "Stimulus matching requires every reference stratum to exist in the candidate pool. "
                f"Missing stratum: {group_label}."
            )

        n_candidate = int(len(candidate_group))
        if n_candidate < n_reference:
            raise ValueError(
                "Stimulus matching requires the candidate pool to have at least as many trials as the "
                f"reference pool in every stratum. Stratum {group_label}: "
                f"reference={n_reference}, candidate={n_candidate}."
            )

        max_bins_for_group = min(
            int(quantile_bins),
            max(1, int(np.floor(n_reference / float(min_trials_per_bin)))),
        )
        quantile_levels = np.linspace(0.0, 1.0, max_bins_for_group + 1)
        bin_edges = np.quantile(reference_group[match_column].to_numpy(dtype=np.float64), quantile_levels)
        bin_edges = np.unique(np.asarray(bin_edges, dtype=np.float64))

        stratum_audit = {
            "group": group_label,
            "n_reference": n_reference,
            "n_candidate": n_candidate,
            "selected_count": n_reference,
            "requested_quantile_bins": int(quantile_bins),
            "effective_quantile_bins": 1,
            "sampling_mode": "uniform_without_bins",
            "bin_edges": None,
            "bin_counts": None,
        }

        if len(bin_edges) < 2:
            sampled_indices = rng.choice(
                candidate_group.index.to_numpy(),
                size=n_reference,
                replace=False,
            )
            matched_group = candidate_group.loc[np.sort(sampled_indices)].copy()
            matched_frames.append(matched_group)
            strata_audit.append(stratum_audit)
            continue

        reference_bins = pd.cut(
            reference_group[match_column],
            bins=bin_edges,
            include_lowest=True,
            right=True,
        )
        candidate_bins = pd.cut(
            candidate_group[match_column],
            bins=bin_edges,
            include_lowest=True,
            right=True,
        )
        reference_bin_counts = reference_bins.value_counts(sort=False)
        candidate_bin_counts = candidate_bins.value_counts(sort=False)

        selected_indices = []
        bin_count_audit = []
        for interval, target_count in reference_bin_counts.items():
            target_count = int(target_count)
            candidate_interval_indices = candidate_group.index[candidate_bins == interval].to_numpy()
            candidate_count = int(len(candidate_interval_indices))
            if candidate_count < target_count:
                raise ValueError(
                    "Stimulus matching failed because a candidate bin has fewer trials than the "
                    "reference bin count. "
                    f"Stratum {group_label}, interval={interval}, "
                    f"reference={target_count}, candidate={candidate_count}."
                )

            chosen_indices = rng.choice(
                candidate_interval_indices,
                size=target_count,
                replace=False,
            )
            selected_indices.extend(chosen_indices.tolist())
            bin_count_audit.append(
                {
                    "interval": {
                        "left": float(interval.left),
                        "right": float(interval.right),
                        "closed": interval.closed,
                    },
                    "reference_count": target_count,
                    "candidate_count": int(candidate_bin_counts.loc[interval]),
                    "selected_count": target_count,
                }
            )

        matched_group = candidate_group.loc[np.sort(np.asarray(selected_indices))].copy()
        matched_frames.append(matched_group)

        stratum_audit["effective_quantile_bins"] = int(len(bin_edges) - 1)
        stratum_audit["sampling_mode"] = "quantile_bin_match"
        stratum_audit["bin_edges"] = [float(edge) for edge in bin_edges.tolist()]
        stratum_audit["bin_counts"] = bin_count_audit
        strata_audit.append(stratum_audit)

    matched_df = pd.concat(matched_frames, axis=0).sort_index().copy()
    return matched_df, strata_audit


def build_matching_audit(reference_df, candidate_df, matched_df, strata_audit):
    return {
        "match_enabled": bool(match_stim_distribution),
        "match_column": stim_match_column,
        "match_group_cols": list(stim_match_group_cols),
        "reference_truncate_rt_wrt_stim_s": float(reference_truncate_rt_wrt_stim_s),
        "target_truncate_rt_wrt_stim_s": float(truncate_rt_wrt_stim_s),
        "quantile_bins_requested": int(stim_match_quantile_bins),
        "min_trials_per_bin": int(stim_match_min_trials_per_bin),
        "seed": int(stim_match_seed),
        "reference_trial_count": int(len(reference_df)),
        "candidate_trial_count": int(len(candidate_df)),
        "matched_trial_count": int(len(matched_df)),
        "reference_counts_by_abl": format_column_counts(reference_df, "ABL"),
        "candidate_counts_by_abl": format_column_counts(candidate_df, "ABL"),
        "matched_counts_by_abl": format_column_counts(matched_df, "ABL"),
        "reference_counts_by_ild": format_column_counts(reference_df, "ILD"),
        "candidate_counts_by_ild": format_column_counts(candidate_df, "ILD"),
        "matched_counts_by_ild": format_column_counts(matched_df, "ILD"),
        "reference_counts_by_group": format_group_counts(reference_df, stim_match_group_cols),
        "candidate_counts_by_group": format_group_counts(candidate_df, stim_match_group_cols),
        "matched_counts_by_group": format_group_counts(matched_df, stim_match_group_cols),
        "ks_reference_vs_candidate": compute_ks_audit(reference_df, candidate_df, stim_match_column),
        "ks_reference_vs_matched": compute_ks_audit(reference_df, matched_df, stim_match_column),
        "strata_sampling_audit": strata_audit,
    }


def compute_empirical_cdf(values):
    values = np.sort(np.asarray(values, dtype=np.float64))
    if len(values) == 0:
        return values, np.array([], dtype=np.float64)
    y = np.arange(1, len(values) + 1, dtype=np.float64) / float(len(values))
    return values, y


def format_ks_annotation(ks_stats):
    pvalue = ks_stats["pvalue"]
    statistic = ks_stats["statistic"]
    pass_text = "yes" if (pvalue is not None and pvalue > 0.05) else "no"
    return (
        f"KS statistic = {statistic:.4f}\n"
        f"KS p-value = {pvalue:.3g}\n"
        f"p > 0.05: {pass_text}"
    )


def save_stim_matching_diagnostic_plot(
    reference_df,
    candidate_df,
    matched_df,
    matching_audit,
    plot_path,
):
    reference_values_ms = reference_df[stim_match_column].to_numpy(dtype=np.float64) * 1e3
    candidate_values_ms = candidate_df[stim_match_column].to_numpy(dtype=np.float64) * 1e3
    matched_values_ms = matched_df[stim_match_column].to_numpy(dtype=np.float64) * 1e3

    combined_values_ms = np.concatenate([
        reference_values_ms,
        candidate_values_ms,
        matched_values_ms,
    ])
    x_min = float(np.min(combined_values_ms))
    x_max = float(np.max(combined_values_ms))
    hist_bin_count = 50
    hist_edges_ms = np.linspace(x_min, x_max, hist_bin_count + 1)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)

    plot_specs = [
        (
            axes[0, 0],
            axes[0, 1],
            candidate_values_ms,
            "Candidate",
            "Before Subsampling",
            matching_audit["ks_reference_vs_candidate"]["overall"],
        ),
        (
            axes[1, 0],
            axes[1, 1],
            matched_values_ms,
            "Matched",
            "After Subsampling",
            matching_audit["ks_reference_vs_matched"]["overall"],
        ),
    ]

    reference_color = "black"
    comparison_colors = {"Candidate": "tab:orange", "Matched": "tab:green"}

    for density_ax, cdf_ax, comparison_values_ms, comparison_label, row_title, ks_stats in plot_specs:
        density_ax.hist(
            reference_values_ms,
            bins=hist_edges_ms,
            density=True,
            histtype="step",
            linewidth=2.0,
            color=reference_color,
            label=f"Reference ({len(reference_values_ms)})",
        )
        density_ax.hist(
            comparison_values_ms,
            bins=hist_edges_ms,
            density=True,
            histtype="step",
            linewidth=2.0,
            color=comparison_colors[comparison_label],
            label=f"{comparison_label} ({len(comparison_values_ms)})",
        )
        density_ax.set_title(f"{row_title}: Density")
        density_ax.set_xlabel("intended_fix (ms)")
        density_ax.set_ylabel("Density")
        density_ax.legend(frameon=False)

        reference_x, reference_y = compute_empirical_cdf(reference_values_ms)
        comparison_x, comparison_y = compute_empirical_cdf(comparison_values_ms)
        cdf_ax.step(reference_x, reference_y, where="post", linewidth=2.0, color=reference_color)
        cdf_ax.step(
            comparison_x,
            comparison_y,
            where="post",
            linewidth=2.0,
            color=comparison_colors[comparison_label],
        )
        cdf_ax.set_title(f"{row_title}: CDF")
        cdf_ax.set_xlabel("intended_fix (ms)")
        cdf_ax.set_ylabel("Empirical CDF")
        cdf_ax.text(
            0.02,
            0.98,
            format_ks_annotation(ks_stats),
            transform=cdf_ax.transAxes,
            ha="left",
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
        )

    fig.suptitle(
        "Stim Distribution Matching Diagnostics\n"
        f"Reference = {reference_truncate_rt_wrt_stim_s*1e3:.0f} ms, "
        f"Target = {truncate_rt_wrt_stim_s*1e3:.0f} ms, "
        f"match column = {stim_match_column}",
        y=1.02,
    )
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)


run_tag = build_run_tag(
    truncate_rt_wrt_stim_s,
    reference_truncate_rt_wrt_stim_s,
    match_stim_distribution,
    stim_match_group_cols,
    stim_match_quantile_bins,
)


# %%
########### Normalized TIED likelihood/prior ##############
def compute_loglike_norm_fn(
    row,
    rate_lambda,
    T_0,
    theta_E,
    Z_E,
    t_E_aff_20,
    t_E_aff_40,
    t_E_aff_60,
    del_go,
    rate_norm_l,
):
    t_E_aff = get_t_E_aff_from_abl(
        row["ABL"],
        t_E_aff_20=t_E_aff_20,
        t_E_aff_40=t_E_aff_40,
        t_E_aff_60=t_E_aff_60,
    )
    return trial_logpdf_proactive_lapse_only_no_trunc_right_truncated_no_choice(
        row=row,
        V_A=V_A,
        theta_A=theta_A,
        t_A_aff=t_A_aff,
        rate_lambda=rate_lambda,
        T0=T_0,
        theta_E=theta_E,
        Z_E=Z_E,
        t_E_aff=t_E_aff,
        del_go=del_go,
        phi_params=phi_params_obj,
        rate_norm_l=rate_norm_l,
        is_norm=is_norm,
        is_time_vary=is_time_vary,
        K_max=K_max,
        lapse_prob=lapse_prob,
        beta_lapse=beta_lapse,
        lapse_choice_prob=0.5,
        truncate_rt_wrt_stim=truncate_rt_wrt_stim_s,
        eps=1e-50,
    )


def vbmc_norm_tied_loglike_fn(params):
    (
        rate_lambda,
        T_0,
        theta_E,
        w,
        t_E_aff_20,
        t_E_aff_40,
        t_E_aff_60,
        del_go,
        rate_norm_l,
    ) = params
    Z_E = (w - 0.5) * 2 * theta_E
    all_loglike = Parallel(n_jobs=n_jobs)(
        delayed(compute_loglike_norm_fn)(
            row,
            rate_lambda,
            T_0,
            theta_E,
            Z_E,
            t_E_aff_20,
            t_E_aff_40,
            t_E_aff_60,
            del_go,
            rate_norm_l,
        )
        for _, row in df_valid_animal_truncated.iterrows()
    )
    return np.sum(all_loglike)


def vbmc_prior_norm_tied_fn(params):
    (
        rate_lambda,
        T_0,
        theta_E,
        w,
        t_E_aff_20,
        t_E_aff_40,
        t_E_aff_60,
        del_go,
        rate_norm_l,
    ) = params
    return (
        trapezoidal_logpdf(
            rate_lambda,
            norm_tied_rate_lambda_bounds[0],
            norm_tied_rate_lambda_plausible_bounds[0],
            norm_tied_rate_lambda_plausible_bounds[1],
            norm_tied_rate_lambda_bounds[1],
        )
        + trapezoidal_logpdf(
            T_0,
            norm_tied_T_0_bounds[0],
            norm_tied_T_0_plausible_bounds[0],
            norm_tied_T_0_plausible_bounds[1],
            norm_tied_T_0_bounds[1],
        )
        + trapezoidal_logpdf(
            theta_E,
            norm_tied_theta_E_bounds[0],
            norm_tied_theta_E_plausible_bounds[0],
            norm_tied_theta_E_plausible_bounds[1],
            norm_tied_theta_E_bounds[1],
        )
        + trapezoidal_logpdf(
            w,
            norm_tied_w_bounds[0],
            norm_tied_w_plausible_bounds[0],
            norm_tied_w_plausible_bounds[1],
            norm_tied_w_bounds[1],
        )
        + trapezoidal_logpdf(
            t_E_aff_20,
            norm_tied_t_E_aff_20_bounds[0],
            norm_tied_t_E_aff_20_plausible_bounds[0],
            norm_tied_t_E_aff_20_plausible_bounds[1],
            norm_tied_t_E_aff_20_bounds[1],
        )
        + trapezoidal_logpdf(
            t_E_aff_40,
            norm_tied_t_E_aff_40_bounds[0],
            norm_tied_t_E_aff_40_plausible_bounds[0],
            norm_tied_t_E_aff_40_plausible_bounds[1],
            norm_tied_t_E_aff_40_bounds[1],
        )
        + trapezoidal_logpdf(
            t_E_aff_60,
            norm_tied_t_E_aff_60_bounds[0],
            norm_tied_t_E_aff_60_plausible_bounds[0],
            norm_tied_t_E_aff_60_plausible_bounds[1],
            norm_tied_t_E_aff_60_bounds[1],
        )
        + trapezoidal_logpdf(
            del_go,
            norm_tied_del_go_bounds[0],
            norm_tied_del_go_plausible_bounds[0],
            norm_tied_del_go_plausible_bounds[1],
            norm_tied_del_go_bounds[1],
        )
        + trapezoidal_logpdf(
            rate_norm_l,
            norm_tied_rate_norm_bounds[0],
            norm_tied_rate_norm_plausible_bounds[0],
            norm_tied_rate_norm_plausible_bounds[1],
            norm_tied_rate_norm_bounds[1],
        )
    )


def vbmc_norm_tied_joint_fn(params):
    return vbmc_prior_norm_tied_fn(params) + vbmc_norm_tied_loglike_fn(params)


############ Bounds ############
norm_tied_rate_lambda_bounds = [0.5, 5]
norm_tied_T_0_bounds = [10e-3, 500e-3]
norm_tied_theta_E_bounds = [1, 15]
norm_tied_w_bounds = [0.3, 0.7]
norm_tied_t_E_aff_20_bounds = [0.01, 0.2]
norm_tied_t_E_aff_40_bounds = [0.01, 0.2]
norm_tied_t_E_aff_60_bounds = [0.01, 0.2]
norm_tied_del_go_bounds = [0, 0.2]
norm_tied_rate_norm_bounds = [0, 2]

norm_tied_rate_lambda_plausible_bounds = [1, 3]
norm_tied_T_0_plausible_bounds = [20e-3, 50e-3]
norm_tied_theta_E_plausible_bounds = [1.5, 10]
norm_tied_w_plausible_bounds = [0.4, 0.6]
norm_tied_t_E_aff_20_plausible_bounds = [0.03, 0.09]
norm_tied_t_E_aff_40_plausible_bounds = [0.03, 0.09]
norm_tied_t_E_aff_60_plausible_bounds = [0.03, 0.09]
norm_tied_del_go_plausible_bounds = [0.05, 0.15]
norm_tied_rate_norm_plausible_bounds = [0.8, 0.99]

norm_tied_lb = np.array([
    norm_tied_rate_lambda_bounds[0],
    norm_tied_T_0_bounds[0],
    norm_tied_theta_E_bounds[0],
    norm_tied_w_bounds[0],
    norm_tied_t_E_aff_20_bounds[0],
    norm_tied_t_E_aff_40_bounds[0],
    norm_tied_t_E_aff_60_bounds[0],
    norm_tied_del_go_bounds[0],
    norm_tied_rate_norm_bounds[0],
])

norm_tied_ub = np.array([
    norm_tied_rate_lambda_bounds[1],
    norm_tied_T_0_bounds[1],
    norm_tied_theta_E_bounds[1],
    norm_tied_w_bounds[1],
    norm_tied_t_E_aff_20_bounds[1],
    norm_tied_t_E_aff_40_bounds[1],
    norm_tied_t_E_aff_60_bounds[1],
    norm_tied_del_go_bounds[1],
    norm_tied_rate_norm_bounds[1],
])

norm_tied_plb = np.array([
    norm_tied_rate_lambda_plausible_bounds[0],
    norm_tied_T_0_plausible_bounds[0],
    norm_tied_theta_E_plausible_bounds[0],
    norm_tied_w_plausible_bounds[0],
    norm_tied_t_E_aff_20_plausible_bounds[0],
    norm_tied_t_E_aff_40_plausible_bounds[0],
    norm_tied_t_E_aff_60_plausible_bounds[0],
    norm_tied_del_go_plausible_bounds[0],
    norm_tied_rate_norm_plausible_bounds[0],
])

norm_tied_pub = np.array([
    norm_tied_rate_lambda_plausible_bounds[1],
    norm_tied_T_0_plausible_bounds[1],
    norm_tied_theta_E_plausible_bounds[1],
    norm_tied_w_plausible_bounds[1],
    norm_tied_t_E_aff_20_plausible_bounds[1],
    norm_tied_t_E_aff_40_plausible_bounds[1],
    norm_tied_t_E_aff_60_plausible_bounds[1],
    norm_tied_del_go_plausible_bounds[1],
    norm_tied_rate_norm_plausible_bounds[1],
])


# %%
############ Load + preprocess LED7-style data ############
if not led_data_csv_path.exists():
    raise FileNotFoundError(f"Could not find data CSV: {led_data_csv_path}")

exp_df = pd.read_csv(led_data_csv_path)
exp_df["RTwrtStim"] = exp_df["timed_fix"] - exp_df["intended_fix"]
exp_df = exp_df.rename(columns={"timed_fix": "TotalFixTime"})
exp_df = exp_df[exp_df["RTwrtStim"] < 1]
exp_df = exp_df[~((exp_df["RTwrtStim"].isna()) & (exp_df["abort_event"] == 3))].copy()

mask_nan = exp_df["response_poke"].isna()
mask_success_1 = exp_df["success"] == 1
mask_success_neg1 = exp_df["success"] == -1
mask_ild_pos = exp_df["ILD"] > 0
mask_ild_neg = exp_df["ILD"] < 0
exp_df.loc[mask_nan & mask_success_1 & mask_ild_pos, "response_poke"] = 3
exp_df.loc[mask_nan & mask_success_1 & mask_ild_neg, "response_poke"] = 2
exp_df.loc[mask_nan & mask_success_neg1 & mask_ild_pos, "response_poke"] = 2
exp_df.loc[mask_nan & mask_success_neg1 & mask_ild_neg, "response_poke"] = 3

mask_led_off = (exp_df["LED_trial"] == 0) | (exp_df["LED_trial"].isna())
mask_repeat = exp_df["repeat_trial"].isin(allowed_repeat_trials) | exp_df["repeat_trial"].isna()
exp_df_led_off = exp_df[
    mask_led_off
    & mask_repeat
    & exp_df["session_type"].isin([session_type])
    & exp_df["training_level"].isin([training_level])
].copy()

exp_df_led_off["choice"] = np.where(
    exp_df_led_off["response_poke"] == 3,
    1,
    np.where(exp_df_led_off["response_poke"] == 2, -1, np.nan),
)
missing_choice = exp_df_led_off["choice"].isna()
if missing_choice.any():
    exp_df_led_off.loc[missing_choice, "choice"] = np.random.choice(
        [1, -1], size=int(missing_choice.sum())
    )
exp_df_led_off["choice"] = exp_df_led_off["choice"].astype(int)

df_valid_and_aborts = exp_df_led_off[
    (exp_df_led_off["success"].isin([1, -1])) | (exp_df_led_off["abort_event"] == 3)
].copy()
observed_abl_values = validate_supported_abl_values(
    df_valid_and_aborts, "filtered LED-OFF valid+aborts dataset"
)

print(
    f"Batch={batch_name}, session_type={session_type}, training_level={training_level}, "
    f"LED OFF trials only"
)
print(f"T_trunc={T_trunc} (not used in likelihood)")
print(
    "Right truncation target: keep trials with "
    f"0 < RTwrtStim <= {truncate_rt_wrt_stim_s:.3f}s for the candidate pool."
)
print(
    "Reference truncation for intended_fix matching: "
    f"0 < RTwrtStim <= {reference_truncate_rt_wrt_stim_s:.3f}s."
)
print(f"Run tag for saved fit artifacts: {run_tag}")
print(f"Supported ABL values in filtered data: {observed_abl_values.tolist()}")
print(f"Filtered LED-OFF trial counts by ABL: {format_column_counts(df_valid_and_aborts, 'ABL')}")


# %%
def fit_aggregate(df_unit, proactive_vp_path):
    global V_A, theta_A, t_A_aff, lapse_prob, beta_lapse, df_valid_animal_truncated

    if not proactive_vp_path.exists():
        print(f"Skipping aggregate fit: proactive VP pickle not found: {proactive_vp_path}")
        return

    observed_unit_abl_values = validate_supported_abl_values(df_unit, "aggregate filtered dataset")
    print(f"aggregate: observed ABL values -> {observed_unit_abl_values.tolist()}")

    with open(proactive_vp_path, "rb") as f:
        proactive_vp = pickle.load(f)

    proactive_samples = proactive_vp.sample(proactive_posterior_sample_count)[0]
    V_A_base = float(np.mean(proactive_samples[:, 0]))
    theta_A = float(np.mean(proactive_samples[:, 2]))
    del_a_minus_del_LED = float(np.mean(proactive_samples[:, 3]))
    del_m_plus_del_LED = float(np.mean(proactive_samples[:, 4]))
    lapse_prob = float(np.mean(proactive_samples[:, 5]))
    beta_lapse = float(np.mean(proactive_samples[:, 6]))

    V_A = V_A_base
    t_A_aff = del_a_minus_del_LED + del_m_plus_del_LED
    print(
        "\naggregate: loaded proactive params -> "
        f"V_A_base={V_A_base:.4f}, theta_A={theta_A:.4f}, "
        f"del_a_minus_del_LED={del_a_minus_del_LED:.4f}, del_m_plus_del_LED={del_m_plus_del_LED:.4f}, "
        f"lapse_prob={lapse_prob:.4f}, beta_lapse={beta_lapse:.4f}, "
        f"derived t_A_aff={t_A_aff:.4f}"
    )

    df_valid_unit = df_unit[df_unit["success"].isin([1, -1])].copy()
    df_valid_unit_for_fit_window = df_valid_unit[
        (df_valid_unit["RTwrtStim"] > 0) & (df_valid_unit["RTwrtStim"] < max_rtwrtstim_for_fit)
    ].copy()

    reference_df = df_valid_unit_for_fit_window[
        df_valid_unit_for_fit_window["RTwrtStim"] <= reference_truncate_rt_wrt_stim_s
    ].copy()
    candidate_df = df_valid_unit_for_fit_window[
        df_valid_unit_for_fit_window["RTwrtStim"] <= truncate_rt_wrt_stim_s
    ].copy()

    pre_trunc_valid_trial_count = len(df_valid_unit_for_fit_window)
    reference_valid_trial_count = len(reference_df)
    candidate_valid_trial_count = len(candidate_df)
    ignored_valid_trial_count = pre_trunc_valid_trial_count - candidate_valid_trial_count

    if reference_valid_trial_count == 0:
        print("Skipping aggregate fit: no valid trials remain after reference truncation")
        return
    if candidate_valid_trial_count == 0:
        print("Skipping aggregate fit: no valid trials remain after target truncation")
        return

    print(
        f"aggregate: valid trials in fit window={pre_trunc_valid_trial_count}, "
        f"reference({reference_truncate_rt_wrt_stim_s:.3f}s)={reference_valid_trial_count}, "
        f"candidate({truncate_rt_wrt_stim_s:.3f}s)={candidate_valid_trial_count}, "
        f"ignored_above_target={ignored_valid_trial_count}"
    )
    print(
        f"aggregate: reference ABL counts={format_column_counts(reference_df, 'ABL')}, "
        f"candidate ABL counts={format_column_counts(candidate_df, 'ABL')}"
    )
    print(
        f"aggregate: reference ILD counts={format_column_counts(reference_df, 'ILD')}, "
        f"candidate ILD counts={format_column_counts(candidate_df, 'ILD')}"
    )
    print("aggregate: fitting ignores observed choice and uses pdf(choice=+1)+pdf(choice=-1).")

    if match_stim_distribution:
        print(
            "aggregate: matching intended_fix distribution from the target pool to the reference pool "
            f"within {stim_match_group_cols} using seed={stim_match_seed}, "
            f"quantile_bins={stim_match_quantile_bins}, "
            f"min_trials_per_bin={stim_match_min_trials_per_bin}"
        )
        df_valid_animal_truncated, strata_audit = sample_candidate_trials_to_match_reference(
            reference_df=reference_df,
            candidate_df=candidate_df,
            match_column=stim_match_column,
            group_cols=stim_match_group_cols,
            quantile_bins=stim_match_quantile_bins,
            min_trials_per_bin=stim_match_min_trials_per_bin,
            seed=stim_match_seed,
        )
    else:
        df_valid_animal_truncated = candidate_df.copy()
        strata_audit = []

    fit_valid_trial_count = len(df_valid_animal_truncated)
    fit_abl_counts = format_column_counts(df_valid_animal_truncated, "ABL")
    fit_ild_counts = format_column_counts(df_valid_animal_truncated, "ILD")
    matching_audit = build_matching_audit(
        reference_df=reference_df,
        candidate_df=candidate_df,
        matched_df=df_valid_animal_truncated,
        strata_audit=strata_audit,
    )

    print(
        f"aggregate: matched trial count={fit_valid_trial_count}, "
        f"ABL counts used for fit={fit_abl_counts}, ILD counts used for fit={fit_ild_counts}"
    )
    pre_match_ks = matching_audit["ks_reference_vs_candidate"]["overall"]
    post_match_ks = matching_audit["ks_reference_vs_matched"]["overall"]
    print(
        "aggregate: intended_fix KS overall "
        f"reference_vs_candidate={pre_match_ks['statistic']:.6f} (p={pre_match_ks['pvalue']:.3g}), "
        f"reference_vs_matched={post_match_ks['statistic']:.6f} (p={post_match_ks['pvalue']:.3g})"
    )

    base_name = (
        f"batch_{batch_name}_aggregate_ledoff_1_"
        f"proactive_loaded_truncate_NOT_censor_ABL_delay_no_choice_match_ref_stim_{run_tag}"
    )

    stim_match_plot_path = None
    if save_stim_match_plot:
        stim_match_plot_path = output_dir / f"stim_match_diagnostic_{base_name}.pdf"
        save_stim_matching_diagnostic_plot(
            reference_df=reference_df,
            candidate_df=candidate_df,
            matched_df=df_valid_animal_truncated,
            matching_audit=matching_audit,
            plot_path=stim_match_plot_path,
        )
        print(f"Saved stim-match diagnostic plot: {stim_match_plot_path}")

    x_0 = np.array([2.3, 100e-3, 3.0, 0.51, 0.071, 0.071, 0.071, 0.13, 0.95])
    x_0 = np.clip(x_0, norm_tied_plb, norm_tied_pub)

    vbmc = VBMC(
        vbmc_norm_tied_joint_fn,
        x_0,
        norm_tied_lb,
        norm_tied_ub,
        norm_tied_plb,
        norm_tied_pub,
        options={"display": "on", "max_fun_evals": vbmc_max_fun_evals},
    )
    vp, results = vbmc.optimize()

    vbmc_obj_path = output_dir / f"vbmc_norm_tied_{base_name}.pkl"
    vp.save(str(vbmc_obj_path), overwrite=True)

    vp_samples = vp.sample(posterior_sample_count)[0]
    rate_lambda = float(vp_samples[:, 0].mean())
    T_0 = float(vp_samples[:, 1].mean())
    theta_E = float(vp_samples[:, 2].mean())
    w = float(vp_samples[:, 3].mean())
    t_E_aff_20 = float(vp_samples[:, 4].mean())
    t_E_aff_40 = float(vp_samples[:, 5].mean())
    t_E_aff_60 = float(vp_samples[:, 6].mean())
    del_go = float(vp_samples[:, 7].mean())
    rate_norm_l = float(vp_samples[:, 8].mean())
    Z_E = (w - 0.5) * 2 * theta_E

    norm_tied_loglike = vbmc_norm_tied_loglike_fn(
        [
            rate_lambda,
            T_0,
            theta_E,
            w,
            t_E_aff_20,
            t_E_aff_40,
            t_E_aff_60,
            del_go,
            rate_norm_l,
        ]
    )

    corner_path = None
    if save_corner_plot:
        corner_samples = vp_samples.copy()
        corner_samples[:, 1] *= 1e3
        labels = [
            r"$\lambda$",
            r"$T_0$ (ms)",
            r"$\theta_E$",
            r"$w$",
            r"$t_{E,20}^{aff}$",
            r"$t_{E,40}^{aff}$",
            r"$t_{E,60}^{aff}$",
            r"$\Delta_{go}$",
            "rate_norm_l",
        ]
        corner_fig = corner.corner(
            corner_samples,
            labels=labels,
            show_titles=True,
            quantiles=[0.025, 0.5, 0.975],
            title_fmt=".3f",
        )
        corner_fig.suptitle(
            "Normalized TIED Posterior (aggregate, matched reference stim, no choice)",
            y=1.02,
        )
        corner_path = output_dir / f"corner_norm_tied_{base_name}.pdf"
        corner_fig.savefig(corner_path, bbox_inches="tight")
        print(f"Saved corner plot: {corner_path}")

    print("Posterior means:")
    print(
        f"rate_lambda={rate_lambda:.5f}, T_0(ms)={1e3*T_0:.5f}, theta_E={theta_E:.5f}, "
        f"w={w:.5f}, Z_E={Z_E:.5f}, t_E_aff_20(ms)={1e3*t_E_aff_20:.5f}, "
        f"t_E_aff_40(ms)={1e3*t_E_aff_40:.5f}, t_E_aff_60(ms)={1e3*t_E_aff_60:.5f}, "
        f"del_go={del_go:.5f}, rate_norm_l={rate_norm_l:.5f}"
    )

    vbmc_norm_tied_results = {
        "rate_lambda_samples": vp_samples[:, 0],
        "T_0_samples": vp_samples[:, 1],
        "theta_E_samples": vp_samples[:, 2],
        "w_samples": vp_samples[:, 3],
        "t_E_aff_20_samples": vp_samples[:, 4],
        "t_E_aff_40_samples": vp_samples[:, 5],
        "t_E_aff_60_samples": vp_samples[:, 6],
        "del_go_samples": vp_samples[:, 7],
        "rate_norm_l_samples": vp_samples[:, 8],
        "message": results.get("message"),
        "elbo": results.get("elbo"),
        "elbo_sd": results.get("elbo_sd"),
        "loglike": norm_tied_loglike,
    }

    fit_trial_counts = {
        "valid_trials_before_right_truncation": pre_trunc_valid_trial_count,
        "valid_trials_at_reference_right_truncation": reference_valid_trial_count,
        "valid_trials_at_target_right_truncation": candidate_valid_trial_count,
        "ignored_valid_trials_above_target_right_truncation": ignored_valid_trial_count,
        "reference_abl_counts": format_column_counts(reference_df, "ABL"),
        "candidate_abl_counts": format_column_counts(candidate_df, "ABL"),
        "matched_abl_counts": fit_abl_counts,
        "reference_ild_counts": format_column_counts(reference_df, "ILD"),
        "candidate_ild_counts": format_column_counts(candidate_df, "ILD"),
        "matched_ild_counts": fit_ild_counts,
        "valid_trials_used_for_fit": fit_valid_trial_count,
    }

    out_results_path = output_dir / f"results_norm_tied_{base_name}.pkl"
    save_dict = {
        "unit_tag": "aggregate",
        "source_proactive_vp_pkl": str(proactive_vp_path),
        "fit_config": {
            "batch_name": batch_name,
            "fit_mode": fit_mode,
            "session_type": session_type,
            "training_level": training_level,
            "allowed_repeat_trials": allowed_repeat_trials,
            "led_data_csv_path": str(led_data_csv_path),
            "likelihood_mode": "proactive_lapse_only_no_trunc_right_truncated_ABL_delay_no_choice",
            "right_truncation_rule": (
                "fit only trials with 0 < RT - t_stim <= truncate_rt_wrt_stim_s; "
                "use the choice-collapsed RT density "
                "pdf(choice=+1)+pdf(choice=-1); "
                "normalize retained trial likelihood by "
                "CDF(t_stim + truncate_rt_wrt_stim_s) - CDF(t_stim)"
            ),
            "choice_mode": "ignore_observed_choice_use_collapsed_rt_density",
            "abl_specific_delay_rule": "ABL=20 -> t_E_aff_20, ABL=40 -> t_E_aff_40, ABL=60 -> t_E_aff_60",
            "supported_ABL_values": list(supported_abl_values),
            "reference_truncate_rt_wrt_stim_s": reference_truncate_rt_wrt_stim_s,
            "truncate_rt_wrt_stim_s": truncate_rt_wrt_stim_s,
            "run_tag": run_tag,
            "match_stim_distribution": match_stim_distribution,
            "stim_match_column": stim_match_column,
            "stim_match_group_cols": list(stim_match_group_cols),
            "stim_match_quantile_bins": stim_match_quantile_bins,
            "stim_match_min_trials_per_bin": stim_match_min_trials_per_bin,
            "stim_match_seed": stim_match_seed,
            "proactive_vp_aggregate_path": str(proactive_vp_aggregate_path),
            "proactive_posterior_sample_count": proactive_posterior_sample_count,
            "max_rtwrtstim_for_fit": max_rtwrtstim_for_fit,
            "save_corner_plot": save_corner_plot,
            "save_stim_match_plot": save_stim_match_plot,
            "corner_path": str(corner_path) if corner_path is not None else None,
            "stim_match_plot_path": (
                str(stim_match_plot_path) if stim_match_plot_path is not None else None
            ),
            "T_trunc": T_trunc,
            "vbmc_max_fun_evals": vbmc_max_fun_evals,
        },
        "fit_trial_counts": fit_trial_counts,
        "matching_audit": matching_audit,
        "loaded_proactive_params": {
            "V_A_base": V_A_base,
            "theta_A": theta_A,
            "del_a_minus_del_LED": del_a_minus_del_LED,
            "del_m_plus_del_LED": del_m_plus_del_LED,
            "lapse_prob": lapse_prob,
            "beta_lapse": beta_lapse,
            "derived_t_A_aff_for_tied_fit": t_A_aff,
        },
        "vbmc_norm_tied_results": vbmc_norm_tied_results,
    }
    with open(out_results_path, "wb") as f:
        pickle.dump(save_dict, f)

    print(f"Saved VBMC object: {vbmc_obj_path}")
    print(f"Saved results: {out_results_path}")


# %%
############ Run mode ############
print("Running aggregate fit with reference-stim-matched truncation and no-choice likelihood.")
fit_aggregate(
    df_unit=df_valid_and_aborts,
    proactive_vp_path=proactive_vp_aggregate_path,
)

print("\nDone.")
