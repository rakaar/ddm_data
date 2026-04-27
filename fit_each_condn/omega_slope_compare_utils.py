# %%
import os

import numpy as np
import pandas as pd


def nanmean_sem_n(values, axis=0):
    values = np.asarray(values, dtype=float)
    n = np.sum(np.isfinite(values), axis=axis)
    mean = np.nanmean(values, axis=axis)
    sem = np.nanstd(values, axis=axis) / np.sqrt(np.where(n == 0, np.nan, n))
    return mean, sem, n


def load_batch_csv_data(batch_dir, desired_batches, csv_suffix=""):
    batch_files = [f"batch_{batch_name}_valid_and_aborts{csv_suffix}.csv" for batch_name in desired_batches]
    dfs = []
    for fname in batch_files:
        fpath = os.path.join(batch_dir, fname)
        if os.path.exists(fpath):
            dfs.append(pd.read_csv(fpath))

    if len(dfs) == 0:
        raise FileNotFoundError(f"No batch CSV files found in {batch_dir}")

    merged_data = pd.concat(dfs, ignore_index=True)
    merged_data["abs_ILD"] = merged_data["ILD"].abs()
    return merged_data


def compute_quantile_slope_summary(
    merged_data,
    batch_animal_pairs,
    abls_to_compare,
    baseline_abl,
    abs_ilds,
    fitting_quantiles,
    min_rt_cut_by_ild,
    max_rt_cut,
    min_trials=5,
):
    raw_slopes = {ABL: np.full((len(batch_animal_pairs), len(abs_ilds)), np.nan) for ABL in abls_to_compare}
    n_trials_by_abl = {
        ABL: np.full((len(batch_animal_pairs), len(abs_ilds)), np.nan)
        for ABL in sorted(set(list(abls_to_compare) + [baseline_abl]))
    }

    for animal_idx, (batch_name, animal_id) in enumerate(batch_animal_pairs):
        animal_df = merged_data[
            (merged_data["batch_name"] == batch_name)
            & (merged_data["animal"] == animal_id)
        ]

        fit_quantiles_by_abl = {}
        for ABL in sorted(set(list(abls_to_compare) + [baseline_abl])):
            fit_q_list = []
            for ild_idx, abs_ild in enumerate(abs_ilds):
                condition_df = animal_df[
                    (animal_df["ABL"] == ABL)
                    & (animal_df["abs_ILD"] == abs_ild)
                    & (animal_df["RTwrtStim"] >= 0)
                    & (animal_df["RTwrtStim"] <= 1)
                    & (animal_df["success"].isin([1, -1]))
                ]
                n_trials_by_abl[ABL][animal_idx, ild_idx] = len(condition_df)

                if len(condition_df) < min_trials:
                    fit_q_list.append(np.full(len(fitting_quantiles), np.nan))
                else:
                    fit_q_list.append(condition_df["RTwrtStim"].quantile(fitting_quantiles).values)

            fit_quantiles_by_abl[ABL] = np.asarray(fit_q_list).T

        baseline_fit_quantiles = fit_quantiles_by_abl[baseline_abl]

        for ABL in abls_to_compare:
            curr_fit_quantiles = fit_quantiles_by_abl[ABL]
            for ild_idx, abs_ild in enumerate(abs_ilds):
                q_baseline = baseline_fit_quantiles[:, ild_idx]
                q_curr = curr_fit_quantiles[:, ild_idx]
                slope = np.nan

                if not np.any(np.isnan(q_baseline)) and not np.any(np.isnan(q_curr)):
                    min_rt_cut = min_rt_cut_by_ild[abs_ild]
                    mask = (q_baseline >= min_rt_cut) & (q_baseline <= max_rt_cut)
                    if np.sum(mask) >= 2:
                        q_curr_minus_baseline = q_curr - q_baseline
                        x_fit = q_baseline[mask] - min_rt_cut
                        y_fit = q_curr_minus_baseline[mask]
                        y_fit_shifted = y_fit - y_fit[0]

                        if np.sum(x_fit**2) > 0:
                            slope = np.sum(x_fit * y_fit_shifted) / np.sum(x_fit**2)

                raw_slopes[ABL][animal_idx, ild_idx] = slope

    raw_slope_mean = {}
    raw_slope_sem = {}
    raw_slope_n = {}
    slope_ratio_mean = {}
    slope_ratio_sem = {}

    for ABL in abls_to_compare:
        raw_slope_mean[ABL], raw_slope_sem[ABL], raw_slope_n[ABL] = nanmean_sem_n(raw_slopes[ABL], axis=0)
        slope_ratio_mean[ABL] = 1 + raw_slope_mean[ABL]
        slope_ratio_sem[ABL] = raw_slope_sem[ABL]

    return {
        "batch_animal_pairs": batch_animal_pairs,
        "abs_ilds": np.asarray(abs_ilds),
        "abls_to_compare": abls_to_compare,
        "baseline_abl": baseline_abl,
        "fitting_quantiles": fitting_quantiles,
        "raw_slopes": raw_slopes,
        "raw_slope_mean": raw_slope_mean,
        "raw_slope_sem": raw_slope_sem,
        "raw_slope_n": raw_slope_n,
        "slope_ratio_mean": slope_ratio_mean,
        "slope_ratio_sem": slope_ratio_sem,
        "n_trials_by_abl": n_trials_by_abl,
    }


def compute_abs_omega_summary(omega_by_abl, abls, signed_ilds, abs_ilds):
    signed_ild_to_idx = {int(ILD): idx for idx, ILD in enumerate(signed_ilds)}

    omega_abs_values = {}
    omega_abs_mean = {}
    omega_abs_sem = {}
    omega_abs_n = {}

    for ABL in abls:
        arr = omega_by_abl[str(ABL)]
        values = np.full((arr.shape[0], len(abs_ilds)), np.nan)

        for ild_idx, abs_ild in enumerate(abs_ilds):
            plus_idx = signed_ild_to_idx[int(abs_ild)]
            minus_idx = signed_ild_to_idx[-int(abs_ild)]
            signed_values = arr[:, [plus_idx, minus_idx]]
            n_signed = np.sum(np.isfinite(signed_values), axis=1)
            summed = np.nansum(signed_values, axis=1)
            values[:, ild_idx] = np.divide(
                summed,
                n_signed,
                out=np.full_like(summed, np.nan, dtype=float),
                where=n_signed > 0,
            )

        omega_abs_values[ABL] = values
        omega_abs_mean[ABL], omega_abs_sem[ABL], omega_abs_n[ABL] = nanmean_sem_n(values, axis=0)

    return {
        "abs_ilds": np.asarray(abs_ilds),
        "omega_abs_values": omega_abs_values,
        "omega_abs_mean": omega_abs_mean,
        "omega_abs_sem": omega_abs_sem,
        "omega_abs_n": omega_abs_n,
    }


def compute_omega_ratio_summary(omega_abs_summary, abls_to_compare, baseline_abl):
    omega_ratio_mean = {}
    omega_ratio_sem = {}
    omega_ratio_n = {}

    baseline_mean = omega_abs_summary["omega_abs_mean"][baseline_abl]
    baseline_sem = omega_abs_summary["omega_abs_sem"][baseline_abl]

    for ABL in abls_to_compare:
        curr_mean = omega_abs_summary["omega_abs_mean"][ABL]
        curr_sem = omega_abs_summary["omega_abs_sem"][ABL]

        ratio = baseline_mean / curr_mean
        ratio_sem = ratio * np.sqrt((baseline_sem / baseline_mean) ** 2 + (curr_sem / curr_mean) ** 2)

        baseline_values = omega_abs_summary["omega_abs_values"][baseline_abl]
        curr_values = omega_abs_summary["omega_abs_values"][ABL]
        ratio_n = np.sum(np.isfinite(baseline_values) & np.isfinite(curr_values), axis=0)

        omega_ratio_mean[ABL] = ratio
        omega_ratio_sem[ABL] = ratio_sem
        omega_ratio_n[ABL] = ratio_n

    return {
        "omega_ratio_mean": omega_ratio_mean,
        "omega_ratio_sem": omega_ratio_sem,
        "omega_ratio_n": omega_ratio_n,
    }


def compute_animal_omega_ratio_values(omega_abs_summary, abls_to_compare, baseline_abl):
    omega_ratio_values = {}
    baseline_values = omega_abs_summary["omega_abs_values"][baseline_abl]

    for ABL in abls_to_compare:
        curr_values = omega_abs_summary["omega_abs_values"][ABL]
        omega_ratio_values[ABL] = np.divide(
            baseline_values,
            curr_values,
            out=np.full_like(baseline_values, np.nan, dtype=float),
            where=np.isfinite(baseline_values) & np.isfinite(curr_values) & (curr_values != 0),
        )

    return omega_ratio_values
