import numpy as np
from joblib import Parallel, delayed

from time_vary_norm_utils import (
    CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn_vec,
    rho_A_t_VEC_fn,
    rho_E_minus_small_t_NORM_rate_norm_time_varying_fn_vec,
)


def validate_supported_values(df, column_name, supported_values):
    observed = np.sort(df[column_name].dropna().astype(float).unique())
    if len(observed) == 0:
        raise ValueError(f"No {column_name} values found in diagnostics dataset.")

    unexpected = [
        float(value)
        for value in observed
        if not any(np.isclose(float(value), float(supported)) for supported in supported_values)
    ]
    if unexpected:
        raise ValueError(
            f"Unexpected {column_name} values in diagnostics dataset: {unexpected}. "
            f"Supported values are {supported_values}."
        )
    return observed


def format_counts(series):
    counts = (
        series.astype(float).round().astype(int).value_counts().sort_index().to_dict()
    )
    return {int(key): int(value) for key, value in counts.items()}


def build_condition_counts(df, abl_values, abs_ild_values):
    counts = df.groupby(["ABL", "abs_ILD"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=abl_values, columns=abs_ild_values, fill_value=0)
    counts.index = counts.index.astype(int)
    counts.columns = counts.columns.astype(int)
    return counts


def format_truncation_labels(truncate_rt_wrt_stim_s):
    truncate_ms = int(round(float(truncate_rt_wrt_stim_s) * 1e3))
    return truncate_ms, f"{truncate_ms} ms", f"{truncate_ms}ms"


def get_t_E_aff_from_abl_abs_ild(
    abl,
    abs_ild,
    bias_ms,
    abl_delay_coeff_ms_per_abl,
    abs_ild_delay_coeff_ms_per_unit,
):
    delay_ms = (
        float(bias_ms)
        + float(abl_delay_coeff_ms_per_abl) * float(abl)
        + float(abs_ild_delay_coeff_ms_per_unit) * float(abs_ild)
    )
    return delay_ms * 1e-3


def _lapse_pdf_vec(t_arr, beta_lapse, eps=1e-50):
    t_arr = np.asarray(t_arr, dtype=np.float64)
    result = np.full_like(t_arr, eps)
    valid = t_arr >= 0
    vals = beta_lapse * np.exp(-beta_lapse * t_arr[valid])
    vals = np.where(np.isfinite(vals) & (vals > 0), vals, eps)
    result[valid] = vals
    return result


def _lapse_cdf_vec(t_arr, beta_lapse):
    t_arr = np.asarray(t_arr, dtype=np.float64)
    result = np.zeros_like(t_arr)
    valid = t_arr > 0
    result[valid] = np.clip(1.0 - np.exp(-beta_lapse * t_arr[valid]), 0.0, 1.0)
    return result


def _cum_A_t_vec(t_arr, V_A, theta_A):
    from scipy.special import erf as _erf

    t_arr = np.asarray(t_arr, dtype=np.float64)
    result = np.zeros_like(t_arr)
    valid = t_arr > 0
    tv = t_arr[valid]
    term1 = 0.5 * (1 + _erf(V_A * (tv - theta_A / V_A) / np.sqrt(2 * tv)))
    term2 = np.exp(2 * V_A * theta_A) * 0.5 * (
        1 + _erf(-V_A * (tv + theta_A / V_A) / np.sqrt(2 * tv))
    )
    result[valid] = term1 + term2
    return result


def build_theory_curve_for_trial(
    t_pts,
    t_stim,
    abl,
    ild,
    rate_lambda,
    T_0,
    theta_E,
    Z_E,
    bias_ms,
    abl_delay_coeff_ms_per_abl,
    abs_ild_delay_coeff_ms_per_unit,
    del_go,
    rate_norm_l,
    V_A,
    theta_A,
    t_A_aff,
    lapse_prob,
    beta_lapse,
    is_norm,
    is_time_vary,
    K_max,
):
    abs_ild = abs(float(ild))
    t_E_aff = get_t_E_aff_from_abl_abs_ild(
        abl=abl,
        abs_ild=abs_ild,
        bias_ms=bias_ms,
        abl_delay_coeff_ms_per_abl=abl_delay_coeff_ms_per_abl,
        abs_ild_delay_coeff_ms_per_unit=abs_ild_delay_coeff_ms_per_unit,
    )
    t_abs = t_pts + t_stim
    curve = np.zeros_like(t_pts, dtype=np.float64)

    valid = t_abs > 0
    if not np.any(valid):
        return curve

    t_v = t_abs[valid]
    n = len(t_v)

    P_A = rho_A_t_VEC_fn(t_v - t_A_aff, V_A, theta_A)
    C_A = _cum_A_t_vec(t_v - t_A_aff, V_A, theta_A)

    t1 = np.maximum(t_v - t_stim - t_E_aff, 1e-6)
    t2 = np.maximum(t_v - t_stim - t_E_aff + del_go, 1e-6)

    ABL_arr = np.full(n, float(abl))
    ILD_arr = np.full(n, float(ild))
    rl_arr = np.full(n, rate_lambda)
    T0_arr = np.full(n, T_0)
    thE_arr = np.full(n, theta_E)
    ZE_arr = np.full(n, Z_E)
    rnl_arr = np.full(n, rate_norm_l)

    int_phi_t_E_g = t_v - t_stim - t_E_aff + del_go
    int_phi_t2 = t2
    int_phi_t1 = t1
    int_phi_t_e = t1.copy()
    phi_t_e = np.ones(n)

    t_cdf_arg = t_v - t_stim - t_E_aff + del_go
    CDF_up = CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn_vec(
        t_cdf_arg,
        1,
        ABL_arr,
        ILD_arr,
        rl_arr,
        T0_arr,
        thE_arr,
        ZE_arr,
        int_phi_t_E_g,
        rnl_arr,
        is_norm,
        is_time_vary,
        K_max,
    )
    CDF_down = CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn_vec(
        t_cdf_arg,
        -1,
        ABL_arr,
        ILD_arr,
        rl_arr,
        T0_arr,
        thE_arr,
        ZE_arr,
        int_phi_t_E_g,
        rnl_arr,
        is_norm,
        is_time_vary,
        K_max,
    )
    random_readout_if_EA_survives = 0.5 * (1.0 - (CDF_up + CDF_down))

    CDF_t2_up = CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn_vec(
        t2,
        1,
        ABL_arr,
        ILD_arr,
        rl_arr,
        T0_arr,
        thE_arr,
        ZE_arr,
        int_phi_t2,
        rnl_arr,
        is_norm,
        is_time_vary,
        K_max,
    )
    CDF_t1_up = CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn_vec(
        t1,
        1,
        ABL_arr,
        ILD_arr,
        rl_arr,
        T0_arr,
        thE_arr,
        ZE_arr,
        int_phi_t1,
        rnl_arr,
        is_norm,
        is_time_vary,
        K_max,
    )
    CDF_t2_down = CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn_vec(
        t2,
        -1,
        ABL_arr,
        ILD_arr,
        rl_arr,
        T0_arr,
        thE_arr,
        ZE_arr,
        int_phi_t2,
        rnl_arr,
        is_norm,
        is_time_vary,
        K_max,
    )
    CDF_t1_down = CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn_vec(
        t1,
        -1,
        ABL_arr,
        ILD_arr,
        rl_arr,
        T0_arr,
        thE_arr,
        ZE_arr,
        int_phi_t1,
        rnl_arr,
        is_norm,
        is_time_vary,
        K_max,
    )
    P_E_plus_cum_up = CDF_t2_up - CDF_t1_up
    P_E_plus_cum_down = CDF_t2_down - CDF_t1_down

    t_rho_arg = t_v - t_stim - t_E_aff
    P_E_plus_up = rho_E_minus_small_t_NORM_rate_norm_time_varying_fn_vec(
        t_rho_arg,
        1,
        ABL_arr,
        ILD_arr,
        rl_arr,
        T0_arr,
        thE_arr,
        ZE_arr,
        phi_t_e,
        int_phi_t_e,
        rnl_arr,
        is_norm,
        is_time_vary,
        K_max,
    )
    P_E_plus_down = rho_E_minus_small_t_NORM_rate_norm_time_varying_fn_vec(
        t_rho_arg,
        -1,
        ABL_arr,
        ILD_arr,
        rl_arr,
        T0_arr,
        thE_arr,
        ZE_arr,
        phi_t_e,
        int_phi_t_e,
        rnl_arr,
        is_norm,
        is_time_vary,
        K_max,
    )

    lp = float(np.clip(lapse_prob, 0.0, 1.0))
    P_A_mix = (1.0 - lp) * P_A + lp * 0.5 * _lapse_pdf_vec(t_v, beta_lapse)
    C_A_mix = np.clip((1.0 - lp) * C_A + lp * _lapse_cdf_vec(t_v, beta_lapse), 0.0, 1.0)

    pdf_up = P_A_mix * (random_readout_if_EA_survives + P_E_plus_cum_up) + P_E_plus_up * (
        1.0 - C_A_mix
    )
    pdf_down = P_A_mix * (random_readout_if_EA_survives + P_E_plus_cum_down) + P_E_plus_down * (
        1.0 - C_A_mix
    )

    pdf_total = pdf_up + pdf_down
    eps = 1e-50
    pdf_total = np.where(np.isfinite(pdf_total) & (pdf_total > 0), pdf_total, eps)
    curve[valid] = np.where(pdf_total > eps, pdf_total, 0.0)
    return curve


def compute_one_sample_curve_from_row(
    row,
    t_pts,
    theory_params,
    is_norm,
    is_time_vary,
    K_max,
):
    (
        rate_lambda,
        T_0,
        theta_E,
        Z_E,
        bias_ms,
        abl_delay_coeff_ms_per_abl,
        abs_ild_delay_coeff_ms_per_unit,
        del_go,
        rate_norm_l,
        V_A,
        theta_A,
        t_A_aff,
        lapse_prob,
        beta_lapse,
    ) = theory_params

    t_stim = float(row["intended_fix"])
    t_led = float(row["t_LED"])
    abl = float(row["ABL"])
    ild = float(row["ILD"])
    abs_ild = float(row["abs_ILD"])
    curve = build_theory_curve_for_trial(
        t_pts=t_pts,
        t_stim=t_stim,
        abl=abl,
        ild=ild,
        rate_lambda=rate_lambda,
        T_0=T_0,
        theta_E=theta_E,
        Z_E=Z_E,
        bias_ms=bias_ms,
        abl_delay_coeff_ms_per_abl=abl_delay_coeff_ms_per_abl,
        abs_ild_delay_coeff_ms_per_unit=abs_ild_delay_coeff_ms_per_unit,
        del_go=del_go,
        rate_norm_l=rate_norm_l,
        V_A=V_A,
        theta_A=theta_A,
        t_A_aff=t_A_aff,
        lapse_prob=lapse_prob,
        beta_lapse=beta_lapse,
        is_norm=is_norm,
        is_time_vary=is_time_vary,
        K_max=K_max,
    )
    return curve, t_stim, t_led, abl, ild, abs_ild


def compute_average_curve_from_rows(
    sampled_rows,
    theory_params,
    t_pts,
    n_jobs,
    is_norm,
    is_time_vary,
    K_max,
):
    if len(sampled_rows) == 0:
        return {
            "sampled_positions": np.array([], dtype=int),
            "theory_density": np.zeros_like(t_pts, dtype=np.float64),
            "sampled_t_stim": np.array([], dtype=np.float64),
            "sampled_t_LED": np.array([], dtype=np.float64),
            "sampled_ABL": np.array([], dtype=np.float64),
            "sampled_ILD": np.array([], dtype=np.float64),
            "sampled_abs_ILD": np.array([], dtype=np.float64),
        }

    mc_results = Parallel(n_jobs=n_jobs)(
        delayed(compute_one_sample_curve_from_row)(
            row,
            t_pts,
            theory_params,
            is_norm,
            is_time_vary,
            K_max,
        )
        for _, row in sampled_rows.iterrows()
    )
    theory_matrix = np.stack([result[0] for result in mc_results], axis=0)

    return {
        "sampled_positions": sampled_rows.index.to_numpy(dtype=int, copy=True),
        "theory_density": np.mean(theory_matrix, axis=0),
        "sampled_t_stim": np.array([result[1] for result in mc_results], dtype=np.float64),
        "sampled_t_LED": np.array([result[2] for result in mc_results], dtype=np.float64),
        "sampled_ABL": np.array([result[3] for result in mc_results], dtype=np.float64),
        "sampled_ILD": np.array([result[4] for result in mc_results], dtype=np.float64),
        "sampled_abs_ILD": np.array([result[5] for result in mc_results], dtype=np.float64),
    }


def compute_mc_average_curve(
    sample_df,
    n_samples,
    rng,
    theory_params,
    t_pts,
    n_jobs,
    is_norm,
    is_time_vary,
    K_max,
):
    if len(sample_df) == 0:
        return {
            "sampled_positions": np.array([], dtype=int),
            "theory_density": np.zeros_like(t_pts, dtype=np.float64),
            "sampled_t_stim": np.array([], dtype=np.float64),
            "sampled_t_LED": np.array([], dtype=np.float64),
            "sampled_ABL": np.array([], dtype=np.float64),
            "sampled_ILD": np.array([], dtype=np.float64),
            "sampled_abs_ILD": np.array([], dtype=np.float64),
        }

    sampled_positions = rng.integers(0, len(sample_df), size=n_samples)
    sampled_rows = sample_df.iloc[sampled_positions].copy()
    result = compute_average_curve_from_rows(
        sampled_rows=sampled_rows,
        theory_params=theory_params,
        t_pts=t_pts,
        n_jobs=n_jobs,
        is_norm=is_norm,
        is_time_vary=is_time_vary,
        K_max=K_max,
    )
    result["sampled_positions"] = sampled_positions
    return result


def compute_empirical_truncated_quantiles(
    rt_values,
    quantile_levels,
    truncate_rt_wrt_stim_s,
    min_trials_for_quantiles,
):
    rt_values = np.asarray(rt_values, dtype=np.float64)
    rt_values_truncated = rt_values[(rt_values >= 0.0) & (rt_values <= truncate_rt_wrt_stim_s)]
    if len(rt_values_truncated) < int(min_trials_for_quantiles):
        return np.full(len(quantile_levels), np.nan, dtype=np.float64), int(len(rt_values_truncated))

    quantiles = np.quantile(rt_values_truncated, quantile_levels).astype(np.float64, copy=False)
    return quantiles, int(len(rt_values_truncated))


def compute_quantiles_from_truncated_density(
    t_pts_truncated,
    density_truncated,
    quantile_levels,
):
    t_pts_truncated = np.asarray(t_pts_truncated, dtype=np.float64)
    density_truncated = np.asarray(density_truncated, dtype=np.float64)
    quantile_levels = np.asarray(quantile_levels, dtype=np.float64)

    if len(t_pts_truncated) == 0 or len(density_truncated) == 0:
        return np.full(len(quantile_levels), np.nan, dtype=np.float64)
    if len(t_pts_truncated) != len(density_truncated):
        raise ValueError("t_pts_truncated and density_truncated must have the same length.")

    density_clean = np.where(np.isfinite(density_truncated) & (density_truncated > 0), density_truncated, 0.0)
    if np.all(density_clean <= 0):
        return np.full(len(quantile_levels), np.nan, dtype=np.float64)

    dt = np.diff(t_pts_truncated)
    if len(dt) == 0:
        return np.full(len(quantile_levels), np.nan, dtype=np.float64)

    cdf = np.zeros(len(t_pts_truncated), dtype=np.float64)
    cdf[1:] = np.cumsum(0.5 * (density_clean[:-1] + density_clean[1:]) * dt)
    total_area = float(cdf[-1])
    if not np.isfinite(total_area) or total_area <= 0:
        return np.full(len(quantile_levels), np.nan, dtype=np.float64)

    cdf /= total_area
    cdf = np.maximum.accumulate(np.clip(cdf, 0.0, 1.0))
    unique_cdf, unique_idx = np.unique(cdf, return_index=True)
    unique_t = t_pts_truncated[unique_idx]
    if len(unique_cdf) < 2:
        return np.full(len(quantile_levels), np.nan, dtype=np.float64)

    return np.interp(quantile_levels, unique_cdf, unique_t).astype(np.float64, copy=False)


def build_balanced_signed_ild_sample_rows(source_df, n_samples, rng, abl, abs_ild):
    if len(source_df) == 0:
        raise ValueError("source_df must contain at least one row.")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    sampled_positions = rng.integers(0, len(source_df), size=n_samples)
    sampled_rows = source_df.iloc[sampled_positions].copy()

    n_positive = n_samples // 2
    n_negative = n_samples // 2
    if n_samples % 2 == 1:
        if float(rng.random()) < 0.5:
            n_positive += 1
        else:
            n_negative += 1

    signs = np.concatenate(
        [
            np.ones(n_positive, dtype=np.float64),
            -np.ones(n_negative, dtype=np.float64),
        ]
    )
    rng.shuffle(signs)

    sampled_rows["ABL"] = float(abl)
    sampled_rows["ILD"] = signs * float(abs_ild)
    sampled_rows["abs_ILD"] = float(abs_ild)
    return sampled_rows


def build_truncated_density_payload(
    rt_values,
    n_total_condition,
    theory_density_full,
    mask_truncated,
    t_pts_truncated,
    hist_edges_truncated,
    data_bin_widths_truncated,
    truncate_rt_wrt_stim_s,
    normalize_within_window,
):
    theory_density_truncated_raw = theory_density_full[mask_truncated].copy()
    theory_area_truncated_raw = float(np.trapz(theory_density_truncated_raw, t_pts_truncated))
    if normalize_within_window and theory_area_truncated_raw > 0:
        theory_density_truncated_plot = theory_density_truncated_raw / theory_area_truncated_raw
    else:
        theory_density_truncated_plot = theory_density_truncated_raw.copy()
    theory_area_truncated_plot = float(np.trapz(theory_density_truncated_plot, t_pts_truncated))

    rt_values = np.asarray(rt_values, dtype=np.float64)
    rt_values_truncated = rt_values[(rt_values >= 0.0) & (rt_values <= truncate_rt_wrt_stim_s)]
    if n_total_condition > 0:
        data_counts_truncated, _ = np.histogram(
            rt_values_truncated,
            bins=hist_edges_truncated,
            density=False,
        )
        data_density_truncated_raw = data_counts_truncated / (
            n_total_condition * data_bin_widths_truncated
        )
    else:
        data_density_truncated_raw = np.zeros(len(hist_edges_truncated) - 1, dtype=np.float64)
    data_area_truncated_raw = float(
        np.sum(data_density_truncated_raw * data_bin_widths_truncated)
    )
    if normalize_within_window and data_area_truncated_raw > 0:
        data_density_truncated_plot = data_density_truncated_raw / data_area_truncated_raw
    else:
        data_density_truncated_plot = data_density_truncated_raw.copy()
    data_area_truncated_plot = float(
        np.sum(data_density_truncated_plot * data_bin_widths_truncated)
    )

    return {
        "n_truncated_points": int(len(rt_values_truncated)),
        "data_density_truncated_raw": data_density_truncated_raw,
        "data_density_truncated_plot": data_density_truncated_plot,
        "data_area_truncated_raw": data_area_truncated_raw,
        "data_area_truncated_plot": data_area_truncated_plot,
        "theory_density_truncated_raw": theory_density_truncated_raw,
        "theory_density_truncated_plot": theory_density_truncated_plot,
        "theory_area_truncated_raw": theory_area_truncated_raw,
        "theory_area_truncated_plot": theory_area_truncated_plot,
    }
