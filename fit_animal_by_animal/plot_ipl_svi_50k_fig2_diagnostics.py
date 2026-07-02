# %%
"""
One-row IPL diagnostic using direct patience12 50k vanilla/IPL SVI parameters.

The target Gamma/Omega scatter comes from the patience12 92-parameter
Gamma/Omega/delay SVI fit. The model row uses posterior means from the direct
IPL/vanilla condition-delay SVI fit for rate_lambda, T_0, theta_E, w, del_go,
and condition-wise t_E_aff.
"""

# %%
# =============================================================================
# Editable parameters
# =============================================================================
from collections import defaultdict
from pathlib import Path
import os
import pickle

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit
import scipy.special as scipy_special
from scipy.stats import sem as scipy_sem

from animal_wise_plotting_utils import calculate_theoretical_curves
from time_vary_norm_utils import (
    CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn_vec,
    rho_A_t_fn,
    up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn_vec,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

RAW_BATCH_DIR = REPO_DIR / "raw_data" / "batch_csvs"
ABORT_PARAMS_DIR = REPO_DIR / "aborts_ipl_npl_time_fit_results"
NPL_SVI_ROOT = (
    SCRIPT_DIR / "numpyro_svi_npl_alpha_condition_delay_patience12_restore_best_outputs"
)
BASE_3X5_PKL = NPL_SVI_ROOT / "three_npl_param_source_comparison" / "three_npl_param_sources_patience12_3x5.pkl"
IPL_SVI_ROOT = SCRIPT_DIR / "numpyro_svi_vanilla_condition_delay_patience12_min50k_restore_best_outputs"
OUTPUT_DIR = IPL_SVI_ROOT / "fig2_like_diagnostics"

OUTPUT_PNG = OUTPUT_DIR / "ipl_svi_50k_fig2_like_diagnostics.png"
OUTPUT_PKL = OUTPUT_DIR / "ipl_svi_50k_fig2_like_diagnostics.pkl"
IPL_PARAM_CSV = OUTPUT_DIR / "ipl_svi_50k_params_by_animal.csv"
IPL_SCALAR_POSTERIOR_CSV = OUTPUT_DIR / "ipl_svi_50k_scalar_posterior_summary.csv"
IPL_CONDITION_VALUES_CSV = OUTPUT_DIR / "ipl_svi_50k_gamma_omega_condition_values.csv"
IPL_CONTINUOUS_VALUES_CSV = OUTPUT_DIR / "ipl_svi_50k_gamma_omega_continuous_values.csv"
MODEL_SUMMARY_CSV = OUTPUT_DIR / "ipl_svi_50k_gamma_omega_model_summary.csv"
METRICS_CSV = OUTPUT_DIR / "ipl_svi_50k_gamma_omega_metrics.csv"

EXPECTED_N_ANIMALS = 30
EXPECTED_N_CONDITION_ROWS = 864
DESIRED_BATCHES = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]
BATCH_ORDER = {batch: idx for idx, batch in enumerate(DESIRED_BATCHES)}
ABLS = [20, 40, 60]
ILD_ARR = np.array([-16.0, -8.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0, 16.0], dtype=float)
ABS_ILD_SORTED = sorted({float(abs(ild)) for ild in ILD_ARR})
SMOOTH_ILDS = np.round(np.arange(-16.0, 16.0 + 0.05, 0.1), 10)

QUANTILES_TO_PLOT = np.round(np.arange(0.1, 1.0, 0.1), 1).tolist()
PAPER_QUANTILES_TO_PLOT = [0.1, 0.3, 0.5, 0.7, 0.9]
CONTINUOUS_ILD_STEP = float(os.environ.get("THREE_NPL_IPL_COMPARE_CONTINUOUS_ILD_STEP", "0.1"))
CONTINUOUS_ABS_ILD = np.round(np.arange(1.0, 16.0 + CONTINUOUS_ILD_STEP / 2, CONTINUOUS_ILD_STEP), 1)

K_MAX = 10
T_PTS = np.arange(-2, 2, 0.001)
N_THEORY = int(os.environ.get("THREE_NPL_IPL_COMPARE_N_THEORY", "1000"))
N_JOBS = int(os.environ.get("THREE_NPL_IPL_COMPARE_N_JOBS", str(max(1, min(8, (os.cpu_count() or 2) - 1)))))
JOBLIB_PREFER = os.environ.get("THREE_NPL_IPL_COMPARE_JOBLIB_PREFER", "processes").strip().lower()
if JOBLIB_PREFER not in {"processes", "threads"}:
    raise ValueError("THREE_NPL_IPL_COMPARE_JOBLIB_PREFER must be 'processes' or 'threads'.")
RNG_SEED = int(os.environ.get("THREE_NPL_IPL_COMPARE_RNG_SEED", "137"))

BATCH_T_TRUNC = {"LED34_even": 0.15}
DEFAULT_T_TRUNC = 0.3
CHI = 17.37

ABL_COLORS = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}


METHODS = [
    {
        "key": "ipl_svi_50k",
        "label": "Direct IPL SVI 50k",
        "short_label": "IPL SVI 50k",
    },
]
METHOD_KEY = METHODS[0]["key"]


# %%
# =============================================================================
# Helpers
# =============================================================================
def _create_innermost_dict():
    return {"empirical": [], "theoretical": []}


def _create_inner_defaultdict():
    return defaultdict(_create_innermost_dict)


def sort_pair(pair):
    batch_name, animal = pair
    return (BATCH_ORDER.get(batch_name, 999), batch_name, int(animal))


def sigmoid(x, upper, lower, x0, k):
    return lower + (upper - lower) / (1 + np.exp(-k * (x - x0)))


def fit_psychometric_sigmoid(ild_values, right_choice_probs):
    ild_values = np.asarray(ild_values, dtype=float)
    right_choice_probs = np.asarray(right_choice_probs, dtype=float)
    valid_idx = np.isfinite(ild_values) & np.isfinite(right_choice_probs)
    if np.sum(valid_idx) < 4:
        return None
    try:
        popt, _ = curve_fit(
            sigmoid,
            ild_values[valid_idx],
            right_choice_probs[valid_idx],
            p0=[1.0, 0.0, 0.0, 1.0],
            bounds=([0, 0, -np.inf, 0], [1, 1, np.inf, np.inf]),
        )
        return popt
    except Exception:
        return None


def nanmean_sem(values):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    n = int(np.sum(finite))
    if n == 0:
        return np.nan, np.nan, 0
    mean = float(np.nanmean(values))
    curr_sem = float(scipy_sem(values, nan_policy="omit")) if n > 1 else np.nan
    return mean, curr_sem, n


def pearson_r_data_vs_model(data_values, model_values):
    data_values = np.asarray(data_values, dtype=float)
    model_values = np.asarray(model_values, dtype=float)
    finite = np.isfinite(data_values) & np.isfinite(model_values)
    if np.sum(finite) < 2:
        return np.nan
    y = data_values[finite]
    yhat = model_values[finite]
    if np.nanstd(y) <= 0 or np.nanstd(yhat) <= 0:
        return np.nan
    return float(np.corrcoef(y, yhat)[0, 1])


def summarize_group_values(df, group_cols, value_cols, prefix):
    rows = []
    for group_key, group in df.groupby(group_cols, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = {col: value for col, value in zip(group_cols, group_key)}
        for value_col in value_cols:
            values = group[value_col].to_numpy(dtype=float)
            finite = np.isfinite(values)
            n = int(np.sum(finite))
            short_name = value_col.replace(prefix, "").strip("_")
            row[f"{prefix}_{short_name}_mean"] = float(np.nanmean(values)) if n else np.nan
            row[f"{prefix}_{short_name}_sd"] = float(np.nanstd(values, ddof=1)) if n > 1 else np.nan
            row[f"{prefix}_{short_name}_sem"] = float(np.nanstd(values, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
            row[f"n_{prefix}_{short_name}"] = n
        rows.append(row)
    return pd.DataFrame(rows)


def Phi_vec(x):
    return 0.5 * (1 + scipy_special.erf(x / np.sqrt(2)))


def cum_A_t_vec(t, V_A, theta_A):
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t, dtype=float)
    valid = t > 0
    t_valid = t[valid]
    out[valid] = (
        Phi_vec(V_A * (t_valid - theta_A / V_A) / np.sqrt(t_valid))
        + np.exp(2 * V_A * theta_A)
        * Phi_vec(-V_A * (t_valid + theta_A / V_A) / np.sqrt(t_valid))
    )
    return out


# %%
# =============================================================================
# IPL Gamma/Omega and vanilla RTD model functions
# =============================================================================
def gamma_omega_ipl_model(abl, ild, rate_lambda, theta_E, T_0):
    abl = np.asarray(abl, dtype=float)
    ild = np.asarray(ild, dtype=float)
    gamma = theta_E * np.tanh(rate_lambda * ild / CHI)
    omega = (1.0 / (T_0 * (theta_E**2))) * (10 ** (rate_lambda * abl / 20.0))
    omega = np.ones(np.broadcast(abl, ild).shape, dtype=float) * omega
    return gamma, omega


def cum_pro_and_reactive_vanilla_vec(t, c_A_trunc_time, abort_params, t_stim, ABL, ILD, tied_params, Z_E, t_E_aff):
    t = np.asarray(t, dtype=float)
    t_stim = np.asarray(t_stim, dtype=float)
    c_A = cum_A_t_vec(t - abort_params["t_A_aff"], abort_params["V_A"], abort_params["theta_A"])
    if c_A_trunc_time is not None:
        trunc_denom = 1 - cum_A_t_vec(
            np.array([c_A_trunc_time - abort_params["t_A_aff"]]),
            abort_params["V_A"],
            abort_params["theta_A"],
        )[0]
        c_A = np.where(t < c_A_trunc_time, 0, c_A / trunc_denom)

    c_E = (
        CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn_vec(
            t - t_stim - t_E_aff,
            1,
            ABL,
            ILD,
            tied_params["rate_lambda"],
            tied_params["T_0"],
            tied_params["theta_E"],
            Z_E,
            np.nan,
            0.0,
            False,
            False,
            K_MAX,
        )
        + CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn_vec(
            t - t_stim - t_E_aff,
            -1,
            ABL,
            ILD,
            tied_params["rate_lambda"],
            tied_params["T_0"],
            tied_params["theta_E"],
            Z_E,
            np.nan,
            0.0,
            False,
            False,
            K_MAX,
        )
    )
    return c_A + c_E - c_A * c_E


def get_theoretical_rtd_components_ipl(P_A_mean, C_A_mean, t_stim_samples, abort_params, tied_params, ABL, ILD, batch_name, t_E_aff):
    T_trunc = BATCH_T_TRUNC.get(batch_name, DEFAULT_T_TRUNC)
    Z_E = (tied_params["w"] - 0.5) * 2 * tied_params["theta_E"]

    trunc_fac_samples = (
        cum_pro_and_reactive_vanilla_vec(
            t_stim_samples + 1,
            T_trunc,
            abort_params,
            t_stim_samples,
            ABL,
            ILD,
            tied_params,
            Z_E,
            t_E_aff,
        )
        - cum_pro_and_reactive_vanilla_vec(
            t_stim_samples,
            T_trunc,
            abort_params,
            t_stim_samples,
            ABL,
            ILD,
            tied_params,
            Z_E,
            t_E_aff,
        )
        + 1e-10
    )
    trunc_factor = np.mean(trunc_fac_samples)

    mask_0_1 = (T_PTS >= 0) & (T_PTS <= 1)
    t_pts_0_1 = T_PTS[mask_0_1]
    P_A_0_1 = P_A_mean[mask_0_1]
    C_A_0_1 = C_A_mean[mask_0_1]

    up = up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn_vec(
        t_pts_0_1,
        1,
        P_A_0_1,
        C_A_0_1,
        ABL,
        ILD,
        tied_params["rate_lambda"],
        tied_params["T_0"],
        tied_params["theta_E"],
        Z_E,
        t_E_aff,
        tied_params["del_go"],
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        0.0,
        False,
        False,
        K_MAX,
    )
    down = up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn_vec(
        t_pts_0_1,
        -1,
        P_A_0_1,
        C_A_0_1,
        ABL,
        ILD,
        tied_params["rate_lambda"],
        tied_params["T_0"],
        tied_params["theta_E"],
        Z_E,
        t_E_aff,
        tied_params["del_go"],
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        0.0,
        False,
        False,
        K_MAX,
    )
    return t_pts_0_1, np.maximum(up / trunc_factor, 0), np.maximum(down / trunc_factor, 0)


# %%
# =============================================================================
# Data loading
# =============================================================================
def load_base_payload():
    if not BASE_3X5_PKL.exists():
        raise FileNotFoundError(f"Run compare_three_npl_param_sources_patience12_3x5.py first: {BASE_3X5_PKL}")
    with BASE_3X5_PKL.open("rb") as handle:
        payload = pickle.load(handle)
    required_keys = [
        "methods",
        "animal_keys",
        "big_condition_rows",
        "big_scalar_rows",
        "method_param_rows",
        "gamma_omega_metrics",
        "gamma_omega_model_summary",
        "psy_by_method",
        "slopes_by_method",
        "quantile_by_method",
    ]
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise KeyError(f"{BASE_3X5_PKL} missing keys: {missing}")
    return payload


def load_raw_data():
    frames = []
    for batch_name in DESIRED_BATCHES:
        csv_path = RAW_BATCH_DIR / f"batch_{batch_name}_valid_and_aborts.csv"
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        frames.append(pd.read_csv(csv_path))
    raw_df = pd.concat(frames, ignore_index=True)
    raw_df["animal"] = raw_df["animal"].astype(int)
    raw_df["ABL"] = raw_df["ABL"].astype(int)
    raw_df["ILD"] = raw_df["ILD"].astype(float)
    return raw_df


def load_abort_params(batch_name, animal):
    pkl_path = ABORT_PARAMS_DIR / f"results_{batch_name}_animal_{int(animal)}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(pkl_path)
    with pkl_path.open("rb") as handle:
        fit_results = pickle.load(handle)
    abort_samples = fit_results["vbmc_aborts_results"]
    return {
        "V_A": float(np.mean(abort_samples["V_A_samples"])),
        "theta_A": float(np.mean(abort_samples["theta_A_samples"])),
        "t_A_aff": float(np.mean(abort_samples["t_A_aff_samp"])),
    }


def load_direct_ipl_svi_params(animal_keys, big_condition_df):
    param_rows = []
    scalar_summary_rows = []
    condition_rows = []
    continuous_rows = []
    params_by_animal = {}

    for batch_name, animal in animal_keys:
        summary_csv = IPL_SVI_ROOT / f"{batch_name}_{int(animal)}" / "main_fullrank_posterior_summary.csv"
        if not summary_csv.exists():
            raise FileNotFoundError(summary_csv)
        summary_df = pd.read_csv(summary_csv)
        summary_df["parameter"] = summary_df["parameter"].astype(str)

        scalar_values = {}
        for param_name in ["rate_lambda", "T_0", "theta_E", "w", "del_go"]:
            rows = summary_df[summary_df["parameter"] == param_name]
            if len(rows) != 1:
                raise RuntimeError(f"{summary_csv}: expected one row for {param_name}, found {len(rows)}")
            row = rows.iloc[0]
            scalar_values[param_name] = float(row["mean"])
            scalar_summary_rows.append(
                {
                    "method_key": METHOD_KEY,
                    "batch_name": batch_name,
                    "animal": int(animal),
                    "parameter": param_name,
                    "mean": float(row["mean"]),
                    "sd": float(row["sd"]),
                    "q025": float(row["q025"]),
                    "q500": float(row["q500"]),
                    "q975": float(row["q975"]),
                    "n_samples": int(row["n_samples"]),
                    "n_finite": int(row["n_finite"]),
                }
            )

        delay_df = summary_df[
            summary_df["parameter"].str.startswith("t_E_aff_")
            & summary_df["ABL"].notna()
            & summary_df["ILD"].notna()
        ].copy()
        if delay_df.empty:
            raise RuntimeError(f"{summary_csv}: no condition-wise t_E_aff rows found.")
        delay_map = {
            (int(float(row.ABL)), float(row.ILD)): float(row.mean)
            for row in delay_df.itertuples(index=False)
        }
        params = dict(scalar_values)
        params["delay_by_condition"] = delay_map
        params_by_animal[(batch_name, int(animal))] = params

        param_rows.append(
            {
                "method_key": METHOD_KEY,
                "method_label": METHODS[0]["label"],
                "batch_name": batch_name,
                "animal": int(animal),
                "rate_lambda": params["rate_lambda"],
                "T_0": params["T_0"],
                "theta_E": params["theta_E"],
                "rate_norm_l": np.nan,
                "alpha": np.nan,
                "w": params["w"],
                "del_go": params["del_go"],
                "n_delay_conditions": int(len(delay_map)),
            }
        )

        animal_cond = big_condition_df[
            (big_condition_df["batch_name"].astype(str) == str(batch_name))
            & (big_condition_df["animal"].astype(int) == int(animal))
        ].copy()
        pred_cond = model_values_for_condition_rows_ipl(METHOD_KEY, {(batch_name, int(animal)): params}, animal_cond)
        condition_rows.extend(pred_cond.to_dict("records"))
        pred_smooth = model_curves_for_params_ipl(METHOD_KEY, {(batch_name, int(animal)): params}, SMOOTH_ILDS)
        continuous_rows.extend(pred_smooth.to_dict("records"))

    param_df = pd.DataFrame(param_rows).sort_values(["batch_name", "animal"]).reset_index(drop=True)
    if len(param_df) != EXPECTED_N_ANIMALS:
        raise RuntimeError(f"Expected {EXPECTED_N_ANIMALS} IPL SVI animals, found {len(param_df)}")
    return (
        param_df,
        pd.DataFrame(scalar_summary_rows),
        pd.DataFrame(condition_rows),
        pd.DataFrame(continuous_rows),
        params_by_animal,
    )


def model_curves_for_params_ipl(method_key, params_by_animal, ild_grid):
    rows = []
    for (batch_name, animal), params in sorted(params_by_animal.items(), key=lambda item: sort_pair(item[0])):
        for abl in ABLS:
            gamma, omega = gamma_omega_ipl_model(
                abl,
                np.asarray(ild_grid, dtype=float),
                params["rate_lambda"],
                params["theta_E"],
                params["T_0"],
            )
            for ild, curr_gamma, curr_omega in zip(ild_grid, gamma, omega):
                rows.append(
                    {
                        "method_key": method_key,
                        "batch_name": batch_name,
                        "animal": int(animal),
                        "ABL": int(abl),
                        "ILD": float(ild),
                        "model_gamma": float(curr_gamma),
                        "model_omega": float(curr_omega),
                    }
                )
    return pd.DataFrame(rows)


def model_values_for_condition_rows_ipl(method_key, params_by_animal, condition_df):
    rows = []
    for (batch_name, animal), animal_cond in condition_df.groupby(["batch_name", "animal"], sort=True):
        pair = (str(batch_name), int(animal))
        params = params_by_animal[pair]
        gamma, omega = gamma_omega_ipl_model(
            animal_cond["ABL"].to_numpy(dtype=float),
            animal_cond["ILD"].to_numpy(dtype=float),
            params["rate_lambda"],
            params["theta_E"],
            params["T_0"],
        )
        curr = animal_cond[["batch_name", "animal", "ABL", "ILD"]].copy()
        curr["method_key"] = method_key
        curr["model_gamma"] = np.asarray(gamma, dtype=float)
        curr["model_omega"] = np.asarray(omega, dtype=float)
        rows.extend(curr.to_dict("records"))
    return pd.DataFrame(rows)


def compute_gamma_omega_metrics(condition_df, model_condition_df):
    merged = condition_df.merge(
        model_condition_df,
        on=["batch_name", "animal", "ABL", "ILD"],
        how="inner",
        validate="one_to_one",
    )
    rows = []
    for method_key in sorted(merged["method_key"].unique()):
        method_subset = merged[merged["method_key"] == method_key]
        for param in ["gamma", "omega"]:
            for abl in ABLS + ["all"]:
                subset = method_subset if abl == "all" else method_subset[method_subset["ABL"] == abl]
                target = subset[f"condition_{param}"].to_numpy(dtype=float)
                pred = subset[f"model_{param}"].to_numpy(dtype=float)
                finite = np.isfinite(target) & np.isfinite(pred)
                if int(np.sum(finite)) >= 2:
                    diff = target[finite] - pred[finite]
                    corr = float(np.corrcoef(target[finite], pred[finite])[0, 1])
                    rows.append(
                        {
                            "method_key": method_key,
                            "parameter": param,
                            "ABL": abl,
                            "n_points": int(np.sum(finite)),
                            "rmse": float(np.sqrt(np.mean(diff**2))),
                            "mae": float(np.mean(np.abs(diff))),
                            "pearson_r": corr,
                        }
                    )
    return pd.DataFrame(rows)


# %%
# =============================================================================
# IPL psychometric and quantile calculations
# =============================================================================
def interpolate_t_e_aff(delay_by_condition, ABL, signed_ild, flat_after_max=False):
    signed_ild = float(signed_ild)
    exact = delay_by_condition.get((int(ABL), signed_ild))
    if exact is not None:
        return exact

    sign = np.sign(signed_ild)
    if sign == 0:
        return None

    branch = sorted(
        [(abs(ild), delay) for (abl, ild), delay in delay_by_condition.items() if int(abl) == int(ABL) and np.sign(ild) == sign],
        key=lambda item: item[0],
    )
    if len(branch) < 2:
        return None

    x = np.array([item[0] for item in branch], dtype=float)
    y = np.array([item[1] for item in branch], dtype=float)
    abs_ild = abs(signed_ild)
    if abs_ild < x.min():
        return None
    if abs_ild > x.max():
        if flat_after_max:
            return float(y[-1])
        return None
    return float(np.interp(abs_ild, x, y))


def get_p_a_c_a(batch_df, animal, abort_params):
    animal_df = batch_df[batch_df["animal"] == int(animal)].copy()
    np.random.seed(RNG_SEED + int(animal))
    return calculate_theoretical_curves(
        animal_df,
        N_THEORY,
        T_PTS,
        abort_params["t_A_aff"],
        abort_params["V_A"],
        abort_params["theta_A"],
        rho_A_t_fn,
    )


def quantiles_from_rtd(t_pts, rtd):
    rtd = np.asarray(rtd, dtype=float)
    rtd = np.where(np.isfinite(rtd), np.maximum(rtd, 0), 0)
    if len(t_pts) < 2 or np.sum(rtd) <= 0:
        return np.full(len(QUANTILES_TO_PLOT), np.nan)
    cdf = np.cumsum(rtd) * (t_pts[1] - t_pts[0])
    if cdf[-1] <= 1e-8:
        return np.full(len(QUANTILES_TO_PLOT), np.nan)
    cdf = cdf / cdf[-1]
    values = []
    for q in QUANTILES_TO_PLOT:
        idx = np.searchsorted(cdf, q, side="left")
        if idx == 0:
            values.append(t_pts[0])
        elif idx >= len(cdf):
            values.append(t_pts[-1])
        else:
            x1, x2 = t_pts[idx - 1], t_pts[idx]
            y1, y2 = cdf[idx - 1], cdf[idx]
            values.append(x1 if y2 == y1 else x1 + (x2 - x1) * (q - y1) / (y2 - y1))
    return np.asarray(values, dtype=float)


def empirical_psychometric(batch_df, animal, ABL):
    df = batch_df[
        (batch_df["animal"] == int(animal))
        & (batch_df["ABL"] == int(ABL))
        & (batch_df["success"].isin([1, -1]))
        & (batch_df["RTwrtStim"] <= 1)
    ].copy()
    if df.empty:
        return {}

    out = {}
    for ild, ild_df in df.groupby("ILD", sort=True):
        out[float(ild)] = float(np.mean(ild_df["choice"] == 1))
    return out


def empirical_rt_quantiles(batch_df, animal, ABL, ILD):
    df = batch_df[
        (batch_df["animal"] == int(animal))
        & (batch_df["ABL"] == int(ABL))
        & (batch_df["ILD"] == float(ILD))
        & (batch_df["success"].isin([1, -1]))
        & (batch_df["RTwrtStim"] >= 0)
        & (batch_df["RTwrtStim"] <= 1)
    ]
    if len(df) <= 5:
        return np.full(len(QUANTILES_TO_PLOT), np.nan)
    return np.quantile(df["RTwrtStim"].to_numpy(dtype=float), QUANTILES_TO_PLOT)


def process_single_ipl(batch_name, animal, tied_params, empirical_psy, empirical_quantiles, P_A_mean, C_A_mean, t_stim_samples, abort_params):
    delay_by_condition = tied_params["delay_by_condition"]
    theory_psy = {ABL: {} for ABL in ABLS}
    discrete_quantiles = {}
    continuous_quantiles = {}
    continuous_quantiles_sd_flat = {}

    for ABL in ABLS:
        for ILD in ILD_ARR:
            raw_q = empirical_quantiles[(ABL, float(ILD))]
            delay = delay_by_condition.get((int(ABL), float(ILD)))
            theory_q = np.full(len(QUANTILES_TO_PLOT), np.nan)

            if delay is not None and not (batch_name == "SD" and abs(ILD) > 8):
                try:
                    t_pts, up, down = get_theoretical_rtd_components_ipl(
                        P_A_mean,
                        C_A_mean,
                        t_stim_samples,
                        abort_params,
                        tied_params,
                        ABL,
                        float(ILD),
                        batch_name,
                        delay,
                    )
                    up_area = trapezoid(up, t_pts)
                    down_area = trapezoid(down, t_pts)
                    if up_area + down_area > 0:
                        theory_psy[ABL][float(ILD)] = float(up_area / (up_area + down_area))
                    theory_q = quantiles_from_rtd(t_pts, up + down)
                except Exception as exc:
                    print(f"  IPL theory failed {batch_name}/{animal} ABL={ABL} ILD={ILD}: {exc}")

            discrete_quantiles[(ABL, float(ILD))] = {"empirical": raw_q, "theoretical": theory_q}

    for ABL in ABLS:
        for abs_ild in CONTINUOUS_ABS_ILD:
            sign_quantiles = []
            sign_quantiles_sd_flat = []
            for sign in [-1, 1]:
                signed_ild = float(sign * abs_ild)
                delay = interpolate_t_e_aff(delay_by_condition, ABL, signed_ild)
                quantile_for_standard_delay = None
                if delay is not None:
                    try:
                        t_pts, up, down = get_theoretical_rtd_components_ipl(
                            P_A_mean,
                            C_A_mean,
                            t_stim_samples,
                            abort_params,
                            tied_params,
                            ABL,
                            signed_ild,
                            batch_name,
                            delay,
                        )
                        quantile_for_standard_delay = quantiles_from_rtd(t_pts, up + down)
                        sign_quantiles.append(quantile_for_standard_delay)
                    except Exception:
                        quantile_for_standard_delay = None

                flat_delay = interpolate_t_e_aff(
                    delay_by_condition,
                    ABL,
                    signed_ild,
                    flat_after_max=(batch_name == "SD"),
                )
                if flat_delay is not None:
                    if (
                        quantile_for_standard_delay is not None
                        and delay is not None
                        and np.isclose(flat_delay, delay)
                    ):
                        sign_quantiles_sd_flat.append(quantile_for_standard_delay)
                    else:
                        try:
                            t_pts, up, down = get_theoretical_rtd_components_ipl(
                                P_A_mean,
                                C_A_mean,
                                t_stim_samples,
                                abort_params,
                                tied_params,
                                ABL,
                                signed_ild,
                                batch_name,
                                flat_delay,
                            )
                            sign_quantiles_sd_flat.append(quantiles_from_rtd(t_pts, up + down))
                        except Exception:
                            pass

            if sign_quantiles:
                continuous_quantiles[(ABL, float(abs_ild))] = np.nanmean(sign_quantiles, axis=0)
            if sign_quantiles_sd_flat:
                continuous_quantiles_sd_flat[(ABL, float(abs_ild))] = np.nanmean(sign_quantiles_sd_flat, axis=0)

    return {
        "pair": (batch_name, animal),
        "empirical_psy": empirical_psy,
        "theory_psy": theory_psy,
        "discrete_quantiles": discrete_quantiles,
        "continuous_quantiles": continuous_quantiles,
        "continuous_quantiles_sd_flat": continuous_quantiles_sd_flat,
    }


def process_animal_ipl(pair, raw_df, ipl_params):
    batch_name, animal = pair
    batch_df = raw_df[raw_df["batch_name"] == batch_name].copy()
    abort_params = load_abort_params(batch_name, animal)
    P_A_mean, C_A_mean, t_stim_samples = get_p_a_c_a(batch_df, animal, abort_params)
    empirical_psy = {ABL: empirical_psychometric(batch_df, animal, ABL) for ABL in ABLS}
    empirical_quantiles = {
        (ABL, float(ILD)): empirical_rt_quantiles(batch_df, animal, ABL, ILD)
        for ABL in ABLS
        for ILD in ILD_ARR
    }
    return process_single_ipl(
        batch_name,
        animal,
        ipl_params[pair],
        empirical_psy,
        empirical_quantiles,
        P_A_mean,
        C_A_mean,
        t_stim_samples,
        abort_params,
    )


def aggregate_psychometric(results, animal_keys):
    empirical_agg = {ABL: np.full((len(animal_keys), len(ILD_ARR)), np.nan) for ABL in ABLS}
    theory_agg = {ABL: np.full((len(animal_keys), len(ILD_ARR)), np.nan) for ABL in ABLS}

    for animal_idx, result in enumerate(results):
        empirical_psy = result["empirical_psy"]
        theory_psy = result["theory_psy"]
        for ABL in ABLS:
            for ild_idx, ild in enumerate(ILD_ARR):
                if float(ild) in empirical_psy[ABL]:
                    empirical_agg[ABL][animal_idx, ild_idx] = empirical_psy[ABL][float(ild)]
                if float(ild) in theory_psy[ABL]:
                    theory_agg[ABL][animal_idx, ild_idx] = theory_psy[ABL][float(ild)]
    return {"empirical_agg": empirical_agg, "theory_agg": theory_agg, "animal_keys": animal_keys}


def aggregate_slopes(psy_data):
    empirical_agg = psy_data["empirical_agg"]
    theory_agg = psy_data["theory_agg"]
    data_means = []
    model_means = []
    slopes_data = {}
    slopes_model = {}
    for animal_idx, pair in enumerate(psy_data["animal_keys"]):
        data_slopes = []
        model_slopes = []
        slopes_data[pair] = {}
        slopes_model[pair] = {}
        for ABL in ABLS:
            data_fit = fit_psychometric_sigmoid(ILD_ARR, empirical_agg[ABL][animal_idx])
            model_fit = fit_psychometric_sigmoid(ILD_ARR, theory_agg[ABL][animal_idx])
            data_slope = float(data_fit[3]) if data_fit is not None else np.nan
            model_slope = float(model_fit[3]) if model_fit is not None else np.nan
            slopes_data[pair][ABL] = data_slope
            slopes_model[pair][ABL] = model_slope
            data_slopes.append(data_slope)
            model_slopes.append(model_slope)
        data_means.append(np.nanmean(data_slopes) if np.any(np.isfinite(data_slopes)) else np.nan)
        model_means.append(np.nanmean(model_slopes) if np.any(np.isfinite(model_slopes)) else np.nan)

    return {
        "data_means": np.asarray(data_means, dtype=float),
        "model_means": np.asarray(model_means, dtype=float),
        "slopes_data": slopes_data,
        "slopes_model": slopes_model,
        "animal_keys": psy_data["animal_keys"],
    }


def aggregate_quantiles(results, animal_keys):
    plot_data = defaultdict(_create_inner_defaultdict)
    continuous_plot_data = defaultdict(_create_inner_defaultdict)
    continuous_plot_data_sd_flat = defaultdict(_create_inner_defaultdict)

    for result in results:
        discrete_quantiles = result["discrete_quantiles"]
        continuous_quantiles = result["continuous_quantiles"]
        continuous_quantiles_sd_flat = result["continuous_quantiles_sd_flat"]

        for ABL in ABLS:
            for abs_ild in ABS_ILD_SORTED:
                emp_sign_values = []
                theory_sign_values = []
                for sign in [-1, 1]:
                    key = (ABL, float(sign * abs_ild))
                    if key not in discrete_quantiles:
                        continue
                    emp_sign_values.append(discrete_quantiles[key]["empirical"])
                    theory_sign_values.append(discrete_quantiles[key]["theoretical"])

                if emp_sign_values:
                    emp_array = np.asarray(emp_sign_values, dtype=float)
                    if np.any(np.isfinite(emp_array)):
                        plot_data[ABL][abs_ild]["empirical"].append(np.nanmean(emp_array, axis=0))
                if theory_sign_values:
                    theory_array = np.asarray(theory_sign_values, dtype=float)
                    if np.any(np.isfinite(theory_array)):
                        theory_mean = np.nanmean(theory_array, axis=0)
                        plot_data[ABL][abs_ild]["theoretical"].append(theory_mean)

            for abs_ild in CONTINUOUS_ABS_ILD:
                key = (ABL, float(abs_ild))
                if key in continuous_quantiles and np.any(np.isfinite(continuous_quantiles[key])):
                    continuous_plot_data[ABL][float(abs_ild)]["theoretical"].append(continuous_quantiles[key])
                if key in continuous_quantiles_sd_flat and np.any(np.isfinite(continuous_quantiles_sd_flat[key])):
                    continuous_plot_data_sd_flat[ABL][float(abs_ild)]["theoretical"].append(
                        continuous_quantiles_sd_flat[key]
                    )

    plot_data_for_pickle = {
        ABL: {
            abs_ild: {
                "empirical": list(plot_data[ABL][abs_ild]["empirical"]),
                "theoretical": list(plot_data[ABL][abs_ild]["theoretical"]),
            }
            for abs_ild in ABS_ILD_SORTED
        }
        for ABL in ABLS
    }
    continuous_plot_data_for_pickle = {
        ABL: {
            float(abs_ild): {
                "empirical": list(continuous_plot_data[ABL][float(abs_ild)]["empirical"]),
                "theoretical": list(continuous_plot_data[ABL][float(abs_ild)]["theoretical"]),
            }
            for abs_ild in CONTINUOUS_ABS_ILD
        }
        for ABL in ABLS
    }
    continuous_plot_data_sd_flat_for_pickle = {
        ABL: {
            float(abs_ild): {
                "empirical": list(continuous_plot_data_sd_flat[ABL][float(abs_ild)]["empirical"]),
                "theoretical": list(continuous_plot_data_sd_flat[ABL][float(abs_ild)]["theoretical"]),
            }
            for abs_ild in CONTINUOUS_ABS_ILD
        }
        for ABL in ABLS
    }

    return {
        "plot_data": plot_data_for_pickle,
        "continuous_plot_data": continuous_plot_data_for_pickle,
        "continuous_plot_data_sd_flat": continuous_plot_data_sd_flat_for_pickle,
        "continuous_abs_ild": CONTINUOUS_ABS_ILD.tolist(),
        "QUANTILES_TO_PLOT": QUANTILES_TO_PLOT,
        "animal_keys": animal_keys,
    }


# %%
# =============================================================================
# Plotting
# =============================================================================
def metric_for(metrics_df, method_key, param):
    row = metrics_df[
        (metrics_df["method_key"] == method_key)
        & (metrics_df["parameter"] == param)
        & (metrics_df["ABL"].astype(str) == "all")
    ]
    if len(row) != 1:
        return None
    return row.iloc[0]


def plot_gamma_omega(ax, param, method_key, condition_summary_df, method_summary_df, metrics_df, show_xlabel, column_title=None):
    ylabel = "Gamma" if param == "gamma" else "Omega"
    for abl in ABLS:
        color = ABL_COLORS[abl]
        cond_subset = condition_summary_df[condition_summary_df["ABL"] == abl].sort_values("ILD")
        ax.errorbar(
            cond_subset["ILD"],
            cond_subset[f"condition_{param}_mean"],
            yerr=cond_subset[f"condition_{param}_sem"],
            fmt="o",
            ms=3.5,
            mfc="white",
            mec=color,
            ecolor=color,
            color=color,
            capsize=1.5,
            linestyle="none",
            alpha=0.95,
        )

        model_subset = method_summary_df[
            (method_summary_df["method_key"] == method_key)
            & (method_summary_df["ABL"] == abl)
        ].sort_values("ILD")
        x = model_subset["ILD"].to_numpy(dtype=float)
        y = model_subset[f"model_{param}_mean"].to_numpy(dtype=float)
        curr_sem = model_subset[f"model_{param}_sem"].to_numpy(dtype=float)
        ax.plot(x, y, color=color, linestyle="-", linewidth=1.3)
        ax.fill_between(x, y - curr_sem, y + curr_sem, color=color, alpha=0.13, linewidth=0)

    metric = metric_for(metrics_df, method_key, param)
    title_text = ""
    if metric is not None:
        title_text = f"RMSE={metric['rmse']:.3g}, r={metric['pearson_r']:.2f}"
    if column_title is not None:
        title_text = f"{column_title}\n{title_text}" if title_text else column_title
    if title_text:
        ax.set_title(title_text, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlim(-17, 17)
    ax.set_xticks([-16, -8, -4, -2, -1, 1, 2, 4, 8, 16])
    ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    ax.tick_params(axis="y", labelsize=8)
    if show_xlabel:
        ax.set_xlabel("ILD", fontsize=9)
    if param == "gamma":
        ax.axhline(0.0, color="0.75", linewidth=0.8, zorder=0)
    if param == "omega":
        ax.set_ylim(bottom=2)
    ax.grid(True, alpha=0.2)


def plot_psychometric(ax, data, show_xlabel):
    empirical_agg = data["empirical_agg"]
    theory_agg = data["theory_agg"]

    for ABL in ABLS:
        color = ABL_COLORS[ABL]
        emp = empirical_agg[ABL]
        theo = theory_agg[ABL]
        emp_mean = np.nanmean(emp, axis=0)
        n_emp = np.sum(np.isfinite(emp), axis=0)
        emp_sem = np.nanstd(emp, axis=0, ddof=1) / np.sqrt(n_emp)

        ax.errorbar(
            ILD_ARR,
            emp_mean,
            yerr=emp_sem,
            fmt="o",
            color=color,
            capsize=0,
            markersize=3.5,
            linestyle="none",
        )

        theo_mean = np.nanmean(theo, axis=0)
        popt = fit_psychometric_sigmoid(ILD_ARR, theo_mean)
        if popt is not None:
            valid_ilds = ILD_ARR[np.isfinite(theo_mean)]
            ilds_smooth = np.linspace(np.nanmin(valid_ilds), np.nanmax(valid_ilds), 200)
            ax.plot(ilds_smooth, sigmoid(ilds_smooth, *popt), "-", color=color, linewidth=1.2)
        ax.plot(ILD_ARR, theo_mean, "x", color=color, markersize=4, alpha=0.75)

    if show_xlabel:
        ax.set_xlabel("ILD", fontsize=9)
    ax.set_ylabel("P(right)", fontsize=9)
    ax.set_xticks([-16, -8, -4, -2, -1, 1, 2, 4, 8, 16])
    ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    ax.tick_params(axis="y", labelsize=8)
    ax.axvline(0, alpha=0.45, color="grey", linestyle="--", linewidth=0.9)
    ax.axhline(0.5, alpha=0.45, color="grey", linestyle="--", linewidth=0.9)
    ax.set_ylim(-0.05, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.16)


def plot_slopes(ax, data, show_xlabel, column_title=None):
    data_means = data["data_means"]
    model_means = data["model_means"]

    ax.scatter(data_means, model_means, marker="o", s=34, facecolors="w", edgecolors="k", linewidths=1.0)
    ax.set_ylabel("Model", fontsize=9)
    if show_xlabel:
        ax.set_xlabel("Data", fontsize=9)
    ax.set_xticks([0.1, 0.5, 0.9])
    ax.set_yticks([0.1, 0.5, 0.9])
    ax.set_xlim(0.1, 0.9)
    ax.set_ylim(0.1, 0.9)
    ax.tick_params(axis="both", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.plot([0.1, 0.9], [0.1, 0.9], color="grey", alpha=0.5, linestyle="--", linewidth=1.2, zorder=0)
    ax.grid(True, alpha=0.16)

    corr = pearson_r_data_vs_model(data_means, model_means)
    title_text = ""
    if np.isfinite(corr):
        title_text = f"r={corr:.2f}"
    if column_title is not None:
        title_text = f"{column_title}\n{title_text}" if title_text else column_title
    if title_text:
        ax.set_title(title_text, fontsize=9)


def quantile_indices_for_plot(data, quantiles_to_show):
    all_quantiles = np.asarray(data["QUANTILES_TO_PLOT"], dtype=float)
    indices = []
    labels = []
    for q in quantiles_to_show:
        matches = np.where(np.isclose(all_quantiles, float(q)))[0]
        if len(matches) != 1:
            raise ValueError(f"Requested quantile {q} not found in {all_quantiles.tolist()}")
        indices.append(int(matches[0]))
        labels.append(float(all_quantiles[matches[0]]))
    return indices, labels


def plot_quantiles(ax, data, show_xlabel):
    plot_data = data["plot_data"]
    theory_source = data["continuous_plot_data_sd_flat"]
    theory_x = data["continuous_abs_ild"]
    quantile_indices, quantile_labels = quantile_indices_for_plot(data, PAPER_QUANTILES_TO_PLOT)

    for plot_idx, (q_idx, q) in enumerate(zip(quantile_indices, quantile_labels)):
        emp_means, emp_sems = [], []
        for abs_ild in ABS_ILD_SORTED:
            values = []
            for ABL in ABLS:
                entries = plot_data[ABL][abs_ild]["empirical"]
                if len(entries) > 0:
                    values.extend(np.asarray(entries, dtype=float)[:, q_idx])
            mean, curr_sem, _n = nanmean_sem(values)
            emp_means.append(mean)
            emp_sems.append(curr_sem)

        ax.errorbar(
            ABS_ILD_SORTED,
            emp_means,
            yerr=emp_sems,
            fmt="o",
            color="black",
            markersize=3.5,
            capsize=0,
            alpha=0.86,
            label=f"Data q={q:.1f}" if plot_idx == 0 else "_nolegend_",
        )

        theo_means, theo_sems, theo_x_valid = [], [], []
        for abs_ild in theory_x:
            values = []
            for ABL in ABLS:
                entries = theory_source[ABL][float(abs_ild)]["theoretical"]
                if len(entries) > 0:
                    values.extend(np.asarray(entries, dtype=float)[:, q_idx])
            mean, curr_sem, n = nanmean_sem(values)
            if n > 0:
                theo_x_valid.append(float(abs_ild))
                theo_means.append(mean)
                theo_sems.append(curr_sem)

        if theo_x_valid:
            ax.plot(
                theo_x_valid,
                theo_means,
                color="tab:red",
                linestyle="-",
                linewidth=1.0,
                label=f"Model q={q:.1f}" if plot_idx == 0 else "_nolegend_",
            )
            ax.fill_between(
                theo_x_valid,
                np.asarray(theo_means) - np.asarray(theo_sems),
                np.asarray(theo_means) + np.asarray(theo_sems),
                color="tab:red",
                alpha=0.12,
                linewidth=0,
            )

    if show_xlabel:
        ax.set_xlabel("|ILD|", fontsize=9)
    ax.set_ylabel("RT quantile (s)", fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ABS_ILD_SORTED)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_yticks([0.1, 0.2, 0.3, 0.4])
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.16)


def make_figure(condition_summary_df, method_summary_df, metrics_df, psy_by_method, slopes_by_method, quantile_by_method):
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.6))
    axes = np.asarray(axes).reshape(1, 5)
    column_titles = [
        "Gamma vs ILD",
        "Omega vs ILD",
        "Psychometric",
        "Psychometric slope",
        "RT quantiles",
    ]
    for row_idx, method in enumerate(METHODS):
        method_key = method["key"]
        show_xlabel = row_idx == len(METHODS) - 1
        plot_gamma_omega(
            axes[row_idx, 0],
            "gamma",
            method_key,
            condition_summary_df,
            method_summary_df,
            metrics_df,
            show_xlabel,
            column_title=column_titles[0] if row_idx == 0 else None,
        )
        plot_gamma_omega(
            axes[row_idx, 1],
            "omega",
            method_key,
            condition_summary_df,
            method_summary_df,
            metrics_df,
            show_xlabel,
            column_title=column_titles[1] if row_idx == 0 else None,
        )
        if row_idx == 0:
            axes[row_idx, 2].set_title(column_titles[2], fontsize=9)
        plot_psychometric(axes[row_idx, 2], psy_by_method[method_key], show_xlabel)
        plot_slopes(
            axes[row_idx, 3],
            slopes_by_method[method_key],
            show_xlabel,
            column_title=column_titles[3] if row_idx == 0 else None,
        )
        if row_idx == 0:
            axes[row_idx, 4].set_title(column_titles[4], fontsize=9)
        plot_quantiles(axes[row_idx, 4], quantile_by_method[method_key], show_xlabel)

        axes[row_idx, 0].text(
            -0.28,
            0.5,
            method["label"],
            transform=axes[row_idx, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

    abl_handles = [
        Line2D([0], [0], color=ABL_COLORS[abl], lw=1.5, marker="o", label=f"ABL {abl}")
        for abl in ABLS
    ]
    source_handles = [
        Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor="black", linestyle="none", label="92-param Gamma/Omega mean"),
        Line2D([0], [0], color="black", lw=1.4, label="model functional mean"),
        Line2D([0], [0], marker="x", color="black", linestyle="none", label="psychometric model points"),
        Line2D([0], [0], marker="o", color="black", linestyle="none", label="RT quantile data"),
        Line2D([0], [0], color="tab:red", lw=1.2, label="RT quantile model"),
    ]
    fig.legend(
        handles=abl_handles + source_handles,
        loc="upper center",
        ncol=7,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        fontsize=9,
    )
    fig.suptitle(
        "Direct IPL SVI 50k tested against Gamma/Omega, psychometric, slopes, and paper RT quantiles",
        fontsize=15,
        y=1.08,
    )
    fig.tight_layout(rect=[0.035, 0.02, 1.0, 0.9], w_pad=1.3, h_pad=1.1)
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    return fig


# %%
# =============================================================================
# Main analysis
# =============================================================================
print(f"Base 3x5 payload: {BASE_3X5_PKL}")
print(f"IPL SVI 50k root: {IPL_SVI_ROOT}")
print(f"Output folder: {OUTPUT_DIR}")
print(f"N_THEORY={N_THEORY}, N_JOBS={N_JOBS}, joblib prefer={JOBLIB_PREFER}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

base_payload = load_base_payload()
animal_keys = [(str(batch), int(animal)) for batch, animal in base_payload["animal_keys"]]
if len(animal_keys) != EXPECTED_N_ANIMALS:
    raise RuntimeError(f"Expected {EXPECTED_N_ANIMALS} animals, found {len(animal_keys)} in base payload.")
animal_keys = sorted(animal_keys, key=sort_pair)

big_condition_df = pd.DataFrame(base_payload["big_condition_rows"])
big_scalar_df = pd.DataFrame(base_payload["big_scalar_rows"])
big_condition_df["batch_name"] = big_condition_df["batch_name"].astype(str)
big_condition_df["animal"] = big_condition_df["animal"].astype(int)
big_condition_df["ABL"] = big_condition_df["ABL"].astype(int)
big_condition_df["ILD"] = big_condition_df["ILD"].astype(float)
if len(big_condition_df) != EXPECTED_N_CONDITION_ROWS:
    raise RuntimeError(f"Expected {EXPECTED_N_CONDITION_ROWS} big condition rows, found {len(big_condition_df)}")

ipl_fit_df, ipl_scalar_summary_df, ipl_condition_model_df, ipl_smooth_model_df, ipl_params = load_direct_ipl_svi_params(
    animal_keys,
    big_condition_df,
)
ipl_fit_df.to_csv(IPL_PARAM_CSV, index=False)
ipl_scalar_summary_df.to_csv(IPL_SCALAR_POSTERIOR_CSV, index=False)
ipl_condition_model_df.to_csv(IPL_CONDITION_VALUES_CSV, index=False)
ipl_smooth_model_df.to_csv(IPL_CONTINUOUS_VALUES_CSV, index=False)
print(f"Saved direct IPL SVI params: {IPL_PARAM_CSV}")
print(f"Saved direct IPL scalar posterior summaries: {IPL_SCALAR_POSTERIOR_CSV}")
print(f"Direct IPL condition Gamma/Omega rows: {len(ipl_condition_model_df)}")
print(f"Direct IPL smooth Gamma/Omega rows: {len(ipl_smooth_model_df)}")

condition_summary_df = summarize_group_values(
    big_condition_df,
    ["ABL", "ILD"],
    ["condition_gamma", "condition_omega"],
    "condition",
)
print("Condition counts at |ILD|=16:")
print(
    condition_summary_df[condition_summary_df["ILD"].abs() == 16][
        ["ABL", "ILD", "n_condition_gamma", "n_condition_omega"]
    ].to_string(index=False)
)

ipl_method_summary_df = summarize_group_values(
    ipl_smooth_model_df,
    ["method_key", "ABL", "ILD"],
    ["model_gamma", "model_omega"],
    "model",
)
ipl_metrics_df = compute_gamma_omega_metrics(big_condition_df, ipl_condition_model_df)

method_summary_df = ipl_method_summary_df
metrics_df = ipl_metrics_df
method_summary_df.to_csv(MODEL_SUMMARY_CSV, index=False)
metrics_df.to_csv(METRICS_CSV, index=False)
print(f"Saved direct IPL Gamma/Omega model summary: {MODEL_SUMMARY_CSV}")
print(f"Saved direct IPL Gamma/Omega metrics: {METRICS_CSV}")
print(metrics_df[metrics_df["ABL"].astype(str) == "all"].to_string(index=False))

raw_df = load_raw_data()
animal_outputs = Parallel(n_jobs=N_JOBS, prefer=JOBLIB_PREFER)(
    delayed(process_animal_ipl)(pair, raw_df, ipl_params) for pair in animal_keys
)

psy_by_method = {}
slopes_by_method = {}
quantile_by_method = {}

psy_by_method[METHOD_KEY] = aggregate_psychometric(animal_outputs, animal_keys)
slopes_by_method[METHOD_KEY] = aggregate_slopes(psy_by_method[METHOD_KEY])
quantile_by_method[METHOD_KEY] = aggregate_quantiles(animal_outputs, animal_keys)

for method in METHODS:
    method_key = method["key"]
    sd_rows = np.array([batch == "SD" for batch, _animal in animal_keys])
    high_cols = np.abs(ILD_ARR) > 8
    sd_high_ild_model_count = 0
    for ABL in ABLS:
        sd_high_ild_model_count += int(
            np.sum(np.isfinite(psy_by_method[method_key]["theory_agg"][ABL][np.ix_(sd_rows, high_cols)]))
        )
    print(f"{method_key}: SD model psychometric entries at |ILD|>8 = {sd_high_ild_model_count}")
    if sd_high_ild_model_count != 0:
        raise RuntimeError(f"{method_key}: SD model psychometric/slope grid should not include |ILD|>8.")

    counts_at_16 = {
        ABL: len(quantile_by_method[method_key]["continuous_plot_data_sd_flat"][ABL][float(16.0)]["theoretical"])
        for ABL in ABLS
    }
    print(f"{method_key}: continuous SD-flat quantile animal counts at |ILD|=16: {counts_at_16}")
    if set(counts_at_16.values()) != {EXPECTED_N_ANIMALS}:
        raise RuntimeError(f"{method_key}: expected 30 SD-flat continuous model entries at |ILD|=16.")

fig = make_figure(condition_summary_df, method_summary_df, metrics_df, psy_by_method, slopes_by_method, quantile_by_method)

payload = {
    "methods": METHODS,
    "animal_keys": animal_keys,
    "base_3x5_pkl": str(BASE_3X5_PKL),
    "ipl_svi_root": str(IPL_SVI_ROOT),
    "big_condition_rows": big_condition_df.to_dict("records"),
    "big_scalar_rows": big_scalar_df.to_dict("records"),
    "ipl_param_rows": ipl_fit_df.to_dict("records"),
    "ipl_scalar_posterior_rows": ipl_scalar_summary_df.to_dict("records"),
    "method_param_rows": ipl_fit_df.to_dict("records"),
    "gamma_omega_metrics": metrics_df.to_dict("records"),
    "gamma_omega_model_summary": method_summary_df.to_dict("records"),
    "psy_by_method": psy_by_method,
    "slopes_by_method": slopes_by_method,
    "quantile_by_method": quantile_by_method,
    "output_png": str(OUTPUT_PNG),
    "config": {
        "N_THEORY": N_THEORY,
        "N_JOBS": N_JOBS,
        "RNG_SEED": RNG_SEED,
        "QUANTILES_TO_PLOT": QUANTILES_TO_PLOT,
        "PAPER_QUANTILES_TO_PLOT": PAPER_QUANTILES_TO_PLOT,
        "CONTINUOUS_ILD_STEP": CONTINUOUS_ILD_STEP,
        "sd_psychometric_model_abs_ild_max": 8,
        "continuous_sd_flat_delay_policy": "SD animals hold signed-branch t_E_aff flat after |ILD|=8; other animals use signed-branch interpolation",
        "ipl_params": ["rate_lambda", "T_0", "theta_E", "w", "del_go", "condition-wise t_E_aff"],
        "ipl_param_source": "direct patience12 50k vanilla/IPL SVI posterior means",
        "ipl_uses_big_svi_w_delgo_teaff": False,
        "gamma_omega_target_source": "patience12 92-param Gamma/Omega/delay SVI posterior means",
    },
}
with OUTPUT_PKL.open("wb") as handle:
    pickle.dump(payload, handle)

print(f"Saved figure: {OUTPUT_PNG}")
print(f"Saved data: {OUTPUT_PKL}")
