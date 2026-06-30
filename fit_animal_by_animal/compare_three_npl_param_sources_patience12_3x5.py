# %%
"""
Compare three sources of per-animal NPL+alpha parameters.

Rows:
1. NPL params from per-animal MSE fit to Gamma+Omega from the 92-param big SVI.
2. NPL params from per-animal MSE fit to Omega only from the 92-param big SVI.
3. NPL params, w, del_go, and t_E_aff from the direct 37-param NPL SVI.

Columns:
1. Gamma vs ILD against the 92-param condition means.
2. Omega vs ILD against the 92-param condition means.
3. Psychometric data vs model.
4. Psychometric slope data vs model.
5. Paper RT quantiles, using continuous theory curves and SD flat delay beyond |ILD|=8.
"""

# %%
# =============================================================================
# Editable parameters
# =============================================================================
from collections import defaultdict
from pathlib import Path
import os
import pickle
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
FIT_EACH_CONDN_DIR = REPO_DIR / "fit_each_condn"
if str(FIT_EACH_CONDN_DIR) not in sys.path:
    sys.path.insert(0, str(FIT_EACH_CONDN_DIR))

RAW_BATCH_DIR = REPO_DIR / "raw_data" / "batch_csvs"
ABORT_PARAMS_DIR = REPO_DIR / "aborts_ipl_npl_time_fit_results"
BIG_SVI_ROOT = (
    FIT_EACH_CONDN_DIR
    / "svi_big_gamma_omega_delay_patience12_restore_best_all_animals_outputs"
)
MSE_VARIANT_ROOT = BIG_SVI_ROOT / "mse_alpha_model_comparison" / "objective_variants"
NPL_SVI_ROOT = (
    SCRIPT_DIR / "numpyro_svi_npl_alpha_condition_delay_patience12_restore_best_outputs"
)
NPL_SVI_LABEL = "main_fullrank"

OUTPUT_DIR = NPL_SVI_ROOT / "three_npl_param_source_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PNG = OUTPUT_DIR / "three_npl_param_sources_patience12_3x5.png"
OUTPUT_PKL = OUTPUT_DIR / "three_npl_param_sources_patience12_3x5.pkl"
GAMMA_OMEGA_METRICS_CSV = OUTPUT_DIR / "three_npl_param_sources_gamma_omega_metrics.csv"
GAMMA_OMEGA_MODEL_SUMMARY_CSV = OUTPUT_DIR / "three_npl_param_sources_gamma_omega_model_summary.csv"
METHOD_PARAM_CSV = OUTPUT_DIR / "three_npl_param_sources_params_by_animal.csv"

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
CONTINUOUS_ILD_STEP = float(os.environ.get("THREE_NPL_COMPARE_CONTINUOUS_ILD_STEP", "0.1"))
CONTINUOUS_ABS_ILD = np.round(np.arange(1.0, 16.0 + CONTINUOUS_ILD_STEP / 2, CONTINUOUS_ILD_STEP), 1)

K_MAX = 10
T_PTS = np.arange(-2, 2, 0.001)
N_THEORY = int(os.environ.get("THREE_NPL_COMPARE_N_THEORY", "1000"))
N_JOBS = int(os.environ.get("THREE_NPL_COMPARE_N_JOBS", str(max(1, min(8, (os.cpu_count() or 2) - 1)))))
JOBLIB_PREFER = os.environ.get("THREE_NPL_COMPARE_JOBLIB_PREFER", "processes").strip().lower()
if JOBLIB_PREFER not in {"processes", "threads"}:
    raise ValueError("THREE_NPL_COMPARE_JOBLIB_PREFER must be 'processes' or 'threads'.")
RNG_SEED = int(os.environ.get("THREE_NPL_COMPARE_RNG_SEED", "137"))

BATCH_T_TRUNC = {"LED34_even": 0.15}
DEFAULT_T_TRUNC = 0.3
P_0 = 20e-6

METHODS = [
    {
        "key": "mse_gamma_omega",
        "label": "MSE fit Gamma + Omega",
        "short_label": "Gamma+Omega MSE",
        "mse_csv": MSE_VARIANT_ROOT / "gamma_omega" / "per_animal_mse_gamma_omega_alpha_params.csv",
        "rt_source": "big_svi",
    },
    {
        "key": "mse_omega_only",
        "label": "MSE fit Omega only",
        "short_label": "Omega-only MSE",
        "mse_csv": MSE_VARIANT_ROOT / "omega_only" / "per_animal_mse_gamma_omega_alpha_params.csv",
        "rt_source": "big_svi",
    },
    {
        "key": "direct_37_param_svi",
        "label": "Direct 37-param NPL SVI",
        "short_label": "Direct 37-param SVI",
        "mse_csv": None,
        "rt_source": "npl_svi",
    },
]

ABL_COLORS = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}


# %%
# =============================================================================
# Imports and helpers
# =============================================================================
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import scipy.special as scipy_special
from joblib import Parallel, delayed
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit
from scipy.stats import sem as scipy_sem

from animal_wise_plotting_utils import calculate_theoretical_curves
from gamma_omega_alpha_utils import gamma_omega_alpha_model
from time_vary_norm_alpha_utils import rho_A_t_fn
from time_vary_norm_utils import M, phi


plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]


def _create_innermost_dict():
    return {"empirical": [], "theoretical": []}


def _create_inner_defaultdict():
    return defaultdict(_create_innermost_dict)


def sort_pair(pair):
    batch_name, animal = pair
    return (BATCH_ORDER.get(batch_name, 999), batch_name, int(animal))


def parse_animal_dir_name(path):
    batch_name, animal_text = path.name.rsplit("_", 1)
    return batch_name, int(animal_text)


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


# %%
# =============================================================================
# Alpha-normalized RT model functions
# =============================================================================
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


def CDF_E_alpha_vec(t, bound, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, rate_norm_l, alpha):
    t_original = np.asarray(t, dtype=float)
    bound = np.asarray(bound)
    v, omega = gamma_omega_alpha_vec(ABL, ILD, rate_lambda, T_0, theta_E, Z_E, rate_norm_l, alpha)

    w = 0.5 + (Z_E / (2.0 * theta_E))
    w = np.asarray(w, dtype=float)
    a = 2
    v = np.where(bound == 1, -v, v)
    w = np.where(bound == 1, 1 - w, w)

    t_eff = omega * t_original
    shape = np.broadcast(t_eff, v, w).shape
    out = np.zeros(shape, dtype=float)
    valid = np.broadcast_to(t_original, shape) > 0
    safe_t = np.where(valid, np.broadcast_to(t_eff, shape), 1e-12)

    v_full = np.broadcast_to(v, shape)
    w_full = np.broadcast_to(w, shape)
    exponent_arg = -v_full * a * w_full - ((v_full**2) * safe_t / 2)
    result = np.exp(exponent_arg)

    k_arr = np.arange(K_MAX + 1)
    t_b = safe_t[..., None]
    v_b = v_full[..., None]
    w_b = w_full[..., None]
    k_b = k_arr.reshape((1,) * len(shape) + (K_MAX + 1,))

    r_k = np.where(k_b % 2 == 0, k_b * a + a * w_b, k_b * a + a * (1 - w_b))
    sqrt_t = np.sqrt(t_b)
    term1 = phi(r_k / sqrt_t)
    term2 = M((r_k - v_b * t_b) / sqrt_t) + M((r_k + v_b * t_b) / sqrt_t)
    summation = np.sum(((-1) ** k_b) * term1 * term2, axis=-1)

    out[valid] = (result * summation)[valid]
    return out


def gamma_omega_alpha_vec(ABL, ILD, rate_lambda, T_0, theta_E, Z_E, rate_norm_l, alpha):
    _ = Z_E
    gamma, omega = gamma_omega_alpha_model(
        ABL,
        ILD,
        rate_lambda,
        rate_norm_l,
        alpha,
        theta_E,
        T_0,
        P_0,
    )
    return gamma, omega


def rho_E_alpha_vec(t, bound, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, rate_norm_l, alpha):
    t_original = np.asarray(t, dtype=float)
    bound = np.asarray(bound)
    v, omega = gamma_omega_alpha_vec(ABL, ILD, rate_lambda, T_0, theta_E, Z_E, rate_norm_l, alpha)

    w = 0.5 + (Z_E / (2.0 * theta_E))
    w = np.asarray(w, dtype=float)
    a = 2
    v = np.where(bound == 1, -v, v)
    w = np.where(bound == 1, 1 - w, w)

    t_eff = omega * t_original
    shape = np.broadcast(t_eff, v, w).shape
    out = np.zeros(shape, dtype=float)
    valid = np.broadcast_to(t_original, shape) > 0
    safe_t = np.where(valid, np.broadcast_to(t_eff, shape), 1e-12)

    v_full = np.broadcast_to(v, shape)
    w_full = np.broadcast_to(w, shape)
    non_sum_term = (
        (1 / a**2)
        * (a**3 / np.sqrt(2 * np.pi * safe_t**3))
        * np.exp(-v_full * a * w_full - (v_full**2 * safe_t) / 2)
    )

    k_half = int(K_MAX / 2)
    k_vals = np.linspace(-k_half, k_half, 2 * k_half + 1)
    t_b = safe_t[..., None]
    w_b = w_full[..., None]
    k_b = k_vals.reshape((1,) * len(shape) + (2 * k_half + 1,))

    sum_w_term = w_b + 2 * k_b
    sum_exp_term = np.exp(-(a**2 * (w_b + 2 * k_b) ** 2) / (2 * t_b))
    sum_result = np.sum(sum_w_term * sum_exp_term, axis=-1)

    density = non_sum_term * sum_result
    density = np.where(density <= 0, 1e-16, density)
    out[valid] = (density * np.broadcast_to(omega, shape))[valid]
    return out


def up_or_down_alpha_pa_ca_vec(t, bound, P_A, C_A, ABL, ILD, tied_params, Z_E, t_E_aff):
    t = np.asarray(t, dtype=float)
    t1 = np.maximum(t - t_E_aff, 1e-6)
    t2 = np.maximum(t - t_E_aff + tied_params["del_go"], 1e-6)

    p_ea_hits_either_bound = (
        CDF_E_alpha_vec(
            t - t_E_aff + tied_params["del_go"],
            1,
            ABL,
            ILD,
            tied_params["rate_lambda"],
            tied_params["T_0"],
            tied_params["theta_E"],
            Z_E,
            tied_params["rate_norm_l"],
            tied_params["alpha"],
        )
        + CDF_E_alpha_vec(
            t - t_E_aff + tied_params["del_go"],
            -1,
            ABL,
            ILD,
            tied_params["rate_lambda"],
            tied_params["T_0"],
            tied_params["theta_E"],
            Z_E,
            tied_params["rate_norm_l"],
            tied_params["alpha"],
        )
    )
    random_readout_if_ea_survives = 0.5 * (1 - p_ea_hits_either_bound)

    p_e_plus_cum = (
        CDF_E_alpha_vec(
            t2,
            bound,
            ABL,
            ILD,
            tied_params["rate_lambda"],
            tied_params["T_0"],
            tied_params["theta_E"],
            Z_E,
            tied_params["rate_norm_l"],
            tied_params["alpha"],
        )
        - CDF_E_alpha_vec(
            t1,
            bound,
            ABL,
            ILD,
            tied_params["rate_lambda"],
            tied_params["T_0"],
            tied_params["theta_E"],
            Z_E,
            tied_params["rate_norm_l"],
            tied_params["alpha"],
        )
    )
    p_e_plus = rho_E_alpha_vec(
        t - t_E_aff,
        bound,
        ABL,
        ILD,
        tied_params["rate_lambda"],
        tied_params["T_0"],
        tied_params["theta_E"],
        Z_E,
        tied_params["rate_norm_l"],
        tied_params["alpha"],
    )

    return P_A * (random_readout_if_ea_survives + p_e_plus_cum) + p_e_plus * (1 - C_A)


def cum_pro_and_reactive_alpha_vec(t, c_A_trunc_time, abort_params, t_stim, ABL, ILD, tied_params, Z_E, t_E_aff):
    t = np.asarray(t, dtype=float)
    c_A = cum_A_t_vec(t - abort_params["t_A_aff"], abort_params["V_A"], abort_params["theta_A"])
    if c_A_trunc_time is not None:
        trunc_denom = 1 - cum_A_t_vec(
            np.array([c_A_trunc_time - abort_params["t_A_aff"]]),
            abort_params["V_A"],
            abort_params["theta_A"],
        )[0]
        c_A = np.where(t < c_A_trunc_time, 0, c_A / trunc_denom)

    c_E = (
        CDF_E_alpha_vec(
            t - t_stim - t_E_aff,
            1,
            ABL,
            ILD,
            tied_params["rate_lambda"],
            tied_params["T_0"],
            tied_params["theta_E"],
            Z_E,
            tied_params["rate_norm_l"],
            tied_params["alpha"],
        )
        + CDF_E_alpha_vec(
            t - t_stim - t_E_aff,
            -1,
            ABL,
            ILD,
            tied_params["rate_lambda"],
            tied_params["T_0"],
            tied_params["theta_E"],
            Z_E,
            tied_params["rate_norm_l"],
            tied_params["alpha"],
        )
    )
    return c_A + c_E - c_A * c_E


# %%
# =============================================================================
# Parameter loading
# =============================================================================
def load_big_svi_params():
    condition_paths = sorted(BIG_SVI_ROOT.glob("*/*_big_gamma_omega_delay_condition_summary.csv"))
    if len(condition_paths) != EXPECTED_N_ANIMALS:
        raise RuntimeError(f"Expected {EXPECTED_N_ANIMALS} big-SVI condition summaries, found {len(condition_paths)}")

    condition_rows = []
    scalar_rows = []
    params = {}

    for condition_path in condition_paths:
        condition_df = pd.read_csv(condition_path)
        required_condition_cols = [
            "batch_name",
            "animal",
            "ABL",
            "ILD",
            "gamma_mean",
            "omega_mean",
            "t_E_aff_mean",
        ]
        missing_cols = [col for col in required_condition_cols if col not in condition_df.columns]
        if missing_cols:
            raise KeyError(f"{condition_path} missing columns: {missing_cols}")

        batch_name = str(condition_df["batch_name"].iloc[0])
        animal = int(condition_df["animal"].iloc[0])
        posterior_path = condition_path.with_name(
            condition_path.name.replace("_condition_summary.csv", "_posterior_summary.csv")
        )
        if not posterior_path.exists():
            raise FileNotFoundError(posterior_path)

        posterior_df = pd.read_csv(posterior_path)
        scalar_df = posterior_df[posterior_df["parameter"].isin(["w", "del_go"])].copy()
        if len(scalar_df) != 2:
            raise RuntimeError(f"{posterior_path} should contain exactly w and del_go scalar rows.")
        scalar_map = {row.parameter: float(row.mean) for row in scalar_df.itertuples(index=False)}

        delay_map = {}
        for row in condition_df.itertuples(index=False):
            delay_map[(int(row.ABL), float(row.ILD))] = float(row.t_E_aff_mean)
            condition_rows.append(
                {
                    "batch_name": batch_name,
                    "animal": animal,
                    "ABL": int(row.ABL),
                    "ILD": float(row.ILD),
                    "condition_gamma": float(row.gamma_mean),
                    "condition_omega": float(row.omega_mean),
                    "t_E_aff": float(row.t_E_aff_mean),
                }
            )

        for param_name, value in scalar_map.items():
            scalar_rows.append({"batch_name": batch_name, "animal": animal, "parameter": param_name, "mean": value})

        params[(batch_name, animal)] = {
            "w": scalar_map["w"],
            "del_go": scalar_map["del_go"],
            "delay_by_condition": delay_map,
        }

    condition_df = pd.DataFrame(condition_rows).sort_values(["batch_name", "animal", "ABL", "ILD"]).reset_index(drop=True)
    scalar_df = pd.DataFrame(scalar_rows).sort_values(["batch_name", "animal", "parameter"]).reset_index(drop=True)

    if len(condition_df) != EXPECTED_N_CONDITION_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_N_CONDITION_ROWS} big-SVI condition rows, found {len(condition_df)}")
    if condition_df[["batch_name", "animal"]].drop_duplicates().shape[0] != EXPECTED_N_ANIMALS:
        raise RuntimeError("Big-SVI summaries do not contain 30 animals.")
    if condition_df.duplicated(["batch_name", "animal", "ABL", "ILD"]).any():
        raise RuntimeError("Big-SVI condition rows contain duplicates.")
    return condition_df, scalar_df, params


def load_mse_npl_params(mse_csv, method_key, method_label):
    mse_df = pd.read_csv(mse_csv)
    required_cols = ["batch_name", "animal", "success", "rate_lambda", "T_0", "theta_E", "rate_norm_l", "alpha"]
    missing_cols = [col for col in required_cols if col not in mse_df.columns]
    if missing_cols:
        raise KeyError(f"{mse_csv} missing columns: {missing_cols}")

    mse_df = mse_df[mse_df["success"] == True].copy()
    if len(mse_df) != EXPECTED_N_ANIMALS:
        raise RuntimeError(f"Expected {EXPECTED_N_ANIMALS} successful MSE rows in {mse_csv}, found {len(mse_df)}")

    params = {}
    rows = []
    for row in mse_df.itertuples(index=False):
        pair = (str(row.batch_name), int(row.animal))
        params[pair] = {
            "rate_lambda": float(row.rate_lambda),
            "T_0": float(row.T_0),
            "theta_E": float(row.theta_E),
            "rate_norm_l": float(row.rate_norm_l),
            "alpha": float(row.alpha),
        }
        rows.append(
            {
                "method_key": method_key,
                "method_label": method_label,
                "batch_name": pair[0],
                "animal": pair[1],
                "rate_lambda": params[pair]["rate_lambda"],
                "T_0": params[pair]["T_0"],
                "theta_E": params[pair]["theta_E"],
                "rate_norm_l": params[pair]["rate_norm_l"],
                "alpha": params[pair]["alpha"],
                "w": np.nan,
                "del_go": np.nan,
            }
        )
    return mse_df, params, rows


def load_direct_npl_svi_params():
    summary_paths = sorted(NPL_SVI_ROOT.glob(f"*/{NPL_SVI_LABEL}_posterior_summary.csv"))
    if len(summary_paths) != EXPECTED_N_ANIMALS:
        raise RuntimeError(f"Expected {EXPECTED_N_ANIMALS} NPL SVI summaries, found {len(summary_paths)}")

    params = {}
    scalar_rows = []
    condition_rows = []
    param_rows = []

    scalar_names = ["rate_lambda", "T_0", "theta_E", "w", "del_go", "rate_norm_l", "alpha"]
    for summary_path in summary_paths:
        batch_name, animal = parse_animal_dir_name(summary_path.parent)
        summary_df = pd.read_csv(summary_path)

        scalar_map = {}
        for param_name in scalar_names:
            rows = summary_df[summary_df["parameter"] == param_name]
            if len(rows) != 1:
                raise RuntimeError(f"{summary_path} has {len(rows)} rows for {param_name}")
            scalar_map[param_name] = float(rows.iloc[0]["mean"])
            scalar_rows.append(
                {
                    "batch_name": batch_name,
                    "animal": animal,
                    "parameter": param_name,
                    "mean": scalar_map[param_name],
                }
            )

        delay_df = summary_df[
            summary_df["parameter"].astype(str).str.startswith("t_E_aff")
            & summary_df["ABL"].notna()
            & summary_df["ILD"].notna()
        ].copy()
        if delay_df.empty:
            raise RuntimeError(f"{summary_path} has no condition t_E_aff rows")

        delay_map = {}
        for row in delay_df.itertuples(index=False):
            delay_map[(int(row.ABL), float(row.ILD))] = float(row.mean)
            condition_rows.append(
                {
                    "batch_name": batch_name,
                    "animal": animal,
                    "ABL": int(row.ABL),
                    "ILD": float(row.ILD),
                    "t_E_aff": float(row.mean),
                }
            )

        pair = (batch_name, animal)
        params[pair] = {
            "rate_lambda": scalar_map["rate_lambda"],
            "T_0": scalar_map["T_0"],
            "theta_E": scalar_map["theta_E"],
            "w": scalar_map["w"],
            "del_go": scalar_map["del_go"],
            "rate_norm_l": scalar_map["rate_norm_l"],
            "alpha": scalar_map["alpha"],
            "delay_by_condition": delay_map,
        }
        param_rows.append(
            {
                "method_key": "direct_37_param_svi",
                "method_label": "Direct 37-param NPL SVI",
                "batch_name": batch_name,
                "animal": animal,
                "rate_lambda": scalar_map["rate_lambda"],
                "T_0": scalar_map["T_0"],
                "theta_E": scalar_map["theta_E"],
                "rate_norm_l": scalar_map["rate_norm_l"],
                "alpha": scalar_map["alpha"],
                "w": scalar_map["w"],
                "del_go": scalar_map["del_go"],
            }
        )

    condition_df = pd.DataFrame(condition_rows).sort_values(["batch_name", "animal", "ABL", "ILD"]).reset_index(drop=True)
    scalar_df = pd.DataFrame(scalar_rows).sort_values(["batch_name", "animal", "parameter"]).reset_index(drop=True)
    if len(condition_df) != EXPECTED_N_CONDITION_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_N_CONDITION_ROWS} NPL SVI delay rows, found {len(condition_df)}")
    if condition_df.duplicated(["batch_name", "animal", "ABL", "ILD"]).any():
        raise RuntimeError("NPL SVI delay rows contain duplicates.")
    return condition_df, scalar_df, params, param_rows


def attach_big_rt_params(npl_params, big_svi_params):
    out = {}
    for pair, params in npl_params.items():
        if pair not in big_svi_params:
            continue
        combined = dict(params)
        combined["w"] = big_svi_params[pair]["w"]
        combined["del_go"] = big_svi_params[pair]["del_go"]
        combined["delay_by_condition"] = big_svi_params[pair]["delay_by_condition"]
        out[pair] = combined
    return out


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


def discover_matched_animals(raw_df, method_param_maps):
    valid_df = raw_df[raw_df["success"].isin([1, -1])].copy()
    raw_pairs = {
        (str(batch), int(animal))
        for batch, animal in valid_df[["batch_name", "animal"]].drop_duplicates().itertuples(index=False)
    }
    common_pairs = set(raw_pairs)
    for params in method_param_maps.values():
        common_pairs &= set(params)
    common_pairs = sorted(common_pairs, key=sort_pair)
    if len(common_pairs) != EXPECTED_N_ANIMALS:
        raise RuntimeError(f"Expected {EXPECTED_N_ANIMALS} matched animals, found {len(common_pairs)}")
    return common_pairs


# %%
# =============================================================================
# Theory and empirical calculations
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


def get_theoretical_rtd_components(P_A_mean, C_A_mean, t_stim_samples, abort_params, tied_params, ABL, ILD, batch_name, t_E_aff):
    T_trunc = BATCH_T_TRUNC.get(batch_name, DEFAULT_T_TRUNC)
    Z_E = (tied_params["w"] - 0.5) * 2 * tied_params["theta_E"]

    trunc_fac_samples = (
        cum_pro_and_reactive_alpha_vec(
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
        - cum_pro_and_reactive_alpha_vec(
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

    up_mean = up_or_down_alpha_pa_ca_vec(T_PTS, 1, P_A_mean, C_A_mean, ABL, ILD, tied_params, Z_E, t_E_aff)
    down_mean = up_or_down_alpha_pa_ca_vec(T_PTS, -1, P_A_mean, C_A_mean, ABL, ILD, tied_params, Z_E, t_E_aff)

    mask_0_1 = (T_PTS >= 0) & (T_PTS <= 1)
    t_pts_0_1 = T_PTS[mask_0_1]
    up = np.maximum(up_mean[mask_0_1] / trunc_factor, 0)
    down = np.maximum(down_mean[mask_0_1] / trunc_factor, 0)
    return t_pts_0_1, up, down


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


def process_single_method(batch_name, animal, method_key, tied_params, empirical_psy, empirical_quantiles, P_A_mean, C_A_mean, t_stim_samples, abort_params):
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
                    t_pts, up, down = get_theoretical_rtd_components(
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
                    print(f"  {method_key} theory failed {batch_name}/{animal} ABL={ABL} ILD={ILD}: {exc}")

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
                        t_pts, up, down = get_theoretical_rtd_components(
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
                if flat_delay is None:
                    continue
                if (
                    quantile_for_standard_delay is not None
                    and delay is not None
                    and np.isclose(flat_delay, delay)
                ):
                    sign_quantiles_sd_flat.append(quantile_for_standard_delay)
                    continue
                try:
                    t_pts, up, down = get_theoretical_rtd_components(
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
                    continue
            if sign_quantiles:
                continuous_quantiles[(ABL, float(abs_ild))] = np.nanmean(sign_quantiles, axis=0)
            if sign_quantiles_sd_flat:
                continuous_quantiles_sd_flat[(ABL, float(abs_ild))] = np.nanmean(sign_quantiles_sd_flat, axis=0)

    return {
        "pair": (batch_name, animal),
        "method_key": method_key,
        "empirical_psy": empirical_psy,
        "theory_psy": theory_psy,
        "discrete_quantiles": discrete_quantiles,
        "continuous_quantiles": continuous_quantiles,
        "continuous_quantiles_sd_flat": continuous_quantiles_sd_flat,
    }


def process_animal_all_methods(pair, raw_df, method_param_maps):
    batch_name, animal = pair
    print(f"Processing {batch_name}/{animal}")

    batch_df = raw_df[raw_df["batch_name"].astype(str) == str(batch_name)].copy()
    abort_params = load_abort_params(batch_name, animal)
    P_A_mean, C_A_mean, t_stim_samples = get_p_a_c_a(batch_df, animal, abort_params)

    empirical_psy = {ABL: empirical_psychometric(batch_df, animal, ABL) for ABL in ABLS}
    empirical_quantiles = {
        (ABL, float(ILD)): empirical_rt_quantiles(batch_df, animal, ABL, ILD)
        for ABL in ABLS
        for ILD in ILD_ARR
    }

    method_results = {}
    for method in METHODS:
        method_key = method["key"]
        tied_params = method_param_maps[method_key][pair]
        method_results[method_key] = process_single_method(
            batch_name,
            animal,
            method_key,
            tied_params,
            empirical_psy,
            empirical_quantiles,
            P_A_mean,
            C_A_mean,
            t_stim_samples,
            abort_params,
        )
    return {"pair": pair, "method_results": method_results}


# %%
# =============================================================================
# Aggregation
# =============================================================================
def model_curves_for_params(method_key, params_by_animal, ild_grid):
    rows = []
    for (batch_name, animal), params in sorted(params_by_animal.items(), key=lambda item: sort_pair(item[0])):
        for abl in ABLS:
            gamma, omega = gamma_omega_alpha_model(
                abl,
                np.asarray(ild_grid, dtype=float),
                params["rate_lambda"],
                params["rate_norm_l"],
                params["alpha"],
                params["theta_E"],
                params["T_0"],
                P_0,
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


def model_values_for_condition_rows(method_key, params_by_animal, condition_df):
    rows = []
    for (batch_name, animal), animal_cond in condition_df.groupby(["batch_name", "animal"], sort=True):
        pair = (str(batch_name), int(animal))
        params = params_by_animal[pair]
        gamma, omega = gamma_omega_alpha_model(
            animal_cond["ABL"].to_numpy(dtype=float),
            animal_cond["ILD"].to_numpy(dtype=float),
            params["rate_lambda"],
            params["rate_norm_l"],
            params["alpha"],
            params["theta_E"],
            params["T_0"],
            P_0,
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
        how="left",
        validate="one_to_many",
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
                    rows.append(
                        {
                            "method_key": method_key,
                            "parameter": param,
                            "ABL": abl,
                            "n_points": int(np.sum(finite)),
                            "rmse": float(np.sqrt(np.mean(diff**2))),
                            "mae": float(np.mean(np.abs(diff))),
                            "pearson_r": float(np.corrcoef(target[finite], pred[finite])[0, 1]),
                        }
                    )
                else:
                    rows.append(
                        {
                            "method_key": method_key,
                            "parameter": param,
                            "ABL": abl,
                            "n_points": int(np.sum(finite)),
                            "rmse": np.nan,
                            "mae": np.nan,
                            "pearson_r": np.nan,
                        }
                    )
    return pd.DataFrame(rows)


def aggregate_psychometric(results, animal_keys):
    empirical_agg = {ABL: np.full((len(animal_keys), len(ILD_ARR)), np.nan) for ABL in ABLS}
    theory_agg = {ABL: np.full((len(animal_keys), len(ILD_ARR)), np.nan) for ABL in ABLS}

    for animal_idx, result in enumerate(results):
        for ABL in ABLS:
            for ild_idx, ILD in enumerate(ILD_ARR):
                empirical_agg[ABL][animal_idx, ild_idx] = result["empirical_psy"][ABL].get(float(ILD), np.nan)
                theory_agg[ABL][animal_idx, ild_idx] = result["theory_psy"][ABL].get(float(ILD), np.nan)

    return {
        "empirical_agg": empirical_agg,
        "theory_agg": theory_agg,
        "ILD_arr": ILD_ARR,
        "animal_keys": animal_keys,
    }


def aggregate_slopes(psy_data):
    empirical_agg = psy_data["empirical_agg"]
    theory_agg = psy_data["theory_agg"]
    animal_keys = psy_data["animal_keys"]

    data_means = []
    model_means = []
    slopes_data = {}
    slopes_model = {}
    for animal_idx, pair in enumerate(animal_keys):
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

    data_means = np.asarray(data_means, dtype=float)
    model_means = np.asarray(model_means, dtype=float)
    sort_idx = np.argsort(data_means)
    return {
        "slopes_data": slopes_data,
        "slopes_model": slopes_model,
        "common_pairs_sorted": [animal_keys[idx] for idx in sort_idx],
        "data_means": data_means[sort_idx],
        "model_means": model_means[sort_idx],
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
                    emp_sign_values = np.asarray(emp_sign_values, dtype=float)
                    if np.any(np.isfinite(emp_sign_values)):
                        plot_data[ABL][abs_ild]["empirical"].append(np.nanmean(emp_sign_values, axis=0))
                if theory_sign_values:
                    theory_sign_values = np.asarray(theory_sign_values, dtype=float)
                    if not np.any(np.isfinite(theory_sign_values)):
                        continue
                    theory_mean = np.nanmean(theory_sign_values, axis=0)
                    if np.any(np.isfinite(theory_mean)):
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
        "QUANTILES_TO_PLOT": QUANTILES_TO_PLOT,
        "abs_ild_sorted": ABS_ILD_SORTED,
        "continuous_abs_ild": CONTINUOUS_ABS_ILD,
        "ABL_arr": ABLS,
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
    fig, axes = plt.subplots(3, 5, figsize=(22, 12.5))
    column_titles = [
        "Gamma vs ILD",
        "Omega vs ILD",
        "Psychometric",
        "Psychometric slope",
        "RT quantiles",
    ]
    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=12, pad=24)

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
            column_title="Gamma vs ILD" if row_idx == 0 else None,
        )
        plot_gamma_omega(
            axes[row_idx, 1],
            "omega",
            method_key,
            condition_summary_df,
            method_summary_df,
            metrics_df,
            show_xlabel,
            column_title="Omega vs ILD" if row_idx == 0 else None,
        )
        plot_psychometric(axes[row_idx, 2], psy_by_method[method_key], show_xlabel)
        plot_slopes(
            axes[row_idx, 3],
            slopes_by_method[method_key],
            show_xlabel,
            column_title="Psychometric slope" if row_idx == 0 else None,
        )
        plot_quantiles(axes[row_idx, 4], quantile_by_method[method_key], show_xlabel)

        axes[row_idx, 0].text(
            -0.34,
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
        Line2D([0], [0], color="black", lw=1.4, label="NPL functional mean"),
        Line2D([0], [0], marker="x", color="black", linestyle="none", label="psychometric model points"),
        Line2D([0], [0], marker="o", color="black", linestyle="none", label="RT quantile data"),
        Line2D([0], [0], color="tab:red", lw=1.2, label="RT quantile model"),
    ]
    fig.legend(
        handles=abl_handles + source_handles,
        loc="upper center",
        ncol=8,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
        fontsize=9,
    )
    fig.suptitle(
        "Three NPL parameter sources tested against Gamma/Omega, psychometric, slopes, and paper RT quantiles",
        fontsize=15,
        y=1.04,
    )
    fig.tight_layout(rect=[0.035, 0.02, 1.0, 0.965], w_pad=1.3, h_pad=1.1)
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    return fig


# %%
# =============================================================================
# Main analysis
# =============================================================================
print(f"Big SVI root: {BIG_SVI_ROOT}")
print(f"NPL SVI root: {NPL_SVI_ROOT}")
print(f"Output folder: {OUTPUT_DIR}")
print(f"N_THEORY={N_THEORY}, N_JOBS={N_JOBS}, joblib prefer={JOBLIB_PREFER}")

big_condition_df, big_scalar_df, big_svi_params = load_big_svi_params()
npl_condition_df, npl_scalar_df, direct_npl_params, direct_param_rows = load_direct_npl_svi_params()

mse_param_rows = []
method_base_npl_params = {}
for method in METHODS:
    if method["mse_csv"] is None:
        method_base_npl_params[method["key"]] = direct_npl_params
        mse_param_rows.extend(direct_param_rows)
    else:
        _mse_df, curr_params, curr_rows = load_mse_npl_params(method["mse_csv"], method["key"], method["label"])
        method_base_npl_params[method["key"]] = curr_params
        mse_param_rows.extend(curr_rows)

method_param_maps = {}
for method in METHODS:
    method_key = method["key"]
    if method["rt_source"] == "big_svi":
        method_param_maps[method_key] = attach_big_rt_params(method_base_npl_params[method_key], big_svi_params)
    elif method["rt_source"] == "npl_svi":
        method_param_maps[method_key] = direct_npl_params
    else:
        raise ValueError(method["rt_source"])

param_df = pd.DataFrame(mse_param_rows).sort_values(["method_key", "batch_name", "animal"]).reset_index(drop=True)
param_df.to_csv(METHOD_PARAM_CSV, index=False)
print(f"Saved method params: {METHOD_PARAM_CSV}")

raw_df = load_raw_data()
animal_keys = discover_matched_animals(raw_df, method_param_maps)
print(f"Matched animals: {len(animal_keys)}")

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

smooth_frames = []
condition_model_frames = []
for method in METHODS:
    method_key = method["key"]
    smooth_frames.append(model_curves_for_params(method_key, method_base_npl_params[method_key], SMOOTH_ILDS))
    condition_model_frames.append(model_values_for_condition_rows(method_key, method_base_npl_params[method_key], big_condition_df))

model_smooth_df = pd.concat(smooth_frames, ignore_index=True)
model_condition_df = pd.concat(condition_model_frames, ignore_index=True)
method_summary_df = summarize_group_values(
    model_smooth_df,
    ["method_key", "ABL", "ILD"],
    ["model_gamma", "model_omega"],
    "model",
)
method_summary_df.to_csv(GAMMA_OMEGA_MODEL_SUMMARY_CSV, index=False)

metrics_df = compute_gamma_omega_metrics(big_condition_df, model_condition_df)
metrics_df.to_csv(GAMMA_OMEGA_METRICS_CSV, index=False)
print(f"Saved Gamma/Omega model summary: {GAMMA_OMEGA_MODEL_SUMMARY_CSV}")
print(f"Saved Gamma/Omega metrics: {GAMMA_OMEGA_METRICS_CSV}")
print(metrics_df[metrics_df["ABL"].astype(str) == "all"].to_string(index=False))

animal_outputs = Parallel(n_jobs=N_JOBS, prefer=JOBLIB_PREFER)(
    delayed(process_animal_all_methods)(pair, raw_df, method_param_maps) for pair in animal_keys
)

results_by_method = {method["key"]: [] for method in METHODS}
for animal_output in animal_outputs:
    for method in METHODS:
        results_by_method[method["key"]].append(animal_output["method_results"][method["key"]])

psy_by_method = {}
slopes_by_method = {}
quantile_by_method = {}
for method in METHODS:
    method_key = method["key"]
    psy_by_method[method_key] = aggregate_psychometric(results_by_method[method_key], animal_keys)
    slopes_by_method[method_key] = aggregate_slopes(psy_by_method[method_key])
    quantile_by_method[method_key] = aggregate_quantiles(results_by_method[method_key], animal_keys)

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
    "big_svi_root": str(BIG_SVI_ROOT),
    "npl_svi_root": str(NPL_SVI_ROOT),
    "big_condition_rows": big_condition_df.to_dict("records"),
    "big_scalar_rows": big_scalar_df.to_dict("records"),
    "npl_condition_rows": npl_condition_df.to_dict("records"),
    "npl_scalar_rows": npl_scalar_df.to_dict("records"),
    "method_param_rows": param_df.to_dict("records"),
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
    },
}
with OUTPUT_PKL.open("wb") as handle:
    pickle.dump(payload, handle)

print(f"Saved figure: {OUTPUT_PNG}")
print(f"Saved data: {OUTPUT_PKL}")

plt.show()

# %%
