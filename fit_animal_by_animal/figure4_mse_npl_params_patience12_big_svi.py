# %%
"""
Figure 4-style diagnostics for MSE-fit NPL parameters.

The shared NPL+alpha parameters come from per-animal MSE fits to the
patience12 big Gamma/Omega/delay SVI condition means. The remaining RT model
parameters use the same patience12 big SVI fit: scalar w/del_go and
condition-wise t_E_aff.
"""

# %%
# =============================================================================
# Parameters
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
import numpy as np
import pandas as pd
import scipy.special as scipy_special
from joblib import Parallel, delayed
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit
from scipy.stats import sem

import figure_template as ft
from animal_wise_plotting_utils import calculate_theoretical_curves
from time_vary_norm_alpha_utils import rho_A_t_fn
from time_vary_norm_utils import M, phi


plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
RAW_BATCH_DIR = REPO_DIR / "raw_data" / "batch_csvs"
ABORT_PARAMS_DIR = REPO_DIR / "aborts_ipl_npl_time_fit_results"

BIG_SVI_ROOT = Path(
    os.environ.get(
        "MSE_FIG4_BIG_SVI_ROOT",
        str(
            REPO_DIR
            / "fit_each_condn"
            / "svi_big_gamma_omega_delay_patience12_restore_best_all_animals_outputs"
        ),
    )
).expanduser()
MSE_COMPARISON_DIR = BIG_SVI_ROOT / "mse_alpha_model_comparison"
MSE_OBJECTIVE = os.environ.get("MSE_FIG4_OBJECTIVE", "gamma_omega").strip().lower()
MSE_OBJECTIVE_LABEL = os.environ.get("MSE_FIG4_OBJECTIVE_LABEL", "fit Gamma + Omega")
MSE_PARAM_CSV = Path(
    os.environ.get(
        "MSE_FIG4_MSE_PARAM_CSV",
        str(MSE_COMPARISON_DIR / "per_animal_mse_gamma_omega_alpha_params.csv"),
    )
).expanduser()
OUTPUT_DIR = Path(
    os.environ.get(
        "MSE_FIG4_OUTPUT_DIR",
        str(MSE_PARAM_CSV.parent / "figure4_mse_params_diagnostics"),
    )
).expanduser()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PNG = OUTPUT_DIR / os.environ.get(
    "MSE_FIG4_FIG_BASENAME",
    "patience12_mse_npl_params_fig4_psychometric_slopes_quantiles.png",
)
OUTPUT_PKL = OUTPUT_DIR / os.environ.get(
    "MSE_FIG4_PKL_BASENAME",
    "patience12_mse_npl_params_fig4_psychometric_slopes_quantiles.pkl",
)

DESIRED_BATCHES = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]
BATCH_ORDER = {batch: idx for idx, batch in enumerate(DESIRED_BATCHES)}
ABL_arr = [20, 40, 60]
ILD_arr = np.array([-16.0, -8.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0, 16.0], dtype=float)
ABS_ILD_SORTED = sorted({float(abs(ild)) for ild in ILD_arr})

QUANTILES_TO_PLOT = np.round(np.arange(0.1, 1.0, 0.1), 1).tolist()
PAPER_QUANTILES_TO_PLOT = [0.1, 0.3, 0.5, 0.7, 0.9]
CONTINUOUS_ILD_STEP = float(os.environ.get("MSE_FIG4_CONTINUOUS_ILD_STEP", "0.1"))
CONTINUOUS_ABS_ILD = np.round(np.arange(1.0, 16.0 + CONTINUOUS_ILD_STEP / 2, CONTINUOUS_ILD_STEP), 1)

K_max = 10
T_PTS = np.arange(-2, 2, 0.001)
N_THEORY = int(os.environ.get("MSE_FIG4_N_THEORY", "1000"))
N_JOBS = int(os.environ.get("MSE_FIG4_N_JOBS", str(max(1, min(8, (os.cpu_count() or 2) - 1)))))
JOBLIB_PREFER = os.environ.get("MSE_FIG4_JOBLIB_PREFER", "processes").strip().lower()
if JOBLIB_PREFER not in {"processes", "threads"}:
    raise ValueError("MSE_FIG4_JOBLIB_PREFER must be 'processes' or 'threads'.")
RNG_SEED = int(os.environ.get("MSE_FIG4_RNG_SEED", "137"))

BATCH_T_TRUNC = {"LED34_even": 0.15}
DEFAULT_T_TRUNC = 0.3
EXPECTED_N_ANIMALS = 30
EXPECTED_N_CONDITION_ROWS = 864


# %%
# =============================================================================
# Small helpers
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
    curr_sem = float(sem(values, nan_policy="omit")) if n > 1 else np.nan
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


# %%
# =============================================================================
# Alpha-normalized model functions
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


def gamma_omega_alpha_vec(ABL, ILD, rate_lambda, T_0, theta_E, rate_norm_l, alpha):
    chi = 17.37
    ABL = np.asarray(ABL, dtype=float)
    ILD = np.asarray(ILD, dtype=float)

    abl_term = 10 ** (rate_lambda * (1 - rate_norm_l) * ABL / 20)
    ild_arg = rate_lambda * ILD / chi
    norm_ild_arg = rate_lambda * rate_norm_l * ILD / chi

    r_r = abl_term * np.exp(ild_arg) / (np.exp(norm_ild_arg) + alpha * np.exp(-norm_ild_arg))
    r_l = abl_term * np.exp(-ild_arg) / (np.exp(-norm_ild_arg) + alpha * np.exp(norm_ild_arg))

    r_sum = r_r + r_l
    gamma = theta_E * (r_r - r_l) / r_sum
    omega = r_sum / (T_0 * (theta_E**2))
    return gamma, omega


def CDF_E_alpha_vec(t, bound, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, rate_norm_l, alpha):
    t_original = np.asarray(t, dtype=float)
    bound = np.asarray(bound)
    v, omega = gamma_omega_alpha_vec(ABL, ILD, rate_lambda, T_0, theta_E, rate_norm_l, alpha)

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

    k_arr = np.arange(K_max + 1)
    t_b = safe_t[..., None]
    v_b = v_full[..., None]
    w_b = w_full[..., None]
    k_b = k_arr.reshape((1,) * len(shape) + (K_max + 1,))

    r_k = np.where(k_b % 2 == 0, k_b * a + a * w_b, k_b * a + a * (1 - w_b))
    sqrt_t = np.sqrt(t_b)
    term1 = phi(r_k / sqrt_t)
    term2 = M((r_k - v_b * t_b) / sqrt_t) + M((r_k + v_b * t_b) / sqrt_t)
    summation = np.sum(((-1) ** k_b) * term1 * term2, axis=-1)

    out[valid] = (result * summation)[valid]
    return out


def rho_E_alpha_vec(t, bound, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, rate_norm_l, alpha):
    t_original = np.asarray(t, dtype=float)
    bound = np.asarray(bound)
    v, omega = gamma_omega_alpha_vec(ABL, ILD, rate_lambda, T_0, theta_E, rate_norm_l, alpha)

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

    K_half = int(K_max / 2)
    k_vals = np.linspace(-K_half, K_half, 2 * K_half + 1)
    t_b = safe_t[..., None]
    w_b = w_full[..., None]
    k_b = k_vals.reshape((1,) * len(shape) + (2 * K_half + 1,))

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
# Data loading
# =============================================================================
def load_mse_params():
    mse_df = pd.read_csv(MSE_PARAM_CSV)
    required_cols = ["batch_name", "animal", "success", "rate_lambda", "T_0", "theta_E", "rate_norm_l", "alpha"]
    missing_cols = [col for col in required_cols if col not in mse_df.columns]
    if missing_cols:
        raise KeyError(f"{MSE_PARAM_CSV} missing columns: {missing_cols}")

    mse_df = mse_df[mse_df["success"] == True].copy()
    if len(mse_df) != EXPECTED_N_ANIMALS:
        raise RuntimeError(f"Expected {EXPECTED_N_ANIMALS} successful MSE rows, found {len(mse_df)}")

    params = {}
    for row in mse_df.itertuples(index=False):
        params[(row.batch_name, int(row.animal))] = {
            "rate_lambda": float(row.rate_lambda),
            "T_0": float(row.T_0),
            "theta_E": float(row.theta_E),
            "rate_norm_l": float(row.rate_norm_l),
            "alpha": float(row.alpha),
        }
    return mse_df, params


def load_big_svi_params():
    condition_paths = sorted(BIG_SVI_ROOT.glob("*/*_big_gamma_omega_delay_condition_summary.csv"))
    if len(condition_paths) != EXPECTED_N_ANIMALS:
        raise RuntimeError(f"Expected {EXPECTED_N_ANIMALS} condition summary CSVs, found {len(condition_paths)}")

    condition_rows = []
    scalar_rows = []
    params = {}

    for condition_path in condition_paths:
        condition_df = pd.read_csv(condition_path)
        required_condition_cols = ["batch_name", "animal", "ABL", "ILD", "t_E_aff_mean"]
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
        raise RuntimeError(f"Expected {EXPECTED_N_CONDITION_ROWS} condition rows, found {len(condition_df)}")
    if condition_df[["batch_name", "animal"]].drop_duplicates().shape[0] != EXPECTED_N_ANIMALS:
        raise RuntimeError("Condition summaries do not contain 30 animals.")
    return condition_df, scalar_df, params


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


def discover_matched_animals(raw_df, mse_params, big_svi_params):
    valid_df = raw_df[raw_df["success"].isin([1, -1])].copy()
    raw_pairs = {
        (str(batch), int(animal))
        for batch, animal in valid_df[["batch_name", "animal"]].drop_duplicates().itertuples(index=False)
    }
    common_pairs = sorted(raw_pairs & set(mse_params) & set(big_svi_params), key=sort_pair)
    if len(common_pairs) != EXPECTED_N_ANIMALS:
        raise RuntimeError(f"Expected {EXPECTED_N_ANIMALS} matched animals, found {len(common_pairs)}")
    return common_pairs


# %%
# =============================================================================
# Theory and empirical per-animal calculations
# =============================================================================
def interpolate_t_e_aff(delay_by_condition, ABL, signed_ild):
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
    if abs_ild < x.min() or abs_ild > x.max():
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


def process_animal(pair, raw_df, mse_params, big_svi_params):
    batch_name, animal = pair
    print(f"Processing {batch_name}/{animal}")

    batch_df = raw_df[raw_df["batch_name"].astype(str) == str(batch_name)].copy()
    abort_params = load_abort_params(batch_name, animal)
    big_params = big_svi_params[(batch_name, int(animal))]
    tied_params = dict(mse_params[(batch_name, int(animal))])
    tied_params["w"] = big_params["w"]
    tied_params["del_go"] = big_params["del_go"]
    delay_by_condition = big_params["delay_by_condition"]

    P_A_mean, C_A_mean, t_stim_samples = get_p_a_c_a(batch_df, animal, abort_params)

    empirical_psy = {ABL: {} for ABL in ABL_arr}
    theory_psy = {ABL: {} for ABL in ABL_arr}
    discrete_quantiles = {}
    continuous_quantiles = {}

    for ABL in ABL_arr:
        empirical_psy[ABL] = empirical_psychometric(batch_df, animal, ABL)

        for ILD in ILD_arr:
            raw_q = empirical_rt_quantiles(batch_df, animal, ABL, ILD)
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
                    print(f"  theory failed {batch_name}/{animal} ABL={ABL} ILD={ILD}: {exc}")

            discrete_quantiles[(ABL, float(ILD))] = {"empirical": raw_q, "theoretical": theory_q}

    for ABL in ABL_arr:
        for abs_ild in CONTINUOUS_ABS_ILD:
            sign_quantiles = []
            for sign in [-1, 1]:
                signed_ild = float(sign * abs_ild)
                delay = interpolate_t_e_aff(delay_by_condition, ABL, signed_ild)
                if delay is None:
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
                        delay,
                    )
                    sign_quantiles.append(quantiles_from_rtd(t_pts, up + down))
                except Exception:
                    continue
            if sign_quantiles:
                continuous_quantiles[(ABL, float(abs_ild))] = np.nanmean(sign_quantiles, axis=0)

    return {
        "pair": pair,
        "empirical_psy": empirical_psy,
        "theory_psy": theory_psy,
        "discrete_quantiles": discrete_quantiles,
        "continuous_quantiles": continuous_quantiles,
    }


# %%
# =============================================================================
# Aggregation
# =============================================================================
def aggregate_psychometric(results, animal_keys):
    empirical_agg = {ABL: np.full((len(animal_keys), len(ILD_arr)), np.nan) for ABL in ABL_arr}
    theory_agg = {ABL: np.full((len(animal_keys), len(ILD_arr)), np.nan) for ABL in ABL_arr}

    for animal_idx, result in enumerate(results):
        for ABL in ABL_arr:
            for ild_idx, ILD in enumerate(ILD_arr):
                empirical_agg[ABL][animal_idx, ild_idx] = result["empirical_psy"][ABL].get(float(ILD), np.nan)
                theory_agg[ABL][animal_idx, ild_idx] = result["theory_psy"][ABL].get(float(ILD), np.nan)

    return {
        "empirical_agg": empirical_agg,
        "theory_agg": theory_agg,
        "ILD_arr": ILD_arr,
        "animal_keys": animal_keys,
        "MODEL_TYPE": "mse_npl_params_patience12_big_svi_delay",
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
        for ABL in ABL_arr:
            data_fit = fit_psychometric_sigmoid(ILD_arr, empirical_agg[ABL][animal_idx])
            model_fit = fit_psychometric_sigmoid(ILD_arr, theory_agg[ABL][animal_idx])

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
        "ABL_arr": ABL_arr,
        "MODEL_TYPE": "mse_npl_params_patience12_big_svi_delay",
    }


def aggregate_quantiles(results, animal_keys):
    plot_data = defaultdict(_create_inner_defaultdict)
    continuous_plot_data = defaultdict(_create_inner_defaultdict)

    for result in results:
        discrete_quantiles = result["discrete_quantiles"]
        continuous_quantiles = result["continuous_quantiles"]

        for ABL in ABL_arr:
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

    plot_data_for_pickle = {
        ABL: {
            abs_ild: {
                "empirical": list(plot_data[ABL][abs_ild]["empirical"]),
                "theoretical": list(plot_data[ABL][abs_ild]["theoretical"]),
            }
            for abs_ild in ABS_ILD_SORTED
        }
        for ABL in ABL_arr
    }
    continuous_plot_data_for_pickle = {
        ABL: {
            float(abs_ild): {
                "empirical": list(continuous_plot_data[ABL][float(abs_ild)]["empirical"]),
                "theoretical": list(continuous_plot_data[ABL][float(abs_ild)]["theoretical"]),
            }
            for abs_ild in CONTINUOUS_ABS_ILD
        }
        for ABL in ABL_arr
    }
    return {
        "plot_data": plot_data_for_pickle,
        "continuous_plot_data": continuous_plot_data_for_pickle,
        "QUANTILES_TO_PLOT": QUANTILES_TO_PLOT,
        "abs_ild_sorted": ABS_ILD_SORTED,
        "continuous_abs_ild": CONTINUOUS_ABS_ILD,
        "ABL_arr": ABL_arr,
        "animal_keys": animal_keys,
        "MODEL_TYPE": "mse_npl_params_patience12_big_svi_delay",
    }


# %%
# =============================================================================
# Plotting
# =============================================================================
def plot_psychometric(ax, data):
    colors = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}
    empirical_agg = data["empirical_agg"]
    theory_agg = data["theory_agg"]

    for ABL in ABL_arr:
        emp = empirical_agg[ABL]
        theo = theory_agg[ABL]
        emp_mean = np.nanmean(emp, axis=0)
        n_emp = np.sum(np.isfinite(emp), axis=0)
        emp_sem = np.nanstd(emp, axis=0, ddof=1) / np.sqrt(n_emp)

        ax.errorbar(
            ILD_arr,
            emp_mean,
            yerr=emp_sem,
            fmt="o",
            color=colors[ABL],
            capsize=0,
            markersize=6,
            label=f"Data ABL={ABL}",
        )

        theo_mean = np.nanmean(theo, axis=0)
        popt = fit_psychometric_sigmoid(ILD_arr, theo_mean)
        if popt is not None:
            valid_ilds = ILD_arr[np.isfinite(theo_mean)]
            ilds_smooth = np.linspace(np.nanmin(valid_ilds), np.nanmax(valid_ilds), 200)
            ax.plot(ilds_smooth, sigmoid(ilds_smooth, *popt), "-", color=colors[ABL], label=f"Model ABL={ABL}")
        ax.plot(ILD_arr, theo_mean, "x", color=colors[ABL], markersize=5, alpha=0.75)

    ax.set_xlabel("ILD (dB)", fontsize=ft.STYLE.LABEL_FONTSIZE)
    ax.set_ylabel("P(choice = right)", fontsize=ft.STYLE.LABEL_FONTSIZE)
    ax.set_xticks([-15, -5, 5, 15])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(axis="both", labelsize=ft.STYLE.TICK_FONTSIZE)
    ax.axvline(0, alpha=0.5, color="grey", linestyle="--")
    ax.axhline(0.5, alpha=0.5, color="grey", linestyle="--")
    ax.set_ylim(-0.05, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_box_aspect(1)


def plot_slopes(ax, data):
    data_means = data["data_means"]
    model_means = data["model_means"]

    ax.scatter(data_means, model_means, marker="o", s=54, facecolors="w", edgecolors="k", linewidths=1.3)
    ax.set_xlabel("Data", fontsize=ft.STYLE.LABEL_FONTSIZE)
    ax.set_ylabel("Model", fontsize=ft.STYLE.LABEL_FONTSIZE)
    ax.set_xticks([0.1, 0.5, 0.9])
    ax.set_yticks([0.1, 0.5, 0.9])
    ax.set_xlim(0.1, 0.9)
    ax.set_ylim(0.1, 0.9)
    ax.tick_params(axis="both", labelsize=ft.STYLE.TICK_FONTSIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.plot([0.1, 0.9], [0.1, 0.9], color="grey", alpha=0.5, linestyle="--", linewidth=2, zorder=0)
    ax.set_box_aspect(1)

    corr = pearson_r_data_vs_model(data_means, model_means)
    if np.isfinite(corr):
        ax.set_title(f"Slope\nr={corr:.2f}", fontsize=ft.STYLE.LABEL_FONTSIZE)


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


def plot_quantiles(ax, data, mode, quantiles_to_show=None, title_suffix=""):
    plot_data = data["plot_data"]
    ABLs = data["ABL_arr"]
    abs_ild_sorted = data["abs_ild_sorted"]
    if quantiles_to_show is None:
        quantiles_to_show = data["QUANTILES_TO_PLOT"]
    quantile_indices, quantile_labels = quantile_indices_for_plot(data, quantiles_to_show)

    if mode == "continuous":
        theory_source = data["continuous_plot_data"]
        theory_x = data["continuous_abs_ild"]
        theory_marker = None
        theory_linestyle = "-"
        title = f"RT quantiles\ninterpolated delay{title_suffix}"
    elif mode == "discrete":
        theory_source = plot_data
        theory_x = abs_ild_sorted
        theory_marker = "x"
        theory_linestyle = "none"
        title = f"RT quantiles\ndiscrete delay{title_suffix}"
    else:
        raise ValueError(mode)

    for plot_idx, (q_idx, q) in enumerate(zip(quantile_indices, quantile_labels)):
        emp_means, emp_sems = [], []
        for abs_ild in abs_ild_sorted:
            values = []
            for ABL in ABLs:
                entries = plot_data[ABL][abs_ild]["empirical"]
                if len(entries) > 0:
                    values.extend(np.asarray(entries, dtype=float)[:, q_idx])
            mean, curr_sem, _n = nanmean_sem(values)
            emp_means.append(mean)
            emp_sems.append(curr_sem)

        ax.errorbar(
            abs_ild_sorted,
            emp_means,
            yerr=emp_sems,
            fmt="o",
            color="black",
            markersize=5,
            capsize=0,
            alpha=0.9,
            label=f"Data q={q:.1f}" if plot_idx == 0 else "_nolegend_",
        )

        theo_means, theo_sems, theo_x_valid = [], [], []
        for abs_ild in theory_x:
            values = []
            for ABL in ABLs:
                entries = theory_source[ABL][float(abs_ild)]["theoretical"]
                if len(entries) > 0:
                    values.extend(np.asarray(entries, dtype=float)[:, q_idx])
            mean, curr_sem, n = nanmean_sem(values)
            if n > 0:
                theo_x_valid.append(float(abs_ild))
                theo_means.append(mean)
                theo_sems.append(curr_sem)

        if theo_x_valid:
            if mode == "continuous":
                ax.plot(
                    theo_x_valid,
                    theo_means,
                    color="tab:red",
                    linestyle=theory_linestyle,
                    linewidth=1.2,
                    label=f"Model q={q:.1f}" if plot_idx == 0 else "_nolegend_",
                )
                ax.fill_between(
                    theo_x_valid,
                    np.array(theo_means) - np.array(theo_sems),
                    np.array(theo_means) + np.array(theo_sems),
                    color="tab:red",
                    alpha=0.16,
                    linewidth=0,
                )
            else:
                ax.errorbar(
                    theo_x_valid,
                    theo_means,
                    yerr=theo_sems,
                    fmt=theory_marker,
                    color="tab:red",
                    markersize=6,
                    capsize=0,
                    linestyle=theory_linestyle,
                    label=f"Model q={q:.1f}" if plot_idx == 0 else "_nolegend_",
                )

    ax.set_title(title, fontsize=ft.STYLE.LABEL_FONTSIZE)
    ax.set_xlabel("|ILD| (dB)", fontsize=ft.STYLE.LABEL_FONTSIZE)
    ax.set_ylabel("RT quantile (s)", fontsize=ft.STYLE.LABEL_FONTSIZE)
    ax.set_xscale("log", base=2)
    ax.set_xticks(abs_ild_sorted)
    ax.set_yticks([0.1, 0.2, 0.3, 0.4])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.tick_params(axis="both", which="major", labelsize=ft.STYLE.TICK_FONTSIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_box_aspect(1)


def make_figure(psy_data, slopes_data, quantile_data):
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.4))
    axes = axes.ravel()

    plot_psychometric(axes[0], psy_data)
    plot_slopes(axes[1], slopes_data)
    plot_quantiles(axes[2], quantile_data, "continuous", title_suffix="\nall deciles")
    plot_quantiles(axes[3], quantile_data, "discrete", title_suffix="\nall deciles")
    plot_quantiles(axes[4], quantile_data, "continuous", PAPER_QUANTILES_TO_PLOT, "\npaper q")
    plot_quantiles(axes[5], quantile_data, "discrete", PAPER_QUANTILES_TO_PLOT, "\npaper q")

    axes[0].set_title("Psychometric", fontsize=ft.STYLE.LABEL_FONTSIZE)
    if not axes[1].get_title():
        axes[1].set_title("Slope", fontsize=ft.STYLE.LABEL_FONTSIZE)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles[:6],
        labels[:6],
        loc="upper center",
        ncol=6,
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(0.5, 1.03),
    )

    fig.suptitle(
        (
            "MSE NPL parameters "
            f"({MSE_OBJECTIVE_LABEL}) with patience12 big-SVI w, del_go, and condition t_E_aff"
        ),
        fontsize=16,
        y=1.07,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    return fig


# %%
# =============================================================================
# Main analysis
# =============================================================================
print(f"MSE params: {MSE_PARAM_CSV}")
print(f"MSE objective: {MSE_OBJECTIVE} ({MSE_OBJECTIVE_LABEL})")
print(f"Big SVI root: {BIG_SVI_ROOT}")
print(f"Output folder: {OUTPUT_DIR}")
print(
    f"N_THEORY={N_THEORY}, N_JOBS={N_JOBS}, joblib prefer={JOBLIB_PREFER}, "
    f"continuous ILD step={CONTINUOUS_ILD_STEP}"
)

mse_df, mse_params = load_mse_params()
big_condition_df, big_scalar_df, big_svi_params = load_big_svi_params()
raw_df = load_raw_data()
animal_keys = discover_matched_animals(raw_df, mse_params, big_svi_params)

print(f"Loaded MSE params for {len(mse_params)} animals")
print(f"Loaded big SVI condition rows: {len(big_condition_df)}")
print(f"Loaded big SVI scalar rows: {len(big_scalar_df)}")
print(f"Matched animals: {len(animal_keys)}")
print("Animal list:")
print(", ".join([f"{batch}/{animal}" for batch, animal in animal_keys]))

results = Parallel(n_jobs=N_JOBS, verbose=10, prefer=JOBLIB_PREFER)(
    delayed(process_animal)(pair, raw_df, mse_params, big_svi_params) for pair in animal_keys
)

psy_data = aggregate_psychometric(results, animal_keys)
slopes_data = aggregate_slopes(psy_data)
quantile_data = aggregate_quantiles(results, animal_keys)

sd_rows = np.array([batch == "SD" for batch, _animal in animal_keys])
sd_high_ild_model_count = 0
for ABL in ABL_arr:
    high_cols = np.abs(ILD_arr) > 8
    sd_high_ild_model_count += int(np.sum(np.isfinite(psy_data["theory_agg"][ABL][np.ix_(sd_rows, high_cols)])))
print(f"SD model psychometric entries at |ILD|>8: {sd_high_ild_model_count}")
if sd_high_ild_model_count != 0:
    raise RuntimeError("SD model psychometric/slope grid should not include |ILD|>8.")

fig = make_figure(psy_data, slopes_data, quantile_data)

payload = {
    "psy_data": psy_data,
    "slopes_data": slopes_data,
    "quantile_data": quantile_data,
    "mse_objective": MSE_OBJECTIVE,
    "mse_objective_label": MSE_OBJECTIVE_LABEL,
    "mse_params_csv": str(MSE_PARAM_CSV),
    "big_svi_root": str(BIG_SVI_ROOT),
    "animal_keys": animal_keys,
    "mse_params": mse_params,
    "big_condition_rows": big_condition_df.to_dict("records"),
    "big_scalar_rows": big_scalar_df.to_dict("records"),
    "output_png": str(OUTPUT_PNG),
    "config": {
        "N_THEORY": N_THEORY,
        "N_JOBS": N_JOBS,
        "RNG_SEED": RNG_SEED,
        "MSE_OBJECTIVE": MSE_OBJECTIVE,
        "MSE_OBJECTIVE_LABEL": MSE_OBJECTIVE_LABEL,
        "CONTINUOUS_ILD_STEP": CONTINUOUS_ILD_STEP,
        "QUANTILES_TO_PLOT": QUANTILES_TO_PLOT,
        "PAPER_QUANTILES_TO_PLOT": PAPER_QUANTILES_TO_PLOT,
        "T_PTS_START": float(T_PTS[0]),
        "T_PTS_STOP": float(T_PTS[-1]),
        "T_PTS_STEP": float(T_PTS[1] - T_PTS[0]),
        "sd_psychometric_model_abs_ild_max": 8,
        "continuous_delay_policy": "signed-branch linear interpolation, no extrapolation",
        "discrete_delay_policy": "exact condition t_E_aff only",
    },
}
with OUTPUT_PKL.open("wb") as handle:
    pickle.dump(payload, handle)

print(f"Saved figure: {OUTPUT_PNG}")
print(f"Saved data: {OUTPUT_PKL}")

plt.show()

# %%
