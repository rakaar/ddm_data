# %%
"""
Compare the animal-wise ABL-specific ILD2 delay model against a diagnostic
variant that substitutes only the condition-fit t_E_aff.

The NPL + alpha + ABL-specific ILD2 fit parameters stay fixed. For the
condition-delay diagnostic, t_E_aff is replaced by the per animal/ABL/ILD
posterior mean from the condition-fit cache.
"""

# %%
import os
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

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
from scipy.stats import sem

try:
    import scipy.special._ufuncs as scipy_ufuncs

    sys.modules.setdefault("scipy.special._special_ufuncs", scipy_ufuncs)
except Exception:
    pass

from time_vary_norm_utils import M, phi, rho_A_t_VEC_fn


# %%
# Parameters
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

UPSTREAM_DIR = REPO_DIR / "all_30_NPL_alpha_ABL_specific_ILD2_delay_fit_results"
CONDITION_CACHE = REPO_DIR / "fit_each_condn" / "abl_specific_ild2_delay_agreement" / "condition_t_E_aff_extraction_cache.csv"
RAW_BATCH_DIR = REPO_DIR / "raw_data" / "batch_csvs"
OUTPUT_DIR = SCRIPT_DIR / "rtd_psychometric_abl_specific_delay_vs_condition_delay"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RTD_OUTPUT_PNG = OUTPUT_DIR / "rtd_abl_specific_delay_vs_condition_delay.png"
RTD_ABS_ILD_OUTPUT_PNG = OUTPUT_DIR / "rtd_by_abl_abs_ild_delay_vs_condition_delay.png"
RTD_ABS_ILD_ZOOM_OUTPUT_PNG = OUTPUT_DIR / "rtd_by_abl_abs_ild_delay_vs_condition_delay_xlim_minus0p6_0p6.png"
PSY_OUTPUT_PNG = OUTPUT_DIR / "psychometric_abl_specific_delay_vs_condition_delay.png"
OUTPUT_PKL = OUTPUT_DIR / "rtd_psychometric_abl_specific_delay_vs_condition_delay.pkl"

MODEL_KEY = "vbmc_norm_alpha_abl_specific_ild2_delay_tied_results"
EXPECTED_N_ANIMALS = 30
EXPECTED_CONDITION_ROWS = 864
DESIRED_BATCHES = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]
ABLS = [20, 40, 60]
ILD_GRID = [-16.0, -8.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0, 16.0]
ABS_ILD_GRID = [1.0, 2.0, 4.0, 8.0, 16.0]
RAW_BATCH_FILE_MAP = {
    "SD": "outExp.csv",
    "LED34": "outExp.csv",
    "LED6": "outExp.csv",
    "LED8": "outLED8.csv",
    "LED7": "out_LED.csv",
    "LED34_even": "outUni.csv",
}
THEORY_ABORT_EVENTS = [3]
RTD_ABORT_EVENTS = [3, 4]

RT_MIN = -1.0
RT_MAX = 1.0
RT_BIN_WIDTH = 0.02
RT_BINS = np.arange(RT_MIN, RT_MAX + RT_BIN_WIDTH, RT_BIN_WIDTH)
RT_BIN_CENTERS = 0.5 * (RT_BINS[:-1] + RT_BINS[1:])
DT_THEORY = 0.001
T_PTS = np.arange(-2.0, 2.0, DT_THEORY)
THEORY_RT_MASK = (T_PTS >= RT_MIN) & (T_PTS <= RT_MAX)
THEORY_RT_PTS = T_PTS[THEORY_RT_MASK]
K_MAX = 10
N_THEORY = int(1e3)
N_JOBS = max(1, min(24, (os.cpu_count() or 2) - 1))

BATCH_T_TRUNC = {"LED34_even": 0.15}
DEFAULT_T_TRUNC = 0.3

COLORS = {
    "data": "black",
    "original": "#D55E00",
    "condition": "#0072B2",
}
LABELS = {
    "data": "Data",
    "original": "NPL ABL-ILD2 delay",
    "condition": "Condition t_E_aff",
}


# %%
# Model density helpers
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

    K_half = int(K_MAX / 2)
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

    P_EA_hits_either_bound = (
        CDF_E_alpha_vec(
            t2,
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
            t2,
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
    random_readout_if_EA_survives = 0.5 * (1 - P_EA_hits_either_bound)

    P_E_plus_cum = (
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
    P_E_plus = rho_E_alpha_vec(
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

    return P_A * (random_readout_if_EA_survives + P_E_plus_cum) + P_E_plus * (1 - C_A)


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
# Loading and summarizing helpers
def parse_upstream_result_name(path):
    match = re.match(
        r"^results_(?P<batch>.+)_animal_(?P<animal>\d+)_NORM_ALPHA_ABL_SPECIFIC_ILD2_DELAY_FROM_ABORTS\.pkl$",
        path.name,
    )
    if match is None:
        return None
    return match.group("batch"), int(match.group("animal"))


def normalize_density(density):
    density = np.asarray(density, dtype=float)
    area = np.nansum(density * np.diff(RT_BINS))
    if np.isfinite(area) and area > 0:
        return density / area
    return np.full_like(density, np.nan, dtype=float)


def normalize_continuous_density(t_pts, density):
    density = np.asarray(density, dtype=float)
    area = trapezoid(density, t_pts)
    if np.isfinite(area) and area > 0:
        return density / area
    return np.full_like(density, np.nan, dtype=float)


def raw_rts_to_hist(raw_rts):
    if len(raw_rts) <= 5:
        return np.full(len(RT_BIN_CENTERS), np.nan)
    hist, _ = np.histogram(raw_rts, bins=RT_BINS, density=True)
    return normalize_density(hist)


def theoretical_rtd_to_bin_density(t_pts, rtd):
    binned_rtd = []
    for bin_idx, (left, right) in enumerate(zip(RT_BINS[:-1], RT_BINS[1:])):
        if bin_idx == len(RT_BINS) - 2:
            in_bin = (t_pts >= left) & (t_pts <= right)
        else:
            in_bin = (t_pts >= left) & (t_pts < right)
        if np.sum(in_bin) < 2:
            binned_rtd.append(np.nan)
            continue
        binned_rtd.append(trapezoid(rtd[in_bin], t_pts[in_bin]) / (right - left))
    return normalize_density(np.asarray(binned_rtd, dtype=float))


def mean_and_sem(curves):
    curves = np.asarray(curves, dtype=float)
    if curves.size == 0:
        return (
            np.full(len(RT_BIN_CENTERS), np.nan),
            np.full(len(RT_BIN_CENTERS), np.nan),
        )
    return np.nanmean(curves, axis=0), sem(curves, axis=0, nan_policy="omit")


def mean_and_sem_continuous(curves):
    curves = np.asarray(curves, dtype=float)
    if curves.size == 0:
        return (
            np.full(len(THEORY_RT_PTS), np.nan),
            np.full(len(THEORY_RT_PTS), np.nan),
        )
    return np.nanmean(curves, axis=0), sem(curves, axis=0, nan_policy="omit")


def psychometric_mean_sem(values):
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values, axis=0)
    err = sem(values, axis=0, nan_policy="omit")
    counts = np.sum(np.isfinite(values), axis=0)
    return mean, err, counts


def average_equal_weight_curves(curves):
    curves = [np.asarray(curve, dtype=float) for curve in curves if np.any(np.isfinite(curve))]
    if not curves:
        return np.full(len(RT_BIN_CENTERS), np.nan)
    return normalize_density(np.nanmean(np.asarray(curves), axis=0))


def average_equal_weight_continuous_curves(curves):
    curves = [np.asarray(curve, dtype=float) for curve in curves if np.any(np.isfinite(curve))]
    if not curves:
        return np.full(len(THEORY_RT_PTS), np.nan)
    return normalize_continuous_density(THEORY_RT_PTS, np.nanmean(np.asarray(curves), axis=0))


def calculate_truncated_theoretical_curves(df_valid_and_aborts, N_theory, t_pts, t_A_aff, V_A, theta_A, T_trunc):
    t_stim_samples = df_valid_and_aborts["intended_fix"].sample(N_theory, replace=True).values

    trunc_cdf = cum_A_t_vec(np.array([T_trunc - t_A_aff]), V_A, theta_A)[0]
    trunc_survival = max(1 - trunc_cdf, 1e-12)

    P_A_samples = np.zeros((N_theory, len(t_pts)))
    C_A_samples = np.zeros((N_theory, len(t_pts)))
    for idx, t_stim in enumerate(t_stim_samples):
        total_time = t_pts + t_stim
        after_trunc = total_time >= T_trunc

        p_a = rho_A_t_VEC_fn(total_time - t_A_aff, V_A, theta_A)
        c_a = cum_A_t_vec(total_time - t_A_aff, V_A, theta_A)

        P_A_samples[idx, :] = np.where(after_trunc, p_a / trunc_survival, 0.0)
        C_A_samples[idx, :] = np.where(after_trunc, (c_a - trunc_cdf) / trunc_survival, 0.0)

    return (
        np.mean(P_A_samples, axis=0),
        np.clip(np.mean(C_A_samples, axis=0), 0.0, 1.0),
        t_stim_samples,
    )


def delay_s_from_abl_specific_params(ABL, ILD, tied_params):
    abl_levels = tied_params["delay_abl_levels"]
    matches = np.where(np.isclose(abl_levels, float(ABL)))[0]
    if len(matches) != 1:
        raise RuntimeError(f"Could not find ABL={ABL} in delay_abl_levels={abl_levels}")
    abl_idx = int(matches[0])
    delay_ms = (
        tied_params["bias_ms_by_abl"][abl_idx]
        + tied_params["abs_ild_delay_coeff_ms_per_unit_by_abl"][abl_idx] * abs(float(ILD))
        + tied_params["ild2_delay_coeff_ms_per_unit2_by_abl"][abl_idx] * (float(ILD) ** 2)
    )
    return delay_ms * 1e-3


def get_theoretical_up_down(P_A_mean, C_A_mean, t_stim_samples, abort_params, tied_params, ABL, ILD, batch_name, t_E_aff):
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

    return (
        T_PTS,
        up_mean / trunc_factor,
        down_mean / trunc_factor,
        float(trunc_factor),
    )


def load_posterior_mean_params(result_path):
    with result_path.open("rb") as handle:
        fit_results = pickle.load(handle)

    abort_samples = fit_results["vbmc_aborts_results"]
    tied_samples = fit_results[MODEL_KEY]
    message = str(tied_samples.get("message", ""))
    if "stable" not in message.lower():
        raise RuntimeError(f"Upstream fit is not stable for {result_path.name}: {message}")

    delay_abl_levels = np.asarray(tied_samples["delay_abl_levels"], dtype=float)
    if len(delay_abl_levels) != len(ABLS) or not np.allclose(delay_abl_levels, ABLS):
        raise RuntimeError(f"{result_path.name} has delay_abl_levels={delay_abl_levels}, expected {ABLS}")

    abort_params = {
        "V_A": float(np.mean(abort_samples["V_A_samples"])),
        "theta_A": float(np.mean(abort_samples["theta_A_samples"])),
        "t_A_aff": float(np.mean(abort_samples["t_A_aff_samp"])),
    }
    tied_params = {
        "rate_lambda": float(np.mean(tied_samples["rate_lambda_samples"])),
        "T_0": float(np.mean(tied_samples["T_0_samples"])),
        "theta_E": float(np.mean(tied_samples["theta_E_samples"])),
        "w": float(np.mean(tied_samples["w_samples"])),
        "del_go": float(np.mean(tied_samples["del_go_samples"])),
        "rate_norm_l": float(np.mean(tied_samples["rate_norm_l_samples"])),
        "alpha": float(np.mean(tied_samples["alpha_samples"])),
        "delay_abl_levels": delay_abl_levels,
        "bias_ms_by_abl": np.mean(tied_samples["bias_ms_by_abl_samples"], axis=0),
        "abs_ild_delay_coeff_ms_per_unit_by_abl": np.mean(
            tied_samples["abs_ild_delay_coeff_ms_per_unit_by_abl_samples"], axis=0
        ),
        "ild2_delay_coeff_ms_per_unit2_by_abl": np.mean(
            tied_samples["ild2_delay_coeff_ms_per_unit2_by_abl_samples"], axis=0
        ),
    }
    return abort_params, tied_params


# %%
# Preflight
print(f"Upstream results: {UPSTREAM_DIR}")
print(f"Condition t_E_aff cache: {CONDITION_CACHE}")
print(f"Raw batch CSVs: {RAW_BATCH_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

condition_cache = pd.read_csv(CONDITION_CACHE)
if len(condition_cache) != EXPECTED_CONDITION_ROWS:
    raise RuntimeError(f"Expected {EXPECTED_CONDITION_ROWS} condition-cache rows, found {len(condition_cache)}")

condition_cache = condition_cache.rename(columns={"batch_name": "batch_name"})
condition_cache["animal"] = condition_cache["animal"].astype(int)
condition_cache["ABL"] = condition_cache["ABL"].astype(int)
condition_cache["ILD"] = condition_cache["ILD"].astype(float)
condition_cache["t_E_aff_s"] = condition_cache["t_E_aff_s"].astype(float)

rerun_checks = {
    -1.0: (53.051370, 0.75),
    1.0: (45.648227, 0.75),
}
for ild, (expected_ms, tol_ms) in rerun_checks.items():
    row = condition_cache[
        (condition_cache["batch_name"] == "LED7")
        & (condition_cache["animal"] == 92)
        & (condition_cache["ABL"] == 20)
        & np.isclose(condition_cache["ILD"], ild)
    ]
    if len(row) != 1:
        raise RuntimeError(f"Expected one LED7/92 ABL=20 ILD={ild:g} cache row, found {len(row)}")
    value_ms = float(row["t_E_aff_ms"].iloc[0])
    print(f"Preflight LED7/92 ABL=20 ILD={ild:+g}: condition cache t_E_aff = {value_ms:.3f} ms")
    if not np.isclose(value_ms, expected_ms, atol=tol_ms):
        raise RuntimeError(
            f"LED7/92 ABL=20 ILD={ild:g} cache value {value_ms:.3f} ms is outside rerun range "
            f"around {expected_ms:.3f} ms"
        )

condition_t_E_aff = {
    (row.batch_name, int(row.animal), int(row.ABL), float(row.ILD)): float(row.t_E_aff_s)
    for row in condition_cache.itertuples(index=False)
}

result_paths = {}
for result_path in sorted(UPSTREAM_DIR.glob("results_*_NORM_ALPHA_ABL_SPECIFIC_ILD2_DELAY_FROM_ABORTS.pkl")):
    parsed = parse_upstream_result_name(result_path)
    if parsed is None:
        continue
    result_paths[parsed] = result_path

if len(result_paths) != EXPECTED_N_ANIMALS:
    raise RuntimeError(f"Expected {EXPECTED_N_ANIMALS} upstream result pickles, found {len(result_paths)}")

fit_config_t_trunc = {}
for (batch_name, animal_id), result_path in sorted(result_paths.items()):
    with result_path.open("rb") as handle:
        fit_results = pickle.load(handle)
    fit_config = fit_results.get("fit_config", {})
    if "T_trunc" not in fit_config:
        raise RuntimeError(f"{result_path.name} is missing fit_config['T_trunc']")
    actual_t_trunc = float(fit_config["T_trunc"])
    expected_t_trunc = BATCH_T_TRUNC.get(batch_name, DEFAULT_T_TRUNC)
    if not np.isclose(actual_t_trunc, expected_t_trunc, atol=1e-12):
        raise RuntimeError(
            f"{result_path.name} has fit_config T_trunc={actual_t_trunc:.3f}; "
            f"expected {expected_t_trunc:.3f}"
        )
    fit_config_t_trunc[(batch_name, animal_id)] = actual_t_trunc

batch_frames = []
for batch_name in DESIRED_BATCHES:
    batch_path = RAW_BATCH_DIR / f"batch_{batch_name}_valid_and_aborts.csv"
    if batch_path.exists():
        batch_frames.append(pd.read_csv(batch_path))
if not batch_frames:
    raise RuntimeError(f"No batch CSVs found in {RAW_BATCH_DIR}")

raw_data = pd.concat(batch_frames, ignore_index=True)
raw_data["animal"] = raw_data["animal"].astype(int)
raw_data["ABL"] = raw_data["ABL"].astype(int)
raw_data["ILD"] = raw_data["ILD"].astype(float)
raw_data["T_trunc"] = raw_data["batch_name"].map(BATCH_T_TRUNC).fillna(DEFAULT_T_TRUNC)
raw_data = raw_data[~(raw_data["RTwrtStim"].isna() & raw_data["abort_event"].isin(THEORY_ABORT_EVENTS))].copy()

valid_or_abort = raw_data["success"].isin([1, -1]) | raw_data["abort_event"].isin(THEORY_ABORT_EVENTS)
valid_and_abort_data = raw_data[valid_or_abort].copy()

valid_data = raw_data[
    raw_data["success"].isin([1, -1])
    & raw_data["RTwrtStim"].between(0, 1, inclusive="both")
    & raw_data["ABL"].isin(ABLS)
].copy()

rtd_source_frames = []
for batch_name in DESIRED_BATCHES:
    raw_batch_path = REPO_DIR / "raw_data" / RAW_BATCH_FILE_MAP[batch_name]
    exp_df = pd.read_csv(raw_batch_path)

    if "timed_fix" in exp_df.columns:
        exp_df.loc[:, "RTwrtStim"] = exp_df["timed_fix"] - exp_df["intended_fix"]
        exp_df = exp_df.rename(columns={"timed_fix": "TotalFixTime"})

    exp_df = exp_df[
        ~(exp_df["RTwrtStim"].isna() & exp_df["abort_event"].isin(RTD_ABORT_EVENTS))
    ].copy()

    led_off = exp_df["LED_trial"].eq(0) | exp_df["LED_trial"].isna()
    if batch_name == "SD":
        exp_df_batch = exp_df[
            exp_df["batch_name"].eq(batch_name)
            & led_off
            & exp_df["session_type"].isin([1, 7])
        ].copy()
    elif batch_name == "LED34":
        exp_df_batch = exp_df[
            exp_df["batch_name"].eq(batch_name)
            & led_off
            & exp_df["session_type"].isin([1, 2])
            & exp_df["animal"].isin([45, 57, 59, 61, 63])
            & exp_df["ABL"].isin(ABLS)
            & exp_df["ILD"].isin(ILD_GRID)
        ].copy()
    elif batch_name == "LED6":
        exp_df_batch = exp_df[
            exp_df["batch_name"].eq(batch_name)
            & led_off
            & exp_df["session_type"].isin([1, 2])
        ].copy()
    elif batch_name == "LED7":
        exp_df_batch = exp_df[
            led_off
            & exp_df["session_type"].isin([7])
            & exp_df["training_level"].isin([16])
            & (exp_df["repeat_trial"].isin([0, 2]) | exp_df["repeat_trial"].isna())
        ].copy()
    elif batch_name == "LED8":
        exp_df_batch = exp_df[
            led_off
            & exp_df["session_type"].isin([1])
            & exp_df["training_level"].isin([16])
            & (exp_df["repeat_trial"].isin([0, 2]) | exp_df["repeat_trial"].isna())
        ].copy()
    elif batch_name == "LED34_even":
        exp_df_batch = exp_df[
            exp_df["batch_name"].eq("LED34")
            & led_off
            & exp_df["session_type"].isin([1, 2])
            & exp_df["animal"].isin([48, 52, 56, 60])
        ].copy()
    else:
        raise RuntimeError(f"Unknown batch for RTD raw-data load: {batch_name}")

    exp_df_batch["batch_name"] = batch_name
    rtd_source_frames.append(exp_df_batch)

rtd_source_data = pd.concat(rtd_source_frames, ignore_index=True)
rtd_source_data["animal"] = rtd_source_data["animal"].astype(int)
rtd_source_data["ABL"] = rtd_source_data["ABL"].astype(int)
rtd_source_data["ILD"] = rtd_source_data["ILD"].astype(float)
rtd_source_data["T_trunc"] = rtd_source_data["batch_name"].map(BATCH_T_TRUNC).fillna(DEFAULT_T_TRUNC)

rtd_valid_or_abort = rtd_source_data["success"].isin([1, -1]) | rtd_source_data["abort_event"].isin(RTD_ABORT_EVENTS)
rtd_trial_pool_data = rtd_source_data[rtd_valid_or_abort].copy()

rtd_data = rtd_trial_pool_data[
    rtd_trial_pool_data["ABL"].isin(ABLS)
    & rtd_trial_pool_data["RTwrtStim"].between(RT_MIN, RT_MAX, inclusive="both")
    & (
        rtd_trial_pool_data["success"].isin([1, -1])
        | rtd_trial_pool_data["TotalFixTime"].ge(rtd_trial_pool_data["T_trunc"])
    )
].copy()

abort_truncation_counts = {}
for batch_name in DESIRED_BATCHES:
    batch_t_trunc = BATCH_T_TRUNC.get(batch_name, DEFAULT_T_TRUNC)
    batch_aborts = rtd_trial_pool_data[
        (rtd_trial_pool_data["batch_name"] == batch_name)
        & rtd_trial_pool_data["abort_event"].isin(RTD_ABORT_EVENTS)
    ].copy()
    kept_after_trunc = batch_aborts["TotalFixTime"].ge(batch_t_trunc)
    kept_in_window = (
        kept_after_trunc
        & batch_aborts["ABL"].isin(ABLS)
        & batch_aborts["RTwrtStim"].between(RT_MIN, RT_MAX, inclusive="both")
    )
    abort_truncation_counts[batch_name] = {
        "T_trunc_s": float(batch_t_trunc),
        "abort_event_rows": {
            str(event): int(batch_aborts["abort_event"].eq(event).sum())
            for event in RTD_ABORT_EVENTS
        },
        "kept_after_trunc_by_event": {
            str(event): int((kept_after_trunc & batch_aborts["abort_event"].eq(event)).sum())
            for event in RTD_ABORT_EVENTS
        },
        "kept_after_trunc_in_rtd_window_by_event": {
            str(event): int((kept_in_window & batch_aborts["abort_event"].eq(event)).sum())
            for event in RTD_ABORT_EVENTS
        },
        "abort_event_rows_total": int(len(batch_aborts)),
        "kept_after_trunc": int(kept_after_trunc.sum()),
        "removed_before_trunc": int((~kept_after_trunc).sum()),
        "kept_after_trunc_in_rtd_window": int(kept_in_window.sum()),
    }

batch_animal_pairs = sorted(
    [(batch, int(animal)) for batch, animal in valid_data[["batch_name", "animal"]].drop_duplicates().values],
    key=lambda pair: (DESIRED_BATCHES.index(pair[0]), pair[1]),
)
batch_animal_pairs = [pair for pair in batch_animal_pairs if pair in result_paths]

if len(batch_animal_pairs) != EXPECTED_N_ANIMALS:
    raise RuntimeError(f"Expected {EXPECTED_N_ANIMALS} matched animals, found {len(batch_animal_pairs)}")

print(f"Loaded condition cache rows: {len(condition_cache)}")
print(f"Found stable upstream result dictionaries: {len(result_paths)}")
print(f"Validated upstream fit_config T_trunc values for {len(fit_config_t_trunc)} animals")
print(f"Using matched animals: {len(batch_animal_pairs)}")
print(f"RTD data histogram bin width: {RT_BIN_WIDTH * 1e3:.0f} ms; window [{RT_MIN:g}, {RT_MAX:g}] s")
print(f"RTD data abort events: {RTD_ABORT_EVENTS}; model/theory abort events: {THEORY_ABORT_EVENTS}")
print(f"Model RTD grid spacing: {DT_THEORY * 1e3:.0f} ms; plotted window [{RT_MIN:g}, {RT_MAX:g}] s")
print("Abort truncation counts used for RTD data")
for batch_name, counts in abort_truncation_counts.items():
    print(
        f"  {batch_name}: T_trunc={counts['T_trunc_s']:.3f}, "
        f"abort rows={counts['abort_event_rows']}, "
        f"kept={counts['kept_after_trunc']}, removed={counts['removed_before_trunc']}, "
        f"kept in RTD window={counts['kept_after_trunc_in_rtd_window']} "
        f"by event={counts['kept_after_trunc_in_rtd_window_by_event']}"
    )
print(f"Running model calculations with {N_JOBS} jobs")


# %%
# Per-animal model and empirical summaries
def process_animal(batch_animal_pair):
    batch_name, animal_id = batch_animal_pair
    print(f"Processing {batch_name}/{animal_id}")

    animal_valid_df = valid_data[
        (valid_data["batch_name"] == batch_name) & (valid_data["animal"] == animal_id)
    ].copy()
    animal_rtd_df = rtd_data[
        (rtd_data["batch_name"] == batch_name) & (rtd_data["animal"] == animal_id)
    ].copy()
    animal_theory_df = valid_and_abort_data[
        (valid_and_abort_data["batch_name"] == batch_name) & (valid_and_abort_data["animal"] == animal_id)
    ].copy()

    abort_params, tied_params = load_posterior_mean_params(result_paths[batch_animal_pair])
    T_trunc = BATCH_T_TRUNC.get(batch_name, DEFAULT_T_TRUNC)
    P_A_mean, C_A_mean, t_stim_samples = calculate_truncated_theoretical_curves(
        animal_theory_df,
        N_THEORY,
        T_PTS,
        abort_params["t_A_aff"],
        abort_params["V_A"],
        abort_params["theta_A"],
        T_trunc,
    )

    condition_results = {}
    for ABL in ABLS:
        valid_abl_df = animal_valid_df[animal_valid_df["ABL"] == ABL]
        rtd_abl_df = animal_rtd_df[animal_rtd_df["ABL"] == ABL]
        if valid_abl_df.empty:
            continue
        available_ilds = sorted(float(ild) for ild in valid_abl_df["ILD"].dropna().unique())
        for ILD in available_ilds:
            cond_key = (batch_name, animal_id, int(ABL), float(ILD))
            if cond_key not in condition_t_E_aff:
                continue

            valid_condition_df = valid_abl_df[valid_abl_df["ILD"] == ILD]
            rtd_condition_df = rtd_abl_df[rtd_abl_df["ILD"] == ILD]

            raw_rts = rtd_condition_df["RTwrtStim"].to_numpy(dtype=float)
            empirical_rtd = raw_rts_to_hist(raw_rts)
            data_right_prob = float(np.mean(valid_condition_df["choice"] == 1))

            original_t_E_aff = delay_s_from_abl_specific_params(ABL, ILD, tied_params)
            condition_t = condition_t_E_aff[cond_key]

            t_pts, original_up, original_down, original_trunc = get_theoretical_up_down(
                P_A_mean,
                C_A_mean,
                t_stim_samples,
                abort_params,
                tied_params,
                ABL,
                ILD,
                batch_name,
                original_t_E_aff,
            )
            t_pts, condition_up, condition_down, condition_trunc = get_theoretical_up_down(
                P_A_mean,
                C_A_mean,
                t_stim_samples,
                abort_params,
                tied_params,
                ABL,
                ILD,
                batch_name,
                condition_t,
            )

            display_mask = (t_pts >= RT_MIN) & (t_pts <= RT_MAX)
            valid_rt_mask = (t_pts >= 0) & (t_pts <= 1)

            original_display = original_up[display_mask] + original_down[display_mask]
            condition_display = condition_up[display_mask] + condition_down[display_mask]
            original_rtd = theoretical_rtd_to_bin_density(t_pts[display_mask], original_display)
            condition_rtd = theoretical_rtd_to_bin_density(t_pts[display_mask], condition_display)
            original_rtd_continuous = normalize_continuous_density(t_pts[display_mask], original_display)
            condition_rtd_continuous = normalize_continuous_density(t_pts[display_mask], condition_display)

            original_up_area = trapezoid(original_up[valid_rt_mask], t_pts[valid_rt_mask])
            original_down_area = trapezoid(original_down[valid_rt_mask], t_pts[valid_rt_mask])
            condition_up_area = trapezoid(condition_up[valid_rt_mask], t_pts[valid_rt_mask])
            condition_down_area = trapezoid(condition_down[valid_rt_mask], t_pts[valid_rt_mask])

            condition_results[(ABL, ILD)] = {
                "empirical_rtd": empirical_rtd,
                "original_rtd": original_rtd,
                "condition_rtd": condition_rtd,
                "original_rtd_continuous": original_rtd_continuous,
                "condition_rtd_continuous": condition_rtd_continuous,
                "n_trials": int(len(valid_condition_df)),
                "n_rtd_trials": int(len(raw_rts)),
                "data_right_prob": data_right_prob,
                "original_right_prob": float(original_up_area / (original_up_area + original_down_area)),
                "condition_right_prob": float(condition_up_area / (condition_up_area + condition_down_area)),
                "original_t_E_aff_s": float(original_t_E_aff),
                "condition_t_E_aff_s": float(condition_t),
                "original_trunc_factor": original_trunc,
                "condition_trunc_factor": condition_trunc,
            }

    rtd_by_abl = {}
    psychometric_by_abl = {}
    for ABL in ABLS:
        available_ilds = sorted([ILD for abl, ILD in condition_results if abl == ABL])
        if not available_ilds:
            continue

        rtd_by_abl[ABL] = {
            "empirical": average_equal_weight_curves(
                [condition_results[(ABL, ILD)]["empirical_rtd"] for ILD in available_ilds]
            ),
            "original": average_equal_weight_curves(
                [condition_results[(ABL, ILD)]["original_rtd"] for ILD in available_ilds]
            ),
            "condition": average_equal_weight_curves(
                [condition_results[(ABL, ILD)]["condition_rtd"] for ILD in available_ilds]
            ),
            "original_continuous": average_equal_weight_continuous_curves(
                [condition_results[(ABL, ILD)]["original_rtd_continuous"] for ILD in available_ilds]
            ),
            "condition_continuous": average_equal_weight_continuous_curves(
                [condition_results[(ABL, ILD)]["condition_rtd_continuous"] for ILD in available_ilds]
            ),
            "ilds": np.array(available_ilds, dtype=float),
        }

        psychometric_by_abl[ABL] = {
            "ilds": np.array(available_ilds, dtype=float),
            "data": np.array([condition_results[(ABL, ILD)]["data_right_prob"] for ILD in available_ilds]),
            "original": np.array([condition_results[(ABL, ILD)]["original_right_prob"] for ILD in available_ilds]),
            "condition": np.array([condition_results[(ABL, ILD)]["condition_right_prob"] for ILD in available_ilds]),
            "n_trials": np.array([condition_results[(ABL, ILD)]["n_trials"] for ILD in available_ilds], dtype=int),
        }

    return {
        "batch_name": batch_name,
        "animal": animal_id,
        "condition_results": condition_results,
        "rtd_by_abl": rtd_by_abl,
        "psychometric_by_abl": psychometric_by_abl,
    }


all_animal_results = Parallel(n_jobs=N_JOBS, verbose=10)(
    delayed(process_animal)(pair) for pair in batch_animal_pairs
)


# %%
# Aggregate RTD and psychometric plot data
rtd_animal_curves = {
    ABL: {"data": [], "original": [], "condition": [], "animal_keys": [], "ilds_by_animal": []}
    for ABL in ABLS
}
rtd_continuous_animal_curves = {
    ABL: {"original": [], "condition": [], "animal_keys": [], "ilds_by_animal": []}
    for ABL in ABLS
}

psy_arrays = {
    ABL: {
        "data": np.full((len(batch_animal_pairs), len(ILD_GRID)), np.nan),
        "original": np.full((len(batch_animal_pairs), len(ILD_GRID)), np.nan),
        "condition": np.full((len(batch_animal_pairs), len(ILD_GRID)), np.nan),
        "n_trials": np.full((len(batch_animal_pairs), len(ILD_GRID)), np.nan),
    }
    for ABL in ABLS
}

animal_key_to_idx = {(result["batch_name"], result["animal"]): idx for idx, result in enumerate(all_animal_results)}

for result in all_animal_results:
    key = (result["batch_name"], result["animal"])
    animal_idx = animal_key_to_idx[key]
    for ABL in ABLS:
        if ABL in result["rtd_by_abl"]:
            rtd_animal_curves[ABL]["data"].append(result["rtd_by_abl"][ABL]["empirical"])
            rtd_animal_curves[ABL]["original"].append(result["rtd_by_abl"][ABL]["original"])
            rtd_animal_curves[ABL]["condition"].append(result["rtd_by_abl"][ABL]["condition"])
            rtd_animal_curves[ABL]["animal_keys"].append(key)
            rtd_animal_curves[ABL]["ilds_by_animal"].append(result["rtd_by_abl"][ABL]["ilds"])
            rtd_continuous_animal_curves[ABL]["original"].append(
                result["rtd_by_abl"][ABL]["original_continuous"]
            )
            rtd_continuous_animal_curves[ABL]["condition"].append(
                result["rtd_by_abl"][ABL]["condition_continuous"]
            )
            rtd_continuous_animal_curves[ABL]["animal_keys"].append(key)
            rtd_continuous_animal_curves[ABL]["ilds_by_animal"].append(result["rtd_by_abl"][ABL]["ilds"])

        if ABL not in result["psychometric_by_abl"]:
            continue
        psy = result["psychometric_by_abl"][ABL]
        for ild_idx, ILD in enumerate(ILD_GRID):
            matches = np.where(np.isclose(psy["ilds"], ILD))[0]
            if len(matches) != 1:
                continue
            src_idx = int(matches[0])
            psy_arrays[ABL]["data"][animal_idx, ild_idx] = psy["data"][src_idx]
            psy_arrays[ABL]["original"][animal_idx, ild_idx] = psy["original"][src_idx]
            psy_arrays[ABL]["condition"][animal_idx, ild_idx] = psy["condition"][src_idx]
            psy_arrays[ABL]["n_trials"][animal_idx, ild_idx] = psy["n_trials"][src_idx]

rtd_summary = {}
rtd_continuous_summary = {}
for ABL in ABLS:
    rtd_summary[ABL] = {}
    for curve_key in ["data", "original", "condition"]:
        mean_curve, sem_curve = mean_and_sem(rtd_animal_curves[ABL][curve_key])
        rtd_summary[ABL][curve_key] = {
            "mean": mean_curve,
            "sem": sem_curve,
            "area": float(np.nansum(mean_curve * np.diff(RT_BINS))),
            "n_animals": int(len(rtd_animal_curves[ABL][curve_key])),
        }
    rtd_continuous_summary[ABL] = {}
    for curve_key in ["original", "condition"]:
        mean_curve, sem_curve = mean_and_sem_continuous(rtd_continuous_animal_curves[ABL][curve_key])
        rtd_continuous_summary[ABL][curve_key] = {
            "mean": mean_curve,
            "sem": sem_curve,
            "area": float(trapezoid(mean_curve, THEORY_RT_PTS)),
            "n_animals": int(len(rtd_continuous_animal_curves[ABL][curve_key])),
        }

psy_summary = {}
for ABL in ABLS:
    psy_summary[ABL] = {}
    for curve_key in ["data", "original", "condition"]:
        mean_values, sem_values, counts = psychometric_mean_sem(psy_arrays[ABL][curve_key])
        psy_summary[ABL][curve_key] = {
            "mean": mean_values,
            "sem": sem_values,
            "counts": counts,
        }

abs_ild_rtd_animal_curves = {
    ABL: {
        abs_ild: {"data": [], "original": [], "condition": [], "animal_keys": [], "signed_ilds_by_animal": []}
        for abs_ild in ABS_ILD_GRID
    }
    for ABL in ABLS
}
abs_ild_rtd_continuous_animal_curves = {
    ABL: {
        abs_ild: {"original": [], "condition": [], "animal_keys": [], "signed_ilds_by_animal": []}
        for abs_ild in ABS_ILD_GRID
    }
    for ABL in ABLS
}

for result in all_animal_results:
    key = (result["batch_name"], result["animal"])
    condition_results = result["condition_results"]
    for ABL in ABLS:
        for abs_ild in ABS_ILD_GRID:
            signed_ilds = [
                ILD
                for ILD in [-abs_ild, abs_ild]
                if (ABL, ILD) in condition_results
            ]
            if not signed_ilds:
                continue

            for curve_key, condition_key in [
                ("data", "empirical_rtd"),
                ("original", "original_rtd"),
                ("condition", "condition_rtd"),
            ]:
                signed_curves = [condition_results[(ABL, ILD)][condition_key] for ILD in signed_ilds]
                abs_ild_rtd_animal_curves[ABL][abs_ild][curve_key].append(
                    average_equal_weight_curves(signed_curves)
                )

            for curve_key, condition_key in [
                ("original", "original_rtd_continuous"),
                ("condition", "condition_rtd_continuous"),
            ]:
                signed_curves = [condition_results[(ABL, ILD)][condition_key] for ILD in signed_ilds]
                abs_ild_rtd_continuous_animal_curves[ABL][abs_ild][curve_key].append(
                    average_equal_weight_continuous_curves(signed_curves)
                )

            abs_ild_rtd_animal_curves[ABL][abs_ild]["animal_keys"].append(key)
            abs_ild_rtd_animal_curves[ABL][abs_ild]["signed_ilds_by_animal"].append(np.array(signed_ilds, dtype=float))
            abs_ild_rtd_continuous_animal_curves[ABL][abs_ild]["animal_keys"].append(key)
            abs_ild_rtd_continuous_animal_curves[ABL][abs_ild]["signed_ilds_by_animal"].append(np.array(signed_ilds, dtype=float))

abs_ild_rtd_summary = {
    ABL: {
        abs_ild: {}
        for abs_ild in ABS_ILD_GRID
    }
    for ABL in ABLS
}
abs_ild_rtd_continuous_summary = {
    ABL: {
        abs_ild: {}
        for abs_ild in ABS_ILD_GRID
    }
    for ABL in ABLS
}
for ABL in ABLS:
    for abs_ild in ABS_ILD_GRID:
        for curve_key in ["data", "original", "condition"]:
            mean_curve, sem_curve = mean_and_sem(abs_ild_rtd_animal_curves[ABL][abs_ild][curve_key])
            abs_ild_rtd_summary[ABL][abs_ild][curve_key] = {
                "mean": mean_curve,
                "sem": sem_curve,
                "area": float(np.nansum(mean_curve * np.diff(RT_BINS))),
                "n_animals": int(len(abs_ild_rtd_animal_curves[ABL][abs_ild][curve_key])),
            }
        for curve_key in ["original", "condition"]:
            mean_curve, sem_curve = mean_and_sem_continuous(
                abs_ild_rtd_continuous_animal_curves[ABL][abs_ild][curve_key]
            )
            abs_ild_rtd_continuous_summary[ABL][abs_ild][curve_key] = {
                "mean": mean_curve,
                "sem": sem_curve,
                "area": float(trapezoid(mean_curve, THEORY_RT_PTS)),
                "n_animals": int(len(abs_ild_rtd_continuous_animal_curves[ABL][abs_ild][curve_key])),
            }


# %%
# Metrics
summary_metrics = {
    "rtd_ise_by_abl": defaultdict(dict),
    "psychometric_rmse_by_abl": defaultdict(dict),
}

all_rtd_ise = {"original": [], "condition": []}
all_psy_errors = {"original": [], "condition": []}

for ABL in ABLS:
    data_curves = np.asarray(rtd_animal_curves[ABL]["data"], dtype=float)
    for model_key in ["original", "condition"]:
        model_curves = np.asarray(rtd_animal_curves[ABL][model_key], dtype=float)
        if data_curves.size and model_curves.size:
            ise_values = np.nansum((model_curves - data_curves) ** 2 * np.diff(RT_BINS), axis=1)
            summary_metrics["rtd_ise_by_abl"][ABL][model_key] = float(np.nanmean(ise_values))
            all_rtd_ise[model_key].extend([float(x) for x in ise_values if np.isfinite(x)])
        else:
            summary_metrics["rtd_ise_by_abl"][ABL][model_key] = np.nan

    data_psy = psy_arrays[ABL]["data"]
    for model_key in ["original", "condition"]:
        errors = psy_arrays[ABL][model_key] - data_psy
        summary_metrics["psychometric_rmse_by_abl"][ABL][model_key] = float(np.sqrt(np.nanmean(errors**2)))
        all_psy_errors[model_key].extend([float(x) for x in errors[np.isfinite(errors)]])

summary_metrics["rtd_ise_overall"] = {
    key: float(np.nanmean(values)) for key, values in all_rtd_ise.items()
}
summary_metrics["psychometric_rmse_overall"] = {
    key: float(np.sqrt(np.nanmean(np.asarray(values, dtype=float) ** 2)))
    for key, values in all_psy_errors.items()
}

print("Summary metrics")
for key, value in summary_metrics["rtd_ise_overall"].items():
    print(f"  RTD ISE overall {key}: {value:.6f}")
for key, value in summary_metrics["psychometric_rmse_overall"].items():
    print(f"  Psychometric RMSE overall {key}: {value:.6f}")
for ABL in ABLS:
    areas = {key: rtd_summary[ABL][key]["area"] for key in ["data", "original", "condition"]}
    print(
        f"RTD mean areas ABL={ABL}: "
        f"data={areas['data']:.6f}, original={areas['original']:.6f}, condition={areas['condition']:.6f}"
    )


# %%
# Save pickle
output_data = {
    "script": str(Path(__file__).relative_to(REPO_DIR)),
    "animal_keys": batch_animal_pairs,
    "rt_bins": RT_BINS,
    "rt_bin_centers": RT_BIN_CENTERS,
    "rt_window_s": (RT_MIN, RT_MAX),
    "rt_bin_width_s": RT_BIN_WIDTH,
    "theory_dt_s": DT_THEORY,
    "theory_rt_pts": THEORY_RT_PTS,
    "ild_grid": np.array(ILD_GRID, dtype=float),
    "abs_ild_grid": np.array(ABS_ILD_GRID, dtype=float),
    "abls": np.array(ABLS, dtype=int),
    "rtd_animal_curves": rtd_animal_curves,
    "rtd_summary": rtd_summary,
    "rtd_continuous_animal_curves": rtd_continuous_animal_curves,
    "rtd_continuous_summary": rtd_continuous_summary,
    "abs_ild_rtd_animal_curves": abs_ild_rtd_animal_curves,
    "abs_ild_rtd_summary": abs_ild_rtd_summary,
    "abs_ild_rtd_continuous_animal_curves": abs_ild_rtd_continuous_animal_curves,
    "abs_ild_rtd_continuous_summary": abs_ild_rtd_continuous_summary,
    "psychometric_arrays": psy_arrays,
    "psychometric_summary": psy_summary,
    "per_animal_results": all_animal_results,
    "summary_metrics": {
        "rtd_ise_by_abl": {int(k): dict(v) for k, v in summary_metrics["rtd_ise_by_abl"].items()},
        "psychometric_rmse_by_abl": {
            int(k): dict(v) for k, v in summary_metrics["psychometric_rmse_by_abl"].items()
        },
        "rtd_ise_overall": summary_metrics["rtd_ise_overall"],
        "psychometric_rmse_overall": summary_metrics["psychometric_rmse_overall"],
    },
    "preflight": {
        "condition_cache_rows": int(len(condition_cache)),
        "upstream_result_count": int(len(result_paths)),
        "matched_animals": int(len(batch_animal_pairs)),
        "valid_data_rows_for_psychometric": int(len(valid_data)),
        "valid_and_abort_rows_for_theory": int(len(valid_and_abort_data)),
        "theory_abort_events": [int(event) for event in THEORY_ABORT_EVENTS],
        "rtd_abort_events": [int(event) for event in RTD_ABORT_EVENTS],
        "rtd_source_files": RAW_BATCH_FILE_MAP,
        "rtd_trial_pool_rows": int(len(rtd_trial_pool_data)),
        "rtd_data_rows_after_abort_truncation": int(len(rtd_data)),
        "batch_t_trunc_s": {batch: float(BATCH_T_TRUNC.get(batch, DEFAULT_T_TRUNC)) for batch in DESIRED_BATCHES},
        "upstream_fit_config_t_trunc_s": {
            f"{batch}/{animal}": float(value) for (batch, animal), value in fit_config_t_trunc.items()
        },
        "abort_truncation_counts": abort_truncation_counts,
        "led7_92_abl20_rerun_ms": {
            str(ild): float(
                condition_cache[
                    (condition_cache["batch_name"] == "LED7")
                    & (condition_cache["animal"] == 92)
                    & (condition_cache["ABL"] == 20)
                    & np.isclose(condition_cache["ILD"], ild)
                ]["t_E_aff_ms"].iloc[0]
            )
            for ild in [-1.0, 1.0]
        },
    },
}

with OUTPUT_PKL.open("wb") as handle:
    pickle.dump(output_data, handle)
print(f"Saved plot data and metrics: {OUTPUT_PKL}")


# %%
# RTD figure
fig, axes = plt.subplots(1, len(ABLS), figsize=(12.5, 3.8), sharex=True, sharey=True)

global_y_max = 0.0
for ax, ABL in zip(axes, ABLS):
    data_mean = rtd_summary[ABL]["data"]["mean"]
    data_sem = rtd_summary[ABL]["data"]["sem"]
    if np.any(np.isfinite(data_mean + data_sem)):
        global_y_max = max(global_y_max, float(np.nanmax(data_mean + data_sem)))

    ax.plot(
        RT_BIN_CENTERS,
        data_mean,
        color=COLORS["data"],
        linewidth=2.0,
        label=LABELS["data"],
    )
    ax.fill_between(
        RT_BIN_CENTERS,
        data_mean - data_sem,
        data_mean + data_sem,
        color=COLORS["data"],
        alpha=0.13,
        linewidth=0,
    )

    for curve_key in ["original", "condition"]:
        mean_curve = rtd_continuous_summary[ABL][curve_key]["mean"]
        sem_curve = rtd_continuous_summary[ABL][curve_key]["sem"]
        if np.any(np.isfinite(mean_curve + sem_curve)):
            global_y_max = max(global_y_max, float(np.nanmax(mean_curve + sem_curve)))

        ax.plot(
            THEORY_RT_PTS,
            mean_curve,
            color=COLORS[curve_key],
            linewidth=1.8,
            label=LABELS[curve_key],
        )
        ax.fill_between(
            THEORY_RT_PTS,
            mean_curve - sem_curve,
            mean_curve + sem_curve,
            color=COLORS[curve_key],
            alpha=0.16,
            linewidth=0,
        )

    ax.axvline(0, color="0.8", linewidth=0.8, zorder=0)
    ax.set_title(f"ABL {ABL}", fontsize=13)
    ax.set_xlim(RT_MIN, RT_MAX)
    ax.set_xticks([RT_MIN, -0.5, 0, 0.5, RT_MAX])
    ax.set_xlabel("RT wrt stimulus (s)", fontsize=11)
    ax.tick_params(axis="both", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Density", fontsize=11)
if global_y_max > 0:
    for ax in axes:
        ax.set_ylim(0, global_y_max * 1.06)
axes[-1].legend(frameon=False, fontsize=9, loc="upper right")
fig.suptitle("RTD agreement after substituting condition t_E_aff (-1 to 1 s)", fontsize=15, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(RTD_OUTPUT_PNG, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved RTD figure: {RTD_OUTPUT_PNG}")


# %%
# RTD by ABL and absolute ILD figure
def save_abs_ild_rtd_grid(output_png, xlim, xticks, title_suffix):
    fig, axes = plt.subplots(len(ABLS), len(ABS_ILD_GRID), figsize=(16.5, 8.4), sharex=True, sharey=True)

    global_y_max = 0.0
    for ABL in ABLS:
        for abs_ild in ABS_ILD_GRID:
            data_mean = abs_ild_rtd_summary[ABL][abs_ild]["data"]["mean"]
            data_sem = abs_ild_rtd_summary[ABL][abs_ild]["data"]["sem"]
            if np.any(np.isfinite(data_mean + data_sem)):
                global_y_max = max(global_y_max, float(np.nanmax(data_mean + data_sem)))
            for curve_key in ["original", "condition"]:
                mean_curve = abs_ild_rtd_continuous_summary[ABL][abs_ild][curve_key]["mean"]
                sem_curve = abs_ild_rtd_continuous_summary[ABL][abs_ild][curve_key]["sem"]
                if np.any(np.isfinite(mean_curve + sem_curve)):
                    global_y_max = max(global_y_max, float(np.nanmax(mean_curve + sem_curve)))

    for row_idx, ABL in enumerate(ABLS):
        for col_idx, abs_ild in enumerate(ABS_ILD_GRID):
            ax = axes[row_idx, col_idx]
            data_mean = abs_ild_rtd_summary[ABL][abs_ild]["data"]["mean"]
            data_sem = abs_ild_rtd_summary[ABL][abs_ild]["data"]["sem"]
            ax.plot(
                RT_BIN_CENTERS,
                data_mean,
                color=COLORS["data"],
                linewidth=1.8,
                label=LABELS["data"],
            )
            ax.fill_between(
                RT_BIN_CENTERS,
                data_mean - data_sem,
                data_mean + data_sem,
                color=COLORS["data"],
                alpha=0.12,
                linewidth=0,
            )

            for curve_key in ["original", "condition"]:
                mean_curve = abs_ild_rtd_continuous_summary[ABL][abs_ild][curve_key]["mean"]
                sem_curve = abs_ild_rtd_continuous_summary[ABL][abs_ild][curve_key]["sem"]
                ax.plot(
                    THEORY_RT_PTS,
                    mean_curve,
                    color=COLORS[curve_key],
                    linewidth=1.5,
                    label=LABELS[curve_key],
                )
                ax.fill_between(
                    THEORY_RT_PTS,
                    mean_curve - sem_curve,
                    mean_curve + sem_curve,
                    color=COLORS[curve_key],
                    alpha=0.15,
                    linewidth=0,
                )

            ax.axvline(0, color="0.8", linewidth=0.7, zorder=0)
            if row_idx == 0:
                ax.set_title(f"|ILD| = {abs_ild:g}", fontsize=12)
            if col_idx == 0:
                ax.set_ylabel(f"ABL {ABL}\nDensity", fontsize=11)
            if row_idx == len(ABLS) - 1:
                ax.set_xlabel("RT wrt stimulus (s)", fontsize=10)

            ax.set_xlim(*xlim)
            ax.set_xticks(xticks)
            ax.tick_params(axis="both", labelsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if global_y_max > 0:
                ax.set_ylim(0, global_y_max * 1.06)

    axes[0, -1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(
        f"RTD agreement by ABL and |ILD| after substituting condition t_E_aff ({title_suffix})",
        fontsize=15,
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved RTD by |ILD| figure: {output_png}")


save_abs_ild_rtd_grid(
    RTD_ABS_ILD_OUTPUT_PNG,
    (RT_MIN, RT_MAX),
    [RT_MIN, 0, RT_MAX],
    "-1 to 1 s",
)
save_abs_ild_rtd_grid(
    RTD_ABS_ILD_ZOOM_OUTPUT_PNG,
    (-0.6, 0.6),
    [-0.6, 0, 0.6],
    "-0.6 to 0.6 s",
)


# %%
# Psychometric figure
fig, axes = plt.subplots(1, len(ABLS), figsize=(12.5, 3.8), sharex=True, sharey=True)

for ax, ABL in zip(axes, ABLS):
    data_mean = psy_summary[ABL]["data"]["mean"]
    data_sem = psy_summary[ABL]["data"]["sem"]
    data_counts = psy_summary[ABL]["data"]["counts"]
    valid_data_pts = np.isfinite(data_mean) & (data_counts > 0)

    ax.errorbar(
        np.asarray(ILD_GRID)[valid_data_pts],
        data_mean[valid_data_pts],
        yerr=data_sem[valid_data_pts],
        fmt="o",
        color=COLORS["data"],
        markersize=4,
        linewidth=1.2,
        capsize=2,
        label=LABELS["data"],
    )

    for curve_key in ["original", "condition"]:
        mean_values = psy_summary[ABL][curve_key]["mean"]
        counts = psy_summary[ABL][curve_key]["counts"]
        valid_pts = np.isfinite(mean_values) & (counts > 0)
        ax.plot(
            np.asarray(ILD_GRID)[valid_pts],
            mean_values[valid_pts],
            "-o",
            color=COLORS[curve_key],
            linewidth=1.6,
            markersize=3.5,
            label=LABELS[curve_key],
        )

    ax.axhline(0.5, color="0.75", linewidth=0.8, zorder=0)
    ax.axvline(0, color="0.85", linewidth=0.8, zorder=0)
    ax.set_title(f"ABL {ABL}", fontsize=13)
    ax.set_xlim(-17, 17)
    ax.set_xticks(ILD_GRID)
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_xlabel("ILD", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("P(right)", fontsize=11)
axes[0].set_ylim(-0.02, 1.02)
axes[-1].legend(frameon=False, fontsize=9, loc="lower right")
fig.suptitle("Psychometric agreement after substituting condition t_E_aff", fontsize=15, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(PSY_OUTPUT_PNG, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved psychometric figure: {PSY_OUTPUT_PNG}")

print("Done.")

# %%
