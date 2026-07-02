# %%
"""
Single-animal diagnostics for the NumPyro SVI NPL+alpha condition-delay fit.

This plots RTDs by ABL and |ILD|, plus psychometric curves by ABL, for one
animal. The data RTD follows the later diagnostic convention: valid trials plus
abort_event 3 and 4, with batch-specific early-abort truncation applied.
"""

# %%
# =============================================================================
# Editable parameters
# =============================================================================
from pathlib import Path
import os
import pickle
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

BATCH_NAME = os.environ.get("NUMPYRO_SVI_BATCH", "LED7")
ANIMAL = int(os.environ.get("NUMPYRO_SVI_ANIMAL", "92"))
SVI_LABEL = os.environ.get("NUMPYRO_SVI_DIAG_LABEL", "main_fullrank")

ABLS = [20, 40, 60]
ABS_ILD_GRID = [1.0, 2.0, 4.0, 8.0, 16.0]

RT_MIN = float(os.environ.get("NUMPYRO_SVI_DIAG_RT_MIN", "-1.0"))
RT_MAX = float(os.environ.get("NUMPYRO_SVI_DIAG_RT_MAX", "1.0"))
RT_BIN_WIDTH = float(os.environ.get("NUMPYRO_SVI_DIAG_RT_BIN_WIDTH", "0.02"))
DT_THEORY = float(os.environ.get("NUMPYRO_SVI_DIAG_DT_THEORY", "0.001"))
N_THEORY = int(os.environ.get("NUMPYRO_SVI_DIAG_N_THEORY", "1000"))
RNG_SEED = int(os.environ.get("NUMPYRO_SVI_DIAG_SEED", "0"))
K_MAX = int(os.environ.get("K_MAX", "10"))

SELECTED_RTD_ABLS_TEXT = os.environ.get("NUMPYRO_SVI_SELECTED_RTD_ABLS", "")
SELECTED_RTD_ABLS = [
    int(value.strip())
    for value in SELECTED_RTD_ABLS_TEXT.split(",")
    if value.strip()
]
SELECTED_RTD_ILD_TEXT = os.environ.get("NUMPYRO_SVI_SELECTED_RTD_ILD", "")
SELECTED_RTD_ILD = float(SELECTED_RTD_ILD_TEXT) if SELECTED_RTD_ILD_TEXT.strip() else None

THEORY_ABORT_EVENTS = [3]
RTD_ABORT_EVENTS = [3, 4]
BATCH_T_TRUNC = {"LED34_even": 0.15}
DEFAULT_T_TRUNC = 0.3

RAW_BATCH_FILE_MAP = {
    "SD": "outExp.csv",
    "LED34": "outExp.csv",
    "LED6": "outExp.csv",
    "LED8": "outLED8.csv",
    "LED7": "out_LED.csv",
    "LED34_even": "outUni.csv",
}

OUTPUT_ROOT = Path(
    os.environ.get(
        "NUMPYRO_SVI_OUTPUT_ROOT",
        str(SCRIPT_DIR / "numpyro_svi_npl_alpha_condition_delay_single_animal_outputs"),
    )
).expanduser()
OUTPUT_DIR = (
    OUTPUT_ROOT
    / f"{BATCH_NAME}_{ANIMAL}"
)
DIAGNOSTIC_DIR = OUTPUT_DIR / "diagnostics"
DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)

POSTERIOR_NPZ = Path(
    os.environ.get(
        "NUMPYRO_SVI_POSTERIOR_NPZ",
        str(OUTPUT_DIR / f"{SVI_LABEL}_posterior_samples.npz"),
    )
)
BATCH_CSV = REPO_DIR / "raw_data" / "batch_csvs" / f"batch_{BATCH_NAME}_valid_and_aborts.csv"
ABORT_RESULT_PKL = REPO_DIR / "aborts_ipl_npl_time_fit_results" / f"results_{BATCH_NAME}_animal_{ANIMAL}.pkl"
SAVED_CONDITION_TABLE = OUTPUT_DIR / "condition_table.csv"

RTD_FULL_PNG = DIAGNOSTIC_DIR / f"{SVI_LABEL}_rtd_by_abl_abs_ild.png"
RTD_ZOOM_PNG = DIAGNOSTIC_DIR / f"{SVI_LABEL}_rtd_by_abl_abs_ild_zoom.png"
PSYCHOMETRIC_PNG = DIAGNOSTIC_DIR / f"{SVI_LABEL}_psychometric_by_abl.png"

RTD_SELECTED_PNG = None
if SELECTED_RTD_ABLS and SELECTED_RTD_ILD is not None:
    selected_abl_label = "_".join(str(abl) for abl in SELECTED_RTD_ABLS)
    selected_ild_label = f"{SELECTED_RTD_ILD:+g}".replace("+", "p").replace("-", "m").replace(".", "p")
    RTD_SELECTED_PNG = DIAGNOSTIC_DIR / f"{SVI_LABEL}_rtd_selected_abl{selected_abl_label}_ild{selected_ild_label}.png"


# %%
# =============================================================================
# Imports and plotting setup
# =============================================================================
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.special as scipy_special
from scipy.integrate import trapezoid

try:
    import scipy.special._ufuncs as scipy_ufuncs

    sys.modules.setdefault("scipy.special._special_ufuncs", scipy_ufuncs)
except Exception:
    pass

sys.path.insert(0, str(SCRIPT_DIR))
from time_vary_norm_utils import M, phi, rho_A_t_VEC_fn

RT_BINS = np.arange(RT_MIN, RT_MAX + RT_BIN_WIDTH, RT_BIN_WIDTH)
RT_BIN_CENTERS = 0.5 * (RT_BINS[:-1] + RT_BINS[1:])
T_PTS = np.arange(-2.0, 2.0, DT_THEORY)
THEORY_RT_MASK = (T_PTS >= RT_MIN) & (T_PTS <= RT_MAX)
THEORY_RT_PTS = T_PTS[THEORY_RT_MASK]

COLORS = {
    "data": "black",
    "model": "#0072B2",
}


# %%
# =============================================================================
# Model density helpers
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
# =============================================================================
# Data and summary helpers
# =============================================================================
def ensure_choice_column(df):
    if "choice" not in df.columns:
        if "response_poke" not in df.columns:
            raise KeyError("Need either `choice` or `response_poke` to compute psychometric data.")
        df = df.copy()
        df["choice"] = df["response_poke"].map({3: 1, 2: -1})
    return df


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
    raw_rts = np.asarray(raw_rts, dtype=float)
    raw_rts = raw_rts[np.isfinite(raw_rts)]
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


def weighted_average_continuous(curves, weights=None):
    clean_curves = []
    clean_weights = []
    if weights is None:
        weights = np.ones(len(curves), dtype=float)
    for curve, weight in zip(curves, weights):
        curve = np.asarray(curve, dtype=float)
        if np.any(np.isfinite(curve)) and np.isfinite(weight) and weight > 0:
            clean_curves.append(curve)
            clean_weights.append(float(weight))
    if not clean_curves:
        return np.full(len(THEORY_RT_PTS), np.nan)
    mean_curve = np.average(np.asarray(clean_curves), axis=0, weights=np.asarray(clean_weights))
    return normalize_continuous_density(THEORY_RT_PTS, mean_curve)


def load_raw_rtd_source_for_batch(batch_name):
    if batch_name not in RAW_BATCH_FILE_MAP:
        raise ValueError(f"No raw-data mapping for batch {batch_name}")
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
    exp_df_batch = ensure_choice_column(exp_df_batch)
    return exp_df_batch


def calculate_truncated_theoretical_curves(df_valid_and_aborts, N_theory, t_pts, t_A_aff, V_A, theta_A, T_trunc):
    if len(df_valid_and_aborts) == 0:
        raise RuntimeError("No valid/abort rows available for proactive truncation samples.")
    n_sample = int(N_theory)
    t_stim_samples = df_valid_and_aborts["intended_fix"].sample(
        n_sample,
        replace=True,
        random_state=RNG_SEED,
    ).to_numpy(dtype=float)

    trunc_cdf = cum_A_t_vec(np.array([T_trunc - t_A_aff]), V_A, theta_A)[0]
    trunc_survival = max(1 - trunc_cdf, 1e-12)

    P_A_samples = np.zeros((n_sample, len(t_pts)))
    C_A_samples = np.zeros((n_sample, len(t_pts)))
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


def finite_mean(values, name):
    values = np.asarray(values, dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise RuntimeError(f"Posterior samples for {name} contain no finite values.")
    return float(np.mean(finite_values))


# %%
# =============================================================================
# Load posterior, data, abort parameters, and condition map
# =============================================================================
print(f"Batch/animal: {BATCH_NAME}/{ANIMAL}")
print(f"SVI posterior samples: {POSTERIOR_NPZ}")
print(f"Processed valid/abort batch CSV: {BATCH_CSV}")
print(f"Abort params: {ABORT_RESULT_PKL}")
print(f"Output diagnostics: {DIAGNOSTIC_DIR}")
print(f"RTD data bin width: {RT_BIN_WIDTH * 1e3:.0f} ms")
if RTD_SELECTED_PNG is not None:
    print("Selected RTD data bin width: 5 ms")
print(f"Model RTD grid spacing: {DT_THEORY * 1e3:.0f} ms")
print(f"RTD data abort events: {RTD_ABORT_EVENTS}; model/theory abort events: {THEORY_ABORT_EVENTS}")

for required_path in [POSTERIOR_NPZ, BATCH_CSV, ABORT_RESULT_PKL]:
    if not required_path.exists():
        raise FileNotFoundError(required_path)

posterior_npz = np.load(POSTERIOR_NPZ)
required_posterior_keys = ["rate_lambda", "T_0", "theta_E", "w", "del_go", "rate_norm_l", "alpha", "t_E_aff"]
missing_keys = [key for key in required_posterior_keys if key not in posterior_npz.files]
if missing_keys:
    raise KeyError(f"Missing posterior sample arrays in {POSTERIOR_NPZ}: {missing_keys}")

tied_params = {
    "rate_lambda": finite_mean(posterior_npz["rate_lambda"], "rate_lambda"),
    "T_0": finite_mean(posterior_npz["T_0"], "T_0"),
    "theta_E": finite_mean(posterior_npz["theta_E"], "theta_E"),
    "w": finite_mean(posterior_npz["w"], "w"),
    "del_go": finite_mean(posterior_npz["del_go"], "del_go"),
    "rate_norm_l": finite_mean(posterior_npz["rate_norm_l"], "rate_norm_l"),
    "alpha": finite_mean(posterior_npz["alpha"], "alpha"),
}
t_E_aff_mean = np.nanmean(np.asarray(posterior_npz["t_E_aff"], dtype=float), axis=0)
if not np.all(np.isfinite(t_E_aff_mean)):
    raise RuntimeError("Some t_E_aff posterior means are non-finite.")

print("\nSVI posterior means used for diagnostics:")
for name, value in tied_params.items():
    if name in {"T_0", "del_go"}:
        print(f"  {name:<12} = {1e3 * value:.3f} ms")
    else:
        print(f"  {name:<12} = {value:.6g}")

batch_df = ensure_choice_column(pd.read_csv(BATCH_CSV))
batch_df["animal"] = batch_df["animal"].astype(int)
batch_df["ABL"] = batch_df["ABL"].astype(float)
batch_df["ILD"] = batch_df["ILD"].astype(float)

valid_df = batch_df[
    (batch_df["animal"] == ANIMAL)
    & batch_df["success"].isin([1, -1])
    & (batch_df["RTwrtStim"] < 1)
    & batch_df["ABL"].isin(ABLS)
].copy()
valid_df = valid_df.dropna(subset=["TotalFixTime", "intended_fix", "ABL", "ILD", "choice", "RTwrtStim"])
if len(valid_df) == 0:
    raise RuntimeError(f"No valid RT<1 trials for {BATCH_NAME}/{ANIMAL}.")
valid_df["choice"] = valid_df["choice"].astype(int)

reconstructed_condition_table = (
    valid_df[["ABL", "ILD"]]
    .drop_duplicates()
    .sort_values(["ABL", "ILD"])
    .reset_index(drop=True)
)
reconstructed_condition_table["condition_id"] = np.arange(len(reconstructed_condition_table), dtype=int)

if SAVED_CONDITION_TABLE.exists():
    condition_table = pd.read_csv(SAVED_CONDITION_TABLE)
    condition_table["ABL"] = condition_table["ABL"].astype(float)
    condition_table["ILD"] = condition_table["ILD"].astype(float)
    condition_table["condition_id"] = condition_table["condition_id"].astype(int)
    condition_table = condition_table.sort_values("condition_id").reset_index(drop=True)

    saved_pairs = condition_table[["ABL", "ILD"]].reset_index(drop=True)
    reconstructed_pairs = reconstructed_condition_table[["ABL", "ILD"]].reset_index(drop=True)
    if len(saved_pairs) != len(reconstructed_pairs) or not np.allclose(saved_pairs.to_numpy(), reconstructed_pairs.to_numpy()):
        raise RuntimeError(
            "Saved condition_table.csv does not match the condition order reconstructed from data. "
            "Refusing to map t_E_aff samples to conditions."
        )
    print(f"Using saved condition table: {SAVED_CONDITION_TABLE}")
else:
    condition_table = reconstructed_condition_table
    print("No saved condition_table.csv found; using reconstructed sorted ABL/ILD condition order.")

if len(condition_table) != t_E_aff_mean.shape[0]:
    raise RuntimeError(
        f"Condition count {len(condition_table)} does not match t_E_aff posterior width {t_E_aff_mean.shape[0]}"
    )

valid_df = valid_df.merge(condition_table[["ABL", "ILD", "condition_id"]], on=["ABL", "ILD"], how="left")
if valid_df["condition_id"].isna().any():
    raise RuntimeError("Failed to assign condition IDs to all valid trials.")

print(f"\nValid psychometric/fitting trials: {len(valid_df)}")
print(f"Observed conditions: {len(condition_table)}")
print(condition_table[["condition_id", "ABL", "ILD"]].to_string(index=False))

with ABORT_RESULT_PKL.open("rb") as f:
    abort_saved = pickle.load(f)
abort_results = abort_saved["vbmc_aborts_results"]
abort_params = {
    "V_A": float(np.mean(abort_results["V_A_samples"])),
    "theta_A": float(np.mean(abort_results["theta_A_samples"])),
    "t_A_aff": float(np.mean(abort_results["t_A_aff_samp"])),
}
print("\nAbort/proactive posterior means:")
print(f"  V_A      = {abort_params['V_A']:.6g}")
print(f"  theta_A  = {abort_params['theta_A']:.6g}")
print(f"  t_A_aff  = {1e3 * abort_params['t_A_aff']:.3f} ms")

theory_df = batch_df[
    (batch_df["animal"] == ANIMAL)
    & (batch_df["success"].isin([1, -1]) | batch_df["abort_event"].isin(THEORY_ABORT_EVENTS))
].copy()
theory_df = theory_df.dropna(subset=["intended_fix"])
if len(theory_df) == 0:
    raise RuntimeError("No theory rows after valid/abort filtering.")

raw_rtd_source = load_raw_rtd_source_for_batch(BATCH_NAME)
raw_rtd_source["animal"] = raw_rtd_source["animal"].astype(int)
raw_rtd_source["ABL"] = raw_rtd_source["ABL"].astype(float)
raw_rtd_source["ILD"] = raw_rtd_source["ILD"].astype(float)
raw_rtd_source["T_trunc"] = raw_rtd_source["batch_name"].map(BATCH_T_TRUNC).fillna(DEFAULT_T_TRUNC)

rtd_trial_pool = raw_rtd_source[
    (raw_rtd_source["animal"] == ANIMAL)
    & (raw_rtd_source["success"].isin([1, -1]) | raw_rtd_source["abort_event"].isin(RTD_ABORT_EVENTS))
].copy()
rtd_data = rtd_trial_pool[
    rtd_trial_pool["ABL"].isin(ABLS)
    & rtd_trial_pool["RTwrtStim"].between(RT_MIN, RT_MAX, inclusive="both")
    & (
        rtd_trial_pool["success"].isin([1, -1])
        | rtd_trial_pool["TotalFixTime"].ge(rtd_trial_pool["T_trunc"])
    )
].copy()

T_trunc = BATCH_T_TRUNC.get(BATCH_NAME, DEFAULT_T_TRUNC)
abort_pool = rtd_trial_pool[rtd_trial_pool["abort_event"].isin(RTD_ABORT_EVENTS)]
kept_abort_pool = abort_pool[
    abort_pool["TotalFixTime"].ge(T_trunc)
    & abort_pool["RTwrtStim"].between(RT_MIN, RT_MAX, inclusive="both")
    & abort_pool["ABL"].isin(ABLS)
]
print("\nRTD data filtering:")
print(f"  T_trunc = {T_trunc:.3f} s")
print(f"  trial pool rows = {len(rtd_trial_pool)}")
print(f"  plotted RTD rows = {len(rtd_data)}")
print(f"  abort rows by event before truncation: {abort_pool['abort_event'].value_counts().sort_index().to_dict()}")
print(f"  abort rows by event after truncation/window: {kept_abort_pool['abort_event'].value_counts().sort_index().to_dict()}")


# %%
# =============================================================================
# Compute condition-wise model/data summaries
# =============================================================================
P_A_mean, C_A_mean, t_stim_samples = calculate_truncated_theoretical_curves(
    theory_df,
    N_THEORY,
    T_PTS,
    abort_params["t_A_aff"],
    abort_params["V_A"],
    abort_params["theta_A"],
    T_trunc,
)
print(f"\nSampled {len(t_stim_samples)} intended-fix values for proactive averaging.")

condition_results = {}
for condition in condition_table.itertuples(index=False):
    ABL = int(condition.ABL)
    ILD = float(condition.ILD)
    condition_id = int(condition.condition_id)

    valid_condition_df = valid_df[(valid_df["ABL"] == ABL) & np.isclose(valid_df["ILD"], ILD)]
    rtd_condition_df = rtd_data[(rtd_data["ABL"] == ABL) & np.isclose(rtd_data["ILD"], ILD)]

    data_right_prob = float(np.mean(valid_condition_df["choice"] == 1)) if len(valid_condition_df) else np.nan
    data_right_se = (
        float(np.sqrt(data_right_prob * (1 - data_right_prob) / len(valid_condition_df)))
        if len(valid_condition_df) and np.isfinite(data_right_prob)
        else np.nan
    )

    t_E_aff = float(t_E_aff_mean[condition_id])
    t_pts, model_up, model_down, trunc_factor = get_theoretical_up_down(
        P_A_mean,
        C_A_mean,
        t_stim_samples,
        abort_params,
        tied_params,
        ABL,
        ILD,
        BATCH_NAME,
        t_E_aff,
    )

    display_mask = (t_pts >= RT_MIN) & (t_pts <= RT_MAX)
    valid_rt_mask = (t_pts >= 0) & (t_pts <= 1)
    model_display = model_up[display_mask] + model_down[display_mask]
    model_rtd_continuous = normalize_continuous_density(t_pts[display_mask], model_display)
    model_rtd_binned = theoretical_rtd_to_bin_density(t_pts[display_mask], model_display)

    model_up_area = trapezoid(model_up[valid_rt_mask], t_pts[valid_rt_mask])
    model_down_area = trapezoid(model_down[valid_rt_mask], t_pts[valid_rt_mask])
    model_denom = model_up_area + model_down_area
    model_right_prob = float(model_up_area / model_denom) if model_denom > 0 else np.nan

    condition_results[(ABL, ILD)] = {
        "condition_id": condition_id,
        "t_E_aff_s": t_E_aff,
        "n_valid_trials": int(len(valid_condition_df)),
        "n_rtd_trials": int(len(rtd_condition_df)),
        "data_rtd": raw_rts_to_hist(rtd_condition_df["RTwrtStim"].to_numpy(dtype=float)),
        "data_right_prob": data_right_prob,
        "data_right_se": data_right_se,
        "model_rtd_binned": model_rtd_binned,
        "model_rtd_continuous": model_rtd_continuous,
        "model_right_prob": model_right_prob,
        "model_trunc_factor": trunc_factor,
    }


# %%
# =============================================================================
# Collapse signed ILDs for RTD panels and signed ILDs for psychometric panels
# =============================================================================
abs_ild_rtd = {
    ABL: {
        abs_ild: {}
        for abs_ild in ABS_ILD_GRID
    }
    for ABL in ABLS
}

for ABL in ABLS:
    for abs_ild in ABS_ILD_GRID:
        signed_ilds = [
            signed_ild
            for signed_ild in [-abs_ild, abs_ild]
            if (ABL, signed_ild) in condition_results
        ]
        if not signed_ilds:
            continue

        pooled_data_df = rtd_data[
            (rtd_data["ABL"] == ABL)
            & np.isclose(np.abs(rtd_data["ILD"].to_numpy(dtype=float)), abs_ild)
        ]
        data_rtd = raw_rts_to_hist(pooled_data_df["RTwrtStim"].to_numpy(dtype=float))

        model_curves = [condition_results[(ABL, signed_ild)]["model_rtd_continuous"] for signed_ild in signed_ilds]
        model_weights = np.array(
            [condition_results[(ABL, signed_ild)]["n_rtd_trials"] for signed_ild in signed_ilds],
            dtype=float,
        )
        if not np.any(model_weights > 0):
            model_weights = np.ones(len(signed_ilds), dtype=float)
        model_rtd = weighted_average_continuous(model_curves, model_weights)

        abs_ild_rtd[ABL][abs_ild] = {
            "signed_ilds": np.array(signed_ilds, dtype=float),
            "data_rtd": data_rtd,
            "model_rtd": model_rtd,
            "n_rtd_trials": int(len(pooled_data_df)),
            "model_weights": model_weights,
            "data_area": float(np.nansum(data_rtd * np.diff(RT_BINS))),
            "model_area": float(trapezoid(model_rtd, THEORY_RT_PTS)),
        }

psychometric_by_abl = {}
for ABL in ABLS:
    ilds = sorted(ILD for abl, ILD in condition_results if abl == ABL)
    psychometric_by_abl[ABL] = {
        "ilds": np.array(ilds, dtype=float),
        "data": np.array([condition_results[(ABL, ILD)]["data_right_prob"] for ILD in ilds], dtype=float),
        "data_se": np.array([condition_results[(ABL, ILD)]["data_right_se"] for ILD in ilds], dtype=float),
        "model": np.array([condition_results[(ABL, ILD)]["model_right_prob"] for ILD in ilds], dtype=float),
        "n_trials": np.array([condition_results[(ABL, ILD)]["n_valid_trials"] for ILD in ilds], dtype=int),
    }

print("\nRTD area check after normalization:")
for ABL in ABLS:
    area_text = []
    for abs_ild in ABS_ILD_GRID:
        entry = abs_ild_rtd[ABL][abs_ild]
        if not entry:
            area_text.append(f"|ILD|{abs_ild:g}: missing")
            continue
        area_text.append(
            f"|ILD|{abs_ild:g}: data={entry['data_area']:.3f}, model={entry['model_area']:.3f}, n={entry['n_rtd_trials']}"
        )
    print(f"  ABL {ABL}: " + "; ".join(area_text))


# %%
# =============================================================================
# Plot RTD grids
# =============================================================================
def save_rtd_grid(output_png, xlim, xticks, title_suffix):
    fig, axes = plt.subplots(len(ABLS), len(ABS_ILD_GRID), figsize=(16.5, 8.4), sharex=True, sharey=True)

    global_y_max = 0.0
    for ABL in ABLS:
        for abs_ild in ABS_ILD_GRID:
            entry = abs_ild_rtd[ABL][abs_ild]
            if not entry:
                continue
            data_mask = (RT_BIN_CENTERS >= xlim[0]) & (RT_BIN_CENTERS <= xlim[1])
            model_mask = (THEORY_RT_PTS >= xlim[0]) & (THEORY_RT_PTS <= xlim[1])
            if np.any(np.isfinite(entry["data_rtd"][data_mask])):
                global_y_max = max(global_y_max, float(np.nanmax(entry["data_rtd"][data_mask])))
            if np.any(np.isfinite(entry["model_rtd"][model_mask])):
                global_y_max = max(global_y_max, float(np.nanmax(entry["model_rtd"][model_mask])))

    for row_idx, ABL in enumerate(ABLS):
        for col_idx, abs_ild in enumerate(ABS_ILD_GRID):
            ax = axes[row_idx, col_idx]
            entry = abs_ild_rtd[ABL][abs_ild]
            if entry:
                ax.plot(
                    RT_BIN_CENTERS,
                    entry["data_rtd"],
                    color=COLORS["data"],
                    linewidth=1.8,
                    label="Data",
                )
                ax.plot(
                    THEORY_RT_PTS,
                    entry["model_rtd"],
                    color=COLORS["model"],
                    linewidth=1.7,
                    label="SVI posterior mean",
                )
                ax.text(
                    0.04,
                    0.90,
                    f"n={entry['n_rtd_trials']}",
                    transform=ax.transAxes,
                    fontsize=8,
                    color="0.35",
                )
            else:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", fontsize=9)

            ax.axvline(0, color="0.82", linewidth=0.7, zorder=0)
            if row_idx == 0:
                ax.set_title(f"|ILD| = {abs_ild:g}", fontsize=12)
            if col_idx == 0:
                ax.set_ylabel(f"ABL {ABL}\nDensity", fontsize=11)
            if row_idx == len(ABLS) - 1:
                ax.set_xlabel("RT wrt stimulus (s)", fontsize=10)

            ax.set_xlim(*xlim)
            ax.set_xticks(xticks)
            if global_y_max > 0:
                ax.set_ylim(0, global_y_max * 1.08)
            ax.tick_params(axis="both", labelsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    axes[0, -1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(
        f"{BATCH_NAME}/{ANIMAL} SVI RTDs by ABL and |ILD| ({title_suffix})",
        fontsize=15,
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    print(f"Saved RTD figure: {output_png}")


save_rtd_grid(
    RTD_FULL_PNG,
    (RT_MIN, RT_MAX),
    [RT_MIN, 0, RT_MAX],
    f"{RT_MIN:g} to {RT_MAX:g} s",
)
save_rtd_grid(
    RTD_ZOOM_PNG,
    (-0.6, 0.6),
    [-0.6, 0, 0.6],
    "-0.6 to 0.6 s",
)


# %%
# =============================================================================
# Optional selected signed-ILD RTD panels
# =============================================================================
if RTD_SELECTED_PNG is not None:
    selected_data_bin_width = 0.005
    selected_top_xlim = (-0.2, 0.6)
    selected_trunc_xlim = (0.0, 0.3)
    selected_reactive_xlim = (0.0, 1.0)
    selected_reactive_plot_xlim = (0.0, 0.3)
    selected_full_bins = np.arange(
        RT_MIN,
        RT_MAX + 0.5 * selected_data_bin_width,
        selected_data_bin_width,
    )
    selected_full_bin_centers = 0.5 * (selected_full_bins[:-1] + selected_full_bins[1:])
    selected_trunc_bins = np.arange(
        selected_trunc_xlim[0],
        selected_trunc_xlim[1] + 0.5 * selected_data_bin_width,
        selected_data_bin_width,
    )
    selected_trunc_bin_centers = 0.5 * (selected_trunc_bins[:-1] + selected_trunc_bins[1:])
    selected_reactive_bins = np.arange(
        selected_reactive_xlim[0],
        selected_reactive_xlim[1] + 0.5 * selected_data_bin_width,
        selected_data_bin_width,
    )
    selected_reactive_bin_centers = 0.5 * (selected_reactive_bins[:-1] + selected_reactive_bins[1:])
    selected_trunc_theory_mask = (
        (THEORY_RT_PTS >= selected_trunc_xlim[0])
        & (THEORY_RT_PTS <= selected_trunc_xlim[1])
    )
    selected_reactive_theory_mask = (
        (THEORY_RT_PTS >= selected_reactive_xlim[0])
        & (THEORY_RT_PTS <= selected_reactive_xlim[1])
    )

    selected_entries = []
    for ABL in SELECTED_RTD_ABLS:
        key = (int(ABL), float(SELECTED_RTD_ILD))
        if key not in condition_results:
            print(f"Skipping selected RTD panel for missing condition ABL={ABL}, ILD={SELECTED_RTD_ILD:g}.")
            continue

        selected_data_df = rtd_data[
            (rtd_data["ABL"] == ABL)
            & np.isclose(rtd_data["ILD"].to_numpy(dtype=float), SELECTED_RTD_ILD)
        ]
        selected_raw_rts = selected_data_df["RTwrtStim"].to_numpy(dtype=float)
        selected_raw_rts = selected_raw_rts[np.isfinite(selected_raw_rts)]
        if len(selected_raw_rts) > 5:
            data_rtd, _ = np.histogram(selected_raw_rts, bins=selected_full_bins, density=True)
            data_area = np.nansum(data_rtd * np.diff(selected_full_bins))
            if np.isfinite(data_area) and data_area > 0:
                data_rtd = data_rtd / data_area
            else:
                data_rtd = np.full(len(selected_full_bin_centers), np.nan)
        else:
            data_rtd = np.full(len(selected_full_bin_centers), np.nan)
        model_rtd = condition_results[key]["model_rtd_continuous"]

        truncated_rts = selected_raw_rts[
            (selected_raw_rts >= selected_trunc_xlim[0])
            & (selected_raw_rts <= selected_trunc_xlim[1])
        ]
        truncated_data_retained_area = (
            len(truncated_rts) / len(selected_raw_rts)
            if len(selected_raw_rts)
            else np.nan
        )
        if len(truncated_rts) > 5:
            truncated_data_rtd, _ = np.histogram(truncated_rts, bins=selected_trunc_bins, density=True)
            truncated_data_area = np.nansum(truncated_data_rtd * np.diff(selected_trunc_bins))
            if np.isfinite(truncated_data_area) and truncated_data_area > 0:
                truncated_data_rtd = truncated_data_rtd / truncated_data_area
            else:
                truncated_data_rtd = np.full(len(selected_trunc_bin_centers), np.nan)
        else:
            truncated_data_rtd = np.full(len(selected_trunc_bin_centers), np.nan)

        truncated_model_t = THEORY_RT_PTS[selected_trunc_theory_mask]
        truncated_model_rtd = normalize_continuous_density(
            truncated_model_t,
            model_rtd[selected_trunc_theory_mask],
        )
        truncated_model_retained_area = trapezoid(model_rtd[selected_trunc_theory_mask], truncated_model_t)

        reactive_data_df = batch_df[
            (batch_df["animal"] == ANIMAL)
            & batch_df["success"].isin([1, -1])
            & (batch_df["ABL"] == ABL)
            & np.isclose(batch_df["ILD"].to_numpy(dtype=float), SELECTED_RTD_ILD)
            & batch_df["RTwrtStim"].between(selected_reactive_xlim[0], selected_reactive_xlim[1], inclusive="both")
        ].copy()
        reactive_raw_rts = reactive_data_df["RTwrtStim"].to_numpy(dtype=float)
        reactive_raw_rts = reactive_raw_rts[np.isfinite(reactive_raw_rts)]
        if len(reactive_raw_rts) > 5:
            reactive_data_rtd, _ = np.histogram(reactive_raw_rts, bins=selected_reactive_bins, density=True)
            reactive_data_area = np.nansum(reactive_data_rtd * np.diff(selected_reactive_bins))
            if np.isfinite(reactive_data_area) and reactive_data_area > 0:
                reactive_data_rtd = reactive_data_rtd / reactive_data_area
            else:
                reactive_data_rtd = np.full(len(selected_reactive_bin_centers), np.nan)
        else:
            reactive_data_rtd = np.full(len(selected_reactive_bin_centers), np.nan)

        reactive_model_t = THEORY_RT_PTS[selected_reactive_theory_mask]
        reactive_zeros = np.zeros_like(reactive_model_t)
        t_E_aff = float(condition_results[key]["t_E_aff_s"])
        reactive_model_raw = (
            up_or_down_alpha_pa_ca_vec(
                reactive_model_t,
                1,
                reactive_zeros,
                reactive_zeros,
                ABL,
                SELECTED_RTD_ILD,
                tied_params,
                (tied_params["w"] - 0.5) * 2 * tied_params["theta_E"],
                t_E_aff,
            )
            + up_or_down_alpha_pa_ca_vec(
                reactive_model_t,
                -1,
                reactive_zeros,
                reactive_zeros,
                ABL,
                SELECTED_RTD_ILD,
                tied_params,
                (tied_params["w"] - 0.5) * 2 * tied_params["theta_E"],
                t_E_aff,
            )
        )
        reactive_model_rtd = normalize_continuous_density(reactive_model_t, reactive_model_raw)

        selected_entries.append(
            {
                "ABL": int(ABL),
                "ILD": float(SELECTED_RTD_ILD),
                "n_rtd_trials": int(len(selected_data_df)),
                "n_truncated_rtd_trials": int(len(truncated_rts)),
                "n_reactive_data_trials": int(len(reactive_raw_rts)),
                "t_E_aff_ms": 1e3 * condition_results[key]["t_E_aff_s"],
                "data_rtd": data_rtd,
                "model_rtd": model_rtd,
                "truncated_data_rtd": truncated_data_rtd,
                "truncated_model_rtd": truncated_model_rtd,
                "reactive_data_rtd": reactive_data_rtd,
                "reactive_model_rtd": reactive_model_rtd,
                "reactive_model_raw": reactive_model_raw,
                "data_area": float(np.nansum(data_rtd * np.diff(selected_full_bins))),
                "model_area": float(trapezoid(model_rtd, THEORY_RT_PTS)),
                "truncated_data_area": float(np.nansum(truncated_data_rtd * np.diff(selected_trunc_bins))),
                "truncated_model_area": float(trapezoid(truncated_model_rtd, truncated_model_t)),
                "truncated_data_retained_area": float(truncated_data_retained_area),
                "truncated_model_retained_area": float(truncated_model_retained_area),
                "reactive_data_area": float(np.nansum(reactive_data_rtd * np.diff(selected_reactive_bins))),
                "reactive_model_raw_area": float(trapezoid(reactive_model_raw, reactive_model_t)),
                "reactive_model_area": float(trapezoid(reactive_model_rtd, reactive_model_t)),
            }
        )

    if selected_entries:
        fig, axes = plt.subplots(
            3,
            len(selected_entries),
            figsize=(5.2 * len(selected_entries), 9.8),
            sharey="row",
        )
        axes = np.asarray(axes)
        if axes.ndim == 1:
            axes = axes.reshape(3, 1)

        top_y_max = 0.0
        bottom_y_max = 0.0
        reactive_y_max = 0.0
        top_data_mask = (
            (selected_full_bin_centers >= selected_top_xlim[0])
            & (selected_full_bin_centers <= selected_top_xlim[1])
        )
        top_model_mask = (THEORY_RT_PTS >= selected_top_xlim[0]) & (THEORY_RT_PTS <= selected_top_xlim[1])
        reactive_data_mask = (
            (selected_reactive_bin_centers >= selected_reactive_plot_xlim[0])
            & (selected_reactive_bin_centers <= selected_reactive_plot_xlim[1])
        )
        reactive_model_mask = (
            (reactive_model_t >= selected_reactive_plot_xlim[0])
            & (reactive_model_t <= selected_reactive_plot_xlim[1])
        )
        for entry in selected_entries:
            if np.any(np.isfinite(entry["data_rtd"][top_data_mask])):
                top_y_max = max(top_y_max, float(np.nanmax(entry["data_rtd"][top_data_mask])))
            if np.any(np.isfinite(entry["model_rtd"][top_model_mask])):
                top_y_max = max(top_y_max, float(np.nanmax(entry["model_rtd"][top_model_mask])))
            if np.any(np.isfinite(entry["truncated_data_rtd"])):
                bottom_y_max = max(bottom_y_max, float(np.nanmax(entry["truncated_data_rtd"])))
            if np.any(np.isfinite(entry["truncated_model_rtd"])):
                bottom_y_max = max(bottom_y_max, float(np.nanmax(entry["truncated_model_rtd"])))
            if np.any(np.isfinite(entry["reactive_data_rtd"][reactive_data_mask])):
                reactive_y_max = max(reactive_y_max, float(np.nanmax(entry["reactive_data_rtd"][reactive_data_mask])))
            if np.any(np.isfinite(entry["reactive_model_rtd"][reactive_model_mask])):
                reactive_y_max = max(reactive_y_max, float(np.nanmax(entry["reactive_model_rtd"][reactive_model_mask])))

        for col_idx, entry in enumerate(selected_entries):
            t_E_aff_s = entry["t_E_aff_ms"] / 1e3
            ax = axes[0, col_idx]
            ax.plot(
                selected_full_bin_centers,
                entry["data_rtd"],
                color=COLORS["data"],
                linewidth=1.8,
                label="Data",
            )
            ax.plot(
                THEORY_RT_PTS,
                entry["model_rtd"],
                color=COLORS["model"],
                linewidth=1.7,
                label="SVI posterior mean",
            )
            ax.axvline(0, color="0.82", linewidth=0.7, zorder=0)
            ax.axvline(
                t_E_aff_s,
                color="#D55E00",
                linestyle=":",
                linewidth=1.5,
                label="t_E_aff",
            )
            ax.set_title(
                f"ABL {entry['ABL']}, ILD {entry['ILD']:+g}\n"
                f"t_E_aff={entry['t_E_aff_ms']:.1f} ms",
                fontsize=12,
            )
            ax.set_xlim(*selected_top_xlim)
            ax.set_xticks([-0.2, 0, 0.2, 0.4, 0.6])
            if top_y_max > 0:
                ax.set_ylim(0, top_y_max * 1.08)
            ax.tick_params(axis="both", labelsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ax = axes[1, col_idx]
            ax.plot(
                selected_trunc_bin_centers,
                entry["truncated_data_rtd"],
                color=COLORS["data"],
                linewidth=1.8,
                label="Data",
            )
            ax.plot(
                truncated_model_t,
                entry["truncated_model_rtd"],
                color=COLORS["model"],
                linewidth=1.7,
                label="SVI posterior mean",
            )
            ax.axvline(0, color="0.82", linewidth=0.7, zorder=0)
            ax.axvline(
                t_E_aff_s,
                color="#D55E00",
                linestyle=":",
                linewidth=1.5,
                label="t_E_aff",
            )
            ax.set_title(
                f"0 to 0.3 truncated from above\n"
                f"Plotted area data/model: {entry['truncated_data_area']:.2f}/{entry['truncated_model_area']:.2f}",
                fontsize=12,
            )
            ax.set_xlim(*selected_trunc_xlim)
            ax.set_xticks([0, 0.1, 0.2, 0.3])
            if bottom_y_max > 0:
                ax.set_ylim(0, bottom_y_max * 1.08)
            ax.set_xlabel("RT wrt stimulus (s)", fontsize=10)
            ax.tick_params(axis="both", labelsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ax = axes[2, col_idx]
            ax.plot(
                selected_reactive_bin_centers,
                entry["reactive_data_rtd"],
                color=COLORS["data"],
                linewidth=1.8,
                label="Valid data",
            )
            ax.plot(
                reactive_model_t,
                entry["reactive_model_rtd"],
                color=COLORS["model"],
                linewidth=1.7,
                label="Reactive only",
            )
            ax.axvline(0, color="0.82", linewidth=0.7, zorder=0)
            ax.axvline(
                t_E_aff_s,
                color="#D55E00",
                linestyle=":",
                linewidth=1.5,
                label="t_E_aff",
            )
            ax.set_title(
                f"0 to 1 s valid trials vs reactive only\n"
                f"Area from 0 to 1: {entry['reactive_model_raw_area']:.2f}",
                fontsize=12,
            )
            ax.set_xlim(*selected_reactive_plot_xlim)
            ax.set_xticks([0, 0.1, 0.2, 0.3])
            if reactive_y_max > 0:
                ax.set_ylim(0, reactive_y_max * 1.08)
            ax.set_xlabel("RT wrt stimulus (s)", fontsize=10)
            ax.tick_params(axis="both", labelsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        axes[0, 0].set_ylabel("Density", fontsize=11)
        axes[1, 0].set_ylabel("Conditional density", fontsize=11)
        axes[2, 0].set_ylabel("Conditional density", fontsize=11)
        axes[0, -1].legend(frameon=False, fontsize=9, loc="upper right")
        axes[2, -1].legend(frameon=False, fontsize=9, loc="upper right")
        selected_abl_text = ", ".join(str(entry["ABL"]) for entry in selected_entries)
        fig.suptitle(
            f"ABL {selected_abl_text}; ILD {SELECTED_RTD_ILD:g}; {BATCH_NAME}/{ANIMAL}",
            fontsize=14,
            y=0.99,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(RTD_SELECTED_PNG, dpi=300, bbox_inches="tight")
        print(f"Saved selected signed-ILD RTD figure: {RTD_SELECTED_PNG}")
        for entry in selected_entries:
            print(
                f"  ABL {entry['ABL']}, ILD {entry['ILD']:+g}: "
                f"t_E_aff={entry['t_E_aff_ms']:.3f} ms, n={entry['n_rtd_trials']}, "
                f"areas data={entry['data_area']:.3f}, model={entry['model_area']:.3f}, "
                f"truncated areas data={entry['truncated_data_area']:.3f}, "
                f"model={entry['truncated_model_area']:.3f}, "
                f"valid data/reactive areas data={entry['reactive_data_area']:.3f}, "
                f"reactive_raw={entry['reactive_model_raw_area']:.3f}, "
                f"reactive_norm={entry['reactive_model_area']:.3f}"
            )


# %%
# =============================================================================
# Plot psychometric curves
# =============================================================================
fig, axes = plt.subplots(1, len(ABLS), figsize=(12.5, 3.8), sharex=True, sharey=True)

for ax, ABL in zip(axes, ABLS):
    psy = psychometric_by_abl[ABL]
    valid_data = np.isfinite(psy["data"]) & (psy["n_trials"] > 0)
    valid_model = np.isfinite(psy["model"])

    ax.errorbar(
        psy["ilds"][valid_data],
        psy["data"][valid_data],
        yerr=psy["data_se"][valid_data],
        fmt="o",
        color=COLORS["data"],
        markersize=4,
        linewidth=1.2,
        capsize=2,
        label="Data",
    )
    ax.plot(
        psy["ilds"][valid_model],
        psy["model"][valid_model],
        "-o",
        color=COLORS["model"],
        linewidth=1.6,
        markersize=3.5,
        label="SVI posterior mean",
    )

    ax.axhline(0.5, color="0.75", linewidth=0.8, zorder=0)
    ax.axvline(0, color="0.85", linewidth=0.8, zorder=0)
    ax.set_title(f"ABL {ABL}", fontsize=13)
    ax.set_xlim(-17, 17)
    ax.set_xticks(psy["ilds"])
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_xlabel("ILD", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("P(right)", fontsize=11)
axes[0].set_ylim(-0.02, 1.02)
axes[-1].legend(frameon=False, fontsize=9, loc="lower right")
fig.suptitle(f"{BATCH_NAME}/{ANIMAL} SVI psychometric", fontsize=15, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(PSYCHOMETRIC_PNG, dpi=300, bbox_inches="tight")
print(f"Saved psychometric figure: {PSYCHOMETRIC_PNG}")

print("Done.")

# %%
