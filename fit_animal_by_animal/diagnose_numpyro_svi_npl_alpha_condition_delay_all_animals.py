# %%
"""
All-animal RTD and psychometric diagnostics for the NumPyro SVI
NPL+alpha condition-delay fits.

This script recomputes posterior-mean model RTDs/psychometrics from the saved
SVI posterior samples and averages the resulting diagnostics across animals.
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

SVI_LABEL = os.environ.get("NUMPYRO_SVI_DIAG_LABEL", "main_fullrank")
SVI_OUTPUT_ROOT = SCRIPT_DIR / "numpyro_svi_npl_alpha_condition_delay_single_animal_outputs"
OUTPUT_DIR = SCRIPT_DIR / "numpyro_svi_npl_alpha_condition_delay_all_animals_diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_N_ANIMALS = int(os.environ.get("NUMPYRO_SVI_EXPECTED_N_ANIMALS", "30"))
ABLS = [20, 40, 60]
ILD_GRID = [-16.0, -8.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0, 16.0]
ABS_ILD_GRID = [1.0, 2.0, 4.0, 8.0, 16.0]
DESIRED_BATCHES = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]

RT_MIN = float(os.environ.get("NUMPYRO_SVI_DIAG_RT_MIN", "-1.0"))
RT_MAX = float(os.environ.get("NUMPYRO_SVI_DIAG_RT_MAX", "1.0"))
RT_BIN_WIDTH = float(os.environ.get("NUMPYRO_SVI_DIAG_RT_BIN_WIDTH", "0.02"))
DT_THEORY = float(os.environ.get("NUMPYRO_SVI_DIAG_DT_THEORY", "0.001"))
N_THEORY = int(os.environ.get("NUMPYRO_SVI_DIAG_N_THEORY", "1000"))
RNG_SEED = int(os.environ.get("NUMPYRO_SVI_DIAG_SEED", "0"))
K_MAX = int(os.environ.get("K_MAX", "10"))

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

RTD_BY_ABL_PNG = OUTPUT_DIR / f"{SVI_LABEL}_all_animals_rtd_by_abl.png"
RTD_ABS_ILD_PNG = OUTPUT_DIR / f"{SVI_LABEL}_all_animals_rtd_by_abl_abs_ild.png"
RTD_ABS_ILD_ZOOM_PNG = OUTPUT_DIR / f"{SVI_LABEL}_all_animals_rtd_by_abl_abs_ild_zoom.png"
PSYCHOMETRIC_PNG = OUTPUT_DIR / f"{SVI_LABEL}_all_animals_psychometric_by_abl.png"
OUTPUT_PKL = OUTPUT_DIR / f"{SVI_LABEL}_all_animals_rtd_psychometric_data.pkl"


# %%
# =============================================================================
# Imports and setup
# =============================================================================
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.special as scipy_special
from scipy.integrate import trapezoid
from scipy.stats import sem

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

COLORS = {"data": "black", "model": "#0072B2"}
LABELS = {"data": "Data", "model": "NumPyro SVI posterior mean"}


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
# Data helpers
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


def mean_and_sem(curves, n_pts):
    curves = np.asarray(curves, dtype=float)
    if curves.size == 0:
        return np.full(n_pts, np.nan), np.full(n_pts, np.nan), 0
    return np.nanmean(curves, axis=0), sem(curves, axis=0, nan_policy="omit"), int(curves.shape[0])


def psychometric_mean_sem(values):
    values = np.asarray(values, dtype=float)
    return (
        np.nanmean(values, axis=0),
        sem(values, axis=0, nan_policy="omit"),
        np.sum(np.isfinite(values), axis=0),
    )


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
    return ensure_choice_column(exp_df_batch)


def calculate_truncated_theoretical_curves(df_valid_and_aborts, N_theory, t_pts, t_A_aff, V_A, theta_A, T_trunc, seed):
    if len(df_valid_and_aborts) == 0:
        raise RuntimeError("No valid/abort rows available for proactive truncation samples.")
    t_stim_samples = df_valid_and_aborts["intended_fix"].sample(
        int(N_theory),
        replace=True,
        random_state=seed,
    ).to_numpy(dtype=float)

    trunc_cdf = cum_A_t_vec(np.array([T_trunc - t_A_aff]), V_A, theta_A)[0]
    trunc_survival = max(1 - trunc_cdf, 1e-12)

    P_A_samples = np.zeros((len(t_stim_samples), len(t_pts)))
    C_A_samples = np.zeros((len(t_stim_samples), len(t_pts)))
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
# Discover and load shared data
# =============================================================================
print(f"SVI output root: {SVI_OUTPUT_ROOT}")
print(f"Output directory: {OUTPUT_DIR}")

animal_dirs = sorted(
    [
        path
        for path in SVI_OUTPUT_ROOT.glob("*_*")
        if path.is_dir() and path.name != "_batch_logs"
    ],
    key=lambda path: path.name,
)
if len(animal_dirs) != EXPECTED_N_ANIMALS:
    raise RuntimeError(f"Expected {EXPECTED_N_ANIMALS} SVI animal directories, found {len(animal_dirs)}")

animal_keys = []
for animal_dir in animal_dirs:
    batch_name, animal_text = animal_dir.name.rsplit("_", 1)
    animal_keys.append((batch_name, int(animal_text), animal_dir))

batch_order = {batch: idx for idx, batch in enumerate(DESIRED_BATCHES)}
animal_keys = sorted(animal_keys, key=lambda item: (batch_order.get(item[0], 999), item[0], item[1]))
print("Animals:")
for batch_name, animal, _ in animal_keys:
    print(f"  {batch_name}/{animal}")

batch_csvs = {}
raw_rtd_sources = {}
for batch_name in sorted({batch for batch, _, _ in animal_keys}):
    batch_csv = REPO_DIR / "raw_data" / "batch_csvs" / f"batch_{batch_name}_valid_and_aborts.csv"
    batch_df = ensure_choice_column(pd.read_csv(batch_csv))
    batch_df["animal"] = batch_df["animal"].astype(int)
    batch_df["ABL"] = batch_df["ABL"].astype(float)
    batch_df["ILD"] = batch_df["ILD"].astype(float)
    batch_csvs[batch_name] = batch_df

    rtd_df = load_raw_rtd_source_for_batch(batch_name)
    rtd_df["animal"] = rtd_df["animal"].astype(int)
    rtd_df["ABL"] = rtd_df["ABL"].astype(float)
    rtd_df["ILD"] = rtd_df["ILD"].astype(float)
    rtd_df["T_trunc"] = rtd_df["batch_name"].map(BATCH_T_TRUNC).fillna(DEFAULT_T_TRUNC)
    raw_rtd_sources[batch_name] = rtd_df


# %%
# =============================================================================
# Per-animal summaries
# =============================================================================
def process_animal(batch_name, animal, animal_dir, animal_idx):
    print(f"Processing {batch_name}/{animal}")

    posterior_npz = animal_dir / f"{SVI_LABEL}_posterior_samples.npz"
    condition_csv = animal_dir / "condition_table.csv"
    abort_result_pkl = REPO_DIR / "aborts_ipl_npl_time_fit_results" / f"results_{batch_name}_animal_{animal}.pkl"
    for path in [posterior_npz, condition_csv, abort_result_pkl]:
        if not path.exists():
            raise FileNotFoundError(path)

    with np.load(posterior_npz) as saved:
        posterior_np = {key: np.asarray(saved[key]) for key in saved.files}

    tied_params = {
        "rate_lambda": finite_mean(posterior_np["rate_lambda"], "rate_lambda"),
        "T_0": finite_mean(posterior_np["T_0"], "T_0"),
        "theta_E": finite_mean(posterior_np["theta_E"], "theta_E"),
        "w": finite_mean(posterior_np["w"], "w"),
        "del_go": finite_mean(posterior_np["del_go"], "del_go"),
        "rate_norm_l": finite_mean(posterior_np["rate_norm_l"], "rate_norm_l"),
        "alpha": finite_mean(posterior_np["alpha"], "alpha"),
    }
    t_E_aff_mean = np.nanmean(np.asarray(posterior_np["t_E_aff"], dtype=float), axis=0)
    if not np.all(np.isfinite(t_E_aff_mean)):
        raise RuntimeError(f"{batch_name}/{animal} has non-finite t_E_aff posterior means.")

    condition_table = pd.read_csv(condition_csv)
    condition_table["ABL"] = condition_table["ABL"].astype(float)
    condition_table["ILD"] = condition_table["ILD"].astype(float)
    condition_table["condition_id"] = condition_table["condition_id"].astype(int)
    condition_table = condition_table.sort_values("condition_id").reset_index(drop=True)
    if len(condition_table) != t_E_aff_mean.shape[0]:
        raise RuntimeError(
            f"{batch_name}/{animal} condition count {len(condition_table)} "
            f"does not match t_E_aff width {t_E_aff_mean.shape[0]}"
        )

    with abort_result_pkl.open("rb") as f:
        abort_saved = pickle.load(f)
    abort_results = abort_saved["vbmc_aborts_results"]
    abort_params = {
        "V_A": float(np.mean(abort_results["V_A_samples"])),
        "theta_A": float(np.mean(abort_results["theta_A_samples"])),
        "t_A_aff": float(np.mean(abort_results["t_A_aff_samp"])),
    }

    batch_df = batch_csvs[batch_name]
    valid_df = batch_df[
        (batch_df["animal"] == animal)
        & batch_df["success"].isin([1, -1])
        & (batch_df["RTwrtStim"] < 1)
        & batch_df["ABL"].isin(ABLS)
    ].copy()
    valid_df = valid_df.dropna(subset=["TotalFixTime", "intended_fix", "ABL", "ILD", "choice", "RTwrtStim"])
    valid_df["choice"] = valid_df["choice"].astype(int)
    valid_df = valid_df.merge(condition_table[["ABL", "ILD", "condition_id"]], on=["ABL", "ILD"], how="left")
    if valid_df["condition_id"].isna().any():
        raise RuntimeError(f"Failed to assign condition IDs to all valid trials for {batch_name}/{animal}.")

    theory_df = batch_df[
        (batch_df["animal"] == animal)
        & (batch_df["success"].isin([1, -1]) | batch_df["abort_event"].isin(THEORY_ABORT_EVENTS))
    ].copy()
    theory_df = theory_df.dropna(subset=["intended_fix"])

    rtd_source = raw_rtd_sources[batch_name]
    rtd_trial_pool = rtd_source[
        (rtd_source["animal"] == animal)
        & (rtd_source["success"].isin([1, -1]) | rtd_source["abort_event"].isin(RTD_ABORT_EVENTS))
    ].copy()
    rtd_data = rtd_trial_pool[
        rtd_trial_pool["ABL"].isin(ABLS)
        & rtd_trial_pool["RTwrtStim"].between(RT_MIN, RT_MAX, inclusive="both")
        & (
            rtd_trial_pool["success"].isin([1, -1])
            | rtd_trial_pool["TotalFixTime"].ge(rtd_trial_pool["T_trunc"])
        )
    ].copy()

    T_trunc = BATCH_T_TRUNC.get(batch_name, DEFAULT_T_TRUNC)
    P_A_mean, C_A_mean, t_stim_samples = calculate_truncated_theoretical_curves(
        theory_df,
        N_THEORY,
        T_PTS,
        abort_params["t_A_aff"],
        abort_params["V_A"],
        abort_params["theta_A"],
        T_trunc,
        RNG_SEED + animal_idx,
    )

    condition_results = {}
    for condition in condition_table.itertuples(index=False):
        ABL = int(condition.ABL)
        ILD = float(condition.ILD)
        condition_id = int(condition.condition_id)

        valid_condition_df = valid_df[(valid_df["ABL"] == ABL) & np.isclose(valid_df["ILD"], ILD)]
        rtd_condition_df = rtd_data[(rtd_data["ABL"] == ABL) & np.isclose(rtd_data["ILD"], ILD)]

        data_right_prob = float(np.mean(valid_condition_df["choice"] == 1)) if len(valid_condition_df) else np.nan
        t_E_aff = float(t_E_aff_mean[condition_id])
        t_pts, model_up, model_down, trunc_factor = get_theoretical_up_down(
            P_A_mean,
            C_A_mean,
            t_stim_samples,
            abort_params,
            tied_params,
            ABL,
            ILD,
            batch_name,
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
            "model_rtd_binned": model_rtd_binned,
            "model_rtd_continuous": model_rtd_continuous,
            "data_right_prob": data_right_prob,
            "model_right_prob": model_right_prob,
            "model_trunc_factor": trunc_factor,
        }

    rtd_by_abl = {}
    psychometric_by_abl = {}
    for ABL in ABLS:
        signed_ilds = sorted([ILD for abl, ILD in condition_results if abl == ABL])
        if not signed_ilds:
            continue
        rtd_by_abl[ABL] = {
            "data": average_equal_weight_curves(
                [condition_results[(ABL, ILD)]["data_rtd"] for ILD in signed_ilds]
            ),
            "model_binned": average_equal_weight_curves(
                [condition_results[(ABL, ILD)]["model_rtd_binned"] for ILD in signed_ilds]
            ),
            "model_continuous": average_equal_weight_continuous_curves(
                [condition_results[(ABL, ILD)]["model_rtd_continuous"] for ILD in signed_ilds]
            ),
            "ilds": np.array(signed_ilds, dtype=float),
        }
        psychometric_by_abl[ABL] = {
            "ilds": np.array(signed_ilds, dtype=float),
            "data": np.array([condition_results[(ABL, ILD)]["data_right_prob"] for ILD in signed_ilds]),
            "model": np.array([condition_results[(ABL, ILD)]["model_right_prob"] for ILD in signed_ilds]),
            "n_trials": np.array([condition_results[(ABL, ILD)]["n_valid_trials"] for ILD in signed_ilds], dtype=int),
        }

    abs_ild_rtd = {ABL: {} for ABL in ABLS}
    for ABL in ABLS:
        for abs_ild in ABS_ILD_GRID:
            signed_ilds = [ILD for ILD in [-abs_ild, abs_ild] if (ABL, ILD) in condition_results]
            if not signed_ilds:
                continue
            abs_ild_rtd[ABL][abs_ild] = {
                "signed_ilds": np.array(signed_ilds, dtype=float),
                "data": average_equal_weight_curves(
                    [condition_results[(ABL, ILD)]["data_rtd"] for ILD in signed_ilds]
                ),
                "model_binned": average_equal_weight_curves(
                    [condition_results[(ABL, ILD)]["model_rtd_binned"] for ILD in signed_ilds]
                ),
                "model_continuous": average_equal_weight_continuous_curves(
                    [condition_results[(ABL, ILD)]["model_rtd_continuous"] for ILD in signed_ilds]
                ),
                "n_rtd_trials": int(sum(condition_results[(ABL, ILD)]["n_rtd_trials"] for ILD in signed_ilds)),
            }

    return {
        "batch_name": batch_name,
        "animal": animal,
        "condition_count": int(len(condition_table)),
        "condition_results": condition_results,
        "rtd_by_abl": rtd_by_abl,
        "abs_ild_rtd": abs_ild_rtd,
        "psychometric_by_abl": psychometric_by_abl,
    }


all_animal_results = [
    process_animal(batch_name, animal, animal_dir, animal_idx)
    for animal_idx, (batch_name, animal, animal_dir) in enumerate(animal_keys)
]


# %%
# =============================================================================
# Aggregate summaries
# =============================================================================
rtd_curves = {ABL: {"data": [], "model_binned": [], "model_continuous": [], "animal_keys": []} for ABL in ABLS}
abs_ild_curves = {
    ABL: {
        abs_ild: {"data": [], "model_binned": [], "model_continuous": [], "animal_keys": []}
        for abs_ild in ABS_ILD_GRID
    }
    for ABL in ABLS
}
psy_arrays = {
    ABL: {
        "data": np.full((len(all_animal_results), len(ILD_GRID)), np.nan),
        "model": np.full((len(all_animal_results), len(ILD_GRID)), np.nan),
        "n_trials": np.full((len(all_animal_results), len(ILD_GRID)), np.nan),
    }
    for ABL in ABLS
}

for animal_idx, result in enumerate(all_animal_results):
    key = (result["batch_name"], result["animal"])
    for ABL in ABLS:
        if ABL in result["rtd_by_abl"]:
            for curve_key in ["data", "model_binned", "model_continuous"]:
                rtd_curves[ABL][curve_key].append(result["rtd_by_abl"][ABL][curve_key])
            rtd_curves[ABL]["animal_keys"].append(key)

        for abs_ild in ABS_ILD_GRID:
            if abs_ild not in result["abs_ild_rtd"].get(ABL, {}):
                continue
            for curve_key in ["data", "model_binned", "model_continuous"]:
                abs_ild_curves[ABL][abs_ild][curve_key].append(result["abs_ild_rtd"][ABL][abs_ild][curve_key])
            abs_ild_curves[ABL][abs_ild]["animal_keys"].append(key)

        if ABL not in result["psychometric_by_abl"]:
            continue
        psy = result["psychometric_by_abl"][ABL]
        for ild_idx, ILD in enumerate(ILD_GRID):
            matches = np.where(np.isclose(psy["ilds"], ILD))[0]
            if len(matches) != 1:
                continue
            src_idx = int(matches[0])
            psy_arrays[ABL]["data"][animal_idx, ild_idx] = psy["data"][src_idx]
            psy_arrays[ABL]["model"][animal_idx, ild_idx] = psy["model"][src_idx]
            psy_arrays[ABL]["n_trials"][animal_idx, ild_idx] = psy["n_trials"][src_idx]

rtd_summary = {}
for ABL in ABLS:
    rtd_summary[ABL] = {}
    data_mean, data_sem, data_n = mean_and_sem(rtd_curves[ABL]["data"], len(RT_BIN_CENTERS))
    model_binned_mean, model_binned_sem, model_binned_n = mean_and_sem(
        rtd_curves[ABL]["model_binned"], len(RT_BIN_CENTERS)
    )
    model_cont_mean, model_cont_sem, model_cont_n = mean_and_sem(
        rtd_curves[ABL]["model_continuous"], len(THEORY_RT_PTS)
    )
    rtd_summary[ABL]["data"] = {
        "mean": data_mean,
        "sem": data_sem,
        "n_animals": data_n,
        "area": float(np.nansum(data_mean * np.diff(RT_BINS))),
    }
    rtd_summary[ABL]["model_binned"] = {
        "mean": model_binned_mean,
        "sem": model_binned_sem,
        "n_animals": model_binned_n,
        "area": float(np.nansum(model_binned_mean * np.diff(RT_BINS))),
    }
    rtd_summary[ABL]["model_continuous"] = {
        "mean": model_cont_mean,
        "sem": model_cont_sem,
        "n_animals": model_cont_n,
        "area": float(trapezoid(model_cont_mean, THEORY_RT_PTS)),
    }

abs_ild_summary = {ABL: {abs_ild: {} for abs_ild in ABS_ILD_GRID} for ABL in ABLS}
for ABL in ABLS:
    for abs_ild in ABS_ILD_GRID:
        data_mean, data_sem, data_n = mean_and_sem(abs_ild_curves[ABL][abs_ild]["data"], len(RT_BIN_CENTERS))
        model_binned_mean, model_binned_sem, model_binned_n = mean_and_sem(
            abs_ild_curves[ABL][abs_ild]["model_binned"], len(RT_BIN_CENTERS)
        )
        model_cont_mean, model_cont_sem, model_cont_n = mean_and_sem(
            abs_ild_curves[ABL][abs_ild]["model_continuous"], len(THEORY_RT_PTS)
        )
        abs_ild_summary[ABL][abs_ild]["data"] = {
            "mean": data_mean,
            "sem": data_sem,
            "n_animals": data_n,
            "area": float(np.nansum(data_mean * np.diff(RT_BINS))),
        }
        abs_ild_summary[ABL][abs_ild]["model_binned"] = {
            "mean": model_binned_mean,
            "sem": model_binned_sem,
            "n_animals": model_binned_n,
            "area": float(np.nansum(model_binned_mean * np.diff(RT_BINS))),
        }
        abs_ild_summary[ABL][abs_ild]["model_continuous"] = {
            "mean": model_cont_mean,
            "sem": model_cont_sem,
            "n_animals": model_cont_n,
            "area": float(trapezoid(model_cont_mean, THEORY_RT_PTS)),
        }

psy_summary = {}
for ABL in ABLS:
    psy_summary[ABL] = {}
    for curve_key in ["data", "model"]:
        mean_values, sem_values, counts = psychometric_mean_sem(psy_arrays[ABL][curve_key])
        psy_summary[ABL][curve_key] = {
            "mean": mean_values,
            "sem": sem_values,
            "counts": counts,
        }

summary_metrics = {"rtd_ise_by_abl": {}, "psychometric_rmse_by_abl": {}}
all_rtd_ise = []
all_psy_errors = []
for ABL in ABLS:
    data_curves = np.asarray(rtd_curves[ABL]["data"], dtype=float)
    model_curves = np.asarray(rtd_curves[ABL]["model_binned"], dtype=float)
    if data_curves.size and model_curves.size:
        ise_values = np.nansum((model_curves - data_curves) ** 2 * np.diff(RT_BINS), axis=1)
        summary_metrics["rtd_ise_by_abl"][ABL] = float(np.nanmean(ise_values))
        all_rtd_ise.extend([float(x) for x in ise_values if np.isfinite(x)])
    errors = psy_arrays[ABL]["model"] - psy_arrays[ABL]["data"]
    summary_metrics["psychometric_rmse_by_abl"][ABL] = float(np.sqrt(np.nanmean(errors**2)))
    all_psy_errors.extend([float(x) for x in errors[np.isfinite(errors)]])
summary_metrics["rtd_ise_overall"] = float(np.nanmean(all_rtd_ise))
summary_metrics["psychometric_rmse_overall"] = float(np.sqrt(np.nanmean(np.asarray(all_psy_errors) ** 2)))

print("\nCondition-count check:")
condition_counts = pd.Series([result["condition_count"] for result in all_animal_results]).value_counts().sort_index()
print(condition_counts.to_string())
print("\nRTD area checks:")
for ABL in ABLS:
    print(
        f"  ABL {ABL}: data={rtd_summary[ABL]['data']['area']:.6f}, "
        f"model={rtd_summary[ABL]['model_continuous']['area']:.6f}, "
        f"n={rtd_summary[ABL]['data']['n_animals']}"
    )
for ABL in ABLS:
    for abs_ild in ABS_ILD_GRID:
        data_n = abs_ild_summary[ABL][abs_ild]["data"]["n_animals"]
        print(
            f"  ABL {ABL} |ILD| {abs_ild:g}: "
            f"data_area={abs_ild_summary[ABL][abs_ild]['data']['area']:.6f}, "
            f"model_area={abs_ild_summary[ABL][abs_ild]['model_continuous']['area']:.6f}, "
            f"n_animals={data_n}"
        )
print("\nSummary metrics:")
print(f"  RTD ISE overall: {summary_metrics['rtd_ise_overall']:.6f}")
print(f"  Psychometric RMSE overall: {summary_metrics['psychometric_rmse_overall']:.6f}")


# %%
# =============================================================================
# Save plot data
# =============================================================================
output_data = {
    "script": str(Path(__file__).relative_to(REPO_DIR)),
    "animal_keys": [(result["batch_name"], result["animal"]) for result in all_animal_results],
    "rt_bins": RT_BINS,
    "rt_bin_centers": RT_BIN_CENTERS,
    "rt_window_s": (RT_MIN, RT_MAX),
    "rt_bin_width_s": RT_BIN_WIDTH,
    "theory_dt_s": DT_THEORY,
    "theory_rt_pts": THEORY_RT_PTS,
    "ild_grid": np.array(ILD_GRID, dtype=float),
    "abs_ild_grid": np.array(ABS_ILD_GRID, dtype=float),
    "abls": np.array(ABLS, dtype=int),
    "rtd_curves": rtd_curves,
    "rtd_summary": rtd_summary,
    "abs_ild_curves": abs_ild_curves,
    "abs_ild_summary": abs_ild_summary,
    "psychometric_arrays": psy_arrays,
    "psychometric_summary": psy_summary,
    "per_animal_results": all_animal_results,
    "summary_metrics": summary_metrics,
    "preflight": {
        "n_animals": int(len(all_animal_results)),
        "condition_counts": {int(k): int(v) for k, v in condition_counts.items()},
        "theory_abort_events": THEORY_ABORT_EVENTS,
        "rtd_abort_events": RTD_ABORT_EVENTS,
        "batch_t_trunc_s": {batch: float(BATCH_T_TRUNC.get(batch, DEFAULT_T_TRUNC)) for batch in DESIRED_BATCHES},
    },
}
with OUTPUT_PKL.open("wb") as handle:
    pickle.dump(output_data, handle)
print(f"Saved plot data: {OUTPUT_PKL}")


# %%
# =============================================================================
# Plotting
# =============================================================================
def finish_axis(ax):
    ax.tick_params(axis="both", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


fig, axes = plt.subplots(1, len(ABLS), figsize=(12.5, 3.8), sharex=True, sharey=True)
global_y_max = 0.0
for ABL in ABLS:
    data_mean = rtd_summary[ABL]["data"]["mean"]
    data_sem = rtd_summary[ABL]["data"]["sem"]
    model_mean = rtd_summary[ABL]["model_continuous"]["mean"]
    model_sem = rtd_summary[ABL]["model_continuous"]["sem"]
    if np.any(np.isfinite(data_mean + data_sem)):
        global_y_max = max(global_y_max, float(np.nanmax(data_mean + data_sem)))
    if np.any(np.isfinite(model_mean + model_sem)):
        global_y_max = max(global_y_max, float(np.nanmax(model_mean + model_sem)))

for ax, ABL in zip(axes, ABLS):
    data_mean = rtd_summary[ABL]["data"]["mean"]
    data_sem = rtd_summary[ABL]["data"]["sem"]
    model_mean = rtd_summary[ABL]["model_continuous"]["mean"]
    model_sem = rtd_summary[ABL]["model_continuous"]["sem"]

    ax.plot(RT_BIN_CENTERS, data_mean, color=COLORS["data"], linewidth=2.0, label=LABELS["data"])
    ax.fill_between(RT_BIN_CENTERS, data_mean - data_sem, data_mean + data_sem, color=COLORS["data"], alpha=0.13)
    ax.plot(THEORY_RT_PTS, model_mean, color=COLORS["model"], linewidth=1.8, label=LABELS["model"])
    ax.fill_between(THEORY_RT_PTS, model_mean - model_sem, model_mean + model_sem, color=COLORS["model"], alpha=0.16)
    ax.axvline(0, color="0.8", linewidth=0.8, zorder=0)
    ax.set_title(f"ABL {ABL}", fontsize=13)
    ax.set_xlim(RT_MIN, RT_MAX)
    ax.set_xticks([RT_MIN, -0.5, 0, 0.5, RT_MAX])
    ax.set_xlabel("RT wrt stimulus (s)", fontsize=11)
    if global_y_max > 0:
        ax.set_ylim(0, global_y_max * 1.06)
    finish_axis(ax)
axes[0].set_ylabel("Density", fontsize=11)
axes[-1].legend(frameon=False, fontsize=9, loc="upper right")
fig.suptitle("All-animal NumPyro SVI RTDs by ABL (-1 to 1 s)", fontsize=15, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(RTD_BY_ABL_PNG, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved RTD by ABL figure: {RTD_BY_ABL_PNG}")


def save_abs_ild_rtd_grid(output_png, xlim, xticks, title_suffix):
    fig, axes = plt.subplots(len(ABLS), len(ABS_ILD_GRID), figsize=(16.5, 8.4), sharex=True, sharey=True)
    global_y_max = 0.0
    for ABL in ABLS:
        for abs_ild in ABS_ILD_GRID:
            data_mean = abs_ild_summary[ABL][abs_ild]["data"]["mean"]
            data_sem = abs_ild_summary[ABL][abs_ild]["data"]["sem"]
            model_mean = abs_ild_summary[ABL][abs_ild]["model_continuous"]["mean"]
            model_sem = abs_ild_summary[ABL][abs_ild]["model_continuous"]["sem"]
            data_mask = (RT_BIN_CENTERS >= xlim[0]) & (RT_BIN_CENTERS <= xlim[1])
            model_mask = (THEORY_RT_PTS >= xlim[0]) & (THEORY_RT_PTS <= xlim[1])
            if np.any(np.isfinite(data_mean[data_mask] + data_sem[data_mask])):
                global_y_max = max(global_y_max, float(np.nanmax(data_mean[data_mask] + data_sem[data_mask])))
            if np.any(np.isfinite(model_mean[model_mask] + model_sem[model_mask])):
                global_y_max = max(global_y_max, float(np.nanmax(model_mean[model_mask] + model_sem[model_mask])))

    for row_idx, ABL in enumerate(ABLS):
        for col_idx, abs_ild in enumerate(ABS_ILD_GRID):
            ax = axes[row_idx, col_idx]
            data_mean = abs_ild_summary[ABL][abs_ild]["data"]["mean"]
            data_sem = abs_ild_summary[ABL][abs_ild]["data"]["sem"]
            model_mean = abs_ild_summary[ABL][abs_ild]["model_continuous"]["mean"]
            model_sem = abs_ild_summary[ABL][abs_ild]["model_continuous"]["sem"]
            n_animals = abs_ild_summary[ABL][abs_ild]["data"]["n_animals"]

            ax.plot(RT_BIN_CENTERS, data_mean, color=COLORS["data"], linewidth=1.8, label=LABELS["data"])
            ax.fill_between(RT_BIN_CENTERS, data_mean - data_sem, data_mean + data_sem, color=COLORS["data"], alpha=0.12)
            ax.plot(THEORY_RT_PTS, model_mean, color=COLORS["model"], linewidth=1.5, label=LABELS["model"])
            ax.fill_between(THEORY_RT_PTS, model_mean - model_sem, model_mean + model_sem, color=COLORS["model"], alpha=0.15)
            ax.text(0.04, 0.90, f"n={n_animals}", transform=ax.transAxes, fontsize=8, color="0.35")

            ax.axvline(0, color="0.8", linewidth=0.7, zorder=0)
            if row_idx == 0:
                ax.set_title(f"|ILD| = {abs_ild:g}", fontsize=12)
            if col_idx == 0:
                ax.set_ylabel(f"ABL {ABL}\nDensity", fontsize=11)
            if row_idx == len(ABLS) - 1:
                ax.set_xlabel("RT wrt stimulus (s)", fontsize=10)
            ax.set_xlim(*xlim)
            ax.set_xticks(xticks)
            if global_y_max > 0:
                ax.set_ylim(0, global_y_max * 1.06)
            finish_axis(ax)

    axes[0, -1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(f"All-animal NumPyro SVI RTDs by ABL and |ILD| ({title_suffix})", fontsize=15, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved RTD by |ILD| figure: {output_png}")


save_abs_ild_rtd_grid(RTD_ABS_ILD_PNG, (RT_MIN, RT_MAX), [RT_MIN, 0, RT_MAX], "-1 to 1 s")
save_abs_ild_rtd_grid(RTD_ABS_ILD_ZOOM_PNG, (-0.6, 0.6), [-0.6, 0, 0.6], "-0.6 to 0.6 s")


fig, axes = plt.subplots(1, len(ABLS), figsize=(12.5, 3.8), sharex=True, sharey=True)
for ax, ABL in zip(axes, ABLS):
    data_mean = psy_summary[ABL]["data"]["mean"]
    data_sem = psy_summary[ABL]["data"]["sem"]
    data_counts = psy_summary[ABL]["data"]["counts"]
    model_mean = psy_summary[ABL]["model"]["mean"]
    model_sem = psy_summary[ABL]["model"]["sem"]
    model_counts = psy_summary[ABL]["model"]["counts"]

    valid_data = np.isfinite(data_mean) & (data_counts > 0)
    valid_model = np.isfinite(model_mean) & (model_counts > 0)
    ild_grid = np.asarray(ILD_GRID, dtype=float)

    ax.errorbar(
        ild_grid[valid_data],
        data_mean[valid_data],
        yerr=data_sem[valid_data],
        fmt="o",
        color=COLORS["data"],
        markersize=4,
        linewidth=1.2,
        capsize=2,
        label=LABELS["data"],
    )
    ax.errorbar(
        ild_grid[valid_model],
        model_mean[valid_model],
        yerr=model_sem[valid_model],
        fmt="-o",
        color=COLORS["model"],
        markersize=3.5,
        linewidth=1.6,
        capsize=2,
        label=LABELS["model"],
    )
    ax.axhline(0.5, color="0.75", linewidth=0.8, zorder=0)
    ax.axvline(0, color="0.85", linewidth=0.8, zorder=0)
    ax.set_title(f"ABL {ABL}", fontsize=13)
    ax.set_xlim(-17, 17)
    ax.set_xticks(ild_grid)
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_xlabel("ILD", fontsize=11)
    finish_axis(ax)
axes[0].set_ylabel("P(right)", fontsize=11)
axes[0].set_ylim(-0.02, 1.02)
axes[-1].legend(frameon=False, fontsize=9, loc="lower right")
fig.suptitle("All-animal NumPyro SVI psychometric", fontsize=15, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(PSYCHOMETRIC_PNG, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved psychometric figure: {PSYCHOMETRIC_PNG}")

print("Done.")

# %%
