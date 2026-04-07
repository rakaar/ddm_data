# %%
"""
Quick check: compare un-normalized theory area with data fractions for each ABL.
"""

# %%
from pathlib import Path
import os
import pickle
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ANIMAL_WISE_DIR = REPO_ROOT / "fit_animal_by_animal"
if str(ANIMAL_WISE_DIR) not in sys.path:
    sys.path.insert(0, str(ANIMAL_WISE_DIR))

from time_vary_norm_utils import (
    CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn_vec,
    rho_A_t_VEC_fn,
    rho_E_minus_small_t_NORM_rate_norm_time_varying_fn_vec,
)

# %%
############ Parameters ############
batch_name = "LED7"
session_type = 7
training_level = 16
allowed_repeat_trials = [0, 2]
max_rtwrtstim_for_fit = 1.0
data_bin_size_s_truncated = 5e-3

t_pts = np.arange(-2.0, 2.001, 0.001)
supported_abl_values = (20, 40, 60)

is_norm = True
is_time_vary = False
phi_params_obj = np.nan
K_max = 10

truncate_rt_wrt_stim_s = 0.130
fix_trial_count_by_abl = True
fixed_trial_counts_by_abl = {20: 1300, 40: 2300, 60: 3400}


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
    raise ValueError(f"Unsupported ABL value {abl_value}.")


def normalize_fixed_trial_counts_by_abl(requested_counts):
    return {int(abl): int(requested_counts[int(abl)]) for abl in supported_abl_values}


def build_run_tag(truncate_rt_wrt_stim_s, fix_trial_count_by_abl, fixed_trial_counts_by_abl):
    truncate_ms = int(round(float(truncate_rt_wrt_stim_s) * 1e3))
    truncate_tag = f"trunc{truncate_ms}ms"
    if not fix_trial_count_by_abl:
        return f"{truncate_tag}_allvalid"
    count_tag = "_".join(
        f"{int(abl)}-{int(fixed_trial_counts_by_abl[int(abl)])}" for abl in supported_abl_values
    )
    return f"{truncate_tag}_fixN_{count_tag}"


normalized_fixed_trial_counts_by_abl = normalize_fixed_trial_counts_by_abl(fixed_trial_counts_by_abl)
requested_run_tag = build_run_tag(truncate_rt_wrt_stim_s, fix_trial_count_by_abl, normalized_fixed_trial_counts_by_abl)


# %%
############ Load fitted parameters ############
results_pkl_path = (
    SCRIPT_DIR
    / "norm_only_led_off_from_loaded_proactive_truncate_NOT_censor_ABL_delay_no_choice"
    / f"results_norm_tied_batch_{batch_name}_aggregate_ledoff_1_proactive_loaded_truncate_NOT_censor_ABL_delay_no_choice_{requested_run_tag}.pkl"
)
with open(results_pkl_path, "rb") as f:
    fit_payload = pickle.load(f)

vbmc_results = fit_payload["vbmc_norm_tied_results"]
loaded_pro = fit_payload["loaded_proactive_params"]

rate_lambda = float(np.mean(vbmc_results["rate_lambda_samples"]))
T_0 = float(np.mean(vbmc_results["T_0_samples"]))
theta_E = float(np.mean(vbmc_results["theta_E_samples"]))
w = float(np.mean(vbmc_results["w_samples"]))
t_E_aff_20 = float(np.mean(vbmc_results["t_E_aff_20_samples"]))
t_E_aff_40 = float(np.mean(vbmc_results["t_E_aff_40_samples"]))
t_E_aff_60 = float(np.mean(vbmc_results["t_E_aff_60_samples"]))
del_go = float(np.mean(vbmc_results["del_go_samples"]))
rate_norm_l = float(np.mean(vbmc_results["rate_norm_l_samples"]))
Z_E = (w - 0.5) * 2.0 * theta_E

V_A = float(loaded_pro["V_A_base"])
theta_A = float(loaded_pro["theta_A"])
del_a_minus_del_LED = float(loaded_pro["del_a_minus_del_LED"])
del_m_plus_del_LED = float(loaded_pro["del_m_plus_del_LED"])
lapse_prob = float(loaded_pro["lapse_prob"])
beta_lapse = float(loaded_pro["beta_lapse"])
t_A_aff = del_a_minus_del_LED + del_m_plus_del_LED


# %%
############ Load data ############
led_data_csv_path = REPO_ROOT / "out_LED.csv"
exp_df = pd.read_csv(led_data_csv_path)
exp_df["RTwrtStim"] = exp_df["timed_fix"] - exp_df["intended_fix"]
exp_df["t_LED"] = exp_df["intended_fix"] - exp_df["LED_onset_time"]
exp_df = exp_df.rename(columns={"timed_fix": "TotalFixTime"})
exp_df = exp_df[exp_df["RTwrtStim"] < 1].copy()
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
    mask_led_off & mask_repeat
    & exp_df["session_type"].isin([session_type])
    & exp_df["training_level"].isin([training_level])
].copy()

df_valid_and_aborts = exp_df_led_off[
    (exp_df_led_off["success"].isin([1, -1])) | (exp_df_led_off["abort_event"].isin([3, 4]))
].copy()
fit_df = df_valid_and_aborts[df_valid_and_aborts["RTwrtStim"] < max_rtwrtstim_for_fit].copy()


# %%
############ Vectorized build_theory_curve ############
from scipy.special import erf as _erf


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
    t_arr = np.asarray(t_arr, dtype=np.float64)
    result = np.zeros_like(t_arr)
    valid = t_arr > 0
    tv = t_arr[valid]
    term1 = 0.5 * (1 + _erf(V_A * (tv - theta_A / V_A) / np.sqrt(2 * tv)))
    term2 = np.exp(2 * V_A * theta_A) * 0.5 * (1 + _erf(-V_A * (tv + theta_A / V_A) / np.sqrt(2 * tv)))
    result[valid] = term1 + term2
    return result


def build_theory_curve_for_trial(t_stim, abl, ild):
    t_E_aff = get_t_E_aff_from_abl(abl, t_E_aff_20, t_E_aff_40, t_E_aff_60)
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
    phi_t_e = np.ones(n)
    int_phi_t_e = t1.copy()

    t_cdf_arg = t_v - t_stim - t_E_aff + del_go
    CDF_up = CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn_vec(
        t_cdf_arg, 1, ABL_arr, ILD_arr, rl_arr, T0_arr, thE_arr, ZE_arr,
        int_phi_t_E_g, rnl_arr, is_norm, is_time_vary, K_max)
    CDF_down = CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn_vec(
        t_cdf_arg, -1, ABL_arr, ILD_arr, rl_arr, T0_arr, thE_arr, ZE_arr,
        int_phi_t_E_g, rnl_arr, is_norm, is_time_vary, K_max)
    random_readout = 0.5 * (1.0 - (CDF_up + CDF_down))

    CDF_t2_up = CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn_vec(
        t2, 1, ABL_arr, ILD_arr, rl_arr, T0_arr, thE_arr, ZE_arr, t2, rnl_arr, is_norm, is_time_vary, K_max)
    CDF_t1_up = CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn_vec(
        t1, 1, ABL_arr, ILD_arr, rl_arr, T0_arr, thE_arr, ZE_arr, t1, rnl_arr, is_norm, is_time_vary, K_max)
    CDF_t2_down = CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn_vec(
        t2, -1, ABL_arr, ILD_arr, rl_arr, T0_arr, thE_arr, ZE_arr, t2, rnl_arr, is_norm, is_time_vary, K_max)
    CDF_t1_down = CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn_vec(
        t1, -1, ABL_arr, ILD_arr, rl_arr, T0_arr, thE_arr, ZE_arr, t1, rnl_arr, is_norm, is_time_vary, K_max)
    P_E_plus_cum_up = CDF_t2_up - CDF_t1_up
    P_E_plus_cum_down = CDF_t2_down - CDF_t1_down

    t_rho_arg = t_v - t_stim - t_E_aff
    P_E_plus_up = rho_E_minus_small_t_NORM_rate_norm_time_varying_fn_vec(
        t_rho_arg, 1, ABL_arr, ILD_arr, rl_arr, T0_arr, thE_arr, ZE_arr,
        phi_t_e, int_phi_t_e, rnl_arr, is_norm, is_time_vary, K_max)
    P_E_plus_down = rho_E_minus_small_t_NORM_rate_norm_time_varying_fn_vec(
        t_rho_arg, -1, ABL_arr, ILD_arr, rl_arr, T0_arr, thE_arr, ZE_arr,
        phi_t_e, int_phi_t_e, rnl_arr, is_norm, is_time_vary, K_max)

    lp = float(np.clip(lapse_prob, 0.0, 1.0))
    P_A_mix = (1.0 - lp) * P_A + lp * 0.5 * _lapse_pdf_vec(t_v, beta_lapse)
    C_A_mix = np.clip((1.0 - lp) * C_A + lp * _lapse_cdf_vec(t_v, beta_lapse), 0.0, 1.0)

    pdf_up = P_A_mix * (random_readout + P_E_plus_cum_up) + P_E_plus_up * (1.0 - C_A_mix)
    pdf_down = P_A_mix * (random_readout + P_E_plus_cum_down) + P_E_plus_down * (1.0 - C_A_mix)
    pdf_total = pdf_up + pdf_down
    eps = 1e-50
    pdf_total = np.where(np.isfinite(pdf_total) & (pdf_total > 0), pdf_total, eps)
    curve[valid] = np.where(pdf_total > eps, pdf_total, 0.0)
    return curve


# %%
############ Compute theory per ABL (mean over MC, un-normalized) ############
from joblib import Parallel, delayed

seed = 12345
rng = np.random.default_rng(seed)
# Use same seed as diagnostics for consistency
# Skip the first N_mc draws (same as diagnostics does for the aggregate MC)
N_mc = 1000
_ = rng.integers(0, len(fit_df), size=N_mc)  # burn the aggregate MC draws

N_mc_per_abl = 1000
n_jobs = 30

mask_truncated = (t_pts >= 0.0) & (t_pts <= truncate_rt_wrt_stim_s)
t_pts_truncated = t_pts[mask_truncated]

hist_edges_truncated = np.arange(0.0, truncate_rt_wrt_stim_s + data_bin_size_s_truncated, data_bin_size_s_truncated)
if hist_edges_truncated[-1] < truncate_rt_wrt_stim_s:
    hist_edges_truncated = np.append(hist_edges_truncated, truncate_rt_wrt_stim_s)
data_bin_widths_truncated = np.diff(hist_edges_truncated)
data_bin_centers_truncated = 0.5 * (hist_edges_truncated[:-1] + hist_edges_truncated[1:])

abl_values_float = fit_df["ABL"].astype(float).to_numpy()

print("=" * 70)
print("Comparing theory area vs data fractions for each ABL")
print(f"Truncation window: [0, {truncate_rt_wrt_stim_s*1e3:.0f} ms]")
print("=" * 70)

for abl in supported_abl_values:
    df_abl = fit_df[np.isclose(abl_values_float, float(abl))].copy()
    n_total_abl = len(df_abl)

    sampled_positions_abl = rng.integers(0, len(df_abl), size=N_mc_per_abl)
    sampled_rows_abl = df_abl.iloc[sampled_positions_abl].copy()

    def _compute_one(row):
        return build_theory_curve_for_trial(float(row["intended_fix"]), float(row["ABL"]), float(row["ILD"]))

    mc_results_abl = Parallel(n_jobs=n_jobs)(delayed(_compute_one)(row) for _, row in sampled_rows_abl.iterrows())
    theory_density_abl = np.mean(np.stack(mc_results_abl, axis=0), axis=0)
    theory_density_trunc = theory_density_abl[mask_truncated]
    theory_area_raw = float(np.trapz(theory_density_trunc, t_pts_truncated))

    # Data
    rt_abl = df_abl["RTwrtStim"].to_numpy(dtype=np.float64)
    rt_abl_trunc = rt_abl[(rt_abl >= 0.0) & (rt_abl <= truncate_rt_wrt_stim_s)]
    n_trunc = len(rt_abl_trunc)
    frac_truncated = n_trunc / n_total_abl

    # Option 1: density=True (area=1 among truncated)
    hist_density_true, _ = np.histogram(rt_abl_trunc, bins=hist_edges_truncated, density=True)
    area_density_true = float(np.sum(hist_density_true * data_bin_widths_truncated))

    # Option 2: counts / (n_total * bin_width) — area = n_trunc/n_total
    hist_counts, _ = np.histogram(rt_abl_trunc, bins=hist_edges_truncated, density=False)
    hist_frac_total = hist_counts / (n_total_abl * data_bin_widths_truncated)
    area_frac_total = float(np.sum(hist_frac_total * data_bin_widths_truncated))

    # Option 3: counts / (n_trunc * bin_width) — same as density=True
    hist_frac_trunc = hist_counts / (n_trunc * data_bin_widths_truncated)
    area_frac_trunc = float(np.sum(hist_frac_trunc * data_bin_widths_truncated))

    print(f"\nABL={abl}:")
    print(f"  n_total={n_total_abl}, n_truncated={n_trunc}, frac_truncated={frac_truncated:.6f}")
    print(f"  Theory area (raw, un-normalized): {theory_area_raw:.6f}")
    print(f"  Data option 1 (density=True):           area={area_density_true:.6f}")
    print(f"  Data option 2 (counts/n_total/binw):    area={area_frac_total:.6f}")
    print(f"  Data option 3 (counts/n_trunc/binw):    area={area_frac_trunc:.6f}")
    print(f"  Theory area / frac_truncated ratio: {theory_area_raw / frac_truncated:.6f}")
