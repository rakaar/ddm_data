# %%
"""
Test vectorized build_theory_curve_for_trial against the scalar version.
Validates correctness by comparing outputs, then benchmarks speed.
"""

# %%
from pathlib import Path
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ANIMAL_WISE_DIR = REPO_ROOT / "fit_animal_by_animal"
if str(ANIMAL_WISE_DIR) not in sys.path:
    sys.path.insert(0, str(ANIMAL_WISE_DIR))

from proactive_plus_lapse_plus_reactive_uitls import (
    rt_pdf_proactive_lapse_only_no_choice_fn,
)
from time_vary_norm_utils import (
    CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn,
    CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn_vec,
    cum_A_t_fn,
    int_phi_fn,
    phi_t_fn,
    rho_A_t_fn,
    rho_A_t_VEC_fn,
    rho_E_minus_small_t_NORM_rate_norm_time_varying_fn,
    rho_E_minus_small_t_NORM_rate_norm_time_varying_fn_vec,
)


# %%
############ Parameters ############
batch_name = "LED7"
session_type = 7
training_level = 16
allowed_repeat_trials = [0, 2]
max_rtwrtstim_for_fit = 1.0

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
############ Helpers (same as diagnostics) ############
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


normalized_fixed_trial_counts_by_abl = normalize_fixed_trial_counts_by_abl(
    fixed_trial_counts_by_abl
)
requested_run_tag = build_run_tag(
    truncate_rt_wrt_stim_s,
    fix_trial_count_by_abl,
    normalized_fixed_trial_counts_by_abl,
)


# %%
############ Load fitted parameters ############
results_pkl_path = (
    SCRIPT_DIR
    / "norm_only_led_off_from_loaded_proactive_truncate_NOT_censor_ABL_delay_no_choice"
    / (
        "results_norm_tied_"
        f"batch_{batch_name}_aggregate_ledoff_1_"
        "proactive_loaded_truncate_NOT_censor_ABL_delay_no_choice_"
        f"{requested_run_tag}.pkl"
    )
)

if not results_pkl_path.exists():
    raise FileNotFoundError(f"Could not find results pickle: {results_pkl_path}")

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

print("Loaded parameters OK")
print(f"  rate_lambda={rate_lambda:.6f}, T_0={T_0:.6f}, theta_E={theta_E:.6f}")
print(f"  t_E_aff_20={t_E_aff_20:.6f}, t_E_aff_40={t_E_aff_40:.6f}, t_E_aff_60={t_E_aff_60:.6f}")
print(f"  V_A={V_A:.6f}, theta_A={theta_A:.6f}, lapse_prob={lapse_prob:.6f}")


# %%
############ Rebuild LED-OFF data (same as diagnostics) ############
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
    mask_led_off
    & mask_repeat
    & exp_df["session_type"].isin([session_type])
    & exp_df["training_level"].isin([training_level])
].copy()

df_valid_and_aborts = exp_df_led_off[
    (exp_df_led_off["success"].isin([1, -1])) | (exp_df_led_off["abort_event"].isin([3, 4]))
].copy()
fit_df = df_valid_and_aborts[df_valid_and_aborts["RTwrtStim"] < max_rtwrtstim_for_fit].copy()
print(f"Data loaded: {len(fit_df)} trials")


# %%
############ SCALAR version (original) ############
def build_theory_curve_for_trial_scalar(t_stim, abl, ild):
    t_E_aff = get_t_E_aff_from_abl(abl, t_E_aff_20, t_E_aff_40, t_E_aff_60)
    t_abs = t_pts + t_stim
    curve = np.zeros_like(t_pts, dtype=np.float64)

    for j, t_abs_j in enumerate(t_abs):
        if t_abs_j <= 0:
            continue
        pdf = rt_pdf_proactive_lapse_only_no_choice_fn(
            t=t_abs_j,
            V_A=V_A,
            theta_A=theta_A,
            t_A_aff=t_A_aff,
            t_stim=t_stim,
            ABL=abl,
            ILD=ild,
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
            eps=1e-50,
        )
        if np.isfinite(pdf) and pdf > 0:
            curve[j] = pdf

    return curve


# %%
############ VECTORIZED version ############
def cum_A_t_vec_fn(t_arr, V_A, theta_A):
    """Vectorized proactive CDF."""
    t_arr = np.asarray(t_arr, dtype=np.float64)
    result = np.zeros_like(t_arr)
    valid = t_arr > 0
    tv = t_arr[valid]
    term1 = 0.5 * (1 + erf(V_A * (tv - theta_A / V_A) / np.sqrt(2 * tv)))
    term2 = np.exp(2 * V_A * theta_A) * 0.5 * (1 + erf(-V_A * (tv + theta_A / V_A) / np.sqrt(2 * tv)))
    result[valid] = term1 + term2
    return result


from scipy.special import erf


def lapse_pdf_vec(t_arr, beta_lapse, eps=1e-50):
    """Vectorized exponential lapse PDF."""
    t_arr = np.asarray(t_arr, dtype=np.float64)
    result = np.full_like(t_arr, eps)
    valid = t_arr >= 0
    vals = beta_lapse * np.exp(-beta_lapse * t_arr[valid])
    vals = np.where(np.isfinite(vals) & (vals > 0), vals, eps)
    result[valid] = vals
    return result


def lapse_cdf_vec(t_arr, beta_lapse):
    """Vectorized exponential lapse CDF."""
    t_arr = np.asarray(t_arr, dtype=np.float64)
    result = np.zeros_like(t_arr)
    valid = t_arr > 0
    result[valid] = np.clip(1.0 - np.exp(-beta_lapse * t_arr[valid]), 0.0, 1.0)
    return result


def build_theory_curve_for_trial_vectorized(t_stim, abl, ild):
    t_E_aff = get_t_E_aff_from_abl(abl, t_E_aff_20, t_E_aff_40, t_E_aff_60)
    t_abs = t_pts + t_stim
    curve = np.zeros_like(t_pts, dtype=np.float64)

    valid = t_abs > 0
    if not np.any(valid):
        return curve

    t_v = t_abs[valid]
    n = len(t_v)

    # --- Proactive PDF (P_A) and CDF (C_A) ---
    P_A = rho_A_t_VEC_fn(t_v - t_A_aff, V_A, theta_A)
    C_A = cum_A_t_vec_fn(t_v - t_A_aff, V_A, theta_A)

    # --- Times for reactive ---
    t1 = np.maximum(t_v - t_stim - t_E_aff, 1e-6)
    t2 = np.maximum(t_v - t_stim - t_E_aff + del_go, 1e-6)

    # is_time_vary=False path: int_phi = t, phi = 1
    # Scalars for CDF/PDF calls
    ABL_arr = np.full(n, float(abl))
    ILD_arr = np.full(n, float(ild))
    rl_arr = np.full(n, rate_lambda)
    T0_arr = np.full(n, T_0)
    thE_arr = np.full(n, theta_E)
    ZE_arr = np.full(n, Z_E)
    rnl_arr = np.full(n, rate_norm_l)

    # int_phi for the various time arguments (is_time_vary=False -> int_phi = t_arg itself)
    int_phi_t_E_g = t_v - t_stim - t_E_aff + del_go  # same as t2 before clamp, but CDF handles t<=0
    int_phi_t2 = t2
    int_phi_t1 = t1
    int_phi_t_e = t1.copy()
    phi_t_e = np.ones(n)

    # --- P_EA_hits_either_bound (for random readout term) ---
    t_cdf_arg = t_v - t_stim - t_E_aff + del_go
    CDF_up = CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn_vec(
        t_cdf_arg, 1, ABL_arr, ILD_arr, rl_arr, T0_arr, thE_arr, ZE_arr,
        int_phi_t_E_g, rnl_arr, is_norm, is_time_vary, K_max,
    )
    CDF_down = CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn_vec(
        t_cdf_arg, -1, ABL_arr, ILD_arr, rl_arr, T0_arr, thE_arr, ZE_arr,
        int_phi_t_E_g, rnl_arr, is_norm, is_time_vary, K_max,
    )
    P_EA_hits_either = CDF_up + CDF_down
    random_readout_if_EA_survives = 0.5 * (1.0 - P_EA_hits_either)

    # --- P_E_plus_cum (CDF difference) ---
    CDF_t2 = CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn_vec(
        t2, 1, ABL_arr, ILD_arr, rl_arr, T0_arr, thE_arr, ZE_arr,
        int_phi_t2, rnl_arr, is_norm, is_time_vary, K_max,
    )
    CDF_t1 = CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn_vec(
        t1, 1, ABL_arr, ILD_arr, rl_arr, T0_arr, thE_arr, ZE_arr,
        int_phi_t1, rnl_arr, is_norm, is_time_vary, K_max,
    )
    CDF_t2_m = CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn_vec(
        t2, -1, ABL_arr, ILD_arr, rl_arr, T0_arr, thE_arr, ZE_arr,
        int_phi_t2, rnl_arr, is_norm, is_time_vary, K_max,
    )
    CDF_t1_m = CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn_vec(
        t1, -1, ABL_arr, ILD_arr, rl_arr, T0_arr, thE_arr, ZE_arr,
        int_phi_t1, rnl_arr, is_norm, is_time_vary, K_max,
    )
    P_E_plus_cum_up = CDF_t2 - CDF_t1
    P_E_plus_cum_down = CDF_t2_m - CDF_t1_m

    # --- P_E_plus (reactive PDF for each bound) ---
    t_rho_arg = t_v - t_stim - t_E_aff
    P_E_plus_up = rho_E_minus_small_t_NORM_rate_norm_time_varying_fn_vec(
        t_rho_arg, 1, ABL_arr, ILD_arr, rl_arr, T0_arr, thE_arr, ZE_arr,
        phi_t_e, int_phi_t_e, rnl_arr, is_norm, is_time_vary, K_max,
    )
    P_E_plus_down = rho_E_minus_small_t_NORM_rate_norm_time_varying_fn_vec(
        t_rho_arg, -1, ABL_arr, ILD_arr, rl_arr, T0_arr, thE_arr, ZE_arr,
        phi_t_e, int_phi_t_e, rnl_arr, is_norm, is_time_vary, K_max,
    )

    # --- Lapse mixing ---
    lp = float(np.clip(lapse_prob, 0.0, 1.0))
    lapse_choice_p = 0.5

    P_A_mix = (1.0 - lp) * P_A + lp * lapse_choice_p * lapse_pdf_vec(t_v, beta_lapse)
    C_A_mix = np.clip((1.0 - lp) * C_A + lp * lapse_cdf_vec(t_v, beta_lapse), 0.0, 1.0)

    # --- Combine for bound=+1 (up) ---
    pdf_up = P_A_mix * (random_readout_if_EA_survives + P_E_plus_cum_up) + P_E_plus_up * (1.0 - C_A_mix)
    # --- Combine for bound=-1 (down) ---
    pdf_down = P_A_mix * (random_readout_if_EA_survives + P_E_plus_cum_down) + P_E_plus_down * (1.0 - C_A_mix)

    pdf_total = pdf_up + pdf_down
    # Apply eps/clamp like scalar version
    eps = 1e-50
    pdf_total = np.where(np.isfinite(pdf_total) & (pdf_total > 0), pdf_total, eps)

    # Only keep positive finite values (match scalar: if pdf > 0 and finite)
    good = np.isfinite(pdf_total) & (pdf_total > 0) & (pdf_total > eps)
    curve_valid = np.where(good, pdf_total, 0.0)
    curve[valid] = curve_valid

    return curve


# %%
############ Validate: compare scalar vs vectorized ############
print("\n=== Validation: scalar vs vectorized ===")
rng = np.random.default_rng(42)

# Pick a few sample trials from each ABL
n_test = 5
abl_values_float = fit_df["ABL"].astype(float).to_numpy()
test_rows = []
for abl in supported_abl_values:
    df_abl = fit_df[np.isclose(abl_values_float, float(abl))]
    idxs = rng.choice(len(df_abl), size=min(n_test, len(df_abl)), replace=False)
    test_rows.append(df_abl.iloc[idxs])
test_df = pd.concat(test_rows, ignore_index=True)

max_abs_err = 0.0
max_rel_err = 0.0
all_pass = True

for i, row in test_df.iterrows():
    t_stim = float(row["intended_fix"])
    abl = float(row["ABL"])
    ild = float(row["ILD"])

    curve_scalar = build_theory_curve_for_trial_scalar(t_stim, abl, ild)
    curve_vec = build_theory_curve_for_trial_vectorized(t_stim, abl, ild)

    abs_err = np.max(np.abs(curve_scalar - curve_vec))
    # relative error where scalar > 0
    nonzero = curve_scalar > 1e-50
    if np.any(nonzero):
        rel_err = np.max(np.abs(curve_scalar[nonzero] - curve_vec[nonzero]) / curve_scalar[nonzero])
    else:
        rel_err = 0.0

    max_abs_err = max(max_abs_err, abs_err)
    max_rel_err = max(max_rel_err, rel_err)

    if abs_err > 1e-10:
        print(f"  FAIL row {i}: ABL={abl}, ILD={ild}, t_stim={t_stim:.4f}, max_abs_err={abs_err:.2e}, max_rel_err={rel_err:.2e}")
        # Show where they differ
        diff_idx = np.where(np.abs(curve_scalar - curve_vec) > 1e-10)[0]
        for di in diff_idx[:5]:
            print(f"    t_pts[{di}]={t_pts[di]:.4f}: scalar={curve_scalar[di]:.10e}, vec={curve_vec[di]:.10e}")
        all_pass = False
    else:
        print(f"  OK   row {i}: ABL={abl}, ILD={ild}, t_stim={t_stim:.4f}, max_abs_err={abs_err:.2e}")

print(f"\nMax absolute error across all tests: {max_abs_err:.2e}")
print(f"Max relative error across all tests: {max_rel_err:.2e}")
if all_pass:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED")


# %%
############ Benchmark ############
print("\n=== Benchmark: scalar vs vectorized ===")

# Use first 20 rows for benchmark
bench_df = test_df.head(10)

# Scalar timing
t0 = time.perf_counter()
for _, row in bench_df.iterrows():
    build_theory_curve_for_trial_scalar(
        float(row["intended_fix"]), float(row["ABL"]), float(row["ILD"])
    )
t_scalar = time.perf_counter() - t0

# Vectorized timing
t0 = time.perf_counter()
for _, row in bench_df.iterrows():
    build_theory_curve_for_trial_vectorized(
        float(row["intended_fix"]), float(row["ABL"]), float(row["ILD"])
    )
t_vec = time.perf_counter() - t0

print(f"Scalar:     {t_scalar:.3f} s for {len(bench_df)} curves")
print(f"Vectorized: {t_vec:.3f} s for {len(bench_df)} curves")
print(f"Speedup:    {t_scalar / t_vec:.1f}x")
