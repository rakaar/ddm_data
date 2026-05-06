# %%
"""
Experiment: compare scalar-loop and vectorized alpha-normalized likelihoods.

The vectorized code here is deliberately local to this test file. If it matches
the scalar reference well, the same blocks can be moved into the fitter later.
"""
import os
import pickle
import time

import numpy as np
import pandas as pd
from scipy.special import erf

from time_vary_norm_alpha_utils import (
    CDF_E_minus_small_t_NORM_alpha_time_varying_fn,
    cum_pro_and_reactive_time_vary_alpha_fn,
    rho_E_minus_small_t_NORM_alpha_time_varying_fn,
    up_or_down_RTs_fit_alpha_fn,
)
from time_vary_norm_utils import M, phi, rho_A_t_VEC_fn


# %%
############ Params #############
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

BATCH_NAME = 'LED34'
ANIMAL = 63
DESIRED_BATCHES = ['SD', 'LED34', 'LED6', 'LED8', 'LED7', 'LED34_even']

BATCH_T_TRUNC = {
    'LED34_even': 0.15,
}
DEFAULT_T_TRUNC = 0.3
T_trunc = BATCH_T_TRUNC.get(BATCH_NAME, DEFAULT_T_TRUNC)

N_TRIALS_FOR_TEST = 1000  # Set to None to compare trial likelihoods on all valid RT<1 trials.
RT_GRID = np.linspace(0, 1, 101)
ALPHA_VALUES = np.array([0.0, 0.5, 1.0])
K_max = 10

is_norm = True
is_time_vary = False
phi_params_obj = np.nan


# %%
############ Vectorized alpha-normalized likelihood helpers #############
def Phi_vec(x):
    return 0.5 * (1 + erf(x / np.sqrt(2)))


def cum_A_t_vec(t, V_A, theta_A):
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t, dtype=float)
    valid = t > 0
    t_valid = t[valid]

    out[valid] = (
        Phi_vec(V_A * (t_valid - theta_A / V_A) / np.sqrt(t_valid)) +
        np.exp(2 * V_A * theta_A) *
        Phi_vec(-V_A * (t_valid + theta_A / V_A) / np.sqrt(t_valid))
    )
    return out


def gamma_omega_alpha_vec(ABL, ILD, rate_lambda, T0, theta_E, rate_norm_l, alpha, is_norm=True):
    if not is_norm:
        rate_norm_l = 0
        alpha = 1

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
    omega = r_sum / (T0 * (theta_E ** 2))
    return gamma, omega


def CDF_E_alpha_vec(t, bound, ABL, ILD, rate_lambda, T0, theta_E, Z_E, rate_norm_l, alpha, K_max):
    t_original = np.asarray(t, dtype=float)
    bound = np.asarray(bound)
    v, omega = gamma_omega_alpha_vec(ABL, ILD, rate_lambda, T0, theta_E, rate_norm_l, alpha, is_norm)

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

    exponent_arg = -np.broadcast_to(v, shape) * a * np.broadcast_to(w, shape) - (
        (np.broadcast_to(v, shape) ** 2) * safe_t / 2
    )
    result = np.exp(exponent_arg)

    k_arr = np.arange(K_max + 1)
    t_b = safe_t[..., None]
    v_b = np.broadcast_to(v, shape)[..., None]
    w_b = np.broadcast_to(w, shape)[..., None]
    k_b = k_arr.reshape((1,) * len(shape) + (K_max + 1,))

    r_k = np.where(k_b % 2 == 0, k_b * a + a * w_b, k_b * a + a * (1 - w_b))
    sqrt_t = np.sqrt(t_b)
    term1 = phi(r_k / sqrt_t)
    term2 = M((r_k - v_b * t_b) / sqrt_t) + M((r_k + v_b * t_b) / sqrt_t)
    summation = np.sum(((-1) ** k_b) * term1 * term2, axis=-1)

    out[valid] = (result * summation)[valid]
    return out


def rho_E_alpha_vec(t, bound, ABL, ILD, rate_lambda, T0, theta_E, Z_E, rate_norm_l, alpha, K_max):
    t_original = np.asarray(t, dtype=float)
    bound = np.asarray(bound)
    v, omega = gamma_omega_alpha_vec(ABL, ILD, rate_lambda, T0, theta_E, rate_norm_l, alpha, is_norm)

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
        (1 / a**2) *
        (a**3 / np.sqrt(2 * np.pi * safe_t**3)) *
        np.exp(-v_full * a * w_full - (v_full**2 * safe_t) / 2)
    )

    K_half = int(K_max / 2)
    k_vals = np.linspace(-K_half, K_half, 2 * K_half + 1)
    t_b = safe_t[..., None]
    w_b = w_full[..., None]
    k_b = k_vals.reshape((1,) * len(shape) + (2 * K_half + 1,))

    sum_w_term = w_b + 2 * k_b
    sum_exp_term = np.exp(-(a**2 * (w_b + 2 * k_b)**2) / (2 * t_b))
    sum_result = np.sum(sum_w_term * sum_exp_term, axis=-1)

    density = non_sum_term * sum_result
    density = np.where(density <= 0, 1e-16, density)
    out[valid] = (density * np.broadcast_to(omega, shape))[valid]
    return out


def up_or_down_alpha_vec(t, bound, V_A, theta_A, t_A_aff, t_stim, ABL, ILD,
                         rate_lambda, T0, theta_E, Z_E, t_E_aff, del_go,
                         rate_norm_l, alpha, K_max):
    t = np.asarray(t, dtype=float)
    t_stim = np.asarray(t_stim, dtype=float)

    t1 = np.maximum(t - t_stim - t_E_aff, 1e-6)
    t2 = np.maximum(t - t_stim - t_E_aff + del_go, 1e-6)

    P_A = rho_A_t_VEC_fn(t - t_A_aff, V_A, theta_A)
    P_EA_hits_either_bound = (
        CDF_E_alpha_vec(t - t_stim - t_E_aff + del_go, 1, ABL, ILD, rate_lambda, T0, theta_E, Z_E, rate_norm_l, alpha, K_max) +
        CDF_E_alpha_vec(t - t_stim - t_E_aff + del_go, -1, ABL, ILD, rate_lambda, T0, theta_E, Z_E, rate_norm_l, alpha, K_max)
    )
    random_readout_if_EA_survives = 0.5 * (1 - P_EA_hits_either_bound)

    P_E_plus_cum = (
        CDF_E_alpha_vec(t2, bound, ABL, ILD, rate_lambda, T0, theta_E, Z_E, rate_norm_l, alpha, K_max) -
        CDF_E_alpha_vec(t1, bound, ABL, ILD, rate_lambda, T0, theta_E, Z_E, rate_norm_l, alpha, K_max)
    )
    P_E_plus = rho_E_alpha_vec(
        t - t_stim - t_E_aff, bound, ABL, ILD, rate_lambda, T0, theta_E, Z_E, rate_norm_l, alpha, K_max
    )
    C_A = cum_A_t_vec(t - t_A_aff, V_A, theta_A)

    return P_A * (random_readout_if_EA_survives + P_E_plus_cum) + P_E_plus * (1 - C_A)


def cum_pro_and_reactive_alpha_vec(t, c_A_trunc_time, V_A, theta_A, t_A_aff,
                                   t_stim, ABL, ILD, rate_lambda, T0, theta_E, Z_E,
                                   t_E_aff, rate_norm_l, alpha, K_max):
    t = np.asarray(t, dtype=float)

    c_A = cum_A_t_vec(t - t_A_aff, V_A, theta_A)
    if c_A_trunc_time is not None:
        trunc_denom = 1 - cum_A_t_vec(np.array([c_A_trunc_time - t_A_aff]), V_A, theta_A)[0]
        c_A = np.where(t < c_A_trunc_time, 0, c_A / trunc_denom)

    c_E = (
        CDF_E_alpha_vec(t - t_stim - t_E_aff, 1, ABL, ILD, rate_lambda, T0, theta_E, Z_E, rate_norm_l, alpha, K_max) +
        CDF_E_alpha_vec(t - t_stim - t_E_aff, -1, ABL, ILD, rate_lambda, T0, theta_E, Z_E, rate_norm_l, alpha, K_max)
    )
    return c_A + c_E - c_A * c_E


# %%
############ Load test animal and starting parameters #############
batch_dfs = []
for batch_name in DESIRED_BATCHES:
    csv_path = os.path.join(SCRIPT_DIR, 'batch_csvs', f'batch_{batch_name}_valid_and_aborts.csv')
    if os.path.exists(csv_path):
        batch_dfs.append(pd.read_csv(csv_path))

merged_data = pd.concat(batch_dfs, ignore_index=True)
df_valid = merged_data[
    (merged_data['batch_name'] == BATCH_NAME) &
    (merged_data['animal'] == ANIMAL) &
    (merged_data['success'].isin([1, -1])) &
    (merged_data['RTwrtStim'] < 1)
].copy()

if N_TRIALS_FOR_TEST is not None and len(df_valid) > N_TRIALS_FOR_TEST:
    row_idx = np.linspace(0, len(df_valid) - 1, N_TRIALS_FOR_TEST, dtype=int)
    df_like = df_valid.iloc[row_idx].copy()
else:
    df_like = df_valid.copy()

with open(f'results_{BATCH_NAME}_animal_{ANIMAL}.pkl', 'rb') as f:
    fit_results_data = pickle.load(f)

abort_samples = fit_results_data['vbmc_aborts_results']
norm_samples = fit_results_data['vbmc_norm_tied_results']

V_A = np.mean(abort_samples['V_A_samples'])
theta_A = np.mean(abort_samples['theta_A_samples'])
t_A_aff = np.mean(abort_samples['t_A_aff_samp'])

rate_lambda = np.mean(norm_samples['rate_lambda_samples'])
T0 = np.mean(norm_samples['T_0_samples'])
theta_E = np.mean(norm_samples['theta_E_samples'])
w = np.mean(norm_samples['w_samples'])
Z_E = (w - 0.5) * 2 * theta_E
t_E_aff = np.mean(norm_samples['t_E_aff_samples'])
del_go = np.mean(norm_samples['del_go_samples'])
rate_norm_l = np.mean(norm_samples['rate_norm_l_samples'])

base_params = np.array([rate_lambda, T0, theta_E, w, t_E_aff, del_go, rate_norm_l, 1.0])

print(f"Testing batch={BATCH_NAME}, animal={ANIMAL}, T_trunc={T_trunc}")
print(f"Using {len(df_like)} valid RT<1 trials out of {len(df_valid)} available")
print("Base params:", base_params)


# %%
############ Scalar and vectorized likelihood functions #############
def scalar_loglike(params, df):
    rate_lambda, T0, theta_E, w, t_E_aff, del_go, rate_norm_l, alpha = params
    Z_E = (w - 0.5) * 2 * theta_E

    loglike = 0.0
    for row in df.itertuples(index=False):
        pdf = up_or_down_RTs_fit_alpha_fn(
            row.TotalFixTime, row.choice,
            V_A, theta_A, t_A_aff,
            row.intended_fix, row.ABL, row.ILD, rate_lambda, T0, theta_E, Z_E, t_E_aff, del_go,
            phi_params_obj, rate_norm_l, alpha,
            is_norm, is_time_vary, K_max
        )
        trunc_factor = cum_pro_and_reactive_time_vary_alpha_fn(
            row.intended_fix + 1, T_trunc,
            V_A, theta_A, t_A_aff,
            row.intended_fix, row.ABL, row.ILD, rate_lambda, T0, theta_E, Z_E, t_E_aff,
            phi_params_obj, rate_norm_l, alpha,
            is_norm, is_time_vary, K_max
        ) - cum_pro_and_reactive_time_vary_alpha_fn(
            row.intended_fix, T_trunc,
            V_A, theta_A, t_A_aff,
            row.intended_fix, row.ABL, row.ILD, rate_lambda, T0, theta_E, Z_E, t_E_aff,
            phi_params_obj, rate_norm_l, alpha,
            is_norm, is_time_vary, K_max
        )

        pdf = max(pdf / (trunc_factor + 1e-20), 1e-50)
        loglike += np.log(pdf)

    return loglike


def vectorized_loglike(params, df):
    rate_lambda, T0, theta_E, w, t_E_aff, del_go, rate_norm_l, alpha = params
    Z_E = (w - 0.5) * 2 * theta_E

    total_fix = df['TotalFixTime'].to_numpy(dtype=float)
    t_stim = df['intended_fix'].to_numpy(dtype=float)
    ABL = df['ABL'].to_numpy(dtype=float)
    ILD = df['ILD'].to_numpy(dtype=float)
    choice = df['choice'].to_numpy(dtype=float)

    pdf = up_or_down_alpha_vec(
        total_fix, choice,
        V_A, theta_A, t_A_aff,
        t_stim, ABL, ILD, rate_lambda, T0, theta_E, Z_E, t_E_aff, del_go,
        rate_norm_l, alpha, K_max
    )
    trunc_factor = (
        cum_pro_and_reactive_alpha_vec(
            t_stim + 1, T_trunc,
            V_A, theta_A, t_A_aff,
            t_stim, ABL, ILD, rate_lambda, T0, theta_E, Z_E, t_E_aff,
            rate_norm_l, alpha, K_max
        ) -
        cum_pro_and_reactive_alpha_vec(
            t_stim, T_trunc,
            V_A, theta_A, t_A_aff,
            t_stim, ABL, ILD, rate_lambda, T0, theta_E, Z_E, t_E_aff,
            rate_norm_l, alpha, K_max
        )
    )

    pdf = np.maximum(pdf / (trunc_factor + 1e-20), 1e-50)
    if np.any(~np.isfinite(pdf)):
        raise ValueError("Vectorized likelihood produced non-finite pdf values.")

    return np.sum(np.log(pdf))


# %%
############ Direct time-grid check from 0 to 1 s #############
test_row = df_like.iloc[0]
t_abs_grid = test_row['intended_fix'] + RT_GRID

time_grid_rows = []
for alpha in ALPHA_VALUES:
    scalar_raw = np.array([
        up_or_down_RTs_fit_alpha_fn(
            t, test_row['choice'],
            V_A, theta_A, t_A_aff,
            test_row['intended_fix'], test_row['ABL'], test_row['ILD'],
            rate_lambda, T0, theta_E, Z_E, t_E_aff, del_go,
            phi_params_obj, rate_norm_l, alpha,
            is_norm, is_time_vary, K_max
        )
        for t in t_abs_grid
    ])
    vec_raw = up_or_down_alpha_vec(
        t_abs_grid, test_row['choice'],
        V_A, theta_A, t_A_aff,
        test_row['intended_fix'], test_row['ABL'], test_row['ILD'],
        rate_lambda, T0, theta_E, Z_E, t_E_aff, del_go,
        rate_norm_l, alpha, K_max
    )

    scalar_trunc = (
        cum_pro_and_reactive_time_vary_alpha_fn(
            test_row['intended_fix'] + 1, T_trunc,
            V_A, theta_A, t_A_aff,
            test_row['intended_fix'], test_row['ABL'], test_row['ILD'],
            rate_lambda, T0, theta_E, Z_E, t_E_aff,
            phi_params_obj, rate_norm_l, alpha,
            is_norm, is_time_vary, K_max
        ) -
        cum_pro_and_reactive_time_vary_alpha_fn(
            test_row['intended_fix'], T_trunc,
            V_A, theta_A, t_A_aff,
            test_row['intended_fix'], test_row['ABL'], test_row['ILD'],
            rate_lambda, T0, theta_E, Z_E, t_E_aff,
            phi_params_obj, rate_norm_l, alpha,
            is_norm, is_time_vary, K_max
        )
    )
    vec_trunc = (
        cum_pro_and_reactive_alpha_vec(
            test_row['intended_fix'] + 1, T_trunc,
            V_A, theta_A, t_A_aff,
            test_row['intended_fix'], test_row['ABL'], test_row['ILD'],
            rate_lambda, T0, theta_E, Z_E, t_E_aff,
            rate_norm_l, alpha, K_max
        ) -
        cum_pro_and_reactive_alpha_vec(
            test_row['intended_fix'], T_trunc,
            V_A, theta_A, t_A_aff,
            test_row['intended_fix'], test_row['ABL'], test_row['ILD'],
            rate_lambda, T0, theta_E, Z_E, t_E_aff,
            rate_norm_l, alpha, K_max
        )
    )

    scalar_norm = np.maximum(scalar_raw / (scalar_trunc + 1e-20), 1e-50)
    vec_norm = np.maximum(vec_raw / (vec_trunc + 1e-20), 1e-50)
    raw_abs_diff = np.abs(scalar_raw - vec_raw)
    norm_abs_diff = np.abs(scalar_norm - vec_norm)

    time_grid_rows.append({
        'alpha': alpha,
        'max_raw_abs_diff': np.max(raw_abs_diff),
        'max_raw_rel_diff': np.max(raw_abs_diff / np.maximum(np.abs(scalar_raw), 1e-12)),
        'max_norm_abs_diff': np.max(norm_abs_diff),
        'max_norm_rel_diff': np.max(norm_abs_diff / np.maximum(np.abs(scalar_norm), 1e-12)),
        'scalar_trunc': scalar_trunc,
        'vectorized_trunc': float(np.asarray(vec_trunc)),
        'trunc_abs_diff': abs(scalar_trunc - float(np.asarray(vec_trunc))),
    })

time_grid_df = pd.DataFrame(time_grid_rows)
print("\nRT-grid likelihood-density check, RTwrtStim in [0, 1]:")
print(time_grid_df.to_string(index=False, float_format=lambda x: f"{x:.6g}"))


# %%
############ Trial likelihood check at three alpha values #############
rows = []
for alpha in ALPHA_VALUES:
    params = base_params.copy()
    params[7] = alpha

    start = time.time()
    scalar_ll = scalar_loglike(params, df_like)
    scalar_seconds = time.time() - start

    start = time.time()
    vectorized_ll = vectorized_loglike(params, df_like)
    vectorized_seconds = time.time() - start

    abs_diff = abs(scalar_ll - vectorized_ll)
    rel_diff = abs_diff / max(abs(scalar_ll), 1e-12)
    rows.append({
        'alpha': alpha,
        'scalar_loglike': scalar_ll,
        'vectorized_loglike': vectorized_ll,
        'abs_diff': abs_diff,
        'rel_diff': rel_diff,
        'scalar_seconds': scalar_seconds,
        'vectorized_seconds': vectorized_seconds,
        'speedup': scalar_seconds / vectorized_seconds if vectorized_seconds > 0 else np.inf,
    })

comparison_df = pd.DataFrame(rows)
print("\nTrial likelihood comparison at three alpha values:")
print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
print("\nSummary:")
print(f"  max abs loglike diff = {comparison_df['abs_diff'].max():.6e}")
print(f"  max rel loglike diff = {comparison_df['rel_diff'].max():.6e}")
print(f"  median speedup        = {comparison_df['speedup'].median():.2f}x")
