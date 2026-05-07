# %%
"""
Animal-wise alpha-normalized TIED fit with an ABL/ILD-dependent evidence delay.

This standalone fit adds alpha to the normalized TIED rate model while loading
V_A, theta_A, and t_A_aff from an existing animal-wise result pickle. The
evidence-afferent delay is fitted as:

    delay_ms = bias_ms + c1 * ABL + c2 * |ILD| + c3 * ILD^2
"""
import os
import pickle
import random
from collections import defaultdict

import corner
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.backends.backend_pdf import PdfPages
from pyvbmc import VBMC
from scipy.special import erf
from tqdm.notebook import tqdm

from animal_wise_plotting_utils import (
    calculate_theoretical_curves,
    create_abort_table,
    create_tied_table,
    plot_grand_summary,
    plot_rt_distributions,
    plot_tachometric_curves,
    prepare_simulation_data,
    render_df_to_pdf,
    save_posterior_summary_page,
)
from time_vary_and_norm_alpha_simulators import psiam_tied_data_gen_wrapper_rate_norm_alpha_fn
from time_vary_norm_alpha_utils import (
    cum_pro_and_reactive_time_vary_alpha_fn,
    rho_A_t_fn,
    up_or_down_RTs_fit_alpha_fn,
    up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_alpha_fn,
)
from time_vary_norm_utils import M, phi, rho_A_t_VEC_fn
from vbmc_animal_wise_fit_utils import trapezoidal_logpdf


# %%
############3 Params #############
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

DESIRED_BATCHES = ['SD', 'LED34', 'LED6', 'LED8', 'LED7', 'LED34_even']
batch_dir = os.path.join(SCRIPT_DIR, 'batch_csvs')
output_dir = os.path.join(SCRIPT_DIR, 'NPL_alpha_animal_fits')
os.makedirs(output_dir, exist_ok=True)
DRY_RUN_ONLY = False
TEST_BATCH_ANIMAL_PAIRS = None  # Set to a list like [('LED34', 63)] for smoke tests.
USE_VECTORIZED_LIKELIHOOD = True
SKIP_FINISHED_FITS = True
FINISHED_ALPHA_RESULT_KEY = 'vbmc_norm_alpha_ild2_delay_tied_results'
K_max = 10

BATCH_T_TRUNC = {
    'LED34_even': 0.15,
}
DEFAULT_T_TRUNC = 0.3
T_trunc = DEFAULT_T_TRUNC

N_theory = int(1e3)
N_sim = int(1e6)
dt = 1e-3
N_print = int(N_sim / 5)


# %%
########### Resume helpers ##############
def alpha_result_pkl_path(batch_name, animal):
    return os.path.join(output_dir, f'results_{batch_name}_animal_{animal}_NORM_ALPHA_ILD2_DELAY_FROM_ABORTS.pkl')


def is_finished_alpha_result(pkl_path):
    if not os.path.exists(pkl_path):
        return False

    try:
        with open(pkl_path, 'rb') as f:
            saved_data = pickle.load(f)
    except Exception as exc:
        print(f"WARNING: Could not read existing result pkl; will rerun: {os.path.relpath(pkl_path, SCRIPT_DIR)} ({exc})")
        return False

    if FINISHED_ALPHA_RESULT_KEY not in saved_data:
        print(f"WARNING: Existing result pkl lacks {FINISHED_ALPHA_RESULT_KEY}; will rerun: {os.path.relpath(pkl_path, SCRIPT_DIR)}")
        return False

    alpha_results = saved_data[FINISHED_ALPHA_RESULT_KEY]
    required_keys = [
        'rate_lambda_samples',
        'T_0_samples',
        'theta_E_samples',
        'w_samples',
        'bias_ms_samples',
        'abl_delay_coeff_ms_per_abl_samples',
        'abs_ild_delay_coeff_ms_per_unit_samples',
        'ild2_delay_coeff_ms_per_unit2_samples',
        'del_go_samples',
        'rate_norm_l_samples',
        'alpha_samples',
        'loglike',
    ]
    missing_keys = [key for key in required_keys if key not in alpha_results]
    if missing_keys:
        print(
            f"WARNING: Existing result pkl is incomplete ({missing_keys}); "
            f"will rerun: {os.path.relpath(pkl_path, SCRIPT_DIR)}"
        )
        return False

    if len(alpha_results['alpha_samples']) == 0:
        print(f"WARNING: Existing result pkl has no alpha samples; will rerun: {os.path.relpath(pkl_path, SCRIPT_DIR)}")
        return False

    return True


# %%
###########  Alpha-normalized TIED ##############
observed_delay_condition_pairs = np.empty((0, 3), dtype=float)


def get_t_E_aff_from_abl_ild(
    abl,
    ild,
    bias_ms,
    abl_delay_coeff_ms_per_abl,
    abs_ild_delay_coeff_ms_per_unit,
    ild2_delay_coeff_ms_per_unit2,
):
    abl = np.asarray(abl, dtype=float)
    ild = np.asarray(ild, dtype=float)
    abs_ild = np.abs(ild)
    delay_ms = (
        float(bias_ms)
        + float(abl_delay_coeff_ms_per_abl) * abl
        + float(abs_ild_delay_coeff_ms_per_unit) * abs_ild
        + float(ild2_delay_coeff_ms_per_unit2) * (ild ** 2)
    )
    return delay_ms * 1e-3


def get_delay_ms_from_condition_pairs(
    condition_pairs,
    bias_ms,
    abl_delay_coeff_ms_per_abl,
    abs_ild_delay_coeff_ms_per_unit,
    ild2_delay_coeff_ms_per_unit2,
):
    condition_pairs = np.asarray(condition_pairs, dtype=float)
    return (
        float(bias_ms)
        + float(abl_delay_coeff_ms_per_abl) * condition_pairs[:, 0]
        + float(abs_ild_delay_coeff_ms_per_unit) * condition_pairs[:, 1]
        + float(ild2_delay_coeff_ms_per_unit2) * condition_pairs[:, 2]
    )


def compute_loglike_norm_alpha_fn(
        row, rate_lambda, T_0, theta_E, Z_E,
        bias_ms, abl_delay_coeff_ms_per_abl, abs_ild_delay_coeff_ms_per_unit,
        ild2_delay_coeff_ms_per_unit2,
        del_go, rate_norm_l, alpha, t_trunc):
    rt = row['TotalFixTime']
    t_stim = row['intended_fix']

    ILD = row['ILD']
    ABL = row['ABL']
    choice = row['choice']
    t_E_aff = get_t_E_aff_from_abl_ild(
        ABL,
        ILD,
        bias_ms,
        abl_delay_coeff_ms_per_abl,
        abs_ild_delay_coeff_ms_per_unit,
        ild2_delay_coeff_ms_per_unit2,
    )

    pdf = up_or_down_RTs_fit_alpha_fn(
        rt, choice,
        V_A, theta_A, t_A_aff,
        t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go,
        phi_params_obj, rate_norm_l, alpha,
        is_norm, is_time_vary, K_max
    )

    trunc_factor_p_joint = cum_pro_and_reactive_time_vary_alpha_fn(
        t_stim + 1, t_trunc,
        V_A, theta_A, t_A_aff,
        t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_E_aff,
        phi_params_obj, rate_norm_l, alpha,
        is_norm, is_time_vary, K_max
    ) - cum_pro_and_reactive_time_vary_alpha_fn(
        t_stim, t_trunc,
        V_A, theta_A, t_A_aff,
        t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_E_aff,
        phi_params_obj, rate_norm_l, alpha,
        is_norm, is_time_vary, K_max
    )

    pdf /= (trunc_factor_p_joint + 1e-20)
    pdf = max(pdf, 1e-50)
    if np.isnan(pdf):
        print(f'row["abort_event"] = {row["abort_event"]}')
        print(f'row["RTwrtStim"] = {row["RTwrtStim"]}')
        raise ValueError(f'nan pdf rt = {rt}, t_stim = {t_stim}')
    return np.log(pdf)


def vbmc_norm_alpha_tied_loglike_fn(params):
    (
        rate_lambda,
        T_0,
        theta_E,
        w,
        bias_ms,
        abl_delay_coeff_ms_per_abl,
        abs_ild_delay_coeff_ms_per_unit,
        ild2_delay_coeff_ms_per_unit2,
        del_go,
        rate_norm_l,
        alpha,
    ) = params
    delay_ms_by_condition = get_delay_ms_from_condition_pairs(
        observed_delay_condition_pairs,
        bias_ms,
        abl_delay_coeff_ms_per_abl,
        abs_ild_delay_coeff_ms_per_unit,
        ild2_delay_coeff_ms_per_unit2,
    )
    if np.any(delay_ms_by_condition < 0):
        return np.log(1e-50)

    Z_E = (w - 0.5) * 2 * theta_E
    all_loglike = Parallel(n_jobs=30)(
        delayed(compute_loglike_norm_alpha_fn)(
            row, rate_lambda, T_0, theta_E, Z_E,
            bias_ms, abl_delay_coeff_ms_per_abl, abs_ild_delay_coeff_ms_per_unit,
            ild2_delay_coeff_ms_per_unit2,
            del_go, rate_norm_l, alpha, T_trunc
        )
        for _, row in df_valid_animal_less_than_1.iterrows()
    )
    return np.sum(all_loglike)


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


def gamma_omega_alpha_vec(ABL, ILD, rate_lambda, T_0, theta_E, rate_norm_l, alpha):
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
    omega = r_sum / (T_0 * (theta_E ** 2))
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
    exponent_arg = -v_full * a * w_full - ((v_full ** 2) * safe_t / 2)
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
                         rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go,
                         rate_norm_l, alpha):
    t = np.asarray(t, dtype=float)
    t_stim = np.asarray(t_stim, dtype=float)

    t1 = np.maximum(t - t_stim - t_E_aff, 1e-6)
    t2 = np.maximum(t - t_stim - t_E_aff + del_go, 1e-6)

    P_A = rho_A_t_VEC_fn(t - t_A_aff, V_A, theta_A)
    P_EA_hits_either_bound = (
        CDF_E_alpha_vec(t - t_stim - t_E_aff + del_go, 1, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, rate_norm_l, alpha) +
        CDF_E_alpha_vec(t - t_stim - t_E_aff + del_go, -1, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, rate_norm_l, alpha)
    )
    random_readout_if_EA_survives = 0.5 * (1 - P_EA_hits_either_bound)

    P_E_plus_cum = (
        CDF_E_alpha_vec(t2, bound, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, rate_norm_l, alpha) -
        CDF_E_alpha_vec(t1, bound, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, rate_norm_l, alpha)
    )
    P_E_plus = rho_E_alpha_vec(
        t - t_stim - t_E_aff, bound, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, rate_norm_l, alpha
    )
    C_A = cum_A_t_vec(t - t_A_aff, V_A, theta_A)

    return P_A * (random_readout_if_EA_survives + P_E_plus_cum) + P_E_plus * (1 - C_A)


def cum_pro_and_reactive_alpha_vec(t, c_A_trunc_time, V_A, theta_A, t_A_aff,
                                   t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E,
                                   t_E_aff, rate_norm_l, alpha):
    t = np.asarray(t, dtype=float)

    c_A = cum_A_t_vec(t - t_A_aff, V_A, theta_A)
    if c_A_trunc_time is not None:
        trunc_denom = 1 - cum_A_t_vec(np.array([c_A_trunc_time - t_A_aff]), V_A, theta_A)[0]
        c_A = np.where(t < c_A_trunc_time, 0, c_A / trunc_denom)

    c_E = (
        CDF_E_alpha_vec(t - t_stim - t_E_aff, 1, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, rate_norm_l, alpha) +
        CDF_E_alpha_vec(t - t_stim - t_E_aff, -1, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, rate_norm_l, alpha)
    )
    return c_A + c_E - c_A * c_E


def vbmc_norm_alpha_tied_loglike_vec_fn(params):
    if is_time_vary:
        return vbmc_norm_alpha_tied_loglike_fn(params)

    (
        rate_lambda,
        T_0,
        theta_E,
        w,
        bias_ms,
        abl_delay_coeff_ms_per_abl,
        abs_ild_delay_coeff_ms_per_unit,
        ild2_delay_coeff_ms_per_unit2,
        del_go,
        rate_norm_l,
        alpha,
    ) = params
    delay_ms_by_condition = get_delay_ms_from_condition_pairs(
        observed_delay_condition_pairs,
        bias_ms,
        abl_delay_coeff_ms_per_abl,
        abs_ild_delay_coeff_ms_per_unit,
        ild2_delay_coeff_ms_per_unit2,
    )
    if np.any(delay_ms_by_condition < 0):
        return np.log(1e-50)

    Z_E = (w - 0.5) * 2 * theta_E

    total_fix = df_valid_animal_less_than_1['TotalFixTime'].to_numpy(dtype=float)
    t_stim = df_valid_animal_less_than_1['intended_fix'].to_numpy(dtype=float)
    ABL = df_valid_animal_less_than_1['ABL'].to_numpy(dtype=float)
    ILD = df_valid_animal_less_than_1['ILD'].to_numpy(dtype=float)
    choice = df_valid_animal_less_than_1['choice'].to_numpy(dtype=float)
    t_E_aff = get_t_E_aff_from_abl_ild(
        ABL,
        ILD,
        bias_ms,
        abl_delay_coeff_ms_per_abl,
        abs_ild_delay_coeff_ms_per_unit,
        ild2_delay_coeff_ms_per_unit2,
    )

    pdf = up_or_down_alpha_vec(
        total_fix, choice,
        V_A, theta_A, t_A_aff,
        t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go,
        rate_norm_l, alpha
    )
    trunc_factor = (
        cum_pro_and_reactive_alpha_vec(
            t_stim + 1, T_trunc,
            V_A, theta_A, t_A_aff,
            t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_E_aff,
            rate_norm_l, alpha
        ) -
        cum_pro_and_reactive_alpha_vec(
            t_stim, T_trunc,
            V_A, theta_A, t_A_aff,
            t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_E_aff,
            rate_norm_l, alpha
        )
    )

    pdf = np.maximum(pdf / (trunc_factor + 1e-20), 1e-50)
    if np.any(~np.isfinite(pdf)):
        raise ValueError("Vectorized alpha likelihood produced non-finite pdf values.")

    return np.sum(np.log(pdf))


def vbmc_norm_alpha_tied_active_loglike_fn(params):
    if USE_VECTORIZED_LIKELIHOOD:
        return vbmc_norm_alpha_tied_loglike_vec_fn(params)
    return vbmc_norm_alpha_tied_loglike_fn(params)


def vbmc_prior_norm_alpha_tied_fn(params):
    (
        rate_lambda,
        T_0,
        theta_E,
        w,
        bias_ms,
        abl_delay_coeff_ms_per_abl,
        abs_ild_delay_coeff_ms_per_unit,
        ild2_delay_coeff_ms_per_unit2,
        del_go,
        rate_norm_l,
        alpha,
    ) = params

    rate_lambda_logpdf = trapezoidal_logpdf(
        rate_lambda,
        norm_tied_rate_lambda_bounds[0],
        norm_tied_rate_lambda_plausible_bounds[0],
        norm_tied_rate_lambda_plausible_bounds[1],
        norm_tied_rate_lambda_bounds[1]
    )

    T_0_logpdf = trapezoidal_logpdf(
        T_0,
        norm_tied_T_0_bounds[0],
        norm_tied_T_0_plausible_bounds[0],
        norm_tied_T_0_plausible_bounds[1],
        norm_tied_T_0_bounds[1]
    )

    theta_E_logpdf = trapezoidal_logpdf(
        theta_E,
        norm_tied_theta_E_bounds[0],
        norm_tied_theta_E_plausible_bounds[0],
        norm_tied_theta_E_plausible_bounds[1],
        norm_tied_theta_E_bounds[1]
    )

    w_logpdf = trapezoidal_logpdf(
        w,
        norm_tied_w_bounds[0],
        norm_tied_w_plausible_bounds[0],
        norm_tied_w_plausible_bounds[1],
        norm_tied_w_bounds[1]
    )

    bias_ms_logpdf = trapezoidal_logpdf(
        bias_ms,
        norm_tied_bias_ms_bounds[0],
        norm_tied_bias_ms_plausible_bounds[0],
        norm_tied_bias_ms_plausible_bounds[1],
        norm_tied_bias_ms_bounds[1]
    )

    abl_delay_coeff_logpdf = trapezoidal_logpdf(
        abl_delay_coeff_ms_per_abl,
        norm_tied_abl_delay_coeff_ms_per_abl_bounds[0],
        norm_tied_abl_delay_coeff_ms_per_abl_plausible_bounds[0],
        norm_tied_abl_delay_coeff_ms_per_abl_plausible_bounds[1],
        norm_tied_abl_delay_coeff_ms_per_abl_bounds[1]
    )

    abs_ild_delay_coeff_logpdf = trapezoidal_logpdf(
        abs_ild_delay_coeff_ms_per_unit,
        norm_tied_abs_ild_delay_coeff_ms_per_unit_bounds[0],
        norm_tied_abs_ild_delay_coeff_ms_per_unit_plausible_bounds[0],
        norm_tied_abs_ild_delay_coeff_ms_per_unit_plausible_bounds[1],
        norm_tied_abs_ild_delay_coeff_ms_per_unit_bounds[1]
    )

    ild2_delay_coeff_logpdf = trapezoidal_logpdf(
        ild2_delay_coeff_ms_per_unit2,
        norm_tied_ild2_delay_coeff_ms_per_unit2_bounds[0],
        norm_tied_ild2_delay_coeff_ms_per_unit2_plausible_bounds[0],
        norm_tied_ild2_delay_coeff_ms_per_unit2_plausible_bounds[1],
        norm_tied_ild2_delay_coeff_ms_per_unit2_bounds[1]
    )

    del_go_logpdf = trapezoidal_logpdf(
        del_go,
        norm_tied_del_go_bounds[0],
        norm_tied_del_go_plausible_bounds[0],
        norm_tied_del_go_plausible_bounds[1],
        norm_tied_del_go_bounds[1]
    )

    rate_norm_l_logpdf = trapezoidal_logpdf(
        rate_norm_l,
        norm_tied_rate_norm_bounds[0],
        norm_tied_rate_norm_plausible_bounds[0],
        norm_tied_rate_norm_plausible_bounds[1],
        norm_tied_rate_norm_bounds[1]
    )
    alpha_logpdf = trapezoidal_logpdf(
        alpha,
        norm_tied_alpha_bounds[0],
        norm_tied_alpha_plausible_bounds[0],
        norm_tied_alpha_plausible_bounds[1],
        norm_tied_alpha_bounds[1]
    )

    return (
        rate_lambda_logpdf +
        T_0_logpdf +
        theta_E_logpdf +
        w_logpdf +
        bias_ms_logpdf +
        abl_delay_coeff_logpdf +
        abs_ild_delay_coeff_logpdf +
        ild2_delay_coeff_logpdf +
        del_go_logpdf +
        rate_norm_l_logpdf +
        alpha_logpdf
    )


def vbmc_norm_alpha_tied_joint_fn(params):
    priors = vbmc_prior_norm_alpha_tied_fn(params)
    loglike = vbmc_norm_alpha_tied_active_loglike_fn(params)

    return priors + loglike


norm_tied_rate_lambda_bounds = [0.5, 5]
norm_tied_T_0_bounds = [20e-3, 800e-3]
norm_tied_theta_E_bounds = [1, 15]
norm_tied_w_bounds = [0.3, 0.7]
norm_tied_bias_ms_bounds = [10, 200]
norm_tied_abl_delay_coeff_ms_per_abl_bounds = [-1, 0.5]
norm_tied_abs_ild_delay_coeff_ms_per_unit_bounds = [-2, 0.5]
norm_tied_ild2_delay_coeff_ms_per_unit2_bounds = [-1, 0.1]
norm_tied_del_go_bounds = [0, 0.2]
norm_tied_rate_norm_bounds = [0, 2]
norm_tied_alpha_bounds = [0, 2]

norm_tied_rate_lambda_plausible_bounds = [1, 3]
norm_tied_T_0_plausible_bounds = [90e-3, 400e-3]
norm_tied_theta_E_plausible_bounds = [1.5, 10]
norm_tied_w_plausible_bounds = [0.4, 0.6]
norm_tied_bias_ms_plausible_bounds = [50, 150]
norm_tied_abl_delay_coeff_ms_per_abl_plausible_bounds = [-0.75, -0.1]
norm_tied_abs_ild_delay_coeff_ms_per_unit_plausible_bounds = [-1, -0.1]
norm_tied_ild2_delay_coeff_ms_per_unit2_plausible_bounds = [-0.1, 0.05]
norm_tied_del_go_plausible_bounds = [0.05, 0.15]
norm_tied_rate_norm_plausible_bounds = [0.8, 0.99]
norm_tied_alpha_plausible_bounds = [0.5, 1.5]

norm_tied_lb = np.array([
    norm_tied_rate_lambda_bounds[0],
    norm_tied_T_0_bounds[0],
    norm_tied_theta_E_bounds[0],
    norm_tied_w_bounds[0],
    norm_tied_bias_ms_bounds[0],
    norm_tied_abl_delay_coeff_ms_per_abl_bounds[0],
    norm_tied_abs_ild_delay_coeff_ms_per_unit_bounds[0],
    norm_tied_ild2_delay_coeff_ms_per_unit2_bounds[0],
    norm_tied_del_go_bounds[0],
    norm_tied_rate_norm_bounds[0],
    norm_tied_alpha_bounds[0]
])

norm_tied_ub = np.array([
    norm_tied_rate_lambda_bounds[1],
    norm_tied_T_0_bounds[1],
    norm_tied_theta_E_bounds[1],
    norm_tied_w_bounds[1],
    norm_tied_bias_ms_bounds[1],
    norm_tied_abl_delay_coeff_ms_per_abl_bounds[1],
    norm_tied_abs_ild_delay_coeff_ms_per_unit_bounds[1],
    norm_tied_ild2_delay_coeff_ms_per_unit2_bounds[1],
    norm_tied_del_go_bounds[1],
    norm_tied_rate_norm_bounds[1],
    norm_tied_alpha_bounds[1]
])

norm_tied_plb = np.array([
    norm_tied_rate_lambda_plausible_bounds[0],
    norm_tied_T_0_plausible_bounds[0],
    norm_tied_theta_E_plausible_bounds[0],
    norm_tied_w_plausible_bounds[0],
    norm_tied_bias_ms_plausible_bounds[0],
    norm_tied_abl_delay_coeff_ms_per_abl_plausible_bounds[0],
    norm_tied_abs_ild_delay_coeff_ms_per_unit_plausible_bounds[0],
    norm_tied_ild2_delay_coeff_ms_per_unit2_plausible_bounds[0],
    norm_tied_del_go_plausible_bounds[0],
    norm_tied_rate_norm_plausible_bounds[0],
    norm_tied_alpha_plausible_bounds[0]
])

norm_tied_pub = np.array([
    norm_tied_rate_lambda_plausible_bounds[1],
    norm_tied_T_0_plausible_bounds[1],
    norm_tied_theta_E_plausible_bounds[1],
    norm_tied_w_plausible_bounds[1],
    norm_tied_bias_ms_plausible_bounds[1],
    norm_tied_abl_delay_coeff_ms_per_abl_plausible_bounds[1],
    norm_tied_abs_ild_delay_coeff_ms_per_unit_plausible_bounds[1],
    norm_tied_ild2_delay_coeff_ms_per_unit2_plausible_bounds[1],
    norm_tied_del_go_plausible_bounds[1],
    norm_tied_rate_norm_plausible_bounds[1],
    norm_tied_alpha_plausible_bounds[1]
])


# %%
### Read batch CSVs and get batch-animal pairs ###
batch_dfs = []
for batch_name in DESIRED_BATCHES:
    csv_path = os.path.join(batch_dir, f'batch_{batch_name}_valid_and_aborts.csv')
    if os.path.exists(csv_path):
        batch_dfs.append(pd.read_csv(csv_path))
    else:
        print(f"WARNING: Missing batch CSV: {csv_path}")

if len(batch_dfs) == 0:
    raise FileNotFoundError(f"No batch CSVs found in {batch_dir} for DESIRED_BATCHES={DESIRED_BATCHES}")

merged_data = pd.concat(batch_dfs, ignore_index=True)

if 'timed_fix' in merged_data.columns and 'TotalFixTime' not in merged_data.columns:
    merged_data.loc[:, 'RTwrtStim'] = merged_data['timed_fix'] - merged_data['intended_fix']
    merged_data = merged_data.rename(columns={'timed_fix': 'TotalFixTime'})

merged_data = merged_data[~((merged_data['RTwrtStim'].isna()) & (merged_data['abort_event'] == 3))].copy()

if 'choice' not in merged_data.columns:
    if 'response_poke' in merged_data.columns:
        mask_nan = merged_data['response_poke'].isna()
        mask_success_1 = (merged_data['success'] == 1)
        mask_success_neg1 = (merged_data['success'] == -1)
        mask_ild_pos = (merged_data['ILD'] > 0)
        mask_ild_neg = (merged_data['ILD'] < 0)

        merged_data.loc[mask_nan & mask_success_1 & mask_ild_pos, 'response_poke'] = 3
        merged_data.loc[mask_nan & mask_success_1 & mask_ild_neg, 'response_poke'] = 2

        merged_data.loc[mask_nan & mask_success_neg1 & mask_ild_pos, 'response_poke'] = 2
        merged_data.loc[mask_nan & mask_success_neg1 & mask_ild_neg, 'response_poke'] = 3

        merged_data['choice'] = merged_data['response_poke'].apply(
            lambda x: 1 if x == 3 else (-1 if x == 2 else random.choice([1, -1]))
        )
    else:
        raise KeyError("No `choice` or `response_poke` column found in merged batch CSV data.")

if 'accuracy' not in merged_data.columns:
    merged_data['accuracy'] = (merged_data['ILD'] * merged_data['choice']).apply(lambda x: 1 if x > 0 else 0)

merged_data['abs_ILD'] = merged_data['ILD'].abs()

df_valid_and_aborts = merged_data[
    (merged_data['success'].isin([1, -1])) |
    (merged_data['abort_event'] == 3)
].copy()

df_aborts = df_valid_and_aborts[df_valid_and_aborts['abort_event'] == 3]
merged_valid = df_valid_and_aborts[df_valid_and_aborts['success'].isin([1, -1])].copy()
batch_animal_pairs = sorted(list(map(tuple, merged_valid[['batch_name', 'animal']].drop_duplicates().values)))

if TEST_BATCH_ANIMAL_PAIRS is not None:
    requested_pairs = set(TEST_BATCH_ANIMAL_PAIRS)
    batch_animal_pairs = [pair for pair in batch_animal_pairs if pair in requested_pairs]
    missing_requested_pairs = sorted(requested_pairs - set(batch_animal_pairs))
    if missing_requested_pairs:
        print(f"WARNING: Requested test pairs were not found in the batch CSVs: {missing_requested_pairs}")
    print(f"TEST_BATCH_ANIMAL_PAIRS active: fitting only {batch_animal_pairs}")

print('####################################')
print(f'Default Aborts Truncation Time: {DEFAULT_T_TRUNC}')
print(f'Batch-specific Aborts Truncation Times: {BATCH_T_TRUNC}')
print(f'Use vectorized likelihood: {USE_VECTORIZED_LIKELIHOOD}')
print(f'Skip finished fits: {SKIP_FINISHED_FITS}')
print('####################################')

print(f"Found {len(batch_animal_pairs)} batch-animal pairs from {len(set(p[0] for p in batch_animal_pairs))} batches:")
if batch_animal_pairs:
    batch_to_animals = defaultdict(list)
    for batch_name, animal in batch_animal_pairs:
        animal_str = str(animal)
        if animal_str not in batch_to_animals[batch_name]:
            batch_to_animals[batch_name].append(animal_str)

    max_batch_len = max(len(b) for b in batch_to_animals.keys())
    animal_strings = {b: ', '.join(sorted(a)) for b, a in batch_to_animals.items()}
    max_animals_len = max(len(s) for s in animal_strings.values())

    print(f"{'Batch':<{max_batch_len}}  {'Animals'}")
    print(f"{'=' * max_batch_len}  {'=' * max_animals_len}")
    for batch_name in sorted(animal_strings.keys()):
        print(f"{batch_name:<{max_batch_len}}  {animal_strings[batch_name]}")

missing_abort_pkls = []
for batch_name, animal in batch_animal_pairs:
    pkl_file = f'results_{batch_name}_animal_{animal}.pkl'
    if not os.path.exists(pkl_file):
        missing_abort_pkls.append(pkl_file)

if missing_abort_pkls:
    print("WARNING: Missing abort-source pickle files:")
    for pkl_file in missing_abort_pkls:
        print(f"  {pkl_file}")
else:
    print("All discovered batch-animal pairs have matching abort-source pickle files.")

if SKIP_FINISHED_FITS:
    finished_pairs = [
        (batch_name, animal)
        for batch_name, animal in batch_animal_pairs
        if is_finished_alpha_result(alpha_result_pkl_path(batch_name, animal))
    ]
    print(f"Found {len(finished_pairs)} finished alpha result pickle files that will be skipped.")

if DRY_RUN_ONLY:
    raise SystemExit("DRY_RUN_ONLY=True: batch discovery completed; skipping VBMC fits.")


# %%
for batch_name, animal in batch_animal_pairs:
    T_trunc = BATCH_T_TRUNC.get(batch_name, DEFAULT_T_TRUNC)
    pkl_filename = alpha_result_pkl_path(batch_name, animal)
    if SKIP_FINISHED_FITS and is_finished_alpha_result(pkl_filename):
        print(f"Skipping finished fit: {batch_name}, animal {animal} ({os.path.relpath(pkl_filename, SCRIPT_DIR)})")
        continue

    df_all_trials_animal = df_valid_and_aborts[
        (df_valid_and_aborts['batch_name'] == batch_name) &
        (df_valid_and_aborts['animal'] == animal)
    ].copy()
    df_aborts_animal = df_aborts[
        (df_aborts['batch_name'] == batch_name) &
        (df_aborts['animal'] == animal)
    ].copy()

    print(f'Batch: {batch_name}, sample animal: {animal}')
    print(f'Using T_trunc = {T_trunc:.3f}')
    pdf_filename = os.path.join(output_dir, f'results_{batch_name}_animal_{animal}_NORM_ALPHA_ILD2_DELAY_FROM_ABORTS.pdf')
    pdf = PdfPages(pdf_filename)

    fig_text = plt.figure(figsize=(8.5, 11))
    fig_text.clf()
    fig_text.text(0.1, 0.9, "Alpha-Normalized TIED With ILD2 Delay From Loaded Abort Parameters", fontsize=18, weight='bold')
    fig_text.text(0.1, 0.8, f"Batch Name: {batch_name}", fontsize=14)
    fig_text.text(0.1, 0.75, f"Animal ID: {animal}", fontsize=14)
    fig_text.text(0.1, 0.68, "Abort params loaded from existing result pickle", fontsize=12)
    fig_text.text(0.1, 0.63, f"T_trunc = {T_trunc:.3f}", fontsize=12)
    fig_text.text(0.1, 0.58, "delay_ms = bias + c1*ABL + c2*|ILD| + c3*ILD^2", fontsize=12)
    fig_text.gca().axis("off")
    pdf.savefig(fig_text, bbox_inches='tight')
    plt.close(fig_text)

    # %%
    ####################################################
    ########### Load Abort Parameters ##################
    ####################################################
    print("\n### Loading Abort Parameters from Pickle ###")

    pkl_file = f'results_{batch_name}_animal_{animal}.pkl'
    try:
        with open(pkl_file, 'rb') as f:
            fit_results_data = pickle.load(f)

        abort_keyname = "vbmc_aborts_results"
        abort_samples = fit_results_data[abort_keyname]

        V_A = np.mean(abort_samples['V_A_samples'])
        theta_A = np.mean(abort_samples['theta_A_samples'])
        t_A_aff = np.mean(abort_samples['t_A_aff_samp'])
        vbmc_aborts_results = abort_samples

        print("Loaded abort parameters:")
        print(f"  V_A = {V_A:.4f}")
        print(f"  theta_A = {theta_A:.4f}")
        print(f"  t_A_aff = {t_A_aff:.4f}")

        fig_text = plt.figure(figsize=(8.5, 11))
        fig_text.clf()
        fig_text.text(0.1, 0.9, f"Abort Parameters (Loaded from {pkl_file})", fontsize=16, weight='bold')
        fig_text.text(0.1, 0.8, f"V_A: {V_A:.4f}", fontsize=12)
        fig_text.text(0.1, 0.75, f"theta_A: {theta_A:.4f}", fontsize=12)
        fig_text.text(0.1, 0.7, f"t_A_aff: {t_A_aff:.4f}", fontsize=12)
        fig_text.gca().axis("off")
        pdf.savefig(fig_text, bbox_inches='tight')
        plt.close(fig_text)

    except FileNotFoundError:
        print(f"ERROR: Pickle file {pkl_file} not found. Please run abort fitting first.")
        pdf.close()
        continue
    except KeyError as e:
        print(f"ERROR: Missing key {e} in pickle file {pkl_file}")
        pdf.close()
        continue

    # %%
    ########################################################
    ########## Alpha-normalized model ######################
    ########################################################
    is_norm = True
    is_time_vary = False
    phi_params_obj = np.nan

    df_valid_animal = df_all_trials_animal[df_all_trials_animal['success'].isin([1, -1])].copy()
    df_valid_animal_less_than_1 = df_valid_animal[df_valid_animal['RTwrtStim'] < 1].copy()
    observed_delay_condition_pairs = (
        df_valid_animal_less_than_1
        .assign(ILD_sq=lambda df: df['ILD'].astype(float) ** 2)
        [['ABL', 'abs_ILD', 'ILD_sq']]
        .drop_duplicates()
        .sort_values(['ABL', 'abs_ILD', 'ILD_sq'])
        .to_numpy(dtype=float)
    )
    print(f"Observed delay conditions used for fit: {observed_delay_condition_pairs.tolist()}")

    ABL_arr = np.sort(df_all_trials_animal['ABL'].unique())
    ILD_arr = np.sort(df_all_trials_animal['ILD'].unique())

    norm_tied_keyname = "vbmc_norm_tied_results"
    norm_tied_start_keys = [
        'rate_lambda_samples',
        'T_0_samples',
        'theta_E_samples',
        'w_samples',
        'del_go_samples',
        'rate_norm_l_samples'
    ]

    if (
            norm_tied_keyname in fit_results_data and
            all(key in fit_results_data[norm_tied_keyname] for key in norm_tied_start_keys)):
        norm_tied_samples = fit_results_data[norm_tied_keyname]
        x_0 = np.array([
            np.mean(norm_tied_samples['rate_lambda_samples']),
            np.mean(norm_tied_samples['T_0_samples']),
            np.mean(norm_tied_samples['theta_E_samples']),
            np.mean(norm_tied_samples['w_samples']),
            100.0,
            -0.425,
            -0.55,
            -0.025,
            np.mean(norm_tied_samples['del_go_samples']),
            np.mean(norm_tied_samples['rate_norm_l_samples']),
            1.0
        ])
        print("Initializing shared parameters from existing normalized TIED posterior mean.")
    else:
        x_0 = np.array([
            2.3,
            100 * 1e-3,
            3,
            0.51,
            100.0,
            -0.425,
            -0.55,
            -0.025,
            0.13,
            0.95,
            1.0
        ])
        print("No normalized TIED posterior found in pickle; using generic plausible initialization.")

    plausible_eps = 1e-6 * (norm_tied_pub - norm_tied_plb)
    x_0 = np.clip(x_0, norm_tied_plb + plausible_eps, norm_tied_pub - plausible_eps)

    print("Initial alpha-normalized parameters:")
    print(f"  rate_lambda = {x_0[0]:.5f}")
    print(f"  T_0         = {1e3*x_0[1]:.5f} ms")
    print(f"  theta_E     = {x_0[2]:.5f}")
    print(f"  w           = {x_0[3]:.5f}")
    print(f"  bias_ms     = {x_0[4]:.5f}")
    print(f"  ABL coeff   = {x_0[5]:.5f} ms/ABL")
    print(f"  |ILD| coeff = {x_0[6]:.5f} ms/unit")
    print(f"  ILD^2 coeff = {x_0[7]:.5f} ms/unit^2")
    print(f"  del_go      = {x_0[8]:.5f}")
    print(f"  rate_norm_l = {x_0[9]:.5f}")
    print(f"  alpha       = {x_0[10]:.5f}")

    vbmc = VBMC(
        vbmc_norm_alpha_tied_joint_fn,
        x_0,
        norm_tied_lb,
        norm_tied_ub,
        norm_tied_plb,
        norm_tied_pub,
        options={'display': 'on', 'max_fun_evals': 200 * (2 + 11)}
    )
    vp, results = vbmc.optimize()
    vbmc.save(
        os.path.join(output_dir, f'vbmc_PKL_file_norm_alpha_ild2_delay_tied_results_batch_{batch_name}_animal_{animal}_FROM_ABORTS.pkl'),
        overwrite=True
    )

    vp_samples = vp.sample(int(1e5))[0]
    vp_samples[:, 1] *= 1e3
    param_labels = [
        r'$\lambda$',
        r'$T_0$ (ms)',
        r'$\theta_E$',
        r'$w$',
        'bias (ms)',
        'ABL coeff (ms/ABL)',
        '|ILD| coeff (ms/unit)',
        'ILD^2 coeff (ms/unit^2)',
        r'$\Delta_{go}$',
        r'rate_norm',
        r'$\alpha$'
    ]
    percentiles = np.percentile(vp_samples, [1, 99], axis=0)
    _ranges = [(percentiles[0, i], percentiles[1, i]) for i in range(vp_samples.shape[1])]

    norm_tied_corner_fig = corner.corner(
        vp_samples,
        labels=param_labels,
        show_titles=True,
        quantiles=[0.025, 0.5, 0.975],
        range=_ranges,
        title_fmt=".3f"
    )
    norm_tied_corner_fig.suptitle(f'Alpha-Normalized Tied ILD2-Delay Posterior (Animal: {animal})', y=1.02)
    vp_samples[:, 1] /= 1e3

    rate_lambda = vp_samples[:, 0].mean()
    T_0 = vp_samples[:, 1].mean()
    theta_E = vp_samples[:, 2].mean()
    w = vp_samples[:, 3].mean()
    Z_E = (w - 0.5) * 2 * theta_E
    bias_ms = vp_samples[:, 4].mean()
    abl_delay_coeff_ms_per_abl = vp_samples[:, 5].mean()
    abs_ild_delay_coeff_ms_per_unit = vp_samples[:, 6].mean()
    ild2_delay_coeff_ms_per_unit2 = vp_samples[:, 7].mean()
    del_go = vp_samples[:, 8].mean()
    rate_norm_l = vp_samples[:, 9].mean()
    alpha = vp_samples[:, 10].mean()

    print("Posterior Means:")
    print(f"rate_lambda  = {rate_lambda:.5f}")
    print(f"T_0 (ms)      = {1e3*T_0:.5f}")
    print(f"theta_E       = {theta_E:.5f}")
    print(f"Z_E           = {Z_E:.5f}")
    print(f"bias_ms       = {bias_ms:.5f}")
    print(f"ABL coeff     = {abl_delay_coeff_ms_per_abl:.5f} ms/ABL")
    print(f"|ILD| coeff   = {abs_ild_delay_coeff_ms_per_unit:.5f} ms/unit")
    print(f"ILD^2 coeff   = {ild2_delay_coeff_ms_per_unit2:.5f} ms/unit^2")
    print(f"del_go        = {del_go:.5f}")
    print(f"rate_norm_l   = {rate_norm_l:.5f}")
    print(f"alpha         = {alpha:.5f}")

    norm_alpha_tied_loglike = vbmc_norm_alpha_tied_active_loglike_fn([
        rate_lambda, T_0, theta_E, w,
        bias_ms, abl_delay_coeff_ms_per_abl, abs_ild_delay_coeff_ms_per_unit,
        ild2_delay_coeff_ms_per_unit2,
        del_go, rate_norm_l, alpha
    ])
    save_posterior_summary_page(
        pdf_pages=pdf,
        title=f'Alpha-Normalized Tied ILD2-Delay Model - Posterior Means ({animal})',
        posterior_means=pd.Series({
            'rate_lambda': rate_lambda,
            'T_0': 1e3*T_0,
            'theta_E': theta_E,
            'w': w,
            'Z_E': Z_E,
            'bias_ms': bias_ms,
            'abl_delay_coeff_ms_per_abl': abl_delay_coeff_ms_per_abl,
            'abs_ild_delay_coeff_ms_per_unit': abs_ild_delay_coeff_ms_per_unit,
            'ild2_delay_coeff_ms_per_unit2': ild2_delay_coeff_ms_per_unit2,
            'del_go': del_go,
            'rate_norm_l': rate_norm_l,
            'alpha': alpha
        }),
        param_labels={
            'rate_lambda': r'$\lambda$',
            'T_0': r'$T_0$ (ms)',
            'theta_E': r'$\theta_E$',
            'w': r'$w$',
            'Z_E': r'$Z_E$',
            'bias_ms': 'bias (ms)',
            'abl_delay_coeff_ms_per_abl': 'ABL coeff (ms/ABL)',
            'abs_ild_delay_coeff_ms_per_unit': '|ILD| coeff (ms/unit)',
            'ild2_delay_coeff_ms_per_unit2': 'ILD^2 coeff (ms/unit^2)',
            'del_go': r'$\Delta_{go}$',
            'rate_norm_l': r'rate_norm',
            'alpha': r'$\alpha$'
        },
        vbmc_results={
            'message': results['message'],
            'elbo': results['elbo'],
            'elbo_sd': results['elbo_sd'],
            'loglike': norm_alpha_tied_loglike,
            'convergence_status': results.get('convergence_status'),
            'r_index': results.get('r_index'),
            'success_flag': results.get('success_flag')
        },
        extra_text=f"T_trunc = {T_trunc:.3f}"
    )

    pdf.savefig(norm_tied_corner_fig, bbox_inches='tight')
    plt.close(norm_tied_corner_fig)

    posterior_mean_delay_ms_by_ABL_absILD = {
        int(abl): {
            int(abs_ild): float(
                1e3 * get_t_E_aff_from_abl_ild(
                    abl,
                    abs_ild,
                    bias_ms,
                    abl_delay_coeff_ms_per_abl,
                    abs_ild_delay_coeff_ms_per_unit,
                    ild2_delay_coeff_ms_per_unit2,
                )
            )
            for abs_ild in np.sort(df_valid_animal_less_than_1['abs_ILD'].dropna().unique())
        }
        for abl in np.sort(df_valid_animal_less_than_1['ABL'].dropna().unique())
    }

    vbmc_norm_alpha_ild2_delay_tied_results = {
        'rate_lambda_samples': vp_samples[:, 0],
        'T_0_samples': vp_samples[:, 1],
        'theta_E_samples': vp_samples[:, 2],
        'w_samples': vp_samples[:, 3],
        'bias_ms_samples': vp_samples[:, 4],
        'abl_delay_coeff_ms_per_abl_samples': vp_samples[:, 5],
        'abs_ild_delay_coeff_ms_per_unit_samples': vp_samples[:, 6],
        'ild2_delay_coeff_ms_per_unit2_samples': vp_samples[:, 7],
        'del_go_samples': vp_samples[:, 8],
        'rate_norm_l_samples': vp_samples[:, 9],
        'alpha_samples': vp_samples[:, 10],
        'message': results['message'],
        'elbo': results['elbo'],
        'elbo_sd': results['elbo_sd'],
        'loglike': norm_alpha_tied_loglike
    }

    # %%
    #######################################
    ##### alpha-normalized tied diagnostics
    #######################################
    print(f'Rate norm is {rate_norm_l}')
    print(f'Alpha is {alpha}')

    t_stim_samples = df_valid_animal_less_than_1['intended_fix'].sample(N_sim, replace=True).values
    ABL_samples = df_valid_animal_less_than_1['ABL'].sample(N_sim, replace=True).values
    ILD_samples = df_valid_animal_less_than_1['ILD'].sample(N_sim, replace=True).values

    t_pts = np.arange(-1, 2, 0.001)
    P_A_mean, C_A_mean, t_stim_samples_for_diag = calculate_theoretical_curves(
        df_valid_and_aborts, N_theory, t_pts, t_A_aff, V_A, theta_A, rho_A_t_fn
    )

    sim_results = Parallel(n_jobs=30)(
        delayed(psiam_tied_data_gen_wrapper_rate_norm_alpha_fn)(
            V_A, theta_A, ABL_samples[iter_num], ILD_samples[iter_num],
            rate_lambda, T_0, theta_E, Z_E, t_A_aff,
            get_t_E_aff_from_abl_ild(
                ABL_samples[iter_num],
                ILD_samples[iter_num],
                bias_ms,
                abl_delay_coeff_ms_per_abl,
                abs_ild_delay_coeff_ms_per_unit,
                ild2_delay_coeff_ms_per_unit2,
            ),
            del_go,
            t_stim_samples[iter_num], rate_norm_l, alpha, iter_num, N_print, dt
        )
        for iter_num in tqdm(range(N_sim))
    )
    sim_results_df = pd.DataFrame(sim_results)
    sim_results_df_valid = sim_results_df[
        (sim_results_df['rt'] > sim_results_df['t_stim']) &
        (sim_results_df['rt'] - sim_results_df['t_stim'] < 1)
    ].copy()

    sim_df_1, data_df_1 = prepare_simulation_data(sim_results_df_valid, df_valid_animal_less_than_1)

    def cum_pro_and_reactive_time_vary_alpha_adapter(
            t, c_A_trunc_time,
            V_A_arg, theta_A_arg, t_A_aff_arg,
            t_stim_arg, ABL_arg, ILD_arg, rate_lambda_arg, T_0_arg, theta_E_arg, Z_E_arg, t_E_aff_arg,
            phi_params_arg, rate_norm_l_arg,
            is_norm_arg, is_time_vary_arg, K_max_arg):
        condition_t_E_aff = get_t_E_aff_from_abl_ild(
            ABL_arg,
            ILD_arg,
            bias_ms,
            abl_delay_coeff_ms_per_abl,
            abs_ild_delay_coeff_ms_per_unit,
            ild2_delay_coeff_ms_per_unit2,
        )
        return cum_pro_and_reactive_time_vary_alpha_fn(
            t, c_A_trunc_time,
            V_A_arg, theta_A_arg, t_A_aff_arg,
            t_stim_arg, ABL_arg, ILD_arg, rate_lambda_arg, T_0_arg, theta_E_arg, Z_E_arg, condition_t_E_aff,
            phi_params_arg, rate_norm_l_arg, alpha,
            is_norm_arg, is_time_vary_arg, K_max_arg
        )

    def up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_alpha_adapter(
            t, bound,
            P_A, C_A,
            ABL_arg, ILD_arg, rate_lambda_arg, T_0_arg, theta_E_arg, Z_E_arg, t_E_aff_arg, del_go_arg,
            phi_params_arg, rate_norm_l_arg,
            is_norm_arg, is_time_vary_arg, K_max_arg):
        condition_t_E_aff = get_t_E_aff_from_abl_ild(
            ABL_arg,
            ILD_arg,
            bias_ms,
            abl_delay_coeff_ms_per_abl,
            abs_ild_delay_coeff_ms_per_unit,
            ild2_delay_coeff_ms_per_unit2,
        )
        return up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_alpha_fn(
            t, bound,
            P_A, C_A,
            ABL_arg, ILD_arg, rate_lambda_arg, T_0_arg, theta_E_arg, Z_E_arg, condition_t_E_aff, del_go_arg,
            phi_params_arg, rate_norm_l_arg, alpha,
            is_norm_arg, is_time_vary_arg, K_max_arg
        )

    theory_results_up_and_down, theory_time_axis, bins, bin_centers = plot_rt_distributions(
        sim_df_1, data_df_1, ILD_arr, ABL_arr, t_pts, P_A_mean, C_A_mean,
        t_stim_samples_for_diag, V_A, theta_A, t_A_aff,
        rate_lambda, T_0, theta_E, Z_E, np.nan, del_go,
        phi_params_obj, rate_norm_l, True, False, K_max, T_trunc,
        cum_pro_and_reactive_time_vary_alpha_adapter, up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_alpha_adapter,
        animal, pdf, model_name="Alpha-Normalized Tied ILD2 Delay"
    )

    plot_tachometric_curves(
        sim_df_1, data_df_1, ILD_arr, ABL_arr, theory_results_up_and_down,
        theory_time_axis, bins, animal, pdf, model_name="Alpha-Normalized Tied ILD2 Delay"
    )

    plot_grand_summary(
        sim_df_1, data_df_1, ILD_arr, ABL_arr, bins, bin_centers,
        animal, pdf, model_name="Alpha-Normalized Tied ILD2 Delay"
    )

    # %%
    print("\nGenerating model comparison tables...")
    save_dict = {
        'vbmc_aborts_results': vbmc_aborts_results,
        'vbmc_norm_alpha_ild2_delay_tied_results': vbmc_norm_alpha_ild2_delay_tied_results,
        'fit_config': {
            'batch_name': batch_name,
            'animal': animal,
            'likelihood_mode': 'choice_aware_alpha_normalized_ild2_delay',
            'delay_rule_ms': 'bias_ms + abl_delay_coeff_ms_per_abl * ABL + abs_ild_delay_coeff_ms_per_unit * |ILD| + ild2_delay_coeff_ms_per_unit2 * ILD^2',
            'negative_delay_rule': 'if delay_ms < 0 for any observed fit condition, return log(1e-50)',
            'delay_return_units': 'seconds',
            'parameter_order': [
                'rate_lambda',
                'T_0',
                'theta_E',
                'w',
                'bias_ms',
                'abl_delay_coeff_ms_per_abl',
                'abs_ild_delay_coeff_ms_per_unit',
                'ild2_delay_coeff_ms_per_unit2',
                'del_go',
                'rate_norm_l',
                'alpha',
            ],
            'delay_coefficient_bounds': {
                'bias_ms': {
                    'hard': norm_tied_bias_ms_bounds,
                    'plausible': norm_tied_bias_ms_plausible_bounds,
                },
                'abl_delay_coeff_ms_per_abl': {
                    'hard': norm_tied_abl_delay_coeff_ms_per_abl_bounds,
                    'plausible': norm_tied_abl_delay_coeff_ms_per_abl_plausible_bounds,
                },
                'abs_ild_delay_coeff_ms_per_unit': {
                    'hard': norm_tied_abs_ild_delay_coeff_ms_per_unit_bounds,
                    'plausible': norm_tied_abs_ild_delay_coeff_ms_per_unit_plausible_bounds,
                },
                'ild2_delay_coeff_ms_per_unit2': {
                    'hard': norm_tied_ild2_delay_coeff_ms_per_unit2_bounds,
                    'plausible': norm_tied_ild2_delay_coeff_ms_per_unit2_plausible_bounds,
                },
            },
            'observed_delay_condition_pairs_used_for_fit': observed_delay_condition_pairs.tolist(),
            'T_trunc': T_trunc,
        },
        'posterior_mean_delay_ms_by_ABL_absILD': posterior_mean_delay_ms_by_ABL_absILD,
    }

    abort_df = create_abort_table(save_dict['vbmc_aborts_results'])
    if abort_df is not None:
        render_df_to_pdf(abort_df, f"Abort Model Results - Animal {animal}", pdf)
        print(f"Added abort model results table to PDF for animal {animal}")

    tied_df = create_tied_table(save_dict)
    if tied_df is not None:
        render_df_to_pdf(tied_df, f"Tied Models Comparison - Animal {animal}", pdf)
        print(f"Added alpha-normalized TIED ILD2-delay results table to PDF for animal {animal}")

    with open(pkl_filename, 'wb') as f:
        pickle.dump(save_dict, f)

    print(f"Saved results to {pkl_filename}")

    pdf.close()
    print(f"Saved PDF report to {pdf_filename}")
