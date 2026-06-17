# %%
"""
Animal-wise alpha-normalized TIED fit with condition t_E_aff fixed.

This standalone fit adds alpha to the normalized TIED rate model while loading
V_A, theta_A, and t_A_aff from an existing animal-wise result pickle. The
evidence-afferent delay is not fitted here; it is fixed to the exact
condition-by-condition t_E_aff posterior mean for each animal/ABL/signed-ILD.
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
import pyvbmc.vbmc.active_sample as pyvbmc_active_sample
import pyvbmc.vbmc.variational_optimization as pyvbmc_variational_optimization
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
########### PyVBMC compatibility patch ##############
# PyVBMC can return one-element arrays for full-ELBO scalar variance terms
# with some GP states. Newer NumPy refuses assigning those arrays into scalar
# slots, so coerce only the scalar diagnostics while leaving gradients and
# per-component arrays unchanged.
if not getattr(pyvbmc_variational_optimization, "_ddm_scalar_elcbo_patch", False):
    _original_neg_elcbo = pyvbmc_variational_optimization._neg_elcbo

    def _scalar_like(value):
        arr = np.asarray(value)
        if arr.ndim == 0:
            return float(arr)
        if arr.size == 1:
            return float(arr.reshape(-1)[0])
        return float(np.mean(arr))

    def _neg_elcbo_with_scalar_diagnostics(*args, **kwargs):
        result = _original_neg_elcbo(*args, **kwargs)
        if not isinstance(result, tuple):
            return result

        result = list(result)
        if len(result) == 11:
            for idx in [0, 2, 3, 4, 6, 7, 8]:
                result[idx] = _scalar_like(result[idx])
        elif len(result) == 5:
            for idx in [0, 2, 3, 4]:
                result[idx] = _scalar_like(result[idx])
        return tuple(result)

    pyvbmc_variational_optimization._neg_elcbo = _neg_elcbo_with_scalar_diagnostics
    pyvbmc_active_sample._neg_elcbo = _neg_elcbo_with_scalar_diagnostics
    pyvbmc_variational_optimization._ddm_scalar_elcbo_patch = True


# %%
############3 Params #############
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
REPO_DIR = os.path.dirname(SCRIPT_DIR)

DESIRED_BATCHES = ['SD', 'LED34', 'LED6', 'LED8', 'LED7', 'LED34_even']
desired_batches_override = os.environ.get('DESIRED_BATCHES_OVERRIDE')
if desired_batches_override:
    DESIRED_BATCHES = [batch.strip() for batch in desired_batches_override.split(',') if batch.strip()]

batch_dir = os.environ.get(
    'BATCH_CSV_DIR_OVERRIDE',
    os.path.join(REPO_DIR, 'raw_data', 'batch_csvs')
)
abort_params_dir = os.environ.get(
    'ABORT_PARAMS_DIR_OVERRIDE',
    os.path.join(REPO_DIR, 'aborts_ipl_npl_time_fit_results')
)
output_dir = os.path.join(SCRIPT_DIR, 'NPL_alpha_condition_t_E_aff_fixed_delay_fit_results')
os.makedirs(output_dir, exist_ok=True)
CONDITION_T_E_AFF_CACHE = os.path.join(
    REPO_DIR,
    'fit_each_condn',
    'abl_specific_ild2_delay_agreement',
    'condition_t_E_aff_extraction_cache.csv',
)
EXPECTED_CONDITION_CACHE_ROWS = 864
DRY_RUN_ONLY = os.environ.get('DRY_RUN_ONLY', '0').strip().lower() in {'1', 'true', 'yes'}
test_pairs_override = os.environ.get('TEST_BATCH_ANIMAL_PAIRS')
if test_pairs_override:
    TEST_BATCH_ANIMAL_PAIRS = [
        (batch.strip(), int(animal.strip()))
        for batch, animal in (item.split(':') for item in test_pairs_override.split(',') if item.strip())
    ]
else:
    TEST_BATCH_ANIMAL_PAIRS = None  # Set env like TEST_BATCH_ANIMAL_PAIRS=LED7:92 for smoke tests.
USE_VECTORIZED_LIKELIHOOD = True
SKIP_FINISHED_FITS = os.environ.get('SKIP_FINISHED_FITS', '1').strip().lower() not in {'0', 'false', 'no'}
RUN_DIAGNOSTICS_AFTER_FIT = os.environ.get('RUN_DIAGNOSTICS_AFTER_FIT', '0').strip().lower() in {'1', 'true', 'yes'}
FINISHED_ALPHA_RESULT_KEY = 'vbmc_norm_alpha_condition_t_E_aff_fixed_delay_tied_results'
HARD_CODED_DELAY_ABL_LEVELS = np.array([20.0, 40.0, 60.0], dtype=float)
K_max = 10

BATCH_T_TRUNC = {
    'LED34_even': 0.15,
}
DEFAULT_T_TRUNC = 0.3
T_trunc = DEFAULT_T_TRUNC

N_theory = int(1e3)
N_sim = int(float(os.environ.get('N_SIM_OVERRIDE', 1e6)))
dt = 1e-3
N_print = max(1, int(N_sim / 5))
DIAGNOSTIC_N_JOBS = int(os.environ.get('DIAGNOSTIC_N_JOBS', 30))
VBMC_FUN_EVAL_SCALE = int(os.environ.get('VBMC_FUN_EVAL_SCALE', 200))
VBMC_MAX_FUN_EVALS_OVERRIDE = os.environ.get('VBMC_MAX_FUN_EVALS')
FIT_RANDOM_SEED = os.environ.get('FIT_RANDOM_SEED')
if FIT_RANDOM_SEED is not None:
    FIT_RANDOM_SEED = int(FIT_RANDOM_SEED)
    random.seed(FIT_RANDOM_SEED)
    np.random.seed(FIT_RANDOM_SEED)


# %%
########### Resume helpers ##############
def alpha_result_pkl_path(batch_name, animal):
    return os.path.join(output_dir, f'results_{batch_name}_animal_{animal}_NORM_ALPHA_CONDITION_T_E_AFF_FIXED_DELAY_FROM_ABORTS.pkl')


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

    vbmc_message = str(alpha_results.get('message', ''))
    if 'stable' not in vbmc_message.lower():
        short_message = vbmc_message.replace('\n', ' ')[:120] or 'missing VBMC message'
        print(
            f"WARNING: Existing result pkl is not a stable VBMC fit ({short_message}); "
            f"will rerun: {os.path.relpath(pkl_path, SCRIPT_DIR)}"
        )
        return False

    return True


# %%
###########  Alpha-normalized TIED ##############
condition_t_E_aff_lookup = {}


def format_abl_suffix(abl):
    abl = float(abl)
    if abl.is_integer():
        return str(int(abl))
    return f"{abl:g}"


def unpack_norm_alpha_condition_fixed_delay_params(params):
    params = np.asarray(params, dtype=float)
    (
        rate_lambda,
        T_0,
        theta_E,
        w,
        del_go,
        rate_norm_l,
        alpha,
    ) = params
    return (
        rate_lambda,
        T_0,
        theta_E,
        w,
        del_go,
        rate_norm_l,
        alpha,
    )


def get_norm_alpha_condition_fixed_delay_param_names():
    return [
        'rate_lambda',
        'T_0',
        'theta_E',
        'w',
        'del_go',
        'rate_norm_l',
        'alpha',
    ]


def get_condition_t_E_aff(batch_name, animal, abl, ild):
    abl_arr, ild_arr = np.broadcast_arrays(np.asarray(abl), np.asarray(ild))
    batch_arr = np.broadcast_to(np.asarray(batch_name, dtype=object), abl_arr.shape)
    animal_arr = np.broadcast_to(np.asarray(animal), abl_arr.shape)
    values = []
    for batch_value, animal_value, abl_value, ild_value in zip(
            batch_arr.ravel(),
            animal_arr.ravel(),
            abl_arr.ravel(),
            ild_arr.ravel()):
        key = (str(batch_value), int(animal_value), int(abl_value), float(ild_value))
        if key not in condition_t_E_aff_lookup:
            raise KeyError(f"Missing fixed condition t_E_aff for {key}")
        values.append(condition_t_E_aff_lookup[key])
    out = np.asarray(values, dtype=float).reshape(abl_arr.shape)
    if np.ndim(abl) == 0 and np.ndim(ild) == 0:
        return float(out.reshape(-1)[0])
    return out


def compute_loglike_norm_alpha_fn(
        row, rate_lambda, T_0, theta_E, Z_E,
        del_go, rate_norm_l, alpha, t_trunc):
    rt = row['TotalFixTime']
    t_stim = row['intended_fix']

    ILD = row['ILD']
    ABL = row['ABL']
    choice = row['choice']
    t_E_aff = get_condition_t_E_aff(row['batch_name'], row['animal'], ABL, ILD)

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
        del_go,
        rate_norm_l,
        alpha,
    ) = unpack_norm_alpha_condition_fixed_delay_params(params)

    Z_E = (w - 0.5) * 2 * theta_E
    all_loglike = Parallel(n_jobs=30)(
        delayed(compute_loglike_norm_alpha_fn)(
            row, rate_lambda, T_0, theta_E, Z_E,
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
        del_go,
        rate_norm_l,
        alpha,
    ) = unpack_norm_alpha_condition_fixed_delay_params(params)

    Z_E = (w - 0.5) * 2 * theta_E

    total_fix = df_valid_animal_less_than_1['TotalFixTime'].to_numpy(dtype=float)
    t_stim = df_valid_animal_less_than_1['intended_fix'].to_numpy(dtype=float)
    ABL = df_valid_animal_less_than_1['ABL'].to_numpy(dtype=float)
    ILD = df_valid_animal_less_than_1['ILD'].to_numpy(dtype=float)
    choice = df_valid_animal_less_than_1['choice'].to_numpy(dtype=float)
    batch_names = df_valid_animal_less_than_1['batch_name'].to_numpy()
    animals = df_valid_animal_less_than_1['animal'].to_numpy(dtype=int)
    t_E_aff = np.array(
        [
            get_condition_t_E_aff(batch_value, animal_value, abl_value, ild_value)
            for batch_value, animal_value, abl_value, ild_value in zip(batch_names, animals, ABL, ILD)
        ],
        dtype=float
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
        del_go,
        rate_norm_l,
        alpha,
    ) = unpack_norm_alpha_condition_fixed_delay_params(params)

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
        del_go_logpdf +
        rate_norm_l_logpdf +
        alpha_logpdf
    )


def vbmc_norm_alpha_tied_joint_fn(params):
    priors = vbmc_prior_norm_alpha_tied_fn(params)
    loglike = vbmc_norm_alpha_tied_active_loglike_fn(params)

    return float(priors + loglike)


norm_tied_rate_lambda_bounds = [0.5, 5]
norm_tied_T_0_bounds = [20e-3, 800e-3]
norm_tied_theta_E_bounds = [1, 15]
norm_tied_w_bounds = [0.3, 0.7]
norm_tied_del_go_bounds = [0, 0.2]
norm_tied_rate_norm_bounds = [0, 2]
norm_tied_alpha_bounds = [0, 2]

norm_tied_rate_lambda_plausible_bounds = [1, 3]
norm_tied_T_0_plausible_bounds = [40e-3, 400e-3]
norm_tied_theta_E_plausible_bounds = [1.5, 10]
norm_tied_w_plausible_bounds = [0.4, 0.6]
norm_tied_del_go_plausible_bounds = [0.02, 0.199]
norm_tied_rate_norm_plausible_bounds = [0.75, 1.05]
norm_tied_alpha_plausible_bounds = [0.1, 1.8]

norm_tied_shared_lb = np.array([
    norm_tied_rate_lambda_bounds[0],
    norm_tied_T_0_bounds[0],
    norm_tied_theta_E_bounds[0],
    norm_tied_w_bounds[0],
    norm_tied_del_go_bounds[0],
    norm_tied_rate_norm_bounds[0],
    norm_tied_alpha_bounds[0]
])

norm_tied_shared_ub = np.array([
    norm_tied_rate_lambda_bounds[1],
    norm_tied_T_0_bounds[1],
    norm_tied_theta_E_bounds[1],
    norm_tied_w_bounds[1],
    norm_tied_del_go_bounds[1],
    norm_tied_rate_norm_bounds[1],
    norm_tied_alpha_bounds[1]
])

norm_tied_shared_plb = np.array([
    norm_tied_rate_lambda_plausible_bounds[0],
    norm_tied_T_0_plausible_bounds[0],
    norm_tied_theta_E_plausible_bounds[0],
    norm_tied_w_plausible_bounds[0],
    norm_tied_del_go_plausible_bounds[0],
    norm_tied_rate_norm_plausible_bounds[0],
    norm_tied_alpha_plausible_bounds[0]
])

norm_tied_shared_pub = np.array([
    norm_tied_rate_lambda_plausible_bounds[1],
    norm_tied_T_0_plausible_bounds[1],
    norm_tied_theta_E_plausible_bounds[1],
    norm_tied_w_plausible_bounds[1],
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
print(f'N_sim for diagnostics: {N_sim}')
print(f'Diagnostic n_jobs: {DIAGNOSTIC_N_JOBS}')
print(f'VBMC fun-eval scale: {VBMC_FUN_EVAL_SCALE}')
print(f'VBMC max_fun_evals override: {VBMC_MAX_FUN_EVALS_OVERRIDE}')
print(f'Fit random seed: {FIT_RANDOM_SEED}')
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
    pkl_file = os.path.join(abort_params_dir, f'results_{batch_name}_animal_{animal}.pkl')
    if not os.path.exists(pkl_file):
        missing_abort_pkls.append(pkl_file)

if missing_abort_pkls:
    print("WARNING: Missing abort-source pickle files:")
    for pkl_file in missing_abort_pkls:
        print(f"  {os.path.relpath(pkl_file, REPO_DIR)}")
else:
    print(f"All discovered batch-animal pairs have matching abort-source pickle files in {abort_params_dir}.")

if SKIP_FINISHED_FITS:
    finished_pairs = [
        (batch_name, animal)
        for batch_name, animal in batch_animal_pairs
        if is_finished_alpha_result(alpha_result_pkl_path(batch_name, animal))
    ]
    print(f"Found {len(finished_pairs)} finished alpha result pickle files that will be skipped.")

print("\n### Loading fixed condition t_E_aff cache ###")
if not os.path.exists(CONDITION_T_E_AFF_CACHE):
    raise FileNotFoundError(f"Missing condition t_E_aff cache: {CONDITION_T_E_AFF_CACHE}")

condition_t_E_aff_cache_df = pd.read_csv(CONDITION_T_E_AFF_CACHE)
required_condition_cache_cols = ['batch_name', 'animal', 'ABL', 'ILD', 't_E_aff_s', 't_E_aff_ms']
missing_condition_cache_cols = [
    col for col in required_condition_cache_cols
    if col not in condition_t_E_aff_cache_df.columns
]
if missing_condition_cache_cols:
    raise KeyError(f"Condition t_E_aff cache is missing columns: {missing_condition_cache_cols}")
if len(condition_t_E_aff_cache_df) != EXPECTED_CONDITION_CACHE_ROWS:
    raise ValueError(
        f"Condition t_E_aff cache has {len(condition_t_E_aff_cache_df)} rows; "
        f"expected {EXPECTED_CONDITION_CACHE_ROWS}."
    )

condition_t_E_aff_cache_df = condition_t_E_aff_cache_df.copy()
condition_t_E_aff_cache_df['batch_name'] = condition_t_E_aff_cache_df['batch_name'].astype(str)
condition_t_E_aff_cache_df['animal'] = condition_t_E_aff_cache_df['animal'].astype(int)
condition_t_E_aff_cache_df['ABL'] = condition_t_E_aff_cache_df['ABL'].astype(int)
condition_t_E_aff_cache_df['ILD'] = condition_t_E_aff_cache_df['ILD'].astype(float)
condition_t_E_aff_cache_df['t_E_aff_s'] = condition_t_E_aff_cache_df['t_E_aff_s'].astype(float)
condition_t_E_aff_cache_df['t_E_aff_ms'] = condition_t_E_aff_cache_df['t_E_aff_ms'].astype(float)

duplicated_condition_cache_keys = condition_t_E_aff_cache_df.duplicated(['batch_name', 'animal', 'ABL', 'ILD'])
if duplicated_condition_cache_keys.any():
    duplicate_rows = condition_t_E_aff_cache_df.loc[
        duplicated_condition_cache_keys,
        ['batch_name', 'animal', 'ABL', 'ILD']
    ]
    raise ValueError(f"Condition t_E_aff cache has duplicate keys:\n{duplicate_rows.to_string(index=False)}")

condition_t_E_aff_lookup.clear()
condition_t_E_aff_lookup.update({
    (row.batch_name, int(row.animal), int(row.ABL), float(row.ILD)): float(row.t_E_aff_s)
    for row in condition_t_E_aff_cache_df.itertuples(index=False)
})

led7_92_minus_ms = 1e3 * get_condition_t_E_aff('LED7', 92, 20, -1)
led7_92_plus_ms = 1e3 * get_condition_t_E_aff('LED7', 92, 20, 1)
print(f"LED7/92 ABL=20 ILD=-1 fixed t_E_aff_ms = {led7_92_minus_ms:.3f}")
print(f"LED7/92 ABL=20 ILD=+1 fixed t_E_aff_ms = {led7_92_plus_ms:.3f}")
if not (52.5 <= led7_92_minus_ms <= 53.7):
    raise ValueError(f"LED7/92 ABL=20 ILD=-1 t_E_aff_ms looks stale/abnormal: {led7_92_minus_ms:.3f}")
if not (45.0 <= led7_92_plus_ms <= 46.3):
    raise ValueError(f"LED7/92 ABL=20 ILD=+1 t_E_aff_ms looks stale/abnormal: {led7_92_plus_ms:.3f}")

batch_animal_pair_df = pd.DataFrame(batch_animal_pairs, columns=['batch_name', 'animal'])
if len(batch_animal_pair_df) > 0:
    batch_animal_pair_df['batch_name'] = batch_animal_pair_df['batch_name'].astype(str)
    batch_animal_pair_df['animal'] = batch_animal_pair_df['animal'].astype(int)
    valid_fit_conditions = merged_valid[
        (merged_valid['RTwrtStim'] < 1) &
        (merged_valid['ABL'].isin(HARD_CODED_DELAY_ABL_LEVELS))
    ][['batch_name', 'animal', 'ABL', 'ILD']].copy()
    valid_fit_conditions['batch_name'] = valid_fit_conditions['batch_name'].astype(str)
    valid_fit_conditions['animal'] = valid_fit_conditions['animal'].astype(int)
    valid_fit_conditions['ABL'] = valid_fit_conditions['ABL'].astype(int)
    valid_fit_conditions['ILD'] = valid_fit_conditions['ILD'].astype(float)
    valid_fit_conditions = valid_fit_conditions.merge(
        batch_animal_pair_df,
        on=['batch_name', 'animal'],
        how='inner'
    ).drop_duplicates()

    cache_key_df = condition_t_E_aff_cache_df[['batch_name', 'animal', 'ABL', 'ILD']].drop_duplicates()
    missing_fit_delay_conditions = valid_fit_conditions.merge(
        cache_key_df,
        on=['batch_name', 'animal', 'ABL', 'ILD'],
        how='left',
        indicator=True
    )
    missing_fit_delay_conditions = missing_fit_delay_conditions[
        missing_fit_delay_conditions['_merge'] == 'left_only'
    ][['batch_name', 'animal', 'ABL', 'ILD']]
    if len(missing_fit_delay_conditions) > 0:
        raise ValueError(
            "Condition t_E_aff cache is missing observed fit conditions:\n"
            f"{missing_fit_delay_conditions.to_string(index=False)}"
        )
    print(
        f"Condition t_E_aff cache loaded with {len(condition_t_E_aff_cache_df)} rows and covers "
        f"{len(valid_fit_conditions)} observed fit conditions for the selected animals."
    )

if DRY_RUN_ONLY:
    print("DRY_RUN_ONLY=True: batch discovery completed; skipping VBMC fits.")
    raise SystemExit(0)


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
    pdf_filename = os.path.join(output_dir, f'results_{batch_name}_animal_{animal}_NORM_ALPHA_CONDITION_T_E_AFF_FIXED_DELAY_FROM_ABORTS.pdf')
    pdf = PdfPages(pdf_filename)

    fig_text = plt.figure(figsize=(8.5, 11))
    fig_text.clf()
    fig_text.text(0.1, 0.9, "Alpha-Normalized TIED With Fixed Condition t_E_aff From Loaded Abort Parameters", fontsize=18, weight='bold')
    fig_text.text(0.1, 0.8, f"Batch Name: {batch_name}", fontsize=14)
    fig_text.text(0.1, 0.75, f"Animal ID: {animal}", fontsize=14)
    fig_text.text(0.1, 0.68, "Abort params loaded from existing result pickle", fontsize=12)
    fig_text.text(0.1, 0.63, f"T_trunc = {T_trunc:.3f}", fontsize=12)
    fig_text.text(0.1, 0.58, "t_E_aff fixed by exact condition cache keyed by batch/animal/ABL/signed ILD", fontsize=12)
    fig_text.text(0.1, 0.53, f"Fit ABLs restricted to {HARD_CODED_DELAY_ABL_LEVELS.astype(int).tolist()}", fontsize=12)
    fig_text.gca().axis("off")
    pdf.savefig(fig_text, bbox_inches='tight')
    plt.close(fig_text)

    # %%
    ####################################################
    ########### Load Abort Parameters ##################
    ####################################################
    print("\n### Loading Abort Parameters from Pickle ###")

    pkl_file = os.path.join(abort_params_dir, f'results_{batch_name}_animal_{animal}.pkl')
    pkl_file_rel = os.path.relpath(pkl_file, REPO_DIR)
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
        fig_text.text(0.1, 0.9, f"Abort Parameters (Loaded from {pkl_file_rel})", fontsize=16, weight='bold')
        fig_text.text(0.1, 0.8, f"V_A: {V_A:.4f}", fontsize=12)
        fig_text.text(0.1, 0.75, f"theta_A: {theta_A:.4f}", fontsize=12)
        fig_text.text(0.1, 0.7, f"t_A_aff: {t_A_aff:.4f}", fontsize=12)
        fig_text.gca().axis("off")
        pdf.savefig(fig_text, bbox_inches='tight')
        plt.close(fig_text)

    except FileNotFoundError:
        print(f"ERROR: Pickle file {pkl_file_rel} not found. Please run abort fitting first.")
        pdf.close()
        continue
    except KeyError as e:
        print(f"ERROR: Missing key {e} in pickle file {pkl_file_rel}")
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
    df_valid_animal_less_than_1_all_abls = df_valid_animal[df_valid_animal['RTwrtStim'] < 1].copy()
    df_valid_animal_less_than_1 = df_valid_animal_less_than_1_all_abls[
        df_valid_animal_less_than_1_all_abls['ABL'].isin(HARD_CODED_DELAY_ABL_LEVELS)
    ].copy()
    n_excluded_non_fit_abl = len(df_valid_animal_less_than_1_all_abls) - len(df_valid_animal_less_than_1)
    if n_excluded_non_fit_abl > 0:
        excluded_abl_levels = np.sort(
            df_valid_animal_less_than_1_all_abls.loc[
                ~df_valid_animal_less_than_1_all_abls['ABL'].isin(HARD_CODED_DELAY_ABL_LEVELS),
                'ABL'
            ].dropna().unique()
        )
        print(
            f"Excluding {n_excluded_non_fit_abl} valid RT<1 trials with non-fit ABLs "
            f"{excluded_abl_levels.tolist()} for this ABL 20/40/60-only fit."
        )
    if len(df_valid_animal_less_than_1) == 0:
        print(f"WARNING: No valid RT<1 trials at ABL 20/40/60 for {batch_name}, animal {animal}; skipping.")
        pdf.close()
        continue

    ABL_arr = np.array([
        abl for abl in HARD_CODED_DELAY_ABL_LEVELS
        if np.any(df_valid_animal_less_than_1['ABL'].to_numpy(dtype=float) == abl)
    ], dtype=float)
    ILD_arr = np.sort(df_valid_animal_less_than_1['ILD'].unique())
    norm_tied_lb = norm_tied_shared_lb.copy()
    norm_tied_ub = norm_tied_shared_ub.copy()
    norm_tied_plb = norm_tied_shared_plb.copy()
    norm_tied_pub = norm_tied_shared_pub.copy()

    fixed_condition_delay_rows = []
    observed_animal_conditions = (
        df_valid_animal_less_than_1[['ABL', 'ILD']]
        .drop_duplicates()
        .sort_values(['ABL', 'ILD'])
    )
    for _, condition_row in observed_animal_conditions.iterrows():
        abl = float(condition_row['ABL'])
        ild = float(condition_row['ILD'])
        fixed_condition_delay_rows.append({
            'ABL': format_abl_suffix(abl),
            'ILD': f"{ild:g}",
            't_E_aff_ms': 1e3 * get_condition_t_E_aff(batch_name, animal, abl, ild),
        })
    fixed_condition_delay_df = pd.DataFrame(fixed_condition_delay_rows)
    print(
        f"Loaded {len(fixed_condition_delay_df)} fixed condition t_E_aff values "
        f"for {batch_name}, animal {animal}."
    )
    if len(fixed_condition_delay_df) > 0:
        print(fixed_condition_delay_df.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
        render_df_to_pdf(fixed_condition_delay_df, f"Fixed Condition t_E_aff Delays - Animal {animal}", pdf)

    norm_tied_keyname = "vbmc_norm_tied_results"
    norm_tied_start_keys = [
        'rate_lambda_samples',
        'T_0_samples',
        'theta_E_samples',
        'w_samples',
        'del_go_samples',
        'rate_norm_l_samples'
    ]

    initialization_source = 'generic plausible defaults'
    if (
            norm_tied_keyname in fit_results_data and
            all(key in fit_results_data[norm_tied_keyname] for key in norm_tied_start_keys)):
        norm_tied_samples = fit_results_data[norm_tied_keyname]
        shared_x_0 = np.array([
            np.mean(norm_tied_samples['rate_lambda_samples']),
            np.mean(norm_tied_samples['T_0_samples']),
            np.mean(norm_tied_samples['theta_E_samples']),
            np.mean(norm_tied_samples['w_samples']),
            np.mean(norm_tied_samples['del_go_samples']),
            np.mean(norm_tied_samples['rate_norm_l_samples']),
            1.0
        ])
        initialization_source = 'existing normalized TIED posterior mean'
        print("Initializing shared parameters from existing normalized TIED posterior mean.")
    else:
        shared_x_0 = np.array([
            2.3,
            100 * 1e-3,
            3,
            0.51,
            0.13,
            0.95,
            1.0
        ])
        print("No normalized TIED posterior found in pickle; using generic plausible initialization.")

    raw_x_0 = shared_x_0.copy()
    x_0 = raw_x_0.copy()

    plausible_eps = 1e-6 * (norm_tied_pub - norm_tied_plb)
    x_0 = np.clip(x_0, norm_tied_plb + plausible_eps, norm_tied_pub - plausible_eps)
    clipped_init = ~np.isclose(raw_x_0, x_0, rtol=0, atol=1e-10)
    if np.any(clipped_init):
        print("Initial parameters clipped to plausible bounds:")
        for param_name, raw_value, clipped_value in zip(get_norm_alpha_condition_fixed_delay_param_names(), raw_x_0, x_0):
            if not np.isclose(raw_value, clipped_value, rtol=0, atol=1e-10):
                print(f"  {param_name}: {raw_value:.6g} -> {clipped_value:.6g}")
    else:
        print("Initial parameters were inside plausible bounds; no clipping applied.")

    print("Initial alpha-normalized parameters:")
    print(f"  rate_lambda = {x_0[0]:.5f}")
    print(f"  T_0         = {1e3*x_0[1]:.5f} ms")
    print(f"  theta_E     = {x_0[2]:.5f}")
    print(f"  w           = {x_0[3]:.5f}")
    print(f"  del_go      = {x_0[4]:.5f}")
    print(f"  rate_norm_l = {x_0[5]:.5f}")
    print(f"  alpha       = {x_0[6]:.5f}")

    max_fun_evals = (
        int(VBMC_MAX_FUN_EVALS_OVERRIDE)
        if VBMC_MAX_FUN_EVALS_OVERRIDE is not None
        else VBMC_FUN_EVAL_SCALE * (2 + len(x_0))
    )
    print(f"VBMC max_fun_evals for this animal: {max_fun_evals}")

    vbmc = VBMC(
        vbmc_norm_alpha_tied_joint_fn,
        x_0,
        norm_tied_lb,
        norm_tied_ub,
        norm_tied_plb,
        norm_tied_pub,
        options={'display': 'on', 'max_fun_evals': max_fun_evals}
    )
    vp, results = vbmc.optimize()
    vbmc.save(
        os.path.join(output_dir, f'vbmc_PKL_file_norm_alpha_condition_t_E_aff_fixed_delay_tied_results_batch_{batch_name}_animal_{animal}_FROM_ABORTS.pkl'),
        overwrite=True
    )

    vp_samples = vp.sample(int(1e5))[0]
    vp_samples[:, 1] *= 1e3
    param_labels = [
        r'$\lambda$',
        r'$T_0$ (ms)',
        r'$\theta_E$',
        r'$w$',
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
    norm_tied_corner_fig.suptitle(f'Alpha-Normalized Tied Fixed Condition t_E_aff Posterior (Animal: {animal})', y=1.02)
    vp_samples[:, 1] /= 1e3

    rate_lambda = vp_samples[:, 0].mean()
    T_0 = vp_samples[:, 1].mean()
    theta_E = vp_samples[:, 2].mean()
    w = vp_samples[:, 3].mean()
    Z_E = (w - 0.5) * 2 * theta_E
    del_go = vp_samples[:, 4].mean()
    rate_norm_l = vp_samples[:, 5].mean()
    alpha = vp_samples[:, 6].mean()

    print("Posterior Means:")
    print(f"rate_lambda  = {rate_lambda:.5f}")
    print(f"T_0 (ms)      = {1e3*T_0:.5f}")
    print(f"theta_E       = {theta_E:.5f}")
    print(f"Z_E           = {Z_E:.5f}")
    print(f"del_go        = {del_go:.5f}")
    print(f"rate_norm_l   = {rate_norm_l:.5f}")
    print(f"alpha         = {alpha:.5f}")

    posterior_mean_params = np.array([rate_lambda, T_0, theta_E, w, del_go, rate_norm_l, alpha], dtype=float)
    norm_alpha_tied_loglike = vbmc_norm_alpha_tied_active_loglike_fn(posterior_mean_params)

    posterior_means = {
        'rate_lambda': rate_lambda,
        'T_0': 1e3*T_0,
        'theta_E': theta_E,
        'w': w,
        'Z_E': Z_E,
        'del_go': del_go,
        'rate_norm_l': rate_norm_l,
        'alpha': alpha,
    }
    posterior_param_labels = {
        'rate_lambda': r'$\lambda$',
        'T_0': r'$T_0$ (ms)',
        'theta_E': r'$\theta_E$',
        'w': r'$w$',
        'Z_E': r'$Z_E$',
        'del_go': r'$\Delta_{go}$',
        'rate_norm_l': r'rate_norm',
        'alpha': r'$\alpha$',
    }

    save_posterior_summary_page(
        pdf_pages=pdf,
        title=f'Alpha-Normalized Tied Fixed Condition t_E_aff Model - Posterior Means ({animal})',
        posterior_means=pd.Series(posterior_means),
        param_labels=posterior_param_labels,
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

    fixed_condition_t_E_aff_ms = [
        {
            'ABL': row['ABL'],
            'ILD': row['ILD'],
            't_E_aff_ms': float(row['t_E_aff_ms']),
        }
        for _, row in fixed_condition_delay_df.iterrows()
    ]

    vbmc_norm_alpha_condition_t_E_aff_fixed_delay_tied_results = {
        'rate_lambda_samples': vp_samples[:, 0],
        'T_0_samples': vp_samples[:, 1],
        'theta_E_samples': vp_samples[:, 2],
        'w_samples': vp_samples[:, 3],
        'del_go_samples': vp_samples[:, 4],
        'rate_norm_l_samples': vp_samples[:, 5],
        'alpha_samples': vp_samples[:, 6],
        'message': results['message'],
        'elbo': results['elbo'],
        'elbo_sd': results['elbo_sd'],
        'loglike': norm_alpha_tied_loglike
    }

    save_dict = {
        'vbmc_aborts_results': vbmc_aborts_results,
        FINISHED_ALPHA_RESULT_KEY: vbmc_norm_alpha_condition_t_E_aff_fixed_delay_tied_results,
        'fit_config': {
            'batch_name': batch_name,
            'animal': animal,
            'likelihood_mode': 'choice_aware_alpha_normalized_condition_t_E_aff_fixed_delay',
            'delay_rule_ms': 'fixed exact condition-cache t_E_aff_s by batch_name/animal/ABL/signed ILD',
            'delay_return_units': 'seconds',
            'condition_t_E_aff_cache': os.path.relpath(CONDITION_T_E_AFF_CACHE, REPO_DIR),
            'condition_t_E_aff_cache_rows': int(len(condition_t_E_aff_cache_df)),
            'fixed_delay_lookup_key': 'batch_name, animal, ABL, signed ILD',
            'fixed_condition_t_E_aff_ms': fixed_condition_t_E_aff_ms,
            'parameter_order': get_norm_alpha_condition_fixed_delay_param_names(),
            'shared_parameter_bounds': {
                'rate_lambda': {
                    'hard': norm_tied_rate_lambda_bounds,
                    'plausible': norm_tied_rate_lambda_plausible_bounds,
                },
                'T_0': {
                    'hard': norm_tied_T_0_bounds,
                    'plausible': norm_tied_T_0_plausible_bounds,
                },
                'theta_E': {
                    'hard': norm_tied_theta_E_bounds,
                    'plausible': norm_tied_theta_E_plausible_bounds,
                },
                'w': {
                    'hard': norm_tied_w_bounds,
                    'plausible': norm_tied_w_plausible_bounds,
                },
                'del_go': {
                    'hard': norm_tied_del_go_bounds,
                    'plausible': norm_tied_del_go_plausible_bounds,
                },
                'rate_norm_l': {
                    'hard': norm_tied_rate_norm_bounds,
                    'plausible': norm_tied_rate_norm_plausible_bounds,
                },
                'alpha': {
                    'hard': norm_tied_alpha_bounds,
                    'plausible': norm_tied_alpha_plausible_bounds,
                },
            },
            'fit_abl_filter': f"valid RT<1 trials restricted to ABLs {HARD_CODED_DELAY_ABL_LEVELS.astype(int).tolist()}",
            'n_excluded_non_fit_abl_valid_rt_lt_1': int(n_excluded_non_fit_abl),
            'T_trunc': T_trunc,
            'initialization_source': initialization_source,
            'raw_initial_x0': raw_x_0.tolist(),
            'initial_x0': x_0.tolist(),
            'initial_x0_was_clipped': bool(np.any(clipped_init)),
            'vbmc_max_fun_evals': int(max_fun_evals),
            'vbmc_fun_eval_scale': int(VBMC_FUN_EVAL_SCALE),
            'N_sim': int(N_sim),
            'diagnostic_n_jobs': int(DIAGNOSTIC_N_JOBS),
            'fit_random_seed': FIT_RANDOM_SEED,
            'diagnostics_status': (
                'posterior_saved_before_diagnostics'
                if RUN_DIAGNOSTICS_AFTER_FIT else
                'not_run_fit_only_default'
            ),
        },
        'fixed_condition_t_E_aff_ms': fixed_condition_t_E_aff_ms,
    }

    with open(pkl_filename, 'wb') as f:
        pickle.dump(save_dict, f)
    print(f"Saved posterior results before diagnostics to {pkl_filename}")

    if not RUN_DIAGNOSTICS_AFTER_FIT:
        pdf.close()
        print("RUN_DIAGNOSTICS_AFTER_FIT=False: saved fit-only result and skipped simulation diagnostics.")
        print(f"Saved PDF report to {pdf_filename}")
        continue

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

    sim_results = Parallel(n_jobs=DIAGNOSTIC_N_JOBS)(
        delayed(psiam_tied_data_gen_wrapper_rate_norm_alpha_fn)(
            V_A, theta_A, ABL_samples[iter_num], ILD_samples[iter_num],
            rate_lambda, T_0, theta_E, Z_E, t_A_aff,
            get_condition_t_E_aff(batch_name, animal, ABL_samples[iter_num], ILD_samples[iter_num]),
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
        condition_t_E_aff = get_condition_t_E_aff(batch_name, animal, ABL_arg, ILD_arg)
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
        condition_t_E_aff = get_condition_t_E_aff(batch_name, animal, ABL_arg, ILD_arg)
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
        animal, pdf, model_name="Alpha-Normalized Tied Fixed Condition t_E_aff"
    )

    plot_tachometric_curves(
        sim_df_1, data_df_1, ILD_arr, ABL_arr, theory_results_up_and_down,
        theory_time_axis, bins, animal, pdf, model_name="Alpha-Normalized Tied Fixed Condition t_E_aff"
    )

    plot_grand_summary(
        sim_df_1, data_df_1, ILD_arr, ABL_arr, bins, bin_centers,
        animal, pdf, model_name="Alpha-Normalized Tied Fixed Condition t_E_aff"
    )

    # %%
    print("\nGenerating model comparison tables...")
    save_dict['fit_config']['diagnostics_status'] = 'diagnostics_complete'

    abort_df = create_abort_table(save_dict['vbmc_aborts_results'])
    if abort_df is not None:
        render_df_to_pdf(abort_df, f"Abort Model Results - Animal {animal}", pdf)
        print(f"Added abort model results table to PDF for animal {animal}")

    tied_df = create_tied_table(save_dict)
    if tied_df is not None:
        render_df_to_pdf(tied_df, f"Tied Models Comparison - Animal {animal}", pdf)
        print(f"Added alpha-normalized TIED fixed condition t_E_aff results table to PDF for animal {animal}")

    with open(pkl_filename, 'wb') as f:
        pickle.dump(save_dict, f)

    print(f"Saved results to {pkl_filename}")

    pdf.close()
    print(f"Saved PDF report to {pdf_filename}")
