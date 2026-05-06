import numpy as np

from time_vary_norm_utils import (
    M,
    cum_A_t_fn,
    int_phi_fn,
    phi,
    phi_t_fn,
    rho_A_t_fn,
    rho_A_t_VEC_fn,
)


def gamma_omega_alpha_fn(ABL, ILD, rate_lambda, T0, theta_E, rate_norm_l, alpha, is_norm=True):
    if not is_norm:
        rate_norm_l = 0
        alpha = 1

    # Use the repo's existing chi convention so alpha=1 matches
    # time_vary_norm_utils.py exactly.
    chi = 17.37
    abl_term = 10 ** (rate_lambda * (1 - rate_norm_l) * ABL / 20)
    ild_arg = rate_lambda * ILD / chi
    norm_ild_arg = rate_lambda * rate_norm_l * ILD / chi

    r_r = abl_term * np.exp(ild_arg) / (np.exp(norm_ild_arg) + alpha * np.exp(-norm_ild_arg))
    r_l = abl_term * np.exp(-ild_arg) / (np.exp(-norm_ild_arg) + alpha * np.exp(norm_ild_arg))

    r_sum = r_r + r_l
    gamma = theta_E * (r_r - r_l) / r_sum
    omega = r_sum / (T0 * (theta_E ** 2))
    return gamma, omega


def rho_E_minus_small_t_NORM_alpha_time_varying_fn(
        t, bound, ABL, ILD, rate_lambda, T0, theta_E, Z_E, phi_t, int_phi_t,
        rate_norm_l, alpha, is_norm, is_time_vary, K_max):
    """
    PDF of EA hitting `bound`, with alpha-normalized rate model.
    """
    if t <= 0:
        return 0

    v, omega = gamma_omega_alpha_fn(ABL, ILD, rate_lambda, T0, theta_E, rate_norm_l, alpha, is_norm)

    w = 0.5 + (Z_E / (2.0 * theta_E))
    a = 2
    if bound == 1:
        v = -v
        w = 1 - w

    if not is_time_vary:
        int_phi_t = t
        phi_t = 1

    t = omega * int_phi_t

    if t == 0:
        raise ValueError(f't = {t}, for int_phi_t = {int_phi_t} or omega = {omega}')

    non_sum_term = (1 / a**2) * (a**3 / np.sqrt(2 * np.pi * t**3)) * np.exp(-v * a * w - (v**2 * t) / 2)
    K_max = int(K_max / 2)
    k_vals = np.linspace(-K_max, K_max, 2 * K_max + 1)
    sum_w_term = w + 2 * k_vals
    sum_exp_term = np.exp(-(a**2 * (w + 2 * k_vals)**2) / (2 * t))
    sum_result = np.sum(sum_w_term * sum_exp_term)

    density = non_sum_term * sum_result
    if density <= 0:
        density = 1e-16

    return density * (omega * phi_t)


def CDF_E_minus_small_t_NORM_alpha_time_varying_fn(
        t, bound, ABL, ILD, rate_lambda, T0, theta_E, Z_E, int_phi_t,
        rate_norm_l, alpha, is_norm, is_time_vary, K_max):
    """
    CDF of EA hitting `bound`, with alpha-normalized rate model.
    """
    if t <= 0:
        return 0

    v, omega = gamma_omega_alpha_fn(ABL, ILD, rate_lambda, T0, theta_E, rate_norm_l, alpha, is_norm)

    w = 0.5 + (Z_E / (2.0 * theta_E))
    a = 2
    if bound == 1:
        v = -v
        w = 1 - w

    if not is_time_vary:
        int_phi_t = t

    t = omega * int_phi_t

    exponent_arg = -v * a * w - (((v**2) * t) / 2)
    result = np.exp(exponent_arg)

    summation = 0
    for k in range(K_max + 1):
        if k % 2 == 0:
            r_k = k * a + a * w
        else:
            r_k = k * a + a * (1 - w)

        term1 = phi(r_k / np.sqrt(t))
        term2 = M((r_k - v * t) / np.sqrt(t)) + M((r_k + v * t) / np.sqrt(t))

        if np.isnan(term2) or np.isinf(term2):
            print(f'omega = {omega}, T0 = {T0}, theta_E = {theta_E}, rate_lambda = {rate_lambda}, alpha = {alpha}')
            raise ValueError(
                f'term2 = {term2}, v = {v}, t = {t}, r_k = {r_k}, '
                f'M(r_k - v*t) = {M((r_k - v * t) / np.sqrt(t))}, '
                f'M(r_k + v*t) = {M((r_k + v * t) / np.sqrt(t))}'
            )

        summation += ((-1) ** k) * term1 * term2

    return result * summation


def up_or_down_RTs_fit_alpha_fn(
        t, bound,
        V_A, theta_A, t_A_aff,
        t_stim, ABL, ILD, rate_lambda, T0, theta_E, Z_E, t_E_aff, del_go,
        phi_params, rate_norm_l, alpha,
        is_norm, is_time_vary, K_max):
    t1 = max(t - t_stim - t_E_aff, 1e-6)
    t2 = max(t - t_stim - t_E_aff + del_go, 1e-6)

    if is_time_vary:
        int_phi_t_E_g = int_phi_fn(max(t - t_stim - t_E_aff + del_go, 1e-6), phi_params.h1, phi_params.a1, phi_params.b1, phi_params.h2, phi_params.a2)

        phi_t_e = phi_t_fn(max(t - t_stim - t_E_aff, 1e-6), phi_params.h1, phi_params.a1, phi_params.b1, phi_params.h2, phi_params.a2)
        int_phi_t_e = int_phi_fn(max(t - t_stim - t_E_aff, 1e-6), phi_params.h1, phi_params.a1, phi_params.b1, phi_params.h2, phi_params.a2)

        int_phi_t2 = int_phi_fn(t2, phi_params.h1, phi_params.a1, phi_params.b1, phi_params.h2, phi_params.a2)
        int_phi_t1 = int_phi_fn(t1, phi_params.h1, phi_params.a1, phi_params.b1, phi_params.h2, phi_params.a2)

        if int_phi_t_E_g * int_phi_t_e * int_phi_t2 * int_phi_t1 == 0:
            raise ValueError(
                f't = {t}, t_stim = {t_stim}, t_E_aff = {t_E_aff}, '
                f'int_phi values = {int_phi_t_E_g}, {int_phi_t_e}, {int_phi_t2}, {int_phi_t1}'
            )
    else:
        int_phi_t_E_g = np.nan
        phi_t_e = np.nan
        int_phi_t_e = np.nan
        int_phi_t2 = np.nan
        int_phi_t1 = np.nan

    P_A = rho_A_t_fn(t - t_A_aff, V_A, theta_A)
    P_EA_hits_either_bound = CDF_E_minus_small_t_NORM_alpha_time_varying_fn(
        t - t_stim - t_E_aff + del_go, 1,
        ABL, ILD, rate_lambda, T0, theta_E, Z_E, int_phi_t_E_g, rate_norm_l, alpha,
        is_norm, is_time_vary, K_max
    ) + CDF_E_minus_small_t_NORM_alpha_time_varying_fn(
        t - t_stim - t_E_aff + del_go, -1,
        ABL, ILD, rate_lambda, T0, theta_E, Z_E, int_phi_t_E_g, rate_norm_l, alpha,
        is_norm, is_time_vary, K_max
    )

    P_EA_survives = 1 - P_EA_hits_either_bound
    random_readout_if_EA_survives = 0.5 * P_EA_survives

    P_E_plus_cum = CDF_E_minus_small_t_NORM_alpha_time_varying_fn(
        t2, bound,
        ABL, ILD, rate_lambda, T0, theta_E, Z_E, int_phi_t2, rate_norm_l, alpha,
        is_norm, is_time_vary, K_max
    ) - CDF_E_minus_small_t_NORM_alpha_time_varying_fn(
        t1, bound,
        ABL, ILD, rate_lambda, T0, theta_E, Z_E, int_phi_t1, rate_norm_l, alpha,
        is_norm, is_time_vary, K_max
    )

    P_E_plus = rho_E_minus_small_t_NORM_alpha_time_varying_fn(
        t - t_stim - t_E_aff, bound, ABL, ILD, rate_lambda, T0, theta_E, Z_E, phi_t_e, int_phi_t_e,
        rate_norm_l, alpha, is_norm, is_time_vary, K_max
    )

    C_A = cum_A_t_fn(t - t_A_aff, V_A, theta_A)
    return P_A * (random_readout_if_EA_survives + P_E_plus_cum) + P_E_plus * (1 - C_A)


def cum_pro_and_reactive_time_vary_alpha_fn(
        t, c_A_trunc_time,
        V_A, theta_A, t_A_aff,
        t_stim, ABL, ILD, rate_lambda, T0, theta_E, Z_E, t_E_aff,
        phi_params, rate_norm_l, alpha,
        is_norm, is_time_vary, K_max):
    c_A = cum_A_t_fn(t - t_A_aff, V_A, theta_A)
    if c_A_trunc_time is not None:
        if t < c_A_trunc_time:
            c_A = 0
        else:
            c_A /= (1 - cum_A_t_fn(c_A_trunc_time - t_A_aff, V_A, theta_A))

    if is_time_vary:
        int_phi_t_E = int_phi_fn(
            t - t_stim - t_E_aff,
            phi_params.h1, phi_params.a1, phi_params.b1, phi_params.h2, phi_params.a2
        )
    else:
        int_phi_t_E = np.nan

    c_E = CDF_E_minus_small_t_NORM_alpha_time_varying_fn(
        t - t_stim - t_E_aff, 1,
        ABL, ILD, rate_lambda, T0, theta_E, Z_E, int_phi_t_E, rate_norm_l, alpha,
        is_norm, is_time_vary, K_max
    ) + CDF_E_minus_small_t_NORM_alpha_time_varying_fn(
        t - t_stim - t_E_aff, -1,
        ABL, ILD, rate_lambda, T0, theta_E, Z_E, int_phi_t_E, rate_norm_l, alpha,
        is_norm, is_time_vary, K_max
    )

    return c_A + c_E - c_A * c_E


def up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_alpha_fn(
        t, bound,
        P_A, C_A,
        ABL, ILD, rate_lambda, T0, theta_E, Z_E, t_E_aff, del_go,
        phi_params, rate_norm_l, alpha,
        is_norm, is_time_vary, K_max):
    t1 = max(t - t_E_aff, 1e-6)
    t2 = max(t - t_E_aff + del_go, 1e-6)

    if is_time_vary:
        int_phi_t_E_g = int_phi_fn(max(t - t_E_aff + del_go, 1e-6), phi_params.h1, phi_params.a1, phi_params.b1, phi_params.h2, phi_params.a2)

        phi_t_e = phi_t_fn(max(t - t_E_aff, 1e-6), phi_params.h1, phi_params.a1, phi_params.b1, phi_params.h2, phi_params.a2)
        int_phi_t_e = int_phi_fn(max(t - t_E_aff, 1e-6), phi_params.h1, phi_params.a1, phi_params.b1, phi_params.h2, phi_params.a2)

        int_phi_t2 = int_phi_fn(t2, phi_params.h1, phi_params.a1, phi_params.b1, phi_params.h2, phi_params.a2)
        int_phi_t1 = int_phi_fn(t1, phi_params.h1, phi_params.a1, phi_params.b1, phi_params.h2, phi_params.a2)

        if int_phi_t_E_g * int_phi_t_e * int_phi_t2 * int_phi_t1 == 0:
            raise ValueError(
                f't = {t}, t_E_aff = {t_E_aff}, '
                f'int_phi values = {int_phi_t_E_g}, {int_phi_t_e}, {int_phi_t2}, {int_phi_t1}'
            )
    else:
        int_phi_t_E_g = np.nan
        phi_t_e = np.nan
        int_phi_t_e = np.nan
        int_phi_t2 = np.nan
        int_phi_t1 = np.nan

    P_EA_hits_either_bound = CDF_E_minus_small_t_NORM_alpha_time_varying_fn(
        t - t_E_aff + del_go, 1,
        ABL, ILD, rate_lambda, T0, theta_E, Z_E, int_phi_t_E_g, rate_norm_l, alpha,
        is_norm, is_time_vary, K_max
    ) + CDF_E_minus_small_t_NORM_alpha_time_varying_fn(
        t - t_E_aff + del_go, -1,
        ABL, ILD, rate_lambda, T0, theta_E, Z_E, int_phi_t_E_g, rate_norm_l, alpha,
        is_norm, is_time_vary, K_max
    )

    P_EA_survives = 1 - P_EA_hits_either_bound
    random_readout_if_EA_survives = 0.5 * P_EA_survives

    P_E_plus_cum = CDF_E_minus_small_t_NORM_alpha_time_varying_fn(
        t2, bound,
        ABL, ILD, rate_lambda, T0, theta_E, Z_E, int_phi_t2, rate_norm_l, alpha,
        is_norm, is_time_vary, K_max
    ) - CDF_E_minus_small_t_NORM_alpha_time_varying_fn(
        t1, bound,
        ABL, ILD, rate_lambda, T0, theta_E, Z_E, int_phi_t1, rate_norm_l, alpha,
        is_norm, is_time_vary, K_max
    )

    P_E_plus = rho_E_minus_small_t_NORM_alpha_time_varying_fn(
        t - t_E_aff, bound, ABL, ILD, rate_lambda, T0, theta_E, Z_E, phi_t_e, int_phi_t_e,
        rate_norm_l, alpha, is_norm, is_time_vary, K_max
    )

    return P_A * (random_readout_if_EA_survives + P_E_plus_cum) + P_E_plus * (1 - C_A)
