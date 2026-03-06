import numpy as np

from time_vary_norm_utils import (
    CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn,
    cum_A_t_fn,
    int_phi_fn,
    phi_t_fn,
    rho_A_t_fn,
    rho_E_minus_small_t_NORM_rate_norm_time_varying_fn,
)


def lapse_pdf(t, beta_lapse, eps=1e-50):
    """Exponential lapse PDF over absolute time t."""
    if t < 0:
        return 0.0
    val = beta_lapse * np.exp(-beta_lapse * t)
    if not np.isfinite(val) or val <= 0:
        return eps
    return float(val)


def lapse_cdf(t, beta_lapse):
    """Exponential lapse CDF over absolute time t."""
    if t <= 0:
        return 0.0
    val = 1.0 - np.exp(-beta_lapse * t)
    return float(np.clip(val, 0.0, 1.0))


def compute_proactive_reactive_blocks_no_lapse(
    t,
    bound,
    V_A,
    theta_A,
    t_A_aff,
    t_stim,
    ABL,
    ILD,
    rate_lambda,
    T0,
    theta_E,
    Z_E,
    t_E_aff,
    del_go,
    phi_params,
    rate_norm_l,
    is_norm,
    is_time_vary,
    K_max,
):
    """
    Compute the no-lapse proactive/reactive decomposition used by up_or_down_RTs_fit_fn.
    """
    t1 = max(t - t_stim - t_E_aff, 1e-6)
    t2 = max(t - t_stim - t_E_aff + del_go, 1e-6)

    if is_time_vary:
        int_phi_t_E_g = int_phi_fn(
            max(t - t_stim - t_E_aff + del_go, 1e-6),
            phi_params.h1,
            phi_params.a1,
            phi_params.b1,
            phi_params.h2,
            phi_params.a2,
        )
        phi_t_e = phi_t_fn(
            max(t - t_stim - t_E_aff, 1e-6),
            phi_params.h1,
            phi_params.a1,
            phi_params.b1,
            phi_params.h2,
            phi_params.a2,
        )
        int_phi_t_e = int_phi_fn(
            max(t - t_stim - t_E_aff, 1e-6),
            phi_params.h1,
            phi_params.a1,
            phi_params.b1,
            phi_params.h2,
            phi_params.a2,
        )
        int_phi_t2 = int_phi_fn(
            t2, phi_params.h1, phi_params.a1, phi_params.b1, phi_params.h2, phi_params.a2
        )
        int_phi_t1 = int_phi_fn(
            t1, phi_params.h1, phi_params.a1, phi_params.b1, phi_params.h2, phi_params.a2
        )
    else:
        int_phi_t_E_g = np.nan
        phi_t_e = np.nan
        int_phi_t_e = np.nan
        int_phi_t2 = np.nan
        int_phi_t1 = np.nan

    P_A = rho_A_t_fn(t - t_A_aff, V_A, theta_A)
    C_A = cum_A_t_fn(t - t_A_aff, V_A, theta_A)

    P_EA_hits_either_bound = CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn(
        t - t_stim - t_E_aff + del_go,
        1,
        ABL,
        ILD,
        rate_lambda,
        T0,
        theta_E,
        Z_E,
        int_phi_t_E_g,
        rate_norm_l,
        is_norm,
        is_time_vary,
        K_max,
    ) + CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn(
        t - t_stim - t_E_aff + del_go,
        -1,
        ABL,
        ILD,
        rate_lambda,
        T0,
        theta_E,
        Z_E,
        int_phi_t_E_g,
        rate_norm_l,
        is_norm,
        is_time_vary,
        K_max,
    )
    P_EA_survives = 1.0 - P_EA_hits_either_bound
    random_readout_if_EA_survives = 0.5 * P_EA_survives

    P_E_plus_cum = CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn(
        t2,
        bound,
        ABL,
        ILD,
        rate_lambda,
        T0,
        theta_E,
        Z_E,
        int_phi_t2,
        rate_norm_l,
        is_norm,
        is_time_vary,
        K_max,
    ) - CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn(
        t1,
        bound,
        ABL,
        ILD,
        rate_lambda,
        T0,
        theta_E,
        Z_E,
        int_phi_t1,
        rate_norm_l,
        is_norm,
        is_time_vary,
        K_max,
    )

    P_E_plus = rho_E_minus_small_t_NORM_rate_norm_time_varying_fn(
        t - t_stim - t_E_aff,
        bound,
        ABL,
        ILD,
        rate_lambda,
        T0,
        theta_E,
        Z_E,
        phi_t_e,
        int_phi_t_e,
        rate_norm_l,
        is_norm,
        is_time_vary,
        K_max,
    )

    proactive_block = P_A * (random_readout_if_EA_survives + P_E_plus_cum)
    reactive_block = P_E_plus * (1.0 - C_A)

    return {
        "P_A": float(P_A),
        "C_A": float(C_A),
        "random_readout_if_EA_survives": float(random_readout_if_EA_survives),
        "P_E_plus_cum": float(P_E_plus_cum),
        "P_E_plus": float(P_E_plus),
        "proactive_block": float(proactive_block),
        "reactive_block": float(reactive_block),
    }


def up_or_down_RTs_fit_proactive_lapse_only_fn(
    t,
    bound,
    V_A,
    theta_A,
    t_A_aff,
    t_stim,
    ABL,
    ILD,
    rate_lambda,
    T0,
    theta_E,
    Z_E,
    t_E_aff,
    del_go,
    phi_params,
    rate_norm_l,
    is_norm,
    is_time_vary,
    K_max,
    lapse_prob,
    beta_lapse,
    lapse_choice_prob=0.5,
    eps=1e-50,
):
    """
    Lapse-aware tied-model PDF where lapse is injected only through the proactive terms.

    P_A_mix = (1-lapse_prob)*P_A + lapse_prob*lapse_choice_prob*lapse_pdf(t)
    C_A_mix = (1-lapse_prob)*C_A + lapse_prob*lapse_cdf(t)
    pdf = P_A_mix*(random_readout_if_EA_survives + P_E_plus_cum) + P_E_plus*(1-C_A_mix)
    """
    blocks = compute_proactive_reactive_blocks_no_lapse(
        t=t,
        bound=bound,
        V_A=V_A,
        theta_A=theta_A,
        t_A_aff=t_A_aff,
        t_stim=t_stim,
        ABL=ABL,
        ILD=ILD,
        rate_lambda=rate_lambda,
        T0=T0,
        theta_E=theta_E,
        Z_E=Z_E,
        t_E_aff=t_E_aff,
        del_go=del_go,
        phi_params=phi_params,
        rate_norm_l=rate_norm_l,
        is_norm=is_norm,
        is_time_vary=is_time_vary,
        K_max=K_max,
    )

    lapse_prob = float(np.clip(lapse_prob, 0.0, 1.0))
    lapse_choice_prob = float(np.clip(lapse_choice_prob, 0.0, 1.0))

    P_A_mix = (1.0 - lapse_prob) * blocks["P_A"] + lapse_prob * lapse_choice_prob * lapse_pdf(
        t, beta_lapse, eps=eps
    )
    C_A_mix = (1.0 - lapse_prob) * blocks["C_A"] + lapse_prob * lapse_cdf(t, beta_lapse)
    C_A_mix = float(np.clip(C_A_mix, 0.0, 1.0))

    pdf = P_A_mix * (
        blocks["random_readout_if_EA_survives"] + blocks["P_E_plus_cum"]
    ) + blocks["P_E_plus"] * (1.0 - C_A_mix)

    if (not np.isfinite(pdf)) or pdf <= 0:
        pdf = eps
    return float(pdf)


def cdf_proactive_with_lapse_plus_reactive_no_trunc_fn(
    t,
    V_A,
    theta_A,
    t_A_aff,
    t_stim,
    ABL,
    ILD,
    rate_lambda,
    T0,
    theta_E,
    Z_E,
    t_E_aff,
    phi_params,
    rate_norm_l,
    is_norm,
    is_time_vary,
    K_max,
    lapse_prob,
    beta_lapse,
):
    """
    Total CDF (any decision) with proactive+lapse and reactive, without truncation correction.
    """
    if t <= 0:
        return 0.0

    lapse_prob = float(np.clip(lapse_prob, 0.0, 1.0))

    C_A = cum_A_t_fn(t - t_A_aff, V_A, theta_A)
    C_A_mix = (1.0 - lapse_prob) * C_A + lapse_prob * lapse_cdf(t, beta_lapse)

    if is_time_vary:
        int_phi_t_E = int_phi_fn(
            t - t_stim - t_E_aff,
            phi_params.h1,
            phi_params.a1,
            phi_params.b1,
            phi_params.h2,
            phi_params.a2,
        )
    else:
        int_phi_t_E = np.nan

    C_E = CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn(
        t - t_stim - t_E_aff,
        1,
        ABL,
        ILD,
        rate_lambda,
        T0,
        theta_E,
        Z_E,
        int_phi_t_E,
        rate_norm_l,
        is_norm,
        is_time_vary,
        K_max,
    ) + CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn(
        t - t_stim - t_E_aff,
        -1,
        ABL,
        ILD,
        rate_lambda,
        T0,
        theta_E,
        Z_E,
        int_phi_t_E,
        rate_norm_l,
        is_norm,
        is_time_vary,
        K_max,
    )

    cdf_total = C_A_mix + C_E - C_A_mix * C_E
    return float(np.clip(cdf_total, 0.0, 1.0))


def trial_logpdf_proactive_lapse_only_no_trunc(
    row,
    V_A,
    theta_A,
    t_A_aff,
    rate_lambda,
    T0,
    theta_E,
    Z_E,
    t_E_aff,
    del_go,
    phi_params,
    rate_norm_l,
    is_norm,
    is_time_vary,
    K_max,
    lapse_prob,
    beta_lapse,
    lapse_choice_prob=0.5,
    censor_rt_wrt_stim=0.15,
    eps=1e-50,
):
    """Row-level log-likelihood for no-truncation proactive-lapse-only tied likelihood."""
    rt = float(row["TotalFixTime"])
    t_stim = float(row["intended_fix"])
    ILD = float(row["ILD"])
    ABL = float(row["ABL"])
    choice = int(row["choice"])

    if (censor_rt_wrt_stim is not None) and ((rt - t_stim) > censor_rt_wrt_stim):
        t_censor = t_stim + float(censor_rt_wrt_stim)
        cdf_at_censor = cdf_proactive_with_lapse_plus_reactive_no_trunc_fn(
            t=t_censor,
            V_A=V_A,
            theta_A=theta_A,
            t_A_aff=t_A_aff,
            t_stim=t_stim,
            ABL=ABL,
            ILD=ILD,
            rate_lambda=rate_lambda,
            T0=T0,
            theta_E=theta_E,
            Z_E=Z_E,
            t_E_aff=t_E_aff,
            phi_params=phi_params,
            rate_norm_l=rate_norm_l,
            is_norm=is_norm,
            is_time_vary=is_time_vary,
            K_max=K_max,
            lapse_prob=lapse_prob,
            beta_lapse=beta_lapse,
        )
        survival = max(1.0 - cdf_at_censor, eps)
        return float(np.log(survival))

    pdf = up_or_down_RTs_fit_proactive_lapse_only_fn(
        t=rt,
        bound=choice,
        V_A=V_A,
        theta_A=theta_A,
        t_A_aff=t_A_aff,
        t_stim=t_stim,
        ABL=ABL,
        ILD=ILD,
        rate_lambda=rate_lambda,
        T0=T0,
        theta_E=theta_E,
        Z_E=Z_E,
        t_E_aff=t_E_aff,
        del_go=del_go,
        phi_params=phi_params,
        rate_norm_l=rate_norm_l,
        is_norm=is_norm,
        is_time_vary=is_time_vary,
        K_max=K_max,
        lapse_prob=lapse_prob,
        beta_lapse=beta_lapse,
        lapse_choice_prob=lapse_choice_prob,
        eps=eps,
    )
    return float(np.log(max(pdf, eps)))


def trial_logpdf_proactive_lapse_only_no_trunc_right_truncated(
    row,
    V_A,
    theta_A,
    t_A_aff,
    rate_lambda,
    T0,
    theta_E,
    Z_E,
    t_E_aff,
    del_go,
    phi_params,
    rate_norm_l,
    is_norm,
    is_time_vary,
    K_max,
    lapse_prob,
    beta_lapse,
    lapse_choice_prob=0.5,
    truncate_rt_wrt_stim=0.13,
    eps=1e-50,
):
    """Row-level log-likelihood with right-truncation in RT relative to stimulus onset."""
    rt = float(row["TotalFixTime"])
    t_stim = float(row["intended_fix"])
    ILD = float(row["ILD"])
    ABL = float(row["ABL"])
    choice = int(row["choice"])

    if truncate_rt_wrt_stim is None:
        truncate_rt_wrt_stim = np.inf
    else:
        truncate_rt_wrt_stim = float(truncate_rt_wrt_stim)

    if (rt - t_stim) > truncate_rt_wrt_stim:
        return -np.inf

    pdf = up_or_down_RTs_fit_proactive_lapse_only_fn(
        t=rt,
        bound=choice,
        V_A=V_A,
        theta_A=theta_A,
        t_A_aff=t_A_aff,
        t_stim=t_stim,
        ABL=ABL,
        ILD=ILD,
        rate_lambda=rate_lambda,
        T0=T0,
        theta_E=theta_E,
        Z_E=Z_E,
        t_E_aff=t_E_aff,
        del_go=del_go,
        phi_params=phi_params,
        rate_norm_l=rate_norm_l,
        is_norm=is_norm,
        is_time_vary=is_time_vary,
        K_max=K_max,
        lapse_prob=lapse_prob,
        beta_lapse=beta_lapse,
        lapse_choice_prob=lapse_choice_prob,
        eps=eps,
    )

    cdf_lower = cdf_proactive_with_lapse_plus_reactive_no_trunc_fn(
        t=t_stim,
        V_A=V_A,
        theta_A=theta_A,
        t_A_aff=t_A_aff,
        t_stim=t_stim,
        ABL=ABL,
        ILD=ILD,
        rate_lambda=rate_lambda,
        T0=T0,
        theta_E=theta_E,
        Z_E=Z_E,
        t_E_aff=t_E_aff,
        phi_params=phi_params,
        rate_norm_l=rate_norm_l,
        is_norm=is_norm,
        is_time_vary=is_time_vary,
        K_max=K_max,
        lapse_prob=lapse_prob,
        beta_lapse=beta_lapse,
    )
    cdf_upper = cdf_proactive_with_lapse_plus_reactive_no_trunc_fn(
        t=t_stim + truncate_rt_wrt_stim,
        V_A=V_A,
        theta_A=theta_A,
        t_A_aff=t_A_aff,
        t_stim=t_stim,
        ABL=ABL,
        ILD=ILD,
        rate_lambda=rate_lambda,
        T0=T0,
        theta_E=theta_E,
        Z_E=Z_E,
        t_E_aff=t_E_aff,
        phi_params=phi_params,
        rate_norm_l=rate_norm_l,
        is_norm=is_norm,
        is_time_vary=is_time_vary,
        K_max=K_max,
        lapse_prob=lapse_prob,
        beta_lapse=beta_lapse,
    )

    truncation_mass = cdf_upper - cdf_lower
    if (not np.isfinite(truncation_mass)) or truncation_mass <= 0:
        truncation_mass = eps

    return float(np.log(max(pdf, eps)) - np.log(truncation_mass))
