# %%
"""
JAX likelihood helpers for direct Gamma/Omega DDM SVI fits.

These functions are the direct Gamma/Omega version of the tied DDM likelihood
used by the condition-by-condition SVI experiments. They intentionally contain
only math helpers; data loading and NumPyro priors stay visible in the driver
scripts.
"""

# %%
import jax.numpy as jnp
from jax.scipy.special import erf, erfc


# %%
def phi_jax(x):
    return (1.0 / jnp.sqrt(2.0 * jnp.pi)) * jnp.exp(-0.5 * x**2)


def normal_cdf_jax(x):
    return 0.5 * (1.0 + erf(x / jnp.sqrt(2.0)))


def mills_ratio_jax(x):
    x = jnp.clip(x, -37.0, 37.0)
    scaled_x = x / jnp.sqrt(2.0)
    erfcx_approx = jnp.exp(scaled_x**2) * erfc(scaled_x)
    return jnp.sqrt(jnp.pi / 2.0) * erfcx_approx


def rho_A_t_jax(t, V_A, theta_A):
    t = jnp.asarray(t, dtype=jnp.float64)
    safe_t = jnp.maximum(t, 1e-12)
    rho = (
        theta_A
        / jnp.sqrt(2.0 * jnp.pi * safe_t**3)
        * jnp.exp(-0.5 * (V_A**2) * ((safe_t - theta_A / V_A) ** 2) / safe_t)
    )
    return jnp.where(t > 0, rho, 0.0)


def cum_A_t_jax(t, V_A, theta_A):
    t = jnp.asarray(t, dtype=jnp.float64)
    safe_t = jnp.maximum(t, 1e-12)
    term1 = normal_cdf_jax(V_A * (safe_t - theta_A / V_A) / jnp.sqrt(safe_t))
    term2 = jnp.exp(2.0 * V_A * theta_A) * normal_cdf_jax(
        -V_A * (safe_t + theta_A / V_A) / jnp.sqrt(safe_t)
    )
    return jnp.where(t > 0, term1 + term2, 0.0)


def CDF_E_gamma_omega_with_w_jax(t, gamma, omega, bound, w, K_max):
    t_original = jnp.asarray(t, dtype=jnp.float64)
    gamma = jnp.asarray(gamma, dtype=jnp.float64)
    omega = jnp.asarray(omega, dtype=jnp.float64)
    w = jnp.asarray(w, dtype=jnp.float64)
    bound = jnp.asarray(bound)

    a = 2.0
    v = jnp.where(bound == 1, -gamma, gamma)
    w_bound = jnp.where(bound == 1, 1.0 - w, w)

    t_eff = omega * t_original
    shape = jnp.broadcast_shapes(jnp.shape(t_eff), jnp.shape(v), jnp.shape(w_bound))
    valid = jnp.broadcast_to(t_original, shape) > 0
    safe_t = jnp.where(valid, jnp.broadcast_to(t_eff, shape), 1e-12)

    v_full = jnp.broadcast_to(v, shape)
    w_full = jnp.broadcast_to(w_bound, shape)
    result = jnp.exp(-v_full * a * w_full - (v_full**2) * safe_t / 2.0)

    k_arr = jnp.arange(K_max + 1)
    extra_shape = (1,) * len(shape) + (K_max + 1,)
    k_b = k_arr.reshape(extra_shape)
    t_b = safe_t[..., None]
    v_b = v_full[..., None]
    w_b = w_full[..., None]

    r_k = jnp.where(k_b % 2 == 0, k_b * a + a * w_b, k_b * a + a * (1.0 - w_b))
    sqrt_t = jnp.sqrt(t_b)
    term1 = phi_jax(r_k / sqrt_t)
    term2 = mills_ratio_jax((r_k - v_b * t_b) / sqrt_t) + mills_ratio_jax(
        (r_k + v_b * t_b) / sqrt_t
    )
    summation = jnp.sum(((-1.0) ** k_b) * term1 * term2, axis=-1)

    return jnp.where(valid, result * summation, 0.0)


def rho_E_gamma_omega_with_w_jax(t, gamma, omega, bound, w, K_max):
    t_original = jnp.asarray(t, dtype=jnp.float64)
    gamma = jnp.asarray(gamma, dtype=jnp.float64)
    omega = jnp.asarray(omega, dtype=jnp.float64)
    w = jnp.asarray(w, dtype=jnp.float64)
    bound = jnp.asarray(bound)

    a = 2.0
    v = jnp.where(bound == 1, -gamma, gamma)
    w_bound = jnp.where(bound == 1, 1.0 - w, w)

    t_eff = omega * t_original
    shape = jnp.broadcast_shapes(jnp.shape(t_eff), jnp.shape(v), jnp.shape(w_bound))
    valid = jnp.broadcast_to(t_original, shape) > 0
    safe_t = jnp.where(valid, jnp.broadcast_to(t_eff, shape), 1e-12)

    v_full = jnp.broadcast_to(v, shape)
    w_full = jnp.broadcast_to(w_bound, shape)
    non_sum_term = (
        (1.0 / a**2)
        * (a**3 / jnp.sqrt(2.0 * jnp.pi * safe_t**3))
        * jnp.exp(-v_full * a * w_full - (v_full**2) * safe_t / 2.0)
    )

    K_half = int(K_max / 2)
    k_vals = jnp.linspace(-K_half, K_half, 2 * K_half + 1)
    extra_shape = (1,) * len(shape) + (2 * K_half + 1,)
    k_b = k_vals.reshape(extra_shape)
    t_b = safe_t[..., None]
    w_b = w_full[..., None]

    sum_w_term = w_b + 2.0 * k_b
    sum_exp_term = jnp.exp(-(a**2 * (w_b + 2.0 * k_b) ** 2) / (2.0 * t_b))
    sum_result = jnp.sum(sum_w_term * sum_exp_term, axis=-1)

    density = non_sum_term * sum_result
    density = jnp.where(density <= 0, 1e-16, density)
    return jnp.where(valid, density * jnp.broadcast_to(omega, shape), 0.0)


def up_or_down_gamma_omega_with_w_jax(
    t,
    bound,
    V_A,
    theta_A,
    t_A_aff,
    t_stim,
    gamma,
    omega,
    t_E_aff,
    del_go,
    w,
    K_max,
):
    t = jnp.asarray(t, dtype=jnp.float64)
    t_stim = jnp.asarray(t_stim, dtype=jnp.float64)

    t2 = t - t_stim - t_E_aff + del_go
    t1 = t - t_stim - t_E_aff

    p_A = rho_A_t_jax(t - t_A_aff, V_A, theta_A)
    p_EA_hits_either_bound = (
        CDF_E_gamma_omega_with_w_jax(t2, gamma, omega, 1, w, K_max)
        + CDF_E_gamma_omega_with_w_jax(t2, gamma, omega, -1, w, K_max)
    )
    random_readout_if_EA_survives = 0.5 * (1.0 - p_EA_hits_either_bound)

    p_E_bound_cum = CDF_E_gamma_omega_with_w_jax(
        t2, gamma, omega, bound, w, K_max
    ) - CDF_E_gamma_omega_with_w_jax(t1, gamma, omega, bound, w, K_max)
    p_E_bound = rho_E_gamma_omega_with_w_jax(
        t - t_stim - t_E_aff, gamma, omega, bound, w, K_max
    )
    c_A = cum_A_t_jax(t - t_A_aff, V_A, theta_A)

    return p_A * (random_readout_if_EA_survives + p_E_bound_cum) + p_E_bound * (1.0 - c_A)


def cum_pro_and_reactive_gamma_omega_with_w_jax(
    t,
    c_A_trunc_time,
    V_A,
    theta_A,
    t_A_aff,
    t_stim,
    gamma,
    omega,
    t_E_aff,
    w,
    K_max,
):
    t = jnp.asarray(t, dtype=jnp.float64)
    c_A = cum_A_t_jax(t - t_A_aff, V_A, theta_A)
    trunc_denom = 1.0 - cum_A_t_jax(c_A_trunc_time - t_A_aff, V_A, theta_A)
    c_A = jnp.where(t < c_A_trunc_time, 0.0, c_A / trunc_denom)

    c_E = CDF_E_gamma_omega_with_w_jax(
        t - t_stim - t_E_aff, gamma, omega, 1, w, K_max
    ) + CDF_E_gamma_omega_with_w_jax(t - t_stim - t_E_aff, gamma, omega, -1, w, K_max)
    return c_A + c_E - c_A * c_E


def gamma_omega_delay_loglike(params, data, K_max):
    condition_id = data["condition_id"]
    gamma = params["gamma"][condition_id]
    omega = params["omega"][condition_id]
    t_E_aff = params["t_E_aff"][condition_id]

    pdf = up_or_down_gamma_omega_with_w_jax(
        data["total_fix"],
        data["choice"],
        data["V_A"],
        data["theta_A"],
        data["t_A_aff"],
        data["t_stim"],
        gamma,
        omega,
        t_E_aff,
        params["del_go"],
        params["w"],
        K_max,
    )
    trunc_factor = (
        cum_pro_and_reactive_gamma_omega_with_w_jax(
            data["t_stim"] + 1.0,
            data["T_trunc"],
            data["V_A"],
            data["theta_A"],
            data["t_A_aff"],
            data["t_stim"],
            gamma,
            omega,
            t_E_aff,
            params["w"],
            K_max,
        )
        - cum_pro_and_reactive_gamma_omega_with_w_jax(
            data["t_stim"],
            data["T_trunc"],
            data["V_A"],
            data["theta_A"],
            data["t_A_aff"],
            data["t_stim"],
            gamma,
            omega,
            t_E_aff,
            params["w"],
            K_max,
        )
    )

    normalized_pdf = jnp.maximum(pdf / (trunc_factor + 1e-20), 1e-50)
    log_pdf = jnp.log(normalized_pdf)
    return jnp.sum(jnp.where(data["mask"], log_pdf, 0.0))


def gamma_omega_delay_lapse_loglike(params, data, K_max):
    condition_id = data["condition_id"]
    gamma = params["gamma"][condition_id]
    omega = params["omega"][condition_id]
    t_E_aff = params["t_E_aff"][condition_id]

    pdf = up_or_down_gamma_omega_with_w_jax(
        data["total_fix"],
        data["choice"],
        data["V_A"],
        data["theta_A"],
        data["t_A_aff"],
        data["t_stim"],
        gamma,
        omega,
        t_E_aff,
        params["del_go"],
        params["w"],
        K_max,
    )
    trunc_factor = (
        cum_pro_and_reactive_gamma_omega_with_w_jax(
            data["t_stim"] + 1.0,
            data["T_trunc"],
            data["V_A"],
            data["theta_A"],
            data["t_A_aff"],
            data["t_stim"],
            gamma,
            omega,
            t_E_aff,
            params["w"],
            K_max,
        )
        - cum_pro_and_reactive_gamma_omega_with_w_jax(
            data["t_stim"],
            data["T_trunc"],
            data["V_A"],
            data["theta_A"],
            data["t_A_aff"],
            data["t_stim"],
            gamma,
            omega,
            t_E_aff,
            params["w"],
            K_max,
        )
    )

    ddm_pdf = jnp.maximum(pdf / (trunc_factor + 1e-10), 1e-10)
    lapse_choice_pdf = jnp.where(
        data["choice"] == 1,
        params["lapse_prob_right"],
        1.0 - params["lapse_prob_right"],
    )
    mixed_pdf = (1.0 - params["lapse_prob"]) * ddm_pdf + params["lapse_prob"] * lapse_choice_pdf
    log_pdf = jnp.log(jnp.maximum(mixed_pdf, 1e-50))
    return jnp.sum(jnp.where(data["mask"], log_pdf, 0.0))
