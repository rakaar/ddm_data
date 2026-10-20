# %%
"""
JAX/NumPyro helpers for vanilla/IPL condition-delay SVI fits.

This is a focused port of the old vanilla branch used through
`time_vary_norm_utils.up_or_down_RTs_fit_fn` with:

    is_norm = False
    is_time_vary = False
    rate_norm_l = 0

The goal is to fit vanilla/IPL parameters directly, not free Gamma/Omega
condition parameters. The condition Gamma/Omega quantities are deterministic
functions of `rate_lambda`, `T_0`, and `theta_E`, matching the old likelihood.
"""

# %%
from collections import OrderedDict

import numpy as np

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import erf, erfc
from numpyro.infer.autoguide import AutoMultivariateNormal, AutoNormal
from numpyro.infer.initialization import init_to_value


# %%
# =============================================================================
# Parameter metadata
# =============================================================================
GLOBAL_PARAM_NAMES = [
    "rate_lambda",
    "T_0",
    "theta_E",
    "w",
    "del_go",
]

GLOBAL_PARAM_LABELS = {
    "rate_lambda": "lambda",
    "T_0": "T_0",
    "theta_E": "theta_E",
    "w": "w",
    "del_go": "del_go",
}

GLOBAL_BOUNDS = OrderedDict(
    [
        ("rate_lambda", {"hard": (0.01, 1.0), "plausible": (0.1, 0.3)}),
        ("T_0", {"hard": (0.1e-3, 2.2e-3), "plausible": (0.5e-3, 1.5e-3)}),
        ("theta_E", {"hard": (5.0, 65.0), "plausible": (15.0, 55.0)}),
        ("w", {"hard": (0.3, 0.7), "plausible": (0.4, 0.6)}),
        ("del_go", {"hard": (0.0, 0.2), "plausible": (0.05, 0.15)}),
    ]
)

DELAY_BOUNDS = {"hard": (0.01, 0.2), "plausible": (0.03, 0.09)}

DEFAULT_INIT_VALUES = {
    "rate_lambda": 0.17,
    "T_0": 1.4e-3,
    "theta_E": 20.0,
    "w": 0.51,
    "del_go": 0.13,
}


# %%
# =============================================================================
# Small JAX math helpers
# =============================================================================
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


def vanilla_gamma_omega_jax(ABL, ILD, rate_lambda, T_0, theta_E):
    """
    Deterministic vanilla/IPL Gamma/Omega used by the old likelihood.

    The `cosh(rate_lambda * ILD / chi)` factor is present because this function
    ports the exact `time_vary_norm_utils.py` branch with `is_norm=False`.
    """
    chi = 17.37
    ABL = jnp.asarray(ABL, dtype=jnp.float64)
    ILD = jnp.asarray(ILD, dtype=jnp.float64)
    ild_arg = rate_lambda * ILD / chi
    gamma = theta_E * jnp.tanh(ild_arg)
    omega = (1.0 / (T_0 * theta_E**2)) * (10.0 ** (rate_lambda * ABL / 20.0)) * jnp.cosh(ild_arg)
    return gamma, omega


def CDF_E_vanilla_jax(t, bound, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, K_max):
    t_original = jnp.asarray(t, dtype=jnp.float64)
    bound = jnp.asarray(bound)
    gamma, omega = vanilla_gamma_omega_jax(ABL, ILD, rate_lambda, T_0, theta_E)

    w = 0.5 + (Z_E / (2.0 * theta_E))
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
    term2 = mills_ratio_jax((r_k - v_b * t_b) / sqrt_t) + mills_ratio_jax((r_k + v_b * t_b) / sqrt_t)
    summation = jnp.sum(((-1.0) ** k_b) * term1 * term2, axis=-1)

    return jnp.where(valid, result * summation, 0.0)


def rho_E_vanilla_jax(t, bound, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, K_max):
    t_original = jnp.asarray(t, dtype=jnp.float64)
    bound = jnp.asarray(bound)
    gamma, omega = vanilla_gamma_omega_jax(ABL, ILD, rate_lambda, T_0, theta_E)

    w = 0.5 + (Z_E / (2.0 * theta_E))
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


# %%
# =============================================================================
# Vanilla/IPL proactive + evidence likelihood pieces
# =============================================================================
def up_or_down_RTs_fit_vanilla_jax(
    t,
    bound,
    V_A,
    theta_A,
    t_A_aff,
    t_stim,
    ABL,
    ILD,
    rate_lambda,
    T_0,
    theta_E,
    Z_E,
    t_E_aff,
    del_go,
    K_max,
):
    t = jnp.asarray(t, dtype=jnp.float64)
    t_stim = jnp.asarray(t_stim, dtype=jnp.float64)

    t1 = jnp.maximum(t - t_stim - t_E_aff, 1e-6)
    t2 = jnp.maximum(t - t_stim - t_E_aff + del_go, 1e-6)

    p_A = rho_A_t_jax(t - t_A_aff, V_A, theta_A)
    p_EA_hits_either_bound = (
        CDF_E_vanilla_jax(t - t_stim - t_E_aff + del_go, 1, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, K_max)
        + CDF_E_vanilla_jax(
            t - t_stim - t_E_aff + del_go, -1, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, K_max
        )
    )
    random_readout_if_EA_survives = 0.5 * (1.0 - p_EA_hits_either_bound)

    p_E_bound_cum = CDF_E_vanilla_jax(
        t2, bound, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, K_max
    ) - CDF_E_vanilla_jax(t1, bound, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, K_max)
    p_E_bound = rho_E_vanilla_jax(
        t - t_stim - t_E_aff,
        bound,
        ABL,
        ILD,
        rate_lambda,
        T_0,
        theta_E,
        Z_E,
        K_max,
    )
    c_A = cum_A_t_jax(t - t_A_aff, V_A, theta_A)

    return p_A * (random_readout_if_EA_survives + p_E_bound_cum) + p_E_bound * (1.0 - c_A)


def cum_pro_and_reactive_vanilla_jax(
    t,
    c_A_trunc_time,
    V_A,
    theta_A,
    t_A_aff,
    t_stim,
    ABL,
    ILD,
    rate_lambda,
    T_0,
    theta_E,
    Z_E,
    t_E_aff,
    K_max,
):
    t = jnp.asarray(t, dtype=jnp.float64)
    c_A = cum_A_t_jax(t - t_A_aff, V_A, theta_A)
    if c_A_trunc_time is not None:
        trunc_denom = 1.0 - cum_A_t_jax(c_A_trunc_time - t_A_aff, V_A, theta_A)
        c_A = jnp.where(t < c_A_trunc_time, 0.0, c_A / trunc_denom)

    c_E = (
        CDF_E_vanilla_jax(t - t_stim - t_E_aff, 1, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, K_max)
        + CDF_E_vanilla_jax(t - t_stim - t_E_aff, -1, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, K_max)
    )
    return c_A + c_E - c_A * c_E


def vanilla_condition_delay_loglike_terms(params, data, K_max=10):
    t_E_aff = params["t_E_aff"][data["condition_id"]]
    Z_E = (params["w"] - 0.5) * 2.0 * params["theta_E"]

    pdf = up_or_down_RTs_fit_vanilla_jax(
        data["total_fix"],
        data["choice"],
        data["V_A"],
        data["theta_A"],
        data["t_A_aff"],
        data["t_stim"],
        data["ABL"],
        data["ILD"],
        params["rate_lambda"],
        params["T_0"],
        params["theta_E"],
        Z_E,
        t_E_aff,
        params["del_go"],
        K_max,
    )
    trunc_factor = (
        cum_pro_and_reactive_vanilla_jax(
            data["t_stim"] + 1.0,
            data["T_trunc"],
            data["V_A"],
            data["theta_A"],
            data["t_A_aff"],
            data["t_stim"],
            data["ABL"],
            data["ILD"],
            params["rate_lambda"],
            params["T_0"],
            params["theta_E"],
            Z_E,
            t_E_aff,
            K_max,
        )
        - cum_pro_and_reactive_vanilla_jax(
            data["t_stim"],
            data["T_trunc"],
            data["V_A"],
            data["theta_A"],
            data["t_A_aff"],
            data["t_stim"],
            data["ABL"],
            data["ILD"],
            params["rate_lambda"],
            params["T_0"],
            params["theta_E"],
            Z_E,
            t_E_aff,
            K_max,
        )
    )

    normalized_pdf = jnp.maximum(pdf / (trunc_factor + 1e-20), 1e-50)
    return pdf, trunc_factor, jnp.log(normalized_pdf)


def vanilla_condition_delay_loglike(params, data, K_max=10):
    _pdf, _trunc_factor, log_pdf = vanilla_condition_delay_loglike_terms(params, data, K_max=K_max)
    return jnp.sum(log_pdf)


# %%
# =============================================================================
# Priors, model, guide, and summary helpers
# =============================================================================
def trapezoidal_logpdf_jax(x, hard_low, plausible_low, plausible_high, hard_high):
    x = jnp.asarray(x)
    area = ((plausible_low - hard_low) + (hard_high - plausible_high)) / 2.0 + (
        plausible_high - plausible_low
    )
    h_max = 1.0 / area

    rising = ((x - hard_low) / (plausible_low - hard_low)) * h_max
    flat = jnp.full_like(x, h_max, dtype=jnp.result_type(x, jnp.float64))
    falling = ((hard_high - x) / (hard_high - plausible_high)) * h_max

    pdf = jnp.where(
        (hard_low <= x) & (x <= plausible_low),
        rising,
        jnp.where(
            (plausible_low < x) & (x < plausible_high),
            flat,
            jnp.where((plausible_high <= x) & (x <= hard_high), falling, 0.0),
        ),
    )
    return jnp.where(pdf > 0, jnp.log(pdf), -jnp.inf)


def sample_trapezoid(name, hard, plausible):
    hard_low, hard_high = hard
    plausible_low, plausible_high = plausible
    value = numpyro.sample(name, dist.Uniform(hard_low, hard_high))
    target_logpdf = trapezoidal_logpdf_jax(value, hard_low, plausible_low, plausible_high, hard_high)
    uniform_logpdf = -jnp.log(hard_high - hard_low)
    numpyro.factor(f"{name}_trapezoid_prior", target_logpdf - uniform_logpdf)
    return value


def sample_trapezoid_vector(name, n, hard, plausible):
    hard_low, hard_high = hard
    plausible_low, plausible_high = plausible
    value = numpyro.sample(name, dist.Uniform(hard_low, hard_high).expand([n]).to_event(1))
    target_logpdf = trapezoidal_logpdf_jax(value, hard_low, plausible_low, plausible_high, hard_high)
    uniform_logpdf = -jnp.log(hard_high - hard_low)
    numpyro.factor(f"{name}_trapezoid_prior", jnp.sum(target_logpdf - uniform_logpdf))
    return value


def vanilla_condition_delay_model(data, n_conditions, K_max=10):
    params = {}
    for name, bounds in GLOBAL_BOUNDS.items():
        params[name] = sample_trapezoid(name, bounds["hard"], bounds["plausible"])

    params["t_E_aff"] = sample_trapezoid_vector(
        "t_E_aff",
        n_conditions,
        DELAY_BOUNDS["hard"],
        DELAY_BOUNDS["plausible"],
    )

    loglike = vanilla_condition_delay_loglike(params, data, K_max=K_max)
    numpyro.factor("ddm_loglike", loglike)


def make_guide(model, guide_kind, init_values, init_scale=0.1):
    init_loc_fn = init_to_value(values=init_values)
    guide_kind = guide_kind.lower()
    if guide_kind in {"meanfield", "autonormal", "normal"}:
        return AutoNormal(model, init_loc_fn=init_loc_fn)
    if guide_kind in {"fullrank", "multivariate", "automultivariate"}:
        return AutoMultivariateNormal(model, init_loc_fn=init_loc_fn, init_scale=init_scale)
    raise ValueError(f"Use fullrank or meanfield guide for vanilla SVI, got {guide_kind!r}.")


def tree_all_finite(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return bool(all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves))


def posterior_samples_to_frame(samples, condition_table):
    import pandas as pd

    rows = []
    for name in GLOBAL_PARAM_NAMES:
        values = np.asarray(samples[name], dtype=float)
        finite_values = values[np.isfinite(values)]
        rows.append(
            {
                "parameter": name,
                "mean": float(np.mean(finite_values)) if finite_values.size else np.nan,
                "sd": float(np.std(finite_values)) if finite_values.size else np.nan,
                "q025": float(np.quantile(finite_values, 0.025)) if finite_values.size else np.nan,
                "q500": float(np.quantile(finite_values, 0.5)) if finite_values.size else np.nan,
                "q975": float(np.quantile(finite_values, 0.975)) if finite_values.size else np.nan,
                "n_samples": int(values.size),
                "n_finite": int(finite_values.size),
            }
        )

    delay_values = np.asarray(samples["t_E_aff"], dtype=float)
    for idx, condition in condition_table.reset_index(drop=True).iterrows():
        values = delay_values[:, idx]
        finite_values = values[np.isfinite(values)]
        rows.append(
            {
                "parameter": f"t_E_aff_ABL{int(condition['ABL'])}_ILD{condition['ILD']:g}",
                "mean": float(np.mean(finite_values)) if finite_values.size else np.nan,
                "sd": float(np.std(finite_values)) if finite_values.size else np.nan,
                "q025": float(np.quantile(finite_values, 0.025)) if finite_values.size else np.nan,
                "q500": float(np.quantile(finite_values, 0.5)) if finite_values.size else np.nan,
                "q975": float(np.quantile(finite_values, 0.975)) if finite_values.size else np.nan,
                "n_samples": int(values.size),
                "n_finite": int(finite_values.size),
                "ABL": int(condition["ABL"]),
                "ILD": float(condition["ILD"]),
                "condition_id": int(idx),
            }
        )

    return pd.DataFrame(rows)


def finite_sample_report(samples):
    import pandas as pd

    rows = []
    all_finite = True
    for key, value in samples.items():
        arr = np.asarray(value)
        finite = np.isfinite(arr)
        all_finite = all_finite and bool(np.all(finite))
        rows.append(
            {
                "parameter": key,
                "shape": str(arr.shape),
                "n_total": int(arr.size),
                "n_finite": int(np.sum(finite)),
                "n_nan": int(np.sum(np.isnan(arr))),
                "n_inf": int(np.sum(np.isinf(arr))),
            }
        )
    return pd.DataFrame(rows), all_finite


def clip_init_to_hard_bounds(init_values, eps=1e-6):
    clipped = dict(init_values)
    for name, bounds in GLOBAL_BOUNDS.items():
        hard_low, hard_high = bounds["hard"]
        width = hard_high - hard_low
        clipped[name] = np.clip(float(clipped[name]), hard_low + eps * width, hard_high - eps * width)

    hard_low, hard_high = DELAY_BOUNDS["hard"]
    width = hard_high - hard_low
    clipped["t_E_aff"] = np.clip(
        np.asarray(clipped["t_E_aff"], dtype=float),
        hard_low + eps * width,
        hard_high - eps * width,
    )
    return clipped


def tree_to_numpy(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(x), tree)

