# %%
"""
Test an analytic integral of the bound-specific reactive DDM CDF.

The intended use is a uniform evidence-afferent delay,

    D ~ Uniform(d_low, d_high),

for which the delay-marginalized reactive CDF can be written as

    [H(t - d_low) - H(t - d_high)] / (d_high - d_low),

where H(x) = integral_0^x F_E(u) du.  This script validates a spectral-series
candidate for H against numerical integration of the CDF currently used by the
NPL+alpha likelihood.  It does not modify the production likelihood.
"""

# %%
from pathlib import Path
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid, quad
from scipy.special import erfcx


SCRIPT_DIR = Path(__file__).resolve().parent
ANIMAL_FIT_DIR = SCRIPT_DIR.parent
REPO_DIR = ANIMAL_FIT_DIR.parent
sys.path.insert(0, str(ANIMAL_FIT_DIR))

import numpyro_npl_alpha_svi_utils as npl_utils


# %%
# =============================================================================
# Editable validation settings
# =============================================================================
SPECTRAL_TERMS = (5, 10, 20)
CURRENT_CDF_K_MAX = 10

GAMMA_GRID = (-3.0, -1.5, 0.0, 1.5, 3.0)
OMEGA_GRID = (1.75, 4.3, 11.6)
W_GRID = (0.35, 0.50, 0.65)
BOUNDS = (-1, 1)
TIME_MS = (0.01, 0.03, 0.1, 0.3, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0)

UNIFORM_DELAY_CENTER_S = 0.080
UNIFORM_DELAY_WIDTHS_S = (0.005, 0.020, 0.100)
UNIFORM_GAMMA_GRID = (-3.0, 0.0, 3.0)
UNIFORM_OMEGA_GRID = (1.75, 4.3, 11.6)
UNIFORM_W_GRID = (0.35, 0.50, 0.65)

REFERENCE_FIT_ROOT = (
    ANIMAL_FIT_DIR
    / "numpyro_svi_npl_alpha_condition_delay_patience12_restore_best_outputs"
)
OUTPUT_DIR = SCRIPT_DIR / "validation_outputs"

QUAD_EPSABS = 1e-11
QUAD_EPSREL = 1e-9
CDF_ERROR_TOL = 1e-6
PDF_REL_ERROR_TOL = 1e-5
PDF_NONNEGLIGIBLE_THRESHOLD = 1e-7
MATERIAL_NEGATIVE_TOL = -1e-10

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Current-CDF image-series K_max: {CURRENT_CDF_K_MAX}")
print(f"Candidate spectral term counts: {SPECTRAL_TERMS}")
print(f"Reference fit root: {REFERENCE_FIT_ROOT}")
print(f"Output directory: {OUTPUT_DIR}")


# %%
# =============================================================================
# Candidate analytic integral
# =============================================================================
def lower_bound_hit_probability_jax(v, w, boundary_separation=2.0):
    """Eventual probability of hitting the transformed lower DDM bound."""
    v = jnp.asarray(v, dtype=jnp.float64)
    w = jnp.asarray(w, dtype=jnp.float64)
    q = -2.0 * v * boundary_separation
    near_zero = jnp.abs(q) < 1e-8
    safe_q = jnp.where(near_zero, 1.0, q)
    drifted_probability = (
        jnp.exp(safe_q * w)
        * jnp.expm1(safe_q * (1.0 - w))
        / jnp.expm1(safe_q)
    )
    return jnp.where(near_zero, 1.0 - w, drifted_probability)


def integrated_CDF_E_gamma_omega_with_w_jax(
    t,
    gamma,
    omega,
    bound,
    w,
    n_terms,
):
    """Spectral candidate for H(t) = integral_0^t F_bound(u) du."""
    t_original = jnp.asarray(t, dtype=jnp.float64)
    gamma = jnp.asarray(gamma, dtype=jnp.float64)
    omega = jnp.asarray(omega, dtype=jnp.float64)
    bound = jnp.asarray(bound)
    w = jnp.asarray(w, dtype=jnp.float64)

    boundary_separation = 2.0
    v = jnp.where(bound == 1, -gamma, gamma)
    transformed_w = jnp.where(bound == 1, 1.0 - w, w)

    shape = jnp.broadcast_shapes(
        jnp.shape(t_original),
        jnp.shape(v),
        jnp.shape(omega),
        jnp.shape(transformed_w),
    )
    t_full = jnp.broadcast_to(t_original, shape)
    v_full = jnp.broadcast_to(v, shape)
    omega_full = jnp.broadcast_to(omega, shape)
    w_full = jnp.broadcast_to(transformed_w, shape)
    valid = t_full > 0.0
    tau = jnp.where(valid, omega_full * t_full, 0.0)

    k = jnp.arange(1, n_terms + 1, dtype=jnp.float64)
    k = k.reshape((1,) * len(shape) + (n_terms,))
    v_terms = v_full[..., None]
    w_terms = w_full[..., None]
    tau_terms = tau[..., None]

    eigenvalue = 0.5 * (
        v_terms**2 + (k * jnp.pi / boundary_separation) ** 2
    )
    coefficient = (
        jnp.pi
        / boundary_separation**2
        * jnp.exp(-v_terms * boundary_separation * w_terms)
        * k
        * jnp.sin(k * jnp.pi * w_terms)
    )
    integrated_transient = jnp.sum(
        coefficient * (-jnp.expm1(-eigenvalue * tau_terms)) / eigenvalue**2,
        axis=-1,
    )
    eventual_probability = lower_bound_hit_probability_jax(
        v_full,
        w_full,
        boundary_separation,
    )
    integrated_cdf = (
        eventual_probability * tau - integrated_transient
    ) / omega_full
    return jnp.where(valid, integrated_cdf, 0.0)


def integrated_CDF_E_alpha_jax(
    t,
    bound,
    ABL,
    ILD,
    rate_lambda,
    T_0,
    theta_E,
    Z_E,
    rate_norm_l,
    alpha,
    n_terms,
):
    """NPL+alpha parameter wrapper around the generic Gamma/Omega integral."""
    gamma, omega = npl_utils.gamma_omega_alpha_jax(
        ABL,
        ILD,
        rate_lambda,
        T_0,
        theta_E,
        rate_norm_l,
        alpha,
    )
    w = 0.5 + Z_E / (2.0 * theta_E)
    return integrated_CDF_E_gamma_omega_with_w_jax(
        t,
        gamma,
        omega,
        bound,
        w,
        n_terms,
    )


# %%
# =============================================================================
# Independent NumPy implementations used as numerical references
# =============================================================================
def lower_bound_hit_probability_numpy(v, w, boundary_separation=2.0):
    v = np.asarray(v, dtype=float)
    w = np.asarray(w, dtype=float)
    q = -2.0 * v * boundary_separation
    near_zero = np.abs(q) < 1e-8
    safe_q = np.where(near_zero, 1.0, q)
    drifted_probability = (
        np.exp(safe_q * w)
        * np.expm1(safe_q * (1.0 - w))
        / np.expm1(safe_q)
    )
    return np.where(near_zero, 1.0 - w, drifted_probability)


def integrated_CDF_E_gamma_omega_with_w_numpy(
    t,
    gamma,
    omega,
    bound,
    w,
    n_terms,
):
    t = np.asarray(t, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    omega = np.asarray(omega, dtype=float)
    bound = np.asarray(bound)
    w = np.asarray(w, dtype=float)

    boundary_separation = 2.0
    v = np.where(bound == 1, -gamma, gamma)
    transformed_w = np.where(bound == 1, 1.0 - w, w)
    shape = np.broadcast(t, v, omega, transformed_w).shape
    t_full = np.broadcast_to(t, shape)
    v_full = np.broadcast_to(v, shape)
    omega_full = np.broadcast_to(omega, shape)
    w_full = np.broadcast_to(transformed_w, shape)
    valid = t_full > 0.0
    tau = np.where(valid, omega_full * t_full, 0.0)

    k = np.arange(1, n_terms + 1, dtype=float)
    k = k.reshape((1,) * len(shape) + (n_terms,))
    eigenvalue = 0.5 * (
        v_full[..., None] ** 2 + (k * np.pi / boundary_separation) ** 2
    )
    coefficient = (
        np.pi
        / boundary_separation**2
        * np.exp(-v_full[..., None] * boundary_separation * w_full[..., None])
        * k
        * np.sin(k * np.pi * w_full[..., None])
    )
    integrated_transient = np.sum(
        coefficient
        * (-np.expm1(-eigenvalue * tau[..., None]))
        / eigenvalue**2,
        axis=-1,
    )
    eventual_probability = lower_bound_hit_probability_numpy(
        v_full,
        w_full,
        boundary_separation,
    )
    integrated_cdf = (
        eventual_probability * tau - integrated_transient
    ) / omega_full
    return np.where(valid, integrated_cdf, 0.0)


def current_CDF_E_gamma_omega_with_w_numpy(
    t,
    gamma,
    omega,
    bound,
    w,
    K_max=CURRENT_CDF_K_MAX,
):
    """Independent NumPy copy of the current small-time image-series CDF."""
    t_original = np.asarray(t, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    omega = np.asarray(omega, dtype=float)
    bound = np.asarray(bound)
    w = np.asarray(w, dtype=float)

    boundary_separation = 2.0
    v = np.where(bound == 1, -gamma, gamma)
    transformed_w = np.where(bound == 1, 1.0 - w, w)
    t_eff = omega * t_original
    shape = np.broadcast(t_eff, v, transformed_w).shape
    valid = np.broadcast_to(t_original, shape) > 0.0
    safe_t = np.where(valid, np.broadcast_to(t_eff, shape), 1e-12)
    v_full = np.broadcast_to(v, shape)
    w_full = np.broadcast_to(transformed_w, shape)
    leading = np.exp(
        -v_full * boundary_separation * w_full - v_full**2 * safe_t / 2.0
    )

    k = np.arange(K_max + 1)
    k = k.reshape((1,) * len(shape) + (K_max + 1,))
    r_k = np.where(
        k % 2 == 0,
        k * boundary_separation + boundary_separation * w_full[..., None],
        k * boundary_separation
        + boundary_separation * (1.0 - w_full[..., None]),
    )
    sqrt_t = np.sqrt(safe_t[..., None])
    z = r_k / sqrt_t
    normal_pdf = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    left_mills = np.sqrt(np.pi / 2.0) * erfcx(
        np.clip(
            (r_k - v_full[..., None] * safe_t[..., None]) / sqrt_t,
            -37.0,
            37.0,
        )
        / np.sqrt(2.0)
    )
    right_mills = np.sqrt(np.pi / 2.0) * erfcx(
        np.clip(
            (r_k + v_full[..., None] * safe_t[..., None]) / sqrt_t,
            -37.0,
            37.0,
        )
        / np.sqrt(2.0)
    )
    summation = np.sum(
        (-1.0) ** k * normal_pdf * (left_mills + right_mills),
        axis=-1,
    )
    return np.where(valid, leading * summation, 0.0)


def current_rho_E_gamma_omega_with_w_numpy(
    t,
    gamma,
    omega,
    bound,
    w,
    K_max=CURRENT_CDF_K_MAX,
):
    """Independent NumPy copy of the current reactive bound density."""
    t_original = np.asarray(t, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    omega = np.asarray(omega, dtype=float)
    bound = np.asarray(bound)
    w = np.asarray(w, dtype=float)

    boundary_separation = 2.0
    v = np.where(bound == 1, -gamma, gamma)
    transformed_w = np.where(bound == 1, 1.0 - w, w)
    t_eff = omega * t_original
    shape = np.broadcast(t_eff, v, transformed_w).shape
    valid = np.broadcast_to(t_original, shape) > 0.0
    safe_t = np.where(valid, np.broadcast_to(t_eff, shape), 1e-12)
    v_full = np.broadcast_to(v, shape)
    w_full = np.broadcast_to(transformed_w, shape)
    leading = (
        boundary_separation
        / np.sqrt(2.0 * np.pi * safe_t**3)
        * np.exp(
            -v_full * boundary_separation * w_full
            - v_full**2 * safe_t / 2.0
        )
    )

    K_half = int(K_max / 2)
    k = np.linspace(-K_half, K_half, 2 * K_half + 1)
    k = k.reshape((1,) * len(shape) + (2 * K_half + 1,))
    shifted_w = w_full[..., None] + 2.0 * k
    summation = np.sum(
        shifted_w
        * np.exp(
            -boundary_separation**2
            * shifted_w**2
            / (2.0 * safe_t[..., None])
        ),
        axis=-1,
    )
    density = leading * summation
    density = np.where(density <= 0.0, 1e-16, density)
    return np.where(valid, density * np.broadcast_to(omega, shape), 0.0)


# %%
# =============================================================================
# Select real posterior-mean cases spanning the fitted parameter range
# =============================================================================
if not REFERENCE_FIT_ROOT.exists():
    raise FileNotFoundError(REFERENCE_FIT_ROOT)

all_real_rows = []
posterior_files = sorted(
    REFERENCE_FIT_ROOT.glob("*/main_fullrank_posterior_samples.npz")
)
if len(posterior_files) != 30:
    raise RuntimeError(
        f"Expected 30 posterior files in {REFERENCE_FIT_ROOT}, found {len(posterior_files)}."
    )

for posterior_file in posterior_files:
    fit_dir = posterior_file.parent
    condition_file = fit_dir / "condition_table.csv"
    if not condition_file.exists():
        raise FileNotFoundError(condition_file)

    with np.load(posterior_file) as saved:
        parameter_means = {
            name: float(np.mean(saved[name]))
            for name in npl_utils.GLOBAL_PARAM_NAMES
        }

    condition_table = pd.read_csv(condition_file).sort_values("condition_id")
    gamma, omega = npl_utils.gamma_omega_alpha_jax(
        condition_table["ABL"].to_numpy(dtype=float),
        condition_table["ILD"].to_numpy(dtype=float),
        parameter_means["rate_lambda"],
        parameter_means["T_0"],
        parameter_means["theta_E"],
        parameter_means["rate_norm_l"],
        parameter_means["alpha"],
    )
    gamma = np.asarray(gamma)
    omega = np.asarray(omega)

    batch_name, animal_text = fit_dir.name.rsplit("_", 1)
    for row, condition_gamma, condition_omega in zip(
        condition_table.itertuples(),
        gamma,
        omega,
    ):
        all_real_rows.append(
            {
                "batch": batch_name,
                "animal": int(animal_text),
                "condition_id": int(row.condition_id),
                "ABL": float(row.ABL),
                "ILD": float(row.ILD),
                "gamma": float(condition_gamma),
                "omega": float(condition_omega),
                "w": parameter_means["w"],
                **parameter_means,
            }
        )

all_real_df = pd.DataFrame(all_real_rows)
selection_reasons = {}

for column in ["gamma", "omega", "w"]:
    for direction, row_index in [
        ("minimum", all_real_df[column].idxmin()),
        ("maximum", all_real_df[column].idxmax()),
    ]:
        selection_reasons.setdefault(int(row_index), []).append(
            f"{column}_{direction}"
        )

median_values = all_real_df[["gamma", "omega", "w"]].median()
scale_values = (
    all_real_df[["gamma", "omega", "w"]].quantile(0.75)
    - all_real_df[["gamma", "omega", "w"]].quantile(0.25)
).replace(0.0, 1.0)
median_distance = (
    (all_real_df[["gamma", "omega", "w"]] - median_values)
    / scale_values
).pow(2).sum(axis=1)
selection_reasons.setdefault(int(median_distance.idxmin()), []).append(
    "closest_to_joint_median"
)

representative_df = all_real_df.loc[sorted(selection_reasons)].copy()
representative_df.insert(
    0,
    "selection_reason",
    [";".join(selection_reasons[int(index)]) for index in representative_df.index],
)
representative_df = representative_df.reset_index(drop=True)

wrapper_differences = []
current_cdf_reference_differences = []
current_pdf_reference_differences = []
wrapper_times = jnp.asarray([0.001, 0.010, 0.100, 0.500, 1.000])
for row in representative_df.itertuples():
    Z_E = (row.w - 0.5) * 2.0 * row.theta_E
    differences = []
    cdf_reference_differences = []
    pdf_reference_differences = []
    for n_terms in SPECTRAL_TERMS:
        for bound in BOUNDS:
            wrapped = integrated_CDF_E_alpha_jax(
                wrapper_times,
                bound,
                row.ABL,
                row.ILD,
                row.rate_lambda,
                row.T_0,
                row.theta_E,
                Z_E,
                row.rate_norm_l,
                row.alpha,
                n_terms,
            )
            direct = integrated_CDF_E_gamma_omega_with_w_jax(
                wrapper_times,
                row.gamma,
                row.omega,
                bound,
                row.w,
                n_terms,
            )
            differences.append(float(jnp.max(jnp.abs(wrapped - direct))))
    for bound in BOUNDS:
        current_jax_cdf = npl_utils.CDF_E_alpha_jax(
            wrapper_times,
            bound,
            row.ABL,
            row.ILD,
            row.rate_lambda,
            row.T_0,
            row.theta_E,
            Z_E,
            row.rate_norm_l,
            row.alpha,
            CURRENT_CDF_K_MAX,
        )
        current_numpy_cdf = current_CDF_E_gamma_omega_with_w_numpy(
            np.asarray(wrapper_times),
            row.gamma,
            row.omega,
            bound,
            row.w,
        )
        current_jax_pdf = npl_utils.rho_E_alpha_jax(
            wrapper_times,
            bound,
            row.ABL,
            row.ILD,
            row.rate_lambda,
            row.T_0,
            row.theta_E,
            Z_E,
            row.rate_norm_l,
            row.alpha,
            CURRENT_CDF_K_MAX,
        )
        current_numpy_pdf = current_rho_E_gamma_omega_with_w_numpy(
            np.asarray(wrapper_times),
            row.gamma,
            row.omega,
            bound,
            row.w,
        )
        cdf_reference_differences.append(
            float(np.max(np.abs(np.asarray(current_jax_cdf) - current_numpy_cdf)))
        )
        pdf_reference_differences.append(
            float(np.max(np.abs(np.asarray(current_jax_pdf) - current_numpy_pdf)))
        )
    wrapper_differences.append(max(differences))
    current_cdf_reference_differences.append(max(cdf_reference_differences))
    current_pdf_reference_differences.append(max(pdf_reference_differences))

representative_df["max_npl_wrapper_abs_difference"] = wrapper_differences
representative_df["max_current_cdf_reference_abs_difference"] = (
    current_cdf_reference_differences
)
representative_df["max_current_pdf_reference_abs_difference"] = (
    current_pdf_reference_differences
)
representative_csv = OUTPUT_DIR / "representative_real_parameter_cases.csv"
representative_df.to_csv(representative_csv, index=False)

print(
    "Real posterior-mean mapped ranges: "
    f"Gamma=[{all_real_df['gamma'].min():.6g}, {all_real_df['gamma'].max():.6g}], "
    f"Omega=[{all_real_df['omega'].min():.6g}, {all_real_df['omega'].max():.6g}], "
    f"w=[{all_real_df['w'].min():.6g}, {all_real_df['w'].max():.6g}]"
)
print(f"Selected {len(representative_df)} representative real conditions.")


# %%
# =============================================================================
# H(t) convergence and dH/dt = F(t) checks
# =============================================================================
base_cases = []
for gamma in GAMMA_GRID:
    for omega in OMEGA_GRID:
        for w in W_GRID:
            for bound in BOUNDS:
                for time_ms in TIME_MS:
                    base_cases.append(
                        {
                            "case_source": "synthetic_grid",
                            "case_label": "synthetic",
                            "batch": "",
                            "animal": np.nan,
                            "ABL": np.nan,
                            "ILD": np.nan,
                            "gamma": gamma,
                            "omega": omega,
                            "w": w,
                            "bound": bound,
                            "time_s": time_ms / 1000.0,
                            "time_ms": time_ms,
                        }
                    )

for row in representative_df.itertuples():
    for bound in BOUNDS:
        for time_ms in TIME_MS:
            base_cases.append(
                {
                    "case_source": "real_posterior_mean",
                    "case_label": row.selection_reason,
                    "batch": row.batch,
                    "animal": row.animal,
                    "ABL": row.ABL,
                    "ILD": row.ILD,
                    "gamma": row.gamma,
                    "omega": row.omega,
                    "w": row.w,
                    "bound": bound,
                    "time_s": time_ms / 1000.0,
                    "time_ms": time_ms,
                }
            )

base_df = pd.DataFrame(base_cases)
reference_h = []
reference_cdf = []
for row in base_df.itertuples():
    cdf_at_t = float(
        current_CDF_E_gamma_omega_with_w_numpy(
            row.time_s,
            row.gamma,
            row.omega,
            row.bound,
            row.w,
        )
    )
    integral, _ = quad(
        lambda integration_time: float(
            current_CDF_E_gamma_omega_with_w_numpy(
                integration_time,
                row.gamma,
                row.omega,
                row.bound,
                row.w,
            )
        ),
        0.0,
        row.time_s,
        epsabs=QUAD_EPSABS,
        epsrel=QUAD_EPSREL,
        limit=200,
    )
    reference_cdf.append(cdf_at_t)
    reference_h.append(integral)

base_df["reference_cdf"] = reference_cdf
base_df["reference_integrated_cdf"] = reference_h

time_jax = jnp.asarray(base_df["time_s"].to_numpy(dtype=float))
gamma_jax = jnp.asarray(base_df["gamma"].to_numpy(dtype=float))
omega_jax = jnp.asarray(base_df["omega"].to_numpy(dtype=float))
bound_jax = jnp.asarray(base_df["bound"].to_numpy(dtype=int))
w_jax = jnp.asarray(base_df["w"].to_numpy(dtype=float))

convergence_frames = []
for n_terms in SPECTRAL_TERMS:
    def scalar_integrated_cdf(time_value, gamma_value, omega_value, bound_value, w_value):
        return integrated_CDF_E_gamma_omega_with_w_jax(
            time_value,
            gamma_value,
            omega_value,
            bound_value,
            w_value,
            n_terms,
        )

    batched_integrated_cdf = jax.jit(jax.vmap(scalar_integrated_cdf))
    batched_derivative = jax.jit(
        jax.vmap(jax.grad(scalar_integrated_cdf, argnums=0))
    )
    analytic_h = np.asarray(
        batched_integrated_cdf(
            time_jax,
            gamma_jax,
            omega_jax,
            bound_jax,
            w_jax,
        )
    )
    derivative = np.asarray(
        batched_derivative(
            time_jax,
            gamma_jax,
            omega_jax,
            bound_jax,
            w_jax,
        )
    )
    numpy_h = integrated_CDF_E_gamma_omega_with_w_numpy(
        base_df["time_s"].to_numpy(dtype=float),
        base_df["gamma"].to_numpy(dtype=float),
        base_df["omega"].to_numpy(dtype=float),
        base_df["bound"].to_numpy(dtype=int),
        base_df["w"].to_numpy(dtype=float),
        n_terms,
    )
    np.testing.assert_allclose(analytic_h, numpy_h, rtol=5e-13, atol=5e-14)

    frame = base_df.copy()
    frame["spectral_terms"] = n_terms
    frame["analytic_integrated_cdf"] = analytic_h
    frame["integrated_cdf_abs_error"] = np.abs(
        analytic_h - frame["reference_integrated_cdf"]
    )
    frame["integrated_cdf_rel_error"] = frame["integrated_cdf_abs_error"] / np.maximum(
        np.abs(frame["reference_integrated_cdf"]),
        1e-12,
    )
    frame["jax_time_derivative"] = derivative
    frame["derivative_cdf_abs_error"] = np.abs(
        derivative - frame["reference_cdf"]
    )
    convergence_frames.append(frame)

convergence_df = pd.concat(convergence_frames, ignore_index=True)
convergence_csv = OUTPUT_DIR / "integrated_cdf_term_convergence.csv"
convergence_df.to_csv(convergence_csv, index=False)


# %%
# =============================================================================
# Uniform-delay CDF and PDF identities
# =============================================================================
uniform_rows = []
for gamma in UNIFORM_GAMMA_GRID:
    for omega in UNIFORM_OMEGA_GRID:
        for w in UNIFORM_W_GRID:
            for bound in BOUNDS:
                for width in UNIFORM_DELAY_WIDTHS_S:
                    delay_low = UNIFORM_DELAY_CENTER_S - width / 2.0
                    delay_high = UNIFORM_DELAY_CENTER_S + width / 2.0
                    offsets = (
                        -0.020,
                        0.0,
                        0.25 * width,
                        0.50 * width,
                        0.75 * width,
                        width,
                        width + 0.020,
                        0.200,
                        0.500,
                        0.920,
                    )
                    for offset in offsets:
                        observed_time = delay_low + offset
                        split_points = (
                            [observed_time]
                            if delay_low < observed_time < delay_high
                            else None
                        )
                        numerical_convolved_cdf, _ = quad(
                            lambda delay: float(
                                current_CDF_E_gamma_omega_with_w_numpy(
                                    observed_time - delay,
                                    gamma,
                                    omega,
                                    bound,
                                    w,
                                )
                            ),
                            delay_low,
                            delay_high,
                            points=split_points,
                            epsabs=QUAD_EPSABS,
                            epsrel=QUAD_EPSREL,
                            limit=200,
                        )
                        numerical_convolved_cdf /= width

                        numerical_convolved_pdf, _ = quad(
                            lambda delay: float(
                                current_rho_E_gamma_omega_with_w_numpy(
                                    observed_time - delay,
                                    gamma,
                                    omega,
                                    bound,
                                    w,
                                )
                            ),
                            delay_low,
                            delay_high,
                            points=split_points,
                            epsabs=QUAD_EPSABS,
                            epsrel=QUAD_EPSREL,
                            limit=200,
                        )
                        numerical_convolved_pdf /= width

                        cdf_difference_pdf = (
                            current_CDF_E_gamma_omega_with_w_numpy(
                                observed_time - delay_low,
                                gamma,
                                omega,
                                bound,
                                w,
                            )
                            - current_CDF_E_gamma_omega_with_w_numpy(
                                observed_time - delay_high,
                                gamma,
                                omega,
                                bound,
                                w,
                            )
                        ) / width
                        pdf_abs_error = abs(
                            float(cdf_difference_pdf) - numerical_convolved_pdf
                        )
                        pdf_rel_error = (
                            pdf_abs_error / abs(numerical_convolved_pdf)
                            if abs(numerical_convolved_pdf)
                            > PDF_NONNEGLIGIBLE_THRESHOLD
                            else np.nan
                        )

                        for n_terms in SPECTRAL_TERMS:
                            convolved_cdf = (
                                integrated_CDF_E_gamma_omega_with_w_numpy(
                                    observed_time - delay_low,
                                    gamma,
                                    omega,
                                    bound,
                                    w,
                                    n_terms,
                                )
                                - integrated_CDF_E_gamma_omega_with_w_numpy(
                                    observed_time - delay_high,
                                    gamma,
                                    omega,
                                    bound,
                                    w,
                                    n_terms,
                                )
                            ) / width
                            uniform_rows.append(
                                {
                                    "gamma": gamma,
                                    "omega": omega,
                                    "w": w,
                                    "bound": bound,
                                    "delay_low_s": delay_low,
                                    "delay_high_s": delay_high,
                                    "delay_width_s": width,
                                    "observed_time_s": observed_time,
                                    "spectral_terms": n_terms,
                                    "numerical_convolved_cdf": numerical_convolved_cdf,
                                    "analytic_convolved_cdf": float(convolved_cdf),
                                    "convolved_cdf_abs_error": abs(
                                        float(convolved_cdf)
                                        - numerical_convolved_cdf
                                    ),
                                    "numerical_convolved_pdf": numerical_convolved_pdf,
                                    "cdf_difference_convolved_pdf": float(
                                        cdf_difference_pdf
                                    ),
                                    "convolved_pdf_abs_error": pdf_abs_error,
                                    "convolved_pdf_rel_error": pdf_rel_error,
                                }
                            )

uniform_df = pd.DataFrame(uniform_rows)
uniform_csv = OUTPUT_DIR / "uniform_delay_convolution_validation.csv"
uniform_df.to_csv(uniform_csv, index=False)


# %%
# =============================================================================
# Width -> 0 recovery of a fixed delay
# =============================================================================
limit_rows = []
limit_gamma = 1.5
limit_omega = 4.3
limit_w = 0.5
limit_bound = -1
limit_delay = 0.080

for width in (1e-3, 1e-4, 1e-5):
    delay_low = limit_delay - width / 2.0
    delay_high = limit_delay + width / 2.0
    for observed_time in (0.081, 0.100, 0.200, 0.500):
        fixed_cdf = float(
            current_CDF_E_gamma_omega_with_w_numpy(
                observed_time - limit_delay,
                limit_gamma,
                limit_omega,
                limit_bound,
                limit_w,
            )
        )
        fixed_pdf = float(
            current_rho_E_gamma_omega_with_w_numpy(
                observed_time - limit_delay,
                limit_gamma,
                limit_omega,
                limit_bound,
                limit_w,
            )
        )
        cdf_difference_pdf = float(
            (
                current_CDF_E_gamma_omega_with_w_numpy(
                    observed_time - delay_low,
                    limit_gamma,
                    limit_omega,
                    limit_bound,
                    limit_w,
                )
                - current_CDF_E_gamma_omega_with_w_numpy(
                    observed_time - delay_high,
                    limit_gamma,
                    limit_omega,
                    limit_bound,
                    limit_w,
                )
            )
            / width
        )
        for n_terms in SPECTRAL_TERMS:
            uniform_cdf = float(
                (
                    integrated_CDF_E_gamma_omega_with_w_numpy(
                        observed_time - delay_low,
                        limit_gamma,
                        limit_omega,
                        limit_bound,
                        limit_w,
                        n_terms,
                    )
                    - integrated_CDF_E_gamma_omega_with_w_numpy(
                        observed_time - delay_high,
                        limit_gamma,
                        limit_omega,
                        limit_bound,
                        limit_w,
                        n_terms,
                    )
                )
                / width
            )
            limit_rows.append(
                {
                    "delay_width_s": width,
                    "observed_time_s": observed_time,
                    "spectral_terms": n_terms,
                    "fixed_delay_cdf": fixed_cdf,
                    "uniform_delay_cdf": uniform_cdf,
                    "cdf_abs_difference": abs(uniform_cdf - fixed_cdf),
                    "fixed_delay_pdf": fixed_pdf,
                    "uniform_delay_pdf": cdf_difference_pdf,
                    "pdf_abs_difference": abs(cdf_difference_pdf - fixed_pdf),
                }
            )

limit_df = pd.DataFrame(limit_rows)
limit_csv = OUTPUT_DIR / "uniform_delay_zero_width_limit.csv"
limit_df.to_csv(limit_csv, index=False)


# %%
# =============================================================================
# Summaries and feasibility decision
# =============================================================================
summary_rows = []
max_wrapper_difference = float(
    representative_df["max_npl_wrapper_abs_difference"].max()
)
max_current_cdf_reference_difference = float(
    representative_df["max_current_cdf_reference_abs_difference"].max()
)
max_current_pdf_reference_difference = float(
    representative_df["max_current_pdf_reference_abs_difference"].max()
)

for n_terms in SPECTRAL_TERMS:
    convergence_subset = convergence_df[
        convergence_df["spectral_terms"] == n_terms
    ]
    uniform_subset = uniform_df[uniform_df["spectral_terms"] == n_terms]
    finite_pdf_relative_errors = uniform_subset[
        "convolved_pdf_rel_error"
    ].dropna()
    row = {
        "spectral_terms": n_terms,
        "max_integrated_cdf_abs_error": convergence_subset[
            "integrated_cdf_abs_error"
        ].max(),
        "p99_integrated_cdf_abs_error": convergence_subset[
            "integrated_cdf_abs_error"
        ].quantile(0.99),
        "max_derivative_cdf_abs_error": convergence_subset[
            "derivative_cdf_abs_error"
        ].max(),
        "max_uniform_cdf_abs_error": uniform_subset[
            "convolved_cdf_abs_error"
        ].max(),
        "max_uniform_pdf_abs_error": uniform_subset[
            "convolved_pdf_abs_error"
        ].max(),
        "max_uniform_pdf_rel_error_non_negligible": finite_pdf_relative_errors.max(),
        "minimum_analytic_integrated_cdf": convergence_subset[
            "analytic_integrated_cdf"
        ].min(),
        "materially_negative_integrated_cdf_count": int(
            (
                convergence_subset["analytic_integrated_cdf"]
                < MATERIAL_NEGATIVE_TOL
            ).sum()
        ),
        "max_npl_wrapper_abs_difference": max_wrapper_difference,
        "max_current_cdf_reference_abs_difference": (
            max_current_cdf_reference_difference
        ),
        "max_current_pdf_reference_abs_difference": (
            max_current_pdf_reference_difference
        ),
        "max_zero_width_cdf_abs_difference": limit_df.loc[
            limit_df["spectral_terms"] == n_terms,
            "cdf_abs_difference",
        ].max(),
        "max_zero_width_pdf_abs_difference": limit_df.loc[
            limit_df["spectral_terms"] == n_terms,
            "pdf_abs_difference",
        ].max(),
    }
    row["passes_main_validation"] = bool(
        row["max_integrated_cdf_abs_error"] < CDF_ERROR_TOL
        and row["max_derivative_cdf_abs_error"] < CDF_ERROR_TOL
        and row["max_uniform_cdf_abs_error"] < CDF_ERROR_TOL
        and row["max_uniform_pdf_rel_error_non_negligible"]
        < PDF_REL_ERROR_TOL
        and row["materially_negative_integrated_cdf_count"] == 0
        and row["max_npl_wrapper_abs_difference"] < 1e-10
        and row["max_current_cdf_reference_abs_difference"] < 1e-10
        and row["max_current_pdf_reference_abs_difference"] < 1e-10
    )
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_csv = OUTPUT_DIR / "integrated_cdf_validation_summary.csv"
summary_df.to_csv(summary_csv, index=False)

passing_terms = summary_df.loc[
    summary_df["passes_main_validation"],
    "spectral_terms",
].tolist()
selected_terms = min(passing_terms) if passing_terms else None

print("\nValidation summary:")
print(summary_df.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
if selected_terms is None:
    print(
        "\nRESULT: No tested term count passes the validation tolerances. "
        "Do not replace the production likelihood with this spectral "
        "antiderivative as written."
    )
else:
    print(
        f"\nRESULT: K={selected_terms} is the smallest tested term count "
        "that passes all main checks."
    )


# %%
# =============================================================================
# Diagnostic figure
# =============================================================================
largest_terms = max(SPECTRAL_TERMS)
largest_terms_df = convergence_df[
    convergence_df["spectral_terms"] == largest_terms
]
worst_row = largest_terms_df.loc[
    largest_terms_df["integrated_cdf_abs_error"].idxmax()
]

dense_time_s = np.concatenate(([0.0], np.geomspace(1e-5, 1.0, 240)))
dense_reference_h = np.array(
    [
        quad(
            lambda integration_time: float(
                current_CDF_E_gamma_omega_with_w_numpy(
                    integration_time,
                    worst_row["gamma"],
                    worst_row["omega"],
                    int(worst_row["bound"]),
                    worst_row["w"],
                )
            ),
            0.0,
            time_value,
            epsabs=QUAD_EPSABS,
            epsrel=QUAD_EPSREL,
            limit=200,
        )[0]
        for time_value in dense_time_s
    ]
)

fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
ax = axes[0, 0]
ax.plot(1e3 * dense_time_s, dense_reference_h, color="black", lw=2.0, label="Numerical integral")
for n_terms in SPECTRAL_TERMS:
    dense_candidate = integrated_CDF_E_gamma_omega_with_w_numpy(
        dense_time_s,
        worst_row["gamma"],
        worst_row["omega"],
        int(worst_row["bound"]),
        worst_row["w"],
        n_terms,
    )
    ax.plot(1e3 * dense_time_s, dense_candidate, lw=1.2, label=f"K={n_terms}")
ax.set_xscale("log")
ax.set_xlabel("Reactive time (ms)")
ax.set_ylabel("Integrated bound CDF, H(t)")
ax.legend(frameon=False, fontsize=8)
ax.set_title(
    "Hardest tested case: "
    f"Gamma={worst_row['gamma']:.3g}, Omega={worst_row['omega']:.3g}, "
    f"w={worst_row['w']:.3g}, bound={int(worst_row['bound'])}"
)

ax = axes[0, 1]
for n_terms in SPECTRAL_TERMS:
    dense_candidate = integrated_CDF_E_gamma_omega_with_w_numpy(
        dense_time_s,
        worst_row["gamma"],
        worst_row["omega"],
        int(worst_row["bound"]),
        worst_row["w"],
        n_terms,
    )
    ax.plot(
        1e3 * dense_time_s[1:],
        np.abs(dense_candidate[1:] - dense_reference_h[1:]),
        lw=1.2,
        label=f"K={n_terms}",
    )
ax.axhline(CDF_ERROR_TOL, color="black", ls="--", lw=1.0, label="Tolerance")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Reactive time (ms)")
ax.set_ylabel("Absolute H(t) error")
ax.legend(frameon=False, fontsize=8)

ax = axes[1, 0]
ax.plot(
    summary_df["spectral_terms"],
    summary_df["max_integrated_cdf_abs_error"],
    "o-",
    label="H vs numerical integral",
)
ax.plot(
    summary_df["spectral_terms"],
    summary_df["max_derivative_cdf_abs_error"],
    "s-",
    label="dH/dt vs current CDF",
)
ax.axhline(CDF_ERROR_TOL, color="black", ls="--", lw=1.0, label="Tolerance")
ax.set_yscale("log")
ax.set_xticks(SPECTRAL_TERMS)
ax.set_xlabel("Positive spectral terms, K")
ax.set_ylabel("Maximum absolute error")
ax.legend(frameon=False, fontsize=8)

ax = axes[1, 1]
ax.plot(
    summary_df["spectral_terms"],
    summary_df["max_uniform_cdf_abs_error"],
    "o-",
    color="tab:purple",
    label="Uniform-delay CDF",
)
ax.axhline(CDF_ERROR_TOL, color="black", ls="--", lw=1.0, label="Tolerance")
ax.set_yscale("log")
ax.set_xticks(SPECTRAL_TERMS)
ax.set_xlabel("Positive spectral terms, K")
ax.set_ylabel("Maximum absolute error")
ax.legend(frameon=False, fontsize=8)

fig.suptitle("Uniform-delay analytic integrated-CDF feasibility", y=0.995)
fig.tight_layout()
figure_png = OUTPUT_DIR / "integrated_cdf_validation.png"
fig.savefig(figure_png, dpi=220, bbox_inches="tight")


# %%
# =============================================================================
# Direct visual comparison for representative fitted parameter sets
# =============================================================================
visual_case_reasons = (
    "closest_to_joint_median",
    "omega_minimum",
    "omega_maximum",
)
visual_cases = []
for reason in visual_case_reasons:
    match = representative_df[
        representative_df["selection_reason"].str.contains(reason, regex=False)
    ]
    if len(match) != 1:
        raise RuntimeError(
            f"Expected one representative case for {reason}, found {len(match)}."
        )
    visual_cases.append(match.iloc[0])

reactive_time_s = np.linspace(0.0, 1.0, 50001)
display_max_s = 0.300
delay_width_s = 0.005
delay_low_s = UNIFORM_DELAY_CENTER_S - delay_width_s / 2.0
delay_high_s = UNIFORM_DELAY_CENTER_S + delay_width_s / 2.0
observed_time_s = np.linspace(0.060, 0.300, 2401)

visual_spectral_terms = (100,)
line_styles = {
    100: {"color": "tab:green", "ls": "--", "lw": 2.5, "zorder": 6},
}

fig_visual, axes_visual = plt.subplots(2, 3, figsize=(13.0, 7.0), sharex="row")
for column, case in enumerate(visual_cases):
    gamma = float(case["gamma"])
    omega = float(case["omega"])
    w = float(case["w"])
    bound = -1 if gamma <= 0.0 else 1

    numerical_cdf = current_CDF_E_gamma_omega_with_w_numpy(
        reactive_time_s,
        gamma,
        omega,
        bound,
        w,
    )
    numerical_h = cumulative_trapezoid(
        numerical_cdf,
        reactive_time_s,
        initial=0.0,
    )

    top_ax = axes_visual[0, column]
    top_ax.plot(
        1e3 * reactive_time_s,
        1e3 * numerical_h,
        color="black",
        lw=1.1,
        label="Numerical",
        zorder=3,
    )
    for n_terms in visual_spectral_terms:
        analytic_h = integrated_CDF_E_gamma_omega_with_w_numpy(
            reactive_time_s,
            gamma,
            omega,
            bound,
            w,
            n_terms,
        )
        top_ax.plot(
            1e3 * reactive_time_s,
            1e3 * analytic_h,
            label=f"Analytic K={n_terms}",
            **line_styles[n_terms],
        )
    top_ax.axhline(0.0, color="0.75", lw=0.7, zorder=0)
    top_ax.set_xlim(0.0, 1e3 * display_max_s)
    top_ax.set_title(
        f"{case['batch']}/{int(case['animal'])}, "
        f"ABL={case['ABL']:.0f}, ILD={case['ILD']:+.0f}\n"
        f"Gamma={gamma:.3f}, Omega={omega:.3f}, w={w:.3f}, bound={bound:+d}",
        fontsize=9,
    )
    if column == 0:
        top_ax.set_ylabel("Integrated bound CDF, H(t) (ms)")

    numerical_uniform_cdf = (
        np.interp(
            np.maximum(observed_time_s - delay_low_s, 0.0),
            reactive_time_s,
            numerical_h,
        )
        - np.interp(
            np.maximum(observed_time_s - delay_high_s, 0.0),
            reactive_time_s,
            numerical_h,
        )
    ) / delay_width_s
    numerical_uniform_cdf = np.where(
        observed_time_s > delay_low_s,
        numerical_uniform_cdf,
        0.0,
    )

    bottom_ax = axes_visual[1, column]
    bottom_ax.plot(
        1e3 * observed_time_s,
        numerical_uniform_cdf,
        color="black",
        lw=1.1,
        label="Numerical",
        zorder=3,
    )
    for n_terms in visual_spectral_terms:
        analytic_uniform_cdf = (
            integrated_CDF_E_gamma_omega_with_w_numpy(
                observed_time_s - delay_low_s,
                gamma,
                omega,
                bound,
                w,
                n_terms,
            )
            - integrated_CDF_E_gamma_omega_with_w_numpy(
                observed_time_s - delay_high_s,
                gamma,
                omega,
                bound,
                w,
                n_terms,
            )
        ) / delay_width_s
        bottom_ax.plot(
            1e3 * observed_time_s,
            analytic_uniform_cdf,
            label=f"Analytic K={n_terms}",
            **line_styles[n_terms],
        )
    bottom_ax.axhline(0.0, color="0.75", lw=0.7, zorder=0)
    bottom_ax.axvspan(
        1e3 * delay_low_s,
        1e3 * delay_high_s,
        color="0.85",
        alpha=0.45,
        zorder=0,
    )
    bottom_ax.set_xlim(60.0, 300.0)
    bottom_ax.set_xlabel("Time after stimulus onset (ms)")
    if column == 0:
        bottom_ax.set_ylabel("Uniform-delay bound CDF")

handles, labels = axes_visual[0, 0].get_legend_handles_labels()
fig_visual.legend(
    handles,
    labels,
    loc="upper center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, 1.01),
)
fig_visual.suptitle(
    "Numerical versus finite-series analytic integrated CDF",
    y=0.955,
)
fig_visual.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
visual_figure_png = OUTPUT_DIR / "integrated_cdf_representative_curve_comparison.png"
fig_visual.savefig(visual_figure_png, dpi=220, bbox_inches="tight")

print("\nSaved:")
for output_path in [
    representative_csv,
    convergence_csv,
    uniform_csv,
    limit_csv,
    summary_csv,
    figure_png,
    visual_figure_png,
]:
    print(f"  {output_path}")
