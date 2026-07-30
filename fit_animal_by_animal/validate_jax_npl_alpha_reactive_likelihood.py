# %%
"""
Validate the retained-window reactive-only NPL+alpha JAX likelihood.

Checks:
1. JAX density, retained-window mass, and log likelihood match an independent
   NumPy implementation on real LED7/92 trials.
2. The choice-collapsed normalized density integrates to one in [0.1, 1] s.
3. The log likelihood and its gradients are finite at the current SVI fit.
"""

# %%
from pathlib import Path
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.special import erfcx


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import numpyro_npl_alpha_reactive_svi_utils as reactive_utils


# %%
# =============================================================================
# Editable validation settings
# =============================================================================
BATCH_NAME = "LED7"
ANIMAL = 92
RT_LOWER = 0.100
RT_UPPER = 1.000
K_MAX = 10
N_REAL_TRIALS = 96
INTEGRATION_POINTS = 20001

BATCH_CSV = REPO_DIR / "raw_data" / "batch_csvs" / f"batch_{BATCH_NAME}_valid_and_aborts.csv"
REFERENCE_ROOT = (
    SCRIPT_DIR
    / "numpyro_svi_npl_alpha_condition_delay_patience12_restore_best_outputs"
)
REFERENCE_DIR = REFERENCE_ROOT / f"{BATCH_NAME}_{ANIMAL}"
REFERENCE_NPZ = REFERENCE_DIR / "main_fullrank_posterior_samples.npz"
REFERENCE_CONDITIONS = REFERENCE_DIR / "condition_table.csv"


# %%
# =============================================================================
# Independent NumPy reactive density/CDF
# =============================================================================
def gamma_omega_alpha_numpy(ABL, ILD, rate_lambda, T_0, theta_E, rate_norm_l, alpha):
    chi = 17.37
    ABL = np.asarray(ABL, dtype=float)
    ILD = np.asarray(ILD, dtype=float)
    abl_term = 10.0 ** (rate_lambda * (1.0 - rate_norm_l) * ABL / 20.0)
    ild_arg = rate_lambda * ILD / chi
    norm_ild_arg = rate_lambda * rate_norm_l * ILD / chi
    r_r = abl_term * np.exp(ild_arg) / (
        np.exp(norm_ild_arg) + alpha * np.exp(-norm_ild_arg)
    )
    r_l = abl_term * np.exp(-ild_arg) / (
        np.exp(-norm_ild_arg) + alpha * np.exp(norm_ild_arg)
    )
    r_sum = r_r + r_l
    return theta_E * (r_r - r_l) / r_sum, r_sum / (T_0 * theta_E**2)


def rho_E_alpha_numpy(
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
    K_max,
):
    t_original = np.asarray(t, dtype=float)
    bound = np.asarray(bound)
    v, omega = gamma_omega_alpha_numpy(
        ABL,
        ILD,
        rate_lambda,
        T_0,
        theta_E,
        rate_norm_l,
        alpha,
    )
    w = 0.5 + Z_E / (2.0 * theta_E)
    a = 2.0
    v = np.where(bound == 1, -v, v)
    w = np.where(bound == 1, 1.0 - w, w)

    t_eff = omega * t_original
    shape = np.broadcast(t_eff, v, w).shape
    valid = np.broadcast_to(t_original, shape) > 0
    safe_t = np.where(valid, np.broadcast_to(t_eff, shape), 1e-12)
    v_full = np.broadcast_to(v, shape)
    w_full = np.broadcast_to(w, shape)
    non_sum_term = (
        (1.0 / a**2)
        * (a**3 / np.sqrt(2.0 * np.pi * safe_t**3))
        * np.exp(-v_full * a * w_full - (v_full**2) * safe_t / 2.0)
    )

    K_half = int(K_max / 2)
    k_vals = np.linspace(-K_half, K_half, 2 * K_half + 1)
    k_b = k_vals.reshape((1,) * len(shape) + (2 * K_half + 1,))
    t_b = safe_t[..., None]
    w_b = w_full[..., None]
    sum_result = np.sum(
        (w_b + 2.0 * k_b)
        * np.exp(-(a**2 * (w_b + 2.0 * k_b) ** 2) / (2.0 * t_b)),
        axis=-1,
    )
    density = non_sum_term * sum_result
    density = np.where(density <= 0, 1e-16, density)
    return np.where(valid, density * np.broadcast_to(omega, shape), 0.0)


def CDF_E_alpha_numpy(
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
    K_max,
):
    t_original = np.asarray(t, dtype=float)
    bound = np.asarray(bound)
    v, omega = gamma_omega_alpha_numpy(
        ABL,
        ILD,
        rate_lambda,
        T_0,
        theta_E,
        rate_norm_l,
        alpha,
    )
    w = 0.5 + Z_E / (2.0 * theta_E)
    a = 2.0
    v = np.where(bound == 1, -v, v)
    w = np.where(bound == 1, 1.0 - w, w)

    t_eff = omega * t_original
    shape = np.broadcast(t_eff, v, w).shape
    valid = np.broadcast_to(t_original, shape) > 0
    safe_t = np.where(valid, np.broadcast_to(t_eff, shape), 1e-12)
    v_full = np.broadcast_to(v, shape)
    w_full = np.broadcast_to(w, shape)
    result = np.exp(-v_full * a * w_full - (v_full**2) * safe_t / 2.0)

    k_arr = np.arange(K_max + 1)
    k_b = k_arr.reshape((1,) * len(shape) + (K_max + 1,))
    t_b = safe_t[..., None]
    v_b = v_full[..., None]
    w_b = w_full[..., None]
    r_k = np.where(
        k_b % 2 == 0,
        k_b * a + a * w_b,
        k_b * a + a * (1.0 - w_b),
    )
    sqrt_t = np.sqrt(t_b)
    normal_pdf = np.exp(-0.5 * (r_k / sqrt_t) ** 2) / np.sqrt(2.0 * np.pi)
    left_mills = np.sqrt(np.pi / 2.0) * erfcx(
        np.clip((r_k - v_b * t_b) / sqrt_t, -37.0, 37.0) / np.sqrt(2.0)
    )
    right_mills = np.sqrt(np.pi / 2.0) * erfcx(
        np.clip((r_k + v_b * t_b) / sqrt_t, -37.0, 37.0) / np.sqrt(2.0)
    )
    summation = np.sum(
        ((-1.0) ** k_b) * normal_pdf * (left_mills + right_mills),
        axis=-1,
    )
    return np.where(valid, result * summation, 0.0)


# %%
# =============================================================================
# Load representative real data and current parameter means
# =============================================================================
for required_path in [BATCH_CSV, REFERENCE_NPZ, REFERENCE_CONDITIONS]:
    if not required_path.exists():
        raise FileNotFoundError(required_path)

raw_df = pd.read_csv(BATCH_CSV)
if "choice" not in raw_df.columns:
    raw_df["choice"] = raw_df["response_poke"].map({3: 1, 2: -1})
valid_df = raw_df[
    (raw_df["animal"].astype(int) == ANIMAL)
    & raw_df["success"].isin([1, -1])
    & (raw_df["RTwrtStim"] >= RT_LOWER)
    & (raw_df["RTwrtStim"] < RT_UPPER)
    & raw_df["ABL"].isin([20, 40, 60])
].dropna(subset=["RTwrtStim", "ABL", "ILD", "choice"]).copy()
valid_df = valid_df.sample(
    min(N_REAL_TRIALS, len(valid_df)),
    random_state=7,
).sort_index()

condition_table = pd.read_csv(REFERENCE_CONDITIONS).sort_values("condition_id")
condition_lookup = {
    (float(row.ABL), float(row.ILD)): int(row.condition_id)
    for row in condition_table.itertuples()
}
valid_df["condition_id"] = [
    condition_lookup[(float(abl), float(ild))]
    for abl, ild in zip(valid_df["ABL"], valid_df["ILD"])
]

with np.load(REFERENCE_NPZ) as saved:
    params = {
        name: float(np.mean(saved[name]))
        for name in reactive_utils.GLOBAL_PARAM_NAMES
    }
    params["t_E_aff"] = np.mean(saved["t_E_aff"], axis=0)

data = {
    "rt_wrt_stim": jnp.asarray(valid_df["RTwrtStim"].to_numpy(dtype=float)),
    "ABL": jnp.asarray(valid_df["ABL"].to_numpy(dtype=float)),
    "ILD": jnp.asarray(valid_df["ILD"].to_numpy(dtype=float)),
    "choice": jnp.asarray(valid_df["choice"].to_numpy(dtype=int)),
    "condition_id": jnp.asarray(valid_df["condition_id"].to_numpy(dtype=int)),
    "rt_lower": jnp.asarray(RT_LOWER, dtype=jnp.float64),
    "rt_upper": jnp.asarray(RT_UPPER, dtype=jnp.float64),
}
params_jax = {
    key: jnp.asarray(value, dtype=jnp.float64)
    for key, value in params.items()
}


# %%
# =============================================================================
# JAX vs NumPy checks
# =============================================================================
jax_terms = reactive_utils.npl_alpha_reactive_condition_delay_loglike_terms(
    params_jax,
    data,
    K_max=K_MAX,
)
jax_terms = {key: np.asarray(value) for key, value in jax_terms.items()}

condition_ids = valid_df["condition_id"].to_numpy(dtype=int)
t_E_aff = params["t_E_aff"][condition_ids]
ABL = valid_df["ABL"].to_numpy(dtype=float)
ILD = valid_df["ILD"].to_numpy(dtype=float)
choice = valid_df["choice"].to_numpy(dtype=int)
Z_E = (params["w"] - 0.5) * 2.0 * params["theta_E"]

numpy_pdf = rho_E_alpha_numpy(
    valid_df["RTwrtStim"].to_numpy(dtype=float) - t_E_aff,
    choice,
    ABL,
    ILD,
    params["rate_lambda"],
    params["T_0"],
    params["theta_E"],
    Z_E,
    params["rate_norm_l"],
    params["alpha"],
    K_MAX,
)
numpy_lower_mass = sum(
    CDF_E_alpha_numpy(
        RT_LOWER - t_E_aff,
        bound,
        ABL,
        ILD,
        params["rate_lambda"],
        params["T_0"],
        params["theta_E"],
        Z_E,
        params["rate_norm_l"],
        params["alpha"],
        K_MAX,
    )
    for bound in (-1, 1)
)
numpy_upper_mass = sum(
    CDF_E_alpha_numpy(
        RT_UPPER - t_E_aff,
        bound,
        ABL,
        ILD,
        params["rate_lambda"],
        params["T_0"],
        params["theta_E"],
        Z_E,
        params["rate_norm_l"],
        params["alpha"],
        K_MAX,
    )
    for bound in (-1, 1)
)
numpy_window_mass = np.maximum(numpy_upper_mass - numpy_lower_mass, 1e-20)
numpy_normalized_pdf = np.maximum(numpy_pdf / numpy_window_mass, 1e-50)
numpy_loglike = np.log(numpy_normalized_pdf)

np.testing.assert_allclose(jax_terms["pdf"], numpy_pdf, rtol=2e-10, atol=1e-12)
np.testing.assert_allclose(
    jax_terms["retained_window_mass"],
    numpy_window_mass,
    rtol=2e-10,
    atol=1e-12,
)
np.testing.assert_allclose(
    jax_terms["normalized_pdf"],
    numpy_normalized_pdf,
    rtol=2e-10,
    atol=1e-12,
)
np.testing.assert_allclose(jax_terms["loglike"], numpy_loglike, rtol=2e-10, atol=1e-12)

jax_loglike = reactive_utils.npl_alpha_reactive_condition_delay_loglike(
    params_jax,
    data,
    K_max=K_MAX,
)
np.testing.assert_allclose(
    float(jax_loglike),
    float(np.sum(numpy_loglike)),
    rtol=2e-10,
    atol=1e-10,
)


# %%
# =============================================================================
# Retained-window mass and finite-gradient checks
# =============================================================================
integration_rows = []
for row in condition_table.iloc[[0, len(condition_table) // 2, len(condition_table) - 1]].itertuples():
    condition_id = int(row.condition_id)
    grid = np.linspace(RT_LOWER, RT_UPPER, INTEGRATION_POINTS)
    delay = float(params["t_E_aff"][condition_id])
    density = sum(
        rho_E_alpha_numpy(
            grid - delay,
            bound,
            float(row.ABL),
            float(row.ILD),
            params["rate_lambda"],
            params["T_0"],
            params["theta_E"],
            Z_E,
            params["rate_norm_l"],
            params["alpha"],
            K_MAX,
        )
        for bound in (-1, 1)
    )
    cdf_mass = sum(
        CDF_E_alpha_numpy(
            RT_UPPER - delay,
            bound,
            float(row.ABL),
            float(row.ILD),
            params["rate_lambda"],
            params["T_0"],
            params["theta_E"],
            Z_E,
            params["rate_norm_l"],
            params["alpha"],
            K_MAX,
        )
        - CDF_E_alpha_numpy(
            RT_LOWER - delay,
            bound,
            float(row.ABL),
            float(row.ILD),
            params["rate_lambda"],
            params["T_0"],
            params["theta_E"],
            Z_E,
            params["rate_norm_l"],
            params["alpha"],
            K_MAX,
        )
        for bound in (-1, 1)
    )
    normalized_integral = float(np.trapz(density / cdf_mass, grid))
    integration_rows.append(
        {
            "ABL": float(row.ABL),
            "ILD": float(row.ILD),
            "cdf_window_mass": float(cdf_mass),
            "normalized_integral": normalized_integral,
        }
    )
    if not np.isclose(normalized_integral, 1.0, rtol=0, atol=2e-4):
        raise AssertionError(
            f"Normalized density integral was {normalized_integral:.8f} "
            f"for ABL={row.ABL:g}, ILD={row.ILD:g}."
        )

gradient = jax.grad(
    lambda values: reactive_utils.npl_alpha_reactive_condition_delay_loglike(
        values,
        data,
        K_max=K_MAX,
    )
)(params_jax)
if not reactive_utils.tree_all_finite(gradient):
    raise AssertionError("Reactive likelihood gradient contains NaN or Inf.")

print(f"Validated {len(valid_df)} real {BATCH_NAME}/{ANIMAL} trials.")
print(f"JAX log likelihood: {float(jax_loglike):.9f}")
print("Normalized two-choice density integration checks:")
print(pd.DataFrame(integration_rows).to_string(index=False))
print("All JAX/NumPy term comparisons and gradient checks passed.")
