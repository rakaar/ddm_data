import numpy as np
from fit_animal_by_animal.time_vary_norm_utils import (
    CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn,
    CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn_vec,
    rho_E_minus_small_t_NORM_rate_norm_time_varying_fn,
    rho_E_minus_small_t_NORM_rate_norm_time_varying_fn_vec
)

# Test array of time points
T = np.arange(-1, 4.02, 0.02)  # from -1 to 1 inclusive

# Fixed parameters for testing
bound = 0
ABL = 20.0
ILD = 1.0
rate_lambda = 0.13
T0 = 0.4 * 1e-3
theta_E = 60
Z_E = 0.0
rate_norm_l = 0
is_norm = False
is_time_vary = False
K_max = 10
phi_params_obj = np.nan
phi_t = np.nan
int_phi_t = np.nan

# --- Test CDF ---
cdf_loop = np.array([
    CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn(
        t, bound, ABL, ILD, rate_lambda, T0, theta_E, Z_E, int_phi_t, rate_norm_l, is_norm, is_time_vary, K_max
    ) for t in T
])
cdf_vec = CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn_vec(
    T, bound, ABL, ILD, rate_lambda, T0, theta_E, Z_E, int_phi_t, rate_norm_l, is_norm, is_time_vary, K_max
)


# --- Test PDF ---
pdf_loop = np.array([
    rho_E_minus_small_t_NORM_rate_norm_time_varying_fn(
        t, bound, ABL, ILD, rate_lambda, T0, theta_E, Z_E, phi_t, int_phi_t, rate_norm_l, is_norm, is_time_vary, K_max
    ) for t in T
])
pdf_vec = rho_E_minus_small_t_NORM_rate_norm_time_varying_fn_vec(
    T, bound, ABL, ILD, rate_lambda, T0, theta_E, Z_E, phi_t, int_phi_t, rate_norm_l, is_norm, is_time_vary, K_max
)


# Optionally, plot for visual inspection
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(T, cdf_loop, label='Loop')
    plt.plot(T, cdf_vec, '--', label='Vectorized')
    plt.title('CDF: Loop vs Vectorized')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(T, pdf_loop, label='Loop')
    plt.plot(T, pdf_vec, '--', label='Vectorized')
    plt.title('PDF: Loop vs Vectorized')
    plt.legend()
    plt.tight_layout()
    plt.show()
except ImportError:
    pass
