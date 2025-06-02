import numpy as np
from fit_animal_by_animal.time_vary_norm_utils import (
    up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn,
    up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn_vec
)

# Test array of time points
T = np.arange(-1, 4.02, 0.02)

# Fixed parameters for testing (same as previous test)
bound = 0
P_A = 0.7
C_A = 0.2
ABL = 20.0
ILD = 1.0
rate_lambda = 0.13
T0 = 0.4 * 1e-3
theta_E = 60
Z_E = 0.0
t_E_aff = 0.05
del_go = 0.1
rate_norm_l = 0
is_norm = False
is_time_vary = False
K_max = 10

# For non-time-varying: phi_params is not used, so pass None
def dummy_phi_params():
    class Dummy:
        h1 = a1 = b1 = h2 = a2 = 0
    return Dummy()
phi_params = dummy_phi_params()

# --- Test original (loop) version ---
loop_results = np.array([
    up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn(
        t, bound, P_A, C_A, ABL, ILD, rate_lambda, T0, theta_E, Z_E,
        t_E_aff, del_go, phi_params, rate_norm_l, is_norm, is_time_vary, K_max
    ) for t in T
])

# --- Test vectorized version ---
vec_results = up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn_vec(
    T, bound, P_A, C_A, ABL, ILD, rate_lambda, T0, theta_E, Z_E,
    t_E_aff, del_go,
    np.nan, np.nan, np.nan, np.nan, np.nan,  # int_phi_t_E_g, phi_t_e, int_phi_t_e, int_phi_t2, int_phi_t1
    rate_norm_l, is_norm, is_time_vary, K_max
)

print("up_or_down: max abs diff:", np.max(np.abs(loop_results - vec_results)))
print("up_or_down: max rel diff:", np.max(np.abs((loop_results - vec_results)/(np.maximum(np.abs(loop_results), 1e-10)))))

# Optionally, plot for visual inspection
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,4))
    plt.plot(T, loop_results, label='Loop')
    plt.plot(T, vec_results, '--', label='Vectorized')
    plt.title('up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn: Loop vs Vectorized')
    plt.legend()
    plt.show()
except ImportError:
    pass
