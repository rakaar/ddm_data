
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the directory containing the utils to the path
sys.path.append('/home/rlab/raghavendra/ddm_data/fit_each_condn')

from led_off_gamma_omega_pdf_utils import (
    rho_E_minus_small_t_NORM_omega_gamma_with_w_fn,
    rho_E_minus_small_t_NORM_omega_gamma_with_w_VEC_fn
)

def test_vectorization():
    # Parameters to test
    # omega_plausible_bounds = [2, 12]
    # gamma_plausible_bounds = [-3, 3]
    # w different fractions from 0.3 to 0.7

    param_sets = [
        {'gamma': 0.0, 'omega': 7.0, 'w': 0.5},
        {'gamma': 2.0, 'omega': 3.0, 'w': 0.3},
        {'gamma': -2.0, 'omega': 10.0, 'w': 0.7},
        {'gamma': 1.0, 'omega': 5.0, 'w': 0.4},
        {'gamma': -1.0, 'omega': 8.0, 'w': 0.6}
    ]

    t_pts = np.arange(0, 1.001, 0.001) # 0 to 1 in steps of 1ms
    bound = 1 # Test for bound 1 (upper bound?) or just one of them. The function takes bound as arg.
    # We should probably test both bounds or just one. Let's test bound=1.
    
    K_max = 50 # Standard K_max

    fig, axes = plt.subplots(len(param_sets), 1, figsize=(10, 4 * len(param_sets)))
    if len(param_sets) == 1:
        axes = [axes]

    for i, params in enumerate(param_sets):
        gamma = params['gamma']
        omega = params['omega']
        w = params['w']
        
        print(f"Testing set {i+1}: gamma={gamma}, omega={omega}, w={w}")

        # Scalar computation
        scalar_pdf = []
        for t in t_pts:
            val = rho_E_minus_small_t_NORM_omega_gamma_with_w_fn(t, gamma, omega, bound, w, K_max)
            scalar_pdf.append(val)
        scalar_pdf = np.array(scalar_pdf)

        # Vectorized computation
        vec_pdf = rho_E_minus_small_t_NORM_omega_gamma_with_w_VEC_fn(t_pts, gamma, omega, bound, w, K_max)

        # Compare
        diff = np.abs(scalar_pdf - vec_pdf)
        max_diff = np.max(diff)
        print(f"  Max difference: {max_diff}")

        # Plot
        ax = axes[i]
        ax.plot(t_pts, scalar_pdf, 'b-', linewidth=3, alpha=0.5, label='Scalar')
        ax.plot(t_pts, vec_pdf, 'r--', linewidth=2, label='Vectorized')
        ax.set_title(f'Set {i+1}: gamma={gamma}, omega={omega}, w={w}, bound={bound}\nMax Diff: {max_diff:.2e}')
        ax.legend()
        ax.grid(True)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('PDF')

    plt.tight_layout()
    save_path = '/home/rlab/raghavendra/ddm_data/vectorization_test_comparison.jpg'
    plt.savefig(save_path)
    print(f"Comparison plot saved to {save_path}")

if __name__ == "__main__":
    test_vectorization()
