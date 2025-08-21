import os
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Local imports from this repo
from animal_wise_plotting_utils import calculate_theoretical_curves
from time_vary_norm_utils import (
    rho_A_t_fn,
    cum_pro_and_reactive_time_vary_fn,
    up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn,
)


ILD_ARR = np.array([-16., -8., -4., -2., -1., 1., 2., 4., 8., 16.])


def load_abort_params(results_pkl_path: str) -> dict:
    """Load abort parameter means from results_{batch}_animal_{id}.pkl.

    Expected keys in 'vbmc_aborts_results': 'V_A_samples', 'theta_A_samples', 't_A_aff_samp'.
    """
    with open(results_pkl_path, 'rb') as f:
        data = pickle.load(f)
    key = 'vbmc_aborts_results'
    if key not in data:
        raise KeyError(f"'{key}' not found in {results_pkl_path}")
    abort = data[key]
    V_A = float(np.mean(np.asarray(abort['V_A_samples'])))
    theta_A = float(np.mean(np.asarray(abort['theta_A_samples'])))
    t_A_aff = float(np.mean(np.asarray(abort['t_A_aff_samp'])))
    return {'V_A': V_A, 'theta_A': theta_A, 't_A_aff': t_A_aff}


def compute_RTD_up_down(P_A_mean, C_A_mean, t_stim_samples, abort_params, tied_params,
                        batch_name: str, ABL: float, ILD: float):
    """Replicates logic from get_theoretical_RTD_up_down without importing heavy modules.

    Returns (t_pts_0_1, up_theory_mean_norm, down_theory_mean_norm)
    """
    # Model switches consistent with vanilla tied usage in existing code
    phi_params_obj = np.nan
    rate_norm_l = 0.0
    is_norm = False
    is_time_vary = False
    K_max = 10

    # Truncation time rule per batch
    T_trunc = 0.15 if batch_name == 'LED34_even' else 0.3

    t_pts = np.arange(-2, 2, 0.001)

    # Truncation factor uses cumulative probability difference including reactive component
    trunc_fac_samples = np.zeros((len(t_stim_samples)))
    Z_E = (tied_params['w'] - 0.5) * 2.0 * tied_params['theta_E']
    for idx, t_stim in enumerate(t_stim_samples):
        trunc_fac_samples[idx] = (
            cum_pro_and_reactive_time_vary_fn(
                t_stim + 1, T_trunc,
                abort_params['V_A'], abort_params['theta_A'], abort_params['t_A_aff'],
                t_stim, ABL, ILD,
                tied_params['rate_lambda'], tied_params['T_0'], tied_params['theta_E'], Z_E, tied_params['t_E_aff'],
                phi_params_obj, rate_norm_l, is_norm, is_time_vary, K_max
            )
            -
            cum_pro_and_reactive_time_vary_fn(
                t_stim, T_trunc,
                abort_params['V_A'], abort_params['theta_A'], abort_params['t_A_aff'],
                t_stim, ABL, ILD,
                tied_params['rate_lambda'], tied_params['T_0'], tied_params['theta_E'], Z_E, tied_params['t_E_aff'],
                phi_params_obj, rate_norm_l, is_norm, is_time_vary, K_max
            )
            + 1e-10
        )
    trunc_factor = float(np.mean(trunc_fac_samples))

    up_mean = np.array([
        up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn(
            t, 1,
            P_A_mean[i], C_A_mean[i],
            ABL, ILD,
            tied_params['rate_lambda'], tied_params['T_0'], tied_params['theta_E'], Z_E, tied_params['t_E_aff'], tied_params['del_go'],
            phi_params_obj, rate_norm_l, is_norm, is_time_vary, K_max
        )
        for i, t in enumerate(t_pts)
    ])
    down_mean = np.array([
        up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn(
            t, -1,
            P_A_mean[i], C_A_mean[i],
            ABL, ILD,
            tied_params['rate_lambda'], tied_params['T_0'], tied_params['theta_E'], Z_E, tied_params['t_E_aff'], tied_params['del_go'],
            phi_params_obj, rate_norm_l, is_norm, is_time_vary, K_max
        )
        for i, t in enumerate(t_pts)
    ])

    mask_0_1 = (t_pts >= 0) & (t_pts <= 1)
    t_pts_0_1 = t_pts[mask_0_1]
    up_theory_mean_norm = up_mean[mask_0_1] / trunc_factor
    down_theory_mean_norm = down_mean[mask_0_1] / trunc_factor
    return t_pts_0_1, up_theory_mean_norm, down_theory_mean_norm


def build_PA_CA_from_data(batch_csv_path: str, batch_name: str, animal_id: int, abort_params: dict,
                          N_theory: int = 1000):
    """Compute P_A_mean, C_A_mean, and t_stim_samples using dataset and abort params."""
    df = pd.read_csv(batch_csv_path)
    # Keep that animal's trials; include both valid and abort trials for t_stim distribution as in reference
    df_animal = df[df['animal'] == animal_id].copy()
    t_pts = np.arange(-2, 2, 0.001)
    P_A_mean, C_A_mean, t_stim_samples = calculate_theoretical_curves(
        df_animal, N_theory, t_pts, abort_params['t_A_aff'], abort_params['V_A'], abort_params['theta_A'], rho_A_t_fn
    )
    return P_A_mean, C_A_mean, t_stim_samples


def run_T0_sweep(batch_name: str, animal_id: int, ABL: float,
                 rate_lambda: float, theta_E: float, w: float, t_E_aff: float, del_go: float,
                 T0_values: np.ndarray, base_dir: str):
    """Compute psychometric curves for each T0 in T0_values and plot them together."""
    # Resolve paths
    batch_csv_path = os.path.join(base_dir, 'batch_csvs', f'batch_{batch_name}_valid_and_aborts.csv')
    results_pkl_path = os.path.join(base_dir, f'results_{batch_name}_animal_{animal_id}.pkl')

    if not os.path.exists(batch_csv_path):
        raise FileNotFoundError(f"CSV not found: {batch_csv_path}")
    if not os.path.exists(results_pkl_path):
        raise FileNotFoundError(f"Results PKL not found: {results_pkl_path}")

    # Abort parameters and P_A/C_A (independent of T0)
    abort_params = load_abort_params(results_pkl_path)
    P_A_mean, C_A_mean, t_stim_samples = build_PA_CA_from_data(
        batch_csv_path, batch_name, animal_id, abort_params, N_theory=1000
    )

    # Fixed tied parameters that do not change across T0 sweep
    tied_fixed = {
        'rate_lambda': float(rate_lambda),
        'theta_E': float(theta_E),
        'w': float(w),
        't_E_aff': float(t_E_aff),
        'del_go': float(del_go),
        # 'T_0' will be set inside the loop
    }

    # Compute psychometric right-choice prob for each T0
    ilds = ILD_ARR.copy()
    curves = {}
    from scipy.integrate import trapezoid

    for T0 in T0_values:
        tied_params = dict(tied_fixed)
        tied_params['T_0'] = float(T0)
        right_choice_probs = []
        for ild in ilds:
            t_pts_0_1, up_mean, down_mean = compute_RTD_up_down(
                P_A_mean, C_A_mean, t_stim_samples, abort_params, tied_params,
                batch_name=batch_name, ABL=ABL, ILD=float(ild)
            )
            up_area = trapezoid(up_mean, t_pts_0_1)
            down_area = trapezoid(down_mean, t_pts_0_1)
            prob_right = up_area / (up_area + down_area)
            right_choice_probs.append(prob_right)
        curves[np.round(T0 * 1000, 3)] = np.array(right_choice_probs)  # store with ms key
        print(f"Computed curve for T0={T0*1000:.3f} ms")

    return ilds, curves


def plot_curves(ilds: np.ndarray, curves: dict, batch_name: str, animal_id: int, ABL: float, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(7, 5))
    # colormap across sorted T0 in ms
    keys_ms = sorted(curves.keys())
    cmap = plt.get_cmap('viridis')
    for idx, k in enumerate(keys_ms):
        color = cmap(float(idx) / max(len(keys_ms) - 1, 1))
        plt.plot(ilds, curves[k], marker='o', linestyle='-', color=color, label=f'T0={k:.1f} ms', alpha=0.9)
    plt.xlabel('ILD (dB)')
    plt.ylabel('P(choice = right)')
    plt.title(f'Psychometric invariance vs T0 (ABL={ABL}, {batch_name}, animal {animal_id})')
    plt.ylim(-0.02, 1.02)
    plt.axhline(0.5, color='grey', alpha=0.5, linestyle='--')
    plt.axvline(0, color='grey', alpha=0.5, linestyle='--')
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f'T0_invariance_psychometric_{batch_name}_animal_{animal_id}_ABL_{int(ABL)}.png')
    plt.savefig(out_path, dpi=200)
    print(f"Saved figure: {out_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot psychometric curves for multiple T0 values to demonstrate invariance.')
    parser.add_argument('--batch', type=str, default='LED8', help='Batch name (e.g., LED8, LED7, SD, LED34_even)')
    parser.add_argument('--animal', type=int, default=108, help='Animal ID present in results_{batch}_animal_{id}.pkl')
    parser.add_argument('--abl', type=float, default=40.0, help='ABL to use for the psychometric curve')

    # Fixed parameters per user request
    parser.add_argument('--rate_lambda', type=float, default=0.2)
    parser.add_argument('--theta_E', type=float, default=20.0)
    parser.add_argument('--w', type=float, default=0.5)
    parser.add_argument('--t_E_aff', type=float, default=0.65)
    parser.add_argument('--del_go', type=float, default=0.13)

    parser.add_argument('--t0_min_ms', type=float, default=0.1, help='Min T0 in ms')
    parser.add_argument('--t0_max_ms', type=float, default=0.5, help='Max T0 in ms')
    parser.add_argument('--t0_step_ms', type=float, default=0.1, help='Step T0 in ms')

    args = parser.parse_args()

    base_dir = os.path.dirname(__file__)

    # Build T0 sweep in seconds
    T0_values_ms = np.arange(args.t0_min_ms, args.t0_max_ms + 1e-12, args.t0_step_ms)
    # Ensure the upper bound is included even if step doesn't land exactly on it
    if T0_values_ms.size == 0 or T0_values_ms[-1] < args.t0_max_ms - 1e-9:
        T0_values_ms = np.append(T0_values_ms, args.t0_max_ms)
    T0_values = T0_values_ms * 1e-3

    ilds, curves = run_T0_sweep(
        batch_name=args.batch,
        animal_id=args.animal,
        ABL=args.abl,
        rate_lambda=args.rate_lambda,
        theta_E=args.theta_E,
        w=args.w,
        t_E_aff=args.t_E_aff,
        del_go=args.del_go,
        T0_values=T0_values,
        base_dir=base_dir,
    )

    plot_dir = os.path.join(base_dir, 'plots')
    plot_curves(ilds, curves, args.batch, args.animal, args.abl, plot_dir)


if __name__ == '__main__':
    main()
