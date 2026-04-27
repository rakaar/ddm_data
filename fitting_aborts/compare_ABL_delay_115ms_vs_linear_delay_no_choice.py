# %%
"""
Compare LED-OFF aggregate RTDs by ABL for:
1) ABL-specific delay, no-choice fit truncated at 115 ms (allvalid)
2) Linear-delay, no-choice fit truncated at 1000 ms

This script rebuilds the LED-OFF diagnostics dataset, reads both saved fit-result pickles,
reconstructs theory curves from posterior-mean parameters, and overlays:
    data + ABL-delay 115ms model + linear-delay model

The densities are built on the ordinary post-stim window [0, 1.0] s with no re-normalization
to 115 ms. The figure is then displayed/saved with xlim = 115 ms.
"""

# %%
from pathlib import Path
import os
import pickle
import sys

SHOW_PLOT = True
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ANIMAL_WISE_DIR = REPO_ROOT / "fit_animal_by_animal"
if str(ANIMAL_WISE_DIR) not in sys.path:
    sys.path.insert(0, str(ANIMAL_WISE_DIR))

from linear_delay_no_choice_diagnostics_utils import (
    _cum_A_t_vec,
    _lapse_cdf_vec,
    _lapse_pdf_vec,
    build_theory_curve_for_trial as build_linear_delay_theory_curve_for_trial,
    build_truncated_density_payload,
    format_counts,
    validate_supported_values,
)
from time_vary_norm_utils import (
    CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn_vec,
    rho_A_t_VEC_fn,
    rho_E_minus_small_t_NORM_rate_norm_time_varying_fn_vec,
)


# %%
############ Parameters (edit here) ############
batch_name = "LED7"
session_type = 7
training_level = 16
allowed_repeat_trials = [0, 2]
max_rtwrtstim_for_fit = 1.0
data_bin_size_s_poststim = 5e-3

seed = 12345
N_mc_per_abl_panel = int(os.getenv("FIT_DIAG_N_MC_PER_ABL", "1000"))
n_jobs = int(os.getenv("FIT_DIAG_N_JOBS", "30"))
show_plot = SHOW_PLOT

t_pts = np.arange(-2.0, 2.001, 0.001)
supported_abl_values = (20, 40, 60)
comparison_xlim_s = 0.115
comparison_figsize = (15.0, 4.8)
comparison_plot_dpi = 300

abl_colors = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}
abl_delay_model_color = "crimson"
linear_delay_model_color = "black"

is_norm = True
is_time_vary = False
K_max = 10

if N_mc_per_abl_panel <= 0:
    raise ValueError("N_mc_per_abl_panel must be positive.")
if comparison_xlim_s <= 0:
    raise ValueError("comparison_xlim_s must be positive.")
if comparison_xlim_s > max_rtwrtstim_for_fit:
    raise ValueError("comparison_xlim_s cannot exceed max_rtwrtstim_for_fit.")

led_data_csv_path = REPO_ROOT / "out_LED.csv"
abl_delay_results_pkl_path = (
    SCRIPT_DIR
    / "norm_only_led_off_from_loaded_proactive_truncate_NOT_censor_ABL_delay_no_choice"
    / (
        "results_norm_tied_batch_LED7_aggregate_ledoff_1_"
        "proactive_loaded_truncate_NOT_censor_ABL_delay_no_choice_trunc115ms_allvalid.pkl"
    )
)
linear_delay_results_pkl_path = (
    SCRIPT_DIR
    / "norm_only_led_off_from_loaded_proactive_truncate_NOT_censor_linear_delay_no_choice"
    / (
        "results_norm_tied_batch_LED7_aggregate_ledoff_1_"
        "proactive_loaded_truncate_NOT_censor_linear_delay_no_choice_trunc1000ms.pkl"
    )
)

output_dir = SCRIPT_DIR / "model_comparisons"
output_dir.mkdir(parents=True, exist_ok=True)
output_base = output_dir / "compare_ABL_delay_115ms_allvalid_vs_linear_delay_no_choice_by_ABL"
output_pdf_path = output_base.with_suffix(".pdf")
output_png_path = output_base.with_suffix(".png")


# %%
############ Helpers ############
def get_t_E_aff_from_abl(abl, t_E_aff_20, t_E_aff_40, t_E_aff_60):
    abl_value = float(abl)
    if np.isclose(abl_value, 20.0):
        return float(t_E_aff_20)
    if np.isclose(abl_value, 40.0):
        return float(t_E_aff_40)
    if np.isclose(abl_value, 60.0):
        return float(t_E_aff_60)
    raise ValueError(
        f"Unsupported ABL value {abl_value}. Expected one of {supported_abl_values}."
    )


def build_abl_delay_theory_curve_for_trial(
    t_pts,
    t_stim,
    abl,
    ild,
    rate_lambda,
    T_0,
    theta_E,
    Z_E,
    t_E_aff_20,
    t_E_aff_40,
    t_E_aff_60,
    del_go,
    rate_norm_l,
    V_A,
    theta_A,
    t_A_aff,
    lapse_prob,
    beta_lapse,
    is_norm,
    is_time_vary,
    K_max,
):
    t_E_aff = get_t_E_aff_from_abl(
        abl=abl,
        t_E_aff_20=t_E_aff_20,
        t_E_aff_40=t_E_aff_40,
        t_E_aff_60=t_E_aff_60,
    )
    t_abs = t_pts + t_stim
    curve = np.zeros_like(t_pts, dtype=np.float64)

    valid = t_abs > 0
    if not np.any(valid):
        return curve

    t_v = t_abs[valid]
    n = len(t_v)

    P_A = rho_A_t_VEC_fn(t_v - t_A_aff, V_A, theta_A)
    C_A = _cum_A_t_vec(t_v - t_A_aff, V_A, theta_A)

    t1 = np.maximum(t_v - t_stim - t_E_aff, 1e-6)
    t2 = np.maximum(t_v - t_stim - t_E_aff + del_go, 1e-6)

    ABL_arr = np.full(n, float(abl))
    ILD_arr = np.full(n, float(ild))
    rl_arr = np.full(n, rate_lambda)
    T0_arr = np.full(n, T_0)
    thE_arr = np.full(n, theta_E)
    ZE_arr = np.full(n, Z_E)
    rnl_arr = np.full(n, rate_norm_l)

    int_phi_t_E_g = t_v - t_stim - t_E_aff + del_go
    int_phi_t2 = t2
    int_phi_t1 = t1
    int_phi_t_e = t1.copy()
    phi_t_e = np.ones(n)

    t_cdf_arg = t_v - t_stim - t_E_aff + del_go
    CDF_up = CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn_vec(
        t_cdf_arg,
        1,
        ABL_arr,
        ILD_arr,
        rl_arr,
        T0_arr,
        thE_arr,
        ZE_arr,
        int_phi_t_E_g,
        rnl_arr,
        is_norm,
        is_time_vary,
        K_max,
    )
    CDF_down = CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn_vec(
        t_cdf_arg,
        -1,
        ABL_arr,
        ILD_arr,
        rl_arr,
        T0_arr,
        thE_arr,
        ZE_arr,
        int_phi_t_E_g,
        rnl_arr,
        is_norm,
        is_time_vary,
        K_max,
    )
    random_readout_if_EA_survives = 0.5 * (1.0 - (CDF_up + CDF_down))

    CDF_t2_up = CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn_vec(
        t2,
        1,
        ABL_arr,
        ILD_arr,
        rl_arr,
        T0_arr,
        thE_arr,
        ZE_arr,
        int_phi_t2,
        rnl_arr,
        is_norm,
        is_time_vary,
        K_max,
    )
    CDF_t1_up = CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn_vec(
        t1,
        1,
        ABL_arr,
        ILD_arr,
        rl_arr,
        T0_arr,
        thE_arr,
        ZE_arr,
        int_phi_t1,
        rnl_arr,
        is_norm,
        is_time_vary,
        K_max,
    )
    CDF_t2_down = CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn_vec(
        t2,
        -1,
        ABL_arr,
        ILD_arr,
        rl_arr,
        T0_arr,
        thE_arr,
        ZE_arr,
        int_phi_t2,
        rnl_arr,
        is_norm,
        is_time_vary,
        K_max,
    )
    CDF_t1_down = CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn_vec(
        t1,
        -1,
        ABL_arr,
        ILD_arr,
        rl_arr,
        T0_arr,
        thE_arr,
        ZE_arr,
        int_phi_t1,
        rnl_arr,
        is_norm,
        is_time_vary,
        K_max,
    )
    P_E_plus_cum_up = CDF_t2_up - CDF_t1_up
    P_E_plus_cum_down = CDF_t2_down - CDF_t1_down

    t_rho_arg = t_v - t_stim - t_E_aff
    P_E_plus_up = rho_E_minus_small_t_NORM_rate_norm_time_varying_fn_vec(
        t_rho_arg,
        1,
        ABL_arr,
        ILD_arr,
        rl_arr,
        T0_arr,
        thE_arr,
        ZE_arr,
        phi_t_e,
        int_phi_t_e,
        rnl_arr,
        is_norm,
        is_time_vary,
        K_max,
    )
    P_E_plus_down = rho_E_minus_small_t_NORM_rate_norm_time_varying_fn_vec(
        t_rho_arg,
        -1,
        ABL_arr,
        ILD_arr,
        rl_arr,
        T0_arr,
        thE_arr,
        ZE_arr,
        phi_t_e,
        int_phi_t_e,
        rnl_arr,
        is_norm,
        is_time_vary,
        K_max,
    )

    lp = float(np.clip(lapse_prob, 0.0, 1.0))
    P_A_mix = (1.0 - lp) * P_A + lp * 0.5 * _lapse_pdf_vec(t_v, beta_lapse)
    C_A_mix = np.clip((1.0 - lp) * C_A + lp * _lapse_cdf_vec(t_v, beta_lapse), 0.0, 1.0)

    pdf_up = P_A_mix * (random_readout_if_EA_survives + P_E_plus_cum_up) + P_E_plus_up * (
        1.0 - C_A_mix
    )
    pdf_down = P_A_mix * (random_readout_if_EA_survives + P_E_plus_cum_down) + P_E_plus_down * (
        1.0 - C_A_mix
    )

    pdf_total = pdf_up + pdf_down
    eps = 1e-50
    pdf_total = np.where(np.isfinite(pdf_total) & (pdf_total > 0), pdf_total, eps)
    curve[valid] = np.where(pdf_total > eps, pdf_total, 0.0)
    return curve


def compute_one_sample_curve_from_row(
    row,
    t_pts,
    model_kind,
    theory_params,
    proactive_params,
    is_norm,
    is_time_vary,
    K_max,
):
    t_stim = float(row["intended_fix"])
    t_led = float(row["t_LED"])
    abl = float(row["ABL"])
    ild = float(row["ILD"])

    if model_kind == "abl_delay":
        curve = build_abl_delay_theory_curve_for_trial(
            t_pts=t_pts,
            t_stim=t_stim,
            abl=abl,
            ild=ild,
            rate_lambda=theory_params["rate_lambda"],
            T_0=theory_params["T_0"],
            theta_E=theory_params["theta_E"],
            Z_E=theory_params["Z_E"],
            t_E_aff_20=theory_params["t_E_aff_20"],
            t_E_aff_40=theory_params["t_E_aff_40"],
            t_E_aff_60=theory_params["t_E_aff_60"],
            del_go=theory_params["del_go"],
            rate_norm_l=theory_params["rate_norm_l"],
            V_A=proactive_params["V_A"],
            theta_A=proactive_params["theta_A"],
            t_A_aff=proactive_params["t_A_aff"],
            lapse_prob=proactive_params["lapse_prob"],
            beta_lapse=proactive_params["beta_lapse"],
            is_norm=is_norm,
            is_time_vary=is_time_vary,
            K_max=K_max,
        )
    elif model_kind == "linear_delay":
        curve = build_linear_delay_theory_curve_for_trial(
            t_pts=t_pts,
            t_stim=t_stim,
            abl=abl,
            ild=ild,
            rate_lambda=theory_params["rate_lambda"],
            T_0=theory_params["T_0"],
            theta_E=theory_params["theta_E"],
            Z_E=theory_params["Z_E"],
            bias_ms=theory_params["bias_ms"],
            abl_delay_coeff_ms_per_abl=theory_params["abl_delay_coeff_ms_per_abl"],
            abs_ild_delay_coeff_ms_per_unit=theory_params["abs_ild_delay_coeff_ms_per_unit"],
            del_go=theory_params["del_go"],
            rate_norm_l=theory_params["rate_norm_l"],
            V_A=proactive_params["V_A"],
            theta_A=proactive_params["theta_A"],
            t_A_aff=proactive_params["t_A_aff"],
            lapse_prob=proactive_params["lapse_prob"],
            beta_lapse=proactive_params["beta_lapse"],
            is_norm=is_norm,
            is_time_vary=is_time_vary,
            K_max=K_max,
        )
    else:
        raise ValueError(f"Unsupported model_kind {model_kind!r}.")

    return curve, t_stim, t_led, abl, ild


def compute_average_curve_from_rows(
    sampled_rows,
    t_pts,
    model_kind,
    theory_params,
    proactive_params,
    n_jobs,
    is_norm,
    is_time_vary,
    K_max,
):
    if len(sampled_rows) == 0:
        return {
            "theory_density": np.zeros_like(t_pts, dtype=np.float64),
            "sampled_positions": np.array([], dtype=int),
            "sampled_t_stim": np.array([], dtype=np.float64),
            "sampled_t_LED": np.array([], dtype=np.float64),
            "sampled_ABL": np.array([], dtype=np.float64),
            "sampled_ILD": np.array([], dtype=np.float64),
        }

    mc_results = Parallel(n_jobs=n_jobs)(
        delayed(compute_one_sample_curve_from_row)(
            row,
            t_pts,
            model_kind,
            theory_params,
            proactive_params,
            is_norm,
            is_time_vary,
            K_max,
        )
        for _, row in sampled_rows.iterrows()
    )
    theory_matrix = np.stack([result[0] for result in mc_results], axis=0)

    return {
        "theory_density": np.mean(theory_matrix, axis=0),
        "sampled_positions": sampled_rows.index.to_numpy(dtype=int, copy=True),
        "sampled_t_stim": np.array([result[1] for result in mc_results], dtype=np.float64),
        "sampled_t_LED": np.array([result[2] for result in mc_results], dtype=np.float64),
        "sampled_ABL": np.array([result[3] for result in mc_results], dtype=np.float64),
        "sampled_ILD": np.array([result[4] for result in mc_results], dtype=np.float64),
    }


# %%
############ Load fit-result PKLs + posterior-mean parameters ############
for pkl_path in [abl_delay_results_pkl_path, linear_delay_results_pkl_path]:
    if not pkl_path.exists():
        raise FileNotFoundError(f"Could not find fit-results pickle: {pkl_path}")

with open(abl_delay_results_pkl_path, "rb") as f:
    abl_delay_fit_payload = pickle.load(f)
with open(linear_delay_results_pkl_path, "rb") as f:
    linear_delay_fit_payload = pickle.load(f)

abl_delay_vbmc = abl_delay_fit_payload["vbmc_norm_tied_results"]
abl_delay_loaded_pro = abl_delay_fit_payload["loaded_proactive_params"]
linear_delay_vbmc = linear_delay_fit_payload["vbmc_norm_tied_results"]
linear_delay_loaded_pro = linear_delay_fit_payload["loaded_proactive_params"]

required_abl_keys = [
    "rate_lambda_samples",
    "T_0_samples",
    "theta_E_samples",
    "w_samples",
    "t_E_aff_20_samples",
    "t_E_aff_40_samples",
    "t_E_aff_60_samples",
    "del_go_samples",
    "rate_norm_l_samples",
]
for key in required_abl_keys:
    if key not in abl_delay_vbmc:
        raise KeyError(f"Missing key in ABL-delay vbmc_norm_tied_results: {key}")

required_linear_keys = [
    "rate_lambda_samples",
    "T_0_samples",
    "theta_E_samples",
    "w_samples",
    "bias_ms_samples",
    "abl_delay_coeff_ms_per_abl_samples",
    "abs_ild_delay_coeff_ms_per_unit_samples",
    "del_go_samples",
    "rate_norm_l_samples",
]
for key in required_linear_keys:
    if key not in linear_delay_vbmc:
        raise KeyError(f"Missing key in linear-delay vbmc_norm_tied_results: {key}")

required_pro_keys = [
    "V_A_base",
    "theta_A",
    "del_a_minus_del_LED",
    "del_m_plus_del_LED",
    "lapse_prob",
    "beta_lapse",
]
for key in required_pro_keys:
    if key not in abl_delay_loaded_pro:
        raise KeyError(f"Missing key in ABL-delay loaded_proactive_params: {key}")
    if key not in linear_delay_loaded_pro:
        raise KeyError(f"Missing key in linear-delay loaded_proactive_params: {key}")

abl_delay_theory_params = {
    "rate_lambda": float(np.mean(abl_delay_vbmc["rate_lambda_samples"])),
    "T_0": float(np.mean(abl_delay_vbmc["T_0_samples"])),
    "theta_E": float(np.mean(abl_delay_vbmc["theta_E_samples"])),
    "w": float(np.mean(abl_delay_vbmc["w_samples"])),
    "t_E_aff_20": float(np.mean(abl_delay_vbmc["t_E_aff_20_samples"])),
    "t_E_aff_40": float(np.mean(abl_delay_vbmc["t_E_aff_40_samples"])),
    "t_E_aff_60": float(np.mean(abl_delay_vbmc["t_E_aff_60_samples"])),
    "del_go": float(np.mean(abl_delay_vbmc["del_go_samples"])),
    "rate_norm_l": float(np.mean(abl_delay_vbmc["rate_norm_l_samples"])),
}
abl_delay_theory_params["Z_E"] = (
    (abl_delay_theory_params["w"] - 0.5) * 2.0 * abl_delay_theory_params["theta_E"]
)

linear_delay_theory_params = {
    "rate_lambda": float(np.mean(linear_delay_vbmc["rate_lambda_samples"])),
    "T_0": float(np.mean(linear_delay_vbmc["T_0_samples"])),
    "theta_E": float(np.mean(linear_delay_vbmc["theta_E_samples"])),
    "w": float(np.mean(linear_delay_vbmc["w_samples"])),
    "bias_ms": float(np.mean(linear_delay_vbmc["bias_ms_samples"])),
    "abl_delay_coeff_ms_per_abl": float(
        np.mean(linear_delay_vbmc["abl_delay_coeff_ms_per_abl_samples"])
    ),
    "abs_ild_delay_coeff_ms_per_unit": float(
        np.mean(linear_delay_vbmc["abs_ild_delay_coeff_ms_per_unit_samples"])
    ),
    "del_go": float(np.mean(linear_delay_vbmc["del_go_samples"])),
    "rate_norm_l": float(np.mean(linear_delay_vbmc["rate_norm_l_samples"])),
}
linear_delay_theory_params["Z_E"] = (
    (linear_delay_theory_params["w"] - 0.5) * 2.0 * linear_delay_theory_params["theta_E"]
)

abl_delay_proactive_params = {
    "V_A": float(abl_delay_loaded_pro["V_A_base"]),
    "theta_A": float(abl_delay_loaded_pro["theta_A"]),
    "lapse_prob": float(abl_delay_loaded_pro["lapse_prob"]),
    "beta_lapse": float(abl_delay_loaded_pro["beta_lapse"]),
}
abl_delay_proactive_params["t_A_aff"] = float(
    abl_delay_loaded_pro["del_a_minus_del_LED"] + abl_delay_loaded_pro["del_m_plus_del_LED"]
)

linear_delay_proactive_params = {
    "V_A": float(linear_delay_loaded_pro["V_A_base"]),
    "theta_A": float(linear_delay_loaded_pro["theta_A"]),
    "lapse_prob": float(linear_delay_loaded_pro["lapse_prob"]),
    "beta_lapse": float(linear_delay_loaded_pro["beta_lapse"]),
}
linear_delay_proactive_params["t_A_aff"] = float(
    linear_delay_loaded_pro["del_a_minus_del_LED"] + linear_delay_loaded_pro["del_m_plus_del_LED"]
)

print("Loaded fit-results PKLs:")
print(f"  ABL-delay 115ms allvalid: {abl_delay_results_pkl_path}")
print(f"  Linear-delay no-choice:   {linear_delay_results_pkl_path}")
print("Posterior-mean ABL-delay no-choice parameters:")
print(
    f"  rate_lambda={abl_delay_theory_params['rate_lambda']:.6f}, "
    f"T_0={abl_delay_theory_params['T_0']:.6f}, "
    f"theta_E={abl_delay_theory_params['theta_E']:.6f}, "
    f"w={abl_delay_theory_params['w']:.6f}, "
    f"Z_E={abl_delay_theory_params['Z_E']:.6f}, "
    f"t_E_aff_20={1e3 * abl_delay_theory_params['t_E_aff_20']:.3f}ms, "
    f"t_E_aff_40={1e3 * abl_delay_theory_params['t_E_aff_40']:.3f}ms, "
    f"t_E_aff_60={1e3 * abl_delay_theory_params['t_E_aff_60']:.3f}ms, "
    f"del_go={abl_delay_theory_params['del_go']:.6f}, "
    f"rate_norm_l={abl_delay_theory_params['rate_norm_l']:.6f}"
)
print(
    f"  V_A={abl_delay_proactive_params['V_A']:.6f}, "
    f"theta_A={abl_delay_proactive_params['theta_A']:.6f}, "
    f"t_A_aff={abl_delay_proactive_params['t_A_aff']:.6f}, "
    f"lapse_prob={abl_delay_proactive_params['lapse_prob']:.6f}, "
    f"beta_lapse={abl_delay_proactive_params['beta_lapse']:.6f}"
)
print("Posterior-mean linear-delay no-choice parameters:")
print(
    f"  rate_lambda={linear_delay_theory_params['rate_lambda']:.6f}, "
    f"T_0={linear_delay_theory_params['T_0']:.6f}, "
    f"theta_E={linear_delay_theory_params['theta_E']:.6f}, "
    f"w={linear_delay_theory_params['w']:.6f}, "
    f"Z_E={linear_delay_theory_params['Z_E']:.6f}, "
    f"bias_ms={linear_delay_theory_params['bias_ms']:.6f}, "
    f"abl_delay_coeff_ms_per_abl={linear_delay_theory_params['abl_delay_coeff_ms_per_abl']:.6f}, "
    f"abs_ild_delay_coeff_ms_per_unit={linear_delay_theory_params['abs_ild_delay_coeff_ms_per_unit']:.6f}, "
    f"del_go={linear_delay_theory_params['del_go']:.6f}, "
    f"rate_norm_l={linear_delay_theory_params['rate_norm_l']:.6f}"
)
print(
    f"  V_A={linear_delay_proactive_params['V_A']:.6f}, "
    f"theta_A={linear_delay_proactive_params['theta_A']:.6f}, "
    f"t_A_aff={linear_delay_proactive_params['t_A_aff']:.6f}, "
    f"lapse_prob={linear_delay_proactive_params['lapse_prob']:.6f}, "
    f"beta_lapse={linear_delay_proactive_params['beta_lapse']:.6f}"
)


# %%
############ Rebuild LED-OFF aggregate diagnostics dataset ############
if not led_data_csv_path.exists():
    raise FileNotFoundError(f"Could not find data CSV: {led_data_csv_path}")

exp_df = pd.read_csv(led_data_csv_path)
exp_df["RTwrtStim"] = exp_df["timed_fix"] - exp_df["intended_fix"]
exp_df["t_LED"] = exp_df["intended_fix"] - exp_df["LED_onset_time"]
exp_df = exp_df.rename(columns={"timed_fix": "TotalFixTime"})
exp_df["abs_ILD"] = exp_df["ILD"].abs()

exp_df = exp_df[exp_df["RTwrtStim"] <= max_rtwrtstim_for_fit].copy()
exp_df = exp_df[~((exp_df["RTwrtStim"].isna()) & (exp_df["abort_event"] == 3))].copy()

mask_nan = exp_df["response_poke"].isna()
mask_success_1 = exp_df["success"] == 1
mask_success_neg1 = exp_df["success"] == -1
mask_ild_pos = exp_df["ILD"] > 0
mask_ild_neg = exp_df["ILD"] < 0
exp_df.loc[mask_nan & mask_success_1 & mask_ild_pos, "response_poke"] = 3
exp_df.loc[mask_nan & mask_success_1 & mask_ild_neg, "response_poke"] = 2
exp_df.loc[mask_nan & mask_success_neg1 & mask_ild_pos, "response_poke"] = 2
exp_df.loc[mask_nan & mask_success_neg1 & mask_ild_neg, "response_poke"] = 3

mask_led_off = (exp_df["LED_trial"] == 0) | (exp_df["LED_trial"].isna())
mask_repeat = exp_df["repeat_trial"].isin(allowed_repeat_trials) | exp_df["repeat_trial"].isna()
exp_df_led_off = exp_df[
    mask_led_off
    & mask_repeat
    & exp_df["session_type"].isin([session_type])
    & exp_df["training_level"].isin([training_level])
].copy()
exp_df_led_off["choice"] = np.where(
    exp_df_led_off["response_poke"] == 3,
    1,
    np.where(exp_df_led_off["response_poke"] == 2, -1, np.nan),
)

df_valid_and_aborts = exp_df_led_off[
    (exp_df_led_off["success"].isin([1, -1])) | (exp_df_led_off["abort_event"].isin([3, 4]))
].copy()
fit_df = df_valid_and_aborts[df_valid_and_aborts["RTwrtStim"] <= max_rtwrtstim_for_fit].copy()

if len(fit_df) == 0:
    raise ValueError("No LED-OFF trials found after filtering.")

observed_abl_values = validate_supported_values(fit_df, "ABL", supported_abl_values)
abl_counts = format_counts(fit_df["ABL"])

print("Rebuilt LED-OFF aggregate diagnostics dataset for comparison:")
print(f"  Total LED-OFF filtered trials (valid+aborts): {len(df_valid_and_aborts)}")
print(f"  LED-OFF trials used for comparison (valid+aborts): {len(fit_df)}")
print(f"  Supported ABL values in comparison dataset: {observed_abl_values.tolist()}")
print(f"  Comparison trial counts by ABL: {abl_counts}")


# %%
############ Build per-ABL data + theory payloads on the post-stim [0, 1]s window ############
rng = np.random.default_rng(seed)
mask_poststim_window = (t_pts >= 0.0) & (t_pts <= max_rtwrtstim_for_fit)
t_pts_poststim = t_pts[mask_poststim_window]
hist_edges_poststim = np.arange(
    0.0,
    max_rtwrtstim_for_fit + data_bin_size_s_poststim,
    data_bin_size_s_poststim,
)
if hist_edges_poststim[-1] < max_rtwrtstim_for_fit:
    hist_edges_poststim = np.append(hist_edges_poststim, max_rtwrtstim_for_fit)
data_bin_widths_poststim = np.diff(hist_edges_poststim)
data_bin_centers_poststim = 0.5 * (hist_edges_poststim[:-1] + hist_edges_poststim[1:])

visible_theory_mask = t_pts_poststim <= comparison_xlim_s
visible_data_mask = data_bin_centers_poststim <= comparison_xlim_s

abl_panel_payload = {}
combined_ax_max = 0.0

print(
    f"Computing per-ABL comparison curves with N_mc_per_abl_panel={N_mc_per_abl_panel}, "
    f"n_jobs={n_jobs} ..."
)

abl_values_float = fit_df["ABL"].astype(float).to_numpy()
for abl in supported_abl_values:
    abl_int = int(abl)
    df_abl = fit_df[np.isclose(abl_values_float, float(abl))].copy()
    if len(df_abl) == 0:
        raise ValueError(f"No rows found for ABL={abl_int} in comparison dataset.")

    sampled_positions = rng.integers(0, len(df_abl), size=N_mc_per_abl_panel)
    sampled_rows = df_abl.iloc[sampled_positions].copy()

    abl_delay_curve_payload = compute_average_curve_from_rows(
        sampled_rows=sampled_rows,
        t_pts=t_pts,
        model_kind="abl_delay",
        theory_params=abl_delay_theory_params,
        proactive_params=abl_delay_proactive_params,
        n_jobs=n_jobs,
        is_norm=is_norm,
        is_time_vary=is_time_vary,
        K_max=K_max,
    )
    linear_delay_curve_payload = compute_average_curve_from_rows(
        sampled_rows=sampled_rows,
        t_pts=t_pts,
        model_kind="linear_delay",
        theory_params=linear_delay_theory_params,
        proactive_params=linear_delay_proactive_params,
        n_jobs=n_jobs,
        is_norm=is_norm,
        is_time_vary=is_time_vary,
        K_max=K_max,
    )

    data_and_abl_delay_payload = build_truncated_density_payload(
        rt_values=df_abl["RTwrtStim"].to_numpy(dtype=np.float64),
        n_total_condition=len(df_abl),
        theory_density_full=abl_delay_curve_payload["theory_density"],
        mask_truncated=mask_poststim_window,
        t_pts_truncated=t_pts_poststim,
        hist_edges_truncated=hist_edges_poststim,
        data_bin_widths_truncated=data_bin_widths_poststim,
        truncate_rt_wrt_stim_s=max_rtwrtstim_for_fit,
        normalize_within_window=False,
    )
    linear_delay_only_payload = build_truncated_density_payload(
        rt_values=df_abl["RTwrtStim"].to_numpy(dtype=np.float64),
        n_total_condition=len(df_abl),
        theory_density_full=linear_delay_curve_payload["theory_density"],
        mask_truncated=mask_poststim_window,
        t_pts_truncated=t_pts_poststim,
        hist_edges_truncated=hist_edges_poststim,
        data_bin_widths_truncated=data_bin_widths_poststim,
        truncate_rt_wrt_stim_s=max_rtwrtstim_for_fit,
        normalize_within_window=False,
    )

    abl_panel_payload[abl_int] = {
        "n_rows": int(len(df_abl)),
        "n_poststim_points": int(data_and_abl_delay_payload["n_truncated_points"]),
        "sampled_positions": sampled_rows.index.to_numpy(dtype=int, copy=True),
        "sampled_t_stim": abl_delay_curve_payload["sampled_t_stim"],
        "sampled_t_LED": abl_delay_curve_payload["sampled_t_LED"],
        "sampled_ILD": abl_delay_curve_payload["sampled_ILD"],
        "data_density_poststim_plot": data_and_abl_delay_payload["data_density_truncated_plot"],
        "abl_delay_density_poststim_plot": data_and_abl_delay_payload["theory_density_truncated_plot"],
        "linear_delay_density_poststim_plot": linear_delay_only_payload["theory_density_truncated_plot"],
    }

    combined_ax_max = max(
        combined_ax_max,
        float(np.max(abl_panel_payload[abl_int]["data_density_poststim_plot"][visible_data_mask])),
        float(np.max(abl_panel_payload[abl_int]["abl_delay_density_poststim_plot"][visible_theory_mask])),
        float(np.max(abl_panel_payload[abl_int]["linear_delay_density_poststim_plot"][visible_theory_mask])),
    )

    print(
        f"ABL={abl_int}: n_rows={len(df_abl)}, "
        f"n_poststim={abl_panel_payload[abl_int]['n_poststim_points']}, "
        f"sampled_rows={len(sampled_rows)}"
    )


# %%
############ Plot 1 x 3 ABL comparison ############
fig, axes = plt.subplots(1, 3, figsize=comparison_figsize, sharey=True)

for ax, abl in zip(axes, supported_abl_values):
    abl_int = int(abl)
    color = abl_colors[abl_int]
    payload_abl = abl_panel_payload[abl_int]

    ax.step(
        data_bin_centers_poststim,
        payload_abl["data_density_poststim_plot"],
        where="mid",
        lw=2.0,
        color=color,
        alpha=0.75,
        label="Data",
    )
    ax.plot(
        t_pts_poststim,
        payload_abl["abl_delay_density_poststim_plot"],
        lw=2.2,
        color=abl_delay_model_color,
        label="ABL-delay 115ms",
    )
    ax.plot(
        t_pts_poststim,
        payload_abl["linear_delay_density_poststim_plot"],
        lw=2.2,
        color=linear_delay_model_color,
        label="Linear delay",
    )
    ax.set_xlim(0.0, comparison_xlim_s)
    ax.set_title(
        f"ABL {abl_int}"
    )
    ax.set_xlabel("RT - t_stim (s)")
    if ax is axes[0]:
        ax.set_ylabel("Density")
    ax.legend(fontsize=9)

if combined_ax_max > 0:
    axes[0].set_ylim(0.0, combined_ax_max * 1.1)

abl_counts_str = ", ".join(
    f"ABL{int(abl)}={abl_panel_payload[int(abl)]['n_rows']}" for abl in supported_abl_values
)
fig.suptitle(
    "trunc + ABL wise delay vs linear delay fit",
    y=1.03,
)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(output_pdf_path, bbox_inches="tight")
fig.savefig(output_png_path, dpi=comparison_plot_dpi, bbox_inches="tight")
if show_plot:
    plt.show()
plt.close(fig)

print(f"Saved comparison plot (PDF): {output_pdf_path}")
print(f"Saved comparison plot (PNG): {output_png_path}")
