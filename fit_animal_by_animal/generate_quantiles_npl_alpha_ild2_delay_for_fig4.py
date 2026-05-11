# %%
import os
import pickle
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import scipy.special as scipy_special
from scipy.stats import sem

from animal_wise_plotting_utils import calculate_theoretical_curves
from time_vary_norm_alpha_utils import rho_A_t_fn
from time_vary_norm_utils import M, phi, rho_A_t_VEC_fn


# %%
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

DESIRED_BATCHES = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]
ABL_arr = [20, 40, 60]
ILD_arr = [-16.0, -8.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0, 16.0]
QUANTILES_TO_PLOT = [0.1, 0.3, 0.5, 0.7, 0.9]
CONTINUOUS_ILD_STEP = 0.1
K_max = 10
T_PTS = np.arange(-2, 2, 0.001)
N_THEORY = int(1e3)
N_JOBS = max(1, min(30, (os.cpu_count() or 2) - 1))

ILD2_RESULTS_DIR = os.path.join(REPO_DIR, "NPL_alpha_ILD2_fit_results", "result_pkls")
OUTPUT_DIR = os.path.join(REPO_DIR, "NPL_alpha_ILD2_fit_results", "figure_4_diagnostics")
os.makedirs(OUTPUT_DIR, exist_ok=True)
QUANT_OUTPUT_PKL = os.path.join(OUTPUT_DIR, "npl_alpha_ild2_quant_fig4_data.pkl")

RESULT_KEY = "vbmc_norm_alpha_ild2_delay_tied_results"
RESULT_RE = re.compile(r"^results_(?P<batch>.+)_animal_(?P<animal>\d+)_NORM_ALPHA_ILD2_DELAY_FROM_ABORTS\.pkl$")
BATCH_T_TRUNC = {"LED34_even": 0.15}
DEFAULT_T_TRUNC = 0.3


# %%
def _create_innermost_dict():
    return {"empirical": [], "theoretical": []}


def _create_inner_defaultdict():
    return defaultdict(_create_innermost_dict)


def delay_s_from_params(ABL, ILD, params):
    delay_ms = (
        params["bias_ms"]
        + params["abl_delay_coeff_ms_per_abl"] * ABL
        + params["abs_ild_delay_coeff_ms_per_unit"] * abs(ILD)
        + params["ild2_delay_coeff_ms_per_unit2"] * (ILD ** 2)
    )
    return delay_ms * 1e-3


def Phi_vec(x):
    return 0.5 * (1 + scipy_special.erf(x / np.sqrt(2)))


def cum_A_t_vec(t, V_A, theta_A):
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t, dtype=float)
    valid = t > 0
    t_valid = t[valid]
    out[valid] = (
        Phi_vec(V_A * (t_valid - theta_A / V_A) / np.sqrt(t_valid))
        + np.exp(2 * V_A * theta_A)
        * Phi_vec(-V_A * (t_valid + theta_A / V_A) / np.sqrt(t_valid))
    )
    return out


def gamma_omega_alpha_vec(ABL, ILD, rate_lambda, T_0, theta_E, rate_norm_l, alpha):
    chi = 17.37
    ABL = np.asarray(ABL, dtype=float)
    ILD = np.asarray(ILD, dtype=float)

    abl_term = 10 ** (rate_lambda * (1 - rate_norm_l) * ABL / 20)
    ild_arg = rate_lambda * ILD / chi
    norm_ild_arg = rate_lambda * rate_norm_l * ILD / chi

    r_r = abl_term * np.exp(ild_arg) / (np.exp(norm_ild_arg) + alpha * np.exp(-norm_ild_arg))
    r_l = abl_term * np.exp(-ild_arg) / (np.exp(-norm_ild_arg) + alpha * np.exp(norm_ild_arg))

    r_sum = r_r + r_l
    gamma = theta_E * (r_r - r_l) / r_sum
    omega = r_sum / (T_0 * (theta_E ** 2))
    return gamma, omega


def CDF_E_alpha_vec(t, bound, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, rate_norm_l, alpha):
    t_original = np.asarray(t, dtype=float)
    bound = np.asarray(bound)
    v, omega = gamma_omega_alpha_vec(ABL, ILD, rate_lambda, T_0, theta_E, rate_norm_l, alpha)

    w = 0.5 + (Z_E / (2.0 * theta_E))
    w = np.asarray(w, dtype=float)
    a = 2
    v = np.where(bound == 1, -v, v)
    w = np.where(bound == 1, 1 - w, w)

    t_eff = omega * t_original
    shape = np.broadcast(t_eff, v, w).shape
    out = np.zeros(shape, dtype=float)
    valid = np.broadcast_to(t_original, shape) > 0
    safe_t = np.where(valid, np.broadcast_to(t_eff, shape), 1e-12)

    v_full = np.broadcast_to(v, shape)
    w_full = np.broadcast_to(w, shape)
    exponent_arg = -v_full * a * w_full - ((v_full ** 2) * safe_t / 2)
    result = np.exp(exponent_arg)

    k_arr = np.arange(K_max + 1)
    t_b = safe_t[..., None]
    v_b = v_full[..., None]
    w_b = w_full[..., None]
    k_b = k_arr.reshape((1,) * len(shape) + (K_max + 1,))

    r_k = np.where(k_b % 2 == 0, k_b * a + a * w_b, k_b * a + a * (1 - w_b))
    sqrt_t = np.sqrt(t_b)
    term1 = phi(r_k / sqrt_t)
    term2 = M((r_k - v_b * t_b) / sqrt_t) + M((r_k + v_b * t_b) / sqrt_t)
    summation = np.sum(((-1) ** k_b) * term1 * term2, axis=-1)

    out[valid] = (result * summation)[valid]
    return out


def rho_E_alpha_vec(t, bound, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, rate_norm_l, alpha):
    t_original = np.asarray(t, dtype=float)
    bound = np.asarray(bound)
    v, omega = gamma_omega_alpha_vec(ABL, ILD, rate_lambda, T_0, theta_E, rate_norm_l, alpha)

    w = 0.5 + (Z_E / (2.0 * theta_E))
    w = np.asarray(w, dtype=float)
    a = 2
    v = np.where(bound == 1, -v, v)
    w = np.where(bound == 1, 1 - w, w)

    t_eff = omega * t_original
    shape = np.broadcast(t_eff, v, w).shape
    out = np.zeros(shape, dtype=float)
    valid = np.broadcast_to(t_original, shape) > 0
    safe_t = np.where(valid, np.broadcast_to(t_eff, shape), 1e-12)

    v_full = np.broadcast_to(v, shape)
    w_full = np.broadcast_to(w, shape)
    non_sum_term = (
        (1 / a**2)
        * (a**3 / np.sqrt(2 * np.pi * safe_t**3))
        * np.exp(-v_full * a * w_full - (v_full**2 * safe_t) / 2)
    )

    K_half = int(K_max / 2)
    k_vals = np.linspace(-K_half, K_half, 2 * K_half + 1)
    t_b = safe_t[..., None]
    w_b = w_full[..., None]
    k_b = k_vals.reshape((1,) * len(shape) + (2 * K_half + 1,))

    sum_w_term = w_b + 2 * k_b
    sum_exp_term = np.exp(-(a**2 * (w_b + 2 * k_b) ** 2) / (2 * t_b))
    sum_result = np.sum(sum_w_term * sum_exp_term, axis=-1)

    density = non_sum_term * sum_result
    density = np.where(density <= 0, 1e-16, density)
    out[valid] = (density * np.broadcast_to(omega, shape))[valid]
    return out


def up_or_down_alpha_pa_ca_vec(t, bound, P_A, C_A, ABL, ILD, tied_params, Z_E, t_E_aff):
    t = np.asarray(t, dtype=float)
    t1 = np.maximum(t - t_E_aff, 1e-6)
    t2 = np.maximum(t - t_E_aff + tied_params["del_go"], 1e-6)

    P_EA_hits_either_bound = (
        CDF_E_alpha_vec(
            t - t_E_aff + tied_params["del_go"],
            1,
            ABL,
            ILD,
            tied_params["rate_lambda"],
            tied_params["T_0"],
            tied_params["theta_E"],
            Z_E,
            tied_params["rate_norm_l"],
            tied_params["alpha"],
        )
        + CDF_E_alpha_vec(
            t - t_E_aff + tied_params["del_go"],
            -1,
            ABL,
            ILD,
            tied_params["rate_lambda"],
            tied_params["T_0"],
            tied_params["theta_E"],
            Z_E,
            tied_params["rate_norm_l"],
            tied_params["alpha"],
        )
    )
    random_readout_if_EA_survives = 0.5 * (1 - P_EA_hits_either_bound)

    P_E_plus_cum = (
        CDF_E_alpha_vec(
            t2,
            bound,
            ABL,
            ILD,
            tied_params["rate_lambda"],
            tied_params["T_0"],
            tied_params["theta_E"],
            Z_E,
            tied_params["rate_norm_l"],
            tied_params["alpha"],
        )
        - CDF_E_alpha_vec(
            t1,
            bound,
            ABL,
            ILD,
            tied_params["rate_lambda"],
            tied_params["T_0"],
            tied_params["theta_E"],
            Z_E,
            tied_params["rate_norm_l"],
            tied_params["alpha"],
        )
    )
    P_E_plus = rho_E_alpha_vec(
        t - t_E_aff,
        bound,
        ABL,
        ILD,
        tied_params["rate_lambda"],
        tied_params["T_0"],
        tied_params["theta_E"],
        Z_E,
        tied_params["rate_norm_l"],
        tied_params["alpha"],
    )

    return P_A * (random_readout_if_EA_survives + P_E_plus_cum) + P_E_plus * (1 - C_A)


def cum_pro_and_reactive_alpha_vec(t, c_A_trunc_time, abort_params, t_stim, ABL, ILD, tied_params, Z_E, t_E_aff):
    t = np.asarray(t, dtype=float)
    c_A = cum_A_t_vec(t - abort_params["t_A_aff"], abort_params["V_A"], abort_params["theta_A"])
    if c_A_trunc_time is not None:
        trunc_denom = 1 - cum_A_t_vec(
            np.array([c_A_trunc_time - abort_params["t_A_aff"]]),
            abort_params["V_A"],
            abort_params["theta_A"],
        )[0]
        c_A = np.where(t < c_A_trunc_time, 0, c_A / trunc_denom)

    c_E = (
        CDF_E_alpha_vec(
            t - t_stim - t_E_aff,
            1,
            ABL,
            ILD,
            tied_params["rate_lambda"],
            tied_params["T_0"],
            tied_params["theta_E"],
            Z_E,
            tied_params["rate_norm_l"],
            tied_params["alpha"],
        )
        + CDF_E_alpha_vec(
            t - t_stim - t_E_aff,
            -1,
            ABL,
            ILD,
            tied_params["rate_lambda"],
            tied_params["T_0"],
            tied_params["theta_E"],
            Z_E,
            tied_params["rate_norm_l"],
            tied_params["alpha"],
        )
    )
    return c_A + c_E - c_A * c_E


def get_ild2_result_paths():
    paths = {}
    for filename in sorted(os.listdir(ILD2_RESULTS_DIR)):
        match = RESULT_RE.match(filename)
        if not match:
            continue
        key = (match.group("batch"), int(match.group("animal")))
        paths[key] = os.path.join(ILD2_RESULTS_DIR, filename)
    return paths


def load_posterior_mean_params(batch_name, animal_id, result_paths):
    with open(result_paths[(batch_name, int(animal_id))], "rb") as handle:
        fit_results = pickle.load(handle)

    abort_samples = fit_results["vbmc_aborts_results"]
    tied_samples = fit_results[RESULT_KEY]

    abort_params = {
        "V_A": float(np.mean(abort_samples["V_A_samples"])),
        "theta_A": float(np.mean(abort_samples["theta_A_samples"])),
        "t_A_aff": float(np.mean(abort_samples["t_A_aff_samp"])),
    }
    tied_params = {
        "rate_lambda": float(np.mean(tied_samples["rate_lambda_samples"])),
        "T_0": float(np.mean(tied_samples["T_0_samples"])),
        "theta_E": float(np.mean(tied_samples["theta_E_samples"])),
        "w": float(np.mean(tied_samples["w_samples"])),
        "bias_ms": float(np.mean(tied_samples["bias_ms_samples"])),
        "abl_delay_coeff_ms_per_abl": float(np.mean(tied_samples["abl_delay_coeff_ms_per_abl_samples"])),
        "abs_ild_delay_coeff_ms_per_unit": float(np.mean(tied_samples["abs_ild_delay_coeff_ms_per_unit_samples"])),
        "ild2_delay_coeff_ms_per_unit2": float(np.mean(tied_samples["ild2_delay_coeff_ms_per_unit2_samples"])),
        "del_go": float(np.mean(tied_samples["del_go_samples"])),
        "rate_norm_l": float(np.mean(tied_samples["rate_norm_l_samples"])),
        "alpha": float(np.mean(tied_samples["alpha_samples"])),
    }
    return abort_params, tied_params


def get_p_a_c_a(batch_name, animal_id, abort_params):
    df = pd.read_csv(os.path.join("batch_csvs", f"batch_{batch_name}_valid_and_aborts.csv"))
    df_animal = df[df["animal"] == int(animal_id)]
    return calculate_theoretical_curves(
        df_animal,
        N_THEORY,
        T_PTS,
        abort_params["t_A_aff"],
        abort_params["V_A"],
        abort_params["theta_A"],
        rho_A_t_fn,
    )


def get_theoretical_rtd_from_params(P_A_mean, C_A_mean, t_stim_samples, abort_params, tied_params, ABL, ILD, batch_name):
    T_trunc = BATCH_T_TRUNC.get(batch_name, DEFAULT_T_TRUNC)
    t_E_aff = delay_s_from_params(ABL, ILD, tied_params)
    Z_E = (tied_params["w"] - 0.5) * 2 * tied_params["theta_E"]

    trunc_fac_samples = (
        cum_pro_and_reactive_alpha_vec(
            t_stim_samples + 1,
            T_trunc,
            abort_params,
            t_stim_samples,
            ABL,
            ILD,
            tied_params,
            Z_E,
            t_E_aff,
        )
        - cum_pro_and_reactive_alpha_vec(
            t_stim_samples,
            T_trunc,
            abort_params,
            t_stim_samples,
            ABL,
            ILD,
            tied_params,
            Z_E,
            t_E_aff,
        )
        + 1e-10
    )
    trunc_factor = np.mean(trunc_fac_samples)

    up_mean = up_or_down_alpha_pa_ca_vec(T_PTS, 1, P_A_mean, C_A_mean, ABL, ILD, tied_params, Z_E, t_E_aff)
    down_mean = up_or_down_alpha_pa_ca_vec(T_PTS, -1, P_A_mean, C_A_mean, ABL, ILD, tied_params, Z_E, t_E_aff)

    mask_0_1 = (T_PTS >= 0) & (T_PTS <= 1)
    t_pts_0_1 = T_PTS[mask_0_1]
    return t_pts_0_1, (up_mean[mask_0_1] + down_mean[mask_0_1]) / trunc_factor


def get_animal_raw_RTs(batch_name, animal_id, ABL, ILD):
    df = pd.read_csv(os.path.join("batch_csvs", f"batch_{batch_name}_valid_and_aborts.csv"))
    df_stim = df[
        (df["animal"] == int(animal_id))
        & (df["ABL"] == ABL)
        & (df["ILD"] == ILD)
        & (df["success"].isin([1, -1]))
        & (df["RTwrtStim"] >= 0)
        & (df["RTwrtStim"] <= 1)
    ]
    return df_stim["RTwrtStim"].values


def find_quantile_from_cdf(q, cdf, x_axis):
    idx = np.searchsorted(cdf, q, side="left")
    if idx == 0:
        return x_axis[0]
    if idx == len(cdf):
        return x_axis[-1]
    x1, x2 = x_axis[idx - 1], x_axis[idx]
    y1, y2 = cdf[idx - 1], cdf[idx]
    if y2 == y1:
        return x1
    return x1 + (x2 - x1) * (q - y1) / (y2 - y1)


def process_animal_for_quantiles(batch_animal_pair, result_paths):
    batch_name, animal_id = batch_animal_pair
    print(f"Processing quantiles for {batch_name}, animal {animal_id}")
    animal_quantile_data = {}
    try:
        abort_params, tied_params = load_posterior_mean_params(batch_name, animal_id, result_paths)
        P_A_mean, C_A_mean, t_stim_samples = get_p_a_c_a(batch_name, animal_id, abort_params)

        for ABL in ABL_arr:
            for ILD in ILD_arr:
                raw_rts = get_animal_raw_RTs(batch_name, animal_id, ABL, ILD)
                if len(raw_rts) > 5:
                    emp_quantiles = np.quantile(raw_rts, QUANTILES_TO_PLOT)
                else:
                    emp_quantiles = [np.nan] * len(QUANTILES_TO_PLOT)
                animal_quantile_data[(ABL, ILD)] = {"empirical": emp_quantiles}

        continuous_ild_values = np.round(np.arange(-16.0, 16.0 + CONTINUOUS_ILD_STEP / 2, CONTINUOUS_ILD_STEP), 1)
        for ABL in ABL_arr:
            for ILD in continuous_ild_values:
                try:
                    t_pts, rtd = get_theoretical_rtd_from_params(
                        P_A_mean, C_A_mean, t_stim_samples, abort_params, tied_params, ABL, float(ILD), batch_name
                    )
                    if np.all(np.isnan(rtd)) or len(t_pts) < 2:
                        raise ValueError("Theoretical RTD is all NaN or too short")
                    cdf = np.cumsum(rtd) * (t_pts[1] - t_pts[0])
                    if cdf[-1] <= 1e-6:
                        raise ValueError("Theoretical CDF sum is close to zero")
                    cdf /= cdf[-1]
                    theo_quantiles = [find_quantile_from_cdf(q, cdf, t_pts) for q in QUANTILES_TO_PLOT]
                    animal_quantile_data[(ABL, float(ILD), "continuous")] = {"theoretical": theo_quantiles}

                    for discrete_ild in ILD_arr:
                        if abs(float(ILD) - discrete_ild) < 0.05 and (ABL, discrete_ild) in animal_quantile_data:
                            animal_quantile_data[(ABL, discrete_ild)]["theoretical"] = theo_quantiles
                            break
                except Exception:
                    pass
    except Exception as exc:
        print(f"ERROR processing {batch_name}/{animal_id}: {exc}")

    return animal_quantile_data


# %%
result_paths = get_ild2_result_paths()
batch_files = [os.path.join("batch_csvs", f"batch_{batch_name}_valid_and_aborts.csv") for batch_name in DESIRED_BATCHES]
merged_data = pd.concat([pd.read_csv(path) for path in batch_files if os.path.exists(path)], ignore_index=True)
merged_valid = merged_data[merged_data["success"].isin([1, -1])].copy()
batch_animal_pairs = sorted(
    [(batch, int(animal)) for batch, animal in merged_valid[["batch_name", "animal"]].drop_duplicates().values]
)
batch_animal_pairs = [pair for pair in batch_animal_pairs if pair in result_paths]

print(f"Found {len(result_paths)} ILD2 result pickles")
print(f"Using {len(batch_animal_pairs)} matched animals for quantiles")
print(f"Continuous ILD step: {CONTINUOUS_ILD_STEP}")
print(f"Running with {N_JOBS} jobs")

all_animal_results = Parallel(n_jobs=N_JOBS, verbose=10)(
    delayed(process_animal_for_quantiles)(pair, result_paths) for pair in batch_animal_pairs
)


# %%
print("Aggregating quantile data for plotting")
abs_ild_sorted = sorted(list(set(abs(ild) for ild in ILD_arr)))
continuous_abs_ild = np.round(np.arange(1.0, 16.0 + CONTINUOUS_ILD_STEP / 2, CONTINUOUS_ILD_STEP), 1)

plot_data = defaultdict(_create_inner_defaultdict)
continuous_plot_data = defaultdict(_create_inner_defaultdict)

for animal_data in all_animal_results:
    if not animal_data:
        continue
    for ABL in ABL_arr:
        for abs_ild in abs_ild_sorted:
            emp_quantiles_combined = []
            for ild_sign in [abs_ild, -abs_ild]:
                stim_key = (ABL, ild_sign)
                if stim_key in animal_data:
                    emp_quantiles_combined.append(animal_data[stim_key]["empirical"])
            if emp_quantiles_combined:
                plot_data[ABL][abs_ild]["empirical"].append(np.nanmean(emp_quantiles_combined, axis=0))

    for ABL in ABL_arr:
        continuous_keys = [
            key
            for key in animal_data.keys()
            if isinstance(key, tuple) and len(key) == 3 and key[2] == "continuous" and key[0] == ABL
        ]
        for key in continuous_keys:
            rounded_abs = np.round(abs(key[1]), 1)
            continuous_plot_data[ABL][rounded_abs]["theoretical"].append(animal_data[key]["theoretical"])

quantile_summary = []
for q_idx, q in enumerate(QUANTILES_TO_PLOT):
    emp_means, emp_sems = [], []
    for abs_ild in abs_ild_sorted:
        all_abl_emp_quantiles = np.concatenate([
            np.array(plot_data[ABL][abs_ild]["empirical"])[:, q_idx] for ABL in ABL_arr
        ])
        emp_means.append(np.nanmean(all_abl_emp_quantiles))
        emp_sems.append(sem(all_abl_emp_quantiles, nan_policy="omit"))

    theo_means, theo_sems, continuous_abs_ild_valid = [], [], []
    for abs_ild in continuous_abs_ild:
        all_abl_theo_quantiles = []
        for ABL in ABL_arr:
            if len(continuous_plot_data[ABL][abs_ild]["theoretical"]) > 0:
                all_abl_theo_quantiles.extend(np.array(continuous_plot_data[ABL][abs_ild]["theoretical"])[:, q_idx])
        if len(all_abl_theo_quantiles) > 0:
            theo_means.append(np.nanmean(all_abl_theo_quantiles))
            theo_sems.append(sem(all_abl_theo_quantiles, nan_policy="omit"))
            continuous_abs_ild_valid.append(abs_ild)

    quantile_summary.append({
        "q": q,
        "emp_abs_ild": abs_ild_sorted,
        "emp_means": emp_means,
        "emp_sems": emp_sems,
        "theo_abs_ild": continuous_abs_ild_valid,
        "theo_means": theo_means,
        "theo_sems": theo_sems,
    })

plot_data_for_pickle = {
    ABL: {
        abs_ild: {
            "empirical": list(plot_data[ABL][abs_ild]["empirical"]),
            "theoretical": list(plot_data[ABL][abs_ild]["theoretical"]),
        }
        for abs_ild in abs_ild_sorted
    }
    for ABL in ABL_arr
}
continuous_plot_data_for_pickle = {
    ABL: {
        abs_ild: {
            "empirical": list(continuous_plot_data[ABL][abs_ild]["empirical"]),
            "theoretical": list(continuous_plot_data[ABL][abs_ild]["theoretical"]),
        }
        for abs_ild in continuous_abs_ild
    }
    for ABL in ABL_arr
}

quantile_plot_data = {
    "plot_data": plot_data_for_pickle,
    "continuous_plot_data": continuous_plot_data_for_pickle,
    "quantile_summary": quantile_summary,
    "QUANTILES_TO_PLOT": QUANTILES_TO_PLOT,
    "abs_ild_sorted": abs_ild_sorted,
    "continuous_abs_ild": continuous_abs_ild,
    "ABL_arr": ABL_arr,
    "MODEL_TYPE": "npl_alpha_ild2_delay",
    "animal_keys": batch_animal_pairs,
}

with open(QUANT_OUTPUT_PKL, "wb") as handle:
    pickle.dump(quantile_plot_data, handle)

print(f"Saved {QUANT_OUTPUT_PKL}")
print(f"Animals processed: {sum(bool(item) for item in all_animal_results)}")

# %%
