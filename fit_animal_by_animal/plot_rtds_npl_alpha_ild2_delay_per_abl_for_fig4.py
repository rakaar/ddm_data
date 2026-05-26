# %%
import os
import pickle
import re
from collections import defaultdict

os.makedirs("/tmp/matplotlib", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import sem
import scipy.special as scipy_special

import figure_template as ft
from animal_wise_plotting_utils import calculate_theoretical_curves
from time_vary_norm_alpha_utils import rho_A_t_fn
from time_vary_norm_utils import M, phi

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]


# %%
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

DESIRED_BATCHES = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]
ABL_ARR = [20, 60]
ABS_ILD_ARR = [1.0, 2.0, 4.0, 8.0, 16.0]
ILD_ARR = sorted([-ild for ild in ABS_ILD_ARR] + ABS_ILD_ARR)
RT_BINS = np.arange(0.0, 1.0 + 0.02, 0.02)
RT_BIN_CENTERS = 0.5 * (RT_BINS[:-1] + RT_BINS[1:])

K_MAX = 10
T_PTS = np.arange(-2, 2, 0.001)
N_THEORY = int(1e3)
N_JOBS = max(1, min(30, (os.cpu_count() or 2) - 1))

ILD2_RESULTS_DIR = os.path.join(REPO_DIR, "NPL_alpha_ILD2_fit_results", "result_pkls")
OUTPUT_DIR = os.path.join(REPO_DIR, "NPL_alpha_ILD2_fit_results", "figure_4_diagnostics_part2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RTD_OUTPUT_PKL = os.path.join(OUTPUT_DIR, "npl_alpha_ild2_delay_rtds_abl20_abl60.pkl")
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "npl_alpha_ild2_delay_rtds_abl20_abl60.png")
OUTPUT_PDF = os.path.join(OUTPUT_DIR, "npl_alpha_ild2_delay_rtds_abl20_abl60.pdf")

RESULT_KEY = "vbmc_norm_alpha_ild2_delay_tied_results"
RESULT_RE = re.compile(r"^results_(?P<batch>.+)_animal_(?P<animal>\d+)_NORM_ALPHA_ILD2_DELAY_FROM_ABORTS\.pkl$")
BATCH_T_TRUNC = {"LED34_even": 0.15}
DEFAULT_T_TRUNC = 0.3


# %%
def _empty_condition():
    return {"empirical": [], "theoretical": []}


def _empty_abl_dict():
    return defaultdict(_empty_condition)


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

    k_arr = np.arange(K_MAX + 1)
    t_b = safe_t[..., None]
    v_b = v_full[..., None]
    w_b = w_full[..., None]
    k_b = k_arr.reshape((1,) * len(shape) + (K_MAX + 1,))

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

    K_half = int(K_MAX / 2)
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


def raw_rts_to_hist(raw_rts):
    if len(raw_rts) <= 5:
        return np.full(len(RT_BIN_CENTERS), np.nan)
    hist, _ = np.histogram(raw_rts, bins=RT_BINS, density=True)
    return hist.astype(float)


def theoretical_rtd_to_bin_density(t_pts, rtd):
    binned_rtd = []
    for bin_idx, (left, right) in enumerate(zip(RT_BINS[:-1], RT_BINS[1:])):
        if bin_idx == len(RT_BINS) - 2:
            in_bin = (t_pts >= left) & (t_pts <= right)
        else:
            in_bin = (t_pts >= left) & (t_pts < right)
        if np.sum(in_bin) < 2:
            binned_rtd.append(np.nan)
            continue
        binned_rtd.append(np.trapz(rtd[in_bin], t_pts[in_bin]) / (right - left))
    binned_rtd = np.asarray(binned_rtd, dtype=float)
    area = np.nansum(binned_rtd * np.diff(RT_BINS))
    if np.isfinite(area) and area > 0:
        binned_rtd = binned_rtd / area
    return binned_rtd


def process_animal_for_rtds(batch_animal_pair, result_paths):
    batch_name, animal_id = batch_animal_pair
    print(f"Processing RTDs for {batch_name}, animal {animal_id}")
    animal_rtd_data = {}
    try:
        abort_params, tied_params = load_posterior_mean_params(batch_name, animal_id, result_paths)
        P_A_mean, C_A_mean, t_stim_samples = get_p_a_c_a(batch_name, animal_id, abort_params)

        for ABL in ABL_ARR:
            for ILD in ILD_ARR:
                raw_rts = get_animal_raw_RTs(batch_name, animal_id, ABL, ILD)
                empirical_rtd = raw_rts_to_hist(raw_rts)

                t_pts, model_rtd = get_theoretical_rtd_from_params(
                    P_A_mean,
                    C_A_mean,
                    t_stim_samples,
                    abort_params,
                    tied_params,
                    ABL,
                    ILD,
                    batch_name,
                )
                theoretical_rtd = theoretical_rtd_to_bin_density(t_pts, model_rtd)

                animal_rtd_data[(ABL, ILD)] = {
                    "empirical": empirical_rtd,
                    "theoretical": theoretical_rtd,
                    "n_trials": len(raw_rts),
                }
    except Exception as exc:
        print(f"ERROR processing {batch_name}/{animal_id}: {exc}")

    return animal_rtd_data


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
print(f"Using {len(batch_animal_pairs)} matched animals for RTDs")
print(f"RT bins: {RT_BINS[0]:.2f} to {RT_BINS[-1]:.2f} s in {RT_BINS[1] - RT_BINS[0]:.2f} s steps")
print(f"Running with {N_JOBS} jobs")

all_animal_results = Parallel(n_jobs=N_JOBS, verbose=10)(
    delayed(process_animal_for_rtds)(pair, result_paths) for pair in batch_animal_pairs
)


# %%
print("Aggregating RTDs for plotting")
plot_data = defaultdict(_empty_abl_dict)
n_trials_by_condition = defaultdict(lambda: defaultdict(list))

for animal_data in all_animal_results:
    if not animal_data:
        continue
    for ABL in ABL_ARR:
        for abs_ild in ABS_ILD_ARR:
            empirical_signed = []
            theoretical_signed = []
            n_trials_signed = 0
            for ILD in [abs_ild, -abs_ild]:
                stim_key = (ABL, ILD)
                if stim_key not in animal_data:
                    continue
                empirical_signed.append(animal_data[stim_key]["empirical"])
                theoretical_signed.append(animal_data[stim_key]["theoretical"])
                n_trials_signed += animal_data[stim_key]["n_trials"]

            empirical_signed = np.asarray(empirical_signed, dtype=float)
            theoretical_signed = np.asarray(theoretical_signed, dtype=float)
            if empirical_signed.size > 0 and np.any(np.isfinite(empirical_signed)):
                plot_data[ABL][abs_ild]["empirical"].append(np.nanmean(empirical_signed, axis=0))
            if theoretical_signed.size > 0 and np.any(np.isfinite(theoretical_signed)):
                plot_data[ABL][abs_ild]["theoretical"].append(np.nanmean(theoretical_signed, axis=0))
            if n_trials_signed > 0:
                n_trials_by_condition[ABL][abs_ild].append(n_trials_signed)

rtd_plot_data = {
    "plot_data": {
        ABL: {
            abs_ild: {
                "empirical": list(plot_data[ABL][abs_ild]["empirical"]),
                "theoretical": list(plot_data[ABL][abs_ild]["theoretical"]),
            }
            for abs_ild in ABS_ILD_ARR
        }
        for ABL in ABL_ARR
    },
    "rt_bins": RT_BINS,
    "rt_bin_centers": RT_BIN_CENTERS,
    "ABL_arr": ABL_ARR,
    "abs_ild_arr": ABS_ILD_ARR,
    "animal_keys": batch_animal_pairs,
    "model": "NPL + alpha + ILD2 delay",
}
with open(RTD_OUTPUT_PKL, "wb") as handle:
    pickle.dump(rtd_plot_data, handle)
print(f"Saved {RTD_OUTPUT_PKL}")


# %%
fig, axes = plt.subplots(2, 5, figsize=(17.5, 6.8), sharex=True, sharey=True)
global_y_max = 0.0

for row_idx, ABL in enumerate(ABL_ARR):
    for col_idx, abs_ild in enumerate(ABS_ILD_ARR):
        ax = axes[row_idx, col_idx]
        empirical_curves = np.asarray(plot_data[ABL][abs_ild]["empirical"], dtype=float)
        theoretical_curves = np.asarray(plot_data[ABL][abs_ild]["theoretical"], dtype=float)
        if empirical_curves.size == 0:
            empirical_curves = np.full((1, len(RT_BIN_CENTERS)), np.nan)
        if theoretical_curves.size == 0:
            theoretical_curves = np.full((1, len(RT_BIN_CENTERS)), np.nan)

        empirical_mean = np.nanmean(empirical_curves, axis=0)
        empirical_sem = sem(empirical_curves, axis=0, nan_policy="omit")
        theoretical_mean = np.nanmean(theoretical_curves, axis=0)
        theoretical_sem = sem(theoretical_curves, axis=0, nan_policy="omit")
        theoretical_area = np.nansum(theoretical_mean * np.diff(RT_BINS))
        panel_y_max = np.nanmax(
            [
                np.nanmax(empirical_mean + empirical_sem),
                np.nanmax(theoretical_mean + theoretical_sem),
            ]
        )
        if np.isfinite(panel_y_max):
            global_y_max = max(global_y_max, panel_y_max)

        ax.plot(RT_BIN_CENTERS, empirical_mean, color="black", linewidth=1.8, label="Data")
        ax.fill_between(
            RT_BIN_CENTERS,
            empirical_mean - empirical_sem,
            empirical_mean + empirical_sem,
            color="black",
            alpha=0.16,
            linewidth=0,
        )

        ax.plot(RT_BIN_CENTERS, theoretical_mean, color="tab:red", linewidth=1.8, label="Model")
        ax.fill_between(
            RT_BIN_CENTERS,
            theoretical_mean - theoretical_sem,
            theoretical_mean + theoretical_sem,
            color="tab:red",
            alpha=0.18,
            linewidth=0,
        )

        if row_idx == 0:
            ax.set_title(f"|ILD| = {abs_ild:g}", fontsize=13)
        if col_idx == 0:
            ax.set_ylabel(f"ABL {ABL}\nDensity", fontsize=14)
        if row_idx == len(ABL_ARR) - 1:
            ax.set_xlabel("RT (s)", fontsize=13)

        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1.0])
        ax.tick_params(axis="both", which="major", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        print(f"Model RTD plotted normalized area: ABL={ABL}, |ILD|={abs_ild:g}, area={theoretical_area:.6f}")

axes[0, -1].legend(frameon=False, fontsize=12, loc="upper right")
for ax in axes.flat:
    ax.set_ylim(0, global_y_max * 1.05)
fig.suptitle(
    "NPL + alpha + ILD2 delay: model vs data RTDs",
    fontsize=18,
    y=0.985,
)
fig.subplots_adjust(left=0.07, right=0.99, bottom=0.11, top=0.88, hspace=0.28, wspace=0.18)
fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUTPUT_PDF, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved {OUTPUT_PNG}")
print(f"Saved {OUTPUT_PDF}")
print(f"Animals processed: {sum(bool(item) for item in all_animal_results)}")

for ABL in ABL_ARR:
    for abs_ild in ABS_ILD_ARR:
        n_animals = len(plot_data[ABL][abs_ild]["empirical"])
        median_trials = np.nanmedian(n_trials_by_condition[ABL][abs_ild])
        print(f"ABL={ABL}, |ILD|={abs_ild:g}: n_animals={n_animals}, median signed-pool trials={median_trials:.0f}")

# %%
