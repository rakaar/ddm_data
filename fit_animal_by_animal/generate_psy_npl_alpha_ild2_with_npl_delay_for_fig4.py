# %%
import os
import pickle
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit

from animal_wise_plotting_utils import calculate_theoretical_curves
from time_vary_norm_alpha_utils import (
    cum_pro_and_reactive_time_vary_alpha_fn,
    rho_A_t_fn,
    up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_alpha_fn,
)


# %%
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

DESIRED_BATCHES = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]
ABL_arr = [20, 40, 60]
ILD_arr = [-16.0, -8.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0, 16.0]
K_max = 10
T_PTS = np.arange(-2, 2, 0.001)
N_THEORY = int(1e3)
N_JOBS = max(1, min(30, (os.cpu_count() or 2) - 1))

ILD2_RESULTS_DIR = os.path.join(REPO_DIR, "NPL_alpha_ILD2_fit_results", "result_pkls")
OUTPUT_DIR = os.path.join(REPO_DIR, "NPL_alpha_ILD2_fit_results", "figure_4_diagnostics")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PSY_OUTPUT_PKL = os.path.join(OUTPUT_DIR, "npl_alpha_ild2_with_npl_delay_psy_fig4_data.pkl")
SLOPES_OUTPUT_PKL = os.path.join(OUTPUT_DIR, "npl_alpha_ild2_with_npl_delay_slopes_fig4_data.pkl")
THEORETICAL_PSY_OUTPUT_PKL = os.path.join(
    OUTPUT_DIR,
    "theoretical_psychometric_data_npl_alpha_ild2_with_npl_delay.pkl",
)
EMPIRICAL_PSY_OUTPUT_PKL = os.path.join(OUTPUT_DIR, "empirical_psychometric_data_for_npl_alpha_ild2.pkl")

ILD2_RESULT_KEY = "vbmc_norm_alpha_ild2_delay_tied_results"
NPL_RESULT_KEY = "vbmc_norm_tied_results"
ILD2_RESULT_RE = re.compile(r"^results_(?P<batch>.+)_animal_(?P<animal>\d+)_NORM_ALPHA_ILD2_DELAY_FROM_ABORTS\.pkl$")
NPL_RESULT_RE = re.compile(r"^results_(?P<batch>.+)_animal_(?P<animal>\d+)\.pkl$")
BATCH_T_TRUNC = {"LED34_even": 0.15}
DEFAULT_T_TRUNC = 0.3


# %%
def fit_psychometric_sigmoid(ild_values, right_choice_probs):
    def sigmoid(x, upper, lower, x0, k):
        return lower + (upper - lower) / (1 + np.exp(-k * (x - x0)))

    ild_values = np.asarray(ild_values, dtype=float)
    right_choice_probs = np.asarray(right_choice_probs, dtype=float)
    valid_idx = np.isfinite(ild_values) & np.isfinite(right_choice_probs)
    if np.sum(valid_idx) < 4:
        return None
    try:
        popt, _ = curve_fit(
            sigmoid,
            ild_values[valid_idx],
            right_choice_probs[valid_idx],
            p0=[1.0, 0.0, 0.0, 1.0],
            bounds=([0, 0, -np.inf, 0], [1, 1, np.inf, np.inf]),
        )
        return {"params": popt}
    except Exception as exc:
        print(f"Could not fit sigmoid: {exc}")
        return None


def _create_innermost_dict():
    return {"empirical": [], "theoretical": []}


def _create_inner_defaultdict():
    return defaultdict(_create_innermost_dict)


def get_result_paths():
    ild2_paths = {}
    for filename in sorted(os.listdir(ILD2_RESULTS_DIR)):
        match = ILD2_RESULT_RE.match(filename)
        if not match:
            continue
        key = (match.group("batch"), int(match.group("animal")))
        ild2_paths[key] = os.path.join(ILD2_RESULTS_DIR, filename)

    npl_paths = {}
    for filename in sorted(os.listdir(SCRIPT_DIR)):
        match = NPL_RESULT_RE.match(filename)
        if not match:
            continue
        key = (match.group("batch"), int(match.group("animal")))
        npl_paths[key] = os.path.join(SCRIPT_DIR, filename)

    return ild2_paths, npl_paths


def load_hybrid_posterior_mean_params(batch_name, animal_id, ild2_paths, npl_paths):
    key = (batch_name, int(animal_id))
    with open(ild2_paths[key], "rb") as handle:
        ild2_fit_results = pickle.load(handle)
    with open(npl_paths[key], "rb") as handle:
        npl_fit_results = pickle.load(handle)

    abort_samples = ild2_fit_results["vbmc_aborts_results"]
    ild2_samples = ild2_fit_results[ILD2_RESULT_KEY]
    npl_samples = npl_fit_results[NPL_RESULT_KEY]

    abort_params = {
        "V_A": float(np.mean(abort_samples["V_A_samples"])),
        "theta_A": float(np.mean(abort_samples["theta_A_samples"])),
        "t_A_aff": float(np.mean(abort_samples["t_A_aff_samp"])),
    }
    tied_params = {
        "rate_lambda": float(np.mean(ild2_samples["rate_lambda_samples"])),
        "T_0": float(np.mean(ild2_samples["T_0_samples"])),
        "theta_E": float(np.mean(ild2_samples["theta_E_samples"])),
        "w": float(np.mean(ild2_samples["w_samples"])),
        "t_E_aff": float(np.mean(npl_samples["t_E_aff_samples"])),
        "del_go": float(np.mean(ild2_samples["del_go_samples"])),
        "rate_norm_l": float(np.mean(ild2_samples["rate_norm_l_samples"])),
        "alpha": float(np.mean(ild2_samples["alpha_samples"])),
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


def get_theoretical_rtd_up_down(P_A_mean, C_A_mean, t_stim_samples, abort_params, tied_params, ABL, ILD, batch_name):
    phi_params_obj = np.nan
    is_norm = True
    is_time_vary = False
    T_trunc = BATCH_T_TRUNC.get(batch_name, DEFAULT_T_TRUNC)
    t_E_aff = tied_params["t_E_aff"]
    Z_E = (tied_params["w"] - 0.5) * 2 * tied_params["theta_E"]

    trunc_fac_samples = np.zeros(len(t_stim_samples))
    for idx, t_stim in enumerate(t_stim_samples):
        trunc_fac_samples[idx] = (
            cum_pro_and_reactive_time_vary_alpha_fn(
                t_stim + 1,
                T_trunc,
                abort_params["V_A"],
                abort_params["theta_A"],
                abort_params["t_A_aff"],
                t_stim,
                ABL,
                ILD,
                tied_params["rate_lambda"],
                tied_params["T_0"],
                tied_params["theta_E"],
                Z_E,
                t_E_aff,
                phi_params_obj,
                tied_params["rate_norm_l"],
                tied_params["alpha"],
                is_norm,
                is_time_vary,
                K_max,
            )
            - cum_pro_and_reactive_time_vary_alpha_fn(
                t_stim,
                T_trunc,
                abort_params["V_A"],
                abort_params["theta_A"],
                abort_params["t_A_aff"],
                t_stim,
                ABL,
                ILD,
                tied_params["rate_lambda"],
                tied_params["T_0"],
                tied_params["theta_E"],
                Z_E,
                t_E_aff,
                phi_params_obj,
                tied_params["rate_norm_l"],
                tied_params["alpha"],
                is_norm,
                is_time_vary,
                K_max,
            )
            + 1e-10
        )
    trunc_factor = np.mean(trunc_fac_samples)

    up_mean = np.array([
        up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_alpha_fn(
            t,
            1,
            P_A_mean[i],
            C_A_mean[i],
            ABL,
            ILD,
            tied_params["rate_lambda"],
            tied_params["T_0"],
            tied_params["theta_E"],
            Z_E,
            t_E_aff,
            tied_params["del_go"],
            phi_params_obj,
            tied_params["rate_norm_l"],
            tied_params["alpha"],
            is_norm,
            is_time_vary,
            K_max,
        )
        for i, t in enumerate(T_PTS)
    ])
    down_mean = np.array([
        up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_alpha_fn(
            t,
            -1,
            P_A_mean[i],
            C_A_mean[i],
            ABL,
            ILD,
            tied_params["rate_lambda"],
            tied_params["T_0"],
            tied_params["theta_E"],
            Z_E,
            t_E_aff,
            tied_params["del_go"],
            phi_params_obj,
            tied_params["rate_norm_l"],
            tied_params["alpha"],
            is_norm,
            is_time_vary,
            K_max,
        )
        for i, t in enumerate(T_PTS)
    ])

    mask_0_1 = (T_PTS >= 0) & (T_PTS <= 1)
    return T_PTS[mask_0_1], up_mean[mask_0_1] / trunc_factor, down_mean[mask_0_1] / trunc_factor


def get_empirical_psychometric_data(batch_name, animal_id, ABL):
    df = pd.read_csv(os.path.join("batch_csvs", f"batch_{batch_name}_valid_and_aborts.csv"))
    df = df[
        (df["animal"] == int(animal_id))
        & (df["ABL"] == ABL)
        & (df["success"].isin([1, -1]))
        & (df["RTwrtStim"] <= 1)
    ]
    if df.empty:
        return None

    ild_values = np.array(sorted(df["ILD"].unique()), dtype=float)
    right_choice_probs = np.array([np.mean(df[df["ILD"] == ild]["choice"] == 1) for ild in ild_values], dtype=float)
    return {"ild_values": ild_values, "right_choice_probs": right_choice_probs}


def process_animal(batch_animal_pair, ild2_paths, npl_paths):
    batch_name, animal_id = batch_animal_pair
    print(f"Processing hybrid psychometric data for {batch_name}, animal {animal_id}")
    abort_params, tied_params = load_hybrid_posterior_mean_params(batch_name, animal_id, ild2_paths, npl_paths)
    print(f"  NPL t_E_aff used = {1e3 * tied_params['t_E_aff']:.3f} ms")
    P_A_mean, C_A_mean, t_stim_samples = get_p_a_c_a(batch_name, animal_id, abort_params)

    empirical_animal = {}
    theoretical_animal = {}
    for ABL in ABL_arr:
        empirical = get_empirical_psychometric_data(batch_name, animal_id, ABL)
        if empirical is not None:
            empirical_animal[ABL] = {
                "empirical": empirical,
                "fit": fit_psychometric_sigmoid(empirical["ild_values"], empirical["right_choice_probs"]),
            }

        theory_probs = []
        for ILD in ILD_arr:
            try:
                t_pts_0_1, up_mean, down_mean = get_theoretical_rtd_up_down(
                    P_A_mean, C_A_mean, t_stim_samples, abort_params, tied_params, ABL, ILD, batch_name
                )
                up_area = trapezoid(up_mean, t_pts_0_1)
                down_area = trapezoid(down_mean, t_pts_0_1)
                theory_probs.append(up_area / (up_area + down_area))
            except Exception as exc:
                print(f"  failed hybrid theory {batch_name}/{animal_id} ABL={ABL} ILD={ILD}: {exc}")
                theory_probs.append(np.nan)

        theoretical = {"ild_values": np.array(ILD_arr, dtype=float), "right_choice_probs": np.array(theory_probs)}
        theoretical_animal[ABL] = {
            "theoretical": theoretical,
            "fit": fit_psychometric_sigmoid(theoretical["ild_values"], theoretical["right_choice_probs"]),
        }

    return batch_animal_pair, empirical_animal, theoretical_animal


# %%
ild2_paths, npl_paths = get_result_paths()
batch_files = [os.path.join("batch_csvs", f"batch_{batch_name}_valid_and_aborts.csv") for batch_name in DESIRED_BATCHES]
merged_data = pd.concat([pd.read_csv(path) for path in batch_files if os.path.exists(path)], ignore_index=True)
merged_valid = merged_data[merged_data["success"].isin([1, -1])].copy()
batch_animal_pairs = sorted(
    [(batch, int(animal)) for batch, animal in merged_valid[["batch_name", "animal"]].drop_duplicates().values]
)
batch_animal_pairs = [pair for pair in batch_animal_pairs if pair in ild2_paths and pair in npl_paths]

print(f"Found {len(ild2_paths)} ILD2 result pickles")
print(f"Found {len(npl_paths)} NPL result pickles")
print(f"Using {len(batch_animal_pairs)} matched animals for hybrid psychometric/slopes")
print(f"Running with {N_JOBS} jobs")

results = Parallel(n_jobs=N_JOBS, verbose=10)(
    delayed(process_animal)(pair, ild2_paths, npl_paths) for pair in batch_animal_pairs
)

psychometric_data = {}
theoretical_psychometric_data = {}
for pair, empirical_animal, theoretical_animal in results:
    if empirical_animal:
        psychometric_data[pair] = empirical_animal
    if theoretical_animal:
        theoretical_psychometric_data[pair] = theoretical_animal


# %%
animal_keys = [pair for pair in batch_animal_pairs if pair in psychometric_data and pair in theoretical_psychometric_data]
theory_agg = {}
empirical_agg = {}
for ABL in ABL_arr:
    theory_agg[ABL] = np.full((len(animal_keys), len(ILD_arr)), np.nan)
    empirical_agg[ABL] = np.full((len(animal_keys), len(ILD_arr)), np.nan)

for idx, key in enumerate(animal_keys):
    for ABL in ABL_arr:
        if ABL in theoretical_psychometric_data[key]:
            theory_agg[ABL][idx] = theoretical_psychometric_data[key][ABL]["theoretical"]["right_choice_probs"]
        if ABL in psychometric_data[key]:
            empirical = psychometric_data[key][ABL]["empirical"]
            for ild_idx, ILD in enumerate(ILD_arr):
                matches = np.where(empirical["ild_values"] == ILD)[0]
                if len(matches) > 0:
                    empirical_agg[ABL][idx, ild_idx] = empirical["right_choice_probs"][matches[0]]

psy_plot_data = {
    "empirical_agg": empirical_agg,
    "theory_agg": theory_agg,
    "ILD_arr": ILD_arr,
    "animal_keys": animal_keys,
    "MODEL_TYPE": "npl_alpha_ild2_with_npl_delay",
}
with open(PSY_OUTPUT_PKL, "wb") as handle:
    pickle.dump(psy_plot_data, handle)
with open(THEORETICAL_PSY_OUTPUT_PKL, "wb") as handle:
    pickle.dump(theoretical_psychometric_data, handle)
with open(EMPIRICAL_PSY_OUTPUT_PKL, "wb") as handle:
    pickle.dump(psychometric_data, handle)


# %%
def extract_slopes(data_dict):
    slopes = {}
    for batch_animal, abl_dict in data_dict.items():
        slopes[batch_animal] = {}
        for ABL in ABL_arr:
            fit = abl_dict.get(ABL, {}).get("fit")
            if fit is not None and "params" in fit:
                slopes[batch_animal][ABL] = float(fit["params"][3])
            else:
                slopes[batch_animal][ABL] = np.nan
    return slopes


slopes_data = extract_slopes(psychometric_data)
slopes_hybrid = extract_slopes(theoretical_psychometric_data)
common_pairs = sorted(set(slopes_data) & set(slopes_hybrid), key=lambda item: (DESIRED_BATCHES.index(item[0]), item[1]))
common_pairs = [pair for pair in common_pairs if str(pair[1]) != "41"]

avg_slope_data = {pair: np.nanmean([slopes_data[pair][ABL] for ABL in ABL_arr]) for pair in common_pairs}
common_pairs_sorted = sorted(common_pairs, key=lambda pair: avg_slope_data[pair])
data_means = np.array([np.nanmean([slopes_data[pair][ABL] for ABL in ABL_arr]) for pair in common_pairs_sorted])
hybrid_means = np.array([np.nanmean([slopes_hybrid[pair][ABL] for ABL in ABL_arr]) for pair in common_pairs_sorted])

slopes_plot_data = {
    "slopes_data": slopes_data,
    "slopes_npl_alpha_ild2_with_npl_delay": slopes_hybrid,
    "common_pairs_sorted": common_pairs_sorted,
    "data_means": data_means,
    "npl_alpha_ild2_with_npl_delay_means": hybrid_means,
    "norm_means": hybrid_means,
    "ABL_arr": ABL_arr,
    "MODEL_TYPE": "npl_alpha_ild2_with_npl_delay",
}
with open(SLOPES_OUTPUT_PKL, "wb") as handle:
    pickle.dump(slopes_plot_data, handle)

print(f"Saved {PSY_OUTPUT_PKL}")
print(f"Saved {SLOPES_OUTPUT_PKL}")
print(f"Saved {THEORETICAL_PSY_OUTPUT_PKL}")
print(f"Matched animals in final psychometric agg: {len(animal_keys)}")
print(f"Matched animals in slope scatter: {len(common_pairs_sorted)}")

# %%
