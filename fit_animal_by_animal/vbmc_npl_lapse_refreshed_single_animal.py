# %%
"""
Refreshed old VBMC NPL+lapse fit for one animal.

This keeps the old norm+lapse VBMC model intact, but fixes stale paths and stops
after saving fit summaries. The model has no alpha and fits one scalar t_E_aff.
"""

# %%
# =============================================================================
# Parameters
# =============================================================================
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pyvbmc import VBMC

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
RAW_BATCH_DIR = REPO_DIR / "raw_data" / "batch_csvs"
ABORT_RESULTS_DIR = REPO_DIR / "aborts_ipl_npl_time_fit_results"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "vbmc_npl_lapse_four_animal_rerun"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_DIR / "lapses"))

from vbmc_animal_wise_fit_utils import trapezoidal_logpdf
from time_vary_norm_utils import up_or_down_RTs_fit_fn, cum_pro_and_reactive_time_vary_fn


parser = argparse.ArgumentParser(description="Fit old VBMC NPL+lapse model for one animal with refreshed paths.")
parser.add_argument("--batch", required=True, help="Batch name, e.g. LED7")
parser.add_argument("--animal", required=True, type=int, help="Animal ID")
parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Output folder root")
parser.add_argument("--init-type", default="norm", choices=["vanilla", "norm"], help="Old initialization mode")
parser.add_argument("--is-stim-filtered", action="store_true", help="Keep only old hardcoded ABL/ILD grid")
parser.add_argument("--posterior-samples", type=int, default=100000, help="Posterior samples for summary CSV")
parser.add_argument("--max-fun-evals", type=int, default=200 * (2 + 8), help="VBMC max_fun_evals")
parser.add_argument("--seed", type=int, default=12345, help="Numpy random seed")
args = parser.parse_args()

batch_name = args.batch
animal = int(args.animal)
init_type = args.init_type
is_stim_filtered = bool(args.is_stim_filtered)
np.random.seed(args.seed)

output_root = Path(args.output_root).expanduser().resolve()
output_dir = output_root / f"{batch_name}_{animal}"
output_dir.mkdir(parents=True, exist_ok=True)

T_trunc = 0.15 if batch_name == "LED34_even" else 0.3
phi_params_obj = np.nan
is_norm = True
is_time_vary = False
K_max = 10
DO_RIGHT_TRUNCATE = True

PARAM_NAMES = [
    "rate_lambda",
    "T_0",
    "theta_E",
    "w",
    "t_E_aff",
    "del_go",
    "rate_norm_l",
    "lapse_prob",
    "lapse_prob_right",
]


# %%
# =============================================================================
# Data and fixed abort/NPL reference parameters
# =============================================================================
csv_path = RAW_BATCH_DIR / f"batch_{batch_name}_valid_and_aborts.csv"
if not csv_path.exists():
    raise FileNotFoundError(csv_path)
exp_df = pd.read_csv(csv_path)

df_valid_and_aborts = exp_df[
    (exp_df["success"].isin([1, -1])) | (exp_df["abort_event"] == 3)
].copy()
df_aborts = df_valid_and_aborts[df_valid_and_aborts["abort_event"] == 3]

df_all_trials_animal = df_valid_and_aborts[df_valid_and_aborts["animal"] == animal].copy()
df_aborts_animal = df_aborts[df_aborts["animal"] == animal].copy()
df_valid_animal = df_all_trials_animal[df_all_trials_animal["success"].isin([1, -1])].copy()

if is_stim_filtered:
    allowed_abls = [20, 40, 60]
    allowed_ilds = [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]
    df_valid_animal = df_valid_animal[
        (df_valid_animal["ABL"].isin(allowed_abls)) & (df_valid_animal["ILD"].isin(allowed_ilds))
    ].copy()

if DO_RIGHT_TRUNCATE:
    df_valid_animal = df_valid_animal[df_valid_animal["RTwrtStim"] < 1].copy()
    max_rt = 1.0
else:
    max_rt = float(df_valid_animal["RTwrtStim"].max())

if df_valid_animal.empty:
    raise RuntimeError(f"No valid fitting trials for {batch_name}/{animal}.")

reference_pkl = ABORT_RESULTS_DIR / f"results_{batch_name}_animal_{animal}.pkl"
if not reference_pkl.exists():
    raise FileNotFoundError(reference_pkl)

with open(reference_pkl, "rb") as f:
    fit_results_data = pickle.load(f)

abort_samples = fit_results_data["vbmc_aborts_results"]
V_A = float(np.mean(abort_samples["V_A_samples"]))
theta_A = float(np.mean(abort_samples["theta_A_samples"]))
t_A_aff = float(np.mean(abort_samples["t_A_aff_samp"]))

norm_reference = fit_results_data["vbmc_norm_tied_results"]
reference_summary = {}
reference_sample_keys = {
    "rate_lambda": "rate_lambda_samples",
    "T_0": "T_0_samples",
    "theta_E": "theta_E_samples",
    "w": "w_samples",
    "t_E_aff": "t_E_aff_samples",
    "del_go": "del_go_samples",
    "rate_norm_l": "rate_norm_l_samples",
}
for param_name, sample_key in reference_sample_keys.items():
    samples = np.asarray(norm_reference[sample_key], dtype=float)
    reference_summary[param_name] = {
        "mean": float(np.mean(samples)),
        "q025": float(np.percentile(samples, 2.5)),
        "q975": float(np.percentile(samples, 97.5)),
    }

print("=" * 80)
print(f"Old VBMC NPL+lapse refreshed fit: {batch_name}/{animal}")
print(f"CSV: {csv_path}")
print(f"Reference pkl: {reference_pkl}")
print(f"Output dir: {output_dir}")
print(f"T_trunc: {T_trunc:.3f} s")
print(f"Valid fitting trials after RT<1 truncation: {len(df_valid_animal)}")
print(f"Abort trials available: {len(df_aborts_animal)}")
print(f"Abort params: V_A={V_A:.6g}, theta_A={theta_A:.6g}, t_A_aff={t_A_aff * 1000:.3f} ms")
print("=" * 80)


# %%
# =============================================================================
# Old NPL+lapse likelihood and bounds
# =============================================================================
def compute_loglike_norm(row, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go, rate_norm_l, lapse_prob, lapse_prob_right):
    rt = row["TotalFixTime"]
    t_stim = row["intended_fix"]
    ILD = row["ILD"]
    ABL = row["ABL"]
    choice = row["choice"]
    lapse_rt_window = max_rt

    pdf = up_or_down_RTs_fit_fn(
        rt,
        choice,
        V_A,
        theta_A,
        t_A_aff,
        t_stim,
        ABL,
        ILD,
        rate_lambda,
        T_0,
        theta_E,
        Z_E,
        t_E_aff,
        del_go,
        phi_params_obj,
        rate_norm_l,
        is_norm,
        is_time_vary,
        K_max,
    )

    if DO_RIGHT_TRUNCATE:
        trunc_factor_p_joint = cum_pro_and_reactive_time_vary_fn(
            t_stim + 1,
            T_trunc,
            V_A,
            theta_A,
            t_A_aff,
            t_stim,
            ABL,
            ILD,
            rate_lambda,
            T_0,
            theta_E,
            Z_E,
            t_E_aff,
            phi_params_obj,
            rate_norm_l,
            is_norm,
            is_time_vary,
            K_max,
        ) - cum_pro_and_reactive_time_vary_fn(
            t_stim,
            T_trunc,
            V_A,
            theta_A,
            t_A_aff,
            t_stim,
            ABL,
            ILD,
            rate_lambda,
            T_0,
            theta_E,
            Z_E,
            t_E_aff,
            phi_params_obj,
            rate_norm_l,
            is_norm,
            is_time_vary,
            K_max,
        )
    else:
        trunc_factor_p_joint = 1 - cum_pro_and_reactive_time_vary_fn(
            t_stim,
            T_trunc,
            V_A,
            theta_A,
            t_A_aff,
            t_stim,
            ABL,
            ILD,
            rate_lambda,
            T_0,
            theta_E,
            Z_E,
            t_E_aff,
            phi_params_obj,
            rate_norm_l,
            is_norm,
            is_time_vary,
            K_max,
        )

    pdf /= trunc_factor_p_joint + 1e-20
    if choice == 1:
        lapse_choice_pdf = lapse_prob_right * (1 / lapse_rt_window)
    else:
        lapse_choice_pdf = (1 - lapse_prob_right) * (1 / lapse_rt_window)

    included_lapse_pdf = (1 - lapse_prob) * pdf + lapse_prob * lapse_choice_pdf
    included_lapse_pdf = max(included_lapse_pdf, 1e-50)
    if np.isnan(included_lapse_pdf):
        raise ValueError(f"nan pdf rt={rt}, t_stim={t_stim}")
    return np.log(included_lapse_pdf)


def vbmc_norm_tied_loglike_fn(params):
    rate_lambda, T_0, theta_E, w, t_E_aff, del_go, rate_norm_l, lapse_prob, lapse_prob_right = params
    Z_E = (w - 0.5) * 2 * theta_E
    all_loglike = Parallel(n_jobs=-5)(
        delayed(compute_loglike_norm)(
            row,
            rate_lambda,
            T_0,
            theta_E,
            Z_E,
            t_E_aff,
            del_go,
            rate_norm_l,
            lapse_prob,
            lapse_prob_right,
        )
        for _, row in df_valid_animal.iterrows()
    )
    return np.sum(all_loglike)


def vbmc_norm_tied_prior_fn(params):
    rate_lambda, T_0, theta_E, w, t_E_aff, del_go, rate_norm_l, lapse_prob, lapse_prob_right = params
    return (
        trapezoidal_logpdf(rate_lambda, norm_rate_lambda_bounds[0], norm_rate_lambda_plausible_bounds[0], norm_rate_lambda_plausible_bounds[1], norm_rate_lambda_bounds[1])
        + trapezoidal_logpdf(T_0, norm_T_0_bounds[0], norm_T_0_plausible_bounds[0], norm_T_0_plausible_bounds[1], norm_T_0_bounds[1])
        + trapezoidal_logpdf(theta_E, norm_theta_E_bounds[0], norm_theta_E_plausible_bounds[0], norm_theta_E_plausible_bounds[1], norm_theta_E_bounds[1])
        + trapezoidal_logpdf(w, norm_w_bounds[0], norm_w_plausible_bounds[0], norm_w_plausible_bounds[1], norm_w_bounds[1])
        + trapezoidal_logpdf(t_E_aff, norm_t_E_aff_bounds[0], norm_t_E_aff_plausible_bounds[0], norm_t_E_aff_plausible_bounds[1], norm_t_E_aff_bounds[1])
        + trapezoidal_logpdf(del_go, norm_del_go_bounds[0], norm_del_go_plausible_bounds[0], norm_del_go_plausible_bounds[1], norm_del_go_bounds[1])
        + trapezoidal_logpdf(rate_norm_l, norm_rate_norm_bounds[0], norm_rate_norm_plausible_bounds[0], norm_rate_norm_plausible_bounds[1], norm_rate_norm_bounds[1])
        + trapezoidal_logpdf(lapse_prob, norm_lapse_prob_bounds[0], norm_lapse_prob_plausible_bounds[0], norm_lapse_prob_plausible_bounds[1], norm_lapse_prob_bounds[1])
        + trapezoidal_logpdf(lapse_prob_right, norm_lapse_prob_right_bounds[0], norm_lapse_prob_right_plausible_bounds[0], norm_lapse_prob_right_plausible_bounds[1], norm_lapse_prob_right_bounds[1])
    )


def vbmc_norm_tied_joint_fn(params):
    return vbmc_norm_tied_prior_fn(params) + vbmc_norm_tied_loglike_fn(params)


# These are the "NEW bounds" from the old norm+lapse script.
norm_rate_lambda_bounds = [0.5, 5]
norm_T_0_bounds = [50e-3, 800e-3]
norm_theta_E_bounds = [1, 15]
norm_w_bounds = [0.3, 0.7]
norm_t_E_aff_bounds = [0.01, 0.2]
norm_del_go_bounds = [0, 0.2]
norm_rate_norm_bounds = [0, 2]
norm_lapse_prob_bounds = [1e-4, 0.2]
norm_lapse_prob_right_bounds = [0.001, 0.999]

norm_rate_lambda_plausible_bounds = [1, 3]
norm_T_0_plausible_bounds = [90e-3, 400e-3]
norm_theta_E_plausible_bounds = [1.5, 10]
norm_w_plausible_bounds = [0.4, 0.6]
norm_t_E_aff_plausible_bounds = [0.03, 0.09]
norm_del_go_plausible_bounds = [0.05, 0.15]
norm_rate_norm_plausible_bounds = [0.8, 0.99]
norm_lapse_prob_plausible_bounds = [1e-3, 0.1]
norm_lapse_prob_right_plausible_bounds = [0.4, 0.6]

norm_tied_lb = np.array([
    norm_rate_lambda_bounds[0],
    norm_T_0_bounds[0],
    norm_theta_E_bounds[0],
    norm_w_bounds[0],
    norm_t_E_aff_bounds[0],
    norm_del_go_bounds[0],
    norm_rate_norm_bounds[0],
    norm_lapse_prob_bounds[0],
    norm_lapse_prob_right_bounds[0],
])
norm_tied_ub = np.array([
    norm_rate_lambda_bounds[1],
    norm_T_0_bounds[1],
    norm_theta_E_bounds[1],
    norm_w_bounds[1],
    norm_t_E_aff_bounds[1],
    norm_del_go_bounds[1],
    norm_rate_norm_bounds[1],
    norm_lapse_prob_bounds[1],
    norm_lapse_prob_right_bounds[1],
])
norm_plb = np.array([
    norm_rate_lambda_plausible_bounds[0],
    norm_T_0_plausible_bounds[0],
    norm_theta_E_plausible_bounds[0],
    norm_w_plausible_bounds[0],
    norm_t_E_aff_plausible_bounds[0],
    norm_del_go_plausible_bounds[0],
    norm_rate_norm_plausible_bounds[0],
    norm_lapse_prob_plausible_bounds[0],
    norm_lapse_prob_right_plausible_bounds[0],
])
norm_pub = np.array([
    norm_rate_lambda_plausible_bounds[1],
    norm_T_0_plausible_bounds[1],
    norm_theta_E_plausible_bounds[1],
    norm_w_plausible_bounds[1],
    norm_t_E_aff_plausible_bounds[1],
    norm_del_go_plausible_bounds[1],
    norm_rate_norm_plausible_bounds[1],
    norm_lapse_prob_plausible_bounds[1],
    norm_lapse_prob_right_plausible_bounds[1],
])


# %%
# =============================================================================
# Fit
# =============================================================================
if init_type == "vanilla":
    x_0 = np.array([0.173793, 1.958508e-3, 14.194689, 0.502231, 87.604997e-3, 169.585535e-3, 0.00001, 0.02, 0.5])
else:
    x_0 = np.array([1.8, 150e-3, 5, 0.51, 0.071, 0.13, 0.9, 0.02, 0.5])

initial_joint = float(vbmc_norm_tied_joint_fn(x_0))
print("Initial values:")
for name, value in zip(PARAM_NAMES, x_0):
    printed = value * 1000 if name in {"T_0", "t_E_aff", "del_go"} else value
    unit = " ms" if name in {"T_0", "t_E_aff", "del_go"} else ""
    print(f"  {name:<16} = {printed:.6g}{unit}")
print(f"Initial joint: {initial_joint:.6f}")
print(f"Running VBMC with max_fun_evals={args.max_fun_evals}...")

vbmc = VBMC(
    vbmc_norm_tied_joint_fn,
    x_0,
    norm_tied_lb,
    norm_tied_ub,
    norm_plb,
    norm_pub,
    options={"display": "on", "max_fun_evals": int(args.max_fun_evals)},
)
vp, results = vbmc.optimize()

stim_filter_suffix = "_stim_filtered" if is_stim_filtered else ""
vbmc_pkl_path = output_dir / f"vbmc_norm_tied_results_batch_{batch_name}_animal_{animal}_lapses_truncate_1s_{init_type}{stim_filter_suffix}.pkl"
vbmc.save(str(vbmc_pkl_path), overwrite=True)
print(f"Saved VBMC object: {vbmc_pkl_path}")


# %%
# =============================================================================
# Posterior summary
# =============================================================================
vp_samples, log_weights = vp.sample(int(args.posterior_samples))
posterior_npz_path = output_dir / "vbmc_norm_lapse_posterior_samples.npz"
np.savez_compressed(posterior_npz_path, samples=vp_samples, log_weights=log_weights, param_names=np.array(PARAM_NAMES, dtype=object))

summary_rows = []
for idx, name in enumerate(PARAM_NAMES):
    samples = np.asarray(vp_samples[:, idx], dtype=float)
    summary_rows.append(
        {
            "parameter": name,
            "mean": float(np.mean(samples)),
            "sd": float(np.std(samples)),
            "q025": float(np.percentile(samples, 2.5)),
            "q500": float(np.percentile(samples, 50)),
            "q975": float(np.percentile(samples, 97.5)),
            "n_samples": int(len(samples)),
        }
    )

summary_df = pd.DataFrame(summary_rows)
summary_csv = output_dir / "vbmc_norm_lapse_posterior_summary.csv"
summary_df.to_csv(summary_csv, index=False)

reference_rows = []
for name, stats in reference_summary.items():
    reference_rows.append({"parameter": name, **stats})
reference_csv = output_dir / "reference_npl_no_lapse_summary.csv"
pd.DataFrame(reference_rows).to_csv(reference_csv, index=False)

iteration_history = getattr(vbmc, "iteration_history", {})
last_elbo = np.nan
last_elbo_sd = np.nan
last_stable = None
last_iter = np.nan
if isinstance(iteration_history, dict):
    if "elbo" in iteration_history and len(iteration_history["elbo"]):
        last_elbo = float(iteration_history["elbo"][-1])
    if "elbo_sd" in iteration_history and len(iteration_history["elbo_sd"]):
        last_elbo_sd = float(iteration_history["elbo_sd"][-1])
    if "stable" in iteration_history and len(iteration_history["stable"]):
        last_stable = bool(iteration_history["stable"][-1])
    if "iter" in iteration_history and len(iteration_history["iter"]):
        last_iter = int(iteration_history["iter"][-1])

run_summary = {
    "batch_name": batch_name,
    "animal": animal,
    "init_type": init_type,
    "csv_path": str(csv_path),
    "reference_pkl": str(reference_pkl),
    "output_dir": str(output_dir),
    "vbmc_pkl_path": str(vbmc_pkl_path),
    "posterior_summary_csv": str(summary_csv),
    "posterior_npz_path": str(posterior_npz_path),
    "n_valid_fit_trials": int(len(df_valid_animal)),
    "n_abort_trials_available": int(len(df_aborts_animal)),
    "T_trunc": float(T_trunc),
    "max_rt": float(max_rt),
    "max_fun_evals": int(args.max_fun_evals),
    "posterior_samples": int(args.posterior_samples),
    "initial_joint": initial_joint,
    "final_elbo": last_elbo,
    "final_elbo_sd": last_elbo_sd,
    "final_stable": last_stable,
    "final_iter": last_iter,
}
run_summary_csv = output_dir / "vbmc_norm_lapse_run_summary.csv"
pd.DataFrame([run_summary]).to_csv(run_summary_csv, index=False)

print("\nPosterior means:")
for row in summary_rows:
    value = row["mean"] * 1000 if row["parameter"] in {"T_0", "t_E_aff", "del_go"} else row["mean"]
    unit = " ms" if row["parameter"] in {"T_0", "t_E_aff", "del_go"} else ""
    print(f"  {row['parameter']:<16} = {value:.6g}{unit}")
print(f"Saved posterior summary: {summary_csv}")
print(f"Saved run summary: {run_summary_csv}")
print(f"final_stable={last_stable}, final_elbo={last_elbo}, final_elbo_sd={last_elbo_sd}, final_iter={last_iter}")

# %%
