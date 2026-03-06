# %%
"""
Compare aggregate LED-OFF normalized-tied fits between:
1) the 150 ms post-stimulus censoring fit
2) the 130 ms right-truncation fit

This script:
- loads both result pickles
- compares posterior summaries and loaded proactive parameters
- rebuilds the LED-OFF valid-trial datasets used by each fit
- cross-evaluates each posterior mean under both likelihood definitions
"""

# %%
from pathlib import Path
import os
import pickle
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ANIMAL_WISE_DIR = REPO_ROOT / "fit_animal_by_animal"
if str(ANIMAL_WISE_DIR) not in sys.path:
    sys.path.insert(0, str(ANIMAL_WISE_DIR))

from proactive_plus_lapse_plus_reactive_uitls import (
    trial_logpdf_proactive_lapse_only_no_trunc,
    trial_logpdf_proactive_lapse_only_no_trunc_right_truncated,
)


# %%
############ Parameters (edit here) ############
batch_name = "LED7"
session_type = 7
training_level = 16
allowed_repeat_trials = [0, 2]

max_rtwrtstim_for_fit = 1.0
censor_rt_wrt_stim_s = 0.150
truncate_rt_wrt_stim_s = 0.130

n_jobs = int(os.getenv("FIT_COMPARE_N_JOBS", "30"))
random_seed = 12345
max_rows_env = os.getenv("FIT_COMPARE_MAX_ROWS")
max_rows_for_loglike_eval = None if max_rows_env is None else int(max_rows_env)

is_norm = True
is_time_vary = False
phi_params_obj = np.nan
K_max = 10

led_data_csv_path = REPO_ROOT / "out_LED.csv"
censor_results_pkl_path = (
    SCRIPT_DIR
    / "norm_only_led_off_from_loaded_proactive"
    / f"results_norm_tied_batch_{batch_name}_aggregate_ledoff_1_proactive_loaded.pkl"
)
truncate_results_pkl_path = (
    SCRIPT_DIR
    / "norm_only_led_off_from_loaded_proactive_truncate_NOT_censor"
    / (
        f"results_norm_tied_batch_{batch_name}_aggregate_ledoff_1_"
        "proactive_loaded_truncate_NOT_censor.pkl"
    )
)

output_dir = SCRIPT_DIR / "censor_vs_truncate_fit_compare"
output_dir.mkdir(parents=True, exist_ok=True)

param_csv_path = output_dir / f"param_comparison_batch_{batch_name}.csv"
objective_csv_path = output_dir / f"objective_comparison_batch_{batch_name}.csv"
payload_path = output_dir / f"comparison_payload_batch_{batch_name}.pkl"


# %%
############ Load fit payloads ############
def load_fit_payload(path, label):
    if not path.exists():
        raise FileNotFoundError(f"Could not find {label} results pickle: {path}")
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if "vbmc_norm_tied_results" not in payload:
        raise KeyError(f"Missing 'vbmc_norm_tied_results' in {label} payload.")
    if "loaded_proactive_params" not in payload:
        raise KeyError(f"Missing 'loaded_proactive_params' in {label} payload.")
    return payload


censor_payload = load_fit_payload(censor_results_pkl_path, "censor")
truncate_payload = load_fit_payload(truncate_results_pkl_path, "truncate")

fit_payloads = {
    "censor": censor_payload,
    "truncate": truncate_payload,
}

print("Loaded fit payloads:")
print(f"  censor:   {censor_results_pkl_path}")
print(f"  truncate: {truncate_results_pkl_path}")


# %%
############ Parameter summaries ############
def get_tied_param_summary(payload, fit_label):
    vbmc_results = payload["vbmc_norm_tied_results"]
    key_map = {
        "rate_lambda": "rate_lambda_samples",
        "T_0": "T_0_samples",
        "theta_E": "theta_E_samples",
        "w": "w_samples",
        "t_E_aff": "t_E_aff_samples",
        "del_go": "del_go_samples",
        "rate_norm_l": "rate_norm_l_samples",
    }
    rows = []
    for param_name, sample_key in key_map.items():
        arr = np.asarray(vbmc_results[sample_key], dtype=np.float64)
        rows.append(
            {
                "group": "normalized_tied",
                "parameter": param_name,
                "fit_label": fit_label,
                "mean": float(np.mean(arr)),
                "sd": float(np.std(arr)),
                "sample_count": int(arr.size),
            }
        )
    return pd.DataFrame(rows)


def get_proactive_param_summary(payload, fit_label):
    loaded_pro = payload["loaded_proactive_params"]
    rows = []
    for param_name in [
        "V_A_base",
        "theta_A",
        "del_a_minus_del_LED",
        "del_m_plus_del_LED",
        "lapse_prob",
        "beta_lapse",
        "derived_t_A_aff_for_tied_fit",
    ]:
        rows.append(
            {
                "group": "loaded_proactive",
                "parameter": param_name,
                "fit_label": fit_label,
                "mean": float(loaded_pro[param_name]),
                "sd": np.nan,
                "sample_count": np.nan,
            }
        )
    return pd.DataFrame(rows)


tied_summary_df = pd.concat(
    [get_tied_param_summary(payload, fit_label) for fit_label, payload in fit_payloads.items()],
    ignore_index=True,
)
proactive_summary_df = pd.concat(
    [get_proactive_param_summary(payload, fit_label) for fit_label, payload in fit_payloads.items()],
    ignore_index=True,
)
param_summary_df = pd.concat([tied_summary_df, proactive_summary_df], ignore_index=True)


def make_wide_comparison(df):
    wide = df.pivot(index=["group", "parameter"], columns="fit_label", values="mean").reset_index()
    if {"censor", "truncate"}.issubset(wide.columns):
        wide["truncate_minus_censor"] = wide["truncate"] - wide["censor"]
        wide["abs_diff"] = np.abs(wide["truncate_minus_censor"])
    return wide


tied_wide_df = tied_summary_df.pivot(index="parameter", columns="fit_label", values=["mean", "sd"])
tied_comparison_rows = []
for param_name in tied_summary_df["parameter"].unique():
    censor_mean = float(tied_wide_df.loc[param_name, ("mean", "censor")])
    truncate_mean = float(tied_wide_df.loc[param_name, ("mean", "truncate")])
    censor_sd = float(tied_wide_df.loc[param_name, ("sd", "censor")])
    truncate_sd = float(tied_wide_df.loc[param_name, ("sd", "truncate")])
    avg_sd = 0.5 * (censor_sd + truncate_sd)
    tied_comparison_rows.append(
        {
            "group": "normalized_tied",
            "parameter": param_name,
            "censor_mean": censor_mean,
            "censor_sd": censor_sd,
            "truncate_mean": truncate_mean,
            "truncate_sd": truncate_sd,
            "truncate_minus_censor": truncate_mean - censor_mean,
            "abs_diff": abs(truncate_mean - censor_mean),
            "abs_diff_over_avg_sd": abs(truncate_mean - censor_mean) / avg_sd if avg_sd > 0 else np.nan,
        }
    )
tied_comparison_df = pd.DataFrame(tied_comparison_rows)
proactive_comparison_df = make_wide_comparison(proactive_summary_df)

print("\nNormalized-tied posterior comparison:")
print(tied_comparison_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

print("\nLoaded proactive parameter comparison:")
print(proactive_comparison_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


# %%
############ Fit metadata + sample equality checks ############
sample_key_map = {
    "rate_lambda": "rate_lambda_samples",
    "T_0": "T_0_samples",
    "theta_E": "theta_E_samples",
    "w": "w_samples",
    "t_E_aff": "t_E_aff_samples",
    "del_go": "del_go_samples",
    "rate_norm_l": "rate_norm_l_samples",
}

sample_equality_rows = []
for param_name, sample_key in sample_key_map.items():
    censor_arr = np.asarray(censor_payload["vbmc_norm_tied_results"][sample_key], dtype=np.float64)
    truncate_arr = np.asarray(truncate_payload["vbmc_norm_tied_results"][sample_key], dtype=np.float64)
    sample_equality_rows.append(
        {
            "parameter": param_name,
            "exact_array_equal": bool(np.array_equal(censor_arr, truncate_arr)),
            "max_abs_sample_diff": float(np.max(np.abs(censor_arr - truncate_arr))),
        }
    )
sample_equality_df = pd.DataFrame(sample_equality_rows)

metadata_rows = []
for fit_label, payload in fit_payloads.items():
    fit_config = payload.get("fit_config", {})
    vbmc_results = payload["vbmc_norm_tied_results"]
    metadata_rows.append(
        {
            "fit_label": fit_label,
            "likelihood_mode": fit_config.get("likelihood_mode"),
            "elbo": vbmc_results.get("elbo"),
            "elbo_sd": vbmc_results.get("elbo_sd"),
            "stored_loglike": vbmc_results.get("loglike"),
            "fit_trial_counts": payload.get("fit_trial_counts"),
        }
    )
metadata_df = pd.DataFrame(metadata_rows)

print("\nFit metadata comparison:")
print(metadata_df.to_string(index=False))

print("\nExact posterior sample equality checks:")
print(sample_equality_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


# %%
############ Rebuild LED-OFF valid-trial datasets ############
if not led_data_csv_path.exists():
    raise FileNotFoundError(f"Could not find data CSV: {led_data_csv_path}")

exp_df = pd.read_csv(led_data_csv_path)
exp_df["RTwrtStim"] = exp_df["timed_fix"] - exp_df["intended_fix"]
exp_df = exp_df.rename(columns={"timed_fix": "TotalFixTime"})
exp_df = exp_df[exp_df["RTwrtStim"] < 1]
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
missing_choice = exp_df_led_off["choice"].isna()
if missing_choice.any():
    rng = np.random.default_rng(random_seed)
    exp_df_led_off.loc[missing_choice, "choice"] = rng.choice(
        [1, -1], size=int(missing_choice.sum())
    )
exp_df_led_off["choice"] = exp_df_led_off["choice"].astype(int)

df_valid_all = exp_df_led_off[exp_df_led_off["success"].isin([1, -1])].copy()
df_valid_censor = df_valid_all[df_valid_all["RTwrtStim"] < max_rtwrtstim_for_fit].copy()
df_valid_truncate = df_valid_censor[
    (df_valid_censor["RTwrtStim"] > 0) & (df_valid_censor["RTwrtStim"] <= truncate_rt_wrt_stim_s)
].copy()

dataset_rows = [
    {"dataset": "valid_full_for_censor_fit", "n_trials": int(len(df_valid_censor))},
    {"dataset": "valid_0_to_130ms_for_truncate_fit", "n_trials": int(len(df_valid_truncate))},
]
dataset_df = pd.DataFrame(dataset_rows)

print("\nRebuilt training dataset counts:")
print(dataset_df.to_string(index=False))

truncate_counts_saved = truncate_payload.get("fit_trial_counts", {})
if truncate_counts_saved:
    print("\nSaved truncation fit trial counts:")
    print(truncate_counts_saved)


# %%
############ Likelihood comparison helpers ############
def get_full_parameter_set(payload):
    vbmc_results = payload["vbmc_norm_tied_results"]
    loaded_pro = payload["loaded_proactive_params"]

    rate_lambda = float(np.mean(vbmc_results["rate_lambda_samples"]))
    T_0 = float(np.mean(vbmc_results["T_0_samples"]))
    theta_E = float(np.mean(vbmc_results["theta_E_samples"]))
    w = float(np.mean(vbmc_results["w_samples"]))
    t_E_aff = float(np.mean(vbmc_results["t_E_aff_samples"]))
    del_go = float(np.mean(vbmc_results["del_go_samples"]))
    rate_norm_l = float(np.mean(vbmc_results["rate_norm_l_samples"]))
    Z_E = (w - 0.5) * 2.0 * theta_E

    return {
        "V_A": float(loaded_pro["V_A_base"]),
        "theta_A": float(loaded_pro["theta_A"]),
        "t_A_aff": float(loaded_pro["derived_t_A_aff_for_tied_fit"]),
        "rate_lambda": rate_lambda,
        "T0": T_0,
        "theta_E": theta_E,
        "Z_E": Z_E,
        "t_E_aff": t_E_aff,
        "del_go": del_go,
        "rate_norm_l": rate_norm_l,
        "lapse_prob": float(loaded_pro["lapse_prob"]),
        "beta_lapse": float(loaded_pro["beta_lapse"]),
    }


param_sets = {
    fit_label: get_full_parameter_set(payload)
    for fit_label, payload in fit_payloads.items()
}


def maybe_subsample_df(df, dataset_label):
    if max_rows_for_loglike_eval is None or len(df) <= max_rows_for_loglike_eval:
        return df.copy(), False
    sampled = (
        df.sample(n=max_rows_for_loglike_eval, random_state=random_seed)
        .copy()
        .reset_index(drop=True)
    )
    print(
        f"Subsampling {dataset_label}: using {len(sampled)} of {len(df)} rows "
        f"for likelihood evaluation."
    )
    return sampled, True


def compute_loglikes(records, params, mode):
    if mode == "censor":
        return Parallel(n_jobs=n_jobs)(
            delayed(trial_logpdf_proactive_lapse_only_no_trunc)(
                row=row,
                V_A=params["V_A"],
                theta_A=params["theta_A"],
                t_A_aff=params["t_A_aff"],
                rate_lambda=params["rate_lambda"],
                T0=params["T0"],
                theta_E=params["theta_E"],
                Z_E=params["Z_E"],
                t_E_aff=params["t_E_aff"],
                del_go=params["del_go"],
                phi_params=phi_params_obj,
                rate_norm_l=params["rate_norm_l"],
                is_norm=is_norm,
                is_time_vary=is_time_vary,
                K_max=K_max,
                lapse_prob=params["lapse_prob"],
                beta_lapse=params["beta_lapse"],
                lapse_choice_prob=0.5,
                censor_rt_wrt_stim=censor_rt_wrt_stim_s,
                eps=1e-50,
            )
            for row in records
        )
    if mode == "truncate":
        return Parallel(n_jobs=n_jobs)(
            delayed(trial_logpdf_proactive_lapse_only_no_trunc_right_truncated)(
                row=row,
                V_A=params["V_A"],
                theta_A=params["theta_A"],
                t_A_aff=params["t_A_aff"],
                rate_lambda=params["rate_lambda"],
                T0=params["T0"],
                theta_E=params["theta_E"],
                Z_E=params["Z_E"],
                t_E_aff=params["t_E_aff"],
                del_go=params["del_go"],
                phi_params=phi_params_obj,
                rate_norm_l=params["rate_norm_l"],
                is_norm=is_norm,
                is_time_vary=is_time_vary,
                K_max=K_max,
                lapse_prob=params["lapse_prob"],
                beta_lapse=params["beta_lapse"],
                lapse_choice_prob=0.5,
                truncate_rt_wrt_stim=truncate_rt_wrt_stim_s,
                eps=1e-50,
            )
            for row in records
        )
    if mode == "raw_no_censor":
        return Parallel(n_jobs=n_jobs)(
            delayed(trial_logpdf_proactive_lapse_only_no_trunc)(
                row=row,
                V_A=params["V_A"],
                theta_A=params["theta_A"],
                t_A_aff=params["t_A_aff"],
                rate_lambda=params["rate_lambda"],
                T0=params["T0"],
                theta_E=params["theta_E"],
                Z_E=params["Z_E"],
                t_E_aff=params["t_E_aff"],
                del_go=params["del_go"],
                phi_params=phi_params_obj,
                rate_norm_l=params["rate_norm_l"],
                is_norm=is_norm,
                is_time_vary=is_time_vary,
                K_max=K_max,
                lapse_prob=params["lapse_prob"],
                beta_lapse=params["beta_lapse"],
                lapse_choice_prob=0.5,
                censor_rt_wrt_stim=None,
                eps=1e-50,
            )
            for row in records
        )
    raise ValueError(f"Unknown mode: {mode}")


evaluation_specs = [
    {
        "evaluation": "censor_objective_on_censor_train",
        "mode": "censor",
        "dataset_label": "valid_full_for_censor_fit",
        "dataset_df": df_valid_censor,
    },
    {
        "evaluation": "truncate_objective_on_truncate_train",
        "mode": "truncate",
        "dataset_label": "valid_0_to_130ms_for_truncate_fit",
        "dataset_df": df_valid_truncate,
    },
    {
        "evaluation": "raw_no_censor_on_full_valid",
        "mode": "raw_no_censor",
        "dataset_label": "valid_full_for_censor_fit",
        "dataset_df": df_valid_censor,
    },
]

objective_rows = []
for eval_spec in evaluation_specs:
    dataset_df_full = eval_spec["dataset_df"]
    dataset_df_eval, was_subsampled = maybe_subsample_df(dataset_df_full, eval_spec["dataset_label"])
    records = dataset_df_eval[
        ["TotalFixTime", "intended_fix", "ILD", "ABL", "choice", "RTwrtStim"]
    ].to_dict("records")

    print(
        f"\nEvaluating {eval_spec['evaluation']} on {len(dataset_df_eval)} rows "
        f"(full dataset size={len(dataset_df_full)}) ..."
    )

    for param_fit_label, params in param_sets.items():
        loglikes = np.asarray(compute_loglikes(records, params, eval_spec["mode"]), dtype=np.float64)
        finite_mask = np.isfinite(loglikes)
        mean_loglike = float(np.mean(loglikes[finite_mask])) if finite_mask.any() else -np.inf
        total_loglike_subset = float(np.sum(loglikes))
        estimated_total_loglike = mean_loglike * len(dataset_df_full) if finite_mask.any() else -np.inf

        objective_rows.append(
            {
                "evaluation": eval_spec["evaluation"],
                "mode": eval_spec["mode"],
                "dataset_label": eval_spec["dataset_label"],
                "params_from_fit": param_fit_label,
                "n_trials_full_dataset": int(len(dataset_df_full)),
                "n_trials_evaluated": int(len(dataset_df_eval)),
                "was_subsampled": bool(was_subsampled),
                "n_finite": int(np.sum(finite_mask)),
                "n_nonfinite": int(np.sum(~finite_mask)),
                "total_loglike_evaluated_rows": total_loglike_subset,
                "mean_loglike_per_trial": mean_loglike,
                "estimated_total_loglike_full_dataset": estimated_total_loglike,
            }
        )

objective_df = pd.DataFrame(objective_rows)
objective_pivot_df = objective_df.pivot(
    index="evaluation",
    columns="params_from_fit",
    values="mean_loglike_per_trial",
).reset_index()
if {"censor", "truncate"}.issubset(objective_pivot_df.columns):
    objective_pivot_df["truncate_minus_censor"] = (
        objective_pivot_df["truncate"] - objective_pivot_df["censor"]
    )

winner_rows = (
    objective_df.sort_values(
        ["evaluation", "mean_loglike_per_trial"],
        ascending=[True, False],
    )
    .groupby("evaluation", as_index=False)
    .first()[["evaluation", "params_from_fit", "mean_loglike_per_trial"]]
    .rename(columns={"params_from_fit": "winner"})
)

print("\nObjective comparison (long form):")
print(objective_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

print("\nObjective comparison (pivoted mean loglike per trial):")
print(objective_pivot_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

print("\nBest parameter set per evaluation:")
print(winner_rows.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


# %%
############ Save comparison outputs ############
param_save_df = pd.concat([tied_comparison_df, proactive_comparison_df], ignore_index=True, sort=False)
param_save_df.to_csv(param_csv_path, index=False)
objective_df.to_csv(objective_csv_path, index=False)

comparison_payload = {
    "config": {
        "batch_name": batch_name,
        "session_type": session_type,
        "training_level": training_level,
        "allowed_repeat_trials": allowed_repeat_trials,
        "max_rtwrtstim_for_fit": max_rtwrtstim_for_fit,
        "censor_rt_wrt_stim_s": censor_rt_wrt_stim_s,
        "truncate_rt_wrt_stim_s": truncate_rt_wrt_stim_s,
        "n_jobs": n_jobs,
        "random_seed": random_seed,
        "max_rows_for_loglike_eval": max_rows_for_loglike_eval,
        "censor_results_pkl_path": str(censor_results_pkl_path),
        "truncate_results_pkl_path": str(truncate_results_pkl_path),
    },
    "metadata": metadata_df,
    "sample_equality": sample_equality_df,
    "param_summary": param_summary_df,
    "tied_param_comparison": tied_comparison_df,
    "proactive_param_comparison": proactive_comparison_df,
    "dataset_counts": dataset_df,
    "objective_comparison": objective_df,
    "objective_comparison_pivot": objective_pivot_df,
    "objective_winners": winner_rows,
}

with open(payload_path, "wb") as f:
    pickle.dump(comparison_payload, f)

print(f"\nSaved parameter comparison CSV: {param_csv_path}")
print(f"Saved objective comparison CSV: {objective_csv_path}")
print(f"Saved comparison payload: {payload_path}")
print("Done.")
