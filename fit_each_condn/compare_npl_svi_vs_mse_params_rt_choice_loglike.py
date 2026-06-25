# %%
"""
Compare RT+choice likelihood for NPL+alpha SVI parameters vs Gamma/Omega MSE parameters.

The per-animal MSE alpha-model fits match condition Gamma/Omega means better than
the animal-wise NPL+alpha SVI posterior means. This diagnostic asks whether those
MSE-derived parameters also improve the original RT+choice data likelihood.

For each animal, both parameter sets are evaluated with the same NPL+alpha
condition-delay likelihood and the same valid [0, 1s] trials used by the SVI fits.
"""

# %%
# =============================================================================
# Editable parameters
# =============================================================================
from pathlib import Path
import os
import pickle
import re
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
ANIMAL_FIT_DIR = REPO_DIR / "fit_animal_by_animal"
sys.path.insert(0, str(ANIMAL_FIT_DIR))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import numpyro_npl_alpha_svi_utils as svi_utils

ANIMAL_SVI_ROOT = ANIMAL_FIT_DIR / "numpyro_svi_npl_alpha_condition_delay_single_animal_outputs"
COMPARISON_DIR = SCRIPT_DIR / "svi_condition_gamma_omega_vs_npl_alpha_svi_comparison"
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

MSE_PARAM_CSV = COMPARISON_DIR / "per_animal_mse_gamma_omega_alpha_params.csv"

OUT_LIKE_CSV = COMPARISON_DIR / "npl_svi_vs_mse_rt_choice_loglike_by_animal.csv"
OUT_SUMMARY_CSV = COMPARISON_DIR / "npl_svi_vs_mse_rt_choice_loglike_summary.csv"
OUT_TOTAL_FIG = COMPARISON_DIR / "npl_svi_vs_mse_total_rt_choice_loglike_by_animal.png"
OUT_PER_TRIAL_FIG = COMPARISON_DIR / "npl_svi_vs_mse_per_trial_rt_choice_loglike_by_animal.png"
OUT_DELTA_FIG = COMPARISON_DIR / "npl_svi_vs_mse_rt_choice_loglike_delta_by_animal.png"

EXPECTED_N_ANIMALS = 30
HARD_CODED_DELAY_ABL_LEVELS = [20.0, 40.0, 60.0]
BATCH_T_TRUNC = {"LED34_even": 0.15}
DEFAULT_T_TRUNC = 0.3
K_MAX = int(os.environ.get("K_MAX", "10"))

NPL_GLOBAL_PARAM_NAMES = ["rate_lambda", "T_0", "theta_E", "w", "del_go", "rate_norm_l", "alpha"]
MSE_TO_NPL_PARAM = {
    "rate_lambda": "rate_lambda",
    "T_0": "T_0",
    "theta": "theta_E",
    "ell": "rate_norm_l",
    "alpha": "alpha",
}


# %%
# =============================================================================
# Helpers
# =============================================================================
def parse_animal_folder(path):
    match = re.match(r"^(?P<batch>.+)_(?P<animal>\d+)$", path.name)
    if match is None:
        return None
    return match.group("batch"), int(match.group("animal"))


def finite_mean(values):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.nan
    return float(np.mean(values[finite]))


def load_mse_params():
    if not MSE_PARAM_CSV.exists():
        raise FileNotFoundError(MSE_PARAM_CSV)

    mse_df = pd.read_csv(MSE_PARAM_CSV)
    mse_df["animal"] = mse_df["animal"].astype(int)
    if len(mse_df) != EXPECTED_N_ANIMALS:
        raise RuntimeError(f"Expected {EXPECTED_N_ANIMALS} MSE rows, found {len(mse_df)}")
    if "success" not in mse_df.columns:
        raise KeyError(f"{MSE_PARAM_CSV} is missing `success`.")

    failed = mse_df[mse_df["success"] != True]
    if len(failed):
        raise RuntimeError(f"MSE fits failed for:\n{failed[['batch_name', 'animal', 'message']].to_string(index=False)}")

    duplicate_count = int(mse_df.duplicated(["batch_name", "animal"]).sum())
    if duplicate_count:
        raise RuntimeError(f"Found {duplicate_count} duplicate MSE animal rows.")
    return mse_df.reset_index(drop=True)


def find_animal_svi_folders(expected_animals):
    folder_by_animal = {}
    for animal_dir in sorted(ANIMAL_SVI_ROOT.iterdir()):
        if not animal_dir.is_dir():
            continue
        parsed = parse_animal_folder(animal_dir)
        if parsed is not None and parsed in expected_animals:
            folder_by_animal[parsed] = animal_dir

    if set(folder_by_animal) != expected_animals:
        missing = sorted(expected_animals - set(folder_by_animal))
        extra = sorted(set(folder_by_animal) - expected_animals)
        raise RuntimeError(f"SVI folder mismatch. Missing={missing}, extra={extra}")
    return folder_by_animal


def load_npl_posterior_mean_params(animal_dir):
    posterior_npz = animal_dir / "main_fullrank_posterior_samples.npz"
    condition_csv = animal_dir / "condition_table.csv"
    if not posterior_npz.exists():
        raise FileNotFoundError(posterior_npz)
    if not condition_csv.exists():
        raise FileNotFoundError(condition_csv)

    samples = np.load(posterior_npz)
    condition_table = pd.read_csv(condition_csv)
    condition_table["ABL"] = condition_table["ABL"].astype(float)
    condition_table["ILD"] = condition_table["ILD"].astype(float)
    condition_table["condition_id"] = condition_table["condition_id"].astype(int)
    condition_table = condition_table.sort_values("condition_id").reset_index(drop=True)

    expected_ids = np.arange(len(condition_table), dtype=int)
    if not np.array_equal(condition_table["condition_id"].to_numpy(dtype=int), expected_ids):
        raise RuntimeError(f"{condition_csv} has non-contiguous condition_id values.")

    params = {}
    for param_name in NPL_GLOBAL_PARAM_NAMES:
        if param_name not in samples.files:
            raise KeyError(f"{posterior_npz} missing {param_name!r}")
        params[param_name] = finite_mean(samples[param_name])

    if "t_E_aff" not in samples.files:
        raise KeyError(f"{posterior_npz} missing 't_E_aff'")
    t_e_aff = np.asarray(samples["t_E_aff"], dtype=float)
    if t_e_aff.ndim != 2 or t_e_aff.shape[1] != len(condition_table):
        raise RuntimeError(
            f"{posterior_npz} t_E_aff shape {t_e_aff.shape} does not match "
            f"{len(condition_table)} conditions."
        )
    params["t_E_aff"] = np.nanmean(t_e_aff, axis=0)

    if not np.all(np.isfinite([params[name] for name in NPL_GLOBAL_PARAM_NAMES])):
        raise RuntimeError(f"Non-finite NPL global posterior mean in {posterior_npz}")
    if not np.all(np.isfinite(params["t_E_aff"])):
        raise RuntimeError(f"Non-finite NPL t_E_aff posterior mean in {posterior_npz}")

    return params, condition_table, posterior_npz


def load_valid_fitting_trials(batch_name, animal_id, condition_table):
    batch_csv = REPO_DIR / "raw_data" / "batch_csvs" / f"batch_{batch_name}_valid_and_aborts.csv"
    abort_pkl = REPO_DIR / "aborts_ipl_npl_time_fit_results" / f"results_{batch_name}_animal_{animal_id}.pkl"
    if not batch_csv.exists():
        raise FileNotFoundError(batch_csv)
    if not abort_pkl.exists():
        raise FileNotFoundError(abort_pkl)

    raw_df = pd.read_csv(batch_csv)
    if "choice" not in raw_df.columns:
        if "response_poke" not in raw_df.columns:
            raise KeyError(f"{batch_csv} needs either `choice` or `response_poke`.")
        raw_df["choice"] = raw_df["response_poke"].map({3: 1, 2: -1})

    valid_df = raw_df[
        (raw_df["animal"].astype(int) == int(animal_id))
        & (raw_df["success"].isin([1, -1]))
        & (raw_df["RTwrtStim"] < 1)
        & (raw_df["ABL"].isin(HARD_CODED_DELAY_ABL_LEVELS))
    ].copy()
    valid_df = valid_df.dropna(subset=["TotalFixTime", "intended_fix", "ABL", "ILD", "choice", "RTwrtStim"])
    if len(valid_df) == 0:
        raise RuntimeError(f"No valid RT<1 fitting trials for {batch_name}/{animal_id}.")

    valid_df["ABL"] = valid_df["ABL"].astype(float)
    valid_df["ILD"] = valid_df["ILD"].astype(float)
    valid_df["choice"] = valid_df["choice"].astype(int)

    observed_conditions = (
        valid_df[["ABL", "ILD"]]
        .drop_duplicates()
        .sort_values(["ABL", "ILD"])
        .reset_index(drop=True)
    )
    saved_conditions = condition_table[["ABL", "ILD"]].sort_values(["ABL", "ILD"]).reset_index(drop=True)
    if not observed_conditions.equals(saved_conditions):
        raise RuntimeError(
            f"Observed conditions do not match saved SVI condition table for {batch_name}/{animal_id}.\n"
            f"Observed:\n{observed_conditions.to_string(index=False)}\n"
            f"Saved:\n{saved_conditions.to_string(index=False)}"
        )

    valid_df = valid_df.merge(
        condition_table[["ABL", "ILD", "condition_id"]],
        on=["ABL", "ILD"],
        how="left",
        validate="many_to_one",
    )
    if valid_df["condition_id"].isna().any():
        raise RuntimeError(f"Failed to assign condition IDs for {batch_name}/{animal_id}.")

    with abort_pkl.open("rb") as f:
        abort_saved = pickle.load(f)
    abort_results = abort_saved["vbmc_aborts_results"]
    V_A = float(np.mean(abort_results["V_A_samples"]))
    theta_A = float(np.mean(abort_results["theta_A_samples"]))
    t_A_aff = float(np.mean(abort_results["t_A_aff_samp"]))
    T_trunc = BATCH_T_TRUNC.get(batch_name, DEFAULT_T_TRUNC)

    data = {
        "total_fix": jnp.asarray(valid_df["TotalFixTime"].to_numpy(dtype=float)),
        "t_stim": jnp.asarray(valid_df["intended_fix"].to_numpy(dtype=float)),
        "ABL": jnp.asarray(valid_df["ABL"].to_numpy(dtype=float)),
        "ILD": jnp.asarray(valid_df["ILD"].to_numpy(dtype=float)),
        "choice": jnp.asarray(valid_df["choice"].to_numpy(dtype=int)),
        "condition_id": jnp.asarray(valid_df["condition_id"].to_numpy(dtype=int)),
        "V_A": jnp.asarray(V_A, dtype=jnp.float64),
        "theta_A": jnp.asarray(theta_A, dtype=jnp.float64),
        "t_A_aff": jnp.asarray(t_A_aff, dtype=jnp.float64),
        "T_trunc": jnp.asarray(T_trunc, dtype=jnp.float64),
    }

    return valid_df, data, batch_csv, abort_pkl, T_trunc


def make_mse_substituted_params(npl_params, mse_row):
    params = dict(npl_params)
    params["t_E_aff"] = np.asarray(npl_params["t_E_aff"], dtype=float).copy()
    for mse_col, npl_name in MSE_TO_NPL_PARAM.items():
        params[npl_name] = float(getattr(mse_row, mse_col))
    return params


def as_jax_params(params):
    out = {}
    for key, value in params.items():
        if key == "t_E_aff":
            out[key] = jnp.asarray(value, dtype=jnp.float64)
        else:
            out[key] = jnp.asarray(float(value), dtype=jnp.float64)
    return out


@jax.jit
def loglike_jit(params, data):
    return svi_utils.npl_alpha_condition_delay_loglike(params, data, K_max=K_MAX)


def evaluate_loglike(params, data):
    value = loglike_jit(as_jax_params(params), data)
    return float(jax.block_until_ready(value))


def count_mse_hard_bound_violations(params):
    violations = []
    for param_name in ["rate_lambda", "T_0", "theta_E", "rate_norm_l", "alpha"]:
        hard_low, hard_high = svi_utils.GLOBAL_BOUNDS[param_name]["hard"]
        value = float(params[param_name])
        if not (hard_low <= value <= hard_high):
            violations.append(param_name)
    return ",".join(violations)


def build_summary_rows(like_df):
    rows = []
    for metric_name, delta_col in [
        ("total_loglike", "delta_total_loglike_mse_minus_npl"),
        ("per_trial_loglike", "delta_per_trial_loglike_mse_minus_npl"),
    ]:
        deltas = like_df[delta_col].to_numpy(dtype=float)
        rows.append(
            {
                "metric": metric_name,
                "n_animals": int(len(like_df)),
                "n_mse_better": int(np.sum(deltas > 0)),
                "n_npl_better": int(np.sum(deltas < 0)),
                "mean_mse_minus_npl": float(np.mean(deltas)),
                "median_mse_minus_npl": float(np.median(deltas)),
                "min_mse_minus_npl": float(np.min(deltas)),
                "max_mse_minus_npl": float(np.max(deltas)),
            }
        )
    return pd.DataFrame(rows)


def plot_likelihood_comparison(like_df, value_suffix, ylabel, title, output_path):
    x = np.arange(len(like_df))
    animal_labels = like_df["animal_label"].tolist()
    npl_values = like_df[f"npl_{value_suffix}"].to_numpy(dtype=float)
    mse_values = like_df[f"mse_{value_suffix}"].to_numpy(dtype=float)
    deltas = mse_values - npl_values

    fig, ax = plt.subplots(figsize=(15, 6))
    for idx in range(len(like_df)):
        ax.plot([x[idx] - 0.08, x[idx] + 0.08], [npl_values[idx], mse_values[idx]], color="0.75", linewidth=1.0, zorder=1)

    ax.scatter(x - 0.08, npl_values, color="tab:blue", s=38, label="NPL+alpha SVI posterior mean", zorder=3)
    ax.scatter(x + 0.08, mse_values, color="tab:red", s=38, label="MSE Gamma/Omega params", zorder=3)
    ax.axhline(np.nanmean(npl_values), color="tab:blue", linewidth=1.2, alpha=0.3)
    ax.axhline(np.nanmean(mse_values), color="tab:red", linewidth=1.2, alpha=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(animal_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{title}\n"
        f"mean MSE-NPL delta={np.mean(deltas):+.4g}; "
        f"MSE better in {int(np.sum(deltas > 0))}/{len(deltas)} animals"
    )
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return output_path


def plot_delta_comparison(like_df, output_path):
    x = np.arange(len(like_df))
    animal_labels = like_df["animal_label"].tolist()
    total_delta = like_df["delta_total_loglike_mse_minus_npl"].to_numpy(dtype=float)
    per_trial_delta = like_df["delta_per_trial_loglike_mse_minus_npl"].to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(15, 8.5), sharex=True)
    for ax, values, ylabel, title in [
        (
            axes[0],
            total_delta,
            "Delta total log likelihood\n(MSE - NPL)",
            "Total RT+choice likelihood delta",
        ),
        (
            axes[1],
            per_trial_delta,
            "Delta log likelihood per trial\n(MSE - NPL)",
            "Per-trial RT+choice likelihood delta",
        ),
    ]:
        colors = np.where(values >= 0, "tab:green", "tab:red")
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.65)
        ax.axhline(np.mean(values), color="tab:red", linewidth=1.2, alpha=0.35)
        ax.scatter(x, values, color=colors, s=34, zorder=3)
        ax.vlines(x, 0.0, values, color=colors, alpha=0.45, linewidth=1.2)
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"{title}: mean={np.mean(values):+.4g}, "
            f"MSE better in {int(np.sum(values > 0))}/{len(values)} animals",
            fontsize=11,
        )
        ax.grid(True, axis="y", alpha=0.25)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(animal_labels, rotation=45, ha="right", fontsize=8)

    handles = [
        Line2D([0], [0], marker="o", color="tab:red", linestyle="none", label="MSE lower likelihood"),
        Line2D([0], [0], marker="o", color="tab:green", linestyle="none", label="MSE higher likelihood"),
        Line2D([0], [0], color="black", linewidth=1.0, label="zero delta"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("RT+choice likelihood delta by animal: MSE Gamma/Omega params minus NPL SVI params", y=1.05)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return output_path


# %%
# =============================================================================
# Main analysis
# =============================================================================
print(f"Animal-wise NPL SVI root: {ANIMAL_SVI_ROOT}")
print(f"MSE parameter CSV: {MSE_PARAM_CSV}")
print(f"Output folder: {COMPARISON_DIR}")
print(f"K_MAX: {K_MAX}")

mse_df = load_mse_params()
expected_animals = {(row.batch_name, int(row.animal)) for row in mse_df[["batch_name", "animal"]].itertuples(index=False)}
folder_by_animal = find_animal_svi_folders(expected_animals)

rows = []
start_time = time.time()
for idx, mse_row in enumerate(mse_df.itertuples(index=False), start=1):
    batch_name = str(mse_row.batch_name)
    animal_id = int(mse_row.animal)
    animal_label = f"{batch_name}/{animal_id}"
    animal_dir = folder_by_animal[(batch_name, animal_id)]

    print(f"[{idx:02d}/{len(mse_df)}] {animal_label}: loading data and evaluating likelihoods...")
    npl_params, condition_table, posterior_npz = load_npl_posterior_mean_params(animal_dir)
    valid_df, data, batch_csv, abort_pkl, T_trunc = load_valid_fitting_trials(batch_name, animal_id, condition_table)
    mse_params = make_mse_substituted_params(npl_params, mse_row)

    npl_total_loglike = evaluate_loglike(npl_params, data)
    mse_total_loglike = evaluate_loglike(mse_params, data)
    n_trials = int(len(valid_df))
    npl_per_trial = npl_total_loglike / n_trials
    mse_per_trial = mse_total_loglike / n_trials

    rows.append(
        {
            "batch_name": batch_name,
            "animal": animal_id,
            "animal_label": animal_label,
            "n_trials": n_trials,
            "n_conditions": int(len(condition_table)),
            "T_trunc": float(T_trunc),
            "npl_total_loglike": npl_total_loglike,
            "mse_total_loglike": mse_total_loglike,
            "delta_total_loglike_mse_minus_npl": mse_total_loglike - npl_total_loglike,
            "npl_per_trial_loglike": npl_per_trial,
            "mse_per_trial_loglike": mse_per_trial,
            "delta_per_trial_loglike_mse_minus_npl": mse_per_trial - npl_per_trial,
            "mse_hard_bound_violations": count_mse_hard_bound_violations(mse_params),
            "posterior_npz": str(posterior_npz),
            "batch_csv": str(batch_csv),
            "abort_pkl": str(abort_pkl),
        }
    )
    print(
        f"    trials={n_trials}, "
        f"NPL={npl_total_loglike:.3f}, MSE={mse_total_loglike:.3f}, "
        f"delta/trial={mse_per_trial - npl_per_trial:+.6g}"
    )

like_df = pd.DataFrame(rows)
summary_df = build_summary_rows(like_df)

if len(like_df) != EXPECTED_N_ANIMALS:
    raise RuntimeError(f"Expected {EXPECTED_N_ANIMALS} likelihood rows, found {len(like_df)}")
if not np.all(np.isfinite(like_df[["npl_total_loglike", "mse_total_loglike", "npl_per_trial_loglike", "mse_per_trial_loglike"]].to_numpy(dtype=float))):
    raise RuntimeError("Likelihood table contains non-finite values.")

like_df.to_csv(OUT_LIKE_CSV, index=False)
summary_df.to_csv(OUT_SUMMARY_CSV, index=False)
print(f"\nSaved likelihood table: {OUT_LIKE_CSV}")
print(f"Saved summary table: {OUT_SUMMARY_CSV}")
print(summary_df.to_string(index=False))
print(f"Elapsed: {(time.time() - start_time) / 60:.2f} min")

total_fig = plot_likelihood_comparison(
    like_df,
    "total_loglike",
    "Total RT+choice log likelihood",
    "Total RT+choice likelihood on NPL SVI fitting trials",
    OUT_TOTAL_FIG,
)
per_trial_fig = plot_likelihood_comparison(
    like_df,
    "per_trial_loglike",
    "RT+choice log likelihood per trial",
    "Per-trial RT+choice likelihood on NPL SVI fitting trials",
    OUT_PER_TRIAL_FIG,
)
delta_fig = plot_delta_comparison(like_df, OUT_DELTA_FIG)
print(f"Saved figure: {total_fig}")
print(f"Saved figure: {per_trial_fig}")
print(f"Saved figure: {delta_fig}")

# %%
