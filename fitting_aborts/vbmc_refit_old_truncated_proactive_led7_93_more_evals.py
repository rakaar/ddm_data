# %%
"""Higher-budget refit of the paper-era truncated proactive model for LED7/93."""

# %%
# =============================================================================
# Editable run settings
# =============================================================================
from pathlib import Path
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from pyvbmc import VBMC
from scipy.special import ndtr


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
FIT_ANIMAL_DIR = REPO_DIR / "fit_animal_by_animal"

BATCH_NAME = "LED7"
ANIMAL = 93
T_TRUNC_S = 0.300
MAX_FUN_EVALS = int(os.environ.get("VBMC_MAX_FUN_EVALS", "10000"))
POSTERIOR_SAMPLES = int(os.environ.get("VBMC_POSTERIOR_SAMPLES", "200000"))
RANDOM_SEED = int(os.environ.get("VBMC_RANDOM_SEED", "20260829"))

DATA_CSV = REPO_DIR / "raw_data" / "batch_csvs" / "batch_LED7_valid_and_aborts.csv"
OLD_RESULT_PKL = REPO_DIR / "aborts_ipl_npl_time_fit_results" / "results_LED7_animal_93.pkl"
OUTPUT_DIR = SCRIPT_DIR / "vbmc_old_truncated_proactive_led7_93_more_evals"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PARAMETER_NAMES = ("V_A", "theta_A", "t_A_aff")

# The historical result used a -5 s delay lower bound. The repository later
# corrected that implausibly broad bound to -0.5 s; this refit uses the fix.
LOWER_BOUNDS = np.array([0.1, 0.1, -0.5], dtype=float)
UPPER_BOUNDS = np.array([10.0, 10.0, 0.1], dtype=float)
PLAUSIBLE_LOWER_BOUNDS = np.array([0.5, 0.5, -0.2], dtype=float)
PLAUSIBLE_UPPER_BOUNDS = np.array([4.0, 4.0, 0.06], dtype=float)
INITIAL_POINT = np.array([2.5, 1.8, -0.14], dtype=float)

sys.path.insert(0, str(FIT_ANIMAL_DIR))
from time_vary_norm_utils import cum_A_t_fn, rho_A_t_fn
from vbmc_animal_wise_fit_utils import trapezoidal_logpdf


# %%
# =============================================================================
# Rebuild the historical LED-OFF fitting dataset
# =============================================================================
raw_df = pd.read_csv(DATA_CSV)
fit_df = raw_df[
    ~((raw_df["RTwrtStim"].isna()) & (raw_df["abort_event"] == 3))
].copy()
fit_df = fit_df[
    (fit_df["animal"].astype(int) == ANIMAL)
    & fit_df["session_type"].isin([1, 7])
    & ((fit_df["LED_trial"] == 0) | fit_df["LED_trial"].isna())
    & (fit_df["success"].isin([1, -1]) | (fit_df["abort_event"] == 3))
].copy()

if len(fit_df) != 10891:
    raise RuntimeError(f"Expected 10,891 historical LED7/93 rows, found {len(fit_df):,}.")
if fit_df[["TotalFixTime", "intended_fix"]].isna().any().any():
    raise RuntimeError("Historical likelihood columns contain missing values.")

fixation_time = fit_df["TotalFixTime"].to_numpy(dtype=float)
stimulus_time = fit_df["intended_fix"].to_numpy(dtype=float)

abort_mask = fit_df["abort_event"].eq(3).to_numpy()
valid_mask = fit_df["success"].isin([1, -1]).to_numpy()
informative_abort_mask = abort_mask & (fixation_time >= T_TRUNC_S)
informative_censored_mask = valid_mask & (stimulus_time > T_TRUNC_S)

print("=" * 78)
print(f"Refreshed paper-era proactive VBMC: {BATCH_NAME}/{ANIMAL}")
print(f"Data: {DATA_CSV}")
print(f"Rows: {len(fit_df):,}")
print(
    f"Valid/censored: {valid_mask.sum():,}; abort-event 3: {abort_mask.sum():,}; "
    f"post-truncation aborts: {informative_abort_mask.sum():,}"
)
print(f"T_trunc: {T_TRUNC_S:.3f} s")
print(f"max_fun_evals: {MAX_FUN_EVALS:,}")
print(f"Output: {OUTPUT_DIR}")
print("=" * 78)


# %%
# =============================================================================
# Historical scalar likelihood and equivalent vectorized likelihood
# =============================================================================
LOG_FLOOR = np.log(1e-50)


def historical_scalar_loglike(params, row_indices=None):
    """Literal scalar form used to validate the faster refit objective."""
    V_A, theta_A, t_A_aff = np.asarray(params, dtype=float)
    trunc_factor = 1.0 - cum_A_t_fn(T_TRUNC_S - t_A_aff, V_A, theta_A)
    indices = range(len(fit_df)) if row_indices is None else row_indices
    total = 0.0
    for index in indices:
        rt = fixation_time[index]
        t_stim = stimulus_time[index]
        if rt < T_TRUNC_S:
            likelihood = 0.0
        elif rt < t_stim:
            likelihood = rho_A_t_fn(rt - t_A_aff, V_A, theta_A) / trunc_factor
        else:
            if t_stim <= T_TRUNC_S:
                likelihood = 1.0
            else:
                likelihood = (
                    1.0 - cum_A_t_fn(t_stim - t_A_aff, V_A, theta_A)
                ) / trunc_factor
        if trunc_factor == 0 or likelihood <= 0:
            likelihood = 1e-50
        if not np.isfinite(likelihood):
            raise RuntimeError(
                f"Non-finite scalar likelihood at row {index}, params={params}."
            )
        total += np.log(likelihood)
    return float(total)


def proactive_cdf_vectorized(t, V_A, theta_A):
    t = np.asarray(t, dtype=float)
    result = np.zeros_like(t)
    positive = t > 0
    positive_t = t[positive]
    sqrt_t = np.sqrt(positive_t)
    result[positive] = ndtr(
        V_A * (positive_t - theta_A / V_A) / sqrt_t
    ) + np.exp(2.0 * V_A * theta_A) * ndtr(
        -V_A * (positive_t + theta_A / V_A) / sqrt_t
    )
    return result


def proactive_pdf_vectorized(t, V_A, theta_A):
    t = np.asarray(t, dtype=float)
    result = np.zeros_like(t)
    positive = t > 0
    positive_t = t[positive]
    result[positive] = (
        theta_A
        / np.sqrt(2.0 * np.pi * positive_t**3)
        * np.exp(
            -0.5
            * V_A**2
            * (positive_t - theta_A / V_A) ** 2
            / positive_t
        )
    )
    return result


def proactive_loglike(params, row_indices=None):
    V_A, theta_A, t_A_aff = np.asarray(params, dtype=float)
    indices = slice(None) if row_indices is None else np.asarray(row_indices, dtype=int)
    rt = fixation_time[indices]
    t_stim = stimulus_time[indices]

    trunc_factor = 1.0 - proactive_cdf_vectorized(
        np.array([T_TRUNC_S - t_A_aff]), V_A, theta_A
    )[0]
    if not np.isfinite(trunc_factor) or trunc_factor <= 0:
        return -np.inf

    likelihood = np.full(rt.shape, 1e-50, dtype=float)
    density_rows = (rt >= T_TRUNC_S) & (rt < t_stim)
    survival_rows = (rt >= T_TRUNC_S) & (rt > t_stim) & (t_stim > T_TRUNC_S)
    unit_rows = (rt >= T_TRUNC_S) & (rt > t_stim) & (t_stim <= T_TRUNC_S)

    likelihood[density_rows] = proactive_pdf_vectorized(
        rt[density_rows] - t_A_aff, V_A, theta_A
    ) / trunc_factor
    likelihood[survival_rows] = (
        1.0
        - proactive_cdf_vectorized(
            t_stim[survival_rows] - t_A_aff, V_A, theta_A
        )
    ) / trunc_factor
    likelihood[unit_rows] = 1.0
    likelihood = np.where(
        np.isfinite(likelihood) & (likelihood > 0), likelihood, 1e-50
    )
    return float(np.log(likelihood).sum())


parity_indices = np.unique(
    np.linspace(0, len(fit_df) - 1, 500, dtype=int)
)
parity_parameter_sets = (
    INITIAL_POINT,
    np.array([1.8, 2.7, -0.20]),
    np.array([2.2, 3.5, -0.05]),
)
parity_rows = []
for parameter_set in parity_parameter_sets:
    scalar_value = historical_scalar_loglike(parameter_set, parity_indices)
    vectorized_value = proactive_loglike(parameter_set, parity_indices)
    difference = vectorized_value - scalar_value
    parity_rows.append(
        {
            **dict(zip(PARAMETER_NAMES, parameter_set)),
            "scalar_loglike": scalar_value,
            "vectorized_loglike": vectorized_value,
            "difference": difference,
        }
    )
    if not np.isclose(vectorized_value, scalar_value, atol=1e-8, rtol=1e-11):
        raise RuntimeError(
            "Vectorized likelihood does not match the historical scalar likelihood: "
            f"params={parameter_set}, difference={difference:.6g}."
        )

parity_csv = OUTPUT_DIR / "likelihood_parity_check.csv"
pd.DataFrame(parity_rows).to_csv(parity_csv, index=False)
print(f"Likelihood parity passed on {len(parity_indices)} rows and 3 parameter sets.")


# %%
# =============================================================================
# Prior, joint density, and higher-budget VBMC fit
# =============================================================================
def proactive_logprior(params):
    V_A, theta_A, t_A_aff = np.asarray(params, dtype=float)
    return float(
        trapezoidal_logpdf(
            V_A,
            LOWER_BOUNDS[0],
            PLAUSIBLE_LOWER_BOUNDS[0],
            PLAUSIBLE_UPPER_BOUNDS[0],
            UPPER_BOUNDS[0],
        )
        + trapezoidal_logpdf(
            theta_A,
            LOWER_BOUNDS[1],
            PLAUSIBLE_LOWER_BOUNDS[1],
            PLAUSIBLE_UPPER_BOUNDS[1],
            UPPER_BOUNDS[1],
        )
        + trapezoidal_logpdf(
            t_A_aff,
            LOWER_BOUNDS[2],
            PLAUSIBLE_LOWER_BOUNDS[2],
            PLAUSIBLE_UPPER_BOUNDS[2],
            UPPER_BOUNDS[2],
        )
    )


def proactive_logjoint(params):
    return proactive_logprior(params) + proactive_loglike(params)


initial_loglike = proactive_loglike(INITIAL_POINT)
initial_logjoint = proactive_logjoint(INITIAL_POINT)
if not np.isfinite(initial_logjoint):
    raise RuntimeError("Initial VBMC log joint is not finite.")
print(f"Initial log likelihood: {initial_loglike:.6f}")
print(f"Initial log joint: {initial_logjoint:.6f}")

np.random.seed(RANDOM_SEED)
fit_started = time.perf_counter()
vbmc = VBMC(
    proactive_logjoint,
    INITIAL_POINT,
    LOWER_BOUNDS,
    UPPER_BOUNDS,
    PLAUSIBLE_LOWER_BOUNDS,
    PLAUSIBLE_UPPER_BOUNDS,
    options={"display": "on", "max_fun_evals": MAX_FUN_EVALS},
)
vp, results = vbmc.optimize()
runtime_s = time.perf_counter() - fit_started


# %%
# =============================================================================
# Posterior summary and old-versus-refreshed comparison
# =============================================================================
posterior_samples = np.asarray(vp.sample(POSTERIOR_SAMPLES)[0], dtype=float)
if posterior_samples.shape != (POSTERIOR_SAMPLES, len(PARAMETER_NAMES)):
    raise RuntimeError(f"Unexpected posterior shape: {posterior_samples.shape}.")
if not np.isfinite(posterior_samples).all():
    raise RuntimeError("Refreshed posterior samples contain non-finite values.")

with OLD_RESULT_PKL.open("rb") as handle:
    old_saved = pickle.load(handle)["vbmc_aborts_results"]
old_samples = np.column_stack(
    [
        np.asarray(old_saved["V_A_samples"], dtype=float),
        np.asarray(old_saved["theta_A_samples"], dtype=float),
        np.asarray(old_saved["t_A_aff_samp"], dtype=float),
    ]
)

summary_rows = []
for fit_label, samples in (("old", old_samples), ("refreshed", posterior_samples)):
    for parameter_index, parameter_name in enumerate(PARAMETER_NAMES):
        values = samples[:, parameter_index]
        q025, median, q975 = np.quantile(values, [0.025, 0.5, 0.975])
        summary_rows.append(
            {
                "fit": fit_label,
                "parameter": parameter_name,
                "mean": float(values.mean()),
                "median": float(median),
                "ci_2.5": float(q025),
                "ci_97.5": float(q975),
                "posterior_sd": float(values.std()),
                "n_samples": int(values.size),
                "units": "s" if parameter_name == "t_A_aff" else "model units",
            }
        )
summary_df = pd.DataFrame(summary_rows)
summary_csv = OUTPUT_DIR / "old_vs_refreshed_posterior_summary.csv"
summary_df.to_csv(summary_csv, index=False)

refreshed_mean = posterior_samples.mean(axis=0)
old_mean = old_samples.mean(axis=0)
comparison = {
    "batch": BATCH_NAME,
    "animal": ANIMAL,
    "status_message": str(results.get("message", "")),
    "convergence_status": results.get("convergence_status"),
    "success_flag": results.get("success_flag"),
    "r_index": results.get("r_index"),
    "elbo": float(results["elbo"]),
    "elbo_sd": float(results["elbo_sd"]),
    "runtime_s": float(runtime_s),
    "max_fun_evals": MAX_FUN_EVALS,
    "posterior_samples": POSTERIOR_SAMPLES,
    "old_status_message": str(old_saved.get("message", "")),
    "n_rows": int(len(fit_df)),
    "n_valid": int(valid_mask.sum()),
    "n_abort_event_3": int(abort_mask.sum()),
    "n_post_truncation_aborts": int(informative_abort_mask.sum()),
    "n_informative_censored": int(informative_censored_mask.sum()),
    "T_trunc_s": T_TRUNC_S,
    "bounds": {
        name: [float(LOWER_BOUNDS[index]), float(UPPER_BOUNDS[index])]
        for index, name in enumerate(PARAMETER_NAMES)
    },
    "plausible_bounds": {
        name: [
            float(PLAUSIBLE_LOWER_BOUNDS[index]),
            float(PLAUSIBLE_UPPER_BOUNDS[index]),
        ]
        for index, name in enumerate(PARAMETER_NAMES)
    },
    "initial_point": dict(zip(PARAMETER_NAMES, INITIAL_POINT.tolist())),
    "old_mean_loglike_on_refit_data": float(proactive_loglike(old_mean)),
    "refreshed_mean_loglike_on_refit_data": float(proactive_loglike(refreshed_mean)),
    "data_csv": str(DATA_CSV.resolve()),
    "old_result_pkl": str(OLD_RESULT_PKL.resolve()),
}

summary_json = OUTPUT_DIR / "run_summary.json"
summary_json.write_text(json.dumps(comparison, indent=2, default=str) + "\n")

samples_npz = OUTPUT_DIR / "refreshed_posterior_samples.npz"
np.savez_compressed(
    samples_npz,
    V_A=posterior_samples[:, 0],
    theta_A=posterior_samples[:, 1],
    t_A_aff=posterior_samples[:, 2],
)

bundle_pkl = OUTPUT_DIR / "refreshed_vbmc_bundle.pkl"
vp_pkl = OUTPUT_DIR / "refreshed_variational_posterior.pkl"
vp.save(vp_pkl, overwrite=True)
with bundle_pkl.open("wb") as handle:
    pickle.dump(
        {
            "results": results,
            "summary": comparison,
            "posterior_samples_npz": str(samples_npz.resolve()),
            "variational_posterior_pkl": str(vp_pkl.resolve()),
        },
        handle,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

print("\nOld versus refreshed posterior summaries")
print(summary_df.to_string(index=False))
print("\nRefreshed VBMC termination")
print(f"  message: {comparison['status_message']}")
print(f"  ELBO: {comparison['elbo']:.6f} +/- {comparison['elbo_sd']:.6f}")
print(f"  runtime: {runtime_s / 60.0:.2f} min")
print(
    "  posterior-mean loglike old/refreshed: "
    f"{comparison['old_mean_loglike_on_refit_data']:.6f} / "
    f"{comparison['refreshed_mean_loglike_on_refit_data']:.6f}"
)
print(f"Saved summary: {summary_csv}")
print(f"Saved run metadata: {summary_json}")
print(f"Saved posterior samples: {samples_npz}")
print(f"Saved variational posterior: {vp_pkl}")
print(f"Saved VBMC bundle: {bundle_pkl}")
