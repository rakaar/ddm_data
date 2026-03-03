"""
VBMC fit of proactive model using PyDDM likelihood.
# this is faster becoz LED OFF is calculated directly
Model summary:
- Proactive process is approximated by PyDDM with:
  1) constant drift (no LED-driven drift change)
  2) bound decreasing linearly after LED effect time until saturation
- Lapse process is exponential and mixed with proactive process.
- Trials are right-censored at t_stim (same censoring logic as before).

Notes:
- Uses t_LED bins for computational tractability (exact per-trial t_LED is too expensive).
- Uses PyDDM FP solver directly for likelihood used by VBMC.
"""

# %%
import os
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyddm
from scipy.special import erf
from joblib import Parallel, delayed
from pyvbmc import VBMC
import corner


# %%
# =============================================================================
# PARAMETERS
# =============================================================================
ANIMAL_ID = None  # None for all animals, or integer index (0, 1, 2...)

# PyDDM numerical settings (as requested)
PYDDM_DT = 1e-3
PYDDM_DX = 1e-3
PYDDM_T_DUR = 3.0

# Condition handling for LED ON trials
# If True, each unique t_LED is treated as its own condition (no binning).
# This is exact but can be very slow in VBMC.
USE_EXACT_T_LED = False
T_LED_BIN_WIDTH = 0.1

# Mapping single-bound -> PyDDM two-bound
THETA_E0_FIXED = 20.0     # must stay greater than theta_A
THETA_A_MIN = 0.0    # minimum upper distance to avoid bound crossing start
LIKELIHOOD_EPS = 1e-50
MODEL_TAG = "bound_drop_saturating"

# Runtime controls
LOAD_SAVED_RESULTS = False
VBMC_OPTIONS = {"display": "iter"}
N_JOBS = 25
PARALLEL_BACKEND = "loky"  # "loky" for processes, "threading" for threads

# Plot/simulation controls
SIM_DT = 1e-4
N_TRIALS_SIM = int(100e3)


def savefig_both(base_path_without_ext, **kwargs):
    plt.savefig(f"{base_path_without_ext}.pdf", **kwargs)
    plt.savefig(f"{base_path_without_ext}.png", **kwargs)


# %%
# =============================================================================
# Load and filter data
# =============================================================================
og_df = pd.read_csv("../out_LED.csv")

df = og_df[og_df["repeat_trial"].isin([0, 2]) | og_df["repeat_trial"].isna()]
df = df[df["session_type"].isin([7])]
df = df[df["training_level"].isin([16])]

df = df.dropna(subset=["intended_fix", "LED_onset_time", "timed_fix"])
df = df[(df["abort_event"] == 3) | (df["success"].isin([1, -1]))]

unique_animals = df["animal"].unique()

if ANIMAL_ID is not None:
    animal_name = unique_animals[ANIMAL_ID]
    print(f"Selected ANIMAL_ID={ANIMAL_ID} -> Animal: {animal_name}")
    df_all = df[df["animal"] == animal_name]
    animal_label = f"Animal {animal_name}"
    file_tag = f"animal_{animal_name}"
else:
    print(f"Aggregating all {len(unique_animals)} animals: {unique_animals}")
    df_all = df
    animal_label = "All Animals Aggregated"
    file_tag = "all_animals"

# Build fitting DataFrame
fit_df = pd.DataFrame(
    {
        "RT": df_all["timed_fix"].values,
        "t_stim": df_all["intended_fix"].values,
        "t_LED": (df_all["intended_fix"] - df_all["LED_onset_time"]).values,
        "LED_trial": np.where(df_all["LED_trial"] == 1, 1, 0),
    }
)

fit_df["is_abort"] = fit_df["RT"] < fit_df["t_stim"]
if USE_EXACT_T_LED:
    fit_df["t_led_cond"] = np.where(
        fit_df["LED_trial"] == 1,
        fit_df["t_LED"].values,
        -999.0,  # sentinel for LED OFF
    )
else:
    fit_df["t_led_cond"] = np.where(
        fit_df["LED_trial"] == 1,
        np.round(fit_df["t_LED"] / T_LED_BIN_WIDTH) * T_LED_BIN_WIDTH,
        -999.0,  # sentinel for LED OFF
    )

print(f"\n{animal_label} fitting summary:")
print(f"  Total trials: {len(fit_df)}")
print(f"  LED ON trials: {np.sum(fit_df['LED_trial'] == 1)}")
print(f"  LED OFF trials: {np.sum(fit_df['LED_trial'] == 0)}")
print(f"  Abort trials (RT < t_stim): {np.sum(fit_df['is_abort'])}")
print(f"  Censored trials (RT >= t_stim): {np.sum(~fit_df['is_abort'])}")
if USE_EXACT_T_LED:
    print(f"  Unique LED ON exact t_LED conditions: {fit_df.loc[fit_df['LED_trial']==1, 't_led_cond'].nunique()}")
    print("  WARNING: exact t_LED mode can be very slow for VBMC.")
else:
    print(f"  Unique LED ON bins (width={T_LED_BIN_WIDTH:.3f}s): {fit_df.loc[fit_df['LED_trial']==1, 't_led_cond'].nunique()}")


# %%
# =============================================================================
# Precompute condition groups for fast likelihood
# =============================================================================
condition_groups = []

for (is_led, t_led_cond), grp in fit_df.groupby(["LED_trial", "t_led_cond"], sort=True):
    abort_rts = grp.loc[grp["is_abort"], "RT"].values.astype(float)
    cens_tstims = grp.loc[~grp["is_abort"], "t_stim"].values.astype(float)

    condition_groups.append(
        {
            "is_led": int(is_led),
            "t_led_cond": float(t_led_cond),
            "abort_rts": abort_rts,
            "cens_tstims": cens_tstims,
            "n_trials": len(grp),
            "n_abort": len(abort_rts),
            "n_cens": len(cens_tstims),
        }
    )

print(f"Condition groups for likelihood: {len(condition_groups)}")


# %%
# =============================================================================
# Utility functions
# =============================================================================
def trapezoidal_logpdf(x, a, b, c, d):
    if x < a or x > d:
        return -np.inf

    area = ((b - a) + (d - c)) / 2 + (c - b)
    h_max = 1.0 / area

    if a <= x <= b:
        pdf_value = ((x - a) / (b - a)) * h_max
    elif b < x < c:
        pdf_value = h_max
    elif c <= x <= d:
        pdf_value = ((d - x) / (d - c)) * h_max
    else:
        pdf_value = 0.0

    if pdf_value <= 0.0:
        return -np.inf
    return np.log(pdf_value)


def lapse_pdf(t, beta):
    return beta * np.exp(-beta * t)


def lapse_survival(t, beta):
    return np.exp(-beta * t)


def _norm_cdf(x):
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def _led_off_pdf_vec(t_abs, V_A_base, theta_A, shift_total):
    """
    Proactive PDF for LED OFF trials (no LED-driven parameter change), vectorized.
    """
    t_proc = np.asarray(t_abs, dtype=float) - shift_total
    out = np.zeros_like(t_proc, dtype=float)
    valid = t_proc > 0.0
    if not np.any(valid):
        return out

    tp = t_proc[valid]
    coef = theta_A / np.sqrt(2.0 * np.pi * tp**3)
    expo = -0.5 * (V_A_base**2) * ((tp - (theta_A / V_A_base)) ** 2) / tp
    out[valid] = coef * np.exp(expo)
    return out


def _led_off_cdf_vec(t_abs, V_A_base, theta_A, shift_total):
    """
    Proactive CDF for LED OFF trials, vectorized.
    """
    t_proc = np.asarray(t_abs, dtype=float) - shift_total
    out = np.zeros_like(t_proc, dtype=float)
    valid = t_proc > 0.0
    if not np.any(valid):
        return out

    tp = t_proc[valid]
    sqrt_tp = np.sqrt(tp)
    theta_over_v = theta_A / V_A_base

    z1 = V_A_base * (tp - theta_over_v) / sqrt_tp
    z2 = -V_A_base * (tp + theta_over_v) / sqrt_tp

    cdf_vals = _norm_cdf(z1) + np.exp(2.0 * V_A_base * theta_A) * _norm_cdf(z2)
    out[valid] = np.clip(cdf_vals, 0.0, 1.0)
    return out


def _led_off_survival_vec(t_abs, V_A_base, theta_A, shift_total):
    return np.clip(1.0 - _led_off_cdf_vec(t_abs, V_A_base, theta_A, shift_total), 0.0, 1.0)


def _pdf_lookup_on_grid(pdf_vals, t_vals, dt, t_dur):
    out = np.zeros_like(t_vals, dtype=float)
    valid = (t_vals >= 0.0) & (t_vals <= t_dur)
    if np.any(valid):
        idx = np.rint(t_vals[valid] / dt).astype(int)
        idx = np.clip(idx, 0, len(pdf_vals) - 1)
        out[valid] = pdf_vals[idx]
    return out


def _upper_survival_lookup(cdf_upper, p_upper, t_vals, dt, t_dur):
    out = np.ones_like(t_vals, dtype=float)

    mid = (t_vals > 0.0) & (t_vals < t_dur)
    if np.any(mid):
        idx = np.rint(t_vals[mid] / dt).astype(int)
        idx = np.clip(idx, 0, len(cdf_upper) - 1)
        out[mid] = 1.0 - cdf_upper[idx]

    late = t_vals >= t_dur
    if np.any(late):
        out[late] = max(0.0, 1.0 - p_upper)

    return np.clip(out, 0.0, 1.0)


# %%
# =============================================================================
# PyDDM model and likelihood
# =============================================================================
def build_pyddm_model(V_A_base, theta_A, bound_slope, theta_A_saturate, t_change):
    """
    Build a PyDDM model for one condition (one t_change).

    - Drift: V_A_base (constant)
    - Bound distance to upper:
      max(theta_A_saturate, theta_A - bound_slope * max(0, t - t_change))
      clipped to avoid crossing the lower bound.
    """
    x = 1.0 - (theta_A / THETA_E0_FIXED)
    x_abs = x * THETA_E0_FIXED
    theta_E_min = x_abs + THETA_A_MIN
    theta_E_sat = THETA_E0_FIXED - (theta_A - theta_A_saturate)
    theta_E_floor = max(theta_E_min, theta_E_sat)

    def drift_fn(t):
        return V_A_base

    def bound_fn(t):
        B_t = THETA_E0_FIXED - bound_slope * max(0.0, t - t_change)
        return max(theta_E_floor, B_t)

    return pyddm.gddm(
        drift=drift_fn,
        noise=1.0,
        bound=bound_fn,
        starting_position=x,
        nondecision=0.0,
        mixture_coef=0.0,
        dt=PYDDM_DT,
        dx=PYDDM_DX,
        T_dur=PYDDM_T_DUR,
        choice_names=("upper_hit", "lower_hit"),
    )


def _group_loglike(
    grp,
    V_A_base,
    theta_A,
    bound_slope,
    theta_A_saturate,
    del_a_minus_del_LED,
    del_m_plus_del_LED,
    lapse_prob,
    beta_lapse,
):
    """Log-likelihood contribution for one (LED_trial, t_led_cond) group."""
    is_led = grp["is_led"]
    t_led_cond = grp["t_led_cond"]
    abort_rts = grp["abort_rts"]
    cens_tstims = grp["cens_tstims"]
    shift_total = del_a_minus_del_LED + del_m_plus_del_LED

    # LED OFF: use direct no-LED formulas (faster, and drift/bound slopes are irrelevant here).
    if is_led == 0:
        group_loglike = 0.0

        if abort_rts.size > 0:
            proactive_pdf_vals = _led_off_pdf_vec(abort_rts, V_A_base, theta_A, shift_total)
            lapse_pdf_vals = lapse_pdf(abort_rts, beta_lapse)
            mix_pdf = (1.0 - lapse_prob) * proactive_pdf_vals + lapse_prob * lapse_pdf_vals
            group_loglike += np.sum(np.log(np.clip(mix_pdf, LIKELIHOOD_EPS, None)))

        if cens_tstims.size > 0:
            proactive_surv_vals = _led_off_survival_vec(cens_tstims, V_A_base, theta_A, shift_total)
            lapse_surv_vals = lapse_survival(cens_tstims, beta_lapse)
            mix_surv = (1.0 - lapse_prob) * proactive_surv_vals + lapse_prob * lapse_surv_vals
            group_loglike += np.sum(np.log(np.clip(mix_surv, LIKELIHOOD_EPS, None)))

        return float(group_loglike)

    t_change = t_led_cond - del_a_minus_del_LED

    model = build_pyddm_model(
        V_A_base=V_A_base,
        theta_A=theta_A,
        bound_slope=bound_slope,
        theta_A_saturate=theta_A_saturate,
        t_change=t_change,
    )
    sol = model.solve()

    pdf_upper = sol.pdf("upper_hit")
    cdf_upper = np.cumsum(pdf_upper) * PYDDM_DT
    p_upper = sol.prob("upper_hit")

    group_loglike = 0.0

    # Abort trials: density term
    if abort_rts.size > 0:
        t_proc_abort = abort_rts - shift_total
        proactive_pdf_vals = _pdf_lookup_on_grid(pdf_upper, t_proc_abort, PYDDM_DT, PYDDM_T_DUR)
        lapse_pdf_vals = lapse_pdf(abort_rts, beta_lapse)
        mix_pdf = (1.0 - lapse_prob) * proactive_pdf_vals + lapse_prob * lapse_pdf_vals
        group_loglike += np.sum(np.log(np.clip(mix_pdf, LIKELIHOOD_EPS, None)))

    # Censored trials: survival term at t_stim
    if cens_tstims.size > 0:
        t_proc_cens = cens_tstims - shift_total
        proactive_surv_vals = _upper_survival_lookup(cdf_upper, p_upper, t_proc_cens, PYDDM_DT, PYDDM_T_DUR)
        lapse_surv_vals = lapse_survival(cens_tstims, beta_lapse)
        mix_surv = (1.0 - lapse_prob) * proactive_surv_vals + lapse_prob * lapse_surv_vals
        group_loglike += np.sum(np.log(np.clip(mix_surv, LIKELIHOOD_EPS, None)))

    return float(group_loglike)


def proactive_led_loglike(params):
    """
    Total log-likelihood over fit_df with right-censoring at t_stim.

    params = [
        V_A_base, theta_A, bound_slope, theta_A_saturate,
        del_a_minus_del_LED, del_m_plus_del_LED,
        lapse_prob, beta_lapse
    ]
    """
    (
        V_A_base,
        theta_A,
        bound_slope,
        theta_A_saturate,
        del_a_minus_del_LED,
        del_m_plus_del_LED,
        lapse_prob,
        beta_lapse,
    ) = params

    # Hard validity checks
    if theta_A <= THETA_A_MIN or theta_A >= THETA_E0_FIXED:
        return -np.inf
    if bound_slope < 0.01:
        return -np.inf
    if theta_A_saturate <= THETA_A_MIN or theta_A_saturate >= THETA_E0_FIXED:
        return -np.inf
    if not (0.0 <= lapse_prob <= 1.0):
        return -np.inf
    if beta_lapse <= 0:
        return -np.inf
    if theta_A_saturate > theta_A:
        return float(np.log(LIKELIHOOD_EPS))

    if N_JOBS == 1:
        ll_by_group = [
            _group_loglike(
                grp,
                V_A_base,
                theta_A,
                bound_slope,
                theta_A_saturate,
                del_a_minus_del_LED,
                del_m_plus_del_LED,
                lapse_prob,
                beta_lapse,
            )
            for grp in condition_groups
        ]
    else:
        ll_by_group = Parallel(n_jobs=N_JOBS, backend=PARALLEL_BACKEND)(
            delayed(_group_loglike)(
                grp,
                V_A_base,
                theta_A,
                bound_slope,
                theta_A_saturate,
                del_a_minus_del_LED,
                del_m_plus_del_LED,
                lapse_prob,
                beta_lapse,
            )
            for grp in condition_groups
        )

    return float(np.sum(ll_by_group))


# %%
# =============================================================================
# Priors and joint
# =============================================================================
V_A_base_bounds = [0.1, 5.0]
theta_A_bounds = [0.1, 5.5]
bound_slope_bounds = [0.01, 2.0]
theta_A_saturate_bounds = [0.1, 4.0]
del_a_minus_del_LED_bounds = [-1.1, 1.1]
del_m_plus_del_LED_bounds = [0.001, 0.2]
lapse_prob_bounds = [0.0, 1.0]
beta_lapse_bounds = [0.001, 20.0]

# #########################################3
V_A_base_plausible = [0.5, 3.0]

theta_A_plausible = [0.8, 4]
bound_slope_plausible = [0.02, 0.5]
theta_A_saturate_plausible = [0.5, 3.0]

del_a_minus_del_LED_plausible = [0.01, 0.07]
del_m_plus_del_LED_plausible = [0.01, 0.07]
lapse_prob_plausible = [0.01, 0.3]
beta_lapse_plausible = [0.5, 5.0]


def vbmc_prior_fn(params):
    (
        V_A_base,
        theta_A,
        bound_slope,
        theta_A_saturate,
        del_a_minus_del_LED,
        del_m_plus_del_LED,
        lapse_prob,
        beta_lapse,
    ) = params

    log_prior = 0.0
    log_prior += trapezoidal_logpdf(
        V_A_base,
        V_A_base_bounds[0],
        V_A_base_plausible[0],
        V_A_base_plausible[1],
        V_A_base_bounds[1],
    )
    log_prior += trapezoidal_logpdf(
        theta_A,
        theta_A_bounds[0],
        theta_A_plausible[0],
        theta_A_plausible[1],
        theta_A_bounds[1],
    )
    log_prior += trapezoidal_logpdf(
        bound_slope,
        bound_slope_bounds[0],
        bound_slope_plausible[0],
        bound_slope_plausible[1],
        bound_slope_bounds[1],
    )
    log_prior += trapezoidal_logpdf(
        theta_A_saturate,
        theta_A_saturate_bounds[0],
        theta_A_saturate_plausible[0],
        theta_A_saturate_plausible[1],
        theta_A_saturate_bounds[1],
    )
    log_prior += trapezoidal_logpdf(
        del_a_minus_del_LED,
        del_a_minus_del_LED_bounds[0],
        del_a_minus_del_LED_plausible[0],
        del_a_minus_del_LED_plausible[1],
        del_a_minus_del_LED_bounds[1],
    )
    log_prior += trapezoidal_logpdf(
        del_m_plus_del_LED,
        del_m_plus_del_LED_bounds[0],
        del_m_plus_del_LED_plausible[0],
        del_m_plus_del_LED_plausible[1],
        del_m_plus_del_LED_bounds[1],
    )
    log_prior += trapezoidal_logpdf(
        lapse_prob,
        lapse_prob_bounds[0],
        lapse_prob_plausible[0],
        lapse_prob_plausible[1],
        lapse_prob_bounds[1],
    )
    log_prior += trapezoidal_logpdf(
        beta_lapse,
        beta_lapse_bounds[0],
        beta_lapse_plausible[0],
        beta_lapse_plausible[1],
        beta_lapse_bounds[1],
    )

    return float(log_prior)


def vbmc_joint(params):
    log_prior = vbmc_prior_fn(params)
    if not np.isfinite(log_prior):
        return -np.inf

    log_like = proactive_led_loglike(params)
    if not np.isfinite(log_like):
        return -np.inf

    return log_prior + log_like


# %%
# =============================================================================
# VBMC setup
# =============================================================================
lb = np.array(
    [
        V_A_base_bounds[0],
        theta_A_bounds[0],
        bound_slope_bounds[0],
        theta_A_saturate_bounds[0],
        del_a_minus_del_LED_bounds[0],
        del_m_plus_del_LED_bounds[0],
        lapse_prob_bounds[0],
        beta_lapse_bounds[0],
    ]
)
ub = np.array(
    [
        V_A_base_bounds[1],
        theta_A_bounds[1],
        bound_slope_bounds[1],
        theta_A_saturate_bounds[1],
        del_a_minus_del_LED_bounds[1],
        del_m_plus_del_LED_bounds[1],
        lapse_prob_bounds[1],
        beta_lapse_bounds[1],
    ]
)
plb = np.array(
    [
        V_A_base_plausible[0],
        theta_A_plausible[0],
        bound_slope_plausible[0],
        theta_A_saturate_plausible[0],
        del_a_minus_del_LED_plausible[0],
        del_m_plus_del_LED_plausible[0],
        lapse_prob_plausible[0],
        beta_lapse_plausible[0],
    ]
)
pub = np.array(
    [
        V_A_base_plausible[1],
        theta_A_plausible[1],
        bound_slope_plausible[1],
        theta_A_saturate_plausible[1],
        del_a_minus_del_LED_plausible[1],
        del_m_plus_del_LED_plausible[1],
        lapse_prob_plausible[1],
        beta_lapse_plausible[1],
    ]
)

x_0 = np.array(
    [
        1.6,   # V_A_base
        2.5,   # theta_A
        0.5,   # bound_slope
        1.2,   # theta_A_saturate
        0.04,  # del_a_minus_del_LED
        0.05,  # del_m_plus_del_LED
        0.05,  # lapse_prob
        5.0,   # beta_lapse
    ]
)
x_0 = np.clip(x_0, plb, pub)

print("\nVBMC setup:")
print(f"  Initial point: {x_0}")
print(f"  Lower bounds: {lb}")
print(f"  Upper bounds: {ub}")
print(f"  Plausible lower: {plb}")
print(f"  Plausible upper: {pub}")


# %%
# =============================================================================
# Run or load VBMC
# =============================================================================
VP_PKL_PATH = f"vbmc_real_{file_tag}_fit_{MODEL_TAG}_PYDDM.pkl"
RESULTS_PKL_PATH = f"vbmc_real_{file_tag}_results_{MODEL_TAG}_PYDDM.pkl"

if LOAD_SAVED_RESULTS and os.path.exists(VP_PKL_PATH) and os.path.getsize(VP_PKL_PATH) > 0:
    print(f"\nLoading saved VBMC results from {VP_PKL_PATH}...")
    with open(VP_PKL_PATH, "rb") as f:
        vp = pickle.load(f)

    results_summary = {}
    if os.path.exists(RESULTS_PKL_PATH) and os.path.getsize(RESULTS_PKL_PATH) > 0:
        with open(RESULTS_PKL_PATH, "rb") as f:
            saved_results = pickle.load(f)
        results_summary = saved_results.get("results_summary", {})
    print("Loaded saved VBMC results.")
else:
    print(f"\nRunning VBMC optimization for {animal_label}...")
    vbmc = VBMC(vbmc_joint, x_0, lb, ub, plb, pub, options=VBMC_OPTIONS)
    vp, results = vbmc.optimize()

    print("\nVBMC optimization complete!")

    vp.save(VP_PKL_PATH, overwrite=True)

    results_summary = {
        "elbo": results.get("elbo", None),
        "elbo_sd": results.get("elbo_sd", None),
        "convergence_status": results.get("convergence_status", None),
    }
    with open(RESULTS_PKL_PATH, "wb") as f:
        pickle.dump(
            {
                "animals": unique_animals.tolist(),
                "animal_id": ANIMAL_ID,
                "results_summary": results_summary,
                "bounds": {"lb": lb, "ub": ub, "plb": plb, "pub": pub},
                "settings": {
                    "PYDDM_DT": PYDDM_DT,
                    "PYDDM_DX": PYDDM_DX,
                    "PYDDM_T_DUR": PYDDM_T_DUR,
                    "USE_EXACT_T_LED": USE_EXACT_T_LED,
                    "T_LED_BIN_WIDTH": T_LED_BIN_WIDTH,
                },
            },
            f,
        )
    print(f"Results saved for {animal_label}!")


# %%
# =============================================================================
# Posterior summary
# =============================================================================
vp_samples = vp.sample(int(1e5))[0]

param_labels = [
    "V_A_base",
    "theta_A",
    "bound_slope",
    "theta_A_saturate",
    "del_a_minus_del_LED",
    "del_m_plus_del_LED",
    "lapse_prob",
    "beta_lapse",
]
param_means = np.mean(vp_samples, axis=0)
param_stds = np.std(vp_samples, axis=0)

print(f"\nPosterior summary ({animal_label}):")
print(f"{'Parameter':<20} {'Mean':<12} {'Std':<12}")
print("-" * 50)
for i, label in enumerate(param_labels):
    print(f"{label:<20} {param_means[i]:<12.4f} {param_stds[i]:<12.4f}")

print(f"\nLog-likelihood at posterior mean: {proactive_led_loglike(param_means):.3f}")


# %%
# =============================================================================
# Corner plot
# =============================================================================
fig = corner.corner(
    vp_samples,
    labels=param_labels,
    show_titles=True,
    title_fmt=".4f",
    quantiles=[0.025, 0.5, 0.975],
)
plt.suptitle(f"{animal_label} Posterior", y=1.02)
corner_base = f"vbmc_real_{file_tag}_corner_{MODEL_TAG}_PYDDM"
savefig_both(corner_base, bbox_inches="tight")
print(f"Corner plot saved as '{corner_base}.pdf' and '{corner_base}.png'")
plt.show()


# %%
# =============================================================================
# Simulation with fitted params (for quick RTD sanity check)
# =============================================================================
def simulate_proactive_single_bound_bound_drop_saturating(
    V_A_base,
    theta_A,
    bound_slope,
    theta_A_saturate,
    t_led,
    del_a_minus_del_LED,
    del_m_plus_del_LED,
    is_led_trial,
    dt=1e-4,
):
    A = 0.0
    t = 0.0
    dB = np.sqrt(dt)
    shift_total = del_m_plus_del_LED + del_a_minus_del_LED

    if is_led_trial:
        t_change = t_led - del_a_minus_del_LED
    else:
        t_change = PYDDM_T_DUR + 10.0

    while True:
        v_t = V_A_base
        theta_t = max(
            THETA_A_MIN,
            max(theta_A_saturate, theta_A - bound_slope * max(0.0, t - t_change)),
        )

        A += v_t * dt + np.random.normal(0.0, dB)
        t += dt

        if A >= theta_t:
            return t + shift_total
        if t > PYDDM_T_DUR:
            return np.nan


def simulate_single_trial_fit():
    trial_idx = np.random.randint(len(fit_df))
    row = fit_df.iloc[trial_idx]

    is_led_trial = bool(row["LED_trial"] == 1)
    t_led = float(row["t_LED"])
    t_stim = float(row["t_stim"])

    lapse_prob = param_means[6]
    beta_lapse = param_means[7]

    if np.random.random() < lapse_prob:
        rt = np.random.exponential(1.0 / beta_lapse)
    else:
        rt = simulate_proactive_single_bound_bound_drop_saturating(
            param_means[0],
            param_means[1],
            param_means[2],
            param_means[3],
            t_led,
            param_means[4],
            param_means[5],
            is_led_trial,
            dt=SIM_DT,
        )

    return rt, is_led_trial, t_led, t_stim


def _chunk_sizes(n_total, n_jobs):
    n_jobs_eff = max(1, min(int(n_jobs), int(n_total)))
    chunks = np.full(n_jobs_eff, n_total // n_jobs_eff, dtype=int)
    chunks[: n_total % n_jobs_eff] += 1
    return [int(c) for c in chunks if c > 0]


def _simulate_chunk_basic(n_chunk):
    return [simulate_single_trial_fit() for _ in range(n_chunk)]


print(f"\nSimulating {N_TRIALS_SIM} trials with fitted params...")
if N_JOBS == 1:
    sim_results = _simulate_chunk_basic(N_TRIALS_SIM)
else:
    sim_chunks = Parallel(n_jobs=min(N_JOBS, N_TRIALS_SIM), backend=PARALLEL_BACKEND)(
        delayed(_simulate_chunk_basic)(n_chunk) for n_chunk in _chunk_sizes(N_TRIALS_SIM, N_JOBS)
    )
    sim_results = [item for chunk in sim_chunks for item in chunk]

sim_rts = [r[0] for r in sim_results]
sim_led = [r[1] for r in sim_results]
sim_t_stim = [r[3] for r in sim_results]

sim_rts_on = [rt for rt, is_led, t_stim in zip(sim_rts, sim_led, sim_t_stim) if is_led and np.isfinite(rt) and rt < t_stim]
sim_rts_off = [rt for rt, is_led, t_stim in zip(sim_rts, sim_led, sim_t_stim) if (not is_led) and np.isfinite(rt) and rt < t_stim]

data_rts_on = fit_df[(fit_df["LED_trial"] == 1) & (fit_df["RT"] < fit_df["t_stim"])]["RT"].values
data_rts_off = fit_df[(fit_df["LED_trial"] == 0) & (fit_df["RT"] < fit_df["t_stim"])]["RT"].values

print(f"Sim abort counts: LED ON={len(sim_rts_on)}, LED OFF={len(sim_rts_off)}")
print(f"Data abort counts: LED ON={len(data_rts_on)}, LED OFF={len(data_rts_off)}")


# %%
# =============================================================================
# Plot: RTD wrt fixation (data vs sim, aborts only)
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

bins = np.arange(0, 3.0, 0.05)
centers = (bins[1:] + bins[:-1]) / 2

data_hist_on, _ = np.histogram(data_rts_on, bins=bins, density=True)
sim_hist_on, _ = np.histogram(sim_rts_on, bins=bins, density=True)

axes[0].plot(centers, data_hist_on, label="Data (aborts)", lw=2, alpha=0.8)
axes[0].plot(centers, sim_hist_on, label="Sim (fitted)", lw=2, alpha=0.8)
axes[0].set_xlabel("RT (s)")
axes[0].set_ylabel("Density")
axes[0].set_title(f"LED ON - {animal_label}")
axes[0].legend()

data_hist_off, _ = np.histogram(data_rts_off, bins=bins, density=True)
sim_hist_off, _ = np.histogram(sim_rts_off, bins=bins, density=True)

axes[1].plot(centers, data_hist_off, label="Data (aborts)", lw=2, alpha=0.8)
axes[1].plot(centers, sim_hist_off, label="Sim (fitted)", lw=2, alpha=0.8)
axes[1].set_xlabel("RT (s)")
axes[1].set_ylabel("Density")
axes[1].set_title(f"LED OFF - {animal_label}")
axes[1].legend()

plt.tight_layout()
rtd_base = f"vbmc_real_{file_tag}_rtd_data_vs_sim_{MODEL_TAG}_PYDDM"
savefig_both(rtd_base, bbox_inches="tight")
print(f"RTD data-vs-sim plot saved as '{rtd_base}.pdf' and '{rtd_base}.png'")
plt.show()

# %%

# =============================================================================
# Plot: RTD wrt LED (area-weighted by abort fraction)
# =============================================================================
def _safe_hist_density(vals, bins):
    if len(vals) == 0:
        return np.zeros(len(bins) - 1, dtype=float)
    hist, _ = np.histogram(vals, bins=bins, density=True)
    return np.nan_to_num(hist, nan=0.0, posinf=0.0, neginf=0.0)

# Data abort RTs wrt LED
df_on_aborts = fit_df[(fit_df["LED_trial"] == 1) & (fit_df["RT"] < fit_df["t_stim"])]
df_off_aborts = fit_df[(fit_df["LED_trial"] == 0) & (fit_df["RT"] < fit_df["t_stim"])]
data_rts_wrt_led_on = (df_on_aborts["RT"] - df_on_aborts["t_LED"]).values
data_rts_wrt_led_off = (df_off_aborts["RT"] - df_off_aborts["t_LED"]).values

# Sim abort RTs wrt LED
sim_rts_wrt_led_on = [
    rt - t_led
    for rt, is_led, t_led, t_stim in sim_results
    if is_led and np.isfinite(rt) and rt < t_stim
]
sim_rts_wrt_led_off = [
    rt - t_led
    for rt, is_led, t_led, t_stim in sim_results
    if (not is_led) and np.isfinite(rt) and rt < t_stim
]

bins_wrt_led = np.arange(-3.0, 3.0, 0.005)
bin_centers_wrt_led = (bins_wrt_led[1:] + bins_wrt_led[:-1]) / 2

# Abort fractions (area targets)
n_all_data_on = int(np.sum(fit_df["LED_trial"] == 1))
n_all_data_off = int(np.sum(fit_df["LED_trial"] == 0))
n_aborts_data_on = len(data_rts_wrt_led_on)
n_aborts_data_off = len(data_rts_wrt_led_off)
frac_data_on = n_aborts_data_on / n_all_data_on if n_all_data_on > 0 else 0.0
frac_data_off = n_aborts_data_off / n_all_data_off if n_all_data_off > 0 else 0.0

n_all_sim_on = sum(1 for _, is_led, _, _ in sim_results if is_led)
n_all_sim_off = sum(1 for _, is_led, _, _ in sim_results if not is_led)
n_aborts_sim_on = len(sim_rts_wrt_led_on)
n_aborts_sim_off = len(sim_rts_wrt_led_off)
frac_sim_on = n_aborts_sim_on / n_all_sim_on if n_all_sim_on > 0 else 0.0
frac_sim_off = n_aborts_sim_off / n_all_sim_off if n_all_sim_off > 0 else 0.0

data_hist_on_scaled = _safe_hist_density(data_rts_wrt_led_on, bins_wrt_led) * frac_data_on
data_hist_off_scaled = _safe_hist_density(data_rts_wrt_led_off, bins_wrt_led) * frac_data_off
sim_hist_on_scaled = _safe_hist_density(sim_rts_wrt_led_on, bins_wrt_led) * frac_sim_on
sim_hist_off_scaled = _safe_hist_density(sim_rts_wrt_led_off, bins_wrt_led) * frac_sim_off

fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(
    bin_centers_wrt_led,
    data_hist_on_scaled,
    label=f"Data LED ON (frac={frac_data_on:.2f})",
    lw=2,
    alpha=0.8,
    color="r",
    linestyle="-",
)
ax.plot(
    bin_centers_wrt_led,
    data_hist_off_scaled,
    label=f"Data LED OFF (frac={frac_data_off:.2f})",
    lw=2,
    alpha=0.8,
    color="b",
    linestyle="-",
)
ax.plot(
    bin_centers_wrt_led,
    sim_hist_on_scaled,
    label=f"Sim LED ON (frac={frac_sim_on:.2f})",
    lw=2,
    alpha=0.8,
    color="r",
    linestyle="--",
)
ax.plot(
    bin_centers_wrt_led,
    sim_hist_off_scaled,
    label=f"Sim LED OFF (frac={frac_sim_off:.2f})",
    lw=2,
    alpha=0.8,
    color="b",
    linestyle="--",
)

ax.axvline(x=0.0, color="k", linestyle="--", alpha=0.5, label="LED onset")
ax.axvline(
    x=param_means[5],
    color="g",
    linestyle=":",
    alpha=0.6,
    label=f"del_m_plus_del_LED={param_means[5]:.3f}",
)
ax.set_xlabel("RT - t_LED (s)")
ax.set_ylabel("Rate (area = fraction)")
ax.set_title(f"RT wrt LED (area-weighted) - {animal_label}")
ax.legend(fontsize=9)
# ax.set_xlim(-0.5, 0.4)
ax.set_xlim(-0.2, 0.3)


plt.tight_layout()
led_rate_base = f"vbmc_real_{file_tag}_rt_wrt_led_rate_{MODEL_TAG}_PYDDM"
savefig_both(led_rate_base, bbox_inches="tight")
print(
    "RT wrt LED rate plot saved as "
    f"'{led_rate_base}.pdf' and '{led_rate_base}.png'"
)
plt.show()

# %%
# =============================================================================
# Plot: Theoretical RTD wrt LED (area-weighted, smooth theory vs data)
# =============================================================================
N_THEORY_SAMPLES = 1000
t_pts_wrt_led_theory = np.arange(-1.0, 1.0, 0.001)

# Fitted params
V_A_base_fit      = param_means[0]
theta_A_fit       = param_means[1]
bound_slope_fit   = param_means[2]
theta_A_sat_fit   = param_means[3]
del_a_minus_fit   = param_means[4]
del_m_plus_fit    = param_means[5]
lapse_prob_fit    = param_means[6]
beta_lapse_fit    = param_means[7]
shift_total_fit   = del_a_minus_fit + del_m_plus_fit

# ---- LED ON theory ----
df_on_all = fit_df[fit_df["LED_trial"] == 1]
sampled_on = df_on_all.sample(n=N_THEORY_SAMPLES, replace=True, random_state=42)
sampled_on_tled = sampled_on["t_LED"].values
sampled_on_tstim = sampled_on["t_stim"].values

# Bin t_LED the same way as in fitting to reuse models
if USE_EXACT_T_LED:
    sampled_on_tled_bin = sampled_on_tled.copy()
else:
    sampled_on_tled_bin = np.round(sampled_on_tled / T_LED_BIN_WIDTH) * T_LED_BIN_WIDTH

# Solve PyDDM once per unique bin
unique_bins_on = np.unique(sampled_on_tled_bin)
print(f"\nTheory wrt LED: solving {len(unique_bins_on)} LED ON conditions ...")
pdf_cache_on = {}
for tled_bin in unique_bins_on:
    t_change = tled_bin - del_a_minus_fit
    model = build_pyddm_model(V_A_base_fit, theta_A_fit, bound_slope_fit, theta_A_sat_fit, t_change)
    sol = model.solve()
    pdf_cache_on[tled_bin] = sol.pdf("upper_hit")

# Accumulate
theory_pdf_on = np.zeros_like(t_pts_wrt_led_theory)
for i in range(N_THEORY_SAMPLES):
    t_LED_i  = sampled_on_tled[i]
    t_stim_i = sampled_on_tstim[i]
    pdf_upper = pdf_cache_on[sampled_on_tled_bin[i]]

    t_abs  = t_pts_wrt_led_theory + t_LED_i
    t_proc = t_abs - shift_total_fit

    pro_vals = _pdf_lookup_on_grid(pdf_upper, t_proc, PYDDM_DT, PYDDM_T_DUR)
    lap_vals = lapse_pdf(np.clip(t_abs, 0, None), beta_lapse_fit)
    lap_vals[t_abs < 0] = 0.0

    mix = (1.0 - lapse_prob_fit) * pro_vals + lapse_prob_fit * lap_vals
    mix[(t_abs < 0) | (t_abs >= t_stim_i)] = 0.0
    theory_pdf_on += mix

theory_pdf_on /= N_THEORY_SAMPLES

# ---- LED OFF theory ----
df_off_all = fit_df[fit_df["LED_trial"] == 0]
sampled_off = df_off_all.sample(n=N_THEORY_SAMPLES, replace=True, random_state=42)
sampled_off_tled = sampled_off["t_LED"].values
sampled_off_tstim = sampled_off["t_stim"].values

# Single model for OFF (no LED effect)
t_change_off = PYDDM_T_DUR + 10.0
model_off = build_pyddm_model(V_A_base_fit, theta_A_fit, bound_slope_fit, theta_A_sat_fit, t_change_off)
sol_off = model_off.solve()
pdf_upper_off = sol_off.pdf("upper_hit")
print("Theory wrt LED: solved LED OFF condition.")

theory_pdf_off = np.zeros_like(t_pts_wrt_led_theory)
for i in range(N_THEORY_SAMPLES):
    t_LED_i  = sampled_off_tled[i]
    t_stim_i = sampled_off_tstim[i]

    t_abs  = t_pts_wrt_led_theory + t_LED_i
    t_proc = t_abs - shift_total_fit

    pro_vals = _pdf_lookup_on_grid(pdf_upper_off, t_proc, PYDDM_DT, PYDDM_T_DUR)
    lap_vals = lapse_pdf(np.clip(t_abs, 0, None), beta_lapse_fit)
    lap_vals[t_abs < 0] = 0.0

    mix = (1.0 - lapse_prob_fit) * pro_vals + lapse_prob_fit * lap_vals
    mix[(t_abs < 0) | (t_abs >= t_stim_i)] = 0.0
    theory_pdf_off += mix

theory_pdf_off /= N_THEORY_SAMPLES

# ---- Plot: data histogram vs theory ----
fig, ax = plt.subplots(figsize=(15, 6))

ax.plot(
    bin_centers_wrt_led, data_hist_on_scaled,
    label=f"Data LED ON (frac={frac_data_on:.2f})",
    lw=2, alpha=0.8, color="r", linestyle="-",
)
ax.plot(
    bin_centers_wrt_led, data_hist_off_scaled,
    label=f"Data LED OFF (frac={frac_data_off:.2f})",
    lw=2, alpha=0.8, color="b", linestyle="-",
)
ax.plot(
    t_pts_wrt_led_theory, theory_pdf_on,
    label="Theory LED ON", lw=2, alpha=0.8, color="r", linestyle="--",
)
ax.plot(
    t_pts_wrt_led_theory, theory_pdf_off,
    label="Theory LED OFF", lw=2, alpha=0.8, color="b", linestyle="--",
)

ax.axvline(x=0.0, color="k", linestyle="--", alpha=0.5, label="LED onset")
ax.axvline(
    x=param_means[5], color="g", linestyle=":", alpha=0.6,
    label=f"del_m_plus_del_LED={param_means[5]:.3f}",
)
ax.set_xlabel("RT - t_LED (s)")
ax.set_ylabel("Rate (area = fraction)")
ax.set_title(f"RT wrt LED (Theory vs Data, area-weighted) - {animal_label}")
ax.legend(fontsize=9)
ax.set_xlim(-0.2, 0.3)

plt.tight_layout()
theory_base = f"vbmc_real_{file_tag}_rt_wrt_led_THEORY_{MODEL_TAG}_PYDDM"
savefig_both(theory_base, bbox_inches="tight")
print(
    "Theory RT wrt LED plot saved as "
    f"'{theory_base}.pdf' and '{theory_base}.png'"
)
plt.show()

# %%
# =============================================================================
# Plot: Drift and Bound over time (wrt LED onset)
# =============================================================================
# Time grid centred on t_change = 0 (ignoring delays)
t_grid = np.linspace(-1.0, 1.0, 2000)

drift_on = np.array([
    V_A_base_fit
    for t in t_grid
])
bound_on = np.array([
    max(THETA_A_MIN, max(theta_A_sat_fit, theta_A_fit - bound_slope_fit * max(0.0, t)))
    for t in t_grid
])

drift_off = np.full_like(t_grid, V_A_base_fit)
bound_off = np.full_like(t_grid, theta_A_fit)

xlims = [-0.4,0.4]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# -- Drift --
axes[0].plot(t_grid, drift_on, color="r", lw=2, label="LED ON")
axes[0].plot(t_grid, drift_off, color="b", lw=2, label="LED OFF")
axes[0].axvline(0.0, color="k", ls="--", alpha=0.5, label="t_change")
axes[0].set_xlabel("Time wrt t_change (s)")
axes[0].set_ylabel("Drift rate")
axes[0].set_title(f"Drift over time - {animal_label}")
axes[0].legend(fontsize=9)
axes[0].set_xlim(xlims)
# -- Bound (distance to upper) --
axes[1].plot(t_grid, bound_on, color="r", lw=2, label="LED ON")
axes[1].plot(t_grid, bound_off, color="b", lw=2, label="LED OFF")
axes[1].axvline(0.0, color="k", ls="--", alpha=0.5, label="t_change")
axes[1].set_xlabel("Time wrt t_change (s)")
axes[1].set_ylabel("Bound (θ_A)")
axes[1].set_title(f"Bound over time - {animal_label}")
axes[1].legend(fontsize=9)
axes[1].set_xlim(xlims)
plt.tight_layout()
drift_bound_base = f"vbmc_real_{file_tag}_drift_bound_over_time_{MODEL_TAG}_PYDDM"
savefig_both(drift_bound_base, bbox_inches="tight")
print(
    "Drift & bound plot saved as "
    f"'{drift_bound_base}.pdf' and '{drift_bound_base}.png'"
)
plt.show()

# %%
# =============================================================================
# Compare likelihood on random LED ON subset: binned t_LED vs exact t_LED
# =============================================================================
N_COMPARE_LED_ON = 1000
COMPARE_RANDOM_SEED = 42

df_on_all = fit_df[fit_df["LED_trial"] == 1]
n_on_sample = min(N_COMPARE_LED_ON, len(df_on_all))
if n_on_sample < N_COMPARE_LED_ON:
    print(
        f"Only {n_on_sample} LED ON trials available; "
        f"using all available trials instead of {N_COMPARE_LED_ON}."
    )

sampled_on_df = df_on_all.sample(
    n=n_on_sample, replace=False, random_state=COMPARE_RANDOM_SEED
).copy()

print("\nLikelihood comparison on random LED ON subset:")
print(f"  Requested LED ON trials: {N_COMPARE_LED_ON}")
print(f"  Used LED ON trials:      {n_on_sample}")
print(f"  Random seed:             {COMPARE_RANDOM_SEED}")


def _build_led_on_condition_groups(df_led_on, use_exact_t_led):
    if use_exact_t_led:
        t_led_cond_vals = df_led_on["t_LED"].values.astype(float)
    else:
        t_led_cond_vals = np.round(df_led_on["t_LED"].values / T_LED_BIN_WIDTH) * T_LED_BIN_WIDTH

    df_tmp = df_led_on.copy()
    df_tmp["t_led_cond_eval"] = t_led_cond_vals

    eval_condition_groups = []
    for t_led_cond, grp in df_tmp.groupby("t_led_cond_eval", sort=True):
        abort_rts = grp.loc[grp["is_abort"], "RT"].values.astype(float)
        cens_tstims = grp.loc[~grp["is_abort"], "t_stim"].values.astype(float)
        eval_condition_groups.append(
            {
                "is_led": 1,
                "t_led_cond": float(t_led_cond),
                "abort_rts": abort_rts,
                "cens_tstims": cens_tstims,
                "n_trials": len(grp),
                "n_abort": len(abort_rts),
                "n_cens": len(cens_tstims),
            }
        )

    return eval_condition_groups


def _sum_loglike_over_groups(eval_condition_groups, params):
    (
        V_A_base_m,
        theta_A_m,
        bound_slope_m,
        theta_A_saturate_m,
        del_a_m,
        del_m_m,
        lapse_prob_m,
        beta_lapse_m,
    ) = params

    if N_JOBS == 1 or len(eval_condition_groups) <= 1:
        ll_parts = [
            _group_loglike(
                grp,
                V_A_base_m,
                theta_A_m,
                bound_slope_m,
                theta_A_saturate_m,
                del_a_m,
                del_m_m,
                lapse_prob_m,
                beta_lapse_m,
            )
            for grp in eval_condition_groups
        ]
    else:
        n_jobs_eff = min(N_JOBS, len(eval_condition_groups))
        ll_parts = Parallel(n_jobs=n_jobs_eff, backend=PARALLEL_BACKEND)(
            delayed(_group_loglike)(
                grp,
                V_A_base_m,
                theta_A_m,
                bound_slope_m,
                theta_A_saturate_m,
                del_a_m,
                del_m_m,
                lapse_prob_m,
                beta_lapse_m,
            )
            for grp in eval_condition_groups
        )

    return float(np.sum(ll_parts))


t0 = time.perf_counter()
binned_condition_groups = _build_led_on_condition_groups(sampled_on_df, use_exact_t_led=False)
ll_binned = _sum_loglike_over_groups(binned_condition_groups, param_means)
binned_elapsed_s = time.perf_counter() - t0

t0 = time.perf_counter()
exact_condition_groups = _build_led_on_condition_groups(sampled_on_df, use_exact_t_led=True)
ll_exact = _sum_loglike_over_groups(exact_condition_groups, param_means)
exact_elapsed_s = time.perf_counter() - t0

ll_diff = ll_binned - ll_exact
rel_diff_pct = abs(ll_diff) / max(abs(ll_exact), 1e-12) * 100.0

print(f"\nBinned t_LED groups (width={T_LED_BIN_WIDTH:.3f}s): {len(binned_condition_groups)}")
print(f"Exact t_LED groups:                                {len(exact_condition_groups)}")
print(f"Log-likelihood (binned t_LED):                     {ll_binned:.4f}")
print(f"Time taken (binned):                               {binned_elapsed_s:.3f} s")
print(f"Log-likelihood (exact  t_LED):                     {ll_exact:.4f}")
print(f"Time taken (exact):                                {exact_elapsed_s:.3f} s")
print(f"Difference (binned - exact):                       {ll_diff:.4f}")
print(f"Relative difference:                               {rel_diff_pct:.4f}%")

# %%
