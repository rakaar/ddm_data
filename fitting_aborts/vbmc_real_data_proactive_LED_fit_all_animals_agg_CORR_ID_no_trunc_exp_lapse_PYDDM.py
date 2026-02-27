"""
VBMC fit of proactive model using PyDDM likelihood.

Model summary:
- Proactive process is approximated by PyDDM with:
  1) drift increasing linearly after LED effect time
  2) bound decreasing linearly after LED effect time
- Lapse process is exponential and mixed with proactive process.
- Trials are right-censored at t_stim (same censoring logic as before).

Notes:
- Uses t_LED bins for computational tractability (exact per-trial t_LED is too expensive).
- Uses PyDDM FP solver directly for likelihood used by VBMC.
"""

# %%
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyddm
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
USE_EXACT_T_LED = True
T_LED_BIN_WIDTH = 0.001

# Mapping single-bound -> PyDDM two-bound
THETA_E0_FIXED = 20.0     # must stay greater than theta_A
THETA_A_MIN = 0.05       # minimum upper distance to avoid bound crossing start
LIKELIHOOD_EPS = 1e-50

# Runtime controls
LOAD_SAVED_RESULTS = False
VBMC_OPTIONS = {"display": "iter"}

# Plot/simulation controls
SIM_DT = 1e-4
N_TRIALS_SIM = int(100e3)


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
def build_pyddm_model(V_A_base, drift_slope, theta_A, bound_slope, t_change):
    """
    Build a PyDDM model for one condition (one t_change).

    - Drift: V_A_base + drift_slope * max(0, t - t_change)
    - Bound distance to upper: theta_A - bound_slope * max(0, t - t_change), clipped at THETA_A_MIN
    """
    x = 1.0 - (theta_A / THETA_E0_FIXED)
    x_abs = x * THETA_E0_FIXED
    theta_E_min = x_abs + THETA_A_MIN

    def drift_fn(t):
        return V_A_base + drift_slope * max(0.0, t - t_change)

    def bound_fn(t):
        B_t = THETA_E0_FIXED - bound_slope * max(0.0, t - t_change)
        return max(theta_E_min, B_t)

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


def proactive_led_loglike(params):
    """
    Total log-likelihood over fit_df with right-censoring at t_stim.

    params = [
        V_A_base, drift_slope, theta_A, bound_slope,
        del_a_minus_del_LED, del_m_plus_del_LED,
        lapse_prob, beta_lapse
    ]
    """
    (
        V_A_base,
        drift_slope,
        theta_A,
        bound_slope,
        del_a_minus_del_LED,
        del_m_plus_del_LED,
        lapse_prob,
        beta_lapse,
    ) = params

    # Hard validity checks
    if theta_A <= THETA_A_MIN or theta_A >= THETA_E0_FIXED:
        return -np.inf
    if drift_slope < 0 or bound_slope < 0:
        return -np.inf
    if not (0.0 <= lapse_prob <= 1.0):
        return -np.inf
    if beta_lapse <= 0:
        return -np.inf

    # Observed RT = process_time + shift_total
    shift_total = del_a_minus_del_LED + del_m_plus_del_LED

    total_loglike = 0.0

    for grp in condition_groups:
        is_led = grp["is_led"]
        t_led_cond = grp["t_led_cond"]
        abort_rts = grp["abort_rts"]
        cens_tstims = grp["cens_tstims"]

        if is_led == 1:
            t_change = t_led_cond - del_a_minus_del_LED
        else:
            # no LED effect for OFF trials
            t_change = PYDDM_T_DUR + 10.0

        model = build_pyddm_model(
            V_A_base=V_A_base,
            drift_slope=drift_slope,
            theta_A=theta_A,
            bound_slope=bound_slope,
            t_change=t_change,
        )
        sol = model.solve()

        pdf_upper = sol.pdf("upper_hit")
        cdf_upper = np.cumsum(pdf_upper) * PYDDM_DT
        p_upper = sol.prob("upper_hit")

        # Abort trials: density term
        if abort_rts.size > 0:
            t_proc_abort = abort_rts - shift_total
            proactive_pdf_vals = _pdf_lookup_on_grid(pdf_upper, t_proc_abort, PYDDM_DT, PYDDM_T_DUR)
            lapse_pdf_vals = lapse_pdf(abort_rts, beta_lapse)
            mix_pdf = (1.0 - lapse_prob) * proactive_pdf_vals + lapse_prob * lapse_pdf_vals
            total_loglike += np.sum(np.log(np.clip(mix_pdf, LIKELIHOOD_EPS, None)))

        # Censored trials: survival term at t_stim
        if cens_tstims.size > 0:
            t_proc_cens = cens_tstims - shift_total
            proactive_surv_vals = _upper_survival_lookup(cdf_upper, p_upper, t_proc_cens, PYDDM_DT, PYDDM_T_DUR)
            lapse_surv_vals = lapse_survival(cens_tstims, beta_lapse)
            mix_surv = (1.0 - lapse_prob) * proactive_surv_vals + lapse_prob * lapse_surv_vals
            total_loglike += np.sum(np.log(np.clip(mix_surv, LIKELIHOOD_EPS, None)))

    return float(total_loglike)


# %%
# =============================================================================
# Priors and joint
# =============================================================================
V_A_base_bounds = [0.1, 5.0]
drift_slope_bounds = [0.0, 0.5]
theta_A_bounds = [0.1, 5.5]
bound_slope_bounds = [0.0, 0.5]
del_a_minus_del_LED_bounds = [-1.1, 1.1]
del_m_plus_del_LED_bounds = [0.001, 0.2]
lapse_prob_bounds = [0.0, 1.0]
beta_lapse_bounds = [0.001, 20.0]

# #########################################3
V_A_base_plausible = [0.5, 3.0]
drift_slope_plausible = [0.05, 0.3]

theta_A_plausible = [0.8, 4]
bound_slope_plausible = [0.01, 0.5]

del_a_minus_del_LED_plausible = [0.01, 0.07]
del_m_plus_del_LED_plausible = [0.01, 0.07]
lapse_prob_plausible = [0.01, 0.3]
beta_lapse_plausible = [0.5, 5.0]


def vbmc_prior_fn(params):
    (
        V_A_base,
        drift_slope,
        theta_A,
        bound_slope,
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
        drift_slope,
        drift_slope_bounds[0],
        drift_slope_plausible[0],
        drift_slope_plausible[1],
        drift_slope_bounds[1],
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
        drift_slope_bounds[0],
        theta_A_bounds[0],
        bound_slope_bounds[0],
        del_a_minus_del_LED_bounds[0],
        del_m_plus_del_LED_bounds[0],
        lapse_prob_bounds[0],
        beta_lapse_bounds[0],
    ]
)
ub = np.array(
    [
        V_A_base_bounds[1],
        drift_slope_bounds[1],
        theta_A_bounds[1],
        bound_slope_bounds[1],
        del_a_minus_del_LED_bounds[1],
        del_m_plus_del_LED_bounds[1],
        lapse_prob_bounds[1],
        beta_lapse_bounds[1],
    ]
)
plb = np.array(
    [
        V_A_base_plausible[0],
        drift_slope_plausible[0],
        theta_A_plausible[0],
        bound_slope_plausible[0],
        del_a_minus_del_LED_plausible[0],
        del_m_plus_del_LED_plausible[0],
        lapse_prob_plausible[0],
        beta_lapse_plausible[0],
    ]
)
pub = np.array(
    [
        V_A_base_plausible[1],
        drift_slope_plausible[1],
        theta_A_plausible[1],
        bound_slope_plausible[1],
        del_a_minus_del_LED_plausible[1],
        del_m_plus_del_LED_plausible[1],
        lapse_prob_plausible[1],
        beta_lapse_plausible[1],
    ]
)

x_0 = np.array(
    [
        1.6,   # V_A_base
        0.6,   # drift_slope
        2.5,   # theta_A
        0.5,   # bound_slope
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
VP_PKL_PATH = f"vbmc_real_{file_tag}_fit_drift_up_bound_drop_PYDDM.pkl"
RESULTS_PKL_PATH = f"vbmc_real_{file_tag}_results_drift_up_bound_drop_PYDDM.pkl"

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
    "drift_slope",
    "theta_A",
    "bound_slope",
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
plt.savefig(f"vbmc_real_{file_tag}_corner_drift_up_bound_drop_PYDDM.pdf", bbox_inches="tight")
print("Corner plot saved.")
plt.show()


# %%
# =============================================================================
# Simulation with fitted params (for quick RTD sanity check)
# =============================================================================
def simulate_proactive_single_bound_drift_up_bound_drop(
    V_A_base,
    drift_slope,
    theta_A,
    bound_slope,
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
        v_t = V_A_base + drift_slope * max(0.0, t - t_change)
        theta_t = max(THETA_A_MIN, theta_A - bound_slope * max(0.0, t - t_change))

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
        rt = simulate_proactive_single_bound_drift_up_bound_drop(
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

    return rt, is_led_trial, t_stim


print(f"\nSimulating {N_TRIALS_SIM} trials with fitted params...")
sim_results = [simulate_single_trial_fit() for _ in range(N_TRIALS_SIM)]

sim_rts = [r[0] for r in sim_results]
sim_led = [r[1] for r in sim_results]
sim_t_stim = [r[2] for r in sim_results]

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
plt.savefig(f"vbmc_real_{file_tag}_rtd_data_vs_sim_drift_up_bound_drop_PYDDM.pdf", bbox_inches="tight")
print("RTD data-vs-sim plot saved.")
plt.show()
