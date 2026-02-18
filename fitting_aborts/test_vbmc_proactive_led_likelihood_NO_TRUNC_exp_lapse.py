# %%
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
import sys
sys.path.append('../fit_each_condn')
from psiam_tied_dv_map_utils_with_PDFs import stupid_f_integral, d_A_RT
from post_LED_censor_utils import cum_A_t_fn

# %%
# Load data to get LED and stimulus timing distributions
og_df = pd.read_csv('../out_LED.csv')
df = og_df[ og_df['repeat_trial'].isin([0,2]) | og_df['repeat_trial'].isna() ]
session_type = 7    
df = df[ df['session_type'].isin([session_type]) ]
training_level = 16
df = df[ df['training_level'].isin([training_level]) ]

# drop rows from df where intended_fix, LED_onset_time and timed_fix are nan
df = df.dropna(subset=['intended_fix', 'LED_onset_time', 'timed_fix'])
df = df[(df['abort_event'] == 3) | (df['success'].isin([1,-1]))]

# %%

# %%
# Get LED ON and OFF trials
df_on = df[df['LED_trial'] == 1]
df_on_1 = df_on.copy()
df_on_1['LED_wrt_fix'] = df_on_1['intended_fix'] - df_on_1['LED_onset_time']

df_off = df[df['LED_trial'] == 0]

# %%
# Plot LED timing distributions
bins = np.arange(0,2,0.05)
plt.hist(df_on_1['LED_wrt_fix'], bins=bins, histtype='step', density=True, label='LED time wrt fix')
plt.hist(df_on_1['intended_fix'], bins=bins, histtype='step', density=True, label='stim time wrt fix')
plt.xlabel('Time (s)')
plt.ylabel('Density')
plt.legend()
plt.title('LED and Stimulus Timing Distributions from Data')
plt.show()

# %%
# Simulate only proactive process (single bound)
def simulate_proactive_single_bound(V_A_base, V_A_post_LED, theta_A, t_LED, t_stim, t_aff, t_effect, motor_delay, is_led_trial, dt=1e-4):
    """
    Simulate proactive process with single bound accumulator.
    Drift changes from V_A_base to V_A_post_LED at t_LED + t_effect (only for LED ON trials).
    Proactive process starts at t = t_aff (no noise before this).
    Returns RT when accumulator hits theta_A.
    """
    AI = 0
    t = t_aff
    dB = np.sqrt(dt)

    while True:
        if is_led_trial and t >= t_LED + t_effect:
            V_A = V_A_post_LED
        else:
            V_A = V_A_base

        AI += V_A * dt + np.random.normal(0, dB)
        t += dt

        if AI >= theta_A:
            RT = t + motor_delay
            return RT

# %%
# Parameters for simulation
V_A_base = 1.8
V_A_post_LED = 3.4
theta_A = 2.5
t_aff =  10*1e-3
t_effect = 20*1e-3
motor_delay = 35*1e-3
lapse_prob = 0.3
beta_lapse = 3.0
N_sim = int(50e3)

# Get LED times from data
LED_times = df_on_1['LED_wrt_fix'].values
stim_times = df_on_1['intended_fix'].values
stim_times_off = df_off['intended_fix'].values

print(f"Number of LED ON trials in data: {len(df_on_1)}")
print(f"Number of LED OFF trials in data: {len(df_off)}")
print(f"Simulating {N_sim} trials...")

# %%
# Run simulation
def simulate_single_trial():
    # 1/3 chance of LED trial
    is_led_trial = np.random.random() < 1/2
    # Sample trial index to preserve (t_LED, t_stim) correlation
    trial_idx = np.random.randint(len(LED_times))
    t_LED = LED_times[trial_idx]
    t_stim = stim_times[trial_idx]

    is_lapse = np.random.random() < lapse_prob
    if is_lapse:
        # np.random.exp treats the input as 1/beta, not beta
        rt = np.random.exponential(1.0 / beta_lapse)
    else:
        rt = simulate_proactive_single_bound(
            V_A_base, V_A_post_LED, theta_A,
            t_LED if is_led_trial else None,
            t_stim,
            t_aff,
            t_effect,
            motor_delay,
            is_led_trial
        )
    return rt, is_led_trial, t_stim, t_LED, is_lapse

sim_results = Parallel(n_jobs=30)(
    delayed(simulate_single_trial)() for _ in tqdm(range(N_sim))
)
sim_rts = [r[0] for r in sim_results]
sim_is_led_trials = [r[1] for r in sim_results]
sim_t_stims = [r[2] for r in sim_results]
sim_t_LEDs = [r[3] for r in sim_results]
sim_is_lapse = [r[4] for r in sim_results]

# %%
# VBMC part
# 1. loglike
def PA_with_LEDON_2(t, v, vON, a, t_aff, motor_delay, tled, t_effect):
    """
    Compute the PA pdf by combining contributions before and after LED onset.
    """
    if (t - motor_delay) <= (tled + t_effect):
        pdf = d_A_RT(v * a, (t - motor_delay - t_aff) / (a**2)) / (a**2)
    else:
        t_post_led = t - motor_delay - tled - t_effect
        tp = tled + t_effect - t_aff

        if tp <= 0:
            pdf = d_A_RT(vON * a, (t - motor_delay - t_aff) / (a**2)) / (a**2)
        else:
            pdf = stupid_f_integral(v, vON, a, t_post_led, tp)

    return pdf


def led_off_cdf(t, v, a, t_aff, motor_delay):
    if t <= motor_delay + t_aff:
        return 0
    return cum_A_t_fn(t - motor_delay - t_aff, v, a)


def led_off_pdf(t, v, a, t_aff, motor_delay):
    if t <= motor_delay + t_aff:
        return 0
    return d_A_RT(v * a, (t - motor_delay - t_aff) / (a**2)) / (a**2)


def led_off_survival(t_stim, v, a, t_aff, motor_delay):
    return 1 - led_off_cdf(t_stim, v, a, t_aff, motor_delay)


def led_on_survival(t_stim, t_led, v, vON, a, t_aff, motor_delay, t_effect):
    t_pts_cdf = np.arange(0, t_stim + 0.001, 0.001)
    pdf_vals_cdf = np.array([
        PA_with_LEDON_2(ti, v, vON, a, t_aff, motor_delay, t_led, t_effect)
        for ti in t_pts_cdf
    ])
    cdf_t_stim = np.trapz(pdf_vals_cdf, t_pts_cdf)
    return 1 - cdf_t_stim


def lapse_pdf(t, beta):
    return beta * np.exp(-beta * t)


def lapse_survival(t_stim, beta):
    return np.exp(-beta * t_stim)


def compute_trial_loglike(row, V_A_base, V_A_post_LED, theta_A, t_aff, t_effect, motor_delay,
                          lapse_prob, beta_lapse):
    t = row['RT']
    t_stim = row['t_stim']
    is_led = row['LED_trial'] == 1
    t_led = row['t_LED']

    if is_led and (t_led is None or (isinstance(t_led, float) and np.isnan(t_led))):
        raise ValueError("LED trial has invalid t_LED (None/NaN).")

    if t < t_stim:
        if is_led:
            proactive_ll = PA_with_LEDON_2(
                t, V_A_base, V_A_post_LED, theta_A, t_aff, motor_delay, t_led, t_effect
            )
        else:
            proactive_ll = led_off_pdf(t, V_A_base, theta_A, t_aff, motor_delay)
        likelihood = (1 - lapse_prob) * proactive_ll + lapse_prob * lapse_pdf(t, beta_lapse)
    else:
        if is_led:
            proactive_surv = led_on_survival(
                t_stim, t_led, V_A_base, V_A_post_LED, theta_A, t_aff, motor_delay, t_effect
            )
        else:
            proactive_surv = led_off_survival(t_stim, V_A_base, theta_A, t_aff, motor_delay)
        likelihood = (1 - lapse_prob) * proactive_surv + lapse_prob * lapse_survival(t_stim, beta_lapse)

    if likelihood <= 0 or np.isnan(likelihood):
        likelihood = 1e-50

    return np.log(likelihood)


def proactive_led_loglike(params):
    V_A_base, V_A_post_LED, theta_A, t_aff, t_effect, motor_delay, lapse_prob, beta_lapse = params
    all_loglike = Parallel(n_jobs=30)(
        delayed(compute_trial_loglike)(
            row, V_A_base, V_A_post_LED, theta_A, t_aff, t_effect, motor_delay,
            lapse_prob, beta_lapse
        ) for _, row in sim_df.iterrows()
    )
    return np.sum(all_loglike)


# %%
# Test the likelihood
def compute_theoretical_rtd_on(t_pts, V_A_base, V_A_post_LED, theta_A, t_aff, t_effect, motor_delay,
                               N_mc=1000):
    pdf_samples = np.zeros((N_mc, len(t_pts)))
    for i in range(N_mc):
        trial_idx = np.random.randint(len(LED_times))
        t_led = LED_times[trial_idx]
        for j, t in enumerate(t_pts):
            pdf_samples[i, j] = PA_with_LEDON_2(
                t, V_A_base, V_A_post_LED, theta_A, t_aff, motor_delay, t_led, t_effect
            )
    proactive_rtd = np.mean(pdf_samples, axis=0)
    return (1 - lapse_prob) * proactive_rtd + lapse_prob * lapse_pdf(t_pts, beta_lapse)


def compute_theoretical_rtd_off(t_pts, V_A_base, theta_A, t_aff, motor_delay):
    proactive_rtd = np.array([
        led_off_pdf(t, V_A_base, theta_A, t_aff, motor_delay)
        for t in t_pts
    ])
    return (1 - lapse_prob) * proactive_rtd + lapse_prob * lapse_pdf(t_pts, beta_lapse)


# Separate simulated RTs into LED ON and OFF
sim_rts_on = [rt for rt, is_led in zip(sim_rts, sim_is_led_trials) if is_led]
sim_rts_off = [rt for rt, is_led in zip(sim_rts, sim_is_led_trials) if not is_led]

sim_t_stims_on = [t_stim for rt, is_led, t_stim in zip(sim_rts, sim_is_led_trials, sim_t_stims)
                  if is_led]
sim_t_stims_off = [t_stim for rt, is_led, t_stim in zip(sim_rts, sim_is_led_trials, sim_t_stims)
                   if not is_led]
sim_t_LEDs_on = [t_led for rt, is_led, t_led in zip(sim_rts, sim_is_led_trials, sim_t_LEDs)
                 if is_led]

sim_df = pd.DataFrame({
    'RT': sim_rts_on + sim_rts_off,
    't_stim': sim_t_stims_on + sim_t_stims_off,
    't_LED': sim_t_LEDs_on + [np.nan] * len(sim_rts_off),
    'LED_trial': [1] * len(sim_rts_on) + [0] * len(sim_rts_off)
})

# Compute theoretical distributions
dt = 0.01
t_max = 5.0
t_pts = np.arange(0, t_max, dt)

pdf_theory_on = compute_theoretical_rtd_on(
    t_pts, V_A_base, V_A_post_LED, theta_A, t_aff, t_effect, motor_delay, N_mc=1000
)
pdf_theory_off = compute_theoretical_rtd_off(
    t_pts, V_A_base, theta_A, t_aff, motor_delay
)

pdf_theory_on_norm = pdf_theory_on / np.trapz(pdf_theory_on, t_pts)
pdf_theory_off_norm = pdf_theory_off / np.trapz(pdf_theory_off, t_pts)

# Plot theoretical vs simulated histogram for LED ON
plt.figure(figsize=(12, 5))
bins = np.arange(0, t_max, dt)
sim_hist_on, _ = np.histogram(sim_rts_on, bins=bins, density=True)
bin_centers = (bins[1:] + bins[:-1]) / 2
plt.plot(bin_centers, sim_hist_on, label='simulated (LED ON)', lw=2, alpha=0.7)
plt.plot(t_pts, pdf_theory_on_norm, label='theoretical (LED ON)', lw=2, ls='--')
area_theory_on = np.trapz(pdf_theory_on_norm, t_pts)
print(f"Theoretical area (LED ON): {area_theory_on:.6f}")
plt.xlabel('RT (s)')
plt.ylabel('Density')
plt.title('Theoretical vs Simulated RT Distribution (LED ON)')
plt.legend()
plt.show()

# Plot theoretical vs simulated histogram for LED OFF
plt.figure(figsize=(12, 5))
sim_hist_off, _ = np.histogram(sim_rts_off, bins=bins, density=True)
plt.plot(bin_centers, sim_hist_off, label='simulated (LED OFF)', lw=2, alpha=0.7)
plt.plot(t_pts, pdf_theory_off_norm, label='theoretical (LED OFF)', lw=2, ls='--')
area_theory_off = np.trapz(pdf_theory_off_norm, t_pts)
print(f"Theoretical area (LED OFF): {area_theory_off:.6f}")
plt.xlabel('RT (s)')
plt.ylabel('Density')
plt.title('Theoretical vs Simulated RT Distribution (LED OFF)')
plt.legend()
plt.show()

# Censoring: fraction of trials after t_stim (simulated)
sim_after_t_stim_off = sum(1 for rt, t_stim in zip(sim_rts_off, sim_t_stims_off) if rt > t_stim)
frac_after_t_stim_off = sim_after_t_stim_off / len(sim_rts_off)
print(f"LED OFF - Fraction of trials after t_stim (simulated): {frac_after_t_stim_off:.6f}")
print(f"LED OFF - Total trials (after truncation): {len(sim_rts_off)}")

sim_after_t_stim_on = sum(1 for rt, t_stim in zip(sim_rts_on, sim_t_stims_on) if rt > t_stim)
frac_after_t_stim_on = sim_after_t_stim_on / len(sim_rts_on)
print(f"LED ON - Fraction of trials after t_stim (simulated): {frac_after_t_stim_on:.6f}")
print(f"LED ON - Total trials (after truncation): {len(sim_rts_on)}")

# Censoring: theoretical survival probability via Monte Carlo
N_mc = 1000

survival_off_samples = []
for _ in range(N_mc):
    t_stim = np.random.choice(stim_times_off)
    survival_off_samples.append(
        led_off_survival(t_stim, V_A_base, theta_A, t_aff, motor_delay)
    )

theoretical_survival_off = np.mean(survival_off_samples)
print(f"LED OFF - Fraction of trials after t_stim (theoretical): {theoretical_survival_off:.6f}")
print(f"LED OFF - Difference (sim - theory): {frac_after_t_stim_off - theoretical_survival_off:.6f}")

survival_on_samples = []
for _ in range(N_mc):
    trial_idx = np.random.randint(len(LED_times))
    t_led = LED_times[trial_idx]
    t_stim = stim_times[trial_idx]
    survival_on_samples.append(
        led_on_survival(t_stim, t_led, V_A_base, V_A_post_LED, theta_A, t_aff,
                        motor_delay, t_effect)
    )

theoretical_survival_on = np.mean(survival_on_samples)
print(f"LED ON - Fraction of trials after t_stim (theoretical): {theoretical_survival_on:.6f}")
print(f"LED ON - Difference (sim - theory): {frac_after_t_stim_on - theoretical_survival_on:.6f}")

# %%
# RT wrt LED: theory vs sim (area-weighted rate = density * abort fraction)

def hist_density_or_zero(values, bins):
    if len(values) == 0:
        return np.zeros(len(bins) - 1)
    hist, _ = np.histogram(values, bins=bins, density=True)
    return hist


def led_on_pdf(t, t_led, v, v_on, a, t_aff, motor_delay, t_effect):
    return PA_with_LEDON_2(t, v, v_on, a, t_aff, motor_delay, t_led, t_effect)

# %%
# Parameters (easy to tune per run)
sim_bw = 0.03
bins_wrt_led = np.arange(-2.0, 2.0, sim_bw)
bin_centers_wrt_led = 0.5 * (bins_wrt_led[:-1] + bins_wrt_led[1:])
t_pts_wrt_led = np.arange(-2,2,0.001)
N_mc_wrt_led = int(5e3)

# Sim RT wrt LED (aborts only: RT < t_stim), area-weighted by abort fraction
n_all_sim_on = len(sim_rts_on)
n_all_sim_off = len(sim_rts_off)

sim_rts_wrt_led_on = [
    rt - t_led
    for rt, t_stim, t_led in zip(sim_rts_on, sim_t_stims_on, sim_t_LEDs_on)
    if rt < t_stim
]

sim_t_LEDs_off = [
    t_led
    for rt, is_led, t_led in zip(sim_rts, sim_is_led_trials, sim_t_LEDs)
    if not is_led
]
sim_rts_wrt_led_off = [
    rt - t_led
    for rt, t_stim, t_led in zip(sim_rts_off, sim_t_stims_off, sim_t_LEDs_off)
    if rt < t_stim
]

frac_sim_on = len(sim_rts_wrt_led_on) / n_all_sim_on if n_all_sim_on > 0 else 0.0
frac_sim_off = len(sim_rts_wrt_led_off) / n_all_sim_off if n_all_sim_off > 0 else 0.0

sim_hist_on_dens = hist_density_or_zero(sim_rts_wrt_led_on, bins_wrt_led)
sim_hist_off_dens = hist_density_or_zero(sim_rts_wrt_led_off, bins_wrt_led)
sim_hist_on_scaled = sim_hist_on_dens * frac_sim_on
sim_hist_off_scaled = sim_hist_off_dens * frac_sim_off

# Theory RT wrt LED (same area-weighted target)
rtd_theory_on_wrt_led = np.zeros(len(t_pts_wrt_led))
rtd_theory_off_wrt_led = np.zeros(len(t_pts_wrt_led))
for _ in tqdm(range(N_mc_wrt_led), desc="Theory RT wrt LED"):
    trial_idx = np.random.randint(len(LED_times))
    t_led = LED_times[trial_idx]
    t_stim = stim_times[trial_idx]
    t_pts_wrt_fix = t_pts_wrt_led + t_led

    mask_on = (t_pts_wrt_fix > 0) & (t_pts_wrt_fix < t_stim)
    if np.any(mask_on):
        proactive_vals = np.array(
            [
                led_on_pdf(
                    t_wrt_fix,
                    t_led,
                    V_A_base,
                    V_A_post_LED,
                    theta_A,
                    t_aff,
                    motor_delay,
                    t_effect,
                )
                for t_wrt_fix in t_pts_wrt_fix[mask_on]
            ]
        )
        lapse_vals = lapse_pdf(t_pts_wrt_fix[mask_on], beta_lapse)
        rtd_theory_on_wrt_led[mask_on] += (1 - lapse_prob) * proactive_vals + lapse_prob * lapse_vals

    mask_off = (t_pts_wrt_fix > 0) & (t_pts_wrt_fix < t_stim)
    if np.any(mask_off):
        proactive_vals = np.array(
            [
                led_off_pdf(
                    t_wrt_fix, V_A_base, theta_A, t_aff, motor_delay
                )
                for t_wrt_fix in t_pts_wrt_fix[mask_off]
            ]
        )
        lapse_vals = lapse_pdf(t_pts_wrt_fix[mask_off], beta_lapse)
        rtd_theory_off_wrt_led[mask_off] += (1 - lapse_prob) * proactive_vals + lapse_prob * lapse_vals

rtd_theory_on_wrt_led /= N_mc_wrt_led
rtd_theory_off_wrt_led /= N_mc_wrt_led
# %%
# Plot
fig, ax = plt.subplots(figsize=(12, 5))
# ax.plot(
#     bin_centers_wrt_led,
#     sim_hist_on_scaled,
#     label=f"Sim LED ON (frac={frac_sim_on:.4f})",
#     lw=2,
#     alpha=0.7,
#     color="r",
# )
# ax.plot(
#     bin_centers_wrt_led,
#     sim_hist_off_scaled,
#     label=f"Sim LED OFF (frac={frac_sim_off:.4f})",
#     lw=2,
#     alpha=0.7,
#     color="b",
# )

ax.plot(
    t_pts_wrt_led,
    rtd_theory_on_wrt_led,
    label="Theory LED ON",
    lw=2,
    alpha=0.5,
    color="r",
    linestyle="--",
)

# ax.plot(
#     t_pts_wrt_led,
#     rtd_theory_off_wrt_led,
#     label="Theory LED OFF",
#     lw=2,
#     alpha=0.5,
#     color="b",
#     linestyle="--",
# )
ax.axvline(x=0, color="k", linestyle="--", alpha=0.5, label="LED onset")
# ax.set_xlim(-1, 1)
ax.set_xlabel("RT - t_LED (s)")
ax.set_ylabel("Rate (area = fraction)")
ax.set_title("RT wrt LED: Theory vs Sim (area-weighted)")
ax.legend()
plt.tight_layout()
plt.show()

bin_width = bins_wrt_led[1] - bins_wrt_led[0]
print(f"\nRT wrt LED areas:")
print(
    f"  Sim ON={np.sum(sim_hist_on_scaled) * bin_width:.4f}, "
    f"OFF={np.sum(sim_hist_off_scaled) * bin_width:.4f}"
)
print(
    f"  Theory ON={np.trapz(rtd_theory_on_wrt_led, t_pts_wrt_led):.4f}, "
    f"OFF={np.trapz(rtd_theory_off_wrt_led, t_pts_wrt_led):.4f}"
)


# %%
