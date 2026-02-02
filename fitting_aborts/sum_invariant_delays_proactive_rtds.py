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
from pyvbmc import VBMC
import corner
import pickle

# =============================================================================
# PDF functions for theoretical RTD
# =============================================================================
def PA_with_LEDON_2_adapted(t, v, vON, a, t_aff, motor_delay, tled, t_effect, T_trunc=None):
    """Compute the PA pdf by combining contributions before and after LED onset."""
    if T_trunc is not None and t <= T_trunc:
        return 0

    if (t - motor_delay) <= (tled + t_effect):
        pdf = d_A_RT(v * a, (t - motor_delay - t_aff) / (a**2)) / (a**2)
    else:
        t_post_led = t - motor_delay - tled - t_effect
        tp = tled + t_effect - t_aff

        if tp <= 0:
            pdf = d_A_RT(vON * a, (t - motor_delay - t_aff) / (a**2)) / (a**2)
        else:
            pdf = stupid_f_integral(v, vON, a, t_post_led, tp)

    if T_trunc is not None:
        t_pts = np.arange(0, T_trunc + 0.001, 0.001)
        pdf_vals = np.array([
            PA_with_LEDON_2_adapted(ti, v, vON, a, t_aff, motor_delay, tled, t_effect, None)
            for ti in t_pts
        ])
        cdf_trunc = np.trapz(pdf_vals, t_pts)
        pdf = pdf / (1 - cdf_trunc)

    return pdf


def led_off_cdf(t, v, a, t_aff, motor_delay):
    if t <= motor_delay + t_aff:
        return 0
    return cum_A_t_fn(t - motor_delay - t_aff, v, a)


def led_off_pdf_truncated(t, v, a, t_aff, motor_delay, T_trunc):
    if t <= T_trunc:
        return 0

    if t <= motor_delay + t_aff:
        return 0

    pdf = d_A_RT(v * a, (t - motor_delay - t_aff) / (a**2)) / (a**2)

    if T_trunc is not None:
        cdf_trunc = led_off_cdf(T_trunc, v, a, t_aff, motor_delay)
        trunc_factor = 1 - cdf_trunc
        if trunc_factor <= 0:
            return 0
        pdf = pdf / trunc_factor

    return pdf

# %%
T_trunc = 0.3   # Left truncation threshold (exclude RT <= T_trunc)

# %%
# =============================================================================
# Load VBMC results from aggregated fit
# =============================================================================
with open('vbmc_real_all_animals_fit.pkl', 'rb') as f:
    vp = pickle.load(f)

# Sample from posterior to get parameter means
vp_samples = vp.sample(int(1e5))[0]

param_labels = ['V_A_base', 'V_A_post_LED', 'theta_A', 't_aff', 't_effect', 'motor_delay']
param_means = np.mean(vp_samples, axis=0)
param_stds = np.std(vp_samples, axis=0)

print("\nPosterior summary (all animals aggregated):")
print(f"{'Parameter':<15} {'Mean':<12} {'Std':<12}")
print("-" * 40)
for i, label in enumerate(param_labels):
    print(f"{label:<15} {param_means[i]:<12.4f} {param_stds[i]:<12.4f}")

# %%
# =============================================================================
# Load data for timing distributions (t_LED, t_stim)
# =============================================================================
og_df = pd.read_csv('../out_LED.csv')

df = og_df[og_df['repeat_trial'].isin([0, 2]) | og_df['repeat_trial'].isna()]
session_type = 7
df = df[df['session_type'].isin([session_type])]
training_level = 16
df = df[df['training_level'].isin([training_level])]

df = df.dropna(subset=['intended_fix', 'LED_onset_time', 'timed_fix'])
df = df[(df['abort_event'] == 3) | (df['success'].isin([1, -1]))]
df = df[~((df['abort_event'] == 3) & (df['timed_fix'] < T_trunc))]

# Get timing distributions from ALL trials
stim_times = df['intended_fix'].values
LED_times = (df['intended_fix'] - df['LED_onset_time']).values
n_trials_data = len(stim_times)

print(f"\nLoaded {len(df)} trials for timing distributions")

# %%
# =============================================================================
# Simulate proactive trials with fitted parameters
# =============================================================================
N_trials_sim = int(50e3)
print(f"\nSimulating {N_trials_sim} trials with fitted parameters...")

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
            return t + motor_delay  # Add motor delay to get full RT

def simulate_single_trial_fit():
    is_led_trial = np.random.random() < 1/3
    # Sample trial index to preserve (t_LED, t_stim) correlation
    trial_idx = np.random.randint(n_trials_data)
    t_LED = LED_times[trial_idx]
    t_stim = stim_times[trial_idx]
    rt = simulate_proactive_single_bound(
        param_means[0], param_means[1], param_means[2],  # V_A_base, V_A_post_LED, theta_A
        t_LED if is_led_trial else None, t_stim,
        param_means[3], param_means[4], param_means[5],  # t_aff, t_effect, motor_delay
        is_led_trial
    )
    return rt, is_led_trial, t_LED, t_stim

sim_results = Parallel(n_jobs=30)(
    delayed(simulate_single_trial_fit)() for _ in range(N_trials_sim)
)

sim_rts = [r[0] for r in sim_results]
sim_is_led = [r[1] for r in sim_results]
sim_t_LEDs = [r[2] for r in sim_results]
sim_t_stims = [r[3] for r in sim_results]

# Separate into LED ON/OFF, apply truncation AND filter aborts only (RT < t_stim)
sim_rts_on = [rt for rt, is_led, t_stim in zip(sim_rts, sim_is_led, sim_t_stims) if is_led and rt > T_trunc and rt < t_stim]
sim_rts_off = [rt for rt, is_led, t_stim in zip(sim_rts, sim_is_led, sim_t_stims) if not is_led and rt > T_trunc and rt < t_stim]

# For RT wrt LED plots (abort rate), same filtering
sim_rts_wrt_led_on = [rt - t_led for rt, is_led, t_led, t_stim in zip(sim_rts, sim_is_led, sim_t_LEDs, sim_t_stims) if is_led and rt > T_trunc and rt < t_stim]
sim_rts_wrt_led_off = [rt - t_led for rt, is_led, t_led, t_stim in zip(sim_rts, sim_is_led, sim_t_LEDs, sim_t_stims) if not is_led and rt > T_trunc and rt < t_stim]

print(f"Simulation (aborts only): {len(sim_rts_on)} LED ON, {len(sim_rts_off)} LED OFF")

# %%
# =============================================================================
# Plot RTD wrt LED (area-weighted by abort rate)
# =============================================================================
# Compute abort rate for sim
n_all_sim_on = sum(1 for is_led in sim_is_led if is_led)
n_all_sim_off = sum(1 for is_led in sim_is_led if not is_led)
n_aborts_sim_on = sum(1 for rt, is_led, t_stim in zip(sim_rts, sim_is_led, sim_t_stims) if is_led and rt < t_stim and rt > T_trunc)
n_aborts_sim_off = sum(1 for rt, is_led, t_stim in zip(sim_rts, sim_is_led, sim_t_stims) if not is_led and rt < t_stim and rt > T_trunc)
frac_sim_on = n_aborts_sim_on / n_all_sim_on if n_all_sim_on > 0 else 0
frac_sim_off = n_aborts_sim_off / n_all_sim_off if n_all_sim_off > 0 else 0

# %%
# theory LED
N_mc = int(1)
t_pts_wrt_led = np.arange(-3, 3, 0.001)
rtd_on_wrt_led = np.zeros((N_mc, len(t_pts_wrt_led)))
rtd_off_wrt_led = np.zeros((N_mc, len(t_pts_wrt_led)))

for i in tqdm(range(N_mc)):
    trial_idx = np.random.randint(n_trials_data)
    t_led = LED_times[trial_idx]
    t_stim = stim_times[trial_idx]
    print(f't_stim={t_stim}, t_LED={t_led}')
    # Skip if t_stim <= T_trunc (no valid abort region)
    if t_stim <= T_trunc:
        continue
    
    t_pts_wrt_fix = t_pts_wrt_led + t_led  # Convert wrt LED -> wrt fixation
    t_stim_wrt_led = t_stim - t_led
    T_trunc_wrt_led = T_trunc - t_led
    
    # OFF: compute PDF for each t (wrt fixation), then store in wrt LED array
    for j, (t_wrt_led, t_wrt_fix) in enumerate(zip(t_pts_wrt_led, t_pts_wrt_fix)):
        # Condition 1: t <= T_trunc and t < t_stim -> 0
        if (t_wrt_fix <= T_trunc) and (t_wrt_fix < t_stim):
            rtd_off_wrt_led[i, j] = 0
        # Condition 2: t >= t_stim -> 0 (not an abort)
        elif t_wrt_fix >= t_stim:
            rtd_off_wrt_led[i, j] = 0
        # Valid abort: T_trunc < t < t_stim
        else:
            rtd_off_wrt_led[i, j] = led_off_pdf_truncated(
                t_wrt_fix, param_means[0], param_means[2], param_means[3], param_means[5], T_trunc
            )
    
    # ON: compute PDF for each t (wrt fixation), then store in wrt LED array
    for j, (t_wrt_led, t_wrt_fix) in enumerate(zip(t_pts_wrt_led, t_pts_wrt_fix)):
        # Condition 1: t <= T_trunc and t < t_stim -> 0
        if (t_wrt_fix <= T_trunc) and (t_wrt_fix < t_stim):
            rtd_on_wrt_led[i, j] = 0
        # Condition 2: t >= t_stim -> 0 (not an abort)
        elif t_wrt_fix >= t_stim:
            rtd_on_wrt_led[i, j] = 0
        # Valid abort: T_trunc < t < t_stim
        else:
            rtd_on_wrt_led[i, j] = PA_with_LEDON_2_adapted(
                t_wrt_fix, param_means[0], param_means[1], param_means[2],
                param_means[3], param_means[5], t_led, param_means[4], T_trunc
            )

# Average over MC samples
theory_rtd_on_wrt_led = np.mean(rtd_on_wrt_led, axis=0)
theory_rtd_off_wrt_led = np.mean(rtd_off_wrt_led, axis=0)

# %%
# area under curve
print(f'area under curve: {np.trapz(theory_rtd_on_wrt_led, t_pts_wrt_led)}')
print(f'area under curve: {np.trapz(theory_rtd_off_wrt_led, t_pts_wrt_led)}')
# %%
# Create histograms with density=True then scale by fraction
bins_wrt_led = np.arange(-3, 3, 0.01)
bin_centers_wrt_led = (bins_wrt_led[1:] + bins_wrt_led[:-1]) / 2

fig, ax = plt.subplots(figsize=(10, 6))

# Sim histograms (area = fraction)
sim_hist_on_wrt_led_dens, _ = np.histogram(sim_rts_wrt_led_on, bins=bins_wrt_led, density=True)
sim_hist_off_wrt_led_dens, _ = np.histogram(sim_rts_wrt_led_off, bins=bins_wrt_led, density=True)
sim_hist_on_scaled = sim_hist_on_wrt_led_dens * frac_sim_on
sim_hist_off_scaled = sim_hist_off_wrt_led_dens * frac_sim_off

# Plot
ax.plot(bin_centers_wrt_led, sim_hist_on_scaled, label=f'Sim LED ON (frac={frac_sim_on:.4f})', lw=2, alpha=0.7, color='r', linestyle='--')
ax.plot(bin_centers_wrt_led, sim_hist_off_scaled, label=f'Sim LED OFF (frac={frac_sim_off:.4f})', lw=2, alpha=0.7, color='b', linestyle='--')

# Plot theoretical curves (scale by sim abort fraction for comparison)
ax.plot(t_pts_wrt_led, theory_rtd_on_wrt_led, label='Theory LED ON', lw=2, alpha=0.3, color='k', linestyle='-')
ax.plot(t_pts_wrt_led, theory_rtd_off_wrt_led, label='Theory LED OFF', lw=2, alpha=0.3, color='k', linestyle='-')

ax.axvline(x=0, color='k', linestyle='--', alpha=0.5, label='LED onset')
ax.axvline(x=param_means[4], color='g', linestyle=':', alpha=0.5, label=f't_effect={param_means[4]:.2f}')
ax.set_xlabel('RT - t_LED (s)', fontsize=12)
ax.set_ylabel('Rate (area = fraction)', fontsize=12)
ax.set_title('RT wrt LED (area-weighted) - All Animals', fontsize=14)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('sum_invariant_delays_proactive_rtds.pdf', bbox_inches='tight')
print("RTD wrt LED plot saved as 'sum_invariant_delays_proactive_rtds.pdf'")
plt.show()

print("\nScript complete!")
# %%
# sum invariant curves
# LED ON and LED OFF - 2 figures
# try plotting RT - t_LED for different sets of params
# the sets of params are such that
# params[3] + params[5] is same as it is now
# params[4] + params[5] is same as it is now
# but the individual valiues are different
# may be 3 sets of params
# the goal is to check if the sum is invariant, are distributions invariant
# =============================================================================
# Sum Invariant Test: Keep t_aff + motor_delay and t_effect + motor_delay constant
# =============================================================================
# Original params: [V_A_base, V_A_post_LED, theta_A, t_aff, t_effect, motor_delay]

# %%
# alter motor delay and totla delay so that both are positive
param_means_og = param_means.copy()
param_means[3] = 10*1e-3
param_means[4] = 60*1e-3
param_means[5] = 20*1e-3

# %%

# Compute sums to keep constant
sum_aff_motor = param_means[3] + param_means[5]      # t_aff + motor_delay
sum_effect_motor = param_means[4] + param_means[5]   # t_effect + motor_delay

print(f"\nOriginal parameters:")
print(f"  t_aff = {param_means[3]:.4f}")
print(f"  t_effect = {param_means[4]:.4f}")
print(f"  motor_delay = {param_means[5]:.4f}")
print(f"\nSum invariant test:")
print(f"  t_aff + motor_delay = {sum_aff_motor:.4f}")
print(f"  t_effect + motor_delay = {sum_effect_motor:.4f}")

# Create 3 parameter sets with same sums but different individual values
# Adjust offsets based on actual motor_delay value to avoid negative params
# motor_delay_offsets = [0, 5*1e-3, 10*1e-3, 15*1e-3, 20*1e-3]  # Decrease motor_delay to increase t_aff/t_effect
motor_delay_offsets = [0, 50*1e-3]  # Decrease motor_delay to increase t_aff/t_effect

param_sets = []
labels = []

for offset in motor_delay_offsets:
    new_motor_delay = param_means[5] + offset
    # NOTE
    # to test LED ON, lets relax the first constraint
    # new_t_aff = sum_aff_motor - new_motor_delay
    new_t_aff = param_means[3] 
    new_t_effect = sum_effect_motor - new_motor_delay
    
    # Only check motor_delay >= 0 (t_aff/t_effect can be negative in this parameterization)
    # if new_motor_delay < 0:
    #     print(f"  Skipping offset {offset}: motor_delay would be negative")
    #     continue

    
    params = param_means.copy()
    params[3] = new_t_aff
    params[4] = new_t_effect
    params[5] = new_motor_delay
    param_sets.append(params)
    print(f'new params')
    print(params)
    labels.append(f't_aff={new_t_aff:.3f}, t_eff={new_t_effect:.3f}, motor={new_motor_delay:.3f}')

print(f"\nParameter sets:")
for idx, params in enumerate(param_sets):
    print(f"  Set {idx+1}: {labels[idx]}")
# %%
# Compute theoretical RTD wrt LED for each parameter set
# IMPORTANT: Use SAME random samples for all parameter sets to test true invariance
# N_mc_inv = int(3e3)
N_mc_inv = int(1)
colors = ['r', 'g', 'b']
theory_on_sets = []
theory_off_sets = []

# Pre-generate random samples ONCE for all parameter sets
np.random.seed(42)
trial_indices = np.random.randint(n_trials_data, size=N_mc_inv)
t_led_samples = LED_times[trial_indices]
t_stim_samples = stim_times[trial_indices]

for idx, params in enumerate(param_sets):
    print(f"\nComputing RTD for parameter set {idx+1}/{len(param_sets)}...")
    print(params)
    rtd_on = np.zeros((N_mc_inv, len(t_pts_wrt_led)))
    rtd_off = np.zeros((N_mc_inv, len(t_pts_wrt_led)))
    
    for i in tqdm(range(N_mc_inv)):
        t_led = t_led_samples[i]
        t_stim = t_stim_samples[i]
        
        if t_stim <= T_trunc:
            continue
        
        t_pts_wrt_fix = t_pts_wrt_led + t_led
        
        for j, (t_wrt_led, t_wrt_fix) in enumerate(zip(t_pts_wrt_led, t_pts_wrt_fix)):
            if (t_wrt_fix <= T_trunc) and (t_wrt_fix < t_stim):
                rtd_off[i, j] = 0
            elif t_wrt_fix >= t_stim:
                rtd_off[i, j] = 0
            else:
                rtd_off[i, j] = led_off_pdf_truncated(t_wrt_fix, params[0], params[2], params[3], params[5], T_trunc)
        
        for j, (t_wrt_led, t_wrt_fix) in enumerate(zip(t_pts_wrt_led, t_pts_wrt_fix)):
            if (t_wrt_fix <= T_trunc) and (t_wrt_fix < t_stim):
                rtd_on[i, j] = 0
            elif t_wrt_fix >= t_stim:
                rtd_on[i, j] = 0
            else:
                rtd_on[i, j] = PA_with_LEDON_2_adapted(t_wrt_fix, params[0], params[1], params[2], params[3], params[5], t_led, params[4], T_trunc)
    
    theory_on_sets.append(np.mean(rtd_on, axis=0))
    theory_off_sets.append(np.mean(rtd_off, axis=0))

# %%
# Plot LED OFF
fig, ax = plt.subplots(figsize=(10, 6))
for idx, (rtd_off, label) in enumerate(zip(theory_off_sets, labels)):
    ax.plot(t_pts_wrt_led, rtd_off, label=label, lw=2, alpha=0.8)
ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
ax.set_xlabel('RT - t_LED (s)')
ax.set_title(f'LED OFF: Sum Invariance (t_aff + motor = {sum_aff_motor:.3f}), Area={np.trapz(theory_off_sets[0], t_pts_wrt_led):.3f}')
ax.legend()
ax.set_xlim(-1.5, 1)
plt.savefig('sum_invariant_LED_OFF.pdf')
plt.show()

# %%
# Plot LED ON
fig, ax = plt.subplots(figsize=(10, 6))
for idx, (rtd_on, label) in enumerate(zip(theory_on_sets, labels)):
    ax.plot(t_pts_wrt_led, rtd_on, label=label, lw=2, alpha=0.8)
    # ax.axvline(0.01 + 0.01, alpha=0.2, color='orange', ls='--')
ax.axvline(0.08, alpha=0.2, color='blue', ls='--', label='80ms(LED eff+ motor)')
ax.axvline(x=0, color='k', linestyle='--', alpha=0.2)
# ax.axvline(0.881-0.741 + 0.08)
ax.set_xlabel('RT - t_LED (s)')
ax.set_title(f'LED ON: Sum Invariance(t_eff + motor = {sum_effect_motor:.3f})')
ax.legend()
ax.set_xlim(-0.5, 0.3)
plt.savefig('sum_invariant_LED_ON.pdf')
plt.show()
# print area 
print(np.trapz(theory_on_sets[0], t_pts_wrt_led))
# %%
# =============================================================================
# Simulation for Sum Invariance Confirmation
# =============================================================================
N_trials_sim_inv = int(900e3)  # per parameter set

def sim_single_trial(V_A_base, V_A_post_LED, theta_A, t_aff, t_effect, motor_delay):
    """Simulate a single trial."""
    dt = 1e-3
    dB = np.sqrt(dt)
    
    is_led_trial = np.random.random() < 1/3
    trial_idx = np.random.randint(n_trials_data)
    t_LED = LED_times[trial_idx]
    t_stim = stim_times[trial_idx]
    
    AI = 0
    t = t_aff
    
    while True:
        if is_led_trial and t >= t_LED + t_effect:
            V_A = V_A_post_LED
        else:
            V_A = V_A_base
        
        AI += V_A * dt + np.random.normal(0, dB)
        t += dt
        
        if AI >= theta_A:
            rt = t + motor_delay
            return rt, is_led_trial, t_LED, t_stim

def simulate_proactive_with_params(params, n_trials, n_jobs=25):
    """Simulate trials with given parameters and return RTs wrt LED for ON/OFF."""
    V_A_base, V_A_post_LED, theta_A, t_aff, t_effect, motor_delay = params
    
    # Let joblib handle distribution across workers
    results = Parallel(n_jobs=n_jobs)(
        delayed(sim_single_trial)(V_A_base, V_A_post_LED, theta_A, t_aff, t_effect, motor_delay)
        for _ in tqdm(range(n_trials), desc="Simulating trials")
    )
    
    # Count total ON and OFF trials
    n_total_on = sum(1 for _, led, _, _ in results if led)
    n_total_off = sum(1 for _, led, _, _ in results if not led)
    
    # Filter aborts
    rts_wrt_led_on = [rt - t_led for rt, led, t_led, t_stim in results 
                      if led and rt > T_trunc and rt < t_stim]
    rts_wrt_led_off = [rt - t_led for rt, led, t_led, t_stim in results 
                       if not led and rt > T_trunc and rt < t_stim]
    
    return rts_wrt_led_on, rts_wrt_led_off, n_total_on, n_total_off

sim_on_sets = []
sim_off_sets = []
n_total_on_sets = []
n_total_off_sets = []

for idx, params in enumerate(param_sets):
    print(f"\nSimulating parameter set {idx+1}/{len(param_sets)}: {labels[idx]}")
    rts_on, rts_off, n_total_on, n_total_off = simulate_proactive_with_params(params, N_trials_sim_inv, n_jobs=25)
    sim_on_sets.append(rts_on)
    sim_off_sets.append(rts_off)
    n_total_on_sets.append(n_total_on)
    n_total_off_sets.append(n_total_off)
    print(f"  LED ON aborts: {len(rts_on)}/{n_total_on}, LED OFF aborts: {len(rts_off)}/{n_total_off}")

# %%
# Plot Simulation LED OFF (scaled by fraction)
bins_inv = np.arange(-1.5, 1, 0.05)
bin_centers_inv = (bins_inv[1:] + bins_inv[:-1]) / 2

fig, ax = plt.subplots(figsize=(10, 6))
for idx, (rts_off, n_tot_off, label) in enumerate(zip(sim_off_sets, n_total_off_sets, labels)):
    frac_off = len(rts_off) / n_tot_off  # aborts_off / total_off
    hist, _ = np.histogram(rts_off, bins=bins_inv, density=True)
    hist_scaled = hist * frac_off
    ax.plot(bin_centers_inv, hist_scaled, label=f'{label} (frac={frac_off:.3f})', lw=2, alpha=0.8)
ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
ax.set_xlabel('RT - t_LED (s)')
ax.set_ylabel('Rate (area = fraction)')
ax.set_title(f'LED OFF (Sim): Sum Invariance (t_aff + motor = {sum_aff_motor:.3f}), Area={np.trapz(hist_scaled, bin_centers_inv):.3f}')
ax.legend(fontsize=8)
ax.set_xlim(-1.5, 1)
plt.savefig('sum_invariant_LED_OFF_sim.pdf')
plt.show()

# %%
# Plot Simulation LED ON (scaled by fraction)
fig, ax = plt.subplots(figsize=(10, 6))
for idx, (rts_on, n_tot_on, label) in enumerate(zip(sim_on_sets, n_total_on_sets, labels)):
    frac_on = len(rts_on) / n_tot_on  # aborts_on / total_on
    hist, _ = np.histogram(rts_on, bins=bins_inv, density=True)
    hist_scaled = hist * frac_on
    ax.plot(bin_centers_inv, hist_scaled, label=f'{label} (frac={frac_on:.3f})', lw=2, alpha=0.8)
ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
ax.set_xlabel('RT - t_LED (s)')
ax.set_ylabel('Rate (area = fraction)')
ax.set_title(f'LED ON (Sim): Sum Invariance(t_eff + motor = {sum_effect_motor:.3f})')
ax.legend(fontsize=8)
ax.set_xlim(-1.5, 1)
plt.savefig('sum_invariant_LED_ON_sim.pdf')
plt.show()
# %%
