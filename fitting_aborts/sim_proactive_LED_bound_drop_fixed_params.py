"""
Simulator for proactive LED model with bound drop (instead of drift increase).

Uses fixed parameters from the all-animals aggregated posterior image.
LED effect here: decision bound decreases after LED effect onset.
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import pandas as pd
from pathlib import Path

# %%
# =============================================================================
# PARAMETERS (edit here)
# =============================================================================
ANIMAL_ID = None  # None for all animals, or integer index (0, 1, 2...)
N_TRIALS_SIM = int(300e3)
N_JOBS = 30
DT = 1e-4
RNG_SEED = 123
LED_ON_PROB = 1 / 3
THETA_FLOOR = 0
THETA_LED_MIN = 1.8  # LED bound lower limit: bound will not go below this value

# Posterior means from image (all animals aggregated)
V_A_BASE = 1.4730
V_A_POST_LED_IMAGE = 3.0935  # kept for reference; not used in bound-drop simulator
THETA_A = 2.16 # og = 2.16
DEL_A_MINUS_DEL_LED = -0.0661
# DEL_M_PLUS_DEL_LED = 0.0260
DEL_M_PLUS_DEL_LED = 0.047

LAPSE_PROB = 0.0420
BETA_LAPSE = 4.1832

# LED effect now acts on bound (decrease magnitude).
BOUND_DECREASE_PARAM = -0.1
USE_DYNAMIC_BOUND_DROP = True  # False: old step drop to theta_post_led, True: linear dynamic drop
# Dynamic drop settings (relative to t_effect): drop starts at 50 ms, and by 400 ms
# theta has dropped by DYNAMIC_DROP_BY_THETA.
DYNAMIC_DROP_START_S = 0.0
DYNAMIC_DROP_TARGET_TIME_S = 0.3
DYNAMIC_DROP_BY_THETA = 0.3
# AI window center relative to bound-switch onset (t_effect).
# 0.0 means T == t_effect.
DELTA_LED_FOR_AI_WINDOW = 0.005 # s, after change
AI_WINDOW_MS = 5.0 # before change

# Plot ranges
T_MAX_RTD = 3
XLIM_WRT_LED = (-0.5, 0.4)

# %%
# =============================================================================
# Data loading and filtering
# =============================================================================
np.random.seed(RNG_SEED)
repo_root = Path(__file__).resolve().parents[1]
input_csv = repo_root / 'out_LED.csv'
og_df = pd.read_csv(input_csv)

df = og_df[og_df['repeat_trial'].isin([0, 2]) | og_df['repeat_trial'].isna()]
session_type = 7
df = df[df['session_type'].isin([session_type])]
training_level = 16
df = df[df['training_level'].isin([training_level])]

df = df.dropna(subset=['intended_fix', 'LED_onset_time', 'timed_fix'])
df = df[(df['abort_event'] == 3) | (df['success'].isin([1, -1]))]

unique_animals = df['animal'].unique()

if ANIMAL_ID is not None:
    animal_name = unique_animals[ANIMAL_ID]
    df_all = df[df['animal'] == animal_name]
    animal_label = f"Animal {animal_name}"
    file_tag = f"animal_{animal_name}"
else:
    df_all = df
    animal_label = 'All Animals Aggregated'
    file_tag = 'all_animals'

df_on = df_all[df_all['LED_trial'] == 1]
df_off = df_all[df_all['LED_trial'].isin([0, np.nan])]

print(f"\n{animal_label} data summary:")
print(f"  Total trials: {len(df_all)}")
print(f"  LED ON trials: {len(df_on)}")
print(f"  LED OFF trials: {len(df_off)}")

# %%
# =============================================================================
# Build fitting-like DataFrame for plotting comparisons
# =============================================================================
df_on_fit = pd.DataFrame({
    'RT': df_on['timed_fix'].values,
    't_stim': df_on['intended_fix'].values,
    't_LED': (df_on['intended_fix'] - df_on['LED_onset_time']).values,
    'LED_trial': 1,
})

df_off_fit = pd.DataFrame({
    'RT': df_off['timed_fix'].values,
    't_stim': df_off['intended_fix'].values,
    't_LED': (df_off['intended_fix'] - df_off['LED_onset_time']).values,
    'LED_trial': 0,
})

fit_df = pd.concat([df_on_fit, df_off_fit], ignore_index=True)

n_aborts = len(fit_df[fit_df['RT'] < fit_df['t_stim']])
n_censored = len(fit_df[fit_df['RT'] >= fit_df['t_stim']])
print(f"  Abort trials (RT < t_stim): {n_aborts}")
print(f"  Censored trials (RT >= t_stim): {n_censored}")

stim_times = df_all['intended_fix'].values
LED_times = (df_all['intended_fix'] - df_all['LED_onset_time']).values
n_trials_data = len(stim_times)

# %%
# =============================================================================
# Simulator (bound drop version)
# =============================================================================
def effective_bound_drop(bound_decrease_param):
    return abs(bound_decrease_param)


def compute_dynamic_drop_slope(dynamic_drop_by_theta, dynamic_drop_start_s, dynamic_drop_target_time_s):
    drop_duration = max(dynamic_drop_target_time_s - dynamic_drop_start_s, 1e-12)
    return abs(dynamic_drop_by_theta) / drop_duration


def simulate_proactive_single_bound_bound_drop(
    V_A_base,
    theta_pre,
    theta_post,
    t_LED,
    del_a_minus_del_LED,
    del_m_plus_del_LED,
    is_led_trial,
    delta_led_for_ai_window=0.0,
    ai_window_seconds=0.005,
    use_dynamic_bound_drop=False,
    dynamic_drop_start_s=0.05,
    dynamic_drop_slope=0.0,
    theta_floor=0.0,
    dt=1e-4,
):
    """
    Proactive single-accumulator simulator with bound drop after LED effect onset.
    Drift stays at V_A_base. For LED ON trials, bound switches from theta_pre to
    theta_post at t_LED - del_a_minus_del_LED (constant mode), or decreases
    linearly after dynamic_drop_start_s in dynamic mode.
    """
    AI = 0.0
    t = 0.0
    dB = np.sqrt(dt)
    t_effect = t_LED - del_a_minus_del_LED if is_led_trial else np.inf
    # Window reference time T; by default this is exactly the bound-switch time.
    t_ref = t_effect + delta_led_for_ai_window if is_led_trial else np.inf
    t_window_start = t_ref - ai_window_seconds
    t_window_end = t_ref
    ai_window_rel_t = []
    ai_window_values = []

    while True:
        if t < t_effect:
            theta_t = theta_pre
        elif not use_dynamic_bound_drop:
            theta_t = theta_post
        else:
            dt_since_effect = t - t_effect
            if dt_since_effect < dynamic_drop_start_s:
                theta_t = theta_pre
            else:
                theta_t = theta_pre - dynamic_drop_slope * (dt_since_effect - dynamic_drop_start_s)
            theta_t = max(theta_floor, theta_t)
        AI += V_A_base * dt + np.random.normal(0.0, dB)
        t += dt

        if is_led_trial and (t_window_start <= t <= t_window_end):
            # Store times relative to T so window is [-ai_window_seconds, 0].
            ai_window_rel_t.append(t - t_ref)
            ai_window_values.append(AI)

        if AI >= theta_t:
            rt = t + (del_m_plus_del_LED + del_a_minus_del_LED)
            return rt, np.asarray(ai_window_rel_t), np.asarray(ai_window_values)


def simulate_single_trial_bound_drop(theta_post_led):
    is_led_trial = np.random.random() < LED_ON_PROB
    trial_idx = np.random.randint(n_trials_data)
    t_LED = LED_times[trial_idx]
    t_stim = stim_times[trial_idx]

    ai_window_seconds = AI_WINDOW_MS / 1000.0
    is_lapse = np.random.random() < LAPSE_PROB

    if is_lapse:
        rt = np.random.exponential(1.0 / BETA_LAPSE)
        ai_rel_t = None
        ai_vals = None
    else:
        rt, ai_rel_t, ai_vals = simulate_proactive_single_bound_bound_drop(
            V_A_BASE,
            THETA_A,
            theta_post_led,
            t_LED,
            DEL_A_MINUS_DEL_LED,
            DEL_M_PLUS_DEL_LED,
            is_led_trial,
            delta_led_for_ai_window=DELTA_LED_FOR_AI_WINDOW,
            ai_window_seconds=ai_window_seconds,
            use_dynamic_bound_drop=USE_DYNAMIC_BOUND_DROP,
            dynamic_drop_start_s=DYNAMIC_DROP_START_S,
            dynamic_drop_slope=dynamic_drop_slope,
            theta_floor=theta_led_min,
            dt=DT,
        )
    return rt, is_led_trial, t_LED, t_stim, ai_rel_t, ai_vals, is_lapse


def safe_hist_density(values, bins):
    values = np.asarray(values)
    if len(values) == 0:
        return np.zeros(len(bins) - 1)
    hist, _ = np.histogram(values, bins=bins, density=True)
    return np.nan_to_num(hist)


bound_drop = effective_bound_drop(BOUND_DECREASE_PARAM)
theta_led_min = max(THETA_FLOOR, THETA_LED_MIN)
theta_post_led = max(theta_led_min, THETA_A - bound_drop)
if USE_DYNAMIC_BOUND_DROP:
    if DYNAMIC_DROP_TARGET_TIME_S <= DYNAMIC_DROP_START_S:
        raise ValueError("DYNAMIC_DROP_TARGET_TIME_S must be greater than DYNAMIC_DROP_START_S.")
    dynamic_drop_slope = compute_dynamic_drop_slope(
        DYNAMIC_DROP_BY_THETA,
        DYNAMIC_DROP_START_S,
        DYNAMIC_DROP_TARGET_TIME_S,
    )
    theta_at_dynamic_target = max(
        theta_led_min,
        THETA_A - abs(DYNAMIC_DROP_BY_THETA),
    )
else:
    dynamic_drop_slope = 0.0
    theta_at_dynamic_target = theta_post_led

print('\nFixed parameters used:')
print(f'  V_A_base = {V_A_BASE:.4f}')
print(f'  V_A_post_LED (image reference, unused here) = {V_A_POST_LED_IMAGE:.4f}')
print(f'  theta_A_pre = {THETA_A:.4f}')
print(f'  bound_decrease_param = {BOUND_DECREASE_PARAM:.4f} -> effective drop = {bound_drop:.4f}')
print(f'  theta_LED_min = {theta_led_min:.4f}')
print(f'  theta_A_post_LED = {theta_post_led:.4f}')
if USE_DYNAMIC_BOUND_DROP:
    print(
        f'  Dynamic bound drop ON: start={DYNAMIC_DROP_START_S*1000:.1f} ms, '
        f'target_time={DYNAMIC_DROP_TARGET_TIME_S*1000:.1f} ms, '
        f'drop_by={abs(DYNAMIC_DROP_BY_THETA):.4f}, slope={dynamic_drop_slope:.4f}/s'
    )
    print(f'  theta at target_time = {theta_at_dynamic_target:.4f}')
else:
    print('  Dynamic bound drop OFF: using constant theta_A_post_LED after t_effect')
print(f'  del_a_minus_del_LED = {DEL_A_MINUS_DEL_LED:.4f}')
print(f'  del_m_plus_del_LED = {DEL_M_PLUS_DEL_LED:.4f}')
print(f'  lapse_prob = {LAPSE_PROB:.4f}, beta_lapse = {BETA_LAPSE:.4f}')

print(f"\nSimulating {N_TRIALS_SIM} trials (bound drop model)...")
sim_results = Parallel(n_jobs=N_JOBS)(
    delayed(simulate_single_trial_bound_drop)(theta_post_led) for _ in range(N_TRIALS_SIM)
)

sim_rts = [r[0] for r in sim_results]
sim_is_led = [r[1] for r in sim_results]
sim_t_LEDs = [r[2] for r in sim_results]
sim_t_stims = [r[3] for r in sim_results]
sim_ai_rel_t = [r[4] for r in sim_results]
sim_ai_vals = [r[5] for r in sim_results]
sim_is_lapse = [r[6] for r in sim_results]

ai_window_seconds = AI_WINDOW_MS / 1000.0
ai_window_rel_grid = np.arange(-ai_window_seconds, 1e-12, DT)
n_led_on_trials = int(np.sum(sim_is_led))
sim_ai_window_led_on_matrix = np.full((n_led_on_trials, len(ai_window_rel_grid)), np.nan)

led_on_row = 0
for is_led, is_lapse, rel_t, ai_vals in zip(sim_is_led, sim_is_lapse, sim_ai_rel_t, sim_ai_vals):
    if not is_led:
        continue

    if (not is_lapse) and (rel_t is not None) and (ai_vals is not None) and (len(rel_t) > 0):
        order = np.argsort(rel_t)
        rel_t_sorted = rel_t[order]
        ai_vals_sorted = ai_vals[order]
        rel_t_unique, unique_idx = np.unique(rel_t_sorted, return_index=True)
        ai_vals_unique = ai_vals_sorted[unique_idx]

        if len(rel_t_unique) >= 2:
            mask = (ai_window_rel_grid >= rel_t_unique[0]) & (ai_window_rel_grid <= rel_t_unique[-1])
            if np.any(mask):
                sim_ai_window_led_on_matrix[led_on_row, mask] = np.interp(
                    ai_window_rel_grid[mask], rel_t_unique, ai_vals_unique
                )
        else:
            nearest_idx = np.argmin(np.abs(ai_window_rel_grid - rel_t_unique[0]))
            sim_ai_window_led_on_matrix[led_on_row, nearest_idx] = ai_vals_unique[0]

    led_on_row += 1

# Aborts only (RT < t_stim), matching data visualization
sim_rts_on = [rt for rt, is_led, t_stim in zip(sim_rts, sim_is_led, sim_t_stims) if is_led and rt < t_stim]
sim_rts_off = [rt for rt, is_led, t_stim in zip(sim_rts, sim_is_led, sim_t_stims) if (not is_led) and rt < t_stim]

sim_rts_wrt_led_on = [
    rt - t_led
    for rt, is_led, t_led, t_stim in zip(sim_rts, sim_is_led, sim_t_LEDs, sim_t_stims)
    if is_led and rt < t_stim
]
sim_rts_wrt_led_off = [
    rt - t_led
    for rt, is_led, t_led, t_stim in zip(sim_rts, sim_is_led, sim_t_LEDs, sim_t_stims)
    if (not is_led) and rt < t_stim
]

print(f"Simulation (aborts only): {len(sim_rts_on)} LED ON, {len(sim_rts_off)} LED OFF")

# %%
# =============================================================================
# Real data arrays for plotting
# =============================================================================
data_rts_on = fit_df[(fit_df['LED_trial'] == 1) & (fit_df['RT'] < fit_df['t_stim'])]['RT'].values
data_rts_off = fit_df[(fit_df['LED_trial'] == 0) & (fit_df['RT'] < fit_df['t_stim'])]['RT'].values

df_on_aborts = fit_df[(fit_df['LED_trial'] == 1) & (fit_df['RT'] < fit_df['t_stim'])]
df_off_aborts = fit_df[(fit_df['LED_trial'] == 0) & (fit_df['RT'] < fit_df['t_stim'])]

data_rts_wrt_led_on = (df_on_aborts['RT'] - df_on_aborts['t_LED']).values
data_rts_wrt_led_off = (df_off_aborts['RT'] - df_off_aborts['t_LED']).values

print(f"Real data (aborts only): {len(data_rts_on)} LED ON, {len(data_rts_off)} LED OFF")

if USE_DYNAMIC_BOUND_DROP:
    mode_tag = (
        f"dyn_s{int(round(DYNAMIC_DROP_START_S*1000)):03d}"
        f"_t{int(round(DYNAMIC_DROP_TARGET_TIME_S*1000)):03d}"
        f"_drop{abs(DYNAMIC_DROP_BY_THETA):.2f}"
    ).replace('.', 'p')
else:
    mode_tag = f"const_{BOUND_DECREASE_PARAM:+.2f}".replace('+', 'p').replace('-', 'm').replace('.', 'p')
out_prefix = f"bound_drop_real_{file_tag}_{mode_tag}"

# %%
# =============================================================================
# Plot 1: RTD wrt fixation (data vs sim)
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

bins = np.arange(0, T_MAX_RTD, 0.05)
bin_centers = (bins[1:] + bins[:-1]) / 2

data_hist_on = safe_hist_density(data_rts_on, bins)
sim_hist_on = safe_hist_density(sim_rts_on, bins)

data_hist_off = safe_hist_density(data_rts_off, bins)
sim_hist_off = safe_hist_density(sim_rts_off, bins)

axes[0].plot(bin_centers, data_hist_on, label='Data (aborts)', lw=2, alpha=0.7, color='b')
axes[0].plot(bin_centers, sim_hist_on, label='Sim (bound-drop)', lw=2, alpha=0.7, color='g')
axes[0].set_xlabel('RT (s)', fontsize=12)
axes[0].set_ylabel('Density', fontsize=12)
axes[0].set_title(f'LED ON Trials - {animal_label}', fontsize=14)
axes[0].legend(fontsize=10)

axes[1].plot(bin_centers, data_hist_off, label='Data (aborts)', lw=2, alpha=0.7, color='b')
axes[1].plot(bin_centers, sim_hist_off, label='Sim (bound-drop)', lw=2, alpha=0.7, color='g')
axes[1].set_xlabel('RT (s)', fontsize=12)
axes[1].set_ylabel('Density', fontsize=12)
axes[1].set_title(f'LED OFF Trials - {animal_label}', fontsize=14)
axes[1].legend(fontsize=10)

plt.tight_layout()
out_rtd = f'{out_prefix}_rtd_comparison.pdf'
plt.savefig(out_rtd, bbox_inches='tight')
print(f"RTD comparison saved as '{out_rtd}'")
plt.show()

# %%
# =============================================================================
# Plot 2: RT wrt LED (data vs sim)
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

bins_wrt_led = np.arange(-3, 3, 0.05)
bin_centers_wrt_led = (bins_wrt_led[1:] + bins_wrt_led[:-1]) / 2

data_hist_on_wrt_led = safe_hist_density(data_rts_wrt_led_on, bins_wrt_led)
sim_hist_on_wrt_led = safe_hist_density(sim_rts_wrt_led_on, bins_wrt_led)

data_hist_off_wrt_led = safe_hist_density(data_rts_wrt_led_off, bins_wrt_led)
sim_hist_off_wrt_led = safe_hist_density(sim_rts_wrt_led_off, bins_wrt_led)

axes[0].plot(bin_centers_wrt_led, data_hist_on_wrt_led, label='Data (aborts)', lw=2, alpha=0.7, color='b')
axes[0].plot(bin_centers_wrt_led, sim_hist_on_wrt_led, label='Sim (bound-drop)', lw=2, alpha=0.7, color='r')
axes[0].axvline(x=0, color='k', linestyle='--', alpha=0.5, label='LED onset')
axes[0].axvline(x=DEL_M_PLUS_DEL_LED, color='r', linestyle=':', alpha=0.5, label=f'del_m_plus_del_LED={DEL_M_PLUS_DEL_LED:.2f}')
axes[0].set_xlabel('RT - t_LED (s)', fontsize=12)
axes[0].set_ylabel('Density', fontsize=12)
axes[0].set_title(f'LED ON Trials - {animal_label}', fontsize=14)
axes[0].legend(fontsize=10)

axes[1].plot(bin_centers_wrt_led, data_hist_off_wrt_led, label='Data (aborts)', lw=2, alpha=0.7, color='b')
axes[1].plot(bin_centers_wrt_led, sim_hist_off_wrt_led, label='Sim (bound-drop)', lw=2, alpha=0.7, color='r')
axes[1].axvline(x=0, color='k', linestyle='--', alpha=0.5, label='LED onset')
axes[1].set_xlabel('RT - t_LED (s)', fontsize=12)
axes[1].set_ylabel('Density', fontsize=12)
axes[1].set_title(f'LED OFF Trials - {animal_label}', fontsize=14)
axes[1].legend(fontsize=10)

plt.tight_layout()
out_wrt = f'{out_prefix}_rt_wrt_led_comparison.pdf'
plt.savefig(out_wrt, bbox_inches='tight')
print(f"RT wrt LED comparison saved as '{out_wrt}'")
plt.show()

# %%
# =============================================================================
# Plot 3: RT wrt LED - abort-rate weighted (area = abort fraction)
# =============================================================================
bins_wrt_led_rate = np.arange(-3, 3, 0.005)
bin_centers_wrt_led_rate = (bins_wrt_led_rate[1:] + bins_wrt_led_rate[:-1]) / 2

n_all_data_on = len(fit_df[fit_df['LED_trial'] == 1])
n_all_data_off = len(fit_df[fit_df['LED_trial'] == 0])
n_aborts_data_on = len(data_rts_wrt_led_on)
n_aborts_data_off = len(data_rts_wrt_led_off)
frac_data_on = n_aborts_data_on / n_all_data_on if n_all_data_on > 0 else 0.0
frac_data_off = n_aborts_data_off / n_all_data_off if n_all_data_off > 0 else 0.0

n_all_sim_on = sum(1 for is_led in sim_is_led if is_led)
n_all_sim_off = sum(1 for is_led in sim_is_led if not is_led)
n_aborts_sim_on = sum(1 for rt, is_led, t_stim in zip(sim_rts, sim_is_led, sim_t_stims) if is_led and rt < t_stim)
n_aborts_sim_off = sum(1 for rt, is_led, t_stim in zip(sim_rts, sim_is_led, sim_t_stims) if (not is_led) and rt < t_stim)
frac_sim_on = n_aborts_sim_on / n_all_sim_on if n_all_sim_on > 0 else 0.0
frac_sim_off = n_aborts_sim_off / n_all_sim_off if n_all_sim_off > 0 else 0.0

fig, ax = plt.subplots(figsize=(15, 6))

data_hist_on_rate = safe_hist_density(data_rts_wrt_led_on, bins_wrt_led_rate) * frac_data_on
data_hist_off_rate = safe_hist_density(data_rts_wrt_led_off, bins_wrt_led_rate) * frac_data_off
sim_hist_on_rate = safe_hist_density(sim_rts_wrt_led_on, bins_wrt_led_rate) * frac_sim_on
sim_hist_off_rate = safe_hist_density(sim_rts_wrt_led_off, bins_wrt_led_rate) * frac_sim_off

# ax.plot(bin_centers_wrt_led_rate, data_hist_on_rate, label=f'Data LED ON (frac={frac_data_on:.2f})', lw=2, alpha=0.7, color='r', linestyle='-')
# ax.plot(bin_centers_wrt_led_rate, data_hist_off_rate, label=f'Data LED OFF (frac={frac_data_off:.2f})', lw=2, alpha=0.7, color='b', linestyle='-')



ax.plot(bin_centers_wrt_led_rate, sim_hist_on_rate, label=f'Sim LED ON (frac={frac_sim_on:.2f})', lw=2, alpha=0.7, color='r', linestyle='--')
ax.plot(bin_centers_wrt_led_rate, sim_hist_off_rate, label=f'Sim LED OFF (frac={frac_sim_off:.2f})', lw=2, alpha=0.7, color='b', linestyle='--')

ax.axvline(x=0, color='k', linestyle='--', alpha=0.5, label='LED onset')
ax.axvline(x=DEL_M_PLUS_DEL_LED, color='g', linestyle=':', alpha=0.5, label=f'del_m_plus_del_LED={DEL_M_PLUS_DEL_LED:.2f}')

ax.set_xlabel('RT - t_LED (s)', fontsize=12)
ax.set_ylabel('Rate (area = fraction)', fontsize=12)
ax.set_title(
    (
        f'RT wrt LED (area-weighted) - {animal_label} | '
        f'{"dynamic drop" if USE_DYNAMIC_BOUND_DROP else f"theta drop={bound_drop:.2f}"} '
        f'(from og theta={THETA_A:.2f})'
    ),
    fontsize=14,
)
ax.legend(fontsize=9)
# ax.set_xlim(*XLIM_WRT_LED)
ax.set_xlim(-0.2,0.5)
plt.tight_layout()
out_rate = f'{out_prefix}_rt_wrt_led_rate.pdf'
plt.savefig(out_rate, bbox_inches='tight')
print(f"RT wrt LED rate plot saved as '{out_rate}'")
plt.show()

print(f"\nScript complete for {animal_label}!")

# %%
# =============================================================================
# AI distribution in LED ON trials for [T-5ms, T]
# T = t_effect + delta_window, where t_effect = t_LED - del_a_minus_del_LED.
# =============================================================================
valid_points_per_trial = np.sum(~np.isnan(sim_ai_window_led_on_matrix), axis=1)
n_led_on_with_samples = int(np.sum(valid_points_per_trial > 0))
n_led_on_total = sim_ai_window_led_on_matrix.shape[0]

ai_vals_all_window = sim_ai_window_led_on_matrix[~np.isnan(sim_ai_window_led_on_matrix)]
ai_vals_at_T = sim_ai_window_led_on_matrix[:, -1]
ai_vals_at_T = ai_vals_at_T[~np.isnan(ai_vals_at_T)]

print(
    f"AI window capture (LED ON): {n_led_on_with_samples}/{n_led_on_total} trials have samples "
    f"in [{-AI_WINDOW_MS:.1f}ms, 0ms] around T=t_effect+delta_window"
)
print(
    f"Window alignment check: T - t_effect = "
    f"{DELTA_LED_FOR_AI_WINDOW * 1000:.3f} ms "
    f"(0 ms means exact bound-switch alignment)"
)
# %%
fig, ax = plt.subplots(figsize=(8, 5))

if len(ai_vals_all_window) > 0:
    ax.hist(ai_vals_all_window, bins=40, density=True, alpha=0.8, color='tab:purple')
ax.set_xlabel('AI value')
ax.set_ylabel('Density')
ax.set_title('DV distribution in [T-5ms, T+5ms], T: time pt of change')

ax.axvline(THETA_A, color='k', ls=':', alpha=0.8, label=f'theta_pre={THETA_A:.2f}')
if USE_DYNAMIC_BOUND_DROP:
    ax.axvline(
        theta_at_dynamic_target,
        color='r',
        ls=':',
        alpha=0.8,
        label=f'theta@{DYNAMIC_DROP_TARGET_TIME_S*1000:.0f}ms={theta_at_dynamic_target:.2f}',
    )
else:
    ax.axvline(theta_post_led, color='r', ls=':', alpha=0.8, label=f'theta_post={theta_post_led:.2f}')
ax.legend(fontsize=9)

plt.tight_layout()
out_ai_window = f'{out_prefix}_ai_window_led_on_distribution.pdf'
plt.savefig(out_ai_window, bbox_inches='tight')
print(f"AI window distribution plot saved as '{out_ai_window}'")
plt.show()

# %%
# =============================================================================
# Survival diagnostic for RT wrt LED (simulation only)
# =============================================================================
def empirical_cdf_survival(values, bins):
    values = np.asarray(values)
    n = len(values)
    centers = (bins[1:] + bins[:-1]) / 2

    if n == 0:
        zeros = np.zeros(len(centers))
        return centers, zeros, zeros

    sorted_vals = np.sort(values)
    cdf = np.searchsorted(sorted_vals, centers, side='right') / n
    survival = 1.0 - cdf
    return centers, cdf, survival


bins_diag = np.arange(-2, 2, 0.005)

centers_diag, cdf_on_diag, surv_on_diag = empirical_cdf_survival(sim_rts_wrt_led_on, bins_diag)
_, cdf_off_diag, surv_off_diag = empirical_cdf_survival(sim_rts_wrt_led_off, bins_diag)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(centers_diag, surv_on_diag, color='r', lw=2, label='Sim LED ON (1-CDF)')
ax.plot(centers_diag, surv_off_diag, color='b', lw=2, label='Sim LED OFF (1-CDF)')
ax.plot(centers_diag, cdf_on_diag, color='r', lw=2, ls='--', label='Sim LED ON (CDF)')
ax.plot(centers_diag, cdf_off_diag, color='b', lw=2, ls='--', label='Sim LED OFF (CDF)')
ax.axvline(0, color='k', ls='--', alpha=0.5)
ax.axvline(DEL_M_PLUS_DEL_LED, color='g', ls=':', alpha=0.6, label='del_m_plus_del_LED')
ax.set_title('CDF and 1 - CDF')
ax.set_xlabel('RT - t_LED (s)')
ax.set_ylabel('Probability')
ax.set_ylim(0, 1.01)
ax.set_xlim(-0.2, 0.5)
ax.legend(fontsize=9)

plt.tight_layout()
out_diag = f'{out_prefix}_rt_wrt_led_survival.pdf'
plt.savefig(out_diag, bbox_inches='tight')
print(f"Survival plot saved as '{out_diag}'")
plt.show()
# %%
# Area-weighted CDF: directly comparable to area-weighted RTD curves.
cdf_on_weighted = frac_sim_on * cdf_on_diag
cdf_off_weighted = frac_sim_off * cdf_off_diag
surv_on_weighted = frac_sim_on * (1.0 - cdf_on_diag)
surv_off_weighted = frac_sim_off * (1.0 - cdf_off_diag)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(centers_diag, cdf_on_weighted, color='r', lw=2, label=f'Sim LED ON weighted CDF (frac={frac_sim_on:.2f})')
ax.plot(centers_diag, cdf_off_weighted, color='b', lw=2, label=f'Sim LED OFF weighted CDF (frac={frac_sim_off:.2f})')
# ax.plot(centers_diag, surv_on_weighted, color='r', lw=2, ls='--', label='Sim LED ON weighted (1-CDF)')
# ax.plot(centers_diag, surv_off_weighted, color='b', lw=2, ls='--', label='Sim LED OFF weighted (1-CDF)')

ax.axvline(0, color='k', ls='--', alpha=0.5)
ax.axvline(DEL_M_PLUS_DEL_LED, color='g', ls=':', alpha=0.6, label='del_m_plus_del_LED')
ax.set_title('Area-weighted CDF and 1-CDF (area = frac of aborts)')
ax.set_xlabel('RT - t_LED (s)')
ax.set_ylabel('Abort fraction')
ax.set_xlim(-0.2, 0.5)
ax.set_ylim(0, max(frac_sim_on, frac_sim_off) * 1.05 + 1e-6)
ax.legend(fontsize=9)

plt.tight_layout()
out_diag_weighted = f'{out_prefix}_rt_wrt_led_weighted_cdf.pdf'
plt.savefig(out_diag_weighted, bbox_inches='tight')
print(f"Weighted CDF plot saved as '{out_diag_weighted}'")
plt.show()


# %%
# =============================================================================
# Plot 4: Decision bound over time (0 = t_LED)
# =============================================================================
bound_time_wrt_led = np.arange(-0.2, 0.6 + 1e-12, 0.001)
bound_led_on = np.full_like(bound_time_wrt_led, THETA_A, dtype=float)
bound_led_off = np.full_like(bound_time_wrt_led, THETA_A, dtype=float)

dt_since_effect_wrt_led = bound_time_wrt_led + DEL_A_MINUS_DEL_LED
effect_onset_wrt_led = -DEL_A_MINUS_DEL_LED

if not USE_DYNAMIC_BOUND_DROP:
    bound_led_on[dt_since_effect_wrt_led >= 0.0] = theta_post_led
else:
    mask_dynamic = dt_since_effect_wrt_led >= DYNAMIC_DROP_START_S
    bound_led_on[mask_dynamic] = (
        THETA_A - dynamic_drop_slope * (dt_since_effect_wrt_led[mask_dynamic] - DYNAMIC_DROP_START_S)
    )
    bound_led_on = np.maximum(bound_led_on, theta_led_min)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(bound_time_wrt_led, bound_led_on, color='r', lw=2, label='LED ON bound')
ax.plot(bound_time_wrt_led, bound_led_off, color='b', lw=2, ls='--', label='LED OFF bound')

ax.axvline(0.0, color='k', ls='--', alpha=0.6, label='t_LED (0)')
ax.axvline(effect_onset_wrt_led, color='tab:green', ls=':', alpha=0.8, label='t_effect')
if USE_DYNAMIC_BOUND_DROP:
    ax.axvline(
        effect_onset_wrt_led + DYNAMIC_DROP_START_S,
        color='tab:orange',
        ls=':',
        alpha=0.8,
        label='dynamic drop start',
    )
    ax.axvline(
        effect_onset_wrt_led + DYNAMIC_DROP_TARGET_TIME_S,
        color='tab:red',
        ls=':',
        alpha=0.8,
        label='dynamic target time',
    )

ax.set_title(f'Bound vs t')
ax.set_xlabel('Time wrt LED onset (s)')
ax.set_ylabel('bound')
ax.set_xlim(-0.2, 0.6)
# ax.set_ylim(bottom=max(0.0, THETA_FLOOR - 0.05))
ax.set_ylim(1.5, 2.5)

ax.legend(fontsize=9)

plt.tight_layout()
out_bound = f'{out_prefix}_bound_wrt_led.pdf'
plt.savefig(out_bound, bbox_inches='tight')
print(f"Bound-vs-time plot saved as '{out_bound}'")
plt.show()

# %%
