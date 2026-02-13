"""
Average RT wrt LED abort-rate plot across all animals.

For each animal:
  1. Load data from out_LED.csv (same filtering as per-animal script).
  2. Load VBMC posterior from vbmc_real_{animal}_CORR_ID_fit.pkl → param_means.
  3. Compute data scaled histogram (density × abort fraction) of RT − t_LED.
  4. Simulate 200k trials with that animal's param_means → compute sim scaled histogram.

Then average the per-animal scaled histograms and plot.
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import pandas as pd
import os
import sys
import pickle
from tqdm import tqdm

sys.path.append('../fit_each_condn')
from psiam_tied_dv_map_utils_with_PDFs import stupid_f_integral, d_A_RT
from post_LED_censor_utils import cum_A_t_fn

# %%
# =============================================================================
# Global settings
# =============================================================================
ANIMALS = [92, 93, 98, 99, 100, 103]
T_trunc = 0.3
N_trials_sim = int(200e3)
SIM_PKL_PATH = 'vbmc_compare_LED_fit_average_animals_simdata.pkl'
LOAD_SAVED_SIM = os.path.exists(SIM_PKL_PATH)

# %%
# =============================================================================
# Simulation function (same as per-animal script)
# =============================================================================
def simulate_proactive_single_bound(V_A_base, V_A_post_LED, theta_A, t_LED, t_stim,
                                     del_a_minus_del_LED, del_m_plus_del_LED, is_led_trial, dt=1e-4):
    AI = 0
    t = 0
    dB = np.sqrt(dt)
    while True:
        if is_led_trial and t >= t_LED - del_a_minus_del_LED:
            V_A = V_A_post_LED
        else:
            V_A = V_A_base
        AI += V_A * dt + np.random.normal(0, dB)
        t += dt
        if AI >= theta_A:
            RT = t + (del_m_plus_del_LED + del_a_minus_del_LED)
            return RT

# %%
# =============================================================================
# Theory PDF functions (CORR_ID parameterization)
# =============================================================================
def PA_with_LEDON_2_adapted(t, v, vON, a, del_a_minus_del_LED, del_m_plus_del_LED, tled, T_trunc=None):
    if T_trunc is not None and t <= T_trunc:
        return 0
    tp = tled - del_a_minus_del_LED
    t_post_led = t - tled - del_m_plus_del_LED
    t_shift_off = t - (del_m_plus_del_LED + del_a_minus_del_LED)
    t_shift_on = t - tled - del_m_plus_del_LED
    if tp > 0 and t_post_led <= 0:
        pdf = d_A_RT(v * a, t_shift_off / (a**2)) / (a**2)
    else:
        if tp <= 0:
            pdf = d_A_RT(vON * a, t_shift_on / (a**2)) / (a**2)
        else:
            pdf = stupid_f_integral(v, vON, a, t_post_led, tp)
    if T_trunc is not None:
        t_pts_cdf = np.arange(0, T_trunc + 0.001, 0.001)
        pdf_vals = np.array([
            PA_with_LEDON_2_adapted(ti, v, vON, a, del_a_minus_del_LED, del_m_plus_del_LED, tled, None)
            for ti in t_pts_cdf
        ])
        cdf_trunc = np.trapz(pdf_vals, t_pts_cdf)
        pdf = pdf / (1 - cdf_trunc)
    return pdf


def led_off_cdf(t, v, a, del_a_minus_del_LED, del_m_plus_del_LED):
    if t <= del_m_plus_del_LED + del_a_minus_del_LED:
        return 0
    return cum_A_t_fn(t - (del_m_plus_del_LED + del_a_minus_del_LED), v, a)


def led_off_pdf_truncated(t, v, a, del_a_minus_del_LED, del_m_plus_del_LED, T_trunc):
    if t <= T_trunc:
        return 0
    if t <= del_m_plus_del_LED + del_a_minus_del_LED:
        return 0
    pdf = d_A_RT(v * a, (t - (del_m_plus_del_LED + del_a_minus_del_LED)) / (a**2)) / (a**2)
    if T_trunc is not None:
        cdf_trunc = led_off_cdf(T_trunc, v, a, del_a_minus_del_LED, del_m_plus_del_LED)
        trunc_factor = 1 - cdf_trunc
        if trunc_factor <= 0:
            return 0
        pdf = pdf / trunc_factor
    return pdf


# %%
# =============================================================================
# Load and filter data (once for all animals)
# =============================================================================
og_df = pd.read_csv('../out_LED.csv')
df = og_df[og_df['repeat_trial'].isin([0, 2]) | og_df['repeat_trial'].isna()]
df = df[df['session_type'].isin([7])]
df = df[df['training_level'].isin([16])]
df = df.dropna(subset=['intended_fix', 'LED_onset_time', 'timed_fix'])
df = df[(df['abort_event'] == 3) | (df['success'].isin([1, -1]))]
df = df[~((df['abort_event'] == 3) & (df['timed_fix'] < T_trunc))]

print(f"Available animals in data: {df['animal'].unique()}")

# %%
# =============================================================================
# Per-animal loop: load data + simulate (or load cached pkl)
# =============================================================================
if LOAD_SAVED_SIM:
    print(f"\nLoading cached simulation data from {SIM_PKL_PATH}...")
    with open(SIM_PKL_PATH, 'rb') as f:
        sim_cache = pickle.load(f)
    per_animal = sim_cache['per_animal']
    print(f"Loaded data for {len(per_animal)} animals.")
else:
    per_animal = {}  # animal -> dict of raw arrays

    for animal in ANIMALS:
        print(f"\n{'='*60}")
        print(f"Processing animal {animal}")
        print(f"{'='*60}")

        # --- Load VP fit and get param_means ---
        vp_path = f'vbmc_real_{animal}_CORR_ID_fit.pkl'
        assert os.path.exists(vp_path), f"Missing {vp_path}"
        with open(vp_path, 'rb') as f:
            vp = pickle.load(f)
        vp_samples = vp.sample(int(1e5))[0]
        param_means = np.mean(vp_samples, axis=0)
        print(f"  param_means: {param_means}")

        # --- Build fit_df for this animal (same as per-animal script) ---
        df_animal = df[df['animal'] == animal]
        df_on = df_animal[df_animal['LED_trial'] == 1]
        df_off = df_animal[df_animal['LED_trial'].isin([0, np.nan])]

        df_on_fit = pd.DataFrame({
            'RT': df_on['timed_fix'].values,
            't_stim': df_on['intended_fix'].values,
            't_LED': (df_on['intended_fix'] - df_on['LED_onset_time']).values,
            'LED_trial': 1
        })
        df_off_fit = pd.DataFrame({
            'RT': df_off['timed_fix'].values,
            't_stim': df_off['intended_fix'].values,
            't_LED': (df_off['intended_fix'] - df_off['LED_onset_time']).values,
            'LED_trial': 0
        })
        fit_df = pd.concat([df_on_fit, df_off_fit], ignore_index=True)
        fit_df = fit_df[~((fit_df['RT'] < fit_df['t_stim']) & (fit_df['RT'] <= T_trunc))]

        # Paired LED/stim times for simulation sampling
        stim_times = df_animal['intended_fix'].values
        LED_times = (df_animal['intended_fix'] - df_animal['LED_onset_time']).values
        n_trials_data = len(stim_times)

        # --- Data: extract raw RT-wrt-LED arrays + counts ---
        df_on_aborts = fit_df[(fit_df['LED_trial'] == 1) & (fit_df['RT'] < fit_df['t_stim']) & (fit_df['RT'] > T_trunc)]
        df_off_aborts = fit_df[(fit_df['LED_trial'] == 0) & (fit_df['RT'] < fit_df['t_stim']) & (fit_df['RT'] > T_trunc)]
        data_rts_wrt_led_on = (df_on_aborts['RT'] - df_on_aborts['t_LED']).values
        data_rts_wrt_led_off = (df_off_aborts['RT'] - df_off_aborts['t_LED']).values

        n_all_data_on = len(fit_df[fit_df['LED_trial'] == 1])
        n_all_data_off = len(fit_df[fit_df['LED_trial'] == 0])

        print(f"  Data: {len(data_rts_wrt_led_on)} ON aborts / {n_all_data_on} total, "
              f"{len(data_rts_wrt_led_off)} OFF aborts / {n_all_data_off} total")

        # --- Simulate with this animal's fitted params ---
        print(f"  Simulating {N_trials_sim} trials...")

        def simulate_single_trial_fit(pm=param_means, lt=LED_times, st=stim_times, nd=n_trials_data):
            is_led_trial = np.random.random() < 1/3
            trial_idx = np.random.randint(nd)
            t_LED = lt[trial_idx]
            t_stim = st[trial_idx]
            rt = simulate_proactive_single_bound(
                pm[0], pm[1], pm[2],
                t_LED if is_led_trial else None, t_stim,
                pm[3], pm[4],
                is_led_trial
            )
            return rt, is_led_trial, t_LED, t_stim

        sim_results = Parallel(n_jobs=30)(
            delayed(simulate_single_trial_fit)() for _ in range(N_trials_sim)
        )

        sim_rts = np.array([r[0] for r in sim_results])
        sim_is_led = np.array([r[1] for r in sim_results])
        sim_t_LEDs = np.array([r[2] for r in sim_results])
        sim_t_stims = np.array([r[3] for r in sim_results])

        # RT wrt LED for sim aborts
        mask_on = sim_is_led & (sim_rts > T_trunc) & (sim_rts < sim_t_stims)
        mask_off = (~sim_is_led) & (sim_rts > T_trunc) & (sim_rts < sim_t_stims)
        sim_rts_wrt_led_on = (sim_rts - sim_t_LEDs)[mask_on]
        sim_rts_wrt_led_off = (sim_rts - sim_t_LEDs)[mask_off]

        n_all_sim_on = int(sim_is_led.sum())
        n_all_sim_off = int((~sim_is_led).sum())

        print(f"  Sim:  {len(sim_rts_wrt_led_on)} ON aborts / {n_all_sim_on} total, "
              f"{len(sim_rts_wrt_led_off)} OFF aborts / {n_all_sim_off} total")

        per_animal[animal] = {
            'param_means': param_means,
            'data_rts_wrt_led_on': data_rts_wrt_led_on,
            'data_rts_wrt_led_off': data_rts_wrt_led_off,
            'n_all_data_on': n_all_data_on,
            'n_all_data_off': n_all_data_off,
            'sim_rts_wrt_led_on': sim_rts_wrt_led_on,
            'sim_rts_wrt_led_off': sim_rts_wrt_led_off,
            'n_all_sim_on': n_all_sim_on,
            'n_all_sim_off': n_all_sim_off,
        }

    # Save to pkl
    with open(SIM_PKL_PATH, 'wb') as f:
        pickle.dump({'per_animal': per_animal, 'T_trunc': T_trunc, 'N_trials_sim': N_trials_sim}, f)
    print(f"\nSaved simulation data to {SIM_PKL_PATH}")

# %%
# =============================================================================
# Compute per-animal scaled histograms and average
# =============================================================================
bins_wrt_led = np.arange(-3, 3, 0.025)
bin_centers_wrt_led = (bins_wrt_led[1:] + bins_wrt_led[:-1]) / 2

all_data_hist_on = []
all_data_hist_off = []
all_sim_hist_on = []
all_sim_hist_off = []
all_del_m_plus_del_LED = []

for animal in ANIMALS:
    d = per_animal[animal]
    all_del_m_plus_del_LED.append(d['param_means'][4])

    # Data scaled histograms
    frac_data_on = len(d['data_rts_wrt_led_on']) / d['n_all_data_on'] if d['n_all_data_on'] > 0 else 0
    frac_data_off = len(d['data_rts_wrt_led_off']) / d['n_all_data_off'] if d['n_all_data_off'] > 0 else 0
    h_on, _ = np.histogram(d['data_rts_wrt_led_on'], bins=bins_wrt_led, density=True)
    h_off, _ = np.histogram(d['data_rts_wrt_led_off'], bins=bins_wrt_led, density=True)
    all_data_hist_on.append(h_on * frac_data_on)
    all_data_hist_off.append(h_off * frac_data_off)

    # Sim scaled histograms
    frac_sim_on = len(d['sim_rts_wrt_led_on']) / d['n_all_sim_on'] if d['n_all_sim_on'] > 0 else 0
    frac_sim_off = len(d['sim_rts_wrt_led_off']) / d['n_all_sim_off'] if d['n_all_sim_off'] > 0 else 0
    h_on_s, _ = np.histogram(d['sim_rts_wrt_led_on'], bins=bins_wrt_led, density=True)
    h_off_s, _ = np.histogram(d['sim_rts_wrt_led_off'], bins=bins_wrt_led, density=True)
    all_sim_hist_on.append(h_on_s * frac_sim_on)
    all_sim_hist_off.append(h_off_s * frac_sim_off)

mean_data_hist_on = np.mean(all_data_hist_on, axis=0)
mean_data_hist_off = np.mean(all_data_hist_off, axis=0)
mean_sim_hist_on = np.mean(all_sim_hist_on, axis=0)
mean_sim_hist_off = np.mean(all_sim_hist_off, axis=0)
mean_del_m_plus_del_LED = np.mean(all_del_m_plus_del_LED)

print(f"\nMean del_m_plus_del_LED across animals: {mean_del_m_plus_del_LED:.4f}")

# %%
# =============================================================================
# Plot: Average RT wrt LED (area-weighted)
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(bin_centers_wrt_led, mean_data_hist_on,
        label='Data LED ON', lw=2, alpha=0.4, color='r', linestyle='-')
ax.plot(bin_centers_wrt_led, mean_data_hist_off,
        label='Data LED OFF', lw=2, alpha=0.4, color='b', linestyle='-')
ax.plot(bin_centers_wrt_led, mean_sim_hist_on,
        label='Sim LED ON', lw=2, alpha=0.7, color='r', linestyle='--')
ax.plot(bin_centers_wrt_led, mean_sim_hist_off,
        label='Sim LED OFF', lw=2, alpha=0.7, color='b', linestyle='--')

ax.set_xlim(-1,1)
ax.axvline(x=0, color='k', linestyle='--', alpha=0.5, label='LED onset')
ax.axvline(x=mean_del_m_plus_del_LED, color='g', linestyle=':', alpha=0.5,
           label=f'mean del_m_plus_del_LED={mean_del_m_plus_del_LED:.2f}')
ax.set_xlabel('RT - t_LED (s)', fontsize=12)
ax.set_ylabel('Rate (area = fraction)', fontsize=12)
ax.set_title('RT wrt LED (area-weighted) — Average across animals', fontsize=14)
ax.legend(fontsize=9)

plt.tight_layout()
outfile = 'vbmc_average_animals_CORR_ID_rt_wrt_led_rate.pdf'
plt.savefig(outfile, bbox_inches='tight')
print(f"\nSaved: {outfile}")
plt.show()

# %%
# =============================================================================
# Theoretical RTD wrt LED (average across animals)
# =============================================================================
N_mc_theory = 3000
t_pts_wrt_led = np.arange(-2, 2, 0.005)

all_theory_on = []
all_theory_off = []

for animal in ANIMALS:
    print(f"\nComputing theory for animal {animal}...")
    pm = per_animal[animal]['param_means']

    # Get LED_times and stim_times from df
    df_animal = df[df['animal'] == animal]
    stim_times_a = df_animal['intended_fix'].values
    LED_times_a = (df_animal['intended_fix'] - df_animal['LED_onset_time']).values
    n_trials_a = len(stim_times_a)

    rtd_on_acc = np.zeros(len(t_pts_wrt_led))
    rtd_off_acc = np.zeros(len(t_pts_wrt_led))

    for i in tqdm(range(N_mc_theory)):
        trial_idx = np.random.randint(n_trials_a)
        t_led = LED_times_a[trial_idx]
        t_stim = stim_times_a[trial_idx]

        if t_stim <= T_trunc:
            continue

        t_pts_wrt_fix = t_pts_wrt_led + t_led

        # Pre-compute LED ON truncation factor for this t_led (avoid recomputing per t-point)
        cdf_pts = np.arange(0, T_trunc + 0.001, 0.001)
        cdf_vals = np.array([
            PA_with_LEDON_2_adapted(ti, pm[0], pm[1], pm[2], pm[3], pm[4], t_led, None)
            for ti in cdf_pts
        ])
        cdf_trunc_on = np.trapz(cdf_vals, cdf_pts)
        trunc_factor_on = 1 - cdf_trunc_on

        for j, (t_wrt_led, t_wrt_fix) in enumerate(zip(t_pts_wrt_led, t_pts_wrt_fix)):
            if (t_wrt_fix <= T_trunc) and (t_wrt_fix < t_stim):
                continue
            elif t_wrt_fix >= t_stim:
                continue
            else:
                # LED OFF
                rtd_off_acc[j] += led_off_pdf_truncated(
                    t_wrt_fix, pm[0], pm[2], pm[3], pm[4], T_trunc
                )
                # LED ON (use T_trunc=None, apply pre-computed factor)
                pdf_on = PA_with_LEDON_2_adapted(
                    t_wrt_fix, pm[0], pm[1], pm[2], pm[3], pm[4], t_led, None
                )
                if trunc_factor_on > 0:
                    pdf_on = pdf_on / trunc_factor_on
                rtd_on_acc[j] += pdf_on

    rtd_on_acc /= N_mc_theory
    rtd_off_acc /= N_mc_theory
    all_theory_on.append(rtd_on_acc)
    all_theory_off.append(rtd_off_acc)
    print(f"  area ON={np.trapz(rtd_on_acc, t_pts_wrt_led):.4f}, "
          f"OFF={np.trapz(rtd_off_acc, t_pts_wrt_led):.4f}")

mean_theory_on = np.mean(all_theory_on, axis=0)
mean_theory_off = np.mean(all_theory_off, axis=0)

# %%
# =============================================================================
# Plot: Theory + Data (average across animals)
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(bin_centers_wrt_led, mean_data_hist_on,
        label='Data LED ON', lw=2, alpha=0.4, color='r', linestyle='-')
ax.plot(bin_centers_wrt_led, mean_data_hist_off,
        label='Data LED OFF', lw=2, alpha=0.4, color='b', linestyle='-')
ax.plot(t_pts_wrt_led, mean_theory_on,
        label='Theory LED ON', lw=2, alpha=0.7, color='r', linestyle='--')
ax.plot(t_pts_wrt_led, mean_theory_off,
        label='Theory LED OFF', lw=2, alpha=0.7, color='b', linestyle='--')

ax.set_xlim(-0.2, 0.4)
ax.axvline(x=0, color='k', linestyle='--', alpha=0.5, label='LED onset')
ax.axvline(x=mean_del_m_plus_del_LED, color='g', linestyle=':', alpha=0.5,
           label=f'mean del_m_plus_del_LED={mean_del_m_plus_del_LED:.2f}')
ax.set_xlabel('RT - t_LED (s)', fontsize=12)
ax.set_ylabel('Rate (area = fraction)', fontsize=12)
ax.set_title('RT wrt LED — Theory vs Data (average across animals)', fontsize=14)
ax.legend(fontsize=9)

plt.tight_layout()
outfile_theory = 'vbmc_average_animals_CORR_ID_rt_wrt_led_theory_vs_data.pdf'
plt.savefig(outfile_theory, bbox_inches='tight')
print(f"\nSaved: {outfile_theory}")
plt.show()
# %%
# =============================================================================
# Plot: Theory vs Sim (confirmation that theory agrees with sim)
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(bin_centers_wrt_led, mean_sim_hist_on,
        label='Sim LED ON', lw=2, alpha=0.7, color='r', linestyle='-')
ax.plot(bin_centers_wrt_led, mean_sim_hist_off,
        label='Sim LED OFF', lw=2, alpha=0.7, color='b', linestyle='-')
ax.plot(t_pts_wrt_led, mean_theory_on,
        label='Theory LED ON', lw=2, alpha=0.4, color='r', linestyle='--')
ax.plot(t_pts_wrt_led, mean_theory_off,
        label='Theory LED OFF', lw=2, alpha=0.4, color='b', linestyle='--')

ax.set_xlim(-1,1)
ax.axvline(x=0, color='k', linestyle='--', alpha=0.5, label='LED onset')
ax.axvline(x=mean_del_m_plus_del_LED, color='g', linestyle=':', alpha=0.5,
           label=f'mean del_m_plus_del_LED={mean_del_m_plus_del_LED:.2f}')
ax.set_xlabel('RT - t_LED (s)', fontsize=12)
ax.set_ylabel('Rate (area = fraction)', fontsize=12)
ax.set_title('RT wrt LED — Theory vs Sim (average across animals)', fontsize=14)
ax.legend(fontsize=9)

plt.tight_layout()
outfile_ts = 'vbmc_average_animals_CORR_ID_rt_wrt_led_theory_vs_sim.pdf'
plt.savefig(outfile_ts, bbox_inches='tight')
print(f"\nSaved: {outfile_ts}")
plt.show()
# %%
# =============================================================================
# Print param means table (each row = animal, columns = params)
# =============================================================================
param_labels = ['V_A_base', 'V_A_post_LED', 'theta_A', 'del_a_minus_del_LED', 'del_m_plus_del_LED(ms)']

print(f"\n{'Animal':<10}", end='')
for lbl in param_labels:
    print(f"{lbl:<22}", end='')
print()
print("-" * (10 + 22 * len(param_labels)))
for animal in ANIMALS:
    pm = per_animal[animal]['param_means']
    print(f"{animal:<10}", end='')
    for vi, val in enumerate(pm):
        if vi == 4:
            print(f"{1000*val:<22.4f}", end='')
        else:
            print(f"{val:<22.4f}", end='')
    print()
# %%
# =============================================================================
# Function: histogram of VP samples for a given parameter per animal
# =============================================================================
ALL_PARAM_LABELS = ['V_A_base', 'V_A_post_LED', 'theta_A', 'del_a_minus_del_LED', 'del_m_plus_del_LED']

def plot_param_posterior(param_name, scale=1.0, unit=''):
    param_idx = ALL_PARAM_LABELS.index(param_name)
    display_name = f'{param_name} ({unit})' if unit else param_name

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(ANIMALS)))

    print(f"\n{'Animal':<10} {'mean':<15} {'std':<15}  [{display_name}]")
    print("-" * 50)
    for idx, animal in enumerate(ANIMALS):
        vp_path = f'vbmc_real_{animal}_CORR_ID_fit.pkl'
        with open(vp_path, 'rb') as f:
            vp = pickle.load(f)
        samples = vp.sample(int(1e5))[0]
        vals = samples[:, param_idx] * scale
        m, s = np.mean(vals), np.std(vals)
        print(f"{animal:<10} {m:<15.4f} {s:<15.4f}")
        ax.hist(vals, bins=50, histtype='step', lw=2, density=True,
                color=colors[idx], label=f'Animal {animal} ({m:.2f} ± {s:.2f})')

    ax.set_xlabel(display_name, fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'{display_name} per animal', fontsize=14)
    ax.legend(fontsize=9)
    plt.tight_layout()
    outfile_hist = f'vbmc_average_animals_CORR_ID_{param_name}_hist.pdf'
    plt.savefig(outfile_hist, bbox_inches='tight')
    print(f"\nSaved: {outfile_hist}")
    plt.show()
# %%
plot_param_posterior('del_m_plus_del_LED', scale=1000, unit='ms')

# %%
plot_param_posterior('V_A_post_LED', scale=1, unit='')

# %%
# =============================================================================
# Posterior of V_A_post_LED - V_A_base per animal
# =============================================================================
def plot_param_diff_posterior(param_name_a, param_name_b, scale=1.0, unit=''):
    idx_a = ALL_PARAM_LABELS.index(param_name_a)
    idx_b = ALL_PARAM_LABELS.index(param_name_b)
    diff_name = f'{param_name_a} - {param_name_b}'
    display_name = f'{diff_name} ({unit})' if unit else diff_name

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(ANIMALS)))

    print(f"\n{'Animal':<10} {'mean':<15} {'std':<15}  [{display_name}]")
    print("-" * 50)
    for idx, animal in enumerate(ANIMALS):
        vp_path = f'vbmc_real_{animal}_CORR_ID_fit.pkl'
        with open(vp_path, 'rb') as f:
            vp = pickle.load(f)
        samples = vp.sample(int(1e5))[0]
        vals = (samples[:, idx_a] - samples[:, idx_b]) * scale
        m, s = np.mean(vals), np.std(vals)
        print(f"{animal:<10} {m:<15.4f} {s:<15.4f}")
        ax.hist(vals, bins=50, histtype='step', lw=2, density=True,
                color=colors[idx], label=f'Animal {animal} ({m:.2f} ± {s:.2f})')

    ax.set_xlabel(display_name, fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'{display_name} per animal', fontsize=14)
    ax.legend(fontsize=9)
    plt.tight_layout()
    outfile_hist = f'vbmc_average_animals_CORR_ID_{param_name_a}_minus_{param_name_b}_hist.pdf'
    plt.savefig(outfile_hist, bbox_inches='tight')
    print(f"\nSaved: {outfile_hist}")
    plt.show()

plot_param_diff_posterior('V_A_post_LED', 'V_A_base')
# %%
# =============================================================================
# Check VBMC convergence status per animal
# =============================================================================
print(f"\n{'Animal':<10} {'ELBO':<15} {'ELBO SD':<12} {'Convergence':<15} {'Stable':<10}")
print("-" * 62)
for animal in ANIMALS:
    with open(f'vbmc_real_{animal}_CORR_ID_results.pkl', 'rb') as f:
        rs = pickle.load(f)['results_summary']
    with open(f'vbmc_real_{animal}_CORR_ID_fit.pkl', 'rb') as f:
        vp = pickle.load(f)
    stable = bool(vp.stats['stable'])
    print(f"{animal:<10} {rs['elbo']:<15.2f} {rs['elbo_sd']:<12.4f} {rs['convergence_status']:<15} {stable}")
# %%
