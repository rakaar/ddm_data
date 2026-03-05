"""
Schematic plot: RT wrt LED (theory vs data), using saved VBMC results.
"""

# %%
from pathlib import Path
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import corner
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
EXPORT_DIR = ROOT / "fct_march_26"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.append(str(ROOT / "fit_each_condn"))
from psiam_tied_dv_map_utils_with_PDFs import stupid_f_integral, d_A_RT

# %%
# =============================================================================
# PARAMETERS
# =============================================================================
ANIMAL_ID = None  # None for all animals, or integer index into unique animals
LOAD_SAVED_RESULTS = True

SESSION_TYPE = 7
TRAINING_LEVEL = 16

N_POSTERIOR_SAMPLES = int(1e5)
N_MC_THEORY_WRT_LED = 5000
THEORY_X_RANGE = (-2.0, 2.0)
THEORY_DT = 0.005
HIST_X_RANGE = (-3.0, 3.0)

RNG_SEED = 42
SHOW_PLOT = True

# %%
# =============================================================================
# Load and filter data
# =============================================================================
og_df = pd.read_csv(ROOT / "out_LED.csv")

df = og_df[og_df["repeat_trial"].isin([0, 2]) | og_df["repeat_trial"].isna()]
df = df[df["session_type"].isin([SESSION_TYPE])]
df = df[df["training_level"].isin([TRAINING_LEVEL])]
df = df.dropna(subset=["intended_fix", "LED_onset_time", "timed_fix"])
df = df[(df["abort_event"] == 3) | (df["success"].isin([1, -1]))]

unique_animals = df["animal"].unique()
if ANIMAL_ID is not None:
    animal_name = unique_animals[ANIMAL_ID]
    df_all = df[df["animal"] == animal_name]
    animal_label = f"Animal {animal_name}"
    file_tag = f"animal_{animal_name}"
else:
    df_all = df
    animal_label = "All Animals Aggregated"
    file_tag = "all_animals"

df_on = df_all[df_all["LED_trial"] == 1]
df_off = df_all[df_all["LED_trial"] == 0]

# For fitting-style columns
df_on_fit = pd.DataFrame(
    {
        "RT": df_on["timed_fix"].values,
        "t_stim": df_on["intended_fix"].values,
        "t_LED": (df_on["intended_fix"] - df_on["LED_onset_time"]).values,
        "LED_trial": 1,
    }
)
df_off_fit = pd.DataFrame(
    {
        "RT": df_off["timed_fix"].values,
        "t_stim": df_off["intended_fix"].values,
        "t_LED": (df_off["intended_fix"] - df_off["LED_onset_time"]).values,
        "LED_trial": 0,
    }
)
fit_df = pd.concat([df_on_fit, df_off_fit], ignore_index=True)

# Keep trial-level pairing
stim_times = df_all["intended_fix"].values
LED_times = (df_all["intended_fix"] - df_all["LED_onset_time"]).values
n_trials_data = len(stim_times)

# %%
# =============================================================================
# Load saved VBMC results and get posterior means
# =============================================================================
if not LOAD_SAVED_RESULTS:
    raise ValueError("This schematic script supports only LOAD_SAVED_RESULTS=True.")

vp_pkl_path = ROOT / "fitting_aborts" / f"vbmc_real_{file_tag}_fit_NO_TRUNC_with_lapse.pkl"
if not vp_pkl_path.exists():
    raise FileNotFoundError(f"Saved fit not found: {vp_pkl_path}")

with open(vp_pkl_path, "rb") as f:
    vp = pickle.load(f)

vp_samples = vp.sample(N_POSTERIOR_SAMPLES)[0]
param_means = np.mean(vp_samples, axis=0)

print(f"Loaded VP: {vp_pkl_path}")
print(f"Posterior means: {np.round(param_means, 4)}")

# %%
# =============================================================================
# Minimal model pieces used by RTD wrt LED theory plot
# =============================================================================
def PA_with_LEDON_2_adapted(t, v, vON, a, del_a_minus_del_LED, del_m_plus_del_LED, tled):
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
    return pdf


def led_off_pdf(t, v, a, del_a_minus_del_LED, del_m_plus_del_LED):
    if t <= del_m_plus_del_LED + del_a_minus_del_LED:
        return 0.0
    return d_A_RT(v * a, (t - (del_m_plus_del_LED + del_a_minus_del_LED)) / (a**2)) / (a**2)


def lapse_pdf(t, beta):
    return beta * np.exp(-beta * t)


def safe_density_hist(values, bins):
    if len(values) == 0:
        return np.zeros(len(bins) - 1)
    hist, _ = np.histogram(values, bins=bins, density=True)
    return np.nan_to_num(hist, nan=0.0, posinf=0.0, neginf=0.0)


def save_plot_payload(plot_name, payload):
    payload_path = EXPORT_DIR / f"{plot_name}_{file_tag}.pkl"
    with open(payload_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"Saved plot data: {payload_path}")


# %%
# =============================================================================
# Compute RTD wrt LED (theory) and RTD wrt LED (data)
# =============================================================================


rng = np.random.default_rng(RNG_SEED)
t_pts_wrt_led_theory = np.arange(THEORY_X_RANGE[0], THEORY_X_RANGE[1], THEORY_DT)
rtd_theory_on_wrt_led = np.zeros(len(t_pts_wrt_led_theory))
rtd_theory_off_wrt_led = np.zeros(len(t_pts_wrt_led_theory))

V_A_base, V_A_post_LED, theta_A, del_a_minus_del_LED, del_m_plus_del_LED, lapse_prob, beta_lapse = param_means

for _ in tqdm(range(N_MC_THEORY_WRT_LED), desc="Theory wrt LED"):
    trial_idx = rng.integers(n_trials_data)
    t_led = LED_times[trial_idx]
    t_stim = stim_times[trial_idx]
    if t_stim <= 0:
        continue

    t_pts_wrt_fix = t_pts_wrt_led_theory + t_led
    mask = (t_pts_wrt_fix > 0) & (t_pts_wrt_fix < t_stim)
    if not np.any(mask):
        continue

    proactive_on = np.array(
        [
            PA_with_LEDON_2_adapted(
                t_wrt_fix,
                V_A_base,
                V_A_post_LED,
                theta_A,
                del_a_minus_del_LED,
                del_m_plus_del_LED,
                t_led,
            )
            for t_wrt_fix in t_pts_wrt_fix[mask]
        ]
    )
    proactive_off = np.array(
        [led_off_pdf(t_wrt_fix, V_A_base, theta_A, del_a_minus_del_LED, del_m_plus_del_LED) for t_wrt_fix in t_pts_wrt_fix[mask]]
    )
    lapse_vals = lapse_pdf(t_pts_wrt_fix[mask], beta_lapse)

    rtd_theory_on_wrt_led[mask] += (1 - lapse_prob) * proactive_on + lapse_prob * lapse_vals
    rtd_theory_off_wrt_led[mask] += (1 - lapse_prob) * proactive_off + lapse_prob * lapse_vals

rtd_theory_on_wrt_led /= N_MC_THEORY_WRT_LED
rtd_theory_off_wrt_led /= N_MC_THEORY_WRT_LED

# %%
HIST_DT = 0.01

# Data aborts (RT < t_stim)
df_on_aborts = fit_df[(fit_df["LED_trial"] == 1) & (fit_df["RT"] < fit_df["t_stim"])]
df_off_aborts = fit_df[(fit_df["LED_trial"] == 0) & (fit_df["RT"] < fit_df["t_stim"])]
data_rts_wrt_led_on = (df_on_aborts["RT"] - df_on_aborts["t_LED"]).values
data_rts_wrt_led_off = (df_off_aborts["RT"] - df_off_aborts["t_LED"]).values

n_all_data_on = len(fit_df[fit_df["LED_trial"] == 1])
n_all_data_off = len(fit_df[fit_df["LED_trial"] == 0])
frac_data_on = len(data_rts_wrt_led_on) / n_all_data_on if n_all_data_on > 0 else 0.0
frac_data_off = len(data_rts_wrt_led_off) / n_all_data_off if n_all_data_off > 0 else 0.0

bins_wrt_led = np.arange(HIST_X_RANGE[0], HIST_X_RANGE[1], HIST_DT)
bin_centers_wrt_led = (bins_wrt_led[1:] + bins_wrt_led[:-1]) / 2
data_hist_on_scaled = safe_density_hist(data_rts_wrt_led_on, bins_wrt_led) * frac_data_on
data_hist_off_scaled = safe_density_hist(data_rts_wrt_led_off, bins_wrt_led) * frac_data_off


# =============================================================================
# Plot: RT wrt LED (data + theory only)
# =============================================================================
PLOT_XLIM = (-0.3, 0.4)
PLOT_XLIM_MS = (PLOT_XLIM[0] * 1000.0, PLOT_XLIM[1] * 1000.0)

data_x_ms = bin_centers_wrt_led * 1000.0
theory_x_ms = t_pts_wrt_led_theory * 1000.0

fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(
    data_x_ms,
    data_hist_on_scaled,
    lw=2,
    alpha=0.4,
    color="r",
    linestyle="-",
)
ax.plot(
    data_x_ms,
    data_hist_off_scaled,
    lw=2,
    alpha=0.4,
    color="b",
    linestyle="-",
)
ax.plot(
    theory_x_ms,
    rtd_theory_on_wrt_led,
    lw=2.4,
    alpha=1.0,
    color="r",
    linestyle="-",
)
ax.plot(
    theory_x_ms,
    rtd_theory_off_wrt_led,
    lw=2.4,
    alpha=1.0,
    color="b",
    linestyle="-",
)

ax.axvline(x=0, color="0.2", linestyle="--", alpha=0.7)
ax.axvline(
    x=del_m_plus_del_LED * 1000.0,
    color="0.2",
    linestyle=":",
    alpha=0.7,
)
ax.set_xlabel("RT wrt LED onset (ms)", fontsize=16)
ax.set_ylabel("Abort Rate (Hz)", fontsize=16)
ax.set_xticks([-400, 0, 400])
ax.set_xlim(PLOT_XLIM_MS)
ax.set_yticks([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_linewidth(2.5)
ax.spines["left"].set_linewidth(2.5)
ax.tick_params(axis="x", labelsize=13, width=2.0, length=7)
ax.tick_params(axis="y", width=0, length=0)
plt.tight_layout()

out_path = ROOT / f"schematic_{file_tag}_rt_wrt_led_theory_data.pdf"
plt.savefig(out_path, bbox_inches="tight")
print(f"Saved plot: {out_path}")
save_plot_payload(
    "rt_wrt_led_theory_data",
    {
        "plot_name": "rt_wrt_led_theory_data",
        "animal_label": animal_label,
        "file_tag": file_tag,
        "pdf_path": str(out_path),
        "data_x_ms": data_x_ms,
        "theory_x_ms": theory_x_ms,
        "data_hist_on_scaled": data_hist_on_scaled,
        "data_hist_off_scaled": data_hist_off_scaled,
        "rtd_theory_on_wrt_led": rtd_theory_on_wrt_led,
        "rtd_theory_off_wrt_led": rtd_theory_off_wrt_led,
        "frac_data_on": frac_data_on,
        "frac_data_off": frac_data_off,
        "data_rts_wrt_led_on": data_rts_wrt_led_on,
        "data_rts_wrt_led_off": data_rts_wrt_led_off,
        "del_m_plus_del_LED_ms": del_m_plus_del_LED * 1000.0,
        "xlim_ms": PLOT_XLIM_MS,
        "xticks_ms": [-400, 0, 400],
        "xlabel": "RT wrt LED onset (ms)",
        "ylabel": "Abort Rate (Hz)",
    },
)

if SHOW_PLOT:
    plt.show()

# =============================================================================
# Schematic: single-bound trajectory with drift change at LED onset + delay
# =============================================================================
SCHEMATIC_X_MIN_MS = -50
SCHEMATIC_X_MAX_MS = 50

SCHEMATIC_DT_MS = 1.0
SCHEMATIC_NOISE_SD = 1.0
SCHEMATIC_POST_MULTIPLIER = 10.0
SCHEMATIC_DELTA_A_SHRINK = 4.0
SCHEMATIC_PRE_LINE_GAIN = 5
SCHEMATIC_DELTA_M_MS = 12.0

# Use fitted drifts and bound.
slope_pre = V_A_base
slope_post = V_A_post_LED * SCHEMATIC_POST_MULTIPLIER
bound_level = theta_A

# Drift switch occurs at LED onset (0) + del_m_plus_del_LED.
# t_switch_s = del_m_plus_del_LED
t_switch_s = 0.01

t_switch_ms = t_switch_s * 1000.0
# Start accumulation close to y-axis so delta_a is short:
# delta_a length = (0 - x_min) / SCHEMATIC_DELTA_A_SHRINK
accum_start_ms = SCHEMATIC_X_MIN_MS + (0.0 - SCHEMATIC_X_MIN_MS) / SCHEMATIC_DELTA_A_SHRINK

t_ms = np.arange(SCHEMATIC_X_MIN_MS, SCHEMATIC_X_MAX_MS + SCHEMATIC_DT_MS, SCHEMATIC_DT_MS)
t_s = t_ms / 1000.0
dt_s = SCHEMATIC_DT_MS / 1000.0

# Piecewise deterministic backbone, continuous at switch.
a_switch = 0.45 * bound_level
det_traj = np.where(
    t_s <= t_switch_s,
    a_switch + slope_pre * (t_s - t_switch_s),
    a_switch + slope_post * (t_s - t_switch_s),
)

# Simulate one evidence trajectory around the deterministic backbone.
rng_schematic = np.random.default_rng(RNG_SEED + 7)
a = np.full_like(t_s, np.nan, dtype=float)
start_idx = np.searchsorted(t_ms, accum_start_ms, side="left")
if start_idx >= len(t_s):
    raise ValueError("delta_a pushes accumulation start outside schematic time range.")

a[start_idx] = max(0.0, det_traj[start_idx] + rng_schematic.normal(scale=0.03 * bound_level))
for i in range(start_idx + 1, len(t_s)):
    if np.isnan(a[i - 1]):
        break
    drift = slope_pre if t_s[i - 1] <= t_switch_s else slope_post
    noise = SCHEMATIC_NOISE_SD * np.sqrt(dt_s) * rng_schematic.normal()
    a[i] = a[i - 1] + drift * dt_s + noise
    if a[i] >= bound_level:
        a[i] = bound_level
        if i + 1 < len(a):
            a[i + 1 :] = np.nan
        break
    if a[i] < 0:
        a[i] = 0.0

mask_pre = t_s <= t_switch_s
mask_post = t_s > t_switch_s

fig, ax = plt.subplots(figsize=(7.2, 4.4))

# One trajectory in two colors (pre/post switch).
ax.plot(t_ms[mask_pre], a[mask_pre], color="blue", lw=3, alpha=0.4, zorder=3)
ax.plot(t_ms[mask_post], a[mask_post], color="red", lw=3, alpha=0.4, zorder=3)

# Bound, LED onset, and switch markers.
ax.axhline(bound_level, color="0.35", lw=1.6, ls="--", zorder=2)
ax.axvline(0, color="0.35", lw=1.2, ls="-.", alpha=0.8, zorder=2)
ax.axvline(t_switch_ms, color="0.35", lw=1.4, ls=":", zorder=2)

# Drift reference lines (thin solid, no arrows).
valid_idx = np.where(np.isfinite(a))[0]
pre_idx = valid_idx[t_ms[valid_idx] <= t_switch_ms]
post_idx = valid_idx[t_ms[valid_idx] > t_switch_ms]

# Pre-LED line starts at delayed accumulation onset.
if len(pre_idx) > 0:
    pre_x0_ms = t_ms[pre_idx[0]]
    pre_y0 = a[pre_idx[0]]
else:
    pre_x0_ms = accum_start_ms
    pre_y0 = np.interp(pre_x0_ms, t_ms, det_traj)

pre_x1_ms = min(t_switch_ms, SCHEMATIC_X_MAX_MS)
pre_line_slope_per_ms = (slope_pre * SCHEMATIC_PRE_LINE_GAIN) / 1000.0
post_line_slope_per_ms = max(slope_post / 1000.0, 1e-9)
pre_line_slope_per_ms = min(pre_line_slope_per_ms, 0.85 * post_line_slope_per_ms)
pre_y1 = pre_y0 + pre_line_slope_per_ms * (pre_x1_ms - pre_x0_ms)
if pre_x1_ms > pre_x0_ms:
    ax.plot(
        [pre_x0_ms, pre_x1_ms],
        [pre_y0, pre_y1],
        color="#1f77b4",
        lw=1.4,
        alpha=0.95,
        zorder=6,
    )

# Post-LED line starts at first point after switch.
if len(post_idx) > 0:
    post_i0 = post_idx[0]
    post_x0_ms, post_y0 = t_ms[post_i0], a[post_i0]
else:
    post_x0_ms, post_y0 = t_switch_ms + 1.0, np.interp(t_switch_ms + 1.0, t_ms, det_traj)

if slope_post > 1e-9:
    x_theta_post = post_x0_ms + 1000.0 * (bound_level - post_y0) / slope_post
    post_x1_ms = np.clip(x_theta_post, post_x0_ms, SCHEMATIC_X_MAX_MS)
else:
    post_x1_ms = post_x0_ms
post_y1 = post_y0 + (slope_post / 1000.0) * (post_x1_ms - post_x0_ms)
if post_x1_ms > post_x0_ms:
    ax.plot(
        [post_x0_ms, post_x1_ms],
        [post_y0, post_y1],
        color="#d62728",
        lw=1.4,
        alpha=0.95,
        zorder=6,
    )
post_guide_x_end = post_x1_ms

# Delay annotations: delta_a (ending at trajectory start), delta_LED (LED -> switch),
# delta_m (decision -> RT), and RT line.
hit_idx = np.where(np.isfinite(a) & (a >= bound_level))[0]
if len(hit_idx) > 0:
    decision_x_ms = t_ms[hit_idx[0]]
else:
    decision_x_ms = min(post_guide_x_end, SCHEMATIC_X_MAX_MS - SCHEMATIC_DELTA_M_MS - 1.0)

rt_x_ms = min(SCHEMATIC_X_MAX_MS - 0.8, decision_x_ms + SCHEMATIC_DELTA_M_MS)
delay_y_led = bound_level * 1.00
delay_y_m = delay_y_led

# delta_a arrow starts at y-axis and ends at trajectory/blue-arrow start.
if pre_x0_ms > SCHEMATIC_X_MIN_MS:
    delta_a_start_ms = SCHEMATIC_X_MIN_MS
    ax.annotate(
        "",
        xy=(delta_a_start_ms, pre_y0),
        xytext=(pre_x0_ms, pre_y0),
        arrowprops=dict(
            arrowstyle="<->",
            lw=1.8,
            color="0.6",
            mutation_scale=16,
        ),
        zorder=7,
    )
    ax.text(
        0.5 * (delta_a_start_ms + pre_x0_ms),
        pre_y0 + 0.02 * bound_level,
        r"$\delta_{a}$",
        color="0.55",
        fontsize=14,
        ha="center",
        va="bottom",
    )

ax.annotate(
    "",
    xy=(t_switch_ms, delay_y_led),
    xytext=(0, delay_y_led),
    arrowprops=dict(arrowstyle="<->", lw=1.9, color="0.55", mutation_scale=17),
    zorder=7,
)
ax.text(
    0.5 * t_switch_ms,
    delay_y_led + 0.02 * bound_level,
    r"$\delta_{LED}$",
    color="0.45",
    fontsize=12,
    ha="center",
    va="bottom",
)

if rt_x_ms > decision_x_ms + 0.5:
    ax.annotate(
        "",
        xy=(rt_x_ms, delay_y_m),
        xytext=(decision_x_ms, delay_y_m),
        arrowprops=dict(arrowstyle="<->", lw=1.9, color="0.55", mutation_scale=17),
        zorder=7,
    )
    ax.text(
        0.5 * (decision_x_ms + rt_x_ms),
        delay_y_m + 0.02 * bound_level,
        r"$\delta_{m}$",
        color="0.45",
        fontsize=12,
        ha="center",
        va="bottom",
    )

ax.axvline(rt_x_ms, color="0.55", lw=1.3, ls=(0, (3, 3)), zorder=2)
ax.text(rt_x_ms, bound_level * 1.10, "RT", color="0.35", fontsize=14, ha="center", va="bottom")

pre_mid_x = 0.5 * (pre_x0_ms + pre_x1_ms)
pre_mid_y = 0.5 * (pre_y0 + pre_y1)
post_mid_x = 0.5 * (post_x0_ms + post_x1_ms)
post_mid_y = 0.5 * (post_y0 + post_y1)

# Labels below arrows for readability.
ax.text(pre_mid_x, pre_mid_y - 0.16 * bound_level, r"$V_A^{pre\!-\!LED}$", color="blue", fontsize=12, ha="center")
ax.text(post_mid_x, post_mid_y - 0.16 * bound_level, r"$V_A^{post\!-\!LED}$", color="red", fontsize=12, ha="center")
ax.text(
    SCHEMATIC_X_MAX_MS + 2,
    bound_level + 0.005 * bound_level,
    r"$\theta$",
    color="0.25",
    fontsize=14,
    ha="left",
    clip_on=False,
)

ax.set_ylim(0, bound_level * 1.12)
if SCHEMATIC_X_MIN_MS <= 0 <= SCHEMATIC_X_MAX_MS:
    ax.set_xticks([SCHEMATIC_X_MIN_MS, 0, SCHEMATIC_X_MAX_MS])
else:
    ax.set_xticks([SCHEMATIC_X_MIN_MS, SCHEMATIC_X_MAX_MS])
ax.set_xlim(SCHEMATIC_X_MIN_MS, SCHEMATIC_X_MAX_MS)
ax.set_yticks([])  # No y ticks (as requested)
ax.set_xlabel("Time from LED onset (ms)", fontsize=18)
ax.set_ylabel("", fontsize=12)

# Publication style: remove top/right spines.
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_bounds(0, bound_level)
ax.spines["left"].set_linewidth(2.6)
ax.spines["bottom"].set_linewidth(2.6)
ax.tick_params(axis="x", labelsize=15, width=2.0, length=7)

plt.tight_layout()
schematic_out = ROOT / f"schematic_{file_tag}_drift_switch_single_bound.pdf"
plt.savefig(schematic_out, bbox_inches="tight")
print(f"Saved drift-switch schematic: {schematic_out}")
save_plot_payload(
    "drift_switch_single_bound",
    {
        "plot_name": "drift_switch_single_bound",
        "animal_label": animal_label,
        "file_tag": file_tag,
        "pdf_path": str(schematic_out),
        "t_ms": t_ms,
        "t_s": t_s,
        "trajectory": a,
        "det_traj": det_traj,
        "mask_pre": mask_pre,
        "mask_post": mask_post,
        "bound_level": bound_level,
        "t_switch_ms": t_switch_ms,
        "accum_start_ms": accum_start_ms,
        "pre_line": {
            "x0_ms": pre_x0_ms,
            "y0": pre_y0,
            "x1_ms": pre_x1_ms,
            "y1": pre_y1,
        },
        "post_line": {
            "x0_ms": post_x0_ms,
            "y0": post_y0,
            "x1_ms": post_x1_ms,
            "y1": post_y1,
        },
        "decision_x_ms": decision_x_ms,
        "rt_x_ms": rt_x_ms,
        "delta_m_ms": SCHEMATIC_DELTA_M_MS,
        "xlim_ms": (SCHEMATIC_X_MIN_MS, SCHEMATIC_X_MAX_MS),
        "xticks_ms": (
            [SCHEMATIC_X_MIN_MS, 0, SCHEMATIC_X_MAX_MS]
            if SCHEMATIC_X_MIN_MS <= 0 <= SCHEMATIC_X_MAX_MS
            else [SCHEMATIC_X_MIN_MS, SCHEMATIC_X_MAX_MS]
        ),
        "xlabel": "Time from LED onset (ms)",
        "labels": {
            "pre": r"$V_A^{pre\!-\!LED}$",
            "post": r"$V_A^{post\!-\!LED}$",
            "theta": r"$\theta$",
            "delta_a": r"$\delta_{a}$",
            "delta_led": r"$\delta_{LED}$",
            "delta_m": r"$\delta_{m}$",
            "rt": "RT",
        },
    },
)

if SHOW_PLOT:
    plt.show()

# =============================================================================
# Corner plot (first 5 params only; lapse params excluded)
# =============================================================================

corner_samples = vp_samples[:, :5].copy()
# Convert delay parameters from seconds to milliseconds.
corner_samples[:, 3] *= 1000.0
corner_samples[:, 4] *= 1000.0

corner_labels = [
    r"$V_A^{pre\!-\!LED}$",
    r"$V_A^{post\!-\!LED}$",
    r"$\theta_A$",
    r"$\delta_a-\delta_{LED}$ (ms)",
    r"$\delta_m+\delta_{LED}$ (ms)",
]

fig = corner.corner(
    corner_samples,
    labels=corner_labels,
    show_titles=False,
    color="tab:blue",
    fill_contours=True,
    plot_datapoints=False,
    plot_density=False,
    bins=40,
    levels=[0.50, 0.80, 0.975],
    # First band is transparent so background outside contours stays clear.
    contourf_kwargs={"colors": [(1, 1, 1, 0), "#deebf7", "#9ecae1", "#4292c6"], "alpha": 1.0},
    contour_kwargs={"colors": "tab:blue", "linewidths": 1.2},
    hist_kwargs={"alpha": 0.8},
    quantiles=[0.025, 0.50, 0.975],
)

# Move parameter titles to top (diagonal titles only).
medians = np.median(corner_samples, axis=0)
param_q = np.quantile(corner_samples, [0.025, 0.975], axis=0)
n_dim = corner_samples.shape[1]
axes = np.array(fig.axes).reshape((n_dim, n_dim))


def _fmt_corner_tick(x, pos):
    ax = abs(x)
    if ax >= 100:
        return f"{x:.0f}"
    if ax >= 10:
        return f"{x:.1f}"
    return f"{x:.2f}"


def _pick_first_third_ticks(ticks, lo, hi):
    ticks = np.asarray(ticks, dtype=float)
    ticks = ticks[np.isfinite(ticks)]
    ticks = ticks[(ticks >= min(lo, hi)) & (ticks <= max(lo, hi))]
    if ticks.size >= 3:
        return [ticks[0], ticks[2]]
    if ticks.size >= 2:
        return [ticks[0], ticks[-1]]
    if ticks.size == 1:
        return [ticks[0]]
    return [lo, hi]


# Remove bottom/left axis labels and increase tick font size.
# Use only 2.5% and 97.5% ticks per parameter; show labels only on outer axes.
for i in range(n_dim):
    for j in range(n_dim):
        ax_ij = axes[i, j]
        if i < j:
            # Upper triangle: keep fully empty (no stray tick labels).
            ax_ij.set_axis_off()
            continue

        ax_ij.set_xlabel("")
        ax_ij.set_ylabel("")
        ax_ij.set_xticks([param_q[0, j], param_q[1, j]])
        ax_ij.xaxis.set_major_formatter(FuncFormatter(_fmt_corner_tick))

        if i > j:
            ax_ij.set_yticks([param_q[0, i], param_q[1, i]])
            ax_ij.yaxis.set_major_formatter(FuncFormatter(_fmt_corner_tick))
        else:
            ax_ij.set_yticks([])

        ax_ij.tick_params(axis="both", labelsize=13)
        if i == n_dim - 1:
            ax_ij.tick_params(axis="x", labelrotation=0)
        else:
            ax_ij.tick_params(axis="x", labelbottom=False)
        if j != 0:
            ax_ij.tick_params(axis="y", labelleft=False)

for i in range(n_dim):
    axes[i, i].set_title(corner_labels[i], fontsize=17, pad=14)
    axes[i, i].axvline(medians[i], color="tab:blue", ls=":", lw=1.8, alpha=0.95)

# Left-side y labels (same text as top titles) for readability.
for i in range(1, n_dim):
    axes[i, 0].set_ylabel(corner_labels[i], fontsize=16, labelpad=16)

corner_out = ROOT / f"schematic_{file_tag}_corner_5params.pdf"
plt.savefig(corner_out, bbox_inches="tight")
print(f"Saved corner plot: {corner_out}")
save_plot_payload(
    "corner_5params",
    {
        "plot_name": "corner_5params",
        "animal_label": animal_label,
        "file_tag": file_tag,
        "pdf_path": str(corner_out),
        "corner_samples": corner_samples,
        "corner_labels": corner_labels,
        "medians": medians,
        "param_q": param_q,
        "levels": [0.50, 0.80, 0.975],
        "bins": 40,
        "quantiles": [0.025, 0.50, 0.975],
        "style": {
            "color": "tab:blue",
            "fill_contours": True,
            "plot_datapoints": False,
            "plot_density": False,
            "contourf_colors": [(1, 1, 1, 0), "#deebf7", "#9ecae1", "#4292c6"],
        },
    },
)

if SHOW_PLOT:
    plt.show()

# %%
# =============================================================================
# Plot: RT wrt LED (data + theory only) - zoomed, 5 ms bins
# =============================================================================
HIST_DT_ZOOM = 0.005
PLOT_XLIM_ZOOM = (-0.2, 0.2)
PLOT_XLIM_ZOOM_MS = (PLOT_XLIM_ZOOM[0] * 1000.0, PLOT_XLIM_ZOOM[1] * 1000.0)

bins_wrt_led_zoom = np.arange(HIST_X_RANGE[0], HIST_X_RANGE[1], HIST_DT_ZOOM)
bin_centers_wrt_led_zoom = (bins_wrt_led_zoom[1:] + bins_wrt_led_zoom[:-1]) / 2.0
data_hist_on_zoom_scaled = safe_density_hist(data_rts_wrt_led_on, bins_wrt_led_zoom) * frac_data_on
data_hist_off_zoom_scaled = safe_density_hist(data_rts_wrt_led_off, bins_wrt_led_zoom) * frac_data_off

data_x_zoom_ms = bin_centers_wrt_led_zoom * 1000.0
theory_x_zoom_ms = t_pts_wrt_led_theory * 1000.0

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(
    data_x_zoom_ms,
    data_hist_on_zoom_scaled,
    lw=2,
    alpha=0.4,
    color="r",
    linestyle="-",
)
ax.plot(
    data_x_zoom_ms,
    data_hist_off_zoom_scaled,
    lw=2,
    alpha=0.4,
    color="b",
    linestyle="-",
)
ax.plot(
    theory_x_zoom_ms,
    rtd_theory_on_wrt_led,
    lw=2.4,
    alpha=1.0,
    color="r",
    linestyle="-",
)
ax.plot(
    theory_x_zoom_ms,
    rtd_theory_off_wrt_led,
    lw=2.4,
    alpha=1.0,
    color="b",
    linestyle="-",
)

ax.axvline(x=0, color="0.2", linestyle="--", alpha=0.7)
ax.axvline(
    x=del_m_plus_del_LED * 1000.0,
    color="0.2",
    linestyle=":",
    alpha=0.7,
)
ax.set_xlabel("RT wrt LED onset (ms)", fontsize=16)
ax.set_ylabel("Abort Rate (Hz)", fontsize=16)
ax.set_xticks([-200, 0, 200])
ax.set_xlim(PLOT_XLIM_ZOOM_MS)
ax.set_yticks([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_linewidth(2.5)
ax.spines["left"].set_linewidth(2.5)
ax.tick_params(axis="x", labelsize=13, width=2.0, length=7)
ax.tick_params(axis="y", width=0, length=0)
plt.tight_layout()

rt_led_zoom_out = ROOT / f"schematic_{file_tag}_rt_wrt_led_theory_data_zoom_5ms.pdf"
plt.savefig(rt_led_zoom_out, bbox_inches="tight")
print(f"Saved zoomed RT wrt LED plot: {rt_led_zoom_out}")
save_plot_payload(
    "rt_wrt_led_theory_data_zoom_5ms",
    {
        "plot_name": "rt_wrt_led_theory_data_zoom_5ms",
        "animal_label": animal_label,
        "file_tag": file_tag,
        "pdf_path": str(rt_led_zoom_out),
        "data_x_ms": data_x_zoom_ms,
        "theory_x_ms": theory_x_zoom_ms,
        "data_hist_on_scaled": data_hist_on_zoom_scaled,
        "data_hist_off_scaled": data_hist_off_zoom_scaled,
        "rtd_theory_on_wrt_led": rtd_theory_on_wrt_led,
        "rtd_theory_off_wrt_led": rtd_theory_off_wrt_led,
        "frac_data_on": frac_data_on,
        "frac_data_off": frac_data_off,
        "bin_size_s": HIST_DT_ZOOM,
        "xlim_ms": PLOT_XLIM_ZOOM_MS,
        "xlabel": "RT wrt LED onset (ms)",
        "ylabel": "Abort Rate (Hz)",
    },
)

if SHOW_PLOT:
    plt.show()

# %%
# =============================================================================
# Plot: RTD wrt fixation (data + theory, ON/OFF together)
# =============================================================================
max_fix_s = float(np.nanmax(stim_times))
t_pts_wrt_fix_theory = np.arange(0.0, max_fix_s + THEORY_DT, THEORY_DT)
rtd_theory_on_wrt_fix = np.zeros(len(t_pts_wrt_fix_theory))
rtd_theory_off_wrt_fix = np.zeros(len(t_pts_wrt_fix_theory))

for _ in tqdm(range(N_MC_THEORY_WRT_LED), desc="Theory wrt fixation"):
    trial_idx = rng.integers(n_trials_data)
    t_led = LED_times[trial_idx]
    t_stim = stim_times[trial_idx]
    if t_stim <= 0:
        continue

    mask = (t_pts_wrt_fix_theory > 0) & (t_pts_wrt_fix_theory < t_stim)
    if not np.any(mask):
        continue

    proactive_on = np.array(
        [
            PA_with_LEDON_2_adapted(
                t_wrt_fix,
                V_A_base,
                V_A_post_LED,
                theta_A,
                del_a_minus_del_LED,
                del_m_plus_del_LED,
                t_led,
            )
            for t_wrt_fix in t_pts_wrt_fix_theory[mask]
        ]
    )
    proactive_off = np.array(
        [led_off_pdf(t_wrt_fix, V_A_base, theta_A, del_a_minus_del_LED, del_m_plus_del_LED) for t_wrt_fix in t_pts_wrt_fix_theory[mask]]
    )
    lapse_vals = lapse_pdf(t_pts_wrt_fix_theory[mask], beta_lapse)

    rtd_theory_on_wrt_fix[mask] += (1 - lapse_prob) * proactive_on + lapse_prob * lapse_vals
    rtd_theory_off_wrt_fix[mask] += (1 - lapse_prob) * proactive_off + lapse_prob * lapse_vals

rtd_theory_on_wrt_fix /= N_MC_THEORY_WRT_LED
rtd_theory_off_wrt_fix /= N_MC_THEORY_WRT_LED

data_rts_on_fix = df_on_aborts["RT"].values
data_rts_off_fix = df_off_aborts["RT"].values
HIST_DT = 0.01
bins_wrt_fix = np.arange(0.0, max_fix_s + HIST_DT, HIST_DT)
bin_centers_wrt_fix = (bins_wrt_fix[1:] + bins_wrt_fix[:-1]) / 2
data_hist_on_fix_scaled = safe_density_hist(data_rts_on_fix, bins_wrt_fix) * frac_data_on
data_hist_off_fix_scaled = safe_density_hist(data_rts_off_fix, bins_wrt_fix) * frac_data_off

fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(
    bin_centers_wrt_fix * 1000.0,
    data_hist_on_fix_scaled,
    lw=2,
    alpha=0.4,
    color="r",
    linestyle="-",
)
ax.plot(
    bin_centers_wrt_fix * 1000.0,
    data_hist_off_fix_scaled,
    lw=2,
    alpha=0.4,
    color="b",
    linestyle="-",
)
ax.plot(
    t_pts_wrt_fix_theory * 1000.0,
    rtd_theory_on_wrt_fix,
    lw=2.4,
    alpha=1.0,
    color="r",
    linestyle="-",
)
ax.plot(
    t_pts_wrt_fix_theory * 1000.0,
    rtd_theory_off_wrt_fix,
    lw=2.4,
    alpha=1.0,
    color="b",
    linestyle="-",
)
ax.set_xlabel("RT wrt fixation (ms)", fontsize=22)
ax.set_ylabel("Abort Rate (Hz)", fontsize=22)
ax.set_xlim(0, 2000)
ax.set_xticks([0, 500, 1000, 1500, 2000])
ax.set_yticks([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_linewidth(2.5)
ax.spines["left"].set_linewidth(2.5)
ax.tick_params(axis="x", labelsize=18, width=2.6, length=10)
ax.tick_params(axis="y", width=0, length=0)
plt.tight_layout()

fix_out = ROOT / f"schematic_{file_tag}_rtd_wrt_fixation_theory_data.pdf"
plt.savefig(fix_out, bbox_inches="tight")
print(f"Saved RTD wrt fixation plot: {fix_out}")
save_plot_payload(
    "rtd_wrt_fixation_theory_data",
    {
        "plot_name": "rtd_wrt_fixation_theory_data",
        "animal_label": animal_label,
        "file_tag": file_tag,
        "pdf_path": str(fix_out),
        "data_x_ms": bin_centers_wrt_fix * 1000.0,
        "theory_x_ms": t_pts_wrt_fix_theory * 1000.0,
        "data_hist_on_scaled": data_hist_on_fix_scaled,
        "data_hist_off_scaled": data_hist_off_fix_scaled,
        "rtd_theory_on_wrt_fix": rtd_theory_on_wrt_fix,
        "rtd_theory_off_wrt_fix": rtd_theory_off_wrt_fix,
        "frac_data_on": frac_data_on,
        "frac_data_off": frac_data_off,
        "data_rts_on_fix": data_rts_on_fix,
        "data_rts_off_fix": data_rts_off_fix,
        "xlim_ms": (0, 2000),
        "xticks_ms": [0, 500, 1000, 1500, 2000],
        "xlabel": "RT wrt fixation (ms)",
        "ylabel": "Abort Rate (Hz)",
    },
)

if SHOW_PLOT:
    plt.show()

# %%
# t_LED  and t_stim
# show relationship between t_LED and t_stim
# =============================================================================
# Plot: t_LED and t_stim distributions
# =============================================================================
dist_bins = np.arange(0.0, 2 + 0.01, 0.01)
dist_centers = (dist_bins[1:] + dist_bins[:-1]) / 2.0

t_led_vals = fit_df["t_LED"].values
t_stim_vals = fit_df["t_stim"].values

t_led_hist = safe_density_hist(t_led_vals, dist_bins)
t_stim_hist = safe_density_hist(t_stim_vals, dist_bins)

fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(dist_centers, t_led_hist, color="tab:blue", lw=2.6, alpha=0.95, label=r"$t_{LED}$")
ax.plot(dist_centers, t_stim_hist, color="tab:red", lw=2.6, alpha=0.95, label=r"$t_{stim}$")

ax.set_xlim(0.0, 1.0)
ax.set_ylim(0, 3)
ax.set_xlabel("Time (s)", fontsize=22)
ax.set_ylabel("Density (1/s)", fontsize=22)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_linewidth(2.5)
ax.spines["left"].set_linewidth(2.5)
ax.tick_params(axis="both", labelsize=18, width=2.0, length=8)
ax.legend(frameon=False, fontsize=16)
plt.tight_layout()

tled_stim_out = ROOT / f"schematic_{file_tag}_t_led_t_stim_distributions.pdf"
plt.savefig(tled_stim_out, bbox_inches="tight")
print(f"Saved t_LED/t_stim distribution plot: {tled_stim_out}")
save_plot_payload(
    "t_led_t_stim_distributions",
    {
        "plot_name": "t_led_t_stim_distributions",
        "animal_label": animal_label,
        "file_tag": file_tag,
        "pdf_path": str(tled_stim_out),
        "bin_centers_s": dist_centers,
        "bins_s": dist_bins,
        "t_led_hist_density": t_led_hist,
        "t_stim_hist_density": t_stim_hist,
        "t_led_values_s": t_led_vals,
        "t_stim_values_s": t_stim_vals,
        "xlabel": "Time (s)",
        "ylabel": "Density (1/s)",
        "xlim_s": (0.0, 2.5),
    },
)

if SHOW_PLOT:
    plt.show()

# %%
