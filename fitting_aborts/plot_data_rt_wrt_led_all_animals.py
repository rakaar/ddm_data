# %%
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append("../fit_each_condn")
from post_LED_censor_utils import cum_A_t_fn
from psiam_tied_dv_map_utils_with_PDFs import d_A_RT, stupid_f_integral


def scaled_hist(values: np.ndarray, n_total: int, bins: np.ndarray) -> np.ndarray:
    """Return density histogram scaled by abort fraction (area = fraction)."""
    if n_total <= 0 or len(values) == 0:
        return np.zeros(len(bins) - 1)
    frac = len(values) / n_total
    h, _ = np.histogram(values, bins=bins, density=True)
    return h * frac


# %%
# =============================================================================
# Parameters
# =============================================================================
T_trunc = 0.3
BIN_WIDTH = 0.005
BINS_WRT_LED = np.arange(-3.0, 3.0 + BIN_WIDTH, BIN_WIDTH)

THEORY_DT = 0.001
THEORY_RANGE = (-3.0, 3.0)
N_MC_THEORY_WRT_LED = 2000
N_POSTERIOR_SAMPLES = int(1e5)
THEORY_T_PTS = np.arange(THEORY_RANGE[0], THEORY_RANGE[1] + THEORY_DT, THEORY_DT)
AGG_VP_PKL_PATH = os.path.join(os.path.dirname(__file__), "vbmc_real_all_animals_fit.pkl")


def PA_with_LEDON_2_adapted(
    t: float,
    v: float,
    vON: float,
    a: float,
    del_a_minus_del_LED: float,
    del_m_plus_del_LED: float,
    tled: float,
    T_trunc_val: float | None = None,
) -> float:
    if T_trunc_val is not None and t <= T_trunc_val:
        return 0.0

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

    if T_trunc_val is not None:
        t_pts = np.arange(0, T_trunc_val + 0.001, 0.001)
        pdf_vals = np.array(
            [
                PA_with_LEDON_2_adapted(
                    ti,
                    v,
                    vON,
                    a,
                    del_a_minus_del_LED,
                    del_m_plus_del_LED,
                    tled,
                    None,
                )
                for ti in t_pts
            ]
        )
        cdf_trunc = np.trapz(pdf_vals, t_pts)
        trunc_factor = 1 - cdf_trunc
        if trunc_factor > 0:
            pdf = pdf / trunc_factor
        else:
            return 0.0

    return float(pdf)


def led_off_cdf(
    t: float, v: float, a: float, del_a_minus_del_LED: float, del_m_plus_del_LED: float
) -> float:
    if t <= del_m_plus_del_LED + del_a_minus_del_LED:
        return 0.0
    return float(cum_A_t_fn(t - (del_m_plus_del_LED + del_a_minus_del_LED), v, a))


def led_off_pdf_truncated(
    t: float,
    v: float,
    a: float,
    del_a_minus_del_LED: float,
    del_m_plus_del_LED: float,
    T_trunc_val: float,
) -> float:
    if t <= T_trunc_val:
        return 0.0
    if t <= del_m_plus_del_LED + del_a_minus_del_LED:
        return 0.0

    pdf = d_A_RT(v * a, (t - (del_m_plus_del_LED + del_a_minus_del_LED)) / (a**2)) / (
        a**2
    )
    cdf_trunc = led_off_cdf(T_trunc_val, v, a, del_a_minus_del_LED, del_m_plus_del_LED)
    trunc_factor = 1 - cdf_trunc
    if trunc_factor <= 0:
        return 0.0
    return float(pdf / trunc_factor)


def compute_theoretical_rtd_wrt_led(
    t_pts_wrt_led: np.ndarray,
    param_means: np.ndarray,
    stim_times: np.ndarray,
    LED_times: np.ndarray,
    T_trunc_val: float,
    n_mc: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute theory RT-wrt-LED curves (area = abort fraction), no simulation."""
    if len(stim_times) == 0:
        return np.zeros_like(t_pts_wrt_led), np.zeros_like(t_pts_wrt_led)

    rng = np.random.default_rng(seed)
    rtd_theory_on = np.zeros(len(t_pts_wrt_led))
    rtd_theory_off = np.zeros(len(t_pts_wrt_led))
    cdf_pts = np.arange(0, T_trunc_val + 0.001, 0.001)

    for _ in range(n_mc):
        trial_idx = rng.integers(0, len(stim_times))
        t_led = LED_times[trial_idx]
        t_stim = stim_times[trial_idx]

        if t_stim <= T_trunc_val:
            continue

        t_pts_wrt_fix = t_pts_wrt_led + t_led

        cdf_vals = np.array(
            [
                PA_with_LEDON_2_adapted(
                    ti,
                    param_means[0],
                    param_means[1],
                    param_means[2],
                    param_means[3],
                    param_means[4],
                    t_led,
                    None,
                )
                for ti in cdf_pts
            ]
        )
        cdf_trunc_on = np.trapz(cdf_vals, cdf_pts)
        trunc_factor_on = 1 - cdf_trunc_on

        for j, t_wrt_fix in enumerate(t_pts_wrt_fix):
            if (t_wrt_fix <= T_trunc_val and t_wrt_fix < t_stim) or (t_wrt_fix >= t_stim):
                continue

            rtd_theory_off[j] += led_off_pdf_truncated(
                t_wrt_fix,
                param_means[0],
                param_means[2],
                param_means[3],
                param_means[4],
                T_trunc_val,
            )

            pdf_on = PA_with_LEDON_2_adapted(
                t_wrt_fix,
                param_means[0],
                param_means[1],
                param_means[2],
                param_means[3],
                param_means[4],
                t_led,
                None,
            )
            if trunc_factor_on > 0:
                pdf_on = pdf_on / trunc_factor_on
            else:
                pdf_on = 0.0
            rtd_theory_on[j] += pdf_on

    rtd_theory_on /= n_mc
    rtd_theory_off /= n_mc
    return rtd_theory_on, rtd_theory_off


# %%
# =============================================================================
# Load and filter data (same logic as vbmc_real_data_proactive_LED_fit_CORR_ID.py)
# =============================================================================
og_df = pd.read_csv("../out_LED.csv")

df = og_df[og_df["repeat_trial"].isin([0, 2]) | og_df["repeat_trial"].isna()]
session_type = 7
df = df[df["session_type"].isin([session_type])]
training_level = 16
df = df[df["training_level"].isin([training_level])]

df = df.dropna(subset=["intended_fix", "LED_onset_time", "timed_fix"])
df = df[(df["abort_event"] == 3) | (df["success"].isin([1, -1]))]

# Filter out aborts < T_trunc
df = df[~((df["abort_event"] == 3) & (df["timed_fix"] < T_trunc))]

unique_animals = df["animal"].unique()
print(f"Animals found ({len(unique_animals)}): {unique_animals}")


# %%
# =============================================================================
# Compute per-animal RT wrt LED distributions (data only, area-weighted by abort fraction)
# =============================================================================
plot_data = []
per_animal_hist_on = []
per_animal_hist_off = []
pooled_on_vals = []
pooled_off_vals = []
pooled_total_on = 0
pooled_total_off = 0
per_animal_theory_on = []
per_animal_theory_off = []
weights_on = []
weights_off = []

for animal in unique_animals:
    df_animal = df[df["animal"] == animal]

    # Match existing ON/OFF split
    df_on = df_animal[df_animal["LED_trial"] == 1]
    df_off = df_animal[(df_animal["LED_trial"] == 0) | (df_animal["LED_trial"].isna())]

    fit_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "RT": df_on["timed_fix"].values,
                    "t_stim": df_on["intended_fix"].values,
                    "t_LED": (df_on["intended_fix"] - df_on["LED_onset_time"]).values,
                    "LED_trial": 1,
                }
            ),
            pd.DataFrame(
                {
                    "RT": df_off["timed_fix"].values,
                    "t_stim": df_off["intended_fix"].values,
                    "t_LED": (df_off["intended_fix"] - df_off["LED_onset_time"]).values,
                    "LED_trial": 0,
                }
            ),
        ],
        ignore_index=True,
    )

    fit_df = fit_df[~((fit_df["RT"] < fit_df["t_stim"]) & (fit_df["RT"] <= T_trunc))]

    # Aborts only (after truncation): RT < t_stim and RT > T_trunc
    df_on_aborts = fit_df[
        (fit_df["LED_trial"] == 1) & (fit_df["RT"] < fit_df["t_stim"]) & (fit_df["RT"] > T_trunc)
    ]
    df_off_aborts = fit_df[
        (fit_df["LED_trial"] == 0) & (fit_df["RT"] < fit_df["t_stim"]) & (fit_df["RT"] > T_trunc)
    ]

    data_rts_wrt_led_on = (df_on_aborts["RT"] - df_on_aborts["t_LED"]).values
    data_rts_wrt_led_off = (df_off_aborts["RT"] - df_off_aborts["t_LED"]).values

    n_all_data_on = len(fit_df[fit_df["LED_trial"] == 1])
    n_all_data_off = len(fit_df[fit_df["LED_trial"] == 0])
    frac_data_on = len(data_rts_wrt_led_on) / n_all_data_on if n_all_data_on > 0 else 0.0
    frac_data_off = len(data_rts_wrt_led_off) / n_all_data_off if n_all_data_off > 0 else 0.0

    hist_on_scaled = scaled_hist(data_rts_wrt_led_on, n_all_data_on, BINS_WRT_LED)
    hist_off_scaled = scaled_hist(data_rts_wrt_led_off, n_all_data_off, BINS_WRT_LED)

    stim_times = df_animal["intended_fix"].values
    LED_times = (df_animal["intended_fix"] - df_animal["LED_onset_time"]).values
    vp_path = os.path.join(os.path.dirname(__file__), f"vbmc_real_{animal}_CORR_ID_fit.pkl")

    has_theory = False
    param_means = None
    theory_on_wrt_led = np.full_like(THEORY_T_PTS, np.nan, dtype=float)
    theory_off_wrt_led = np.full_like(THEORY_T_PTS, np.nan, dtype=float)

    if os.path.exists(vp_path):
        with open(vp_path, "rb") as f:
            vp = pickle.load(f)
        vp_samples = vp.sample(N_POSTERIOR_SAMPLES)[0]
        param_means = np.mean(vp_samples, axis=0)
        theory_on_wrt_led, theory_off_wrt_led = compute_theoretical_rtd_wrt_led(
            THEORY_T_PTS,
            param_means,
            stim_times,
            LED_times,
            T_trunc,
            n_mc=N_MC_THEORY_WRT_LED,
            seed=int(animal),
        )
        has_theory = True
        print(
            f"Animal {animal}: theory area ON={np.trapz(theory_on_wrt_led, THEORY_T_PTS):.4f}, "
            f"OFF={np.trapz(theory_off_wrt_led, THEORY_T_PTS):.4f}"
        )
    else:
        print(f"Warning: missing fit file for animal {animal}: {vp_path}")

    plot_data.append(
        {
            "animal": animal,
            "hist_on_scaled": hist_on_scaled,
            "hist_off_scaled": hist_off_scaled,
            "frac_data_on": frac_data_on,
            "frac_data_off": frac_data_off,
            "n_on": n_all_data_on,
            "n_off": n_all_data_off,
            "data_rts_wrt_led_on": data_rts_wrt_led_on,
            "data_rts_wrt_led_off": data_rts_wrt_led_off,
            "theory_on_wrt_led": theory_on_wrt_led,
            "theory_off_wrt_led": theory_off_wrt_led,
            "has_theory": has_theory,
            "param_means": param_means,
            "t_led_on": (df_on["intended_fix"] - df_on["LED_onset_time"]).values,
            "t_led_off": (df_off["intended_fix"] - df_off["LED_onset_time"]).values,
            "t_led_all": (df_animal["intended_fix"] - df_animal["LED_onset_time"]).values,
        }
    )

    per_animal_hist_on.append(hist_on_scaled)
    per_animal_hist_off.append(hist_off_scaled)
    pooled_on_vals.append(data_rts_wrt_led_on)
    pooled_off_vals.append(data_rts_wrt_led_off)
    pooled_total_on += n_all_data_on
    pooled_total_off += n_all_data_off
    if has_theory:
        per_animal_theory_on.append(theory_on_wrt_led)
        per_animal_theory_off.append(theory_off_wrt_led)
        weights_on.append(n_all_data_on)
        weights_off.append(n_all_data_off)

# %%

avg_hist_on = np.mean(per_animal_hist_on, axis=0)
avg_hist_off = np.mean(per_animal_hist_off, axis=0)

pooled_on_vals = np.concatenate(pooled_on_vals) if pooled_on_vals else np.array([])
pooled_off_vals = np.concatenate(pooled_off_vals) if pooled_off_vals else np.array([])
agg_hist_on = scaled_hist(pooled_on_vals, pooled_total_on, BINS_WRT_LED)
agg_hist_off = scaled_hist(pooled_off_vals, pooled_total_off, BINS_WRT_LED)

if per_animal_theory_on and np.sum(weights_on) > 0 and np.sum(weights_off) > 0:
    avg_theory_on = np.mean(per_animal_theory_on, axis=0)
    avg_theory_off = np.mean(per_animal_theory_off, axis=0)
else:
    avg_theory_on = np.full_like(THEORY_T_PTS, np.nan, dtype=float)
    avg_theory_off = np.full_like(THEORY_T_PTS, np.nan, dtype=float)

# Aggregate theory: prefer dedicated aggregate fit pkl.
if os.path.exists(AGG_VP_PKL_PATH):
    with open(AGG_VP_PKL_PATH, "rb") as f:
        agg_vp = pickle.load(f)
    agg_vp_samples = agg_vp.sample(N_POSTERIOR_SAMPLES)[0]
    agg_param_means = np.mean(agg_vp_samples, axis=0)
    agg_stim_times = df["intended_fix"].values
    agg_LED_times = (df["intended_fix"] - df["LED_onset_time"]).values
    agg_theory_on, agg_theory_off = compute_theoretical_rtd_wrt_led(
        THEORY_T_PTS,
        agg_param_means,
        agg_stim_times,
        agg_LED_times,
        T_trunc,
        n_mc=N_MC_THEORY_WRT_LED,
        seed=12345,
    )
    print(
        f"Aggregate theory from pkl: area ON={np.trapz(agg_theory_on, THEORY_T_PTS):.4f}, "
        f"OFF={np.trapz(agg_theory_off, THEORY_T_PTS):.4f}"
    )
elif per_animal_theory_on and np.sum(weights_on) > 0 and np.sum(weights_off) > 0:
    agg_theory_on = np.average(per_animal_theory_on, axis=0, weights=weights_on)
    agg_theory_off = np.average(per_animal_theory_off, axis=0, weights=weights_off)
    print("Warning: aggregate pkl missing, using weighted average of per-animal theory.")
else:
    agg_theory_on = np.full_like(THEORY_T_PTS, np.nan, dtype=float)
    agg_theory_off = np.full_like(THEORY_T_PTS, np.nan, dtype=float)


# %%
# =============================================================================
# 1x6 plot (data + theory, no simulation)
# =============================================================================
bin_centers = (BINS_WRT_LED[1:] + BINS_WRT_LED[:-1]) / 2
n_animals = len(plot_data)
PLOT_XLIM = (-0.05, 0.2)

if n_animals == 0:
    raise RuntimeError("No animals available after filtering.")

n_panels = n_animals + 2  # per-animal + aggregate + average
fig, axes = plt.subplots(n_panels, 1, figsize=(6.5, 2.4 * n_panels), sharex=True, sharey=True)
if n_panels == 1:
    axes = [axes]

theory_label_added = False
for i, (ax, animal_data) in enumerate(zip(axes, plot_data)):
    ax.plot(
        bin_centers,
        animal_data["hist_on_scaled"],
        alpha=0.8,
        color="r",
        linestyle="-",
        label=f"LED ON ({animal_data['frac_data_on']:.3f})",
    )
    ax.plot(
        bin_centers,
        animal_data["hist_off_scaled"],
        alpha=0.8,
        color="b",
        linestyle="-",
        label=f"LED OFF ({animal_data['frac_data_off']:.3f})",
    )
    if animal_data["has_theory"]:
        theory_on_label = None
        theory_off_label = None
        if not theory_label_added:
            theory_on_label = "Theory ON"
            theory_off_label = "Theory OFF"
            theory_label_added = True
        ax.plot(
            THEORY_T_PTS,
            animal_data["theory_on_wrt_led"],
            lw=2,
            alpha=0.9,
            color="r",
            linestyle="--",
            label=theory_on_label,
        )
        ax.plot(
            THEORY_T_PTS,
            animal_data["theory_off_wrt_led"],
            lw=2,
            alpha=0.9,
            color="b",
            linestyle="--",
            label=theory_off_label,
        )

    ax.axvline(x=0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlim(PLOT_XLIM)
    ax.tick_params(axis="x", labelbottom=True)
    if i == n_animals - 1:
        ax.set_xlabel("RT - t_LED (s)")
    ax.set_title(str(animal_data["animal"]))

    if i == 0:
        ax.set_ylabel("Rate (area = fraction)")
        ax.legend(fontsize=9, loc="upper right")

# 7th panel: aggregate-all-data
ax_agg = axes[n_animals]
ax_agg.plot(bin_centers, agg_hist_on, lw=2, alpha=0.9, color="r", linestyle="-", label="LED ON")
ax_agg.plot(bin_centers, agg_hist_off, lw=2, alpha=0.9, color="b", linestyle="-", label="LED OFF")
if np.any(np.isfinite(agg_theory_on)):
    ax_agg.plot(THEORY_T_PTS, agg_theory_on, lw=2, alpha=0.9, color="r", linestyle="--", label="Theory ON")
    ax_agg.plot(THEORY_T_PTS, agg_theory_off, lw=2, alpha=0.9, color="b", linestyle="--", label="Theory OFF")
ax_agg.axvline(x=0, color="k", linestyle="--", alpha=0.5)
ax_agg.set_xlim(PLOT_XLIM)
ax_agg.tick_params(axis="x", labelbottom=True)
ax_agg.set_title("Aggregate")
# ax_agg.legend(fontsize=9, loc="upper right")

# 8th panel: average-of-animals
ax_avg = axes[n_animals + 1]
ax_avg.plot(bin_centers, avg_hist_on, lw=2, alpha=0.9, color="r", linestyle="-", label="LED ON")
ax_avg.plot(bin_centers, avg_hist_off, lw=2, alpha=0.9, color="b", linestyle="-", label="LED OFF")
if np.any(np.isfinite(avg_theory_on)):
    ax_avg.plot(THEORY_T_PTS, avg_theory_on, lw=2, alpha=0.9, color="r", linestyle="--", label="Theory ON")
    ax_avg.plot(THEORY_T_PTS, avg_theory_off, lw=2, alpha=0.9, color="b", linestyle="--", label="Theory OFF")
ax_avg.axvline(x=0, color="k", linestyle="--", alpha=0.5)
ax_avg.set_xlim(PLOT_XLIM)
ax_avg.tick_params(axis="x", labelbottom=True)
ax_avg.set_xlabel("RT - t_LED (s)")
ax_avg.set_title("Average")
# ax_avg.legend(fontsize=9, loc="upper right")

plt.suptitle("Data + theory RT wrt LED (aborts only, area-weighted)", y=1.03)
plt.tight_layout()

output_path = os.path.join(os.path.dirname(__file__), "data_theory_rt_wrt_led_all_animals_1x6.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved: {output_path}")
plt.show()

# %%
fig, ax = plt.subplots(figsize=(8, 5))

# Aggregate (solid)
ax.plot(bin_centers, agg_hist_on, lw=2, alpha=0.4, color="r", linestyle="-", label="Aggregate LED ON")
ax.plot(bin_centers, agg_hist_off, lw=2, alpha=0.4, color="b", linestyle="-", label="Aggregate LED OFF")

# Average (dashed)
ax.plot(bin_centers, avg_hist_on, lw=2, alpha=0.9, color="r", linestyle="--", label="Average LED ON")
ax.plot(bin_centers, avg_hist_off, lw=2, alpha=0.9, color="b", linestyle="--", label="Average LED OFF")

ax.axvline(x=0, color="k", linestyle=":", alpha=0.6)
ax.set_xlim(PLOT_XLIM)
ax.set_xlabel("RT - t_LED (s)")
ax.set_ylabel("Rate (area = fraction)")
ax.set_title("Aggregate (solid) vs Average (dashed)")
ax.legend(fontsize=9)

plt.tight_layout()
output_path_combined = os.path.join(os.path.dirname(__file__), "data_rt_wrt_led_aggregate_vs_average.png")
plt.savefig(output_path_combined, dpi=300, bbox_inches="tight")
print(f"Saved: {output_path_combined}")
plt.show()



# %%
# =============================================================================
# Per-animal t_LED distributions

# =============================================================================
# BINS_T_LED = np.arange(0.0, 2.0 + 0.02, 0.02)
# n_animals = len(plot_data)

# fig, axes = plt.subplots(n_animals, 1, figsize=(8, 2.2 * n_animals), sharex=True, sharey=True)
# if n_animals == 1:
#     axes = [axes]

# for i, (ax, animal_data) in enumerate(zip(axes, plot_data)):
#     t_led_on = animal_data["t_led_on"]
#     t_led_off = animal_data["t_led_off"]
#     t_led_all = animal_data["t_led_all"]

#     if len(t_led_on) > 0:
#         ax.hist(
#             t_led_on,
#             bins=BINS_T_LED,
#             density=True,
#             histtype="step",
#             color="r",
#             label=f"ON",
#         )
#     if len(t_led_off) > 0:
#         ax.hist(
#             t_led_off,
#             bins=BINS_T_LED,
#             density=True,
#             histtype="step",
#             color="b",
#             label=f"OFF",
#         )
#     if len(t_led_all) > 0:
#         ax.hist(
#             t_led_all,
#             bins=BINS_T_LED,
#             density=True,
#             histtype="step",
#             alpha=0.6,
#             ls="--",
#             color="k",
#             label=f"ALL",
#         )

#     ax.set_ylabel("Density")
#     ax.set_title(f"Animal {animal_data['animal']}")
#     if i == 0:
#         ax.legend(fontsize=8, loc="upper right")

# axes[-1].set_xlabel("t_LED (s)")

# plt.tight_layout()
# output_path_tled = os.path.join(
#     os.path.dirname(__file__),
#     "data_t_led_distributions_all_animals.png",
# )
# plt.savefig(output_path_tled, dpi=300, bbox_inches="tight")
# print(f"Saved: {output_path_tled}")
# plt.show()

# %%
