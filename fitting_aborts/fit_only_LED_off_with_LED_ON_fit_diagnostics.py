# %%
"""
Diagnostics for aggregate LED-OFF normalized-tied fit with proactive+lapse parameters loaded.

This script:
1) Rebuilds the same LED-OFF aggregate fitting dataset.
2) Loads fitted parameters from the aggregate results pickle.
3) Computes Monte Carlo-averaged theoretical RTwrtStim density on t_pts in [-2, 2] (1 ms).
4) Compares theory vs empirical data and marks the 150 ms post-stim censor threshold.
"""

# %%
from pathlib import Path
import os
import pickle
import sys

import matplotlib
SHOW_PLOT = True
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ANIMAL_WISE_DIR = REPO_ROOT / "fit_animal_by_animal"
if str(ANIMAL_WISE_DIR) not in sys.path:
    sys.path.insert(0, str(ANIMAL_WISE_DIR))

from proactive_plus_lapse_plus_reactive_uitls import (
    up_or_down_RTs_fit_proactive_lapse_only_fn,
)


# %%
############ Parameters (edit here) ############
batch_name = "LED7"
session_type = 7
training_level = 16
allowed_repeat_trials = [0, 2]
max_rtwrtstim_for_fit = 1.0
data_bin_size_s = 0.005

seed = 12345
N_mc = int(os.getenv("FIT_DIAG_N_MC", "1000"))
n_jobs = int(os.getenv("FIT_DIAG_N_JOBS", "30"))
show_plot = SHOW_PLOT

t_pts = np.arange(-2.0, 2.001, 0.001)
censor_rt_wrt_stim_s = 0.150

is_norm = True
is_time_vary = False
phi_params_obj = np.nan
K_max = 10

led_data_csv_path = REPO_ROOT / "out_LED.csv"
results_pkl_path = (
    SCRIPT_DIR
    / "norm_only_led_off_from_loaded_proactive"
    / f"results_norm_tied_batch_{batch_name}_aggregate_ledoff_1_proactive_loaded.pkl"
)

output_dir = SCRIPT_DIR / "norm_only_led_off_from_loaded_proactive" / "diagnostics"
output_dir.mkdir(parents=True, exist_ok=True)

plot_base = f"diag_norm_tied_batch_{batch_name}_aggregate_ledoff_raw_rtwrtstim"
plot_pdf_path = output_dir / f"{plot_base}.pdf"
plot_png_path = output_dir / f"{plot_base}.png"
payload_path = output_dir / f"{plot_base}.pkl"


# %%
############ Load fitted parameters ############
if not results_pkl_path.exists():
    raise FileNotFoundError(f"Could not find aggregate results pickle: {results_pkl_path}")

with open(results_pkl_path, "rb") as f:
    fit_payload = pickle.load(f)

if "vbmc_norm_tied_results" not in fit_payload:
    raise KeyError("Missing 'vbmc_norm_tied_results' in aggregate results pickle.")
if "loaded_proactive_params" not in fit_payload:
    raise KeyError("Missing 'loaded_proactive_params' in aggregate results pickle.")

vbmc_results = fit_payload["vbmc_norm_tied_results"]
loaded_pro = fit_payload["loaded_proactive_params"]

required_norm_keys = [
    "rate_lambda_samples",
    "T_0_samples",
    "theta_E_samples",
    "w_samples",
    "t_E_aff_samples",
    "del_go_samples",
    "rate_norm_l_samples",
]
for key in required_norm_keys:
    if key not in vbmc_results:
        raise KeyError(f"Missing key in vbmc_norm_tied_results: {key}")

required_pro_keys = [
    "V_A_base",
    "theta_A",
    "del_a_minus_del_LED",
    "del_m_plus_del_LED",
    "lapse_prob",
    "beta_lapse",
]
for key in required_pro_keys:
    if key not in loaded_pro:
        raise KeyError(f"Missing key in loaded_proactive_params: {key}")

rate_lambda = float(np.mean(vbmc_results["rate_lambda_samples"]))
T_0 = float(np.mean(vbmc_results["T_0_samples"]))
theta_E = float(np.mean(vbmc_results["theta_E_samples"]))
w = float(np.mean(vbmc_results["w_samples"]))
t_E_aff = float(np.mean(vbmc_results["t_E_aff_samples"]))
del_go = float(np.mean(vbmc_results["del_go_samples"]))
rate_norm_l = float(np.mean(vbmc_results["rate_norm_l_samples"]))
Z_E = (w - 0.5) * 2.0 * theta_E

V_A = float(loaded_pro["V_A_base"])
theta_A = float(loaded_pro["theta_A"])
del_a_minus_del_LED = float(loaded_pro["del_a_minus_del_LED"])
del_m_plus_del_LED = float(loaded_pro["del_m_plus_del_LED"])
lapse_prob = float(loaded_pro["lapse_prob"])
beta_lapse = float(loaded_pro["beta_lapse"])
t_A_aff = del_a_minus_del_LED + del_m_plus_del_LED

print("Loaded aggregate fit parameters:")
print(
    f"  rate_lambda={rate_lambda:.6f}, T_0={T_0:.6f}, theta_E={theta_E:.6f}, "
    f"w={w:.6f}, Z_E={Z_E:.6f}, t_E_aff={t_E_aff:.6f}, del_go={del_go:.6f}, "
    f"rate_norm_l={rate_norm_l:.6f}"
)
print(
    f"  V_A={V_A:.6f}, theta_A={theta_A:.6f}, del_a_minus_del_LED={del_a_minus_del_LED:.6f}, "
    f"del_m_plus_del_LED={del_m_plus_del_LED:.6f}, t_A_aff={t_A_aff:.6f}, "
    f"lapse_prob={lapse_prob:.6f}, beta_lapse={beta_lapse:.6f}"
)


# %%
############ Rebuild LED-OFF aggregate fit dataset ############
if not led_data_csv_path.exists():
    raise FileNotFoundError(f"Could not find data CSV: {led_data_csv_path}")

exp_df = pd.read_csv(led_data_csv_path)
exp_df["RTwrtStim"] = exp_df["timed_fix"] - exp_df["intended_fix"]
exp_df["t_LED"] = exp_df["intended_fix"] - exp_df["LED_onset_time"]
exp_df = exp_df.rename(columns={"timed_fix": "TotalFixTime"})

# Keep same preprocessing behavior as fit script.
exp_df = exp_df[exp_df["RTwrtStim"] < 1].copy()
exp_df = exp_df[~((exp_df["RTwrtStim"].isna()) & (exp_df["abort_event"] == 3))].copy()

mask_nan = exp_df["response_poke"].isna()
mask_success_1 = exp_df["success"] == 1
mask_success_neg1 = exp_df["success"] == -1
mask_ild_pos = exp_df["ILD"] > 0
mask_ild_neg = exp_df["ILD"] < 0
exp_df.loc[mask_nan & mask_success_1 & mask_ild_pos, "response_poke"] = 3
exp_df.loc[mask_nan & mask_success_1 & mask_ild_neg, "response_poke"] = 2
exp_df.loc[mask_nan & mask_success_neg1 & mask_ild_pos, "response_poke"] = 2
exp_df.loc[mask_nan & mask_success_neg1 & mask_ild_neg, "response_poke"] = 3

mask_led_off = (exp_df["LED_trial"] == 0) | (exp_df["LED_trial"].isna())
mask_repeat = exp_df["repeat_trial"].isin(allowed_repeat_trials) | exp_df["repeat_trial"].isna()
exp_df_led_off = exp_df[
    mask_led_off
    & mask_repeat
    & exp_df["session_type"].isin([session_type])
    & exp_df["training_level"].isin([training_level])
].copy()

df_valid_and_aborts = exp_df_led_off[
    (exp_df_led_off["success"].isin([1, -1])) | (exp_df_led_off["abort_event"] == 3)
].copy()
fit_df = df_valid_and_aborts[df_valid_and_aborts["RTwrtStim"] < max_rtwrtstim_for_fit].copy()

if len(fit_df) == 0:
    raise ValueError("No valid LED-OFF trials found after filtering.")

print("Rebuilt LED-OFF aggregate fit dataset:")
print(f"  Total LED-OFF filtered trials (valid+aborts): {len(df_valid_and_aborts)}")
print(f"  LED-OFF trials used for diagnostics (valid+aborts): {len(fit_df)}")


# %%
############ Monte Carlo theoretical RTwrtStim density (collapsed over choices) ############
rng = np.random.default_rng(seed)
sampled_positions = rng.integers(0, len(fit_df), size=N_mc)


def compute_one_sample_curve(row_pos):
    row = fit_df.iloc[int(row_pos)]
    t_stim = float(row["intended_fix"])
    t_led = float(row["t_LED"])
    abl = float(row["ABL"])
    ild = float(row["ILD"])

    t_abs = t_pts + t_stim
    curve = np.zeros_like(t_pts, dtype=np.float64)

    for j, t_abs_j in enumerate(t_abs):
        if t_abs_j <= 0:
            continue

        pdf_plus = up_or_down_RTs_fit_proactive_lapse_only_fn(
            t=t_abs_j,
            bound=1,
            V_A=V_A,
            theta_A=theta_A,
            t_A_aff=t_A_aff,
            t_stim=t_stim,
            ABL=abl,
            ILD=ild,
            rate_lambda=rate_lambda,
            T0=T_0,
            theta_E=theta_E,
            Z_E=Z_E,
            t_E_aff=t_E_aff,
            del_go=del_go,
            phi_params=phi_params_obj,
            rate_norm_l=rate_norm_l,
            is_norm=is_norm,
            is_time_vary=is_time_vary,
            K_max=K_max,
            lapse_prob=lapse_prob,
            beta_lapse=beta_lapse,
            lapse_choice_prob=0.5,
            eps=1e-50,
        )

        pdf_minus = up_or_down_RTs_fit_proactive_lapse_only_fn(
            t=t_abs_j,
            bound=-1,
            V_A=V_A,
            theta_A=theta_A,
            t_A_aff=t_A_aff,
            t_stim=t_stim,
            ABL=abl,
            ILD=ild,
            rate_lambda=rate_lambda,
            T0=T_0,
            theta_E=theta_E,
            Z_E=Z_E,
            t_E_aff=t_E_aff,
            del_go=del_go,
            phi_params=phi_params_obj,
            rate_norm_l=rate_norm_l,
            is_norm=is_norm,
            is_time_vary=is_time_vary,
            K_max=K_max,
            lapse_prob=lapse_prob,
            beta_lapse=beta_lapse,
            lapse_choice_prob=0.5,
            eps=1e-50,
        )

        val = pdf_plus + pdf_minus
        if np.isfinite(val) and val > 0:
            curve[j] = val

    return curve, t_stim, t_led, abl, ild


print(f"Computing Monte Carlo theory with N_mc={N_mc}, n_jobs={n_jobs} ...")
mc_results = Parallel(n_jobs=n_jobs)(
    delayed(compute_one_sample_curve)(int(pos)) for pos in sampled_positions
)

theory_matrix = np.stack([r[0] for r in mc_results], axis=0)
theory_density = np.mean(theory_matrix, axis=0)

sampled_t_stim = np.array([r[1] for r in mc_results], dtype=np.float64)
sampled_t_led = np.array([r[2] for r in mc_results], dtype=np.float64)
sampled_abl = np.array([r[3] for r in mc_results], dtype=np.float64)
sampled_ild = np.array([r[4] for r in mc_results], dtype=np.float64)


# %%
############ Empirical RTwrtStim density on same grid ############
data_rtwrtstim = fit_df["RTwrtStim"].to_numpy(dtype=np.float64)
dt = float(t_pts[1] - t_pts[0])
hist_edges = np.arange(t_pts[0], t_pts[-1] + data_bin_size_s, data_bin_size_s)
if hist_edges[-1] < t_pts[-1]:
    hist_edges = np.append(hist_edges, t_pts[-1])
data_density, _ = np.histogram(data_rtwrtstim, bins=hist_edges, density=True)
data_bin_centers = 0.5 * (hist_edges[:-1] + hist_edges[1:])
data_bin_widths = np.diff(hist_edges)

data_area = float(np.sum(data_density * data_bin_widths))
theory_area = float(np.trapz(theory_density, t_pts))
print(f"Density areas on t_pts grid: data={data_area:.6f}, theory={theory_area:.6f}")


# %%
############ Plot diagnostics ############
fig, ax = plt.subplots(figsize=(8, 5))
ax.step(
    data_bin_centers,
    data_density,
    where="mid",
    lw=2.0,
    color="tab:blue",
    label=f"Data, bin={1e3*data_bin_size_s:.0f} ms)",
)
ax.plot(t_pts, theory_density, lw=2.0, color="tab:orange", label=f"Theory MC average (N_mc={N_mc})")
ax.axvline(
    x=censor_rt_wrt_stim_s,
    color="crimson",
    linestyle="--",
    linewidth=1.8,
    label=f"Censor threshold = {censor_rt_wrt_stim_s*1e3:.0f} ms + t_stim",
)
ax.set_xlabel("RT - t_stim (s)")
ax.set_ylabel("Density")
ax.set_title(f"LED-OFF Aggregate RTD ({batch_name})")
# ax.set_xlim(float(t_pts[0]), float(t_pts[-1]))
ax.set_xlim(0.05, 0.4)
ax.legend()

fig.tight_layout()
fig.savefig(plot_pdf_path, bbox_inches="tight")
fig.savefig(plot_png_path, dpi=200, bbox_inches="tight")
if show_plot:
    plt.show()
plt.close(fig)

print(f"Saved diagnostics plot (PDF): {plot_pdf_path}")
print(f"Saved diagnostics plot (PNG): {plot_png_path}")


# %%
############ Save diagnostics payload ############
diagnostics_payload = {
    "config": {
        "batch_name": batch_name,
        "session_type": session_type,
        "training_level": training_level,
        "allowed_repeat_trials": allowed_repeat_trials,
        "max_rtwrtstim_for_fit": max_rtwrtstim_for_fit,
        "seed": seed,
        "N_mc": N_mc,
        "n_jobs": n_jobs,
        "show_plot": show_plot,
        "data_bin_size_s": data_bin_size_s,
        "t_pts_start_s": float(t_pts[0]),
        "t_pts_end_s": float(t_pts[-1]),
        "dt_s": dt,
        "grid_size": int(len(t_pts)),
        "censor_rt_wrt_stim_s": censor_rt_wrt_stim_s,
        "is_norm": is_norm,
        "is_time_vary": is_time_vary,
        "K_max": K_max,
        "results_pkl_path": str(results_pkl_path),
        "led_data_csv_path": str(led_data_csv_path),
        "compare_mode": "raw_no_censor_in_plot",
        "choice_mode": "collapsed_pdf_plus_plus_pdf_minus",
    },
    "parameter_snapshot": {
        "normalized_tied_means": {
            "rate_lambda": rate_lambda,
            "T_0": T_0,
            "theta_E": theta_E,
            "w": w,
            "Z_E": Z_E,
            "t_E_aff": t_E_aff,
            "del_go": del_go,
            "rate_norm_l": rate_norm_l,
        },
        "loaded_proactive_params": {
            "V_A_base": V_A,
            "theta_A": theta_A,
            "del_a_minus_del_LED": del_a_minus_del_LED,
            "del_m_plus_del_LED": del_m_plus_del_LED,
            "derived_t_A_aff": t_A_aff,
            "lapse_prob": lapse_prob,
            "beta_lapse": beta_lapse,
        },
    },
    "dataset_summary": {
        "n_ledoff_filtered_valid_plus_aborts": int(len(df_valid_and_aborts)),
        "n_valid_used_for_diagnostics": int(len(fit_df)),
        "rtwrtstim_min": float(np.min(data_rtwrtstim)),
        "rtwrtstim_max": float(np.max(data_rtwrtstim)),
        "rtwrtstim_mean": float(np.mean(data_rtwrtstim)),
    },
    "summary_stats": {
        "data_area_on_grid": data_area,
        "theory_area_on_grid": theory_area,
        "theory_min": float(np.min(theory_density)),
        "theory_max": float(np.max(theory_density)),
        "theory_has_nan": bool(np.isnan(theory_density).any()),
        "theory_has_inf": bool(np.isinf(theory_density).any()),
    },
    "t_pts": t_pts,
    "data_hist_edges": hist_edges,
    "data_hist_centers": data_bin_centers,
    "data_density": data_density,
    "theory_density_mc_avg": theory_density,
    "sampled_row_positions": sampled_positions,
    "sampled_t_stim": sampled_t_stim,
    "sampled_t_LED": sampled_t_led,
    "sampled_ABL": sampled_abl,
    "sampled_ILD": sampled_ild,
}

with open(payload_path, "wb") as f:
    pickle.dump(diagnostics_payload, f)

print(f"Saved diagnostics payload: {payload_path}")
print("Done.")
# %%
