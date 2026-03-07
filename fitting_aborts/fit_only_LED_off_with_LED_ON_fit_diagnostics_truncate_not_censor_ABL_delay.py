# %%
"""
Diagnostics for aggregate LED-OFF normalized-tied fit with proactive+lapse parameters loaded.

This truncation + ABL-delay variant:
1) Rebuilds the same LED-OFF aggregate diagnostics dataset.
2) Loads fitted parameters from the aggregate truncation-fit results pickle with ABL-specific t_E_aff.
3) Computes Monte Carlo-averaged theoretical RTwrtStim density on t_pts in [-2, 2] (1 ms).
4) Compares the raw untruncated theory vs empirical data and marks the fit-aligned post-stim
   truncation threshold.
5) Also compares theory vs data after truncating both to [0, truncate_rt_wrt_stim_s] and
   area-normalizing them.
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
data_bin_size_s_full = 0.005
data_bin_size_s_truncated = 5e-3

seed = 12345
N_mc = int(os.getenv("FIT_DIAG_N_MC", "1000"))
N_mc_per_abl_panel = int(os.getenv("FIT_DIAG_N_MC_PER_ABL", str(N_mc)))
n_jobs = int(os.getenv("FIT_DIAG_N_JOBS", "30"))
show_plot = SHOW_PLOT

t_pts = np.arange(-2.0, 2.001, 0.001)
truncate_rt_wrt_stim_s_override = None  # None -> load from fit results pickle
supported_abl_values = (20, 40, 60)
abl_colors = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}

is_norm = True
is_time_vary = False
phi_params_obj = np.nan
K_max = 10

if N_mc_per_abl_panel <= 0:
    raise ValueError("N_mc_per_abl_panel must be positive.")

led_data_csv_path = REPO_ROOT / "out_LED.csv"
results_pkl_path = (
    SCRIPT_DIR
    / "norm_only_led_off_from_loaded_proactive_truncate_NOT_censor_ABL_delay"
    / f"results_norm_tied_batch_{batch_name}_aggregate_ledoff_1_proactive_loaded_truncate_NOT_censor_ABL_delay.pkl"
)

output_dir = (
    SCRIPT_DIR
    / "norm_only_led_off_from_loaded_proactive_truncate_NOT_censor_ABL_delay"
    / "diagnostics"
)
output_dir.mkdir(parents=True, exist_ok=True)

plot_base = (
    f"diag_norm_tied_batch_{batch_name}_aggregate_ledoff_raw_rtwrtstim_"
    "truncate_not_censor_ABL_delay"
)
plot_pdf_path = output_dir / f"{plot_base}.pdf"
plot_png_path = output_dir / f"{plot_base}.png"

payload_path = output_dir / f"{plot_base}.pkl"


# %%
############ Helpers ############
def get_t_E_aff_from_abl(abl, t_E_aff_20, t_E_aff_40, t_E_aff_60):
    abl_value = float(abl)
    if np.isclose(abl_value, 20.0):
        return t_E_aff_20
    if np.isclose(abl_value, 40.0):
        return t_E_aff_40
    if np.isclose(abl_value, 60.0):
        return t_E_aff_60
    raise ValueError(
        f"Unsupported ABL value {abl_value}. Expected one of {supported_abl_values}."
    )


def validate_supported_abl_values(df, df_name):
    observed = np.sort(df["ABL"].dropna().astype(float).unique())
    if len(observed) == 0:
        raise ValueError(f"No ABL values found in {df_name}.")

    unexpected = [
        float(abl)
        for abl in observed
        if not any(np.isclose(float(abl), float(supported)) for supported in supported_abl_values)
    ]
    if unexpected:
        raise ValueError(
            f"Unexpected ABL values in {df_name}: {unexpected}. "
            f"Supported values are {supported_abl_values}."
        )
    return observed


def format_abl_counts(df):
    counts = (
        df["ABL"]
        .astype(float)
        .round()
        .astype(int)
        .value_counts()
        .sort_index()
        .to_dict()
    )
    return {int(k): int(v) for k, v in counts.items()}


def format_truncation_labels(truncate_rt_wrt_stim_s):
    truncate_ms = int(round(float(truncate_rt_wrt_stim_s) * 1e3))
    return truncate_ms, f"{truncate_ms} ms", f"{truncate_ms}ms"


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

fit_config = fit_payload.get("fit_config", {})
vbmc_results = fit_payload["vbmc_norm_tied_results"]
loaded_pro = fit_payload["loaded_proactive_params"]

truncate_rt_wrt_stim_s_from_fit = fit_config.get("truncate_rt_wrt_stim_s")
if truncate_rt_wrt_stim_s_override is None:
    if truncate_rt_wrt_stim_s_from_fit is None:
        raise KeyError(
            "Missing 'fit_config.truncate_rt_wrt_stim_s' in aggregate results pickle. "
            "Set truncate_rt_wrt_stim_s_override near the top to override manually."
        )
    truncate_rt_wrt_stim_s = float(truncate_rt_wrt_stim_s_from_fit)
else:
    truncate_rt_wrt_stim_s = float(truncate_rt_wrt_stim_s_override)

if truncate_rt_wrt_stim_s <= 0:
    raise ValueError("truncate_rt_wrt_stim_s must be positive.")
if truncate_rt_wrt_stim_s > max_rtwrtstim_for_fit:
    raise ValueError("truncate_rt_wrt_stim_s cannot exceed max_rtwrtstim_for_fit.")

truncate_rt_wrt_stim_ms, truncate_label_ms, truncate_label_tag = format_truncation_labels(
    truncate_rt_wrt_stim_s
)
truncated_plot_base = (
    f"diag_norm_tied_batch_{batch_name}_aggregate_ledoff_truncated_{truncate_label_tag}_rtwrtstim_"
    "truncate_not_censor_ABL_delay"
)
truncated_plot_pdf_path = output_dir / f"{truncated_plot_base}.pdf"
truncated_plot_png_path = output_dir / f"{truncated_plot_base}.png"

abl_split_plot_base = (
    f"diag_norm_tied_batch_{batch_name}_aggregate_ledoff_truncated_{truncate_label_tag}_rtwrtstim_"
    "by_ABL_truncate_not_censor_ABL_delay"
)
abl_split_plot_pdf_path = output_dir / f"{abl_split_plot_base}.pdf"
abl_split_plot_png_path = output_dir / f"{abl_split_plot_base}.png"

if truncate_rt_wrt_stim_s_from_fit is None:
    print(f"Using diagnostics truncation threshold: {truncate_rt_wrt_stim_s:.3f} s ({truncate_label_ms})")
else:
    print(
        "Using diagnostics truncation threshold: "
        f"{truncate_rt_wrt_stim_s:.3f} s ({truncate_label_ms}); "
        f"fit results saved {float(truncate_rt_wrt_stim_s_from_fit):.3f} s"
    )

required_norm_keys = [
    "rate_lambda_samples",
    "T_0_samples",
    "theta_E_samples",
    "w_samples",
    "t_E_aff_20_samples",
    "t_E_aff_40_samples",
    "t_E_aff_60_samples",
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
t_E_aff_20 = float(np.mean(vbmc_results["t_E_aff_20_samples"]))
t_E_aff_40 = float(np.mean(vbmc_results["t_E_aff_40_samples"]))
t_E_aff_60 = float(np.mean(vbmc_results["t_E_aff_60_samples"]))
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

print("Loaded aggregate truncation-fit ABL-delay parameters:")
print(
    f"  rate_lambda={rate_lambda:.6f}, T_0={T_0:.6f}, theta_E={theta_E:.6f}, "
    f"w={w:.6f}, Z_E={Z_E:.6f}, t_E_aff_20={t_E_aff_20:.6f}, "
    f"t_E_aff_40={t_E_aff_40:.6f}, t_E_aff_60={t_E_aff_60:.6f}, "
    f"del_go={del_go:.6f}, rate_norm_l={rate_norm_l:.6f}"
)
print(
    f"  V_A={V_A:.6f}, theta_A={theta_A:.6f}, del_a_minus_del_LED={del_a_minus_del_LED:.6f}, "
    f"del_m_plus_del_LED={del_m_plus_del_LED:.6f}, t_A_aff={t_A_aff:.6f}, "
    f"lapse_prob={lapse_prob:.6f}, beta_lapse={beta_lapse:.6f}"
)


# %%
############ Rebuild LED-OFF aggregate diagnostics dataset ############
if not led_data_csv_path.exists():
    raise FileNotFoundError(f"Could not find data CSV: {led_data_csv_path}")

exp_df = pd.read_csv(led_data_csv_path)
exp_df["RTwrtStim"] = exp_df["timed_fix"] - exp_df["intended_fix"]
exp_df["t_LED"] = exp_df["intended_fix"] - exp_df["LED_onset_time"]
exp_df = exp_df.rename(columns={"timed_fix": "TotalFixTime"})

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
    (exp_df_led_off["success"].isin([1, -1])) | (exp_df_led_off["abort_event"].isin([3, 4]))
].copy()
fit_df = df_valid_and_aborts[df_valid_and_aborts["RTwrtStim"] < max_rtwrtstim_for_fit].copy()

if len(fit_df) == 0:
    raise ValueError("No LED-OFF trials found after filtering.")

observed_abl_values = validate_supported_abl_values(
    fit_df, "LED-OFF aggregate diagnostics dataset"
)
abl_counts = format_abl_counts(fit_df)

print("Rebuilt LED-OFF aggregate diagnostics dataset:")
print(f"  Total LED-OFF filtered trials (valid+aborts): {len(df_valid_and_aborts)}")
print(f"  LED-OFF trials used for diagnostics (valid+aborts): {len(fit_df)}")
print(f"  Supported ABL values in diagnostics dataset: {observed_abl_values.tolist()}")
print(f"  Diagnostics trial counts by ABL: {abl_counts}")


# %%
############ Monte Carlo theoretical RTwrtStim density (collapsed over choices) ############
rng = np.random.default_rng(seed)
sampled_positions = rng.integers(0, len(fit_df), size=N_mc)
sampled_rows = fit_df.iloc[sampled_positions].copy()


def build_theory_curve_for_trial(t_stim, abl, ild):
    t_E_aff = get_t_E_aff_from_abl(
        abl,
        t_E_aff_20=t_E_aff_20,
        t_E_aff_40=t_E_aff_40,
        t_E_aff_60=t_E_aff_60,
    )
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

    return curve


def compute_one_sample_curve_from_row(row):
    t_stim = float(row["intended_fix"])
    t_led = float(row["t_LED"])
    abl = float(row["ABL"])
    ild = float(row["ILD"])
    curve = build_theory_curve_for_trial(t_stim=t_stim, abl=abl, ild=ild)
    return curve, t_stim, t_led, abl, ild


print(f"Computing Monte Carlo theory with N_mc={N_mc}, n_jobs={n_jobs} ...")
mc_results = Parallel(n_jobs=n_jobs)(
    delayed(compute_one_sample_curve_from_row)(row) for _, row in sampled_rows.iterrows()
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
hist_edges = np.arange(t_pts[0], t_pts[-1] + data_bin_size_s_full, data_bin_size_s_full)
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
    label=f"Data, bin={1e3*data_bin_size_s_full:.0f} ms",
)
ax.plot(
    t_pts,
    theory_density,
    lw=2.0,
    color="tab:orange",
    label=f"Theory MC average (N_mc={N_mc})",
)
ax.axvline(
    x=truncate_rt_wrt_stim_s,
    color="crimson",
    linestyle="--",
    linewidth=1.8,
    label=f"Trunc threshold = {truncate_rt_wrt_stim_s*1e3:.0f} ms + t_stim",
)
ax.set_xlabel("RT - t_stim (s)")
ax.set_ylabel("Density")
ax.set_title(f"LED-OFF Aggregate RTD from Trunc-Fit ABL-delay ({batch_name})")
ax.set_xlim(-0.1, 0.4)
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
############ Plot truncated-and-renormalized RTwrtStim diagnostics ############
mask_truncated = (t_pts >= 0.0) & (t_pts <= truncate_rt_wrt_stim_s)
t_pts_truncated = t_pts[mask_truncated]

theory_density_truncated_raw = theory_density[mask_truncated].copy()
theory_area_truncated_raw = float(np.trapz(theory_density_truncated_raw, t_pts_truncated))
if theory_area_truncated_raw <= 0:
    raise ValueError("Truncated theory area is non-positive; cannot normalize.")
theory_density_truncated_norm = theory_density_truncated_raw / theory_area_truncated_raw

data_rtwrtstim_truncated = data_rtwrtstim[
    (data_rtwrtstim >= 0.0) & (data_rtwrtstim <= truncate_rt_wrt_stim_s)
]
if len(data_rtwrtstim_truncated) == 0:
    raise ValueError(
        f"No data points remain after truncating RTwrtStim to [0, {truncate_label_ms}]."
    )

hist_edges_truncated = np.arange(
    0.0,
    truncate_rt_wrt_stim_s + data_bin_size_s_truncated,
    data_bin_size_s_truncated,
)
if hist_edges_truncated[-1] < truncate_rt_wrt_stim_s:
    hist_edges_truncated = np.append(hist_edges_truncated, truncate_rt_wrt_stim_s)
data_density_truncated, _ = np.histogram(
    data_rtwrtstim_truncated,
    bins=hist_edges_truncated,
    density=True,
)
data_bin_centers_truncated = 0.5 * (
    hist_edges_truncated[:-1] + hist_edges_truncated[1:]
)
data_bin_widths_truncated = np.diff(hist_edges_truncated)

data_area_truncated = float(np.sum(data_density_truncated * data_bin_widths_truncated))
if data_area_truncated > 0:
    data_density_truncated /= data_area_truncated
theory_area_truncated_norm = float(np.trapz(theory_density_truncated_norm, t_pts_truncated))
data_area_truncated_norm = float(np.sum(data_density_truncated * data_bin_widths_truncated))

print(
    "Truncated density areas: "
    f"data_raw={data_area_truncated:.6f}, data_norm={data_area_truncated_norm:.6f}, "
    f"theory_raw={theory_area_truncated_raw:.6f}, theory_norm={theory_area_truncated_norm:.6f}"
)

fig_trunc, ax_trunc = plt.subplots(figsize=(8, 5))
ax_trunc.step(
    data_bin_centers_truncated,
    data_density_truncated,
    where="mid",
    lw=2.0,
    color="tab:blue",
    label=f"Data [0, {truncate_rt_wrt_stim_ms} ms]",
)
ax_trunc.plot(
    t_pts_truncated,
    theory_density_truncated_norm,
    lw=2.0,
    color="tab:orange",
    label="Theory trunc + area norm",
)
ax_trunc.axvline(
    x=truncate_rt_wrt_stim_s,
    color="crimson",
    linestyle="--",
    linewidth=1.8,
    label=f"Trunc threshold = {truncate_rt_wrt_stim_ms} ms + t_stim",
)
ax_trunc.set_xlabel("RT - t_stim (s)")
ax_trunc.set_ylabel("Density")
ax_trunc.set_title(
    f"LED-OFF Aggregate RTD truncated {truncate_label_ms} ABL-delay ({batch_name})"
)
ax_trunc.set_xlim(0.0, truncate_rt_wrt_stim_s)
ax_trunc.legend()

fig_trunc.tight_layout()
fig_trunc.savefig(truncated_plot_pdf_path, bbox_inches="tight")
fig_trunc.savefig(truncated_plot_png_path, dpi=200, bbox_inches="tight")
if show_plot:
    plt.show()
plt.close(fig_trunc)

print(f"Saved truncated diagnostics plot (PDF): {truncated_plot_pdf_path}")
print(f"Saved truncated diagnostics plot (PNG): {truncated_plot_png_path}")


# %%
############ Plot truncated-and-renormalized RTwrtStim diagnostics by ABL ############
print(
    f"Computing per-ABL truncated theory with N_mc_per_abl_panel={N_mc_per_abl_panel}, "
    f"n_jobs={n_jobs} ..."
)

abl_panel_payload = {}
combined_ax_max = 0.0
abl_values_float = fit_df["ABL"].astype(float).to_numpy()

for abl in supported_abl_values:
    df_abl = fit_df[np.isclose(abl_values_float, float(abl))].copy()
    if len(df_abl) == 0:
        raise ValueError(f"No diagnostics rows found for ABL={abl}.")

    sampled_positions_abl = rng.integers(0, len(df_abl), size=N_mc_per_abl_panel)
    sampled_rows_abl = df_abl.iloc[sampled_positions_abl].copy()
    mc_results_abl = Parallel(n_jobs=n_jobs)(
        delayed(compute_one_sample_curve_from_row)(row)
        for _, row in sampled_rows_abl.iterrows()
    )
    theory_matrix_abl = np.stack([r[0] for r in mc_results_abl], axis=0)
    theory_density_abl = np.mean(theory_matrix_abl, axis=0)
    theory_density_abl_truncated_raw = theory_density_abl[mask_truncated].copy()
    theory_area_abl_raw = float(np.trapz(theory_density_abl_truncated_raw, t_pts_truncated))
    if theory_area_abl_raw <= 0:
        raise ValueError(f"Truncated theory area is non-positive for ABL={abl}.")
    theory_density_abl_truncated_norm = theory_density_abl_truncated_raw / theory_area_abl_raw

    data_rtwrtstim_abl = df_abl["RTwrtStim"].to_numpy(dtype=np.float64)
    data_rtwrtstim_abl_truncated = data_rtwrtstim_abl[
        (data_rtwrtstim_abl >= 0.0) & (data_rtwrtstim_abl <= truncate_rt_wrt_stim_s)
    ]
    if len(data_rtwrtstim_abl_truncated) == 0:
        raise ValueError(f"No truncated data points remain for ABL={abl}.")

    data_density_abl_truncated, _ = np.histogram(
        data_rtwrtstim_abl_truncated,
        bins=hist_edges_truncated,
        density=True,
    )
    data_area_abl_raw = float(
        np.sum(data_density_abl_truncated * data_bin_widths_truncated)
    )
    if data_area_abl_raw > 0:
        data_density_abl_truncated /= data_area_abl_raw
    data_area_abl_norm = float(
        np.sum(data_density_abl_truncated * data_bin_widths_truncated)
    )
    theory_area_abl_norm = float(
        np.trapz(theory_density_abl_truncated_norm, t_pts_truncated)
    )

    combined_ax_max = max(
        combined_ax_max,
        float(np.max(data_density_abl_truncated)),
        float(np.max(theory_density_abl_truncated_norm)),
    )

    abl_panel_payload[int(abl)] = {
        "n_rows": int(len(df_abl)),
        "n_truncated_points": int(len(data_rtwrtstim_abl_truncated)),
        "sampled_positions": sampled_positions_abl,
        "data_density_truncated": data_density_abl_truncated,
        "theory_density_truncated_raw": theory_density_abl_truncated_raw,
        "theory_density_truncated_norm": theory_density_abl_truncated_norm,
        "data_area_truncated_raw": data_area_abl_raw,
        "data_area_truncated_norm": data_area_abl_norm,
        "theory_area_truncated_raw": theory_area_abl_raw,
        "theory_area_truncated_norm": theory_area_abl_norm,
    }

    print(
        f"ABL={abl}: n_rows={len(df_abl)}, n_truncated={len(data_rtwrtstim_abl_truncated)}, "
        f"data_norm={data_area_abl_norm:.6f}, theory_norm={theory_area_abl_norm:.6f}"
    )

fig_abl, axes_abl = plt.subplots(1, 4, figsize=(20, 4.8), sharex=True, sharey=True)

for ax, abl in zip(axes_abl[:3], supported_abl_values):
    abl_int = int(abl)
    color = abl_colors[abl_int]
    payload_abl = abl_panel_payload[abl_int]

    ax.step(
        data_bin_centers_truncated,
        payload_abl["data_density_truncated"],
        where="mid",
        lw=2.0,
        color=color,
        alpha=0.75,
        label="Data",
    )
    ax.plot(
        t_pts_truncated,
        payload_abl["theory_density_truncated_norm"],
        lw=2.4,
        color=color,
        label="Theory",
    )
    ax.axvline(
        x=truncate_rt_wrt_stim_s,
        color="crimson",
        linestyle="--",
        linewidth=1.4,
        label=truncate_label_ms,
    )
    ax.set_title(f"ABL {abl_int}")
    ax.set_xlim(0.0, truncate_rt_wrt_stim_s)
    ax.set_xlabel("RT - t_stim (s)")
    if ax is axes_abl[0]:
        ax.set_ylabel("Density")
    ax.legend(fontsize=9)

ax_combined = axes_abl[3]
for abl in supported_abl_values:
    abl_int = int(abl)
    color = abl_colors[abl_int]
    payload_abl = abl_panel_payload[abl_int]
    ax_combined.step(
        data_bin_centers_truncated,
        payload_abl["data_density_truncated"],
        where="mid",
        lw=1.8,
        color=color,
        alpha=0.55,
        label=f"ABL {abl_int} data",
    )
    ax_combined.plot(
        t_pts_truncated,
        payload_abl["theory_density_truncated_norm"],
        lw=2.2,
        color=color,
        label=f"ABL {abl_int} theory",
    )

ax_combined.axvline(
    x=truncate_rt_wrt_stim_s,
    color="crimson",
    linestyle="--",
    linewidth=1.4,
    label=truncate_label_ms,
)
ax_combined.set_title("All ABLs")
ax_combined.set_xlim(0.0, truncate_rt_wrt_stim_s)
ax_combined.set_xlabel("RT - t_stim (s)")
ax_combined.legend(fontsize=8, ncol=2)

if combined_ax_max > 0:
    axes_abl[0].set_ylim(0.0, combined_ax_max * 1.1)

fig_abl.suptitle(
    f"LED-OFF Aggregate RTD truncated {truncate_label_ms} by ABL ({batch_name})",
    y=1.02,
)
fig_abl.tight_layout(rect=[0, 0, 1, 0.97])
fig_abl.savefig(abl_split_plot_pdf_path, bbox_inches="tight")
fig_abl.savefig(abl_split_plot_png_path, dpi=200, bbox_inches="tight")
if show_plot:
    plt.show()
plt.close(fig_abl)

print(f"Saved ABL-split truncated diagnostics plot (PDF): {abl_split_plot_pdf_path}")
print(f"Saved ABL-split truncated diagnostics plot (PNG): {abl_split_plot_png_path}")


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
        "N_mc_per_abl_panel": N_mc_per_abl_panel,
        "n_jobs": n_jobs,
        "show_plot": show_plot,
        "data_bin_size_s_full": data_bin_size_s_full,
        "data_bin_size_s_truncated": data_bin_size_s_truncated,
        "t_pts_start_s": float(t_pts[0]),
        "t_pts_end_s": float(t_pts[-1]),
        "dt_s": dt,
        "grid_size": int(len(t_pts)),
        "truncate_rt_wrt_stim_s": truncate_rt_wrt_stim_s,
        "is_norm": is_norm,
        "is_time_vary": is_time_vary,
        "K_max": K_max,
        "results_pkl_path": str(results_pkl_path),
        "led_data_csv_path": str(led_data_csv_path),
        "compare_mode": "raw_full_rtd_from_trunc_fit",
        "compare_mode_truncated": f"clipped_to_{truncate_label_tag}_and_area_normalized",
        "choice_mode": "collapsed_pdf_plus_plus_pdf_minus",
        "theory_curve_is_untruncated": True,
        "theory_curve_truncated_is_area_normalized": True,
        "theory_curve_truncated_by_abl_is_area_normalized": True,
        "abl_specific_delay_rule": "ABL=20 -> t_E_aff_20, ABL=40 -> t_E_aff_40, ABL=60 -> t_E_aff_60",
        "supported_ABL_values": list(supported_abl_values),
        "truncated_plot_pdf_path": str(truncated_plot_pdf_path),
        "truncated_plot_png_path": str(truncated_plot_png_path),
        "abl_split_plot_pdf_path": str(abl_split_plot_pdf_path),
        "abl_split_plot_png_path": str(abl_split_plot_png_path),
    },
    "parameter_snapshot": {
        "normalized_tied_means": {
            "rate_lambda": rate_lambda,
            "T_0": T_0,
            "theta_E": theta_E,
            "w": w,
            "Z_E": Z_E,
            "t_E_aff_20": t_E_aff_20,
            "t_E_aff_40": t_E_aff_40,
            "t_E_aff_60": t_E_aff_60,
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
        "n_valid_plus_aborts_used_for_diagnostics": int(len(fit_df)),
        "n_points_in_truncated_window": int(len(data_rtwrtstim_truncated)),
        "truncated_window_label": f"0_to_{truncate_label_tag}",
        "abl_counts_used_for_diagnostics": abl_counts,
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
        "data_area_truncated_raw": data_area_truncated,
        "data_area_truncated_norm": data_area_truncated_norm,
        "theory_area_truncated_raw": theory_area_truncated_raw,
        "theory_area_truncated_norm": theory_area_truncated_norm,
        "abl_split_summary": {
            int(abl): {
                "data_area_truncated_raw": abl_panel_payload[int(abl)]["data_area_truncated_raw"],
                "data_area_truncated_norm": abl_panel_payload[int(abl)]["data_area_truncated_norm"],
                "theory_area_truncated_raw": abl_panel_payload[int(abl)]["theory_area_truncated_raw"],
                "theory_area_truncated_norm": abl_panel_payload[int(abl)]["theory_area_truncated_norm"],
            }
            for abl in supported_abl_values
        },
    },
    "t_pts": t_pts,
    "data_hist_edges": hist_edges,
    "data_hist_centers": data_bin_centers,
    "data_density": data_density,
    "theory_density_mc_avg": theory_density,
    "t_pts_truncated": t_pts_truncated,
    "data_hist_edges_truncated": hist_edges_truncated,
    "data_hist_centers_truncated": data_bin_centers_truncated,
    "data_density_truncated": data_density_truncated,
    "theory_density_truncated_raw": theory_density_truncated_raw,
    "theory_density_truncated_norm": theory_density_truncated_norm,
    "abl_split_truncated": {
        int(abl): {
            "n_rows": abl_panel_payload[int(abl)]["n_rows"],
            "n_truncated_points": abl_panel_payload[int(abl)]["n_truncated_points"],
            "sampled_positions": abl_panel_payload[int(abl)]["sampled_positions"],
            "data_density_truncated": abl_panel_payload[int(abl)]["data_density_truncated"],
            "theory_density_truncated_raw": abl_panel_payload[int(abl)]["theory_density_truncated_raw"],
            "theory_density_truncated_norm": abl_panel_payload[int(abl)]["theory_density_truncated_norm"],
        }
        for abl in supported_abl_values
    },
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
