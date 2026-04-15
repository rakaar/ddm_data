# %%
"""
Diagnostics for aggregate LED-OFF normalized-tied with choice fit with proactive+lapse
parameters loaded.

This truncation + linear-delay + with choice variant:
1) Rebuilds the same LED-OFF aggregate diagnostics dataset used by the old FAST diagnostics flow.
2) Loads fitted parameters from the aggregate linear-delay with choice results pickle.
3) Computes Monte Carlo-averaged theoretical RTwrtStim density on t_pts in [-2, 2] (1 ms).
4) Compares the raw untruncated theory vs empirical data and marks the fit-aligned post-stim
   truncation threshold.
5) Also compares theory vs data after truncating both to [0, truncate_rt_wrt_stim_s].
6) Adds ABL-only and ABL x abs_ILD truncated diagnostics panels.
7) Adds a derived delay-surface plot and a quantile-vs-|ILD| comparison plot.
"""

# %%
from pathlib import Path
import os
import pickle
import sys

SHOW_PLOT = True
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ANIMAL_WISE_DIR = REPO_ROOT / "fit_animal_by_animal"
if str(ANIMAL_WISE_DIR) not in sys.path:
    sys.path.insert(0, str(ANIMAL_WISE_DIR))

from linear_delay_no_choice_diagnostics_utils import (
    build_condition_counts,
    build_balanced_signed_ild_sample_rows,
    build_truncated_density_payload,
    compute_average_curve_from_rows,
    compute_empirical_truncated_quantiles,
    compute_mc_average_curve,
    compute_quantiles_from_truncated_density,
    format_counts,
    format_truncation_labels,
    get_t_E_aff_from_abl_abs_ild,
    validate_supported_values,
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
N_mc_per_condition = int(os.getenv("FIT_DIAG_N_MC_PER_COND", "300"))
n_jobs = int(os.getenv("FIT_DIAG_N_JOBS", "30"))
show_plot = SHOW_PLOT

t_pts = np.arange(-2.0, 2.001, 0.001)
supported_abl_values = (20, 40, 60)
default_supported_abs_ild_values = (1, 2, 4, 8, 16)
abl_colors = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}

publication_figsize = (6.2, 4.0)
publication_plot_dpi = 300
publication_data_linewidth = 1.2
publication_theory_linewidth = 2.5
publication_data_alpha = 1.0
publication_xlabel = r"RT - $t_{stim}$ (ms)"
publication_ylabel = "Density"

condition_panel_width = 3.8
condition_panel_height = 3.0
combined_abs_ild_figsize = (17.5, 3.6)
combined_abl_figsize = (12.8, 3.6)
combined_condition_xlim_s = 0.115
combined_condition_plot_dpi = 300
delay_surface_figsize = (7.0, 3.8)
delay_surface_dpi = 300
quantile_levels = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float64)
model_abs_ild_grid = np.arange(0.0, 16.0 + 0.5, 0.5)
N_mc_per_quantile_grid = int(os.getenv("FIT_DIAG_N_MC_PER_QUANT", str(N_mc_per_condition)))
min_trials_for_data_quantiles = 5
quantile_figsize = (13.5, 4.2)
quantile_plot_dpi = 300

normalize_per_abl = False
normalize_per_condition = False

is_norm = True
is_time_vary = False
K_max = 10

fit_truncate_rt_wrt_stim_s = 1.0
truncate_rt_wrt_stim_s_override = None  # None -> use fit truncation saved in the results pickle

if N_mc <= 0:
    raise ValueError("N_mc must be positive.")
if N_mc_per_abl_panel <= 0:
    raise ValueError("N_mc_per_abl_panel must be positive.")
if N_mc_per_condition <= 0:
    raise ValueError("N_mc_per_condition must be positive.")
if N_mc_per_quantile_grid <= 0:
    raise ValueError("N_mc_per_quantile_grid must be positive.")
if fit_truncate_rt_wrt_stim_s <= 0:
    raise ValueError("fit_truncate_rt_wrt_stim_s must be positive.")
if min_trials_for_data_quantiles <= 0:
    raise ValueError("min_trials_for_data_quantiles must be positive.")
if combined_condition_xlim_s <= 0:
    raise ValueError("combined_condition_xlim_s must be positive.")

led_data_csv_path = REPO_ROOT / "out_LED.csv"
fit_results_dir = (
    SCRIPT_DIR / "norm_only_led_off_from_loaded_proactive_truncate_NOT_censor_linear_delay_with_choice"
)
output_dir = fit_results_dir / "diagnostics"
output_dir.mkdir(parents=True, exist_ok=True)


# %%
############ Load fitted parameters ############
fit_run_tag_requested = f"trunc{int(round(float(fit_truncate_rt_wrt_stim_s) * 1e3))}ms"
results_pkl_path = (
    fit_results_dir
    / (
        "results_norm_tied_"
        f"batch_{batch_name}_aggregate_ledoff_1_"
        "proactive_loaded_truncate_NOT_censor_linear_delay_with_choice_"
        f"{fit_run_tag_requested}.pkl"
    )
)

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

fit_run_tag = fit_config.get("run_tag")
if fit_run_tag is None:
    raise KeyError("Missing 'fit_config.run_tag' in aggregate results pickle.")
if fit_run_tag != fit_run_tag_requested:
    raise ValueError(
        "Diagnostics run-tag mismatch. "
        f"Requested {fit_run_tag_requested}, but loaded fit config resolves to {fit_run_tag}."
    )

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

plot_abs_ild_values = tuple(
    int(round(float(value)))
    for value in fit_config.get("observed_abs_ILD_values", default_supported_abs_ild_values)
)
if len(plot_abs_ild_values) == 0:
    plot_abs_ild_values = default_supported_abs_ild_values

truncate_rt_wrt_stim_ms, truncate_label_ms, truncate_label_tag = format_truncation_labels(
    truncate_rt_wrt_stim_s
)
raw_plot_base = (
    f"diag_norm_tied_batch_{batch_name}_aggregate_ledoff_raw_rtwrtstim_"
    f"truncate_not_censor_linear_delay_with_choice_{fit_run_tag}"
)
raw_plot_pdf_path = output_dir / f"{raw_plot_base}.pdf"
raw_plot_png_path = output_dir / f"{raw_plot_base}.png"

truncated_plot_base = (
    f"diag_norm_tied_batch_{batch_name}_aggregate_ledoff_truncated_{truncate_label_tag}_rtwrtstim_"
    f"truncate_not_censor_linear_delay_with_choice_{fit_run_tag}"
)
truncated_plot_pdf_path = output_dir / f"{truncated_plot_base}.pdf"
truncated_plot_png_path = output_dir / f"{truncated_plot_base}.png"

abl_split_plot_base = (
    f"diag_norm_tied_batch_{batch_name}_aggregate_ledoff_truncated_{truncate_label_tag}_rtwrtstim_"
    f"by_ABL_truncate_not_censor_linear_delay_with_choice_{fit_run_tag}"
)
abl_split_plot_pdf_path = output_dir / f"{abl_split_plot_base}.pdf"
abl_split_plot_png_path = output_dir / f"{abl_split_plot_base}.png"

condition_split_plot_base = (
    f"diag_norm_tied_batch_{batch_name}_aggregate_ledoff_truncated_{truncate_label_tag}_rtwrtstim_"
    f"by_ABL_abs_ILD_truncate_not_censor_linear_delay_with_choice_{fit_run_tag}"
)
condition_split_plot_pdf_path = output_dir / f"{condition_split_plot_base}.pdf"
condition_split_plot_png_path = output_dir / f"{condition_split_plot_base}.png"

publication_plot_base = (
    f"diag_norm_tied_batch_{batch_name}_aggregate_ledoff_truncated_{truncate_label_tag}_rtwrtstim_"
    f"by_ABL_publication_truncate_not_censor_linear_delay_with_choice_{fit_run_tag}"
)
publication_plot_pdf_path = output_dir / f"{publication_plot_base}.pdf"
publication_plot_png_path = output_dir / f"{publication_plot_base}.png"
publication_plot_pkl_path = output_dir / f"{publication_plot_base}.pkl"

delay_surface_base = (
    f"diag_norm_tied_batch_{batch_name}_aggregate_ledoff_delay_surface_"
    f"truncate_not_censor_linear_delay_with_choice_{fit_run_tag}"
)
delay_surface_pdf_path = output_dir / f"{delay_surface_base}.pdf"
delay_surface_png_path = output_dir / f"{delay_surface_base}.png"
delay_surface_pkl_path = output_dir / f"{delay_surface_base}.pkl"

quantile_plot_base = (
    f"diag_norm_tied_batch_{batch_name}_aggregate_ledoff_truncated_{truncate_label_tag}_quantiles_"
    f"by_ABL_abs_ILD_truncate_not_censor_linear_delay_with_choice_{fit_run_tag}"
)
quantile_plot_png_path = output_dir / f"{quantile_plot_base}.png"

combined_abs_ild_plot_base = (
    f"diag_norm_tied_batch_{batch_name}_aggregate_ledoff_truncated_{truncate_label_tag}_rtwrtstim_"
    f"by_abs_ILD_overlay_ABL_truncate_not_censor_linear_delay_with_choice_{fit_run_tag}"
)
combined_abs_ild_plot_pdf_path = output_dir / f"{combined_abs_ild_plot_base}.pdf"
combined_abs_ild_plot_png_path = output_dir / f"{combined_abs_ild_plot_base}.png"

combined_abl_plot_base = (
    f"diag_norm_tied_batch_{batch_name}_aggregate_ledoff_truncated_{truncate_label_tag}_rtwrtstim_"
    f"by_ABL_overlay_abs_ILD_truncate_not_censor_linear_delay_with_choice_{fit_run_tag}"
)
combined_abl_plot_pdf_path = output_dir / f"{combined_abl_plot_base}.pdf"
combined_abl_plot_png_path = output_dir / f"{combined_abl_plot_base}.png"

payload_path = output_dir / f"{raw_plot_base}.pkl"

print(f"Requested diagnostics run tag: {fit_run_tag_requested}")
print(f"Resolved fit run tag: {fit_run_tag}")
print(
    f"Using diagnostics truncation threshold: {truncate_rt_wrt_stim_s:.3f} s ({truncate_label_ms})"
)

required_norm_keys = [
    "rate_lambda_samples",
    "T_0_samples",
    "theta_E_samples",
    "w_samples",
    "bias_ms_samples",
    "abl_delay_coeff_ms_per_abl_samples",
    "abs_ild_delay_coeff_ms_per_unit_samples",
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
bias_ms = float(np.mean(vbmc_results["bias_ms_samples"]))
abl_delay_coeff_ms_per_abl = float(np.mean(vbmc_results["abl_delay_coeff_ms_per_abl_samples"]))
abs_ild_delay_coeff_ms_per_unit = float(
    np.mean(vbmc_results["abs_ild_delay_coeff_ms_per_unit_samples"])
)
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

delay_surface_ms = pd.DataFrame(
    {
        int(abs_ild): [
            1e3
            * get_t_E_aff_from_abl_abs_ild(
                abl=abl,
                abs_ild=abs_ild,
                bias_ms=bias_ms,
                abl_delay_coeff_ms_per_abl=abl_delay_coeff_ms_per_abl,
                abs_ild_delay_coeff_ms_per_unit=abs_ild_delay_coeff_ms_per_unit,
            )
            for abl in supported_abl_values
        ]
        for abs_ild in plot_abs_ild_values
    },
    index=[int(abl) for abl in supported_abl_values],
    dtype=np.float64,
)
delay_surface_ms.index.name = "ABL"
delay_surface_ms.columns.name = "abs_ILD"

print("Loaded aggregate linear-delay with choice parameters:")
print(
    f"  rate_lambda={rate_lambda:.6f}, T_0={T_0:.6f}, theta_E={theta_E:.6f}, "
    f"w={w:.6f}, Z_E={Z_E:.6f}, bias_ms={bias_ms:.6f}, "
    f"abl_delay_coeff_ms_per_abl={abl_delay_coeff_ms_per_abl:.6f}, "
    f"abs_ild_delay_coeff_ms_per_unit={abs_ild_delay_coeff_ms_per_unit:.6f}, "
    f"del_go={del_go:.6f}, rate_norm_l={rate_norm_l:.6f}"
)
print(
    f"  V_A={V_A:.6f}, theta_A={theta_A:.6f}, del_a_minus_del_LED={del_a_minus_del_LED:.6f}, "
    f"del_m_plus_del_LED={del_m_plus_del_LED:.6f}, t_A_aff={t_A_aff:.6f}, "
    f"lapse_prob={lapse_prob:.6f}, beta_lapse={beta_lapse:.6f}"
)
print("Derived posterior-mean delay surface (ms) by ABL x abs_ILD:")
print(delay_surface_ms.round(3))

theory_params = (
    rate_lambda,
    T_0,
    theta_E,
    Z_E,
    bias_ms,
    abl_delay_coeff_ms_per_abl,
    abs_ild_delay_coeff_ms_per_unit,
    del_go,
    rate_norm_l,
    V_A,
    theta_A,
    t_A_aff,
    lapse_prob,
    beta_lapse,
)


# %%
############ Rebuild LED-OFF aggregate diagnostics dataset ############
if not led_data_csv_path.exists():
    raise FileNotFoundError(f"Could not find data CSV: {led_data_csv_path}")

exp_df = pd.read_csv(led_data_csv_path)
exp_df["RTwrtStim"] = exp_df["timed_fix"] - exp_df["intended_fix"]
exp_df["t_LED"] = exp_df["intended_fix"] - exp_df["LED_onset_time"]
exp_df = exp_df.rename(columns={"timed_fix": "TotalFixTime"})
exp_df["abs_ILD"] = exp_df["ILD"].abs()

exp_df = exp_df[exp_df["RTwrtStim"] <= max_rtwrtstim_for_fit].copy()
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
fit_df = df_valid_and_aborts[df_valid_and_aborts["RTwrtStim"] <= max_rtwrtstim_for_fit].copy()

if len(fit_df) == 0:
    raise ValueError("No LED-OFF trials found after filtering.")

observed_abl_values = validate_supported_values(fit_df, "ABL", supported_abl_values)
observed_abs_ild_values = validate_supported_values(fit_df, "abs_ILD", plot_abs_ild_values)
abl_counts = format_counts(fit_df["ABL"])
abs_ild_counts = format_counts(fit_df["abs_ILD"])
condition_counts = build_condition_counts(fit_df, supported_abl_values, plot_abs_ild_values)
mask_truncated_data = (fit_df["RTwrtStim"] >= 0.0) & (fit_df["RTwrtStim"] <= truncate_rt_wrt_stim_s)
condition_counts_truncated = build_condition_counts(
    fit_df.loc[mask_truncated_data].copy(),
    supported_abl_values,
    plot_abs_ild_values,
)

print("Rebuilt LED-OFF aggregate diagnostics dataset for linear-delay with choice fit:")
print(f"  Total LED-OFF filtered trials (valid+aborts): {len(df_valid_and_aborts)}")
print(f"  LED-OFF trials used for diagnostics (valid+aborts): {len(fit_df)}")
print(f"  Supported ABL values in diagnostics dataset: {observed_abl_values.tolist()}")
print(f"  Supported abs_ILD values in diagnostics dataset: {observed_abs_ild_values.tolist()}")
print(f"  Diagnostics trial counts by ABL: {abl_counts}")
print(f"  Diagnostics trial counts by abs_ILD: {abs_ild_counts}")
print("  Counts by ABL x abs_ILD:")
print(condition_counts)
print("  Truncated counts by ABL x abs_ILD:")
print(condition_counts_truncated)


# %%
############ Monte Carlo theoretical RTwrtStim density (collapsed over choices) ############
rng = np.random.default_rng(seed)
print(f"Computing Monte Carlo theory with N_mc={N_mc}, n_jobs={n_jobs} ...")
overall_mc = compute_mc_average_curve(
    sample_df=fit_df,
    n_samples=N_mc,
    rng=rng,
    theory_params=theory_params,
    t_pts=t_pts,
    n_jobs=n_jobs,
    is_norm=is_norm,
    is_time_vary=is_time_vary,
    K_max=K_max,
)
theory_density = overall_mc["theory_density"]


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
############ Plot raw RTwrtStim diagnostics ############
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
ax.set_title(f"LED-OFF Aggregate RTD from linear-delay with choice fit ({batch_name})")
ax.set_xlim(-0.1, max(0.4, truncate_rt_wrt_stim_s + 0.05))
ax.legend()

fig.tight_layout()
fig.savefig(raw_plot_pdf_path, bbox_inches="tight")
fig.savefig(raw_plot_png_path, dpi=200, bbox_inches="tight")
if show_plot:
    plt.show()
plt.close(fig)

print(f"Saved diagnostics plot (PDF): {raw_plot_pdf_path}")
print(f"Saved diagnostics plot (PNG): {raw_plot_png_path}")


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
    f"LED-OFF Aggregate RTD truncated {truncate_label_ms} from linear-delay fit ({batch_name})"
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
############ Plot truncated RTwrtStim diagnostics by ABL ############
print(
    f"Computing per-ABL truncated theory with N_mc_per_abl_panel={N_mc_per_abl_panel}, "
    f"n_jobs={n_jobs} ..."
)

abl_panel_payload = {}
combined_ax_max = 0.0
abl_values_float = fit_df["ABL"].astype(float).to_numpy()

for abl in supported_abl_values:
    df_abl = fit_df[np.isclose(abl_values_float, float(abl))].copy()
    mc_payload_abl = compute_mc_average_curve(
        sample_df=df_abl,
        n_samples=N_mc_per_abl_panel,
        rng=rng,
        theory_params=theory_params,
        t_pts=t_pts,
        n_jobs=n_jobs,
        is_norm=is_norm,
        is_time_vary=is_time_vary,
        K_max=K_max,
    )
    trunc_payload_abl = build_truncated_density_payload(
        rt_values=df_abl["RTwrtStim"].to_numpy(dtype=np.float64),
        n_total_condition=len(df_abl),
        theory_density_full=mc_payload_abl["theory_density"],
        mask_truncated=mask_truncated,
        t_pts_truncated=t_pts_truncated,
        hist_edges_truncated=hist_edges_truncated,
        data_bin_widths_truncated=data_bin_widths_truncated,
        truncate_rt_wrt_stim_s=truncate_rt_wrt_stim_s,
        normalize_within_window=normalize_per_abl,
    )

    combined_ax_max = max(
        combined_ax_max,
        float(np.max(trunc_payload_abl["data_density_truncated_plot"])),
        float(np.max(trunc_payload_abl["theory_density_truncated_plot"])),
    )

    abl_panel_payload[int(abl)] = {
        "n_rows": int(len(df_abl)),
        "sampled_positions": mc_payload_abl["sampled_positions"],
        **trunc_payload_abl,
    }

    print(
        f"ABL={abl}: n_rows={len(df_abl)}, "
        f"n_truncated={trunc_payload_abl['n_truncated_points']}, "
        f"data_plot_area={trunc_payload_abl['data_area_truncated_plot']:.6f}, "
        f"theory_plot_area={trunc_payload_abl['theory_area_truncated_plot']:.6f}"
    )
# %%
fig_abl, axes_abl = plt.subplots(1, 4, figsize=(20, 4.8), sharey=True)
for ax, abl in zip(axes_abl[:3], supported_abl_values):
    abl_int = int(abl)
    color = abl_colors[abl_int]
    payload_abl = abl_panel_payload[abl_int]

    ax.step(
        data_bin_centers_truncated,
        payload_abl["data_density_truncated_plot"],
        where="mid",
        lw=2.0,
        color=color,
        alpha=0.75,
        label="Data",
    )
    ax.plot(
        t_pts_truncated,
        payload_abl["theory_density_truncated_plot"],
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
    ax.set_title(
        f"ABL {abl_int}  (n={payload_abl['n_rows']}, trunc={payload_abl['n_truncated_points']})"
    )
    # ax.set_xlim(0.0, truncate_rt_wrt_stim_s)
    ax.set_xlim(0.0, 0.115)

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
        payload_abl["data_density_truncated_plot"],
        where="mid",
        lw=1.8,
        color=color,
        alpha=0.55,
        label=f"ABL {abl_int} data",
    )
    ax_combined.plot(
        t_pts_truncated,
        payload_abl["theory_density_truncated_plot"],
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
# ax_combined.set_xlim(0.0, truncate_rt_wrt_stim_s)
ax_combined.set_xlim(0.0, 0.115)

ax_combined.set_xlabel("RT - t_stim (s)")
ax_combined.legend(fontsize=8, ncol=2)

if combined_ax_max > 0:
    axes_abl[0].set_ylim(0.0, combined_ax_max * 1.1)

abl_counts_str = ", ".join(
    f"ABL{int(abl)}={abl_panel_payload[int(abl)]['n_rows']}" for abl in supported_abl_values
)
fig_abl.suptitle(
    f"LED-OFF Aggregate RTD truncated {truncate_label_ms} by ABL from linear-delay fit ({batch_name})\n"
    f"Trials: {abl_counts_str}",
    y=1.04,
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
############ Plot truncated RTwrtStim diagnostics by ABL x abs_ILD ############
print(
    f"Computing per-condition truncated theory with N_mc_per_condition={N_mc_per_condition}, "
    f"n_jobs={n_jobs} ..."
)

condition_panel_payload = {}
combined_condition_ax_max = 0.0

for abl in supported_abl_values:
    for abs_ild in plot_abs_ild_values:
        condition_df = fit_df[
            np.isclose(fit_df["ABL"], float(abl))
            & np.isclose(fit_df["abs_ILD"], float(abs_ild))
        ].copy()

        mc_payload_condition = compute_mc_average_curve(
            sample_df=condition_df,
            n_samples=N_mc_per_condition,
            rng=rng,
            theory_params=theory_params,
            t_pts=t_pts,
            n_jobs=n_jobs,
            is_norm=is_norm,
            is_time_vary=is_time_vary,
            K_max=K_max,
        )
        trunc_payload_condition = build_truncated_density_payload(
            rt_values=condition_df["RTwrtStim"].to_numpy(dtype=np.float64),
            n_total_condition=len(condition_df),
            theory_density_full=mc_payload_condition["theory_density"],
            mask_truncated=mask_truncated,
            t_pts_truncated=t_pts_truncated,
            hist_edges_truncated=hist_edges_truncated,
            data_bin_widths_truncated=data_bin_widths_truncated,
            truncate_rt_wrt_stim_s=truncate_rt_wrt_stim_s,
            normalize_within_window=normalize_per_condition,
        )

        combined_condition_ax_max = max(
            combined_condition_ax_max,
            float(np.max(trunc_payload_condition["data_density_truncated_plot"])),
            float(np.max(trunc_payload_condition["theory_density_truncated_plot"])),
        )

        condition_panel_payload[(int(abl), int(abs_ild))] = {
            "n_rows": int(len(condition_df)),
            "sampled_positions": mc_payload_condition["sampled_positions"],
            **trunc_payload_condition,
        }

        print(
            f"ABL={int(abl)}, abs_ILD={int(abs_ild)}: "
            f"n_rows={len(condition_df)}, "
            f"n_truncated={trunc_payload_condition['n_truncated_points']}, "
            f"data_plot_area={trunc_payload_condition['data_area_truncated_plot']:.6f}, "
            f"theory_plot_area={trunc_payload_condition['theory_area_truncated_plot']:.6f}"
        )

fig_cond, axes_cond = plt.subplots(
    len(supported_abl_values),
    len(plot_abs_ild_values),
    figsize=(
        condition_panel_width * len(plot_abs_ild_values),
        condition_panel_height * len(supported_abl_values),
    ),
    sharex=True,
    sharey=True,
    squeeze=False,
)

for row_idx, abl in enumerate(supported_abl_values):
    for col_idx, abs_ild in enumerate(plot_abs_ild_values):
        ax = axes_cond[row_idx, col_idx]
        payload_condition = condition_panel_payload[(int(abl), int(abs_ild))]
        color = abl_colors[int(abl)]

        ax.step(
            data_bin_centers_truncated,
            payload_condition["data_density_truncated_plot"],
            where="mid",
            lw=1.6,
            color=color,
            alpha=0.75,
        )
        ax.plot(
            t_pts_truncated,
            payload_condition["theory_density_truncated_plot"],
            lw=2.0,
            color=color,
        )
        ax.axvline(
            x=truncate_rt_wrt_stim_s,
            color="crimson",
            linestyle="--",
            linewidth=1.0,
        )
        ax.set_xlim(0.0, truncate_rt_wrt_stim_s)
        ax.set_title(
            f"ABL {int(abl)}, |ILD| {int(abs_ild)}\n"
            f"n={payload_condition['n_rows']}, trunc={payload_condition['n_truncated_points']}",
            fontsize=10,
        )
        ax.grid(alpha=0.15, linewidth=0.6)
        if row_idx == len(supported_abl_values) - 1:
            ax.set_xlabel("RT - t_stim (s)")
        if col_idx == 0:
            ax.set_ylabel("Density")

if combined_condition_ax_max > 0:
    axes_cond[0, 0].set_ylim(0.0, combined_condition_ax_max * 1.08)

fig_cond.suptitle(
    f"LED-OFF Aggregate RTD truncated {truncate_label_ms} by ABL x abs_ILD from linear-delay fit ({batch_name})",
    y=1.02,
)
fig_cond.tight_layout(rect=[0, 0, 1, 0.98])
fig_cond.savefig(condition_split_plot_pdf_path, bbox_inches="tight")
fig_cond.savefig(condition_split_plot_png_path, dpi=200, bbox_inches="tight")
if show_plot:
    plt.show()
plt.close(fig_cond)

print(f"Saved condition-split diagnostics plot (PDF): {condition_split_plot_pdf_path}")
print(f"Saved condition-split diagnostics plot (PNG): {condition_split_plot_png_path}")




# %%
############ Delay surface plot ############
delay_surface_payload = {
    "config": {
        "batch_name": batch_name,
        "supported_ABL_values": list(supported_abl_values),
        "supported_abs_ILD_values": list(plot_abs_ild_values),
        "delay_rule_ms": fit_config.get("delay_rule_ms"),
        "negative_delay_rule": fit_config.get("negative_delay_rule"),
    },
    "bias_ms": bias_ms,
    "abl_delay_coeff_ms_per_abl": abl_delay_coeff_ms_per_abl,
    "abs_ild_delay_coeff_ms_per_unit": abs_ild_delay_coeff_ms_per_unit,
    "delay_surface_ms": delay_surface_ms,
    "pdf_path": str(delay_surface_pdf_path),
    "png_path": str(delay_surface_png_path),
}

with open(delay_surface_pkl_path, "wb") as f:
    pickle.dump(delay_surface_payload, f)

fig_delay, ax_delay = plt.subplots(figsize=delay_surface_figsize)
delay_matrix = delay_surface_ms.to_numpy(dtype=np.float64)
im = ax_delay.imshow(delay_matrix, aspect="auto", cmap="viridis")

ax_delay.set_xticks(np.arange(len(plot_abs_ild_values)))
ax_delay.set_xticklabels([int(value) for value in plot_abs_ild_values])
ax_delay.set_yticks(np.arange(len(supported_abl_values)))
ax_delay.set_yticklabels([int(value) for value in supported_abl_values])
ax_delay.set_xlabel("abs_ILD")
ax_delay.set_ylabel("ABL")
delay_surface_title = (
    "Posterior-mean evidence delay (ms)\n"
    f"delay = {bias_ms:.2f} {abl_delay_coeff_ms_per_abl:+.3f}*ABL "
    f"{abs_ild_delay_coeff_ms_per_unit:+.3f}*|ILD|"
)
ax_delay.set_title(delay_surface_title)

for row_idx, abl in enumerate(supported_abl_values):
    for col_idx, abs_ild in enumerate(plot_abs_ild_values):
        ax_delay.text(
            col_idx,
            row_idx,
            f"{delay_surface_ms.loc[int(abl), int(abs_ild)]:.1f}",
            ha="center",
            va="center",
            color="white",
            fontsize=9,
        )

colorbar = fig_delay.colorbar(im, ax=ax_delay, shrink=0.92)
colorbar.set_label("Delay (ms)")

fig_delay.tight_layout()
fig_delay.savefig(delay_surface_pdf_path, bbox_inches="tight")
fig_delay.savefig(delay_surface_png_path, dpi=delay_surface_dpi, bbox_inches="tight")
if show_plot:
    plt.show()
plt.close(fig_delay)

print(f"Saved delay surface payload: {delay_surface_pkl_path}")
print(f"Saved delay surface plot (PDF): {delay_surface_pdf_path}")
print(f"Saved delay surface plot (PNG): {delay_surface_png_path}")


# %%
############ Quantile-vs-abs_ILD diagnostics ############
print(
    f"Computing quantile diagnostics with N_mc_per_quantile_grid={N_mc_per_quantile_grid}, "
    f"n_jobs={n_jobs} ..."
)

fit_df_truncated = fit_df[
    (fit_df["RTwrtStim"] >= 0.0) & (fit_df["RTwrtStim"] <= truncate_rt_wrt_stim_s)
].copy()

data_quantiles_ms_by_abl = {}
data_truncated_counts_by_abl = {}
for abl in supported_abl_values:
    abl_int = int(abl)
    data_quantiles_ms_by_abl[abl_int] = {}
    data_truncated_counts_by_abl[abl_int] = {}

    for abs_ild in plot_abs_ild_values:
        abs_ild_int = int(abs_ild)
        condition_df = fit_df_truncated[
            np.isclose(fit_df_truncated["ABL"], float(abl))
            & np.isclose(fit_df_truncated["abs_ILD"], float(abs_ild))
        ].copy()
        data_quantiles_s, n_truncated_condition = compute_empirical_truncated_quantiles(
            rt_values=condition_df["RTwrtStim"].to_numpy(dtype=np.float64),
            quantile_levels=quantile_levels,
            truncate_rt_wrt_stim_s=truncate_rt_wrt_stim_s,
            min_trials_for_quantiles=min_trials_for_data_quantiles,
        )
        data_quantiles_ms_by_abl[abl_int][abs_ild_int] = 1e3 * data_quantiles_s
        data_truncated_counts_by_abl[abl_int][abs_ild_int] = int(n_truncated_condition)

    print(
        f"ABL={abl_int}: truncated data counts by abs_ILD for quantiles = "
        f"{data_truncated_counts_by_abl[abl_int]}"
    )

model_quantiles_ms_by_abl = {}
for abl in supported_abl_values:
    abl_int = int(abl)
    abl_source_df = fit_df[np.isclose(fit_df["ABL"], float(abl))].copy()
    if len(abl_source_df) == 0:
        raise ValueError(f"No source rows available for ABL={abl_int} in quantile diagnostics.")

    model_quantiles_ms = np.full(
        (len(quantile_levels), len(model_abs_ild_grid)),
        np.nan,
        dtype=np.float64,
    )
    for abs_ild_idx, abs_ild in enumerate(model_abs_ild_grid):
        sampled_rows = build_balanced_signed_ild_sample_rows(
            source_df=abl_source_df,
            n_samples=N_mc_per_quantile_grid,
            rng=rng,
            abl=abl,
            abs_ild=abs_ild,
        )
        mc_payload_quant = compute_average_curve_from_rows(
            sampled_rows=sampled_rows,
            theory_params=theory_params,
            t_pts=t_pts,
            n_jobs=n_jobs,
            is_norm=is_norm,
            is_time_vary=is_time_vary,
            K_max=K_max,
        )

        theory_density_truncated_quant = mc_payload_quant["theory_density"][mask_truncated].copy()
        theory_area_truncated_quant = float(
            np.trapz(theory_density_truncated_quant, t_pts_truncated)
        )
        if theory_area_truncated_quant <= 0:
            continue

        theory_density_truncated_quant /= theory_area_truncated_quant
        model_quantiles_ms[:, abs_ild_idx] = 1e3 * compute_quantiles_from_truncated_density(
            t_pts_truncated=t_pts_truncated,
            density_truncated=theory_density_truncated_quant,
            quantile_levels=quantile_levels,
        )

    model_quantiles_ms_by_abl[abl_int] = model_quantiles_ms
    print(
        f"ABL={abl_int}: computed model quantiles on abs_ILD grid "
        f"{model_abs_ild_grid[0]:.1f} to {model_abs_ild_grid[-1]:.1f}"
    )
# %%
fig_quant, axes_quant = plt.subplots(
    1,
    len(supported_abl_values),
    figsize=quantile_figsize,
    sharex=True,
    sharey=True,
)

if len(supported_abl_values) == 1:
    axes_quant = [axes_quant]

for ax, abl in zip(axes_quant, supported_abl_values):
    abl_int = int(abl)
    model_quantiles_ms = model_quantiles_ms_by_abl[abl_int]

    for q_idx, q_level in enumerate(quantile_levels):
        ax.plot(
            model_abs_ild_grid,
            model_quantiles_ms[q_idx],
            color="black",
            lw=2.0,
            alpha=0.6,
        )

        data_x = []
        data_y = []
        for abs_ild in plot_abs_ild_values:
            abs_ild_int = int(abs_ild)
            data_quantile_value = data_quantiles_ms_by_abl[abl_int][abs_ild_int][q_idx]
            if np.isfinite(data_quantile_value):
                data_x.append(abs_ild_int)
                data_y.append(data_quantile_value)

        if len(data_x) > 0:
            ax.scatter(
                data_x,
                data_y,
                s=55,
                color="red",
                edgecolors="white",
                linewidths=0.6,
                zorder=3,
            )

    ax.set_title(f"ABL {abl_int}")
    # ax.set_xlim(float(model_abs_ild_grid[0]), float(model_abs_ild_grid[-1]))
    ax.set_xlim(float(model_abs_ild_grid[0]), float(model_abs_ild_grid[-1] + 0.5))

    ax.set_xlabel("abs_ILD")
    ax.grid(alpha=0.18, linewidth=0.6)

axes_quant[0].set_ylabel("RT quantile (ms)")

fig_quant.suptitle(
    f"Aggregate LED off: [0, {truncate_label_ms}] delay linear, with choice\n",
    y=1.04,
)
fig_quant.tight_layout(rect=[0, 0, 1, 0.95])
fig_quant.savefig(quantile_plot_png_path, dpi=quantile_plot_dpi, bbox_inches="tight")
if show_plot:
    plt.show()
plt.close(fig_quant)

print(f"Saved quantile diagnostics plot (PNG): {quantile_plot_png_path}")


# %%
############ Combined truncated RTwrtStim overlays ############
print(
    "Building combined truncated RTD overlays from the ABL x abs_ILD condition payloads "
    f"with xlim={int(round(1e3 * combined_condition_xlim_s))} ms ..."
)

combined_condition_xlim_s_effective = min(combined_condition_xlim_s, truncate_rt_wrt_stim_s)
combined_data_xlim_mask = data_bin_centers_truncated <= combined_condition_xlim_s_effective
combined_theory_xlim_mask = t_pts_truncated <= combined_condition_xlim_s_effective
abs_ild_color_values = plt.cm.viridis(np.linspace(0.12, 0.88, len(plot_abs_ild_values)))
abs_ild_colors = {
    int(abs_ild): abs_ild_color_values[idx]
    for idx, abs_ild in enumerate(plot_abs_ild_values)
}

combined_abs_ild_ax_max = 0.0
for abs_ild in plot_abs_ild_values:
    for abl in supported_abl_values:
        payload_condition = condition_panel_payload[(int(abl), int(abs_ild))]
        if np.any(combined_data_xlim_mask):
            combined_abs_ild_ax_max = max(
                combined_abs_ild_ax_max,
                float(np.max(payload_condition["data_density_truncated_plot"][combined_data_xlim_mask])),
            )
        if np.any(combined_theory_xlim_mask):
            combined_abs_ild_ax_max = max(
                combined_abs_ild_ax_max,
                float(
                    np.max(
                        payload_condition["theory_density_truncated_plot"][
                            combined_theory_xlim_mask
                        ]
                    )
                ),
            )

fig_abs_ild, axes_abs_ild = plt.subplots(
    1,
    len(plot_abs_ild_values),
    figsize=combined_abs_ild_figsize,
    sharex=True,
    sharey=True,
)

if len(plot_abs_ild_values) == 1:
    axes_abs_ild = [axes_abs_ild]

for ax, abs_ild in zip(axes_abs_ild, plot_abs_ild_values):
    for abl in supported_abl_values:
        payload_condition = condition_panel_payload[(int(abl), int(abs_ild))]
        color = abl_colors[int(abl)]
        ax.step(
            data_bin_centers_truncated,
            payload_condition["data_density_truncated_plot"],
            where="mid",
            lw=1.3,
            color=color,
            alpha=0.55,
        )
        ax.plot(
            t_pts_truncated,
            payload_condition["theory_density_truncated_plot"],
            lw=2.0,
            color=color,
            label=f"ABL {int(abl)}" if int(abs_ild) == int(plot_abs_ild_values[0]) else None,
        )

    ax.set_xlim(0.0, combined_condition_xlim_s_effective)
    ax.set_title(f"|ILD| {int(abs_ild)}")
    ax.set_xlabel("RT - t_stim (s)")
    ax.grid(alpha=0.16, linewidth=0.6)

axes_abs_ild[0].set_ylabel("Density")
if combined_abs_ild_ax_max > 0:
    axes_abs_ild[0].set_ylim(0.0, combined_abs_ild_ax_max * 1.08)
axes_abs_ild[0].legend(loc="upper right", fontsize=9, frameon=False)

fig_abs_ild.suptitle(
    f"LED-OFF Aggregate RTD up to {int(round(1e3 * combined_condition_xlim_s_effective))} ms by |ILD| "
    f"(step=data, line=model; {batch_name})",
    y=1.04,
)
fig_abs_ild.tight_layout(rect=[0, 0, 1, 0.95])
fig_abs_ild.savefig(combined_abs_ild_plot_pdf_path, bbox_inches="tight")
fig_abs_ild.savefig(
    combined_abs_ild_plot_png_path,
    dpi=combined_condition_plot_dpi,
    bbox_inches="tight",
)
if show_plot:
    plt.show()
plt.close(fig_abs_ild)

print(f"Saved combined abs_ILD overlay plot (PDF): {combined_abs_ild_plot_pdf_path}")
print(f"Saved combined abs_ILD overlay plot (PNG): {combined_abs_ild_plot_png_path}")


combined_abl_ax_max = 0.0
for abl in supported_abl_values:
    for abs_ild in plot_abs_ild_values:
        payload_condition = condition_panel_payload[(int(abl), int(abs_ild))]
        if np.any(combined_data_xlim_mask):
            combined_abl_ax_max = max(
                combined_abl_ax_max,
                float(np.max(payload_condition["data_density_truncated_plot"][combined_data_xlim_mask])),
            )
        if np.any(combined_theory_xlim_mask):
            combined_abl_ax_max = max(
                combined_abl_ax_max,
                float(
                    np.max(
                        payload_condition["theory_density_truncated_plot"][
                            combined_theory_xlim_mask
                        ]
                    )
                ),
            )

fig_combined_abl, axes_combined_abl = plt.subplots(
    1,
    len(supported_abl_values),
    figsize=combined_abl_figsize,
    sharex=True,
    sharey=True,
)

if len(supported_abl_values) == 1:
    axes_combined_abl = [axes_combined_abl]

for ax, abl in zip(axes_combined_abl, supported_abl_values):
    for abs_ild in plot_abs_ild_values:
        payload_condition = condition_panel_payload[(int(abl), int(abs_ild))]
        color = abs_ild_colors[int(abs_ild)]
        ax.step(
            data_bin_centers_truncated,
            payload_condition["data_density_truncated_plot"],
            where="mid",
            lw=1.3,
            color=color,
            alpha=0.55,
        )
        ax.plot(
            t_pts_truncated,
            payload_condition["theory_density_truncated_plot"],
            lw=2.0,
            color=color,
            label=(
                f"|ILD| {int(abs_ild)}"
                if int(abl) == int(supported_abl_values[0])
                else None
            ),
        )

    ax.set_xlim(0.0, combined_condition_xlim_s_effective)
    ax.set_title(f"ABL {int(abl)}")
    ax.set_xlabel("RT - t_stim (s)")
    ax.grid(alpha=0.16, linewidth=0.6)

axes_combined_abl[0].set_ylabel("Density")
if combined_abl_ax_max > 0:
    axes_combined_abl[0].set_ylim(0.0, combined_abl_ax_max * 1.08)
axes_combined_abl[0].legend(loc="upper right", fontsize=9, frameon=False)

fig_combined_abl.suptitle(
    f"LED-OFF Aggregate RTD up to {int(round(1e3 * combined_condition_xlim_s_effective))} ms by ABL "
    f"(step=data, line=model; {batch_name})",
    y=1.04,
)
fig_combined_abl.tight_layout(rect=[0, 0, 1, 0.95])
fig_combined_abl.savefig(combined_abl_plot_pdf_path, bbox_inches="tight")
fig_combined_abl.savefig(
    combined_abl_plot_png_path,
    dpi=combined_condition_plot_dpi,
    bbox_inches="tight",
)
if show_plot:
    plt.show()
plt.close(fig_combined_abl)

print(f"Saved combined ABL overlay plot (PDF): {combined_abl_plot_pdf_path}")
print(f"Saved combined ABL overlay plot (PNG): {combined_abl_plot_png_path}")
