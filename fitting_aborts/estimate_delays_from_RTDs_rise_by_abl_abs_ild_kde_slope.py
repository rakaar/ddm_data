# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# %%
############ Parameters (edit here) ############
SHOW_PLOT = True

batch_name = "LED7"
session_type = 7
training_level = 16
allowed_repeat_trials = [0, 2]

supported_abl_values = (20, 40, 60)
supported_abs_ild_values = (1, 2, 4, 8, 16)
abl_colors = {
    20: "tab:blue",
    40: "tab:orange",
    60: "tab:green",
}
abs_ild_colors = {
    1: "tab:blue",
    2: "tab:orange",
    4: "tab:green",
    8: "tab:red",
    16: "tab:purple",
}

max_rtwrtstim_for_fit = 1.0
truncate_rt_wrt_stim_s = 0.140
data_bin_size_s_truncated = 5e-3
kde_bandwidth_s = 15e-3
kde_eval_step_s = 5e-4
reflect_kde_at_zero = True
slope_window_n_points = 10
slope_onset_fraction_of_peak = 0.10
baseline_window_end_s = 40e-3
baseline_sigma_multiplier = 5.0
baseline_min_consecutive_duration_s = 5e-3

panel_width = 3.8
panel_height = 3.0
png_dpi = 300
show_plot = SHOW_PLOT

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
led_data_csv_path = REPO_ROOT / "out_LED.csv"
output_dir = SCRIPT_DIR / "estimate_delays_from_RTDs_rise_by_abl_abs_ild_kde_slope"
manual_pick_input_dir = SCRIPT_DIR / "manual_pick_delays_from_RTDs_by_abl_abs_ild"


# %%
############ Helpers ############
def epanechnikov_kernel(u: np.ndarray) -> np.ndarray:
    kernel_values = np.zeros_like(u, dtype=float)
    mask = np.abs(u) <= 1.0
    kernel_values[mask] = 0.75 * (1.0 - u[mask] ** 2)
    return kernel_values


def compute_epanechnikov_kde_density(
    eval_points: np.ndarray,
    sample_points: np.ndarray,
    bandwidth_s: float,
    reflect_at_zero: bool,
):
    if len(sample_points) == 0:
        return np.zeros_like(eval_points, dtype=float)

    u = (eval_points[:, None] - sample_points[None, :]) / bandwidth_s
    density = epanechnikov_kernel(u).sum(axis=1) / (len(sample_points) * bandwidth_s)

    if reflect_at_zero:
        u_reflected = (eval_points[:, None] + sample_points[None, :]) / bandwidth_s
        density += epanechnikov_kernel(u_reflected).sum(axis=1) / (len(sample_points) * bandwidth_s)

    return density


def compute_rolling_linear_slope(
    x_points: np.ndarray,
    y_points: np.ndarray,
    window_n_points: int,
):
    if window_n_points <= 1:
        raise ValueError("slope_window_n_points must be at least 2.")
    if len(x_points) < window_n_points:
        return np.array([], dtype=float), np.array([], dtype=float)

    slope_times = []
    slope_values = []

    for start_idx in range(len(x_points) - window_n_points + 1):
        x_window = x_points[start_idx : start_idx + window_n_points]
        y_window = y_points[start_idx : start_idx + window_n_points]
        slope_value = np.polyfit(x_window, y_window, deg=1)[0]
        slope_times.append(float(np.mean(x_window)))
        slope_values.append(float(slope_value))

    return np.asarray(slope_times, dtype=float), np.asarray(slope_values, dtype=float)


def build_rtd_kde_and_slope_payload(
    rt_values: np.ndarray,
    n_total_condition: int,
    hist_edges_truncated: np.ndarray,
    hist_bin_widths_truncated: np.ndarray,
    kde_eval_points: np.ndarray,
    kde_bandwidth_s: float,
    truncate_rt_wrt_stim_s: float,
    reflect_kde_at_zero: bool,
    slope_window_n_points: int,
    slope_onset_fraction_of_peak: float,
    baseline_window_end_s: float,
    baseline_sigma_multiplier: float,
    baseline_min_consecutive_duration_s: float,
):
    truncated_rt_values = rt_values[(rt_values >= 0.0) & (rt_values <= truncate_rt_wrt_stim_s)]
    n_hist_bins = len(hist_edges_truncated) - 1

    if n_total_condition <= 0:
        return {
            "n_truncated_points": 0,
            "data_density_truncated": np.zeros(n_hist_bins, dtype=float),
            "data_area_truncated": 0.0,
            "kde_density_truncated": np.zeros_like(kde_eval_points, dtype=float),
            "kde_area_truncated": 0.0,
            "slope_times_s": np.array([], dtype=float),
            "slope_values_per_ms": np.array([], dtype=float),
            "peak_slope_per_ms": np.nan,
            "peak_slope_time_s": np.nan,
            "baseline_slope_mean_per_ms": np.nan,
            "baseline_slope_std_per_ms": np.nan,
            "baseline_slope_threshold_per_ms": np.nan,
            "baseline_significant_onset_time_s": np.nan,
        }

    data_counts_truncated, _ = np.histogram(
        truncated_rt_values,
        bins=hist_edges_truncated,
        density=False,
    )
    data_density_truncated = data_counts_truncated / (n_total_condition * hist_bin_widths_truncated)
    data_area_truncated = float(np.sum(data_density_truncated * hist_bin_widths_truncated))

    if len(truncated_rt_values) >= 2:
        kde_unit_density = compute_epanechnikov_kde_density(
            eval_points=kde_eval_points,
            sample_points=truncated_rt_values,
            bandwidth_s=kde_bandwidth_s,
            reflect_at_zero=reflect_kde_at_zero,
        )
        kde_density_truncated = kde_unit_density * (len(truncated_rt_values) / n_total_condition)
    else:
        kde_density_truncated = np.zeros_like(kde_eval_points, dtype=float)

    kde_area_truncated = float(np.trapz(kde_density_truncated, kde_eval_points))
    slope_times_s, slope_values_per_s = compute_rolling_linear_slope(
        x_points=kde_eval_points,
        y_points=kde_density_truncated,
        window_n_points=slope_window_n_points,
    )
    slope_values_per_ms = slope_values_per_s * 1e-3

    if len(slope_values_per_ms) == 0:
        peak_slope_per_ms = np.nan
        peak_slope_time_s = np.nan
        slope_onset_time_s = np.nan
        slope_onset_threshold_per_ms = np.nan
        baseline_slope_mean_per_ms = np.nan
        baseline_slope_std_per_ms = np.nan
        baseline_slope_threshold_per_ms = np.nan
        baseline_significant_onset_time_s = np.nan
    else:
        peak_idx = int(np.argmax(slope_values_per_ms))
        peak_slope_per_ms = float(slope_values_per_ms[peak_idx])
        peak_slope_time_s = float(slope_times_s[peak_idx])
        slope_onset_threshold_per_ms = slope_onset_fraction_of_peak * peak_slope_per_ms
        slope_onset_time_s = np.nan
        for time_s, slope_value in zip(slope_times_s, slope_values_per_ms):
            if slope_value >= slope_onset_threshold_per_ms:
                slope_onset_time_s = float(time_s)
                break

        baseline_mask = slope_times_s <= baseline_window_end_s
        if np.any(baseline_mask):
            baseline_slope_values = slope_values_per_ms[baseline_mask]
            baseline_slope_mean_per_ms = float(np.mean(baseline_slope_values))
            baseline_slope_std_per_ms = float(np.std(baseline_slope_values))
            baseline_slope_threshold_per_ms = (
                baseline_slope_mean_per_ms
                + baseline_sigma_multiplier * baseline_slope_std_per_ms
            )

            if len(slope_times_s) > 1:
                slope_time_step_s = float(np.median(np.diff(slope_times_s)))
            else:
                slope_time_step_s = np.nan

            if np.isfinite(slope_time_step_s) and slope_time_step_s > 0:
                min_consecutive_points = int(
                    np.ceil(baseline_min_consecutive_duration_s / slope_time_step_s)
                )
            else:
                min_consecutive_points = 1
            min_consecutive_points = max(min_consecutive_points, 1)

            above_baseline_threshold = slope_values_per_ms >= baseline_slope_threshold_per_ms
            baseline_significant_onset_time_s = np.nan
            consecutive_points = 0
            onset_start_idx = None

            for idx, is_above in enumerate(above_baseline_threshold):
                if is_above:
                    consecutive_points += 1
                    if consecutive_points == 1:
                        onset_start_idx = idx
                    if consecutive_points >= min_consecutive_points:
                        baseline_significant_onset_time_s = float(slope_times_s[onset_start_idx])
                        break
                else:
                    consecutive_points = 0
                    onset_start_idx = None
        else:
            baseline_slope_mean_per_ms = np.nan
            baseline_slope_std_per_ms = np.nan
            baseline_slope_threshold_per_ms = np.nan
            baseline_significant_onset_time_s = np.nan

    return {
        "n_truncated_points": int(len(truncated_rt_values)),
        "data_density_truncated": data_density_truncated,
        "data_area_truncated": data_area_truncated,
        "kde_density_truncated": kde_density_truncated,
        "kde_area_truncated": kde_area_truncated,
        "slope_times_s": slope_times_s,
        "slope_values_per_ms": slope_values_per_ms,
        "peak_slope_per_ms": peak_slope_per_ms,
        "peak_slope_time_s": peak_slope_time_s,
        "slope_onset_threshold_per_ms": slope_onset_threshold_per_ms,
        "slope_onset_time_s": slope_onset_time_s,
        "baseline_slope_mean_per_ms": baseline_slope_mean_per_ms,
        "baseline_slope_std_per_ms": baseline_slope_std_per_ms,
        "baseline_slope_threshold_per_ms": baseline_slope_threshold_per_ms,
        "baseline_significant_onset_time_s": baseline_significant_onset_time_s,
    }


# %%
############ Rebuild LED-OFF aggregate diagnostics dataset ############
if not led_data_csv_path.exists():
    raise FileNotFoundError(f"Could not find data CSV: {led_data_csv_path}")

output_dir.mkdir(parents=True, exist_ok=True)

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
plot_df = df_valid_and_aborts[df_valid_and_aborts["RTwrtStim"] < max_rtwrtstim_for_fit].copy()
plot_df["abs_ILD"] = plot_df["ILD"].abs()

if len(plot_df) == 0:
    raise ValueError("No LED-OFF trials found after filtering.")

observed_abl_values = np.sort(plot_df["ABL"].dropna().astype(float).unique())
unexpected_abl_values = [
    float(abl)
    for abl in observed_abl_values
    if not any(np.isclose(float(abl), float(supported)) for supported in supported_abl_values)
]
if unexpected_abl_values:
    raise ValueError(
        f"Unexpected ABL values in LED-OFF RTD dataset: {unexpected_abl_values}. "
        f"Supported values are {supported_abl_values}."
    )

observed_abs_ild_values = np.sort(plot_df["abs_ILD"].dropna().astype(float).unique())
unexpected_abs_ild_values = [
    float(abs_ild)
    for abs_ild in observed_abs_ild_values
    if not any(np.isclose(float(abs_ild), float(supported)) for supported in supported_abs_ild_values)
]
if unexpected_abs_ild_values:
    raise ValueError(
        f"Unexpected abs_ILD values in LED-OFF RTD dataset: {unexpected_abs_ild_values}. "
        f"Supported values are {supported_abs_ild_values}."
    )

print("Rebuilt LED-OFF aggregate RTD dataset for KDE slope exploration by ABL x abs_ILD:")
print(f"  batch_name={batch_name}")
print(f"  Total LED-OFF filtered trials (valid+aborts): {len(df_valid_and_aborts)}")
print(f"  LED-OFF trials used for plots (valid+aborts): {len(plot_df)}")
print(f"  Supported ABL values in plot dataset: {observed_abl_values.tolist()}")
print(f"  Supported abs_ILD values in plot dataset: {observed_abs_ild_values.tolist()}")


# %%
############ Build per-condition histogram + KDE + slope payload ############
truncate_rt_wrt_stim_ms = int(round(float(truncate_rt_wrt_stim_s) * 1e3))
truncate_label_ms = f"{truncate_rt_wrt_stim_ms} ms"
truncate_label_tag = f"{truncate_rt_wrt_stim_ms}ms"
hist_edges_truncated = np.linspace(
    0.0,
    truncate_rt_wrt_stim_s,
    int(round(truncate_rt_wrt_stim_s / data_bin_size_s_truncated)) + 1,
)
hist_bin_widths_truncated = np.diff(hist_edges_truncated)
data_bin_centers_truncated = hist_edges_truncated[:-1] + 0.5 * hist_bin_widths_truncated
kde_eval_points = np.arange(0.0, truncate_rt_wrt_stim_s + kde_eval_step_s, kde_eval_step_s)
manual_pick_csv_path = (
    manual_pick_input_dir
    / f"manual_delay_picks_batch_{batch_name}_truncated_{truncate_label_tag}_by_ABL_abs_ILD.csv"
)

condition_payload = {}
combined_density_max = 0.0
combined_slope_abs_max = 0.0

for abl in supported_abl_values:
    for abs_ild in supported_abs_ild_values:
        condition_df = plot_df[
            np.isclose(plot_df["ABL"], float(abl))
            & np.isclose(plot_df["abs_ILD"], float(abs_ild))
        ].copy()

        payload = build_rtd_kde_and_slope_payload(
            rt_values=condition_df["RTwrtStim"].to_numpy(dtype=np.float64),
            n_total_condition=len(condition_df),
            hist_edges_truncated=hist_edges_truncated,
            hist_bin_widths_truncated=hist_bin_widths_truncated,
            kde_eval_points=kde_eval_points,
            kde_bandwidth_s=kde_bandwidth_s,
            truncate_rt_wrt_stim_s=truncate_rt_wrt_stim_s,
            reflect_kde_at_zero=reflect_kde_at_zero,
            slope_window_n_points=slope_window_n_points,
            slope_onset_fraction_of_peak=slope_onset_fraction_of_peak,
            baseline_window_end_s=baseline_window_end_s,
            baseline_sigma_multiplier=baseline_sigma_multiplier,
            baseline_min_consecutive_duration_s=baseline_min_consecutive_duration_s,
        )
        payload["n_rows"] = int(len(condition_df))
        condition_payload[(int(abl), int(abs_ild))] = payload
        combined_density_max = max(
            combined_density_max,
            float(np.max(payload["data_density_truncated"])),
            float(np.max(payload["kde_density_truncated"])),
        )
        if len(payload["slope_values_per_ms"]) > 0:
            combined_slope_abs_max = max(
                combined_slope_abs_max,
                float(np.max(np.abs(payload["slope_values_per_ms"]))),
            )

        peak_slope_text = "nan" if np.isnan(payload["peak_slope_per_ms"]) else f"{payload['peak_slope_per_ms']:.4f}"
        peak_time_text = "nan" if np.isnan(payload["peak_slope_time_s"]) else f"{payload['peak_slope_time_s'] * 1e3:.1f} ms"
        onset_time_text = "nan" if np.isnan(payload["slope_onset_time_s"]) else f"{payload['slope_onset_time_s'] * 1e3:.1f} ms"
        baseline_threshold_text = (
            "nan"
            if np.isnan(payload["baseline_slope_threshold_per_ms"])
            else f"{payload['baseline_slope_threshold_per_ms']:.4f}"
        )
        baseline_onset_text = (
            "nan"
            if np.isnan(payload["baseline_significant_onset_time_s"])
            else f"{payload['baseline_significant_onset_time_s'] * 1e3:.1f} ms"
        )
        print(
            f"ABL={int(abl)}, abs_ILD={int(abs_ild)}: "
            f"n_rows={payload['n_rows']}, "
            f"n_truncated={payload['n_truncated_points']}, "
            f"hist_area={payload['data_area_truncated']:.6f}, "
            f"kde_area={payload['kde_area_truncated']:.6f}, "
            f"peak_slope_per_ms={peak_slope_text}, "
            f"peak_slope_time={peak_time_text}, "
            f"slope_onset_time={onset_time_text}, "
            f"baseline_threshold_per_ms={baseline_threshold_text}, "
            f"baseline_onset_time={baseline_onset_text}"
        )

manual_delay_s_by_condition = {}
if manual_pick_csv_path.exists():
    manual_pick_df = pd.read_csv(manual_pick_csv_path)
    for abl in supported_abl_values:
        for abs_ild in supported_abs_ild_values:
            condition_key = (int(abl), int(abs_ild))
            manual_delay_s_by_condition[condition_key] = np.nan

    for _, row in manual_pick_df.iterrows():
        condition_key = (int(row["ABL"]), int(row["abs_ILD"]))
        if condition_key not in manual_delay_s_by_condition:
            continue
        if "manual_delay_s" in row and pd.notna(row["manual_delay_s"]):
            manual_delay_s_by_condition[condition_key] = float(row["manual_delay_s"])
        elif "manual_delay_ms" in row and pd.notna(row["manual_delay_ms"]):
            manual_delay_s_by_condition[condition_key] = float(row["manual_delay_ms"]) * 1e-3

    n_loaded_manual_picks = sum(
        int(np.isfinite(delay_s)) for delay_s in manual_delay_s_by_condition.values()
    )
    print(
        f"Loaded manual picks from {manual_pick_csv_path}: "
        f"{n_loaded_manual_picks}/{len(manual_delay_s_by_condition)} conditions."
    )
else:
    print(f"Manual pick CSV not found for overlay plot: {manual_pick_csv_path}")


# %%
############ Plot 3 x 5 truncated RTDs with smoother KDE overlay ############
overlay_plot_base = (
    f"estimate_delays_from_RTDs_rise_batch_{batch_name}_aggregate_ledoff_"
    f"truncated_{truncate_label_tag}_rtwrtstim_by_ABL_abs_ILD_kde_overlay_slope_explore"
)
overlay_plot_output_base = output_dir / overlay_plot_base

fig_overlay, axes_overlay = plt.subplots(
    len(supported_abl_values),
    len(supported_abs_ild_values),
    figsize=(panel_width * len(supported_abs_ild_values), panel_height * len(supported_abl_values)),
    sharex=True,
    sharey=True,
    squeeze=False,
)

for row_idx, abl in enumerate(supported_abl_values):
    for col_idx, abs_ild in enumerate(supported_abs_ild_values):
        ax = axes_overlay[row_idx, col_idx]
        color = abl_colors[int(abl)]
        payload = condition_payload[(int(abl), int(abs_ild))]

        ax.step(
            data_bin_centers_truncated,
            payload["data_density_truncated"],
            where="mid",
            lw=1.8,
            color=color,
            alpha=0.9,
            label="Data",
        )
        ax.plot(
            kde_eval_points,
            payload["kde_density_truncated"],
            color="black",
            linewidth=1.8,
            alpha=0.95,
            label=f"Epanechnikov KDE (h={kde_bandwidth_s * 1e3:.0f} ms)",
        )
        if np.isfinite(payload["slope_onset_time_s"]):
            ax.axvline(
                x=payload["slope_onset_time_s"],
                color="black",
                linestyle="-",
                linewidth=1.5,
                label=f"10% peak {payload['slope_onset_time_s'] * 1e3:.1f} ms",
            )
        if np.isfinite(payload["baseline_significant_onset_time_s"]):
            ax.axvline(
                x=payload["baseline_significant_onset_time_s"],
                color="darkmagenta",
                linestyle="-",
                linewidth=1.5,
                label=f"Baseline+{baseline_sigma_multiplier:.0f}σ {payload['baseline_significant_onset_time_s'] * 1e3:.1f} ms",
            )

        if row_idx == 0:
            ax.set_title(f"|ILD| = {int(abs_ild)}")
        if row_idx == len(supported_abl_values) - 1:
            ax.set_xlabel("RT - t_stim (s)")
        if col_idx == 0:
            ax.set_ylabel(f"Density\nABL = {int(abl)}")

        ax.set_xlim(0.0, truncate_rt_wrt_stim_s)
        ax.grid(alpha=0.2, linewidth=0.6)
        ax.legend(fontsize=6)

if combined_density_max > 0:
    axes_overlay[0, 0].set_ylim(0.0, combined_density_max * 1.1)

fig_overlay.suptitle(
    f"LED-OFF Aggregate RTD truncated {truncate_label_ms} with smoother Epanechnikov KDE by ABL x |ILD| ({batch_name})",
    y=1.02,
)
fig_overlay.tight_layout(rect=[0, 0, 1, 0.97])
fig_overlay.savefig(overlay_plot_output_base.with_suffix(".png"), dpi=png_dpi, bbox_inches="tight")
print(f"Saved smoother RTD KDE overlay plot (PNG): {overlay_plot_output_base.with_suffix('.png')}")


# %%
############ Plot 1 x 3 smoother RTD overlays by ABL ############
overlay_by_abl_plot_base = (
    f"estimate_delays_from_RTDs_rise_batch_{batch_name}_aggregate_ledoff_"
    f"truncated_{truncate_label_tag}_kde_overlay_by_ABL"
)
overlay_by_abl_plot_output_base = output_dir / overlay_by_abl_plot_base

fig_overlay_by_abl, axes_overlay_by_abl = plt.subplots(
    1,
    len(supported_abl_values),
    figsize=(20, 5),
    sharex=True,
    sharey=True,
    squeeze=False,
)

for col_idx, abl in enumerate(supported_abl_values):
    ax = axes_overlay_by_abl[0, col_idx]

    for abs_ild in supported_abs_ild_values:
        color = abs_ild_colors[int(abs_ild)]
        payload = condition_payload[(int(abl), int(abs_ild))]
        manual_delay_s = manual_delay_s_by_condition.get(
            (int(abl), int(abs_ild)),
            payload["baseline_significant_onset_time_s"],
        )
        legend_label = f"|ILD| = {int(abs_ild)}"
        if np.isfinite(manual_delay_s):
            legend_label = f"|ILD| = {int(abs_ild)} ({manual_delay_s * 1e3:.1f} ms)"

        ax.plot(
            kde_eval_points,
            payload["kde_density_truncated"],
            color=color,
            linewidth=2.0,
            alpha=0.95,
            label=legend_label,
        )
        if np.isfinite(manual_delay_s):
            ax.axvline(
                x=manual_delay_s,
                color=color,
                linestyle="-",
                linewidth=1.6,
                alpha=0.9,
            )
    ax.set_title(f"ABL = {int(abl)}")
    ax.set_xlabel("RT - t_stim (s)")
    if col_idx == 0:
        ax.set_ylabel("KDE Density")
    ax.set_xlim(0.0, truncate_rt_wrt_stim_s)
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.legend(fontsize=7)

if combined_density_max > 0:
    axes_overlay_by_abl[0, 0].set_ylim(0.0, combined_density_max * 1.1)

fig_overlay_by_abl.suptitle(
    f"LED-OFF smoothed RTD overlays by ABL with manual delay picks ({batch_name})",
    y=1.02,
)
fig_overlay_by_abl.tight_layout(rect=[0, 0, 1, 0.97])
fig_overlay_by_abl.savefig(overlay_by_abl_plot_output_base.with_suffix(".png"), dpi=png_dpi, bbox_inches="tight")
print(f"Saved 1x3 ABL KDE overlay plot (PNG): {overlay_by_abl_plot_output_base.with_suffix('.png')}")


# %%
############ Plot 3 x 5 slope curves from smoothed KDE ############
slope_plot_base = (
    f"estimate_delays_from_RTDs_rise_batch_{batch_name}_aggregate_ledoff_"
    f"truncated_{truncate_label_tag}_kde_slope_by_ABL_abs_ILD"
)
slope_plot_output_base = output_dir / slope_plot_base

fig_slope, axes_slope = plt.subplots(
    len(supported_abl_values),
    len(supported_abs_ild_values),
    figsize=(panel_width * len(supported_abs_ild_values), panel_height * len(supported_abl_values)),
    sharex=True,
    sharey=True,
    squeeze=False,
)

for row_idx, abl in enumerate(supported_abl_values):
    for col_idx, abs_ild in enumerate(supported_abs_ild_values):
        ax = axes_slope[row_idx, col_idx]
        color = abl_colors[int(abl)]
        payload = condition_payload[(int(abl), int(abs_ild))]

        ax.axhline(0.0, color="0.35", linewidth=1.0, alpha=0.8)
        if len(payload["slope_values_per_ms"]) > 0:
            ax.plot(
                payload["slope_times_s"],
                payload["slope_values_per_ms"],
                color=color,
                linewidth=1.8,
                label=f"Slope ({slope_window_n_points} pts)",
            )
            if np.isfinite(payload["slope_onset_time_s"]):
                ax.axvline(
                    x=payload["slope_onset_time_s"],
                    color="black",
                    linestyle="-",
                    linewidth=1.5,
                    label=f"10% peak {payload['slope_onset_time_s'] * 1e3:.1f} ms",
                )
            if np.isfinite(payload["baseline_significant_onset_time_s"]):
                ax.axvline(
                    x=payload["baseline_significant_onset_time_s"],
                    color="darkmagenta",
                    linestyle="-",
                    linewidth=1.5,
                    label=f"Baseline+{baseline_sigma_multiplier:.0f}σ {payload['baseline_significant_onset_time_s'] * 1e3:.1f} ms",
                )

        if row_idx == 0:
            ax.set_title(f"|ILD| = {int(abs_ild)}")
        if row_idx == len(supported_abl_values) - 1:
            ax.set_xlabel("RT - t_stim (s)")
        if col_idx == 0:
            ax.set_ylabel(f"KDE slope\n(density/ms)\nABL = {int(abl)}")

        ax.set_xlim(0.0, truncate_rt_wrt_stim_s)
        ax.grid(alpha=0.2, linewidth=0.6)
        ax.legend(fontsize=6)

if combined_slope_abs_max > 0:
    axes_slope[0, 0].set_ylim(-combined_slope_abs_max * 1.1, combined_slope_abs_max * 1.1)

fig_slope.suptitle(
    f"LED-OFF Aggregate smoothed RTD slope ({slope_window_n_points}-point window) by ABL x |ILD| ({batch_name})",
    y=1.02,
)
fig_slope.tight_layout(rect=[0, 0, 1, 0.97])
fig_slope.savefig(slope_plot_output_base.with_suffix(".png"), dpi=png_dpi, bbox_inches="tight")
print(f"Saved KDE slope plot (PNG): {slope_plot_output_base.with_suffix('.png')}")


# %%
############ Plot 5 x 3 transposed slope curves from smoothed KDE ############
slope_plot_transposed_base = (
    f"estimate_delays_from_RTDs_rise_batch_{batch_name}_aggregate_ledoff_"
    f"truncated_{truncate_label_tag}_kde_slope_by_abs_ILD_ABL"
)
slope_plot_transposed_output_base = output_dir / slope_plot_transposed_base

fig_slope_transposed, axes_slope_transposed = plt.subplots(
    len(supported_abs_ild_values),
    len(supported_abl_values),
    figsize=(panel_width * len(supported_abl_values), panel_height * len(supported_abs_ild_values)),
    sharex=True,
    sharey=True,
    squeeze=False,
)

for row_idx, abs_ild in enumerate(supported_abs_ild_values):
    for col_idx, abl in enumerate(supported_abl_values):
        ax = axes_slope_transposed[row_idx, col_idx]
        color = abl_colors[int(abl)]
        payload = condition_payload[(int(abl), int(abs_ild))]

        ax.axhline(0.0, color="0.35", linewidth=1.0, alpha=0.8)
        if len(payload["slope_values_per_ms"]) > 0:
            ax.plot(
                payload["slope_times_s"],
                payload["slope_values_per_ms"],
                color=color,
                linewidth=1.8,
                label=f"Slope ({slope_window_n_points} pts)",
            )
            if np.isfinite(payload["slope_onset_time_s"]):
                ax.axvline(
                    x=payload["slope_onset_time_s"],
                    color="black",
                    linestyle="-",
                    linewidth=1.5,
                    label=f"10% peak {payload['slope_onset_time_s'] * 1e3:.1f} ms",
                )
            if np.isfinite(payload["baseline_significant_onset_time_s"]):
                ax.axvline(
                    x=payload["baseline_significant_onset_time_s"],
                    color="darkmagenta",
                    linestyle="-",
                    linewidth=1.5,
                    label=f"Baseline+{baseline_sigma_multiplier:.0f}σ {payload['baseline_significant_onset_time_s'] * 1e3:.1f} ms",
                )

        if row_idx == 0:
            ax.set_title(f"ABL = {int(abl)}")
        if row_idx == len(supported_abs_ild_values) - 1:
            ax.set_xlabel("RT - t_stim (s)")
        if col_idx == 0:
            ax.set_ylabel(f"KDE slope\n(density/ms)\n|ILD| = {int(abs_ild)}")

        ax.set_xlim(0.0, truncate_rt_wrt_stim_s)
        ax.grid(alpha=0.2, linewidth=0.6)
        ax.legend(fontsize=6)

if combined_slope_abs_max > 0:
    axes_slope_transposed[0, 0].set_ylim(-combined_slope_abs_max * 1.1, combined_slope_abs_max * 1.1)

fig_slope_transposed.suptitle(
    f"LED-OFF Aggregate smoothed RTD slope ({slope_window_n_points}-point window) by |ILD| x ABL ({batch_name})",
    y=1.02,
)
fig_slope_transposed.tight_layout(rect=[0, 0, 1, 0.97])
fig_slope_transposed.savefig(
    slope_plot_transposed_output_base.with_suffix(".png"),
    dpi=png_dpi,
    bbox_inches="tight",
)
print(f"Saved transposed KDE slope plot (PNG): {slope_plot_transposed_output_base.with_suffix('.png')}")

slope_onset_table_ms = pd.DataFrame(
    index=supported_abl_values,
    columns=supported_abs_ild_values,
    dtype=float,
)
for abl in supported_abl_values:
    for abs_ild in supported_abs_ild_values:
        slope_onset_time_s = condition_payload[(int(abl), int(abs_ild))]["slope_onset_time_s"]
        slope_onset_table_ms.loc[int(abl), int(abs_ild)] = (
            np.nan if np.isnan(slope_onset_time_s) else float(slope_onset_time_s) * 1e3
        )

slope_onset_table_display = slope_onset_table_ms.apply(
    lambda column: column.map(
        lambda value: "nan" if pd.isna(value) else f"{float(value):.1f}"
    )
)
slope_onset_table_display.index.name = "ABL"
slope_onset_table_display.columns.name = "abs_ILD"
print(f"Slope-onset estimates (ms) by ABL x abs_ILD at {int(round(slope_onset_fraction_of_peak * 100))}% of peak slope:")
print(slope_onset_table_display.to_string())

baseline_onset_table_ms = pd.DataFrame(
    index=supported_abl_values,
    columns=supported_abs_ild_values,
    dtype=float,
)
for abl in supported_abl_values:
    for abs_ild in supported_abs_ild_values:
        baseline_onset_time_s = condition_payload[(int(abl), int(abs_ild))]["baseline_significant_onset_time_s"]
        baseline_onset_table_ms.loc[int(abl), int(abs_ild)] = (
            np.nan if np.isnan(baseline_onset_time_s) else float(baseline_onset_time_s) * 1e3
        )

baseline_onset_table_display = baseline_onset_table_ms.apply(
    lambda column: column.map(
        lambda value: "nan" if pd.isna(value) else f"{float(value):.1f}"
    )
)
baseline_onset_csv_path = (
    output_dir
    / f"baseline_significant_slope_onset_mean_plus_{int(round(baseline_sigma_multiplier))}sigma_ms_by_ABL_abs_ILD.csv"
)
baseline_onset_table_ms_rounded = baseline_onset_table_ms.round(1)
baseline_onset_table_ms_rounded.index.name = "ABL"
baseline_onset_table_ms_rounded.columns.name = "abs_ILD"
baseline_onset_table_ms_rounded.to_csv(baseline_onset_csv_path)
baseline_onset_table_display.index.name = "ABL"
baseline_onset_table_display.columns.name = "abs_ILD"
print(
    "Baseline-significant slope onset estimates (ms) by ABL x abs_ILD "
    f"(baseline <= {baseline_window_end_s * 1e3:.0f} ms, threshold=mean+{baseline_sigma_multiplier:.0f}σ):"
)
print(baseline_onset_table_display.to_string())
print(f"Saved baseline-significant delay table CSV: {baseline_onset_csv_path}")

if show_plot:
    plt.show()
else:
    plt.close(fig_overlay)
    plt.close(fig_overlay_by_abl)
    plt.close(fig_slope)
    plt.close(fig_slope_transposed)

# %%
