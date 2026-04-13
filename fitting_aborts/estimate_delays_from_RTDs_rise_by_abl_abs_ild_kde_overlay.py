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

max_rtwrtstim_for_fit = 1.0
truncate_rt_wrt_stim_s = 0.140
data_bin_size_s_truncated = 5e-3
kde_bandwidth_s = 12e-3
kde_eval_step_s = 5e-4
reflect_kde_at_zero = True
rise_threshold_peak_fraction = 0.18
rise_density_floor = 0.4
rise_min_consecutive_duration_s = 5e-3

panel_width = 3.8
panel_height = 3.0
png_dpi = 300
show_plot = SHOW_PLOT

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
led_data_csv_path = REPO_ROOT / "out_LED.csv"
output_dir = SCRIPT_DIR / "estimate_delays_from_RTDs_rise_by_abl_abs_ild_kde_overlay"


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


def estimate_rise_time_from_smoothed_curve(
    smoothed_density: np.ndarray,
    eval_points: np.ndarray,
    rise_threshold_peak_fraction: float,
    rise_density_floor: float,
    rise_min_consecutive_duration_s: float,
    eval_step_s: float,
):
    if len(smoothed_density) == 0:
        return np.nan, np.nan

    peak_density = float(np.max(smoothed_density))
    if not np.isfinite(peak_density) or peak_density <= 0:
        return np.nan, np.nan

    rise_threshold = max(rise_density_floor, rise_threshold_peak_fraction * peak_density)
    rise_min_consecutive_points = max(1, int(round(rise_min_consecutive_duration_s / eval_step_s)))
    rise_candidate_mask = smoothed_density >= rise_threshold

    rise_estimate_s = np.nan
    consecutive_count = 0
    for idx, is_candidate in enumerate(rise_candidate_mask):
        consecutive_count = consecutive_count + 1 if is_candidate else 0
        if consecutive_count >= rise_min_consecutive_points:
            rise_estimate_s = eval_points[idx - rise_min_consecutive_points + 1]
            break

    return rise_estimate_s, rise_threshold


def build_rtd_and_kde_payload(
    rt_values: np.ndarray,
    n_total_condition: int,
    hist_edges_truncated: np.ndarray,
    hist_bin_widths_truncated: np.ndarray,
    kde_eval_points: np.ndarray,
    kde_bandwidth_s: float,
    truncate_rt_wrt_stim_s: float,
    reflect_kde_at_zero: bool,
    rise_threshold_peak_fraction: float,
    rise_density_floor: float,
    rise_min_consecutive_duration_s: float,
    kde_eval_step_s: float,
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

    rise_estimate_s, rise_threshold = estimate_rise_time_from_smoothed_curve(
        smoothed_density=kde_density_truncated,
        eval_points=kde_eval_points,
        rise_threshold_peak_fraction=rise_threshold_peak_fraction,
        rise_density_floor=rise_density_floor,
        rise_min_consecutive_duration_s=rise_min_consecutive_duration_s,
        eval_step_s=kde_eval_step_s,
    )

    kde_area_truncated = float(np.trapz(kde_density_truncated, kde_eval_points))
    return {
        "n_truncated_points": int(len(truncated_rt_values)),
        "data_density_truncated": data_density_truncated,
        "data_area_truncated": data_area_truncated,
        "kde_density_truncated": kde_density_truncated,
        "kde_area_truncated": kde_area_truncated,
        "rise_estimate_s": rise_estimate_s,
        "rise_threshold": rise_threshold,
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

print("Rebuilt LED-OFF aggregate RTD dataset for KDE overlay by ABL x abs_ILD:")
print(f"  batch_name={batch_name}")
print(f"  Total LED-OFF filtered trials (valid+aborts): {len(df_valid_and_aborts)}")
print(f"  LED-OFF trials used for RTD plot (valid+aborts): {len(plot_df)}")
print(f"  Supported ABL values in plot dataset: {observed_abl_values.tolist()}")
print(f"  Supported abs_ILD values in plot dataset: {observed_abs_ild_values.tolist()}")


# %%
############ Build per-condition histogram + KDE payload ############
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

condition_payload = {}
combined_ax_max = 0.0

for abl in supported_abl_values:
    for abs_ild in supported_abs_ild_values:
        condition_df = plot_df[
            np.isclose(plot_df["ABL"], float(abl))
            & np.isclose(plot_df["abs_ILD"], float(abs_ild))
        ].copy()

        payload = build_rtd_and_kde_payload(
            rt_values=condition_df["RTwrtStim"].to_numpy(dtype=np.float64),
            n_total_condition=len(condition_df),
            hist_edges_truncated=hist_edges_truncated,
            hist_bin_widths_truncated=hist_bin_widths_truncated,
            kde_eval_points=kde_eval_points,
            kde_bandwidth_s=kde_bandwidth_s,
            truncate_rt_wrt_stim_s=truncate_rt_wrt_stim_s,
            reflect_kde_at_zero=reflect_kde_at_zero,
            rise_threshold_peak_fraction=rise_threshold_peak_fraction,
            rise_density_floor=rise_density_floor,
            rise_min_consecutive_duration_s=rise_min_consecutive_duration_s,
            kde_eval_step_s=kde_eval_step_s,
        )
        payload["n_rows"] = int(len(condition_df))
        condition_payload[(int(abl), int(abs_ild))] = payload
        combined_ax_max = max(
            combined_ax_max,
            float(np.max(payload["data_density_truncated"])),
            float(np.max(payload["kde_density_truncated"])),
        )

        rise_text = "nan" if np.isnan(payload["rise_estimate_s"]) else f"{payload['rise_estimate_s'] * 1e3:.1f} ms"
        print(
            f"ABL={int(abl)}, abs_ILD={int(abs_ild)}: "
            f"n_rows={payload['n_rows']}, "
            f"n_truncated={payload['n_truncated_points']}, "
            f"hist_area={payload['data_area_truncated']:.6f}, "
            f"kde_area={payload['kde_area_truncated']:.6f}, "
            f"kde_rise_estimate={rise_text}"
        )


# %%
############ Plot 3 x 5 truncated RTDs with KDE overlay ############
plot_base = (
    f"estimate_delays_from_RTDs_rise_batch_{batch_name}_aggregate_ledoff_"
    f"truncated_{truncate_label_tag}_rtwrtstim_by_ABL_abs_ILD_kde_overlay"
)
plot_output_base = output_dir / plot_base

fig, axes = plt.subplots(
    len(supported_abl_values),
    len(supported_abs_ild_values),
    figsize=(panel_width * len(supported_abs_ild_values), panel_height * len(supported_abl_values)),
    sharex=True,
    sharey=True,
    squeeze=False,
)

for row_idx, abl in enumerate(supported_abl_values):
    for col_idx, abs_ild in enumerate(supported_abs_ild_values):
        ax = axes[row_idx, col_idx]
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
        if np.isfinite(payload["rise_estimate_s"]):
            rise_estimate_ms = payload["rise_estimate_s"] * 1e3
            ax.axvline(
                x=payload["rise_estimate_s"],
                color="black",
                linestyle=":",
                linewidth=1.5,
                label=f"KDE rise {rise_estimate_ms:.1f} ms",
            )
        ax.axvline(
            x=truncate_rt_wrt_stim_s,
            color="crimson",
            linestyle="--",
            linewidth=1.2,
            label=truncate_label_ms,
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

if combined_ax_max > 0:
    axes[0, 0].set_ylim(0.0, combined_ax_max * 1.1)

fig.suptitle(
    f"LED-OFF Aggregate RTD truncated {truncate_label_ms} with Epanechnikov KDE by ABL x |ILD| ({batch_name})",
    y=1.02,
)
fig.tight_layout(rect=[0, 0, 1, 0.97])

fig.savefig(plot_output_base.with_suffix(".png"), dpi=png_dpi, bbox_inches="tight")
print(f"Saved RTD KDE overlay plot (PNG): {plot_output_base.with_suffix('.png')}")

delay_table_ms = pd.DataFrame(
    index=supported_abl_values,
    columns=supported_abs_ild_values,
    dtype=float,
)
for abl in supported_abl_values:
    for abs_ild in supported_abs_ild_values:
        rise_estimate_s = condition_payload[(int(abl), int(abs_ild))]["rise_estimate_s"]
        delay_table_ms.loc[int(abl), int(abs_ild)] = (
            np.nan if np.isnan(rise_estimate_s) else float(rise_estimate_s) * 1e3
        )

delay_table_display = delay_table_ms.apply(
    lambda column: column.map(
        lambda value: "nan" if pd.isna(value) else f"{float(value):.1f}"
    )
)
delay_table_display.index.name = "ABL"
delay_table_display.columns.name = "abs_ILD"
print("KDE delay estimates (ms) by ABL x abs_ILD:")
print(delay_table_display.to_string())

if show_plot:
    plt.show()
else:
    plt.close(fig)
