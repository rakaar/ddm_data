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
abl_colors = {
    20: "tab:blue",
    40: "tab:orange",
    60: "tab:green",
}

max_rtwrtstim_for_fit = 1.0
truncate_rt_wrt_stim_s = 0.115
data_bin_size_s_truncated = 5e-3
rise_estimation_bin_size_s = 1e-3
rise_moving_average_window_s = 10e-3
rise_diff_threshold = 0.02
rise_density_floor = 0.4
rise_min_consecutive_bins = 2

panel_figsize = (15.0, 4.8)
png_dpi = 300
show_plot = SHOW_PLOT

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
led_data_csv_path = REPO_ROOT / "out_LED.csv"
output_dir = SCRIPT_DIR / "estimate_delays_from_RTDs_rise"


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

if len(plot_df) == 0:
    raise ValueError("No LED-OFF trials found after filtering.")

observed_abl_values = np.sort(plot_df["ABL"].dropna().astype(float).unique())
if len(observed_abl_values) == 0:
    raise ValueError("No ABL values found in LED-OFF RTD dataset.")

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

abl_counts = (
    plot_df["ABL"]
    .astype(float)
    .round()
    .astype(int)
    .value_counts()
    .sort_index()
    .to_dict()
)
abl_counts = {int(key): int(value) for key, value in abl_counts.items()}

print("Rebuilt LED-OFF aggregate RTD dataset:")
print(f"  batch_name={batch_name}")
print(f"  Total LED-OFF filtered trials (valid+aborts): {len(df_valid_and_aborts)}")
print(f"  LED-OFF trials used for RTD plot (valid+aborts): {len(plot_df)}")
print(f"  Supported ABL values in plot dataset: {observed_abl_values.tolist()}")
print(f"  Plot trial counts by ABL: {abl_counts}")


# %%
############ Build per-ABL truncated RTD payload ############
truncate_rt_wrt_stim_ms = int(round(float(truncate_rt_wrt_stim_s) * 1e3))
truncate_label_ms = f"{truncate_rt_wrt_stim_ms} ms"
truncate_label_tag = f"{truncate_rt_wrt_stim_ms}ms"
hist_edges_truncated = np.linspace(
    0.0,
    truncate_rt_wrt_stim_s,
    int(round(truncate_rt_wrt_stim_s / data_bin_size_s_truncated)) + 1,
)
data_bin_widths_truncated = np.diff(hist_edges_truncated)
data_bin_centers_truncated = hist_edges_truncated[:-1] + 0.5 * data_bin_widths_truncated
rise_hist_edges = np.arange(
    0.0,
    truncate_rt_wrt_stim_s + rise_estimation_bin_size_s,
    rise_estimation_bin_size_s,
)
rise_bin_centers = rise_hist_edges[:-1] + 0.5 * rise_estimation_bin_size_s
rise_window_bins = int(round(rise_moving_average_window_s / rise_estimation_bin_size_s))
rise_kernel = np.ones(rise_window_bins, dtype=float) / rise_window_bins

abl_panel_payload: dict[int, dict[str, np.ndarray | int | float]] = {}
combined_ax_max = 0.0
abl_values_float = plot_df["ABL"].astype(float).to_numpy()

for abl in supported_abl_values:
    df_abl = plot_df[np.isclose(abl_values_float, float(abl))].copy()
    if len(df_abl) == 0:
        raise ValueError(f"No plot rows found for ABL={abl}.")

    data_rtwrtstim_abl = df_abl["RTwrtStim"].to_numpy(dtype=np.float64)
    data_rtwrtstim_abl_truncated = data_rtwrtstim_abl[
        (data_rtwrtstim_abl >= 0.0) & (data_rtwrtstim_abl <= truncate_rt_wrt_stim_s)
    ]
    if len(data_rtwrtstim_abl_truncated) == 0:
        raise ValueError(f"No truncated data points remain for ABL={abl}.")

    data_counts_abl_truncated, _ = np.histogram(
        data_rtwrtstim_abl_truncated,
        bins=hist_edges_truncated,
        density=False,
    )
    n_total_abl = len(df_abl)
    data_density_abl_truncated = data_counts_abl_truncated / (n_total_abl * data_bin_widths_truncated)
    data_area_abl_truncated = float(np.sum(data_density_abl_truncated * data_bin_widths_truncated))

    rise_counts_abl, _ = np.histogram(
        data_rtwrtstim_abl_truncated,
        bins=rise_hist_edges,
        density=False,
    )
    rise_density_abl = rise_counts_abl / (n_total_abl * rise_estimation_bin_size_s)
    rise_density_smoothed = np.convolve(rise_density_abl, rise_kernel, mode="same")
    rise_density_diff = np.diff(rise_density_smoothed)
    rise_candidate_mask = (
        (rise_density_diff > rise_diff_threshold)
        & (rise_density_smoothed[1:] > rise_density_floor)
    )

    rise_estimate_s = np.nan
    consecutive_count = 0
    for idx, is_candidate in enumerate(rise_candidate_mask):
        consecutive_count = consecutive_count + 1 if is_candidate else 0
        if consecutive_count >= rise_min_consecutive_bins:
            rise_estimate_s = rise_bin_centers[idx - rise_min_consecutive_bins + 2]
            break

    combined_ax_max = max(combined_ax_max, float(np.max(data_density_abl_truncated)))

    abl_panel_payload[int(abl)] = {
        "n_rows": int(n_total_abl),
        "n_truncated_points": int(len(data_rtwrtstim_abl_truncated)),
        "data_density_truncated": data_density_abl_truncated,
        "data_area_truncated": data_area_abl_truncated,
        "rise_estimate_s": rise_estimate_s,
    }

    rise_text = "nan" if np.isnan(rise_estimate_s) else f"{rise_estimate_s * 1e3:.1f} ms"
    print(
        f"ABL={abl}: n_rows={len(df_abl)}, "
        f"n_truncated={len(data_rtwrtstim_abl_truncated)}, "
        f"truncated_area={data_area_abl_truncated:.6f}, "
        f"rise_estimate={rise_text}"
    )


# %%
############ Plot 1 x 3 truncated RTDs by ABL ############
plot_base = (
    f"estimate_delays_from_RTDs_rise_batch_{batch_name}_aggregate_ledoff_"
    f"truncated_{truncate_label_tag}_rtwrtstim_by_ABL"
)
plot_output_base = output_dir / plot_base

fig, axes = plt.subplots(
    1,
    len(supported_abl_values),
    figsize=panel_figsize,
    sharex=True,
    sharey=True,
)

if len(supported_abl_values) == 1:
    axes = [axes]

for ax, abl in zip(axes, supported_abl_values):
    abl_int = int(abl)
    color = abl_colors[abl_int]
    payload_abl = abl_panel_payload[abl_int]

    ax.step(
        data_bin_centers_truncated,
        payload_abl["data_density_truncated"],
        where="mid",
        lw=2.0,
        color=color,
        alpha=0.85,
        label="Data",
    )
    ax.axvline(
        x=truncate_rt_wrt_stim_s,
        color="crimson",
        linestyle="--",
        linewidth=1.4,
        label=truncate_label_ms,
    )
    if np.isfinite(payload_abl["rise_estimate_s"]):
        rise_estimate_ms = payload_abl["rise_estimate_s"] * 1e3
        ax.axvline(
            x=payload_abl["rise_estimate_s"],
            color="black",
            linestyle=":",
            linewidth=1.8,
            label=f"Rise {rise_estimate_ms:.1f} ms",
        )
        print(f'ABL={abl_int}: Rise estimate = {rise_estimate_ms:.1f} ms')
    ax.set_title(
        f"ABL {abl_int}  (n={payload_abl['n_rows']}, trunc={payload_abl['n_truncated_points']})"
    )
    ax.set_xlim(0.0, truncate_rt_wrt_stim_s)
    ax.set_xlabel("RT - t_stim (s)")
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.legend(fontsize=9)

axes[0].set_ylabel("Density")
if combined_ax_max > 0:
    axes[0].set_ylim(0.0, combined_ax_max * 1.1)

abl_counts_str = ", ".join(
    f"ABL{int(abl)}={abl_panel_payload[int(abl)]['n_rows']}"
    for abl in supported_abl_values
)
fig.suptitle(
    f"LED-OFF Aggregate RTD truncated {truncate_label_ms} by ABL ({batch_name})\n"
    f"Trials: {abl_counts_str}",
    y=1.04,
)
fig.tight_layout(rect=[0, 0, 1, 0.97])

fig.savefig(plot_output_base.with_suffix(".png"), dpi=png_dpi, bbox_inches="tight")
print(f"Saved RTD rise plot (PNG): {plot_output_base.with_suffix('.png')}")

if show_plot:
    plt.show()
else:
    plt.close(fig)
