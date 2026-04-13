# %%
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
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
slope_window_n_points = 50
slope_onset_fraction_of_peak = 0.10
baseline_window_end_s = 40e-3
baseline_sigma_multiplier = 3.0
baseline_min_consecutive_duration_s = 5e-3

manual_fine_step_s = 1e-3
manual_coarse_step_s = 5e-3
show_auto_reference_lines_on_start = True
start_from_first_unpicked = True

show_plot = SHOW_PLOT

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
led_data_csv_path = REPO_ROOT / "out_LED.csv"
output_dir = SCRIPT_DIR / "manual_pick_delays_from_RTDs_by_abl_abs_ild"


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
            "n_rows": 0,
            "n_truncated_points": 0,
            "data_density_truncated": np.zeros(n_hist_bins, dtype=float),
            "kde_density_truncated": np.zeros_like(kde_eval_points, dtype=float),
            "slope_times_s": np.array([], dtype=float),
            "slope_values_per_ms": np.array([], dtype=float),
            "slope_onset_time_s": np.nan,
            "baseline_significant_onset_time_s": np.nan,
            "baseline_slope_threshold_per_ms": np.nan,
        }

    data_counts_truncated, _ = np.histogram(
        truncated_rt_values,
        bins=hist_edges_truncated,
        density=False,
    )
    data_density_truncated = data_counts_truncated / (n_total_condition * hist_bin_widths_truncated)

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

    slope_times_s, slope_values_per_s = compute_rolling_linear_slope(
        x_points=kde_eval_points,
        y_points=kde_density_truncated,
        window_n_points=slope_window_n_points,
    )
    slope_values_per_ms = slope_values_per_s * 1e-3

    if len(slope_values_per_ms) == 0:
        slope_onset_time_s = np.nan
        baseline_significant_onset_time_s = np.nan
        baseline_slope_threshold_per_ms = np.nan
    else:
        peak_slope_per_ms = float(np.max(slope_values_per_ms))
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

            baseline_significant_onset_time_s = np.nan
            above_baseline_threshold = slope_values_per_ms >= baseline_slope_threshold_per_ms
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
            baseline_slope_threshold_per_ms = np.nan
            baseline_significant_onset_time_s = np.nan

    return {
        "n_rows": int(n_total_condition),
        "n_truncated_points": int(len(truncated_rt_values)),
        "data_density_truncated": data_density_truncated,
        "kde_density_truncated": kde_density_truncated,
        "slope_times_s": slope_times_s,
        "slope_values_per_ms": slope_values_per_ms,
        "slope_onset_time_s": slope_onset_time_s,
        "baseline_significant_onset_time_s": baseline_significant_onset_time_s,
        "baseline_slope_threshold_per_ms": baseline_slope_threshold_per_ms,
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

print("Loaded LED-OFF RTD dataset for manual delay picking:")
print(f"  batch_name={batch_name}")
print(f"  LED-OFF trials used for picker (valid+aborts): {len(plot_df)}")


# %%
############ Build condition payloads and load saved picks ############
truncate_rt_wrt_stim_ms = int(round(float(truncate_rt_wrt_stim_s) * 1e3))
truncate_label_tag = f"{truncate_rt_wrt_stim_ms}ms"
hist_edges_truncated = np.linspace(
    0.0,
    truncate_rt_wrt_stim_s,
    int(round(truncate_rt_wrt_stim_s / data_bin_size_s_truncated)) + 1,
)
hist_bin_widths_truncated = np.diff(hist_edges_truncated)
data_bin_centers_truncated = hist_edges_truncated[:-1] + 0.5 * hist_bin_widths_truncated
kde_eval_points = np.arange(0.0, truncate_rt_wrt_stim_s + kde_eval_step_s, kde_eval_step_s)

condition_keys = [
    (int(abl), int(abs_ild))
    for abl in supported_abl_values
    for abs_ild in supported_abs_ild_values
]
condition_payload = {}
for abl, abs_ild in condition_keys:
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
    condition_payload[(abl, abs_ild)] = payload

manual_pick_csv_path = (
    output_dir
    / f"manual_delay_picks_batch_{batch_name}_truncated_{truncate_label_tag}_by_ABL_abs_ILD.csv"
)
manual_pick_s_by_condition = {condition_key: np.nan for condition_key in condition_keys}

if manual_pick_csv_path.exists():
    saved_pick_df = pd.read_csv(manual_pick_csv_path)
    for _, row in saved_pick_df.iterrows():
        condition_key = (int(row["ABL"]), int(row["abs_ILD"]))
        if condition_key not in manual_pick_s_by_condition:
            continue
        if "manual_delay_s" in row and pd.notna(row["manual_delay_s"]):
            manual_pick_s_by_condition[condition_key] = float(row["manual_delay_s"])
        elif "manual_delay_ms" in row and pd.notna(row["manual_delay_ms"]):
            manual_pick_s_by_condition[condition_key] = float(row["manual_delay_ms"]) * 1e-3

current_condition_index = 0
if start_from_first_unpicked:
    for idx, condition_key in enumerate(condition_keys):
        if np.isnan(manual_pick_s_by_condition[condition_key]):
            current_condition_index = idx
            break

print(f"Manual pick CSV path: {manual_pick_csv_path}")
print(
    "Existing manual picks loaded: "
    f"{sum(int(np.isfinite(value)) for value in manual_pick_s_by_condition.values())}/{len(condition_keys)}"
)


# %%
############ Manual picker UI ############
fig = plt.figure(figsize=(10.5, 8.3))
grid = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.3], hspace=0.18)
ax_rtd = fig.add_subplot(grid[0, 0])
ax_slope = fig.add_subplot(grid[1, 0], sharex=ax_rtd)
fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.20)

button_prev_ax = fig.add_axes([0.10, 0.07, 0.10, 0.05])
button_next_ax = fig.add_axes([0.22, 0.07, 0.10, 0.05])
button_clear_ax = fig.add_axes([0.34, 0.07, 0.10, 0.05])
button_auto_ax = fig.add_axes([0.46, 0.07, 0.12, 0.05])
button_save_ax = fig.add_axes([0.60, 0.07, 0.10, 0.05])
button_quit_ax = fig.add_axes([0.72, 0.07, 0.10, 0.05])

button_prev = Button(button_prev_ax, "Prev [p]")
button_next = Button(button_next_ax, "Next [n]")
button_clear = Button(button_clear_ax, "Clear [r]")
button_auto = Button(button_auto_ax, "Auto [a]")
button_save = Button(button_save_ax, "Save [s]")
button_quit = Button(button_quit_ax, "Quit [q]")

status_text = fig.text(0.08, 0.14, "", fontsize=10)
help_text = fig.text(
    0.08,
    0.015,
    "Click in either panel to place the manual delay. Drag to adjust. "
    "Left/Right = +/-1 ms, Shift+Left/Right = +/-5 ms, n/p = next/prev, r = clear, a = toggle auto refs, s = save, q = quit.",
    fontsize=9,
)

state = {
    "condition_index": current_condition_index,
    "show_auto_lines": show_auto_reference_lines_on_start,
    "dragging_line": False,
    "manual_pick_s_by_condition": manual_pick_s_by_condition,
    "manual_line_artists": [],
    "last_saved_message": "Not saved in this session yet.",
}


def clamp_delay_s(delay_s: float) -> float:
    return float(np.clip(delay_s, 0.0, truncate_rt_wrt_stim_s))


def current_condition_key():
    return condition_keys[state["condition_index"]]


def current_payload():
    return condition_payload[current_condition_key()]


def default_pick_s():
    payload = current_payload()
    if np.isfinite(payload["baseline_significant_onset_time_s"]):
        return float(payload["baseline_significant_onset_time_s"])
    if np.isfinite(payload["slope_onset_time_s"]):
        return float(payload["slope_onset_time_s"])
    return float(0.5 * truncate_rt_wrt_stim_s)


def current_manual_pick_s():
    return state["manual_pick_s_by_condition"][current_condition_key()]


def update_manual_line_artists():
    manual_pick_s = current_manual_pick_s()
    for line_artist in state["manual_line_artists"]:
        if np.isfinite(manual_pick_s):
            line_artist.set_xdata([manual_pick_s, manual_pick_s])
            line_artist.set_visible(True)
        else:
            line_artist.set_visible(False)


def save_manual_picks():
    save_timestamp_utc = datetime.now(timezone.utc).isoformat()
    save_rows = []
    for order_idx, (abl, abs_ild) in enumerate(condition_keys, start=1):
        payload = condition_payload[(abl, abs_ild)]
        manual_delay_s = state["manual_pick_s_by_condition"][(abl, abs_ild)]
        save_rows.append(
            {
                "order_idx": order_idx,
                "batch_name": batch_name,
                "ABL": abl,
                "abs_ILD": abs_ild,
                "n_rows": payload["n_rows"],
                "n_truncated_points": payload["n_truncated_points"],
                "auto_10pct_peak_ms": (
                    np.nan
                    if np.isnan(payload["slope_onset_time_s"])
                    else float(payload["slope_onset_time_s"]) * 1e3
                ),
                "auto_baseline_ms": (
                    np.nan
                    if np.isnan(payload["baseline_significant_onset_time_s"])
                    else float(payload["baseline_significant_onset_time_s"]) * 1e3
                ),
                "manual_delay_s": manual_delay_s,
                "manual_delay_ms": (
                    np.nan if np.isnan(manual_delay_s) else float(manual_delay_s) * 1e3
                ),
                "updated_utc": save_timestamp_utc,
            }
        )

    save_df = pd.DataFrame(save_rows)
    manual_pick_tmp_csv_path = manual_pick_csv_path.with_suffix(".tmp.csv")
    save_df.to_csv(manual_pick_tmp_csv_path, index=False)
    manual_pick_tmp_csv_path.replace(manual_pick_csv_path)
    n_completed = sum(
        int(np.isfinite(delay_s)) for delay_s in state["manual_pick_s_by_condition"].values()
    )
    state["last_saved_message"] = (
        f"Saved {n_completed}/{len(condition_keys)} picks to {manual_pick_csv_path.name} at {save_timestamp_utc}"
    )


def refresh_status_text():
    condition_key = current_condition_key()
    payload = current_payload()
    manual_pick_s = current_manual_pick_s()
    n_completed = sum(
        int(np.isfinite(delay_s)) for delay_s in state["manual_pick_s_by_condition"].values()
    )
    manual_text = "manual = not set"
    if np.isfinite(manual_pick_s):
        manual_text = f"manual = {manual_pick_s * 1e3:.1f} ms"
    auto_10pct_text = "10% peak = nan"
    if np.isfinite(payload["slope_onset_time_s"]):
        auto_10pct_text = f"10% peak = {payload['slope_onset_time_s'] * 1e3:.1f} ms"
    auto_baseline_text = "baseline+3sigma = nan"
    if np.isfinite(payload["baseline_significant_onset_time_s"]):
        auto_baseline_text = (
            f"baseline+3sigma = {payload['baseline_significant_onset_time_s'] * 1e3:.1f} ms"
        )

    fig.suptitle(
        f"Manual Delay Picker | {state['condition_index'] + 1}/{len(condition_keys)} | "
        f"ABL = {condition_key[0]} | |ILD| = {condition_key[1]}",
        y=0.965,
    )
    status_text.set_text(
        f"{manual_text} | {auto_10pct_text} | {auto_baseline_text} | "
        f"completed = {n_completed}/{len(condition_keys)} | auto refs = {'on' if state['show_auto_lines'] else 'off'}\n"
        f"{state['last_saved_message']}"
    )


def render_current_condition():
    condition_key = current_condition_key()
    abl, abs_ild = condition_key
    payload = current_payload()
    condition_color = abs_ild_colors[int(abs_ild)]

    ax_rtd.clear()
    ax_slope.clear()

    ax_rtd.step(
        data_bin_centers_truncated,
        payload["data_density_truncated"],
        where="mid",
        color=condition_color,
        linewidth=1.8,
        alpha=0.55,
        label="Data",
    )
    ax_rtd.plot(
        kde_eval_points,
        payload["kde_density_truncated"],
        color="black",
        linewidth=2.0,
        alpha=0.95,
        label=f"KDE (h={kde_bandwidth_s * 1e3:.0f} ms)",
    )
    ax_slope.axhline(0.0, color="0.35", linewidth=1.0, alpha=0.8)
    if len(payload["slope_values_per_ms"]) > 0:
        ax_slope.plot(
            payload["slope_times_s"],
            payload["slope_values_per_ms"],
            color=condition_color,
            linewidth=2.0,
            alpha=0.95,
            label=f"Slope ({slope_window_n_points} pts)",
        )

    if state["show_auto_lines"]:
        if np.isfinite(payload["slope_onset_time_s"]):
            ax_rtd.axvline(
                x=payload["slope_onset_time_s"],
                color="black",
                linestyle="--",
                linewidth=1.1,
                alpha=0.75,
                label=f"10% peak {payload['slope_onset_time_s'] * 1e3:.1f} ms",
            )
            ax_slope.axvline(
                x=payload["slope_onset_time_s"],
                color="black",
                linestyle="--",
                linewidth=1.1,
                alpha=0.75,
            )
        if np.isfinite(payload["baseline_significant_onset_time_s"]):
            ax_rtd.axvline(
                x=payload["baseline_significant_onset_time_s"],
                color="darkmagenta",
                linestyle="--",
                linewidth=1.1,
                alpha=0.75,
                label=f"Baseline+{baseline_sigma_multiplier:.0f}sigma {payload['baseline_significant_onset_time_s'] * 1e3:.1f} ms",
            )
            ax_slope.axvline(
                x=payload["baseline_significant_onset_time_s"],
                color="darkmagenta",
                linestyle="--",
                linewidth=1.1,
                alpha=0.75,
            )

    manual_pick_s = current_manual_pick_s()
    manual_pick_is_finite = np.isfinite(manual_pick_s)
    if not manual_pick_is_finite:
        manual_pick_s = default_pick_s()

    manual_line_rtd = ax_rtd.axvline(
        x=manual_pick_s,
        color="crimson",
        linestyle="-",
        linewidth=2.4,
        alpha=0.95,
        label=(
            f"Manual {manual_pick_s * 1e3:.1f} ms"
            if manual_pick_is_finite
            else "Manual pick"
        ),
    )
    manual_line_slope = ax_slope.axvline(
        x=manual_pick_s,
        color="crimson",
        linestyle="-",
        linewidth=2.4,
        alpha=0.95,
        label=(
            f"Manual {manual_pick_s * 1e3:.1f} ms"
            if manual_pick_is_finite
            else "Manual pick"
        ),
    )
    if not manual_pick_is_finite:
        manual_line_rtd.set_visible(False)
        manual_line_slope.set_visible(False)

    state["manual_line_artists"] = [manual_line_rtd, manual_line_slope]

    rtd_y_max = float(
        max(
            np.max(payload["data_density_truncated"]) if len(payload["data_density_truncated"]) > 0 else 0.0,
            np.max(payload["kde_density_truncated"]) if len(payload["kde_density_truncated"]) > 0 else 0.0,
        )
    )
    slope_abs_max = float(
        np.max(np.abs(payload["slope_values_per_ms"])) if len(payload["slope_values_per_ms"]) > 0 else 0.0
    )

    ax_rtd.set_ylabel("Density")
    ax_slope.set_ylabel("KDE slope\n(density/ms)")
    ax_slope.set_xlabel("RT - t_stim (s)")
    ax_rtd.set_xlim(0.0, truncate_rt_wrt_stim_s)
    ax_rtd.set_ylim(0.0, max(rtd_y_max * 1.15, 0.1))
    ax_slope.set_ylim(-max(slope_abs_max * 1.15, 0.03), max(slope_abs_max * 1.15, 0.03))
    ax_rtd.grid(alpha=0.2, linewidth=0.6)
    ax_slope.grid(alpha=0.2, linewidth=0.6)
    ax_rtd.legend(fontsize=8, loc="upper left")
    ax_slope.legend(fontsize=8, loc="upper left")
    refresh_status_text()
    fig.canvas.draw_idle()


def set_manual_pick(delay_s: float, autosave: bool):
    state["manual_pick_s_by_condition"][current_condition_key()] = clamp_delay_s(delay_s)
    update_manual_line_artists()
    refresh_status_text()
    fig.canvas.draw_idle()
    if autosave:
        save_manual_picks()
        refresh_status_text()
        fig.canvas.draw_idle()


def clear_manual_pick(event=None):
    state["manual_pick_s_by_condition"][current_condition_key()] = np.nan
    update_manual_line_artists()
    save_manual_picks()
    refresh_status_text()
    fig.canvas.draw_idle()


def go_to_condition(index: int):
    state["condition_index"] = int(np.clip(index, 0, len(condition_keys) - 1))
    render_current_condition()


def go_prev(event=None):
    go_to_condition(state["condition_index"] - 1)


def go_next(event=None):
    go_to_condition(state["condition_index"] + 1)


def toggle_auto_lines(event=None):
    state["show_auto_lines"] = not state["show_auto_lines"]
    render_current_condition()


def save_only(event=None):
    save_manual_picks()
    refresh_status_text()
    fig.canvas.draw_idle()


def quit_picker(event=None):
    save_manual_picks()
    plt.close(fig)


def on_mouse_press(event):
    if event.inaxes not in (ax_rtd, ax_slope):
        return
    if event.xdata is None:
        return
    state["dragging_line"] = True
    set_manual_pick(float(event.xdata), autosave=False)


def on_mouse_move(event):
    if not state["dragging_line"]:
        return
    if event.inaxes not in (ax_rtd, ax_slope):
        return
    if event.xdata is None:
        return
    set_manual_pick(float(event.xdata), autosave=False)


def on_mouse_release(event):
    if not state["dragging_line"]:
        return
    state["dragging_line"] = False
    if event.inaxes in (ax_rtd, ax_slope) and event.xdata is not None:
        set_manual_pick(float(event.xdata), autosave=True)
    else:
        save_manual_picks()
        refresh_status_text()
        fig.canvas.draw_idle()


def on_key_press(event):
    current_pick = current_manual_pick_s()
    if not np.isfinite(current_pick):
        current_pick = default_pick_s()

    if event.key == "left":
        set_manual_pick(current_pick - manual_fine_step_s, autosave=True)
    elif event.key == "right":
        set_manual_pick(current_pick + manual_fine_step_s, autosave=True)
    elif event.key == "shift+left":
        set_manual_pick(current_pick - manual_coarse_step_s, autosave=True)
    elif event.key == "shift+right":
        set_manual_pick(current_pick + manual_coarse_step_s, autosave=True)
    elif event.key in ("n", "enter", "down"):
        go_next()
    elif event.key in ("p", "backspace", "up"):
        go_prev()
    elif event.key == "r":
        clear_manual_pick()
    elif event.key == "a":
        toggle_auto_lines()
    elif event.key == "s":
        save_only()
    elif event.key == "q":
        quit_picker()


def on_close(event):
    save_manual_picks()


button_prev.on_clicked(go_prev)
button_next.on_clicked(go_next)
button_clear.on_clicked(clear_manual_pick)
button_auto.on_clicked(toggle_auto_lines)
button_save.on_clicked(save_only)
button_quit.on_clicked(quit_picker)

fig.canvas.mpl_connect("button_press_event", on_mouse_press)
fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)
fig.canvas.mpl_connect("button_release_event", on_mouse_release)
fig.canvas.mpl_connect("key_press_event", on_key_press)
fig.canvas.mpl_connect("close_event", on_close)

render_current_condition()
save_manual_picks()
refresh_status_text()

print("Opened manual picker.")
print("Controls:")
print("  click in either panel -> place manual delay")
print("  drag after click -> adjust line continuously")
print("  left/right -> +/-1 ms")
print("  shift+left/right -> +/-5 ms")
print("  n/p -> next/prev condition")
print("  r -> clear current manual pick")
print("  a -> toggle auto reference lines")
print("  s -> save CSV")
print("  q -> save and quit")

if show_plot:
    plt.show()
else:
    plt.close(fig)
