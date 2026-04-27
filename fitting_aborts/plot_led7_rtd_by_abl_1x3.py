# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# %%
############ Parameters (edit here) ############
batch_name = "LED7"
session_type = 7
training_level = 16
allowed_repeat_trials = [0, 2]
supported_abl_values = (20, 40, 60)
xlim_s = (0.0, 0.120)
data_bin_size_s = 5e-3
max_rtwrtstim_for_plot = 1.0

show_plot = True
png_dpi = 300
panel_width = 4.2
panel_height = 3.6

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
led_data_csv_path = REPO_ROOT / "out_LED.csv"
output_dir = SCRIPT_DIR / "led7_rtd_by_abl_1x3"
plot_output_base = output_dir / "led7_rtd_by_abl_1x3_xlim_0_120ms"


# %%
############ Helpers ############
def format_abl_counts(df: pd.DataFrame) -> dict[int, int]:
    counts = (
        df["ABL"]
        .astype(float)
        .round()
        .astype(int)
        .value_counts()
        .sort_index()
        .to_dict()
    )
    return {int(key): int(value) for key, value in counts.items()}


def build_bin_edges(rt_max_s: float, bin_width_s: float) -> np.ndarray:
    n_bins_float = rt_max_s / bin_width_s
    n_bins = int(round(n_bins_float))
    if not np.isclose(n_bins * bin_width_s, rt_max_s):
        raise ValueError(
            f"rt_max_s={rt_max_s} must be an integer multiple of bin_width_s={bin_width_s}."
        )
    return np.linspace(0.0, rt_max_s, n_bins + 1)


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=png_dpi, bbox_inches="tight")


# %%
############ Load and preprocess LED7 LED-OFF data ############
if not led_data_csv_path.exists():
    raise FileNotFoundError(f"Could not find data CSV: {led_data_csv_path}")

output_dir.mkdir(parents=True, exist_ok=True)

exp_df = pd.read_csv(led_data_csv_path)
exp_df["RTwrtStim"] = exp_df["timed_fix"] - exp_df["intended_fix"]
exp_df = exp_df.rename(columns={"timed_fix": "TotalFixTime"})
exp_df = exp_df[~((exp_df["RTwrtStim"].isna()) & (exp_df["abort_event"] == 3))].copy()

mask_led_off = (exp_df["LED_trial"] == 0) | (exp_df["LED_trial"].isna())
mask_repeat = exp_df["repeat_trial"].isin(allowed_repeat_trials) | exp_df["repeat_trial"].isna()
exp_df_led_off = exp_df[
    mask_led_off
    & mask_repeat
    & exp_df["session_type"].isin([session_type])
    & exp_df["training_level"].isin([training_level])
].copy()

valid_or_early_response = (
    exp_df_led_off["success"].isin([1, -1]) | exp_df_led_off["abort_event"].isin([3, 4])
)
plot_df = exp_df_led_off[
    valid_or_early_response
    & exp_df_led_off["ABL"].isin(supported_abl_values)
    & (exp_df_led_off["RTwrtStim"] >= 0.0)
    & (exp_df_led_off["RTwrtStim"] < max_rtwrtstim_for_plot)
].copy()

if len(plot_df) == 0:
    raise ValueError("No LED7 LED-OFF rows found after filtering.")

observed_abl_values = np.sort(plot_df["ABL"].dropna().astype(float).unique())
unexpected_abl_values = [
    float(abl)
    for abl in observed_abl_values
    if not any(np.isclose(float(abl), float(supported)) for supported in supported_abl_values)
]
if unexpected_abl_values:
    raise ValueError(
        f"Unexpected ABL values in filtered data: {unexpected_abl_values}. "
        f"Supported values are {supported_abl_values}."
    )

print("Rebuilt LED7 LED-OFF aggregate RTD dataset:")
print(f"  Data CSV: {led_data_csv_path}")
print(f"  Batch={batch_name}, session_type={session_type}, training_level={training_level}")
print(f"  Repeat trials kept: {allowed_repeat_trials} plus missing")
print("  Trial pool: success in {1, -1} or abort_event in {3, 4}")
print(f"  Broad RTwrtStim window used for histogramming: [0, {max_rtwrtstim_for_plot:.3f}) s")
print(f"  Display x-limit only: [{xlim_s[0]:.3f}, {xlim_s[1]:.3f}] s")
print(f"  Total filtered LED-OFF rows: {len(plot_df)}")
print(f"  Counts by ABL: {format_abl_counts(plot_df)}")

for abl_value in supported_abl_values:
    n_abl = int(np.isclose(plot_df["ABL"].astype(float), float(abl_value)).sum())
    if n_abl == 0:
        raise ValueError(f"No rows found for ABL={abl_value} after filtering.")


# %%
############ Build RTD densities by ABL ############
bin_edges_s = build_bin_edges(max_rtwrtstim_for_plot, data_bin_size_s)
bin_width_s = float(np.diff(bin_edges_s)[0])

abl_colors = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}
abl_density_payload: dict[int, dict[str, np.ndarray | int | float]] = {}
global_max_density_display = 0.0

for abl_value in supported_abl_values:
    abl_df = plot_df[np.isclose(plot_df["ABL"].astype(float), float(abl_value))].copy()
    rt_values = abl_df["RTwrtStim"].to_numpy(dtype=float)
    counts, _ = np.histogram(rt_values, bins=bin_edges_s)
    density = counts / (len(abl_df) * bin_width_s)
    area = float(np.sum(density * np.diff(bin_edges_s)))

    display_bin_mask = (bin_edges_s[:-1] < xlim_s[1]) & (bin_edges_s[1:] > xlim_s[0])
    if display_bin_mask.any():
        global_max_density_display = max(global_max_density_display, float(np.max(density[display_bin_mask])))

    abl_density_payload[int(abl_value)] = {
        "n_total": int(len(abl_df)),
        "counts": counts,
        "density": density,
        "area": area,
    }

    n_visible = int(((rt_values >= xlim_s[0]) & (rt_values < xlim_s[1])).sum())
    print(
        f"  ABL={abl_value}: N={len(abl_df)}, "
        f"N in displayed window={n_visible}, density area over [0, 1s)={area:.6f}"
    )


# %%
############ Build and save 1 x 3 figure ############
fig, axes = plt.subplots(
    1,
    len(supported_abl_values),
    figsize=(panel_width * len(supported_abl_values), panel_height),
    sharex=True,
    sharey=True,
)

if len(supported_abl_values) == 1:
    axes = [axes]

y_limit = 1.08 * global_max_density_display if global_max_density_display > 0 else 1.0

for ax, abl_value in zip(axes, supported_abl_values):
    payload = abl_density_payload[int(abl_value)]
    density = payload["density"]
    n_total = int(payload["n_total"])

    ax.stairs(
        density,
        bin_edges_s * 1e3,
        baseline=0.0,
        color=abl_colors[int(abl_value)],
        linewidth=2.0,
    )
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_xlim(xlim_s[0] * 1e3, xlim_s[1] * 1e3)
    ax.set_ylim(0.0, y_limit)
    ax.set_title(f"ABL = {abl_value}")
    ax.set_xlabel("RT wrt stim (ms)")
    ax.text(
        0.97,
        0.92,
        f"N = {n_total}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )

axes[0].set_ylabel("Density (1/s)")
fig.suptitle(
    f"{batch_name} LED-OFF RTD by ABL "
    f"(session_type={session_type}, training_level={training_level})",
    y=1.02,
)
fig.tight_layout()

save_figure(fig, plot_output_base)

print("Saved RTD by ABL figure:")
print(f"  {plot_output_base.with_suffix('.pdf')}")
print(f"  {plot_output_base.with_suffix('.png')}")

if show_plot:
    plt.show()
else:
    plt.close(fig)

# %%
