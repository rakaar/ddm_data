# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


# %%
############ Parameters (edit here) ############
SHOW_PLOT = True

session_type = 7
training_level = 16
allowed_repeat_trials = [0, 2]
supported_abl_values = (20, 40, 60)
truncation_times_s = (0.10, 0.13)
bin_size_s = 5e-3
max_rtwrtstim_for_fit = 1.0

show_plot = SHOW_PLOT
png_dpi = 300
panel_width = 4.3
panel_height = 3.8

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
led_data_csv_path = REPO_ROOT / "out_LED.csv"
output_dir = SCRIPT_DIR / "led_off_signed_rtd_by_abl_two_truncations"
plot_output_base = output_dir / "led_off_signed_rtd_by_abl_two_truncations"


# %%
############ Helpers ############
def validate_supported_abl_values(df: pd.DataFrame, df_name: str) -> np.ndarray:
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


def build_bin_edges(truncation_s: float, bin_width_s: float) -> np.ndarray:
    n_bins_float = truncation_s / bin_width_s
    n_bins = int(round(n_bins_float))
    if not np.isclose(n_bins * bin_width_s, truncation_s):
        raise ValueError(
            f"truncation_s={truncation_s} must be an integer multiple of bin_width_s={bin_width_s}."
        )
    return np.linspace(0.0, truncation_s, n_bins + 1)


def compute_signed_densities(df: pd.DataFrame, bin_edges_s: np.ndarray) -> dict[str, np.ndarray | int | float]:
    bin_widths_s = np.diff(bin_edges_s)
    if not np.allclose(bin_widths_s, bin_widths_s[0]):
        raise ValueError("Expected uniform bin widths.")

    bin_width_s = float(bin_widths_s[0])
    n_total = int(len(df))
    up_df = df[df["response_poke"] == 3].copy()
    down_df = df[df["response_poke"] == 2].copy()
    n_up = int(len(up_df))
    n_down = int(len(down_df))

    up_counts, _ = np.histogram(up_df["RTwrtStim"].to_numpy(dtype=float), bins=bin_edges_s)
    down_counts, _ = np.histogram(down_df["RTwrtStim"].to_numpy(dtype=float), bins=bin_edges_s)

    if n_total == 0:
        up_density = np.zeros(len(bin_edges_s) - 1, dtype=float)
        down_density = np.zeros(len(bin_edges_s) - 1, dtype=float)
    else:
        up_density = up_counts / (n_total * bin_width_s)
        down_density = -(down_counts / (n_total * bin_width_s))

    up_fraction = (n_up / n_total) if n_total > 0 else np.nan
    down_fraction = (n_down / n_total) if n_total > 0 else np.nan
    up_area = float(np.sum(up_density * bin_widths_s))
    down_abs_area = float(np.sum(np.abs(down_density) * bin_widths_s))
    total_abs_area = float(np.sum((up_density + np.abs(down_density)) * bin_widths_s))

    return {
        "n_total": n_total,
        "n_up": n_up,
        "n_down": n_down,
        "up_fraction": up_fraction,
        "down_fraction": down_fraction,
        "up_density": up_density,
        "down_density": down_density,
        "up_area": up_area,
        "down_abs_area": down_abs_area,
        "total_abs_area": total_abs_area,
    }


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=png_dpi, bbox_inches="tight")


# %%
############ Load and preprocess data ############
if not led_data_csv_path.exists():
    raise FileNotFoundError(f"Could not find data CSV: {led_data_csv_path}")

output_dir.mkdir(parents=True, exist_ok=True)

exp_df = pd.read_csv(led_data_csv_path)
exp_df["RTwrtStim"] = exp_df["timed_fix"] - exp_df["intended_fix"]
exp_df = exp_df.rename(columns={"timed_fix": "TotalFixTime"})
exp_df = exp_df[exp_df["RTwrtStim"] < max_rtwrtstim_for_fit]
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

plot_df = exp_df_led_off[
    (exp_df_led_off["success"].isin([1, -1]))
    & (exp_df_led_off["ABL"].isin(supported_abl_values))
    & (exp_df_led_off["RTwrtStim"] > 0)
    & (exp_df_led_off["RTwrtStim"] < max_rtwrtstim_for_fit)
].copy()

valid_response_pokes = {2.0, 3.0}
observed_response_pokes = set(plot_df["response_poke"].dropna().astype(float).unique().tolist())
unexpected_response_pokes = sorted(observed_response_pokes - valid_response_pokes)
if unexpected_response_pokes:
    raise ValueError(
        f"Unexpected response_poke values in filtered LED-OFF valid dataset: {unexpected_response_pokes}. "
        "Expected only 2 (down) and 3 (up)."
    )

observed_abl_values = validate_supported_abl_values(plot_df, "filtered LED-OFF valid dataset")

print("Rebuilt LED-OFF aggregate valid-trial dataset for signed RTD plot:")
print(f"  Total filtered LED-OFF valid trials: {len(plot_df)}")
print(f"  Supported ABL values in filtered data: {observed_abl_values.tolist()}")
print(f"  Counts by ABL before truncation: {format_abl_counts(plot_df)}")


# %%
############ Build signed-density payload ############
truncation_plot_kwargs = {
    0.10: {"color": "tab:blue", "linestyle": "--", "linewidth": 2.0},
    0.13: {"color": "tab:orange", "linestyle": "-", "linewidth": 2.0},
}

signed_density_results: dict[tuple[int, float], dict[str, np.ndarray | int | float]] = {}
global_max_abs_density = 0.0

for truncation_s in truncation_times_s:
    truncated_df = plot_df[plot_df["RTwrtStim"] <= truncation_s].copy()
    print(f"trunc={truncation_s:.2f} s: retained counts by ABL -> {format_abl_counts(truncated_df)}")

    bin_edges_s = build_bin_edges(truncation_s, bin_size_s)
    for abl_value in supported_abl_values:
        abl_df = truncated_df[np.isclose(truncated_df["ABL"], abl_value)].copy()
        density_result = compute_signed_densities(abl_df, bin_edges_s)
        density_result["bin_edges_s"] = bin_edges_s
        signed_density_results[(int(abl_value), float(truncation_s))] = density_result

        max_abs_density = max(
            float(np.max(np.abs(density_result["up_density"]))),
            float(np.max(np.abs(density_result["down_density"]))),
        )
        global_max_abs_density = max(global_max_abs_density, max_abs_density)

        print(
            f"  ABL={abl_value}, trunc={truncation_s:.2f} s: "
            f"N={density_result['n_total']}, up={density_result['n_up']}, down={density_result['n_down']}, "
            f"up_frac={density_result['up_fraction']:.6f}, down_frac={density_result['down_fraction']:.6f}, "
            f"up_area={density_result['up_area']:.6f}, down_abs_area={density_result['down_abs_area']:.6f}, "
            f"total_abs_area={density_result['total_abs_area']:.6f}"
        )


# %%
############ Build and save figure ############
fig, axes = plt.subplots(
    1,
    len(supported_abl_values),
    figsize=(panel_width * len(supported_abl_values), panel_height),
    sharex=True,
    sharey=True,
)

if len(supported_abl_values) == 1:
    axes = [axes]

y_limit = 1.05 * global_max_abs_density if global_max_abs_density > 0 else 1.0

for ax, abl_value in zip(axes, supported_abl_values):
    for truncation_s in truncation_times_s:
        result = signed_density_results[(int(abl_value), float(truncation_s))]
        bin_edges_s = result["bin_edges_s"]

        ax.stairs(
            result["up_density"],
            bin_edges_s * 1e3,
            baseline=0.0,
            **truncation_plot_kwargs[float(truncation_s)],
        )
        ax.stairs(
            result["down_density"],
            bin_edges_s * 1e3,
            baseline=0.0,
            **truncation_plot_kwargs[float(truncation_s)],
        )

    ax.axhline(0.0, color="0.35", linewidth=1.0, alpha=0.8)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_xlim(0, 130)
    ax.set_ylim(-y_limit, y_limit)
    ax.set_title(f"ABL = {abl_value}")
    ax.set_xlabel("RT wrt stim (ms)")

axes[0].set_ylabel("Signed density")

legend_handles = [
    Line2D([0], [0], label=f"trunc = {truncation_s:.2f} s", **truncation_plot_kwargs[float(truncation_s)])
    for truncation_s in truncation_times_s
]
fig.legend(handles=legend_handles, loc="upper center", ncol=len(truncation_times_s), frameon=False)
fig.suptitle("LED-OFF signed up/down RTD by ABL for two truncation windows", y=1.03)
fig.text(0.5, 0.96, "Up plotted positive, down plotted negative", ha="center", va="center")

fig.tight_layout(rect=(0, 0, 1, 0.88))
save_figure(fig, plot_output_base)

print("Saved signed RTD figure:")
print(f"  {plot_output_base.with_suffix('.pdf')}")
print(f"  {plot_output_base.with_suffix('.png')}")

if show_plot:
    plt.show()
else:
    plt.close(fig)

# %%
