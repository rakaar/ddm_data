# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
led_data_csv_path = REPO_ROOT / "out_LED.csv"
output_dir = SCRIPT_DIR / "led_off_tachometric_by_abl_two_truncations"
plot_output_base = output_dir / "led_off_tachometric_by_abl_two_truncations"


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


def compute_tachometric(df: pd.DataFrame, bin_edges_s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bin_right_edges_s = bin_edges_s[1:]
    if len(df) == 0:
        return (
            bin_right_edges_s,
            np.full(len(bin_right_edges_s), np.nan),
            np.zeros(len(bin_right_edges_s), dtype=int),
        )

    rt_values = df["RTwrtStim"].to_numpy(dtype=float)
    accuracy_values = df["accuracy"].to_numpy(dtype=float)

    counts, _ = np.histogram(rt_values, bins=bin_edges_s)
    weighted_accuracy_sum, _ = np.histogram(rt_values, bins=bin_edges_s, weights=accuracy_values)

    mean_accuracy = np.full(len(bin_right_edges_s), np.nan, dtype=float)
    valid_mask = counts > 0
    mean_accuracy[valid_mask] = weighted_accuracy_sum[valid_mask] / counts[valid_mask]
    return bin_right_edges_s, mean_accuracy, counts


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
plot_df["accuracy"] = (plot_df["success"] == 1).astype(float)

observed_abl_values = validate_supported_abl_values(plot_df, "filtered LED-OFF valid dataset")

print("Rebuilt LED-OFF aggregate valid-trial dataset for tachometric plot:")
print(f"  Total filtered LED-OFF valid trials: {len(plot_df)}")
print(f"  Supported ABL values in filtered data: {observed_abl_values.tolist()}")
print(f"  Counts by ABL before truncation: {format_abl_counts(plot_df)}")


# %%
############ Build and save tachometric figure ############
abl_colors = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}
truncation_plot_kwargs = {
    0.10: {
        "linestyle": "--",
        "linewidth": 2.6,
        "alpha": 1.0,
        "zorder": 3,
        "marker": "o",
        "markersize": 3.2,
        "markerfacecolor": "white",
        "markeredgewidth": 0.9,
    },
    0.13: {
        "linestyle": "-",
        "linewidth": 2.2,
        "alpha": 0.45,
        "zorder": 2,
    },
}

fig, ax = plt.subplots(figsize=(8.4, 5.2))
truncation_counts_summary: dict[float, dict[int, int]] = {}

for truncation_s in truncation_times_s:
    truncated_df = plot_df[plot_df["RTwrtStim"] <= truncation_s].copy()
    truncation_counts_summary[float(truncation_s)] = format_abl_counts(truncated_df)

    for abl_value in supported_abl_values:
        abl_df = truncated_df[np.isclose(truncated_df["ABL"], abl_value)].copy()
        bin_edges_s = build_bin_edges(truncation_s, bin_size_s)
        bin_right_edges_s, mean_accuracy, bin_counts = compute_tachometric(abl_df, bin_edges_s)

        ax.plot(
            bin_right_edges_s * 1e3,
            mean_accuracy,
            color=abl_colors[abl_value],
            label=f"ABL={abl_value}, trunc={truncation_s:.2f} s (N={len(abl_df)})",
            **truncation_plot_kwargs[float(truncation_s)],
        )

        if len(abl_df) == 0:
            print(f"  trunc={truncation_s:.2f} s, ABL={abl_value}: no retained trials")
            continue

        nonempty_bins = int(np.sum(bin_counts > 0))
        print(
            f"  trunc={truncation_s:.2f} s, ABL={abl_value}: "
            f"N={len(abl_df)}, nonempty_bins={nonempty_bins}"
        )

ax.axhline(0.5, color="0.4", linestyle="--", linewidth=1.0, alpha=0.7)
ax.set_xlim(0, 130)
ax.set_ylim(0, 1)
ax.set_xlabel("RT wrt stim (ms)")
ax.set_ylabel("Accuracy")
ax.set_title("LED-OFF tachometric by ABL for two truncation windows")
ax.grid(alpha=0.25, linewidth=0.6)
ax.legend(frameon=False, ncol=2)

fig.tight_layout()
save_figure(fig, plot_output_base)

print("Retained trial counts by truncation and ABL:")
for truncation_s in truncation_times_s:
    print(f"  {truncation_s:.2f} s: {truncation_counts_summary[float(truncation_s)]}")
print("  Note: 0.10 s curves overlap the 0.13 s curves up to 100 ms by construction.")

print("Saved tachometric figure:")
print(f"  {plot_output_base.with_suffix('.pdf')}")
print(f"  {plot_output_base.with_suffix('.png')}")

if show_plot:
    plt.show()
else:
    plt.close(fig)

# %%
