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
num_intended_fix_segments = 2

supported_abl_values = (20, 40, 60)
supported_abs_ild_values = (1, 2, 4, 8, 16)

rt_min_s = 0.0
rt_max_s = 1.0
intended_fix_max_s = 0.5
bin_size_s = 5e-3
xlim_ms = (0, 100)
ylabel_rtd = "Density"
ylabel_cdf = "CDF"
rtd_ylim = (0, 6)
cdf_ylim = (0, 0.1)

panel_width = 5.0
panel_height = 3.2
png_dpi = 300
show_plot = SHOW_PLOT

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
led_data_csv_path = REPO_ROOT / "out_LED.csv"
output_dir = SCRIPT_DIR / "led_off_valid_rtd_cdf_diagnostics"

rtd_plot_base = "led_off_valid_rtd_by_abl_abs_ild"
cdf_plot_base = "led_off_valid_cdf_by_abl_abs_ild"


# %%
############ Helpers ############
def validate_supported_values(df: pd.DataFrame, column_name: str, supported_values: tuple[int, ...]):
    observed = sorted(df[column_name].dropna().astype(float).unique().tolist())
    unexpected = [
        value
        for value in observed
        if not any(np.isclose(float(value), float(supported)) for supported in supported_values)
    ]
    if unexpected:
        raise ValueError(
            f"Unexpected {column_name} values after filtering: {unexpected}. "
            f"Supported values are {supported_values}."
        )


def format_counts(series: pd.Series) -> dict[int, int]:
    counts = (
        series.astype(float).round().astype(int).value_counts().sort_index().to_dict()
    )
    return {int(key): int(value) for key, value in counts.items()}


def build_condition_counts(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby(["ABL", "abs_ILD"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=supported_abl_values, columns=supported_abs_ild_values, fill_value=0)
    counts.index = counts.index.astype(int)
    counts.columns = counts.columns.astype(int)
    return counts


def build_segment_condition_counts(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df.groupby(["intended_fix_segment", "ABL", "abs_ILD"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    return counts


def compute_density_histogram(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(len(bins) - 1, dtype=float)
    hist, _ = np.histogram(values, bins=bins, density=True)
    return hist


def compute_binned_cdf(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(len(bins) - 1, dtype=float)
    counts, _ = np.histogram(values, bins=bins, density=False)
    return np.cumsum(counts) / len(values)


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=png_dpi, bbox_inches="tight")


def add_intended_fix_segments(df: pd.DataFrame, n_segments: int) -> tuple[pd.DataFrame, np.ndarray]:
    if n_segments <= 0:
        raise ValueError("num_intended_fix_segments must be positive.")

    intended_fix_min = float(df["intended_fix"].min())
    intended_fix_max = float(df["intended_fix"].max())
    if np.isclose(intended_fix_min, intended_fix_max):
        raise ValueError("Cannot segment intended_fix because all filtered values are identical.")

    segment_edges = np.linspace(intended_fix_min, intended_fix_max, n_segments + 1)
    segment_ids = pd.cut(
        df["intended_fix"],
        bins=segment_edges,
        include_lowest=True,
        labels=False,
        duplicates="raise",
    )
    if segment_ids.isna().any():
        raise ValueError("Failed to assign intended_fix segments to some trials.")

    segmented_df = df.copy()
    segmented_df["intended_fix_segment"] = segment_ids.astype(int)
    return segmented_df, segment_edges


def apply_intended_fix_upper_bound(df: pd.DataFrame, intended_fix_max: float | None) -> pd.DataFrame:
    if intended_fix_max is None:
        return df.copy()
    return df[df["intended_fix"] <= intended_fix_max].copy()


def format_segment_label(segment_idx: int, segment_edges: np.ndarray) -> str:
    left_edge = segment_edges[segment_idx]
    right_edge = segment_edges[segment_idx + 1]
    return f"intended_fix [{left_edge:.3f}, {right_edge:.3f}] s"


def make_condition_plot(
    df: pd.DataFrame,
    bins_s: np.ndarray,
    y_mode: str,
    colors_by_abs_ild: dict[int, str],
    segment_edges: np.ndarray,
):
    if y_mode not in {"rtd", "cdf"}:
        raise ValueError(f"Unsupported y_mode: {y_mode}")

    n_segments = len(segment_edges) - 1
    figure_size = (panel_width * len(supported_abl_values), panel_height * n_segments)
    fig, axes = plt.subplots(
        n_segments,
        len(supported_abl_values),
        figsize=figure_size,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    x_edges_ms = bins_s * 1e3

    for segment_idx in range(n_segments):
        segment_df = df[df["intended_fix_segment"] == segment_idx].copy()
        segment_label = format_segment_label(segment_idx, segment_edges)

        for col_idx, abl_value in enumerate(supported_abl_values):
            ax = axes[segment_idx, col_idx]
            abl_df = segment_df[np.isclose(segment_df["ABL"], abl_value)].copy()

            for abs_ild_value in supported_abs_ild_values:
                condition_df = abl_df[np.isclose(abl_df["abs_ILD"], abs_ild_value)].copy()
                values = condition_df["RTwrtStim"].to_numpy()

                if y_mode == "rtd":
                    y_values = compute_density_histogram(values, bins_s)
                else:
                    y_values = compute_binned_cdf(values, bins_s)

                ax.stairs(
                    y_values,
                    x_edges_ms,
                    label=f"|ILD| = {abs_ild_value}",
                    color=colors_by_abs_ild[abs_ild_value],
                    linewidth=1.8,
                )

            if segment_idx == 0:
                ax.set_title(f"ABL = {abl_value}")
            ax.set_xlim(*xlim_ms)
            ax.grid(alpha=0.2, linewidth=0.6)
            if segment_idx == n_segments - 1:
                ax.set_xlabel("RT wrt stim (ms)")

            if col_idx == 0:
                ax.set_ylabel(
                    f"{ylabel_rtd}\n{segment_label}" if y_mode == "rtd" else f"{ylabel_cdf}\n{segment_label}"
                )

            if y_mode == "rtd" and rtd_ylim is not None:
                ax.set_ylim(*rtd_ylim)
            if y_mode == "cdf" and cdf_ylim is not None:
                ax.set_ylim(*cdf_ylim)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(supported_abs_ild_values), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


# %%
############ Load and filter LED-OFF valid-trial dataset ############
if not led_data_csv_path.exists():
    raise FileNotFoundError(f"Could not find data CSV: {led_data_csv_path}")

output_dir.mkdir(parents=True, exist_ok=True)

exp_df = pd.read_csv(led_data_csv_path)
exp_df["RTwrtStim"] = exp_df["timed_fix"] - exp_df["intended_fix"]
exp_df["abs_ILD"] = exp_df["ILD"].abs()

mask_led_off = (exp_df["LED_trial"] == 0) | (exp_df["LED_trial"].isna())
mask_repeat = exp_df["repeat_trial"].isin(allowed_repeat_trials) | exp_df["repeat_trial"].isna()
mask_valid = exp_df["success"].isin([1, -1])
mask_rt = (exp_df["RTwrtStim"] >= rt_min_s) & (exp_df["RTwrtStim"] < rt_max_s)
mask_abl = exp_df["ABL"].isin(supported_abl_values)
mask_abs_ild = exp_df["abs_ILD"].isin(supported_abs_ild_values)

plot_df = exp_df[
    mask_led_off
    & mask_repeat
    & mask_valid
    & mask_rt
    & mask_abl
    & mask_abs_ild
    & exp_df["session_type"].isin([session_type])
    & exp_df["training_level"].isin([training_level])
].copy()
plot_df = apply_intended_fix_upper_bound(plot_df, intended_fix_max_s)

if len(plot_df) == 0:
    raise ValueError("No LED-OFF valid trials found after filtering.")

validate_supported_values(plot_df, "ABL", supported_abl_values)
validate_supported_values(plot_df, "abs_ILD", supported_abs_ild_values)
plot_df, intended_fix_segment_edges = add_intended_fix_segments(plot_df, num_intended_fix_segments)

abl_counts = format_counts(plot_df["ABL"])
abs_ild_counts = format_counts(plot_df["abs_ILD"])
condition_counts = build_condition_counts(plot_df)
segment_counts = (
    plot_df["intended_fix_segment"].value_counts().sort_index().to_dict()
)
segment_condition_counts = build_segment_condition_counts(plot_df)

print("Rebuilt LED-OFF valid-trial dataset for RTD/CDF plots:")
print(f"  Total filtered LED-OFF valid trials: {len(plot_df)}")
print(f"  Counts by ABL: {abl_counts}")
print(f"  Counts by abs_ILD: {abs_ild_counts}")
print(
    "  intended_fix upper bound (s): "
    f"{float(plot_df['intended_fix'].max()) if intended_fix_max_s is None else intended_fix_max_s}"
)
print(f"  intended_fix segment count: {num_intended_fix_segments}")
print(f"  intended_fix segment edges (s): {[float(edge) for edge in intended_fix_segment_edges]}")
print(f"  Counts by intended_fix segment: {segment_counts}")
print("  Counts by ABL x abs_ILD:")
print(condition_counts.to_string())
print("  Counts by intended_fix segment x ABL x abs_ILD:")
print(segment_condition_counts.to_string())


# %%
############ Build and save RTD/CDF plots ############
rt_bins_s = np.arange(rt_min_s, rt_max_s + bin_size_s, bin_size_s)
abs_ild_colors = {
    1: "tab:blue",
    2: "tab:orange",
    4: "tab:green",
    8: "tab:red",
    16: "tab:purple",
}

rtd_fig = make_condition_plot(
    plot_df,
    bins_s=rt_bins_s,
    y_mode="rtd",
    colors_by_abs_ild=abs_ild_colors,
    segment_edges=intended_fix_segment_edges,
)
cdf_fig = make_condition_plot(
    plot_df,
    bins_s=rt_bins_s,
    y_mode="cdf",
    colors_by_abs_ild=abs_ild_colors,
    segment_edges=intended_fix_segment_edges,
)

rtd_output_base = output_dir / rtd_plot_base
cdf_output_base = output_dir / cdf_plot_base

save_figure(rtd_fig, rtd_output_base)
save_figure(cdf_fig, cdf_output_base)

print("Saved RTD figure:")
print(f"  {rtd_output_base.with_suffix('.pdf')}")
print(f"  {rtd_output_base.with_suffix('.png')}")
print("Saved CDF figure:")
print(f"  {cdf_output_base.with_suffix('.pdf')}")
print(f"  {cdf_output_base.with_suffix('.png')}")

if show_plot:
    plt.show()
else:
    plt.close(rtd_fig)
    plt.close(cdf_fig)

# %%
