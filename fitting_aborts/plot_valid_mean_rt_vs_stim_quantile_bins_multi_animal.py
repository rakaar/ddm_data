# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# %%
SHOW_PLOT = True

# DESIRED_BATCHES = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]
DESIRED_BATCHES = ["LED7"]

EXCLUDED_BATCH_ANIMAL_PAIRS = []

INCLUDE_ABORT_EVENT_4 = True

ABL_VALUES = (20, 40, 60)
ABS_ILD_VALUES = (1, 2, 4, 8, 16)

NUM_STIM_BINS = 11

INTENDED_FIX_MIN_S = 0.2
INTENDED_FIX_MAX_S = 2.3
RT_MIN_S = 0.0
RT_MAX_S = None

png_dpi = 300
show_plot = SHOW_PLOT


# %%
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
batch_csv_dir = REPO_ROOT / "fit_animal_by_animal" / "batch_csvs"
output_dir = SCRIPT_DIR / "multi_animal_valid_mean_rt_stim_quantiles"

output_suffix = "_valid_plus_abort4" if INCLUDE_ABORT_EVENT_4 else "_valid_only"
summary_csv_path = output_dir / f"mean_rt_vs_stim_qcut_{NUM_STIM_BINS}bins{output_suffix}_summary.csv"
figure_output_base = output_dir / f"mean_rt_vs_stim_qcut_{NUM_STIM_BINS}bins{output_suffix}_combined"

abl_colors = {
    20: "tab:blue",
    40: "tab:orange",
    60: "tab:green",
}


# %%
def validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def validate_supported_values(df: pd.DataFrame, column_name: str, supported_values: tuple[int, ...]) -> None:
    observed = sorted(df[column_name].dropna().astype(float).unique().tolist())
    unexpected = [
        value
        for value in observed
        if not any(np.isclose(float(value), float(supported)) for supported in supported_values)
    ]
    if unexpected:
        raise ValueError(
            f"Unexpected {column_name} values after filtering: {unexpected}. Supported values are {supported_values}."
        )


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=png_dpi, bbox_inches="tight")


def get_batch_csv_path(batch_name: str) -> Path:
    batch_file_with_4 = batch_csv_dir / f"batch_{batch_name}_valid_and_aborts_and_4.csv"
    batch_file = batch_csv_dir / f"batch_{batch_name}_valid_and_aborts.csv"

    if INCLUDE_ABORT_EVENT_4 and batch_file_with_4.exists():
        return batch_file_with_4
    if batch_file.exists():
        return batch_file

    raise FileNotFoundError(f"Missing batch CSV for batch '{batch_name}'.")


def load_merged_trials() -> pd.DataFrame:
    batch_files = []
    missing_batch_files = []

    for batch_name in DESIRED_BATCHES:
        try:
            batch_files.append(get_batch_csv_path(batch_name))
        except FileNotFoundError:
            missing_batch_files.append(batch_name)

    if not batch_files:
        raise FileNotFoundError(
            f"Could not find any batch CSVs in {batch_csv_dir} for DESIRED_BATCHES={DESIRED_BATCHES}"
        )

    merged_data = pd.concat([pd.read_csv(batch_file) for batch_file in batch_files], ignore_index=True)
    validate_required_columns(
        merged_data,
        ["batch_name", "animal", "success", "abort_event", "RTwrtStim", "ABL", "ILD", "intended_fix"],
    )

    for column_name in ["animal", "abort_event", "RTwrtStim", "ABL", "ILD", "intended_fix"]:
        merged_data[column_name] = pd.to_numeric(merged_data[column_name], errors="coerce")

    if INCLUDE_ABORT_EVENT_4:
        keep_mask = merged_data["success"].isin([1, -1]) | np.isclose(merged_data["abort_event"], 4)
    else:
        keep_mask = merged_data["success"].isin([1, -1])

    pooled_df = merged_data[keep_mask].copy()
    pooled_df["abs_ILD"] = pooled_df["ILD"].abs()
    pooled_df["is_abort4"] = np.isclose(pooled_df["abort_event"], 4)

    if missing_batch_files:
        print(f"Skipped missing batches: {missing_batch_files}")

    if EXCLUDED_BATCH_ANIMAL_PAIRS:
        excluded_pairs = {(str(batch), str(animal)) for batch, animal in EXCLUDED_BATCH_ANIMAL_PAIRS}
        batch_animal_keys = list(
            zip(pooled_df["batch_name"].astype(str), pooled_df["animal"].astype("Int64").astype(str))
        )
        pooled_df = pooled_df[[key not in excluded_pairs for key in batch_animal_keys]].copy()
        print(f"Excluded batch-animal pairs: {sorted(excluded_pairs)}")

    return pooled_df


def prepare_plot_df(pooled_df: pd.DataFrame) -> pd.DataFrame:
    mask = pooled_df["intended_fix"].notna() & pooled_df["RTwrtStim"].notna()
    mask &= pooled_df["ABL"].isin(ABL_VALUES)
    mask &= pooled_df["abs_ILD"].isin(ABS_ILD_VALUES)
    mask &= pooled_df["intended_fix"] >= INTENDED_FIX_MIN_S
    if INTENDED_FIX_MAX_S is not None:
        mask &= pooled_df["intended_fix"] <= INTENDED_FIX_MAX_S
    if RT_MIN_S is not None:
        mask &= pooled_df["RTwrtStim"] >= RT_MIN_S
    if RT_MAX_S is not None:
        mask &= pooled_df["RTwrtStim"] <= RT_MAX_S

    plot_df = pooled_df[mask].copy()
    if len(plot_df) == 0:
        raise ValueError("No trials found after filtering.")

    validate_supported_values(plot_df, "ABL", ABL_VALUES)
    validate_supported_values(plot_df, "abs_ILD", ABS_ILD_VALUES)
    return plot_df


def add_global_stim_bins(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    if NUM_STIM_BINS <= 0:
        raise ValueError("NUM_STIM_BINS must be positive.")
    if df["intended_fix"].isna().any():
        raise ValueError("Cannot build quantile bins with NaN intended_fix values.")
    if float(df["intended_fix"].nunique()) <= 1:
        raise ValueError("Cannot build quantile bins because all intended_fix values are identical.")

    stim_bin, stim_edges = pd.qcut(
        df["intended_fix"],
        q=NUM_STIM_BINS,
        labels=False,
        retbins=True,
        duplicates="drop",
    )
    if len(stim_edges) - 1 != NUM_STIM_BINS:
        raise ValueError(
            f"Requested {NUM_STIM_BINS} bins, but only {len(stim_edges) - 1} unique bins could be formed."
        )
    if stim_bin.isna().any():
        raise ValueError("Failed to assign some trials to qcut bins.")

    binned_df = df.copy()
    binned_df["stim_bin"] = stim_bin.astype(int)
    return binned_df, np.asarray(stim_edges, dtype=float)


def build_subset_summary(df: pd.DataFrame, subset_name: str, stim_edges: np.ndarray) -> pd.DataFrame:
    bin_index = list(range(len(stim_edges) - 1))

    counts = df.groupby("stim_bin").size().reindex(bin_index, fill_value=0)
    mean_rt = df.groupby("stim_bin")["RTwrtStim"].mean().reindex(bin_index)
    std_rt = df.groupby("stim_bin")["RTwrtStim"].std().reindex(bin_index)
    sem_rt = (std_rt / np.sqrt(counts.replace(0, np.nan))).fillna(0.0)
    frac_abs_ild = (
        df.groupby(["stim_bin", "abs_ILD"]).size().unstack("abs_ILD")
        .reindex(index=bin_index, columns=ABS_ILD_VALUES, fill_value=0)
    )
    frac_abs_ild = frac_abs_ild.div(frac_abs_ild.sum(axis=1).replace(0, np.nan), axis=0)

    summary_df = pd.DataFrame(
        {
            "subset": subset_name,
            "stim_bin": bin_index,
            "stim_left_s": stim_edges[:-1],
            "stim_right_s": stim_edges[1:],
            "stim_mid_s": 0.5 * (stim_edges[:-1] + stim_edges[1:]),
            "n_trials": counts.values,
            "mean_rt_s": mean_rt.values,
            "sem_rt_s": sem_rt.values,
        }
    )
    for abs_ild_value in ABS_ILD_VALUES:
        summary_df[f"frac_abs_ild_{abs_ild_value}"] = frac_abs_ild[int(abs_ild_value)].values

    return summary_df


def format_fraction_block(summary_df: pd.DataFrame, label: str) -> str:
    lines = [f"{label}  p(|ILD|)"]
    for row in summary_df.itertuples(index=False):
        parts = [
            f"Q{int(row.stim_bin) + 1:02d}",
            f"mid={float(row.stim_mid_s) * 1e3:6.1f}ms",
            f"n={int(row.n_trials):5d}",
        ]
        for abs_ild_value in ABS_ILD_VALUES:
            frac_value = getattr(row, f"frac_abs_ild_{int(abs_ild_value)}")
            parts.append(f"{int(abs_ild_value):>2d}:{float(frac_value):.3f}")
        lines.append("  ".join(parts))
    return "\n".join(lines)


def make_combined_figure(summary_df: pd.DataFrame) -> plt.Figure:
    aggregate_df = summary_df[summary_df["subset"] == "aggregate"].copy()
    abl_summary_dfs = {
        int(abl_value): summary_df[summary_df["subset"] == f"abl_{int(abl_value)}"].copy()
        for abl_value in ABL_VALUES
    }

    left_text = "\n\n".join(
        [format_fraction_block(abl_summary_dfs[int(abl_value)], f"ABL {int(abl_value)}") for abl_value in ABL_VALUES]
    )
    right_text = format_fraction_block(aggregate_df, "Aggregate")

    left_line_count = len(left_text.splitlines())
    right_line_count = len(right_text.splitlines())
    max_line_count = max(left_line_count, right_line_count)

    fig_height = max(8.5, 4.8 + 0.15 * max_line_count)
    bottom_margin = min(0.60, 0.14 + 0.012 * max_line_count)

    fig, axes = plt.subplots(1, 2, figsize=(17.0, fig_height), sharey=True)

    left_ax = axes[0]
    for abl_value in ABL_VALUES:
        abl_df = abl_summary_dfs[int(abl_value)]
        left_ax.errorbar(
            abl_df["stim_mid_s"] * 1e3,
            abl_df["mean_rt_s"] * 1e3,
            yerr=abl_df["sem_rt_s"] * 1e3,
            fmt="o-",
            color=abl_colors[int(abl_value)],
            linewidth=2,
            markersize=5,
            capsize=3,
            label=f"ABL {int(abl_value)}",
        )
    left_ax.set_title(
        "Mean RT vs stim bin midpoint by ABL (error bars = SEM)\n"
        f"qcut bins = {NUM_STIM_BINS}"
    )
    left_ax.set_xlabel("Stim-bin midpoint (ms)")
    left_ax.set_ylabel("Mean RTwrtStim (ms)")
    left_ax.grid(alpha=0.3)
    left_ax.legend(frameon=False)
    left_ax.text(
        0.0,
        -0.23,
        left_text,
        transform=left_ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.7,
        family="monospace",
    )
    left_ax.set_xlim(200, 800)

    right_ax = axes[1]
    right_ax.errorbar(
        aggregate_df["stim_mid_s"] * 1e3,
        aggregate_df["mean_rt_s"] * 1e3,
        yerr=aggregate_df["sem_rt_s"] * 1e3,
        fmt="o-",
        color="k",
        linewidth=2,
        markersize=5,
        capsize=3,
    )
    right_ax.set_title(
        "Aggregate mean RT vs stim bin midpoint (error bars = SEM)\n"
        f"qcut bins = {NUM_STIM_BINS}"
    )
    right_ax.set_xlabel("Stim-bin midpoint (ms)")
    right_ax.grid(alpha=0.3)
    right_ax.text(
        0.0,
        -0.23,
        right_text,
        transform=right_ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.9,
        family="monospace",
    )
    right_ax.set_xlim(200, 800)

    fig.subplots_adjust(bottom=bottom_margin, wspace=0.28)
    return fig


def print_summary_report(plot_df: pd.DataFrame, stim_edges: np.ndarray, summary_df: pd.DataFrame) -> None:
    valid_count = int(plot_df["success"].isin([1, -1]).sum())
    abort4_count = int(plot_df["is_abort4"].sum())

    print("Mean RT vs stim qcut bins")
    print(f"  Total filtered trials: {len(plot_df)}")
    print(f"  Included valid trials: {valid_count}")
    print(f"  Included abort_event == 4 trials: {abort4_count}")
    print(
        "  intended_fix range (s): "
        f"[{float(plot_df['intended_fix'].min()):.6f}, {float(plot_df['intended_fix'].max()):.6f}]"
    )
    print(
        "  RTwrtStim range (s): "
        f"[{float(plot_df['RTwrtStim'].min()):.6f}, {float(plot_df['RTwrtStim'].max()):.6f}]"
    )
    print(f"  NUM_STIM_BINS: {NUM_STIM_BINS}")
    print(f"  qcut edges (s): {[float(edge) for edge in stim_edges]}")
    print("")
    for subset_name, group_df in summary_df.groupby("subset", sort=False):
        print(subset_name)
        print(
            group_df[
                ["stim_bin", "n_trials", "stim_mid_s", "mean_rt_s", "sem_rt_s"]
            ].to_string(index=False)
        )
        print("")


# %%
output_dir.mkdir(parents=True, exist_ok=True)

pooled_df = load_merged_trials()
plot_df = prepare_plot_df(pooled_df)
plot_df, stim_edges = add_global_stim_bins(plot_df)

summary_frames = [build_subset_summary(plot_df, "aggregate", stim_edges)]
for abl_value in ABL_VALUES:
    abl_df = plot_df[np.isclose(plot_df["ABL"], abl_value)].copy()
    summary_frames.append(build_subset_summary(abl_df, f"abl_{int(abl_value)}", stim_edges))

summary_df = pd.concat(summary_frames, ignore_index=True)
summary_df.to_csv(summary_csv_path, index=False)

print_summary_report(plot_df, stim_edges, summary_df)
print(f"Saved summary CSV: {summary_csv_path}")

fig = make_combined_figure(summary_df)
save_figure(fig, figure_output_base)
print(f"Saved figure: {figure_output_base.with_suffix('.pdf')}")

if show_plot:
    plt.show()
else:
    plt.close(fig)
