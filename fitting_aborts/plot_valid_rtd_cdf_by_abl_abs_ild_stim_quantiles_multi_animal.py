# %%
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# %%
SHOW_PLOT = True

# DESIRED_BATCHES = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]
DESIRED_BATCHES = ["LED8"]

EXCLUDED_BATCH_ANIMAL_PAIRS = []
num_intended_fix_quantile_bins = 2
TRIAL_POOL_MODE = "valid"  # "valid" or "valid_plus_abort3"

if TRIAL_POOL_MODE not in {"valid", "valid_plus_abort3"}:
    raise ValueError(f"Unsupported TRIAL_POOL_MODE: {TRIAL_POOL_MODE}")

supported_abl_values = (20, 40, 60)
supported_abs_ild_values = (1, 2, 4, 8, 16)

rt_min_s = -3
rt_max_s = 3.0
intended_fix_max_s = 1.5
bin_size_s = 5e-3
tachometric_bin_size_s = 5e-3
xlim_ms = (0, 1000)
ylabel_rtd = "Density"
ylabel_cdf = "CDF"
ylabel_tacho = "P(success = 1)"
rtd_ylim = (0, 10)
cdf_ylim = (0, 1)
tacho_ylim = (0, 1)
tachometric_random_choice_seed = 12345

panel_width = 5.0
panel_height = 3.2
png_dpi = 300
show_plot = SHOW_PLOT

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
batch_csv_dir = REPO_ROOT / "fit_animal_by_animal" / "batch_csvs"
output_dir = SCRIPT_DIR / "multi_animal_valid_rtd_cdf_stim_quantiles"

output_suffix = "_plus_abort3" if TRIAL_POOL_MODE == "valid_plus_abort3" else ""
rtd_plot_base = f"multi_animal_valid_rtd_by_abl_abs_ild_quantile_segments{output_suffix}"
cdf_plot_base = f"multi_animal_valid_cdf_by_abl_abs_ild_quantile_segments{output_suffix}"
rtd_plot_base_all_ild = f"multi_animal_valid_rtd_by_abl_all_ild_quantile_segments{output_suffix}"
rtd_plot_base_all_ild_segment_overlay = f"multi_animal_valid_rtd_by_abl_all_ild_segment_overlay_quantile_segments{output_suffix}"
rtd_plot_base_condition_grid_segment_overlay = f"multi_animal_valid_rtd_by_abl_abs_ild_condition_grid_segment_overlay_quantile_segments{output_suffix}"
cdf_plot_base_all_ild = f"multi_animal_valid_cdf_by_abl_all_ild_quantile_segments{output_suffix}"
cdf_plot_base_segment_overlay = f"multi_animal_valid_cdf_by_abl_abs_ild_segment_overlay_quantile_segments{output_suffix}"
cdf_plot_base_abl_segment_overlay = f"multi_animal_valid_cdf_by_abl_segment_overlay_quantile_segments{output_suffix}"
tacho_plot_base_all_ild = f"multi_animal_valid_tachometric_by_abl_all_ild_quantile_segments{output_suffix}"
tacho_plot_base_abs_ild = f"multi_animal_valid_tachometric_by_abl_abs_ild_quantile_segments{output_suffix}"


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


def format_counts(series: pd.Series) -> dict[int, int]:
    counts = series.astype(float).round().astype(int).value_counts().sort_index().to_dict()
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


def compute_binned_accuracy(rt_values: np.ndarray, correct_values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    if len(rt_values) == 0:
        return np.full(len(bins) - 1, np.nan, dtype=float)

    trial_counts, _ = np.histogram(rt_values, bins=bins, density=False)
    correct_counts, _ = np.histogram(rt_values[correct_values == 1], bins=bins, density=False)

    accuracy = np.full(len(bins) - 1, np.nan, dtype=float)
    nonzero_mask = trial_counts > 0
    accuracy[nonzero_mask] = correct_counts[nonzero_mask] / trial_counts[nonzero_mask]
    return accuracy


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=png_dpi, bbox_inches="tight")


def add_intended_fix_quantile_segments(df: pd.DataFrame, n_quantile_bins: int) -> tuple[pd.DataFrame, np.ndarray]:
    if n_quantile_bins <= 0:
        raise ValueError("num_intended_fix_quantile_bins must be positive.")

    if df["intended_fix"].isna().any():
        raise ValueError("Cannot build quantile segments with NaN intended_fix values.")

    if float(df["intended_fix"].nunique()) <= 1:
        raise ValueError("Cannot segment intended_fix because all filtered values are identical.")

    segment_ids, segment_edges = pd.qcut(
        df["intended_fix"],
        q=n_quantile_bins,
        labels=False,
        retbins=True,
        duplicates="drop",
    )

    if len(segment_edges) - 1 != n_quantile_bins:
        raise ValueError(
            f"Requested {n_quantile_bins} quantile bins, but only {len(segment_edges) - 1} unique bins could be formed."
        )

    if segment_ids.isna().any():
        raise ValueError("Failed to assign intended_fix quantile segments to some trials.")

    segmented_df = df.copy()
    segmented_df["intended_fix_segment"] = segment_ids.astype(int)
    return segmented_df, np.asarray(segment_edges, dtype=float)


def apply_intended_fix_upper_bound(df: pd.DataFrame, intended_fix_max: float | None) -> pd.DataFrame:
    if intended_fix_max is None:
        return df.copy()
    return df[df["intended_fix"] <= intended_fix_max].copy()


def format_segment_label(segment_idx: int, segment_edges: np.ndarray, n_segments: int) -> str:
    left_edge = segment_edges[segment_idx]
    right_edge = segment_edges[segment_idx + 1]
    return f"intended_fix Q{segment_idx + 1}/{n_segments} [{left_edge:.3f}, {right_edge:.3f}] s"


def get_batch_animal_pairs(df: pd.DataFrame) -> list[tuple[str, str]]:
    pairs_df = df[["batch_name", "animal"]].drop_duplicates().copy()
    pairs_df["batch_name"] = pairs_df["batch_name"].astype(str)
    pairs_df["animal"] = pairs_df["animal"].astype(str)
    return sorted(list(map(tuple, pairs_df[["batch_name", "animal"]].values)))


def print_batch_animal_table(batch_animal_pairs: list[tuple[str, str]]) -> None:
    print(
        f"Found {len(batch_animal_pairs)} batch-animal pairs from "
        f"{len(set(pair[0] for pair in batch_animal_pairs))} batches:"
    )

    if not batch_animal_pairs:
        return

    batch_to_animals = defaultdict(list)
    for batch, animal in batch_animal_pairs:
        if animal not in batch_to_animals[batch]:
            batch_to_animals[batch].append(animal)

    max_batch_len = max(len(batch) for batch in batch_to_animals.keys()) if batch_to_animals else 0
    animal_strings = {batch: ", ".join(sorted(animals)) for batch, animals in batch_to_animals.items()}
    max_animals_len = max(len(animal_string) for animal_string in animal_strings.values()) if animal_strings else 0

    print(f"{'Batch':<{max_batch_len}}  {'Animals'}")
    print(f"{'=' * max_batch_len}  {'=' * max_animals_len}")

    for batch in sorted(animal_strings.keys()):
        print(f"{batch:<{max_batch_len}}  {animal_strings[batch]}")


def load_merged_valid_trials() -> pd.DataFrame:
    batch_files = []
    missing_batch_files = []

    for batch_name in DESIRED_BATCHES:
        batch_file = batch_csv_dir / f"batch_{batch_name}_valid_and_aborts.csv"
        if batch_file.exists():
            batch_files.append(batch_file)
        else:
            missing_batch_files.append(batch_file.name)

    if not batch_files:
        raise FileNotFoundError(
            f"Could not find any batch CSVs in {batch_csv_dir} for DESIRED_BATCHES={DESIRED_BATCHES}"
        )

    merged_data = pd.concat([pd.read_csv(batch_file) for batch_file in batch_files], ignore_index=True)
    required_columns = ["batch_name", "animal", "success", "RTwrtStim", "ABL", "ILD", "intended_fix"]
    if TRIAL_POOL_MODE == "valid_plus_abort3":
        required_columns.append("abort_event")
    validate_required_columns(merged_data, required_columns)

    numeric_columns = ["RTwrtStim", "ABL", "ILD", "intended_fix"]
    if TRIAL_POOL_MODE == "valid_plus_abort3":
        numeric_columns.append("abort_event")
    for column_name in numeric_columns:
        merged_data[column_name] = pd.to_numeric(merged_data[column_name], errors="coerce")

    if TRIAL_POOL_MODE == "valid_plus_abort3":
        merged_valid = merged_data[
            merged_data["success"].isin([1, -1]) | np.isclose(merged_data["abort_event"], 3)
        ].copy()
    else:
        merged_valid = merged_data[merged_data["success"].isin([1, -1])].copy()
    merged_valid["abs_ILD"] = merged_valid["ILD"].abs()

    if missing_batch_files:
        print(f"Skipped missing batch files: {missing_batch_files}")

    discovered_pairs = get_batch_animal_pairs(merged_valid)
    print_batch_animal_table(discovered_pairs)

    if EXCLUDED_BATCH_ANIMAL_PAIRS:
        excluded_pairs = {(str(batch), str(animal)) for batch, animal in EXCLUDED_BATCH_ANIMAL_PAIRS}
        batch_animal_keys = list(zip(merged_valid["batch_name"].astype(str), merged_valid["animal"].astype(str)))
        merged_valid = merged_valid[[key not in excluded_pairs for key in batch_animal_keys]].copy()
        print(f"Excluded batch-animal pairs: {sorted(excluded_pairs)}")

    return merged_valid


def load_merged_tachometric_trials() -> pd.DataFrame:
    batch_files = []
    missing_batch_files = []

    for batch_name in DESIRED_BATCHES:
        batch_file_with_4 = batch_csv_dir / f"batch_{batch_name}_valid_and_aborts_and_4.csv"
        batch_file = batch_csv_dir / f"batch_{batch_name}_valid_and_aborts.csv"
        if batch_file_with_4.exists():
            batch_files.append(batch_file_with_4)
        elif batch_file.exists():
            batch_files.append(batch_file)
        else:
            missing_batch_files.append(batch_file_with_4.name)

    if not batch_files:
        raise FileNotFoundError(
            f"Could not find any batch CSVs in {batch_csv_dir} for DESIRED_BATCHES={DESIRED_BATCHES}"
        )

    merged_data = pd.concat([pd.read_csv(batch_file) for batch_file in batch_files], ignore_index=True)
    validate_required_columns(
        merged_data,
        ["batch_name", "animal", "success", "abort_event", "choice", "RTwrtStim", "ABL", "ILD", "intended_fix"],
    )

    for column_name in ["success", "abort_event", "choice", "RTwrtStim", "ABL", "ILD", "intended_fix"]:
        merged_data[column_name] = pd.to_numeric(merged_data[column_name], errors="coerce")

    merged_tachometric = merged_data[
        merged_data["success"].isin([1, -1])
        | np.isclose(merged_data["abort_event"], 4)
        | np.isclose(merged_data["abort_event"], 3)
    ].copy()

    abort_event_3_mask = np.isclose(merged_tachometric["abort_event"], 3)
    if abort_event_3_mask.any():
        rng = np.random.default_rng(tachometric_random_choice_seed)
        merged_tachometric.loc[abort_event_3_mask, "choice"] = rng.choice(
            [-1, 1],
            size=int(abort_event_3_mask.sum()),
        )

    merged_tachometric["abs_ILD"] = merged_tachometric["ILD"].abs()

    correct_mask = (
        ((merged_tachometric["ILD"] > 0) & np.isclose(merged_tachometric["choice"], 1))
        | ((merged_tachometric["ILD"] < 0) & np.isclose(merged_tachometric["choice"], -1))
    )
    merged_tachometric["derived_success"] = correct_mask.astype(int)

    if missing_batch_files:
        print(f"Skipped missing tachometric batch files: {missing_batch_files}")

    discovered_pairs = get_batch_animal_pairs(merged_tachometric)
    print_batch_animal_table(discovered_pairs)

    if EXCLUDED_BATCH_ANIMAL_PAIRS:
        excluded_pairs = {(str(batch), str(animal)) for batch, animal in EXCLUDED_BATCH_ANIMAL_PAIRS}
        batch_animal_keys = list(
            zip(merged_tachometric["batch_name"].astype(str), merged_tachometric["animal"].astype(str))
        )
        merged_tachometric = merged_tachometric[[key not in excluded_pairs for key in batch_animal_keys]].copy()
        print(f"Excluded batch-animal pairs from tachometric data: {sorted(excluded_pairs)}")

    return merged_tachometric


def prepare_plot_df(valid_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    mask_rt = (valid_df["RTwrtStim"] >= rt_min_s) & (valid_df["RTwrtStim"] < rt_max_s)
    mask_abl = valid_df["ABL"].isin(supported_abl_values)
    mask_abs_ild = valid_df["abs_ILD"].isin(supported_abs_ild_values)
    mask_intended_fix = valid_df["intended_fix"].notna()

    plot_df = valid_df[mask_rt & mask_abl & mask_abs_ild & mask_intended_fix].copy()
    plot_df = apply_intended_fix_upper_bound(plot_df, intended_fix_max_s)

    if len(plot_df) == 0:
        raise ValueError("No RTD/CDF plot trials found after filtering.")

    validate_supported_values(plot_df, "ABL", supported_abl_values)
    validate_supported_values(plot_df, "abs_ILD", supported_abs_ild_values)
    plot_df, intended_fix_segment_edges = add_intended_fix_quantile_segments(
        plot_df,
        num_intended_fix_quantile_bins,
    )
    return plot_df, intended_fix_segment_edges


def prepare_tachometric_df(tachometric_df: pd.DataFrame, segment_edges: np.ndarray) -> pd.DataFrame:
    mask_rt = (tachometric_df["RTwrtStim"] >= rt_min_s) & (tachometric_df["RTwrtStim"] < rt_max_s)
    mask_abl = tachometric_df["ABL"].isin(supported_abl_values)
    mask_abs_ild = tachometric_df["abs_ILD"].isin(supported_abs_ild_values)
    mask_intended_fix = tachometric_df["intended_fix"].notna()
    mask_choice = tachometric_df["choice"].isin([-1, 1])

    prepared_df = tachometric_df[mask_rt & mask_abl & mask_abs_ild & mask_intended_fix & mask_choice].copy()
    prepared_df = apply_intended_fix_upper_bound(prepared_df, intended_fix_max_s)

    if len(prepared_df) == 0:
        raise ValueError("No tachometric trials found after filtering.")

    validate_supported_values(prepared_df, "ABL", supported_abl_values)
    validate_supported_values(prepared_df, "abs_ILD", supported_abs_ild_values)

    segment_cut_edges = np.asarray(segment_edges, dtype=float).copy()
    segment_cut_edges[0] = np.nextafter(segment_cut_edges[0], -np.inf)
    segment_cut_edges[-1] = np.nextafter(segment_cut_edges[-1], np.inf)
    clipped_intended_fix = prepared_df["intended_fix"].clip(
        lower=float(segment_edges[0]),
        upper=float(segment_edges[-1]),
    )

    segment_ids = pd.cut(
        clipped_intended_fix,
        bins=segment_cut_edges,
        include_lowest=True,
        labels=False,
        duplicates="raise",
    )
    if segment_ids.isna().any():
        raise ValueError("Failed to assign intended_fix quantile segments to some tachometric trials.")

    prepared_df["intended_fix_segment"] = segment_ids.astype(int)
    return prepared_df


def make_condition_plot(
    df: pd.DataFrame,
    bins_s: np.ndarray,
    y_mode: str,
    colors_by_abs_ild: dict[int, str],
    segment_edges: np.ndarray,
) -> plt.Figure:
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
        segment_label = format_segment_label(segment_idx, segment_edges, n_segments)
        segment_total = int(len(segment_df))

        for col_idx, abl_value in enumerate(supported_abl_values):
            ax = axes[segment_idx, col_idx]
            abl_df = segment_df[np.isclose(segment_df["ABL"], abl_value)].copy()
            abl_total = int(len(abl_df))

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

            ax.set_title(f"ABL = {abl_value}\nn={abl_total}")
            ax.set_xlim(*xlim_ms)
            ax.grid(alpha=0.2, linewidth=0.6)
            if segment_idx == n_segments - 1:
                ax.set_xlabel("RT wrt stim (ms)")

            if col_idx == 0:
                if y_mode == "rtd":
                    ax.set_ylabel(f"{ylabel_rtd}\n{segment_label}\nsegment n={segment_total}")
                else:
                    ax.set_ylabel(f"{ylabel_cdf}\n{segment_label}\nsegment n={segment_total}")

            if y_mode == "rtd" and rtd_ylim is not None:
                ax.set_ylim(*rtd_ylim)
            if y_mode == "cdf" and cdf_ylim is not None:
                ax.set_ylim(*cdf_ylim)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(supported_abs_ild_values), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def make_abl_collapsed_plot(
    df: pd.DataFrame,
    bins_s: np.ndarray,
    y_mode: str,
    colors_by_abl: dict[int, str],
    segment_edges: np.ndarray,
) -> plt.Figure:
    if y_mode not in {"rtd", "cdf"}:
        raise ValueError(f"Unsupported y_mode: {y_mode}")

    n_segments = len(segment_edges) - 1
    fig, axes = plt.subplots(
        n_segments,
        1,
        figsize=(panel_width, panel_height * n_segments),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    x_edges_ms = bins_s * 1e3

    for segment_idx in range(n_segments):
        ax = axes[segment_idx, 0]
        segment_df = df[df["intended_fix_segment"] == segment_idx].copy()
        segment_label = format_segment_label(segment_idx, segment_edges, n_segments)
        segment_counts_by_abl = {
            abl_value: int(np.isclose(segment_df["ABL"], abl_value).sum())
            for abl_value in supported_abl_values
        }
        segment_total = int(len(segment_df))

        for abl_value in supported_abl_values:
            abl_df = segment_df[np.isclose(segment_df["ABL"], abl_value)].copy()
            values = abl_df["RTwrtStim"].to_numpy()

            if y_mode == "rtd":
                y_values = compute_density_histogram(values, bins_s)
            else:
                y_values = compute_binned_cdf(values, bins_s)

            ax.stairs(
                y_values,
                x_edges_ms,
                label=f"ABL = {abl_value}",
                color=colors_by_abl[abl_value],
                linewidth=1.8,
            )

        ax.set_xlim(*xlim_ms)
        ax.grid(alpha=0.2, linewidth=0.6)
        ax.set_title(
            f"{segment_label}\n"
            f"n20={segment_counts_by_abl[20]}, "
            f"n40={segment_counts_by_abl[40]}, "
            f"n60={segment_counts_by_abl[60]}, "
            f"total={segment_total}"
        )

        if segment_idx == n_segments - 1:
            ax.set_xlabel("RT wrt stim (ms)")

        if y_mode == "rtd":
            ax.set_ylabel(ylabel_rtd)
        else:
            ax.set_ylabel(ylabel_cdf)

        if y_mode == "rtd" and rtd_ylim is not None:
            ax.set_ylim(*rtd_ylim)
        if y_mode == "cdf" and cdf_ylim is not None:
            ax.set_ylim(*cdf_ylim)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(supported_abl_values), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def make_abl_segment_overlay_rtd_plot(
    df: pd.DataFrame,
    bins_s: np.ndarray,
    colors_by_abl: dict[int, str],
    segment_edges: np.ndarray,
) -> plt.Figure:
    n_segments = len(segment_edges) - 1
    fig, axes = plt.subplots(
        1,
        len(supported_abl_values),
        figsize=(panel_width * len(supported_abl_values), panel_height),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    x_edges_ms = bins_s * 1e3
    segment_colors = ["tab:blue", "tab:red", "tab:green", "tab:purple"]

    for col_idx, abl_value in enumerate(supported_abl_values):
        ax = axes[0, col_idx]
        abl_df = df[np.isclose(df["ABL"], abl_value)].copy()
        segment_count_labels = []

        for segment_idx in range(n_segments):
            segment_df = abl_df[abl_df["intended_fix_segment"] == segment_idx].copy()
            values = segment_df["RTwrtStim"].to_numpy()
            y_values = compute_density_histogram(values, bins_s)
            line_alpha = 1.0 if segment_idx == 0 else 0.5

            if n_segments == 2:
                segment_name = "Early stim" if segment_idx == 0 else "Late stim"
                segment_count_name = "early" if segment_idx == 0 else "late"
            else:
                segment_name = f"Stim seg {segment_idx + 1}/{n_segments}"
                segment_count_name = f"seg{segment_idx + 1}"

            segment_count_labels.append(f"{segment_count_name}={len(segment_df)}")

            ax.stairs(
                y_values,
                x_edges_ms,
                label=(
                    f"{segment_name} "
                    f"[{segment_edges[segment_idx]:.3f}, {segment_edges[segment_idx + 1]:.3f}] s"
                ),
                color=segment_colors[segment_idx % len(segment_colors)],
                linewidth=1.8,
                linestyle="-",
                alpha=line_alpha,
            )

        ax.set_title(f"ABL = {abl_value}\n{', '.join(segment_count_labels)}, total={len(abl_df)}")
        ax.set_xlim(*xlim_ms)
        ax.grid(alpha=0.2, linewidth=0.6)
        ax.set_xlabel("RT wrt stim (ms)")

        if col_idx == 0:
            ax.set_ylabel(ylabel_rtd)

        if rtd_ylim is not None:
            ax.set_ylim(*rtd_ylim)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(n_segments, 4), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def make_abl_segment_overlay_cdf_plot(
    df: pd.DataFrame,
    bins_s: np.ndarray,
    segment_edges: np.ndarray,
) -> plt.Figure:
    n_segments = len(segment_edges) - 1
    fig, axes = plt.subplots(
        1,
        len(supported_abl_values),
        figsize=(panel_width * len(supported_abl_values), panel_height),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    x_edges_ms = bins_s * 1e3
    segment_colors = ["tab:blue", "tab:red", "tab:green", "tab:purple"]

    for col_idx, abl_value in enumerate(supported_abl_values):
        ax = axes[0, col_idx]
        abl_df = df[np.isclose(df["ABL"], abl_value)].copy()
        segment_count_labels = []

        for segment_idx in range(n_segments):
            segment_df = abl_df[abl_df["intended_fix_segment"] == segment_idx].copy()
            values = segment_df["RTwrtStim"].to_numpy()
            y_values = compute_binned_cdf(values, bins_s)

            if n_segments == 2:
                segment_name = "Early stim" if segment_idx == 0 else "Late stim"
                segment_count_name = "early" if segment_idx == 0 else "late"
            else:
                segment_name = f"Stim seg {segment_idx + 1}/{n_segments}"
                segment_count_name = f"seg{segment_idx + 1}"

            segment_count_labels.append(f"{segment_count_name}={len(segment_df)}")

            ax.stairs(
                y_values,
                x_edges_ms,
                label=(
                    f"{segment_name} "
                    f"[{segment_edges[segment_idx]:.3f}, {segment_edges[segment_idx + 1]:.3f}] s"
                ),
                color=segment_colors[segment_idx % len(segment_colors)],
                linewidth=1.8,
                linestyle="-",
            )

        ax.set_title(f"ABL = {abl_value}\n{', '.join(segment_count_labels)}, total={len(abl_df)}")
        ax.set_xlim(*xlim_ms)
        ax.grid(alpha=0.2, linewidth=0.6)
        ax.set_xlabel("RT wrt stim (ms)")

        if col_idx == 0:
            ax.set_ylabel(ylabel_cdf)

        if cdf_ylim is not None:
            ax.set_ylim(*cdf_ylim)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(n_segments, 4), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def make_condition_segment_overlay_plot(
    df: pd.DataFrame,
    bins_s: np.ndarray,
    y_mode: str,
    colors_by_abs_ild: dict[int, str],
    segment_edges: np.ndarray,
) -> plt.Figure:
    if y_mode not in {"rtd", "cdf"}:
        raise ValueError(f"Unsupported y_mode: {y_mode}")

    n_segments = len(segment_edges) - 1
    fig, axes = plt.subplots(
        1,
        len(supported_abl_values),
        figsize=(panel_width * len(supported_abl_values), panel_height),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    x_edges_ms = bins_s * 1e3
    segment_linestyles = [":", "-", "--", "-."]
    segment_colors = ["tab:blue", "tab:red", "tab:green", "tab:purple"]

    for col_idx, abl_value in enumerate(supported_abl_values):
        ax = axes[0, col_idx]
        abl_df = df[np.isclose(df["ABL"], abl_value)].copy()
        segment_count_labels = []

        for segment_idx in range(n_segments):
            segment_df = abl_df[abl_df["intended_fix_segment"] == segment_idx].copy()
            line_alpha = 1.0 if segment_idx == 0 else 0.5

            if n_segments == 2:
                segment_count_name = "early" if segment_idx == 0 else "late"
            else:
                segment_count_name = f"seg{segment_idx + 1}"
            segment_count_labels.append(f"{segment_count_name}={len(segment_df)}")

            for abs_ild_value in supported_abs_ild_values:
                condition_df = segment_df[np.isclose(segment_df["abs_ILD"], abs_ild_value)].copy()
                values = condition_df["RTwrtStim"].to_numpy()

                if y_mode == "rtd":
                    y_values = compute_density_histogram(values, bins_s)
                else:
                    y_values = compute_binned_cdf(values, bins_s)

                ax.stairs(
                    y_values,
                    x_edges_ms,
                    color=colors_by_abs_ild[abs_ild_value],
                    linewidth=1.8,
                    linestyle=segment_linestyles[segment_idx % len(segment_linestyles)],
                    alpha=line_alpha,
                )

        ax.set_title(f"ABL = {abl_value}\n{', '.join(segment_count_labels)}")
        ax.set_xlim(*xlim_ms)
        ax.grid(alpha=0.2, linewidth=0.6)
        ax.set_xlabel("RT wrt stim (ms)")

        if col_idx == 0:
            if y_mode == "rtd":
                ax.set_ylabel(ylabel_rtd)
            else:
                ax.set_ylabel(ylabel_cdf)

        if y_mode == "rtd" and rtd_ylim is not None:
            ax.set_ylim(*rtd_ylim)
        if y_mode == "cdf" and cdf_ylim is not None:
            ax.set_ylim(*cdf_ylim)

    abs_ild_handles = [
        Line2D([0], [0], color=colors_by_abs_ild[abs_ild_value], linewidth=1.8, linestyle="-")
        for abs_ild_value in supported_abs_ild_values
    ]
    abs_ild_labels = [f"|ILD| = {abs_ild_value}" for abs_ild_value in supported_abs_ild_values]
    segment_handles = [
        Line2D([0], [0], color=segment_colors[segment_idx], linewidth=1.8, linestyle="-")
        for segment_idx in range(min(n_segments, len(segment_colors)))
    ]
    if n_segments == 2:
        segment_labels = ["Early stim", "Late stim"]
    else:
        segment_labels = [f"Stim seg {segment_idx + 1}/{n_segments}" for segment_idx in range(min(n_segments, len(segment_colors)))]

    color_legend = fig.legend(
        abs_ild_handles,
        abs_ild_labels,
        loc="upper center",
        ncol=len(supported_abs_ild_values),
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.add_artist(color_legend)
    fig.legend(
        segment_handles,
        segment_labels,
        loc="upper center",
        ncol=len(segment_labels),
        frameon=False,
        bbox_to_anchor=(0.5, 0.96),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig


def make_condition_grid_segment_overlay_plot(
    df: pd.DataFrame,
    bins_s: np.ndarray,
    y_mode: str,
    segment_edges: np.ndarray,
) -> plt.Figure:
    if y_mode not in {"rtd", "cdf"}:
        raise ValueError(f"Unsupported y_mode: {y_mode}")

    n_segments = len(segment_edges) - 1
    fig, axes = plt.subplots(
        len(supported_abs_ild_values),
        len(supported_abl_values),
        figsize=(panel_width * len(supported_abl_values), panel_height * len(supported_abs_ild_values)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    x_edges_ms = bins_s * 1e3
    segment_colors = ["tab:blue", "tab:red", "tab:green", "tab:purple"]

    for row_idx, abs_ild_value in enumerate(supported_abs_ild_values):
        for col_idx, abl_value in enumerate(supported_abl_values):
            ax = axes[row_idx, col_idx]
            condition_df = df[
                np.isclose(df["abs_ILD"], abs_ild_value) & np.isclose(df["ABL"], abl_value)
            ].copy()
            segment_count_labels = []

            for segment_idx in range(n_segments):
                segment_df = condition_df[condition_df["intended_fix_segment"] == segment_idx].copy()
                values = segment_df["RTwrtStim"].to_numpy()

                if n_segments == 2:
                    segment_name = "Early stim" if segment_idx == 0 else "Late stim"
                    segment_count_name = "early" if segment_idx == 0 else "late"
                else:
                    segment_name = f"Stim seg {segment_idx + 1}/{n_segments}"
                    segment_count_name = f"seg{segment_idx + 1}"

                segment_count_labels.append(f"{segment_count_name}={len(segment_df)}")

                if y_mode == "rtd":
                    y_values = compute_density_histogram(values, bins_s)
                else:
                    y_values = compute_binned_cdf(values, bins_s)

                ax.stairs(
                    y_values,
                    x_edges_ms,
                    color=segment_colors[segment_idx % len(segment_colors)],
                    linewidth=1.8,
                    linestyle="-",
                    label=segment_name,
                )

            ax.set_title(f"ABL = {abl_value}\n{', '.join(segment_count_labels)}")
            ax.set_xlim(*xlim_ms)
            ax.grid(alpha=0.2, linewidth=0.6)

            if row_idx == len(supported_abs_ild_values) - 1:
                ax.set_xlabel("RT wrt stim (ms)")

            if col_idx == 0:
                if y_mode == "rtd":
                    ax.set_ylabel(f"{ylabel_rtd}\n|ILD| = {abs_ild_value}")
                else:
                    ax.set_ylabel(f"{ylabel_cdf}\n|ILD| = {abs_ild_value}")

            if y_mode == "rtd" and rtd_ylim is not None:
                ax.set_ylim(*rtd_ylim)
            if y_mode == "cdf" and cdf_ylim is not None:
                ax.set_ylim(*cdf_ylim)

    segment_handles = [
        Line2D([0], [0], color=segment_colors[segment_idx], linewidth=1.8, linestyle="-")
        for segment_idx in range(min(n_segments, len(segment_colors)))
    ]
    if n_segments == 2:
        segment_labels = ["Early stim", "Late stim"]
    else:
        segment_labels = [f"Stim seg {segment_idx + 1}/{n_segments}" for segment_idx in range(min(n_segments, len(segment_colors)))]

    fig.legend(
        segment_handles,
        segment_labels,
        loc="upper center",
        ncol=len(segment_labels),
        frameon=False,
        bbox_to_anchor=(0.5, 0.99),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def make_tachometric_plot(
    df: pd.DataFrame,
    bins_s: np.ndarray,
    colors_by_abl: dict[int, str],
    segment_edges: np.ndarray,
) -> plt.Figure:
    n_segments = len(segment_edges) - 1
    fig, axes = plt.subplots(
        n_segments,
        1,
        figsize=(panel_width, panel_height * n_segments),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    bin_centers_ms = (bins_s[:-1] + 0.5 * np.diff(bins_s)) * 1e3

    for segment_idx in range(n_segments):
        ax = axes[segment_idx, 0]
        segment_df = df[df["intended_fix_segment"] == segment_idx].copy()
        segment_label = format_segment_label(segment_idx, segment_edges, n_segments)
        segment_counts_by_abl = {
            abl_value: int(np.isclose(segment_df["ABL"], abl_value).sum())
            for abl_value in supported_abl_values
        }
        segment_total = int(len(segment_df))

        for abl_value in supported_abl_values:
            abl_df = segment_df[np.isclose(segment_df["ABL"], abl_value)].copy()
            rt_values = abl_df["RTwrtStim"].to_numpy()
            correct_values = abl_df["derived_success"].to_numpy()
            accuracy = compute_binned_accuracy(rt_values, correct_values, bins_s)

            ax.plot(
                bin_centers_ms,
                accuracy,
                label=f"ABL = {abl_value}",
                color=colors_by_abl[abl_value],
                linewidth=1.8,
                marker="o",
                markersize=3,
            )

        ax.set_xlim(*xlim_ms)
        ax.grid(alpha=0.2, linewidth=0.6)
        ax.set_title(
            f"{segment_label}\n"
            f"n20={segment_counts_by_abl[20]}, "
            f"n40={segment_counts_by_abl[40]}, "
            f"n60={segment_counts_by_abl[60]}, "
            f"total={segment_total}"
        )

        if segment_idx == n_segments - 1:
            ax.set_xlabel("RT wrt stim (ms)")

        ax.set_ylabel(ylabel_tacho)

        if tacho_ylim is not None:
            ax.set_ylim(*tacho_ylim)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(supported_abl_values), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def make_tachometric_plot_by_abs_ild(
    df: pd.DataFrame,
    bins_s: np.ndarray,
    colors_by_abs_ild: dict[int, str],
    segment_edges: np.ndarray,
) -> plt.Figure:
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
    bin_centers_ms = (bins_s[:-1] + 0.5 * np.diff(bins_s)) * 1e3

    for segment_idx in range(n_segments):
        segment_df = df[df["intended_fix_segment"] == segment_idx].copy()
        segment_label = format_segment_label(segment_idx, segment_edges, n_segments)
        segment_total = int(len(segment_df))

        for col_idx, abl_value in enumerate(supported_abl_values):
            ax = axes[segment_idx, col_idx]
            abl_df = segment_df[np.isclose(segment_df["ABL"], abl_value)].copy()
            abl_total = int(len(abl_df))

            for abs_ild_value in supported_abs_ild_values:
                condition_df = abl_df[np.isclose(abl_df["abs_ILD"], abs_ild_value)].copy()
                rt_values = condition_df["RTwrtStim"].to_numpy()
                correct_values = condition_df["derived_success"].to_numpy()
                accuracy = compute_binned_accuracy(rt_values, correct_values, bins_s)

                ax.plot(
                    bin_centers_ms,
                    accuracy,
                    label=f"|ILD| = {abs_ild_value}",
                    color=colors_by_abs_ild[abs_ild_value],
                    linewidth=1.8,
                    marker="o",
                    markersize=3,
                )

            ax.set_title(f"ABL = {abl_value}\nn={abl_total}")
            ax.set_xlim(*xlim_ms)
            ax.grid(alpha=0.2, linewidth=0.6)

            if segment_idx == n_segments - 1:
                ax.set_xlabel("RT wrt stim (ms)")

            if col_idx == 0:
                ax.set_ylabel(f"{ylabel_tacho}\n{segment_label}\nsegment n={segment_total}")

            if tacho_ylim is not None:
                ax.set_ylim(*tacho_ylim)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(supported_abs_ild_values), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


# %%
output_dir.mkdir(parents=True, exist_ok=True)

merged_valid = load_merged_valid_trials()
plot_df, intended_fix_segment_edges = prepare_plot_df(merged_valid)
merged_tachometric = load_merged_tachometric_trials()
tacho_df = prepare_tachometric_df(merged_tachometric, intended_fix_segment_edges)

filtered_batch_animal_pairs = get_batch_animal_pairs(plot_df)
batch_counts = plot_df["batch_name"].astype(str).value_counts().sort_index().to_dict()
abl_counts = format_counts(plot_df["ABL"])
abs_ild_counts = format_counts(plot_df["abs_ILD"])
condition_counts = build_condition_counts(plot_df)
segment_counts = plot_df["intended_fix_segment"].value_counts().sort_index().to_dict()
segment_condition_counts = build_segment_condition_counts(plot_df)
batch_animal_counts = (
    plot_df.assign(
        batch_name=plot_df["batch_name"].astype(str),
        animal=plot_df["animal"].astype(str),
    )
    .groupby(["batch_name", "animal"])
    .size()
    .rename("n_trials")
)

plot_trial_label = "valid+abort3" if TRIAL_POOL_MODE == "valid_plus_abort3" else "valid-only"

print(f"Rebuilt multi-animal {plot_trial_label} dataset for quantile-segment RTD/CDF plots:")
print(f"  Batch CSV dir: {batch_csv_dir}")
print(f"  Loaded batches: {sorted(plot_df['batch_name'].astype(str).unique().tolist())}")
print(f"  Total filtered RTD/CDF plot trials: {len(plot_df)}")
print(f"  Filtered batch-animal pairs: {len(filtered_batch_animal_pairs)}")
print(f"  Counts by batch: {batch_counts}")
print(f"  Counts by ABL: {abl_counts}")
print(f"  Counts by abs_ILD: {abs_ild_counts}")
print(f"  TRIAL_POOL_MODE: {TRIAL_POOL_MODE}")
print(
    "  intended_fix upper bound (s): "
    f"{float(plot_df['intended_fix'].max()) if intended_fix_max_s is None else intended_fix_max_s}"
)
print(f"  intended_fix quantile bin count: {num_intended_fix_quantile_bins}")
print(f"  intended_fix quantile edges (s): {[float(edge) for edge in intended_fix_segment_edges]}")
print(f"  Counts by intended_fix quantile segment: {segment_counts}")
print("  Counts by batch x animal:")
print(batch_animal_counts.to_string())
print("  Counts by ABL x abs_ILD:")
print(condition_counts.to_string())
print("  Counts by intended_fix quantile segment x ABL x abs_ILD:")
print(segment_condition_counts.to_string())


# %%
rt_bins_s = np.arange(rt_min_s, rt_max_s + bin_size_s, bin_size_s)
tacho_bins_s = np.arange(rt_min_s, rt_max_s + tachometric_bin_size_s, tachometric_bin_size_s)
abs_ild_colors = {
    1: "tab:blue",
    2: "tab:orange",
    4: "tab:green",
    8: "tab:red",
    16: "tab:purple",
}
abl_colors = {
    20: "tab:blue",
    40: "tab:orange",
    60: "tab:green",
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
rtd_all_ild_fig = make_abl_collapsed_plot(
    plot_df,
    bins_s=rt_bins_s,
    y_mode="rtd",
    colors_by_abl=abl_colors,
    segment_edges=intended_fix_segment_edges,
)
rtd_all_ild_segment_overlay_fig = make_abl_segment_overlay_rtd_plot(
    plot_df,
    bins_s=rt_bins_s,
    colors_by_abl=abl_colors,
    segment_edges=intended_fix_segment_edges,
)
cdf_all_ild_fig = make_abl_collapsed_plot(
    plot_df,
    bins_s=rt_bins_s,
    y_mode="cdf",
    colors_by_abl=abl_colors,
    segment_edges=intended_fix_segment_edges,
)
cdf_segment_overlay_fig = make_condition_segment_overlay_plot(
    plot_df,
    bins_s=rt_bins_s,
    y_mode="cdf",
    colors_by_abs_ild=abs_ild_colors,
    segment_edges=intended_fix_segment_edges,
)
cdf_abl_segment_overlay_fig = make_abl_segment_overlay_cdf_plot(
    plot_df,
    bins_s=rt_bins_s,
    segment_edges=intended_fix_segment_edges,
)
rtd_condition_grid_segment_overlay_fig = make_condition_grid_segment_overlay_plot(
    plot_df,
    bins_s=rt_bins_s,
    y_mode="rtd",
    segment_edges=intended_fix_segment_edges,
)
tacho_all_ild_fig = make_tachometric_plot(
    tacho_df,
    bins_s=tacho_bins_s,
    colors_by_abl=abl_colors,
    segment_edges=intended_fix_segment_edges,
)
tacho_abs_ild_fig = make_tachometric_plot_by_abs_ild(
    tacho_df,
    bins_s=tacho_bins_s,
    colors_by_abs_ild=abs_ild_colors,
    segment_edges=intended_fix_segment_edges,
)

rtd_output_base = output_dir / rtd_plot_base
cdf_output_base = output_dir / cdf_plot_base
rtd_output_base_all_ild = output_dir / rtd_plot_base_all_ild
rtd_output_base_all_ild_segment_overlay = output_dir / rtd_plot_base_all_ild_segment_overlay
rtd_output_base_condition_grid_segment_overlay = output_dir / rtd_plot_base_condition_grid_segment_overlay
cdf_output_base_all_ild = output_dir / cdf_plot_base_all_ild
cdf_output_base_segment_overlay = output_dir / cdf_plot_base_segment_overlay
cdf_output_base_abl_segment_overlay = output_dir / cdf_plot_base_abl_segment_overlay
tacho_output_base_all_ild = output_dir / tacho_plot_base_all_ild
tacho_output_base_abs_ild = output_dir / tacho_plot_base_abs_ild

save_figure(rtd_fig, rtd_output_base)
save_figure(cdf_fig, cdf_output_base)
save_figure(rtd_all_ild_fig, rtd_output_base_all_ild)
save_figure(rtd_all_ild_segment_overlay_fig, rtd_output_base_all_ild_segment_overlay)
save_figure(rtd_condition_grid_segment_overlay_fig, rtd_output_base_condition_grid_segment_overlay)
save_figure(cdf_all_ild_fig, cdf_output_base_all_ild)
save_figure(cdf_segment_overlay_fig, cdf_output_base_segment_overlay)
save_figure(cdf_abl_segment_overlay_fig, cdf_output_base_abl_segment_overlay)
save_figure(tacho_all_ild_fig, tacho_output_base_all_ild)
save_figure(tacho_abs_ild_fig, tacho_output_base_abs_ild)

print("Saved RTD figure:")
print(f"  {rtd_output_base.with_suffix('.pdf')}")
print(f"  {rtd_output_base.with_suffix('.png')}")
print("Saved CDF figure:")
print(f"  {cdf_output_base.with_suffix('.pdf')}")
print(f"  {cdf_output_base.with_suffix('.png')}")
print("Saved RTD figure collapsed across ILD:")
print(f"  {rtd_output_base_all_ild.with_suffix('.pdf')}")
print(f"  {rtd_output_base_all_ild.with_suffix('.png')}")
print("Saved RTD figure with intended_fix segments overlaid within each ABL:")
print(f"  {rtd_output_base_all_ild_segment_overlay.with_suffix('.pdf')}")
print(f"  {rtd_output_base_all_ild_segment_overlay.with_suffix('.png')}")
print("Saved RTD figure with abs ILD rows, ABL columns, and intended_fix segment overlays:")
print(f"  {rtd_output_base_condition_grid_segment_overlay.with_suffix('.pdf')}")
print(f"  {rtd_output_base_condition_grid_segment_overlay.with_suffix('.png')}")
print("Saved CDF figure collapsed across ILD:")
print(f"  {cdf_output_base_all_ild.with_suffix('.pdf')}")
print(f"  {cdf_output_base_all_ild.with_suffix('.png')}")
print("Saved CDF figure with intended_fix segments overlaid within each ABL:")
print(f"  {cdf_output_base_segment_overlay.with_suffix('.pdf')}")
print(f"  {cdf_output_base_segment_overlay.with_suffix('.png')}")
print("Saved CDF figure by ABL with early/late segment overlay:")
print(f"  {cdf_output_base_abl_segment_overlay.with_suffix('.pdf')}")
print(f"  {cdf_output_base_abl_segment_overlay.with_suffix('.png')}")
print("Saved tachometric figure collapsed across ILD:")
print(f"  {tacho_output_base_all_ild.with_suffix('.pdf')}")
print(f"  {tacho_output_base_all_ild.with_suffix('.png')}")
print("Saved tachometric figure by abs ILD:")
print(f"  {tacho_output_base_abs_ild.with_suffix('.pdf')}")
print(f"  {tacho_output_base_abs_ild.with_suffix('.png')}")

if show_plot:
    plt.show()
else:
    plt.close(rtd_fig)
    plt.close(cdf_fig)
    plt.close(rtd_all_ild_fig)
    plt.close(rtd_all_ild_segment_overlay_fig)
    plt.close(rtd_condition_grid_segment_overlay_fig)
    plt.close(cdf_all_ild_fig)
    plt.close(cdf_segment_overlay_fig)
    plt.close(cdf_abl_segment_overlay_fig)
    plt.close(tacho_all_ild_fig)
    plt.close(tacho_abs_ild_fig)

# %%
