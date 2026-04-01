# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# %%
SHOW_PLOT = True

DESIRED_BATCHES = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]
# DESIRED_BATCHES = ["LED8"]
EXCLUDED_BATCH_ANIMAL_PAIRS = []
TRIAL_POOL_MODE = "valid_plus_abort3_and_4"  # "valid" or "valid_plus_abort3" or "valid_plus_abort3_and_4"
PROACTIVE_TRUNC_FIX_TIME_S = {"default": 0.3, "LED34_even": 0.15}
# Set to None to disable truncation entirely

num_intended_fix_quantile_bins = 2
supported_abl_values = (20, 40, 60)
supported_abs_ild_values = (1, 2, 4, 8, 16)
rt_min_s = -3.0
rt_max_s = 3.0
intended_fix_max_s = 1.5
stim_bin_size_s = 0.01
stim_xlim_s = (0.18, 1.5)
panel_width = 5.6
panel_height = 4.0
png_dpi = 300

UNFILTERED_COLOR = "tab:blue"
FILTERED_COLOR = "tab:red"
REMOVED_COLOR = "black"


if TRIAL_POOL_MODE not in {"valid", "valid_plus_abort3", "valid_plus_abort3_and_4"}:
    raise ValueError(f"Unsupported TRIAL_POOL_MODE: {TRIAL_POOL_MODE}")
if num_intended_fix_quantile_bins != 2:
    raise ValueError("This script expects num_intended_fix_quantile_bins == 2 for a 1 x 2 plot.")


def get_trunc_time(batch_name):
    if PROACTIVE_TRUNC_FIX_TIME_S is None:
        return None
    return PROACTIVE_TRUNC_FIX_TIME_S.get(
        str(batch_name), PROACTIVE_TRUNC_FIX_TIME_S["default"]
    )


def get_trial_pool_abort_events() -> list[int]:
    if TRIAL_POOL_MODE == "valid":
        return []
    if TRIAL_POOL_MODE == "valid_plus_abort3":
        return [3]
    if TRIAL_POOL_MODE == "valid_plus_abort3_and_4":
        return [3, 4]
    raise ValueError(f"Unsupported TRIAL_POOL_MODE: {TRIAL_POOL_MODE}")


# %%
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

REPO_ROOT = SCRIPT_DIR.parent
batch_csv_dir = REPO_ROOT / "fit_animal_by_animal" / "batch_csvs"
output_dir = SCRIPT_DIR / "stim_distribution_early_vs_late_filtered_vs_unfiltered"
output_base = output_dir / "stim_distribution_early_vs_late_filtered_vs_unfiltered"
removed_output_base = output_dir / "stim_distribution_removed_by_abort3_truncation"
overall_output_base = output_dir / "stim_distribution_before_vs_after_abort3_truncation"
cdf_output_base = output_dir / "stim_cdf_before_vs_after_abort3_truncation"
difference_output_base = output_dir / "stim_count_difference_filtered_minus_unfiltered"
removed_fraction_output_base = output_dir / "stim_removed_fraction_by_bin"


# %%
def validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def add_intended_fix_quantile_segments(
    df: pd.DataFrame, n_quantile_bins: int
) -> tuple[pd.DataFrame, np.ndarray]:
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


def apply_intended_fix_upper_bound(
    df: pd.DataFrame, intended_fix_max: float | None
) -> pd.DataFrame:
    if intended_fix_max is None:
        return df.copy()
    return df[df["intended_fix"] <= intended_fix_max].copy()


def compute_density_histogram(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.zeros(len(bins) - 1, dtype=float)
    hist, _ = np.histogram(values, bins=bins, density=False)
    return hist.astype(float)


def compute_empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    sorted_values = np.sort(values)
    cdf = np.arange(1, len(sorted_values) + 1, dtype=float)
    return sorted_values, cdf


def save_figure(fig: plt.Figure, output_base_path: Path) -> None:
    fig.savefig(output_base_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base_path.with_suffix(".png"), dpi=png_dpi, bbox_inches="tight")


def format_edge_list(edges: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(edge):.6f}" for edge in edges) + "]"


def format_case_label(case_label: str, edges: np.ndarray, count: int) -> str:
    return (
        f"{case_label} [{float(edges[0]):.3f}, {float(edges[1]):.3f}] s"
        f" (n={int(count)})"
    )


def filter_excluded_batch_animal_pairs(df: pd.DataFrame) -> pd.DataFrame:
    if not EXCLUDED_BATCH_ANIMAL_PAIRS:
        return df.copy()

    excluded_pairs = {(str(batch), str(animal)) for batch, animal in EXCLUDED_BATCH_ANIMAL_PAIRS}
    batch_animal_keys = list(zip(df["batch_name"].astype(str), df["animal"].astype(str)))
    return df[[key not in excluded_pairs for key in batch_animal_keys]].copy()


def load_merged_pooled_trials(apply_truncation: bool) -> pd.DataFrame:
    batch_files = []
    missing_batch_files = []

    for batch_name in DESIRED_BATCHES:
        batch_file_with_4 = batch_csv_dir / f"batch_{batch_name}_valid_and_aborts_and_4.csv"
        batch_file = batch_csv_dir / f"batch_{batch_name}_valid_and_aborts.csv"
        if TRIAL_POOL_MODE == "valid_plus_abort3_and_4" and batch_file_with_4.exists():
            batch_files.append(batch_file_with_4)
        elif batch_file.exists():
            batch_files.append(batch_file)
        else:
            missing_batch_files.append(batch_file.name)

    if not batch_files:
        raise FileNotFoundError(
            f"Could not find any batch CSVs in {batch_csv_dir} for DESIRED_BATCHES={DESIRED_BATCHES}"
        )

    merged_data = pd.concat(
        [pd.read_csv(batch_file) for batch_file in batch_files],
        ignore_index=True,
    )
    merged_data["source_row_id"] = np.arange(len(merged_data), dtype=int)

    required_columns = ["batch_name", "animal", "success", "RTwrtStim", "ABL", "ILD", "intended_fix"]
    abort_events = get_trial_pool_abort_events()
    if abort_events:
        required_columns.append("abort_event")
    if apply_truncation and 3 in abort_events:
        required_columns.append("TotalFixTime")
    validate_required_columns(merged_data, required_columns)

    numeric_columns = ["RTwrtStim", "ABL", "ILD", "intended_fix"]
    if abort_events:
        numeric_columns.append("abort_event")
    if apply_truncation and 3 in abort_events:
        numeric_columns.append("TotalFixTime")
    for column_name in numeric_columns:
        merged_data[column_name] = pd.to_numeric(merged_data[column_name], errors="coerce")

    if abort_events:
        abort_mask = np.column_stack(
            [np.isclose(merged_data["abort_event"], abort_event) for abort_event in abort_events]
        ).any(axis=1)
        pooled_df = merged_data[merged_data["success"].isin([1, -1]) | abort_mask].copy()
    else:
        pooled_df = merged_data[merged_data["success"].isin([1, -1])].copy()

    if apply_truncation and 3 in abort_events and PROACTIVE_TRUNC_FIX_TIME_S is not None:
        early_abort_mask = pd.Series(False, index=pooled_df.index)
        for batch_name in pooled_df["batch_name"].astype(str).unique():
            trunc_t = get_trunc_time(batch_name)
            if trunc_t is None:
                continue
            batch_mask = pooled_df["batch_name"].astype(str) == batch_name
            early_abort_mask |= (
                batch_mask
                & np.isclose(pooled_df["abort_event"], 3)
                & (pooled_df["TotalFixTime"] < trunc_t)
            )
        pooled_df = pooled_df[~early_abort_mask].copy()

    pooled_df["abs_ILD"] = pooled_df["ILD"].abs()
    pooled_df = filter_excluded_batch_animal_pairs(pooled_df)

    if missing_batch_files:
        print(f"Skipped missing batch files: {missing_batch_files}")

    return pooled_df


def prepare_plot_df(pooled_df: pd.DataFrame) -> pd.DataFrame:
    mask_rt = (pooled_df["RTwrtStim"] >= rt_min_s) & (pooled_df["RTwrtStim"] < rt_max_s)
    mask_abl = pooled_df["ABL"].isin(supported_abl_values)
    mask_abs_ild = pooled_df["abs_ILD"].isin(supported_abs_ild_values)
    mask_intended_fix = pooled_df["intended_fix"].notna()

    plot_df = pooled_df[mask_rt & mask_abl & mask_abs_ild & mask_intended_fix].copy()
    plot_df = apply_intended_fix_upper_bound(plot_df, intended_fix_max_s)

    if len(plot_df) == 0:
        raise ValueError("No pooled RTD/CDF plot trials found after filtering.")

    return plot_df


def build_case_result(case_name: str, color: str, apply_truncation: bool) -> dict:
    pooled_df = load_merged_pooled_trials(apply_truncation=apply_truncation)
    plot_df = prepare_plot_df(pooled_df)
    segmented_df, segment_edges = add_intended_fix_quantile_segments(
        plot_df,
        num_intended_fix_quantile_bins,
    )

    return {
        "case_name": case_name,
        "color": color,
        "apply_truncation": apply_truncation,
        "pooled_df": pooled_df,
        "plot_df": plot_df,
        "segmented_df": segmented_df,
        "segment_edges": segment_edges,
    }


def build_stim_bins_from_arrays(value_arrays: list[np.ndarray]) -> np.ndarray:
    all_values = np.concatenate(value_arrays)
    data_min = float(np.nanmin(all_values))
    data_max = float(np.nanmax(all_values))
    left_edge = min(stim_xlim_s[0], data_min)
    right_edge = max(stim_xlim_s[1], data_max)
    return np.arange(left_edge, right_edge + stim_bin_size_s, stim_bin_size_s)


def build_stim_bins(case_results: list[dict]) -> np.ndarray:
    return build_stim_bins_from_arrays(
        [case_result["plot_df"]["intended_fix"].to_numpy(dtype=float) for case_result in case_results]
    )


def get_removed_plot_df(unfiltered_result: dict, filtered_result: dict) -> pd.DataFrame:
    unfiltered_plot_df = unfiltered_result["plot_df"]
    filtered_plot_df = filtered_result["plot_df"]
    removed_plot_df = unfiltered_plot_df[
        ~unfiltered_plot_df["source_row_id"].isin(filtered_plot_df["source_row_id"])
    ].copy()

    if len(removed_plot_df):
        valid_removed_mask = np.isclose(removed_plot_df["abort_event"], 3)
        trunc_mask = pd.Series(False, index=removed_plot_df.index)
        for batch_name in removed_plot_df["batch_name"].astype(str).unique():
            trunc_t = get_trunc_time(batch_name)
            batch_mask = removed_plot_df["batch_name"].astype(str) == batch_name
            if trunc_t is None:
                continue
            trunc_mask |= batch_mask & (removed_plot_df["TotalFixTime"] < trunc_t)
        valid_removed_mask &= trunc_mask
        if not valid_removed_mask.all():
            raise ValueError(
                "Filtered dataset removed trials outside abort_event == 3 with TotalFixTime < trunc_t."
            )

    return removed_plot_df


def make_overlay_plot(unfiltered_result: dict, filtered_result: dict) -> plt.Figure:
    case_results = [unfiltered_result, filtered_result]
    stim_bins_s = build_stim_bins(case_results)
    x_edges_ms = stim_bins_s * 1e3

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(2 * panel_width, panel_height),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    y_max = 0.0
    panel_specs = [
        (0, "Early stim"),
        (1, "Late stim"),
    ]

    for ax, (segment_idx, panel_title) in zip(axes, panel_specs):
        for case_result in case_results:
            segment_df = case_result["segmented_df"][
                case_result["segmented_df"]["intended_fix_segment"] == segment_idx
            ].copy()
            values = segment_df["intended_fix"].to_numpy(dtype=float)
            density = compute_density_histogram(values, stim_bins_s)
            if len(density):
                y_max = max(y_max, float(np.nanmax(density)))

            label = format_case_label(
                case_result["case_name"],
                case_result["segment_edges"][segment_idx : segment_idx + 2],
                len(segment_df),
            )
            ax.stairs(
                density,
                x_edges_ms,
                label=label,
                color=case_result["color"],
                linewidth=2.0,
            )

        ax.set_title(panel_title)
        ax.set_xlim(stim_xlim_s[0] * 1e3, stim_xlim_s[1] * 1e3)
        ax.grid(alpha=0.2, linewidth=0.6)
        ax.set_xlabel("intended_fix (ms)")
        ax.legend(frameon=False, fontsize=9)

    if y_max <= 0:
        y_max = 1.0

    for ax in axes:
        ax.set_ylim(0, 1.05 * y_max)

    axes[0].set_ylabel("Count")
    fig.suptitle("Early/Late stim distributions before vs after abort-3 truncation")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def make_removed_trials_plot(removed_plot_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(panel_width, panel_height),
        squeeze=False,
    )
    ax = ax[0, 0]

    if len(removed_plot_df) == 0:
        raise ValueError("No removed trials found for the removed-trial stimulus distribution plot.")

    stim_bins_s = build_stim_bins_from_arrays(
        [removed_plot_df["intended_fix"].to_numpy(dtype=float)]
    )
    x_edges_ms = stim_bins_s * 1e3
    values = removed_plot_df["intended_fix"].to_numpy(dtype=float)
    density = compute_density_histogram(values, stim_bins_s)

    trunc_counts = (
        removed_plot_df.assign(
            trunc_time_s=removed_plot_df["batch_name"].astype(str).map(get_trunc_time)
        )["trunc_time_s"]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    trunc_label = ", ".join(
        f"n(t<{float(trunc_t):.2f})={int(count)}"
        for trunc_t, count in trunc_counts.items()
    )

    ax.stairs(
        density,
        x_edges_ms,
        color=REMOVED_COLOR,
        linewidth=2.0,
        label=f"Removed trials (n={len(removed_plot_df)})",
    )
    ax.set_title(
        "Stim distribution removed by abort-3 truncation\n"
        f"{trunc_label}"
    )
    ax.set_xlim(stim_xlim_s[0] * 1e3, stim_xlim_s[1] * 1e3)
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.set_xlabel("intended_fix (ms)")
    ax.set_ylabel("Count")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def make_overall_distribution_plot(unfiltered_result: dict, filtered_result: dict) -> plt.Figure:
    case_results = [unfiltered_result, filtered_result]
    stim_bins_s = build_stim_bins(case_results)
    x_edges_ms = stim_bins_s * 1e3

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(panel_width, panel_height),
        squeeze=False,
    )
    ax = ax[0, 0]

    y_max = 0.0
    for case_result in case_results:
        values = case_result["plot_df"]["intended_fix"].to_numpy(dtype=float)
        density = compute_density_histogram(values, stim_bins_s)
        if len(density):
            y_max = max(y_max, float(np.nanmax(density)))

        ax.stairs(
            density,
            x_edges_ms,
            label=f"{case_result['case_name']} distribution (n={len(values)})",
            color=case_result["color"],
            linewidth=2.0,
        )

    if y_max <= 0:
        y_max = 1.0

    for case_result in case_results:
        cutoff_s = float(case_result["segment_edges"][1])
        ax.axvline(
            cutoff_s * 1e3,
            color=case_result["color"],
            linestyle="--",
            linewidth=1.8,
            label=f"{case_result['case_name']} cutoff = {cutoff_s:.3f} s",
        )

    ax.set_title("Overall intended_fix distribution before vs after abort-3 truncation")
    ax.set_xlim(stim_xlim_s[0] * 1e3, stim_xlim_s[1] * 1e3)
    ax.set_ylim(0, 1.05 * y_max)
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.set_xlabel("intended_fix (ms)")
    ax.set_ylabel("Count")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def make_overall_cdf_plot(unfiltered_result: dict, filtered_result: dict) -> plt.Figure:
    case_results = [unfiltered_result, filtered_result]

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(panel_width, panel_height),
        squeeze=False,
    )
    ax = ax[0, 0]

    for case_result in case_results:
        x_values_s, cdf_values = compute_empirical_cdf(
            case_result["plot_df"]["intended_fix"].to_numpy(dtype=float)
        )
        ax.step(
            x_values_s * 1e3,
            cdf_values,
            where="post",
            color=case_result["color"],
            linewidth=2.0,
            label=f"{case_result['case_name']} CDF (n={len(x_values_s)})",
        )

    for case_result in case_results:
        cutoff_s = float(case_result["segment_edges"][1])
        ax.axvline(
            cutoff_s * 1e3,
            color=case_result["color"],
            linestyle="--",
            linewidth=1.8,
            label=f"{case_result['case_name']} cutoff = {cutoff_s:.3f} s",
        )

    ax.set_title("Overall intended_fix CDF before vs after abort-3 truncation")
    ax.set_xlim(stim_xlim_s[0] * 1e3, stim_xlim_s[1] * 1e3)
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.set_xlabel("intended_fix (ms)")
    ax.set_ylabel("Cumulative count")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def make_overall_difference_plot(unfiltered_result: dict, filtered_result: dict) -> plt.Figure:
    stim_bins_s = build_stim_bins([unfiltered_result, filtered_result])
    x_edges_ms = stim_bins_s * 1e3

    unfiltered_counts = compute_density_histogram(
        unfiltered_result["plot_df"]["intended_fix"].to_numpy(dtype=float),
        stim_bins_s,
    )
    filtered_counts = compute_density_histogram(
        filtered_result["plot_df"]["intended_fix"].to_numpy(dtype=float),
        stim_bins_s,
    )
    count_difference = filtered_counts - unfiltered_counts

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(panel_width, panel_height),
        squeeze=False,
    )
    ax = ax[0, 0]

    ax.stairs(
        count_difference,
        x_edges_ms,
        color="tab:purple",
        linewidth=2.0,
        label="Filtered - unfiltered",
    )
    ax.axhline(0.0, color="0.3", linestyle="--", linewidth=1.2)
    ax.set_title("Overall intended_fix count difference after abort-3 truncation")
    ax.set_xlim(stim_xlim_s[0] * 1e3, stim_xlim_s[1] * 1e3)
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.set_xlabel("intended_fix (ms)")
    ax.set_ylabel("Count difference")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def make_removed_fraction_plot(unfiltered_result: dict, filtered_result: dict) -> plt.Figure:
    stim_bins_s = build_stim_bins([unfiltered_result, filtered_result])
    x_edges_ms = stim_bins_s * 1e3

    unfiltered_counts = compute_density_histogram(
        unfiltered_result["plot_df"]["intended_fix"].to_numpy(dtype=float),
        stim_bins_s,
    )
    filtered_counts = compute_density_histogram(
        filtered_result["plot_df"]["intended_fix"].to_numpy(dtype=float),
        stim_bins_s,
    )
    removed_counts = unfiltered_counts - filtered_counts
    removed_fraction = np.divide(
        removed_counts,
        unfiltered_counts,
        out=np.full_like(removed_counts, np.nan, dtype=float),
        where=unfiltered_counts > 0,
    )

    global_removed_fraction = (
        len(get_removed_plot_df(unfiltered_result, filtered_result)) / len(unfiltered_result["plot_df"])
        if len(unfiltered_result["plot_df"]) > 0
        else float("nan")
    )

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(panel_width, panel_height),
        squeeze=False,
    )
    ax = ax[0, 0]

    ax.stairs(
        removed_fraction * 100.0,
        x_edges_ms,
        color="tab:green",
        linewidth=2.0,
        label="Removed fraction by bin",
    )
    if np.isfinite(global_removed_fraction):
        ax.axhline(
            global_removed_fraction * 100.0,
            color="0.3",
            linestyle="--",
            linewidth=1.2,
            label=f"Overall removed fraction = {100.0 * global_removed_fraction:.2f}%",
        )

    ax.set_title("Removed trial percentage by intended_fix bin")
    ax.set_xlim(stim_xlim_s[0] * 1e3, stim_xlim_s[1] * 1e3)
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.set_xlabel("intended_fix (ms)")
    ax.set_ylabel("Removed trials (%)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def print_summary(unfiltered_result: dict, filtered_result: dict, removed_plot_df: pd.DataFrame) -> None:
    unfiltered_plot_df = unfiltered_result["plot_df"]
    filtered_plot_df = filtered_result["plot_df"]

    if len(unfiltered_plot_df) == 0:
        removed_fraction = float("nan")
    else:
        removed_fraction = len(removed_plot_df) / len(unfiltered_plot_df)

    edge_deltas = filtered_result["segment_edges"] - unfiltered_result["segment_edges"]

    print("Stim-distribution comparison summary:")
    print(f"  TRIAL_POOL_MODE: {TRIAL_POOL_MODE}")
    print(f"  Plot trials before truncation: {len(unfiltered_plot_df)}")
    print(f"  Plot trials after truncation:  {len(filtered_plot_df)}")
    print(f"  Removed plot trials:           {len(removed_plot_df)}")
    print(f"  Fraction removed:              {removed_fraction:.6f}")
    print(
        "  Unfiltered segment edges (s): "
        f"{format_edge_list(unfiltered_result['segment_edges'])}"
    )
    print(
        "  Filtered segment edges (s):   "
        f"{format_edge_list(filtered_result['segment_edges'])}"
    )
    print(f"  Edge deltas (filtered - unfiltered) (s): {format_edge_list(edge_deltas)}")


# %%
output_dir.mkdir(parents=True, exist_ok=True)

unfiltered_result = build_case_result(
    case_name="Unfiltered",
    color=UNFILTERED_COLOR,
    apply_truncation=False,
)
filtered_result = build_case_result(
    case_name="Filtered",
    color=FILTERED_COLOR,
    apply_truncation=True,
)

removed_plot_df = get_removed_plot_df(unfiltered_result, filtered_result)

print_summary(unfiltered_result, filtered_result, removed_plot_df)

fig = make_overlay_plot(unfiltered_result, filtered_result)
removed_fig = make_removed_trials_plot(removed_plot_df)
overall_fig = make_overall_distribution_plot(unfiltered_result, filtered_result)
cdf_fig = make_overall_cdf_plot(unfiltered_result, filtered_result)
difference_fig = make_overall_difference_plot(unfiltered_result, filtered_result)
removed_fraction_fig = make_removed_fraction_plot(unfiltered_result, filtered_result)

save_figure(fig, output_base)
save_figure(removed_fig, removed_output_base)
save_figure(overall_fig, overall_output_base)
save_figure(cdf_fig, cdf_output_base)
save_figure(difference_fig, difference_output_base)
save_figure(removed_fraction_fig, removed_fraction_output_base)

print(f"Saved: {output_base.with_suffix('.pdf')}")
print(f"Saved: {output_base.with_suffix('.png')}")
print(f"Saved: {removed_output_base.with_suffix('.pdf')}")
print(f"Saved: {removed_output_base.with_suffix('.png')}")
print(f"Saved: {overall_output_base.with_suffix('.pdf')}")
print(f"Saved: {overall_output_base.with_suffix('.png')}")
print(f"Saved: {cdf_output_base.with_suffix('.pdf')}")
print(f"Saved: {cdf_output_base.with_suffix('.png')}")
print(f"Saved: {difference_output_base.with_suffix('.pdf')}")
print(f"Saved: {difference_output_base.with_suffix('.png')}")
print(f"Saved: {removed_fraction_output_base.with_suffix('.pdf')}")
print(f"Saved: {removed_fraction_output_base.with_suffix('.png')}")

if SHOW_PLOT:
    plt.show()
else:
    plt.close(fig)
    plt.close(removed_fig)
    plt.close(overall_fig)
    plt.close(cdf_fig)
    plt.close(difference_fig)
    plt.close(removed_fraction_fig)

# %%
