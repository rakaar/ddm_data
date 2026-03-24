# %%
SHOW_PLOT = True

DESIRED_BATCHES = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]
# DESIRED_BATCHES = ["LED7"]

EXCLUDED_BATCH_ANIMAL_PAIRS = []
TRIAL_POOL_MODE = "valid"  # "valid" or "valid_plus_abort3"
SEGMENT_MODE = "quantile"  # "quantile" or "fixed"
NUM_INTENDED_FIX_QUANTILE_BINS = 2
FIXED_SEGMENT_EDGES_S = (0.2, 0.4, 1.5)

if TRIAL_POOL_MODE not in {"valid", "valid_plus_abort3"}:
    raise ValueError(f"Unsupported TRIAL_POOL_MODE: {TRIAL_POOL_MODE}")
if SEGMENT_MODE not in {"quantile", "fixed"}:
    raise ValueError(f"Unsupported SEGMENT_MODE: {SEGMENT_MODE}")

ABL_VALUES = (20, 40, 60)
ILD_VALUES = (-16, -8, -4, -2, -1, 1, 2, 4, 8, 16)

intended_fix_min_s = 0.2
intended_fix_max_s = 1.5
rt_min_s = -1.0
rt_max_s = 1.0
bin_size_s = 25e-3
xlim_s = (0, 1)
figure_size = (5.0, 6.6)
png_dpi = 300


# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# %%
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
batch_csv_dir = REPO_ROOT / "fit_animal_by_animal" / "batch_csvs"
output_dir = SCRIPT_DIR / "valid_rtd_stim_segments_all_animals_avg_animal_rtds"

output_suffix_parts = []
if TRIAL_POOL_MODE == "valid_plus_abort3":
    output_suffix_parts.append("plus_abort3")
output_suffix_parts.append(f"{SEGMENT_MODE}_segments")
output_suffix_parts.append("avg_animal_rtds")
output_suffix = "".join(f"_{part}" for part in output_suffix_parts)
output_base = output_dir / f"valid_rtd_by_abl_stim_segments_all_animals{output_suffix}"

abl_colors = {
    20: "tab:blue",
    40: "tab:orange",
    60: "tab:green",
}


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


def get_batch_animal_pairs(df: pd.DataFrame) -> list[tuple[str, int]]:
    pairs_df = df[["batch_name", "animal"]].dropna().copy()
    pairs_df["batch_name"] = pairs_df["batch_name"].astype(str)
    pairs_df["animal"] = pd.to_numeric(pairs_df["animal"], errors="coerce")
    pairs_df = pairs_df.dropna().copy()
    pairs_df = pairs_df.drop_duplicates(subset=["batch_name", "animal"]).copy()
    return sorted((str(row.batch_name), int(row.animal)) for row in pairs_df.itertuples(index=False))


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

    numeric_columns = ["animal", "RTwrtStim", "ABL", "ILD", "intended_fix"]
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

    if missing_batch_files:
        print(f"Skipped missing batch files: {missing_batch_files}")

    if EXCLUDED_BATCH_ANIMAL_PAIRS:
        excluded_pairs = {(str(batch), int(animal)) for batch, animal in EXCLUDED_BATCH_ANIMAL_PAIRS}
        batch_animal_keys = list(zip(merged_valid["batch_name"].astype(str), merged_valid["animal"].astype(int)))
        merged_valid = merged_valid[[key not in excluded_pairs for key in batch_animal_keys]].copy()
        print(f"Excluded batch-animal pairs: {sorted(excluded_pairs)}")

    return merged_valid


def prepare_plot_df(valid_df: pd.DataFrame) -> pd.DataFrame:
    plot_df = valid_df[
        (valid_df["RTwrtStim"] >= rt_min_s)
        & (valid_df["RTwrtStim"] <= rt_max_s)
        & (valid_df["intended_fix"] >= intended_fix_min_s)
        & (valid_df["intended_fix"] <= intended_fix_max_s)
        & (valid_df["ABL"].isin(ABL_VALUES))
        & (valid_df["ILD"].isin(ILD_VALUES))
    ].copy()

    if len(plot_df) == 0:
        raise ValueError("No RTD plot trials found after filtering.")

    validate_supported_values(plot_df, "ABL", ABL_VALUES)
    validate_supported_values(plot_df, "ILD", ILD_VALUES)
    return plot_df


def add_intended_fix_segments(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    if SEGMENT_MODE == "quantile":
        if NUM_INTENDED_FIX_QUANTILE_BINS <= 0:
            raise ValueError("NUM_INTENDED_FIX_QUANTILE_BINS must be positive.")
        if df["intended_fix"].isna().any():
            raise ValueError("Cannot build quantile segments with NaN intended_fix values.")
        if float(df["intended_fix"].nunique()) <= 1:
            raise ValueError("Cannot segment intended_fix because all filtered values are identical.")

        segment_ids, segment_edges = pd.qcut(
            df["intended_fix"],
            q=NUM_INTENDED_FIX_QUANTILE_BINS,
            labels=False,
            retbins=True,
            duplicates="drop",
        )
        if len(segment_edges) - 1 != NUM_INTENDED_FIX_QUANTILE_BINS:
            raise ValueError(
                "Requested "
                f"{NUM_INTENDED_FIX_QUANTILE_BINS} quantile bins, but only {len(segment_edges) - 1} unique bins could be formed."
            )
        if segment_ids.isna().any():
            raise ValueError("Failed to assign intended_fix quantile segments to some trials.")

        segmented_df = df.copy()
        segmented_df["intended_fix_segment"] = segment_ids.astype(int)
        return segmented_df, np.asarray(segment_edges, dtype=float)

    segment_edges = np.asarray(FIXED_SEGMENT_EDGES_S, dtype=float)
    if len(segment_edges) < 3:
        raise ValueError("FIXED_SEGMENT_EDGES_S must define at least two segments.")

    cut_edges = segment_edges.copy()
    cut_edges[0] = np.nextafter(cut_edges[0], -np.inf)
    cut_edges[-1] = np.nextafter(cut_edges[-1], np.inf)
    segment_ids = pd.cut(
        df["intended_fix"],
        bins=cut_edges,
        labels=False,
        include_lowest=True,
        right=True,
    )
    if segment_ids.isna().any():
        raise ValueError("Failed to assign intended_fix fixed segments to some trials.")

    segmented_df = df.copy()
    segmented_df["intended_fix_segment"] = segment_ids.astype(int)
    return segmented_df, segment_edges


def build_segment_specs(segment_edges: np.ndarray) -> list[dict[str, float]]:
    n_segments = len(segment_edges) - 1
    segment_specs = []
    for segment_idx in range(n_segments):
        segment_specs.append(
            {
                "index": int(segment_idx),
                "name": f"Q{segment_idx + 1}/{n_segments}",
                "left": float(segment_edges[segment_idx]),
                "right": float(segment_edges[segment_idx + 1]),
            }
        )
    return segment_specs


def compute_density_histogram(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.full(len(bins) - 1, np.nan, dtype=float)
    hist, _ = np.histogram(values, bins=bins, density=True)
    return hist


def compute_one_animal_rtd(batch_name: str, animal_id: int, animal_df: pd.DataFrame, segment_specs, bins_s: np.ndarray):
    animal_segment_results = []

    for segment_spec in segment_specs:
        segment_df = animal_df[animal_df["intended_fix_segment"] == segment_spec["index"]].copy()
        segment_trial_count = int(len(segment_df))

        densities_by_abl = {}
        trial_counts_by_abl = {}
        for abl_value in ABL_VALUES:
            abl_df = segment_df[np.isclose(segment_df["ABL"], abl_value)].copy()
            trial_counts_by_abl[int(abl_value)] = int(len(abl_df))
            densities_by_abl[int(abl_value)] = compute_density_histogram(
                abl_df["RTwrtStim"].to_numpy(),
                bins_s,
            )

        animal_segment_results.append(
            {
                "segment_spec": segment_spec,
                "total": segment_trial_count,
                "densities_by_abl": densities_by_abl,
                "trial_counts_by_abl": trial_counts_by_abl,
            }
        )

    return {
        "pair": (str(batch_name), int(animal_id)),
        "segment_results": animal_segment_results,
    }


def aggregate_animal_results(per_animal_results, segment_specs, bins_s):
    aggregated_segment_results = []

    for segment_idx, segment_spec in enumerate(segment_specs):
        segment_totals = np.asarray(
            [animal_result["segment_results"][segment_idx]["total"] for animal_result in per_animal_results],
            dtype=int,
        )

        densities_by_abl = {}
        contributing_animals_by_abl = {}
        trial_counts_by_abl = {}

        for abl_value in ABL_VALUES:
            density_list = []
            abl_trial_counts = []

            for animal_result in per_animal_results:
                animal_segment_result = animal_result["segment_results"][segment_idx]
                density_list.append(animal_segment_result["densities_by_abl"][int(abl_value)])
                abl_trial_counts.append(animal_segment_result["trial_counts_by_abl"][int(abl_value)])

            density_stack = np.stack(density_list, axis=0)
            abl_trial_counts = np.asarray(abl_trial_counts, dtype=int)
            contributing_mask = abl_trial_counts > 0

            if np.any(contributing_mask):
                densities_by_abl[int(abl_value)] = np.nanmean(density_stack[contributing_mask], axis=0)
            else:
                densities_by_abl[int(abl_value)] = np.full(len(bins_s) - 1, np.nan, dtype=float)

            contributing_animals_by_abl[int(abl_value)] = int(np.sum(contributing_mask))
            trial_counts_by_abl[int(abl_value)] = int(np.sum(abl_trial_counts))

        aggregated_segment_results.append(
            {
                "segment_spec": segment_spec,
                "total": int(np.sum(segment_totals)),
                "contributing_animals_total": int(np.sum(segment_totals > 0)),
                "densities_by_abl": densities_by_abl,
                "contributing_animals_by_abl": contributing_animals_by_abl,
                "trial_counts_by_abl": trial_counts_by_abl,
            }
        )

    return aggregated_segment_results


def load_data():
    valid_df = load_merged_valid_trials()
    plot_df = prepare_plot_df(valid_df)
    plot_df, segment_edges = add_intended_fix_segments(plot_df)
    segment_specs = build_segment_specs(segment_edges)

    included_pairs = get_batch_animal_pairs(plot_df)
    if not included_pairs:
        raise ValueError("No eligible batch-animal pairs remained after filtering.")

    pair_to_animal_df = {}
    for batch_name, animal_id in included_pairs:
        animal_df = plot_df[
            (plot_df["batch_name"].astype(str) == str(batch_name))
            & np.isclose(plot_df["animal"], int(animal_id))
        ].copy()
        if len(animal_df) == 0:
            continue
        pair_to_animal_df[(str(batch_name), int(animal_id))] = animal_df

    if not pair_to_animal_df:
        raise ValueError("No per-animal dataframes found after filtering.")

    bins_s = np.arange(rt_min_s, rt_max_s + bin_size_s, bin_size_s)
    per_animal_results = [
        compute_one_animal_rtd(
            batch_name,
            animal_id,
            pair_to_animal_df[(str(batch_name), int(animal_id))],
            segment_specs,
            bins_s,
        )
        for batch_name, animal_id in included_pairs
    ]
    segment_results = aggregate_animal_results(per_animal_results, segment_specs, bins_s)

    return {
        "included_pairs": included_pairs,
        "plot_df": plot_df,
        "bins_s": bins_s,
        "segment_edges": segment_edges,
        "segment_results": segment_results,
        "per_animal_results": per_animal_results,
    }


data = load_data()


# %%
def save_figure(fig, output_base_path):
    fig.savefig(output_base_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base_path.with_suffix(".png"), dpi=png_dpi, bbox_inches="tight")


def plot_data(data):
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        len(data["segment_results"]),
        1,
        figsize=figure_size,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    x_edges_s = data["bins_s"]

    visible_mask = (x_edges_s[:-1] >= xlim_s[0]) & (x_edges_s[1:] <= xlim_s[1])
    global_max = 0.0
    for segment_result in data["segment_results"]:
        for abl_value in ABL_VALUES:
            density = segment_result["densities_by_abl"][int(abl_value)]
            if np.any(np.isfinite(density)):
                global_max = max(global_max, float(np.nanmax(density[visible_mask])))
    y_max = 1.05 * global_max if global_max > 0 else 1.0

    for row_idx, segment_result in enumerate(data["segment_results"]):
        ax = axes[row_idx, 0]
        segment_spec = segment_result["segment_spec"]
        for abl_value in ABL_VALUES:
            ax.stairs(
                segment_result["densities_by_abl"][int(abl_value)],
                x_edges_s,
                label=f"ABL = {abl_value}",
                color=abl_colors[int(abl_value)],
                linewidth=1.8,
            )
        ax.set_xlim(*xlim_s)
        ax.set_ylim(0, y_max)
        ax.grid(alpha=0.2, linewidth=0.6)
        ax.set_ylabel("Density")
        ax.set_title(
            f"avg per-animal data {segment_spec['name']} [{segment_spec['left']:.3f}, {segment_spec['right']:.3f}] s\n"
            f"animals={segment_result['contributing_animals_total']}"
        )
        if row_idx == len(data["segment_results"]) - 1:
            ax.set_xlabel("RT wrt stim (s)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(ABL_VALUES), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    tagged_output_base = output_dir / f"{output_base.name}_{len(data['included_pairs'])}animals"
    save_figure(fig, tagged_output_base)

    fig_by_abl, axes_by_abl = plt.subplots(1, len(ABL_VALUES), figsize=(12.0, 3.8), sharex=True, sharey=True, squeeze=False)
    early_segment_result = data["segment_results"][0]
    late_segment_result = data["segment_results"][-1]

    global_max_by_abl = 0.0
    for abl_value in ABL_VALUES:
        early_density = early_segment_result["densities_by_abl"][int(abl_value)]
        late_density = late_segment_result["densities_by_abl"][int(abl_value)]
        if np.any(np.isfinite(early_density)):
            global_max_by_abl = max(global_max_by_abl, float(np.nanmax(early_density[visible_mask])))
        if np.any(np.isfinite(late_density)):
            global_max_by_abl = max(global_max_by_abl, float(np.nanmax(late_density[visible_mask])))
    y_max_by_abl = 1.05 * global_max_by_abl if global_max_by_abl > 0 else 1.0

    for col_idx, abl_value in enumerate(ABL_VALUES):
        ax = axes_by_abl[0, col_idx]
        ax.stairs(
            early_segment_result["densities_by_abl"][int(abl_value)],
            x_edges_s,
            label=early_segment_result["segment_spec"]["name"],
            color="tab:blue",
            linewidth=1.8,
        )
        ax.stairs(
            late_segment_result["densities_by_abl"][int(abl_value)],
            x_edges_s,
            label=late_segment_result["segment_spec"]["name"],
            color="tab:red",
            linewidth=1.8,
        )
        ax.set_xlim(*xlim_s)
        ax.set_ylim(0, y_max_by_abl)
        ax.grid(alpha=0.2, linewidth=0.6)
        ax.set_title(f"ABL = {abl_value}")
        ax.set_xlabel("RT wrt stim (s)")
        if col_idx == 0:
            ax.set_ylabel("Density")

    handles_by_abl, labels_by_abl = axes_by_abl[0, 0].get_legend_handles_labels()
    fig_by_abl.legend(handles_by_abl, labels_by_abl, loc="upper center", ncol=2, frameon=False)
    fig_by_abl.tight_layout(rect=(0, 0, 1, 0.90))

    tagged_output_base_by_abl = output_dir / f"{output_base.name}_{len(data['included_pairs'])}animals_by_abl"
    save_figure(fig_by_abl, tagged_output_base_by_abl)

    print(f"Included animals ({len(data['included_pairs'])}):")
    for batch_name, animal_id in data["included_pairs"]:
        print(f"  {batch_name}-{animal_id}")
    print(f"TRIAL_POOL_MODE: {TRIAL_POOL_MODE}")
    print(f"SEGMENT_MODE: {SEGMENT_MODE}")
    print(f"Filtered pooled plot trials: {len(data['plot_df'])}")
    print(f"Segment edges (s): {[float(edge) for edge in data['segment_edges']]}")
    for segment_result in data["segment_results"]:
        segment_spec = segment_result["segment_spec"]
        print(
            f"Segment {segment_spec['name']} [{segment_spec['left']:.3f}, {segment_spec['right']:.3f}] s, "
            f"animals={segment_result['contributing_animals_total']}, total_trials={segment_result['total']}"
        )
        for abl_value in ABL_VALUES:
            print(
                f"  ABL={abl_value}, animals={segment_result['contributing_animals_by_abl'][int(abl_value)]}, "
                f"trials={segment_result['trial_counts_by_abl'][int(abl_value)]}"
            )
    print(f"Saved: {tagged_output_base.with_suffix('.pdf')}")
    print(f"Saved: {tagged_output_base.with_suffix('.png')}")
    print(f"Saved: {tagged_output_base_by_abl.with_suffix('.pdf')}")
    print(f"Saved: {tagged_output_base_by_abl.with_suffix('.png')}")

    return fig, fig_by_abl


fig, fig_by_abl = plot_data(data)

if SHOW_PLOT:
    plt.show()
else:
    plt.close(fig)
    plt.close(fig_by_abl)

# %%
