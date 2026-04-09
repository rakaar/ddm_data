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

ABL_VALUES = (20, 40, 60)
ABS_ILD_VALUES = (1, 2, 4, 8, 16)

NUM_STIM_BINS = 11

INTENDED_FIX_MIN_S = 0.2
INTENDED_FIX_MAX_S = 2.3

Y_LIM = (0.45, 1.02)

png_dpi = 300
show_plot = SHOW_PLOT


# %%
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
batch_csv_dir = REPO_ROOT / "fit_animal_by_animal" / "batch_csvs"
output_dir = SCRIPT_DIR / "multi_animal_valid_accuracy_stim_quantiles_compare_abl_per_abs_ild"

summary_csv_path = output_dir / f"accuracy_vs_stim_qcut_{NUM_STIM_BINS}bins_compare_abl_per_abs_ild_summary.csv"
figure_output_base = output_dir / f"accuracy_vs_stim_qcut_{NUM_STIM_BINS}bins_compare_abl_per_abs_ild"

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

    if batch_file_with_4.exists():
        return batch_file_with_4
    if batch_file.exists():
        return batch_file

    raise FileNotFoundError(f"Missing batch CSV for batch '{batch_name}'.")


def load_merged_valid_trials() -> pd.DataFrame:
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
        ["batch_name", "animal", "success", "ABL", "ILD", "intended_fix"],
    )

    for column_name in ["animal", "success", "ABL", "ILD", "intended_fix"]:
        merged_data[column_name] = pd.to_numeric(merged_data[column_name], errors="coerce")

    valid_df = merged_data[merged_data["success"].isin([1, -1])].copy()
    valid_df["abs_ILD"] = valid_df["ILD"].abs()
    valid_df["correct"] = np.isclose(valid_df["success"], 1).astype(float)

    if missing_batch_files:
        print(f"Skipped missing batches: {missing_batch_files}")

    if EXCLUDED_BATCH_ANIMAL_PAIRS:
        excluded_pairs = {(str(batch), str(animal)) for batch, animal in EXCLUDED_BATCH_ANIMAL_PAIRS}
        batch_animal_keys = list(
            zip(valid_df["batch_name"].astype(str), valid_df["animal"].astype("Int64").astype(str))
        )
        valid_df = valid_df[[key not in excluded_pairs for key in batch_animal_keys]].copy()
        print(f"Excluded batch-animal pairs: {sorted(excluded_pairs)}")

    return valid_df


def prepare_plot_df(valid_df: pd.DataFrame) -> pd.DataFrame:
    mask = valid_df["intended_fix"].notna()
    mask &= valid_df["ABL"].isin(ABL_VALUES)
    mask &= valid_df["abs_ILD"].isin(ABS_ILD_VALUES)
    mask &= valid_df["intended_fix"] >= INTENDED_FIX_MIN_S
    if INTENDED_FIX_MAX_S is not None:
        mask &= valid_df["intended_fix"] <= INTENDED_FIX_MAX_S

    plot_df = valid_df[mask].copy()
    if len(plot_df) == 0:
        raise ValueError("No valid trials found after filtering.")

    validate_supported_values(plot_df, "ABL", ABL_VALUES)
    validate_supported_values(plot_df, "abs_ILD", ABS_ILD_VALUES)
    return plot_df


def build_group_summary(group_df: pd.DataFrame, abs_ild_value: int, abl_value: int) -> pd.DataFrame:
    if len(group_df) == 0:
        raise ValueError(f"No trials found for abs_ILD={abs_ild_value}, ABL={abl_value}.")

    stim_bin, stim_edges = pd.qcut(
        group_df["intended_fix"],
        q=NUM_STIM_BINS,
        labels=False,
        retbins=True,
        duplicates="drop",
    )
    if len(stim_edges) - 1 != NUM_STIM_BINS:
        raise ValueError(
            f"Requested {NUM_STIM_BINS} bins, but only {len(stim_edges) - 1} unique bins could be formed "
            f"for abs_ILD={abs_ild_value}, ABL={abl_value}."
        )
    if stim_bin.isna().any():
        raise ValueError(f"Failed qcut assignment for abs_ILD={abs_ild_value}, ABL={abl_value}.")

    binned_df = group_df.copy()
    binned_df["stim_bin"] = stim_bin.astype(int)

    bin_index = list(range(NUM_STIM_BINS))
    counts = binned_df.groupby("stim_bin").size().reindex(bin_index, fill_value=0)
    accuracy = binned_df.groupby("stim_bin")["correct"].mean().reindex(bin_index)
    sem_accuracy = np.sqrt(accuracy * (1.0 - accuracy) / counts.replace(0, np.nan)).fillna(0.0)

    return pd.DataFrame(
        {
            "abs_ILD": int(abs_ild_value),
            "ABL": int(abl_value),
            "stim_bin": bin_index,
            "stim_left_s": stim_edges[:-1],
            "stim_right_s": stim_edges[1:],
            "stim_mid_s": 0.5 * (stim_edges[:-1] + stim_edges[1:]),
            "n_trials": counts.values,
            "accuracy": accuracy.values,
            "sem_accuracy": sem_accuracy.values,
        }
    )


def build_all_summaries(plot_df: pd.DataFrame) -> pd.DataFrame:
    summary_frames = []

    for abs_ild_value in ABS_ILD_VALUES:
        abs_ild_df = plot_df[np.isclose(plot_df["abs_ILD"], abs_ild_value)].copy()
        for abl_value in ABL_VALUES:
            group_df = abs_ild_df[np.isclose(abs_ild_df["ABL"], abl_value)].copy()
            summary_frames.append(build_group_summary(group_df, int(abs_ild_value), int(abl_value)))

    return pd.concat(summary_frames, ignore_index=True)


def format_count_block(summary_df: pd.DataFrame, abs_ild_value: int) -> str:
    lines = [f"|ILD| = {int(abs_ild_value)}  qcut counts per ABL"]
    abs_ild_df = summary_df[np.isclose(summary_df["abs_ILD"], abs_ild_value)].copy()

    for abl_value in ABL_VALUES:
        abl_df = abs_ild_df[np.isclose(abs_ild_df["ABL"], abl_value)].sort_values("stim_bin")
        counts_text = "  ".join(
            [f"Q{int(row.stim_bin) + 1:02d}={int(row.n_trials)}" for row in abl_df.itertuples(index=False)]
        )
        lines.append(f"ABL {int(abl_value)}: {counts_text}")

    return "\n".join(lines)


def make_figure(summary_df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, len(ABS_ILD_VALUES), figsize=(30.0, 8.2), sharey=True)

    for ax, abs_ild_value in zip(axes, ABS_ILD_VALUES):
        abs_ild_df = summary_df[np.isclose(summary_df["abs_ILD"], abs_ild_value)].copy()

        for abl_value in ABL_VALUES:
            abl_df = abs_ild_df[np.isclose(abs_ild_df["ABL"], abl_value)].sort_values("stim_bin")
            ax.errorbar(
                abl_df["stim_mid_s"] * 1e3,
                abl_df["accuracy"],
                yerr=abl_df["sem_accuracy"],
                fmt="o-",
                color=abl_colors[int(abl_value)],
                linewidth=2,
                markersize=5,
                capsize=3,
                label=f"ABL {int(abl_value)}",
            )

        ax.set_title(f"|ILD| = {int(abs_ild_value)}")
        ax.set_xlabel("Stim-bin midpoint (ms)")
        ax.set_ylim(*Y_LIM)
        ax.grid(alpha=0.3)
        ax.text(
            0.0,
            -0.27,
            format_count_block(summary_df, int(abs_ild_value)),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.6,
            family="monospace",
        )
        ax.set_xlim(200, 800)

    axes[0].set_ylabel("Accuracy")
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle(
        "Accuracy vs stim bin midpoint by ABL within each |ILD| (error bars = SEM)\n",
        y=1.02,
    )
    fig.subplots_adjust(bottom=0.36, wspace=0.25)
    return fig


def print_summary_report(plot_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    print("Accuracy vs stim qcut bins by ABL within each abs_ILD")
    print(f"  Total filtered valid trials: {len(plot_df)}")
    print(
        "  intended_fix range (s): "
        f"[{float(plot_df['intended_fix'].min()):.6f}, {float(plot_df['intended_fix'].max()):.6f}]"
    )
    print(f"  NUM_STIM_BINS: {NUM_STIM_BINS}")
    print("")

    for abs_ild_value in ABS_ILD_VALUES:
        print(f"|ILD| = {int(abs_ild_value)}")
        abs_ild_df = summary_df[np.isclose(summary_df["abs_ILD"], abs_ild_value)].copy()
        print(
            abs_ild_df[
                ["ABL", "stim_bin", "n_trials", "stim_mid_s", "accuracy", "sem_accuracy"]
            ].to_string(index=False)
        )
        print("")


# %%
output_dir.mkdir(parents=True, exist_ok=True)

valid_df = load_merged_valid_trials()
plot_df = prepare_plot_df(valid_df)
summary_df = build_all_summaries(plot_df)
summary_df.to_csv(summary_csv_path, index=False)

print_summary_report(plot_df, summary_df)
print(f"Saved summary CSV: {summary_csv_path}")

fig = make_figure(summary_df)
save_figure(fig, figure_output_base)
print(f"Saved figure: {figure_output_base.with_suffix('.pdf')}")

if show_plot:
    plt.show()
else:
    plt.close(fig)
