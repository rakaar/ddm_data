# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


# %%
############ Parameters (edit here) ############
batch_name = "LED7"
session_type = 7
training_level = 16
allowed_repeat_trials = [0, 2]
allowed_abort_events = [3, 4]
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
expert_data_csv_path = REPO_ROOT / "LED7_expert_data.csv"
raw_data_csv_path = REPO_ROOT / "out_LED.csv"
output_dir = SCRIPT_DIR / "led7_control_opto_rtd_by_abl_1x3"
plot_output_base = output_dir / "led7_control_opto_rtd_by_abl_1x3_xlim_0_120ms"


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
    n_bins = int(round(rt_max_s / bin_width_s))
    if not np.isclose(n_bins * bin_width_s, rt_max_s):
        raise ValueError(
            f"rt_max_s={rt_max_s} must be an integer multiple of bin_width_s={bin_width_s}."
        )
    return np.linspace(0.0, rt_max_s, n_bins + 1)


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=png_dpi, bbox_inches="tight")


# %%
############ Load control rows from LED7 expert CSV ############
if not expert_data_csv_path.exists():
    raise FileNotFoundError(f"Could not find expert CSV: {expert_data_csv_path}")

output_dir.mkdir(parents=True, exist_ok=True)
expert_df = pd.read_csv(expert_data_csv_path)

required_expert_columns = [
    "batch_name",
    "session_type",
    "training_level",
    "repeat_trial",
    "LED_trial",
    "success",
    "abort_event",
    "ABL",
    "ILD",
    "abs_ILD",
    "intended_fix",
    "TotalFixTime",
    "RTwrtStim",
    "LED_onset_time",
    "t_LED",
]
missing_columns = [column for column in required_expert_columns if column not in expert_df.columns]
if missing_columns:
    raise ValueError(f"Expert CSV is missing required columns: {missing_columns}")

condition_failures = []
if not expert_df["batch_name"].eq(batch_name).all():
    condition_failures.append(f"batch_name is not all {batch_name}")
if not expert_df["session_type"].isin([session_type]).all():
    condition_failures.append(f"session_type is not all {session_type}")
if not expert_df["training_level"].isin([training_level]).all():
    condition_failures.append(f"training_level is not all {training_level}")
if not (expert_df["repeat_trial"].isin(allowed_repeat_trials) | expert_df["repeat_trial"].isna()).all():
    condition_failures.append(f"repeat_trial is not in {allowed_repeat_trials} or missing")
if not (expert_df["success"].isin([1, -1]) | expert_df["abort_event"].isin(allowed_abort_events)).all():
    condition_failures.append("trial pool is not success in {1, -1} or abort_event in {3, 4}")
if expert_df["RTwrtStim"].isna().any():
    condition_failures.append("RTwrtStim contains missing values")
if not (expert_df["RTwrtStim"] < max_rtwrtstim_for_plot).all():
    condition_failures.append(f"RTwrtStim has values >= {max_rtwrtstim_for_plot}")
if not np.allclose(
    expert_df["RTwrtStim"].to_numpy(dtype=float),
    expert_df["TotalFixTime"].to_numpy(dtype=float) - expert_df["intended_fix"].to_numpy(dtype=float),
):
    condition_failures.append("RTwrtStim is not TotalFixTime - intended_fix")
if not np.allclose(
    expert_df["t_LED"].to_numpy(dtype=float),
    expert_df["intended_fix"].to_numpy(dtype=float) - expert_df["LED_onset_time"].to_numpy(dtype=float),
):
    condition_failures.append("t_LED is not intended_fix - LED_onset_time")
if not np.allclose(
    expert_df["abs_ILD"].to_numpy(dtype=float),
    np.abs(expert_df["ILD"].to_numpy(dtype=float)),
):
    condition_failures.append("abs_ILD is not abs(ILD)")

unexpected_abl_values = [
    float(abl)
    for abl in np.sort(expert_df["ABL"].dropna().astype(float).unique())
    if not any(np.isclose(float(abl), float(supported)) for supported in supported_abl_values)
]
if unexpected_abl_values:
    condition_failures.append(
        f"unexpected expert CSV ABL values {unexpected_abl_values}; supported values are {supported_abl_values}"
    )

if condition_failures:
    raise ValueError("Expert CSV condition check failed:\n  - " + "\n  - ".join(condition_failures))

control_df_all = expert_df[
    ((expert_df["LED_trial"] == 0) | expert_df["LED_trial"].isna())
    & expert_df["ABL"].isin(supported_abl_values)
].copy()
control_plot_df = control_df_all[
    (control_df_all["RTwrtStim"] >= 0.0)
    & (control_df_all["RTwrtStim"] < max_rtwrtstim_for_plot)
].copy()

if len(control_plot_df) == 0:
    raise ValueError("No control rows found in LED7 expert CSV after RTwrtStim filtering.")


# %%
############ Load opto rows ############
expert_opto_rows = expert_df[expert_df["LED_trial"] == 1].copy()
if len(expert_opto_rows) > 0:
    opto_df_all = expert_opto_rows[expert_opto_rows["ABL"].isin(supported_abl_values)].copy()
    opto_source = expert_data_csv_path
else:
    if not raw_data_csv_path.exists():
        raise FileNotFoundError(f"Could not find raw CSV for opto rows: {raw_data_csv_path}")

    raw_df = pd.read_csv(raw_data_csv_path)
    raw_df["batch_name"] = batch_name
    raw_df["RTwrtStim"] = raw_df["timed_fix"] - raw_df["intended_fix"]
    raw_df["t_LED"] = raw_df["intended_fix"] - raw_df["LED_onset_time"]
    raw_df = raw_df.rename(columns={"timed_fix": "TotalFixTime"})
    raw_df["abs_ILD"] = raw_df["ILD"].abs()

    raw_df = raw_df[raw_df["RTwrtStim"] < max_rtwrtstim_for_plot].copy()
    raw_df = raw_df[~((raw_df["RTwrtStim"].isna()) & (raw_df["abort_event"] == 3))].copy()

    mask_led_on = raw_df["LED_trial"] == 1
    mask_repeat = raw_df["repeat_trial"].isin(allowed_repeat_trials) | raw_df["repeat_trial"].isna()
    opto_df_all = raw_df[
        mask_led_on
        & mask_repeat
        & raw_df["session_type"].isin([session_type])
        & raw_df["training_level"].isin([training_level])
        & (raw_df["success"].isin([1, -1]) | raw_df["abort_event"].isin(allowed_abort_events))
        & raw_df["ABL"].isin(supported_abl_values)
    ].copy()
    opto_source = raw_data_csv_path

opto_plot_df = opto_df_all[
    (opto_df_all["RTwrtStim"] >= 0.0)
    & (opto_df_all["RTwrtStim"] < max_rtwrtstim_for_plot)
].copy()

if len(opto_plot_df) == 0:
    raise ValueError(
        "No opto rows found after filtering. The current LED7_expert_data.csv is control-only, "
        "so opto rows are rebuilt from out_LED.csv."
    )

print("Loaded LED7 control/opto RTD datasets:")
print(f"  Control source: {expert_data_csv_path}")
print(f"  Opto source: {opto_source}")
print(f"  Batch={batch_name}, session_type={session_type}, training_level={training_level}")
print("  Trial pool: success in {1, -1} or abort_event in {3, 4}")
print(f"  Broad RTwrtStim window used for histogramming: [0, {max_rtwrtstim_for_plot:.3f}) s")
print(f"  Display x-limit only: [{xlim_s[0]:.3f}, {xlim_s[1]:.3f}] s")
print(f"  Control rows before nonnegative RT filter: {len(control_df_all)}")
print(f"  Opto rows before nonnegative RT filter: {len(opto_df_all)}")
print(f"  Control rows used for RTD histogramming: {len(control_plot_df)}")
print(f"  Opto rows used for RTD histogramming: {len(opto_plot_df)}")
print(f"  Control counts by ABL: {format_abl_counts(control_plot_df)}")
print(f"  Opto counts by ABL: {format_abl_counts(opto_plot_df)}")

for label, df in [("control", control_plot_df), ("opto", opto_plot_df)]:
    for abl_value in supported_abl_values:
        n_abl = int(np.isclose(df["ABL"].astype(float), float(abl_value)).sum())
        if n_abl == 0:
            raise ValueError(f"No {label} rows found for ABL={abl_value} after filtering.")


# %%
############ Build RTD densities ############
bin_edges_s = build_bin_edges(max_rtwrtstim_for_plot, data_bin_size_s)
bin_width_s = float(np.diff(bin_edges_s)[0])
condition_plot_dfs = {"control": control_plot_df, "opto": opto_plot_df}
condition_colors = {"control": "black", "opto": "red"}

density_payload: dict[tuple[str, int], dict[str, np.ndarray | int | float]] = {}
global_max_density_display = 0.0

for condition_name, condition_df in condition_plot_dfs.items():
    for abl_value in supported_abl_values:
        abl_df = condition_df[np.isclose(condition_df["ABL"].astype(float), float(abl_value))].copy()
        rt_values = abl_df["RTwrtStim"].to_numpy(dtype=float)
        counts, _ = np.histogram(rt_values, bins=bin_edges_s)
        density = counts / (len(abl_df) * bin_width_s)
        area = float(np.sum(density * np.diff(bin_edges_s)))

        display_bin_mask = (bin_edges_s[:-1] < xlim_s[1]) & (bin_edges_s[1:] > xlim_s[0])
        if display_bin_mask.any():
            global_max_density_display = max(
                global_max_density_display,
                float(np.max(density[display_bin_mask])),
            )

        density_payload[(condition_name, int(abl_value))] = {
            "n_total": int(len(abl_df)),
            "counts": counts,
            "density": density,
            "area": area,
        }

        n_visible = int(((rt_values >= xlim_s[0]) & (rt_values < xlim_s[1])).sum())
        print(
            f"  {condition_name}, ABL={abl_value}: N={len(abl_df)}, "
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
    panel_counts = []
    for condition_name in ["control", "opto"]:
        payload = density_payload[(condition_name, int(abl_value))]
        density = payload["density"]
        n_total = int(payload["n_total"])
        panel_counts.append(f"{condition_name}: N={n_total}")

        ax.stairs(
            density,
            bin_edges_s * 1e3,
            baseline=0.0,
            color=condition_colors[condition_name],
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
        "\n".join(panel_counts),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )

axes[0].set_ylabel("Density (1/s)")

legend_handles = [
    Line2D([0], [0], color=condition_colors["control"], lw=2.0, label="control"),
    Line2D([0], [0], color=condition_colors["opto"], lw=2.0, label="opto"),
]
fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
fig.suptitle(
    f"{batch_name} control vs opto RTD by ABL "
    f"(session_type={session_type}, training_level={training_level})",
    y=1.12,
)
fig.tight_layout()

save_figure(fig, plot_output_base)

print("Saved control/opto RTD by ABL figure:")
print(f"  {plot_output_base.with_suffix('.pdf')}")
print(f"  {plot_output_base.with_suffix('.png')}")

if show_plot:
    plt.show()
else:
    plt.close(fig)

# %%
