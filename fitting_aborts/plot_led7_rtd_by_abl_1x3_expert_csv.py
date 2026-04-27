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
allowed_abort_events = [3, 4]
supported_abl_values = (20, 40, 60)
xlim_s = (0.0, 0.120)
data_bin_size_s = 5e-3
max_rtwrtstim_for_plot = 1.0

check_against_raw_out_led = True
show_plot = False
png_dpi = 300
panel_width = 4.2
panel_height = 3.6

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
expert_data_csv_path = REPO_ROOT / "LED7_expert_data.csv"
raw_data_csv_path = REPO_ROOT / "out_LED.csv"
output_dir = SCRIPT_DIR / "led7_rtd_by_abl_1x3_expert_csv"
plot_output_base = output_dir / "led7_rtd_by_abl_1x3_expert_csv_xlim_0_120ms"


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
############ Load and validate LED7 expert CSV ############
if not expert_data_csv_path.exists():
    raise FileNotFoundError(f"Could not find expert CSV: {expert_data_csv_path}")

output_dir.mkdir(parents=True, exist_ok=True)

expert_df = pd.read_csv(expert_data_csv_path)

required_columns = [
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
missing_columns = [column for column in required_columns if column not in expert_df.columns]
if missing_columns:
    raise ValueError(f"Expert CSV is missing required columns: {missing_columns}")

condition_failures = []
if not expert_df["batch_name"].eq(batch_name).all():
    condition_failures.append(f"batch_name is not all {batch_name}")
if not expert_df["session_type"].isin([session_type]).all():
    condition_failures.append(f"session_type is not all {session_type}")
if not expert_df["training_level"].isin([training_level]).all():
    condition_failures.append(f"training_level is not all {training_level}")
if not ((expert_df["LED_trial"] == 0) | expert_df["LED_trial"].isna()).all():
    condition_failures.append("LED_trial is not all LED-OFF")
if not (expert_df["repeat_trial"].isin(allowed_repeat_trials) | expert_df["repeat_trial"].isna()).all():
    condition_failures.append(f"repeat_trial is not in {allowed_repeat_trials} or missing")
if not (
    expert_df["success"].isin([1, -1]) | expert_df["abort_event"].isin(allowed_abort_events)
).all():
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

observed_abl_values = np.sort(expert_df["ABL"].dropna().astype(float).unique())
unexpected_abl_values = [
    float(abl)
    for abl in observed_abl_values
    if not any(np.isclose(float(abl), float(supported)) for supported in supported_abl_values)
]
if unexpected_abl_values:
    condition_failures.append(
        f"unexpected ABL values {unexpected_abl_values}; supported values are {supported_abl_values}"
    )

if condition_failures:
    raise ValueError("Expert CSV condition check failed:\n  - " + "\n  - ".join(condition_failures))

plot_df = expert_df[
    expert_df["ABL"].isin(supported_abl_values)
    & (expert_df["RTwrtStim"] >= 0.0)
    & (expert_df["RTwrtStim"] < max_rtwrtstim_for_plot)
].copy()

if len(plot_df) == 0:
    raise ValueError("No LED7 expert rows found after nonnegative RTwrtStim filtering.")

print("Loaded LED7 expert RTD dataset:")
print(f"  Expert CSV: {expert_data_csv_path}")
print(f"  Batch={batch_name}, session_type={session_type}, training_level={training_level}")
print("  Expert CSV condition check: PASS")
print(f"  Expert CSV rows before nonnegative RT filter: {len(expert_df)}")
print(f"  Rows used for RTD histogramming: {len(plot_df)}")
print(f"  Broad RTwrtStim window used for histogramming: [0, {max_rtwrtstim_for_plot:.3f}) s")
print(f"  Display x-limit only: [{xlim_s[0]:.3f}, {xlim_s[1]:.3f}] s")
print(f"  Counts by ABL: {format_abl_counts(plot_df)}")

for abl_value in supported_abl_values:
    n_abl = int(np.isclose(plot_df["ABL"].astype(float), float(abl_value)).sum())
    if n_abl == 0:
        raise ValueError(f"No rows found for ABL={abl_value} after filtering.")


# %%
############ Build RTD densities by ABL from expert CSV ############
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
############ Exact RTD check against raw out_LED rebuild ############
if check_against_raw_out_led:
    if not raw_data_csv_path.exists():
        raise FileNotFoundError(f"Could not find raw CSV for RTD check: {raw_data_csv_path}")

    raw_df = pd.read_csv(raw_data_csv_path)
    raw_df["RTwrtStim"] = raw_df["timed_fix"] - raw_df["intended_fix"]
    raw_df = raw_df.rename(columns={"timed_fix": "TotalFixTime"})
    raw_df = raw_df[~((raw_df["RTwrtStim"].isna()) & (raw_df["abort_event"] == 3))].copy()

    mask_led_off = (raw_df["LED_trial"] == 0) | (raw_df["LED_trial"].isna())
    mask_repeat = raw_df["repeat_trial"].isin(allowed_repeat_trials) | raw_df["repeat_trial"].isna()
    raw_led_off_df = raw_df[
        mask_led_off
        & mask_repeat
        & raw_df["session_type"].isin([session_type])
        & raw_df["training_level"].isin([training_level])
    ].copy()

    raw_plot_df = raw_led_off_df[
        (raw_led_off_df["success"].isin([1, -1]) | raw_led_off_df["abort_event"].isin(allowed_abort_events))
        & raw_led_off_df["ABL"].isin(supported_abl_values)
        & (raw_led_off_df["RTwrtStim"] >= 0.0)
        & (raw_led_off_df["RTwrtStim"] < max_rtwrtstim_for_plot)
    ].copy()

    if len(raw_plot_df) != len(plot_df):
        raise AssertionError(
            f"Expert plot rows ({len(plot_df)}) do not match raw rebuild rows ({len(raw_plot_df)})."
        )

    mismatch_messages = []
    for abl_value in supported_abl_values:
        raw_abl_df = raw_plot_df[np.isclose(raw_plot_df["ABL"].astype(float), float(abl_value))].copy()
        expert_payload = abl_density_payload[int(abl_value)]
        raw_counts, _ = np.histogram(raw_abl_df["RTwrtStim"].to_numpy(dtype=float), bins=bin_edges_s)
        raw_density = raw_counts / (len(raw_abl_df) * bin_width_s)

        if len(raw_abl_df) != expert_payload["n_total"]:
            mismatch_messages.append(
                f"ABL={abl_value}: raw N={len(raw_abl_df)} vs expert N={expert_payload['n_total']}"
            )
            continue
        if not np.array_equal(raw_counts, expert_payload["counts"]):
            first_mismatch = int(np.flatnonzero(raw_counts != expert_payload["counts"])[0])
            mismatch_messages.append(
                f"ABL={abl_value}: first count mismatch at bin {first_mismatch}, "
                f"raw={raw_counts[first_mismatch]}, expert={expert_payload['counts'][first_mismatch]}"
            )
        if not np.array_equal(raw_density, expert_payload["density"]):
            first_mismatch = int(np.flatnonzero(raw_density != expert_payload["density"])[0])
            mismatch_messages.append(
                f"ABL={abl_value}: first density mismatch at bin {first_mismatch}, "
                f"raw={raw_density[first_mismatch]}, expert={expert_payload['density'][first_mismatch]}"
            )

    if mismatch_messages:
        raise AssertionError("Raw rebuild and expert CSV RTD check failed:\n  - " + "\n  - ".join(mismatch_messages))

    print("RTD exact check against raw out_LED rebuild: PASS")
    print(f"  Raw rebuild rows used for RTD histogramming: {len(raw_plot_df)}")
    print(f"  Raw rebuild counts by ABL: {format_abl_counts(raw_plot_df)}")
    print("  Per-ABL histogram counts and density arrays match exactly.")


# %%
############ Build and save 1 x 3 expert-CSV figure ############
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
    f"(expert CSV, session_type={session_type}, training_level={training_level})",
    y=1.02,
)
fig.tight_layout()

save_figure(fig, plot_output_base)

print("Saved expert-CSV RTD by ABL figure:")
print(f"  {plot_output_base.with_suffix('.pdf')}")
print(f"  {plot_output_base.with_suffix('.png')}")

if show_plot:
    plt.show()
else:
    plt.close(fig)

# %%
