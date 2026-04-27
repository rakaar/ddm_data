# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# %%
############ Parameters (edit here) ############
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

input_csv_path = REPO_ROOT / "LED8_session8_training16_repeat_filtered.csv"

bin_width_s = 0.05
x_min_s = -2.0
x_max_s = 2.0
bins_wrt_led = np.arange(x_min_s, x_max_s + bin_width_s, bin_width_s)

show_plots = True
save_ext = "pdf"

animal_grid_output_path = SCRIPT_DIR / f"led8_abort_rate_rt_wrt_led_onset_animals_and_aggregate.{save_ext}"


# %%
if not input_csv_path.exists():
    raise FileNotFoundError(f"Could not find input CSV: {input_csv_path}")

df = pd.read_csv(input_csv_path)

required_columns = [
    "animal",
    "LED_trial",
    "abort_event",
    "timed_fix",
    "t_LED",
]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns in {input_csv_path}: {missing_columns}")

n_rows_before_drop = len(df)
df = df.dropna(subset=required_columns).copy()
df["animal"] = df["animal"].astype(int)
df["LED_trial"] = df["LED_trial"].astype(int)
n_rows_dropped = n_rows_before_drop - len(df)

animals = np.sort(df["animal"].unique())
bin_centers_wrt_led = (bins_wrt_led[1:] + bins_wrt_led[:-1]) / 2

print(f"Loaded {input_csv_path}")
print(f"Rows after dropna on {required_columns}: {len(df):,}")
print(f"Dropped rows: {n_rows_dropped:,}")
print(f"Animals: {animals.tolist()}")


# %%
def scaled_hist(values: np.ndarray, n_total_trials: int) -> tuple[np.ndarray, float]:
    if n_total_trials <= 0 or len(values) == 0:
        return np.zeros(len(bins_wrt_led) - 1), 0.0

    hist_density, _ = np.histogram(values, bins=bins_wrt_led, density=True)
    abort_fraction = len(values) / n_total_trials
    return hist_density * abort_fraction, abort_fraction


# %%
summary_rows = []

row_defs = [(f"Animal {animal}", df[df["animal"] == animal].copy()) for animal in animals]
row_defs.append(("Aggregate", df.copy()))

panel_results = []
global_y_max = 0.0

for row_label, row_df in row_defs:
    panel_text_lines = []
    cond_results = []

    for led_value, cond_label, color in [(1, "LED ON", "tab:red"), (0, "LED OFF", "tab:blue")]:
        cond_df = row_df[row_df["LED_trial"] == led_value].copy()
        abort_df = cond_df[cond_df["abort_event"] == 3].copy()
        abort_rts_wrt_led = (abort_df["timed_fix"] - abort_df["t_LED"]).to_numpy()

        hist_scaled, abort_fraction = scaled_hist(abort_rts_wrt_led, len(cond_df))
        global_y_max = max(global_y_max, float(hist_scaled.max(initial=0.0)))

        cond_results.append(
            {
                "cond_label": cond_label,
                "color": color,
                "hist_scaled": hist_scaled,
            }
        )
        panel_text_lines.append(
            f"{cond_label}: n={len(cond_df):,}, a={len(abort_df):,}, f={abort_fraction:.3f}"
        )

        summary_rows.append(
            {
                "group": row_label,
                "condition": cond_label,
                "n_total_trials": len(cond_df),
                "n_abort_event_3": len(abort_df),
                "abort_fraction": abort_fraction,
                "in_range_abort_count": int(
                    ((abort_rts_wrt_led >= x_min_s) & (abort_rts_wrt_led <= x_max_s)).sum()
                ),
            }
        )

    panel_results.append(
        {
            "row_label": row_label,
            "panel_text_lines": panel_text_lines,
            "cond_results": cond_results,
        }
    )

y_max = 1.05 * global_y_max if global_y_max > 0 else 1.0

fig_width = max(3.2 * len(panel_results), 12.0)
fig, axes = plt.subplots(1, len(panel_results), figsize=(fig_width, 3.8), sharex=True, sharey=True)
if len(panel_results) == 1:
    axes = np.array([axes])

for panel_idx, panel_result in enumerate(panel_results):
    ax = axes[panel_idx]

    for cond_result in panel_result["cond_results"]:
        label = cond_result["cond_label"] if panel_idx == 0 else None
        ax.stairs(
            cond_result["hist_scaled"],
            bins_wrt_led,
            color=cond_result["color"],
            linewidth=1.8,
            label=label,
        )

    ax.axvline(0.0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlim(x_min_s, x_max_s)
    ax.set_ylim(0.0, y_max)
    ax.set_title(panel_result["row_label"])
    ax.set_xlabel("timed_fix - t_LED (s)")
    if panel_idx == 0:
        ax.set_ylabel("Abort rate")



fig.suptitle(
    "LED8 abort rate vs RT wrt LED onset by animal and aggregate\n"
    "area of each histogram equals abort fraction within the LED condition",
    y=1.02,
)
axes[0].legend(fontsize=8, loc="upper right")
fig.tight_layout(rect=(0.02, 0.0, 1.0, 0.96))
fig.savefig(animal_grid_output_path, bbox_inches="tight")

if show_plots:
    plt.show()
else:
    plt.close(fig)

print(f"Saved {animal_grid_output_path}")


# %%
summary_df = pd.DataFrame(summary_rows)
print("\nSummary")
print(summary_df.to_string(index=False))
