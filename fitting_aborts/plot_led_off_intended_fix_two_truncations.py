# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# %%
# =============================================================================
# Parameters
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

batch_name = "LED7"
session_type = 7
training_level = 16
allowed_repeat_trials = [0, 2]
max_rtwrtstim_for_fit = 1.0

truncate_values_s = [0.100, 0.115, 0.130, 0.145, 0.160]

led_data_csv_path = REPO_ROOT / "out_LED.csv"
output_dir = SCRIPT_DIR / "led_off_intended_fix_truncation_sweep"
output_dir.mkdir(parents=True, exist_ok=True)

n_bins = 60
hist_density = True
show_plot = True

truncate_colors = plt.cm.tab10(np.linspace(0, 0.9, len(truncate_values_s)))
added_interval_colors = plt.cm.Greys(np.linspace(0.45, 0.85, max(len(truncate_values_s) - 1, 1)))


# %%
# =============================================================================
# Helpers
# =============================================================================
def empirical_cdf(values):
    sorted_values = np.sort(np.asarray(values, dtype=float))
    cdf = np.arange(1, len(sorted_values) + 1, dtype=float) / float(len(sorted_values))
    return sorted_values, cdf


def ks_statistic(x, y):
    x = np.sort(np.asarray(x, dtype=float))
    y = np.sort(np.asarray(y, dtype=float))
    support = np.sort(np.unique(np.concatenate([x, y])))
    x_cdf = np.searchsorted(x, support, side="right") / float(len(x))
    y_cdf = np.searchsorted(y, support, side="right") / float(len(y))
    return float(np.max(np.abs(x_cdf - y_cdf)))


def summarize(values):
    values = np.asarray(values, dtype=float)
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=0)),
        "q025": float(np.quantile(values, 0.025)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "q975": float(np.quantile(values, 0.975)),
    }


def format_abl_counts(df):
    counts = (
        df["ABL"]
        .astype(float)
        .round()
        .astype(int)
        .value_counts()
        .sort_index()
        .to_dict()
    )
    return {int(k): int(v) for k, v in counts.items()}


def trunc_label_ms(truncate_s):
    return f"{int(round(truncate_s * 1e3))} ms"


def trunc_slug_ms(truncate_s):
    return f"{int(round(truncate_s * 1e3))}ms"


# %%
# =============================================================================
# Load and filter LED-OFF valid trials exactly like the fit scripts
# =============================================================================
if not led_data_csv_path.exists():
    raise FileNotFoundError(f"Could not find data CSV: {led_data_csv_path}")

exp_df = pd.read_csv(led_data_csv_path)
exp_df["RTwrtStim"] = exp_df["timed_fix"] - exp_df["intended_fix"]
exp_df = exp_df.rename(columns={"timed_fix": "TotalFixTime"})
exp_df = exp_df[exp_df["RTwrtStim"] < 1]
exp_df = exp_df[~((exp_df["RTwrtStim"].isna()) & (exp_df["abort_event"] == 3))].copy()

mask_led_off = (exp_df["LED_trial"] == 0) | (exp_df["LED_trial"].isna())
mask_repeat = exp_df["repeat_trial"].isin(allowed_repeat_trials) | exp_df["repeat_trial"].isna()
exp_df_led_off = exp_df[
    mask_led_off
    & mask_repeat
    & exp_df["session_type"].isin([session_type])
    & exp_df["training_level"].isin([training_level])
].copy()

df_valid = exp_df_led_off[exp_df_led_off["success"].isin([1, -1])].copy()
df_valid_fit_window = df_valid[
    (df_valid["RTwrtStim"] > 0) & (df_valid["RTwrtStim"] < max_rtwrtstim_for_fit)
].copy()

truncate_values_s = sorted({float(t) for t in truncate_values_s})
if len(truncate_values_s) < 2:
    raise ValueError("truncate_values_s must contain at least two truncation values.")
if any(t <= 0 for t in truncate_values_s):
    raise ValueError(f"truncate_values_s must all be positive. Got {truncate_values_s}.")

truncation_data = []
for truncate_s in truncate_values_s:
    df_trunc = df_valid_fit_window[df_valid_fit_window["RTwrtStim"] <= truncate_s].copy()
    values = df_trunc["intended_fix"].dropna().to_numpy(dtype=float)
    if len(values) == 0:
        raise RuntimeError(f"Truncated valid-trial dataset is empty for {trunc_label_ms(truncate_s)}.")

    truncation_data.append(
        {
            "truncate_s": truncate_s,
            "label": trunc_label_ms(truncate_s),
            "slug": trunc_slug_ms(truncate_s),
            "df": df_trunc,
            "values": values,
            "stats": summarize(values),
            "abl_counts": format_abl_counts(df_trunc),
        }
    )

added_interval_data = []
for idx, (lower_s, upper_s) in enumerate(zip(truncate_values_s[:-1], truncate_values_s[1:])):
    df_upper = truncation_data[idx + 1]["df"]
    df_added = df_upper[df_upper["RTwrtStim"] > lower_s].copy()
    values = df_added["intended_fix"].dropna().to_numpy(dtype=float)
    added_interval_data.append(
        {
            "lower_s": lower_s,
            "upper_s": upper_s,
            "label": f"({trunc_label_ms(lower_s)}, {trunc_label_ms(upper_s)}]",
            "color": added_interval_colors[idx],
            "df": df_added,
            "values": values,
            "stats": summarize(values) if len(values) > 0 else None,
        }
    )

adjacent_ks = []
for previous, current in zip(truncation_data[:-1], truncation_data[1:]):
    adjacent_ks.append(
        {
            "left_label": previous["label"],
            "right_label": current["label"],
            "ks": ks_statistic(previous["values"], current["values"]),
        }
    )

print(
    f"LED-OFF valid-trial intended_fix comparison across truncations, batch={batch_name}, "
    f"session_type={session_type}, training_level={training_level}"
)
print(f"Valid trials in fit window (0 < RTwrtStim < {max_rtwrtstim_for_fit:.3f}s): {len(df_valid_fit_window)}")
print()
print(
    f"{'Dataset':<20} {'n':>8} {'mean':>10} {'median':>10} "
    f"{'std':>10} {'q2.5':>10} {'q97.5':>10}"
)
print("-" * 82)
for entry in truncation_data:
    stats = entry["stats"]
    print(
        f"{entry['label']:<20} {stats['n']:>8d} {stats['mean']:>10.4f} {stats['median']:>10.4f} "
        f"{stats['std']:>10.4f} {stats['q025']:>10.4f} {stats['q975']:>10.4f}"
    )
print()
for entry in truncation_data:
    print(f"ABL counts at {entry['label']}: {entry['abl_counts']}")
print()
print("Adjacent KS statistics on intended_fix distributions:")
for ks_entry in adjacent_ks:
    print(f"  {ks_entry['left_label']} vs {ks_entry['right_label']}: KS={ks_entry['ks']:.6f}")
print()
print(
    f"{'Added interval':<20} {'n':>8} {'mean':>10} {'median':>10} "
    f"{'std':>10} {'q2.5':>10} {'q97.5':>10}"
)
print("-" * 82)
for interval in added_interval_data:
    stats = interval["stats"]
    if stats is None:
        continue
    print(
        f"{interval['label']:<20} {stats['n']:>8d} {stats['mean']:>10.4f} "
        f"{stats['median']:>10.4f} {stats['std']:>10.4f} "
        f"{stats['q025']:>10.4f} {stats['q975']:>10.4f}"
    )


# %%
# =============================================================================
# Plot intended_fix distributions
# =============================================================================
all_values = np.concatenate([entry["values"] for entry in truncation_data])
bins = np.linspace(float(all_values.min()), float(all_values.max()), n_bins + 1)

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

for color, entry in zip(truncate_colors, truncation_data):
    axes[0].hist(
        entry["values"],
        bins=bins,
        density=hist_density,
        histtype="step",
        linewidth=2.0,
        color=color,
        label=(
            f"{entry['label']} (n={entry['stats']['n']}, "
            f"mean={entry['stats']['mean']:.3f}s)"
        ),
    )
axes[0].set_xlabel("intended_fix (s)")
axes[0].set_ylabel("Density")
axes[0].set_title("Histogram")
axes[0].grid(axis="y", alpha=0.25)
axes[0].legend(fontsize=9)

for color, entry in zip(truncate_colors, truncation_data):
    sorted_values, cdf = empirical_cdf(entry["values"])
    axes[1].plot(
        sorted_values,
        cdf,
        color=color,
        linewidth=2.0,
        label=entry["label"],
    )
axes[1].set_xlabel("intended_fix (s)")
axes[1].set_ylabel("Empirical CDF")
axes[1].set_title("CDF")
axes[1].grid(alpha=0.25)
axes[1].legend(fontsize=9)

fig.suptitle(
    "LED-OFF valid-trial intended_fix distributions after RT truncation\n"
    "No fixed-trial subsampling; success in [1, -1] only; "
    f"truncations={', '.join(entry['label'] for entry in truncation_data)}",
    y=1.02,
)
fig.tight_layout()

plot_base = "led_off_intended_fix_compare_" + "_".join(entry["slug"] for entry in truncation_data)
pdf_path = output_dir / f"{plot_base}.pdf"
png_path = output_dir / f"{plot_base}.png"
fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, dpi=300, bbox_inches="tight")
print(f"\nSaved plot (PDF): {pdf_path}")
print(f"Saved plot (PNG): {png_path}")

if show_plot:
    plt.show()
plt.close(fig)


# %%
# =============================================================================
# Added-trials-only plots for adjacent truncation intervals
# =============================================================================
non_empty_added_intervals = [interval for interval in added_interval_data if len(interval["values"]) > 0]
if non_empty_added_intervals:
    n_panels = len(non_empty_added_intervals)
    n_cols = 2
    n_rows = int(np.ceil(n_panels / n_cols))
    fig_added, axes_added = plt.subplots(n_rows, n_cols, figsize=(12.5, 4.4 * n_rows))
    axes_added = np.atleast_1d(axes_added).ravel()

    for ax_added, interval in zip(axes_added, non_empty_added_intervals):
        stats = interval["stats"]
        ax_added.hist(
            interval["values"],
            bins=bins,
            density=hist_density,
            histtype="step",
            linewidth=2.0,
            color=interval["color"],
            label=f"n={stats['n']}, mean={stats['mean']:.3f}s",
        )
        ax_added.set_xlabel("intended_fix (s)")
        ax_added.set_ylabel("Density")
        ax_added.set_title(f"Added trials in {interval['label']}")
        ax_added.grid(axis="y", alpha=0.25)
        ax_added.legend(fontsize=9)

    for ax_added in axes_added[n_panels:]:
        ax_added.axis("off")

    fig_added.suptitle(
        "intended_fix of valid trials added between adjacent RT truncations",
        y=1.01,
    )
    fig_added.tight_layout()

    added_pdf_path = output_dir / f"{plot_base}_added_intervals.pdf"
    added_png_path = output_dir / f"{plot_base}_added_intervals.png"
    fig_added.savefig(added_pdf_path, bbox_inches="tight")
    fig_added.savefig(added_png_path, dpi=300, bbox_inches="tight")
    print(f"Saved added-intervals plot (PDF): {added_pdf_path}")
    print(f"Saved added-intervals plot (PNG): {added_png_path}")

    if show_plot:
        plt.show()
    plt.close(fig_added)
