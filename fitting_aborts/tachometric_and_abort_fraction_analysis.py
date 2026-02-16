# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, MultipleLocator


# %%
# =============================================================================
# Parameters (edit these in-cell when running interactively)
# =============================================================================
CSV_PATH = Path(__file__).resolve().parent.parent / "out_LED.csv"
SESSION_TYPE = 7
TRAINING_LEVEL = 16
T_TRUNC = 0.3


def load_filtered_data(
    csv_path: Path,
    session_type: int,
    training_level: int,
    t_trunc: float,
) -> pd.DataFrame:
    """Match the filtering used in plot_data_rt_wrt_led_all_animals.py."""
    og_df = pd.read_csv(csv_path)

    df = og_df[og_df["repeat_trial"].isin([0, 2]) | og_df["repeat_trial"].isna()]
    df = df[df["session_type"].isin([session_type])]
    df = df[df["training_level"].isin([training_level])]
    df = df.dropna(subset=["intended_fix", "LED_onset_time", "timed_fix"])
    df = df[(df["abort_event"] == 3) | (df["success"].isin([1, -1]))]

    # Same truncation as the plotting script.
    df = df[~((df["abort_event"] == 3) & (df["timed_fix"] < t_trunc))]
    return df


def compute_abort_fractions(df_animal: pd.DataFrame, t_trunc: float) -> dict:
    df_on = df_animal[df_animal["LED_trial"] == 1]
    df_off = df_animal[(df_animal["LED_trial"] == 0) | (df_animal["LED_trial"].isna())]

    fit_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "RT": df_on["timed_fix"].values,
                    "t_stim": df_on["intended_fix"].values,
                    "LED_trial": 1,
                }
            ),
            pd.DataFrame(
                {
                    "RT": df_off["timed_fix"].values,
                    "t_stim": df_off["intended_fix"].values,
                    "LED_trial": 0,
                }
            ),
        ],
        ignore_index=True,
    )

    on_abort_mask = (
        (fit_df["LED_trial"] == 1) & (fit_df["RT"] < fit_df["t_stim"]) & (fit_df["RT"] > t_trunc)
    )
    off_abort_mask = (
        (fit_df["LED_trial"] == 0) & (fit_df["RT"] < fit_df["t_stim"]) & (fit_df["RT"] > t_trunc)
    )

    n_total_on = int((fit_df["LED_trial"] == 1).sum())
    n_total_off = int((fit_df["LED_trial"] == 0).sum())
    n_abort_on = int(on_abort_mask.sum())
    n_abort_off = int(off_abort_mask.sum())

    frac_on = (n_abort_on / n_total_on) if n_total_on > 0 else 0.0
    frac_off = (n_abort_off / n_total_off) if n_total_off > 0 else 0.0

    return {
        "n_on": n_total_on,
        "abort_on": n_abort_on,
        "frac_on": frac_on,
        "n_off": n_total_off,
        "abort_off": n_abort_off,
        "frac_off": frac_off,
        "delta_on_minus_off": frac_on - frac_off,
    }


# %%
# =============================================================================
# Load + filter data (same logic as plot_data_rt_wrt_led_all_animals.py)
# =============================================================================
df = load_filtered_data(
    csv_path=CSV_PATH,
    session_type=SESSION_TYPE,
    training_level=TRAINING_LEVEL,
    t_trunc=T_TRUNC,
)


# %%
# =============================================================================
# Per-animal abort fraction table
# =============================================================================
animals = sorted(df["animal"].dropna().unique())
print(f"CSV: {CSV_PATH}")
print(
    "Filters: repeat_trial in {0,2,NaN}, "
    f"session_type={SESSION_TYPE}, training_level={TRAINING_LEVEL}, "
    f"valid trials={(df.shape[0])}, T_trunc={T_TRUNC}"
)
print("")

rows = []
for animal in animals:
    stats = compute_abort_fractions(df[df["animal"] == animal], t_trunc=T_TRUNC)
    row = {"animal": animal, **stats}
    rows.append(row)

out_df = pd.DataFrame(rows).sort_values("animal")
out_df["delta_pct"] = 100.0 * (out_df["frac_on"] - out_df["frac_off"])

display_df = out_df[["animal", "frac_on", "frac_off", "delta_pct"]].copy()
print("Per-animal abort fractions (same denominator as histogram scaling):")
print(
    display_df.to_string(
        index=False,
        formatters={
            "frac_on": "{:.4f}".format,
            "frac_off": "{:.4f}".format,
            "delta_pct": lambda x: f"{x:+.2f}%",
        },
    )
)


# %%
# =============================================================================
# Aggregate summary
# =============================================================================
total_on = int(out_df["n_on"].sum())
total_off = int(out_df["n_off"].sum())
total_abort_on = int(out_df["abort_on"].sum())
total_abort_off = int(out_df["abort_off"].sum())
agg_frac_on = (total_abort_on / total_on) if total_on > 0 else 0.0
agg_frac_off = (total_abort_off / total_off) if total_off > 0 else 0.0
agg_delta_pct = 100.0 * (agg_frac_on - agg_frac_off)

mean_frac_on = out_df["frac_on"].mean()
mean_frac_off = out_df["frac_off"].mean()
mean_delta_pct = 100.0 * (mean_frac_on - mean_frac_off)

print("")
print("Aggregate (pooled across animals):")
print(
    f"frac_on={agg_frac_on:.4f}, frac_off={agg_frac_off:.4f}, delta_pct={agg_delta_pct:+.2f}%"
)
print(
    f"Across-animal mean: frac_on={mean_frac_on:.4f}, "
    f"frac_off={mean_frac_off:.4f}, delta_pct={mean_delta_pct:+.2f}%"
)

# %%
# =============================================================================
# Tachometric parameters
# =============================================================================
EASY_ABS_ILD = {8, 16}
HARD_ABS_ILD = {1, 2, 4}
LED_CONDITION = "ON"  # "ON" or "OFF"
PLOT_LED_OFF_DOTTED_IN_6X1 = True
RT_BIN_WIDTH = 0.02
RT_MIN = 0.0
RT_MAX = 1
MIN_TRIALS_PER_BIN = 5
RT_BINS = np.arange(RT_MIN, RT_MAX + RT_BIN_WIDTH, RT_BIN_WIDTH)


def binned_accuracy(rt_vals: np.ndarray, correct_vals: np.ndarray, bin_edges: np.ndarray):
    counts, _ = np.histogram(rt_vals, bins=bin_edges)
    correct_counts, _ = np.histogram(rt_vals, bins=bin_edges, weights=correct_vals)
    acc = np.divide(
        correct_counts,
        counts,
        out=np.full_like(correct_counts, np.nan, dtype=float),
        where=counts > 0,
    )
    acc[counts < MIN_TRIALS_PER_BIN] = np.nan
    centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    return centers, acc, counts


def filter_by_led_condition(df_in: pd.DataFrame, led_condition: str) -> tuple[pd.DataFrame, str]:
    led_condition_upper = led_condition.upper()
    if led_condition_upper == "ON":
        return df_in[df_in["LED_trial"] == 1].copy(), "LED ON"
    if led_condition_upper == "OFF":
        return df_in[(df_in["LED_trial"] == 0) | (df_in["LED_trial"].isna())].copy(), "LED OFF"
    raise ValueError(f"Invalid LED_CONDITION={led_condition!r}. Use 'ON' or 'OFF'.")


# %%
# =============================================================================
# Build valid-trial tachometric dataset
# =============================================================================
valid_df_all = df[df["success"].isin([1, -1])].copy()
valid_df_all["rt_rel_stim"] = valid_df_all["timed_fix"] - valid_df_all["intended_fix"]
valid_df_all["abs_ILD"] = valid_df_all["ILD"].abs()
valid_df_all["is_correct"] = (valid_df_all["success"] == 1).astype(float)
valid_df_all = valid_df_all.dropna(subset=["rt_rel_stim", "abs_ILD", "is_correct"])

valid_df, led_label = filter_by_led_condition(valid_df_all, LED_CONDITION)
valid_df_off, _ = filter_by_led_condition(valid_df_all, "OFF")
print(f"Tachometric LED condition: {led_label} | valid trials: {len(valid_df)}")


# %%
# =============================================================================
# Plot tachometric: accuracy vs RT, easy vs hard, per animal (6 x 1)
# =============================================================================
animals = sorted(valid_df_all["animal"].dropna().unique())
fig, axes = plt.subplots(len(animals), 1, figsize=(7.5, 2.8 * len(animals)), sharex=True, sharey=True)
if len(animals) == 1:
    axes = [axes]

for ax, animal in zip(axes, animals):
    df_animal = valid_df[valid_df["animal"] == animal]

    df_easy = df_animal[df_animal["abs_ILD"].isin(EASY_ABS_ILD)]
    df_hard = df_animal[df_animal["abs_ILD"].isin(HARD_ABS_ILD)]

    centers_easy, acc_easy, counts_easy = binned_accuracy(
        df_easy["rt_rel_stim"].values,
        df_easy["is_correct"].values,
        RT_BINS,
    )
    centers_hard, acc_hard, counts_hard = binned_accuracy(
        df_hard["rt_rel_stim"].values,
        df_hard["is_correct"].values,
        RT_BINS,
    )

    ax.plot(
        centers_easy,
        acc_easy,
        color="tab:green",
        markersize=3,
        linewidth=1.8,
        label=f"Easy {led_label} |ILD|=8,16",
    )
    ax.plot(
        centers_hard,
        acc_hard,
        color="tab:orange",
        markersize=3,
        linewidth=1.8,
        label=f"Hard {led_label} |ILD|=1,2,4",
    )

    if PLOT_LED_OFF_DOTTED_IN_6X1 and led_label != "LED OFF":
        df_animal_off = valid_df_off[valid_df_off["animal"] == animal]
        df_easy_off = df_animal_off[df_animal_off["abs_ILD"].isin(EASY_ABS_ILD)]
        df_hard_off = df_animal_off[df_animal_off["abs_ILD"].isin(HARD_ABS_ILD)]

        centers_easy_off, acc_easy_off, _ = binned_accuracy(
            df_easy_off["rt_rel_stim"].values,
            df_easy_off["is_correct"].values,
            RT_BINS,
        )
        centers_hard_off, acc_hard_off, _ = binned_accuracy(
            df_hard_off["rt_rel_stim"].values,
            df_hard_off["is_correct"].values,
            RT_BINS,
        )

        ax.plot(
            centers_easy_off,
            acc_easy_off,
            color="tab:green",
            linestyle="--",
            linewidth=1.8,
            label="Easy LED OFF |ILD|=8,16",
        )
        ax.plot(
            centers_hard_off,
            acc_hard_off,
            color="tab:orange",
            linestyle="--",
            linewidth=1.8,
            label="Hard LED OFF |ILD|=1,2,4",
        )

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.8)
    ax.set_title(f"{int(animal)}")
    ax.set_xlim(0, 0.6)
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(MultipleLocator(0.025))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.tick_params(axis="x", labelbottom=True)
    ax.set_ylim(0.45, 1.02)
    ax.set_xlabel("RT")
    ax.grid(True, which="major", axis="x", alpha=0.5)
    ax.grid(True, which="minor", axis="x", linestyle="-", alpha=0.5)
    ax.grid(True, which="major", axis="y", alpha=0.25)

axes[0].set_ylabel("Accuracy")
axes[0].legend(loc="lower right", fontsize=8)
if PLOT_LED_OFF_DOTTED_IN_6X1 and led_label != "LED OFF":
    fig.suptitle(
        f"Per-animal tachometric (solid: {led_label}, dotted: LED OFF)",
        y=1.04,
    )
else:
    fig.suptitle(f"Per-animal tachometric ({led_label})", y=1.04)
fig.tight_layout()
plt.show()

# %%
# =============================================================================
# 1 x 2 tachometric: all animals together (easy vs hard)
# =============================================================================
animals = sorted(valid_df["animal"].dropna().unique())
animal_colors = plt.cm.tab10(np.linspace(0, 1, len(animals)))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True, sharey=True)
ax_easy, ax_hard = axes

for color, animal in zip(animal_colors, animals):
    df_animal = valid_df[valid_df["animal"] == animal]
    df_easy = df_animal[df_animal["abs_ILD"].isin(EASY_ABS_ILD)]
    df_hard = df_animal[df_animal["abs_ILD"].isin(HARD_ABS_ILD)]

    centers_easy, acc_easy, _ = binned_accuracy(
        df_easy["rt_rel_stim"].values,
        df_easy["is_correct"].values,
        RT_BINS,
    )
    centers_hard, acc_hard, _ = binned_accuracy(
        df_hard["rt_rel_stim"].values,
        df_hard["is_correct"].values,
        RT_BINS,
    )

    ax_easy.plot(centers_easy, acc_easy, color=color, linewidth=1.8, label=f"{int(animal)}")
    ax_hard.plot(centers_hard, acc_hard, color=color, linewidth=1.8, label=f"{int(animal)}")

for ax, title in zip(axes, ["Easy |ILD| = 8,16", "Hard |ILD| = 1,2,4"]):
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.8)
    ax.set_title(title)
    ax.set_xlim(0, 0.6)
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(MultipleLocator(0.025))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_ylim(0.45, 1.02)
    ax.set_xlabel("RT")
    ax.grid(True, which="major", axis="x", alpha=0.5)
    ax.grid(True, which="minor", axis="x", linestyle="-", alpha=0.5)
    ax.grid(True, which="major", axis="y", alpha=0.25)

ax_easy.set_ylabel("Accuracy")
ax_easy.legend(title="Animal", loc="lower right", fontsize=8)
fig.suptitle(f"Tachometric ({led_label})", y=1.04)
fig.tight_layout()
plt.show()

# %%
