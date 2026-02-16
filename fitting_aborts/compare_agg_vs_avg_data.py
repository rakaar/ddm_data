"""
Compare two ways of building the RT wrt LED data histogram across animals:

1) Average-of-animals:
   - Build per-animal scaled histogram (density * abort fraction)
   - Average those histograms across animals

2) Aggregate-all-data:
   - Pool all animals' RT wrt LED abort samples
   - Build one scaled histogram using pooled abort fraction

This script uses the same filtering and RT definitions as
vbmc_compare_LED_fit_average_animals.py.
"""
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ANIMALS = [92, 93, 98, 99, 100, 103]
T_TRUNC = 0.3
BINS_WRT_LED = np.arange(-3, 3, 0.025)
OUTFILE = "compare_agg_vs_avg_data_rt_wrt_led.pdf"


def scaled_hist(values: np.ndarray, n_total: int, bins: np.ndarray) -> np.ndarray:
    """Return density histogram scaled by abort fraction (area = fraction)."""
    if n_total <= 0 or len(values) == 0:
        return np.zeros(len(bins) - 1)
    frac = len(values) / n_total
    h, _ = np.histogram(values, bins=bins, density=True)
    return h * frac


# %%
# Load and filter data + initialize containers
df = pd.read_csv("../out_LED.csv")
df = df[df["repeat_trial"].isin([0, 2]) | df["repeat_trial"].isna()]
df = df[df["session_type"].isin([7])]
df = df[df["training_level"].isin([16])]
df = df.dropna(subset=["intended_fix", "LED_onset_time", "timed_fix"])
df = df[(df["abort_event"] == 3) | (df["success"].isin([1, -1]))]
df = df[~((df["abort_event"] == 3) & (df["timed_fix"] < T_TRUNC))]

bin_centers = (BINS_WRT_LED[1:] + BINS_WRT_LED[:-1]) / 2

per_animal_hist_on = []
per_animal_hist_off = []
weights_on = []
weights_off = []

pooled_on_vals = []
pooled_off_vals = []
pooled_total_on = 0
pooled_total_off = 0


# %%
# Build per-animal hist inputs + pooled arrays (fully inline for easier debugging)
print("Animal-wise counts:")
for animal in ANIMALS:
    df_animal = df[df["animal"] == animal]
    if len(df_animal) == 0:
        print(f"  Animal {animal}: no rows after filtering, skipping")
        continue

    df_on = df_animal[df_animal["LED_trial"] == 1]
    df_off = df_animal[(df_animal["LED_trial"] == 0) | (df_animal["LED_trial"].isna())]

    df_on_fit = pd.DataFrame(
        {
            "RT": df_on["timed_fix"].values,
            "t_stim": df_on["intended_fix"].values,
            "t_LED": (df_on["intended_fix"] - df_on["LED_onset_time"]).values,
            "LED_trial": 1,
        }
    )
    df_off_fit = pd.DataFrame(
        {
            "RT": df_off["timed_fix"].values,
            "t_stim": df_off["intended_fix"].values,
            "t_LED": (df_off["intended_fix"] - df_off["LED_onset_time"]).values,
            "LED_trial": 0,
        }
    )

    fit_df = pd.concat([df_on_fit, df_off_fit], ignore_index=True)
    fit_df = fit_df[~((fit_df["RT"] < fit_df["t_stim"]) & (fit_df["RT"] <= T_TRUNC))]

    df_on_aborts = fit_df[
        (fit_df["LED_trial"] == 1)
        & (fit_df["RT"] < fit_df["t_stim"])
        & (fit_df["RT"] > T_TRUNC)
    ]
    df_off_aborts = fit_df[
        (fit_df["LED_trial"] == 0)
        & (fit_df["RT"] < fit_df["t_stim"])
        & (fit_df["RT"] > T_TRUNC)
    ]

    on_vals = (df_on_aborts["RT"] - df_on_aborts["t_LED"]).values
    off_vals = (df_off_aborts["RT"] - df_off_aborts["t_LED"]).values
    n_on = int((fit_df["LED_trial"] == 1).sum())
    n_off = int((fit_df["LED_trial"] == 0).sum())

    print(
        f"  Animal {animal}: ON aborts={len(on_vals)}/{n_on}, "
        f"OFF aborts={len(off_vals)}/{n_off}"
    )

    per_animal_hist_on.append(scaled_hist(on_vals, n_on, BINS_WRT_LED))
    per_animal_hist_off.append(scaled_hist(off_vals, n_off, BINS_WRT_LED))
    weights_on.append(n_on)
    weights_off.append(n_off)

    pooled_on_vals.append(on_vals)
    pooled_off_vals.append(off_vals)
    pooled_total_on += n_on
    pooled_total_off += n_off

if len(per_animal_hist_on) == 0:
    raise RuntimeError("No animals available after filtering.")


# %%
# Compute average-of-animals vs aggregate-all-data histograms
avg_hist_on = np.mean(per_animal_hist_on, axis=0)
avg_hist_off = np.mean(per_animal_hist_off, axis=0)

# Weighted-by-trial-count average of per-animal scaled histograms
if np.sum(weights_on) > 0:
    weighted_hist_on = np.average(per_animal_hist_on, axis=0, weights=weights_on)
else:
    weighted_hist_on = np.zeros_like(avg_hist_on)

if np.sum(weights_off) > 0:
    weighted_hist_off = np.average(per_animal_hist_off, axis=0, weights=weights_off)
else:
    weighted_hist_off = np.zeros_like(avg_hist_off)

pooled_on_vals = np.concatenate(pooled_on_vals) if pooled_on_vals else np.array([])
pooled_off_vals = np.concatenate(pooled_off_vals) if pooled_off_vals else np.array([])

agg_hist_on = scaled_hist(pooled_on_vals, pooled_total_on, BINS_WRT_LED)
agg_hist_off = scaled_hist(pooled_off_vals, pooled_total_off, BINS_WRT_LED)

print("\nAreas (should equal abort fractions):")
print(f"  avg_hist_on area  = {np.trapz(avg_hist_on, bin_centers):.4f}")
print(f"  wt_hist_on area   = {np.trapz(weighted_hist_on, bin_centers):.4f}")
print(f"  agg_hist_on area  = {np.trapz(agg_hist_on, bin_centers):.4f}")
print(f"  avg_hist_off area = {np.trapz(avg_hist_off, bin_centers):.4f}")
print(f"  wt_hist_off area  = {np.trapz(weighted_hist_off, bin_centers):.4f}")
print(f"  agg_hist_off area = {np.trapz(agg_hist_off, bin_centers):.4f}")


# %%
# Plot
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    bin_centers,
    avg_hist_on,
    color="r",
    lw=2,
    alpha=0.8,
    linestyle=":",
    label="Average-of-animals: LED ON",
)
ax.plot(
    bin_centers,
    agg_hist_on,
    color="r",
    lw=2,
    alpha=0.8,
    linestyle=":",
    label="Aggregate-all-data: LED ON",
)
ax.plot(
    bin_centers,
    weighted_hist_on,
    color="k",
    lw=2,
    alpha=0.4,
    linestyle="-",
    label="Weighted-by-trials: LED ON",
)
ax.plot(
    bin_centers,
    avg_hist_off,
    color="k",
    lw=2,
    alpha=0.4,
    linestyle="-",
    label="Average-of-animals: LED OFF",
)
ax.plot(
    bin_centers,
    agg_hist_off,
    color="g",
    lw=2,
    alpha=0.8,
    linestyle="--",
    label="Aggregate-all-data: LED OFF",
)
ax.plot(
    bin_centers,
    weighted_hist_off,
    color="g",
    lw=2,
    alpha=0.8,
    linestyle="--",
    label="Weighted-by-trials: LED OFF",
)

ax.axvline(0, color="k", linestyle=":", alpha=0.6, label="LED onset")
ax.set_xlim(-0.1, 0.2)
ax.set_xlabel("RT - t_LED (s)")
ax.set_ylabel("Rate (area = fraction)")
ax.set_title("RT wrt LED data histograms: aggregate vs average")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUTFILE, bbox_inches="tight")
print(f"\nSaved: {OUTFILE}")
plt.show()

# %%
