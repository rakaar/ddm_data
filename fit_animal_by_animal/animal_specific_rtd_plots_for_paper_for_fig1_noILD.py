# %%
"""
Animal-specific RTD plots (no |ILD| dimension)
================================================
This script reproduces the analysis in
`animal_specific_rtd_plots_for_paper_for_fig1_ILD_COMBINED.py` but collapses
across ILD.  For each animal we plot:
    1. Original RTDs for ABL 20/40/60 (overlaid)
    2. Q–Q analysis (ABL 20 & 40 vs 60) to obtain a slope in the min/max RT window
    3. Rescaled RTDs using the slope so that curves overlap
Output is a single-column figure per animal and saved to one PDF.
The original ILD-based script is left intact.
"""
# %%
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from sklearn.neighbors import KernelDensity

# Flag to include abort_event == 4. If True, data with these aborts is loaded
# and filenames are updated accordingly.
INCLUDE_ABORT_EVENT_4 = True
if INCLUDE_ABORT_EVENT_4:
    CSV_SUFFIX = "_and_4"
    ABORT_EVENTS = [3, 4]
    FILENAME_SUFFIX = "_with_abort4"
else:
    CSV_SUFFIX = ""
    ABORT_EVENTS = [3]
    FILENAME_SUFFIX = ""

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
DESIRED_BATCHES = [
    "SD",
    "LED34",
    "LED6",
    "LED8",
    "LED7",
    "LED34_even",  # keep same selection as original script
]
ABL_ARR = [20, 40, 60]
RT_BIN_SIZE = 0.02
RT_BINS = np.arange(0, 1 + RT_BIN_SIZE, RT_BIN_SIZE)
MIN_RT_CUT = 0.09  # lower bound for Q–Q slope fit
MAX_RT_CUT = 0.30  # upper bound for Q–Q slope fit

ABL_COLORS = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
this_dir = os.path.dirname(__file__)
csv_dir = os.path.join(this_dir, "batch_csvs")

batch_files = [
    f"batch_{batch_name}_valid_and_aborts{CSV_SUFFIX}.csv" for batch_name in DESIRED_BATCHES
]

all_data = []
for fname in batch_files:
    fpath = os.path.join(csv_dir, fname)
    if os.path.exists(fpath):
        print(f"Loading {fpath} …")
        all_data.append(pd.read_csv(fpath))

if not all_data:
    raise FileNotFoundError(
        f"No batch CSV files found for {DESIRED_BATCHES} in '{csv_dir}' with suffix '{CSV_SUFFIX}'"
    )

merged_data = pd.concat(all_data, ignore_index=True)

# Identify batch–animal pairs (same logic as original script)
excluded_animals_led2 = {40, 41, 43}
base_pairs = set(map(tuple, merged_data[["batch_name", "animal"]].drop_duplicates().values))
batch_animal_pairs = sorted(
    (b, a) for b, a in base_pairs if not (b == "LED2" and a in excluded_animals_led2)
)

print(
    f"Found {len(batch_animal_pairs)} batch-animal pairs from {len(set(b for b, _ in batch_animal_pairs))} batches."
)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def get_animal_RTD_data(df: pd.DataFrame, abl: int):
    """Return (bin_centers, hist_density, n_trials) for one ABL, ignoring ILD."""
    condition = (
        (df["ABL"] == abl)
        & (df["RTwrtStim"] >= 0)
        & (df["RTwrtStim"] <= 1)
        & (df["success"].isin([1, -1]))
    )
    subset = df.loc[condition, "RTwrtStim"].dropna()

    bin_centers = (RT_BINS[:-1] + RT_BINS[1:]) / 2
    n_trials = len(subset)

    if n_trials == 0:
        hist = np.full_like(bin_centers, np.nan, dtype=float)
    else:
        hist, _ = np.histogram(subset, bins=RT_BINS, density=True)

    return bin_centers, hist, n_trials


def process_batch_animal(pair, animal_df):
    """Compute RT histograms & quantiles for one animal."""
    batch_name, animal_id = pair
    result = {}

    q_levels = np.arange(0.01, 1.0, 0.01)

    for abl in ABL_ARR:
        # Histogram
        bin_centers, rtd_hist, n_trials = get_animal_RTD_data(animal_df, abl)

        # Quantiles (no ILD split)
        condition = (
            (animal_df["ABL"] == abl)
            & (animal_df["RTwrtStim"] >= 0)
            & (animal_df["RTwrtStim"] <= 1)
            & (animal_df["success"].isin([1, -1]))
        )
        subset = animal_df.loc[condition, "RTwrtStim"].dropna()
        if subset.empty:
            q_vals = np.full(len(q_levels), np.nan)
        else:
            q_vals = np.quantile(subset, q_levels)

        result[abl] = {
            "empirical": {
                "bin_centers": bin_centers,
                "rtd_hist": rtd_hist,
                "n_trials": n_trials,
            },
            "quantiles": q_vals,
        }

    return pair, result


# -----------------------------------------------------------------------------
# Parallel processing per animal
# -----------------------------------------------------------------------------
animal_groups = merged_data.groupby(["batch_name", "animal"])

n_jobs = max(1, os.cpu_count() - 4)
print(f"Processing {len(animal_groups)} animal groups on {n_jobs} cores …")

results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(process_batch_animal)(name, grp) for name, grp in animal_groups if name in batch_animal_pairs
)
RTD_DATA = {pair: data for pair, data in results if data}
print(f"Completed processing {len(RTD_DATA)} animals.")

# -----------------------------------------------------------------------------
# Plot per animal
# -----------------------------------------------------------------------------
output_filename = (
    f"animal_specific_rtd_plots_noILD{FILENAME_SUFFIX}_minRT_{MIN_RT_CUT}_maxRT_{MAX_RT_CUT}_binsz_{RT_BIN_SIZE}.pdf"
)

all_fit_results = {}

with PdfPages(output_filename) as pdf:
    for (batch_name, animal_id), data in RTD_DATA.items():
        fig, axes = plt.subplots(3, 1, figsize=(5, 12), sharex=False)
        fig.suptitle(f"Animal {animal_id}  (Batch {batch_name})", fontsize=14)

        # ------------------------------------------------------------------
        # Row 1 – original RTDs
        # ------------------------------------------------------------------
        ax1 = axes[0]
        for abl in ABL_ARR:
            emp = data[abl]["empirical"]
            if emp["n_trials"] > 0:
                ax1.plot(
                    emp["bin_centers"], emp["rtd_hist"], color=ABL_COLORS[abl], lw=1.5, label=f"ABL={abl}"
                )
        ax1.set_ylabel("Density")
        ax1.set_xlim(0, 0.7)
        ax1.set_title("Original RTDs")
        ax1.legend(frameon=False)

        # ------------------------------------------------------------------
        # Row 2 – Q-Q plot and slope fit
        # ------------------------------------------------------------------
        ax2 = axes[1]
        slopes = {}
        quant60 = data[60]["quantiles"]
        q_levels = np.arange(0.01, 1.0, 0.01)
        for abl in (20, 40):
            quant_other = data[abl]["quantiles"]
            if not (np.all(np.isnan(quant60)) or np.all(np.isnan(quant_other))):
                mask = (quant60 >= MIN_RT_CUT) & (quant60 <= MAX_RT_CUT)
                if mask.sum() >= 2:
                    x = quant60[mask] - MIN_RT_CUT
                    y = (quant_other - quant60)[mask]
                    y0 = y[0]
                    slope = np.sum(x * (y - y0)) / np.sum(x**2) if np.sum(x**2) > 0 else np.nan
                    slopes[abl] = slope

                    ax2.plot(x, y - y0, "o" if abl == 20 else "s", color=ABL_COLORS[abl], alpha=0.3)
                    ax2.plot([0, x.max()], [0, slope * x.max()], color=ABL_COLORS[abl], lw=2)
                else:
                    slopes[abl] = np.nan
            else:
                slopes[abl] = np.nan
        title_parts = [f"m{abl}={slopes[abl]:.3f}" for abl in (20, 40) if not np.isnan(slopes[abl])]
        ax2.set_title("Q–Q slope " + ", ".join(title_parts) if title_parts else "Q–Q plot")
        ax2.set_ylabel("(Q_AL - Q_60)[min RT:max RT] ")
        ax2.set_xlim(0, MAX_RT_CUT)
        ax2.set_xlabel('Q_60 - min RT')
        ax2.axhline(0, color="k", ls="--", lw=0.8)

        # ------------------------------------------------------------------
        # Row 3 – rescaled RTDs
        # ------------------------------------------------------------------
        ax3 = axes[2]
        for abl in ABL_ARR:
            emp = data[abl]["empirical"]
            bc = emp["bin_centers"]
            hist = emp["rtd_hist"]
            if emp["n_trials"] == 0:
                continue

            if abl == 60:
                ax3.plot(bc, hist, color=ABL_COLORS[abl], lw=1.5)
            else:
                slope = slopes.get(abl, np.nan)
                if not np.isnan(slope) and (1 + slope) != 0:
                    xvals = np.where(bc > MIN_RT_CUT, ((bc - MIN_RT_CUT) / (1 + slope)) + MIN_RT_CUT, bc)
                    mult = np.ones_like(hist)
                    mult[bc > MIN_RT_CUT] = slope + 1
                    rescaled = hist * mult
                    ax3.plot(xvals, rescaled, color=ABL_COLORS[abl], lw=1.5)
                else:
                    ax3.plot(bc, hist, color=ABL_COLORS[abl], lw=1.5, ls=":")
        ax3.set_xlabel("RT (s)")
        ax3.set_ylabel("Density (rescaled)")
        ax3.set_xlim(0, 0.7)
        ax3.set_title("Rescaled RTDs")

        # Sync y-limits row-wise
        for ax in axes:
            ymin, ymax = ax.get_ylim()
            if ymin > 0:
                ax.set_ylim(0, ymax)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

        # store slopes
        all_fit_results[(batch_name, animal_id)] = slopes

print(f"PDF saved to {output_filename}")

# %%
# Average across animals (no ILD)
# -----------------------------------------------------------------------------
print("Generating average animal plot (no ILD)...")

BIN_CENTERS = (RT_BINS[:-1] + RT_BINS[1:]) / 2
aggregated_rtds = {abl: [] for abl in ABL_ARR}
aggregated_rescaled_data = {abl: [] for abl in ABL_ARR}

for (batch_name, animal_id), data in RTD_DATA.items():
    slopes = all_fit_results.get((batch_name, animal_id), {})
    for abl in ABL_ARR:
        emp = data[abl]["empirical"]
        if emp["n_trials"] > 0:
            rtd_hist = emp["rtd_hist"]
        else:
            rtd_hist = np.full(len(BIN_CENTERS), np.nan)
        aggregated_rtds[abl].append(rtd_hist)
        # Rescale histogram
        if abl == 60:
            rescaled = rtd_hist
        else:
            slope = slopes.get(abl, np.nan)
            if not np.isnan(slope) and (1 + slope) != 0:
                mult = np.ones_like(rtd_hist)
                mult[BIN_CENTERS > MIN_RT_CUT] = slope + 1
                rescaled_tmp = rtd_hist * mult
                xvals = np.where(
                    BIN_CENTERS > MIN_RT_CUT,
                    ((BIN_CENTERS - MIN_RT_CUT) / (1 + slope)) + MIN_RT_CUT,
                    BIN_CENTERS,
                )
                rescaled = np.interp(BIN_CENTERS, xvals, rescaled_tmp, left=0, right=0)
            else:
                rescaled = rtd_hist
        aggregated_rescaled_data[abl].append(rescaled)

avg_output_filename = (
    f"average_animal_rtd_plots_noILD{FILENAME_SUFFIX}_minRT_{MIN_RT_CUT}_maxRT_{MAX_RT_CUT}_binsz_{RT_BIN_SIZE}.png"
)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), sharex=False)
fig.suptitle("Average Animal RTD Analysis (no |ILD|)", fontsize=14)

for abl in ABL_ARR:
    avg_rtd = np.nanmean(np.array(aggregated_rtds[abl]), axis=0)
    ax1.plot(BIN_CENTERS, avg_rtd, color=ABL_COLORS[abl], lw=1.5, label=f"ABL={abl}")

    avg_rescaled = np.nanmean(np.array(aggregated_rescaled_data[abl]), axis=0)
    ax2.plot(BIN_CENTERS, avg_rescaled, color=ABL_COLORS[abl], lw=1.5)

ax1.set_ylabel("Density")
ax1.set_xlim(0, 0.7)
ax1.legend(frameon=False)

ax2.set_xlabel("RT (s)")
ax2.set_ylabel("Density (rescaled)")
ax2.set_xlim(0, 0.7)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(avg_output_filename, dpi=300)
# plt.close(fig)
plt.show(fig)

print(f"Average plot PNG saved to {avg_output_filename}")
# %%
# Average Q–Q plots and line fits across animals (no ILD)
print("Generating average Q-Q plot (no ILD)...")

x_grid = np.arange(0, MAX_RT_CUT, 0.001)
aggregated_qq = {20: [], 40: []}
slope_lists = {20: [], 40: []}

for (batch_name, animal_id), data in RTD_DATA.items():
    quant60 = data[60]["quantiles"]
    for abl in (20, 40):
        quant_other = data[abl]["quantiles"]
        if np.all(np.isnan(quant60)) or np.all(np.isnan(quant_other)):
            continue
        mask = (quant60 >= MIN_RT_CUT) & (quant60 <= MAX_RT_CUT)
        if mask.sum() < 2:
            continue
        x = quant60[mask] - MIN_RT_CUT
        y = (quant_other - quant60)[mask]
        y0 = y[0]
        y_adj = y - y0
        y_grid = np.interp(x_grid, x, y_adj, left=np.nan, right=np.nan)
        aggregated_qq[abl].append(y_grid)

        slope = all_fit_results.get((batch_name, animal_id), {}).get(abl, np.nan)
        if not np.isnan(slope):
            slope_lists[abl].append(slope)

avg_qq_diff = {
    abl: (np.nanmean(np.vstack(aggregated_qq[abl]), axis=0) if aggregated_qq[abl] else np.full_like(x_grid, np.nan))
    for abl in (20, 40)
}
avg_slope = {abl: (np.nanmean(slope_lists[abl]) if slope_lists[abl] else np.nan) for abl in (20, 40)}

avg_qq_output_filename = f"average_QQ_plots_noILD{FILENAME_SUFFIX}_minRT_{MIN_RT_CUT}_maxRT_{MAX_RT_CUT}.png"

fig, ax = plt.subplots(figsize=(6, 4))
fig.suptitle("Average Animal Q–Q Differences and Line Fits (no |ILD|)", fontsize=14)

for abl in (20, 40):
    ax.scatter(x_grid, avg_qq_diff[abl], color=ABL_COLORS[abl], s=5, alpha=0.5, label=f"ABL={abl} avg ΔQ")
    if not np.isnan(avg_slope[abl]):
        ax.plot([0, MAX_RT_CUT], [0, avg_slope[abl] * MAX_RT_CUT], color=ABL_COLORS[abl], ls="--", lw=2, label=f"ABL={abl} avg line (m={avg_slope[abl]:.3f})")

ax.set_xlabel('Q_60 - min RT')
ax.set_ylabel('(Q_AL - Q_60)[min RT:max RT]')
ax.set_xlim(0, MAX_RT_CUT)
ax.axhline(0, color='k', ls='--', lw=0.8)
# ax.legend(frameon=False)

plt.tight_layout()
plt.savefig(avg_qq_output_filename, dpi=300)
plt.show(fig)

print(f"Average Q-Q plot PNG saved to {avg_qq_output_filename}")

# %%
# =============================================================================
# KDE on RAW data across animals (no ILD)
# =============================================================================

from collections import defaultdict

aggregated_raw_rts = defaultdict(list)
aggregated_raw_rescaled_rts = defaultdict(list)

print("Aggregating raw RT data for KDE (no ILD)...")

for (batch_name, animal_id), slopes in tqdm(all_fit_results.items(), desc="Loading Raw RTs"):
    csv_path = f"batch_csvs/batch_{batch_name}_valid_and_aborts{CSV_SUFFIX}.csv"
    try:
        df_animal = pd.read_csv(csv_path)
        df_animal = df_animal[
            (df_animal["animal"] == int(animal_id))
            & (
                (df_animal["abort_event"].isin(ABORT_EVENTS))
                | (df_animal["success"].isin([1, -1]))
            )
        ]
    except FileNotFoundError:
        print(f"Warning: Could not find {csv_path}. Skipping.")
        continue

    for abl in ABL_ARR:
        df_stim = df_animal[df_animal["ABL"] == abl].copy()
        raw_rts = df_stim["RTwrtStim"].dropna().values
        valid_rts = raw_rts[(raw_rts >= -0.1) & (raw_rts <= 1)]
        if len(valid_rts) == 0:
            continue

        aggregated_raw_rts[abl].extend(valid_rts)

        slope = 0.0 if abl == 60 else slopes.get(abl, 0.0)
        if (1 + slope) == 0:
            continue  # avoid undefined scaling when slope == -1
        rescaled = np.where(
            valid_rts > MIN_RT_CUT,
            ((valid_rts - MIN_RT_CUT) / (1 + slope)) + MIN_RT_CUT,
            valid_rts,
        )
        aggregated_raw_rescaled_rts[abl].extend(rescaled)

# --- Plotting KDE ---
print("Plotting KDE curves (no ILD)...")

x_grid = np.arange(-0.1, 1, 0.001).reshape(-1, 1)
bandwidth = 0.01

kde_output_filename_raw = (
    f"average_animal_rtd_plots_KDE_RAW_DATA_noILD{FILENAME_SUFFIX}_minRT_{MIN_RT_CUT}_maxRT_{MAX_RT_CUT}.png"
)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), sharex=False)
fig.suptitle("Average Animal RTDs – KDE on Raw Data (no |ILD|)", fontsize=14)

for abl in ABL_ARR:
    # Original
    all_raw = np.array(aggregated_raw_rts.get(abl, []))
    if all_raw.size > 1:
        try:
            kde = KernelDensity(kernel="epanechnikov", bandwidth=bandwidth)
            kde.fit(all_raw.reshape(-1, 1))
            kde_y = np.exp(kde.score_samples(x_grid))
            kde_y /= np.trapz(kde_y, x_grid.ravel())
            ax1.plot(x_grid.ravel(), kde_y, color=ABL_COLORS[abl], lw=1.5, label=f"ABL={abl}")
        except Exception as e:
            print(f"KDE failed for original ABL={abl}: {e}")

    # Rescaled
    all_rescaled = np.array(aggregated_raw_rescaled_rts.get(abl, []))
    if all_rescaled.size > 1:
        try:
            kde_r = KernelDensity(kernel="epanechnikov", bandwidth=bandwidth)
            kde_r.fit(all_rescaled.reshape(-1, 1))
            kde_y_r = np.exp(kde_r.score_samples(x_grid))
            kde_y_r /= np.trapz(kde_y_r, x_grid.ravel())
            ax2.plot(x_grid.ravel(), kde_y_r, color=ABL_COLORS[abl], lw=1.5)
        except Exception as e:
            print(f"KDE failed for rescaled ABL={abl}: {e}")

ax1.set_ylabel("Density (KDE)")
ax1.set_xlim(-0.1, 0.7)
ax1.legend(frameon=False)

ax2.set_xlabel("RT (s)")
ax2.set_ylabel("Density (rescaled, KDE)")
ax2.set_xlim(-0.1, 0.7)

# Sync y-lims
y_max1 = ax1.get_ylim()[1]
y_max2 = ax2.get_ylim()[1]
ax1.set_ylim(0, y_max1 * 1.05)
ax2.set_ylim(0, y_max2 * 1.05)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(kde_output_filename_raw, dpi=300)
# plt.close(fig)
plt.show(fig)

print(f"KDE plot PNG saved to {kde_output_filename_raw}")

# %%
