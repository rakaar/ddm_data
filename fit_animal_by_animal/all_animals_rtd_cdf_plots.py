# %%
"""
all_animals_rtd_cdf_plots.py

Compute reaction-time distributions (RTDs) at 1 ms resolution for every animal, for
all |ILD| ∈ {1, 2, 4, 8, 16} and ABL ∈ {20, 40, 60}.  Then:
1. Plot the *average* RTD across animals (density) – five sub-plots (one per
   |ILD|) each showing the three ABL curves.
2. Convert each average RTD to a cumulative distribution function (CDF) and plot
   the CDF curves with the same layout.

Two PDFs are saved in the current directory:
    • all_animals_rtd_1ms.pdf
    • all_animals_rtd_cdf_1ms.pdf

The data-loading logic mirrors animal_specific_rtd_plots_for_paper_for_fig1.py so
that both scripts work on identical datasets.
"""
from __future__ import annotations

import os
import glob
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.optimize import curve_fit

# ============ CONFIGURATION ===================================================

# Whether to include abort_event == 4 trials (matches original script behaviour)
INCLUDE_ABORT_EVENT_4 = False

# Batches to analyse (identical to original script, minus LED1)
DESIRED_BATCHES = ['SD', 'LED34', 'LED6', 'LED8', 'LED7', 'LED34_even']

# Bin resolution (seconds) – 1 ms
BIN_SIZE = 0.001
BINS = np.arange(0, 1.0 + BIN_SIZE, BIN_SIZE)  # inclusive upper edge
BIN_CENTERS = (BINS[:-1] + BINS[1:]) / 2

ABL_VALUES = [20, 40, 60]
ABS_ILD_VALUES = [1, 2, 4, 8, 16]

# Output
RTD_PDF = 'all_animals_rtd_1ms.pdf'
CDF_PDF = 'all_animals_rtd_cdf_1ms.pdf'

# Plotting colours (consistent with other scripts)
ABL_COLOURS = {20: 'tab:blue', 40: 'tab:orange', 60: 'tab:green'}

# ==============================================================================

def load_data() -> pd.DataFrame:
    """Load and concatenate all desired batch CSVs into one DataFrame."""
    csv_dir = os.path.join(os.path.dirname(__file__), 'batch_csvs')

    suffix = '_and_4' if INCLUDE_ABORT_EVENT_4 else ''
    batch_files = [f'batch_{bn}_valid_and_aborts{suffix}.csv' for bn in DESIRED_BATCHES]

    dfs = []
    for fname in batch_files:
        path = os.path.join(csv_dir, fname)
        if os.path.exists(path):
            print(f"Loading {path} …")
            dfs.append(pd.read_csv(path))
        else:
            print(f"WARNING: {path} not found – skipping.")

    if not dfs:
        raise FileNotFoundError(f"None of the batch CSVs were found in {csv_dir}")

    merged = pd.concat(dfs, ignore_index=True)
    merged['abs_ILD'] = merged['ILD'].abs()
    # Keep only sensible RTs (0–1 s as in the original script)
    merged = merged[(merged['RTwrtStim'] >= 0) & (merged['RTwrtStim'] <= 1) & (merged['success'].isin([1, -1]))]
    return merged

# -----------------------------------------------------------------------------

def compute_histograms(df: pd.DataFrame) -> dict[tuple[int, int, int], np.ndarray]:
    """Return {(animal_id, abl, abs_ild) → density-hist array}."""
    def _hist(group: pd.DataFrame, ani: int):
        res = {}
        for abl in ABL_VALUES:
            for ild in ABS_ILD_VALUES:
                subset = group[(group['ABL'] == abl) & (group['abs_ILD'] == ild)]
                if len(subset):
                    hist, _ = np.histogram(subset['RTwrtStim'], bins=BINS, density=True)
                else:
                    hist = np.full(len(BIN_CENTERS), np.nan)
                res[(ani, abl, ild)] = hist
        return res

    grouped = df.groupby(['batch_name', 'animal'])
    n_jobs = max(1, os.cpu_count() - 4)
    print(f"Computing per-animal RTDs using {n_jobs} cores …")

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_hist)(grp, ani) for (_, ani), grp in grouped
    )

    hist_dict: dict[tuple[int, int, int], np.ndarray] = {}
    for d in results:
        hist_dict.update(d)
    return hist_dict

# -----------------------------------------------------------------------------

def aggregate_histograms(hists: dict[tuple[int, int, int], np.ndarray]):
    """Return {(abl, abs_ild) → [hist_ani0, hist_ani1, …]}."""
    agg = {(abl, ild): [] for abl in ABL_VALUES for ild in ABS_ILD_VALUES}
    for (ani, abl, ild), hist in hists.items():
        agg[(abl, ild)].append(hist)
    return agg

# -----------------------------------------------------------------------------
# Sigmoid fitting helpers ------------------------------------------------------

def _logistic(t: np.ndarray, L: float, t0: float, k: float) -> np.ndarray:
    """Simple 3-param logistic that rises from 0 to L."""
    return L / (1 + np.exp(-(t - t0) / k))


def fit_sigmoid_onset(t: np.ndarray, y: np.ndarray, *, threshold: float = 0.01, relative: bool = False):
    """Fit logistic to *y(t)* and return ``(onset_time, params)``.

    Parameters
    ----------
    threshold : float
        If ``relative`` is True the threshold is interpreted as a fraction of
        the asymptote *L* (e.g. 0.01 ⇒ 1 %).  Otherwise it is an *absolute*
        |ΔCDF| value (default 0.01).
    relative : bool
        Whether to treat *threshold* as relative to *L*.
    """
        # Ignore NaNs
    mask = ~np.isnan(y)
    t_fit, y_fit = t[mask], y[mask]
    if len(t_fit) < 5 or np.nanmax(y_fit) == 0:
        raise RuntimeError("Not enough data for sigmoid fit")

    L0 = np.nanmax(y_fit)
    t0_guess = t_fit[np.argmax(y_fit > L0 / 2)] if np.any(y_fit > L0 / 2) else 0.1
    k_guess = 0.02
    p0 = [L0, t0_guess, k_guess]

    bounds = ([0, 0, 1e-4], [1.5 * L0, 0.5, 1])
    popt, _ = curve_fit(_logistic, t_fit, y_fit, p0=p0, bounds=bounds, maxfev=10000)
    L, t0, k = popt

    y_target = threshold * L if relative else threshold
    if y_target <= 0 or y_target >= L:
        raise RuntimeError("Invalid threshold for onset calculation")

    onset_time = t0 + k * np.log(L / y_target - 1)
    return onset_time, popt

# -----------------------------------------------------------------------------
# Moving-average onset detection ---------------------------------------------

def detect_onset_moving_average(y: np.ndarray, *, window_bins: int = 5, threshold: float = 0.005):
    """Return first index i where mean(next window) - mean(prev window) >= threshold.
    Ideal for detecting initial rise in |ΔCDF| curves.
    """
    n = len(y)
    if n < 2 * window_bins + 1:
        return None
    for i in range(window_bins, n - window_bins):
        prev_mean = np.nanmean(y[i - window_bins : i])
        next_mean = np.nanmean(y[i + 1 : i + 1 + window_bins])
        if np.isnan(prev_mean) or np.isnan(next_mean):
            continue
        if next_mean - prev_mean >= threshold:
            return i
    return None

# -----------------------------------------------------------------------------

def plot_average_rtd(agg: dict[tuple[int, int], list[np.ndarray]]):
    with PdfPages(RTD_PDF) as pdf:
        fig, axes = plt.subplots(1, len(ABS_ILD_VALUES), figsize=(18, 4), sharey=True)
        fig.suptitle('Average RTD across animals (1 ms bins)')

        for j, ild in enumerate(ABS_ILD_VALUES):
            ax = axes[j]
            for abl in ABL_VALUES:
                key = (abl, ild)
                avg = np.nanmean(np.vstack(agg[key]), axis=0)
                ax.plot(BIN_CENTERS, avg, color=ABL_COLOURS[abl], lw=1.5, label=f'ABL {abl}')
            ax.set_title(f'|ILD|={ild}')
            ax.set_xlim(0, 1)
            if j == 0:
                ax.set_ylabel('Density')
            ax.set_xlabel('RT (s)')
        axes[0].legend()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)
    print(f"Saved average RTD plot → {RTD_PDF}")

# -----------------------------------------------------------------------------

def plot_average_cdf(agg: dict[tuple[int, int], list[np.ndarray]]):
    """Plot average CDF curves (row 1) and, beneath them, the mean absolute
    pair-wise CDF differences between ABLs (row 2).  Both rows share the same
    RT axis so users can directly compare spread between loudness levels.
    """
    with PdfPages(CDF_PDF) as pdf:
        # 2 rows: (0) CDF curves; (1) mean |ΔCDF| curves
        fig, axes = plt.subplots(2, len(ABS_ILD_VALUES), figsize=(18, 8), sharey="row")
        fig.suptitle("Average CDFs and mean pairwise differences (1 ms bins)")
        # Store mean |ΔCDF| curves by |ILD|
        mean_diff_by_ild = {}

        for j, ild in enumerate(ABS_ILD_VALUES):
            ax_cdf = axes[0, j]
            ax_diff = axes[1, j]

            # Store CDF curves for this |ILD| to compute pair-wise diffs
            cdf_by_abl = {}
            for abl in ABL_VALUES:
                key = (abl, ild)
                avg = np.nanmean(np.vstack(agg[key]), axis=0)
                cdf = np.cumsum(avg) * BIN_SIZE
                cdf_by_abl[abl] = cdf
                ax_cdf.plot(
                    BIN_CENTERS,
                    cdf,
                    color=ABL_COLOURS[abl],
                    lw=1.5,
                    label=f"ABL {abl}",
                )

            # Compute mean absolute pair-wise differences between CDF curves
            diffs = [
                np.abs(cdf_by_abl[20] - cdf_by_abl[40]),
                np.abs(cdf_by_abl[40] - cdf_by_abl[60]),
                np.abs(cdf_by_abl[20] - cdf_by_abl[60]),
            ]
            mean_diff = np.nanmean(np.vstack(diffs), axis=0)
            ax_diff.plot(BIN_CENTERS, mean_diff, color="k", lw=1.5)

            # Save for combined figure
            mean_diff_by_ild[ild] = mean_diff

            # Formatting
            ax_cdf.set_title(f"|ILD|={ild}")
            ax_cdf.set_xlim(0, 0.15)
            ax_cdf.set_ylim(0, 1)
            if j == 0:
                ax_cdf.set_ylabel("Cumulative probability")

            ax_cdf.set_xlabel("RT (s)")  # only visible for bottom row, but okay

            ax_diff.set_xlim(0, 0.15)
            if j == 0:
                ax_diff.set_ylabel("Mean |ΔCDF|")
            ax_diff.set_xlabel("RT (s)")

        # --- Onset detection by moving-average criterion ---------------------------
        onset_by_ild: dict[int, float] = {}
        cmap = plt.cm.viridis
        window_bins = int(0.010 / BIN_SIZE)  # 10-ms window
        for idx, ild in enumerate(ABS_ILD_VALUES):
            curve = mean_diff_by_ild[ild]
            onset_idx = detect_onset_moving_average(curve, window_bins=window_bins, threshold=0.005)
            onset_time = BIN_CENTERS[onset_idx] if onset_idx is not None else np.nan
            onset_by_ild[ild] = onset_time

        # Add vertical onset lines to each ΔCDF subplot
        for j, ild in enumerate(ABS_ILD_VALUES):
            onset = onset_by_ild.get(ild)
            if onset and not np.isnan(onset):
                color = plt.cm.viridis(j / (len(ABS_ILD_VALUES) - 1))
                axes[1, j].axvline(onset, color=color, ls='--', lw=1)

        # Legends and layout
        axes[0, 0].legend()
        plt.tight_layout(rect=[0, 0.04, 1, 0.96])
        pdf.savefig(fig)
        plt.close(fig)

        # ---------------- Combined ΔCDF curves (all ILDs) ----------------
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        colour_cycle = plt.cm.viridis(np.linspace(0, 1, len(ABS_ILD_VALUES)))
        for ild, col in zip(ABS_ILD_VALUES, colour_cycle):
            ax2.plot(BIN_CENTERS, mean_diff_by_ild[ild], color=col, lw=1.5, label=f"|ILD|={ild}")
            onset = onset_by_ild.get(ild)
            if onset and not np.isnan(onset):
                ax2.axvline(onset, color=col, ls='--', lw=1)
        ax2.set_xlim(0, 0.15)
        ax2.set_xlabel("RT (s)")
        ax2.set_ylabel("Mean |ΔCDF|")
        ax2.set_title("Mean |ΔCDF| curves across ILDs")
        ax2.legend()
        plt.tight_layout()
        pdf.savefig(fig2)
        plt.close(fig2)

    print("Onset times (s) by |ILD| (±10 ms windows):",
          {ild: (None if np.isnan(t) else round(float(t), 4)) for ild, t in onset_by_ild.items()})
    print(f"Saved CDF & ΔCDF plot → {CDF_PDF}")

# ==============================================================================
# %%
df = load_data()
histograms = compute_histograms(df)
aggregated = aggregate_histograms(histograms)

# %%
plot_average_rtd(aggregated)
plot_average_cdf(aggregated)

# %%
