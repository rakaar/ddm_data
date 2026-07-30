# %%
"""
Recreate paper Fig. 2a and 2d diagnostics for each LED7 animal.

Fig. 2a-style panels show the valid-trial RT CDF by absolute ILD.
Fig. 2d-style panels show 10 ms-binned accuracy as a function of RT.
Both figures use the animal-wise pooled-ABL stimulus-modulation onset from
find_led7_stimulus_modulation_onset_by_animal.py.
"""

# %%
# =============================================================================
# Editable parameters
# =============================================================================
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

BATCH_NAME = "LED7"
ANIMALS = (92, 93, 98, 99, 100, 103)
ABLS = (20, 40, 60)
ABS_ILDS = (1, 2, 4, 8, 16)

RT_MIN_S = 0.0
RT_MAX_S = 1.0
PLOT_RT_MAX_S = 0.300
CDF_STEP_S = 0.001
TACHOMETRIC_BIN_S = 0.010
MIN_TRIALS_PER_TACHOMETRIC_BIN = 8
CDF_REFERENCE_CUTOFF_MS = 100.0
PLOT_DPI = 300

DATA_CSV = (
    REPO_ROOT
    / "raw_data"
    / "batch_csvs"
    / "batch_LED7_valid_and_aborts.csv"
)
ONSET_SUMMARY_CSV = (
    SCRIPT_DIR
    / "led7_stimulus_modulation_onset"
    / "led7_stimulus_modulation_onset_summary.csv"
)
OUTPUT_DIR = SCRIPT_DIR / "led7_stimulus_modulation_onset"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CDF_PNG = OUTPUT_DIR / "led7_fig2a_rtcdf_by_animal.png"
TACHOMETRIC_PNG = OUTPUT_DIR / "led7_fig2d_tachometric_by_animal.png"
CDF_CSV = OUTPUT_DIR / "led7_fig2a_rtcdf_by_animal.csv"
TACHOMETRIC_CSV = (
    OUTPUT_DIR / "led7_fig2d_tachometric_by_animal.csv"
)


# %%
# =============================================================================
# Imports and paper-style plotting defaults
# =============================================================================
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Helvetica",
            "Nimbus Sans",
            "Helvetica Neue",
            "Arial",
            "Liberation Sans",
            "sans-serif",
        ],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
    }
)

ILD_COLORS = {
    1: "#000000",
    2: "#7F0000",
    4: "#C51B00",
    8: "#FF4B00",
    16: "#FFC000",
}
ONSET_COLOR = "#00BFC4"


# %%
# =============================================================================
# Load the valid LED7 trials and animal-wise modulation onsets
# =============================================================================
if not DATA_CSV.exists():
    raise FileNotFoundError(DATA_CSV)
if not ONSET_SUMMARY_CSV.exists():
    raise FileNotFoundError(
        f"{ONSET_SUMMARY_CSV} is missing. Run "
        "fitting_aborts/find_led7_stimulus_modulation_onset_by_animal.py "
        "first."
    )

raw_df = pd.read_csv(DATA_CSV)
required_columns = ["animal", "success", "RTwrtStim", "ABL", "ILD"]
missing_columns = [
    column for column in required_columns if column not in raw_df.columns
]
if missing_columns:
    raise KeyError(f"Missing columns in {DATA_CSV}: {missing_columns}")

valid_df = raw_df[
    raw_df["animal"].astype(int).isin(ANIMALS)
    & raw_df["success"].isin([1, -1])
    & raw_df["RTwrtStim"].ge(RT_MIN_S)
    & raw_df["RTwrtStim"].lt(RT_MAX_S)
    & raw_df["ABL"].isin(ABLS)
    & raw_df["ILD"].abs().isin(ABS_ILDS)
].copy()
valid_df = valid_df.dropna(subset=required_columns)
valid_df["animal"] = valid_df["animal"].astype(int)
valid_df["ABL"] = valid_df["ABL"].astype(int)
valid_df["abs_ILD"] = valid_df["ILD"].abs().astype(int)
valid_df["correct"] = valid_df["success"].eq(1).astype(int)

onset_df = pd.read_csv(ONSET_SUMMARY_CSV)
onset_df = onset_df[
    onset_df["batch_name"].eq(BATCH_NAME)
    & onset_df["animal"].astype(int).isin(ANIMALS)
].copy()
onset_df["animal"] = onset_df["animal"].astype(int)

observed_animals = tuple(sorted(valid_df["animal"].unique()))
onset_animals = tuple(sorted(onset_df["animal"].unique()))
expected_animals = tuple(sorted(ANIMALS))
if observed_animals != expected_animals:
    raise RuntimeError(
        f"Expected data for animals {expected_animals}, found "
        f"{observed_animals}."
    )
if onset_animals != expected_animals:
    raise RuntimeError(
        f"Expected cutoffs for animals {expected_animals}, found "
        f"{onset_animals}."
    )

cutoff_ms_by_animal = onset_df.set_index("animal")[
    "robust_cutoff_ms"
].to_dict()

print(f"Data CSV: {DATA_CSV}")
print(f"Cutoff CSV: {ONSET_SUMMARY_CSV}")
print(f"Valid LED7 trials in [0, 1) s: {len(valid_df):,}")
print(
    "Animal-wise pooled-ABL cutoffs (ms): "
    + ", ".join(
        f"{animal}={cutoff_ms_by_animal[animal]:.0f}"
        for animal in ANIMALS
    )
)


# %%
# =============================================================================
# Build reusable RT CDF and tachometric tables
# =============================================================================
cdf_grid_s = np.arange(
    RT_MIN_S,
    PLOT_RT_MAX_S + 0.5 * CDF_STEP_S,
    CDF_STEP_S,
)
tachometric_edges_s = np.arange(
    RT_MIN_S,
    PLOT_RT_MAX_S + TACHOMETRIC_BIN_S,
    TACHOMETRIC_BIN_S,
)

cdf_rows = []
tachometric_rows = []

for animal in ANIMALS:
    animal_df = valid_df[valid_df["animal"].eq(animal)]

    for abs_ild in ABS_ILDS:
        condition_df = animal_df[animal_df["abs_ILD"].eq(abs_ild)]
        condition_rts = np.sort(
            condition_df["RTwrtStim"].to_numpy(dtype=float)
        )
        condition_cdf = (
            np.searchsorted(condition_rts, cdf_grid_s, side="right")
            / len(condition_rts)
        )
        cdf_rows.extend(
            {
                "batch_name": BATCH_NAME,
                "animal": animal,
                "abs_ILD": abs_ild,
                "RT_s": rt_s,
                "RT_ms": rt_s * 1000.0,
                "cdf": cdf_value,
                "n_condition": len(condition_rts),
            }
            for rt_s, cdf_value in zip(cdf_grid_s, condition_cdf)
        )

        condition_rts = condition_df["RTwrtStim"].to_numpy(dtype=float)
        condition_correct = condition_df["correct"].to_numpy(dtype=int)
        bin_indices = np.digitize(
            condition_rts,
            tachometric_edges_s,
            right=False,
        ) - 1

        for bin_idx in range(len(tachometric_edges_s) - 1):
            in_bin = bin_indices == bin_idx
            n_trials = int(in_bin.sum())
            if n_trials:
                accuracy = float(condition_correct[in_bin].mean())
                accuracy_sem = float(
                    np.sqrt(accuracy * (1.0 - accuracy) / n_trials)
                )
            else:
                accuracy = np.nan
                accuracy_sem = np.nan

            tachometric_rows.append(
                {
                    "batch_name": BATCH_NAME,
                    "animal": animal,
                    "abs_ILD": abs_ild,
                    "RT_bin_left_s": tachometric_edges_s[bin_idx],
                    "RT_bin_right_s": tachometric_edges_s[bin_idx + 1],
                    "RT_bin_center_s": (
                        tachometric_edges_s[bin_idx]
                        + 0.5 * TACHOMETRIC_BIN_S
                    ),
                    "RT_bin_center_ms": (
                        tachometric_edges_s[bin_idx]
                        + 0.5 * TACHOMETRIC_BIN_S
                    )
                    * 1000.0,
                    "n_trials": n_trials,
                    "accuracy": accuracy,
                    "accuracy_sem": accuracy_sem,
                }
            )

cdf_df = pd.DataFrame(cdf_rows)
tachometric_df = pd.DataFrame(tachometric_rows)
cdf_df.to_csv(CDF_CSV, index=False)
tachometric_df.to_csv(TACHOMETRIC_CSV, index=False)


# %%
# =============================================================================
# Fig. 2a-style animal-wise RT CDFs
# =============================================================================
fig_cdf, axes_cdf = plt.subplots(
    3,
    2,
    figsize=(7.2, 8.4),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)

for ax, animal in zip(axes_cdf.flat, ANIMALS):
    animal_cdf = cdf_df[cdf_df["animal"].eq(animal)]
    for abs_ild in ABS_ILDS:
        condition_cdf = animal_cdf[
            animal_cdf["abs_ILD"].eq(abs_ild)
        ]
        ax.plot(
            condition_cdf["RT_ms"],
            condition_cdf["cdf"],
            color=ILD_COLORS[abs_ild],
            lw=1.6,
            label=rf"$|ILD|={abs_ild}$",
        )

    ax.axvline(CDF_REFERENCE_CUTOFF_MS, color=ONSET_COLOR, lw=1.2)
    ax.set_title(f"LED7/{animal}")
    ax.set_xlim(0, PLOT_RT_MAX_S * 1000.0)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([0, 100, 200, 300])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

handles, labels = axes_cdf.flat[0].get_legend_handles_labels()
axes_cdf.flat[1].legend(
    handles[::-1],
    labels[::-1],
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=False,
)
fig_cdf.supxlabel("RT (ms)")
fig_cdf.supylabel("RT c.d.f.")
fig_cdf.savefig(CDF_PNG, dpi=PLOT_DPI, bbox_inches="tight")


# %%
# =============================================================================
# Fig. 2d-style animal-wise tachometric curves
# =============================================================================
fig_tach, axes_tach = plt.subplots(
    3,
    2,
    figsize=(7.2, 8.4),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)

for ax, animal in zip(axes_tach.flat, ANIMALS):
    animal_tach = tachometric_df[
        tachometric_df["animal"].eq(animal)
    ]
    for abs_ild in ABS_ILDS:
        condition_tach = animal_tach[
            animal_tach["abs_ILD"].eq(abs_ild)
        ].copy()
        enough_trials = condition_tach["n_trials"].ge(
            MIN_TRIALS_PER_TACHOMETRIC_BIN
        )
        condition_tach.loc[
            ~enough_trials,
            ["accuracy", "accuracy_sem"],
        ] = np.nan

        rt_ms = condition_tach["RT_bin_center_ms"].to_numpy(dtype=float)
        accuracy_pct = (
            condition_tach["accuracy"].to_numpy(dtype=float) * 100.0
        )
        sem_pct = (
            condition_tach["accuracy_sem"].to_numpy(dtype=float) * 100.0
        )
        ax.fill_between(
            rt_ms,
            np.clip(accuracy_pct - sem_pct, 0.0, 100.0),
            np.clip(accuracy_pct + sem_pct, 0.0, 100.0),
            color=ILD_COLORS[abs_ild],
            alpha=0.16,
            linewidth=0,
        )
        ax.plot(
            rt_ms,
            accuracy_pct,
            color=ILD_COLORS[abs_ild],
            lw=1.6,
            label=rf"$|ILD|={abs_ild}$",
        )

    cutoff_ms = float(cutoff_ms_by_animal[animal])
    ax.axvline(cutoff_ms, color=ONSET_COLOR, lw=1.2)
    ax.set_title(f"LED7/{animal}  onset {cutoff_ms:.0f} ms")
    ax.set_xlim(0, PLOT_RT_MAX_S * 1000.0)
    ax.set_ylim(40, 100)
    ax.set_xticks([0, 100, 200, 300])
    ax.set_yticks([50, 75, 100])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

handles, labels = axes_tach.flat[0].get_legend_handles_labels()
axes_tach.flat[1].legend(
    handles[::-1],
    labels[::-1],
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=False,
)
fig_tach.supxlabel("RT (ms)")
fig_tach.supylabel("Accuracy (%)")
fig_tach.savefig(TACHOMETRIC_PNG, dpi=PLOT_DPI, bbox_inches="tight")

print(f"Saved: {CDF_PNG}")
print(f"Saved: {TACHOMETRIC_PNG}")
print(f"Saved: {CDF_CSV}")
print(f"Saved: {TACHOMETRIC_CSV}")
