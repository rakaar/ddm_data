# %%
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gamma_omega_alpha_utils import (
    build_cond_fit_arrays,
    load_batch_animal_pairs,
    print_batch_animal_table,
)
from omega_slope_compare_utils import (
    compute_animal_omega_ratio_values,
    compute_abs_omega_summary,
    compute_omega_ratio_summary,
    compute_quantile_slope_summary,
    load_batch_csv_data,
)


# %% Parameters
SCRIPT_DIR = os.path.dirname(__file__)
REPO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DESIRED_BATCHES = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]
BATCH_DIR = os.path.join(REPO_DIR, "fit_animal_by_animal", "batch_csvs")
COND_FIT_PKL_DIR = os.path.join(SCRIPT_DIR, "each_animal_cond_fit_gama_omega_pkl_files")

ABLS = [20, 40, 60]
ABLS_TO_COMPARE = [20, 40]
BASELINE_ABL = 60
ABS_ILDS = [1, 2, 4, 8, 16]
SIGNED_ILDS = np.sort([-16, -8, -4, -2, -1, 1, 2, 4, 8, 16])

FITTING_QUANTILES = np.arange(0.01, 1.0, 0.01)
MIN_RT_CUT_BY_ILD = {1: 0.0865, 2: 0.0865, 4: 0.0885, 8: 0.0785, 16: 0.0615}
MAX_RT_CUT = 0.3
MIN_TRIALS = 5

N_POSTERIOR_SAMPLES = int(1e5)

COLORS = {20: "tab:blue", 40: "tab:orange"}
FIG_PATH = os.path.join(SCRIPT_DIR, "quantile_slope_vs_cond_omega_ratio.png")
ANIMALWISE_FIG_PATH = os.path.join(SCRIPT_DIR, "animalwise_quantile_slope_vs_cond_omega_ratio.png")
ANIMALWISE_CSV_PATH = os.path.join(SCRIPT_DIR, "animalwise_quantile_slope_vs_cond_omega_ratio.csv")
PKL_PATH = os.path.join(SCRIPT_DIR, "quantile_slope_vs_cond_omega_ratio.pkl")


# %% Load animal list and recompute quantile slopes
batch_animal_pairs = load_batch_animal_pairs(BATCH_DIR, DESIRED_BATCHES)
print_batch_animal_table(batch_animal_pairs)

merged_data = load_batch_csv_data(BATCH_DIR, DESIRED_BATCHES)
slope_summary = compute_quantile_slope_summary(
    merged_data,
    batch_animal_pairs,
    ABLS_TO_COMPARE,
    BASELINE_ABL,
    ABS_ILDS,
    FITTING_QUANTILES,
    MIN_RT_CUT_BY_ILD,
    MAX_RT_CUT,
    min_trials=MIN_TRIALS,
)


# %% Load condition-by-condition omega fits and compute omega ratios
_, omega_cond_by_cond_fit_all_animals, missing_files = build_cond_fit_arrays(
    batch_animal_pairs,
    ABLS,
    SIGNED_ILDS,
    COND_FIT_PKL_DIR,
    n_samples=N_POSTERIOR_SAMPLES,
)

if len(missing_files) > 0:
    print(f"Missing condition-fit pickle files: {len(missing_files)}")

omega_abs_summary = compute_abs_omega_summary(
    omega_cond_by_cond_fit_all_animals,
    ABLS,
    SIGNED_ILDS,
    ABS_ILDS,
)
omega_ratio_summary = compute_omega_ratio_summary(
    omega_abs_summary,
    ABLS_TO_COMPARE,
    BASELINE_ABL,
)
animal_omega_ratio_values = compute_animal_omega_ratio_values(
    omega_abs_summary,
    ABLS_TO_COMPARE,
    BASELINE_ABL,
)


# %% Print comparison table
print("\nQuantile slope-derived ratio vs condition-fit omega ratio")
print("x = 1 + quantile slope, y = omega_60 / omega_ABL")
for ABL in ABLS_TO_COMPARE:
    print(f"\nABL {ABL} compared with baseline ABL {BASELINE_ABL}")
    print("abs_ILD  raw_slope  1+slope  slope_n  omega_ratio  omega_n")
    for ild_idx, abs_ild in enumerate(ABS_ILDS):
        print(
            f"{abs_ild:>7}  "
            f"{slope_summary['raw_slope_mean'][ABL][ild_idx]:>9.4f}  "
            f"{slope_summary['slope_ratio_mean'][ABL][ild_idx]:>8.4f}  "
            f"{int(slope_summary['raw_slope_n'][ABL][ild_idx]):>7}  "
            f"{omega_ratio_summary['omega_ratio_mean'][ABL][ild_idx]:>11.4f}  "
            f"{int(omega_ratio_summary['omega_ratio_n'][ABL][ild_idx]):>7}"
        )


# %% Plot quantile slope ratio vs omega ratio
fig, ax = plt.subplots(1, 2, figsize=(11, 5), sharex=False, sharey=False)

for panel_idx, ABL in enumerate(ABLS_TO_COMPARE):
    curr_ax = ax[panel_idx]
    color = COLORS[ABL]

    x = slope_summary["slope_ratio_mean"][ABL]
    y = omega_ratio_summary["omega_ratio_mean"][ABL]

    curr_ax.plot(
        x,
        y,
        marker="o",
        linestyle="-",
        color=color,
    )

    finite = np.isfinite(x) & np.isfinite(y)
    if np.any(finite):
        min_lim = np.nanmin([np.nanmin(x[finite]), np.nanmin(y[finite])])
        max_lim = np.nanmax([np.nanmax(x[finite]), np.nanmax(y[finite])])
        padding = 0.08 * (max_lim - min_lim)
        curr_ax.plot(
            [min_lim - padding, max_lim + padding],
            [min_lim - padding, max_lim + padding],
            color="black",
            linestyle="--",
            linewidth=1,
            alpha=0.5,
        )
        curr_ax.set_xlim(min_lim - padding, max_lim + padding)
        curr_ax.set_ylim(min_lim - padding, max_lim + padding)

    for ild_idx, abs_ild in enumerate(ABS_ILDS):
        if np.isfinite(x[ild_idx]) and np.isfinite(y[ild_idx]):
            curr_ax.annotate(
                f"{abs_ild}",
                (x[ild_idx], y[ild_idx]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )

    curr_ax.set_title(f"ABL {ABL}: slope ratio vs omega ratio")
    curr_ax.set_xlabel("Quantile scaling ratio (1 + slope)")
    curr_ax.set_ylabel(r"Condition-fit $\omega_{60} / \omega_{\mathrm{ABL}}$")
    curr_ax.grid(True, alpha=0.25)

fig.suptitle("Quantile scaling compared with condition-fit omega ratio")
fig.tight_layout()
fig.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
print(f"\nSaved figure: {FIG_PATH}")


# %% Plot animal-wise quantile slope ratio vs animal-wise omega ratio
animalwise_rows = []

for animal_idx, (batch_name, animal_id) in enumerate(batch_animal_pairs):
    for ABL in ABLS_TO_COMPARE:
        for ild_idx, abs_ild in enumerate(ABS_ILDS):
            raw_slope = slope_summary["raw_slopes"][ABL][animal_idx, ild_idx]
            slope_ratio = 1 + raw_slope if np.isfinite(raw_slope) else np.nan
            omega_ratio = animal_omega_ratio_values[ABL][animal_idx, ild_idx]

            animalwise_rows.append(
                {
                    "batch_name": batch_name,
                    "animal": animal_id,
                    "ABL": ABL,
                    "baseline_ABL": BASELINE_ABL,
                    "abs_ILD": abs_ild,
                    "raw_slope": raw_slope,
                    "slope_ratio_1_plus_slope": slope_ratio,
                    "omega_abs_baseline_ABL_60": omega_abs_summary["omega_abs_values"][BASELINE_ABL][animal_idx, ild_idx],
                    "omega_abs_current_ABL": omega_abs_summary["omega_abs_values"][ABL][animal_idx, ild_idx],
                    "omega_ratio_60_over_ABL": omega_ratio,
                    "n_trials_current_ABL": slope_summary["n_trials_by_abl"][ABL][animal_idx, ild_idx],
                    "n_trials_baseline_ABL_60": slope_summary["n_trials_by_abl"][BASELINE_ABL][animal_idx, ild_idx],
                }
            )

animalwise_df = pd.DataFrame(animalwise_rows)
animalwise_df.to_csv(ANIMALWISE_CSV_PATH, index=False)
print(f"Saved animal-wise data: {ANIMALWISE_CSV_PATH}")

fig_animal, ax_animal = plt.subplots(
    len(ABLS_TO_COMPARE),
    len(ABS_ILDS),
    figsize=(18, 7),
    sharex=False,
    sharey=False,
)

for row_idx, ABL in enumerate(ABLS_TO_COMPARE):
    for col_idx, abs_ild in enumerate(ABS_ILDS):
        curr_ax = ax_animal[row_idx, col_idx]
        color = COLORS[ABL]

        x = 1 + slope_summary["raw_slopes"][ABL][:, col_idx]
        y = animal_omega_ratio_values[ABL][:, col_idx]
        finite = np.isfinite(x) & np.isfinite(y)

        curr_ax.scatter(
            x[finite],
            y[finite],
            color=color,
            alpha=0.75,
            s=28,
            edgecolor="none",
        )

        if np.any(finite):
            min_lim = np.nanmin([np.nanmin(x[finite]), np.nanmin(y[finite])])
            max_lim = np.nanmax([np.nanmax(x[finite]), np.nanmax(y[finite])])
            padding = 0.08 * (max_lim - min_lim)
            curr_ax.plot(
                [min_lim - padding, max_lim + padding],
                [min_lim - padding, max_lim + padding],
                color="black",
                linestyle="--",
                linewidth=1,
                alpha=0.5,
            )
            curr_ax.set_xlim(min_lim - padding, max_lim + padding)
            curr_ax.set_ylim(min_lim - padding, max_lim + padding)

        if row_idx == 0:
            curr_ax.set_title(f"|ILD|={abs_ild}")
        if col_idx == 0:
            curr_ax.set_ylabel(f"ABL {ABL}\n" + r"$\omega_{60} / \omega_{\mathrm{ABL}}$")
        if row_idx == len(ABLS_TO_COMPARE) - 1:
            curr_ax.set_xlabel("1 + quantile slope")

        curr_ax.text(
            0.05,
            0.95,
            f"n={np.sum(finite)}",
            transform=curr_ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
        )
        curr_ax.grid(True, alpha=0.25)

fig_animal.suptitle("Animal-wise quantile scaling vs condition-fit omega ratio")
fig_animal.tight_layout()
fig_animal.savefig(ANIMALWISE_FIG_PATH, dpi=300, bbox_inches="tight")
print(f"Saved animal-wise figure: {ANIMALWISE_FIG_PATH}")


# %% Save plotted arrays
plot_data = {
    "ABLS": ABLS,
    "ABLS_TO_COMPARE": ABLS_TO_COMPARE,
    "BASELINE_ABL": BASELINE_ABL,
    "ABS_ILDS": ABS_ILDS,
    "SIGNED_ILDS": SIGNED_ILDS,
    "FITTING_QUANTILES": FITTING_QUANTILES,
    "MIN_RT_CUT_BY_ILD": MIN_RT_CUT_BY_ILD,
    "MAX_RT_CUT": MAX_RT_CUT,
    "N_POSTERIOR_SAMPLES": N_POSTERIOR_SAMPLES,
    "batch_animal_pairs": batch_animal_pairs,
    "missing_files": missing_files,
    "slope_summary": slope_summary,
    "omega_abs_summary": omega_abs_summary,
    "omega_ratio_summary": omega_ratio_summary,
    "animal_omega_ratio_values": animal_omega_ratio_values,
    "animalwise_csv_path": ANIMALWISE_CSV_PATH,
}

with open(PKL_PATH, "wb") as f:
    pickle.dump(plot_data, f)
print(f"Saved data: {PKL_PATH}")

plt.show()

# %%
