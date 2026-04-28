# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gamma_omega_alpha_utils import (
    get_param_means_by_ABL_ILD,
    load_batch_animal_pairs,
    print_batch_animal_table,
)


# %% Parameters
SCRIPT_DIR = os.path.dirname(__file__)
REPO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DESIRED_BATCHES = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]
BATCH_DIR = os.path.join(REPO_DIR, "fit_animal_by_animal", "batch_csvs")
COND_FIT_PKL_DIR = os.path.join(SCRIPT_DIR, "each_animal_cond_fit_5_params_pkl_files")
COND_FIT_FILENAME_SUFFIX = "_5_params"
COND_FIT_EXPECTED_N_PARAMS = 5

ABLS = [20, 40, 60]
ILDS = np.sort([-16, -8, -4, -2, -1, 1, 2, 4, 8, 16])

PARAM_NAMES = ["gamma", "omega", "t_E_aff", "w", "del_go"]
PLOT_PARAM_NAMES = ["t_E_aff", "del_go", "w"]
PLOT_LABELS = {
    "t_E_aff": "delta_e (ms)",
    "del_go": "del_go (ms)",
    "w": "Starting point, w",
}

N_POSTERIOR_SAMPLES = int(1e5)
COLORS = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}

FIG_PATH = os.path.join(SCRIPT_DIR, "five_param_delay_by_abl_ild.png")
ANIMALWISE_CSV_PATH = os.path.join(SCRIPT_DIR, "five_param_delay_by_abl_ild_animalwise.csv")
SUMMARY_CSV_PATH = os.path.join(SCRIPT_DIR, "five_param_delay_by_abl_ild_summary.csv")


# %% Load 5-param condition fits
batch_animal_pairs = load_batch_animal_pairs(BATCH_DIR, DESIRED_BATCHES)
print_batch_animal_table(batch_animal_pairs)
print(f"Loading 5-param condition fits from: {COND_FIT_PKL_DIR}")
print(f"Using {N_POSTERIOR_SAMPLES} posterior samples per condition.")

plot_values = {
    name: {str(ABL): np.full((len(batch_animal_pairs), len(ILDS)), np.nan) for ABL in ABLS}
    for name in PLOT_PARAM_NAMES
}
missing_files = []
animalwise_rows = []

for animal_idx, (batch_name, animal_id) in enumerate(batch_animal_pairs):
    print("##########################################")
    print(f"Batch: {batch_name}, Animal: {animal_id}")
    print("##########################################")

    param_dict, missing_for_animal = get_param_means_by_ABL_ILD(
        batch_name,
        animal_id,
        ABLS,
        ILDS,
        COND_FIT_PKL_DIR,
        n_samples=N_POSTERIOR_SAMPLES,
        param_names=PARAM_NAMES,
        filename_suffix=COND_FIT_FILENAME_SUFFIX,
        expected_n_params=COND_FIT_EXPECTED_N_PARAMS,
    )
    missing_files.extend(missing_for_animal)

    for ABL in ABLS:
        for ild_idx, ILD in enumerate(ILDS):
            row = {
                "batch_name": batch_name,
                "animal": animal_id,
                "ABL": ABL,
                "ILD": ILD,
            }
            if (ABL, ILD) in param_dict:
                for param_name in PARAM_NAMES:
                    row[param_name] = param_dict[(ABL, ILD)][param_name]
                for plot_name in PLOT_PARAM_NAMES:
                    plot_values[plot_name][str(ABL)][animal_idx, ild_idx] = param_dict[(ABL, ILD)][plot_name]
            else:
                for param_name in PARAM_NAMES:
                    row[param_name] = np.nan
            animalwise_rows.append(row)

if len(missing_files) > 0:
    print(f"Missing 5-param condition-fit pickle files: {len(missing_files)}")


# %% Average fitted values across animals
summary_rows = []
plot_summary = {}

for plot_name in PLOT_PARAM_NAMES:
    plot_summary[plot_name] = {}
    for ABL in ABLS:
        arr = plot_values[plot_name][str(ABL)]
        if plot_name in ["t_E_aff", "del_go"]:
            arr = 1e3 * arr
        n = np.sum(np.isfinite(arr), axis=0)
        mean_value = np.nanmean(arr, axis=0)
        sem_value = np.nanstd(arr, axis=0) / np.sqrt(n)

        plot_summary[plot_name][str(ABL)] = {
            "mean": mean_value,
            "sem": sem_value,
            "n": n,
        }

        for ild_idx, ILD in enumerate(ILDS):
            summary_rows.append(
                {
                    "param": plot_name,
                    "ABL": ABL,
                    "ILD": ILD,
                    "mean": mean_value[ild_idx],
                    "sem": sem_value[ild_idx],
                    "n_animals": n[ild_idx],
                }
            )

animalwise_df = pd.DataFrame(animalwise_rows)
animalwise_df.to_csv(ANIMALWISE_CSV_PATH, index=False)
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_CSV_PATH, index=False)
print(f"Saved animal-wise delay data: {ANIMALWISE_CSV_PATH}")
print(f"Saved delay summary data: {SUMMARY_CSV_PATH}")


# %% Plot fitted values by ILD, colored by ABL
fig, ax = plt.subplots(1, len(PLOT_PARAM_NAMES), figsize=(16, 5), sharex=True)

for plot_idx, plot_name in enumerate(PLOT_PARAM_NAMES):
    curr_ax = ax[plot_idx]
    for ABL in ABLS:
        summary = plot_summary[plot_name][str(ABL)]
        curr_ax.errorbar(
            ILDS,
            summary["mean"],
            yerr=summary["sem"],
            marker="o",
            linestyle="none",
            capsize=3,
            color=COLORS[ABL],
            label=f"ABL={ABL}",
        )

    curr_ax.set_title(PLOT_LABELS[plot_name])
    curr_ax.set_xlabel("ILD")
    curr_ax.set_ylabel("ms" if plot_name in ["t_E_aff", "del_go"] else "w")
    curr_ax.grid(True, alpha=0.25)
    curr_ax.legend(fontsize=8)

fig.suptitle("5-param condition fits averaged across animals")
fig.tight_layout()
fig.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
print(f"Saved figure: {FIG_PATH}")

plt.show()

# %%
