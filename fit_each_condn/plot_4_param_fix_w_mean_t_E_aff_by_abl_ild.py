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

COND_FIT_SOURCE = "gamma_omega_t_E_aff_del_go_fix_w_mean"
COND_FIT_PKL_DIR = os.path.join(SCRIPT_DIR, "each_animal_cond_fit_4_params_fix_w_mean_pkl_files")
COND_FIT_FILENAME_SUFFIX = "_FIX_w_mean_4_params"
COND_FIT_EXPECTED_N_PARAMS = 4

ABLS = [20, 40, 60]
ILDS = np.sort([-16, -8, -4, -2, -1, 1, 2, 4, 8, 16])

PARAM_NAMES = ["gamma", "omega", "t_E_aff", "del_go"]
N_POSTERIOR_SAMPLES = int(1e5)
COLORS = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}

FIG_PATH = os.path.join(SCRIPT_DIR, "four_param_fix_w_mean_t_E_aff_by_abl_ild.png")
ANIMALWISE_CSV_PATH = os.path.join(
    SCRIPT_DIR, "four_param_fix_w_mean_t_E_aff_by_abl_ild_animalwise.csv"
)
SUMMARY_CSV_PATH = os.path.join(SCRIPT_DIR, "four_param_fix_w_mean_t_E_aff_by_abl_ild_summary.csv")


# %% Load 4-param fixed animal-mean-w condition fits
batch_animal_pairs = load_batch_animal_pairs(BATCH_DIR, DESIRED_BATCHES)
print_batch_animal_table(batch_animal_pairs)
print(f"Using condition-fit source: {COND_FIT_SOURCE}")
print(f"Loading condition fits from: {COND_FIT_PKL_DIR}")
print(f"Filename suffix: {COND_FIT_FILENAME_SUFFIX}")
print(f"Using {N_POSTERIOR_SAMPLES} posterior samples per condition.")

t_E_aff_by_abl = {
    str(ABL): np.full((len(batch_animal_pairs), len(ILDS)), np.nan) for ABL in ABLS
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
                row["t_E_aff_ms"] = 1e3 * param_dict[(ABL, ILD)]["t_E_aff"]
                row["del_go_ms"] = 1e3 * param_dict[(ABL, ILD)]["del_go"]
                t_E_aff_by_abl[str(ABL)][animal_idx, ild_idx] = row["t_E_aff_ms"]
            else:
                for param_name in PARAM_NAMES:
                    row[param_name] = np.nan
                row["t_E_aff_ms"] = np.nan
                row["del_go_ms"] = np.nan
            animalwise_rows.append(row)

print(f"Missing condition-fit pickle files: {len(missing_files)}")


# %% Average t_E_aff across animals
summary_by_abl = {}
summary_rows = []
total_finite_values = 0

for ABL in ABLS:
    arr = t_E_aff_by_abl[str(ABL)]
    n = np.sum(np.isfinite(arr), axis=0)
    total_finite_values += int(np.sum(n))

    mean_value = np.full(len(ILDS), np.nan)
    sem_value = np.full(len(ILDS), np.nan)
    finite_condition = n > 0
    mean_value[finite_condition] = np.nanmean(arr[:, finite_condition], axis=0)
    sem_value[finite_condition] = np.nanstd(arr[:, finite_condition], axis=0) / np.sqrt(
        n[finite_condition]
    )

    summary_by_abl[str(ABL)] = {
        "mean": mean_value,
        "sem": sem_value,
        "n": n,
    }

    for ild_idx, ILD in enumerate(ILDS):
        summary_rows.append(
            {
                "ABL": ABL,
                "ILD": ILD,
                "t_E_aff_mean_ms": mean_value[ild_idx],
                "t_E_aff_sem_ms": sem_value[ild_idx],
                "n_animals": n[ild_idx],
            }
        )

if total_finite_values == 0:
    raise RuntimeError("No finite t_E_aff values were loaded from the condition-fit pickles.")

animalwise_df = pd.DataFrame(animalwise_rows)
animalwise_df.to_csv(ANIMALWISE_CSV_PATH, index=False)
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_CSV_PATH, index=False)
print(f"Saved animal-wise t_E_aff data: {ANIMALWISE_CSV_PATH}")
print(f"Saved t_E_aff summary data: {SUMMARY_CSV_PATH}")


# %% Plot t_E_aff by ILD, colored by ABL
fig, ax = plt.subplots(figsize=(6.5, 4.5))

for ABL in ABLS:
    summary = summary_by_abl[str(ABL)]
    ax.errorbar(
        ILDS,
        summary["mean"],
        yerr=summary["sem"],
        marker="o",
        linestyle="none",
        capsize=3,
        color=COLORS[ABL],
        label=f"ABL={ABL}",
    )

ax.set_xlabel("ILD")
ax.set_ylabel("t_E_aff (ms)")
ax.set_title("4-param condition fits with fixed animal-mean w")
ax.set_xticks(ILDS)
ax.grid(True, alpha=0.25)
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
print(f"Saved figure: {FIG_PATH}")

plt.show()

# %%
