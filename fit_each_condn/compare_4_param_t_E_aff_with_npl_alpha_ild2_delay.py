# %%
"""
Compare 4-param condition-fit t_E_aff with NPL+alpha+ILD2 delay curves.

Condition-fit t_E_aff is estimated separately for each ABL/ILD. The ILD2 model
uses one animal-wise delay function:
delay_ms = bias + abl_coeff*ABL + abs_ild_coeff*|ILD| + ild2_coeff*|ILD|^2.
"""
import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from gamma_omega_alpha_utils import (
    get_param_means_by_ABL_ILD,
    load_batch_animal_pairs,
    print_batch_animal_table,
)


# %% Parameters
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DESIRED_BATCHES = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]
BATCH_DIR = os.path.join(REPO_DIR, "fit_animal_by_animal", "batch_csvs")

COND_FIT_SOURCE = "gamma_omega_t_E_aff_del_go_fix_w_mean"
COND_FIT_PKL_DIR = os.path.join(SCRIPT_DIR, "each_animal_cond_fit_4_params_fix_w_mean_pkl_files")
COND_FIT_FILENAME_SUFFIX = "_FIX_w_mean_4_params"
COND_FIT_EXPECTED_N_PARAMS = 4
COND_PARAM_NAMES = ["gamma", "omega", "t_E_aff", "del_go"]

MODEL_RESULTS_DIR = os.path.join(REPO_DIR, "NPL_alpha_ILD2_fit_results", "result_pkls")
MODEL_RESULT_KEY = "vbmc_norm_alpha_ild2_delay_tied_results"
MODEL_LABEL = "NPL + alpha + ILD2 delay"
MODEL_RESULT_PATTERN = re.compile(
    r"^results_(?P<batch>.+)_animal_(?P<animal>\d+)_NORM_ALPHA_ILD2_DELAY_FROM_ABORTS\.pkl$"
)
DELAY_SAMPLE_KEYS = {
    "bias_ms": "bias_ms_samples",
    "abl_coeff": "abl_delay_coeff_ms_per_abl_samples",
    "abs_ild_coeff": "abs_ild_delay_coeff_ms_per_unit_samples",
    "ild2_coeff": "ild2_delay_coeff_ms_per_unit2_samples",
}

OUTPUT_DIR = os.path.join(REPO_DIR, "NPL_alpha_ILD2_fit_results", "delay_comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ABLS = [20, 40, 60]
ILDS = np.sort([-16, -8, -4, -2, -1, 1, 2, 4, 8, 16])
CONTINUOUS_ILDS = np.round(np.arange(-16, 16 + 0.05, 0.1), 10)

N_POSTERIOR_SAMPLES = int(1e5)
COLORS = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}

FIG_PATH = os.path.join(OUTPUT_DIR, "cond_t_E_aff_vs_npl_alpha_ild2_delay.png")
COND_SUMMARY_CSV_PATH = os.path.join(
    OUTPUT_DIR, "cond_t_E_aff_vs_npl_alpha_ild2_delay_condition_summary.csv"
)
MODEL_SUMMARY_CSV_PATH = os.path.join(
    OUTPUT_DIR, "cond_t_E_aff_vs_npl_alpha_ild2_delay_model_summary.csv"
)


# %% Helpers
def parse_model_result_filename(fname):
    match = MODEL_RESULT_PATTERN.match(fname)
    if match is None:
        return None
    return match.group("batch"), int(match.group("animal"))


def load_model_result_paths():
    result_paths = {}
    for fname in os.listdir(MODEL_RESULTS_DIR):
        parsed = parse_model_result_filename(fname)
        if parsed is None:
            continue
        result_paths[parsed] = os.path.join(MODEL_RESULTS_DIR, fname)
    return result_paths


def posterior_mean_delay_coefficients(pkl_path):
    with open(pkl_path, "rb") as f:
        saved_data = pickle.load(f)

    if MODEL_RESULT_KEY not in saved_data:
        raise KeyError(f"{pkl_path} is missing `{MODEL_RESULT_KEY}`")

    model_results = saved_data[MODEL_RESULT_KEY]
    missing_keys = [
        sample_key
        for sample_key in DELAY_SAMPLE_KEYS.values()
        if sample_key not in model_results
    ]
    if missing_keys:
        raise KeyError(f"{pkl_path} is missing delay coefficient sample keys: {missing_keys}")

    return {
        coeff_name: float(np.mean(np.asarray(model_results[sample_key], dtype=float)))
        for coeff_name, sample_key in DELAY_SAMPLE_KEYS.items()
    }


def compute_delay_ms(abl, ild, coeffs):
    abs_ild = np.abs(ild)
    return (
        coeffs["bias_ms"]
        + coeffs["abl_coeff"] * abl
        + coeffs["abs_ild_coeff"] * abs_ild
        + coeffs["ild2_coeff"] * (abs_ild ** 2)
    )


def mean_sem_n(arr, axis=0):
    arr = np.asarray(arr, dtype=float)
    n = np.sum(np.isfinite(arr), axis=axis)
    mean = np.full(arr.shape[1], np.nan)
    sem = np.full(arr.shape[1], np.nan)
    finite_condition = n > 0
    mean[finite_condition] = np.nanmean(arr[:, finite_condition], axis=0)
    sem[finite_condition] = np.nanstd(arr[:, finite_condition], axis=0) / np.sqrt(
        n[finite_condition]
    )
    return mean, sem, n


# %%
# Load matched animals.
print(f"Using condition-fit source: {COND_FIT_SOURCE}")
print(f"Condition-fit pickle directory: {COND_FIT_PKL_DIR}")
print(f"Model result directory: {MODEL_RESULTS_DIR}")

all_batch_animal_pairs = load_batch_animal_pairs(BATCH_DIR, DESIRED_BATCHES)
model_result_paths = load_model_result_paths()
matched_pairs = [
    (batch_name, int(animal_id))
    for batch_name, animal_id in all_batch_animal_pairs
    if (batch_name, int(animal_id)) in model_result_paths
]

print_batch_animal_table(matched_pairs)
print(f"Matched animals with {MODEL_LABEL} fits: {len(matched_pairs)}")
if len(matched_pairs) == 0:
    raise RuntimeError(f"No matched animals have model result pickles in {MODEL_RESULTS_DIR}")


# %%
# Load condition-fit t_E_aff posterior means.
condition_t_E_aff_by_abl = {
    str(ABL): np.full((len(matched_pairs), len(ILDS)), np.nan) for ABL in ABLS
}
missing_condition_files = []

for animal_idx, (batch_name, animal_id) in enumerate(matched_pairs):
    print("##########################################")
    print(f"Condition fits: Batch {batch_name}, Animal {animal_id}")
    print("##########################################")

    param_dict, missing_for_animal = get_param_means_by_ABL_ILD(
        batch_name,
        animal_id,
        ABLS,
        ILDS,
        COND_FIT_PKL_DIR,
        n_samples=N_POSTERIOR_SAMPLES,
        param_names=COND_PARAM_NAMES,
        filename_suffix=COND_FIT_FILENAME_SUFFIX,
        expected_n_params=COND_FIT_EXPECTED_N_PARAMS,
    )
    missing_condition_files.extend(missing_for_animal)

    for ABL in ABLS:
        for ild_idx, ILD in enumerate(ILDS):
            if (ABL, ILD) in param_dict:
                condition_t_E_aff_by_abl[str(ABL)][animal_idx, ild_idx] = (
                    1e3 * param_dict[(ABL, ILD)]["t_E_aff"]
                )

print(f"Missing condition-fit pickle files: {len(missing_condition_files)}")


# %%
# Average condition-fit delays across matched animals.
condition_summary_by_abl = {}
condition_summary_rows = []

for ABL in ABLS:
    mean_value, sem_value, n = mean_sem_n(condition_t_E_aff_by_abl[str(ABL)], axis=0)
    condition_summary_by_abl[str(ABL)] = {
        "mean": mean_value,
        "sem": sem_value,
        "n": n,
    }
    for ild_idx, ILD in enumerate(ILDS):
        condition_summary_rows.append(
            {
                "ABL": ABL,
                "ILD": ILD,
                "condition_t_E_aff_mean_ms": mean_value[ild_idx],
                "condition_t_E_aff_sem_ms": sem_value[ild_idx],
                "n_animals": n[ild_idx],
            }
        )

if not any(np.any(np.isfinite(condition_summary_by_abl[str(ABL)]["mean"])) for ABL in ABLS):
    raise RuntimeError("No finite condition-fit t_E_aff values were loaded.")

condition_summary_df = pd.DataFrame(condition_summary_rows)
condition_summary_df.to_csv(COND_SUMMARY_CSV_PATH, index=False)
print(f"Saved condition-fit summary: {COND_SUMMARY_CSV_PATH}")


# %%
# Load model delay functions and average continuous curves across animals.
model_delay_by_abl = {
    str(ABL): np.full((len(matched_pairs), len(CONTINUOUS_ILDS)), np.nan) for ABL in ABLS
}
coeff_rows = []

for animal_idx, (batch_name, animal_id) in enumerate(matched_pairs):
    pkl_path = model_result_paths[(batch_name, animal_id)]
    coeffs = posterior_mean_delay_coefficients(pkl_path)
    coeff_rows.append({"batch_name": batch_name, "animal": animal_id, **coeffs})

    for ABL in ABLS:
        model_delay_by_abl[str(ABL)][animal_idx, :] = compute_delay_ms(
            ABL,
            CONTINUOUS_ILDS,
            coeffs,
        )

model_summary_by_abl = {}
model_summary_rows = []

for ABL in ABLS:
    mean_value, sem_value, n = mean_sem_n(model_delay_by_abl[str(ABL)], axis=0)
    model_summary_by_abl[str(ABL)] = {
        "mean": mean_value,
        "sem": sem_value,
        "n": n,
    }
    for ild_idx, ILD in enumerate(CONTINUOUS_ILDS):
        model_summary_rows.append(
            {
                "ABL": ABL,
                "ILD": ILD,
                "model_delay_mean_ms": mean_value[ild_idx],
                "model_delay_sem_ms": sem_value[ild_idx],
                "n_animals": n[ild_idx],
            }
        )

model_summary_df = pd.DataFrame(model_summary_rows)
model_summary_df.to_csv(MODEL_SUMMARY_CSV_PATH, index=False)
print(f"Saved model delay summary: {MODEL_SUMMARY_CSV_PATH}")

coeff_df = pd.DataFrame(coeff_rows)
print("Delay coefficient means across matched animals:")
for coeff_name in ["bias_ms", "abl_coeff", "abs_ild_coeff", "ild2_coeff"]:
    values = coeff_df[coeff_name].to_numpy(dtype=float)
    print(
        f"  {coeff_name}: {np.nanmean(values):.6g} "
        f"+/- {np.nanstd(values) / np.sqrt(np.sum(np.isfinite(values))):.6g} SEM"
    )


# %%
# Plot condition-fit points and model delay curves.
fig, ax = plt.subplots(figsize=(7.5, 5))

for ABL in ABLS:
    color = COLORS[ABL]
    cond_summary = condition_summary_by_abl[str(ABL)]
    model_summary = model_summary_by_abl[str(ABL)]

    ax.fill_between(
        CONTINUOUS_ILDS,
        model_summary["mean"] - model_summary["sem"],
        model_summary["mean"] + model_summary["sem"],
        color=color,
        alpha=0.12,
        linewidth=0,
    )
    ax.plot(
        CONTINUOUS_ILDS,
        model_summary["mean"],
        color=color,
        linewidth=2,
    )
    ax.errorbar(
        ILDS,
        cond_summary["mean"],
        yerr=cond_summary["sem"],
        marker="o",
        linestyle="none",
        capsize=3,
        color=color,
        zorder=3,
    )

ax.set_xlabel("ILD")
ax.set_ylabel("t_E_aff / delay (ms)")
ax.set_xlim(-16.2, 16.2)
ax.set_xticks([-16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16])
ax.grid(True, alpha=0.25)

abl_handles = [
    Line2D([0], [0], color=COLORS[ABL], linewidth=2, label=f"ABL={ABL}")
    for ABL in ABLS
]
source_handles = [
    Line2D([0], [0], marker="o", color="black", linestyle="none", label="condition fit mean +/- SEM"),
    Line2D([0], [0], color="black", linewidth=2, label=f"{MODEL_LABEL} mean +/- SEM"),
]
ax.legend(handles=abl_handles + source_handles, fontsize=8)
ax.set_title("Condition-fit t_E_aff vs NPL+alpha+ILD2 delay")

fig.tight_layout()
fig.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
print(f"Saved figure: {FIG_PATH}")

plt.show()

# %%
