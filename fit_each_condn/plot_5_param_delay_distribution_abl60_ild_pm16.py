# %%
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gamma_omega_alpha_utils import load_batch_animal_pairs, print_batch_animal_table


# %% Parameters
SCRIPT_DIR = os.path.dirname(__file__)
REPO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DESIRED_BATCHES = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]
BATCH_DIR = os.path.join(REPO_DIR, "fit_animal_by_animal", "batch_csvs")
PKL_DIR = os.path.join(SCRIPT_DIR, "each_animal_cond_fit_5_params_pkl_files")

ABL_TO_LOAD = 60
ILDS_TO_LOAD = [-16, 16]
N_POSTERIOR_SAMPLES = int(1e5)
DELAY_BINS_MS = np.arange(30, 90, 2.5)
REFERENCE_DELAY_MS = 72.5

PARAM_NAMES = ["gamma", "omega", "t_E_aff", "w", "del_go"]
DELAY_COLUMNS_MS = ["t_E_aff_ms"]
DELAY_LABELS = {
    "t_E_aff_ms": "t_E_aff (ms)",
}
COLORS = {
    "pooled": "black",
}

ANIMALWISE_CSV_PATH = os.path.join(
    SCRIPT_DIR,
    "five_param_delay_distribution_ABL60_ILD_pm16_animalwise.csv",
)
FIG_PATH = os.path.join(
    SCRIPT_DIR,
    "five_param_delay_distribution_ABL60_ILD_pm16.png",
)


# %% Load VP posterior means for ABL 60, ILD +/-16
batch_animal_pairs = load_batch_animal_pairs(BATCH_DIR, DESIRED_BATCHES)
print_batch_animal_table(batch_animal_pairs)
print(f"Loading 5-param condition fits from: {PKL_DIR}")
print(f"Using {N_POSTERIOR_SAMPLES} VP posterior samples per condition.")

rows = []
missing_files = []

for batch_name, animal_id in batch_animal_pairs:
    for ILD in ILDS_TO_LOAD:
        pkl_file = os.path.join(
            PKL_DIR,
            f"vbmc_cond_by_cond_{batch_name}_{animal_id}_{ABL_TO_LOAD}_ILD_{ILD}_5_params.pkl",
        )
        if not os.path.exists(pkl_file):
            missing_files.append(pkl_file)
            continue

        with open(pkl_file, "rb") as f:
            vbmc = pickle.load(f)
        samples = vbmc.vp.sample(N_POSTERIOR_SAMPLES)[0]
        if samples.shape[1] < len(PARAM_NAMES):
            raise ValueError(
                f"{pkl_file} has {samples.shape[1]} sampled params, expected at least {len(PARAM_NAMES)}"
            )

        posterior_means = {name: float(np.mean(samples[:, idx])) for idx, name in enumerate(PARAM_NAMES)}
        rows.append(
            {
                "batch_name": batch_name,
                "animal": animal_id,
                "ABL": ABL_TO_LOAD,
                "ILD": ILD,
                "gamma": posterior_means["gamma"],
                "omega": posterior_means["omega"],
                "t_E_aff_s": posterior_means["t_E_aff"],
                "w": posterior_means["w"],
                "del_go_s": posterior_means["del_go"],
                "t_E_aff_ms": 1e3 * posterior_means["t_E_aff"],
                "del_go_ms": 1e3 * posterior_means["del_go"],
                "total_delay_ms": 1e3 * (posterior_means["t_E_aff"] + posterior_means["del_go"]),
            }
        )

animalwise_df = pd.DataFrame(rows)
animalwise_df.to_csv(ANIMALWISE_CSV_PATH, index=False)
print(f"Loaded {len(animalwise_df)} condition files for {animalwise_df['animal'].nunique()} animals.")
if len(missing_files) > 0:
    print(f"Missing ABL={ABL_TO_LOAD}, ILD=+/-16 5-param pickle files: {len(missing_files)}")
print(f"Saved animal-wise posterior means: {ANIMALWISE_CSV_PATH}")


# %% Club ILD +/-16 together and summarize
summary_rows = []
for delay_col in DELAY_COLUMNS_MS:
    values = animalwise_df[delay_col].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    summary_rows.append(
        {
            "delay": delay_col,
            "group": "ILD +/-16 pooled",
            "n": len(values),
            "mean_ms": float(np.mean(values)) if len(values) else np.nan,
            "median_ms": float(np.median(values)) if len(values) else np.nan,
            "std_ms": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
        }
    )

summary_df = pd.DataFrame(summary_rows)
print("\nDelay posterior-mean summaries:")
print(summary_df.to_string(index=False))


# %% Plot across-animal distributions
fig, ax = plt.subplots(figsize=(6.5, 4.5))

for delay_col in DELAY_COLUMNS_MS:
    curr_ax = ax
    all_values = animalwise_df[delay_col].to_numpy(dtype=float)
    all_values = all_values[np.isfinite(all_values)]

    if len(all_values) == 0:
        curr_ax.set_visible(False)
        continue

    curr_ax.hist(
        all_values,
        bins=DELAY_BINS_MS,
        density=True,
        histtype="step",
        linewidth=2.0,
        color=COLORS["pooled"],
        label=f"ILD +/-16 pooled (n={len(all_values)})",
    )
    curr_ax.axvline(
        REFERENCE_DELAY_MS,
        color="crimson",
        linestyle="--",
        linewidth=1.8,
        label=f"{REFERENCE_DELAY_MS:g} ms",
    )
    curr_ax.set_title(DELAY_LABELS[delay_col])
    curr_ax.set_xlabel("Posterior mean delay (ms)")
    curr_ax.grid(True, alpha=0.25)

ax.set_ylabel("Density")
ax.legend(fontsize=8)
fig.suptitle(f"5-param VP posterior mean t_E_aff, ABL={ABL_TO_LOAD}, ILD=+/-16")
fig.tight_layout()
fig.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
print(f"Saved figure: {FIG_PATH}")

plt.show()

# %%
