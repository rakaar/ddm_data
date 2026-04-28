# %%
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from gamma_omega_alpha_utils import load_batch_animal_pairs, print_batch_animal_table


# %% Parameters
SCRIPT_DIR = os.path.dirname(__file__)
REPO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DESIRED_BATCHES = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]
BATCH_DIR = os.path.join(REPO_DIR, "fit_animal_by_animal", "batch_csvs")
PKL_DIR = os.path.join(SCRIPT_DIR, "each_animal_cond_fit_5_params_pkl_files")

ABL_TO_PLOT = 60
ILD_TO_PLOT = 16
N_POSTERIOR_SAMPLES = int(1e5)

FIG_PATH = os.path.join(
    SCRIPT_DIR,
    f"five_param_ABL{ABL_TO_PLOT}_ILD{ILD_TO_PLOT}_omega_t_E_aff_animal_lines.png",
)


# %% Load posterior means for each animal
batch_animal_pairs = load_batch_animal_pairs(BATCH_DIR, DESIRED_BATCHES)
print_batch_animal_table(batch_animal_pairs)

animal_rows = []
missing_files = []

for batch_name, animal_id in batch_animal_pairs:
    pkl_file = os.path.join(
        PKL_DIR,
        f"vbmc_cond_by_cond_{batch_name}_{animal_id}_{ABL_TO_PLOT}_ILD_{ILD_TO_PLOT}_5_params.pkl",
    )
    if not os.path.exists(pkl_file):
        missing_files.append(pkl_file)
        continue

    with open(pkl_file, "rb") as f:
        vbmc = pickle.load(f)
    samples = vbmc.vp.sample(N_POSTERIOR_SAMPLES)[0]

    animal_rows.append(
        {
            "label": f"{batch_name}/{animal_id}",
            "omega": float(np.mean(samples[:, 1])),
            "t_E_aff_ms": float(1e3 * np.mean(samples[:, 2])),
            "w": float(np.mean(samples[:, 3])),
        }
    )

if len(missing_files) > 0:
    print(f"Missing ABL={ABL_TO_PLOT}, ILD={ILD_TO_PLOT} 5-param pickle files: {len(missing_files)}")
print(f"Loaded {len(animal_rows)} animals for ABL={ABL_TO_PLOT}, ILD={ILD_TO_PLOT}")


# %% Plot animal-wise posterior-mean parameter values
fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
x = np.arange(len(animal_rows))
labels = [row["label"] for row in animal_rows]

plot_specs = [
    ("omega", "Omega", "tab:red", ax[0]),
    ("t_E_aff_ms", "t_E_aff (ms)", "tab:blue", ax[1]),
    ("w", "w", "tab:green", ax[2]),
]

for key, ylabel, color, curr_ax in plot_specs:
    values = np.asarray([row[key] for row in animal_rows], dtype=float)
    mean_value = float(np.nanmean(values))
    median_value = float(np.nanmedian(values))

    curr_ax.scatter(
        x,
        values,
        marker="o",
        color=color,
        edgecolors="black",
        s=50,
    )
    curr_ax.axhline(
        mean_value,
        color="black",
        linestyle="-",
        linewidth=1.2,
        label=f"mean={mean_value:.4g}",
    )
    curr_ax.axhline(
        median_value,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"median={median_value:.4g}",
    )
    curr_ax.set_title(f"{ylabel}: mean={mean_value:.4g}, median={median_value:.4g}")
    curr_ax.set_ylabel(ylabel)
    curr_ax.grid(True, alpha=0.25)
    curr_ax.legend(fontsize=8)

ax[-1].set_xticks(x)
ax[-1].set_xticklabels(labels, rotation=90, fontsize=7)
ax[-1].set_xlabel("Animal")

fig.suptitle(f"Posterior-mean parameters per animal: ABL={ABL_TO_PLOT}, ILD={ILD_TO_PLOT}")
fig.tight_layout()
fig.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
print(f"Saved figure: {FIG_PATH}")

plt.show()

# %%
