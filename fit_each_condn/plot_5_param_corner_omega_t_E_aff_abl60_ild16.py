# %%
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from gamma_omega_alpha_utils import load_batch_animal_pairs, print_batch_animal_table


# %% Parameters
SCRIPT_DIR = os.path.dirname(__file__)
REPO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DESIRED_BATCHES = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]
BATCH_DIR = os.path.join(REPO_DIR, "fit_animal_by_animal", "batch_csvs")
PKL_DIR = os.path.join(SCRIPT_DIR, "each_animal_cond_fit_5_params_pkl_files")

ABL_TO_PLOT = 60
ILD_TO_PLOT = 4
N_POSTERIOR_SAMPLES = int(1e5)
N_SCATTER_SAMPLES = 5000
N_BINS = 50
N_GRID_ROWS = 6
N_GRID_COLS = 4
RNG_SEED = 12345

FIG_PATH = os.path.join(
    SCRIPT_DIR,
    "five_param_ABL60_ILD16_omega_t_E_aff_corner_all_animals.png",
)


# %% Load posterior samples
batch_animal_pairs = load_batch_animal_pairs(BATCH_DIR, DESIRED_BATCHES)
print_batch_animal_table(batch_animal_pairs)

rng = np.random.default_rng(RNG_SEED)
animal_posteriors = []
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

    omega_samples = samples[:, 1]
    t_E_aff_ms_samples = 1e3 * samples[:, 2]
    omega_ci = np.percentile(omega_samples, [2.5, 97.5])
    t_E_aff_ci = np.percentile(t_E_aff_ms_samples, [2.5, 97.5])

    n_scatter = min(N_SCATTER_SAMPLES, len(omega_samples))
    scatter_idx = rng.choice(len(omega_samples), size=n_scatter, replace=False)

    animal_posteriors.append(
        {
            "label": f"{batch_name}/{animal_id}",
            "omega": omega_samples,
            "t_E_aff_ms": t_E_aff_ms_samples,
            "omega_ci": omega_ci,
            "t_E_aff_ms_ci": t_E_aff_ci,
            "scatter_omega": omega_samples[scatter_idx],
            "scatter_t_E_aff_ms": t_E_aff_ms_samples[scatter_idx],
        }
    )

if len(missing_files) > 0:
    print(f"Missing ABL={ABL_TO_PLOT}, ILD={ILD_TO_PLOT} 5-param pickle files: {len(missing_files)}")
print(f"Loaded {len(animal_posteriors)} animals for ABL={ABL_TO_PLOT}, ILD={ILD_TO_PLOT}")


# %% Plot mini-corners for all loaded animals
fig = plt.figure(figsize=(16, 22))
outer_grid = fig.add_gridspec(
    N_GRID_ROWS,
    N_GRID_COLS,
    wspace=0.32,
    hspace=0.42,
)

for animal_idx, animal_data in enumerate(animal_posteriors[: N_GRID_ROWS * N_GRID_COLS]):
    row_idx = animal_idx // N_GRID_COLS
    col_idx = animal_idx % N_GRID_COLS
    inner_grid = outer_grid[row_idx, col_idx].subgridspec(
        2,
        2,
        wspace=0.08,
        hspace=0.08,
    )

    ax_omega = fig.add_subplot(inner_grid[0, 0])
    ax_blank = fig.add_subplot(inner_grid[0, 1])
    ax_scatter = fig.add_subplot(inner_grid[1, 0])
    ax_t_E_aff = fig.add_subplot(inner_grid[1, 1])

    ax_omega.hist(
        animal_data["omega"],
        bins=N_BINS,
        density=True,
        histtype="step",
        color="tab:red",
        linewidth=1.1,
    )
    ax_scatter.scatter(
        animal_data["scatter_omega"],
        animal_data["scatter_t_E_aff_ms"],
        s=1,
        alpha=0.12,
        color="0.15",
        rasterized=True,
    )
    ax_t_E_aff.hist(
        animal_data["t_E_aff_ms"],
        bins=N_BINS,
        density=True,
        histtype="step",
        color="tab:blue",
        linewidth=1.1,
    )
    ax_omega.axvline(animal_data["omega_ci"][0], color="tab:red", linestyle="--", linewidth=0.8)
    ax_omega.axvline(animal_data["omega_ci"][1], color="tab:red", linestyle="--", linewidth=0.8)
    ax_t_E_aff.axvline(animal_data["t_E_aff_ms_ci"][0], color="tab:blue", linestyle="--", linewidth=0.8)
    ax_t_E_aff.axvline(animal_data["t_E_aff_ms_ci"][1], color="tab:blue", linestyle="--", linewidth=0.8)
    ci_rect = Rectangle(
        (
            animal_data["omega_ci"][0],
            animal_data["t_E_aff_ms_ci"][0],
        ),
        animal_data["omega_ci"][1] - animal_data["omega_ci"][0],
        animal_data["t_E_aff_ms_ci"][1] - animal_data["t_E_aff_ms_ci"][0],
        facecolor="tab:purple",
        edgecolor="tab:purple",
        alpha=0.12,
        linewidth=0.8,
        linestyle="--",
    )
    ax_scatter.add_patch(ci_rect)
    ax_scatter.axvline(animal_data["omega_ci"][0], color="tab:red", linestyle="--", linewidth=0.6, alpha=0.7)
    ax_scatter.axvline(animal_data["omega_ci"][1], color="tab:red", linestyle="--", linewidth=0.6, alpha=0.7)
    ax_scatter.axhline(animal_data["t_E_aff_ms_ci"][0], color="tab:blue", linestyle="--", linewidth=0.6, alpha=0.7)
    ax_scatter.axhline(animal_data["t_E_aff_ms_ci"][1], color="tab:blue", linestyle="--", linewidth=0.6, alpha=0.7)

    ax_blank.set_visible(False)

    ax_omega.set_title(animal_data["label"], fontsize=8)
    ax_scatter.set_xlabel("omega", fontsize=7)
    ax_scatter.set_ylabel("t_E_aff (ms)", fontsize=7)
    ax_t_E_aff.set_xlabel("t_E_aff (ms)", fontsize=7)

    for curr_ax in [ax_omega, ax_scatter, ax_t_E_aff]:
        curr_ax.grid(True, alpha=0.2)
        curr_ax.tick_params(labelsize=6)

    ax_omega.tick_params(labelbottom=False)
    ax_t_E_aff.tick_params(labelleft=False)

for empty_idx in range(len(animal_posteriors), N_GRID_ROWS * N_GRID_COLS):
    row_idx = empty_idx // N_GRID_COLS
    col_idx = empty_idx % N_GRID_COLS
    empty_ax = fig.add_subplot(outer_grid[row_idx, col_idx])
    empty_ax.set_visible(False)

fig.suptitle(
    f"5-param mini-corner posteriors: ABL={ABL_TO_PLOT}, ILD={ILD_TO_PLOT}, "
    f"omega vs t_E_aff",
    fontsize=16,
)
fig.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
print(f"Saved figure: {FIG_PATH}")

plt.show()

# %%
