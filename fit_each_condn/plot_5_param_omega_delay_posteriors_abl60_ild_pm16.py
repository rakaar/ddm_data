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
ILDS_TO_PLOT = [16, -16]
N_POSTERIOR_SAMPLES = int(1e5)
N_BINS = 50

FIG_PATH = os.path.join(
    SCRIPT_DIR,
    "five_param_ABL60_omega_t_E_aff_posteriors_ILD_pm16.png",
)
AVERAGE_FIG_PATH = os.path.join(
    SCRIPT_DIR,
    "five_param_ABL60_omega_t_E_aff_average_posteriors_ILD_pm16.png",
)


# %% Load posterior samples
batch_animal_pairs = load_batch_animal_pairs(BATCH_DIR, DESIRED_BATCHES)
print_batch_animal_table(batch_animal_pairs)

posterior_samples = {
    ILD: {
        "omega": [],
        "t_E_aff_ms": [],
        "labels": [],
    }
    for ILD in ILDS_TO_PLOT
}
missing_files = []

for batch_name, animal_id in batch_animal_pairs:
    for ILD in ILDS_TO_PLOT:
        pkl_file = os.path.join(
            PKL_DIR,
            f"vbmc_cond_by_cond_{batch_name}_{animal_id}_{ABL_TO_PLOT}_ILD_{ILD}_5_params.pkl",
        )
        if not os.path.exists(pkl_file):
            missing_files.append(pkl_file)
            continue

        with open(pkl_file, "rb") as f:
            vbmc = pickle.load(f)
        samples = vbmc.vp.sample(N_POSTERIOR_SAMPLES)[0]

        posterior_samples[ILD]["omega"].append(samples[:, 1])
        posterior_samples[ILD]["t_E_aff_ms"].append(1e3 * samples[:, 2])
        posterior_samples[ILD]["labels"].append(f"{batch_name}/{animal_id}")

if len(missing_files) > 0:
    print(f"Missing ABL={ABL_TO_PLOT}, ILD=+/-16 5-param pickle files: {len(missing_files)}")

for ILD in ILDS_TO_PLOT:
    print(
        f"ABL={ABL_TO_PLOT}, ILD={ILD}: "
        f"{len(posterior_samples[ILD]['omega'])} animals loaded"
    )


# %% Plot omega and t_E_aff posteriors
fig, ax = plt.subplots(2, 2, figsize=(10, 7), sharey="row")

for col_idx, ILD in enumerate(ILDS_TO_PLOT):
    omega_samples_all_animals = posterior_samples[ILD]["omega"]
    t_E_aff_samples_all_animals = posterior_samples[ILD]["t_E_aff_ms"]

    for omega_samples in omega_samples_all_animals:
        ax[0, col_idx].hist(
            omega_samples,
            bins=N_BINS,
            density=True,
            histtype="step",
            alpha=0.35,
            linewidth=1,
            color="tab:red",
        )

    for t_E_aff_samples in t_E_aff_samples_all_animals:
        ax[1, col_idx].hist(
            t_E_aff_samples,
            bins=N_BINS,
            density=True,
            histtype="step",
            alpha=0.35,
            linewidth=1,
            color="tab:blue",
        )

    ax[0, col_idx].set_title(
        f"ABL={ABL_TO_PLOT}, ILD={ILD}: omega (n={len(omega_samples_all_animals)})"
    )
    ax[1, col_idx].set_title(
        f"ABL={ABL_TO_PLOT}, ILD={ILD}: t_E_aff (n={len(t_E_aff_samples_all_animals)})"
    )
    ax[1, col_idx].set_xlabel("t_E_aff (ms)")

ax[0, 0].set_ylabel("Density")
ax[1, 0].set_ylabel("Density")
ax[0, 0].set_xlabel("omega")
ax[0, 1].set_xlabel("omega")

for curr_ax in ax.ravel():
    curr_ax.grid(True, alpha=0.25)

fig.suptitle("5-param posterior distributions for ABL=60, ILD=+/-16")
fig.tight_layout()
fig.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
print(f"Saved figure: {FIG_PATH}")

plt.show()

# %%
# %% Plot animal-averaged omega and t_E_aff posterior densities
fig_avg, ax_avg = plt.subplots(2, 2, figsize=(10, 7), sharey="row")

for col_idx, ILD in enumerate(ILDS_TO_PLOT):
    omega_samples_all_animals = posterior_samples[ILD]["omega"]
    t_E_aff_samples_all_animals = posterior_samples[ILD]["t_E_aff_ms"]

    for row_idx, (param_name, samples_all_animals, color, xlabel) in enumerate(
        [
            ("omega", omega_samples_all_animals, "tab:red", "omega"),
            ("t_E_aff", t_E_aff_samples_all_animals, "tab:blue", "t_E_aff (ms)"),
        ]
    ):
        curr_ax = ax_avg[row_idx, col_idx]

        if len(samples_all_animals) == 0:
            curr_ax.set_visible(False)
            continue

        combined_samples = np.concatenate(samples_all_animals)
        bins = np.histogram_bin_edges(combined_samples, bins=N_BINS)
        density_by_animal = []
        for samples in samples_all_animals:
            density, _ = np.histogram(samples, bins=bins, density=True)
            density_by_animal.append(density)

        density_by_animal = np.asarray(density_by_animal)
        mean_density = np.nanmean(density_by_animal, axis=0)
        sem_density = np.nanstd(density_by_animal, axis=0) / np.sqrt(density_by_animal.shape[0])

        curr_ax.stairs(
            mean_density,
            bins,
            color=color,
            linewidth=2,
            label="mean density",
        )
        curr_ax.fill_between(
            bins[:-1],
            mean_density - sem_density,
            mean_density + sem_density,
            step="post",
            color=color,
            alpha=0.2,
            linewidth=0,
            label="SEM",
        )
        curr_ax.set_title(
            f"ABL={ABL_TO_PLOT}, ILD={ILD}: {param_name} average (n={len(samples_all_animals)})"
        )
        curr_ax.set_xlabel(xlabel)
        curr_ax.grid(True, alpha=0.25)

ax_avg[0, 0].set_ylabel("Mean density")
ax_avg[1, 0].set_ylabel("Mean density")

fig_avg.suptitle("Animal-averaged 5-param posterior densities for ABL=60, ILD=+/-16")
fig_avg.tight_layout()
fig_avg.savefig(AVERAGE_FIG_PATH, dpi=300, bbox_inches="tight")
print(f"Saved figure: {AVERAGE_FIG_PATH}")

plt.show()
