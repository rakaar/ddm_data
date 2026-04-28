# %%
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from matplotlib.ticker import FormatStrFormatter

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
N_BINS = 50
POSTERIOR_INTERVAL = [2.5, 97.5]
ELLIPSE_QUANTILE = 0.95

FIG_PATH = os.path.join(
    SCRIPT_DIR,
    f"five_param_ABL{ABL_TO_PLOT}_ILD{ILD_TO_PLOT}_omega_t_E_aff_w_corner_ellipses_all_animals.png",
)


# %% Load posterior samples
batch_animal_pairs = load_batch_animal_pairs(BATCH_DIR, DESIRED_BATCHES)
print_batch_animal_table(batch_animal_pairs)

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
    w_samples = samples[:, 3]
    omega_ci = np.percentile(omega_samples, POSTERIOR_INTERVAL)
    t_E_aff_ms_ci = np.percentile(t_E_aff_ms_samples, POSTERIOR_INTERVAL)
    w_ci = np.percentile(w_samples, POSTERIOR_INTERVAL)

    animal_posteriors.append(
        {
            "label": f"{batch_name}/{animal_id}",
            "omega": omega_samples,
            "t_E_aff_ms": t_E_aff_ms_samples,
            "w": w_samples,
            "omega_mean": float(np.mean(omega_samples)),
            "t_E_aff_ms_mean": float(np.mean(t_E_aff_ms_samples)),
            "w_mean": float(np.mean(w_samples)),
            "omega_ci": omega_ci,
            "t_E_aff_ms_ci": t_E_aff_ms_ci,
            "w_ci": w_ci,
        }
    )

if len(missing_files) > 0:
    print(f"Missing ABL={ABL_TO_PLOT}, ILD={ILD_TO_PLOT} 5-param pickle files: {len(missing_files)}")
print(f"Loaded {len(animal_posteriors)} animals for ABL={ABL_TO_PLOT}, ILD={ILD_TO_PLOT}")


# %% Compute shared limits
param_specs = [
    {"key": "omega", "label": "omega", "mean_key": "omega_mean", "ci_key": "omega_ci"},
    {"key": "t_E_aff_ms", "label": "t_E_aff (ms)", "mean_key": "t_E_aff_ms_mean", "ci_key": "t_E_aff_ms_ci"},
    {"key": "w", "label": "w", "mean_key": "w_mean", "ci_key": "w_ci"},
]

lims_by_param = {}
for spec in param_specs:
    all_values_for_lims = np.concatenate(
        [animal[spec["key"]] for animal in animal_posteriors]
    ) if len(animal_posteriors) > 0 else np.array([])
    curr_lims = np.percentile(all_values_for_lims, [0.5, 99.5])
    curr_pad = 0.05 * (curr_lims[1] - curr_lims[0])
    if curr_pad == 0 or not np.isfinite(curr_pad):
        curr_pad = 0.05
    lims_by_param[spec["key"]] = [curr_lims[0] - curr_pad, curr_lims[1] + curr_pad]


# %% Plot one corner-style summary with covariance ellipses
n_params = len(param_specs)
fig, ax = plt.subplots(n_params, n_params, figsize=(9, 9))

for row_idx, y_spec in enumerate(param_specs):
    for col_idx, x_spec in enumerate(param_specs):
        curr_ax = ax[row_idx, col_idx]

        if row_idx < col_idx:
            curr_ax.axis("off")
            continue

        if row_idx > col_idx:
            s_chi2 = -2.0 * np.log(1.0 - ELLIPSE_QUANTILE)
            for animal in animal_posteriors:
                xy_samples = np.vstack([animal[x_spec["key"]], animal[y_spec["key"]]])
                cov = np.cov(xy_samples)
                if not np.all(np.isfinite(cov)):
                    continue

                try:
                    evals, evecs = np.linalg.eigh(cov)
                except np.linalg.LinAlgError:
                    continue

                order = np.argsort(evals)[::-1]
                evals = np.maximum(evals[order], 0.0)
                evecs = evecs[:, order]
                width = 2.0 * float(np.sqrt(s_chi2 * evals[0]))
                height = 2.0 * float(np.sqrt(s_chi2 * evals[1]))
                if width == 0.0 or height == 0.0:
                    continue

                angle = float(np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0])))
                ellipse = Ellipse(
                    (animal[x_spec["mean_key"]], animal[y_spec["mean_key"]]),
                    width=width,
                    height=height,
                    angle=angle,
                    facecolor="none",
                    edgecolor="#2b6cb0",
                    linewidth=1.0,
                    alpha=0.65,
                    zorder=3,
                )
                curr_ax.add_patch(ellipse)

            curr_ax.scatter(
                [animal[x_spec["mean_key"]] for animal in animal_posteriors],
                [animal[y_spec["mean_key"]] for animal in animal_posteriors],
                s=28,
                c="#8B0000",
                edgecolor="black",
                linewidths=0.5,
                zorder=5,
            )
            curr_ax.set_xlim(lims_by_param[x_spec["key"]])
            curr_ax.set_ylim(lims_by_param[y_spec["key"]])
            curr_ax.grid(True, alpha=0.2)
            curr_ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            curr_ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            if row_idx == n_params - 1:
                curr_ax.set_xlabel(x_spec["label"])
            else:
                curr_ax.set_xticklabels([])
            if col_idx == 0:
                curr_ax.set_ylabel(y_spec["label"])
            else:
                curr_ax.set_yticklabels([])
            continue

        param_name = x_spec["label"]
        lims = lims_by_param[x_spec["key"]]
        mean_key = x_spec["mean_key"]
        ci_key = x_spec["ci_key"]
        stats = []
        for animal in animal_posteriors:
            mean_value = animal[mean_key]
            ci = animal[ci_key]
            stats.append((animal["label"], mean_value, ci[0], ci[1]))
        stats.sort(key=lambda row: row[1], reverse=True)

        for rank_idx, (_, mean_value, lo_value, hi_value) in enumerate(stats):
            curr_ax.errorbar(
                mean_value,
                rank_idx,
                xerr=[[mean_value - lo_value], [hi_value - mean_value]],
                fmt="o",
                color="#8B0000",
                ecolor="#8B0000",
                markeredgecolor="black",
                markersize=4,
                elinewidth=1.2,
                linewidth=1.0,
                alpha=0.95,
            )

        curr_ax.set_xlim(lims)
        curr_ax.set_ylim(-0.5, len(stats) - 0.5)
        curr_ax.set_yticks([])
        curr_ax.set_xlabel(param_name)
        curr_ax.grid(True, alpha=0.2)
        curr_ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))

fig.suptitle(
    f"ABL={ABL_TO_PLOT}, ILD={ILD_TO_PLOT}: 95% covariance ellipses and posterior means",
    fontsize=13,
)
fig.tight_layout()
fig.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
print(f"Saved figure: {FIG_PATH}")

plt.show()

# %%
