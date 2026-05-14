# %%
import copy
import os
import pickle
from collections import defaultdict

os.makedirs("/tmp/matplotlib", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import sem

import figure_template as ft

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]


# %%
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

DESIRED_BATCHES = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]
QUANT_PKL = os.path.join(SCRIPT_DIR, "norm_quant_fig2_data.pkl")
OUTPUT_DIR = os.path.join(REPO_DIR, "NPL_alpha_ILD2_fit_results", "figure_4_diagnostics")
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "fig4_quantile_sd_ild_mismatch_check.png")
OUTPUT_PDF = os.path.join(OUTPUT_DIR, "fig4_quantile_sd_ild_mismatch_check.pdf")


# %%
def _create_innermost_dict():
    return {"empirical": [], "theoretical": []}


def _create_inner_defaultdict():
    return defaultdict(_create_innermost_dict)


def load_quantile_data():
    with open(QUANT_PKL, "rb") as handle:
        return pickle.load(handle)


def get_figure4_batch_animal_pairs():
    batch_dir = os.path.join(SCRIPT_DIR, "batch_csvs")
    batch_files = [os.path.join(batch_dir, f"batch_{batch_name}_valid_and_aborts.csv") for batch_name in DESIRED_BATCHES]

    merged_data = pd.concat(
        [pd.read_csv(path) for path in batch_files if os.path.exists(path)],
        ignore_index=True,
    )
    merged_valid = merged_data[merged_data["success"].isin([1, -1])].copy()
    batch_animal_pairs = sorted(list(map(tuple, merged_valid[["batch_name", "animal"]].drop_duplicates().values)))
    return [(batch_name, int(animal)) for batch_name, animal in batch_animal_pairs]


# %%
def plot_quantiles(ax, data):
    plot_data = data["plot_data"]
    continuous_plot_data = data.get("continuous_plot_data", None)
    continuous_abs_ild = data.get("continuous_abs_ild", None)
    QUANTILES_TO_PLOT = data["QUANTILES_TO_PLOT"]
    abs_ild_sorted = data["abs_ild_sorted"]
    ABL_arr = data["ABL_arr"]

    for q_idx, q in enumerate(QUANTILES_TO_PLOT):
        emp_means, emp_sems = [], []
        theo_means, theo_sems = [], []
        theo_abs_ild_plot = []

        for abs_ild in abs_ild_sorted:
            all_abl_emp_quantiles = np.concatenate(
                [np.array(plot_data[abl][abs_ild]["empirical"])[:, q_idx] for abl in ABL_arr]
            )
            emp_means.append(np.nanmean(all_abl_emp_quantiles))
            emp_sems.append(sem(all_abl_emp_quantiles, nan_policy="omit"))

        if continuous_plot_data is not None and continuous_abs_ild is not None:
            for abs_ild in continuous_abs_ild:
                all_abl_theo_q = []
                for abl in ABL_arr:
                    if len(continuous_plot_data[abl][abs_ild]["theoretical"]) > 0:
                        all_abl_theo_q.extend(np.array(continuous_plot_data[abl][abs_ild]["theoretical"])[:, q_idx])
                if len(all_abl_theo_q) > 0:
                    theo_abs_ild_plot.append(abs_ild)
                    theo_means.append(np.nanmean(all_abl_theo_q))
                    theo_sems.append(sem(all_abl_theo_q, nan_policy="omit"))
        else:
            for abs_ild in abs_ild_sorted:
                all_abl_theo_quantiles = np.concatenate(
                    [np.array(plot_data[abl][abs_ild]["theoretical"])[:, q_idx] for abl in ABL_arr]
                )
                theo_abs_ild_plot.append(abs_ild)
                theo_means.append(np.nanmean(all_abl_theo_quantiles))
                theo_sems.append(sem(all_abl_theo_quantiles, nan_policy="omit"))

        ax.errorbar(
            abs_ild_sorted,
            emp_means,
            yerr=emp_sems,
            fmt="o",
            color="black",
            markersize=8,
            capsize=0,
            label=f"Data q={q:.2f}" if q_idx == 0 else "_nolegend_",
        )

        if len(theo_abs_ild_plot) > 0:
            ax.plot(
                theo_abs_ild_plot,
                theo_means,
                "-",
                color="tab:red",
                linewidth=1.5,
                label=f"Theory q={q:.2f}" if q_idx == 0 else "_nolegend_",
            )
            ax.fill_between(
                theo_abs_ild_plot,
                np.array(theo_means) - np.array(theo_sems),
                np.array(theo_means) + np.array(theo_sems),
                color="tab:red",
                alpha=0.2,
                linewidth=0,
            )

    ax.set_xlabel("|ILD| (dB)", fontsize=ft.STYLE.LABEL_FONTSIZE)
    ax.set_ylabel("RT Quantile (s)", fontsize=ft.STYLE.LABEL_FONTSIZE)
    ax.set_xscale("log", base=2)
    ax.set_xticks(abs_ild_sorted)
    ax.set_yticks([0.1, 0.2, 0.3, 0.4])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.tick_params(axis="both", which="major", labelsize=ft.STYLE.TICK_FONTSIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.set_box_aspect(1)


# %%
def filter_sd_theory_above_ild8(data, sd_animal_indices, n_animals):
    filtered_data = copy.deepcopy(data)
    continuous_plot_data = filtered_data["continuous_plot_data"]

    for abl in filtered_data["ABL_arr"]:
        for abs_ild in filtered_data["continuous_abs_ild"]:
            if float(abs_ild) <= 8.0:
                continue

            theory_entries = list(continuous_plot_data[abl][abs_ild]["theoretical"])
            if len(theory_entries) != 2 * n_animals:
                print(
                    f"Warning: ABL={abl}, |ILD|={abs_ild} has {len(theory_entries)} theory entries; "
                    f"expected {2 * n_animals}. Leaving this condition unchanged."
                )
                continue

            kept_entries = []
            for entry_idx, theory_quantiles in enumerate(theory_entries):
                animal_idx = entry_idx // 2
                if animal_idx not in sd_animal_indices:
                    kept_entries.append(theory_quantiles)

            continuous_plot_data[abl][abs_ild]["theoretical"] = kept_entries

    return filtered_data


def print_sanity_counts(label, data):
    print(f"\n{label}")
    for abl in data["ABL_arr"]:
        left_or_right_theory_n = len(data["continuous_plot_data"][abl][16.0]["theoretical"])
        empirical_quantiles = np.asarray(data["plot_data"][abl][16.0]["empirical"], dtype=float)
        empirical_finite_n = int(np.sum(np.any(np.isfinite(empirical_quantiles), axis=1)))
        print(
            f"  ABL={abl}: theory signed entries at |ILD|=16 = {left_or_right_theory_n}; "
            f"empirical finite animal entries at |ILD|=16 = {empirical_finite_n}"
        )


# %%
quantile_data = load_quantile_data()
batch_animal_pairs = get_figure4_batch_animal_pairs()
sd_animal_indices = [idx for idx, (batch_name, _animal) in enumerate(batch_animal_pairs) if batch_name == "SD"]

print(f"Figure 4 batch-animal pairs ({len(batch_animal_pairs)}):")
for idx, pair in enumerate(batch_animal_pairs):
    print(f"  {idx:02d}: {pair[0]} {pair[1]}")
print(f"SD animal indices: {sd_animal_indices}")

matched_sd_data = filter_sd_theory_above_ild8(
    quantile_data,
    sd_animal_indices=sd_animal_indices,
    n_animals=len(batch_animal_pairs),
)

print_sanity_counts("Paper data as saved", quantile_data)
print_sanity_counts("Matched SD grid data", matched_sd_data)


# %%
os.makedirs(OUTPUT_DIR, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharex=True, sharey=True)
plot_quantiles(axes[0], quantile_data)
axes[0].set_title("All 30 animals: paper model includes SD to |ILD|=16", fontsize=12)

plot_quantiles(axes[1], matched_sd_data)
axes[1].set_title("All 30 animals: SD model included only to |ILD|=8", fontsize=12)
axes[1].set_ylabel("")

for ax in axes:
    ax.set_ylim(0.06, 0.5)

fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.84, wspace=0.28)
fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUTPUT_PDF, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved {OUTPUT_PNG}")
print(f"Saved {OUTPUT_PDF}")

# %%
