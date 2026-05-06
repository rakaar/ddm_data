# %%
# Compare animal-wise NPL norm parameters against NPL + alpha parameters.

import os
import pickle
import re

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


# %%
# Editable parameters

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_RESULTS_DIR = SCRIPT_DIR
ALPHA_RESULTS_DIR = os.path.join(SCRIPT_DIR, "NPL_alpha_animal_fits")
OUTPUT_PDF = os.path.join(ALPHA_RESULTS_DIR, "compare_npl_vs_npl_alpha_params.pdf")

BASELINE_RESULT_KEY = "vbmc_norm_tied_results"
ALPHA_RESULT_KEY = "vbmc_norm_alpha_tied_results"

DESIRED_BATCHES = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]
BATCH_ORDER = {batch: idx for idx, batch in enumerate(DESIRED_BATCHES)}

PARAM_CONFIGS = [
    ("rate_lambda_samples", "rate_lambda", r"$\lambda$", 1.0),
    ("T_0_samples", "T_0", r"$T_0$ (ms)", 1e3),
    ("theta_E_samples", "theta_E", r"$\theta_E$", 1.0),
    ("w_samples", "w", r"$w$", 1.0),
    ("t_E_aff_samples", "t_E_aff", r"$t_E^{aff}$ (ms)", 1e3),
    ("del_go_samples", "del_go", r"$\Delta_{go}$", 1.0),
    ("rate_norm_l_samples", "rate_norm_l", "rate_norm_l", 1.0),
    ("alpha_samples", "alpha", r"$\alpha$", 1.0),
]

BASELINE_COLOR = "tab:blue"
ALPHA_COLOR = "tab:red"
X_OFFSET = 0.16


# %%
# Discover matching baseline NPL and NPL + alpha result pickle files.

baseline_pattern = re.compile(r"^results_(?P<batch>.+)_animal_(?P<animal>\d+)\.pkl$")
alpha_pattern = re.compile(
    r"^results_(?P<batch>.+)_animal_(?P<animal>\d+)_NORM_ALPHA_FROM_ABORTS\.pkl$"
)

baseline_paths = {}
for fname in os.listdir(BASELINE_RESULTS_DIR):
    match = baseline_pattern.match(fname)
    if match is None:
        continue
    batch = match.group("batch")
    animal = int(match.group("animal"))
    baseline_paths[(batch, animal)] = os.path.join(BASELINE_RESULTS_DIR, fname)

alpha_paths = {}
for fname in os.listdir(ALPHA_RESULTS_DIR):
    match = alpha_pattern.match(fname)
    if match is None:
        continue
    batch = match.group("batch")
    animal = int(match.group("animal"))
    alpha_paths[(batch, animal)] = os.path.join(ALPHA_RESULTS_DIR, fname)

matched_pairs = sorted(
    set(baseline_paths) & set(alpha_paths),
    key=lambda pair: (BATCH_ORDER.get(pair[0], len(BATCH_ORDER)), pair[1]),
)

if len(matched_pairs) == 0:
    raise RuntimeError(
        "No matched baseline NPL and NPL + alpha result pickle files were found."
    )

print(f"Found {len(baseline_paths)} baseline NPL result pickle files.")
print(f"Found {len(alpha_paths)} NPL + alpha result pickle files.")
print(f"Using {len(matched_pairs)} matched batch-animal pairs.")

missing_baseline = sorted(set(alpha_paths) - set(baseline_paths))
if missing_baseline:
    print("NPL + alpha files without matching baseline NPL files:")
    for batch, animal in missing_baseline:
        print(f"  {batch}-{animal}")

missing_alpha = sorted(set(baseline_paths) - set(alpha_paths))
if missing_alpha:
    print("Baseline NPL files without matching NPL + alpha files:")
    for batch, animal in missing_alpha:
        print(f"  {batch}-{animal}")


# %%
# Load posterior sample summaries.

def sample_summary(results, result_key, sample_key, scale, pkl_path):
    if result_key not in results:
        raise KeyError(f"{pkl_path} is missing `{result_key}`")
    model_results = results[result_key]
    if sample_key not in model_results:
        raise KeyError(f"{pkl_path} is missing `{result_key}[{sample_key}]`")

    samples = np.asarray(model_results[sample_key], dtype=float) * scale
    if samples.size == 0:
        raise ValueError(f"{pkl_path} has empty samples for `{result_key}[{sample_key}]`")

    q025, q975 = np.percentile(samples, [2.5, 97.5])
    mean = np.mean(samples)
    return mean, q025, q975


rows = []
for batch, animal in matched_pairs:
    baseline_path = baseline_paths[(batch, animal)]
    alpha_path = alpha_paths[(batch, animal)]

    with open(baseline_path, "rb") as f:
        baseline_results = pickle.load(f)
    with open(alpha_path, "rb") as f:
        alpha_results = pickle.load(f)

    row = {
        "batch": batch,
        "animal": animal,
        "label": f"{batch}-{animal}",
        "baseline": {},
        "alpha": {},
    }

    for sample_key, param_name, _, scale in PARAM_CONFIGS:
        if param_name != "alpha":
            row["baseline"][param_name] = sample_summary(
                baseline_results,
                BASELINE_RESULT_KEY,
                sample_key,
                scale,
                baseline_path,
            )

        row["alpha"][param_name] = sample_summary(
            alpha_results,
            ALPHA_RESULT_KEY,
            sample_key,
            scale,
            alpha_path,
        )

    rows.append(row)

print("Loaded posterior means and 2.5/97.5 percentile intervals for all matched pairs.")


# %%
# Plot one PDF page per parameter.

labels = [row["label"] for row in rows]
x = np.arange(len(labels))

with PdfPages(OUTPUT_PDF) as pdf:
    for _, param_name, param_label, _ in PARAM_CONFIGS:
        fig_width = max(12, 0.35 * len(labels))
        fig, ax = plt.subplots(figsize=(fig_width, 6))

        if param_name != "alpha":
            baseline_mean = np.array([row["baseline"][param_name][0] for row in rows])
            baseline_low = np.array([row["baseline"][param_name][1] for row in rows])
            baseline_high = np.array([row["baseline"][param_name][2] for row in rows])
            baseline_yerr = np.vstack([
                baseline_mean - baseline_low,
                baseline_high - baseline_mean,
            ])

            ax.errorbar(
                x - X_OFFSET,
                baseline_mean,
                yerr=baseline_yerr,
                fmt="o",
                color=BASELINE_COLOR,
                ecolor=BASELINE_COLOR,
                elinewidth=1.3,
                capsize=3,
                markersize=4,
                label="NPL",
            )

        alpha_mean = np.array([row["alpha"][param_name][0] for row in rows])
        alpha_low = np.array([row["alpha"][param_name][1] for row in rows])
        alpha_high = np.array([row["alpha"][param_name][2] for row in rows])
        alpha_yerr = np.vstack([
            alpha_mean - alpha_low,
            alpha_high - alpha_mean,
        ])

        alpha_x = x if param_name == "alpha" else x + X_OFFSET
        ax.errorbar(
            alpha_x,
            alpha_mean,
            yerr=alpha_yerr,
            fmt="o",
            color=ALPHA_COLOR,
            ecolor=ALPHA_COLOR,
            elinewidth=1.3,
            capsize=3,
            markersize=4,
            label="NPL + alpha",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=55, ha="right")
        ax.set_xlabel("Animal")
        ax.set_ylabel(param_label)

        if param_name == "alpha":
            ax.set_title(f"{param_label}: NPL + alpha only")
        else:
            ax.set_title(f"{param_label}: NPL vs NPL + alpha")

        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

print(f"Saved comparison PDF: {OUTPUT_PDF}")
