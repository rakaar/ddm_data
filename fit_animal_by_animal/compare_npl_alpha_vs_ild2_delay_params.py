# %%
# Compare animal-wise NPL + alpha parameters against NPL + alpha + ILD2-delay parameters.

import os
import pickle
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


# %%
# Editable parameters

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

ALPHA_RESULTS_DIR = os.path.join(SCRIPT_DIR, "NPL_alpha_animal_fits")
ILD2_RESULTS_DIR = os.path.join(REPO_DIR, "NPL_alpha_ILD2_fit_results", "result_pkls")
OUTPUT_DIR = os.path.join(REPO_DIR, "NPL_alpha_ILD2_fit_results", "param_comparison")
OUTPUT_PDF = os.path.join(OUTPUT_DIR, "compare_npl_alpha_vs_ild2_delay_params.pdf")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALPHA_RESULT_KEY = "vbmc_norm_alpha_tied_results"
ILD2_RESULT_KEY = "vbmc_norm_alpha_ild2_delay_tied_results"

DESIRED_BATCHES = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]
BATCH_ORDER = {batch: idx for idx, batch in enumerate(DESIRED_BATCHES)}

PARAM_CONFIGS = [
    ("rate_lambda_samples", "rate_lambda", r"$\lambda$", 1.0),
    ("T_0_samples", "T_0", r"$T_0$ (ms)", 1e3),
    ("theta_E_samples", "theta_E", r"$\theta_E$", 1.0),
    ("w_samples", "w", r"$w$", 1.0),
    ("del_go_samples", "del_go", r"$\Delta_{go}$", 1.0),
    ("rate_norm_l_samples", "rate_norm_l", "rate_norm_l", 1.0),
    ("alpha_samples", "alpha", r"$\alpha$", 1.0),
]

ALPHA_COLOR = "tab:blue"
ILD2_COLOR = "tab:red"
X_OFFSET = 0.16


# %%
# Discover matching NPL + alpha and NPL + alpha + ILD2-delay result pickle files.

alpha_pattern = re.compile(
    r"^results_(?P<batch>.+)_animal_(?P<animal>\d+)_NORM_ALPHA_FROM_ABORTS\.pkl$"
)
ild2_pattern = re.compile(
    r"^results_(?P<batch>.+)_animal_(?P<animal>\d+)_NORM_ALPHA_ILD2_DELAY_FROM_ABORTS\.pkl$"
)

alpha_paths = {}
for fname in os.listdir(ALPHA_RESULTS_DIR):
    match = alpha_pattern.match(fname)
    if match is None:
        continue
    batch = match.group("batch")
    animal = int(match.group("animal"))
    alpha_paths[(batch, animal)] = os.path.join(ALPHA_RESULTS_DIR, fname)

ild2_paths = {}
for fname in os.listdir(ILD2_RESULTS_DIR):
    match = ild2_pattern.match(fname)
    if match is None:
        continue
    batch = match.group("batch")
    animal = int(match.group("animal"))
    ild2_paths[(batch, animal)] = os.path.join(ILD2_RESULTS_DIR, fname)

matched_pairs = sorted(
    set(alpha_paths) & set(ild2_paths),
    key=lambda pair: (BATCH_ORDER.get(pair[0], len(BATCH_ORDER)), pair[1]),
)

if len(matched_pairs) == 0:
    raise RuntimeError(
        "No matched NPL + alpha and NPL + alpha + ILD2-delay result pickle files were found."
    )

print(f"Found {len(alpha_paths)} NPL + alpha result pickle files.")
print(f"Found {len(ild2_paths)} NPL + alpha + ILD2-delay result pickle files.")
print(f"Using {len(matched_pairs)} matched batch-animal pairs.")

missing_alpha = sorted(set(ild2_paths) - set(alpha_paths))
if missing_alpha:
    print("ILD2-delay files without matching NPL + alpha files:")
    for batch, animal in missing_alpha:
        print(f"  {batch}-{animal}")

missing_ild2 = sorted(set(alpha_paths) - set(ild2_paths))
if missing_ild2:
    print("NPL + alpha files without matching ILD2-delay files:")
    for batch, animal in missing_ild2:
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
    alpha_path = alpha_paths[(batch, animal)]
    ild2_path = ild2_paths[(batch, animal)]

    with open(alpha_path, "rb") as f:
        alpha_results = pickle.load(f)
    with open(ild2_path, "rb") as f:
        ild2_results = pickle.load(f)

    row = {
        "batch": batch,
        "animal": animal,
        "label": f"{batch}-{animal}",
        "alpha": {},
        "ild2": {},
    }

    for sample_key, param_name, _, scale in PARAM_CONFIGS:
        row["alpha"][param_name] = sample_summary(
            alpha_results,
            ALPHA_RESULT_KEY,
            sample_key,
            scale,
            alpha_path,
        )
        row["ild2"][param_name] = sample_summary(
            ild2_results,
            ILD2_RESULT_KEY,
            sample_key,
            scale,
            ild2_path,
        )

    rows.append(row)

print("Loaded posterior means and 2.5/97.5 percentile intervals for all matched pairs.")
print("Delay parameters are intentionally excluded because ILD2 delay is a stimulus-dependent function.")


# %%
# Plot one PDF page per comparable parameter.

labels = [row["label"] for row in rows]
x = np.arange(len(labels))

with PdfPages(OUTPUT_PDF) as pdf:
    for _, param_name, param_label, _ in PARAM_CONFIGS:
        fig_width = max(12, 0.35 * len(labels))
        fig, ax = plt.subplots(figsize=(fig_width, 6))

        alpha_mean = np.array([row["alpha"][param_name][0] for row in rows])
        alpha_low = np.array([row["alpha"][param_name][1] for row in rows])
        alpha_high = np.array([row["alpha"][param_name][2] for row in rows])
        alpha_yerr = np.vstack([
            alpha_mean - alpha_low,
            alpha_high - alpha_mean,
        ])

        ild2_mean = np.array([row["ild2"][param_name][0] for row in rows])
        ild2_low = np.array([row["ild2"][param_name][1] for row in rows])
        ild2_high = np.array([row["ild2"][param_name][2] for row in rows])
        ild2_yerr = np.vstack([
            ild2_mean - ild2_low,
            ild2_high - ild2_mean,
        ])

        ax.errorbar(
            x - X_OFFSET,
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
        ax.errorbar(
            x + X_OFFSET,
            ild2_mean,
            yerr=ild2_yerr,
            fmt="o",
            color=ILD2_COLOR,
            ecolor=ILD2_COLOR,
            elinewidth=1.3,
            capsize=3,
            markersize=4,
            label="NPL + alpha + ILD2 delay",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=55, ha="right")
        ax.set_xlabel("Animal")
        ax.set_ylabel(param_label)
        ax.set_title(f"{param_label}: NPL + alpha vs NPL + alpha + ILD2 delay")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

print(f"Saved comparison PDF: {OUTPUT_PDF}")

# %%
