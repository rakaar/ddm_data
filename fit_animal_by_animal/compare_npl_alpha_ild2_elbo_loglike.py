# %%
# Compare raw ELBO and loglike for NPL, NPL + alpha, and NPL + alpha + ILD2-delay fits.

import os
import pickle
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


# %%
# Editable parameters

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

NPL_RESULTS_DIR = SCRIPT_DIR
ALPHA_RESULTS_DIR = os.path.join(SCRIPT_DIR, "NPL_alpha_animal_fits")
ILD2_RESULTS_DIR = os.path.join(REPO_DIR, "NPL_alpha_ILD2_fit_results", "result_pkls")

OUTPUT_DIR = os.path.join(REPO_DIR, "NPL_alpha_ILD2_fit_results", "elbo_loglike_comparison")
OUTPUT_PDF = os.path.join(OUTPUT_DIR, "compare_npl_alpha_ild2_elbo_loglike.pdf")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "compare_npl_alpha_ild2_elbo_loglike.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DESIRED_BATCHES = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]
BATCH_ORDER = {batch: idx for idx, batch in enumerate(DESIRED_BATCHES)}

MODEL_CONFIGS = [
    {
        "model_name": "NPL",
        "result_key": "vbmc_norm_tied_results",
        "results_dir": NPL_RESULTS_DIR,
        "pattern": re.compile(r"^results_(?P<batch>.+)_animal_(?P<animal>\d+)\.pkl$"),
        "color": "0.25",
        "x_offset": -0.22,
    },
    {
        "model_name": "NPL + alpha",
        "result_key": "vbmc_norm_alpha_tied_results",
        "results_dir": ALPHA_RESULTS_DIR,
        "pattern": re.compile(
            r"^results_(?P<batch>.+)_animal_(?P<animal>\d+)_NORM_ALPHA_FROM_ABORTS\.pkl$"
        ),
        "color": "tab:blue",
        "x_offset": 0.0,
    },
    {
        "model_name": "NPL + alpha + ILD2",
        "result_key": "vbmc_norm_alpha_ild2_delay_tied_results",
        "results_dir": ILD2_RESULTS_DIR,
        "pattern": re.compile(
            r"^results_(?P<batch>.+)_animal_(?P<animal>\d+)_NORM_ALPHA_ILD2_DELAY_FROM_ABORTS\.pkl$"
        ),
        "color": "tab:red",
        "x_offset": 0.22,
    },
]


# %%
# Helpers

def discover_result_paths(model_config):
    paths = {}
    for fname in os.listdir(model_config["results_dir"]):
        match = model_config["pattern"].match(fname)
        if match is None:
            continue
        batch = match.group("batch")
        animal = int(match.group("animal"))
        paths[(batch, animal)] = os.path.join(model_config["results_dir"], fname)
    return paths


def load_model_metrics(model_config, pkl_path):
    with open(pkl_path, "rb") as f:
        saved_data = pickle.load(f)

    result_key = model_config["result_key"]
    if result_key not in saved_data:
        raise KeyError(f"{pkl_path} is missing `{result_key}`")

    model_results = saved_data[result_key]
    return {
        "elbo": float(model_results.get("elbo", np.nan)),
        "elbo_sd": float(model_results.get("elbo_sd", np.nan)),
        "loglike": float(model_results.get("loglike", np.nan)),
        "message": model_results.get("message", ""),
    }


def plot_metric_page(pdf, rows_df, matched_pairs, metric_name, ylabel, title):
    labels = [f"{batch}-{animal}" for batch, animal in matched_pairs]
    x = np.arange(len(labels))
    fig_width = max(12, 0.36 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    for model_config in MODEL_CONFIGS:
        model_name = model_config["model_name"]
        model_df = rows_df[rows_df["model"] == model_name].set_index(["batch", "animal"])

        y_values = np.array(
            [model_df.loc[(batch, animal), metric_name] for batch, animal in matched_pairs],
            dtype=float,
        )
        x_values = x + model_config["x_offset"]

        if metric_name == "elbo":
            yerr = np.array(
                [model_df.loc[(batch, animal), "elbo_sd"] for batch, animal in matched_pairs],
                dtype=float,
            )
            yerr = np.where(np.isfinite(yerr), yerr, 0.0)
            ax.errorbar(
                x_values,
                y_values,
                yerr=yerr,
                fmt="o",
                color=model_config["color"],
                ecolor=model_config["color"],
                elinewidth=1.2,
                capsize=3,
                markersize=4,
                label=model_name,
            )
        else:
            ax.plot(
                x_values,
                y_values,
                "o",
                color=model_config["color"],
                markersize=4,
                label=model_name,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right")
    ax.set_xlabel("Animal")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# %%
# Discover and match result files.

paths_by_model = {
    model_config["model_name"]: discover_result_paths(model_config)
    for model_config in MODEL_CONFIGS
}

for model_config in MODEL_CONFIGS:
    model_name = model_config["model_name"]
    print(f"Found {len(paths_by_model[model_name])} {model_name} result pickle files.")

matched_pairs = sorted(
    set.intersection(*(set(paths.keys()) for paths in paths_by_model.values())),
    key=lambda pair: (BATCH_ORDER.get(pair[0], len(BATCH_ORDER)), pair[1]),
)

if len(matched_pairs) == 0:
    raise RuntimeError("No animals are matched across NPL, NPL + alpha, and ILD2 result files.")

print(f"Using {len(matched_pairs)} matched batch-animal pairs.")

for model_config in MODEL_CONFIGS:
    model_name = model_config["model_name"]
    missing = sorted(set(matched_pairs) ^ set(paths_by_model[model_name]))
    if missing:
        print(f"{model_name} has extra or missing files relative to matched set:")
        for batch, animal in missing:
            print(f"  {batch}-{animal}")


# %%
# Load metrics and save CSV.

rows = []
for batch, animal in matched_pairs:
    for model_config in MODEL_CONFIGS:
        model_name = model_config["model_name"]
        pkl_path = paths_by_model[model_name][(batch, animal)]
        metrics = load_model_metrics(model_config, pkl_path)
        rows.append(
            {
                "batch": batch,
                "animal": animal,
                "label": f"{batch}-{animal}",
                "model": model_name,
                "result_key": model_config["result_key"],
                "pkl_path": pkl_path,
                **metrics,
            }
        )

rows_df = pd.DataFrame(rows)
rows_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved CSV: {OUTPUT_CSV}")

print("Mean raw metrics across matched animals:")
for model_name, group_df in rows_df.groupby("model", sort=False):
    print(
        f"  {model_name}: "
        f"loglike={group_df['loglike'].mean():.6g}, "
        f"ELBO={group_df['elbo'].mean():.6g}"
    )


# %%
# Save two-page PDF.

with PdfPages(OUTPUT_PDF) as pdf:
    plot_metric_page(
        pdf,
        rows_df,
        matched_pairs,
        "loglike",
        "Raw loglike",
        "Raw loglike by animal",
    )
    plot_metric_page(
        pdf,
        rows_df,
        matched_pairs,
        "elbo",
        "Raw ELBO",
        "Raw ELBO by animal",
    )

print(f"Saved PDF: {OUTPUT_PDF}")

# %%
