# %%
"""
Plot fitted ILD2-delay surfaces from NPL+alpha+ILD2 animal-wise result pickles.

Each animal gets one PDF page with abs(ILD) on the x-axis, ABL on the y-axis,
and fitted evidence delay in milliseconds as color. Final pages show the
across-animal mean and median for each observed stimulus cell.
"""
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

RESULTS_DIR = os.path.join(REPO_DIR, "NPL_alpha_ILD2_fit_results", "result_pkls")
OUTPUT_DIR = os.path.join(REPO_DIR, "NPL_alpha_ILD2_fit_results", "delay_heatmaps")
OUTPUT_PDF = os.path.join(OUTPUT_DIR, "npl_alpha_ild2_delay_heatmaps_by_animal.pdf")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RESULT_KEY = "vbmc_norm_alpha_ild2_delay_tied_results"
RESULT_PATTERN = re.compile(
    r"^results_(?P<batch>.+)_animal_(?P<animal>\d+)_NORM_ALPHA_ILD2_DELAY_FROM_ABORTS\.pkl$"
)

DESIRED_BATCHES = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]
BATCH_ORDER = {batch: idx for idx, batch in enumerate(DESIRED_BATCHES)}

DELAY_SAMPLE_KEYS = {
    "bias_ms": "bias_ms_samples",
    "abl_coeff": "abl_delay_coeff_ms_per_abl_samples",
    "abs_ild_coeff": "abs_ild_delay_coeff_ms_per_unit_samples",
    "ild2_coeff": "ild2_delay_coeff_ms_per_unit2_samples",
}

CMAP_NAME = "viridis"
FIGSIZE = (8.5, 6.5)
ANNOTATION_FONTSIZE = 9


# %%
# Helpers

def parse_result_filename(fname):
    match = RESULT_PATTERN.match(fname)
    if match is None:
        return None
    return match.group("batch"), int(match.group("animal"))


def posterior_mean_coefficients(model_results, pkl_path):
    missing = [
        sample_key
        for sample_key in DELAY_SAMPLE_KEYS.values()
        if sample_key not in model_results
    ]
    if missing:
        raise KeyError(f"{pkl_path} is missing delay coefficient sample keys: {missing}")

    return {
        coeff_name: float(np.mean(np.asarray(model_results[sample_key], dtype=float)))
        for coeff_name, sample_key in DELAY_SAMPLE_KEYS.items()
    }


def format_signed_term(value, label):
    sign = "+" if value >= 0 else "-"
    return f" {sign} {abs(value):.5g}*{label}"


def delay_expression(coeffs):
    return (
        f"delay_ms = {coeffs['bias_ms']:.5g}"
        f"{format_signed_term(coeffs['abl_coeff'], 'ABL')}"
        f"{format_signed_term(coeffs['abs_ild_coeff'], '|ILD|')}"
        f"{format_signed_term(coeffs['ild2_coeff'], '|ILD|^2')}"
    )


def compute_delay_ms(abl, abs_ild, coeffs):
    return (
        coeffs["bias_ms"]
        + coeffs["abl_coeff"] * abl
        + coeffs["abs_ild_coeff"] * abs_ild
        + coeffs["ild2_coeff"] * (abs_ild ** 2)
    )


def heatmap_matrix_from_cells(cells, abls, abs_ilds):
    matrix = np.full((len(abls), len(abs_ilds)), np.nan)
    abl_to_idx = {abl: idx for idx, abl in enumerate(abls)}
    abs_ild_to_idx = {abs_ild: idx for idx, abs_ild in enumerate(abs_ilds)}

    for abl, abs_ild, delay_ms in cells:
        if abl in abl_to_idx and abs_ild in abs_ild_to_idx:
            matrix[abl_to_idx[abl], abs_ild_to_idx[abs_ild]] = delay_ms
    return matrix


def draw_delay_heatmap(pdf, matrix, abls, abs_ilds, title, subtitle, vmin, vmax):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    cmap = plt.get_cmap(CMAP_NAME).copy()
    cmap.set_bad("white")

    masked_matrix = np.ma.masked_invalid(matrix)
    image = ax.imshow(
        masked_matrix,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xticks(np.arange(len(abs_ilds)))
    ax.set_xticklabels([str(abs_ild) for abs_ild in abs_ilds])
    ax.set_yticks(np.arange(len(abls)))
    ax.set_yticklabels([str(abl) for abl in abls])
    ax.set_xlabel("|ILD|")
    ax.set_ylabel("ABL")
    ax.set_title(title, fontsize=12, pad=12)
    fig.suptitle(subtitle, fontsize=10, y=0.96)

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if not np.isfinite(value):
                continue
            text_color = "white" if value > (vmin + vmax) / 2 else "black"
            ax.text(
                col_idx,
                row_idx,
                f"{value:.1f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=ANNOTATION_FONTSIZE,
            )

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Delay (ms)")
    ax.grid(False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    pdf.savefig(fig)
    plt.close(fig)


# %%
# Load result pickles and build animal delay matrices.

result_paths = {}
for fname in os.listdir(RESULTS_DIR):
    parsed = parse_result_filename(fname)
    if parsed is None:
        continue
    result_paths[parsed] = os.path.join(RESULTS_DIR, fname)

matched_pairs = sorted(
    result_paths.keys(),
    key=lambda pair: (BATCH_ORDER.get(pair[0], len(BATCH_ORDER)), pair[1]),
)

if len(matched_pairs) == 0:
    raise RuntimeError(f"No ILD2 delay result pickles found in {RESULTS_DIR}")

animal_rows = []
all_observed_abls = set()
all_observed_abs_ilds = set()

for batch_name, animal_id in matched_pairs:
    pkl_path = result_paths[(batch_name, animal_id)]
    with open(pkl_path, "rb") as f:
        saved_data = pickle.load(f)

    if RESULT_KEY not in saved_data:
        raise KeyError(f"{pkl_path} is missing `{RESULT_KEY}`")

    coeffs = posterior_mean_coefficients(saved_data[RESULT_KEY], pkl_path)
    fit_config = saved_data.get("fit_config", {})
    condition_pairs = np.asarray(
        fit_config.get("observed_delay_condition_pairs_used_for_fit", []),
        dtype=float,
    )

    if condition_pairs.size == 0:
        raise ValueError(f"{pkl_path} has no observed delay condition pairs in fit_config")

    animal_cells = []
    for abl, abs_ild, _ild2 in condition_pairs:
        abl = int(abl)
        abs_ild = int(abs_ild)
        all_observed_abls.add(abl)
        all_observed_abs_ilds.add(abs_ild)
        animal_cells.append((abl, abs_ild, compute_delay_ms(abl, abs_ild, coeffs)))

    animal_rows.append(
        {
            "batch_name": batch_name,
            "animal": animal_id,
            "coeffs": coeffs,
            "cells": animal_cells,
            "pkl_path": pkl_path,
        }
    )

abls = sorted(all_observed_abls)
abs_ilds = sorted(all_observed_abs_ilds)

for row in animal_rows:
    row["matrix"] = heatmap_matrix_from_cells(row["cells"], abls, abs_ilds)

delay_stack = np.stack([row["matrix"] for row in animal_rows], axis=0)
all_finite_delays = delay_stack[np.isfinite(delay_stack)]
vmin = float(np.nanmin(all_finite_delays))
vmax = float(np.nanmax(all_finite_delays))

mean_matrix = np.full((len(abls), len(abs_ilds)), np.nan)
median_matrix = np.full((len(abls), len(abs_ilds)), np.nan)
n_matrix = np.sum(np.isfinite(delay_stack), axis=0)

for row_idx in range(len(abls)):
    for col_idx in range(len(abs_ilds)):
        cell_values = delay_stack[:, row_idx, col_idx]
        cell_values = cell_values[np.isfinite(cell_values)]
        if cell_values.size == 0:
            continue
        mean_matrix[row_idx, col_idx] = np.mean(cell_values)
        median_matrix[row_idx, col_idx] = np.median(cell_values)

print(f"Found {len(animal_rows)} ILD2 delay result pickles.")
print(f"ABL grid: {abls}")
print(f"|ILD| grid: {abs_ilds}")
print(f"Delay color scale: {vmin:.3f} to {vmax:.3f} ms")

print("Delay coefficient summary across animals:")
for coeff_name, label in [
    ("bias_ms", "intercept_ms"),
    ("abl_coeff", "abl_delay_coeff_ms_per_abl"),
    ("abs_ild_coeff", "abs_ild_delay_coeff_ms_per_unit"),
    ("ild2_coeff", "ild2_delay_coeff_ms_per_unit2"),
]:
    values = np.asarray([row["coeffs"][coeff_name] for row in animal_rows], dtype=float)
    print(
        f"  {label}: mean={np.mean(values):.6g}, "
        f"median={np.median(values):.6g}, "
        f"sem={np.std(values) / np.sqrt(len(values)):.6g}"
    )


# %%
# Save multipage delay heatmap PDF.

with PdfPages(OUTPUT_PDF) as pdf:
    for row in animal_rows:
        batch_name = row["batch_name"]
        animal_id = row["animal"]
        title = f"{batch_name} animal {animal_id}"
        subtitle = delay_expression(row["coeffs"])
        draw_delay_heatmap(
            pdf,
            row["matrix"],
            abls,
            abs_ilds,
            title,
            subtitle,
            vmin,
            vmax,
        )

    draw_delay_heatmap(
        pdf,
        mean_matrix,
        abls,
        abs_ilds,
        "Across-animal mean delay",
        "Mean uses only animals where each stimulus cell was observed in the fit grid",
        vmin,
        vmax,
    )
    draw_delay_heatmap(
        pdf,
        median_matrix,
        abls,
        abs_ilds,
        "Across-animal median delay",
        "Median uses only animals where each stimulus cell was observed in the fit grid",
        vmin,
        vmax,
    )

print(f"Saved PDF: {OUTPUT_PDF}")
print("Number of animals per summary cell:")
print("Rows are ABL, columns are |ILD|")
print(n_matrix.astype(int))

# %%
