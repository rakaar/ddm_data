# %%
"""Plot choice-collapsed proactive+lapse+reactive RTDs from -1 to 2 s."""

# %%
from pathlib import Path
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# %%
# =============================================================================
# Editable paths and display settings
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
VALIDATION_DIR = SCRIPT_DIR / "proactive_led_step_jump_npl_alpha_validation"
PAYLOAD_FILE = VALIDATION_DIR / "simulation_vs_likelihood_plot_data.pkl"
DISPLAY_LOW_S = -1.0
DISPLAY_HIGH_S = 2.0

if not PAYLOAD_FILE.exists():
    raise FileNotFoundError(PAYLOAD_FILE)

with PAYLOAD_FILE.open("rb") as handle:
    payload = pickle.load(handle)

if payload["settings"]["n_sim"] != 100_000:
    raise RuntimeError(
        "Expected the final 100,000-trial validation payload, found "
        f"{payload['settings']['n_sim']:,}."
    )
if payload["settings"]["lapse_semantics"] != (
    "direct exponential-time random choice"
):
    raise RuntimeError("Unexpected lapse semantics in validation payload.")

all_case_rows = payload["parameter_cases"]
selected_cases = [
    (case_index, case)
    for case_index, case in enumerate(all_case_rows)
    if case["case_key"] != "downward_jump"
]
plot_payload = payload["plot_payload"]
hist_bin_s = float(payload["settings"]["hist_bin_s"])

print(f"Validation payload: {PAYLOAD_FILE}")
print("Model: (proactive + exponential random-choice lapses) + reactive")
print("Density scaling: raw probability per original simulated trial")
print("Display only: -1 to 2 s; no truncation or renormalization")
print("Cases: no-jump control, upward-jump model, LED7/93 fitted upward jump")


# %%
# =============================================================================
# Sum choice-specific densities and draw separate OFF and ON figures
# =============================================================================
area_rows = []
figure_paths = []

for category in ["off", "on"]:
    model_color = "tab:blue" if category == "off" else "tab:red"
    fig, axes = plt.subplots(
        1,
        len(selected_cases),
        figsize=(12.2, 3.8),
        sharex=True,
        sharey=True,
    )
    category_cells = []
    y_max = 0.0

    for source_case_index, case in selected_cases:
        cell = plot_payload[(source_case_index, category, True)]
        histogram = cell["raw_hist_plus"] + cell["raw_hist_minus"]
        theory = cell["raw_theory_plus"] + cell["raw_theory_minus"]
        histogram_centers = np.asarray(cell["raw_centers"], dtype=float)
        theory_rt = np.asarray(cell["raw_grid"], dtype=float)

        histogram_window = (
            (histogram_centers >= DISPLAY_LOW_S)
            & (histogram_centers < DISPLAY_HIGH_S)
        )
        theory_window = (
            (theory_rt >= DISPLAY_LOW_S)
            & (theory_rt <= DISPLAY_HIGH_S)
        )
        simulation_area = float(
            np.sum(histogram[histogram_window]) * hist_bin_s
        )
        theory_area = float(
            np.trapz(theory[theory_window], theory_rt[theory_window])
        )
        if not np.all(np.isfinite(histogram)) or np.any(histogram < 0.0):
            raise RuntimeError(f"Invalid simulated density for {case['case_label']}.")
        if not np.all(np.isfinite(theory)) or np.any(theory < -1e-10):
            raise RuntimeError(f"Invalid theory density for {case['case_label']}.")

        category_cells.append(
            {
                "histogram": histogram,
                "histogram_centers": histogram_centers,
                "theory": theory,
                "theory_rt": theory_rt,
                "simulation_area": simulation_area,
                "theory_area": theory_area,
            }
        )
        y_max = max(
            y_max,
            float(np.max(histogram[histogram_window])),
            float(np.max(theory[theory_window])),
        )
        area_rows.append(
            {
                "category": category,
                "case_index": source_case_index,
                "case_key": case["case_key"],
                "case_label": case["case_label"],
                "display_low_s": DISPLAY_LOW_S,
                "display_high_s": DISPLAY_HIGH_S,
                "simulation_area": simulation_area,
                "theory_area": theory_area,
                "absolute_area_error": abs(simulation_area - theory_area),
            }
        )

    for panel_index, ((_, case), cell) in enumerate(
        zip(selected_cases, category_cells)
    ):
        ax = axes[panel_index]
        ax.step(
            cell["histogram_centers"],
            cell["histogram"],
            where="mid",
            color="black",
            lw=0.9,
            alpha=0.62,
            zorder=3,
        )
        ax.plot(
            cell["theory_rt"],
            cell["theory"],
            color=model_color,
            lw=1.5,
            alpha=0.68,
            zorder=4,
        )
        ax.axvline(0.0, color="0.55", lw=0.8, ls=":", zorder=1)
        ax.set_xlim(DISPLAY_LOW_S, DISPLAY_HIGH_S)
        ax.set_ylim(0.0, 1.08 * y_max)
        panel_title = case["case_label"]
        if case["case_key"] == "no_jump":
            panel_title += " (control)"
        ax.set_title(panel_title, fontsize=10)
        ax.set_xlabel("RT relative to stimulus (s)")
        ax.text(
            0.98,
            0.96,
            "Area in [-1, 2] s\n"
            f"simulation={cell['simulation_area']:.3f}\n"
            f"likelihood={cell['theory_area']:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
        )
        if panel_index == 0:
            ax.set_ylabel("Choice-collapsed RT density")

    fig.suptitle(
        f"LED {category.upper()}: (proactive + lapses) + reactive RTD",
        y=0.995,
        fontsize=13,
    )
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="black",
                lw=0.9,
                alpha=0.62,
                label="Simulation",
            ),
            Line2D(
                [0],
                [0],
                color=model_color,
                lw=1.5,
                alpha=0.68,
                label="Analytic likelihood",
            ),
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.93),
    )
    fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.84))
    figure_path = VALIDATION_DIR / (
        f"led_{category}_with_lapse_choice_collapsed_rtd_minus1_to2s.png"
    )
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    figure_paths.append(figure_path)


# %%
# =============================================================================
# Save the displayed-window area audit
# =============================================================================
area_df = pd.DataFrame(area_rows)
area_csv = VALIDATION_DIR / (
    "choice_collapsed_rtd_minus1_to2s_area_summary.csv"
)
area_df.to_csv(area_csv, index=False)

if area_df["absolute_area_error"].max() > 0.015:
    raise AssertionError(
        "Simulation/theory area mismatch exceeded 0.015 in the displayed window."
    )

print("\nDisplayed-window area comparison:")
print(
    area_df[
        [
            "category",
            "case_label",
            "simulation_area",
            "theory_area",
            "absolute_area_error",
        ]
    ].to_string(index=False, float_format=lambda value: f"{value:.6g}")
)
print("\nSaved:")
for output_path in [*figure_paths, area_csv]:
    print(f"  {output_path}")
