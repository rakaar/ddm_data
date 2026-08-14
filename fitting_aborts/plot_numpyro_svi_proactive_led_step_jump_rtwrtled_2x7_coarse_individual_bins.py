# %%
"""Replot the LED7 RT-wrt-LED diagnostic with coarser individual data bins."""

# %%
# =============================================================================
# Parameters
# =============================================================================
from pathlib import Path
import os
import pickle

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_ROOT = (
    SCRIPT_DIR
    / "numpyro_svi_proactive_led_step_jump_bilateral_unweighted_"
    "patience12_min50k_restore_best_outputs"
    / "summary_figures"
)
SOURCE_PAYLOAD = SOURCE_ROOT / "led7_proactive_led_step_jump_rtwrtled_data_model_2x7.pkl"
FIG_PATH = (
    SOURCE_ROOT
    / "led7_proactive_led_step_jump_rtwrtled_data_model_"
    "2x7_coarse_individual_bins.png"
)

ANIMALS = (92, 93, 98, 99, 100, 103)
HIST_RANGE_S = (-3.0, 3.0)
INDIVIDUAL_REGULAR_BIN_S = 0.040
INDIVIDUAL_ZOOM_BIN_S = 0.020
POOLED_REGULAR_BIN_S = 0.010
POOLED_ZOOM_BIN_S = 0.005
REGULAR_XLIM_MS = (-300.0, 400.0)
ZOOM_XLIM_MS = (-200.0, 200.0)

LED_ON_COLOR = "#D62728"
LED_OFF_COLOR = "#1F77B4"
DATA_ALPHA = 0.48


# %%
# =============================================================================
# Imports and source payload
# =============================================================================
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

if not SOURCE_PAYLOAD.exists():
    raise FileNotFoundError(SOURCE_PAYLOAD)

with SOURCE_PAYLOAD.open("rb") as handle:
    source = pickle.load(handle)

if tuple(source["config"]["animals"]) != ANIMALS:
    raise RuntimeError("Source payload does not contain the expected six LED7 animals.")


# %%
# =============================================================================
# Re-bin only the empirical data; preserve the source theory exactly
# =============================================================================
def histogram_rate(values, n_total, bin_width_s):
    edges = np.arange(
        HIST_RANGE_S[0],
        HIST_RANGE_S[1] + 0.5 * bin_width_s,
        bin_width_s,
    )
    counts, _ = np.histogram(np.asarray(values, dtype=float), bins=edges)
    density = counts.astype(float) / (float(n_total) * bin_width_s)
    centers = 0.5 * (edges[:-1] + edges[1:])
    expected_area = len(values) / float(n_total)
    observed_area = float(np.sum(density * np.diff(edges)))
    if not np.isclose(observed_area, expected_area, atol=1e-12, rtol=1e-12):
        raise RuntimeError("Re-binned data do not preserve the abort-fraction area.")
    return centers, density


columns = []
for animal in ANIMALS:
    payload = source["per_animal"][animal]
    plotted = {
        "title": f"LED7/{animal}",
        "delay_s": payload["posterior_means"]["del_m_plus_del_LED"],
        "conditions": {},
    }
    for condition in ("on", "off"):
        condition_source = payload["conditions"][condition]
        regular_x, regular_density = histogram_rate(
            condition_source["rt_wrt_led_s"],
            condition_source["n_total"],
            INDIVIDUAL_REGULAR_BIN_S,
        )
        zoom_x, zoom_density = histogram_rate(
            condition_source["rt_wrt_led_s"],
            condition_source["n_total"],
            INDIVIDUAL_ZOOM_BIN_S,
        )
        plotted["conditions"][condition] = {
            "regular_x_s": regular_x,
            "regular_data_density": regular_density,
            "zoom_x_s": zoom_x,
            "zoom_data_density": zoom_density,
            "theory_x_s": condition_source["theory_x_s"],
            "theory_density": condition_source["theory_density"],
        }
    columns.append(plotted)

pooled_source = source["pooled_all_animals"]
pooled = {
    "title": "All animals\npooled",
    "delay_s": pooled_source["posterior_mean_del_m_plus_del_LED"],
    "conditions": {},
}
for condition in ("on", "off"):
    condition_source = pooled_source["conditions"][condition]
    regular_x, regular_density = histogram_rate(
        condition_source["rt_wrt_led_s"],
        condition_source["n_total"],
        POOLED_REGULAR_BIN_S,
    )
    zoom_x, zoom_density = histogram_rate(
        condition_source["rt_wrt_led_s"],
        condition_source["n_total"],
        POOLED_ZOOM_BIN_S,
    )
    pooled["conditions"][condition] = {
        "regular_x_s": regular_x,
        "regular_data_density": regular_density,
        "zoom_x_s": zoom_x,
        "zoom_data_density": zoom_density,
        "theory_x_s": condition_source["theory_x_s"],
        "theory_density": condition_source["theory_density"],
    }
columns.append(pooled)
pooled_switch_delays_ms = [
    1000.0
    * source["per_animal"][animal]["posterior_means"]["del_m_plus_del_LED"]
    for animal in ANIMALS
]


# %%
# =============================================================================
# Plot
# =============================================================================
fig, axes = plt.subplots(2, 7, figsize=(24.0, 7.0), sharey="row")

for column_index, payload in enumerate(columns):
    delay_ms = 1000.0 * payload["delay_s"]
    if column_index < len(ANIMALS):
        switch_delays_ms = [delay_ms]
        panel_title = f"{payload['title']}\nswitch = {delay_ms:.0f} ms"
    else:
        switch_delays_ms = pooled_switch_delays_ms
        panel_title = payload["title"]

    for row_index, view in enumerate(("regular", "zoom")):
        ax = axes[row_index, column_index]
        for condition, color in (("on", LED_ON_COLOR), ("off", LED_OFF_COLOR)):
            condition_payload = payload["conditions"][condition]
            ax.step(
                1000.0 * condition_payload[f"{view}_x_s"],
                condition_payload[f"{view}_data_density"],
                where="mid",
                color=color,
                alpha=DATA_ALPHA,
                lw=1.1,
                zorder=2,
            )
            ax.plot(
                1000.0 * condition_payload["theory_x_s"],
                condition_payload["theory_density"],
                color=color,
                lw=1.8,
                zorder=3,
            )

        ax.axvline(0.0, color="0.25", ls="--", lw=0.9, alpha=0.7, zorder=1)
        for switch_delay_ms in switch_delays_ms:
            ax.axvline(
                switch_delay_ms,
                color=LED_ON_COLOR,
                ls=":",
                lw=1.1,
                alpha=0.75 if column_index < len(ANIMALS) else 0.28,
                zorder=1,
            )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.12, lw=0.5)
        ax.tick_params(axis="both", labelsize=8)
        if row_index == 0:
            ax.set_xlim(REGULAR_XLIM_MS)
            ax.set_xticks([-300, 0, 400])
            ax.set_title(panel_title, fontsize=10)
        else:
            ax.set_xlim(ZOOM_XLIM_MS)
            ax.set_xticks([-200, 0, 200])
            ax.set_xlabel("RT wrt LED (ms)", fontsize=9)

axes[0, 0].set_ylabel("Abort rate (Hz)", fontsize=10)
axes[1, 0].set_ylabel("Abort rate (Hz)", fontsize=10)

for row_index, xlim in enumerate((REGULAR_XLIM_MS, ZOOM_XLIM_MS)):
    row_max = 0.0
    view = "regular" if row_index == 0 else "zoom"
    for payload in columns:
        for condition in ("on", "off"):
            condition_payload = payload["conditions"][condition]
            data_x_ms = 1000.0 * condition_payload[f"{view}_x_s"]
            theory_x_ms = 1000.0 * condition_payload["theory_x_s"]
            data_mask = (data_x_ms >= xlim[0]) & (data_x_ms <= xlim[1])
            theory_mask = (theory_x_ms >= xlim[0]) & (theory_x_ms <= xlim[1])
            row_max = max(
                row_max,
                float(np.max(condition_payload[f"{view}_data_density"][data_mask])),
                float(np.max(condition_payload["theory_density"][theory_mask])),
            )
    axes[row_index, 0].set_ylim(0.0, 1.08 * row_max)

legend_handles = [
    Line2D([0], [0], color=LED_ON_COLOR, lw=1.1, alpha=DATA_ALPHA, label="Data LED ON"),
    Line2D([0], [0], color=LED_ON_COLOR, lw=1.8, label="Model LED ON"),
    Line2D([0], [0], color=LED_OFF_COLOR, lw=1.1, alpha=DATA_ALPHA, label="Data LED OFF"),
    Line2D([0], [0], color=LED_OFF_COLOR, lw=1.8, label="Model LED OFF"),
    Line2D([0], [0], color="0.25", lw=0.9, ls="--", label="LED onset"),
    Line2D(
        [0],
        [0],
        color=LED_ON_COLOR,
        lw=1.1,
        ls=":",
        label=r"Fitted $\delta_m + \delta_{LED}$",
    ),
]
fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.955),
    ncol=6,
    frameon=False,
    fontsize=9,
)
fig.suptitle(
    "LED7 proactive LED step-jump SVI: coarser individual data bins",
    fontsize=13,
    y=0.995,
)
fig.subplots_adjust(
    left=0.055,
    right=0.995,
    bottom=0.10,
    top=0.86,
    wspace=0.20,
    hspace=0.32,
)
fig.savefig(FIG_PATH, dpi=250, bbox_inches="tight")
print(f"Saved figure: {FIG_PATH}")

# %%
