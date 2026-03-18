"""
Standalone ABL-wise RT density overlay with area normalization undone.
"""

# %%
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np

# %%
# =============================================================================
# PARAMETERS
# =============================================================================
ROOT = Path(__file__).resolve().parent
RAW_DIAG_PKL = (
    ROOT
    / "fitting_aborts"
    / "norm_only_led_off_from_loaded_proactive_truncate_NOT_censor_ABL_delay"
    / "diagnostics"
    / "diag_norm_tied_batch_LED7_aggregate_ledoff_raw_rtwrtstim_truncate_not_censor_ABL_delay.pkl"
)
OUTPUT_DIR = ROOT / "fct_march_26_nonorm"
OUTPUT_BASENAME = "abl_density_nonorm_LED7_115ms"
OUTPUT_DIR_V2 = ROOT / "fct_march_26_nonorm_v2"

SHOW_PLOT = True
SAVE_DPI = 450

PANEL_FIGSIZE = (6.4, 4.8)
AXIS_LABEL_FONTSIZE = 25
AXIS_TICK_FONTSIZE = 24
LINEWIDTH_THEORY = 2.5
LINEWIDTH_DATA = 1.2
DATA_ALPHA = 0.65
AXIS_SPINE_LW = 2.2
TICK_WIDTH = 1.8
TICK_LENGTH = 6
XLABEL_PAD = 3
YLABEL_PAD = 2
Y_LIM = (0.0, 7.0)

ABL_COLORS = {
    20: "tab:blue",
    40: "tab:orange",
    60: "tab:green",
}


# %%
# =============================================================================
# Helpers
# =============================================================================
def load_payload(payload_path):
    if not payload_path.exists():
        raise FileNotFoundError(f"Payload not found: {payload_path}")
    with open(payload_path, "rb") as f:
        return pickle.load(f)


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(AXIS_SPINE_LW)
    ax.spines["left"].set_linewidth(AXIS_SPINE_LW)
    ax.tick_params(
        axis="x",
        labelsize=AXIS_TICK_FONTSIZE,
        width=TICK_WIDTH,
        length=TICK_LENGTH,
    )
    ax.tick_params(
        axis="y",
        labelsize=AXIS_TICK_FONTSIZE,
        width=TICK_WIDTH,
        length=TICK_LENGTH,
    )


def build_non_normalized_curves(payload):
    t_ms = 1e3 * np.asarray(payload["t_pts_truncated"], dtype=float)
    data_x_ms = 1e3 * np.asarray(payload["data_hist_centers_truncated"], dtype=float)
    data_edges_s = np.asarray(payload["data_hist_edges_truncated"], dtype=float)
    bin_widths_s = np.diff(data_edges_s)

    curves_by_abl = {}
    for abl in payload["config"]["supported_ABL_values"]:
        abl_int = int(abl)
        abl_payload = payload["abl_split_truncated"][abl_int]
        truncated_fraction = abl_payload["n_truncated_points"] / abl_payload["n_rows"]

        data_density_conditional = np.asarray(
            abl_payload["data_density_truncated"], dtype=float
        )
        data_density_all_trials = data_density_conditional * truncated_fraction
        theory_density_raw = np.asarray(
            abl_payload["theory_density_truncated_raw"], dtype=float
        )

        curves_by_abl[abl_int] = {
            "data_y": data_density_all_trials,
            "theory_y": theory_density_raw,
            "data_area": float(np.sum(data_density_all_trials * bin_widths_s)),
            "theory_area": float(
                np.trapz(theory_density_raw, np.asarray(payload["t_pts_truncated"], dtype=float))
            ),
            "truncated_fraction": float(truncated_fraction),
        }

    return t_ms, data_x_ms, curves_by_abl


# %%
# =============================================================================
# Load payload + reconstruct non-normalized curves
# =============================================================================
payload = load_payload(RAW_DIAG_PKL)
t_ms, data_x_ms, curves_by_abl = build_non_normalized_curves(payload)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_V2.mkdir(parents=True, exist_ok=True)
truncate_ms = int(round(float(payload["config"]["truncate_rt_wrt_stim_s"]) * 1e3))


# %%
# =============================================================================
# Build figure
# =============================================================================
fig, ax = plt.subplots(figsize=PANEL_FIGSIZE, constrained_layout=True)

peak_density = 0.0
for abl in payload["config"]["supported_ABL_values"]:
    abl_int = int(abl)
    color = ABL_COLORS.get(abl_int, "0.35")
    curves = curves_by_abl[abl_int]

    ax.step(
        data_x_ms,
        curves["data_y"],
        where="mid",
        color=color,
        lw=LINEWIDTH_DATA,
        alpha=DATA_ALPHA,
    )
    ax.plot(
        t_ms,
        curves["theory_y"],
        color=color,
        lw=LINEWIDTH_THEORY,
    )
    peak_density = max(
        peak_density,
        float(np.max(curves["data_y"])),
        float(np.max(curves["theory_y"])),
    )

ax.set_xlim(0.0, truncate_ms)
ax.set_xticks([0, 100])
ax.set_xlabel(r"RT - $t_{stim}$ (ms)", fontsize=AXIS_LABEL_FONTSIZE, labelpad=XLABEL_PAD)
ax.set_ylabel("Density", fontsize=AXIS_LABEL_FONTSIZE, labelpad=YLABEL_PAD)

ax.set_ylim(*Y_LIM)
ax.set_yticks([0, 6])

style_axes(ax)


# %%
# =============================================================================
# Save + diagnostics
# =============================================================================
output_paths = {
    "pdf": OUTPUT_DIR / f"{OUTPUT_BASENAME}.pdf",
    "png": OUTPUT_DIR / f"{OUTPUT_BASENAME}.png",
    "svg": OUTPUT_DIR / f"{OUTPUT_BASENAME}.svg",
    "eps": OUTPUT_DIR / f"{OUTPUT_BASENAME}.eps",
}

fig.savefig(output_paths["pdf"], bbox_inches="tight")
fig.savefig(output_paths["png"], dpi=SAVE_DPI, bbox_inches="tight")
fig.savefig(output_paths["svg"], bbox_inches="tight")
fig.savefig(output_paths["eps"], bbox_inches="tight")

for suffix, path in output_paths.items():
    print(f"Saved {suffix.upper()}: {path}")

output_paths_v2 = {
    "pdf": OUTPUT_DIR_V2 / f"{OUTPUT_BASENAME}.pdf",
    "png": OUTPUT_DIR_V2 / f"{OUTPUT_BASENAME}.png",
    "svg": OUTPUT_DIR_V2 / f"{OUTPUT_BASENAME}.svg",
    "eps": OUTPUT_DIR_V2 / f"{OUTPUT_BASENAME}.eps",
}

fig.savefig(output_paths_v2["pdf"], bbox_inches="tight")
fig.savefig(output_paths_v2["png"], dpi=SAVE_DPI, bbox_inches="tight")
fig.savefig(output_paths_v2["svg"], bbox_inches="tight")
fig.savefig(output_paths_v2["eps"], bbox_inches="tight")

for suffix, path in output_paths_v2.items():
    print(f"Saved V2 {suffix.upper()}: {path}")

for abl in payload["config"]["supported_ABL_values"]:
    abl_int = int(abl)
    curves = curves_by_abl[abl_int]
    print(
        f"ABL {abl_int}: "
        f"data_area={curves['data_area']:.6f}, "
        f"truncated_fraction={curves['truncated_fraction']:.6f}, "
        f"theory_area={curves['theory_area']:.6f}"
    )

if SHOW_PLOT:
    plt.show()
else:
    plt.close(fig)
