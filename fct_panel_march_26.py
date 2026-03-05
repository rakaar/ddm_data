"""
Build a publication-style 2x2 panel from saved plot payloads in fct_march_26/.
"""

# %%
from io import BytesIO
from pathlib import Path
import pickle

import corner
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# %%
# =============================================================================
# PARAMETERS
# =============================================================================
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "fct_march_26"
FILE_TAG = "all_animals"

SHOW_PLOT = True
SAVE_DPI = 450

PANEL_FIGSIZE = (16, 12.5)
AXIS_LABEL_FONTSIZE = 18
AXIS_TICK_FONTSIZE = 15
ANNOT_FONTSIZE = 14
CORNER_TITLE_FONTSIZE = 16
CORNER_TICK_FONTSIZE = 13
CORNER_YLABEL_FONTSIZE = 15

# %%
# =============================================================================
# Helpers
# =============================================================================
def load_payload(prefix, file_tag):
    payload_path = DATA_DIR / f"{prefix}_{file_tag}.pkl"
    if not payload_path.exists():
        matches = sorted(DATA_DIR.glob(f"{prefix}_*.pkl"))
        if len(matches) == 0:
            raise FileNotFoundError(f"No payload found for prefix '{prefix}' in {DATA_DIR}")
        payload_path = matches[0]
        print(f"[WARN] Using {payload_path.name} (requested tag: {file_tag})")

    with open(payload_path, "rb") as f:
        payload = pickle.load(f)
    return payload, payload_path


def fmt_corner_tick(x, _):
    ax = abs(x)
    if ax >= 100:
        return f"{x:.0f}"
    if ax >= 10:
        return f"{x:.1f}"
    return f"{x:.2f}"


def render_corner_image(corner_payload):
    samples = corner_payload["corner_samples"]
    labels = corner_payload["corner_labels"]
    medians = corner_payload["medians"]
    param_q = corner_payload["param_q"]
    levels = corner_payload.get("levels", [0.50, 0.80, 0.975])
    bins = int(corner_payload.get("bins", 40))
    quantiles = corner_payload.get("quantiles", [0.025, 0.50, 0.975])
    style = corner_payload.get("style", {})

    fig_tmp = corner.corner(
        samples,
        labels=labels,
        show_titles=False,
        color=style.get("color", "tab:blue"),
        fill_contours=style.get("fill_contours", True),
        plot_datapoints=style.get("plot_datapoints", False),
        plot_density=style.get("plot_density", False),
        bins=bins,
        levels=levels,
        contourf_kwargs={"colors": style.get("contourf_colors", [(1, 1, 1, 0), "#deebf7", "#9ecae1", "#4292c6"]), "alpha": 1.0},
        contour_kwargs={"colors": style.get("color", "tab:blue"), "linewidths": 1.1},
        hist_kwargs={"alpha": 0.8},
        quantiles=quantiles,
    )
    # Keep corner text readable when embedded in a multi-panel figure.
    fig_tmp.set_size_inches(11.5, 11.5)

    n_dim = samples.shape[1]
    axes = np.array(fig_tmp.axes).reshape((n_dim, n_dim))

    for i in range(n_dim):
        for j in range(n_dim):
            ax_ij = axes[i, j]
            if i < j:
                ax_ij.set_axis_off()
                continue
            ax_ij.set_xlabel("")
            ax_ij.set_ylabel("")
            ax_ij.set_xticks([param_q[0, j], param_q[1, j]])
            ax_ij.xaxis.set_major_formatter(FuncFormatter(fmt_corner_tick))
            if i > j:
                ax_ij.set_yticks([param_q[0, i], param_q[1, i]])
                ax_ij.yaxis.set_major_formatter(FuncFormatter(fmt_corner_tick))
            else:
                ax_ij.set_yticks([])
            ax_ij.tick_params(axis="both", labelsize=CORNER_TICK_FONTSIZE)
            if i != n_dim - 1:
                ax_ij.tick_params(axis="x", labelbottom=False)
            if j != 0:
                ax_ij.tick_params(axis="y", labelleft=False)

    for i in range(n_dim):
        axes[i, i].set_title(labels[i], fontsize=CORNER_TITLE_FONTSIZE, pad=10)
        axes[i, i].axvline(medians[i], color="tab:blue", ls=":", lw=1.5, alpha=0.95)
    for i in range(1, n_dim):
        axes[i, 0].set_ylabel(labels[i], fontsize=CORNER_YLABEL_FONTSIZE, labelpad=12)

    buf = BytesIO()
    fig_tmp.savefig(buf, format="png", dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig_tmp)
    buf.seek(0)
    return plt.imread(buf)


# %%
# =============================================================================
# Load payloads
# =============================================================================
rt_led_payload, _ = load_payload("rt_wrt_led_theory_data", FILE_TAG)
schematic_payload, _ = load_payload("drift_switch_single_bound", FILE_TAG)
corner_payload, _ = load_payload("corner_5params", FILE_TAG)
rt_fix_payload, _ = load_payload("rtd_wrt_fixation_theory_data", FILE_TAG)

file_tag = rt_led_payload.get("file_tag", FILE_TAG)

# %%
# =============================================================================
# Build combined 2x2 panel
# =============================================================================
fig = plt.figure(figsize=PANEL_FIGSIZE, constrained_layout=True)
outer = fig.add_gridspec(2, 2, wspace=0.18, hspace=0.18)

# Panel A: Model schematic
ax_a = fig.add_subplot(outer[0, 0])
t_ms = schematic_payload["t_ms"]
a = schematic_payload["trajectory"]
mask_pre = schematic_payload["mask_pre"]
mask_post = schematic_payload["mask_post"]
bound_level = float(schematic_payload["bound_level"])
t_switch_ms = float(schematic_payload["t_switch_ms"])
xmin, xmax = schematic_payload["xlim_ms"]
pre_line = schematic_payload["pre_line"]
post_line = schematic_payload["post_line"]
rt_x_ms = float(schematic_payload["rt_x_ms"])
decision_x_ms = float(schematic_payload["decision_x_ms"])
labels = schematic_payload["labels"]

ax_a.plot(t_ms[mask_pre], a[mask_pre], color="blue", lw=2.8, alpha=0.42, zorder=3)
ax_a.plot(t_ms[mask_post], a[mask_post], color="red", lw=2.8, alpha=0.42, zorder=3)
ax_a.axhline(bound_level, color="0.35", lw=1.5, ls="--", zorder=2)
ax_a.axvline(0, color="0.35", lw=1.1, ls="-.", alpha=0.8, zorder=2)
ax_a.axvline(t_switch_ms, color="0.35", lw=1.3, ls=":", zorder=2)
ax_a.plot([pre_line["x0_ms"], pre_line["x1_ms"]], [pre_line["y0"], pre_line["y1"]], color="#1f77b4", lw=1.4, zorder=6)
ax_a.plot([post_line["x0_ms"], post_line["x1_ms"]], [post_line["y0"], post_line["y1"]], color="#d62728", lw=1.4, zorder=6)
ax_a.axvline(rt_x_ms, color="0.55", lw=1.2, ls=(0, (3, 3)), zorder=2)

delta_led_y = bound_level
ax_a.annotate(
    "",
    xy=(t_switch_ms, delta_led_y),
    xytext=(0, delta_led_y),
    arrowprops=dict(arrowstyle="<->", lw=1.6, color="0.55", mutation_scale=14),
    zorder=7,
)
ax_a.text(0.5 * t_switch_ms, delta_led_y + 0.02 * bound_level, labels["delta_led"], color="0.45", fontsize=ANNOT_FONTSIZE, ha="center", va="bottom")

if rt_x_ms > decision_x_ms:
    ax_a.annotate(
        "",
        xy=(rt_x_ms, delta_led_y - 0.02 * bound_level),
        xytext=(decision_x_ms, delta_led_y - 0.02 * bound_level),
        arrowprops=dict(arrowstyle="<->", lw=1.6, color="0.55", mutation_scale=14),
        zorder=7,
    )
    ax_a.text(0.5 * (decision_x_ms + rt_x_ms), delta_led_y + 0.005 * bound_level, labels["delta_m"], color="0.45", fontsize=ANNOT_FONTSIZE, ha="center", va="bottom")

if pre_line["x0_ms"] > xmin:
    ax_a.annotate(
        "",
        xy=(xmin, pre_line["y0"]),
        xytext=(pre_line["x0_ms"], pre_line["y0"]),
        arrowprops=dict(arrowstyle="<->", lw=1.6, color="0.6", mutation_scale=14),
        zorder=7,
    )
    ax_a.text(0.5 * (xmin + pre_line["x0_ms"]), pre_line["y0"] + 0.02 * bound_level, labels["delta_a"], color="0.55", fontsize=ANNOT_FONTSIZE, ha="center", va="bottom")

ax_a.text(0.5 * (pre_line["x0_ms"] + pre_line["x1_ms"]), 0.5 * (pre_line["y0"] + pre_line["y1"]) - 0.16 * bound_level, labels["pre"], color="blue", fontsize=ANNOT_FONTSIZE, ha="center")
ax_a.text(0.5 * (post_line["x0_ms"] + post_line["x1_ms"]), 0.5 * (post_line["y0"] + post_line["y1"]) - 0.16 * bound_level, labels["post"], color="red", fontsize=ANNOT_FONTSIZE, ha="center")
ax_a.text(xmax + 2, bound_level + 0.005 * bound_level, labels["theta"], color="0.25", fontsize=ANNOT_FONTSIZE + 1, ha="left", clip_on=False)
ax_a.text(rt_x_ms, bound_level * 1.10, labels["rt"], color="0.35", fontsize=ANNOT_FONTSIZE + 1, ha="center", va="bottom")

ax_a.set_xlim(xmin, xmax)
ax_a.set_ylim(0, bound_level * 1.12)
ax_a.set_xticks(schematic_payload["xticks_ms"])
ax_a.set_yticks([])
ax_a.set_xlabel(schematic_payload.get("xlabel", "Time from LED onset (ms)"), fontsize=AXIS_LABEL_FONTSIZE)
ax_a.spines["top"].set_visible(False)
ax_a.spines["right"].set_visible(False)
ax_a.spines["left"].set_bounds(0, bound_level)
ax_a.spines["left"].set_linewidth(2.0)
ax_a.spines["bottom"].set_linewidth(2.0)
ax_a.tick_params(axis="x", labelsize=AXIS_TICK_FONTSIZE, width=2.0, length=7)

# Panel B: Corner plot image (from payload samples)
ax_b = fig.add_subplot(outer[0, 1])
corner_img = render_corner_image(corner_payload)
ax_b.imshow(corner_img)
ax_b.set_aspect("auto")
ax_b.axis("off")

# Panel C: RT wrt LED
ax_c = fig.add_subplot(outer[1, 0])
ax_c.plot(rt_led_payload["data_x_ms"], rt_led_payload["data_hist_on_scaled"], color="r", alpha=0.4, lw=2.0)
ax_c.plot(rt_led_payload["data_x_ms"], rt_led_payload["data_hist_off_scaled"], color="b", alpha=0.4, lw=2.0)
ax_c.plot(rt_led_payload["theory_x_ms"], rt_led_payload["rtd_theory_on_wrt_led"], color="r", alpha=1.0, lw=2.4)
ax_c.plot(rt_led_payload["theory_x_ms"], rt_led_payload["rtd_theory_off_wrt_led"], color="b", alpha=1.0, lw=2.4)
ax_c.axvline(0, color="0.2", ls="--", lw=1.2, alpha=0.7)
ax_c.axvline(rt_led_payload["del_m_plus_del_LED_ms"], color="0.2", ls=":", lw=1.2, alpha=0.7)
ax_c.set_xlim(rt_led_payload["xlim_ms"])
ax_c.set_xticks(rt_led_payload["xticks_ms"])
ax_c.set_yticks([])
ax_c.set_xlabel(rt_led_payload.get("xlabel", "RT wrt LED onset (ms)"), fontsize=AXIS_LABEL_FONTSIZE)
ax_c.set_ylabel(rt_led_payload.get("ylabel", "Abort Rate (Hz)"), fontsize=AXIS_LABEL_FONTSIZE)
ax_c.spines["top"].set_visible(False)
ax_c.spines["right"].set_visible(False)
ax_c.spines["left"].set_linewidth(2.0)
ax_c.spines["bottom"].set_linewidth(2.0)
ax_c.tick_params(axis="x", labelsize=AXIS_TICK_FONTSIZE, width=2.0, length=7)
ax_c.tick_params(axis="y", width=0, length=0)

# Panel D: RT wrt fixation
ax_d = fig.add_subplot(outer[1, 1])
ax_d.plot(rt_fix_payload["data_x_ms"], rt_fix_payload["data_hist_on_scaled"], color="r", alpha=0.4, lw=2.0)
ax_d.plot(rt_fix_payload["data_x_ms"], rt_fix_payload["data_hist_off_scaled"], color="b", alpha=0.4, lw=2.0)
ax_d.plot(rt_fix_payload["theory_x_ms"], rt_fix_payload["rtd_theory_on_wrt_fix"], color="r", alpha=1.0, lw=2.4)
ax_d.plot(rt_fix_payload["theory_x_ms"], rt_fix_payload["rtd_theory_off_wrt_fix"], color="b", alpha=1.0, lw=2.4)
ax_d.set_xlim(rt_fix_payload["xlim_ms"])
ax_d.set_xticks(rt_fix_payload["xticks_ms"])
ax_d.set_yticks([])
ax_d.set_xlabel(rt_fix_payload.get("xlabel", "RT wrt fixation (ms)"), fontsize=AXIS_LABEL_FONTSIZE)
ax_d.set_ylabel(rt_fix_payload.get("ylabel", "Abort Rate (Hz)"), fontsize=AXIS_LABEL_FONTSIZE)
ax_d.spines["top"].set_visible(False)
ax_d.spines["right"].set_visible(False)
ax_d.spines["left"].set_linewidth(2.0)
ax_d.spines["bottom"].set_linewidth(2.0)
ax_d.tick_params(axis="x", labelsize=AXIS_TICK_FONTSIZE, width=2.0, length=7)
ax_d.tick_params(axis="y", width=0, length=0)

# %%
# =============================================================================
# Save
# =============================================================================
panel_pdf = DATA_DIR / f"fct_panel_march_26_{file_tag}.pdf"
panel_png = DATA_DIR / f"fct_panel_march_26_{file_tag}.png"
fig.savefig(panel_pdf, bbox_inches="tight")
fig.savefig(panel_png, dpi=SAVE_DPI, bbox_inches="tight")
print(f"Saved panel PDF: {panel_pdf}")
print(f"Saved panel PNG: {panel_png}")

if SHOW_PLOT:
    plt.show()
else:
    plt.close(fig)


