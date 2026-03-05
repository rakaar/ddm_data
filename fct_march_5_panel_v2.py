"""
Publication-style composite panel:
- Left: 2x2 square panels (schematic, RT wrt LED + inset, RT wrt fixation, RT wrt LED zoomed)
- Right: one large corner panel aligned to full left-panel height
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
SAVE_DPI = 500

FIGSIZE = (21, 11.2)
LABEL_FS = 14
TICK_FS = 11
INSET_LABEL_FS = 9
INSET_TICK_FS = 7

# Corner plot typography tuned for readability without overlap.
CORNER_TICK_FS = 25
CORNER_TITLE_FS = 24
CORNER_YLABEL_FS = 28


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
        return pickle.load(f)


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
    fig_tmp.set_size_inches(19.0, 19.0)

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
            ax_ij.tick_params(axis="both", labelsize=CORNER_TICK_FS)
            ax_ij.tick_params(axis="x", pad=2)
            ax_ij.tick_params(axis="y", pad=2)
            if i != n_dim - 1:
                ax_ij.tick_params(axis="x", labelbottom=False)
            if j != 0:
                ax_ij.tick_params(axis="y", labelleft=False)

    for i in range(n_dim):
        axes[i, i].set_title(labels[i], fontsize=CORNER_TITLE_FS, pad=10)
        axes[i, i].axvline(medians[i], color="tab:blue", ls=":", lw=1.6, alpha=0.95)
    for i in range(1, n_dim):
        axes[i, 0].set_ylabel(labels[i], fontsize=CORNER_YLABEL_FS, labelpad=28)

    buf = BytesIO()
    fig_tmp.savefig(buf, format="png", dpi=520, bbox_inches="tight", facecolor="white")
    plt.close(fig_tmp)
    buf.seek(0)
    return plt.imread(buf)


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(2.2)
    ax.spines["left"].set_linewidth(2.2)
    ax.tick_params(axis="x", labelsize=TICK_FS, width=1.8, length=6)
    ax.tick_params(axis="y", labelsize=TICK_FS, width=1.8, length=6)


# %%
# =============================================================================
# Load payloads
# =============================================================================
schematic_payload = load_payload("drift_switch_single_bound", FILE_TAG)
rt_led_payload = load_payload("rt_wrt_led_theory_data", FILE_TAG)
rt_fix_payload = load_payload("rtd_wrt_fixation_theory_data", FILE_TAG)
rt_led_zoom_payload = load_payload("rt_wrt_led_theory_data_zoom_5ms", FILE_TAG)
tled_stim_payload = load_payload("t_led_t_stim_distributions", FILE_TAG)
corner_payload = load_payload("corner_5params", FILE_TAG)

file_tag = rt_led_payload.get("file_tag", FILE_TAG)


# =============================================================================
# Figure layout + all panels + save (single cell-friendly block)
# =============================================================================
plt.close("all")
fig = plt.figure(figsize=FIGSIZE)
outer = fig.add_gridspec(1, 2, width_ratios=[1.50, 1.00], wspace=0.02)

left = outer[0, 0].subgridspec(2, 2, wspace=0.08, hspace=0.16)
ax_schematic = fig.add_subplot(left[0, 0])
ax_rt_led = fig.add_subplot(left[0, 1])
ax_rt_fix = fig.add_subplot(left[1, 0])
ax_rt_led_zoom = fig.add_subplot(left[1, 1])

# Keep all four left panels square and same size.
for ax in [ax_schematic, ax_rt_led, ax_rt_fix, ax_rt_led_zoom]:
    ax.set_box_aspect(1)

ax_corner = fig.add_subplot(outer[0, 1])


# =============================================================================
# Panel 1: Model schematic
# =============================================================================
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
decision_x_ms = float(schematic_payload.get("decision_x_ms", rt_x_ms - 12.0))
labels = schematic_payload["labels"]

ax_schematic.plot(t_ms[mask_pre], a[mask_pre], color="blue", lw=2.6, alpha=0.42, zorder=3)
ax_schematic.plot(t_ms[mask_post], a[mask_post], color="red", lw=2.6, alpha=0.42, zorder=3)
ax_schematic.axhline(bound_level, color="0.35", lw=1.4, ls="--", zorder=2)
ax_schematic.axvline(0, color="0.35", lw=1.0, ls="-.", alpha=0.8, zorder=2)
ax_schematic.axvline(t_switch_ms, color="0.35", lw=1.2, ls=":", zorder=2)
ax_schematic.plot([pre_line["x0_ms"], pre_line["x1_ms"]], [pre_line["y0"], pre_line["y1"]], color="#1f77b4", lw=1.4, zorder=6)
ax_schematic.plot([post_line["x0_ms"], post_line["x1_ms"]], [post_line["y0"], post_line["y1"]], color="#d62728", lw=1.4, zorder=6)
ax_schematic.axvline(rt_x_ms, color="0.55", lw=1.1, ls=(0, (3, 3)), zorder=2)

# Delay annotations (restore full schematic semantics).
delay_y = bound_level * 1.00
if pre_line["x0_ms"] > xmin + 1e-9:
    ax_schematic.annotate(
        "",
        xy=(xmin, pre_line["y0"]),
        xytext=(pre_line["x0_ms"], pre_line["y0"]),
        arrowprops=dict(arrowstyle="<->", lw=1.6, color="0.6", mutation_scale=14),
        zorder=7,
    )
    ax_schematic.text(
        0.5 * (xmin + pre_line["x0_ms"]),
        pre_line["y0"] + 0.018 * bound_level,
        labels["delta_a"],
        color="0.55",
        fontsize=12,
        ha="center",
        va="bottom",
    )

ax_schematic.annotate(
    "",
    xy=(t_switch_ms, delay_y),
    xytext=(0, delay_y),
    arrowprops=dict(arrowstyle="<->", lw=1.7, color="0.55", mutation_scale=14),
    zorder=7,
)
ax_schematic.text(
    0.5 * t_switch_ms,
    delay_y + 0.015 * bound_level,
    labels["delta_led"],
    color="0.45",
    fontsize=12,
    ha="center",
    va="bottom",
)

if rt_x_ms > decision_x_ms + 0.3:
    ax_schematic.annotate(
        "",
        xy=(rt_x_ms, delay_y - 0.02 * bound_level),
        xytext=(decision_x_ms, delay_y - 0.02 * bound_level),
        arrowprops=dict(arrowstyle="<->", lw=1.7, color="0.55", mutation_scale=14),
        zorder=7,
    )
    ax_schematic.text(
        0.5 * (decision_x_ms + rt_x_ms),
        delay_y + 0.003 * bound_level,
        labels["delta_m"],
        color="0.45",
        fontsize=12,
        ha="center",
        va="bottom",
    )

ax_schematic.text(0.5 * (pre_line["x0_ms"] + pre_line["x1_ms"]), 0.5 * (pre_line["y0"] + pre_line["y1"]) - 0.15 * bound_level, labels["pre"], color="blue", fontsize=12, ha="center")
ax_schematic.text(0.5 * (post_line["x0_ms"] + post_line["x1_ms"]), 0.5 * (post_line["y0"] + post_line["y1"]) - 0.15 * bound_level, labels["post"], color="red", fontsize=12, ha="center")
ax_schematic.text(xmax + 1.5, bound_level + 0.01 * bound_level, labels["theta"], color="0.25", fontsize=13, ha="left", clip_on=False)
ax_schematic.text(rt_x_ms, bound_level * 1.085, labels["rt"], color="0.35", fontsize=13, ha="center", va="bottom")

ax_schematic.set_xlim(xmin, xmax)
ax_schematic.set_ylim(0, bound_level * 1.10)
ax_schematic.set_xticks(schematic_payload["xticks_ms"])
ax_schematic.set_yticks([])
ax_schematic.set_xlabel(schematic_payload.get("xlabel", "Time from LED onset (ms)"), fontsize=LABEL_FS)
style_axes(ax_schematic)
ax_schematic.tick_params(axis="y", width=0, length=0)


# =============================================================================
# Panel 2: RT wrt LED (main) + inset t_stim/t_LED
# =============================================================================
ax_rt_led.plot(rt_led_payload["data_x_ms"], rt_led_payload["data_hist_on_scaled"], color="r", alpha=0.4, lw=2.0)
ax_rt_led.plot(rt_led_payload["data_x_ms"], rt_led_payload["data_hist_off_scaled"], color="b", alpha=0.4, lw=2.0)
ax_rt_led.plot(rt_led_payload["theory_x_ms"], rt_led_payload["rtd_theory_on_wrt_led"], color="r", alpha=1.0, lw=2.4)
ax_rt_led.plot(rt_led_payload["theory_x_ms"], rt_led_payload["rtd_theory_off_wrt_led"], color="b", alpha=1.0, lw=2.4)
ax_rt_led.axvline(0, color="0.2", ls="--", lw=1.1, alpha=0.7)
ax_rt_led.axvline(rt_led_payload["del_m_plus_del_LED_ms"], color="0.2", ls=":", lw=1.1, alpha=0.7)
ax_rt_led.set_xlim(rt_led_payload["xlim_ms"])
ax_rt_led.set_xticks(rt_led_payload["xticks_ms"])
ax_rt_led.set_yticks([])
ax_rt_led.set_xlabel(rt_led_payload.get("xlabel", "RT wrt LED onset (ms)"), fontsize=LABEL_FS)
style_axes(ax_rt_led)
ax_rt_led.tick_params(axis="y", width=0, length=0)

# Inset: t_stim / t_LED
ax_inset = ax_rt_led.inset_axes([0.06, 0.63, 0.31, 0.31], zorder=15)
ax_inset.plot(tled_stim_payload["bin_centers_s"], tled_stim_payload["t_led_hist_density"], color="tab:blue", lw=1.8, alpha=0.95)
ax_inset.plot(tled_stim_payload["bin_centers_s"], tled_stim_payload["t_stim_hist_density"], color="tab:red", lw=1.8, alpha=0.95)
xlim_s = tled_stim_payload.get("xlim_s", (0.0, 1.0))
ax_inset.set_xlim(xlim_s)
ax_inset.set_ylim(0, 3)
ax_inset.set_yticks([])
ax_inset.set_xlabel("Time (s)", fontsize=INSET_LABEL_FS)
ax_inset.tick_params(axis="x", labelsize=INSET_TICK_FS, width=1.0, length=4)
ax_inset.tick_params(axis="y", width=0, length=0)
ax_inset.spines["top"].set_visible(False)
ax_inset.spines["right"].set_visible(False)
ax_inset.spines["left"].set_linewidth(1.2)
ax_inset.spines["bottom"].set_linewidth(1.2)
ax_inset.set_title(r"$t_{stim}$ and $t_{LED}$", fontsize=INSET_LABEL_FS + 1, pad=2)
ax_inset.set_facecolor("white")
ax_inset.patch.set_alpha(0.95)


# =============================================================================
# Panel 3: RT wrt fixation
# =============================================================================
ax_rt_fix.plot(rt_fix_payload["data_x_ms"], rt_fix_payload["data_hist_on_scaled"], color="r", alpha=0.4, lw=2.0)
ax_rt_fix.plot(rt_fix_payload["data_x_ms"], rt_fix_payload["data_hist_off_scaled"], color="b", alpha=0.4, lw=2.0)
ax_rt_fix.plot(rt_fix_payload["theory_x_ms"], rt_fix_payload["rtd_theory_on_wrt_fix"], color="r", alpha=1.0, lw=2.4)
ax_rt_fix.plot(rt_fix_payload["theory_x_ms"], rt_fix_payload["rtd_theory_off_wrt_fix"], color="b", alpha=1.0, lw=2.4)
ax_rt_fix.set_xlim(rt_fix_payload["xlim_ms"])
ax_rt_fix.set_xticks(rt_fix_payload["xticks_ms"])
ax_rt_fix.set_yticks([])
ax_rt_fix.set_xlabel(rt_fix_payload.get("xlabel", "RT wrt fixation (ms)"), fontsize=LABEL_FS)
style_axes(ax_rt_fix)
ax_rt_fix.tick_params(axis="y", width=0, length=0)


# =============================================================================
# Panel 4: RT wrt LED zoomed
# =============================================================================
ax_rt_led_zoom.plot(rt_led_zoom_payload["data_x_ms"], rt_led_zoom_payload["data_hist_on_scaled"], color="r", alpha=0.4, lw=2.0)
ax_rt_led_zoom.plot(rt_led_zoom_payload["data_x_ms"], rt_led_zoom_payload["data_hist_off_scaled"], color="b", alpha=0.4, lw=2.0)
ax_rt_led_zoom.plot(rt_led_zoom_payload["theory_x_ms"], rt_led_zoom_payload["rtd_theory_on_wrt_led"], color="r", alpha=1.0, lw=2.4)
ax_rt_led_zoom.plot(rt_led_zoom_payload["theory_x_ms"], rt_led_zoom_payload["rtd_theory_off_wrt_led"], color="b", alpha=1.0, lw=2.4)
ax_rt_led_zoom.axvline(0, color="0.2", ls="--", lw=1.1, alpha=0.7)
ax_rt_led_zoom.set_xlim(rt_led_zoom_payload["xlim_ms"])
ax_rt_led_zoom.set_xticks([-100, 0, 100])
ax_rt_led_zoom.set_yticks([])
ax_rt_led_zoom.set_xlabel(rt_led_zoom_payload.get("xlabel", "RT wrt LED onset (ms)"), fontsize=LABEL_FS)
style_axes(ax_rt_led_zoom)
ax_rt_led_zoom.tick_params(axis="y", width=0, length=0)


# =============================================================================
# Right panel: corner plot image (full height, aligned with left block)
# =============================================================================
corner_img = render_corner_image(corner_payload)
ax_corner.imshow(corner_img)
ax_corner.set_aspect("auto")
ax_corner.axis("off")


# =============================================================================
# Save
# =============================================================================
panel_pdf = DATA_DIR / f"fct_march_5_panel_v2_{file_tag}.pdf"
panel_png = DATA_DIR / f"fct_march_5_panel_v2_{file_tag}.png"
fig.savefig(panel_pdf, bbox_inches="tight")
fig.savefig(panel_png, dpi=SAVE_DPI, bbox_inches="tight")
print(f"Saved panel PDF: {panel_pdf}")
print(f"Saved panel PNG: {panel_png}")

if SHOW_PLOT:
    plt.show()
else:
    plt.close(fig)
