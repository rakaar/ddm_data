"""
Publication-style composite panel:
- Top-left: timing schematic + stacked timing distributions
- Bottom-left: 2x2 square panels (schematic, RT wrt fixation, RT wrt LED, RT wrt LED zoomed)
- Bottom-right: one large corner panel aligned to the lower 2x2 block
- Bottom row: ABL-wise delay-fit overlay + delay bar plot
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
ABL_DELAY_DIAG_DIR = ROOT / "fitting_aborts" / "norm_only_led_off_from_loaded_proactive_truncate_NOT_censor_ABL_delay" / "diagnostics"
FILE_TAG = "all_animals"
ABL_DELAY_TRUNC_TAG = "115ms"

SHOW_PLOT = True
SAVE_DPI = 500
SHOW_TIMING_HEADER = True
SHOW_TIMING_DISTRIBUTIONS = True

FIGSIZE = (24, 17.5)
FIG_BOUNDS = dict(left=0.05, right=0.95, top=0.95, bottom=0.08)
# Choose width ratios so the bottom-right corner slot is close to square
# after accounting for the top timing-header row height.
OUTER_WIDTH_RATIOS = (0.70, 1.0)
OUTER_WSPACE = 0.0
OUTER_HSPACE = 0.22
TIMING_HEADER_HEIGHT_FRAC = 0.34
BOTTOM_ROW_HEIGHT_FRAC = 0.38
TIMING_HEADER_WIDTH_RATIOS = (3, 1)
TIMING_HEADER_WSPACE = 0.12
TIMING_DISTS_HSPACE = 0.42
LEFT_GRID_WSPACE = 0.08
LEFT_GRID_HSPACE = 0.32
BOTTOM_ROW_LAYOUT_RATIOS = (0.13, 0.62, 0.18, 0.72, 0.69)
MAIN_HSPACE = 0.34
BOTTOM_ROW_WSPACE = 0.08

LABEL_FS = 25
TICK_FS = 24
BOTTOM_LABEL_FS = LABEL_FS
BOTTOM_TICK_FS = TICK_FS
BOTTOM_AXIS_LW = 2.2
BOTTOM_TICK_LW = 1.8
BOTTOM_TICK_LEN = 6
TIMING_LABEL_FS = 18
TIMING_DIST_LABEL_FS = LABEL_FS
TIMING_DIST_TICK_FS = TICK_FS
TIMING_LINE_COLOR = "black"
TIMING_LINE_WIDTH = 2.0
TIMING_DIST_FILL_COLOR = "0.75"
TIMING_DIST_LINE_COLOR = "0.35"
SCHEMATIC_DELAY_LABEL_FS = 12
DELTA_LED_VISUAL_MIN_SPAN_MS = 18.0
DELTA_M_VISUAL_MIN_SPAN_MS = 18.0
PANEL_XLABEL_PAD = 3
PANEL_YLABEL_PAD = 2
BOTTOM_XLABEL_PAD = 3
BOTTOM_YLABEL_PAD = 2
TIMING_DIST_XLABEL_PAD = 2
TIMING_DIST_YLABEL_PAD = 2

# Corner plot typography tuned for readability without overlap.
CORNER_TICK_FS = 30
CORNER_TITLE_FS = 34
CORNER_YLABEL_FS = 32
CORNER_YLABEL_PAD = 40


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


def load_pickle_payload(payload_path):
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
        axes[i, 0].set_ylabel(labels[i], fontsize=CORNER_YLABEL_FS, labelpad=CORNER_YLABEL_PAD)

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


def compute_density(values, bins=80, data_range=None):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("Cannot compute density for empty values.")

    if data_range is None:
        lo = max(0.0, float(np.percentile(arr, 0.2)))
        hi = float(np.percentile(arr, 99.5))
    else:
        lo, hi = data_range

    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi) or hi <= lo:
        hi = float(np.max(arr))
    if hi <= lo:
        hi = lo + 1e-3

    edges = np.linspace(lo, hi, bins + 1)
    hist, _ = np.histogram(arr, bins=edges, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist, (lo, hi)


def expanded_interval(x0, x1, min_span):
    center = 0.5 * (x0 + x1)
    half_span = max(0.5 * abs(x1 - x0), 0.5 * min_span)
    return center - half_span, center + half_span


def plot_timing_header(ax, tled_stim_payload):
    t_led = np.asarray(tled_stim_payload["t_led_values_s"], dtype=float)
    t_stim = np.asarray(tled_stim_payload["t_stim_values_s"], dtype=float)
    t_stim_wrt_led = t_stim - t_led
    valid = np.isfinite(t_led) & np.isfinite(t_stim_wrt_led)
    t_led = t_led[valid]
    t_stim_wrt_led = t_stim_wrt_led[valid]

    t_led_ref = float(np.percentile(t_led, 60))
    t_stim_ref = float(np.percentile(t_stim_wrt_led, 60))
    total_ref = max(float(np.percentile(t_led + t_stim_wrt_led, 95)), t_led_ref + t_stim_ref)

    x_fix = 0.22
    x_end = 0.95
    span = x_end - x_fix
    scale = span / total_ref if total_ref > 0 else 1.0
    led_gap = max(t_led_ref * scale, 0.24)
    stim_gap = max(t_stim_ref * scale, 0.24)
    x_led = min(x_fix + led_gap, 0.58)
    x_stim = min(x_led + stim_gap, 0.82)

    y_fix = 2.15
    y_led = 1.20
    y_stim = 0.25
    step_h = 0.52

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.05, 2.95)
    ax.axis("off")

    def draw_step(y0, x_step, x_start):
        ax.plot([x_start, x_step], [y0, y0], color=TIMING_LINE_COLOR, lw=TIMING_LINE_WIDTH, solid_capstyle="round")
        ax.plot([x_step, x_step], [y0, y0 + step_h], color=TIMING_LINE_COLOR, lw=TIMING_LINE_WIDTH, solid_capstyle="round")
        ax.plot([x_step, x_end], [y0 + step_h, y0 + step_h], color=TIMING_LINE_COLOR, lw=TIMING_LINE_WIDTH, solid_capstyle="round")

    draw_step(y_fix, x_fix, 0.08)
    draw_step(y_led, x_led, 0.11)
    draw_step(y_stim, x_stim, 0.13)

    ax.text(0.00, y_fix + 0.03, "Fixation", color=TIMING_LINE_COLOR, fontsize=TIMING_LABEL_FS, ha="left", va="bottom")
    ax.text(0.03, y_led + 0.03, "LED", color=TIMING_LINE_COLOR, fontsize=TIMING_LABEL_FS, ha="left", va="bottom")
    ax.text(0.00, y_stim + 0.03, "Stimulus", color=TIMING_LINE_COLOR, fontsize=TIMING_LABEL_FS, ha="left", va="bottom")

    arrow_y_led = y_led + 0.20
    arrow_y_stim = y_stim + 0.36
    ax.annotate("", xy=(x_led, arrow_y_led), xytext=(x_fix, arrow_y_led), arrowprops=dict(arrowstyle="<->", lw=1.6, color=TIMING_LINE_COLOR, mutation_scale=16))
    ax.annotate("", xy=(x_stim, arrow_y_stim), xytext=(x_led, arrow_y_stim), arrowprops=dict(arrowstyle="<->", lw=1.6, color=TIMING_LINE_COLOR, mutation_scale=16))
    ax.text(0.5 * (x_fix + x_led), arrow_y_led + 0.12, r"$t_{LED}$", color=TIMING_LINE_COLOR, fontsize=TIMING_LABEL_FS - 1, ha="center", va="center")
    ax.text(0.5 * (x_led + x_stim), arrow_y_stim + 0.12, r"$t_{stim}$", color=TIMING_LINE_COLOR, fontsize=TIMING_LABEL_FS - 1, ha="center", va="center")


def style_timing_distribution_axis(ax, centers, density, xlim, label, show_xlabel=False, show_left_spine=False):
    ax.step(centers, density, where="mid", color=TIMING_DIST_LINE_COLOR, lw=1.4)
    ax.fill_between(centers, 0, density, step="mid", color=TIMING_DIST_FILL_COLOR, alpha=0.9)
    ax.set_xlim(*xlim)
    ax.set_ylim(0, max(float(np.max(density)) * 1.18, 1e-6))
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(show_left_spine)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="x", labelsize=TIMING_DIST_TICK_FS, width=1.0, length=3)
    ax.tick_params(axis="y", width=0, length=0)
    ax.set_ylabel(label, fontsize=TIMING_DIST_LABEL_FS, rotation=90, labelpad=TIMING_DIST_YLABEL_PAD)
    if not show_xlabel:
        ax.tick_params(axis="x", labelbottom=False)
    else:
        ax.set_xlabel("Time (ms)", fontsize=TIMING_DIST_LABEL_FS, labelpad=TIMING_DIST_XLABEL_PAD)


def plot_timing_distributions(ax_t_led, ax_t_stim, tled_stim_payload):
    t_led = 1e3 * np.asarray(tled_stim_payload["t_led_values_s"], dtype=float)
    t_stim = 1e3 * np.asarray(tled_stim_payload["t_stim_values_s"], dtype=float)
    t_stim_wrt_led = t_stim - t_led
    valid_led = np.isfinite(t_led)
    valid_stim = np.isfinite(t_stim_wrt_led)

    led_centers, led_density, led_xlim = compute_density(t_led[valid_led], bins=90, data_range=(0.0, float(np.percentile(t_led[valid_led], 99.5))))
    stim_hi = float(np.percentile(t_stim_wrt_led[valid_stim], 99.5))
    stim_centers, stim_density, stim_xlim = compute_density(t_stim_wrt_led[valid_stim], bins=70, data_range=(0.0, stim_hi))

    style_timing_distribution_axis(ax_t_led, led_centers, led_density, led_xlim, r"$P(t_{LED})$", show_xlabel=False, show_left_spine=False)
    style_timing_distribution_axis(ax_t_stim, stim_centers, stim_density, stim_xlim, r"$P(t_{stim})$", show_xlabel=True, show_left_spine=True)


def plot_abl_delay_overlay(ax, payload):
    t_ms = np.asarray(payload["t_pts_truncated_ms"], dtype=float)
    bin_centers_ms = np.asarray(payload["data_hist_centers_truncated_ms"], dtype=float)
    supported_abls = [int(abl) for abl in payload["config"]["supported_ABL_values"]]
    abl_colors = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}

    peak_density = 0.0
    for abl in supported_abls:
        curves = payload["abl_curves"][abl]
        data_density = np.asarray(curves["data_density_truncated"], dtype=float)
        theory_density = np.asarray(curves["theory_density_truncated_norm"], dtype=float)
        color = abl_colors.get(abl, "0.35")

        ax.step(bin_centers_ms, data_density, where="mid", lw=1.2, color=color, alpha=0.65)
        ax.plot(t_ms, theory_density, lw=2.5, color=color)
        peak_density = max(peak_density, float(np.max(data_density)), float(np.max(theory_density)))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(BOTTOM_AXIS_LW)
    ax.spines["bottom"].set_linewidth(BOTTOM_AXIS_LW)
    ax.set_xlim(0.0, float(payload["config"]["truncate_rt_wrt_stim_ms"]))
    ax.set_xlabel(payload["config"]["xlabel"], fontsize=BOTTOM_LABEL_FS, labelpad=BOTTOM_XLABEL_PAD)
    ax.set_ylabel(payload["config"]["ylabel"], fontsize=BOTTOM_LABEL_FS, labelpad=BOTTOM_YLABEL_PAD)
    ax.margins(x=0.0)
    ax.tick_params(axis="both", labelsize=BOTTOM_TICK_FS, direction="out", length=BOTTOM_TICK_LEN, width=BOTTOM_TICK_LW)
    ax.set_xticks([0, 100])
    ax.set_yticks([0, 40])
    ax.set_ylim(0.0, max(peak_density * 1.05, 40.0) if peak_density > 0 else 40.0)


def plot_delay_bar(ax, payload):
    labels = payload["labels"]
    values_ms = np.asarray(payload["values_ms"], dtype=float)
    x = np.arange(len(labels))

    ax.bar(x, values_ms, width=0.68, color="#808080", edgecolor="black", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_position(("data", 0.0))
    ax.spines["left"].set_linewidth(BOTTOM_AXIS_LW)
    ax.spines["bottom"].set_linewidth(BOTTOM_AXIS_LW)
    ax.set_ylabel("Delay (ms)", fontsize=BOTTOM_LABEL_FS, labelpad=BOTTOM_YLABEL_PAD)
    ax.set_xticks(x)
    ax.set_xticklabels(["", *labels[1:]], fontsize=BOTTOM_TICK_FS, rotation=0)
    ax.tick_params(axis="y", labelsize=BOTTOM_TICK_FS, direction="out", length=BOTTOM_TICK_LEN, width=BOTTOM_TICK_LW)
    ax.tick_params(axis="x", direction="out", length=0, width=BOTTOM_TICK_LW)
    ax.set_xlim(-0.55, len(labels) - 0.45)
    ax.set_axisbelow(True)

    ymin = float(np.min(values_ms))
    ymax = float(np.max(values_ms))
    ymin_plot = ymin * 1.18 if ymin < 0 else 0.0
    ymax_plot = ymax * 1.18 if ymax > 0 else 1.0
    ax.set_ylim(ymin_plot, ymax_plot)
    ax.set_yticks([-60, 0, 60])

    first_label_y = 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.text(x[0], first_label_y, labels[0], ha="center", va="bottom", fontsize=BOTTOM_TICK_FS)


# %%
# =============================================================================
# Load payloads
# =============================================================================
abl_delay_overlay_payload_path = (
    ABL_DELAY_DIAG_DIR
    / f"diag_norm_tied_batch_LED7_aggregate_ledoff_truncated_{ABL_DELAY_TRUNC_TAG}_rtwrtstim_by_ABL_publication_truncate_not_censor_ABL_delay.pkl"
)
delay_bar_payload_path = (
    ABL_DELAY_DIAG_DIR
    / "diag_norm_tied_batch_LED7_aggregate_ledoff_delay_bars_truncate_not_censor_ABL_delay.pkl"
)

schematic_payload = load_payload("drift_switch_single_bound", FILE_TAG)
rt_led_payload = load_payload("rt_wrt_led_theory_data", FILE_TAG)
rt_fix_payload = load_payload("rtd_wrt_fixation_theory_data", FILE_TAG)
rt_led_zoom_payload = load_payload("rt_wrt_led_theory_data_zoom_5ms", FILE_TAG)
tled_stim_payload = load_payload("t_led_t_stim_distributions", FILE_TAG) if (SHOW_TIMING_HEADER or SHOW_TIMING_DISTRIBUTIONS) else None
corner_payload = load_payload("corner_5params", FILE_TAG)
abl_delay_overlay_payload = load_pickle_payload(abl_delay_overlay_payload_path)
delay_bar_payload = load_pickle_payload(delay_bar_payload_path)

file_tag = rt_led_payload.get("file_tag", FILE_TAG)


# =============================================================================
# Figure layout + all panels + save (single cell-friendly block)
# =============================================================================
plt.close("all")
fig = plt.figure(figsize=FIGSIZE)
show_top_strip = SHOW_TIMING_HEADER or SHOW_TIMING_DISTRIBUTIONS
main = fig.add_gridspec(
    2,
    1,
    height_ratios=[1.0, BOTTOM_ROW_HEIGHT_FRAC],
    hspace=MAIN_HSPACE,
    **FIG_BOUNDS,
)

if show_top_strip:
    outer = main[0, 0].subgridspec(
        2,
        2,
        width_ratios=OUTER_WIDTH_RATIOS,
        height_ratios=[TIMING_HEADER_HEIGHT_FRAC, 1.0],
        wspace=OUTER_WSPACE,
        hspace=OUTER_HSPACE,
    )
    header = outer[0, 0].subgridspec(1, 2, width_ratios=TIMING_HEADER_WIDTH_RATIOS, wspace=TIMING_HEADER_WSPACE)
    header_dists = header[0, 1].subgridspec(2, 1, hspace=TIMING_DISTS_HSPACE)
    left = outer[1, 0].subgridspec(2, 2, wspace=LEFT_GRID_WSPACE, hspace=LEFT_GRID_HSPACE)

    ax_timing_header = fig.add_subplot(header[0, 0])
    ax_timing_dist_led = fig.add_subplot(header_dists[0, 0])
    ax_timing_dist_stim = fig.add_subplot(header_dists[1, 0])
    ax_corner = fig.add_subplot(outer[1, 1])
else:
    outer = main[0, 0].subgridspec(1, 2, width_ratios=OUTER_WIDTH_RATIOS, wspace=OUTER_WSPACE)
    left = outer[0, 0].subgridspec(2, 2, wspace=LEFT_GRID_WSPACE, hspace=LEFT_GRID_HSPACE)
    ax_corner = fig.add_subplot(outer[0, 1])
    ax_timing_header = None
    ax_timing_dist_led = None
    ax_timing_dist_stim = None

bottom = main[1, 0].subgridspec(1, 5, width_ratios=BOTTOM_ROW_LAYOUT_RATIOS, wspace=BOTTOM_ROW_WSPACE)
ax_abl_delay = fig.add_subplot(bottom[0, 1])
ax_delay_bar = fig.add_subplot(bottom[0, 3])

ax_schematic = fig.add_subplot(left[0, 0])
ax_rt_fix = fig.add_subplot(left[0, 1])
ax_rt_led = fig.add_subplot(left[1, 0])
ax_rt_led_zoom = fig.add_subplot(left[1, 1])

for ax in [ax_schematic, ax_rt_led, ax_rt_fix, ax_rt_led_zoom]:
    ax.set_box_aspect(1)


# =============================================================================
# Top strip: task timing schematic + timing distributions
# =============================================================================
if SHOW_TIMING_HEADER and ax_timing_header is not None:
    plot_timing_header(ax_timing_header, tled_stim_payload)
elif ax_timing_header is not None:
    ax_timing_header.axis("off")

if SHOW_TIMING_DISTRIBUTIONS and ax_timing_dist_led is not None and ax_timing_dist_stim is not None:
    plot_timing_distributions(ax_timing_dist_led, ax_timing_dist_stim, tled_stim_payload)
else:
    if ax_timing_dist_led is not None:
        ax_timing_dist_led.axis("off")
    if ax_timing_dist_stim is not None:
        ax_timing_dist_stim.axis("off")


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
    xy=(expanded_interval(0.0, t_switch_ms, DELTA_LED_VISUAL_MIN_SPAN_MS)[1], delay_y),
    xytext=(expanded_interval(0.0, t_switch_ms, DELTA_LED_VISUAL_MIN_SPAN_MS)[0], delay_y),
    arrowprops=dict(arrowstyle="<->", lw=1.7, color="0.55", mutation_scale=14),
    zorder=7,
)
ax_schematic.text(
    0.5 * t_switch_ms,
    delay_y + 0.015 * bound_level,
    labels["delta_led"],
    color="0.45",
    fontsize=SCHEMATIC_DELAY_LABEL_FS,
    ha="center",
    va="bottom",
)

if rt_x_ms > decision_x_ms + 0.3:
    delta_m_x0, delta_m_x1 = expanded_interval(decision_x_ms, rt_x_ms, DELTA_M_VISUAL_MIN_SPAN_MS)
    ax_schematic.annotate(
        "",
        xy=(delta_m_x1, delay_y - 0.02 * bound_level),
        xytext=(delta_m_x0, delay_y - 0.02 * bound_level),
        arrowprops=dict(arrowstyle="<->", lw=1.7, color="0.55", mutation_scale=14),
        zorder=7,
    )
    ax_schematic.text(
        0.5 * (decision_x_ms + rt_x_ms),
        delay_y + 0.003 * bound_level,
        labels["delta_m"],
        color="0.45",
        fontsize=SCHEMATIC_DELAY_LABEL_FS,
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
ax_schematic.set_xlabel(schematic_payload.get("xlabel", "Time from LED onset (ms)"), fontsize=LABEL_FS, labelpad=PANEL_XLABEL_PAD)
style_axes(ax_schematic)
ax_schematic.tick_params(axis="y", width=0, length=0)


# =============================================================================
# Panel 2: RT wrt LED
# =============================================================================
ax_rt_led.step(rt_led_payload["data_x_ms"], rt_led_payload["data_hist_on_scaled"], where="mid", color="r", alpha=0.4, lw=2.0)
ax_rt_led.step(rt_led_payload["data_x_ms"], rt_led_payload["data_hist_off_scaled"], where="mid", color="b", alpha=0.4, lw=2.0)
ax_rt_led.plot(rt_led_payload["theory_x_ms"], rt_led_payload["rtd_theory_on_wrt_led"], color="r", alpha=1.0, lw=2.4)
ax_rt_led.plot(rt_led_payload["theory_x_ms"], rt_led_payload["rtd_theory_off_wrt_led"], color="b", alpha=1.0, lw=2.4)
ax_rt_led.axvline(0, color="0.2", ls="--", lw=1.1, alpha=0.7)
ax_rt_led.axvline(rt_led_payload["del_m_plus_del_LED_ms"], color="0.2", ls=":", lw=1.1, alpha=0.7)
ax_rt_led.set_xlim(rt_led_payload["xlim_ms"])
ax_rt_led.set_xticks(rt_led_payload["xticks_ms"])
ax_rt_led.set_yticks([])
ax_rt_led.set_xlabel(rt_led_payload.get("xlabel", "RT wrt LED onset (ms)"), fontsize=LABEL_FS, labelpad=PANEL_XLABEL_PAD)
ax_rt_led.set_ylabel("Density", fontsize=LABEL_FS, labelpad=PANEL_YLABEL_PAD)
style_axes(ax_rt_led)
ax_rt_led.tick_params(axis="y", width=0, length=0)


# =============================================================================
# Panel 3: RT wrt fixation
# =============================================================================
ax_rt_fix.step(rt_fix_payload["data_x_ms"], rt_fix_payload["data_hist_on_scaled"], where="mid", color="r", alpha=0.4, lw=2.0)
ax_rt_fix.step(rt_fix_payload["data_x_ms"], rt_fix_payload["data_hist_off_scaled"], where="mid", color="b", alpha=0.4, lw=2.0)
ax_rt_fix.plot(rt_fix_payload["theory_x_ms"], rt_fix_payload["rtd_theory_on_wrt_fix"], color="r", alpha=1.0, lw=2.4)
ax_rt_fix.plot(rt_fix_payload["theory_x_ms"], rt_fix_payload["rtd_theory_off_wrt_fix"], color="b", alpha=1.0, lw=2.4)
ax_rt_fix.set_xlim(rt_fix_payload["xlim_ms"])
ax_rt_fix.set_xticks(rt_fix_payload["xticks_ms"])
ax_rt_fix.set_yticks([])
ax_rt_fix.set_xlabel(rt_fix_payload.get("xlabel", "RT wrt fixation (ms)"), fontsize=LABEL_FS, labelpad=PANEL_XLABEL_PAD)
ax_rt_fix.set_ylabel("Density", fontsize=LABEL_FS, labelpad=PANEL_YLABEL_PAD)
style_axes(ax_rt_fix)
ax_rt_fix.tick_params(axis="y", width=0, length=0)


# =============================================================================
# Panel 4: RT wrt LED zoomed
# =============================================================================
ax_rt_led_zoom.step(rt_led_zoom_payload["data_x_ms"], rt_led_zoom_payload["data_hist_on_scaled"], where="mid", color="r", alpha=0.4, lw=2.0)
ax_rt_led_zoom.step(rt_led_zoom_payload["data_x_ms"], rt_led_zoom_payload["data_hist_off_scaled"], where="mid", color="b", alpha=0.4, lw=2.0)
ax_rt_led_zoom.plot(rt_led_zoom_payload["theory_x_ms"], rt_led_zoom_payload["rtd_theory_on_wrt_led"], color="r", alpha=1.0, lw=2.4)
ax_rt_led_zoom.plot(rt_led_zoom_payload["theory_x_ms"], rt_led_zoom_payload["rtd_theory_off_wrt_led"], color="b", alpha=1.0, lw=2.4)
ax_rt_led_zoom.axvline(0, color="0.2", ls="--", lw=1.1, alpha=0.7)
ax_rt_led_zoom.set_xlim(rt_led_zoom_payload["xlim_ms"])
ax_rt_led_zoom.set_xticks([-100, 0, 100])
ax_rt_led_zoom.set_yticks([])
ax_rt_led_zoom.set_xlabel(rt_led_zoom_payload.get("xlabel", "RT wrt LED onset (ms)"), fontsize=LABEL_FS, labelpad=PANEL_XLABEL_PAD)
style_axes(ax_rt_led_zoom)
ax_rt_led_zoom.tick_params(axis="y", width=0, length=0)


# =============================================================================
# Right panel: corner plot image (full height, aligned with left block)
# =============================================================================
corner_img = render_corner_image(corner_payload)
ax_corner.imshow(corner_img)
ax_corner.set_aspect("equal")
ax_corner.set_anchor("W")
ax_corner.axis("off")


# =============================================================================
# Bottom row: ABL-wise delay fit + delay bar plot
# =============================================================================
plot_abl_delay_overlay(ax_abl_delay, abl_delay_overlay_payload)
ax_abl_delay.set_ylabel("Density", fontsize=BOTTOM_LABEL_FS, labelpad=BOTTOM_YLABEL_PAD)
plot_delay_bar(ax_delay_bar, delay_bar_payload)


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
