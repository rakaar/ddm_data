# %%
"""Global-posterior and fitted uniform-delay diagnostics for one animal."""

# %%
# =============================================================================
# Editable parameters
# =============================================================================
from pathlib import Path
import os

SCRIPT_DIR = Path(__file__).resolve().parent

BATCH_NAME = os.environ.get("NUMPYRO_SVI_BATCH", "LED7")
ANIMAL = int(os.environ.get("NUMPYRO_SVI_ANIMAL", "92"))
ABLS = (20, 40, 60)
ABS_ILDS = (1.0, 2.0, 4.0, 8.0, 16.0)
SIGNED_ILDS = (-16.0, -8.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0, 16.0)
ABL_COLORS = {
    20: "#0072B2",
    40: "#E69F00",
    60: "#009E73",
}
DELAY_FILL_ALPHA = 0.4
PLOT_DPI = 300

OUTPUT_ROOT = Path(
    os.environ.get(
        "NUMPYRO_SVI_OUTPUT_ROOT",
        str(
            SCRIPT_DIR
            / (
                "numpyro_svi_npl_alpha_uniform_delay_rt_choice_"
                "patience12_min50k_restore_best_outputs"
            )
        ),
    )
).expanduser()
OUTPUT_DIR = OUTPUT_ROOT / f"{BATCH_NAME}_{ANIMAL}"
DIAGNOSTIC_DIR = OUTPUT_DIR / "diagnostics"
DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)

POSTERIOR_NPZ = OUTPUT_DIR / "main_fullrank_posterior_samples.npz"
CONDITION_CSV = OUTPUT_DIR / "condition_table.csv"
GLOBAL_CORNER_PNG = (
    DIAGNOSTIC_DIR
    / f"{BATCH_NAME.lower()}_{ANIMAL}_npl_alpha_uniform_delay_global_corner.png"
)
DELAY_DISTRIBUTIONS_PNG = (
    DIAGNOSTIC_DIR
    / (
        f"{BATCH_NAME.lower()}_{ANIMAL}_npl_alpha_uniform_delay_"
        "condition_distributions_by_ild.png"
    )
)
DELAY_SUPPORT_BY_ILD_PNG = (
    DIAGNOSTIC_DIR
    / (
        f"{BATCH_NAME.lower()}_{ANIMAL}_npl_alpha_uniform_delay_"
        "center_and_support_vs_ild_by_abl.png"
    )
)


# %%
# =============================================================================
# Imports and input checks
# =============================================================================
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import corner
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Helvetica",
            "Nimbus Sans",
            "Helvetica Neue",
            "Arial",
            "Liberation Sans",
            "sans-serif",
        ],
        "axes.linewidth": 0.8,
    }
)

for required_path in (POSTERIOR_NPZ, CONDITION_CSV):
    if not required_path.exists():
        raise FileNotFoundError(required_path)

posterior = np.load(POSTERIOR_NPZ)
condition_table = (
    pd.read_csv(CONDITION_CSV)
    .sort_values("condition_id")
    .reset_index(drop=True)
)
if len(condition_table) != 30:
    raise RuntimeError(f"Expected 30 conditions, found {len(condition_table)}.")


# %%
# =============================================================================
# Seven-global-parameter corner plot
# =============================================================================
global_samples = np.column_stack(
    [
        np.asarray(posterior["rate_lambda"], dtype=float),
        1e3 * np.asarray(posterior["T_0"], dtype=float),
        np.asarray(posterior["theta_E"], dtype=float),
        np.asarray(posterior["w"], dtype=float),
        1e3 * np.asarray(posterior["del_go"], dtype=float),
        np.asarray(posterior["rate_norm_l"], dtype=float),
        np.asarray(posterior["alpha"], dtype=float),
    ]
)
global_labels = [
    r"$\lambda^\prime$",
    r"$T_0$ (ms)",
    r"$\theta_E$",
    r"$w$",
    r"$\delta_{\mathrm{go}}$ (ms)",
    r"$\ell$",
    r"$\alpha$",
]

finite_rows = np.all(np.isfinite(global_samples), axis=1)
global_samples = global_samples[finite_rows]
if len(global_samples) < 100:
    raise RuntimeError(
        f"Only {len(global_samples)} finite posterior rows remain for the corner plot."
    )

corner_ranges = []
for column_index in range(global_samples.shape[1]):
    low, high = np.quantile(global_samples[:, column_index], [0.01, 0.99])
    if np.isclose(low, high, rtol=0, atol=1e-12):
        padding = max(1e-6, abs(low) * 1e-3)
        low -= padding
        high += padding
    corner_ranges.append((float(low), float(high)))

corner_figure = corner.corner(
    global_samples,
    labels=global_labels,
    range=corner_ranges,
    color="#0072B2",
    show_titles=True,
    title_quantiles=[0.025, 0.5, 0.975],
    quantiles=[0.025, 0.5, 0.975],
    title_fmt=".3f",
    smooth=0.8,
    smooth1d=0.8,
    plot_datapoints=False,
    fill_contours=True,
    hist_kwargs={"linewidth": 1.0},
    contour_kwargs={"linewidths": 0.8},
    label_kwargs={"fontsize": 11},
    title_kwargs={"fontsize": 8},
)
corner_figure.suptitle(
    f"{BATCH_NAME}/{ANIMAL} NPL+alpha uniform-delay global posterior",
    fontsize=14,
    y=1.01,
)
corner_figure.savefig(GLOBAL_CORNER_PNG, dpi=PLOT_DPI, bbox_inches="tight")


# %%
# =============================================================================
# Posterior-mean uniform delay distributions by signed ILD and ABL
# =============================================================================
center_samples = np.asarray(posterior["t_E_aff_center"], dtype=float)
width_samples = np.asarray(posterior["t_E_aff_width"], dtype=float)
if center_samples.shape != width_samples.shape or center_samples.shape[1] != 30:
    raise RuntimeError(
        "Expected matching 10,000 x 30 center and width posterior arrays."
    )

center_mean_ms = 1e3 * np.mean(center_samples, axis=0)
width_mean_ms = 1e3 * np.mean(width_samples, axis=0)
low_mean_ms = center_mean_ms - 0.5 * width_mean_ms
high_mean_ms = center_mean_ms + 0.5 * width_mean_ms
if not (
    np.isfinite(low_mean_ms).all()
    and np.isfinite(high_mean_ms).all()
    and np.all(low_mean_ms >= 0.0)
    and np.all(high_mean_ms <= 1000.0)
    and np.all(width_mean_ms > 0.0)
):
    raise RuntimeError("Invalid posterior-mean uniform delay intervals.")

condition_lookup = {
    (int(row.ABL), float(row.ILD)): int(row.condition_id)
    for row in condition_table.itertuples(index=False)
}
expected_conditions = {
    (abl, signed_ild)
    for abl in ABLS
    for abs_ild in ABS_ILDS
    for signed_ild in (-abs_ild, abs_ild)
}
if set(condition_lookup) != expected_conditions:
    raise RuntimeError("Condition table does not contain the expected ABL/ILD grid.")

x_max_ms = 5.0 * np.ceil((float(np.max(high_mean_ms)) + 5.0) / 5.0)
uniform_areas = []
delay_figure, axes = plt.subplots(
    2,
    5,
    figsize=(17.0, 6.2),
    sharex=True,
    sharey=True,
)

# Positive ILDs occupy the first row and negative ILDs the second row.
for row_index, sign in enumerate((1.0, -1.0)):
    for column_index, abs_ild in enumerate(ABS_ILDS):
        ax = axes[row_index, column_index]
        signed_ild = sign * abs_ild

        # Draw wider rectangles first so narrow distributions remain visible.
        intervals = []
        for abl in ABLS:
            condition_id = condition_lookup[(abl, signed_ild)]
            intervals.append(
                (
                    width_mean_ms[condition_id],
                    abl,
                    low_mean_ms[condition_id],
                    high_mean_ms[condition_id],
                )
            )
        for width_ms, abl, delay_low_ms, delay_high_ms in sorted(
            intervals,
            reverse=True,
        ):
            density_per_ms = 1.0 / width_ms
            color = ABL_COLORS[abl]
            ax.fill_between(
                [delay_low_ms, delay_high_ms],
                [density_per_ms, density_per_ms],
                [0.0, 0.0],
                color=color,
                alpha=DELAY_FILL_ALPHA,
                linewidth=0,
            )
            ax.plot(
                [delay_low_ms, delay_low_ms, delay_high_ms, delay_high_ms],
                [0.0, density_per_ms, density_per_ms, 0.0],
                color=color,
                linewidth=1.1,
                alpha=0.9,
            )
            uniform_areas.append(density_per_ms * width_ms)

        ax.set_xlim(0.0, x_max_ms)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        if row_index == 0:
            ax.set_title(f"ILD = +{abs_ild:g} dB", fontsize=10)
        else:
            ax.set_title(f"ILD = -{abs_ild:g} dB", fontsize=10)
            ax.set_xlabel(r"$t_{E,\mathrm{aff}}$ (ms)")
        if column_index == 0:
            row_label = "Positive ILD" if sign > 0 else "Negative ILD"
            ax.set_ylabel(f"{row_label}\nDensity (ms$^{{-1}}$)")

if not np.allclose(uniform_areas, 1.0, atol=1e-12):
    raise RuntimeError("A displayed uniform density does not integrate to one.")

legend_handles = [
    Line2D(
        [0],
        [0],
        color=ABL_COLORS[abl],
        linewidth=4,
        alpha=DELAY_FILL_ALPHA,
        label=f"ABL {abl} dB",
    )
    for abl in ABLS
]
delay_figure.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.01),
    ncol=3,
    frameon=False,
)
delay_figure.suptitle(
    (
        f"{BATCH_NAME}/{ANIMAL} fitted uniform evidence-delay distributions "
        "at posterior-mean center and width"
    ),
    fontsize=13,
    y=1.075,
)
delay_figure.tight_layout(rect=(0, 0, 1, 0.94), w_pad=1.1, h_pad=1.2)
delay_figure.savefig(
    DELAY_DISTRIBUTIONS_PNG,
    dpi=PLOT_DPI,
    bbox_inches="tight",
)


# %%
# =============================================================================
# Delay center and fitted Uniform support versus signed ILD
# =============================================================================
support_figure, support_ax = plt.subplots(
    figsize=(8.5, 5.2),
    constrained_layout=True,
)
for abl in ABLS:
    condition_ids = [condition_lookup[(abl, ild)] for ild in SIGNED_ILDS]
    means = center_mean_ms[condition_ids]
    lows = low_mean_ms[condition_ids]
    highs = high_mean_ms[condition_ids]
    asymmetric_support = np.vstack([means - lows, highs - means])
    support_ax.errorbar(
        SIGNED_ILDS,
        means,
        yerr=asymmetric_support,
        fmt="o-",
        color=ABL_COLORS[abl],
        ecolor=ABL_COLORS[abl],
        linewidth=1.3,
        elinewidth=1.1,
        capsize=3,
        markersize=5,
        label=f"ABL {abl} dB",
    )

support_padding_ms = max(
    2.0,
    0.06 * (float(np.max(high_mean_ms)) - float(np.min(low_mean_ms))),
)
support_ax.axvline(0.0, color="0.82", linewidth=1.0, zorder=0)
support_ax.set_xticks(SIGNED_ILDS)
support_ax.set_xticklabels(
    [f"{ild:g}" for ild in SIGNED_ILDS],
    rotation=45,
    ha="right",
)
support_ax.set_ylim(
    max(0.0, float(np.min(low_mean_ms)) - support_padding_ms),
    float(np.max(high_mean_ms)) + support_padding_ms,
)
support_ax.set_xlabel("ILD (dB)")
support_ax.set_ylabel(r"$t_{E,\mathrm{aff}}$ (ms)")
support_ax.set_title(
    f"{BATCH_NAME}/{ANIMAL} NPL+alpha uniform-delay condition delays"
)
support_ax.grid(axis="y", alpha=0.22)
support_ax.legend(frameon=False)
support_ax.spines[["top", "right"]].set_visible(False)
support_figure.savefig(
    DELAY_SUPPORT_BY_ILD_PNG,
    dpi=PLOT_DPI,
    bbox_inches="tight",
)

print(f"Finite global posterior draws: {len(global_samples):,}")
print(
    "Posterior-mean uniform widths: "
    f"{width_mean_ms.min():.3f}--{width_mean_ms.max():.3f} ms"
)
print(
    "Posterior-mean uniform support: "
    f"{low_mean_ms.min():.3f}--{high_mean_ms.max():.3f} ms"
)
print(f"Saved: {GLOBAL_CORNER_PNG}")
print(f"Saved: {DELAY_DISTRIBUTIONS_PNG}")
print(f"Saved: {DELAY_SUPPORT_BY_ILD_PNG}")
