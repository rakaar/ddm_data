# %%
"""
LED7 t_E_aff posterior histograms from the direct patience-12 NPL+alpha fit.

Rows are the ten signed ILDs, columns are the six LED7 animals, and each
panel overlays the ABL 20/40/60 variational-posterior sample histograms.
"""

# %%
# =============================================================================
# Editable parameters
# =============================================================================
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

BATCH_NAME = "LED7"
ANIMALS = (92, 93, 98, 99, 100, 103)
ABLS = (20, 40, 60)
SIGNED_ILDS = (-16.0, -8.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0, 16.0)
EXPECTED_POSTERIOR_SAMPLES = 10_000
ABL_PAIRS = ((20, 40), (40, 60), (20, 60))
KS_ALPHA = 0.05

HIST_MIN_MS = 0.0
HIST_MAX_MS = 165.0
HIST_BIN_WIDTH_MS = 1.0
PLOT_DPI = 250

FIT_ROOT = Path(
    os.environ.get(
        "NPL_ALPHA_PATIENCE12_FIT_ROOT",
        str(
            REPO_ROOT
            / "fit_animal_by_animal"
            / "numpyro_svi_npl_alpha_condition_delay_patience12_restore_best_outputs"
        ),
    )
).expanduser()
OUTPUT_DIR = SCRIPT_DIR / "led7_npl_alpha_patience12_t_E_aff_posteriors"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_PATH = (
    OUTPUT_DIR
    / "led7_npl_alpha_patience12_t_E_aff_posterior_histograms_10x6.png"
)
SUMMARY_CSV = (
    OUTPUT_DIR
    / "led7_npl_alpha_patience12_t_E_aff_posterior_histogram_summary.csv"
)
PAIRWISE_KS_CSV = (
    OUTPUT_DIR
    / "led7_npl_alpha_patience12_t_E_aff_pairwise_ks_by_animal_ild.csv"
)
PAIRWISE_KS_SUMMARY_CSV = (
    OUTPUT_DIR
    / "led7_npl_alpha_patience12_t_E_aff_pairwise_ks_count_summary.csv"
)


# %%
# =============================================================================
# Imports and plotting defaults
# =============================================================================
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

ABL_COLORS = {
    20: "#1f77b4",
    40: "#ff7f0e",
    60: "#2ca02c",
}

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
        "axes.linewidth": 0.7,
    }
)


# %%
# =============================================================================
# Load and validate all 180 condition posterior distributions
# =============================================================================
if not FIT_ROOT.exists():
    raise FileNotFoundError(FIT_ROOT)

hist_edges_ms = np.arange(
    HIST_MIN_MS,
    HIST_MAX_MS + HIST_BIN_WIDTH_MS,
    HIST_BIN_WIDTH_MS,
)
if not np.isclose(hist_edges_ms[-1], HIST_MAX_MS):
    raise RuntimeError("Histogram bounds are not divisible by the bin width.")

expected_conditions = pd.MultiIndex.from_product(
    [ABLS, SIGNED_ILDS],
    names=["ABL", "ILD"],
).to_frame(index=False)

posterior_by_condition = {}
summary_rows = []

for animal in ANIMALS:
    fit_dir = FIT_ROOT / f"{BATCH_NAME}_{animal}"
    posterior_path = fit_dir / "main_fullrank_posterior_samples.npz"
    condition_path = fit_dir / "condition_table.csv"
    finite_path = fit_dir / "main_fullrank_posterior_finite_report.csv"
    for required_path in (posterior_path, condition_path, finite_path):
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    condition_df = (
        pd.read_csv(condition_path)
        .sort_values("condition_id")
        .reset_index(drop=True)
    )
    if len(condition_df) != len(expected_conditions) or not np.allclose(
        condition_df[["ABL", "ILD"]].to_numpy(dtype=float),
        expected_conditions[["ABL", "ILD"]].to_numpy(dtype=float),
        atol=1e-12,
        rtol=0,
    ):
        raise RuntimeError(f"Unexpected condition table for LED7/{animal}.")

    finite_df = pd.read_csv(finite_path)
    if not (
        finite_df["n_total"].to_numpy(dtype=int)
        == finite_df["n_finite"].to_numpy(dtype=int)
    ).all():
        raise RuntimeError(f"Non-finite posterior report for LED7/{animal}.")

    with np.load(posterior_path) as posterior:
        if "t_E_aff" not in posterior.files:
            raise KeyError(f"Missing t_E_aff in {posterior_path}.")
        delay_samples_ms = np.asarray(posterior["t_E_aff"], dtype=float) * 1e3

    expected_shape = (EXPECTED_POSTERIOR_SAMPLES, len(expected_conditions))
    if delay_samples_ms.shape != expected_shape:
        raise RuntimeError(
            f"Expected t_E_aff shape {expected_shape} for LED7/{animal}, "
            f"found {delay_samples_ms.shape}."
        )
    if not np.isfinite(delay_samples_ms).all():
        raise RuntimeError(f"Non-finite t_E_aff samples for LED7/{animal}.")
    if (
        np.min(delay_samples_ms) < HIST_MIN_MS
        or np.max(delay_samples_ms) > HIST_MAX_MS
    ):
        raise RuntimeError(
            f"LED7/{animal} samples exceed the configured histogram range: "
            f"{np.min(delay_samples_ms):.3f}--"
            f"{np.max(delay_samples_ms):.3f} ms."
        )

    for condition in condition_df.itertuples(index=False):
        condition_id = int(condition.condition_id)
        abl = int(condition.ABL)
        ild = float(condition.ILD)
        samples_ms = delay_samples_ms[:, condition_id]
        counts, _ = np.histogram(samples_ms, bins=hist_edges_ms)
        density = counts.astype(float) / (
            len(samples_ms) * HIST_BIN_WIDTH_MS
        )
        histogram_area = float(np.sum(density * np.diff(hist_edges_ms)))
        if not np.isclose(histogram_area, 1.0, atol=1e-12, rtol=0):
            raise RuntimeError(
                f"Histogram area differs from one for LED7/{animal}, "
                f"ABL={abl}, ILD={ild:g}: {histogram_area:.12f}."
            )

        posterior_by_condition[(animal, abl, ild)] = {
            "samples_ms": samples_ms,
            "density": density,
        }
        ci_low_ms, ci_high_ms = np.quantile(samples_ms, [0.025, 0.975])
        summary_rows.append(
            {
                "batch_name": BATCH_NAME,
                "animal": animal,
                "ABL": abl,
                "ILD": ild,
                "condition_id": condition_id,
                "n_samples": len(samples_ms),
                "mean_ms": float(np.mean(samples_ms)),
                "median_ms": float(np.median(samples_ms)),
                "ci_2_5_ms": float(ci_low_ms),
                "ci_97_5_ms": float(ci_high_ms),
                "sample_min_ms": float(np.min(samples_ms)),
                "sample_max_ms": float(np.max(samples_ms)),
                "histogram_area": histogram_area,
            }
        )

    print(
        f"LED7/{animal}: {delay_samples_ms.shape[0]:,} samples x "
        f"{delay_samples_ms.shape[1]} conditions; range "
        f"{np.min(delay_samples_ms):.2f}--"
        f"{np.max(delay_samples_ms):.2f} ms"
    )

expected_distributions = len(ANIMALS) * len(ABLS) * len(SIGNED_ILDS)
if len(posterior_by_condition) != expected_distributions:
    raise RuntimeError(
        f"Expected {expected_distributions} distributions, "
        f"found {len(posterior_by_condition)}."
    )

summary_df = pd.DataFrame(summary_rows).sort_values(
    ["ILD", "animal", "ABL"]
).reset_index(drop=True)
summary_df.to_csv(SUMMARY_CSV, index=False)

print(f"Fit root: {FIT_ROOT}")
print(f"Loaded distributions: {len(posterior_by_condition)}")
print(
    f"Histogram bins: {HIST_MIN_MS:g}--{HIST_MAX_MS:g} ms in "
    f"{HIST_BIN_WIDTH_MS:g} ms steps"
)
print(f"Saved summary: {SUMMARY_CSV}")


# %%
# =============================================================================
# Pairwise two-sample KS tests requested for each animal x signed-ILD panel
# =============================================================================
ks_rows = []
for abl_a, abl_b in ABL_PAIRS:
    for animal in ANIMALS:
        for ild in SIGNED_ILDS:
            samples_a_ms = posterior_by_condition[(animal, abl_a, ild)][
                "samples_ms"
            ]
            samples_b_ms = posterior_by_condition[(animal, abl_b, ild)][
                "samples_ms"
            ]
            ks_result = ks_2samp(
                samples_a_ms,
                samples_b_ms,
                alternative="two-sided",
                method="asymp",
            )
            is_different = bool(ks_result.pvalue < KS_ALPHA)
            ks_rows.append(
                {
                    "batch_name": BATCH_NAME,
                    "animal": animal,
                    "ILD": ild,
                    "ABL_a": abl_a,
                    "ABL_b": abl_b,
                    "pair": f"{abl_a} vs {abl_b}",
                    "mean_a_ms": float(np.mean(samples_a_ms)),
                    "mean_b_ms": float(np.mean(samples_b_ms)),
                    "mean_diff_a_minus_b_ms": float(
                        np.mean(samples_a_ms) - np.mean(samples_b_ms)
                    ),
                    "ks_statistic": float(ks_result.statistic),
                    "p_value_uncorrected": float(ks_result.pvalue),
                    "alpha": KS_ALPHA,
                    "classification": "different" if is_different else "same",
                }
            )

ks_df = pd.DataFrame(ks_rows)
ks_df.to_csv(PAIRWISE_KS_CSV, index=False)

ks_summary_rows = []
for abl_a, abl_b in ABL_PAIRS:
    pair_df = ks_df.loc[
        (ks_df["ABL_a"] == abl_a) & (ks_df["ABL_b"] == abl_b)
    ]
    n_total = len(pair_df)
    n_different = int((pair_df["classification"] == "different").sum())
    n_same = int((pair_df["classification"] == "same").sum())
    if n_total != len(ANIMALS) * len(SIGNED_ILDS):
        raise RuntimeError(
            f"Expected 60 KS cases for ABL {abl_a} vs {abl_b}, found {n_total}."
        )
    ks_summary_rows.append(
        {
            "ABL_pair": f"{abl_a} vs {abl_b}",
            "n_cases": n_total,
            "n_different_p_lt_0_05": n_different,
            "n_same_p_ge_0_05": n_same,
            "percent_different": 100.0 * n_different / n_total,
        }
    )

ks_summary_df = pd.DataFrame(ks_summary_rows)
ks_summary_df.to_csv(PAIRWISE_KS_SUMMARY_CSV, index=False)

print("\nPairwise two-sample KS classifications (uncorrected alpha = 0.05):")
print(ks_summary_df.to_string(index=False))
print(f"Saved detailed KS table: {PAIRWISE_KS_CSV}")
print(f"Saved KS count table: {PAIRWISE_KS_SUMMARY_CSV}")


# %%
# =============================================================================
# Plot signed ILD rows x animal columns
# =============================================================================
fig, axes = plt.subplots(
    len(SIGNED_ILDS),
    len(ANIMALS),
    figsize=(18.0, 24.0),
    sharex=True,
    sharey=False,
)

for row_idx, ild in enumerate(SIGNED_ILDS):
    for col_idx, animal in enumerate(ANIMALS):
        ax = axes[row_idx, col_idx]

        for abl in ABLS:
            density = posterior_by_condition[(animal, abl, ild)]["density"]
            ax.stairs(
                density,
                hist_edges_ms,
                fill=True,
                color=ABL_COLORS[abl],
                alpha=0.13,
                linewidth=0,
                zorder=1,
            )
            ax.stairs(
                density,
                hist_edges_ms,
                color=ABL_COLORS[abl],
                linewidth=1.05,
                zorder=2,
            )

        if row_idx == 0:
            ax.set_title(f"LED7/{animal}", fontsize=10)
        if col_idx == 0:
            ax.set_ylabel(
                f"ILD = {ild:g} dB",
                fontsize=8.5,
                labelpad=8,
            )

        ax.set_xlim(HIST_MIN_MS, HIST_MAX_MS)
        ax.set_xticks((0, 50, 100, 150))
        ax.set_ylim(bottom=0)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=7, length=2.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("0.65")
        ax.spines["bottom"].set_color("0.65")

legend_handles = [
    Line2D(
        [0],
        [0],
        color=ABL_COLORS[abl],
        linewidth=1.5,
        label=f"ABL {abl}",
    )
    for abl in ABLS
]
fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.984),
    ncol=3,
    frameon=False,
    fontsize=10,
)
fig.suptitle(
    "LED7 direct NPL+alpha SVI condition-delay posteriors",
    fontsize=14,
    y=0.998,
)
fig.supxlabel(r"$t_{E,\mathrm{aff}}$ (ms)", fontsize=11, y=0.018)
fig.supylabel("Posterior density", fontsize=11, x=0.012)
fig.subplots_adjust(
    left=0.065,
    right=0.995,
    bottom=0.035,
    top=0.955,
    wspace=0.13,
    hspace=0.26,
)
fig.savefig(FIG_PATH, dpi=PLOT_DPI, bbox_inches="tight")
print(f"Saved figure: {FIG_PATH}")

# %%
