# %%
"""
LED7 reactive-only NPL+alpha condition delays by signed ILD and ABL.

For each condition, each animal contributes one posterior-mean t_E_aff value.
The plotted error bars are SEM across the six LED7 animals.
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
PLOT_DPI = 300

FIT_ROOT = (
    REPO_ROOT
    / "fit_animal_by_animal"
    / "numpyro_svi_npl_alpha_reactive_only_rt_ge_100ms_condition_delay_"
    "patience12_min50k_restore_best_outputs"
)
OUTPUT_DIR = (
    SCRIPT_DIR
    / "led7_npl_alpha_reactive_ge100ms_fit_aligned_rtds"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_PATH = (
    OUTPUT_DIR
    / "led7_npl_alpha_reactive_ge100ms_t_E_aff_vs_ild_by_abl.png"
)
ANIMAL_DELAY_CSV = (
    OUTPUT_DIR
    / "led7_npl_alpha_reactive_ge100ms_t_E_aff_posterior_means_by_animal.csv"
)
SUMMARY_CSV = (
    OUTPUT_DIR
    / "led7_npl_alpha_reactive_ge100ms_t_E_aff_across_animal_summary.csv"
)


# %%
# =============================================================================
# Load posterior delays
# =============================================================================
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

delay_rows = []

for animal in ANIMALS:
    fit_dir = FIT_ROOT / f"{BATCH_NAME}_{animal}"
    posterior_path = fit_dir / "main_fullrank_posterior_samples.npz"
    condition_path = fit_dir / "condition_table.csv"

    for required_path in (posterior_path, condition_path):
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    condition_df = (
        pd.read_csv(condition_path)
        .sort_values("condition_id")
        .reset_index(drop=True)
    )
    expected_conditions = pd.MultiIndex.from_product(
        [ABLS, SIGNED_ILDS],
        names=["ABL", "ILD"],
    ).to_frame(index=False)

    if len(condition_df) != len(expected_conditions) or not np.allclose(
        condition_df[["ABL", "ILD"]].to_numpy(dtype=float),
        expected_conditions[["ABL", "ILD"]].to_numpy(dtype=float),
        atol=1e-12,
        rtol=0,
    ):
        raise RuntimeError(
            f"Unexpected condition table for {BATCH_NAME}/{animal}."
        )

    with np.load(posterior_path) as posterior:
        if "t_E_aff" not in posterior.files:
            raise KeyError(f"Missing t_E_aff in {posterior_path}")
        delay_samples_s = np.asarray(posterior["t_E_aff"], dtype=float)

    if (
        delay_samples_s.ndim != 2
        or delay_samples_s.shape[1] != len(condition_df)
        or not np.isfinite(delay_samples_s).all()
    ):
        raise RuntimeError(
            f"Invalid t_E_aff posterior shape/values for "
            f"{BATCH_NAME}/{animal}: {delay_samples_s.shape}"
        )

    delay_mean_ms = np.mean(delay_samples_s, axis=0) * 1e3
    delay_ci_ms = np.quantile(delay_samples_s, [0.025, 0.975], axis=0) * 1e3

    for condition, mean_ms, ci_low_ms, ci_high_ms in zip(
        condition_df.itertuples(index=False),
        delay_mean_ms,
        delay_ci_ms[0],
        delay_ci_ms[1],
    ):
        delay_rows.append(
            {
                "batch_name": BATCH_NAME,
                "animal": animal,
                "ABL": int(condition.ABL),
                "ILD": float(condition.ILD),
                "posterior_mean_ms": float(mean_ms),
                "posterior_ci_low_ms": float(ci_low_ms),
                "posterior_ci_high_ms": float(ci_high_ms),
            }
        )

    print(
        f"{BATCH_NAME}/{animal}: {delay_samples_s.shape[0]:,} posterior "
        f"samples; mean delay range "
        f"{delay_mean_ms.min():.1f}-{delay_mean_ms.max():.1f} ms"
    )

delay_df = pd.DataFrame(delay_rows)
delay_df.to_csv(ANIMAL_DELAY_CSV, index=False)

delay_summary_df = (
    delay_df.groupby(["ABL", "ILD"], as_index=False)
    .agg(
        mean_ms=("posterior_mean_ms", "mean"),
        std_ms=("posterior_mean_ms", "std"),
        n_animals=("animal", "nunique"),
    )
    .sort_values(["ABL", "ILD"])
    .reset_index(drop=True)
)
delay_summary_df["sem_ms"] = (
    delay_summary_df["std_ms"]
    / np.sqrt(delay_summary_df["n_animals"])
)

if len(delay_summary_df) != len(ABLS) * len(SIGNED_ILDS):
    raise RuntimeError(
        f"Expected 30 ABL/ILD summary rows, found {len(delay_summary_df)}."
    )
if not np.all(delay_summary_df["n_animals"].to_numpy() == len(ANIMALS)):
    raise RuntimeError("Every delay point must contain all six LED7 animals.")

delay_summary_df.to_csv(SUMMARY_CSV, index=False)
print(f"Saved animal posterior means: {ANIMAL_DELAY_CSV}")
print(f"Saved across-animal summary: {SUMMARY_CSV}")


# %%
# =============================================================================
# Plot across-animal posterior means
# =============================================================================
abl_colors = {
    20: "#1f77b4",
    40: "#ff7f0e",
    60: "#2ca02c",
}

fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)

for abl in ABLS:
    sub = delay_summary_df[delay_summary_df["ABL"].eq(abl)]
    ax.errorbar(
        sub["ILD"],
        sub["mean_ms"],
        yerr=sub["sem_ms"],
        fmt="o-",
        color=abl_colors[abl],
        ecolor=abl_colors[abl],
        linewidth=1.3,
        elinewidth=1.1,
        capsize=3,
        markersize=5,
        label=f"ABL {abl}",
    )

ax.axvline(0, color="0.82", linewidth=1, zorder=0)
ax.set_xticks(SIGNED_ILDS)
ax.set_xticklabels(
    [f"{ild:g}" for ild in SIGNED_ILDS],
    rotation=45,
    ha="right",
)
ax.set_xlabel("ILD (dB)")
ax.set_ylabel(r"$t_{E,\mathrm{aff}}$ (ms)")
ax.set_title("LED7 reactive-only NPL+alpha SVI condition delays")
ax.grid(axis="y", alpha=0.22)
ax.legend(frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.savefig(FIG_PATH, dpi=PLOT_DPI, bbox_inches="tight")
print(f"Saved figure: {FIG_PATH}")

# %%
