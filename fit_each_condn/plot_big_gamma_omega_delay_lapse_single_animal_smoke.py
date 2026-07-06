# %%
"""
Single-animal diagnostic for the Gamma/Omega/delay+lapse SVI smoke test.

The figure overlays the new lapse fit against the completed no-lapse 92-param
big SVI fit and compares the fitted lapse scalars with the accepted IPL+lapse
and NPL+alpha+lapse animal-wise SVI fits.
"""

# %%
from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# %%
# =============================================================================
# Editable parameters
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

BATCH_NAME = os.environ.get("BIG_GAMMA_OMEGA_DELAY_LAPSE_SMOKE_BATCH", "LED8")
ANIMAL = int(os.environ.get("BIG_GAMMA_OMEGA_DELAY_LAPSE_SMOKE_ANIMAL", "105"))

LAPSE_OUTPUT_ROOT = Path(
    os.environ.get(
        "BIG_GAMMA_OMEGA_DELAY_LAPSE_SMOKE_OUTPUT_ROOT",
        str(SCRIPT_DIR / "svi_big_gamma_omega_delay_lapse_single_animal_outputs"),
    )
).expanduser()
NO_LAPSE_OUTPUT_ROOT = (
    SCRIPT_DIR / "svi_big_gamma_omega_delay_patience12_restore_best_all_animals_outputs"
)
VANILLA_LAPSE_OUTPUT_ROOT = (
    REPO_DIR
    / "fit_animal_by_animal"
    / "numpyro_svi_vanilla_lapse_condition_delay_patience12_min50k_restore_best_outputs"
)
NPL_ALPHA_LAPSE_OUTPUT_ROOT = (
    REPO_DIR
    / "fit_animal_by_animal"
    / "numpyro_svi_npl_alpha_lapse_condition_delay_patience12_min50k_restore_best_outputs"
)

if not LAPSE_OUTPUT_ROOT.is_absolute():
    LAPSE_OUTPUT_ROOT = (REPO_DIR / LAPSE_OUTPUT_ROOT).resolve()

LAPSE_OUTPUT_DIR = LAPSE_OUTPUT_ROOT / f"{BATCH_NAME}_{ANIMAL}"
LAPSE_PREFIX = f"{BATCH_NAME}_{ANIMAL}_big_gamma_omega_delay_lapse"
NO_LAPSE_OUTPUT_DIR = NO_LAPSE_OUTPUT_ROOT / f"{BATCH_NAME}_{ANIMAL}"
NO_LAPSE_PREFIX = f"{BATCH_NAME}_{ANIMAL}_big_gamma_omega_delay"

LAPSE_CONDITION_SUMMARY = LAPSE_OUTPUT_DIR / f"{LAPSE_PREFIX}_condition_summary.csv"
LAPSE_POSTERIOR_SUMMARY = LAPSE_OUTPUT_DIR / f"{LAPSE_PREFIX}_posterior_summary.csv"
LAPSE_LOSS = LAPSE_OUTPUT_DIR / f"{LAPSE_PREFIX}_loss.csv"
LAPSE_CONVERGENCE = LAPSE_OUTPUT_DIR / f"{LAPSE_PREFIX}_convergence_checks.csv"
NO_LAPSE_CONDITION_SUMMARY = NO_LAPSE_OUTPUT_DIR / f"{NO_LAPSE_PREFIX}_condition_summary.csv"

OUTPUT_FIG = LAPSE_OUTPUT_DIR / f"{LAPSE_PREFIX}_smoke_diagnostic.png"


# %%
def read_scalar(summary_csv, parameter):
    summary_csv = Path(summary_csv)
    if not summary_csv.exists():
        return None
    summary_df = pd.read_csv(summary_csv)
    match = summary_df[summary_df["parameter"].astype(str) == str(parameter)]
    if len(match) != 1:
        return None
    row = match.iloc[0]
    return {
        "mean": float(row["mean"]),
        "q025": float(row["q025"]),
        "q975": float(row["q975"]),
    }


# %%
for required_path in [
    LAPSE_CONDITION_SUMMARY,
    LAPSE_POSTERIOR_SUMMARY,
    LAPSE_LOSS,
    LAPSE_CONVERGENCE,
    NO_LAPSE_CONDITION_SUMMARY,
]:
    if not required_path.exists():
        raise FileNotFoundError(required_path)

lapse_condition_df = pd.read_csv(LAPSE_CONDITION_SUMMARY).sort_values(["ABL", "ILD"])
no_lapse_condition_df = pd.read_csv(NO_LAPSE_CONDITION_SUMMARY).sort_values(["ABL", "ILD"])
loss_df = pd.read_csv(LAPSE_LOSS)
convergence_df = pd.read_csv(LAPSE_CONVERGENCE)

lapse_scalar_sources = {
    "Big lapse": LAPSE_POSTERIOR_SUMMARY,
    "IPL+lapse": VANILLA_LAPSE_OUTPUT_ROOT / f"{BATCH_NAME}_{ANIMAL}" / "main_fullrank_posterior_summary.csv",
    "NPL+alpha+lapse": NPL_ALPHA_LAPSE_OUTPUT_ROOT
    / f"{BATCH_NAME}_{ANIMAL}"
    / "main_fullrank_posterior_summary.csv",
}

abl_colors = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}

fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
axes = axes.ravel()

axes[0].plot(loss_df["step"], loss_df["loss"], color="tab:blue", linewidth=0.8)
axes[0].scatter(
    convergence_df["end_step"],
    convergence_df["mean_loss"],
    color="tab:orange",
    s=20,
    label="1k mean",
    zorder=3,
)
if "best_end_step_so_far" in convergence_df.columns and len(convergence_df):
    best_step = int(convergence_df["best_end_step_so_far"].iloc[-1])
    final_step = int(convergence_df["end_step"].iloc[-1])
    axes[0].axvline(best_step, color="tab:green", linewidth=1.5, label="restored best")
    axes[0].axvline(final_step, color="tab:red", linestyle="--", linewidth=1.2, label="final checked")
axes[0].set_title("ELBO")
axes[0].set_xlabel("SVI step")
axes[0].set_ylabel("negative ELBO")
axes[0].grid(True, alpha=0.25)
axes[0].legend(frameon=False, fontsize=8, loc="best")

plot_specs = [
    (axes[1], "gamma_mean", "Gamma", 1.0),
    (axes[2], "omega_mean", "Omega", 1.0),
    (axes[3], "t_E_aff_mean", "t_E_aff (ms)", 1000.0),
]
for ax, mean_col, title, scale in plot_specs:
    for abl, no_lapse_abl_df in no_lapse_condition_df.groupby("ABL", sort=True):
        abl = int(abl)
        lapse_abl_df = lapse_condition_df[lapse_condition_df["ABL"].astype(int) == abl]
        ax.plot(
            no_lapse_abl_df["ILD"],
            no_lapse_abl_df[mean_col] * scale,
            linestyle="--",
            marker="x",
            color=abl_colors.get(abl),
            alpha=0.45,
            linewidth=2.2,
            markersize=4,
            label=f"ABL {abl} no lapse" if title == "Gamma" else None,
        )
        ax.plot(
            lapse_abl_df["ILD"],
            lapse_abl_df[mean_col] * scale,
            linestyle="-",
            marker="o",
            color=abl_colors.get(abl),
            linewidth=1.0,
            markersize=4,
            label=f"ABL {abl} lapse" if title == "Gamma" else None,
        )
    ax.set_title(title)
    ax.set_xlabel("ILD")
    ax.set_ylabel(title)
    ax.grid(True, alpha=0.25)
    unique_ilds = np.sort(lapse_condition_df["ILD"].unique())
    ax.set_xticks(unique_ilds)
    ax.set_xticklabels([f"{int(ild):+d}" for ild in unique_ilds], rotation=45, ha="right", fontsize=8)
    if title == "Gamma":
        ax.axhline(0, color="0.75", linewidth=0.8, zorder=0)
        ax.legend(frameon=False, fontsize=7, ncol=2)

for ax, parameter, title, scale in [
    (axes[4], "lapse_prob", "lapse rate (%)", 100.0),
    (axes[5], "lapse_prob_right", "lapse prob right", 1.0),
]:
    labels = []
    xs = []
    means = []
    yerr_low = []
    yerr_high = []
    for source_name, summary_csv in lapse_scalar_sources.items():
        scalar = read_scalar(summary_csv, parameter)
        if scalar is None:
            continue
        labels.append(source_name)
        xs.append(len(xs))
        mean = scalar["mean"] * scale
        means.append(mean)
        yerr_low.append(mean - scalar["q025"] * scale)
        yerr_high.append(scalar["q975"] * scale - mean)
    ax.errorbar(
        xs,
        means,
        yerr=np.vstack([yerr_low, yerr_high]),
        fmt="o",
        color="tab:purple",
        ecolor="tab:purple",
        capsize=3,
        markersize=5,
        linestyle="none",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(title)
    ax.grid(True, axis="y", alpha=0.25)

fig.suptitle(f"{BATCH_NAME}/{ANIMAL} Gamma/Omega/delay+lapse SVI smoke diagnostic", y=1.02)
fig.tight_layout()
fig.savefig(OUTPUT_FIG, dpi=200, bbox_inches="tight")
print(f"Saved {OUTPUT_FIG}")

# %%
