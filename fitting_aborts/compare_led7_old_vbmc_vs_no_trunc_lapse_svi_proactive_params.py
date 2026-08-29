# %%
"""Compare paper-era proactive VBMC parameters with the current LED7 SVI fits."""

# %%
# =============================================================================
# Parameters
# =============================================================================
from pathlib import Path
import json
import os
import pickle

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

OLD_VBMC_ROOT = REPO_DIR / "aborts_ipl_npl_time_fit_results"
BASE_SVI_ROOT = (
    SCRIPT_DIR
    / "numpyro_svi_proactive_led_step_jump_all_on_no_trunc_exp_lapse_"
    "patience12_min50k_restore_best_outputs"
)
AUDIT_SVI_ROOT = (
    SCRIPT_DIR
    / "numpyro_svi_proactive_led_step_jump_all_on_no_trunc_exp_lapse_"
    "patience12_min150k_restore_best_audit_outputs"
)

ANIMALS = (92, 93, 98, 99, 100, 103)
AUDIT_ANIMALS = (92, 99, 100, 103)

OUTPUT_DIR = BASE_SVI_ROOT / "summary_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_PATH = OUTPUT_DIR / "led7_old_vbmc_vs_no_trunc_lapse_svi_proactive_params.png"
SUMMARY_CSV = OUTPUT_DIR / "led7_old_vbmc_vs_no_trunc_lapse_svi_proactive_params.csv"
CORRECTED_93_PROVENANCE = (
    SCRIPT_DIR
    / "vbmc_old_truncated_proactive_led7_93_more_evals"
    / "canonical_replacement"
    / "replacement_provenance.json"
)

OLD_LABEL = "Old VBMC"
NEW_LABEL = "New SVI"
OLD_COLOR = "#D62728"
NEW_COLOR = "#1F77B4"


# %%
# =============================================================================
# Load posterior samples and build a tidy summary
# =============================================================================
summary_rows = []

for animal in ANIMALS:
    old_path = OLD_VBMC_ROOT / f"results_LED7_animal_{animal}.pkl"
    selected_svi_root = AUDIT_SVI_ROOT if animal in AUDIT_ANIMALS else BASE_SVI_ROOT
    svi_dir = selected_svi_root / f"LED7_{animal}"
    posterior_path = svi_dir / "main_fullrank_posterior_samples.npz"
    run_summary_path = svi_dir / "run_summary.json"

    for required_path in (old_path, posterior_path, run_summary_path):
        if not required_path.exists():
            raise FileNotFoundError(f"Missing required fit artifact: {required_path}")

    with old_path.open("rb") as handle:
        old_saved = pickle.load(handle)
    if "vbmc_aborts_results" not in old_saved:
        raise KeyError(f"Missing vbmc_aborts_results in {old_path}")
    old_result = old_saved["vbmc_aborts_results"]

    svi_summary = json.loads(run_summary_path.read_text())
    if svi_summary.get("status") not in {"complete", "max_steps_restore_best"}:
        raise RuntimeError(
            f"LED7/{animal} SVI status is {svi_summary.get('status')!r}, not complete."
        )
    if svi_summary.get("config", {}).get("truncation") != "none":
        raise RuntimeError(f"LED7/{animal} selected SVI fit is not no-truncation.")
    if not svi_summary.get("all_posterior_samples_finite", False):
        raise RuntimeError(f"LED7/{animal} SVI summary reports non-finite samples.")

    with np.load(posterior_path) as svi_saved:
        required_svi_keys = {
            "V_A_base",
            "theta_A",
            "del_a_minus_del_LED",
            "del_m_plus_del_LED",
        }
        missing_svi_keys = required_svi_keys.difference(svi_saved.files)
        if missing_svi_keys:
            raise KeyError(
                f"Missing SVI posterior arrays for LED7/{animal}: "
                f"{sorted(missing_svi_keys)}"
            )

        old_samples = {
            "V_A": np.asarray(old_result["V_A_samples"], dtype=float),
            "theta_A": np.asarray(old_result["theta_A_samples"], dtype=float),
            "t_A_aff": 1000.0 * np.asarray(old_result["t_A_aff_samp"], dtype=float),
        }
        del_a_samples = np.asarray(
            svi_saved["del_a_minus_del_LED"], dtype=float
        )
        del_m_samples = np.asarray(
            svi_saved["del_m_plus_del_LED"], dtype=float
        )
        if del_a_samples.shape != del_m_samples.shape:
            raise RuntimeError(
                f"LED7/{animal} delay-component posterior shapes differ: "
                f"{del_a_samples.shape} versus {del_m_samples.shape}."
            )
        new_delay_ms = 1000.0 * (del_a_samples + del_m_samples)
        new_samples = {
            "V_A": np.asarray(svi_saved["V_A_base"], dtype=float),
            "theta_A": np.asarray(svi_saved["theta_A"], dtype=float),
            "t_A_aff": new_delay_ms,
        }

    for model_samples, model_label in (
        (old_samples, OLD_LABEL),
        (new_samples, NEW_LABEL),
    ):
        for parameter, samples in model_samples.items():
            if samples.ndim != 1 or samples.size == 0 or not np.isfinite(samples).all():
                raise RuntimeError(
                    f"Invalid {model_label} {parameter} samples for LED7/{animal}."
                )
            ci_low, ci_high = np.quantile(samples, [0.025, 0.975])
            summary_rows.append(
                {
                    "animal": animal,
                    "model": model_label,
                    "parameter": parameter,
                    "mean": float(np.mean(samples)),
                    "ci_2.5": float(ci_low),
                    "ci_97.5": float(ci_high),
                    "units": "ms" if parameter == "t_A_aff" else "model units",
                    "n_posterior_samples": int(samples.size),
                    "source_path": str(
                        (old_path if model_label == OLD_LABEL else posterior_path).resolve()
                    ),
                    "fit_status": (
                        str(old_result.get("message", ""))
                        if model_label == OLD_LABEL
                        else str(svi_summary["status"])
                    ),
                    "stop_reason": (
                        ""
                        if model_label == OLD_LABEL
                        else str(svi_summary.get("stop_reason", ""))
                    ),
                    "completed_steps": (
                        np.nan
                        if model_label == OLD_LABEL
                        else int(svi_summary["completed_steps"])
                    ),
                    "restored_best_step": (
                        np.nan
                        if model_label == OLD_LABEL
                        else int(svi_summary["restored_best_step"])
                    ),
                    "derived_from": (
                        "t_A_aff_samp"
                        if model_label == OLD_LABEL
                        else (
                            "del_a_minus_del_LED + del_m_plus_del_LED"
                            if parameter == "t_A_aff"
                            else ("V_A_base" if parameter == "V_A" else "theta_A")
                        )
                    ),
                }
            )

summary_df = pd.DataFrame(summary_rows)
expected_rows = len(ANIMALS) * 2 * 3
if len(summary_df) != expected_rows:
    raise RuntimeError(f"Expected {expected_rows} summary rows, found {len(summary_df)}.")

old_93_rows = summary_df[
    (summary_df["animal"] == 93) & (summary_df["model"] == OLD_LABEL)
]
if len(old_93_rows) != 3 or not old_93_rows["fit_status"].str.contains(
    "stable", case=False, regex=False
).all():
    raise RuntimeError("Expected the corrected LED7/93 VBMC fit to report stability.")

if not CORRECTED_93_PROVENANCE.exists():
    raise FileNotFoundError(CORRECTED_93_PROVENANCE)
corrected_93_provenance = json.loads(CORRECTED_93_PROVENANCE.read_text())
expected_93_means = corrected_93_provenance["new_posterior_means"]
observed_93_means = old_93_rows.set_index("parameter")["mean"]
for parameter, provenance_name in (
    ("V_A", "V_A"),
    ("theta_A", "theta_A"),
    ("t_A_aff", "t_A_aff_s"),
):
    expected_mean = float(expected_93_means[provenance_name])
    if parameter == "t_A_aff":
        expected_mean *= 1000.0
    if not np.isclose(observed_93_means.loc[parameter], expected_mean, atol=1e-10):
        raise RuntimeError(
            f"LED7/93 {parameter} does not match the promoted corrected posterior."
        )

summary_df.to_csv(SUMMARY_CSV, index=False)


# %%
# =============================================================================
# Plot posterior means and 95% credible intervals
# =============================================================================
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

parameter_panels = (
    ("V_A", r"$V_A$"),
    ("theta_A", r"$\theta_A$"),
    ("t_A_aff", r"$t_{A,\mathrm{aff}} = \delta_a + \delta_m$ (ms)"),
)
x = np.arange(len(ANIMALS), dtype=float)
model_styles = {
    OLD_LABEL: {"offset": -0.10, "color": OLD_COLOR, "mfc": "white"},
    NEW_LABEL: {"offset": 0.10, "color": NEW_COLOR, "mfc": NEW_COLOR},
}


def draw_parameter_points(ax, parameter):
    """Draw the two posterior summaries with a consistent paired layout."""
    for model_label, style in model_styles.items():
        model_rows = summary_df[
            (summary_df["parameter"] == parameter)
            & (summary_df["model"] == model_label)
        ].set_index("animal")
        rows = model_rows.loc[list(ANIMALS)]
        means = rows["mean"].to_numpy(dtype=float)
        ci_low = rows["ci_2.5"].to_numpy(dtype=float)
        ci_high = rows["ci_97.5"].to_numpy(dtype=float)
        ax.errorbar(
            x + style["offset"],
            means,
            yerr=np.vstack((means - ci_low, ci_high - means)),
            fmt="o",
            ms=5.5,
            mfc=style["mfc"],
            mec=style["color"],
            mew=1.2,
            ecolor=style["color"],
            elinewidth=1.0,
            capsize=2.5,
            capthick=1.0,
            linestyle="none",
            zorder=3,
        )


fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.7))

for ax, (parameter, ylabel) in zip(axes, parameter_panels):
    draw_parameter_points(ax, parameter)
    ax.set_xticks(x)
    ax.set_xticklabels([str(animal) for animal in ANIMALS], rotation=45, ha="right")
    ax.set_xlabel("Animal")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel.replace(" (ms)", ""), fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.85", alpha=0.55, lw=0.6)

axes[2].axhline(0.0, color="0.55", lw=0.7, ls=":", zorder=1)

fig.legend(
    handles=[
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=OLD_COLOR,
            markeredgewidth=1.2,
            color=OLD_COLOR,
            label="Paper VBMC",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=NEW_COLOR,
            markeredgecolor=NEW_COLOR,
            color=NEW_COLOR,
            label="No-truncation+lapse SVI",
        ),
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, 1.015),
    ncol=2,
    frameon=False,
)
fig.subplots_adjust(left=0.065, right=0.985, bottom=0.20, top=0.82, wspace=0.32)
fig.savefig(FIG_PATH, dpi=240, bbox_inches="tight")

print(f"Saved figure: {FIG_PATH}")
print(f"Saved summary: {SUMMARY_CSV}")
print("Selected SVI source roots:")
for animal in ANIMALS:
    root = AUDIT_SVI_ROOT if animal in AUDIT_ANIMALS else BASE_SVI_ROOT
    print(f"  LED7/{animal}: {root}")
print("LED7/93 paper-VBMC source: corrected stable posterior promoted 2026-08-29.")
print(f"LED7/93 replacement provenance: {CORRECTED_93_PROVENANCE}")
