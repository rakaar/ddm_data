# %%
"""Plot displaced-start convergence for the pooled LED7 valid-trial fits."""

# %%
# =============================================================================
# Editable paths and plotting settings
# =============================================================================
from pathlib import Path
import json
import os


SCRIPT_DIR = Path(__file__).resolve().parent
FIT_ROOT = Path(
    os.environ.get(
        "LED7_VALID_NPL_AGG_FIT_ROOT",
        str(
            SCRIPT_DIR
            / "numpyro_svi_proactive_led_step_jump_npl_alpha_valid_led7_"
            "aggregate_patience12_min50k_restore_best_outputs"
            / "LED7_all_animals"
        ),
    )
).expanduser()
OUTPUT_DIR = Path(
    os.environ.get(
        "LED7_VALID_NPL_AGG_DIAGNOSTIC_OUTPUT_DIR",
        str(FIT_ROOT / "diagnostics"),
    )
).expanduser()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORY_SETTINGS = {
    "off": {"label": "LED OFF", "color": "tab:blue"},
    "on_bilateral": {"label": "bilateral LED ON", "color": "tab:red"},
}


# %%
# =============================================================================
# Imports
# =============================================================================
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# %%
# =============================================================================
# Load and validate the two displaced-start branches
# =============================================================================
branch_payloads = {}
summary_rows = []

for category in CATEGORY_SETTINGS:
    branch_dir = FIT_ROOT / category / "displaced"
    summary_path = branch_dir / "run_summary.json"
    convergence_path = branch_dir / "main_fullrank_convergence_checks.csv"
    posterior_path = branch_dir / "main_fullrank_posterior_samples.npz"
    for path in [summary_path, convergence_path, posterior_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    with summary_path.open() as handle:
        summary = json.load(handle)
    convergence = pd.read_csv(convergence_path)

    if summary.get("status") != "complete":
        raise RuntimeError(f"{category}/displaced is not complete: {summary}")
    if summary.get("initialization_mode") != "displaced":
        raise RuntimeError(f"{category}: expected displaced initialization.")
    if summary.get("stop_reason") != "patience12_restore_best":
        raise RuntimeError(f"{category}: unexpected stop reason {summary.get('stop_reason')}")
    if summary.get("n_nonfinite_losses") != 0:
        raise RuntimeError(f"{category}: fit recorded non-finite losses.")
    if not summary.get("all_posterior_samples_finite", False):
        raise RuntimeError(f"{category}: posterior samples were not all finite.")

    required_columns = {"end_step", "mean_loss"}
    if not required_columns.issubset(convergence.columns):
        missing = sorted(required_columns - set(convergence.columns))
        raise RuntimeError(f"{category}: convergence CSV lacks {missing}.")
    if not np.all(
        np.isfinite(convergence[["end_step", "mean_loss"]].to_numpy(dtype=float))
    ):
        raise RuntimeError(f"{category}: convergence trace contains non-finite values.")

    restored_step = int(summary["restored_best_step"])
    checked_step = int(summary["completed_steps"])
    if restored_step not in set(convergence["end_step"].astype(int)):
        raise RuntimeError(
            f"{category}: restored step {restored_step} is absent from the trace."
        )
    if checked_step != int(convergence["end_step"].iloc[-1]):
        raise RuntimeError(
            f"{category}: checked step does not match the final convergence window."
        )

    with np.load(posterior_path) as saved:
        if not all(np.all(np.isfinite(saved[name])) for name in saved.files):
            raise RuntimeError(f"{category}: posterior archive contains non-finite values.")

    branch_payloads[category] = {
        "fit_dir": branch_dir,
        "summary": summary,
        "convergence": convergence,
    }
    summary_rows.append(
        {
            "category": category,
            "initialization_mode": "displaced",
            "status": summary["status"],
            "stop_reason": summary["stop_reason"],
            "restored_best_step": restored_step,
            "final_checked_step": checked_step,
            "best_window_negative_elbo": summary["best_window_negative_elbo"],
            "valid_trial_count": summary["valid_trial_count"],
            "fit_elapsed_minutes": summary["fit_elapsed_minutes"],
            "fit_root": str(branch_dir.resolve()),
        }
    )

print(f"Fit root: {FIT_ROOT.resolve()}")
for category, payload in branch_payloads.items():
    summary = payload["summary"]
    print(
        f"{category}/displaced: restored {summary['restored_best_step']:,}, "
        f"checked {summary['completed_steps']:,}, "
        f"best mean negative ELBO {summary['best_window_negative_elbo']:.3f}"
    )


# %%
# =============================================================================
# Displaced-start-only convergence figure
# =============================================================================
output_png = (
    OUTPUT_DIR
    / "led7_super_animal_valid_npl_alpha_convergence_displaced_only.png"
)
output_csv = (
    OUTPUT_DIR
    / "led7_super_animal_valid_npl_alpha_convergence_displaced_only_summary.csv"
)

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), sharey=False)

for ax, category in zip(axes, CATEGORY_SETTINGS):
    settings = CATEGORY_SETTINGS[category]
    payload = branch_payloads[category]
    summary = payload["summary"]
    convergence = payload["convergence"]

    restored_step = int(summary["restored_best_step"])
    checked_step = int(summary["completed_steps"])
    restored_row = convergence.loc[
        convergence["end_step"].astype(int) == restored_step
    ].iloc[-1]

    ax.plot(
        convergence["end_step"],
        convergence["mean_loss"],
        color=settings["color"],
        linewidth=1.25,
        label="displaced start",
    )
    ax.scatter(
        [restored_step],
        [restored_row["mean_loss"]],
        color="tab:green",
        s=24,
        zorder=4,
    )
    ax.axvline(restored_step, color="tab:green", linewidth=1.1)
    ax.axvline(
        checked_step,
        color="tab:red",
        linestyle="--",
        linewidth=1.1,
    )
    ax.set_title(
        f"{settings['label']}\n"
        f"restored {restored_step // 1000}k; checked {checked_step // 1000}k"
    )
    ax.set_xlabel("SVI step")
    ax.set_ylabel("negative ELBO")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[
            Line2D([0], [0], color=settings["color"], linewidth=1.25),
            Line2D([0], [0], color="tab:green", linewidth=1.1),
            Line2D(
                [0],
                [0],
                color="tab:red",
                linestyle="--",
                linewidth=1.1,
            ),
        ],
        labels=["displaced start", "restored best", "final checked"],
        frameon=False,
        fontsize=8,
        loc="upper right",
    )

fig.suptitle(
    "LED7 super-animal valid RT+choice NPL+alpha fits: displaced start only",
    y=1.02,
)
fig.tight_layout()
fig.savefig(output_png, dpi=220, bbox_inches="tight")

pd.DataFrame(summary_rows).to_csv(output_csv, index=False)

print(f"Saved {output_png.resolve()}")
print(f"Saved {output_csv.resolve()}")
