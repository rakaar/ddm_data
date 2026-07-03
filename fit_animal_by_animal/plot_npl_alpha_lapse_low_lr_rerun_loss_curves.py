# %%
"""
Plot NPL+alpha+lapse low-LR rerun loss curves.

This produces:

1. A 2 x 3 grid for the six animals rerun with lower learning rate.
2. A 5 x 6 hybrid grid for all animals, using low-LR reruns where available
   and the original min-50k NPL+alpha+lapse run for the rest.
"""

# %%
# =============================================================================
# Parameters
# =============================================================================
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ORIGINAL_ROOT = Path(
    os.environ.get(
        "NPL_ALPHA_LAPSE_ORIGINAL_ROOT",
        str(SCRIPT_DIR / "numpyro_svi_npl_alpha_lapse_condition_delay_patience12_min50k_restore_best_outputs"),
    )
).expanduser()
LOW_LR_ROOT = Path(
    os.environ.get(
        "NPL_ALPHA_LAPSE_LOW_LR_ROOT",
        str(
            SCRIPT_DIR
            / "numpyro_svi_npl_alpha_lapse_condition_delay_low_lr_patience12_min12k_restore_best_reruns"
        ),
    )
).expanduser()
LOW_LR_LEDGER = LOW_LR_ROOT / "_batch_logs" / "low_lr_firstwin_rerun_status.csv"
ORIGINAL_LEDGER = ORIGINAL_ROOT / "_batch_logs" / "batch_run_status.csv"

SUMMARY_DIR = LOW_LR_ROOT / "summary_figures"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

LOW_LR_FIG_PATH = SUMMARY_DIR / "npl_alpha_lapse_low_lr_firstwin_rerun_loss_grid.png"
HYBRID_FIG_PATH = SUMMARY_DIR / "npl_alpha_lapse_hybrid_low_lr_firstwin_loss_grid.png"
LOW_LR_SUMMARY_CSV = SUMMARY_DIR / "npl_alpha_lapse_low_lr_firstwin_rerun_loss_summary.csv"
HYBRID_SUMMARY_CSV = SUMMARY_DIR / "npl_alpha_lapse_hybrid_low_lr_firstwin_loss_summary.csv"

LOSS_COLOR = "0.15"
BEST_COLOR = "tab:green"
STOP_COLOR = "tab:red"
LOW_LR_TITLE_COLOR = "tab:blue"


# %%
# =============================================================================
# Helpers
# =============================================================================
def load_ledger(path, expected_rows=None):
    if not path.exists():
        raise FileNotFoundError(path)
    ledger_df = pd.read_csv(path)
    if expected_rows is not None and len(ledger_df) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows in {path}, found {len(ledger_df)}.")

    accepted_statuses = {"completed", "skipped_existing"}
    bad_status = ledger_df[~ledger_df["status"].isin(accepted_statuses)]
    if not bad_status.empty:
        raise RuntimeError("Some animals are not completed or skipped-existing:\n" + bad_status.to_string(index=False))
    return ledger_df


def load_payload(root, batch, animal, source_label):
    output_dir = root / f"{batch}_{animal}"
    convergence_csv = output_dir / "main_fullrank_convergence_checks.csv"
    if not convergence_csv.exists():
        raise FileNotFoundError(convergence_csv)

    conv_df = pd.read_csv(convergence_csv).sort_values("end_step").reset_index(drop=True)
    final_row = conv_df.iloc[-1]
    checked_step = int(final_row["end_step"])
    best_step = int(final_row["best_end_step_so_far"])
    best_loss = float(final_row["best_mean_loss_so_far"])
    final_loss = float(final_row["mean_loss"])
    no_improve_windows = int(final_row["no_improve_window_count"])

    if int(final_row.get("n_nonfinite", 0)) > 0:
        stop_reason = "nonfinite_loss"
    elif bool(final_row.get("no_improve_stop_candidate", False)):
        stop_reason = "patience_restore_best_12_windows"
    elif bool(final_row.get("early_stop_candidate", False)):
        stop_reason = "stable_12_windows"
    else:
        stop_reason = "max_steps_or_manual_stop"

    return {
        "label": f"{batch}/{animal}",
        "batch_name": batch,
        "animal": int(animal),
        "source": source_label,
        "steps": conv_df["end_step"].to_numpy(dtype=float),
        "mean_loss": conv_df["mean_loss"].to_numpy(dtype=float),
        "best_step": best_step,
        "checked_step": checked_step,
        "best_loss": best_loss,
        "final_loss": final_loss,
        "loss_rebound_from_best": final_loss - best_loss,
        "no_improve_windows": no_improve_windows,
        "stop_reason": stop_reason,
        "n_windows": len(conv_df),
    }


def payloads_to_summary(payloads):
    rows = []
    for idx, payload in enumerate(payloads, start=1):
        rows.append(
            {
                "plot_index": idx,
                "batch_name": payload["batch_name"],
                "animal": payload["animal"],
                "source": payload["source"],
                "stop_reason": payload["stop_reason"],
                "best_end_step": payload["best_step"],
                "checked_end_step": payload["checked_step"],
                "best_mean_loss": payload["best_loss"],
                "final_checked_mean_loss": payload["final_loss"],
                "loss_rebound_from_best": payload["loss_rebound_from_best"],
                "no_improve_windows": payload["no_improve_windows"],
                "n_windows": payload["n_windows"],
            }
        )
    return pd.DataFrame(rows)


def plot_grid(payloads, n_rows, n_cols, fig_path, title):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 2.45 * n_rows), sharex=False, sharey=False)
    axes = np.asarray(axes).ravel()

    for ax, payload in zip(axes, payloads):
        ax.plot(payload["steps"], payload["mean_loss"], color=LOSS_COLOR, lw=1.1)
        ax.axvline(payload["best_step"], color=BEST_COLOR, lw=1.4)
        ax.axvline(payload["checked_step"], color=STOP_COLOR, lw=1.4, ls="--")
        ax.scatter([payload["best_step"]], [payload["best_loss"]], s=18, color=BEST_COLOR, zorder=3)
        title_lines = [
            f"{payload['label']} ({payload['source']})",
            f"min {payload['best_step'] / 1000:.0f}k, checked {payload['checked_step'] / 1000:.0f}k",
        ]
        if payload["best_step"] == payload["checked_step"]:
            title_lines.append(f"no >0.1%: {payload['no_improve_windows']}w")
        title_color = LOW_LR_TITLE_COLOR if payload["source"] == "low_lr" else "black"
        ax.set_title("\n".join(title_lines), fontsize=8, color=title_color)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(alpha=0.18, lw=0.5)
        ax.set_xlabel("SVI step", fontsize=7)
        ax.set_ylabel("mean -ELBO", fontsize=7)

    for ax in axes[len(payloads) :]:
        ax.axis("off")

    legend_handles = [
        Line2D([0], [0], color=LOSS_COLOR, lw=1.4, label="1k-window mean -ELBO"),
        Line2D([0], [0], color=BEST_COLOR, lw=1.6, label="restored-best checkpoint"),
        Line2D([0], [0], color=STOP_COLOR, lw=1.6, ls="--", label="final checked step"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    fig.suptitle(title, y=0.995, fontsize=14)
    fig.tight_layout(rect=[0, 0.05, 1, 0.955])
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure: {fig_path}")


# %%
# =============================================================================
# Load rerun payloads
# =============================================================================
low_lr_ledger_df = load_ledger(LOW_LR_LEDGER, expected_rows=6)
low_lr_payloads = []
for _, ledger_row in low_lr_ledger_df.sort_values("run_index").iterrows():
    low_lr_payloads.append(
        load_payload(
            LOW_LR_ROOT,
            str(ledger_row["batch_name"]),
            int(ledger_row["animal"]),
            "low_lr",
        )
    )

low_lr_summary_df = payloads_to_summary(low_lr_payloads)
low_lr_summary_df.to_csv(LOW_LR_SUMMARY_CSV, index=False)
print("Low-LR rerun summaries:")
print(
    low_lr_summary_df[
        [
            "batch_name",
            "animal",
            "best_end_step",
            "checked_end_step",
            "loss_rebound_from_best",
            "no_improve_windows",
        ]
    ].to_string(index=False)
)
print(f"Saved summary CSV: {LOW_LR_SUMMARY_CSV}")

plot_grid(
    low_lr_payloads,
    n_rows=2,
    n_cols=3,
    fig_path=LOW_LR_FIG_PATH,
    title="Low-LR reruns for first-window-best NPL+alpha+lapse SVI loss curves",
)


# %%
# =============================================================================
# Build hybrid all-animal payloads: low-LR reruns replace original panels
# =============================================================================
original_ledger_df = load_ledger(ORIGINAL_LEDGER, expected_rows=30)
low_lr_pairs = {(payload["batch_name"], payload["animal"]) for payload in low_lr_payloads}

hybrid_payloads = []
for _, ledger_row in original_ledger_df.sort_values("run_index").iterrows():
    batch = str(ledger_row["batch_name"])
    animal = int(ledger_row["animal"])
    if (batch, animal) in low_lr_pairs:
        hybrid_payloads.append(load_payload(LOW_LR_ROOT, batch, animal, "low_lr"))
    else:
        hybrid_payloads.append(load_payload(ORIGINAL_ROOT, batch, animal, "original"))

hybrid_summary_df = payloads_to_summary(hybrid_payloads)
hybrid_summary_df.to_csv(HYBRID_SUMMARY_CSV, index=False)
print("\nHybrid all-animal summaries:")
print(
    hybrid_summary_df[
        [
            "batch_name",
            "animal",
            "source",
            "best_end_step",
            "checked_end_step",
            "loss_rebound_from_best",
            "no_improve_windows",
        ]
    ].to_string(index=False)
)
print(f"Saved summary CSV: {HYBRID_SUMMARY_CSV}")

plot_grid(
    hybrid_payloads,
    n_rows=5,
    n_cols=6,
    fig_path=HYBRID_FIG_PATH,
    title="Hybrid NPL+alpha+lapse SVI loss curves: low-LR reruns replace first-window-best animals",
)

# %%
