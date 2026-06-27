# %%
"""
Compare original old-stop big SVI parameters against patience8 restore-best.

The new-rule parameter summaries come from the existing 50k no-early-stop audit
outputs. That run returned/sampled the tracked best state, and the minima-method
diagnostic verifies that `patience8_restore_best` selects that same best window
for these six audit animals.
"""

# %%
# =============================================================================
# Parameters
# =============================================================================
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ORIGINAL_OUTPUT_ROOT = SCRIPT_DIR / "svi_big_gamma_omega_delay_all_animals_outputs"
AUDIT_OUTPUT_ROOT = (
    SCRIPT_DIR
    / "svi_big_gamma_omega_delay_convergence_audit_outputs"
    / "no_early_stop_50k"
)
MINIMA_SUMMARY_CSV = (
    AUDIT_OUTPUT_ROOT
    / "loss_minima_method_diagnostics"
    / "big_gamma_omega_delay_loss_minima_method_summary.csv"
)
OUTPUT_DIR = AUDIT_OUTPUT_ROOT / "old_stop_vs_patience8_restore_best_diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SELECTED_PAIRS = [
    ("SD", 49),
    ("LED8", 105),
    ("SD", 52),
    ("LED6", 81),
    ("LED34", 63),
    ("LED7", 103),
]

ABL_COLORS = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}

LOSS_FIG = OUTPUT_DIR / "old_stop_vs_patience8_restore_best_loss_curves.png"
PARAM_FIG = OUTPUT_DIR / "old_stop_vs_patience8_restore_best_gamma_omega_curves.png"
SUMMARY_CSV = OUTPUT_DIR / "old_stop_vs_patience8_restore_best_summary.csv"
CONDITION_PARAM_CSV = OUTPUT_DIR / "old_stop_vs_patience8_restore_best_condition_params.csv"


# %%
# =============================================================================
# Helpers
# =============================================================================
def animal_prefix(batch: str, animal: int) -> str:
    return f"{batch}_{animal}_big_gamma_omega_delay"


def animal_paths(root: Path, batch: str, animal: int):
    output_dir = root / f"{batch}_{animal}"
    prefix = animal_prefix(batch, animal)
    return {
        "convergence": output_dir / f"{prefix}_convergence_checks.csv",
        "condition_summary": output_dir / f"{prefix}_condition_summary.csv",
    }


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def rmse(values: pd.Series) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values**2)))


# %%
# =============================================================================
# Load old/new summaries and compute diagnostics
# =============================================================================
minima_summary = load_csv(MINIMA_SUMMARY_CSV)
summary_rows = []
condition_rows = []
loss_payloads = []
param_payloads = []

for batch, animal in SELECTED_PAIRS:
    label = f"{batch}/{animal}"
    original_paths = animal_paths(ORIGINAL_OUTPUT_ROOT, batch, animal)
    audit_paths = animal_paths(AUDIT_OUTPUT_ROOT, batch, animal)

    old_conv = load_csv(original_paths["convergence"]).sort_values("end_step").reset_index(drop=True)
    new_conv = load_csv(audit_paths["convergence"]).sort_values("end_step").reset_index(drop=True)
    old_params = load_csv(original_paths["condition_summary"])
    new_params = load_csv(audit_paths["condition_summary"])

    patience_rows = minima_summary[
        (minima_summary["animal_label"] == label)
        & (minima_summary["method"] == "patience8_restore_best")
    ]
    if len(patience_rows) != 1:
        raise RuntimeError(f"{label}: expected one patience8_restore_best row in {MINIMA_SUMMARY_CSV}")
    patience_row = patience_rows.iloc[0]

    if abs(float(patience_row["delta_from_raw_best_per_trial"])) > 1e-12:
        raise RuntimeError(
            f"{label}: patience8_restore_best does not select the audit raw-best checkpoint; "
            f"delta/trial={patience_row['delta_from_raw_best_per_trial']}"
        )

    old_stop_step = int(old_conv["end_step"].max())
    new_stop_step = int(patience_row["stop_step"])
    new_selected_step = int(patience_row["selected_step"])

    old_loss_for_plot = old_conv[old_conv["end_step"] <= old_stop_step].copy()
    new_loss_for_plot = new_conv[new_conv["end_step"] <= new_stop_step].copy()

    merge_cols = ["batch_name", "animal", "condition_id", "ABL", "ILD"]
    merged = old_params.merge(
        new_params,
        on=merge_cols,
        suffixes=("_old", "_new"),
        validate="one_to_one",
    )
    if len(merged) != len(old_params):
        raise RuntimeError(f"{label}: old/new condition merge lost rows.")

    merged["gamma_delta"] = merged["gamma_mean_new"] - merged["gamma_mean_old"]
    merged["omega_delta"] = merged["omega_mean_new"] - merged["omega_mean_old"]

    for _, row in merged.iterrows():
        condition_rows.append(
            {
                "batch_name": batch,
                "animal": animal,
                "animal_label": label,
                "condition_id": int(row["condition_id"]),
                "ABL": int(row["ABL"]),
                "ILD": int(row["ILD"]),
                "gamma_old": float(row["gamma_mean_old"]),
                "gamma_new": float(row["gamma_mean_new"]),
                "gamma_delta": float(row["gamma_delta"]),
                "omega_old": float(row["omega_mean_old"]),
                "omega_new": float(row["omega_mean_new"]),
                "omega_delta": float(row["omega_delta"]),
            }
        )

    summary_rows.append(
        {
            "batch_name": batch,
            "animal": animal,
            "animal_label": label,
            "n_conditions": int(len(merged)),
            "n_trials": int(old_params["n_trials"].sum()),
            "old_stop_step": old_stop_step,
            "new_patience_stop_step": new_stop_step,
            "new_restore_best_step": new_selected_step,
            "gamma_rmse_new_minus_old": rmse(merged["gamma_delta"]),
            "omega_rmse_new_minus_old": rmse(merged["omega_delta"]),
            "gamma_max_abs_delta": float(np.max(np.abs(merged["gamma_delta"]))),
            "omega_max_abs_delta": float(np.max(np.abs(merged["omega_delta"]))),
        }
    )

    loss_payloads.append(
        {
            "label": label,
            "old_conv": old_loss_for_plot,
            "new_conv": new_loss_for_plot,
            "old_stop_step": old_stop_step,
            "new_stop_step": new_stop_step,
            "new_selected_step": new_selected_step,
        }
    )
    param_payloads.append({"label": label, "condition_params": merged})

summary_df = pd.DataFrame(summary_rows)
condition_df = pd.DataFrame(condition_rows)
summary_df.to_csv(SUMMARY_CSV, index=False)
condition_df.to_csv(CONDITION_PARAM_CSV, index=False)

print("Saved summaries:")
print(f"  {SUMMARY_CSV}")
print(f"  {CONDITION_PARAM_CSV}")
print("\nOld stop vs patience8 restore-best:")
print(
    summary_df[
        [
            "animal_label",
            "old_stop_step",
            "new_restore_best_step",
            "new_patience_stop_step",
            "gamma_rmse_new_minus_old",
            "omega_rmse_new_minus_old",
        ]
    ].to_string(index=False)
)


# %%
# =============================================================================
# Plot old/new stopping loss curves
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.0), sharex=False, sharey=False)
axes_flat = axes.ravel()

for ax, payload in zip(axes_flat, loss_payloads):
    old_conv = payload["old_conv"]
    new_conv = payload["new_conv"]

    ax.plot(
        old_conv["end_step"],
        old_conv["mean_loss"],
        color="tab:red",
        linestyle="-",
        linewidth=1.0,
        label="old rule",
    )
    ax.plot(
        new_conv["end_step"],
        new_conv["mean_loss"],
        color="tab:blue",
        linestyle="--",
        linewidth=3.0,
        alpha=0.4,
        label="patience8 restore-best run",
    )
    ax.axvline(payload["old_stop_step"], color="tab:red", linestyle="-", linewidth=1.2, alpha=0.9)
    ax.axvline(payload["new_stop_step"], color="tab:blue", linestyle="--", linewidth=1.8, alpha=0.6)
    ax.axvline(payload["new_selected_step"], color="tab:green", linestyle=":", linewidth=1.8, alpha=0.9)

    text = (
        f"old stop={payload['old_stop_step'] / 1000:.0f}k\n"
        f"new best={payload['new_selected_step'] / 1000:.0f}k\n"
        f"new stop={payload['new_stop_step'] / 1000:.0f}k"
    )
    ax.text(
        0.98,
        0.96,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "0.85", "alpha": 0.9},
    )

    ax.set_title(payload["label"])
    ax.set_xlabel("SVI step")
    ax.set_ylabel("1k-window mean negative ELBO")
    ax.grid(True, alpha=0.25)

loss_handles = [
    Line2D([0], [0], color="tab:red", lw=1.0, linestyle="-", label="old stopping rule"),
    Line2D([0], [0], color="tab:blue", lw=3.0, linestyle="--", alpha=0.4, label="patience8 run"),
    Line2D([0], [0], color="tab:green", lw=1.8, linestyle=":", label="new restored-best step"),
]
fig.legend(
    handles=loss_handles,
    frameon=False,
    fontsize=9,
    loc="upper right",
    bbox_to_anchor=(0.995, 1.02),
    ncol=3,
)
fig.suptitle("Old stopping rule vs patience8 restore-best loss curves", y=1.02)
fig.tight_layout()
fig.savefig(LOSS_FIG, dpi=200, bbox_inches="tight")
print(f"Saved loss figure: {LOSS_FIG}")


# %%
# =============================================================================
# Plot Gamma/Omega old/new condition curves
# =============================================================================
animal_labels = [f"{batch}/{animal}" for batch, animal in SELECTED_PAIRS]

fig, axes = plt.subplots(2, 6, figsize=(22, 7.0), sharex=False, sharey="row")
curve_specs = [
    ("gamma", "Gamma", "gamma_mean_old", "gamma_mean_new"),
    ("omega", "Omega", "omega_mean_old", "omega_mean_new"),
]
for col_idx, payload in enumerate(param_payloads):
    label = payload["label"]
    params = payload["condition_params"]
    for row_idx, (_, y_label, old_col, new_col) in enumerate(curve_specs):
        ax = axes[row_idx, col_idx]
        for abl in sorted(params["ABL"].unique()):
            abl_rows = params[params["ABL"] == abl].sort_values("ILD")
            color = ABL_COLORS.get(int(abl), "0.3")
            ax.plot(
                abl_rows["ILD"],
                abl_rows[old_col],
                color=color,
                linestyle="-",
                linewidth=1.0,
            )
            ax.plot(
                abl_rows["ILD"],
                abl_rows[new_col],
                color=color,
                linestyle="--",
                linewidth=3.0,
                alpha=0.4,
            )
        if row_idx == 0:
            ax.set_title(label)
        if col_idx == 0:
            ax.set_ylabel(y_label)
        if row_idx == 1:
            ax.set_xlabel("ILD")
        ax.grid(True, alpha=0.22)

param_handles = [
    Line2D([0], [0], color=ABL_COLORS[20], lw=1.0, label="ABL 20"),
    Line2D([0], [0], color=ABL_COLORS[40], lw=1.0, label="ABL 40"),
    Line2D([0], [0], color=ABL_COLORS[60], lw=1.0, label="ABL 60"),
    Line2D([0], [0], color="0.15", lw=1.0, linestyle="-", label="old rule"),
    Line2D([0], [0], color="0.15", lw=3.0, linestyle="--", alpha=0.4, label="patience8 restore-best"),
]
fig.legend(
    handles=param_handles,
    frameon=False,
    fontsize=9,
    loc="upper right",
    bbox_to_anchor=(0.995, 1.02),
    ncol=5,
)
fig.suptitle("Gamma/Omega condition means: old stop vs patience8 restore-best", y=1.04)
fig.tight_layout()
fig.savefig(PARAM_FIG, dpi=200, bbox_inches="tight")
print(f"Saved parameter figure: {PARAM_FIG}")

plt.show()

# %%
