# %%
"""Compare warm and deliberately displaced initializations for LED7/98 OFF."""

# %%
# =============================================================================
# Editable paths and plotting settings
# =============================================================================
from pathlib import Path
import json
import pickle


SCRIPT_DIR = Path(__file__).resolve().parent
WARM_ROOT = (
    SCRIPT_DIR
    / "numpyro_svi_proactive_led_step_jump_npl_alpha_valid_led7_98_pilot_outputs"
    / "off"
)
DISPLACED_ROOT = (
    SCRIPT_DIR
    / "numpyro_svi_proactive_led_step_jump_npl_alpha_valid_"
    "led7_98_displaced_init_recovery_outputs"
    / "off"
)
WARM_DIAGNOSTIC_ROOT = WARM_ROOT.parent / "diagnostics"
DISPLACED_DIAGNOSTIC_ROOT = DISPLACED_ROOT.parent / "diagnostics"
OUTPUT_DIR = DISPLACED_DIAGNOSTIC_ROOT
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOSS_XMAX = 25_000
ABL_LEVELS = [20, 40, 60]
ABS_ILD_LEVELS = [1, 2, 4, 8, 16]
GLOBAL_PARAMETERS = [
    ("rate_lambda", r"$\lambda'$", 1.0),
    ("T_0", r"$T_0$ (ms)", 1000.0),
    ("theta_E", r"$\theta_E$", 1.0),
    ("w", r"$w$", 1.0),
    ("del_go", r"$t_{motor}$ (ms)", 1000.0),
    ("rate_norm_l", r"$\ell$", 1.0),
    ("alpha", r"$\alpha$", 1.0),
]


# %%
# =============================================================================
# Imports and source loading
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_fit(root):
    with (root / "run_summary.json").open() as handle:
        summary = json.load(handle)
    with np.load(root / "main_fullrank_posterior_samples.npz") as saved:
        posterior = {name: np.asarray(saved[name]) for name in saved.files}
    convergence = pd.read_csv(root / "main_fullrank_convergence_checks.csv")
    if summary.get("status") != "complete":
        raise RuntimeError(f"Incomplete fit at {root}: {summary.get('status')!r}")
    if not all(np.all(np.isfinite(values)) for values in posterior.values()):
        raise RuntimeError(f"Non-finite posterior samples at {root}")
    return summary, posterior, convergence


warm_summary, warm_posterior, warm_convergence = load_fit(WARM_ROOT)
displaced_summary, displaced_posterior, displaced_convergence = load_fit(DISPLACED_ROOT)

if displaced_summary["config"].get("initialization_mode") != "displaced":
    raise RuntimeError("The recovery fit was not initialized in displaced mode.")
if warm_summary["valid_trial_count"] != displaced_summary["valid_trial_count"]:
    raise RuntimeError("Warm and displaced fits used different trial counts.")

with (WARM_DIAGNOSTIC_ROOT / "led7_98_valid_npl_alpha_pilot_rtd_payload.pkl").open(
    "rb"
) as handle:
    warm_rtd = pickle.load(handle)["categories"]["off"]["cells"]
with (
    DISPLACED_DIAGNOSTIC_ROOT / "led7_98_valid_npl_alpha_pilot_rtd_payload.pkl"
).open("rb") as handle:
    displaced_rtd = pickle.load(handle)["categories"]["off"]["cells"]


# %%
# =============================================================================
# Loss and global-parameter recovery figure
# =============================================================================
recovery_png = OUTPUT_DIR / "led7_98_off_valid_npl_alpha_displaced_init_recovery.png"
fig, axes = plt.subplots(2, 4, figsize=(13.0, 6.7))

loss_ax = axes[0, 0]
for convergence, summary, color, label in [
    (warm_convergence, warm_summary, "tab:blue", "posterior warm start"),
    (displaced_convergence, displaced_summary, "tab:orange", "displaced start"),
]:
    visible = convergence["end_step"] <= LOSS_XMAX
    loss_ax.plot(
        convergence.loc[visible, "end_step"],
        convergence.loc[visible, "mean_loss"],
        marker="o",
        markersize=3.0,
        linewidth=1.1,
        color=color,
        label=label,
    )
    loss_ax.axvline(
        summary["restored_best_step"],
        color=color,
        linestyle="--",
        linewidth=1.0,
        alpha=0.8,
    )
loss_ax.set_title("1k-window negative ELBO")
loss_ax.set_xlabel("SVI step")
loss_ax.set_ylabel("negative ELBO")
loss_ax.set_xlim(0, LOSS_XMAX)
loss_ax.legend(frameon=False, fontsize=7.5, loc="upper right")

parameter_rows = []
for ax, (name, label, scale) in zip(axes.flat[1:], GLOBAL_PARAMETERS):
    for x, posterior, color, marker, source in [
        (0, warm_posterior, "tab:blue", "o", "posterior warm start"),
        (1, displaced_posterior, "tab:orange", "s", "displaced start"),
    ]:
        values = np.asarray(posterior[name], dtype=float) * scale
        mean = float(np.mean(values))
        low, high = np.quantile(values, [0.025, 0.975])
        ax.errorbar(
            x,
            mean,
            yerr=[[mean - low], [high - mean]],
            color=color,
            marker=marker,
            markersize=4.5,
            linewidth=1.0,
            capsize=2.5,
        )
        parameter_rows.append(
            {
                "source": source,
                "parameter": name,
                "mean": mean,
                "q025": float(low),
                "q975": float(high),
            }
        )
    ax.set_title(label)
    ax.set_xticks([0, 1], ["warm", "displaced"], rotation=20, ha="right")
    ax.set_xlim(-0.45, 1.45)

for ax in axes.flat:
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("LED7/98 LED-OFF initialization recovery check", y=1.01)
fig.tight_layout()
fig.savefig(recovery_png, dpi=220, bbox_inches="tight")


# %%
# =============================================================================
# RTD overlay and curve-difference metrics
# =============================================================================
rtd_png = OUTPUT_DIR / "led7_98_off_valid_npl_alpha_initialization_rtd_overlay.png"
metric_rows = []
max_density = 0.0

for ABL in ABL_LEVELS:
    for abs_ILD in ABS_ILD_LEVELS:
        key = f"ABL{ABL}_absILD{abs_ILD}"
        warm_cell = warm_rtd[key]
        displaced_cell = displaced_rtd[key]
        warm_time = np.asarray(warm_cell["model_rt_s"], dtype=float)
        displaced_time = np.asarray(displaced_cell["model_rt_s"], dtype=float)
        if not np.array_equal(warm_time, displaced_time):
            raise RuntimeError(f"RT grids differ for {key}.")
        if not np.array_equal(
            np.asarray(warm_cell["data_density"]),
            np.asarray(displaced_cell["data_density"]),
        ):
            raise RuntimeError(f"Empirical histograms differ for {key}.")

        warm_density = np.asarray(warm_cell["model_density"], dtype=float)
        displaced_density = np.asarray(displaced_cell["model_density"], dtype=float)
        absolute_difference = np.abs(displaced_density - warm_density)
        metric_rows.append(
            {
                "ABL": ABL,
                "abs_ILD": abs_ILD,
                "n_trials": warm_cell["n_trials"],
                "integrated_absolute_difference": float(
                    np.trapz(absolute_difference, warm_time)
                ),
                "maximum_absolute_density_difference": float(
                    np.max(absolute_difference)
                ),
                "displayed_mass_difference": float(
                    displaced_cell["model_displayed_mass"]
                    - warm_cell["model_displayed_mass"]
                ),
            }
        )
        max_density = max(
            max_density,
            float(np.max(warm_cell["data_density"])),
            float(np.max(warm_density)),
            float(np.max(displaced_density)),
        )

fig, axes = plt.subplots(3, 5, figsize=(14.5, 8.0), sharex=True, sharey=True)
for row_index, ABL in enumerate(ABL_LEVELS):
    for col_index, abs_ILD in enumerate(ABS_ILD_LEVELS):
        ax = axes[row_index, col_index]
        key = f"ABL{ABL}_absILD{abs_ILD}"
        warm_cell = warm_rtd[key]
        displaced_cell = displaced_rtd[key]
        ax.step(
            warm_cell["data_centers_s"],
            warm_cell["data_density"],
            where="mid",
            color="black",
            alpha=0.55,
            linewidth=0.9,
            label="data" if row_index == 0 and col_index == 0 else None,
        )
        ax.plot(
            warm_cell["model_rt_s"],
            warm_cell["model_density"],
            color="tab:blue",
            linewidth=1.35,
            label="posterior warm start" if row_index == 0 and col_index == 0 else None,
        )
        ax.plot(
            displaced_cell["model_rt_s"],
            displaced_cell["model_density"],
            color="tab:orange",
            linestyle="--",
            linewidth=2.2,
            alpha=0.60,
            label="displaced start" if row_index == 0 and col_index == 0 else None,
        )
        if row_index == 0:
            ax.set_title(rf"$|ILD|={abs_ILD}$ dB")
        if col_index == 0:
            ax.set_ylabel(f"ABL {ABL} dB\ndensity")
        if row_index == 2:
            ax.set_xlabel("RT wrt stimulus (s)")
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(0.0, 1.06 * max_density)
        ax.spines[["top", "right"]].set_visible(False)

axes[0, 0].legend(frameon=False, fontsize=7.5, loc="upper left")
fig.suptitle("LED7/98 LED-OFF RTD initialization recovery", y=1.01)
fig.tight_layout()
fig.savefig(rtd_png, dpi=220, bbox_inches="tight")


# %%
# =============================================================================
# Save the numerical audit
# =============================================================================
parameter_csv = OUTPUT_DIR / "led7_98_off_valid_npl_alpha_initialization_parameter_summary.csv"
rtd_metric_csv = OUTPUT_DIR / "led7_98_off_valid_npl_alpha_initialization_rtd_metrics.csv"
audit_json = OUTPUT_DIR / "led7_98_off_valid_npl_alpha_initialization_recovery_summary.json"

parameter_table = pd.DataFrame(parameter_rows)
rtd_metric_table = pd.DataFrame(metric_rows)
parameter_table.to_csv(parameter_csv, index=False)
rtd_metric_table.to_csv(rtd_metric_csv, index=False)

warm_delay_means = np.mean(warm_posterior["t_E_aff"], axis=0)
displaced_delay_means = np.mean(displaced_posterior["t_E_aff"], axis=0)
delay_difference = displaced_delay_means - warm_delay_means
summary = {
    "warm_fit_root": str(WARM_ROOT.resolve()),
    "displaced_fit_root": str(DISPLACED_ROOT.resolve()),
    "warm_best_step": int(warm_summary["restored_best_step"]),
    "displaced_best_step": int(displaced_summary["restored_best_step"]),
    "warm_best_window_negative_elbo": float(warm_convergence["mean_loss"].min()),
    "displaced_best_window_negative_elbo": float(
        displaced_convergence["mean_loss"].min()
    ),
    "delay_mean_rmse_ms": float(1000.0 * np.sqrt(np.mean(delay_difference**2))),
    "delay_mean_max_abs_difference_ms": float(
        1000.0 * np.max(np.abs(delay_difference))
    ),
    "delay_mean_correlation": float(
        np.corrcoef(warm_delay_means, displaced_delay_means)[0, 1]
    ),
    "mean_cell_integrated_absolute_rtd_difference": float(
        rtd_metric_table["integrated_absolute_difference"].mean()
    ),
    "max_cell_integrated_absolute_rtd_difference": float(
        rtd_metric_table["integrated_absolute_difference"].max()
    ),
    "mean_abs_displayed_mass_difference": float(
        rtd_metric_table["displayed_mass_difference"].abs().mean()
    ),
    "max_abs_displayed_mass_difference": float(
        rtd_metric_table["displayed_mass_difference"].abs().max()
    ),
}
with audit_json.open("w") as handle:
    json.dump(summary, handle, indent=2)

print(json.dumps(summary, indent=2))
print(f"Recovery figure: {recovery_png}")
print(f"RTD overlay: {rtd_png}")
print(f"Parameter summary: {parameter_csv}")
print(f"RTD metrics: {rtd_metric_csv}")
print(f"Audit summary: {audit_json}")
