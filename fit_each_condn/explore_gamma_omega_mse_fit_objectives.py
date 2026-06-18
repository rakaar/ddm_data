# %%
"""
Explore Gamma/Omega alpha-model fits under different MSE objectives.

For each animal, fit the same NPL+alpha Gamma/Omega expression three ways:
1. minimize both Gamma and Omega residuals;
2. minimize only Gamma residuals and inspect Omega prediction;
3. minimize only Omega residuals and inspect Gamma prediction.
"""

# %%
# =============================================================================
# Imports
# =============================================================================
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.optimize import least_squares

from gamma_omega_alpha_utils import gamma_omega_alpha_model


# %%
# =============================================================================
# Parameters
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

COND_ANIMAL_VALUES_CSV = (
    SCRIPT_DIR
    / "condition_t_E_aff_fixed_gamma_omega_comparison"
    / "condition_gamma_omega_animal_values.csv"
)
FALLBACK_COND_ANIMAL_VALUES_CSV = (
    SCRIPT_DIR
    / "abl_specific_ild2_gamma_omega_comparison"
    / "condition_gamma_omega_animal_values.csv"
)

OUTPUT_DIR = SCRIPT_DIR / "gamma_omega_mse_fit_objective_exploration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIT_PARAMS_CSV = OUTPUT_DIR / "per_animal_fit_params_by_objective.csv"
CONDITION_PREDICTIONS_CSV = OUTPUT_DIR / "condition_grid_predictions_by_objective.csv"
CONDITION_AGREEMENT_CSV = OUTPUT_DIR / "condition_grid_agreement_by_objective.csv"
METRICS_CSV = OUTPUT_DIR / "condition_grid_metrics_by_objective.csv"
CONTINUOUS_SUMMARY_CSV = OUTPUT_DIR / "continuous_curve_summary_by_objective.csv"
CONDITION_SUMMARY_CSV = OUTPUT_DIR / "condition_gamma_omega_summary.csv"
FIG_PNG = OUTPUT_DIR / "gamma_omega_mse_fit_objective_exploration.png"
FIG_PDF = OUTPUT_DIR / "gamma_omega_mse_fit_objective_exploration.pdf"

ABLS = [20, 40, 60]
FITTED_ILDS = [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]
SMOOTH_ILDS = np.round(np.arange(-16.0, 16.0 + 0.05, 0.1), 10)
P_0 = 20e-6
COLORS = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}

PARAM_NAMES = ["rate_lambda", "ell", "alpha", "theta", "T_0"]
P0_FIT = np.array([2.0, 0.9, 0.5, 2.4, 100e-3])
LOWER_BOUNDS = np.array([0.1, 0.0, 0.0, 0.5, 1e-3])
UPPER_BOUNDS = np.array([6.0, 2.0, 5.0, 15.0, 2.0])

FIT_OBJECTIVES = [
    {
        "objective": "gamma_omega",
        "label": "fit Gamma + Omega",
        "fit_params": ["gamma", "omega"],
    },
    {
        "objective": "gamma_only",
        "label": "fit Gamma only",
        "fit_params": ["gamma"],
    },
    {
        "objective": "omega_only",
        "label": "fit Omega only",
        "fit_params": ["omega"],
    },
]

EXPECTED_N_CONDITION_ROWS = 864
EXPECTED_N_ANIMALS = 30


# %%
# =============================================================================
# Helpers
# =============================================================================
def mean_sem(values):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    n = int(np.sum(finite))
    if n == 0:
        return np.nan, np.nan, 0
    return float(np.nanmean(values)), float(np.nanstd(values) / np.sqrt(n)), n


def gamma_omega_rows(params, ild_grid):
    rows = []
    for abl in ABLS:
        gamma, omega = gamma_omega_alpha_model(
            abl,
            np.asarray(ild_grid, dtype=float),
            params["rate_lambda"],
            params["ell"],
            params["alpha"],
            params["theta"],
            params["T_0"],
            P_0,
        )
        for ild, curr_gamma, curr_omega in zip(ild_grid, gamma, omega):
            rows.append(
                {
                    "ABL": int(abl),
                    "ILD": float(ild),
                    "pred_gamma": float(curr_gamma),
                    "pred_omega": float(curr_omega),
                }
            )
    return pd.DataFrame(rows)


def params_from_fit_result(fit_result):
    return {name: float(value) for name, value in zip(PARAM_NAMES, fit_result.x)}


def fit_alpha_model(fit_abls, fit_ilds, target_gamma, target_omega, fit_params):
    gamma_scale = np.nanstd(target_gamma)
    omega_scale = np.nanstd(target_omega)
    if gamma_scale == 0 or not np.isfinite(gamma_scale):
        gamma_scale = 1.0
    if omega_scale == 0 or not np.isfinite(omega_scale):
        omega_scale = 1.0

    def residuals(params):
        rate_lambda, ell, alpha, theta, T_0 = params
        pred_gamma, pred_omega = gamma_omega_alpha_model(
            fit_abls,
            fit_ilds,
            rate_lambda,
            ell,
            alpha,
            theta,
            T_0,
            P_0,
        )
        pieces = []
        if "gamma" in fit_params:
            pieces.append((pred_gamma - target_gamma) / gamma_scale)
        if "omega" in fit_params:
            pieces.append((pred_omega - target_omega) / omega_scale)
        return np.concatenate(pieces)

    return least_squares(
        residuals,
        P0_FIT,
        bounds=(LOWER_BOUNDS, UPPER_BOUNDS),
    )


# %%
# =============================================================================
# Load condition Gamma/Omega values
# =============================================================================
if COND_ANIMAL_VALUES_CSV.exists():
    cond_path = COND_ANIMAL_VALUES_CSV
elif FALLBACK_COND_ANIMAL_VALUES_CSV.exists():
    cond_path = FALLBACK_COND_ANIMAL_VALUES_CSV
else:
    raise FileNotFoundError(
        "Could not find condition Gamma/Omega animal values CSV. "
        "Run compare_cond_gamma_omega_with_npl_alpha_condition_t_E_aff_fixed_delay.py first."
    )

cond_df = pd.read_csv(cond_path)
if len(cond_df) != EXPECTED_N_CONDITION_ROWS:
    raise RuntimeError(f"Expected {EXPECTED_N_CONDITION_ROWS} condition rows, found {len(cond_df)} in {cond_path}")

cond_df = cond_df.sort_values(["batch_name", "animal", "ABL", "ILD"]).reset_index(drop=True)
animal_df = cond_df[["batch_name", "animal"]].drop_duplicates().sort_values(["batch_name", "animal"])
if len(animal_df) != EXPECTED_N_ANIMALS:
    raise RuntimeError(f"Expected {EXPECTED_N_ANIMALS} animals, found {len(animal_df)}")

condition_summary_rows = []
for (abl, ild), group in cond_df.groupby(["ABL", "ILD"], sort=True):
    gamma_mean, gamma_sem, n_gamma = mean_sem(group["condition_gamma"])
    omega_mean, omega_sem, n_omega = mean_sem(group["condition_omega"])
    condition_summary_rows.append(
        {
            "ABL": int(abl),
            "ILD": int(ild),
            "condition_gamma_mean": gamma_mean,
            "condition_gamma_sem": gamma_sem,
            "n_condition_gamma": n_gamma,
            "condition_omega_mean": omega_mean,
            "condition_omega_sem": omega_sem,
            "n_condition_omega": n_omega,
        }
    )
condition_summary_df = pd.DataFrame(condition_summary_rows)
condition_summary_df.to_csv(CONDITION_SUMMARY_CSV, index=False)

print(f"Loaded condition Gamma/Omega values: {cond_path}")
print(f"Condition rows: {len(cond_df)}")
print("Condition counts at |ILD|=16:")
print(
    condition_summary_df[condition_summary_df["ILD"].abs() == 16][
        ["ABL", "ILD", "n_condition_gamma", "n_condition_omega"]
    ].to_string(index=False)
)


# %%
# =============================================================================
# Fit per-animal models under each objective
# =============================================================================
fit_rows = []
smooth_rows = []
condition_prediction_rows = []

for _, animal_row in animal_df.iterrows():
    batch_name = animal_row["batch_name"]
    animal = int(animal_row["animal"])
    animal_cond = cond_df[
        (cond_df["batch_name"] == batch_name)
        & (cond_df["animal"].astype(int) == animal)
    ].copy()
    finite = (
        np.isfinite(animal_cond["condition_gamma"].to_numpy(dtype=float))
        & np.isfinite(animal_cond["condition_omega"].to_numpy(dtype=float))
    )
    animal_cond = animal_cond.loc[finite]

    fit_abls = animal_cond["ABL"].to_numpy(dtype=float)
    fit_ilds = animal_cond["ILD"].to_numpy(dtype=float)
    target_gamma = animal_cond["condition_gamma"].to_numpy(dtype=float)
    target_omega = animal_cond["condition_omega"].to_numpy(dtype=float)

    for objective_config in FIT_OBJECTIVES:
        try:
            fit_result = fit_alpha_model(
                fit_abls,
                fit_ilds,
                target_gamma,
                target_omega,
                objective_config["fit_params"],
            )
            fit_params = params_from_fit_result(fit_result)
            fit_rows.append(
                {
                    "batch_name": batch_name,
                    "animal": animal,
                    "objective": objective_config["objective"],
                    "objective_label": objective_config["label"],
                    "fit_params": ",".join(objective_config["fit_params"]),
                    "success": bool(fit_result.success),
                    "n_points": len(animal_cond),
                    "cost": float(fit_result.cost),
                    "message": fit_result.message,
                    **fit_params,
                }
            )
            if not fit_result.success:
                continue

            smooth_df = gamma_omega_rows(fit_params, SMOOTH_ILDS)
            smooth_df.insert(0, "objective_label", objective_config["label"])
            smooth_df.insert(0, "objective", objective_config["objective"])
            smooth_df.insert(0, "animal", animal)
            smooth_df.insert(0, "batch_name", batch_name)
            smooth_rows.extend(smooth_df.to_dict("records"))

            condition_df = gamma_omega_rows(fit_params, FITTED_ILDS)
            condition_df["ILD"] = condition_df["ILD"].astype(int)
            condition_df.insert(0, "objective_label", objective_config["label"])
            condition_df.insert(0, "objective", objective_config["objective"])
            condition_df.insert(0, "animal", animal)
            condition_df.insert(0, "batch_name", batch_name)
            condition_prediction_rows.extend(condition_df.to_dict("records"))
        except Exception as exc:
            fit_rows.append(
                {
                    "batch_name": batch_name,
                    "animal": animal,
                    "objective": objective_config["objective"],
                    "objective_label": objective_config["label"],
                    "fit_params": ",".join(objective_config["fit_params"]),
                    "success": False,
                    "n_points": len(animal_cond),
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )

fit_df = pd.DataFrame(fit_rows).sort_values(["objective", "batch_name", "animal"]).reset_index(drop=True)
smooth_df = pd.DataFrame(smooth_rows).sort_values(["objective", "batch_name", "animal", "ABL", "ILD"]).reset_index(drop=True)
condition_prediction_df = (
    pd.DataFrame(condition_prediction_rows)
    .sort_values(["objective", "batch_name", "animal", "ABL", "ILD"])
    .reset_index(drop=True)
)

fit_df.to_csv(FIT_PARAMS_CSV, index=False)
condition_prediction_df.to_csv(CONDITION_PREDICTIONS_CSV, index=False)
print(f"Saved fit parameters: {FIT_PARAMS_CSV}")
print(f"Saved condition-grid predictions: {CONDITION_PREDICTIONS_CSV}")
print("Successful fits by objective:")
print(fit_df.groupby("objective")["success"].sum().to_string())

if fit_df["success"].sum() == 0:
    raise RuntimeError("No MSE objective fits succeeded.")


# %%
# =============================================================================
# Summarize curves and condition-grid agreement
# =============================================================================
continuous_summary_rows = []
for (objective, objective_label, abl, ild), group in smooth_df.groupby(
    ["objective", "objective_label", "ABL", "ILD"], sort=True
):
    gamma_mean, gamma_sem, n_gamma = mean_sem(group["pred_gamma"])
    omega_mean, omega_sem, n_omega = mean_sem(group["pred_omega"])
    continuous_summary_rows.append(
        {
            "objective": objective,
            "objective_label": objective_label,
            "ABL": int(abl),
            "ILD": float(ild),
            "pred_gamma_mean": gamma_mean,
            "pred_gamma_sem": gamma_sem,
            "n_pred_gamma": n_gamma,
            "pred_omega_mean": omega_mean,
            "pred_omega_sem": omega_sem,
            "n_pred_omega": n_omega,
        }
    )
continuous_summary_df = pd.DataFrame(continuous_summary_rows)
continuous_summary_df.to_csv(CONTINUOUS_SUMMARY_CSV, index=False)
print(f"Saved continuous curve summary: {CONTINUOUS_SUMMARY_CSV}")

agreement_df = cond_df.merge(
    condition_prediction_df,
    on=["batch_name", "animal", "ABL", "ILD"],
    how="left",
    validate="one_to_many",
)
if agreement_df["pred_gamma"].isna().any() or agreement_df["pred_omega"].isna().any():
    missing = agreement_df[agreement_df["pred_gamma"].isna() | agreement_df["pred_omega"].isna()]
    raise RuntimeError(f"Missing model predictions for {len(missing)} condition rows")
agreement_df.to_csv(CONDITION_AGREEMENT_CSV, index=False)
print(f"Saved condition-grid agreement rows: {CONDITION_AGREEMENT_CSV}")

metric_rows = []
for objective_config in FIT_OBJECTIVES:
    objective = objective_config["objective"]
    objective_label = objective_config["label"]
    objective_subset = agreement_df[agreement_df["objective"] == objective]
    for param in ["gamma", "omega"]:
        for abl in ABLS + ["all"]:
            subset = objective_subset if abl == "all" else objective_subset[objective_subset["ABL"] == abl]
            y = subset[f"condition_{param}"].to_numpy(dtype=float)
            x = subset[f"pred_{param}"].to_numpy(dtype=float)
            finite = np.isfinite(x) & np.isfinite(y)
            if np.sum(finite) >= 2:
                diff = x[finite] - y[finite]
                rmse = float(np.sqrt(np.mean(diff**2)))
                mean_model_minus_condition = float(np.mean(diff))
                pearson_r = float(np.corrcoef(x[finite], y[finite])[0, 1])
            else:
                rmse = mean_model_minus_condition = pearson_r = np.nan
            metric_rows.append(
                {
                    "objective": objective,
                    "objective_label": objective_label,
                    "parameter": param,
                    "ABL": abl,
                    "n_condition_rows": int(np.sum(finite)),
                    "in_fit_objective": param in objective_config["fit_params"],
                    "rmse": rmse,
                    "mean_model_minus_condition": mean_model_minus_condition,
                    "pearson_r": pearson_r,
                }
            )

metrics_df = pd.DataFrame(metric_rows)
metrics_df.to_csv(METRICS_CSV, index=False)
print(f"Saved objective metrics: {METRICS_CSV}")
print(metrics_df[metrics_df["ABL"].astype(str) == "all"].to_string(index=False))


# %%
# =============================================================================
# Plot 3 x 2 objective diagnostics
# =============================================================================
display_limits = {}
for param in ["gamma", "omega"]:
    values_for_limits = []
    values_for_limits.extend(condition_summary_df[f"condition_{param}_mean"].to_numpy(dtype=float))
    values_for_limits.extend(
        (
            condition_summary_df[f"condition_{param}_mean"].to_numpy(dtype=float)
            - condition_summary_df[f"condition_{param}_sem"].to_numpy(dtype=float)
        )
    )
    values_for_limits.extend(
        (
            condition_summary_df[f"condition_{param}_mean"].to_numpy(dtype=float)
            + condition_summary_df[f"condition_{param}_sem"].to_numpy(dtype=float)
        )
    )

    model_for_limits = continuous_summary_df.copy()
    # Gamma-only fits can make held-out Omega explode; keep the main Omega scale readable.
    if param == "omega":
        model_for_limits = model_for_limits[model_for_limits["objective"] != "gamma_only"]
    values_for_limits.extend(model_for_limits[f"pred_{param}_mean"].to_numpy(dtype=float))
    values_for_limits.extend(
        (
            model_for_limits[f"pred_{param}_mean"].to_numpy(dtype=float)
            - model_for_limits[f"pred_{param}_sem"].to_numpy(dtype=float)
        )
    )
    values_for_limits.extend(
        (
            model_for_limits[f"pred_{param}_mean"].to_numpy(dtype=float)
            + model_for_limits[f"pred_{param}_sem"].to_numpy(dtype=float)
        )
    )

    values_for_limits = np.asarray(values_for_limits, dtype=float)
    values_for_limits = values_for_limits[np.isfinite(values_for_limits)]
    y_min = float(np.nanmin(values_for_limits))
    y_max = float(np.nanmax(values_for_limits))
    pad = 0.08 * (y_max - y_min)
    if pad == 0 or not np.isfinite(pad):
        pad = 1.0
    display_limits[param] = (y_min - pad, y_max + pad)

fig, axes = plt.subplots(3, 2, figsize=(13.5, 11.0), sharex=True)
column_specs = [
    ("gamma", "Gamma"),
    ("omega", "Omega"),
]

for row_idx, objective_config in enumerate(FIT_OBJECTIVES):
    objective = objective_config["objective"]
    objective_label = objective_config["label"]
    for col_idx, (param, param_label) in enumerate(column_specs):
        ax = axes[row_idx, col_idx]
        for abl in ABLS:
            color = COLORS[abl]
            cond = condition_summary_df[condition_summary_df["ABL"] == abl].sort_values("ILD")
            model = continuous_summary_df[
                (continuous_summary_df["objective"] == objective)
                & (continuous_summary_df["ABL"] == abl)
            ].sort_values("ILD")

            ax.errorbar(
                cond["ILD"].to_numpy(dtype=float),
                cond[f"condition_{param}_mean"].to_numpy(dtype=float),
                yerr=cond[f"condition_{param}_sem"].to_numpy(dtype=float),
                marker="o",
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.2,
                linestyle="none",
                color=color,
                ecolor=color,
                capsize=2.2,
                markersize=4.2,
                zorder=3,
            )

            x = model["ILD"].to_numpy(dtype=float)
            y = model[f"pred_{param}_mean"].to_numpy(dtype=float)
            sem = model[f"pred_{param}_sem"].to_numpy(dtype=float)
            ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.10, linewidth=0)
            ax.plot(x, y, color=color, linewidth=2.0)

        y_min, y_max = display_limits[param]
        ax.set_ylim(y_min, y_max)
        panel_model = continuous_summary_df[
            (continuous_summary_df["objective"] == objective)
        ].copy()
        panel_y = panel_model[f"pred_{param}_mean"].to_numpy(dtype=float)
        panel_sem = panel_model[f"pred_{param}_sem"].to_numpy(dtype=float)
        panel_high = np.nanmax(panel_y + panel_sem)
        panel_low = np.nanmin(panel_y - panel_sem)
        if np.isfinite(panel_high) and np.isfinite(panel_low) and (panel_high > y_max or panel_low < y_min):
            ax.text(
                0.58,
                0.96,
                f"model off scale\nrange {panel_low:.2g} to {panel_high:.2g}",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.85},
            )

        metric = metrics_df[
            (metrics_df["objective"] == objective)
            & (metrics_df["parameter"] == param)
            & (metrics_df["ABL"].astype(str) == "all")
        ].iloc[0]
        status = "fit target" if bool(metric["in_fit_objective"]) else "held out"
        ax.text(
            0.04,
            0.96,
            f"{status}\nRMSE={metric['rmse']:.3g}\nr={metric['pearson_r']:.2f}",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.85},
        )
        if row_idx == 0:
            ax.set_title(param_label)
        if col_idx == 0:
            ax.set_ylabel(f"{objective_label}\n{param_label}")
        else:
            ax.set_ylabel(param_label)
        ax.set_xticks([-16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16])
        ax.grid(True, alpha=0.25)

axes[2, 0].set_xlabel("ILD")
axes[2, 1].set_xlabel("ILD")
axes[0, 0].axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.35)
axes[1, 0].axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.35)
axes[2, 0].axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.35)

abl_handles = [Line2D([0], [0], color=COLORS[abl], linewidth=2.0, label=f"ABL={abl}") for abl in ABLS]
source_handles = [
    Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor="black", linestyle="none", label="condition fit"),
    Line2D([0], [0], color="black", linewidth=2.0, label="alpha-model MSE fit"),
]
fig.legend(
    handles=abl_handles + source_handles,
    loc="upper center",
    ncol=5,
    frameon=False,
    bbox_to_anchor=(0.5, 1.02),
)
fig.suptitle("Gamma/Omega MSE fit objective cross-prediction", y=1.055)
fig.tight_layout(rect=[0, 0, 1, 0.985])
fig.savefig(FIG_PNG, dpi=300, bbox_inches="tight")
fig.savefig(FIG_PDF, bbox_inches="tight")
plt.close(fig)
print(f"Saved figure: {FIG_PNG}")
print(f"Saved figure: {FIG_PDF}")

# %%
