# %%
"""
Fit the Gamma/Omega alpha model to the big SVI condition parameters per animal.

Targets are posterior means from the completed big Gamma/Omega/delay SVI fits.
Each animal gets a least-squares fit for rate_lambda, T_0, theta_E,
rate_norm_l, and alpha.
"""

# %%
# =============================================================================
# Parameters
# =============================================================================
from pathlib import Path
import os
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.optimize import least_squares

from gamma_omega_alpha_utils import gamma_omega_alpha_model


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / "svi_big_gamma_omega_delay_all_animals_outputs"
OUTPUT_DIR = OUTPUT_ROOT / "mse_alpha_model_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ANIMAL_VALUES_CSV = OUTPUT_DIR / "big_svi_gamma_omega_animal_values.csv"
CONDITION_SUMMARY_CSV = OUTPUT_DIR / "big_svi_gamma_omega_summary.csv"
MSE_PARAM_CSV = OUTPUT_DIR / "per_animal_mse_gamma_omega_alpha_params.csv"
MSE_CONDITION_VALUES_CSV = OUTPUT_DIR / "per_animal_mse_gamma_omega_condition_values.csv"
MSE_CONTINUOUS_VALUES_CSV = OUTPUT_DIR / "per_animal_mse_gamma_omega_continuous_values.csv"
MSE_CONTINUOUS_SUMMARY_CSV = OUTPUT_DIR / "per_animal_mse_gamma_omega_continuous_summary.csv"
METRICS_CSV = OUTPUT_DIR / "per_animal_mse_gamma_omega_metrics.csv"
FIG_PNG = OUTPUT_DIR / "big_svi_gamma_omega_with_per_animal_mse_alpha_model.png"

EXPECTED_N_ANIMALS = 30
EXPECTED_N_CONDITION_ROWS = 864
ABLS = [20, 40, 60]
ILDS = np.array([-16, -8, -4, -2, -1, 1, 2, 4, 8, 16], dtype=int)
SMOOTH_ILDS = np.round(np.arange(-16.0, 16.0 + 0.05, 0.1), 10)
COLORS = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}

P_0 = 20e-6
PARAM_NAMES = ["rate_lambda", "rate_norm_l", "alpha", "theta_E", "T_0"]
P0_FIT = np.array([2.0, 0.9, 0.5, 2.4, 100e-3])
LOWER_BOUNDS = np.array([0.1, 0.0, 0.0, 0.5, 1e-3])
UPPER_BOUNDS = np.array([6.0, 2.0, 5.0, 15.0, 2.0])


# %%
# =============================================================================
# Helpers
# =============================================================================
def load_big_svi_condition_values():
    summary_paths = sorted(OUTPUT_ROOT.glob("*/*_big_gamma_omega_delay_condition_summary.csv"))
    if len(summary_paths) != EXPECTED_N_ANIMALS:
        raise RuntimeError(f"Expected {EXPECTED_N_ANIMALS} condition-summary CSVs, found {len(summary_paths)}")

    rows = []
    for summary_path in summary_paths:
        match = re.match(r"^(?P<batch>.+)_(?P<animal>\d+)$", summary_path.parent.name)
        if match is None:
            raise RuntimeError(f"Could not parse animal folder name: {summary_path.parent}")

        df = pd.read_csv(summary_path)
        required_cols = ["batch_name", "animal", "ABL", "ILD", "gamma_mean", "omega_mean"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise KeyError(f"{summary_path} missing columns: {missing_cols}")

        df = df[required_cols].copy()
        df["animal"] = df["animal"].astype(int)
        df["ABL"] = df["ABL"].astype(int)
        df["ILD"] = df["ILD"].astype(int)
        df = df.rename(columns={"gamma_mean": "condition_gamma", "omega_mean": "condition_omega"})
        rows.append(df)

    cond_df = pd.concat(rows, ignore_index=True)
    cond_df = cond_df.sort_values(["batch_name", "animal", "ABL", "ILD"]).reset_index(drop=True)
    if len(cond_df) != EXPECTED_N_CONDITION_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_N_CONDITION_ROWS} condition rows, found {len(cond_df)}")
    n_animals = cond_df[["batch_name", "animal"]].drop_duplicates().shape[0]
    if n_animals != EXPECTED_N_ANIMALS:
        raise RuntimeError(f"Expected {EXPECTED_N_ANIMALS} animals, found {n_animals}")
    duplicate_count = int(cond_df.duplicated(["batch_name", "animal", "ABL", "ILD"]).sum())
    if duplicate_count:
        raise RuntimeError(f"Found {duplicate_count} duplicate animal/condition rows.")
    if not np.all(np.isfinite(cond_df[["condition_gamma", "condition_omega"]].to_numpy(dtype=float))):
        raise RuntimeError("Condition Gamma/Omega values contain NaN/Inf.")
    return cond_df


def summarize_group_values(df, group_cols, value_cols, prefix):
    rows = []
    for group_key, group in df.groupby(group_cols, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = {col: value for col, value in zip(group_cols, group_key)}
        for value_col in value_cols:
            values = group[value_col].to_numpy(dtype=float)
            finite = np.isfinite(values)
            n = int(np.sum(finite))
            short_name = value_col.replace(prefix, "").strip("_")
            row[f"{prefix}_{short_name}_mean"] = float(np.nanmean(values)) if n else np.nan
            row[f"{prefix}_{short_name}_sd"] = float(np.nanstd(values, ddof=1)) if n > 1 else np.nan
            row[f"{prefix}_{short_name}_sem"] = float(np.nanstd(values, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
            row[f"n_{prefix}_{short_name}"] = n
        rows.append(row)
    return pd.DataFrame(rows)


def model_curves_for_params(params, ild_grid):
    rows = []
    for abl in ABLS:
        gamma, omega = gamma_omega_alpha_model(
            abl,
            np.asarray(ild_grid, dtype=float),
            params["rate_lambda"],
            params["rate_norm_l"],
            params["alpha"],
            params["theta_E"],
            params["T_0"],
            P_0,
        )
        for ild, curr_gamma, curr_omega in zip(ild_grid, gamma, omega):
            rows.append(
                {
                    "ABL": int(abl),
                    "ILD": float(ild),
                    "mse_model_gamma": float(curr_gamma),
                    "mse_model_omega": float(curr_omega),
                }
            )
    return pd.DataFrame(rows)


def model_values_for_condition_rows(params, condition_rows):
    gamma, omega = gamma_omega_alpha_model(
        condition_rows["ABL"].to_numpy(dtype=float),
        condition_rows["ILD"].to_numpy(dtype=float),
        params["rate_lambda"],
        params["rate_norm_l"],
        params["alpha"],
        params["theta_E"],
        params["T_0"],
        P_0,
    )
    values = condition_rows[["batch_name", "animal", "ABL", "ILD"]].copy()
    values["mse_model_gamma"] = np.asarray(gamma, dtype=float)
    values["mse_model_omega"] = np.asarray(omega, dtype=float)
    return values


def fit_gamma_omega_alpha(fit_abls, fit_ilds, target_gamma, target_omega):
    gamma_scale = np.nanstd(target_gamma)
    omega_scale = np.nanstd(target_omega)
    if gamma_scale == 0 or not np.isfinite(gamma_scale):
        gamma_scale = 1.0
    if omega_scale == 0 or not np.isfinite(omega_scale):
        omega_scale = 1.0

    def residuals(params):
        rate_lambda, rate_norm_l, alpha, theta_E, T_0 = params
        pred_gamma, pred_omega = gamma_omega_alpha_model(
            fit_abls,
            fit_ilds,
            rate_lambda,
            rate_norm_l,
            alpha,
            theta_E,
            T_0,
            P_0,
        )
        gamma_residuals = (pred_gamma - target_gamma) / gamma_scale
        omega_residuals = (pred_omega - target_omega) / omega_scale
        return np.concatenate([gamma_residuals, omega_residuals])

    return least_squares(residuals, P0_FIT, bounds=(LOWER_BOUNDS, UPPER_BOUNDS))


def fit_per_animal_mse_models(cond_df):
    fit_rows = []
    smooth_rows = []
    condition_rows = []

    animals = cond_df[["batch_name", "animal"]].drop_duplicates().sort_values(["batch_name", "animal"])
    for animal_row in animals.itertuples(index=False):
        batch_name = animal_row.batch_name
        animal_id = int(animal_row.animal)
        animal_cond = cond_df[
            (cond_df["batch_name"] == batch_name)
            & (cond_df["animal"].astype(int) == animal_id)
        ].copy()
        finite = (
            np.isfinite(animal_cond["condition_gamma"].to_numpy(dtype=float))
            & np.isfinite(animal_cond["condition_omega"].to_numpy(dtype=float))
        )
        animal_cond = animal_cond.loc[finite]

        if len(animal_cond) < len(PARAM_NAMES):
            fit_rows.append(
                {
                    "batch_name": batch_name,
                    "animal": animal_id,
                    "success": False,
                    "n_points": len(animal_cond),
                    "message": "too few finite condition points",
                }
            )
            continue

        try:
            fit_result = fit_gamma_omega_alpha(
                animal_cond["ABL"].to_numpy(dtype=float),
                animal_cond["ILD"].to_numpy(dtype=float),
                animal_cond["condition_gamma"].to_numpy(dtype=float),
                animal_cond["condition_omega"].to_numpy(dtype=float),
            )
            params = {name: float(value) for name, value in zip(PARAM_NAMES, fit_result.x)}
            fit_rows.append(
                {
                    "batch_name": batch_name,
                    "animal": animal_id,
                    "success": bool(fit_result.success),
                    "n_points": len(animal_cond),
                    "message": fit_result.message,
                    "cost": float(fit_result.cost),
                    "rate_lambda": params["rate_lambda"],
                    "T_0": params["T_0"],
                    "theta_E": params["theta_E"],
                    "rate_norm_l": params["rate_norm_l"],
                    "alpha": params["alpha"],
                }
            )
            if not fit_result.success:
                continue

            smooth_df = model_curves_for_params(params, SMOOTH_ILDS)
            smooth_df.insert(0, "animal", animal_id)
            smooth_df.insert(0, "batch_name", batch_name)
            smooth_rows.extend(smooth_df.to_dict("records"))

            condition_df = model_values_for_condition_rows(params, animal_cond)
            condition_rows.extend(condition_df.to_dict("records"))
        except Exception as exc:
            fit_rows.append(
                {
                    "batch_name": batch_name,
                    "animal": animal_id,
                    "success": False,
                    "n_points": len(animal_cond),
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )

    fit_df = pd.DataFrame(fit_rows).sort_values(["batch_name", "animal"]).reset_index(drop=True)
    smooth_df = pd.DataFrame(smooth_rows).sort_values(["batch_name", "animal", "ABL", "ILD"]).reset_index(drop=True)
    condition_df = pd.DataFrame(condition_rows).sort_values(["batch_name", "animal", "ABL", "ILD"]).reset_index(drop=True)
    return fit_df, smooth_df, condition_df


def compute_metrics(cond_df, mse_condition_df):
    merged = cond_df.merge(
        mse_condition_df,
        on=["batch_name", "animal", "ABL", "ILD"],
        how="left",
        validate="one_to_one",
    )
    rows = []
    for param in ["gamma", "omega"]:
        for abl in ABLS + ["all"]:
            subset = merged if abl == "all" else merged[merged["ABL"] == abl]
            target = subset[f"condition_{param}"].to_numpy(dtype=float)
            pred = subset[f"mse_model_{param}"].to_numpy(dtype=float)
            finite = np.isfinite(target) & np.isfinite(pred)
            if int(np.sum(finite)) >= 2:
                diff = target[finite] - pred[finite]
                rows.append(
                    {
                        "parameter": param,
                        "ABL": abl,
                        "n_points": int(np.sum(finite)),
                        "rmse": float(np.sqrt(np.mean(diff**2))),
                        "mae": float(np.mean(np.abs(diff))),
                        "pearson_r": float(np.corrcoef(target[finite], pred[finite])[0, 1]),
                    }
                )
            else:
                rows.append(
                    {
                        "parameter": param,
                        "ABL": abl,
                        "n_points": int(np.sum(finite)),
                        "rmse": np.nan,
                        "mae": np.nan,
                        "pearson_r": np.nan,
                    }
                )
    return pd.DataFrame(rows)


def metric_for(metrics_df, param):
    row = metrics_df[
        (metrics_df["parameter"] == param)
        & (metrics_df["ABL"].astype(str) == "all")
    ]
    if len(row) != 1:
        return None
    return row.iloc[0]


def plot_gamma_omega(condition_summary_df, mse_continuous_summary_df, metrics_df):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), sharex=True)
    plot_specs = [
        ("gamma", "Gamma"),
        ("omega", "Omega"),
    ]

    for ax, (param, ylabel) in zip(axes, plot_specs):
        for abl in ABLS:
            color = COLORS[abl]
            cond_subset = condition_summary_df[condition_summary_df["ABL"] == abl]
            ax.errorbar(
                cond_subset["ILD"],
                cond_subset[f"condition_{param}_mean"],
                yerr=cond_subset[f"condition_{param}_sem"],
                fmt="o",
                ms=4.5,
                mfc="white",
                mec=color,
                ecolor=color,
                color=color,
                capsize=2,
                linestyle="none",
                alpha=0.95,
            )

            mse_subset = mse_continuous_summary_df[mse_continuous_summary_df["ABL"] == abl]
            x = mse_subset["ILD"].to_numpy(dtype=float)
            y = mse_subset[f"mse_model_{param}_mean"].to_numpy(dtype=float)
            sem = mse_subset[f"mse_model_{param}_sem"].to_numpy(dtype=float)
            ax.plot(x, y, color=color, linestyle="-", linewidth=2.1)
            ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.16, linewidth=0)

        metric = metric_for(metrics_df, param)
        metric_text = ""
        if metric is not None:
            metric_text = f"\nRMSE={metric['rmse']:.3g}, r={metric['pearson_r']:.2f}"
        ax.set_title(f"{ylabel}: big SVI condition means vs per-animal MSE alpha model{metric_text}", fontsize=11)
        ax.set_xlabel("ILD")
        ax.set_ylabel(ylabel)
        ax.set_xlim(-17, 17)
        if param == "gamma":
            ax.axhline(0.0, color="0.75", linewidth=0.8, zorder=0)
        if param == "omega":
            ax.set_ylim(bottom=2)
        ax.grid(True, alpha=0.22)

    abl_handles = [Line2D([0], [0], color=COLORS[abl], linewidth=2.3, label=f"ABL {abl}") for abl in ABLS]
    source_handles = [
        Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor="black", linestyle="none", label="big SVI condition mean +/- SEM"),
        Line2D([0], [0], color="black", linewidth=2.1, label="per-animal MSE alpha model mean +/- SEM"),
    ]
    fig.legend(
        handles=abl_handles + source_handles,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle("Big Gamma/Omega/delay SVI condition Gamma/Omega with per-animal MSE alpha-model fits", y=1.11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_PNG, dpi=300, bbox_inches="tight")
    return FIG_PNG


# %%
# =============================================================================
# Main analysis
# =============================================================================
print(f"Big SVI output root: {OUTPUT_ROOT}")
print(f"MSE comparison output folder: {OUTPUT_DIR}")

cond_df = load_big_svi_condition_values()
cond_df.to_csv(ANIMAL_VALUES_CSV, index=False)
print(f"Loaded condition rows: {len(cond_df)}")
print(f"Loaded animals: {cond_df[['batch_name', 'animal']].drop_duplicates().shape[0]}")
print(f"Saved animal condition values: {ANIMAL_VALUES_CSV}")

condition_summary_df = summarize_group_values(
    cond_df,
    ["ABL", "ILD"],
    ["condition_gamma", "condition_omega"],
    "condition",
)
condition_summary_df.to_csv(CONDITION_SUMMARY_CSV, index=False)
print(f"Saved condition summary: {CONDITION_SUMMARY_CSV}")

edge_counts = condition_summary_df[condition_summary_df["ILD"].abs() == 16][
    ["ABL", "ILD", "n_condition_gamma", "n_condition_omega"]
]
print("Condition counts at |ILD|=16:")
print(edge_counts.to_string(index=False))

mse_param_df, mse_continuous_df, mse_condition_df = fit_per_animal_mse_models(cond_df)
mse_param_df.to_csv(MSE_PARAM_CSV, index=False)
mse_condition_df.to_csv(MSE_CONDITION_VALUES_CSV, index=False)
mse_continuous_df.to_csv(MSE_CONTINUOUS_VALUES_CSV, index=False)
print(f"Saved per-animal MSE params: {MSE_PARAM_CSV}")
print(f"Saved MSE condition values: {MSE_CONDITION_VALUES_CSV}")
print(f"Saved MSE continuous values: {MSE_CONTINUOUS_VALUES_CSV}")

successful_mse = mse_param_df[mse_param_df["success"] == True].copy()
print(f"Successful per-animal MSE fits: {len(successful_mse)} / {len(mse_param_df)}")
if len(successful_mse) != EXPECTED_N_ANIMALS:
    raise RuntimeError("Not all per-animal MSE fits succeeded.")

for param in ["rate_lambda", "T_0", "theta_E", "rate_norm_l", "alpha"]:
    print(f"  {param}: {successful_mse[param].mean():.6g} +/- {successful_mse[param].sem():.6g}")

mse_continuous_summary_df = summarize_group_values(
    mse_continuous_df,
    ["ABL", "ILD"],
    ["mse_model_gamma", "mse_model_omega"],
    "mse_model",
)
mse_continuous_summary_df.to_csv(MSE_CONTINUOUS_SUMMARY_CSV, index=False)
print(f"Saved MSE continuous summary: {MSE_CONTINUOUS_SUMMARY_CSV}")

metrics_df = compute_metrics(cond_df, mse_condition_df)
metrics_df.to_csv(METRICS_CSV, index=False)
print(f"Saved metrics: {METRICS_CSV}")
print(metrics_df.to_string(index=False))

fig_path = plot_gamma_omega(condition_summary_df, mse_continuous_summary_df, metrics_df)
print(f"Saved figure: {fig_path}")

plt.show()

# %%
