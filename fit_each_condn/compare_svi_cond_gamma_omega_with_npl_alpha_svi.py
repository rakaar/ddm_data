# %%
"""
Compare condition-by-condition SVI Gamma/Omega fits with animal-wise NPL+alpha SVI.

The condition fits are the consolidated Gamma/Omega-only NumPyro SVI fits, where
all non-Gamma/Omega decision-side parameters were fixed from the animal-wise
NPL+alpha condition-delay SVI fits. This script compares those condition
Gamma/Omega values against:

1. Gamma/Omega implied by the same animal-wise NPL+alpha SVI posterior means;
2. per-animal least-squares Gamma/Omega alpha-model fits to the condition means.
"""

# %%
# =============================================================================
# Imports
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


# %%
# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

ANIMAL_SVI_ROOT = (
    REPO_DIR
    / "fit_animal_by_animal"
    / "numpyro_svi_npl_alpha_condition_delay_single_animal_outputs"
)
COND_SVI_DIR = (
    SCRIPT_DIR
    / "svi_gamma_omega_fixed_from_animal_svi_condition_delay_results"
    / "all_observed_with_30k_reruns"
)
COND_CACHE_CSV = COND_SVI_DIR / "condition_gamma_omega_extraction_cache.csv"

OUTPUT_DIR = SCRIPT_DIR / "svi_condition_gamma_omega_vs_npl_alpha_svi_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COND_ANIMAL_VALUES_CSV = OUTPUT_DIR / "svi_condition_gamma_omega_animal_values.csv"
COND_SUMMARY_CSV = OUTPUT_DIR / "svi_condition_gamma_omega_summary.csv"
MODEL_PARAM_CSV = OUTPUT_DIR / "npl_alpha_svi_params_by_animal.csv"
MODEL_CONTINUOUS_SUMMARY_CSV = OUTPUT_DIR / "npl_alpha_svi_gamma_omega_summary.csv"
MODEL_CONDITION_VALUES_CSV = OUTPUT_DIR / "npl_alpha_svi_gamma_omega_condition_values.csv"
MSE_PARAM_CSV = OUTPUT_DIR / "per_animal_mse_gamma_omega_alpha_params.csv"
MSE_CONTINUOUS_SUMMARY_CSV = OUTPUT_DIR / "per_animal_mse_gamma_omega_summary.csv"
MSE_CONDITION_VALUES_CSV = OUTPUT_DIR / "per_animal_mse_gamma_omega_condition_values.csv"
AGREEMENT_POINTS_CSV = OUTPUT_DIR / "gamma_omega_agreement_points.csv"
AGREEMENT_SUMMARY_CSV = OUTPUT_DIR / "gamma_omega_agreement_summary.csv"
METRICS_CSV = OUTPUT_DIR / "gamma_omega_agreement_metrics.csv"

FIG_PNG = OUTPUT_DIR / "svi_cond_gamma_omega_vs_npl_alpha_svi.png"
FIG_PDF = OUTPUT_DIR / "svi_cond_gamma_omega_vs_npl_alpha_svi.pdf"

MODEL_LABEL = "NPL+alpha SVI"
MSE_LABEL = "animal-wise MSE alpha model"

PARAM_NAMES = ["rate_lambda", "ell", "alpha", "theta", "T_0"]
SVI_SAMPLE_KEYS = {
    "rate_lambda": "rate_lambda",
    "ell": "rate_norm_l",
    "alpha": "alpha",
    "theta": "theta_E",
    "T_0": "T_0",
}

P0_FIT = np.array([2.0, 0.9, 0.5, 2.4, 100e-3])
LOWER_BOUNDS = np.array([0.1, 0.0, 0.0, 0.5, 1e-3])
UPPER_BOUNDS = np.array([6.0, 2.0, 5.0, 15.0, 2.0])

ABLS = [20, 40, 60]
SMOOTH_ILDS = np.round(np.arange(-16.0, 16.0 + 0.05, 0.1), 10)
P_0 = 20e-6
COLORS = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}

EXPECTED_N_ANIMALS = 30
EXPECTED_N_COND_FITS = 864


# %%
# =============================================================================
# Helpers
# =============================================================================
def finite_mean(values):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.nan
    return float(np.mean(values[finite]))


def parse_animal_folder(path):
    match = re.match(r"^(?P<batch>.+)_(?P<animal>\d+)$", path.name)
    if match is None:
        return None
    return match.group("batch"), int(match.group("animal"))


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
            row[f"{prefix}_{short_name}_mean"] = float(np.nanmean(values)) if n > 0 else np.nan
            row[f"{prefix}_{short_name}_sem"] = float(np.nanstd(values) / np.sqrt(n)) if n > 0 else np.nan
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
                    "gamma": float(curr_gamma),
                    "omega": float(curr_omega),
                }
            )
    return pd.DataFrame(rows)


def model_values_for_condition_rows(params, condition_rows):
    gamma, omega = gamma_omega_alpha_model(
        condition_rows["ABL"].to_numpy(dtype=float),
        condition_rows["ILD"].to_numpy(dtype=float),
        params["rate_lambda"],
        params["ell"],
        params["alpha"],
        params["theta"],
        params["T_0"],
        P_0,
    )
    values = condition_rows[["batch_name", "animal", "ABL", "ILD"]].copy()
    values["gamma"] = np.asarray(gamma, dtype=float)
    values["omega"] = np.asarray(omega, dtype=float)
    return values


def load_condition_svi_cache():
    if not COND_CACHE_CSV.exists():
        raise FileNotFoundError(COND_CACHE_CSV)

    cond_df = pd.read_csv(COND_CACHE_CSV)
    required_cols = ["batch_name", "animal", "ABL", "ILD", "condition_gamma", "condition_omega"]
    missing_cols = [col for col in required_cols if col not in cond_df.columns]
    if missing_cols:
        raise KeyError(f"{COND_CACHE_CSV} missing columns: {missing_cols}")

    cond_df["animal"] = cond_df["animal"].astype(int)
    cond_df["ABL"] = cond_df["ABL"].astype(int)
    cond_df["ILD"] = cond_df["ILD"].astype(int)
    cond_df = cond_df.sort_values(["batch_name", "animal", "ABL", "ILD"]).reset_index(drop=True)

    if len(cond_df) != EXPECTED_N_COND_FITS:
        raise RuntimeError(f"Expected {EXPECTED_N_COND_FITS} condition rows, found {len(cond_df)}")
    duplicate_count = int(cond_df.duplicated(["batch_name", "animal", "ABL", "ILD"]).sum())
    if duplicate_count:
        raise RuntimeError(f"Found {duplicate_count} duplicate condition rows in {COND_CACHE_CSV}")
    n_animals = cond_df[["batch_name", "animal"]].drop_duplicates().shape[0]
    if n_animals != EXPECTED_N_ANIMALS:
        raise RuntimeError(f"Expected {EXPECTED_N_ANIMALS} animals, found {n_animals}")
    if not np.all(np.isfinite(cond_df[["condition_gamma", "condition_omega"]].to_numpy(dtype=float))):
        raise RuntimeError("Condition Gamma/Omega cache contains non-finite values.")
    return cond_df


def load_svi_model_gamma_omega(cond_df):
    param_rows = []
    smooth_rows = []
    condition_rows = []
    expected_animals = {
        (row.batch_name, int(row.animal))
        for row in cond_df[["batch_name", "animal"]].drop_duplicates().itertuples(index=False)
    }

    found_animals = {}
    for animal_dir in sorted(ANIMAL_SVI_ROOT.iterdir()):
        if not animal_dir.is_dir():
            continue
        parsed = parse_animal_folder(animal_dir)
        if parsed is None:
            continue
        if parsed in expected_animals:
            found_animals[parsed] = animal_dir

    if set(found_animals) != expected_animals:
        missing = sorted(expected_animals - set(found_animals))
        extra = sorted(set(found_animals) - expected_animals)
        raise RuntimeError(f"Animal SVI folder mismatch. missing={missing}; extra={extra}")

    for batch_name, animal_id in sorted(expected_animals):
        animal_dir = found_animals[(batch_name, animal_id)]
        posterior_npz = animal_dir / "main_fullrank_posterior_samples.npz"
        condition_table_csv = animal_dir / "condition_table.csv"
        if not posterior_npz.exists():
            raise FileNotFoundError(posterior_npz)
        if not condition_table_csv.exists():
            raise FileNotFoundError(condition_table_csv)

        posterior = np.load(posterior_npz)
        missing_keys = [sample_key for sample_key in SVI_SAMPLE_KEYS.values() if sample_key not in posterior.files]
        if missing_keys:
            raise KeyError(f"{posterior_npz} missing keys: {missing_keys}")

        params = {
            param_name: finite_mean(posterior[sample_key])
            for param_name, sample_key in SVI_SAMPLE_KEYS.items()
        }
        if not np.all(np.isfinite(list(params.values()))):
            raise RuntimeError(f"Non-finite SVI parameter mean for {batch_name}/{animal_id}: {params}")
        param_rows.append(
            {
                "batch_name": batch_name,
                "animal": animal_id,
                **params,
                "source_npz": str(posterior_npz),
                "source_condition_table": str(condition_table_csv),
            }
        )

        smooth_df = model_curves_for_params(params, SMOOTH_ILDS)
        smooth_df.insert(0, "animal", animal_id)
        smooth_df.insert(0, "batch_name", batch_name)
        smooth_df = smooth_df.rename(columns={"gamma": "svi_model_gamma", "omega": "svi_model_omega"})
        smooth_rows.extend(smooth_df.to_dict("records"))

        animal_cond = cond_df[
            (cond_df["batch_name"] == batch_name)
            & (cond_df["animal"].astype(int) == animal_id)
        ].copy()
        condition_df = model_values_for_condition_rows(params, animal_cond)
        condition_df = condition_df.rename(columns={"gamma": "svi_model_gamma", "omega": "svi_model_omega"})
        condition_rows.extend(condition_df.to_dict("records"))

    param_df = pd.DataFrame(param_rows).sort_values(["batch_name", "animal"]).reset_index(drop=True)
    smooth_df = pd.DataFrame(smooth_rows).sort_values(["batch_name", "animal", "ABL", "ILD"]).reset_index(drop=True)
    condition_df = pd.DataFrame(condition_rows).sort_values(["batch_name", "animal", "ABL", "ILD"]).reset_index(drop=True)
    return param_df, smooth_df, condition_df


def fit_gamma_omega_alpha(fit_abls, fit_ilds, target_gamma, target_omega):
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
        gamma_residuals = (pred_gamma - target_gamma) / gamma_scale
        omega_residuals = (pred_omega - target_omega) / omega_scale
        return np.concatenate([gamma_residuals, omega_residuals])

    return least_squares(residuals, P0_FIT, bounds=(LOWER_BOUNDS, UPPER_BOUNDS))


def fit_per_animal_mse_models(cond_df, animal_df):
    fit_rows = []
    smooth_rows = []
    condition_rows = []

    for _, animal_row in animal_df[["batch_name", "animal"]].drop_duplicates().iterrows():
        batch_name = animal_row["batch_name"]
        animal_id = int(animal_row["animal"])
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
                    **params,
                }
            )
            if not fit_result.success:
                continue

            smooth_df = model_curves_for_params(params, SMOOTH_ILDS)
            smooth_df.insert(0, "animal", animal_id)
            smooth_df.insert(0, "batch_name", batch_name)
            smooth_df = smooth_df.rename(columns={"gamma": "mse_model_gamma", "omega": "mse_model_omega"})
            smooth_rows.extend(smooth_df.to_dict("records"))

            condition_df = model_values_for_condition_rows(params, animal_cond)
            condition_df = condition_df.rename(columns={"gamma": "mse_model_gamma", "omega": "mse_model_omega"})
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


def build_agreement_points(cond_df, svi_condition_df, mse_condition_df):
    agreement = cond_df.merge(
        svi_condition_df,
        on=["batch_name", "animal", "ABL", "ILD"],
        how="left",
        validate="one_to_one",
    )
    agreement = agreement.merge(
        mse_condition_df,
        on=["batch_name", "animal", "ABL", "ILD"],
        how="left",
        validate="one_to_one",
    )
    return agreement


def build_agreement_summary(agreement_df):
    value_cols = [
        "condition_gamma",
        "condition_omega",
        "svi_model_gamma",
        "svi_model_omega",
        "mse_model_gamma",
        "mse_model_omega",
    ]
    rows = []
    for (abl, ild), group in agreement_df.groupby(["ABL", "ILD"], sort=True):
        row = {"ABL": int(abl), "ILD": int(ild)}
        for col in value_cols:
            values = group[col].to_numpy(dtype=float)
            finite = np.isfinite(values)
            n = int(np.sum(finite))
            row[f"{col}_mean"] = float(np.nanmean(values)) if n > 0 else np.nan
            row[f"{col}_sem"] = float(np.nanstd(values) / np.sqrt(n)) if n > 0 else np.nan
            row[f"n_{col}"] = n
        rows.append(row)
    return pd.DataFrame(rows)


def compute_agreement_metrics(agreement_summary_df):
    rows = []
    sources = [("svi_model", MODEL_LABEL), ("mse_model", MSE_LABEL)]
    for source, source_label in sources:
        for param in ["gamma", "omega"]:
            for abl in ABLS + ["all"]:
                subset = agreement_summary_df if abl == "all" else agreement_summary_df[agreement_summary_df["ABL"] == abl]
                x = subset[f"{source}_{param}_mean"].to_numpy(dtype=float)
                y = subset[f"condition_{param}_mean"].to_numpy(dtype=float)
                finite = np.isfinite(x) & np.isfinite(y)
                if np.sum(finite) >= 2:
                    diff = y[finite] - x[finite]
                    r = float(np.corrcoef(x[finite], y[finite])[0, 1])
                    rmse = float(np.sqrt(np.mean(diff**2)))
                    mae = float(np.mean(np.abs(diff)))
                else:
                    r = np.nan
                    rmse = np.nan
                    mae = np.nan
                rows.append(
                    {
                        "source": source,
                        "source_label": source_label,
                        "parameter": param,
                        "ABL": abl,
                        "n_points": int(np.sum(finite)),
                        "pearson_r": r,
                        "rmse": rmse,
                        "mae": mae,
                    }
                )
    return pd.DataFrame(rows)


def print_param_means(label, param_df):
    print(f"{label} parameter means across animals:")
    for param in PARAM_NAMES:
        if param in param_df:
            print(f"  {param}: {param_df[param].mean():.6g} +/- {param_df[param].sem():.6g}")


def metric_for(metrics_df, source, param):
    row = metrics_df[
        (metrics_df["source"] == source)
        & (metrics_df["parameter"] == param)
        & (metrics_df["ABL"].astype(str) == "all")
    ]
    if len(row) != 1:
        return None
    return row.iloc[0]


def plot_gamma_omega(cond_summary_df, svi_summary_df, mse_summary_df, metrics_df):
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.0))
    plot_specs = [
        ("gamma", "Gamma: SVI condition fits vs NPL+alpha SVI"),
        ("omega", "Omega: SVI condition fits vs NPL+alpha SVI"),
    ]

    for ax, (param, title) in zip(axes, plot_specs):
        for abl in ABLS:
            color = COLORS[abl]
            cond_subset = cond_summary_df[cond_summary_df["ABL"] == abl]
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

            svi_subset = svi_summary_df[svi_summary_df["ABL"] == abl]
            x = svi_subset["ILD"].to_numpy(dtype=float)
            y = svi_subset[f"svi_model_{param}_mean"].to_numpy(dtype=float)
            sem = svi_subset[f"svi_model_{param}_sem"].to_numpy(dtype=float)
            ax.plot(x, y, color=color, linestyle="-", linewidth=2.3)
            ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.14, linewidth=0)

            mse_subset = mse_summary_df[mse_summary_df["ABL"] == abl]
            x = mse_subset["ILD"].to_numpy(dtype=float)
            y = mse_subset[f"mse_model_{param}_mean"].to_numpy(dtype=float)
            sem = mse_subset[f"mse_model_{param}_sem"].to_numpy(dtype=float)
            ax.plot(x, y, color=color, linestyle="--", linewidth=2.0)
            ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.10, linewidth=0)

        svi_metric = metric_for(metrics_df, "svi_model", param)
        mse_metric = metric_for(metrics_df, "mse_model", param)
        metric_text = ""
        if svi_metric is not None and mse_metric is not None:
            metric_text = (
                "\nall ABLs RMSE: "
                f"SVI={svi_metric['rmse']:.3g}, MSE={mse_metric['rmse']:.3g}; "
                f"r: SVI={svi_metric['pearson_r']:.2f}, MSE={mse_metric['pearson_r']:.2f}"
            )
        ax.set_title(title + metric_text, fontsize=11)
        ax.set_xlabel("ILD")
        ax.set_ylabel(param)
        ax.axhline(0, color="0.75", linewidth=0.8, zorder=0)
        ax.set_xlim(-17, 17)
        if param == "omega":
            ax.set_ylim(bottom=2)
        ax.grid(True, alpha=0.22)

    abl_handles = [
        Line2D([0], [0], color=COLORS[abl], linewidth=2.5, label=f"ABL {abl}")
        for abl in ABLS
    ]
    source_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            markerfacecolor="white",
            markeredgecolor="black",
            linestyle="none",
            label="condition SVI fit",
        ),
        Line2D([0], [0], color="black", linestyle="-", linewidth=2.3, label=MODEL_LABEL),
        Line2D([0], [0], color="black", linestyle="--", linewidth=2.0, label=MSE_LABEL),
    ]
    fig.legend(
        handles=abl_handles + source_handles,
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(
        "Condition-by-condition SVI Gamma/Omega vs animal-wise NPL+alpha SVI",
        y=1.12,
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(FIG_PDF, bbox_inches="tight")
    return FIG_PNG, FIG_PDF


# %%
# =============================================================================
# Main analysis
# =============================================================================
print(f"Animal-wise SVI folder: {ANIMAL_SVI_ROOT}")
print(f"Condition SVI cache: {COND_CACHE_CSV}")
print(f"Output folder: {OUTPUT_DIR}")

cond_df = load_condition_svi_cache()
cond_df.to_csv(COND_ANIMAL_VALUES_CSV, index=False)
print(f"Loaded condition SVI rows: {len(cond_df)}")
print(f"Condition animals: {cond_df[['batch_name', 'animal']].drop_duplicates().shape[0]}")
print(f"Saved condition animal values: {COND_ANIMAL_VALUES_CSV}")

param_df, svi_continuous_df, svi_condition_df = load_svi_model_gamma_omega(cond_df)
param_df.to_csv(MODEL_PARAM_CSV, index=False)
svi_condition_df.to_csv(MODEL_CONDITION_VALUES_CSV, index=False)
print(f"Loaded animal-wise {MODEL_LABEL} fits for {len(param_df)} animals")
print(f"Saved NPL+alpha SVI parameter summary: {MODEL_PARAM_CSV}")
print(f"Saved NPL+alpha SVI condition values: {MODEL_CONDITION_VALUES_CSV}")
print_param_means(MODEL_LABEL, param_df)

cond_summary_df = summarize_group_values(
    cond_df,
    ["ABL", "ILD"],
    ["condition_gamma", "condition_omega"],
    "condition",
)
cond_summary_df.to_csv(COND_SUMMARY_CSV, index=False)
edge_ild_counts = cond_summary_df[cond_summary_df["ILD"].abs() == 16][
    ["ABL", "ILD", "n_condition_gamma", "n_condition_omega"]
]
print("Condition counts at |ILD|=16:")
print(edge_ild_counts.to_string(index=False))
print(f"Saved condition summary: {COND_SUMMARY_CSV}")

svi_continuous_summary_df = summarize_group_values(
    svi_continuous_df,
    ["ABL", "ILD"],
    ["svi_model_gamma", "svi_model_omega"],
    "svi_model",
)
svi_continuous_summary_df.to_csv(MODEL_CONTINUOUS_SUMMARY_CSV, index=False)
print(f"Saved NPL+alpha SVI continuous summary: {MODEL_CONTINUOUS_SUMMARY_CSV}")

mse_param_df, mse_continuous_df, mse_condition_df = fit_per_animal_mse_models(cond_df, param_df)
mse_param_df.to_csv(MSE_PARAM_CSV, index=False)
mse_condition_df.to_csv(MSE_CONDITION_VALUES_CSV, index=False)
print(f"Saved per-animal MSE fit summary: {MSE_PARAM_CSV}")
print(f"Saved MSE condition values: {MSE_CONDITION_VALUES_CSV}")

successful_mse = mse_param_df[mse_param_df["success"] == True].copy()
print(f"Successful per-animal MSE fits: {len(successful_mse)} / {len(mse_param_df)}")
if len(successful_mse) == 0:
    raise RuntimeError("No per-animal MSE fits succeeded.")
print_param_means(MSE_LABEL, successful_mse)

mse_continuous_summary_df = summarize_group_values(
    mse_continuous_df,
    ["ABL", "ILD"],
    ["mse_model_gamma", "mse_model_omega"],
    "mse_model",
)
mse_continuous_summary_df.to_csv(MSE_CONTINUOUS_SUMMARY_CSV, index=False)
print(f"Saved MSE continuous summary: {MSE_CONTINUOUS_SUMMARY_CSV}")

agreement_df = build_agreement_points(cond_df, svi_condition_df, mse_condition_df)
if agreement_df[["svi_model_gamma", "svi_model_omega", "mse_model_gamma", "mse_model_omega"]].isna().any().any():
    missing_counts = agreement_df[
        ["svi_model_gamma", "svi_model_omega", "mse_model_gamma", "mse_model_omega"]
    ].isna().sum()
    raise RuntimeError(f"Missing model values after merge:\n{missing_counts}")

agreement_df.to_csv(AGREEMENT_POINTS_CSV, index=False)
agreement_summary_df = build_agreement_summary(agreement_df)
agreement_summary_df.to_csv(AGREEMENT_SUMMARY_CSV, index=False)
metrics_df = compute_agreement_metrics(agreement_summary_df)
metrics_df.to_csv(METRICS_CSV, index=False)
print(f"Saved agreement points: {AGREEMENT_POINTS_CSV}")
print(f"Saved agreement summary: {AGREEMENT_SUMMARY_CSV}")
print(f"Saved agreement metrics: {METRICS_CSV}")
print(metrics_df.to_string(index=False))

png_path, pdf_path = plot_gamma_omega(
    cond_summary_df,
    svi_continuous_summary_df,
    mse_continuous_summary_df,
    metrics_df,
)
print(f"Saved figure: {png_path}")
print(f"Saved figure: {pdf_path}")

# %%
