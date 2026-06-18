# %%
"""
Compare condition-by-condition Gamma/Omega fits with NPL+alpha fits that fixed
condition-specific t_E_aff.

The animal-wise fits in this comparison refit NPL+alpha parameters while holding
t_E_aff fixed to the condition-by-condition delay estimates. Gamma and Omega are
still implied by the NPL+alpha firing-rate expression:

    gamma, omega = gamma_omega_alpha_model(ABL, ILD, rate_lambda, ell, alpha, theta, T_0, P_0)

This script compares:
1. condition-by-condition Gamma/Omega posterior means;
2. Gamma/Omega curves implied by the fixed-condition-t_E_aff animal-wise fit;
3. Gamma/Omega curves implied by the previous ABL-specific ILD2-delay animal-wise fit;
4. per-animal least-squares Gamma/Omega alpha-model fits to the condition means.
"""

# %%
# =============================================================================
# Imports
# =============================================================================
import os
import pickle
import re
import subprocess
import sys
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
# Configuration
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

UPSTREAM_DIR = (
    REPO_DIR
    / "fit_animal_by_animal"
    / "NPL_alpha_condition_t_E_aff_fixed_delay_fit_results_all_30"
)
PREVIOUS_UPSTREAM_DIR = REPO_DIR / "all_30_NPL_alpha_ABL_specific_ILD2_delay_fit_results"
COND_PKL_DIR = (
    REPO_DIR
    / "all_30_cond_by_cond_gamma_omega_t_E_aff_fix_w_del_go_from_NPL_alpha_ABL_specific_ILD2_fit_results"
    / "pkl_files"
)

OUTPUT_DIR = SCRIPT_DIR / "condition_t_E_aff_fixed_gamma_omega_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COND_CACHE_CSV = OUTPUT_DIR / "condition_gamma_omega_extraction_cache.csv"
SHARED_COND_CACHE_CSV = (
    SCRIPT_DIR
    / "abl_specific_ild2_gamma_omega_comparison"
    / "condition_gamma_omega_extraction_cache.csv"
)
COND_ANIMAL_VALUES_CSV = OUTPUT_DIR / "condition_gamma_omega_animal_values.csv"
COND_SUMMARY_CSV = OUTPUT_DIR / "condition_gamma_omega_summary.csv"
MODEL_PARAM_CSV = OUTPUT_DIR / "npl_alpha_condition_t_E_aff_fixed_params_by_animal.csv"
MODEL_CONTINUOUS_SUMMARY_CSV = OUTPUT_DIR / "npl_alpha_condition_t_E_aff_fixed_gamma_omega_summary.csv"
MODEL_CONDITION_VALUES_CSV = OUTPUT_DIR / "npl_alpha_condition_t_E_aff_fixed_gamma_omega_condition_values.csv"
PREVIOUS_MODEL_PARAM_CSV = OUTPUT_DIR / "npl_alpha_abl_specific_ild2_params_by_animal.csv"
PREVIOUS_MODEL_CONTINUOUS_SUMMARY_CSV = OUTPUT_DIR / "npl_alpha_abl_specific_ild2_gamma_omega_summary.csv"
PREVIOUS_MODEL_CONDITION_VALUES_CSV = OUTPUT_DIR / "npl_alpha_abl_specific_ild2_gamma_omega_condition_values.csv"
MSE_PARAM_CSV = OUTPUT_DIR / "per_animal_mse_gamma_omega_alpha_params.csv"
MSE_CONTINUOUS_SUMMARY_CSV = OUTPUT_DIR / "per_animal_mse_gamma_omega_summary.csv"
MSE_CONDITION_VALUES_CSV = OUTPUT_DIR / "per_animal_mse_gamma_omega_condition_values.csv"
AGREEMENT_POINTS_CSV = OUTPUT_DIR / "gamma_omega_agreement_points.csv"
AGREEMENT_SUMMARY_CSV = OUTPUT_DIR / "gamma_omega_agreement_summary.csv"
METRICS_CSV = OUTPUT_DIR / "gamma_omega_agreement_metrics.csv"

FIG_PNG = OUTPUT_DIR / "cond_gamma_omega_vs_npl_alpha_condition_t_E_aff_fixed_delay.png"
FIG_PDF = OUTPUT_DIR / "cond_gamma_omega_vs_npl_alpha_condition_t_E_aff_fixed_delay.pdf"

MODEL_KEY = "vbmc_norm_alpha_condition_t_E_aff_fixed_delay_tied_results"
MODEL_LABEL = "NPL+alpha fixed condition t_E_aff"
PREVIOUS_MODEL_KEY = "vbmc_norm_alpha_abl_specific_ild2_delay_tied_results"
PREVIOUS_MODEL_LABEL = "NPL+alpha ABL-specific ILD2 delay"
MSE_LABEL = "animal-wise MSE alpha model"
PARAM_SAMPLES_TO_MEAN = {
    "rate_lambda": "rate_lambda_samples",
    "ell": "rate_norm_l_samples",
    "alpha": "alpha_samples",
    "theta": "theta_E_samples",
    "T_0": "T_0_samples",
}
PARAM_NAMES = ["rate_lambda", "ell", "alpha", "theta", "T_0"]
P0_FIT = np.array([2.0, 0.9, 0.5, 2.4, 100e-3])
LOWER_BOUNDS = np.array([0.1, 0.0, 0.0, 0.5, 1e-3])
UPPER_BOUNDS = np.array([6.0, 2.0, 5.0, 15.0, 2.0])

ABLS = [20, 40, 60]
FITTED_ILDS = [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]
SMOOTH_ILDS = np.round(np.arange(-16.0, 16.0 + 0.05, 0.1), 10)
P_0 = 20e-6
COLORS = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}

N_CONDITION_POSTERIOR_SAMPLES = int(float(os.environ.get("N_CONDITION_POSTERIOR_SAMPLES", "50000")))
REFRESH_COND_CACHE = os.environ.get("REFRESH_COND_CACHE", "0").lower() in {"1", "true", "yes", "y"}
AUTO_CREATE_COMPAT_ENV = os.environ.get("AUTO_CREATE_COMPAT_ENV", "1").lower() in {"1", "true", "yes", "y"}
COMPAT_ENV_DIR = Path(os.environ.get("COND_PICKLE_COMPAT_ENV", REPO_DIR / ".venv_cond_pickle_read"))
EXPECTED_N_UPSTREAM_ANIMALS = 30
EXPECTED_N_COND_FITS = 864


# %%
# =============================================================================
# Helpers
# =============================================================================
def parse_upstream_result_name(path):
    match = re.match(
        (
            r"^results_(?P<batch>.+)_animal_(?P<animal>\d+)_"
            r"NORM_ALPHA_CONDITION_T_E_AFF_FIXED_DELAY_FROM_ABORTS\.pkl$"
        ),
        path.name,
    )
    if match is None:
        return None
    return match.group("batch"), int(match.group("animal"))


def parse_previous_upstream_result_name(path):
    match = re.match(
        (
            r"^results_(?P<batch>.+)_animal_(?P<animal>\d+)_"
            r"NORM_ALPHA_ABL_SPECIFIC_ILD2_DELAY_FROM_ABORTS\.pkl$"
        ),
        path.name,
    )
    if match is None:
        return None
    return match.group("batch"), int(match.group("animal"))


def parse_condition_fit_name(path):
    match = re.match(
        (
            r"^vbmc_cond_by_cond_(?P<batch>.+)_(?P<animal>\d+)_(?P<abl>20|40|60)_"
            r"ILD_(?P<ild>-?\d+)_FIX_w_del_go_FROM_ABL_SPECIFIC_ILD2_3_params\.pkl$"
        ),
        path.name,
    )
    if match is None:
        return None
    return (
        match.group("batch"),
        int(match.group("animal")),
        int(match.group("abl")),
        int(match.group("ild")),
    )


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


def load_model_gamma_omega(model_dir, result_glob, parse_result_name, model_key, value_prefix, expected_n, label):
    param_rows = []
    smooth_rows = []
    condition_rows = []

    result_paths = sorted(model_dir.glob(result_glob))
    if len(result_paths) != expected_n:
        raise RuntimeError(f"Expected {expected_n} {label} result pkls, found {len(result_paths)}")

    for result_path in result_paths:
        parsed = parse_result_name(result_path)
        if parsed is None:
            raise RuntimeError(f"Could not parse {label} result filename: {result_path.name}")
        batch_name, animal_id = parsed

        with result_path.open("rb") as f:
            saved = pickle.load(f)
        if model_key not in saved:
            raise KeyError(f"{result_path} is missing `{model_key}`")

        result = saved[model_key]
        message = str(result.get("message", ""))
        if "stable" not in message.lower():
            raise RuntimeError(f"{label} fit is not stable for {batch_name}/{animal_id}: {message}")

        missing_keys = [
            sample_key
            for sample_key in PARAM_SAMPLES_TO_MEAN.values()
            if sample_key not in result
        ]
        if missing_keys:
            raise KeyError(f"{result_path} is missing keys: {missing_keys}")

        params = {
            param_name: float(np.mean(np.asarray(result[sample_key], dtype=float)))
            for param_name, sample_key in PARAM_SAMPLES_TO_MEAN.items()
        }
        param_rows.append(
            {
                "batch_name": batch_name,
                "animal": animal_id,
                **params,
                "message": message,
                "source_pkl": str(result_path),
            }
        )

        smooth_df = model_curves_for_params(params, SMOOTH_ILDS)
        smooth_df.insert(0, "animal", animal_id)
        smooth_df.insert(0, "batch_name", batch_name)
        smooth_df = smooth_df.rename(columns={"gamma": f"{value_prefix}_gamma", "omega": f"{value_prefix}_omega"})
        smooth_rows.extend(smooth_df.to_dict("records"))

        condition_df = model_curves_for_params(params, FITTED_ILDS)
        condition_df.insert(0, "animal", animal_id)
        condition_df.insert(0, "batch_name", batch_name)
        condition_df["ILD"] = condition_df["ILD"].astype(int)
        condition_df = condition_df.rename(columns={"gamma": f"{value_prefix}_gamma", "omega": f"{value_prefix}_omega"})
        condition_rows.extend(condition_df.to_dict("records"))

    param_df = pd.DataFrame(param_rows).sort_values(["batch_name", "animal"]).reset_index(drop=True)
    smooth_df = pd.DataFrame(smooth_rows).sort_values(["batch_name", "animal", "ABL", "ILD"]).reset_index(drop=True)
    condition_df = pd.DataFrame(condition_rows).sort_values(["batch_name", "animal", "ABL", "ILD"]).reset_index(drop=True)
    return param_df, smooth_df, condition_df


def try_extract_condition_means_with_current_python():
    rows = []
    failures = []
    sys.path.insert(0, str(SCRIPT_DIR))
    sys.path.insert(0, str(REPO_DIR))

    try:
        import scipy.special._ufuncs as scipy_ufuncs

        sys.modules.setdefault("scipy.special._special_ufuncs", scipy_ufuncs)
    except Exception:
        pass

    condition_paths = sorted(COND_PKL_DIR.glob("vbmc_cond_by_cond_*_FIX_w_del_go_FROM_ABL_SPECIFIC_ILD2_3_params.pkl"))
    for pkl_path in condition_paths:
        parsed = parse_condition_fit_name(pkl_path)
        if parsed is None:
            failures.append((pkl_path, "filename_parse_failed"))
            continue
        batch_name, animal_id, abl, ild = parsed
        try:
            with pkl_path.open("rb") as f:
                vbmc = pickle.load(f)
            samples = vbmc.vp.sample(N_CONDITION_POSTERIOR_SAMPLES)[0]
            if samples.shape[1] < 3:
                raise ValueError(f"expected at least 3 columns, got {samples.shape[1]}")
            rows.append(
                {
                    "batch_name": batch_name,
                    "animal": animal_id,
                    "ABL": abl,
                    "ILD": ild,
                    "condition_gamma": float(np.mean(samples[:, 0])),
                    "condition_omega": float(np.mean(samples[:, 1])),
                    "condition_t_E_aff_s": float(np.mean(samples[:, 2])),
                    "condition_t_E_aff_ms": float(1e3 * np.mean(samples[:, 2])),
                    "source_pkl": str(pkl_path),
                    "extraction_python": sys.executable,
                }
            )
        except Exception as exc:
            failures.append((pkl_path, f"{type(exc).__name__}: {exc}"))

    return rows, failures


def ensure_compat_env():
    python_path = COMPAT_ENV_DIR / "bin" / "python"
    if python_path.exists():
        return python_path
    if not AUTO_CREATE_COMPAT_ENV:
        return None

    print(f"Creating condition-pickle compatibility env: {COMPAT_ENV_DIR}")
    subprocess.run(["uv", "venv", str(COMPAT_ENV_DIR), "--python", "3.12"], cwd=REPO_DIR, check=True)
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python_path),
            "numpy>=2.0",
            "pandas>=2.3",
            "scipy>=1.14",
            "matplotlib",
            "pyvbmc==1.0.4",
            "corner",
            "joblib",
        ],
        cwd=REPO_DIR,
        check=True,
    )
    return python_path


def extract_condition_means_with_compat_env(cache_csv):
    compat_python = ensure_compat_env()
    if compat_python is None:
        return False

    print(f"Extracting condition means with compatibility Python: {compat_python}")
    cmd = [
        str(compat_python),
        str(Path(__file__).resolve()),
        "--extract-condition-cache",
        str(cache_csv),
    ]
    subprocess.run(cmd, cwd=REPO_DIR, check=True)
    return True


def load_or_extract_condition_means():
    if COND_CACHE_CSV.exists() and not REFRESH_COND_CACHE:
        df = pd.read_csv(COND_CACHE_CSV)
        if len(df) == EXPECTED_N_COND_FITS:
            print(f"Loaded cached condition means: {COND_CACHE_CSV}")
            return df
        print(f"Refreshing incomplete cache {COND_CACHE_CSV}: {len(df)} rows")

    if SHARED_COND_CACHE_CSV.exists() and not REFRESH_COND_CACHE:
        df = pd.read_csv(SHARED_COND_CACHE_CSV)
        if len(df) == EXPECTED_N_COND_FITS:
            print(f"Loaded shared cached condition means: {SHARED_COND_CACHE_CSV}")
            df.to_csv(COND_CACHE_CSV, index=False)
            print(f"Copied condition extraction cache: {COND_CACHE_CSV}")
            return df
        print(f"Ignoring incomplete shared cache {SHARED_COND_CACHE_CSV}: {len(df)} rows")

    rows, failures = try_extract_condition_means_with_current_python()
    if len(rows) == EXPECTED_N_COND_FITS:
        df = pd.DataFrame(rows).sort_values(["batch_name", "animal", "ABL", "ILD"])
        df.to_csv(COND_CACHE_CSV, index=False)
        print(f"Saved condition extraction cache: {COND_CACHE_CSV}")
        return df

    print(
        f"Current Python extracted {len(rows)}/{EXPECTED_N_COND_FITS} condition fits; "
        f"using compatibility extraction for the full set."
    )
    if failures:
        print("First extraction failures:")
        for pkl_path, error in failures[:5]:
            print(f"  {pkl_path.name}: {error[:180]}")

    if extract_condition_means_with_compat_env(COND_CACHE_CSV):
        df = pd.read_csv(COND_CACHE_CSV)
        if len(df) == EXPECTED_N_COND_FITS:
            return df
        raise RuntimeError(f"Compatibility extraction wrote {len(df)} rows, expected {EXPECTED_N_COND_FITS}")

    raise RuntimeError(
        "Could not extract all condition-fit posterior means. "
        "Set COND_PICKLE_COMPAT_ENV to a Python env that can unpickle the ganon VBMC files, "
        "or allow AUTO_CREATE_COMPAT_ENV=1."
    )


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

    return least_squares(
        residuals,
        P0_FIT,
        bounds=(LOWER_BOUNDS, UPPER_BOUNDS),
    )


def params_from_fit_result(fit_result):
    return {name: float(value) for name, value in zip(PARAM_NAMES, fit_result.x)}


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
            params = params_from_fit_result(fit_result)
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

            condition_df = model_curves_for_params(params, FITTED_ILDS)
            condition_df.insert(0, "animal", animal_id)
            condition_df.insert(0, "batch_name", batch_name)
            condition_df["ILD"] = condition_df["ILD"].astype(int)
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


def build_agreement_points(cond_df, fixed_condition_df, previous_condition_df, mse_condition_df):
    agreement_df = cond_df.merge(
        fixed_condition_df,
        on=["batch_name", "animal", "ABL", "ILD"],
        how="left",
        validate="one_to_one",
    )
    if agreement_df["fixed_model_gamma"].isna().any() or agreement_df["fixed_model_omega"].isna().any():
        missing = agreement_df[agreement_df["fixed_model_gamma"].isna() | agreement_df["fixed_model_omega"].isna()]
        raise RuntimeError(f"Missing fixed-delay model condition values for {len(missing)} condition rows")

    agreement_df = agreement_df.merge(
        previous_condition_df,
        on=["batch_name", "animal", "ABL", "ILD"],
        how="left",
        validate="one_to_one",
    )
    if agreement_df["previous_model_gamma"].isna().any() or agreement_df["previous_model_omega"].isna().any():
        missing = agreement_df[
            agreement_df["previous_model_gamma"].isna() | agreement_df["previous_model_omega"].isna()
        ]
        raise RuntimeError(f"Missing previous ABL-specific ILD2 model condition values for {len(missing)} condition rows")

    agreement_df = agreement_df.merge(
        mse_condition_df,
        on=["batch_name", "animal", "ABL", "ILD"],
        how="left",
        validate="one_to_one",
    )
    missing_mse = agreement_df["mse_model_gamma"].isna() | agreement_df["mse_model_omega"].isna()
    if missing_mse.any():
        print(f"Warning: missing MSE model values for {int(np.sum(missing_mse))} condition rows")

    return agreement_df.sort_values(["batch_name", "animal", "ABL", "ILD"]).reset_index(drop=True)


def build_agreement_summary(agreement_df):
    rows = []
    value_prefixes = ["condition", "fixed_model", "previous_model", "mse_model"]
    for (abl, ild), group in agreement_df.groupby(["ABL", "ILD"], sort=True):
        row = {"ABL": int(abl), "ILD": int(ild)}
        for prefix in value_prefixes:
            for param in ["gamma", "omega"]:
                col = f"{prefix}_{param}"
                values = group[col].to_numpy(dtype=float)
                finite = np.isfinite(values)
                n = int(np.sum(finite))
                row[f"{prefix}_{param}_mean"] = float(np.nanmean(values)) if n > 0 else np.nan
                row[f"{prefix}_{param}_sem"] = float(np.nanstd(values) / np.sqrt(n)) if n > 0 else np.nan
                row[f"n_{prefix}_{param}"] = n
        rows.append(row)
    return pd.DataFrame(rows)


def compute_agreement_metrics(agreement_summary_df):
    rows = []
    sources = [
        ("fixed_model", MODEL_LABEL),
        ("previous_model", PREVIOUS_MODEL_LABEL),
        ("mse_model", MSE_LABEL),
    ]
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
                    mean_diff = float(np.mean(diff))
                    rmse = float(np.sqrt(np.mean(diff**2)))
                else:
                    r = mean_diff = rmse = np.nan
                rows.append(
                    {
                        "source": source,
                        "source_label": source_label,
                        "parameter": param,
                        "ABL": abl,
                        "n_points": int(np.sum(finite)),
                        "pearson_r": r,
                        "mean_condition_minus_model": mean_diff,
                        "rmse": rmse,
                    }
                )
    return pd.DataFrame(rows)


def print_param_means(label, param_df):
    print(f"{label} parameter means across animals:")
    for param_name in PARAM_NAMES:
        values = param_df[param_name].to_numpy(dtype=float)
        finite = np.isfinite(values)
        if param_name == "T_0":
            print(
                f"  {param_name}: {np.nanmean(values) * 1e3:.6g} ms "
                f"+/- {np.nanstd(values[finite]) / np.sqrt(np.sum(finite)) * 1e3:.6g} SEM"
            )
        else:
            print(
                f"  {param_name}: {np.nanmean(values):.6g} "
                f"+/- {np.nanstd(values[finite]) / np.sqrt(np.sum(finite)):.6g} SEM"
            )


def plot_gamma_omega(
    cond_summary_df,
    fixed_continuous_summary_df,
    previous_continuous_summary_df,
    mse_continuous_summary_df,
    metrics_df,
):
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.0))
    plot_specs = [
        ("gamma", "Gamma"),
        ("omega", "Omega"),
    ]

    for ax, (param, title) in zip(axes, plot_specs):
        for abl in ABLS:
            color = COLORS[abl]
            fixed_model = fixed_continuous_summary_df[
                fixed_continuous_summary_df["ABL"] == abl
            ].sort_values("ILD")
            previous_model = previous_continuous_summary_df[
                previous_continuous_summary_df["ABL"] == abl
            ].sort_values("ILD")
            mse_model = mse_continuous_summary_df[
                mse_continuous_summary_df["ABL"] == abl
            ].sort_values("ILD")
            cond = cond_summary_df[cond_summary_df["ABL"] == abl].sort_values("ILD")

            x = fixed_model["ILD"].to_numpy(dtype=float)
            y = fixed_model[f"fixed_model_{param}_mean"].to_numpy(dtype=float)
            sem = fixed_model[f"fixed_model_{param}_sem"].to_numpy(dtype=float)
            ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.12, linewidth=0)
            ax.plot(x, y, color=color, linestyle="-", linewidth=2.2)

            x = previous_model["ILD"].to_numpy(dtype=float)
            y = previous_model[f"previous_model_{param}_mean"].to_numpy(dtype=float)
            sem = previous_model[f"previous_model_{param}_sem"].to_numpy(dtype=float)
            ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.06, linewidth=0)
            ax.plot(x, y, color=color, linestyle=":", linewidth=2.4)

            x = mse_model["ILD"].to_numpy(dtype=float)
            y = mse_model[f"mse_model_{param}_mean"].to_numpy(dtype=float)
            sem = mse_model[f"mse_model_{param}_sem"].to_numpy(dtype=float)
            ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.07, linewidth=0)
            ax.plot(x, y, color=color, linestyle="--", linewidth=2.0)

            ax.errorbar(
                cond["ILD"].to_numpy(dtype=float),
                cond[f"condition_{param}_mean"].to_numpy(dtype=float),
                yerr=cond[f"condition_{param}_sem"].to_numpy(dtype=float),
                marker="o",
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.4,
                linestyle="none",
                color=color,
                ecolor=color,
                capsize=2.5,
                markersize=4.8,
                zorder=3,
            )

        fixed_metric = metrics_df[
            (metrics_df["source"] == "fixed_model")
            & (metrics_df["parameter"] == param)
            & (metrics_df["ABL"].astype(str) == "all")
        ].iloc[0]
        previous_metric = metrics_df[
            (metrics_df["source"] == "previous_model")
            & (metrics_df["parameter"] == param)
            & (metrics_df["ABL"].astype(str) == "all")
        ].iloc[0]
        mse_metric = metrics_df[
            (metrics_df["source"] == "mse_model")
            & (metrics_df["parameter"] == param)
            & (metrics_df["ABL"].astype(str) == "all")
        ].iloc[0]
        ax.text(
            0.04,
            0.96,
            (
                "all ABLs RMSE\n"
                f"fixed={fixed_metric['rmse']:.3g}\n"
                f"ILD2={previous_metric['rmse']:.3g}\n"
                f"MSE={mse_metric['rmse']:.3g}"
            ),
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.85},
        )
        ax.set_title(title)
        ax.set_xlabel("ILD")
        ax.set_xticks([-16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16])
        ax.grid(True, alpha=0.25)

    axes[0].axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.4)
    axes[0].set_ylabel("Gamma")
    axes[1].set_ylabel("Omega")

    abl_handles = [Line2D([0], [0], color=COLORS[abl], linewidth=2.2, label=f"ABL={abl}") for abl in ABLS]
    source_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            markerfacecolor="white",
            markeredgecolor="black",
            linestyle="none",
            label="condition fit",
        ),
        Line2D([0], [0], color="black", linestyle="-", linewidth=2.2, label=MODEL_LABEL),
        Line2D([0], [0], color="black", linestyle=":", linewidth=2.4, label=PREVIOUS_MODEL_LABEL),
        Line2D([0], [0], color="black", linestyle="--", linewidth=2.0, label=MSE_LABEL),
    ]
    fig.legend(
        handles=abl_handles + source_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
    )
    fig.suptitle("Condition Gamma/Omega vs NPL+alpha model variants", y=1.17)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(FIG_PDF, bbox_inches="tight")
    plt.close(fig)
    return FIG_PNG, FIG_PDF


# %%
# =============================================================================
# Extraction-only mode for compatibility env
# =============================================================================
if len(sys.argv) >= 3 and sys.argv[1] == "--extract-condition-cache":
    out_csv = Path(sys.argv[2])
    rows, failures = try_extract_condition_means_with_current_python()
    if failures:
        print(f"Failed to extract {len(failures)} condition fit files.")
        for pkl_path, error in failures[:20]:
            print(f"  {pkl_path}: {error}")
        raise SystemExit(1)
    df = pd.DataFrame(rows).sort_values(["batch_name", "animal", "ABL", "ILD"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} condition rows to {out_csv}")
    raise SystemExit(0)


# %%
# =============================================================================
# Main analysis
# =============================================================================
print(f"Upstream folder: {UPSTREAM_DIR}")
print(f"Previous ABL-specific ILD2 folder: {PREVIOUS_UPSTREAM_DIR}")
print(f"Condition pickle folder: {COND_PKL_DIR}")
print(f"Output folder: {OUTPUT_DIR}")

param_df, fixed_continuous_df, fixed_condition_df = load_model_gamma_omega(
    UPSTREAM_DIR,
    "results_*_NORM_ALPHA_CONDITION_T_E_AFF_FIXED_DELAY_FROM_ABORTS.pkl",
    parse_upstream_result_name,
    MODEL_KEY,
    "fixed_model",
    EXPECTED_N_UPSTREAM_ANIMALS,
    MODEL_LABEL,
)
param_df.to_csv(MODEL_PARAM_CSV, index=False)
fixed_condition_df.to_csv(MODEL_CONDITION_VALUES_CSV, index=False)
print(f"Loaded stable {MODEL_LABEL} fits for {param_df[['batch_name', 'animal']].drop_duplicates().shape[0]} animals")
print(f"Saved fixed-delay model parameter summary: {MODEL_PARAM_CSV}")
print(f"Saved fixed-delay model condition values: {MODEL_CONDITION_VALUES_CSV}")
print_param_means(MODEL_LABEL, param_df)

previous_param_df, previous_continuous_df, previous_condition_df = load_model_gamma_omega(
    PREVIOUS_UPSTREAM_DIR,
    "results_*_NORM_ALPHA_ABL_SPECIFIC_ILD2_DELAY_FROM_ABORTS.pkl",
    parse_previous_upstream_result_name,
    PREVIOUS_MODEL_KEY,
    "previous_model",
    EXPECTED_N_UPSTREAM_ANIMALS,
    PREVIOUS_MODEL_LABEL,
)
fixed_animals = set(zip(param_df["batch_name"], param_df["animal"].astype(int)))
previous_animals = set(zip(previous_param_df["batch_name"], previous_param_df["animal"].astype(int)))
if fixed_animals != previous_animals:
    missing_previous = sorted(fixed_animals - previous_animals)
    missing_fixed = sorted(previous_animals - fixed_animals)
    raise RuntimeError(
        "Fixed-delay and previous ABL-specific ILD2 animal sets differ. "
        f"Missing previous={missing_previous}; missing fixed={missing_fixed}"
    )
previous_param_df.to_csv(PREVIOUS_MODEL_PARAM_CSV, index=False)
previous_condition_df.to_csv(PREVIOUS_MODEL_CONDITION_VALUES_CSV, index=False)
print(
    f"Loaded stable {PREVIOUS_MODEL_LABEL} fits for "
    f"{previous_param_df[['batch_name', 'animal']].drop_duplicates().shape[0]} animals"
)
print(f"Saved previous ABL-specific ILD2 parameter summary: {PREVIOUS_MODEL_PARAM_CSV}")
print(f"Saved previous ABL-specific ILD2 condition values: {PREVIOUS_MODEL_CONDITION_VALUES_CSV}")
print_param_means(PREVIOUS_MODEL_LABEL, previous_param_df)

cond_df = load_or_extract_condition_means()
if len(cond_df) != EXPECTED_N_COND_FITS:
    raise RuntimeError(f"Expected {EXPECTED_N_COND_FITS} condition rows, found {len(cond_df)}")
cond_df = cond_df.sort_values(["batch_name", "animal", "ABL", "ILD"]).reset_index(drop=True)
cond_df.to_csv(COND_ANIMAL_VALUES_CSV, index=False)

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
print(f"Loaded condition rows: {len(cond_df)}")
print("Condition counts at |ILD|=16:")
print(edge_ild_counts.to_string(index=False))
print(f"Saved condition animal values: {COND_ANIMAL_VALUES_CSV}")
print(f"Saved condition summary: {COND_SUMMARY_CSV}")

fixed_continuous_summary_df = summarize_group_values(
    fixed_continuous_df,
    ["ABL", "ILD"],
    ["fixed_model_gamma", "fixed_model_omega"],
    "fixed_model",
)
fixed_continuous_summary_df.to_csv(MODEL_CONTINUOUS_SUMMARY_CSV, index=False)
print(f"Saved fixed-delay model continuous summary: {MODEL_CONTINUOUS_SUMMARY_CSV}")

previous_continuous_summary_df = summarize_group_values(
    previous_continuous_df,
    ["ABL", "ILD"],
    ["previous_model_gamma", "previous_model_omega"],
    "previous_model",
)
previous_continuous_summary_df.to_csv(PREVIOUS_MODEL_CONTINUOUS_SUMMARY_CSV, index=False)
print(f"Saved previous ABL-specific ILD2 continuous summary: {PREVIOUS_MODEL_CONTINUOUS_SUMMARY_CSV}")

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

agreement_df = build_agreement_points(cond_df, fixed_condition_df, previous_condition_df, mse_condition_df)
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
    fixed_continuous_summary_df,
    previous_continuous_summary_df,
    mse_continuous_summary_df,
    metrics_df,
)
print(f"Saved figure: {png_path}")
print(f"Saved figure: {pdf_path}")

# %%
