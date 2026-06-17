# %%
"""
Compare condition-by-condition Gamma/Omega fits with NPL+alpha ABL-specific ILD2 fits.

The condition fits estimate gamma, omega, and t_E_aff per ABL/ILD while fixing
w and del_go from the animal-wise NPL+alpha+ABL-specific ILD2-delay fit. Gamma
and Omega are still implied by the NPL+alpha firing-rate expression:

    gamma, omega = gamma_omega_alpha_model(ABL, ILD, rate_lambda, ell, alpha, theta, T_0, P_0)

This script compares those condition-wise Gamma/Omega posterior means against
the animal-wise model-implied Gamma/Omega curves.
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

from gamma_omega_alpha_utils import gamma_omega_alpha_model


# %%
# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

UPSTREAM_DIR = REPO_DIR / "all_30_NPL_alpha_ABL_specific_ILD2_delay_fit_results"
COND_PKL_DIR = (
    REPO_DIR
    / "all_30_cond_by_cond_gamma_omega_t_E_aff_fix_w_del_go_from_NPL_alpha_ABL_specific_ILD2_fit_results"
    / "pkl_files"
)

OUTPUT_DIR = SCRIPT_DIR / "abl_specific_ild2_gamma_omega_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COND_CACHE_CSV = OUTPUT_DIR / "condition_gamma_omega_extraction_cache.csv"
COND_ANIMAL_VALUES_CSV = OUTPUT_DIR / "condition_gamma_omega_animal_values.csv"
COND_SUMMARY_CSV = OUTPUT_DIR / "condition_gamma_omega_summary.csv"
MODEL_PARAM_CSV = OUTPUT_DIR / "npl_alpha_abl_specific_ild2_params_by_animal.csv"
MODEL_CONTINUOUS_SUMMARY_CSV = OUTPUT_DIR / "npl_alpha_abl_specific_ild2_gamma_omega_summary.csv"
MODEL_CONDITION_VALUES_CSV = OUTPUT_DIR / "npl_alpha_abl_specific_ild2_gamma_omega_condition_values.csv"
AGREEMENT_POINTS_CSV = OUTPUT_DIR / "gamma_omega_agreement_points.csv"
METRICS_CSV = OUTPUT_DIR / "gamma_omega_agreement_metrics.csv"

FIG_PNG = OUTPUT_DIR / "cond_gamma_omega_vs_npl_alpha_abl_specific_ild2_delay.png"
FIG_PDF = OUTPUT_DIR / "cond_gamma_omega_vs_npl_alpha_abl_specific_ild2_delay.pdf"

MODEL_KEY = "vbmc_norm_alpha_abl_specific_ild2_delay_tied_results"
PARAM_SAMPLES_TO_MEAN = {
    "rate_lambda": "rate_lambda_samples",
    "ell": "rate_norm_l_samples",
    "alpha": "alpha_samples",
    "theta": "theta_E_samples",
    "T_0": "T_0_samples",
}

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
        r"^results_(?P<batch>.+)_animal_(?P<animal>\d+)_NORM_ALPHA_ABL_SPECIFIC_ILD2_DELAY_FROM_ABORTS\.pkl$",
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


def load_upstream_gamma_omega():
    param_rows = []
    smooth_rows = []
    condition_rows = []

    result_paths = sorted(UPSTREAM_DIR.glob("results_*_NORM_ALPHA_ABL_SPECIFIC_ILD2_DELAY_FROM_ABORTS.pkl"))
    if len(result_paths) != EXPECTED_N_UPSTREAM_ANIMALS:
        raise RuntimeError(f"Expected {EXPECTED_N_UPSTREAM_ANIMALS} upstream result pkls, found {len(result_paths)}")

    for result_path in result_paths:
        parsed = parse_upstream_result_name(result_path)
        if parsed is None:
            continue
        batch_name, animal_id = parsed

        with result_path.open("rb") as f:
            saved = pickle.load(f)
        if MODEL_KEY not in saved:
            raise KeyError(f"{result_path} is missing `{MODEL_KEY}`")

        result = saved[MODEL_KEY]
        message = str(result.get("message", ""))
        if "stable" not in message.lower():
            raise RuntimeError(f"Upstream fit is not stable for {batch_name}/{animal_id}: {message}")

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

        for abl in ABLS:
            gamma_smooth, omega_smooth = gamma_omega_alpha_model(
                abl,
                SMOOTH_ILDS,
                params["rate_lambda"],
                params["ell"],
                params["alpha"],
                params["theta"],
                params["T_0"],
                P_0,
            )
            for ild, gamma, omega in zip(SMOOTH_ILDS, gamma_smooth, omega_smooth):
                smooth_rows.append(
                    {
                        "batch_name": batch_name,
                        "animal": animal_id,
                        "ABL": abl,
                        "ILD": float(ild),
                        "model_gamma": float(gamma),
                        "model_omega": float(omega),
                    }
                )

            gamma_cond, omega_cond = gamma_omega_alpha_model(
                abl,
                np.asarray(FITTED_ILDS, dtype=float),
                params["rate_lambda"],
                params["ell"],
                params["alpha"],
                params["theta"],
                params["T_0"],
                P_0,
            )
            for ild, gamma, omega in zip(FITTED_ILDS, gamma_cond, omega_cond):
                condition_rows.append(
                    {
                        "batch_name": batch_name,
                        "animal": animal_id,
                        "ABL": abl,
                        "ILD": int(ild),
                        "model_gamma": float(gamma),
                        "model_omega": float(omega),
                    }
                )

    return pd.DataFrame(param_rows), pd.DataFrame(smooth_rows), pd.DataFrame(condition_rows)


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


def build_agreement_points(cond_df, model_condition_df):
    agreement_df = cond_df.merge(
        model_condition_df,
        on=["batch_name", "animal", "ABL", "ILD"],
        how="left",
        validate="one_to_one",
    )
    if agreement_df["model_gamma"].isna().any() or agreement_df["model_omega"].isna().any():
        missing = agreement_df[agreement_df["model_gamma"].isna() | agreement_df["model_omega"].isna()]
        raise RuntimeError(f"Missing model condition values for {len(missing)} condition rows")

    rows = []
    for (abl, ild), group in agreement_df.groupby(["ABL", "ILD"], sort=True):
        row = {"ABL": int(abl), "ILD": int(ild)}
        for param in ["gamma", "omega"]:
            cond_values = group[f"condition_{param}"].to_numpy(dtype=float)
            model_values = group[f"model_{param}"].to_numpy(dtype=float)
            finite = np.isfinite(cond_values) & np.isfinite(model_values)
            n = int(np.sum(finite))
            row[f"condition_{param}_mean"] = float(np.nanmean(cond_values)) if n > 0 else np.nan
            row[f"condition_{param}_sem"] = float(np.nanstd(cond_values) / np.sqrt(n)) if n > 0 else np.nan
            row[f"model_{param}_mean"] = float(np.nanmean(model_values)) if n > 0 else np.nan
            row[f"model_{param}_sem"] = float(np.nanstd(model_values) / np.sqrt(n)) if n > 0 else np.nan
            row[f"n_{param}_paired_animals"] = n
        rows.append(row)
    return pd.DataFrame(rows)


def compute_agreement_metrics(agreement_summary_df):
    rows = []
    for param in ["gamma", "omega"]:
        for abl in ABLS + ["all"]:
            subset = agreement_summary_df if abl == "all" else agreement_summary_df[agreement_summary_df["ABL"] == abl]
            x = subset[f"model_{param}_mean"].to_numpy(dtype=float)
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
                    "parameter": param,
                    "ABL": abl,
                    "n_points": int(np.sum(finite)),
                    "pearson_r": r,
                    "mean_condition_minus_model": mean_diff,
                    "rmse": rmse,
                }
            )
    return pd.DataFrame(rows)


def plot_gamma_omega(cond_summary_df, model_continuous_summary_df, metrics_df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    plot_specs = [
        ("gamma", "Gamma"),
        ("omega", "Omega"),
    ]

    for ax, (param, title) in zip(axes, plot_specs):
        for abl in ABLS:
            color = COLORS[abl]
            model = model_continuous_summary_df[model_continuous_summary_df["ABL"] == abl].sort_values("ILD")
            cond = cond_summary_df[cond_summary_df["ABL"] == abl].sort_values("ILD")

            x = model["ILD"].to_numpy(dtype=float)
            y = model[f"model_{param}_mean"].to_numpy(dtype=float)
            sem = model[f"model_{param}_sem"].to_numpy(dtype=float)
            ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.12, linewidth=0)
            ax.plot(x, y, color=color, linewidth=2.2)

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

        metric = metrics_df[(metrics_df["parameter"] == param) & (metrics_df["ABL"].astype(str) == "all")].iloc[0]
        ax.text(
            0.04,
            0.96,
            f"all ABLs\nr={metric['pearson_r']:.2f}\nRMSE={metric['rmse']:.3g}",
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
        Line2D([0], [0], color="black", linewidth=2.2, label="NPL+alpha ABL-specific ILD2"),
        Line2D(
            [0],
            [0],
            marker="o",
            markerfacecolor="white",
            markeredgecolor="black",
            linestyle="none",
            label="condition fit",
        ),
    ]
    fig.legend(
        handles=abl_handles + source_handles,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
    )
    fig.suptitle("Condition Gamma/Omega vs NPL+alpha ABL-specific ILD2", y=1.11)
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
print(f"Condition pickle folder: {COND_PKL_DIR}")
print(f"Output folder: {OUTPUT_DIR}")

param_df, model_continuous_df, model_condition_df = load_upstream_gamma_omega()
param_df.to_csv(MODEL_PARAM_CSV, index=False)
model_condition_df.to_csv(MODEL_CONDITION_VALUES_CSV, index=False)
print(f"Loaded stable upstream fits for {param_df[['batch_name', 'animal']].drop_duplicates().shape[0]} animals")
print(f"Saved model parameter summary: {MODEL_PARAM_CSV}")
print(f"Saved model condition values: {MODEL_CONDITION_VALUES_CSV}")

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
print(f"Saved condition animal values: {COND_ANIMAL_VALUES_CSV}")
print(f"Saved condition summary: {COND_SUMMARY_CSV}")

model_continuous_summary_df = summarize_group_values(
    model_continuous_df,
    ["ABL", "ILD"],
    ["model_gamma", "model_omega"],
    "model",
)
model_continuous_summary_df.to_csv(MODEL_CONTINUOUS_SUMMARY_CSV, index=False)
print(f"Saved model continuous summary: {MODEL_CONTINUOUS_SUMMARY_CSV}")

agreement_summary_df = build_agreement_points(cond_df, model_condition_df)
agreement_summary_df.to_csv(AGREEMENT_POINTS_CSV, index=False)
metrics_df = compute_agreement_metrics(agreement_summary_df)
metrics_df.to_csv(METRICS_CSV, index=False)
print(f"Saved agreement points: {AGREEMENT_POINTS_CSV}")
print(f"Saved agreement metrics: {METRICS_CSV}")
print(metrics_df.to_string(index=False))

png_path, pdf_path = plot_gamma_omega(cond_summary_df, model_continuous_summary_df, metrics_df)
print(f"Saved figure: {png_path}")
print(f"Saved figure: {pdf_path}")

# %%
