# %%
"""
Compare 3-param condition-fit t_E_aff with NPL+alpha ABL-specific ILD2 delays.

The condition fits estimate gamma, omega, and t_E_aff per ABL/ILD while fixing
w and del_go from the animal-wise ABL-specific NPL+alpha+ILD2 fit. This script
compares those condition-wise t_E_aff values against the animal-wise delay
curve:

    delay_ms = bias_ms[ABL] + abs_ild_coeff[ABL] * |ILD| + ild2_coeff[ABL] * ILD^2

Two NPL averaging policies are plotted:
- observed_range: SD animals contribute only through |ILD| <= 8.
- full_range: all animals contribute through |ILD| <= 16.
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


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

OUTPUT_DIR = SCRIPT_DIR / "abl_specific_ild2_delay_agreement"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COND_CACHE_CSV = OUTPUT_DIR / "condition_t_E_aff_extraction_cache.csv"
COND_ANIMAL_VALUES_CSV = OUTPUT_DIR / "condition_t_E_aff_animal_values.csv"
COND_SUMMARY_CSV = OUTPUT_DIR / "condition_t_E_aff_summary.csv"
METRICS_CSV = OUTPUT_DIR / "delay_agreement_metrics.csv"

MODEL_KEY = "vbmc_norm_alpha_abl_specific_ild2_delay_tied_results"
ABLS = [20, 40, 60]
CONTINUOUS_ILDS = np.round(np.arange(-16.0, 16.0 + 0.05, 0.1), 10)
FITTED_ILDS = [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]
COLORS = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}
POLICIES = ["observed_range", "full_range"]

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


def mean_sem_n(arr, axis=0):
    arr = np.asarray(arr, dtype=float)
    n = np.sum(np.isfinite(arr), axis=axis)
    mean = np.full(arr.shape[1], np.nan)
    sem = np.full(arr.shape[1], np.nan)
    ok = n > 0
    if np.any(ok):
        mean[ok] = np.nanmean(arr[:, ok], axis=0)
        sem[ok] = np.nanstd(arr[:, ok], axis=0) / np.sqrt(n[ok])
    return mean, sem, n


def load_upstream_delay_curves():
    rows = []
    animal_curves = {policy: [] for policy in POLICIES}

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

        abl_levels = np.asarray(result["delay_abl_levels"], dtype=float)
        bias = np.mean(np.asarray(result["bias_ms_by_abl_samples"], dtype=float), axis=0)
        abs_coeff = np.mean(
            np.asarray(result["abs_ild_delay_coeff_ms_per_unit_by_abl_samples"], dtype=float),
            axis=0,
        )
        ild2_coeff = np.mean(
            np.asarray(result["ild2_delay_coeff_ms_per_unit2_by_abl_samples"], dtype=float),
            axis=0,
        )

        for abl in ABLS:
            abl_matches = np.where(np.isclose(abl_levels, abl))[0]
            if len(abl_matches) != 1:
                raise RuntimeError(f"{batch_name}/{animal_id} has ABL levels {abl_levels}, cannot find ABL={abl}")
            abl_idx = int(abl_matches[0])
            delay_ms = (
                bias[abl_idx]
                + abs_coeff[abl_idx] * np.abs(CONTINUOUS_ILDS)
                + ild2_coeff[abl_idx] * (CONTINUOUS_ILDS ** 2)
            )

            full_curve = delay_ms.copy()
            observed_curve = delay_ms.copy()
            if batch_name == "SD":
                observed_curve[np.abs(CONTINUOUS_ILDS) > 8.0] = np.nan

            for policy, curve in [("full_range", full_curve), ("observed_range", observed_curve)]:
                for ild, value in zip(CONTINUOUS_ILDS, curve):
                    animal_curves[policy].append(
                        {
                            "policy": policy,
                            "batch_name": batch_name,
                            "animal": animal_id,
                            "ABL": abl,
                            "ILD": float(ild),
                            "npl_delay_ms": float(value) if np.isfinite(value) else np.nan,
                        }
                    )

            rows.append(
                {
                    "batch_name": batch_name,
                    "animal": animal_id,
                    "ABL": abl,
                    "bias_ms": bias[abl_idx],
                    "abs_ild_coeff_ms_per_unit": abs_coeff[abl_idx],
                    "ild2_coeff_ms_per_unit2": ild2_coeff[abl_idx],
                    "message": message,
                }
            )

    return pd.DataFrame(rows), {policy: pd.DataFrame(animal_curves[policy]) for policy in POLICIES}


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
                    "gamma": float(np.mean(samples[:, 0])),
                    "omega": float(np.mean(samples[:, 1])),
                    "t_E_aff_s": float(np.mean(samples[:, 2])),
                    "t_E_aff_ms": float(1e3 * np.mean(samples[:, 2])),
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


def summarize_condition_values(cond_df):
    rows = []
    for (abl, ild), group in cond_df.groupby(["ABL", "ILD"], sort=True):
        values = group["t_E_aff_ms"].to_numpy(dtype=float)
        n = int(np.sum(np.isfinite(values)))
        rows.append(
            {
                "ABL": int(abl),
                "ILD": int(ild),
                "condition_t_E_aff_mean_ms": float(np.nanmean(values)),
                "condition_t_E_aff_sem_ms": float(np.nanstd(values) / np.sqrt(n)),
                "n_animals": n,
            }
        )
    return pd.DataFrame(rows)


def summarize_model_curves(model_animal_df, policy):
    rows = []
    for (abl, ild), group in model_animal_df.groupby(["ABL", "ILD"], sort=True):
        values = group["npl_delay_ms"].to_numpy(dtype=float)
        n = int(np.sum(np.isfinite(values)))
        rows.append(
            {
                "policy": policy,
                "ABL": int(abl),
                "ILD": float(ild),
                "npl_delay_mean_ms": float(np.nanmean(values)) if n > 0 else np.nan,
                "npl_delay_sem_ms": float(np.nanstd(values) / np.sqrt(n)) if n > 0 else np.nan,
                "n_animals": n,
            }
        )
    return pd.DataFrame(rows)


def compute_animal_agreement_points(policy, cond_df, model_animal_df):
    point_rows = []
    model_lookup = {
        (row.batch_name, int(row.animal), int(row.ABL), round(float(row.ILD), 10)): row.npl_delay_ms
        for row in model_animal_df.itertuples(index=False)
    }

    for row in cond_df.itertuples(index=False):
        key = (row.batch_name, int(row.animal), int(row.ABL), round(float(row.ILD), 10))
        model_value = model_lookup.get(key, np.nan)
        point_rows.append(
            {
                "policy": policy,
                "batch_name": row.batch_name,
                "animal": int(row.animal),
                "ABL": int(row.ABL),
                "ILD": int(row.ILD),
                "condition_t_E_aff_ms": float(row.t_E_aff_ms),
                "npl_delay_ms": float(model_value) if np.isfinite(model_value) else np.nan,
            }
        )
    return pd.DataFrame(point_rows)


def build_summary_agreement_points(policy, cond_summary_df, model_summary_df):
    rows = []
    for cond_row in cond_summary_df.itertuples(index=False):
        model_match = model_summary_df[
            (model_summary_df["ABL"] == int(cond_row.ABL))
            & np.isclose(model_summary_df["ILD"].to_numpy(dtype=float), float(cond_row.ILD))
        ]
        if len(model_match) != 1:
            raise RuntimeError(f"Could not find model summary point for ABL={cond_row.ABL}, ILD={cond_row.ILD}")
        model_row = model_match.iloc[0]
        rows.append(
            {
                "policy": policy,
                "ABL": int(cond_row.ABL),
                "ILD": int(cond_row.ILD),
                "condition_t_E_aff_mean_ms": float(cond_row.condition_t_E_aff_mean_ms),
                "condition_t_E_aff_sem_ms": float(cond_row.condition_t_E_aff_sem_ms),
                "condition_n_animals": int(cond_row.n_animals),
                "npl_delay_mean_ms": float(model_row.npl_delay_mean_ms),
                "npl_delay_sem_ms": float(model_row.npl_delay_sem_ms),
                "npl_n_animals": int(model_row.n_animals),
            }
        )
    return pd.DataFrame(rows)


def compute_agreement_metrics(policy, summary_point_df):
    rows = []
    for abl in ABLS + ["all"]:
        subset = summary_point_df if abl == "all" else summary_point_df[summary_point_df["ABL"] == abl]
        x = subset["npl_delay_mean_ms"].to_numpy(dtype=float)
        y = subset["condition_t_E_aff_mean_ms"].to_numpy(dtype=float)
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
                "policy": policy,
                "ABL": abl,
                "n_points": int(np.sum(finite)),
                "pearson_r": r,
                "mean_condition_minus_npl_ms": mean_diff,
                "rmse_ms": rmse,
            }
        )
    return pd.DataFrame(rows)


def plot_policy(policy, cond_summary_df, model_summary_df, summary_agreement_df, metrics_df):
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(14, 8),
        gridspec_kw={"height_ratios": [1.15, 1.0]},
        sharex=False,
    )

    for col, abl in enumerate(ABLS):
        color = COLORS[abl]
        ax = axes[0, col]
        model = model_summary_df[model_summary_df["ABL"] == abl].sort_values("ILD")
        cond = cond_summary_df[cond_summary_df["ABL"] == abl].sort_values("ILD")

        x = model["ILD"].to_numpy(dtype=float)
        y = model["npl_delay_mean_ms"].to_numpy(dtype=float)
        sem = model["npl_delay_sem_ms"].to_numpy(dtype=float)
        ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.15, linewidth=0)
        ax.plot(x, y, color=color, linewidth=2)
        ax.errorbar(
            cond["ILD"].to_numpy(dtype=float),
            cond["condition_t_E_aff_mean_ms"].to_numpy(dtype=float),
            yerr=cond["condition_t_E_aff_sem_ms"].to_numpy(dtype=float),
            marker="o",
            linestyle="none",
            color=color,
            capsize=3,
            zorder=3,
        )
        ax.set_title(f"ABL={abl}")
        ax.set_xlabel("ILD")
        if col == 0:
            ax.set_ylabel("delay / t_E_aff (ms)")
        ax.set_xlim(-16.5, 16.5)
        ax.set_xticks([-16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16])
        ax.grid(True, alpha=0.25)

        scatter_ax = axes[1, col]
        points = summary_agreement_df[summary_agreement_df["ABL"] == abl]
        scatter_ax.errorbar(
            points["npl_delay_mean_ms"],
            points["condition_t_E_aff_mean_ms"],
            xerr=points["npl_delay_sem_ms"],
            yerr=points["condition_t_E_aff_sem_ms"],
            marker="o",
            linestyle="none",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=5,
            alpha=0.9,
        )
        finite_x = points["npl_delay_mean_ms"].to_numpy(dtype=float)
        finite_y = points["condition_t_E_aff_mean_ms"].to_numpy(dtype=float)
        finite = np.isfinite(finite_x) & np.isfinite(finite_y)
        if np.any(finite):
            lo = float(np.nanmin([np.nanmin(finite_x[finite]), np.nanmin(finite_y[finite])]))
            hi = float(np.nanmax([np.nanmax(finite_x[finite]), np.nanmax(finite_y[finite])]))
            pad = max(2.0, 0.05 * (hi - lo))
            scatter_ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.4", linestyle="--", linewidth=1)
            scatter_ax.set_xlim(lo - pad, hi + pad)
            scatter_ax.set_ylim(lo - pad, hi + pad)
        metric = metrics_df[(metrics_df["ABL"].astype(str) == str(abl))].iloc[0]
        scatter_ax.text(
            0.04,
            0.96,
            (
                f"n={int(metric['n_points'])}\n"
                f"r={metric['pearson_r']:.2f}\n"
                f"RMSE={metric['rmse_ms']:.1f} ms"
            ),
            transform=scatter_ax.transAxes,
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.8},
        )
        scatter_ax.set_xlabel("NPL delay (ms)")
        if col == 0:
            scatter_ax.set_ylabel("condition t_E_aff (ms)")
        scatter_ax.grid(True, alpha=0.25)

    policy_title = "SD observed range (|ILD| <= 8)" if policy == "observed_range" else "Full range for all animals"
    handles = [
        Line2D([0], [0], color="black", linewidth=2, label="NPL+alpha ABL-specific ILD2 mean +/- SEM"),
        Line2D([0], [0], marker="o", color="black", linestyle="none", label="condition t_E_aff mean +/- SEM"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle(f"Condition t_E_aff vs NPL delay: {policy_title}", y=1.03)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    png_path = OUTPUT_DIR / f"cond_t_E_aff_vs_npl_alpha_abl_specific_ild2_delay_{policy}.png"
    pdf_path = OUTPUT_DIR / f"cond_t_E_aff_vs_npl_alpha_abl_specific_ild2_delay_{policy}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def plot_combined_policy_comparison(cond_summary_df, model_summary_by_policy):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    panel_specs = [
        ("full_range", "All ILDs"),
        ("observed_range", "SD limited to |ILD| <= 8"),
    ]

    for ax, (policy, title) in zip(axes, panel_specs):
        model_summary_df = model_summary_by_policy[policy]
        for abl in ABLS:
            color = COLORS[abl]
            model = model_summary_df[model_summary_df["ABL"] == abl].sort_values("ILD")
            cond = cond_summary_df[cond_summary_df["ABL"] == abl].sort_values("ILD")

            x = model["ILD"].to_numpy(dtype=float)
            y = model["npl_delay_mean_ms"].to_numpy(dtype=float)
            sem = model["npl_delay_sem_ms"].to_numpy(dtype=float)
            ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.12, linewidth=0)
            ax.plot(x, y, color=color, linewidth=2.2)

            ax.errorbar(
                cond["ILD"].to_numpy(dtype=float),
                cond["condition_t_E_aff_mean_ms"].to_numpy(dtype=float),
                yerr=cond["condition_t_E_aff_sem_ms"].to_numpy(dtype=float),
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

        ax.set_title(title)
        ax.set_xlabel("ILD")
        ax.set_xlim(-16.5, 16.5)
        ax.set_xticks([-16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16])
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("delay / t_E_aff (ms)")
    abl_handles = [Line2D([0], [0], color=COLORS[abl], linewidth=2.2, label=f"ABL={abl}") for abl in ABLS]
    model_handles = [
        Line2D([0], [0], color="black", linewidth=2.2, label="NPL+alpha ABL-specific ILD2"),
        Line2D(
            [0],
            [0],
            marker="o",
            markerfacecolor="white",
            markeredgecolor="black",
            linestyle="none",
            label="condition t_E_aff",
        ),
    ]
    fig.legend(
        handles=abl_handles + model_handles,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
    )
    fig.suptitle("Condition t_E_aff vs NPL delay", y=1.11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    png_path = OUTPUT_DIR / "cond_t_E_aff_vs_npl_alpha_abl_specific_ild2_delay_1x2_policy_comparison.png"
    pdf_path = OUTPUT_DIR / "cond_t_E_aff_vs_npl_alpha_abl_specific_ild2_delay_1x2_policy_comparison.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


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

coeff_df, model_animal_by_policy = load_upstream_delay_curves()
print(f"Loaded stable upstream fits for {coeff_df[['batch_name', 'animal']].drop_duplicates().shape[0]} animals")

cond_df = load_or_extract_condition_means()
if len(cond_df) != EXPECTED_N_COND_FITS:
    raise RuntimeError(f"Expected {EXPECTED_N_COND_FITS} condition rows, found {len(cond_df)}")
cond_df = cond_df.sort_values(["batch_name", "animal", "ABL", "ILD"]).reset_index(drop=True)
cond_df.to_csv(COND_ANIMAL_VALUES_CSV, index=False)

cond_summary_df = summarize_condition_values(cond_df)
cond_summary_df.to_csv(COND_SUMMARY_CSV, index=False)
print(f"Saved condition animal values: {COND_ANIMAL_VALUES_CSV}")
print(f"Saved condition summary: {COND_SUMMARY_CSV}")

all_metrics = []
model_summary_by_policy = {}
for policy in POLICIES:
    model_animal_df = model_animal_by_policy[policy]
    model_animal_csv = OUTPUT_DIR / f"npl_delay_animal_curves_{policy}.csv"
    model_summary_csv = OUTPUT_DIR / f"npl_delay_continuous_summary_{policy}.csv"

    model_summary_df = summarize_model_curves(model_animal_df, policy)
    model_summary_by_policy[policy] = model_summary_df
    model_animal_df.to_csv(model_animal_csv, index=False)
    model_summary_df.to_csv(model_summary_csv, index=False)

    animal_agreement_df = compute_animal_agreement_points(policy, cond_df, model_animal_df)
    summary_agreement_df = build_summary_agreement_points(policy, cond_summary_df, model_summary_df)
    metrics_df = compute_agreement_metrics(policy, summary_agreement_df)
    all_metrics.append(metrics_df)
    agreement_point_csv = OUTPUT_DIR / f"delay_agreement_points_{policy}.csv"
    animal_agreement_point_csv = OUTPUT_DIR / f"delay_agreement_animal_points_{policy}.csv"
    summary_agreement_point_csv = OUTPUT_DIR / f"delay_agreement_summary_points_{policy}.csv"
    summary_agreement_df.to_csv(agreement_point_csv, index=False)
    summary_agreement_df.to_csv(summary_agreement_point_csv, index=False)
    animal_agreement_df.to_csv(animal_agreement_point_csv, index=False)

    png_path, pdf_path = plot_policy(policy, cond_summary_df, model_summary_df, summary_agreement_df, metrics_df)

    print(f"\nPolicy: {policy}")
    print(f"  Saved model animal curves: {model_animal_csv}")
    print(f"  Saved model summary: {model_summary_csv}")
    print(f"  Saved agreement summary points: {agreement_point_csv}")
    print(f"  Saved agreement animal points: {animal_agreement_point_csv}")
    print(f"  Saved figure: {png_path}")
    print(f"  Saved figure: {pdf_path}")
    print(metrics_df.to_string(index=False))

metrics_out = pd.concat(all_metrics, ignore_index=True)
metrics_out.to_csv(METRICS_CSV, index=False)
print(f"\nSaved agreement metrics: {METRICS_CSV}")

combined_png_path, combined_pdf_path = plot_combined_policy_comparison(cond_summary_df, model_summary_by_policy)
print(f"Saved compact 1x2 comparison figure: {combined_png_path}")
print(f"Saved compact 1x2 comparison figure: {combined_pdf_path}")

# %%
