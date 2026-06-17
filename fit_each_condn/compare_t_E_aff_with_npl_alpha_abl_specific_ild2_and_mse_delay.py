# %%
"""
Compare condition t_E_aff with NPL+alpha ABL-specific ILD2 delay and MSE delay fits.

This script uses the full-ILD policy: no SD-specific clipping is applied to the
NPL or MSE curves. SD animals simply contribute no condition points at ILD
+/-16 because those condition fits do not exist.

The delay function is fit separately for each animal and ABL:

    delay_ms = bias_ms + abs_ild_coeff_ms_per_unit * |ILD| + ild2_coeff_ms_per_unit2 * ILD^2
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
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.optimize import least_squares


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

OUTPUT_DIR = SCRIPT_DIR / "abl_specific_ild2_delay_mse_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_PNG = OUTPUT_DIR / "cond_t_E_aff_vs_npl_alpha_abl_specific_ild2_and_mse_delay.png"

MODEL_KEY = "vbmc_norm_alpha_abl_specific_ild2_delay_tied_results"
ABLS = [20, 40, 60]
CONTINUOUS_ILDS = np.round(np.arange(-16.0, 16.0 + 0.05, 0.1), 10)
FITTED_ILDS = [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]
COLORS = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}

N_CONDITION_POSTERIOR_SAMPLES = int(float(os.environ.get("N_CONDITION_POSTERIOR_SAMPLES", "50000")))
AUTO_CREATE_COMPAT_ENV = os.environ.get("AUTO_CREATE_COMPAT_ENV", "1").lower() in {"1", "true", "yes", "y"}
COMPAT_ENV_DIR = Path(os.environ.get("COND_PICKLE_COMPAT_ENV", REPO_DIR / ".venv_cond_pickle_read"))
EXISTING_COND_CACHE_CSV = SCRIPT_DIR / "abl_specific_ild2_delay_agreement" / "condition_t_E_aff_extraction_cache.csv"
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


def delay_curve_ms(params, ild_values):
    bias_ms, abs_ild_coeff, ild2_coeff = np.asarray(params, dtype=float)
    ild_values = np.asarray(ild_values, dtype=float)
    return bias_ms + abs_ild_coeff * np.abs(ild_values) + ild2_coeff * (ild_values ** 2)


def format_signed_term(value, suffix):
    sign = "+" if value >= 0 else "-"
    return f" {sign} {abs(value):.3g}{suffix}"


def format_average_expression(label, coeff_df):
    parts = []
    avg_coeff_df = coeff_df.groupby("ABL", sort=True)[
        ["bias_ms", "abs_ild_coeff_ms_per_unit", "ild2_coeff_ms_per_unit2"]
    ].mean()
    for abl in ABLS:
        row = avg_coeff_df.loc[abl]
        expr = (
            f"A{abl}: {row['bias_ms']:.3g}"
            f"{format_signed_term(row['abs_ild_coeff_ms_per_unit'], '|I|')}"
            f"{format_signed_term(row['ild2_coeff_ms_per_unit2'], 'I^2')}"
        )
        parts.append(expr)
    return f"{label} avg d(I) ms = " + "; ".join(parts)


def summarize_values(rows_df, value_col, out_col_prefix):
    rows = []
    for (abl, ild), group in rows_df.groupby(["ABL", "ILD"], sort=True):
        values = group[value_col].to_numpy(dtype=float)
        n = int(np.sum(np.isfinite(values)))
        rows.append(
            {
                "ABL": int(abl),
                "ILD": float(ild),
                f"{out_col_prefix}_mean_ms": float(np.nanmean(values)) if n > 0 else np.nan,
                f"{out_col_prefix}_sem_ms": float(np.nanstd(values) / np.sqrt(n)) if n > 0 else np.nan,
                "n_animals": n,
            }
        )
    return pd.DataFrame(rows)


def load_upstream_delay_coefficients_and_curves():
    coeff_rows = []
    curve_rows = []

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
            params = np.array([bias[abl_idx], abs_coeff[abl_idx], ild2_coeff[abl_idx]], dtype=float)

            coeff_rows.append(
                {
                    "batch_name": batch_name,
                    "animal": animal_id,
                    "ABL": abl,
                    "bias_ms": params[0],
                    "abs_ild_coeff_ms_per_unit": params[1],
                    "ild2_coeff_ms_per_unit2": params[2],
                    "source_pkl": str(result_path),
                }
            )
            for ild, delay in zip(CONTINUOUS_ILDS, delay_curve_ms(params, CONTINUOUS_ILDS)):
                curve_rows.append(
                    {
                        "batch_name": batch_name,
                        "animal": animal_id,
                        "ABL": abl,
                        "ILD": float(ild),
                        "npl_delay_ms": float(delay),
                    }
                )

    return pd.DataFrame(coeff_rows), pd.DataFrame(curve_rows)


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


def extract_condition_means_with_compat_env():
    compat_python = ensure_compat_env()
    if compat_python is None:
        return None

    print(f"Extracting condition means with compatibility Python: {compat_python}")
    with tempfile.TemporaryDirectory(prefix="ddm_cond_delay_extract_") as temp_dir:
        temp_csv = Path(temp_dir) / "condition_means.csv"
        cmd = [
            str(compat_python),
            str(Path(__file__).resolve()),
            "--extract-condition-cache",
            str(temp_csv),
        ]
        subprocess.run(cmd, cwd=REPO_DIR, check=True)
        return pd.read_csv(temp_csv)


def load_or_extract_condition_means():
    if EXISTING_COND_CACHE_CSV.exists():
        df = pd.read_csv(EXISTING_COND_CACHE_CSV)
        if len(df) == EXPECTED_N_COND_FITS:
            print(f"Loaded existing condition means cache: {EXISTING_COND_CACHE_CSV}")
            return df
        print(f"Ignoring incomplete existing cache {EXISTING_COND_CACHE_CSV}: {len(df)} rows")

    rows, failures = try_extract_condition_means_with_current_python()
    if len(rows) == EXPECTED_N_COND_FITS:
        print("Extracted all condition means with current Python")
        return pd.DataFrame(rows).sort_values(["batch_name", "animal", "ABL", "ILD"])

    print(
        f"Current Python extracted {len(rows)}/{EXPECTED_N_COND_FITS} condition fits; "
        f"using compatibility extraction for the full set."
    )
    if failures:
        print("First extraction failures:")
        for pkl_path, error in failures[:5]:
            print(f"  {pkl_path.name}: {error[:180]}")

    compat_df = extract_condition_means_with_compat_env()
    if compat_df is not None and len(compat_df) == EXPECTED_N_COND_FITS:
        return compat_df
    raise RuntimeError("Could not extract all condition-fit posterior means.")


def fit_mse_delay_curves(cond_df, coeff_df):
    coeff_lookup = {
        (row.batch_name, int(row.animal), int(row.ABL)): np.array(
            [row.bias_ms, row.abs_ild_coeff_ms_per_unit, row.ild2_coeff_ms_per_unit2],
            dtype=float,
        )
        for row in coeff_df.itertuples(index=False)
    }

    fit_rows = []
    curve_rows = []
    for (batch_name, animal_id, abl), group in cond_df.groupby(["batch_name", "animal", "ABL"], sort=True):
        start_params = coeff_lookup[(batch_name, int(animal_id), int(abl))]
        x0 = start_params
        ild_values = group["ILD"].to_numpy(dtype=float)
        target_ms = group["t_E_aff_ms"].to_numpy(dtype=float)

        finite = np.isfinite(ild_values) & np.isfinite(target_ms)
        ild_values = ild_values[finite]
        target_ms = target_ms[finite]
        if len(target_ms) < 3:
            raise RuntimeError(f"Too few points for MSE delay fit: {batch_name}/{animal_id}, ABL={abl}")

        def residuals(params):
            return delay_curve_ms(params, ild_values) - target_ms

        fit_result = least_squares(residuals, x0)
        params = fit_result.x
        fitted_values = delay_curve_ms(params, ild_values)
        rmse = float(np.sqrt(np.mean((fitted_values - target_ms) ** 2)))
        fit_rows.append(
            {
                "batch_name": batch_name,
                "animal": int(animal_id),
                "ABL": int(abl),
                "n_condition_points": int(len(target_ms)),
                "bias_ms": float(params[0]),
                "abs_ild_coeff_ms_per_unit": float(params[1]),
                "ild2_coeff_ms_per_unit2": float(params[2]),
                "rmse_ms": rmse,
                "success": bool(fit_result.success),
                "message": fit_result.message,
            }
        )

        for ild, delay in zip(CONTINUOUS_ILDS, delay_curve_ms(params, CONTINUOUS_ILDS)):
            curve_rows.append(
                {
                    "batch_name": batch_name,
                    "animal": int(animal_id),
                    "ABL": int(abl),
                    "ILD": float(ild),
                    "mse_delay_ms": float(delay),
                }
            )

    return pd.DataFrame(fit_rows), pd.DataFrame(curve_rows)


def compute_summary_metrics(cond_summary_df, npl_summary_df, mse_summary_df):
    rows = []
    npl_lookup = {(int(row.ABL), round(float(row.ILD), 10)): row.npl_delay_mean_ms for row in npl_summary_df.itertuples()}
    mse_lookup = {(int(row.ABL), round(float(row.ILD), 10)): row.mse_delay_mean_ms for row in mse_summary_df.itertuples()}

    point_rows = []
    for row in cond_summary_df.itertuples(index=False):
        key = (int(row.ABL), round(float(row.ILD), 10))
        point_rows.append(
            {
                "ABL": int(row.ABL),
                "ILD": float(row.ILD),
                "condition": float(row.condition_delay_mean_ms),
                "npl": float(npl_lookup[key]),
                "mse": float(mse_lookup[key]),
            }
        )
    points = pd.DataFrame(point_rows)

    for abl in ABLS + ["all"]:
        subset = points if abl == "all" else points[points["ABL"] == abl]
        for left, right in [("condition", "npl"), ("condition", "mse"), ("npl", "mse")]:
            x = subset[left].to_numpy(dtype=float)
            y = subset[right].to_numpy(dtype=float)
            finite = np.isfinite(x) & np.isfinite(y)
            if np.sum(finite) >= 2:
                diff = x[finite] - y[finite]
                rows.append(
                    {
                        "ABL": abl,
                        "comparison": f"{left}_minus_{right}",
                        "n_points": int(np.sum(finite)),
                        "mean_diff_ms": float(np.mean(diff)),
                        "rmse_ms": float(np.sqrt(np.mean(diff**2))),
                        "pearson_r": float(np.corrcoef(x[finite], y[finite])[0, 1]),
                    }
                )
    return pd.DataFrame(rows)


def plot_delay_comparison(cond_summary_df, npl_summary_df, mse_summary_df, coeff_df, mse_fit_df):
    fig, ax = plt.subplots(1, 1, figsize=(12.5, 6.4))

    for abl in ABLS:
        color = COLORS[abl]
        npl = npl_summary_df[npl_summary_df["ABL"] == abl].sort_values("ILD")
        mse = mse_summary_df[mse_summary_df["ABL"] == abl].sort_values("ILD")
        cond = cond_summary_df[cond_summary_df["ABL"] == abl].sort_values("ILD")

        npl_x = npl["ILD"].to_numpy(dtype=float)
        npl_y = npl["npl_delay_mean_ms"].to_numpy(dtype=float)
        npl_sem = npl["npl_delay_sem_ms"].to_numpy(dtype=float)
        ax.fill_between(npl_x, npl_y - npl_sem, npl_y + npl_sem, color=color, alpha=0.10, linewidth=0)
        ax.plot(npl_x, npl_y, color=color, linewidth=2.3)

        mse_x = mse["ILD"].to_numpy(dtype=float)
        mse_y = mse["mse_delay_mean_ms"].to_numpy(dtype=float)
        mse_sem = mse["mse_delay_sem_ms"].to_numpy(dtype=float)
        ax.fill_between(mse_x, mse_y - mse_sem, mse_y + mse_sem, color=color, alpha=0.06, linewidth=0)
        ax.plot(mse_x, mse_y, color=color, linewidth=2.3, linestyle="--")

        ax.errorbar(
            cond["ILD"].to_numpy(dtype=float),
            cond["condition_delay_mean_ms"].to_numpy(dtype=float),
            yerr=cond["condition_delay_sem_ms"].to_numpy(dtype=float),
            marker="o",
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.4,
            linestyle="none",
            color=color,
            ecolor=color,
            capsize=2.5,
            markersize=5.0,
            zorder=3,
        )

    title_lines = [
        "Condition t_E_aff vs NPL delay and unconstrained per-animal MSE delay",
        format_average_expression("NPL+alpha+ILD2", coeff_df),
        format_average_expression("MSE", mse_fit_df),
    ]
    fig.suptitle("\n".join(title_lines), fontsize=9.5, y=0.99)
    ax.set_xlabel("ILD")
    ax.set_ylabel("delay / t_E_aff (ms)")
    ax.set_xlim(-16.5, 16.5)
    ax.set_xticks([-16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16])
    ax.grid(True, alpha=0.25)

    abl_handles = [Line2D([0], [0], color=COLORS[abl], linewidth=2.3, label=f"ABL={abl}") for abl in ABLS]
    method_handles = [
        Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor="black", linestyle="none", label="condition fit"),
        Line2D([0], [0], color="black", linewidth=2.3, label="NPL+alpha ABL-specific ILD2"),
        Line2D([0], [0], color="black", linewidth=2.3, linestyle="--", label="per-animal MSE delay fit"),
    ]
    fig.legend(
        handles=abl_handles + method_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.90),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.82])
    fig.savefig(FIG_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return FIG_PNG


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
print(f"Output figure: {FIG_PNG}")

coeff_df, npl_animal_curve_df = load_upstream_delay_coefficients_and_curves()
print(f"Loaded stable upstream fits for {coeff_df[['batch_name', 'animal']].drop_duplicates().shape[0]} animals")

cond_df = load_or_extract_condition_means()
if len(cond_df) != EXPECTED_N_COND_FITS:
    raise RuntimeError(f"Expected {EXPECTED_N_COND_FITS} condition rows, found {len(cond_df)}")
cond_df = cond_df.sort_values(["batch_name", "animal", "ABL", "ILD"]).reset_index(drop=True)
print(f"Loaded condition rows: {len(cond_df)}")

led7_check = cond_df[
    (cond_df["batch_name"] == "LED7")
    & (cond_df["animal"] == 92)
    & (cond_df["ABL"] == 20)
    & (cond_df["ILD"] == -1)
]
if len(led7_check) == 1:
    print(f"LED7/92 ABL=20 ILD=-1 t_E_aff_ms: {float(led7_check.iloc[0]['t_E_aff_ms']):.3f}")

mse_fit_df, mse_animal_curve_df = fit_mse_delay_curves(cond_df, coeff_df)
print(f"MSE delay fits completed: {len(mse_fit_df)} animal-ABL fits")
print(f"MSE fit success count: {int(mse_fit_df['success'].sum())}/{len(mse_fit_df)}")
print("MSE fits are unconstrained least-squares fits initialized from each animal's NPL delay coefficients.")
print("\nAverage delay expressions:")
print("  " + format_average_expression("NPL+alpha+ILD2", coeff_df))
print("  " + format_average_expression("MSE", mse_fit_df))
print("\nMSE coefficient ranges by ABL:")
for abl in ABLS:
    subset = mse_fit_df[mse_fit_df["ABL"] == abl]
    print(
        f"  ABL={abl}: "
        f"bias [{subset['bias_ms'].min():.3g}, {subset['bias_ms'].max():.3g}], "
        f"abs_coeff [{subset['abs_ild_coeff_ms_per_unit'].min():.3g}, {subset['abs_ild_coeff_ms_per_unit'].max():.3g}], "
        f"ild2_coeff [{subset['ild2_coeff_ms_per_unit2'].min():.3g}, {subset['ild2_coeff_ms_per_unit2'].max():.3g}], "
        f"rmse mean={subset['rmse_ms'].mean():.3g} ms"
    )

cond_summary_df = summarize_values(cond_df.rename(columns={"t_E_aff_ms": "condition_delay_ms"}), "condition_delay_ms", "condition_delay")
npl_summary_df = summarize_values(npl_animal_curve_df, "npl_delay_ms", "npl_delay")
mse_summary_df = summarize_values(mse_animal_curve_df, "mse_delay_ms", "mse_delay")

metrics_df = compute_summary_metrics(cond_summary_df, npl_summary_df, mse_summary_df)
print("\nSummary metrics at fitted ILDs:")
print(metrics_df.to_string(index=False))

png_path = plot_delay_comparison(cond_summary_df, npl_summary_df, mse_summary_df, coeff_df, mse_fit_df)
print(f"\nSaved figure: {png_path}")

# %%
