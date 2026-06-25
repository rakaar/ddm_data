# %%
"""
Compare animal-wise NPL+alpha SVI parameters with per-animal MSE alpha-model fits.

The MSE fits are the least-squares Gamma/Omega alpha-model fits from:

    compare_svi_cond_gamma_omega_with_npl_alpha_svi.py

This diagnostic asks which NPL+alpha parameters differ when the MSE model better
matches the condition-by-condition SVI Gamma/Omega curves.
"""

# %%
# =============================================================================
# Editable parameters
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

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

ANIMAL_SVI_ROOT = (
    REPO_DIR
    / "fit_animal_by_animal"
    / "numpyro_svi_npl_alpha_condition_delay_single_animal_outputs"
)
COMPARISON_DIR = SCRIPT_DIR / "svi_condition_gamma_omega_vs_npl_alpha_svi_comparison"
MSE_PARAM_CSV = COMPARISON_DIR / "per_animal_mse_gamma_omega_alpha_params.csv"

OUT_PARAM_CSV = COMPARISON_DIR / "npl_svi_vs_mse_alpha_params_by_animal.csv"
OUT_METRIC_CSV = COMPARISON_DIR / "npl_svi_vs_mse_alpha_param_metrics.csv"
OUT_FIG_PNG = COMPARISON_DIR / "npl_svi_vs_mse_alpha_params_by_animal.png"
OUT_FIG_PDF = COMPARISON_DIR / "npl_svi_vs_mse_alpha_params_by_animal.pdf"

EXPECTED_N_ANIMALS = 30

PARAM_SPECS = [
    {
        "key": "rate_lambda",
        "title": "rate_lambda",
        "ylabel": "rate_lambda",
        "npl_sample_key": "rate_lambda",
        "mse_col": "rate_lambda",
        "scale": 1.0,
    },
    {
        "key": "T_0_ms",
        "title": "T_0",
        "ylabel": "T_0 (ms)",
        "npl_sample_key": "T_0",
        "mse_col": "T_0",
        "scale": 1000.0,
    },
    {
        "key": "theta_E",
        "title": "theta_E",
        "ylabel": "theta_E",
        "npl_sample_key": "theta_E",
        "mse_col": "theta",
        "scale": 1.0,
    },
    {
        "key": "rate_norm_l",
        "title": "rate_norm_l",
        "ylabel": "rate_norm_l",
        "npl_sample_key": "rate_norm_l",
        "mse_col": "ell",
        "scale": 1.0,
    },
    {
        "key": "alpha",
        "title": "alpha",
        "ylabel": "alpha",
        "npl_sample_key": "alpha",
        "mse_col": "alpha",
        "scale": 1.0,
    },
]


# %%
# =============================================================================
# Helpers
# =============================================================================
def parse_animal_folder(path):
    match = re.match(r"^(?P<batch>.+)_(?P<animal>\d+)$", path.name)
    if match is None:
        return None
    return match.group("batch"), int(match.group("animal"))


def summarize_samples(values, scale):
    values = np.asarray(values, dtype=float) * scale
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return {"mean": np.nan, "q025": np.nan, "q975": np.nan}
    return {
        "mean": float(np.mean(finite_values)),
        "q025": float(np.quantile(finite_values, 0.025)),
        "q975": float(np.quantile(finite_values, 0.975)),
    }


def load_mse_params():
    if not MSE_PARAM_CSV.exists():
        raise FileNotFoundError(MSE_PARAM_CSV)
    mse_df = pd.read_csv(MSE_PARAM_CSV)
    mse_df["animal"] = mse_df["animal"].astype(int)
    if len(mse_df) != EXPECTED_N_ANIMALS:
        raise RuntimeError(f"Expected {EXPECTED_N_ANIMALS} MSE rows, found {len(mse_df)}")
    if "success" not in mse_df.columns:
        raise KeyError(f"{MSE_PARAM_CSV} is missing `success`.")
    failed = mse_df[mse_df["success"] != True]
    if len(failed):
        raise RuntimeError(f"MSE fits failed for:\n{failed[['batch_name', 'animal', 'message']].to_string(index=False)}")
    duplicate_count = int(mse_df.duplicated(["batch_name", "animal"]).sum())
    if duplicate_count:
        raise RuntimeError(f"Found {duplicate_count} duplicate MSE animal rows.")
    return mse_df.reset_index(drop=True)


def load_npl_svi_for_animals(mse_df):
    expected_animals = {
        (row.batch_name, int(row.animal))
        for row in mse_df[["batch_name", "animal"]].itertuples(index=False)
    }
    folder_by_animal = {}
    for animal_dir in sorted(ANIMAL_SVI_ROOT.iterdir()):
        if not animal_dir.is_dir():
            continue
        parsed = parse_animal_folder(animal_dir)
        if parsed is not None and parsed in expected_animals:
            folder_by_animal[parsed] = animal_dir

    if set(folder_by_animal) != expected_animals:
        missing = sorted(expected_animals - set(folder_by_animal))
        raise RuntimeError(f"Missing NPL SVI folders for: {missing}")

    rows = []
    for row in mse_df.itertuples(index=False):
        batch_name = row.batch_name
        animal_id = int(row.animal)
        posterior_npz = folder_by_animal[(batch_name, animal_id)] / "main_fullrank_posterior_samples.npz"
        if not posterior_npz.exists():
            raise FileNotFoundError(posterior_npz)
        samples = np.load(posterior_npz)
        out_row = {
            "batch_name": batch_name,
            "animal": animal_id,
            "animal_label": f"{batch_name}/{animal_id}",
            "source_npz": str(posterior_npz),
        }
        for spec in PARAM_SPECS:
            if spec["npl_sample_key"] not in samples.files:
                raise KeyError(f"{posterior_npz} missing {spec['npl_sample_key']!r}")
            npl_summary = summarize_samples(samples[spec["npl_sample_key"]], spec["scale"])
            mse_value = float(getattr(row, spec["mse_col"])) * spec["scale"]
            out_row[f"{spec['key']}_npl_mean"] = npl_summary["mean"]
            out_row[f"{spec['key']}_npl_q025"] = npl_summary["q025"]
            out_row[f"{spec['key']}_npl_q975"] = npl_summary["q975"]
            out_row[f"{spec['key']}_mse"] = mse_value
            out_row[f"{spec['key']}_mse_minus_npl_mean"] = mse_value - npl_summary["mean"]
        rows.append(out_row)
    return pd.DataFrame(rows)


def compute_param_metrics(param_df):
    rows = []
    for spec in PARAM_SPECS:
        npl = param_df[f"{spec['key']}_npl_mean"].to_numpy(dtype=float)
        mse = param_df[f"{spec['key']}_mse"].to_numpy(dtype=float)
        finite = np.isfinite(npl) & np.isfinite(mse)
        diff = mse[finite] - npl[finite]
        if np.sum(finite) >= 2:
            corr = float(np.corrcoef(npl[finite], mse[finite])[0, 1])
        else:
            corr = np.nan
        rows.append(
            {
                "parameter": spec["key"],
                "label": spec["ylabel"],
                "n_animals": int(np.sum(finite)),
                "npl_mean_across_animals": float(np.mean(npl[finite])),
                "mse_mean_across_animals": float(np.mean(mse[finite])),
                "mean_mse_minus_npl": float(np.mean(diff)),
                "rmse_mse_vs_npl": float(np.sqrt(np.mean(diff**2))),
                "pearson_r": corr,
            }
        )
    return pd.DataFrame(rows)


def plot_param_comparison(param_df, metrics_df):
    x = np.arange(len(param_df))
    animal_labels = param_df["animal_label"].tolist()

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), sharex=True)
    axes_flat = axes.ravel()
    for ax, spec in zip(axes_flat, PARAM_SPECS):
        npl_mean = param_df[f"{spec['key']}_npl_mean"].to_numpy(dtype=float)
        npl_q025 = param_df[f"{spec['key']}_npl_q025"].to_numpy(dtype=float)
        npl_q975 = param_df[f"{spec['key']}_npl_q975"].to_numpy(dtype=float)
        mse_value = param_df[f"{spec['key']}_mse"].to_numpy(dtype=float)
        yerr = np.vstack([npl_mean - npl_q025, npl_q975 - npl_mean])

        ax.errorbar(
            x - 0.08,
            npl_mean,
            yerr=yerr,
            fmt="o",
            color="tab:blue",
            ecolor="tab:blue",
            elinewidth=1.0,
            capsize=2.0,
            markersize=4.2,
            label="NPL+alpha SVI mean +/- 95% CI",
        )
        ax.scatter(
            x + 0.08,
            mse_value,
            color="tab:red",
            s=23,
            alpha=0.9,
            label="MSE alpha model",
            zorder=3,
        )

        metric = metrics_df[metrics_df["parameter"] == spec["key"]].iloc[0]
        ax.set_title(
            f"{spec['title']}\nRMSE={metric['rmse_mse_vs_npl']:.3g}, r={metric['pearson_r']:.2f}",
            fontsize=10.5,
        )
        ax.set_ylabel(spec["ylabel"])
        ax.grid(True, axis="y", alpha=0.25)

    axes_flat[-1].axis("off")
    for ax in axes_flat[:-1]:
        ax.set_xticks(x)
        ax.set_xticklabels(animal_labels, rotation=45, ha="right", fontsize=8)
        ax.tick_params(axis="x", labelbottom=True)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="tab:blue",
            linestyle="none",
            label="NPL+alpha SVI mean +/- 95% CI",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="tab:red",
            linestyle="none",
            label="MSE alpha model",
        ),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("NPL+alpha SVI vs per-animal MSE alpha-model parameters", y=1.07, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(OUT_FIG_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG_PDF, bbox_inches="tight")
    return OUT_FIG_PNG, OUT_FIG_PDF


# %%
# =============================================================================
# Main analysis
# =============================================================================
print(f"Animal-wise NPL SVI root: {ANIMAL_SVI_ROOT}")
print(f"MSE parameter CSV: {MSE_PARAM_CSV}")
print(f"Output folder: {COMPARISON_DIR}")

mse_df = load_mse_params()
param_df = load_npl_svi_for_animals(mse_df)
metric_df = compute_param_metrics(param_df)

if len(param_df) != EXPECTED_N_ANIMALS:
    raise RuntimeError(f"Expected {EXPECTED_N_ANIMALS} parameter rows, found {len(param_df)}")
if not np.all(np.isfinite(param_df.filter(regex="(_npl_mean|_npl_q025|_npl_q975|_mse)$").to_numpy(dtype=float))):
    raise RuntimeError("Parameter comparison table contains non-finite values.")

param_df.to_csv(OUT_PARAM_CSV, index=False)
metric_df.to_csv(OUT_METRIC_CSV, index=False)
print(f"Saved parameter comparison table: {OUT_PARAM_CSV}")
print(f"Saved parameter metrics: {OUT_METRIC_CSV}")
print(metric_df.to_string(index=False))

png_path, pdf_path = plot_param_comparison(param_df, metric_df)
print(f"Saved figure: {png_path}")
print(f"Saved figure: {pdf_path}")

# %%
