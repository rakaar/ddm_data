# %%
"""
Compare old NPL+alpha condition-delay SVI parameters with MSE alpha-model fits.

Blue points: old animal-wise NPL+alpha SVI posterior means with 95% intervals.
Red points: per-animal MSE alpha-model fits to the big SVI condition Gamma/Omega.
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


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

ANIMAL_SVI_ROOT = (
    REPO_DIR
    / "fit_animal_by_animal"
    / "numpyro_svi_npl_alpha_condition_delay_single_animal_outputs"
)
COMPARISON_DIR = (
    SCRIPT_DIR
    / "svi_big_gamma_omega_delay_all_animals_outputs"
    / "mse_alpha_model_comparison"
)
MSE_PARAM_CSV = COMPARISON_DIR / "per_animal_mse_gamma_omega_alpha_params.csv"

OUT_PARAM_CSV = COMPARISON_DIR / "npl_svi_vs_big_svi_mse_alpha_params_by_animal.csv"
OUT_METRIC_CSV = COMPARISON_DIR / "npl_svi_vs_big_svi_mse_alpha_param_metrics.csv"
OUT_FIG_PNG = COMPARISON_DIR / "npl_svi_vs_big_svi_mse_alpha_params_by_animal.png"

EXPECTED_N_ANIMALS = 30
DESIRED_BATCHES = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]

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
        "mse_col": "theta_E",
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
    {
        "key": "rate_norm_l",
        "title": "rate_norm_l",
        "ylabel": "rate_norm_l",
        "npl_sample_key": "rate_norm_l",
        "mse_col": "rate_norm_l",
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


def batch_sort_key(batch_name, animal_id):
    batch_idx = DESIRED_BATCHES.index(batch_name) if batch_name in DESIRED_BATCHES else len(DESIRED_BATCHES)
    return batch_idx, batch_name, int(animal_id)


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

    success_mask = mse_df["success"].astype(str).str.lower().isin(["true", "1"])
    failed = mse_df[~success_mask]
    if len(failed):
        raise RuntimeError(f"MSE fits failed for:\n{failed[['batch_name', 'animal', 'message']].to_string(index=False)}")

    duplicate_count = int(mse_df.duplicated(["batch_name", "animal"]).sum())
    if duplicate_count:
        raise RuntimeError(f"Found {duplicate_count} duplicate MSE animal rows.")

    required_cols = ["batch_name", "animal", "rate_lambda", "T_0", "theta_E", "alpha", "rate_norm_l"]
    missing_cols = [col for col in required_cols if col not in mse_df.columns]
    if missing_cols:
        raise KeyError(f"{MSE_PARAM_CSV} missing columns: {missing_cols}")

    mse_df = mse_df.sort_values(
        by=["batch_name", "animal"],
        key=lambda col: col.map(
            {batch: idx for idx, batch in enumerate(DESIRED_BATCHES)}
        )
        if col.name == "batch_name"
        else col,
    ).reset_index(drop=True)
    return mse_df


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
        missing = sorted(expected_animals - set(folder_by_animal), key=lambda pair: batch_sort_key(*pair))
        raise RuntimeError(f"Missing NPL SVI folders for: {missing}")

    rows = []
    mse_rows = sorted(
        list(mse_df.itertuples(index=False)),
        key=lambda row: batch_sort_key(row.batch_name, int(row.animal)),
    )
    for row in mse_rows:
        batch_name = row.batch_name
        animal_id = int(row.animal)
        posterior_npz = folder_by_animal[(batch_name, animal_id)] / "main_fullrank_posterior_samples.npz"
        if not posterior_npz.exists():
            raise FileNotFoundError(posterior_npz)

        with np.load(posterior_npz) as samples:
            out_row = {
                "batch_name": batch_name,
                "animal": animal_id,
                "animal_label": f"{batch_name}/{animal_id}",
                "source_npz": str(posterior_npz),
                "source_mse_csv": str(MSE_PARAM_CSV),
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
        corr = float(np.corrcoef(npl[finite], mse[finite])[0, 1]) if np.sum(finite) >= 2 else np.nan
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

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.4), sharex=True)
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
            label="MSE fit to big SVI Gamma/Omega",
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
            label="NPL+alpha + condition-delay SVI mean +/- 95% CI",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="tab:red",
            linestyle="none",
            label="MSE alpha model fit to big-SVI Gamma/Omega",
        ),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("NPL+alpha condition-delay SVI vs MSE alpha-model parameters from big-SVI Gamma/Omega", y=1.07, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(OUT_FIG_PNG, dpi=300, bbox_inches="tight")
    return OUT_FIG_PNG


# %%
# =============================================================================
# Main analysis
# =============================================================================
print(f"Animal-wise NPL SVI root: {ANIMAL_SVI_ROOT}")
print(f"Big-SVI MSE parameter CSV: {MSE_PARAM_CSV}")
print(f"Output folder: {COMPARISON_DIR}")

mse_df = load_mse_params()
param_df = load_npl_svi_for_animals(mse_df)
metric_df = compute_param_metrics(param_df)

if len(param_df) != EXPECTED_N_ANIMALS:
    raise RuntimeError(f"Expected {EXPECTED_N_ANIMALS} parameter rows, found {len(param_df)}")
finite_cols = [
    col
    for col in param_df.columns
    if col.endswith("_npl_mean") or col.endswith("_npl_q025") or col.endswith("_npl_q975") or col.endswith("_mse")
]
if not np.all(np.isfinite(param_df[finite_cols].to_numpy(dtype=float))):
    raise RuntimeError("Parameter comparison table contains non-finite values.")

param_df.to_csv(OUT_PARAM_CSV, index=False)
metric_df.to_csv(OUT_METRIC_CSV, index=False)
print(f"Saved parameter comparison table: {OUT_PARAM_CSV}")
print(f"Saved parameter metrics: {OUT_METRIC_CSV}")
print(metric_df.to_string(index=False))

fig_path = plot_param_comparison(param_df, metric_df)
print(f"Saved figure: {fig_path}")

plt.show()

# %%
