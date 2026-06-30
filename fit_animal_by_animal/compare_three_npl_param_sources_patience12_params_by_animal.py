# %%
"""
Compare animal-wise NPL+alpha parameters from three current parameter sources.

Markers:
- red x: per-animal MSE fit to big-SVI Gamma + Omega condition means
- blue open triangle: per-animal MSE fit to big-SVI Omega condition means
- green dot with 95% CI: direct patience-12 37-param NPL SVI posterior
"""

# %%
# =============================================================================
# Editable parameters
# =============================================================================
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

NPL_SVI_ROOT = (
    SCRIPT_DIR / "numpyro_svi_npl_alpha_condition_delay_patience12_restore_best_outputs"
)
COMPARISON_DIR = NPL_SVI_ROOT / "three_npl_param_source_comparison"
METHOD_PARAM_CSV = COMPARISON_DIR / "three_npl_param_sources_params_by_animal.csv"

OUT_LONG_CSV = COMPARISON_DIR / "three_npl_param_sources_params_by_animal_2x3_long.csv"
OUT_METRIC_CSV = COMPARISON_DIR / "three_npl_param_sources_params_by_animal_2x3_metrics.csv"
OUT_FIG_PNG = COMPARISON_DIR / "three_npl_param_sources_params_by_animal_2x3.png"

EXPECTED_N_ANIMALS = 30
DESIRED_BATCHES = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]
BATCH_ORDER = {batch: idx for idx, batch in enumerate(DESIRED_BATCHES)}

METHOD_SPECS = [
    {
        "key": "mse_gamma_omega",
        "label": "Gamma+Omega MSE",
        "color": "tab:red",
        "marker": "x",
        "x_offset": -0.2,
    },
    {
        "key": "mse_omega_only",
        "label": "Omega-only MSE",
        "color": "tab:blue",
        "marker": "^",
        "x_offset": 0.0,
    },
    {
        "key": "direct_37_param_svi",
        "label": "Direct NPL patience-12 SVI",
        "color": "tab:green",
        "marker": "o",
        "x_offset": 0.2,
    },
]

PARAM_SPECS = [
    {"key": "rate_lambda", "title": "rate_lambda", "ylabel": "rate_lambda", "scale": 1.0},
    {"key": "T_0", "title": "T_0", "ylabel": "T_0 (ms)", "scale": 1000.0},
    {"key": "theta_E", "title": "theta_E", "ylabel": "theta_E", "scale": 1.0},
    {"key": "alpha", "title": "alpha", "ylabel": "alpha", "scale": 1.0},
    {"key": "rate_norm_l", "title": "rate_norm_l", "ylabel": "rate_norm_l", "scale": 1.0},
]


# %%
# =============================================================================
# Load and validate inputs
# =============================================================================
def sort_pair(batch_name, animal):
    return (BATCH_ORDER.get(str(batch_name), 999), str(batch_name), int(animal))


if not METHOD_PARAM_CSV.exists():
    raise FileNotFoundError(METHOD_PARAM_CSV)

method_df = pd.read_csv(METHOD_PARAM_CSV)
method_df["batch_name"] = method_df["batch_name"].astype(str)
method_df["animal"] = method_df["animal"].astype(int)

required_cols = ["method_key", "method_label", "batch_name", "animal"] + [
    spec["key"] for spec in PARAM_SPECS
]
missing_cols = [col for col in required_cols if col not in method_df.columns]
if missing_cols:
    raise KeyError(f"{METHOD_PARAM_CSV} missing columns: {missing_cols}")

expected_method_keys = [spec["key"] for spec in METHOD_SPECS]
unexpected_methods = sorted(set(method_df["method_key"]) - set(expected_method_keys))
missing_methods = sorted(set(expected_method_keys) - set(method_df["method_key"]))
if unexpected_methods or missing_methods:
    raise RuntimeError(
        f"Unexpected methods={unexpected_methods}; missing methods={missing_methods}"
    )

duplicate_rows = int(method_df.duplicated(["method_key", "batch_name", "animal"]).sum())
if duplicate_rows:
    raise RuntimeError(f"Found {duplicate_rows} duplicate method/animal rows in {METHOD_PARAM_CSV}")

animals_by_method = {
    method_key: {
        (row.batch_name, int(row.animal))
        for row in group[["batch_name", "animal"]].itertuples(index=False)
    }
    for method_key, group in method_df.groupby("method_key")
}
for method_key, animal_set in animals_by_method.items():
    if len(animal_set) != EXPECTED_N_ANIMALS:
        raise RuntimeError(
            f"Expected {EXPECTED_N_ANIMALS} animals for {method_key}, found {len(animal_set)}"
        )

animal_set_values = list(animals_by_method.values())
if any(curr != animal_set_values[0] for curr in animal_set_values[1:]):
    raise RuntimeError("The three methods do not contain the same animal set.")

animal_pairs = sorted(animal_set_values[0], key=lambda pair: sort_pair(pair[0], pair[1]))
animal_labels = [f"{batch}/{animal}" for batch, animal in animal_pairs]
animal_index = {pair: idx for idx, pair in enumerate(animal_pairs)}

long_rows = []
for method_spec in METHOD_SPECS:
    method_key = method_spec["key"]
    curr_df = method_df[method_df["method_key"] == method_key].copy()
    for row in curr_df.itertuples(index=False):
        pair = (str(row.batch_name), int(row.animal))
        for param_spec in PARAM_SPECS:
            value = float(getattr(row, param_spec["key"])) * param_spec["scale"]
            if not np.isfinite(value):
                raise RuntimeError(f"Non-finite {param_spec['key']} for {method_key} {pair}")
            long_rows.append(
                {
                    "method_key": method_key,
                    "method_label": method_spec["label"],
                    "batch_name": pair[0],
                    "animal": pair[1],
                    "animal_label": f"{pair[0]}/{pair[1]}",
                    "animal_order": animal_index[pair],
                    "parameter": param_spec["key"],
                    "parameter_label": param_spec["ylabel"],
                    "value": value,
                    "q025": np.nan,
                    "q975": np.nan,
                }
            )

long_df = pd.DataFrame(long_rows)

for batch_name, animal in animal_pairs:
    summary_path = NPL_SVI_ROOT / f"{batch_name}_{animal}" / "main_fullrank_posterior_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary_df = pd.read_csv(summary_path)
    for param_spec in PARAM_SPECS:
        param_rows = summary_df[summary_df["parameter"] == param_spec["key"]]
        if len(param_rows) != 1:
            raise RuntimeError(f"{summary_path} has {len(param_rows)} rows for {param_spec['key']}")
        param_row = param_rows.iloc[0]
        q025 = float(param_row["q025"]) * param_spec["scale"]
        q975 = float(param_row["q975"]) * param_spec["scale"]
        if not np.isfinite(q025) or not np.isfinite(q975):
            raise RuntimeError(f"Non-finite CI for {param_spec['key']} in {summary_path}")
        mask = (
            (long_df["method_key"] == "direct_37_param_svi")
            & (long_df["batch_name"] == batch_name)
            & (long_df["animal"] == animal)
            & (long_df["parameter"] == param_spec["key"])
        )
        if int(mask.sum()) != 1:
            raise RuntimeError(f"Could not attach CI for {batch_name}/{animal} {param_spec['key']}")
        long_df.loc[mask, "q025"] = q025
        long_df.loc[mask, "q975"] = q975

long_df = long_df.sort_values(["parameter", "animal_order", "method_key"]).reset_index(drop=True)
long_df.to_csv(OUT_LONG_CSV, index=False)
print(f"Saved plotted values: {OUT_LONG_CSV}")


# %%
# =============================================================================
# Metrics against the direct 37-param SVI means
# =============================================================================
metric_rows = []
direct_df = long_df[long_df["method_key"] == "direct_37_param_svi"].copy()
for method_spec in METHOD_SPECS:
    if method_spec["key"] == "direct_37_param_svi":
        continue
    curr_df = long_df[long_df["method_key"] == method_spec["key"]].copy()
    for param_spec in PARAM_SPECS:
        direct_values = (
            direct_df[direct_df["parameter"] == param_spec["key"]]
            .sort_values("animal_order")["value"]
            .to_numpy(dtype=float)
        )
        curr_values = (
            curr_df[curr_df["parameter"] == param_spec["key"]]
            .sort_values("animal_order")["value"]
            .to_numpy(dtype=float)
        )
        if direct_values.shape != curr_values.shape:
            raise RuntimeError(f"Shape mismatch for {method_spec['key']} {param_spec['key']}")
        diff = curr_values - direct_values
        finite = np.isfinite(diff) & np.isfinite(curr_values) & np.isfinite(direct_values)
        pearson_r = (
            float(np.corrcoef(curr_values[finite], direct_values[finite])[0, 1])
            if np.sum(finite) >= 2
            and np.nanstd(curr_values[finite]) > 0
            and np.nanstd(direct_values[finite]) > 0
            else np.nan
        )
        metric_rows.append(
            {
                "method_key": method_spec["key"],
                "method_label": method_spec["label"],
                "parameter": param_spec["key"],
                "parameter_label": param_spec["ylabel"],
                "n_animals": int(np.sum(finite)),
                "mean_method_minus_direct": float(np.mean(diff[finite])),
                "rmse_method_vs_direct": float(np.sqrt(np.mean(diff[finite] ** 2))),
                "pearson_r_method_vs_direct": pearson_r,
            }
        )

metric_df = pd.DataFrame(metric_rows)
metric_df.to_csv(OUT_METRIC_CSV, index=False)
print(f"Saved metrics: {OUT_METRIC_CSV}")


# %%
# =============================================================================
# Plot
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(17.5, 8.8), sharex=True)
axes_flat = axes.ravel()
x_base = np.arange(len(animal_pairs), dtype=float)

for ax, param_spec in zip(axes_flat, PARAM_SPECS):
    ax.set_title(param_spec["title"])
    ax.set_ylabel(param_spec["ylabel"])

    for method_spec in METHOD_SPECS:
        curr = long_df[
            (long_df["method_key"] == method_spec["key"])
            & (long_df["parameter"] == param_spec["key"])
        ].sort_values("animal_order")

        x_vals = x_base + method_spec["x_offset"]
        y_vals = curr["value"].to_numpy(dtype=float)

        if method_spec["key"] == "direct_37_param_svi":
            q025 = curr["q025"].to_numpy(dtype=float)
            q975 = curr["q975"].to_numpy(dtype=float)
            yerr = np.vstack([y_vals - q025, q975 - y_vals])
            ax.errorbar(
                x_vals,
                y_vals,
                yerr=yerr,
                fmt=method_spec["marker"],
                color=method_spec["color"],
                ecolor=method_spec["color"],
                elinewidth=1.0,
                capsize=2.0,
                markersize=4.4,
                linestyle="none",
                alpha=0.95,
            )
        elif method_spec["key"] == "mse_omega_only":
            ax.scatter(
                x_vals,
                y_vals,
                marker=method_spec["marker"],
                facecolors="none",
                edgecolors=method_spec["color"],
                linewidths=1.2,
                s=42,
                alpha=0.95,
            )
        else:
            ax.scatter(
                x_vals,
                y_vals,
                marker=method_spec["marker"],
                color=method_spec["color"],
                s=38,
                linewidths=1.3,
                alpha=0.95,
            )

    ax.grid(alpha=0.22)
    ax.set_xlim(-0.7, len(animal_pairs) - 0.3)

axes_flat[len(PARAM_SPECS)].axis("off")

legend_handles = [
    Line2D(
        [0],
        [0],
        color="tab:red",
        marker="x",
        linestyle="none",
        markersize=6,
        label="Gamma+Omega MSE",
    ),
    Line2D(
        [0],
        [0],
        color="tab:blue",
        marker="^",
        linestyle="none",
        markerfacecolor="none",
        markeredgewidth=1.2,
        markersize=6,
        label="Omega-only MSE",
    ),
    Line2D(
        [0],
        [0],
        color="tab:green",
        marker="o",
        linestyle="none",
        markersize=6,
        label="Direct NPL patience-12 SVI mean +/- 95% CI",
    ),
]
axes_flat[len(PARAM_SPECS)].legend(
    handles=legend_handles,
    loc="center",
    frameon=False,
    fontsize=10,
)

for ax in axes[-1, :]:
    ax.set_xticks(x_base)
    ax.set_xticklabels(animal_labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Animal")

fig.suptitle(
    "Three current NPL+alpha parameter sources by animal",
    y=0.985,
    fontsize=14,
)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT_FIG_PNG, dpi=200, bbox_inches="tight")
print(f"Saved figure: {OUT_FIG_PNG}")
