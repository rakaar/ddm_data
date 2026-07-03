# %%
"""
Compare vanilla/IPL condition-delay SVI parameters with and without lapses.

The scalar panels show posterior means for each animal. The delay panel shows
the across-animal mean of posterior-mean t_E_aff values with SEM bars.
"""

# %%
# =============================================================================
# Editable parameters
# =============================================================================
from pathlib import Path
import os

SCRIPT_DIR = Path(__file__).resolve().parent

IPL_OUTPUT_ROOT = (
    SCRIPT_DIR / "numpyro_svi_vanilla_condition_delay_patience12_min50k_restore_best_outputs"
)
IPL_LAPSE_OUTPUT_ROOT = (
    SCRIPT_DIR / "numpyro_svi_vanilla_lapse_condition_delay_patience12_min50k_restore_best_outputs"
)

OUTPUT_DIR = IPL_LAPSE_OUTPUT_ROOT / "comparison_with_ipl_no_lapse"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_N_ANIMALS = 30
EXPECTED_N_CONDITION_ROWS = 864
DESIRED_BATCHES = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]
ABLS = [20, 40, 60]
ILDS = [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]

FIG_PATH = OUTPUT_DIR / "ipl_vs_ipl_lapse_condition_delay_params.png"
SCALAR_VALUES_CSV = OUTPUT_DIR / "ipl_vs_ipl_lapse_scalar_values.csv"
CONDITION_VALUES_CSV = OUTPUT_DIR / "ipl_vs_ipl_lapse_teaff_animal_values.csv"
TEAFF_SUMMARY_CSV = OUTPUT_DIR / "ipl_vs_ipl_lapse_teaff_summary.csv"

MODEL_IPL = "IPL SVI 50k"
MODEL_LAPSE = "IPL + lapse SVI 50k"
MODEL_ORDER = [MODEL_IPL, MODEL_LAPSE]
MODEL_COLORS = {
    MODEL_IPL: "#0072B2",
    MODEL_LAPSE: "#D55E00",
}
MODEL_MARKERS = {
    MODEL_IPL: "o",
    MODEL_LAPSE: "x",
}
MODEL_OFFSETS = {
    MODEL_IPL: -0.11,
    MODEL_LAPSE: 0.11,
}
MODEL_LINESTYLES = {
    MODEL_IPL: "-",
    MODEL_LAPSE: "--",
}
ABL_COLORS = {
    20: "tab:blue",
    40: "tab:orange",
    60: "tab:green",
}


# %%
# =============================================================================
# Imports and helpers
# =============================================================================
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def parse_animal_dir_name(path):
    batch_name, animal_text = path.name.rsplit("_", 1)
    return batch_name, int(animal_text)


def animal_sort_key(item):
    batch_name, animal = item
    batch_idx = DESIRED_BATCHES.index(batch_name) if batch_name in DESIRED_BATCHES else len(DESIRED_BATCHES)
    return batch_idx, int(animal)


def sem(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return np.nan
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def load_condition_delay_svi_root(output_root, model_label):
    summary_paths = sorted(output_root.glob("*/main_fullrank_posterior_summary.csv"))
    if len(summary_paths) != EXPECTED_N_ANIMALS:
        raise RuntimeError(
            f"Expected {EXPECTED_N_ANIMALS} {model_label} posterior summaries, "
            f"found {len(summary_paths)} in {output_root}"
        )

    scalar_rows = []
    delay_rows = []

    for summary_path in summary_paths:
        batch_name, animal = parse_animal_dir_name(summary_path.parent)
        summary_df = pd.read_csv(summary_path)

        for param_name in ["rate_lambda", "T_0", "theta_E", "w", "del_go"]:
            param_rows = summary_df[summary_df["parameter"] == param_name]
            if len(param_rows) != 1:
                raise RuntimeError(f"{summary_path} has {len(param_rows)} rows for {param_name}")
            row = param_rows.iloc[0]
            scale = 1000.0 if param_name in {"T_0", "del_go"} else 1.0
            scalar_rows.append(
                {
                    "model": model_label,
                    "batch_name": batch_name,
                    "animal": animal,
                    "animal_label": f"{batch_name}/{animal}",
                    "parameter": param_name,
                    "mean": float(row["mean"]) * scale,
                    "q025": float(row["q025"]) * scale,
                    "q975": float(row["q975"]) * scale,
                    "unit": "ms" if param_name in {"T_0", "del_go"} else "unitless",
                }
            )

        delay_df = summary_df[
            summary_df["parameter"].astype(str).str.startswith("t_E_aff")
            & summary_df["ABL"].notna()
            & summary_df["ILD"].notna()
        ].copy()
        if delay_df.empty:
            raise RuntimeError(f"{summary_path} has no condition t_E_aff rows")

        for row in delay_df.itertuples(index=False):
            delay_rows.append(
                {
                    "model": model_label,
                    "batch_name": batch_name,
                    "animal": animal,
                    "animal_label": f"{batch_name}/{animal}",
                    "ABL": int(row.ABL),
                    "ILD": int(row.ILD),
                    "t_E_aff_mean_ms": float(row.mean) * 1000.0,
                    "t_E_aff_q025_ms": float(row.q025) * 1000.0,
                    "t_E_aff_q975_ms": float(row.q975) * 1000.0,
                }
            )

    return scalar_rows, delay_rows


# %%
# =============================================================================
# Load and validate both model sources
# =============================================================================
ipl_scalar_rows, ipl_delay_rows = load_condition_delay_svi_root(IPL_OUTPUT_ROOT, MODEL_IPL)
lapse_scalar_rows, lapse_delay_rows = load_condition_delay_svi_root(IPL_LAPSE_OUTPUT_ROOT, MODEL_LAPSE)

scalar_df = pd.DataFrame(ipl_scalar_rows + lapse_scalar_rows)
delay_df = pd.DataFrame(ipl_delay_rows + lapse_delay_rows)

animal_sets = {}
for model in MODEL_ORDER:
    animal_sets[model] = set(
        scalar_df.loc[scalar_df["model"] == model, ["batch_name", "animal"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

reference_animals = animal_sets[MODEL_IPL]
for model, model_animals in animal_sets.items():
    if model_animals != reference_animals:
        missing_from_model = sorted(reference_animals - model_animals, key=animal_sort_key)
        extra_in_model = sorted(model_animals - reference_animals, key=animal_sort_key)
        raise RuntimeError(
            f"Animal set mismatch for {model}. Missing: {missing_from_model}; extra: {extra_in_model}"
        )

animal_keys = sorted(reference_animals, key=animal_sort_key)
animal_labels = [f"{batch}/{animal}" for batch, animal in animal_keys]

if len(animal_keys) != EXPECTED_N_ANIMALS:
    raise RuntimeError(f"Expected {EXPECTED_N_ANIMALS} matched animals, found {len(animal_keys)}")

expected_scalar_rows = EXPECTED_N_ANIMALS * len(MODEL_ORDER) * 5
if len(scalar_df) != expected_scalar_rows:
    raise RuntimeError(f"Expected {expected_scalar_rows} scalar rows, found {len(scalar_df)}")
if scalar_df.duplicated(["model", "batch_name", "animal", "parameter"]).any():
    raise RuntimeError("Scalar values contain duplicate model/animal/parameter rows")

expected_delay_rows = EXPECTED_N_CONDITION_ROWS * len(MODEL_ORDER)
if len(delay_df) != expected_delay_rows:
    raise RuntimeError(f"Expected {expected_delay_rows} delay rows, found {len(delay_df)}")
if delay_df.duplicated(["model", "batch_name", "animal", "ABL", "ILD"]).any():
    raise RuntimeError("Delay values contain duplicate model/animal/condition rows")

if not np.all(np.isfinite(scalar_df[["mean", "q025", "q975"]].to_numpy(dtype=float))):
    raise RuntimeError("Scalar values contain NaN/Inf")
if not np.all(
    np.isfinite(delay_df[["t_E_aff_mean_ms", "t_E_aff_q025_ms", "t_E_aff_q975_ms"]].to_numpy(dtype=float))
):
    raise RuntimeError("Delay values contain NaN/Inf")

model_order_map = {model: idx for idx, model in enumerate(MODEL_ORDER)}
scalar_df["model_order"] = scalar_df["model"].map(model_order_map)
delay_df["model_order"] = delay_df["model"].map(model_order_map)
scalar_df = scalar_df.sort_values(["model_order", "batch_name", "animal", "parameter"]).reset_index(drop=True)
delay_df = delay_df.sort_values(["model_order", "batch_name", "animal", "ABL", "ILD"]).reset_index(drop=True)
scalar_df.drop(columns=["model_order"]).to_csv(SCALAR_VALUES_CSV, index=False)
delay_df.drop(columns=["model_order"]).to_csv(CONDITION_VALUES_CSV, index=False)

summary_rows = []
for (model, abl, ild), group in delay_df.groupby(["model", "ABL", "ILD"], sort=True):
    values = group["t_E_aff_mean_ms"].to_numpy(dtype=float)
    summary_rows.append(
        {
            "model": model,
            "model_order": model_order_map[model],
            "ABL": int(abl),
            "ILD": int(ild),
            "n_animals": int(values.size),
            "t_E_aff_mean_ms": float(np.mean(values)),
            "t_E_aff_sd_ms": float(np.std(values, ddof=1)) if values.size > 1 else np.nan,
            "t_E_aff_sem_ms": sem(values),
        }
    )

teaff_summary_df = (
    pd.DataFrame(summary_rows)
    .sort_values(["model_order", "ABL", "ILD"])
    .reset_index(drop=True)
)

for row in teaff_summary_df.itertuples(index=False):
    expected_n = 24 if abs(int(row.ILD)) == 16 else 30
    if int(row.n_animals) != expected_n:
        raise RuntimeError(
            f"{row.model}, ABL={row.ABL}, ILD={row.ILD} has n={row.n_animals}; expected {expected_n}"
        )

teaff_summary_df.drop(columns=["model_order"]).to_csv(TEAFF_SUMMARY_CSV, index=False)

print(f"Matched animals: {len(animal_keys)}")
print(f"Scalar rows: {len(scalar_df)}")
print(f"Delay rows: {len(delay_df)}")
print("Delay animal counts by model/ILD:")
print(teaff_summary_df.pivot_table(index="ILD", columns="model", values="n_animals", aggfunc="first"))
print(f"Saved scalar values: {SCALAR_VALUES_CSV}")
print(f"Saved delay animal values: {CONDITION_VALUES_CSV}")
print(f"Saved t_E_aff summary: {TEAFF_SUMMARY_CSV}")


# %%
# =============================================================================
# Plot comparison
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(22, 10), constrained_layout=True)
axes = axes.ravel()

x = np.arange(len(animal_labels))
scalar_panels = [
    ("rate_lambda", "rate_lambda"),
    ("T_0", "T_0 (ms)"),
    ("theta_E", "theta_E"),
    ("w", "w"),
    ("del_go", "del_go (ms)"),
]

for ax, (param_name, ylabel) in zip(axes[:5], scalar_panels):
    for model in MODEL_ORDER:
        model_df = (
            scalar_df[(scalar_df["model"] == model) & (scalar_df["parameter"] == param_name)]
            .set_index(["batch_name", "animal"])
            .loc[animal_keys]
            .reset_index()
        )
        ax.scatter(
            x + MODEL_OFFSETS[model],
            model_df["mean"],
            s=36,
            marker=MODEL_MARKERS[model],
            color=MODEL_COLORS[model],
            label=model,
            linewidths=1.5,
            alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(animal_labels, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Animal")
    ax.set_title(ylabel)
    ax.grid(axis="y", alpha=0.25)

axes[0].legend(frameon=False, loc="best", fontsize=8)

delay_ax = axes[5]
for abl in ABLS:
    for model in MODEL_ORDER:
        sub = teaff_summary_df[
            (teaff_summary_df["model"] == model) & (teaff_summary_df["ABL"] == abl)
        ].sort_values("ILD")
        delay_ax.errorbar(
            sub["ILD"] + MODEL_OFFSETS[model],
            sub["t_E_aff_mean_ms"],
            yerr=sub["t_E_aff_sem_ms"],
            fmt=MODEL_MARKERS[model],
            linestyle=MODEL_LINESTYLES[model],
            lw=1.0,
            color=ABL_COLORS[abl],
            ecolor=ABL_COLORS[abl],
            capsize=2.0,
            markersize=5.0,
            markeredgewidth=1.5,
            alpha=0.9,
        )

delay_ax.axvline(0, color="0.85", linewidth=0.9, zorder=0)
delay_ax.set_xticks(ILDS)
delay_ax.set_xticklabels([f"{ild:+d}" for ild in ILDS], rotation=45, ha="right", fontsize=8)
delay_ax.set_xlabel("ILD")
delay_ax.set_ylabel("t_E_aff (ms)")
delay_ax.set_title("t_E_aff across animals")
delay_ax.grid(axis="y", alpha=0.25)

abl_handles = [
    Line2D([0], [0], color=ABL_COLORS[abl], marker="o", lw=1.0, label=f"ABL {abl}")
    for abl in ABLS
]
model_handles = [
    Line2D(
        [0],
        [0],
        color="0.2",
        marker=MODEL_MARKERS[model],
        linestyle=MODEL_LINESTYLES[model],
        lw=1.0,
        label=model,
        markersize=6,
        markeredgewidth=1.5,
    )
    for model in MODEL_ORDER
]
delay_ax.legend(handles=abl_handles + model_handles, frameon=False, loc="best", fontsize=8)

fig.suptitle(
    "Patience-12 min-50k IPL versus IPL+lapse condition-delay SVI parameters",
    fontsize=14,
)
fig.savefig(FIG_PATH, dpi=200, bbox_inches="tight")
print(f"Saved figure: {FIG_PATH}")

plt.show()


# %%
