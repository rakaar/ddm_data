# %%
"""
Recreate the lapses supplementary figure from the current SVI fit families.

This mirrors the old 2 x 4 lapses supplementary figure, but replaces the old
VBMC/IPL/NPL/lapse sources with the current all-animal SVI fits:

- IPL condition-delay SVI
- NPL+alpha condition-delay SVI
- IPL+lapse condition-delay SVI
- NPL+alpha+lapse condition-delay SVI

The old Gamma panel required condition-by-condition lapse Gamma/Omega fits, so
that panel is intentionally left blank and annotated.
"""

# %%
# =============================================================================
# Editable parameters
# =============================================================================
from pathlib import Path
import os
import pickle
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

IPL_ROOT = SCRIPT_DIR / "numpyro_svi_vanilla_condition_delay_patience12_min50k_restore_best_outputs"
NPL_ROOT = SCRIPT_DIR / "numpyro_svi_npl_alpha_condition_delay_patience12_restore_best_outputs"
IPL_LAPSE_ROOT = SCRIPT_DIR / "numpyro_svi_vanilla_lapse_condition_delay_patience12_min50k_restore_best_outputs"
NPL_LAPSE_ROOT = SCRIPT_DIR / "numpyro_svi_npl_alpha_lapse_condition_delay_patience12_min50k_restore_best_outputs"

OUTPUT_DIR = NPL_LAPSE_ROOT / "summary_figures" / "svi_lapses_supp_figure"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_PNG = OUTPUT_DIR / "svi_lapses_supp_figure_2x4.png"
FIG_PDF = OUTPUT_DIR / "svi_lapses_supp_figure_2x4.pdf"
PLOT_DATA_PKL = OUTPUT_DIR / "svi_lapses_supp_figure_plot_data.pkl"
LAPSE_RATES_CSV = OUTPUT_DIR / "svi_lapses_supp_lapse_rates.csv"
LOGLIKE_CSV = OUTPUT_DIR / "svi_lapses_supp_loglike_by_animal.csv"
LL_DIFF_CSV = OUTPUT_DIR / "svi_lapses_supp_loglike_differences.csv"
PARAM_CSV = OUTPUT_DIR / "svi_lapses_supp_npl_param_comparison.csv"

EXPECTED_N_ANIMALS = 30
ABLS = [20.0, 40.0, 60.0]
DESIRED_BATCHES = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]
BATCH_T_TRUNC = {"LED34_even": 0.15}
DEFAULT_T_TRUNC = 0.30
K_MAX = 10

MODEL_IPL = "IPL"
MODEL_NPL = "NPL+alpha"
MODEL_IPL_LAPSE = "IPL+lapse"
MODEL_NPL_LAPSE = "NPL+alpha+lapse"

MODEL_SPECS = {
    MODEL_IPL: {
        "root": IPL_ROOT,
        "global_params": ["rate_lambda", "T_0", "theta_E", "w", "del_go"],
        "likelihood": "ipl",
    },
    MODEL_NPL: {
        "root": NPL_ROOT,
        "global_params": ["rate_lambda", "T_0", "theta_E", "w", "del_go", "rate_norm_l", "alpha"],
        "likelihood": "npl_alpha",
    },
    MODEL_IPL_LAPSE: {
        "root": IPL_LAPSE_ROOT,
        "global_params": ["rate_lambda", "T_0", "theta_E", "w", "del_go", "lapse_prob", "lapse_prob_right"],
        "likelihood": "ipl_lapse",
    },
    MODEL_NPL_LAPSE: {
        "root": NPL_LAPSE_ROOT,
        "global_params": [
            "rate_lambda",
            "T_0",
            "theta_E",
            "w",
            "del_go",
            "rate_norm_l",
            "alpha",
            "lapse_prob",
            "lapse_prob_right",
        ],
        "likelihood": "npl_alpha_lapse",
    },
}

MODEL_ORDER = [MODEL_IPL, MODEL_NPL, MODEL_IPL_LAPSE, MODEL_NPL_LAPSE]

NPL_PARAM_PANELS = [
    ("rate_norm_l", r"$\ell$", None),
    ("rate_lambda", r"$\lambda'$", None),
    ("theta_E", r"$\theta_E$", None),
    ("T_0", r"$T_0$ (s)", None),
]


# %%
# =============================================================================
# Imports that depend on local paths
# =============================================================================
sys.path.insert(0, str(SCRIPT_DIR))

import numpyro_vanilla_condition_delay_svi_utils as ipl_utils
import numpyro_vanilla_lapse_condition_delay_svi_utils as ipl_lapse_utils
import numpyro_npl_alpha_svi_utils as npl_alpha_utils
import numpyro_npl_alpha_lapse_svi_utils as npl_alpha_lapse_utils


# %%
# =============================================================================
# Helpers
# =============================================================================
def animal_key(batch_name, animal):
    return f"{batch_name}_{int(animal)}"


def animal_label(batch_name, animal):
    return f"{batch_name}/{int(animal)}"


def parse_animal_dir(path):
    batch_name, animal_text = path.name.rsplit("_", 1)
    return batch_name, int(animal_text)


def animal_sort_key(item):
    batch_name, animal = item
    batch_idx = DESIRED_BATCHES.index(batch_name) if batch_name in DESIRED_BATCHES else len(DESIRED_BATCHES)
    return batch_idx, int(animal)


def ensure_choice_column(df):
    if "choice" not in df.columns:
        if "response_poke" not in df.columns:
            raise KeyError("Need either `choice` or `response_poke` in the batch CSV.")
        df = df.copy()
        df["choice"] = df["response_poke"].map({3: 1, 2: -1})
    return df


def sem(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return np.nan
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def load_summary(root, batch_name, animal):
    path = root / animal_key(batch_name, animal) / "main_fullrank_posterior_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_condition_table(root, batch_name, animal):
    path = root / animal_key(batch_name, animal) / "condition_table.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    table = pd.read_csv(path)
    needed_cols = ["ABL", "ILD", "condition_id"]
    missing = [col for col in needed_cols if col not in table.columns]
    if missing:
        raise RuntimeError(f"{path} missing columns: {missing}")
    table = table[needed_cols].copy()
    table["ABL"] = table["ABL"].astype(float)
    table["ILD"] = table["ILD"].astype(float)
    table["condition_id"] = table["condition_id"].astype(int)
    if table.duplicated(["ABL", "ILD"]).any():
        raise RuntimeError(f"{path} has duplicate ABL/ILD rows")
    return table.sort_values("condition_id").reset_index(drop=True)


def summary_row(summary_df, parameter):
    rows = summary_df[summary_df["parameter"].astype(str) == parameter]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one posterior-summary row for {parameter!r}, found {len(rows)}")
    return rows.iloc[0]


def read_global_param(summary_df, parameter):
    row = summary_row(summary_df, parameter)
    return {
        "mean": float(row["mean"]),
        "q025": float(row["q025"]),
        "q975": float(row["q975"]),
    }


def delay_vector_from_summary(summary_df, condition_table):
    delay_rows = summary_df[
        summary_df["parameter"].astype(str).str.startswith("t_E_aff")
        & summary_df["ABL"].notna()
        & summary_df["ILD"].notna()
    ].copy()
    if delay_rows.empty:
        raise RuntimeError("No condition-wise t_E_aff rows found in posterior summary.")
    delay_rows["ABL"] = delay_rows["ABL"].astype(float)
    delay_rows["ILD"] = delay_rows["ILD"].astype(float)
    merged = condition_table.merge(
        delay_rows[["ABL", "ILD", "mean"]],
        on=["ABL", "ILD"],
        how="left",
        validate="one_to_one",
    )
    if merged["mean"].isna().any():
        missing = merged.loc[merged["mean"].isna(), ["ABL", "ILD", "condition_id"]]
        raise RuntimeError(f"Missing t_E_aff rows for conditions:\n{missing}")
    return merged.sort_values("condition_id")["mean"].to_numpy(dtype=float)


def load_abort_means(batch_name, animal):
    path = REPO_DIR / "aborts_ipl_npl_time_fit_results" / f"results_{batch_name}_animal_{int(animal)}.pkl"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        saved = pickle.load(handle)
    abort = saved["vbmc_aborts_results"]
    return {
        "V_A": float(np.mean(abort["V_A_samples"])),
        "theta_A": float(np.mean(abort["theta_A_samples"])),
        "t_A_aff": float(np.mean(abort["t_A_aff_samp"])),
    }


def load_valid_trials(batch_name, animal):
    path = REPO_DIR / "raw_data" / "batch_csvs" / f"batch_{batch_name}_valid_and_aborts.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    raw_df = ensure_choice_column(pd.read_csv(path))
    valid_df = raw_df[
        (raw_df["animal"].astype(int) == int(animal))
        & (raw_df["success"].isin([1, -1]))
        & (raw_df["RTwrtStim"] < 1)
        & (raw_df["ABL"].isin(ABLS))
    ].copy()
    valid_df = valid_df.dropna(subset=["TotalFixTime", "intended_fix", "ABL", "ILD", "choice", "RTwrtStim"])
    if valid_df.empty:
        raise RuntimeError(f"No valid RT<1 trials for {batch_name}/{animal}")
    valid_df["ABL"] = valid_df["ABL"].astype(float)
    valid_df["ILD"] = valid_df["ILD"].astype(float)
    valid_df["choice"] = valid_df["choice"].astype(int)
    return valid_df.reset_index(drop=True)


def make_jax_data(valid_df, condition_table, abort_params, T_trunc):
    model_df = valid_df.merge(condition_table, on=["ABL", "ILD"], how="left", validate="many_to_one")
    if model_df["condition_id"].isna().any():
        missing = model_df.loc[model_df["condition_id"].isna(), ["ABL", "ILD"]].drop_duplicates()
        raise RuntimeError(f"Could not assign condition_id for:\n{missing}")
    return {
        "total_fix": jnp.asarray(model_df["TotalFixTime"].to_numpy(dtype=float)),
        "t_stim": jnp.asarray(model_df["intended_fix"].to_numpy(dtype=float)),
        "ABL": jnp.asarray(model_df["ABL"].to_numpy(dtype=float)),
        "ILD": jnp.asarray(model_df["ILD"].to_numpy(dtype=float)),
        "choice": jnp.asarray(model_df["choice"].to_numpy(dtype=int)),
        "condition_id": jnp.asarray(model_df["condition_id"].to_numpy(dtype=int)),
        "V_A": jnp.asarray(abort_params["V_A"], dtype=jnp.float64),
        "theta_A": jnp.asarray(abort_params["theta_A"], dtype=jnp.float64),
        "t_A_aff": jnp.asarray(abort_params["t_A_aff"], dtype=jnp.float64),
        "T_trunc": jnp.asarray(T_trunc, dtype=jnp.float64),
        "lapse_rt_window": jnp.asarray(1.0, dtype=jnp.float64),
    }


def params_from_summary(summary_df, condition_table, global_params):
    params = {name: read_global_param(summary_df, name)["mean"] for name in global_params}
    params["t_E_aff"] = delay_vector_from_summary(summary_df, condition_table)
    return params


def common_loglike(model_label, params, data):
    kind = MODEL_SPECS[model_label]["likelihood"]
    if kind == "ipl":
        value = ipl_utils.vanilla_condition_delay_loglike(params, data, K_max=K_MAX)
    elif kind == "ipl_lapse":
        value = ipl_lapse_utils.vanilla_condition_delay_loglike(params, data, K_max=K_MAX)
    elif kind == "npl_alpha":
        value = npl_alpha_utils.npl_alpha_condition_delay_loglike(params, data, K_max=K_MAX)
    elif kind == "npl_alpha_lapse":
        value = npl_alpha_lapse_utils.npl_alpha_lapse_condition_delay_loglike(params, data, K_max=K_MAX)
    else:
        raise ValueError(f"Unknown likelihood kind {kind!r}")
    return float(jax.device_get(value))


def set_panel_style(ax):
    ax.tick_params(axis="both", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    try:
        ax.set_box_aspect(1)
    except Exception:
        pass


def finite_ylim_with_zero(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -1.0, 1.0
    ymin = min(float(np.min(values)), 0.0)
    ymax = max(float(np.max(values)), 0.0)
    if ymin == ymax:
        pad = max(abs(ymin) * 0.1, 1.0)
    else:
        pad = 0.08 * (ymax - ymin)
    return ymin - pad, ymax + pad


# %%
# =============================================================================
# Load and validate the four SVI fit roots
# =============================================================================
print("Using SVI roots:")
for model_label in MODEL_ORDER:
    root = MODEL_SPECS[model_label]["root"]
    print(f"  {model_label}: {root}")
    if not root.exists():
        raise FileNotFoundError(root)

animal_sets = {}
for model_label in MODEL_ORDER:
    root = MODEL_SPECS[model_label]["root"]
    paths = sorted(root.glob("*/main_fullrank_posterior_summary.csv"))
    animal_sets[model_label] = {parse_animal_dir(path.parent) for path in paths}
    print(f"{model_label}: {len(animal_sets[model_label])} posterior summaries")

reference_animals = animal_sets[MODEL_IPL]
for model_label, animals in animal_sets.items():
    if animals != reference_animals:
        missing = sorted(reference_animals - animals, key=animal_sort_key)
        extra = sorted(animals - reference_animals, key=animal_sort_key)
        raise RuntimeError(f"Animal set mismatch for {model_label}. Missing={missing}; extra={extra}")

animal_keys = sorted(reference_animals, key=animal_sort_key)
if len(animal_keys) != EXPECTED_N_ANIMALS:
    raise RuntimeError(f"Expected {EXPECTED_N_ANIMALS} matched animals, found {len(animal_keys)}")

print(f"Matched animals across all four SVI roots: {len(animal_keys)}")


# %%
# =============================================================================
# Collect posterior summaries, lapse rates, and common log likelihoods
# =============================================================================
param_rows = []
lapse_rows = []
loglike_rows = []

for batch_name, animal in animal_keys:
    label = animal_label(batch_name, animal)
    print(f"Processing {label}")
    valid_df = load_valid_trials(batch_name, animal)
    abort_params = load_abort_means(batch_name, animal)
    T_trunc = BATCH_T_TRUNC.get(batch_name, DEFAULT_T_TRUNC)

    animal_model_params = {}
    animal_summaries = {}

    for model_label in MODEL_ORDER:
        spec = MODEL_SPECS[model_label]
        summary_df = load_summary(spec["root"], batch_name, animal)
        condition_table = load_condition_table(spec["root"], batch_name, animal)
        params = params_from_summary(summary_df, condition_table, spec["global_params"])
        data = make_jax_data(valid_df, condition_table, abort_params, T_trunc)
        loglike = common_loglike(model_label, params, data)

        if not np.isfinite(loglike):
            raise RuntimeError(f"Non-finite log likelihood for {model_label} {label}: {loglike}")

        animal_model_params[model_label] = params
        animal_summaries[model_label] = summary_df
        loglike_rows.append(
            {
                "batch_name": batch_name,
                "animal": int(animal),
                "animal_label": label,
                "model": model_label,
                "loglike_at_posterior_mean": loglike,
                "n_trials": int(len(valid_df)),
                "T_trunc": float(T_trunc),
            }
        )

    ipl_lapse_prob = animal_model_params[MODEL_IPL_LAPSE]["lapse_prob"]
    npl_lapse_prob = animal_model_params[MODEL_NPL_LAPSE]["lapse_prob"]
    avg_lapse_prob = 0.5 * (ipl_lapse_prob + npl_lapse_prob)
    lapse_rows.append(
        {
            "batch_name": batch_name,
            "animal": int(animal),
            "animal_label": label,
            "ipl_lapse_prob": float(ipl_lapse_prob),
            "npl_alpha_lapse_prob": float(npl_lapse_prob),
            "avg_lapse_prob": float(avg_lapse_prob),
            "ipl_lapse_rate_pct": float(ipl_lapse_prob * 100.0),
            "npl_alpha_lapse_rate_pct": float(npl_lapse_prob * 100.0),
            "avg_lapse_rate_pct": float(avg_lapse_prob * 100.0),
        }
    )

    for model_label in [MODEL_NPL, MODEL_NPL_LAPSE]:
        summary_df = animal_summaries[model_label]
        for param_name, _label, _ticks in NPL_PARAM_PANELS:
            stats = read_global_param(summary_df, param_name)
            param_rows.append(
                {
                    "batch_name": batch_name,
                    "animal": int(animal),
                    "animal_label": label,
                    "model": model_label,
                    "parameter": param_name,
                    "mean": stats["mean"],
                    "q025": stats["q025"],
                    "q975": stats["q975"],
                }
            )

lapse_df = pd.DataFrame(lapse_rows)
loglike_df = pd.DataFrame(loglike_rows)
param_df = pd.DataFrame(param_rows)

if len(lapse_df) != EXPECTED_N_ANIMALS:
    raise RuntimeError(f"Expected {EXPECTED_N_ANIMALS} lapse rows, found {len(lapse_df)}")
if len(loglike_df) != EXPECTED_N_ANIMALS * len(MODEL_ORDER):
    raise RuntimeError(f"Expected {EXPECTED_N_ANIMALS * len(MODEL_ORDER)} loglike rows, found {len(loglike_df)}")
if loglike_df["loglike_at_posterior_mean"].isna().any():
    raise RuntimeError("Some log likelihood values are NaN")

loglike_wide = loglike_df.pivot_table(
    index=["batch_name", "animal", "animal_label"],
    columns="model",
    values="loglike_at_posterior_mean",
    aggfunc="first",
).reset_index()
ll_diff_df = loglike_wide.merge(lapse_df, on=["batch_name", "animal", "animal_label"], validate="one_to_one")
ll_diff_df["npl_minus_ipl_lapse"] = ll_diff_df[MODEL_NPL] - ll_diff_df[MODEL_IPL_LAPSE]
ll_diff_df["npl_lapse_minus_ipl_lapse"] = ll_diff_df[MODEL_NPL_LAPSE] - ll_diff_df[MODEL_IPL_LAPSE]

param_order_df = lapse_df.sort_values("npl_alpha_lapse_rate_pct").reset_index(drop=True)
param_order_df["x_pos"] = np.arange(len(param_order_df))
param_df = param_df.merge(
    param_order_df[["batch_name", "animal", "x_pos", "npl_alpha_lapse_rate_pct"]],
    on=["batch_name", "animal"],
    how="left",
    validate="many_to_one",
)
median_lapse_rate_pct = float(np.median(lapse_df["avg_lapse_rate_pct"]))
median_order_lapse_rate_pct = float(np.median(param_order_df["npl_alpha_lapse_rate_pct"]))
median_lapse_x = float(
    np.interp(
        median_order_lapse_rate_pct,
        param_order_df["npl_alpha_lapse_rate_pct"].to_numpy(dtype=float),
        param_order_df["x_pos"].to_numpy(dtype=float),
    )
)

lapse_df.to_csv(LAPSE_RATES_CSV, index=False)
loglike_df.to_csv(LOGLIKE_CSV, index=False)
ll_diff_df.to_csv(LL_DIFF_CSV, index=False)
param_df.to_csv(PARAM_CSV, index=False)

plot_data = {
    "model_roots": {model: str(MODEL_SPECS[model]["root"]) for model in MODEL_ORDER},
    "lapse_df": lapse_df,
    "loglike_df": loglike_df,
    "ll_diff_df": ll_diff_df,
    "param_df": param_df,
    "median_lapse_rate_pct": median_lapse_rate_pct,
    "median_lapse_x": median_lapse_x,
    "notes": {
        "gamma_panel": "Omitted because the old panel requires condition-by-condition lapse Gamma/Omega fits.",
        "likelihood": "Posterior-mean log likelihood on valid RT<1 fit trials, using each model's own likelihood.",
    },
}
with PLOT_DATA_PKL.open("wb") as handle:
    pickle.dump(plot_data, handle)

print(f"Saved lapse rates: {LAPSE_RATES_CSV}")
print(f"Saved log likelihoods: {LOGLIKE_CSV}")
print(f"Saved LL differences: {LL_DIFF_CSV}")
print(f"Saved parameter comparison: {PARAM_CSV}")
print(f"Saved plot data: {PLOT_DATA_PKL}")
print("\nLog-likelihood difference ranges:")
print(ll_diff_df[["npl_minus_ipl_lapse", "npl_lapse_minus_ipl_lapse"]].describe().to_string())


# %%
# =============================================================================
# Plot the SVI-based supplementary figure
# =============================================================================
fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)

# Panel (0,0): lapse-rate distribution
ax = axes[0, 0]
sorted_lapse = np.sort(lapse_df["avg_lapse_rate_pct"].to_numpy(dtype=float))
x_animals = np.arange(1, len(sorted_lapse) + 1)
ax.scatter(x_animals, sorted_lapse, color="black", s=45, alpha=0.75)
ax.axhline(median_lapse_rate_pct, color="gray", linestyle="--", linewidth=1.5, label=f"median={median_lapse_rate_pct:.2f}%")
ax.set_xlabel("Rat", fontsize=11)
ax.set_ylabel("Lapse rate (%)", fontsize=11)
ax.set_xticks([])
upper_lapse = max(5.0, float(np.nanmax(sorted_lapse)) * 1.15)
ax.set_ylim(0, upper_lapse)
ax.legend(frameon=False, fontsize=9, loc="best")
set_panel_style(ax)

# Panel (0,1): NPL - IPL+lapse LL diff
ax = axes[0, 1]
x = ll_diff_df["ipl_lapse_rate_pct"].to_numpy(dtype=float)
y = ll_diff_df["npl_minus_ipl_lapse"].to_numpy(dtype=float)
colors = np.where(y > 0, "green", "red")
ax.scatter(x, y, c=colors, alpha=0.75, s=45, edgecolors="black", linewidths=0.4)
ax.axhline(0, color="black", linestyle="--", linewidth=1)
ax.axvline(float(np.median(x)), color="black", linestyle=":", linewidth=1)
ax.set_xlabel("IPL+lapse rate (%)", fontsize=11)
ax.set_ylabel(r"$\Delta$LL (NPL+alpha $-$ IPL+lapse)", fontsize=11)
ax.set_ylim(*finite_ylim_with_zero(y))
set_panel_style(ax)

# Panel (0,2): NPL+lapse - IPL+lapse LL diff
ax = axes[0, 2]
y = ll_diff_df["npl_lapse_minus_ipl_lapse"].to_numpy(dtype=float)
colors = np.where(y > 0, "green", "red")
ax.scatter(x, y, c=colors, alpha=0.75, s=45, edgecolors="black", linewidths=0.4)
ax.axhline(0, color="black", linestyle="--", linewidth=1)
ax.axvline(float(np.median(x)), color="black", linestyle=":", linewidth=1)
ax.set_xlabel("IPL+lapse rate (%)", fontsize=11)
ax.set_ylabel(r"$\Delta$LL (NPL+alpha+lapse $-$ IPL+lapse)", fontsize=11)
ax.set_ylim(*finite_ylim_with_zero(y))
set_panel_style(ax)

# Panel (0,3): omitted condition-by-condition lapse Gamma panel
ax = axes[0, 3]
ax.axis("off")
ax.text(
    0.5,
    0.55,
    "Gamma panel omitted",
    ha="center",
    va="center",
    fontsize=13,
    fontweight="bold",
    transform=ax.transAxes,
)
ax.text(
    0.5,
    0.40,
    "Requires condition-by-condition\nlapse Gamma/Omega fits",
    ha="center",
    va="center",
    fontsize=10,
    transform=ax.transAxes,
)

# Bottom row: NPL+alpha vs NPL+alpha+lapse parameter comparison
model_style = {
    MODEL_NPL: {"color": "black", "marker": "o", "offset": -0.10, "label": "NPL+alpha"},
    MODEL_NPL_LAPSE: {"color": "red", "marker": "s", "offset": 0.10, "label": "NPL+alpha+lapse"},
}

for col_idx, (param_name, ylabel, yticks) in enumerate(NPL_PARAM_PANELS):
    ax = axes[1, col_idx]
    for model_label in [MODEL_NPL, MODEL_NPL_LAPSE]:
        style = model_style[model_label]
        sub = param_df[(param_df["parameter"] == param_name) & (param_df["model"] == model_label)].sort_values("x_pos")
        means = sub["mean"].to_numpy(dtype=float)
        q025 = sub["q025"].to_numpy(dtype=float)
        q975 = sub["q975"].to_numpy(dtype=float)
        x_pos = sub["x_pos"].to_numpy(dtype=float) + style["offset"]
        ax.errorbar(
            x_pos,
            means,
            yerr=[means - q025, q975 - means],
            fmt=style["marker"],
            color=style["color"],
            alpha=0.75,
            capsize=0,
            markersize=4.5,
            linewidth=1.0,
            label=style["label"],
        )
    ax.axvline(median_lapse_x, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Rat ordered by NPL+alpha+lapse lapse rate", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks([])
    if yticks is not None:
        ax.set_yticks(yticks)
    if col_idx == 0:
        ax.legend(frameon=False, fontsize=9, loc="best")
    set_panel_style(ax)

fig.suptitle("SVI recreation of lapses supplementary figure", fontsize=15)
fig.savefig(FIG_PNG, dpi=300, bbox_inches="tight")
fig.savefig(FIG_PDF, bbox_inches="tight")

print("\nFigure saved to:")
print(f"  {FIG_PNG}")
print(f"  {FIG_PDF}")
print("Done.")

# %%
