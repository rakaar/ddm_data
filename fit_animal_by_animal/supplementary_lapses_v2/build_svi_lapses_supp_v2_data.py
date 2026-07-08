# %%
"""
Build data products for the SVI-based lapses supplementary figure v2.

The figure mirrors the old 2 x 4 lapses supplementary figure, but reads the
current all-animal SVI outputs instead of the older VBMC pickles. NPL is shown
with the paper-facing label in plots, but the source fit is the current
NPL+alpha condition-delay SVI family.
"""

# %%
# =============================================================================
# Editable parameters
# =============================================================================
from pathlib import Path
import os
import pickle
import re
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ANIMAL_DIR = SCRIPT_DIR.parent
REPO_DIR = ANIMAL_DIR.parent
COND_DIR = REPO_DIR / "fit_each_condn"

IPL_ROOT = ANIMAL_DIR / "numpyro_svi_vanilla_condition_delay_patience12_min50k_restore_best_outputs"
NPL_ROOT = ANIMAL_DIR / "numpyro_svi_npl_alpha_condition_delay_patience12_restore_best_outputs"
IPL_LAPSE_ROOT = ANIMAL_DIR / "numpyro_svi_vanilla_lapse_condition_delay_patience12_min50k_restore_best_outputs"
NPL_LAPSE_ROOT = ANIMAL_DIR / "numpyro_svi_npl_alpha_lapse_condition_delay_patience12_min50k_restore_best_outputs"

BIG_NO_LAPSE_ROOT = COND_DIR / "svi_big_gamma_omega_delay_patience12_restore_best_all_animals_outputs"
BIG_LAPSE_ROOT = COND_DIR / "svi_big_gamma_omega_delay_lapse_patience12_restore_best_all_animals_outputs"

OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PLOT_DATA_PKL = OUTPUT_DIR / "svi_lapses_supp_v2_plot_data.pkl"
LAPSE_RATES_CSV = OUTPUT_DIR / "svi_lapses_supp_v2_lapse_rates.csv"
LOGLIKE_CSV = OUTPUT_DIR / "svi_lapses_supp_v2_loglike_by_animal.csv"
LL_DIFF_CSV = OUTPUT_DIR / "svi_lapses_supp_v2_loglike_differences.csv"
PARAM_CSV = OUTPUT_DIR / "svi_lapses_supp_v2_npl_param_comparison.csv"
GAMMA_CSV = OUTPUT_DIR / "svi_lapses_supp_v2_big_condition_gamma_summary.csv"

EXPECTED_N_ANIMALS = 30
EXPECTED_N_CONDITION_ROWS_PER_MODEL = 864
ABLS = [20.0, 40.0, 60.0]
ILDS = np.array([-16, -8, -4, -2, -1, 1, 2, 4, 8, 16], dtype=int)
DESIRED_BATCHES = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]
BATCH_T_TRUNC = {"LED34_even": 0.15}
DEFAULT_T_TRUNC = 0.30
K_MAX = 10

MODEL_IPL = "IPL"
MODEL_NPL = "NPL"
MODEL_IPL_LAPSE = "IPL_L"
MODEL_NPL_LAPSE = "NPL_L"

MODEL_SPECS = {
    MODEL_IPL: {
        "root": IPL_ROOT,
        "global_params": ["rate_lambda", "T_0", "theta_E", "w", "del_go"],
        "likelihood": "ipl",
        "caption_name": "IPL/vanilla condition-delay SVI",
    },
    MODEL_NPL: {
        "root": NPL_ROOT,
        "global_params": ["rate_lambda", "T_0", "theta_E", "w", "del_go", "rate_norm_l", "alpha"],
        "likelihood": "npl_alpha",
        "caption_name": "NPL+alpha condition-delay SVI",
    },
    MODEL_IPL_LAPSE: {
        "root": IPL_LAPSE_ROOT,
        "global_params": ["rate_lambda", "T_0", "theta_E", "w", "del_go", "lapse_prob", "lapse_prob_right"],
        "likelihood": "ipl_lapse",
        "caption_name": "IPL/vanilla+lapse condition-delay SVI",
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
        "caption_name": "NPL+alpha+lapse condition-delay SVI",
    },
}
MODEL_ORDER = [MODEL_IPL, MODEL_NPL, MODEL_IPL_LAPSE, MODEL_NPL_LAPSE]

NPL_PARAM_PANELS = [
    ("rate_norm_l", r"$\ell$"),
    ("rate_lambda", r"$\lambda'$"),
    ("theta_E", r"$\theta_E$"),
    ("T_0", r"$T_0$ (s)"),
]

BIG_GAMMA_SPECS = {
    "No lapse": {
        "root": BIG_NO_LAPSE_ROOT,
        "file_suffix": "_big_gamma_omega_delay_condition_summary.csv",
    },
    "Lapse": {
        "root": BIG_LAPSE_ROOT,
        "file_suffix": "_big_gamma_omega_delay_lapse_condition_summary.csv",
    },
}


# %%
# =============================================================================
# Imports that depend on local paths
# =============================================================================
sys.path.insert(0, str(ANIMAL_DIR))

import numpyro_npl_alpha_lapse_svi_utils as npl_alpha_lapse_utils
import numpyro_npl_alpha_svi_utils as npl_alpha_utils
import numpyro_vanilla_condition_delay_svi_utils as ipl_utils
import numpyro_vanilla_lapse_condition_delay_svi_utils as ipl_lapse_utils


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


def sem(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return np.nan
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def ensure_choice_column(df):
    if "choice" not in df.columns:
        if "response_poke" not in df.columns:
            raise KeyError("Need either `choice` or `response_poke` in the batch CSV.")
        df = df.copy()
        df["choice"] = df["response_poke"].map({3: 1, 2: -1})
    return df


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


def validate_condition_counts(df, model_name):
    n_animals = df[["batch_name", "animal"]].drop_duplicates().shape[0]
    if n_animals != EXPECTED_N_ANIMALS:
        raise RuntimeError(f"{model_name}: expected {EXPECTED_N_ANIMALS} animals, found {n_animals}")
    if len(df) != EXPECTED_N_CONDITION_ROWS_PER_MODEL:
        raise RuntimeError(
            f"{model_name}: expected {EXPECTED_N_CONDITION_ROWS_PER_MODEL} condition rows, found {len(df)}"
        )
    duplicate_count = int(df.duplicated(["batch_name", "animal", "ABL", "ILD"]).sum())
    if duplicate_count:
        raise RuntimeError(f"{model_name}: found {duplicate_count} duplicate animal/condition rows")

    counts = df.groupby(["ABL", "ILD"])[["batch_name", "animal"]].apply(
        lambda values: values.drop_duplicates().shape[0]
    )
    for (abl, ild), n_animals_for_condition in counts.items():
        expected_n = 24 if abs(int(ild)) == 16 else 30
        if int(n_animals_for_condition) != expected_n:
            raise RuntimeError(
                f"{model_name}: ABL={abl}, ILD={ild} has n={n_animals_for_condition}, expected {expected_n}"
            )


def load_big_gamma_condition_df():
    all_dfs = []
    for model_name, spec in BIG_GAMMA_SPECS.items():
        root = spec["root"]
        if not root.exists():
            raise FileNotFoundError(root)
        summary_paths = sorted(root.glob(f"*/*{spec['file_suffix']}"))
        if len(summary_paths) != EXPECTED_N_ANIMALS:
            raise RuntimeError(f"{model_name}: expected {EXPECTED_N_ANIMALS} summaries, found {len(summary_paths)}")

        model_dfs = []
        for summary_path in summary_paths:
            match = re.match(r"^(?P<batch>.+)_(?P<animal>\d+)$", summary_path.parent.name)
            if match is None:
                raise RuntimeError(f"{model_name}: could not parse animal folder name {summary_path.parent}")
            df = pd.read_csv(summary_path)
            required = ["batch_name", "animal", "ABL", "ILD", "gamma_mean"]
            missing = [col for col in required if col not in df.columns]
            if missing:
                raise KeyError(f"{summary_path} missing columns: {missing}")
            df = df[required].copy()
            df["model"] = model_name
            df["batch_name"] = df["batch_name"].astype(str)
            df["animal"] = df["animal"].astype(int)
            df["ABL"] = df["ABL"].astype(int)
            df["ILD"] = df["ILD"].astype(int)
            model_dfs.append(df)

        model_df = pd.concat(model_dfs, ignore_index=True)
        validate_condition_counts(model_df, model_name)
        all_dfs.append(model_df)
        print(f"{model_name} big Gamma condition summaries: {len(model_df)} rows")

    condition_df = pd.concat(all_dfs, ignore_index=True)
    gamma_by_animal_ablavg = (
        condition_df.groupby(["model", "batch_name", "animal", "ILD"], as_index=False)["gamma_mean"].mean()
    )

    summary_rows = []
    for (model_name, ild), group in gamma_by_animal_ablavg.groupby(["model", "ILD"], sort=True):
        values = group["gamma_mean"].to_numpy(dtype=float)
        n = group[["batch_name", "animal"]].drop_duplicates().shape[0]
        expected_n = 24 if abs(int(ild)) == 16 else 30
        if n != expected_n:
            raise RuntimeError(f"{model_name}: ILD={ild} has n={n}, expected {expected_n}")
        summary_rows.append(
            {
                "model": model_name,
                "ILD": int(ild),
                "n_animals": int(n),
                "mean": float(np.nanmean(values)),
                "sd": float(np.nanstd(values, ddof=1)) if n > 1 else np.nan,
                "sem": sem(values),
            }
        )

    gamma_summary_df = pd.DataFrame(summary_rows).sort_values(["model", "ILD"]).reset_index(drop=True)
    return condition_df, gamma_by_animal_ablavg, gamma_summary_df


# %%
# =============================================================================
# Validate animal-wise fit roots
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

print(f"Matched animals across all four animal-wise SVI roots: {len(animal_keys)}")


# %%
# =============================================================================
# Collect posterior summaries, average lapse rates, and common log likelihoods
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
            "npl_lapse_prob": float(npl_lapse_prob),
            "avg_lapse_prob": float(avg_lapse_prob),
            "ipl_lapse_rate_pct": float(ipl_lapse_prob * 100.0),
            "npl_lapse_rate_pct": float(npl_lapse_prob * 100.0),
            "avg_lapse_rate_pct": float(avg_lapse_prob * 100.0),
        }
    )

    for model_label in [MODEL_NPL, MODEL_NPL_LAPSE]:
        summary_df = animal_summaries[model_label]
        for param_name, _label in NPL_PARAM_PANELS:
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

param_order_df = lapse_df.sort_values("avg_lapse_rate_pct").reset_index(drop=True)
param_order_df["x_pos"] = np.arange(len(param_order_df))
param_df = param_df.merge(
    param_order_df[["batch_name", "animal", "x_pos", "avg_lapse_rate_pct"]],
    on=["batch_name", "animal"],
    how="left",
    validate="many_to_one",
)
median_lapse_rate_pct = float(np.median(lapse_df["avg_lapse_rate_pct"]))
median_lapse_x = float(
    np.interp(
        median_lapse_rate_pct,
        param_order_df["avg_lapse_rate_pct"].to_numpy(dtype=float),
        param_order_df["x_pos"].to_numpy(dtype=float),
    )
)


# %%
# =============================================================================
# Add the condition-wise Gamma panel data from big SVI roots
# =============================================================================
print("Loading big condition-wise Gamma summaries:")
for label, spec in BIG_GAMMA_SPECS.items():
    print(f"  {label}: {spec['root']}")

gamma_condition_df, gamma_by_animal_ablavg_df, gamma_summary_df = load_big_gamma_condition_df()

condition_animals = set(map(tuple, gamma_condition_df[["batch_name", "animal"]].drop_duplicates().to_numpy()))
if condition_animals != set(animal_keys):
    missing = sorted(set(animal_keys) - condition_animals, key=animal_sort_key)
    extra = sorted(condition_animals - set(animal_keys), key=animal_sort_key)
    raise RuntimeError(f"Animal set mismatch between animal-wise and big condition roots. Missing={missing}; extra={extra}")


# %%
# =============================================================================
# Save compact data products
# =============================================================================
lapse_df.to_csv(LAPSE_RATES_CSV, index=False)
loglike_df.to_csv(LOGLIKE_CSV, index=False)
ll_diff_df.to_csv(LL_DIFF_CSV, index=False)
param_df.to_csv(PARAM_CSV, index=False)
gamma_summary_df.to_csv(GAMMA_CSV, index=False)

plot_data = {
    "model_roots": {model: str(MODEL_SPECS[model]["root"]) for model in MODEL_ORDER},
    "model_caption_names": {model: MODEL_SPECS[model]["caption_name"] for model in MODEL_ORDER},
    "big_gamma_roots": {model: str(spec["root"]) for model, spec in BIG_GAMMA_SPECS.items()},
    "lapse_df": lapse_df,
    "loglike_df": loglike_df,
    "ll_diff_df": ll_diff_df,
    "param_df": param_df,
    "param_order_df": param_order_df,
    "gamma_summary_df": gamma_summary_df,
    "gamma_by_animal_ablavg_df": gamma_by_animal_ablavg_df,
    "median_lapse_rate_pct": median_lapse_rate_pct,
    "median_lapse_x": median_lapse_x,
    "n_animals": EXPECTED_N_ANIMALS,
    "notes": {
        "npl_label": "Paper label NPL denotes the current NPL+alpha SVI fit.",
        "lapse_rate": "Average of IPL_L and NPL_L posterior-mean lapse_prob values.",
        "likelihood": "Posterior-mean log likelihood on valid RT<1 fit trials, using each model's own likelihood.",
        "gamma_panel": "Condition-wise Gamma posterior means from big Gamma/Omega/delay SVI with and without lapses; ABLs averaged within animal before SEM across animals.",
    },
}
with PLOT_DATA_PKL.open("wb") as handle:
    pickle.dump(plot_data, handle)

print("\nSaved data products:")
for path in [PLOT_DATA_PKL, LAPSE_RATES_CSV, LOGLIKE_CSV, LL_DIFF_CSV, PARAM_CSV, GAMMA_CSV]:
    print(f"  {path}")

print("\nLog-likelihood difference ranges:")
print(ll_diff_df[["npl_minus_ipl_lapse", "npl_lapse_minus_ipl_lapse"]].describe().to_string())
print("\nDone.")

# %%
