# %%
"""
Compare suspicious NPL+alpha+lapse SVI reruns against refreshed VBMC NPL+alpha+lapse.

This makes two result-book figures:
1. posterior parameter means with 95% intervals;
2. ELBO/objective and a common posterior-mean log likelihood.

VBMC is the refreshed scalar-delay NPL+alpha+lapse fit, so alpha is read from
the VBMC posterior summary instead of being fixed to 1.
"""

# %%
# =============================================================================
# Parameters
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
from matplotlib.lines import Line2D

import numpy as np
import pandas as pd

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

EARLY_SVI_ROOT = SCRIPT_DIR / "numpyro_svi_npl_alpha_lapse_condition_delay_low_lr_patience12_min12k_restore_best_reruns"
RANDOM_100K_SVI_ROOT = SCRIPT_DIR / "numpyro_svi_npl_alpha_lapse_condition_delay_random_plausible_low_lr_100k_earlybest_reruns"
VBMC_ROOT = SCRIPT_DIR / "vbmc_npl_alpha_lapse_scalar_delay_rerun"

OUTPUT_DIR = VBMC_ROOT / "comparison_to_svi"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PARAM_FIG = OUTPUT_DIR / "npl_alpha_lapse_svi_vbmc_alpha_scalar_parameter_means_ci.png"
OBJECTIVE_FIG = OUTPUT_DIR / "npl_alpha_lapse_svi_vbmc_alpha_scalar_elbo_likelihood.png"
PARAM_CSV = OUTPUT_DIR / "npl_alpha_lapse_svi_vbmc_alpha_scalar_parameter_means_ci.csv"
OBJECTIVE_CSV = OUTPUT_DIR / "npl_alpha_lapse_svi_vbmc_alpha_scalar_elbo_likelihood.csv"
VALIDATION_CSV = OUTPUT_DIR / "npl_alpha_lapse_svi_vbmc_alpha_scalar_likelihood_validation.csv"

ANIMALS = [
    ("LED7", 98),
    ("LED7", 100),
    ("LED34_even", 52),
    ("LED34_even", 60),
]

ABLS = [20.0, 40.0, 60.0]
BATCH_T_TRUNC = {"LED34_even": 0.15}
DEFAULT_T_TRUNC = 0.3
K_MAX = 10
VALIDATION_MAX_ROWS = 300
RNG_SEED = 20260705

MODEL_SPECS = [
    {
        "key": "early_svi",
        "label": "early SVI",
        "root": EARLY_SVI_ROOT,
        "kind": "svi",
        "color": "tab:blue",
        "marker": "o",
        "offset": -0.20,
    },
    {
        "key": "random100k_svi",
        "label": "random 100k SVI",
        "root": RANDOM_100K_SVI_ROOT,
        "kind": "svi",
        "color": "tab:red",
        "marker": "x",
        "offset": 0.0,
    },
    {
        "key": "vbmc",
        "label": "VBMC+alpha scalar delay",
        "root": VBMC_ROOT,
        "kind": "vbmc",
        "color": "tab:green",
        "marker": "^",
        "offset": 0.20,
    },
]

PARAM_SPECS = [
    ("rate_lambda", "rate_lambda", 1.0),
    ("T_0", "T_0 (ms)", 1000.0),
    ("theta_E", "theta_E", 1.0),
    ("rate_norm_l", "rate_norm_l", 1.0),
    ("alpha", "alpha", 1.0),
    ("lapse_prob", "lapse rate (%)", 100.0),
    ("lapse_prob_right", "lapse_prob_right", 1.0),
]

LIKELIHOOD_PARAM_NAMES = [
    "rate_lambda",
    "T_0",
    "theta_E",
    "w",
    "del_go",
    "rate_norm_l",
    "alpha",
    "lapse_prob",
    "lapse_prob_right",
]

sys.path.insert(0, str(SCRIPT_DIR))
from time_vary_norm_alpha_utils import cum_pro_and_reactive_time_vary_alpha_fn, up_or_down_RTs_fit_alpha_fn
import numpyro_npl_alpha_lapse_svi_utils as npl_alpha_lapse_utils


# %%
# =============================================================================
# Helpers
# =============================================================================
def animal_key(batch_name, animal):
    return f"{batch_name}_{int(animal)}"


def animal_label(batch_name, animal):
    return f"{batch_name}/{int(animal)}"


def ensure_choice_column(df):
    if "choice" not in df.columns:
        if "response_poke" not in df.columns:
            raise KeyError("Need either `choice` or `response_poke` in the batch CSV.")
        df = df.copy()
        df["choice"] = df["response_poke"].map({3: 1, 2: -1})
    return df


def load_abort_means(batch_name, animal):
    abort_pkl = REPO_DIR / "aborts_ipl_npl_time_fit_results" / f"results_{batch_name}_animal_{int(animal)}.pkl"
    if not abort_pkl.exists():
        raise FileNotFoundError(abort_pkl)
    with abort_pkl.open("rb") as handle:
        saved = pickle.load(handle)
    abort = saved["vbmc_aborts_results"]
    return {
        "V_A": float(np.mean(abort["V_A_samples"])),
        "theta_A": float(np.mean(abort["theta_A_samples"])),
        "t_A_aff": float(np.mean(abort["t_A_aff_samp"])),
    }


def load_valid_trials(batch_name, animal):
    csv_path = REPO_DIR / "raw_data" / "batch_csvs" / f"batch_{batch_name}_valid_and_aborts.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    raw_df = ensure_choice_column(pd.read_csv(csv_path))
    valid_df = raw_df[
        (raw_df["animal"].astype(int) == int(animal))
        & (raw_df["success"].isin([1, -1]))
        & (raw_df["RTwrtStim"] < 1)
        & (raw_df["ABL"].isin(ABLS))
    ].copy()
    valid_df = valid_df.dropna(subset=["TotalFixTime", "intended_fix", "ABL", "ILD", "choice", "RTwrtStim"])
    if valid_df.empty:
        raise RuntimeError(f"No valid RT<1 trials for {batch_name}/{animal}.")
    valid_df["ABL"] = valid_df["ABL"].astype(float)
    valid_df["ILD"] = valid_df["ILD"].astype(float)
    valid_df["choice"] = valid_df["choice"].astype(int)
    return valid_df.reset_index(drop=True)


def observed_condition_table(valid_df):
    condition_table = (
        valid_df[["ABL", "ILD"]]
        .drop_duplicates()
        .sort_values(["ABL", "ILD"])
        .reset_index(drop=True)
    )
    condition_table["condition_id"] = np.arange(len(condition_table), dtype=int)
    return condition_table


def load_summary(model_spec, batch_name, animal):
    key = animal_key(batch_name, animal)
    if model_spec["kind"] == "svi":
        path = model_spec["root"] / key / "main_fullrank_posterior_summary.csv"
    else:
        path = model_spec["root"] / key / "vbmc_norm_alpha_lapse_scalar_t_E_aff_posterior_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_svi_condition_table(model_spec, batch_name, animal):
    path = model_spec["root"] / animal_key(batch_name, animal) / "condition_table.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    condition_table = pd.read_csv(path)
    condition_table["ABL"] = condition_table["ABL"].astype(float)
    condition_table["ILD"] = condition_table["ILD"].astype(float)
    condition_table["condition_id"] = condition_table["condition_id"].astype(int)
    return condition_table.sort_values("condition_id").reset_index(drop=True)


def summary_row(summary_df, parameter):
    rows = summary_df[summary_df["parameter"].astype(str) == parameter]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one row for {parameter!r}, found {len(rows)}.")
    return rows.iloc[0]


def parameter_stats(summary_df, parameter, model_kind):
    row = summary_row(summary_df, parameter)
    return float(row["mean"]), float(row["q025"]), float(row["q975"]), True


def delay_vector_from_summary(summary_df, condition_table, scalar_delay=None):
    if scalar_delay is not None:
        return np.full(len(condition_table), float(scalar_delay), dtype=float)

    delay_rows = summary_df[summary_df["parameter"].astype(str).str.startswith("t_E_aff_")].copy()
    if delay_rows.empty:
        raise RuntimeError("No t_E_aff rows found in SVI posterior summary.")
    delay_rows["ABL"] = delay_rows["ABL"].astype(float)
    delay_rows["ILD"] = delay_rows["ILD"].astype(float)
    merged = condition_table[["condition_id", "ABL", "ILD"]].merge(
        delay_rows[["ABL", "ILD", "mean"]],
        on=["ABL", "ILD"],
        how="left",
        validate="one_to_one",
    )
    if merged["mean"].isna().any():
        raise RuntimeError(f"Missing delay means:\n{merged[merged['mean'].isna()]}")
    return merged.sort_values("condition_id")["mean"].to_numpy(dtype=float)


def make_jax_data(valid_df, condition_table, abort_params, T_trunc):
    condition_map = condition_table[["ABL", "ILD", "condition_id"]].copy()
    model_df = valid_df.merge(condition_map, on=["ABL", "ILD"], how="left", validate="many_to_one")
    if model_df["condition_id"].isna().any():
        raise RuntimeError(f"Failed to assign all conditions:\n{model_df[model_df['condition_id'].isna()].head()}")
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


def params_from_summary(summary_df, model_kind, condition_table):
    params = {}
    for parameter in LIKELIHOOD_PARAM_NAMES:
        params[parameter] = float(summary_row(summary_df, parameter)["mean"])

    if model_kind == "vbmc":
        scalar_delay = float(summary_row(summary_df, "t_E_aff")["mean"])
        params["t_E_aff"] = delay_vector_from_summary(summary_df, condition_table, scalar_delay=scalar_delay)
    else:
        params["t_E_aff"] = delay_vector_from_summary(summary_df, condition_table)
    return params


def common_loglike(params, data):
    value = npl_alpha_lapse_utils.npl_alpha_lapse_condition_delay_loglike(params, data, K_max=K_MAX)
    return float(jax.device_get(value))


def vbmc_alpha_scalar_loglike_numpy(valid_df, abort_params, T_trunc, params):
    z_e = (params["w"] - 0.5) * 2.0 * params["theta_E"]
    loglikes = []
    for _, row in valid_df.iterrows():
        pdf = up_or_down_RTs_fit_alpha_fn(
            float(row["TotalFixTime"]),
            int(row["choice"]),
            abort_params["V_A"],
            abort_params["theta_A"],
            abort_params["t_A_aff"],
            float(row["intended_fix"]),
            float(row["ABL"]),
            float(row["ILD"]),
            params["rate_lambda"],
            params["T_0"],
            params["theta_E"],
            z_e,
            float(params["t_E_aff"][0]),
            params["del_go"],
            np.nan,
            params["rate_norm_l"],
            params["alpha"],
            True,
            False,
            K_MAX,
        )
        trunc_factor = cum_pro_and_reactive_time_vary_alpha_fn(
            float(row["intended_fix"]) + 1.0,
            T_trunc,
            abort_params["V_A"],
            abort_params["theta_A"],
            abort_params["t_A_aff"],
            float(row["intended_fix"]),
            float(row["ABL"]),
            float(row["ILD"]),
            params["rate_lambda"],
            params["T_0"],
            params["theta_E"],
            z_e,
            float(params["t_E_aff"][0]),
            np.nan,
            params["rate_norm_l"],
            params["alpha"],
            True,
            False,
            K_MAX,
        ) - cum_pro_and_reactive_time_vary_alpha_fn(
            float(row["intended_fix"]),
            T_trunc,
            abort_params["V_A"],
            abort_params["theta_A"],
            abort_params["t_A_aff"],
            float(row["intended_fix"]),
            float(row["ABL"]),
            float(row["ILD"]),
            params["rate_lambda"],
            params["T_0"],
            params["theta_E"],
            z_e,
            float(params["t_E_aff"][0]),
            np.nan,
            params["rate_norm_l"],
            params["alpha"],
            True,
            False,
            K_MAX,
        )
        normalized_pdf = max(pdf / (trunc_factor + 1e-20), 1e-50)
        if int(row["choice"]) == 1:
            lapse_choice_pdf = params["lapse_prob_right"]
        else:
            lapse_choice_pdf = 1.0 - params["lapse_prob_right"]
        mixture_pdf = (1.0 - params["lapse_prob"]) * normalized_pdf + params["lapse_prob"] * lapse_choice_pdf
        loglikes.append(np.log(max(mixture_pdf, 1e-50)))
    return float(np.sum(loglikes))


def representative_validation_rows(valid_df, batch_name, animal):
    if len(valid_df) <= VALIDATION_MAX_ROWS:
        return valid_df.copy()
    n_conditions = valid_df[["ABL", "ILD"]].drop_duplicates().shape[0]
    per_condition = max(1, int(np.ceil(VALIDATION_MAX_ROWS / n_conditions)))
    sampled = (
        valid_df.groupby(["ABL", "ILD"], group_keys=False)
        .sample(n=per_condition, replace=True, random_state=RNG_SEED + int(animal))
        .drop_duplicates()
        .head(VALIDATION_MAX_ROWS)
        .sort_index()
        .copy()
    )
    return sampled.reset_index(drop=True)


def svi_elbo_from_convergence(model_spec, batch_name, animal):
    path = model_spec["root"] / animal_key(batch_name, animal) / "main_fullrank_convergence_checks.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    conv_df = pd.read_csv(path)
    final_row = conv_df.iloc[-1]
    return -float(final_row["best_mean_loss_so_far"]), int(final_row["best_end_step_so_far"]), int(final_row["end_step"])


def vbmc_elbo_from_summary(batch_name, animal):
    path = VBMC_ROOT / animal_key(batch_name, animal) / "vbmc_norm_alpha_lapse_scalar_t_E_aff_run_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    row = pd.read_csv(path).iloc[0]
    return float(row["final_elbo"]), float(row["final_elbo_sd"]), int(row["final_iter"])


# %%
# =============================================================================
# Collect parameters, ELBOs, common log likelihoods, and validation rows
# =============================================================================
param_rows = []
objective_rows = []
validation_rows = []

for batch_name, animal in ANIMALS:
    label = animal_label(batch_name, animal)
    valid_df = load_valid_trials(batch_name, animal)
    abort_params = load_abort_means(batch_name, animal)
    T_trunc = BATCH_T_TRUNC.get(batch_name, DEFAULT_T_TRUNC)

    for model_spec in MODEL_SPECS:
        model_key = model_spec["key"]
        summary_df = load_summary(model_spec, batch_name, animal)
        if model_spec["kind"] == "svi":
            condition_table = load_svi_condition_table(model_spec, batch_name, animal)
        else:
            condition_table = observed_condition_table(valid_df)

        params = params_from_summary(summary_df, model_spec["kind"], condition_table)
        data = make_jax_data(valid_df, condition_table, abort_params, T_trunc)
        loglike = common_loglike(params, data)

        if model_spec["kind"] == "svi":
            elbo, best_step, checked_step = svi_elbo_from_convergence(model_spec, batch_name, animal)
            elbo_sd = np.nan
            fit_iter = np.nan
        else:
            elbo, elbo_sd, fit_iter = vbmc_elbo_from_summary(batch_name, animal)
            best_step = np.nan
            checked_step = np.nan

        objective_rows.append(
            {
                "batch_name": batch_name,
                "animal": int(animal),
                "animal_label": label,
                "model_key": model_key,
                "model_label": model_spec["label"],
                "fit_kind": model_spec["kind"],
                "n_valid_trials": int(len(valid_df)),
                "n_conditions": int(len(condition_table)),
                "T_trunc": float(T_trunc),
                "elbo": float(elbo),
                "elbo_sd": float(elbo_sd) if np.isfinite(elbo_sd) else np.nan,
                "best_step": best_step,
                "checked_step": checked_step,
                "fit_iter": fit_iter,
                "common_loglike_at_posterior_mean": float(loglike),
                "common_likelihood_alpha_policy": "VBMC and SVI alpha from posterior mean",
            }
        )

        for parameter, display_name, scale in PARAM_SPECS:
            mean, q025, q975, has_interval = parameter_stats(summary_df, parameter, model_spec["kind"])
            param_rows.append(
                {
                    "batch_name": batch_name,
                    "animal": int(animal),
                    "animal_label": label,
                    "model_key": model_key,
                    "model_label": model_spec["label"],
                    "fit_kind": model_spec["kind"],
                    "parameter": parameter,
                    "display_parameter": display_name,
                    "mean": float(mean),
                    "q025": float(q025),
                    "q975": float(q975),
                    "display_mean": float(mean) * scale,
                    "display_q025": float(q025) * scale,
                    "display_q975": float(q975) * scale,
                    "has_interval": bool(has_interval),
                    "scale": float(scale),
                }
            )

        if model_spec["kind"] == "vbmc":
            validation_df = representative_validation_rows(valid_df, batch_name, animal)
            validation_condition_table = observed_condition_table(validation_df)
            scalar_delay = float(summary_row(summary_df, "t_E_aff")["mean"])
            validation_params = dict(params)
            validation_params["t_E_aff"] = np.full(len(validation_condition_table), scalar_delay, dtype=float)
            validation_data = make_jax_data(validation_df, validation_condition_table, abort_params, T_trunc)
            jax_loglike = common_loglike(validation_params, validation_data)
            numpy_loglike = vbmc_alpha_scalar_loglike_numpy(validation_df, abort_params, T_trunc, validation_params)
            validation_rows.append(
                {
                    "batch_name": batch_name,
                    "animal": int(animal),
                    "n_validation_rows": int(len(validation_df)),
                    "n_conditions": int(len(validation_condition_table)),
                    "T_trunc": float(T_trunc),
                    "vbmc_alpha": float(validation_params["alpha"]),
                    "vbmc_scalar_t_E_aff_s": scalar_delay,
                    "numpy_alpha_loglike": float(numpy_loglike),
                    "common_jax_alpha_loglike": float(jax_loglike),
                    "abs_diff": float(abs(numpy_loglike - jax_loglike)),
                }
            )

param_df = pd.DataFrame(param_rows)
objective_df = pd.DataFrame(objective_rows)
validation_df = pd.DataFrame(validation_rows)

param_df.to_csv(PARAM_CSV, index=False)
objective_df.to_csv(OBJECTIVE_CSV, index=False)
validation_df.to_csv(VALIDATION_CSV, index=False)

print("Parameter rows:")
print(param_df[["animal_label", "model_key", "display_parameter", "display_mean", "display_q025", "display_q975"]].to_string(index=False))
print(f"\nSaved parameter CSV: {PARAM_CSV}")
print("\nObjective rows:")
print(objective_df[["animal_label", "model_key", "elbo", "elbo_sd", "common_loglike_at_posterior_mean"]].to_string(index=False))
print(f"\nSaved objective CSV: {OBJECTIVE_CSV}")
print("\nVBMC likelihood validation:")
print(validation_df.to_string(index=False))
print(f"\nSaved validation CSV: {VALIDATION_CSV}")


# %%
# =============================================================================
# Plot parameter means and intervals
# =============================================================================
animal_labels = [animal_label(batch, animal) for batch, animal in ANIMALS]
x = np.arange(len(animal_labels), dtype=float)
model_by_key = {spec["key"]: spec for spec in MODEL_SPECS}

fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.8), sharex=True)
axes = axes.ravel()

for ax_idx, (parameter, display_name, _scale) in enumerate(PARAM_SPECS):
    ax = axes[ax_idx]
    subset = param_df[param_df["parameter"] == parameter]
    for model_spec in MODEL_SPECS:
        rows = (
            subset[subset["model_key"] == model_spec["key"]]
            .set_index("animal_label")
            .loc[animal_labels]
            .reset_index()
        )
        y = rows["display_mean"].to_numpy(dtype=float)
        low = rows["display_q025"].to_numpy(dtype=float)
        high = rows["display_q975"].to_numpy(dtype=float)
        has_interval = rows["has_interval"].to_numpy(dtype=bool)
        yerr = np.vstack([y - low, high - y])
        yerr[:, ~has_interval] = 0.0
        ax.errorbar(
            x + model_spec["offset"],
            y,
            yerr=yerr,
            fmt=model_spec["marker"],
            color=model_spec["color"],
            ecolor=model_spec["color"],
            elinewidth=1.1,
            capsize=2.5,
            markersize=5.0,
            linestyle="none",
            alpha=0.95,
            label=model_spec["label"],
        )
    ax.set_title(display_name)
    ax.grid(axis="y", alpha=0.22, linewidth=0.6)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)

for ax in axes[: len(PARAM_SPECS)]:
    ax.set_xticks(x)
    ax.set_xticklabels(animal_labels, rotation=45, ha="right")

axes[-1].axis("off")
handles = [
    Line2D([0], [0], marker=spec["marker"], color=spec["color"], linestyle="none", label=spec["label"])
    for spec in MODEL_SPECS
]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
fig.suptitle("NPL+alpha+lapse SVI vs refreshed VBMC+alpha scalar-delay parameter means", fontsize=14, y=0.99)
fig.tight_layout(rect=[0, 0.06, 1, 0.95])
fig.savefig(PARAM_FIG, dpi=200, bbox_inches="tight")
print(f"Saved parameter figure: {PARAM_FIG}")


# %%
# =============================================================================
# Plot ELBO and common likelihood
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True)

for model_spec in MODEL_SPECS:
    rows = (
        objective_df[objective_df["model_key"] == model_spec["key"]]
        .set_index("animal_label")
        .loc[animal_labels]
        .reset_index()
    )
    elbo = rows["elbo"].to_numpy(dtype=float)
    elbo_sd = rows["elbo_sd"].to_numpy(dtype=float)
    elbo_yerr = np.where(np.isfinite(elbo_sd), elbo_sd, 0.0)
    axes[0].errorbar(
        x + model_spec["offset"],
        elbo,
        yerr=elbo_yerr,
        fmt=model_spec["marker"],
        color=model_spec["color"],
        ecolor=model_spec["color"],
        elinewidth=1.0,
        capsize=2.5,
        markersize=5.5,
        linestyle="none",
        alpha=0.95,
    )
    axes[1].scatter(
        x + model_spec["offset"],
        rows["common_loglike_at_posterior_mean"].to_numpy(dtype=float),
        marker=model_spec["marker"],
        color=model_spec["color"],
        s=38,
        alpha=0.95,
    )

axes[0].set_title("ELBO / restored objective")
axes[0].set_ylabel("ELBO")
axes[1].set_title("Common log likelihood at posterior mean")
axes[1].set_ylabel("log likelihood")

for ax in axes:
    ax.set_xticks(x)
    ax.set_xticklabels(animal_labels, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.22, linewidth=0.6)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)

fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
fig.suptitle("NPL+alpha+lapse SVI/VBMC+alpha objectives and common likelihoods", fontsize=14, y=0.98)
fig.tight_layout(rect=[0, 0.10, 1, 0.90])
fig.savefig(OBJECTIVE_FIG, dpi=200, bbox_inches="tight")
print(f"Saved objective figure: {OBJECTIVE_FIG}")

# %%
