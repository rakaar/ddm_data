# %%
"""
Build compact Figure 4 supplementary data from the direct NPL+alpha SVI fit.

The paper panels reuse the direct_37_param_svi row from the existing three-way
diagnostic. Posterior corner summaries are rebuilt from each animal's saved
10,000-sample variational posterior.
"""

# %%
# =============================================================================
# Editable parameters
# =============================================================================
from pathlib import Path
import pickle

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ANIMAL_DIR = SCRIPT_DIR.parent
REPO_ROOT = ANIMAL_DIR.parent

NPL_SVI_ROOT = (
    ANIMAL_DIR
    / "numpyro_svi_npl_alpha_condition_delay_patience12_restore_best_outputs"
)
BIG_SVI_ROOT = (
    REPO_ROOT
    / "fit_each_condn"
    / "svi_big_gamma_omega_delay_patience12_restore_best_all_animals_outputs"
)
INPUT_PAYLOAD = (
    NPL_SVI_ROOT
    / "three_npl_param_source_comparison"
    / "three_npl_param_sources_patience12_3x5.pkl"
)
OUTPUT_PKL = SCRIPT_DIR / "npl_svi_patience12_fig4_supplementary_bundle.pkl"

METHOD_KEY = "direct_37_param_svi"
MODEL_LABEL = "Direct patience-12 NPL+alpha condition-delay SVI"
PARAMS = ["rate_lambda", "T_0", "theta_E", "rate_norm_l", "alpha"]
EXPECTED_N_ANIMALS = 30
EXPECTED_N_CONDITION_ROWS = 864
EXPECTED_N_POSTERIOR_SAMPLES = 10_000
ABLS = [20, 40, 60]
ILD_ARR = np.array([-16.0, -8.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0, 16.0])
PAPER_QUANTILES = [0.1, 0.3, 0.5, 0.7, 0.9]


# %%
# =============================================================================
# Helpers used for both discrete and continuous quantile dictionaries
# =============================================================================
def subset_quantile_entries(entries, indices):
    if len(entries) == 0:
        return []
    values = np.asarray(entries, dtype=float)
    if values.ndim != 2:
        raise RuntimeError(f"Expected 2-D quantile entries, found {values.shape}")
    return values[:, indices].tolist()


def subset_quantile_plot_dict(source, indices):
    subset = {}
    for abl in ABLS:
        subset[abl] = {}
        for abs_ild, values in source[abl].items():
            subset[abl][float(abs_ild)] = {
                "empirical": subset_quantile_entries(values.get("empirical", []), indices),
                "theoretical": subset_quantile_entries(
                    values.get("theoretical", []), indices
                ),
            }
    return subset


# %%
# =============================================================================
# Load and validate the existing direct-SVI panel payload
# =============================================================================
print(f"Direct NPL SVI root: {NPL_SVI_ROOT}")
print(f"Big Gamma/Omega SVI root: {BIG_SVI_ROOT}")
print(f"Input diagnostic payload: {INPUT_PAYLOAD}")
print(f"Output bundle: {OUTPUT_PKL}")

with INPUT_PAYLOAD.open("rb") as handle:
    payload = pickle.load(handle)

required_keys = [
    "animal_keys",
    "npl_svi_root",
    "big_svi_root",
    "npl_condition_rows",
    "big_condition_rows",
    "method_param_rows",
    "gamma_omega_model_summary",
    "psy_by_method",
    "slopes_by_method",
    "quantile_by_method",
]
missing_keys = [key for key in required_keys if key not in payload]
if missing_keys:
    raise RuntimeError(f"Source payload is missing keys: {missing_keys}")

if Path(payload["npl_svi_root"]).resolve() != NPL_SVI_ROOT.resolve():
    raise RuntimeError("Source payload points to a different NPL SVI fit root")
if Path(payload["big_svi_root"]).resolve() != BIG_SVI_ROOT.resolve():
    raise RuntimeError("Source payload points to a different big-SVI fit root")

animal_keys = [(str(batch), int(animal)) for batch, animal in payload["animal_keys"]]
if len(animal_keys) != EXPECTED_N_ANIMALS or len(set(animal_keys)) != EXPECTED_N_ANIMALS:
    raise RuntimeError(f"Expected 30 unique animals, found {len(set(animal_keys))}")

npl_condition_df = pd.DataFrame(payload["npl_condition_rows"])
big_condition_df = pd.DataFrame(payload["big_condition_rows"])
if len(npl_condition_df) != EXPECTED_N_CONDITION_ROWS:
    raise RuntimeError(f"Expected 864 NPL condition rows, found {len(npl_condition_df)}")
if len(big_condition_df) != EXPECTED_N_CONDITION_ROWS:
    raise RuntimeError(f"Expected 864 big-SVI condition rows, found {len(big_condition_df)}")

method_param_df = pd.DataFrame(payload["method_param_rows"])
method_param_df = method_param_df[method_param_df["method_key"] == METHOD_KEY].copy()
if len(method_param_df) != EXPECTED_N_ANIMALS:
    raise RuntimeError(f"Expected 30 direct-SVI parameter rows, found {len(method_param_df)}")

model_summary_df = pd.DataFrame(payload["gamma_omega_model_summary"])
model_summary_df = model_summary_df[model_summary_df["method_key"] == METHOD_KEY].copy()
if model_summary_df.empty:
    raise RuntimeError(f"No Gamma/Omega model rows found for {METHOD_KEY}")

for section in ["psy_by_method", "slopes_by_method", "quantile_by_method"]:
    if METHOD_KEY not in payload[section]:
        raise RuntimeError(f"{METHOD_KEY} is missing from payload['{section}']")


# %%
# =============================================================================
# Compact paper-panel data
# =============================================================================
psy_source = payload["psy_by_method"][METHOD_KEY]
psy_data = {
    "model_label": MODEL_LABEL,
    "animal_keys": animal_keys,
    "ILD_arr": ILD_ARR.copy(),
    "ABL_arr": ABLS.copy(),
    "empirical_agg": psy_source["empirical_agg"],
    "theory_agg": psy_source["theory_agg"],
    "sd_psychometric_model_abs_ild_max": 8,
}

sd_rows = np.array([batch == "SD" for batch, _animal in animal_keys])
high_ild_cols = np.abs(ILD_ARR) > 8
sd_high_model_entries = 0
for abl in ABLS:
    theory = np.asarray(psy_data["theory_agg"][abl], dtype=float)
    if theory.shape != (EXPECTED_N_ANIMALS, len(ILD_ARR)):
        raise RuntimeError(f"Unexpected ABL {abl} psychometric shape: {theory.shape}")
    sd_high_model_entries += int(
        np.sum(np.isfinite(theory[np.ix_(sd_rows, high_ild_cols)]))
    )
if sd_high_model_entries != 0:
    raise RuntimeError(
        f"Found {sd_high_model_entries} SD psychometric entries above |ILD|=8"
    )

slopes_source = payload["slopes_by_method"][METHOD_KEY]
slopes_data = {
    "model_label": MODEL_LABEL,
    "animal_keys": animal_keys,
    "data_means": np.asarray(slopes_source["data_means"], dtype=float),
    "model_means": np.asarray(slopes_source["model_means"], dtype=float),
    "slopes_data": slopes_source["slopes_data"],
    "slopes_model": slopes_source["slopes_model"],
    "sd_psychometric_model_abs_ild_max": 8,
}
if len(slopes_data["data_means"]) != EXPECTED_N_ANIMALS:
    raise RuntimeError("Slope data does not contain all 30 animals")
if len(slopes_data["model_means"]) != EXPECTED_N_ANIMALS:
    raise RuntimeError("Slope model does not contain all 30 animals")

quantile_source = payload["quantile_by_method"][METHOD_KEY]
available_quantiles = np.asarray(quantile_source["QUANTILES_TO_PLOT"], dtype=float)
quantile_indices = []
for quantile in PAPER_QUANTILES:
    matches = np.flatnonzero(np.isclose(available_quantiles, quantile))
    if len(matches) != 1:
        raise RuntimeError(
            f"Quantile {quantile} not found exactly once in {available_quantiles.tolist()}"
        )
    quantile_indices.append(int(matches[0]))

quant_data = {
    "model_label": MODEL_LABEL,
    "animal_keys": animal_keys,
    "ABL_arr": ABLS.copy(),
    "abs_ild_sorted": [float(value) for value in quantile_source["abs_ild_sorted"]],
    "continuous_abs_ild": [
        float(value) for value in quantile_source["continuous_abs_ild"]
    ],
    "QUANTILES_TO_PLOT": PAPER_QUANTILES.copy(),
    "plot_data": subset_quantile_plot_dict(
        quantile_source["plot_data"], quantile_indices
    ),
    "continuous_plot_data": subset_quantile_plot_dict(
        quantile_source["continuous_plot_data_sd_flat"], quantile_indices
    ),
    "sd_flat_delay_policy": (
        "For SD animals, model t_E_aff is held flat after |ILD|=8 in the "
        "continuous quantile curve."
    ),
}

sd_flat_counts_at_16 = {
    abl: len(quant_data["continuous_plot_data"][abl][16.0]["theoretical"])
    for abl in ABLS
}
if set(sd_flat_counts_at_16.values()) != {EXPECTED_N_ANIMALS}:
    raise RuntimeError(
        f"Expected 30 SD-flat model entries at |ILD|=16, got {sd_flat_counts_at_16}"
    )

gamma_omega_data = {
    "model_label": MODEL_LABEL,
    "animal_keys": animal_keys,
    "ABL_arr": ABLS.copy(),
    "ILD_arr": ILD_ARR.copy(),
    "condition_rows": big_condition_df.to_dict("records"),
    "model_summary_rows": model_summary_df.to_dict("records"),
    "condition_source": (
        "patience12 92-param Gamma/Omega/delay SVI condition posterior means"
    ),
    "model_source": (
        "direct patience12 37-param NPL+alpha condition-delay SVI posterior means"
    ),
}


# %%
# =============================================================================
# Compact animal-wise posterior summaries for the upper-triangular corner
# =============================================================================
posterior_rows = []
for batch_name, animal in animal_keys:
    posterior_path = (
        NPL_SVI_ROOT
        / f"{batch_name}_{animal}"
        / "main_fullrank_posterior_samples.npz"
    )
    if not posterior_path.exists():
        raise FileNotFoundError(posterior_path)

    with np.load(posterior_path) as posterior:
        missing_params = [param for param in PARAMS if param not in posterior.files]
        if missing_params:
            raise RuntimeError(f"{posterior_path} is missing {missing_params}")
        sample_matrix = np.column_stack(
            [np.asarray(posterior[param], dtype=float).reshape(-1) for param in PARAMS]
        )

    if sample_matrix.shape != (EXPECTED_N_POSTERIOR_SAMPLES, len(PARAMS)):
        raise RuntimeError(
            f"Unexpected posterior shape for {batch_name}/{animal}: {sample_matrix.shape}"
        )
    if not np.all(np.isfinite(sample_matrix)):
        raise RuntimeError(f"Non-finite posterior samples for {batch_name}/{animal}")

    covariance = np.cov(sample_matrix, rowvar=False)
    if covariance.shape != (len(PARAMS), len(PARAMS)):
        raise RuntimeError(f"Unexpected covariance shape for {batch_name}/{animal}")
    if not np.all(np.isfinite(covariance)):
        raise RuntimeError(f"Non-finite covariance for {batch_name}/{animal}")
    if np.min(np.linalg.eigvalsh(covariance)) < -1e-10:
        raise RuntimeError(f"Non-positive-semidefinite covariance for {batch_name}/{animal}")

    means = np.mean(sample_matrix, axis=0)
    q025 = np.quantile(sample_matrix, 0.025, axis=0)
    q975 = np.quantile(sample_matrix, 0.975, axis=0)
    posterior_rows.append(
        {
            "batch_name": batch_name,
            "animal": animal,
            "animal_label": f"{batch_name}/{animal}",
            "n_samples": int(sample_matrix.shape[0]),
            "mean": dict(zip(PARAMS, means.tolist())),
            "q025": dict(zip(PARAMS, q025.tolist())),
            "q975": dict(zip(PARAMS, q975.tolist())),
            "covariance": covariance.tolist(),
        }
    )

direct_mean_lookup = {
    (str(row.batch_name), int(row.animal)): row
    for row in method_param_df.itertuples(index=False)
}
max_direct_mean_difference = 0.0
for row in posterior_rows:
    source_row = direct_mean_lookup[(row["batch_name"], row["animal"])]
    for param in PARAMS:
        difference = abs(row["mean"][param] - float(getattr(source_row, param)))
        max_direct_mean_difference = max(max_direct_mean_difference, difference)
if max_direct_mean_difference > 1e-6:
    raise RuntimeError(
        "Posterior means disagree with the direct-SVI diagnostic payload: "
        f"max difference={max_direct_mean_difference:.3g}"
    )


# %%
# =============================================================================
# Save one self-contained supplementary-figure bundle
# =============================================================================
bundle = {
    "model_label": MODEL_LABEL,
    "method_key": METHOD_KEY,
    "source_payload": str(INPUT_PAYLOAD),
    "npl_svi_root": str(NPL_SVI_ROOT),
    "big_svi_root": str(BIG_SVI_ROOT),
    "params": PARAMS.copy(),
    "psy_data": psy_data,
    "quant_data": quant_data,
    "slopes_data": slopes_data,
    "gamma_omega_data": gamma_omega_data,
    "posterior_rows": posterior_rows,
    "checks": {
        "n_animals": len(animal_keys),
        "n_npl_condition_rows": len(npl_condition_df),
        "n_big_condition_rows": len(big_condition_df),
        "posterior_samples_per_animal": EXPECTED_N_POSTERIOR_SAMPLES,
        "sd_psychometric_high_ild_model_entries": sd_high_model_entries,
        "sd_flat_quantile_entries_at_abs_ild_16": sd_flat_counts_at_16,
        "paper_quantiles": PAPER_QUANTILES.copy(),
        "max_direct_mean_difference": max_direct_mean_difference,
    },
}

with OUTPUT_PKL.open("wb") as handle:
    pickle.dump(bundle, handle)

print(f"Validated animals: {len(animal_keys)}")
print(f"Validated NPL condition rows: {len(npl_condition_df)}")
print(f"Validated big-SVI condition rows: {len(big_condition_df)}")
print(f"Posterior samples per animal: {EXPECTED_N_POSTERIOR_SAMPLES}")
print(f"Maximum direct-mean cross-check difference: {max_direct_mean_difference:.3g}")
print(f"SD psychometric entries above |ILD|=8: {sd_high_model_entries}")
print(f"SD-flat quantile entries at |ILD|=16: {sd_flat_counts_at_16}")
print(f"Saved supplementary bundle: {OUTPUT_PKL}")
