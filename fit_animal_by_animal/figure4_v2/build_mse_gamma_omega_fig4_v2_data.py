# %%
"""
Build paper Fig. 4-style data products for the NPL+alpha Gamma+Omega MSE params.

This repackages the first row of
compare_three_npl_param_sources_patience12_3x5.py into compact Fig. 4 v2
pickles. It does not rerun the expensive RTD integrations.
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

BIG_SVI_ROOT = REPO_ROOT / "fit_each_condn" / "svi_big_gamma_omega_delay_patience12_restore_best_all_animals_outputs"
NPL_SOURCE_ROOT = ANIMAL_DIR / "numpyro_svi_npl_alpha_condition_delay_patience12_restore_best_outputs"
INPUT_PAYLOAD = (
    NPL_SOURCE_ROOT
    / "three_npl_param_source_comparison"
    / "three_npl_param_sources_patience12_3x5.pkl"
)
MSE_PARAM_CSV = (
    BIG_SVI_ROOT
    / "mse_alpha_model_comparison"
    / "objective_variants"
    / "gamma_omega"
    / "per_animal_mse_gamma_omega_alpha_params.csv"
)

OUTPUT_DIR = SCRIPT_DIR
PSY_OUTPUT = OUTPUT_DIR / "mse_gamma_omega_npl_alpha_psy_fig4_v2_data.pkl"
QUANT_OUTPUT = OUTPUT_DIR / "mse_gamma_omega_npl_alpha_quant_fig4_v2_data.pkl"
SLOPES_OUTPUT = OUTPUT_DIR / "mse_gamma_omega_npl_alpha_slopes_fig4_v2_data.pkl"
GAMMA_OUTPUT = OUTPUT_DIR / "mse_gamma_omega_npl_alpha_gamma_fig4_v2_data.pkl"
BUNDLE_OUTPUT = OUTPUT_DIR / "mse_gamma_omega_npl_alpha_fig4_v2_bundle.pkl"

METHOD_KEY = "mse_gamma_omega"
MODEL_LABEL = "NPL+alpha parameters fit by Gamma+Omega MSE"

EXPECTED_N_ANIMALS = 30
EXPECTED_N_CONDITION_ROWS = 864
ABLS = [20, 40, 60]
ILD_ARR = np.array([-16.0, -8.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0, 16.0], dtype=float)
ABS_ILD_SORTED = sorted({float(abs(ild)) for ild in ILD_ARR})
PAPER_QUANTILES = [0.1, 0.3, 0.5, 0.7, 0.9]


# %%
# =============================================================================
# Helpers
# =============================================================================
def subset_quantile_entries(entries, indices):
    if len(entries) == 0:
        return []
    arr = np.asarray(entries, dtype=float)
    if arr.ndim != 2:
        raise RuntimeError(f"Expected quantile entries to be 2-D, found shape {arr.shape}")
    return arr[:, indices].tolist()


def subset_quantile_plot_dict(source, indices):
    out = {}
    for abl in ABLS:
        out[abl] = {}
        for abs_ild, values in source[abl].items():
            out[abl][float(abs_ild)] = {
                "empirical": subset_quantile_entries(values.get("empirical", []), indices),
                "theoretical": subset_quantile_entries(values.get("theoretical", []), indices),
            }
    return out


def to_float_list(values):
    return [float(v) for v in values]


# %%
# =============================================================================
# Load and validate source payload
# =============================================================================
print(f"Input 3x5 payload: {INPUT_PAYLOAD}")
print(f"Big 92-param SVI root: {BIG_SVI_ROOT}")
print(f"Gamma+Omega MSE parameter CSV: {MSE_PARAM_CSV}")
print(f"Output folder: {OUTPUT_DIR}")

with INPUT_PAYLOAD.open("rb") as handle:
    payload = pickle.load(handle)

required_keys = [
    "animal_keys",
    "big_svi_root",
    "methods",
    "big_condition_rows",
    "method_param_rows",
    "gamma_omega_model_summary",
    "psy_by_method",
    "slopes_by_method",
    "quantile_by_method",
]
missing = [key for key in required_keys if key not in payload]
if missing:
    raise RuntimeError(f"Missing required payload keys: {missing}")

payload_big_root = Path(payload["big_svi_root"]).resolve()
if payload_big_root != BIG_SVI_ROOT.resolve():
    raise RuntimeError(f"Payload big-SVI root mismatch: {payload_big_root} != {BIG_SVI_ROOT.resolve()}")

method_entries = [method for method in payload["methods"] if method["key"] == METHOD_KEY]
if len(method_entries) != 1:
    raise RuntimeError(f"Expected exactly one {METHOD_KEY} method entry, found {len(method_entries)}")
method_mse_csv = Path(method_entries[0]["mse_csv"]).resolve()
if method_mse_csv != MSE_PARAM_CSV.resolve():
    raise RuntimeError(f"MSE CSV mismatch: {method_mse_csv} != {MSE_PARAM_CSV.resolve()}")

animal_keys = [(str(batch), int(animal)) for batch, animal in payload["animal_keys"]]
if len(animal_keys) != EXPECTED_N_ANIMALS:
    raise RuntimeError(f"Expected {EXPECTED_N_ANIMALS} animals, found {len(animal_keys)}")

big_condition_df = pd.DataFrame(payload["big_condition_rows"])
if len(big_condition_df) != EXPECTED_N_CONDITION_ROWS:
    raise RuntimeError(
        f"Expected {EXPECTED_N_CONDITION_ROWS} condition rows, found {len(big_condition_df)}"
    )
big_condition_df["batch_name"] = big_condition_df["batch_name"].astype(str)
big_condition_df["animal"] = big_condition_df["animal"].astype(int)
big_condition_df["ABL"] = big_condition_df["ABL"].astype(int)
big_condition_df["ILD"] = big_condition_df["ILD"].astype(float)

method_param_df = pd.DataFrame(payload["method_param_rows"])
method_param_df = method_param_df[method_param_df["method_key"] == METHOD_KEY].copy()
if len(method_param_df) != EXPECTED_N_ANIMALS:
    raise RuntimeError(f"Expected {EXPECTED_N_ANIMALS} MSE parameter rows, found {len(method_param_df)}")

method_summary_df = pd.DataFrame(payload["gamma_omega_model_summary"])
method_summary_df["method_key"] = method_summary_df["method_key"].astype(str)
method_summary_df["ABL"] = method_summary_df["ABL"].astype(int)
method_summary_df["ILD"] = method_summary_df["ILD"].astype(float)
method_summary_df = method_summary_df[method_summary_df["method_key"] == METHOD_KEY].copy()
if method_summary_df.empty:
    raise RuntimeError(f"Method {METHOD_KEY} not found in Gamma/Omega model summary")

for top_key in ["psy_by_method", "slopes_by_method", "quantile_by_method"]:
    if METHOD_KEY not in payload[top_key]:
        raise RuntimeError(f"{METHOD_KEY} not found in payload['{top_key}']")

print(f"Validated animals: {len(animal_keys)}")
print(f"Validated condition rows: {len(big_condition_df)}")
print(f"Validated MSE parameter rows: {len(method_param_df)}")


# %%
# =============================================================================
# Psychometric and slope payloads
# =============================================================================
psy_src = payload["psy_by_method"][METHOD_KEY]
psy_data = {
    "model_label": MODEL_LABEL,
    "method_key": METHOD_KEY,
    "animal_keys": animal_keys,
    "ILD_arr": ILD_ARR.copy(),
    "ABL_arr": ABLS.copy(),
    "empirical_agg": psy_src["empirical_agg"],
    "theory_agg": psy_src["theory_agg"],
    "sd_psychometric_model_abs_ild_max": 8,
}

sd_rows = np.array([batch == "SD" for batch, _animal in animal_keys])
high_cols = np.abs(ILD_ARR) > 8
sd_high_model_entries = 0
for abl in ABLS:
    theory = np.asarray(psy_data["theory_agg"][abl], dtype=float)
    if theory.shape != (EXPECTED_N_ANIMALS, len(ILD_ARR)):
        raise RuntimeError(f"Unexpected psychometric theory shape for ABL {abl}: {theory.shape}")
    sd_high_model_entries += int(np.sum(np.isfinite(theory[np.ix_(sd_rows, high_cols)])))

if sd_high_model_entries != 0:
    raise RuntimeError(f"Expected zero SD psychometric model entries at |ILD|>8, found {sd_high_model_entries}")

slopes_src = payload["slopes_by_method"][METHOD_KEY]
slopes_data = {
    "model_label": MODEL_LABEL,
    "method_key": METHOD_KEY,
    "animal_keys": animal_keys,
    "data_means": np.asarray(slopes_src["data_means"], dtype=float),
    "model_means": np.asarray(slopes_src["model_means"], dtype=float),
    "norm_means": np.asarray(slopes_src["model_means"], dtype=float),
    "slopes_data": slopes_src["slopes_data"],
    "slopes_model": slopes_src["slopes_model"],
    "sd_psychometric_model_abs_ild_max": 8,
}

if slopes_data["data_means"].shape[0] != EXPECTED_N_ANIMALS:
    raise RuntimeError(f"Unexpected slope data count: {slopes_data['data_means'].shape[0]}")
if slopes_data["model_means"].shape[0] != EXPECTED_N_ANIMALS:
    raise RuntimeError(f"Unexpected slope model count: {slopes_data['model_means'].shape[0]}")


# %%
# =============================================================================
# RT quantile payload
# =============================================================================
quant_src = payload["quantile_by_method"][METHOD_KEY]
all_quantiles = np.asarray(quant_src["QUANTILES_TO_PLOT"], dtype=float)
quantile_indices = []
for q in PAPER_QUANTILES:
    matches = np.where(np.isclose(all_quantiles, float(q)))[0]
    if len(matches) != 1:
        raise RuntimeError(f"Paper quantile {q} not found in source quantiles {all_quantiles.tolist()}")
    quantile_indices.append(int(matches[0]))

plot_data = subset_quantile_plot_dict(quant_src["plot_data"], quantile_indices)
continuous_sd_flat = subset_quantile_plot_dict(quant_src["continuous_plot_data_sd_flat"], quantile_indices)
continuous_standard = subset_quantile_plot_dict(quant_src["continuous_plot_data"], quantile_indices)
continuous_abs_ild = to_float_list(quant_src["continuous_abs_ild"])

sd_flat_counts_at_16 = {
    abl: len(continuous_sd_flat[abl][float(16.0)]["theoretical"])
    for abl in ABLS
}
if set(sd_flat_counts_at_16.values()) != {EXPECTED_N_ANIMALS}:
    raise RuntimeError(f"Expected 30 SD-flat quantile entries at |ILD|=16, got {sd_flat_counts_at_16}")

quant_data = {
    "model_label": MODEL_LABEL,
    "method_key": METHOD_KEY,
    "animal_keys": animal_keys,
    "ABL_arr": ABLS.copy(),
    "abs_ild_sorted": ABS_ILD_SORTED,
    "continuous_abs_ild": continuous_abs_ild,
    "QUANTILES_TO_PLOT": PAPER_QUANTILES.copy(),
    "plot_data": plot_data,
    "continuous_plot_data": continuous_sd_flat,
    "continuous_plot_data_sd_flat": continuous_sd_flat,
    "continuous_plot_data_standard": continuous_standard,
    "sd_flat_delay_policy": "For SD animals, model t_E_aff is held flat after |ILD|=8 in the continuous quantile curve.",
}


# %%
# =============================================================================
# Gamma payload
# =============================================================================
gamma_data = {
    "model_label": MODEL_LABEL,
    "method_key": METHOD_KEY,
    "animal_keys": animal_keys,
    "ABL_arr": ABLS.copy(),
    "ILD_arr": ILD_ARR.copy(),
    "condition_rows": big_condition_df.to_dict("records"),
    "model_summary_rows": method_summary_df.to_dict("records"),
    "method_param_rows": method_param_df.to_dict("records"),
    "condition_source": "patience12 92-param Gamma/Omega/delay SVI condition posterior means",
    "model_source": "NPL+alpha parameters fit animal-wise by MSE to 92-param Gamma+Omega posterior means",
}


# %%
# =============================================================================
# Save outputs
# =============================================================================
with PSY_OUTPUT.open("wb") as handle:
    pickle.dump(psy_data, handle)
with QUANT_OUTPUT.open("wb") as handle:
    pickle.dump(quant_data, handle)
with SLOPES_OUTPUT.open("wb") as handle:
    pickle.dump(slopes_data, handle)
with GAMMA_OUTPUT.open("wb") as handle:
    pickle.dump(gamma_data, handle)

bundle = {
    "model_label": MODEL_LABEL,
    "method_key": METHOD_KEY,
    "source_payload": str(INPUT_PAYLOAD),
    "big_svi_root": str(BIG_SVI_ROOT),
    "mse_param_csv": str(MSE_PARAM_CSV),
    "psy_data": psy_data,
    "quant_data": quant_data,
    "slopes_data": slopes_data,
    "gamma_data": gamma_data,
    "checks": {
        "n_animals": len(animal_keys),
        "n_condition_rows": len(big_condition_df),
        "n_mse_param_rows": len(method_param_df),
        "sd_psychometric_high_ild_model_entries": sd_high_model_entries,
        "sd_flat_quantile_entries_at_abs_ild_16": sd_flat_counts_at_16,
        "quantiles": PAPER_QUANTILES.copy(),
    },
}
with BUNDLE_OUTPUT.open("wb") as handle:
    pickle.dump(bundle, handle)

print(f"Saved psychometric data: {PSY_OUTPUT}")
print(f"Saved RT quantile data: {QUANT_OUTPUT}")
print(f"Saved slope data: {SLOPES_OUTPUT}")
print(f"Saved Gamma data: {GAMMA_OUTPUT}")
print(f"Saved bundle: {BUNDLE_OUTPUT}")
print(f"SD psychometric/slope model entries at |ILD| > 8: {sd_high_model_entries}")
print(f"SD-flat RT quantile model entries at |ILD| = 16: {sd_flat_counts_at_16}")
print(f"Paper quantiles: {PAPER_QUANTILES}")
