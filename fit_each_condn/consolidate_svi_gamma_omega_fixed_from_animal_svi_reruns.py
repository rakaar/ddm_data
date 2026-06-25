# %%
"""
Consolidate condition-by-condition NumPyro SVI gamma/omega fits.

The full all-animal run lives in:

    svi_gamma_omega_fixed_from_animal_svi_condition_delay_results/all_observed/

The extra 30k rerun for originally non-stable conditions lives in:

    svi_gamma_omega_fixed_from_animal_svi_condition_delay_results/nonstable_extra_steps_30k/

This script creates a new consolidated folder containing all 30 animals, with
only the rerun conditions replacing the original all_observed rows/samples.
"""

# %%
# =============================================================================
# Editable parameters
# =============================================================================
from pathlib import Path
import json
import os
import pickle
import re
import shutil

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / "svi_gamma_omega_fixed_from_animal_svi_condition_delay_results"

BASE_LABEL = os.environ.get("COND_SVI_CONSOLIDATE_BASE_LABEL", "all_observed")
RERUN_LABELS = [
    label.strip()
    for label in os.environ.get(
        "COND_SVI_CONSOLIDATE_RERUN_LABELS",
        "nonstable_extra_steps_30k",
    ).split(",")
    if label.strip()
]
CONSOLIDATED_LABEL = os.environ.get(
    "COND_SVI_CONSOLIDATE_OUTPUT_LABEL",
    "all_observed_with_30k_reruns",
)
OVERWRITE_OUTPUT = os.environ.get("COND_SVI_CONSOLIDATE_OVERWRITE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}

BASE_DIR = OUTPUT_ROOT / BASE_LABEL
RERUN_DIRS = [OUTPUT_ROOT / label for label in RERUN_LABELS]
CONSOLIDATED_DIR = OUTPUT_ROOT / CONSOLIDATED_LABEL

FIT_SCRIPT = SCRIPT_DIR / "fit_svi_cond_by_cond_gamma_omega_fixed_from_animal_svi.py"
THIS_SCRIPT = Path(__file__).resolve()

BATCH_ORDER = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]


# %%
# =============================================================================
# Helpers
# =============================================================================
def animal_prefix(batch_name, animal):
    return f"{batch_name}_{int(animal)}_gamma_omega_fixed_from_animal_svi"


def output_paths(folder, batch_name, animal):
    prefix = animal_prefix(batch_name, animal)
    return {
        "summary": folder / f"{prefix}_summary.csv",
        "posterior_summary": folder / f"{prefix}_posterior_summary.csv",
        "loss": folder / f"{prefix}_loss_by_condition.csv",
        "convergence": folder / f"{prefix}_convergence_checks.csv",
        "samples": folder / f"{prefix}_posterior_samples.npz",
        "bundle": folder / f"{prefix}_fit_bundle.pkl",
    }


def condition_key(abl, ild):
    return int(abl), int(ild)


def safe_condition_key(condition_index, abl, ild):
    sign = "m" if int(ild) < 0 else "p"
    return f"cond{int(condition_index):02d}_ABL{int(abl)}_ILD{sign}{abs(int(ild))}"


def replace_condition_rows(base_df, rerun_df, key_cols, original_condition_index_by_key):
    merged_df = base_df.copy()
    rerun_fixed = rerun_df.copy()
    for idx, row in rerun_fixed.iterrows():
        key = condition_key(row["ABL"], row["ILD"])
        if key not in original_condition_index_by_key:
            raise RuntimeError(f"Rerun condition {key} was not found in the base animal output.")
        original_condition_index, original_condition_id = original_condition_index_by_key[key]
        rerun_fixed.loc[idx, "condition_index"] = original_condition_index
        if "condition_id_from_animal_svi" in rerun_fixed.columns:
            rerun_fixed.loc[idx, "condition_id_from_animal_svi"] = original_condition_id

    base_index = pd.MultiIndex.from_frame(merged_df[key_cols])
    rerun_index = pd.MultiIndex.from_frame(rerun_fixed[key_cols])
    merged_df = merged_df.loc[~base_index.isin(rerun_index)].copy()
    merged_df = pd.concat([merged_df, rerun_fixed], ignore_index=True)
    sort_cols = [col for col in ["condition_index", "parameter", "step", "chunk"] if col in merged_df.columns]
    if sort_cols:
        merged_df = merged_df.sort_values(sort_cols).reset_index(drop=True)
    return merged_df


def load_rerun_animal_files(batch_name, animal):
    rerun_payloads = []
    for rerun_label, rerun_dir in zip(RERUN_LABELS, RERUN_DIRS):
        paths = output_paths(rerun_dir, batch_name, animal)
        if paths["summary"].exists():
            required_paths = list(paths.values())
            missing = [path for path in required_paths if not path.exists()]
            if missing:
                raise FileNotFoundError(f"Partial rerun output for {batch_name}/{animal}: {missing}")
            rerun_payloads.append((rerun_label, paths))
    return rerun_payloads


def read_csv_if_exists(path):
    if path.exists() and path.stat().st_size > 1:
        return pd.read_csv(path)
    return pd.DataFrame()


# %%
# =============================================================================
# Preflight
# =============================================================================
if not BASE_DIR.exists():
    raise FileNotFoundError(BASE_DIR)
for rerun_dir in RERUN_DIRS:
    if not rerun_dir.exists():
        raise FileNotFoundError(rerun_dir)

if CONSOLIDATED_DIR.exists():
    if not OVERWRITE_OUTPUT:
        raise RuntimeError(
            f"{CONSOLIDATED_DIR} already exists. Set COND_SVI_CONSOLIDATE_OVERWRITE=1 to replace it."
        )
    shutil.rmtree(CONSOLIDATED_DIR)
CONSOLIDATED_DIR.mkdir(parents=True, exist_ok=False)

base_aggregate_summary = pd.read_csv(BASE_DIR / "all_animals_gamma_omega_fixed_from_animal_svi_summary.csv")
base_aggregate_summary["animal"] = base_aggregate_summary["animal"].astype(int)
animal_pairs = (
    base_aggregate_summary[["batch_name", "animal"]]
    .drop_duplicates()
    .assign(batch_order=lambda df: df["batch_name"].map({batch: i for i, batch in enumerate(BATCH_ORDER)}))
    .sort_values(["batch_order", "batch_name", "animal"])
    [["batch_name", "animal"]]
    .itertuples(index=False, name=None)
)
animal_pairs = list(animal_pairs)
print(f"Base folder: {BASE_DIR}")
print(f"Rerun folders: {[str(path) for path in RERUN_DIRS]}")
print(f"Consolidated folder: {CONSOLIDATED_DIR}")
print(f"Found {len(animal_pairs)} animals in base aggregate.")


# %%
# =============================================================================
# Merge per-animal files
# =============================================================================
all_summary_frames = []
all_posterior_summary_frames = []
replacement_rows = []

for batch_name, animal in animal_pairs:
    base_paths = output_paths(BASE_DIR, batch_name, animal)
    output_animal_paths = output_paths(CONSOLIDATED_DIR, batch_name, animal)
    for path in base_paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    base_summary = pd.read_csv(base_paths["summary"])
    base_posterior_summary = pd.read_csv(base_paths["posterior_summary"])
    base_loss = read_csv_if_exists(base_paths["loss"])
    base_convergence = read_csv_if_exists(base_paths["convergence"])
    original_condition_index_by_key = {
        condition_key(row.ABL, row.ILD): (
            int(row.condition_index),
            int(row.condition_id_from_animal_svi),
        )
        for row in base_summary.itertuples(index=False)
    }

    with np.load(base_paths["samples"]) as base_npz:
        merged_samples = {key: base_npz[key] for key in base_npz.files}
    with open(base_paths["bundle"], "rb") as f:
        base_bundle = pickle.load(f)

    merged_summary = base_summary.copy()
    merged_posterior_summary = base_posterior_summary.copy()
    merged_loss = base_loss.copy()
    merged_convergence = base_convergence.copy()
    replaced_for_animal = []

    for rerun_label, rerun_paths in load_rerun_animal_files(batch_name, animal):
        rerun_summary = pd.read_csv(rerun_paths["summary"])
        rerun_posterior_summary = pd.read_csv(rerun_paths["posterior_summary"])
        rerun_loss = read_csv_if_exists(rerun_paths["loss"])
        rerun_convergence = read_csv_if_exists(rerun_paths["convergence"])

        rerun_conditions = [condition_key(row.ABL, row.ILD) for row in rerun_summary.itertuples(index=False)]
        replaced_for_animal.extend((rerun_label, abl, ild) for abl, ild in rerun_conditions)

        merged_summary = replace_condition_rows(
            merged_summary,
            rerun_summary,
            ["batch_name", "animal", "ABL", "ILD"],
            original_condition_index_by_key,
        )
        merged_posterior_summary = replace_condition_rows(
            merged_posterior_summary,
            rerun_posterior_summary,
            ["batch_name", "animal", "ABL", "ILD", "parameter"],
            original_condition_index_by_key,
        )
        if not rerun_loss.empty:
            merged_loss = replace_condition_rows(
                merged_loss,
                rerun_loss,
                ["batch_name", "animal", "ABL", "ILD", "step"],
                original_condition_index_by_key,
            )
        if not rerun_convergence.empty:
            merged_convergence = replace_condition_rows(
                merged_convergence,
                rerun_convergence,
                ["batch_name", "animal", "ABL", "ILD", "chunk"],
                original_condition_index_by_key,
            )

        with np.load(rerun_paths["samples"]) as rerun_npz:
            for row in rerun_summary.itertuples(index=False):
                original_condition_index, _ = original_condition_index_by_key[condition_key(row.ABL, row.ILD)]
                source_prefix = safe_condition_key(row.condition_index, row.ABL, row.ILD)
                target_prefix = safe_condition_key(original_condition_index, row.ABL, row.ILD)
                for param_name in ["gamma", "omega"]:
                    source_key = f"{source_prefix}_{param_name}"
                    target_key = f"{target_prefix}_{param_name}"
                    if source_key not in rerun_npz.files:
                        raise KeyError(f"{source_key} missing from {rerun_paths['samples']}")
                    merged_samples[target_key] = rerun_npz[source_key]

    expected_n_conditions = len(base_summary)
    if len(merged_summary) != expected_n_conditions:
        raise RuntimeError(f"{batch_name}/{animal}: merged summary row count changed.")
    if len(merged_posterior_summary) != 2 * expected_n_conditions:
        raise RuntimeError(f"{batch_name}/{animal}: merged posterior summary row count changed.")
    if len(merged_samples) != 2 * expected_n_conditions:
        raise RuntimeError(f"{batch_name}/{animal}: merged NPZ key count changed.")

    # Point summary/cache consumers to the consolidated sample file.
    merged_summary["source_script"] = str(FIT_SCRIPT.resolve())

    merged_summary.to_csv(output_animal_paths["summary"], index=False)
    merged_posterior_summary.to_csv(output_animal_paths["posterior_summary"], index=False)
    merged_loss.to_csv(output_animal_paths["loss"], index=False)
    merged_convergence.to_csv(output_animal_paths["convergence"], index=False)
    np.savez(output_animal_paths["samples"], **merged_samples)

    base_bundle["run_label"] = CONSOLIDATED_LABEL
    base_bundle["summary_rows"] = merged_summary.to_dict("records")
    base_bundle["posterior_summary_rows"] = merged_posterior_summary.to_dict("records")
    base_bundle["consolidation"] = {
        "base_label": BASE_LABEL,
        "rerun_labels": RERUN_LABELS,
        "consolidated_label": CONSOLIDATED_LABEL,
        "consolidation_script": str(THIS_SCRIPT),
        "replaced_conditions": [
            {"rerun_label": label, "ABL": int(abl), "ILD": int(ild)}
            for label, abl, ild in replaced_for_animal
        ],
    }
    with open(output_animal_paths["bundle"], "wb") as f:
        pickle.dump(base_bundle, f)

    for rerun_label, abl, ild in replaced_for_animal:
        replacement_rows.append(
            {
                "batch_name": batch_name,
                "animal": animal,
                "ABL": int(abl),
                "ILD": int(ild),
                "rerun_label": rerun_label,
                "output_samples_npz": str(output_animal_paths["samples"]),
            }
        )

    all_summary_frames.append(merged_summary)
    all_posterior_summary_frames.append(merged_posterior_summary)
    print(
        f"{batch_name}/{animal}: {expected_n_conditions} conditions, "
        f"{len(replaced_for_animal)} replaced from reruns."
    )


# %%
# =============================================================================
# Aggregate outputs
# =============================================================================
all_summary = pd.concat(all_summary_frames, ignore_index=True)
all_posterior_summary = pd.concat(all_posterior_summary_frames, ignore_index=True)
batch_order_map = {batch: i for i, batch in enumerate(BATCH_ORDER)}
all_summary = (
    all_summary.assign(batch_order=all_summary["batch_name"].map(batch_order_map))
    .sort_values(["batch_order", "batch_name", "animal", "condition_index"])
    .drop(columns=["batch_order"])
    .reset_index(drop=True)
)
all_posterior_summary = (
    all_posterior_summary.assign(batch_order=all_posterior_summary["batch_name"].map(batch_order_map))
    .sort_values(["batch_order", "batch_name", "animal", "condition_index", "parameter"])
    .drop(columns=["batch_order"])
    .reset_index(drop=True)
)

condition_cache_rows = []
for row in all_summary.itertuples(index=False):
    samples_path = output_paths(CONSOLIDATED_DIR, row.batch_name, int(row.animal))["samples"]
    condition_cache_rows.append(
        {
            "batch_name": row.batch_name,
            "animal": int(row.animal),
            "ABL": int(row.ABL),
            "ILD": int(row.ILD),
            "condition_gamma": float(row.svi_gamma_mean),
            "condition_omega": float(row.svi_omega_mean),
            "fixed_t_E_aff_s": float(row.fixed_t_E_aff_s),
            "fixed_t_E_aff_ms": float(row.fixed_t_E_aff_ms),
            "source_npz": str(samples_path),
            "source_script": str(FIT_SCRIPT.resolve()),
        }
    )
condition_cache = pd.DataFrame(condition_cache_rows)
replacement_table = pd.DataFrame(replacement_rows)

all_summary.to_csv(
    CONSOLIDATED_DIR / "all_animals_gamma_omega_fixed_from_animal_svi_summary.csv",
    index=False,
)
all_posterior_summary.to_csv(
    CONSOLIDATED_DIR / "all_animals_gamma_omega_fixed_from_animal_svi_posterior_summary.csv",
    index=False,
)
condition_cache.to_csv(CONSOLIDATED_DIR / "condition_gamma_omega_extraction_cache.csv", index=False)
pd.DataFrame().to_csv(CONSOLIDATED_DIR / "failures.csv", index=False)
replacement_table.to_csv(CONSOLIDATED_DIR / "rerun_replaced_conditions.csv", index=False)

manifest = {
    "base_label": BASE_LABEL,
    "rerun_labels": RERUN_LABELS,
    "consolidated_label": CONSOLIDATED_LABEL,
    "base_dir": str(BASE_DIR),
    "rerun_dirs": [str(path) for path in RERUN_DIRS],
    "consolidated_dir": str(CONSOLIDATED_DIR),
    "n_animals": int(all_summary[["batch_name", "animal"]].drop_duplicates().shape[0]),
    "n_conditions": int(len(all_summary)),
    "n_replaced_conditions": int(len(replacement_table)),
    "stop_reason_counts": {
        key: int(value)
        for key, value in all_summary["stop_reason"].value_counts().sort_index().items()
    },
    "script": str(THIS_SCRIPT),
}
(CONSOLIDATED_DIR / "consolidation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


# %%
# =============================================================================
# Validation summary
# =============================================================================
duplicates = all_summary.duplicated(["batch_name", "animal", "ABL", "ILD"]).sum()
posterior_duplicates = all_posterior_summary.duplicated(
    ["batch_name", "animal", "ABL", "ILD", "parameter"]
).sum()
if duplicates:
    raise RuntimeError(f"Found {duplicates} duplicated aggregate condition rows.")
if posterior_duplicates:
    raise RuntimeError(f"Found {posterior_duplicates} duplicated posterior summary rows.")
if len(all_summary) != len(base_aggregate_summary):
    raise RuntimeError("Consolidated condition count does not match base condition count.")
if len(replacement_table) == 0:
    raise RuntimeError("No rerun rows were found; consolidation did not replace anything.")

print("\nConsolidated outputs written:")
for path in [
    CONSOLIDATED_DIR / "all_animals_gamma_omega_fixed_from_animal_svi_summary.csv",
    CONSOLIDATED_DIR / "all_animals_gamma_omega_fixed_from_animal_svi_posterior_summary.csv",
    CONSOLIDATED_DIR / "condition_gamma_omega_extraction_cache.csv",
    CONSOLIDATED_DIR / "rerun_replaced_conditions.csv",
    CONSOLIDATED_DIR / "consolidation_manifest.json",
]:
    print(f"  {path}")

print("\nValidation:")
print(f"  animals: {all_summary[['batch_name', 'animal']].drop_duplicates().shape[0]}")
print(f"  conditions: {len(all_summary)}")
print(f"  posterior rows: {len(all_posterior_summary)}")
print(f"  replaced rerun conditions: {len(replacement_table)}")
print("  stop reasons:")
print(all_summary["stop_reason"].value_counts().to_string())
