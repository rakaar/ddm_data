# %%
"""Replace only LED7/93's canonical proactive VBMC block with the converged refit."""

# %%
# =============================================================================
# Editable paths and reproducibility settings
# =============================================================================
from datetime import datetime
from pathlib import Path
import hashlib
import json
import os
import pickle
import shutil

import numpy as np
from pyvbmc import VariationalPosterior


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

CANONICAL_PKL = (
    REPO_DIR / "aborts_ipl_npl_time_fit_results" / "results_LED7_animal_93.pkl"
)
REFIT_DIR = SCRIPT_DIR / "vbmc_old_truncated_proactive_led7_93_more_evals"
REFIT_VP_PKL = REFIT_DIR / "refreshed_variational_posterior.pkl"
REFIT_SAMPLES_NPZ = REFIT_DIR / "refreshed_posterior_samples.npz"
REFIT_SUMMARY_JSON = REFIT_DIR / "run_summary.json"

REPLACEMENT_DIR = REFIT_DIR / "canonical_replacement"
ORIGINAL_BACKUP_PKL = (
    REPLACEMENT_DIR / "results_LED7_animal_93_before_corrected_proactive_refit.pkl"
)
PROVENANCE_JSON = REPLACEMENT_DIR / "replacement_provenance.json"

EXPECTED_ORIGINAL_SHA256 = (
    "1a38888c20612e9ed54abe8d56f88f02b0a42b9660eb730c923d09243c047860"
)
POSTERIOR_SAMPLES = 1_000_000
RANDOM_SEED = 20260829

ABORT_BLOCK_KEYS = (
    "V_A_samples",
    "theta_A_samples",
    "t_A_aff_samp",
    "message",
    "elbo",
    "elbo_sd",
    "loglike",
)
UNTOUCHED_BLOCKS = (
    "vbmc_vanilla_tied_results",
    "vbmc_norm_tied_results",
    "vbmc_time_vary_norm_tied_results",
)


# %%
# =============================================================================
# Load and validate the exact source artifacts
# =============================================================================
def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def block_sha256(block):
    return hashlib.sha256(
        pickle.dumps(block, protocol=pickle.HIGHEST_PROTOCOL)
    ).hexdigest()


required_paths = (
    CANONICAL_PKL,
    REFIT_VP_PKL,
    REFIT_SAMPLES_NPZ,
    REFIT_SUMMARY_JSON,
)
missing_paths = [str(path) for path in required_paths if not path.is_file()]
if missing_paths:
    raise FileNotFoundError(f"Missing required replacement inputs: {missing_paths}")

current_sha256 = sha256_file(CANONICAL_PKL)
if current_sha256 != EXPECTED_ORIGINAL_SHA256:
    if PROVENANCE_JSON.is_file():
        prior_provenance = json.loads(PROVENANCE_JSON.read_text())
        if current_sha256 == prior_provenance.get("new_canonical_sha256"):
            print("Canonical LED7/93 proactive block is already the converged refit.")
            print(f"Canonical SHA256: {current_sha256}")
            raise SystemExit(0)
    raise RuntimeError(
        "Refusing to replace an unexpected canonical file: "
        f"expected SHA256 {EXPECTED_ORIGINAL_SHA256}, found {current_sha256}."
    )

with CANONICAL_PKL.open("rb") as handle:
    original_combined = pickle.load(handle)

expected_top_level_keys = (
    "vbmc_aborts_results",
    "vbmc_vanilla_tied_results",
    "vbmc_norm_tied_results",
    "vbmc_time_vary_norm_tied_results",
)
if tuple(original_combined) != expected_top_level_keys:
    raise RuntimeError(
        f"Unexpected canonical top-level schema: {tuple(original_combined)}"
    )
if tuple(original_combined["vbmc_aborts_results"]) != ABORT_BLOCK_KEYS:
    raise RuntimeError("Unexpected canonical proactive-block schema.")

with REFIT_SUMMARY_JSON.open() as handle:
    refit_summary = json.load(handle)
if not refit_summary.get("success_flag"):
    raise RuntimeError("The corrected proactive VBMC refit is not marked successful.")
if "stable" not in refit_summary.get("status_message", "").lower():
    raise RuntimeError("The corrected proactive VBMC refit is not marked stable.")

source_samples = np.load(REFIT_SAMPLES_NPZ)
source_sample_means = np.array(
    [
        source_samples["V_A"].mean(),
        source_samples["theta_A"].mean(),
        source_samples["t_A_aff"].mean(),
    ],
    dtype=float,
)


# %%
# =============================================================================
# Draw a canonical-size posterior sample and construct the replacement block
# =============================================================================
np.random.seed(RANDOM_SEED)
refit_vp = VariationalPosterior.load(REFIT_VP_PKL)
replacement_samples = np.asarray(refit_vp.sample(POSTERIOR_SAMPLES)[0], dtype=float)
if replacement_samples.shape != (POSTERIOR_SAMPLES, 3):
    raise RuntimeError(
        f"Unexpected replacement posterior shape: {replacement_samples.shape}."
    )
if not np.isfinite(replacement_samples).all():
    raise RuntimeError("Replacement posterior contains non-finite samples.")

replacement_sample_means = replacement_samples.mean(axis=0)
mean_tolerances = np.array([0.01, 0.01, 0.001], dtype=float)
if not np.all(np.abs(replacement_sample_means - source_sample_means) < mean_tolerances):
    raise RuntimeError(
        "Canonical-size posterior means differ unexpectedly from the saved refit "
        f"sample means: source={source_sample_means}, replacement={replacement_sample_means}."
    )

replacement_abort_block = {
    "V_A_samples": replacement_samples[:, 0].copy(),
    "theta_A_samples": replacement_samples[:, 1].copy(),
    "t_A_aff_samp": replacement_samples[:, 2].copy(),
    "message": str(refit_summary["status_message"]),
    "elbo": np.float64(refit_summary["elbo"]),
    "elbo_sd": np.float64(refit_summary["elbo_sd"]),
    "loglike": np.float64(refit_summary["refreshed_mean_loglike_on_refit_data"]),
}

old_abort_block = original_combined["vbmc_aborts_results"]
old_sample_means = np.array(
    [
        np.mean(old_abort_block["V_A_samples"]),
        np.mean(old_abort_block["theta_A_samples"]),
        np.mean(old_abort_block["t_A_aff_samp"]),
    ],
    dtype=float,
)
untouched_hashes_before = {
    name: block_sha256(original_combined[name]) for name in UNTOUCHED_BLOCKS
}


# %%
# =============================================================================
# Preserve the original and atomically replace the canonical pickle
# =============================================================================
REPLACEMENT_DIR.mkdir(parents=True, exist_ok=True)
if ORIGINAL_BACKUP_PKL.exists():
    if sha256_file(ORIGINAL_BACKUP_PKL) != EXPECTED_ORIGINAL_SHA256:
        raise RuntimeError(f"Existing backup has an unexpected hash: {ORIGINAL_BACKUP_PKL}")
else:
    shutil.copy2(CANONICAL_PKL, ORIGINAL_BACKUP_PKL)
if sha256_file(ORIGINAL_BACKUP_PKL) != EXPECTED_ORIGINAL_SHA256:
    raise RuntimeError("The preserved original canonical pickle failed SHA256 validation.")

updated_combined = original_combined.copy()
updated_combined["vbmc_aborts_results"] = replacement_abort_block

temporary_pkl = CANONICAL_PKL.with_suffix(CANONICAL_PKL.suffix + ".tmp")
try:
    with temporary_pkl.open("wb") as handle:
        pickle.dump(updated_combined, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary_pkl, CANONICAL_PKL.stat().st_mode)
    os.replace(temporary_pkl, CANONICAL_PKL)
finally:
    if temporary_pkl.exists():
        temporary_pkl.unlink()


# %%
# =============================================================================
# Reload and prove that only the proactive block changed
# =============================================================================
with CANONICAL_PKL.open("rb") as handle:
    verified_combined = pickle.load(handle)

if tuple(verified_combined) != expected_top_level_keys:
    raise RuntimeError("Top-level canonical schema changed during replacement.")
if tuple(verified_combined["vbmc_aborts_results"]) != ABORT_BLOCK_KEYS:
    raise RuntimeError("Proactive-block schema changed during replacement.")

verified_abort = verified_combined["vbmc_aborts_results"]
for key in ABORT_BLOCK_KEYS:
    expected_value = replacement_abort_block[key]
    observed_value = verified_abort[key]
    if isinstance(expected_value, np.ndarray):
        if not np.array_equal(observed_value, expected_value):
            raise RuntimeError(f"Replacement array failed exact validation: {key}")
    elif observed_value != expected_value:
        raise RuntimeError(f"Replacement metadata failed exact validation: {key}")

untouched_hashes_after = {
    name: block_sha256(verified_combined[name]) for name in UNTOUCHED_BLOCKS
}
if untouched_hashes_after != untouched_hashes_before:
    raise RuntimeError("At least one downstream TIED result block changed.")

new_sha256 = sha256_file(CANONICAL_PKL)
provenance = {
    "updated_at_local": datetime.now().astimezone().isoformat(),
    "canonical_pkl": str(CANONICAL_PKL.resolve()),
    "original_backup_pkl": str(ORIGINAL_BACKUP_PKL.resolve()),
    "source_variational_posterior": str(REFIT_VP_PKL.resolve()),
    "source_summary_json": str(REFIT_SUMMARY_JSON.resolve()),
    "source_posterior_samples": str(REFIT_SAMPLES_NPZ.resolve()),
    "replacement_script": str(Path(__file__).resolve()),
    "random_seed": RANDOM_SEED,
    "posterior_samples": POSTERIOR_SAMPLES,
    "original_canonical_sha256": EXPECTED_ORIGINAL_SHA256,
    "new_canonical_sha256": new_sha256,
    "old_posterior_means": dict(
        zip(("V_A", "theta_A", "t_A_aff_s"), old_sample_means.tolist())
    ),
    "new_posterior_means": dict(
        zip(("V_A", "theta_A", "t_A_aff_s"), replacement_sample_means.tolist())
    ),
    "refit_status_message": refit_summary["status_message"],
    "refit_convergence_status": refit_summary["convergence_status"],
    "refit_elbo": refit_summary["elbo"],
    "refit_elbo_sd": refit_summary["elbo_sd"],
    "refit_mean_loglike": refit_summary["refreshed_mean_loglike_on_refit_data"],
    "untouched_block_sha256": untouched_hashes_after,
}
PROVENANCE_JSON.write_text(json.dumps(provenance, indent=2) + "\n")

print("Canonical LED7/93 proactive VBMC block replaced successfully.")
print(f"Old SHA256: {EXPECTED_ORIGINAL_SHA256}")
print(f"New SHA256: {new_sha256}")
print(f"Old means [V_A, theta_A, t_A_aff_s]: {old_sample_means}")
print(f"New means [V_A, theta_A, t_A_aff_s]: {replacement_sample_means}")
print(f"Preserved original: {ORIGINAL_BACKUP_PKL}")
print(f"Replacement provenance: {PROVENANCE_JSON}")
print("Verified unchanged downstream blocks:")
for block_name, block_hash in untouched_hashes_after.items():
    print(f"  {block_name}: {block_hash}")
