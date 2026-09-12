# %%
"""Overlay pooled LED7 s9 timing on the relaxed matched s7 samples."""

# %%
# =============================================================================
# Editable parameters
# =============================================================================
from pathlib import Path
import hashlib
import json

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

INPUT_MAT_PATH = REPO_DIR / "raw_data" / "outMatrix_LED7_latest.mat"
TABLE_NAME = "totalout_stGtACRII"
FIT_ROOT = (
    SCRIPT_DIR
    / "numpyro_svi_led7_s7_relaxed_s9_timing_matched_"
    "no_trunc_exp_lapse_outputs"
)
MATCHED_FIT_DIR = FIT_ROOT / "s7_matched_to_s9"
MANIFEST_PATH = MATCHED_FIT_DIR / "source_row_manifest.csv"
MATCHING_AUDIT_PATH = MATCHED_FIT_DIR / "matching_audit.json"
OUTPUT_FIGURE_PATH = SCRIPT_DIR / "led7_s7_relaxed_s9_timing_match_overlay_1x2.png"
OUTPUT_AUDIT_PATH = SCRIPT_DIR / "led7_s7_relaxed_s9_timing_match_overlay_audit.csv"

TRAINING_LEVEL = 16
REFERENCE_SESSION_TYPE = 9
ALLOWED_REPEAT_TRIALS = {0, 2}
LED_TRIAL_VALUES = (0, 1)
EXPECTED_REFERENCE_ROWS = 100_788
EXPECTED_SAMPLE_SIZES = {0: 13_750, 1: 7_000}
EXPECTED_SELECTION_HASHES = {
    0: "0524b9c51119f79fb90f2160972a7e85b56d9cf40928b237ab433e637d7f0371",
    1: "8e40159aec2cfa01cb4eb336a9e18a64a9222af0236ebd89247aabc66b4b4077",
}
MAX_KS_DISTANCE = 0.030
MAX_WASSERSTEIN_MS = 10.0

BIN_WIDTH_S = 0.020
INTENDED_BINS = np.arange(0.2, 2.2 + BIN_WIDTH_S, BIN_WIDTH_S)
ONSET_BINS = np.arange(0.1, 1.1 + BIN_WIDTH_S, BIN_WIDTH_S)
OUTPUT_DPI = 250
SHOW_PLOT = False


# %%
# =============================================================================
# Imports and small reusable calculations
# =============================================================================
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

try:
    from matio import load_from_mat
except ImportError as exc:
    raise ImportError(
        "Run with: env PYTHONPATH=/tmp/codex_mat_io .venv/bin/python "
        "fitting_aborts/plot_led7_s7_relaxed_s9_timing_match_overlay.py"
    ) from exc


def density_histogram(values, bins):
    values = np.asarray(values, dtype=float)
    counts, _ = np.histogram(values, bins=bins)
    if int(counts.sum()) != len(values):
        raise RuntimeError(
            f"Histogram retained {counts.sum():,} of {len(values):,} values."
        )
    density = counts / (len(values) * np.diff(bins))
    if not np.isclose(np.sum(density * np.diff(bins)), 1.0, atol=1e-12):
        raise RuntimeError("Density histogram does not integrate to one.")
    return 0.5 * (bins[:-1] + bins[1:]), density


def ordered_index_hash(frame):
    ordered = frame.sort_values("selection_order")["source_row_index"].astype(int)
    payload = "\n".join(str(value) for value in ordered).encode()
    return hashlib.sha256(payload).hexdigest()


# %%
# =============================================================================
# Load the exact relaxed s7 samples and pooled s9 reference
# =============================================================================
for path in (INPUT_MAT_PATH, MANIFEST_PATH, MATCHING_AUDIT_PATH):
    if not path.exists():
        raise FileNotFoundError(path)

matching_audit = json.loads(MATCHING_AUDIT_PATH.read_text())
if matching_audit.get("matching_profile") != "relaxed_unequal":
    raise RuntimeError("Manifest is not from the relaxed unequal matching profile.")
if matching_audit.get("match_sample_sizes_by_led") != {"0": 13_750, "1": 7_000}:
    raise RuntimeError("Relaxed matching sample sizes changed.")

matched = pd.read_csv(MANIFEST_PATH)
required_manifest_columns = {
    "source_row_index",
    "selection_order",
    "LED_trial",
    "intended_fix",
    "effective_scheduled_onset",
}
missing_manifest_columns = sorted(required_manifest_columns - set(matched.columns))
if missing_manifest_columns:
    raise RuntimeError(f"Matched manifest is missing {missing_manifest_columns}.")
if not matched["source_row_index"].is_unique:
    raise RuntimeError("Matched manifest repeats source rows.")

mat_contents = load_from_mat(INPUT_MAT_PATH, variable_names=[TABLE_NAME])
if TABLE_NAME not in mat_contents:
    raise KeyError(f"{TABLE_NAME!r} is absent from {INPUT_MAT_PATH}.")
raw = mat_contents[TABLE_NAME]
if not isinstance(raw, pd.DataFrame):
    raise TypeError(f"Expected a DataFrame, found {type(raw).__name__}.")

required_raw_columns = {
    "training_level",
    "session_type",
    "repeat_trial",
    "LED_trial",
    "intended_fix",
    "LED_onset_time",
}
missing_raw_columns = sorted(required_raw_columns - set(raw.columns))
if missing_raw_columns:
    raise RuntimeError(f"Source table is missing {missing_raw_columns}.")

# Keep the original filter order explicit.
reference = raw.loc[raw["training_level"].eq(TRAINING_LEVEL)].copy()
reference = reference.loc[
    reference["session_type"].eq(REFERENCE_SESSION_TYPE)
].copy()
reference = reference.loc[
    reference["repeat_trial"].isin(ALLOWED_REPEAT_TRIALS)
    | reference["repeat_trial"].isna()
].copy()
reference = reference.loc[
    reference["LED_trial"].isin(LED_TRIAL_VALUES)
].copy()
reference["effective_scheduled_onset"] = reference["LED_onset_time"]

if len(reference) != EXPECTED_REFERENCE_ROWS:
    raise RuntimeError(
        f"Expected {EXPECTED_REFERENCE_ROWS:,} s9 rows, found {len(reference):,}."
    )
if not np.isfinite(
    reference[["intended_fix", "effective_scheduled_onset"]]
).all(axis=None):
    raise RuntimeError("Reference timing contains non-finite values.")


# %%
# =============================================================================
# Validate unbinned distances and prepare unit-area curves
# =============================================================================
VARIABLE_SPECS = (
    {
        "column": "intended_fix",
        "title": "intended_fix",
        "xlabel": "Time (s)",
        "bins": INTENDED_BINS,
    },
    {
        "column": "effective_scheduled_onset",
        "title": "LED onset from fixation",
        "xlabel": "Time (s)",
        "bins": ONSET_BINS,
    },
)

audit_rows = []
plot_payload = {}
for variable_spec in VARIABLE_SPECS:
    column = variable_spec["column"]
    reference_centers, reference_density = density_histogram(
        reference[column], variable_spec["bins"]
    )
    plot_payload[column] = {
        "reference_centers": reference_centers,
        "reference_density": reference_density,
        "groups": {},
    }

    for led_trial in LED_TRIAL_VALUES:
        group = matched.loc[matched["LED_trial"].eq(led_trial)].copy()
        expected_n = EXPECTED_SAMPLE_SIZES[led_trial]
        if len(group) != expected_n:
            raise RuntimeError(
                f"LED {led_trial}: expected {expected_n:,} rows, found {len(group):,}."
            )
        if ordered_index_hash(group) != EXPECTED_SELECTION_HASHES[led_trial]:
            raise RuntimeError(f"LED {led_trial}: selected-row identity changed.")

        ks = float(ks_2samp(group[column], reference[column], method="asymp").statistic)
        wasserstein_ms = float(
            1000.0 * wasserstein_distance(group[column], reference[column])
        )
        if ks > MAX_KS_DISTANCE or wasserstein_ms > MAX_WASSERSTEIN_MS:
            raise RuntimeError(
                f"LED {led_trial} {column}: matching threshold failed."
            )
        centers, density = density_histogram(group[column], variable_spec["bins"])
        plot_payload[column]["groups"][led_trial] = {
            "centers": centers,
            "density": density,
            "ks": ks,
            "wasserstein_ms": wasserstein_ms,
            "n": len(group),
        }
        audit_rows.append(
            {
                "variable": column,
                "LED_trial": led_trial,
                "sample_n": len(group),
                "reference_n": len(reference),
                "KS_distance": ks,
                "Wasserstein_ms": wasserstein_ms,
            }
        )

audit = pd.DataFrame(audit_rows)
audit.to_csv(OUTPUT_AUDIT_PATH, index=False)


# %%
# =============================================================================
# Draw the two timing overlays
# =============================================================================
colors = {0: "tab:blue", 1: "tab:red"}
labels = {0: "Sampled s7 LED OFF", 1: "Sampled s7 LED ON"}

fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1))
for ax, variable_spec in zip(axes, VARIABLE_SPECS):
    column = variable_spec["column"]
    payload = plot_payload[column]
    ax.step(
        payload["reference_centers"],
        payload["reference_density"],
        where="mid",
        color="black",
        linestyle="--",
        linewidth=1.8,
        label=f"Session type 9 (n = {len(reference):,})",
        zorder=4,
    )
    for led_trial in LED_TRIAL_VALUES:
        group = payload["groups"][led_trial]
        ax.step(
            group["centers"],
            group["density"],
            where="mid",
            color=colors[led_trial],
            linewidth=1.35,
            label=f"{labels[led_trial]} (n = {group['n']:,})",
            zorder=2,
        )
    ax.set_title(variable_spec["title"])
    ax.set_xlabel(variable_spec["xlabel"])
    ax.set_ylabel("Density")
    ax.set_xlim(variable_spec["bins"][[0, -1]])
    ax.grid(axis="y", color="0.88", linewidth=0.55)
    ax.spines[["top", "right"]].set_visible(False)

axes[1].legend(frameon=False, fontsize=8.3)
fig.suptitle("LED7 relaxed timing match: sampled session type 7 and session type 9")
fig.tight_layout()
fig.savefig(OUTPUT_FIGURE_PATH, dpi=OUTPUT_DPI, bbox_inches="tight")
if SHOW_PLOT:
    plt.show()
else:
    plt.close(fig)

print(audit.to_string(index=False))
print(f"\nFigure: {OUTPUT_FIGURE_PATH.resolve()}")
print(f"Audit: {OUTPUT_AUDIT_PATH.resolve()}")
