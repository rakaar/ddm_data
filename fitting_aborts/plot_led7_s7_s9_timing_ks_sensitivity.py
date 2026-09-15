# %%
"""Compare LED7 session-type-7 timing matches at several retained row counts."""

# %%
# Editable parameters
from pathlib import Path
import hashlib
import json

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
INPUT_MAT_PATH = REPO_DIR / "raw_data" / "outMatrix_LED7_latest.mat"
TABLE_NAME = "totalout_stGtACRII"

STRICT_FIT_DIR = (
    SCRIPT_DIR
    / "numpyro_svi_led7_s7_all_vs_s9_timing_matched_no_trunc_exp_lapse_outputs"
    / "s7_matched_to_s9"
)
RELAXED_FIT_DIR = (
    SCRIPT_DIR
    / "numpyro_svi_led7_s7_relaxed_s9_timing_matched_no_trunc_exp_lapse_outputs"
    / "s7_matched_to_s9"
)
OUTPUT_FIGURE_PATH = SCRIPT_DIR / "led7_s7_s9_timing_ks_sensitivity_2x5.png"
OUTPUT_AUDIT_PATH = SCRIPT_DIR / "led7_s7_s9_timing_ks_sensitivity_audit.csv"

TRAINING_LEVEL = 16
SESSION_TYPES = (7, 9)
REPEAT_TRIAL_VALUES = {0, 2}
LED_TRIAL_VALUES = (0, 1)
EXPECTED_S7_COUNTS = {0: 92_057, 1: 46_383}
EXPECTED_S9_ROWS = 100_788

STAGES = (
    ("All s7", EXPECTED_S7_COUNTS, "all"),
    ("30k / 15k", {0: 30_000, 1: 15_000}, "new"),
    ("20k / 10k", {0: 20_000, 1: 10_000}, "new"),
    ("Relaxed", {0: 13_750, 1: 7_000}, "relaxed"),
    ("Strict", {0: 5_500, 1: 5_500}, "strict"),
)
SAVED_SELECTION_HASHES = {
    "relaxed": {
        0: "0524b9c51119f79fb90f2160972a7e85b56d9cf40928b237ab433e637d7f0371",
        1: "8e40159aec2cfa01cb4eb336a9e18a64a9222af0236ebd89247aabc66b4b4077",
    },
    "strict": {
        0: "475dbc17f6230a311756c1f9693cf295358f3d323d80710e5ed766a3ecf9592d",
        1: "57eec5c5f2de1c8f195a62c9e729138dbaf8fc5c1bece572be8a96f1ffaa908b",
    },
}

MATCH_RANDOM_SEED = 20_260_909
MATCHING_COLUMNS = ["intended_fix", "effective_scheduled_onset"]
BIN_WIDTH_S = 0.020
MATCH_INTENDED_BINS = np.arange(200, 2200 + 20, 20) / 1000
MATCH_ONSET_BINS = np.arange(100, 1100 + 20, 20) / 1000
DISPLAY_BINS = np.arange(0, 2200 + 20, 20) / 1000
OUTPUT_DPI = 250
SHOW_PLOT = False


# %%
# Imports and repeated calculations
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from led7_scheduled_timing_matching_utils import (
    largest_remainder_counts,
    select_rows_matching_two_margins,
)

try:
    from matio import load_from_mat
except ImportError as exc:
    raise ImportError(
        "Run with: env PYTHONPATH=/tmp/codex_mat_io "
        ".venv/bin/python fitting_aborts/plot_led7_s7_s9_timing_ks_sensitivity.py"
    ) from exc


def density_histogram(values, bins):
    values = np.asarray(values, dtype=float)
    counts, _ = np.histogram(values, bins=bins)
    if int(counts.sum()) != len(values):
        raise RuntimeError("The displayed timing range omitted source rows.")
    density = counts / (len(values) * np.diff(bins))
    area = float(np.sum(density * np.diff(bins)))
    if not np.isclose(area, 1.0, atol=1e-12):
        raise RuntimeError("Timing histogram does not have unit area.")
    return density, area


# %%
# Load and filter source timing before any row selection
if not INPUT_MAT_PATH.exists():
    raise FileNotFoundError(INPUT_MAT_PATH)

mat_contents = load_from_mat(INPUT_MAT_PATH, variable_names=[TABLE_NAME])
if TABLE_NAME not in mat_contents:
    raise KeyError(f"{TABLE_NAME!r} is absent from {INPUT_MAT_PATH}.")
raw = mat_contents[TABLE_NAME]
if not isinstance(raw, pd.DataFrame):
    raise TypeError(f"Expected a DataFrame, found {type(raw).__name__}.")

required_columns = {
    "training_level",
    "session_type",
    "repeat_trial",
    "LED_trial",
    "intended_fix",
    "LED_onset_time",
}
missing_columns = sorted(required_columns - set(raw.columns))
if missing_columns:
    raise RuntimeError(f"Source table is missing {missing_columns}.")

filtered = raw.loc[raw["training_level"].eq(TRAINING_LEVEL)].copy()
filtered = filtered.loc[filtered["session_type"].isin(SESSION_TYPES)].copy()
filtered = filtered.loc[
    filtered["repeat_trial"].isin(REPEAT_TRIAL_VALUES)
    | filtered["repeat_trial"].isna()
].copy()
filtered = filtered.loc[filtered["LED_trial"].isin(LED_TRIAL_VALUES)].copy()

s7 = filtered.loc[filtered["session_type"].eq(7)].copy()
s9 = filtered.loc[filtered["session_type"].eq(9)].copy()
s7["effective_scheduled_onset"] = s7["intended_fix"] - s7["LED_onset_time"]
s9["effective_scheduled_onset"] = s9["LED_onset_time"]

if not s7.index.is_unique or not s9.index.is_unique:
    raise RuntimeError("Source-row indices must be unique.")
if len(s9) != EXPECTED_S9_ROWS:
    raise RuntimeError(f"Expected {EXPECTED_S9_ROWS:,} s9 rows, found {len(s9):,}.")
for led in LED_TRIAL_VALUES:
    actual = int(s7["LED_trial"].eq(led).sum())
    if actual != EXPECTED_S7_COUNTS[led]:
        raise RuntimeError(f"LED {led}: expected {EXPECTED_S7_COUNTS[led]:,} s7 rows, found {actual:,}.")
for label, frame in (("s7", s7), ("s9", s9)):
    timing = frame[MATCHING_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(timing).all():
        raise RuntimeError(f"{label} contains non-finite timing.")
    if (frame["effective_scheduled_onset"] < 0).any():
        raise RuntimeError(f"{label} contains a negative scheduled onset.")
    if (
        frame["effective_scheduled_onset"]
        > frame["intended_fix"] + 1e-12
    ).any():
        raise RuntimeError(f"{label} has scheduled onset after intended_fix.")

reference_intended_counts, _ = np.histogram(s9["intended_fix"], MATCH_INTENDED_BINS)
reference_onset_counts, _ = np.histogram(
    s9["effective_scheduled_onset"], MATCH_ONSET_BINS
)
if int(reference_intended_counts.sum()) != len(s9):
    raise RuntimeError("The s9 intended_fix reference exceeds matching support.")
if int(reference_onset_counts.sum()) != len(s9):
    raise RuntimeError("The s9 onset reference exceeds matching support.")


# %%
# Keep all eligible rows, reproduce the two saved selections, and make two new ones
stage_groups = {}
for stage_label, sizes, source in STAGES:
    groups = {}
    if source in {"strict", "relaxed"}:
        fit_dir = STRICT_FIT_DIR if source == "strict" else RELAXED_FIT_DIR
        manifest_path = fit_dir / "source_row_manifest.csv"
        matching_audit_path = fit_dir / "matching_audit.json"
        for path in (manifest_path, matching_audit_path):
            if not path.exists():
                raise FileNotFoundError(path)
        saved_audit = json.loads(matching_audit_path.read_text())
        if int(saved_audit["matching_seed"]) != MATCH_RANDOM_SEED:
            raise RuntimeError(f"{source} fit used another matching seed.")
        manifest = pd.read_csv(manifest_path)
        if not manifest["source_row_index"].is_unique:
            raise RuntimeError(f"{source} manifest repeats source rows.")
        if not {"selection_order", "LED_trial", "source_row_index"}.issubset(
            manifest.columns
        ):
            raise RuntimeError(f"{source} manifest is missing selection columns.")

    for led in LED_TRIAL_VALUES:
        candidate = s7.loc[s7["LED_trial"].eq(led)].copy()
        expected_n = sizes[led]
        if source == "all":
            group = candidate
        elif source == "new":
            target_intended = largest_remainder_counts(
                reference_intended_counts, expected_n
            )
            target_onset = largest_remainder_counts(
                reference_onset_counts, expected_n
            )
            selected_indices, _ = select_rows_matching_two_margins(
                timing_frame=candidate[MATCHING_COLUMNS].copy(),
                target_intended_counts=target_intended,
                target_onset_counts=target_onset,
                rng=np.random.default_rng(MATCH_RANDOM_SEED + expected_n + led),
                matching_columns=MATCHING_COLUMNS,
                intended_bins=MATCH_INTENDED_BINS,
                onset_bins=MATCH_ONSET_BINS,
                sample_size=expected_n,
                bin_width_s=BIN_WIDTH_S,
            )
            group = candidate.loc[selected_indices].copy()
        else:
            saved_group = manifest.loc[manifest["LED_trial"].eq(led)]
            if len(saved_group) != expected_n:
                raise RuntimeError(f"{source} LED {led} has the wrong row count.")
            ordered_indices = (
                saved_group.sort_values("selection_order")["source_row_index"]
                .astype(int)
                .to_numpy()
            )
            selection_hash = hashlib.sha256(
                "\n".join(str(index) for index in ordered_indices).encode()
            ).hexdigest()
            if selection_hash != SAVED_SELECTION_HASHES[source][led]:
                raise RuntimeError(f"{source} LED {led} row identity changed.")
            if not np.isin(ordered_indices, candidate.index.to_numpy()).all():
                raise RuntimeError(f"{source} LED {led} includes a noncandidate row.")
            group = candidate.loc[ordered_indices].copy()
            for column in MATCHING_COLUMNS:
                if not np.allclose(
                    group[column].to_numpy(),
                    saved_group.sort_values("selection_order")[column].to_numpy(),
                    rtol=0,
                    atol=1e-12,
                ):
                    raise RuntimeError(f"{source} LED {led} timing changed.")
        if len(group) != expected_n or not group.index.is_unique:
            raise RuntimeError(f"{stage_label} LED {led} has repeated or missing rows.")
        groups[led] = group

    if not groups[0].index.intersection(groups[1].index).empty:
        raise RuntimeError(f"{stage_label} OFF and ON selections overlap.")
    stage_groups[stage_label] = groups


# %%
# Compute full-support densities and unbinned discrepancies
variables = (
    ("intended_fix", "intended_fix (s)"),
    ("effective_scheduled_onset", "LED onset from fixation (s)"),
)
reference_densities = {
    column: density_histogram(s9[column], DISPLAY_BINS)[0]
    for column, _ in variables
}
plot_payload = {}
audit_rows = []
for stage_label, sizes, source in STAGES:
    for led in LED_TRIAL_VALUES:
        group = stage_groups[stage_label][led]
        for column, _ in variables:
            density, area = density_histogram(group[column], DISPLAY_BINS)
            ks = float(
                ks_2samp(group[column], s9[column], method="asymp").statistic
            )
            wasserstein_ms = float(
                1000 * wasserstein_distance(group[column], s9[column])
            )
            plot_payload[(stage_label, led, column)] = {
                "density": density,
                "ks": ks,
            }
            audit_rows.append(
                {
                    "stage": stage_label,
                    "selection_source": source,
                    "variable": column,
                    "LED_trial": led,
                    "selected_n": len(group),
                    "reference_n": len(s9),
                    "KS_distance": ks,
                    "Wasserstein_ms": wasserstein_ms,
                    "histogram_area": area,
                }
            )

audit = pd.DataFrame(audit_rows)
if len(audit) != 2 * len(STAGES) * len(LED_TRIAL_VALUES):
    raise RuntimeError("Timing audit is incomplete.")
audit.to_csv(OUTPUT_AUDIT_PATH, index=False)


# %%
# Draw the 2 x 5 comparison on common full-support axes
colors = {0: "tab:blue", 1: "tab:red"}
bin_centers = 0.5 * (DISPLAY_BINS[:-1] + DISPLAY_BINS[1:])

fig, axes = plt.subplots(2, 5, figsize=(22.0, 8.8), sharex=True, sharey="row")
for column_index, (stage_label, sizes, _) in enumerate(STAGES):
    for row_index, (column, xlabel) in enumerate(variables):
        ax = axes[row_index, column_index]
        ax.step(
            bin_centers,
            reference_densities[column],
            where="mid",
            color="black",
            linestyle="--",
            linewidth=1.7,
            zorder=3,
        )
        for led in LED_TRIAL_VALUES:
            payload = plot_payload[(stage_label, led, column)]
            ax.step(
                bin_centers,
                payload["density"],
                where="mid",
                color=colors[led],
                linewidth=1.3,
                zorder=2,
            )
            ax.text(
                0.97,
                0.95 - 0.09 * led,
                f"{'OFF' if led == 0 else 'ON'} KS = {payload['ks']:.3f}",
                color=colors[led],
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
            )
        if row_index == 0:
            ax.set_title(
                f"{stage_label}\nOFF {sizes[0]:,} · ON {sizes[1]:,}",
                fontsize=12,
            )
        ax.set_xlabel(xlabel)
        if column_index == 0:
            ax.set_ylabel("Density (s$^{-1}$)")
        ax.set_xlim(0, 2.2)
        ax.grid(axis="y", color="0.88", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)

legend_handles = [
    Line2D([0], [0], color="black", linestyle="--", linewidth=1.7, label="Pooled s9"),
    Line2D([0], [0], color=colors[0], linewidth=1.3, label="s7 LED OFF"),
    Line2D([0], [0], color=colors[1], linewidth=1.3, label="s7 LED ON"),
]
fig.suptitle("LED7 session type 7 versus session type 9: timing-match sensitivity", y=0.995)
fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=3, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.savefig(OUTPUT_FIGURE_PATH, dpi=OUTPUT_DPI, bbox_inches="tight")
if SHOW_PLOT:
    plt.show()
else:
    plt.close(fig)

print(audit.to_string(index=False))
print(f"\nFigure: {OUTPUT_FIGURE_PATH.resolve()}")
print(f"Audit: {OUTPUT_AUDIT_PATH.resolve()}")
