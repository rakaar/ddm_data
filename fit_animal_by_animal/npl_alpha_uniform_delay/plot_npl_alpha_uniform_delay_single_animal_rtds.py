# %%
"""Fit-aligned RTDs for one NPL+alpha uniform-delay SVI fit.

The empirical and posterior-mean model RTDs use the exact successful 0--1 s
RT+choice fitting pool.  Signed ILDs are normalized separately and then
averaged equally into ABL by absolute-ILD panels.
"""

# %%
# =============================================================================
# Editable parameters
# =============================================================================
from pathlib import Path
import os
import pickle
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
ANIMAL_FIT_DIR = SCRIPT_DIR.parent
REPO_DIR = ANIMAL_FIT_DIR.parent

BATCH_NAME = os.environ.get("NUMPYRO_SVI_BATCH", "LED7")
ANIMAL = int(os.environ.get("NUMPYRO_SVI_ANIMAL", "92"))
ABLS = (20, 40, 60)
ABS_ILDS = (1.0, 2.0, 4.0, 8.0, 16.0)

FIT_RT_MIN_S = 0.0
FIT_RT_MAX_S = 1.0
MODEL_STEP_S = 0.001
DATA_BIN_S = float(os.environ.get("NUMPYRO_SVI_DIAG_DATA_BIN_S", "0.005"))
DISPLAY_RT_MAX_S = 0.600
T_TRUNC_S = 0.300
K_MAX = int(os.environ.get("K_MAX", "10"))
INTEGRATED_CDF_TERMS = int(os.environ.get("INTEGRATED_CDF_TERMS", "200"))
TRIAL_CHUNK_SIZE = 512
PLOT_DPI = 300

OUTPUT_ROOT = Path(
    os.environ.get(
        "NUMPYRO_SVI_OUTPUT_ROOT",
        str(
            SCRIPT_DIR
            / (
                "numpyro_svi_npl_alpha_uniform_delay_rt_choice_"
                "patience12_min50k_restore_best_outputs"
            )
        ),
    )
).expanduser()
OUTPUT_DIR = OUTPUT_ROOT / f"{BATCH_NAME}_{ANIMAL}"
DIAGNOSTIC_DIR = OUTPUT_DIR / "diagnostics"
DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)

DATA_CSV = (
    REPO_DIR
    / "raw_data"
    / "batch_csvs"
    / f"batch_{BATCH_NAME}_valid_and_aborts.csv"
)
ABORT_RESULT_PKL = (
    REPO_DIR
    / "aborts_ipl_npl_time_fit_results"
    / f"results_{BATCH_NAME}_animal_{ANIMAL}.pkl"
)
POSTERIOR_NPZ = OUTPUT_DIR / "main_fullrank_posterior_samples.npz"
CONDITION_CSV = OUTPUT_DIR / "condition_table.csv"
DATA_BIN_MS_TEXT = f"{1e3 * DATA_BIN_S:g}".replace(".", "p")
DATA_BIN_FILENAME_PART = (
    ""
    if abs(DATA_BIN_S - 0.005) < 1e-12
    else f"_data_bin_{DATA_BIN_MS_TEXT}ms"
)
RTD_PNG = (
    DIAGNOSTIC_DIR
    / (
        f"{BATCH_NAME.lower()}_{ANIMAL}_npl_alpha_uniform_delay_"
        "fit_aligned_rtds_by_abl_abs_ild"
        f"{DATA_BIN_FILENAME_PART}_0_600ms_xlim.png"
    )
)
RTD_PKL = (
    DIAGNOSTIC_DIR
    / (
        f"{BATCH_NAME.lower()}_{ANIMAL}_npl_alpha_uniform_delay_"
        "fit_aligned_rtds_by_abl_abs_ild"
        f"{DATA_BIN_FILENAME_PART}.pkl"
    )
)


# %%
# =============================================================================
# Imports and plotting setup
# =============================================================================
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

for import_path in (ANIMAL_FIT_DIR, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import numpyro_npl_alpha_svi_utils as point_utils
import uniform_delay_likelihood_utils as uniform_utils
import uniform_delay_svi_utils as svi_utils

fit_window_bins = (FIT_RT_MAX_S - FIT_RT_MIN_S) / DATA_BIN_S
if DATA_BIN_S <= 0.0 or not np.isclose(fit_window_bins, round(fit_window_bins)):
    raise ValueError(
        "NUMPYRO_SVI_DIAG_DATA_BIN_S must be positive and divide the 0--1 s "
        f"fitting window exactly; got {DATA_BIN_S:g} s."
    )

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Helvetica",
            "Nimbus Sans",
            "Helvetica Neue",
            "Arial",
            "Liberation Sans",
            "sans-serif",
        ],
    }
)


# %%
# =============================================================================
# Load exact fitting rows and posterior means
# =============================================================================
for required_path in (DATA_CSV, ABORT_RESULT_PKL, POSTERIOR_NPZ, CONDITION_CSV):
    if not required_path.exists():
        raise FileNotFoundError(required_path)

raw_df = pd.read_csv(DATA_CSV)
if "choice" not in raw_df.columns:
    raw_df["choice"] = raw_df["response_poke"].map({3: 1, 2: -1})
for column in (
    "animal",
    "success",
    "RTwrtStim",
    "TotalFixTime",
    "intended_fix",
    "ABL",
    "ILD",
    "choice",
):
    raw_df[column] = pd.to_numeric(raw_df[column], errors="coerce")

valid_df = raw_df[
    raw_df["animal"].eq(ANIMAL)
    & raw_df["success"].isin([1, -1])
    & raw_df["RTwrtStim"].ge(FIT_RT_MIN_S)
    & raw_df["RTwrtStim"].lt(FIT_RT_MAX_S)
    & raw_df["ABL"].isin(ABLS)
].copy()
valid_df = valid_df.dropna(
    subset=[
        "RTwrtStim",
        "TotalFixTime",
        "intended_fix",
        "ABL",
        "ILD",
        "choice",
    ]
)
valid_df["ABL"] = valid_df["ABL"].astype(int)
valid_df["ILD"] = valid_df["ILD"].astype(float)

condition_table = (
    pd.read_csv(CONDITION_CSV)
    .sort_values("condition_id")
    .reset_index(drop=True)
)
reconstructed_conditions = (
    valid_df[["ABL", "ILD"]]
    .drop_duplicates()
    .sort_values(["ABL", "ILD"])
    .reset_index(drop=True)
)
reconstructed_conditions["condition_id"] = np.arange(
    len(reconstructed_conditions),
    dtype=int,
)
if len(condition_table) != 30 or not np.allclose(
    condition_table[["ABL", "ILD", "condition_id"]].to_numpy(dtype=float),
    reconstructed_conditions[["ABL", "ILD", "condition_id"]].to_numpy(
        dtype=float
    ),
):
    raise RuntimeError("Saved and reconstructed condition tables differ.")

valid_df = valid_df.merge(
    reconstructed_conditions,
    on=["ABL", "ILD"],
    how="left",
    validate="many_to_one",
)
posterior_file = np.load(POSTERIOR_NPZ)
required_keys = tuple(svi_utils.GLOBAL_PARAM_NAMES) + (
    "t_E_aff_center",
    "t_E_aff_width",
)
missing_keys = [key for key in required_keys if key not in posterior_file.files]
if missing_keys:
    raise KeyError(f"Missing posterior keys: {missing_keys}")

params = {
    key: float(np.mean(np.asarray(posterior_file[key], dtype=float)))
    for key in svi_utils.GLOBAL_PARAM_NAMES
}
delay_centers = np.mean(
    np.asarray(posterior_file["t_E_aff_center"], dtype=float),
    axis=0,
)
delay_widths = np.mean(
    np.asarray(posterior_file["t_E_aff_width"], dtype=float),
    axis=0,
)
delay_lows = delay_centers - 0.5 * delay_widths
delay_highs = delay_centers + 0.5 * delay_widths
if not (
    np.isfinite(delay_lows).all()
    and np.isfinite(delay_highs).all()
    and np.all(delay_lows >= 0.0)
    and np.all(delay_highs <= 1.0)
):
    raise RuntimeError("Posterior-mean delay intervals are invalid.")

with ABORT_RESULT_PKL.open("rb") as file:
    abort_fit = pickle.load(file)["vbmc_aborts_results"]
V_A = float(np.mean(np.asarray(abort_fit["V_A_samples"], dtype=float)))
theta_A = float(np.mean(np.asarray(abort_fit["theta_A_samples"], dtype=float)))
t_A_aff = float(np.mean(np.asarray(abort_fit["t_A_aff_samp"], dtype=float)))

if BATCH_NAME == "LED7" and ANIMAL == 92 and len(valid_df) != 12137:
    raise RuntimeError(f"Expected 12,137 fitting rows, found {len(valid_df):,}.")
print(f"Fitting rows: {len(valid_df):,}; conditions: {len(condition_table)}")
print(
    "Posterior-mean delay widths: "
    f"{1e3 * delay_widths.min():.3f}--{1e3 * delay_widths.max():.3f} ms"
)


# %%
# =============================================================================
# Condition-level evidence terms and trial-averaged race RTDs
# =============================================================================
rt_grid_s = np.arange(
    FIT_RT_MIN_S,
    FIT_RT_MAX_S + 0.5 * MODEL_STEP_S,
    MODEL_STEP_S,
)
data_bins_s = np.arange(
    FIT_RT_MIN_S,
    FIT_RT_MAX_S + 0.5 * DATA_BIN_S,
    DATA_BIN_S,
)
data_bin_centers_s = 0.5 * (data_bins_s[:-1] + data_bins_s[1:])

condition_abls = condition_table["ABL"].to_numpy(dtype=float)[:, None]
condition_ilds = condition_table["ILD"].to_numpy(dtype=float)[:, None]
Z_E = (params["w"] - 0.5) * 2.0 * params["theta_E"]

elapsed_jax = jnp.asarray(rt_grid_s[None, :], dtype=jnp.float64)
abl_jax = jnp.asarray(condition_abls, dtype=jnp.float64)
ild_jax = jnp.asarray(condition_ilds, dtype=jnp.float64)
low_jax = jnp.asarray(delay_lows[:, None], dtype=jnp.float64)
high_jax = jnp.asarray(delay_highs[:, None], dtype=jnp.float64)

evidence_cdf = np.asarray(
    uniform_utils.uniform_delay_bound_cdf_alpha_jax(
        elapsed_jax,
        1,
        low_jax,
        high_jax,
        abl_jax,
        ild_jax,
        params["rate_lambda"],
        params["T_0"],
        params["theta_E"],
        Z_E,
        params["rate_norm_l"],
        params["alpha"],
        INTEGRATED_CDF_TERMS,
    )
    + uniform_utils.uniform_delay_bound_cdf_alpha_jax(
        elapsed_jax,
        -1,
        low_jax,
        high_jax,
        abl_jax,
        ild_jax,
        params["rate_lambda"],
        params["T_0"],
        params["theta_E"],
        Z_E,
        params["rate_norm_l"],
        params["alpha"],
        INTEGRATED_CDF_TERMS,
    )
)
evidence_pdf = np.asarray(
    uniform_utils.uniform_delay_bound_pdf_alpha_jax(
        elapsed_jax,
        1,
        low_jax,
        high_jax,
        abl_jax,
        ild_jax,
        params["rate_lambda"],
        params["T_0"],
        params["theta_E"],
        Z_E,
        params["rate_norm_l"],
        params["alpha"],
        K_MAX,
    )
    + uniform_utils.uniform_delay_bound_pdf_alpha_jax(
        elapsed_jax,
        -1,
        low_jax,
        high_jax,
        abl_jax,
        ild_jax,
        params["rate_lambda"],
        params["T_0"],
        params["theta_E"],
        Z_E,
        params["rate_norm_l"],
        params["alpha"],
        K_MAX,
    )
)
evidence_cdf_at_zero = evidence_cdf[:, 0]
evidence_cdf_at_one = evidence_cdf[:, -1]

model_sum_by_condition = np.zeros((30, len(rt_grid_s)), dtype=float)
model_n_by_condition = np.zeros(30, dtype=int)
normalization_denominators = []

for start in range(0, len(valid_df), TRIAL_CHUNK_SIZE):
    chunk = valid_df.iloc[start : start + TRIAL_CHUNK_SIZE]
    condition_ids = chunk["condition_id"].to_numpy(dtype=int)
    t_stim = chunk["intended_fix"].to_numpy(dtype=float)
    absolute_time = t_stim[:, None] + rt_grid_s[None, :]

    proactive_pdf = np.asarray(
        point_utils.rho_A_t_jax(
            jnp.asarray(absolute_time - t_A_aff),
            V_A,
            theta_A,
        )
    )
    proactive_cdf = np.asarray(
        point_utils.cum_A_t_jax(
            jnp.asarray(absolute_time - t_A_aff),
            V_A,
            theta_A,
        )
    )

    def legacy_proactive_cdf(absolute_values):
        raw_cdf = np.asarray(
            point_utils.cum_A_t_jax(
                jnp.asarray(absolute_values - t_A_aff),
                V_A,
                theta_A,
            )
        )
        survival = float(
            1.0
            - point_utils.cum_A_t_jax(
                T_TRUNC_S - t_A_aff,
                V_A,
                theta_A,
            )
        )
        return np.where(absolute_values < T_TRUNC_S, 0.0, raw_cdf / survival)

    proactive_cdf_at_zero = legacy_proactive_cdf(t_stim)
    proactive_cdf_at_one = legacy_proactive_cdf(t_stim + FIT_RT_MAX_S)
    cdf_lower = (
        proactive_cdf_at_zero
        + evidence_cdf_at_zero[condition_ids]
        - proactive_cdf_at_zero * evidence_cdf_at_zero[condition_ids]
    )
    cdf_upper = (
        proactive_cdf_at_one
        + evidence_cdf_at_one[condition_ids]
        - proactive_cdf_at_one * evidence_cdf_at_one[condition_ids]
    )
    retained_mass = cdf_upper - cdf_lower
    if not np.isfinite(retained_mass).all() or np.any(retained_mass <= 0.0):
        raise RuntimeError("Invalid retained 0--1 s race mass.")
    normalization_denominators.extend(retained_mass.tolist())

    numerator = (
        proactive_pdf * (1.0 - evidence_cdf[condition_ids])
        + evidence_pdf[condition_ids] * (1.0 - proactive_cdf)
    )
    normalized_density = numerator / retained_mass[:, None]
    if not np.isfinite(normalized_density).all():
        raise RuntimeError("Non-finite model RTD.")

    for condition_id in np.unique(condition_ids):
        condition_mask = condition_ids == condition_id
        model_sum_by_condition[condition_id] += normalized_density[
            condition_mask
        ].sum(axis=0)
        model_n_by_condition[condition_id] += int(condition_mask.sum())

expected_counts = (
    valid_df.groupby("condition_id")
    .size()
    .reindex(range(30), fill_value=0)
    .to_numpy(dtype=int)
)
if not np.array_equal(model_n_by_condition, expected_counts):
    raise RuntimeError("Model averaging counts do not match fitting rows.")


# %%
# =============================================================================
# Normalize signed conditions and average signs, ABLs, and absolute ILDs
# =============================================================================
def normalize_histogram(values):
    values = np.asarray(values, dtype=float)
    area = float(np.sum(values * np.diff(data_bins_s)))
    if not np.isfinite(area) or area <= 0.0:
        raise RuntimeError(f"Invalid data RTD area: {area}.")
    return values / area


def normalize_density(values):
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    area = float(trapezoid(values, rt_grid_s))
    if not np.isfinite(area) or area <= 0.0:
        raise RuntimeError(f"Invalid model RTD area: {area}.")
    return values / area


data_signed = {}
model_signed = {}
for condition in condition_table.itertuples(index=False):
    condition_id = int(condition.condition_id)
    condition_rows = valid_df[
        valid_df["ABL"].eq(int(condition.ABL))
        & np.isclose(valid_df["ILD"], float(condition.ILD))
    ]
    counts, _ = np.histogram(
        condition_rows["RTwrtStim"].to_numpy(dtype=float),
        bins=data_bins_s,
    )
    key = (int(condition.ABL), float(condition.ILD))
    data_signed[key] = normalize_histogram(counts.astype(float) / DATA_BIN_S)
    model_signed[key] = normalize_density(
        model_sum_by_condition[condition_id] / model_n_by_condition[condition_id]
    )

data_panel = np.empty((len(ABLS), len(ABS_ILDS), len(data_bin_centers_s)))
model_panel = np.empty((len(ABLS), len(ABS_ILDS), len(rt_grid_s)))
for abl_idx, abl in enumerate(ABLS):
    for ild_idx, abs_ild in enumerate(ABS_ILDS):
        signed_keys = ((abl, -abs_ild), (abl, abs_ild))
        data_panel[abl_idx, ild_idx] = normalize_histogram(
            np.mean([data_signed[key] for key in signed_keys], axis=0)
        )
        model_panel[abl_idx, ild_idx] = normalize_density(
            np.mean([model_signed[key] for key in signed_keys], axis=0)
        )

data_by_abl = np.empty((len(ABLS), len(data_bin_centers_s)))
model_by_abl = np.empty((len(ABLS), len(rt_grid_s)))
for abl_idx in range(len(ABLS)):
    data_by_abl[abl_idx] = normalize_histogram(
        np.mean(data_panel[abl_idx], axis=0)
    )
    model_by_abl[abl_idx] = normalize_density(
        np.mean(model_panel[abl_idx], axis=0)
    )

data_by_ild = np.empty((len(ABS_ILDS), len(data_bin_centers_s)))
model_by_ild = np.empty((len(ABS_ILDS), len(rt_grid_s)))
for ild_idx in range(len(ABS_ILDS)):
    data_by_ild[ild_idx] = normalize_histogram(
        np.mean(data_panel[:, ild_idx], axis=0)
    )
    model_by_ild[ild_idx] = normalize_density(
        np.mean(model_panel[:, ild_idx], axis=0)
    )

data_grand = normalize_histogram(np.mean(data_panel, axis=(0, 1)))
model_grand = normalize_density(np.mean(model_panel, axis=(0, 1)))

data_areas = np.sum(data_panel * np.diff(data_bins_s), axis=-1)
model_areas = trapezoid(model_panel, rt_grid_s, axis=-1)
if not np.allclose(data_areas, 1.0, atol=1e-10):
    raise RuntimeError(f"Data panels are not normalized: {data_areas}.")
if not np.allclose(model_areas, 1.0, atol=1e-8):
    raise RuntimeError(f"Model panels are not normalized: {model_areas}.")


# %%
# =============================================================================
# Direct bound-sum identity check
# =============================================================================
check_trial = valid_df.iloc[0]
check_condition_id = int(check_trial["condition_id"])
check_t_stim = float(check_trial["intended_fix"])
direct_sum = np.zeros_like(rt_grid_s)
for bound in (-1, 1):
    direct_sum += np.asarray(
        svi_utils.up_or_down_alpha_uniform_delay_legacy_jax(
            jnp.asarray(check_t_stim + rt_grid_s),
            bound,
            V_A,
            theta_A,
            t_A_aff,
            check_t_stim,
            float(check_trial["ABL"]),
            float(check_trial["ILD"]),
            params["rate_lambda"],
            params["T_0"],
            params["theta_E"],
            Z_E,
            delay_lows[check_condition_id],
            delay_highs[check_condition_id],
            params["del_go"],
            params["rate_norm_l"],
            params["alpha"],
            K_MAX,
            INTEGRATED_CDF_TERMS,
        )
    )
check_absolute_time = check_t_stim + rt_grid_s
check_proactive_pdf = np.asarray(
    point_utils.rho_A_t_jax(check_absolute_time - t_A_aff, V_A, theta_A)
)
check_proactive_cdf = np.asarray(
    point_utils.cum_A_t_jax(check_absolute_time - t_A_aff, V_A, theta_A)
)
collapsed_sum = (
    check_proactive_pdf * (1.0 - evidence_cdf[check_condition_id])
    + evidence_pdf[check_condition_id] * (1.0 - check_proactive_cdf)
)
identity_scale = max(1.0, float(np.max(np.abs(direct_sum))))
identity_error = float(np.max(np.abs(direct_sum - collapsed_sum))) / identity_scale
if identity_error > 1e-10:
    raise RuntimeError(f"Choice-collapsed identity failed: {identity_error:.3g}.")


# %%
# =============================================================================
# Plot 3 ABL rows plus averages, 5 absolute ILDs plus averages
# =============================================================================
fig, axes = plt.subplots(4, 6, figsize=(19.5, 10.4), sharex=True)
for row_idx in range(4):
    for column_idx in range(6):
        ax = axes[row_idx, column_idx]
        if row_idx < 3 and column_idx < 5:
            data_curve = data_panel[row_idx, column_idx]
            model_curve = model_panel[row_idx, column_idx]
        elif row_idx < 3:
            data_curve = data_by_abl[row_idx]
            model_curve = model_by_abl[row_idx]
        elif column_idx < 5:
            data_curve = data_by_ild[column_idx]
            model_curve = model_by_ild[column_idx]
        else:
            data_curve = data_grand
            model_curve = model_grand

        ax.step(
            1e3 * data_bin_centers_s,
            data_curve,
            where="mid",
            color="black",
            linewidth=0.8,
            alpha=0.62,
        )
        ax.plot(
            1e3 * rt_grid_s,
            model_curve,
            color="#0072B2",
            linewidth=1.5,
        )
        ax.set_xlim(0.0, 1e3 * DISPLAY_RT_MAX_S)
        ax.set_xticks([0, 300, 600])
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)

        if row_idx == 0:
            if column_idx < 5:
                ax.set_title(f"|ILD| = {ABS_ILDS[column_idx]:g} dB", fontsize=10)
            else:
                ax.set_title("Mean across |ILD|", fontsize=10)
        if column_idx == 0:
            if row_idx < 3:
                ax.set_ylabel(f"ABL {ABLS[row_idx]} dB\nDensity (s$^{{-1}}$)")
            else:
                ax.set_ylabel("Mean across ABL\nDensity (s$^{-1}$)")
        if row_idx == 3:
            ax.set_xlabel("RT from stimulus (ms)")

legend_handles = [
    Line2D([0], [0], color="black", linewidth=0.8, alpha=0.62, label="Data"),
    Line2D([0], [0], color="#0072B2", linewidth=1.5, label="Model"),
]
axes[0, 5].legend(handles=legend_handles, frameon=False, fontsize=9)
fig.suptitle(
    (
        f"{BATCH_NAME}/{ANIMAL} NPL+alpha uniform-delay RT+choice fit "
        f"({1e3 * DATA_BIN_S:g} ms data bins)"
    ),
    fontsize=14,
    y=0.995,
)
fig.tight_layout(rect=(0, 0, 1, 0.985), w_pad=1.0, h_pad=1.0)
fig.savefig(RTD_PNG, dpi=PLOT_DPI, bbox_inches="tight")

payload = {
    "batch_name": BATCH_NAME,
    "animal": ANIMAL,
    "fit_rows": len(valid_df),
    "rt_grid_s": rt_grid_s,
    "data_bins_s": data_bins_s,
    "data_bin_centers_s": data_bin_centers_s,
    "ABLs": ABLS,
    "abs_ilds": ABS_ILDS,
    "data_rtd_by_abl_abs_ild": data_panel,
    "model_rtd_by_abl_abs_ild": model_panel,
    "data_rtd_by_abl": data_by_abl,
    "model_rtd_by_abl": model_by_abl,
    "data_rtd_by_abs_ild": data_by_ild,
    "model_rtd_by_abs_ild": model_by_ild,
    "data_grand_rtd": data_grand,
    "model_grand_rtd": model_grand,
    "posterior_mean_params": params,
    "delay_center_mean_s": delay_centers,
    "delay_width_mean_s": delay_widths,
    "delay_low_mean_s": delay_lows,
    "delay_high_mean_s": delay_highs,
    "retained_mass_min": float(np.min(normalization_denominators)),
    "retained_mass_max": float(np.max(normalization_denominators)),
    "choice_collapsed_identity_relative_error": identity_error,
    "data_panel_areas": data_areas,
    "model_panel_areas": model_areas,
    "data_bin_width_s": DATA_BIN_S,
    "model_step_s": MODEL_STEP_S,
}
with RTD_PKL.open("wb") as file:
    pickle.dump(payload, file)

print(f"Choice-collapsed identity relative error: {identity_error:.3e}")
print(
    "Retained-mass range: "
    f"{np.min(normalization_denominators):.6f}--"
    f"{np.max(normalization_denominators):.6f}"
)
print(f"Saved: {RTD_PNG}")
print(f"Saved: {RTD_PKL}")
