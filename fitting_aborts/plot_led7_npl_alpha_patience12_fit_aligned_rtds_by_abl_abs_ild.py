# %%
"""
LED7 fit-aligned RTDs by ABL and absolute ILD.

The direct patience-12 NPL+alpha SVI posterior means are evaluated for the
exact valid 0--1 s fitting rows. Signed-condition RTDs are normalized first,
then averaged equally across ILD signs and across the six LED7 animals.
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

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
FIT_UTILS_DIR = REPO_ROOT / "fit_animal_by_animal"

BATCH_NAME = "LED7"
ANIMALS = (92, 93, 98, 99, 100, 103)
ABLS = (20, 40, 60)
SIGNED_ILDS = (-16.0, -8.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0, 16.0)
ABS_ILDS = (1.0, 2.0, 4.0, 8.0, 16.0)

FIT_RT_MIN_S = 0.0
FIT_RT_MAX_S = 1.0
MODEL_STEP_S = 0.001
DATA_BIN_S = 0.005
RIGHT_TRUNCATION_S = 0.120
T_TRUNC_S = 0.300
RTD_ABORT_EVENTS = (3, 4)
K_MAX = 10
TRIAL_CHUNK_SIZE = 512
PLOT_DPI = 300

DATA_CSV = REPO_ROOT / "raw_data" / "batch_csvs" / "batch_LED7_valid_and_aborts.csv"
RTD_DATA_CSV = REPO_ROOT / "raw_data" / "out_LED.csv"
FIT_ROOT = (
    REPO_ROOT
    / "fit_animal_by_animal"
    / "numpyro_svi_npl_alpha_condition_delay_patience12_restore_best_outputs"
)
ABORT_ROOT = REPO_ROOT / "aborts_ipl_npl_time_fit_results"
OUTPUT_DIR = SCRIPT_DIR / "led7_npl_alpha_patience12_fit_aligned_valid_rtds"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OVERVIEW_PNG = (
    OUTPUT_DIR
    / "led7_npl_alpha_patience12_fit_aligned_rtds_by_abl_abs_ild_0_1s.png"
)
ZOOM_PNG = (
    OUTPUT_DIR
    / "led7_npl_alpha_patience12_fit_aligned_rtds_by_abl_abs_ild_0_160ms_xlim.png"
)
ZOOM_120MS_PNG = (
    OUTPUT_DIR
    / "led7_npl_alpha_patience12_fit_aligned_rtds_by_abl_abs_ild_0_120ms_xlim.png"
)
RIGHT_TRUNCATED_120MS_PNG = (
    OUTPUT_DIR
    / (
        "led7_npl_alpha_patience12_fit_aligned_rtds_by_abl_abs_ild_"
        "right_truncated_120ms_0_1s.png"
    )
)
DELAY_BY_ILD_PNG = (
    OUTPUT_DIR
    / "led7_npl_alpha_patience12_t_E_aff_vs_ild_by_abl.png"
)
OUTPUT_PKL = (
    OUTPUT_DIR
    / "led7_npl_alpha_patience12_fit_aligned_rtds_by_abl_abs_ild.pkl"
)


# %%
# =============================================================================
# Imports and exact likelihood helpers
# =============================================================================
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

if str(FIT_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(FIT_UTILS_DIR))
import numpyro_npl_alpha_svi_utils as svi_utils

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
# Reused numerical operations
# =============================================================================
def normalize_histogram_density(values):
    values = np.asarray(values, dtype=float)
    area = float(np.sum(values * np.diff(data_bins_s)))
    if not np.isfinite(area) or area <= 0:
        raise RuntimeError(f"Invalid histogram area: {area}.")
    return values / area


def normalize_continuous_density(values):
    values = np.asarray(values, dtype=float)
    values = np.where(np.isfinite(values), np.maximum(values, 0), 0)
    area = float(trapezoid(values, rt_grid_s))
    if not np.isfinite(area) or area <= 0:
        raise RuntimeError(f"Invalid continuous-density area: {area}.")
    return values / area


def normalize_right_truncated_continuous_density(values):
    values = np.asarray(values, dtype=float)
    values = np.where(np.isfinite(values), np.maximum(values, 0), 0)
    keep = rt_grid_s <= RIGHT_TRUNCATION_S
    values = np.where(keep, values, 0)
    area = float(trapezoid(values[keep], rt_grid_s[keep]))
    if not np.isfinite(area) or area <= 0:
        raise RuntimeError(f"Invalid right-truncated density area: {area}.")
    return values / area


def mean_sem(values, axis=0):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    n = np.sum(finite, axis=axis)
    mean = np.nanmean(values, axis=axis)
    sd = np.nanstd(values, axis=axis, ddof=1)
    curr_sem = sd / np.sqrt(np.maximum(n, 1))
    curr_sem = np.where(n > 1, curr_sem, np.nan)
    return mean, curr_sem, n


# %%
# =============================================================================
# Load and validate the exact LED7 fitting rows
# =============================================================================
for required_path in [DATA_CSV, RTD_DATA_CSV, FIT_ROOT, ABORT_ROOT]:
    if not required_path.exists():
        raise FileNotFoundError(required_path)

batch_df = pd.read_csv(DATA_CSV)
required_columns = [
    "animal",
    "success",
    "RTwrtStim",
    "TotalFixTime",
    "intended_fix",
    "ABL",
    "ILD",
]
missing_columns = [column for column in required_columns if column not in batch_df.columns]
if missing_columns:
    raise KeyError(f"Missing columns in {DATA_CSV}: {missing_columns}")

valid_df = batch_df[
    batch_df["animal"].astype(int).isin(ANIMALS)
    & batch_df["success"].isin([1, -1])
    & batch_df["RTwrtStim"].between(FIT_RT_MIN_S, FIT_RT_MAX_S)
    & batch_df["ABL"].isin(ABLS)
    & batch_df["ILD"].isin(SIGNED_ILDS)
].copy()
valid_df = valid_df.dropna(subset=required_columns)
valid_df["animal"] = valid_df["animal"].astype(int)
valid_df["ABL"] = valid_df["ABL"].astype(int)
valid_df["ILD"] = valid_df["ILD"].astype(float)

if not np.allclose(
    valid_df["RTwrtStim"].to_numpy(dtype=float),
    valid_df["TotalFixTime"].to_numpy(dtype=float)
    - valid_df["intended_fix"].to_numpy(dtype=float),
    atol=2e-6,
    rtol=0,
):
    raise RuntimeError("RTwrtStim is inconsistent with TotalFixTime - intended_fix.")

rtd_source_df = pd.read_csv(RTD_DATA_CSV)
if "timed_fix" not in rtd_source_df.columns:
    raise KeyError(f"Missing timed_fix in {RTD_DATA_CSV}.")
rtd_source_df["RTwrtStim"] = (
    rtd_source_df["timed_fix"] - rtd_source_df["intended_fix"]
)
rtd_source_df = rtd_source_df.rename(columns={"timed_fix": "TotalFixTime"})

led_off = rtd_source_df["LED_trial"].eq(0) | rtd_source_df["LED_trial"].isna()
repeat_ok = (
    rtd_source_df["repeat_trial"].isin([0, 2])
    | rtd_source_df["repeat_trial"].isna()
)
led7_panel_rows = rtd_source_df[
    led_off
    & rtd_source_df["session_type"].eq(7)
    & rtd_source_df["training_level"].eq(16)
    & repeat_ok
    & rtd_source_df["animal"].isin(ANIMALS)
    & rtd_source_df["ABL"].isin(ABLS)
    & rtd_source_df["ILD"].isin(SIGNED_ILDS)
].copy()

rtd_trial_pool = led7_panel_rows[
    led7_panel_rows["success"].isin([1, -1])
    | led7_panel_rows["abort_event"].isin(RTD_ABORT_EVENTS)
].copy()
rtd_df = rtd_trial_pool[
    rtd_trial_pool["RTwrtStim"].between(
        FIT_RT_MIN_S,
        FIT_RT_MAX_S,
        inclusive="both",
    )
    & (
        rtd_trial_pool["success"].isin([1, -1])
        | rtd_trial_pool["TotalFixTime"].ge(T_TRUNC_S)
    )
].copy()
rtd_df = rtd_df.dropna(
    subset=["animal", "RTwrtStim", "TotalFixTime", "ABL", "ILD", "abort_event"]
)
rtd_df["animal"] = rtd_df["animal"].astype(int)
rtd_df["ABL"] = rtd_df["ABL"].astype(int)
rtd_df["ILD"] = rtd_df["ILD"].astype(float)

event_counts_before_truncation = {
    int(event): int(rtd_trial_pool["abort_event"].eq(event).sum())
    for event in RTD_ABORT_EVENTS
}
event_counts_after_truncation = {
    int(event): int(rtd_df["abort_event"].eq(event).sum())
    for event in RTD_ABORT_EVENTS
}
if event_counts_after_truncation[4] == 0:
    raise RuntimeError("Expected retained LED7 abort_event == 4 RTD rows.")

observed_animals = tuple(sorted(valid_df["animal"].unique()))
if observed_animals != tuple(sorted(ANIMALS)):
    raise RuntimeError(f"Expected LED7 animals {ANIMALS}, found {observed_animals}.")

condition_counts = (
    valid_df[["animal", "ABL", "ILD"]]
    .drop_duplicates()
    .groupby("animal")
    .size()
    .reindex(ANIMALS)
)
if not np.all(condition_counts.to_numpy() == 30):
    raise RuntimeError(
        "Each LED7 animal should have 30 signed ABL/ILD conditions:\n"
        f"{condition_counts.to_string()}"
    )

print(f"Data CSV: {DATA_CSV}")
print(f"RTD source CSV: {RTD_DATA_CSV}")
print(f"SVI fit root: {FIT_ROOT}")
print(f"Exact LED7 fitting rows: {len(valid_df):,}")
print(f"LED7 RTD rows after abort truncation: {len(rtd_df):,}")
print(
    f"RTD abort events {RTD_ABORT_EVENTS}: "
    f"before truncation={event_counts_before_truncation}, "
    f"after truncation={event_counts_after_truncation}"
)
print(f"Animals: {ANIMALS}")


# %%
# =============================================================================
# Per-animal, per-signed-condition fit-aligned RTDs
# =============================================================================
rt_grid_s = np.arange(
    round(FIT_RT_MIN_S / MODEL_STEP_S),
    round(FIT_RT_MAX_S / MODEL_STEP_S) + 1,
) * MODEL_STEP_S
data_bins_s = np.arange(
    round(FIT_RT_MIN_S / DATA_BIN_S),
    round(FIT_RT_MAX_S / DATA_BIN_S) + 1,
) * DATA_BIN_S
data_bin_centers_s = 0.5 * (data_bins_s[:-1] + data_bins_s[1:])
data_right_truncation_mask = data_bins_s[1:] <= RIGHT_TRUNCATION_S
model_right_truncation_mask = rt_grid_s <= RIGHT_TRUNCATION_S

n_animals = len(ANIMALS)
n_abls = len(ABLS)
n_abs_ilds = len(ABS_ILDS)
n_data_bins = len(data_bin_centers_s)
n_model_points = len(rt_grid_s)

data_rtd_by_animal = np.full(
    (n_animals, n_abls, n_abs_ilds, n_data_bins),
    np.nan,
)
model_rtd_by_animal = np.full(
    (n_animals, n_abls, n_abs_ilds, n_model_points),
    np.nan,
)
right_truncated_data_rtd_by_animal = np.full_like(data_rtd_by_animal, np.nan)
right_truncated_model_rtd_by_animal = np.full_like(model_rtd_by_animal, np.nan)
condition_data_areas = []
condition_model_areas = []
sign_averaged_data_areas = []
sign_averaged_model_areas = []
right_truncated_sign_averaged_data_areas = []
right_truncated_sign_averaged_model_areas = []
normalization_denominators = []
formula_checks = []
parameter_rows = []
delay_rows = []
condition_trial_rows = []

for animal_idx, animal in enumerate(ANIMALS):
    print(f"\nProcessing LED7/{animal}")
    animal_df = valid_df[valid_df["animal"] == animal].copy()
    animal_rtd_df = rtd_df[rtd_df["animal"] == animal].copy()
    fit_dir = FIT_ROOT / f"{BATCH_NAME}_{animal}"
    posterior_path = fit_dir / "main_fullrank_posterior_samples.npz"
    condition_path = fit_dir / "condition_table.csv"
    abort_path = ABORT_ROOT / f"results_{BATCH_NAME}_animal_{animal}.pkl"
    for required_path in [posterior_path, condition_path, abort_path]:
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    posterior = np.load(posterior_path)
    scalar_names = (
        "rate_lambda",
        "T_0",
        "theta_E",
        "w",
        "del_go",
        "rate_norm_l",
        "alpha",
    )
    required_posterior_keys = scalar_names + ("t_E_aff",)
    missing_keys = [
        key for key in required_posterior_keys if key not in posterior.files
    ]
    if missing_keys:
        raise KeyError(f"Missing posterior keys for LED7/{animal}: {missing_keys}")
    if any(
        not np.isfinite(np.asarray(posterior[key], dtype=float)).all()
        for key in required_posterior_keys
    ):
        raise RuntimeError(f"Non-finite posterior samples for LED7/{animal}.")

    params = {
        key: float(np.mean(np.asarray(posterior[key], dtype=float)))
        for key in scalar_names
    }
    for key, value in params.items():
        parameter_rows.append(
            {
                "batch_name": BATCH_NAME,
                "animal": animal,
                "parameter": key,
                "posterior_mean": value,
            }
        )

    saved_conditions = (
        pd.read_csv(condition_path)
        .sort_values("condition_id")
        .reset_index(drop=True)
    )
    reconstructed_conditions = (
        animal_df[["ABL", "ILD"]]
        .drop_duplicates()
        .sort_values(["ABL", "ILD"])
        .reset_index(drop=True)
    )
    reconstructed_conditions["condition_id"] = np.arange(
        len(reconstructed_conditions),
        dtype=int,
    )
    if len(saved_conditions) != 30 or not np.allclose(
        saved_conditions[["ABL", "ILD", "condition_id"]].to_numpy(dtype=float),
        reconstructed_conditions[["ABL", "ILD", "condition_id"]].to_numpy(
            dtype=float
        ),
        atol=1e-12,
        rtol=0,
    ):
        raise RuntimeError(
            f"Saved and reconstructed condition tables differ for LED7/{animal}."
        )

    delay_means = np.mean(
        np.asarray(posterior["t_E_aff"], dtype=float),
        axis=0,
    )
    if len(delay_means) != len(saved_conditions):
        raise RuntimeError(
            f"Delay vector length does not match condition table for LED7/{animal}."
        )

    for condition, delay in zip(
        saved_conditions.itertuples(index=False),
        delay_means,
    ):
        delay_rows.append(
            {
                "batch_name": BATCH_NAME,
                "animal": animal,
                "ABL": int(condition.ABL),
                "ILD": float(condition.ILD),
                "condition_id": int(condition.condition_id),
                "t_E_aff_s": float(delay),
            }
        )

    with abort_path.open("rb") as handle:
        abort_fit = pickle.load(handle)["vbmc_aborts_results"]
    V_A = float(np.mean(np.asarray(abort_fit["V_A_samples"], dtype=float)))
    theta_A = float(np.mean(np.asarray(abort_fit["theta_A_samples"], dtype=float)))
    t_A_aff = float(np.mean(np.asarray(abort_fit["t_A_aff_samp"], dtype=float)))

    animal_df = animal_df.merge(
        reconstructed_conditions,
        on=["ABL", "ILD"],
        how="left",
        validate="many_to_one",
    )
    if animal_df["condition_id"].isna().any():
        raise RuntimeError(f"Failed to assign condition IDs for LED7/{animal}.")

    condition_abls = saved_conditions["ABL"].to_numpy(dtype=float)
    condition_ilds = saved_conditions["ILD"].to_numpy(dtype=float)
    Z_E = (params["w"] - 0.5) * 2.0 * params["theta_E"]

    relative_evidence_time = jnp.asarray(
        rt_grid_s[None, :] - delay_means[:, None],
        dtype=jnp.float64,
    )
    relative_go_time = relative_evidence_time + params["del_go"]
    abl_jax = jnp.asarray(condition_abls[:, None], dtype=jnp.float64)
    ild_jax = jnp.asarray(condition_ilds[:, None], dtype=jnp.float64)

    c_e_t1 = np.asarray(
        svi_utils.CDF_E_alpha_jax(
            relative_evidence_time,
            1,
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
        + svi_utils.CDF_E_alpha_jax(
            relative_evidence_time,
            -1,
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
    c_e_t2 = np.asarray(
        svi_utils.CDF_E_alpha_jax(
            relative_go_time,
            1,
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
        + svi_utils.CDF_E_alpha_jax(
            relative_go_time,
            -1,
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
    rho_e = np.asarray(
        svi_utils.rho_E_alpha_jax(
            relative_evidence_time,
            1,
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
        + svi_utils.rho_E_alpha_jax(
            relative_evidence_time,
            -1,
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
    proactive_multiplier = (1.0 - c_e_t2) + (c_e_t2 - c_e_t1)
    c_e_lower = c_e_t1[:, 0]
    c_e_upper = c_e_t1[:, -1]

    model_sum_by_condition = np.zeros((30, n_model_points), dtype=float)
    model_n_by_condition = np.zeros(30, dtype=int)

    for start in range(0, len(animal_df), TRIAL_CHUNK_SIZE):
        chunk = animal_df.iloc[start : start + TRIAL_CHUNK_SIZE]
        condition_ids = chunk["condition_id"].to_numpy(dtype=int)
        t_stim = chunk["intended_fix"].to_numpy(dtype=float)

        absolute_t = t_stim[:, None] + rt_grid_s[None, :]
        p_a = np.asarray(
            svi_utils.rho_A_t_jax(
                jnp.asarray(absolute_t - t_A_aff),
                V_A,
                theta_A,
            )
        )
        c_a = np.asarray(
            svi_utils.cum_A_t_jax(
                jnp.asarray(absolute_t - t_A_aff),
                V_A,
                theta_A,
            )
        )

        c_a_lower_raw = np.asarray(
            svi_utils.cum_A_t_jax(
                jnp.asarray(t_stim - t_A_aff),
                V_A,
                theta_A,
            )
        )
        c_a_upper_raw = np.asarray(
            svi_utils.cum_A_t_jax(
                jnp.asarray(t_stim + FIT_RT_MAX_S - t_A_aff),
                V_A,
                theta_A,
            )
        )
        proactive_trunc_denom = float(
            1.0
            - svi_utils.cum_A_t_jax(
                T_TRUNC_S - t_A_aff,
                V_A,
                theta_A,
            )
        )
        c_a_lower = np.where(
            t_stim < T_TRUNC_S,
            0.0,
            c_a_lower_raw / proactive_trunc_denom,
        )
        c_a_upper = np.where(
            t_stim + FIT_RT_MAX_S < T_TRUNC_S,
            0.0,
            c_a_upper_raw / proactive_trunc_denom,
        )

        c_e_lo = c_e_lower[condition_ids]
        c_e_hi = c_e_upper[condition_ids]
        combined_lower = c_a_lower + c_e_lo - c_a_lower * c_e_lo
        combined_upper = c_a_upper + c_e_hi - c_a_upper * c_e_hi
        trunc_denom = combined_upper - combined_lower
        if not np.isfinite(trunc_denom).all() or np.any(trunc_denom <= 0):
            raise RuntimeError(
                f"Invalid SVI truncation denominator for LED7/{animal}."
            )
        normalization_denominators.extend(trunc_denom.tolist())

        numerator = (
            p_a * proactive_multiplier[condition_ids]
            + rho_e[condition_ids] * (1.0 - c_a)
        )
        normalized_density = numerator / trunc_denom[:, None]
        if not np.isfinite(normalized_density).all():
            raise RuntimeError(f"Non-finite model density for LED7/{animal}.")

        for condition_id in np.unique(condition_ids):
            condition_mask = condition_ids == condition_id
            model_sum_by_condition[condition_id] += normalized_density[
                condition_mask
            ].sum(axis=0)
            model_n_by_condition[condition_id] += int(condition_mask.sum())

    reconstructed_counts = (
        animal_df.groupby("condition_id")
        .size()
        .reindex(range(30), fill_value=0)
        .to_numpy(dtype=int)
    )
    if not np.array_equal(model_n_by_condition, reconstructed_counts):
        raise RuntimeError(
            f"Model condition counts do not match data rows for LED7/{animal}."
        )

    data_rtd_by_signed_condition = {}
    model_rtd_by_signed_condition = {}
    for condition in saved_conditions.itertuples(index=False):
        condition_id = int(condition.condition_id)
        abl = int(condition.ABL)
        signed_ild = float(condition.ILD)
        condition_rtd_rows = animal_rtd_df[
            animal_rtd_df["ABL"].eq(abl)
            & np.isclose(animal_rtd_df["ILD"], signed_ild)
        ]
        condition_rts = condition_rtd_rows["RTwrtStim"].to_numpy(dtype=float)

        counts, _ = np.histogram(condition_rts, bins=data_bins_s)
        data_density = normalize_histogram_density(
            counts.astype(float) / DATA_BIN_S
        )
        model_density = normalize_continuous_density(
            model_sum_by_condition[condition_id]
            / model_n_by_condition[condition_id]
        )
        data_rtd_by_signed_condition[(abl, signed_ild)] = data_density
        model_rtd_by_signed_condition[(abl, signed_ild)] = model_density

        condition_data_areas.append(
            float(np.sum(data_density * np.diff(data_bins_s)))
        )
        condition_model_areas.append(
            float(trapezoid(model_density, rt_grid_s))
        )
        condition_trial_rows.append(
            {
                "batch_name": BATCH_NAME,
                "animal": animal,
                "ABL": abl,
                "ILD": signed_ild,
                "n_fit_trials": int(
                    model_n_by_condition[condition_id]
                ),
                "n_rtd_trials": len(condition_rts),
                "n_abort_event_4": int(
                    condition_rtd_rows["abort_event"].eq(4).sum()
                ),
            }
        )

    for abl_idx, abl in enumerate(ABLS):
        for abs_idx, abs_ild in enumerate(ABS_ILDS):
            signed_keys = [(abl, -abs_ild), (abl, abs_ild)]
            if any(key not in data_rtd_by_signed_condition for key in signed_keys):
                raise RuntimeError(
                    f"Missing signed condition for LED7/{animal}, "
                    f"ABL={abl}, |ILD|={abs_ild}."
                )

            data_sign_mean = normalize_histogram_density(
                np.mean(
                    [data_rtd_by_signed_condition[key] for key in signed_keys],
                    axis=0,
                )
            )
            model_sign_mean = normalize_continuous_density(
                np.mean(
                    [model_rtd_by_signed_condition[key] for key in signed_keys],
                    axis=0,
                )
            )
            right_truncated_data_sign_mean = normalize_histogram_density(
                np.mean(
                    [
                        normalize_histogram_density(
                            np.where(
                                data_right_truncation_mask,
                                data_rtd_by_signed_condition[key],
                                0,
                            )
                        )
                        for key in signed_keys
                    ],
                    axis=0,
                )
            )
            right_truncated_model_sign_mean = (
                normalize_right_truncated_continuous_density(
                    np.mean(
                        [
                            normalize_right_truncated_continuous_density(
                                model_rtd_by_signed_condition[key]
                            )
                            for key in signed_keys
                        ],
                        axis=0,
                    )
                )
            )
            data_rtd_by_animal[animal_idx, abl_idx, abs_idx] = data_sign_mean
            model_rtd_by_animal[animal_idx, abl_idx, abs_idx] = model_sign_mean
            right_truncated_data_rtd_by_animal[
                animal_idx, abl_idx, abs_idx
            ] = right_truncated_data_sign_mean
            right_truncated_model_rtd_by_animal[
                animal_idx, abl_idx, abs_idx
            ] = right_truncated_model_sign_mean
            sign_averaged_data_areas.append(
                float(np.sum(data_sign_mean * np.diff(data_bins_s)))
            )
            sign_averaged_model_areas.append(
                float(trapezoid(model_sign_mean, rt_grid_s))
            )
            right_truncated_sign_averaged_data_areas.append(
                float(
                    np.sum(
                        right_truncated_data_sign_mean
                        * np.diff(data_bins_s)
                    )
                )
            )
            right_truncated_sign_averaged_model_areas.append(
                float(
                    trapezoid(
                        right_truncated_model_sign_mean[
                            model_right_truncation_mask
                        ],
                        rt_grid_s[model_right_truncation_mask],
                    )
                )
            )

    check_trial = animal_df.iloc[0]
    check_condition_id = int(check_trial["condition_id"])
    check_t_stim = float(check_trial["intended_fix"])
    direct_up = np.asarray(
        svi_utils.up_or_down_alpha_jax(
            jnp.asarray(check_t_stim + rt_grid_s),
            1,
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
            delay_means[check_condition_id],
            params["del_go"],
            params["rate_norm_l"],
            params["alpha"],
            K_MAX,
        )
    )
    direct_down = np.asarray(
        svi_utils.up_or_down_alpha_jax(
            jnp.asarray(check_t_stim + rt_grid_s),
            -1,
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
            delay_means[check_condition_id],
            params["del_go"],
            params["rate_norm_l"],
            params["alpha"],
            K_MAX,
        )
    )
    check_p_a = np.asarray(
        svi_utils.rho_A_t_jax(
            jnp.asarray(check_t_stim + rt_grid_s - t_A_aff),
            V_A,
            theta_A,
        )
    )
    check_c_a = np.asarray(
        svi_utils.cum_A_t_jax(
            jnp.asarray(check_t_stim + rt_grid_s - t_A_aff),
            V_A,
            theta_A,
        )
    )
    collapsed_numerator = (
        check_p_a * proactive_multiplier[check_condition_id]
        + rho_e[check_condition_id] * (1.0 - check_c_a)
    )
    max_abs_difference = float(
        np.max(np.abs(direct_up + direct_down - collapsed_numerator))
    )
    scale = max(1.0, float(np.max(np.abs(direct_up + direct_down))))
    formula_checks.append(max_abs_difference / scale)
    if formula_checks[-1] > 1e-10:
        raise RuntimeError(
            f"Choice-collapsed formula check failed for LED7/{animal}: "
            f"{formula_checks[-1]:.3g}."
        )

    print(
        f"  {len(animal_df):,} rows; 30 fitted delays; "
        f"formula check={formula_checks[-1]:.2e}"
    )

delay_df = pd.DataFrame(delay_rows)
delay_df["t_E_aff_ms"] = delay_df["t_E_aff_s"] * 1e3
delay_summary_df = (
    delay_df.groupby(["ABL", "ILD"], as_index=False)
    .agg(
        mean_ms=("t_E_aff_ms", "mean"),
        std_ms=("t_E_aff_ms", "std"),
        n_animals=("animal", "nunique"),
    )
    .sort_values(["ABL", "ILD"])
    .reset_index(drop=True)
)
delay_summary_df["sem_ms"] = (
    delay_summary_df["std_ms"]
    / np.sqrt(delay_summary_df["n_animals"])
)
if len(delay_summary_df) != len(ABLS) * len(SIGNED_ILDS):
    raise RuntimeError(
        f"Expected 30 delay summary rows, found {len(delay_summary_df)}."
    )
if not np.all(delay_summary_df["n_animals"].to_numpy() == n_animals):
    raise RuntimeError(
        "Every ABL/ILD delay point should contain all six LED7 animals."
    )


# %%
# =============================================================================
# Equal-animal and within-animal ABL averages
# =============================================================================
if not np.isfinite(data_rtd_by_animal).all():
    raise RuntimeError("Non-finite data RTDs after sign averaging.")
if not np.isfinite(model_rtd_by_animal).all():
    raise RuntimeError("Non-finite model RTDs after sign averaging.")
if not np.isfinite(right_truncated_data_rtd_by_animal).all():
    raise RuntimeError("Non-finite right-truncated data RTDs.")
if not np.isfinite(right_truncated_model_rtd_by_animal).all():
    raise RuntimeError("Non-finite right-truncated model RTDs.")

data_mean, data_sem, data_n = mean_sem(data_rtd_by_animal, axis=0)
model_mean, model_sem, model_n = mean_sem(model_rtd_by_animal, axis=0)
(
    right_truncated_data_mean,
    right_truncated_data_sem,
    right_truncated_data_n,
) = mean_sem(right_truncated_data_rtd_by_animal, axis=0)
(
    right_truncated_model_mean,
    right_truncated_model_sem,
    right_truncated_model_n,
) = mean_sem(right_truncated_model_rtd_by_animal, axis=0)

# Average ABLs inside each animal first so the last-row SEM has six independent
# animal contributors rather than 18 repeated animal-ABL entries.
data_abl_average_by_animal = np.mean(data_rtd_by_animal, axis=1)
model_abl_average_by_animal = np.mean(model_rtd_by_animal, axis=1)
right_truncated_data_abl_average_by_animal = np.mean(
    right_truncated_data_rtd_by_animal,
    axis=1,
)
right_truncated_model_abl_average_by_animal = np.mean(
    right_truncated_model_rtd_by_animal,
    axis=1,
)
for animal_idx in range(n_animals):
    for abs_idx in range(n_abs_ilds):
        data_abl_average_by_animal[animal_idx, abs_idx] = (
            normalize_histogram_density(
                data_abl_average_by_animal[animal_idx, abs_idx]
            )
        )
        model_abl_average_by_animal[animal_idx, abs_idx] = (
            normalize_continuous_density(
                model_abl_average_by_animal[animal_idx, abs_idx]
            )
        )
        right_truncated_data_abl_average_by_animal[animal_idx, abs_idx] = (
            normalize_histogram_density(
                right_truncated_data_abl_average_by_animal[
                    animal_idx, abs_idx
                ]
            )
        )
        right_truncated_model_abl_average_by_animal[animal_idx, abs_idx] = (
            normalize_right_truncated_continuous_density(
                right_truncated_model_abl_average_by_animal[
                    animal_idx, abs_idx
                ]
            )
        )

# Average the five |ILD| values inside each animal and ABL. The grand average
# gives equal weight to all 15 ABL-by-|ILD| curves inside each animal.
data_ild_average_by_animal = np.mean(data_rtd_by_animal, axis=2)
model_ild_average_by_animal = np.mean(model_rtd_by_animal, axis=2)
right_truncated_data_ild_average_by_animal = np.mean(
    right_truncated_data_rtd_by_animal,
    axis=2,
)
right_truncated_model_ild_average_by_animal = np.mean(
    right_truncated_model_rtd_by_animal,
    axis=2,
)
data_grand_average_by_animal = np.mean(data_rtd_by_animal, axis=(1, 2))
model_grand_average_by_animal = np.mean(model_rtd_by_animal, axis=(1, 2))
right_truncated_data_grand_average_by_animal = np.mean(
    right_truncated_data_rtd_by_animal,
    axis=(1, 2),
)
right_truncated_model_grand_average_by_animal = np.mean(
    right_truncated_model_rtd_by_animal,
    axis=(1, 2),
)

for animal_idx in range(n_animals):
    for abl_idx in range(n_abls):
        data_ild_average_by_animal[animal_idx, abl_idx] = (
            normalize_histogram_density(
                data_ild_average_by_animal[animal_idx, abl_idx]
            )
        )
        model_ild_average_by_animal[animal_idx, abl_idx] = (
            normalize_continuous_density(
                model_ild_average_by_animal[animal_idx, abl_idx]
            )
        )
        right_truncated_data_ild_average_by_animal[
            animal_idx, abl_idx
        ] = normalize_histogram_density(
            right_truncated_data_ild_average_by_animal[
                animal_idx, abl_idx
            ]
        )
        right_truncated_model_ild_average_by_animal[
            animal_idx, abl_idx
        ] = normalize_right_truncated_continuous_density(
            right_truncated_model_ild_average_by_animal[
                animal_idx, abl_idx
            ]
        )

    data_grand_average_by_animal[animal_idx] = (
        normalize_histogram_density(
            data_grand_average_by_animal[animal_idx]
        )
    )
    model_grand_average_by_animal[animal_idx] = (
        normalize_continuous_density(
            model_grand_average_by_animal[animal_idx]
        )
    )
    right_truncated_data_grand_average_by_animal[animal_idx] = (
        normalize_histogram_density(
            right_truncated_data_grand_average_by_animal[animal_idx]
        )
    )
    right_truncated_model_grand_average_by_animal[animal_idx] = (
        normalize_right_truncated_continuous_density(
            right_truncated_model_grand_average_by_animal[animal_idx]
        )
    )

data_abl_mean, data_abl_sem, data_abl_n = mean_sem(
    data_abl_average_by_animal,
    axis=0,
)
model_abl_mean, model_abl_sem, model_abl_n = mean_sem(
    model_abl_average_by_animal,
    axis=0,
)
(
    right_truncated_data_abl_mean,
    right_truncated_data_abl_sem,
    right_truncated_data_abl_n,
) = mean_sem(right_truncated_data_abl_average_by_animal, axis=0)
(
    right_truncated_model_abl_mean,
    right_truncated_model_abl_sem,
    right_truncated_model_abl_n,
) = mean_sem(right_truncated_model_abl_average_by_animal, axis=0)
data_ild_mean, data_ild_sem, data_ild_n = mean_sem(
    data_ild_average_by_animal,
    axis=0,
)
model_ild_mean, model_ild_sem, model_ild_n = mean_sem(
    model_ild_average_by_animal,
    axis=0,
)
(
    right_truncated_data_ild_mean,
    right_truncated_data_ild_sem,
    right_truncated_data_ild_n,
) = mean_sem(right_truncated_data_ild_average_by_animal, axis=0)
(
    right_truncated_model_ild_mean,
    right_truncated_model_ild_sem,
    right_truncated_model_ild_n,
) = mean_sem(right_truncated_model_ild_average_by_animal, axis=0)
data_grand_mean, data_grand_sem, data_grand_n = mean_sem(
    data_grand_average_by_animal,
    axis=0,
)
model_grand_mean, model_grand_sem, model_grand_n = mean_sem(
    model_grand_average_by_animal,
    axis=0,
)
(
    right_truncated_data_grand_mean,
    right_truncated_data_grand_sem,
    right_truncated_data_grand_n,
) = mean_sem(right_truncated_data_grand_average_by_animal, axis=0)
(
    right_truncated_model_grand_mean,
    right_truncated_model_grand_sem,
    right_truncated_model_grand_n,
) = mean_sem(right_truncated_model_grand_average_by_animal, axis=0)

if not np.all(data_n == n_animals):
    raise RuntimeError(f"Expected six data contributors, found {data_n}.")
if not np.all(model_n == n_animals):
    raise RuntimeError(f"Expected six model contributors, found {model_n}.")
if not np.all(data_abl_n == n_animals):
    raise RuntimeError(
        f"Expected six ABL-averaged data contributors, found {data_abl_n}."
    )
if not np.all(model_abl_n == n_animals):
    raise RuntimeError(
        f"Expected six ABL-averaged model contributors, found {model_abl_n}."
    )
if not np.all(right_truncated_data_n == n_animals):
    raise RuntimeError(
        "Expected six right-truncated data contributors, "
        f"found {right_truncated_data_n}."
    )
if not np.all(right_truncated_model_n == n_animals):
    raise RuntimeError(
        "Expected six right-truncated model contributors, "
        f"found {right_truncated_model_n}."
    )
if not np.all(right_truncated_data_abl_n == n_animals):
    raise RuntimeError(
        "Expected six ABL-averaged right-truncated data contributors, "
        f"found {right_truncated_data_abl_n}."
    )
if not np.all(right_truncated_model_abl_n == n_animals):
    raise RuntimeError(
        "Expected six ABL-averaged right-truncated model contributors, "
        f"found {right_truncated_model_abl_n}."
    )
for label, counts in {
    "ILD-averaged data": data_ild_n,
    "ILD-averaged model": model_ild_n,
    "right-truncated ILD-averaged data": right_truncated_data_ild_n,
    "right-truncated ILD-averaged model": right_truncated_model_ild_n,
    "grand-average data": data_grand_n,
    "grand-average model": model_grand_n,
    "right-truncated grand-average data": right_truncated_data_grand_n,
    "right-truncated grand-average model": right_truncated_model_grand_n,
}.items():
    if not np.all(counts == n_animals):
        raise RuntimeError(
            f"Expected six {label} contributors, found {counts}."
        )

all_area_checks = {
    "condition_data": np.asarray(condition_data_areas),
    "condition_model": np.asarray(condition_model_areas),
    "sign_averaged_data": np.asarray(sign_averaged_data_areas),
    "sign_averaged_model": np.asarray(sign_averaged_model_areas),
    "right_truncated_sign_averaged_data": np.asarray(
        right_truncated_sign_averaged_data_areas
    ),
    "right_truncated_sign_averaged_model": np.asarray(
        right_truncated_sign_averaged_model_areas
    ),
    "animal_abl_average_data": np.sum(
        data_abl_average_by_animal * np.diff(data_bins_s),
        axis=-1,
    ),
    "animal_abl_average_model": trapezoid(
        model_abl_average_by_animal,
        rt_grid_s,
        axis=-1,
    ),
    "across_animal_data_mean": np.sum(
        data_mean * np.diff(data_bins_s),
        axis=-1,
    ),
    "across_animal_model_mean": trapezoid(
        model_mean,
        rt_grid_s,
        axis=-1,
    ),
    "abl_average_data_mean": np.sum(
        data_abl_mean * np.diff(data_bins_s),
        axis=-1,
    ),
    "abl_average_model_mean": trapezoid(
        model_abl_mean,
        rt_grid_s,
        axis=-1,
    ),
    "animal_ild_average_data": np.sum(
        data_ild_average_by_animal * np.diff(data_bins_s),
        axis=-1,
    ),
    "animal_ild_average_model": trapezoid(
        model_ild_average_by_animal,
        rt_grid_s,
        axis=-1,
    ),
    "ild_average_data_mean": np.sum(
        data_ild_mean * np.diff(data_bins_s),
        axis=-1,
    ),
    "ild_average_model_mean": trapezoid(
        model_ild_mean,
        rt_grid_s,
        axis=-1,
    ),
    "animal_grand_average_data": np.sum(
        data_grand_average_by_animal * np.diff(data_bins_s),
        axis=-1,
    ),
    "animal_grand_average_model": trapezoid(
        model_grand_average_by_animal,
        rt_grid_s,
        axis=-1,
    ),
    "grand_average_data_mean": np.sum(
        data_grand_mean * np.diff(data_bins_s),
    ),
    "grand_average_model_mean": trapezoid(
        model_grand_mean,
        rt_grid_s,
    ),
    "right_truncated_animal_abl_average_data": np.sum(
        right_truncated_data_abl_average_by_animal * np.diff(data_bins_s),
        axis=-1,
    ),
    "right_truncated_animal_abl_average_model": trapezoid(
        right_truncated_model_abl_average_by_animal[
            ..., model_right_truncation_mask
        ],
        rt_grid_s[model_right_truncation_mask],
        axis=-1,
    ),
    "right_truncated_across_animal_data_mean": np.sum(
        right_truncated_data_mean * np.diff(data_bins_s),
        axis=-1,
    ),
    "right_truncated_across_animal_model_mean": trapezoid(
        right_truncated_model_mean[..., model_right_truncation_mask],
        rt_grid_s[model_right_truncation_mask],
        axis=-1,
    ),
    "right_truncated_abl_average_data_mean": np.sum(
        right_truncated_data_abl_mean * np.diff(data_bins_s),
        axis=-1,
    ),
    "right_truncated_abl_average_model_mean": trapezoid(
        right_truncated_model_abl_mean[..., model_right_truncation_mask],
        rt_grid_s[model_right_truncation_mask],
        axis=-1,
    ),
    "right_truncated_animal_ild_average_data": np.sum(
        right_truncated_data_ild_average_by_animal * np.diff(data_bins_s),
        axis=-1,
    ),
    "right_truncated_animal_ild_average_model": trapezoid(
        right_truncated_model_ild_average_by_animal[
            ..., model_right_truncation_mask
        ],
        rt_grid_s[model_right_truncation_mask],
        axis=-1,
    ),
    "right_truncated_ild_average_data_mean": np.sum(
        right_truncated_data_ild_mean * np.diff(data_bins_s),
        axis=-1,
    ),
    "right_truncated_ild_average_model_mean": trapezoid(
        right_truncated_model_ild_mean[..., model_right_truncation_mask],
        rt_grid_s[model_right_truncation_mask],
        axis=-1,
    ),
    "right_truncated_animal_grand_average_data": np.sum(
        right_truncated_data_grand_average_by_animal
        * np.diff(data_bins_s),
        axis=-1,
    ),
    "right_truncated_animal_grand_average_model": trapezoid(
        right_truncated_model_grand_average_by_animal[
            ..., model_right_truncation_mask
        ],
        rt_grid_s[model_right_truncation_mask],
        axis=-1,
    ),
    "right_truncated_grand_average_data_mean": np.sum(
        right_truncated_data_grand_mean * np.diff(data_bins_s),
    ),
    "right_truncated_grand_average_model_mean": trapezoid(
        right_truncated_model_grand_mean[model_right_truncation_mask],
        rt_grid_s[model_right_truncation_mask],
    ),
}
for label, areas in all_area_checks.items():
    if not np.allclose(areas, 1.0, atol=1e-10, rtol=0):
        raise RuntimeError(
            f"{label} RTD areas are not one: "
            f"min={np.min(areas):.12f}, max={np.max(areas):.12f}."
        )

print("\nContributor counts")
print(f"  Per ABL/|ILD| data: {np.unique(data_n).tolist()}")
print(f"  Per ABL/|ILD| model: {np.unique(model_n).tolist()}")
print(f"  ABL-averaged data: {np.unique(data_abl_n).tolist()}")
print(f"  ABL-averaged model: {np.unique(model_abl_n).tolist()}")
print(f"  |ILD|-averaged data: {np.unique(data_ild_n).tolist()}")
print(f"  |ILD|-averaged model: {np.unique(model_ild_n).tolist()}")
print(f"  Grand-average data: {np.unique(data_grand_n).tolist()}")
print(f"  Grand-average model: {np.unique(model_grand_n).tolist()}")
print(
    "  Right-truncated per ABL/|ILD| data: "
    f"{np.unique(right_truncated_data_n).tolist()}"
)
print(
    "  Right-truncated per ABL/|ILD| model: "
    f"{np.unique(right_truncated_model_n).tolist()}"
)
print(
    "  Right-truncated |ILD|-averaged data: "
    f"{np.unique(right_truncated_data_ild_n).tolist()}"
)
print(
    "  Right-truncated |ILD|-averaged model: "
    f"{np.unique(right_truncated_model_ild_n).tolist()}"
)
print(
    "  Right-truncated grand-average data: "
    f"{np.unique(right_truncated_data_grand_n).tolist()}"
)
print(
    "  Right-truncated grand-average model: "
    f"{np.unique(right_truncated_model_grand_n).tolist()}"
)
print(
    "SVI truncation denominators: "
    f"min={np.min(normalization_denominators):.6g}, "
    f"max={np.max(normalization_denominators):.6g}"
)
for label, areas in all_area_checks.items():
    print(
        f"  {label} areas: "
        f"{np.min(areas):.12f}--{np.max(areas):.12f}"
    )


# %%
# =============================================================================
# Save the full-window payload before plotting either x-axis view
# =============================================================================
payload = {
    "batch_name": BATCH_NAME,
    "animals": ANIMALS,
    "abls": ABLS,
    "signed_ilds": SIGNED_ILDS,
    "abs_ilds": ABS_ILDS,
    "rt_grid_s": rt_grid_s,
    "data_bins_s": data_bins_s,
    "data_bin_centers_s": data_bin_centers_s,
    "data_bin_width_s": DATA_BIN_S,
    "model_step_s": MODEL_STEP_S,
    "data_rtd_by_animal": data_rtd_by_animal,
    "model_rtd_by_animal": model_rtd_by_animal,
    "data_mean": data_mean,
    "data_sem": data_sem,
    "data_n": data_n,
    "model_mean": model_mean,
    "model_sem": model_sem,
    "model_n": model_n,
    "data_abl_average_by_animal": data_abl_average_by_animal,
    "model_abl_average_by_animal": model_abl_average_by_animal,
    "data_abl_mean": data_abl_mean,
    "data_abl_sem": data_abl_sem,
    "data_abl_n": data_abl_n,
    "model_abl_mean": model_abl_mean,
    "model_abl_sem": model_abl_sem,
    "model_abl_n": model_abl_n,
    "data_ild_average_by_animal": data_ild_average_by_animal,
    "model_ild_average_by_animal": model_ild_average_by_animal,
    "data_ild_mean": data_ild_mean,
    "data_ild_sem": data_ild_sem,
    "data_ild_n": data_ild_n,
    "model_ild_mean": model_ild_mean,
    "model_ild_sem": model_ild_sem,
    "model_ild_n": model_ild_n,
    "data_grand_average_by_animal": data_grand_average_by_animal,
    "model_grand_average_by_animal": model_grand_average_by_animal,
    "data_grand_mean": data_grand_mean,
    "data_grand_sem": data_grand_sem,
    "data_grand_n": data_grand_n,
    "model_grand_mean": model_grand_mean,
    "model_grand_sem": model_grand_sem,
    "model_grand_n": model_grand_n,
    "right_truncation_s": RIGHT_TRUNCATION_S,
    "right_truncated_data_rtd_by_animal": right_truncated_data_rtd_by_animal,
    "right_truncated_model_rtd_by_animal": right_truncated_model_rtd_by_animal,
    "right_truncated_data_mean": right_truncated_data_mean,
    "right_truncated_data_sem": right_truncated_data_sem,
    "right_truncated_data_n": right_truncated_data_n,
    "right_truncated_model_mean": right_truncated_model_mean,
    "right_truncated_model_sem": right_truncated_model_sem,
    "right_truncated_model_n": right_truncated_model_n,
    "right_truncated_data_abl_average_by_animal": (
        right_truncated_data_abl_average_by_animal
    ),
    "right_truncated_model_abl_average_by_animal": (
        right_truncated_model_abl_average_by_animal
    ),
    "right_truncated_data_abl_mean": right_truncated_data_abl_mean,
    "right_truncated_data_abl_sem": right_truncated_data_abl_sem,
    "right_truncated_data_abl_n": right_truncated_data_abl_n,
    "right_truncated_model_abl_mean": right_truncated_model_abl_mean,
    "right_truncated_model_abl_sem": right_truncated_model_abl_sem,
    "right_truncated_model_abl_n": right_truncated_model_abl_n,
    "right_truncated_data_ild_average_by_animal": (
        right_truncated_data_ild_average_by_animal
    ),
    "right_truncated_model_ild_average_by_animal": (
        right_truncated_model_ild_average_by_animal
    ),
    "right_truncated_data_ild_mean": right_truncated_data_ild_mean,
    "right_truncated_data_ild_sem": right_truncated_data_ild_sem,
    "right_truncated_data_ild_n": right_truncated_data_ild_n,
    "right_truncated_model_ild_mean": right_truncated_model_ild_mean,
    "right_truncated_model_ild_sem": right_truncated_model_ild_sem,
    "right_truncated_model_ild_n": right_truncated_model_ild_n,
    "right_truncated_data_grand_average_by_animal": (
        right_truncated_data_grand_average_by_animal
    ),
    "right_truncated_model_grand_average_by_animal": (
        right_truncated_model_grand_average_by_animal
    ),
    "right_truncated_data_grand_mean": right_truncated_data_grand_mean,
    "right_truncated_data_grand_sem": right_truncated_data_grand_sem,
    "right_truncated_data_grand_n": right_truncated_data_grand_n,
    "right_truncated_model_grand_mean": right_truncated_model_grand_mean,
    "right_truncated_model_grand_sem": right_truncated_model_grand_sem,
    "right_truncated_model_grand_n": right_truncated_model_grand_n,
    "area_checks": all_area_checks,
    "formula_check_relative_max": np.asarray(formula_checks),
    "truncation_denominator_min": float(np.min(normalization_denominators)),
    "truncation_denominator_max": float(np.max(normalization_denominators)),
    "parameter_rows": pd.DataFrame(parameter_rows),
    "delay_rows": delay_df,
    "delay_summary": delay_summary_df,
    "condition_trial_rows": pd.DataFrame(condition_trial_rows),
    "data_csv": str(DATA_CSV.relative_to(REPO_ROOT)),
    "rtd_data_csv": str(RTD_DATA_CSV.relative_to(REPO_ROOT)),
    "rtd_abort_events": RTD_ABORT_EVENTS,
    "rtd_abort_total_fix_truncation_s": T_TRUNC_S,
    "rtd_abort_event_counts_before_truncation": (
        event_counts_before_truncation
    ),
    "rtd_abort_event_counts_after_truncation": (
        event_counts_after_truncation
    ),
    "fit_root": str(FIT_ROOT.relative_to(REPO_ROOT)),
    "abort_root": str(ABORT_ROOT.relative_to(REPO_ROOT)),
    "sign_average": (
        "Normalize each signed-condition RTD, then average -ILD and +ILD "
        "equally within animal."
    ),
    "animal_average": "Equal mean across the six LED7 animals.",
    "data_trial_pool": (
        "Data RTDs use successful trials plus abort_event 3/4 from raw "
        "out_LED.csv. Abort rows must have TotalFixTime >= 300 ms; all "
        "rows must have RTwrtStim in 0--1 s. The model remains evaluated "
        "on the successful-trial SVI fit pool."
    ),
    "abl_average": (
        "Average ABL 20/40/60 within each animal, then calculate mean and "
        "SEM across six animals."
    ),
    "ild_average": (
        "Average |ILD| 1/2/4/8/16 equally within each animal and ABL, "
        "then calculate mean and SEM across six animals."
    ),
    "grand_average": (
        "Average all 15 ABL-by-|ILD| RTDs equally within each animal, "
        "then calculate mean and SEM across six animals."
    ),
    "plot_windows_ms": {
        "overview": (0, 600),
        "zoom": (0, 160),
        "zoom_120ms": (0, 120),
        "right_truncated_120ms": (0, 120),
    },
    "display_policy": (
        "The overview and x-axis-only views reuse the full 0--1 s "
        "normalized densities and change only the displayed x-axis."
    ),
    "right_truncation_policy": (
        "Within each signed ABL/ILD condition, remove RT density after "
        "120 ms and renormalize over 0--120 ms before equal-sign, "
        "equal-animal, and within-animal ABL averaging."
    ),
}
with OUTPUT_PKL.open("wb") as handle:
    pickle.dump(payload, handle)


# %%
# =============================================================================
# Plot full and zoomed views from the same arrays
# =============================================================================
global_y_max = max(
    float(np.nanmax(data_mean + data_sem)),
    float(np.nanmax(model_mean + model_sem)),
    float(np.nanmax(data_abl_mean + data_abl_sem)),
    float(np.nanmax(model_abl_mean + model_abl_sem)),
    float(np.nanmax(data_ild_mean + data_ild_sem)),
    float(np.nanmax(model_ild_mean + model_ild_sem)),
    float(np.nanmax(data_grand_mean + data_grand_sem)),
    float(np.nanmax(model_grand_mean + model_grand_sem)),
)

column_labels = [
    f"|ILD| = {abs_ild:g}" for abs_ild in ABS_ILDS
] + ["|ILD| average"]

row_data = []
for abl_idx, abl in enumerate(ABLS):
    row_data.append(
        {
            "label": f"ABL {abl}",
            "data_mean": np.concatenate(
                [data_mean[abl_idx], data_ild_mean[abl_idx][None, :]],
                axis=0,
            ),
            "data_sem": np.concatenate(
                [data_sem[abl_idx], data_ild_sem[abl_idx][None, :]],
                axis=0,
            ),
            "model_mean": np.concatenate(
                [model_mean[abl_idx], model_ild_mean[abl_idx][None, :]],
                axis=0,
            ),
            "model_sem": np.concatenate(
                [model_sem[abl_idx], model_ild_sem[abl_idx][None, :]],
                axis=0,
            ),
        }
    )
row_data.append(
    {
        "label": "ABL average",
        "data_mean": np.concatenate(
            [data_abl_mean, data_grand_mean[None, :]],
            axis=0,
        ),
        "data_sem": np.concatenate(
            [data_abl_sem, data_grand_sem[None, :]],
            axis=0,
        ),
        "model_mean": np.concatenate(
            [model_abl_mean, model_grand_mean[None, :]],
            axis=0,
        ),
        "model_sem": np.concatenate(
            [model_abl_sem, model_grand_sem[None, :]],
            axis=0,
        ),
    }
)

right_truncated_global_y_max = max(
    float(np.nanmax(right_truncated_data_mean + right_truncated_data_sem)),
    float(np.nanmax(right_truncated_model_mean + right_truncated_model_sem)),
    float(
        np.nanmax(
            right_truncated_data_abl_mean + right_truncated_data_abl_sem
        )
    ),
    float(
        np.nanmax(
            right_truncated_model_abl_mean + right_truncated_model_abl_sem
        )
    ),
    float(
        np.nanmax(
            right_truncated_data_ild_mean + right_truncated_data_ild_sem
        )
    ),
    float(
        np.nanmax(
            right_truncated_model_ild_mean + right_truncated_model_ild_sem
        )
    ),
    float(
        np.nanmax(
            right_truncated_data_grand_mean
            + right_truncated_data_grand_sem
        )
    ),
    float(
        np.nanmax(
            right_truncated_model_grand_mean
            + right_truncated_model_grand_sem
        )
    ),
)

right_truncated_row_data = []
for abl_idx, abl in enumerate(ABLS):
    right_truncated_row_data.append(
        {
            "label": f"ABL {abl}",
            "data_mean": np.concatenate(
                [
                    right_truncated_data_mean[abl_idx],
                    right_truncated_data_ild_mean[abl_idx][None, :],
                ],
                axis=0,
            ),
            "data_sem": np.concatenate(
                [
                    right_truncated_data_sem[abl_idx],
                    right_truncated_data_ild_sem[abl_idx][None, :],
                ],
                axis=0,
            ),
            "model_mean": np.concatenate(
                [
                    right_truncated_model_mean[abl_idx],
                    right_truncated_model_ild_mean[abl_idx][None, :],
                ],
                axis=0,
            ),
            "model_sem": np.concatenate(
                [
                    right_truncated_model_sem[abl_idx],
                    right_truncated_model_ild_sem[abl_idx][None, :],
                ],
                axis=0,
            ),
        }
    )
right_truncated_row_data.append(
    {
        "label": "ABL average",
        "data_mean": np.concatenate(
            [
                right_truncated_data_abl_mean,
                right_truncated_data_grand_mean[None, :],
            ],
            axis=0,
        ),
        "data_sem": np.concatenate(
            [
                right_truncated_data_abl_sem,
                right_truncated_data_grand_sem[None, :],
            ],
            axis=0,
        ),
        "model_mean": np.concatenate(
            [
                right_truncated_model_abl_mean,
                right_truncated_model_grand_mean[None, :],
            ],
            axis=0,
        ),
        "model_sem": np.concatenate(
            [
                right_truncated_model_abl_sem,
                right_truncated_model_grand_sem[None, :],
            ],
            axis=0,
        ),
    }
)


def save_rtd_grid(
    output_path,
    xlim_ms,
    xticks_ms,
    title,
    plot_rows,
    y_max,
    truncation_ms=None,
):
    fig, axes = plt.subplots(
        4,
        len(column_labels),
        figsize=(19.8, 10.3),
        sharex=True,
        sharey=True,
    )

    for row_idx, row in enumerate(plot_rows):
        for col_idx, column_label in enumerate(column_labels):
            ax = axes[row_idx, col_idx]
            data_curve = row["data_mean"][col_idx]
            data_error = row["data_sem"][col_idx]
            model_curve = row["model_mean"][col_idx]
            model_error = row["model_sem"][col_idx]

            ax.fill_between(
                data_bin_centers_s * 1e3,
                np.maximum(data_curve - data_error, 0),
                data_curve + data_error,
                step="mid",
                color="black",
                alpha=0.12,
                linewidth=0,
                zorder=1,
            )
            ax.stairs(
                data_curve,
                data_bins_s * 1e3,
                color="black",
                linewidth=0.9,
                alpha=0.7,
                label="Data",
                zorder=3,
            )
            ax.fill_between(
                rt_grid_s * 1e3,
                np.maximum(model_curve - model_error, 0),
                model_curve + model_error,
                color="#0072B2",
                alpha=0.16,
                linewidth=0,
                zorder=1,
            )
            ax.plot(
                rt_grid_s * 1e3,
                model_curve,
                color="#0072B2",
                linewidth=1.6,
                label="NPL+alpha SVI",
                zorder=2,
            )
            if truncation_ms is not None:
                ax.axvline(
                    truncation_ms,
                    color="0.45",
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.7,
                    zorder=0,
                )

            if row_idx == 0:
                ax.set_title(column_label, fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(
                    f"{row['label']}\nDensity (s$^{{-1}}$)",
                    fontsize=10,
                )
            if row_idx == len(plot_rows) - 1:
                ax.set_xlabel(r"RT - $t_{stim}$ (ms)", fontsize=10)

            ax.set_xlim(*xlim_ms)
            ax.set_xticks(xticks_ms)
            ax.set_ylim(0, y_max * 1.06)
            ax.tick_params(axis="both", labelsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=0.9,
            alpha=0.7,
            label="Data mean +/- SEM",
        ),
        Line2D(
            [0],
            [0],
            color="#0072B2",
            linewidth=1.6,
            label="NPL+alpha SVI mean +/- SEM",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    fig.suptitle(title, fontsize=13, y=0.995)
    fig.subplots_adjust(
        left=0.065,
        right=0.995,
        bottom=0.065,
        top=0.90,
        wspace=0.16,
        hspace=0.18,
    )
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    print(f"Saved figure: {output_path}")


save_rtd_grid(
    OVERVIEW_PNG,
    (0, 600),
    (0, 300, 600),
    "LED7 direct NPL+alpha SVI RTDs by ABL and |ILD| (0--600 ms x-axis)",
    row_data,
    global_y_max,
)
save_rtd_grid(
    ZOOM_PNG,
    (0, 160),
    (0, 80, 160),
    "LED7 direct NPL+alpha SVI RTDs by ABL and |ILD| (0--160 ms x-axis)",
    row_data,
    global_y_max,
)
save_rtd_grid(
    ZOOM_120MS_PNG,
    (0, 120),
    (0, 60, 120),
    "LED7 direct NPL+alpha SVI RTDs by ABL and |ILD| (0--120 ms x-axis)",
    row_data,
    global_y_max,
)
save_rtd_grid(
    RIGHT_TRUNCATED_120MS_PNG,
    (0, 120),
    (0, 60, 120),
    (
        "LED7 direct NPL+alpha SVI RTDs right-truncated at 120 ms "
        "(renormalized over 0--120 ms)"
    ),
    right_truncated_row_data,
    right_truncated_global_y_max,
    truncation_ms=RIGHT_TRUNCATION_S * 1e3,
)

# %%
# =============================================================================
# Across-animal condition delay by signed ILD and ABL
# =============================================================================
abl_colors = {
    20: "#1f77b4",
    40: "#ff7f0e",
    60: "#2ca02c",
}

fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
for abl in ABLS:
    sub = delay_summary_df[delay_summary_df["ABL"].eq(abl)]
    ax.errorbar(
        sub["ILD"],
        sub["mean_ms"],
        yerr=sub["sem_ms"],
        fmt="o-",
        color=abl_colors[abl],
        ecolor=abl_colors[abl],
        linewidth=1.3,
        elinewidth=1.1,
        capsize=3,
        markersize=5,
        label=f"ABL {abl}",
    )

ax.axvline(0, color="0.82", linewidth=1, zorder=0)
ax.set_xticks(SIGNED_ILDS)
ax.set_xticklabels(
    [f"{ild:g}" for ild in SIGNED_ILDS],
    rotation=45,
    ha="right",
)
ax.set_xlabel("ILD (dB)")
ax.set_ylabel(r"$t_{E,\mathrm{aff}}$ (ms)")
ax.set_title("LED7 NPL+alpha SVI condition delays")
ax.grid(axis="y", alpha=0.22)
ax.legend(frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.savefig(DELAY_BY_ILD_PNG, dpi=PLOT_DPI, bbox_inches="tight")
print(f"Saved figure: {DELAY_BY_ILD_PNG}")

print(f"Saved payload: {OUTPUT_PKL}")

# %%
