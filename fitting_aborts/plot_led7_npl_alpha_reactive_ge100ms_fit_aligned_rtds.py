# %%
"""
LED7 reactive-only NPL+alpha RTDs by ABL and absolute ILD.

Both empirical and model RTDs are conditioned on the exact fitting window:

    0.100 <= RTwrtStim < 1.000 seconds

Each signed-condition RTD is normalized to unit area before equal-sign,
equal-condition, and equal-animal averaging.
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
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
FIT_UTILS_DIR = REPO_ROOT / "fit_animal_by_animal"

BATCH_NAME = "LED7"
ANIMALS = (92, 93, 98, 99, 100, 103)
ABLS = (20, 40, 60)
SIGNED_ILDS = (-16.0, -8.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0, 16.0)
ABS_ILDS = (1.0, 2.0, 4.0, 8.0, 16.0)

RT_MIN_S = 0.100
RT_MAX_S = 1.000
DISPLAY_RT_MAX_S = 0.600
MODEL_STEP_S = 0.001
DATA_BIN_S = 0.005
K_MAX = 10
PLOT_DPI = 300

DATA_CSV = (
    REPO_ROOT
    / "raw_data"
    / "batch_csvs"
    / "batch_LED7_valid_and_aborts.csv"
)
FIT_ROOT = (
    REPO_ROOT
    / "fit_animal_by_animal"
    / "numpyro_svi_npl_alpha_reactive_only_rt_ge_100ms_condition_delay_"
    "patience12_min50k_restore_best_outputs"
)
OUTPUT_DIR = (
    SCRIPT_DIR
    / "led7_npl_alpha_reactive_ge100ms_fit_aligned_rtds"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_PATH = (
    OUTPUT_DIR
    / "led7_npl_alpha_reactive_ge100ms_rtds_by_abl_abs_ild_100ms_1s.png"
)
OUTPUT_PKL = (
    OUTPUT_DIR
    / "led7_npl_alpha_reactive_ge100ms_rtds_by_abl_abs_ild.pkl"
)
AREA_AUDIT_CSV = (
    OUTPUT_DIR
    / "led7_npl_alpha_reactive_ge100ms_rtd_area_audit.csv"
)


# %%
# =============================================================================
# Imports and exact likelihood utility
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
import numpyro_npl_alpha_reactive_svi_utils as reactive_utils

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


def mean_sem(values, axis=0):
    values = np.asarray(values, dtype=float)
    n = np.sum(np.isfinite(values), axis=axis)
    mean = np.nanmean(values, axis=axis)
    sd = np.nanstd(values, axis=axis, ddof=1)
    sem = np.where(n > 1, sd / np.sqrt(np.maximum(n, 1)), np.nan)
    return mean, sem, n


# %%
# =============================================================================
# Exact fitting rows and grids
# =============================================================================
for required_path in [DATA_CSV, FIT_ROOT]:
    if not required_path.exists():
        raise FileNotFoundError(required_path)

batch_df = pd.read_csv(DATA_CSV)
required_columns = [
    "animal",
    "success",
    "RTwrtStim",
    "ABL",
    "ILD",
]
missing_columns = [
    column for column in required_columns if column not in batch_df.columns
]
if missing_columns:
    raise KeyError(f"Missing columns in {DATA_CSV}: {missing_columns}")

valid_df = batch_df[
    batch_df["animal"].astype(int).isin(ANIMALS)
    & batch_df["success"].isin([1, -1])
    & (batch_df["RTwrtStim"] >= RT_MIN_S)
    & (batch_df["RTwrtStim"] < RT_MAX_S)
    & batch_df["ABL"].isin(ABLS)
    & batch_df["ILD"].isin(SIGNED_ILDS)
].dropna(subset=required_columns).copy()
valid_df["animal"] = valid_df["animal"].astype(int)
valid_df["ABL"] = valid_df["ABL"].astype(int)
valid_df["ILD"] = valid_df["ILD"].astype(float)

observed_animals = tuple(sorted(valid_df["animal"].unique()))
if observed_animals != tuple(sorted(ANIMALS)):
    raise RuntimeError(
        f"Expected LED7 animals {ANIMALS}, found {observed_animals}."
    )
condition_counts = (
    valid_df[["animal", "ABL", "ILD"]]
    .drop_duplicates()
    .groupby("animal")
    .size()
    .reindex(ANIMALS)
)
if not np.all(condition_counts.to_numpy() == 30):
    raise RuntimeError(
        "Each LED7 animal should have 30 retained signed conditions:\n"
        + condition_counts.to_string()
    )

rt_grid_s = np.arange(
    round(RT_MIN_S / MODEL_STEP_S),
    round(RT_MAX_S / MODEL_STEP_S) + 1,
) * MODEL_STEP_S
data_bins_s = np.arange(
    round(RT_MIN_S / DATA_BIN_S),
    round(RT_MAX_S / DATA_BIN_S) + 1,
) * DATA_BIN_S
data_bin_centers_s = 0.5 * (data_bins_s[:-1] + data_bins_s[1:])

n_animals = len(ANIMALS)
n_abls = len(ABLS)
n_abs_ilds = len(ABS_ILDS)
data_rtd_by_animal = np.full(
    (n_animals, n_abls, n_abs_ilds, len(data_bin_centers_s)),
    np.nan,
)
model_rtd_by_animal = np.full(
    (n_animals, n_abls, n_abs_ilds, len(rt_grid_s)),
    np.nan,
)

print(f"Data CSV: {DATA_CSV}")
print(f"Reactive SVI fit root: {FIT_ROOT}")
print(
    f"Valid fitting rows in [{RT_MIN_S:.3f}, {RT_MAX_S:.3f}) s: "
    f"{len(valid_df):,}"
)
print(f"Animals: {ANIMALS}")


# %%
# =============================================================================
# Per-animal, per-signed-condition conditional RTDs
# =============================================================================
condition_area_rows = []
parameter_rows = []

for animal_idx, animal in enumerate(ANIMALS):
    print(f"\nProcessing LED7/{animal}")
    animal_df = valid_df[valid_df["animal"] == animal].copy()
    fit_dir = FIT_ROOT / f"{BATCH_NAME}_{animal}"
    posterior_path = fit_dir / "main_fullrank_posterior_samples.npz"
    condition_path = fit_dir / "condition_table.csv"
    metadata_path = fit_dir / "main_fullrank_run_metadata.json"
    for required_path in [
        posterior_path,
        condition_path,
        metadata_path,
    ]:
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    with np.load(posterior_path) as posterior:
        scalar_names = (
            "rate_lambda",
            "T_0",
            "theta_E",
            "w",
            "rate_norm_l",
            "alpha",
        )
        required_keys = scalar_names + ("t_E_aff",)
        missing_keys = [
            key for key in required_keys if key not in posterior.files
        ]
        if missing_keys:
            raise KeyError(
                f"Missing posterior keys for LED7/{animal}: {missing_keys}"
            )
        if any(
            not np.isfinite(np.asarray(posterior[key], dtype=float)).all()
            for key in required_keys
        ):
            raise RuntimeError(
                f"Non-finite posterior samples for LED7/{animal}."
            )
        params = {
            key: float(np.mean(np.asarray(posterior[key], dtype=float)))
            for key in scalar_names
        }
        delay_means = np.mean(
            np.asarray(posterior["t_E_aff"], dtype=float),
            axis=0,
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
        saved_conditions[
            ["ABL", "ILD", "condition_id"]
        ].to_numpy(dtype=float),
        reconstructed_conditions[
            ["ABL", "ILD", "condition_id"]
        ].to_numpy(dtype=float),
        atol=1e-12,
        rtol=0,
    ):
        raise RuntimeError(
            f"Saved and reconstructed conditions differ for LED7/{animal}."
        )
    if len(delay_means) != len(saved_conditions):
        raise RuntimeError(
            f"Delay vector length mismatch for LED7/{animal}."
        )

    for name, value in params.items():
        parameter_rows.append(
            {
                "batch_name": BATCH_NAME,
                "animal": animal,
                "parameter": name,
                "posterior_mean": value,
            }
        )

    condition_abls = saved_conditions["ABL"].to_numpy(dtype=float)
    condition_ilds = saved_conditions["ILD"].to_numpy(dtype=float)
    Z_E = (params["w"] - 0.5) * 2.0 * params["theta_E"]
    relative_time = jnp.asarray(
        rt_grid_s[None, :] - delay_means[:, None],
        dtype=jnp.float64,
    )
    abl_jax = jnp.asarray(condition_abls[:, None], dtype=jnp.float64)
    ild_jax = jnp.asarray(condition_ilds[:, None], dtype=jnp.float64)

    raw_model_density = np.asarray(
        reactive_utils.rho_E_alpha_jax(
            relative_time,
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
        + reactive_utils.rho_E_alpha_jax(
            relative_time,
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
    lower_relative_time = jnp.asarray(
        RT_MIN_S - delay_means,
        dtype=jnp.float64,
    )
    upper_relative_time = jnp.asarray(
        RT_MAX_S - delay_means,
        dtype=jnp.float64,
    )
    abl_vector_jax = jnp.asarray(condition_abls, dtype=jnp.float64)
    ild_vector_jax = jnp.asarray(condition_ilds, dtype=jnp.float64)
    retained_window_mass = np.asarray(
        sum(
            reactive_utils.CDF_E_alpha_jax(
                upper_relative_time,
                bound,
                abl_vector_jax,
                ild_vector_jax,
                params["rate_lambda"],
                params["T_0"],
                params["theta_E"],
                Z_E,
                params["rate_norm_l"],
                params["alpha"],
                K_MAX,
            )
            - reactive_utils.CDF_E_alpha_jax(
                lower_relative_time,
                bound,
                abl_vector_jax,
                ild_vector_jax,
                params["rate_lambda"],
                params["T_0"],
                params["theta_E"],
                Z_E,
                params["rate_norm_l"],
                params["alpha"],
                K_MAX,
            )
            for bound in (-1, 1)
        )
    )
    if (
        not np.isfinite(retained_window_mass).all()
        or np.any(retained_window_mass <= 0)
    ):
        raise RuntimeError(
            f"Invalid retained-window mass for LED7/{animal}."
        )

    model_density_from_likelihood = (
        raw_model_density / retained_window_mass[:, None]
    )
    data_rtd_by_signed_condition = {}
    model_rtd_by_signed_condition = {}

    for condition in saved_conditions.itertuples(index=False):
        condition_id = int(condition.condition_id)
        abl = int(condition.ABL)
        signed_ild = float(condition.ILD)
        condition_rows = animal_df[
            animal_df["ABL"].eq(abl)
            & np.isclose(animal_df["ILD"], signed_ild)
        ]
        condition_rts = condition_rows["RTwrtStim"].to_numpy(dtype=float)
        counts, _ = np.histogram(condition_rts, bins=data_bins_s)
        data_density = normalize_histogram_density(
            counts.astype(float) / DATA_BIN_S
        )

        model_before_numeric_normalization = (
            model_density_from_likelihood[condition_id]
        )
        model_area_before_numeric_normalization = float(
            trapezoid(model_before_numeric_normalization, rt_grid_s)
        )
        model_density = normalize_continuous_density(
            model_before_numeric_normalization
        )

        data_area = float(
            np.sum(data_density * np.diff(data_bins_s))
        )
        model_area = float(trapezoid(model_density, rt_grid_s))
        data_rtd_by_signed_condition[(abl, signed_ild)] = data_density
        model_rtd_by_signed_condition[(abl, signed_ild)] = model_density
        condition_area_rows.append(
            {
                "batch_name": BATCH_NAME,
                "animal": animal,
                "ABL": abl,
                "ILD": signed_ild,
                "n_valid_trials": len(condition_rts),
                "t_E_aff_ms": 1e3 * float(delay_means[condition_id]),
                "retained_window_mass_from_cdf": float(
                    retained_window_mass[condition_id]
                ),
                "model_area_after_likelihood_normalization": (
                    model_area_before_numeric_normalization
                ),
                "data_area_final": data_area,
                "model_area_final": model_area,
            }
        )

    for abl_idx, abl in enumerate(ABLS):
        for abs_idx, abs_ild in enumerate(ABS_ILDS):
            signed_keys = [(abl, -abs_ild), (abl, abs_ild)]
            data_sign_mean = normalize_histogram_density(
                np.mean(
                    [
                        data_rtd_by_signed_condition[key]
                        for key in signed_keys
                    ],
                    axis=0,
                )
            )
            model_sign_mean = normalize_continuous_density(
                np.mean(
                    [
                        model_rtd_by_signed_condition[key]
                        for key in signed_keys
                    ],
                    axis=0,
                )
            )
            data_rtd_by_animal[
                animal_idx,
                abl_idx,
                abs_idx,
            ] = data_sign_mean
            model_rtd_by_animal[
                animal_idx,
                abl_idx,
                abs_idx,
            ] = model_sign_mean

    print(
        f"  retained trials={len(animal_df):,}; "
        f"delay range={1e3 * np.min(delay_means):.1f}-"
        f"{1e3 * np.max(delay_means):.1f} ms"
    )


# %%
# =============================================================================
# Equal-animal, equal-ABL, and equal-|ILD| averages
# =============================================================================
if not np.isfinite(data_rtd_by_animal).all():
    raise RuntimeError("Non-finite empirical RTDs after sign averaging.")
if not np.isfinite(model_rtd_by_animal).all():
    raise RuntimeError("Non-finite model RTDs after sign averaging.")

data_mean, data_sem, data_n = mean_sem(data_rtd_by_animal, axis=0)
model_mean, model_sem, model_n = mean_sem(model_rtd_by_animal, axis=0)

data_abl_average_by_animal = np.mean(data_rtd_by_animal, axis=1)
model_abl_average_by_animal = np.mean(model_rtd_by_animal, axis=1)
data_ild_average_by_animal = np.mean(data_rtd_by_animal, axis=2)
model_ild_average_by_animal = np.mean(model_rtd_by_animal, axis=2)
data_grand_average_by_animal = np.mean(
    data_rtd_by_animal,
    axis=(1, 2),
)
model_grand_average_by_animal = np.mean(
    model_rtd_by_animal,
    axis=(1, 2),
)

for animal_idx in range(n_animals):
    for abs_idx in range(n_abs_ilds):
        data_abl_average_by_animal[
            animal_idx,
            abs_idx,
        ] = normalize_histogram_density(
            data_abl_average_by_animal[animal_idx, abs_idx]
        )
        model_abl_average_by_animal[
            animal_idx,
            abs_idx,
        ] = normalize_continuous_density(
            model_abl_average_by_animal[animal_idx, abs_idx]
        )
    for abl_idx in range(n_abls):
        data_ild_average_by_animal[
            animal_idx,
            abl_idx,
        ] = normalize_histogram_density(
            data_ild_average_by_animal[animal_idx, abl_idx]
        )
        model_ild_average_by_animal[
            animal_idx,
            abl_idx,
        ] = normalize_continuous_density(
            model_ild_average_by_animal[animal_idx, abl_idx]
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

data_abl_mean, data_abl_sem, data_abl_n = mean_sem(
    data_abl_average_by_animal,
    axis=0,
)
model_abl_mean, model_abl_sem, model_abl_n = mean_sem(
    model_abl_average_by_animal,
    axis=0,
)
data_ild_mean, data_ild_sem, data_ild_n = mean_sem(
    data_ild_average_by_animal,
    axis=0,
)
model_ild_mean, model_ild_sem, model_ild_n = mean_sem(
    model_ild_average_by_animal,
    axis=0,
)
data_grand_mean, data_grand_sem, data_grand_n = mean_sem(
    data_grand_average_by_animal,
    axis=0,
)
model_grand_mean, model_grand_sem, model_grand_n = mean_sem(
    model_grand_average_by_animal,
    axis=0,
)

for name, contributor_counts in {
    "condition data": data_n,
    "condition model": model_n,
    "ABL-average data": data_abl_n,
    "ABL-average model": model_abl_n,
    "ILD-average data": data_ild_n,
    "ILD-average model": model_ild_n,
    "grand-average data": data_grand_n,
    "grand-average model": model_grand_n,
}.items():
    if not np.all(np.asarray(contributor_counts) == n_animals):
        raise RuntimeError(
            f"{name} does not have six animal contributors."
        )

condition_area_df = pd.DataFrame(condition_area_rows)
condition_area_df.to_csv(AREA_AUDIT_CSV, index=False)
for column in ["data_area_final", "model_area_final"]:
    if not np.allclose(
        condition_area_df[column].to_numpy(dtype=float),
        1.0,
        atol=2e-10,
        rtol=0,
    ):
        raise RuntimeError(f"{column} is not exactly unit normalized.")

aggregate_area_checks = {
    "animal_condition_data": np.sum(
        data_rtd_by_animal * np.diff(data_bins_s),
        axis=-1,
    ),
    "animal_condition_model": trapezoid(
        model_rtd_by_animal,
        rt_grid_s,
        axis=-1,
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
    "animal_ild_average_data": np.sum(
        data_ild_average_by_animal * np.diff(data_bins_s),
        axis=-1,
    ),
    "animal_ild_average_model": trapezoid(
        model_ild_average_by_animal,
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
    "across_animal_condition_data": np.sum(
        data_mean * np.diff(data_bins_s),
        axis=-1,
    ),
    "across_animal_condition_model": trapezoid(
        model_mean,
        rt_grid_s,
        axis=-1,
    ),
    "across_animal_grand_data": float(
        np.sum(data_grand_mean * np.diff(data_bins_s))
    ),
    "across_animal_grand_model": float(
        trapezoid(model_grand_mean, rt_grid_s)
    ),
}
for name, values in aggregate_area_checks.items():
    if not np.allclose(
        np.asarray(values, dtype=float),
        1.0,
        atol=2e-10,
        rtol=0,
    ):
        raise RuntimeError(f"Area check failed for {name}: {values}")

print(f"\nSaved area audit: {AREA_AUDIT_CSV}")
print(
    "Model area after likelihood CDF normalization, before final numerical "
    "normalization: "
    f"range={condition_area_df['model_area_after_likelihood_normalization'].min():.8f}-"
    f"{condition_area_df['model_area_after_likelihood_normalization'].max():.8f}"
)
print("All final empirical and model RTD areas equal 1.")


# %%
# =============================================================================
# Save reusable plot payload
# =============================================================================
payload = {
    "batch_name": BATCH_NAME,
    "animals": ANIMALS,
    "abls": ABLS,
    "signed_ilds": SIGNED_ILDS,
    "abs_ilds": ABS_ILDS,
    "rt_window_s": (RT_MIN_S, RT_MAX_S),
    "model_step_s": MODEL_STEP_S,
    "data_bin_s": DATA_BIN_S,
    "rt_grid_s": rt_grid_s,
    "data_bins_s": data_bins_s,
    "data_bin_centers_s": data_bin_centers_s,
    "data_rtd_by_animal": data_rtd_by_animal,
    "model_rtd_by_animal": model_rtd_by_animal,
    "data_mean": data_mean,
    "data_sem": data_sem,
    "model_mean": model_mean,
    "model_sem": model_sem,
    "data_abl_average_by_animal": data_abl_average_by_animal,
    "model_abl_average_by_animal": model_abl_average_by_animal,
    "data_abl_mean": data_abl_mean,
    "data_abl_sem": data_abl_sem,
    "model_abl_mean": model_abl_mean,
    "model_abl_sem": model_abl_sem,
    "data_ild_average_by_animal": data_ild_average_by_animal,
    "model_ild_average_by_animal": model_ild_average_by_animal,
    "data_ild_mean": data_ild_mean,
    "data_ild_sem": data_ild_sem,
    "model_ild_mean": model_ild_mean,
    "model_ild_sem": model_ild_sem,
    "data_grand_average_by_animal": data_grand_average_by_animal,
    "model_grand_average_by_animal": model_grand_average_by_animal,
    "data_grand_mean": data_grand_mean,
    "data_grand_sem": data_grand_sem,
    "model_grand_mean": model_grand_mean,
    "model_grand_sem": model_grand_sem,
    "condition_area_audit": condition_area_df,
    "aggregate_area_checks": aggregate_area_checks,
    "parameter_rows": pd.DataFrame(parameter_rows),
    "data_csv": str(DATA_CSV.relative_to(REPO_ROOT)),
    "fit_root": str(FIT_ROOT.relative_to(REPO_ROOT)),
    "data_trial_pool": (
        "Only successful/valid trials with 0.100 <= RTwrtStim < 1.000 s; "
        "no abort-event rows."
    ),
    "model_density": (
        "Posterior-mean pure reactive bound-collapsed density, divided by "
        "the likelihood CDF mass in 0.100-1.000 s and then numerically "
        "renormalized on the 1 ms grid."
    ),
    "averaging": (
        "Normalize signed conditions first; average signs equally within "
        "animal; average ABLs and |ILD| values equally within animal; "
        "calculate mean and SEM across six animals."
    ),
}
with OUTPUT_PKL.open("wb") as handle:
    pickle.dump(payload, handle)
print(f"Saved payload: {OUTPUT_PKL}")


# %%
# =============================================================================
# Plot 4 x 6 RTD grid
# =============================================================================
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

fig, axes = plt.subplots(
    4,
    len(column_labels),
    figsize=(19.8, 10.3),
    sharex=True,
    sharey=True,
)
for row_idx, row in enumerate(row_data):
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
            alpha=0.10,
            linewidth=0,
            zorder=1,
        )
        ax.stairs(
            data_curve,
            data_bins_s * 1e3,
            color="black",
            linewidth=0.75,
            alpha=0.62,
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
            linewidth=1.55,
            label="Reactive NPL+alpha SVI",
            zorder=2,
        )

        if row_idx == 0:
            ax.set_title(column_label, fontsize=11)
        if col_idx == 0:
            ax.set_ylabel(
                f"{row['label']}\nDensity (s$^{{-1}}$)",
                fontsize=10,
            )
        if row_idx == len(row_data) - 1:
            ax.set_xlabel(r"RT - $t_{stim}$ (ms)", fontsize=10)

        ax.set_xlim(RT_MIN_S * 1e3, DISPLAY_RT_MAX_S * 1e3)
        ax.set_xticks([100, 350, 600])
        ax.set_ylim(0, global_y_max * 1.06)
        ax.tick_params(axis="both", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

legend_handles = [
    Line2D(
        [0],
        [0],
        color="black",
        linewidth=0.75,
        alpha=0.62,
        label="Data mean +/- SEM",
    ),
    Line2D(
        [0],
        [0],
        color="#0072B2",
        linewidth=1.55,
        label="Reactive NPL+alpha SVI mean +/- SEM",
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
fig.suptitle(
    "LED7 reactive-only NPL+alpha RTDs by ABL and |ILD| "
    "(conditional on 100-1000 ms)",
    fontsize=13,
    y=0.995,
)
fig.subplots_adjust(
    left=0.065,
    right=0.995,
    bottom=0.065,
    top=0.90,
    wspace=0.16,
    hspace=0.18,
)
fig.savefig(FIG_PATH, dpi=PLOT_DPI, bbox_inches="tight")
print(f"Saved figure: {FIG_PATH}")

# %%
