# %%
"""
LED7 RT quantiles from the direct patience-12 NPL+alpha condition-delay SVI.

The data and intended-fix distributions use the same valid 0--1 s fitting rows
as the LED7 fit-aligned RTD diagnostic. Quantiles are first calculated for each
animal, ABL, and signed ILD. The two ILD signs are then averaged within animal
before taking across-animal means and SEMs.
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
QUANTILES = (0.1, 0.3, 0.5, 0.7, 0.9)

FIT_RT_MIN_S = 0.0
FIT_RT_MAX_S = 1.0
MODEL_STEP_S = 0.001
CONTINUOUS_ILD_STEP = 0.1
K_MAX = 10
PROACTIVE_CHUNK_SIZE = 512
PLOT_DPI = 300

DATA_CSV = REPO_ROOT / "raw_data" / "batch_csvs" / "batch_LED7_valid_and_aborts.csv"
FIT_ROOT = (
    REPO_ROOT
    / "fit_animal_by_animal"
    / "numpyro_svi_npl_alpha_condition_delay_patience12_restore_best_outputs"
)
ABORT_ROOT = REPO_ROOT / "aborts_ipl_npl_time_fit_results"
OUTPUT_DIR = SCRIPT_DIR / "led7_npl_alpha_patience12_fit_aligned_valid_rtds"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PNG = (
    OUTPUT_DIR
    / "led7_npl_alpha_patience12_fit_aligned_rt_quantiles_equal_animal_1x4.png"
)
OUTPUT_PKL = OUTPUT_PNG.with_suffix(".pkl")


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
from scipy.integrate import cumulative_trapezoid, trapezoid

if str(FIT_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(FIT_UTILS_DIR))
import numpyro_npl_alpha_svi_utils as svi_utils

CONTINUOUS_ABS_ILDS = np.round(
    np.arange(
        min(ABS_ILDS),
        max(ABS_ILDS) + CONTINUOUS_ILD_STEP / 2,
        CONTINUOUS_ILD_STEP,
    ),
    10,
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
# Reused numerical operations
# =============================================================================
def mean_sem(values, axis=0):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    n = np.sum(finite, axis=axis)
    mean = np.nanmean(values, axis=axis)
    sd = np.nanstd(values, axis=axis, ddof=1)
    curr_sem = sd / np.sqrt(np.maximum(n, 1))
    curr_sem = np.where(n > 1, curr_sem, np.nan)
    return mean, curr_sem, n


def interpolate_delay(delay_by_condition, abl, signed_ild):
    signed_ild = float(signed_ild)
    exact = delay_by_condition.get((int(abl), signed_ild))
    if exact is not None:
        return float(exact)

    sign = np.sign(signed_ild)
    branch = sorted(
        [
            (abs(float(ild)), float(delay))
            for (condition_abl, ild), delay in delay_by_condition.items()
            if int(condition_abl) == int(abl) and np.sign(float(ild)) == sign
        ]
    )
    if len(branch) < 2:
        raise RuntimeError(
            f"Cannot interpolate t_E_aff for ABL={abl}, ILD={signed_ild}."
        )

    branch_ild = np.asarray([item[0] for item in branch], dtype=float)
    branch_delay = np.asarray([item[1] for item in branch], dtype=float)
    abs_ild = abs(signed_ild)
    if abs_ild < branch_ild.min() or abs_ild > branch_ild.max():
        raise RuntimeError(
            f"Requested |ILD|={abs_ild} outside the fitted delay branch "
            f"[{branch_ild.min()}, {branch_ild.max()}]."
        )
    return float(np.interp(abs_ild, branch_ild, branch_delay))


def quantiles_from_density(rt_grid_s, density):
    density = np.asarray(density, dtype=float)
    density = np.where(np.isfinite(density), np.maximum(density, 0), 0)
    area = float(trapezoid(density, rt_grid_s))
    if area <= 0:
        return np.full(len(QUANTILES), np.nan), np.nan

    normalized_density = density / area
    normalized_area = float(trapezoid(normalized_density, rt_grid_s))
    cdf = cumulative_trapezoid(normalized_density, rt_grid_s, initial=0)
    cdf /= cdf[-1]
    return np.interp(QUANTILES, cdf, rt_grid_s), normalized_area


# %%
# =============================================================================
# Load the exact LED7 fitting rows
# =============================================================================
for required_path in [DATA_CSV, FIT_ROOT, ABORT_ROOT]:
    if not required_path.exists():
        raise FileNotFoundError(required_path)

batch_df = pd.read_csv(DATA_CSV)
required_columns = [
    "animal",
    "success",
    "RTwrtStim",
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

observed_animals = tuple(sorted(valid_df["animal"].unique()))
if observed_animals != tuple(sorted(ANIMALS)):
    raise RuntimeError(f"Expected LED7 animals {ANIMALS}, found {observed_animals}.")

condition_counts = (
    valid_df[["animal", "ABL", "ILD"]]
    .drop_duplicates()
    .groupby("animal")
    .size()
)
if not np.all(condition_counts.reindex(ANIMALS).to_numpy() == 30):
    raise RuntimeError(
        "Each LED7 animal should have 30 signed ABL/ILD conditions:\n"
        f"{condition_counts.to_string()}"
    )

print(f"Data CSV: {DATA_CSV}")
print(f"SVI fit root: {FIT_ROOT}")
print(f"Exact LED7 fitting rows: {len(valid_df):,}")
print(f"Animals: {ANIMALS}")
print(f"Quantiles: {QUANTILES}")


# %%
# =============================================================================
# Per-animal data and model quantiles
# =============================================================================
rt_grid_s = (
    np.arange(
        round(FIT_RT_MIN_S / MODEL_STEP_S),
        round(FIT_RT_MAX_S / MODEL_STEP_S) + 1,
    )
    * MODEL_STEP_S
)
proactive_grid_s = np.arange(
    round(-2.0 / MODEL_STEP_S),
    round(2.0 / MODEL_STEP_S) + 1,
) * MODEL_STEP_S
rt_grid_mask = (proactive_grid_s >= FIT_RT_MIN_S) & (
    proactive_grid_s <= FIT_RT_MAX_S
)
if not np.allclose(proactive_grid_s[rt_grid_mask], rt_grid_s):
    raise RuntimeError("The proactive and RT grids do not align over 0--1 s.")

n_animals = len(ANIMALS)
n_abls = len(ABLS)
n_abs_ilds = len(ABS_ILDS)
n_continuous_ilds = len(CONTINUOUS_ABS_ILDS)
n_quantiles = len(QUANTILES)

data_quantiles = np.full(
    (n_animals, n_abls, n_abs_ilds, n_quantiles),
    np.nan,
)
model_quantiles = np.full(
    (n_animals, n_abls, n_continuous_ilds, n_quantiles),
    np.nan,
)
normalization_areas = []
parameter_rows = []
delay_rows = []

for animal_idx, animal in enumerate(ANIMALS):
    print(f"\nProcessing LED7/{animal}")
    animal_df = valid_df[valid_df["animal"] == animal].copy()
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

    delay_by_condition = {}
    for condition, delay in zip(
        saved_conditions.itertuples(index=False),
        delay_means,
    ):
        key = (int(condition.ABL), float(condition.ILD))
        delay_by_condition[key] = float(delay)
        delay_rows.append(
            {
                "batch_name": BATCH_NAME,
                "animal": animal,
                "ABL": key[0],
                "ILD": key[1],
                "t_E_aff_s": float(delay),
            }
        )

    with abort_path.open("rb") as handle:
        abort_fit = pickle.load(handle)["vbmc_aborts_results"]
    V_A = float(np.mean(np.asarray(abort_fit["V_A_samples"], dtype=float)))
    theta_A = float(np.mean(np.asarray(abort_fit["theta_A_samples"], dtype=float)))
    t_A_aff = float(np.mean(np.asarray(abort_fit["t_A_aff_samp"], dtype=float)))

    # Average the proactive density over every valid fitting row for this animal.
    # This is deterministic and gives each fitting row equal weight.
    proactive_density_sum = np.zeros_like(proactive_grid_s)
    intended_fix = animal_df["intended_fix"].to_numpy(dtype=float)
    for start in range(0, len(intended_fix), PROACTIVE_CHUNK_SIZE):
        stimulus_chunk = intended_fix[start : start + PROACTIVE_CHUNK_SIZE]
        proactive_time = (
            proactive_grid_s[None, :]
            + stimulus_chunk[:, None]
            - t_A_aff
        )
        positive = proactive_time > 0
        safe_time = np.where(positive, proactive_time, 1.0)
        proactive_density = (
            theta_A
            / np.sqrt(2 * np.pi * safe_time**3)
            * np.exp(
                -0.5
                * V_A**2
                * (safe_time - theta_A / V_A) ** 2
                / safe_time
            )
        )
        proactive_density[~positive] = 0.0
        proactive_density_sum += proactive_density.sum(axis=0)

    p_a_mean = proactive_density_sum / len(intended_fix)
    c_a_mean = cumulative_trapezoid(
        p_a_mean,
        proactive_grid_s,
        initial=0,
    )
    p_a_rt = p_a_mean[rt_grid_mask]
    c_a_rt = c_a_mean[rt_grid_mask]

    # Data quantiles are computed per signed condition, then averaged across
    # the two signs within animal to match the Figure 4 quantile convention.
    for abl_idx, abl in enumerate(ABLS):
        for abs_idx, abs_ild in enumerate(ABS_ILDS):
            sign_quantiles = []
            for sign in (-1, 1):
                signed_ild = float(sign * abs_ild)
                condition_rts = animal_df[
                    (animal_df["ABL"] == abl)
                    & (animal_df["ILD"] == signed_ild)
                ]["RTwrtStim"].to_numpy(dtype=float)
                if len(condition_rts) <= 5:
                    raise RuntimeError(
                        f"Too few data rows for LED7/{animal}, ABL={abl}, "
                        f"ILD={signed_ild}: {len(condition_rts)}."
                    )
                sign_quantiles.append(np.quantile(condition_rts, QUANTILES))
            data_quantiles[animal_idx, abl_idx, abs_idx] = np.mean(
                sign_quantiles,
                axis=0,
            )

    # Evaluate all ABL/signed-ILD curves together at 1 ms resolution.
    condition_abls = []
    condition_ilds = []
    condition_delays = []
    condition_indices = []
    for abl_idx, abl in enumerate(ABLS):
        for abs_idx, abs_ild in enumerate(CONTINUOUS_ABS_ILDS):
            for sign in (-1, 1):
                signed_ild = float(sign * abs_ild)
                condition_abls.append(float(abl))
                condition_ilds.append(signed_ild)
                condition_delays.append(
                    interpolate_delay(delay_by_condition, abl, signed_ild)
                )
                condition_indices.append((abl_idx, abs_idx, sign))

    condition_abls = np.asarray(condition_abls, dtype=float)
    condition_ilds = np.asarray(condition_ilds, dtype=float)
    condition_delays = np.asarray(condition_delays, dtype=float)
    relative_evidence_time = jnp.asarray(
        rt_grid_s[None, :] - condition_delays[:, None],
        dtype=jnp.float64,
    )
    abl_jax = jnp.asarray(condition_abls[:, None], dtype=jnp.float64)
    ild_jax = jnp.asarray(condition_ilds[:, None], dtype=jnp.float64)
    Z_E = (params["w"] - 0.5) * 2.0 * params["theta_E"]

    c_e = np.asarray(
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

    model_density = (
        p_a_rt[None, :] * (1.0 - c_e)
        + rho_e * (1.0 - c_a_rt[None, :])
    )
    sign_model_quantiles = {}
    for row_idx, (abl_idx, abs_idx, sign) in enumerate(condition_indices):
        row_quantiles, normalized_area = quantiles_from_density(
            rt_grid_s,
            model_density[row_idx],
        )
        sign_model_quantiles[(abl_idx, abs_idx, sign)] = row_quantiles
        normalization_areas.append(normalized_area)

    for abl_idx, _abl in enumerate(ABLS):
        for abs_idx, _abs_ild in enumerate(CONTINUOUS_ABS_ILDS):
            model_quantiles[animal_idx, abl_idx, abs_idx] = np.mean(
                [
                    sign_model_quantiles[(abl_idx, abs_idx, -1)],
                    sign_model_quantiles[(abl_idx, abs_idx, 1)],
                ],
                axis=0,
            )

    print(
        f"  {len(animal_df):,} rows; 30 fitted delays; "
        f"proactive CDF endpoint={c_a_mean[-1]:.6f}"
    )


# %%
# =============================================================================
# Validate and aggregate across animals
# =============================================================================
if not np.isfinite(data_quantiles).all():
    raise RuntimeError("Non-finite empirical quantiles.")
if not np.isfinite(model_quantiles).all():
    raise RuntimeError("Non-finite model quantiles.")
if np.any(np.diff(data_quantiles, axis=-1) < -1e-12):
    raise RuntimeError("Empirical quantiles are not monotonically ordered.")
if np.any(np.diff(model_quantiles, axis=-1) < -1e-12):
    raise RuntimeError("Model quantiles are not monotonically ordered.")
if not np.allclose(normalization_areas, 1.0, atol=1e-10, rtol=0):
    raise RuntimeError("One or more normalized model RTDs do not integrate to one.")

data_mean, data_sem, data_n = mean_sem(data_quantiles, axis=0)
model_mean, model_sem, model_n = mean_sem(model_quantiles, axis=0)

# Figure 4 collapses ABL by pooling each animal-ABL entry. Because LED7 is
# balanced across all three ABLs, every animal contributes exactly three times.
collapsed_data_values = data_quantiles.reshape(
    n_animals * n_abls,
    n_abs_ilds,
    n_quantiles,
)
collapsed_model_values = model_quantiles.reshape(
    n_animals * n_abls,
    n_continuous_ilds,
    n_quantiles,
)
collapsed_data_mean, collapsed_data_sem, collapsed_data_n = mean_sem(
    collapsed_data_values,
    axis=0,
)
collapsed_model_mean, collapsed_model_sem, collapsed_model_n = mean_sem(
    collapsed_model_values,
    axis=0,
)

if not np.all(data_n == n_animals):
    raise RuntimeError(f"Expected six data animals per ABL point, found {data_n}.")
if not np.all(model_n == n_animals):
    raise RuntimeError(f"Expected six model animals per ABL point, found {model_n}.")
if not np.all(collapsed_data_n == n_animals * n_abls):
    raise RuntimeError(
        f"Expected 18 collapsed data contributors, found {collapsed_data_n}."
    )
if not np.all(collapsed_model_n == n_animals * n_abls):
    raise RuntimeError(
        f"Expected 18 collapsed model contributors, found {collapsed_model_n}."
    )

fitted_indices = []
for abs_ild in ABS_ILDS:
    matches = np.where(np.isclose(CONTINUOUS_ABS_ILDS, abs_ild))[0]
    if len(matches) != 1:
        raise RuntimeError(
            f"Could not locate fitted |ILD|={abs_ild} on the continuous grid."
        )
    fitted_indices.append(int(matches[0]))

print("\nContributor counts")
for abl_idx, abl in enumerate(ABLS):
    print(
        f"  ABL={abl}: data={np.unique(data_n[abl_idx]).tolist()}, "
        f"model={np.unique(model_n[abl_idx]).tolist()}"
    )
print(
    "  ABL collapsed: "
    f"data={np.unique(collapsed_data_n).tolist()}, "
    f"model={np.unique(collapsed_model_n).tolist()}"
)
print(
    "Normalized model RTD areas: "
    f"min={np.min(normalization_areas):.12f}, "
    f"max={np.max(normalization_areas):.12f}"
)


# %%
# =============================================================================
# Plot the three ABLs and the Figure 4-style ABL collapse
# =============================================================================
fig, axes = plt.subplots(
    1,
    4,
    figsize=(18.0, 4.5),
    sharex=True,
    sharey=True,
)

panel_data = [
    (
        f"ABL = {abl}",
        data_mean[abl_idx],
        data_sem[abl_idx],
        model_mean[abl_idx],
        model_sem[abl_idx],
    )
    for abl_idx, abl in enumerate(ABLS)
]
panel_data.append(
    (
        "ABL collapsed",
        collapsed_data_mean,
        collapsed_data_sem,
        collapsed_model_mean,
        collapsed_model_sem,
    )
)

for ax, (
    title,
    panel_data_mean,
    panel_data_sem,
    panel_model_mean,
    panel_model_sem,
) in zip(axes, panel_data):
    for quantile_idx, _quantile in enumerate(QUANTILES):
        ax.errorbar(
            ABS_ILDS,
            panel_data_mean[:, quantile_idx],
            yerr=panel_data_sem[:, quantile_idx],
            fmt="o",
            color="black",
            markersize=4.5,
            capsize=0,
            alpha=0.85,
            linestyle="none",
            zorder=4,
        )
        ax.plot(
            CONTINUOUS_ABS_ILDS,
            panel_model_mean[:, quantile_idx],
            color="tab:red",
            linewidth=1.4,
            zorder=2,
        )
        ax.fill_between(
            CONTINUOUS_ABS_ILDS,
            panel_model_mean[:, quantile_idx]
            - panel_model_sem[:, quantile_idx],
            panel_model_mean[:, quantile_idx]
            + panel_model_sem[:, quantile_idx],
            color="tab:red",
            alpha=0.10,
            linewidth=0,
            zorder=1,
        )
        ax.scatter(
            ABS_ILDS,
            panel_model_mean[fitted_indices, quantile_idx],
            marker="x",
            s=28,
            linewidths=1.2,
            color="tab:red",
            zorder=5,
        )

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("|ILD| (dB)", fontsize=11)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ABS_ILDS)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.tick_params(axis="both", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.14)

axes[0].set_ylabel("RT quantile (s)", fontsize=11)
all_lower = min(
    float(np.nanmin(data_mean - data_sem)),
    float(np.nanmin(model_mean - model_sem)),
    float(np.nanmin(collapsed_data_mean - collapsed_data_sem)),
    float(np.nanmin(collapsed_model_mean - collapsed_model_sem)),
)
all_upper = max(
    float(np.nanmax(data_mean + data_sem)),
    float(np.nanmax(model_mean + model_sem)),
    float(np.nanmax(collapsed_data_mean + collapsed_data_sem)),
    float(np.nanmax(collapsed_model_mean + collapsed_model_sem)),
)
padding = max(0.02, 0.08 * (all_upper - all_lower))
axes[0].set_ylim(max(0, all_lower - padding), all_upper + padding)

legend_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        color="black",
        markersize=5,
        label="Data mean +/- SEM",
    ),
    Line2D(
        [0],
        [0],
        color="tab:red",
        linewidth=1.4,
        label="Model interpolated delay",
    ),
    Line2D(
        [0],
        [0],
        marker="x",
        linestyle="none",
        color="tab:red",
        markersize=6,
        label="Model fitted |ILD|",
    ),
]
fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.995),
    ncol=3,
    frameon=False,
    fontsize=10,
)
fig.suptitle(
    "LED7 direct NPL+alpha SVI: equal-animal RT quantiles",
    y=1.08,
    fontsize=12,
)
fig.subplots_adjust(
    left=0.065,
    right=0.99,
    bottom=0.16,
    top=0.80,
    wspace=0.20,
)
fig.savefig(OUTPUT_PNG, dpi=PLOT_DPI, bbox_inches="tight")


# %%
# =============================================================================
# Save the compact numerical payload
# =============================================================================
payload = {
    "batch_name": BATCH_NAME,
    "animals": ANIMALS,
    "abls": ABLS,
    "signed_ilds": SIGNED_ILDS,
    "abs_ilds": ABS_ILDS,
    "continuous_abs_ilds": CONTINUOUS_ABS_ILDS,
    "quantiles": QUANTILES,
    "data_quantiles_by_animal": data_quantiles,
    "model_quantiles_by_animal": model_quantiles,
    "data_mean_by_abl": data_mean,
    "data_sem_by_abl": data_sem,
    "data_n_by_abl": data_n,
    "model_mean_by_abl": model_mean,
    "model_sem_by_abl": model_sem,
    "model_n_by_abl": model_n,
    "collapsed_data_mean": collapsed_data_mean,
    "collapsed_data_sem": collapsed_data_sem,
    "collapsed_data_n": collapsed_data_n,
    "collapsed_model_mean": collapsed_model_mean,
    "collapsed_model_sem": collapsed_model_sem,
    "collapsed_model_n": collapsed_model_n,
    "parameter_rows": pd.DataFrame(parameter_rows),
    "delay_rows": pd.DataFrame(delay_rows),
    "data_csv": str(DATA_CSV.relative_to(REPO_ROOT)),
    "fit_root": str(FIT_ROOT.relative_to(REPO_ROOT)),
    "abort_root": str(ABORT_ROOT.relative_to(REPO_ROOT)),
    "model_step_s": MODEL_STEP_S,
    "sign_collapse": "Mean of negative- and positive-ILD quantiles within animal.",
    "abl_collapse": (
        "Figure 4 convention: pool the 18 animal-ABL quantile values "
        "before calculating mean and SEM."
    ),
}
with OUTPUT_PKL.open("wb") as handle:
    pickle.dump(payload, handle)

print(f"\nSaved figure: {OUTPUT_PNG}")
print(f"Saved payload: {OUTPUT_PKL}")

# %%
