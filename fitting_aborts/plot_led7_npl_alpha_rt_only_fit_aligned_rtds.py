# %%
"""
LED7 fit-aligned RTDs for the two completed RT-only NPL+alpha SVI fits.

The script makes one ABL x |ILD| grid for each process model:

1. proactive + reactive RT-only likelihood;
2. reactive-only likelihood.

Data use the exact successful 0 <= RTwrtStim < 1 s fitting rows. Each signed
condition is normalized before equal averaging across ILD signs, conditions,
ABLs, and animals. Model curves use posterior-mean parameters, the fitted
condition delays, and the matching choice-collapsed RT-only likelihood.
"""

# %%
# =============================================================================
# Editable parameters
# =============================================================================
from pathlib import Path
import json
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

FIT_RT_MIN_S = 0.0
FIT_RT_MAX_S = 1.0
MODEL_STEP_S = 0.001
DATA_BIN_S = 0.005
DISPLAY_RT_MAX_MS = 600
K_MAX = 10
TRIAL_CHUNK_SIZE = 512
PLOT_DPI = 300

DATA_CSV = REPO_ROOT / "raw_data" / "batch_csvs" / "batch_LED7_valid_and_aborts.csv"
OUTPUT_DIR = SCRIPT_DIR / "led7_npl_alpha_rt_only_fit_aligned_rtds"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIT_CONFIGS = (
    {
        "mode": "proactive_reactive",
        "label": "proactive + reactive",
        "root": FIT_UTILS_DIR
        / (
            "numpyro_svi_npl_alpha_rt_only_proactive_reactive_0_to_1s_"
            "condition_delay_patience12_min50k_restore_best_outputs"
        ),
        "figure": OUTPUT_DIR
        / (
            "led7_npl_alpha_rt_only_proactive_reactive_fit_aligned_rtds_"
            "by_abl_abs_ild_0_600ms_xlim.png"
        ),
        "payload": OUTPUT_DIR
        / "led7_npl_alpha_rt_only_proactive_reactive_fit_aligned_rtds.pkl",
    },
    {
        "mode": "reactive_only",
        "label": "reactive only",
        "root": FIT_UTILS_DIR
        / (
            "numpyro_svi_npl_alpha_rt_only_reactive_only_0_to_1s_"
            "condition_delay_patience12_min50k_restore_best_outputs"
        ),
        "figure": OUTPUT_DIR
        / (
            "led7_npl_alpha_rt_only_reactive_only_fit_aligned_rtds_"
            "by_abl_abs_ild_0_600ms_xlim.png"
        ),
        "payload": OUTPUT_DIR
        / "led7_npl_alpha_rt_only_reactive_only_fit_aligned_rtds.pkl",
    },
)


# %%
# =============================================================================
# Imports and plotting defaults
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
import numpyro_npl_alpha_rt_only_svi_utils as rt_utils

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
def normalize_histogram_density(values, data_bins_s):
    values = np.asarray(values, dtype=float)
    area = float(np.sum(values * np.diff(data_bins_s)))
    if not np.isfinite(area) or area <= 0:
        raise RuntimeError(f"Invalid histogram area: {area}.")
    return values / area


def normalize_continuous_density(values, rt_grid_s):
    values = np.asarray(values, dtype=float)
    values = np.where(np.isfinite(values), np.maximum(values, 0), 0)
    area = float(trapezoid(values, rt_grid_s))
    if not np.isfinite(area) or area <= 0:
        raise RuntimeError(f"Invalid continuous-density area: {area}.")
    return values / area


def mean_sem(values, axis=0):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    n = np.sum(finite, axis=axis)
    mean = np.nanmean(values, axis=axis)
    sd = np.nanstd(values, axis=axis, ddof=1)
    sem = sd / np.sqrt(np.maximum(n, 1))
    sem = np.where(n > 1, sem, np.nan)
    return mean, sem, n


# %%
# =============================================================================
# Load the exact successful RT-only fitting rows and build data RTDs once
# =============================================================================
if not DATA_CSV.exists():
    raise FileNotFoundError(DATA_CSV)
for fit_config in FIT_CONFIGS:
    if not fit_config["root"].exists():
        raise FileNotFoundError(fit_config["root"])

batch_df = pd.read_csv(DATA_CSV)
required_columns = (
    "animal",
    "success",
    "RTwrtStim",
    "TotalFixTime",
    "intended_fix",
    "ABL",
    "ILD",
)
missing_columns = [column for column in required_columns if column not in batch_df]
if missing_columns:
    raise KeyError(f"Missing columns in {DATA_CSV}: {missing_columns}")

valid_df = batch_df[
    batch_df["animal"].astype(int).isin(ANIMALS)
    & batch_df["success"].isin([1, -1])
    & (batch_df["RTwrtStim"] >= FIT_RT_MIN_S)
    & (batch_df["RTwrtStim"] < FIT_RT_MAX_S)
    & batch_df["ABL"].isin(ABLS)
    & batch_df["ILD"].isin(SIGNED_ILDS)
].dropna(subset=required_columns).copy()
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
        + condition_counts.to_string()
    )

rt_grid_s = np.arange(
    round(FIT_RT_MIN_S / MODEL_STEP_S),
    round(FIT_RT_MAX_S / MODEL_STEP_S) + 1,
) * MODEL_STEP_S
data_bins_s = np.arange(
    round(FIT_RT_MIN_S / DATA_BIN_S),
    round(FIT_RT_MAX_S / DATA_BIN_S) + 1,
) * DATA_BIN_S
data_bin_centers_s = 0.5 * (data_bins_s[:-1] + data_bins_s[1:])

n_animals = len(ANIMALS)
n_abls = len(ABLS)
n_abs_ilds = len(ABS_ILDS)
n_data_bins = len(data_bin_centers_s)
n_model_points = len(rt_grid_s)

data_rtd_by_animal = np.full(
    (n_animals, n_abls, n_abs_ilds, n_data_bins),
    np.nan,
)
data_signed_rtds = {}

for animal_idx, animal in enumerate(ANIMALS):
    animal_df = valid_df[valid_df["animal"].eq(animal)]
    for abl in ABLS:
        for signed_ild in SIGNED_ILDS:
            condition_rts = animal_df.loc[
                animal_df["ABL"].eq(abl)
                & np.isclose(animal_df["ILD"], signed_ild),
                "RTwrtStim",
            ].to_numpy(dtype=float)
            if len(condition_rts) == 0:
                raise RuntimeError(
                    f"No data for LED7/{animal}, ABL={abl}, ILD={signed_ild}."
                )
            counts, _ = np.histogram(condition_rts, bins=data_bins_s)
            data_signed_rtds[(animal, abl, signed_ild)] = (
                normalize_histogram_density(
                    counts.astype(float) / DATA_BIN_S,
                    data_bins_s,
                )
            )

    for abl_idx, abl in enumerate(ABLS):
        for abs_idx, abs_ild in enumerate(ABS_ILDS):
            data_rtd_by_animal[animal_idx, abl_idx, abs_idx] = (
                normalize_histogram_density(
                    np.mean(
                        [
                            data_signed_rtds[(animal, abl, -abs_ild)],
                            data_signed_rtds[(animal, abl, abs_ild)],
                        ],
                        axis=0,
                    ),
                    data_bins_s,
                )
            )

print(f"Data CSV: {DATA_CSV}")
print(f"Exact successful LED7 RT-only fitting rows: {len(valid_df):,}")
print(f"Data bins: {1e3 * DATA_BIN_S:g} ms")
print(f"Model grid: {1e3 * MODEL_STEP_S:g} ms")


# %%
# =============================================================================
# Evaluate each RT-only posterior-mean model and make its RTD grid
# =============================================================================
audit_rows = []

for fit_config in FIT_CONFIGS:
    process_mode = fit_config["mode"]
    fit_root = fit_config["root"]
    ledger_csv = fit_root / "_batch_logs" / "batch_run_status.csv"
    if not ledger_csv.exists():
        raise FileNotFoundError(ledger_csv)
    ledger_df = pd.read_csv(ledger_csv).sort_values("run_index")
    if tuple(ledger_df["animal"].astype(int)) != ANIMALS:
        raise RuntimeError(f"Unexpected animal order in {ledger_csv}.")
    if not ledger_df["status"].eq("completed").all():
        raise RuntimeError(f"Not all fits are completed in {fit_root}.")

    print(f"\nEvaluating {fit_config['label']} RT-only fits")
    print(f"Fit root: {fit_root}")
    model_rtd_by_animal = np.full(
        (n_animals, n_abls, n_abs_ilds, n_model_points),
        np.nan,
    )
    parameter_rows = []
    delay_rows = []

    for animal_idx, animal in enumerate(ANIMALS):
        print(f"  LED7/{animal}")
        fit_dir = fit_root / f"LED7_{animal}"
        posterior_path = fit_dir / "main_fullrank_posterior_samples.npz"
        condition_path = fit_dir / "condition_table.csv"
        metadata_path = fit_dir / "main_fullrank_run_metadata.json"
        finite_path = fit_dir / "main_fullrank_posterior_finite_report.csv"
        for required_path in (
            posterior_path,
            condition_path,
            metadata_path,
            finite_path,
        ):
            if not required_path.exists():
                raise FileNotFoundError(required_path)

        posterior = np.load(posterior_path)
        required_keys = tuple(rt_utils.GLOBAL_PARAM_NAMES) + ("t_E_aff",)
        missing_keys = [key for key in required_keys if key not in posterior.files]
        if missing_keys:
            raise KeyError(f"Missing posterior keys for LED7/{animal}: {missing_keys}")
        if any(
            not np.isfinite(np.asarray(posterior[key], dtype=float)).all()
            for key in required_keys
        ):
            raise RuntimeError(f"Non-finite posterior samples for LED7/{animal}.")

        finite_df = pd.read_csv(finite_path)
        if not (
            finite_df["n_total"].to_numpy(dtype=int)
            == finite_df["n_finite"].to_numpy(dtype=int)
        ).all():
            raise RuntimeError(f"Non-finite posterior report for LED7/{animal}.")

        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata["config"]["process_mode"] != process_mode:
            raise RuntimeError(f"Process-mode mismatch for LED7/{animal}.")
        if not np.isclose(metadata["config"]["rt_lower_s"], FIT_RT_MIN_S):
            raise RuntimeError(f"Lower RT bound mismatch for LED7/{animal}.")
        if not np.isclose(metadata["config"]["rt_upper_s"], FIT_RT_MAX_S):
            raise RuntimeError(f"Upper RT bound mismatch for LED7/{animal}.")

        params_np = {
            name: float(np.mean(np.asarray(posterior[name], dtype=float)))
            for name in rt_utils.GLOBAL_PARAM_NAMES
        }
        delay_means = np.mean(
            np.asarray(posterior["t_E_aff"], dtype=float),
            axis=0,
        )
        params_jax = {
            name: jnp.asarray(value, dtype=jnp.float64)
            for name, value in params_np.items()
        }
        params_jax["t_E_aff"] = jnp.asarray(delay_means, dtype=jnp.float64)

        for name, value in params_np.items():
            parameter_rows.append(
                {
                    "batch_name": BATCH_NAME,
                    "animal": animal,
                    "parameter": name,
                    "posterior_mean": value,
                }
            )

        saved_conditions = (
            pd.read_csv(condition_path)
            .sort_values("condition_id")
            .reset_index(drop=True)
        )
        animal_df = valid_df[valid_df["animal"].eq(animal)].copy()
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
                f"Saved and reconstructed conditions differ for LED7/{animal}."
            )
        if len(delay_means) != len(saved_conditions):
            raise RuntimeError(f"Delay-vector length mismatch for LED7/{animal}.")

        animal_df = animal_df.merge(
            reconstructed_conditions,
            on=["ABL", "ILD"],
            how="left",
            validate="many_to_one",
        )
        saved_trial_counts = (
            saved_conditions.set_index("condition_id")["n_retained_trials"]
            .reindex(range(30))
            .to_numpy(dtype=int)
        )
        reconstructed_trial_counts = (
            animal_df.groupby("condition_id")
            .size()
            .reindex(range(30), fill_value=0)
            .to_numpy(dtype=int)
        )
        if not np.array_equal(saved_trial_counts, reconstructed_trial_counts):
            raise RuntimeError(f"Trial counts differ for LED7/{animal}.")
        if int(metadata["n_retained_trials"]) != len(animal_df):
            raise RuntimeError(f"Metadata trial count differs for LED7/{animal}.")

        condition_abls = saved_conditions["ABL"].to_numpy(dtype=float)
        condition_ilds = saved_conditions["ILD"].to_numpy(dtype=float)
        Z_E = (params_np["w"] - 0.5) * 2.0 * params_np["theta_E"]
        relative_time = jnp.asarray(
            rt_grid_s[None, :] - delay_means[:, None],
            dtype=jnp.float64,
        )
        abl_jax = jnp.asarray(condition_abls[:, None], dtype=jnp.float64)
        ild_jax = jnp.asarray(condition_ilds[:, None], dtype=jnp.float64)

        p_e = np.asarray(
            rt_utils.base_utils.rho_E_alpha_jax(
                relative_time,
                1,
                abl_jax,
                ild_jax,
                params_np["rate_lambda"],
                params_np["T_0"],
                params_np["theta_E"],
                Z_E,
                params_np["rate_norm_l"],
                params_np["alpha"],
                K_MAX,
            )
            + rt_utils.base_utils.rho_E_alpha_jax(
                relative_time,
                -1,
                abl_jax,
                ild_jax,
                params_np["rate_lambda"],
                params_np["T_0"],
                params_np["theta_E"],
                Z_E,
                params_np["rate_norm_l"],
                params_np["alpha"],
                K_MAX,
            )
        )
        c_e = np.asarray(
            rt_utils.base_utils.CDF_E_alpha_jax(
                relative_time,
                1,
                abl_jax,
                ild_jax,
                params_np["rate_lambda"],
                params_np["T_0"],
                params_np["theta_E"],
                Z_E,
                params_np["rate_norm_l"],
                params_np["alpha"],
                K_MAX,
            )
            + rt_utils.base_utils.CDF_E_alpha_jax(
                relative_time,
                -1,
                abl_jax,
                ild_jax,
                params_np["rate_lambda"],
                params_np["T_0"],
                params_np["theta_E"],
                Z_E,
                params_np["rate_norm_l"],
                params_np["alpha"],
                K_MAX,
            )
        )
        c_e_lower = c_e[:, 0]
        c_e_upper = c_e[:, -1]

        if process_mode == "reactive_only":
            retained_mass = c_e_upper - c_e_lower
            if not np.isfinite(retained_mass).all() or np.any(retained_mass <= 0):
                raise RuntimeError(f"Invalid reactive RT mass for LED7/{animal}.")
            model_by_signed_condition = p_e / retained_mass[:, None]
            fixed_proactive = None
        else:
            fixed_proactive = metadata["fixed_proactive_parameters"]
            if fixed_proactive is None:
                raise RuntimeError(f"Missing proactive parameters for LED7/{animal}.")
            V_A = float(fixed_proactive["V_A"])
            theta_A = float(fixed_proactive["theta_A"])
            t_A_aff = float(fixed_proactive["t_A_aff_s"])
            T_trunc = float(fixed_proactive["T_trunc_s"])
            proactive_trunc_denom = float(
                1.0
                - rt_utils.base_utils.cum_A_t_jax(
                    T_trunc - t_A_aff,
                    V_A,
                    theta_A,
                )
            )
            if not np.isfinite(proactive_trunc_denom) or proactive_trunc_denom <= 0:
                raise RuntimeError(
                    f"Invalid proactive truncation denominator for LED7/{animal}."
                )

            model_sum = np.zeros((30, n_model_points), dtype=float)
            model_counts = np.zeros(30, dtype=int)
            for start in range(0, len(animal_df), TRIAL_CHUNK_SIZE):
                chunk = animal_df.iloc[start : start + TRIAL_CHUNK_SIZE]
                condition_ids = chunk["condition_id"].to_numpy(dtype=int)
                t_stim = chunk["intended_fix"].to_numpy(dtype=float)
                absolute_t = t_stim[:, None] + rt_grid_s[None, :]

                p_a = np.asarray(
                    rt_utils.base_utils.rho_A_t_jax(
                        jnp.asarray(absolute_t - t_A_aff),
                        V_A,
                        theta_A,
                    )
                )
                c_a = np.asarray(
                    rt_utils.base_utils.cum_A_t_jax(
                        jnp.asarray(absolute_t - t_A_aff),
                        V_A,
                        theta_A,
                    )
                )
                numerator = (
                    p_a * (1.0 - c_e[condition_ids])
                    + p_e[condition_ids] * (1.0 - c_a)
                )

                c_a_lower_raw = np.asarray(
                    rt_utils.base_utils.cum_A_t_jax(
                        jnp.asarray(t_stim - t_A_aff),
                        V_A,
                        theta_A,
                    )
                )
                c_a_upper_raw = np.asarray(
                    rt_utils.base_utils.cum_A_t_jax(
                        jnp.asarray(t_stim + FIT_RT_MAX_S - t_A_aff),
                        V_A,
                        theta_A,
                    )
                )
                c_a_lower = np.where(
                    t_stim < T_trunc,
                    0.0,
                    c_a_lower_raw / proactive_trunc_denom,
                )
                c_a_upper = np.where(
                    t_stim + FIT_RT_MAX_S < T_trunc,
                    0.0,
                    c_a_upper_raw / proactive_trunc_denom,
                )
                race_lower = (
                    c_a_lower
                    + c_e_lower[condition_ids]
                    - c_a_lower * c_e_lower[condition_ids]
                )
                race_upper = (
                    c_a_upper
                    + c_e_upper[condition_ids]
                    - c_a_upper * c_e_upper[condition_ids]
                )
                retained_mass = race_upper - race_lower
                if not np.isfinite(retained_mass).all() or np.any(retained_mass <= 0):
                    raise RuntimeError(
                        f"Invalid proactive+reactive RT mass for LED7/{animal}."
                    )
                normalized_density = numerator / retained_mass[:, None]
                if not np.isfinite(normalized_density).all():
                    raise RuntimeError(f"Non-finite RT density for LED7/{animal}.")

                for condition_id in np.unique(condition_ids):
                    mask = condition_ids == condition_id
                    model_sum[condition_id] += normalized_density[mask].sum(axis=0)
                    model_counts[condition_id] += int(mask.sum())

            if not np.array_equal(model_counts, reconstructed_trial_counts):
                raise RuntimeError(f"Model trial counts differ for LED7/{animal}.")
            model_by_signed_condition = model_sum / model_counts[:, None]

        check_trial = animal_df.iloc[0]
        check_condition_id = int(check_trial["condition_id"])
        check_t_stim = float(check_trial["intended_fix"])
        check_abl = float(check_trial["ABL"])
        check_ild = float(check_trial["ILD"])
        if process_mode == "reactive_only":
            manual_check_density = model_by_signed_condition[check_condition_id]
            V_A_check = theta_A_check = 1.0
            t_A_aff_check = 0.0
            T_trunc_check = 0.30
        else:
            check_absolute_t = check_t_stim + rt_grid_s
            check_p_a = np.asarray(
                rt_utils.base_utils.rho_A_t_jax(
                    jnp.asarray(check_absolute_t - t_A_aff),
                    V_A,
                    theta_A,
                )
            )
            check_c_a = np.asarray(
                rt_utils.base_utils.cum_A_t_jax(
                    jnp.asarray(check_absolute_t - t_A_aff),
                    V_A,
                    theta_A,
                )
            )
            check_numerator = (
                check_p_a * (1.0 - c_e[check_condition_id])
                + p_e[check_condition_id] * (1.0 - check_c_a)
            )
            check_c_a_lower = (
                0.0
                if check_t_stim < T_trunc
                else float(
                    rt_utils.base_utils.cum_A_t_jax(
                        check_t_stim - t_A_aff,
                        V_A,
                        theta_A,
                    )
                )
                / proactive_trunc_denom
            )
            check_c_a_upper = (
                0.0
                if check_t_stim + FIT_RT_MAX_S < T_trunc
                else float(
                    rt_utils.base_utils.cum_A_t_jax(
                        check_t_stim + FIT_RT_MAX_S - t_A_aff,
                        V_A,
                        theta_A,
                    )
                )
                / proactive_trunc_denom
            )
            check_race_lower = (
                check_c_a_lower
                + c_e_lower[check_condition_id]
                - check_c_a_lower * c_e_lower[check_condition_id]
            )
            check_race_upper = (
                check_c_a_upper
                + c_e_upper[check_condition_id]
                - check_c_a_upper * c_e_upper[check_condition_id]
            )
            manual_check_density = check_numerator / (
                check_race_upper - check_race_lower
            )
            V_A_check = V_A
            theta_A_check = theta_A
            t_A_aff_check = t_A_aff
            T_trunc_check = T_trunc

        check_data = {
            "rt_wrt_stim": jnp.asarray(rt_grid_s[None, :]),
            "total_fix": jnp.asarray((check_t_stim + rt_grid_s)[None, :]),
            "t_stim": jnp.asarray([[check_t_stim]]),
            "ABL": jnp.asarray([[check_abl]]),
            "ILD": jnp.asarray([[check_ild]]),
            "condition_id": jnp.asarray([[check_condition_id]], dtype=jnp.int32),
            "V_A": jnp.asarray(V_A_check),
            "theta_A": jnp.asarray(theta_A_check),
            "t_A_aff": jnp.asarray(t_A_aff_check),
            "T_trunc": jnp.asarray(T_trunc_check),
            "rt_lower": jnp.asarray(FIT_RT_MIN_S),
            "rt_upper": jnp.asarray(FIT_RT_MAX_S),
        }
        direct_check_density = np.asarray(
            rt_utils.npl_alpha_rt_only_condition_delay_loglike_terms(
                params_jax,
                check_data,
                process_mode,
                K_max=K_MAX,
            )["normalized_pdf"]
        ).reshape(-1)
        max_abs_difference = float(
            np.max(np.abs(direct_check_density - manual_check_density))
        )
        comparison_scale = max(1.0, float(np.max(np.abs(direct_check_density))))
        relative_difference = max_abs_difference / comparison_scale
        if relative_difference > 1e-10:
            raise RuntimeError(
                f"Optimized likelihood check failed for {process_mode} "
                f"LED7/{animal}: {relative_difference:.3g}."
            )

        normalized_signed_model = {}
        for condition in saved_conditions.itertuples(index=False):
            condition_id = int(condition.condition_id)
            abl = int(condition.ABL)
            signed_ild = float(condition.ILD)
            normalized_signed_model[(abl, signed_ild)] = (
                normalize_continuous_density(
                    model_by_signed_condition[condition_id],
                    rt_grid_s,
                )
            )
            delay_rows.append(
                {
                    "batch_name": BATCH_NAME,
                    "animal": animal,
                    "ABL": abl,
                    "ILD": signed_ild,
                    "condition_id": condition_id,
                    "t_E_aff_s": float(delay_means[condition_id]),
                }
            )

        for abl_idx, abl in enumerate(ABLS):
            for abs_idx, abs_ild in enumerate(ABS_ILDS):
                model_rtd_by_animal[animal_idx, abl_idx, abs_idx] = (
                    normalize_continuous_density(
                        np.mean(
                            [
                                normalized_signed_model[(abl, -abs_ild)],
                                normalized_signed_model[(abl, abs_ild)],
                            ],
                            axis=0,
                        ),
                        rt_grid_s,
                    )
                )

        audit_rows.append(
            {
                "process_mode": process_mode,
                "batch_name": BATCH_NAME,
                "animal": animal,
                "n_retained_trials": len(animal_df),
                "best_step": int(metadata["best_step"]),
                "final_checked_step": int(metadata["final_checked_step"]),
                "optimized_vs_likelihood_relative_max": relative_difference,
                "all_posterior_samples_finite": True,
            }
        )
        print(
            f"    {len(animal_df):,} trials; likelihood check="
            f"{relative_difference:.2e}"
        )

    if not np.isfinite(model_rtd_by_animal).all():
        raise RuntimeError(f"Incomplete model RTD array for {process_mode}.")

    data_mean, data_sem, data_n = mean_sem(data_rtd_by_animal, axis=0)
    model_mean, model_sem, model_n = mean_sem(model_rtd_by_animal, axis=0)

    data_abl_average_by_animal = np.mean(data_rtd_by_animal, axis=1)
    model_abl_average_by_animal = np.mean(model_rtd_by_animal, axis=1)
    data_abl_mean, data_abl_sem, data_abl_n = mean_sem(
        data_abl_average_by_animal,
        axis=0,
    )
    model_abl_mean, model_abl_sem, model_abl_n = mean_sem(
        model_abl_average_by_animal,
        axis=0,
    )

    data_ild_average_by_animal = np.mean(data_rtd_by_animal, axis=2)
    model_ild_average_by_animal = np.mean(model_rtd_by_animal, axis=2)
    data_ild_mean, data_ild_sem, data_ild_n = mean_sem(
        data_ild_average_by_animal,
        axis=0,
    )
    model_ild_mean, model_ild_sem, model_ild_n = mean_sem(
        model_ild_average_by_animal,
        axis=0,
    )

    data_grand_average_by_animal = np.mean(data_rtd_by_animal, axis=(1, 2))
    model_grand_average_by_animal = np.mean(model_rtd_by_animal, axis=(1, 2))
    data_grand_mean, data_grand_sem, data_grand_n = mean_sem(
        data_grand_average_by_animal,
        axis=0,
    )
    model_grand_mean, model_grand_sem, model_grand_n = mean_sem(
        model_grand_average_by_animal,
        axis=0,
    )

    area_checks = {
        "animal condition data": np.sum(
            data_rtd_by_animal * np.diff(data_bins_s),
            axis=-1,
        ),
        "animal condition model": trapezoid(
            model_rtd_by_animal,
            rt_grid_s,
            axis=-1,
        ),
        "animal ABL-average data": np.sum(
            data_abl_average_by_animal * np.diff(data_bins_s),
            axis=-1,
        ),
        "animal ABL-average model": trapezoid(
            model_abl_average_by_animal,
            rt_grid_s,
            axis=-1,
        ),
        "animal ILD-average data": np.sum(
            data_ild_average_by_animal * np.diff(data_bins_s),
            axis=-1,
        ),
        "animal ILD-average model": trapezoid(
            model_ild_average_by_animal,
            rt_grid_s,
            axis=-1,
        ),
        "animal grand-average data": np.sum(
            data_grand_average_by_animal * np.diff(data_bins_s),
            axis=-1,
        ),
        "animal grand-average model": trapezoid(
            model_grand_average_by_animal,
            rt_grid_s,
            axis=-1,
        ),
    }
    for label, areas in area_checks.items():
        if not np.allclose(areas, 1.0, atol=1e-10, rtol=0):
            raise RuntimeError(
                f"{process_mode} {label} areas differ from one: "
                f"{np.min(areas):.12f}--{np.max(areas):.12f}."
            )

    contributor_checks = {
        "condition data": data_n,
        "condition model": model_n,
        "ABL-average data": data_abl_n,
        "ABL-average model": model_abl_n,
        "ILD-average data": data_ild_n,
        "ILD-average model": model_ild_n,
        "grand-average data": data_grand_n,
        "grand-average model": model_grand_n,
    }
    for label, contributors in contributor_checks.items():
        if not np.all(contributors == n_animals):
            raise RuntimeError(
                f"{process_mode} {label} does not have six animal contributors."
            )

    column_labels = [
        f"|ILD| = {abs_ild:g}" for abs_ild in ABS_ILDS
    ] + ["|ILD| average"]
    plot_rows = []
    for abl_idx, abl in enumerate(ABLS):
        plot_rows.append(
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
    plot_rows.append(
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
                alpha=0.10,
                linewidth=0,
                zorder=1,
            )
            ax.stairs(
                data_curve,
                data_bins_s * 1e3,
                color="black",
                linewidth=0.8,
                alpha=0.62,
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
                zorder=2,
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

            ax.set_xlim(0, DISPLAY_RT_MAX_MS)
            ax.set_xticks((0, 300, 600))
            ax.set_ylim(0, global_y_max * 1.06)
            ax.tick_params(axis="both", labelsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=0.8,
            alpha=0.62,
            label="Data mean +/- SEM",
        ),
        Line2D(
            [0],
            [0],
            color="#0072B2",
            linewidth=1.6,
            label="RT-only model mean +/- SEM",
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
        f"LED7 RT-only NPL+alpha SVI RTDs: {fit_config['label']} "
        "(0--600 ms x-axis)",
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
    fig.savefig(fit_config["figure"], dpi=PLOT_DPI, bbox_inches="tight")
    print(f"Saved figure: {fit_config['figure']}")

    payload = {
        "batch_name": BATCH_NAME,
        "process_mode": process_mode,
        "animals": ANIMALS,
        "abls": ABLS,
        "signed_ilds": SIGNED_ILDS,
        "abs_ilds": ABS_ILDS,
        "rt_grid_s": rt_grid_s,
        "data_bins_s": data_bins_s,
        "data_bin_centers_s": data_bin_centers_s,
        "data_bin_width_s": DATA_BIN_S,
        "model_step_s": MODEL_STEP_S,
        "display_rt_max_ms": DISPLAY_RT_MAX_MS,
        "data_rtd_by_animal": data_rtd_by_animal,
        "model_rtd_by_animal": model_rtd_by_animal,
        "data_mean": data_mean,
        "data_sem": data_sem,
        "model_mean": model_mean,
        "model_sem": model_sem,
        "data_abl_average_by_animal": data_abl_average_by_animal,
        "model_abl_average_by_animal": model_abl_average_by_animal,
        "data_ild_average_by_animal": data_ild_average_by_animal,
        "model_ild_average_by_animal": model_ild_average_by_animal,
        "data_grand_average_by_animal": data_grand_average_by_animal,
        "model_grand_average_by_animal": model_grand_average_by_animal,
        "area_checks": area_checks,
        "parameter_rows": pd.DataFrame(parameter_rows),
        "delay_rows": pd.DataFrame(delay_rows),
        "data_csv": str(DATA_CSV.relative_to(REPO_ROOT)),
        "fit_root": str(fit_root.relative_to(REPO_ROOT)),
        "data_trial_pool": (
            "Exact successful trials with 0 <= RTwrtStim < 1 s used by the "
            "RT-only fits; no abort trials are included."
        ),
        "model_construction": (
            "Posterior-mean choice-collapsed RT-only density normalized on "
            "0--1 s. Proactive+reactive curves are averaged over every "
            "fitting trial's intended_fix value within signed condition."
        ),
        "sign_average": (
            "Normalize each signed-condition RTD, then average -ILD and +ILD "
            "equally within animal."
        ),
        "animal_average": "Equal mean and SEM across six LED7 animals.",
        "abl_average": "Equal ABL average within animal before animal averaging.",
        "ild_average": "Equal |ILD| average within animal before animal averaging.",
    }
    with fit_config["payload"].open("wb") as handle:
        pickle.dump(payload, handle)
    print(f"Saved payload: {fit_config['payload']}")


# %%
# =============================================================================
# Save compact likelihood and completion audit
# =============================================================================
AUDIT_CSV = OUTPUT_DIR / "led7_npl_alpha_rt_only_fit_aligned_rtd_audit.csv"
audit_df = pd.DataFrame(audit_rows)
audit_df.to_csv(AUDIT_CSV, index=False)
print("\nLikelihood checks:")
print(
    audit_df[
        [
            "process_mode",
            "animal",
            "n_retained_trials",
            "best_step",
            "final_checked_step",
            "optimized_vs_likelihood_relative_max",
        ]
    ].to_string(index=False)
)
print(f"Saved audit: {AUDIT_CSV}")

# %%
