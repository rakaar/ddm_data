# %%
"""Validate separate LED-OFF/ON proactive + NPL+alpha likelihoods.

The simulator is independent of the analytic proactive PDF/CDF.  Four figures
are produced: LED OFF and LED ON, each without and with direct random-choice
exponential lapses.  Choice +1 is plotted above zero and choice -1 below.  All
cases use the same stratified LED7/93 stimulus-timing rows.
"""

# %%
from functools import partial
from pathlib import Path
import hashlib
import json
import os
import pickle
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from joblib import Parallel, delayed
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
ANIMAL_FIT_DIR = REPO_DIR / "fit_animal_by_animal"
for import_dir in [SCRIPT_DIR, ANIMAL_FIT_DIR]:
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

import numpyro_proactive_led_step_jump_npl_alpha_utils as combined
import numpyro_proactive_led_step_jump_svi_utils as step_jump
from simulate_proactive_led_step_jump_npl_alpha import simulate_led_category
from time_vary_norm_alpha_utils import (
    CDF_E_minus_small_t_NORM_alpha_time_varying_fn,
    rho_E_minus_small_t_NORM_alpha_time_varying_fn,
)


# %%
# =============================================================================
# Editable validation settings
# =============================================================================
DATA_CSV = REPO_DIR / "out_LED.csv"
PROACTIVE_FIT_DIR = (
    SCRIPT_DIR
    / "numpyro_svi_proactive_led_step_jump_all_on_no_trunc_exp_lapse_"
    "patience12_min50k_restore_best_outputs"
    / "LED7_93"
)
NPL_FIT_DIR = (
    ANIMAL_FIT_DIR
    / "numpyro_svi_npl_alpha_condition_delay_patience12_restore_best_outputs"
    / "LED7_93"
)
OUTPUT_DIR = SCRIPT_DIR / "proactive_led_step_jump_npl_alpha_validation"

N_SIM = int(float(os.environ.get("N_SIM_OVERRIDE", 100_000)))
SIM_DT_S = float(os.environ.get("SIM_DT_OVERRIDE", 0.0001))
N_JOBS = int(os.environ.get("N_JOBS_OVERRIDE", min(4, os.cpu_count() or 1)))
FORCE_RESIMULATE = os.environ.get("FORCE_RESIMULATE", "0") == "1"
BASE_SEED = 20260828
N_REALISTIC_TIMING_ROWS = 128
HIST_BIN_S = 0.005
THEORY_DT_S = 0.0005
THEORY_CHUNK = 128
K_MAX = 10
N_QUAD = 64
RAW_TAIL_TARGET = 1e-8
RT_LOW_S = 0.0
RT_HIGH_S = 1.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Proactive source: {PROACTIVE_FIT_DIR}")
print(f"NPL+alpha source: {NPL_FIT_DIR}")
print(f"Output: {OUTPUT_DIR}")
print(f"Simulation trials per cell: {N_SIM:,}")
print(f"Simulation dt: {1e3 * SIM_DT_S:.3f} ms")
print("LED OFF and LED ON likelihoods are evaluated separately.")


# %%
# =============================================================================
# Controlled and posterior-derived parameter cases
# =============================================================================
COMMON_SYNTHETIC = {
    "theta_A": 2.0,
    "del_a_minus_del_LED": 0.040,
    "del_m_plus_del_LED": 0.050,
    "lapse_prob": 0.10,
    "beta_lapse": 3.0,
    "rate_lambda": 2.0,
    "T_0": 0.150,
    "theta_E": 3.0,
    "w": 0.5,
    "del_go": 0.080,
    "rate_norm_l": 0.95,
    "alpha": 0.8,
}

case_configs = [
    {
        "case_key": "no_jump",
        "case_label": "No proactive drift jump",
        "ABL": 20.0,
        "ILD": 0.0,
        "t_E_aff": 0.080,
        "params": {
            **COMMON_SYNTHETIC,
            "V_A_base": 1.5,
            "V_A_post_LED": 1.5,
        },
    },
    {
        "case_key": "upward_jump",
        "case_label": "Upward proactive drift jump",
        "ABL": 40.0,
        "ILD": 4.0,
        "t_E_aff": 0.080,
        "params": {
            **COMMON_SYNTHETIC,
            "V_A_base": 1.2,
            "V_A_post_LED": 3.0,
        },
    },
]

proactive_file = PROACTIVE_FIT_DIR / "main_fullrank_posterior_samples.npz"
npl_file = NPL_FIT_DIR / "main_fullrank_posterior_samples.npz"
condition_file = NPL_FIT_DIR / "condition_table.csv"
for required_file in [DATA_CSV, proactive_file, npl_file, condition_file]:
    if not required_file.exists():
        raise FileNotFoundError(required_file)

proactive_names = [
    "V_A_base",
    "V_A_post_LED",
    "theta_A",
    "del_a_minus_del_LED",
    "del_m_plus_del_LED",
    "lapse_prob",
    "beta_lapse",
]
reactive_names = [
    "rate_lambda",
    "T_0",
    "theta_E",
    "w",
    "del_go",
    "rate_norm_l",
    "alpha",
]
with np.load(proactive_file) as posterior:
    realistic_params = {
        name: float(np.mean(posterior[name])) for name in proactive_names
    }
with np.load(npl_file) as posterior:
    realistic_params.update(
        {name: float(np.mean(posterior[name])) for name in reactive_names}
    )
    condition_table = pd.read_csv(condition_file)
    condition_row = condition_table[
        np.isclose(condition_table["ABL"], 40.0)
        & np.isclose(condition_table["ILD"], 4.0)
    ]
    if len(condition_row) != 1:
        raise RuntimeError("Expected one LED7/93 ABL 40, ILD +4 condition.")
    condition_id = int(condition_row.iloc[0]["condition_id"])
    realistic_t_E_aff = float(
        np.mean(posterior["t_E_aff"][:, condition_id])
    )

case_configs.append(
    {
        "case_key": "led7_93_posterior",
        "case_label": "LED7/93 posterior means",
        "ABL": 40.0,
        "ILD": 4.0,
        "t_E_aff": realistic_t_E_aff,
        "params": realistic_params,
        "provenance": {
            "proactive": str(proactive_file.relative_to(REPO_DIR)),
            "reactive": str(npl_file.relative_to(REPO_DIR)),
            "condition_id": condition_id,
        },
    }
)


# %%
# =============================================================================
# Timing banks: fixed medians for synthetic cases, intact rows for LED7/93
# =============================================================================
raw_df = pd.read_csv(DATA_CSV)
valid_df = raw_df[
    (raw_df["animal"].astype(int) == 93)
    & (raw_df["session_type"] == 7)
    & (raw_df["training_level"] == 16)
    & raw_df["success"].isin([1, -1])
    & (raw_df["repeat_trial"].isin([0, 2]) | raw_df["repeat_trial"].isna())
].dropna(subset=["intended_fix", "LED_onset_time", "timed_fix"])
real_on_df = valid_df[valid_df["LED_trial"] == 1].copy()
real_off_df = valid_df[
    (valid_df["LED_trial"] == 0) | valid_df["LED_trial"].isna()
].copy()
if real_on_df.empty or real_off_df.empty:
    raise RuntimeError("LED7/93 needs both LED-ON and LED-OFF valid rows.")


def stratified_timing_bank(frame, led_on):
    ordered = frame.sort_values("intended_fix").reset_index(drop=True)
    indices = np.linspace(
        0,
        len(ordered) - 1,
        N_REALISTIC_TIMING_ROWS,
    ).round().astype(int)
    selected = ordered.iloc[indices]
    t_stim = selected["intended_fix"].to_numpy(dtype=float)
    if led_on:
        t_LED = (
            selected["intended_fix"] - selected["LED_onset_time"]
        ).to_numpy(dtype=float)
    else:
        t_LED = np.zeros(len(selected), dtype=float)
    return {"t_stim": t_stim, "t_LED": t_LED}


realistic_timing_banks = {
    "off": stratified_timing_bank(real_off_df, False),
    "on": stratified_timing_bank(real_on_df, True),
}
for case in case_configs:
    case["timing_banks"] = realistic_timing_banks

print("\nShared LED7/93 timing-bank ranges:")
for category, timing in realistic_timing_banks.items():
    print(
        f"  LED {category.upper()}: t_stim "
        f"{np.min(timing['t_stim']):.3f}.."
        f"{np.max(timing['t_stim']):.3f} s"
    )

parameter_rows = []
for case in case_configs:
    parameter_rows.append(
        {
            "case_key": case["case_key"],
            "case_label": case["case_label"],
            "ABL": case["ABL"],
            "ILD": case["ILD"],
            "t_E_aff": case["t_E_aff"],
            **case["params"],
        }
    )
parameter_df = pd.DataFrame(parameter_rows)
parameter_csv = OUTPUT_DIR / "parameter_cases.csv"
parameter_df.to_csv(parameter_csv, index=False)
print("\nParameter cases:")
print(
    parameter_df[
        [
            "case_label",
            "ABL",
            "ILD",
            "V_A_base",
            "V_A_post_LED",
            "lapse_prob",
            "rate_lambda",
            "alpha",
            "t_E_aff",
        ]
    ].to_string(index=False, float_format=lambda value: f"{value:.6g}")
)


# %%
# =============================================================================
# Independent simulations with deterministic caches
# =============================================================================
def timing_hash(timing_bank):
    digest = hashlib.sha256()
    digest.update(np.asarray(timing_bank["t_stim"]).tobytes())
    digest.update(np.asarray(timing_bank["t_LED"]).tobytes())
    return digest.hexdigest()[:12]


scenario_configs = []
for case_index, case in enumerate(case_configs):
    for category_index, category in enumerate(["off", "on"]):
        for lapse_index, include_lapse in enumerate([False, True]):
            scenario_configs.append(
                {
                    "case_index": case_index,
                    "case": case,
                    "category": category,
                    "include_lapse": include_lapse,
                    "seed": (
                        BASE_SEED
                        + 1000 * case_index
                        + 100 * category_index
                        + 10 * lapse_index
                    ),
                }
            )


def simulate_scenario(scenario):
    case = scenario["case"]
    timing_bank = case["timing_banks"][scenario["category"]]
    rng = np.random.default_rng(scenario["seed"])
    timing_index = np.resize(
        np.arange(len(timing_bank["t_stim"]), dtype=np.int16),
        N_SIM,
    )
    rng.shuffle(timing_index)
    t_stim = timing_bank["t_stim"][timing_index]
    t_LED = timing_bank["t_LED"][timing_index]
    start = time.perf_counter()
    result = simulate_led_category(
        case["params"],
        case["ABL"],
        case["ILD"],
        case["t_E_aff"],
        t_stim,
        t_LED,
        scenario["category"] == "on",
        N_SIM,
        scenario["include_lapse"],
        dt=SIM_DT_S,
        seed=scenario["seed"],
    )
    result["timing_index"] = timing_index
    result["runtime_s"] = time.perf_counter() - start
    return result


# Compile the Numba kernels before worker processes start.
simulate_led_category(
    case_configs[0]["params"],
    case_configs[0]["ABL"],
    case_configs[0]["ILD"],
    case_configs[0]["t_E_aff"],
    np.full(2, 0.4182),
    np.full(2, 0.2337),
    True,
    2,
    False,
    dt=SIM_DT_S,
    seed=BASE_SEED,
)

simulation_results = {}
missing = []
for scenario in scenario_configs:
    case = scenario["case"]
    lapse_tag = "lapse" if scenario["include_lapse"] else "no_lapse"
    cache_path = OUTPUT_DIR / (
        f"sim_{case['case_key']}_{scenario['category']}_{lapse_tag}_"
        f"n{N_SIM}_dt{int(round(1e6 * SIM_DT_S))}us.npz"
    )
    scenario["cache_path"] = cache_path
    config = {
        "case": case["case_key"],
        "params": case["params"],
        "ABL": case["ABL"],
        "ILD": case["ILD"],
        "t_E_aff": case["t_E_aff"],
        "category": scenario["category"],
        "include_lapse": scenario["include_lapse"],
        "n_sim": N_SIM,
        "dt": SIM_DT_S,
        "seed": scenario["seed"],
        "timing_hash": timing_hash(case["timing_banks"][scenario["category"]]),
    }
    scenario["config_json"] = json.dumps(config, sort_keys=True)
    loaded = False
    if cache_path.exists() and not FORCE_RESIMULATE:
        with np.load(cache_path, allow_pickle=False) as cached:
            if str(cached["config_json"].item()) == scenario["config_json"]:
                simulation_results[
                    (
                        scenario["case_index"],
                        scenario["category"],
                        scenario["include_lapse"],
                    )
                ] = {
                    name: cached[name]
                    for name in [
                        "choice",
                        "total_fix",
                        "rt_stim",
                        "source",
                        "t_stim",
                        "t_LED",
                        "timing_index",
                    ]
                } | {"runtime_s": float(cached["runtime_s"])}
                loaded = True
    if not loaded:
        missing.append(scenario)

print(
    f"\nLoaded {len(scenario_configs) - len(missing)} of "
    f"{len(scenario_configs)} simulation caches."
)
if missing:
    print(f"Simulating {len(missing)} cells with {N_JOBS} workers...")
    results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(simulate_scenario)(scenario) for scenario in missing
    )
    for scenario, result in zip(missing, results):
        key = (
            scenario["case_index"],
            scenario["category"],
            scenario["include_lapse"],
        )
        simulation_results[key] = result
        np.savez_compressed(
            scenario["cache_path"],
            config_json=np.asarray(scenario["config_json"]),
            choice=result["choice"],
            total_fix=result["total_fix"],
            rt_stim=result["rt_stim"],
            source=result["source"],
            t_stim=result["t_stim"],
            t_LED=result["t_LED"],
            timing_index=result["timing_index"],
            runtime_s=np.asarray(result["runtime_s"]),
        )
        print(
            f"  {scenario['case']['case_label']}, "
            f"LED {scenario['category'].upper()}, "
            f"lapse={scenario['include_lapse']}: "
            f"{result['runtime_s']:.1f} s"
        )


# %%
# =============================================================================
# Batched analytic densities, kept separate for LED OFF and LED ON
# =============================================================================
@partial(jax.jit, static_argnames=("include_lapse", "normalized"))
def off_mean_density(
    rt_chunk,
    bound,
    params,
    t_stim,
    ABL,
    ILD,
    t_E_aff,
    normalizer,
    include_lapse,
    normalized,
):
    total_fix = t_stim[:, None] + rt_chunk[None, :]
    density = combined.led_off_npl_alpha_choice_pdf_jax(
        total_fix,
        bound,
        params,
        t_stim[:, None],
        ABL,
        ILD,
        t_E_aff,
        K_max=K_MAX,
        include_lapse=include_lapse,
    )
    if normalized:
        density = density / normalizer[:, None]
    return jnp.mean(density, axis=0)


@partial(jax.jit, static_argnames=("include_lapse", "normalized"))
def on_mean_density(
    rt_chunk,
    bound,
    params,
    t_stim,
    t_LED,
    ABL,
    ILD,
    t_E_aff,
    normalizer,
    include_lapse,
    normalized,
):
    total_fix = t_stim[:, None] + rt_chunk[None, :]
    density = combined.led_on_npl_alpha_choice_pdf_jax(
        total_fix,
        bound,
        params,
        t_stim[:, None],
        t_LED[:, None],
        ABL,
        ILD,
        t_E_aff,
        K_max=K_MAX,
        n_quad=N_QUAD,
        include_lapse=include_lapse,
    )
    if normalized:
        density = density / normalizer[:, None]
    return jnp.mean(density, axis=0)


def timing_window_normalizers(case, category, include_lapse):
    timing = case["timing_banks"][category]
    t_stim = jnp.asarray(timing["t_stim"])
    if category == "off":
        cdf = combined.led_off_npl_alpha_total_cdf_jax
        upper = cdf(
            t_stim + RT_HIGH_S,
            case["params"],
            t_stim,
            case["ABL"],
            case["ILD"],
            case["t_E_aff"],
            K_max=K_MAX,
            include_lapse=include_lapse,
        )
        lower = cdf(
            t_stim + RT_LOW_S,
            case["params"],
            t_stim,
            case["ABL"],
            case["ILD"],
            case["t_E_aff"],
            K_max=K_MAX,
            include_lapse=include_lapse,
        )
    else:
        t_LED = jnp.asarray(timing["t_LED"])
        cdf = combined.led_on_npl_alpha_total_cdf_jax
        upper = cdf(
            t_stim + RT_HIGH_S,
            case["params"],
            t_stim,
            t_LED,
            case["ABL"],
            case["ILD"],
            case["t_E_aff"],
            K_max=K_MAX,
            n_quad=N_QUAD,
            include_lapse=include_lapse,
        )
        lower = cdf(
            t_stim + RT_LOW_S,
            case["params"],
            t_stim,
            t_LED,
            case["ABL"],
            case["ILD"],
            case["t_E_aff"],
            K_max=K_MAX,
            n_quad=N_QUAD,
            include_lapse=include_lapse,
        )
    values = np.asarray(upper - lower)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise RuntimeError("Invalid per-timing valid-window normalizer.")
    return values


def mean_total_cdf(case, category, include_lapse, rt_stim):
    timing = case["timing_banks"][category]
    t_stim = jnp.asarray(timing["t_stim"])
    total_fix = t_stim + rt_stim
    if category == "off":
        values = combined.led_off_npl_alpha_total_cdf_jax(
            total_fix,
            case["params"],
            t_stim,
            case["ABL"],
            case["ILD"],
            case["t_E_aff"],
            K_max=K_MAX,
            include_lapse=include_lapse,
        )
    else:
        values = combined.led_on_npl_alpha_total_cdf_jax(
            total_fix,
            case["params"],
            t_stim,
            jnp.asarray(timing["t_LED"]),
            case["ABL"],
            case["ILD"],
            case["t_E_aff"],
            K_max=K_MAX,
            n_quad=N_QUAD,
            include_lapse=include_lapse,
        )
    return float(jnp.mean(values))


def evaluate_mean_density(
    rt_grid,
    bound,
    case,
    category,
    include_lapse,
    normalizer,
    normalized,
):
    timing = case["timing_banks"][category]
    pieces = []
    for start in range(0, len(rt_grid), THEORY_CHUNK):
        values = rt_grid[start : start + THEORY_CHUNK]
        valid_length = len(values)
        if valid_length < THEORY_CHUNK:
            values = np.pad(
                values,
                (0, THEORY_CHUNK - valid_length),
                mode="edge",
            )
        if category == "off":
            result = off_mean_density(
                jnp.asarray(values),
                bound,
                case["params"],
                jnp.asarray(timing["t_stim"]),
                case["ABL"],
                case["ILD"],
                case["t_E_aff"],
                jnp.asarray(normalizer),
                include_lapse=include_lapse,
                normalized=normalized,
            )
        else:
            result = on_mean_density(
                jnp.asarray(values),
                bound,
                case["params"],
                jnp.asarray(timing["t_stim"]),
                jnp.asarray(timing["t_LED"]),
                case["ABL"],
                case["ILD"],
                case["t_E_aff"],
                jnp.asarray(normalizer),
                include_lapse=include_lapse,
                normalized=normalized,
            )
        pieces.append(np.asarray(result)[:valid_length])
    return np.concatenate(pieces)


# %%
# =============================================================================
# Raw and fitted-window validation metrics
# =============================================================================
summary_rows = []
plot_payload = {}

for scenario in scenario_configs:
    case_index = scenario["case_index"]
    case = scenario["case"]
    category = scenario["category"]
    include_lapse = scenario["include_lapse"]
    key = (case_index, category, include_lapse)
    simulation = simulation_results[key]
    timing = case["timing_banks"][category]
    normalizer = timing_window_normalizers(
        case, category, include_lapse
    )

    raw_min = min(
        float(np.min(simulation["rt_stim"])) - HIST_BIN_S,
        float(
            min(0.0, case["params"]["del_a_minus_del_LED"] + case["params"]["del_m_plus_del_LED"])
            - np.max(timing["t_stim"])
            - HIST_BIN_S
        ),
    )
    raw_max = max(3.0, float(np.max(simulation["rt_stim"])) + HIST_BIN_S)
    while mean_total_cdf(
        case, category, include_lapse, raw_max
    ) < 1.0 - RAW_TAIL_TARGET:
        raw_max = 1.5 * raw_max + 0.5
        if raw_max > 30.0:
            raise RuntimeError("Could not cover the analytic response tail.")
    raw_min = HIST_BIN_S * np.floor(raw_min / HIST_BIN_S)
    raw_max = HIST_BIN_S * np.ceil(raw_max / HIST_BIN_S)
    raw_grid = np.arange(
        raw_min,
        raw_max + 0.5 * THEORY_DT_S,
        THEORY_DT_S,
    )
    raw_edges = np.arange(
        raw_min,
        raw_max + 1.01 * HIST_BIN_S,
        HIST_BIN_S,
    )
    raw_centers = 0.5 * (raw_edges[:-1] + raw_edges[1:])

    window_grid = np.arange(
        RT_LOW_S,
        RT_HIGH_S + 0.5 * THEORY_DT_S,
        THEORY_DT_S,
    )
    window_edges = np.arange(
        RT_LOW_S,
        RT_HIGH_S + 1.01 * HIST_BIN_S,
        HIST_BIN_S,
    )
    window_centers = 0.5 * (window_edges[:-1] + window_edges[1:])

    cell = {
        "raw_grid": raw_grid,
        "raw_centers": raw_centers,
        "window_grid": window_grid,
        "window_centers": window_centers,
    }
    metrics = {
        "case_index": case_index,
        "case_key": case["case_key"],
        "case_label": case["case_label"],
        "category": category,
        "include_lapse": include_lapse,
        "n_sim": N_SIM,
        "simulation_runtime_s": simulation["runtime_s"],
        "mean_window_normalizer": float(np.mean(normalizer)),
        "minimum_window_normalizer": float(np.min(normalizer)),
        "raw_tail_mass": 1.0
        - mean_total_cdf(case, category, include_lapse, raw_max),
    }

    in_window = (
        (simulation["rt_stim"] >= RT_LOW_S)
        & (simulation["rt_stim"] < RT_HIGH_S)
    )
    inverse_normalizer_weights = 1.0 / normalizer[
        simulation["timing_index"]
    ]

    for bound, suffix in [(1, "plus"), (-1, "minus")]:
        bound_mask = simulation["choice"] == bound
        raw_counts, _ = np.histogram(
            simulation["rt_stim"][bound_mask], bins=raw_edges
        )
        raw_hist = raw_counts / (N_SIM * HIST_BIN_S)
        raw_theory = evaluate_mean_density(
            raw_grid,
            bound,
            case,
            category,
            include_lapse,
            np.ones_like(normalizer),
            normalized=False,
        )
        raw_theory_cdf = cumulative_trapezoid(
            raw_theory, raw_grid, initial=0.0
        )
        raw_sim_cdf = np.cumsum(raw_hist) * HIST_BIN_S
        raw_theory_at_hist = np.interp(
            raw_centers, raw_grid, raw_theory_cdf
        )

        window_mask = bound_mask & in_window
        window_counts, _ = np.histogram(
            simulation["rt_stim"][window_mask],
            bins=window_edges,
            weights=inverse_normalizer_weights[window_mask],
        )
        window_hist = window_counts / (N_SIM * HIST_BIN_S)
        window_theory = evaluate_mean_density(
            window_grid,
            bound,
            case,
            category,
            include_lapse,
            normalizer,
            normalized=True,
        )
        window_theory_cdf = cumulative_trapezoid(
            window_theory, window_grid, initial=0.0
        )
        window_sim_cdf = np.cumsum(window_hist) * HIST_BIN_S
        window_theory_at_hist = np.interp(
            window_centers, window_grid, window_theory_cdf
        )

        cell[f"raw_hist_{suffix}"] = raw_hist
        cell[f"raw_theory_{suffix}"] = raw_theory
        cell[f"window_hist_{suffix}"] = window_hist
        cell[f"window_theory_{suffix}"] = window_theory

        metrics[f"raw_sim_area_{suffix}"] = float(np.mean(bound_mask))
        metrics[f"raw_theory_area_{suffix}"] = float(raw_theory_cdf[-1])
        metrics[f"raw_subcdf_error_{suffix}"] = float(
            np.max(np.abs(raw_sim_cdf - raw_theory_at_hist))
        )
        metrics[f"window_sim_area_{suffix}"] = float(
            np.sum(inverse_normalizer_weights[window_mask]) / N_SIM
        )
        metrics[f"window_theory_area_{suffix}"] = float(
            window_theory_cdf[-1]
        )
        metrics[f"window_subcdf_error_{suffix}"] = float(
            np.max(np.abs(window_sim_cdf - window_theory_at_hist))
        )
        metrics[f"minimum_density_{suffix}"] = float(
            min(np.min(raw_theory), np.min(window_theory))
        )

    metrics["raw_sim_total"] = (
        metrics["raw_sim_area_plus"] + metrics["raw_sim_area_minus"]
    )
    metrics["raw_theory_total"] = (
        metrics["raw_theory_area_plus"]
        + metrics["raw_theory_area_minus"]
    )
    metrics["window_sim_total"] = (
        metrics["window_sim_area_plus"]
        + metrics["window_sim_area_minus"]
    )
    metrics["window_theory_total"] = (
        metrics["window_theory_area_plus"]
        + metrics["window_theory_area_minus"]
    )
    metrics["max_raw_choice_area_error"] = max(
        abs(metrics["raw_sim_area_plus"] - metrics["raw_theory_area_plus"]),
        abs(metrics["raw_sim_area_minus"] - metrics["raw_theory_area_minus"]),
    )
    metrics["max_window_choice_area_error"] = max(
        abs(
            metrics["window_sim_area_plus"]
            - metrics["window_theory_area_plus"]
        ),
        abs(
            metrics["window_sim_area_minus"]
            - metrics["window_theory_area_minus"]
        ),
    )
    metrics["max_raw_subcdf_error"] = max(
        metrics["raw_subcdf_error_plus"],
        metrics["raw_subcdf_error_minus"],
    )
    metrics["max_window_subcdf_error"] = max(
        metrics["window_subcdf_error_plus"],
        metrics["window_subcdf_error_minus"],
    )
    metrics["minimum_density"] = min(
        metrics["minimum_density_plus"],
        metrics["minimum_density_minus"],
    )
    metrics["pass_raw_theory_mass"] = (
        abs(metrics["raw_theory_total"] - 1.0) <= 0.002
    )
    metrics["pass_raw_choice_areas"] = (
        metrics["max_raw_choice_area_error"] <= 0.01
    )
    metrics["pass_raw_subcdf"] = metrics["max_raw_subcdf_error"] <= 0.01
    metrics["pass_window_theory_mass"] = (
        abs(metrics["window_theory_total"] - 1.0) <= 0.002
    )
    metrics["pass_window_sim_mass"] = (
        abs(metrics["window_sim_total"] - 1.0) <= 0.02
    )
    metrics["pass_window_choice_areas"] = (
        metrics["max_window_choice_area_error"] <= 0.015
    )
    metrics["pass_window_subcdf"] = (
        metrics["max_window_subcdf_error"] <= 0.015
    )
    metrics["pass_nonnegative"] = metrics["minimum_density"] >= -1e-8
    pass_columns = [
        "pass_raw_theory_mass",
        "pass_raw_choice_areas",
        "pass_raw_subcdf",
        "pass_window_theory_mass",
        "pass_window_sim_mass",
        "pass_window_choice_areas",
        "pass_window_subcdf",
        "pass_nonnegative",
    ]
    metrics["pass_all"] = all(metrics[name] for name in pass_columns)
    summary_rows.append(metrics)
    plot_payload[key] = cell
    print(
        f"Computed {case['case_label']}, LED {category.upper()}, "
        f"lapse={include_lapse}: raw total={metrics['raw_theory_total']:.6f}, "
        f"window total={metrics['window_theory_total']:.6f}, "
        f"max CDF errors={metrics['max_raw_subcdf_error']:.4f}/"
        f"{metrics['max_window_subcdf_error']:.4f}, pass={metrics['pass_all']}"
    )

summary_df = pd.DataFrame(summary_rows).sort_values(
    ["include_lapse", "category", "case_index"]
)
summary_csv = OUTPUT_DIR / "simulation_vs_likelihood_summary.csv"
summary_df.to_csv(summary_csv, index=False)


# %%
# =============================================================================
# Component parity and gradient checks
# =============================================================================
def numpy_evidence_cdf(t, bound, case):
    p = case["params"]
    z_e = (p["w"] - 0.5) * 2.0 * p["theta_E"]
    return CDF_E_minus_small_t_NORM_alpha_time_varying_fn(
        t,
        bound,
        case["ABL"],
        case["ILD"],
        p["rate_lambda"],
        p["T_0"],
        p["theta_E"],
        z_e,
        np.nan,
        p["rate_norm_l"],
        p["alpha"],
        True,
        False,
        K_MAX,
    )


def numpy_evidence_pdf(t, bound, case):
    p = case["params"]
    z_e = (p["w"] - 0.5) * 2.0 * p["theta_E"]
    return rho_E_minus_small_t_NORM_alpha_time_varying_fn(
        t,
        bound,
        case["ABL"],
        case["ILD"],
        p["rate_lambda"],
        p["T_0"],
        p["theta_E"],
        z_e,
        np.nan,
        np.nan,
        p["rate_norm_l"],
        p["alpha"],
        True,
        False,
        K_MAX,
    )


parity_errors = []
for case in case_configs:
    for category in ["off", "on"]:
        timing = case["timing_banks"][category]
        t_stim = float(timing["t_stim"][len(timing["t_stim"]) // 2])
        t_LED = float(timing["t_LED"][len(timing["t_LED"]) // 2])
        p = case["params"]
        for include_lapse in [False, True]:
            for bound in [1, -1]:
                for rt_stim in [0.03, 0.15, 0.40]:
                    t = t_stim + rt_stim
                    if category == "off":
                        proactive_pdf = float(
                            step_jump.led_off_pdf_jax(
                                t,
                                p["V_A_base"],
                                p["theta_A"],
                                p["del_a_minus_del_LED"],
                                p["del_m_plus_del_LED"],
                            )
                        )
                        proactive_cdf = float(
                            step_jump.led_off_cdf_jax(
                                t,
                                p["V_A_base"],
                                p["theta_A"],
                                p["del_a_minus_del_LED"],
                                p["del_m_plus_del_LED"],
                            )
                        )
                        jax_pdf = combined.led_off_npl_alpha_choice_pdf_jax(
                            t,
                            bound,
                            p,
                            t_stim,
                            case["ABL"],
                            case["ILD"],
                            case["t_E_aff"],
                            include_lapse=include_lapse,
                        )
                    else:
                        proactive_pdf = float(
                            step_jump.led_on_pdf_jax(
                                t,
                                t_LED,
                                p["V_A_base"],
                                p["V_A_post_LED"],
                                p["theta_A"],
                                p["del_a_minus_del_LED"],
                                p["del_m_plus_del_LED"],
                            )
                        )
                        proactive_cdf = float(
                            step_jump.led_on_cdf_jax(
                                t,
                                t_LED,
                                p["V_A_base"],
                                p["V_A_post_LED"],
                                p["theta_A"],
                                p["del_a_minus_del_LED"],
                                p["del_m_plus_del_LED"],
                                n_quad=N_QUAD,
                            )
                        )
                        jax_pdf = combined.led_on_npl_alpha_choice_pdf_jax(
                            t,
                            bound,
                            p,
                            t_stim,
                            t_LED,
                            case["ABL"],
                            case["ILD"],
                            case["t_E_aff"],
                            include_lapse=include_lapse,
                        )

                    evidence_time = t - t_stim - case["t_E_aff"]
                    evidence_time_go = evidence_time + p["del_go"]
                    cdf_up_go = numpy_evidence_cdf(
                        evidence_time_go, 1, case
                    )
                    cdf_down_go = numpy_evidence_cdf(
                        evidence_time_go, -1, case
                    )
                    ordinary = proactive_pdf * (
                        0.5 * max(0.0, 1.0 - cdf_up_go - cdf_down_go)
                        + numpy_evidence_cdf(
                            evidence_time_go, bound, case
                        )
                        - numpy_evidence_cdf(evidence_time, bound, case)
                    ) + numpy_evidence_pdf(
                        evidence_time, bound, case
                    ) * (1.0 - proactive_cdf)
                    if include_lapse:
                        lapse_pdf = p["beta_lapse"] * np.exp(
                            -p["beta_lapse"] * t
                        )
                        reference = (
                            (1.0 - p["lapse_prob"]) * ordinary
                            + 0.5 * p["lapse_prob"] * lapse_pdf
                        )
                    else:
                        reference = ordinary
                    parity_errors.append(abs(float(jax_pdf) - reference))

max_parity_error = max(parity_errors)
if max_parity_error > 1e-8:
    raise AssertionError(f"JAX/NumPy PDF parity error {max_parity_error:.3e}")

gradient_case = case_configs[1]
gradient_names = [
    "rate_lambda",
    "T_0",
    "theta_E",
    "w",
    "del_go",
    "rate_norm_l",
    "alpha",
]
gradient_start = jnp.asarray(
    [gradient_case["params"][name] for name in gradient_names]
    + [gradient_case["t_E_aff"]]
)


def gradient_probe(vector, category, include_lapse):
    params = dict(gradient_case["params"])
    for index, name in enumerate(gradient_names):
        params[name] = vector[index]
    t_E_aff = vector[-1]
    timing = gradient_case["timing_banks"][category]
    t_stim = timing["t_stim"][0]
    total_fix = t_stim + 0.2
    if category == "off":
        value = combined.led_off_npl_alpha_choice_pdf_jax(
            total_fix,
            1,
            params,
            t_stim,
            gradient_case["ABL"],
            gradient_case["ILD"],
            t_E_aff,
            include_lapse=include_lapse,
        )
    else:
        value = combined.led_on_npl_alpha_choice_pdf_jax(
            total_fix,
            1,
            params,
            t_stim,
            timing["t_LED"][0],
            gradient_case["ABL"],
            gradient_case["ILD"],
            t_E_aff,
            include_lapse=include_lapse,
        )
    return jnp.log(jnp.maximum(value, combined.LIKELIHOOD_FLOOR))


for category in ["off", "on"]:
    for include_lapse in [False, True]:
        gradient = jax.grad(gradient_probe)(
            gradient_start, category, include_lapse
        )
        if not np.all(np.isfinite(np.asarray(gradient))):
            raise AssertionError(
                f"Non-finite {category} lapse={include_lapse} gradient."
            )

# Exact lapse-off identity and no-jump ON/OFF invariance.
identity_case = case_configs[0]
identity_t = jnp.linspace(0.05, 2.0, 500)
identity_stim = 0.4182
for category in ["off", "on"]:
    for bound in [1, -1]:
        if category == "off":
            ordinary = combined.led_off_npl_alpha_choice_pdf_jax(
                identity_t,
                bound,
                identity_case["params"],
                identity_stim,
                identity_case["ABL"],
                identity_case["ILD"],
                identity_case["t_E_aff"],
                include_lapse=False,
            )
            zero_lapse_params = {
                **identity_case["params"],
                "lapse_prob": 0.0,
            }
            zero_lapse = combined.led_off_npl_alpha_choice_pdf_jax(
                identity_t,
                bound,
                zero_lapse_params,
                identity_stim,
                identity_case["ABL"],
                identity_case["ILD"],
                identity_case["t_E_aff"],
                include_lapse=True,
            )
        else:
            ordinary = combined.led_on_npl_alpha_choice_pdf_jax(
                identity_t,
                bound,
                identity_case["params"],
                identity_stim,
                0.2337,
                identity_case["ABL"],
                identity_case["ILD"],
                identity_case["t_E_aff"],
                include_lapse=False,
            )
            zero_lapse_params = {
                **identity_case["params"],
                "lapse_prob": 0.0,
            }
            zero_lapse = combined.led_on_npl_alpha_choice_pdf_jax(
                identity_t,
                bound,
                zero_lapse_params,
                identity_stim,
                0.2337,
                identity_case["ABL"],
                identity_case["ILD"],
                identity_case["t_E_aff"],
                include_lapse=True,
            )
        if not np.array_equal(np.asarray(ordinary), np.asarray(zero_lapse)):
            raise AssertionError("lapse_prob=0 did not exactly recover no lapse.")

off_no_jump = combined.led_off_npl_alpha_choice_pdf_jax(
    identity_t,
    1,
    identity_case["params"],
    identity_stim,
    identity_case["ABL"],
    identity_case["ILD"],
    identity_case["t_E_aff"],
    include_lapse=False,
)
on_no_jump = combined.led_on_npl_alpha_choice_pdf_jax(
    identity_t,
    1,
    identity_case["params"],
    identity_stim,
    0.2337,
    identity_case["ABL"],
    identity_case["ILD"],
    identity_case["t_E_aff"],
    include_lapse=False,
)
no_jump_error = float(np.max(np.abs(off_no_jump - on_no_jump)))
if no_jump_error > 1e-7:
    raise AssertionError(f"No-jump ON/OFF mismatch {no_jump_error:.3e}")


# %%
# =============================================================================
# Four separate figures: OFF/ON and no-lapse/lapse
# =============================================================================
figure_paths = []
for category in ["off", "on"]:
    for include_lapse in [False, True]:
        color = "tab:blue" if category == "off" else "tab:red"
        fig, axes = plt.subplots(
            2,
            len(case_configs),
            figsize=(15.5, 6.2),
            sharey="row",
        )
        row_y_max = [0.0, 0.0]
        for case_index in range(len(case_configs)):
            cell = plot_payload[(case_index, category, include_lapse)]
            row_y_max[0] = max(
                row_y_max[0],
                np.max(cell["raw_hist_plus"]),
                np.max(cell["raw_hist_minus"]),
                np.max(cell["raw_theory_plus"]),
                np.max(cell["raw_theory_minus"]),
            )
            row_y_max[1] = max(
                row_y_max[1],
                np.max(cell["window_hist_plus"]),
                np.max(cell["window_hist_minus"]),
                np.max(cell["window_theory_plus"]),
                np.max(cell["window_theory_minus"]),
            )

        for case_index, case in enumerate(case_configs):
            cell = plot_payload[(case_index, category, include_lapse)]
            metrics = summary_df[
                (summary_df["case_index"] == case_index)
                & (summary_df["category"] == category)
                & (summary_df["include_lapse"] == include_lapse)
            ].iloc[0]
            for row_index, prefix in enumerate(["raw", "window"]):
                ax = axes[row_index, case_index]
                centers = cell[f"{prefix}_centers"]
                grid = cell[f"{prefix}_grid"]
                ax.step(
                    centers,
                    cell[f"{prefix}_hist_plus"],
                    where="mid",
                    color="black",
                    lw=0.9,
                    alpha=0.65,
                )
                ax.step(
                    centers,
                    -cell[f"{prefix}_hist_minus"],
                    where="mid",
                    color="black",
                    lw=0.9,
                    alpha=0.65,
                )
                ax.plot(
                    grid,
                    cell[f"{prefix}_theory_plus"],
                    color=color,
                    lw=1.5,
                    alpha=0.65,
                )
                ax.plot(
                    grid,
                    -cell[f"{prefix}_theory_minus"],
                    color=color,
                    lw=1.5,
                    alpha=0.65,
                )
                ax.axhline(0.0, color="0.25", lw=0.7)
                ax.axvline(0.0, color="0.65", lw=0.8, ls=":")
                ax.set_ylim(-1.08 * row_y_max[row_index], 1.08 * row_y_max[row_index])
                if prefix == "raw":
                    display_low = max(float(grid[0]), -1.5)
                    display_high = min(float(grid[-1]), 2.0)
                    ax.set_xlim(display_low, display_high)
                    ax.set_title(case["case_label"], fontsize=10)
                    ax.text(
                        0.98,
                        0.96,
                        f"area={metrics['raw_theory_total']:.3f}\n"
                        f"CDF err={metrics['max_raw_subcdf_error']:.3f}",
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=7.5,
                    )
                else:
                    ax.set_xlim(RT_LOW_S, RT_HIGH_S)
                    ax.set_xlabel("RT relative to stimulus (s)")
                    ax.text(
                        0.98,
                        0.96,
                        f"area={metrics['window_theory_total']:.3f}\n"
                        f"CDF err={metrics['max_window_subcdf_error']:.3f}",
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=7.5,
                    )
                if case_index == 0:
                    row_label = (
                        "Raw joint RT density"
                        if prefix == "raw"
                        else "Conditioned 0-1 s density"
                    )
                    ax.set_ylabel(row_label)
                if not bool(metrics["pass_all"]):
                    ax.text(
                        0.98,
                        0.05,
                        "CHECK FAILED",
                        transform=ax.transAxes,
                        ha="right",
                        color="tab:red",
                        fontsize=8,
                        weight="bold",
                    )

        lapse_label = "with random-choice lapse" if include_lapse else "without lapse"
        fig.suptitle(
            f"LED {category.upper()} proactive step-jump + NPL+alpha, {lapse_label}",
            y=0.995,
            fontsize=13,
        )
        fig.legend(
            handles=[
                Line2D([0], [0], color="black", lw=0.9, alpha=0.65, label="Simulation"),
                Line2D([0], [0], color=color, lw=1.5, alpha=0.65, label="Analytic likelihood"),
            ],
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 0.955),
        )
        fig.text(0.008, 0.73, "choice +1", rotation=90, va="center", fontsize=9)
        fig.text(0.008, 0.25, "choice -1 (mirrored)", rotation=90, va="center", fontsize=9)
        fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.92))
        lapse_tag = "with_lapse" if include_lapse else "no_lapse"
        figure_path = OUTPUT_DIR / (
            f"led_{category}_{lapse_tag}_simulation_vs_likelihood_rtds.png"
        )
        fig.savefig(figure_path, dpi=220, bbox_inches="tight")
        figure_paths.append(figure_path)


# %%
# =============================================================================
# Save reusable validation payload and enforce the acceptance checks
# =============================================================================
payload_path = OUTPUT_DIR / "simulation_vs_likelihood_plot_data.pkl"
with payload_path.open("wb") as handle:
    pickle.dump(
        {
            "parameter_cases": parameter_rows,
            "summary": summary_df,
            "plot_payload": plot_payload,
            "simulation_caches": [
                str(scenario["cache_path"].relative_to(REPO_DIR))
                for scenario in scenario_configs
            ],
            "settings": {
                "n_sim": N_SIM,
                "sim_dt_s": SIM_DT_S,
                "hist_bin_s": HIST_BIN_S,
                "theory_dt_s": THEORY_DT_S,
                "K_max": K_MAX,
                "n_quad": N_QUAD,
                "lapse_semantics": "direct exponential-time random choice",
                "valid_rt_window_s": [RT_LOW_S, RT_HIGH_S],
                "led_off_on_likelihoods_separate": True,
            },
            "checks": {
                "max_jax_numpy_pdf_error": max_parity_error,
                "no_jump_on_off_max_error": no_jump_error,
            },
        },
        handle,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

print("\nValidation summary:")
print(
    summary_df[
        [
            "case_label",
            "category",
            "include_lapse",
            "raw_theory_total",
            "window_theory_total",
            "max_raw_subcdf_error",
            "max_window_subcdf_error",
            "pass_all",
        ]
    ].to_string(index=False, float_format=lambda value: f"{value:.6g}")
)
print(f"Maximum JAX/NumPy PDF error: {max_parity_error:.3e}")
print(f"No-jump ON/OFF maximum density error: {no_jump_error:.3e}")
print("\nSaved:")
for path in [parameter_csv, summary_csv, payload_path, *figure_paths]:
    print(f"  {path}")

failed = summary_df[~summary_df["pass_all"]]
if len(failed):
    raise AssertionError(f"{len(failed)} validation cells failed checks.")
print("\nAll separate LED-OFF and LED-ON likelihood checks passed.")
