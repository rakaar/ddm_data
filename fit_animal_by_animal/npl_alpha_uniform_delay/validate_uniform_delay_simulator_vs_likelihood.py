# %%
"""Compare uniform-delay simulations with the analytic NPL+alpha likelihood.

Rows use three posterior-derived proactive/reactive parameter sets. Columns use
three uniform t_E_aff distributions. The plotted densities are joint RT/choice
densities: choice +1 is positive and choice -1 is mirrored below zero. The
script validates both the complete response distribution and the data-like
left-truncated, stimulus-relative 0-1 s fitting window.
"""

# %%
from pathlib import Path
import json
import os
import pickle
import sys
import time

from joblib import Parallel, delayed
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.integrate import cumulative_trapezoid


SCRIPT_DIR = Path(__file__).resolve().parent
ANIMAL_FIT_DIR = SCRIPT_DIR.parent
REPO_DIR = ANIMAL_FIT_DIR.parent
sys.path.insert(0, str(ANIMAL_FIT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from time_vary_and_norm_alpha_simulators import (
    simulate_psiam_tied_rate_norm_alpha,
)
from time_vary_norm_alpha_utils import gamma_omega_alpha_fn


# %%
# =============================================================================
# Editable validation settings
# =============================================================================
REFERENCE_FIT_ROOT = (
    ANIMAL_FIT_DIR
    / "numpyro_svi_npl_alpha_condition_delay_patience12_restore_best_outputs"
)
ABORT_FIT_ROOT = REPO_DIR / "aborts_ipl_npl_time_fit_results"
OUTPUT_DIR = SCRIPT_DIR / "validation_outputs"

T_STIM_S = 0.500
DEFAULT_T_TRUNC_S = 0.300
BATCH_T_TRUNC_S = {"LED34_even": 0.150}
RT_WINDOW_LOW_S = 0.0
RT_WINDOW_HIGH_S = 1.0
MIN_RETAINED_WINDOW_SIM = int(
    float(os.environ.get("MIN_RETAINED_WINDOW_SIM", 100_000))
)
N_SIM = int(float(os.environ.get("N_SIM_OVERRIDE", 100_000)))
SIM_DT_OVERRIDE = os.environ.get("SIM_DT_OVERRIDE")
BASE_SIM_DT_S = float(SIM_DT_OVERRIDE or 0.00025)
SLOW_SIM_DT_S = float(
    os.environ.get(
        "SLOW_SIM_DT_OVERRIDE",
        SIM_DT_OVERRIDE or 0.00010,
    )
)
FAST_SIM_DT_S = float(
    os.environ.get(
        "FAST_SIM_DT_OVERRIDE",
        SIM_DT_OVERRIDE or 0.00005,
    )
)
N_JOBS = int(os.environ.get("N_JOBS_OVERRIDE", 3))
FORCE_RESIMULATE = os.environ.get("FORCE_RESIMULATE", "0") == "1"
BASE_SEED = 20260805

HIST_BIN_S = 0.005
THEORY_DT_S = 0.00025
MEAN_DELAY_REFERENCE_S = 0.080
ANALYTIC_SPECTRAL_TERMS = 100
CURRENT_CDF_K_MAX = 10
QUADRATURE_NODES = 64
QUADRATURE_NODE_CHUNK = 16
TAIL_MASS_TARGET = 1e-8
MAX_DISPLAY_RT_STIM_S = 20.0

GLOBAL_PARAM_NAMES = [
    "rate_lambda",
    "T_0",
    "theta_E",
    "w",
    "del_go",
    "rate_norm_l",
    "alpha",
]

PARAMETER_CASES = [
    {
        "case_key": "near_median",
        "case_label": "Near median",
        "batch": "LED34",
        "animal": 57,
        "ABL": 40.0,
        "ILD": -1.0,
    },
    {
        "case_key": "slow_negative",
        "case_label": "Slow evidence",
        "batch": "LED7",
        "animal": 92,
        "ABL": 20.0,
        "ILD": -16.0,
    },
    {
        "case_key": "fast_positive",
        "case_label": "Fast positive evidence",
        "batch": "LED8",
        "animal": 105,
        "ABL": 60.0,
        "ILD": 16.0,
    },
]

DELAY_DISTRIBUTIONS = [
    {
        "delay_key": "narrow_5ms",
        "delay_label": "Uniform(77.5, 82.5) ms",
        "delay_low_s": 0.0775,
        "delay_high_s": 0.0825,
    },
    {
        "delay_key": "medium_20ms",
        "delay_label": "Uniform(70, 90) ms",
        "delay_low_s": 0.070,
        "delay_high_s": 0.090,
    },
    {
        "delay_key": "wide_100ms",
        "delay_label": "Uniform(30, 130) ms",
        "delay_low_s": 0.030,
        "delay_high_s": 0.130,
    },
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Reference NPL+alpha fits: {REFERENCE_FIT_ROOT}")
print(f"Abort fits: {ABORT_FIT_ROOT}")
print(f"Simulation trials per cell: {N_SIM:,}")
print(
    "Simulation dt: "
    f"{1e3 * BASE_SIM_DT_S:.3f} ms for median; "
    f"{1e3 * SLOW_SIM_DT_S:.3f} ms for slow evidence; "
    f"{1e3 * FAST_SIM_DT_S:.3f} ms for fast evidence"
)
print(f"Simulation workers: {N_JOBS}")
print(f"Theory dt: {1e3 * THEORY_DT_S:.3f} ms")
print(f"Analytic spectral terms: {ANALYTIC_SPECTRAL_TERMS}")


# %%
# =============================================================================
# Load the three complete posterior-derived parameter sets
# =============================================================================
loaded_cases = []
for case_config in PARAMETER_CASES:
    fit_dir = (
        REFERENCE_FIT_ROOT
        / f"{case_config['batch']}_{case_config['animal']}"
    )
    posterior_file = fit_dir / "main_fullrank_posterior_samples.npz"
    abort_file = (
        ABORT_FIT_ROOT
        / f"results_{case_config['batch']}_animal_{case_config['animal']}.pkl"
    )
    for required_file in [posterior_file, abort_file]:
        if not required_file.exists():
            raise FileNotFoundError(required_file)

    with np.load(posterior_file) as posterior:
        global_means = {
            name: float(np.mean(posterior[name]))
            for name in GLOBAL_PARAM_NAMES
        }
    with abort_file.open("rb") as handle:
        abort_samples = pickle.load(handle)["vbmc_aborts_results"]

    case = {
        **case_config,
        **global_means,
        "V_A": float(np.mean(abort_samples["V_A_samples"])),
        "theta_A": float(np.mean(abort_samples["theta_A_samples"])),
        "t_A_aff": float(np.mean(abort_samples["t_A_aff_samp"])),
        "T_trunc": BATCH_T_TRUNC_S.get(
            case_config["batch"],
            DEFAULT_T_TRUNC_S,
        ),
    }
    case["Z_E"] = (case["w"] - 0.5) * 2.0 * case["theta_E"]
    gamma, omega = gamma_omega_alpha_fn(
        case["ABL"],
        case["ILD"],
        case["rate_lambda"],
        case["T_0"],
        case["theta_E"],
        case["rate_norm_l"],
        case["alpha"],
        True,
    )
    case["gamma"] = float(gamma)
    case["omega"] = float(omega)
    loaded_cases.append(case)

parameter_columns = [
    "case_key",
    "case_label",
    "batch",
    "animal",
    "ABL",
    "ILD",
    "V_A",
    "theta_A",
    "t_A_aff",
    "T_trunc",
    "rate_lambda",
    "T_0",
    "theta_E",
    "w",
    "Z_E",
    "del_go",
    "rate_norm_l",
    "alpha",
    "gamma",
    "omega",
]
parameter_df = pd.DataFrame(loaded_cases)[parameter_columns]
parameter_csv = OUTPUT_DIR / "uniform_delay_simulator_parameter_sets.csv"
parameter_df.to_csv(parameter_csv, index=False)
print("\nComplete parameter sets:")
print(
    parameter_df[
        [
            "case_label",
            "batch",
            "animal",
            "ABL",
            "ILD",
            "gamma",
            "omega",
            "w",
            "V_A",
            "theta_A",
            "t_A_aff",
        ]
    ].to_string(index=False, float_format=lambda value: f"{value:.6g}")
)


# %%
# =============================================================================
# Simulation helpers and cache
# =============================================================================
def simulate_uniform_delay_scenario(
    case,
    delay_config,
    n_sim,
    sim_dt_s,
    t_stim_s,
    seed,
):
    """Run the existing scalar simulator with one delay draw per trial."""
    np.random.seed(seed)
    rt_stim = np.empty(n_sim, dtype=np.float64)
    choices = np.empty(n_sim, dtype=np.int8)
    is_proactive = np.empty(n_sim, dtype=np.int8)
    sampled_delays = np.empty(n_sim, dtype=np.float32)

    start_time = time.perf_counter()
    for trial_index in range(n_sim):
        trial_delay = np.random.uniform(
            delay_config["delay_low_s"],
            delay_config["delay_high_s"],
        )
        choice, total_rt, trial_is_proactive = (
            simulate_psiam_tied_rate_norm_alpha(
                case["V_A"],
                case["theta_A"],
                case["ABL"],
                case["ILD"],
                case["rate_lambda"],
                case["T_0"],
                case["theta_E"],
                case["Z_E"],
                t_stim_s,
                case["t_A_aff"],
                trial_delay,
                case["del_go"],
                case["rate_norm_l"],
                case["alpha"],
                sim_dt_s,
            )
        )
        rt_stim[trial_index] = total_rt - t_stim_s
        choices[trial_index] = choice
        is_proactive[trial_index] = trial_is_proactive
        sampled_delays[trial_index] = trial_delay

    return {
        "rt_stim": rt_stim,
        "choice": choices,
        "is_proactive": is_proactive,
        "sampled_delay": sampled_delays,
        "runtime_s": time.perf_counter() - start_time,
        "seed": seed,
    }


scenario_configs = []
for row_index, case in enumerate(loaded_cases):
    if case["case_key"] == "slow_negative":
        case_sim_dt_s = SLOW_SIM_DT_S
    elif case["case_key"] == "fast_positive":
        case_sim_dt_s = FAST_SIM_DT_S
    else:
        case_sim_dt_s = BASE_SIM_DT_S
    for column_index, delay_config in enumerate(DELAY_DISTRIBUTIONS):
        scenario_configs.append(
            {
                "row_index": row_index,
                "column_index": column_index,
                "case": case,
                "delay": delay_config,
                "sim_dt_s": case_sim_dt_s,
                "seed": BASE_SEED + 100 * row_index + column_index,
            }
        )

simulation_results = {}
missing_scenarios = []
simulation_cache_paths = []
for scenario in scenario_configs:
    dt_microseconds = int(round(1e6 * scenario["sim_dt_s"]))
    simulation_cache = OUTPUT_DIR / (
        f"uniform_delay_{scenario['case']['case_key']}_"
        f"{scenario['delay']['delay_key']}_n{N_SIM}_"
        f"dt{dt_microseconds}us_seed{scenario['seed']}.npz"
    )
    scenario["simulation_cache"] = simulation_cache
    simulation_cache_paths.append(simulation_cache)
    cache_metadata = {
        "n_sim": N_SIM,
        "sim_dt_s": scenario["sim_dt_s"],
        "t_stim_s": T_STIM_S,
        "seed": scenario["seed"],
        "case": {
            key: scenario["case"][key]
            for key in scenario["case"]
            if key not in {"case_label", "T_trunc"}
        },
        "delay": scenario["delay"],
    }
    scenario["cache_metadata_json"] = json.dumps(
        cache_metadata,
        sort_keys=True,
    )

    cache_loaded = False
    if simulation_cache.exists() and not FORCE_RESIMULATE:
        with np.load(simulation_cache, allow_pickle=False) as cached:
            if str(cached["config_json"].item()) == scenario["cache_metadata_json"]:
                simulation_results[
                    (scenario["row_index"], scenario["column_index"])
                ] = {
                    "rt_stim": cached["rt_stim"],
                    "choice": cached["choice"],
                    "is_proactive": cached["is_proactive"],
                    "sampled_delay": cached["sampled_delay"],
                    "runtime_s": float(cached["runtime_s"]),
                    "seed": int(cached["seed"]),
                }
                cache_loaded = True
    if not cache_loaded:
        missing_scenarios.append(scenario)

print(
    f"\nLoaded {len(scenario_configs) - len(missing_scenarios)} "
    f"of {len(scenario_configs)} per-cell simulation caches."
)

if missing_scenarios:
    print(f"Simulating {len(missing_scenarios)} missing scenarios...")
    parallel_results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(simulate_uniform_delay_scenario)(
            scenario["case"],
            scenario["delay"],
            N_SIM,
            scenario["sim_dt_s"],
            T_STIM_S,
            scenario["seed"],
        )
        for scenario in missing_scenarios
    )
    for scenario, result in zip(missing_scenarios, parallel_results):
        key = (scenario["row_index"], scenario["column_index"])
        simulation_results[key] = result
        np.savez_compressed(
            scenario["simulation_cache"],
            config_json=np.asarray(scenario["cache_metadata_json"]),
            rt_stim=result["rt_stim"],
            choice=result["choice"],
            is_proactive=result["is_proactive"],
            sampled_delay=result["sampled_delay"],
            runtime_s=np.asarray(result["runtime_s"]),
            seed=np.asarray(result["seed"]),
        )
        print(
            f"  row {key[0] + 1}, column {key[1] + 1}: "
            f"{result['runtime_s']:.1f} s"
        )
        print(f"    {scenario['simulation_cache']}")


# %%
# =============================================================================
# Analytic and independent numerical delay-mixture helpers
# =============================================================================
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import numpyro_npl_alpha_svi_utils as npl_utils
import uniform_delay_likelihood_utils as uniform_utils


legendre_x, legendre_w = leggauss(QUADRATURE_NODES)


def analytic_bound_density(
    total_time_s,
    bound,
    case,
    delay_config,
    proactive_truncation_time=None,
):
    values = uniform_utils.up_or_down_alpha_uniform_delay_jax(
        jnp.asarray(total_time_s),
        bound,
        case["V_A"],
        case["theta_A"],
        case["t_A_aff"],
        T_STIM_S,
        case["ABL"],
        case["ILD"],
        case["rate_lambda"],
        case["T_0"],
        case["theta_E"],
        case["Z_E"],
        delay_config["delay_low_s"],
        delay_config["delay_high_s"],
        case["del_go"],
        case["rate_norm_l"],
        case["alpha"],
        CURRENT_CDF_K_MAX,
        ANALYTIC_SPECTRAL_TERMS,
        proactive_truncation_time,
    )
    return np.asarray(values)


def quadrature_bound_density(
    total_time_s,
    bound,
    case,
    delay_config,
    proactive_truncation_time=None,
):
    delay_low = delay_config["delay_low_s"]
    delay_high = delay_config["delay_high_s"]
    delay_nodes = (
        0.5 * (delay_high - delay_low) * legendre_x
        + 0.5 * (delay_high + delay_low)
    )
    expectation_weights = 0.5 * legendre_w
    total_time_jax = jnp.asarray(total_time_s)[None, :]
    mixture = np.zeros_like(total_time_s, dtype=float)

    for start in range(0, QUADRATURE_NODES, QUADRATURE_NODE_CHUNK):
        stop = min(start + QUADRATURE_NODE_CHUNK, QUADRATURE_NODES)
        node_values = uniform_utils.up_or_down_alpha_fixed_delay_stable_proactive_jax(
            total_time_jax,
            bound,
            case["V_A"],
            case["theta_A"],
            case["t_A_aff"],
            T_STIM_S,
            case["ABL"],
            case["ILD"],
            case["rate_lambda"],
            case["T_0"],
            case["theta_E"],
            case["Z_E"],
            jnp.asarray(delay_nodes[start:stop])[:, None],
            case["del_go"],
            case["rate_norm_l"],
            case["alpha"],
            CURRENT_CDF_K_MAX,
            proactive_truncation_time,
        )
        mixture += np.sum(
            expectation_weights[start:stop, None] * np.asarray(node_values),
            axis=0,
        )
    return mixture


def analytic_race_cdf(
    rt_stim_s,
    case,
    delay_config,
    proactive_truncation_time=None,
):
    value = uniform_utils.cum_pro_and_reactive_alpha_uniform_delay_jax(
        rt_stim_s + T_STIM_S,
        case["V_A"],
        case["theta_A"],
        case["t_A_aff"],
        T_STIM_S,
        case["ABL"],
        case["ILD"],
        case["rate_lambda"],
        case["T_0"],
        case["theta_E"],
        case["Z_E"],
        delay_config["delay_low_s"],
        delay_config["delay_high_s"],
        case["rate_norm_l"],
        case["alpha"],
        ANALYTIC_SPECTRAL_TERMS,
        proactive_truncation_time,
    )
    return float(value)


def quadrature_race_cdf(
    rt_stim_s,
    case,
    delay_config,
    proactive_truncation_time=None,
):
    """Independent fixed-delay quadrature reference for the race CDF."""
    delay_low = delay_config["delay_low_s"]
    delay_high = delay_config["delay_high_s"]
    delay_nodes = (
        0.5 * (delay_high - delay_low) * legendre_x
        + 0.5 * (delay_high + delay_low)
    )
    values = uniform_utils.cum_pro_and_reactive_alpha_fixed_delay_stable_proactive_jax(
        rt_stim_s + T_STIM_S,
        case["V_A"],
        case["theta_A"],
        case["t_A_aff"],
        T_STIM_S,
        case["ABL"],
        case["ILD"],
        case["rate_lambda"],
        case["T_0"],
        case["theta_E"],
        case["Z_E"],
        jnp.asarray(delay_nodes),
        case["rate_norm_l"],
        case["alpha"],
        CURRENT_CDF_K_MAX,
        proactive_truncation_time,
    )
    return float(np.sum(0.5 * legendre_w * np.asarray(values)))


# %%
# =============================================================================
# Use the complete support numerically: exact left support and negligible tail
# =============================================================================
row_grids = {}
row_histogram_edges = {}
for row_index, case in enumerate(loaded_cases):
    row_simulations = [
        simulation_results[(row_index, column_index)]["rt_stim"]
        for column_index in range(len(DELAY_DISTRIBUTIONS))
    ]
    simulation_min = min(float(np.min(values)) for values in row_simulations)
    simulation_max = max(float(np.max(values)) for values in row_simulations)
    proactive_support_start = case["t_A_aff"] - T_STIM_S
    row_min = min(simulation_min, proactive_support_start) - HIST_BIN_S
    row_min = HIST_BIN_S * np.floor(row_min / HIST_BIN_S)

    row_max = max(2.0, simulation_max + HIST_BIN_S)
    while True:
        cdf_values = [
            analytic_race_cdf(row_max, case, delay_config)
            for delay_config in DELAY_DISTRIBUTIONS
        ]
        if min(cdf_values) >= 1.0 - TAIL_MASS_TARGET:
            break
        row_max = 1.5 * row_max + 0.5
        if row_max > MAX_DISPLAY_RT_STIM_S:
            raise RuntimeError(
                f"Could not cover the analytic tail for {case['case_label']} "
                f"before {MAX_DISPLAY_RT_STIM_S} s."
            )
    row_max = max(row_max, simulation_max + HIST_BIN_S)
    row_max = HIST_BIN_S * np.ceil(row_max / HIST_BIN_S)

    row_grids[row_index] = np.arange(
        row_min,
        row_max + 0.5 * THEORY_DT_S,
        THEORY_DT_S,
    )
    row_histogram_edges[row_index] = np.arange(
        row_min,
        row_max + 1.01 * HIST_BIN_S,
        HIST_BIN_S,
    )
    tail_values = [1.0 - value for value in cdf_values]
    print(
        f"\n{case['case_label']} support: {row_min:.3f} to {row_max:.3f} s; "
        f"largest analytic tail={max(tail_values):.3e}; "
        f"simulation range={simulation_min:.3f} to {simulation_max:.3f} s"
    )


# %%
# =============================================================================
# Compute raw areas, sub-CDF agreement, and plotting arrays
# =============================================================================
summary_rows = []
cell_plot_data = {}

for scenario in scenario_configs:
    row_index = scenario["row_index"]
    column_index = scenario["column_index"]
    case = scenario["case"]
    delay_config = scenario["delay"]
    simulation = simulation_results[(row_index, column_index)]
    rt_stim = simulation["rt_stim"]
    choices = simulation["choice"]
    theory_rt_stim = row_grids[row_index]
    total_time_s = theory_rt_stim + T_STIM_S
    histogram_edges = row_histogram_edges[row_index]
    histogram_centers = 0.5 * (
        histogram_edges[:-1] + histogram_edges[1:]
    )

    cell_data = {
        "histogram_centers": histogram_centers,
        "theory_rt_stim": theory_rt_stim,
    }
    metrics = {
        "row_index": row_index,
        "column_index": column_index,
        "case_key": case["case_key"],
        "case_label": case["case_label"],
        "batch": case["batch"],
        "animal": case["animal"],
        "ABL": case["ABL"],
        "ILD": case["ILD"],
        "gamma": case["gamma"],
        "omega": case["omega"],
        "w": case["w"],
        "delay_key": delay_config["delay_key"],
        "delay_low_ms": 1e3 * delay_config["delay_low_s"],
        "delay_high_ms": 1e3 * delay_config["delay_high_s"],
        "delay_width_ms": 1e3
        * (delay_config["delay_high_s"] - delay_config["delay_low_s"]),
        "n_sim": N_SIM,
        "sim_dt_ms": 1e3 * scenario["sim_dt_s"],
        "seed": simulation["seed"],
        "simulation_runtime_s": simulation["runtime_s"],
        "simulation_proactive_fraction": float(
            np.mean(simulation["is_proactive"])
        ),
        "theory_rt_min_s": float(theory_rt_stim[0]),
        "theory_rt_max_s": float(theory_rt_stim[-1]),
        "analytic_tail_mass_at_max": 1.0
        - analytic_race_cdf(
            float(theory_rt_stim[-1]),
            case,
            delay_config,
        ),
    }

    for bound, suffix in [(1, "plus"), (-1, "minus")]:
        bound_mask = choices == bound
        histogram_counts, _ = np.histogram(
            rt_stim[bound_mask],
            bins=histogram_edges,
        )
        histogram_density = histogram_counts / (N_SIM * HIST_BIN_S)
        analytic_density = analytic_bound_density(
            total_time_s,
            bound,
            case,
            delay_config,
        )
        quadrature_density = quadrature_bound_density(
            total_time_s,
            bound,
            case,
            delay_config,
        )
        analytic_subcdf = cumulative_trapezoid(
            analytic_density,
            theory_rt_stim,
            initial=0.0,
        )
        quadrature_subcdf = cumulative_trapezoid(
            quadrature_density,
            theory_rt_stim,
            initial=0.0,
        )
        sorted_bound_rt = np.sort(rt_stim[bound_mask])
        simulation_subcdf = np.searchsorted(
            sorted_bound_rt,
            theory_rt_stim,
            side="right",
        ) / N_SIM

        cell_data[f"histogram_{suffix}"] = histogram_density
        cell_data[f"analytic_{suffix}"] = analytic_density
        cell_data[f"quadrature_{suffix}"] = quadrature_density

        metrics[f"simulation_area_{suffix}"] = float(np.mean(bound_mask))
        metrics[f"analytic_area_{suffix}"] = float(analytic_subcdf[-1])
        metrics[f"quadrature_area_{suffix}"] = float(
            quadrature_subcdf[-1]
        )
        metrics[f"simulation_analytic_area_abs_error_{suffix}"] = abs(
            metrics[f"simulation_area_{suffix}"]
            - metrics[f"analytic_area_{suffix}"]
        )
        metrics[f"analytic_quadrature_area_abs_error_{suffix}"] = abs(
            metrics[f"analytic_area_{suffix}"]
            - metrics[f"quadrature_area_{suffix}"]
        )
        metrics[f"simulation_analytic_subcdf_max_abs_error_{suffix}"] = (
            float(np.max(np.abs(simulation_subcdf - analytic_subcdf)))
        )
        metrics[f"analytic_quadrature_subcdf_max_abs_error_{suffix}"] = (
            float(np.max(np.abs(analytic_subcdf - quadrature_subcdf)))
        )
        metrics[f"analytic_density_min_{suffix}"] = float(
            np.min(analytic_density)
        )
        probability_for_se = np.clip(
            metrics[f"analytic_area_{suffix}"],
            0.0,
            1.0,
        )
        metrics[f"four_mc_se_{suffix}"] = 4.0 * np.sqrt(
            probability_for_se * (1.0 - probability_for_se) / N_SIM
        )

    metrics["simulation_total_area"] = (
        metrics["simulation_area_plus"]
        + metrics["simulation_area_minus"]
    )
    metrics["analytic_total_area"] = (
        metrics["analytic_area_plus"] + metrics["analytic_area_minus"]
    )
    metrics["quadrature_total_area"] = (
        metrics["quadrature_area_plus"]
        + metrics["quadrature_area_minus"]
    )
    metrics["simulation_total_area_abs_error"] = abs(
        metrics["simulation_total_area"] - 1.0
    )
    metrics["analytic_total_area_abs_error"] = abs(
        metrics["analytic_total_area"] - 1.0
    )
    metrics["quadrature_total_area_abs_error"] = abs(
        metrics["quadrature_total_area"] - 1.0
    )
    metrics["max_simulation_analytic_area_abs_error"] = max(
        metrics["simulation_analytic_area_abs_error_plus"],
        metrics["simulation_analytic_area_abs_error_minus"],
    )
    metrics["max_analytic_quadrature_area_abs_error"] = max(
        metrics["analytic_quadrature_area_abs_error_plus"],
        metrics["analytic_quadrature_area_abs_error_minus"],
    )
    metrics["max_simulation_analytic_subcdf_abs_error"] = max(
        metrics["simulation_analytic_subcdf_max_abs_error_plus"],
        metrics["simulation_analytic_subcdf_max_abs_error_minus"],
    )
    metrics["max_analytic_quadrature_subcdf_abs_error"] = max(
        metrics["analytic_quadrature_subcdf_max_abs_error_plus"],
        metrics["analytic_quadrature_subcdf_max_abs_error_minus"],
    )
    metrics["min_analytic_density"] = min(
        metrics["analytic_density_min_plus"],
        metrics["analytic_density_min_minus"],
    )
    metrics["pass_simulation_total_area"] = (
        metrics["simulation_total_area_abs_error"] <= 1e-12
    )
    metrics["pass_analytic_total_area"] = (
        metrics["analytic_total_area_abs_error"] <= 0.002
    )
    metrics["pass_analytic_quadrature_area"] = (
        metrics["max_analytic_quadrature_area_abs_error"] <= 0.002
    )
    metrics["pass_analytic_quadrature_subcdf"] = (
        metrics["max_analytic_quadrature_subcdf_abs_error"] <= 0.002
    )
    metrics["pass_simulation_choice_areas"] = (
        metrics["simulation_analytic_area_abs_error_plus"]
        <= metrics["four_mc_se_plus"]
        and metrics["simulation_analytic_area_abs_error_minus"]
        <= metrics["four_mc_se_minus"]
    )
    metrics["pass_simulation_analytic_subcdf"] = (
        metrics["max_simulation_analytic_subcdf_abs_error"] <= 0.01
    )
    metrics["pass_nonnegative_analytic_density"] = (
        metrics["min_analytic_density"] >= -1e-6
    )
    pass_columns = [
        "pass_simulation_total_area",
        "pass_analytic_total_area",
        "pass_analytic_quadrature_area",
        "pass_analytic_quadrature_subcdf",
        "pass_simulation_choice_areas",
        "pass_simulation_analytic_subcdf",
        "pass_nonnegative_analytic_density",
    ]
    metrics["pass_all_checks"] = all(metrics[name] for name in pass_columns)

    summary_rows.append(metrics)
    cell_plot_data[(row_index, column_index)] = cell_data
    print(
        f"Computed row {row_index + 1}, column {column_index + 1}: "
        f"simulation areas="
        f"{metrics['simulation_area_plus']:.4f}/"
        f"{metrics['simulation_area_minus']:.4f}, "
        f"analytic areas="
        f"{metrics['analytic_area_plus']:.4f}/"
        f"{metrics['analytic_area_minus']:.4f}, "
        f"pass={metrics['pass_all_checks']}"
    )

summary_df = pd.DataFrame(summary_rows).sort_values(
    ["row_index", "column_index"]
)
summary_csv = OUTPUT_DIR / "uniform_delay_simulator_vs_likelihood_summary.csv"
summary_df.to_csv(summary_csv, index=False)

print("\nValidation summary:")
print(
    summary_df[
        [
            "case_label",
            "delay_width_ms",
            "simulation_area_plus",
            "analytic_area_plus",
            "analytic_total_area",
            "max_simulation_analytic_subcdf_abs_error",
            "max_analytic_quadrature_subcdf_abs_error",
            "min_analytic_density",
            "pass_all_checks",
        ]
    ].to_string(index=False, float_format=lambda value: f"{value:.6g}")
)


# %%
# =============================================================================
# Mirrored RTD figure: raw +1 area above zero, raw -1 area below zero
# =============================================================================
fig, axes = plt.subplots(
    3,
    3,
    figsize=(14.5, 10.0),
    sharex="row",
    sharey="row",
)

for row_index, case in enumerate(loaded_cases):
    row_y_max = 0.0
    for column_index in range(len(DELAY_DISTRIBUTIONS)):
        plot_data = cell_plot_data[(row_index, column_index)]
        row_y_max = max(
            row_y_max,
            float(np.max(plot_data["histogram_plus"])),
            float(np.max(plot_data["histogram_minus"])),
            float(np.max(np.abs(plot_data["analytic_plus"]))),
            float(np.max(np.abs(plot_data["analytic_minus"]))),
        )

    for column_index, delay_config in enumerate(DELAY_DISTRIBUTIONS):
        ax = axes[row_index, column_index]
        plot_data = cell_plot_data[(row_index, column_index)]
        metrics = summary_df[
            (summary_df["row_index"] == row_index)
            & (summary_df["column_index"] == column_index)
        ].iloc[0]

        ax.step(
            plot_data["histogram_centers"],
            plot_data["histogram_plus"],
            where="mid",
            color="black",
            lw=0.9,
            alpha=0.68,
            zorder=3,
        )
        ax.step(
            plot_data["histogram_centers"],
            -plot_data["histogram_minus"],
            where="mid",
            color="black",
            lw=0.9,
            alpha=0.68,
            zorder=3,
        )
        ax.plot(
            plot_data["theory_rt_stim"],
            plot_data["analytic_plus"],
            color="tab:blue",
            lw=1.4,
            alpha=0.5,
            zorder=4,
        )
        ax.plot(
            plot_data["theory_rt_stim"],
            -plot_data["analytic_minus"],
            color="tab:blue",
            lw=1.4,
            alpha=0.5,
            zorder=4,
        )
        ax.axhline(0.0, color="0.25", lw=0.7, zorder=1)
        ax.axvline(0.0, color="0.65", lw=0.8, ls=":", zorder=1)
        ax.set_ylim(-1.08 * row_y_max, 1.08 * row_y_max)
        ax.set_xlim(-1.0, 1.0)
        ax.text(
            0.985,
            0.965,
            "Simulation + / -: "
            f"{metrics['simulation_area_plus']:.3f} / "
            f"{metrics['simulation_area_minus']:.3f}\n"
            "Analytic + / -: "
            f"{metrics['analytic_area_plus']:.3f} / "
            f"{metrics['analytic_area_minus']:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
            color="0.15",
        )
        if not bool(metrics["pass_all_checks"]):
            ax.text(
                0.985,
                0.04,
                "CHECK FAILED",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7.5,
                color="tab:red",
                weight="bold",
            )
        if row_index == 0:
            ax.set_title(delay_config["delay_label"], fontsize=10)
        if row_index == 2:
            ax.set_xlabel("RT relative to stimulus (s)")
        if column_index == 0:
            ax.set_ylabel(
                f"{case['case_label']}\n"
                f"{case['batch']}/{case['animal']}, "
                f"ABL {case['ABL']:.0f}, ILD {case['ILD']:+.0f}\n"
                "Joint RT density",
            )

legend_handles = [
    Line2D([0], [0], color="black", lw=0.9, alpha=0.68, label="Simulation"),
    Line2D(
        [0],
        [0],
        color="tab:blue",
        lw=1.4,
        alpha=0.5,
        label="Analytic K=100",
    ),
]
fig.legend(
    handles=legend_handles,
    loc="upper center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, 0.975),
)
fig.text(
    0.012,
    0.72,
    "choice +1",
    rotation=90,
    va="center",
    ha="center",
    fontsize=9,
)
fig.text(
    0.012,
    0.29,
    "choice -1 (mirrored)",
    rotation=90,
    va="center",
    ha="center",
    fontsize=9,
)
fig.suptitle(
    "Uniform-delay NPL+alpha: simulation versus analytic joint RT/choice density",
    y=0.998,
    fontsize=13,
)
fig.tight_layout(rect=(0.025, 0.02, 1.0, 0.94))

figure_png = (
    OUTPUT_DIR / "uniform_delay_simulator_vs_analytic_likelihood_3x3.png"
)
fig.savefig(figure_png, dpi=220, bbox_inches="tight")

print("\nSaved:")
for output_path in [
    parameter_csv,
    summary_csv,
    figure_png,
]:
    print(f"  {output_path}")

failed_rows = summary_df.loc[~summary_df["pass_all_checks"]]
if len(failed_rows):
    print(f"\nWARNING: {len(failed_rows)} of 9 cells failed at least one check.")
    for row in failed_rows.itertuples():
        failed_checks = [
            column
            for column in summary_df.columns
            if column.startswith("pass_")
            and column != "pass_all_checks"
            and not bool(getattr(row, column))
        ]
        print(
            f"  {row.case_label}, width={row.delay_width_ms:g} ms: "
            + ", ".join(failed_checks)
        )
else:
    print("\nAll nine simulation/analytic likelihood checks passed.")


# %%
# =============================================================================
# Data-like left truncation followed by the stimulus-relative 0-1 s window
# =============================================================================
window_theory_rt_stim = np.arange(
    RT_WINDOW_LOW_S,
    RT_WINDOW_HIGH_S + 0.5 * THEORY_DT_S,
    THEORY_DT_S,
)
window_total_time_s = window_theory_rt_stim + T_STIM_S
window_histogram_edges = np.arange(
    RT_WINDOW_LOW_S,
    RT_WINDOW_HIGH_S + 1.01 * HIST_BIN_S,
    HIST_BIN_S,
)
window_histogram_centers = 0.5 * (
    window_histogram_edges[:-1] + window_histogram_edges[1:]
)

# Keep the original full-support simulations untouched. Add deterministic,
# separately cached trials when conditioning leaves fewer than 100k responses.
window_simulation_results = {}
missing_window_supplements = []
for scenario in scenario_configs:
    key = (scenario["row_index"], scenario["column_index"])
    initial_simulation = simulation_results[key]
    initial_rt_stim = initial_simulation["rt_stim"]
    initial_retained = (
        (initial_rt_stim + T_STIM_S >= scenario["case"]["T_trunc"])
        & (initial_rt_stim >= RT_WINDOW_LOW_S)
        & (initial_rt_stim < RT_WINDOW_HIGH_S)
    )
    initial_n_retained = int(np.sum(initial_retained))
    if initial_n_retained >= MIN_RETAINED_WINDOW_SIM:
        window_simulation_results[key] = initial_simulation
        continue

    retained_fraction = max(initial_n_retained / len(initial_rt_stim), 0.05)
    n_supplement = int(
        np.ceil(
            (MIN_RETAINED_WINDOW_SIM - initial_n_retained)
            / retained_fraction
        )
        + 10_000
    )
    supplement_seed = (
        BASE_SEED
        + 100_000
        + 100 * scenario["row_index"]
        + scenario["column_index"]
    )
    dt_microseconds = int(round(1e6 * scenario["sim_dt_s"]))
    supplement_cache = OUTPUT_DIR / (
        f"uniform_delay_window_supplement_"
        f"{scenario['case']['case_key']}_"
        f"{scenario['delay']['delay_key']}_n{n_supplement}_"
        f"dt{dt_microseconds}us_seed{supplement_seed}.npz"
    )
    supplement_metadata = {
        "n_sim": n_supplement,
        "sim_dt_s": scenario["sim_dt_s"],
        "t_stim_s": T_STIM_S,
        "seed": supplement_seed,
        "case": {
            name: scenario["case"][name]
            for name in scenario["case"]
            if name not in {"case_label", "T_trunc"}
        },
        "delay": scenario["delay"],
    }
    supplement_metadata_json = json.dumps(supplement_metadata, sort_keys=True)

    supplement_result = None
    if supplement_cache.exists() and not FORCE_RESIMULATE:
        with np.load(supplement_cache, allow_pickle=False) as cached:
            if str(cached["config_json"].item()) == supplement_metadata_json:
                supplement_result = {
                    "rt_stim": cached["rt_stim"],
                    "choice": cached["choice"],
                    "is_proactive": cached["is_proactive"],
                    "sampled_delay": cached["sampled_delay"],
                    "runtime_s": float(cached["runtime_s"]),
                    "seed": int(cached["seed"]),
                }

    supplement_job = {
        "scenario": scenario,
        "n_supplement": n_supplement,
        "seed": supplement_seed,
        "cache": supplement_cache,
        "metadata_json": supplement_metadata_json,
        "result": supplement_result,
    }
    if supplement_result is None:
        missing_window_supplements.append(supplement_job)
    scenario["window_supplement_job"] = supplement_job

print(
    f"\nRetained-window target: at least {MIN_RETAINED_WINDOW_SIM:,} "
    "simulations per cell."
)
if missing_window_supplements:
    print(
        f"Simulating {len(missing_window_supplements)} missing deterministic "
        "window supplements..."
    )
    supplemental_results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(simulate_uniform_delay_scenario)(
            job["scenario"]["case"],
            job["scenario"]["delay"],
            job["n_supplement"],
            job["scenario"]["sim_dt_s"],
            T_STIM_S,
            job["seed"],
        )
        for job in missing_window_supplements
    )
    for job, result in zip(missing_window_supplements, supplemental_results):
        job["result"] = result
        np.savez_compressed(
            job["cache"],
            config_json=np.asarray(job["metadata_json"]),
            rt_stim=result["rt_stim"],
            choice=result["choice"],
            is_proactive=result["is_proactive"],
            sampled_delay=result["sampled_delay"],
            runtime_s=np.asarray(result["runtime_s"]),
            seed=np.asarray(result["seed"]),
        )
        print(f"  {job['cache']}")

for scenario in scenario_configs:
    key = (scenario["row_index"], scenario["column_index"])
    initial_simulation = simulation_results[key]
    supplement_job = scenario.get("window_supplement_job")
    if supplement_job is None:
        window_simulation_results[key] = initial_simulation
        continue
    supplement = supplement_job["result"]
    window_simulation_results[key] = {
        "rt_stim": np.concatenate(
            [initial_simulation["rt_stim"], supplement["rt_stim"]]
        ),
        "choice": np.concatenate(
            [initial_simulation["choice"], supplement["choice"]]
        ),
        "is_proactive": np.concatenate(
            [initial_simulation["is_proactive"], supplement["is_proactive"]]
        ),
        "sampled_delay": np.concatenate(
            [initial_simulation["sampled_delay"], supplement["sampled_delay"]]
        ),
        "runtime_s": (
            initial_simulation["runtime_s"] + supplement["runtime_s"]
        ),
        "seed": initial_simulation["seed"],
    }

window_summary_rows = []
window_cell_plot_data = {}

for scenario in scenario_configs:
    row_index = scenario["row_index"]
    column_index = scenario["column_index"]
    case = scenario["case"]
    delay_config = scenario["delay"]
    simulation = window_simulation_results[(row_index, column_index)]
    rt_stim = simulation["rt_stim"]
    choices = simulation["choice"]
    total_rt = rt_stim + T_STIM_S
    truncation_time = case["T_trunc"]

    after_left_truncation = total_rt >= truncation_time
    in_stimulus_window = (
        (rt_stim >= RT_WINDOW_LOW_S)
        & (rt_stim < RT_WINDOW_HIGH_S)
    )
    retained = after_left_truncation & in_stimulus_window

    if not np.all(total_rt[retained] >= truncation_time):
        raise AssertionError("Retained simulation contains a left-truncated abort.")
    if not np.all(
        (rt_stim[retained] >= RT_WINDOW_LOW_S)
        & (rt_stim[retained] < RT_WINDOW_HIGH_S)
    ):
        raise AssertionError("Retained simulation lies outside [0, 1) s.")
    if T_STIM_S >= truncation_time and not np.array_equal(
        retained,
        in_stimulus_window,
    ):
        raise AssertionError(
            "With stimulus onset after the abort cutoff, the [0, 1) s window "
            "must already be a subset of the left-truncated responses."
        )
    before_cutoff = ~after_left_truncation
    pre_cutoff_all_proactive = bool(
        np.all(simulation["is_proactive"][before_cutoff] == 1)
    )
    if not pre_cutoff_all_proactive:
        raise AssertionError("A pre-stimulus-cutoff response was not proactive.")

    n_after_left_truncation = int(np.sum(after_left_truncation))
    n_retained = int(np.sum(retained))
    if n_after_left_truncation == 0 or n_retained == 0:
        raise RuntimeError("No simulations survived the requested conditioning.")
    if n_retained < MIN_RETAINED_WINDOW_SIM:
        raise RuntimeError(
            f"Only {n_retained:,} simulations survived the retained window; "
            f"expected at least {MIN_RETAINED_WINDOW_SIM:,}."
        )

    analytic_window_mass = (
        analytic_race_cdf(
            RT_WINDOW_HIGH_S,
            case,
            delay_config,
            truncation_time,
        )
        - analytic_race_cdf(
            RT_WINDOW_LOW_S,
            case,
            delay_config,
            truncation_time,
        )
    )
    quadrature_window_mass = (
        quadrature_race_cdf(
            RT_WINDOW_HIGH_S,
            case,
            delay_config,
            truncation_time,
        )
        - quadrature_race_cdf(
            RT_WINDOW_LOW_S,
            case,
            delay_config,
            truncation_time,
        )
    )
    if analytic_window_mass <= 0.0 or quadrature_window_mass <= 0.0:
        raise RuntimeError("The retained-window normalization must be positive.")

    simulation_window_mass = n_retained / n_after_left_truncation
    window_probability_for_se = np.clip(analytic_window_mass, 0.0, 1.0)
    four_mc_se_window_mass = 4.0 * np.sqrt(
        window_probability_for_se
        * (1.0 - window_probability_for_se)
        / n_after_left_truncation
    )

    cell_data = {
        "histogram_centers": window_histogram_centers,
        "theory_rt_stim": window_theory_rt_stim,
    }
    metrics = {
        "row_index": row_index,
        "column_index": column_index,
        "case_key": case["case_key"],
        "case_label": case["case_label"],
        "batch": case["batch"],
        "animal": case["animal"],
        "ABL": case["ABL"],
        "ILD": case["ILD"],
        "delay_key": delay_config["delay_key"],
        "delay_low_ms": 1e3 * delay_config["delay_low_s"],
        "delay_high_ms": 1e3 * delay_config["delay_high_s"],
        "delay_width_ms": 1e3
        * (delay_config["delay_high_s"] - delay_config["delay_low_s"]),
        "T_stim_ms": 1e3 * T_STIM_S,
        "T_trunc_ms": 1e3 * truncation_time,
        "rt_window_low_s": RT_WINDOW_LOW_S,
        "rt_window_high_s": RT_WINDOW_HIGH_S,
        "n_sim": len(rt_stim),
        "n_before_left_cutoff": int(np.sum(before_cutoff)),
        "n_after_left_truncation": n_after_left_truncation,
        "n_retained": n_retained,
        "simulation_pre_cutoff_fraction": float(np.mean(before_cutoff)),
        "simulation_window_mass_after_left_truncation": simulation_window_mass,
        "analytic_window_mass_after_left_truncation": analytic_window_mass,
        "quadrature_window_mass_after_left_truncation": quadrature_window_mass,
        "simulation_analytic_window_mass_abs_error": abs(
            simulation_window_mass - analytic_window_mass
        ),
        "analytic_quadrature_window_mass_abs_error": abs(
            analytic_window_mass - quadrature_window_mass
        ),
        "four_mc_se_window_mass": four_mc_se_window_mass,
        "pre_cutoff_all_proactive": pre_cutoff_all_proactive,
        "left_truncation_mask_redundant_after_windowing": bool(
            np.array_equal(retained, in_stimulus_window)
        ),
    }

    for bound, suffix in [(1, "plus"), (-1, "minus")]:
        retained_bound = retained & (choices == bound)
        histogram_counts, _ = np.histogram(
            rt_stim[retained_bound],
            bins=window_histogram_edges,
        )
        histogram_density = histogram_counts / (n_retained * HIST_BIN_S)

        analytic_density_raw = analytic_bound_density(
            window_total_time_s,
            bound,
            case,
            delay_config,
            truncation_time,
        )
        quadrature_density_raw = quadrature_bound_density(
            window_total_time_s,
            bound,
            case,
            delay_config,
            truncation_time,
        )
        analytic_density = analytic_density_raw / analytic_window_mass
        quadrature_density = quadrature_density_raw / quadrature_window_mass
        analytic_subcdf = cumulative_trapezoid(
            analytic_density,
            window_theory_rt_stim,
            initial=0.0,
        )
        quadrature_subcdf = cumulative_trapezoid(
            quadrature_density,
            window_theory_rt_stim,
            initial=0.0,
        )
        sorted_bound_rt = np.sort(rt_stim[retained_bound])
        simulation_subcdf = np.searchsorted(
            sorted_bound_rt,
            window_theory_rt_stim,
            side="right",
        ) / n_retained

        cell_data[f"histogram_{suffix}"] = histogram_density
        cell_data[f"analytic_{suffix}"] = analytic_density
        cell_data[f"quadrature_{suffix}"] = quadrature_density

        metrics[f"simulation_area_{suffix}"] = float(
            np.sum(retained_bound) / n_retained
        )
        metrics[f"analytic_area_{suffix}"] = float(analytic_subcdf[-1])
        metrics[f"quadrature_area_{suffix}"] = float(
            quadrature_subcdf[-1]
        )
        metrics[f"simulation_analytic_area_abs_error_{suffix}"] = abs(
            metrics[f"simulation_area_{suffix}"]
            - metrics[f"analytic_area_{suffix}"]
        )
        metrics[f"analytic_quadrature_area_abs_error_{suffix}"] = abs(
            metrics[f"analytic_area_{suffix}"]
            - metrics[f"quadrature_area_{suffix}"]
        )
        metrics[f"simulation_analytic_subcdf_max_abs_error_{suffix}"] = (
            float(np.max(np.abs(simulation_subcdf - analytic_subcdf)))
        )
        metrics[f"analytic_quadrature_subcdf_max_abs_error_{suffix}"] = (
            float(np.max(np.abs(analytic_subcdf - quadrature_subcdf)))
        )
        metrics[f"analytic_density_min_{suffix}"] = float(
            np.min(analytic_density)
        )
        choice_probability_for_se = np.clip(
            metrics[f"analytic_area_{suffix}"],
            0.0,
            1.0,
        )
        metrics[f"four_mc_se_{suffix}"] = 4.0 * np.sqrt(
            choice_probability_for_se
            * (1.0 - choice_probability_for_se)
            / n_retained
        )

    metrics["simulation_total_area"] = (
        metrics["simulation_area_plus"] + metrics["simulation_area_minus"]
    )
    metrics["analytic_total_area"] = (
        metrics["analytic_area_plus"] + metrics["analytic_area_minus"]
    )
    metrics["quadrature_total_area"] = (
        metrics["quadrature_area_plus"] + metrics["quadrature_area_minus"]
    )
    metrics["simulation_total_area_abs_error"] = abs(
        metrics["simulation_total_area"] - 1.0
    )
    metrics["analytic_total_area_abs_error"] = abs(
        metrics["analytic_total_area"] - 1.0
    )
    metrics["quadrature_total_area_abs_error"] = abs(
        metrics["quadrature_total_area"] - 1.0
    )
    metrics["max_simulation_analytic_area_abs_error"] = max(
        metrics["simulation_analytic_area_abs_error_plus"],
        metrics["simulation_analytic_area_abs_error_minus"],
    )
    metrics["max_analytic_quadrature_area_abs_error"] = max(
        metrics["analytic_quadrature_area_abs_error_plus"],
        metrics["analytic_quadrature_area_abs_error_minus"],
    )
    metrics["max_simulation_analytic_subcdf_abs_error"] = max(
        metrics["simulation_analytic_subcdf_max_abs_error_plus"],
        metrics["simulation_analytic_subcdf_max_abs_error_minus"],
    )
    metrics["max_analytic_quadrature_subcdf_abs_error"] = max(
        metrics["analytic_quadrature_subcdf_max_abs_error_plus"],
        metrics["analytic_quadrature_subcdf_max_abs_error_minus"],
    )
    metrics["min_analytic_density"] = min(
        metrics["analytic_density_min_plus"],
        metrics["analytic_density_min_minus"],
    )
    metrics["pass_left_truncation_semantics"] = (
        metrics["pre_cutoff_all_proactive"]
        and metrics["left_truncation_mask_redundant_after_windowing"]
    )
    metrics["pass_simulation_window_mass"] = (
        metrics["simulation_analytic_window_mass_abs_error"]
        <= metrics["four_mc_se_window_mass"]
    )
    metrics["pass_analytic_quadrature_window_mass"] = (
        metrics["analytic_quadrature_window_mass_abs_error"] <= 0.002
    )
    metrics["pass_simulation_total_area"] = (
        metrics["simulation_total_area_abs_error"] <= 1e-12
    )
    metrics["pass_analytic_total_area"] = (
        metrics["analytic_total_area_abs_error"] <= 0.002
    )
    metrics["pass_quadrature_total_area"] = (
        metrics["quadrature_total_area_abs_error"] <= 0.002
    )
    metrics["pass_analytic_quadrature_area"] = (
        metrics["max_analytic_quadrature_area_abs_error"] <= 0.002
    )
    metrics["pass_analytic_quadrature_subcdf"] = (
        metrics["max_analytic_quadrature_subcdf_abs_error"] <= 0.002
    )
    metrics["pass_simulation_choice_areas"] = (
        metrics["simulation_analytic_area_abs_error_plus"]
        <= metrics["four_mc_se_plus"]
        and metrics["simulation_analytic_area_abs_error_minus"]
        <= metrics["four_mc_se_minus"]
    )
    metrics["pass_simulation_analytic_subcdf"] = (
        metrics["max_simulation_analytic_subcdf_abs_error"] <= 0.01
    )
    metrics["pass_nonnegative_analytic_density"] = (
        metrics["min_analytic_density"] >= -1e-6
    )
    window_pass_columns = [
        "pass_left_truncation_semantics",
        "pass_simulation_window_mass",
        "pass_analytic_quadrature_window_mass",
        "pass_simulation_total_area",
        "pass_analytic_total_area",
        "pass_quadrature_total_area",
        "pass_analytic_quadrature_area",
        "pass_analytic_quadrature_subcdf",
        "pass_simulation_choice_areas",
        "pass_simulation_analytic_subcdf",
        "pass_nonnegative_analytic_density",
    ]
    metrics["pass_all_checks"] = all(
        metrics[name] for name in window_pass_columns
    )

    window_summary_rows.append(metrics)
    window_cell_plot_data[(row_index, column_index)] = cell_data
    print(
        f"Retained-window row {row_index + 1}, column {column_index + 1}: "
        f"N={n_retained:,}, simulation areas="
        f"{metrics['simulation_area_plus']:.4f}/"
        f"{metrics['simulation_area_minus']:.4f}, analytic areas="
        f"{metrics['analytic_area_plus']:.4f}/"
        f"{metrics['analytic_area_minus']:.4f}, "
        f"pass={metrics['pass_all_checks']}"
    )

window_summary_df = pd.DataFrame(window_summary_rows).sort_values(
    ["row_index", "column_index"]
)
window_summary_csv = (
    OUTPUT_DIR
    / "uniform_delay_left_truncated_0_1s_simulator_vs_likelihood_summary.csv"
)
window_summary_df.to_csv(window_summary_csv, index=False)


# %%
# =============================================================================
# Mirrored retained-window RTD figure
# =============================================================================
fig_window, axes_window = plt.subplots(
    3,
    3,
    figsize=(14.5, 10.0),
    sharex="row",
    sharey="row",
)

for row_index, case in enumerate(loaded_cases):
    row_y_max = 0.0
    for column_index in range(len(DELAY_DISTRIBUTIONS)):
        plot_data = window_cell_plot_data[(row_index, column_index)]
        row_y_max = max(
            row_y_max,
            float(np.max(plot_data["histogram_plus"])),
            float(np.max(plot_data["histogram_minus"])),
            float(np.max(np.abs(plot_data["analytic_plus"]))),
            float(np.max(np.abs(plot_data["analytic_minus"]))),
        )

    for column_index, delay_config in enumerate(DELAY_DISTRIBUTIONS):
        ax = axes_window[row_index, column_index]
        plot_data = window_cell_plot_data[(row_index, column_index)]
        metrics = window_summary_df[
            (window_summary_df["row_index"] == row_index)
            & (window_summary_df["column_index"] == column_index)
        ].iloc[0]

        ax.step(
            plot_data["histogram_centers"],
            plot_data["histogram_plus"],
            where="mid",
            color="black",
            lw=0.9,
            alpha=0.68,
            zorder=3,
        )
        ax.step(
            plot_data["histogram_centers"],
            -plot_data["histogram_minus"],
            where="mid",
            color="black",
            lw=0.9,
            alpha=0.68,
            zorder=3,
        )
        ax.plot(
            plot_data["theory_rt_stim"],
            plot_data["analytic_plus"],
            color="tab:blue",
            lw=1.4,
            alpha=0.5,
            zorder=4,
        )
        ax.plot(
            plot_data["theory_rt_stim"],
            -plot_data["analytic_minus"],
            color="tab:blue",
            lw=1.4,
            alpha=0.5,
            zorder=4,
        )
        ax.axhline(0.0, color="0.25", lw=0.7, zorder=1)
        ax.axvline(
            MEAN_DELAY_REFERENCE_S,
            color="tab:red",
            lw=0.9,
            ls="--",
            alpha=0.65,
            zorder=2,
        )
        ax.set_ylim(-1.08 * row_y_max, 1.08 * row_y_max)
        ax.set_xlim(RT_WINDOW_LOW_S, RT_WINDOW_HIGH_S)
        ax.text(
            0.985,
            0.965,
            f"Retained N = {int(metrics['n_retained']):,}\n"
            "Simulation + / -: "
            f"{metrics['simulation_area_plus']:.3f} / "
            f"{metrics['simulation_area_minus']:.3f}\n"
            "Analytic + / -: "
            f"{metrics['analytic_area_plus']:.3f} / "
            f"{metrics['analytic_area_minus']:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
            color="0.15",
        )
        if not bool(metrics["pass_all_checks"]):
            ax.text(
                0.985,
                0.04,
                "CHECK FAILED",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7.5,
                color="tab:red",
                weight="bold",
            )
        if row_index == 0:
            ax.set_title(delay_config["delay_label"], fontsize=10)
        if row_index == 2:
            ax.set_xlabel("RT relative to stimulus (s)")
        if column_index == 0:
            ax.set_ylabel(
                f"{case['case_label']}\n"
                f"{case['batch']}/{case['animal']}, "
                f"ABL {case['ABL']:.0f}, ILD {case['ILD']:+.0f}\n"
                "Conditional RT density",
            )

window_legend_handles = legend_handles + [
    Line2D(
        [0],
        [0],
        color="tab:red",
        lw=0.9,
        ls="--",
        alpha=0.65,
        label="Mean delay (80 ms)",
    )
]
fig_window.legend(
    handles=window_legend_handles,
    loc="upper center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 0.975),
)
fig_window.text(
    0.012,
    0.72,
    "choice +1",
    rotation=90,
    va="center",
    ha="center",
    fontsize=9,
)
fig_window.text(
    0.012,
    0.29,
    "choice -1 (mirrored)",
    rotation=90,
    va="center",
    ha="center",
    fontsize=9,
)
fig_window.suptitle(
    "Uniform-delay NPL+alpha after proactive left truncation: "
    "simulation versus analytic 0-1 s density",
    y=0.998,
    fontsize=13,
)
fig_window.tight_layout(rect=(0.025, 0.02, 1.0, 0.94))

window_figure_png = (
    OUTPUT_DIR
    / "uniform_delay_simulator_vs_analytic_likelihood_left_truncated_0_1s_3x3.png"
)
fig_window.savefig(window_figure_png, dpi=220, bbox_inches="tight")

print("\nRetained-window validation summary:")
print(
    window_summary_df[
        [
            "case_label",
            "delay_width_ms",
            "T_trunc_ms",
            "n_retained",
            "simulation_window_mass_after_left_truncation",
            "analytic_window_mass_after_left_truncation",
            "analytic_total_area",
            "max_simulation_analytic_subcdf_abs_error",
            "max_analytic_quadrature_subcdf_abs_error",
            "pass_all_checks",
        ]
    ].to_string(index=False, float_format=lambda value: f"{value:.6g}")
)
print("\nSaved retained-window outputs:")
for output_path in [window_summary_csv, window_figure_png]:
    print(f"  {output_path}")

failed_window_rows = window_summary_df.loc[
    ~window_summary_df["pass_all_checks"]
]
if len(failed_window_rows):
    print(
        f"\nWARNING: {len(failed_window_rows)} of 9 retained-window cells "
        "failed at least one check."
    )
    for row in failed_window_rows.itertuples():
        failed_checks = [
            column
            for column in window_summary_df.columns
            if column.startswith("pass_")
            and column != "pass_all_checks"
            and not bool(getattr(row, column))
        ]
        print(
            f"  {row.case_label}, width={row.delay_width_ms:g} ms: "
            + ", ".join(failed_checks)
        )
else:
    print("\nAll nine retained-window simulation/analytic checks passed.")
